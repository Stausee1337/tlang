
use std::{marker::PhantomData, ops::Deref, array::TryFromSliceError, slice::SliceIndex};

use ahash::HashMap;
use tlang_macros::define_instructions;

use crate::{tvalue::TValue, symbol::Symbol, parse::Ident, interpreter::CodeStream};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum CGValue {
    Null,
    Bool(bool),
    Int32(i32),
    Register(u32),
    Descriptor(u32),
}

struct Local {
    constant: bool,
    declared: bool,
}

struct BasicBlock {
    id: usize,
    parent: Option<usize>,
    data: Vec<u8>,
    locals: HashMap<Symbol, Local>
}

impl BasicBlock {
    fn new(id: usize, parent: Option<usize>) -> Self {
        Self {
            id,
            parent,
            data: vec![],
            locals: Default::default()
        }
    }
}

#[derive(Clone, Copy)]
pub enum Scope {
    Module,
    Function,
    Closure
}

pub struct CGFunction {
    scope: Scope,
    descriptor_table: Vec<TValue>,
    blocks: Vec<BasicBlock>,
    current_block: usize
}

impl CGFunction {
    fn new(kind: Scope) -> Self {
        Self {
            scope: kind,
            descriptor_table: Default::default(),
            blocks: vec![BasicBlock::new(0, None)],
            current_block: 0
        }
    }

    fn current_block(&mut self) -> &mut BasicBlock {
        &mut self.blocks[self.current_block]
    }

    fn fork_block(&mut self) -> usize {
        let new_block = BasicBlock::new(self.blocks.len(), Some(self.current_block));
        let id = new_block.id;
        self.blocks.push(new_block);
        id
    }

    fn register_variable(&mut self, symbol: Symbol, constant: bool) -> Result<(), ()> {
        let local = Local {
            constant,
            declared: false
        };
        if let Some(..) = self.current_block().locals.insert(symbol, local) {
            return Err(());
        }
        Ok(())
    }

    fn descriptor(&mut self, tvalue: TValue) -> CGValue {
        let idx = self.descriptor_table.len();
        self.descriptor_table.push(tvalue);
        CGValue::Descriptor(idx as u32)
    }
}

pub struct BytecodeGenerator {
    current_fn: CGFunction
}

impl BytecodeGenerator {
    pub fn new() -> Self {
        Self {
            current_fn: CGFunction::new(Scope::Module)
        }
    }

    pub fn current_fn(&mut self) -> &mut CGFunction {
        return &mut self.current_fn;
    }

    pub fn register_variable(&mut self, ident: Ident, constant: bool) -> Result<(), ()> {
        self.current_fn()
            .register_variable(ident.symbol, constant)
    }

    pub fn make_string_literal(&mut self, literal: &str) -> Result<CGValue, snailquote::UnescapeError> {
        let string = snailquote::unescape(literal)?;
        let tvalue = TValue::string(&string);
        Ok(self.current_fn().descriptor(tvalue))
    }

    pub fn make_int(&mut self, int: u64) -> CGValue {
        if let Ok(int) = i32::try_from(int) {
            return CGValue::Int32(int);
        }
        self.current_fn().descriptor(TValue::bigint(&int.to_le_bytes()))
    }

    pub fn make_float(&mut self, float: f64) -> CGValue {
        self.current_fn().descriptor(TValue::float(float))
    }
}

pub trait InstructionSerializer<I: Instruction> {
    fn serialize(inst: I, vec: &mut Vec<u8>);
    fn deserialize(data: &[u8]) -> Option<I>;
}

pub struct BitSerializer<I>(PhantomData<I>);

impl<T: Copy> BitSerializer<T> {
    #[inline(always)]
    fn serialize(inst: T, vec: &mut Vec<u8>) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&inst as *const T) as * const u8,
                std::mem::size_of::<T>()
            )
        };
        vec.extend(bytes);
    }

    #[inline(always)]
    fn deserialize(data: &[u8]) -> Option<T> {
        if data.len() < std::mem::size_of::<Self>() {
            return None;
        }
        Some(*unsafe { std::mem::transmute::<*const u8, &T>(data.as_ptr()) })
    }
}

impl<I: Instruction> InstructionSerializer<I> for BitSerializer<I> {
    #[inline(always)]
    fn serialize(inst: I, vec: &mut Vec<u8>) {
        BitSerializer::serialize(inst, vec)
    }

    #[inline(always)]
    fn deserialize(data: &[u8]) -> Option<I> {
        BitSerializer::deserialize(data)
    }
}

pub struct CallSerializer;

impl InstructionSerializer<instructions::Call> for CallSerializer {
    #[inline(always)]
    fn serialize(instructions::Call { callee, arguments }: instructions::Call, vec: &mut Vec<u8>) {
        BitSerializer::serialize(callee, vec);
        arguments.serialize(vec);
    }

    #[inline(always)]
    fn deserialize(data: &[u8]) -> Option<instructions::Call> {
        const VAL_SIZE: usize = std::mem::size_of::<CGValue>();

        let callee = BitSerializer::deserialize(data)?;
        let arguments = DynamicArray::deserialize(&data[VAL_SIZE..])?;

        Some(instructions::Call {
            callee,
            arguments
        })
    }
}

#[derive(Clone, Copy)]
pub struct DynamicArray<T: Copy> {
    length: usize,
    data: *const T
}

impl<T: Copy> DynamicArray<T> {
    fn serialize(&self, vec: &mut Vec<u8>) {
        let size = self.length * std::mem::size_of::<T>();
        let bytes = unsafe {
            std::slice::from_raw_parts(self.data as *const u8, size)
        };
        vec.extend(size.to_le_bytes());
        vec.extend(bytes);
    }

    fn deserialize(data: &[u8]) -> Option<Self> {
        let item_size = std::mem::size_of::<T>();
        let header_size = std::mem::size_of::<usize>();

        if data.len() < header_size {
            return None;
        }

        let size: [u8; 8] = data[0..header_size].try_into().unwrap();
        let size = usize::from_le_bytes(size);

        let remaining = data.len() - header_size;
        if size % item_size != 0 || remaining < size {
            return None;
        }

        let length = size / item_size;
        Some(Self {
            length,
            data: data[header_size..].as_ptr() as *const T
        })
    }
}

impl<T: Copy, const SIZE: usize> From<[T; SIZE]> for DynamicArray<T> {
    fn from(value: [T; SIZE]) -> Self {
        DynamicArray {
            length: SIZE,
            data: value.as_ptr()
        }
    }
}

impl<T: Copy> Deref for DynamicArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.data, self.length) }
    }
}

define_instructions! {
    Add {
        lhs: CGValue,
        rhs: CGValue,
    } = 0x00,
    #[serializer(CallSerializer)]
    Call {
        callee: CGValue,
        arguments: DynamicArray<CGValue>
    },
    Noop = 0x80
}
