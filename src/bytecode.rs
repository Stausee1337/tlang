
use std::{marker::PhantomData, ops::Deref};

use ahash::HashMap;
use tlang_macros::define_instructions;

use crate::{tvalue::TValue, symbol::Symbol, parse::Ident, interpreter::CodeStream};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CGKind {
    Null,
    Bool(bool),
    Register(u32),
    Descriptor(u32),
    Int32(i32),
}

#[derive(Clone, Copy)]
pub struct CGValue(u32);

macro_rules! slice {
    ($($expr:expr),*) => {
         &[$($expr),*] as &[_]
    };
}

impl CGValue {
    pub const CG_MAX_U32: u32 = 1 << 30;

    const NULL_TAG:  u32 = 0x0 << 29;
    const BOOL_TAG:  u32 = 0x1 << 29;
    const REG_TAG:   u32 = 0x2 << 29;
    const DESC_TAG:  u32 = 0x3 << 29;
    const INT32_TAG: u32 = 0x4 << 29;

    const VALUE_MASK: u32 = 0x1fffffff;
    const TAG_MASK:   u32 = 0xe0000000;

    pub const fn null() -> Self {
        CGValue(Self::NULL_TAG)
    }

    pub const fn bool(bool: bool) -> Self {
        let bool = bool as u32;
        CGValue((bool & Self::VALUE_MASK) | Self::BOOL_TAG)
    }

    pub const fn register(idx: u32) -> Self {
        debug_assert!(idx < Self::CG_MAX_U32);
        CGValue((idx & Self::VALUE_MASK) | Self::REG_TAG)
    }

    pub const fn descriptor(desc: u32) -> Self {
        debug_assert!(desc < Self::CG_MAX_U32);
        CGValue((desc & Self::VALUE_MASK) | Self::DESC_TAG)
    }

    pub const fn int32(int: i32) -> Self {
        let int = int as u32;
        debug_assert!(int < Self::CG_MAX_U32);
        CGValue((int & Self::VALUE_MASK) | Self::INT32_TAG)
    }

    pub fn to_rust(self) -> CGKind {
        match self.0 & Self::TAG_MASK {
            Self::NULL_TAG => CGKind::Null,
            Self::BOOL_TAG => CGKind::Bool((self.0 & Self::VALUE_MASK) != 0),
            Self::REG_TAG => CGKind::Register(self.0 & Self::VALUE_MASK),
            Self::DESC_TAG => CGKind::Descriptor(self.0 & Self::VALUE_MASK),
            Self::INT32_TAG => CGKind::Int32((self.0 & Self::VALUE_MASK) as i32),
            _ => unreachable!()
        }
    }
}

impl std::fmt::Debug for CGValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.to_rust() {
            CGKind::Null => f.write_str("Null"),
            CGKind::Bool(bool) => write!(f, "Bool({bool})"),
            CGKind::Register(reg) => write!(f, "Register({reg})"),
            CGKind::Descriptor(desc) => write!(f, "Descriptor({desc})"),
            CGKind::Int32(int) => write!(f, "Int32({int})"),
        }
    }
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
        CGValue::descriptor(idx as u32)
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
            return CGValue::int32(int);
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

impl<T: Copy> From<&[T]> for DynamicArray<T> {
    fn from(value: &[T]) -> Self {
        DynamicArray {
            length: value.len(),
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
