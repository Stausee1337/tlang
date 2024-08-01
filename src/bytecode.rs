
use std::{marker::PhantomData, ops::Deref, fmt::{Write, Result as FmtResult}};

use ahash::HashMap;
use tlang_macros::define_instructions;

use crate::{tvalue::TValue, symbol::Symbol, parse::Ident, interpreter::CodeStream};
use index_vec::{IndexVec, define_index_type};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OperandKind {
    Null,
    Bool(bool),
    Register(Register),
    Descriptor(Descriptor),
    Int32(i32),
}

#[derive(Clone, Copy)]
pub struct Operand(u32);

macro_rules! slice {
    ($($expr:expr),*) => {
         &[$($expr),*] as &[_]
    };
}

impl Operand {
    pub const CG_MAX_U32: u32 = 1 << 30;

    const NULL_TAG:  u32 = 0x0 << 29;
    const BOOL_TAG:  u32 = 0x1 << 29;
    const REG_TAG:   u32 = 0x2 << 29;
    const DESC_TAG:  u32 = 0x3 << 29;
    const INT32_TAG: u32 = 0x4 << 29;

    const VALUE_MASK: u32 = 0x1fffffff;
    const TAG_MASK:   u32 = 0xe0000000;

    pub const fn null() -> Self {
        Operand(Self::NULL_TAG)
    }

    pub const fn bool(bool: bool) -> Self {
        let bool = bool as u32;
        Operand((bool & Self::VALUE_MASK) | Self::BOOL_TAG)
    }

    pub const fn register(reg: Register) -> Self {
        let reg = reg._raw;
        debug_assert!(reg < Self::CG_MAX_U32);
        Operand((reg & Self::VALUE_MASK) | Self::REG_TAG)
    }

    pub const fn descriptor(desc: Descriptor) -> Self {
        let desc = desc._raw;
        debug_assert!(desc < Self::CG_MAX_U32);
        Operand((desc & Self::VALUE_MASK) | Self::DESC_TAG)
    }

    pub const fn int32(int: i32) -> Self {
        let int = int as u32;
        debug_assert!(int < Self::CG_MAX_U32);
        Operand((int & Self::VALUE_MASK) | Self::INT32_TAG)
    }

    pub fn to_rust(self) -> OperandKind {
        match self.0 & Self::TAG_MASK {
            Self::NULL_TAG => OperandKind::Null,
            Self::BOOL_TAG => OperandKind::Bool((self.0 & Self::VALUE_MASK) != 0),
            Self::REG_TAG => OperandKind::Register(Register::from_raw(self.0 & Self::VALUE_MASK)),
            Self::DESC_TAG => OperandKind::Descriptor(Descriptor::from_raw(self.0 & Self::VALUE_MASK)),
            Self::INT32_TAG => OperandKind::Int32((self.0 & Self::VALUE_MASK) as i32),
            _ => unreachable!()
        }
    }
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> FmtResult {
        match self.to_rust() {
            OperandKind::Null => f.write_str("Null"),
            OperandKind::Bool(bool) => write!(f, "{bool}"),
            OperandKind::Register(reg) => write!(f, "{:?}", reg),
            OperandKind::Descriptor(desc) => write!(f, "{desc:?}"),
            OperandKind::Int32(int) => write!(f, "Int32({int})"),
        }
    }
}

define_index_type! {
    pub struct Register = u32;
    DEBUG_FORMAT = "reg{}";
}

define_index_type! {
    pub struct Descriptor = u32;
    DEBUG_FORMAT = "${}";
}

struct RegisterAllocator(u32);

impl RegisterAllocator {
    fn prefill(amount: u32) -> Self {
        RegisterAllocator(amount)
    }

    fn next(&mut self) -> Register {
        let reg = Register::from_raw(self.0);
        self.0 += 1;
        reg
    }
}

#[derive(Clone, Copy)]
pub struct Local {
    pub constant: bool,
    pub declared: bool,
}

define_index_type! {
    pub struct CodeLabel = u32;
    DEBUG_FORMAT = "bb{}";
}

struct BasicBlock {
    label: CodeLabel,
    parent: Option<CodeLabel>,
    data: Vec<u8>,
    locals: HashMap<Symbol, Local>,
    terminated: bool
} 

impl BasicBlock {
    fn new(parent: Option<CodeLabel>) -> Self {
        Self {
            label: CodeLabel::from_raw(0),
            parent,
            data: vec![],
            locals: Default::default(),
            terminated: false
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Module,
    Function,
}

pub struct FunctionDisassembler<'f> {
    function: &'f CGFunction,
    stream: &'f mut dyn Write,
    indentation: usize,
    last_indent_pending: bool
}

impl<'f> FunctionDisassembler<'f> {
    pub fn dissassemble(function: &'f CGFunction, stream: &'f mut dyn Write) -> FmtResult {
        let mut disasm = Self {
            function,
            stream,
            indentation: 0,
            last_indent_pending: false
        };
        disasm.function_head()?;

        for block in &disasm.function.blocks {
            disasm.basic_block(block)?;
        }

        disasm.dedent();
        write!(disasm, "}}")
    }

    fn indent(&mut self) {
        self.indentation += 1;
    }

    fn dedent(&mut self) {
        self.indentation -= 1;
    }

    fn basic_block(&mut self, block: &'f BasicBlock) -> FmtResult {
        writeln!(self, "bb{} {{", block.label._raw)?;
        self.indent();

        let mut stream = CodeStream::new(&block.data);
        while !stream.eos() {
            let op = OpCode::decode(stream.current());
            let instruction = op.deserialize_for_debug(&mut stream);
            writeln!(self, "{instruction:?}")?;
        }

        self.dedent();
        writeln!(self, "}}")
    }

    fn function_head(&mut self) -> FmtResult {
        writeln!(self, "function \"\" ({}) {{", self.function.num_params)?;
        self.indent();
        writeln!(self, ".locals {}", self.function.local2reg.len())?;
        writeln!(self, ".registers {}", self.function.register_allocator.0)?;

        Ok(())
    }
}

impl<'f> Write for FunctionDisassembler<'f> {
    fn write_str(&mut self, mut s: &str) -> FmtResult {
        if self.last_indent_pending {
            self.stream.write_str(&"    ".repeat(self.indentation))?;
            self.last_indent_pending = false;
        }

        while let Some(idx) = s.find("\n") {
            self.stream.write_str(&s[..idx + 1])?;
            s = &s[idx + 1..];
            if !s.is_empty() {
                self.stream.write_str(&"    ".repeat(self.indentation))?;
            } else {
                self.last_indent_pending = true;
            }
        }
        self.stream.write_str(s)
    }
}

pub struct CGFunction {
    scope: Scope,
    num_params: u32,
    descriptor_table: IndexVec<Descriptor, TValue>,
    blocks: IndexVec<CodeLabel, BasicBlock>,
    current_block: CodeLabel,
    register_allocator: RegisterAllocator,
    local2reg: HashMap<(CodeLabel, Symbol), Register>
}

impl CGFunction {
    fn create() -> Self {
        Self {
            num_params: 0,
            scope: Scope::Module,
            descriptor_table: Default::default(),
            blocks: vec![BasicBlock::new(None)].into(),
            current_block: CodeLabel::from_raw(0),
            register_allocator: RegisterAllocator::prefill(0),
            local2reg: Default::default() 
        }
    }

    fn module() -> Self {
        Self::create()
    }

    fn function(params: &[Symbol]) -> Self {
        let mut rv = Self {
            num_params: params.len() as u32,
            scope: Scope::Function,
            register_allocator: RegisterAllocator::prefill(params.len() as u32),
            ..Self::create()
        };

        for param in params {
            rv.register_local(*param, false).expect("no locals before arguments");
            rv.declare_local(*param);
        }

        rv
    }

    fn current_block(&self) -> &BasicBlock {
        &self.blocks[self.current_block]
    }

    fn current_block_mut(&mut self) -> &mut BasicBlock {
        &mut self.blocks[self.current_block]
    }

    pub fn set_current_block(&mut self, label: CodeLabel) -> CodeLabel {
        let prev_block = self.current_block;
        debug_assert!(label < self.blocks.len_idx());
        self.current_block = label;
        prev_block
    }

    pub fn find_local(&self, symbol: Symbol) -> Option<Local> {
        let mut block = &self.blocks[self.current_block];
        loop {
            if let Some(local) = block.locals.get(&symbol) {
                if local.declared {
                    return Some(*local);
                }
            }
            let Some(parent) = block.parent else {
                return None;
            };
            block = &self.blocks[parent];
        }
    }

    pub fn fork_block(&mut self, inherit: bool) -> CodeLabel {
        let prev_block = self.current_block;

        let new_block = BasicBlock::new(
            if inherit { Some(self.current_block) } else { None });
        let label = self.blocks.push(new_block);
        self.blocks[label].label = label;

        self.current_block = label;

        prev_block
    }

    pub fn is_block_terminated(&self) -> bool {
        self.current_block().terminated
    }

    pub fn register_local(&mut self, symbol: Symbol, constant: bool) -> Result<(), ()> {
        let local = Local {
            constant,
            declared: false
        };
        if let Some(..) = self.current_block_mut().locals.insert(symbol, local) {
            return Err(());
        }
        Ok(())
    }

    fn declare_local(&mut self, symbol: Symbol) -> Operand {
        let local = self.current_block_mut().locals.get_mut(&symbol)
            .expect("register local before declare");
        debug_assert!(!local.declared);
        local.declared = true;

        let reg = self.register_allocator.next();
        let result = self.local2reg.insert((self.current_block, symbol), reg);
        debug_assert!(result.is_none());
        Operand::register(reg)
    }

    pub fn get_local_reg(&self, symbol: Symbol) -> Operand {
        let reg = *self.local2reg.get(&(self.current_block, symbol))
            .expect("symbol corresponds to actual local");
        Operand::register(reg)
    }

    fn descriptor(&mut self, tvalue: TValue) -> Operand {
        let idx = self.descriptor_table.len_idx();
        self.descriptor_table.push(tvalue);
        Operand::descriptor(idx)
    }
}

pub struct BytecodeGenerator {
    current_fn: CGFunction
}

impl BytecodeGenerator {
    pub fn new() -> Self {
        Self {
            current_fn: CGFunction::module()
        }
    }

    pub fn current_fn(&self) -> &CGFunction {
        return &self.current_fn;
    }

    pub fn current_fn_mut(&mut self) -> &mut CGFunction {
        return &mut self.current_fn;
    }

    pub fn fork_block(&mut self, inherit: bool) -> CodeLabel {
        self.current_fn_mut()
            .fork_block(inherit)
    }

    pub fn is_terminated(&self) -> bool {
        self.current_fn()
            .is_block_terminated()
    }

    pub fn set_current_block(&mut self, label: CodeLabel) -> CodeLabel {
        self.current_fn_mut()
            .set_current_block(label)
    }

    pub fn register_local(&mut self, ident: Ident, constant: bool) -> Result<(), ()> {
        self.current_fn_mut()
            .register_local(ident.symbol, constant)
    }

    pub fn find_local(&self, symbol: Symbol) -> Option<Local> {
        self.current_fn().find_local(symbol)
    }

    pub fn allocate_reg(&mut self) -> Operand {
        let reg = self.current_fn_mut()
            .register_allocator
            .next();
        Operand::register(reg)
    }

    fn declare_local(&mut self, symbol: Symbol) -> Operand {
        self.current_fn_mut()
            .declare_local(symbol)
    }

    pub fn get_local_reg(&self, symbol: Symbol) -> Operand {
        self.current_fn()
            .get_local_reg(symbol)
    }

    pub fn make_string_literal(&mut self, literal: &str) -> Result<Operand, snailquote::UnescapeError> {
        let string = snailquote::unescape(literal)?;
        let tvalue = TValue::string(&string);
        Ok(self.current_fn_mut().descriptor(tvalue))
    }

    pub fn make_int(&mut self, int: u64) -> Operand {
        if let Ok(int) = i32::try_from(int) {
            return Operand::int32(int);
        }
        self.current_fn_mut().descriptor(TValue::bigint(&int.to_le_bytes()))
    }

    pub fn make_float(&mut self, float: f64) -> Operand {
        self.current_fn_mut().descriptor(TValue::float(float))
    }
}

pub trait InstructionSerializer<I: Instruction> {
    fn serialize(inst: I, vec: &mut Vec<u8>);
    fn deserialize(data: &[u8]) -> Option<(I, usize)>;
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
    fn deserialize(data: &[u8]) -> Option<(I, usize)> {
        BitSerializer::deserialize(data).map(|i| (i, std::mem::size_of::<I>()))
    }
}

pub struct CallSerializer;

impl InstructionSerializer<instructions::Call> for CallSerializer {
    #[inline(always)]
    fn serialize(instructions::Call { callee, arguments, dst }: instructions::Call, vec: &mut Vec<u8>) {
        BitSerializer::serialize(callee, vec);
        BitSerializer::serialize(dst, vec);
        arguments.serialize(vec);
    }

    #[inline(always)]
    fn deserialize(data: &[u8]) -> Option<(instructions::Call, usize)> {
        const VAL_SIZE: usize = std::mem::size_of::<Operand>();

        let callee = BitSerializer::deserialize(data)?;
        let dst = BitSerializer::deserialize(&data[VAL_SIZE..])?;
        let (arguments, size) = DynamicArray::deserialize(&data[VAL_SIZE * 2..])?;

        Some((instructions::Call {
            callee,
            arguments,
            dst
        }, VAL_SIZE * 2 + size))
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

    fn deserialize(data: &[u8]) -> Option<(Self, usize)> {
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
        Some((Self {
            length,
            data: data[header_size..].as_ptr() as *const T
        }, header_size + size))
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

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for DynamicArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> FmtResult {
        let x: &[_] = Deref::deref(&self);
        write!(f, "{:?}", x)
    }
}


define_instructions! {
    Mov {
        dst: Operand,
        src: Operand,
    },

    Add { dst: Operand, lhs: Operand, rhs: Operand },
    Sub { dst: Operand, lhs: Operand, rhs: Operand },
    Mul { dst: Operand, lhs: Operand, rhs: Operand },
    Div { dst: Operand, lhs: Operand, rhs: Operand },
    Mod { dst: Operand, lhs: Operand, rhs: Operand },

    LeftShift { dst: Operand, lhs: Operand, rhs: Operand },
    RightShift { dst: Operand, lhs: Operand, rhs: Operand },

    BitwiseAnd { dst: Operand, lhs: Operand, rhs: Operand },
    BitwiseOr { dst: Operand, lhs: Operand, rhs: Operand },
    BitwiseXor { dst: Operand, lhs: Operand, rhs: Operand },

    Neg { dst: Operand, src: Operand },
    Not { dst: Operand, src: Operand },
    Invert { dst: Operand, src: Operand },

    GetGlobal { dst: Operand, symbol: Symbol  },
    SetGlobal { symbol: Symbol, src: Operand, },

    GetAttribute {
        dst: Operand,
        base: Operand,
        attribute: Symbol,
    },
    SetAttribute {
        base: Operand,
        attribute: Symbol,
        src: Operand,
    },

    GetSubscript {
        dst: Operand,
        base: Operand,
        index: Operand,
    },
    SetSubscript {
        base: Operand,
        index: Operand,
        src: Operand,
    },

    #[terminator(true)]
    Branch { target: CodeLabel },
 
    #[terminator(true)]
    BranchIf {
        condition: Operand,
        true_target: CodeLabel,
        false_target: CodeLabel
    },
    #[terminator(true)]
    BranchEq {
        lhs: Operand, rhs: Operand,
        true_target: CodeLabel, false_target: CodeLabel
    },
    #[terminator(true)]
    BranchNe {
        lhs: Operand, rhs: Operand,
        true_target: CodeLabel, false_target: CodeLabel
    },
    #[terminator(true)]
    BranchGt {
        lhs: Operand, rhs: Operand,
        true_target: CodeLabel, false_target: CodeLabel
    },
    #[terminator(true)]
    BranchGe {
        lhs: Operand, rhs: Operand,
        true_target: CodeLabel, false_target: CodeLabel
    },
    #[terminator(true)]
    BranchLt {
        lhs: Operand, rhs: Operand,
        true_target: CodeLabel, false_target: CodeLabel
    },
    #[terminator(true)]
    BranchLe {
        lhs: Operand, rhs: Operand,
        true_target: CodeLabel, false_target: CodeLabel
    },

    #[terminator(true)]
    Return { value: Operand },
    #[terminator(true)]
    Fallthrough,

    #[serializer(CallSerializer)]
    Call {
        dst: Operand,
        callee: Operand,
        arguments: DynamicArray<Operand>,
    },
    Error,
    Noop = 0x80
}
