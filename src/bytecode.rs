
use ahash::HashMap;
use tlang_macros::define_instructions;

use crate::{tvalue::TValue, symbol::Symbol, lexer::Span, parse::Ident};


#[derive(Clone, Copy)]
#[repr(C)]
pub enum ImmValue {
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
pub enum ScopeKind {
    Module,
    Function,
    Closure
}

pub struct Scope {
    kind: ScopeKind,
    descriptor_table: Vec<TValue>,
    blocks: Vec<BasicBlock>,
    current_block: usize
}

impl Scope {
    fn new(kind: ScopeKind) -> Self {
        Self {
            kind,
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

    fn descriptor(&mut self, tvalue: TValue) -> ImmValue {
        let idx = self.descriptor_table.len();
        self.descriptor_table.push(tvalue);
        ImmValue::Descriptor(idx as u32)
    }
}

pub struct BytecodeGenerator {
    current_scope: Scope
}

impl BytecodeGenerator {
    pub fn new() -> Self {
        Self {
            current_scope: Scope::new(ScopeKind::Module)
        }
    }

    pub fn current_scope(&mut self) -> &mut Scope {
        return &mut self.current_scope;
    }

    pub fn scope_kind(&mut self) -> ScopeKind {
        self.current_scope().kind
    }

    pub fn register_variable(&mut self, ident: Ident, constant: bool) -> Result<(), ()> {
        self.current_scope()
            .register_variable(ident.symbol, constant)
    }

    pub fn make_string_literal(&mut self, literal: &str) -> Result<ImmValue, snailquote::UnescapeError> {
        let string = snailquote::unescape(literal)?;
        let tvalue = TValue::string(&string);
        Ok(self.current_scope()
            .descriptor(tvalue))
    }

    pub fn make_int(&mut self, int: u64) -> ImmValue {
        if let Ok(int) = i32::try_from(int) {
            return ImmValue::Int32(int);
        }
        self.current_scope().
            descriptor(TValue::bigint(&int.to_le_bytes()))
    }

    pub fn make_float(&mut self, float: f64) -> ImmValue {
        todo!()
    }
}

pub trait Instruction: Sized + Copy {
    const CODE: instructions::OpCode;

    #[inline(always)]
    fn serialize(&self, vec: &mut Vec<u8>) {
        vec.push(Self::CODE as u8);
        let bytes = unsafe { as_raw_bytes(&self) };
        vec.extend(bytes);
    }

    #[inline(always)]
    fn deserialize(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return None;
        }
        *unsafe { from_raw_bytes(bytes.as_ptr()) }
    }
}

#[inline(always)]
pub unsafe fn as_raw_bytes<T>(obj: &T) -> &[u8] {
    std::slice::from_raw_parts(
        (obj as *const T) as * const u8,
        std::mem::size_of::<T>()
    )
}

#[inline(always)]
pub unsafe fn from_raw_bytes<'a, T>(ptr: *const u8) -> &'a T {
    unsafe { std::mem::transmute::<*const u8, &'a T>(ptr) }
}

define_instructions! {
    Add {
        lhs: ImmValue,
        rhs: ImmValue,
    } = 0x00,
    Noop = 0x80
}
