
use std::{ops::IndexMut, fmt::{Write, Result as FmtResult}, usize, cell::OnceCell, rc::Rc};

use ahash::HashMap;
use tlang_macros::define_instructions;

use crate::{tvalue::{TFunction, TValue, TString, TInteger, TFloat, TFnKind, TBool, Typed}, symbol::Symbol, parse::Ident, codegen, memory::GCRef, interpreter::VM};
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
            OperandKind::Bool(bool) => write!(f, "Bool({bool})"),
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

pub struct CodeStream<'l> {
    position: usize,
    code: &'l [u8],
    blocks: &'l [u32],
}

impl<'l> CodeStream<'l> {
    pub fn from_raw(raw: &'l TRawCode) -> Self {
        Self {
            position: 0,
            code: raw.code(),
            blocks: raw.blocks()
        }
    }

    fn debug_from_data(code: &'l [u8])  -> Self {
        Self {
            code,
            position: 0,
            blocks: &[]
        }
    }

    #[inline(always)]
    pub fn eos(&self) -> bool {
        self.code().len() == 0
    }

    #[inline(always)]
    pub fn data(&self) -> &[u8] {
        self.code
    }

    #[inline(always)]
    pub fn code(&self) -> &[u8] {
        &self.data()[self.position..]
    }

    #[inline(always)]
    pub fn current(&self) -> u8 {
        self.data()[self.position]
    }

    #[inline(always)]
    pub fn jump(&mut self, to: CodeLabel) {
        self.position = self.blocks[to.index()] as usize;
    }

    #[inline(always)]
    pub fn bump(&mut self, amount: usize) {
        self.position += amount;
    }
}


struct RegisterAllocator(u32);

impl RegisterAllocator {
    fn new() -> Self {
        RegisterAllocator(0)
    }

    fn next(&mut self) -> Register {
        let reg = Register::from_raw(self.0);
        self.0 += 1;
        reg
    }
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

        let mut stream = CodeStream::debug_from_data(&block.data);
        while !stream.eos() {
            let op = OpCode::decode(stream.current());
            let instruction = op.deserialize_for_debug(&mut stream);
            writeln!(self, "{instruction:?}")?;
        }

        self.dedent();
        writeln!(self, "}}")
    }

    fn function_head(&mut self) -> FmtResult {
        writeln!(self, "function {:?} ({}) {{", self.function.name, self.function.num_params)?;
        self.indent();
        
        let mut code_size = 0;
        for block in &self.function.blocks {
            code_size += block.data.len();
        }

        writeln!(self, ".locals {}", self.function.num_locals)?;
        writeln!(self, ".registers {}", self.function.register_allocator.0)?;
        writeln!(self, ".codeSize {}", code_size)?;

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

define_index_type! {
    pub struct CodeLabel = u32;
    DEBUG_FORMAT = "bb{}";
}

#[derive(Clone, Copy)]
pub struct Local {
    pub constant: bool,
    pub register: Option<Register>,
}

struct BasicBlock {
    label: CodeLabel,
    data: Vec<u8>,
    terminated: bool
} 

impl BasicBlock {
    fn new() -> Self {
        Self {
            label: CodeLabel::from_raw(0),
            data: vec![],
            terminated: false
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RibKind {
    Module,
    Function,
    If,
    Loop,
}

pub struct Rib {
    kind: RibKind,
    locals: HashMap<Symbol, Local>,
    loop_ctx: OnceCell<(CodeLabel, CodeLabel)>
}

impl Rib {
    fn new(kind: RibKind) -> Self {
        Self {
            kind,
            locals: Default::default(),
            loop_ctx: OnceCell::new()
        }
    }

    pub fn loop_ctx(&self) -> (CodeLabel, CodeLabel) {
        debug_assert!(self.kind == RibKind::Loop);
        *self.loop_ctx.get().expect("loop_ctx set in loop")
    }
}

pub struct CGFunction {
    name: Symbol,
    num_params: u32,
    num_locals: u32,
    descriptor_table: IndexVec<Descriptor, TValue>,
    blocks: IndexVec<CodeLabel, BasicBlock>,
    current_block: CodeLabel,
    register_allocator: RegisterAllocator,
    ribs: Vec<Rib>,
}

impl CGFunction {

    fn module() -> Self {
        Self {
            name: todo!(),
            ribs: vec![Rib::new(RibKind::Module)],
            num_params: 0,
            num_locals: 0,
            descriptor_table: Default::default(),
            blocks: vec![BasicBlock::new()].into(),
            current_block: CodeLabel::from_raw(0),
            register_allocator: RegisterAllocator::new(),
        }
    }

    fn function(name: Symbol, _closure: bool, params: &[&Ident]) -> Result<Self, Ident> {
        let mut rv = Self {
            name,
            num_params: params.len() as u32,
            ribs: vec![Rib::new(RibKind::Function)],
            register_allocator: RegisterAllocator::new(),
            ..Self::module()
        };

        for param in params {
            rv.register_local(param.symbol, false).map_err(|_| **param)?;
            rv.declare_local(param.symbol);
        }

        Ok(rv)
    }

    fn current_block(&self) -> &BasicBlock {
        &self.blocks[self.current_block]
    }

    fn current_block_mut(&mut self) -> &mut BasicBlock {
        &mut self.blocks[self.current_block]
    }

    fn current_rib(&self) -> &Rib {
        self.ribs.last().expect("at least one root rib")
    }

    fn current_rib_mut(&mut self) -> &mut Rib {
        self.ribs.last_mut().expect("at least one root rib")
    }

    pub fn set_current_block(&mut self, label: CodeLabel) -> CodeLabel {
        let prev_block = self.current_block;
        debug_assert!(label < self.blocks.len_idx());
        self.current_block = label;
        prev_block
    }


    pub fn find_rib(&mut self, kind: RibKind, depth: i32) -> Option<&mut Rib> {
        let size = self.ribs.len() as i32;
        let mut iteration = 0;
        while iteration != depth {
            let idx = size - (iteration + 1);
            if idx < 0 {
                return None;
            }
            let rib_kind = self.ribs[idx as usize].kind;
            if kind == rib_kind {
                let rib = self.ribs.index_mut(idx as usize);
                return Some(rib);
            }

            iteration += 1;
        }
        None
    }

    pub fn set_loop_ctx(&mut self, continue_target: CodeLabel, break_target: CodeLabel) {
        let loop_rib = self.find_rib(RibKind::Loop, 1).expect("set_loop_ctxt with loop rib");
        loop_rib.loop_ctx.set((continue_target, break_target)).expect("loop_ctx not set");
    }

    pub fn find_local(&self, symbol: Symbol) -> Option<Local> {
        for rib in self.ribs.iter().rev() {
            if let Some(local) = rib.locals.get(&symbol) {
                if local.register.is_some() {
                    return Some(*local);
                }
            }
        }
        None
    }

    pub fn make_block(&mut self) -> CodeLabel {
        let prev_block = self.current_block;

        let label = self.blocks.push(BasicBlock::new());
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
            register: None
        };
        if let Some(..) = self.current_rib_mut().locals.insert(symbol, local) {
            return Err(());
        }
        Ok(())
    }

    pub fn declare_local(&mut self, symbol: Symbol) -> Operand {
        self.num_locals += 1;
        let reg = self.register_allocator.next();
        let local = self.current_rib_mut().locals.get_mut(&symbol)
            .expect("register local before declare");
        debug_assert!(local.register.is_none());
        local.register = Some(reg);
        Operand::register(reg)
    }

    pub fn get_local_reg(&self, symbol: Symbol) -> Operand {
        let reg = self.find_local(symbol).expect("symbol corresponds to actual local");
        Operand::register(reg.register.expect("find_local finds declared locals"))
    }

    fn descriptor<T: Into<TValue>>(&mut self, tvalue: T) -> Operand {
        let idx = self.descriptor_table.len_idx();
        self.descriptor_table.push(tvalue.into());
        Operand::descriptor(idx)
    }
}

pub struct BytecodeGenerator {
    root_fn: CGFunction,
    function_stack: Vec<CGFunction>
}

pub struct ClosureScope;

impl BytecodeGenerator {
    pub fn new() -> Self {
        Self {
            root_fn: CGFunction::module(),
            function_stack: Default::default()
        }
    }

    pub fn vm(&self) -> Rc<VM> {
        todo!()
    }

    pub fn current_fn(&self) -> &CGFunction {
        return self.function_stack.last().unwrap_or_else(|| &self.root_fn);
    }

    pub fn current_fn_mut(&mut self) -> &mut CGFunction {
        return self.function_stack.last_mut().unwrap_or_else(|| &mut self.root_fn);
    }

    pub fn with_function<F: FnOnce(&mut Self) -> Result<(), codegen::CodegenErr>>(
        &mut self, name: Ident, params: &[&Ident], do_work: F) -> Result<Option<ClosureScope>, codegen::CodegenErr> {
        if self.current_fn().current_rib().kind != RibKind::Module {
            todo!("closures")
        }
        let func = CGFunction::function(name.symbol, false, params)
            .map_err(|param| codegen::CodegenErr::SyntaxError {
                message: Some(format!("param {:?} has already been declared", param.symbol)),
                span: param.span
            })?;
        self.function_stack.push(func);
        do_work(self)?;
        let func = self.function_stack.pop().unwrap();

        let mut string = String::new();
        FunctionDisassembler::dissassemble(&func, &mut string).unwrap();
        println!("{string}");

        let func = TFunction::from_codegen(&self.vm(), func);

        let result = func.call(&mut [
            TBool::from_bool(false).into(),
        ]);
        let int = TBool::try_from(result).unwrap();
        println!("{int}");

        Ok(None)
    }

    pub fn with_rib<F: FnOnce(&mut Self) -> R, R>(&mut self, kind: RibKind, do_work: F) -> R {
        debug_assert!(kind != RibKind::Module);
        self.current_fn_mut().ribs.push(Rib::new(kind));
        let rv = do_work(self);
        self.current_fn_mut().ribs.pop();
        rv
    }

    pub fn find_rib(&mut self, kind: RibKind, depth: i32) -> Option<&mut Rib> {
        self.current_fn_mut()
            .find_rib(kind, depth)
    }

    pub fn set_loop_ctx(&mut self, continue_target: CodeLabel, break_target: CodeLabel) {
        self.current_fn_mut()
            .set_loop_ctx(continue_target, break_target)
    }

    pub fn make_block(&mut self) -> CodeLabel {
        self.current_fn_mut()
            .make_block()
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

    pub fn declare_local(&mut self, symbol: Symbol) -> Operand {
        self.current_fn_mut()
            .declare_local(symbol)
    }

    pub fn get_local_reg(&self, symbol: Symbol) -> Operand {
        self.current_fn()
            .get_local_reg(symbol)
    }

    pub fn make_string(&mut self, string: GCRef<TString>) -> Operand {
        self.current_fn_mut().descriptor(string)
    }

    pub fn make_int(&mut self, int: u64) -> Operand {
        if let Ok(int) = i32::try_from(int) {
            return Operand::int32(int);
        }
        self.current_fn_mut().descriptor(TInteger::from_usize(int as usize))
    }

    pub fn make_float(&mut self, float: f64) -> Operand {
        self.current_fn_mut().descriptor(TFloat::from_float(float))
    }
}

impl TFunction {
    pub fn from_codegen(vm: &VM, func: CGFunction) -> GCRef<Self> {
        let mut codesize = 0;
        for block in &func.blocks {
            codesize += block.data.len();
        }

        let (function, tcode) = TFunction::create_presized(
            vm,
            vm.symbols.get(func.name),
            codesize,
            func.num_params,
            func.register_allocator.0,
            func.descriptor_table.len_idx()._raw,
            func.blocks.len_idx()._raw
        );

        let mut offset = 0;
        for block in &func.blocks {
            assert!(block.terminated);

            tcode.blocks_mut()[block.label.index()] = offset as u32;
            let length = block.data.len();
            let codebuf = &mut tcode.code_mut()[offset..offset + length];
            codebuf.copy_from_slice(&block.data);
            offset += block.data.len();
        }

        for (descriptor, value) in func.descriptor_table.iter_enumerated() {
            tcode.descriptors_mut()[descriptor.index()] = *value;
        }

        function
    }

    fn create_presized(
        vm: &VM,
        name: GCRef<TString>,
        codesize: usize,
        params: u32,
        registers: u32,
        descriptors: u32,
        blocks: u32
    ) -> (GCRef<TFunction>, &mut TRawCode) {
        let extra_size = codesize
            + descriptors as usize * std::mem::size_of::<TValue>()
            + blocks as usize * std::mem::size_of::<u32>();
        let code = TRawCode::new(codesize, params, registers, descriptors, blocks);
        let mut function = Self::ttype(vm).allocate_var_object(
            Self {
                name,
                kind: TFnKind::Function(code)
            },
            extra_size
        );
        let function2 = function.clone();
        match function.kind {
            TFnKind::Function(ref mut code) => {
                (function2, unsafe { std::mem::transmute(code) })
            }
        }
    }
}

#[repr(C)]
pub struct TRawCode {
    code_size: TInteger,
    num_params: TInteger,
    num_registers: TInteger,
    num_descriptors: TInteger,
    num_blocks: TInteger,
    data: [u8; 0]
}

impl TRawCode {
    fn new(
        codesize: usize,
        params: u32,
        registers: u32,
        descriptors: u32,
        blocks: u32
    ) -> Self {
        Self {
            code_size: TInteger::from_usize(codesize),
            num_params: TInteger::from_usize(params as usize),
            num_registers: TInteger::from_usize(registers as usize),
            num_descriptors: TInteger::from_usize(descriptors as usize),
            num_blocks: TInteger::from_usize(blocks as usize),
            data: [0u8; 0]
        }
    }
 
    fn create_presized(
        codesize: usize,
        params: u32,
        registers: u32,
        descriptors: u32,
        blocks: u32
    ) -> GCRef<Self> {
        let extra_size = codesize
            + descriptors as usize * std::mem::size_of::<TValue>()
            + blocks as usize * std::mem::size_of::<u32>();
        /*let interpreter = get_interpeter();
        interpreter.block_allocator.allocate_var_atom(
            TRawCode::new(codesize, params, registers, descriptors, blocks),
            extra_size
        )*/
        todo!()
    }

    pub fn registers(&self) -> usize {
        self.num_registers.as_usize().expect("TRawCode sensible num_registers")
    }

    pub fn params(&self) -> usize {
        self.num_params.as_usize().expect("TRawCode sensible num_params")
    }

    pub fn blocks(&self) -> &[u32] {
        unsafe {
            let len = self.num_blocks.as_usize().expect("TRawCode sensible num_blocks");
            let offset = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors")
                * std::mem::size_of::<TValue>();
            std::slice::from_raw_parts(
                self.data.as_ptr().add(offset) as *const u32,
                len
            )
        }
    }

    pub fn blocks_mut(&mut self) -> &mut [u32] {
        unsafe {
            let len = self.num_blocks.as_usize().expect("TRawCode sensible num_blocks");
            let offset = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors")
                * std::mem::size_of::<TValue>();
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(offset) as *mut u32,
                len
            )
        }
    }

    pub fn code(&self) -> &[u8] {
        unsafe {
            let len = self.code_size.as_usize().expect("TRawCode sensible code_size");
            let offset = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors")
                * std::mem::size_of::<TValue>()
                + self.num_blocks.as_usize().expect("TRawCode sensible num_blocks")
                * std::mem::size_of::<u32>();
            std::slice::from_raw_parts(
                self.data.as_ptr().add(offset),
                len
            )
        }
    }

    pub fn code_mut(&mut self) -> &mut [u8] {
        unsafe {
            let len = self.code_size.as_usize().expect("TRawCode sensible code_size");
            let offset = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors")
                * std::mem::size_of::<TValue>()
                + self.num_blocks.as_usize().expect("TRawCode sensible num_blocks")
                * std::mem::size_of::<u32>();
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(offset),
                len
            )
        }
    }

    pub fn descriptors(&self) -> &[TValue] {
        unsafe {
            let len = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors");
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const TValue,
                len
            )
        }
    }

    pub fn descriptors_mut(&mut self) -> &mut [TValue] {
        unsafe {
            let len = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors");
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut TValue,
                len
            )
        }
    }
}

pub trait InstructionSerializer<I: Instruction> {
    fn serialize(inst: I, vec: &mut Vec<u8>);
    fn deserialize<'c>(data: &'c mut CodeStream) -> Option<&'c I>;
}

pub struct BitSerializer;

impl BitSerializer {
    #[inline(always)]
    fn serialize<T: Copy>(inst: T, vec: &mut Vec<u8>) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&inst as *const T) as * const u8,
                std::mem::size_of::<T>()
            )
        };
        vec.extend(bytes);
    }

    #[inline(always)]
    fn deserialize<T: Copy>(data: &[u8]) -> Option<&T> {
        if data.len() < std::mem::size_of::<Self>() {
            return None;
        }
        Some(unsafe { std::mem::transmute::<*const u8, &T>(data.as_ptr()) })
    }
}

impl<I: Instruction> InstructionSerializer<I> for BitSerializer {
    #[inline(always)]
    fn serialize(inst: I, vec: &mut Vec<u8>) {
        BitSerializer::serialize::<I>(inst, vec)
    }

    #[inline(always)]
    fn deserialize<'c>(data: &'c mut CodeStream) -> Option<&'c I> {
        let codestream: *mut _ = &mut *data;

        let instr = BitSerializer::deserialize::<I>(data.code())?;
        CodeStream::bump(unsafe { &mut *codestream }, std::mem::size_of::<I>());

        Some(instr)
    }
}

pub struct CallSerializer;

impl CallSerializer {
    fn serialize_slice<'a, T>(slice: &'a [T], vec: &mut Vec<u8>) {
        let size = slice.len() * std::mem::size_of::<T>();
        let bytes = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, size)
        };
        vec.extend(size.to_le_bytes());
        vec.extend(bytes);
    }

    fn deserialize_slice<'a, T>(data: &'a [u8]) -> Option<(&'a [T], usize)> {
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
        unsafe {
            Some((std::slice::from_raw_parts(
                data[header_size..].as_ptr() as *const T,
                length
            ), header_size + size))
        }
    }
}

impl<'s> InstructionSerializer<instructions::Call_1<'s>> for CallSerializer {
    #[inline(always)]
    fn serialize(instr: instructions::Call_1<'s>, vec: &mut Vec<u8>) {
        let instructions::Call_1 { callee, arguments, dst } = instr;
        BitSerializer::serialize(callee, vec);
        BitSerializer::serialize(dst, vec);
        Self::serialize_slice(arguments, vec);
    }

    #[inline(always)]
    fn deserialize<'c>(_data: &'c mut CodeStream) -> Option<&'c instructions::Call_1<'s>> {
        panic!()
    }
}

impl InstructionSerializer<instructions::Call> for CallSerializer {
    #[inline(always)]
    fn serialize(_instr: instructions::Call, _vec: &mut Vec<u8>) {
        panic!()
    }

    #[inline(always)]
    fn deserialize<'c>(data: &'c mut CodeStream) -> Option<&'c instructions::Call> {
        const SIZE: usize = std::mem::size_of::<instructions::Call>();
        let codestream: *mut _ = &mut *data;

        let instr = BitSerializer::deserialize::<instructions::Call>(data.code());
        CodeStream::bump(unsafe { &mut *codestream }, SIZE);

        let (_arguments, size) = Self::deserialize_slice::<Operand>(data.code())?;
        CodeStream::bump(unsafe { &mut *codestream }, size);

        instr
    }
}

impl instructions::Call {
    pub fn arguments(&self) -> &[Operand] {
        let data = unsafe {
            std::slice::from_raw_parts(
                self.arguments.as_ptr() as *const u8,
                usize::MAX
            )
        };
        CallSerializer::deserialize_slice(data).unwrap().0
    }
}

fn call_format(call: &instructions::Call, f: &mut std::fmt::Formatter<'_>) -> FmtResult {
    let instructions::Call { dst, callee, .. } = *call;

    f.write_str("Call { ")?;

    write!(f, "dst: {:?}, ", dst)?;
    write!(f, "callee: {:?}, ", callee)?;
    write!(f, "arguments: {:?}", call.arguments())?;

    f.write_str(" }")
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

    DeclareGlobal { symbol: Symbol, init: Operand, constant: bool },

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
    #[formatter(call_format)]
    Call<'s> {
        dst: Operand,
        callee: Operand,
        arguments: &'s [Operand],
    },

    GetIterator { dst: Operand, iterable: Operand },
    NextIterator {
        iterator: Operand,
        loop_target: CodeLabel,
        end_target: CodeLabel
    },

    Error,
    Noop = 0x80
}
