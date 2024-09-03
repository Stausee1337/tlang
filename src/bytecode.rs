
use std::{ops::IndexMut, fmt::{Write, Result as FmtResult}, usize, cell::OnceCell, io::Write as IOWrite};

use ahash::HashMap;
use tlang_macros::define_instructions;

use crate::{tvalue::{TFunction, TValue, TString, TInteger, TFloat, TFnKind, TObject, FunctionFlags, TBool}, symbol::Symbol, parse::Ident, codegen, memory::GCRef, vm::{VM, TModule, Eternal}};
use index_vec::{IndexVec, define_index_type};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    // Null,
    // Bool(bool),
    // Int32(i32),
    Parameter(Parameter),
    Register(Register),
    Descriptor(Descriptor),
}

impl Operand {
    pub fn null() -> Self {
        // under the assumption that the first descriptor of every function is TValue::null()
        Operand::Descriptor(Descriptor::from_raw(0))
    }
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> FmtResult {
        match self {
            // Operand::Null => f.write_str("Null"),
            // Operand::Bool(bool) => write!(f, "Bool({bool})"),
            // Operand::Int32(int) => write!(f, "Int32({int})"),
            Operand::Parameter(arg) => write!(f, "{:?}", arg),
            Operand::Register(reg) => write!(f, "{:?}", reg),
            Operand::Descriptor(desc) => write!(f, "{desc:?}"),
        }
    }
}

define_index_type! {
    pub struct Parameter = u32;
    DEBUG_FORMAT = "~{}";
}

define_index_type! {
    pub struct Register = u32;
    DEBUG_FORMAT = "reg{}";
}

define_index_type! {
    pub struct Descriptor = u32;
    DEBUG_FORMAT = "${}";
}

#[derive(Clone, Copy)]
pub struct OperandList(pub *const [Operand]);

impl std::fmt::Debug for OperandList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> FmtResult {
        let slice = unsafe { &*self.0 };
        std::fmt::Debug::fmt(slice, f)
    }
}

pub struct CodeStream<'l> {
    begin: *const u8,
    current: *const u8,
    end: *const u8,
    blocks: &'l [u32],
}

impl<'l> CodeStream<'l> {
    pub fn from_raw(raw: &'l TRawCode) -> Self {
        let code = raw.code();
        let begin = code.as_ptr();
        Self {
            begin,
            current: begin,
            end: unsafe { code.as_ptr().add(code.len()) },
            blocks: raw.blocks()
        }
    }

    fn debug_from_data(code: &'l [u8])  -> Self {
        let begin = code.as_ptr();
        Self {
            begin,
            current: begin,
            end: unsafe { code.as_ptr().add(code.len()) },
            blocks: &[]
        }
    }

    #[inline(always)]
    pub fn eos(&self) -> bool {
        std::ptr::eq(self.current, self.end)
    }

    #[inline(always)]
    pub fn current(&self) -> u8 {
        unsafe { *self.current }
    }

    #[inline(always)]
    pub fn jump(&mut self, to: CodeLabel) {
        unsafe {
            let offset = self.blocks[to.index()] as usize;
            self.current = self.begin.add(offset);
        }
    }

    #[inline(always)]
    pub fn bump(&mut self, amount: usize) {
        unsafe {
            self.current = self.current.add(amount);
        }
    }

    #[inline(always)]
    pub fn read<T: Sized>(&self) -> &T {
        unsafe {
            &*(self.current as *const T)
        }
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
            let op: OpCode = unsafe { std::mem::transmute(stream.current()) };
            stream.bump(1);
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackingData {
    Param(Parameter),
    Reg(Register),
    Undefined,
}

#[derive(Clone, Copy)]
pub struct Local {
    pub constant: bool,
    pub backing_data: BackingData,
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
    name: Option<Symbol>,
    num_params: u32,
    num_locals: u32,
    max_arguments: u32,
    descriptor_table: IndexVec<Descriptor, TValue>,
    blocks: IndexVec<CodeLabel, BasicBlock>,
    current_block: CodeLabel,
    register_allocator: RegisterAllocator,
    ribs: Vec<Rib>,
}

impl CGFunction {

    fn module() -> Self {
        Self {
            name: None,
            ribs: vec![Rib::new(RibKind::Module)],
            num_params: 0,
            num_locals: 0,
            max_arguments: 1,
            descriptor_table: IndexVec::from_vec(vec![TValue::null()]),
            blocks: vec![BasicBlock::new()].into(),
            current_block: CodeLabel::from_raw(0),
            register_allocator: RegisterAllocator::new(),
        }
    }

    fn function(name: Symbol, _closure: bool, params: &[&Ident]) -> Result<Self, Ident> {
        let mut this = Self {
            name: Some(name),
            num_params: params.len() as u32,
            ribs: vec![Rib::new(RibKind::Function)],
            register_allocator: RegisterAllocator::new(),
            ..Self::module()
        };

        for (idx, param) in params.iter().enumerate() {
            let local = Local {
                constant: false,
                backing_data: BackingData::Param(Parameter::from_usize(idx))
            };
            if let Some(..) = this.current_rib_mut().locals.insert(param.symbol, local) {
                return Err((*param).clone());
            }
        }

        Ok(this)
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
                if let BackingData::Param(..) | BackingData::Reg(..) = local.backing_data {
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
            backing_data: BackingData::Undefined
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
        debug_assert!(local.backing_data == BackingData::Undefined);
        local.backing_data = BackingData::Reg(reg);
        Operand::Register(reg)
    }

    pub fn get_local_reg(&self, symbol: Symbol) -> Operand {
        let local = self.find_local(symbol).expect("symbol corresponds to actual local");
        match local.backing_data {
            BackingData::Param(arg) => Operand::Parameter(arg),
            BackingData::Reg(reg) => Operand::Register(reg),
            BackingData::Undefined => panic!("find_local finds declared locals")
        }
    }

    fn descriptor<T: Into<TValue>>(&mut self, tvalue: T) -> Operand {
        let idx = self.descriptor_table.len_idx();
        self.descriptor_table.push(tvalue.into());
        Operand::Descriptor(idx)
    }

    fn grow_args_buffer(&mut self, size: usize) {
        const MAX_ARGS: usize = 16;
        if size <= MAX_ARGS  {
            self.max_arguments = (size as u32).max(self.max_arguments);
        }
    }
}

pub struct BytecodeGenerator {
    module: GCRef<TModule>,
    root_fn: CGFunction,
    function_stack: Vec<CGFunction>
}

pub struct ClosureScope;

impl BytecodeGenerator {
    pub fn new(module: GCRef<TModule>) -> Self {
        Self {
            module,
            root_fn: CGFunction::module(),
            function_stack: Default::default()
        }
    }

    pub fn vm(&self) -> Eternal<VM> {
        self.module.vm()
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
    
        // struct N;
        // let mut n = N;
        // impl std::fmt::Write for N {
        //     fn write_str(&mut self, s: &str) -> FmtResult {
        //         print!("{s}");
        //         Ok(())
        //     }
        // }
        // FunctionDisassembler::dissassemble(&func, &mut n).unwrap();

        let func = TFunction::from_codegen(&self.vm(), func, self.module);
        if let Err(crate::vm::GlobalErr::Redeclared(..)) = self.module.set_global(name.symbol, func.into(), true) {
            todo!("codegen errors that are more like runtime errors? Keep going?");
        }

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
        Operand::Register(reg)
    }

    pub fn allocate_list(&mut self, operands: Vec<Operand>) -> OperandList {
        let operands = operands.into_boxed_slice();
        let operands = Box::leak(operands);
        OperandList(unsafe { &*operands })
    }

    pub fn declare_local(&mut self, symbol: Symbol) -> Operand {
        self.current_fn_mut()
            .declare_local(symbol)
    }

    pub fn get_local_reg(&self, symbol: Symbol) -> Operand {
        self.current_fn()
            .get_local_reg(symbol)
    }

    pub fn grow_args_buffer(&mut self, size: usize) {
        self.current_fn_mut()
            .grow_args_buffer(size)
    }

    pub fn make_string(&mut self, string: GCRef<TString>) -> Operand {
        self.current_fn_mut().descriptor(string)
    }

    pub fn make_i32(&mut self, int: i32) -> Operand {
        self.current_fn_mut().descriptor(TInteger::from_int32(int))
    }

    pub fn make_int(&mut self, int: u64) -> Operand {
        if let Ok(int) = i32::try_from(int) {
            return self.make_i32(int);
        }
        self.current_fn_mut().descriptor(TInteger::from_usize(int as usize))
    }

    pub fn make_bool(&mut self, bool: bool) -> Operand {
        self.current_fn_mut().descriptor(TBool::from_bool(bool))
    }

    pub fn make_float(&mut self, float: f64) -> Operand {
        self.current_fn_mut().descriptor(TFloat::from_float(float))
    }

    pub fn root_function(self) -> GCRef<TFunction> {
        TFunction::from_codegen(&self.vm(), self.root_fn, self.module)
    }
}

impl TFunction {
    pub fn from_codegen(vm: &VM, func: CGFunction, module: GCRef<TModule>) -> GCRef<Self> {
        let mut codesize = 0;
        for block in &func.blocks {
            codesize += block.data.len();
        }

        let name = func.name.map(|name| vm.symbols.get(name));
        let (function, tcode) = TFunction::create_presized(
            vm,
            name,
            module,
            codesize,
            func.num_params,
            func.register_allocator.0,
            func.max_arguments,
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
        name: Option<GCRef<TString>>,
        module: GCRef<TModule>,
        codesize: usize,
        params: u32,
        registers: u32,
        max_args: u32,
        descriptors: u32,
        blocks: u32
    ) -> (GCRef<TFunction>, &mut TRawCode) {
        let extra_size = codesize
            + descriptors as usize * std::mem::size_of::<TValue>()
            + blocks as usize * std::mem::size_of::<u32>();
        let code = TRawCode::new(codesize, params, registers, max_args, descriptors, blocks);

        let mut function: GCRef<Self> = vm.heap().allocate_var_atom(
            Self {
                base: TObject::base(vm, vm.types().query::<Self>()),
                name,
                module,
                kind: TFnKind::Function(code),
                flags: FunctionFlags::empty(),
            },
            extra_size
        );
        let function2 = function.clone();
        match function.kind {
            TFnKind::Function(ref mut code) => {
                (function2, unsafe { std::mem::transmute(code) })
            },
            _ => unreachable!()
        }
    }
}

#[repr(C)]
pub struct TRawCode {
    num_params: u32,
    num_registers: u32,
    max_args: u32,
    num_descriptors: u32,
    num_blocks: u32,
    codesize: usize,
    data: [u8; 0]
}

impl TRawCode {
    fn new(
        codesize: usize,
        num_params: u32,
        num_registers: u32,
        max_args: u32,
        num_descriptors: u32,
        num_blocks: u32
    ) -> Self {
        Self {
            codesize,
            num_params,
            num_registers,
            max_args,
            num_descriptors,
            num_blocks,
            data: [0u8; 0]
        }
    }

    pub fn registers(&self) -> usize {
        self.num_registers as usize
    }

    pub fn params(&self) -> usize {
        self.num_params as usize
    }

    pub fn max_args(&self) -> usize {
        self.max_args as usize
    }

    pub fn blocks(&self) -> &[u32] {
        unsafe {
            let len = self.num_blocks as usize;
            let offset = self.num_descriptors as usize
                * std::mem::size_of::<TValue>();
            std::slice::from_raw_parts(
                self.data.as_ptr().add(offset) as *const u32,
                len
            )
        }
    }

    pub fn blocks_mut(&mut self) -> &mut [u32] {
        unsafe {
            let len = self.num_blocks as usize;
            let offset = self.num_descriptors as usize
                * std::mem::size_of::<TValue>();
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(offset) as *mut u32,
                len
            )
        }
    }

    pub fn code(&self) -> &[u8] {
        unsafe {
            let len = self.codesize;
            let offset = self.num_descriptors as usize
                * std::mem::size_of::<TValue>()
                + self.num_blocks as usize
                * std::mem::size_of::<u32>();
            std::slice::from_raw_parts(
                self.data.as_ptr().add(offset),
                len
            )
        }
    }

    pub fn code_mut(&mut self) -> &mut [u8] {
        unsafe {
            let len = self.codesize as usize;
            let offset = self.num_descriptors as usize
                * std::mem::size_of::<TValue>()
                + self.num_blocks as usize
                * std::mem::size_of::<u32>();
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(offset),
                len
            )
        }
    }

    pub fn descriptors(&self) -> &[TValue] {
        unsafe {
            let len = self.num_descriptors as usize;
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const TValue,
                len as usize
            )
        }
    }

    pub fn descriptors_mut(&mut self) -> &mut [TValue] {
        unsafe {
            let len = self.num_descriptors as usize;
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut TValue,
                len
            )
        }
    }
}

pub trait Instruction: Sized + Copy {
    const CODE: OpCode;
    const IS_TERMINATOR: bool;
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

    Call {
        dst: Operand,
        callee: Operand,
        arguments: OperandList,
    },

    MethodCall {
        dst: Operand,
        this: Operand,
        callee: Symbol,
        arguments: OperandList,
    },

    GetIterator { dst: Operand, iterable: Operand },
    #[terminator(true)]
    NextIterator {
        dst: Operand,
        iterator: Operand,
        loop_target: CodeLabel,
        end_target: CodeLabel
    },

    MakeList {
        dst: Operand,
        items: OperandList
    },
    MakeEmptyList { dst: Operand },

    Error,
    Noop
}
