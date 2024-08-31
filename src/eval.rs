
use bytemuck::Zeroable;
use tlang_macros::{decode, tcall, tget};

use crate::{
    bytecode::{
        CodeStream, OpCode, Operand, TRawCode, OperandList
    },
    memory::GCRef,
    tvalue::{TBool, TFunction, TInteger, TValue, resolve_by_symbol, TList},
    interop::TPropertyAccess,
    vm::{TModule, VM}, debug
};

macro_rules! get_stack_ptr {
    () => {
        unsafe {
            let x: usize;
            std::arch::asm! {
                "mov {}, rsp",
                out(reg) x
            };
            x
        }
    };
}

macro_rules! set_stack_ptr {
    ($x:expr) => {
        unsafe {
            let x = $x;
            std::arch::asm! {
                "mov rsp, {}",
                in(reg) x
            };
        }
    };
}

#[derive(Clone, Copy)]
struct FrameData {
    descriptors: *const [TValue],
    arguments: *mut [TValue],
    registers: *mut [TValue],
    buffer: *mut [TValue],
}

#[derive(Clone, Copy)]
pub struct StackFrame<'l> {
    vm: &'l VM,
    function: GCRef<TFunction>,
    data: FrameData
}

impl<'l> StackFrame<'l> {
    pub fn new(function: GCRef<TFunction>, args: TArgsBuffer) -> Self {
        let vm = function.vm();
        StackFrame {
            vm: unsafe { std::mem::transmute::<_, &'l VM>(vm) },
            function,
            data: FrameData {
                arguments: args.0,
                descriptors: &[],
                registers: &mut [],
                buffer: &mut []
            }
        }
    }

    #[inline(always)]
    fn registers(self) -> &'l mut [TValue] {
        unsafe { &mut *self.data.registers }
    }

    #[inline(always)]
    fn buffer(self) -> &'l mut [TValue] {
        unsafe { &mut *self.data.buffer }
    }

    #[inline(always)]
    fn arguments(self) -> &'l mut [TValue] {
        unsafe { &mut *self.data.arguments }
    }

    #[inline(always)]
    fn descriptors(self) -> &'l [TValue] {
        unsafe { &*self.data.descriptors }
    }

    #[inline(always)]
    fn module(self) -> GCRef<TModule> {
        self.function.module
    }

    #[inline(always)]
    fn alloc<F: FnOnce(Self) -> R, R>(mut self, code: &TRawCode, f: F) -> R {
        assert!(self.arguments().len() >= code.params());

        let mut aligned_num = code.registers() + code.max_args() + 3;
        if aligned_num & 0b1 != 0 {
            // padding
            aligned_num = aligned_num + 1;
        }

        let alloc_size = aligned_num << 3;

        let rsp = get_stack_ptr!();
        set_stack_ptr!(rsp - alloc_size);

        let (regs, buffer) = unsafe {
            let begin = (rsp - alloc_size) as *mut TValue;
            *(begin as *mut usize) = 0xdeadbeef;
            *begin.add(1) = TValue::object(self.function);
            // *(begin.add(2) as *mut usize) = 5;

            let regs = std::slice::from_raw_parts_mut(
                begin.add(2), code.registers());

            let buffer = std::slice::from_raw_parts_mut(
                regs.as_mut_ptr().add(code.registers()), code.max_args());
            assert!(code.max_args() > 0);

            *(rsp as *mut TValue) = TValue::zeroed(); // Sentinel
            (regs, buffer)
        };

        regs.fill(TValue::null());
        buffer.fill(TValue::zeroed());

        self.data.registers = regs;
        self.data.buffer = buffer;
        self.data.descriptors = code.descriptors();
        let rv = f(self);

        let rsp = get_stack_ptr!();
        set_stack_ptr!(rsp + alloc_size);

        unsafe {
            let begin = rsp as *mut usize;
            assert!(*begin == 0xdeadbeef);
            *begin = 0x0;
        }

        rv
    }

    #[inline(always)]
    fn decode(self, op: Operand) -> TValue {
        match op {
            Operand::Null => TValue::null(),
            Operand::Bool(bool) => TBool::from_bool(bool).into(),
            Operand::Register(reg) => {
                let idx = reg.index();
                let num_args = self.arguments().len();
                if idx < num_args {
                    return self.arguments()[idx];
                }
                self.registers()[idx - num_args]
            }
            Operand::Descriptor(desc) => self.descriptors()[desc.index()],
            Operand::Int32(int) => TInteger::from_int32(int).into(),
        }
    }

    #[inline(always)]
    fn decode_mut(mut self, op: Operand) -> &'l mut TValue {
        let Operand::Register(reg) = op else {
            panic!("cannot be decoded as mutable");
        };
        let idx = reg.index();
        let num_args = self.arguments().len();
        if idx < num_args {
            return &mut self.arguments()[idx];
        }
        &mut self.registers()[idx - num_args]
    }
}

pub struct TArgsBuffer(*mut [TValue]);

impl TArgsBuffer {
    pub fn empty() -> Self {
        unsafe {
            TArgsBuffer(&mut [])
        }
    }

    pub fn create(x: &mut [TValue]) -> Self {
        unsafe { Self(&mut *x) }
    }

    pub fn gc_alloc<const SIZE: usize>(vm: &VM, args: [TValue; SIZE]) -> Self {
        let mut list = TList::new_with_capacity(vm, SIZE + 1);
        for val in args {
            list.push(val);
        }
        list.push(list.into());
        
        unsafe {
            Self(
                std::slice::from_raw_parts_mut(
                    list.data_ptr(),
                    SIZE
                )
            )
        }
    }

    #[inline]
    pub fn into_iter(self, min: usize, varags: bool) -> TArgsIterator {
        unsafe {
            if !varags {
                assert_eq!(min, self.0.len());
            } else {
                assert!(min <= self.0.len());
            }
        }
        TArgsIterator { inner: self.0, current: 0 }
    }    
}

pub struct TArgsIterator {
    inner: *mut [TValue],
    current: usize
}

impl Iterator for TArgsIterator {
    type Item = TValue;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.inner.len() {
            return None; 
        }
        unsafe {
            let argument = (&*self.inner)[self.current];
            self.current += 1;
            Some(argument)
        }
    }
}

impl TArgsIterator {
    fn remaining(&mut self) -> &mut [TValue] {
        unsafe { &mut (&mut *self.inner)[self.current..] }
    }
}

extern "C" fn get_stack_ptr() -> *const u8 {
    unsafe {
        std::arch::asm! {
            "mov rax, QWORD PTR [rsp]",
            "ret",
            options(noreturn)
        };
    }
}

macro_rules! stack_push {
    ($x:expr) => {
        unsafe {
            let x = $x;
            std::arch::asm! {
                "push {}",
                in(reg) x
            };
        }
    };
}

macro_rules! stack_pop {
    () => {
        unsafe {
            let x: usize;
            std::arch::asm! {
                "pop {}",
                out(reg) x
            };
            x
        }
    };
}

// StackFrame::new(GCRef<TFunction> function /* function, module & vm */, TArgsBuffer arguments)

impl TRawCode {
    #[inline(never)]
    pub fn evaluate(&self, mut frame: StackFrame) -> TValue {
        // CUSTOM STACK LAYOUT
        // +------------------+
        // +    0xdeadbeef    +
        // +------------------+
        // + GCRef<FrameInfo> +
        // +------------------+
        // +       reg0       +
        // +------------------+
        // +       reg1       +
        // +------------------+
        // +       ....       +
        // +------------------+
        // +       regN       +
        // +------------------+
        // +      argBuf0     +
        // +------------------+
        // +      argBuf1     +
        // +------------------+
        // +       ....       +
        // +------------------+
        // +      argBufN     +
        // +------------------+
        // +  0x0 (sentinel)  +
        // +------------------+
        // +                  +
        // +     *padding*    +
        // +                  +
        // +------------------+
        
        let mut stream = CodeStream::from_raw(self);
        frame.alloc(self, |frame| { 
            Self::inner_eval(frame, &mut stream)
        })
    }

    #[inline(always)]
    fn inner_eval(mut frame: StackFrame, stream: &mut CodeStream) -> TValue {
        loop {
            let op: OpCode = unsafe { std::mem::transmute(stream.current()) };
            stream.bump(1);
            // let now = Instant::now();
            match op {
                OpCode::Mov => {
                    decode!(stream, frame, Mov { &src, &mut dst });
                    *dst = src;
                }
                OpCode::Add => impls::add(frame, stream),
                OpCode::Sub => impls::sub(frame, stream),
                OpCode::Mul => impls::mul(frame, stream),
                OpCode::Div => impls::div(frame, stream),
                OpCode::Mod => impls::rem(frame, stream),

                OpCode::LeftShift => impls::shl(frame, stream),
                OpCode::RightShift => impls::shr(frame, stream),

                OpCode::BitwiseAnd => impls::bitand(frame, stream),
                OpCode::BitwiseOr => impls::bitor(frame, stream),
                OpCode::BitwiseXor => impls::bitxor(frame, stream),

                OpCode::Neg => impls::neg(frame, stream),
                OpCode::Not => impls::not(frame, stream),
                OpCode::Invert => impls::invert(frame, stream),

                OpCode::DeclareGlobal => {
                    let mut module = frame.module();
                    decode!(stream, frame, DeclareGlobal { symbol, &init, constant });
                    module.set_global(symbol, init, constant).unwrap();
                }

                OpCode::GetGlobal => {
                    let module = frame.module();
                    decode!(stream, frame, GetGlobal { symbol, &mut dst });
                    *dst = module.get_global(symbol).unwrap();
                }

                OpCode::SetGlobal => {
                    let mut module = frame.module();
                    decode!(stream, frame, SetGlobal { symbol, &src });
                    *module.get_global_mut(symbol).unwrap() = src;
                }

                OpCode::GetAttribute => {
                    let vm = frame.vm;
                    decode!(stream, frame, GetAttribute { &base, attribute, &mut dst });
                    let access: TPropertyAccess<TValue> = resolve_by_symbol(vm, attribute, base, true);
                    if let Some(tfunction) = access.as_method() {
                        *dst = tfunction.bind(base).into();
                    } else {
                        *dst = *access;
                    }
                }

                OpCode::SetAttribute => {
                    let vm = frame.vm;
                    decode!(stream, frame, SetAttribute { &base, attribute, &src });
                    let mut access: TPropertyAccess<TValue> = resolve_by_symbol(vm, attribute, base, true);
                    *access = src;
                }

                OpCode::GetSubscript => {
                    let vm = frame.vm;
                    decode!(stream, frame, GetSubscript { &base, &index, &mut dst });
                    if let Some(index) = index.query_integer() {
                        if let Some(list) = base.query_list() {
                            *dst = list[index];
                            continue;
                        }
                    }
                    *dst = tcall!(vm, TValue::get_index(base, index));
                }

                OpCode::SetSubscript => {
                    let vm = frame.vm;
                    decode!(stream, frame, SetSubscript { &base, &index, &src });
                    if let Some(index) = index.query_integer() {
                        if let Some(mut list) = base.query_list() {
                            list[index] = src;
                            continue;
                        }
                    }
                    let _: () = tcall!(vm, TValue::set_index(base, index, value));
                }

                OpCode::Branch => {
                    decode!(stream, frame, Branch { target });
                    stream.jump(target);
                }
                OpCode::BranchIf => {
                    decode!(stream, frame, BranchIf { &condition, true_target, false_target });
                    if impls::truthy(condition, frame) {
                        stream.jump(true_target);
                    } else {
                        stream.jump(false_target);
                    }
                }

                OpCode::BranchEq => impls::eq(frame, stream),
                OpCode::BranchNe => impls::ne(frame, stream),
                OpCode::BranchGt => impls::gt(frame, stream),
                OpCode::BranchGe => impls::ge(frame, stream),
                OpCode::BranchLt => impls::lt(frame, stream),
                OpCode::BranchLe => impls::le(frame, stream),

                OpCode::Return => {
                    decode!(stream, frame, Return { &value });
                    return value;
                }

                OpCode::Call => {
                    let vm = frame.vm;
                    decode!(stream, frame, Call { arguments, &callee, &mut dst });

                    if let Some(callee) = callee.query_object::<TFunction>(vm) {
                        arguments.decode_into(0, frame);
                        *dst = callee.call(TArgsBuffer::create(&mut frame.buffer()[..arguments.len()]));
                        frame.buffer()[0] = TValue::zeroed(); // Sentinel
                    } else {
                        todo!("dispatch other callable");
                    }
                }

                OpCode::MethodCall => {
                    let vm = frame.vm;
                    decode!(stream, frame, MethodCall { arguments, &this, callee, &mut dst });

                    let access: TPropertyAccess<TValue> = resolve_by_symbol(vm, callee, this, true);
                    if let Some(callee) = access.as_method() {
                        frame.buffer()[0] = this;
                        arguments.decode_into(1, frame);
                        *dst = callee.call(TArgsBuffer::create(&mut frame.buffer()[..arguments.len() + 1]));
                        frame.buffer()[0] = TValue::zeroed(); // Sentinel
                    } else if let Some(callee) = (*access).query_object::<TFunction>(vm) {
                        arguments.decode_into(0, frame);
                        *dst = callee.call(TArgsBuffer::create(&mut frame.buffer()[..arguments.len()]));
                        frame.buffer()[0] = TValue::zeroed(); // Sentinel
                    } else {
                        todo!("dispatch other callable");
                    }
                }

                OpCode::GetIterator => {
                    let vm = frame.vm;
                    decode!(stream, frame, GetIterator { &iterable, &mut dst });
                    *dst = tcall!(vm, TValue::get_iterator(iterable));
                }

                OpCode::NextIterator => {
                    let vm = frame.vm;
                    decode!(stream, frame, NextIterator { &iterator, loop_target, end_target, &mut dst });
                    if TBool::as_bool(tcall!(vm, TValue::next(iterator))) {
                        *dst = *tget!(vm, TValue::current(iterator));
                        stream.jump(loop_target);
                    } else {
                        stream.jump(end_target);
                    }
                }

                OpCode::MakeList => {
                    let vm = frame.vm;
                    decode!(stream, frame, MakeList { items, &mut dst });
                    let list = TList::new_with_capacity(vm, items.len());
                    for item in items.iter() {
                        list.push(frame.decode(*item));
                    }
                    *dst = list.into();
                }

                OpCode::MakeEmptyList => {
                    let vm = frame.vm;
                    decode!(stream, frame, MakeEmptyList { &mut dst });
                    *dst = TList::new_empty(vm).into();
                }

                OpCode::Error => {
                    panic!("Error");
                }
                OpCode::Noop => (),
            }
            // debug!("perf {:?}", now.elapsed());
        }
    }
}

impl OperandList {
    #[inline(always)]
    fn decode_into(&self, offset: usize, mut frame: StackFrame) {
        let operands = unsafe { &*self.0 };
        for (idx, op) in operands.iter().enumerate() {
            frame.buffer()[offset + idx] = frame.decode(*op);
        }
        if (operands.len() + offset) < frame.buffer().len() {
            frame.buffer()[offset + operands.len()] = TValue::zeroed();
        }
    }

    fn len(&self) -> usize {
        unsafe { &*self.0 }.len()
    }

    fn iter(&self) -> impl std::iter::Iterator<Item = &Operand> {
        unsafe { &*self.0 }.iter()
    }
}

mod impls {
    use tlang_macros::tcall;

    use super::*;
    use std::ops::*;

    pub fn truthy(bool: TValue, frame: StackFrame) -> bool {
        if let Some(tbool) = bool.query_bool() {
            return tbool.as_bool();
        }
        todo!()
    }

    macro_rules! float_arithmetic {
        (add => $($body:tt)*) => { $($body)* };
        (sub => $($body:tt)*) => { $($body)* };
        (mul => $($body:tt)*) => { $($body)* };
        (div => $($body:tt)*) => { $($body)* };
        (rem => $($body:tt)*) => { $($body)* };
        ($_:ident => $($body:tt)*) => { };
    }

    macro_rules! arithmetic_impl {
        ($fnname: ident, $inname: ident) => {
            #[inline(always)]
            pub fn $fnname<'l>(mut frame: StackFrame, stream: &mut CodeStream<'l>) {
                let vm = frame.vm;
                decode!(stream, frame, $inname { &lhs, &rhs, &mut dst });
                let lhs_int: Option<TInteger> = lhs.query_integer();
                let rhs_int: Option<TInteger> = rhs.query_integer();
                if let Some((lhs, rhs)) = lhs_int.zip(rhs_int) {
                    *dst = lhs.$fnname(rhs).into();
                    return;
                }
                float_arithmetic! { $fnname =>
                    if let Some(lhs) = lhs.query_float() {
                        *dst = lhs.$fnname(rhs);
                        return;
                    }
                }
                *dst = tcall!(vm, TValue::$fnname(lhs, rhs));
            }
        };
    }

    macro_rules! iterate_arithmetics {
        ($(impl $fnname:ident for $inname:ident;)*) => {
            $(arithmetic_impl!($fnname, $inname);)*
        };
    }

    iterate_arithmetics! {
        impl add for Add; impl sub for Sub;
        impl mul for Mul; impl div for Div; impl rem for Mod;

        impl shl for LeftShift; impl shr for RightShift;
        impl bitand for BitwiseAnd; impl bitor for BitwiseOr; impl bitxor for BitwiseXor;
    }

    macro_rules! if_integer_comp {
        (eq => $($body:tt)*) => {};
        (ne => $($body:tt)*) => {};
        ($_:ident => $($body:tt)*) => {
            { $($body)* }
        };
    }

    macro_rules! if_general_comp {
        (eq => $($body:tt)*) => { { $($body)* } };
        (ne => $($body:tt)*) => { { $($body)* } };
        ($_:ident => $($body:tt)*) => {};
    }

    macro_rules! branch_impl {
        ($fnname:ident, $inname:ident) => {    
            #[inline(always)]
            pub fn $fnname<'l>(frame: StackFrame, stream: &mut CodeStream<'l>) {
                let vm = frame.vm;
                decode!(stream, frame, $inname { &lhs, &rhs, true_target, false_target });
                if_general_comp! { $fnname =>
                    use std::cmp::*;

                    if lhs.is_null() || rhs.is_null() {
                        if lhs.encoded().$fnname(&rhs.encoded()) {
                            stream.jump(true_target);
                        } else {
                            stream.jump(false_target);
                        }
                        return;
                    }
                }
                if_integer_comp! { $fnname =>
                    use $crate::interop::vmops::*;

                    let lhs_int: Option<TInteger> = lhs.query_integer();
                    let rhs_int: Option<TInteger> = rhs.query_integer();
                    if let Some((lhs, rhs)) = lhs_int.zip(rhs_int) {
                        if lhs.$fnname(rhs).as_bool() {
                            stream.jump(true_target);
                        } else {
                            stream.jump(false_target);
                        }
                        return;
                    }
                }
                if TBool::as_bool(tcall!(vm, TValue::$fnname(lhs, rhs))) {
                    stream.jump(true_target);
                } else {
                    stream.jump(false_target);
                }
            }
        };
    }

    macro_rules! iterate_branches {
        ($(impl $fnname:ident for $inname:ident;)*) => {
            $(branch_impl!($fnname, $inname);)*
        };
    }

    iterate_branches! {
        impl eq for BranchEq; impl ne for BranchNe;
        impl gt for BranchGt; impl ge for BranchGe;
        impl lt for BranchLt; impl le for BranchLe;
    }

    #[inline(always)]
    pub fn neg<'l>(mut frame: StackFrame, stream: &mut CodeStream<'l>) {
        let vm = frame.vm;
        decode!(stream, frame, Neg { &src, &mut dst });
        if let Some(int) = src.query_integer() {
            *dst = int.neg().into();
            return;
        } else if let Some(float) = src.query_float() {
            *dst = float.neg().into();
            return;
        }
        *dst = tcall!(vm, TValue::neg(src));
    }

    #[inline(always)]
    pub fn invert<'l>(mut frame: StackFrame, stream: &mut CodeStream<'l>) {
        use crate::interop::vmops::Invert;

        let vm = frame.vm;
        decode!(stream, frame, Invert { &src, &mut dst });
        if let Some(int) = src.query_integer() {
            *dst = int.invert().into();
            return;
        }
        *dst = tcall!(vm, TValue::invert(src));
    }

    #[inline(always)]
    pub fn not<'l>(mut frame: StackFrame, stream: &mut CodeStream<'l>) {
        let vm = frame.vm;
        decode!(stream, frame, Not { &src, &mut dst });
        if let Some(int) = src.query_bool() {
            *dst = int.not().into();
            return;
        }
        *dst = tcall!(vm, TValue::not(src));
    }
}
