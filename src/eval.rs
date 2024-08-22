
use tlang_macros::{decode, tcall, tget};

use crate::{
    bytecode::{
        CodeStream, OpCode, Operand, OperandKind, TRawCode, OperandList
    },
    memory::GCRef,
    tvalue::{TBool, TFunction, TInteger, TValue, resolve_by_symbol},
    interop::TPropertyAccess,
    vm::{TModule, VM},
    debug
};

struct ExecutionEnvironment<'l> {
    descriptors: &'l [TValue],
    registers: &'l mut [TValue],
}

impl<'l> ExecutionEnvironment<'l> {
    #[inline(always)]
    pub fn decode(&self, op: Operand) -> TValue {
        match op.to_rust() {
            OperandKind::Null => TValue::null(),
            OperandKind::Bool(bool) => TBool::from_bool(bool).into(),
            OperandKind::Register(reg) => self.registers[reg.index()],
            OperandKind::Descriptor(desc) => self.descriptors[desc.index()],
            OperandKind::Int32(int) => TInteger::from_int32(int).into(),
        }
    }

    pub fn decode_mut(&mut self, op: Operand) -> &mut TValue {
        let OperandKind::Register(reg) = op.to_rust() else {
            panic!("cannot be decoded as mutable");
        };
        &mut self.registers[reg.index()]
    }
}

pub struct TArgsBuffer(Vec<TValue>);

impl TArgsBuffer {
    pub fn empty() -> Self {
        TArgsBuffer(vec![])
    }

    pub fn new(v: Vec<TValue>) -> Self {
        TArgsBuffer(v)
    }

    pub fn into_iter(self, min: usize, varags: bool) -> TArgsIterator {
        if !varags {
            assert!(min == self.0.len());
        } else {
            assert!(min <= self.0.len());
        }
        TArgsIterator { inner: self.0, current: 0 }
    }
    
    pub fn prepend(self, tvalue: TValue) -> Self {
        let mut args = self.0;
        args.insert(0, tvalue);
        TArgsBuffer(args)
    }
}

pub struct TArgsIterator {
    inner: Vec<TValue>,
    current: usize
}

impl Iterator for TArgsIterator {
    type Item = TValue;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.inner.len() {
            return None; 
        }
        let argument = self.inner[self.current];
        self.current += 1;
        Some(argument)
    }
}

impl TArgsIterator {
    fn remaining(&mut self) -> &mut [TValue] {
        &mut self.inner[self.current..]
    }
}

impl TRawCode {
    pub fn evaluate<'a>(&self, mut module: GCRef<TModule>, arguments: TArgsBuffer) -> TValue {
        let mut registers = vec![TValue::null(); self.registers() + arguments.0.len()];
        for (idx, arg) in arguments.0.iter().enumerate() {
            registers[idx] = *arg;
        }
        let mut env = ExecutionEnvironment {
            descriptors: self.descriptors(),
            registers: registers.as_mut_slice(),
        };
        let vm = module.vm();
        let mut stream = CodeStream::from_raw(self);
        Self::inner_eval(module, &vm, &mut env, stream)
    }

    fn inner_eval(mut module: GCRef<TModule>, vm: &VM, env: &mut ExecutionEnvironment, mut stream: CodeStream) -> TValue {
        loop {
            let op: OpCode = unsafe { std::mem::transmute(stream.current()) };
            stream.bump(1);
            debug!("{op:?}");
            // let now = Instant::now();
            match op {
                OpCode::Mov => {
                    decode!(&mut stream, env, Mov { &src, &mut dst });
                    *dst = src;
                }
                OpCode::Add => impls::add(vm, env, &mut stream),
                OpCode::Sub => impls::sub(vm, env, &mut stream),
                OpCode::Mul => impls::mul(vm, env, &mut stream),
                OpCode::Div => impls::div(vm, env, &mut stream),
                OpCode::Mod => impls::rem(vm, env, &mut stream),

                OpCode::LeftShift => impls::shl(vm, env, &mut stream),
                OpCode::RightShift => impls::shr(vm, env, &mut stream),

                OpCode::BitwiseAnd => impls::bitand(vm, env, &mut stream),
                OpCode::BitwiseOr => impls::bitor(vm, env, &mut stream),
                OpCode::BitwiseXor => impls::bitxor(vm, env, &mut stream),

                OpCode::Neg => impls::neg(vm, env, &mut stream),
                OpCode::Not => impls::not(vm, env, &mut stream),
                OpCode::Invert => impls::invert(vm, env, &mut stream),

                OpCode::DeclareGlobal => {
                    decode!(&mut stream, env, DeclareGlobal { symbol, &init, constant });
                    module.set_global(symbol, init, constant);
                }

                OpCode::GetGlobal => {
                    decode!(&mut stream, env, GetGlobal { symbol, &mut dst });
                    *dst = module.get_global(symbol).unwrap();
                }

                OpCode::SetGlobal => {
                    decode!(&mut stream, env, SetGlobal { symbol, &src });
                    *module.get_global_mut(symbol).unwrap() = src;
                }

                OpCode::GetAttribute => {
                    decode!(&mut stream, env, GetAttribute { &base, attribute, &mut dst });
                    let access: TPropertyAccess<TValue> = resolve_by_symbol(vm, attribute, base, true);
                    if let Some(tfunction) = access.as_method() {
                        *dst = tfunction.bind(base).into();
                    } else {
                        *dst = *access;
                    }
                }

                OpCode::SetAttribute => {
                    decode!(&mut stream, env, SetAttribute { &base, attribute, &src });
                    let mut access: TPropertyAccess<TValue> = resolve_by_symbol(vm, attribute, base, true);
                    *access = src;
                }

                OpCode::GetSubscript => {
                    todo!("GetSubscript");
                }

                OpCode::SetSubscript => {
                    todo!("SetSubscript");
                }

                OpCode::Branch => {
                    decode!(&mut stream, env, Branch { target });
                    stream.jump(target);
                }
                OpCode::BranchIf => {
                    decode!(&mut stream, env, BranchIf { &condition, true_target, false_target });
                    if impls::truthy(condition, env) {
                        stream.jump(true_target);
                    } else {
                        stream.jump(false_target);
                    }
                }

                OpCode::BranchEq => impls::eq(vm, env, &mut stream),
                OpCode::BranchNe => impls::ne(vm, env, &mut stream),
                OpCode::BranchGt => impls::gt(vm, env, &mut stream),
                OpCode::BranchGe => impls::ge(vm, env, &mut stream),
                OpCode::BranchLt => impls::lt(vm, env, &mut stream),
                OpCode::BranchLe => impls::le(vm, env, &mut stream),

                OpCode::Return => {
                    decode!(&mut stream, env, Return { &value });
                    return value;
                }

                OpCode::Call => {
                    let envcopy: *const _ = &*env;
                    decode!(&mut stream, env, Call { arguments, &callee, &mut dst });
                    let arguments = arguments.decode(unsafe { &*envcopy });

                    if let Some(callee) = callee.query_object::<TFunction>(vm) {
                        *dst = callee.call(arguments);
                    } else {
                        todo!("dispatch other callable");
                    }
                }

                OpCode::MethodCall => {
                    let envcopy: *const _ = &*env;
                    decode!(&mut stream, env, MethodCall { arguments, &this, callee, &mut dst });
                    let arguments = arguments.decode(unsafe { &*envcopy });

                    let access: TPropertyAccess<TValue> = resolve_by_symbol(vm, callee, this, true);
                    if let Some(tfunction) = access.as_method() {
                        let arguments = arguments.prepend(this);
                        *dst = tfunction.call(arguments);
                    } else if let Some(tfunction) = (*access).query_object::<TFunction>(vm) {
                        *dst = tfunction.call(arguments);
                    } else {
                        todo!("dispatch other callable");
                    }
                }

                OpCode::GetIterator => {
                    decode!(&mut stream, env, GetIterator { &iterable, &mut dst });
                    *dst = tcall!(vm, TValue::get_iterator(iterable));
                }

                OpCode::NextIterator => {
                    decode!(&mut stream, env, NextIterator { &iterator, loop_target, end_target, &mut dst });
                    if TBool::as_bool(tcall!(vm, TValue::next(iterator))) {
                        *dst = *tget!(vm, TValue::current(iterator));
                        stream.jump(loop_target);
                    } else {
                        stream.jump(end_target);
                    }
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
    fn decode(&self, env: &ExecutionEnvironment) -> TArgsBuffer {
        let operands = unsafe { &*self.0 };
        let mut arguments = Vec::new();
        for op in operands {
            arguments.push(env.decode(*op));
        }
        TArgsBuffer(arguments)
    }
}

mod impls {
    use tlang_macros::tcall;

    use super::*;
    use std::ops::*;

    pub fn truthy(bool: TValue, env: &ExecutionEnvironment) -> bool {
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
            pub fn $fnname<'l>(vm: &VM, env: &mut ExecutionEnvironment, stream: &mut CodeStream<'l>) {
                decode!(stream, env, $inname { &lhs, &rhs, &mut dst });
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

    macro_rules! branch_impl {
        ($fnname:ident, $inname:ident) => {    
            #[inline(always)]
            pub fn $fnname<'l>(vm: &VM, env: &ExecutionEnvironment, stream: &mut CodeStream<'l>) {
                decode!(stream, env, $inname { &lhs, &rhs, true_target, false_target });
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
    pub fn neg<'l>(vm: &VM, env: &mut ExecutionEnvironment, stream: &mut CodeStream<'l>) {
        decode!(stream, env, Neg { &src, &mut dst });
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
    pub fn invert<'l>(vm: &VM, env: &mut ExecutionEnvironment, stream: &mut CodeStream<'l>) {
        use crate::interop::vmops::Invert;

        decode!(stream, env, Invert { &src, &mut dst });
        if let Some(int) = src.query_integer() {
            *dst = int.invert().into();
            return;
        }
        *dst = tcall!(vm, TValue::invert(src));
    }

    #[inline(always)]
    pub fn not<'l>(vm: &VM, env: &mut ExecutionEnvironment, stream: &mut CodeStream<'l>) {
        decode!(stream, env, Not { &src, &mut dst });
        if let Some(int) = src.query_bool() {
            *dst = int.not().into();
            return;
        }
        *dst = tcall!(vm, TValue::not(src));
    }
}
