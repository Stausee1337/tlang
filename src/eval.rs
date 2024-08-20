use tlang_macros::decode;

use crate::{
    bytecode::{
        CodeLabel, CodeStream, Descriptor, OpCode, Operand, OperandKind, Register, TRawCode, Deserializer
    },
    memory::GCRef,
    symbol::Symbol,
    tvalue::{TBool, TFunction, TInteger, TValue, resolve_by_symbol},
    interop::{VMDowncast, TPropertyAccess},
    vm::{TModule, VM},
    debug
};

struct ExecutionEnvironment<'l> {
    descriptors: &'l [TValue],
    arguments: &'l mut [TValue],
    registers: &'l mut [TValue],
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

trait Decode {
    type Output;

    fn decode(&self, env: &ExecutionEnvironment) -> Self::Output;
}

trait DecodeMut {
    #[inline(always)]
    fn decode_mut<'l>(&self, _env: &'l mut ExecutionEnvironment) -> &'l mut TValue {
        panic!(
            "{} cannot be decoded as mutable",
            std::any::type_name_of_val(self)
        )
    }
}

trait DecodeDeref {
    fn decode_deref<'l>(
        &self,
        env: &'l ExecutionEnvironment,
    ) -> &'l impl Iterator<Item = &'l TValue>;
}

impl TRawCode {
    pub fn evaluate<'a>(&self, mut module: GCRef<TModule>, args: TArgsBuffer) -> TValue {
        self.with_environment(module, args, |vm, env, mut deserializer| {
            loop {
                let op: OpCode = deserializer.next().unwrap();
                debug!("{op:?}");
                match op {
                    OpCode::Mov => {
                        decode!(&mut deserializer, env, Mov { src, mut dst });
                        *dst = src;
                    }
                    OpCode::Add => impls::add(vm, env, &mut deserializer),
                    OpCode::Sub => impls::sub(vm, env, &mut deserializer),
                    OpCode::Mul => impls::mul(vm, env, &mut deserializer),
                    OpCode::Div => impls::mul(vm, env, &mut deserializer),
                    OpCode::Mod => impls::rem(vm, env, &mut deserializer),

                    OpCode::LeftShift => impls::shl(vm, env, &mut deserializer),
                    OpCode::RightShift => impls::shr(vm, env, &mut deserializer),

                    OpCode::BitwiseAnd => impls::bitand(vm, env, &mut deserializer),
                    OpCode::BitwiseOr => impls::bitor(vm, env, &mut deserializer),
                    OpCode::BitwiseXor => impls::bitxor(vm, env, &mut deserializer),

                    OpCode::Neg => impls::neg(vm, env, &mut deserializer),
                    OpCode::Not => impls::not(vm, env, &mut deserializer),
                    OpCode::Invert => impls::invert(vm, env, &mut deserializer),

                    OpCode::DeclareGlobal => {
                        decode!(&mut deserializer, env, DeclareGlobal { symbol, init, constant });
                        module.set_global(symbol, init, constant);
                    }

                    OpCode::GetGlobal => {
                        decode!(&mut deserializer, env, GetGlobal { symbol, mut dst });
                        *dst = module.get_global(symbol).unwrap();
                    }

                    OpCode::SetGlobal => {
                        decode!(&mut deserializer, env, SetGlobal { symbol, src });
                        *module.get_global_mut(symbol).unwrap() = src;
                    }

                    OpCode::GetAttribute => {
                        decode!(&mut deserializer, env, GetAttribute { base, attribute, mut dst });
                        let access: TPropertyAccess<TValue> = resolve_by_symbol(vm, attribute, base);
                        *dst = *access;
                    }

                    OpCode::SetAttribute => {
                        decode!(&mut deserializer, env, SetAttribute { base, attribute, src });
                        let mut access: TPropertyAccess<TValue> = resolve_by_symbol(vm, attribute, base);
                        *access = src;
                    }

                    OpCode::Branch => {
                        decode!(&mut deserializer, env, Branch { target });
                        deserializer.stream().jump(target);
                    }
                    OpCode::BranchIf => {
                        decode!(&mut deserializer, env, BranchIf { condition, true_target, false_target });
                        if impls::truthy(condition, env) {
                            deserializer.stream().jump(true_target);
                        } else {
                            deserializer.stream().jump(false_target);
                        }
                    }

                    OpCode::BranchEq => impls::eq(vm, env, &mut deserializer),
                    OpCode::BranchNe => impls::ne(vm, env, &mut deserializer),
                    OpCode::BranchGt => impls::gt(vm, env, &mut deserializer),
                    OpCode::BranchGe => impls::ge(vm, env, &mut deserializer),
                    OpCode::BranchLt => impls::lt(vm, env, &mut deserializer),
                    OpCode::BranchLe => impls::le(vm, env, &mut deserializer),

                    OpCode::Return => {
                        decode!(&mut deserializer, env, Return { value });
                        return value;
                    }

                    OpCode::Call => {
                        decode!(&mut deserializer, env, Call { arguments, callee, mut dst });
                        let callee: GCRef<TFunction> = VMDowncast::vmdowncast(callee, vm).unwrap();
                        *dst = callee.call(arguments);
                    }
                    _ => todo!(),
                }
            }
        })
    }

    fn with_environment<'a, F: FnOnce(&VM, &mut ExecutionEnvironment, Deserializer) -> TValue>(
        &self,
        module: GCRef<TModule>,
        mut arguments: TArgsBuffer,
        executor: F,
    ) -> TValue {
        let mut registers = vec![TValue::null(); self.registers()];
        let mut env = ExecutionEnvironment {
            descriptors: self.descriptors(),
            arguments: &mut arguments.0,
            registers: registers.as_mut_slice(),
        };
        let vm = module.vm();
        let mut stream = CodeStream::from_raw(self);
        let deserializer = Deserializer::new(&mut stream);
        executor(&vm, &mut env, deserializer)
    }
}

impl Decode for Operand {
    type Output = TValue;

    fn decode(&self, env: &ExecutionEnvironment) -> TValue {
        match self.to_rust() {
            OperandKind::Null => TValue::null(),
            OperandKind::Bool(bool) => TBool::from_bool(bool).into(),
            OperandKind::Register(reg) => reg.decode(env),
            OperandKind::Descriptor(desc) => desc.decode(env),
            OperandKind::Int32(int) => int.decode(env),
        }
    }
}

impl<'a> Decode for &'a [Operand] {
    type Output = TArgsBuffer;

    fn decode(&self, env: &ExecutionEnvironment) -> Self::Output {
        let mut arguments = Vec::new();
        for op in self.iter() {
            arguments.push(Decode::decode(op, env));
        }
        TArgsBuffer(arguments)
    }
}

impl DecodeMut for Operand {
    fn decode_mut<'l>(&self, env: &'l mut ExecutionEnvironment) -> &'l mut TValue {
        match self.to_rust() {
            OperandKind::Null => panic!("TValue::null() cannot be decoded as mutable"),
            OperandKind::Bool(bool) => bool.decode_mut(env),
            OperandKind::Register(reg) => reg.decode_mut(env),
            OperandKind::Descriptor(desc) => desc.decode_mut(env),
            OperandKind::Int32(int) => int.decode_mut(env),
        }
    }
}

impl Decode for bool {
    type Output = bool;

    #[inline(always)]
    fn decode(&self, _env: &ExecutionEnvironment) -> bool {
        *self
    }
}

impl DecodeMut for bool {}

impl Decode for Descriptor {
    type Output = TValue;

    #[inline(always)]
    fn decode(&self, env: &ExecutionEnvironment) -> TValue {
        env.descriptors[self.index()]
    }
}

impl DecodeMut for Descriptor {}

impl Decode for Register {
    type Output = TValue;

    #[inline(always)]
    fn decode(&self, env: &ExecutionEnvironment) -> TValue {
        let idx = self.index();
        debug!("  -> Decode {idx}");
        idx.checked_sub(env.arguments.len())
            .map(|idx| env.registers[idx])
            .unwrap_or_else(|| env.arguments[idx])
    }
}

impl DecodeMut for Register {
    #[inline(always)]
    fn decode_mut<'l>(&self, env: &'l mut ExecutionEnvironment) -> &'l mut TValue {
        let idx = self.index();
        debug!("  -> DecodeMut {idx}");
        idx.checked_sub(env.arguments.len())
            .map(|idx| env.registers.get_mut(idx).unwrap())
            .unwrap_or_else(|| env.arguments.get_mut(idx).unwrap())
    }
}

impl Decode for i32 {
    type Output = TValue;

    #[inline(always)]
    fn decode(&self, _env: &ExecutionEnvironment) -> TValue {
        TInteger::from_int32(*self).into()
    }
}

impl DecodeMut for i32 {}

impl Decode for CodeLabel {
    type Output = CodeLabel;

    fn decode(&self, _env: &ExecutionEnvironment) -> Self::Output {
        *self
    }
}

impl Decode for Symbol {
    type Output = Symbol;

    fn decode(&self, _env: &ExecutionEnvironment) -> Self::Output {
        *self
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

    macro_rules! arithmetic_impl {
        ($fnname: ident, $inname: ident) => {
            #[inline(always)]
            pub fn $fnname<'de>(vm: &VM, env: &mut ExecutionEnvironment, deserializer: &mut Deserializer<'de>) {
                decode!(deserializer, env, $inname { lhs, rhs, mut dst });
                let lhs_int: Option<TInteger> = lhs.query_integer();
                let rhs_int: Option<TInteger> = rhs.query_integer();
                if let Some((lhs, rhs)) = lhs_int.zip(rhs_int) {
                    *dst = lhs.$fnname(rhs).into();
                    return;
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

    macro_rules! branch_impl {
        ($fnname:ident, $inname:ident) => {    
            #[inline(always)]
            pub fn $fnname<'de>(vm: &VM, env: &ExecutionEnvironment, deserializer: &mut Deserializer<'de>) {
                decode!(deserializer, env, $inname { lhs, rhs, true_target, false_target });
                if TBool::as_bool(tcall!(vm, TValue::$fnname(lhs, rhs))) {
                    deserializer.stream().jump(true_target);
                } else {
                    deserializer.stream().jump(false_target);
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
    pub fn neg<'de>(vm: &VM, env: &mut ExecutionEnvironment, deserializer: &mut Deserializer<'de>) {
        decode!(deserializer, env, Neg { src, mut dst });
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
    pub fn invert<'de>(vm: &VM, env: &mut ExecutionEnvironment, deserializer: &mut Deserializer<'de>) {
        use crate::interop::vmops::Invert;

        decode!(deserializer, env, Invert { src, mut dst });
        if let Some(int) = src.query_integer() {
            *dst = int.invert().into();
            return;
        }
        *dst = tcall!(vm, TValue::invert(src));
    }

    #[inline(always)]
    pub fn not<'de>(vm: &VM, env: &mut ExecutionEnvironment, deserializer: &mut Deserializer<'de>) {
        decode!(deserializer, env, Not { src, mut dst });
        if let Some(int) = src.query_bool() {
            *dst = int.not().into();
            return;
        }
        *dst = tcall!(vm, TValue::not(src));
    }
}
