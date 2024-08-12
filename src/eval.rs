use tlang_macros::decode;

use crate::{
    bytecode::{
        CodeLabel, CodeStream, Descriptor, OpCode, Operand, OperandKind, Register, TRawCode,
    },
    memory::GCRef,
    symbol::Symbol,
    tvalue::{TBool, TFunction, TInteger, TValue, VMDowncast},
    vm::{TModule, VM},
};

struct ExecutionEnvironment<'l> {
    stream: CodeStream<'l>,
    vm: &'l VM,
    descriptors: &'l [TValue],
    arguments: &'l mut [TValue],
    registers: &'l mut [TValue],
}

pub struct TArgsBuffer(Vec<TValue>);

impl TArgsBuffer {
    pub fn empty() -> Self {
        TArgsBuffer(vec![])
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
    pub fn evaluate<'a>(&self, module: GCRef<TModule>, args: TArgsBuffer) -> TValue {
        self.with_environment(module, args, |env| {
            loop {
                let opcode = OpCode::decode(env.stream.current());
                println!("{opcode:?}");
                match opcode {
                    OpCode::Mov => {
                        decode!(Mov { src, mut dst } in env);
                        *dst = src;
                    }
                    OpCode::Add => impls::add(env),
                    OpCode::Sub => impls::sub(env),
                    OpCode::Mul => impls::mul(env),
                    OpCode::Div => impls::mul(env),
                    OpCode::Mod => impls::rem(env),

                    OpCode::LeftShift => impls::shl(env),
                    OpCode::RightShift => impls::shr(env),

                    OpCode::BitwiseAnd => impls::bitand(env),
                    OpCode::BitwiseOr => impls::bitor(env),
                    OpCode::BitwiseXor => impls::bitxor(env),

                    OpCode::GetGlobal => {
                        decode!(GetGlobal { symbol, mut dst } in env);
                        *dst = module.get_global(symbol).unwrap();
                    }

                    OpCode::Branch => {
                        decode!(Branch { target } in env);
                        env.stream.jump(target);
                    }
                    OpCode::BranchIf => {
                        decode!(BranchIf { condition, true_target, false_target } in env);
                        if impls::truthy(condition, env) {
                            env.stream.jump(true_target);
                        } else {
                            env.stream.jump(false_target);
                        }
                    }

                    OpCode::Return => {
                        decode!(Return { value } in env);
                        return value;
                    }

                    OpCode::Fallthrough => (),

                    OpCode::Call => {
                        decode!(Call { &arguments, callee, mut dst } in env);
                        let callee: GCRef<TFunction> = VMDowncast::vmdowncast(callee, &module.vm()).unwrap();
                        *dst = callee.call(arguments);
                    }
                    _ => todo!(),
                }
            }
        })
    }

    fn with_environment<'a, F: FnOnce(&mut ExecutionEnvironment) -> TValue>(
        &self,
        module: GCRef<TModule>,
        mut arguments: TArgsBuffer,
        executor: F,
    ) -> TValue {
        let vm = module.vm();

        let mut registers = vec![TValue::null(); self.registers()];
        let mut env = ExecutionEnvironment {
            vm: &vm,
            stream: CodeStream::from_raw(self),
            descriptors: self.descriptors(),
            arguments: &mut arguments.0,
            registers: registers.as_mut_slice(),
        };
        executor(&mut env)
    }
}

impl crate::bytecode::instructions::Call {
    fn decode_deref_arguments(&self, env: &ExecutionEnvironment) -> TArgsBuffer {
        let mut arguments = Vec::new();
        for op in self.arguments() {
            arguments.push(Decode::decode(op, env));
        }
        TArgsBuffer(arguments)
    }
}

impl Decode for Operand {
    type Output = TValue;

    fn decode(&self, env: &ExecutionEnvironment) -> TValue {
        match self.to_rust() {
            OperandKind::Null => TValue::null(),
            OperandKind::Bool(bool) => bool.decode(env),
            OperandKind::Register(reg) => reg.decode(env),
            OperandKind::Descriptor(desc) => desc.decode(env),
            OperandKind::Int32(int) => int.decode(env),
        }
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
    type Output = TValue;

    #[inline(always)]
    fn decode(&self, _env: &ExecutionEnvironment) -> TValue {
        TBool::from_bool(*self).into()
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
        println!("  -> Decode {idx}");
        idx.checked_sub(env.arguments.len())
            .map(|idx| env.registers[idx])
            .unwrap_or_else(|| env.arguments[idx])
    }
}

impl DecodeMut for Register {
    #[inline(always)]
    fn decode_mut<'l>(&self, env: &'l mut ExecutionEnvironment) -> &'l mut TValue {
        let idx = self.index();
        println!("  -> DecodeMut {idx}");
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
            pub fn $fnname(env: &mut ExecutionEnvironment) {
                let vm = env.vm;
                decode!($inname { lhs, rhs, mut dst } in env);
                let lhs_int: Option<TInteger> = lhs.query_integer(vm);
                let rhs_int: Option<TInteger> = rhs.query_integer(vm);
                if let Some((lhs, rhs)) = lhs_int.zip(rhs_int) {
                    *dst = lhs.$fnname(rhs).into();
                    return;
                }
                todo!()
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
}
