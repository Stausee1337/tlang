
use std::{rc::Rc, hash::BuildHasher};

use tlang_macros::decode;

use crate::{memory::Heap, tvalue::{TInteger, TValue, TBool}, bytecode::{TRawCode, OpCode, CodeStream, Operand, OperandKind, Descriptor, Register, CodeLabel}, symbol::SymbolInterner};

pub struct VM {
    heap: Box<Heap>,
    pub hash_state: ahash::RandomState,
    pub symbols: SymbolInterner
}

impl VM {
    pub fn init() -> Rc<VM> {
        let vm = Rc::new_cyclic(|me| {
            let heap = Box::new(Heap::init(me.clone()));
            let hash_state = ahash::RandomState::new();
            VM {
                hash_state,
                heap,
                symbols: SymbolInterner::new(),
            }
        });
        vm
    }

    pub fn heap(&self) -> &Heap {
        &self.heap
    }
}

struct ExecutionEnvironment<'l> {
    stream: CodeStream<'l>,
    vm: &'l VM,
    descriptors: &'l [TValue],
    arguments: &'l mut [TValue],
    registers: &'l mut [TValue]
}

trait Decode {
    type Output;

    fn decode(&self, env: &ExecutionEnvironment) -> Self::Output;
}

trait DecodeMut {
    #[inline(always)]
    fn decode_mut<'l>(&self, env: &'l mut ExecutionEnvironment) -> &'l mut TValue {
        panic!("{} cannot be decoded as mutable", std::any::type_name_of_val(self))
    }
}

trait DecodeDeref {
    fn decode_deref<'l>(&self, env: &'l ExecutionEnvironment) -> &'l impl Iterator<Item = &'l TValue>;
}

impl TRawCode {
    pub fn evaluate<'a>(&self, vm: &VM, arguments: &'a mut [TValue]) -> TValue {
        self.with_environment(vm, arguments, |env| {
            loop {
                match OpCode::decode(env.stream.current()) {
                    OpCode::Mov => {
                        decode!(Mov { src, mut dst } in env);
                        *dst = src;
                    }
                    OpCode::Add => impls::add(env), OpCode::Sub => impls::sub(env),
                    OpCode::Mul => impls::mul(env), OpCode::Div => impls::mul(env),
                    OpCode::Mod => impls::rem(env),

                    OpCode::LeftShift => impls::shl(env), OpCode::RightShift => impls::shr(env),

                    OpCode::BitwiseAnd => impls::bitand(env), OpCode::BitwiseOr => impls::bitor(env),
                    OpCode::BitwiseXor => impls::bitxor(env),

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
                    _ => todo!()
                }
            }
        })
    }

    fn with_environment<'a, F: FnOnce(&mut ExecutionEnvironment) -> TValue>(
        &self, vm: &VM, arguments: &'a mut [TValue], executor: F) -> TValue {
        debug_assert!(arguments.len() == self.params());

        let mut registers = vec![TValue::null(); self.registers()];
        let mut env = ExecutionEnvironment {
            vm,
            stream: CodeStream::from_raw(self),
            descriptors: self.descriptors(),
            arguments,
            registers: registers.as_mut_slice()
        };
        executor(&mut env)
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
        idx.checked_sub(env.arguments.len())
            .map(|idx| env.registers[idx])
            .unwrap_or_else(|| env.arguments[idx])
    }
}

impl DecodeMut for Register {
    #[inline(always)]
    fn decode_mut<'l>(&self, env: &'l mut ExecutionEnvironment) -> &'l mut TValue { 
        let idx = self.index();
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
                    *dst = TInteger::$fnname(lhs, rhs).into();
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
