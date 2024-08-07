use std::{mem::MaybeUninit, slice::Iter};

use tlang_macros::decode;

use crate::{memory::BlockAllocator, tvalue::{TInteger, TValue, TBool}, bytecode::{TRawCode, OpCode, CodeStream, Operand, OperandKind, Descriptor, Register, CodeLabel}};

static mut INTERPTETER: Wrapper = Wrapper(false, MaybeUninit::uninit());

struct Wrapper(bool, MaybeUninit<TlInterpreter>);

pub struct TlInterpreter {
    pub block_allocator: BlockAllocator,
}

pub fn make_interpreter() -> &'static TlInterpreter {
    let interpreter = TlInterpreter {
        block_allocator: BlockAllocator::init()
    };
    unsafe {
        INTERPTETER = Wrapper(
            true,
            MaybeUninit::new(interpreter)
        );
    }
    get_interpeter()
}

pub fn get_interpeter() -> &'static TlInterpreter {
    unsafe { assert!(INTERPTETER.0, "Interpreter initialized") };
    unsafe { INTERPTETER.1.assume_init_ref() }
}

struct ExecutionEnvironment<'l> {
    stream: CodeStream<'l>,
    descriptors: &'l [TValue],
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
    pub fn evaluate<'a>(&self, arguments: &'a Iter<'a, TValue>) -> TValue {
        Self::with_environment(arguments, |env| {
            loop {
                match OpCode::decode(env.stream.current()) {
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

                    OpCode::Fallthrough => (),
                    _ => todo!()
                }
            }
        })
    }

    fn with_environment<'a, F: FnOnce(&mut ExecutionEnvironment) -> TValue>(
        arguments: &'a Iter<'a, TValue>, executor: F) -> TValue {
        todo!()
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
        env.registers[self.index()]
    }
}

impl DecodeMut for Register {
    #[inline(always)]
    fn decode_mut<'l>(&self, env: &'l mut ExecutionEnvironment) -> &'l mut TValue { 
        let index = self.index();
        &mut env.registers[index]
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

    macro_rules! arithmetic_impl {
        ($fnname: ident, $inname: ident) => {
            #[inline(always)]
            pub fn $fnname(env: &mut ExecutionEnvironment) {
                decode!($inname { mut dst, lhs, rhs } in env);
                let lhs_int: Option<TInteger> = TInteger::try_from(lhs).ok();
                let rhs_int: Option<TInteger> = TInteger::try_from(rhs).ok();
                if let Some((lhs, rhs)) = lhs_int.zip(rhs_int) {
                    *dst = TInteger::$fnname(lhs, rhs).into();
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
