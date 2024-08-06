use std::{mem::MaybeUninit, ops::Deref, slice::Iter};

use index_vec::Idx;
use tlang_macros::decode;

use crate::{memory::{BlockAllocator, GCRef}, tvalue::{TInteger, TValue, TBool}, bytecode::{TRawCode, OpCode, CodeStream, Operand, OperandKind, Descriptor, Register, DynamicArray}};

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
    fn decode(&self, env: &ExecutionEnvironment) -> TValue;
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
                    OpCode::Add => {
                        decode!(Add { mut dst, lhs, rhs } in env);
                        *dst = add_helper(lhs, rhs);
                    }
                    OpCode::Call => {
                        decode!(Call { mut dst, callee, &arguments } in env);
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

#[inline(always)]
fn add_helper(lhs: TValue, rhs: TValue) -> TValue {
    todo!()
}

impl Decode for Operand {
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
    #[inline(always)]
    fn decode(&self, _env: &ExecutionEnvironment) -> TValue {
        TBool::from_bool(*self).into()
    }
}

impl DecodeMut for bool {}

impl Decode for Descriptor {
    #[inline(always)]
    fn decode(&self, env: &ExecutionEnvironment) -> TValue {
        env.descriptors[self.index()]
    }
}

impl DecodeMut for Descriptor {}

impl Decode for Register {
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
    #[inline(always)]
    fn decode(&self, _env: &ExecutionEnvironment) -> TValue {
        TInteger::from_int32(*self).into() 
    }
}

impl DecodeMut for i32 {}


impl DecodeDeref for DynamicArray<Operand> {
    fn decode_deref<'l>(&self, _env: &'l ExecutionEnvironment) -> &'l Iter<'l, TValue> {
        todo!()
    }
}
