use std::mem::MaybeUninit;

use index_vec::Idx;

use crate::{memory::{BlockAllocator, GCRef}, tvalue::{TInteger, TValue, TBool}, bytecode::{TRawCode, OpCode, CodeStream, Operand, OperandKind, Descriptor, Register}};

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

impl TRawCode {
    pub fn evaluate(&self, arguments: &[TValue]) -> TValue {
        Self::with_environment(arguments, |env| {
            loop {
                match OpCode::decode(env.stream.current()) {
                    OpCode::Add => {
                        decode!(Add { mut dst, lhs, rhs } in env);
                    }
                    OpCode::Call => {
                        decode!(Call { mut dst, callee, &arguments } in env);
                    }
                    _ => todo!()
                }
            }
        })
    }

    fn with_environment<F: FnOnce(&mut ExecutionEnvironment) -> TValue>(arguments: &[TValue], executor: F) -> TValue {
        todo!()
    }
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
