use std::mem::MaybeUninit;

use crate::memory::BlockAllocator;

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

