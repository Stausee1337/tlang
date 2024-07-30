#[derive(Clone, Copy)]
pub enum ImmValue {
    Null,
    Bool(bool),
    Int32(i32),
    Register(u8),
    Descriptor(u32),
}

pub struct BasicBlock {
    data: Vec<u8>
}

pub struct BytecodeGenerator {
    blocks: Vec<BasicBlock>
}

impl BytecodeGenerator {
    pub fn new() -> Self {
        Self {
            blocks: Default::default()
        }
    }
}
