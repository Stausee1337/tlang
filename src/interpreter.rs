
pub struct CodeStream {
    position: usize,
    code: *const [u8]
}

impl CodeStream {
    pub fn debug_from_data(code: &[u8])  -> Self {
        Self {
            code,
            position: 0,
        }
    }

    #[inline(always)]
    pub fn eos(&self) -> bool {
        self.code().len() == 0
    }

    #[inline(always)]
    pub fn data(&self) -> &[u8] {
        unsafe { &*self.code }
    }

    #[inline(always)]
    pub fn code(&self) -> &[u8] {
        &self.data()[self.position..]
    }

    #[inline(always)]
    pub fn current(&self) -> u8 {
        self.data()[self.position]
    }

    #[inline(always)]
    pub fn bump(&mut self, amount: usize) {
        self.position += amount;
    }
}

