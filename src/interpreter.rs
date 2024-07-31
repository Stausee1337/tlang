
pub struct CodeStream {
    position: usize,
    code: *const [u8]
}

impl CodeStream {
    pub fn new(code: &[u8])  -> Self {
        Self {
            code,
            position: 0,
        }
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
        self.code()[self.position]
    }

    #[inline(always)]
    pub fn bump(&mut self, amount: usize) {
        debug_assert!(amount > 0);
        self.position += amount;
    }
}
