use std::{
    mem::{ManuallyDrop, transmute},
    cell::RefCell,
    fmt::Debug
};

use bumpalo::Bump;
use ahash::RandomState;

type StringSet = indexmap::IndexSet<&'static str, RandomState>;

struct StringInterner {
    arena: ManuallyDrop<Bump>,
    strings: StringSet
}

impl StringInterner {
    fn new() -> Self {
        Self {
            arena: ManuallyDrop::new(Bump::new()),
            strings: Default::default() 
        }
    }

    fn add(&mut self, str: &str) -> u32 {
        if let Some(idx) = self.strings.get_index_of(str) {
            return idx as u32;
        }

        let str = self.arena.alloc_str(str);
        let str: &'static str = unsafe { transmute(str) };

        let (idx, newly_inserted) = self.strings.insert_full(str);
        assert!(newly_inserted);

        idx as u32
    }

    fn get(&self, idx: u32) -> &str {
        self.strings.get_index(idx as usize).unwrap()
    }
}


thread_local! {
    static INTERNED_STRINGS: RefCell<StringInterner> = RefCell::new(StringInterner::new());
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

impl Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}`", self.get())
    }
}

impl Symbol {
    pub fn intern(str: &str) -> Self {
        Self(INTERNED_STRINGS.with(|interner| interner.borrow_mut().add(str)))
    }

    pub fn get(&self) -> &str {
        INTERNED_STRINGS.with(|interner| unsafe {
            transmute(interner.borrow().get(self.0))
        })
    }
}
