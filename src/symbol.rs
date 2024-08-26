use std::fmt::Debug;

use ahash::RandomState;
use hashbrown::raw::RawTable;

use crate::{memory::{GCRef, Atom, Visitor}, tvalue::TString};

pub use tlang_macros::Symbol;

const HASH0: RandomState = RandomState::with_seeds(
    0x6735611e020820df,
    0x43abdc326e8e3a2c,
    0x80a2f5f207cf871e,
    0x43d1fac44245c038
);

const HASH1: RandomState = RandomState::with_seeds(
    0xb15ceb7218c4c1b5,
    0xae5bd30505b9c17e,
    0xb1c5d5e97339544a,
    0xedae3f360619d8f6
);

const HASH2: RandomState = RandomState::with_seeds(
    0x229533896b4d1f57,
    0x82d963a13dca2bb5,
    0x51a9b15708317482,
    0x382e6f20e2020ddf
);

const HASH3: RandomState = RandomState::with_seeds(
    0xd6cc5b074c253c8e,
    0x591a296cf00ad299,
    0x47848ccc54e51f03,
    0x29e940477b79df10
);

pub struct SymbolCache {
    table: RawTable<(Symbol, GCRef<TString>)>,
}

impl SymbolCache {
    pub fn new() -> Self {
        SymbolCache {
            table: RawTable::new(),
        }
    }

    #[inline]
    fn mkhash(&self, str: &str) -> u64 {
        HASH0.hash_one(str)
    }

    #[inline]
    fn mkid(&self, str: &str) -> u64 {
        HASH1.hash_one(str) ^ HASH2.hash_one(str) ^ HASH3.hash_one(str)
    }

    #[inline]
    fn mksym(&self, str: &str) -> Symbol {
        Symbol {
            id: self.mkid(str),
            hash: self.mkhash(str),
        }
    }

    /// Inserts a string into the cache, without checking for
    /// uniqness of the string
    fn cache_insert(&mut self, str: GCRef<TString>) -> Symbol {
        let symbol = self.mksym(str.as_slice());
        self.table.insert(symbol.hash, (symbol, str), |val| val.0.hash);
        symbol
    }

    fn cache_find(&self, str: &str) -> Option<Symbol> {
        let hash = self.mkhash(str);
        if let Some(bucket) = self.table.find(hash, 
            |val| val.1.as_slice().eq(str)) {
            return Some(unsafe { bucket.as_ref().0 });
        }
        None
    }

    pub fn intern(&mut self, str: GCRef<TString>) -> Symbol {
        if let Some(sym) = self.cache_find(str.as_slice()) {
            return sym;
        }
        self.cache_insert(str)
    }

    pub fn get(&self, sym: Symbol) -> GCRef<TString> {
        if let Some(bucket) = self.table.find(sym.hash,
            |val| val.0 == sym) {
            return unsafe { bucket.as_ref().1 };
        }
        panic!("tried to get unknown symbol");
    }
}

impl GCRef<SymbolCache> {
    pub fn intern_slice(&mut self, str: &str) -> Symbol {
        let vm = self.vm();
        if let Some(sym) = self.cache_find(str) {
            return sym;
        }
        self.cache_insert(TString::from_slice(&vm, str))
    }
}

impl Atom for SymbolCache {
    fn visit(&self, visitor: &mut Visitor) {
        unsafe {
            for bucket in self.table.iter() {
                visitor.feed(bucket.as_ref().1);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol {
    pub id: u64,
    pub hash: u64,
}

static_assertions::const_assert!(std::mem::size_of::<Symbol>() == 16);

impl Symbol {
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

