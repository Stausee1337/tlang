use std::{fmt::Debug, sync::Mutex};

use hashbrown::raw::RawTable;

use crate::{memory::GCRef, tvalue::{TString, GetHash}};

struct Cache {
    table: RawTable<usize>,
    entries: Vec<(GCRef<TString>, u64)>
}

impl Cache {
    fn new() -> Self {
        Cache {
            table: RawTable::new(),
            entries: Vec::new()
        }
    }

    pub fn cache(&mut self, str: GCRef<TString>) -> usize {
        let hash = str.get_hash_code();
        if let Some(bucket) = self.table.find(hash, |idx| {
            self.entries[*idx].1 == hash
        }) {
            return unsafe { *bucket.as_ref() };
        }
        let str = str.make_static();
        let idx = self.entries.len();
        self.entries.push((str, hash));
        self.table.insert(hash, idx, |idx| self.entries[*idx].1);
        idx
    }

    pub fn query(&self, idx: usize) -> Option<&(GCRef<TString>, u64)> {
        self.entries.get(idx)
    }
}

pub struct SymbolInterner {
    cache: Cache
}

impl SymbolInterner {
    pub fn new() -> Self {
        Self {
            cache: Cache::new()
        }
    }

    pub fn intern(&mut self, str: GCRef<TString>) -> Symbol {
        Symbol(self.cache.cache(str) as u32)
    }

    pub fn hash(&self, symbol: Symbol) -> u64 {
        let Some(cached) = self.cache.query(symbol.0 as usize) else {
            panic!("Invalid symbol");
        };
        cached.1
    }

    pub fn get(&self, symbol: Symbol) -> GCRef<TString> {
        let Some(cached) = self.cache.query(symbol.0 as usize) else {
            panic!("Invalid symbol");
        };
        cached.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

impl Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}`", self.0)
    }
}

