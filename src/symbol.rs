use std::{fmt::Debug, cell::RefCell, sync::Mutex};

use hashbrown::raw::RawTable;

use crate::{memory::GCRef, tvalue::{TString, GetHash}};

struct Cache {
    table: RawTable<(GCRef<TString>, u64)>
}

impl Cache {
    fn new() -> Self {
        Cache {
            table: RawTable::new()
        }
    }

    pub fn cache(&mut self, str: GCRef<TString>) -> usize {
        let hash = str.get_hash_code();
        if let Some(bucket) = self.table.find(hash, |val| val.0 == str) {
            return unsafe { self.table.bucket_index(&bucket) };
        }
        let str = str.make_static();
        let bucket = self.table.insert(hash, (str, hash), |val| val.1);
        unsafe { self.table.bucket_index(&bucket) }
    }

    pub fn query(&self, idx: usize) -> Option<&(GCRef<TString>, u64)> {
        if idx < self.table.buckets() {
            unsafe {
                let bucket = self.table.bucket(idx);
                return Some(bucket.as_ref())
            }
        }
        None
    }
}

pub struct SymbolInterner {
    cache: Mutex<Cache>
}

impl SymbolInterner {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(Cache::new())
        }
    }

    pub fn intern(&mut self, str: GCRef<TString>) -> Symbol {
        let cache = self.cache.get_mut().unwrap();
        Symbol(cache.cache(str) as u32)
    }

    pub fn hash(&self, symbol: Symbol) -> u64 {
        let cache = self.cache.lock().unwrap();
        let Some(cached) = cache.query(symbol.0 as usize) else {
            panic!("Invalid symbol");
        };
        cached.1
    }

    pub fn get(&self, symbol: Symbol) -> GCRef<TString> {
        let cache = self.cache.lock().unwrap();
        let Some(cached) = cache.query(symbol.0 as usize) else {
            panic!("Invalid symbol");
        };
        cached.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

pub fn test() -> Symbol {
    Symbol(0)
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}`", self.0)
    }
}

