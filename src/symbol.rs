use std::{
    mem::{ManuallyDrop, transmute},
    fmt::Debug
};

use bumpalo::Bump;

use crate::{memory::GCRef, tvalue::TString};

pub struct SymbolInterner {
}

impl SymbolInterner {
    pub fn new() -> Self {
        Self {
        }
    }

    pub fn intern(&self, str: GCRef<TString>) -> Symbol {
        let str = str.make_static();
        todo!();
        /*if let Some(idx) = self.strings.get_index_of(str) {
            return Symbol(idx as u32);
        }

        let str = self.arena.alloc_str(str);
        let str: &'static str = unsafe { transmute(str) };

        let (idx, newly_inserted) = self.strings.insert_full(str);
        assert!(newly_inserted);

        Symbol(idx as u32)*/
    }

    pub fn get(&self, symbol: Symbol) -> GCRef<TString> {
        todo!();
        // self.strings.get_index(symbol.0 as usize).unwrap()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

impl Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}`", self.0)
    }
}

