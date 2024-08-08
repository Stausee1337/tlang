use std::{rc::Rc, cell::OnceCell};

use crate::{memory::{Heap, GCRef, StaticAtom}, symbol::SymbolInterner};

pub struct VM {
    heap: Box<Heap>,
    pub hash_state: ahash::RandomState,
    pub symbols: OnceCell<GCRef<SymbolInterner>>
}

impl VM {
    pub fn init() -> Rc<VM> {
        let vm = Rc::new_cyclic(|me| {
            let heap = Box::new(Heap::init(me.clone()));
            let hash_state = ahash::RandomState::new();
            VM {
                hash_state,
                heap,
                symbols: OnceCell::new(),
            }
        });
        vm
    }

    pub fn heap(&self) -> &Heap {
        &self.heap
    }

    pub fn symbols(&self) -> GCRef<SymbolInterner> {
        *self.symbols.get_or_init(
            || StaticAtom::allocate(self.heap(), SymbolInterner::new()))
    }
}
