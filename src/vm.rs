use std::rc::Rc;

use crate::{memory::Heap, symbol::SymbolInterner};

pub struct VM {
    heap: Box<Heap>,
    pub hash_state: ahash::RandomState,
    pub symbols: SymbolInterner
}

impl VM {
    pub fn init() -> Rc<VM> {
        let vm = Rc::new_cyclic(|me| {
            let heap = Box::new(Heap::init(me.clone()));
            let hash_state = ahash::RandomState::new();
            VM {
                hash_state,
                heap,
                symbols: SymbolInterner::new(),
            }
        });
        vm
    }

    pub fn heap(&self) -> &Heap {
        &self.heap
    }
}
