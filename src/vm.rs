use std::{rc::Rc, cell::OnceCell, any::TypeId};

use crate::{memory::{Heap, GCRef, StaticAtom}, symbol::SymbolInterner, tvalue::{TType, self}};

pub struct VM {
    heap: Box<Heap>,
    symbols: OnceCell<GCRef<SymbolInterner>>,
    types: OnceCell<GCRef<RustTypeInterner>>,

    pub hash_state: ahash::RandomState,
}

impl VM {
    pub fn init() -> Rc<VM> {
        let vm = Rc::new_cyclic(|me| {
            let heap = Box::new(Heap::init(me.clone()));
            let hash_state = ahash::RandomState::new();
            VM {
                heap,
                symbols: OnceCell::new(),

                hash_state,
                types: OnceCell::new() 
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

    pub fn types(&self) -> GCRef<RustTypeInterner> {
        *self.types.get_or_init(
            || StaticAtom::allocate(self.heap(), RustTypeInterner::new()))
    }
}

pub struct RustTypeInterner(hashbrown::HashMap<TypeId, GCRef<TType>>);

impl RustTypeInterner {
    fn new() -> Self {
        RustTypeInterner(Default::default())
    }

    pub fn query(&self, id: TypeId) -> GCRef<TType> {
        *self.0.get(&id).expect(
            &format!("expected query of interned rust type"))
    }
}
