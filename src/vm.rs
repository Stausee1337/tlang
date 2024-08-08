use std::{rc::Rc, cell::OnceCell, any::TypeId};

use crate::{memory::{Heap, GCRef, StaticAtom}, symbol::{SymbolInterner, Symbol}, tvalue::{TType, self, TString, TValue}};

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

pub enum GlobalErr {
    Redeclared(Symbol),
    NotFound(Symbol),
    Constant(Symbol)
}

pub struct TModule {
    name: GCRef<TString>,
    source: TValue,
    table: hashbrown::HashTable<(Symbol, TValue, bool)>
}

impl TModule {
    pub fn new_from_rust(vm: &VM, modname: &str) -> GCRef<Self> {
        let modname = TString::from_slice(vm, modname);
        StaticAtom::allocate(vm.heap(), 
            Self {
                name: modname,
                source: TValue::null(),
                table: Default::default()
            }
        )
    }

    pub fn set_source(&mut self, source: GCRef<TString>) {
        self.source = source.into();
    }
}

impl GCRef<TModule> {
    pub fn set_global(&mut self, name: Symbol, value: TValue, constant: bool) -> Result<(), GlobalErr> {
        let symbols = self.vm().symbols();

        let Err(..) = self.get_global(name) else {
            return Err(GlobalErr::Redeclared(name));
        };

        self.table.insert_unique(
            symbols.hash(name),
            (name, value, constant),
            |value| symbols.hash(value.0)
        );
        Ok(())
    }

    pub fn get_global(&self, name: Symbol) -> Result<TValue, GlobalErr> {
        self.table.find(
            self.vm().symbols().hash(name),
            |entry| entry.0 == name
        ).map(|entry| entry.1).ok_or(GlobalErr::NotFound(name))
    }

    pub fn get_global_mut(&mut self, name: Symbol) -> Result<&mut TValue, GlobalErr> {
        let symbols = self.vm().symbols();
        let Some(entry) = self.table.find_mut(
            symbols.hash(name),
            |entry| entry.0 == name
        ) else {
            return Err(GlobalErr::NotFound(name))
        };

        if entry.2 {
            return Err(GlobalErr::Constant(name));
        }

        Ok(&mut entry.1)
    }
}
