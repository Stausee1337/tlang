use std::{rc::Rc, any::TypeId};

use crate::{memory::{Heap, GCRef, StaticAtom}, symbol::{SymbolInterner, Symbol}, tvalue::{TType, self, TString, TValue, Typed}};

pub struct VM {
    heap: Box<Heap>,

    pub symbols: GCRef<SymbolInterner>,
    pub types: GCRef<RustTypeInterner>,
    pub hash_state: ahash::RandomState,
}

impl VM {
    pub fn init() -> (Rc<VM>, GCRef<TModule>) {
        let vm = Rc::new_cyclic(|me| {
            let heap = Box::new(Heap::init(me.clone()));
            Self::create(heap)
        });

        let prelude = TModule::new_from_rust(&vm);
        tvalue::prelude::module_init(prelude); 

        (vm, prelude)
    }

    fn create(heap: Box<Heap>) -> Self {
        let hash_state = ahash::RandomState::new();

        let symbols = StaticAtom::allocate(&heap, SymbolInterner::new()).make_static();
        let types = StaticAtom::allocate(&heap, RustTypeInterner::new()).make_static();

        VM {
            heap,
            symbols,

            hash_state,
            types
        }
    }

    pub fn heap(&self) -> &Heap {
        &self.heap
    }

    pub fn types(&self) -> GCRef<RustTypeInterner> {
        self.types
    }

    pub fn symbols(&self) -> GCRef<SymbolInterner> {
        self.symbols
    }
}

pub struct RustTypeInterner(hashbrown::HashMap<TypeId, GCRef<TType>>);

impl RustTypeInterner {
    fn new() -> Self {
        RustTypeInterner(Default::default())
    }
    
    fn intern<T: Typed>(&mut self, ttype: GCRef<TType>) {
        let Some(ttype2) = self.0.insert(TypeId::of::<T>(), ttype) else {
            return;
        };
        if ttype2.refrence_eq(ttype) {
            panic!("{:?} was tried to be associated with two different types", TypeId::of::<T>());
        }
    }

    pub fn query<T: Typed>(&self) -> GCRef<TType> {
        *self.0.get(&TypeId::of::<T>()).expect(
            &format!("expected query of interned rust type"))
    }
}

#[derive(Debug)]
pub enum GlobalErr {
    Redeclared(Symbol),
    NotFound(Symbol),
    Constant(Symbol)
}

pub struct TModule {
    name: TValue,
    source: TValue,
    table: hashbrown::HashTable<(Symbol, TValue, bool)>
}

impl TModule {
    pub fn new_from_rust(vm: &VM) -> GCRef<Self> {
        StaticAtom::allocate(vm.heap(), 
            Self {
                name: TValue::null(),
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
    pub fn set_name(&mut self, name: &str) {
        self.name = TString::from_slice(&self.vm(), name).into();
    }

    pub fn set_rust_ttype<T: Typed>(
        &mut self,
        value: GCRef<TType>,
    ) -> Result<(), GlobalErr> {
        let vm = self.vm();
        vm.types().intern::<T>(value);
        let name = vm.symbols().intern_slice(T::NAME);
        self.set_global(name, TValue::ttype_object(value), true)
    }

    pub fn set_global(&mut self, name: Symbol, value: TValue, constant: bool) -> Result<(), GlobalErr> {
        let symbols = self.vm().symbols;

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
            self.vm().symbols.hash(name),
            |entry| entry.0 == name
        ).map(|entry| entry.1).ok_or(GlobalErr::NotFound(name))
    }

    pub fn get_global_mut(&mut self, name: Symbol) -> Result<&mut TValue, GlobalErr> {
        let symbols = self.vm().symbols;
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
