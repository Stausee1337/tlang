use std::{rc::Rc, any::TypeId};

use crate::{memory::{Heap, GCRef, Atom, Visitor}, symbol::{SymbolCache, Symbol}, tvalue::{TType, self, TString, TValue, Typed}};

pub struct VM {
    heap: Box<Heap>,

    pub symbols: GCRef<SymbolCache>,
    pub types: GCRef<RustTypeInterner>,
    pub hash_state: ahash::RandomState,

    pub primitives: Primitives
}

impl VM {
    pub fn init() -> Rc<VM> {
        let vm = Rc::new_cyclic(|me| {
            let heap = Box::new(Heap::init(me.clone()));
            Self::create(heap)
        });
        vm
    }

    fn create(heap: Box<Heap>) -> Self {
        let hash_state = ahash::RandomState::new();

        let symbols = heap.allocate_atom(SymbolCache::new());
        let types = heap.allocate_atom(RustTypeInterner::new());

        VM {
            heap,

            symbols,
            hash_state,
            types,
            primitives: Primitives::lazy(),
        }
    }

    pub fn heap(&self) -> &Heap {
        &self.heap
    }

    pub fn primitives(&self) -> &Primitives {
        &self.primitives
    }

    pub fn types(&self) -> GCRef<RustTypeInterner> {
        self.types
    }

    pub fn symbols(&self) -> GCRef<SymbolCache> {
        self.symbols
    }
}

pub struct Primitives {
}

impl Primitives {
    fn lazy() -> Self {
        Primitives {}
    }

    pub fn float_type(&self) -> GCRef<TType> {
        todo!()
    }

    pub fn int_type(&self) -> GCRef<TType> {
        todo!()
    }

    pub fn bool_type(&self) -> GCRef<TType> {
        todo!()
    }

    pub fn string_type(&self) -> GCRef<TType> {
        todo!()
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
}

impl GCRef<RustTypeInterner> {
    pub fn query<T: Typed>(mut self) -> GCRef<TType> {
        if let Some(ty) = self.0.get(&TypeId::of::<T>()) {
            return *ty;
        }
        let ty = T::initialize(&self.vm());
        self.intern::<T>(ty);
        ty
    }
}

impl Atom for RustTypeInterner {
    fn visit(&self, visitor: &mut Visitor) {
        todo!()
    }
}

#[derive(Debug)]
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

impl Atom for TModule {
    fn visit(&self, visitor: &mut Visitor) {
        todo!()
    }
}

impl TModule {
    pub fn new_from_rust(vm: &VM, name: GCRef<TString>) -> GCRef<Self> {
        vm.heap().allocate_atom(
            Self {
                name,
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

    pub fn set_global(&mut self, name: Symbol, value: TValue, constant: bool) -> Result<(), GlobalErr> {
        let Err(..) = self.get_global(name) else {
            return Err(GlobalErr::Redeclared(name));
        };

        self.table.insert_unique(
            name.hash(),
            (name, value, constant),
            |value| value.0.hash()
        );
        Ok(())
    }

    pub fn get_global(&self, name: Symbol) -> Result<TValue, GlobalErr> {
        println!("{name:?} {}", self.vm().symbols().get(name));
        self.table.find(
            name.hash(),
            |entry| entry.0 == name
        ).map(|entry| entry.1).ok_or(GlobalErr::NotFound(name))
    }

    pub fn get_global_mut(&mut self, name: Symbol) -> Result<&mut TValue, GlobalErr> {
        let Some(entry) = self.table.find_mut(
            name.hash(),
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
