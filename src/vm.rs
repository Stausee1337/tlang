use std::{rc::Rc, any::TypeId, sync::OnceLock};

use hashbrown::hash_map::RawEntryMut;

use crate::{memory::{Heap, GCRef, Atom, Visitor}, symbol::{SymbolCache, Symbol}, tvalue::{TType, self, TString, TValue, Typed, TProperty, Accessor, TObject, TFunction, TInteger, TBool}};

pub struct VM {
    heap: Box<Heap>,

    pub symbols: GCRef<SymbolCache>,
    pub types: GCRef<RustTypeInterner>,
    pub modules: GCRef<TModules>,
    pub hash_state: ahash::RandomState,

    pub primitives: GCRef<Primitives>
}

impl VM {
    pub fn init() -> Rc<VM> {
        let vm = Rc::new_cyclic(|me| {
            let heap = Box::new(Heap::init(me.clone()));
            Self::create(heap)
        });
        let ttype = vm.types().query::<TType>();
        vm
    }

    fn create(heap: Box<Heap>) -> Self {
        let hash_state = ahash::RandomState::new();

        let symbols = heap.allocate_atom(SymbolCache::new());
        let types = heap.allocate_atom(RustTypeInterner::new());
        let modules = heap.allocate_atom(TModules::new());
        let primitives = heap.allocate_atom(Primitives::lazy());

        VM {
            heap,
            modules,

            symbols,
            hash_state,
            types,
            primitives,
        }
    }

    pub fn heap(&self) -> &Heap {
        &self.heap
    }

    pub fn primitives(&self) -> GCRef<Primitives> {
        self.primitives
    }

    pub fn types(&self) -> GCRef<RustTypeInterner> {
        self.types
    }

    pub fn symbols(&self) -> GCRef<SymbolCache> {
        self.symbols
    }

    pub fn modules(&self) -> GCRef<TModules> {
        self.modules
    }
}

pub struct Primitives {
    int: OnceLock<GCRef<TType>>,
    bool: OnceLock<GCRef<TType>>,
    string: OnceLock<GCRef<TType>>
}

impl Primitives {
    fn lazy() -> Self {
        Primitives {
            int: OnceLock::new(),
            string: OnceLock::new(),
            bool: OnceLock::new()
        }
    }
}

impl GCRef<Primitives> {
    pub fn float_type(&self) -> GCRef<TType> {
        todo!()
    }

    pub fn int_type(self) -> GCRef<TType> {
        *self.int.get_or_init(|| {
            let vm = self.vm();
            let mut ttype = vm.heap().allocate_atom(TType {
                base: TObject::base(&vm, vm.types().query::<TType>()),
                basety: Some(vm.types().query::<TObject>()),
                basesize: 0, // primitive
                name: TString::from_slice(&vm, "int"),
                modname: TString::from_slice(&vm, "prelude"),
                variable: false
            });

            ttype.define_method(Symbol![toString], TFunction::rustfunc(
                    vm.modules().empty(), Some("int::toString"),
                    move |this: TInteger| {
                        let vm = self.vm();
                        TString::from_slice(&vm, &format!("{this}"))
                    }));

            ttype
        })
    }

    pub fn bool_type(self) -> GCRef<TType> {
        *self.bool.get_or_init(|| {
            let vm = self.vm();
            let mut ttype = vm.heap().allocate_atom(TType {
                base: TObject::base(&vm, vm.types().query::<TType>()),
                basety: Some(vm.types().query::<TObject>()),
                basesize: 0, // primitive
                name: TString::from_slice(&vm, "bool"),
                modname: TString::from_slice(&vm, "prelude"),
                variable: false
            });

            ttype.define_method(Symbol![toString], TFunction::rustfunc(
                    vm.modules().empty(), Some("bool::toString"),
                    move |this: TBool| {
                        let vm = self.vm();
                        TString::from_slice(&vm, &format!("{this}"))
                    }));

            ttype
        })
    }

    pub fn string_type(&self) -> GCRef<TType> {
        *self.string.get_or_init(|| {
            let vm = self.vm();
            let mut ttype = vm.heap().allocate_atom(TType {
                base: TObject::base(&vm, vm.types().query::<TType>()),
                basety: Some(vm.types().query::<TObject>()),
                basesize: 0, // primitive
                name: TString::from_slice(&vm, "string"),
                modname: TString::from_slice(&vm, "prelude"),
                variable: false
            });

            ttype.define_method(Symbol![toString], TFunction::rustfunc(
                    vm.modules().empty(), Some("string::toString"), |this: GCRef<TString>| this));

            ttype
        })
    }
}

impl Atom for Primitives {
    fn visit(&self, visitor: &mut Visitor) {
        todo!()
    }
}

pub struct TModules {
    empty: OnceLock<GCRef<TModule>>
}

impl TModules {
    pub fn new() -> Self {
        Self {
            empty: OnceLock::new()
        }
    }
}

impl GCRef<TModules> {
    pub fn empty(&self) -> GCRef<TModule> {
        *self.empty.get_or_init(|| {
            let vm = self.vm();
            TModule::new(&vm, TString::from_slice(&vm, "empty"))
        })
    }
}

impl Atom for TModules {
    fn visit(&self, visitor: &mut Visitor) {
        todo!()
    }
}

pub struct RustTypeInterner(hashbrown::HashMap<TypeId, GCRef<TType>>);

impl RustTypeInterner {
    fn new() -> Self {
        RustTypeInterner(Default::default())
    }    
}

impl GCRef<RustTypeInterner> {
    pub fn query<T: Typed>(mut self) -> GCRef<TType> {
        let vm = self.vm();

        let key = TypeId::of::<T>();
        let entry = self.0.raw_entry_mut().from_key(&key);
        match entry {
            RawEntryMut::Occupied(ty) => *ty.get(),
            RawEntryMut::Vacant(vacant) => {
                T::initialize_entry(&vm, vacant)
            }
        }
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
    source: Option<GCRef<TString>>,
    table: hashbrown::HashTable<(Symbol, TValue, bool)>
}

impl Atom for TModule {
    fn visit(&self, visitor: &mut Visitor) {
        todo!()
    }
}

impl TModule {
    pub fn new(vm: &VM, name: GCRef<TString>) -> GCRef<Self> {
        vm.heap().allocate_atom(
            Self {
                name,
                source: None,
                table: Default::default()
            }
        )
    }

    pub fn set_source(&mut self, source: Option<GCRef<TString>>) {
        self.source = source;
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
