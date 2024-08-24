use std::{rc::Rc, any::TypeId, sync::OnceLock, mem::offset_of};

use hashbrown::hash_map::RawEntryMut;

use crate::{memory::{Heap, GCRef, Atom, Visitor}, symbol::{SymbolCache, Symbol}, tvalue::{TType, TString, TValue, Typed, TObject, TFunction, TInteger, TBool, TFloat, self, TProperty, Accessor, TList}};

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! debug {
    ($($arg:tt)*) => { println!($($arg)*) }
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! debug {
    ($($arg:tt)*) => { }
}

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

        let _ttype = vm.types().query::<TType>();
        vm.primitives().float_type();
        vm.primitives().int_type();
        vm.primitives().bool_type();
        vm.primitives().string_type();
        vm.primitives().list_type();

        init_prelude_functions(&vm);

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

fn init_prelude_functions(vm: &VM) {
    let mut prelude = vm.modules().prelude();

    let printfn = TFunction::rustfunc(prelude, Some("print"), move |msg| {
        tvalue::print(prelude, msg);
    });
    prelude.set_global(Symbol![print], printfn.into(), true);
}

pub struct Primitives {
    float: OnceLock<GCRef<TType>>,
    int: OnceLock<GCRef<TType>>,
    bool: OnceLock<GCRef<TType>>,
    string: OnceLock<GCRef<TType>>,
    list: OnceLock<GCRef<TType>>
}

impl Primitives {
    fn lazy() -> Self {
        Primitives {
            float: OnceLock::new(),
            int: OnceLock::new(),
            string: OnceLock::new(),
            bool: OnceLock::new(),
            list: OnceLock::new()
        }
    }
}

impl GCRef<Primitives> {
    pub fn float_type(self) -> GCRef<TType> {
        *self.float.get_or_init(|| {
            let vm = self.vm();
            let mut ttype = vm.heap().allocate_atom(TType {
                base: TObject::base(&vm, vm.types().query::<TType>()),
                basety: Some(vm.types().query::<TObject>()),
                basesize: 0, // primitive
                name: TString::from_slice(&vm, "float"),
                modname: TString::from_slice(&vm, "prelude"),
                variable: false
            });

            let mut prelude = vm.modules().prelude();
            prelude.set_global(Symbol![float], ttype.into(), true);

            ttype.define_method(Symbol![fmt], TFunction::rustfunc(
                    prelude, Some("float.fmt"),
                    move |this: TFloat| {
                        let vm = self.vm();
                        TString::from_format(&vm, format_args!("{this}"))
                    }));

            tvalue::float_init_arithmetics(ttype);
            tvalue::float_init_cmps(ttype);

            ttype
        })
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

            let mut prelude = vm.modules().prelude();
            prelude.set_global(Symbol![int], ttype.into(), true);

            ttype.define_method(Symbol![fmt], TFunction::rustfunc(
                    prelude, Some("int.fmt"),
                    move |this: TInteger| {
                        let vm = self.vm();
                        TString::from_format(&vm, format_args!("{this}"))
                    }));

            ttype.define_method(Symbol![pow], TFunction::rustfunc(
                    prelude, Some("int.pow"), TInteger::pow));

            tvalue::int_init_arithmetics(ttype);
            tvalue::int_init_cmps(ttype);

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

            let mut prelude = vm.modules().prelude();
            prelude.set_global(Symbol![bool], ttype.into(), true);

            ttype.define_method(Symbol![fmt], TFunction::rustfunc(
                    prelude, Some("bool.fmt"),
                    move |this: TBool| {
                        let vm = self.vm();
                        TString::from_format(&vm, format_args!("{this}"))
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

            let mut prelude = vm.modules().prelude();
            prelude.set_global(Symbol![string], ttype.into(), true);

            ttype.define_method(Symbol![fmt], TFunction::rustfunc(
                    prelude, Some("string.fmt"), |this: GCRef<TString>| this));
            ttype.define_method(Symbol![eq], TFunction::rustfunc(
                    prelude, Some("string.eq"), |this: GCRef<TString>, other: GCRef<TString>| this.eq(&other)));

            ttype.define_method(
                Symbol![get_iterator],
                TFunction::rustfunc(prelude, Some("string.get_iterator"), GCRef::<TString>::get_iterator));

            ttype.define_property(
                Symbol![length],
                TProperty::get(
                    prelude,
                    TFunction::rustfunc(prelude, None, |string: GCRef<TString>| unsafe {
                        let ptr = string.as_ptr() as *mut u8;
                        *(ptr.add(offset_of!(TString, length)) as *mut TInteger)
                    })));

            ttype
        })
    }

    pub fn list_type(self) -> GCRef<TType> {
        *self.list.get_or_init(|| {
            let vm = self.vm();
            let mut ttype = vm.heap().allocate_atom(TType {
                base: TObject::base(&vm, vm.types().query::<TType>()),
                basety: Some(vm.types().query::<TObject>()),
                basesize: 0, // primitive
                name: TString::from_slice(&vm, "list"),
                modname: TString::from_slice(&vm, "prelude"),
                variable: false
            });

            let mut prelude = vm.modules().prelude();
            prelude.set_global(Symbol![list], ttype.into(), true);

            ttype.define_method(
                Symbol![fmt],
                TFunction::rustfunc(prelude, Some("list.fmt"), |this: GCRef<TList>| {
                    TString::from_format(&this.vm(), format_args!("{this}"))
                }));

            ttype.define_method(
                Symbol![push],
                TFunction::rustfunc(prelude, Some("list.push"), GCRef::<TList>::push));

            ttype.define_method(
                Symbol![get_index],
                TFunction::rustfunc(prelude, Some("list.get_index"),
                |this: GCRef<TList>, index: TInteger| this[index]));

            ttype.define_method(
                Symbol![set_index],
                TFunction::rustfunc(prelude, Some("list.set_index"),
                |mut this: GCRef<TList>, index: TInteger, value: TValue| { this[index] = value; }));

            ttype.define_property(
                Symbol![length],
                TProperty::get(
                    prelude,
                    TFunction::rustfunc(prelude, None, |list: GCRef<TList>| unsafe {
                        let ptr = list.as_ptr() as *mut u8;
                        *(ptr.add(offset_of!(TList, length)) as *mut TInteger)
                    })));

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
    imported: hashbrown::HashMap<String, GCRef<TModule>>,
    prelude: OnceLock<GCRef<TModule>>,
}

impl TModules {
    pub fn new() -> Self {
        Self {
            imported: Default::default(),
            prelude: OnceLock::new()
        }
    }
}

impl GCRef<TModules> {
    pub fn prelude(&self) -> GCRef<TModule> {
        *self.prelude.get_or_init(|| {
            let vm = self.vm();
            let module = TModule::new(&vm, TString::from_slice(&vm, "prelude"));
            vm.modules().insert("tlang:prelude", module);
            module
        })
    }

    pub fn get(&mut self, key: &str) -> Option<GCRef<TModule>> {
        self.imported.get(key).map(|module| *module)
    }

    pub fn insert(&mut self, key: &str, module: GCRef<TModule>) {
        match self.imported
            .raw_entry_mut()
            .from_key(key) {
            RawEntryMut::Occupied(..) => {
                panic!("module is already imported");
            }
            RawEntryMut::Vacant(entry) => {
                entry.insert(key.to_string(), module);
            }
        }
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

    #[inline(never)]
    pub fn query<T: Typed>(&mut self) -> GCRef<TType> {
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

    pub fn iter(&mut self) -> impl Iterator<Item = (Symbol, TValue)> + '_ {
        self.table.iter().map(|entry| (entry.0, entry.1))
    }

    pub fn import(&mut self, path: &str, what: Option<&mut dyn Iterator<Item = Symbol>>) {
        // FIXME: call into import logic here to resolve
        // file bound modules as well
        let Some(mut module) = self.vm().modules().get(path) else {
            panic!("could not find module {path}");
        };
        if let Some(filter) = what {
            for name in filter.into_iter() {
                let value = module.get_global(name).unwrap();
                self.set_global(name, value, true);
            }
        } else {
            for (name, value) in module.iter() {
                self.set_global(name, value, true);
            }
        }
    }
}
