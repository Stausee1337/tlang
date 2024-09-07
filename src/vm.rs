use std::{any::TypeId, sync::OnceLock, mem::ManuallyDrop, ptr::NonNull, ops::Deref};

use hashbrown::hash_map::RawEntryMut;

use crate::{memory::{Heap, GCRef, Atom, Visitor}, symbol::{SymbolCache, Symbol}, tvalue::{TType, TString, TValue, Typed, TObject, TFunction, TInteger, TBool, TFloat, self, TProperty, TList, TypeFlags}};

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

pub struct Eternal<T> {
    ptr: NonNull<T>
}

impl<T> Clone for Eternal<T> {
    #[inline]
    fn clone(&self) -> Self {
        Eternal { ptr: self.ptr }
    }
}

impl<T> Deref for Eternal<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: 'static> Eternal<T> {
    unsafe fn uninit() -> Self {
        let ptr = std::alloc::alloc(std::alloc::Layout::new::<T>());
        Eternal { ptr: NonNull::new_unchecked(ptr as *mut T) }
    }

    pub fn make<F>(init: F) -> Self
    where
        F: FnOnce(Eternal<T>) -> T
    {
        unsafe {
            let blueprint = Self::uninit();
            let instance = ManuallyDrop::new(init(blueprint.clone()));
            std::ptr::copy_nonoverlapping(instance.deref(), blueprint.ptr.as_ptr(), 1);
            blueprint
        }
    }
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
    // FIXME: the vm shouldn't actually be created and deleted by
    // init() and shutdown(vm) functions, but rather using a callback.
    // e.g. vm::with_vm(...), which gets rid of the vm safely after use.
    // It should only take a function pointer (NOT a closure), to make (accedentally) safely
    // leaking it outside of its scope hard to do
    pub fn init() -> Eternal<VM> {
        let vm = Eternal::make(|place| {
            let heap = Box::new(Heap::init(place));
            Self::create(heap)
        });

        vm.heap().defer_gc();

        let _ttype = vm.types().query::<TType>();
        vm.primitives().float_type();
        vm.primitives().int_type();
        vm.primitives().bool_type();
        vm.primitives().string_type();
        vm.primitives().list_type();

        init_prelude_functions(&vm);

        vm.heap().undefer_gc();

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

pub fn shutdown(vm: Eternal<VM>) {
    unsafe {
        std::ptr::drop_in_place(vm.ptr.as_ptr());
    }
}

fn init_prelude_functions(vm: &VM) {
    let mut prelude = vm.modules().prelude();

    let printfn = TFunction::rustfunc(prelude, Some("print"), move |msg| {
        tvalue::print(prelude, msg);
    });
    prelude.set_global(Symbol![print], printfn.into(), true).unwrap();
}

type Initializer = fn(vm: &VM) -> TValue;

pub struct Primitives {
    float: OnceLock<GCRef<TType>>,
    int: OnceLock<GCRef<TType>>,
    bool: OnceLock<GCRef<TType>>,
    string: OnceLock<GCRef<TType>>,
    list: OnceLock<GCRef<TType>>,

    zeroed_initializers: [Initializer; 5]
}

impl Primitives {
    fn lazy() -> Self {
        let zeroed_initializers: [Initializer; 5] = [
            |_vm| TFloat::from_float(0.0).into(),
            |_vm| TInteger::from_int32(0).into(),
            |_vm| TBool::from_bool(false).into(),
            |vm| TString::from_slice(vm, "").into(),
            |vm| TList::new_empty(vm).into(),
        ];
        Primitives {
            float: OnceLock::new(),
            int: OnceLock::new(),
            string: OnceLock::new(),
            bool: OnceLock::new(),
            list: OnceLock::new(),

            zeroed_initializers
        }
    }
}

impl Primitives {
    pub fn float_type(self: GCRef<Self>) -> GCRef<TType> {
        *self.float.get_or_init(|| TFloat::initialize_type(&self.vm()))
    }

    pub fn int_type(self: GCRef<Self>) -> GCRef<TType> {
        *self.int.get_or_init(|| TInteger::initialize_type(&self.vm()))
    }

    pub fn bool_type(self: GCRef<Self>) -> GCRef<TType> {
        *self.bool.get_or_init(|| TBool::initialize_type(&self.vm()))
    }

    pub fn string_type(self: GCRef<Self>) -> GCRef<TType> {
        *self.string.get_or_init(|| TString::initialize_type(&self.vm()))
    }

    pub fn list_type(self: GCRef<Self>) -> GCRef<TType> {
        *self.list.get_or_init(|| TList::initialize_type(&self.vm()))
    }

    pub fn zeroed(self: GCRef<Self>, ptype: GCRef<TType>) -> TValue {
        let list = [
            self.float_type(),
            self.int_type(),
            self.bool_type(),
            self.string_type(),
            self.list_type(),
        ];

        for (idx, ty) in list.into_iter().enumerate() {
            if GCRef::refrence_eq(ty, ptype) {
                return self.zeroed_initializers[idx](&self.vm());
            }
        }

        eprintln!("{} is not a primitive, yet it as a basesize < sizeof(TObject)", ptype.name);
        std::process::abort();
    }
}

impl Atom for Primitives {
    fn visit(&self, visitor: &mut Visitor) {
        self.float.get().map(|ty| visitor.feed(*ty));
        self.int.get().map(|ty| visitor.feed(*ty));
        self.bool.get().map(|ty| visitor.feed(*ty));
        self.string.get().map(|ty| visitor.feed(*ty));
        self.list.get().map(|ty| visitor.feed(*ty));
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

impl TModules {
    pub fn prelude(self: GCRef<Self>) -> GCRef<TModule> {
        *self.prelude.get_or_init(|| {
            let vm = self.vm();
            let module = TModule::new(&vm, TString::from_slice(&vm, "prelude"));
            vm.modules().insert("tlang:prelude", module);
            module
        })
    }

    pub fn get(mut self: GCRef<Self>, key: &str) -> Option<GCRef<TModule>> {
        self.imported.get(key).map(|module| *module)
    }

    pub fn insert(mut self: GCRef<Self>, key: &str, module: GCRef<TModule>) {
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
        self.prelude.get().map(|ty| visitor.feed(*ty));
        for (_, import) in self.imported.iter() {
            visitor.feed(*import);
        }
    }
}

pub struct RustTypeInterner(hashbrown::HashMap<TypeId, GCRef<TType>>);

impl RustTypeInterner {
    fn new() -> Self {
        RustTypeInterner(Default::default())
    }    

    #[inline(never)]
    pub fn query<T: Typed>(mut self: GCRef<Self>) -> GCRef<TType> {
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
        for (_, ty) in self.0.iter() {
            visitor.feed(*ty);
        }
    }
}

#[derive(Debug)]
pub enum GlobalErr {
    Redeclared(Symbol),
    NotFound(Symbol),
    Constant(Symbol)
}

pub struct TModule {
    pub(crate) name: GCRef<TString>,
    source: Option<GCRef<TString>>,
    table: hashbrown::HashTable<(Symbol, TValue, bool)>
}

impl Atom for TModule {
    fn visit(&self, visitor: &mut Visitor) {
        visitor.feed(self.name);
        if let Some(source) = self.source {
            visitor.feed(source);
        }
        for (_, val, _) in self.table.iter() {
            val.visit(visitor);
        }
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

    pub fn import(mut self: GCRef<Self>, path: &str, what: Option<&mut dyn Iterator<Item = Symbol>>) {
        // FIXME: call into import logic here to resolve
        // file bound modules as well
        let Some(mut module) = self.vm().modules().get(path) else {
            panic!("could not find module {path}");
        };
        if let Some(filter) = what {
            for name in filter.into_iter() {
                let value = module.get_global(name).unwrap();
                self.set_global(name, value, true).unwrap();
            }
        } else {
            for (name, value) in module.iter() {
                self.set_global(name, value, true).unwrap();
            }
        }
    }
}
