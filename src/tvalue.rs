use std::{mem::{transmute, offset_of}, fmt::{Display, Write}, alloc::Layout, any::TypeId, hash::BuildHasherDefault};


use ahash::AHasher;
use allocator_api2::alloc::Global;
use bitflags::{bitflags, Flags};
use hashbrown::{hash_map::RawVacantEntryMut, HashTable, hash_table::Entry};
use tlang_macros::tcall;

use crate::{memory::{GCRef, Atom, Visitor}, symbol::Symbol, bytecode::TRawCode, bigint::{TBigint, self, to_bigint}, vm::{VM, TModule}, eval::{TArgsBuffer, StackFrame}, interop::{TPolymorphicObject, VMDowncast, TPolymorphicWrapper, VMArgs, VMCast, TPolymorphicCallable, TVariadicArguments}, debug};


macro_rules! __tobject_struct {
    ($vis:vis, $name:ident, [], { $($fields:tt)* }) => {
        #[repr(C)]
        $vis struct $name {
            #[doc(hidden)]
            pub base: TObject,
            $($fields)*
        }
    };
    ($vis:vis, $name:ident, [$base:ident], { $($fields:tt)* }) => {
        #[repr(C)]
        $vis struct $name {
            #[doc(hidden)]
            pub base: $base,
            $($fields)*
        }
    };
    ($vis:vis, $name:ident, [()], { $($fields:tt)* }) => {
        #[repr(C)]
        $vis struct $name {
            $($fields)*
        }
    };
}

macro_rules! __tobject_marker {
    ($name:ident,) => {
        unsafe impl $crate::interop::TPolymorphicObject for $name {
            type Base = TObject;
        }
    };
    ($name:ident, ()) => {
        unsafe impl $crate::interop::TPolymorphicObject for $name {
            type Base = TObject;
        }
    };
    ($name:ident, $base:ident) => {
        unsafe impl $crate::interop::TPolymorphicObject for $name {
            type Base = $base;
        }
    };
}

macro_rules! tobject {
    (
        $(#[$outer:meta])*
        $vis:vis struct $name:ident $(: $base:tt)?
        {
            $(
                $(#[$inner:ident $($args:tt)*])*
                $fvis:vis $fname:ident: $fty:ty
            ),*
        }
    ) => {
        __tobject_struct!(
            $vis, $name,
            [$($base)?],
            {
                $(
                    $(#[$inner $($args )*])*
                    $fvis $fname : $fty
                ),*
            }
        );

        __tobject_marker!($name, $($base)?);

        impl Into<TValue> for GCRef<$name> {
            #[inline(always)]
            fn into(self) -> TValue {
                TValue::object(self)
            }
        }

        impl $crate::interop::VMDowncast for GCRef<$name> {
            #[inline(always)]
            fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
                value.query_object::<$name>(vm)
            }
        }
    };
    () => {};
}


#[repr(u64)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TValueKind {
    Object   = 0b101 << 49,
    Int32    = 0b001 << 49,
    Bool     = 0b010 << 49,
    Float    = 0b100 << 49,
    String   = 0b000 << 49,
    BigInt   = 0b110 << 49,
    List     = 0b011 << 49,
}

/// 64bit Float:
/// S eeeeeeeeeee FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
/// (S) 1bit sign
/// (e) 11bit exponent
/// (F) 52 fraction
///
/// NaN's for boxing:
/// X 11111111111 TTTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
#[derive(Clone, Copy)]
pub struct TValue(u64);

impl TValue {
    const FLOAT_TAG_MASK: u64 = 0x000e000000000000;
    const FLOAT_NAN_TAG:  u64 = 0x7ff0000000000000;
    const NAN_VALUE_MASK: u64 = 0x0001ffffffffffff;

    pub const fn encoded(&self) -> u64 {
        return self.0;
    }

    /// Constructors
    
    #[inline(always)]
    pub const fn null() -> Self {
        let null = unsafe { GCRef::<()>::null() };
        Self::object_impl(null, TValueKind::Object)
    }

    #[inline(always)]
    pub const fn object<T: TPolymorphicObject>(object: GCRef<T>) -> Self {
        Self::object_impl(object, TValueKind::Object)
    }

    #[inline(always)]
    const fn string(string: GCRef<TString>) -> Self {
        Self::object_impl(string, TValueKind::String)
    }

    #[inline(always)]
    const fn bigint(bigint: GCRef<bigint::TBigint>) -> Self {
        Self::object_impl(bigint, TValueKind::BigInt)
    }

    #[inline(always)]
    const fn bool(bool: bool) -> Self {
        let bool = bool as u64;
        TValue(Self::FLOAT_NAN_TAG | TValueKind::Bool as u64 | bool)
    }

    #[inline(always)]
    fn int32(int: i32) -> Self {
        let int = int as u32 as u64;
        TValue(Self::FLOAT_NAN_TAG | TValueKind::Int32 as u64 | int)
    }

    #[inline(always)]
    const fn float(float: f64) -> Self {
        return TValue(unsafe { transmute(float) });
    }

    #[inline(always)]
    const fn object_impl<T>(object: GCRef<T>, kind: TValueKind) -> Self {
        // TODO: use object.as_ptr() instead
        let object: u64 = unsafe { transmute(object) };
        assert!(object & (!Self::NAN_VALUE_MASK) == 0);
        TValue(Self::FLOAT_NAN_TAG | kind as u64 | object)
    }

    /// Public Helpers 

    #[inline(always)]
    pub fn query_object<T: Typed>(&self, vm: &VM) -> Option<GCRef<T>> {
        if let TValueKind::Object = self.kind() {
            let ty = vm.types().query::<T>();
            if !GCRef::refrence_eq(self.ttype(vm)?, ty) {
                return None;
            }
            return self.as_object();
        }
        None
    }

    #[inline(always)]
    pub fn query_tobject(&self) -> Option<GCRef<TObject>> {
        if let TValueKind::Object = self.kind() {
            return self.as_object();
        }
        None
    }

    #[inline(always)]
    pub fn query_string(&self) -> Option<GCRef<TString>> {
        if let TValueKind::String = self.kind() {
            return Some(self.as_gcref());
        }
        None
    }

    #[inline(always)]
    pub fn query_list(&self) -> Option<GCRef<TList>> {
        if let TValueKind::List = self.kind() {
            return Some(self.as_gcref());
        }
        None
    }

    #[inline(always)]
    pub fn query_integer(&self) -> Option<TInteger> {
        match self.kind() {
            TValueKind::Int32 =>
                Some(TInteger(IntegerKind::Int32(self.as_int32()))),
            TValueKind::BigInt =>
                Some(TInteger(IntegerKind::Bigint(self.as_gcref()))),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn query_float(&self) -> Option<TFloat> {
        match self.kind() {
            TValueKind::Float =>
                Some(TFloat(self.as_float())),
            _ => None
        }
    }

    #[inline(always)]
    pub fn query_bool(&self) -> Option<TBool> {
        match self.kind() {
            TValueKind::Bool =>
                Some(TBool(self.as_bool())),
            _ => None
        }
    }

    #[inline(always)]
    pub fn ttype(&self, vm: &VM) -> Option<GCRef<TType>> {
        Some(match self.kind() {
            TValueKind::Bool => vm.primitives().bool_type(),
            TValueKind::Int32 | TValueKind::BigInt => vm.primitives().int_type(),
            TValueKind::Float => vm.primitives().float_type(),
            TValueKind::String => vm.primitives().string_type(),
            TValueKind::List => vm.primitives().list_type(),
            TValueKind::Object => {
                let Some(object) = self.as_object::<TObject>() else {
                    return None; // TValue::null() doesn't have a type
                };
                // `object` should be (at least) a subtype of `TObject`, because the
                // object constructor `TValue::object` requires object to be `TPolymorphicObject`,
                // which when implemented safely, keeps up polymorphism.
                object.ty
            }
        })
    }

    #[inline(always)]
    pub const fn is_null(&self) -> bool {
        self.encoded() == TValue::null().encoded()
    }

    /// Private Helpers

    #[inline(always)]
    pub(crate) fn kind(&self) -> TValueKind {
        let float: f64 = unsafe { transmute(self.0) };
        if !float.is_nan() {
            return TValueKind::Float;
        }
        unsafe { transmute(self.0 & Self::FLOAT_TAG_MASK) }
    }

    #[inline(always)]
    const fn as_int32(&self) -> i32 {
        (self.0 & Self::NAN_VALUE_MASK) as u32 as i32
    }

    #[inline(always)]
    const fn as_bool(&self) -> bool {
        (self.0 & Self::NAN_VALUE_MASK) != 0
    }

    #[inline(always)]
    const fn as_float(&self) -> f64 {
        unsafe { transmute(self.0) }
    }

    #[inline(always)]
    const fn as_gcref<T>(&self) -> GCRef<T> {
        unsafe { GCRef::from_raw((self.0 & Self::NAN_VALUE_MASK) as *mut T) }
    }

    #[inline(always)]
    const fn as_object<T>(&self) -> Option<GCRef<T>> {
        unsafe { self.as_object_unsafe() }
    }

    #[inline(always)]
    const unsafe fn as_object_unsafe<T>(&self) -> Option<GCRef<T>> {
        // This will still yield None, if the GCRef<..>,
        // which contains a NonNull, actually is null
        // So `TValue::null()` -> None
        transmute(self.as_gcref::<T>())
    }

    pub fn visit(self, visitor: &mut Visitor) {
        match self.kind() {
            TValueKind::Object | TValueKind::List |
            TValueKind::BigInt | TValueKind::String => unsafe {
                if let Some(object) = self.as_object_unsafe::<()>() {
                    visitor.feed(object);
                }
            }
            _ => ()
        }
    }
}

pub fn visit_polymorphic<T>(poly: &T, visitor: &mut Visitor)
where
    T: Typed + TPolymorphicObject
{
    unsafe {
        let ttype_ptr: *const *const TType =
            poly as *const _  as *const *const TType;
        let ttype = GCRef::from_raw(*ttype_ptr);
        let this = GCRef::from_raw(poly as *const T as *const TObject);

        if let Some(descriptor) = this.descriptor {
            visitor.feed(descriptor);
        }

        visitor.feed(ttype);

        ttype.properties(|prop| {
            let Some(get) = prop.get else {
                return;
            };
            let getter: TPolymorphicCallable<_, TValue> = get.into();
            let ret = getter(TValue::object(this));
            ret.visit(visitor);
        });
    }
}

pub trait Typed: Atom + Sized + 'static {
    fn initialize_entry(
        vm: &VM,
        entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
    ) -> GCRef<TType>;

    fn visit_override(&self, visitor: &mut Visitor)
    where
        Self: TPolymorphicObject 
    {
        visit_polymorphic(self, visitor);
    }
}

impl<T: Typed + TPolymorphicObject> Atom for T {
    fn visit(&self, visitor: &mut Visitor) {
        T::visit_override(self, visitor);
    }
}

tobject! {
pub struct TObject: () {
    pub ty: GCRef<TType>,
    descriptor: Option<GCRef<HashTable<(Symbol, TValue)>>>
}
}

impl TObject {
    unsafe fn ctor(this: *mut TObject, ty: GCRef<TType>) { 
    }

    pub fn new_with_capacity(vm: &VM, capacity: usize) -> GCRef<Self> {
        let descriptor = vm.heap().allocate_atom(HashTable::with_capacity(capacity));
        vm.heap().allocate_atom(Self {
            ty: vm.types().query::<Self>(),
            descriptor: Some(descriptor),
        })
    }

    pub fn base(vm: &VM, ty: GCRef<TType>) -> Self {
        let descriptor = if ty.flags.contains(TypeFlags::VARIABLE) {
            Some(vm.heap().allocate_atom(HashTable::new()))
        } else {
            None
        };
        Self {
            ty, descriptor,
        }
    }

    fn base_with_descriptor(
        vm: &VM, ty: GCRef<TType>, descriptor: Option<GCRef<HashTable<(Symbol, TValue)>>>) -> Self {
        Self {
            ty, descriptor,
        }
    }

    pub fn get_attribute(&mut self, name: Symbol, allows_insert: bool) -> Option<&mut TValue> {
        let Some(descriptor) = &mut self.descriptor else {
            debug_assert!(!self.ty.flags.contains(TypeFlags::VARIABLE));
            panic!("cannot get attribute on fixed type");
        };
        match descriptor.entry(
            name.hash,
            |val| val.0 == name,
            |val| val.0.hash) {
            Entry::Occupied(mut entry) => {
                Some(&mut entry.into_mut().1)
            }
            Entry::Vacant(entry) if allows_insert => {
                let mut entry = entry.insert((name, TValue::null()));
                Some(&mut entry.into_mut().1)
            }
            _ => None
        }
    }

    pub fn set_attribute(&mut self, name: Symbol, value: TValue) {
        let Some(descriptor) = &mut self.descriptor else {
            debug_assert!(!self.ty.flags.contains(TypeFlags::VARIABLE));
            panic!("cannot set attribute on fixed type {:p} {:p}", std::ptr::addr_of!(self.ty), GCRef::as_ptr(self.ty));
        };
        match descriptor.entry(
            name.hash,
            |val| val.0 == name,
            |val| val.0.hash) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = (name, value);
            }
            Entry::Vacant(entry) => {
                entry.insert((name, value));
            }
        }
    }

    pub fn attributes<F: FnMut(TValue)>(&self, mut f: F) {
        if let Some(descriptor) = self.descriptor {
            descriptor.iter().for_each(|val| f(val.1));
        }
    }

    pub fn instanciate(ty: GCRef<TType>) -> GCRef<TObject> {
        let vm = ty.vm();
        if ty.basesize < std::mem::size_of::<TObject>() {
            panic!("cannot instanciate primitive type {}", ty.name);
        }
        if ty.flags.contains(TypeFlags::ABSTRACT) { 
            panic!("cannot instanciate abstract type {}", ty.name);
        }
        let Some(ctor) = ty.ctor else {
            eprintln!("non-abstract constructorless type {}", ty.name);
            std::process::abort();
        };

        let extra_bytes = ty.basesize - std::mem::size_of::<TObject>();
        let vm = ty.vm();
        let descriptor = if ty.flags.contains(TypeFlags::VARIABLE) {
            Some(vm.heap().allocate_atom(HashTable::new()))
        } else {
            None
        };
        let result = vm.heap().allocate_var_atom(TObject {
            ty,
            descriptor
        }, extra_bytes);

        unsafe { ctor(GCRef::as_ptr(result), ty) }
        result
    }
}

impl Typed for TObject {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: Self::base_with_descriptor(vm, unsafe { GCRef::null() }, None),
            basesize: std::mem::size_of::<Self>(),
            basety: None, // There's nothing above Object in the hierarchy
            name: TString::from_slice(vm, "Object"),
            modname: TString::from_slice(vm, "prelude"),
            flags: TypeFlags::OPEN | TypeFlags::VARIABLE,
            ctor: Some(TObject::ctor)
        });
        entry.insert(TypeId::of::<Self>(), ttype);

        ttype.base = TObject::base(vm, vm.types().query::<TType>());

        let mut builder = TTypeBuilder::new(vm);
        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![Object], ttype.into(), true).unwrap();

        builder.define_method(Symbol![eq], TFunction::rustfunc(
                prelude, Some("Object.eq"), |this: TValue, other: TValue| {
                    TBool::from_bool(this.encoded() == other.encoded())
                }));

        builder.define_method(Symbol![ne], TFunction::rustfunc(
                prelude, Some("Object.ne"), move |this: TValue, other: TValue| {
                    let vm = ttype.vm();
                    let eq: TBool = tcall!(&vm, TValue::eq(this, other));
                    !eq
                }));

        builder.define_method(Symbol![fmt], TFunction::rustfunc(
                prelude, Some("Object.fmt"), move |this: TValue| {
                    let vm = ttype.vm();
                    let ty = this.ttype(&vm).unwrap();

                    let object = this.as_gcref::<TObject>();
                    if GCRef::refrence_eq(ty, ttype) && object.descriptor.is_some() {
                        let formatter = TObjectFormatter { object };
                        return TString::from_format(&vm, format_args!("{formatter}"))
                    }

                    TString::from_format(&vm, format_args!("{} {{}}", ty.name))
                }));

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
    }
}

impl Atom for HashTable<(Symbol, TValue)> {
    fn visit(&self, visitor: &mut Visitor) {
        let this = unsafe { GCRef::from_raw(self as *const Self) };
        let vm = this.vm();
        for (_, val) in self.iter() {
            // debug!("Visit {}", val.ttype(&vm).map(|ty| ty.name.as_slice()).unwrap_or("null"));
            val.visit(visitor);
        }
    }
}

struct TObjectFormatter {
    object: GCRef<TObject>,
}

impl std::fmt::Display for TObjectFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('{')?;
        let vm = self.object.vm();

        if let Some(descriptor) = self.object.descriptor {
            if !descriptor.is_empty() {
                f.write_char(' ')?;
                for (idx, (key, value)) in descriptor.iter().enumerate() {
                    if idx > 0 {
                        f.write_str(", ")?;
                    }
                    f.write_str(vm.symbols().get(*key).as_slice())?;
                    f.write_str(": ")?;
                    let string: GCRef<TString> = tcall!(&vm, TValue::fmt(*value));
                    f.write_str(string.as_slice())?;
                }
                f.write_char(' ')?;
            }
        }

        f.write_char('}')
    }
}

#[derive(Clone, Copy)]
pub struct TBool(bool);

impl TBool {
    #[inline(always)]
    pub const fn as_bool(self) -> bool {
        self.0
    }

    #[inline(always)]
    pub const fn from_bool(bool: bool) -> Self {
        TBool(bool)
    }
}

impl Into<TValue> for TBool {
    #[inline(always)]
    fn into(self) -> TValue {
        TValue::bool(self.0)
    }
}

impl From<bool> for TBool {
    #[inline(always)]
    fn from(value: bool) -> Self {
        TBool(value)
    }
}

impl Display for TBool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.as_bool() {
            "true"
        } else {
            "false"
        })
    }
}

impl std::fmt::Debug for TBool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl VMDowncast for TBool {
    fn vmdowncast(value: TValue, _vm: &VM) -> Option<Self> {
        value.query_bool()
    }
}

impl std::ops::Not for TBool {
    type Output = Self;

    fn not(self) -> Self::Output {
        TBool(!self.0)
    }
}

pub fn bool_init_unarys(builder: &mut TTypeBuilder) {
    let vm = builder.vm();
    let prelude = vm.modules().prelude();

    builder.define_method(Symbol![not], TFunction::rustfunc(
            prelude, Some("bool.not"),
            <TBool as std::ops::Not>::not));
}

impl TBool {
    pub fn initialize_type(vm: &VM) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base(&vm, vm.types().query::<TType>()),
            basety: Some(vm.types().query::<TObject>()),
            basesize: 0, // primitive
            name: TString::from_slice(&vm, "bool"),
            modname: TString::from_slice(&vm, "prelude"),
            flags: TypeFlags::empty(),
            ctor: None
        });

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![bool], ttype.into(), true).unwrap();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_method(Symbol![fmt], TFunction::rustfunc(
                prelude, Some("bool.fmt"),
                move |this: TBool| {
                    let vm = prelude.vm();
                    TString::from_format(&vm, format_args!("{this}"))
                }));
        
        bool_init_unarys(&mut builder);

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
    }
}

#[repr(C)]
#[repr(align(8))]
pub struct TString {
    data: StringData,
    pub(crate) length: TInteger,

    prev: Option<GCRef<TString>>,
    next: Option<GCRef<TString>>,

    size: usize,
}

union StringData {
    small: [u8; std::mem::size_of::<&()>()],
    ptr: *const u8
}

unsafe impl std::marker::Sync for StringData {}
unsafe impl std::marker::Send for StringData {}

static_assertions::const_assert!(std::mem::size_of::<StringData>() == 8);

impl TString {
    pub fn from_slice(vm: &VM, slice: &str) -> GCRef<Self> {
        let size = slice.len();
        let length = TInteger::from_usize(slice.chars().count());

        unsafe {
            let mut string = vm.heap().allocate_atom(
                Self {
                    data: StringData { ptr: std::ptr::null() },
                    length,
                    prev: None, next: None,
                    size
                },
            );

            if size < std::mem::size_of::<&()>() {
                string.data.small[0] = 1;
            } else {
                let align = 2;
                string.data.ptr = std::alloc::alloc(Layout::from_size_align_unchecked(size, align)) as *const u8;
            }

            std::ptr::copy_nonoverlapping(slice.as_ptr(), string.data_ptr_mut(), size);

            string
        }
    }


    pub fn from_format(vm: &VM, arguments: std::fmt::Arguments) -> GCRef<Self> {
        const STACK_CAP: usize = std::mem::size_of::<&()>() - 1;
        let mut stack_storage = [0u8; STACK_CAP];
        struct FormatWriter {
            begin: *mut u8,
            current: *mut u8,
            end: *mut u8,
            capacity: usize
        }
        impl FormatWriter {
            unsafe fn realloc(&mut self, amount: usize) {
                let new_capacity = (self.capacity + amount).next_power_of_two();
                let offset = self.current.sub_ptr(self.begin);

                self.begin = if self.capacity != STACK_CAP {
                    let layout = Layout::from_size_align_unchecked(self.capacity, 2);
                    std::alloc::realloc(self.begin, layout, new_capacity)
                } else {
                    let layout = Layout::from_size_align_unchecked(new_capacity, 2);
                    let dst = std::alloc::alloc(layout);
                    std::ptr::copy_nonoverlapping(self.begin, dst, STACK_CAP);
                    dst
                };
                self.capacity = new_capacity;
                self.end = self.begin.add(new_capacity);
                self.current = self.begin.add(offset);
            }

            fn shrink_to_fit(self) -> (&'static str, bool) {
                unsafe {
                    let len = self.current.sub_ptr(self.begin);
                    let mut begin = self.begin;
                    if self.capacity != STACK_CAP {
                        let layout = Layout::from_size_align_unchecked(self.capacity, 2);
                        begin = std::alloc::realloc(begin, layout, len);
                    }

                    (
                        std::str::from_utf8_unchecked(
                            std::slice::from_raw_parts(begin, len)),
                        self.capacity == STACK_CAP
                    )
                }
            }
        }
        impl std::fmt::Write for FormatWriter {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                unsafe {
                    let s = s.as_bytes();
                    if self.current.add(s.len()) > self.end {
                        self.realloc(s.len());
                    }
                    std::ptr::copy_nonoverlapping(s.as_ptr(), self.current, s.len());
                    self.current = self.current.add(s.len());
                }
                Ok(())
            }
        }

        let mut fwriter = FormatWriter {
            begin: stack_storage.as_mut_ptr(),
            current: stack_storage.as_mut_ptr(),
            end: unsafe { stack_storage.as_mut_ptr().add(STACK_CAP) },
            capacity: STACK_CAP,
        };

        let mut formatter = std::fmt::Formatter::new(&mut fwriter);
        formatter.write_fmt(arguments).unwrap();

        let (slice, stack_allocated) = fwriter.shrink_to_fit();

        let mut string = vm.heap().allocate_atom(
            Self {
                data: StringData { ptr: std::ptr::null() },
                length: TInteger::from_usize(slice.chars().count()),
                prev: None, next: None,
                size: slice.len()
            },
        );

        unsafe {
            if stack_allocated {
                string.data.small[0] = 1;
                std::ptr::copy_nonoverlapping(
                    stack_storage.as_ptr(),
                    std::ptr::addr_of_mut!(string.data.small[1]),
                    slice.len()
                );
            } else {
                string.data.ptr = slice.as_bytes().as_ptr();
            }
        }

        string
    }
}

impl TString {
    pub fn as_slice<'a>(&self) -> &'a str {
        unsafe {
            let bytes = std::slice::from_raw_parts(self.data_ptr(), self.size());
            let str = std::str::from_utf8_unchecked(bytes);
            str
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn is_small(&self) -> bool {
        unsafe { (self.data.ptr as usize & 0b1) == 1 }
    }

    #[inline]
    unsafe fn data_ptr(&self) -> *const u8 {
        if self.is_small() {
            return std::ptr::addr_of!(self.data.small[1]);
        }
        self.data.ptr
    }

    #[inline]
    unsafe fn data_ptr_mut(&mut self) -> *mut u8 {
        if self.is_small() {
            return std::ptr::addr_of_mut!(self.data.small[1]);
        }
        self.data.ptr as *mut u8
    }

    pub fn get_iterator(self: GCRef<Self>) -> GCRef<TStringIterator> {
        TStringIterator::new(self)
    }
}

impl Atom for TString {
    fn visit(&self, visitor: &mut Visitor) {
        if let Some(prev) = self.prev {
            visitor.feed(prev);
        }
        if let Some(next) = self.next {
            visitor.feed(next);
        }
        self.length.visit(visitor);
    }
}

impl Drop for TString {
    fn drop(&mut self) {
        if !self.is_small() {
            unsafe {
                let align = 2;
                std::alloc::dealloc(
                    self.data_ptr_mut(),
                    Layout::from_size_align_unchecked(self.size(), align)
                );
            }
        }
    }
}

impl Into<TValue> for GCRef<TString> {
    fn into(self) -> TValue {
        TValue::string(self)
    }
}

impl VMDowncast for GCRef<TString> {
    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
        value.query_string()
    }
}

impl PartialEq for GCRef<TString> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl Display for GCRef<TString> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_slice())
    }
}

impl std::fmt::Debug for GCRef<TString> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_slice())
    }
}

impl TString {
    pub(crate) fn initialize_type(vm: &VM) -> GCRef<TType>{
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base(&vm, vm.types().query::<TType>()),
            basety: Some(vm.types().query::<TObject>()),
            basesize: 0, // primitive
            name: TString::from_slice(&vm, "string"),
            modname: TString::from_slice(&vm, "prelude"),
            flags: TypeFlags::empty(),
            ctor: None
        });

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![string], ttype.into(), true).unwrap();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_method(Symbol![fmt], TFunction::rustfunc(
                prelude, Some("string.fmt"), |this: GCRef<TString>| this));
        builder.define_method(Symbol![eq], TFunction::rustfunc(
                prelude, Some("string.eq"), |this: GCRef<TString>, other: GCRef<TString>| this.eq(&other)));

        builder.define_method(
            Symbol![get_iterator],
            TFunction::rustfunc(prelude, Some("string.get_iterator"), TString::get_iterator));

        builder.define_property(
            Symbol![length],
            TProperty::get(
                prelude,
                TFunction::rustfunc(prelude, None, |string: GCRef<TString>| unsafe {
                    let ptr = GCRef::as_ptr(string) as *mut u8;
                    *(ptr.add(offset_of!(TString, length)) as *mut TInteger)
                })));

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
    } 
}

tobject! {
pub struct TStringIterator {
    backing_string: GCRef<TString>,
    byte_offset: usize,
    current_codepoint: Option<char>
}
}

impl TStringIterator {
    pub fn new(string: GCRef<TString>) -> GCRef<TStringIterator> {
        let vm = string.vm();
        vm.heap().allocate_atom(Self {
            base: TObject::base(&vm, vm.types().query::<Self>()),
            backing_string: string,
            byte_offset: 0,
            current_codepoint: None
        })
    }
}

impl TStringIterator {
    pub fn next(mut self: GCRef<Self>) -> bool {
        struct InnerIterator<'a> {
            bytes: &'a [u8],
            offset: &'a mut usize
        }

        impl<'a> Iterator for InnerIterator<'a> {
            type Item = &'a u8;
            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                if *self.offset < self.bytes.len() {
                    let old = &self.bytes[*self.offset];
                    *self.offset += 1;
                    return Some(old);
                }
                None
            }
        }

        let mut bytes_iter = InnerIterator {
            bytes: self.backing_string.as_slice().as_bytes(),
            offset: &mut self.byte_offset
        };

        let codepoint = unsafe {
            core::str::next_code_point(&mut bytes_iter)
        };
        let Some(codepoint) = codepoint else {
            return false;
        };
        self.current_codepoint = Some(unsafe { char::from_u32_unchecked(codepoint) });
        true
    }

    pub fn current(self: GCRef<Self>) -> GCRef<TString> {
        let Some(codepoint) = self.current_codepoint else {
            panic!("Iterator is not even initialized. Call next() first");
        };
        TString::from_format(&self.vm(), format_args!("{codepoint}"))
    }
}

impl Typed for TStringIterator {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> {
        let prelude = vm.modules().prelude();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_method(
            Symbol![next],
            TFunction::rustfunc(prelude, Some("StringIterator.next"), TStringIterator::next));

        builder.define_property(
            Symbol![current],
            TProperty::get(
                prelude,
                TFunction::rustfunc(prelude, Some("StringIterator.current"),
                TStringIterator::current)));

        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base_with_descriptor(vm, vm.types().query::<TType>(), Some(builder.descriptor)),
            basety: Some(vm.types().query::<TObject>()),
            basesize: std::mem::size_of::<Self>(),
            name: TString::from_slice(&vm, "StringIterator"),
            modname: TString::from_slice(&vm, "prelude"),
            flags: TypeFlags::ABSTRACT,
            ctor: None
        });

        entry.insert(TypeId::of::<Self>(), ttype);

        ttype
    }

    fn visit_override(&self, visitor: &mut Visitor)
        where
            Self: TPolymorphicObject {
        visit_polymorphic(self, visitor);
        visitor.feed(self.backing_string);
    }
}

#[derive(Clone, Copy)]
enum IntegerKind {
    Int32(i32),
    Bigint(GCRef<TBigint>),
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TInteger(IntegerKind);

impl TInteger {
    pub fn as_i32(&self) -> Option<i32> {
        match self.0 {
            IntegerKind::Int32(int) => Some(int),
            IntegerKind::Bigint(bigint) => bigint::try_as_i32(bigint)
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self.0 {
            IntegerKind::Int32(int) => usize::try_from(int).ok(),
            IntegerKind::Bigint(bigint) => bigint::try_as_usize(bigint)
        }
    }

    pub fn as_isize(&self) -> Option<isize> {
        match self.0 {
            IntegerKind::Int32(int) => isize::try_from(int).ok(),
            IntegerKind::Bigint(bigint) => bigint::try_as_isize(bigint)
        }
    }

    #[inline(always)]
    pub const fn from_int32(int: i32) -> Self {
        TInteger(IntegerKind::Int32(int))
    }

    #[inline(always)]
    pub const fn from_bigint(bigint: GCRef<TBigint>) -> Self {
        TInteger(IntegerKind::Bigint(bigint))
    }

    #[inline(always)]
    pub fn from_usize(size: usize) -> Self {
        let Ok(size) = isize::try_from(size) else {
            return Self::from_signed_bytes(&(size as i128).to_le_bytes());
        };
        if let Ok(int) = i32::try_from(size) {
            return TInteger(IntegerKind::Int32(int));
        }
        Self::from_signed_bytes(&size.to_le_bytes()) 
    }

    pub fn inc(&mut self) {
        match self.0 {
            IntegerKind::Int32(int) => {
                *self = int.checked_add(1).map(Self::from_int32).unwrap_or_else(|| {
                    Self::from_bigint(bigint::add(&to_bigint(int), &to_bigint(1)))
                });
            }
            IntegerKind::Bigint(bigint) => {
                *self = Self::from_bigint(bigint::add(bigint, &to_bigint(1)));
            }
        }
    }

    pub fn from_signed_bytes(bytes: &[u8]) -> Self {
        todo!("real bigint support")
    }

    pub fn pow(self, exp: TInteger) -> Self {
        if let (IntegerKind::Int32(base), IntegerKind::Int32(exp)) = (self.0, exp.0) {
            assert!(exp > 0);
            if let Some(res) = base.checked_pow(exp as u32) {
                return TInteger(IntegerKind::Int32(res));
            }
        }
        todo!("bigint pow")
    }

    pub fn visit(&self, visitor: &mut Visitor) {
        if let IntegerKind::Bigint(bigint) = self.0 {
            visitor.feed(bigint);
        }
    }
}

impl VMDowncast for TInteger {
    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
        value.query_integer()
    }
}

impl Display for TInteger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            IntegerKind::Int32(int) => f.write_fmt(format_args!("{int}")),
            IntegerKind::Bigint(bigint) => bigint.fmt(f)
        }
    }
}

impl std::ops::Neg for TInteger {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        match self.0 {
            IntegerKind::Int32(int) => {
                let kind = int.checked_neg()
                    .map(|int| IntegerKind::Int32(int))
                    .unwrap_or_else(|| IntegerKind::Bigint(bigint::neg(&to_bigint(int))));
                TInteger(kind)
            }
            IntegerKind::Bigint(bigint) =>
                TInteger(IntegerKind::Bigint(bigint::neg(bigint)))
        }
    }
}

impl crate::interop::vmops::Invert for TInteger {
    type Output = Self;

    #[inline(always)]
    fn invert(self) -> Self::Output {
        match self.0 {
            IntegerKind::Int32(int) =>
                TInteger(IntegerKind::Int32(!int)),
            IntegerKind::Bigint(bigint) =>
                TInteger(IntegerKind::Bigint(bigint::invert(bigint)))
        }
    }
}

pub fn int_init_unarys(builder: &mut TTypeBuilder) {
    let vm = builder.vm();

    let prelude = vm.modules().prelude();

    builder.define_method(Symbol![neg], TFunction::rustfunc(
            prelude, Some("int.neg"),
            <TInteger as std::ops::Neg>::neg));

    builder.define_method(Symbol![invert], TFunction::rustfunc(
            prelude, Some("int.invert"),
            <TInteger as crate::interop::vmops::Invert>::invert));
}

macro_rules! impl_int_safe {
    ($lhs:ident, $rhs:ident, $fn:ident,) => {
        Self::from_int32($lhs.$fn($rhs))
    };
    ($lhs:ident, $rhs:ident, $fn:ident, $checked_fn:ident) => {
        $lhs.$checked_fn($rhs as _).map(Self::from_int32).unwrap_or_else(|| {
            Self::from_bigint(bigint::$fn(&to_bigint($lhs), &to_bigint($rhs)))
        })
    };
}

macro_rules! impl_int_arithmetic {
    ($op:ident, $ty:ident, $fn:ident, $($checked_fn:ident)?) => { 
        impl std::ops::$op for $ty {
            type Output = $ty;

            #[inline(always)]
            fn $fn(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (IntegerKind::Int32(lhs), IntegerKind::Int32(rhs)) =>
                        impl_int_safe!(lhs, rhs, $fn, $($checked_fn)?),
                    (IntegerKind::Int32(lhs), IntegerKind::Bigint(rhs)) =>
                        Self::from_bigint(bigint::$fn(&to_bigint(lhs), rhs)),
                    (IntegerKind::Bigint(lhs), IntegerKind::Int32(rhs)) =>
                        Self::from_bigint(bigint::$fn(lhs, &to_bigint(rhs))),
                    (IntegerKind::Bigint(lhs), IntegerKind::Bigint(rhs)) =>
                        Self::from_bigint(bigint::$fn(lhs, rhs)),
                }
            }
        }

        impl std::ops::$op<TValue> for $ty {
            type Output = TValue;

            #[inline(always)]
            fn $fn(self, rhs: TValue) -> TValue {
                use std::ops::$op;

                if let Some(rhs) = rhs.query_integer() {
                    return $op::$fn(self, rhs).into();
                }
                panic!("Not implemented");
            }
        }
    };
}

macro_rules! iter_int_arithmetics {
    ($(impl $op:ident for $ty:ident in ($fn:ident$(, $checked_fn:ident)?);)*) => {
        $(impl_int_arithmetic!($op, $ty, $fn, $($checked_fn)?);)*

        pub(crate) fn int_init_arithmetics(builder: &mut TTypeBuilder) {
            let vm = builder.vm();
            let prelude = vm.modules().prelude();
            $(
                builder.define_method(Symbol![$fn], TFunction::rustfunc(
                        prelude, Some(concat!("int.", stringify!($fn))),
                        <TInteger as std::ops::$op<TValue>>::$fn));
            )*
        }
    };
}

iter_int_arithmetics! {
    impl Add for TInteger in (add, checked_add);
    impl Sub for TInteger in (sub, checked_sub);
    impl Mul for TInteger in (mul, checked_mul);
    impl Div for TInteger in (div, checked_div);
    impl Rem for TInteger in (rem, checked_rem);

    impl Shl for TInteger in (shl, checked_shl);
    impl Shr for TInteger in (shr, checked_shr);

    impl BitAnd for TInteger in (bitand);
    impl BitOr for TInteger in (bitor);
    impl BitXor for TInteger in (bitxor);
}

macro_rules! impl_int_cmp {
    ($op:ident, $ty:ident, $fn:ident) => { 
        impl $crate::interop::vmops::$op for $ty {
            #[inline(always)]
            fn $fn(self, rhs: Self) -> TBool {
                match (self.0, rhs.0) {
                    (IntegerKind::Int32(lhs), IntegerKind::Int32(rhs)) =>
                        lhs.$fn(&rhs).into(),
                    (IntegerKind::Int32(lhs), IntegerKind::Bigint(rhs)) =>
                        bigint::$fn(&to_bigint(lhs), rhs).into(),
                    (IntegerKind::Bigint(lhs), IntegerKind::Int32(rhs)) =>
                        bigint::$fn(lhs, &to_bigint(rhs)).into(),
                    (IntegerKind::Bigint(lhs), IntegerKind::Bigint(rhs)) =>
                        bigint::$fn(lhs, rhs).into(),
                }
            }
        }

        impl $crate::interop::vmops::$op<TValue> for $ty {
            #[inline(always)]
            fn $fn(self, rhs: TValue) -> TBool {
                use $crate::interop::vmops::$op;

                if let Some(rhs) = rhs.query_integer() {
                    return $op::$fn(self, rhs).into();
                }
                panic!("Not implemented");
            }
        }
    };
}

macro_rules! iter_int_cmps {
    ($(impl $op:ident for $ty:ident in $fn:ident;)*) => {
        $(impl_int_cmp!($op, $ty, $fn);)*

        pub(crate) fn int_init_cmps(builder: &mut TTypeBuilder) {
            let vm = builder.vm();
            let prelude = vm.modules().prelude();
            $(
                builder.define_method(Symbol![$fn], TFunction::rustfunc(
                        prelude, Some(concat!("int.", stringify!($fn))),
                        <TInteger as $crate::interop::vmops::$op<TValue>>::$fn));
            )*
        }
    };
}

iter_int_cmps! {
    impl Lt for TInteger in lt; impl Le for TInteger in le;
    impl Gt for TInteger in gt; impl Ge for TInteger in ge;
}

impl Into<TValue> for TInteger {
    #[inline(always)]
    fn into(self) -> TValue {
        match self.0 {
            IntegerKind::Int32(int) => TValue::int32(int),
            IntegerKind::Bigint(bigint) => todo!()
        }
    }
}

impl Into<TFloat> for TInteger {
    fn into(self) -> TFloat {
        match self.0 {
            IntegerKind::Int32(int) => TFloat(int as f64),
            IntegerKind::Bigint(bigint) =>
                TFloat(bigint::smart_to_f64(bigint))
        }
    }
}

impl TInteger {
    pub(crate) fn initialize_type(vm: &VM) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base(&vm, vm.types().query::<TType>()),
            basety: Some(vm.types().query::<TObject>()),
            basesize: 0, // primitive
            name: TString::from_slice(&vm, "int"),
            modname: TString::from_slice(&vm, "prelude"),
            flags: TypeFlags::empty(),
            ctor: None
        });

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![int], ttype.into(), true).unwrap();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_method(Symbol![fmt], TFunction::rustfunc(
                prelude, Some("int.fmt"),
                move |this: TInteger| {
                    let vm = prelude.vm();
                    TString::from_format(&vm, format_args!("{this}"))
                }));

        builder.define_method(Symbol![pow], TFunction::rustfunc(
                prelude, Some("int.pow"), TInteger::pow));

        int_init_arithmetics(&mut builder);
        int_init_cmps(&mut builder);

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TFloat(f64);

impl TFloat {
    pub const fn as_float(self) -> f64 {
        self.0
    }

    pub const fn from_float(float: f64) -> Self {
        TFloat(float)
    }
}

impl Display for TFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.0))
    }
}

impl Into<TValue> for TFloat {
    fn into(self) -> TValue {
        TValue::float(self.0)
    }
}

impl VMDowncast for TFloat {
    #[inline(always)]
    fn vmdowncast(value: TValue, _vm: &VM) -> Option<Self> {
        value.query_float()
    }
}

impl std::ops::Neg for TFloat {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        TFloat(-self.0)
    }
}

pub fn float_init_unarys(builder: &mut TTypeBuilder) {
    let vm = builder.vm();

    builder.define_method(Symbol![neg], TFunction::rustfunc(
            vm.modules().prelude(), Some("float::neg"),
            <TInteger as std::ops::Neg>::neg));
}

macro_rules! impl_float_arithmetic {
    ($op:ident, $ty:ident, $fn:ident) => { 
        impl std::ops::$op for $ty {
            type Output = $ty;

            #[inline(always)]
            fn $fn(self, rhs: Self) -> Self::Output {
                Self(self.0.$fn(rhs.0))
            }
        }

        impl std::ops::$op<TValue> for $ty {
            type Output = TValue;

            #[inline(always)]
            fn $fn(self, rhs: TValue) -> TValue {
                use std::ops::$op;

                if let Some(rhs) = rhs.query_float() {
                    return $op::$fn(self, rhs).into();
                } else if let Some(rhs) = rhs.query_integer() {
                    return $op::<TFloat>::$fn(self, rhs.into()).into();
                }
                panic!("Not implemented");
            }
        }
    };
}

macro_rules! iter_float_arithmetics {
    ($(impl $op:ident for $ty:ident in $fn:ident;)*) => {
        $(impl_float_arithmetic!($op, $ty, $fn);)*

        pub(crate) fn float_init_arithmetics(builder: &mut TTypeBuilder) {
            let vm = builder.vm();
            let prelude = vm.modules().prelude();
            $(
                builder.define_method(Symbol![$fn], TFunction::rustfunc(
                        prelude, Some(concat!("float.", stringify!($fn))),
                        <TFloat as std::ops::$op<TValue>>::$fn));
            )*
        }
    };
}

iter_float_arithmetics! {
    impl Add for TFloat in add;
    impl Sub for TFloat in sub;
    impl Mul for TFloat in mul;
    impl Div for TFloat in div;
    impl Rem for TFloat in rem;
}

macro_rules! impl_float_cmp {
    ($op:ident, $ty:ident, $fn:ident) => { 
        impl $crate::interop::vmops::$op for $ty {
            #[inline(always)]
            fn $fn(self, rhs: Self) -> TBool {
                TBool(self.0.$fn(&rhs.0))
            }
        }

        impl $crate::interop::vmops::$op<TValue> for $ty {
            #[inline(always)]
            fn $fn(self, rhs: TValue) -> TBool {
                use $crate::interop::vmops::$op;

                if let Some(rhs) = rhs.query_float() {
                    return $op::$fn(self, rhs).into();
                } else if let Some(rhs) = rhs.query_integer() {
                    return $op::<TFloat>::$fn(self, rhs.into()).into();
                }
                panic!("Not implemented");
            }
        }
    };
}

macro_rules! iter_float_cmps {
    ($(impl $op:ident for $ty:ident in $fn:ident;)*) => {
        $(impl_float_cmp!($op, $ty, $fn);)*

        pub(crate) fn float_init_cmps(builder: &mut TTypeBuilder) {
            let vm = builder.vm();
            let mut prelude = vm.modules().prelude();
            $(
                builder.define_method(Symbol![$fn], TFunction::rustfunc(
                        prelude, Some(concat!("float::", stringify!($fn))),
                        <TFloat as $crate::interop::vmops::$op<TValue>>::$fn));
            )*
        }
    };
}

iter_float_cmps! {
    impl Lt for TFloat in lt; impl Le for TFloat in le;
    impl Gt for TFloat in gt; impl Ge for TFloat in ge;
}

impl TFloat {
    pub(crate) fn initialize_type(vm: &VM) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base(&vm, vm.types().query::<TType>()),
            basety: Some(vm.types().query::<TObject>()),
            basesize: 0, // primitive
            name: TString::from_slice(&vm, "float"),
            modname: TString::from_slice(&vm, "prelude"),
            flags: TypeFlags::empty(),
            ctor: None
        });

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![float], ttype.into(), true).unwrap();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_method(Symbol![fmt], TFunction::rustfunc(
                prelude, Some("float.fmt"),
                move |this: TFloat| {
                    let vm = prelude.vm();
                    TString::from_format(&vm, format_args!("{this}"))
                }));

        float_init_arithmetics(&mut builder);
        float_init_unarys(&mut builder);
        float_init_cmps(&mut builder);

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
    }
}

pub struct TList {
    pub length: TInteger,
    capacity: isize,
    data: ListData

}

union ListData {
    immediate: [TValue; 0],
    ptr: *mut TValue
}

unsafe impl std::marker::Sync for ListData {}
unsafe impl std::marker::Send for ListData {}

static_assertions::const_assert!(std::mem::size_of::<TList>() == 4 * std::mem::size_of::<&()>());

impl TList {
    pub fn new_empty(vm: &VM) -> GCRef<Self> {
        vm.heap().allocate_atom(Self {
            length: TInteger::from_int32(0),
            capacity: 0,
            data: ListData { ptr: std::ptr::null_mut() }
        })
    }

    pub fn new_with_capacity(vm: &VM, capacity: usize) -> GCRef<Self> {
        let extra_bytes = capacity * std::mem::size_of::<TValue>();
        if extra_bytes >= 4096 /* page size */ {
            const ITEM_SIZE: usize  = std::mem::size_of::<TValue>();
            unsafe {
                let layout = Layout::from_size_align_unchecked(
                    capacity * ITEM_SIZE, std::mem::align_of::<TValue>());
                return vm.heap().allocate_atom(Self {
                    length: TInteger::from_int32(0),
                    capacity: capacity as isize,
                    data: ListData { ptr: std::alloc::alloc(layout) as *mut _ }
                });
            }
        }
        vm.heap().allocate_var_atom(Self {
            length: TInteger::from_int32(0),
            capacity: -(capacity as isize),
            data: ListData { ptr: std::ptr::null_mut() }
        }, extra_bytes)
    }
}

const AMORTIZED_LIST_INITIAL_CAP: usize = std::mem::size_of::<&()>();

impl TList {
    #[inline]
    pub unsafe fn data_ptr(&self) -> *mut TValue {
        if self.capacity < 0 {
            return self.data.immediate.as_ptr() as *mut _;
        }
        self.data.ptr
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity.abs() as usize
    }

    #[inline]
    fn index_access_helper<'a>(&self, index: usize) -> &'a mut TValue {
        if index >= self.length.as_usize().unwrap() {
            panic!("list out of bounds access");
        }
        unsafe {
            let ptr = self.data_ptr();
            &mut *ptr.add(index)
        }
    }

    #[inline]
    fn wrapping_index_translation(&self, index: TInteger) -> usize {
        let index = index.as_isize().unwrap();
        if index >= 0 {
            index as usize 
        } else {
            let index = self.length.as_isize().unwrap() + index;
            usize::try_from(index).unwrap() 
        }
    }

    pub fn grow(mut self: GCRef<Self>, new_capacity: usize) {
        let mut new_capacity = (new_capacity).next_power_of_two();
        if new_capacity < AMORTIZED_LIST_INITIAL_CAP {
            new_capacity = AMORTIZED_LIST_INITIAL_CAP;
        }

        const ITEM_SIZE: usize  = std::mem::size_of::<TValue>();
        unsafe {
            let new_ptr = if self.capacity >= 0 {
                let layout = Layout::from_size_align_unchecked(
                    self.capacity() * ITEM_SIZE, std::mem::align_of::<TValue>());
                std::alloc::realloc(self.data_ptr() as *mut u8, layout, new_capacity * ITEM_SIZE) as *mut TValue
            } else {
                // FIXME: here we leak some memory, from the previous, direct list allocation.
                // Currently this will remain around, until this list gets fully garbage collected.
                // In the future this should probably call into the GC, shrinking its size to
                // just std::mem::size_of::<TList>()

                let layout = Layout::from_size_align_unchecked(
                    new_capacity * ITEM_SIZE, std::mem::align_of::<TValue>());
                let dst = std::alloc::alloc(layout) as *mut TValue;
                std::ptr::copy_nonoverlapping(self.data_ptr(), dst, self.capacity());
                dst
            };
            self.data.ptr = new_ptr;
            self.capacity = new_capacity as isize;
        }
    }

    pub fn push(mut self: GCRef<Self>, value: TValue) {
        unsafe {
            let length = self.length.as_usize().unwrap();
            if length == self.capacity() {
                self.grow(self.capacity() + 1);
            }

            *self.data_ptr().add(length) = value;
            self.length.inc();
        }
    }

    pub fn repeat(self: GCRef<Self>, count: TInteger) -> GCRef<Self> {
        let length = self.length.as_usize().expect("sensible size");
        let count = count.as_usize().expect("sensible size");

        let new_length = count * length;
        let mut new_list = Self::new_with_capacity(&self.vm(), new_length);

        for _ in 0..count {
            for idx in 0..length {
                new_list.push(self[idx]);
            }
        }

        new_list
    }

    pub(crate) fn initialize_type(vm: &VM) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base(&vm, vm.types().query::<TType>()),
            basety: Some(vm.types().query::<TObject>()),
            basesize: 0, // primitive
            name: TString::from_slice(&vm, "list"),
            modname: TString::from_slice(&vm, "prelude"),
            flags: TypeFlags::empty(),
            ctor: None
        });

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![list], ttype.into(), true).unwrap();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_method(
            Symbol![fmt],
            TFunction::rustfunc(prelude, Some("list.fmt"), |this: GCRef<TList>| {
                TString::from_format(&this.vm(), format_args!("{this}"))
            }));

        builder.define_method(
            Symbol![push],
            TFunction::rustfunc(prelude, Some("list.push"), TList::push));

        builder.define_method(
            Symbol![repeat],
            TFunction::rustfunc(prelude, Some("list.repeat"), TList::repeat));

        builder.define_method(
            Symbol![get_index],
            TFunction::rustfunc(prelude, Some("list.get_index"),
            |this: GCRef<TList>, index: TInteger| this[index]));

        builder.define_method(
            Symbol![set_index],
            TFunction::rustfunc(prelude, Some("list.set_index"),
            |mut this: GCRef<TList>, index: TInteger, value: TValue| { this[index] = value; }));

        builder.define_property(
            Symbol![length],
            TProperty::get(
                prelude,
                TFunction::rustfunc(prelude, None, |list: GCRef<TList>| unsafe {
                    let ptr = GCRef::as_ptr(list) as *mut u8;
                    *(ptr.add(offset_of!(TList, length)) as *mut TInteger)
                })));

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
    }
}

impl std::ops::Index<usize> for TList {
    type Output = TValue;

    fn index(&self, index: usize) -> &Self::Output {
        Self::index_access_helper(self, index)
    }
}

impl std::ops::IndexMut<usize> for TList {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        Self::index_access_helper(self, index) 
    }
}

impl std::ops::Index<TInteger> for TList {
    type Output = TValue;

    fn index(&self, index: TInteger) -> &Self::Output {
        Self::index_access_helper(self, self.wrapping_index_translation(index))
    }
}

impl std::ops::IndexMut<TInteger> for TList {
    fn index_mut(&mut self, index: TInteger) -> &mut Self::Output { 
        Self::index_access_helper(self, self.wrapping_index_translation(index))
    }
}

impl std::fmt::Display for GCRef<TList> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vm = self.vm();
        let length = self.length.as_usize().unwrap();
        f.write_char('[')?;
        for i in 0..length {
            if i != 0 {
                f.write_str(", ")?;
            }
            let item = self[i];
            let fmt: GCRef<TString> = tcall!(&vm, TValue::fmt(item));
            f.write_str(fmt.as_slice())?;
        }
        f.write_char(']')?;
        Ok(())
    }
}

impl Into<TValue> for GCRef<TList> {
    #[inline]
    fn into(self) -> TValue {
        TValue::object_impl(self, TValueKind::List)
    }
}

impl VMDowncast for GCRef<TList> {
    #[inline]
    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
        value.query_list()
    }
}

impl Atom for TList {
    fn visit(&self, visitor: &mut Visitor) {
        let this = unsafe { GCRef::from_raw(self as *const Self) };
        let length = this.length.as_usize().unwrap();
        for idx in 0..length {
            this[idx].visit(visitor);
        }
    }
}

bitflags! {
    #[derive(Debug)]
    pub struct TypeFlags: u8 {
        /// Allows a type to be extended
        const OPEN     = 0b100;
        /// Makes a type uninstanciatable
        /// specifically, it makes `Object.new` refuse to create an instance 
        const ABSTRACT = 0b010;
        /// Allows a type's instance to be altered
        const VARIABLE = 0b001;
    }
}


tobject! {
pub struct TType {
    pub basety: Option<GCRef<TType>>,
    pub basesize: usize,
    pub name: GCRef<TString>,
    pub modname: GCRef<TString>,
    pub flags: TypeFlags,
    pub ctor: TCtor
}
}

pub type TCtor = Option<unsafe fn(*mut TObject, GCRef<TType>)>;
static_assertions::const_assert!(std::mem::size_of::<TCtor>() == std::mem::size_of::<&()>());

impl TType {
    pub fn is_subclass(self: GCRef<Self>, needle: GCRef<TType>) -> bool {
        let mut current = self;
        loop {
            if GCRef::refrence_eq(current, needle) {
                return true;
            }
            let Some(base) = current.basety else {
                break;
            };
            current = base;
        }
        false
    }

    pub fn properties<F: FnMut(GCRef<TProperty>)>(self: GCRef<Self>, mut f: F) {
        let vm = self.vm();
        self.base.attributes(|attr| {
            if let Some(prop) = attr.query_object::<TProperty>(&vm) {
                f(prop);
            }
        })
    }

    unsafe fn dynamic_ctor(this: *mut TObject, ty: GCRef<TType>) {
        let basety = ty.basety.unwrap_unchecked();
        basety.ctor.unwrap_unchecked()(this, basety);
        let offset = basety.basesize;
        let size = ty.basesize - offset;
        debug_assert!(size % std::mem::size_of::<TValue>() == 0);
        let this = std::slice::from_raw_parts_mut(
            (this as *mut TValue).byte_add(offset),
            size / std::mem::size_of::<TValue>()
        );
        this.fill(TValue::null());
    }
}

impl Typed for TType {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base_with_descriptor(vm, unsafe { GCRef::null() }, None),
            basety: None,
            basesize: std::mem::size_of::<Self>(),
            name: TString::from_slice(vm, "Type"),
            modname: TString::from_slice(vm, "prelude"),
            flags: TypeFlags::ABSTRACT,
            ctor: None
        });

        entry.insert(TypeId::of::<Self>(), ttype);

        ttype.base = TObject::base(vm, ttype);
        ttype.basety = Some(vm.types().query::<TObject>());

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![Type], ttype.into(), true).unwrap();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_property(
            Symbol![base],
            TProperty::offset::<TType, Option<GCRef<TType>>>(prelude, Accessor::GET, offset_of!(TType, basety)));
        builder.define_property(
            Symbol![name],
            TProperty::offset::<TType, GCRef<TString>>(prelude, Accessor::GET, offset_of!(TType, name)));
        builder.define_property(
            Symbol![modname],
            TProperty::offset::<TType, GCRef<TString>>(prelude, Accessor::GET, offset_of!(TType, modname)));
        builder.define_method(
            Symbol![fmt],
            TFunction::rustfunc(prelude, Some("type.fmt"), |this: GCRef<TType>| {
                TString::from_format(&this.vm(), format_args!("[type {}] {{}}", this.name))
            }));
        builder.define_static_method(
            Symbol![of],
            TFunction::rustfunc(prelude, Some("type.of"), move |value: TValue| {
                value.ttype(&prelude.vm())
            }));

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
        
    }
}

pub struct TTypeBuilder<'l> {
    vm: &'l VM,
    descriptor: GCRef<HashTable<(Symbol, TValue)>>
}

impl<'l> TTypeBuilder<'l> {
    pub fn new(vm: &'l VM) -> Self {
        Self {
            vm,
            descriptor: vm.heap().allocate_atom(Default::default())
        }
    }

    pub fn vm(&self) -> &VM {
        self.vm
    }

    pub fn build(self) -> GCRef<TType> {
        todo!()
    }

    fn insert(&mut self, name: Symbol, value: TValue) {
        match self.descriptor.entry(
            name.hash,
            |val| val.0 == name,
            |val| val.0.hash) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = (name, value);
            }
            Entry::Vacant(entry) => {
                entry.insert((name, value));
            }
        }
    }

    pub fn define_property(&mut self, name: Symbol, property: GCRef<TProperty>) {
        self.insert(name, property.into());
    }

    pub fn define_method(&mut self, name: Symbol, mut method: GCRef<TFunction>) {
        method.flags.insert(FunctionFlags::METHOD);
        self.insert(name, method.into());
    }

    pub fn define_static_method(&mut self, name: Symbol, method: GCRef<TFunction>) { 
        self.insert(name, method.into());
    }
}

tobject! {
pub struct WeirdBuilder {
    pub module: GCRef<TModule>,
    ttype: GCRef<TType>,
    descriptor: GCRef<HashTable<(Symbol, TValue)>>
}
}

impl WeirdBuilder {
    pub fn new(module: GCRef<TModule>, name: GCRef<TString>, base: GCRef<TType>) -> GCRef<WeirdBuilder> {
        let vm = module.vm();
        let descriptor = vm.heap().allocate_atom(Default::default());
        if !base.flags.contains(TypeFlags::OPEN) {
            panic!("cannot extend closed type {}", base.name);
        }
        let ttype = vm.heap().allocate_atom(TType {
            base: TObject::base_with_descriptor(&vm, vm.types().query::<TType>(), Some(descriptor)),
            basety: Some(base),
            basesize: base.basesize,
            name,
            modname: module.name,
            flags: TypeFlags::OPEN,
            ctor: Some(TType::dynamic_ctor)
        });
        vm.heap().allocate_atom(WeirdBuilder {
            base: TObject::base(&vm, vm.types().query::<Self>()),
            module, ttype,
            descriptor
        })
    }
}

impl WeirdBuilder {
    pub fn declare(mut self: GCRef<WeirdBuilder>, kind: i32, name: GCRef<TString>, value: TValue) {
        const METHOD: i32 = 0;
        const OFFSET_PROPERTY: i32 = 1;
        const CONSTANT: i32 = 2;

        let name = self.vm().symbols().intern(name);

        let value = match kind {
            METHOD =>
                self.insert(name, value),
            OFFSET_PROPERTY => {
                let accessor = value.query_integer()
                    .unwrap()
                    .as_usize()
                    .unwrap();
                let accessor = Accessor::from_bits(accessor as u8).unwrap();
                TProperty::offset2(self, name, accessor);
            }
            CONSTANT =>
                self.insert(name, value),
            _ =>
                panic!("Invalid kind"),
        };
    }

    pub fn finalize(self: GCRef<WeirdBuilder>) -> GCRef<TType> {
        self.ttype
    }

    fn insert(mut self: GCRef<WeirdBuilder>, name: Symbol, value: TValue) {
        match self.descriptor.entry(
            name.hash,
            |val| val.0 == name,
            |val| val.0.hash) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = (name, value);
            }
            Entry::Vacant(entry) => {
                entry.insert((name, value));
            }
        }
    }
}

impl Typed for WeirdBuilder {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> {
        let prelude = vm.modules().prelude();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_method(
            Symbol![declare],
            TFunction::rustfunc(prelude, Some("TypeBuilder.declare"), WeirdBuilder::declare));

        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base_with_descriptor(vm, vm.types().query::<TType>(), Some(builder.descriptor)),
            basety: Some(vm.types().query::<TObject>()),
            basesize: std::mem::size_of::<Self>(),
            name: TString::from_slice(&vm, "TypeBuilder"),
            modname: TString::from_slice(&vm, "prelude"),
            flags: TypeFlags::ABSTRACT,
            ctor: None
        });

        entry.insert(TypeId::of::<Self>(), ttype);

        ttype
    }

    fn visit_override(&self, visitor: &mut Visitor)
        where
            Self: TPolymorphicObject {
        visitor.feed(self.module); 
        visitor.feed(self.descriptor); 
        visitor.feed(self.ttype); 
    }
}

bitflags! {
    pub struct Accessor: u8 {
        const GET = 0b10;
        const SET = 0b01;
    }
}

tobject! {
pub struct TProperty {
    pub get: Option<GCRef<TFunction>>,
    pub set: Option<GCRef<TFunction>>
}
}

impl TProperty {
    unsafe fn ctor(this: *mut TObject, ty: GCRef<TType>) {
        TObject::ctor(this, ty);
        let this = &mut *(this as *mut TProperty);
        this.get = None;
        this.set = None;
    }

    pub fn offset<Slf, P>(module: GCRef<TModule>, accessor: Accessor, offset: usize) -> GCRef<Self>
    where
        Slf: TPolymorphicObject,
        P: VMCast + VMDowncast + Copy + 'static
    {
        let get = if accessor.contains(Accessor::GET) {
            Some(TFunction::rustfunc(module, None, move |object: TPolymorphicWrapper<Slf>| {
                unsafe { *object.raw_access::<P>(offset) }
            }))
        } else {
            None
        };
        let set = if accessor.contains(Accessor::SET) {
            Some(TFunction::rustfunc(module, None, move |object: TPolymorphicWrapper<Slf>, value: P| {
                unsafe { *object.raw_access::<P>(offset) = value; }
            }))
        } else {
            None
        };

        let vm = module.vm();
        vm.heap().allocate_atom(Self {
            base: TObject::base(&vm, vm.types().query::<Self>()),
            get, set
        })
    }

    pub fn offset2<'a>(mut builder: GCRef<WeirdBuilder>, name: Symbol, accessor: Accessor) {
        let module = builder.module;
        let mut ttype = builder.ttype;
        let offset = ttype.basesize;
        ttype.basesize += std::mem::size_of::<TValue>();

        let get = if accessor.contains(Accessor::GET) {
            Some(TFunction::rustfunc(module, None, move |this: TValue| {
                let object = this.query_tobject()
                    .filter(|obj| obj.ty.is_subclass(ttype))
                    .unwrap();
                unsafe {
                    let ptr = GCRef::as_ptr(object) as *mut u8;
                    *(ptr.add(offset) as *mut TValue)
                }
            }))
        } else {
            None
        };

        let set = if accessor.contains(Accessor::SET) {
            Some(TFunction::rustfunc(module, None, move |this: TValue, value: TValue| {
                let object = this.query_tobject()
                    .filter(|obj| obj.ty.is_subclass(ttype))
                    .unwrap();
                unsafe {
                    let ptr = GCRef::as_ptr(object) as *mut u8;
                    *(ptr.add(offset) as *mut TValue) = value;
                }
            }))
        } else {
            None
        };

        let vm = module.vm();
        let property = vm.heap().allocate_atom(Self {
            base: TObject::base(&vm, vm.types().query::<Self>()),
            get, set
        });

        builder.insert(name, property.into());
    }

    pub fn get(module: GCRef<TModule>, get: GCRef<TFunction>) -> GCRef<Self> {
        let vm = module.vm();
        vm.heap().allocate_atom(Self {
            base: TObject::base(&vm, vm.types().query::<Self>()),
            get: Some(get),
            set: None
        })
    }
}

impl Typed for TProperty {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> {  
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base(vm, vm.types().query::<TType>()),
            basety: Some(vm.types().query::<TObject>()),
            basesize: std::mem::size_of::<Self>(),
            name: TString::from_slice(vm, "Property"),
            modname: TString::from_slice(vm, "prelude"),
            flags: TypeFlags::empty(),
            ctor: Some(TProperty::ctor)
        });
        entry.insert(TypeId::of::<Self>(), ttype);

        let mut builder = TTypeBuilder::new(vm);
        let prelude = vm.modules().prelude();

        builder.define_property(
            Symbol![get],
            TProperty::offset::<Self, Option<GCRef<TFunction>>>(prelude, Accessor::GET, offset_of!(TProperty, get)));

        builder.define_property(
            Symbol![set],
            TProperty::offset::<Self, Option<GCRef<TFunction>>>(prelude, Accessor::GET, offset_of!(TProperty, set)));

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
    }
}

bitflags! {
pub struct FunctionFlags: u32 {
    const METHOD = 0b01;
}
}

tobject! {
pub struct TFunction {
    pub name: Option<GCRef<TString>>,
    pub module: GCRef<TModule>,
    pub flags: FunctionFlags,
    pub kind: TFnKind
}
}

#[repr(C)]
pub enum TFnKind {
    Function(TRawCode),
    Nativefunc(Nativefunc),
    BoundMethod(GCRef<TFunction>, TValue),
}

unsafe impl std::marker::Sync for TFnKind {}
unsafe impl std::marker::Send for TFnKind {}

impl Typed for TFunction {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::base(vm, vm.types().query::<TType>()),
            basety: Some(vm.types().query::<TObject>()),
            basesize: std::mem::size_of::<Self>(),
            name: TString::from_slice(vm, "Function"),
            modname: TString::from_slice(vm, "prelude"),
            flags: TypeFlags::ABSTRACT,
            ctor: None
        });
        entry.insert(TypeId::of::<Self>(), ttype);

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![Function], ttype.into(), true).unwrap();
        let mut builder = TTypeBuilder::new(vm);

        builder.define_property(
            Symbol![name],
            TProperty::offset::<Self, Option<GCRef<TString>>>(prelude, Accessor::GET, offset_of!(Self, name)));

        builder.define_method(
            Symbol![fmt],
            TFunction::rustfunc(prelude, Some("Function.fmt"), |this: GCRef<TFunction>| {
                let name = if let Some(name) = this.name {
                    name.as_slice()
                } else {
                    "(anonymous)"
                };

                let mut string = format!("def {name} ");
                if let TFnKind::Nativefunc(..) = this.kind {
                    string.push_str("[Native Code]");
                } else if let TFnKind::BoundMethod(..) = this.kind {
                    string.push_str("[Bound Method]");
                } else {
                    string.push_str("{}");
                }
                TString::from_slice(&this.vm(), &string)
            }));

        ttype.base.descriptor = Some(builder.descriptor);

        ttype
    }

    fn visit_override(&self, visitor: &mut Visitor)
        where
            Self: TPolymorphicObject { 
        visit_polymorphic(self, visitor);
        match &self.kind {
            TFnKind::Function(code) => {
                for desc in code.descriptors() {
                    desc.visit(visitor);
                }
            }
            TFnKind::BoundMethod(tfunction, this) => {
                visitor.feed(*tfunction);
                this.visit(visitor);
            }
            TFnKind::Nativefunc(..) => {}
        }
    }
}

impl TFunction {
    pub fn rustfunc<In: VMArgs + 'static, R: VMCast + 'static, F: Fn<In, Output = R> + Sync + Send + 'static>(
        module: GCRef<TModule>,
        name: Option<&str>,
        func: F
    ) -> GCRef<TFunction> {
        let vm = module.vm();

        let id = TypeId::of::<(In, R)>();
        let closure = Box::new(func);

        let closure: *const _ = Box::leak(closure);

        let mut func = vm.heap().allocate_atom(Self {
            base: TObject::base(&vm, vm.types().query::<Self>()),
            name: name.map(|name| TString::from_slice(&vm, name)),
            module,
            flags: FunctionFlags::empty(),
            kind: TFnKind::Nativefunc(Nativefunc {
                id,
                closure: closure as *const (),
                fastcall: fastcall::<In, R, F> as *const (),
                traitfn: traitfn::<In, R>
            })
        });

        func
    }

    #[inline(always)]
    pub fn call(self: GCRef<TFunction>, arguments: TArgsBuffer) -> TValue {
        match &self.kind {
            TFnKind::Function(code) =>
                code.evaluate(StackFrame::new(self, arguments)),
            TFnKind::Nativefunc(n @ Nativefunc { traitfn, .. })=>
                traitfn(n, self.module, arguments),
            TFnKind::BoundMethod(tfunction, base) => {
                let mut arguments: Vec<_> = arguments.into_iter(0, true)
                    .collect();
                arguments.insert(0, *base);
                tfunction.call(TArgsBuffer::create(&mut arguments))
            }
        }
    }

    #[inline(always)]
    pub fn fastcall<In, R>(self: GCRef<TFunction>, args: In) -> CallResult<In, R>
    where
        In: std::marker::Tuple + 'static,
        R: 'static
    {
        match &self.kind {
            TFnKind::Function(..) => CallResult::NotImplemented(args),
            TFnKind::BoundMethod(..) => CallResult::NotImplemented(args),
            TFnKind::Nativefunc(n) => n.call(args)
        }
    }

    pub fn bind(self: GCRef<TFunction>, this: TValue) -> GCRef<TFunction> {
        debug!("created method wrapper for function {:?}", self.name);
        let vm = self.vm();
        vm.heap().allocate_atom(TFunction {
            base: TObject::base(&vm, vm.types().query::<TFunction>()),
            name: self.name,
            module: self.module,
            kind: TFnKind::BoundMethod(self, this),
            flags: FunctionFlags::empty()
        })
    }
}

fn traitfn<In, R>(this: &Nativefunc, module: GCRef<TModule>, args: TArgsBuffer) -> TValue
where
    In: VMArgs + 'static,
    R: VMCast + 'static
{
    let vm = module.vm();
    let decoded_args = In::try_decode(&vm, args).unwrap();
    let result = unsafe { this.call_unchecked(decoded_args) };
    R::vmcast(result, &vm)
}

pub enum CallResult<In, R> {
    NotImplemented(In),
    Result(R)
}

#[repr(C)]
pub struct Nativefunc {
    id: TypeId,
    closure: *const (),
    fastcall: *const (),
    traitfn: fn(&Nativefunc, GCRef<TModule>, TArgsBuffer) -> TValue,
}

impl Nativefunc {
    unsafe fn call_unchecked<In, R>(&self, args: In) -> R
    where
        In: std::marker::Tuple + 'static,
        R: 'static,
    {
        let fn_ptr: unsafe fn(*const(), In) -> R = transmute(self.fastcall);
        fn_ptr(self.closure, args)
    }

    fn call<In, R>(&self, args: In) -> CallResult<In ,R>
    where
        In: std::marker::Tuple + 'static,
        R: 'static,
    {
        if self.id != TypeId::of::<(In, R)>() {
            return CallResult::NotImplemented(args);
        }
        CallResult::Result(unsafe { self.call_unchecked(args) })
    }
}

unsafe fn fastcall<In, R, F>(func: *const (), args: In) -> R
where
    In: std::marker::Tuple + 'static,
    R: 'static,
    F: Fn<In, Output = R> + Sync + Send + 'static
{
    let unerased_func = &*(func as *const F);
    unerased_func.call(args)
}

bitflags! {
    pub struct ResolveFlags: u8 {
        const ATTRIBUTE = 0b100;
        const INSERT    = 0b010;
        const SHADOW    = 0b001;
    }
}

#[inline(always)]
pub fn resolve_by_symbol<'v, T, R>(vm: &VM, name: Symbol, value: T, flags: ResolveFlags) -> R
where
    T: VMCast,
    R: tlang::interop::VMPropertyCast
{
    let value = value.vmcast(vm);

    if flags.contains(ResolveFlags::ATTRIBUTE) {
        if let Some(mut tobject) = value.query_tobject() {
            if tobject.descriptor.is_some() {
                let variable_type = tobject.ty.flags.contains(TypeFlags::VARIABLE);
                let ttype = tobject.ty;
                if let Some(found) = tobject.get_attribute(name, variable_type && flags.contains(ResolveFlags::INSERT)) {
                    let access = if variable_type {
                        if flags.contains(ResolveFlags::SHADOW) {
                            tlang::interop::AccessType::WriteOnly
                        } else {
                            tlang::interop::AccessType::Writeable
                        }
                    } else {
                        tlang::interop::AccessType::ReadOnly
                    };
                    return R::propcast(value, found, access, vm);
                }
            }
        }
    }

    let mut current_ty = value.ttype(vm);
    while let Some(mut current) = current_ty {
        if let Some(val) = current.base.get_attribute(name, false) {
            return R::propcast(value, val, tlang::interop::AccessType::TypeChain, vm)
        }
        current_ty = current.basety.clone();
    }

    panic!("Could not find property {} on {:?}", vm.symbols().get(name), value.ttype(vm).map(|ty| ty.name));
}

pub fn print(module: GCRef<TModule>, args: TVariadicArguments) {
    let vm = module.vm();

    for (idx, msg) in args.iter().enumerate() {
        if idx > 0 {
            print!(", ");
        }

        if msg.encoded() == TValue::null().encoded() {
            print!("null");
            continue;
        }

        let msg_str: GCRef<TString> = tcall!(&vm, TValue::fmt(*msg));
        print!("{msg_str}");
    }
    println!();
}
