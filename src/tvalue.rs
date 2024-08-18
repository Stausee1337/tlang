use std::{mem::{transmute, MaybeUninit, offset_of, ManuallyDrop}, fmt::Display, alloc::Layout, ptr::NonNull, any::TypeId, hash::BuildHasherDefault, io::Write};


use ahash::AHasher;
use allocator_api2::alloc::Global;
use hashbrown::{hash_map::RawVacantEntryMut, HashTable, hash_table::Entry};

use crate::{memory::{GCRef, Atom, Visitor, self}, symbol::Symbol, bytecode::TRawCode, bigint::{TBigint, self, to_bigint}, vm::{VM, TModule}, eval::TArgsBuffer, interop::{TPolymorphicObject, VMDowncast, TPolymorphicWrapper, VMArgs, VMCast, TPolymorphicCallable}};


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
enum TValueKind {
    Object   = 0b101 << 49,
    Int32    = 0b001 << 49,
    Bool     = 0b010 << 49,
    Float    = 0b100 << 49,
    String   = 0b000 << 49,
    BigInt   = 0b110 << 49,
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

    pub fn encoded(&self) -> u64 {
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
            if !self.ttype(vm)?.refrence_eq(ty) {
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

    /// Private Helpers

    #[inline(always)]
    fn kind(&self) -> TValueKind {
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
        // This will still yield None, if the GCRef<..>,
        // which contains a NonNull, actually is null
        // So `TValue::null()` -> None
        Some(self.as_gcref())
    }
}

pub trait Typed: Atom + 'static {
    fn initialize_entry(
        vm: &VM,
        entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
    ) -> GCRef<TType> {
        *entry.insert(TypeId::of::<Self>(), Self::initialize(vm)).1
    }

    fn initialize(vm: &VM) -> GCRef<TType>;

    fn visit_override(&self, visitor: &mut Visitor) {
        todo!()
    }
}

impl<T: Typed> Atom for T {
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
    pub fn new(vm: &VM) -> GCRef<Self> {
        let descriptor = vm.heap().allocate_atom(HashTable::new());
        vm.heap().allocate_atom(Self {
            ty: vm.types().query::<Self>(),
            descriptor: Some(descriptor),
        })
    }

    pub fn base(vm: &VM, ty: GCRef<TType>) -> Self {
        let descriptor = if ty.variable {
            Some(vm.heap().allocate_atom(HashTable::new()))
        } else {
            None
        };
        Self {
            ty, descriptor,
        }
    }

    pub fn get_attribute(&self, name: Symbol, value: TValue) -> Option<TValue> {
        if !self.ty.variable {
            panic!("cannot set attribute on fixed type");
        }
        let Some(descriptor) = &self.descriptor else {
            unreachable!()
        };
        descriptor
            .find(name.hash, |val| val.0 == name)
            .map(|val| val.1)
    }

    pub fn set_attribute(&mut self, name: Symbol, value: TValue) {
        if !self.ty.variable {
            panic!("cannot set attribute on fixed type");
        }
        let Some(descriptor) = &mut self.descriptor else {
            unreachable!()
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

    fn uninit() -> Self {
        unsafe { MaybeUninit::uninit().assume_init() }
    }
}

impl Typed for TObject {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: Self::uninit(),
            basesize: std::mem::size_of::<Self>(),
            basety: None, // There's nothing above Object in the hierarchy
            name: TString::from_slice(vm, "object"),
            modname: TString::from_slice(vm, "prelude"),
            variable: true
        });
        entry.insert(TypeId::of::<Self>(), ttype);
        ttype.base = TObject::base(vm, ttype);

        ttype.define_method(Symbol![toString], TFunction::rustfunc(
                vm.modules().empty(), Some("object::toString"), |_value: TValue| "object {}"));

        ttype
    }

    fn initialize(vm: &VM) -> GCRef<TType> {
        unreachable!()
    }
}

impl Atom for HashTable<(Symbol, TValue)> {
    fn visit(&self, visitor: &mut Visitor) {
        todo!()
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

#[repr(C)]
#[repr(align(8))]
pub struct TString {
    data: StringData,
    length: TInteger,

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
    pub(crate) fn from_slice(vm: &VM, slice: &str) -> GCRef<Self> {
        let size = slice.len();
        let length = TInteger::from_usize(slice.chars().count());

        unsafe {
            let mut string = vm.heap().allocate_var_atom(
                Self {
                    data: StringData { ptr: std::ptr::null() },
                    length,
                    prev: None, next: None,
                    size
                },
                slice.len()
            );

            let mut string = if size < std::mem::size_of::<&()>() {
                GCRef::from_raw(
                    std::ptr::addr_of!(string.data.small[1]) as *const TString
                )
            } else {
                let align = std::mem::align_of::<u8>();
                string.data.ptr = std::alloc::alloc(Layout::from_size_align_unchecked(size, align)) as *const u8;
                string
            };

            std::ptr::copy_nonoverlapping(slice.as_ptr(), string.data_ptr_mut(), size);

            string
        }
    }
}

pub trait GetHash {
    fn get_hash_code(&self) -> u64;
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
        self.normalize().size
    }

    #[inline]
    pub fn length(&self) -> TInteger {
        self.normalize().length
    }

    #[inline]
    fn is_small(&self) -> bool {
        let ptr: *const _ = self;
        (ptr as usize & 0b1) == 1
    }

    #[inline]
    fn normalize(&self) -> &Self {
        let ptr: *const _ = self;
        unsafe {
            &*((ptr as usize & !0b1) as *const TString)
        }
    }

    #[inline]
    unsafe fn data_ptr(&self) -> *const u8 {
        if self.is_small() {
            return std::ptr::addr_of!(self.data.small[0]);
        }
        self.data.ptr
    }

    #[inline]
    unsafe fn data_ptr_mut(&mut self) -> *mut u8 {
        if self.is_small() {
            return std::ptr::addr_of_mut!(self.data.small[0]);
        }
        self.data.ptr as *mut u8
    }
}

impl Atom for TString {
    fn visit(&self, visitor: &mut Visitor) {
        todo!()
    }
}

impl Drop for TString {
    fn drop(&mut self) {
        if self.is_small() {
            unsafe {
                let align = std::mem::align_of::<u8>();
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

impl GetHash for GCRef<TString> {
    fn get_hash_code(&self) -> u64 {
        self.vm().hash_state.hash_one(self.as_slice())
    }
}

#[derive(Clone, Copy)]
pub(super) enum IntegerKind {
    Int32(i32),
    Bigint(GCRef<TBigint>),
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TInteger(pub(super) IntegerKind);

impl TInteger {
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

    pub fn from_signed_bytes(bytes: &[u8]) -> Self {
        todo!("real bigint support")
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
        use std::ops::$op;
        impl $op for $ty {
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
    };
}

macro_rules! iter_int_arithmetics {
    ($(impl $op:ident for $ty:ident in ($fn:ident$(, $checked_fn:ident)?);)*) => {
        $(impl_int_arithmetic!($op, $ty, $fn, $($checked_fn)?);)*
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


impl Into<TValue> for TInteger {
    #[inline(always)]
    fn into(self) -> TValue {
        match self.0 {
            IntegerKind::Int32(int) => TValue::int32(int),
            IntegerKind::Bigint(bigint) => todo!()
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TFloat(pub(super) f64);


impl TFloat {
    pub const fn as_float(self) -> f64 {
        self.0
    }

    pub const fn from_float(float: f64) -> Self {
        TFloat(float)
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


macro_rules! impl_float_arithmetic {
    ($op:ident, $ty:ident, $fn:ident) => { 
        impl $op for $ty {
            type Output = $ty;

            #[inline(always)]
            fn $fn(self, rhs: Self) -> Self::Output {
                Self(self.0.$fn(rhs.0))
            }
        }
    };
}

macro_rules! iter_float_arithmetics {
    ($(impl $op:ident for $ty:ident in $fn:ident;)*) => {
        $(impl_float_arithmetic!($op, $ty, $fn);)*
    };
}

iter_float_arithmetics! {
    impl Add for TFloat in add;
    impl Sub for TFloat in sub;
    impl Mul for TFloat in mul;
    impl Div for TFloat in div;
    impl Rem for TFloat in rem;
}

tobject! {
pub struct TType {
    pub basety: Option<GCRef<TType>>,
    pub basesize: usize,
    pub name: GCRef<TString>,
    pub modname: GCRef<TString>,
    pub variable: bool
}
}

impl GCRef<TType> {
    pub fn is_subclass(&self, needle: GCRef<TType>) -> bool {
        let mut current = *self;
        loop {
            if current.refrence_eq(needle) {
                return true;
            }
            let Some(base) = self.basety else {
                break;
            };
            current = base;
        }
        false
    }

    pub fn define_property(&mut self, name: Symbol, property: GCRef<TProperty>) {
        self.base.set_attribute(name, property.into());
    }

    pub fn define_method(&mut self, name: Symbol, method: GCRef<TFunction>) {
        // TODO: make method a method
        self.base.set_attribute(name, method.into());
    }
}

impl Typed for TType {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> {
        let mut ttype = vm.heap().allocate_atom(TType {
            base: TObject::uninit(),
            basety: None,
            basesize: std::mem::size_of::<Self>(),
            name: TString::from_slice(vm, "type"),
            modname: TString::from_slice(vm, "prelude"),
            variable: true
        });
        entry.insert(TypeId::of::<Self>(), ttype);

        ttype.basety = Some(vm.types().query::<TObject>());
        ttype.base = TObject::base(vm, ttype);

        ttype.define_property(
            Symbol![basety],
            TProperty::offset::<TType, Option<GCRef<TType>>>(&vm, Accessor::GET, offset_of!(TType, basety)));
        ttype.define_property(
            Symbol![name],
            TProperty::offset::<TType, GCRef<TString>>(&vm, Accessor::GET, offset_of!(TType, name)));
        ttype.define_property(
            Symbol![modname],
            TProperty::offset::<TType, GCRef<TString>>(&vm, Accessor::GET, offset_of!(TType, modname)));

        ttype
        
    }

    fn initialize(vm: &VM) -> GCRef<TType> {
        unreachable!()
    }
}

bitflags::bitflags! {
    pub struct Accessor: u8 {
        const GET = 0b1;
        const SET = 0b1;
    }
}

tobject! {
pub struct TProperty {
    pub get: Option<GCRef<TFunction>>,
    pub set: Option<GCRef<TFunction>>
}
}

impl TProperty {
    pub fn offset<Slf, P>(vm: &VM, accessor: Accessor, offset: usize) -> GCRef<Self>
    where
        Slf: TPolymorphicObject,
        P: VMCast + VMDowncast + Copy + 'static
    {
        let get = if accessor.contains(Accessor::GET) {
            Some(TFunction::rustfunc(vm.modules().empty(), None, move |object: TPolymorphicWrapper<Slf>| {
                unsafe { *object.raw_access::<P>(offset) }
            }))
        } else {
            None
        };
        let set = if accessor.contains(Accessor::SET) {
            Some(TFunction::rustfunc(vm.modules().empty(), None, move |object: TPolymorphicWrapper<Slf>, value: P| {
                unsafe { *object.raw_access::<P>(offset) = value; }
            }))
        } else {
            None
        };

        vm.heap().allocate_atom(Self {
            base: TObject::base(vm, vm.types().query::<Self>()),
            get, set
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
            name: TString::from_slice(vm, "property"),
            modname: TString::from_slice(vm, "prelude"),
            variable: false
        });
        entry.insert(TypeId::of::<Self>(), ttype);

        ttype.define_property(
            Symbol![get],
            TProperty::offset::<Self, Option<GCRef<TFunction>>>(vm, Accessor::GET, offset_of!(TProperty, get)));

        ttype.define_property(
            Symbol![set],
            TProperty::offset::<Self, Option<GCRef<TFunction>>>(vm, Accessor::GET, offset_of!(TProperty, set)));

        ttype
    }

    fn initialize(vm: &VM) -> GCRef<TType> {
        unreachable!()
    }
}

tobject! {
pub struct TFunction {
    pub name: Option<GCRef<TString>>,
    pub module: GCRef<TModule>,
    pub kind: TFnKind
}
}

impl Typed for TFunction {
    fn initialize_entry(
            vm: &VM,
            entry: RawVacantEntryMut<'_, TypeId, GCRef<TType>, BuildHasherDefault<AHasher>, Global>
        ) -> GCRef<TType> { 
        let ttype = vm.heap().allocate_atom(TType {
            base: TObject::base(vm, vm.types().query::<TType>()),
            basety: None,
            basesize: std::mem::size_of::<Self>(),
            name: TString::from_slice(vm, "function"),
            modname: TString::from_slice(vm, "prelude"),
            variable: false
        });
        entry.insert(TypeId::of::<Self>(), ttype);

        ttype
    }
    fn initialize(vm: &VM) -> GCRef<TType> {
        unreachable!()
    }
}

#[repr(C)]
struct Nativefunc {
    id: TypeId,
    closure: *const (),
    fastcall: *const (),
    traitfn: fn(&Nativefunc, GCRef<TModule>, TArgsBuffer) -> TValue,
}

impl Nativefunc {
    unsafe fn call_unchecked<In, R>(&self, args: In) -> R
    where
        In: VMArgs + 'static,
        R: VMCast + 'static,
    {
        let fn_ptr: unsafe fn(*const(), In) -> R = transmute(self.fastcall);
        fn_ptr(self.closure, args)
    }
}


#[repr(C)]
pub enum TFnKind {
    Function(TRawCode),
    Nativefunc(Nativefunc)
}

unsafe impl std::marker::Sync for TFnKind {}
unsafe impl std::marker::Send for TFnKind {}

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

        /*let ptr: unsafe fn(args: In) -> R = |args| {
        };*/

        // let fastcall = Fastcall::new(func);

        let mut func = vm.heap().allocate_atom(Self {
            base: TObject::base(&vm, vm.types().query::<Self>()),
            name: name.map(|name| TString::from_slice(&vm, name)),
            module,
            kind: TFnKind::Nativefunc(Nativefunc {
                id,
                closure: closure as *const (),
                fastcall: fastcall::<In, R, F> as *const (),
                traitfn: traitfn::<In, R>
            })
        });

        func
    }
}

unsafe fn fastcall<In, R, F>(func: *const (), args: In) -> R
where
    In: VMArgs + 'static,
    R: VMCast + 'static,
    F: Fn<In, Output = R> + Sync + Send + 'static
{
    let unerased_func = &*(func as *const F);
    unerased_func.call(args)
}

impl GCRef<TFunction> {
    #[inline(always)]
    pub fn call(&self, arguments: TArgsBuffer) -> TValue {
        match &self.kind {
            TFnKind::Function(code) =>
                code.evaluate(self.module, arguments),
            TFnKind::Nativefunc(n @ Nativefunc { traitfn, .. })=>
                traitfn(n, self.module, arguments)
        }
    }

    #[inline(always)]
    pub fn fastcall<In, R>(&self, args: In) -> CallResult<In, R>
    where
        In: std::marker::Tuple + 'static,
        R: 'static
    {
        match &self.kind {
            TFnKind::Function(..) => CallResult::NotImplemented(args),
            TFnKind::Nativefunc(..) => todo!()
        }
    }
}

pub enum CallResult<In, R> {
    NotImplemented(In),
    Result(R)
}

#[repr(C)]
struct Fastcall<F> {
    id: TypeId,
    fast_drop: unsafe fn(NonNull<Fastcall<()>>),
    fast_call: *const (),
    func: F
}

impl<F> Fastcall<F> {
    pub fn new<In, R>(func: F) -> Fastcall<F>
    where
        In: std::marker::Tuple + 'static,
        R: 'static,
        F: FnOnce<In, Output = R> + Copy
    {
        let id = TypeId::of::<(In, R)>();
        Fastcall {
            id,
            fast_drop: fast_drop::<In, R, F>,
            fast_call: fast_call::<In, R, F> as *const (),
            func
        }
    }
}

impl Fastcall<()> {
    #[inline(always)]
    fn try_call<In, R>(&self, args: In) -> CallResult<In, R> 
    where
        In: std::marker::Tuple + 'static,
        R: 'static,
    {
        let id = TypeId::of::<(In, R)>();
        if self.id != id {
            return CallResult::NotImplemented(args);
        }
        unsafe {
            let fast_call: unsafe fn(&Fastcall<()>, In) -> R = transmute(self.fast_call);
            CallResult::Result(fast_call(self, args))
        }
    }
}

impl<F> Drop for Fastcall<F> {
    fn drop(&mut self) {
        unsafe {
            let ereased_ref: &mut Fastcall<()> = memory::mutcast(self);
            let erased_owned = ereased_ref.into();
            (self.fast_drop)(erased_owned);
        }
    }
}

unsafe fn fast_drop<In, R, F>(mut fastcall: NonNull<Fastcall<()>>)
where
    In: std::marker::Tuple,
    F: FnOnce<In, Output = R> + Copy
{
    let unerased_type: &mut Fastcall<F> = memory::mutcast(fastcall.as_mut());
    std::ptr::drop_in_place(unerased_type);
}

unsafe fn fast_call<In, R, F>(fastcall: &Fastcall<()>, args: In) -> R
where
    In: std::marker::Tuple,
    F: FnOnce<In, Output = R> + Copy
{
    let unerased_type: &Fastcall<F> = memory::refcast(fastcall);
    unerased_type.func.call_once(args)
}

#[repr(C)]
struct FnOnceTrait<F = ()> {
    vtable: &'static FnOnceTable,
    func: F,
}

#[repr(C)]
struct FnOnceTable {
    func_drop: unsafe fn(NonNull<FnOnceTrait>),
    func_call: unsafe fn(&'_ FnOnceTrait, GCRef<TModule>, TArgsBuffer) -> TValue,
}

impl<F: FnOnce(GCRef<TModule>, TArgsBuffer) -> TValue + Copy> FnOnceTrait<F> {
    fn new(func: F) -> Self {
        let vtable = &FnOnceTable {
            func_drop: func_drop::<F>,
            func_call: func_call::<F>
        };
        Self {
            vtable,
            func,
        }
    }
}

impl FnOnceTrait {
    fn call(&self, module: GCRef<TModule>, args: TArgsBuffer) -> TValue {
        unsafe {
            (self.vtable.func_call)(self, module, args)
        }
    }
}

impl<T> Drop for FnOnceTrait<T> {
    fn drop(&mut self) {
        unsafe {
            let erased_ref: &mut FnOnceTrait<()> = memory::mutcast(self);
            let erased_owned = erased_ref.into();
            (self.vtable.func_drop)(erased_owned)
        }
    }
}

unsafe fn func_drop<F>(mut f: NonNull<FnOnceTrait>)
where
    F: FnOnce(GCRef<TModule>, TArgsBuffer) -> TValue + Copy
{
    let unerased_type: &mut FnOnceTrait<F> = memory::mutcast(f.as_mut());
    std::ptr::drop_in_place(unerased_type);
}

unsafe fn func_call<F>(f: &'_ FnOnceTrait, module: GCRef<TModule>, args: TArgsBuffer) -> TValue
where
    F: FnOnce(GCRef<TModule>, TArgsBuffer) -> TValue + Copy
{
    let unerased_type: &'_ FnOnceTrait<F> = memory::refcast(f);
    (unerased_type.func)(module, args)
}

#[inline(always)]
pub fn resolve_by_symbol<T, R>(vm: &VM, name: Symbol, value: T) -> R
where
    T: VMCast,
    R: VMDowncast
{
    let value = value.vmcast(vm);
    if let Some(tobject) = value.query_tobject() {
        if let Some(found) = tobject.get_attribute(name, value) {
            return R::vmdowncast(found, vm).unwrap();
        }
    }
    
    let mut found = None;
    let mut mut_current = value.ttype(vm);

    while let Some(current) = mut_current {
        if let Some(val) = current.base.get_attribute(name, value) {
            found = Some(val);
            break;
        }
        mut_current = current.basety;
    }

    R::vmdowncast(found.unwrap(), vm).unwrap()
}

pub fn print(module: GCRef<TModule>, msg: TValue) {
    let vm = module.vm();
    println!("print({:?})", msg.kind());
    // let msg: GCRef<TString> = tcall!(&vm, TValue::toString(msg));
    let msg: GCRef<TString> = {
        let resolved_func: TPolymorphicCallable<_, _> = resolve_by_symbol(&vm, Symbol![toString], msg);
        resolved_func(msg)
    };

    println!("{msg}");
}

pub fn module_init(mut module: GCRef<TModule>) {
    // let print_function = TFunction::nativefunc(module, Some("print"), print_impl);
    // module.set_global(module.vm().symbols().intern_slice("print"), print_function.into(), true).unwrap();
    todo!()
}


// #[repr(C)]

// --- TType's ---
// TBool
// TInt
// TFloat
// TString
// TFunction
// TObject
// TType (TType as a subtype of TObject makes sense)

