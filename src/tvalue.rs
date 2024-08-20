use std::{mem::{transmute, MaybeUninit, offset_of}, fmt::Display, alloc::Layout, any::TypeId, hash::BuildHasherDefault, io::Write};


use ahash::AHasher;
use allocator_api2::alloc::Global;
use hashbrown::{hash_map::RawVacantEntryMut, HashTable, hash_table::Entry};
use tlang_macros::tcall;

use crate::{memory::{GCRef, Atom, Visitor}, symbol::Symbol, bytecode::TRawCode, bigint::{TBigint, self, to_bigint}, vm::{VM, TModule}, eval::TArgsBuffer, interop::{TPolymorphicObject, VMDowncast, TPolymorphicWrapper, VMArgs, VMCast, TPolymorphicCallable, TPropertyAccess}, debug};


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

    fn base_with_descriptor(
        vm: &VM, ty: GCRef<TType>, descriptor: Option<GCRef<HashTable<(Symbol, TValue)>>>) -> Self {
        Self {
            ty, descriptor,
        }
    }

    pub fn get_attribute(&mut self, name: Symbol) -> Option<&mut TValue> {
        let Some(descriptor) = &mut self.descriptor else {
            debug_assert!(!self.ty.variable);
            panic!("cannot get attribute on fixed type");
        };
        if let Some(val) = descriptor.find_mut(name.hash, |val| val.0 == name) {
            return Some(&mut val.1);
        }
        None
    }

    pub fn set_attribute(&mut self, name: Symbol, value: TValue) {
        let Some(descriptor) = &mut self.descriptor else {
            debug_assert!(!self.ty.variable);
            panic!("cannot set attribute on fixed type {:p} {:p}", std::ptr::addr_of!(self.ty), self.ty.as_ptr());
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
            variable: true
        });
        entry.insert(TypeId::of::<Self>(), ttype);

        ttype.base = TObject::base(vm, vm.types().query::<TType>());

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![Object], ttype.into(), true);

        ttype.define_method(Symbol![eq], TFunction::rustfunc(
                prelude, Some("object::eq"), |this: TValue, other: TValue| {
                    TBool::from_bool(this.encoded() == other.encoded())
                }));

        ttype.define_method(Symbol![ne], TFunction::rustfunc(
                prelude, Some("object::ne"), move |this: TValue, other: TValue| {
                    let vm = ttype.vm();
                    let eq: TBool = tcall!(&vm, TValue::eq(this, other));
                    !eq
                }));

        ttype.define_method(Symbol![toString], TFunction::rustfunc(
                prelude, Some("object::toString"), move |this: TValue| {
                    let vm = ttype.vm();
                    TString::from_slice(&vm, &format!("{} {{}}", this.ttype(&vm).unwrap().name))
                }));


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

pub fn bool_init_unarys(mut ttype: GCRef<TType>) {
    let vm = ttype.vm();

    let prelude = vm.modules().prelude();

    ttype.define_method(Symbol![not], TFunction::rustfunc(
            prelude, Some("bool::not"),
            <TBool as std::ops::Not>::not));
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
enum IntegerKind {
    Int32(i32),
    Bigint(GCRef<TBigint>),
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TInteger(IntegerKind);

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

pub fn int_init_unarys(mut ttype: GCRef<TType>) {
    let vm = ttype.vm();

    let prelude = vm.modules().prelude();

    ttype.define_method(Symbol![neg], TFunction::rustfunc(
            prelude, Some("int::neg"),
            <TInteger as std::ops::Neg>::neg));

    ttype.define_method(Symbol![invert], TFunction::rustfunc(
            prelude, Some("int::invert"),
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

        pub(crate) fn int_init_arithmetics(mut ttype: GCRef<TType>) {
            let vm = ttype.vm();
            let prelude = vm.modules().prelude();
            $(
                ttype.define_method(Symbol![$fn], TFunction::rustfunc(
                        prelude, Some(concat!("int::", stringify!($op))),
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

        pub(crate) fn int_init_cmps(mut ttype: GCRef<TType>) {
            let vm = ttype.vm();
            let prelude = vm.modules().prelude();
            $(
                ttype.define_method(Symbol![$fn], TFunction::rustfunc(
                        prelude, Some(concat!("int::", stringify!($op))),
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

pub fn float_init_unarys(mut ttype: GCRef<TType>) {
    let vm = ttype.vm();

    ttype.define_method(Symbol![neg], TFunction::rustfunc(
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

        pub(crate) fn float_init_arithmetics(mut ttype: GCRef<TType>) {
            let vm = ttype.vm();
            let prelude = vm.modules().prelude();
            $(
                ttype.define_method(Symbol![$fn], TFunction::rustfunc(
                        prelude, Some(concat!("float::", stringify!($op))),
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

        pub(crate) fn float_init_cmps(mut ttype: GCRef<TType>) {
            let vm = ttype.vm();
            let mut prelude = vm.modules().prelude();
            $(
                ttype.define_method(Symbol![$fn], TFunction::rustfunc(
                        prelude, Some(concat!("float::", stringify!($op))),
                        <TFloat as $crate::interop::vmops::$op<TValue>>::$fn));
            )*
        }
    };
}

iter_float_cmps! {
    impl Lt for TFloat in lt; impl Le for TFloat in le;
    impl Gt for TFloat in gt; impl Ge for TFloat in ge;
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

impl Atom for i32 {
    fn visit(&self, visitor: &mut Visitor) {
        todo!()
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
            name: TString::from_slice(vm, "type"),
            modname: TString::from_slice(vm, "prelude"),
            variable: true
        });

        let int = vm.heap().allocate_atom(12);
        entry.insert(TypeId::of::<Self>(), ttype);

        ttype.base = TObject::base(vm, ttype);
        ttype.basety = Some(vm.types().query::<TObject>());

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![Type], ttype.into(), true);

        ttype.define_property(
            Symbol![basety],
            TProperty::offset::<TType, Option<GCRef<TType>>>(prelude, Accessor::GET, offset_of!(TType, basety)));
        ttype.define_property(
            Symbol![name],
            TProperty::offset::<TType, GCRef<TString>>(prelude, Accessor::GET, offset_of!(TType, name)));
        ttype.define_property(
            Symbol![modname],
            TProperty::offset::<TType, GCRef<TString>>(prelude, Accessor::GET, offset_of!(TType, modname)));
        ttype.define_method(
            Symbol![toString],
            TFunction::rustfunc(prelude, Some("type::toString"), |this: GCRef<TType>| {
                TString::from_slice(&this.vm(), &format!("[type {}] {{}}", this.name))
            }));

        ttype
        
    }

    fn initialize(vm: &VM) -> GCRef<TType> {
        unreachable!()
    }
}

bitflags::bitflags! {
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

        let prelude = vm.modules().prelude();

        ttype.define_property(
            Symbol![get],
            TProperty::offset::<Self, Option<GCRef<TFunction>>>(prelude, Accessor::GET, offset_of!(TProperty, get)));

        ttype.define_property(
            Symbol![set],
            TProperty::offset::<Self, Option<GCRef<TFunction>>>(prelude, Accessor::GET, offset_of!(TProperty, set)));

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

#[repr(C)]
pub enum TFnKind {
    Function(TRawCode),
    Nativefunc(Nativefunc)
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
            name: TString::from_slice(vm, "function"),
            modname: TString::from_slice(vm, "prelude"),
            variable: false
        });
        entry.insert(TypeId::of::<Self>(), ttype);

        let mut prelude = vm.modules().prelude();
        prelude.set_global(Symbol![Function], ttype.into(), true);

        ttype.define_property(
            Symbol![name],
            TProperty::offset::<Self, Option<GCRef<TString>>>(prelude, Accessor::GET, offset_of!(Self, name)));

        ttype.define_method(
            Symbol![toString],
            TFunction::rustfunc(prelude, Some("function::toString"), |this: GCRef<TFunction>| {
                let name = if let Some(name) = this.name {
                    name.as_slice()
                } else {
                    "(anonymous)"
                };
                let mut string = format!("def {name} ");
                if let TFnKind::Nativefunc(..) = this.kind {
                    string.push_str("[Native Code]");
                } else {
                    string.push_str("{}");
                }
                TString::from_slice(&this.vm(), &string)
            }));

        ttype
    }

    fn initialize(vm: &VM) -> GCRef<TType> {
        unreachable!()
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
            TFnKind::Nativefunc(n) => n.call(args)
        }
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
struct Nativefunc {
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

#[inline(always)]
pub fn resolve_by_symbol<'v, T, R>(vm: &VM, name: Symbol, value: T, resolve_to_attribute: bool) -> R
where
    T: VMCast,
    R: tlang::interop::VMPropertyCast
{
    let value = value.vmcast(vm);
    if resolve_to_attribute {
        if let Some(mut tobject) = value.query_tobject() {
            if tobject.ty.variable {
                if let Some(found) = tobject.get_attribute(name) {
                    return R::propcast(value, found, true, vm);
                }
            }
        }
    }
    
    let mut mut_current = value.ttype(vm);

    while let Some(mut current) = mut_current {
        if let Some(val) = current.base.get_attribute(name) {
            return R::propcast(value, val, false, vm)
        }
        mut_current = current.basety.clone();
    }
    
    panic!("Could not find property");
}

pub fn print(module: GCRef<TModule>, msg: TValue) {
    let vm = module.vm();
    if msg.encoded() == TValue::null().encoded() {
        println!("null");
        return;
    }

    let msg_str: GCRef<TString> = tcall!(&vm, TValue::toString(msg));

    println!("{msg_str}");
}
