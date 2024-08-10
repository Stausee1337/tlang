use std::{mem::transmute, fmt::Display, u64, any::TypeId, cell::OnceCell, process::Output, marker::PhantomData};


use hashbrown::raw::RawTable;

use crate::{memory::{GCRef, Atom}, symbol::Symbol, bytecode::TRawCode, bigint::{TBigint, self, to_bigint}, vm::{VM, TModule}};
use tlang_macros::tmodule;

#[repr(u64)]
#[derive(Debug)]
enum TValueKind {
    Object   = 0b101 << 49,
    Int32    = 0b001 << 49,
    Bool     = 0b010 << 49,
    String   = 0b110 << 49,
    Function = 0b000 << 49,
    Float    = 0b100 << 49,
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

    /// Constructors
    
    #[inline(always)]
    pub const fn null() -> Self {
        let null = unsafe { GCRef::<()>::null() };
        Self::object_tagged(null, TValueKind::Object)
    }

    #[inline(always)]
    pub fn object<T: Typed>(object: GCRef<T>) -> Self {
        debug_assert!(std::ptr::addr_eq(
            &T::ttype(&object.vm()),
            object.kind::<TType>().unwrap()
        ));
        Self::object_tagged(object, TValueKind::Object)
    }

    pub(crate) fn ttype_object(ttype: GCRef<TType>) -> Self {
        Self::object_tagged(ttype, TValueKind::Object)
    }

    #[inline(always)]
    const fn bool(bool: bool) -> Self {
        let bool = bool as u64;
        TValue(Self::FLOAT_NAN_TAG | TValueKind::Bool as u64 | bool)
    }

    #[inline(always)]
    const fn int32(int: i32) -> Self {
        let int = int as u64;
        TValue(Self::FLOAT_NAN_TAG | TValueKind::Int32 as u64 | int)
    }

    #[inline(always)]
    const fn float(float: f64) -> Self {
        return TValue(unsafe { transmute(float) });
    }

    #[inline(always)]
    const fn object_tagged<T>(object: GCRef<T>, kind: TValueKind) -> Self {
        // TODO: use object.as_ptr() instead
        let object: u64 = unsafe { transmute(object) };
        assert!(object & (!Self::NAN_VALUE_MASK) == 0);
        TValue(Self::FLOAT_NAN_TAG | kind as u64 | object)
    }

    #[inline(always)]
    const fn string(string: GCRef<TString>) -> Self {
        Self::object_tagged(string, TValueKind::String)
    }

    /// Public Helpers 

    #[inline(always)]
    pub fn query_object<T: Typed>(&self, vm: &VM) -> Option<GCRef<T>> {
        if let TValueKind::Object = self.kind() {
            let object_type = self.ttype(vm);
            if std::ptr::addr_eq(object_type.as_ptr(), T::ttype(vm).as_ptr()) {
                return Some(self.as_object::<T>());
            }
        }
        None
    }

    #[inline(always)]
    pub fn query_integer(&self, vm: &VM) -> Option<TInteger> {
        match self.kind() {
            TValueKind::Int32 =>
                Some(TInteger(IntegerKind::Int32(self.as_int32()))),
            TValueKind::Object =>
                Some(TInteger(IntegerKind::Bigint(self.query_object::<TBigint>(vm)?))),
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
    pub fn ttype(&self, vm: &VM) -> GCRef<TType> {
        match self.kind() {
            TValueKind::Bool => TBool::ttype(vm),
            TValueKind::Int32 => TInteger::ttype(vm),
            TValueKind::Float => TFloat::ttype(vm),
            _ => {
                let object = self.as_object::<()>();
                let object_type = object.kind::<TType>()
                    .expect("TValue have TType");
                unsafe { GCRef::from_raw(&*object_type) }
            }
        }
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
    const fn as_object<T>(&self) -> GCRef<T> {
        unsafe { GCRef::from_raw((self.0 & Self::NAN_VALUE_MASK) as *mut T) }
    }
}

pub trait Typed: 'static {
    const NAME: &'static str;

    fn ttype(vm: &VM) -> GCRef<TType>;
}

pub trait VMCast {
    fn vmcast(self, vm: &VM) -> TValue;
}

pub trait VMDowncast: Sized {
    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self>;
}

impl VMCast for &str {
    fn vmcast(self, vm: &VM) -> TValue {
        TString::from_slice(vm, self).into()
    }
}

impl<T: Into<TValue>> VMCast for T {
    fn vmcast(self, _vm: &VM) -> TValue {
        self.into()
    }
}

pub trait GetHash {
    fn get_hash_code(&self) -> u64;
}

/// FIXME: better TTypeBuilder 
pub struct TTypeBuilder<'vm> {
    vm: &'vm VM,
    attributes: Vec<(Symbol, TValue)>,
    empty: Option<GCRef<TType>>
}

impl<'vm> TTypeBuilder<'vm> {
    pub fn new(vm: &'vm VM, name: impl VMCast, modname: impl VMCast) -> Self {
        let mut attributes = Vec::new();
        attributes.push((
            vm.symbols().intern_slice("name"),
            name.vmcast(vm)
        ));
        attributes.push((
            vm.symbols().intern_slice("modname"),
            modname.vmcast(vm)
        ));
        Self {
            vm,
            attributes,
            empty: None
        }
    }

    fn build_empty(vm: &'vm VM, name: impl VMCast, modname: impl VMCast, empty: GCRef<TType>) -> Self {
        let mut builder = TTypeBuilder::new(vm, name, modname);
        builder.empty = Some(empty);
        builder
    }

    pub fn insert_attribute(&mut self, name: Symbol, value: TValue) {
        self.attributes.push((name, value));
    }

    pub fn extend<F: Fn(&mut Self)>(&mut self, builder: F) {
        builder(self);
    }

    pub fn build(self) -> GCRef<TType> {
        unsafe {
            if let Some(ttype) = self.empty {
                let mut object = ttype.cast::<TAnonObject>();
                object.fill(&self.attributes);
                return ttype;
            }
            let object = TAnonObject::create(self.vm, &self.attributes);
            object.cast()
        }
    }
}

#[repr(C)]
pub struct TType(TAnonObject);

impl TType {
    fn empty(vm: &VM, count: usize) -> GCRef<Self> {
        unsafe {
            let object = TAnonObject::make(vm, count);
            object.cast()
        }
    }
}

impl GCRef<TType> {
    fn get_static(&self) -> &'static TType {
        unsafe { &*self.as_ptr() }
    }

    pub fn allocate_object<T: Typed>(self, object: T) -> GCRef<T> {
        debug_assert!(std::ptr::addr_eq(self.as_ptr(), T::ttype(&self.vm()).as_ptr()));
        self.heap().allocate_atom(
            self.get_static(),
            object
        )
    }

    pub fn allocate_var_object<T: Typed>(self, object: T, extra_bytes: usize) -> GCRef<T> {
        debug_assert!(std::ptr::addr_eq(self.as_ptr(), T::ttype(&self.vm()).as_ptr()));
        self.heap().allocate_var_atom(
            self.get_static(),
            object,
            extra_bytes
        )
    }
}

impl Atom for TType {
    fn iterate_children(&self, p: *const ()) -> Box<dyn Iterator<Item = *const ()>> {
        todo!()
    }
}

pub struct TypeCollector;

impl Atom for TypeCollector {
    fn iterate_children(&self, p: *const ()) -> Box<dyn Iterator<Item = *const ()>> {
        todo!()
    }
}

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

impl Typed for TFloat {
    const NAME: &'static str = "float";

    fn ttype(vm: &VM) -> GCRef<TType> {
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

impl Typed for TBool {
    const NAME: &'static str = "bool";

    fn ttype(vm: &VM) -> GCRef<TType> {
        todo!()
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

use prelude::IntegerKind;
pub use prelude::{TFunction, TFnKind, TInteger, TString};

#[tmodule("prelude")]
pub mod prelude {
    use super::*;

    #[primitive("string")]
    pub struct TString {
        size: TInteger,
        length: TInteger,
        data: [u8; 0]
    }

    impl TString {
        pub fn as_slice<'a>(&self) -> &'a str {
            let size = self.size.as_usize().expect("TString sensible size");
            unsafe {
                let bytes = std::slice::from_raw_parts(self.data.as_ptr(), size);
                let str = std::str::from_utf8_unchecked(bytes);
                str
            }
        }

        pub fn from_slice(vm: &VM, slice: &str) -> GCRef<Self> {
            Self::from_slice_impl(vm, slice, TString::ttype(vm).get_static())
        }

        pub(crate) fn from_slice_impl<A: Atom>(vm: &VM, slice: &str, atom: &'static A) -> GCRef<Self> {
            let size = TInteger::from_usize(slice.len());

            let length = TInteger::from_usize(slice.chars().count());

            let mut string = vm.heap().allocate_var_atom(
                atom,
                Self { size, length, data: [0u8; 0] },
                slice.len()
            );

            unsafe {
                std::ptr::copy_nonoverlapping(slice.as_ptr(), string.data.as_mut_ptr(), slice.len());
            }

            string
        }

        pub fn deepcopy(&self, vm: &VM) -> GCRef<TString> {
            Self::from_slice(vm, self.as_slice())
        }
    }

    impl Display for TString {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.as_slice())
        }
    }

    impl std::fmt::Debug for TString {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.as_slice())
        }
    }

    impl Into<TValue> for GCRef<TString> {
        fn into(self) -> TValue {
            TValue::string(self)
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

    #[primitive("int")]
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
            value.query_integer(vm)
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
                IntegerKind::Bigint(bigint) =>
                    TValue::object_tagged(bigint, TValueKind::Object)
            }
        }
    }

    #[primitive("function")]
    #[repr(C)] 
    pub struct TFunction {
        pub name: TValue, /// Optional<TString>
        pub module: GCRef<TModule>,
        pub kind: TFnKind
    }

    pub enum TFnKind {
        Function(TRawCode),
    }

    impl GCRef<TFunction> {
        #[inline(always)]
        pub fn call<'a>(&self, arguments: &'a mut [TValue]) -> TValue {
            match &self.kind {
                TFnKind::Function(code) => {
                    // TODO: check len(arguments) == len(params)
                    code.evaluate(&self.vm(), arguments)
                }
            }
        }
    }

    impl Into<TValue> for GCRef<TFunction> {
        #[inline(always)]
        fn into(self) -> TValue {
            TValue::object_tagged(self, TValueKind::Function)
        }
    }
}


#[repr(C)]
struct TAnonObject {
    descriptor: RawTable<(Symbol, u64, usize)>,
    data: [u8; 0]
}

impl TAnonObject {
    fn create(vm: &VM, attributes: &[(Symbol, TValue)]) -> GCRef<Self> {
        unsafe {
            let mut object = Self::make(vm, attributes.len());
            object.fill(attributes);
            object
        }
    }

    unsafe fn make(vm: &VM, count: usize) -> GCRef<Self> {
        let object = TAnonObject {
            descriptor: RawTable::with_capacity(count),
            data: [0u8; 0]
        };
        vm.heap().allocate_var_atom(
            &TypeCollector,
            object,
            count * std::mem::size_of::<TValue>())
    }
}

impl GCRef<TAnonObject> {
    unsafe fn fill(&mut self, attributes: &[(Symbol, TValue)]) {
        debug_assert!(self.descriptor.capacity() >= attributes.len());
        for (idx, (name, value)) in attributes.iter().enumerate() {
            if let Some(bucket) = self.query_attribute(*name) {
                *bucket = *value;
                continue;
            }
            self.insert_attribute(idx, *name, *value);
        }
    }

    unsafe fn insert_attribute(&mut self, idx: usize, attribute: Symbol, value: TValue) {
        let hash = self.vm().symbols.hash(attribute);
        self.descriptor.insert(
            hash, (attribute, hash, idx),
            |val| val.1
        );
        *(self.data.as_mut_ptr() as *mut TValue).add(idx) = value;
    }

    unsafe fn query_attribute(&mut self, attribute: Symbol) -> Option<&mut TValue> { 
        let hash = self.vm().symbols.hash(attribute);
        if let Some(&(_, _, idx)) = self.descriptor.get(hash, |val| val.0 == attribute) { 
            let ptr = (self.data.as_mut_ptr() as *mut TValue).add(idx);
            return Some(&mut *ptr)
        }
        None
    }
}

// --- TType's ---
// TBool
// TInt
// TFloat
// TString
// TFunction
// TObject
// TType (TType as a subtype of TObject makes sense)

