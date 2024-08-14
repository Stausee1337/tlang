use std::{mem::transmute, fmt::Display, alloc::Layout};


use hashbrown::raw::RawTable;

use crate::{memory::{GCRef, Atom, Visitor}, symbol::Symbol, bytecode::TRawCode, bigint::{TBigint, self, to_bigint}, vm::{VM, TModule}, eval::TArgsBuffer};

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

/// This is a _marker_ trait for polymorphic Objects use `tobject!` to implement
///
/// Marking a struct means it obeys 2 main criteria
///     * the first attribute is a struct implementig `TPolymorphicObject`
///     * the struct layout is `#[repr(C)]`
/// This ensures that the first N attributes are the same as those in
/// `TObject`, furthermore it allows any base types to directly extend the
/// inheritance hirachy for decendent objects.
/// Most importantly it allows for our type queries, as the first field now
/// always points to a TType
unsafe trait TPolymorphicObject: Typed {
    type Base: TPolymorphicObject;
}

#[repr(C)]
pub struct TObject {
    pub ty: GCRef<TType>
}

impl Typed for TObject {
    const NAME: &'static str = "object";
}

unsafe impl TPolymorphicObject for TObject {
    type Base = TObject;
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
        Self::object_impl(null, TValueKind::Object)
    }

    #[inline(always)]
    pub const fn object<T: TPolymorphicObject>(object: GCRef<T>) -> Self {
        /*debug_assert!(std::ptr::addr_eq(
            T::ttype(&object.vm()).as_ptr(),
            object.kind::<TType>().unwrap()
        ));*/
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
    const fn int32(int: i32) -> Self {
        let int = int as u64;
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
            vm.types().query::<T>();
            todo!()
        }
        None
    }

    #[inline(always)]
    pub fn query_string(&self) -> Option<GCRef<TString>> {
        if let TValueKind::String = self.kind() {
            return Some(self.as_object());
        }
        None
    }

    #[inline(always)]
    pub fn query_integer(&self) -> Option<TInteger> {
        match self.kind() {
            TValueKind::Int32 =>
                Some(TInteger(IntegerKind::Int32(self.as_int32()))),
            TValueKind::BigInt =>
                Some(TInteger(IntegerKind::Bigint(self.as_object()))),
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
            TValueKind::Bool => vm.primitives().bool_type(),
            TValueKind::Int32 => vm.primitives().int_type(),
            TValueKind::Float => vm.primitives().float_type(),
            _ => {
                todo!()
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

pub trait Typed: Atom + 'static {
    const NAME: &'static str;

    fn visit_override(&self, visitor: &mut Visitor) {
        todo!()
    }
}

impl<T: Typed> Atom for T {
    fn visit(&self, visitor: &mut Visitor) {
        T::visit_override(self, visitor);
    }
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

impl VMCast for () {
    fn vmcast(self, _vm: &VM) -> TValue {
        TValue::null()
    }
}

impl<T: Into<TValue>> VMCast for T {
    fn vmcast(self, _vm: &VM) -> TValue {
        self.into()
    }
}

impl<T: VMCast> VMCast for Option<T> {
    fn vmcast(self, vm: &VM) -> TValue {
        if let Some(value) = self {
            return value.vmcast(vm);
        }
        TValue::null()
    }
}

impl VMDowncast for TValue {
    fn vmdowncast(value: TValue, _vm: &VM) -> Option<Self> {
        Some(value)
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
            let object = TAnonObject::make(vm, count + 2);
            object.cast()
        }
    }
}

impl GCRef<TType> {
    fn get_static(&self) -> &'static TType {
        unsafe { &*self.as_ptr() }
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

macro_rules! impl_typed {
    (for $ty:ty as $name:ident) => {
        impl Typed for $ty {
            const NAME: &'static str = stringify!($name);
        }
    };
}

macro_rules! into_object {
    (for $ty:ty) => {
        impl Into<TValue> for GCRef<$ty> {
            #[inline(always)]
            fn into(self) -> TValue {
                TValue::object(self)
            }
        }
    };
}

macro_rules! query_object {
    (for $ty:ty) => {
        impl VMDowncast for GCRef<$ty> {
            #[inline(always)]
            fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
                value.query_object::<$ty>(vm)
            }
        }
    };
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
    pub fn from_slice(vm: &VM, slice: &str) -> GCRef<Self> {
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

impl_typed!(for TInteger as int);

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

impl_typed!(for TFloat as float);

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

#[repr(C)]
pub struct TFunction {
    pub name: TValue, /// Optional<TString>
    pub module: GCRef<TModule>,
    pub kind: TFnKind
}

impl_typed!(for TFunction as function);
into_object!(for TFunction);
query_object!(for TFunction);

pub enum TFnKind {
    Function(TRawCode),
    Nativefunc(fn(GCRef<TModule>, TArgsBuffer) -> TValue),
}

impl TFunction {
    pub fn nativefunc(
        module: GCRef<TModule>,
        name: Option<&str>,
        nativefunc: fn(GCRef<TModule>, TArgsBuffer) -> TValue
    ) -> GCRef<TFunction> {
        let vm = module.vm();
        vm.heap().allocate_atom(Self {
            name: name.vmcast(&vm),
            module,
            kind: TFnKind::Nativefunc(nativefunc)
        })
    }
}

impl GCRef<TFunction> {
    #[inline(always)]
    pub fn call(&self, arguments: TArgsBuffer) -> TValue {
        match &self.kind {
            TFnKind::Function(code) =>
                code.evaluate(self.module, arguments),
            TFnKind::Nativefunc(func) =>
                func(self.module, arguments),
        }
    }
}

pub fn print_impl(module: GCRef<TModule>, args: TArgsBuffer) -> TValue {
    let vm = module.vm();
    let mut iter = args.into_iter(1, false);

    let message: GCRef<TString> = iter.next().and_then(|value| VMDowncast::vmdowncast(value, &vm)).unwrap();
    println!("{}", message);
    TValue::null()
}

pub fn module_init(mut module: GCRef<TModule>) {
    let print_function = TFunction::nativefunc(module, Some("print"), print_impl);
    module.set_global(module.vm().symbols().intern_slice("print"), print_function.into(), true).unwrap();
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
        /*vm.heap().allocate_var_atom(
            object,
            count * std::mem::size_of::<TValue>())*/
        todo!()
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
        let hash = attribute.hash();
        self.descriptor.insert(
            hash, (attribute, hash, idx),
            |val| val.1
        );
        *(self.data.as_mut_ptr() as *mut TValue).add(idx) = value;
    }

    unsafe fn query_attribute(&mut self, attribute: Symbol) -> Option<&mut TValue> { 
        let hash = attribute.hash();
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

