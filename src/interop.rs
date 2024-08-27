use std::{marker::PhantomData, cell::OnceCell, mem::MaybeUninit};

use crate::{memory::GCRef, tvalue::{TObject, TValue, TString, Typed, TInteger, TFunction, CallResult, TProperty, FunctionFlags, TBool}, vm::{VM, Eternal}, eval::TArgsBuffer};


pub struct TPolymorphicWrapper<T: TPolymorphicObject> {
    object: GCRef<TObject>,
    _phantom: PhantomData<T>
}

impl<T: TPolymorphicObject> VMCast for TPolymorphicWrapper<T> {
    fn vmcast(self, _vm: &VM) -> TValue {
        TValue::object(self.object)
    }
}

impl<T: TPolymorphicObject> VMDowncast for TPolymorphicWrapper<T> {
    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
        let object = value.query_tobject()?;
        if !object.ty.is_subclass(vm.types().query::<T>()) {
            return None;
        }
        Some(TPolymorphicWrapper {
            object,
            _phantom: PhantomData::default()
        })
    }
}

impl<T: TPolymorphicObject> TPolymorphicWrapper<T> {
    pub(crate) unsafe fn raw_access<P: Sized>(&self, offset: usize) -> *mut P {
        let ptr = self.object.as_ptr() as *mut u8;
        ptr.add(offset) as *mut P
    }
}

///
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
pub unsafe trait TPolymorphicObject: Typed {
    type Base: TPolymorphicObject;
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

impl VMCast for bool {
    fn vmcast(self, _vm: &VM) -> TValue {
        TBool::from_bool(self).into()
    }
}

impl VMCast for i32 {
    fn vmcast(self, _vm: &VM) -> TValue {
        TInteger::from_int32(self).into()
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

impl<T: VMDowncast> VMDowncast for Option<T> {
    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
        if value.encoded() == TValue::null().encoded() {
            return None;
        }
        Some(T::vmdowncast(value, vm))
    }
}

pub trait VMArgs: std::marker::Tuple + Sized {
    const SIZE: usize;

    fn try_decode(vm: &VM, args: TArgsBuffer) -> Option<Self>;
    fn encode_into(self, vm: &VM, slice: &mut [TValue]);
}


impl VMArgs for () {
    const SIZE: usize = 0;

    fn try_decode(_vm: &VM, args: TArgsBuffer) -> Option<Self> {
        let _iter = args.into_iter(0, false);
        Some(())
    }

    fn encode_into(self, _vm: &VM, _slice: &mut [TValue])
    { }
}

impl VMDowncast for () {
    fn vmdowncast(value: TValue, _vm: &VM) -> Option<Self> {
        if value.encoded() == TValue::null().encoded() {
            return Some(());
        }
        None
    }
}

impl<T1> VMArgs for (T1,)
where
    T1: VMDowncast + VMCast
{
    const SIZE: usize = 1;

    fn try_decode(vm: &VM, args: TArgsBuffer) -> Option<Self> {
        let mut iter = args.into_iter(1, false);
        let arg1 = iter.next()?;
        Some((
            T1::vmdowncast(arg1, vm)?,
        ))
    }

    fn encode_into(self, vm: &VM, slice: &mut [TValue]) {
        slice[0] = self.0.vmcast(vm);
    }
}

impl<T1, T2> VMArgs for (T1, T2)
where
    T1: VMDowncast + VMCast,
    T2: VMDowncast + VMCast
{
    const SIZE: usize = 2;

    fn try_decode(vm: &VM, args: TArgsBuffer) -> Option<Self> {
        let mut iter = args.into_iter(2, false);
        let arg1 = iter.next()?;
        let arg2 = iter.next()?;
        Some((
            T1::vmdowncast(arg1, vm)?,
            T2::vmdowncast(arg2, vm)?,
        ))
    }

    fn encode_into(self, vm: &VM, slice: &mut [TValue]) { 
        slice[0] = self.0.vmcast(vm);
        slice[1] = self.1.vmcast(vm);
    }
}

impl<T1, T2, T3> VMArgs for (T1, T2, T3)
where
    T1: VMDowncast + VMCast,
    T2: VMDowncast + VMCast,
    T3: VMDowncast + VMCast
{
    const SIZE: usize = 3;

    fn try_decode(vm: &VM, args: TArgsBuffer) -> Option<Self> {  
        let mut iter = args.into_iter(3, false);
        let arg1 = iter.next()?;
        let arg2 = iter.next()?;
        let arg3 = iter.next()?;
        Some((
            T1::vmdowncast(arg1, vm)?,
            T2::vmdowncast(arg2, vm)?,
            T3::vmdowncast(arg3, vm)?,
        ))
    }

    fn encode_into(self, vm: &VM, slice: &mut [TValue]) { 
        slice[0] = self.0.vmcast(vm);
        slice[1] = self.1.vmcast(vm);
    }
}

#[derive(Clone, Copy)]
enum CallableInner {
    Polymorph(GCRef<TObject>),
    Function(GCRef<TFunction>)
}

#[derive(Clone, Copy)]
pub struct TPolymorphicCallable<In: VMArgs, R: VMDowncast> {
    inner: CallableInner,
    _phantom: PhantomData<(In, R)>
}

impl<In: VMArgs, R: VMDowncast> TPolymorphicCallable<In, R> {
    #[inline]
    pub fn is_method(&self) -> bool {
        match self.inner {
            CallableInner::Polymorph(..) =>
                true, // callable object are always implemented via methods
            CallableInner::Function(tfunction) =>
                tfunction.flags.contains(FunctionFlags::METHOD),
        }
    }

    #[inline]
    pub fn reencode<In2: VMArgs, R2: VMDowncast>(self) -> TPolymorphicCallable<In2, R2> {
        TPolymorphicCallable::<In2, R2> {
            inner: self.inner,
            _phantom: PhantomData::default()
        }
    }
}

impl<In: VMArgs, R: VMDowncast> VMCast for TPolymorphicCallable<In, R> {
    fn vmcast(self, _vm: &VM) -> TValue {
        match self.inner {
            CallableInner::Polymorph(tobject) =>
                TValue::object(tobject),
            CallableInner::Function(tfunction) =>
                TValue::object(tfunction),
        }
    }
}

impl<In: VMArgs, R: VMDowncast> VMDowncast for TPolymorphicCallable<In, R> {
    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
        if let Some(tfunction) = value.query_object::<TFunction>(vm) {
            return Some(Self {
                inner: CallableInner::Function(tfunction),
                _phantom: PhantomData::default()
            });
        }
        let _object = value.query_tobject()?;
        todo!("querry object::call and decide based on this");
    }
}

impl<In: VMArgs + 'static, R: VMDowncast + 'static> FnOnce<In> for TPolymorphicCallable<In, R> {
    type Output = R;

    #[inline(always)]
    extern "rust-call" fn call_once(self, args: In) -> Self::Output {
        const MAX_ARGS: usize = 8;

        match self.inner {
            CallableInner::Function(tfunction) => {
                let vm = tfunction.vm();
                match tfunction.fastcall(args) {
                    CallResult::Result(result) => result,
                    CallResult::NotImplemented(args) => {
                        let mut buffer = [TValue::null(); MAX_ARGS];
                        args.encode_into(&vm, &mut buffer[..In::SIZE]);
                        R::vmdowncast(tfunction.call(TArgsBuffer::create(&mut buffer[..In::SIZE])), &vm).unwrap()
                    }
                }
            }
            CallableInner::Polymorph(..) => todo!()
        }
    }
}

impl<In: VMArgs, R: VMDowncast> From<GCRef<TFunction>> for TPolymorphicCallable<In, R> {
    fn from(value: GCRef<TFunction>) -> Self {
        Self {
            inner: CallableInner::Function(value),
            _phantom: PhantomData::default()
        }
    }
}

pub trait VMPropertyCast {
    fn propcast(this: TValue, value: &mut TValue, writeable: bool, vm: &VM) -> Self;
}

impl<T: VMDowncast> VMPropertyCast for T {
    fn propcast(_this: TValue, value: &mut TValue, _writeable: bool, vm: &VM) -> T {
        T::vmdowncast(*value, vm).unwrap()
    }
}

enum AccessKind {
    Property {
        property: GCRef<TProperty>,
        this: TValue,
    },
    Attribute {
        vm: Eternal<VM>,
        attribute_val: *mut TValue,
    }
}

pub struct TPropertyAccess<T: VMDowncast + VMCast + Copy + 'static> {
    copy: OnceCell<T>,
    place: MaybeUninit<T>,
    write: bool,
    /// Flag to see if an attribute is considered writeable
    /// Properties provide thier own mechanism to check if a write is legal
    writeable: bool,
    kind: AccessKind
}

impl<T: VMDowncast + VMCast + Copy + 'static> TPropertyAccess<T> {
    pub fn as_method(&self) -> Option<GCRef<TFunction>> {
        let AccessKind::Attribute { vm, attribute_val } = &self.kind else {
            return None;
        };
        if self.writeable {
            return None;
        }
        let value = unsafe { **attribute_val };
        let Some(tfunction) = value.query_object::<TFunction>(vm) else {
            return None;
        };
        if !tfunction.flags.contains(FunctionFlags::METHOD) {
            return None;
        }
        Some(tfunction)
    }
}

impl<T: VMDowncast + VMCast + Copy + 'static> VMPropertyCast for TPropertyAccess<T> {
    fn propcast(this: TValue, value: &mut TValue, writeable: bool, vm: &VM) -> Self {
        let kind = 'kind: {
            if !writeable { // if properties are accessed as writeable, these are direct
                            // accesses, and need to be resolved as attributes
                if let Some(property) = GCRef::<TProperty>::vmdowncast(*value, vm) {
                    break 'kind AccessKind::Property {
                        property, this,
                    };
                }
            }
            AccessKind::Attribute {
                vm: vm.heap().vm(),
                attribute_val: &mut *value,
            }
        };
        return TPropertyAccess {
            kind,
            copy: OnceCell::new(),
            place: MaybeUninit::uninit(),
            write: false,
            writeable
        }
    }
}

impl<T: VMDowncast + VMCast + Copy + 'static> TPropertyAccess<T> {
    fn get(&self) -> &T {
        if let Some(value) = self.copy.get() {
            return value;
        }
        match &self.kind {
            AccessKind::Property { property, this } => {
                let Some(getter) = property.get else {
                    panic!("cannot get property");
                };
                let getter: TPolymorphicCallable<_, T> = getter.into();
                let value = getter(*this);
                self.copy.set(value).map_err(|_err| ()).expect("copy is empty"); 
                return self.get();
            }
            AccessKind::Attribute { attribute_val: value, vm } => {
                let value = T::vmdowncast(unsafe { **value }, &vm).unwrap(); 
                self.copy.set(value).map_err(|_err| ()).expect("copy is empty"); 
                return self.get();
            }
        }
    }
}

impl<T: VMDowncast + VMCast + Copy + 'static> std::ops::Deref for TPropertyAccess<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T: VMDowncast + VMCast + Copy + 'static> std::ops::DerefMut for TPropertyAccess<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if !self.writeable && matches!(self.kind, AccessKind::Attribute { .. }) {
            panic!("cannot set property");
        }
        self.write = true;
        unsafe { self.place.assume_init_mut() }
    }
}

impl<T: VMDowncast + VMCast + Copy + 'static> std::ops::Drop for TPropertyAccess<T> {
    fn drop(&mut self) {
        if !self.write {
            return;
        }
        let value = unsafe { self.place.assume_init() };
        match &mut self.kind {
            AccessKind::Property { property, this } => {
                let Some(setter) = property.set else {
                    panic!("cannot set property");
                };
                let setter: TPolymorphicCallable<_, ()> = setter.into();
                setter(*this, value);
            }
            AccessKind::Attribute { attribute_val, vm } => {
                unsafe {
                    **attribute_val = T::vmcast(value, vm);
                }
            }
        }
    }
}

pub mod vmops {
    macro_rules! define_trait {
        ($tname:ident, $fname:ident) => {
            pub trait $tname<Rhs = Self> {
                fn $fname(self, rhs: Rhs) -> crate::tvalue::TBool;
            }
        };
    }

    /*define_trait!(Add, add);
    define_trait!(Sub, sub);
    define_trait!(Mul, mul);
    define_trait!(Div, div);
    define_trait!(Rem, rem);

    define_trait!(Shl, shl);
    define_trait!(Shr, shr);

    define_trait!(BitAnd, bitand);
    define_trait!(BitOr, bitor);
    define_trait!(BitXor, bitxor);*/

    // define_trait!(Eq, eq);
    // define_trait!(Ne, ne);
    define_trait!(Gt, gt);
    define_trait!(Ge, ge);
    define_trait!(Lt, lt);
    define_trait!(Le, le);

    pub trait Invert {
        type Output;
        fn invert(self) -> Self::Output;
    }
}

