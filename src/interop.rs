use std::marker::PhantomData;

use crate::{memory::GCRef, tvalue::{TObject, TValue, TString, Typed, TInteger, TFunction, CallResult}, vm::VM, eval::TArgsBuffer};


pub struct TPolymorphicWrapper<T: TPolymorphicObject> {
    object: GCRef<TObject>,
    _phantom: PhantomData<T>
}

impl<T: TPolymorphicObject> VMCast for TPolymorphicWrapper<T> {
    fn vmcast(self, vm: &VM) -> TValue {
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
    fn try_decode(vm: &VM, args: TArgsBuffer) -> Option<Self>;
    fn encode(self, vm: &VM) -> TArgsBuffer;
}


impl VMArgs for () {
    fn try_decode(_vm: &VM, args: TArgsBuffer) -> Option<Self> {
        let _iter = args.into_iter(0, false);
        Some(())
    }

    fn encode(self, vm: &VM) -> TArgsBuffer {
        TArgsBuffer::empty()
    }
}

impl VMDowncast for () {
    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
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
    fn try_decode(vm: &VM, args: TArgsBuffer) -> Option<Self> {
        let mut iter = args.into_iter(1, false);
        let arg1 = iter.next()?;
        Some((
            T1::vmdowncast(arg1, vm)?,
        ))
    }

    fn encode(self, vm: &VM) -> TArgsBuffer {
        TArgsBuffer::debug(vec![self.0.vmcast(vm)])
    }
}

impl<T1, T2> VMArgs for (T1, T2)
where
    T1: VMDowncast + VMCast,
    T2: VMDowncast + VMCast
{
    fn try_decode(vm: &VM, args: TArgsBuffer) -> Option<Self> {
        let mut iter = args.into_iter(2, false);
        let arg1 = iter.next()?;
        let arg2 = iter.next()?;
        Some((
            T1::vmdowncast(arg1, vm)?,
            T2::vmdowncast(arg2, vm)?,
        ))
    }

    fn encode(self, vm: &VM) -> TArgsBuffer {
        TArgsBuffer::debug(vec![
            self.0.vmcast(vm),
            self.1.vmcast(vm)
        ])
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

impl<In: VMArgs, R: VMDowncast> VMCast for TPolymorphicCallable<In, R> {
    fn vmcast(self, vm: &VM) -> TValue {
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
        let object = value.query_tobject()?;
        todo!("querry object::call and decide based on this");
    }
}

impl<In: VMArgs + 'static, R: VMDowncast + 'static> FnOnce<In> for TPolymorphicCallable<In, R> {
    type Output = R;

    #[inline(always)]
    extern "rust-call" fn call_once(self, args: In) -> Self::Output {
        match self.inner {
            CallableInner::Function(tfunction) => {
                let vm = tfunction.vm();
                match tfunction.fastcall(args) {
                    CallResult::Result(result) => result,
                    CallResult::NotImplemented(args) => {
                        let buffer = args.encode(&vm);
                        R::vmdowncast(tfunction.call(buffer), &vm).unwrap()
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
