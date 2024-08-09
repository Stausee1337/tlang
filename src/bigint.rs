use std::{marker::PhantomData, ops::Index, fmt::Display};

use crate::{tvalue::{TInteger, TType, Typed}, memory::GCRef, vm::VM};

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Sign {
    Positive, Negative
}

impl Sign {
    #[inline(always)]
    fn isize(&self) -> isize {
        match self {
            Sign::Positive => 1, 
            Sign::Negative => -1, 
        }
    }
}

#[repr(C)]
pub struct SignedSlice<'a> {
    len: isize,
    data: *const u8,
    _phantom: PhantomData<&'a ()>
}

impl<'a> SignedSlice<'a> {
    #[inline(always)]
    fn from_slice_with_sign(sign: Sign, slice: &'a [u8]) -> Self {
        Self {
            len: slice.len() as isize * sign.isize(),
            data: slice.as_ptr(),
            _phantom: Default::default()
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len.abs() as usize
    }
}

impl<'l> From<GCRef<TBigint>> for SignedSlice<'l> {
    fn from(value: GCRef<TBigint>) -> Self {
        Self {
            len: value.size.as_isize().unwrap(),
            data: value.bytes.as_ptr(),
            _phantom: Default::default()
        }
    }
}

impl<'l> Index<usize> for SignedSlice<'l> {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len.abs() as usize {
            panic!("signed slice out of bounds access");
        }
        unsafe { &*self.data.add(index) }
    }
}

impl<'a, const LENGTH: usize> From<&'a (Sign, [u8; LENGTH])> for SignedSlice<'a> {
    fn from(value: &'a (Sign, [u8; LENGTH])) -> Self {
        SignedSlice::from_slice_with_sign(value.0, &value.1)
    }
}

#[repr(C)]
pub struct TBigint {
    size: TInteger,
    bytes: [u8; 0]
}

/// FIXME: this is incorrect: BigInt should be stored using the TInteger type
impl Typed for TBigint {
    const NAME: &'static str = "int";

    fn ttype(vm: &VM) -> GCRef<TType> {
        todo!()
    }
}

impl Display for TBigint {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

pub fn to_bigint(int: i32) -> (Sign, [u8; 4]) {
    let sign = if int < 0 {
        Sign::Negative
    } else {
        Sign::Positive
    };
    (sign, int.to_le_bytes())
}

pub fn try_as_usize<'a>(_bigint: impl Into<SignedSlice<'a>>) -> Option<usize> {
    todo!()
}

pub fn try_as_isize<'a>(_bigint: impl Into<SignedSlice<'a>>) -> Option<isize> {
    todo!()
}

// ARITHMETIC

pub fn add<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

pub fn sub<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

pub fn mul<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

pub fn div<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

pub fn rem<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

// BITSHIFT

pub fn shl<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

pub fn shr<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

// BITWISE

pub fn bitand<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

pub fn bitor<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}

pub fn bitxor<'a>(_lhs: impl Into<SignedSlice<'a>>, _rhs: impl Into<SignedSlice<'a>>) -> GCRef<TBigint> {
    todo!()
}
