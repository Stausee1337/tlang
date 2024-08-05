use std::{mem::{transmute, MaybeUninit}, hash::{BuildHasher, Hash, Hasher}};


use hashbrown::raw::RawTable;

use crate::{memory, symbol::Symbol};

#[repr(u64)]
#[derive(Debug)]
enum TValueKind {
    Object   = 0b000 << 49,
    Int32    = 0b001 << 49,
    Bool     = 0b010 << 49,
    String   = 0b110 << 49,
    Function = 0b101 << 49,
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

    const fn bool(bool: bool) -> Self {
        let bool = bool as u64;
        TValue(Self::FLOAT_NAN_TAG | TValueKind::Bool as u64 | bool)
    }

    const fn int32(int: i32) -> Self {
        let int = int as u64;
        TValue(Self::FLOAT_NAN_TAG | TValueKind::Int32 as u64 | int)
    }

    const fn float(float: f64) -> Self {
        return TValue(unsafe { transmute(float) });
    }

    const fn object_tagged<T: memory::Atom>(object: memory::GCRef<T>, kind: TValueKind) -> Self {
        let object: u64 = unsafe { transmute(object) };
        assert!(object & (!Self::NAN_VALUE_MASK) == 0);
        TValue(Self::FLOAT_NAN_TAG | kind as u64 | object)
    }

    const fn string(string: memory::GCRef<TString>) -> Self {
        Self::object_tagged(string, TValueKind::String)
    }

    /// Private Helpers

    fn kind(&self) -> TValueKind {
        let float: f64 = unsafe { transmute(self.0) };
        if !float.is_nan() {
            return TValueKind::Float;
        }
        unsafe { transmute(self.0 & Self::FLOAT_TAG_MASK) }
    }

    const fn as_int32(&self) -> i32 {
        (self.0 & Self::NAN_VALUE_MASK) as u32 as i32
    }

    const fn as_bool(&self) -> bool {
        (self.0 & Self::NAN_VALUE_MASK) != 0
    }

    const fn as_float(&self) -> f64 {
        unsafe { transmute(self.0) }
    }

    const fn as_object<T: memory::Atom>(&self) -> memory::GCRef<T> {
        memory::GCRef::from_raw((self.0 & Self::NAN_VALUE_MASK) as *mut T)
    }

}

pub struct TType {

}

impl memory::Atom for TType {}

#[derive(Clone, Copy)]
pub struct TInteger(TValue);

impl TInteger {
    pub fn as_usize(self) -> Option<usize> {
        match self.0.kind() {
            TValueKind::Int32 => usize::try_from(self.0.as_int32()).ok(),
            TValueKind::Object => {
                // debug_assert!(self.0.type() == Self::type());
                todo!()
            },
            _ => unreachable!()
        }
    }

    pub const fn from_int32(int: i32) -> Self {
        TInteger(TValue::int32(int))
    }

    /// Converts a sequence of signed little endian bytes
    /// into a TInteger representation
    pub const fn from_bytes(bytes: &[u8]) -> Self {
        /*if bytes.len() <= std::mem::size_of::<i32>() {

        }*/
        todo!()
    }

}

impl Into<TValue> for TInteger {
    fn into(self) -> TValue {
        self.0
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

#[derive(Clone, Copy)]
pub struct TBool(pub(super) bool);

impl TBool {
    pub const fn as_bool(self) -> bool {
        self.0
    }

    pub const fn from_bool(bool: bool) -> Self {
        TBool(bool)
    }

}

impl Into<TValue> for TBool {
    fn into(self) -> TValue {
        TValue::bool(self.0)
    }
}

#[repr(C)]
pub struct TString {
    pub size: TInteger,
    pub data: [u8; 1]
}
impl memory::Atom for TString {}

impl TString {
    pub fn as_slice(self) -> &'static str {
        let size = self.size.as_usize().expect("TString sensible size");
        unsafe {
            let bytes = std::slice::from_raw_parts(self.data.as_ptr(), size);
            let str = std::str::from_utf8_unchecked(bytes);
            str
        }
    }

    pub fn from_slice(slice: &str) -> memory::GCRef<Self> {
        let size = (slice.len() as isize).to_le_bytes();
        let size = TInteger::from_bytes(&size);
        todo!()
    }
}

impl Into<TValue> for memory::GCRef<TString> {
    fn into(self) -> TValue {
        TValue::string(self)
    }
}

#[repr(C)]
struct TObjectHead {
    ttype: TType,
    descriptor: RawTable<Symbol, memory::BlockAllocator>,
    data: [u8; 1]
}

impl TObjectHead {
    fn getattr(&self, attribute: Symbol) -> Option<TValue> {
        let builder = ahash::RandomState::new();
        let mut hasher = builder.build_hasher();
        attribute.get().hash(&mut hasher);
        let hash = hasher.finish();

        let bucket = self.descriptor.find(hash, |key| attribute == *key);
        let Some(bucket) = bucket else {
            return None;
        };

        unsafe {
            let size = self.descriptor.len() * std::mem::size_of::<TValue>();
            let values = std::slice::from_raw_parts(
                self.data.as_ptr() as *const TValue,
                size
            );
            let idx = self.descriptor.bucket_index(&bucket);
            Some(values[idx])
        }
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

