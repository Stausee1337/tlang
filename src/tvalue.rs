use std::mem::transmute;


use crate::memory;

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

    const fn object_tagged<T>(object: memory::GCRef<T>, kind: TValueKind) -> Self {
        let object: u64 = unsafe { transmute(object) };
        TValue(Self::FLOAT_NAN_TAG | kind as u64 | object)
    }

    const fn string(string: TString) -> Self {
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

    const fn as_object<T>(&self) -> memory::GCRef<T> {
        memory::GCRef::from_raw((self.0 & Self::NAN_VALUE_MASK) as *mut T)
    }

    fn ttype(&self) -> TType {
        todo!()
    }
}

pub type TType = memory::GCRef<raw_objects::TType>;
pub type TString = memory::GCRef<raw_objects::TString>;
pub type TInteger = raw_objects::TInteger;
pub type TFloat = raw_objects::TFloat;
pub type TBool = raw_objects::TBool;

impl TInteger {
    pub fn as_usize(self) -> Option<usize> {
        match self.0.kind() {
            TValueKind::Int32 => usize::try_from(self.0.as_int32()).ok(),
            TValueKind::Object => {
                debug_assert!(self.0.ttype() == Self::ttype());
                todo!()
            },
            _ => unreachable!()
        }
    }

    pub const fn from_int32(int: i32) -> Self {
        raw_objects::TInteger(TValue::int32(int))
    }

    pub const fn from_bytes(bytes: &[u8]) -> Self {
        todo!()
    }

    pub fn ttype() -> TType {
        todo!()
    }
}

impl Into<TValue> for TInteger {
    fn into(self) -> TValue {
        self.0
    }
}

impl TFloat {
    pub const fn as_float(self) -> f64 {
        self.0
    }

    pub const fn from_float(float: f64) -> Self {
        raw_objects::TFloat(float)
    }

    pub fn ttype() -> TType {
        todo!()
    }
}

impl Into<TValue> for TFloat {
    fn into(self) -> TValue {
        TValue::float(self.0)
    }
}

impl TBool {
    pub const fn as_bool(self) -> bool {
        self.0
    }

    pub const fn from_bool(bool: bool) -> Self {
        raw_objects::TBool(bool)
    }

    pub fn ttype() -> TType {
        todo!()
    }
}

impl Into<TValue> for TBool {
    fn into(self) -> TValue {
        TValue::bool(self.0)
    }
}

impl TString {
    pub fn as_slice(self) -> &'static str {
        let size = self.size.as_usize().expect("TString sensible size");
        todo!()
    }

    pub fn from_slice(slice: &str) -> Self {
        todo!()
    }
}

impl Into<TValue> for TString {
    fn into(self) -> TValue {
        TValue::string(self)
    }
}

// --- TType's ---
// TBool
// TInt
// TFloat
// TString
// TFunction
// TObject

mod raw_objects {
    pub struct TType {
        pub name: super::TString,
    } 

    #[derive(Clone, Copy)]
    pub struct TBool(pub(super) bool);

    #[derive(Clone, Copy)]
    pub struct TInteger(pub(super) super::TValue);

    #[derive(Clone, Copy)]
    pub struct TFloat(pub(super) f64);

    pub struct TString {
        pub size: super::TInteger,
        pub data: [u8; 1]
    }
}

