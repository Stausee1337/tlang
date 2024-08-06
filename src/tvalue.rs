use std::{mem::{transmute, MaybeUninit}, hash::{BuildHasher, Hash, Hasher}, cell::OnceCell};


use hashbrown::raw::RawTable;

use crate::{memory::{self, GCRef}, symbol::Symbol, interpreter::get_interpeter, bytecode};

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

    const fn as_object<T>(&self) -> memory::GCRef<T> {
        memory::GCRef::from_raw((self.0 & Self::NAN_VALUE_MASK) as *mut T)
    }
}

#[repr(C)]
pub struct TType {
    ty: GCRef<TType>,
}

impl TType {
    pub fn create() -> GCRef<Self> {  
        let interpreter = get_interpeter();
        interpreter.block_allocator.allocate_object(
            Self { ty: Self::ttype() }
        ) 
    }

    pub fn ttype() -> GCRef<TType> {
        static mut TYPE: OnceCell<GCRef<TType>> = OnceCell::new();
        unsafe {
            if let Some(ty) = TYPE.get_mut() {
                if ty.ty == GCRef::from_raw(std::ptr::null_mut::<TType>()) {
                    ty.ty = *ty;
                }
            }
            *TYPE.get_or_init(|| {
                let interpreter = get_interpeter();
                let ty = GCRef::from_raw(std::ptr::null_mut());
                interpreter.block_allocator.allocate_object(
                    Self { ty }
                )
            })
        }
    }
}

#[derive(Clone, Copy)]
pub struct TInteger(TValue);

#[repr(C)]
struct TIntObject {
    ty: GCRef<TType>,
    bytes: [u8; 0]
}

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
    pub fn from_bytes(bytes: &[u8]) -> Self {
        // TODO: real bigintegers
        if bytes.len() <= std::mem::size_of::<i32>() {
            let int = match bytes {
                [b1] => i8::from_le_bytes([*b1]) as i32,
                [b1, b2] => i16::from_le_bytes([*b1, *b2]) as i32,
                [b1, b2, b3, b4] => i32::from_le_bytes([*b1, *b2, *b3, *b4]) as i32,
                _ => unreachable!()
            };
            return Self::from_int32(int);
        }

        match bytes {
            [b1, b2, b3, b4, b5, b6, b7, b8] => {
                let int = i64::from_le_bytes([*b1, *b2, *b3, *b4, *b5, *b6, *b7, *b8]);
                if let Ok(int) = i32::try_from(int) {
                    return Self::from_int32(int);
                }
                let object = Self::as_object(bytes);
                TInteger(TValue::object_tagged(object, TValueKind::Object))
            },
            _ => todo!("real bigint support")
        }
    }

    fn as_object(bytes: &[u8]) -> GCRef<TIntObject> {
        let interpreter = get_interpeter();
        let mut object = interpreter.block_allocator.allocate_var_object(
            TIntObject { ty: Self::ttype(), bytes: [0u8; 0] },
            bytes.len()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), object.bytes.as_mut_ptr(), bytes.len());
        }
        object
    }

    pub fn ttype() -> GCRef<TType> {
        static mut TYPE: OnceCell<GCRef<TType>> = OnceCell::new();
        unsafe {
            *TYPE.get_or_init(|| TType::create())
        }
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
    ty: GCRef<TType>,
    pub size: TInteger,
    pub length: TInteger,
    pub data: [u8; 0]
}

impl TString {
    pub fn as_slice<'a>(self) -> &'a str {
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

        let length = (slice.chars().count() as isize).to_le_bytes();
        let length = TInteger::from_bytes(&length);

        let interpreter = get_interpeter();
        let mut string = interpreter.block_allocator.allocate_var_object(
            Self { ty: Self::ttype(), size, length, data: [0u8; 0] },
            slice.len()
        );

        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), string.data.as_mut_ptr(), slice.len());
        }

        string
    }

    pub fn ttype() -> GCRef<TType> {
        static mut TYPE: OnceCell<GCRef<TType>> = OnceCell::new();
        unsafe {
            *TYPE.get_or_init(|| TType::create())
        }
    }
}

impl Into<TValue> for memory::GCRef<TString> {
    fn into(self) -> TValue {
        TValue::string(self)
    }
}

#[repr(C)]
pub struct TFunction {
    ty: GCRef<TType>,
    pub name: GCRef<TString>,
    kind: TFnKind
}

enum TFnKind {
    Function(TRawCode),
}

impl TFunction {
    pub fn create_presized(
        name: &str,
        codesize: usize,
        registers: u32,
        descriptors: u32,
        blocks: u32
    ) -> (GCRef<TFunction>, &mut TRawCode) {
        let extra_size = codesize
            + descriptors as usize * std::mem::size_of::<TValue>()
            + blocks as usize * std::mem::size_of::<u32>();
        let interpreter = get_interpeter();
        let code = TRawCode::new(codesize, registers, descriptors, blocks);
        let mut function = interpreter.block_allocator.allocate_var_object(
            Self {
                ty: Self::ttype(),
                name: TString::from_slice(name),
                kind: TFnKind::Function(code)
            },
            extra_size
        );
        let function2 = function.clone();
        match function.kind {
            TFnKind::Function(ref mut code) => {
                (function2, unsafe { transmute(code) })
            }
        }
    }

    pub fn ttype() -> GCRef<TType> {
        static mut TYPE: OnceCell<GCRef<TType>> = OnceCell::new();
        unsafe {
            *TYPE.get_or_init(|| TType::create())
        }
    }
}

#[repr(C)]
pub struct TRawCode {
    code_size: TInteger,
    num_registers: TInteger,
    num_descriptors: TInteger,
    num_blocks: TInteger,
    data: [u8; 0]
}

impl TRawCode {
    fn new(
        codesize: usize,
        registers: u32,
        descriptors: u32,
        blocks: u32
    ) -> Self {
        Self {
            code_size: TInteger::from_bytes(&(codesize as isize).to_le_bytes()),
            num_registers: TInteger::from_bytes(&(registers as i32).to_le_bytes()),
            num_descriptors: TInteger::from_bytes(&(descriptors as i32).to_le_bytes()),
            num_blocks: TInteger::from_bytes(&(blocks as i32).to_le_bytes()),
            data: [0u8; 0]
        }
    }
    
    fn create_presized(
        codesize: usize,
        registers: u32,
        descriptors: u32,
        blocks: u32
    ) -> GCRef<Self> {
        let extra_size = codesize
            + descriptors as usize * std::mem::size_of::<TValue>()
            + blocks as usize * std::mem::size_of::<u32>();
        let interpreter = get_interpeter();
        interpreter.block_allocator.allocate_var_object(
            TRawCode::new(codesize, registers, descriptors, blocks),
            extra_size
        )
    }

    pub fn blocks(&self) -> &[u32] {
        unsafe {
            let len = self.num_blocks.as_usize().expect("TRawCode sensible num_blocks");
            let offset = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors")
                * std::mem::size_of::<TValue>();
            std::slice::from_raw_parts(
                self.data.as_ptr().add(offset) as *const u32,
                len
            )
        }
    }

    pub fn blocks_mut(&mut self) -> &mut [u32] {
        unsafe {
            let len = self.num_blocks.as_usize().expect("TRawCode sensible num_blocks");
            let offset = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors")
                * std::mem::size_of::<TValue>();
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(offset) as *mut u32,
                len
            )
        }
    }

    pub fn code(&self) -> &[u8] {
        unsafe {
            let len = self.code_size.as_usize().expect("TRawCode sensible code_size");
            let offset = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors")
                * std::mem::size_of::<TValue>()
                + self.num_blocks.as_usize().expect("TRawCode sensible num_blocks")
                * std::mem::size_of::<u32>();
            std::slice::from_raw_parts(
                self.data.as_ptr().add(offset),
                len
            )
        }
    }

    pub fn code_mut(&mut self) -> &mut [u8] {
        unsafe {
            let len = self.code_size.as_usize().expect("TRawCode sensible code_size");
            let offset = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors")
                * std::mem::size_of::<TValue>()
                + self.num_blocks.as_usize().expect("TRawCode sensible num_blocks")
                * std::mem::size_of::<u32>();
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(offset),
                len
            )
        }
    }

    pub fn descriptors(&self) -> &[TValue] {
        unsafe {
            let len = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors");
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const TValue,
                len
            )
        }
    }

    pub fn descriptors_mut(&mut self) -> &mut [TValue] {
        unsafe {
            let len = self.num_descriptors.as_usize().expect("TRawCode sensible num_descriptors");
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut TValue,
                len
            )
        }
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

