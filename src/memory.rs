use std::{ops::{Deref, DerefMut}, ffi::c_void, ptr::NonNull, alloc::Layout, cell::Cell};
use rustix::{
    io,
    mm::{mmap_anonymous, munmap, MapFlags, ProtFlags}
};

use allocator_api2::alloc::{Allocator, AllocError};
use static_assertions::const_assert_eq;

#[repr(C)]
struct AtomHead {
    size: u32, alive: u16, tag: u16,
}

const_assert_eq!(std::mem::size_of::<AtomHead>(), 8);

#[inline(always)]
fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

#[derive(Clone, Copy)]
struct HeapBlock {
    data: NonNull<u8>,
    allocated_bytes: usize
}

impl HeapBlock {
    const PAGE_SIZE: usize = 4096;
    const ALIGN: usize = std::mem::size_of::<usize>();

    unsafe fn map() -> io::Result<NonNull<HeapBlock>> {
        let memory = mmap_anonymous(
            std::ptr::null_mut(),
            Self::PAGE_SIZE,
            ProtFlags::READ | ProtFlags::WRITE,
            MapFlags::PRIVATE,
        )?;
        let mut block = HeapBlock {
            data: NonNull::new(memory as *mut u8).unwrap(),
            allocated_bytes: 0
        };

        let raw = block.alloc_raw(Layout::new::<HeapBlock>()).unwrap() as *mut HeapBlock;
        *raw = block;

        Ok(NonNull::new_unchecked(raw))
    }

    unsafe fn unmap(&self) -> io::Result<()> {
        munmap(self.data.as_ptr() as *mut c_void, HeapBlock::PAGE_SIZE)
    }

    #[inline(always)]
    unsafe fn alloc_raw(&mut self, layout: Layout) -> Option<*mut u8> {
        assert!(layout.size() != 0);

        let size = layout.size();
        let align = layout.align();

        let start = self.data.as_ptr();
        let end = start.add(Self::PAGE_SIZE);

        let head = start.add(self.allocated_bytes); 
        let body = head.add(std::mem::size_of::<AtomHead>());

        let body = align_up(body as usize, align) as *mut u8; /*- (body as usize)*/

        let alloc_size = align_up(size, Self::ALIGN);
        if body.add(alloc_size) > end {
            return None;
        }

        let prev_size = self.allocated_bytes;
        self.allocated_bytes = (body.add(alloc_size) as usize) - (start as usize);

        *(head as *mut AtomHead) = AtomHead {
            size: (self.allocated_bytes - prev_size) as u32,
            alive: 1,
            tag: 0
        };

        // self.allocated_bytes += padding;

        // let addr = self.data.as_ptr().add(self.allocated_bytes);

        // self.allocated_bytes += alloc_size;

        println!("{layout:?}, {alloc_size}, {}", self.allocated_bytes);

        Some(body)
    }
}

pub struct BlockAllocator {
    current_block: Cell<NonNull<HeapBlock>>
}

impl BlockAllocator {
    pub fn init() -> Self {
        let root_block = unsafe { HeapBlock::map().unwrap() };
        Self { current_block: Cell::new(root_block) }
    }

    pub fn allocate_object<T>(&self, object: T) -> GCRef<T> {
        unsafe {
            let mut data = self.allocate(Layout::new::<T>()).unwrap().cast::<T>();
            *(data.as_mut()) = object;
            GCRef(data.as_ptr())
        }
    }

    pub fn allocate_var_object<T>(&self, object: T, extra_bytes: usize) -> GCRef<T> {
        unsafe {
            let layout = Layout::new::<T>();
            let layout = Layout::from_size_align_unchecked(layout.size() + extra_bytes, layout.align());

            let mut data = self.allocate(layout).unwrap().cast::<T>();
            *(data.as_mut()) = object;
            GCRef(data.as_ptr())
        }
    }
}

unsafe impl Allocator for BlockAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let block = self.current_block.get().as_mut();

            let raw: *mut u8;
            if layout.size() == 0 {
                raw = std::ptr::null_mut();
            } else if let Some(ptr) = block.alloc_raw(layout) {
                raw = ptr;
            } else {
                return Err(AllocError);
            }

            let raw = NonNull::new(raw).unwrap();
            Ok(NonNull::slice_from_raw_parts(raw, layout.size()))
        }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) { }
}

#[derive(Eq)]
pub struct GCRef<T>(*mut T);

impl<T> GCRef<T> {
    pub const fn from_raw(raw: *mut T) -> Self {
        GCRef(raw)
    }
}

unsafe impl<T> Sync for GCRef<T> {}
unsafe impl<T> Send for GCRef<T> {}

impl<T> PartialEq for GCRef<T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(self.0, other.0)
    }
}

impl<T> Clone for GCRef<T> {
    fn clone(&self) -> Self {
        GCRef(self.0)
    }
}
impl<T> std::marker::Copy for GCRef<T> {}

impl<T> Deref for GCRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl<T> DerefMut for GCRef<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

