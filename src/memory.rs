use std::{ops::{Deref, DerefMut}, ffi::c_void, ptr::NonNull, alloc::Layout, cell::Cell, any::TypeId, mem::{transmute, ManuallyDrop}, rc::{Weak, Rc}};
use rustix::{
    io,
    mm::{mmap_anonymous, munmap, MapFlags, ProtFlags}
};

use allocator_api2::alloc::{Allocator, AllocError};
use static_assertions::const_assert_eq;

use crate::vm::VM;

#[repr(u16)]
enum State {
    /// The region of memory does not contain any
    /// used object and is free for allocation
    DEAD  = 0,
    /// The region of memory is considered a live object
    /// it can be moved or collected (and become DEAD)
    ALIVE = 1,
}

#[repr(C)]
struct AllocHead {
    size: u32, state: State, tag: u16,
}

const_assert_eq!(std::mem::size_of::<AllocHead>(), 8);

#[inline(always)]
fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

#[derive(Clone, Copy)]
struct HeapBlock {
    heap: *const Heap,
    previous: *const HeapBlock,
    allocated_bytes: usize
}

impl HeapBlock {
    const PAGE_SIZE: usize = 4096;
    const ALIGN: usize = std::mem::size_of::<usize>();

    const EMPTY: &'static HeapBlock = &HeapBlock {
        heap: std::ptr::null(),
        previous: std::ptr::null(),
        allocated_bytes: Self::PAGE_SIZE
    };

    unsafe fn map(heap: &Heap, previous: &Self) -> io::Result<NonNull<HeapBlock>> {
        let block = mmap_anonymous(
            std::ptr::null_mut(),
            Self::PAGE_SIZE,
            ProtFlags::READ | ProtFlags::WRITE,
            MapFlags::PRIVATE,
        )? as *mut Self;

        *block = HeapBlock {
            heap: &*heap,
            previous: &*previous,
            allocated_bytes: std::mem::size_of::<Self>()
        };

        let block = NonNull::new_unchecked(block);
        Ok(block)
    }

    unsafe fn unmap(&mut self) -> io::Result<()> {
        munmap(self.data() as *mut c_void, HeapBlock::PAGE_SIZE)
    }

    unsafe fn fork(&mut self, heap: &Heap) -> io::Result<NonNull<HeapBlock>> {
        Self::map(heap, self)
    }

    #[inline(always)]
    unsafe fn data(&mut self) -> *mut u8 {
        let raw: *mut Self = &mut *self;
        raw as *mut u8
    }

    #[inline(always)]
    unsafe fn alloc_raw(&mut self, layout: Layout) -> Option<*mut u8> {
        assert!(layout.size() != 0);

        let size = layout.size();
        let align = layout.align();

        let start = self.data();
        let end = start.add(Self::PAGE_SIZE);

        let head = start.add(self.allocated_bytes); 
        let body = head.add(std::mem::size_of::<AllocHead>());

        let body = align_up(body as usize, align) as *mut u8; /*- (body as usize)*/

        let alloc_size = align_up(size, Self::ALIGN);
        if body.add(alloc_size) > end {
            return None;
        }

        let prev_size = self.allocated_bytes;
        self.allocated_bytes = (body.add(alloc_size) as usize) - (start as usize);

        *(head as *mut AllocHead) = AllocHead {
            size: (self.allocated_bytes - prev_size) as u32,
            state: State::ALIVE,
            tag: 0
        };

        println!("{layout:?}, {alloc_size}, {} {:p}", self.allocated_bytes, body);

        Some(body)
    }

    unsafe fn from_allocation<T: Sized>(ptr: *const T) -> &'static HeapBlock {
        let ptr = ((ptr as usize) & !(HeapBlock::PAGE_SIZE - 1)) as *const HeapBlock;
        &*ptr
    }

    fn previous(&self) -> Option<NonNull<Self>> {
        NonNull::new(self.previous as *mut Self)
    }
}

pub struct Heap {
    vm: Weak<VM>,
    current_block: Cell<NonNull<HeapBlock>>
}

impl Heap {
    pub fn init(vm: Weak<VM>) -> Self {
        let block: *const HeapBlock = &*HeapBlock::EMPTY;
        let block = unsafe { NonNull::new_unchecked(block as *mut HeapBlock) };
        Self {
            vm,
            current_block: Cell::new(block)
        }
    }

    pub fn allocate_atom<A: Atom>(&self, atom: A) -> GCRef<A> {
        let mut atom = ManuallyDrop::new(AtomTrait::new(atom));
        unsafe {
            let mut data = self.allocate(
                Layout::new::<AtomTrait<A>>()).unwrap().cast::<AtomTrait<A>>();
            *(data.as_mut()) = ManuallyDrop::take(&mut atom);
            GCRef::from_raw(std::ptr::addr_of!(data.as_mut().atom))
        }
    }

    pub fn allocate_var_atom<A: Atom>(&self, atom: A, extra_bytes: usize) -> GCRef<A> {
        let mut atom = ManuallyDrop::new(AtomTrait::new(atom));
        unsafe {
            let layout = Layout::new::<AtomTrait<A>>();
            let layout = Layout::from_size_align_unchecked(layout.size() + extra_bytes, layout.align());

            let mut data = self.allocate(layout).unwrap().cast::<AtomTrait<A>>();
            *(data.as_mut()) = ManuallyDrop::take(&mut atom);
            GCRef::from_raw(std::ptr::addr_of!(data.as_mut().atom))
        }
    }

    pub fn vm(&self) -> Rc<VM> {
        self.vm.upgrade().expect("we should have dropped to")
    }

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let block = self.current_block.get().as_mut();

            let raw: *mut u8;
            if layout.size() == 0 {
                raw = std::ptr::null_mut();
            } else if let Some(ptr) = block.alloc_raw(layout) {
                raw = ptr;
            } else {
                self.current_block.set(block.fork(self).unwrap());
                let block = self.current_block.get().as_mut();
                raw = block.alloc_raw(layout).ok_or(AllocError)?;
            }

            let raw = NonNull::new(raw).unwrap();
            Ok(NonNull::slice_from_raw_parts(raw, layout.size()))
        }
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        let mut current_block: NonNull<HeapBlock> = self.current_block.get();
        unsafe {
            loop {
                let block = current_block.as_mut();
                let Some(previous) = block.previous() else {
                    break;
                };
                block.unmap().unwrap(); 
                current_block = previous;
            }
        }
    }
}

pub struct Visitor;

pub trait Atom: Send + Sync + 'static {
    fn visit(&self, visitor: &mut Visitor);
}

#[repr(C)]
struct AtomTrait<Atom = ()> {
    vtable: &'static AtomTraitVTable,
    atom: Atom
}

static_assertions::const_assert!(std::mem::size_of::<AtomTrait>() == 8);

impl<A: Atom> AtomTrait<A> {
    fn new(atom: A) -> Self {
        let vtable = &AtomTraitVTable {
            atom_drop: atom_drop::<A>,
            atom_visit: atom_visit::<A>,
            atom_downcast: atom_downcast::<A>,
        };
        AtomTrait { vtable, atom }
    }
}

impl AtomTrait {
    fn downcast<A: Atom>(&self) -> Option<&A> {
        let target = TypeId::of::<A>();
        unsafe {
            if let Some(addr) = (self.vtable.atom_downcast)(self, target) {
                return Some(refcast::<(), A>(addr));
            }
            None
        }
    }

    fn visit(&self, visitor: &mut Visitor) {
        unsafe { (self.vtable.atom_visit)(self, visitor); }
    }

    fn drop(&mut self) {
        unsafe { (self.vtable.atom_drop)(self.into()); }
    }
}

#[repr(C)]
struct AtomTraitVTable {
    atom_drop: unsafe fn(NonNull<AtomTrait>),
    atom_visit: unsafe fn(&'_ AtomTrait, &'_ mut Visitor),
    atom_downcast: unsafe fn(&'_ AtomTrait, TypeId) -> Option<&'_ ()>,
}

unsafe fn atom_drop<A: Atom>(mut a: NonNull<AtomTrait>) {
    let unerased_type: &mut AtomTrait<A> = mutcast(a.as_mut());
    std::ptr::drop_in_place(unerased_type);
}

unsafe fn atom_visit<A: Atom>(a: &'_ AtomTrait, visitor: &'_ mut Visitor) {
    let unerased_type: &AtomTrait<A> = refcast(a);
    unerased_type.atom.visit(visitor);
}

unsafe fn atom_downcast<A: Atom>(a: &'_ AtomTrait, target: TypeId) -> Option<&'_ ()> {
    if TypeId::of::<A>() == target {
        let unerased_type: &'_ AtomTrait<A> = refcast(a);
        let atom = std::ptr::addr_of!(unerased_type.atom);
        return Some(&*(atom as *const ()));
    }
    None
}

unsafe fn refcast<Src, Dst>(a: &Src) -> &Dst {
    transmute::<&Src, &Dst>(a)
}

unsafe fn mutcast<Src, Dst>(a: &mut Src) -> &mut Dst {
    transmute::<&mut Src, &mut Dst>(a)
}

pub struct GCRef<T>(NonNull<T>);

impl<T> GCRef<T> {
    pub(crate) const unsafe fn from_raw(raw: *const T) -> Self {
        GCRef(NonNull::new_unchecked(raw as *mut T))
    }

    pub(crate) unsafe fn cast<U>(&self) -> GCRef<U> {
        GCRef::<U>::from_raw(self.as_ptr() as *const U)
    }

    pub const unsafe fn null() -> Self {
        GCRef::from_raw(std::ptr::null_mut())
    }

    pub const fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }

    pub fn heap(&self) -> &Heap {
        unsafe {
            let block = HeapBlock::from_allocation(self.as_ptr());
            &*block.heap
        }
    }

    pub fn vm(&self) -> Rc<VM> {
        self.heap().vm().clone()
    }

    unsafe fn head(&self) -> *mut AllocHead {
        self.as_ptr().byte_sub(
            std::mem::size_of::<AtomTrait>()
            + std::mem::size_of::<AllocHead>()
        ) as *mut AllocHead
    }

    unsafe fn atom(&self) -> &AtomTrait {
        &*(self.0.as_ptr().byte_sub(std::mem::size_of::<AtomTrait>()) as *const AtomTrait)
    }

    pub fn refrence_eq(&self, other: Self) -> bool {
        std::ptr::addr_eq(self.0.as_ptr(), other.0.as_ptr())
    }
}


unsafe impl<T> Sync for GCRef<T> {}
unsafe impl<T> Send for GCRef<T> {}

impl<T> Clone for GCRef<T> {
    fn clone(&self) -> Self {
        GCRef(self.0)
    }
}
impl<T> std::marker::Copy for GCRef<T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for GCRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.deref(), f)
    }
}

impl<T: std::fmt::Display> std::fmt::Display for GCRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.deref(), f)
    }
}

impl<T> Deref for GCRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl<T> DerefMut for GCRef<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut() }
    }
}

