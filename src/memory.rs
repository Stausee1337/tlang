use std::{ops::{Deref, DerefMut}, ffi::c_void, ptr::NonNull, alloc::Layout, cell::Cell, any::TypeId, mem::{transmute, ManuallyDrop}, rc::{Weak, Rc}};
use rustix::{
    io,
    mm::{mmap_anonymous, munmap, MapFlags, ProtFlags}
};

use allocator_api2::alloc::{Allocator, AllocError};
use static_assertions::const_assert_eq;

use crate::vm::VM;

struct State(u16);

impl State {
    /// The region of memory does not contain any
    /// used object and is free for allocation
    const DEAD:   State = State(0);
    /// The region of memory is considered a live object
    /// it can be moved or collected (and become DEAD)
    const ALIVE:  State = State(1);
    /// The region of memory is considered a live object
    /// it can be moved but never be collected. It is considered
    /// an entry point for finding live objects
    const STATIC: State = State(2);
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

    pub fn allocate_atom<A: Atom, T>(&self, kind: &'static A, object: T) -> GCRef<T> {
        let atom = AtomTrait::new(kind);
        let mut allocation = ManuallyDrop::new(Allocation(atom, object));
        unsafe {
            let mut data = self.allocate(
                Layout::new::<Allocation<A, T>>()).unwrap().cast::<Allocation<A, T>>();
            *(data.as_mut()) = ManuallyDrop::take(&mut allocation);
            GCRef::from_allocation(data.as_ptr())
        }
    }

    pub fn allocate_var_atom<A: Atom, T>(&self, kind: &'static A, object: T, extra_bytes: usize) -> GCRef<T> {
        let atom = AtomTrait::new(kind);
        let mut allocation = ManuallyDrop::new(Allocation(atom, object));
        unsafe {
            let layout = Layout::new::<Allocation<A, T>>();
            let layout = Layout::from_size_align_unchecked(layout.size() + extra_bytes, layout.align());

            let mut data = self.allocate(layout).unwrap().cast::<Allocation<A, T>>();
            *(data.as_mut()) = ManuallyDrop::take(&mut allocation);
            GCRef::from_allocation(data.as_ptr())
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

pub struct StaticAtom;
static STATIC_ALLOCATOR: &'static StaticAtom = &StaticAtom;

impl Atom for StaticAtom {
    fn iterate_children(&self, _p: *const ()) -> Box<dyn Iterator<Item = *const ()>> {
        Box::new(std::iter::empty())
    }
}

impl StaticAtom {
    pub fn atom() -> &'static Self {
        STATIC_ALLOCATOR
    }

    pub fn allocate<T>(heap: &Heap, object: T) -> GCRef<T> {
        heap.allocate_atom(STATIC_ALLOCATOR, object)
    }
}

pub trait Atom: Send + Sync + 'static {
    fn iterate_children(&self, p: *const ()) -> Box<dyn Iterator<Item = *const ()>>;
}

#[repr(C)]
struct AtomTrait<Atom: 'static = ()> {
    test: &'static str,
    vtable: &'static AtomTraitVTable,
    atom: &'static Atom
}

const_assert_eq!(std::mem::size_of::<AtomTrait>(), 32);

impl<A: Atom> AtomTrait<A> {
    fn new(atom: &'static A) -> Self {
        let vtable = &AtomTraitVTable {
            atom_ref: atom_ref::<A>,
            atom_downcast: atom_downcast::<A>,
        };
        AtomTrait { test: "Hello, World", vtable, atom }
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
}

#[repr(C)]
struct AtomTraitVTable {
    atom_downcast: unsafe fn(&'_ AtomTrait, TypeId) -> Option<&'_ ()>,
    atom_ref: unsafe fn(&'_ AtomTrait) -> &'_ dyn Atom,
}

unsafe fn atom_ref<A: Atom>(a: &AtomTrait) -> &dyn Atom {
    let unerased_a = refcast::<AtomTrait, AtomTrait<A>>(a);
    unerased_a.atom
}

unsafe fn atom_downcast<A: Atom>(a: &'_ AtomTrait, target: TypeId) -> Option<&'_ ()> {
    if TypeId::of::<A>() == target {
        return Some(a.atom);
    }
    None
}

unsafe fn refcast<Src, Dst>(a: &Src) -> &Dst {
    transmute::<&Src, &Dst>(a)
}

#[repr(C)]
struct Allocation<A: 'static, T>(AtomTrait<A>, T);

#[derive(Eq)]
pub struct GCRef<T>(*mut T);

impl<T> GCRef<T> {
    pub(crate) const unsafe fn from_raw(raw: *const T) -> Self {
        GCRef(raw as *mut T)
    }

    pub(crate) unsafe fn cast<U>(&self) -> GCRef<U> {
        GCRef::<U>::from_raw(self.as_ptr() as *const U)
    }

    pub const unsafe fn null() -> Self {
        GCRef(std::ptr::null_mut())
    }

    pub const fn as_ptr(&self) -> *mut T {
        self.0
    }

    pub fn kind<A: Atom>(&self) -> Option<&A> {
        unsafe {
            let atom = self.atom();
            println!("calced atom {:p}", atom);
            atom.downcast()
        }
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

    pub fn drop_static(&self) -> GCRef<T> {
        unsafe {
            let head = self.head();
            (&mut *head).state = State::ALIVE;
        }
        *self
    }

    pub fn make_static(&self) -> GCRef<T> {
        unsafe {
            let head = self.head();
            (&mut *head).state = State::STATIC;
        }
        *self
    }

    unsafe fn head(&self) -> *mut AllocHead {
        self.as_ptr().byte_sub(
            std::mem::size_of::<AtomTrait>()
            + std::mem::size_of::<AllocHead>()
        ) as *mut AllocHead
    }

    unsafe fn atom(&self) -> &AtomTrait {
        &*(self.0.byte_sub(std::mem::size_of::<AtomTrait>()) as *const AtomTrait)
    }

    unsafe fn from_allocation<A: Atom>(allocation: *mut Allocation<A, T>) -> GCRef<T> {
        GCRef(&mut (*allocation).1)
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
        unsafe { &*self.0 }
    }
}

impl<T> DerefMut for GCRef<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

