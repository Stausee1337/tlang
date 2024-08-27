use std::{ops::{Deref, DerefMut}, ffi::c_void, ptr::{NonNull, copy_nonoverlapping}, alloc::Layout, cell::Cell, any::TypeId, mem::{transmute, ManuallyDrop}};
use rustix::{
    io,
    mm::{mmap_anonymous, munmap, MapFlags, ProtFlags}
};

use allocator_api2::alloc::AllocError;
use static_assertions::const_assert_eq;

use crate::{vm::{VM, Eternal}, debug};

#[repr(u16)]
#[derive(Debug, PartialEq, Eq)]
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

impl AllocHead {
    unsafe fn free(&mut self, freelist_end: &mut *mut FreeListEntry, last_empty: *mut AllocHead) {
        let atom = &mut *(self as *mut Self as *mut AtomTrait)
            .byte_add(std::mem::size_of::<Self>());
        atom.drop();
        self.state = State::DEAD;

        if last_empty.is_null() {
            let current_entry = self as *mut _ as *mut FreeListEntry;
            (*current_entry).next = *freelist_end;
            *freelist_end = current_entry;
        } else {
            (*last_empty).size += self.size;
        }

    }
}

const_assert_eq!(std::mem::size_of::<AllocHead>(), 8);

#[inline(always)]
fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

#[repr(C)]
struct FreeListEntry {
    head: AllocHead,
    next: *mut FreeListEntry
}

#[derive(Clone, Copy)]
struct HeapBlock {
    heap: *const Heap,
    previous: *const HeapBlock,
    allocated_bytes: usize,
    freelist: *mut FreeListEntry
}

impl HeapBlock {
    const PAGE_SIZE: usize = 4096;
    const ALIGN: usize = std::mem::size_of::<usize>();

    const EMPTY: &'static HeapBlock = &HeapBlock {
        heap: std::ptr::null(),
        previous: std::ptr::null(),
        allocated_bytes: Self::PAGE_SIZE,
        freelist: std::ptr::null_mut()
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
            allocated_bytes: std::mem::size_of::<Self>(),
            freelist: std::ptr::null_mut()
        };

        // debug!("Create Block @ {:p} ", block);

        let block = NonNull::new_unchecked(block);
        Ok(block)
    }

    unsafe fn unmap(&mut self) -> io::Result<()> {
        // debug!("Free Block @ {:p} ", self.data());
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

        let mut previous: *mut *mut FreeListEntry = &mut self.freelist;
        let mut entry = self.freelist;
        while !entry.is_null() {
            let head_size = std::mem::size_of::<AllocHead>();

            let head = &mut (*entry).head;
            debug_assert!(head.state == State::DEAD);
            let avialable_size = (head.size as usize) - head_size;

            if avialable_size < size {
                entry = (*entry).next;
                previous = &mut entry;
                continue;
            }
            // print!("Found spot w/, {avialable_size} bytes, alloc {size}, ");

            // First fit
            head.size = (size + head_size) as u32;
            head.state = State::ALIVE;
            let body = (entry as *mut u8).byte_add(head_size);

            let rem = avialable_size - size;
            if (rem / std::mem::size_of::<usize>()) >= 3 {
                // split the atom
                let new_entry = (body as *mut FreeListEntry).byte_add(size);
                (*new_entry).head = AllocHead {
                    size: rem as u32,
                    state: State::DEAD,
                    tag: u16::MAX
                };
                // print!("Split the Atom {} @ {new_entry:p}", rem);

                // replace self with new_entry in the linked list
                (*new_entry).next = (*entry).next;
                *previous = new_entry;
            } else {
                // just remove self from the linked list
                *previous = (*entry).next;
            }

            return Some(body);
        }

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
            tag: u16::MAX
        };

        // debug!("{layout:?}, {alloc_size}, {} {:p}", self.allocated_bytes, body);

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
    tag: Cell<u16>,
    vm: Eternal<VM>,
    current_block: Cell<NonNull<HeapBlock>>,
    defer_gc: Cell<bool>
}

impl Heap {
    pub fn init(vm: Eternal<VM>) -> Self {
        let block: *const HeapBlock = &*HeapBlock::EMPTY;
        let block = unsafe { NonNull::new_unchecked(block as *mut HeapBlock) };
        Self {
            tag: Cell::new(0),
            vm,
            current_block: Cell::new(block),
            defer_gc: Cell::new(false)
        }
    }

    pub fn defer_gc(&self) {
        self.defer_gc.set(true);
    }

    pub fn undefer_gc(&self) {
        self.defer_gc.set(false);
    }

    pub fn allocate_atom<A: Atom>(&self, atom: A) -> GCRef<A> {
        let mut atom = ManuallyDrop::new(AtomTrait::new(atom));
        unsafe {
            let mut data = self.allocate(
                Layout::new::<AtomTrait<A>>()).unwrap().cast::<AtomTrait<A>>();
            copy_nonoverlapping(atom.deref(), data.as_ptr(), 1);
            GCRef::from_raw(std::ptr::addr_of!(data.as_mut().atom))
        }
    }

    pub fn allocate_var_atom<A: Atom>(&self, atom: A, extra_bytes: usize) -> GCRef<A> {
        let mut atom = ManuallyDrop::new(AtomTrait::new(atom));
        unsafe {
            let layout = Layout::new::<AtomTrait<A>>();
            let layout = Layout::from_size_align_unchecked(layout.size() + extra_bytes, layout.align());

            let mut data = self.allocate(layout).unwrap().cast::<AtomTrait<A>>();
            copy_nonoverlapping(atom.deref(), data.as_ptr(), 1);
            GCRef::from_raw(std::ptr::addr_of!(data.as_mut().atom))
        }
    }

    pub fn vm(&self) -> Eternal<VM> {
        self.vm.clone()
    }

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let block = self.current_block.get().as_mut();
            assert!(layout.size() != 0, "ZST are unsupported");

            let raw: *mut u8;
            if let Some(ptr) = block.alloc_raw(layout) {
                raw = ptr;
            } else {
                if block.previous().is_some() && !self.defer_gc.get() {
                    let vm = self.vm();
                    let collector = GarbageCollector::with_last_tag(self.tag.get());
                    collector.mark_phase(&vm);
                    collector.sweep_phase(&vm);

                    self.tag.set(collector.current_tag);

                    if let Some(ptr) = block.alloc_raw(layout) {
                        raw = ptr;
                    } else {
                        self.current_block.set(block.fork(self).unwrap());
                        let block = self.current_block.get().as_mut();
                        raw = block.alloc_raw(layout).ok_or(AllocError)?;
                    }
                } else {
                    self.current_block.set(block.fork(self).unwrap());
                    let block = self.current_block.get().as_mut();
                    raw = block.alloc_raw(layout).ok_or(AllocError)?;
                }
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

#[repr(C)]
struct StackFrame {
    previous: *const StackFrame,
    return_adress: *const (),
    data: [u8; 0]
}

impl StackFrame {
    #[inline]
    unsafe fn size(&self) -> usize {
        (self.previous() as *const Self as *const u8).sub_ptr(self.data.as_ptr()) 
    }

    unsafe fn data(&self) -> &[u8] {
        std::slice::from_raw_parts(
            self.data.as_ptr(),
            self.size()
        )
    }

    #[inline]
    const unsafe fn previous(&self) -> &StackFrame {
        &*self.previous
    }
}

unsafe impl bytemuck::Zeroable for crate::tvalue::TValue {}
unsafe impl bytemuck::Pod for crate::tvalue::TValue {}

struct StackWalker;

impl StackWalker {
    unsafe fn get_frame<'a>() -> &'a StackFrame {
        let ptr: *const StackFrame;
        std::arch::asm! {
            "mov {}, rbp",
            out(reg) ptr
        };
        &*ptr
    }

    fn walk<F: FnMut(crate::tvalue::TValue)>(mut f: F) {
        unsafe {
            let mut frame = Self::get_frame();
            let mut count = 0;
            while frame.size() > 0 {
                let data: &[usize] = bytemuck::cast_slice(frame.data());
                if data[0] != 0xdeadbeef {
                    frame = frame.previous();
                    continue;
                }

                // debug!("Stack walk frame {:p}", frame);

                let values: &[crate::tvalue::TValue] = bytemuck::cast_slice(&data[1..]);

                let mut idx = 0;
                loop {
                    let value = values[idx];
                    if values[idx].encoded() == 0x0 {
                        break;
                    }
                    f(value);

                    // debug!("  - {:?} 0x{:x}", value.kind(), value.encoded());

                    idx += 1;
                }

                frame = frame.previous();
                count += 1;
            }
            // debug!("Visted {count} stack frames");
        }
    }
}

struct GarbageCollector {
    current_tag: u16,
}

impl GarbageCollector {
    fn with_last_tag(prev: u16) -> Self {
        Self {
            current_tag: prev + 1
        }
    }

    fn mark_phase(&self, vm: &VM) {
        // recursivly visit every life object, starting at the static (eternal) objects
        let mut visitor = Visitor {
            count: 0,
            current_tag: self.current_tag
        };
        visitor.feed(vm.symbols);
        visitor.feed(vm.types);
        visitor.feed(vm.modules);
        visitor.feed(vm.primitives);

        // vist objects on custom stack
        StackWalker::walk(|value| {
            value.visit(&mut visitor);
        });
    }

    fn sweep_phase(&self, vm: &VM) {
        let heap = vm.heap(); 
        let mut current_block: NonNull<HeapBlock> = heap.current_block.get();
        unsafe {
            let mut block_count = 0usize;
            let mut atom_count = 0usize;
            let mut freed_count = 0usize;
            loop {
                let block = current_block.as_mut();
                let Some(previous) = block.previous() else {
                    break;
                };

                let mut head = (block as *mut _ as *mut AllocHead)
                    .byte_add(std::mem::size_of::<HeapBlock>());

                let mut last_empty: *mut AllocHead = std::ptr::null_mut();
                while (*head).size != 0 {
                    if (*head).tag != self.current_tag && (*head).state == State::ALIVE {
                        freed_count += 1;
                        (*head).free(&mut block.freelist, last_empty);
                        if last_empty.is_null() {
                            last_empty = head;
                        }
                    } else {
                        last_empty = std::ptr::null_mut();
                    }
                    // debug!("Head @ {:p} w/ {} bytes is {:?}", head, (*head).size, (*head).state);
                    head = head.byte_add((*head).size as usize);
                    // offset += (*head).size as usize;
                    atom_count += 1;
                }

                block_count += 1;

                current_block = previous;
            }
            println!("Swept across {block_count} blocks w/ {atom_count} atoms; Collected {freed_count}");
        }
    }
}

pub struct Visitor {
    count: usize,
    current_tag: u16,
}

impl Visitor {
    pub fn feed<T>(&mut self, ptr: GCRef<T>) {
        unsafe {
            let head = GCRef::head(ptr);
            if (*head).tag == self.current_tag {
                return;
            }

            (*head).tag = self.current_tag;

            let atom = GCRef::atom(ptr);
            atom.visit(self);
            self.count += 1;
        }
    }
}

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
    // debug!("{}::Drop::drop", std::any::type_name::<A>());
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

pub(crate) unsafe fn refcast<Src, Dst>(a: &Src) -> &Dst {
    transmute::<&Src, &Dst>(a)
}

pub(crate) unsafe fn mutcast<Src, Dst>(a: &mut Src) -> &mut Dst {
    transmute::<&mut Src, &mut Dst>(a)
}

pub struct GCRef<T>(NonNull<T>);

impl<T> GCRef<T> {
    pub(crate) const unsafe fn from_raw(raw: *const T) -> Self {
        GCRef(transmute(raw))
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

    pub fn vm(&self) -> Eternal<VM> {
        self.heap().vm().clone()
    }

    pub fn refrence_eq(&self, other: Self) -> bool {
        std::ptr::addr_eq(self.0.as_ptr(), other.0.as_ptr())
    }

    unsafe fn head(this: Self) -> *mut AllocHead {
        this.as_ptr().byte_sub(
            std::mem::size_of::<AtomTrait>()
            + std::mem::size_of::<AllocHead>()
        ) as *mut AllocHead
    }

    unsafe fn atom<'a>(this: Self) -> &'a AtomTrait {
        &*(this.as_ptr().byte_sub(std::mem::size_of::<AtomTrait>()) as *const AtomTrait)
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

