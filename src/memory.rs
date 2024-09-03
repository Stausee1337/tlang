use std::{ops::{Deref, DerefMut}, ffi::c_void, ptr::{NonNull, copy_nonoverlapping}, alloc::Layout, cell::{Cell, UnsafeCell}, any::TypeId, mem::{transmute, ManuallyDrop}, usize, io::Write};
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
    const MAX_SIZE: u32 = (HeapBlock::BLOCK_SIZE - std::mem::size_of::<HeapBlock>()) as u32;

    unsafe fn free(&mut self, freelist_end: &mut *mut FreeListEntry) {
        let atom = &mut *(self as *mut Self as *mut AtomTrait)
            .byte_add(std::mem::size_of::<Self>());
        atom.drop();
        self.state = State::DEAD;
    }
}

const_assert_eq!(std::mem::size_of::<AllocHead>(), 8);

#[inline(always)]
fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
enum Generations {
    Gen0 = 0,
    Gen1 = 1,
    Gen2 = 2,
}

impl Generations {
    fn next(self) -> Self {
        unsafe { transmute((self as usize) + 1) }
    }
}

#[repr(C)]
struct FreeListEntry {
    head: AllocHead,
    next: *mut FreeListEntry
}

#[derive(Clone, Copy)]
#[repr(C)]
struct HeapBlock {
    sentinel: usize,
    heap: *const Heap,
    previous: *mut HeapBlock,
    generation: Generations,
    freelist: *mut FreeListEntry
}

impl HeapBlock {
    const PAGE_SIZE: usize = 4096;
    const BLOCK_SIZE: usize = 4 * Self::PAGE_SIZE; // 16 KiB
    const ALIGN: usize = std::mem::size_of::<usize>();

    const EMPTY: &'static HeapBlock = &HeapBlock {
        sentinel: 0xdeadbeef,
        heap: std::ptr::null(),
        previous: std::ptr::null_mut(),
        generation: Generations::Gen2,
        freelist: std::ptr::null_mut(),
    };

    const CAPACITY: usize = Self::BLOCK_SIZE - std::mem::size_of::<Self>();

    unsafe fn map(heap: &Heap, previous: &mut Self) -> io::Result<*mut HeapBlock> {
        let block = mmap_anonymous(
            std::ptr::null_mut(),
            Self::BLOCK_SIZE,
            ProtFlags::READ | ProtFlags::WRITE,
            MapFlags::PRIVATE,
        )? as *mut Self;

        *block = HeapBlock {
            sentinel: 0xdeadbeef,
            heap: &*heap,
            previous: &mut *previous,
            generation: Generations::Gen0,
            freelist: std::ptr::null_mut(),
        };

        (*block).reset(previous);

        // debug!("Create Block @ {:p} ", block);

        Ok(block)
    }

    unsafe fn reset(&mut self, previous: &mut Self) {
        let freelist = (self as *mut _ as *mut FreeListEntry).byte_add(std::mem::size_of::<Self>());

        (*freelist).head = AllocHead {
            state: State::DEAD,
            size: Self::CAPACITY as u32,
            tag: u16::MAX
        };
        (*freelist).next = std::ptr::null_mut();

        self.previous = &mut *previous;
        self.generation = Generations::Gen0;
        self.freelist = freelist;
    }

    unsafe fn unmap(&mut self) -> io::Result<()> {
        // debug!("Free Block @ {:p} ", self.data());
        munmap(self.data() as *mut c_void, HeapBlock::BLOCK_SIZE)
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

        let mut previous: *mut *mut FreeListEntry = std::ptr::addr_of_mut!(self.freelist);
        let mut entry = self.freelist;
        let mut iteration = 0;
        while !entry.is_null() {
            let head_size = std::mem::size_of::<AllocHead>();

            let head = &mut (*entry).head;
            assert!(head.state == State::DEAD);
            let avialable_size = (head.size as usize) - head_size;
            
            let align_ptr = (head as *mut _ as usize) + std::mem::size_of::<AllocHead>();
            let padding = align_up(align_ptr, align) - align_ptr;

            let size = align_up(size + padding, Self::ALIGN);

            if avialable_size < size {
                previous = std::ptr::addr_of_mut!((*entry).next);
                entry = (*entry).next;
                iteration += 1;
                continue;
            }

            // First fit
            head.size = (size + head_size) as u32;
            head.state = State::ALIVE;
            let body = (entry as *mut u8).byte_add(head_size + padding);

            let rem = avialable_size - size;
            if (rem / std::mem::size_of::<usize>()) >= 3 {
                // split the atom
                let new_entry = (body as *mut FreeListEntry).byte_add(size);
                (*new_entry).head = AllocHead {
                    size: rem as u32,
                    state: State::DEAD,
                    tag: u16::MAX
                };

                // replace self with new_entry in the linked list
                (*new_entry).next = (*entry).next;
                *previous = new_entry;
            } else {
                // just remove self from the linked list
                *previous = (*entry).next;
                head.size += rem as u32;
            }

            return Some(body);
        }

        None
    }

    unsafe fn from_allocation<T: Sized>(ptr: *const T) -> &'static HeapBlock {
        let mut ptr = ptr as *const HeapBlock;

        for _ in 0..4 {
            ptr = ((ptr as usize) & !(HeapBlock::PAGE_SIZE - 1)) as *const HeapBlock;
            if (*ptr).sentinel == 0xdeadbeef {
                return &*ptr;
            }
            ptr = ptr.byte_sub(1);
        }
        std::process::abort();
    }

    fn previous(&self) -> Option<NonNull<Self>> {
        NonNull::new(self.previous as *mut Self)
    }
}

pub struct Heap {
    vm: Eternal<VM>,
    data: UnsafeCell<HeapData>
}

struct HeapData {
    tag: u16,
    current_block: *mut HeapBlock,
    block_cache: *mut HeapBlock,
    defer_gc: bool,

    // GC Collection params
    allocations: usize,
    previous_nodes_count: usize,
    /// amount of allocations to start a collection
    gc_allocations_threshold: usize,
    /// counts the number of collections, in which no memory was reclamed
    fruitless_collections: usize,
    /// amount of fruitless collections needed
    /// to double the number of allocations needed to start a collection,
    /// as well as doubling this number
    gc_collections_threshold: usize,
}

impl Heap {
    pub fn init(vm: Eternal<VM>) -> Self {
        let block: *const HeapBlock = &*HeapBlock::EMPTY;
        Self {
            vm,
            data: UnsafeCell::new(HeapData {
                current_block: block as *mut HeapBlock,
                block_cache: std::ptr::null_mut(),
                tag: 0,
                defer_gc: false,

                allocations: 0,
                previous_nodes_count: 0,
                gc_allocations_threshold: 1536,
                fruitless_collections: 0,
                gc_collections_threshold: 8
            })
        }
    }

    #[inline(always)]
    pub fn data(&self) -> &mut HeapData {
        unsafe { &mut *self.data.get() }
    }

    pub fn defer_gc(&self) {
        self.data().defer_gc = true;
    }

    pub fn undefer_gc(&self) {
        self.data().defer_gc = false;
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

    unsafe fn fork_block(&self) -> io::Result<*mut HeapBlock> {
        let this = self.data();
        if !this.block_cache.is_null() {
            let block = this.block_cache;
            this.block_cache = (*block).previous;

            (*block).reset(&mut *this.current_block);

            return Ok(block);
        }
        HeapBlock::map(self, &mut *this.current_block)
    }

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let this = self.data();
        unsafe {
            assert!(layout.size() != 0, "ZST are unsupported");

            let raw: *mut u8;
            if let Some(ptr) = (*this.current_block).alloc_raw(layout) {
                raw = ptr;
            } else {
                if (*this.current_block).previous().is_some() &&
                        !this.defer_gc &&
                        this.allocations >= this.gc_allocations_threshold {
                    let vm = self.vm();
                    let collector = GarbageCollector::with_last_tag(this.tag);
                    let nodes_count = collector.mark_phase(&vm);

                    let mut sweep_mode = SweepMode::Ephermeral;
                    if nodes_count < this.previous_nodes_count {
                        sweep_mode = SweepMode::Extended( (this.previous_nodes_count - nodes_count)/2 );
                    }

                    if this.fruitless_collections == (this.gc_collections_threshold - 1) {
                        sweep_mode = SweepMode::Full;
                    }

                    let freed_count = collector.sweep_phase(&vm, sweep_mode);
                    /*println!("Run GC after {} allocations counting {} fruitless collections",
                             this.allocations, this.fruitless_collections);*/

                    if freed_count == 0 {
                        this.fruitless_collections += 1;
                    } else {
                        this.fruitless_collections = 0;
                        /*if this.gc_collections_threshold > 8 {
                            this.gc_collections_threshold /= 2;
                        }
                        if this.gc_allocations_threshold > 192 {
                            this.gc_allocations_threshold /= 2;
                        }*/
                    }

                    if this.fruitless_collections == this.gc_collections_threshold {
                        this.fruitless_collections = 0;
                        this.gc_collections_threshold *= 2;
                        this.gc_allocations_threshold *= 2;
                    }

                    this.tag = collector.current_tag;
                    this.allocations = 0;
                    this.previous_nodes_count = nodes_count;

                    if let Some(ptr) = (*this.current_block).alloc_raw(layout) {
                        raw = ptr;
                    } else {
                        this.current_block = self.fork_block().unwrap();
                        raw = (*this.current_block).alloc_raw(layout).ok_or(AllocError)?;
                    }
                } else {
                    this.current_block = self.fork_block().unwrap();
                    raw = (*this.current_block).alloc_raw(layout).ok_or(AllocError)?;
                }
            }

            this.allocations += 1;

            let raw = NonNull::new(raw).unwrap();
            Ok(NonNull::slice_from_raw_parts(raw, layout.size()))
        }
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        let this = self.data.get_mut();
        let mut current_block = this.current_block;
        let mut count = 0usize;
        unsafe {
            loop {
                let Some(previous) = (*current_block).previous() else {
                    break;
                };
                count += 1;
                (*current_block).unmap().unwrap(); 
                current_block = previous.as_ptr();
            }
            println!("Unmaped {count} blocks");

            current_block = this.block_cache;
            count = 0;



            if !this.block_cache.is_null() {
                loop {
                    let Some(previous) = (*current_block).previous() else {
                        break;
                    };
                    count += 1;
                    (*current_block).unmap().unwrap(); 
                    current_block = previous.as_ptr();
                }
            }

            println!("And {count} cached blocks");
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

struct StackWalker<'l> {
    vm: &'l VM
}

impl<'l> StackWalker<'l> {
    unsafe fn get_frame<'a>(&self) -> &'a StackFrame {
        let ptr: *const StackFrame;
        std::arch::asm! {
            "mov {}, rbp",
            out(reg) ptr
        };
        &*ptr
    }

    fn walk<F: FnMut(crate::tvalue::TValue)>(&self, mut f: F) {
        unsafe {
            let mut frame = self.get_frame();
            let mut count = 0;
            let mut vmcount = 0;
            while frame.size() > 0 {
                let data: &[usize] = bytemuck::cast_slice(frame.data());
                count += 1;
                if data[0] != 0xdeadbeef {
                    frame = frame.previous();
                    continue;
                }

                // debug!("Stack walk frame {:p}", frame);

                let values: &[crate::tvalue::TValue] = bytemuck::cast_slice(&data[1..]);
                let mut is_entry_frame = false;
                if let Some(tfn) = values[0].query_object::<crate::tvalue::TFunction>(self.vm) {
                    is_entry_frame = tfn.name.is_none();
                } else {
                    debug!(" -> WARNING: Frame without conclusive fn object");
                }

                let mut idx = 0;
                for value in values {
                    if value.encoded() == 0x0 {
                        break;
                    }
                    f(*value);

                    // debug!("  - {:?} 0x{:x}", value.kind(), value.encoded());
                }

                frame = frame.previous();
                vmcount += 1;

                if is_entry_frame {
                    break;
                }
            }
            // debug!("Visted {vmcount} vm frames, {count} total");
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SweepMode {
    /// Only collect the ephermeral generations (Gen0)
    Ephermeral,
    /// Collect Gen0 + Gen1
    Extended(/* threshold: */ usize),
    /// Collect all generations Gen0, Gen1 and Gen2
    Full
}

struct GarbageCollector {
    current_tag: u16,
}

impl GarbageCollector {
    fn with_last_tag(prev: u16) -> Self {
        Self {
            current_tag: prev ^ 1
        }
    }

    fn mark_phase(&self, vm: &VM) -> usize {
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
        let walker = StackWalker { vm };
        walker.walk(|value| {
            value.visit(&mut visitor);
        });
        
        visitor.count
    }

    fn sweep_phase(&self, vm: &VM, sweep_mode: SweepMode) -> usize {
        let heap = vm.heap().data();
        let mut all_blocks: Vec<*mut HeapBlock> = Vec::new();
        unsafe {
            let mut block_count = 0usize;
            let mut atom_count = 0usize;
            let mut freed_count = 0usize;

            let mut max_gen = if sweep_mode == SweepMode::Full {
                Generations::Gen2
            } else {
                // always starts out as Gen0, dynamically adjusted during sweep
                Generations::Gen0
            };

            let mut current_block: NonNull<HeapBlock> = NonNull::new_unchecked(heap.current_block);
            let mut last: *mut *mut HeapBlock = std::ptr::addr_of_mut!(heap.current_block);
            loop {
                let block = current_block.as_mut();
                let Some(previous) = block.previous() else {
                    break;
                };

                if block.generation > max_gen {
                    break;
                }

                let mut head = (block as *mut _ as *mut AllocHead)
                    .byte_add(std::mem::size_of::<HeapBlock>());
                let first_head = head;

                let mut last_empty: *mut AllocHead = std::ptr::null_mut();
                let mut sweep_size = 0;
                block.freelist = std::ptr::null_mut();
                while sweep_size < HeapBlock::CAPACITY {
                    if (*head).tag != self.current_tag && (*head).state == State::ALIVE {
                        freed_count += 1;
                        (*head).free(&mut block.freelist);
                    }

                    if (*head).state == State::DEAD {
                        if last_empty.is_null() {
                            let current_entry = head as *mut FreeListEntry;
                            (*current_entry).next = block.freelist;
                            block.freelist = current_entry;

                            last_empty = head;
                        } else {
                            (*last_empty).size += (*head).size;
                        }
                    } else {
                        last_empty = std::ptr::null_mut();
                    }
                    sweep_size += (*head).size as usize;
                    // debug!("Head @ {:p} w/ {} bytes is {:?} {sweep_size}/{}",
                    // head, (*head).size, (*head).state, HeapBlock::CAPACITY);
                    head = head.byte_add((*head).size as usize);
                    atom_count += 1;
                }

                let mut smallest_entry_size = HeapBlock::CAPACITY;
                let mut entry = block.freelist;
                let mut freelist_size = 0;
                let mut entry_count = 0;
                while !entry.is_null() {
                    let size = (*entry).head.size as usize;
                    freelist_size += size;
                    if size < smallest_entry_size {
                        smallest_entry_size = size;
                    }

                    entry_count += 1;
                    entry = (*entry).next;
                }

                // let fragmentation_ratio = (freelist_size/smallest_entry_size) * entry_count;
                // if entry_count > 0 && fragmentation_ratio != 1 {
                //     println!("{fragmentation_ratio} Ratio; {freelist_size}/{smallest_entry_size}");
                // }
                if block.generation < Generations::Gen2 && freelist_size < 1376 {
                    block.generation = block.generation.next(); // Upgrade block to the next higher
                                                                // generation
                }

                let first_entry = first_head as *mut FreeListEntry;
                if (*first_entry).head.state == State::DEAD && (*first_entry).head.size == HeapBlock::CAPACITY as u32 {
                    block.previous = heap.block_cache;
                    heap.block_cache = block;
                    *last = previous.as_ptr();
                } else {
                    all_blocks.push(block);
                    last = std::ptr::addr_of_mut!(block.previous);
                }

                block_count += 1;

                // println!("{previous:p}");
                if previous.as_ref().generation > max_gen {
                    if let SweepMode::Extended(threshold) = sweep_mode {
                        if freed_count < threshold {
                            max_gen = Generations::Gen1;
                        }
                    }
                }

                current_block = previous;
            }

            if block_count == 0 {
                return 0;
            }

            all_blocks.sort_by(|&a, &b| (*a).generation.cmp(&(*b).generation));

            let mut last = all_blocks.pop().unwrap();
            (*last).previous = current_block.as_ptr();
            for &block in all_blocks.iter().rev() {
                // println!("Updating previous as {last:p} {:?}", (*block).generation);
                (*block).previous = last;
                last = block;
            }

            (*last).generation = Generations::Gen0;

            heap.current_block = last;

            // println!("Swept {sweep_mode:?} across {block_count} blocks w/ {atom_count} atoms; Collected {freed_count}");
            freed_count
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

    pub fn refrence_eq(this: Self, other: Self) -> bool {
        std::ptr::addr_eq(this.0.as_ptr(), other.0.as_ptr())
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

