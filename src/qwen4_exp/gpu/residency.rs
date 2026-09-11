//! The expert pool: which records the GPU may touch, LRU managed here,
//! with two backings behind one interface.
//!
//! `Set`: a Metal residency set over the packed expert file's mapping.
//! Every record is its own zero-copy sub-buffer (wrapped on first use);
//! the set holds up to a budget of them and the GPU reaches a record
//! through its GPU address. A record leaving the set keeps its pages in
//! the page cache until the OS needs the memory, so a miss on a recently
//! evicted record is served from RAM (an unwired second tier with no
//! copies); a cold miss reads the file through the cache before the
//! record is added. Records added while a command buffer waits on an
//! event are resident by the time the GPU is released (measured: 1 to
//! 4 ms for 150 records, no faults).
//!
//! `Copy`: one wired Metal buffer of record slots, filled by uncached
//! parallel reads. No second tier, but no page-cache churn either, which
//! matters when the machine is short of memory.

use crate::metal::MetalContext;
use crate::units::BYTES_PER_KIB;
use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor,
};
use std::ffi::c_void;
use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Per-record completion retained by the deadline policy until its slot
/// is safe to reuse. All record IO uses the measured pread path.
#[derive(Clone)]
pub struct Landed(Arc<AtomicBool>);

impl Landed {
    pub fn done(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    pub fn wait(&self) -> Result<()> {
        let t = std::time::Instant::now();

        while !self.done() {
            std::hint::spin_loop();
            anyhow::ensure!(
                t.elapsed().as_secs() < 30,
                "a record read did not land within 30 s"
            );
        }

        Ok(())
    }
}

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

pub enum Pool {
    Set(Residency),
    Copy(CopyPool),
}

/// One record read. A zero destination means a page-cache read for a
/// mapped residency-set record; otherwise it is a CPU address in the copy
/// pool. `need_index` connects completion to the caller's acquired records.
struct PlannedRead {
    destination: usize,
    file_offset: usize,
    bytes: usize,
    /// 0 = base 4-bit store, 1 = 3-bit, 2 = 2-bit.
    kind: u8,
    need_index: usize,
}

/// Reads required between pool acquisition and residency completion.
/// Built by the service thread, then moved or borrowed by reader threads.
pub struct ReadPlan {
    items: Vec<PlannedRead>,
    low_file: Option<File>,
    /// Base record size for reads through the page cache.
    stride: usize,
}

impl ReadPlan {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Run the reads: copies go to their slots past the cache
    /// (`nocache`), cache reads pull pages in through `cached`.
    /// Run every read on its own detached thread; returns one flag per
    /// entry of the `need` list the plan was built for (`need_len`), set
    /// once that record is in memory (records that needed no read start
    /// set). The caller keeps the flags of reads it stops waiting for and
    /// checks them before the slot can be reused.
    pub fn run_tracked(&self, cached: &File, nocache: &File, need_len: usize) -> Vec<Landed> {
        let flags: Vec<Arc<AtomicBool>> = (0..need_len)
            .map(|_| Arc::new(AtomicBool::new(true)))
            .collect();

        for read in &self.items {
            let (dst, off, len) = (read.destination, read.file_offset, read.bytes);
            let flag = flags[read.need_index].clone();

            flag.store(false, Ordering::Release);

            let stride = self.stride;
            let cached = cached.try_clone().expect("dup fd");
            let src = if read.kind != 0 {
                self.low_file.as_ref().expect("low-bit store")
            } else {
                nocache
            }
            .try_clone()
            .expect("dup fd");

            std::thread::spawn(move || {
                if dst != 0 {
                    fetch_into_slots(&src, &[(dst, off, len)]);
                } else {
                    read_through_cache(&cached, stride, &[off]);
                }

                flag.store(true, Ordering::Release);
            });
        }

        flags.into_iter().map(Landed).collect()
    }

    pub fn run(&self, cached: &File, nocache: &File) {
        let pulls: Vec<usize> = self
            .items
            .iter()
            .filter(|read| read.destination == 0)
            .map(|read| read.file_offset)
            .collect();
        let copies4: Vec<(usize, usize, usize)> = self
            .items
            .iter()
            .filter(|read| read.destination != 0 && read.kind == 0)
            .map(|read| (read.destination, read.file_offset, read.bytes))
            .collect();
        let low_copies: Vec<(usize, usize, usize)> = self
            .items
            .iter()
            .filter(|read| read.destination != 0 && read.kind != 0)
            .map(|read| (read.destination, read.file_offset, read.bytes))
            .collect();

        std::thread::scope(|s| {
            if !pulls.is_empty() {
                s.spawn(move || read_through_cache(cached, self.stride, &pulls));
            }

            if !copies4.is_empty() {
                s.spawn(move || fetch_into_slots(nocache, &copies4));
            }

            if !low_copies.is_empty() {
                let low_file = self.low_file.as_ref().expect("low-bit store");

                s.spawn(move || fetch_into_slots(low_file, &low_copies));
            }
        });
    }
}

/// Fill independent slots in parallel, one complete record per read.
/// Splitting reads did not improve measured latency.
pub fn fetch_into_slots(file: &File, fetch: &[(usize, usize, usize)]) {
    use std::os::unix::fs::FileExt as _;

    std::thread::scope(|s| {
        for &(dst, off, len) in fetch {
            s.spawn(move || {
                // Each slot has one writer and is not yet visible to the GPU.
                let buf = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) };
                let mut done = 0;

                while done < buf.len() {
                    match file.read_at(&mut buf[done..], (off + done) as u64) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => done += n,
                    }
                }
            });
        }
    });
}

/// Pull records' pages into memory through the page cache (parallel
/// reads into per-thread scratch; the mapping's pages are what the
/// residency set will pin).
fn read_through_cache(file: &File, stride: usize, offsets: &[usize]) {
    use std::os::unix::fs::FileExt as _;

    const PIECE: usize = 256 * BYTES_PER_KIB;

    std::thread::scope(|s| {
        for &off in offsets {
            s.spawn(move || {
                let mut scratch = vec![0u8; PIECE];
                let mut done = 0usize;

                while done < stride {
                    let n = PIECE.min(stride - done);

                    match file.read_at(&mut scratch[..n], (off + done) as u64) {
                        Ok(0) | Err(_) => break,
                        Ok(k) => done += k,
                    }
                }
            });
        }
    });
}

impl Pool {
    /// `budget` records over the mapping at `base`; `copy` selects the
    /// wired-copy backing (allocation may fail: the caller shrinks).
    /// `slot_stride` is the pitch of a pool slot, which is the low-bit
    /// record stride when every record is low-bit (more slots in the same
    /// wired budget); `stride` stays the 4-bit stride for file offsets.
    pub fn new(
        ctx: &MetalContext,
        base: *const u8,
        stride: usize,
        slot_stride: usize,
        n_records: usize,
        budget: usize,
        copy: bool,
    ) -> Result<Self> {
        Ok(if copy {
            Pool::Copy(CopyPool::new(ctx, stride, slot_stride, n_records, budget)?)
        } else {
            Pool::Set(Residency::new(ctx, base, stride, n_records, budget)?)
        })
    }

    /// Attach a low-bit store (copy pool only): its file and record
    /// stride. With `default_kind` non-zero every slot holds a low-bit
    /// record, so nothing reads the 4-bit file.
    pub fn set_low_bit_store(&mut self, file: File, low_stride: usize, default_kind: u8) {
        if let Pool::Copy(c) = self {
            c.low_file = Some(file);
            c.low_stride = low_stride;
            c.default_kind = default_kind;

            if default_kind != 0 {
                for k in c.slot_kind.iter_mut() {
                    *k = default_kind;
                }
            }
        }
    }

    /// Which store a record's slot holds (0 = 4-bit, 1 = 3-bit, 2 = 2-bit); 0 when
    /// not resident or not a copy pool.
    pub fn kind(&self, rid: usize) -> u8 {
        match self {
            Pool::Copy(c) if c.rec_slot[rid] != u32::MAX => c.slot_kind[c.rec_slot[rid] as usize],
            _ => 0,
        }
    }

    /// Choose the store for a record acquired this step (before
    /// `plan_reads`); copy pool only.
    pub fn set_kind(&mut self, rid: usize, kind: u8) {
        if let Pool::Copy(c) = self
            && c.rec_slot[rid] != u32::MAX
            && (kind == 0 || c.low_file.is_some())
        {
            c.slot_kind[c.rec_slot[rid] as usize] = kind;
        }
    }

    /// Store backing a prefill ring read. Its precision is also used to
    /// select the GEMM, so compressed bytes never reach a Q4 kernel.
    pub fn store_file<'a>(&'a self, original: &'a File, kind: u8) -> &'a File {
        if kind == 0 {
            original
        } else if let Pool::Copy(c) = self {
            c.low_file.as_ref().expect("low-bit store attached")
        } else {
            unreachable!("low-bit records require the copy pool")
        }
    }

    pub fn budget(&self) -> usize {
        match self {
            Pool::Set(r) => r.budget,
            Pool::Copy(c) => c.budget,
        }
    }

    pub fn resident(&self) -> usize {
        match self {
            Pool::Set(r) => r.resident(),
            Pool::Copy(c) => c.slot_rec.iter().filter(|&&r| r != u32::MAX).count(),
        }
    }

    pub fn is_member(&self, rid: usize) -> bool {
        match self {
            Pool::Set(r) => r.is_member(rid),
            Pool::Copy(c) => c.rec_slot[rid] != u32::MAX,
        }
    }

    /// Mark records used this step and reserve room for them, evicting
    /// least recently used records not used this step. Returns the
    /// records that still have to be read (`plan_reads`, run the plan,
    /// then `finish`).
    pub fn acquire(&mut self, ctx: &MetalContext, rids: &[usize], step: u64) -> Result<Vec<usize>> {
        match self {
            Pool::Set(r) => r.acquire(ctx, rids, step),
            Pool::Copy(c) => c.acquire(rids, step),
        }
    }

    /// The reads for records returned by `acquire`, and how many of them
    /// need none (their pages are still in memory).
    pub fn plan_reads(&self, need: &[usize]) -> (ReadPlan, usize) {
        match self {
            Pool::Set(r) => {
                let mut items = Vec::new();

                for (need_index, &rid) in need.iter().enumerate() {
                    if !r.cached(rid) {
                        items.push(PlannedRead {
                            destination: 0,
                            file_offset: rid * r.stride,
                            bytes: r.stride,
                            kind: 0,
                            need_index,
                        });
                    }
                }

                let warm = need.len() - items.len();

                (
                    ReadPlan {
                        items,
                        low_file: None,
                        stride: r.stride,
                    },
                    warm,
                )
            }
            Pool::Copy(c) => {
                let items: Vec<PlannedRead> = need
                    .iter()
                    .enumerate()
                    .map(|(need_index, &rid)| {
                        let slot = c.rec_slot[rid] as usize;
                        let kind = c.slot_kind[slot];
                        let (off, len) = if kind != 0 {
                            (rid * c.low_stride, c.low_stride)
                        } else {
                            (rid * c.stride, c.stride)
                        };

                        PlannedRead {
                            destination: c.base + slot * c.slot_stride,
                            file_offset: off,
                            bytes: len,
                            kind,
                            need_index,
                        }
                    })
                    .collect();
                let low_file = if items.iter().any(|read| read.kind != 0) {
                    c.low_file.as_ref().map(|f| f.try_clone().expect("dup fd"))
                } else {
                    None
                };

                (
                    ReadPlan {
                        items,
                        low_file,
                        stride: c.stride,
                    },
                    0,
                )
            }
        }
    }

    /// Logical resident budget in bytes (low-bit slots are shorter).
    pub fn bytes(&self) -> usize {
        match self {
            Pool::Copy(c) => c.budget * c.slot_stride,
            Pool::Set(r) => r.budget * r.stride,
        }
    }

    /// Records from `acquire` whose reads are done become usable.
    pub fn finish(&mut self, ctx: &MetalContext, rids: &[usize]) -> Result<()> {
        match self {
            Pool::Set(r) => r.finish(ctx, rids),
            Pool::Copy(_) => Ok(()),
        }
    }

    /// GPU address of a usable record.
    pub fn addr(&mut self, ctx: &MetalContext, rid: usize) -> Result<u64> {
        match self {
            Pool::Set(r) => r.addr(ctx, rid),
            Pool::Copy(c) => {
                let slot = c.rec_slot[rid] as usize;
                let flag = match c.slot_kind[slot] {
                    1 => 1u64 << 63,
                    2 => 1u64 << 62,
                    _ => 0,
                };

                Ok((c.gpu_base + slot as u64 * c.slot_stride as u64) | flag)
            }
        }
    }

    /// A usable record as (buffer, byte offset) for binding.
    pub fn buf(&mut self, ctx: &MetalContext, rid: usize) -> Result<(Buf, usize)> {
        match self {
            Pool::Set(r) => Ok((r.buf(ctx, rid)?, 0)),
            Pool::Copy(c) => Ok((c.pool.clone(), c.rec_slot[rid] as usize * c.slot_stride)),
        }
    }
}

pub struct CopyPool {
    pool: Buf,
    /// This set keeps the pool resident for the queue. The GPU reaches it
    /// through addresses rather than a binding.
    _set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    base: usize,
    gpu_base: u64,
    stride: usize,
    /// Bytes per slot (the low-bit stride when every record is low-bit).
    slot_stride: usize,
    /// Record id held by each slot (u32::MAX = empty).
    slot_rec: Vec<u32>,
    /// Step number of each slot's last use (eviction guard and LRU key).
    slot_used: Vec<u64>,
    /// Slot of each record id (u32::MAX = not resident).
    rec_slot: Vec<u32>,
    /// Store held by each slot: 0 = 4-bit record, 1 = 3-bit, 2 = 2-bit
    /// (one low-bit store is attached at a time; `low_file`/`low_stride`).
    slot_kind: Vec<u8>,
    low_file: Option<File>,
    low_stride: usize,
    /// Kind a freshly taken slot gets (0 unless every record is low-bit).
    default_kind: u8,
    pub budget: usize,
}

impl CopyPool {
    fn new(
        ctx: &MetalContext,
        stride: usize,
        slot_stride: usize,
        n_records: usize,
        budget: usize,
    ) -> Result<Self> {
        let pool = ctx
            .new_buffer(budget * slot_stride)
            .context("allocating the expert pool")?;
        let desc = MTLResidencySetDescriptor::new();
        let set = ctx
            .device
            .newResidencySetWithDescriptor_error(&desc)
            .map_err(|e| anyhow::anyhow!("residency set: {e}"))?;

        set.addAllocation(ProtocolObject::from_ref(&*pool));
        set.commit();
        set.requestResidency();
        ctx.queue.addResidencySet(&set);

        Ok(CopyPool {
            base: pool.contents().cast::<u8>().as_ptr() as usize,
            gpu_base: pool.gpuAddress(),
            pool,
            _set: set,
            stride,
            slot_stride,
            slot_rec: vec![u32::MAX; budget],
            slot_used: vec![0; budget],
            rec_slot: vec![u32::MAX; n_records],
            slot_kind: vec![0; budget],
            low_file: None,
            low_stride: 0,
            default_kind: 0,
            budget,
        })
    }

    fn acquire(&mut self, rids: &[usize], step: u64) -> Result<Vec<usize>> {
        let mut need = Vec::new();

        for &rid in rids {
            let mut slot = self.rec_slot[rid];

            if slot == u32::MAX {
                // Victim: the least recently used slot not used this step.
                let mut best = u32::MAX;
                let mut best_used = u64::MAX;

                for s in 0..self.budget {
                    let u = self.slot_used[s];

                    if u == step || u >= best_used {
                        continue;
                    }

                    best_used = u;
                    best = s as u32;

                    if u == 0 {
                        break;
                    }
                }

                anyhow::ensure!(best != u32::MAX, "expert pool too small for one step");

                let old = self.slot_rec[best as usize];

                if old != u32::MAX {
                    self.rec_slot[old as usize] = u32::MAX;
                }

                self.slot_rec[best as usize] = rid as u32;
                self.slot_kind[best as usize] = self.default_kind;
                self.rec_slot[rid] = best;
                slot = best;

                need.push(rid);
            }

            self.slot_used[slot as usize] = step;
        }

        Ok(need)
    }
}

pub struct Residency {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    base: *const u8,
    stride: usize,
    bufs: Vec<Option<Buf>>,
    /// Position in `members`, or u32::MAX when not in the set.
    member_pos: Vec<u32>,
    members: Vec<u32>,
    /// Step of last use per record (LRU key; also the eviction guard).
    last_used: Vec<u64>,
    /// Records added to the set object since the last commit.
    dirty: bool,
    pub budget: usize,
}

// The set is only touched from the service thread; the mapping is
// read-only for the process lifetime.
unsafe impl Send for Residency {}

impl Residency {
    pub fn new(
        ctx: &MetalContext,
        base: *const u8,
        stride: usize,
        n_records: usize,
        budget: usize,
    ) -> Result<Self> {
        let desc = MTLResidencySetDescriptor::new();

        unsafe { desc.setInitialCapacity(budget + 256) };

        let set = ctx
            .device
            .newResidencySetWithDescriptor_error(&desc)
            .map_err(|e| anyhow::anyhow!("residency set: {e}"))?;

        ctx.queue.addResidencySet(&set);

        Ok(Residency {
            set,
            base,
            stride,
            bufs: vec![None; n_records],
            member_pos: vec![u32::MAX; n_records],
            members: Vec::with_capacity(budget),
            last_used: vec![0; n_records],
            dirty: false,
            budget,
        })
    }

    pub fn resident(&self) -> usize {
        self.members.len()
    }

    pub fn is_member(&self, rid: usize) -> bool {
        self.member_pos[rid] != u32::MAX
    }

    /// The record's sub-buffer (wrapped on first use).
    pub fn buf(&mut self, ctx: &MetalContext, rid: usize) -> Result<Buf> {
        if let Some(b) = &self.bufs[rid] {
            return Ok(b.clone());
        }

        // Safety: page-aligned record inside the live read-only mapping.
        let b =
            unsafe { ctx.wrap_region(self.base.add(rid * self.stride).cast_mut(), self.stride)? };
        self.bufs[rid] = Some(b.clone());

        Ok(b)
    }

    /// GPU address of the record.
    pub fn addr(&mut self, ctx: &MetalContext, rid: usize) -> Result<u64> {
        Ok(self.buf(ctx, rid)?.gpuAddress())
    }

    /// This method reserves membership for the step's records, evicting the
    /// least recently used members outside the step as needed. It returns
    /// records that still need to be read and added. Call `finish` once their
    /// pages are in memory.
    pub fn acquire(&mut self, ctx: &MetalContext, rids: &[usize], step: u64) -> Result<Vec<usize>> {
        let mut need = Vec::new();

        for &rid in rids {
            self.last_used[rid] = step;

            if self.is_member(rid) {
                continue;
            }

            if self.members.len() >= self.budget {
                self.evict_one(ctx, step)?;
            }

            need.push(rid);
        }

        Ok(need)
    }

    fn evict_one(&mut self, ctx: &MetalContext, step: u64) -> Result<()> {
        let mut best = usize::MAX;
        let mut best_used = u64::MAX;

        for (i, &m) in self.members.iter().enumerate() {
            let u = self.last_used[m as usize];

            if u != step && u < best_used {
                best_used = u;
                best = i;

                if u == 0 {
                    break;
                }
            }
        }

        anyhow::ensure!(
            best != usize::MAX,
            "expert residency budget too small for one step"
        );

        let victim = self.members[best] as usize;
        let last = self.members.len() - 1;

        self.members.swap(best, last);

        self.member_pos[self.members[best] as usize] = best as u32;

        self.members.pop();

        self.member_pos[victim] = u32::MAX;
        let b = self.buf(ctx, victim)?;

        self.set.removeAllocation(ProtocolObject::from_ref(&*b));

        self.dirty = true;

        Ok(())
    }

    /// Whether all of the record's pages are in memory (page cache or
    /// wired), so no read is needed before adding it.
    pub fn cached(&self, rid: usize) -> bool {
        let pages = self.stride / 16384;
        let mut vec = vec![0u8; pages];
        let r = unsafe {
            libc::mincore(
                self.base.add(rid * self.stride) as *mut c_void,
                self.stride,
                vec.as_mut_ptr() as *mut libc::c_char,
            )
        };

        r == 0 && vec.iter().all(|v| v & 1 == 1)
    }

    /// Add records whose pages are in memory to the set and commit.
    pub fn finish(&mut self, ctx: &MetalContext, rids: &[usize]) -> Result<()> {
        for &rid in rids {
            if self.is_member(rid) {
                continue;
            }

            let b = self.buf(ctx, rid)?;

            self.set.addAllocation(ProtocolObject::from_ref(&*b));

            self.member_pos[rid] = self.members.len() as u32;

            self.members.push(rid as u32);

            self.dirty = true;
        }

        if self.dirty {
            self.set.commit();
            self.set.requestResidency();

            self.dirty = false;
        }

        Ok(())
    }
}
