//! Profiling counters (measurement only; no effect on any result). Built only with the feature `prof`; without it every
//! call here is empty and inlined away, so the released worker carries none of it. Per region: the wall time (ns) its
//! calls took, inclusive of the regions inside it, its calls, and a free count (the heap allocations made inside
//! it, inclusive, counted by the feature's allocator) -- each thread's own, flushed to
//! the process's every 256 updates and when a report is made; the report is one line per region entered, to stderr with
//! every result's counts line (adapt.rs `counted`).

/// The regions instrumented (a region whose instrumentation went with its code is gone with it; the others keep their
/// numbers, so that reports of different builds read alike).
pub const STEP_NEW: usize = 0;
pub const ATTEMPT: usize = 1;
pub const STRETCH: usize = 3;
pub const PICTURED: usize = 4;
pub const PIC_GENERATOR: usize = 5;
pub const PIC_SEPARATE: usize = 6;
pub const PIC_PREPARE: usize = 7;
pub const LAUNCH: usize = 8;
pub const AT_NODES: usize = 9;
pub const GATHER: usize = 10;
pub const ROOT: usize = 11;
pub const FALLEN: usize = 12;
pub const HAZARD_AT: usize = 13;
pub const JUMP: usize = 14;
pub const OPTICAL: usize = 15;
pub const NODES_CALL: usize = 16;
pub const LOOKED: usize = 18;
pub const STEP_OTHER_MIDDLE: usize = 19;
pub const SOLVE_TRAJECTORY: usize = 65;
pub const SOLVE_RECORD: usize = 66;
pub const MID_STEP_MIDDLE: usize = 81;
#[cfg(feature = "prof")]
pub const SLOTS: usize = 96;

#[cfg(feature = "prof")]
pub const NAMES: [&str; SLOTS] = {
    let mut n = [""; SLOTS];
    n[STEP_NEW] = "step: Midway::new (middle + exists)";
    n[ATTEMPT] = "step: attempt";
    n[STRETCH] = "step: stretch";
    n[PICTURED] = "step: pictured";
    n[PIC_GENERATOR] = "step: pictured's generator calls";
    n[PIC_SEPARATE] = "step: picture.separate (beats, envelopes)";
    n[PIC_PREPARE] = "step: picture.prepare";
    n[LAUNCH] = "step: picture.launch";
    n[AT_NODES] = "step: picture.at (the nodes and stops)";
    n[GATHER] = "step: Sums.gather";
    n[ROOT] = "step: root";
    n[FALLEN] = "step: fallen";
    n[HAZARD_AT] = "step: hazard_at";
    n[JUMP] = "step: jump (emit, land, enter)";
    n[OPTICAL] = "step: optical sector";
    n[NODES_CALL] = "step: middle.nodes (stretch)";
    n[LOOKED] = "step: looked (reads)";
    n[STEP_OTHER_MIDDLE] = "step: middle of a landed cluster";
    n[SOLVE_TRAJECTORY] = "solve: a trajectory (run)";
    n[SOLVE_RECORD] = "solve: record at outputs/reads";
    n[MID_STEP_MIDDLE] = "step: the step's middle (scattered, optical, centre)";
    n
};

#[cfg(feature = "prof")]
mod on {
    use super::{NAMES, SLOTS};
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    static NS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static CALLS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static MORE: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];

    struct Local {
        ns: [u64; SLOTS],
        calls: [u64; SLOTS],
        more: [u64; SLOTS],
        updates: u32,
    }

    impl Drop for Local {
        fn drop(&mut self) {
            flush_local(self);
        }
    }

    thread_local! {
        static LOCAL: RefCell<Local> = const { RefCell::new(Local { ns: [0; SLOTS], calls: [0; SLOTS], more: [0; SLOTS], updates: 0 }) };
    }

    pub fn add(k: usize, ns: u64, calls: u64, more: u64) {
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            l.ns[k] += ns;
            l.calls[k] += calls;
            l.more[k] += more;
            l.updates += 1;
            if l.updates >= 256 {
                flush_local(&mut l);
            }
        });
    }

    fn flush_local(l: &mut Local) {
        for k in 0..SLOTS {
            if l.calls[k] != 0 || l.ns[k] != 0 || l.more[k] != 0 {
                NS[k].fetch_add(l.ns[k], Relaxed);
                CALLS[k].fetch_add(l.calls[k], Relaxed);
                MORE[k].fetch_add(l.more[k], Relaxed);
                (l.ns[k], l.calls[k], l.more[k]) = (0, 0, 0);
            }
        }
        l.updates = 0;
    }

    pub fn flush() {
        LOCAL.with(|l| flush_local(&mut l.borrow_mut()));
    }

    // The heap allocations a thread has made (`Counting`): a region's free count is those made inside it.
    thread_local! {
        static ALLOCATED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// The system allocator, each allocation counted on its thread (measurement only: the feature's alone).
    struct Counting;
    unsafe impl std::alloc::GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            ALLOCATED.with(|c| c.set(c.get() + 1));
            unsafe { std::alloc::System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
            ALLOCATED.with(|c| c.set(c.get() + 1));
            unsafe { std::alloc::System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
            ALLOCATED.with(|c| c.set(c.get() + 1));
            unsafe { std::alloc::System.realloc(ptr, layout, size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static COUNTING: Counting = Counting;

    fn allocated() -> u64 {
        ALLOCATED.with(|c| c.get())
    }

    /// A region's guard: its start, its region, and the thread's allocations at its start -- its free count the heap
    /// allocations (and reallocations) made inside it.
    pub struct Scope(std::time::Instant, usize, u64);
    impl Drop for Scope {
        #[inline]
        fn drop(&mut self) {
            add(self.1, self.0.elapsed().as_nanos() as u64, 1, allocated() - self.2);
        }
    }

    pub fn scope(k: usize) -> Scope {
        Scope(std::time::Instant::now(), k, allocated())
    }

    pub fn report() -> String {
        flush();
        let ns: Vec<u64> = NS.iter().map(|x| x.load(Relaxed)).collect();
        let calls: Vec<u64> = CALLS.iter().map(|x| x.load(Relaxed)).collect();
        let more: Vec<u64> = MORE.iter().map(|x| x.load(Relaxed)).collect();
        let mut out = String::new();
        for k in 0..SLOTS {
            if calls[k] == 0 && ns[k] == 0 && more[k] == 0 {
                continue;
            }
            out += &format!("\nPROF {k:3} {:<58} {:>10.3} s {:>11} calls {:>10.2} µs/call more {}",NAMES[k], 1e-9 * ns[k] as f64, calls[k], 1e-3 * ns[k] as f64 / calls[k].max(1) as f64, more[k]);
        }
        out
    }
}

/// A guard timing region k (nothing without the feature).
#[cfg(feature = "prof")]
pub use on::Scope;
#[cfg(not(feature = "prof"))]
pub struct Scope;

#[cfg(feature = "prof")]
#[inline]
pub fn scope(k: usize) -> Scope {
    on::scope(k)
}
#[cfg(not(feature = "prof"))]
#[inline(always)]
pub fn scope(_k: usize) -> Scope {
    Scope
}

/// A note to stderr (the feature's alone).
#[inline(always)]
pub fn note(_what: impl FnOnce() -> String) {
    #[cfg(feature = "prof")]
    eprintln!("PROFNOTE {}", _what());
}

/// The report, empty without the feature.
pub fn report() -> String {
    #[cfg(feature = "prof")]
    return on::report();
    #[cfg(not(feature = "prof"))]
    String::new()
}
