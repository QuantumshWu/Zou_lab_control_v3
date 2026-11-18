//! Profiling counters (measurement only; no effect on any result). Built only with the feature `prof`; without it every
//! call here is empty and inlined away, so the released worker carries none of it. Per region: the wall time (ns) its
//! calls took, inclusive of the regions inside it, its calls, and a free count (`more`: the heap allocations made inside
//! it, inclusive, counted by the feature's allocator) -- each thread's own, flushed to
//! the process's every 256 updates and when a report is made; the report is one line per region entered, to stderr with
//! every result's counts line (adapt.rs `counted`).

/// The regions.
pub const STEP_NEW: usize = 0;
pub const ATTEMPT: usize = 1;
pub const DECIDED: usize = 2;
pub const STRETCH: usize = 3;
pub const PICTURED: usize = 4;
pub const PIC_GENERATOR: usize = 5;
pub const PIC_SEPARATE: usize = 6;
pub const PIC_PREPARE: usize = 7;
pub const LAUNCH: usize = 8;
pub const ADVANCE: usize = 9;
pub const GATHER: usize = 10;
pub const ROOT: usize = 11;
pub const FALLEN: usize = 12;
pub const HAZARD_AT: usize = 13;
pub const JUMP: usize = 14;
pub const OPTICAL: usize = 15;
pub const NODES_CALL: usize = 16;
pub const SCATTER_CALL: usize = 17;
pub const LOOKED: usize = 18;
pub const STEP_OTHER_MIDDLE: usize = 19;
pub const MID_FORWARD: usize = 20;
pub const MID_FWD_DECIDE: usize = 21;
pub const MID_P_NODES: usize = 22;
pub const MID_P_TANGENT: usize = 23;
pub const MID_P_FORCE: usize = 24;
pub const MID_GENERATOR: usize = 25;
pub const MID_ENVELOPES: usize = 26;
pub const MID_BEATS: usize = 27;
pub const MID_NODES: usize = 28;
pub const MID_FORCE: usize = 29;
pub const MID_TONE_FORCE: usize = 30;
pub const MID_TONES: usize = 31;
pub const MID_SCATTERING: usize = 32;
pub const MID_OPTICAL: usize = 33;
pub const MID_GRADIENT: usize = 34;
pub const MID_BOOKS: usize = 35;
pub const MID_EXISTS: usize = 36;
pub const MID_DECIDE: usize = 37;
pub const MID_EMITTED: usize = 38;
pub const MID_TRANSIENT: usize = 39;
pub const MID_LANDING: usize = 40;
pub const LAND_TAKEN: usize = 41;
pub const LAND_CLOCK: usize = 42;
pub const FOLDED_PART: usize = 43;
pub const P_JUMPS: usize = 44;
pub const P_PLACE: usize = 45;
pub const P_PHOTONS: usize = 46;
pub const DYN_MIDDLE: usize = 47;
pub const DYN_EMIT: usize = 48;
pub const DYN_TRANSIENT: usize = 49;
pub const DYN_FORMS: usize = 50;
pub const DYN_STARTED: usize = 51;
pub const DYN_LANDED: usize = 52;
pub const DYN_ENERGY: usize = 53;
pub const DYN_LOST: usize = 54;
pub const DYN_READOUT: usize = 55;
pub const DYN_ENTERED: usize = 56;
pub const TR_MODES: usize = 57;
pub const TR_CROSSING: usize = 58;
pub const TR_FORCES: usize = 59;
pub const TR_MODE_FORCES: usize = 60;
pub const TR_LEFT: usize = 61;
pub const TR_KICK: usize = 62;
pub const TR_FAST_PHOTON: usize = 63;
pub const TR_CONDITIONAL: usize = 64;
pub const SOLVE_TRAJECTORY: usize = 65;
pub const SOLVE_RECORD: usize = 66;
pub const FWD_HELD: usize = 67;
pub const FWD_DECIDING: usize = 68;
pub const FWD_BASES: usize = 69;
pub const FWD_NEAR_BASIS: usize = 70;
pub const FWD_FAR_BASIS: usize = 71;
pub const FWD_ACCOUNTS: usize = 72;
pub const FWD_SWEEPS: usize = 73;
pub const NB_SCHUR: usize = 74;
pub const NB_TERMS: usize = 75;
pub const NB_REST: usize = 76;
pub const SP_FOLLOW: usize = 77;
pub const SP_HOP: usize = 78;
pub const SP_STRIDE: usize = 79;
pub const SP_BETWEEN: usize = 80;
pub const MID_STEP_MIDDLE: usize = 81;
pub const LAND_RESPONSES: usize = 82;
pub const MID_READOUT: usize = 83;
pub const SLOTS: usize = 96;

pub const NAMES: [&str; SLOTS] = {
    let mut n = [""; SLOTS];
    n[STEP_NEW] = "step: Midway::new (middle + exists)";
    n[ATTEMPT] = "step: attempt";
    n[DECIDED] = "step: decided (cell end)";
    n[STRETCH] = "step: stretch";
    n[PICTURED] = "step: pictured";
    n[PIC_GENERATOR] = "step: pictured's generator calls";
    n[PIC_SEPARATE] = "step: picture.separate (beats, envelopes)";
    n[PIC_PREPARE] = "step: picture.prepare";
    n[LAUNCH] = "step: picture.launch";
    n[ADVANCE] = "step: picture.advance (nodes)";
    n[GATHER] = "step: Sums.gather";
    n[ROOT] = "step: root";
    n[FALLEN] = "step: fallen";
    n[HAZARD_AT] = "step: hazard_at";
    n[JUMP] = "step: jump (emit, land, enter)";
    n[OPTICAL] = "step: optical sector";
    n[NODES_CALL] = "step: middle.nodes (stretch)";
    n[SCATTER_CALL] = "step: middle.scattering (stretch)";
    n[LOOKED] = "step: looked (reads)";
    n[STEP_OTHER_MIDDLE] = "step: middle of a landed cluster";
    n[MID_FORWARD] = "run: Middle::forward (made)";
    n[MID_FWD_DECIDE] = "run: Middle::forward deciding";
    n[MID_P_NODES] = "run: prototype_nodes";
    n[MID_P_TANGENT] = "run: prototype_tangent";
    n[MID_P_FORCE] = "run: prototype_force (forms)";
    n[MID_GENERATOR] = "run: Middle::generator";
    n[MID_ENVELOPES] = "run: Middle::envelopes";
    n[MID_BEATS] = "run: Middle::beats";
    n[MID_NODES] = "run: Middle::nodes";
    n[MID_FORCE] = "run: Middle::force";
    n[MID_TONE_FORCE] = "run: Middle::tone_force";
    n[MID_TONES] = "run: Middle::tones";
    n[MID_SCATTERING] = "run: Middle::scattering";
    n[MID_OPTICAL] = "run: Middle::optical";
    n[MID_GRADIENT] = "run: Middle::gradient";
    n[MID_BOOKS] = "run: Middle::books";
    n[MID_EXISTS] = "run: Middle::exists";
    n[MID_DECIDE] = "run: Middle::decide";
    n[MID_EMITTED] = "run: Middle::emitted";
    n[MID_TRANSIENT] = "run: Middle::transient";
    n[MID_LANDING] = "run: Middle::landing";
    n[LAND_TAKEN] = "run: landing ws.taken";
    n[LAND_CLOCK] = "run: landing ws.clock";
    n[FOLDED_PART] = "run: folded_part (§7.6)";
    n[P_JUMPS] = "proto: prototype_excited (cross decays)";
    n[P_PLACE] = "proto: prototype_place";
    n[P_PHOTONS] = "proto: prototype_photons";
    n[DYN_MIDDLE] = "run: Dynamics::middle";
    n[DYN_EMIT] = "run: Dynamics::emit";
    n[DYN_TRANSIENT] = "run: Dynamics::transient";
    n[DYN_FORMS] = "run: Dynamics::forms";
    n[DYN_STARTED] = "run: Dynamics::started";
    n[DYN_LANDED] = "run: Dynamics::landed";
    n[DYN_ENERGY] = "run: Dynamics::energy";
    n[DYN_LOST] = "run: Dynamics::lost";
    n[DYN_READOUT] = "run: Dynamics::readout";
    n[DYN_ENTERED] = "run: Dynamics::entered";
    n[TR_MODES] = "transient: modes (Transient::of)";
    n[TR_CROSSING] = "transient: crossing";
    n[TR_FORCES] = "transient: transient_forces";
    n[TR_MODE_FORCES] = "transient: mode_forces";
    n[TR_LEFT] = "transient: left_tangent";
    n[TR_KICK] = "transient: K_inf loop";
    n[TR_FAST_PHOTON] = "transient: fast_photon";
    n[TR_CONDITIONAL] = "transient: conditional";
    n[SOLVE_TRAJECTORY] = "solve: a trajectory (run)";
    n[SOLVE_RECORD] = "solve: record at outputs/reads";
    n[FWD_HELD] = "proto: forward held";
    n[FWD_DECIDING] = "proto: forward deciding";
    n[FWD_BASES] = "proto: bases (refresh)";
    n[FWD_NEAR_BASIS] = "proto: near_basis";
    n[FWD_FAR_BASIS] = "proto: far_basis";
    n[FWD_ACCOUNTS] = "proto: accounts";
    n[FWD_SWEEPS] = "proto: sweeps";
    n[NB_SCHUR] = "near_basis: K and its Schur form";
    n[NB_TERMS] = "near_basis: per-term forms";
    n[NB_REST] = "near_basis: the rest";
    n[SP_FOLLOW] = "spectral: follow";
    n[SP_HOP] = "spectral: hop";
    n[SP_STRIDE] = "spectral: stride";
    n[SP_BETWEEN] = "spectral: between";
    n[MID_STEP_MIDDLE] = "step: the step's middle (scattered, optical, centre)";
    n[LAND_RESPONSES] = "run: landing left responses";
    n[MID_READOUT] = "run: Middle::readout (a read on the step)";
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

    /// The heap allocations a thread has made (`Counting`): a region's free count is those made inside it.
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

/// Adds to region k a free count.
#[inline(always)]
pub fn more(_k: usize, _more: u64) {
    #[cfg(feature = "prof")]
    on::add(_k, 0, 0, _more);
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
