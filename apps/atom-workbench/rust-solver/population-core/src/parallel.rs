//! Persistent worker pool for the population kernels.
//!
//! Threading policy: work is split by output modes into contiguous chunks
//! (a few per worker, balanced by work stealing); each task owns its packing
//! scratch and calls BLAS
//! single-threaded (`Lapack::load` fixes OpenBLAS at one thread), so BLAS
//! and the pool never oversubscribe.
//! Every output entry is accumulated by exactly one task. Small stages run
//! serially on the calling thread.
//!
//! ATOM_PENCIL_THREADS sets the pool size (1..=32). When it is unset the
//! pool uses the available hardware threads, capped at 8.
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

/// Default minimum complex multiply-adds per task; below it a stage stays
/// serial. Scheduling threshold only; it never changes the arithmetic.
pub const MINIMUM_WORK_PER_TASK: usize = 262_144;
/// Tasks per worker, so work stealing evens out uneven chunks and workers
/// that are preempted or run on slower cores.
const TASKS_PER_THREAD: usize = 4;
const DEFAULT_THREAD_CAP: usize = 8;

pub struct Execution {
    threads: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl Execution {
    pub fn new() -> Result<Self, String> {
        let threads = match std::env::var("ATOM_PENCIL_THREADS") {
            Ok(value) => value.parse::<usize>().map_err(|_| "ATOM_PENCIL_THREADS must be an integer in 1..32")?,
            Err(std::env::VarError::NotPresent) => default_threads(),
            Err(_) => return Err("Invalid ATOM_PENCIL_THREADS environment value".into()),
        };
        Self::with_threads(threads)
    }

    pub(crate) fn with_threads(threads: usize) -> Result<Self, String> {
        if !(1..=32).contains(&threads) {
            return Err("ATOM_PENCIL_THREADS must be in 1..32".into());
        }
        let pool = if threads == 1 { None } else { Some(shared_pool(threads)?) };
        Ok(Self { threads, pool })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `a` and `b` concurrently on the pool (serially without one). Both
    /// stay on pool workers, which already own their BLAS buffers.
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        match &self.pool {
            Some(pool) => pool.install(|| rayon::join(a, b)),
            None => (a(), b()),
        }
    }

    /// Number of tasks for `items` independent output items, `work` complex
    /// multiply-adds in total and at least `minimum_work` per task.
    pub fn tasks(&self, items: usize, work: usize, minimum_work: usize) -> usize {
        if self.threads == 1 {
            return 1;
        }
        (self.threads * TASKS_PER_THREAD).min(items).min(work / minimum_work.max(1)).max(1)
    }

    /// Split `out` into contiguous chunks of whole modes (`width` entries per
    /// mode) and run `action(first_mode, chunk, scratch)` on each, in parallel
    /// when `work` allows tasks of at least `minimum_work`. Scratch buffers are kept in `scratch` across
    /// calls; a running task borrows one, so at most one exists per worker.
    /// Returns the first error of any task.
    pub fn for_mode_chunks<T, S, F>(
        &self,
        out: &mut [T],
        width: usize,
        work: usize,
        minimum_work: usize,
        scratch: &mut Vec<S>,
        action: F,
    ) -> Result<(), String>
    where
        T: Send,
        S: Default + Send,
        F: Fn(usize, &mut [T], &mut S) -> Result<(), String> + Sync,
    {
        let modes = if width == 0 { 0 } else { out.len() / width };
        let tasks = self.tasks(modes, work, minimum_work);
        if let (Some(pool), true) = (&self.pool, tasks > 1) {
            let per_task = modes.div_ceil(tasks);
            let idle = Mutex::new(std::mem::take(scratch));
            let borrow = || idle.lock().ok().and_then(|mut idle| idle.pop()).unwrap_or_default();
            let result = pool.install(|| {
                out.par_chunks_mut(per_task * width).enumerate().try_for_each(|(task, chunk)| {
                    let mut own = borrow();
                    let result = action(task * per_task, chunk, &mut own);
                    if let Ok(mut idle) = idle.lock() {
                        idle.push(own);
                    }
                    result
                })
            });
            *scratch = idle.into_inner().unwrap_or_default();
            return result;
        }
        if scratch.is_empty() {
            scratch.push(S::default());
        }
        action(0, out, &mut scratch[0])
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, DEFAULT_THREAD_CAP)
}

/// Workers are shared by every plan with the same size, not spawned per
/// apply. The cache holds execution infrastructure only, no numerical data.
fn shared_pool(threads: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS.get_or_init(|| Mutex::new(HashMap::new())).lock().map_err(|_| "Worker pool lock poisoned")?;
    if let Some(pool) = pools.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("awb-pencil-{threads}-{i}"))
        .build()
        .map_err(|e| format!("Cannot prepare pencil workers: {e}"))?;
    let pool = Arc::new(pool);
    pools.insert(threads, Arc::clone(&pool));
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    const WORK: usize = MINIMUM_WORK_PER_TASK;

    #[test]
    fn mode_chunks_cover_every_mode_once_and_keep_scratch() {
        let execution = Execution::with_threads(4).unwrap();
        let mut scratch: Vec<Vec<usize>> = Vec::new();
        let mut out = vec![0usize; 10 * 3];
        execution
            .for_mode_chunks(&mut out, 3, 4 * WORK, WORK, &mut scratch, |first, chunk, seen| {
                for (k, mode) in chunk.chunks_exact_mut(3).enumerate() {
                    mode.fill(first + k);
                    seen.push(first + k);
                }
                Ok(())
            })
            .unwrap();
        let expected: Vec<usize> = (0..10).flat_map(|m| [m; 3]).collect();
        assert_eq!(out, expected);
        assert!(!scratch.is_empty() && scratch.len() <= 4);
        let mut all: Vec<usize> = scratch.concat();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        // Too little work: one serial task.
        let mut small = vec![0usize; 8];
        execution.for_mode_chunks(&mut small, 1, 8, WORK, &mut scratch, |first, chunk, _| {
            assert_eq!((first, chunk.len()), (0, 8));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn mode_chunks_report_errors() {
        let execution = Execution::with_threads(3).unwrap();
        let mut scratch: Vec<()> = Vec::new();
        let mut out = vec![0u8; 9];
        let result = execution.for_mode_chunks(&mut out, 1, 9 * WORK, WORK, &mut scratch, |first, chunk, _| {
            if (first..first + chunk.len()).contains(&4) { Err("mode 4".into()) } else { Ok(()) }
        });
        assert_eq!(result, Err("mode 4".to_string()));
    }
}
