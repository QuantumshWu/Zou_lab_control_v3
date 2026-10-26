//! Persistent worker pool for the population kernels.
//!
//! Threading policy: work is split by output modes into contiguous chunks,
//! one task per worker; each task owns its packing scratch and calls BLAS
//! single-threaded (the population core loads OpenBLAS with one thread, see
//! ATOM_POPULATION_BLAS_THREADS), so BLAS and the pool never oversubscribe.
//! Every output entry is accumulated by exactly one task. Small stages run
//! serially on the calling thread.
//!
//! ATOM_PENCIL_THREADS sets the pool size (1..=32). When it is unset the
//! pool uses the available hardware threads, capped at 8.
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

/// Below this many complex multiply-adds per task a stage stays serial.
/// Scheduling threshold only; it never changes which arithmetic is done.
const MINIMUM_WORK_PER_TASK: usize = 262_144;
const DEFAULT_THREAD_CAP: usize = 8;

pub struct Execution {
    threads: usize,
    #[cfg(feature = "parallel")]
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
        #[cfg(not(feature = "parallel"))]
        if threads != 1 {
            return Err("Parallel execution was requested but this build lacks feature parallel".into());
        }
        #[cfg(feature = "parallel")]
        let pool = if threads == 1 { None } else { Some(shared_pool(threads)?) };
        Ok(Self {
            threads,
            #[cfg(feature = "parallel")]
            pool,
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Number of tasks for `items` independent output items and `work`
    /// complex multiply-adds.
    pub fn tasks(&self, items: usize, work: usize) -> usize {
        self.threads.min(items).min(work / MINIMUM_WORK_PER_TASK).max(1)
    }

    /// Split `out` into contiguous chunks of whole modes (`width` entries per
    /// mode) and run `action(first_mode, chunk, scratch)` on each, in parallel
    /// when `work` justifies it. `scratch` keeps one private buffer per task
    /// across calls. Returns the first error of any task.
    pub fn for_mode_chunks<T, S, F>(&self, out: &mut [T], width: usize, work: usize, scratch: &mut Vec<S>, action: F) -> Result<(), String>
    where
        T: Send,
        S: Default + Send,
        F: Fn(usize, &mut [T], &mut S) -> Result<(), String> + Sync,
    {
        let modes = if width == 0 { 0 } else { out.len() / width };
        let tasks = self.tasks(modes, work);
        if scratch.len() < tasks {
            scratch.resize_with(tasks, S::default);
        }
        #[cfg(feature = "parallel")]
        if let (Some(pool), true) = (&self.pool, tasks > 1) {
            let per_task = modes.div_ceil(tasks);
            return pool.install(|| {
                out.par_chunks_mut(per_task * width)
                    .zip(scratch[..tasks].par_iter_mut())
                    .enumerate()
                    .try_for_each(|(task, (chunk, scratch))| action(task * per_task, chunk, scratch))
            });
        }
        action(0, out, &mut scratch[0])
    }

    /// Run `action(item, scratch)` for `items` independent items, in parallel
    /// when `work` justifies it; items are split into contiguous ranges.
    pub fn for_ranges<S, F>(&self, items: usize, work: usize, scratch: &mut Vec<S>, action: F) -> Result<(), String>
    where
        S: Default + Send,
        F: Fn(std::ops::Range<usize>, &mut S) -> Result<(), String> + Sync,
    {
        let tasks = self.tasks(items, work);
        if scratch.len() < tasks {
            scratch.resize_with(tasks, S::default);
        }
        #[cfg(feature = "parallel")]
        if let (Some(pool), true) = (&self.pool, tasks > 1) {
            let per_task = items.div_ceil(tasks);
            return pool.install(|| {
                scratch[..tasks].par_iter_mut().enumerate().try_for_each(|(task, scratch)| {
                    let start = (task * per_task).min(items);
                    action(start..(start + per_task).min(items), scratch)
                })
            });
        }
        action(0..items, &mut scratch[0])
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, DEFAULT_THREAD_CAP)
}

/// Workers are shared by every plan with the same size, not spawned per
/// apply. The cache holds execution infrastructure only, no numerical data.
#[cfg(feature = "parallel")]
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

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    #[test]
    fn mode_chunks_cover_every_mode_once_and_keep_scratch() {
        let execution = Execution::with_threads(4).unwrap();
        let mut scratch: Vec<Vec<usize>> = Vec::new();
        let mut out = vec![0usize; 10 * 3];
        execution
            .for_mode_chunks(&mut out, 3, 4 * MINIMUM_WORK_PER_TASK, &mut scratch, |first, chunk, seen| {
                for (k, mode) in chunk.chunks_exact_mut(3).enumerate() {
                    mode.fill(first + k);
                    seen.push(first + k);
                }
                Ok(())
            })
            .unwrap();
        let expected: Vec<usize> = (0..10).flat_map(|m| [m; 3]).collect();
        assert_eq!(out, expected);
        assert_eq!(scratch.len(), 4);
        let mut all: Vec<usize> = scratch.concat();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        // Too little work: one serial task.
        let mut small = vec![0usize; 8];
        execution.for_mode_chunks(&mut small, 1, 8, &mut scratch, |first, chunk, _| {
            assert_eq!((first, chunk.len()), (0, 8));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn ranges_report_errors() {
        let execution = Execution::with_threads(3).unwrap();
        let mut scratch: Vec<()> = Vec::new();
        let result = execution.for_ranges(9, 3 * MINIMUM_WORK_PER_TASK, &mut scratch, |range, _| {
            if range.contains(&4) { Err("item 4".into()) } else { Ok(()) }
        });
        assert_eq!(result, Err("item 4".to_string()));
    }
}
