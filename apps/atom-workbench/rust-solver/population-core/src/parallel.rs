//! Optional persistent output-partition executor. No coefficient, physical
//! parameter or previously solved state affects the eligibility decision.
use std::sync::atomic::{AtomicU64,Ordering};
#[cfg(feature="parallel")]
use std::{collections::HashMap,sync::{Arc,Mutex,OnceLock}};
#[cfg(feature="parallel")]
use rayon::prelude::*;

// Scheduling threshold, not an accuracy parameter. Below this amount the
// same native serial arithmetic is used to avoid dispatch dominating work.
const MINIMUM_MACS_PER_TASK:usize=262_144;

#[repr(C)]
#[derive(Clone,Copy,Default)]
pub struct ParallelStats {pub configured_threads:u64,pub parallel_stages:u64,pub serial_stages:u64,pub minimum_macs_per_task:u64}
pub struct Execution {
    threads:usize,
    #[cfg(feature="parallel")]
    pool:Option<Arc<rayon::ThreadPool>>,
    parallel_stages:AtomicU64,
    serial_stages:AtomicU64,
}
impl Execution {
    pub fn new()->Result<Self,String>{
        let threads=match std::env::var("ATOM_PENCIL_THREADS"){
            Ok(value)=>value.parse::<usize>().map_err(|_|"ATOM_PENCIL_THREADS must be an integer in 1..32")?,
            Err(std::env::VarError::NotPresent)=>1,
            Err(_)=>return Err("Invalid ATOM_PENCIL_THREADS environment value".into()),
        };
        Self::with_threads(threads)
    }
    pub(crate) fn with_threads(threads:usize)->Result<Self,String>{
        if !(1..=32).contains(&threads){return Err("ATOM_PENCIL_THREADS must be in 1..32".into());}
        #[cfg(not(feature="parallel"))]
        if threads!=1{return Err("Parallel execution was requested but this DLL lacks feature parallel".into());}
        #[cfg(feature="parallel")]
        let pool=if threads==1{None}else{
            // Workers are shared by handles/domains, not spawned per apply.
            // This cache contains execution infrastructure only, no factors,
            // input fields, model identities or simulation answers.
            static POOLS:OnceLock<Mutex<HashMap<usize,Arc<rayon::ThreadPool>>>>=OnceLock::new();
            let mut pools=POOLS.get_or_init(||Mutex::new(HashMap::new())).lock().map_err(|_|"Persistent worker pool lock poisoned")?;
            if let Some(pool)=pools.get(&threads){Some(Arc::clone(pool))}else{
                let pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads)
                    .thread_name(move |i|format!("awb-pencil-{threads}-{i}"))
                    .build().map_err(|e|format!("Cannot prepare output workers: {e}"))?);
                pools.insert(threads,Arc::clone(&pool));Some(pool)
            }
        };
        Ok(Self{threads,#[cfg(feature="parallel")]pool,parallel_stages:AtomicU64::new(0),serial_stages:AtomicU64::new(0)})
    }
    pub fn output_chunks<T,F>(&self,out:&mut[T],mode_width:usize,macs:usize,action:F)
    where T:Send,F:Fn(usize,&mut[T])+Send+Sync {
        #[cfg(feature="parallel")]
        if let Some(pool)=&self.pool{
            let modes=if mode_width==0{0}else{out.len()/mode_width};
            let tasks=self.threads.min(modes).min(macs/MINIMUM_MACS_PER_TASK);
            if tasks>=2{
                let chunk_modes=modes.div_ceil(tasks);
                self.parallel_stages.fetch_add(1,Ordering::Relaxed);
                pool.install(||out.par_chunks_mut(chunk_modes*mode_width).enumerate()
                    .for_each(|(part,slice)|action(part*chunk_modes,slice)));
                return;
            }
        }
        let _=(mode_width,macs);
        self.serial_stages.fetch_add(1,Ordering::Relaxed);action(0,out);
    }
    pub fn mark_serial(&self){self.serial_stages.fetch_add(1,Ordering::Relaxed);}
    pub fn stats(&self)->ParallelStats{ParallelStats{configured_threads:self.threads as u64,
        parallel_stages:self.parallel_stages.load(Ordering::Relaxed),serial_stages:self.serial_stages.load(Ordering::Relaxed),
        minimum_macs_per_task:MINIMUM_MACS_PER_TASK as u64}}
}
