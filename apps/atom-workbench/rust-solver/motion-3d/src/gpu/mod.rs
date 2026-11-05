//! The trajectories on an NVIDIA GPU: a batch of lib.rs `simulate` -- the same groups, tasks,
//! initial states, random streams and sums -- run by the kernel of kernels/motion3d.cu, its model
//! flattened by `program`, through the driver API alone (`driver`), with the PTX embedded (`ptx`).
//!
//! # Device
//! The GPU is an NVIDIA one of compute capability 7.5 (Turing) or newer with a driver of CUDA 12.0
//! (R525, Windows 527.41) or newer, the one of the most multiprocessors. Without one -- no driver,
//! an older one, no such GPU, a module the driver cannot compile -- `gpu` says why in Chinese and
//! the solve runs on the CPU with that note (solve.rs).
//!
//! # Context
//! The process makes the CUDA context and loads the module once, in the background (`prepare`: the
//! worker's `gpu` operation, which waits for it, or a solve at its start), and keeps them for every later solve.
//! The driver compiles the PTX for the GPU at its first load on a machine, once per version of the
//! kernels (about a minute on a laptop's GPU), and caches the result (`Startup` records the times). A
//! solve asked for the GPU waits for it to start (`gpu`): the first compile is waited out.
//!
//! # Batches
//! The initial states are drawn on the CPU (lib.rs `initial_trajectory` and a held twin's start,
//! exactly); the tasks run as blocks, as many at once as the device memory comfortably holds (a
//! wave), each launch of about `LAUNCH_MS` (well inside the display watchdog) by a cycle budget: every trajectory on
//! from where it stands, on its own, at most `MOST_INTERVALS` past the first interval not yet summed, and only as far as its group's last output (lib.rs `simulate_to`); a warp of cheap steps
//! does not wait for one of dear ones. The intervals every slot has passed are summed per task after the launch; the
//! tasks' sums come back per output and are merged per group in order, as the CPU's are.
pub mod driver;
pub mod program;
pub mod ptx;

use crate::engine::{propagate,Settings,Trajectory};
use crate::model::Model;
use crate::stats::{Group,Sums};
use crate::{bare_trajectory,dressed,Complex,Initial,Outcome,Stratum,Variates,STATES,TASK};
use driver::{Buffer,Context,DeviceInfo,Event,Function,HostWords,Module,Stream};
use program::{Program,MAXM,MAXN,MOST_LISTED};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64,Ordering};
use std::sync::{Condvar,Mutex};
use std::time::Instant;

/// The oldest GPUs the PTX targets: compute capability 7.5 (Turing).
pub const MINIMUM_CAPABILITY:(i32,i32)=(7,5);
/// A launch's target wall time (ms).
const LAUNCH_MS:f64=60.;
/// A block's warps (the kernel's WARPS: lib.rs `TASK`), an output's sums (its RECORD), a trajectory's
/// record of an output interval (its SLOT_RECORD: the point record and per lab axis two closing
/// windows' held flag and means) and the most intervals of a launch (its MOST_INTERVALS: the room of
/// each trajectory's records, which the block sums at the launch's end).
pub(crate) const WARPS:usize=TASK as usize;
pub(crate) const RECORD:usize=85;
pub(crate) const SLOT_RECORD:usize=70+3*2*6;
pub(crate) const MOST_INTERVALS:usize=64;
/// Where an output's sums (the kernel's record) keep the held twin's, the folded excited population's,
/// the motion's and the harmonic control's (`sums_of`).
pub(crate) const TWIN:usize=35;
pub(crate) const FOLDED:usize=69;
pub(crate) const MOTION:usize=70;
pub(crate) const HARMONIC:usize=76;
/// The profiled kernel's counters per trajectory (its SPANS): the cycles of the point, the generator,
/// the exponential, the evolution with the kick, the jumps -- of them the exact responses, the
/// photons' matrices with the classes' weights and the absorbed light --, the loss, the steps, the
/// readouts, the windows, the held twin's steps and records, and the launches; the advances, the jumps and
/// the jumps with the exact responses; of the jumps the point, the draws, the emission's direction and polarization,
/// the kicks and the transient with the slow start; of the photons the matrices and the classes' weights, of the absorbed light
/// its items and its beams; an exact evolution's nodes and products by a matrix, and of an exact step the segment's
/// components and the panels' forces; a launch's load and store of the warp's state, and the launches.
pub const SPANS:[&str;32]=["point","generator","exponential","evolve","jump","exact","photons","absorption","loss","step","readout","windows","twin","total","advances","jumps","exact jumps","jump point","draw","emit","kick","transient","items","beams","matrices","weights","nodes","products","components","forces","checkpoint","launches"];

/// The times the process's GPU took to start (ms): the driver, the context, the module (the driver's
/// JIT compile of the PTX at its first load on a machine, its cache after).
#[derive(Clone,Copy,Debug)]
pub struct Startup{pub driver_ms:f64,pub context_ms:f64,pub module_ms:f64}

/// The process's GPU: its context, the kernel, a stream.
pub struct Gpu{
    pub info:DeviceInfo,
    pub startup:Startup,
    pub driver_version:i32,
    motion:Function,
    /// Where the module keeps the program its kernels read (the kernel's `P`).
    program:driver::DevicePtr,
    stream:Stream,
    blocks_per_multiprocessor:i32,
    merge:Function,
    sums:Function,
    /// The kernel was compiled with PROFILE: its counters summed over every trajectory run.
    profiled:Option<Mutex<Vec<u64>>>,
    /// The motion kernel's launches, each waited for, summed (us): the solves' time on the GPU.
    kernel_us:AtomicU64,
    module:Module,
    context:Context,
}

/// The process's GPU: not asked for yet, starting (in the background), or made -- the GPU, or why
/// there is none (Chinese).
enum Start{Idle,Starting,Done(Result<&'static Mutex<Gpu>,String>)}
static START:Mutex<Start>=Mutex::new(Start::Idle);
static STARTED:Condvar=Condvar::new();

/// Starts the process's GPU in the background, once.
pub fn prepare(){
    let mut start=START.lock().unwrap_or_else(|e|e.into_inner());
    if matches!(*start,Start::Idle){
        *start=Start::Starting;
        std::thread::spawn(||{
            let made=Gpu::new().map(|g|&*Box::leak(Box::new(Mutex::new(g))));
            *START.lock().unwrap_or_else(|e|e.into_inner())=Start::Done(made);STARTED.notify_all();
        });
    }
}
/// The process's GPU, started if it was not, waiting for it to start (the driver's first compile of the
/// kernels included); Err: why there is none (Chinese).
pub fn gpu()->Result<&'static Mutex<Gpu>,String>{
    prepare();
    let start=STARTED.wait_while(START.lock().unwrap_or_else(|e|e.into_inner()),|s|matches!(s,Start::Starting)).unwrap_or_else(|e|e.into_inner());
    match &*start{Start::Done(made)=>made.clone(),_=>unreachable!("`prepare` has started the GPU")}
}
/// Makes the process's GPU from PTX compiled with PROFILE (examples/profile.rs), before anything
/// starts it: its solves then sum the kernel's counters (`Gpu::spans`).
pub fn install_profiled(ptx:&str)->Result<(),String>{
    let mut start=START.lock().unwrap_or_else(|e|e.into_inner());
    if !matches!(*start,Start::Idle){return Err("the GPU has started already".into());}
    *start=Start::Done(Gpu::with(ptx,true).map(|g|&*Box::leak(Box::new(Mutex::new(g)))));
    match &*start{Start::Done(Err(e))=>Err(e.clone()),_=>Ok(())}
}

impl Gpu{
    fn new()->Result<Gpu,String>{Gpu::with(&ptx::ptx(),false)}
    fn with(ptx:&str,profiled:bool)->Result<Gpu,String>{
        let clock=Instant::now();
        let version=driver::driver_version()?;
        if version<ptx::DRIVER_VERSION{return Err(format!("NVIDIA 驱动太旧（CUDA {}.{}；需要 CUDA 12.0，即 R525 / 527.41 或更新）",version/1000,version%1000/10));}
        let devices=driver::devices()?;
        let device=devices.iter().filter(|d|d.compute_capability>=MINIMUM_CAPABILITY).max_by_key(|d|d.multiprocessors).ok_or_else(||if devices.is_empty(){"没有 NVIDIA 显卡".to_string()}else{
            format!("{} 太旧（需要计算能力 7.5 或更新）",devices.iter().map(|d|format!("{}（计算能力 {}.{}）",d.name,d.compute_capability.0,d.compute_capability.1)).collect::<Vec<_>>().join("、"))})?;
        let driver_ms=clock.elapsed().as_secs_f64()*1e3;
        let clock=Instant::now();let context=Context::new(device)?;let context_ms=clock.elapsed().as_secs_f64()*1e3;
        let clock=Instant::now();
        let module=context.load(ptx).map_err(|e|format!("驱动无法编译 GPU 内核（{e}）"))?;
        let motion=module.function("motion3d")?;motion.prefer_shared()?;let merge=module.function("merge")?;let sums=module.function("task_sums")?;let program=module.global("P")?;
        let module_ms=clock.elapsed().as_secs_f64()*1e3;
        let blocks_per_multiprocessor=(motion.occupancy(32)?/WARPS as i32).max(1);
        Ok(Gpu{info:device.clone(),startup:Startup{driver_ms,context_ms,module_ms},driver_version:version,motion,program,stream:Stream::new()?,blocks_per_multiprocessor,merge,sums,
            profiled:profiled.then(||Mutex::new(vec![0;SPANS.len()])),kernel_us:AtomicU64::new(0),module,context})
    }
    /// A kernel of the module by name (the checks' kernels).
    pub fn function(&self,name:&str)->Result<Function,String>{self.context.make_current()?;self.module.function(name)}
    pub fn stream(&self)->&Stream{&self.stream}
    /// The kernel's registers and spilled bytes per thread, and its blocks per multiprocessor.
    pub fn resources(&self)->Result<(i32,i32,i32),String>{let(r,l)=self.motion.resources()?;Ok((r,l,self.blocks_per_multiprocessor))}
    /// The widest batch of `groups` groups of whole tasks the GPU runs at once: as many tasks per
    /// group as its resident blocks hold (one at least). More wait for a block to leave.
    pub fn width(&self,groups:u64)->u64{groups*TASK*((self.info.multiprocessors*self.blocks_per_multiprocessor) as u64/groups).max(1)}
    /// The motion kernel's time over every launch so far (s).
    pub fn kernel_seconds(&self)->f64{self.kernel_us.load(Ordering::Relaxed) as f64*1e-6}
    /// The profiled kernel's counters summed over every trajectory it ran (`SPANS`); None unprofiled.
    pub fn spans(&self)->Option<Vec<u64>>{self.profiled.as_ref().map(|s|s.lock().unwrap_or_else(|e|e.into_inner()).clone())}
    /// The program the kernels read until the next call, on the calling thread's context (one solve holds the process's GPU at a time).
    fn set(&self,program:&Program)->Result<(),String>{self.context.make_current()?;self.stream.put(self.program,&program.params)}
    /// A model and its step's settings on the device.
    pub fn program(&self,model:&Model,settings:&Settings)->Result<Program,String>{self.context.make_current()?;Program::new(&self.stream,model,settings)}

    /// Trajectories [first, first + count) of the run `seed` (lib.rs `simulate`): `steps` steps of the
    /// settings' dt between `outputs` output times, in groups of `group`, as tasks of at most `TASK`
    /// trajectories inside a group, (r, κ) kept at `listed`, with the control variates of `variates`; with `ends`, the g-th group's
    /// trajectories only through the output `ends[g]` (lib.rs `simulate_to`), half a period past it, its sums counting none beyond.
    pub fn simulate(&self,program:&Program,model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],
        variates:Option<&Variates>,ends:Option<&[usize]>)->Result<Outcome,String>{
        self.set(program)?;let clock=Instant::now();
        let controls=Controls::new(&self.stream,model,settings,variates)?;let mut cycles=[0u64;2];
        // The tasks in order: (their group, their first trajectory, the one after their last), as lib.rs `run`.
        let bounds=|g:u64|(first+g*group,(first+(g+1)*group).min(first+count));
        let tasks:Vec<(u64,u64,u64)>=(0..count.div_ceil(group)).flat_map(|g|{let(start,end)=bounds(g);(start..end).step_by(TASK as usize).map(move|s|(g,s,(s+TASK).min(end)))}).collect();
        let windows=Windows::new(settings,steps,outputs,listed)?;let through=|g:u64|ends.map_or(outputs-1,|e|e[g as usize]);
        let(mut groups,mut samples,mut jumps,mut beats)=(Vec::<Group>::new(),vec![vec![];listed.len()],0u64,0f64);let mut last=None;
        for wave in tasks.chunks(self.wave(program,&windows)?){
            // The initial states, drawn on the CPU; each task's trajectories in its block's slots.
            // The atom from its bare state's slow part (lib.rs `initial_trajectory`), the held twin from
            // the bare state's at the twin's point (lib.rs `hold`, control.rs `Twin::start`).
            let twin=variates.and_then(|v|v.twin);
            let states:Vec<Option<program::State>>=wave.par_iter().flat_map_iter(|&(g,from,to)|{
                let(start,end)=bounds(g);
                (0..WARPS as u64).map(move|w|{let index=from+w;(index<to).then(||{
                    let bare=bare_trajectory(model,settings,initial,seed,index,Stratum{first:start,size:end-start});
                    let held=twin.map(|h|{let mut x=bare.clone();h.start(&mut x);x});let atom=dressed(model,bare);
                    program::State::new(&atom,held.as_ref().unwrap_or(&atom),settings)
                })})
            }).collect();
            // The wave's runs of tasks of one group, merged on the device in order (lib.rs `run`).
            let mut runs:Vec<(u64,usize,usize)>=vec![];
            for(i,&(g,_,_))in wave.iter().enumerate(){match runs.last_mut(){Some(r)if r.0==g=>r.2+=1,_=>runs.push((g,i,1))}}
            let(group_sums,kept,finished,costs)=self.run(program,&windows,&controls,seed,&states,&wave.iter().map(|t|through(t.0) as i32).collect::<Vec<_>>(),&runs.iter().map(|r|[r.1 as i32,r.2 as i32]).collect::<Vec<_>>())?;
            cycles[0]+=costs[0];cycles[1]+=costs[1];
            for(k,&(g,i,count))in runs.iter().enumerate(){
                let trajectories=wave[i..i+count].iter().map(|t|t.2-t.1).sum::<u64>() as f64;
                let sums=Group{sums:(0..outputs).map(|o|sums_of(&group_sums[(k*outputs+o)*RECORD..][..RECORD],if o<=through(g){trajectories}else{0.})).collect()};
                if last==Some(g){groups.last_mut().unwrap().merge(&sums);}else{groups.push(sums);last=Some(g);}
            }
            // The samples in the trajectories' order.
            for(i,&(_,from,to))in wave.iter().enumerate(){
                for w in 0..(to-from) as usize{
                    let slot=WARPS*i+w;let state=&finished[slot];jumps+=state.jumps as u64;beats=beats.max(state.beats as f64);
                    for(k,kept_k)in samples.iter_mut().enumerate(){let x=&kept[(slot*listed.len()+k)*7..][..7];if x[6]>0.{kept_k.push(([x[0],x[1],x[2]],[x[3],x[4],x[5]]));}}
                }
            }
        }
        // The held twins' share of the time: their cycles in the warps' (lib.rs `Outcome::seconds`).
        let(wall,share)=(clock.elapsed().as_secs_f64(),if cycles[1]>0{cycles[0] as f64/cycles[1] as f64}else{0.});
        Ok(Outcome{groups,samples,jumps,beats,seconds:[wall*(1.-share),wall*share]})
    }

    /// The trajectories after `steps` steps of the settings' dt on the GPU (engine.rs `Engine::step`
    /// each): the checks' entry, one launch without the motion windows.
    pub fn advance(&self,program:&Program,settings:&Settings,trajectories:&[Trajectory],steps:usize)->Result<Vec<Trajectory>,String>{
        self.set(program)?;
        let seed=trajectories.first().map_or(0,|t|t.seed);
        if trajectories.iter().any(|t|t.seed!=seed){return Err("the trajectories of a launch share their run's seed".into());}
        let mut plain=settings.clone();plain.average=false;
        let windows=Windows::new(&plain,steps,2,&[])?;
        let states:Vec<Option<program::State>>=trajectories.iter().map(|t|Some(program::State::new(t,t,settings))).chain(std::iter::repeat(None)).take(trajectories.len().div_ceil(WARPS)*WARPS).collect();
        let(_,_,finished,_)=self.run(program,&windows,&Controls::none(),seed,&states,&vec![1;states.len()/WARPS],&[])?;
        Ok(trajectories.iter().zip(&finished).map(|(t,s)|s.trajectory(settings,t.seed)).collect())
    }

    /// A trajectory's cluster's components at its point as the exact step takes them (the kernel's
    /// `exact_components`, a check's entry): per part H_q, then per part and axis G_q, each n × n row-major.
    pub fn exact_probe(&self,program:&Program,model:&Model,settings:&Settings,t:&Trajectory)->Result<Vec<Complex>,String>{
        self.set(program)?;
        let(n,q)=(model.dimension(t.cluster),1+model.beats(t.cluster).len());let size=5*q*n*n;
        let state=Buffer::with(&self.stream,&[program::State::new(t,t,settings)])?;let out=Buffer::new(8*size.max(program.exact))?;
        let f=self.function("exact_probe")?;self.stream.launch(&f,1,32,&[&state.pointer,&out.pointer])?;
        let mut words=vec![[0f32;2];size];self.stream.download(&out,&mut words)?;
        Ok(words[..4*q*n*n].iter().map(|z|Complex::new(z[0] as f64,z[1] as f64)).collect())
    }

    /// A trajectory's cluster's exact responses at its point (the kernel's `exact_responses`, a check's
    /// entry): per near beam (in the beams' order), cluster column, axis a and ground state `to`, (C_a R_l)[to, b];
    /// Err where a resolvent is singular.
    pub fn response_probe(&self,program:&Program,settings:&Settings,t:&Trajectory)->Result<Vec<Complex>,String>{
        self.set(program)?;
        let size=program.params.nnear as usize*program::MAXN*3*crate::model::GROUND;
        let state=Buffer::with(&self.stream,&[program::State::new(t,t,settings)])?;let out=Buffer::new(8*(size+program::NE*32+program.params.ncarriers as usize*program::MAXN*program::MAXN))?;
        let f=self.function("response_probe")?;self.stream.launch(&f,1,32,&[&state.pointer,&out.pointer])?;
        let mut words=vec![[0f32;2];size];self.stream.download(&out,&mut words)?;
        if words[0][0].is_nan(){return Err("The excited states' resolvent is singular".into());}
        Ok(words.iter().map(|z|Complex::new(z[0] as f64,z[1] as f64)).collect())
    }

    /// A trajectory after a jump at its position and time on the GPU (engine.rs `Engine::jump`, the first
    /// order in the Doppler shift or, where `exact`, the exact response; a check's entry): the kernel's `jump_probe`.
    pub fn jump_probe(&self,program:&Program,settings:&Settings,t:&Trajectory,exact:bool)->Result<Trajectory,String>{
        self.set(program)?;
        let state=Buffer::with(&self.stream,&[program::State::new(t,t,settings)])?;let scratch=Buffer::new(8*program.scratch())?;
        let mut run=program::Run::empty();run.seed=t.seed;
        let f=self.function("jump_probe")?;self.stream.launch(&f,1,32,&[&run,&state.pointer,&scratch.pointer,&(exact as i32)])?;
        let mut out=[program::State::default()];self.stream.download(&state,&mut out)?;
        Ok(out[0].trajectory(settings,t.seed))
    }

    /// The tasks a wave holds: as many as a quarter of the free memory takes (at most half a gigabyte).
    fn wave(&self,program:&Program,w:&Windows)->Result<usize,String>{
        let slot=std::mem::size_of::<program::State>()+8*w.ring_stride+56*w.listed.len()+8*program.scratch()+8*MOST_INTERVALS*SLOT_RECORD;
        let(free,_)=self.context.memory()?;
        Ok(((free/4).min(1<<29)/(WARPS*slot+8*w.outputs*RECORD)).max(1))
    }
    /// One wave: blocks of `WARPS` slots (None: no trajectory), per block its last output, its trajectories through that output's interval and its reach in launches
    /// of about LAUNCH_MS: the first a step per warp (its budget zero), each next one's budget the cycles the slowest warp's clock
    /// made in the last per millisecond of it on the device, times LAUNCH_MS. Two launches stand in the stream at a time, each
    /// reporting into its own words of host memory, so that the device does not wait for the host's turn between them. The sums per
    /// output of each run of tasks ([first task, tasks]), the samples kept per slot, the slots' final
    /// states, and the held twins' cycles and all the warps'.
    fn run(&self,program:&Program,w:&Windows,controls:&Controls,seed:u64,slots:&[Option<program::State>],through:&[i32],runs:&[[i32;2]])->Result<(Vec<f64>,Vec<f64>,Vec<program::State>,[u64;2]),String>{
        let n=slots.len()/WARPS;
        let states:Vec<program::State>=slots.iter().map(|s|s.unwrap_or_default()).collect();
        let tasks:Vec<i32>=(0..n).flat_map(|i|{let count=slots[WARPS*i..][..WARPS].iter().filter(|s|s.is_some()).count();[through[i],count as i32,(WARPS*i) as i32]}).collect();
        let state_buffer=Buffer::with(&self.stream,&states)?;let task_buffer=Buffer::with(&self.stream,&tasks)?;
        let rings=Buffer::new(8*w.ring_stride*WARPS*n)?;let samples=Buffer::new(56*w.listed.len().max(1)*WARPS*n)?;
        let sums=Buffer::new(8*w.outputs*RECORD*n)?;let scratch=Buffer::new(8*program.scratch()*WARPS*n)?;let records=Buffer::new(8*MOST_INTERVALS*SLOT_RECORD*WARPS*n)?;
        let profile=Buffer::new(if self.profiled.is_some(){8*SPANS.len()*WARPS*n}else{0})?;
        // A launch's report per slot: the held twin's cycles, all its cycles, and the interval it stands in (a slot without a
        // trajectory, or past its task's last, past them all).
        let mut cycles:Vec<u64>=slots.iter().flat_map(|s|[0,0,if s.is_some(){0}else{u64::MAX}]).collect();
        let reports=[HostWords::new(&cycles)?,HostWords::new(&cycles)?];
        self.stream.zero(&rings)?;self.stream.zero(&samples)?;self.stream.zero(&sums)?;self.stream.zero(&profile)?;
        let mut run=program::Run{states:state_buffer.pointer,rings:rings.pointer,samples:samples.pointer,sums:sums.pointer,scratch:scratch.pointer,tasks:task_buffer.pointer,
            profile:if self.profiled.is_some(){profile.pointer}else{0},records:records.pointer,
            harmonic:controls.harmonic.as_ref().map_or(0,|b|b.pointer),costs:reports[0].address(),held:controls.held,
            seed,budget:0,outputs:w.outputs as i32,steps:w.steps as i32,reach:w.reach as i32,begin:0,end:0,half:w.half.map(|h|h as i32),ring:w.ring.map(|l|l as i32),
            ring_at:w.ring_at.map(|x|x as i32),ring_stride:w.ring_stride as i32,scratch_stride:program.scratch() as i32,nlisted:w.listed.len() as i32,pad:0,listed:[0;MOST_LISTED]};
        for(k,&l)in w.listed.iter().enumerate(){run.listed[k]=l as i32;}
        let mut totals=[0u64;2];let(mut begin,mut launched)=(0,0usize);
        let mut standing:VecDeque<(Event,Event,usize)>=VecDeque::new();let clock=Instant::now();
        while begin<w.intervals||!standing.is_empty(){
            while begin<w.intervals&&standing.len()<reports.len(){
                run.begin=begin as i32;run.end=(begin+MOST_INTERVALS).min(w.intervals) as i32;run.costs=reports[launched%reports.len()].address();
                let(start,end)=(Event::new()?,Event::new()?);
                self.stream.record(&start)?;self.stream.launch(&self.motion,(WARPS*n) as u32,32,&[&run])?;self.stream.record(&end)?;
                standing.push_back((start,end,launched%reports.len()));launched+=1;
            }
            let(start,end,report)=standing.pop_front().unwrap();end.synchronize()?;
            reports[report].read(&mut cycles);
            let most=cycles.chunks_exact(3).fold(0,|m,c|{totals[0]+=c[0];totals[1]+=c[1];m.max(c[1])});
            run.budget=(most as f64*LAUNCH_MS/end.since(&start)?.max(1e-3)) as i64;
            // The intervals every slot has passed.
            let done=cycles.chunks_exact(3).map(|c|c[2]).min().map_or(w.intervals,|d|(d.min(w.intervals as u64)) as usize);
            if done>begin{let mut passed=run;passed.begin=begin as i32;passed.end=done as i32;self.stream.launch(&self.sums,n as u32,(WARPS*32) as u32,&[&passed])?;begin=done;}
        }
        self.stream.synchronize()?;self.kernel_us.fetch_add((clock.elapsed().as_secs_f64()*1e6) as u64,Ordering::Relaxed);
        let mut merged=vec![0f64;w.outputs*RECORD*runs.len()];
        if !runs.is_empty(){
            let(run_buffer,out)=(Buffer::with(&self.stream,runs)?,Buffer::new(8*merged.len())?);let total=merged.len() as u32;
            self.stream.launch(&self.merge,total.div_ceil(256),256,&[&sums.pointer,&run_buffer.pointer,&(runs.len() as i32),&(w.outputs as i32),&out.pointer])?;
            self.stream.download(&out,&mut merged)?;
        }
        let mut kept=vec![0f64;7*w.listed.len()*WARPS*n];self.stream.download(&samples,&mut kept)?;
        let mut finished=vec![program::State::default();WARPS*n];self.stream.download(&state_buffer,&mut finished)?;
        if let Some(spans)=&self.profiled{
            let mut counters=vec![0u64;SPANS.len()*WARPS*n];self.stream.download(&profile,&mut counters)?;
            let mut spans=spans.lock().unwrap_or_else(|e|e.into_inner());for(i,c)in counters.iter().enumerate(){spans[i%SPANS.len()]+=c;}
        }
        Ok((merged,kept,finished,totals))
    }
}

/// A batch's motion windows (lib.rs `Windows`): half a trap period per lab axis, the reach past the
/// last output, the kernel's rings (per axis a power of two of at least 2h + steps + 1 steps of three
/// sums) and its intervals.
struct Windows{outputs:usize,steps:usize,listed:Vec<usize>,half:[usize;3],reach:usize,ring:[usize;3],ring_at:[usize;3],ring_stride:usize,intervals:usize}
impl Windows{
    fn new(settings:&Settings,steps:usize,outputs:usize,listed:&[usize])->Result<Windows,String>{
        if listed.len()>MOST_LISTED{return Err("too many listed outputs".into());}
        let half:[usize;3]=std::array::from_fn(|a|{let w=settings.lab[a];if settings.average&&w>0.{(std::f64::consts::PI/w/settings.dt).round() as usize}else{0}});
        let reach=half.into_iter().max().unwrap();let ring=half.map(|h|(2*h+steps+1).next_power_of_two());
        Ok(Windows{outputs,steps,listed:listed.to_vec(),half,reach,ring,ring_at:[0,3*ring[0],3*(ring[0]+ring[1])],ring_stride:3*(ring[0]+ring[1]+ring[2]),intervals:outputs+reach.div_ceil(steps)})
    }
}

/// An output's sums of a task (the kernel's record: held, profile, doppler, the populations, the held
/// twin's, the motion's means and the harmonic control's per axis) as stats.rs `Sums`.
fn sums_of(x:&[f64],trajectories:f64)->Sums{
    let mut s=Sums::zero();s.trajectories=trajectories;s.held=x[0];s.profile=x[1];s.doppler=x[2];s.folded=x[FOLDED];
    s.populations.copy_from_slice(&x[3..3+STATES]);s.twin.copy_from_slice(&x[TWIN..TWIN+STATES]);s.twin2=[x[TWIN+STATES],x[TWIN+STATES+1]];
    for a in 0..3{
        let m=&x[MOTION+2*a..][..2];s.kappa[a]=m[0];s.kappa2[a]=m[1];
        let h=&x[HARMONIC+3*a..][..3];s.harmonic[a]=[h[0],h[1],h[2]];
    }
    s
}

/// A batch's control variates on the device (lib.rs `Variates`): the harmonic control's A and B per
/// output and lab axis, and the held twin's half step's propagators -- engine.rs `Engine::held`'s,
/// by the same series --, generators, jump operators and populations' readouts.
struct Controls{harmonic:Option<Buffer>,held:program::Held,_buffers:Vec<Buffer>}
impl Controls{
    fn none()->Controls{Controls{harmonic:None,held:program::Held::default(),_buffers:vec![]}}
    fn new(stream:&Stream,model:&Model,settings:&Settings,variates:Option<&Variates>)->Result<Controls,String>{
        let Some(v)=variates else{return Ok(Controls::none())};
        let c32=|z:Complex|[z.re as f32,z.im as f32];
        let tables:Vec<f64>=v.harmonic.a.iter().zip(&v.harmonic.b).flat_map(|(a,b)|(0..3).flat_map(move|x|a[x].iter().chain(&b[x]).flatten().copied())).collect();
        let mut out=Controls{harmonic:Some(Buffer::with(stream,&tables)?),held:program::Held::default(),_buffers:vec![]};
        let Some(twin)=v.twin else{return Ok(out)};
        let(mut u,mut h,mut readout,mut blocks)=(vec![],vec![],vec![],vec![]);
        for(k,local)in twin.held.locals.iter().enumerate(){
            let n=model.dimension(k);blocks.push(u.len() as i32);
            let mut half=vec![[0f32;2];n*n];
            for j in 0..n{let mut e=[Complex::ZERO;MAXN];e[j]=Complex::ONE;propagate(&local.hamiltonian,n,0.5*settings.dt,&mut e,settings.truncation);for i in 0..n{half[i*n+j]=c32(e[i]);}}
            u.extend(half);h.extend(local.hamiltonian[..n*n].iter().map(|&z|c32(z)));
            readout.extend(twin.readout[k].iter().flatten().copied());
        }
        let(mut l,mut jumps)=(vec![],vec![]);
        for cluster in &twin.held.jumps{for class in cluster{jumps.push(l.len() as i32);for x in class{l.extend(x.iter().map(|&z|c32(z)));}}}
        let mut put=|buffer:Buffer|{let p=buffer.pointer;out._buffers.push(buffer);p};
        out.held=program::Held{u:put(Buffer::with(stream,&u)?),h:put(Buffer::with(stream,&h)?),l:put(Buffer::with(stream,&l)?),readout:put(Buffer::with(stream,&readout)?),
            blocks:put(Buffer::with(stream,&blocks)?),jumps:put(Buffer::with(stream,&jumps)?)};
        Ok(out)
    }
}

impl program::State{
    /// A trajectory's initial state for the kernel, with its held twin's internal state and threshold
    /// (`twin`: at fourth order the bare state's slow part at the twin's point, control.rs
    /// `Twin::start`): on the trap's principal axes, ψ normalized.
    fn new(t:&crate::engine::Trajectory,twin:&crate::engine::Trajectory,settings:&Settings)->program::State{
        let along=|v:[f64;3]|->[f64;3]{std::array::from_fn(|i|settings.axes[i][0]*v[0]+settings.axes[i][1]*v[1]+settings.axes[i][2]*v[2])};
        fn unit<const N:usize>(t:&crate::engine::Trajectory)->(f64,[[f32;2];N]){let norm:f64=t.psi.iter().map(|z|z.norm_sqr()).sum();let scale=1./norm.sqrt();
            (norm.ln(),std::array::from_fn(|x|[(t.psi[x].re*scale) as f32,(t.psi[x].im*scale) as f32]))}
        let((logn,psi),(twin_logn,held))=(unit::<MAXM>(t),unit::<MAXN>(twin));
        program::State{q:along(t.r),p:along(t.kappa),time:t.time,logn,threshold:t.threshold,log_threshold:t.threshold.ln(),prefix:[0.;9],
            q0:along(settings.about_site(t.r)),p0:along(t.kappa),twin_logn,twin_threshold:twin.threshold,twin_log_threshold:twin.threshold.ln(),
            psi,twin:held,cluster:t.cluster as i32,jumps:t.jumps as i32,death:if t.alive{-1}else{0},
            index:t.index as u32,twin_cluster:twin.cluster as i32,twin_jumps:twin.jumps as i32,beats:t.beats as f32,interval:0,step:-1}
    }
    /// The trajectory of a state: on the lab axes, ψ with its norm e^{ln‖ψ‖²/2} as the CPU keeps it.
    fn trajectory(&self,settings:&Settings,seed:u64)->Trajectory{
        let lab=|v:[f64;3]|->[f64;3]{std::array::from_fn(|a|(0..3).map(|i|v[i]*settings.axes[i][a]).sum())};
        let scale=(0.5*self.logn).exp();
        let mut psi=[Complex::ZERO;crate::model::MOST_MIXED];for(x,z)in self.psi.iter().enumerate(){psi[x]=Complex::new(z[0] as f64*scale,z[1] as f64*scale);}
        Trajectory{cluster:self.cluster as usize,psi,r:lab(self.q),kappa:lab(self.p),time:self.time,threshold:self.threshold,alive:self.death<0,
            jumps:self.jumps as u64,beats:self.beats as f64,seed,index:self.index as u64}
    }
}
const _:()=assert!(MAXN==crate::model::MOST_STATES);
