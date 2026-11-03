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
//! trajectory lane's worker at its start, a solve at its own), and keeps them for every later solve.
//! The driver compiles the PTX for the GPU at its first load on a machine, a few seconds once per
//! version of the kernels, and caches the result (`Startup` records the times). A solve waits for
//! the GPU at most `START_WAIT` (`gpu_within`): one still starting runs on the CPU, with the reason.
//!
//! # Batches
//! The initial states are drawn on the CPU (lib.rs `initial_trajectory`, exactly); the tasks run as
//! blocks, as many at once as the device memory comfortably holds (a wave), each launch a range of
//! output intervals of about `LAUNCH_MS` (well inside the display watchdog). The tasks' sums come
//! back per output and are merged per group in order, as the CPU's are.
pub mod driver;
pub mod program;
pub mod ptx;

use crate::engine::{Settings,Trajectory};
use crate::model::Model;
use crate::stats::{Group,Sums};
use crate::{initial_trajectory,Complex,Initial,Outcome,Stratum,STATES,TASK};
use driver::{Buffer,Context,DeviceInfo,Function,Module,Stream};
use program::{Program,MAXN,MOST_LISTED};
use rayon::prelude::*;
use std::sync::{Condvar,Mutex};
use std::time::{Duration,Instant};

/// The oldest GPUs the PTX targets: compute capability 7.5 (Turing).
pub const MINIMUM_CAPABILITY:(i32,i32)=(7,5);
/// A launch's target wall time (ms).
const LAUNCH_MS:f64=60.;
/// The longest a solve waits for the GPU to start (the context's few hundred milliseconds, not the
/// driver's first compile of the kernels).
pub const START_WAIT:Duration=Duration::from_secs(2);
/// Why a solve did not wait for a GPU still starting.
pub const STARTING:&str="GPU 正在首次准备（驱动编译内核，新版本首次使用时一次）";
/// A block's warps (the kernel's WARPS: lib.rs `TASK`) and an output's sums (its RECORD).
const WARPS:usize=TASK as usize;
const RECORD:usize=47;

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
    stream:Stream,
    blocks_per_multiprocessor:i32,
    merge:Function,
    _module:Module,
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
/// The process's GPU, started if it was not, waiting for it at most `wait`; Err: why there is none
/// (Chinese), `STARTING` when it is not ready by then.
pub fn gpu_within(wait:Duration)->Result<&'static Mutex<Gpu>,String>{
    prepare();
    let start=START.lock().unwrap_or_else(|e|e.into_inner());
    let(start,_)=STARTED.wait_timeout_while(start,wait,|s|matches!(s,Start::Starting)).unwrap_or_else(|e|e.into_inner());
    match &*start{Start::Done(made)=>made.clone(),_=>Err(STARTING.into())}
}
/// The process's GPU, waiting for it to start.
pub fn gpu()->Result<&'static Mutex<Gpu>,String>{gpu_within(Duration::MAX)}

impl Gpu{
    fn new()->Result<Gpu,String>{
        let clock=Instant::now();
        let version=driver::driver_version()?;
        if version<ptx::DRIVER_VERSION{return Err(format!("NVIDIA 驱动太旧（CUDA {}.{}；需要 CUDA 12.0，即 R525 / 527.41 或更新）",version/1000,version%1000/10));}
        let devices=driver::devices()?;
        let device=devices.iter().filter(|d|d.compute_capability>=MINIMUM_CAPABILITY).max_by_key(|d|d.multiprocessors).ok_or_else(||if devices.is_empty(){"没有 NVIDIA 显卡".to_string()}else{
            format!("{} 太旧（需要计算能力 7.5 或更新）",devices.iter().map(|d|format!("{}（计算能力 {}.{}）",d.name,d.compute_capability.0,d.compute_capability.1)).collect::<Vec<_>>().join("、"))})?;
        let driver_ms=clock.elapsed().as_secs_f64()*1e3;
        let clock=Instant::now();let context=Context::new(device)?;let context_ms=clock.elapsed().as_secs_f64()*1e3;
        let clock=Instant::now();
        let module=context.load(&ptx::ptx()).map_err(|e|format!("驱动无法编译 GPU 内核（{e}）"))?;
        let motion=module.function("motion3d")?;motion.prefer_shared()?;let merge=module.function("merge")?;
        let module_ms=clock.elapsed().as_secs_f64()*1e3;
        let blocks_per_multiprocessor=motion.occupancy((WARPS*32) as i32)?.max(1);
        Ok(Gpu{info:device.clone(),startup:Startup{driver_ms,context_ms,module_ms},driver_version:version,motion,stream:Stream::new()?,blocks_per_multiprocessor,merge,_module:module,context})
    }
    /// A kernel of the module by name (the checks' kernels).
    pub fn function(&self,name:&str)->Result<Function,String>{self.context.make_current()?;self._module.function(name)}
    pub fn stream(&self)->&Stream{&self.stream}
    /// The kernel's registers and spilled bytes per thread, and its blocks per multiprocessor.
    pub fn resources(&self)->Result<(i32,i32,i32),String>{let(r,l)=self.motion.resources()?;Ok((r,l,self.blocks_per_multiprocessor))}
    /// The widest batch of `groups` groups of whole tasks the GPU runs at once: as many tasks per
    /// group as its resident blocks hold (one at least).
    pub fn width(&self,groups:u64)->u64{groups*TASK*((self.info.multiprocessors*self.blocks_per_multiprocessor) as u64/groups).max(1)}
    /// A model and its step's settings on the device.
    pub fn program(&self,model:&Model,settings:&Settings)->Result<Program,String>{self.context.make_current()?;Program::new(&self.stream,model,settings)}

    /// Trajectories [first, first + count) of the run `seed` (lib.rs `simulate`): `steps` steps of the
    /// settings' dt between `outputs` output times, in groups of `group`, as tasks of at most `TASK`
    /// trajectories inside a group, (r, κ) kept at `listed`.
    pub fn simulate(&self,program:&Program,model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize])->Result<Outcome,String>{
        self.context.make_current()?;
        // The tasks in order: (their group, their first trajectory, the one after their last), as lib.rs `run`.
        let bounds=|g:u64|(first+g*group,(first+(g+1)*group).min(first+count));
        let tasks:Vec<(u64,u64,u64)>=(0..count.div_ceil(group)).flat_map(|g|{let(start,end)=bounds(g);(start..end).step_by(TASK as usize).map(move|s|(g,s,(s+TASK).min(end)))}).collect();
        let windows=Windows::new(settings,steps,outputs,listed)?;
        let(mut groups,mut samples,mut jumps,mut doppler)=(Vec::<Group>::new(),vec![vec![];listed.len()],0u64,0f64);let mut last=None;
        for wave in tasks.chunks(self.wave(program,&windows)?){
            // The initial states, drawn on the CPU; each task's trajectories in its block's slots.
            let states:Vec<Option<program::State>>=wave.par_iter().flat_map_iter(|&(g,from,to)|{
                let(start,end)=bounds(g);
                (0..WARPS as u64).map(move|w|{let index=from+w;(index<to).then(||program::State::new(&initial_trajectory(model,settings,initial,seed,index,Stratum{first:start,size:end-start}),settings))})
            }).collect();
            // The wave's runs of tasks of one group, merged on the device in order (lib.rs `run`).
            let mut runs:Vec<(u64,usize,usize)>=vec![];
            for(i,&(g,_,_))in wave.iter().enumerate(){match runs.last_mut(){Some(r)if r.0==g=>r.2+=1,_=>runs.push((g,i,1))}}
            let(group_sums,kept,finished)=self.run(program,&windows,seed,&states,&runs.iter().map(|r|[r.1 as i32,r.2 as i32]).collect::<Vec<_>>())?;
            for(k,&(g,i,count))in runs.iter().enumerate(){
                let trajectories=wave[i..i+count].iter().map(|t|t.2-t.1).sum::<u64>() as f64;
                let sums=Group{sums:(0..outputs).map(|o|sums_of(&group_sums[(k*outputs+o)*RECORD..][..RECORD],trajectories)).collect()};
                if last==Some(g){groups.last_mut().unwrap().merge(&sums);}else{groups.push(sums);last=Some(g);}
            }
            // The samples in the trajectories' order.
            for(i,&(_,from,to))in wave.iter().enumerate(){
                for w in 0..(to-from) as usize{
                    let slot=WARPS*i+w;let state=&finished[slot];jumps+=state.jumps as u64;doppler=doppler.max(state.doppler as f64);
                    for(k,kept_k)in samples.iter_mut().enumerate(){let x=&kept[(slot*listed.len()+k)*7..][..7];if x[6]>0.{kept_k.push(([x[0],x[1],x[2]],[x[3],x[4],x[5]]));}}
                }
            }
        }
        Ok(Outcome{groups,samples,jumps,doppler})
    }

    /// The trajectories after `steps` steps of the settings' dt on the GPU (engine.rs `Engine::step`
    /// each): the checks' entry, one launch without the motion windows.
    pub fn advance(&self,program:&Program,settings:&Settings,trajectories:&[Trajectory],steps:usize)->Result<Vec<Trajectory>,String>{
        self.context.make_current()?;
        let seed=trajectories.first().map_or(0,|t|t.seed);
        if trajectories.iter().any(|t|t.seed!=seed){return Err("the trajectories of a launch share their run's seed".into());}
        let mut plain=settings.clone();plain.average=false;
        let windows=Windows::new(&plain,steps,2,&[])?;
        let states:Vec<Option<program::State>>=trajectories.iter().map(|t|Some(program::State::new(t,settings))).chain(std::iter::repeat(None)).take(trajectories.len().div_ceil(WARPS)*WARPS).collect();
        let(_,_,finished)=self.run(program,&windows,seed,&states,&[])?;
        Ok(trajectories.iter().zip(&finished).map(|(t,s)|s.trajectory(settings,t.seed)).collect())
    }

    /// The tasks a wave holds: as many as a quarter of the free memory takes (at most half a gigabyte).
    fn wave(&self,program:&Program,w:&Windows)->Result<usize,String>{
        let slot=std::mem::size_of::<program::State>()+8*w.ring_stride+56*w.listed.len()+8*program.scratch();
        let(free,_)=self.context.memory()?;
        Ok(((free/4).min(1<<29)/(WARPS*slot+8*w.outputs*RECORD)).max(1))
    }
    /// One wave: blocks of `WARPS` slots (None: no trajectory), every output interval in launches of
    /// about LAUNCH_MS -- the first a few steps, the next ones sized by the last's pace. The sums per
    /// output of each run of tasks ([first task, tasks]), the samples kept per slot and the slots'
    /// final states.
    fn run(&self,program:&Program,w:&Windows,seed:u64,slots:&[Option<program::State>],runs:&[[i32;2]])->Result<(Vec<f64>,Vec<f64>,Vec<program::State>),String>{
        let n=slots.len()/WARPS;
        let states:Vec<program::State>=slots.iter().map(|s|s.unwrap_or_default()).collect();
        let tasks:Vec<i32>=(0..n).flat_map(|i|{let count=slots[WARPS*i..][..WARPS].iter().filter(|s|s.is_some()).count();[0,count as i32,(WARPS*i) as i32]}).collect();
        let state_buffer=Buffer::with(&self.stream,&states)?;let task_buffer=Buffer::with(&self.stream,&tasks)?;
        let rings=Buffer::new(8*w.ring_stride*WARPS*n)?;let samples=Buffer::new(56*w.listed.len().max(1)*WARPS*n)?;
        let sums=Buffer::new(8*w.outputs*RECORD*n)?;let scratch=Buffer::new(8*program.scratch()*WARPS*n)?;
        self.stream.zero(&rings)?;self.stream.zero(&samples)?;self.stream.zero(&sums)?;
        let mut run=program::Run{states:state_buffer.pointer,rings:rings.pointer,samples:samples.pointer,sums:sums.pointer,scratch:scratch.pointer,tasks:task_buffer.pointer,
            seed,outputs:w.outputs as i32,steps:w.steps as i32,reach:w.reach as i32,begin:0,end:0,half:w.half.map(|h|h as i32),ring:w.ring.map(|l|l as i32),
            ring_at:w.ring_at.map(|x|x as i32),ring_stride:w.ring_stride as i32,scratch_stride:program.scratch() as i32,nlisted:w.listed.len() as i32,pad:0,listed:[0;MOST_LISTED]};
        for(k,&l)in w.listed.iter().enumerate(){run.listed[k]=l as i32;}
        let mut begin=0;let mut chunk=8usize.div_ceil(w.steps).max(1);
        while begin<w.intervals{
            run.begin=begin as i32;run.end=(begin+chunk).min(w.intervals) as i32;
            let clock=Instant::now();
            self.stream.launch(&self.motion,n as u32,(WARPS*32) as u32,&[&program.params,&run])?;self.stream.synchronize()?;
            let ms=clock.elapsed().as_secs_f64()*1e3;let done=run.end as usize-begin;begin=run.end as usize;
            chunk=((done as f64*LAUNCH_MS/ms.max(1e-3)).floor() as usize).clamp(1,(4*done).max(1));
        }
        let mut merged=vec![0f64;w.outputs*RECORD*runs.len()];
        if !runs.is_empty(){
            let(run_buffer,out)=(Buffer::with(&self.stream,runs)?,Buffer::new(8*merged.len())?);let total=merged.len() as u32;
            self.stream.launch(&self.merge,total.div_ceil(256),256,&[&sums.pointer,&run_buffer.pointer,&(runs.len() as i32),&(w.outputs as i32),&out.pointer])?;
            self.stream.download(&out,&mut merged)?;
        }
        let mut kept=vec![0f64;7*w.listed.len()*WARPS*n];self.stream.download(&samples,&mut kept)?;
        let mut finished=vec![program::State::default();WARPS*n];self.stream.download(&state_buffer,&mut finished)?;
        Ok((merged,kept,finished))
    }
}

/// A batch's motion windows (lib.rs `Windows`): half a trap period per lab axis, the reach past the
/// last output, the kernel's rings (per axis a power of two of at least 2h + steps + 1 steps of five
/// sums) and its intervals.
struct Windows{outputs:usize,steps:usize,listed:Vec<usize>,half:[usize;3],reach:usize,ring:[usize;3],ring_at:[usize;3],ring_stride:usize,intervals:usize}
impl Windows{
    fn new(settings:&Settings,steps:usize,outputs:usize,listed:&[usize])->Result<Windows,String>{
        if listed.len()>MOST_LISTED{return Err("too many listed outputs".into());}
        let half:[usize;3]=std::array::from_fn(|a|{let w=settings.lab[a];if settings.average&&w>0.{(std::f64::consts::PI/w/settings.dt).round() as usize}else{0}});
        let reach=half.into_iter().max().unwrap();let ring=half.map(|h|(2*h+steps+1).next_power_of_two());
        Ok(Windows{outputs,steps,listed:listed.to_vec(),half,reach,ring,ring_at:[0,5*ring[0],5*(ring[0]+ring[1])],ring_stride:5*(ring[0]+ring[1]+ring[2]),intervals:outputs+reach.div_ceil(steps)})
    }
}

/// An output's sums of a task (the kernel's record: held, profile, doppler, the populations and the
/// motion's means per axis) as stats.rs `Sums`.
fn sums_of(x:&[f64],trajectories:f64)->Sums{
    let mut s=Sums::zero();s.trajectories=trajectories;s.held=x[0];s.profile=x[1];s.doppler=x[2];
    s.populations.copy_from_slice(&x[3..3+STATES]);
    for a in 0..3{let m=&x[3+STATES+4*a..][..4];s.kappa[a]=m[0];s.kappa2[a]=m[1];s.position[a]=m[2];s.position2[a]=m[3];}
    s
}

impl program::State{
    /// A trajectory's initial state for the kernel: on the trap's principal axes, ψ normalized.
    fn new(t:&crate::engine::Trajectory,settings:&Settings)->program::State{
        let along=|v:[f64;3]|->[f64;3]{std::array::from_fn(|i|settings.axes[i][0]*v[0]+settings.axes[i][1]*v[1]+settings.axes[i][2]*v[2])};
        let norm:f64=t.psi.iter().map(|z|z.norm_sqr()).sum();let scale=1./norm.sqrt();
        program::State{q:along(t.r),p:along(t.kappa),time:t.time,logn:norm.ln(),threshold:t.threshold,log_threshold:t.threshold.ln(),prefix:[0.;15],
            psi:std::array::from_fn(|x|[(t.psi[x].re*scale) as f32,(t.psi[x].im*scale) as f32]),doppler:t.doppler as f32,cluster:t.cluster as i32,jumps:t.jumps as i32,
            death:if t.alive{-1}else{0},index:t.index as u32,pad:0}
    }
    /// The trajectory of a state: on the lab axes, ψ with its norm e^{ln‖ψ‖²/2} as the CPU keeps it.
    fn trajectory(&self,settings:&Settings,seed:u64)->Trajectory{
        let lab=|v:[f64;3]|->[f64;3]{std::array::from_fn(|a|(0..3).map(|i|v[i]*settings.axes[i][a]).sum())};
        let scale=(0.5*self.logn).exp();
        let psi=std::array::from_fn(|x|Complex::new(self.psi[x][0] as f64*scale,self.psi[x][1] as f64*scale));
        Trajectory{cluster:self.cluster as usize,psi,r:lab(self.q),kappa:lab(self.p),time:self.time,threshold:self.threshold,alive:self.death<0,
            jumps:self.jumps as u64,doppler:self.doppler as f64,seed,index:self.index as u64}
    }
}
const _:()=assert!(MAXN==crate::model::MOST_STATES);
