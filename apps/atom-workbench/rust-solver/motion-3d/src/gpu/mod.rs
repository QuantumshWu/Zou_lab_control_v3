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
//! The initial states are drawn on the CPU (lib.rs `initial_trajectory` and a held twin's start,
//! exactly); the tasks run as blocks, as many at once as the device memory comfortably holds (a
//! wave), each launch a range of output intervals of about `LAUNCH_MS` (well inside the display
//! watchdog), at most `MOST_INTERVALS`: each trajectory through them on its own, its records of the
//! intervals summed per task at the launch's end. The tasks' sums come back per output and are
//! merged per group in order, as the CPU's are.
pub mod driver;
pub mod program;
pub mod ptx;

use crate::engine::{propagate,Settings,Trajectory};
use crate::model::Model;
use crate::stats::{Group,Sums};
use crate::{bare_trajectory,dressed,Complex,Initial,Outcome,Stratum,Variates,STATES,TASK};
use driver::{Buffer,Context,DeviceInfo,Function,Module,Stream};
use program::{Program,MAXN,MOST_LISTED};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64,Ordering};
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
/// A block's warps (the kernel's WARPS: lib.rs `TASK`), an output's sums (its RECORD), a trajectory's
/// record of an output interval (its SLOT_RECORD: the point record and per lab axis two closing
/// windows' held flag and means) and the most intervals of a launch (its MOST_INTERVALS: the room of
/// each trajectory's records, which the block sums at the launch's end).
pub(crate) const WARPS:usize=TASK as usize;
pub(crate) const RECORD:usize=91;
pub(crate) const SLOT_RECORD:usize=70+3*2*8;
pub(crate) const MOST_INTERVALS:usize=64;
/// Where an output's sums (the kernel's record) keep the held twin's, the folded excited population's,
/// the motion's and the harmonic control's (`sums_of`).
pub(crate) const TWIN:usize=35;
pub(crate) const FOLDED:usize=69;
pub(crate) const MOTION:usize=70;
pub(crate) const HARMONIC:usize=82;
/// The profiled kernel's counters per trajectory (its SPANS): the cycles of the point, the generator,
/// the exponential, the evolution with the kick, the jumps -- of them the exact responses, the
/// photons' matrices with the classes' weights and the absorbed light --, the loss, the steps, the
/// readouts, the windows, the block's sums at a launch's end with the wait for its slowest warp, the
/// held twin's steps and records, and the launches; the advances, the jumps and the jumps with the
/// exact responses.
pub const SPANS:[&str;18]=["point","generator","exponential","evolve","jump","exact","photons","absorption","loss","step","readout","windows","sums","twin","total","advances","jumps","exact jumps"];

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
        let motion=module.function("motion3d")?;motion.prefer_shared()?;let merge=module.function("merge")?;
        let module_ms=clock.elapsed().as_secs_f64()*1e3;
        let blocks_per_multiprocessor=motion.occupancy((WARPS*32) as i32)?.max(1);
        Ok(Gpu{info:device.clone(),startup:Startup{driver_ms,context_ms,module_ms},driver_version:version,motion,stream:Stream::new()?,blocks_per_multiprocessor,merge,
            profiled:profiled.then(||Mutex::new(vec![0;SPANS.len()])),kernel_us:AtomicU64::new(0),module,context})
    }
    /// A kernel of the module by name (the checks' kernels).
    pub fn function(&self,name:&str)->Result<Function,String>{self.context.make_current()?;self.module.function(name)}
    pub fn stream(&self)->&Stream{&self.stream}
    /// The kernel's registers and spilled bytes per thread, and its blocks per multiprocessor.
    pub fn resources(&self)->Result<(i32,i32,i32),String>{let(r,l)=self.motion.resources()?;Ok((r,l,self.blocks_per_multiprocessor))}
    /// The widest batch of `groups` groups of whole tasks the GPU runs at once: as many tasks per
    /// group as its resident blocks hold (one at least).
    pub fn width(&self,groups:u64)->u64{groups*TASK*((self.info.multiprocessors*self.blocks_per_multiprocessor) as u64/groups).max(1)}
    /// The motion kernel's time over every launch so far (s).
    pub fn kernel_seconds(&self)->f64{self.kernel_us.load(Ordering::Relaxed) as f64*1e-6}
    /// The profiled kernel's counters summed over every trajectory it ran (`SPANS`); None unprofiled.
    pub fn spans(&self)->Option<Vec<u64>>{self.profiled.as_ref().map(|s|s.lock().unwrap_or_else(|e|e.into_inner()).clone())}
    /// A model and its step's settings on the device.
    pub fn program(&self,model:&Model,settings:&Settings)->Result<Program,String>{self.context.make_current()?;Program::new(&self.stream,model,settings)}

    /// Trajectories [first, first + count) of the run `seed` (lib.rs `simulate`): `steps` steps of the
    /// settings' dt between `outputs` output times, in groups of `group`, as tasks of at most `TASK`
    /// trajectories inside a group, (r, κ) kept at `listed`, with the control variates of `variates`.
    pub fn simulate(&self,program:&Program,model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],
        variates:Option<&Variates>)->Result<Outcome,String>{
        self.context.make_current()?;let clock=Instant::now();
        let controls=Controls::new(&self.stream,model,settings,variates)?;let mut cycles=[0u64;2];
        // The tasks in order: (their group, their first trajectory, the one after their last), as lib.rs `run`.
        let bounds=|g:u64|(first+g*group,(first+(g+1)*group).min(first+count));
        let tasks:Vec<(u64,u64,u64)>=(0..count.div_ceil(group)).flat_map(|g|{let(start,end)=bounds(g);(start..end).step_by(TASK as usize).map(move|s|(g,s,(s+TASK).min(end)))}).collect();
        let windows=Windows::new(settings,steps,outputs,listed)?;
        let(mut groups,mut samples,mut jumps,mut doppler)=(Vec::<Group>::new(),vec![vec![];listed.len()],0u64,0f64);let mut last=None;
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
            let(group_sums,kept,finished,costs)=self.run(program,&windows,&controls,seed,&states,&runs.iter().map(|r|[r.1 as i32,r.2 as i32]).collect::<Vec<_>>())?;
            cycles[0]+=costs[0];cycles[1]+=costs[1];
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
        // The held twins' share of the time: their cycles in the warps' (lib.rs `Outcome::seconds`).
        let(wall,share)=(clock.elapsed().as_secs_f64(),if cycles[1]>0{cycles[0] as f64/cycles[1] as f64}else{0.});
        Ok(Outcome{groups,samples,jumps,doppler,beats:0.,seconds:[wall*(1.-share),wall*share]})
    }

    /// The trajectories after `steps` steps of the settings' dt on the GPU (engine.rs `Engine::step`
    /// each): the checks' entry, one launch without the motion windows.
    pub fn advance(&self,program:&Program,settings:&Settings,trajectories:&[Trajectory],steps:usize)->Result<Vec<Trajectory>,String>{
        self.context.make_current()?;
        let seed=trajectories.first().map_or(0,|t|t.seed);
        if trajectories.iter().any(|t|t.seed!=seed){return Err("the trajectories of a launch share their run's seed".into());}
        let mut plain=settings.clone();plain.average=false;
        let windows=Windows::new(&plain,steps,2,&[])?;
        let states:Vec<Option<program::State>>=trajectories.iter().map(|t|Some(program::State::new(t,t,settings))).chain(std::iter::repeat(None)).take(trajectories.len().div_ceil(WARPS)*WARPS).collect();
        let(_,_,finished,_)=self.run(program,&windows,&Controls::none(),seed,&states,&[])?;
        Ok(trajectories.iter().zip(&finished).map(|(t,s)|s.trajectory(settings,t.seed)).collect())
    }

    /// The tasks a wave holds: as many as a quarter of the free memory takes (at most half a gigabyte).
    fn wave(&self,program:&Program,w:&Windows)->Result<usize,String>{
        let slot=std::mem::size_of::<program::State>()+8*w.ring_stride+56*w.listed.len()+8*program.scratch()+8*MOST_INTERVALS*SLOT_RECORD;
        let(free,_)=self.context.memory()?;
        Ok(((free/4).min(1<<29)/(WARPS*slot+8*w.outputs*RECORD)).max(1))
    }
    /// One wave: blocks of `WARPS` slots (None: no trajectory), every output interval in launches of
    /// about LAUNCH_MS -- the first a few steps, the next ones sized by the last's pace. The sums per
    /// output of each run of tasks ([first task, tasks]), the samples kept per slot, the slots' final
    /// states, and the held twins' cycles and all the warps'.
    fn run(&self,program:&Program,w:&Windows,controls:&Controls,seed:u64,slots:&[Option<program::State>],runs:&[[i32;2]])->Result<(Vec<f64>,Vec<f64>,Vec<program::State>,[u64;2]),String>{
        let n=slots.len()/WARPS;
        let states:Vec<program::State>=slots.iter().map(|s|s.unwrap_or_default()).collect();
        let tasks:Vec<i32>=(0..n).flat_map(|i|{let count=slots[WARPS*i..][..WARPS].iter().filter(|s|s.is_some()).count();[0,count as i32,(WARPS*i) as i32]}).collect();
        let state_buffer=Buffer::with(&self.stream,&states)?;let task_buffer=Buffer::with(&self.stream,&tasks)?;
        let rings=Buffer::new(8*w.ring_stride*WARPS*n)?;let samples=Buffer::new(56*w.listed.len().max(1)*WARPS*n)?;
        let sums=Buffer::new(8*w.outputs*RECORD*n)?;let scratch=Buffer::new(8*program.scratch()*WARPS*n)?;let records=Buffer::new(8*MOST_INTERVALS*SLOT_RECORD*WARPS*n)?;
        let profile=Buffer::new(if self.profiled.is_some(){8*SPANS.len()*WARPS*n}else{0})?;let costs=Buffer::new(16*WARPS*n)?;
        self.stream.zero(&rings)?;self.stream.zero(&samples)?;self.stream.zero(&sums)?;self.stream.zero(&profile)?;self.stream.zero(&costs)?;
        let mut run=program::Run{states:state_buffer.pointer,rings:rings.pointer,samples:samples.pointer,sums:sums.pointer,scratch:scratch.pointer,tasks:task_buffer.pointer,
            profile:if self.profiled.is_some(){profile.pointer}else{0},records:records.pointer,
            harmonic:controls.harmonic.as_ref().map_or(0,|b|b.pointer),costs:costs.pointer,held:controls.held,
            seed,outputs:w.outputs as i32,steps:w.steps as i32,reach:w.reach as i32,begin:0,end:0,half:w.half.map(|h|h as i32),ring:w.ring.map(|l|l as i32),
            ring_at:w.ring_at.map(|x|x as i32),ring_stride:w.ring_stride as i32,scratch_stride:program.scratch() as i32,nlisted:w.listed.len() as i32,pad:0,listed:[0;MOST_LISTED]};
        for(k,&l)in w.listed.iter().enumerate(){run.listed[k]=l as i32;}
        let mut begin=0;let mut chunk=8usize.div_ceil(w.steps).clamp(1,MOST_INTERVALS);
        while begin<w.intervals{
            run.begin=begin as i32;run.end=(begin+chunk).min(w.intervals) as i32;
            let clock=Instant::now();
            self.stream.launch(&self.motion,n as u32,(WARPS*32) as u32,&[&program.params,&run])?;self.stream.synchronize()?;
            let ms=clock.elapsed().as_secs_f64()*1e3;self.kernel_us.fetch_add((ms*1e3) as u64,Ordering::Relaxed);
            let done=run.end as usize-begin;begin=run.end as usize;
            chunk=((done as f64*LAUNCH_MS/ms.max(1e-3)).floor() as usize).clamp(1,(4*done).max(1)).min(MOST_INTERVALS);
        }
        let mut merged=vec![0f64;w.outputs*RECORD*runs.len()];
        if !runs.is_empty(){
            let(run_buffer,out)=(Buffer::with(&self.stream,runs)?,Buffer::new(8*merged.len())?);let total=merged.len() as u32;
            self.stream.launch(&self.merge,total.div_ceil(256),256,&[&sums.pointer,&run_buffer.pointer,&(runs.len() as i32),&(w.outputs as i32),&out.pointer])?;
            self.stream.download(&out,&mut merged)?;
        }
        let mut kept=vec![0f64;7*w.listed.len()*WARPS*n];self.stream.download(&samples,&mut kept)?;
        let mut finished=vec![program::State::default();WARPS*n];self.stream.download(&state_buffer,&mut finished)?;
        let mut cycles=vec![0u64;2*WARPS*n];self.stream.download(&costs,&mut cycles)?;
        let costs=cycles.chunks_exact(2).fold([0u64;2],|s,c|[s[0]+c[0],s[1]+c[1]]);
        if let Some(spans)=&self.profiled{
            let mut counters=vec![0u64;SPANS.len()*WARPS*n];self.stream.download(&profile,&mut counters)?;
            let mut spans=spans.lock().unwrap_or_else(|e|e.into_inner());for(i,c)in counters.iter().enumerate(){spans[i%SPANS.len()]+=c;}
        }
        Ok((merged,kept,finished,costs))
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

/// An output's sums of a task (the kernel's record: held, profile, doppler, the populations, the held
/// twin's, the motion's means and the harmonic control's per axis) as stats.rs `Sums`.
fn sums_of(x:&[f64],trajectories:f64)->Sums{
    let mut s=Sums::zero();s.trajectories=trajectories;s.held=x[0];s.profile=x[1];s.doppler=x[2];s.folded=x[FOLDED];
    s.populations.copy_from_slice(&x[3..3+STATES]);s.twin.copy_from_slice(&x[TWIN..TWIN+STATES]);s.twin2=[x[TWIN+STATES],x[TWIN+STATES+1]];
    for a in 0..3{
        let m=&x[MOTION+4*a..][..4];s.kappa[a]=m[0];s.kappa2[a]=m[1];s.position[a]=m[2];s.position2[a]=m[3];
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
        let unit=|t:&crate::engine::Trajectory|->(f64,[[f32;2];MAXN]){let norm:f64=t.psi.iter().map(|z|z.norm_sqr()).sum();let scale=1./norm.sqrt();
            (norm.ln(),std::array::from_fn(|x|[(t.psi[x].re*scale) as f32,(t.psi[x].im*scale) as f32]))};
        let((logn,psi),(twin_logn,held))=(unit(t),unit(twin));
        program::State{q:along(t.r),p:along(t.kappa),time:t.time,logn,threshold:t.threshold,log_threshold:t.threshold.ln(),prefix:[0.;15],
            q0:along(t.r),p0:along(t.kappa),twin_logn,twin_threshold:twin.threshold,twin_log_threshold:twin.threshold.ln(),
            psi,twin:held,doppler:t.doppler as f32,cluster:t.cluster as i32,jumps:t.jumps as i32,death:if t.alive{-1}else{0},
            index:t.index as u32,twin_cluster:twin.cluster as i32,twin_jumps:twin.jumps as i32,pad:0}
    }
    /// The trajectory of a state: on the lab axes, ψ with its norm e^{ln‖ψ‖²/2} as the CPU keeps it.
    fn trajectory(&self,settings:&Settings,seed:u64)->Trajectory{
        let lab=|v:[f64;3]|->[f64;3]{std::array::from_fn(|a|(0..3).map(|i|v[i]*settings.axes[i][a]).sum())};
        let scale=(0.5*self.logn).exp();
        let mut psi=[Complex::ZERO;crate::model::MOST_MIXED];for(x,z)in self.psi.iter().enumerate(){psi[x]=Complex::new(z[0] as f64*scale,z[1] as f64*scale);}
        Trajectory{cluster:self.cluster as usize,psi,r:lab(self.q),kappa:lab(self.p),time:self.time,threshold:self.threshold,alive:self.death<0,
            jumps:self.jumps as u64,doppler:self.doppler as f64,beats:0.,seed,index:self.index as u64}
    }
}
const _:()=assert!(MAXN==crate::model::MOST_STATES);
