//! The trajectories on the GPU: one batch of B wavefunctions on one grid, stepped together.
//!
//! # Step
//! Strang splitting with the kinetic halves merged: the stored state is φ = K(dt/2) ψ, and a step
//! is V(dt), the jumps, then K(dt). V is the exact local exponential of the block's generator at
//! each point (model.rs), K the exact kinetic factor per momentum. Every `doppler_every` steps the
//! Doppler term W·κ̂ is applied over that span, explicitly, after V. An output splits the step's K
//! into halves and observes ψ between them.
//!
//! # Jumps
//! Waiting times (Dalibard, Castin, Mølmer 1992; QuTiP's mcsolve): each trajectory draws ln r
//! and accumulates the log of its norm's decay, renormalizing every step; when the sum reaches
//! ln r within a step, the jump is made at the step's midpoint -- back by V(−dt/2), the jump,
//! forward by V(dt/2) -- so the timing error averages out to second order in dt. The channel
//! (carrier and target block, or the absorbing boundary) is chosen by its rate, the photon's
//! direction and polarization exactly from the dipole patterns (kernels `jump_*`).
use crate::driver::{Buffer,Event,Function,Stream};
use crate::gpu::Gpu;
use std::rc::Rc;
use crate::model::{propagator,Grid,Model,Point};
use atom_motion_core::linear::Dense;
use atom_motion_core::Complex;
use rayon::prelude::*;

pub const COMPONENTS:usize=5;
pub const RECORD:usize=52;
pub const REC_GROUND:usize=0;
pub const REC_EXCITED:usize=8;
pub const REC_NORM:usize=32;
pub const REC_X:usize=33;
pub const REC_KX:usize=37;
pub const REC_TAIL:usize=41;
pub const REC_TAIL2:usize=45;
const THREADS:usize=256;
const SUB:usize=8;
const CHOICE:usize=8;
const G_SLOTS:usize=25;
/// Grids the kernels are compiled for (nx, ny): planes in shared memory, and the large ones
/// (N x N, N >= 128) in two passes.
pub const GRIDS:&[(usize,usize)]=&[(16,16),(32,32),(64,64),(64,1),(128,1),(256,1),(512,1),(1024,1),(128,128),(256,256)];
const TILE:usize=8;

/// The passes over the planes of one grid: in shared memory, one block per plane, or for a large
/// grid as a rows pass and a columns pass.
enum Planes{
    Small{kinetic:Function,kinetic_derive:Function,observe:Function,derive:Function,shared:usize,observe_shared:usize,threads:u32},
    Big{n:usize,kinetic:[Function;2],observe:[Function;2],derive:[Function;2]},
}
impl Planes{
    fn new(gpu:&Gpu,nx:usize,ny:usize)->Result<Planes,String>{
        if nx==ny&&nx>=128{
            let f=|gpu:&Gpu,s:&str|->Result<Function,String>{gpu.function(&format!("{s}_{nx}"))};
            let shared=nx*(TILE+1)*8;
            let out=Planes::Big{n:nx,kinetic:[f(gpu,"kinetic_rows")?,f(gpu,"kinetic_cols")?],observe:[f(gpu,"observe_rows")?,f(gpu,"observe_cols")?],
                derive:[f(gpu,"derive_rows")?,f(gpu,"derive_cols")?]};
            if let Planes::Big{kinetic,observe,derive,..}=&out{kinetic[1].allow_shared(shared)?;observe[1].allow_shared(shared)?;derive[1].allow_shared(shared)?;}
            return Ok(out);
        }
        let name=|s:&str|format!("{s}_{nx}x{ny}");
        let shared=ny*(nx+1)*8;let observe_shared=shared+(nx+ny+32)*4;
        let kinetic=gpu.function(&name("kinetic"))?;kinetic.allow_shared(shared)?;
        let kinetic_derive=gpu.function(&name("kinetic_derive"))?;kinetic_derive.allow_shared(shared)?;
        let observe=gpu.function(&name("observe_kinetic"))?;observe.allow_shared(observe_shared)?;
        let derive=gpu.function(&name("derive"))?;derive.allow_shared(shared)?;
        Ok(Planes::Small{kinetic,kinetic_derive,observe,derive,shared,observe_shared,threads:if ny==1{32}else{THREADS as u32}})
    }
    /// K on every plane of `b` trajectory slots, renormalizing by `scale`.
    fn kinetic(&self,s:&Stream,state:&Buffer,components:&Buffer,scale:&Buffer,tables:&[Buffer;2],b:usize)->Result<(),String>{
        match self{
            Planes::Small{kinetic,shared,threads,..}=>s.launch(kinetic,((b*COMPONENTS) as u32,1,1),*threads,*shared,&[&state.pointer,&components.pointer,&scale.pointer,&tables[0].pointer,&tables[1].pointer]),
            Planes::Big{n,kinetic,..}=>{
                s.launch(&kinetic[0],((b*COMPONENTS*n/8) as u32,1,1),THREADS as u32,0,&[&state.pointer,&components.pointer,&scale.pointer,&tables[0].pointer])?;
                s.launch(&kinetic[1],((b*COMPONENTS*n/TILE) as u32,1,1),THREADS as u32,n*(TILE+1)*8,&[&state.pointer,&components.pointer,&tables[1].pointer])
            }
        }
    }
    /// K on every plane, and κ_x, κ_y of the result into dx, dy (`derivative`: κ/N): planes in
    /// shared memory only (false: not done).
    #[allow(clippy::too_many_arguments)]
    fn kinetic_derive(&self,s:&Stream,state:&Buffer,components:&Buffer,scale:&Buffer,tables:&[Buffer;2],derivative:&[Buffer;2],dx:&Buffer,dy:&Buffer,b:usize)->Result<bool,String>{
        let Planes::Small{kinetic_derive,shared,threads,..}=self else{return Ok(false)};
        s.launch(kinetic_derive,((b*COMPONENTS) as u32,1,1),*threads,*shared,&[&state.pointer,&components.pointer,&scale.pointer,&tables[0].pointer,&tables[1].pointer,
            &derivative[0].pointer,&derivative[1].pointer,&dx.pointer,&dy.pointer])?;
        Ok(true)
    }
    /// K with the momentum observables added to the records (and, non-null, the marginals).
    #[allow(clippy::too_many_arguments)]
    fn observe(&self,s:&Stream,state:&Buffer,components:&Buffer,scale:&Buffer,tables:&[Buffer;2],kappa:&[Buffer;2],tails:[f32;2],records:&Buffer,weights:&Buffer,marginals:[u64;2],b:usize)->Result<(),String>{
        match self{
            Planes::Small{observe,observe_shared,threads,..}=>s.launch(observe,((b*COMPONENTS) as u32,1,1),*threads,*observe_shared,&[&state.pointer,&components.pointer,&scale.pointer,
                &tables[0].pointer,&tables[1].pointer,&kappa[0].pointer,&kappa[1].pointer,&tails[0],&tails[1],&records.pointer,&weights.pointer,&marginals[0],&marginals[1]]),
            Planes::Big{n,observe,..}=>{
                s.launch(&observe[0],((b*COMPONENTS*n/8) as u32,1,1),THREADS as u32,0,&[&state.pointer,&components.pointer,&scale.pointer,&tables[0].pointer,&kappa[0].pointer,&tails[0],&records.pointer,&weights.pointer,&marginals[0]])?;
                s.launch(&observe[1],((b*COMPONENTS*n/TILE) as u32,1,1),THREADS as u32,n*(TILE+1)*8,&[&state.pointer,&components.pointer,&tables[1].pointer,&kappa[1].pointer,&tails[1],&records.pointer,&weights.pointer,&marginals[1]])
            }
        }
    }
    /// κ_x ψ into dx and κ_y ψ into dy for every trajectory (list None), or for the `count` listed
    /// ones, at most `b`.
    #[allow(clippy::too_many_arguments)]
    fn derive(&self,s:&Stream,state:&Buffer,components:&Buffer,list:Option<(&Buffer,&Buffer)>,b:usize,dx:&Buffer,dy:&Buffer,tables:&[Buffer;2])->Result<(),String>{
        let(l,c)=list.map_or((0u64,0u64),|(l,c)|(l.pointer,c.pointer));
        match self{
            Planes::Small{derive,shared,threads,..}=>{
                let blocks=if list.is_some(){b.min(64)*COMPONENTS}else{b*COMPONENTS};
                s.launch(derive,(blocks as u32,1,1),*threads,*shared,&[&state.pointer,&components.pointer,&l,&c,&(b as i32),&dx.pointer,&dy.pointer,&tables[0].pointer,&tables[1].pointer])
            }
            Planes::Big{n,derive,..}=>{
                s.launch(&derive[0],((b*COMPONENTS*n/8) as u32,1,1),THREADS as u32,0,&[&state.pointer,&components.pointer,&l,&c,&dx.pointer,&tables[0].pointer])?;
                s.launch(&derive[1],((b*COMPONENTS*n/TILE) as u32,1,1),THREADS as u32,n*(TILE+1)*8,&[&state.pointer,&components.pointer,&l,&c,&dy.pointer,&tables[1].pointer])
            }
        }
    }
}

fn bitrev(j:usize,n:usize)->usize{let bits=n.trailing_zeros();if bits==0{0}else{j.reverse_bits()>>(usize::BITS-bits)}}

/// A complex table in single precision.
fn complex32(z:Complex)->[f32;2]{[z.re as f32,z.im as f32]}
/// Matrices per point as the kernels read them: entry (i, j) of point p at [(i n + j) points + p].
fn planar(matrices:&[&Dense])->Vec<[f32;2]>{
    let(n,points)=(matrices[0].rows,matrices.len());let mut out=vec![[0f32;2];n*n*points];
    for(p,m)in matrices.iter().enumerate(){for i in 0..n{for j in 0..n{out[(i*n+j)*points+p]=complex32(m[(i,j)]);}}}
    out
}

/// A pair of device tables for the two blocks (F = 2: 5 × 5, F = 1: 3 × 3).
struct Local{m5:Buffer,m3:Buffer}
impl Local{
    fn new(gpu:&Gpu,points:&[Point],matrix:impl Fn(&Point,usize)->Dense+Sync)->Result<Local,String>{
        let m5:Vec<Dense>=points.par_iter().map(|p|matrix(p,1)).collect();let m3:Vec<Dense>=points.par_iter().map(|p|matrix(p,0)).collect();
        let upload=|ms:&Vec<Dense>|->Result<Buffer,String>{let data=planar(&ms.iter().collect::<Vec<_>>());let b=Buffer::new(data.len()*8)?;gpu.stream.upload(&b,&data)?;Ok(b)};
        Ok(Local{m5:upload(&m5)?,m3:upload(&m3)?})
    }
}

/// The kernels' `Trajectories`, `Local` and `Jumps` structs as launch arguments.
#[repr(C)]#[derive(Clone,Copy)]
pub(crate) struct TrajectoriesArg{ids:u64,components:u64,scale:u64,log_norm:u64,threshold:u64,jumps:u64}
#[repr(C)]#[derive(Clone,Copy)]
struct LocalArg{m5:u64,m3:u64}
#[repr(C)]#[derive(Clone,Copy)]
struct JumpsArg{jump:u64,response:u64,step:f32,radii:i32,beams:u64,beam_count:i32,cap:u64,wave_number:u64,carriers:i32,nx:i32,dx:f32}
#[repr(C)]#[derive(Clone,Copy)]
struct Seed{lo:u32,hi:u32}
macro_rules! arg_struct{($($t:ty),*)=>{$(impl crate::driver::Arg for $t{fn pointer(&self)->*mut std::ffi::c_void{self as *const $t as *mut std::ffi::c_void}})*};}
impl crate::driver::Arg for [i32;2]{fn pointer(&self)->*mut std::ffi::c_void{self as *const [i32;2] as *mut std::ffi::c_void}}
arg_struct!(TrajectoriesArg,LocalArg,JumpsArg,Seed);
impl Local{fn arg(&self)->LocalArg{LocalArg{m5:self.m5.pointer,m3:self.m3.pointer}}}

/// What the engine needs of a run.
#[derive(Clone,Debug)]
pub struct Settings{
    pub grid:Grid,
    /// The time step (us).
    pub dt:f64,
    /// The Doppler term's span, in steps.
    pub doppler_every:usize,
    pub trajectories:usize,
    pub seed:u64,
    /// The absorbing boundary, if any.
    pub cap:Option<Cap>,
}

/// An absorbing potential: rate × min(((ρ − start)/(half − start)), 1.5)² beyond the radius `start`
/// (um), `half` the grid's half width; an atom it absorbs is lost.
#[derive(Clone,Copy,Debug)]
pub struct Cap{pub start:f64,pub rate:f64}
impl Cap{
    pub fn at(&self,grid:&Grid,r:[f64;3])->f64{
        let rho=(r[0]*r[0]+r[1]*r[1]).sqrt();let u=((rho-self.start)/(grid.half_width()-self.start)).max(0.);self.rate*u.min(1.5).powi(2)
    }
}



/// Profiling's events and times.
struct Profile{events:Vec<Event>,phases:Vec<&'static str>,times:Vec<(&'static str,f64)>}

/// A trajectory's initial internal state: its block (0: F = 1, 1: F = 2) and components.
#[derive(Clone,Debug)]
pub struct Internal{pub block:usize,pub amplitudes:Vec<Complex>}

pub struct Engine{
    pub gpu:Rc<Gpu>,
    pub settings:Settings,
    points:usize,tiles:usize,
    state:Buffer,dx:Buffer,dy:Buffer,
    ids:Buffer,components:Buffer,scale:Buffer,log_norm:Buffer,threshold:Buffer,jumps:Buffer,weights:Buffer,
    partial:Buffer,jumpers:Buffer,count:Buffer,rates:Buffer,choices:Buffer,
    pub records:Buffer,marginal_x:Buffer,marginal_y:Buffer,density:Buffer,scratch_x:Buffer,scratch_y:Buffer,
    step_v:Local,back_v:Local,forward_v:Local,doppler_x:Local,doppler_y:Local,
    jump_table:Buffer,response_table:Buffer,radial_step:f32,radii:usize,beams:Buffer,beam_count:usize,cap:Buffer,wave_number:Buffer,carriers:usize,
    energies:Vec<Vec<f64>>,
    kinetic_full:[Buffer;2],kinetic_half:[Buffer;2],derivative:[Buffer;2],kappa:[Buffer;2],
    tails:[f32;4],
    planes:Planes,f_local:Function,f_kick_local:Function,f_epilogue:Function,
    /// Whether dx, dy hold κ_x φ, κ_y φ of the current φ (the last kinetic pass made them).
    derived:bool,
    f_jump_back:Function,f_jump_rates:Function,f_jump_choose:Function,f_jump_apply:Function,f_jump_finish:Function,
    f_observe_points:Function,f_thresholds:Function,
    /// The weight of each trajectory (thermal states' norms), and the steps made.
    pub trajectory_weights:Vec<f64>,
    /// The slots in use: the first `active` (the passes cover only them).
    pub active:usize,
    pub steps:u64,
    hbar_over_mass:f64,
    /// With profiling on: the step's events after each phase, and the milliseconds per phase.
    profile:Option<Profile>,
}

impl Engine{
    pub fn new(gpu:Rc<Gpu>,model:&Model,settings:Settings)->Result<Engine,String>{
        let g=settings.grid;let(nx,ny)=(g.nx,g.ny);
        if !GRIDS.contains(&(nx,ny)){return Err(format!("no kernels for a {nx} x {ny} grid"));}
        let points=g.points();let tiles=points.div_ceil(THREADS);let b=settings.trajectories;
        if model.radial.eliminations.is_empty()||model.radial.reach()<g.reach(){return Err("Model::tabulate must reach the grid's corners".into());}
        let cap=settings.cap;
        let cap_at=|r:[f64;3]|cap.map_or(0.,|c|c.at(&g,r));
        let table=model.points(&g,&cap_at);
        let dt=settings.dt;
        let step_v=Local::new(&gpu,&table,|p,k|propagator(&p.generator[k],dt))?;
        let back_v=Local::new(&gpu,&table,|p,k|propagator(&p.generator[k],-dt/2.))?;
        let forward_v=Local::new(&gpu,&table,|p,k|propagator(&p.generator[k],dt/2.))?;
        let doppler_x=Local::new(&gpu,&table,|p,k|p.doppler[k][0].clone())?;
        let doppler_y=Local::new(&gpu,&table,|p,k|p.doppler[k][1].clone())?;
        let ground=&model.ground;let carriers=ground.carriers.len();
        if carriers>2{return Err("2D 运动最多支持两组近共振载波".into());}
        let(ne,ng)=(ground.excited_states.len(),ground.ground_states.len());
        if ne!=24||ng!=8{return Err("2D 运动需要完整的 Rb87 32 态模型".into());}
        // The near beams' responses along the radius, linear between radii a quarter of the
        // model's radial step apart (its Catmull-Rom interpolation to RADIAL_TOLERANCE between
        // them): R_l and C_a R_l, C_a R2_l on the ground states in block order (F = 1, then F = 2).
        let near:Vec<(usize,usize)>=ground.carriers.iter().enumerate().flat_map(|(c,carrier)|carrier.beams.iter().map(move|&l|(c,l))).collect();
        let order:Vec<usize>=model.block_states[0].iter().chain(&model.block_states[1]).copied().collect();
        let step=model.radial.step/4.;let radii=(g.reach()/step).ceil() as usize+2;
        let slots=near.len().max(1);
        let columns:Vec<(Vec<[f32;2]>,Vec<[f32;2]>)>=(0..radii).into_par_iter().map(|k|{
            let e=model.radial.at(k as f64*step);
            let(mut jump,mut response)=(vec![[0f32;2];slots*2*3*8*8],vec![[0f32;2];slots*ne*8]);
            for(slot,&(_,l))in near.iter().enumerate(){
                for(term,m)in [&e.responses[l],&e.derivatives[l]].into_iter().enumerate(){for a in 0..3{
                    let cm=ground.emission[a].product(m);
                    for gr in 0..8{for h in 0..8{jump[(((slot*2+term)*3+a)*8+gr)*8+h]=complex32(cm[(order[gr],order[h])]);}}
                }}
                for i in 0..ne{for h in 0..8{response[(slot*ne+i)*8+h]=complex32(e.responses[l][(i,order[h])]);}}
            }
            (jump,response)
        }).collect();
        let mut jump_table=vec![[0f32;2];slots*2*3*8*8*radii];let mut response_table=vec![[0f32;2];slots*ne*8*radii];
        for(k,(jump,response))in columns.iter().enumerate(){
            for(i,z)in jump.iter().enumerate(){jump_table[i*radii+k]=*z;}
            for(i,z)in response.iter().enumerate(){response_table[i*radii+k]=*z;}
        }
        let hm=model.hbar_over_mass;
        let mut beam_table=vec![0f32;near.len().max(1)*8];
        for(slot,&(c,l))in near.iter().enumerate(){
            let k=model.in_plane[l];let beam=&ground.beams[l];
            beam_table[slot*8..slot*8+7].copy_from_slice(&[c as f32,k[0] as f32,k[1] as f32,beam.waist_um.map_or(0.,|w|1./(w*w)) as f32,
                (hm*(k[0]*k[0]+k[1]*k[1])/2.) as f32,(hm*k[0]) as f32,(hm*k[1]) as f32]);
        }
        let caps:Vec<f32>=(0..points).map(|p|cap_at(g.position(p)) as f32).collect();
        let wave_numbers:Vec<f32>=ground.carriers.iter().map(|c|c.wave_number as f32).collect();
        let up=|data:&[u8]|->Result<Buffer,String>{let b=Buffer::new(data.len())?;gpu.stream.upload(&b,data)?;Ok(b)};
        let bytes=|v:&[[f32;2]]|->Vec<u8>{v.iter().flat_map(|z|[z[0].to_le_bytes(),z[1].to_le_bytes()].concat()).collect()};
        let jump_table=up(&bytes(&jump_table))?;let response_table=up(&bytes(&response_table))?;
        let beams=up(&beam_table.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<u8>>())?;
        let cap_buffer=up(&caps.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<u8>>())?;
        let wave_number=up(&wave_numbers.iter().chain([0f32].iter()).flat_map(|x|x.to_le_bytes()).collect::<Vec<u8>>())?;
        // Kinetic factors per frequency position (bit-reversed), with each inverse transform's 1/N.
        let factors=|n:usize,f:&dyn Fn(f64)->Complex|->Vec<[f32;2]>{(0..n).map(|j|{let k=if n==1{0.}else{g.wave_number(bitrev(j,n),n)};complex32(f(k)*(1./n as f64))}).collect()};
        let pair=|f:&dyn Fn(f64)->Complex|->Result<[Buffer;2],String>{Ok([up(&bytes(&factors(nx,f)))?,up(&bytes(&factors(ny,f)))?])};
        let kinetic_full=pair(&|k|Complex::cis(-hm*k*k*dt/2.))?;
        let kinetic_half=pair(&|k|Complex::cis(-hm*k*k*dt/4.))?;
        let derivative=pair(&|k|Complex::new(k,0.))?;
        let kappa_of=|n:usize|->Vec<u8>{(0..n).flat_map(|j|(if n==1{0.}else{g.wave_number(bitrev(j,n),n)} as f32).to_le_bytes()).collect()};
        let kappa=[up(&kappa_of(nx))?,up(&kappa_of(ny))?];
        let reach=0.7;let kmax=std::f64::consts::PI/g.dx;
        let tails=[(reach*g.half_width()) as f32,(reach*g.half_width()) as f32,(reach*kmax) as f32,(reach*kmax) as f32];
        let energies:Vec<Vec<f64>>=(0..ng).map(|b|table.iter().map(|p|p.energies[b]).collect()).collect();
        let planes=Planes::new(&gpu,nx,ny)?;
        let zeroed=|bytes:usize|->Result<Buffer,String>{let b=Buffer::new(bytes)?;gpu.stream.zero(&b)?;Ok(b)};
        let state_bytes=b*COMPONENTS*points*8;
        Ok(Engine{
            state:zeroed(state_bytes)?,dx:zeroed(state_bytes)?,dy:zeroed(state_bytes)?,
            ids:{let v:Vec<u32>=(0..b as u32).collect();let x=Buffer::new(b*4)?;gpu.stream.upload(&x,&v)?;x},components:zeroed(b*4)?,scale:zeroed(b*4)?,log_norm:zeroed(b*8)?,threshold:zeroed(b*8)?,jumps:zeroed(b*4)?,weights:zeroed(b*4)?,
            partial:zeroed(b*tiles*4)?,jumpers:zeroed(b*4)?,count:zeroed(4)?,rates:zeroed(b*tiles*G_SLOTS*4)?,choices:zeroed(b*CHOICE*4)?,
            records:zeroed(b*RECORD*4)?,marginal_x:zeroed(nx*4)?,marginal_y:zeroed(ny*4)?,density:zeroed(points*4)?,scratch_x:zeroed(nx*4)?,scratch_y:zeroed(ny*4)?,
            step_v,back_v,forward_v,doppler_x,doppler_y,jump_table,response_table,radial_step:step as f32,radii,beams,beam_count:near.len(),
            cap:cap_buffer,wave_number,carriers,energies,
            kinetic_full,kinetic_half,derivative,kappa,tails,
            planes,
            f_local:gpu.function("local_step")?,f_kick_local:gpu.function("kick_local_step")?,f_epilogue:gpu.function("epilogue")?,derived:false,
            f_jump_back:gpu.function("jump_back")?,f_jump_rates:gpu.function("jump_rates")?,f_jump_choose:gpu.function("jump_choose")?,
            f_jump_apply:gpu.function("jump_apply")?,f_jump_finish:gpu.function("jump_finish")?,f_observe_points:gpu.function("observe_points")?,
            f_thresholds:gpu.function("thresholds")?,
            points,tiles,trajectory_weights:vec![1.;b],active:b,steps:0,hbar_over_mass:model.hbar_over_mass,profile:None,gpu,settings,
        })
    }

    fn seed(&self,stream:u64)->Seed{let s=self.settings.seed^stream.wrapping_mul(0x9E3779B97F4A7C15);Seed{lo:s as u32,hi:(s>>32) as u32}}
    /// Trajectories moved in or out on the device: the derivatives no longer match.
    pub(crate) fn touched(&mut self){self.derived=false;}
    /// The device addresses the move kernels take: the trajectories' state, their planes and weights.
    pub(crate) fn device(&self)->(TrajectoriesArg,u64,u64,u64){(self.trajectories_arg(),self.state.pointer,self.weights.pointer,self.components.pointer)}
    pub(crate) fn trajectories_arg(&self)->TrajectoriesArg{
        TrajectoriesArg{ids:self.ids.pointer,components:self.components.pointer,scale:self.scale.pointer,log_norm:self.log_norm.pointer,threshold:self.threshold.pointer,jumps:self.jumps.pointer}
    }
    fn jumps_arg(&self)->JumpsArg{
        let g=self.settings.grid;
        JumpsArg{jump:self.jump_table.pointer,response:self.response_table.pointer,step:self.radial_step,radii:self.radii as i32,beams:self.beams.pointer,beam_count:self.beam_count as i32,
            cap:self.cap.pointer,wave_number:self.wave_number.pointer,carriers:self.carriers as i32,nx:g.nx as i32,dx:g.dx as f32}
    }

    /// The thermal initial states of the first slots (one per given internal state; the others
    /// emptied): per trajectory its internal state, its motion the random-phase
    /// thermal pure state of temperature T (uK) in the potential of its internal state at the
    /// grid points, ⟨φ|diag E_b(x)|φ⟩; each trajectory weighted by its norm, normalized within the
    /// trajectories of one internal state.
    pub fn initialize(&mut self,internal:&[Internal],temperature_uk:f64,model:&Model)->Result<(),String>{
        let b=internal.len();assert!(b<=self.settings.trajectories);self.derived=false;
        let(points,g)=(self.points,self.settings.grid);
        // The distinct internal states and their potentials.
        let mut kinds:Vec<&Internal>=vec![];let mut kind=vec![0i32;b];
        for(t,i)in internal.iter().enumerate(){
            let k=kinds.iter().position(|x|x.block==i.block&&x.amplitudes==i.amplitudes).unwrap_or_else(||{kinds.push(i);kinds.len()-1});kind[t]=k as i32;
        }
        let beta=crate::model::HBAR/(crate::model::BOLTZMANN*temperature_uk*1e-6)*1e6; // 1/(k_B T) in us (energies in rad/us)
        let tau=beta/2.;let steps=((tau/self.settings.dt).ceil() as usize).max(1);let h=tau/steps as f64;
        let mut factors=vec![0f32;kinds.len()*points];
        for(k,i)in kinds.iter().enumerate(){
            let u:Vec<f64>=(0..points).map(|p|i.amplitudes.iter().enumerate().map(|(c,a)|a.norm_sqr()*self.energies[model.block_states[i.block][c]][p]).sum()).collect();
            let low=u.iter().copied().fold(f64::INFINITY,f64::min);
            for p in 0..points{factors[k*points+p]=(-(u[p]-low)*h/2.).exp() as f32;}
        }
        let noise=self.gpu.function("thermal_noise")?;let potential=self.gpu.function("thermal_potential")?;
        let norms=self.gpu.function("plane_norms")?;let expand=self.gpu.function("thermal_expand")?;
        let s=&self.gpu.stream;
        let factor_buffer=Buffer::new(factors.len()*4)?;s.upload(&factor_buffer,&factors)?;
        let kind_buffer=Buffer::new(b*4)?;s.upload(&kind_buffer,&kind)?;
        s.zero(&self.components)?;s.upload(&self.components,&vec![1i32;b])?;s.upload(&self.scale,&vec![1f32;b])?;self.active=b;
        let hm=model.hbar_over_mass;
        let real=|n:usize|->Vec<[f32;2]>{(0..n).map(|j|{let k=if n==1{0.}else{g.wave_number(bitrev(j,n),n)};[((-hm*k*k*h/2.).exp()/n as f64) as f32,0.]}).collect()};
        let imaginary=[Buffer::new(g.nx*8)?,Buffer::new(g.ny*8)?];s.upload(&imaginary[0],&real(g.nx))?;s.upload(&imaginary[1],&real(g.ny))?;
        let grid2=(self.tiles as u32,b as u32,1);
        s.launch(&noise,grid2,THREADS as u32,0,&[&self.state.pointer,&(points as i32),&(b as i32),&self.ids.pointer,&self.seed(1)])?;
        for _ in 0..steps{
            s.launch(&potential,grid2,THREADS as u32,0,&[&self.state.pointer,&(points as i32),&(b as i32),&factor_buffer.pointer,&kind_buffer.pointer])?;
            self.planes.kinetic(s,&self.state,&self.components,&self.scale,&imaginary,b)?;
            s.launch(&potential,grid2,THREADS as u32,0,&[&self.state.pointer,&(points as i32),&(b as i32),&factor_buffer.pointer,&kind_buffer.pointer])?;
        }
        s.launch(&norms,grid2,THREADS as u32,0,&[&self.state.pointer,&(points as i32),&self.partial.pointer])?;
        let mut partial=vec![0f32;b*self.tiles];s.download(&self.partial,&mut partial)?;
        let norm2:Vec<f64>=(0..b).map(|t|partial[t*self.tiles..(t+1)*self.tiles].iter().map(|&x|x as f64).sum()).collect();
        // Weights: each trajectory's norm relative to the mean of its internal state's.
        let mut weights=vec![0f64;b];
        for k in 0..kinds.len(){
            let members:Vec<usize>=(0..b).filter(|&t|kind[t]==k as i32).collect();
            let mean=members.iter().map(|&t|norm2[t]).sum::<f64>()/members.len() as f64;
            for &t in &members{weights[t]=norm2[t]/mean;}
        }
        let inverse:Vec<f32>=norm2.iter().map(|&n|(1./n.sqrt()) as f32).collect();
        let mut amplitudes=vec![[0f32;2];b*5];
        for(t,i)in internal.iter().enumerate(){for(c,a)in i.amplitudes.iter().enumerate(){amplitudes[t*5+c]=complex32(*a);}}
        let(inv_buffer,amp_buffer)=(Buffer::new(b*4)?,Buffer::new(b*40)?);s.upload(&inv_buffer,&inverse)?;s.upload(&amp_buffer,&amplitudes)?;
        let components:Vec<i32>=internal.iter().map(|i|if i.block==1{5}else{3}).collect();s.upload(&self.components,&components)?;
        s.launch(&expand,grid2,THREADS as u32,0,&[&self.state.pointer,&(points as i32),&(b as i32),&self.components.pointer,&amp_buffer.pointer,&inv_buffer.pointer])?;
        s.upload(&self.weights,&weights.iter().map(|&w|w as f32).collect::<Vec<f32>>())?;
        self.trajectory_weights=weights;
        s.zero(&self.log_norm)?;s.zero(&self.jumps)?;s.upload(&self.scale,&vec![1f32;b])?;
        s.launch(&self.f_thresholds,(b.div_ceil(THREADS) as u32,1,1),THREADS as u32,0,&[&self.threshold.pointer,&self.ids.pointer,&self.jumps.pointer,&(b as i32),&self.seed(0)])?;
        s.synchronize()
    }

    fn launch_kinetic(&self,tables:&[Buffer;2])->Result<(),String>{
        self.planes.kinetic(&self.gpu.stream,&self.state,&self.components,&self.scale,tables,self.active)
    }
    /// One step of φ; with `observe`, its K split around ψ's observation (records, and with
    /// `distributions` the marginals and the density).
    pub fn step(&mut self,observe:bool,distributions:bool)->Result<(),String>{
        let b=self.active;let(points,tiles)=(self.points,self.tiles);if b==0{return Ok(());}
        let t=THREADS as u32;
        let(p,n,bb)=(points as i32,tiles as i32,b as i32);
        let grid_traj=(tiles as u32,b.div_ceil(SUB) as u32,1);
        self.mark("")?;
        // Every `every` steps the Doppler term over them, an explicit kick of φ before the local
        // step (κφ from the last kinetic pass, or derived here).
        let every=self.settings.doppler_every.max(1) as u64;
        if (self.steps+1)%every==0{
            if !self.derived{self.planes.derive(&self.gpu.stream,&self.state,&self.components,None,b,&self.dx,&self.dy,&self.derivative)?;self.mark("derive")?;}
            let tau=(every as f64*self.settings.dt) as f32;
            self.gpu.stream.launch(&self.f_kick_local,grid_traj,t,0,&[&self.state.pointer,&self.dx.pointer,&self.dy.pointer,&p,&bb,&self.components.pointer,
                &self.doppler_x.arg(),&self.doppler_y.arg(),&self.step_v.arg(),&tau,&self.partial.pointer])?;
            self.mark("doppler+local")?;
        }else{
            self.gpu.stream.launch(&self.f_local,grid_traj,t,0,&[&self.state.pointer,&p,&bb,&self.components.pointer,&self.step_v.arg(),&self.partial.pointer])?;
            self.mark("local")?;
        }
        self.derived=false;
        self.gpu.stream.zero(&self.count)?;
        self.gpu.stream.launch(&self.f_epilogue,(b.div_ceil(THREADS) as u32,1,1),t,0,&[&self.trajectories_arg(),&bb,&self.partial.pointer,&n,&self.jumpers.pointer,&self.count.pointer])?;
        self.mark("epilogue")?;
        self.jump()?;
        if observe{
            self.gpu.stream.zero(&self.records)?;
            self.observe_momentum(distributions)?;
            self.observe_points(distributions)?;
            self.launch_kinetic(&self.kinetic_half)?;
            self.mark("observe+kinetic")?;
        }else if (self.steps+2)%every==0&&self.planes.kinetic_derive(&self.gpu.stream,&self.state,&self.components,&self.scale,&self.kinetic_full,&self.derivative,&self.dx,&self.dy,b)?{
            // The next step kicks: κφ with the kinetic pass.
            self.derived=true;self.mark("kinetic+derive")?;
        }else{self.launch_kinetic(&self.kinetic_full)?;self.mark("kinetic")?;}
        self.timed()?;
        self.steps+=1;
        Ok(())
    }
    /// Profiling: on, every step is timed by phase (and synchronized).
    pub fn set_profiling(&mut self,on:bool)->Result<(),String>{
        self.profile=if on{Some(Profile{events:(0..16).map(|_|Event::new()).collect::<Result<Vec<_>,_>>()?,phases:vec![],times:vec![]})}else{None};Ok(())
    }
    /// Profiling: an event after the phase (an empty phase starts a step); the phases run on
    /// without waiting, timed when the step ends (`timed`).
    fn mark(&mut self,phase:&'static str)->Result<(),String>{
        let Some(p)=&mut self.profile else{return Ok(())};
        if phase.is_empty(){p.phases.clear();}
        self.gpu.stream.record(&p.events[p.phases.len()])?;p.phases.push(phase);Ok(())
    }
    /// Profiling: the ended step's phases added to the times.
    fn timed(&mut self)->Result<(),String>{
        let Some(p)=&mut self.profile else{return Ok(())};
        let n=p.phases.len();if n<2{return Ok(());}
        p.events[n-1].synchronize()?;
        for i in 1..n{
            let(phase,ms)=(p.phases[i],p.events[i].since(&p.events[i-1])?);
            match p.times.iter_mut().find(|(q,_)|*q==phase){Some(e)=>e.1+=ms,None=>p.times.push((phase,ms))}
        }
        Ok(())
    }
    /// The milliseconds per phase since profiling was turned on.
    pub fn profile(&self)->Vec<(&'static str,f64)>{self.profile.as_ref().map_or(vec![],|p|p.times.clone())}
    /// The observation of the initial state ψ, and the first half kick φ = K(dt/2) ψ.
    pub fn observe_initial(&mut self,distributions:bool)->Result<(),String>{
        self.derived=false;
        self.gpu.stream.zero(&self.records)?;
        self.observe_points(distributions)?;
        self.observe_momentum(distributions)
    }

    /// The step's jumps (kernels `jump_*`), whatever their number: the grids stride over them.
    fn jump(&mut self)->Result<(),String>{
        let s=&self.gpu.stream;let t=THREADS as u32;let(points,tiles)=(self.points,self.tiles);
        let lanes=(self.active.min(64)) as u32;
        let(p,n)=(points as i32,tiles as i32);let seed=self.seed(0);let b=self.active as i32;
        s.launch(&self.f_jump_back,(tiles as u32,lanes,1),t,0,&[&self.state.pointer,&p,&self.components.pointer,&self.jumpers.pointer,&self.count.pointer,&self.back_v.arg()])?;
        self.mark("jump back")?;let s=&self.gpu.stream;
        self.planes.derive(s,&self.state,&self.components,Some((&self.jumpers,&self.count)),b as usize,&self.dx,&self.dy,&self.derivative)?;
        self.mark("jump derive")?;let s=&self.gpu.stream;
        s.launch(&self.f_jump_rates,(tiles as u32,lanes,1),t,0,&[&self.state.pointer,&self.dx.pointer,&self.dy.pointer,&p,&self.components.pointer,&self.jumpers.pointer,&self.count.pointer,&self.jumps_arg(),&self.rates.pointer])?;
        self.mark("jump rates")?;let s=&self.gpu.stream;
        s.launch(&self.f_jump_choose,(self.active.div_ceil(2) as u32,1,1),64,0,&[&self.trajectories_arg(),&self.jumpers.pointer,&self.count.pointer,&self.rates.pointer,&n,&self.jumps_arg(),&seed,&self.choices.pointer])?;
        self.mark("jump choose")?;let s=&self.gpu.stream;
        s.launch(&self.f_jump_apply,(tiles as u32,lanes,1),t,0,&[&self.state.pointer,&self.dx.pointer,&self.dy.pointer,&p,&self.components.pointer,&self.jumpers.pointer,&self.count.pointer,&self.jumps_arg(),&self.choices.pointer,&self.forward_v.arg(),&self.partial.pointer])?;
        self.mark("jump apply")?;let s=&self.gpu.stream;
        s.launch(&self.f_jump_finish,(self.active.div_ceil(64) as u32,1,1),64,0,&[&self.trajectories_arg(),&self.jumpers.pointer,&self.count.pointer,&self.choices.pointer,&self.partial.pointer,&n,&seed])?;
        self.mark("jump finish")
    }

    /// A half kick K(dt/2) that adds the momentum observables to the records (and with
    /// `distributions` the weighted momentum marginals); it applies the pending renormalization.
    fn observe_momentum(&self,distributions:bool)->Result<(),String>{
        let s=&self.gpu.stream;let b=self.active;
        if !distributions{s.zero(&self.scratch_x)?;s.zero(&self.scratch_y)?;}
        let(mx,my)=if distributions{(self.marginal_x.pointer,self.marginal_y.pointer)}else{(self.scratch_x.pointer,self.scratch_y.pointer)};
        self.planes.observe(s,&self.state,&self.components,&self.scale,&self.kinetic_half,&self.kappa,[self.tails[2],self.tails[3]],&self.records,&self.weights,[mx,my],b)?;
        s.upload(&self.scale,&vec![1f32;b])
    }
    /// The position observables of ψ added to the records (with `distributions` the density).
    fn observe_points(&self,distributions:bool)->Result<(),String>{
        let s=&self.gpu.stream;let b=self.active;
        let density=if distributions{self.density.pointer}else{0u64};
        s.launch(&self.f_observe_points,(self.tiles as u32,b.div_ceil(SUB) as u32,1),THREADS as u32,0,&[&self.state.pointer,&(self.points as i32),&(b as i32),
            &self.components.pointer,&self.jumps_arg(),&self.tails[0],&self.tails[1],&self.records.pointer,&self.weights.pointer,&density])
    }

    pub fn capacity(&self)->usize{self.settings.trajectories}
    pub fn hbar_over_mass(&self)->f64{self.hbar_over_mass}
    /// A new random seed for the trajectories' streams.
    pub fn reseed(&mut self,seed:u64){self.settings.seed=seed;}
    /// The trajectories' weights (the distributions' sums use them).
    pub fn set_weights(&mut self,weights:&[f64])->Result<(),String>{
        self.gpu.stream.upload(&self.weights,&weights.iter().map(|&w|w as f32).collect::<Vec<f32>>())?;self.trajectory_weights=weights.to_vec();Ok(())
    }
    /// The trajectory each slot holds (its random streams).
    pub fn set_ids(&self,ids:&[u32])->Result<(),String>{self.gpu.stream.upload(&self.ids,ids)}
    /// Copies slot `from`'s trajectory into slot `to` and empties `from` (on the device, in order).
    pub fn copy_slot(&mut self,from:usize,to:usize)->Result<(),String>{
        self.derived=false;
        let s=&self.gpu.stream;let plane=COMPONENTS*self.points*8;
        s.copy(&self.state,to*plane,&self.state,from*plane,plane)?;
        for(buffer,size)in [(&self.ids,4),(&self.components,4),(&self.scale,4),(&self.log_norm,8),(&self.threshold,8),(&self.jumps,4),(&self.weights,4)]{s.copy(buffer,to*size,buffer,from*size,size)?;}
        s.upload_at(&self.components,from*4,&[0i32])
    }
    /// Empties every slot.
    pub fn clear(&self)->Result<(),String>{self.gpu.stream.zero(&self.components)}
    pub fn grid(&self)->Grid{self.settings.grid}

    /// The records of the last observation, and the trajectories' blocks (0: lost).
    pub fn read_records(&self)->Result<(Vec<f32>,Vec<i32>),String>{
        let b=self.active;let mut r=vec![0f32;b*RECORD];if b>0{self.gpu.stream.download(&self.records,&mut r)?;}
        let mut c=vec![0i32;b];if b>0{self.gpu.stream.download(&self.components,&mut c)?;}Ok((r,c))
    }
    /// The accumulated distributions: momentum marginals by frequency (FFT order) and the density.
    pub fn read_distributions(&self)->Result<(Vec<f32>,Vec<f32>,Vec<f32>),String>{
        let g=self.settings.grid;let(mut x,mut y,mut d)=(vec![0f32;g.nx],vec![0f32;g.ny],vec![0f32;g.points()]);
        self.gpu.stream.download(&self.marginal_x,&mut x)?;self.gpu.stream.download(&self.marginal_y,&mut y)?;self.gpu.stream.download(&self.density,&mut d)?;
        // From bit-reversed positions to frequencies.
        let unscramble=|v:&[f32]|->Vec<f32>{let n=v.len();let mut out=vec![0f32;n];for j in 0..n{out[bitrev(j,n)]=v[j];}out};
        Ok((unscramble(&x),unscramble(&y),d))
    }
    pub fn clear_distributions(&self)->Result<(),String>{let s=&self.gpu.stream;s.zero(&self.marginal_x)?;s.zero(&self.marginal_y)?;s.zero(&self.density)}
    /// The whole state, for tests.
    pub fn download_state(&self)->Result<Vec<[f32;2]>,String>{let mut v=vec![[0f32;2];self.state.bytes/8];self.gpu.stream.download(&self.state,&mut v)?;Ok(v)}
    pub fn upload_state(&self,data:&[[f32;2]])->Result<(),String>{self.gpu.stream.upload(&self.state,data)}
    pub fn points(&self)->usize{self.points}
}
