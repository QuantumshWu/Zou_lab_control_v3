//! Semiclassical 3D motion of a single Rb87 atom in an optical tweezer: a classical point (r, p)
//! in 3D carrying the internal state of the eliminated ground manifold (`ground`), evolved by
//! quantum jumps. The request (`request`) is the scene's model with its motion block; `ground` is
//! its front end, the generator with the excited states eliminated, on the small dense linear
//! algebra of `linear`.
pub mod control;
pub mod eigen;
pub mod engine;
pub mod gpu;
pub mod ground;
pub mod ledger;
pub mod linear;
pub mod model;
pub mod request;
pub mod rng;
pub mod solve;
pub mod stats;

pub use request::{Beam,Complex,Matrix,Request,STATES};
use engine::{Engine,Settings,Trajectory};
use model::{Model,Trap};
use rayon::prelude::*;
use rng::{normal_quantile,Purpose,Rng};
use control::{Harmonic,Twin};
use stats::{Atom,Controls,Group,KT_PER_UK};
use std::sync::Mutex;
use std::time::Instant;

/// The initial internal states (cluster, position in it, probability), each one's trap (the
/// harmonic trap its initial positions are drawn in) and depth (`Model::depth`), the temperature,
/// and the share of the traps' thermal positions beyond their regions (`thermal_position`), booked
/// as an a-priori estimate: what a region twice as deep would add, the initial states' mean.
#[derive(Clone,Debug)]
pub struct Initial{pub states:Vec<(usize,usize,f64)>,pub traps:Vec<Trap>,pub depths:Vec<f64>,pub temperature_uk:f64,pub outside:f64}
impl Initial{
    /// From the request's initial density: its ground populations (the density is diagonal in the
    /// model basis, as the app writes it).
    pub fn new(model:&Model,request:&Request)->Result<Initial,String>{
        let g=&model.ground;let rho=&request.initial_density;let mut states=vec![];
        for i in 0..STATES{for j in 0..STATES{if i!=j&&rho[i*STATES+j].abs()>0.{return Err("The initial density must be diagonal in the model basis".into());}}}
        for(k,cluster)in g.clusters.iter().enumerate(){for(x,&b)in cluster.states.iter().enumerate(){
            let p=rho[g.ground_states[b]*STATES+g.ground_states[b]].re;if p>0.{states.push((k,x,p));}
        }}
        let total:f64=states.iter().map(|s|s.2).sum();
        if !(total>0.){return Err("The initial density has no ground population".into());}
        for s in &mut states{s.2/=total;}
        let traps:Vec<Trap>=states.iter().map(|&(k,x,_)|model.trap(k,x)).collect();
        let depths:Vec<f64>=states.iter().map(|&(k,x,_)|model.depth(k,x)).collect();
        let kt=KT_PER_UK*request.temperature_uk;
        let outside=states.iter().zip(&traps).zip(&depths).enumerate().map(|(i,((&(k,x,p),trap),&depth))|p*outside_share(model,k,x,trap,depth,kt,i as u64)).sum();
        Ok(Initial{states,traps,depths,temperature_uk:request.temperature_uk,outside})
    }
}

/// The draws of a region's share (`outside_share`): where the share reaches the a-priori budget,
/// to about a fifth of it.
const REGION_DRAWS:u64=4096;

/// A state's harmonic trap at 2T: the scales σ_i √2 of its trapped principal axes (√(2 k_BT ħ/m)/ω_i,
/// zero along an untrapped one), in which its harmonic potential is U_h = k_BT |z|² (z = q/scale).
fn doubled_scales(trap:&Trap,kt:f64,hm:f64)->[f64;3]{std::array::from_fn(|i|{let w=trap.frequencies[i];if w>0.{(2.*kt*hm).sqrt()/w}else{0.}})}

/// The trap's thermal positions at T (`thermal_position`) that its region leaves out: the share of
/// e^{−U/k_BT} over the region twice as deep (U_h ≤ 2D) that lies beyond the trap's (U_h ≤ D), by
/// importance sampling from the proposal `thermal_position` takes there (on a fixed stream: the
/// estimate is deterministic). One where the region holds nothing of it; zero for a state the trap
/// does not hold (its positions are the focus).
fn outside_share(model:&Model,cluster:usize,state:usize,trap:&Trap,depth:f64,kt:f64,index:u64)->f64{
    let scales=doubled_scales(trap,kt,model.hbar_over_mass);let trapped=scales.iter().filter(|&&s|s>0.).count();
    if trapped==0||!(depth>0.){return 0.;}
    let(inner,outer)=(depth/kt,2.*depth/kt);let normal=outer>=trapped as f64;let edge=outer.sqrt();
    let mut rng=Rng::new(0x3D_5EED,Purpose::Region,index,0);let(mut all,mut beyond)=(0.,0.);
    for _ in 0..REGION_DRAWS{
        let z:[f64;3]=std::array::from_fn(|i|if scales[i]>0.{if normal{normal_quantile(rng.interior())}else{edge*(2.*rng.uniform()-1.)}}else{0.});
        let z2:f64=z.iter().map(|x|x*x).sum();if z2>outer{continue;}
        let q:[f64;3]=std::array::from_fn(|i|z[i]*scales[i]);let lab=std::array::from_fn(|a|(0..3).map(|i|q[i]*trap.axes[i][a]).sum());
        let w=(-model.potential(cluster,state,lab)/kt+if normal{0.5*z2}else{0.}).exp();
        all+=w;if z2>inner{beyond+=w;}
    }
    if all>0.{beyond/all}else{1.}
}

/// A state's thermal position on its trap's principal axes: the trap's Boltzmann distribution at T,
/// e^{−U/k_BT} (U the state's potential, `Model::potential`), over the trap's region, where its
/// harmonic approximation lies below its depth, U_h ≤ D (`Model::depth`; a solve's loss takes the
/// reference state's, engine.rs `Settings::region`). A finite trap's Boltzmann factor does not fall
/// off past its depth -- U saturates there --, so no region holds all of it: the region is the trap's,
/// the share a twice as deep one would add booked (`outside_share`). Drawn
/// exactly, by rejection, in the coordinates z of the harmonic trap at 2T (U_h = k_BT |z|², the
/// region |z|² ≤ D/k_BT): from the normal distribution there, accepted with e^{−U/k_BT + |z|²/2} --
/// at most one where U ≥ U_h/2, as on the region of a Gaussian beam (it lies below its harmonic
/// potential by at most half, on the axis at U_h = D) and of its standing wave; or, where the region
/// is shallower than the normal's mean |z|² (the trapped axes' number), uniformly over the cube
/// about it, accepted with e^{−U/k_BT} on it. The first proposal's deviates are the member's of its
/// group's Latin hypercube (`next`), the further ones its own stream's. Untrapped axes: zero; a state
/// the trap does not hold: the focus.
fn thermal_position(model:&Model,cluster:usize,state:usize,trap:&Trap,depth:f64,kt:f64,rng:&mut Rng,next:&mut dyn FnMut(&mut Rng)->[f64;3])->[f64;3]{
    let scales=doubled_scales(trap,kt,model.hbar_over_mass);let trapped=scales.iter().filter(|&&s|s>0.).count();
    // The member's Latin deviates are taken whatever follows, so that its group's stay in step.
    let first=next(rng);
    if trapped==0||!(depth>0.){return [0.;3];}
    let region=depth/kt;let normal=region>=trapped as f64;let edge=region.sqrt();
    let mut deviates=first;
    loop{
        let z:[f64;3]=std::array::from_fn(|i|if scales[i]>0.{if normal{normal_quantile(deviates[i])}else{edge*(2.*deviates[i]-1.)}}else{0.});
        let z2:f64=z.iter().map(|x|x*x).sum();
        if z2<=region{
            let q:[f64;3]=std::array::from_fn(|i|z[i]*scales[i]);let lab=std::array::from_fn(|a|(0..3).map(|i|q[i]*trap.axes[i][a]).sum());
            let accept=-model.potential(cluster,state,lab)/kt+if normal{0.5*z2}else{0.};
            if rng.open_uniform().ln()<accept{return q;}
        }
        deviates=std::array::from_fn(|_|rng.interior());
    }
}

/// A trajectory's group: its first trajectory and its size. A group's initial states form a Latin
/// hypercube: every variable drawn from a uniform deviate -- the internal state's pick, each
/// coordinate of each Metropolis proposal, each momentum component -- falls for each member in
/// another of `size` equal-probability strata (the members' order a random permutation of the
/// strata, new for every variable) at a uniform point inside it. Each member is distributed as an
/// independent draw would be; the group spreads over every variable's distribution, so its means of
/// the initial energies, and of what they leave at early times, vary much less than independent
/// draws'. Groups are independent of each other, so the jackknife over them still measures the
/// error. A group of one is an independent draw.
#[derive(Clone,Copy,Debug)]
pub struct Stratum{pub first:u64,pub size:u64}
impl Stratum{
    /// A trajectory alone.
    pub fn single(index:u64)->Stratum{Stratum{first:index,size:1}}
}
/// One member's walk through its group's Latin hypercube.
struct Latin{shared:Rng,order:Vec<usize>,member:usize}
impl Latin{
    fn new(seed:u64,stratum:Stratum,index:u64)->Latin{
        Latin{shared:Rng::new(seed,Purpose::Latin,stratum.first,0),order:(0..stratum.size as usize).collect(),member:(index-stratum.first) as usize}
    }
    /// The next variable's uniform deviate: the member's stratum in a new random order of the
    /// group's strata (Fisher–Yates, the same for every member), a uniform point inside it.
    fn next(&mut self,own:&mut Rng)->f64{
        let n=self.order.len();
        for i in (1..n).rev(){let j=(self.shared.next_u64()%(i as u64+1)) as usize;self.order.swap(i,j);}
        (self.order[self.member] as f64+own.interior())/n as f64
    }
}

/// A trajectory's initial state, its group's member (`Stratum`): an internal basis state drawn from
/// the initial populations; the position from the Boltzmann distribution at T in that state's trap
/// over the trap's region (`thermal_position`, on the trap's principal axes); the momentum Maxwell,
/// apart from the position -- an atom the trap cannot hold is lost at the first step --; position
/// and momentum along each principal axis scaled to the harmonic oscillator's thermal Wigner
/// function, T* = (ħω/2k)coth(ħω/2kT). Along frozen lab axes (a validation's), nothing. At fourth
/// order the internal state is the bare state's slow part (engine.rs `slow_start`).
pub fn initial_trajectory(model:&Model,settings:&Settings,initial:&Initial,seed:u64,index:u64,stratum:Stratum)->Trajectory{
    dressed(model,initial_from(model,settings,initial,seed,Purpose::Initial,index,stratum))
}
/// A bare initial state's slow part at fourth order, where it starts (engine.rs `slow_start`).
pub(crate) fn dressed(model:&Model,mut t:Trajectory)->Trajectory{let point=model.point(t.r,t.kappa);engine::slow_start(model,&mut t,&point,0.);t}
/// A pilot trajectory's initial state (`sample_states`; solve.rs, the fold's charge at t = 0).
pub(crate) fn pilot_start(model:&Model,settings:&Settings,initial:&Initial,seed:u64,index:u64)->Trajectory{
    dressed(model,initial_from(model,settings,initial,seed,Purpose::Pilot,index,Stratum::single(index)))
}
/// The bare initial state of a solve's trajectory, which a held twin starts from (`hold`; gpu/mod.rs).
pub(crate) fn bare_trajectory(model:&Model,settings:&Settings,initial:&Initial,seed:u64,index:u64,stratum:Stratum)->Trajectory{
    initial_from(model,settings,initial,seed,Purpose::Initial,index,stratum)
}
/// The bare initial state: the internal basis state, its position and momentum (`initial_trajectory`).
fn initial_from(model:&Model,settings:&Settings,initial:&Initial,seed:u64,purpose:Purpose,index:u64,stratum:Stratum)->Trajectory{
    let mut rng=Rng::new(seed,purpose,index,0);let mut latin=Latin::new(seed,stratum,index);
    let mut x=latin.next(&mut rng);let mut pick=initial.states.len()-1;
    for(i,s)in initial.states.iter().enumerate(){if x<s.2{pick=i;break;}x-=s.2;}
    let(cluster,state,_)=initial.states[pick];let trap=&initial.traps[pick];
    // The axes: the trap's principal ones when every lab axis is free, else the free lab axes.
    let all=settings.free==[true;3];
    let(axes,frequencies)=if all{(trap.axes,trap.frequencies)}else{([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],std::array::from_fn(|a|if settings.free[a]{trap.lab[a]}else{0.}))};
    let kt=KT_PER_UK*initial.temperature_uk;let hm=model.hbar_over_mass;
    let sigma:[f64;3]=std::array::from_fn(|i|if frequencies[i]>0.{(kt*hm).sqrt()/frequencies[i]}else{0.});
    let lab=|q:[f64;3]|->[f64;3]{std::array::from_fn(|a|(0..3).map(|i|q[i]*axes[i][a]).sum())};
    // The position: along the trap's principal axes in the trap's region; along the free lab axes of
    // a validation, that trap with the frozen axes' frequencies zero (none there).
    let held=Trap{axes,frequencies,lab:trap.lab};
    let mut q=thermal_position(model,cluster,state,&held,initial.depths[pick],kt,&mut rng,&mut |rng|std::array::from_fn(|_|latin.next(rng)));
    let quantum:[f64;3]=std::array::from_fn(|i|{let f=frequencies[i];if f>0.{let half=f/(2.*kt);(half/half.tanh()).sqrt()}else{1.}});
    for i in 0..3{q[i]*=quantum[i];}
    let p:[f64;3]=std::array::from_fn(|i|{let z=normal_quantile(latin.next(&mut rng));if sigma[i]>0.{(kt/hm).sqrt()*quantum[i]*z}else{0.}});
    let mut psi=[Complex::ZERO;model::MOST_MIXED];psi[state]=Complex::ONE;
    let threshold=engine::jump_stream(seed,index,0).open_uniform();
    Trajectory{cluster,psi,r:lab(q),kappa:lab(p),time:0.,threshold,alive:true,jumps:0,doppler:0.,beats:0.,seed,index}
}

/// The outputs of a run: one group per `group` trajectories, sums at every output time, the held
/// atoms' (r, κ) at the listed outputs, the jumps made, the largest first-order (kv/σ_min)², the
/// largest bound per unit time of the beats the exact steps left out (`Trajectory::beats`), and the
/// seconds (summed over the threads) its trajectories and their held twins took.
pub struct Outcome{pub groups:Vec<Group>,pub samples:Vec<Vec<([f64;3],[f64;3])>>,pub jumps:u64,pub doppler:f64,pub beats:f64,pub seconds:[f64;2]}

/// The control variates a run computes (control.rs): the harmonic twin's control of every trajectory
/// and, with a held twin, the twin of every trajectory.
pub struct Variates<'a>{pub harmonic:&'a Harmonic,pub twin:Option<&'a Twin>}

/// The most trajectories a task sums one after another: its sums (one per output time) stay few
/// while the trajectories run one at a time on whichever thread is free.
pub(crate) const TASK:u64=8;

/// Trajectories [first, first + count) of the run `seed`, `steps` steps of the settings' dt between
/// `outputs` output times (the first at t = 0), in groups of `group` (each a Latin hypercube of
/// initial states, `Stratum`, and a jackknife group), summed as tasks of at most `TASK` trajectories
/// inside a group; (r, κ) kept at `listed`. The trajectories run one at a time, an idle thread taking
/// the next, so the threads finish within a trajectory of each other; each task adds its
/// trajectories in their order, a trajectory that finishes early waiting for those before it, so
/// the sums do not depend on the threads.
///
/// With `Settings::average` an atom's κ, κ², r and r² along each lab axis are its means over the
/// steps within half a trap period of each output time (a window cut at t = 0 to stay centred, and
/// the trajectory run half a period past the last output): an atom's kinetic energy along a trap
/// axis sloshes into potential energy and back twice a period, and over a period its mean is half
/// its energy on that axis -- the spread of T between atoms then comes from their energies alone,
/// not from where each is in its oscillation. T stays m Var(v)/k_B, averaged over one trap period.
///
/// With `variates`, every trajectory's control variates at every output (stats.rs `Controls`).
pub fn simulate(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],variates:Option<&Variates>)->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,None,variates,None)
}
/// The same with the g-th group's trajectories run only through the output `ends[g]` (solve.rs # Batches): half a period past
/// it, as past a run's last, so that T's windows there are whole; its sums count no trajectories beyond.
pub fn simulate_to(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],variates:Option<&Variates>,ends:&[usize])->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,None,variates,Some(ends))
}
/// The same with every atom started at rest at r.
pub fn simulate_from(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],r:[f64;3])->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,Some(r),None,None)
}
fn run(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],origin:Option<[f64;3]>,variates:Option<&Variates>,ends:Option<&[usize]>)->Outcome{
    let engine=Engine::new(model,settings.clone());
    // The held twins' engine: the same settings and the twin's generator and jumps.
    let twins=variates.and_then(|v|v.twin).map(|twin|(Engine::held(model,settings,&twin.held),twin));
    // The tasks in order: (their group, their first trajectory, the one after their last).
    let bounds=|g:u64|(first+g*group,(first+(g+1)*group).min(first+count));
    let tasks:Vec<(u64,u64,u64)>=(0..count.div_ceil(group)).flat_map(|g|{let(start,end)=bounds(g);(start..end).step_by(TASK as usize).map(move|s|(g,s,(s+TASK).min(end)))}).collect();
    // How many outputs a group's trajectories run through.
    let through=|g:u64|ends.map_or(outputs,|e|e[g as usize]+1);
    let slots:Vec<Mutex<Slot>>=tasks.iter().map(|&(g,from,to)|Mutex::new(Slot{sums:Group::reaching(outputs,through(g),(to-from) as usize),samples:vec![vec![];listed.len()],next:from,
        waiting:(from..to).map(|_|None).collect(),jumps:0,doppler:0.,beats:0.,seconds:[0.;2]})).collect();
    (0..count as usize).into_par_iter().with_max_len(1).for_each_init(||(Windows::new(settings,steps,outputs),vec![None;outputs]),|(windows,held),n|{
        let index=first+n as u64;let k=tasks.partition_point(|t|t.1<=index)-1;let(g,from,to)=tasks[k];let(start,end)=bounds(g);
        let mut bare=initial_from(model,settings,initial,seed,Purpose::Initial,index,Stratum{first:start,size:end-start});
        if let Some(r)=origin{bare.r=r;bare.kappa=[0.;3];}
        let(r0,k0)=(bare.r,bare.kappa);
        let clock=Instant::now();
        let reach=through(g);let twin=twins.as_ref().map(|(e,twin)|hold(e,twin,&bare,steps,reach));
        let held_seconds=clock.elapsed().as_secs_f64();let clock=Instant::now();
        let mut t=dressed(model,bare);
        let(mut atoms,samples,_,_)=walk(model,&engine,&mut t,steps,reach,listed,windows,held,None);atoms.truncate(reach);
        let record=Record{atoms,start:(r0,k0),twin,samples,jumps:t.jumps,doppler:t.doppler,beats:t.beats,seconds:[clock.elapsed().as_secs_f64(),held_seconds]};
        // Into its task in order: it and every trajectory after it that already waits.
        let mut guard=slots[k].lock().unwrap_or_else(|e|e.into_inner());let slot=&mut *guard;slot.waiting[(index-from) as usize]=Some(record);
        while slot.next<to{
            let at=(slot.next-from) as usize;let Some(r)=slot.waiting[at].take() else{break};
            for(i,atom)in r.atoms.iter().enumerate(){
                let controls=Controls{harmonic:variates.map_or([0.;3],|v|v.harmonic.value(i,r.start.0,r.start.1)),twin:r.twin.as_ref().map(|p|&p[i])};
                slot.sums.add(i,atom.as_ref().map(atom_of).as_ref(),&controls);
            }
            for(k,x)in r.samples.into_iter().enumerate(){if let Some(x)=x{slot.samples[k].push(x);}}
            slot.jumps+=r.jumps;slot.doppler=slot.doppler.max(r.doppler);slot.beats=slot.beats.max(r.beats);slot.seconds[0]+=r.seconds[0];slot.seconds[1]+=r.seconds[1];slot.next+=1;
        }
    });
    // A group's tasks merged in order: the sums do not depend on the threads.
    let mut samples=vec![vec![];listed.len()];let(mut jumps,mut doppler,mut beats,mut seconds)=(0,0_f64,0_f64,[0.;2]);let mut out:Vec<Group>=vec![];let mut last=None;
    for(&(g,_,_),slot)in tasks.iter().zip(slots){
        let slot=slot.into_inner().unwrap_or_else(|e|e.into_inner());
        if last==Some(g){out.last_mut().unwrap().merge(&slot.sums);}else{out.push(slot.sums);last=Some(g);}
        for(k,v)in slot.samples.into_iter().enumerate(){samples[k].extend(v);}jumps+=slot.jumps;doppler=doppler.max(slot.doppler);beats=beats.max(slot.beats);seconds[0]+=slot.seconds[0];seconds[1]+=slot.seconds[1];
    }
    Outcome{groups:out,samples,jumps,doppler,beats,seconds}
}

/// A held atom at an output: its motion's window means, populations, folded excited population (η²),
/// profile error and (kv/σ)².
type Moment=([[f64;4];3],[f64;STATES],f64,f64,f64);
fn atom_of(m:&Moment)->Atom<'_>{Atom{motion:m.0,populations:&m.1,folded:m.2,profile:m.3,doppler:m.4}}

/// A trajectory's walk through the outputs (`run`, `sample_states`): `steps` steps between outputs and
/// half a period past the last (its windows' far halves); per output its atom if held, its (r, κ) at
/// the listed outputs, with `keep` its state at every keep-th output after t = 0 (staggered by its
/// index), and the jumps it made up to the last output.
fn walk(model:&Model,engine:&Engine,t:&mut Trajectory,steps:usize,outputs:usize,listed:&[usize],windows:&mut Windows,held:&mut [Option<([f64;STATES],f64,f64,f64)>],keep:Option<usize>)
    ->(Vec<Option<Moment>>,Vec<Option<([f64;3],[f64;3])>>,Vec<Trajectory>,u64){
    windows.clear();held.fill(None);let mut samples=vec![None;listed.len()];let mut kept=vec![];
    let mut record=|t:&Trajectory,i:usize,held:&mut [Option<([f64;STATES],f64,f64,f64)>]|{
        if !t.alive{return;}
        let point=model.point(t.r,t.kappa);
        let(populations,folded)=model.readout(t.cluster,&point,&t.psi,t.time);held[i]=Some((populations,folded,model.profile_error(t.r),model.doppler_ratio(t.kappa)));
        if let Some(k)=listed.iter().position(|&l|l==i){samples[k]=Some((t.r,t.kappa));}
    };
    record(t,0,held);windows.add(0,t);
    for i in 1..outputs{
        for k in 1..=steps{if t.alive{engine.step(t);if t.alive{windows.add((i-1)*steps+k,t);}}}
        record(t,i,held);
        if let Some(every)=keep{if t.alive&&(i+t.index as usize)%every==0{kept.push(t.clone());}}
    }
    let jumps=t.jumps;
    for k in 1..=windows.reach{if t.alive{engine.step(t);if t.alive{windows.add((outputs-1)*steps+k,t);}}}
    (held.iter().enumerate().map(|(i,p)|p.map(|(populations,folded,profile,doppler)|(windows.means(i),populations,folded,profile,doppler))).collect(),samples,kept,jumps)
}

/// A trajectory's part of its task's sums, kept until the trajectories before it in the task are added
/// (`run`): per output the held atom's motion means, populations, folded excited population (η²),
/// profile error and (kv/σ)², its
/// initial (r, κ) (the harmonic control's), its twin's populations, its (r, κ) at the listed outputs,
/// its jumps, largest (kv/σ)² and seconds (its own and its twin's).
struct Record{atoms:Vec<Option<Moment>>,start:([f64;3],[f64;3]),twin:Option<Vec<[f64;STATES]>>,samples:Vec<Option<([f64;3],[f64;3])>>,jumps:u64,doppler:f64,beats:f64,seconds:[f64;2]}
/// A task's sums so far: its trajectories added in order up to `next`, the ones after it that
/// finished early waiting.
struct Slot{sums:Group,samples:Vec<Vec<([f64;3],[f64;3])>>,next:u64,waiting:Vec<Option<Record>>,jumps:u64,doppler:f64,beats:f64,seconds:[f64;2]}

/// A trajectory's held twin (control.rs): the same bare initial internal state (at fourth order its
/// slow part at the twin's point, `Twin::start`), thresholds and jumps' draws (its streams are the
/// trajectory's), at the twin's point without motion; its populations there at every output.
fn hold(engine:&Engine,twin:&Twin,bare:&Trajectory,steps:usize,outputs:usize)->Vec<[f64;STATES]>{
    let mut t=bare.clone();t.r=twin.held.point.r;t.kappa=[0.;3];twin.start(&mut t);
    let read=|t:&Trajectory|twin.populations(t.cluster,&t.psi);
    let mut out=Vec::with_capacity(outputs);out.push(read(&t));
    for _ in 1..outputs{for _ in 0..steps{engine.step(&mut t);}out.push(read(&t));}
    out
}

/// The T windows' half widths per lab axis (steps): half a trap period with `Settings::average`, else
/// none.
pub(crate) fn half_windows(settings:&Settings)->[usize;3]{
    std::array::from_fn(|a|{let w=settings.lab[a];if settings.average&&w>0.{(std::f64::consts::PI/w/settings.dt).round() as usize}else{0}})
}

/// One trajectory's motion sums about every output time: per output and lab axis the steps counted
/// and Σκ, Σκ², Σr, Σr² over its window -- the steps within `half` of the output's (at most as many
/// as the output's own, so a window stays centred), one step without averaging.
///
/// Each step is added once, to the output interval it lies in (interval j holds steps
/// (j − 1)·steps + 1 ..= j·steps; interval 0 the step at t = 0): to the running total through that
/// interval, and, when it lies there, to the interval's first `half mod steps` steps or its last
/// `half mod steps + 1` -- a window's edges fall at those places of every interval. A window is the
/// whole intervals it covers, the difference of two running totals, and the two partial intervals
/// at its ends: O(1) per step and per output, O(outputs) memory.
struct Windows{half:[usize;3],steps:usize,reach:usize,total:Vec<[[f64;5];3]>,head:Vec<[[f64;5];3]>,tail:Vec<[[f64;5];3]>}
impl Windows{
    fn new(settings:&Settings,steps:usize,outputs:usize)->Windows{
        Windows::with(half_windows(settings),steps,outputs)
    }
    /// Windows of `half` steps per axis around outputs `steps` apart.
    fn with(half:[usize;3],steps:usize,outputs:usize)->Windows{
        let reach=half.into_iter().max().unwrap();let intervals=outputs+reach.div_ceil(steps);
        Windows{half,steps,reach,total:Vec::with_capacity(intervals),head:vec![[[0.;5];3];intervals],tail:vec![[[0.;5];3];intervals]}
    }
    fn clear(&mut self){self.total.clear();self.head.fill([[0.;5];3]);self.tail.fill([[0.;5];3]);}
    /// The trajectory at step k (from t = 0): into its interval's sums.
    fn add(&mut self,k:usize,t:&Trajectory){
        let(j,m)=if k==0{(0,self.steps)}else{let j=k.div_ceil(self.steps);(j,k-(j-1)*self.steps)};
        let last=self.total.last().copied().unwrap_or([[0.;5];3]);while self.total.len()<=j{self.total.push(last);}
        for a in 0..3{
            let(x,q)=(t.r[a],t.kappa[a]);let v=[1.,q,q*q,x,x*x];let r=self.half[a]%self.steps;
            for(s,y)in self.total[j][a].iter_mut().zip(v){*s+=y;}
            if m<=r{for(s,y)in self.head[j][a].iter_mut().zip(v){*s+=y;}}
            if m+r>=self.steps{for(s,y)in self.tail[j][a].iter_mut().zip(v){*s+=y;}}
        }
    }
    /// Output i's means: κ, κ², r, r² per axis.
    fn means(&self,i:usize)->[[f64;4];3]{
        let at=i*self.steps;let total=|j:usize,a:usize|self.total[j.min(self.total.len()-1)][a];
        std::array::from_fn(|a|{
            let h=self.half[a];
            // Within half a window of t = 0 the window is cut there: the whole intervals 0 ..= 2i.
            let s:[f64;5]=if at<h{total(2*i,a)}else{
                let(lo,hi)=(i-h/self.steps,i+h/self.steps);let(whole,first)=((total(hi,a),total(lo,a)),self.tail[lo][a]);
                let end=self.head.get(hi+1).map_or([0.;5],|x|x[a]);
                std::array::from_fn(|v|whole.0[v]-whole.1[v]+first[v]+end[v])
            };
            [s[1]/s[0],s[2]/s[0],s[3]/s[0],s[4]/s[0]]
        })
    }
}

/// A pilot: `count` trajectories over the outputs, the held atoms' states at every output kept (at
/// most a few thousand, evenly thinned), the jumps they made, and each trajectory's sums (a group of
/// one, with its harmonic control): the statistics' first prediction (solve.rs, # Batches).
pub fn sample_states(model:&Model,settings:&Settings,initial:&Initial,seed:u64,count:u64,steps:usize,outputs:usize,harmonic:&Harmonic)->(Vec<Trajectory>,u64,Vec<Group>){
    let engine=Engine::new(model,settings.clone());
    let every=((count as usize*outputs)/4096).max(1);
    let runs:Vec<(Vec<Trajectory>,u64,Group)>=(0..count).into_par_iter().map_init(||(Windows::new(settings,steps,outputs),vec![None;outputs]),|(windows,held),index|{
        let mut t=pilot_start(model,settings,initial,seed,index);let(r0,k0)=(t.r,t.kappa);
        let(atoms,_,kept,jumps)=walk(model,&engine,&mut t,steps,outputs,&[],windows,held,Some(every));
        let mut sums=Group::new(outputs,1);
        for(i,atom)in atoms.iter().enumerate(){sums.add(i,atom.as_ref().map(atom_of).as_ref(),&Controls{harmonic:harmonic.value(i,r0,k0),twin:None});}
        (kept,jumps,sums)
    }).collect();
    let jumps=runs.iter().map(|r|r.1).sum();let mut states=vec![];let mut groups=vec![];
    for(kept,_,sums)in runs{states.extend(kept);groups.push(sums);}
    (states,jumps,groups)
}

#[cfg(test)]
mod tests{
    use super::*;
    /// The windows' means against the sums over each window's steps, with the atom lost before the
    /// end: windows cut at t = 0, half a window shorter than an interval, a whole number of them, none.
    #[test]
    fn the_windows_sum_each_window_s_steps(){
        let mut rng=Rng::new(9,Purpose::Initial,0,0);
        for(steps,half,outputs)in [(1,[0,3,419],50),(5,[7,10,0],20),(44,[5,110,113],6),(3,[2,2,2],10),(4,[8,1,4],9)]{
            let mut w=Windows::with(half,steps,outputs);let last=(outputs-1)*steps+w.reach;let lost=last-last/5;
            let values:Vec<([f64;3],[f64;3])>=(0..=last).map(|_|(std::array::from_fn(|_|rng.uniform()-0.5),std::array::from_fn(|_|rng.uniform()-0.5))).collect();
            let mut t=Trajectory{cluster:0,psi:[Complex::ZERO;model::MOST_MIXED],r:[0.;3],kappa:[0.;3],time:0.,threshold:1.,alive:true,jumps:0,doppler:0.,beats:0.,seed:0,index:0};
            for k in 0..=lost{(t.r,t.kappa)=values[k];w.add(k,&t);}
            for i in 0..outputs{
                let at=i*steps;if at>lost{continue;}let m=w.means(i);
                for a in 0..3{
                    let h=half[a].min(at);let(mut n,mut s)=(0.,[0.;4]);
                    for k in at-h..=(at+h).min(lost){let(r,q)=values[k];n+=1.;s[0]+=q[a];s[1]+=q[a]*q[a];s[2]+=r[a];s[3]+=r[a]*r[a];}
                    for v in 0..4{assert!((m[a][v]-s[v]/n).abs()<=1e-12,"steps {steps} half {half:?} output {i} axis {a}: {} against {}",m[a][v],s[v]/n);}
                }
            }
        }
    }
}
