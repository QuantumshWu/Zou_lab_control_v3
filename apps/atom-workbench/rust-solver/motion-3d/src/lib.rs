//! Semiclassical 3D motion of a single Rb87 atom in an optical tweezer: a classical point (r, p)
//! in 3D carrying the internal state of the eliminated ground manifold (`ground`), evolved by
//! quantum jumps. The request (`request`) is the scene's model with its motion block; `ground` is
//! its front end, the generator with the excited states eliminated, on the small dense linear
//! algebra of `linear`.
pub mod control;
pub mod check;
pub(crate) mod cost;
pub mod eigen;
pub mod engine;
pub mod ensemble;
pub mod gpu;
pub mod ground;
pub mod ledger;
pub mod linear;
pub mod model;
pub mod request;
pub(crate) mod read;
pub mod rng;
pub mod solve;
pub mod stats;

pub use request::{Beam,Complex,Matrix,OutsideForm,Request,STATES};
use engine::{Engine,Settings,Trajectory};
use ensemble::Ensemble;
use model::{Model,Trap};
use rayon::prelude::*;
use rng::{Purpose,Rng};
use control::Harmonic;
use stats::{Atom,Group,KT_PER_UK};
use std::sync::Mutex;

/// The initial internal states (cluster, position in it, probability), each one's trap (the harmonic
/// trap at its centre, `Model::trap`), depth (`Model::depth`: the loss's region U_h ≤ D) and bound
/// thermal ensemble (ensemble.rs; none for a state the trap does not hold, which starts there at
/// rest), the temperature asked for -- the ensembles' parameter T --, and the share of the traps'
/// thermal distributions over their regions that the ensembles leave out (`Ensemble::cut`, weighted by
/// the populations).
#[derive(Clone,Debug)]
pub struct Initial{pub states:Vec<(usize,usize,f64)>,pub traps:Vec<Trap>,pub depths:Vec<f64>,pub ensembles:Vec<Option<Ensemble>>,pub temperature_uk:f64,pub cut:f64}
impl Initial{
    /// From the request's initial density: its ground populations (the density is diagonal in the
    /// model basis, as the app writes it).
    pub fn new(model:&Model,request:&Request)->Result<Initial,String>{let mut i=Initial::deferred(model,request)?;i.integrate(model);Ok(i)}
    /// `new` with its ensembles' kinetic temperatures and cut still to come (`integrate`): what a fold's candidate
    /// needs until it is the pilot's or the chosen, the quadrature touching none of the fold's choices.
    pub fn deferred(model:&Model,request:&Request)->Result<Initial,String>{
        let g=&model.ground;let rho=&request.initial_density;let mut states=vec![];
        for i in 0..STATES{for j in 0..STATES{if i!=j&&rho[i*STATES+j].abs()>0.{return Err("The initial density must be diagonal in the model basis".into());}}}
        for(k,cluster)in g.clusters.iter().enumerate(){for(x,&b)in cluster.states.iter().enumerate(){
            let p=rho[g.ground_states[b]*STATES+g.ground_states[b]].re;if p>0.{states.push((k,x,p));}
        }}
        let total:f64=states.iter().map(|s|s.2).sum();
        if !(total>0.){return Err("The initial density has no ground population".into());}
        for s in &mut states{s.2/=total;}
        let traps:Vec<Trap>=states.iter().map(|&(k,x,_)|model.trap(k,x)).collect();
        let depths:Vec<f64>=states.iter().zip(&traps).map(|(&(k,x,_),trap)|model.depth(k,x,trap.centre)).collect();
        let kt=KT_PER_UK*request.temperature_uk;
        let ensembles=states.iter().zip(&traps).map(|(&(k,x,_),trap)|Ensemble::new(model,k,x,kt,trap.centre)).collect::<Result<Vec<_>,_>>()?;
        Ok(Initial{states,traps,depths,ensembles,temperature_uk:request.temperature_uk,cut:0.})
    }
    /// The reference state, the largest initial population: its place in `states`.
    pub fn reference(&self)->usize{(0..self.states.len()).max_by(|&a,&b|self.states[a].2.total_cmp(&self.states[b].2)).unwrap()}
    /// The ensembles' kinetic temperatures and share left out, by quadrature (ensemble.rs).
    pub fn integrate(&mut self,model:&Model){
        for((e,trap),&depth)in self.ensembles.iter_mut().zip(&self.traps).zip(&self.depths){if let Some(e)=e{e.integrate(model,trap,depth);}}
        self.cut=self.states.iter().zip(&self.ensembles).map(|(s,e)|s.2*e.as_ref().map_or(0.,|e|e.cut)).sum();
    }
    /// The initial kinetic temperature per lab axis (uK), exactly: per state its ensemble's relative to
    /// T (`Ensemble::kinetic`) along each principal axis of its trap with the harmonic oscillator's
    /// thermal Wigner factor (ħω/2k_BT) coth(ħω/2k_BT) (`initial_trajectory`), weighted by its population
    /// and the axis' share of the lab axis; a state the trap does not hold at rest.
    pub fn temperature(&self)->[f64;3]{
        let kt=KT_PER_UK*self.temperature_uk;let mut t=[0.;3];
        for((&(_,_,p),trap),e)in self.states.iter().zip(&self.traps).zip(&self.ensembles){
            let Some(e)=e else{continue};
            for a in 0..3{for i in 0..3{let half=trap.frequencies[i]/(2.*kt);
                t[a]+=p*e.kinetic*trap.axes[i][a]*trap.axes[i][a]*if half>0.{self.temperature_uk*half/half.tanh()}else{self.temperature_uk};}}
        }
        t
    }
    /// The ensembles' kinetic temperature relative to T (`Ensemble::kinetic`, classical: without the
    /// Wigner factor), weighted by the populations.
    pub fn kinetic(&self)->f64{self.states.iter().zip(&self.ensembles).map(|(s,e)|s.2*e.as_ref().map_or(0.,|e|e.kinetic)).sum()}
}

/// A trajectory's group: its first trajectory and its size. A group's initial states form a Latin
/// hypercube: every variable drawn from a uniform deviate -- the internal state's pick, the position's
/// first proposal (its radial deviate and direction), the kinetic energy's quantile at the position
/// and the momentum's direction (ensemble.rs) -- falls for each member in another of `size`
/// equal-probability strata (the members' order a random permutation of the strata, new for every
/// variable) at a uniform point inside it. A member whose first proposal is rejected draws its
/// position from its own stream; its kinetic energy keeps its stratum, the quantile of its distribution
/// at whatever position it takes. Each member is distributed as an independent draw would be; the
/// group spreads over every variable's distribution, so its means of the initial energies, and of what
/// they leave at early times, vary much less than independent draws'. Groups are independent of each
/// other, so the jackknife over them still measures the error. A group of one is an independent
/// draw.
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
/// the initial populations; its position and momentum from that state's bound thermal ensemble
/// (ensemble.rs: the lowered isothermal at T over its trap's bound states), along each principal axis of its trap scaled to the harmonic oscillator's
/// thermal Wigner function, T* = (ħω/2k)coth(ħω/2kT): the momentum by √(T*/T), and the position's displacement from
/// the bottom of the atom's own well likewise, that displacement x = K⁻¹∇V at the atom (V the ensemble's potential,
/// K the trap's stiffness at its centre): r itself in a harmonic trap; about its own site in a standing wave, whose
/// sites are as stiff, ∝ sin(2kz) -- continuous, no site to pick, zero at the nodes where no oscillator holds it.
/// (Scaled about the focus a neighbouring site's atom left its site's bottom by (√(T*/T) − 1)λ/2: 975 uK of potential
/// energy at the retro tweezer's site 1, against 14 uK in the centre.) A scaled draw the trap would not bind
/// (the factors lift its energy by their excess over one) is drawn again from its own stream. Along
/// frozen lab axes (a validation's), nothing; a state the trap does not hold at rest at the trap's centre. At
/// fourth order the internal state is the bare state's slow part (engine.rs `slow_start`).
pub fn initial_trajectory(model:&Model,settings:&Settings,initial:&Initial,seed:u64,index:u64,stratum:Stratum)->Trajectory{
    dressed(model,initial_from(model,settings,initial,seed,Purpose::Initial,index,stratum))
}
/// A bare initial state's slow part at fourth order, where it starts (engine.rs `slow_start`).
pub(crate) fn dressed(model:&Model,mut t:Trajectory)->Trajectory{let point=model.point(t.r,t.kappa);engine::slow_start(model,&mut t,&point,0.);t}
/// A pilot trajectory's initial state (`sample_states`; solve.rs, the fold's charge at t = 0).
pub(crate) fn pilot_start(model:&Model,settings:&Settings,initial:&Initial,seed:u64,index:u64)->Trajectory{
    dressed(model,initial_from(model,settings,initial,seed,Purpose::Pilot,index,Stratum::single(index)))
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
    let kt=KT_PER_UK*initial.temperature_uk;
    let quantum:[f64;3]=std::array::from_fn(|i|{let f=frequencies[i];if f>0.{let half=f/(2.*kt);(half/half.tanh()).sqrt()}else{1.}});
    let scaled=|v:[f64;3]|->[f64;3]{std::array::from_fn(|a|(0..3).map(|i|quantum[i]*(0..3).map(|b|v[b]*axes[i][b]).sum::<f64>()*axes[i][a]).sum())};
    // The position's displacement from its own well's bottom scaled likewise, the slope by central differences as `Model::trap`'s.
    let displaced=|r:[f64;3]|->[f64;3]{
        let h=1e-3*model.geometry[model.trap_beam].waist.unwrap_or(1.);let mut out=r;
        for i in 0..3{let f=frequencies[i];if !(f>0.){continue;}
            let at=|s:f64|->[f64;3]{std::array::from_fn(|a|r[a]+s*axes[i][a])};
            let slope=(model.trap_level(cluster,state,at(h))-model.trap_level(cluster,state,at(-h)))/(2.*h);
            let shift=(quantum[i]-1.)*slope*model.hbar_over_mass/(f*f);for a in 0..3{out[a]+=shift*axes[i][a];}}
        out};
    let mut psi=[Complex::ZERO;model::MOST_MIXED];psi[state]=Complex::ONE;
    let(r,kappa)=match &initial.ensembles[pick]{
        // The member's Latin deviates are taken whatever its state, so that its group's stay in step.
        None=>{for _ in 0..Ensemble::deviates(settings.free){latin.next(&mut rng);}(trap.centre,[0.;3])}
        Some(e)=>{
            let mut draw=e.draw(model,settings.free,&mut rng,&mut |rng|latin.next(rng));
            loop{
                let(r,kappa)=(displaced(draw.0),scaled(draw.1));
                let kinetic=0.5*model.hbar_over_mass*kappa.iter().map(|k|k*k).sum::<f64>();
                if kinetic+model.trap_level(cluster,state,r)<0.{break(r,kappa);}
                draw=e.draw(model,settings.free,&mut rng,&mut |rng|rng.interior());
            }
        }
    };
    let threshold=engine::jump_stream(seed,index,0).open_uniform();
    Trajectory{cluster,psi,r,kappa,time:0.,threshold,alive:true,jumps:0,beats:0.,seed,index}
}

/// The outputs of a run: one group per `group` trajectories, sums at every output time, the held
/// atoms' (r, κ) at the listed outputs, the jumps made, and the
/// largest bound per unit time of the beats the exact steps left out (`Trajectory::beats`); with the loss's check's sampling, per
/// trajectory in their order what it leaves the check (check.rs `Trace`).
pub struct Outcome{pub groups:Vec<Group>,pub samples:Vec<Vec<([f64;3],[f64;3])>>,pub jumps:u64,pub beats:f64,pub traces:Vec<check::Trace>}
impl Outcome{
    /// A later batch's after it (solve.rs # Batches).
    pub(crate) fn extend(&mut self,more:Outcome){
        self.groups.extend(more.groups);for(a,b)in self.samples.iter_mut().zip(more.samples){a.extend(b);}self.jumps+=more.jumps;self.beats=self.beats.max(more.beats);self.traces.extend(more.traces);
    }
}

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
/// With `Settings::average` an atom's κ and κ² along each lab axis are its means over the
/// steps within half a trap period of each output time (a window cut at t = 0 to stay centred, and
/// the trajectory run half a period past the last output): an atom's kinetic energy along a trap
/// axis sloshes into potential energy and back twice a period, and over a period its mean is half
/// its energy on that axis -- the spread of T between atoms then comes from their energies alone,
/// not from where each is in its oscillation. T stays m Var(v)/k_B, averaged over one trap period.
///
/// With `harmonic`, every trajectory's harmonic control at every output (control.rs).
pub fn simulate(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],harmonic:Option<&Harmonic>)->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,None,harmonic,None,None)
}
/// The same with the g-th group's trajectories run only through the output `ends[g]` (solve.rs # Batches): half a period past
/// it, as past a run's last, so that T's windows there are whole; its sums count no trajectories beyond. With `check`, the
/// loss's check's reads in its intervals and each trajectory's trace (check.rs).
pub fn simulate_to(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],harmonic:Option<&Harmonic>,ends:&[usize],check:Option<&check::Sampling>)->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,None,harmonic,Some(ends),check)
}
/// The same with every atom started at rest at r.
pub fn simulate_from(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],r:[f64;3])->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,Some(r),None,None,None)
}
fn run(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],origin:Option<[f64;3]>,harmonic:Option<&Harmonic>,ends:Option<&[usize]>,check:Option<&check::Sampling>)->Outcome{
    let engine=Engine::new(model,settings.clone());
    // The tasks in order: (their group, their first trajectory, the one after their last).
    let bounds=|g:u64|(first+g*group,(first+(g+1)*group).min(first+count));
    let tasks:Vec<(u64,u64,u64)>=(0..count.div_ceil(group)).flat_map(|g|{let(start,end)=bounds(g);(start..end).step_by(TASK as usize).map(move|s|(g,s,(s+TASK).min(end)))}).collect();
    // How many outputs a group's trajectories run through.
    let through=|g:u64|ends.map_or(outputs,|e|e[g as usize]+1);
    let slots:Vec<Mutex<Slot>>=tasks.iter().map(|&(g,from,to)|Mutex::new(Slot{sums:Group::reaching(outputs,through(g),(to-from) as usize),samples:vec![vec![];listed.len()],next:from,
        waiting:(from..to).map(|_|None).collect(),jumps:0,beats:0.,traces:vec![]})).collect();
    (0..count as usize).into_par_iter().with_max_len(1).for_each_init(||(Windows::new(settings,steps,outputs),vec![None;outputs]),|(windows,held),n|{
        let index=first+n as u64;let k=tasks.partition_point(|t|t.1<=index)-1;let(g,from,to)=tasks[k];let(start,end)=bounds(g);
        let mut bare=initial_from(model,settings,initial,seed,Purpose::Initial,index,Stratum{first:start,size:end-start});
        if let Some(r)=origin{bare.r=r;bare.kappa=[0.;3];}
        let(r0,k0)=(settings.about_site(bare.r),bare.kappa);
        let reach=through(g);let mut t=dressed(model,bare);
        let(mut atoms,samples,_,_,trace)=walk(model,&engine,&mut t,steps,reach,listed,windows,held,None,check);atoms.truncate(reach);
        let record=Record{atoms,start:(r0,k0),samples,jumps:t.jumps,beats:t.beats,trace};
        // Into its task in order: it and every trajectory after it that already waits.
        let mut guard=slots[k].lock().unwrap_or_else(|e|e.into_inner());let slot=&mut *guard;slot.waiting[(index-from) as usize]=Some(record);
        while slot.next<to{
            let at=(slot.next-from) as usize;let Some(r)=slot.waiting[at].take() else{break};
            for(i,atom)in r.atoms.iter().enumerate(){slot.sums.add(i,atom.as_ref().map(atom_of).as_ref(),harmonic.map_or([0.;3],|h|h.value(i,r.start.0,r.start.1)));}
            for(k,x)in r.samples.into_iter().enumerate(){if let Some(x)=x{slot.samples[k].push(x);}}
            slot.jumps+=r.jumps;slot.beats=slot.beats.max(r.beats);if let Some(x)=r.trace{slot.traces.push(x);}slot.next+=1;
        }
    });
    // A group's tasks merged in order: the sums do not depend on the threads.
    let mut samples=vec![vec![];listed.len()];let(mut jumps,mut beats)=(0,0_f64);let mut out:Vec<Group>=vec![];let mut last=None;let mut traces=vec![];
    for(&(g,_,_),slot)in tasks.iter().zip(slots){
        let slot=slot.into_inner().unwrap_or_else(|e|e.into_inner());
        if last==Some(g){out.last_mut().unwrap().merge(&slot.sums);}else{out.push(slot.sums);last=Some(g);}
        for(k,v)in slot.samples.into_iter().enumerate(){samples[k].extend(v);}jumps+=slot.jumps;beats=beats.max(slot.beats);traces.extend(slot.traces);
    }
    Outcome{groups:out,samples,jumps,beats,traces}
}

/// A held atom at an output: its motion's window means, populations, folded excited population (η²),
/// profile error and (kv/σ)².
type Moment=([[f64;2];3],[f64;STATES],f64,f64,f64);
fn atom_of(m:&Moment)->Atom<'_>{Atom{motion:m.0,populations:&m.1,folded:m.2,profile:m.3,doppler:m.4}}

/// A trajectory's walk through the outputs (`run`, `sample_states`): `steps` steps between outputs and
/// half a period past the last (its windows' far halves); per output its atom if held, its (r, κ) at
/// the listed outputs, with `keep` its state at every keep-th output after t = 0 (staggered by its
/// index), and the jumps it made up to the last output; with `check`, what it leaves the loss's check (check.rs `Trace`): its
/// energy at every output, and in every interval at the step drawn there its read where drawn.
fn walk(model:&Model,engine:&Engine,t:&mut Trajectory,steps:usize,outputs:usize,listed:&[usize],windows:&mut Windows,held:&mut [Option<([f64;STATES],f64,f64,f64)>],keep:Option<usize>,check:Option<&check::Sampling>)
    ->(Vec<Option<Moment>>,Vec<Option<([f64;3],[f64;3])>>,Vec<Trajectory>,u64,Option<check::Trace>){
    windows.clear();held.fill(None);let mut samples=vec![None;listed.len()];let mut kept=vec![];
    let mut trace=check.map(|_|check::Trace::new(outputs));
    // An output: the held atom's record, and with the check its energy at the same point.
    let mut record=|t:&Trajectory,i:usize,held:&mut [Option<([f64;STATES],f64,f64,f64)>],trace:&mut Option<check::Trace>|{
        let point=t.alive.then(||model.point(t.r,t.kappa));
        if let Some(x)=trace{x.output(point.as_ref().map(|p|engine.loss_energy_at(t,p)));}
        let Some(point)=point else{return};
        let(populations,folded)=model.readout(t.cluster,&point,&t.psi,t.time);held[i]=Some((populations,folded,model.profile_error(t.r),model.doppler_ratio(t.kappa)));
        if let Some(k)=listed.iter().position(|&l|l==i){samples[k]=Some((t.r,t.kappa));}
    };
    record(t,0,held,&mut trace);windows.add(0,t);
    for i in 1..outputs{
        let drawn=check.map(|_|check::drawn(t.seed,t.index,i,steps));
        for k in 1..=steps{
            if let(Some(sampling),Some((at,deviate)),Some(x))=(check,drawn,trace.as_mut()){if k-1==at&&t.alive{x.look(engine,sampling,Some(t),i,deviate);}}
            if t.alive{engine.step(t);if t.alive{windows.add((i-1)*steps+k,t);}}
        }
        record(t,i,held,&mut trace);
        if let Some(every)=keep{if t.alive&&(i+t.index as usize)%every==0{kept.push(t.clone());}}
    }
    let jumps=t.jumps;
    for k in 1..=windows.reach{if t.alive{engine.step(t);if t.alive{windows.add((outputs-1)*steps+k,t);}}}
    (held.iter().enumerate().map(|(i,p)|p.map(|(populations,folded,profile,doppler)|(windows.means(i),populations,folded,profile,doppler))).collect(),samples,kept,jumps,trace)
}

/// A trajectory's part of its task's sums, kept until the trajectories before it in the task are added
/// (`run`): per output the held atom's motion means, populations, folded excited population (η²),
/// profile error and (kv/σ)², its
/// initial (r, κ) (the harmonic control's), its (r, κ) at the listed outputs and
/// its jumps and its trace for the loss's check.
struct Record{atoms:Vec<Option<Moment>>,start:([f64;3],[f64;3]),samples:Vec<Option<([f64;3],[f64;3])>>,jumps:u64,beats:f64,trace:Option<check::Trace>}
/// A task's sums so far: its trajectories added in order up to `next`, the ones after it that
/// finished early waiting.
struct Slot{sums:Group,samples:Vec<Vec<([f64;3],[f64;3])>>,next:u64,waiting:Vec<Option<Record>>,jumps:u64,beats:f64,traces:Vec<check::Trace>}

/// The T windows' half widths per lab axis (steps): half a trap period with `Settings::average`, else
/// none.
pub(crate) fn half_windows(settings:&Settings)->[usize;3]{
    std::array::from_fn(|a|{let w=settings.lab[a];if settings.average&&w>0.{(std::f64::consts::PI/w/settings.dt).round() as usize}else{0}})
}

/// One trajectory's motion sums about every output time: per output and lab axis the steps counted
/// and Σκ, Σκ² over its window -- the steps within `half` of the output's (at most as many
/// as the output's own, so a window stays centred), one step without averaging.
///
/// Each step is added once, to the output interval it lies in (interval j holds steps
/// (j − 1)·steps + 1 ..= j·steps; interval 0 the step at t = 0): to the running total through that
/// interval, and, when it lies there, to the interval's first `half mod steps` steps or its last
/// `half mod steps + 1` -- a window's edges fall at those places of every interval. A window is the
/// whole intervals it covers, the difference of two running totals, and the two partial intervals
/// at its ends: O(1) per step and per output, O(outputs) memory.
struct Windows{half:[usize;3],steps:usize,reach:usize,total:Vec<[[f64;3];3]>,head:Vec<[[f64;3];3]>,tail:Vec<[[f64;3];3]>}
impl Windows{
    fn new(settings:&Settings,steps:usize,outputs:usize)->Windows{
        Windows::with(half_windows(settings),steps,outputs)
    }
    /// Windows of `half` steps per axis around outputs `steps` apart.
    fn with(half:[usize;3],steps:usize,outputs:usize)->Windows{
        let reach=half.into_iter().max().unwrap();let intervals=outputs+reach.div_ceil(steps);
        Windows{half,steps,reach,total:Vec::with_capacity(intervals),head:vec![[[0.;3];3];intervals],tail:vec![[[0.;3];3];intervals]}
    }
    fn clear(&mut self){self.total.clear();self.head.fill([[0.;3];3]);self.tail.fill([[0.;3];3]);}
    /// The trajectory at step k (from t = 0): into its interval's sums.
    fn add(&mut self,k:usize,t:&Trajectory){
        let(j,m)=if k==0{(0,self.steps)}else{let j=k.div_ceil(self.steps);(j,k-(j-1)*self.steps)};
        let last=self.total.last().copied().unwrap_or([[0.;3];3]);while self.total.len()<=j{self.total.push(last);}
        for a in 0..3{
            let q=t.kappa[a];let v=[1.,q,q*q];let r=self.half[a]%self.steps;
            for(s,y)in self.total[j][a].iter_mut().zip(v){*s+=y;}
            if m<=r{for(s,y)in self.head[j][a].iter_mut().zip(v){*s+=y;}}
            if m+r>=self.steps{for(s,y)in self.tail[j][a].iter_mut().zip(v){*s+=y;}}
        }
    }
    /// Output i's means: κ, κ² per axis.
    fn means(&self,i:usize)->[[f64;2];3]{
        let at=i*self.steps;let total=|j:usize,a:usize|self.total[j.min(self.total.len()-1)][a];
        std::array::from_fn(|a|{
            let h=self.half[a];
            // Within half a window of t = 0 the window is cut there: the whole intervals 0 ..= 2i.
            let s:[f64;3]=if at<h{total(2*i,a)}else{
                let(lo,hi)=(i-h/self.steps,i+h/self.steps);let(whole,first)=((total(hi,a),total(lo,a)),self.tail[lo][a]);
                let end=self.head.get(hi+1).map_or([0.;3],|x|x[a]);
                std::array::from_fn(|v|whole.0[v]-whole.1[v]+first[v]+end[v])
            };
            [s[1]/s[0],s[2]/s[0]]
        })
    }
}

/// A pilot: `count` trajectories over the outputs, the held atoms' states at every output kept (at
/// most a few thousand, evenly thinned), the jumps they made, and each trajectory's sums (a group of
/// one, with its harmonic control): the statistics' first prediction (solve.rs, # Batches).
pub fn sample_states(model:&Model,settings:&Settings,initial:&Initial,seed:u64,count:u64,steps:usize,outputs:usize,harmonic:&Harmonic,check:Option<&check::Sampling>)->(Vec<Trajectory>,u64,Vec<Group>,Vec<check::Trace>){
    let engine=Engine::new(model,settings.clone());
    let every=((count as usize*outputs)/4096).max(1);
    let runs:Vec<(Vec<Trajectory>,u64,Group,Option<check::Trace>)>=(0..count).into_par_iter().map_init(||(Windows::new(settings,steps,outputs),vec![None;outputs]),|(windows,held),index|{
        let mut t=pilot_start(model,settings,initial,seed,index);let(r0,k0)=(settings.about_site(t.r),t.kappa);
        let(atoms,_,kept,jumps,trace)=walk(model,&engine,&mut t,steps,outputs,&[],windows,held,Some(every),check);
        let mut sums=Group::new(outputs,1);
        for(i,atom)in atoms.iter().enumerate(){sums.add(i,atom.as_ref().map(atom_of).as_ref(),harmonic.value(i,r0,k0));}
        (kept,jumps,sums,trace)
    }).collect();
    let jumps=runs.iter().map(|r|r.1).sum();let(mut states,mut groups,mut traces)=(vec![],vec![],vec![]);
    for(kept,_,sums,trace)in runs{states.extend(kept);groups.push(sums);traces.extend(trace);}
    (states,jumps,groups,traces)
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
            let values:Vec<[f64;3]>=(0..=last).map(|_|std::array::from_fn(|_|rng.uniform()-0.5)).collect();
            let mut t=Trajectory{cluster:0,psi:[Complex::ZERO;model::MOST_MIXED],r:[0.;3],kappa:[0.;3],time:0.,threshold:1.,alive:true,jumps:0,beats:0.,seed:0,index:0};
            for k in 0..=lost{t.kappa=values[k];w.add(k,&t);}
            for i in 0..outputs{
                let at=i*steps;if at>lost{continue;}let m=w.means(i);
                for a in 0..3{
                    let h=half[a].min(at);let(mut n,mut s)=(0.,[0.;2]);
                    for k in at-h..=(at+h).min(lost){let q=values[k];n+=1.;s[0]+=q[a];s[1]+=q[a]*q[a];}
                    for v in 0..2{assert!((m[a][v]-s[v]/n).abs()<=1e-12,"steps {steps} half {half:?} output {i} axis {a}: {} against {}",m[a][v],s[v]/n);}
                }
            }
        }
    }
}
