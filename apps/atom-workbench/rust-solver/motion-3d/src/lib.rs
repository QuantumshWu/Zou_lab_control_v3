//! Semiclassical 3D motion of a single Rb87 atom in an optical tweezer: a classical point (r, p)
//! in 3D carrying the internal state of the eliminated ground manifold (`ground`), evolved by
//! quantum jumps. The request (`request`) is the scene's model with its motion block; `ground` is
//! its front end, the generator with the excited states eliminated, on the small dense linear
//! algebra of `linear`.
pub mod engine;
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
use stats::{Group,KT_PER_UK};

/// The initial internal states (cluster, position in it, probability), each one's trap (the
/// harmonic trap its initial positions are drawn in) and the temperature.
#[derive(Clone,Debug)]
pub struct Initial{pub states:Vec<(usize,usize,f64)>,pub traps:Vec<Trap>,pub temperature_uk:f64}
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
        let traps=states.iter().map(|&(k,x,_)|model.trap(k,x)).collect();
        Ok(Initial{states,traps,temperature_uk:request.temperature_uk})
    }
}

/// The independence Metropolis chain's length for the initial positions: its proposal, the
/// harmonic trap's Boltzmann distribution, is the target to a few percent, so the chain forgets
/// its start within a few accepted moves.
const METROPOLIS_STEPS:usize=32;

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
/// potential (independence Metropolis from its harmonic approximation, on the trap's principal
/// axes); the momentum Maxwell; position and momentum along each principal axis scaled to the
/// harmonic oscillator's thermal Wigner function, T* = (ħω/2k)coth(ħω/2kT). Along frozen lab axes (a
/// validation's), nothing.
pub fn initial_trajectory(model:&Model,settings:&Settings,initial:&Initial,seed:u64,index:u64,stratum:Stratum)->Trajectory{
    initial_from(model,settings,initial,seed,Purpose::Initial,index,stratum)
}
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
    let draw=|rng:&mut Rng,latin:&mut Latin|->[f64;3]{std::array::from_fn(|i|sigma[i]*normal_quantile(latin.next(rng)))};
    let weight=|q:[f64;3]|->f64{-model.potential(cluster,state,lab(q))/kt+(0..3).filter(|&i|sigma[i]>0.).map(|i|q[i]*q[i]/(2.*sigma[i]*sigma[i])).sum::<f64>()};
    let mut q=draw(&mut rng,&mut latin);let mut w=weight(q);
    for _ in 0..METROPOLIS_STEPS{let next=draw(&mut rng,&mut latin);let wn=weight(next);if rng.open_uniform().ln()<wn-w{q=next;w=wn;}}
    let quantum:[f64;3]=std::array::from_fn(|i|{let f=frequencies[i];if f>0.{let half=f/(2.*kt);(half/half.tanh()).sqrt()}else{1.}});
    for i in 0..3{q[i]*=quantum[i];}
    let p:[f64;3]=std::array::from_fn(|i|{let z=normal_quantile(latin.next(&mut rng));if sigma[i]>0.{(kt/hm).sqrt()*quantum[i]*z}else{0.}});
    let mut psi=[Complex::ZERO;model::MOST_STATES];psi[state]=Complex::ONE;
    let threshold=engine::jump_stream(seed,index,0).open_uniform();
    Trajectory{cluster,psi,r:lab(q),kappa:lab(p),time:0.,threshold,alive:true,jumps:0,correlation:[[Complex::ZERO;engine::SQUARE];3],pending:[[0.;3];3],steps:0,
        doppler:0.,seed,index}
}

/// The outputs of a run: one group per `group` trajectories, sums at every output time, the held
/// atoms' (r, κ) at the listed outputs, the jumps made and the largest first-order (kv/σ_min)².
pub struct Outcome{pub groups:Vec<Group>,pub samples:Vec<Vec<([f64;3],[f64;3])>>,pub jumps:u64,pub doppler:f64}

/// The most trajectories a task runs one after another: a batch's work is cut this fine so the
/// threads finish together, while its sums (one per output time) stay few.
const TASK:u64=8;

/// Trajectories [first, first + count) of the run `seed`, `steps` steps of the settings' dt between
/// `outputs` output times (the first at t = 0), in groups of `group` (each a Latin hypercube of
/// initial states, `Stratum`, and a jackknife group), run as tasks of at most `TASK` trajectories
/// inside a group; (r, κ) kept at `listed`.
///
/// With `Settings::average` an atom's κ, κ², r and r² along each lab axis are its means over the
/// steps within half a trap period of each output time (a window cut at t = 0 to stay centred, and
/// the trajectory run half a period past the last output): an atom's kinetic energy along a trap
/// axis sloshes into potential energy and back twice a period, and over a period its mean is half
/// its energy on that axis -- the spread of T between atoms then comes from their energies alone,
/// not from where each is in its oscillation. T stays m Var(v)/k_B, averaged over one trap period.
pub fn simulate(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize])->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,None)
}
/// The same with every atom started at rest at r.
pub fn simulate_from(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],r:[f64;3])->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,Some(r))
}
fn run(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],origin:Option<[f64;3]>)->Outcome{
    let engine=Engine{model,settings:settings.clone()};
    // The tasks in order: (their group, their first trajectory, the one after their last).
    let bounds=|g:u64|(first+g*group,(first+(g+1)*group).min(first+count));
    let tasks:Vec<(u64,u64,u64)>=(0..count.div_ceil(group)).flat_map(|g|{let(start,end)=bounds(g);(start..end).step_by(TASK as usize).map(move|s|(g,s,(s+TASK).min(end)))}).collect();
    let results:Vec<(u64,Group,Vec<Vec<([f64;3],[f64;3])>>,u64,f64)>=tasks.par_iter().map(|&(g,from,to)|{
        let(start,end)=bounds(g);
        let mut sums=Group::new(outputs,(to-from) as usize);let mut samples=vec![vec![];listed.len()];let(mut jumps,mut doppler)=(0,0_f64);
        let mut windows=Windows::new(settings,steps,outputs);let mut held=vec![None;outputs];
        for index in from..to{
            let mut t=initial_trajectory(model,settings,initial,seed,index,Stratum{first:start,size:end-start});
            if let Some(r)=origin{t.r=r;t.kappa=[0.;3];}
            windows.clear();held.fill(None);
            let mut record=|t:&Trajectory,i:usize,held:&mut Vec<Option<([f64;STATES],f64,f64)>>|{
                if !t.alive{return;}
                let point=model.point(t.r,t.kappa);
                held[i]=Some((model.populations(t.cluster,&point,&t.psi,t.time),model.profile_error(t.r),model.doppler_ratio(t.kappa)));
                if let Some(k)=listed.iter().position(|&l|l==i){samples[k].push((t.r,t.kappa));}
            };
            record(&t,0,&mut held);windows.add(0,&t);
            for i in 1..outputs{
                for k in 1..=steps{if t.alive{engine.step(&mut t);if t.alive{windows.add((i-1)*steps+k,&t);}}}
                record(&t,i,&mut held);
            }
            // Half a period past the last output: its windows' far halves.
            for k in 1..=windows.reach{if t.alive{engine.step(&mut t);if t.alive{windows.add((outputs-1)*steps+k,&t);}}}
            for(i,p)in held.iter().enumerate(){if let Some((p,profile,doppler))=p{sums.add(i,&windows.means(i),p,*profile,*doppler);}}
            jumps+=t.jumps;doppler=doppler.max(t.doppler);
        }
        (g,sums,samples,jumps,doppler)
    }).collect();
    // A group's tasks merged in order: the sums do not depend on the threads.
    let mut samples=vec![vec![];listed.len()];let(mut jumps,mut doppler)=(0,0_f64);let mut out:Vec<Group>=vec![];let mut last=None;
    for(g,sums,s,j,d)in results{
        if last==Some(g){out.last_mut().unwrap().merge(&sums);}else{out.push(sums);last=Some(g);}
        for(k,v)in s.into_iter().enumerate(){samples[k].extend(v);}jumps+=j;doppler=doppler.max(d);
    }
    Outcome{groups:out,samples,jumps,doppler}
}

/// One trajectory's motion sums about every output time: per output and lab axis the steps counted
/// and Σκ, Σκ², Σr, Σr² over its window -- the steps within `half` of the output's (at most as many
/// as the output's own, so a window stays centred), one step without averaging.
struct Windows{half:[usize;3],steps:usize,reach:usize,sums:Vec<[[f64;5];3]>}
impl Windows{
    fn new(settings:&Settings,steps:usize,outputs:usize)->Windows{
        let half:[usize;3]=std::array::from_fn(|a|{let w=settings.lab[a];if settings.average&&w>0.{(std::f64::consts::PI/w/settings.dt).round() as usize}else{0}});
        Windows{half,steps,reach:half.into_iter().max().unwrap(),sums:vec![[[0.;5];3];outputs]}
    }
    fn clear(&mut self){for s in &mut self.sums{*s=[[0.;5];3];}}
    /// The trajectory at step k (from t = 0) into every window holding it.
    fn add(&mut self,k:usize,t:&Trajectory){
        let last=self.sums.len()-1;
        for a in 0..3{
            let h=self.half[a];let(x,q)=(t.r[a],t.kappa[a]);
            for i in k.saturating_sub(h).div_ceil(self.steps)..=((k+h)/self.steps).min(last){
                let at=i*self.steps;if k.abs_diff(at)<=h.min(at){let s=&mut self.sums[i][a];s[0]+=1.;s[1]+=q;s[2]+=q*q;s[3]+=x;s[4]+=x*x;}
            }
        }
    }
    /// Output i's means: κ, κ², r, r² per axis.
    fn means(&self,i:usize)->[[f64;4];3]{std::array::from_fn(|a|{let s=&self.sums[i][a];[s[1]/s[0],s[2]/s[0],s[3]/s[0],s[4]/s[0]]})}
}

/// A pilot: `count` trajectories over the outputs, the held atoms' states at every output kept (at
/// most a few thousand, evenly thinned), and the jumps they made.
pub fn sample_states(model:&Model,settings:&Settings,initial:&Initial,seed:u64,count:u64,steps:usize,outputs:usize)->(Vec<Trajectory>,u64){
    let engine=Engine{model,settings:settings.clone()};
    let every=((count as usize*outputs)/4096).max(1);
    let runs:Vec<(Vec<Trajectory>,u64)>=(0..count).into_par_iter().map(|index|{
        let mut t=initial_from(model,settings,initial,seed,Purpose::Pilot,index,Stratum::single(index));let mut kept=vec![];
        for i in 1..outputs{for _ in 0..steps{if t.alive{engine.step(&mut t);}}if t.alive&&(i+index as usize)%every==0{kept.push(t.clone());}}
        (kept,t.jumps)
    }).collect();
    let jumps=runs.iter().map(|r|r.1).sum();(runs.into_iter().flat_map(|r|r.0).collect(),jumps)
}
