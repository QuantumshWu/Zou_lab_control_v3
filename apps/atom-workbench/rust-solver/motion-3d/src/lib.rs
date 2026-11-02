//! Semiclassical 3D motion of a single Rb87 atom in an optical tweezer: a classical point (r, p)
//! in 3D carrying the internal state of the eliminated ground manifold (motion-core `ground`),
//! evolved by quantum jumps.
pub mod engine;
pub mod model;
pub mod rng;
pub mod solve;
pub mod stats;

use engine::{Engine,Settings,Trajectory};
use model::Model;
use rayon::prelude::*;
use rng::Rng;
use stats::{Group,KT_PER_UK};
use atom_motion_core::{Complex,Request,STATES};

/// The initial internal states (block, position in the block, probability) and temperature.
#[derive(Clone,Debug)]
pub struct Initial{pub states:Vec<(usize,usize,f64)>,pub temperature_uk:f64}
impl Initial{
    /// From the request's initial density: its ground populations (the density is diagonal in the
    /// model basis, as the app writes it).
    pub fn new(model:&Model,request:&Request)->Result<Initial,String>{
        let g=&model.ground;let rho=&request.initial_density;let mut states=vec![];
        for i in 0..STATES{for j in 0..STATES{if i!=j&&rho[i*STATES+j].abs()>0.{return Err("The initial density must be diagonal in the model basis".into());}}}
        for(b,block)in model.blocks.iter().enumerate(){for(x,&k)in block.states.iter().enumerate(){
            let p=rho[g.ground_states[k]*STATES+g.ground_states[k]].re;if p>0.{states.push((b,x,p));}
        }}
        let total:f64=states.iter().map(|s|s.2).sum();
        if !(total>0.){return Err("The initial density has no ground population".into());}
        for s in &mut states{s.2/=total;}
        Ok(Initial{states,temperature_uk:request.temperature_uk})
    }
}

/// The independence Metropolis chain's length for the initial positions: its proposal, the
/// harmonic trap's Boltzmann distribution, is the target to a few percent, so the chain forgets
/// its start within a few accepted moves.
const METROPOLIS_STEPS:usize=32;

/// A trajectory's initial state: an internal basis state drawn from the initial populations; the
/// position from the Boltzmann distribution at T in that state's trap potential (independence
/// Metropolis from its harmonic approximation); the momentum Maxwell; position and momentum along
/// each axis scaled to the harmonic oscillator's thermal Wigner function, T* = (ħω/2k)coth(ħω/2kT).
pub fn initial_trajectory(model:&Model,settings:&Settings,initial:&Initial,seed:u64,index:u64)->Trajectory{
    let mut rng=Rng::new(seed,index);
    let mut x=rng.uniform();let mut pick=*initial.states.last().unwrap();
    for s in &initial.states{if x<s.2{pick=*s;break;}x-=s.2;}
    let(block,state,_)=pick;
    let frequencies=model.trap_frequencies(block,state);let kt=KT_PER_UK*initial.temperature_uk;let hm=model.hbar_over_mass;
    let sigma:[f64;3]=std::array::from_fn(|a|if settings.free[a]&&frequencies[a]>0.{(kt*hm).sqrt()/frequencies[a]}else{0.});
    let draw=|rng:&mut Rng|->[f64;3]{std::array::from_fn(|a|sigma[a]*rng.normal())};
    let weight=|r:[f64;3]|->f64{-model.potential(block,state,r)/kt+(0..3).filter(|&a|sigma[a]>0.).map(|a|r[a]*r[a]/(2.*sigma[a]*sigma[a])).sum::<f64>()};
    let mut r=draw(&mut rng);let mut w=weight(r);
    for _ in 0..METROPOLIS_STEPS{let next=draw(&mut rng);let wn=weight(next);if rng.open_uniform().ln()<wn-w{r=next;w=wn;}}
    let quantum:[f64;3]=std::array::from_fn(|a|{let f=frequencies[a];if f>0.{let half=f/(2.*kt);(half/half.tanh()).sqrt()}else{1.}});
    for a in 0..3{r[a]*=quantum[a];}
    let kappa:[f64;3]=std::array::from_fn(|a|if settings.free[a]{(kt/hm).sqrt()*quantum[a]*rng.normal()}else{0.});
    let mut psi=[Complex::ZERO;5];psi[state]=Complex::ONE;
    let threshold=engine::jump_stream(seed,index,0).open_uniform();
    Trajectory{block,psi,r,kappa,threshold,alive:true,jumps:0,seed,index}
}

/// The outputs of a run: one group per `group` trajectories, sums at every output time, and the
/// held atoms' (r, κ) at the listed outputs.
pub struct Outcome{pub groups:Vec<Group>,pub samples:Vec<Vec<([f64;3],[f64;3])>>,pub jumps:u64}

/// Trajectories [first, first + count) of the run `seed`, `steps` steps of the settings' dt between
/// `outputs` output times (the first at t = 0), in groups of `group`; (r, κ) kept at `listed`.
pub fn simulate(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize])->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,None)
}
/// The same with every atom started at rest at r.
pub fn simulate_from(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],r:[f64;3])->Outcome{
    run(model,settings,initial,seed,first,count,group,steps,outputs,listed,Some(r))
}
fn run(model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,steps:usize,outputs:usize,listed:&[usize],origin:Option<[f64;3]>)->Outcome{
    let engine=Engine{model,settings:settings.clone()};
    let groups:Vec<u64>=(0..count.div_ceil(group)).collect();
    let results:Vec<(Group,Vec<Vec<([f64;3],[f64;3])>>,u64)>=groups.par_iter().map(|&g|{
        let(start,end)=(first+g*group,(first+(g+1)*group).min(first+count));
        let mut sums=Group::new(outputs,(end-start) as usize);let mut samples=vec![vec![];listed.len()];let mut jumps=0;
        for index in start..end{
            let mut t=initial_trajectory(model,settings,initial,seed,index);
            if let Some(r)=origin{t.r=r;t.kappa=[0.;3];}
            let mut record=|t:&Trajectory,i:usize|{
                if !t.alive{return;}
                let point=model.point(t.r,t.kappa);let p=model.populations(t.block,&point,&t.psi);sums.add(i,t.r,t.kappa,&p);
                if let Some(k)=listed.iter().position(|&l|l==i){samples[k].push((t.r,t.kappa));}
            };
            record(&t,0);
            for i in 1..outputs{
                for _ in 0..steps{if t.alive{engine.step(&mut t);}}
                record(&t,i);
            }
            jumps+=t.jumps;
        }
        (sums,samples,jumps)
    }).collect();
    let mut samples=vec![vec![];listed.len()];let mut jumps=0;let mut out=vec![];
    for(g,s,j)in results{out.push(g);for(k,v)in s.into_iter().enumerate(){samples[k].extend(v);}jumps+=j;}
    Outcome{groups:out,samples,jumps}
}

/// A pilot: `count` trajectories over the outputs, the held atoms' states at every output kept (at
/// most a few thousand, evenly thinned), and the jumps they made.
pub fn sample_states(model:&Model,settings:&Settings,initial:&Initial,seed:u64,count:u64,steps:usize,outputs:usize)->(Vec<Trajectory>,u64){
    let engine=Engine{model,settings:settings.clone()};
    let every=((count as usize*outputs)/4096).max(1);
    let runs:Vec<(Vec<Trajectory>,u64)>=(0..count).into_par_iter().map(|index|{
        let mut t=initial_trajectory(model,settings,initial,seed^0x5A3E_5EED,index);let mut kept=vec![];
        for i in 1..outputs{for _ in 0..steps{if t.alive{engine.step(&mut t);}}if t.alive&&(i+index as usize)%every==0{kept.push(t.clone());}}
        (kept,t.jumps)
    }).collect();
    let jumps=runs.iter().map(|r|r.1).sum();(runs.into_iter().flat_map(|r|r.0).collect(),jumps)
}
