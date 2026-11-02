//! The ensemble's observables at an output time from the trajectories' records: weighted means
//! over the atoms still held, each with its standard error (the weighted spread over the
//! trajectories), and the probability of loss.
use crate::engine::{RECORD,REC_EXCITED,REC_GROUND,REC_KX,REC_TAIL,REC_X};
use crate::model::Model;
use atom_motion_core::STATES;

/// One output time's estimates; `*_error` are standard errors.
#[derive(Clone,Debug,Default)]
pub struct Sample{
    /// The 32 internal populations (model basis) of the held atom.
    pub populations:Vec<f64>,
    pub population_errors:Vec<f64>,
    /// Per plane axis: T = m Var(v)/k_B (uK), the mean velocity (m/s), ⟨x²⟩ (um²), ⟨κ²⟩ (1/um²).
    pub temperature:[f64;2],
    pub temperature_errors:[f64;2],
    pub mean_velocity:[f64;2],
    pub position_variance:[f64;2],
    pub momentum_square:[f64;2],
    /// The parts of ⟨x²⟩, ⟨y²⟩, ⟨κx²⟩, ⟨κy²⟩ beyond 0.7 of the grid's reach, relative to the whole.
    pub tails:[f64;4],
    /// The probability that the atom has been lost, and its error.
    pub loss:f64,
    pub loss_error:f64,
    /// The weight of the held trajectories, the number held.
    pub held:usize,
}

/// μK per (rad/um)² of κ variance: ħ²/(m k_B).
pub fn temperature_unit(model:&Model)->f64{model.hbar_over_mass*crate::model::HBAR/crate::model::BOLTZMANN*1e12}

/// The weighted mean of v over the held trajectories and its standard error.
fn mean(weights:&[f64],held:&[usize],v:&dyn Fn(usize)->f64)->(f64,f64){
    let total:f64=held.iter().map(|&t|weights[t]).sum();
    if total<=0.{return(0.,0.);}
    let m=held.iter().map(|&t|weights[t]*v(t)).sum::<f64>()/total;
    let var=held.iter().map(|&t|(weights[t]*(v(t)-m)).powi(2)).sum::<f64>()/(total*total);
    (m,var.sqrt())
}

/// The sample of one output time: `records` (RECORD floats per trajectory) and `components`
/// (0: lost) as the engine read them, the trajectories' weights.
pub fn sample(model:&Model,records:&[f32],components:&[i32],weights:&[f64])->Sample{
    let b=components.len();let g=&model.ground;
    let held:Vec<usize>=(0..b).filter(|&t|components[t]>0).collect();
    let r=|t:usize,i:usize|records[t*RECORD+i] as f64;
    let mut s=Sample{populations:vec![0.;STATES],population_errors:vec![0.;STATES],held:held.len(),..Default::default()};
    // The records' ground states are in block order: F = 1's, then F = 2's.
    for(k,&b)in model.block_states[0].iter().chain(&model.block_states[1]).enumerate(){
        let state=g.ground_states[b];let(m,e)=mean(weights,&held,&|t|r(t,REC_GROUND+k));s.populations[state]=m;s.population_errors[state]=e;}
    for(k,&state)in g.excited_states.iter().enumerate(){let(m,e)=mean(weights,&held,&|t|r(t,REC_EXCITED+k));s.populations[state]=m;s.population_errors[state]=e;}
    let unit=temperature_unit(model);
    for d in 0..2{
        let(k1,_)=mean(weights,&held,&|t|r(t,REC_KX+d));let(k2,_)=mean(weights,&held,&|t|r(t,REC_KX+2+d));
        // T's error by the delta method: the spread of κ² − 2⟨κ⟩κ.
        let(_,e)=mean(weights,&held,&|t|r(t,REC_KX+2+d)-2.*k1*r(t,REC_KX+d));
        s.temperature[d]=unit*(k2-k1*k1);s.temperature_errors[d]=unit*e;
        s.mean_velocity[d]=model.hbar_over_mass*k1; // um/us = m/s
        s.momentum_square[d]=k2;
        let(x2,_)=mean(weights,&held,&|t|r(t,REC_X+2+d));s.position_variance[d]=x2;
        let(xt,_)=mean(weights,&held,&|t|r(t,REC_TAIL+d));let(kt,_)=mean(weights,&held,&|t|r(t,REC_TAIL+2+d));
        s.tails[d]=if x2>0.{xt/x2}else{0.};s.tails[2+d]=if k2>0.{kt/k2}else{0.};
    }
    let total:f64=weights.iter().sum();
    let lost:f64=(0..b).filter(|&t|components[t]==0).map(|t|weights[t]).sum();
    s.loss=lost/total;
    s.loss_error=((0..b).map(|t|(weights[t]*((components[t]==0) as i32 as f64-s.loss)).powi(2)).sum::<f64>()).sqrt()/total;
    s
}

/// n̄ = ħ⟨κ²⟩/(2mω) + mω⟨x²⟩/(2ħ) − ½ along one axis, for the trap frequency ω (rad/us).
pub fn occupation(model:&Model,momentum_square:f64,position_variance:f64,omega:f64)->f64{
    model.hbar_over_mass*momentum_square/(2.*omega)+omega*position_variance/(2.*model.hbar_over_mass)-0.5
}
