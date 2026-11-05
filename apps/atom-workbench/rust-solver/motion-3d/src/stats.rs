//! The ensemble's observables: sums over groups of trajectories at every output time, and their
//! estimates with standard errors by the jackknife over the groups.
use crate::STATES;
use rayon::prelude::*;

/// kB/ħ in rad/us per μK: a temperature T as the rate kT/ħ.
pub const KT_PER_UK:f64=crate::model::BOLTZMANN/crate::model::HBAR*1e-12;

/// Sums over the atoms still held at one output time: their number, Σκ, Σκ² per axis,
/// Σ populations, Σ of the folded excited population (the elimination's η²), Σ of the Gaussian
/// beams' profile error at the atoms (model.rs
/// `Model::profile_error`) and Σ (kv/σ)² (`Model::doppler_ratio`), with the number of trajectories
/// they come from; and over every trajectory, held or lost, the harmonic control's (control.rs): per
/// lab axis its Σc, Σc² and Σ c κ² (κ² the held atom's, none when lost).
#[derive(Clone,Debug)]
pub struct Sums{
    pub trajectories:f64,
    pub held:f64,
    pub kappa:[f64;3],
    pub kappa2:[f64;3],
    pub populations:[f64;STATES],
    pub folded:f64,
    pub profile:f64,
    pub doppler:f64,
    pub harmonic:[[f64;3];3],
}
impl Sums{
    pub fn zero()->Sums{Sums{trajectories:0.,held:0.,kappa:[0.;3],kappa2:[0.;3],populations:[0.;STATES],folded:0.,profile:0.,doppler:0.,
        harmonic:[[0.;3];3]}}
    fn combine(&self,o:&Sums,sign:f64)->Sums{let mut out=self.clone();out.add(o,sign);out}
    fn add(&mut self,o:&Sums,sign:f64){
        self.trajectories+=sign*o.trajectories;self.held+=sign*o.held;
        for a in 0..3{self.kappa[a]+=sign*o.kappa[a];self.kappa2[a]+=sign*o.kappa2[a];
            for j in 0..3{self.harmonic[a][j]+=sign*o.harmonic[a][j];}}
        for(x,y)in self.populations.iter_mut().zip(&o.populations){*x+=sign*y;}
        self.folded+=sign*o.folded;self.profile+=sign*o.profile;self.doppler+=sign*o.doppler;
    }
}

/// An atom held at an output: its motion's means about it -- κ and κ² per axis, at the
/// output time or over a window around it --, its populations and folded excited population, its
/// profile error and (kv/σ)².
pub struct Atom<'a>{pub motion:[[f64;2];3],pub populations:&'a [f64;STATES],pub folded:f64,pub profile:f64,pub doppler:f64}
/// One group of trajectories: its sums at every output time.
#[derive(Clone,Debug)]
pub struct Group{pub sums:Vec<Sums>}
impl Group{
    pub fn new(outputs:usize,trajectories:usize)->Group{
        let mut s=Sums::zero();s.trajectories=trajectories as f64;Group{sums:vec![s;outputs]}
    }
    /// A group whose trajectories run through its first `reach` outputs alone: none are counted beyond.
    pub fn reaching(outputs:usize,reach:usize,trajectories:usize)->Group{
        let mut g=Group::new(outputs,trajectories);for s in g.sums.iter_mut().skip(reach){s.trajectories=0.;}g
    }
    /// Another group's sums added in: the two as one group.
    pub fn merge(&mut self,o:&Group){for(s,x)in self.sums.iter_mut().zip(&o.sums){*s=s.combine(x,1.);}}
    /// One trajectory at output i: the atom when it is held there, and its harmonic control per lab axis.
    pub fn add(&mut self,i:usize,atom:Option<&Atom>,harmonic:[f64;3]){
        let s=&mut self.sums[i];
        if let Some(x)=atom{
            s.held+=1.;s.folded+=x.folded;s.profile+=x.profile;s.doppler+=x.doppler;
            for a in 0..3{s.kappa[a]+=x.motion[a][0];s.kappa2[a]+=x.motion[a][1];}
            for(y,p)in s.populations.iter_mut().zip(x.populations){*y+=p;}
        }
        for a in 0..3{let(c,y)=(harmonic[a],atom.map_or(0.,|x|x.motion[a][1]));s.harmonic[a][0]+=c;s.harmonic[a][1]+=c*c;s.harmonic[a][2]+=c*y;}
    }
}

/// The estimates at one output time.
#[derive(Clone,Debug,Default)]
pub struct Estimate{
    /// T = m Var(v)/k_B per axis (μK), its standard error, the mean velocity (um/us).
    pub temperature:[f64;3],
    pub temperature_error:[f64;3],
    /// T_total = (T_x + T_y + T_z)/3, the temperature of the 3D velocity variance (μK), and its
    /// standard error from the same jackknife, so the axes' correlations are in it.
    pub temperature_total:f64,
    pub temperature_total_error:f64,
    pub mean_velocity:[f64;3],
    /// n̄ per axis relative to its trap frequency, ħ⟨κ²⟩/mω − ½ -- the energy on the axis over ħω, twice its kinetic part in a harmonic well (the virial theorem), which takes no
    /// place: the same in whichever well of a standing wave the atom sits --, and its standard error.
    pub occupation:[f64;3],
    pub occupation_error:[f64;3],
    pub populations:Vec<f64>,
    pub population_errors:Vec<f64>,
    /// The probability that the atom has been lost, and its standard error.
    pub loss:f64,
    pub loss_error:f64,
    /// The held atoms' mean folded excited population (the elimination's η²), mean profile error
    /// and mean (kv/σ)² of the first order in the Doppler shift.
    pub folded:f64,
    pub profile:f64,
    pub doppler:f64,
    /// The trajectories and the groups behind the estimate.
    pub trajectories:f64,
    pub groups:f64,
}

/// What the estimates need of the model: ħ/m (um²/us) and the trap frequencies per axis (rad/us).
#[derive(Clone,Copy,Debug)]
pub struct Scales{pub hbar_over_mass:f64,pub frequencies:[f64;3]}

/// The harmonic control's correction at one output: its exact means per lab axis (control.rs) and β, the regression of the
/// held atoms' κ² per lab axis on the control over every trajectory so far.
pub struct Correction{pub means:[f64;3],pub harmonic:[f64;3]}
impl Correction{
    /// The regression of sums on their control (none where it does not vary).
    pub fn new(g:&Sums,means:[f64;3])->Correction{
        let all=g.trajectories;
        let harmonic=std::array::from_fn(|a|{let[c,cc,cy]=g.harmonic[a];let variance=cc-c*c/all;if variance>0.{(cy-c*g.kappa2[a]/all)/variance}else{0.}});
        Correction{means,harmonic}
    }
}

/// The estimates of sums, with a correction by the harmonic control: the held atoms' κ²
/// less β times its excess over its means -- the same expectation, the
/// variance less the part the control accounts for.
pub fn estimate(g:&Sums,s:&Scales,correction:Option<&Correction>)->Estimate{
    let n=g.held;let m=moments(g,s,correction);
    Estimate{temperature:m.temperature,temperature_total:m.temperature_total,occupation:m.occupation,populations:m.populations.to_vec(),loss:m.loss,trajectories:g.trajectories,
        mean_velocity:if n>0.{std::array::from_fn(|a|g.kappa[a]/n*s.hbar_over_mass)}else{[0.;3]},
        folded:if n>0.{g.folded/n}else{0.},profile:if n>0.{g.profile/n}else{0.},doppler:if n>0.{g.doppler/n}else{0.},..Default::default()}
}
/// What the jackknife spreads (`summarize`), of `estimate`: no allocation per group left out.
struct Moments{temperature:[f64;3],temperature_total:f64,occupation:[f64;3],populations:[f64;STATES],loss:f64}
fn moments(g:&Sums,s:&Scales,correction:Option<&Correction>)->Moments{
    let n=g.held;let all=g.trajectories;let mut e=Moments{temperature:[0.;3],temperature_total:0.,occupation:[0.;3],populations:[0.;STATES],loss:1.-n/all};
    if n<=0.{return e;}
    let mut kappa2=g.kappa2;
    if let Some(c)=correction{for a in 0..3{kappa2[a]-=c.harmonic[a]*(g.harmonic[a][0]-all*c.means[a]);}}
    for a in 0..3{
        let(mean,second)=(g.kappa[a]/n,kappa2[a]/n);let variance=(second-mean*mean).max(0.);
        e.temperature[a]=variance*s.hbar_over_mass/KT_PER_UK;
        let w=s.frequencies[a];e.occupation[a]=if w>0.{second*s.hbar_over_mass/w-0.5}else{0.};
    }
    e.temperature_total=e.temperature.iter().sum::<f64>()/3.;
    for k in 0..STATES{e.populations[k]=g.populations[k]/n;}
    e
}
/// The jackknife's standard error of a value over the groups left out.
fn spread(leave:&[Moments],factor:f64,f:impl Fn(&Moments)->f64)->f64{let m=leave.iter().map(&f).sum::<f64>()/leave.len() as f64;(factor*leave.iter().map(|x|(f(x)-m).powi(2)).sum::<f64>()).sqrt()}

/// The pooled estimates at every output time, with jackknife standard errors over the groups that reach it (a batch's
/// trajectories run only as far as the outputs need them, solve.rs # Batches), the
/// harmonic control's means per output and lab axis where given: its β from every trajectory, held when a group
/// is left out -- recomputed with it, a β that rests on a few trajectories at an early output
/// swings from group to group and its spread swamps the error.
pub fn summarize<G:std::borrow::Borrow<Group>+Sync>(groups:&[G],s:&Scales,means:&[[f64;3]])->Vec<Estimate>{
    let outputs=groups[0].borrow().sums.len();
    (0..outputs).into_par_iter().map(|i|{
        let present:Vec<&Sums>=groups.iter().map(|g|&g.borrow().sums[i]).filter(|x|x.trajectories>0.).collect();let count=present.len();
        let mut all=Sums::zero();for x in &present{all.add(x,1.);}
        let correction=means.get(i).map(|&m|Correction::new(&all,m));let m=correction.as_ref();
        let mut e=estimate(&all,s,m);e.groups=count as f64;
        if count<2{e.population_errors=vec![f64::INFINITY;STATES];e.temperature_error=[f64::INFINITY;3];e.temperature_total_error=f64::INFINITY;e.occupation_error=[f64::INFINITY;3];e.loss_error=f64::INFINITY;return e;}
        let leave:Vec<Moments>=present.iter().map(|x|moments(&all.combine(x,-1.),s,m)).collect();
        let factor=(count-1) as f64/count as f64;
        for a in 0..3{e.temperature_error[a]=spread(&leave,factor,|x|x.temperature[a]);e.occupation_error[a]=spread(&leave,factor,|x|x.occupation[a]);}
        e.temperature_total_error=spread(&leave,factor,|x|x.temperature_total);
        e.population_errors=(0..STATES).map(|k|spread(&leave,factor,|x|x.populations[k])).collect();
        e.loss_error=spread(&leave,factor,|x|x.loss);
        e
    }).collect()
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::rng::{normal_quantile,Purpose,Rng};
    /// The harmonic control's estimate on drawn sums: an atom's κ² on one axis 1 + 0.1 x + noise and its
    /// control 1 + 0.1 x with the known mean 1; and an uncorrelated control. The estimate keeps
    /// the expectation, its error shrinks by the correlation's, and an
    /// uncorrelated control leaves it to within its error.
    #[test]
    fn the_control_variates_keep_the_mean_and_shrink_the_error(){
        let mut rng=Rng::new(3,Purpose::Initial,0,0);let mut normal=||normal_quantile(rng.interior());
        let scales=Scales{hbar_over_mass:1.,frequencies:[1.;3]};
        for correlated in [true,false]{
            let groups:Vec<Group>=(0..32).map(|_|{let mut g=Group::new(1,64);
                for _ in 0..64{
                    let x=normal();let(y,c)=(1.+0.1*x+0.01*normal(),if correlated{1.+0.1*x}else{1.+0.1*normal()});
                    let atom=Atom{motion:[[0.,y];3],populations:&[0.;STATES],folded:0.,profile:0.,doppler:0.};
                    g.add(0,Some(&atom),[c;3]);
                }g}).collect();
            let(with,plain)=(&summarize(&groups,&scales,&[[1.;3]])[0],&summarize(&groups,&scales,&[])[0]);
            let(t,e)=(with.temperature[0]*KT_PER_UK,with.temperature_error[0]*KT_PER_UK);let(p,o)=(plain.temperature[0]*KT_PER_UK,plain.temperature_error[0]*KT_PER_UK);
            assert!((t-1.).abs()<=4.*e,"{t} ± {e}");
            if correlated{assert!(e<0.2*o,"{e} against {o}");}
            else{assert!((t-p).abs()<=o&&e<=1.1*o);}
        }
    }
}
