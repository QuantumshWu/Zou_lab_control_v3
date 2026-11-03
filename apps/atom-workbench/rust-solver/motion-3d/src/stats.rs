//! The ensemble's observables: sums over groups of trajectories at every output time, and their
//! estimates with standard errors by the jackknife over the groups.
use crate::STATES;

/// kB/ħ in rad/us per μK: a temperature T as the rate kT/ħ.
pub const KT_PER_UK:f64=1.380649e-23/1.0545718176461565e-34*1e-12;

/// Sums over the atoms still held at one output time: their number, Σκ, Σκ², Σr, Σr² per axis,
/// Σ populations, Σ of the Gaussian beams' profile error at the atoms (model.rs
/// `Model::profile_error`) and Σ (kv/σ)² (`Model::doppler_ratio`), with the number of trajectories
/// they come from.
#[derive(Clone,Debug)]
pub struct Sums{
    pub trajectories:f64,
    pub held:f64,
    pub kappa:[f64;3],
    pub kappa2:[f64;3],
    pub position:[f64;3],
    pub position2:[f64;3],
    pub populations:[f64;STATES],
    pub profile:f64,
    pub doppler:f64,
}
impl Sums{
    pub fn zero()->Sums{Sums{trajectories:0.,held:0.,kappa:[0.;3],kappa2:[0.;3],position:[0.;3],position2:[0.;3],populations:[0.;STATES],profile:0.,doppler:0.}}
    fn combine(&self,o:&Sums,sign:f64)->Sums{
        let mut out=self.clone();out.trajectories+=sign*o.trajectories;out.held+=sign*o.held;
        for a in 0..3{out.kappa[a]+=sign*o.kappa[a];out.kappa2[a]+=sign*o.kappa2[a];out.position[a]+=sign*o.position[a];out.position2[a]+=sign*o.position2[a];}
        for(x,y)in out.populations.iter_mut().zip(&o.populations){*x+=sign*y;}
        out.profile+=sign*o.profile;out.doppler+=sign*o.doppler;
        out
    }
}

/// One group of trajectories: its sums at every output time.
#[derive(Clone,Debug)]
pub struct Group{pub sums:Vec<Sums>}
impl Group{
    pub fn new(outputs:usize,trajectories:usize)->Group{
        let mut s=Sums::zero();s.trajectories=trajectories as f64;Group{sums:vec![s;outputs]}
    }
    /// Another group's sums added in: the two as one group.
    pub fn merge(&mut self,o:&Group){for(s,x)in self.sums.iter_mut().zip(&o.sums){*s=s.combine(x,1.);}}
    /// One atom held at output i: its populations there and its motion's means about it -- κ, κ², r
    /// and r² per axis, at the output time or over a window around it -- and its profile error.
    pub fn add(&mut self,i:usize,motion:&[[f64;4];3],populations:&[f64;STATES],profile:f64,doppler:f64){
        let s=&mut self.sums[i];s.held+=1.;s.profile+=profile;s.doppler+=doppler;
        for a in 0..3{s.kappa[a]+=motion[a][0];s.kappa2[a]+=motion[a][1];s.position[a]+=motion[a][2];s.position2[a]+=motion[a][3];}
        for(x,p)in s.populations.iter_mut().zip(populations){*x+=p;}
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
    /// n̄ per axis relative to its trap frequency, and its standard error.
    pub occupation:[f64;3],
    pub occupation_error:[f64;3],
    pub populations:Vec<f64>,
    pub population_errors:Vec<f64>,
    /// The probability that the atom has been lost, and its standard error.
    pub loss:f64,
    pub loss_error:f64,
    /// ⟨r²⟩ per axis (um²).
    pub position2:[f64;3],
    /// The held atoms' mean profile error and mean (kv/σ)² of the first order in the Doppler shift.
    pub profile:f64,
    pub doppler:f64,
    /// The trajectories behind the estimate.
    pub trajectories:f64,
}

/// What the estimates need of the model: ħ/m (um²/us) and the trap frequencies per axis (rad/us).
#[derive(Clone,Copy,Debug)]
pub struct Scales{pub hbar_over_mass:f64,pub frequencies:[f64;3]}

/// The estimates of sums.
pub fn estimate(g:&Sums,s:&Scales)->Estimate{
    let n=g.held;let mut e=Estimate{loss:1.-n/g.trajectories,trajectories:g.trajectories,..Default::default()};
    if n<=0.{e.populations=vec![0.;STATES];return e;}
    for a in 0..3{
        let(mean,second)=(g.kappa[a]/n,g.kappa2[a]/n);let variance=(second-mean*mean).max(0.);
        e.temperature[a]=variance*s.hbar_over_mass/KT_PER_UK;e.mean_velocity[a]=mean*s.hbar_over_mass;
        let x2=g.position2[a]/n;e.position2[a]=x2;let w=s.frequencies[a];
        e.occupation[a]=if w>0.{(second*s.hbar_over_mass/2.+w*w/s.hbar_over_mass*x2/2.)/w-0.5}else{0.};
    }
    e.temperature_total=e.temperature.iter().sum::<f64>()/3.;e.profile=g.profile/n;e.doppler=g.doppler/n;
    e.populations=g.populations.iter().map(|p|p/n).collect();
    e
}

/// The pooled estimates at every output time, with jackknife standard errors over the groups.
pub fn summarize(groups:&[Group],s:&Scales)->Vec<Estimate>{
    let outputs=groups[0].sums.len();let count=groups.len();
    (0..outputs).map(|i|{
        let all=groups.iter().skip(1).fold(groups[0].sums[i].clone(),|acc,g|acc.combine(&g.sums[i],1.));
        let mut e=estimate(&all,s);
        if count<2{e.population_errors=vec![f64::INFINITY;STATES];e.temperature_error=[f64::INFINITY;3];e.temperature_total_error=f64::INFINITY;e.occupation_error=[f64::INFINITY;3];e.loss_error=f64::INFINITY;return e;}
        let leave:Vec<Estimate>=groups.iter().map(|g|estimate(&all.combine(&g.sums[i],-1.),s)).collect();
        let factor=(count-1) as f64/count as f64;
        let spread=|f:&dyn Fn(&Estimate)->f64|->f64{let m=leave.iter().map(|x|f(x)).sum::<f64>()/count as f64;(factor*leave.iter().map(|x|(f(x)-m).powi(2)).sum::<f64>()).sqrt()};
        for a in 0..3{e.temperature_error[a]=spread(&|x|x.temperature[a]);e.occupation_error[a]=spread(&|x|x.occupation[a]);}
        e.temperature_total_error=spread(&|x|x.temperature_total);
        e.population_errors=(0..STATES).map(|k|spread(&|x|x.populations[k])).collect();
        e.loss_error=spread(&|x|x.loss);
        e
    }).collect()
}
