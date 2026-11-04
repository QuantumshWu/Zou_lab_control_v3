//! The ensemble's observables: sums over groups of trajectories at every output time, and their
//! estimates with standard errors by the jackknife over the groups.
use crate::STATES;

/// kB/ħ in rad/us per μK: a temperature T as the rate kT/ħ.
pub const KT_PER_UK:f64=1.380649e-23/1.0545718176461565e-34*1e-12;

/// Sums over the atoms still held at one output time: their number, Σκ, Σκ², Σr, Σr² per axis,
/// Σ populations, Σ of the folded excited population (the elimination's η²), Σ of the Gaussian
/// beams' profile error at the atoms (model.rs
/// `Model::profile_error`) and Σ (kv/σ)² (`Model::doppler_ratio`), with the number of trajectories
/// they come from; and over every trajectory, held or lost, the control variates' (control.rs): per
/// lab axis the harmonic control's Σc, Σc² and Σ c κ² (κ² the held atom's, none when lost), and the
/// held twin's Σc per state with, over the states, Σ c² and Σ c p (p the held atom's population).
#[derive(Clone,Debug)]
pub struct Sums{
    pub trajectories:f64,
    pub held:f64,
    pub kappa:[f64;3],
    pub kappa2:[f64;3],
    pub position:[f64;3],
    pub position2:[f64;3],
    pub populations:[f64;STATES],
    pub folded:f64,
    pub profile:f64,
    pub doppler:f64,
    pub harmonic:[[f64;3];3],
    pub twin:[f64;STATES],
    pub twin2:[f64;2],
}
impl Sums{
    pub fn zero()->Sums{Sums{trajectories:0.,held:0.,kappa:[0.;3],kappa2:[0.;3],position:[0.;3],position2:[0.;3],populations:[0.;STATES],folded:0.,profile:0.,doppler:0.,
        harmonic:[[0.;3];3],twin:[0.;STATES],twin2:[0.;2]}}
    fn combine(&self,o:&Sums,sign:f64)->Sums{
        let mut out=self.clone();out.trajectories+=sign*o.trajectories;out.held+=sign*o.held;
        for a in 0..3{out.kappa[a]+=sign*o.kappa[a];out.kappa2[a]+=sign*o.kappa2[a];out.position[a]+=sign*o.position[a];out.position2[a]+=sign*o.position2[a];
            for j in 0..3{out.harmonic[a][j]+=sign*o.harmonic[a][j];}}
        for(x,y)in out.populations.iter_mut().zip(&o.populations){*x+=sign*y;}
        for(x,y)in out.twin.iter_mut().zip(&o.twin){*x+=sign*y;}
        out.folded+=sign*o.folded;out.profile+=sign*o.profile;out.doppler+=sign*o.doppler;out.twin2[0]+=sign*o.twin2[0];out.twin2[1]+=sign*o.twin2[1];
        out
    }
}

/// An atom held at an output: its motion's means about it -- κ, κ², r and r² per axis, at the
/// output time or over a window around it --, its populations and folded excited population, its
/// profile error and (kv/σ)².
pub struct Atom<'a>{pub motion:[[f64;4];3],pub populations:&'a [f64;STATES],pub folded:f64,pub profile:f64,pub doppler:f64}
/// A trajectory's control variates at an output (control.rs): the harmonic control per lab axis and,
/// with a held twin, the twin's populations.
pub struct Controls<'a>{pub harmonic:[f64;3],pub twin:Option<&'a [f64;STATES]>}

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
    /// One trajectory at output i: the atom when it is held there, and its control variates.
    pub fn add(&mut self,i:usize,atom:Option<&Atom>,controls:&Controls){
        let s=&mut self.sums[i];
        if let Some(x)=atom{
            s.held+=1.;s.folded+=x.folded;s.profile+=x.profile;s.doppler+=x.doppler;
            for a in 0..3{s.kappa[a]+=x.motion[a][0];s.kappa2[a]+=x.motion[a][1];s.position[a]+=x.motion[a][2];s.position2[a]+=x.motion[a][3];}
            for(y,p)in s.populations.iter_mut().zip(x.populations){*y+=p;}
        }
        for a in 0..3{let(c,y)=(controls.harmonic[a],atom.map_or(0.,|x|x.motion[a][1]));s.harmonic[a][0]+=c;s.harmonic[a][1]+=c*c;s.harmonic[a][2]+=c*y;}
        if let Some(twin)=controls.twin{
            for k in 0..STATES{let(c,y)=(twin[k],atom.map_or(0.,|x|x.populations[k]));s.twin[k]+=c;s.twin2[0]+=c*c;s.twin2[1]+=c*y;}
        }
    }
}

/// The control variates' exact means at one output (control.rs): the harmonic control's per lab axis
/// and, with a held twin, its populations'.
#[derive(Clone,Debug)]
pub struct Means{pub harmonic:[f64;3],pub twin:Option<[f64;STATES]>}

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
    /// The held atoms' mean folded excited population (the elimination's η²), mean profile error
    /// and mean (kv/σ)² of the first order in the Doppler shift.
    pub folded:f64,
    pub profile:f64,
    pub doppler:f64,
    /// The trajectories behind the estimate.
    pub trajectories:f64,
}

/// What the estimates need of the model: ħ/m (um²/us) and the trap frequencies per axis (rad/us).
#[derive(Clone,Copy,Debug)]
pub struct Scales{pub hbar_over_mass:f64,pub frequencies:[f64;3]}

/// A control variate's correction at one output: its means (`Means`) and β, the regression of the
/// held atoms' κ² per lab axis and of their populations on the controls over every trajectory so far
/// -- one β for all the populations, so they keep their sum.
pub struct Correction<'a>{pub means:&'a Means,pub harmonic:[f64;3],pub twin:f64}
impl Correction<'_>{
    /// The regression of sums on their controls (none where a control does not vary).
    pub fn new<'a>(g:&Sums,means:&'a Means)->Correction<'a>{
        let all=g.trajectories;
        let harmonic=std::array::from_fn(|a|{let[c,cc,cy]=g.harmonic[a];let variance=cc-c*c/all;if variance>0.{(cy-c*g.kappa2[a]/all)/variance}else{0.}});
        let variance=g.twin2[0]-g.twin.iter().map(|c|c*c).sum::<f64>()/all;
        let covariance=g.twin2[1]-g.twin.iter().zip(&g.populations).map(|(c,y)|c*y).sum::<f64>()/all;
        Correction{means,harmonic,twin:if means.twin.is_some()&&variance>0.{covariance/variance}else{0.}}
    }
}

/// The estimates of sums, with a correction by the control variates: the held atoms' κ² and
/// populations less β times their controls' excess over their means -- the same expectation, the
/// variance less the part the controls account for.
pub fn estimate(g:&Sums,s:&Scales,correction:Option<&Correction>)->Estimate{
    let n=g.held;let all=g.trajectories;let mut e=Estimate{loss:1.-n/all,trajectories:all,..Default::default()};
    if n<=0.{e.populations=vec![0.;STATES];return e;}
    let(mut kappa2,mut populations)=(g.kappa2,g.populations);
    if let Some(c)=correction{
        for a in 0..3{kappa2[a]-=c.harmonic[a]*(g.harmonic[a][0]-all*c.means.harmonic[a]);}
        if let Some(mean)=&c.means.twin{for k in 0..STATES{populations[k]-=c.twin*(g.twin[k]-all*mean[k]);}}
    }
    for a in 0..3{
        let(mean,second)=(g.kappa[a]/n,kappa2[a]/n);let variance=(second-mean*mean).max(0.);
        e.temperature[a]=variance*s.hbar_over_mass/KT_PER_UK;e.mean_velocity[a]=mean*s.hbar_over_mass;
        let x2=g.position2[a]/n;e.position2[a]=x2;let w=s.frequencies[a];
        e.occupation[a]=if w>0.{(second*s.hbar_over_mass/2.+w*w/s.hbar_over_mass*x2/2.)/w-0.5}else{0.};
    }
    e.temperature_total=e.temperature.iter().sum::<f64>()/3.;e.folded=g.folded/n;e.profile=g.profile/n;e.doppler=g.doppler/n;
    e.populations=populations.iter().map(|p|p/n).collect();
    e
}

/// The pooled estimates at every output time, with jackknife standard errors over the groups that reach it (a batch's
/// trajectories run only as far as the outputs need them, solve.rs # Batches), the
/// control variates' means per output where given: their β from every trajectory, held when a group
/// is left out -- recomputed with it, a β that rests on a few trajectories at an early output (a few
/// twins that have jumped) swings from group to group and its spread swamps the error.
pub fn summarize(groups:&[Group],s:&Scales,means:&[Means])->Vec<Estimate>{
    let outputs=groups[0].sums.len();
    (0..outputs).map(|i|{
        let present:Vec<&Sums>=groups.iter().map(|g|&g.sums[i]).filter(|x|x.trajectories>0.).collect();let count=present.len();
        let all=present.iter().fold(Sums::zero(),|acc,x|acc.combine(x,1.));
        let correction=means.get(i).map(|m|Correction::new(&all,m));let m=correction.as_ref();
        let mut e=estimate(&all,s,m);
        if count<2{e.population_errors=vec![f64::INFINITY;STATES];e.temperature_error=[f64::INFINITY;3];e.temperature_total_error=f64::INFINITY;e.occupation_error=[f64::INFINITY;3];e.loss_error=f64::INFINITY;return e;}
        let leave:Vec<Estimate>=present.iter().map(|x|estimate(&all.combine(x,-1.),s,m)).collect();
        let factor=(count-1) as f64/count as f64;
        let spread=|f:&dyn Fn(&Estimate)->f64|->f64{let m=leave.iter().map(|x|f(x)).sum::<f64>()/count as f64;(factor*leave.iter().map(|x|(f(x)-m).powi(2)).sum::<f64>()).sqrt()};
        for a in 0..3{e.temperature_error[a]=spread(&|x|x.temperature[a]);e.occupation_error[a]=spread(&|x|x.occupation[a]);}
        e.temperature_total_error=spread(&|x|x.temperature_total);
        e.population_errors=(0..STATES).map(|k|spread(&|x|x.populations[k])).collect();
        e.loss_error=spread(&|x|x.loss);
        e
    }).collect()
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::rng::{normal_quantile,Purpose,Rng};
    /// The control variates' estimate on drawn sums: a population 0.5 + 0.1 x + noise and its
    /// control 0.5 + 0.1 x with the known mean 0.5; and an uncorrelated control. The estimate keeps
    /// the expectation and the populations' sum, its error shrinks by the correlation's, and an
    /// uncorrelated control leaves it to within its error.
    #[test]
    fn the_control_variates_keep_the_mean_and_shrink_the_error(){
        let mut rng=Rng::new(3,Purpose::Initial,0,0);let mut normal=||normal_quantile(rng.interior());
        let scales=Scales{hbar_over_mass:1.,frequencies:[1.;3]};
        for correlated in [true,false]{
            let groups:Vec<Group>=(0..32).map(|_|{let mut g=Group::new(1,64);
                for _ in 0..64{
                    let x=normal();let(p,c)=(0.5+0.1*x+0.01*normal(),if correlated{0.5+0.1*x}else{0.5+0.1*normal()});
                    let mut populations=[0.;STATES];populations[0]=p;populations[1]=1.-p;let mut twin=[0.;STATES];twin[0]=c;twin[1]=1.-c;
                    let atom=Atom{motion:[[0.,1.,0.,0.];3],populations:&populations,folded:0.,profile:0.,doppler:0.};
                    g.add(0,Some(&atom),&Controls{harmonic:[0.;3],twin:Some(&twin)});
                }g}).collect();
            let mut mean=[0.;STATES];mean[0]=0.5;mean[1]=0.5;
            let(with,plain)=(&summarize(&groups,&scales,&[Means{harmonic:[0.;3],twin:Some(mean)}])[0],&summarize(&groups,&scales,&[])[0]);
            assert!((with.populations.iter().sum::<f64>()-1.).abs()<1e-12,"the populations keep their sum");
            assert!((with.populations[0]-0.5).abs()<=4.*with.population_errors[0],"{} ± {}",with.populations[0],with.population_errors[0]);
            if correlated{assert!(with.population_errors[0]<0.2*plain.population_errors[0],"{} against {}",with.population_errors[0],plain.population_errors[0]);}
            else{assert!((with.populations[0]-plain.populations[0]).abs()<=plain.population_errors[0]&&with.population_errors[0]<=1.1*plain.population_errors[0]);}
        }
    }
}
