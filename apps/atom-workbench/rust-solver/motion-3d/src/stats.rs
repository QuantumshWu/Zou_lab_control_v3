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

/// A quantity's statistical error, from its jackknife over the groups behind an estimate (`summarize`): its standard error,
/// the groups that read it -- every group that reaches the output for the loss and the rows as shown, those holding an atom
/// there for the held atoms' quantities -- and the skew of the group values, their third central moment over their second
/// per group (`skew`). An error at z standard errors reads it through one definition: the interval's distances below and
/// above the value (`below`, `above`), the larger of them (`at`).
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct Spread{pub error:f64,pub groups:f64,pub skew:f64}
impl Default for Spread{fn default()->Spread{Spread::EXACT}}
impl Spread{
    /// No error: a value known exactly (t = 0, the request's initial state).
    pub const EXACT:Spread=Spread{error:0.,groups:f64::INFINITY,skew:0.};
    /// Too few groups to read a spread on (`reads_spread`): its error unknown.
    fn unread(groups:f64)->Spread{Spread{error:f64::INFINITY,groups,skew:0.}}
    /// Its quantile at z: Student's t at one degree of freedom fewer than the groups that read it, at the normal's tail beyond z
    /// (`student`) -- the jackknife's t interval (Tukey's), the batch means' t of their batches less one --; z on unlimited groups.
    fn quantile(&self,z:f64)->f64{if self.groups==f64::INFINITY{z}else{student(z,self.groups-1.)}}
    /// Hall's shift of its interval at z for the skew of the group values, (2z² + 1)/6 of it: the mean of skewed values lies short of
    /// its long tail more often than its standard error says, the studentized interval's both ends moved toward that tail.
    fn shift(&self,z:f64)->f64{(2.*z*z+1.)/6.*self.skew}
    /// The distances of its interval at z standard errors below the value and above it, [v + d − q SE, v + d + q SE] (q its
    /// `quantile`, d its `shift`); unknown on fewer than two groups.
    pub fn below(&self,z:f64)->f64{if reads_spread(self.groups){self.quantile(z)*self.error-self.shift(z)}else{f64::INFINITY}}
    pub fn above(&self,z:f64)->f64{if reads_spread(self.groups){self.quantile(z)*self.error+self.shift(z)}else{f64::INFINITY}}
    /// Its error at z standard errors: the larger distance.
    pub fn at(&self,z:f64)->f64{self.below(z).max(self.above(z))}
}

/// Student's t at ν degrees of freedom (a whole number, at least one) at the normal's two-sided tail beyond z: the q with
/// P(|t_ν| > q) = P(|N| > z), by bisection on t's tail (`student_tail`) to the last bit; kept per thread, a pure function of both.
pub(crate) fn student(z:f64,nu:f64)->f64{
    thread_local!{static KEPT:std::cell::RefCell<std::collections::HashMap<(u64,u64),f64>>=std::cell::RefCell::new(std::collections::HashMap::new());}
    let n=nu.round().max(1.) as u64;
    KEPT.with(|kept|*kept.borrow_mut().entry((z.to_bits(),n)).or_insert_with(||{
        let p=normal_tail(z);let(mut low,mut high)=(z,2.*z);
        while student_tail(high,n)>p{low=high;high*=2.;}
        loop{let m=0.5*(low+high);if !(m>low&&m<high){break high;}if student_tail(m,n)>p{low=m;}else{high=m;}}
    }))
}
/// The normal's two-sided tail beyond z, P(|N| > z) = erfc(z/√2) = Q(½, z²/2), the regularized upper incomplete gamma at ½:
/// below ½ + 1 one less its lower part's series, above by its continued fraction (Lentz; Numerical Recipes' gser and gcf),
/// with Γ(½) = √π.
fn normal_tail(z:f64)->f64{
    let(a,x)=(0.5,0.5*z*z);if !(x>0.){return 1.;}
    let lead=(a*x.ln()-x).exp()/std::f64::consts::PI.sqrt();
    if x<a+1.{
        let(mut term,mut sum,mut b)=(1./a,1./a,a);
        loop{b+=1.;term*=x/b;sum+=term;if term.abs()<=sum.abs()*f64::EPSILON{break;}}
        1.-sum*lead
    }else{
        let tiny=f64::MIN_POSITIVE/f64::EPSILON;let(mut b,mut c)=(x+1.-a,1./tiny);let mut d=1./b;let mut h=d;let mut i=1.;
        loop{let an=-i*(i-a);b+=2.;d=an*d+b;if d.abs()<tiny{d=tiny;}c=b+an/c;if c.abs()<tiny{c=tiny;}d=1./d;let step=d*c;h*=step;if (step-1.).abs()<=f64::EPSILON{break;}i+=1.;}
        lead*h
    }
}
/// Student's t at ν degrees of freedom (whole): its two-sided tail beyond q, 1 − A(q|ν), A by the finite sums of Abramowitz and
/// Stegun 26.7.3–4 in θ = atan(q/√ν): (2/π)(θ + sin θ (cos θ + (2/3) cos³θ + … + (2·4…(ν−3))/(3·5…(ν−2)) cos^(ν−2) θ)) at odd ν
/// (θ alone at one), sin θ (1 + (1/2) cos²θ + … + (1·3…(ν−3))/(2·4…(ν−2)) cos^(ν−2) θ) at even.
fn student_tail(q:f64,nu:u64)->f64{
    let theta=(q/(nu as f64).sqrt()).atan();let(s,c)=theta.sin_cos();let c2=c*c;
    let a=if nu%2==1{
        let(mut term,mut sum,mut k)=(c,if nu>1{c}else{0.},1);
        while 2*k+1<nu{term*=c2*(2*k) as f64/(2*k+1) as f64;sum+=term;k+=1;}
        2./std::f64::consts::PI*(theta+s*sum)
    }else{
        let(mut term,mut sum,mut k)=(1.,1.,1);
        while 2*k<nu{term*=c2*(2*k-1) as f64/(2*k) as f64;sum+=term;k+=1;}
        s*sum
    };
    1.-a
}

/// The estimates at one output time.
#[derive(Clone,Debug,Default)]
pub struct Estimate{
    /// T = m Var(v)/k_B per axis (μK), its error, the mean velocity (um/us).
    pub temperature:[f64;3],
    pub temperature_error:[Spread;3],
    /// T_total = (T_x + T_y + T_z)/3, the temperature of the 3D velocity variance (μK), and its
    /// error from the same jackknife, so the axes' correlations are in it.
    pub temperature_total:f64,
    pub temperature_total_error:Spread,
    pub mean_velocity:[f64;3],
    /// n̄ per axis relative to its trap frequency, ħ⟨κ²⟩/mω − ½ -- the energy on the axis over ħω, twice its kinetic part in a harmonic well (the virial theorem), which takes no
    /// place: the same in whichever well of a standing wave the atom sits --, and its error.
    pub occupation:[f64;3],
    pub occupation_error:[Spread;3],
    pub populations:Vec<f64>,
    pub population_errors:Vec<Spread>,
    /// The errors of the rows as the app shows them, p (1 − L) the probability that the atom is in a state and held: each
    /// row's own jackknife, of its sum over every trajectory reaching the output, so that the loss's variance and
    /// covariance are in it -- (1 − L) times its error among the held atoms left them out, RETRO's dominant state at
    /// 100 us 0.0008 where its rows spread 0.0110 between disjoint runs of the solve's size (the own jackknife 0.0100).
    pub shown_errors:Vec<Spread>,
    /// The probability that the atom has been lost, and its error.
    pub loss:f64,
    pub loss_error:Spread,
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
struct Moments{temperature:[f64;3],temperature_total:f64,occupation:[f64;3],populations:[f64;STATES],shown:[f64;STATES],loss:f64}
fn moments(g:&Sums,s:&Scales,correction:Option<&Correction>)->Moments{
    let n=g.held;let all=g.trajectories;let mut e=Moments{temperature:[0.;3],temperature_total:0.,occupation:[0.;3],populations:[0.;STATES],shown:[0.;STATES],loss:1.-n/all};
    if n<=0.{return e;}
    let mut kappa2=g.kappa2;
    if let Some(c)=correction{for a in 0..3{kappa2[a]-=c.harmonic[a]*(g.harmonic[a][0]-all*c.means[a]);}}
    for a in 0..3{
        let(mean,second)=(g.kappa[a]/n,kappa2[a]/n);let variance=(second-mean*mean).max(0.);
        e.temperature[a]=variance*s.hbar_over_mass/KT_PER_UK;
        let w=s.frequencies[a];e.occupation[a]=if w>0.{second*s.hbar_over_mass/w-0.5}else{0.};
    }
    e.temperature_total=e.temperature.iter().sum::<f64>()/3.;
    for k in 0..STATES{e.populations[k]=g.populations[k]/n;e.shown[k]=g.populations[k]/all;}
    e
}
/// Whether so many independent samples read a spread: a variance has one degree of freedom fewer than its samples, so it takes two. What is too few to read
/// is this wherever the read is a spread -- a temperature over its atoms (solve.rs `reads_temperature`), a jackknife over its groups (`summarize`), that jackknife's own over the
/// groups left out (solve.rs `need_error`), a read's error over its windows (solve.rs `drift`); where it is a count, what none met, it is what the samples leave open (solve.rs `unseen`).
pub(crate) fn reads_spread(samples:f64)->bool{samples>=2.}
/// The relative error of a jackknife's variance over g groups (χ² with g − 1 degrees of freedom): √(2/(g − 1)), of its standard error half that, √(1/(2(g − 1))): 8 % on 72 groups.
pub(crate) fn variance_noise(groups:f64)->f64{(2./(groups-1.).max(1.)).sqrt()}
/// The largest of n noisy looks at one quantity, in standard errors: z = √(2 ln n), what a read that takes their largest must allow for.
pub(crate) fn largest_look(looks:f64)->f64{(2.*looks.ln()).sqrt()}
/// A value's spread over the groups left out (`Spread`), `groups` of them reading it: the jackknife's standard error and the skew;
/// unknown where fewer than two groups read it (`reads_spread`) -- a held atoms' quantity whose atoms one group holds.
fn spread(leave:&[Moments],f:impl Fn(&Moments)->f64,groups:f64)->Spread{
    if !reads_spread(groups){return Spread::unread(groups);}
    Spread{error:crate::read::jackknife_error(leave.iter().map(&f)),groups,skew:skew(leave.iter().map(&f))}
}
/// The skew of a jackknife's group values (`Spread::skew`), of the values left out: a group's value moves the estimate against
/// its own leave-one-out, so with d a value left out less their mean, −(g − 1) Σd³/(g Σd²); none without a spread.
fn skew(leave:impl ExactSizeIterator<Item=f64>+Clone)->f64{
    let n=leave.len() as f64;let mean=leave.clone().sum::<f64>()/n;
    let(two,three)=leave.fold((0.,0.),|(a,b),x|{let d=x-mean;(a+d*d,b+d*d*d)});
    if two>0.{-(n-1.)*three/(n*two)}else{0.}
}

/// The pooled estimates at every output time, with jackknife errors over the groups that reach it (a batch's
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
        // The groups that read each quantity: every one the loss and the rows as shown, those holding an atom the held atoms'.
        let(every,holding)=(count as f64,present.iter().filter(|x|x.held>0.).count() as f64);
        if !reads_spread(count as f64){let(held,all)=(Spread::unread(holding),Spread::unread(every));
            e.population_errors=vec![held;STATES];e.temperature_error=[held;3];e.temperature_total_error=held;e.occupation_error=[held;3];e.shown_errors=vec![all;STATES];e.loss_error=all;return e;}
        let leave:Vec<Moments>=present.iter().map(|x|moments(&all.combine(x,-1.),s,m)).collect();
        for a in 0..3{e.temperature_error[a]=spread(&leave,|x|x.temperature[a],holding);e.occupation_error[a]=spread(&leave,|x|x.occupation[a],holding);}
        e.temperature_total_error=spread(&leave,|x|x.temperature_total,holding);
        e.population_errors=(0..STATES).map(|k|spread(&leave,|x|x.populations[k],holding)).collect();
        e.shown_errors=(0..STATES).map(|k|spread(&leave,|x|x.shown[k],every)).collect();
        e.loss_error=spread(&leave,|x|x.loss,every);
        e
    }).collect()
}

#[derive(Clone,Debug)]
pub(crate) struct PairedOutput{
    pub temperature:[crate::read::Read;3],
    pub populations:[crate::read::Read;STATES],
    pub loss:crate::read::Read,
    pub independent:usize,
    pub held:[f64;2],
}

/// The same moments, control correction and independent groups as `summarize`.
/// Temperature differences are relative to the fine output; populations and
/// loss are absolute. No Richardson factor or acceptance decision lives here.
pub(crate) fn paired_outputs(pairs:&[[Group;2]],s:&Scales,means:[&[[f64;3]];2])->Vec<PairedOutput>{
    use crate::read::Read;
    let Some(first)=pairs.first()else{return vec![]};
    let outputs=first[0].sums.len();
    assert!(pairs.iter().all(|p|p.iter().all(|g|g.sums.len()==outputs)));
    (0..outputs).into_par_iter().map(|i|{
        let present:Vec<_>=pairs.iter().filter(|p|p[0].sums[i].trajectories>0.&&p[1].sums[i].trajectories>0.).collect();
        let mut all=[Sums::zero(),Sums::zero()];
        for pair in &present{for side in 0..2{all[side].add(&pair[side].sums[i],1.);}}
        let correction:[Option<Correction>;2]=std::array::from_fn(|side|means[side].get(i).map(|&m|Correction::new(&all[side],m)));
        let at=|sums:&[Sums;2]|->([f64;3],[f64;STATES],f64){
            let m:[Moments;2]=std::array::from_fn(|side|moments(&sums[side],s,correction[side].as_ref()));
            let temperature=std::array::from_fn(|a|{
                if sums.iter().any(|g|!reads_spread(g.held)){return f64::INFINITY;}
                let(c,f)=(m[0].temperature[a],m[1].temperature[a]);
                if f>0.{(c-f)/f}else if c==0.{0.}else{f64::INFINITY}
            });
            (temperature,std::array::from_fn(|a|m[0].populations[a]-m[1].populations[a]),m[0].loss-m[1].loss)
        };
        let value=at(&all);
        let leave:Vec<_>=present.iter().map(|p|at(&std::array::from_fn(|side|all[side].combine(&p[side].sums[i],-1.)))).collect();
        PairedOutput{temperature:std::array::from_fn(|a|Read::jackknife(value.0[a],leave.iter().map(|v|v.0[a]))),
            populations:std::array::from_fn(|a|Read::jackknife(value.1[a],leave.iter().map(|v|v.1[a]))),
            loss:Read::jackknife(value.2,leave.iter().map(|v|v.2)),independent:present.len(),held:[all[0].held,all[1].held]}
    }).collect()
}

#[cfg(test)]
mod tests{
    #[test]
    fn paired_outputs_use_the_product_survivors_and_temperature(){
        use super::*;
        let scales=Scales{hbar_over_mass:1.,frequencies:[1.;3]};
        let mut population=[0.;STATES];population[0]=1.;
        for boost in [0.,10.]{
            let make=|fine:bool|{let mut group=Group::new(1,4);
                for (i,q) in [-1.,1.,-3.,3.].into_iter().enumerate(){let q=q+boost;
                    let atom=Atom{motion:[[q,q*q];3],populations:&population,folded:0.,profile:0.,doppler:0.};
                    group.add(0,(fine||i<2).then_some(&atom),[0.;3]);}group};
            let pairs:Vec<_>=(0..4).map(|_|[make(false),make(true)]).collect();
            let read=paired_outputs(&pairs,&scales,[&[],&[]]);
            let coarse:Vec<_>=pairs.iter().map(|p|&p[0]).collect();let fine:Vec<_>=pairs.iter().map(|p|&p[1]).collect();
            let(a,b)=(summarize(&coarse,&scales,&[]),summarize(&fine,&scales,&[]));
            for axis in 0..3{assert_eq!(read[0].temperature[axis].value,(a[0].temperature[axis]-b[0].temperature[axis])/b[0].temperature[axis]);
                assert!((read[0].temperature[axis].value+0.8).abs()<1e-13);assert_eq!(read[0].temperature[axis].error,0.);}
            assert_eq!(read[0].loss.value,a[0].loss-b[0].loss);assert_eq!((read[0].loss.value,read[0].loss.error),(0.5,0.));
            assert_eq!((read[0].independent,read[0].held),(4,[8.,16.]));
        }
    }
    #[test]
    fn paired_loss_keeps_covariance_across_separate_survivors(){
        use super::*;
        let mut population=[0.;STATES];population[0]=1.;
        let atom=Atom{motion:[[0.,1.];3],populations:&population,folded:0.,profile:0.,doppler:0.};
        let pairs:Vec<_>=[[true,false],[true,true],[false,true],[false,false]].into_iter().map(|held|std::array::from_fn(|side|{
            let mut g=Group::new(1,1);g.add(0,held[side].then_some(&atom),[0.;3]);g})).collect();
        let read=paired_outputs(&pairs,&Scales{hbar_over_mass:1.,frequencies:[1.;3]},[&[],&[]]);
        assert_eq!(read[0].loss.value,0.);assert!((read[0].loss.error-(1_f64/6.).sqrt()).abs()<1e-14);
        assert!(read[0].temperature.iter().all(|r|r.error.is_infinite()),"one survivor in a leave-out group cannot certify T");
    }
    #[test]
    fn paired_outputs_hold_the_product_control_fixed_and_skip_uncovered_groups(){
        use super::*;
        let scales=Scales{hbar_over_mass:1.,frequencies:[1.;3]};
        let means=[[2.;3]];let mut population=[0.;STATES];population[0]=1.;
        let mut pairs:Vec<_>=(0..6).map(|i|std::array::from_fn(|side|{
            let mut g=Group::new(1,2);
            for sign in [-1.,1.]{let q=sign*(1.+i as f64*0.2+side as f64*(i%3) as f64*0.1);
                g.add(0,Some(&Atom{motion:[[q,q*q];3],populations:&population,folded:0.,profile:0.,doppler:0.}),[1.+i as f64*0.4;3]);}g
        })).collect();
        let read=paired_outputs(&pairs,&scales,[&means,&means]);
        let mut all=[Group::new(1,0),Group::new(1,0)];for p in &pairs{for side in 0..2{all[side].merge(&p[side]);}}
        let correction:Vec<_>=all.iter().map(|g|Correction::new(&g.sums[0],means[0])).collect();
        let difference=|g:&[Sums;2]|{let(c,f)=(estimate(&g[0],&scales,Some(&correction[0])),estimate(&g[1],&scales,Some(&correction[1])));(c.temperature[0]-f.temperature[0])/f.temperature[0]};
        let pooled=std::array::from_fn(|side|all[side].sums[0].clone());
        let expected=crate::read::Read::jackknife(difference(&pooled),pairs.iter().map(|p|difference(&std::array::from_fn(|side|pooled[side].combine(&p[side].sums[0],-1.)))));
        assert_eq!((read[0].temperature[0].value,read[0].temperature[0].error),(expected.value,expected.error));
        pairs.push([Group::reaching(1,0,1),Group::reaching(1,0,1)]);
        let skipped=paired_outputs(&pairs,&scales,[&means,&means]);
        assert_eq!(skipped[0].independent,6);assert_eq!(skipped[0].loss.value,0.);
        assert_eq!((skipped[0].temperature[0].value,skipped[0].temperature[0].error),(expected.value,expected.error));
        assert_eq!(skipped[0].populations[0].value,0.);
    }

    /// A spread reads its groups and their skew off the jackknife: a plain mean of five groups of one atom, one of them bright,
    /// 0 0 0 0 1 -- its standard error √(Σ(x − x̄)²/(g(g − 1))) = 0.2 and the third central moment over the second per group,
    /// 0.096/(0.16·5) = 0.12, positive (the bright group pulls the mean up) --; the loss read on every group, the held atoms'
    /// quantities on those holding one.
    #[test]
    fn a_spread_reads_its_groups_and_skew(){
        use super::*;
        let scales=Scales{hbar_over_mass:1.,frequencies:[1.;3]};
        let mut groups:Vec<Group>=[0.,0.,0.,0.,1.].into_iter().map(|x|{let mut g=Group::new(1,1);let mut population=[0.;STATES];population[0]=x;
            g.add(0,Some(&Atom{motion:[[x,1.+x*x];3],populations:&population,folded:0.,profile:0.,doppler:0.}),[0.;3]);g}).collect();
        let e=&summarize(&groups,&scales,&[])[0];let p=e.population_errors[0];
        assert!((p.error-0.2).abs()<1e-15&&(p.skew-0.12).abs()<1e-15,"{p:?}");
        // Its interval at 2: t at four degrees of freedom, moved up by (2·4 + 1)/6 of the skew toward the bright group.
        let(q,d)=(student(2.,4.),1.5*0.12);assert!((p.above(2.)-(q*0.2+d)).abs()<1e-14&&(p.below(2.)-(q*0.2-d)).abs()<1e-14&&p.at(2.)==p.above(2.));
        assert_eq!(Spread::EXACT.at(2.),0.);
        let mut lost=Group::new(1,1);lost.add(0,None,[0.;3]);groups.push(lost);
        let e=&summarize(&groups,&scales,&[])[0];
        assert_eq!((e.population_errors[0].groups,e.temperature_error[0].groups,e.shown_errors[0].groups,e.loss_error.groups),(5.,5.,6.,6.));
    }
    /// Student's t at the normal's tail: the tables' t at 97.5 % at z = Φ⁻¹(0.975) for ν = 1, 10, 30, its tail there the normal's to
    /// round-off; at z = 2 the closed forms of one and two degrees of freedom, tan(π(1 − p)/2) and (1 − p)√(2/(1 − (1 − p)²)),
    /// p = P(|N| > 2); the normal's own 2 as ν grows, (z³ + z)/(4ν) apart.
    #[test]
    fn student_s_t_is_the_quantile_at_the_groups_read(){
        use super::*;
        let z=1.959963984540054;assert!((normal_tail(z)-0.05).abs()<1e-16,"{}",normal_tail(z));
        for(nu,q)in [(1.,12.706205),(10.,2.228139),(30.,2.042272)]{let t=student(z,nu);assert!((t-q).abs()<5e-7&&(student_tail(t,nu as u64)-0.05).abs()<1e-15,"{nu}: {t} against {q}");}
        let p=normal_tail(2.);assert!((p-0.04550026389635842).abs()<1e-16,"{p}");
        assert!((student(2.,1.)-(std::f64::consts::FRAC_PI_2*(1.-p)).tan()).abs()<1e-10);
        assert!((student(2.,2.)-(1.-p)*(2./(1.-(1.-p)*(1.-p))).sqrt()).abs()<1e-13);
        let t=student(2.,1e4);assert!((t-2.-10./4e4).abs()<1e-6,"{t}");
    }
    /// A row as shown, p (1 − L), carries the loss's variance: every held atom in one state shows that state at 1 − L, its error
    /// the loss's own, where its error among the held atoms is none (the held share times that read nothing of it).
    #[test]
    fn a_row_as_shown_carries_the_loss_s_variance(){
        use super::*;
        let scales=Scales{hbar_over_mass:1.,frequencies:[1.;3]};let mut population=[0.;STATES];population[0]=1.;
        let groups:Vec<Group>=[true,true,false,true,false,true].into_iter().map(|held|{let mut g=Group::new(1,1);
            g.add(0,held.then_some(&Atom{motion:[[0.,1.];3],populations:&population,folded:0.,profile:0.,doppler:0.}),[0.;3]);g}).collect();
        let e=&summarize(&groups,&scales,&[])[0];
        assert_eq!(e.population_errors[0].error,0.);assert!(e.loss_error.error>0.1);
        assert!((e.shown_errors[0].error-e.loss_error.error).abs()<1e-15&&(e.shown_errors[0].skew+e.loss_error.skew).abs()<1e-15,"{:?} {:?}",e.shown_errors[0],e.loss_error);
    }

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
            let(t,e)=(with.temperature[0]*KT_PER_UK,with.temperature_error[0].error*KT_PER_UK);let(p,o)=(plain.temperature[0]*KT_PER_UK,plain.temperature_error[0].error*KT_PER_UK);
            assert!((t-1.).abs()<=4.*e,"{t} ± {e}");
            if correlated{assert!(e<0.2*o,"{e} against {o}");}
            else{assert!((t-p).abs()<=o&&e<=1.1*o);}
        }
    }
}
