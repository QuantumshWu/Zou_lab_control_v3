//! The harmonic control of T: per trajectory a quantity whose mean is known exactly and which follows what
//! the solve estimates, so that each output's κ² -- the trajectories' mean less β times its
//! control's excess over the known mean (stats.rs) -- keeps its expectation with a smaller spread.
//!
//! The trajectory's initial r₀ (about its own site in a standing wave, engine.rs `Settings::about_site`) and κ₀ carried by the trap's harmonic part alone (the engine's flow,
//! along the trap's principal axes): over each output's window (lib.rs `Windows`) its κ² along a lab
//! axis, less the same with κ₀ = 0, is κ₀ᵀ A κ₀ + 2 κ₀ᵀ B r₀ -- A and B the window's means of products
//! of the axes' cosines and sines. κ₀ is centred and symmetric at each r₀ (the bound ensemble's
//! momenta are isotropic wherever the atom starts, ensemble.rs), so the mean is tr(A Σ), Σ = E[κ₀κ₀ᵀ]
//! of the initial states (each state's ensemble's kinetic temperature, its trap's axes and Wigner
//! factor), exactly, whatever r₀'s distribution. While the atom still moves in the harmonic trap the
//! control follows its window's κ² closely; the light and the trap's anharmonic part decorrelate it.
//!
use crate::engine::Settings;
use crate::model::Model;
use crate::stats::KT_PER_UK;
use crate::Initial;

/// The harmonic twin's control: per output and lab axis A and B (on the principal axes of the
/// settings' trap) and its exact mean.
pub struct Harmonic{axes:[[f64;3];3],pub(crate) a:Vec<[[[f64;3];3];3]>,pub(crate) b:Vec<[[[f64;3];3];3]>,pub mean:Vec<[f64;3]>}
impl Harmonic{
    /// The control of a run of `steps` steps of the settings' dt per output, `outputs` outputs, with
    /// the settings' T windows (lib.rs `Windows`: `half` steps per lab axis, cut at t = 0). Each
    /// window's means of cos_p cos_q and cos_p sin_q over its steps in closed form (`window`): O(1)
    /// time and memory per output whatever the step.
    pub fn new(model:&Model,settings:&Settings,initial:&Initial,steps:usize,outputs:usize)->Harmonic{
        let(hm,dt,w)=(model.hbar_over_mass,settings.dt,settings.frequencies);
        let half=crate::half_windows(settings);
        let sigma=initial_momenta(model,settings,initial);
        let(mut a,mut b,mut mean)=(vec![[[[0.;3];3];3];outputs],vec![[[[0.;3];3];3];outputs],vec![[0.;3];outputs]);
        for i in 0..outputs{
            let at=i*steps;
            for x in 0..3{
                let h=half[x].min(at);let(lo,hi)=(at-h,at+h);let n=(hi-lo+1) as f64;let u:[f64;3]=std::array::from_fn(|p|settings.axes[p][x]);
                for p in 0..3{for q in 0..3{
                    // cos_p cos_q = (cos(ω_p − ω_q)t + cos(ω_p + ω_q)t)/2, cos_p sin_q = (sin(ω_q + ω_p)t + sin(ω_q − ω_p)t)/2.
                    let(difference,sum)=(window((w[p]-w[q])*dt,lo,hi),window((w[p]+w[q])*dt,lo,hi));
                    let(cc,cs)=(0.5*(difference.0+sum.0)/n,0.5*(sum.1-difference.1)/n);
                    a[i][x][p][q]=u[p]*u[q]*cc;b[i][x][p][q]=-u[p]*u[q]*w[q]/hm*cs;mean[i][x]+=a[i][x][p][q]*sigma[p][q];
                }}
            }
        }
        Harmonic{axes:settings.axes,a,b,mean}
    }
    /// The control at output i of a trajectory that started at r₀ with κ₀ (lab axes), per lab axis.
    pub fn value(&self,i:usize,r0:[f64;3],k0:[f64;3])->[f64;3]{
        let on=|v:[f64;3]|->[f64;3]{std::array::from_fn(|p|(0..3).map(|x|self.axes[p][x]*v[x]).sum())};let(k,r)=(on(k0),on(r0));
        std::array::from_fn(|x|{let(a,b)=(&self.a[i][x],&self.b[i][x]);let mut c=0.;
            for p in 0..3{for q in 0..3{c+=k[p]*(a[p][q]*k[q]+2.*b[p][q]*r[q]);}}c})
    }
}

/// Σ_{k=lo}^{hi} (cos θk, sin θk): e^{iθ(lo+hi)/2} D(θ) with the Dirichlet kernel D(θ) =
/// sin(nθ/2)/sin(θ/2), n = hi − lo + 1 terms. θ/2 is taken as mπ + δ, |δ| ≤ π/2, where the kernel
/// is (−1)^{m(n−1)} sin(nδ)/sin δ -- n at δ = 0 --, so it keeps its relative accuracy near every
/// multiple of 2π that θ comes to.
fn window(theta:f64,lo:usize,hi:usize)->(f64,f64){
    let n=hi-lo+1;let m=(0.5*theta/std::f64::consts::PI).round();let delta=0.5*theta-m*std::f64::consts::PI;
    let kernel=if delta==0.{n as f64}else{(n as f64*delta).sin()/delta.sin()};
    let sign=if (m as i64).rem_euclid(2)==1&&(n-1)%2==1{-1.}else{1.};
    let(s,c)=(0.5*theta*(lo+hi) as f64).sin_cos();
    (sign*kernel*c,sign*kernel*s)
}

/// E[κ₀κ₀ᵀ] on the settings' principal axes: each initial state's bound ensemble's momenta, isotropic
/// at its kinetic temperature (ensemble.rs `Ensemble::kinetic`), along its own trap's axes with the
/// harmonic oscillator's Wigner factor (lib.rs `initial_trajectory`), weighted by its population; a
/// state the trap does not hold at rest. (A draw the Wigner factors leave unbound by the loss's measure,
/// drawn again, moves it by the square of their excess over one, below 10⁻⁶ of it.)
fn initial_momenta(model:&Model,settings:&Settings,initial:&Initial)->[[f64;3];3]{
    let(kt,hm)=(KT_PER_UK*initial.temperature_uk,model.hbar_over_mass);let mut lab=[[0.;3];3];
    for((&(_,_,p),trap),e)in initial.states.iter().zip(&initial.traps).zip(&initial.ensembles){
        let Some(e)=e else{continue};
        for i in 0..3{let half=trap.frequencies[i]/(2.*kt);let quantum=if half>0.{half/half.tanh()}else{1.};
            for x in 0..3{for y in 0..3{lab[x][y]+=p*e.kinetic*kt/hm*quantum*trap.axes[i][x]*trap.axes[i][y];}}}
    }
    std::array::from_fn(|p|std::array::from_fn(|q|(0..3).map(|x|(0..3).map(|y|settings.axes[p][x]*lab[x][y]*settings.axes[q][y]).sum::<f64>()).sum()))
}

#[cfg(test)]
mod tests{
    use super::window;
    use crate::rng::{Purpose,Rng};
    /// The closed-form window sums against the sums over the window's steps: angles at random and
    /// next to multiples of 2π (where the kernel's sines vanish), windows at the start and far out.
    #[test]
    fn the_window_sums_are_the_sums_over_the_window_s_steps(){
        let mut rng=Rng::new(5,Purpose::Initial,0,0);let tau=std::f64::consts::TAU;
        let mut angles:Vec<f64>=(0..40).map(|_|8.*(rng.uniform()-0.5)).collect();
        for m in [-2.,-1.,0.,1.,3.]{for offset in [0.,1e-13,-3e-9,2e-5]{angles.push(m*tau+offset);}}
        for &theta in &angles{for(lo,hi)in [(0,0),(0,1),(0,2),(3,3),(7,20),(0,4097),(123_457,140_000),(1_000_000,1_000_513)]{
            let(mut c,mut s)=(0.,0.);for k in lo..=hi{let(y,x)=(theta*k as f64).sin_cos();c+=x;s+=y;}
            let(wc,ws)=window(theta,lo,hi);let scale=(hi-lo+1) as f64*(1.+(theta*hi as f64).abs())*1e-14;
            assert!((wc-c).abs()<=scale&&(ws-s).abs()<=scale,"θ {theta:e}, {lo}..={hi}: ({wc}, {ws}) against ({c}, {s})");
        }}
    }
}
