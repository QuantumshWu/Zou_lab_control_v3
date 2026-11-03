//! Control variates: per trajectory, quantities whose means are known exactly and which follow what
//! the solve estimates, so that each output's estimate -- the trajectories' mean less β times its
//! control's excess over the known mean (stats.rs) -- keeps its expectation with a smaller spread.
//!
//! # T: the harmonic twin
//! The trajectory's initial r₀ and κ₀ carried by the trap's harmonic part alone (the engine's flow,
//! along the trap's principal axes): over each output's window (lib.rs `Windows`) its κ² along a lab
//! axis, less the same with κ₀ = 0, is κ₀ᵀ A κ₀ + 2 κ₀ᵀ B r₀ -- A and B the window's means of products
//! of the axes' cosines and sines. κ₀ is centred and drawn apart from r₀, so the mean is tr(A Σ),
//! Σ = E[κ₀κ₀ᵀ] of the initial states (each state's Maxwell momenta, its trap's axes and Wigner
//! factor), exactly, whatever r₀'s distribution. While the atom still moves in the harmonic trap the
//! control follows its window's κ² closely; the light and the trap's anharmonic part decorrelate it.
//!
//! # Populations: the held twin
//! The same trajectory -- its initial internal state, thresholds and jumps' draws (common random
//! numbers, engine.rs) -- held without motion at a point where the trap beam's intensity is the mean
//! the moving atoms see (the pilot's states): its populations' mean is the eliminated master equation
//! there (`Twin::means`): the engine's generator there, its decay its own jumps' (so that its trace
//! stays one, `Twin::new`), and the engine's jumps. While the moving
//! atom's light differs little from the point's, its jumps follow the twin's; a light pattern on the
//! scale of the motion decorrelates them at once. A twin exists when the generator is static (no
//! residual beats) and every cluster is of ground states (the exact step's clusters with explicit
//! excited states have none yet): its exponentials are then computed once, and the master equation is
//! one exponential of the superoperator per output interval.
//!
//! # The twin at fourth order
//! A jump leaves the bare state f, and at fourth order the atom goes on from its slow part: Πf/√A,
//! A = ‖Πf‖², its threshold θ/A (engine.rs `slow_start`). That unraveling is no master equation's:
//! where A < 1 the jump it makes at once with probability 1 − A draws its class by the slow part's
//! own photons, normalized by their total, and where A > 1 none comes until A‖ψ‖² falls to 1 -- the
//! waiting time's law is not linear in the state. The twin takes the post-jump map into its jumps
//! instead: K = Π L̃ for each class, Π its target cluster's at the twin's point, so that the jump term
//! is Σ K ρ K† = Σ Π (L̃ ρ L̃†) Π† -- the map ρ ↦ Π ρ Π† after every jump --, and its decay is their
//! own Σ K†K = Σ L̃†Π†ΠL̃, so that the equation stays Lindblad's with trace one. Its unraveling goes on
//! after a jump from Π f/√A, as the atom does, with the plain threshold, and starts from Π e/√A
//! (`Twin::start`); its decay differs from the atom's by Σ L̃†(Π†Π − 1)L̃, a share of order η² of the
//! jumps' rate, which in the atom's unraveling the threshold's scaling carries.
use crate::engine::{Held,Settings,Trajectory};
use crate::linear::Dense;
use crate::model::{Model,MOST_MIXED,MOST_STATES};
use crate::stats::KT_PER_UK;
use crate::{Complex,Initial,STATES};

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

/// E[κ₀κ₀ᵀ] on the settings' principal axes: each initial state's Maxwell momenta along its own trap's
/// axes with the harmonic oscillator's Wigner factor (lib.rs `initial_trajectory`), weighted by its
/// population.
fn initial_momenta(model:&Model,settings:&Settings,initial:&Initial)->[[f64;3];3]{
    let(kt,hm)=(KT_PER_UK*initial.temperature_uk,model.hbar_over_mass);let mut lab=[[0.;3];3];
    for(&(_,_,p),trap)in initial.states.iter().zip(&initial.traps){
        for i in 0..3{let half=trap.frequencies[i]/(2.*kt);let quantum=if half>0.{half/half.tanh()}else{1.};
            for x in 0..3{for y in 0..3{lab[x][y]+=p*kt/hm*quantum*trap.axes[i][x]*trap.axes[i][y];}}}
    }
    std::array::from_fn(|p|std::array::from_fn(|q|(0..3).map(|x|(0..3).map(|y|settings.axes[p][x]*lab[x][y]*settings.axes[q][y]).sum::<f64>()).sum()))
}

/// The held twin: its point, every cluster's generator and jumps there (engine.rs `Held`), its
/// populations' readout there, a linear map of the cluster's density, and at fourth order each
/// cluster's post-jump map Π there (n × n, row-major; none without the fourth order's tables).
pub struct Twin{pub held:Held,pub(crate) readout:Vec<Vec<[f64;STATES]>>,pub slow:Vec<Option<Vec<Complex>>>}
impl Twin{
    /// The twin where the trap beam's intensity is `intensity`, on the line through its focus across
    /// it; none when the generator has residual beats.
    pub fn new(model:&Model,intensity:f64)->Option<Twin>{
        if !static_generator(model)||(0..model.clusters()).any(|k|model.mixed(k)){return None;}
        let g=&model.geometry[model.trap_beam];let n=g.direction;
        let pick=if n[0].abs()<0.9{[1.,0.,0.]}else{[0.,1.,0.]};let along=pick[0]*n[0]+pick[1]*n[1]+pick[2]*n[2];
        let across:[f64;3]=std::array::from_fn(|i|pick[i]-along*n[i]);let l=(across[0]*across[0]+across[1]*across[1]+across[2]*across[2]).sqrt();
        let at=|d:f64|->[f64;3]{across.map(|x|x/l*d)};
        let(mut lo,mut hi)=(0.,g.waist?);while g.intensity(at(hi))>intensity{hi*=2.;}
        for _ in 0..60{let mid=0.5*(lo+hi);if g.intensity(at(mid))>intensity{lo=mid;}else{hi=mid;}}
        let point=model.point(at(lo),[0.;3]);
        // At fourth order each cluster's post-jump map Π (the module documentation, # The twin at
        // fourth order).
        let slow:Vec<Option<Vec<Complex>>>=(0..model.clusters()).map(|k|model.slow_part(k,&point,0.)).collect();
        // Each class's jump operators: column j the photon amplitudes of the basis state j, then the
        // target's Π: K = Π L̃.
        let jumps:Vec<Vec<[Vec<Complex>;3]>>=(0..model.clusters()).map(|k|{
            let n=model.dimension(k);let columns:Vec<Vec<[[Complex;MOST_STATES];3]>>=(0..n).map(|j|{let mut psi=[Complex::ZERO;MOST_STATES];psi[j]=Complex::ONE;
                model.photons(k,&point,&psi,0.,None).amplitude}).collect();
            model.ground.clusters[k].classes.iter().enumerate().map(|(c,class)|{let nt=model.dimension(class.target);
                std::array::from_fn(|x|{let l:Vec<Complex>=(0..nt*n).map(|e|columns[e%n][c][x][e/n]).collect();
                    match &slow[class.target]{None=>l,Some(pi)=>(0..nt*n).map(|e|{let(i,j)=(e/n,e%n);(0..nt).fold(Complex::ZERO,|z,m|z+pi[i*nt+m]*l[m*n+j])}).collect()}})}).collect()}).collect();
        // Each cluster's generator: the Hermitian part of the engine's H_eff there, its decay its own
        // jumps' Σ L†L. The engine's decay comes from the tables of the responses and its jumps from
        // the photons, which take the recoil's (and the Doppler's) first order: the two differ by ~1e-4
        // of the scattering rate, and a generator of the one with the jumps of the other leaks trace.
        // With its jumps' decay the twin's unraveling is exactly the Lindblad equation of its H and L
        // (`means`): its mean exact, its trace one, every correction's trace zero.
        let locals=(0..model.clusters()).map(|k|{
            let n=model.dimension(k);let mut local=model.local(k,&point,0.);let h=local.hamiltonian;
            for p in 0..n{for q in 0..n{
                let mut decay=Complex::ZERO;
                for(class,l)in model.ground.clusters[k].classes.iter().zip(&jumps[k]){let nt=model.dimension(class.target);for x in 0..3{for i in 0..nt{decay+=l[x][i*n+p].conj()*l[x][i*n+q];}}}
                local.hamiltonian[p*n+q]=(h[p*n+q]+h[q*n+p].conj())*0.5-decay*Complex::new(0.,0.5);
            }}
            local}).collect();
        // The readout's images of the density's basis: ψψ† for ψ = e_i, e_i + e_j and e_i + i e_j.
        let readout=(0..model.clusters()).map(|k|{
            let n=model.dimension(k);let pure=|psi:&[Complex;MOST_STATES]|->[f64;STATES]{let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();model.populations(k,&point,psi,0.).map(|x|x*norm)};
            let unit=|i:usize|{let mut v=[Complex::ZERO;MOST_STATES];v[i]=Complex::ONE;v};
            let diagonal:Vec<[f64;STATES]>=(0..n).map(|i|pure(&unit(i))).collect();
            let mut out=vec![[0.;STATES];n*n];
            for i in 0..n{out[i*n+i]=diagonal[i];for j in i+1..n{
                let(mut x,mut y)=(unit(i),unit(i));x[j]=Complex::ONE;y[j]=Complex::I;let(px,py)=(pure(&x),pure(&y));
                // ρ_ij e_ij + ρ_ji e_ji = Re ρ_ij (e_ij + e_ji) + Im ρ_ij (i e_ij − i e_ji): their images.
                out[i*n+j]=std::array::from_fn(|s|px[s]-diagonal[i][s]-diagonal[j][s]);out[j*n+i]=std::array::from_fn(|s|diagonal[i][s]+diagonal[j][s]-py[s]);
            }}
            out}).collect();
        Some(Twin{held:Held{point,locals,jumps},readout,slow})
    }
    /// A twin's start from the bare initial state of its trajectory: at fourth order its slow part at the
    /// twin's point, Π e/√A, with the plain threshold (the module documentation).
    pub fn start(&self,t:&mut Trajectory){
        let Some(pi)=&self.slow[t.cluster] else{return};let n=(pi.len() as f64).sqrt().round() as usize;
        let mut y=[Complex::ZERO;MOST_MIXED];for a in 0..n{for b in 0..n{y[a]+=pi[a*n+b]*t.psi[b];}}
        let(w,before):(f64,f64)=(y[..n].iter().map(|z|z.norm_sqr()).sum(),t.psi[..n].iter().map(|z|z.norm_sqr()).sum());
        for z in y[..n].iter_mut(){*z=*z*(before/w).sqrt();}t.psi=y;
    }
    /// A twin's populations: those of its state ψ (in cluster k) at its point.
    pub fn populations(&self,k:usize,psi:&[Complex])->[f64;STATES]{
        let n=(self.readout[k].len() as f64).sqrt().round() as usize;let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        let rho:Vec<Complex>=(0..n*n).map(|e|psi[e/n]*psi[e%n].conj()*(1./norm)).collect();self.density(k,&rho)
    }
    /// The populations of a cluster's density ρ (n × n, row-major) at the twin's point.
    fn density(&self,k:usize,rho:&[Complex])->[f64;STATES]{
        let m=&self.readout[k];let n=(m.len() as f64).sqrt().round() as usize;let mut out=[0.;STATES];
        for i in 0..n{for s in 0..STATES{out[s]+=rho[i*n+i].re*m[i*n+i][s];}
            for j in i+1..n{let z=rho[i*n+j];for s in 0..STATES{out[s]+=z.re*m[i*n+j][s]+z.im*m[j*n+i][s];}}}
        out
    }
    /// The populations' exact means at `outputs` outputs `interval` apart: the eliminated master
    /// equation at the twin's point -- H_eff and the jumps of the engine's tables there -- from the
    /// initial density's ground populations (at fourth order each basis state's slow part Π e/√A, as
    /// the twins start), by the exponential of its superoperator over an interval.
    pub fn means(&self,model:&Model,initial_density:&[Complex],interval:f64,outputs:usize)->Vec<[f64;STATES]>{
        let g=&model.ground;let sizes:Vec<usize>=(0..model.clusters()).map(|k|model.dimension(k)).collect();
        let offsets:Vec<usize>=sizes.iter().scan(0,|o,&n|{let x=*o;*o+=n*n;Some(x)}).collect();let d:usize=sizes.iter().map(|n|n*n).sum();
        let mut generator=Dense::zeros(d,d);
        for(k,&n)in sizes.iter().enumerate(){
            let h=&self.held.locals[k].hamiltonian;let o=offsets[k];
            // −i(Hρ − ρH†).
            for i in 0..n{for j in 0..n{for m in 0..n{
                generator[(o+i*n+j,o+m*n+j)]+=Complex::new(0.,-1.)*h[i*n+m];
                generator[(o+i*n+j,o+i*n+m)]+=Complex::new(0.,1.)*h[j*n+m].conj();
            }}}
            // Σ L ρ L† into each class's target cluster.
            for(class,c)in g.clusters[k].classes.iter().enumerate(){
                let(nt,ot)=(sizes[c.target],offsets[c.target]);
                for a in 0..3{let l=&self.held.jumps[k][class][a];for i in 0..nt{for j in 0..nt{for p in 0..n{for q in 0..n{
                    generator[(ot+i*nt+j,o+p*n+q)]+=l[i*n+p]*l[j*n+q].conj();
                }}}}}
            }
        }
        let propagator=exponential(&generator,interval);
        let mut rho=vec![Complex::ZERO;d];
        for(k,c)in g.clusters.iter().enumerate(){let n=sizes[k];for(x,&b)in c.states.iter().enumerate(){let s=g.ground_states[b];let p=initial_density[s*STATES+s];
            match &self.slow[k]{
                None=>rho[offsets[k]+x*n+x]=p,
                Some(pi)=>{let a:f64=(0..n).map(|i|pi[i*n+x].norm_sqr()).sum();for i in 0..n{for j in 0..n{rho[offsets[k]+i*n+j]+=p*pi[i*n+x]*pi[j*n+x].conj()*(1./a);}}}
            }}}
        let read=|rho:&[Complex]|->[f64;STATES]{let mut p=[0.;STATES];for k in 0..sizes.len(){let x=self.density(k,&rho[offsets[k]..offsets[k]+sizes[k]*sizes[k]]);for s in 0..STATES{p[s]+=x[s];}}p};
        let mut out=vec![read(&rho)];
        for _ in 1..outputs{rho=(0..d).map(|i|(0..d).fold(Complex::ZERO,|z,j|z+propagator[(i,j)]*rho[j])).collect();out.push(read(&rho));}
        out
    }
}

/// Whether the generator is static: no residual beat in the kept couplings, the paths, the photons or
/// the readout's coherent sums (the frames make the strongest couplings static, ground.rs).
pub fn static_generator(model:&Model)->bool{
    model.ground.clusters.iter().all(|c|c.groups.iter().all(|x|x.entries.iter().all(|e|e.2==0.))&&c.paths.iter().all(|p|p.residual==0.)
        &&c.classes.iter().all(|k|k.photons.iter().all(|p|p.1==0.))&&c.coherent.iter().all(|g|g.iter().all(|x|x.1==0.)))
}

/// exp(M τ) by scaling and squaring of its Taylor series, summed to double precision.
fn exponential(m:&Dense,tau:f64)->Dense{
    let n=m.rows;let norm=(0..n).map(|j|(0..n).map(|i|m[(i,j)].abs()).sum::<f64>()).fold(0.,f64::max)*tau.abs();
    let squarings=if norm>0.5{(norm/0.5).log2().ceil() as u32}else{0};let scale=tau/2f64.powi(squarings as i32);
    let a=m.scaled(Complex::new(scale,0.));let mut sum=Dense::from_fn(n,n,|i,j|if i==j{Complex::ONE}else{Complex::ZERO});let mut term=sum.clone();
    for k in 1..=30{term=term.product(&a).scaled(Complex::new(1./k as f64,0.));sum.add_scaled(Complex::ONE,&term);if term.largest()<=f64::EPSILON*sum.largest(){break;}}
    for _ in 0..squarings{sum=sum.product(&sum);}
    sum
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
