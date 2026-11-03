//! Trajectories: a classical point (r, ħκ) and the internal state ψ on one cluster (ground.rs),
//! unnormalized between quantum jumps (the waiting-time unraveling of the cluster's generator).
//!
//! # One step (flow–kick–flow)
//! The trap's harmonic part along each of its principal axes, ½ m ω_i² x_i² (`Settings::axes`,
//! `Settings::frequencies`), is moved exactly: (r, κ) turns in phase space for dt/2 to (r_h, κ_h);
//! there the cluster's H_eff at the step's middle time gives U = exp(−i H_eff dt/2); ψ_h = U ψ;
//! κ_h += dt (F − F_harmonic) with F the Ehrenfest force −∇Re⟨H_eff⟩ (the trap's full force and
//! the light's) and F_harmonic its harmonic part; ψ ← U ψ_h; and the harmonic flow for dt/2 again.
//! Only the anharmonic rest and the light are split, so the step is not bound to the trap period.
//! The scalar part of H_herm (the trap's depth) is a phase and left out of U.
//!
//! # The exact step (a cluster with explicit excited states)
//! An explicit excited state rotates at its detuning in the frame, and its saturation, Rabi
//! oscillation and decay set the internal state's pace: the step takes it exactly instead. The flow
//! for dt/2 to (r_h, κ_h); there, the position held, every cluster the trajectory visits in the step
//! takes its generator at the step's middle time once (`Segment`), and ψ(t) = exp(−i H t) ψ is
//! propagated exactly through the whole step -- on a grid whose spacing δ has ‖H‖₁δ ≤ `NODE_PHASE`,
//! by the propagator exp(−i H δ) (its Taylor series to double precision) and the series on the vector
//! for the grid's last part --, every jump at the exact root of ‖ψ(t)‖² = θ (safeguarded Newton on
//! ln‖ψ‖², whose derivative −⟨ψ|i(H − H†)|ψ⟩/‖ψ‖² is exact), the trajectory going on from the state
//! the jump left in the cluster it landed in. The impulse is the conditional force's time integral
//! along that evolution, ∫ ⟨ψ|F|ψ⟩/⟨ψ|ψ⟩ dt (composite Simpson on the grid, its error within
//! `QUADRATURE` of the impulse: (2‖H‖·2δ)⁴/2880 bounds it), applied with the jumps' kicks and less
//! dt F_harmonic(r_h) before the flow for dt/2 again. The internal state is exact for the held
//! generator; what is left is the motion's own splitting, which the step-doubling check measures.
//! A trajectory of ground states whose jump lands it in such a cluster takes the rest of its step so.
//!
//! # Jumps
//! When ‖ψ‖² falls to the trajectory's threshold θ (uniform in (0, 1]), the crossing time t* is
//! found on ln‖ψ‖² (linear on the half step where it lies), the step redone from its start to t*
//! (flow, kick and flow of that length) and the jump made there; the rest of the step is taken from
//! the state the jump left, so the force follows the internal state on either side of the jump --
//! with the jump inside a step the force of the state before it would act on the whole step, an
//! error first order in dt that the jump-free step doubling cannot see. The jump draws the class φ
//! of photons of one frequency -- and the cluster it lands the atom in -- with probability
//! Σ_a ‖u_a‖², u_a = J_{a,φ} ψ: the folded drives' C_a R_c ψ less an explicit state's amplitude
//! C_a ψ_e (the folded amplitude A_c = −R_c with its physical sign, the jump's overall sign the
//! release's, ground.rs); the photon's direction n from the dipole pattern tr G − n·Re(G)·n,
//! G_ab = ⟨u_a|u_b⟩, exactly (rejection from the uniform sphere); its polarization ε across n from
//! ‖Σ_a ε_a u_a‖² on a real orthonormal pair; then ψ ← f = Σ_a ε_a u_a and the emission recoil
//! κ −= k_φ n.
//!
//! # Absorption
//! The light absorbed with the photon moves the atom too: by ħ k_l of the beam l it came from,
//! drawn by its share of the photon (bounded kicks, the beams' own recoil) -- for a folded part
//! ‖Σ_a ε_a (u_a)_l‖², for an explicit state's part its weight shared by the beams in proportion to
//! their drives into the explicit states --, plus the difference between the exact conditional mean
//! of this photon, Im⟨f|∇f⟩/⟨f|f⟩ (the jump term's first Wigner moment: ∇ of the folded part; an
//! explicit state's amplitude does not depend on r), and the drawn kicks' mean, so that the jumps
//! carry the exact mean force of the absorbed light. An explicit state's absorbed light acts through
//! the Ehrenfest force instead -- the stimulated force of its kept drives, −∇Re⟨ψ|V|ψ⟩ --, and its
//! draw adds the beams' spread about that mean at the jump: the mean force stays exact. The
//! momentum diffusion beyond these kicks is the hook `Diffusion`'s.
//!
//! # The transient after a jump into a cluster with explicit states
//! A jump leaves the atom in a bare ground state, while the folded drives' amplitude of the state it
//! landed in, X ψ′ (X = Σ e^{−iε̃t} A_c per coherent group of channels, model.rs `Transient`), is
//! dressed: it builds up from zero, and with probability ‖X ψ′‖² the transient really excites the
//! folded states and they emit one more photon at once (their decay damps the rest). Where the jump
//! lands in a cluster with explicit excited states, the map after it is ρ → ρ − ½{Y, ρ} + Σ_s S_s ρ S_s†,
//! S_s = P_B C_a D_g^{−1/2} X_g per group g of nearly degenerate excited states (which ring apart),
//! Σ_s S_s†S_s = Y: with probability ψ′†Yψ′ one more photon from the S_s (its class by ‖S_s ψ′‖², its
//! kicks as a jump's), else ψ′ ← (1 − ½Y)ψ′; its draws come from the jump's stream before the
//! directions' loops. In a cluster of ground states alone the transient is left out: it is of the
//! order of the fold's own error, which the fold's charge bounds as validated without it (ground.rs),
//! and the engine there is the release's, draw for draw (the GPU's kernel, gpu/mod.rs).
//!
//! # What the kicks leave out
//! The Ehrenfest force averages the dipole force over a superposition of ground states, and a kick
//! drawn per beam is not correlated with the state the jump leaves: at a fixed point the momentum
//! diffusion is a part of the second-order Wigner expansion's (0.23-0.27 of it at a lin-perp-lin
//! focus). The expansion's own trajectory scheme -- a drift tr A(rho), a momentum-internal
//! correlation M and Gaussian noise of rate 2 tr A(M) + B -- is exact on a prescribed path but not
//! for an atom moving freely in the trap, where M shears into a position correlation that acts back
//! through the force's gradient (part of the Sisyphus friction) and the variance it prescribes
//! falls as well as rises: it heats the tweezer PGC far above the GMW reference
//! (claude/motion3d-m-scheme). The trapped T of this engine stays within a few percent of the 1D
//! GMW references in the validated regime (solve.rs).
//!
//! # First order in the Doppler shift
//! The folded photon amplitudes take the kinetic shift of absorption to first order, R − w R2
//! (model.rs), good to (kv/σ)², the largest (w_l/σ_c)² over the near beams, σ_c the nearest excited
//! pole of beam l's carrier (model.rs `Model::doppler_ratio`). A step whose (kv/σ)² exceeds the
//! solve's allowance (`Settings::doppler`) takes the exact response at its jumps (model.rs `Exact`);
//! the largest first-order (kv/σ)² a trajectory used is its share of the ledger
//! (`Trajectory::doppler`). An explicit drive needs neither: the atom's motion through e^{ik_l·r} is
//! its Doppler shift.
//!
//! # Loss
//! An atom is lost when it is no longer bound to the trap beam: its mechanical energy, ħ²κ²/2m plus
//! its potential (model.rs `Model::trap_potential`: Re⟨ψ|H_eff|ψ⟩ of its current internal state
//! less the same with the trap beam off, so U → 0 far from it), is positive at the end of a step.
//!
//! # Randomness
//! Every jump draws from its own counter-based stream (rng.rs: the run's seed, the trajectory, the
//! jump's number), so runs that differ only in the time step make the same choices: the
//! step-doubling check compares them. A jump draws its class, the next threshold, the polarization,
//! the absorbed beam (and into a cluster with explicit states the transient's four) first and the
//! directions' rejection loops last, so that
//! a trajectory's twin on the same streams (control.rs) keeps its thresholds and choices whatever its
//! loops take.
//!
//! # A held atom
//! A held twin (control.rs) does not move and its generator is static: its engine (`Engine::held`)
//! takes every cluster's generator and jump operators at its point once, steps by the
//! half step's propagators, and draws a jump's direction and polarization without the kicks.
use crate::model::{Exact,Generator,Local,Model,Point,MOST_MIXED,MOST_STATES};
use crate::rng::{Purpose,Rng};
use crate::Complex;
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;

/// The grid spacing of the exact step: ‖H‖₁ δ ≤ NODE_PHASE, so that composite Simpson's error,
/// (2‖H‖·2δ)⁴/2880 of the impulse at most, is within `QUADRATURE`.
pub const QUADRATURE:f64=A_PRIORI_SHARE*A_PRIORI_SHARE*crate::solve::TEMPERATURE_TARGET;
fn node_phase()->f64{(2880.*QUADRATURE).powf(0.25)/4.}

/// A trajectory's state.
#[derive(Clone,Debug)]
pub struct Trajectory{
    pub cluster:usize,
    pub psi:[Complex;MOST_MIXED],
    pub r:[f64;3],
    /// The momentum over ħ (1/um).
    pub kappa:[f64;3],
    /// The time (us).
    pub time:f64,
    pub threshold:f64,
    pub alive:bool,
    pub jumps:u64,
    /// The largest (kv/σ_min)² of its first-order steps.
    pub doppler:f64,
    /// The run's seed and the trajectory's number: its jumps' streams.
    pub seed:u64,
    pub index:u64,
}

/// What the engine needs besides the model.
#[derive(Clone,Debug)]
pub struct Settings{
    pub dt:f64,
    /// The lab axes the atom moves along (the others stay at 0).
    pub free:[bool;3],
    /// The trap's harmonic part, moved exactly: its principal axes (unit rows) and frequencies
    /// ω_i (rad/us), when the atom moves along every axis; along the free lab axes alone, the
    /// frequencies per lab axis.
    pub axes:[[f64;3];3],
    pub frequencies:[f64;3],
    pub lab:[f64;3],
    /// The bound on each exponential's truncated Taylor remainder relative to ‖ψ‖ (`propagate`):
    /// a share of the step's own error budget in a solve, f64::EPSILON for double precision.
    pub truncation:f64,
    /// The motion's sums (T, ⟨v⟩, n̄, ⟨r²⟩) averaged over one trap period of each axis centred on
    /// each output time (lib.rs `simulate`, a solve's), or taken at the output time.
    pub average:bool,
    /// The loss by the mechanical energy (a solve's); off where a validation holds the atom.
    pub loss:bool,
    /// The largest (kv/σ_min)² the first order in the Doppler shift is kept to.
    pub doppler:f64,
}
impl Settings{
    /// The settings of a trap (model.rs `Trap`), every axis free, the loss on, the first order in
    /// the Doppler shift everywhere.
    pub fn new(dt:f64,trap:&crate::model::Trap,truncation:f64,average:bool)->Settings{
        Settings{dt,free:[true;3],axes:trap.axes,frequencies:trap.frequencies,lab:trap.lab,truncation,average,loss:true,doppler:f64::INFINITY}
    }
}

/// A momentum diffusion scheme beyond the per-beam kicks: the hook for one (none by default). It
/// may change κ at the end of each exact step's internal evolution, given the step's point and
/// length, and at each photon, given its amplitude f (the target cluster's ground states) and the
/// kick the engine has drawn.
pub trait Diffusion:Sync{
    fn step(&self,_model:&Model,_t:&mut Trajectory,_point:&Point,_tau:f64){}
    fn photon(&self,_model:&Model,_t:&Trajectory,_point:&Point,_f:&[Complex],_kick:&mut [f64;3]){}
}

/// The harmonic flow along one axis for τ: (x, κ) turned by ωτ in phase space (x in um, κ in 1/um,
/// hm = ħ/m); free flight when ω is 0.
fn flow(x:f64,kappa:f64,omega:f64,hm:f64,tau:f64)->(f64,f64){
    if omega==0.{return(x+hm*kappa*tau,kappa);}
    let(s,c)=(omega*tau).sin_cos();(x*c+hm*kappa/omega*s,kappa*c-omega/hm*x*s)
}
fn dot(a:[f64;3],b:[f64;3])->f64{a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}

fn norm2(psi:&[Complex],n:usize)->f64{psi[..n].iter().map(|z|z.norm_sqr()).sum()}

/// ψ ← exp(−i H τ) ψ for an n × n matrix H (row-major in an 8 × 8 array): Taylor series on the vector
/// in sub-steps of ‖H‖τ ≤ ½ (‖H‖ the largest column sum of |re| + |im|, above the 1-norm), each
/// summed through the term whose bound (‖H‖τ)^m/m! falls below its share of `truncation` (relative
/// to ‖ψ‖; the remainder past it is at most half that bound).
pub fn propagate(h:&[Complex;MOST_STATES*MOST_STATES],n:usize,tau:f64,psi:&mut [Complex],truncation:f64){
    macro_rules! sized{($($n:literal)*)=>{match n{$($n=>propagate_sized::<$n>(h,tau,psi,truncation),)*_=>unreachable!("a cluster of ground states has 1 to 8 states")}}}
    sized!(1 2 3 4 5 6 7 8)
}
fn propagate_sized<const N:usize>(h:&[Complex;MOST_STATES*MOST_STATES],tau:f64,psi:&mut [Complex],truncation:f64){
    let mut norm=0_f64;for b in 0..N{let mut column=0.;for a in 0..N{column+=h[a*N+b].re.abs()+h[a*N+b].im.abs();}norm=norm.max(column);}
    let x=norm*tau.abs();let parts=((x/0.5).ceil() as usize).max(1);let step=tau/parts as f64;
    // The terms for this step's y = ‖H‖τ/parts ≤ ½: through the first whose bound y^m/m! is within
    // the sub-step's share of the truncation bound.
    let y=x/parts as f64;let share=truncation/parts as f64;let mut terms=1;let mut size=y;while size>share&&terms<40{terms+=1;size*=y/terms as f64;}
    let mut v=[Complex::ZERO;N];v.copy_from_slice(&psi[..N]);
    for _ in 0..parts{
        let mut term=v;let mut sum=v;
        for k in 1..=terms{
            let scale=step/k as f64;let mut next=[Complex::ZERO;N];
            for a in 0..N{let mut s=Complex::ZERO;for b in 0..N{s+=h[a*N+b]*term[b];}next[a]=Complex::new(s.im*scale,-s.re*scale);}
            term=next;for a in 0..N{sum[a]+=term[a];}
        }
        v=sum;
    }
    psi[..N].copy_from_slice(&v);
}

/// ‖H‖₁ bounded above: the largest column sum of |re| + |im| of an n × n row-major matrix.
fn column_norm(h:&[Complex],n:usize)->f64{(0..n).map(|b|(0..n).map(|a|h[a*n+b].re.abs()+h[a*n+b].im.abs()).sum::<f64>()).fold(0.,f64::max)}
/// ψ ← exp(−i H τ) ψ on any number of states, to double precision: `propagate`'s series.
pub fn propagate_dyn(h:&[Complex],n:usize,tau:f64,psi:&mut [Complex]){
    let x=column_norm(h,n)*tau.abs();let parts=((x/0.5).ceil() as usize).max(1);let step=tau/parts as f64;
    let y=x/parts as f64;let share=f64::EPSILON/parts as f64;let mut terms=1;let mut size=y;while size>share&&terms<40{terms+=1;size*=y/terms as f64;}
    let(mut term,mut next)=([Complex::ZERO;MOST_MIXED],[Complex::ZERO;MOST_MIXED]);
    for _ in 0..parts{
        term[..n].copy_from_slice(&psi[..n]);
        for k in 1..=terms{
            let scale=step/k as f64;
            for a in 0..n{let mut s=Complex::ZERO;for b in 0..n{s+=h[a*n+b]*term[b];}next[a]=Complex::new(s.im*scale,-s.re*scale);}
            term[..n].copy_from_slice(&next[..n]);for a in 0..n{psi[a]+=term[a];}
        }
    }
}
/// exp(−i H τ) of an n × n matrix with ‖H‖₁τ ≤ 1 (row-major): its Taylor series to double precision.
fn exponential(h:&[Complex],n:usize,tau:f64)->Vec<Complex>{
    let x:Vec<Complex>=h.iter().map(|z|Complex::new(z.im*tau,-z.re*tau)).collect();
    let mut sum=vec![Complex::ZERO;n*n];for a in 0..n{sum[a*n+a]=Complex::ONE;}let mut term=sum.clone();let mut next=vec![Complex::ZERO;n*n];
    let size=column_norm(h,n)*tau.abs();let mut bound=1.;
    for k in 1..40{
        for a in 0..n{for b in 0..n{let mut s=Complex::ZERO;for c in 0..n{s+=term[a*n+c]*x[c*n+b];}next[a*n+b]=s*(1./k as f64);}}
        std::mem::swap(&mut term,&mut next);for(s,t)in sum.iter_mut().zip(&term){*s+=*t;}
        bound*=size/k as f64;if bound<=f64::EPSILON{break;}
    }
    sum
}
fn multiply(u:&[Complex],n:usize,psi:&mut [Complex]){
    let mut v=[Complex::ZERO;MOST_MIXED];v[..n].copy_from_slice(&psi[..n]);
    for a in 0..n{let mut s=Complex::ZERO;for b in 0..n{s+=u[a*n+b]*v[b];}psi[a]=s;}
}

/// One cluster's internal evolution in an exact step (module documentation): its generator at the
/// held point and the step's middle time, its decay operator i(H − H†), the grid's spacing and the
/// propagator over it.
struct Segment{generator:Generator,decay:Vec<Complex>,spacing:f64,propagator:Vec<Complex>}
impl Segment{
    fn new(generator:Generator,tau:f64)->Segment{
        let n=generator.n;let h=&generator.hamiltonian;
        let decay:Vec<Complex>=(0..n*n).map(|k|{let(a,b)=(k/n,k%n);(h[a*n+b]-h[b*n+a].conj())*Complex::I}).collect();
        let steps=((column_norm(h,n)*tau/node_phase()).ceil() as usize).max(1);let spacing=tau/steps as f64;
        let propagator=exponential(h,n,spacing);
        Segment{generator,decay,spacing,propagator}
    }
    fn n(&self)->usize{self.generator.n}
    /// The conditional force ⟨ψ|F|ψ⟩/⟨ψ|ψ⟩ and ‖ψ‖².
    fn force(&self,psi:&[Complex])->([f64;3],f64){(self.generator.force(psi),norm2(psi,self.n()))}
    /// From ψ (unnormalized), the evolution for at most `remaining` until ‖ψ‖² reaches `threshold`:
    /// the time taken, whether it reached it (ψ then at the root), and the impulse ∫ ⟨F⟩ dt.
    fn evolve(&self,psi:&mut [Complex],remaining:f64,threshold:f64)->(f64,bool,[f64;3]){
        let n=self.n();let mut impulse=[0.;3];let mut elapsed=0.;
        let(mut fa,mut na)=self.force(psi);
        while elapsed<remaining{
            // A panel of two spacings (or what is left), Simpson on its ends and middle.
            let width=(2.*self.spacing).min(remaining-elapsed);let full=width==2.*self.spacing;
            let(mut middle,mut end)=([Complex::ZERO;MOST_MIXED],[Complex::ZERO;MOST_MIXED]);middle[..n].copy_from_slice(&psi[..n]);
            if full{multiply(&self.propagator,n,&mut middle);}else{propagate_dyn(&self.generator.hamiltonian,n,0.5*width,&mut middle);}
            end[..n].copy_from_slice(&middle[..n]);
            if full{multiply(&self.propagator,n,&mut end);}else{propagate_dyn(&self.generator.hamiltonian,n,0.5*width,&mut end);}
            let(fm,nm)=self.force(&middle);let(fb,nb)=self.force(&end);
            if nb<=threshold{
                // The root in the panel: Newton on ln‖ψ‖², safeguarded by its bracket.
                let s=self.root(psi,na,nm,nb,width,threshold);
                let(mut half,mut at)=([Complex::ZERO;MOST_MIXED],[Complex::ZERO;MOST_MIXED]);half[..n].copy_from_slice(&psi[..n]);
                propagate_dyn(&self.generator.hamiltonian,n,0.5*s,&mut half);at[..n].copy_from_slice(&half[..n]);propagate_dyn(&self.generator.hamiltonian,n,0.5*s,&mut at);
                let(fh,_)=self.force(&half);let(fs,_)=self.force(&at);
                for a in 0..3{impulse[a]+=s/6.*(fa[a]+4.*fh[a]+fs[a]);}
                psi[..n].copy_from_slice(&at[..n]);return(elapsed+s,true,impulse);
            }
            for a in 0..3{impulse[a]+=width/6.*(fa[a]+4.*fm[a]+fb[a]);}
            psi[..n].copy_from_slice(&end[..n]);fa=fb;na=nb;elapsed+=width;
        }
        (remaining,false,impulse)
    }
    /// The time s in (0, width] at which ‖ψ(s)‖² = threshold, ψ(s) = exp(−iHs) ψ: Newton on
    /// ln‖ψ(s)‖² with its exact derivative −⟨ψ|Γ|ψ⟩/‖ψ‖², from linear interpolation of the logarithm
    /// on the panel's half where it lies, kept inside the bracket (bisection when it steps out).
    fn root(&self,psi:&[Complex],na:f64,nm:f64,nb:f64,width:f64,threshold:f64)->f64{
        let n=self.n();let target=threshold.ln();
        let(mut lo,mut hi)=if nm<=threshold{(0.,0.5*width)}else{(0.5*width,width)};
        let(la,lb)=if nm<=threshold{(na.ln(),nm.ln())}else{(nm.ln(),nb.ln())};
        let mut s=if lb<la{(lo+(target-la)/(lb-la)*(hi-lo)).clamp(lo,hi)}else{hi};
        for _ in 0..64{
            let mut at=[Complex::ZERO;MOST_MIXED];at[..n].copy_from_slice(&psi[..n]);propagate_dyn(&self.generator.hamiltonian,n,s,&mut at);
            let norm=norm2(&at,n);let g=norm.ln()-target;
            if g>0.{lo=s;}else{hi=s;}
            let rate=crate::model::hermitian_form(&self.decay,&at,n)/norm;
            let next=if rate>0.{s+g/rate}else{0.5*(lo+hi)};
            let next=if next>lo&&next<hi{next}else{0.5*(lo+hi)};
            if (next-s).abs()<=f64::EPSILON*width.max(s){return next;}
            s=next;
        }
        s
    }
}

/// A real orthonormal pair across the unit vector n.
fn across(n:[f64;3])->[[f64;3];2]{
    let a=if n[0].abs()<=n[1].abs()&&n[0].abs()<=n[2].abs(){[1.,0.,0.]}else if n[1].abs()<=n[2].abs(){[0.,1.,0.]}else{[0.,0.,1.]};
    let cross=|u:[f64;3],v:[f64;3]|[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
    let e1=cross(n,a);let l=(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]).sqrt();let e1=e1.map(|x|x/l);[e1,cross(n,e1)]
}

/// A photon's direction n from the dipole pattern of its amplitudes' Gram matrix G
/// (G_ab = ⟨u_a|u_b⟩): density tr G − n·Re(G)·n, by rejection from the uniform sphere (the density
/// is at most tr G and at least a third of its mean).
pub fn dipole_direction(rng:&mut Rng,gram:&[[Complex;3];3])->[f64;3]{
    let trace=gram[0][0].re+gram[1][1].re+gram[2][2].re;
    loop{
        let d=rng.direction();let mut q=0.;for a in 0..3{for b in 0..3{q+=d[a]*gram[a][b].re*d[b];}}
        if rng.uniform()*trace<=trace-q{return d;}
    }
}

/// A photon of amplitudes u_a on nt states: its direction from the dipole pattern, its polarization
/// across it (by `polarization`, uniform), the state it leaves f = Σ ε_a u_a and ‖f‖².
fn emit(rng:&mut Rng,u:&[[Complex;MOST_STATES];3],nt:usize,polarization:f64)->([f64;3],[f64;3],[Complex;MOST_STATES],f64){
    let gram:[[Complex;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|(0..nt).fold(Complex::ZERO,|acc,i|acc+u[a][i].conj()*u[b][i])));
    let direction=dipole_direction(rng,&gram);
    let pair=across(direction);
    let combine=|e:&[f64;3]|->[Complex;MOST_STATES]{let mut out=[Complex::ZERO;MOST_STATES];for i in 0..nt{for a in 0..3{out[i]+=u[a][i]*e[a];}}out};
    let amplitudes=pair.map(|e|combine(&e));let weights=amplitudes.map(|v|norm2(&v,nt));
    let chosen=if polarization*(weights[0]+weights[1])<weights[0]{0}else{1};
    (direction,pair[chosen],amplitudes[chosen],weights[chosen])
}

/// The stream of a trajectory's jump number `jump` (the initial threshold is jump 0's).
pub fn jump_stream(seed:u64,index:u64,jump:u64)->Rng{Rng::new(seed,Purpose::Jump,index,jump)}

/// An atom held at a point (control.rs `Twin`), its generator static there: every cluster's H_eff,
/// per class and photon axis its jump operator L (target × cluster, row-major), and its post-jump
/// transient, computed once.
pub struct Held{pub point:Point,pub locals:Vec<Local>,pub jumps:Vec<Vec<[Vec<Complex>;3]>>}

/// The engine: a model and its settings; for a held atom its generator and jumps there, and each
/// cluster's propagator exp(−i H dt/2) over half the settings' step; the momentum diffusion's hook.
pub struct Engine<'a>{pub model:&'a Model,pub settings:Settings,held:Option<(&'a Held,Vec<[Complex;MOST_STATES*MOST_STATES]>)>,pub diffusion:Option<&'a dyn Diffusion>}

impl<'a> Engine<'a>{
    pub fn new(model:&'a Model,settings:Settings)->Engine<'a>{Engine{model,settings,held:None,diffusion:None}}
    /// The engine of an atom held at a point: no axis free, no loss, the held generator and jumps;
    /// the half step's propagators by the exponentials' own series on the basis states. Its clusters
    /// are of ground states (a twin exists for those alone, control.rs).
    pub fn held(model:&'a Model,settings:&Settings,held:&'a Held)->Engine<'a>{
        let mut s=settings.clone();s.free=[false;3];s.loss=false;
        let half=held.locals.iter().enumerate().map(|(k,local)|{let n=model.dimension(k);let mut u=[Complex::ZERO;MOST_STATES*MOST_STATES];
            for j in 0..n{let mut e=[Complex::ZERO;MOST_STATES];e[j]=Complex::ONE;propagate(&local.hamiltonian,n,0.5*s.dt,&mut e,s.truncation);for i in 0..n{u[i*n+j]=e[i];}}u}).collect();
        Engine{model,settings:s,held:Some((held,half)),diffusion:None}
    }
    /// The trap's harmonic flow for τ: along its principal axes when every lab axis is free, else
    /// along the free lab axes.
    fn flow(&self,t:&mut Trajectory,tau:f64){
        let(s,hm)=(&self.settings,self.model.hbar_over_mass);
        if s.free==[true;3]{
            let(mut r,mut kappa)=([0.;3],[0.;3]);
            for i in 0..3{let(x,q)=flow(dot(s.axes[i],t.r),dot(s.axes[i],t.kappa),s.frequencies[i],hm,tau);for a in 0..3{r[a]+=x*s.axes[i][a];kappa[a]+=q*s.axes[i][a];}}
            (t.r,t.kappa)=(r,kappa);
        }else{for a in 0..3{if s.free[a]{(t.r[a],t.kappa[a])=flow(t.r[a],t.kappa[a],s.lab[a],hm,tau);}}}
    }
    /// The harmonic part's force on κ at r (1/um per us).
    fn harmonic(&self,r:[f64;3])->[f64;3]{
        let(s,hm)=(&self.settings,self.model.hbar_over_mass);let mut f=[0.;3];
        if s.free==[true;3]{for i in 0..3{let x=dot(s.axes[i],r)*s.frequencies[i]*s.frequencies[i]/hm;for a in 0..3{f[a]-=x*s.axes[i][a];}}}
        else{for a in 0..3{f[a]=-s.lab[a]*s.lab[a]/hm*r[a];}}
        f
    }

    /// One step's flow, kick and internal evolution of length dt without jumps: the norms of ψ at
    /// the start, the middle and the end, ψ at the start, the generator and point of the middle,
    /// and the kick's momentum (the light's and the trap's anharmonic part). A cluster of ground
    /// states (`exact` takes the others).
    fn advance(&self,t:&mut Trajectory,dt:f64)->(f64,f64,f64,Local,crate::model::Point,[f64;3]){
        let(m,s)=(self.model,&self.settings);
        if let Some((h,half))=&self.held{
            // Held: the internal state alone, by the generator there -- a whole step by the half
            // step's propagator, a step's part by the series.
            let n=m.dimension(t.cluster);let local=&h.locals[t.cluster];let n0=norm2(&t.psi,n);let mut psi=t.psi;
            let half_step=|psi:&mut [Complex;MOST_MIXED]|if dt==s.dt{let u=&half[t.cluster];let v=*psi;for i in 0..n{let mut z=Complex::ZERO;for j in 0..n{z+=u[i*n+j]*v[j];}psi[i]=z;}}
                else{propagate(&local.hamiltonian,n,0.5*dt,psi,s.truncation);};
            half_step(&mut psi);let nh=norm2(&psi,n);half_step(&mut psi);let n1=norm2(&psi,n);
            t.psi=psi;t.time+=dt;return(n0,nh,n1,*local,h.point,[0.;3]);
        }
        self.flow(t,0.5*dt);
        let point=m.point(t.r,t.kappa);
        let n=m.dimension(t.cluster);let local=m.local(t.cluster,&point,t.time+0.5*dt);
        let n0=norm2(&t.psi,n);
        let mut psi=t.psi;propagate(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let nh=norm2(&psi,n);
        let force=m.force(t.cluster,&local,&psi);let harmonic=self.harmonic(t.r);let mut kick=[0.;3];
        for a in 0..3{if s.free[a]{kick[a]=dt*(force[a]-harmonic[a]);t.kappa[a]+=kick[a];}}
        propagate(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let n1=norm2(&psi,n);
        t.psi=psi;
        self.flow(t,0.5*dt);
        t.time+=dt;
        (n0,nh,n1,local,point,kick)
    }

    /// The exact step of τ (module documentation): the flow for τ/2; at the held point every
    /// cluster's generator at the middle time once, the internal state propagated exactly through
    /// the jumps (`jumps` false: none, the step-doubling check's), the impulse along it; the kick; the
    /// flow for τ/2. The kick's momentum (the light's and the trap's anharmonic part).
    fn exact(&self,t:&mut Trajectory,tau:f64,exact:bool,jumps:bool)->[f64;3]{
        let(m,s)=(self.model,&self.settings);
        self.flow(t,0.5*tau);
        let point=m.point(t.r,t.kappa);let(start,middle)=(t.time,t.time+0.5*tau);
        let mut segments:Vec<(usize,Segment)>=vec![];let mut impulse=[0.;3];let mut elapsed=0.;
        while elapsed<tau{
            let k=t.cluster;
            if !segments.iter().any(|x|x.0==k){segments.push((k,Segment::new(m.generator(k,&point,middle),tau)));}
            let segment=&segments.iter().find(|x|x.0==k).unwrap().1;
            let(taken,jumped,part)=segment.evolve(&mut t.psi,tau-elapsed,if jumps{t.threshold}else{0.});
            for a in 0..3{impulse[a]+=part[a];}elapsed+=taken;
            if !jumped{break;}
            t.time=start+elapsed;self.jump(t,point.r,t.time,exact);
        }
        let harmonic=self.harmonic(t.r);let mut kick=[0.;3];
        for a in 0..3{if s.free[a]{kick[a]=impulse[a]-tau*harmonic[a];t.kappa[a]+=kick[a];}}
        if let Some(d)=self.diffusion{d.step(m,t,&point,tau);}
        self.flow(t,0.5*tau);
        t.time=start+tau;
        kick
    }

    /// The mechanical energy of a trajectory's state (rad/us): ħ²κ²/2m and its potential, the
    /// Hermitian part of its generator's expectation (the trap's and the light's, the force's).
    pub fn energy(&self,t:&Trajectory)->f64{
        let m=self.model;let g=m.generator(t.cluster,&m.point(t.r,t.kappa),t.time);let n=g.n;let norm=norm2(&t.psi,n);
        let mut e=Complex::ZERO;for a in 0..n{for b in 0..n{e+=t.psi[a].conj()*g.hamiltonian[a*n+b]*t.psi[b];}}
        0.5*m.hbar_over_mass*t.kappa.iter().map(|k|k*k).sum::<f64>()+g.scalar+e.re/norm
    }
    /// One jump-free step of dt from a trajectory's state, by the step its cluster takes: the norms
    /// at its start and end and its kick.
    fn free_step(&self,t:&mut Trajectory,dt:f64)->(f64,f64,[f64;3]){
        if self.held.is_none()&&self.model.mixed(t.cluster){
            let n=self.model.dimension(t.cluster);let n0=norm2(&t.psi,n);let kick=self.exact(t,dt,false,false);(n0,norm2(&t.psi,n),kick)
        }else{let(n0,_,n1,_,_,kick)=self.advance(t,dt);(n0,n1,kick)}
    }
    /// A trajectory's state after `count` jump-free steps of dt (the deterministic flow the step check
    /// compares): flow, kick and the internal state's no-jump evolution, no jump, no loss.
    pub fn coast(&self,t:&Trajectory,dt:f64,count:usize)->Trajectory{let mut x=t.clone();for _ in 0..count{self.free_step(&mut x,dt);}x}
    /// The step-doubling error of one step of dt from a trajectory's state, without jumps: one step
    /// against two of dt/2, as (|Δκ|, |kick|) -- the momentum's difference and the step's
    /// non-harmonic impulse (1/um) --, (max |Δp|, |Δ ln‖ψ‖²|, |ln‖ψ‖²|) -- the largest difference of
    /// the cluster's state populations, of the norm's decay and the decay itself -- and (ΔE, E₂ − E₀),
    /// the mechanical energy's difference and its change over the two half steps: conserved by the
    /// trap's and the light's conservative forces, it changes by the light's friction alone.
    pub fn doubling(&self,t:&Trajectory,dt:f64)->([f64;2],[f64;3],[f64;2]){
        let n=self.model.dimension(t.cluster);
        let mut one=t.clone();let(n0,n1,kick)=self.free_step(&mut one,dt);
        let mut two=t.clone();self.free_step(&mut two,0.5*dt);let(_,n2,_)=self.free_step(&mut two,0.5*dt);
        let dk=(0..3).map(|a|(one.kappa[a]-two.kappa[a]).powi(2)).sum::<f64>().sqrt();let impulse=kick.iter().map(|x|x*x).sum::<f64>().sqrt();
        let(a,b)=(norm2(&one.psi,n),norm2(&two.psi,n));
        let dp=(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max);
        let(e0,e1,e2)=(self.energy(t),self.energy(&one),self.energy(&two));
        ([dk,impulse],[dp,((n1/n0).ln()-(n2/n0).ln()).abs(),(n1/n0).ln().abs()],[e1-e2,e2-e0])
    }

    /// One step of dt: flow, kick and flow (`advance`), or the exact step in a cluster with explicit
    /// excited states. Where ‖ψ‖² crosses the threshold, the step is redone to the crossing, the jump
    /// made there and the rest of the step taken from it in the state the jump left -- exactly when
    /// it lands in a cluster with explicit states: the force and the motion follow the internal state
    /// on either side of the jump. Then the loss.
    pub fn step(&self,t:&mut Trajectory){
        let(m,s)=(self.model,&self.settings);
        // The first order in the Doppler shift where it holds; the exact response beyond.
        let doppler=m.doppler_ratio(t.kappa);let exact=doppler>s.doppler;
        if !exact{t.doppler=t.doppler.max(doppler);}
        let point=if self.held.is_none()&&m.mixed(t.cluster){self.exact(t,s.dt,exact,true);m.point(t.r,t.kappa)}else{
            let mut left=s.dt;
            loop{
                let start=t.clone();
                let(n0,nh,n1,_,point,_)=self.advance(t,left);
                if n1>t.threshold{break point;}
                // The crossing on ln‖ψ‖², linear on the half where it lies; the step redone to it.
                let(ta,tb,la,lb)=if nh<=t.threshold{(0.,0.5*left,n0.ln(),nh.ln())}else{(0.5*left,left,nh.ln(),n1.ln())};
                let at=if lb<la{(ta+(t.threshold.ln()-la)/(lb-la)*(tb-ta)).clamp(ta,tb)}else{tb};
                *t=start;if at>0.{self.advance(t,at);}
                self.jump(t,t.r,t.time,exact);
                left-=at;
                if !(left>0.){break m.point(t.r,t.kappa);}
                if self.held.is_none()&&m.mixed(t.cluster){self.exact(t,left,exact,true);break m.point(t.r,t.kappa);}
            }
        };
        // The loss: the mechanical energy above the trap's far level.
        if s.loss{
            let kinetic=0.5*m.hbar_over_mass*t.kappa.iter().map(|k|k*k).sum::<f64>();
            if kinetic+m.trap_bound(t.cluster,&point,&t.psi)>0.&&kinetic+m.trap_potential(t.cluster,&point,&t.psi,t.time)>0.{t.alive=false;}
        }
    }

    /// A quantum jump at r and time `time` with the trajectory's (unnormalized) state, with the exact
    /// response when `exact`; then the post-jump transient.
    pub fn jump(&self,t:&mut Trajectory,r:[f64;3],time:f64,exact:bool){
        let m=self.model;t.jumps+=1;let mut rng=jump_stream(t.seed,t.index,t.jumps);
        let n=m.dimension(t.cluster);let norm=norm2(&t.psi,n).sqrt();
        let mut psi=t.psi;for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        let classes=&m.ground.clusters[t.cluster].classes;
        // Each class's amplitudes u_x = J_{φ,x} ψ: from the photons at the point, or a held atom's
        // jump operators; the photons kept for the kicks of a moving one.
        let point=self.held.as_ref().map_or_else(||m.point(r,t.kappa),|(h,_)|h.point);
        let photons=self.held.is_none().then(||{let responses=if exact{Exact::new(m,r,t.kappa).ok()}else{None};m.photons(t.cluster,&point,&psi,time,responses)});
        let held:Vec<[[Complex;MOST_STATES];3]>=self.held.as_ref().map_or(vec![],|(h,_)|h.jumps[t.cluster].iter().zip(classes).map(|(l,c)|{let nt=m.ground_dimension(c.target);
            std::array::from_fn(|x|{let mut u=[Complex::ZERO;MOST_STATES];for i in 0..nt{for j in 0..n{u[i]+=l[x][i*n+j]*psi[j];}}u})}).collect());
        let amplitude=photons.as_ref().map_or(&held,|p|&p.amplitude);
        // The class, and the cluster it lands the atom in.
        let weights:Vec<f64>=classes.iter().zip(amplitude).map(|(c,u)|{let nt=m.ground_dimension(c.target);(0..3).map(|a|norm2(&u[a],nt)).sum()}).collect();
        let total:f64=weights.iter().sum();
        if !(total>0.){t.threshold=rng.open_uniform();return;}
        // The draws the state and the kicks depend on first -- the class, the next threshold, the
        // polarization, the absorbed beam, the transient's --, the directions' rejection loops after
        // them: a twin trajectory on the same stream (control.rs) keeps its thresholds and choices
        // whatever its loops take.
        let(mut x,threshold,polarization,absorbed)=(rng.uniform()*total,rng.open_uniform(),rng.uniform(),rng.uniform());
        let mut pick=weights.len()-1;
        for(i,&w)in weights.iter().enumerate(){if x<w{pick=i;break;}x-=w;}
        let class=&classes[pick];let nt=m.ground_dimension(class.target);
        let transient:Option<[f64;4]>=(self.held.is_none()&&m.mixed(class.target)).then(||std::array::from_fn(|_|rng.uniform()));
        let(direction,e,f,size)=emit(&mut rng,&amplitude[pick],nt,polarization);
        // The recoils of a moving atom: the photon's, the absorbed beam's and the exact conditional
        // mean's correction.
        if let Some(photons)=&photons{
            let shares=photons.shares(pick,&e);
            self.kick(t,&point,direction,class.wave_number,&e,&f,size,&photons.gradient(pick),&shares,absorbed);
        }
        let mut next=[Complex::ZERO;MOST_MIXED];for i in 0..nt{next[i]=f[i]*(1./size.sqrt());}
        t.cluster=class.target;t.psi=next;t.threshold=threshold;
        if let Some(draws)=transient{self.transient(t,&point,time,draws,&mut rng);}
    }

    /// A moving atom's recoils from a photon (direction n, wave number k, polarization ε, the state it
    /// leaves f and ‖f‖², the gradient of its amplitudes and the beams' shares): −k n, the absorbed
    /// beam's ħk_l drawn by its share (`absorbed`, uniform), and the exact conditional mean's
    /// difference from the drawn mean.
    fn kick(&self,t:&mut Trajectory,point:&Point,direction:[f64;3],k:f64,e:&[f64;3],f:&[Complex;MOST_STATES],size:f64,du:&[[[Complex;MOST_STATES];3];3],shares:&[(usize,f64)],absorbed:f64){
        let m=self.model;let mut kick=direction.map(|x|-k*x);
        let mean:[f64;3]=std::array::from_fn(|x|{let mut s=0.;for i in 0..MOST_STATES{let mut g=Complex::ZERO;for a in 0..3{g+=du[a][x][i]*e[a];}s+=(f[i].conj()*g).im;}s/size});
        let sum:f64=shares.iter().map(|x|x.1).sum();
        if sum>0.{
            let mut x=absorbed*sum;let mut l=shares[shares.len()-1].0;
            for &(beam,w)in shares{if x<w{l=beam;break;}x-=w;}
            let drawn=m.geometry[l].wave_vector();
            let average:[f64;3]=std::array::from_fn(|a|shares.iter().map(|&(beam,w)|w*m.geometry[beam].wave_vector()[a]).sum::<f64>()/sum);
            for a in 0..3{kick[a]+=drawn[a]+mean[a]-average[a];}
        }else{for a in 0..3{kick[a]+=mean[a];}}
        if let Some(d)=self.diffusion{d.photon(m,t,point,f,&mut kick);}
        for a in 0..3{if self.settings.free[a]{t.kappa[a]+=kick[a];}}
    }

    /// The post-jump transient (module documentation) from the normalized ground state the jump
    /// left: with probability ψ′†Yψ′ one more photon of a piece S_s (by ‖S_s ψ′‖², its kicks a
    /// jump's), else ψ′ ← (1 − ½Y)ψ′. `draws`: whether, which piece, the polarization, the beam.
    fn transient(&self,t:&mut Trajectory,point:&Point,time:f64,draws:[f64;4],rng:&mut Rng){
        let m=self.model;let k=t.cluster;
        let ops=m.transient(k,point,time);let n0=ops.n0;let psi=t.psi;
        let mut y=[Complex::ZERO;MOST_STATES];for a in 0..n0{for b in 0..n0{y[a]+=ops.y[a*n0+b]*psi[b];}}
        let p:f64=(0..n0).map(|a|(psi[a].conj()*y[a]).re).sum();
        if !(p>0.){return;}
        if draws[0]>=p{
            // No photon: ψ′ ← (1 − ½Y)ψ′.
            let mut next=t.psi;for a in 0..n0{next[a]=psi[a]-y[a]*0.5;}let norm=norm2(&next,n0).sqrt();for z in next[..n0].iter_mut(){*z=*z*(1./norm);}
            t.psi=next;return;
        }
        // One more photon: the piece by its weight.
        let amplitudes:Vec<[[Complex;MOST_STATES];3]>=ops.pieces.iter().map(|piece|{let nt=m.ground_dimension(piece.target);
            std::array::from_fn(|a|{let mut u=[Complex::ZERO;MOST_STATES];for i in 0..nt{for b in 0..n0{u[i]+=piece.ops[a][i*n0+b]*psi[b];}}u})}).collect();
        let weights:Vec<f64>=ops.pieces.iter().zip(&amplitudes).map(|(piece,u)|{let nt=m.ground_dimension(piece.target);(0..3).map(|a|norm2(&u[a],nt)).sum()}).collect();
        let total:f64=weights.iter().sum();if !(total>0.){return;}
        let mut x=draws[1]*total;let mut pick=weights.len()-1;for(i,&w)in weights.iter().enumerate(){if x<w{pick=i;break;}x-=w;}
        let piece=&ops.pieces[pick];let nt=m.ground_dimension(piece.target);
        let(direction,e,f,size)=emit(rng,&amplitudes[pick],nt,draws[2]);
        let(_,du,beams)=m.transient_photon(k,point,time,piece,&psi);
        let shares:Vec<(usize,f64)>=beams.iter().map(|(l,v)|(*l,(0..nt).map(|i|(0..3).fold(Complex::ZERO,|s,a|s+v[a][i]*e[a]).norm_sqr()).sum())).collect();
        self.kick(t,point,direction,piece.wave_number,&e,&f,size,&du,&shares,draws[3]);
        let mut next=[Complex::ZERO;MOST_MIXED];for i in 0..nt{next[i]=f[i]*(1./size.sqrt());}
        t.cluster=piece.target;t.psi=next;
    }
}
