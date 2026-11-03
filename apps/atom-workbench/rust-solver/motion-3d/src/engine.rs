//! Trajectories: a classical point (r, ħκ), the internal state ψ on one cluster (ground.rs),
//! unnormalized between quantum jumps (the waiting-time unraveling of the eliminated generator),
//! and the momentum's correlation with the internal state M.
//!
//! # The second-order expansion
//! The eliminated generator ℒ_r at the atom's position (ground.rs) has the Hermitian part H, the
//! decay Γ (H_eff = H − (i/2)Γ) and the jumps L_{φ,x} of each class φ of photons and photon axis x.
//! Its Wigner function in the atom's momentum, expanded to second order in the photon momenta ħk --
//! exact to that order for every single-time moment --, moves the momentum by the superoperators
//!
//! ```text
//! 𝒜_i(Y)  = ½{F_i, Y} + (i/4)[∂_iΓ, Y] − (i/2) Σ_{φ,x} (∂_iL Y L† − L Y ∂_iL†),     F_i = −∂_i H,
//! 𝓑_ij(Y) = Σ_{φ,x} ½ tr[(∂_iL†∂_jL + ∂_jL†∂_iL) Y]   (the absorbed light's second moment),
//! ```
//!
//! κ's mean by tr 𝒜_i(ρ) and its covariance by tr 𝓑_ij(ρ) plus tr 𝒜_i(W_j) + tr 𝒜_j(W_i), W_j the
//! density's first moment in κ_j about the mean: the internal state's memory of the momentum.
//!
//! # A trajectory
//! Its conditional state ρ̃ = ψψ†/‖ψ‖² follows the unraveling. κ takes the drift
//! f_i = tr 𝒜_i(ρ̃) = ⟨F_i⟩ + Σ Im⟨u|∂_i u⟩ (u = L ψ/‖ψ‖: the Ehrenfest force and the absorbed
//! photons' mean momentum, their radiation pressure and its Doppler imbalance) and Gaussian noise of
//! covariance rate tr 𝒜_i(M_j) + tr 𝒜_j(M_i) + 𝓑_ij(ρ̃). The correlation M_j -- traceless, Hermitian,
//! block-diagonal on the clusters -- follows
//!
//! ```text
//! dM_j/dt = ℒ M_j + 𝒜_j(ρ̃) − f_j ρ̃,        M_j(0) = 0,
//! ```
//!
//! the part of κ's correlation with the internal state that the drift and the noise leave out: the
//! force's own fluctuations in a superposition of ground states (the dipole force's diffusion), the
//! absorbed photon's momentum correlated with the state it leaves the atom in. Then the ensemble's
//! E[(κ_j − κ̄_j) ρ̃] + E[M_j] follows the expansion's first moment W_j exactly (E[dρ̃] = ℒρ̃ dt and
//! the noise is independent of the jumps), and the covariance of κ grows at the expansion's rate:
//! T, the populations and the momentum's distribution at every single time are exact to second
//! order. Two-time quantities -- κ's autocorrelation, its spectrum -- are not: the noise is white
//! where the force's fluctuations keep the internal state's memory.
//!
//! The emission recoil stays at the jumps: −ħk n, n from the dipole pattern. Its mean vanishes for
//! every state the jump leaves (the pattern is even in n), so it adds nothing to 𝒜, and its second
//! moment is the expansion's.
//!
//! # The covariance owed
//! The rate is not positive at every instant: the memory term subtracts where the drift's own
//! fluctuations -- a radiation pressure that lasts as long as the state does -- count more than the
//! photons gave. Each segment's increment is added to the trajectory's pending covariance S; at the
//! end of every step S's positive part S₊ is drawn as noise, κ += S₊^½ ξ with ξ three standard
//! normals of the step's own stream (rng.rs: the trajectory and the step's number, so a twin
//! trajectory of the same steps draws the same ξ), and taken out of S; a negative part waits for the
//! increments that follow. The noise drawn exceeds the rate's integral by the debt still open, a
//! bounded offset (the debt's distribution is stationary where the rate's mean is positive) that
//! the friction of a cooling scene damps away.
//!
//! # One segment of τ (flow–kick–flow)
//! 1. The trap's harmonic part along each of its principal axes, ½ m ω_i² x_i² (`Settings::axes`,
//!    `Settings::frequencies`), moves (r, κ) exactly for τ/2.
//! 2. At the midpoint (r, κ, t + τ/2): ψ's cluster's H_eff gives ψ_h = U ψ and ψ_e = U ψ_h,
//!    U = exp(−i H_eff τ/2); a step stops here when ‖ψ_e‖² falls below the jump's threshold (below).
//!    The scalar part of H (the trap's depth) is a phase and left out of U.
//! 3. Every cluster's generator and jumps at the midpoint (model.rs `Local`, `Jumps`: H_eff, ∇H,
//!    ∇Γ for ψ's cluster, L and ∂L of every class) and its U.
//! 4. At ρ̃ = ψ_hψ_h†/‖ψ_h‖²: the drift f, the source X_j = 𝒜_j(ρ̃) − f_j ρ̃ and 𝓑(ρ̃).
//! 5. M_j over τ, Strang: M ← U M U† per cluster (its coherent evolution and decay, exact); the
//!    jumps' inflow J M = Σ_{φ,x} L M L† (from each class's cluster into its target) and X by the
//!    midpoint rule, M̄ = M + τ/2 (J M + X), M ← M + τ (J M̄ + X); M ← U M U†. Second order in τ.
//!    The trace the step leaves (its third order, and the elimination's second-order decay without
//!    jumps) is taken out along ρ̃ at the segment's end, as the unraveling's normalization does.
//! 6. The covariance increment τ [tr 𝒜_i(M̄_j) + tr 𝒜_j(M̄_i) + 𝓑_ij(ρ̃)] (midpoint rule; M̄ is the
//!    correlation at τ/2 to second order) into S; κ += τ (f − F_harmonic), F_harmonic the trap's
//!    harmonic part moved by the flow.
//! 7. ψ ← ψ_e; the harmonic flow for τ/2.
//!
//! Only the anharmonic rest and the light are split, so the step is not bound to the trap period.
//! Along held lab axes (a validation's `Settings::free`) neither κ nor M moves. The step's segments
//! done (below, at its jumps), S's positive part is drawn.
//!
//! # Jumps
//! When ‖ψ‖² falls to the trajectory's threshold θ (uniform in (0, 1]), the crossing time t* is
//! found on ln‖ψ‖² (linear on the half step where it lies), the segment redone from its start to t*
//! and the jump made there; the rest of the step is taken from the state the jump left, so the
//! drift follows the internal state on either side of the jump -- with the jump inside a segment
//! the force of the state before it would act on the whole step, an error first order in dt that
//! the jump-free step doubling cannot see. The jump draws the class φ of photons of one frequency
//! -- and the cluster it lands the atom in -- with probability Σ_x ‖u_x‖², u_x = L_{φ,x} ψ; the
//! photon's direction n from the dipole pattern tr G − n·Re(G)·n, G_xy = ⟨u_x|u_y⟩, exactly
//! (rejection from the uniform sphere); its polarization ε across n from ‖Σ_x ε_x u_x‖² on a real
//! orthonormal pair; then ψ ← f = Σ_x ε_x u_x and the emission recoil κ −= k_φ n. M is continuous.
//!
//! # First order in the Doppler shift
//! The jump operators take the kinetic shift of absorption to first order, R − w R2 (model.rs),
//! good to (kv/σ)², the largest (w_l/σ_c)² over the near beams, σ_c the nearest excited pole of
//! beam l's carrier (model.rs `Model::doppler_ratio`). A step whose (kv/σ)² exceeds the solve's
//! allowance (`Settings::doppler`) takes the exact responses (model.rs `Exact`); the largest
//! first-order (kv/σ)² a trajectory used is its share of the ledger (`Trajectory::doppler`).
//!
//! # Loss
//! An atom is lost when it is no longer bound to the trap beam: its mechanical energy, ħ²κ²/2m plus
//! its potential (model.rs `Model::trap_potential`: Re⟨ψ|H_eff|ψ⟩ of its current internal state
//! less the same with the trap beam off, so U → 0 far from it), is positive at the end of a step.
//!
//! # Randomness
//! Every jump draws from its own counter-based stream (rng.rs: the run's seed, the trajectory, the
//! jump's number), and every noise draw from its own (the draw's number), so runs that differ only
//! in the time step make the same jump choices.
use crate::model::{symmetric_eigen,Exact,Jumps,Local,Model,MOST_STATES};
use crate::rng::{normal_quantile,Purpose,Rng};
use crate::Complex;

/// A cluster's block's largest size: every cluster's block of a matrix block-diagonal on the
/// clusters fits (the eight ground states split among them).
pub const SQUARE:usize=MOST_STATES*MOST_STATES;

/// A trajectory's state.
#[derive(Clone,Debug)]
pub struct Trajectory{
    pub cluster:usize,
    pub psi:[Complex;MOST_STATES],
    pub r:[f64;3],
    /// The momentum over ħ (1/um).
    pub kappa:[f64;3],
    /// The time (us).
    pub time:f64,
    pub threshold:f64,
    pub alive:bool,
    pub jumps:u64,
    /// M_j per lab axis j (1/um): κ_j's correlation with the internal state, traceless and
    /// Hermitian, block-diagonal on the clusters (cluster k's block n_k × n_k, row-major, from
    /// `Model::block(k)`).
    pub correlation:[[Complex;SQUARE];3],
    /// The covariance of κ owed (1/um²): the increments not drawn yet.
    pub pending:[[f64;3];3],
    /// The steps made: the next one's noise stream.
    pub steps:u64,
    /// The largest (kv/σ_min)² of its first-order steps.
    pub doppler:f64,
    /// The run's seed and the trajectory's number: its streams.
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
    /// A validation's prescribed path: the atom moves at this velocity (um/us) whatever its
    /// momentum, which takes the drift and the noise (the diffusion at a point, or along a line).
    pub path:Option<[f64;3]>,
    /// The covariance owed drawn as noise (a solve's); a validation may keep it pending and add its
    /// mean, the noise's expectation without the open debt.
    pub noise:bool,
}
impl Settings{
    /// The settings of a trap (model.rs `Trap`), every axis free, the loss on, the first order in
    /// the Doppler shift everywhere.
    pub fn new(dt:f64,trap:&crate::model::Trap,truncation:f64,average:bool)->Settings{
        Settings{dt,free:[true;3],axes:trap.axes,frequencies:trap.frequencies,lab:trap.lab,truncation,average,loss:true,doppler:f64::INFINITY,path:None,noise:true}
    }
}

/// The harmonic flow along one axis for τ: (x, κ) turned by ωτ in phase space (x in um, κ in 1/um,
/// hm = ħ/m); free flight when ω is 0.
fn flow(x:f64,kappa:f64,omega:f64,hm:f64,tau:f64)->(f64,f64){
    if omega==0.{return(x+hm*kappa*tau,kappa);}
    let(s,c)=(omega*tau).sin_cos();(x*c+hm*kappa/omega*s,kappa*c-omega/hm*x*s)
}
fn dot3(a:[f64;3],b:[f64;3])->f64{a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}

fn norm2(psi:&[Complex],n:usize)->f64{psi[..n].iter().map(|z|z.norm_sqr()).sum()}
/// ⟨a|b⟩ over n entries.
fn inner(a:&[Complex],b:&[Complex],n:usize)->Complex{(0..n).fold(Complex::ZERO,|s,i|s+a[i].conj()*b[i])}
/// m v for a rows × cols matrix m (row-major).
fn apply(m:&[Complex],v:&[Complex],rows:usize,cols:usize)->[Complex;MOST_STATES]{
    let mut out=[Complex::ZERO;MOST_STATES];for p in 0..rows{let mut s=Complex::ZERO;for q in 0..cols{s+=m[p*cols+q]*v[q];}out[p]=s;}out
}
/// ½(G + G†) v for an n × n matrix G re/im interleaved (`Local`'s gradients).
fn hermitian_apply(g:&[f64],v:&[Complex],n:usize)->[Complex;MOST_STATES]{
    let at=|a:usize,b:usize|Complex::new(g[2*(a*n+b)],g[2*(a*n+b)+1]);
    let mut out=[Complex::ZERO;MOST_STATES];
    for a in 0..n{let mut s=Complex::ZERO;for b in 0..n{s+=at(a,b)*v[b]+at(b,a).conj()*v[b];}out[a]=s*0.5;}
    out
}
/// Re tr(G M) for an n × n matrix G re/im interleaved and a Hermitian M (row-major): tr(½(G + G†) M).
fn real_trace(g:&[f64],m:&[Complex],n:usize)->f64{
    let mut s=0.;for a in 0..n{for b in 0..n{let k=2*(a*n+b);let x=m[b*n+a];s+=g[k]*x.re-g[k+1]*x.im;}}s
}

/// ψ ← exp(−i H τ) ψ for an n × n matrix H (row-major in an 8 × 8 array): Taylor series on the vector
/// in sub-steps of ‖H‖τ ≤ ½ (‖H‖ the largest column sum of |re| + |im|, above the 1-norm), each
/// summed through the term whose bound (‖H‖τ)^m/m! falls below its share of `truncation` (relative
/// to ‖ψ‖; the remainder past it is at most half that bound).
pub fn propagate(h:&[Complex;SQUARE],n:usize,tau:f64,psi:&mut [Complex;MOST_STATES],truncation:f64){
    macro_rules! sized{($($n:literal)*)=>{match n{$($n=>propagate_sized::<$n>(h,tau,psi,truncation),)*_=>unreachable!("a cluster has 1 to 8 states")}}}
    sized!(1 2 3 4 5 6 7 8)
}
fn propagate_sized<const N:usize>(h:&[Complex;SQUARE],tau:f64,psi:&mut [Complex;MOST_STATES],truncation:f64){
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
/// exp(−i H τ) as a matrix (row-major): `propagate` of each basis vector.
fn exponential(h:&[Complex;SQUARE],n:usize,tau:f64,truncation:f64)->[Complex;SQUARE]{
    let mut u=[Complex::ZERO;SQUARE];
    for j in 0..n{let mut v=[Complex::ZERO;MOST_STATES];v[j]=Complex::ONE;propagate(h,n,tau,&mut v,truncation);for i in 0..n{u[i*n+j]=v[i];}}
    u
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

/// The stream of a trajectory's jump number `jump` (the initial threshold is jump 0's).
pub fn jump_stream(seed:u64,index:u64,jump:u64)->Rng{Rng::new(seed,Purpose::Jump,index,jump)}

/// A cluster at a segment's midpoint: its generator, its jumps and U = exp(−i H_eff τ/2).
struct Here{local:Local,jumps:Jumps,u:[Complex;SQUARE]}

/// A segment's outcome: ψ's norm reached the threshold (its norms at the start, the middle and the
/// end, the segment undone), or the segment made: its norms, the drift's impulse on κ (less the
/// trap's harmonic part) and κ's covariance increment.
pub enum Segment{Crossing([f64;3]),Made{norms:[f64;3],impulse:[f64;3],covariance:[[f64;3];3]}}
/// How a segment runs: a step's first try (it stops at a crossing), its redo to a crossing (it
/// does not), or a step-doubling check (it neither stops nor owes its covariance: deterministic).
#[derive(Clone,Copy,PartialEq,Eq)]
enum Mode{Try,Redo,Check}

/// The step-doubling difference of one step of dt against two of dt/2 from a state, without jumps
/// and without noise (`Engine::doubling`).
#[derive(Clone,Copy,Debug)]
pub struct Doubling{
    /// |Δκ| and the step's non-harmonic impulse |kick| (1/um).
    pub momentum:[f64;2],
    /// The largest difference of the cluster's state populations, of the norm's decay
    /// |Δ ln‖ψ‖²|, and the decay |ln‖ψ‖²| itself.
    pub internal:[f64;3],
    /// The mechanical energy's difference and its change over the two half steps (rad/us):
    /// conserved by the trap's and the light's conservative forces, it changes by the light's
    /// friction alone.
    pub energy:[f64;2],
    /// The covariance increments' traces: their difference and the two half steps' (1/um²).
    pub heating:[f64;2],
}

/// The engine: a model and its settings.
pub struct Engine<'a>{pub model:&'a Model,pub settings:Settings}

impl<'a> Engine<'a>{
    /// The trap's harmonic flow for τ: along its principal axes when every lab axis is free, else
    /// along the free lab axes; a validation's path instead.
    fn flow(&self,t:&mut Trajectory,tau:f64){
        let(s,hm)=(&self.settings,self.model.hbar_over_mass);
        if let Some(v)=s.path{for a in 0..3{t.r[a]+=v[a]*tau;}return;}
        if s.free==[true;3]{
            let(mut r,mut kappa)=([0.;3],[0.;3]);
            for i in 0..3{let(x,q)=flow(dot3(s.axes[i],t.r),dot3(s.axes[i],t.kappa),s.frequencies[i],hm,tau);for a in 0..3{r[a]+=x*s.axes[i][a];kappa[a]+=q*s.axes[i][a];}}
            (t.r,t.kappa)=(r,kappa);
        }else{for a in 0..3{if s.free[a]{(t.r[a],t.kappa[a])=flow(t.r[a],t.kappa[a],s.lab[a],hm,tau);}}}
    }
    /// The harmonic part's force on κ at r (1/um per us): none on a validation's path.
    fn harmonic(&self,r:[f64;3])->[f64;3]{
        let(s,hm)=(&self.settings,self.model.hbar_over_mass);let mut f=[0.;3];
        if s.path.is_some(){return f;}
        if s.free==[true;3]{for i in 0..3{let x=dot3(s.axes[i],r)*s.frequencies[i]*s.frequencies[i]/hm;for a in 0..3{f[a]-=x*s.axes[i][a];}}}
        else{for a in 0..3{f[a]=-s.lab[a]*s.lab[a]/hm*r[a];}}
        f
    }
    /// The free lab axes.
    fn axes(&self)->([usize;3],usize){let mut out=[0;3];let mut n=0;for a in 0..3{if self.settings.free[a]{out[n]=a;n+=1;}}(out,n)}

    /// One segment of τ (module documentation): flow, kick and flow, ψ, M and the noise.
    fn segment(&self,t:&mut Trajectory,tau:f64,exact:bool,mode:Mode)->Segment{
        let(m,s)=(self.model,&self.settings);
        let start=if mode==Mode::Try{Some(t.clone())}else{None};
        self.flow(t,0.5*tau);
        let mid=t.time+0.5*tau;let point=m.point(t.r,t.kappa);
        let k=t.cluster;let n=m.dimension(k);
        let local=m.local(k,&point,mid,true);
        let n0=norm2(&t.psi,n);
        let mut half=t.psi;propagate(&local.hamiltonian,n,0.5*tau,&mut half,s.truncation);let nh=norm2(&half,n);
        let mut end=half;propagate(&local.hamiltonian,n,0.5*tau,&mut end,s.truncation);let n1=norm2(&end,n);
        if let Some(start)=start{if n1<=t.threshold{*t=start;return Segment::Crossing([n0,nh,n1]);}}
        let(axes,count)=self.axes();let axes=&axes[..count];
        let(mut impulse,mut covariance)=([0.;3],[[0.;3];3]);
        if count>0{
            let responses=if exact{Exact::new(m,t.r,t.kappa).ok()}else{None};
            let mut local=Some(local);
            let here:Vec<Here>=(0..m.clusters()).map(|c|{
                let l=if c==k{local.take().unwrap()}else{m.local(c,&point,mid,false)};
                let u=exponential(&l.hamiltonian,m.dimension(c),0.5*tau,s.truncation);
                Here{jumps:m.jumps(c,&point,mid,responses.as_ref(),true),local:l,u}}).collect();
            let mut phi=half;for z in phi[..n].iter_mut(){*z=*z*(1./nh.sqrt());}
            let(drift,source,absorption)=self.source(&here,k,&phi,axes);
            let memory=self.correlate(t,&here,&source,tau,axes);
            self.untrace(t,&end,axes);
            for &i in axes{for &j in axes{covariance[i][j]=tau*(memory[i][j]+memory[j][i]+absorption[i][j]);}}
            let harmonic=self.harmonic(t.r);
            for &a in axes{impulse[a]=tau*(drift[a]-harmonic[a]);t.kappa[a]+=impulse[a];}
            if mode!=Mode::Check{for a in 0..3{for b in 0..3{t.pending[a][b]+=covariance[a][b];}}}
        }
        t.psi=end;
        self.flow(t,0.5*tau);
        t.time+=tau;
        Segment::Made{norms:[n0,nh,n1],impulse,covariance}
    }

    /// At ρ̃ = φφ† (φ normalized, on cluster k): the drift f_i = tr 𝒜_i(ρ̃) = ⟨F_i⟩ + Σ Im⟨u|∂_i u⟩,
    /// the source X_i = 𝒜_i(ρ̃) − f_i ρ̃ on every cluster's block, and the absorbed light's second
    /// moment 𝓑_ij(ρ̃) = Σ Re⟨∂_i u|∂_j u⟩, u = L_{φ,x} φ over the classes and the photon's axes.
    fn source(&self,here:&[Here],k:usize,phi:&[Complex;MOST_STATES],axes:&[usize])->([f64;3],[[Complex;SQUARE];3],[[f64;3];3]){
        let m=self.model;let(n,at)=(m.dimension(k),m.block(k));let h=&here[k];
        let(mut drift,mut x,mut b)=([0.;3],[[Complex::ZERO;SQUARE];3],[[0.;3];3]);
        let quarter=Complex::new(0.,0.25);let half=Complex::new(0.,-0.5);
        for &i in axes{
            // ½{F_i, ρ̃} + (i/4)[∂_iΓ, ρ̃]: F_i φ = −∂_iH φ, ∂_iΓ φ.
            let f=hermitian_apply(&h.local.gradient[i],phi,n).map(|z|-z);let d=hermitian_apply(&h.local.decay_gradient[i],phi,n);
            drift[i]=inner(phi,&f,n).re;
            for p in 0..n{for q in 0..n{x[i][at+p*n+q]=(f[p]*phi[q].conj()+phi[p]*f[q].conj())*0.5+(d[p]*phi[q].conj()-phi[p]*d[q].conj())*quarter;}}
        }
        // −(i/2)(∂_iL ρ̃ L† − L ρ̃ ∂_iL†) into each class's target, its trace the photons' momentum.
        for(c,class)in h.jumps.classes.iter().enumerate(){
            let(rows,to)=(class.rows,m.block(class.target));
            for a in 0..3{
                let u=apply(h.jumps.l(c,a),phi,rows,n);
                let mut du=[[Complex::ZERO;MOST_STATES];3];for &i in axes{du[i]=apply(h.jumps.dl(c,i,a),phi,rows,n);}
                for &i in axes{
                    drift[i]+=inner(&u,&du[i],rows).im;
                    for p in 0..rows{for q in 0..rows{x[i][to+p*rows+q]+=(du[i][p]*u[q].conj()-u[p]*du[i][q].conj())*half;}}
                    for &j in axes{b[i][j]+=inner(&du[i],&du[j],rows).re;}
                }
            }
        }
        for &i in axes{for p in 0..n{for q in 0..n{x[i][at+p*n+q]-=phi[p]*phi[q].conj()*drift[i];}}}
        (drift,x,b)
    }

    /// M ← U M U† on every cluster's block.
    fn conjugate(&self,here:&[Here],mm:&mut [Complex;SQUARE]){
        for(c,h)in here.iter().enumerate(){
            let(n,at)=(self.model.dimension(c),self.model.block(c));let u=&h.u;
            let mut left=[Complex::ZERO;SQUARE];
            for p in 0..n{for q in 0..n{let mut s=Complex::ZERO;for y in 0..n{s+=u[p*n+y]*mm[at+y*n+q];}left[p*n+q]=s;}}
            for p in 0..n{for q in 0..n{let mut s=Complex::ZERO;for y in 0..n{s+=left[p*n+y]*u[q*n+y].conj();}mm[at+p*n+q]=s;}}
        }
    }
    /// out += J M = Σ_{φ,x} L M L† from each class's cluster into its target's block; with `memory`,
    /// tr 𝒜_i(M) per free axis i: the force's −Re tr(∂_iH M) and the photons' Σ Im⟨L M|∂_iL⟩.
    fn inflow(&self,here:&[Here],mm:&[Complex;SQUARE],out:&mut [Complex;SQUARE],mut memory:Option<(&mut [f64;3],&[usize])>){
        let m=self.model;
        for(c,h)in here.iter().enumerate(){
            let(n,at)=(m.dimension(c),m.block(c));let block=&mm[at..at+n*n];
            if let Some((sums,axes))=memory.as_mut(){for &i in axes.iter(){sums[i]-=real_trace(&h.local.gradient[i],block,n);}}
            for(ci,class)in h.jumps.classes.iter().enumerate(){
                let(rows,to)=(class.rows,m.block(class.target));
                for a in 0..3{
                    let l=h.jumps.l(ci,a);
                    let mut lm=[Complex::ZERO;SQUARE];
                    for p in 0..rows{for q in 0..n{let mut s=Complex::ZERO;for y in 0..n{s+=l[p*n+y]*block[y*n+q];}lm[p*n+q]=s;}}
                    for p in 0..rows{for q in 0..rows{let mut s=Complex::ZERO;for y in 0..n{s+=lm[p*n+y]*l[q*n+y].conj();}out[to+p*rows+q]+=s;}}
                    if let Some((sums,axes))=memory.as_mut(){for &i in axes.iter(){
                        let dl=h.jumps.dl(ci,i,a);sums[i]+=inner(&lm[..rows*n],dl,rows*n).im;
                    }}
                }
            }
        }
    }
    /// The correlation M_j of each free axis over τ (module documentation, step 5), and the
    /// covariance rate's memory term at its midpoint, tr 𝒜_i(M̄_j) (memory[i][j]).
    fn correlate(&self,t:&mut Trajectory,here:&[Here],source:&[[Complex;SQUARE];3],tau:f64,axes:&[usize])->[[f64;3];3]{
        let mut memory=[[0.;3];3];let size=here.iter().enumerate().map(|(c,_)|{let n=self.model.dimension(c);n*n}).sum::<usize>();
        for &j in axes{
            let mut mm=t.correlation[j];
            self.conjugate(here,&mut mm);
            let mut rate=source[j];self.inflow(here,&mm,&mut rate,None);
            let mut bar=mm;for i in 0..size{bar[i]+=rate[i]*(0.5*tau);}
            let mut rate=source[j];let mut sums=[0.;3];self.inflow(here,&bar,&mut rate,Some((&mut sums,axes)));
            for &i in axes{memory[i][j]=sums[i];}
            for i in 0..size{mm[i]+=rate[i]*tau;}
            self.conjugate(here,&mut mm);
            t.correlation[j]=mm;
        }
        memory
    }
    /// M's trace, zero for the generator's exact flow, taken out along the conditional state ρ̃ at the
    /// segment's end: what the time step's second order and the elimination's second-order decay
    /// without jumps (the van Vleck paths') leave, removed as the unraveling's normalization removes
    /// it from ψ.
    fn untrace(&self,t:&mut Trajectory,psi:&[Complex;MOST_STATES],axes:&[usize]){
        let m=self.model;let(n,at)=(m.dimension(t.cluster),m.block(t.cluster));let norm=norm2(psi,n);
        for &j in axes{
            let trace:f64=(0..m.clusters()).map(|k|{let(nk,b)=(m.dimension(k),m.block(k));(0..nk).map(|a|t.correlation[j][b+a*nk+a].re).sum::<f64>()}).sum();
            for p in 0..n{for q in 0..n{t.correlation[j][at+p*n+q]-=psi[p]*psi[q].conj()*(trace/norm);}}
        }
    }
    /// The pending covariance's positive part S₊ drawn at the end of a step (module documentation):
    /// κ += S₊^½ ξ, ξ the step's three standard normals, and S₊ taken out.
    fn realize(&self,t:&mut Trajectory){
        let(values,vectors)=symmetric_eigen(t.pending);
        if values.iter().all(|&l|!(l>0.)){return;}
        let mut rng=Rng::new(t.seed,Purpose::Noise,t.index,t.steps);let xi:[f64;3]=std::array::from_fn(|_|normal_quantile(rng.interior()));
        for(&lambda,e)in values.iter().zip(&vectors){
            if !(lambda>0.){continue;}
            let x=lambda.sqrt()*dot3(*e,xi);
            for a in 0..3{t.kappa[a]+=x*e[a];for b in 0..3{t.pending[a][b]-=lambda*e[a]*e[b];}}
        }
    }

    /// The mechanical energy of a trajectory's state (rad/us): ħ²κ²/2m and its potential, the
    /// Hermitian part of its generator's expectation (the trap's and the light's, the force's).
    pub fn energy(&self,t:&Trajectory)->f64{
        let m=self.model;let n=m.dimension(t.cluster);let local=m.local(t.cluster,&m.point(t.r,t.kappa),t.time,false);let norm=norm2(&t.psi,n);
        let mut e=Complex::ZERO;for a in 0..n{for b in 0..n{e+=t.psi[a].conj()*local.hamiltonian[a*n+b]*t.psi[b];}}
        0.5*m.hbar_over_mass*t.kappa.iter().map(|k|k*k).sum::<f64>()+local.scalar+e.re/norm
    }
    /// The step-doubling difference of one step of dt from a trajectory's state against two of dt/2,
    /// without jumps and without noise (`Doubling`).
    pub fn doubling(&self,t:&Trajectory,dt:f64)->Doubling{
        let n=self.model.dimension(t.cluster);
        let made=|s:Segment|match s{Segment::Made{norms,impulse,covariance}=>(norms,impulse,covariance),Segment::Crossing(_)=>unreachable!("a check does not stop")};
        let trace=|c:&[[f64;3];3]|c[0][0]+c[1][1]+c[2][2];
        let mut one=t.clone();let(n0,kick,c1)=made(self.segment(&mut one,dt,false,Mode::Check));
        let mut two=t.clone();let(_,_,ca)=made(self.segment(&mut two,0.5*dt,false,Mode::Check));let(n2,_,cb)=made(self.segment(&mut two,0.5*dt,false,Mode::Check));
        let dk=(0..3).map(|a|(one.kappa[a]-two.kappa[a]).powi(2)).sum::<f64>().sqrt();let impulse=kick.iter().map(|x|x*x).sum::<f64>().sqrt();
        let(a,b)=(norm2(&one.psi,n),norm2(&two.psi,n));
        let dp=(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max);
        let(e0,e1,e2)=(self.energy(t),self.energy(&one),self.energy(&two));
        let(v1,v2)=(trace(&c1),trace(&ca)+trace(&cb));
        Doubling{momentum:[dk,impulse],internal:[dp,((n0[2]/n0[0]).ln()-(n2[2]/n0[0]).ln()).abs(),(n0[2]/n0[0]).ln().abs()],energy:[e1-e2,e2-e0],heating:[(v1-v2).abs(),v2.abs()]}
    }

    /// One step of dt: segments (`segment`) to each crossing of ψ's norm with its threshold, the
    /// jump made there, the rest of the step from the state the jump left; the covariance owed drawn
    /// (`realize`). Then the loss.
    pub fn step(&self,t:&mut Trajectory){
        let(m,s)=(self.model,&self.settings);
        // The first order in the Doppler shift where it holds; the exact responses beyond.
        let doppler=m.doppler_ratio(t.kappa);let exact=doppler>s.doppler;
        if !exact{t.doppler=t.doppler.max(doppler);}
        let mut left=s.dt;
        loop{
            let Segment::Crossing([n0,nh,n1])=self.segment(t,left,exact,Mode::Try)else{break};
            // The crossing on ln‖ψ‖², linear on the half where it lies; the segment redone to it.
            let(ta,tb,la,lb)=if nh<=t.threshold{(0.,0.5*left,n0.ln(),nh.ln())}else{(0.5*left,left,nh.ln(),n1.ln())};
            let at=if lb<la{(ta+(t.threshold.ln()-la)/(lb-la)*(tb-ta)).clamp(ta,tb)}else{tb};
            if at>0.{self.segment(t,at,exact,Mode::Redo);}
            self.jump(t,exact);
            left-=at;
            if !(left>0.){break;}
        }
        if s.noise{self.realize(t);}
        t.steps+=1;
        // The loss: the mechanical energy above the trap's far level.
        if s.loss{
            let point=m.point(t.r,t.kappa);
            let kinetic=0.5*m.hbar_over_mass*t.kappa.iter().map(|k|k*k).sum::<f64>();
            if kinetic+m.trap_bound(t.cluster,&point,&t.psi)>0.&&kinetic+m.trap_potential(t.cluster,&point,&t.psi,t.time)>0.{t.alive=false;}
        }
    }

    /// A quantum jump at the trajectory's point and time with its (unnormalized) state, with the
    /// exact responses when `exact`: the class, the photon's direction and polarization, the
    /// emission recoil.
    pub fn jump(&self,t:&mut Trajectory,exact:bool){
        let m=self.model;let s=&self.settings;t.jumps+=1;let mut rng=jump_stream(t.seed,t.index,t.jumps);
        let point=m.point(t.r,t.kappa);let n=m.dimension(t.cluster);let norm=norm2(&t.psi,n).sqrt();
        let mut psi=t.psi;for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        let responses=if exact{Exact::new(m,t.r,t.kappa).ok()}else{None};
        let jumps=m.jumps(t.cluster,&point,t.time,responses.as_ref(),false);
        // Each class's amplitudes u_x = L_{φ,x} ψ; the class, and the cluster it lands the atom in.
        let amplitudes:Vec<[[Complex;MOST_STATES];3]>=jumps.classes.iter().enumerate().map(|(c,x)|std::array::from_fn(|a|apply(jumps.l(c,a),&psi,x.rows,n))).collect();
        let weights:Vec<f64>=amplitudes.iter().zip(&jumps.classes).map(|(u,x)|(0..3).map(|a|norm2(&u[a],x.rows)).sum()).collect();
        let total:f64=weights.iter().sum();
        if !(total>0.){t.threshold=rng.open_uniform();return;}
        let mut x=rng.uniform()*total;let mut pick=weights.len()-1;
        for(i,&w)in weights.iter().enumerate(){if x<w{pick=i;break;}x-=w;}
        let class=&m.ground.clusters[t.cluster].classes[pick];let nt=jumps.classes[pick].rows;let u=&amplitudes[pick];
        let gram:[[Complex;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|inner(&u[a],&u[b],nt)));
        let direction=dipole_direction(&mut rng,&gram);
        // The polarization across it.
        let pair=across(direction);
        let combine=|e:&[f64;3]|->[Complex;MOST_STATES]{let mut out=[Complex::ZERO;MOST_STATES];for i in 0..nt{for a in 0..3{out[i]+=u[a][i]*e[a];}}out};
        let amplitudes=pair.map(|e|combine(&e));let weights=amplitudes.map(|v|norm2(&v,nt));
        let chosen=if rng.uniform()*(weights[0]+weights[1])<weights[0]{0}else{1};
        let(f,size)=(amplitudes[chosen],weights[chosen]);
        // The emission recoil; the absorbed light's momentum is the drift's and the noise's.
        let k=class.wave_number;for a in 0..3{if s.free[a]{t.kappa[a]-=k*direction[a];}}
        let mut next=[Complex::ZERO;MOST_STATES];for i in 0..nt{next[i]=f[i]*(1./size.sqrt());}
        t.cluster=class.target;t.psi=next;t.threshold=rng.open_uniform();
    }
}
