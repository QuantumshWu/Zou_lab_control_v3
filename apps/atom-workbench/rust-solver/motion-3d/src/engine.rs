//! Trajectories: a classical point (r, ħκ) and the internal state ψ on one cluster (ground.rs),
//! unnormalized between quantum jumps (the waiting-time unraveling of the eliminated generator).
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
//! # Jumps
//! When ‖ψ‖² falls to the trajectory's threshold θ (uniform in (0, 1]), the crossing time t* is
//! found on ln‖ψ‖² (linear on the half step where it lies), the step redone from its start to t*
//! (flow, kick and flow of that length) and the jump made there; the rest of the step is taken from
//! the state the jump left, so the force follows the internal state on either side of the jump --
//! with the jump inside a step the force of the state before it would act on the whole step, an
//! error first order in dt that the jump-free step doubling cannot see. The jump draws the class φ
//! of photons of one frequency -- and the cluster it
//! lands the atom in -- is drawn with probability Σ_a ‖u_a‖², u_a = L_{a,φ} ψ (ground.rs); the
//! photon's direction n from the dipole pattern tr G − n·Re(G)·n, G_ab = ⟨u_a|u_b⟩, exactly
//! (rejection from the uniform sphere); its polarization ε across n from ‖Σ_a ε_a u_a‖² on a real
//! orthonormal pair; then ψ ← f = Σ_a ε_a u_a and the emission recoil κ −= k_φ n.
//!
//! # Absorption
//! The light absorbed with the photon moves the atom too: by ħ k_l of the beam l it came from,
//! drawn by its share of the photon, ‖Σ_a ε_a (u_a)_l‖² (bounded kicks, the beams' own recoil),
//! plus the difference between the exact conditional mean of this photon, Im⟨f|∇f⟩/⟨f|f⟩ (the
//! jump term's first Wigner moment), and the drawn kicks' mean, so that the jumps carry the exact
//! mean force of the absorbed light -- its radiation pressure, its Doppler imbalance, and the
//! excited states' own force while the photon is held, which the light shift alone gets wrong near
//! resonance.
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
//! The photon amplitudes take the kinetic shift of absorption to first order, R − w R2 (model.rs),
//! good to (kv/σ)², the largest (w_l/σ_c)² over the near beams, σ_c the nearest excited pole of
//! beam l's carrier (model.rs `Model::doppler_ratio`). A step whose (kv/σ)² exceeds the solve's
//! allowance (`Settings::doppler`) takes the exact response at its jumps (model.rs `Exact`); the
//! largest first-order (kv/σ)² a trajectory used is its share of the ledger (`Trajectory::doppler`).
//!
//! # Loss
//! An atom is lost when it is no longer bound to the trap beam: its mechanical energy, ħ²κ²/2m plus
//! its potential (model.rs `Model::trap_potential`: Re⟨ψ|H_eff|ψ⟩ of its current internal state
//! less the same with the trap beam off, so U → 0 far from it), is positive at the end of a step.
//!
//! # Randomness
//! Every jump draws from its own counter-based stream (rng.rs: the run's seed, the trajectory, the
//! jump's number), so runs that differ only in the time step make the same choices: the
//! step-doubling check compares them. A jump draws its class, the next threshold, the polarization
//! and the absorbed beam first and the direction's rejection loop last, so that a trajectory's twin
//! on the same streams (control.rs) keeps its thresholds and choices whatever its loop takes.
//!
//! # A held atom
//! A held twin (control.rs) does not move and its generator is static: its engine (`Engine::held`)
//! takes every cluster's generator and jump operators at its point once, steps by the half step's
//! propagators, and draws a jump's direction and polarization without the kicks.
use crate::model::{Exact,Local,Model,Point,MOST_STATES};
use crate::rng::{Purpose,Rng};
use crate::Complex;

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
pub fn propagate(h:&[Complex;MOST_STATES*MOST_STATES],n:usize,tau:f64,psi:&mut [Complex;MOST_STATES],truncation:f64){
    macro_rules! sized{($($n:literal)*)=>{match n{$($n=>propagate_sized::<$n>(h,tau,psi,truncation),)*_=>unreachable!("a cluster has 1 to 8 states")}}}
    sized!(1 2 3 4 5 6 7 8)
}
fn propagate_sized<const N:usize>(h:&[Complex;MOST_STATES*MOST_STATES],tau:f64,psi:&mut [Complex;MOST_STATES],truncation:f64){
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

/// An atom held at a point (control.rs `Twin`), its generator static there: every cluster's H_eff
/// and, per class and photon axis, its jump operator L (target × cluster, row-major), computed once.
pub struct Held{pub point:Point,pub locals:Vec<Local>,pub jumps:Vec<Vec<[Vec<Complex>;3]>>}

/// The engine: a model and its settings; for a held atom its generator and jumps there, and each
/// cluster's propagator exp(−i H dt/2) over half the settings' step.
pub struct Engine<'a>{pub model:&'a Model,pub settings:Settings,held:Option<(&'a Held,Vec<[Complex;MOST_STATES*MOST_STATES]>)>}

impl<'a> Engine<'a>{
    pub fn new(model:&'a Model,settings:Settings)->Engine<'a>{Engine{model,settings,held:None}}
    /// The engine of an atom held at a point: no axis free, no loss, the held generator and jumps;
    /// the half step's propagators by the exponentials' own series on the basis states.
    pub fn held(model:&'a Model,settings:&Settings,held:&'a Held)->Engine<'a>{
        let mut s=settings.clone();s.free=[false;3];s.loss=false;
        let half=held.locals.iter().enumerate().map(|(k,local)|{let n=model.dimension(k);let mut u=[Complex::ZERO;MOST_STATES*MOST_STATES];
            for j in 0..n{let mut e=[Complex::ZERO;MOST_STATES];e[j]=Complex::ONE;propagate(&local.hamiltonian,n,0.5*s.dt,&mut e,s.truncation);for i in 0..n{u[i*n+j]=e[i];}}u}).collect();
        Engine{model,settings:s,held:Some((held,half))}
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
    /// and the kick's momentum (the light's and the trap's anharmonic part).
    fn advance(&self,t:&mut Trajectory,dt:f64)->(f64,f64,f64,[Complex;MOST_STATES],Local,crate::model::Point,[f64;3]){
        let(m,s)=(self.model,&self.settings);
        if let Some((h,half))=&self.held{
            // Held: the internal state alone, by the generator there -- a whole step by the half
            // step's propagator, a step's part by the series.
            let n=m.dimension(t.cluster);let local=&h.locals[t.cluster];let psi0=t.psi;let n0=norm2(&t.psi,n);let mut psi=t.psi;
            let half_step=|psi:&mut [Complex;MOST_STATES]|if dt==s.dt{let u=&half[t.cluster];let v=*psi;for i in 0..n{let mut z=Complex::ZERO;for j in 0..n{z+=u[i*n+j]*v[j];}psi[i]=z;}}
                else{propagate(&local.hamiltonian,n,0.5*dt,psi,s.truncation);};
            half_step(&mut psi);let nh=norm2(&psi,n);half_step(&mut psi);let n1=norm2(&psi,n);
            t.psi=psi;t.time+=dt;return(n0,nh,n1,psi0,*local,h.point,[0.;3]);
        }
        self.flow(t,0.5*dt);
        let point=m.point(t.r,t.kappa);
        let n=m.dimension(t.cluster);let local=m.local(t.cluster,&point,t.time+0.5*dt);
        let psi0=t.psi;let n0=norm2(&t.psi,n);
        let mut psi=t.psi;propagate(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let nh=norm2(&psi,n);
        let force=m.force(t.cluster,&local,&psi);let harmonic=self.harmonic(t.r);let mut kick=[0.;3];
        for a in 0..3{if s.free[a]{kick[a]=dt*(force[a]-harmonic[a]);t.kappa[a]+=kick[a];}}
        propagate(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let n1=norm2(&psi,n);
        t.psi=psi;
        self.flow(t,0.5*dt);
        t.time+=dt;
        (n0,nh,n1,psi0,local,point,kick)
    }

    /// The mechanical energy of a trajectory's state (rad/us): ħ²κ²/2m and its potential, the
    /// Hermitian part of its generator's expectation (the trap's and the light's, the force's).
    pub fn energy(&self,t:&Trajectory)->f64{
        let m=self.model;let n=m.dimension(t.cluster);let local=m.local(t.cluster,&m.point(t.r,t.kappa),t.time);let norm=norm2(&t.psi,n);
        let mut e=Complex::ZERO;for a in 0..n{for b in 0..n{e+=t.psi[a].conj()*local.hamiltonian[a*n+b]*t.psi[b];}}
        0.5*m.hbar_over_mass*t.kappa.iter().map(|k|k*k).sum::<f64>()+local.scalar+e.re/norm
    }
    /// A trajectory's state after `count` jump-free steps of dt (the deterministic flow the step check
    /// compares): flow, kick and the internal state's no-jump evolution, no jump, no loss.
    pub fn coast(&self,t:&Trajectory,dt:f64,count:usize)->Trajectory{let mut x=t.clone();for _ in 0..count{self.advance(&mut x,dt);}x}
    /// The step-doubling error of one step of dt from a trajectory's state, without jumps: one step
    /// against two of dt/2, as (|Δκ|, |kick|) -- the momentum's difference and the step's
    /// non-harmonic impulse (1/um) --, (max |Δp|, |Δ ln‖ψ‖²|, |ln‖ψ‖²|) -- the largest difference of
    /// the cluster's state populations, of the norm's decay and the decay itself -- and (ΔE, E₂ − E₀),
    /// the mechanical energy's difference and its change over the two half steps: conserved by the
    /// trap's and the light's conservative forces, it changes by the light's friction alone.
    pub fn doubling(&self,t:&Trajectory,dt:f64)->([f64;2],[f64;3],[f64;2]){
        let n=self.model.dimension(t.cluster);
        let mut one=t.clone();let(n0,_,n1,_,_,_,kick)=self.advance(&mut one,dt);
        let mut two=t.clone();self.advance(&mut two,0.5*dt);let(_,_,n2,_,_,_,_)=self.advance(&mut two,0.5*dt);
        let dk=(0..3).map(|a|(one.kappa[a]-two.kappa[a]).powi(2)).sum::<f64>().sqrt();let impulse=kick.iter().map(|x|x*x).sum::<f64>().sqrt();
        let(a,b)=(norm2(&one.psi,n),norm2(&two.psi,n));
        let dp=(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max);
        let(e0,e1,e2)=(self.energy(t),self.energy(&one),self.energy(&two));
        ([dk,impulse],[dp,((n1/n0).ln()-(n2/n0).ln()).abs(),(n1/n0).ln().abs()],[e1-e2,e2-e0])
    }

    /// One step of dt: flow, kick and flow (`advance`). Where ‖ψ‖² crosses the threshold, the step
    /// is redone to the crossing, the jump made there and the rest of the step taken from it in the
    /// state the jump left: the force and the motion follow the internal state on either side of
    /// the jump. Then the loss.
    pub fn step(&self,t:&mut Trajectory){
        let(m,s)=(self.model,&self.settings);
        // The first order in the Doppler shift where it holds; the exact response beyond.
        let doppler=m.doppler_ratio(t.kappa);let exact=doppler>s.doppler;
        if !exact{t.doppler=t.doppler.max(doppler);}
        let mut left=s.dt;
        let point=loop{
            let start=t.clone();
            let(n0,nh,n1,_,_,point,_)=self.advance(t,left);
            if n1>t.threshold{break point;}
            // The crossing on ln‖ψ‖², linear on the half where it lies; the step redone to it.
            let(ta,tb,la,lb)=if nh<=t.threshold{(0.,0.5*left,n0.ln(),nh.ln())}else{(0.5*left,left,nh.ln(),n1.ln())};
            let at=if lb<la{(ta+(t.threshold.ln()-la)/(lb-la)*(tb-ta)).clamp(ta,tb)}else{tb};
            *t=start;if at>0.{self.advance(t,at);}
            self.jump(t,t.r,t.time,exact);
            left-=at;
            if !(left>0.){break m.point(t.r,t.kappa);}
        };
        // The loss: the mechanical energy above the trap's far level.
        if s.loss{
            let kinetic=0.5*m.hbar_over_mass*t.kappa.iter().map(|k|k*k).sum::<f64>();
            if kinetic+m.trap_bound(t.cluster,&point,&t.psi)>0.&&kinetic+m.trap_potential(t.cluster,&point,&t.psi,t.time)>0.{t.alive=false;}
        }
    }

    /// A quantum jump at r and time `time` with the trajectory's (unnormalized) state, with the exact
    /// response when `exact`.
    pub fn jump(&self,t:&mut Trajectory,r:[f64;3],time:f64,exact:bool){
        let m=self.model;let s=&self.settings;t.jumps+=1;let mut rng=jump_stream(t.seed,t.index,t.jumps);
        let n=m.dimension(t.cluster);let norm=norm2(&t.psi,n).sqrt();
        let mut psi=t.psi;for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        let classes=&m.ground.clusters[t.cluster].classes;
        // Each class's amplitudes u_x = L_{φ,x} ψ: from the photons at the point, or a held atom's
        // jump operators; the photons kept for the kicks of a moving one.
        let photons=self.held.is_none().then(||{let point=m.point(r,t.kappa);let responses=if exact{Exact::new(m,r,t.kappa).ok()}else{None};m.photons(t.cluster,&point,&psi,time,responses)});
        let held:Vec<[[Complex;MOST_STATES];3]>=self.held.as_ref().map_or(vec![],|(h,_)|h.jumps[t.cluster].iter().zip(classes).map(|(l,c)|{let nt=m.dimension(c.target);
            std::array::from_fn(|x|{let mut u=[Complex::ZERO;MOST_STATES];for i in 0..nt{for j in 0..n{u[i]+=l[x][i*n+j]*psi[j];}}u})}).collect());
        let amplitude=photons.as_ref().map_or(&held,|p|&p.amplitude);
        // The class, and the cluster it lands the atom in.
        let weights:Vec<f64>=classes.iter().zip(amplitude).map(|(c,u)|{let nt=m.dimension(c.target);(0..3).map(|a|norm2(&u[a],nt)).sum()}).collect();
        let total:f64=weights.iter().sum();
        if !(total>0.){t.threshold=rng.open_uniform();return;}
        // The draws the state and the kicks depend on first -- the class, the next threshold, the
        // polarization, the absorbed beam --, the direction's rejection loop after them: a twin
        // trajectory on the same stream (control.rs) keeps its thresholds and choices whatever its
        // loop takes.
        let(mut x,threshold,polarization,absorbed)=(rng.uniform()*total,rng.open_uniform(),rng.uniform(),rng.uniform());let mut pick=weights.len()-1;
        for(i,&w)in weights.iter().enumerate(){if x<w{pick=i;break;}x-=w;}
        let class=&classes[pick];let nt=m.dimension(class.target);let u=&amplitude[pick];
        let gram:[[Complex;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|(0..nt).fold(Complex::ZERO,|acc,i|acc+u[a][i].conj()*u[b][i])));
        let direction=dipole_direction(&mut rng,&gram);
        // The polarization across it.
        let pair=across(direction);
        let combine=|v:&[[Complex;MOST_STATES];3],e:&[f64;3]|->[Complex;MOST_STATES]{let mut out=[Complex::ZERO;MOST_STATES];for i in 0..nt{for a in 0..3{out[i]+=v[a][i]*e[a];}}out};
        let amplitudes=pair.map(|e|combine(u,&e));let weights=amplitudes.map(|v|norm2(&v,nt));
        let chosen=if polarization*(weights[0]+weights[1])<weights[0]{0}else{1};let e=pair[chosen];
        let f=amplitudes[chosen];let size=weights[chosen];
        // The recoils of a moving atom: the photon's, the absorbed beam's and the exact conditional
        // mean's correction.
        if let Some(photons)=&photons{
            let du=photons.gradient(pick);let k=class.wave_number;let mut kick=direction.map(|x|-k*x);
            let mean:[f64;3]=std::array::from_fn(|x|{let mut s=0.;for i in 0..nt{let mut g=Complex::ZERO;for a in 0..3{g+=du[a][x][i]*e[a];}s+=(f[i].conj()*g).im;}s/size});
            let shares:Vec<(usize,f64)>=photons.shares(pick).iter().map(|(l,v)|(*l,norm2(&combine(v,&e),nt))).collect();
            let sum:f64=shares.iter().map(|x|x.1).sum();
            if sum>0.{
                let mut x=absorbed*sum;let mut l=shares[shares.len()-1].0;
                for &(beam,w)in &shares{if x<w{l=beam;break;}x-=w;}
                let drawn=m.geometry[l].wave_vector();
                let average:[f64;3]=std::array::from_fn(|a|shares.iter().map(|&(beam,w)|w*m.geometry[beam].wave_vector()[a]).sum::<f64>()/sum);
                for a in 0..3{kick[a]+=drawn[a]+mean[a]-average[a];}
            }else{for a in 0..3{kick[a]+=mean[a];}}
            for a in 0..3{if s.free[a]{t.kappa[a]+=kick[a];}}
        }
        let mut next=[Complex::ZERO;MOST_STATES];for i in 0..nt{next[i]=f[i]*(1./size.sqrt());}
        t.cluster=class.target;t.psi=next;t.threshold=threshold;
    }
}
