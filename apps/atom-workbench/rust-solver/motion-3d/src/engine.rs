//! Trajectories: a classical point (r, ħκ) and the internal state ψ of one hyperfine block,
//! unnormalized between quantum jumps (the waiting-time unraveling of the eliminated generator).
//!
//! # One step (flow–kick–flow)
//! The trap's harmonic part along each axis, ½ m ω_a² x_a² (`Settings::frequencies`), is moved
//! exactly: (r, κ) turns in phase space for dt/2 to (r_h, κ_h); there the block's H_eff gives
//! U = exp(−i H_eff dt/2); ψ_h = U ψ; κ_h += dt (F − F_harmonic) with F the Ehrenfest force
//! −⟨∇H_herm⟩ (the trap's full force and the light's) and F_harmonic = −(m ω_a²/ħ) x_a its harmonic
//! part; ψ ← U ψ_h; and the harmonic flow for dt/2 again. Only the anharmonic rest and the light
//! are split, so the step is not bound to the trap period. The scalar part of H_herm (the trap's
//! depth) is a phase and left out of U.
//!
//! # Jumps
//! When ‖ψ‖² falls to the trajectory's threshold θ (uniform in (0, 1]), the crossing time t* is
//! found on ln‖ψ‖² (linear on the half step where it lies), ψ(t*) re-propagated from the step's
//! start and the jump made at r(t*): the carrier c and the block F' it ends in are drawn with
//! probability Σ_a ‖u_a‖², u_a = P_F' C_a R_c ψ; the photon's direction n from the dipole pattern
//! tr G − n·Re(G)·n, G_ab = ⟨u_a|u_b⟩, exactly (rejection from the uniform sphere); its polarization
//! ε across n from ‖Σ_a ε_a u_a‖² on a real orthonormal pair; then ψ ← f = Σ_a ε_a u_a and the
//! emission recoil κ −= k_c n. The rest of the step is propagated in the new block.
//!
//! # Absorption
//! The light absorbed with the photon moves the atom too: by ħ k_l of the beam l it came from,
//! drawn by its share of the photon, ‖Σ_a ε_a P_F' C_a a_l R_l ψ‖² (bounded kicks, the beams' own
//! recoil), plus the difference between the exact conditional mean of this photon,
//! Im⟨f|∇f⟩/⟨f|f⟩ (the jump term's first Wigner moment), and the drawn kicks' mean, so that the
//! jumps carry the exact mean force of the absorbed light -- its radiation pressure, its Doppler
//! imbalance, and the excited states' own force while the photon is held, which the light shift
//! alone gets wrong near resonance.
//!
//! # Randomness
//! Every jump draws from its own stream (the trajectory's and the jump's number), so runs that
//! differ only in the time step make the same choices: the step-doubling check compares them.
use crate::model::{Model,GROUND,MOST_STATES};
use crate::rng::Rng;
use crate::Complex;

/// A trajectory's state.
#[derive(Clone,Debug)]
pub struct Trajectory{
    pub block:usize,
    pub psi:[Complex;MOST_STATES],
    pub r:[f64;3],
    /// The momentum over ħ (1/um).
    pub kappa:[f64;3],
    pub threshold:f64,
    pub alive:bool,
    pub jumps:u64,
    /// The run's seed and the trajectory's number: its jumps' streams.
    pub seed:u64,
    pub index:u64,
}

/// What the engine needs besides the model.
#[derive(Clone,Debug)]
pub struct Settings{
    pub dt:f64,
    /// The axes the atom moves along (the others stay at 0).
    pub free:[bool;3],
    /// The atom is lost once every focused beam's intensity at it is below this share of its peak.
    pub escape_intensity:f64,
    /// ω_a (rad/us): the trap's harmonic part along each axis, moved exactly.
    pub frequencies:[f64;3],
    /// The bound on each exponential's truncated Taylor remainder relative to ‖ψ‖ (`propagate`):
    /// a share of the step's own error budget in a solve, f64::EPSILON for double precision.
    pub truncation:f64,
    /// The motion's sums (T, ⟨v⟩, n̄, ⟨r²⟩) averaged over one trap period of each axis centred on
    /// each output time (lib.rs `simulate`, a solve's), or taken at the output time.
    pub average:bool,
}

/// The harmonic flow along one axis for τ: (x, κ) turned by ωτ in phase space (x in um, κ in 1/um,
/// hm = ħ/m); free flight when ω is 0.
fn flow(x:f64,kappa:f64,omega:f64,hm:f64,tau:f64)->(f64,f64){
    if omega==0.{return(x+hm*kappa*tau,kappa);}
    let(s,c)=(omega*tau).sin_cos();(x*c+hm*kappa/omega*s,kappa*c-omega/hm*x*s)
}

fn norm2(psi:&[Complex],n:usize)->f64{psi[..n].iter().map(|z|z.norm_sqr()).sum()}

/// ψ ← exp(−i H τ) ψ for an n × n matrix H (row-major in a 5 × 5 array): Taylor series on the vector
/// in sub-steps of ‖H‖τ ≤ ½ (‖H‖ the largest column sum of |re| + |im|, above the 1-norm), each
/// summed through the term whose bound (‖H‖τ)^m/m! falls below its share of `truncation` (relative
/// to ‖ψ‖; the remainder past it is at most half that bound).
pub fn propagate(h:&[Complex;25],n:usize,tau:f64,psi:&mut [Complex;MOST_STATES],truncation:f64){
    macro_rules! sized{($($n:literal)*)=>{match n{$($n=>propagate_sized::<$n>(h,tau,psi,truncation),)*_=>unreachable!("a hyperfine block has 1 to 5 states")}}}
    sized!(1 2 3 4 5)
}
fn propagate_sized<const N:usize>(h:&[Complex;25],tau:f64,psi:&mut [Complex;MOST_STATES],truncation:f64){
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
pub fn jump_stream(seed:u64,index:u64,jump:u64)->Rng{Rng::new(seed^0x6A75_6D70_5F73_7472,(index<<24)^jump)}

/// The engine: a model and its settings.
pub struct Engine<'a>{pub model:&'a Model,pub settings:Settings}

impl<'a> Engine<'a>{
    /// One step's flow, kick and internal evolution of length dt without jumps: the norms of ψ at
    /// the start, the middle and the end, ψ at the start, the generator and point of the middle,
    /// and the kick's momentum (the light's and the trap's anharmonic part).
    fn advance(&self,t:&mut Trajectory,dt:f64)->(f64,f64,f64,[Complex;MOST_STATES],crate::model::Local,crate::model::Point,[f64;3]){
        let(m,s)=(self.model,&self.settings);let hm=m.hbar_over_mass;
        for a in 0..3{if s.free[a]{(t.r[a],t.kappa[a])=flow(t.r[a],t.kappa[a],s.frequencies[a],hm,0.5*dt);}}
        let point=m.point(t.r,t.kappa);
        let n=m.dimension(t.block);let local=m.local(t.block,&point);
        let psi0=t.psi;let n0=norm2(&t.psi,n);
        let mut psi=t.psi;propagate(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let nh=norm2(&psi,n);
        let force=m.force(t.block,&point,&local,&psi);let mut kick=[0.;3];
        for a in 0..3{if s.free[a]{let w=s.frequencies[a];kick[a]=dt*(force[a]+w*w/hm*t.r[a]);t.kappa[a]+=kick[a];}}
        propagate(&local.hamiltonian,n,0.5*dt,&mut psi,s.truncation);let n1=norm2(&psi,n);
        t.psi=psi;
        for a in 0..3{if s.free[a]{(t.r[a],t.kappa[a])=flow(t.r[a],t.kappa[a],s.frequencies[a],hm,0.5*dt);}}
        (n0,nh,n1,psi0,local,point,kick)
    }

    /// The step-doubling error of one step of dt from a trajectory's state, without jumps: one step
    /// against two of dt/2, as (|Δκ|, |kick|) -- the momentum's difference and the step's
    /// non-harmonic impulse (1/um) -- and (max |Δp|, |Δ ln‖ψ‖²|, |ln‖ψ‖²|) -- the largest difference of
    /// the block's state populations, of the norm's decay and the decay itself.
    pub fn doubling(&self,t:&Trajectory,dt:f64)->([f64;2],[f64;3]){
        let n=self.model.dimension(t.block);
        let mut one=t.clone();let(n0,_,n1,_,_,_,kick)=self.advance(&mut one,dt);
        let mut two=t.clone();self.advance(&mut two,0.5*dt);let(_,_,n2,_,_,_,_)=self.advance(&mut two,0.5*dt);
        let dk=(0..3).map(|a|(one.kappa[a]-two.kappa[a]).powi(2)).sum::<f64>().sqrt();let impulse=kick.iter().map(|x|x*x).sum::<f64>().sqrt();
        let(a,b)=(norm2(&one.psi,n),norm2(&two.psi,n));
        let dp=(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max);
        ([dk,impulse],[dp,((n1/n0).ln()-(n2/n0).ln()).abs(),(n1/n0).ln().abs()])
    }

    /// One step of dt: flow, kick and flow (`advance`); a jump where ‖ψ‖² crosses the threshold,
    /// the rest of the step after it; the escape.
    pub fn step(&self,t:&mut Trajectory){
        let(m,s)=(self.model,&self.settings);let dt=s.dt;
        let r0=t.r;
        let(n0,nh,n1,psi0,local,point,_)=self.advance(t,dt);let n=m.dimension(t.block);
        if n1<=t.threshold{
            // The crossing on ln‖ψ‖², linear on the half where it lies.
            let(ta,tb,la,lb)=if nh<=t.threshold{(0.,0.5*dt,n0.ln(),nh.ln())}else{(0.5*dt,dt,nh.ln(),n1.ln())};
            let at=if lb<la{(ta+(t.threshold.ln()-la)/(lb-la)*(tb-ta)).clamp(ta,tb)}else{tb};
            let mut before=psi0;propagate(&local.hamiltonian,n,at,&mut before,s.truncation);
            let rj:[f64;3]=std::array::from_fn(|a|r0[a]+(t.r[a]-r0[a])*at/dt);
            let block=t.block;t.psi=before;self.jump(t,rj);
            // The rest of the step in the block the jump left the atom in.
            let rest=dt-at;
            if rest>0.{let nb=m.dimension(t.block);let l2=if t.block==block{local}else{m.local(t.block,&point)};
                propagate(&l2.hamiltonian,nb,rest,&mut t.psi,s.truncation);
                // A second crossing within the step (rare): made at the step's end.
                if norm2(&t.psi,nb)<=t.threshold{self.jump(t,t.r);}}
        }
        // Escape: every focused beam's intensity below the share.
        if !m.focused.is_empty()&&m.focused.iter().all(|&l|m.geometry[l].field(t.r).0.norm_sqr()<s.escape_intensity){t.alive=false;}
    }

    /// A quantum jump at r with the trajectory's (unnormalized) state.
    pub fn jump(&self,t:&mut Trajectory,r:[f64;3]){
        let m=self.model;let s=&self.settings;t.jumps+=1;let mut rng=jump_stream(t.seed,t.index,t.jumps);
        let point=m.point(r,t.kappa);let n=m.dimension(t.block);let norm=norm2(&t.psi,n).sqrt();
        let mut psi=t.psi;for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        let photons=m.photons(t.block,&point,&psi);
        // The carrier and the block it ends in.
        let mut options:Vec<(usize,usize,f64)>=vec![];
        for c in 0..m.carriers.len(){for(b,block)in m.blocks.iter().enumerate(){
            let w:f64=(0..3).map(|a|block.states.iter().map(|&row|photons.amplitude[c][a][row].norm_sqr()).sum::<f64>()).sum();
            if w>0.{options.push((c,b,w));}
        }}
        let total:f64=options.iter().map(|o|o.2).sum();
        if !(total>0.){t.threshold=rng.open_uniform();return;}
        let mut x=rng.uniform()*total;let mut pick=options[options.len()-1];
        for o in &options{if x<o.2{pick=*o;break;}x-=o.2;}
        let(c,b,_)=pick;let states=&m.blocks[b].states;let nb=states.len();
        let rows=|v:&[Complex;GROUND]|->[Complex;MOST_STATES]{let mut out=[Complex::ZERO;MOST_STATES];for(i,&row)in states.iter().enumerate(){out[i]=v[row];}out};
        let u:[[Complex;MOST_STATES];3]=std::array::from_fn(|a|rows(&photons.amplitude[c][a]));
        let gram:[[Complex;3];3]=std::array::from_fn(|a|std::array::from_fn(|b2|(0..nb).fold(Complex::ZERO,|acc,i|acc+u[a][i].conj()*u[b2][i])));
        let direction=dipole_direction(&mut rng,&gram);
        // The polarization across it.
        let pair=across(direction);
        let combine=|v:&[[Complex;MOST_STATES];3],e:&[f64;3]|->[Complex;MOST_STATES]{let mut out=[Complex::ZERO;MOST_STATES];for i in 0..nb{for a in 0..3{out[i]+=v[a][i]*e[a];}}out};
        let amplitudes=pair.map(|e|combine(&u,&e));let weights=amplitudes.map(|v|norm2(&v,nb));
        let chosen=if rng.uniform()*(weights[0]+weights[1])<weights[0]{0}else{1};let e=pair[chosen];
        let f=amplitudes[chosen];let size=weights[chosen];
        // The recoils: the photon's, the absorbed beam's and the exact conditional mean's correction.
        let k=m.carriers[c].wave_number;let mut kick=direction.map(|x|-k*x);
        let gradient=photons.gradient(m,&point,&psi,c);let mean=absorption_mean(&gradient,states,&e,&f,size);
        let shares:Vec<f64>=photons.beams(m,&point,c).iter().map(|v|norm2(&combine(&std::array::from_fn(|a|rows(&v[a])),&e),nb)).collect();
        let sum:f64=shares.iter().sum();
        if sum>0.{
            let mut x=rng.uniform()*sum;let mut l=m.carriers[c].beams.len()-1;
            for(j,&w)in shares.iter().enumerate(){if x<w{l=j;break;}x-=w;}
            let drawn=m.geometry[m.carriers[c].beams[l]].wave_vector();
            let average:[f64;3]=std::array::from_fn(|a|m.carriers[c].beams.iter().zip(&shares).map(|(&beam,&w)|w*m.geometry[beam].wave_vector()[a]).sum::<f64>()/sum);
            for a in 0..3{kick[a]+=drawn[a]+mean[a]-average[a];}
        }else{for a in 0..3{kick[a]+=mean[a];}}
        for a in 0..3{if s.free[a]{t.kappa[a]+=kick[a];}}
        let mut next=f;for z in next[..nb].iter_mut(){*z=*z*(1./size.sqrt());}
        t.block=b;t.psi=next;t.threshold=rng.open_uniform();
    }
}

/// Im⟨f|∇f⟩/⟨f|f⟩ of the emitted photon f = Σ_a ε_a P_F' C_a R_c ψ (`gradient` the carrier's
/// ∇(C_a R_c ψ) [a][axis]): the exact mean momentum (1/um) the absorbed light gives with it.
fn absorption_mean(gradient:&[[[Complex;GROUND];3];3],states:&[usize],e:&[f64;3],f:&[Complex;MOST_STATES],size:f64)->[f64;3]{
    std::array::from_fn(|x|{
        let mut s=0.;
        for(i,&row)in states.iter().enumerate(){let mut g=Complex::ZERO;for a in 0..3{g+=gradient[a][x][row]*e[a];}s+=(f[i].conj()*g).im;}
        s/size
    })
}
