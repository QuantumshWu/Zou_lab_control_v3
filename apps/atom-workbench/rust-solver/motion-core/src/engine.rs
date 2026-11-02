//! The engine of 1D motion: GMW-o1 in a box (the radial tweezer pipeline study of 2026-11-02).
//!
//! # State
//! The eliminated ground manifold's density ρ(x1, x2) on its block-diagonal elements (the secular
//! step's, `Engine::elements`: 34 for Rb87), held in the mixed representation
//! S_e(z, κ) = Σ_y ρ_e(z + y/2, z − y/2) e^{−iκy} of the centre z and of κ (rad/um, p = ħκ), the
//! momentum conjugate to the displacement y, on a box periodic in both. ρ is Hermitian,
//! ρ_ba(z, y) = ρ_ab(z, −y)*, so only the elements a ≤ b are stored (`Engine::fields`: 21 of 34);
//! at a point the elements below the diagonal are those above at −y, conjugated, and every per-point
//! operator is stored for y ≥ 0 only, its pair at −y following from the same symmetry (checked by
//! `the_operators_at_minus_y_mirror_those_at_y`).
//!
//! The grid (`Grid`): z on dz = λ_ref/32 (λ_ref = 2π/k_ref, k_ref the largest wave number of the
//! near carriers), κ on [−P k_ref, P k_ref) with dκ = 2 k_ref/M, y on dy = π/(P k_ref), so that the
//! y box is M λ_ref/2 wide. M starts at 2: the box then holds one standing-wave period of the near
//! carriers on either side, and the coherence between neighbouring sites of their light-shift
//! potential (λ_ref/4 apart for lin⊥lin) stays inside it (with M = 1 its edge sits on the site
//! spacing and that coherence aliases: an early loss of 2e-4 and a 1 % drift of T in TW2). Both
//! positions x1 = z + y/2 and x2 = z − y/2 lie on the lattice h = dy/2, of which dz is a whole
//! multiple (P a multiple of 8), so the generator is evaluated once per lattice position (`Site`).
//! P/8 and nz/2 have no prime factors beyond 5 (fast transforms).
//!
//! # Step
//! Strang, K(dt/2) E K(dt/2) with the kinetic halves merged. K = exp(−i(ħ/m)Qκτ) is exact in
//! (Q, κ), Q the momentum difference conjugate to z (an FFT along z); E = exp(dt L_loc) is exact at
//! every point (z, y) (an FFT along κ), with
//!
//! ```text
//! L_loc ρ = −i(H_eff(x1) ρ − ρ H_eff(x2)†) + Σ_j K_j(k_c y) L_j(x1) ρ L_j(x2)† − ½(W(x1) + W(x2)) ρ:
//! ```
//!
//! the eliminated generator at both positions (`Ground::local`), the dipole kernel of jump j's
//! photon (`dipole_kernel`) and the absorbing layer W. E − 1 is kept in f32 and applied in f64
//! arithmetic (its rounding is then fixed from step to step, the state's f64); on the line y = 0,
//! the trace's, it is kept in f64. E is dense (the jumps carry populations between the hyperfine
//! levels both ways), so the local step streams d² numbers per point: it is bound by memory
//! bandwidth, and the kick's operators are stored the same way.
//!
//! The first order in the Doppler and recoil shifts (`Ground::moving`) is an Euler kick at the
//! middle of each quarter of the radial period T_r: S += T_r/4 L1 S. With κ̂₁ = κ + Q/2 the ket's
//! momentum and ρκ̂₂ (κ̂₂ = κ − Q/2) the bra's, and as (κ̂₁ρ)(x2, x1)† = (ρκ̂₂)(x1, x2),
//! L1 = Y + Y† = 𝓐 ρ + 𝓑 κ̂₁ρ + 𝓒 ρκ̂₂ at every point: three superoperators per point (`Engine::point`),
//! from Y = (i G_0 + G_L′) ρ + i(G_R + G_L)(κ̂₁ρ) − Σ_j K_j (F_j ρ + M_j κ̂₁ρ) L_j(x2)† (κ̂₁ G_L(x1) =
//! G_L(x1) κ̂₁ − i G_L′(x1), G_L′ by central differences on the lattice). The midpoint rule over the
//! quarters integrates the motion's harmonics up to the third exactly over a period; two kicks per
//! period would alias the 2ω one onto the mean.
//!
//! # Box
//! The box ends where the trap does: the absorbing layer starts where the depth has fallen to
//! 1/`ESCAPE` of the focus's, and an atom beyond counts as lost. (A layer inside the bound region
//! absorbs atoms at their turning points within a grid step; the ringing it leaves goes negative,
//! and absorbing it gains trace.) The layer fills the box's outer fifth and absorbs an atom at the
//! escape speed to within the loss's share. The box and the momentum range do not shrink as the atom
//! cools: the atoms heated by the light fill both out to the escape radius and momentum at 1e-6, and
//! the κ² tail beyond 0.7 × 24 k_ref still holds a quarter to a half of ⟨κ²⟩ in TW2's cold phase.
//!
//! # Error control
//! Two tolerances: `Request::tolerance` (populations and the loss, absolute) and
//! `MOTION_TOLERANCE` (T and n̄, relative). Each a-posteriori discretization estimate (the time
//! step, the momentum range, the momentum step, the layer) may take `SHARE` of them, so that the four
//! together stay within them:
//! - the momentum range grows ×1.5 while the share of ⟨κ²⟩ beyond `EDGE` P exceeds it, up to the
//!   escape momentum;
//! - the momentum step halves while the coherence at the y box's edge exceeds it;
//! - the time step (`Engine::run`): it starts where one period's error, from a period with 2 dt
//!   (E applied twice), is within the share; over the first checkpoint interval a companion runs
//!   with 2 dt, and the difference of their outputs, /3, is dt's error. Past the full tolerance the
//!   start is not resolved: dt halves and the run starts again. Otherwise the interval stands with
//!   its error as measured, and the later intervals take dt from the error at its end, past the
//!   start's transient: dt halves above the share and doubles when four times it is within half the
//!   share.
//!
//! # Continuation
//! The outputs (momentum and position distributions, populations, the trace) are snapshotted once
//! per radial period, stroboscopically. Every `CHECKPOINT_PERIODS` periods a dynamic mode
//! decomposition of the last half of the snapshots predicts every remaining output: singular
//! directions below √ε σ_max cut (the reduced operator's condition within 1/√ε), the operator
//! balanced by Σ^{1/2}, its eigen-decomposition this crate's own (`eigen`, so that every build gives
//! the same continuation), modes growing put on the unit circle. The prediction is compared with the
//! one made at half the run -- at the times simulated since and at every later one -- and the run
//! stops when they agree within the tolerances. The two come from disjoint halves of the history:
//! consecutive checkpoints share most of theirs, and their change underestimated the error 2-4×
//! (the predictions converge geometrically, about 0.8 per checkpoint, so a consecutive change is
//! ≈ (1 − r)/r ≈ ¼ of the error, the factor that rule needed); the doubling rule needs no factor, and
//! the change it measures is mostly the older prediction's error, the newer one's being smaller.
//!
//! # Remaining time
//! A cost model: the run's wall time per simulated us and grid point at the current dt (the
//! companion's apart), measured from the first period, times the grid and the simulated time left,
//! plus each checkpoint's own; the time left runs to the end, or to the acceptance the changes'
//! geometric decrease since their largest projects (they rise while the continuation learns the
//! slow modes, then fall; a least-squares line through their logarithms, from two changes after the
//! largest on). Replayed on TW2 and TW1, the projection is 0.4-1.8× the time actually left; before
//! it, the estimate to the end is 2-4×.
use crate::ground::{Ground,Local,dipole_kernel};
use crate::linear::{Dense,Lu};
use crate::{Complex,Distributions,Request,Solution,STATES};
use atom_population_core::{Complex64,Lapack};
use rayon::prelude::*;
use rustfft::{Fft,FftPlanner};
use std::collections::HashMap;
use std::f64::consts::{PI,TAU};
use std::sync::Arc;
use std::time::Instant;

type Z=rustfft::num_complex::Complex64;

/// ATOM_MOTION_TRACE set: the engine's decisions on stderr.
fn tracing()->bool{static ON:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*ON.get_or_init(||std::env::var_os("ATOM_MOTION_TRACE").is_some())}
macro_rules! trace{($($a:tt)*)=>{if tracing(){eprintln!($($a)*);}}}

const HBAR:f64=1.054_571_817e-34;
const BOLTZMANN:f64=1.380_649e-23;
/// The relative tolerance of T and n̄.
pub const MOTION_TOLERANCE:f64=0.1;
/// The share of a tolerance each a-posteriori discretization estimate may take.
const SHARE:f64=0.25;
/// A range's tail is measured beyond this fraction of it, so that it grows before its edge is reached.
const EDGE:f64=0.7;
/// The absorbing layer fills the box's outer fifth.
const LAYER:f64=0.8;
/// The trap ends where its depth has fallen to 1/ESCAPE of the focus's (the pipeline study's choice).
const ESCAPE:f64=20.;
/// Radial periods between checkpoints.
const CHECKPOINT_PERIODS:usize=25;
/// First-order kicks per radial period: the midpoint rule misses (2π/4)²/24 ≈ 10 % of the term
/// (of a few per cent of T), within its share.
const KICKS:usize=4;
/// The coarsest time step, steps per radial period: the Strang phase error of the harmonic trap,
/// (2π/n)²/24, within the share of the T tolerance.
const COARSEST:usize=8;
/// The finest time step tried, steps per radial period.
const FINEST:usize=4096;

/// The box: the momentum range ±p k_ref (p a multiple of 8), the momentum step dκ = 2 k_ref/m2,
/// nz centres.
#[derive(Clone,Copy,Debug,PartialEq)]
struct Grid{p:usize,m2:usize,nz:usize}
impl Grid{
    fn ny(self)->usize{self.p*self.m2}
    fn points(self)->usize{self.nz*self.ny()}
    /// dz / h.
    fn stride(self)->i64{(self.p/8) as i64}
}
/// The smallest n′ ≥ max(n, 1) without prime factors beyond 5.
fn smooth(n:usize)->usize{(n.max(1)..).find(|&m|five_smooth(m)).unwrap()}
/// The largest n′ ≤ max(n, 1) without prime factors beyond 5.
fn smooth_below(n:usize)->usize{(1..=n.max(1)).rev().find(|&m|five_smooth(m)).unwrap()}
fn five_smooth(mut m:usize)->bool{for p in [2,3,5]{while m%p==0{m/=p;}}m==1}
/// An FFT-order index as a signed one.
fn signed(i:usize,n:usize)->i64{if i<n/2{i as i64}else{i as i64-n as i64}}
/// The FFT-order index of a signed one.
fn unsigned(s:i64,n:usize)->usize{s.rem_euclid(n as i64) as usize}

/// The generator at one lattice position: H_eff and the driven jumps (ground ← ground, row-major),
/// the first-order operators, and at the centres the readout; the kick's `hold` = i G_0 + G_L′ (on ρ)
/// and `shift` = i (G_R + G_L) (on κ̂₁ρ), ng × ng.
struct Site{heff:Vec<Z>,jumps:Vec<Vec<Z>>,right:Vec<Z>,left:Vec<Z>,constant:Vec<Z>,momentum:Vec<Vec<Z>>,fixed:Vec<Vec<Z>>,local:Option<Local>,
    hold:Vec<Z>,shift:Vec<Z>}
fn flat(m:&Dense)->Vec<Z>{m.entries.iter().map(|z|Z::new(z.re,z.im)).collect()}

/// A synchronized readout: the populations, densities and moments of the atom still held
/// (normalized by the trace), and the trace relative to the start.
#[derive(Clone,Debug)]
struct Reading{t:f64,populations:Vec<f64>,trace:f64,kappa:Vec<f64>,p_kappa:Vec<f64>,z:Vec<f64>,p_z:Vec<f64>,mean:f64,square:f64,spread:f64}
/// The outputs the request reports at one time.
#[derive(Clone,Debug)]
struct Outputs{populations:Vec<f64>,temperature:f64,velocity:f64,occupation:f64,loss:f64}

/// A dynamic mode decomposition's prediction at the snapshot times after `t`.
/// `errors`: per time, the population change since the previous checkpoint's prediction (1 without one).
#[derive(Clone,Debug)]
struct Prediction{t:f64,times:Vec<f64>,outputs:Vec<Outputs>,errors:Vec<f64>,kappa:Vec<f64>,z:Vec<f64>,p_kappa:Vec<Vec<f64>>,p_z:Vec<Vec<f64>>}

pub struct Engine<'a>{
    ground:&'a Ground,lapack:&'a Lapack,
    /// k_ref (rad/um), ħ/m (um²/us), m (kg).
    k:f64,hbar_over_m:f64,mass:f64,
    ng:usize,d:usize,elements:Vec<(usize,usize)>,element:Vec<Option<usize>>,transposed:Vec<usize>,diagonal:Vec<usize>,
    /// The stored fields: the elements (a, b) with a ≤ b (ρ is Hermitian: ρ_ba(x1, x2) = ρ_ab(x2, x1)*);
    /// per element its field and whether it is that field's conjugate at −y; the diagonal's fields.
    fields:Vec<usize>,source:Vec<(usize,bool)>,diagonal_fields:Vec<usize>,
    /// The driven jumps (indices into `Ground::jumps`) and each one's kernel: along the axis, k_c.
    kept:Vec<usize>,kernels:Vec<(bool,f64)>,
    grid:Grid,h:f64,dz:f64,dy:f64,dkappa:f64,dq:f64,
    /// The lattice sites n ∈ [−reach, reach] (`site_reach`), at index n + reach.
    sites:Vec<Site>,site_offset:i64,
    /// The absorbing layer: from x_abs to the box's edge, rising quadratically to `cap` (1/us).
    x_abs:f64,edge:f64,cap:f64,
    /// The trap: ω (rad/us), the escape momentum (rad/um), the initial state's ground weights.
    omega:f64,escape_momentum:f64,weights:Vec<f64>,
    period:f64,steps:usize,dt:f64,
    /// E − 1 and the kick's 𝓐, 𝓑, 𝓒 at the points y ≥ 0 (the pair at −y follows by Hermiticity;
    /// d × d each, f32, real then imaginary parts).
    exponential:Vec<f32>,kicks:Vec<f32>,
    /// E − 1 and 𝓐, 𝓑, 𝓒 on the line y = 0 (per centre, d × d each, f64).
    exact:Vec<Z>,exact_kicks:Vec<Z>,
    /// The state, [z][field][κ], and the scratch of the passes along z, [field][κ][z].
    state:Vec<Z>,planes:Vec<Z>,
    /// The kick's scratch: κ̂₁ρ and ρκ̂₂ (fields), and L1 ρ.
    ket:Vec<Z>,bra:Vec<Z>,kicked:Vec<Z>,
    /// exp(−i(ħ/m)Qκτ)/nz per (κ, Q) for the kinetic steps τ used on this grid.
    phase_tables:HashMap<u64,Vec<Z>>,
    plans:[Arc<dyn Fft<f64>>;4],
    trace0:f64,
    /// Wall time (s) in the kinetic steps, the local steps, the kicks and the readouts, when tracing.
    clock:[f64;4],
}

impl<'a> Engine<'a>{
    pub fn new(ground:&'a Ground,request:&Request,lapack:&'a Lapack)->Result<Engine<'a>,String>{
        let ng=ground.ground_states.len();
        let elements:Vec<(usize,usize)>=(0..ng).flat_map(|a|(0..ng).map(move|b|(a,b))).filter(|&(a,b)|ground.block[a]==ground.block[b]).collect();
        let d=elements.len();let mut element=vec![None;ng*ng];for(e,&(a,b))in elements.iter().enumerate(){element[a*ng+b]=Some(e);}
        let transposed=elements.iter().map(|&(a,b)|element[b*ng+a].unwrap()).collect();
        let diagonal:Vec<usize>=(0..ng).map(|a|element[a*ng+a].unwrap()).collect();
        // The elements run block by block, row-major within each (the kick reads them as blocks).
        let mut at=0;for block in &ground.blocks{for &a in &block.states{for &b in &block.states{
            if elements.get(at)!=Some(&(a,b)){return Err("The motion core needs the ground states of each hyperfine level in one run".into());}at+=1;}}}
        let fields:Vec<usize>=(0..elements.len()).filter(|&e|elements[e].0<=elements[e].1).collect();
        let source=elements.iter().map(|&(a,b)|{let e=element[a.min(b)*ng+a.max(b)].unwrap();(fields.iter().position(|&x|x==e).unwrap(),a>b)}).collect();
        let diagonal_fields=diagonal.iter().map(|e|fields.iter().position(|x|x==e).unwrap()).collect();
        let hbar_over_m=HBAR/request.mass_kg*1e6;
        let k=ground.carriers.iter().map(|c|c.wave_number).fold(0_f64,f64::max);
        let k=if k>0.{k}else{ground.beams.iter().map(|b|b.wave_number_per_um).fold(0_f64,f64::max)};
        // The initial state: its ground populations weigh the potential the motion starts thermal in.
        let rho=&request.initial_density;
        let excited:f64=ground.excited_states.iter().map(|&i|rho[i*STATES+i].re).sum();
        if excited>1e-9{return Err(format!("1D 运动从基态开始：初态有激发态布居 {excited:.1e}"));}
        let weights:Vec<f64>=ground.ground_states.iter().map(|&i|rho[i*STATES+i].re).collect();
        let total:f64=weights.iter().sum();let weights:Vec<f64>=weights.iter().map(|w|w/total).collect();
        let curvature:f64=(0..ng).map(|b|weights[b]*ground.curvature(b)).sum();
        if !(curvature>0.){return Err(crate::NOT_CONFINED.into());}
        let omega=(hbar_over_m*curvature).sqrt();
        let potential=|x:f64|{let h=ground.hamiltonian(x);(0..ng).map(|b|weights[b]*h[(ground.ground_states[b],ground.ground_states[b])].re).sum::<f64>()};
        let(bottom,top)=(potential(0.),potential(1e6));let depth=top-bottom;
        if !(depth>0.){return Err(crate::NOT_CONFINED.into());}
        let escape_momentum=(2.*depth/hbar_over_m).sqrt();
        let dz=PI/(16.*k);
        // The escape radius: where the depth has fallen to 1/ESCAPE on both sides.
        let mut escape_radius=dz/16.;
        while [escape_radius,-escape_radius].iter().any(|&x|potential(x)-bottom<(1.-1./ESCAPE)*depth){escape_radius+=dz/16.;
            if escape_radius>1e4{return Err(crate::NOT_CONFINED.into());}}
        // The initial momentum range from the thermal state's width and the tail criterion of a
        // Gaussian; the box to the escape radius.
        let temperature=request.temperature_uk;
        let quantum=(HBAR*omega*1e6/(BOLTZMANN*temperature*1e-6)/2.).tanh().recip();
        let sigma_kappa=(omega/(2.*hbar_over_m)*quantum).sqrt();
        // The sizes along κ and z have no prime factors beyond 5 (fast transforms): P/8 and nz/2 smooth.
        let p_cap=8*smooth((escape_momentum/k/8.).ceil() as usize);
        let p=(8*smooth((gaussian_tail(SHARE*MOTION_TOLERANCE)*sigma_kappa/(EDGE*k)/8.).ceil() as usize)).min(p_cap);
        let nz=2*smooth(((escape_radius/LAYER/dz).ceil() as usize).max(4));
        let period=TAU/omega;
        let kept={let e=ground.eliminate(&ground.hamiltonian(0.))?;ground.driven_jumps(&e)};
        let kernels=kept.iter().map(|&j|{let jump=&ground.jumps[j];(jump.axis==2,ground.carriers[jump.carrier].wave_number)}).collect();
        // The y box holds one standing-wave period of the near carriers, λ_ref/2 on either side: the
        // coherence between neighbouring sites of their light-shift potential (λ_ref/4 apart for
        // lin⊥lin) stays inside it.
        let grid=Grid{p,m2:2,nz};
        let mut engine=Engine{ground,lapack,k,hbar_over_m,mass:request.mass_kg,ng,d,elements,element,transposed,diagonal,fields,source,diagonal_fields,kept,kernels,
            grid,h:0.,dz,dy:0.,dkappa:0.,dq:0.,sites:vec![],site_offset:0,x_abs:0.,edge:0.,cap:0.,omega,escape_momentum,weights,
            period,steps:COARSEST,dt:period/COARSEST as f64,exponential:vec![],kicks:vec![],exact:vec![],exact_kicks:vec![],state:vec![],planes:vec![],ket:vec![],bra:vec![],kicked:vec![],phase_tables:HashMap::new(),plans:plans(8,8),trace0:1.,clock:[0.;4]};
        engine.regrid(grid)?;
        engine.state=engine.thermal(temperature,rho)?;
        engine.trace0=engine.readout(0.).trace;
        trace!("engine: k_ref {k:.4} rad/um, omega/2pi {:.3} kHz, escape {escape_radius:.3} um / {escape_momentum:.1} rad/um, grid {:?}, sites {}, kept jumps {}, d {}",omega/TAU*1e3,engine.grid,engine.sites.len(),engine.kept.len(),engine.d);
        Ok(engine)
    }

    /// Sets the grid's derived quantities, the sites and the per-point operators.
    fn regrid(&mut self,grid:Grid)->Result<(),String>{
        let k=self.k;
        if grid.p!=self.grid.p||self.h==0.{self.sites.clear();self.site_offset=0;}
        self.grid=grid;self.h=PI/(2.*grid.p as f64*k);self.dy=2.*self.h;self.dkappa=2.*k/grid.m2 as f64;
        self.dq=TAU/(grid.nz as f64*self.dz);
        self.edge=grid.nz as f64/2.*self.dz;self.x_abs=LAYER*self.edge;
        // The layer absorbs an atom at the escape speed within SHARE × the population tolerance:
        // transmission exp(−2 ∫W dx / v) with ∫W dx = cap (edge − x_abs)/3 across it.
        let speed=self.hbar_over_m*self.escape_momentum;
        self.cap=(1./(SHARE*1e-2)).ln()*3.*speed/(2.*(self.edge-self.x_abs));
        self.plans=plans(grid.nz,grid.ny());self.phase_tables.clear();self.planes=vec![Z::default();self.fields.len()*grid.points()];
        self.ensure_sites()?;
        self.build();
        Ok(())
    }

    fn reach(&self)->i64{(self.grid.nz/2) as i64*self.grid.stride()+(self.grid.ny()/2) as i64}
    /// The sites the points use, and one beyond (the derivative of G_L).
    fn site_reach(&self)->i64{self.reach()+1}

    /// Computes every lattice site the grid's points use that is not cached.
    fn ensure_sites(&mut self)->Result<(),String>{
        let(reach,stride)=(self.site_reach(),self.grid.stride());let(old,offset)=(std::mem::take(&mut self.sites),self.site_offset);
        let length=old.len();let have=|n:i64|n+offset>=0&&((n+offset) as usize)<length;
        let(ground,kept,h,hbar_over_m,ng)=(self.ground,&self.kept,self.h,self.hbar_over_m,self.ng);
        let mut fresh:Vec<(i64,Result<Site,String>)>=(-reach..=reach).filter(|&n|!have(n)).collect::<Vec<_>>().par_iter().map(|&n|{
            let x=n as f64*h;
            let site=(||{let e=ground.eliminate(&ground.hamiltonian(x))?;let local=ground.local(&e,x);let moving=ground.moving(&e,x,hbar_over_m);
                Ok(Site{heff:flat(&local.effective_hamiltonian(&e)),jumps:kept.iter().map(|&j|flat(&local.jumps[j])).collect(),
                    right:flat(&moving.right),left:flat(&moving.left),constant:flat(&moving.constant),
                    momentum:kept.iter().map(|&j|flat(&moving.jumps_momentum[j])).collect(),fixed:kept.iter().map(|&j|flat(&moving.jumps_constant[j])).collect(),
                    local:(n.rem_euclid(stride)==0).then_some(local),hold:vec![],shift:vec![]})})();
            (n,site)}).collect();
        let mut old:Vec<Option<Site>>=old.into_iter().map(Some).collect();
        let mut sites=Vec::with_capacity((2*reach+1) as usize);
        for n in -reach..=reach{
            if have(n){sites.push(old[(n+offset) as usize].take().unwrap());}
            else{let k=fresh.iter().position(|f|f.0==n).unwrap();sites.push(fresh.swap_remove(k).1?);}
        }
        // The kick's operators, G_L′ by central differences on the lattice (one-sided at the ends).
        let lefts:Vec<Vec<Z>>=sites.iter().map(|s|s.left.clone()).collect();let last=lefts.len()-1;
        for(k,site)in sites.iter_mut().enumerate(){
            let(lo,hi)=(k.saturating_sub(1),(k+1).min(last));let width=(hi-lo) as f64*h;
            site.hold=(0..ng*ng).map(|t|Z::new(0.,1.)*site.constant[t]+(lefts[hi][t]-lefts[lo][t])/width).collect();
            site.shift=(0..ng*ng).map(|t|Z::new(0.,1.)*(site.right[t]+site.left[t])).collect();
        }
        (self.sites,self.site_offset)=(sites,reach);
        Ok(())
    }
    fn site(&self,n:i64)->&Site{&self.sites[(n+self.site_offset) as usize]}

    /// The absorbing layer's rate at x (1/us).
    fn layer(&self,x:f64)->f64{let r=((x.abs()-self.x_abs)/(self.edge-self.x_abs)).max(0.);self.cap*r.min(1.5).powi(2)}

    /// E − 1 at point (i, j) (d × d, row-major), for the time step dt, and the kick's superoperators
    /// there, L1 = 𝓐 ρ + 𝓑 κ̂₁ρ + 𝓒 ρκ̂₂ (Y + Y† at one point: Y†(x1, x2) = Y(x2, x1)†, with ρ Hermitian
    /// and (κ̂₁ρ)(x2, x1)† = (ρκ̂₂)(x1, x2)):
    /// 𝓐 σ = hold(x1) σ + σ hold(x2)† − Σ_q K_q (fixed_q(x1) σ L_q(x2)† + L_q(x1) σ fixed_q(x2)†),
    /// 𝓑 σ = shift(x1) σ − Σ_q K_q momentum_q(x1) σ L_q(x2)†,  𝓒 σ = σ shift(x2)† − Σ_q K_q L_q(x1) σ momentum_q(x2)†.
    fn point(&self,i:usize,j:usize)->(Vec<Z>,[Vec<Z>;3]){
        let(d,ng,grid)=(self.d,self.ng,self.grid);let dd=d*d;let(si,sj)=(signed(i,grid.nz),signed(j,grid.ny()));
        let(n1,n2)=(si*grid.stride()+sj,si*grid.stride()-sj);let(x1,x2)=(n1 as f64*self.h,n2 as f64*self.h);
        let(one,two)=(self.site(n1),self.site(n2));let y=sj as f64*self.dy;
        let kernel:Vec<f64>=self.kernels.iter().map(|&(along,k)|dipole_kernel(along,k*y)).collect();
        let mut l=vec![Z::default();dd];let mut abc=[vec![Z::default();dd],vec![Z::default();dd],vec![Z::default();dd]];
        let(oh,th)=(&one.hold,&two.hold);let(os,ts)=(&one.shift,&two.shift);
        let damping=-0.5*(self.layer(x1)+self.layer(x2));
        for(r,&(a,b))in self.elements.iter().enumerate(){
            l[r*d+r]+=Z::new(damping,0.);
            for c in 0..ng{if let Some(col)=self.element[c*ng+b]{l[r*d+col]+=Z::new(0.,-1.)*one.heff[a*ng+c];abc[0][r*d+col]+=oh[a*ng+c];abc[1][r*d+col]+=os[a*ng+c];}}
            for c in 0..ng{if let Some(col)=self.element[a*ng+c]{abc[0][r*d+col]+=th[b*ng+c].conj();abc[2][r*d+col]+=ts[b*ng+c].conj();}}
            for c in 0..ng{if let Some(col)=self.element[a*ng+c]{l[r*d+col]+=Z::new(0.,1.)*two.heff[b*ng+c].conj();}}
            for(q,&weight)in kernel.iter().enumerate(){
                let(j1,j2,f1,f2,m1,m2)=(&one.jumps[q],&two.jumps[q],&one.fixed[q],&two.fixed[q],&one.momentum[q],&two.momentum[q]);
                for c in 0..ng{for e2 in 0..ng{if let Some(col)=self.element[c*ng+e2]{
                    let(u,w)=(j1[a*ng+c],j2[b*ng+e2].conj());
                    l[r*d+col]+=u*w*weight;
                    abc[0][r*d+col]-=(f1[a*ng+c]*w+u*f2[b*ng+e2].conj())*weight;
                    abc[1][r*d+col]-=m1[a*ng+c]*w*weight;abc[2][r*d+col]-=u*m2[b*ng+e2].conj()*weight;}}}
            }
        }
        for x in l.iter_mut(){*x*=self.dt;}
        (expm1(&l,d),abc)
    }

    /// E − 1 and 𝓐, 𝓑, 𝓒 at the points y ≥ 0 (f32, real then imaginary parts), and in f64 on the line
    /// y = 0, the trace's: there the f32 rounding, fixed from step to step, would drift it.
    fn build(&mut self){
        let(grid,dd)=(self.grid,self.d*self.d);let half=grid.ny()/2+1;
        // The old tables go first (≈ 300 MB at P = 64, nz = 128).
        (self.exponential,self.kicks)=(vec![],vec![]);
        let(mut exponential,mut kicks)=(vec![0f32;grid.nz*half*2*dd],vec![0f32;grid.nz*half*6*dd]);
        let this=&*self;
        let store=|m:&[Z],out:&mut [f32]|for(t,z)in m.iter().enumerate(){out[t]=z.re as f32;out[dd+t]=z.im as f32;};
        exponential.par_chunks_mut(2*dd).zip(kicks.par_chunks_mut(6*dd)).enumerate().for_each(|(point,(e,k))|{
            let(exp,abc)=this.point(point/half,point%half);store(&exp,e);for(x,out)in abc.iter().zip(k.chunks_mut(2*dd)){store(x,out);}
        });
        let lines:Vec<(Vec<Z>,[Vec<Z>;3])>=(0..grid.nz).into_par_iter().map(|i|this.point(i,0)).collect();
        self.exact=lines.iter().flat_map(|l|l.0.iter().copied()).collect();
        self.exact_kicks=lines.iter().flat_map(|l|l.1.iter().flat_map(|x|x.iter().copied())).collect();
        (self.exponential,self.kicks)=(exponential,kicks);
    }

    /// The thermal state of the potential the initial ground populations weigh, on the lattice (a
    /// sinc DVR), times the initial internal density's secular part.
    fn thermal(&self,temperature:f64,rho:&[Complex])->Result<Vec<Z>,String>{
        let(reach,h,ng)=(self.reach(),self.h,self.ng);let n=(2*reach+1) as usize;
        let potential:Vec<f64>=(-reach..=reach).map(|i|{let hm=self.ground.hamiltonian(i as f64*h);(0..ng).map(|b|self.weights[b]*hm[(self.ground.ground_states[b],self.ground.ground_states[b])].re).sum()}).collect();
        let kinetic=self.hbar_over_m/2./(h*h);
        let mut a=vec![Complex64::default();n*n];
        for i in 0..n{for j in 0..n{let m=i as i64-j as i64;
            let t=if m==0{PI*PI/3.}else{2.*if m%2==0{1.}else{-1.}/(m*m) as f64};
            a[j*n+i].re=kinetic*t+if i==j{potential[i]}else{0.};}}
        let(values,vectors)=self.lapack.hermitian_eigen(&a,n)?;
        let beta=HBAR*1e6/(BOLTZMANN*temperature*1e-6);
        let mut w:Vec<f64>=values.iter().map(|&v|(-beta*(v-values[0])).exp()).collect();let total:f64=w.iter().sum();for x in w.iter_mut(){*x/=total;}
        let used=w.iter().rposition(|&x|x>1e-18).map_or(1,|i|i+1);
        let density=|i:usize,j:usize|->f64{(0..used).map(|m|{let(a,b)=(vectors[m*n+i],vectors[m*n+j]);w[m]*(a.re*b.re+a.im*b.im)}).sum::<f64>()/h};
        let(grid,f)=(self.grid,self.fields.len());let(ny,nz)=(grid.ny(),grid.nz);
        let internal:Vec<Z>=self.fields.iter().map(|&e|{let(a,b)=self.elements[e];let z=rho[self.ground.ground_states[a]*STATES+self.ground.ground_states[b]];Z::new(z.re,z.im)}).collect();
        let mut state=vec![Z::default();f*grid.points()];
        let fft=&self.plans[2];
        state.par_chunks_mut(f*ny).enumerate().for_each(|(i,block)|{
            let si=signed(i,nz);
            for j in 0..ny{let sj=signed(j,ny);let(n1,n2)=(si*grid.stride()+sj,si*grid.stride()-sj);
                let value=density((n1+reach) as usize,(n2+reach) as usize);
                for u in 0..f{block[u*ny+j]=internal[u]*value;}}
            for row in block.chunks_mut(ny){fft.process(row);}
        });
        Ok(state)
    }

    /// exp(−i(ħ/m)Qκτ)/nz per (κ, Q), FFT order.
    fn phases(&self,tau:f64)->Vec<Z>{
        let(ny,nz)=(self.grid.ny(),self.grid.nz);let mut out=vec![Z::default();ny*nz];
        for k in 0..ny{let kappa=signed(k,ny) as f64*self.dkappa;for l in 0..nz{let q=signed(l,nz) as f64*self.dq;out[k*nz+l]=Z::from_polar(1./nz as f64,-self.hbar_over_m*q*kappa*tau);}}
        out
    }


    /// The kinetic step K(τ): along z to Q, the phase, back.
    fn kinetic(&mut self,tau:f64){
        if !self.phase_tables.contains_key(&tau.to_bits()){let table=self.phases(tau);self.phase_tables.insert(tau.to_bits(),table);}
        let phases=std::mem::take(&mut self.phase_tables);let table=&phases[&tau.to_bits()];let nz=self.grid.nz;
        let mut state=std::mem::take(&mut self.state);self.along_z(&mut state,None,&|k,l|table[k*nz+l]);self.state=state;self.phase_tables=phases;
    }

    /// Per row along z of every field and κ: W(Q) → factor(κ, Q) W(Q) (the factor with 1/nz), from
    /// `state` into `out` (or in place).
    fn along_z(&mut self,state:&mut [Z],out:Option<&mut [Z]>,factor:&(dyn Fn(usize,usize)->Z+Sync)){
        let(f,ny,nz)=(self.fields.len(),self.grid.ny(),self.grid.nz);let(fwd,inv)=(&self.plans[0],&self.plans[1]);
        let source:&[Z]=state;
        // Tasks of ROWS κ rows of one field (ny is a multiple of 8).
        const ROWS:usize=8;
        self.planes.par_chunks_mut(ROWS*nz).enumerate().for_each(|(c,plane)|{
            let(u,k0)=(c*ROWS/ny,c*ROWS%ny);
            for i in 0..nz{let row=&source[(i*f+u)*ny+k0..(i*f+u)*ny+k0+ROWS];for(k,&z)in row.iter().enumerate(){plane[k*nz+i]=z;}}
            let mut scratch=vec![Z::default();fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len())];
            for(k,row)in plane.chunks_mut(nz).enumerate(){fwd.process_with_scratch(row,&mut scratch);for(l,z)in row.iter_mut().enumerate(){*z*=factor(k0+k,l);}inv.process_with_scratch(row,&mut scratch);}
        });
        let planes:&[Z]=&self.planes;
        let target=match out{Some(o)=>o,None=>state};
        target.par_chunks_mut(f*ny).enumerate().for_each(|(i,block)|{for(u,row)in block.chunks_mut(ny).enumerate(){for(k,z)in row.iter_mut().enumerate(){*z=planes[(u*ny+k)*nz+i];}}});
    }

    /// Applies E (`times` times) at every point; with `edge`, returns the share of the diagonal
    /// elements' coherence beyond EDGE of the y box's half-width.
    fn local(&mut self,times:usize,edge:bool)->f64{
        let(d,f,ny)=(self.d,self.fields.len(),self.grid.ny());let(dd,half)=(d*d,ny/2+1);let(fwd,inv)=(&self.plans[2],&self.plans[3]);
        let(exponential,exact,diagonal,source,fields,transposed)=(&self.exponential,&self.exact,&self.diagonal,&self.source,&self.fields,&self.transposed);
        let cut=EDGE*(ny/2) as f64;
        let sums:Vec<(f64,f64)>=self.state.par_chunks_mut(f*ny).enumerate().map(|(i,block)|{
            let mut scratch=vec![Z::default();fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len())];
            for row in block.chunks_mut(ny){inv.process_with_scratch(row,&mut scratch);}
            let(mut total,mut outer)=(0.,0.);let(mut vr,mut vi,mut xr,mut xi)=(vec![0.;d],vec![0.;d],vec![0.;d],vec![0.;d]);
            let scale=1./ny as f64;
            // The point y and its pair −y: the elements below the diagonal at y are those above at −y, conjugated.
            for j in 0..half{
                let m=(ny-j)%ny;
                for e in 0..d{let(u,conjugate)=source[e];let z=if conjugate{block[u*ny+m].conj()}else{block[u*ny+j]}*scale;(vr[e],vi[e])=(z.re,z.im);}
                if edge{let w:f64=diagonal.iter().map(|&a|vr[a].hypot(vi[a])).sum::<f64>()*if m==j{1.}else{2.};total+=w;if signed(j,ny).abs() as f64>cut{outer+=w;}}
                // The f32 entries in f64 arithmetic: the rounding of E is fixed, the state's is f64.
                for _ in 0..times{
                    xr.copy_from_slice(&vr);xi.copy_from_slice(&vi);
                    if j==0{let m=&exact[i*dd..(i+1)*dd];
                        for r in 0..d{let row=&m[r*d..(r+1)*d];let(mut ar,mut ai)=(0.,0.);for c in 0..d{ar+=row[c].re*xr[c]-row[c].im*xi[c];ai+=row[c].re*xi[c]+row[c].im*xr[c];}vr[r]+=ar;vi[r]+=ai;}
                    }else{apply(&exponential[(i*half+j)*2*dd..(i*half+j+1)*2*dd],d,&xr,&xi,&mut vr,&mut vi);}
                }
                for(u,&e)in fields.iter().enumerate(){block[u*ny+j]=Z::new(vr[e],vi[e]);if m!=j{let t=transposed[e];block[u*ny+m]=Z::new(vr[t],-vi[t]);}}
            }
            for row in block.chunks_mut(ny){fwd.process_with_scratch(row,&mut scratch);}
            (total,outer)
        }).collect();
        let(total,outer)=sums.iter().fold((0.,0.),|a,b|(a.0+b.0,a.1+b.1));
        if total>0.{outer/total}else{0.}
    }

    /// Qρ/2 into `out` ([z][field][κ]), Q the momentum difference conjugate to z: κ̂₁ρ = κρ + Qρ/2 and
    /// ρκ̂₂ = κρ − Qρ/2. The Nyquist Q is both ±Q_max: its (odd) factor is zero, so that κ̂₁ρ's elements
    /// below the diagonal are exactly ρκ̂₂'s above it, conjugated.
    fn difference(&mut self,out:&mut [Z]){
        let(nz,dq)=(self.grid.nz,self.dq);
        let mut state=std::mem::take(&mut self.state);
        self.along_z(&mut state,Some(out),&|_,l|if 2*l==nz{Z::default()}else{Z::new(0.5*signed(l,nz) as f64*dq/nz as f64,0.)});
        self.state=state;
    }

    /// The first-order Euler kick of span `span` (us): S += span L1 S, L1 = 𝓐 ρ + 𝓑 κ̂₁ρ + 𝓒 ρκ̂₂ per point
    /// (`point`), at y ≥ 0 and its pair −y by Hermiticity.
    fn kick(&mut self,span:f64){
        let(d,f,grid)=(self.d,self.fields.len(),self.grid);let(ny,dd)=(grid.ny(),self.d*self.d);let half=ny/2+1;
        let(mut ket,mut bra,mut y)=(std::mem::take(&mut self.ket),std::mem::take(&mut self.bra),std::mem::take(&mut self.kicked));
        for x in [&mut ket,&mut bra,&mut y]{x.resize(self.state.len(),Z::default());}
        self.difference(&mut ket);
        let kappa:Vec<f64>=(0..ny).map(|k|signed(k,ny) as f64*self.dkappa).collect();
        let(fwd,inv,source,fields,transposed,kicks,exact)=(&self.plans[2],&self.plans[3],&self.source,&self.fields,&self.transposed,&self.kicks,&self.exact_kicks);
        y.par_chunks_mut(f*ny).zip(self.state.par_chunks(f*ny)).zip(ket.par_chunks_mut(f*ny).zip(bra.par_chunks_mut(f*ny))).enumerate().for_each(|(i,((out,rho),(k1,k2)))|{
            let mut scratch=vec![Z::default();fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len())];
            out.copy_from_slice(rho);
            // κ̂₁ρ = κρ + Qρ/2 and ρκ̂₂ = κρ − Qρ/2 (k1 holds Qρ/2).
            for(u,(a,b))in k1.chunks_mut(ny).zip(k2.chunks_mut(ny)).enumerate(){for k in 0..ny{let(q,r)=(a[k],rho[u*ny+k]*kappa[k]);a[k]=r+q;b[k]=r-q;}}
            for row in out.chunks_mut(ny).chain(k1.chunks_mut(ny)).chain(k2.chunks_mut(ny)){inv.process_with_scratch(row,&mut scratch);}
            let scale=1./ny as f64;
            let mut x=[vec![0.;d],vec![0.;d],vec![0.;d],vec![0.;d],vec![0.;d],vec![0.;d]];let(mut or,mut oi)=(vec![0.;d],vec![0.;d]);
            // At y and its pair −y: ρ, κ̂₁ρ and ρκ̂₂, the elements below the diagonal from the fields at −y
            // (κ̂₁ρ's from ρκ̂₂'s and ρκ̂₂'s from κ̂₁ρ's); L1 there by 𝓐, 𝓑, 𝓒; its pair by Hermiticity.
            for j in 0..half{
                let m=(ny-j)%ny;
                for e in 0..d{let(u,conjugate)=source[e];
                    let(a,b,c)=if conjugate{(out[u*ny+m].conj(),k2[u*ny+m].conj(),k1[u*ny+m].conj())}else{(out[u*ny+j],k1[u*ny+j],k2[u*ny+j])};
                    (x[0][e],x[1][e],x[2][e],x[3][e],x[4][e],x[5][e])=(a.re*scale,a.im*scale,b.re*scale,b.im*scale,c.re*scale,c.im*scale);}
                or.fill(0.);oi.fill(0.);
                if j==0{let m3=&exact[i*3*dd..(i+1)*3*dd];
                    for(k,mat)in m3.chunks(dd).enumerate(){let(xr,xi)=(&x[2*k],&x[2*k+1]);
                        for r in 0..d{let row=&mat[r*d..(r+1)*d];let(mut ar,mut ai)=(0.,0.);for c in 0..d{ar+=row[c].re*xr[c]-row[c].im*xi[c];ai+=row[c].re*xi[c]+row[c].im*xr[c];}or[r]+=ar;oi[r]+=ai;}}
                }else{let p=(i*half+j)*6*dd;for k in 0..3{apply(&kicks[p+k*2*dd..p+(k+1)*2*dd],d,&x[2*k],&x[2*k+1],&mut or,&mut oi);}}
                for(u,&e)in fields.iter().enumerate(){let t=transposed[e];out[u*ny+j]=Z::new(or[e],oi[e]);if m!=j{out[u*ny+m]=Z::new(or[t],-oi[t]);}}
            }
            for row in out.chunks_mut(ny){fwd.process_with_scratch(row,&mut scratch);}
        });
        self.state.par_iter_mut().zip(y.par_iter()).for_each(|(s,y)|*s+=y*span);
        (self.ket,self.bra,self.kicked)=(ket,bra,y);
    }

    /// The readout of the synchronized state at t.
    fn readout(&self,t:f64)->Reading{
        let(f,ng,grid)=(self.fields.len(),self.ng,self.grid);let(ny,nz)=(grid.ny(),grid.nz);
        let mut p_kappa=vec![0.;ny];let mut p_z=vec![0.;nz];let mut populations=vec![0.;STATES];
        for i in 0..nz{
            let block=&self.state[i*f*ny..(i+1)*f*ny];
            let field:Vec<Z>=(0..f).map(|u|block[u*ny..(u+1)*ny].iter().sum::<Z>()/ny as f64).collect();
            let rho:Vec<Z>=self.source.iter().map(|&(u,conjugate)|if conjugate{field[u].conj()}else{field[u]}).collect();
            for &u in &self.diagonal_fields{for k in 0..ny{p_kappa[k]+=block[u*ny+k].re;}p_z[i]+=field[u].re;}
            let n=signed(i,nz)*grid.stride();
            if let Some(local)=&self.site(n).local{
                let mut m=Dense::zeros(ng,ng);for(e,&(a,b))in self.elements.iter().enumerate(){m[(a,b)]=Complex::new(rho[e].re,rho[e].im);}
                for(x,p)in populations.iter_mut().zip(self.ground.populations(local,&m)){*x+=p*self.dz;}
            }
        }
        let norm=self.dz*self.dy/TAU;for x in p_kappa.iter_mut(){*x*=norm;}
        let trace:f64=p_z.iter().sum::<f64>()*self.dz;
        for x in p_kappa.iter_mut().chain(p_z.iter_mut()).chain(populations.iter_mut()){*x/=trace;}
        let(kappa,p_kappa)=sorted(&p_kappa,self.dkappa);let(z,p_z)=sorted(&p_z,self.dz);
        let mean=kappa.iter().zip(&p_kappa).map(|(k,p)|k*p).sum::<f64>()*self.dkappa;
        let square=kappa.iter().zip(&p_kappa).map(|(k,p)|k*k*p).sum::<f64>()*self.dkappa;
        let spread=z.iter().zip(&p_z).map(|(x,p)|x*x*p).sum::<f64>()*self.dz;
        Reading{t,populations,trace:trace/self.trace0,kappa,p_kappa,z,p_z,mean,square,spread}
    }

    fn outputs(&self,r:&Reading)->Outputs{
        let variance=r.square-r.mean*r.mean;
        Outputs{populations:r.populations.clone(),temperature:self.temperature(variance),velocity:self.hbar_over_m*r.mean,
            occupation:self.occupation(r.square,r.spread),loss:1.-r.trace}
    }
    /// T (uK) of a variance of κ (rad²/um²): m (ħ/m)² Var(κ)/k_B, ħ/m in m²/s.
    fn temperature(&self,variance:f64)->f64{self.mass*self.hbar_over_m.powi(2)*variance/BOLTZMANN*1e6}
    /// n̄ from ⟨κ²⟩ and ⟨x²⟩.
    fn occupation(&self,square:f64,spread:f64)->f64{(self.hbar_over_m*square/2.+self.omega*self.omega*spread/(2.*self.hbar_over_m))/self.omega-0.5}
}

/// out += M x at one point: M d × d complex in f32 (real then imaginary parts, row-major), x and out
/// f64 with their real and imaginary parts apart. AVX2 and FMA when the processor has them.
fn apply(m:&[f32],d:usize,xr:&[f64],xi:&[f64],or:&mut [f64],oi:&mut [f64]){
    #[cfg(target_arch="x86_64")]
    {static WIDE:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
        if *WIDE.get_or_init(||is_x86_feature_detected!("avx2")&&is_x86_feature_detected!("fma")){
            // SAFETY: the processor has AVX2 and FMA; the slices hold d × d (twice) and d values.
            unsafe{return apply_wide(m,d,xr,xi,or,oi);}}}
    let(mr,mi)=m.split_at(d*d);
    for r in 0..d{let(er,ei)=(&mr[r*d..(r+1)*d],&mi[r*d..(r+1)*d]);let(mut ar,mut ai)=(0.,0.);
        for c in 0..d{let(a,b)=(er[c] as f64,ei[c] as f64);ar+=a*xr[c]-b*xi[c];ai+=a*xi[c]+b*xr[c];}
        or[r]+=ar;oi[r]+=ai;}
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2,fma")]
unsafe fn apply_wide(m:&[f32],d:usize,xr:&[f64],xi:&[f64],or:&mut [f64],oi:&mut [f64]){
    use std::arch::x86_64::*;
    assert!(m.len()>=2*d*d&&xr.len()>=d&&xi.len()>=d&&or.len()>=d&&oi.len()>=d);
    let(full,dd)=(d/4*4,d*d);
    let sum=|v:__m256d|{let s=_mm_add_pd(_mm256_castpd256_pd128(v),_mm256_extractf128_pd(v,1));_mm_cvtsd_f64(_mm_add_sd(s,_mm_unpackhi_pd(s,s)))};
    for r in 0..d{
        let(er,ei)=(m.as_ptr().add(r*d),m.as_ptr().add(dd+r*d));
        let(mut ar,mut ai)=(_mm256_setzero_pd(),_mm256_setzero_pd());
        let mut c=0;
        while c<full{
            let(a,b)=(_mm256_cvtps_pd(_mm_loadu_ps(er.add(c))),_mm256_cvtps_pd(_mm_loadu_ps(ei.add(c))));
            let(x,y)=(_mm256_loadu_pd(xr.as_ptr().add(c)),_mm256_loadu_pd(xi.as_ptr().add(c)));
            ar=_mm256_fnmadd_pd(b,y,_mm256_fmadd_pd(a,x,ar));ai=_mm256_fmadd_pd(b,x,_mm256_fmadd_pd(a,y,ai));
            c+=4;
        }
        let(mut sr,mut si)=(sum(ar),sum(ai));
        while c<d{let(a,b)=(*er.add(c) as f64,*ei.add(c) as f64);sr+=a*xr[c]-b*xi[c];si+=a*xi[c]+b*xr[c];c+=1;}
        or[r]+=sr;oi[r]+=si;
    }
}

/// Rustfft plans: along z forward and inverse, along κ forward and inverse.
fn plans(nz:usize,ny:usize)->[Arc<dyn Fft<f64>>;4]{
    let mut planner=FftPlanner::new();
    [planner.plan_fft_forward(nz),planner.plan_fft_inverse(nz),planner.plan_fft_forward(ny),planner.plan_fft_inverse(ny)]
}

/// An FFT-order density as (grid, density) in increasing order.
fn sorted(values:&[f64],step:f64)->(Vec<f64>,Vec<f64>){
    let n=values.len();let grid=(0..n).map(|m|(m as i64-(n/2) as i64) as f64*step).collect();
    let density=(0..n).map(|m|values[unsigned(m as i64-(n/2) as i64,n)]).collect();(grid,density)
}

/// exp(a) − 1 of a d × d matrix (row-major): Taylor to order 14 of a scaled to norm ≤ 1/2, then squared.
fn expm1(a:&[Z],d:usize)->Vec<Z>{
    let norm=(0..d).map(|c|(0..d).map(|r|a[r*d+c].norm()).sum::<f64>()).fold(0.,f64::max);
    let squarings=if norm>0.5{(norm/0.5).log2().ceil() as i32}else{0};let scale=0.5_f64.powi(squarings);
    let x:Vec<Z>=a.iter().map(|z|z*scale).collect();
    // exp(x) − 1 = x (1 + x/2 (1 + x/3 (… ))).
    let mut p=vec![Z::default();d*d];
    for order in (2..=14).rev(){let mut q=multiply(&x,&p,d);for z in q.iter_mut(){*z/=order as f64;}for i in 0..d{q[i*d+i]+=1.;}p=q;}
    let mut m=multiply(&x,&p,d);
    // exp(2y) − 1 = (exp(y) − 1)(exp(y) − 1 + 2).
    for _ in 0..squarings{let mut plus=m.clone();for i in 0..d{plus[i*d+i]+=2.;}m=multiply(&m,&plus,d);}
    m
}
fn multiply(a:&[Z],b:&[Z],d:usize)->Vec<Z>{
    let mut out=vec![Z::default();d*d];
    for i in 0..d{for k in 0..d{let x=a[i*d+k];if x==Z::default(){continue;}let(row,o)=(&b[k*d..(k+1)*d],&mut out[i*d..(i+1)*d]);for j in 0..d{o[j]+=x*row[j];}}}
    out
}

/// The c at which a Gaussian's κ²-weighted tail beyond c σ is `share`: 2(c φ(c) + Q(c)) = share.
fn gaussian_tail(share:f64)->f64{
    let tail=|c:f64|2.*(c*(-c*c/2.).exp()/(TAU).sqrt()+0.5*erfc(c/2_f64.sqrt()));
    let(mut lo,mut hi)=(0.,40.);for _ in 0..200{let mid=0.5*(lo+hi);if tail(mid)>share{lo=mid;}else{hi=mid;}}hi
}
/// erfc to 1.2e-7 relative (Numerical Recipes, Chebyshev fit).
fn erfc(x:f64)->f64{
    let z=x.abs();let t=1./(1.+0.5*z);
    let r=t*(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277))))))))).exp();
    if x>=0.{r}else{2.-r}
}

/// A snapshot of the outputs' linear sector: densities and populations times the trace (relative
/// to the start), and the trace.
#[derive(Clone,Debug)]
struct Snapshot{t:f64,kappa:Vec<f64>,p_kappa:Vec<f64>,z:Vec<f64>,p_z:Vec<f64>,populations:Vec<f64>,trace:f64}
impl Snapshot{
    fn of(r:&Reading)->Snapshot{let s=|v:&[f64]|v.iter().map(|x|x*r.trace).collect();
        Snapshot{t:r.t,kappa:r.kappa.clone(),p_kappa:s(&r.p_kappa),z:r.z.clone(),p_z:s(&r.p_z),populations:s(&r.populations),trace:r.trace}}
    /// Its features on a basis: probabilities per κ bin and per z bin, the populations, the trace.
    fn features(&self,kappa:&[f64],z:&[f64])->Vec<f64>{
        let step=|g:&[f64]|g[1]-g[0];let(dk,dz)=(step(kappa),step(z));
        let mut f:Vec<f64>=resample(&self.kappa,&self.p_kappa,kappa).iter().map(|p|p*dk).collect();
        f.extend(resample(&self.z,&self.p_z,z).iter().map(|p|p*dz));f.extend(&self.populations);f.push(self.trace);f
    }
}
/// Linear interpolation of a density on an increasing grid at other points; zero outside it.
fn resample(grid:&[f64],values:&[f64],at:&[f64])->Vec<f64>{
    let(first,last,step)=(grid[0],grid[grid.len()-1],grid[1]-grid[0]);
    at.iter().map(|&x|{if x<first-1e-12*step.abs()||x>last+1e-12*step.abs(){return 0.;}
        let u=((x-first)/step).max(0.);let i=(u.floor() as usize).min(grid.len()-1);let f=u-i as f64;
        if i+1<grid.len(){values[i]*(1.-f)+values[i+1]*f}else{values[i]}}).collect()
}

/// The step-doubling estimate: per output, dt's error, |O_2dt − O_dt|/3 (Strang is second order).
#[derive(Clone,Debug,Default)]
struct StepError{temperature:f64,occupation:f64,populations:f64,loss:f64}
impl StepError{
    fn of(a:&Outputs,b:&Outputs)->StepError{
        StepError{temperature:(a.temperature-b.temperature).abs()/3.,occupation:(a.occupation-b.occupation).abs()/3.,
            populations:a.populations.iter().zip(&b.populations).map(|(p,q)|(p-q).abs()).fold(0.,f64::max)/3.,loss:(a.loss-b.loss).abs()/3.}
    }
    fn max(&self,o:&StepError)->StepError{
        StepError{temperature:self.temperature.max(o.temperature),occupation:self.occupation.max(o.occupation),populations:self.populations.max(o.populations),loss:self.loss.max(o.loss)}
    }
}

impl<'a> Engine<'a>{
    /// One radial period from the synchronized state at t (`double`: steps/2 steps of 2 dt, E applied
    /// twice), kicked eight times, read out at the boundaries `wanted` asks for and at its end;
    /// returns the y edge share measured in its last step.
    fn period_run(&mut self,t:f64,double:bool,wanted:&dyn Fn(f64,f64)->bool,readings:&mut Vec<Reading>,end:f64)->f64{
        let(n,tau,times)=if double{(self.steps/2,2.*self.dt,2)}else{(self.steps,self.dt,1)};
        let per_kick=(n/KICKS).max(1);let mut pending=tau/2.;let mut edge=0.;
        for s in 0..n{
            let now=t+s as f64*tau;if now>=end-1e-9*tau{break;}
            // The kick acts on the synchronized state.
            if s%per_kick==per_kick/2{if pending>tau*0.75{self.timed(0,|e|e.kinetic(tau/2.));}self.timed(2,|e|e.kick(per_kick as f64*tau));pending=tau/2.;}
            self.timed(0,|e|e.kinetic(pending));edge=self.timed(1,|e|e.local(times,s+1==n));
            let after=t+(s+1) as f64*tau;
            if s+1==n||after>=end-1e-9*tau||wanted(after,tau){self.timed(0,|e|e.kinetic(tau/2.));let r=self.timed(3,|e|e.readout(after));readings.push(r);pending=tau/2.;}else{pending=tau;}
        }
        edge
    }
    /// Runs f, its wall time added to the clock's slot (kinetic, local, kick, readout) when tracing.
    fn timed<T>(&mut self,slot:usize,f:impl FnOnce(&mut Self)->T)->T{
        if !tracing(){return f(self);}let start=Instant::now();let out=f(self);self.clock[slot]+=start.elapsed().as_secs_f64();out
    }
}

impl<'a> Engine<'a>{
    /// dt's projected error against its share of the tolerances.
    fn ratio(&self,bias:&StepError,tolerance:f64,at:&Outputs)->f64{
        [bias.temperature/(MOTION_TOLERANCE*at.temperature),bias.occupation/(MOTION_TOLERANCE*(at.occupation+0.5)),bias.populations/tolerance,bias.loss/tolerance]
            .iter().fold(0_f64,|a,&b|a.max(b))/SHARE
    }
    /// One radial period from the synchronized state at t, with dt or with 2 dt: its end's reading.
    fn one_period(&mut self,t:f64,double:bool)->Reading{let mut r=vec![];self.period_run(t,double,&|_,_|false,&mut r,f64::INFINITY);r.pop().unwrap()}
    fn set_steps(&mut self,steps:usize){self.steps=steps;self.dt=self.period/steps as f64;self.build();}

    /// The outputs a feature vector on the basis (κ, z) stands for, and its densities.
    fn predicted(&self,f:&[f64],kappa:&[f64],z:&[f64])->(Outputs,Vec<f64>,Vec<f64>){
        let(nk,nz)=(kappa.len(),z.len());let(pk,pz)=(&f[..nk],&f[nk..nk+nz]);let populations=&f[nk+nz..nk+nz+STATES];let trace=f[nk+nz+STATES];
        let(sk,sz)=(pk.iter().sum::<f64>(),pz.iter().sum::<f64>());
        let mean=kappa.iter().zip(pk).map(|(k,p)|k*p).sum::<f64>()/sk;let square=kappa.iter().zip(pk).map(|(k,p)|k*k*p).sum::<f64>()/sk;
        let spread=z.iter().zip(pz).map(|(x,p)|x*x*p).sum::<f64>()/sz;
        let(dk,dz)=(kappa[1]-kappa[0],z[1]-z[0]);
        (Outputs{populations:populations.iter().map(|p|p/trace).collect(),temperature:self.temperature(square-mean*mean),velocity:self.hbar_over_m*mean,
            occupation:self.occupation(square,spread),loss:1.-trace},pk.iter().map(|p|p/(sk*dk)).collect(),pz.iter().map(|p|p/(sz*dz)).collect())
    }

    /// The continuation from the snapshots of the last half of the run, to the end.
    fn continuation(&self,snapshots:&[Snapshot],last:&Reading,end:f64)->Result<Option<Prediction>,String>{
        let t=last.t;let used:Vec<&Snapshot>=snapshots.iter().filter(|s|s.t>=t/2.-1e-9).collect();
        let steps=((end-t)/self.period-1e-9).ceil().max(0.) as usize;
        if used.len()<3||steps==0{return Ok(None);}
        let x:Vec<Vec<f64>>=used.iter().map(|s|s.features(&last.kappa,&last.z)).collect();
        // A decomposition that does not resolve leaves this checkpoint without a continuation.
        let future=match dmd(&x,steps){Ok(f)=>f,Err(e)=>{trace!("t {t:.1} continuation: {e}");return Ok(None);}};
        let mut prediction=Prediction{t,times:vec![],outputs:vec![],errors:vec![],kappa:last.kappa.clone(),z:last.z.clone(),p_kappa:vec![],p_z:vec![]};
        for(k,f)in future.iter().enumerate(){
            let(o,pk,pz)=self.predicted(f,&last.kappa,&last.z);
            prediction.times.push(t+(k+1) as f64*self.period);prediction.outputs.push(o);prediction.p_kappa.push(pk);prediction.p_z.push(pz);
        }
        Ok(Some(prediction))
    }
}

/// The largest change between two outputs, relative to the tolerances: populations and the loss
/// absolute, T and n̄ (n̄ + ½, the energy in ħω) relative.
fn change(a:&Outputs,b:&Outputs,tolerance:f64)->f64{
    let populations=a.populations.iter().zip(&b.populations).map(|(p,q)|(p-q).abs()).fold(0.,f64::max)/tolerance;
    let temperature=(a.temperature-b.temperature).abs()/(MOTION_TOLERANCE*b.temperature.abs().max(f64::MIN_POSITIVE));
    let occupation=(a.occupation-b.occupation).abs()/(MOTION_TOLERANCE*(b.occupation+0.5).abs().max(f64::MIN_POSITIVE));
    [populations,temperature,occupation,(a.loss-b.loss).abs()/tolerance].iter().fold(0_f64,|x,&y|x.max(y))
}
/// The population error of a change: the largest of every state's and every group's.
fn population_change(a:&[f64],b:&[f64],groups:&[Vec<usize>])->f64{
    let states=a.iter().zip(b).map(|(p,q)|(p-q).abs()).fold(0.,f64::max);
    groups.iter().map(|g|g.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs()).fold(states,f64::max)
}
/// The states grouped by manifold and F (`manifold-F-m`).
fn groups(states:&[String])->Vec<Vec<usize>>{
    let mut keys:Vec<String>=vec![];let mut out:Vec<Vec<usize>>=vec![];
    for(i,s)in states.iter().enumerate(){let key=s.rsplitn(2,'-').nth(1).unwrap_or(s).to_string();
        match keys.iter().position(|k|*k==key){Some(g)=>out[g].push(i),None=>{keys.push(key);out.push(vec![i]);}}}
    out
}
/// The samples with distributions: twenty-one evenly spread, the first and the last among them.
fn listed(n:usize)->Vec<usize>{let mut v:Vec<usize>=(0..=20).map(|i|i*(n-1)/20).collect();v.dedup();v}

fn interpolate(a:&Outputs,b:&Outputs,f:f64)->Outputs{
    let mix=|x:f64,y:f64|x+(y-x)*f;
    Outputs{populations:a.populations.iter().zip(&b.populations).map(|(x,y)|mix(*x,*y)).collect(),temperature:mix(a.temperature,b.temperature),
        velocity:mix(a.velocity,b.velocity),occupation:mix(a.occupation,b.occupation),loss:mix(a.loss,b.loss)}
}

/// One-sided Jacobi SVD of the columns: (U's columns, σ, V's columns) for σ > rcond σ_max, decreasing.
fn svd(mut a:Vec<Vec<f64>>,rcond:f64)->(Vec<Vec<f64>>,Vec<f64>,Vec<Vec<f64>>){
    let n=a.len();let mut v:Vec<Vec<f64>>=(0..n).map(|i|(0..n).map(|j|if i==j{1.}else{0.}).collect()).collect();
    for _ in 0..80{
        let mut rotated=false;
        for p in 0..n{for q in p+1..n{
            let(alpha,beta,gamma)=(dot(&a[p],&a[p]),dot(&a[q],&a[q]),dot(&a[p],&a[q]));
            if gamma==0.||gamma.abs()<=1e-15*(alpha*beta).sqrt(){continue;}
            rotated=true;let zeta=(beta-alpha)/(2.*gamma);let t=zeta.signum()/(zeta.abs()+(1.+zeta*zeta).sqrt());let c=1./(1.+t*t).sqrt();
            rotate(&mut a,p,q,c,c*t);rotate(&mut v,p,q,c,c*t);
        }}
        if !rotated{break;}
    }
    let sigma:Vec<f64>=a.iter().map(|x|dot(x,x).sqrt()).collect();let mut order:Vec<usize>=(0..n).collect();order.sort_by(|&i,&j|sigma[j].total_cmp(&sigma[i]));
    let top=sigma[order[0]];let kept:Vec<usize>=order.into_iter().filter(|&i|sigma[i]>rcond*top&&sigma[i]>0.).collect();
    (kept.iter().map(|&i|a[i].iter().map(|x|x/sigma[i]).collect()).collect(),kept.iter().map(|&i|sigma[i]).collect(),kept.iter().map(|&i|v[i].clone()).collect())
}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn rotate(m:&mut [Vec<f64>],p:usize,q:usize,c:f64,s:f64){
    let(left,right)=m.split_at_mut(q);
    for(a,b)in left[p].iter_mut().zip(right[0].iter_mut()){let(u,w)=(*a,*b);*a=c*u-s*w;*b=s*u+c*w;}
}
/// The dynamic mode decomposition of equally spaced snapshots x_0 … x_n: x_{n+k} for k = 1..=steps,
/// growing modes put on the unit circle. The reduced operator Ã = Uᵀ X1 V Σ⁻¹ is diagonalized as
/// Σ^{-1/2} Uᵀ X1 V Σ^{-1/2}, similar to it and balanced; when that does not resolve, the smallest
/// singular directions are dropped.
fn dmd(x:&[Vec<f64>],steps:usize)->Result<Vec<Vec<f64>>,String>{
    // Singular directions below √ε σ_max are cut: the reduced operator's condition stays within 1/√ε.
    let n=x.len();let(u,sigma,v)=svd(x[..n-1].to_vec(),f64::EPSILON.sqrt());let m=x[0].len();
    // Uᵀ X1 V, once.
    let full=sigma.len();let mut core=vec![0.;full*full];
    for j in 0..full{let mut column=vec![0.;m];for(i,w)in v[j].iter().enumerate(){for(c,y)in column.iter_mut().zip(&x[i+1]){*c+=w*y;}}
        for i in 0..full{core[i*full+j]=dot(&u[i],&column);}}
    let mut r=full;
    loop{
        let a:Vec<Z>=(0..r*r).map(|t|{let(i,j)=(t/r,t%r);Z::new(core[i*full+j]/(sigma[i]*sigma[j]).sqrt(),0.)}).collect();
        match eigen(&a,r){
            Ok((values,vectors))=>{
                let mu:Vec<Complex>=values.iter().map(|z|{let w=Complex::new(z.re,z.im);let size=w.abs();if size>1.{w*(1./size)}else{w}}).collect();
                let modes=Dense::from_fn(r,r,|i,k|Complex::new(vectors[i*r+k].re,vectors[i*r+k].im));
                // W = Σ^{1/2} W_s; b = W⁻¹ Uᵀ x_n.
                let mut b:Vec<Complex>=(0..r).map(|i|Complex::new(dot(&u[i],&x[n-1])/sigma[i].sqrt(),0.)).collect();
                let Some(lu)=Lu::new(&modes)else{if r<=2{return Err("The continuation's modes are singular".into());}r=r*3/4;continue;};lu.solve(&mut b);
                let mut power=b;let mut out=vec![];
                for _ in 0..steps{
                    for(p,z)in power.iter_mut().zip(&mu){*p=*p**z;}
                    let mut f=vec![0.;m];
                    for i in 0..r{let c:f64=(0..r).map(|k|(modes[(i,k)]*power[k]).re).sum::<f64>()*sigma[i].sqrt();for(y,w)in f.iter_mut().zip(&u[i]){*y+=c*w;}}
                    out.push(f);
                }
                return Ok(out);
            }
            Err(e)=>{if r<=2{return Err(e);}r=r*3/4;}
        }
    }
}

/// The eigenvalues and right eigenvectors of a complex r × r matrix (row-major), without LAPACK, so that the
/// continuation is the same in every build: Householder reduction to Hessenberg form, the shifted QR iteration
/// (Wilkinson shifts, exceptional ones every 11th) to the Schur form A = Q T Qᴴ, T's eigenvectors by
/// back-substitution. The vectors are the columns of the r × r row-major result, each of unit 2-norm.
fn eigen(a:&[Z],r:usize)->Result<(Vec<Z>,Vec<Z>),String>{
    let zero=Z::default();let mut h=a.to_vec();let mut q=vec![zero;r*r];for i in 0..r{q[i*r+i]=Z::new(1.,0.);}
    // Hessenberg: the reflector I − 2vvᴴ/vᴴv maps the column below the subdiagonal to α e1, |α| its norm.
    for k in 0..r.saturating_sub(2){
        let size=(k+1..r).map(|i|h[i*r+k].norm_sqr()).sum::<f64>().sqrt();if size==0.{continue;}
        let x0=h[(k+1)*r+k];let alpha=-(if x0.norm()>0.{x0/x0.norm()}else{Z::new(1.,0.)})*size;
        let mut v:Vec<Z>=(k+1..r).map(|i|h[i*r+k]).collect();v[0]-=alpha;let vv:f64=v.iter().map(|z|z.norm_sqr()).sum();if vv==0.{continue;}
        for j in k..r{let f=v.iter().enumerate().map(|(l,vl)|vl.conj()*h[(k+1+l)*r+j]).sum::<Z>()*(2./vv);for(l,vl)in v.iter().enumerate(){h[(k+1+l)*r+j]-=f*vl;}}
        for m in [&mut h,&mut q]{for i in 0..r{let f=v.iter().enumerate().map(|(l,vl)|m[i*r+k+1+l]*vl).sum::<Z>()*(2./vv);for(l,vl)in v.iter().enumerate(){m[i*r+k+1+l]-=f*vl.conj();}}}
        for i in k+2..r{h[i*r+k]=zero;}
    }
    // QR steps on the unreduced block [lo, hi], the rotations applied across the whole matrix (the Schur form).
    let scale=h.iter().fold(0_f64,|m,z|m.max(z.norm()));
    let(mut hi,mut since,mut total)=(r.saturating_sub(1),0,0);
    while hi>0{
        let mut lo=hi;
        while lo>0{let s=h[(lo-1)*r+lo-1].norm()+h[lo*r+lo].norm();if h[lo*r+lo-1].norm()<=f64::EPSILON*if s>0.{s}else{scale}{h[lo*r+lo-1]=zero;break;}lo-=1;}
        if lo==hi{hi-=1;since=0;continue;}
        since+=1;total+=1;if total>60*r{return Err("The continuation's eigenvalues do not converge".into());}
        let(p,b,c,d)=(h[(hi-1)*r+hi-1],h[(hi-1)*r+hi],h[hi*r+hi-1],h[hi*r+hi]);
        let shift=if since%11==10{d+Z::new(c.re.abs()+if hi>=2{h[(hi-1)*r+hi-2].re.abs()}else{0.},0.)}
            else{let(mean,root)=((p+d)*0.5,(((p-d)*0.5).powi(2)+b*c).sqrt());let(u,w)=(mean+root,mean-root);if(u-d).norm()<=(w-d).norm(){u}else{w}};
        for k in lo..=hi{h[k*r+k]-=shift;}
        let mut rotations=Vec::with_capacity(hi-lo);
        for k in lo..hi{
            let(x,y)=(h[k*r+k],h[(k+1)*r+k]);let n=(x.norm_sqr()+y.norm_sqr()).sqrt();
            let(cs,sn)=if n==0.{(1.,zero)}else if x.norm()==0.{(0.,Z::new(1.,0.))}else{(x.norm()/n,x/x.norm()*y.conj()/n)};
            for j in k..r{let(u,w)=(h[k*r+j],h[(k+1)*r+j]);h[k*r+j]=u*cs+sn*w;h[(k+1)*r+j]=w*cs-sn.conj()*u;}
            rotations.push((cs,sn));
        }
        for(k,&(cs,sn))in(lo..hi).zip(&rotations){
            for i in 0..=(k+1).min(hi){let(u,w)=(h[i*r+k],h[i*r+k+1]);h[i*r+k]=u*cs+w*sn.conj();h[i*r+k+1]=w*cs-u*sn;}
            for i in 0..r{let(u,w)=(q[i*r+k],q[i*r+k+1]);q[i*r+k]=u*cs+w*sn.conj();q[i*r+k+1]=w*cs-u*sn;}
        }
        for k in lo..=hi{h[k*r+k]+=shift;}
    }
    // T x = λ_k x with x_k = 1 and x_j = 0 beyond k (denominators kept off zero, x rescaled before it overflows); v = Q x.
    let values:Vec<Z>=(0..r).map(|k|h[k*r+k]).collect();let small=f64::EPSILON*scale.max(f64::MIN_POSITIVE);
    let mut vectors=vec![zero;r*r];
    for k in 0..r{
        let mut x=vec![zero;k+1];x[k]=Z::new(1.,0.);
        for j in(0..k).rev(){
            let sum:Z=(j+1..=k).map(|i|h[j*r+i]*x[i]).sum();let mut den=h[j*r+j]-values[k];if den.norm()<small{den=Z::new(small,0.);}
            x[j]=-sum/den;let size=x[j].norm();if size>1e150{for z in x.iter_mut(){*z/=size;}}
        }
        let v:Vec<Z>=(0..r).map(|i|(0..=k).map(|j|q[i*r+j]*x[j]).sum()).collect();let size=v.iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();
        for i in 0..r{vectors[i*r+k]=v[i]/size;}
    }
    Ok((values,vectors))
}

/// The samples filled so far: their outputs, and the readings of the listed ones.
struct Record{samples:Vec<Option<Outputs>>,shapes:HashMap<usize,Reading>}

impl<'a> Engine<'a>{
    /// The state (and `others`) on a new grid: the momentum range zero-padded in κ, the momentum step
    /// halved by zero-padding in y; then the grid's operators.
    fn resize(&mut self,new:Grid,others:&mut [&mut Vec<Z>])->Result<(),String>{
        let old=self.grid;let state=std::mem::take(&mut self.state);self.state=self.transform(state,old,new);
        for other in others.iter_mut(){let state=std::mem::take(&mut **other);**other=self.transform(state,old,new);}
        self.regrid(new)
    }
    /// A state on the grid `old` on the grid `new` (the momentum range larger or smaller, the momentum
    /// step equal or finer, the same box).
    fn transform(&self,mut state:Vec<Z>,old:Grid,new:Grid)->Vec<Z>{
        let d=self.fields.len();
        if new.p!=old.p{
            // The κ range padded or cut, S = Σ_y B e^{−iκy} over the new y lattice scaled to keep B.
            let(ny,width)=(old.ny(),new.p*old.m2);let(scale,kept)=(width as f64/ny as f64,ny.min(width));
            let mut resized=vec![Z::default();d*old.nz*width];
            for row in 0..old.nz*d{for s in -((kept/2) as i64)..(kept/2) as i64{resized[row*width+unsigned(s,width)]=state[row*ny+unsigned(s,ny)]*scale;}}
            state=resized;
        }
        if new.m2!=old.m2{
            let(ny,wide)=(new.p*old.m2,new.ny());
            let mut planner=FftPlanner::new();let(inv,fwd)=(planner.plan_fft_inverse(ny),planner.plan_fft_forward(wide));
            let mut finer=vec![Z::default();d*old.nz*wide];
            finer.par_chunks_mut(wide).zip(state.par_chunks(ny)).for_each(|(to,from)|{
                let mut b=from.to_vec();inv.process(&mut b);
                for j in 0..ny{to[if j<ny/2{j}else{j+wide-ny}]=b[j]/ny as f64;}
                fwd.process(to);
            });
            state=finer;
        }
        state
    }

    /// The grid the last reading asks for: the momentum range and the momentum step grow while
    /// their tails exceed their share (`edge`: the y box's, from the last step); the momentum range
    /// shrinks ×2/3 when the smaller one would hold the tail with the margin again (the share beyond
    /// EDGE² of it), so that it does not grow back at once.
    fn wanted_grid(&self,last:&Reading,edge:f64)->Grid{
        let tail=|g:&[f64],p:&[f64],cut:f64|{let(mut out,mut all)=(0.,0.);for(x,q)in g.iter().zip(p){let w=x*x*q.abs();all+=w;if x.abs()>cut{out+=w;}}if all>0.{out/all}else{0.}};
        let mut grid=self.grid;
        let p_cap=8*smooth((self.escape_momentum/self.k/8.).ceil() as usize);
        let smaller=8*smooth_below((grid.p as f64/1.5/8.).floor() as usize);
        if tail(&last.kappa,&last.p_kappa,EDGE*grid.p as f64*self.k)>SHARE*MOTION_TOLERANCE&&grid.p<p_cap{grid.p=(8*smooth((1.5*grid.p as f64/8.).ceil() as usize)).min(p_cap);}
        else if smaller<grid.p&&tail(&last.kappa,&last.p_kappa,EDGE*EDGE*smaller as f64*self.k)<=SHARE*MOTION_TOLERANCE{grid.p=smaller;}
        if edge>SHARE*MOTION_TOLERANCE&&PI*grid.m2 as f64/(2.*self.k)<self.edge{grid.m2*=2;}
        grid
    }

    /// Fills the samples between two consecutive readings.
    fn fill(&self,record:&mut Record,times:&[f64],listed:&[usize],a:&Reading,b:&Reading){
        let(x,y)=(self.outputs(a),self.outputs(b));let span=b.t-a.t;
        for(s,&t)in times.iter().enumerate(){
            if record.samples[s].is_some()||t>b.t+1e-9||t<a.t-1e-9{continue;}
            let f=if span>0.{((t-a.t)/span).clamp(0.,1.)}else{0.};
            record.samples[s]=Some(interpolate(&x,&y,f));
            if listed.contains(&s){record.shapes.insert(s,if f<0.5{a.clone()}else{b.clone()});}
        }
    }

    /// The solution: the samples simulated, then the continuation (or, without one, the last
    /// outputs held, their error unknown); the distributions on the current grid.
    /// `bias`: dt's population error (the step-doubling check's).
    fn assemble(&self,request:&Request,record:&Record,last:&Reading,prediction:Option<&Prediction>,bias:f64,listed:&[usize],method:&str)->Solution{
        let times=&request.times_us;let held=self.outputs(last);
        let mut rows:Vec<Outputs>=vec![];let mut errors=vec![];
        for(s,&t)in times.iter().enumerate(){
            if let Some(o)=&record.samples[s]{rows.push(o.clone());errors.push(bias);continue;}
            match prediction{
                Some(p)=>{
                    let k=p.times.partition_point(|&x|x<t-1e-9).min(p.times.len()-1);
                    let(t0,o0,e0)=if k==0{(p.t,&held,0.)}else{(p.times[k-1],&p.outputs[k-1],p.errors[k-1])};
                    let f=((t-t0)/(p.times[k]-t0)).clamp(0.,1.);
                    rows.push(interpolate(o0,&p.outputs[k],f));errors.push(e0.max(p.errors[k])+bias);
                }
                None=>{rows.push(held.clone());errors.push(1.);}
            }
        }
        let(kappa,z)=(&last.kappa,&last.z);let(dk,dz)=(kappa[1]-kappa[0],z[1]-z[0]);
        let probabilities=|density:Vec<f64>,step:f64|{let mut p:Vec<f64>=density.iter().map(|x|(x*step).max(0.)).collect();let s:f64=p.iter().sum();for x in p.iter_mut(){*x/=s;}p};
        let(mut pk,mut pz)=(vec![],vec![]);
        for &s in listed{
            let t=times[s];
            let(a,b)=match(record.shapes.get(&s),prediction){
                (Some(r),_)=>(resample(&r.kappa,&r.p_kappa,kappa),resample(&r.z,&r.p_z,z)),
                (None,Some(p))=>{let k=p.times.partition_point(|&x|x<t-1e-9).min(p.times.len()-1);
                    if k==0&&t-p.t<p.times[0]-t{(last.p_kappa.clone(),last.p_z.clone())}else{(resample(&p.kappa,&p.p_kappa[k],kappa),resample(&p.z,&p.p_z[k],z))}}
                (None,None)=>(last.p_kappa.clone(),last.p_z.clone()),
            };
            pk.push(probabilities(a,dk));pz.push(probabilities(b,dz));
        }
        let mut loss=0_f64;
        Solution{
            populations:rows.iter().map(|o|o.populations.clone()).collect(),sample_errors:errors,
            temperature_uk:rows.iter().map(|o|o.temperature.max(0.)).collect(),mean_velocity_mps:rows.iter().map(|o|o.velocity).collect(),
            trap_frequency_khz:self.omega/TAU*1e3,mean_occupation:rows.iter().map(|o|o.occupation).collect(),
            // The loss never decreases: its round-off is clipped to the running maximum.
            loss_probability:rows.iter().map(|o|{loss=loss.max(o.loss.clamp(0.,1.));loss}).collect(),
            momentum:Distributions{grid:kappa.iter().map(|k|self.hbar_over_m*k).collect(),samples:listed.to_vec(),probabilities:pk},
            position:Distributions{grid:z.clone(),samples:listed.to_vec(),probabilities:pz},
            method:method.into(),reason:None,continued_from_us:prediction.map(|p|p.t).filter(|&t|t<times[times.len()-1]),
        }
    }

    /// Runs the request to its end, or until the continuation is accepted; every checkpoint's
    /// solution goes to `checkpoint` with the estimated remaining wall time (ms).
    ///
    /// dt starts where one period's error, from one period with 2 dt, is within its share. Over the
    /// first checkpoint interval a companion runs with 2 dt; at its end the two runs' outputs and
    /// continuations give dt's error over the run (Richardson): above the share dt halves and the
    /// run starts again, below an eighth of it dt doubles.
    pub fn run(&mut self,request:&Request,checkpoint:&mut dyn FnMut(&Solution,Option<f64>))->Result<Solution,String>{
        let started=Instant::now();
        let times=&request.times_us;let n=times.len();let end=times[n-1];let tolerance=request.tolerance;let period=self.period;
        let groups=groups(&request.states);let listed=listed(n);
        let(initial,initial_grid)=(self.state.clone(),self.grid);
        loop{
            let a=self.one_period(0.,false);self.state=initial.clone();let b=self.one_period(0.,true);self.state=initial.clone();
            let ratio=self.ratio(&StepError::of(&self.outputs(&a),&self.outputs(&b)),tolerance,&self.outputs(&a));
            trace!("start: steps/period {} one-period ratio {ratio:.3e} [{:.2} s]",self.steps,started.elapsed().as_secs_f64());
            if ratio>1.&&self.steps<FINEST{let steps=2*self.steps;self.set_steps(steps);continue;}
            break;
        }
        let wanted=|t:f64,tau:f64|{let i=times.partition_point(|&s|s<=t-tau);i<n&&times[i]<=t+tau};
        let method=format!("GMW-o1 in a box: the ground manifold with the excited states eliminated at every position (the far components folded), ρ(x1, x2) on a box in centre and momentum, Strang steps with exact per-point exponentials and the first order in the Doppler and recoil shifts as kicks; the grids grown and the time step chosen by step doubling for {MOTION_TOLERANCE} relative in T and n̄ and the population tolerance; continued by dynamic mode decomposition once its prediction agrees with the one made at half the run");
        'attempt:loop{
            let mut last=self.readout(0.);
            let mut record=Record{samples:vec![None;n],shapes:HashMap::new()};
            self.fill(&mut record,times,&listed,&last.clone(),&last.clone());
            // At t = 0 the populations are the initial state's: the excited states' adiabatic admixture,
            // ≈ the elimination's η, builds up within their lifetime, which the eliminated model does not resolve.
            if let Some(first)=record.samples[0].as_mut(){first.populations=(0..STATES).map(|i|request.initial_density[i*STATES+i].re).collect();}
            let mut snapshots=vec![(Snapshot::of(&last),self.outputs(&last))];
            let mut companion=Some((self.state.clone(),vec![(Snapshot::of(&last),self.outputs(&last))]));
            let(mut previous,mut accepted):(Option<Prediction>,bool)=(None,false);let mut changes:Vec<f64>=vec![];
            // Every checkpoint's prediction of the outputs (without its distributions).
            let mut history:Vec<Prediction>=vec![];
            let mut bias=StepError::default();let mut periods=0;
            // The cost model of the remaining time: the run's wall time per simulated us and grid point at
            // the current dt (the companion's apart), and a checkpoint's own.
            let(mut rate,mut rate_steps,mut per_checkpoint)=((0_f64,0_f64),self.steps,0_f64);
            while last.t<end-1e-9*period{
                let(began,from)=(Instant::now(),last.t);
                let mut fresh=vec![];let edge=self.period_run(last.t,false,&wanted,&mut fresh,end);
                for r in fresh{self.fill(&mut record,times,&listed,&last,&r);last=r;}
                if self.steps!=rate_steps{rate=(0.,0.);rate_steps=self.steps;}
                rate.0+=began.elapsed().as_secs_f64();rate.1+=(last.t-from)*self.grid.points() as f64;
                let t=last.t;periods+=1;snapshots.push((Snapshot::of(&last),self.outputs(&last)));
                if let Some((state,history))=companion.as_mut(){
                    if (t/period-(t/period).round()).abs()<1e-6{
                        let main=std::mem::replace(&mut self.state,std::mem::take(state));let r=self.one_period(t-period,true);
                        *state=std::mem::replace(&mut self.state,main);history.push((Snapshot::of(&r),self.outputs(&r)));
                    }
                }
                // The grid.
                let grid=self.wanted_grid(&last,edge);
                if periods%5==0{trace!("  clock: kinetic {:.2} s, local {:.2} s, kick {:.2} s, readout {:.2} s",self.clock[0],self.clock[1],self.clock[2],self.clock[3]);trace!("t {t:.1} T {:.3} nbar {:.3} loss {:.3e} grid {:?} edge {edge:.2e} [{:.2} s]",self.temperature(last.square-last.mean*last.mean),self.occupation(last.square,last.spread),1.-last.trace,self.grid,started.elapsed().as_secs_f64());}
                if grid!=self.grid{trace!("t {t:.1} regrid {:?} -> {grid:?} (edge {edge:.2e})",self.grid);
                    match companion.as_mut(){Some((state,_))=>self.resize(grid,&mut [state])?,None=>self.resize(grid,&mut [])?}}
                let finished=last.t>=end-1e-9*period;
                // The remaining wall time to `expected` (us): the periods at the measured rate, the checkpoints at theirs.
                let remaining=move|engine:&Self,expected:f64,t:f64|(expected-t).max(0.)*(rate.0/rate.1.max(f64::MIN_POSITIVE)*engine.grid.points() as f64
                    +per_checkpoint/(CHECKPOINT_PERIODS as f64*period));
                // An estimate from the start: after the first period, to the end.
                if periods==1&&!finished{let solution=self.assemble(request,&record,&last,None,1.,&listed,&method);checkpoint(&solution,Some(remaining(self,end,last.t)*1e3));}
                if !finished&&periods%CHECKPOINT_PERIODS!=0{continue;}
                let checkpoint_began=Instant::now();
                // A checkpoint: the continuation, compared with the previous one and with the outputs since.
                let continued=self.continuation(&snapshots.iter().map(|s|s.0.clone()).collect::<Vec<_>>(),&last,end)?;
                // dt's check at the first checkpoint (or the end): the companion's outputs against the run's
                // over the interval. (Their continuations from one interval differ by more than dt does.)
                if let Some((_,history))=companion.take(){
                    for((_,a),(_,b))in snapshots.iter().zip(&history){bias=bias.max(&StepError::of(a,b));}
                    // dt's error at the interval's end, past the start's transient: what the later intervals see.
                    let settled=snapshots.iter().zip(&history).last().map(|((_,a),(_,b))|StepError::of(a,b)).unwrap_or_default();
                    let ratio=self.ratio(&bias,tolerance,&self.outputs(&last));
                    trace!("t {t:.1} dt check: steps/period {} ratio {ratio:.3e} {bias:?} [{:.2} s]",self.steps,started.elapsed().as_secs_f64());
                    // Past the full tolerance the start is not resolved: dt halves and the run starts again.
                    // Within it the interval stands, its error as measured; the later intervals take dt from
                    // the error at its end, past the start's transient (2 dt has four times it).
                    if ratio*SHARE>1.&&self.steps<FINEST{
                        let steps=2*self.steps;self.state=initial.clone();if self.grid!=initial_grid{self.regrid(initial_grid)?;}self.set_steps(steps);continue 'attempt;
                    }
                    let ahead=self.ratio(&settled,tolerance,&self.outputs(&last));
                    if ahead>1.&&self.steps<FINEST{let steps=2*self.steps;trace!("  dt halved for the later intervals: steps/period {steps}");self.set_steps(steps);}
                    else if ahead*4.<=0.5&&self.steps>COARSEST{let steps=self.steps/2;trace!("  dt doubled for the later intervals: steps/period {steps} (settled {settled:?})");self.set_steps(steps);
                        bias=bias.max(&StepError{temperature:4.*settled.temperature,occupation:4.*settled.occupation,populations:4.*settled.populations,loss:4.*settled.loss});}
                    else{trace!("  dt kept: steps/period {} (settled ratio {ahead:.3e})",self.steps);}
                }
                if finished{break;}
                if let Some(mut prediction)=continued{
                    prediction.errors=vec![1.;prediction.times.len()];
                    // Against the prediction made at half the run, from the half before this one's window:
                    // two estimates from disjoint histories (consecutive checkpoints share most of theirs).
                    let half=history.iter().rev().find(|p|p.t<=t/2.+1e-9*period);
                    if let Some(p)=half{
                        let offset=((prediction.t-p.t)/period).round() as usize;let mut c=0_f64;
                        for(s,o)in snapshots.iter().filter(|s|s.0.t>p.t+1e-9*period){let k=((s.t-p.t)/period).round() as usize;if k>=1&&k<=p.outputs.len(){c=c.max(change(o,&p.outputs[k-1],tolerance));}}
                        for(k,o)in prediction.outputs.iter().enumerate(){if let Some(q)=p.outputs.get(k+offset){c=c.max(change(o,q,tolerance));prediction.errors[k]=population_change(&o.populations,&q.populations,&groups);}}
                        changes.push(c);accepted=c<=1.;trace!("t {t:.1} checkpoint change {c:.3e} against t {:.1} [{:.2} s]",p.t,started.elapsed().as_secs_f64());
                    }
                    history.push(Prediction{p_kappa:vec![],p_z:vec![],..prediction.clone()});
                    previous=Some(prediction);
                }
                if accepted{break;}
                let solution=self.assemble(request,&record,&last,previous.as_ref(),bias.populations,&listed,&method);
                // To the end, or to the acceptance the changes' geometric decrease since their largest projects
                // (they rise while the continuation learns the slow modes, then fall): a least-squares line
                // through their logarithms, from two changes after the largest on.
                let top=changes.iter().enumerate().fold(0,|m,(k,&c)|if c>changes[m]{k}else{m});
                let expected=if changes.len()>=top+3{
                    let ys:Vec<f64>=changes[top..].iter().map(|c|c.ln()).collect();let n=ys.len() as f64;
                    let(xm,ym)=((n-1.)/2.,ys.iter().sum::<f64>()/n);
                    let slope=ys.iter().enumerate().map(|(i,y)|(i as f64-xm)*(y-ym)).sum::<f64>()/ys.iter().enumerate().map(|(i,_)|(i as f64-xm).powi(2)).sum::<f64>();
                    if slope<0.{let now=ym+slope*(n-1.-xm);(t+(now.max(0.)/-slope).ceil()*CHECKPOINT_PERIODS as f64*period).min(end)}else{end}
                }else{end};
                per_checkpoint=checkpoint_began.elapsed().as_secs_f64();
                checkpoint(&solution,Some(remaining(self,expected,t)*1e3));
            }
            let prediction=if accepted{previous.as_ref()}else{None};
            return Ok(self.assemble(request,&record,&last,prediction,bias.populations,&listed,&method));
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use serde_json::Value;

    /// The app example's tweezer alone (tests/fixtures/tweezer_focus.json): |2,+2⟩ at 30 μK.
    fn tweezer(times:Vec<f64>)->Request{
        let f:Value=serde_json::from_str(include_str!("../tests/fixtures/tweezer_focus.json")).unwrap();
        let mut r=crate::ground::tests::request(&f["request"]);let i=r.states.iter().position(|s|s=="ground-2-2").unwrap();
        r.initial_density[i*STATES+i]=Complex::new(1.,0.);r.times_us=times;r.tolerance=1e-2;r
    }

    /// The app's tweezer PGC example (tests/fixtures/tweezer_pgc.json, TW2): |2,+2⟩ at 30 μK.
    fn tweezer_pgc()->Request{
        let f:Value=serde_json::from_str(include_str!("../tests/fixtures/tweezer_pgc.json")).unwrap();
        let mut r=crate::ground::tests::request(&f["request"]);let i=r.states.iter().position(|s|s=="ground-2-2").unwrap();
        r.initial_density[i*STATES+i]=Complex::new(1.,0.);r.times_us=vec![0.,10.];r.tolerance=1e-2;r
    }

    /// Half storage rests on the per-point operators carrying ρ's Hermiticity: at −y, E and 𝓐 are their
    /// mirror at y (element (a, b) ← (c, d) the conjugate of (b, a) ← (d, c)) and 𝓑 is 𝓒's; checked across
    /// the y box on lines through the centre and off it, with the cooling light's jumps and kicks.
    #[test]
    fn the_operators_at_minus_y_mirror_those_at_y(){
        let r=tweezer_pgc();let ground=Ground::new(&r).unwrap();let lapack=Lapack::load().unwrap();
        let engine=Engine::new(&ground,&r,&lapack).unwrap();
        let(ny,nz,d,t)=(engine.grid.ny(),engine.grid.nz,engine.d,&engine.transposed);
        assert!(!engine.kept.is_empty());
        for i in [0,1,nz/4,nz-3]{for j in(1..ny/2).step_by(5){
            let((e1,[a1,b1,c1]),(e2,[a2,b2,c2]))=(engine.point(i,j),engine.point(i,ny-j));
            for(at_minus,at_plus)in[(&e2,&e1),(&a2,&a1),(&b2,&c1),(&c2,&b1)]{
                let size=at_plus.iter().fold(0_f64,|m,z|m.max(z.norm()));assert!(size>0.);
                for e in 0..d{for f in 0..d{let(u,w)=(at_minus[e*d+f],at_plus[t[e]*d+t[f]].conj());
                    assert!((u-w).norm()<=1e-12*size,"point ({i}, {j}), element {e} <- {f}: {u} vs {w}");}}
            }
        }}
    }

    /// The continuation's eigen-solver: A v = λ v to rounding for a non-normal real matrix with complex pairs,
    /// close and repeated eigenvalues, and the trace kept.
    #[test]
    fn the_eigen_solver_resolves_a_non_normal_matrix(){
        let r=40;let mut seed=12345_u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);((seed>>11) as f64/(1u64<<53) as f64)-0.5};
        let mut cases:Vec<Vec<Z>>=vec![(0..r*r).map(|_|Z::new(next(),0.)).collect()];
        // A Jordan-like block and a repeated eigenvalue among the random ones.
        let mut near=cases[0].clone();for i in 0..r{for j in 0..r{near[i*r+j]*=1e-3;}near[i*r+i]+=Z::new(if i<3{0.9}else{0.5+0.01*i as f64},0.);if i+1<r&&i<3{near[i*r+i+1]+=Z::new(1.,0.);}}
        cases.push(near);
        for a in &cases{
            let(values,vectors)=eigen(a,r).unwrap();let norm=a.iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();
            for k in 0..r{
                let residual=(0..r).map(|i|((0..r).map(|j|a[i*r+j]*vectors[j*r+k]).sum::<Z>()-values[k]*vectors[i*r+k]).norm_sqr()).sum::<f64>().sqrt();
                assert!(residual<=1e-10*norm,"eigenpair {k}: residual {residual:e}");
            }
            let trace:Z=(0..r).map(|i|a[i*r+i]).sum();let sum:Z=values.iter().sum();
            assert!((trace-sum).norm()<=1e-12*norm,"trace {trace} vs {sum}");
        }
    }

    /// Without light near resonance the thermal state of the trap stays over ten radial periods: n̄
    /// (its energy) to 1e-3, its temperature within dt's share of the tolerance (Strang's trap is
    /// H + O(dt²): the kinetic share of the energy moves by ≈ dt's error), its populations exactly;
    /// its loss, atoms beyond the momentum range, within the loss's share.
    #[test]
    fn the_thermal_state_of_the_trap_is_stationary_without_light(){
        let r=tweezer((0..=10).map(|i|10.*i as f64).collect());let state=r.states.iter().position(|s|s=="ground-2-2").unwrap();
        let solution=crate::solve(&r,&mut |_,_|{}).unwrap();
        let(t,n)=(&solution.temperature_uk,&solution.mean_occupation);
        assert!((t[0]-30.).abs()<=0.2,"the initial temperature {}",t[0]);
        for k in 1..t.len(){
            assert!((t[k]/t[0]-1.).abs()<=SHARE*MOTION_TOLERANCE&&((n[k]+0.5)/(n[0]+0.5)-1.).abs()<=1e-3,"t {} us: T {} vs {}, n̄ {} vs {}",r.times_us[k],t[k],t[0],n[k],n[0]);
            assert!(solution.loss_probability[k]<=SHARE*r.tolerance&&(solution.populations[k][state]-1.).abs()<=1e-9,"t {} us: loss {}",r.times_us[k],solution.loss_probability[k]);
        }
    }
}
