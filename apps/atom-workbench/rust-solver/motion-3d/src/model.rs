//! The 3D model: the eliminated generator of `ground` on each cluster at any point r and momentum
//! ħκ, from tables over the focused beams' intensities and the near carriers' fields at r.
//!
//! # Where the generator depends on r
//! H(r) = H0 + Σ_u (s_u(r) − s_u(0)) O_u + Σ_u s_u(r) F_u (ground.rs) depends on r only through
//! the units' intensities s_u(r) -- a beam's |a_l(r)|², or the summed field's of beams of one
//! frequency and polarization, their standing wave --, and so does the elimination: the energies, K
//! and every near beam's responses. The near beams enter through their amplitudes a_l(r) (phase and
//! profile, a Gaussian beam's exact), linearly per carrier. The tables hold the elimination on a
//! uniform grid of the focused units' intensities s ∈ [0, 1]^D, cubic in each s_f, the grid doubled
//! while a finer grid costs less than the trajectories its smaller error saves (`tabulate`), the error
//! the interpolation's at the new nodes where the atoms go, against the exact elimination, relative to
//! every table's scale there. The focused units are the trap beam's and
//! every varying unit whose profile matters (`Model::new`): one whose intensity changes over the
//! atom's reach by more than the budget leaves -- its shifts O_u + F_u against the excited poles --
//! gets an axis; the others' shifts are taken at the focus.
//!
//! # A carrier's field basis
//! Each near beam's coupling V_l is the dipole operator contracted with the beam's field, so the
//! couplings of one carrier span at most three matrices. An orthonormal basis U_k of that span
//! (Gram–Schmidt, Frobenius inner product) gives V_l = Σ_k c_lk U_k and -- the responses being linear
//! in the coupling column by column -- R_l = Σ_k c_lk T_k, T_k the responses of U_k's folded part
//! U^f_k (the kept drives' entries zeroed, the same for every beam of the carrier, ground.rs). At r
//! the carrier drives V_c = Σ_k E_k U_k and responds R_c = Σ_k E_k T_k(s), E_k(r) = Σ_l a_l(r) c_lk:
//! the carrier's field on its basis, three numbers whatever its number of beams.
//!
//! # The generator of a cluster at r and t
//! With Φ_k′k = E^d_k′* E^c_k for the kept couplings absorbing c and emitting d (`ground::Group`),
//! S_k′k = ½(U^fd_k′† T^c_k + T^d_k′† U^fc_k) and Q_k′k = T^d_k′† D T^c_k on the group's entries
//! between ground states,
//!
//! ```text
//! H_eff = h0 − Σ_groups e^{iΩ̃t} Σ_k′k Φ_k′k (S_k′k + (i/2) Q_k′k)
//!       + Σ_drives e^{iΩ̃t} Σ_k E_k U_k[e,b] |e⟩⟨b| + h.c.  + (i/2) Σ_crosses e^{iΩ̃t} Σ_k E_k X_k |e⟩⟨b| + h.c.  − (i/2) Σ_decays
//! ```
//!
//! h0 holding the states' energies less their frames, the kept static couplings and the folded
//! static couplings' van Vleck shift -- for the explicit excited states too, their energies in H(r)
//! (the trap's anti-trapping of the excited states among them). A group of one carrier with itself
//! and no beat is Hermitian in (k′, k), S_kk′ = S_k′k†, Q_kk′ = Q_k′k†: its tables hold k′ ≤ k, the
//! pairs k′ < k standing for both. Between two explicit states its S is their folded drives' van
//! Vleck shift, −½ Σ_b U^f_k[e,b] U^f_k′[e′,b]* (1/(E_e − δ − E_b) + 1/(E_e′ − δ − E_b)), and Q zero.
//! The kept drives are linear in the fields and need no table: U_k at the focus is exact. The decay
//! of the explicit states, −(i/2) Σ_φ J_φ†J_φ on their rows and columns, is constant between two of
//! them (`decays`) and linear in the fields between one and a ground state (`crosses`: the folded
//! amplitude A_c = −R_c of the class's photons, X_k = w·T_k[:,b] with w = Σ_{to,a} C_a[to,e]* C_a[to,·]
//! over the photons the two share). Only the Hermitian part acts on the motion: the Ehrenfest force
//! is −∇Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩, with the kept drives' gradients Σ_k ∇E_k U_k the stimulated force. The
//! decay is the jumps' product by construction: where a class joins two folded photons of channels
//! without a kept coupling (through an explicit state's photon of their frequency), their cross term
//! Q = Σ_{to,a} (C_a T^d)[to,a]†(C_a T^c)[to,b] over the landing states they share enters alone, in a
//! group of its own without S (ground.rs `FoldedCross`).
//!
//! # The generator's beats
//! Every term's residual Ω̃ is a beat of the generator: H_eff(t) = H₀ + Σ_j e^{iΩ_j t} H_j (`Components`),
//! H₀ the static part less its scalar (a phase), H_j the terms at Ω_j, the gradients alike. Residuals
//! within `same_frame`'s tolerance at the cluster's largest frame and detuning (the frames' round-off)
//! are static, and are one beat where within it of each other; the beats come in pairs ±Ω (a term and
//! its adjoint), and the gradient's Hermitian part pairs them, G_{−Ω} = G_Ω†. Where the beats are
//! integer multiples of one frequency ω₀ (Euclid's algorithm within the round-off) their period 2π/ω₀
//! lets the exact step cache one period's propagators (engine.rs). A cluster with beats or explicit
//! excited states takes the exact step; `local` is the release's static clusters of ground states'.
//!
//! # The near light's dressing
//! Every near carrier's response is dressed by the other near light (ground.rs, # The near light's
//! dressing): the elimination at each node takes it, each near carrier's field at its units'
//! intensities there, and the tables hold the dressed responses -- the folded pairs' far channels in
//! their denominators, where the release added their first order as van Vleck paths. What the
//! dressing leaves out (a carrier's beams' interference, its next order) is bounded at each node
//! against the pole, its largest over the tables in the ledger (`Model::dressing`).
//!
//! # The photons and the Doppler term
//! The photon of axis x scattered from carrier c leaves the atom in C_x A_c ψ = −C_x R_c ψ, from the
//! jump tables J_xk = C_x T_k (all ground rows: the class chooses the target there); an explicit
//! state's photon in C_x ψ_e, the two added in one class. Absorbing from beam l
//! raises the excited state's energy by its kinetic shift w_l = (ħ/m) k_l·κ + ħ|k_l|²/2m; to first
//! order R_l → R_l − w_l R2_l (ground.rs), so the photon is C_x Σ_k (E_k T_k − E^D_k T2_k) ψ with
//! J2_xk = C_x T2_k and E^D_k = Σ_l a_l w_l c_lk: the scattering's velocity dependence, which the
//! jumps' mean momentum turns into Doppler friction. In the no-jump generator the term changes the
//! light shift and the scattering by kv/Δ, odd in v, and is left out: the ensemble's error is second
//! order, (kv/σ_min)², as the photons' (engine.rs).
//!
//! # The trap
//! The trap beam's focus is the origin. Its trap is each state's potential with the near beams'
//! interference left out (`Model::potential`; a unit's standing wave is in it); its Hessian at the
//! focus gives the trap's principal axes and frequencies (`Trap`). An atom is held while its
//! mechanical energy, ħ²κ²/2m plus its potential relative to far from the trap, is negative -- or,
//! unbound, until it leaves the trap's region (engine.rs # Loss): the potential the engine's force
//! derives from, Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩ of its current internal state, less the same with the trap's light
//! off (`Model::trap_potential`) -- the trap beam and the beams of its mode, along its axis with its
//! waist and frequency (its reflections, `Model::trap_beams`).
use crate::ground::{self,Ground};
use crate::ledger::BUDGET;
use crate::linear::{hermitian_eigenvalues,Dense};
use crate::solve::{MATRIX_NS,MOST_TRAJECTORIES,POPULATION_TARGET};
use crate::stats::KT_PER_UK;
use crate::{Complex,Request,STATES};
use rayon::prelude::*;

pub const HBAR:f64=1.0545718176461565e-34;
pub const BOLTZMANN:f64=1.380649e-23;
/// The most focused beams the tables span (one axis of intensity each).
pub const MOST_FOCUSED:usize=3;
/// The ground manifold's states (5S₁/₂, F = 1 and 2): a photon's amplitude on each.
pub const GROUND:usize=8;
/// The most states of a cluster of ground states, and its generator's interleaved size.
pub const MOST_STATES:usize=8;
const MOST_W:usize=2*MOST_STATES*MOST_STATES;
/// The most states of a cluster with explicit excited states: the 32 of the model.
pub const MOST_MIXED:usize=STATES;
/// The largest rank of a carrier's couplings (the dipole operator's three components).
pub const MOST_RANK:usize=3;
/// The coarsest grid of each intensity axis (nodes), the fewest a cubic stencil takes doubled once.
const FIRST_NODES:usize=5;
/// An instant at which the generator's beats take generic phases (us, the golden ratio's inverse: no
/// rational multiple of a period), where `Model::lindblad` checks it besides t = 0.
const GENERIC:f64=0.618_033_988_749_894_8;
/// The head of a model's refusal for a generator that gains beyond the budget (`Model::lindblad`): solve.rs stops
/// with it where the ladder's first rung, every drive folded, meets it.
pub const NOT_LINDBLAD:&str="消去激发态后的生成元不是 Lindblad 形式";
/// The most a model's tables may take while a grid is refined, the coarse grid's and the fine one's
/// together (bytes), and the most nodes one grid may have: the model's work bounds, as `MOST_STEPS`
/// and `MOST_TRAJECTORIES` are a solve's (solve.rs). A resonance across the focused units'
/// intensities -- a near probe crossing its line where far beams shift it, its ridge a tenth of the
/// intensities wide -- can ask for a grid finer than a machine holds. A grid beyond either bound is
/// not made, whatever it would save (`tabulate`): the tables keep the last one, its measured
/// interpolation error booked (`Model::table_error`, the ledger's), the same on any machine.
pub const MOST_TABLE_BYTES:usize=1<<32;
pub const MOST_TABLE_NODES:usize=1<<20;
/// The nodes of a grid computed at once, written into its tables before the next: a grid's nodes
/// are held once, in its tables.
const NODE_BATCH:usize=4096;

/// A beam's geometry: its direction and wave number, and its waist when Gaussian.
#[derive(Clone,Debug)]
pub struct Geometry{pub direction:[f64;3],pub wave_number:f64,pub waist:Option<f64>}
impl Geometry{
    /// The wave vector (rad/um).
    pub fn wave_vector(&self)->[f64;3]{self.direction.map(|n|n*self.wave_number)}
    /// a(r) and ∇a(r) (1/um): the plane wave e^{ikζ}, or the paraxial Gaussian beam focused at the
    /// origin e^{ikζ} exp(−ρ²/(w² q))/q, q = 1 + iζ/z_R, z_R = k w²/2 (request.rs `Beam::amplitude_at`),
    /// whose ∇ln a = ik n − 2(r − ζn)/(w² q) + i(ρ²/(w² q²) − 1/q) n / z_R.
    pub fn field(&self,r:[f64;3])->(Complex,[Complex;3]){
        let(k,n)=(self.wave_number,self.direction);let zeta=r[0]*n[0]+r[1]*n[1]+r[2]*n[2];let wave=Complex::cis(k*zeta);
        match self.waist{
            None=>{let g=Complex::new(0.,k)*wave;(wave,n.map(|x|g*x))}
            Some(w)=>{
                let rayleigh=k*w*w/2.;let q=Complex::new(1.,zeta/rayleigh);let inverse=Complex::ONE/q;
                let rho2=(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]-zeta*zeta).max(0.);
                let a=wave*(Complex::new(-rho2/(w*w),0.)*inverse).exp()*inverse;
                let along=Complex::new(0.,k)+(Complex::new(0.,rho2/(w*w))*inverse*inverse-Complex::new(0.,1.)*inverse)*(1./rayleigh);
                let across=inverse*(-2./(w*w));
                (a,std::array::from_fn(|i|a*(along*n[i]+across*(r[i]-zeta*n[i]))))
            }
        }
    }
    /// The intensity |a(r)|².
    pub fn intensity(&self,r:[f64;3])->f64{self.field(r).0.norm_sqr()}
}

/// A carrier: its beams, their couplings' basis and coefficients, its photons' wave number.
#[derive(Clone,Debug)]
pub struct Carrier{
    pub beams:Vec<usize>,
    pub rank:usize,
    /// c_lk per beam of the carrier (in `beams` order).
    pub coefficients:Vec<[Complex;MOST_RANK]>,
    pub wave_number:f64,
}

/// A table group's place in a node's slice: a kept group of `ground::Group` with one residual beat,
/// its field pairs (k′ ≤ k for a Hermitian one), its entries -- the (row, column) places of the
/// cluster's generator its couplings at this beat reach; S and Q vanish elsewhere -- and where its S
/// and Q lie (the entries' values per pair). A `decay` group holds the decay alone, its S zero: the
/// cross terms of folded photons a class joins without a kept coupling (`ground::FoldedCross`).
#[derive(Clone,Debug)]
pub(crate) struct GroupLayout{pub(crate) absorbed:usize,pub(crate) emitted:usize,pub(crate) residual:f64,pub(crate) hermitian:bool,pub(crate) pairs:Vec<(usize,usize)>,pub(crate) entries:Vec<(usize,usize)>,pub(crate) s:usize,pub(crate) q:usize,
    pub(crate) decay:bool}
/// Where a carrier's photon tables lie: J and J2 (3 axes × rank × ng × n0) and T (rank × ne × n0),
/// on the cluster's n0 ground states.
#[derive(Clone,Copy,Debug)]
pub(crate) struct CarrierLayout{pub(crate) j:usize,pub(crate) j2:usize,pub(crate) t:usize}
/// A kept drive: row (explicit state), column (ground state), carrier, residual beat and U_k[e,b]
/// per rank at the focus.
#[derive(Clone,Copy,Debug)]
pub struct DriveLayout{pub row:usize,pub col:usize,pub carrier:usize,pub residual:f64,pub u:[Complex;MOST_RANK]}
/// A decay cross term between an explicit state (row) and a ground state (column) whose folded
/// photons share a class with its: carrier, residual beat, the landing states the two share, and
/// where its X_k = w·T_k[:,b] lie (rank values), w = Σ_{to,a} ĉ_a[to]* C_a[to,·] with the explicit
/// state's dressed photon ĉ (`Layout::emission`).
#[derive(Clone,Debug)]
struct CrossLayout{row:usize,col:usize,carrier:usize,residual:f64,tos:Vec<usize>,at:usize}
/// The decay between two explicit states of a class, −(i/2) Σ_{to,a} ĉ_a[to]* ĉ′_a[to] over the
/// landing states they share: row, column, residual beat, those states, where its value lies.
#[derive(Clone,Debug)]
struct DecayLayout{row:usize,col:usize,residual:f64,tos:Vec<usize>,at:usize}
/// A drive the dressing brings, from ground state b (column) to explicit state e (row) through the
/// folded excited states F: ½(H_eF A_c[F,b] + x_e†V^f_c[F,b]) = Σ_k E_k Y_k, carrier, residual beat,
/// where Y_k lie (rank values).
#[derive(Clone,Debug)]
struct EffectiveLayout{row:usize,col:usize,carrier:usize,residual:f64,at:usize}
/// Where one cluster's tables lie in a node's slice, and its explicit states' terms: the kept drives
/// (without a table), the decays, the cross terms, the dressing's drives and the explicit states'
/// dressed photons ĉ_a[to] (explicit × 3 axes × ng, at `emission`).
#[derive(Clone,Debug)]
pub(crate) struct Layout{pub(crate) n:usize,pub(crate) n0:usize,pub(crate) size:usize,pub(crate) h0:usize,pub(crate) groups:Vec<GroupLayout>,pub(crate) carriers:Vec<CarrierLayout>,crosses:Vec<CrossLayout>,drives:Vec<DriveLayout>,
    decays:Vec<DecayLayout>,effective:Vec<EffectiveLayout>,emission:usize,pub(crate) fourth:Option<FourthLayout>,beats:Beats}
/// A cluster's beats (module documentation, # The generator's beats): the distinct frequencies of its terms'
/// residuals, ascending and closed under negation (−Ω_max … −Ω_min, Ω_min … Ω_max); the round-off within
/// which a residual is static or one of them; their common period where they are harmonics of one frequency.
#[derive(Clone,Debug,Default)]
pub(crate) struct Beats{pub(crate) omegas:Vec<f64>,round_off:f64,pub(crate) period:Option<f64>}
impl Beats{
    /// The beats of the residuals: those beyond the round-off, merged where within it of each other.
    fn new(residuals:impl Iterator<Item=f64>,round_off:f64)->Beats{
        let mut positive:Vec<f64>=residuals.map(f64::abs).filter(|&w|w>round_off).collect();positive.sort_by(f64::total_cmp);
        let mut merged:Vec<f64>=vec![];for w in positive{if merged.last().map_or(true,|&m|w-m>round_off){merged.push(w);}}
        let omegas:Vec<f64>=merged.iter().rev().map(|&w|-w).chain(merged.iter().copied()).collect();
        let period=Beats::period_of(&merged,round_off);
        Beats{omegas,round_off,period}
    }
    /// The period 2π/ω₀ of beats that are integer multiples of one frequency ω₀: Euclid's algorithm with
    /// nearest-integer quotients on the positive beats, each remainder's bound of its round-off carried
    /// along (|a − q b| errs by at most e_a + |q| e_b); a remainder within its bound ends it, one whose
    /// divisor is within its own bound has no common frequency. Every beat is then checked to be its
    /// multiple within the bounds. Within the round-off incommensurate beats are harmonics too, of a
    /// fundamental far below them: a period too long for a step's cache, which declines it (engine.rs).
    fn period_of(positive:&[f64],round_off:f64)->Option<f64>{
        let(&first,rest)=positive.split_first()?;let(mut g,mut error)=(first,round_off);
        for &w in rest{
            let(mut a,mut ea,mut b,mut eb)=(w,round_off,g,error);
            loop{
                if b<=eb{return None;}
                let q=(a/b).round();let(r,er)=((a-q*b).abs(),ea+q.abs()*eb);
                if r<=er{break;}
                (a,ea,b,eb)=(b,eb,r,er);
            }
            (g,error)=(b,eb);
        }
        positive.iter().all(|&w|{let q=(w/g).round();(w-q*g).abs()<=round_off+q*error}).then(||std::f64::consts::TAU/g)
    }
    /// The beat of a residual: None where it is static, else its index in `omegas`.
    fn of(&self,residual:f64)->Option<usize>{
        if residual.abs()<=self.round_off{return None;}
        let half=self.omegas.len()/2;let i=self.omegas[half..].iter().position(|&w|(residual.abs()-w).abs()<=self.round_off).expect("every residual has its beat");
        Some(if residual>0.{half+i}else{half-1-i})
    }
}
/// The fourth order's tables of a cluster of ground states (ground.rs, # Fourth order), on
/// the carriers that drive it, `fields` their field indices (carrier, k):
/// - H^(4) per monomial E*_p E*_p′ E_q E_q′ (`QuarticSlab`: the conjugated fields' pair p ≤ p′, the plain
///   ones' q ≤ q′; the monomial is symmetric in either, so one carrier of rank r has (r(r+1)/2)² slabs and a
///   carrier pair's two orders share theirs), n × n each: the Hermitian part of the van Vleck form
///   V†W3 − ½[X, V†R] and −i/2 times the fourth order of Σ L̃†L̃, the jumps' own (X the metric's coherent
///   part, the readout's);
/// - per carrier pair (c1, c2) R_c1†R_c2 on the coherent entries (X's) and YW_c1c2 on all (`SlowSlab`,
///   r_c1 r_c2 × n × n each);
/// - for the jumps' W3^c at a carrier c's frequency, per carrier c1 inside it C_a G(δ_c) T^c1_k[:,b′] from each source
///   place b to each column y, c's block's (`JumpSlab`, r_c1 × 3 × ng × ng × n × n: k, axis, row, b′, y, b) and the
///   middle factors of the triples (c1, c2, c4)
///   it takes, each P^{c2c4} = U^c2†T^c4 less what the dressing holds (`Middle`, r_c2 r_c4 × ng × n, at `p`: one
///   table per pair and kind);
/// - where two photon channels' jumps meet: per pair of channels (carrier, place) the landing states their photons
///   share a class in (`landing`, a bit per ground state), where the fourth order's decay takes their products.
/// Every G(ω) at a carrier c's frequency is c's block's dressed resolvent (ground.rs `fourth_column`).
/// An entry (x, y) of a slab of lab frequency ν beats at Ω = θ_x − θ_y + ν in the cluster's frames. H^(4)
/// keeps it where the second order keeps a coupling at that beat (below the cut, or at a kept coupling's
/// residual on the entry); the others are zero in the tables, their mixings' squares booked (`dropped`).
#[derive(Clone,Debug)]
pub(crate) struct FourthLayout{carriers:Vec<usize>,fields:Vec<(usize,usize)>,quartic:Vec<QuarticSlab>,slow:Vec<SlowSlab>,jumps:Vec<JumpSlab>,
    p:Vec<(usize,usize,Middle,usize)>,landing:Vec<u32>,pub(crate) dropped:f64}
/// A W3 triple (c1, c2, c4)'s middle factor (V_c2†R_c4)[b′, b] (ground.rs `Ground::excluded`): whole; less the far channels
/// of (c4, b) through c2 (c1 = c2, Σ's first order); or less c2's light shift (c2 = c4 ≠ c1) that c1's block holds:
/// b's own, an unmixed block's (`Shift`, ground.rs `Ground::unmixed`), or between the cluster's states (`Shifts`).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) enum Middle{Whole,Vias,Shift,Shifts}
impl Middle{
    fn of(c1:usize,c2:usize,c4:usize,unmixed:bool)->Middle{if c1==c2{Middle::Vias}else if c2!=c4{Middle::Whole}else if unmixed{Middle::Shift}else{Middle::Shifts}}
}
/// One monomial's slab of H^(4): its conjugated and plain field pairs (into `FourthLayout::fields`), its
/// lab frequency ν (rad/us), its kept entries, their parts among the cluster's beats (`Model::components`:
/// 0 the static part, 1 + j beat j's) and where it lies.
#[derive(Clone,Debug)]
pub(crate) struct QuarticSlab{conj:(usize,usize),plain:(usize,usize),nu:f64,keep:Vec<bool>,part:Vec<usize>,at:usize}
impl FourthLayout{
    /// Its slabs' kept entries' beats Ω = θ_x − θ_y + ν in a cluster's frames.
    fn residuals(&self,frames:&[f64])->Vec<f64>{
        let n=frames.len();self.quartic.iter().flat_map(|q|(0..n*n).filter(|&i|q.keep[i]).map(|i|frames[i/n]-frames[i%n]+q.nu).collect::<Vec<f64>>()).collect()
    }
    /// Each slab's kept entries' parts among the cluster's beats.
    fn place(&mut self,frames:&[f64],beats:&Beats){
        let n=frames.len();
        for q in &mut self.quartic{q.part=(0..n*n).map(|i|if q.keep[i]{beats.of(frames[i/n]-frames[i%n]+q.nu).map_or(0,|j|j+1)}else{0}).collect();}
    }
}
/// One carrier pair's slabs: R_c1†R_c2 on the coherent entries (`coherent`) at `rr`, YW_c1c2 at `yw`; their
/// lab frequency δ_c1 − δ_c2.
#[derive(Clone,Debug)]
pub(crate) struct SlowSlab{carriers:(usize,usize),nu:f64,coherent:Vec<bool>,rr:usize,yw:usize}
/// The jumps' W3 at a photon carrier's frequency through one inner carrier c1: its table -- each column from every source
/// place, or from its own alone where the photon carrier's block is unmixed (`diagonal`, ground.rs `Ground::unmixed`) --
/// and the middle factors of the triples (c1, c2, c4) at that frequency (into `FourthLayout::p`).
#[derive(Clone,Debug)]
pub(crate) struct JumpSlab{photon:usize,inner:usize,pairs:Vec<usize>,diagonal:bool,at:usize}
impl JumpSlab{
    /// The source places per column: all n of the cluster's, or the column's own.
    fn sources(&self,n:usize)->usize{if self.diagonal{1}else{n}}
}

/// e^{iωt} of a run of residuals ω at one time, a value's rotation computed once while it repeats (the
/// photons of a class share theirs): the same rotation as Complex::cis of each.
struct Rotation{t:f64,last:u64,value:Complex}
impl Rotation{
    fn new(t:f64)->Rotation{Rotation{t,last:f64::NAN.to_bits(),value:Complex::ONE}}
    fn at(&mut self,w:f64)->Complex{if w.to_bits()!=self.last{self.last=w.to_bits();self.value=Complex::cis(w*self.t);}self.value}
}

/// A cluster's generator against its jumps at a point and time (`Model::decay_balance`): D = i(H − H†) − Σ L†L
/// (at fourth order less the corrections' δ†δ), the jumps' whole Σ L̃†L̃, D's smallest eigenvalue, Σ L†L's largest
/// and D's round-off scale.
struct Balance{d:Dense,jumps:Dense,low:f64,rate:f64,scale:f64}
/// The largest ε with −D ≤ ε M, M = J + floor (Hermitian, J ⪰ 0): the largest eigenvalue of L⁻¹(−D − round)L⁻†,
/// M = LL† (Cholesky); 0 where none is positive, and infinite where M is not positive definite.
fn excess(d:&Dense,j:&Dense,round:f64,floor:f64)->f64{
    let n=d.rows;let mut l=Dense::zeros(n,n);
    for c in 0..n{
        let mut diagonal=j[(c,c)].re+floor;for k in 0..c{diagonal-=l[(c,k)].norm_sqr();}
        if !(diagonal>0.){return f64::INFINITY;}let diagonal=diagonal.sqrt();l[(c,c)]=Complex::new(diagonal,0.);
        for i in c+1..n{let mut z=j[(i,c)];for k in 0..c{z-=l[(i,k)]*l[(c,k)].conj();}l[(i,c)]=z*(1./diagonal);}
    }
    // L⁻¹ B, column by column (forward substitution).
    let solve=|b:&Dense|->Dense{let mut x=b.clone();for c in 0..n{for i in 0..n{let mut z=x[(i,c)];for k in 0..i{z-=l[(i,k)]*x[(k,c)];}x[(i,c)]=z*(1./l[(i,i)].re);}}x};
    let a=Dense::from_fn(n,n,|i,k|-d[(i,k)]-if i==k{Complex::new(round,0.)}else{Complex::ZERO});
    let y=solve(&a);let x=solve(&y.adjoint()).adjoint();
    hermitian_eigenvalues(&x).into_iter().fold(0_f64,f64::max)
}

/// A cluster's work per call in complex multiply-adds (`Model::work`), and its table pairs.
#[derive(Clone,Copy,Debug)]
pub struct Work{pub generator:f64,pub force:f64,pub photons:f64,pub absorption:f64,pub pairs:usize}

/// The interpolation stencil at s: per focused beam the first of its four nodes, their cubic
/// weights and the weights' derivatives in s_f; the nodes are their tensor product.
#[derive(Clone,Copy,Debug)]
pub struct Stencil{pub dimensions:usize,pub first:[usize;MOST_FOCUSED],pub weights:[[f64;4];MOST_FOCUSED],pub derivatives:[[f64;4];MOST_FOCUSED],pub stride:[usize;MOST_FOCUSED]}
impl Stencil{
    /// The number of nodes (4^D).
    pub fn count(&self)->usize{1<<(2*self.dimensions)}
    /// Node i's index, its weight and its weights' derivatives in each s_f.
    pub fn node(&self,i:usize)->(usize,f64,[f64;MOST_FOCUSED]){
        let(mut index,mut weight,mut derivative)=(0,1.,[1.;MOST_FOCUSED]);
        for f in 0..self.dimensions{let m=(i>>(2*f))&3;index+=(self.first[f]+m)*self.stride[f];weight*=self.weights[f][m];
            for g in 0..self.dimensions{derivative[g]*=if g==f{self.derivatives[f][m]}else{self.weights[f][m]};}}
        (index,weight,derivative)
    }
}

/// The fields of one carrier at (r, κ): E_k, ∇E_k (per k, per axis) and E^D_k.
#[derive(Clone,Copy,Debug,Default)]
pub struct CarrierField{pub e:[Complex;MOST_RANK],pub grad:[[Complex;3];MOST_RANK],pub doppler:[Complex;MOST_RANK]}

/// The most carriers (frequencies of near light) and beams a scene may have.
pub const MOST_CARRIERS:usize=8;
pub const MOST_BEAMS:usize=32;

/// Everything at (r, κ) that does not depend on the internal state: the stencils at the focused
/// beams' intensities there and with the trap beam off, their gradients and the carriers' fields.
#[derive(Clone,Copy,Debug)]
pub struct Point{
    pub r:[f64;3],
    pub kappa:[f64;3],
    pub stencil:Stencil,
    pub far:Stencil,
    /// ∇s_f (1/um).
    pub intensity_gradient:[[f64;3];MOST_FOCUSED],
    pub fields:[CarrierField;MOST_CARRIERS],
}

/// A cluster's generator at a point and a time: H_eff less `scalar`, the scalar part of its
/// Hermitian part (a phase); and what the gradient of its Hermitian part takes, for the force on the
/// state of the step's middle (`Model::force`): per table pair, in the groups' order, its S and every
/// ∂S/∂s_f at the point on its group's entries (`values`, re/im interleaved) and its coefficients
/// −∇_aΦ and −Φ (each twice for a pair standing for two); h0's ∂/∂s_f (n × n); and ∇s_f. A cluster of
/// ground states' (`Model::local`; its gradients as matrices `Model::gradients`).
#[derive(Clone,Debug)]
pub struct Local{pub hamiltonian:[Complex;MOST_STATES*MOST_STATES],pub scalar:f64,values:Vec<f64>,coefficients:Vec<[Complex;4]>,h0:[[f64;MOST_W];MOST_FOCUSED],intensity_gradient:[[f64;3];MOST_FOCUSED]}
/// A cluster's generator at a point and a time on any number of states, row-major n × n: H_eff
/// less `scalar` (the scalar part of its Hermitian part) and the gradient of its Hermitian part per
/// axis -- the explicit excited states' clusters', and any cluster's in the exact step (engine.rs).
#[derive(Clone,Debug)]
pub struct Generator{pub n:usize,pub hamiltonian:Vec<Complex>,pub scalar:f64,pub gradient:[Vec<Complex>;3]}
impl Generator{
    /// −Re⟨ψ|∇H|ψ⟩/⟨ψ|ψ⟩: the Ehrenfest force on ħκ (rad/us per um).
    pub fn force(&self,psi:&[Complex])->[f64;3]{
        let n=self.n;let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        std::array::from_fn(|a|-hermitian_form(&self.gradient[a],psi,n)/norm)
    }
}
/// A cluster's generator at a point split by its beats (`Model::components`): H(t) = H₀ + Σ_j e^{iΩ_j t} H_j
/// and the Hermitian part's gradient G(t) = G₀ + Σ_j e^{iΩ_j t} G_j, Hermitian at every t (G_{−Ω} = G_Ω†); H₀
/// less the static part's scalar; the beats' common period where they are harmonics of one frequency.
#[derive(Clone,Debug)]
pub struct Components{pub statics:Generator,pub beats:Vec<Beat>,pub period:Option<f64>}
/// One beat of a generator: its frequency Ω (rad/us) and its parts of H and of the gradient (n × n).
#[derive(Clone,Debug)]
pub struct Beat{pub omega:f64,pub hamiltonian:Vec<Complex>,pub gradient:[Vec<Complex>;3]}
impl Components{
    pub fn n(&self)->usize{self.statics.n}
    /// The generator at time t.
    pub fn at(&self,t:f64)->Generator{
        let mut g=self.statics.clone();
        for b in &self.beats{
            let phase=Complex::cis(b.omega*t);
            for(o,x)in g.hamiltonian.iter_mut().zip(&b.hamiltonian){*o+=phase**x;}
            for a in 0..3{for(o,x)in g.gradient[a].iter_mut().zip(&b.gradient[a]){*o+=phase**x;}}
        }
        g
    }
}

/// Re⟨ψ|X|ψ⟩ of a Hermitian n × n matrix X (row-major).
#[inline]
pub fn hermitian_form(x:&[Complex],psi:&[Complex],n:usize)->f64{
    let mut s=0.;
    for a in 0..n{let p=psi[a];s+=x[a*n+a].re*p.norm_sqr();for b in a+1..n{let z=x[a*n+b]*psi[b];s+=2.*(p.re*z.re+p.im*z.im);}}
    s
}

/// acc += c x for interleaved complex vectors of W/2 entries.
#[inline(always)]
fn caxpy<const W:usize>(c:Complex,x:&[f64;MOST_W],acc:&mut [f64;MOST_W]){
    for j in (0..W).step_by(2){let(u,v)=(x[j],x[j+1]);acc[j]+=c.re*u-c.im*v;acc[j+1]+=c.re*v+c.im*u;}
}
/// ⟨ψ|X|ψ⟩ for an interleaved N × N matrix X.
#[inline(always)]
fn quadratic<const N:usize>(x:&[f64;MOST_W],psi:&[Complex])->Complex{
    let mut s=Complex::ZERO;
    for a in 0..N{let mut row=Complex::ZERO;for b in 0..N{let k=2*(a*N+b);row+=Complex::new(x[k],x[k+1])*psi[b];}s+=psi[a].conj()*row;}
    s
}

/// One node of the tables: every cluster's tables there, and its bounds: the dressing's (what it leaves
/// out, `Model::dressing`), the elimination's (`Model::resonance`) and the undressed response's
/// (`Model::undressed`).
type Node=(Vec<Vec<Complex>>,[f64;3]);

/// The tables on one grid of n nodes per focused unit: per cluster, node-major, its layout's size per
/// node (re/im interleaved); per node its bounds (`Node`); and each bound's largest over the nodes
/// inside the reach (`Model::dressing`, `Model::resonance`, `Model::undressed`); the work its computed
/// nodes took, counted (linear.rs `counted`: complex multiply-adds).
#[derive(Clone,Debug)]
pub(crate) struct Tables{pub(crate) nodes:usize,pub(crate) data:Vec<Vec<f64>>,bounds:Vec<[f64;3]>,reach:[f64;3],work:f64}

/// The trap of the trap beam: its principal axes (unit rows on the lab axes) and frequencies
/// (rad/us) at the focus for a reference state, and the frequencies per lab axis.
#[derive(Clone,Copy,Debug)]
pub struct Trap{pub axes:[[f64;3];3],pub frequencies:[f64;3],pub lab:[f64;3],pub lattice:Option<Lattice>}

/// The trap's standing wave, its beam and reflection one unit (ground.rs `Unit`): on the beam's axis the field is
/// γ₀ a + γ₁ ā, a = e^{ikz}/(1 + iz/z_R), its antinodes -- the sites -- where Φ(z) = kz − atan(z/z_R) + ½ arg(γ₀ γ̄₁)
/// is a multiple of π, its nodes half way between. Its axis (a unit vector), k, z_R and that phase.
#[derive(Clone,Copy,Debug)]
pub struct Lattice{pub axis:[f64;3],pub wave_number:f64,pub rayleigh:f64,pub phase:f64}
impl Lattice{
    /// The nearest site's number at z on the axis, and Φ's distance from it there, within ±π/2.
    pub fn nearest(&self,z:f64)->(f64,f64){let phase=self.wave_number*z-(z/self.rayleigh).atan()+self.phase;let n=(phase/std::f64::consts::PI).round();(n,phase-n*std::f64::consts::PI)}
    /// Site n's place on the axis: Φ's root by Newton's steps, from a start that depends on n alone.
    pub fn site(&self,n:f64)->f64{
        let(k,target)=(self.wave_number,n*std::f64::consts::PI-self.phase);let mut z=target/k;
        loop{let x=z/self.rayleigh;let dz=(k*z-x.atan()-target)/(k-1./(self.rayleigh*(1.+x*x)));z-=dz;if dz.abs()<=4.*f64::EPSILON*(z.abs()+1./k){return z;}}
    }
}

/// The 3D model of a request.
#[derive(Clone,Debug)]
pub struct Model{
    pub ground:Ground,
    pub geometry:Vec<Geometry>,
    /// The trap beam and the trap's beams (the trap beam and the beams of its mode, `trap_beams`), and
    /// the focused units (indices into `Ground::units`), one intensity axis each, the trap beam's first.
    pub trap_beam:usize,
    pub trap_beams:Vec<usize>,
    /// The trap's light (`trap_light`): the initial ensemble's potential (`trap_level`).
    pub trap_light:Vec<usize>,
    pub focused:Vec<usize>,
    /// Each varying unit taken at the focus: the unit, its estimated error over the reach and its
    /// shifts' weight, the error per unit of intensity changed (`choose_profiles`).
    pub profiles:Vec<(usize,f64,f64)>,
    pub carriers:Vec<Carrier>,
    /// ħ/m (um²/us).
    pub hbar_over_mass:f64,
    pub mass_kg:f64,
    pub(crate) layouts:Vec<Layout>,
    pub(crate) tables:Tables,
    /// The interpolation's measured largest relative error where the atoms go (`tabulate`), and where the
    /// work bounds -- or a doubling that did not shrink the error (`table_stalled`), whose next one is
    /// predicted to save nothing -- stopped the grid's refinement the nodes per axis it kept.
    pub table_error:f64,
    pub table_limit:Option<usize>,
    pub table_stalled:bool,
    /// The near light's dressing (ground.rs `Ground::dressing_bounds`), relative to the generator's
    /// scale, its largest over the tables' nodes inside the reach's intensities (as `resonance`): the bound
    /// of what it leaves out, and what a bare
    /// response -- each ground state at its own energy, undressed -- errs by where a block mixes, measured (the
    /// GPU's exact response, gpu/program.rs: the static block's eigenbasis is pending).
    pub dressing:f64,
    pub undressed:f64,
    /// The fourth order's H^(4) entries left out at beats the second order folds: their mixings' squares
    /// (`FourthLayout`).
    pub fourth_dropped:f64,
    /// The elimination's a-priori bound over the atoms' reach: the largest excited weight a ground
    /// state b can take, Σ_l ‖R_l[:,b]‖² over the near beams at their peak amplitude (each one's
    /// (Ω_l/2σ)², the beams' interference averaged out), over the tables' nodes inside the reach's
    /// intensities (`reach`). Where it exceeds the population target the eliminated model is outside
    /// its validity somewhere the atoms go -- light near resonance, which the excited population
    /// measured along the trajectories can miss when the atoms sit in dark states (solve.rs flags it).
    pub resonance:f64,
    /// The reach the profiles and the bound are judged over: the focus and the points ±reach_a along
    /// each principal axis of the trap (`choose_profiles`).
    reach:Vec<[f64;3]>,
    /// The processor has AVX2 and FMA: the generator is built with them.
    wide:bool,
    /// Its clusters of ground states folded at fourth order (`Model::with_order`).
    pub fourth:bool,
    /// The kinetic shift of absorption (`kinetic_shift`), always on in a solve; off only where a
    /// validation compares with the recoil-heating formula, which has no Doppler friction.
    pub doppler:bool,
}

fn frobenius_inner(a:&Dense,b:&Dense)->Complex{a.entries.iter().zip(&b.entries).fold(Complex::ZERO,|s,(x,y)|s+x.conj()**y)}

/// The interpolation weights at s on a uniform grid of `nodes` points on [0, 1], on four consecutive nodes from `first`,
/// and their derivatives in s: the cubic Hermite interpolant with Catmull-Rom's central slopes, (f_{i+1} − f_{i−1})/2h,
/// and at the grid's ends the second-order one-sided ones -- C¹, so that the force the tables' derivative gives is
/// continuous. (The piecewise 4-point Lagrange cubic before it was C⁰: its force jumped at every node by O(h³), and a
/// step that straddles a jump errs at first order in dt -- the 3D PGC's cold atoms crossed so many that the step
/// doubling read twice T's step error on 65 nodes per axis, its excess over 513 nodes falling as n⁻².) Its error is
/// O(h³) (`tabulate`'s first prediction 2⁻³ per doubling).
fn cubic(nodes:usize,s:f64)->(usize,[f64;4],[f64;4]){
    let h=1./(nodes-1) as f64;let x=s.clamp(0.,1.)/h;let i=(x.floor() as usize).min(nodes-2);let u=x-i as f64;
    // The Hermite basis on the interval [i, i + 1] and its derivatives in u.
    let(h00,h10,h01,h11)=(2.*u*u*u-3.*u*u+1.,u*u*u-2.*u*u+u,-2.*u*u*u+3.*u*u,u*u*u-u*u);
    let(d00,d10,d01,d11)=(6.*u*u-6.*u,3.*u*u-4.*u+1.,-6.*u*u+6.*u,3.*u*u-2.*u);
    // The values' weights on the stencil from the end slopes h m_i: the first interval's (−3f₀ + 4f₁ − f₂)/2 and
    // (f₂ − f₀)/2, an inner one's central pair, the last one's (f₃ − f₁)/2 and (3f₃ − 4f₂ + f₁)/2.
    let weights=|a:f64,b:f64,c:f64,d:f64|->[f64;4]{if i==0{[a-1.5*b-0.5*d,2.*b+c,-0.5*b+0.5*d,0.]}
        else if i==nodes-2{[0.,-0.5*b+0.5*d,a-2.*d,0.5*b+c+1.5*d]}else{[-0.5*b,a-0.5*d,c+0.5*b,0.5*d]}};
    let first=if i==0{0}else if i==nodes-2{nodes-4}else{i-1};
    (first,weights(h00,h10,h01,h11),weights(d00,d10,d01,d11).map(|x|x/h))
}

/// The pairs k′ ≤ k of a Hermitian group of rank r, or every (k′, k) of ranks (r′, r).
fn field_pairs(hermitian:bool,emitted:usize,absorbed:usize)->Vec<(usize,usize)>{
    if hermitian{(0..absorbed).flat_map(|k1|(k1..absorbed).map(move|k|(k1,k))).collect()}else{(0..emitted).flat_map(|k1|(0..absorbed).map(move|k|(k1,k))).collect()}
}

/// The trap's beams: the trap beam and the beams of its mode -- of its frequency, along its axis
/// either way, with its waist: its reflections --, whose light falls off with its own far from the
/// focus (`Model::trap_potential`).
fn trap_beams(ground:&Ground,geometry:&[Geometry],trap_beam:usize)->Vec<usize>{
    let(t,index)=(&geometry[trap_beam],&ground.beams[trap_beam].frequency_index);
    (0..geometry.len()).filter(|&l|l==trap_beam||(ground.beams[l].frequency_index==*index&&geometry[l].waist==t.waist&&{
        let c:f64=(0..3).map(|i|geometry[l].direction[i]*t.direction[i]).sum();(c.abs()-1.).abs()<=1e-12})).collect()
}

/// The trap's light, what holds the atoms before t = 0 (`Model::trap_level`): the trap's beams and every far Gaussian
/// beam -- a crossed beam's shift is part of the potential the atom's motion conserves --; not a near beam, whose light
/// the solve turns on at t = 0, nor a beam whose unit holds a plane wave, which does not fall off with it.
fn trap_light(ground:&Ground,geometry:&[Geometry],trap_beams:&[usize])->Vec<usize>{
    (0..geometry.len()).filter(|l|trap_beams.contains(l)||(geometry[*l].waist.is_some()&&!ground.carriers.iter().any(|c|c.beams.contains(l))
        &&ground.units.iter().filter(|u|u.beams.contains(l)).all(|u|u.beams.iter().all(|&b|geometry[b].waist.is_some())))).collect()
}

/// The largest eigenvalue's eigenvector of a real symmetric 3 × 3 matrix's eigen-decomposition:
/// its eigenvalues and orthonormal eigenvectors (rows), by Jacobi rotations.
pub(crate) fn symmetric_eigen(m:[[f64;3];3])->([f64;3],[[f64;3];3]){
    let mut a=m;let mut v=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
    for _ in 0..64{
        let off=a[0][1].abs()+a[0][2].abs()+a[1][2].abs();if off<=1e-15*(a[0][0].abs()+a[1][1].abs()+a[2][2].abs()){break;}
        for(p,q)in [(0,1),(0,2),(1,2)]{
            if a[p][q]==0.{continue;}
            let theta=(a[q][q]-a[p][p])/(2.*a[p][q]);let t=theta.signum()/(theta.abs()+(theta*theta+1.).sqrt());let c=1./(t*t+1.).sqrt();let s=t*c;
            for k in 0..3{let(x,y)=(a[k][p],a[k][q]);a[k][p]=c*x-s*y;a[k][q]=s*x+c*y;}
            for k in 0..3{let(x,y)=(a[p][k],a[q][k]);a[p][k]=c*x-s*y;a[q][k]=s*x+c*y;}
            for k in 0..3{let(x,y)=(v[p][k],v[q][k]);v[p][k]=c*x-s*y;v[q][k]=s*x+c*y;}
        }
    }
    ([a[0][0],a[1][1],a[2][2]],v)
}

impl Model{
    /// The model of a request with every drive folded (`Model::with` at an infinite allowance).
    pub fn new(request:&Request)->Result<Model,String>{Model::with(request,f64::INFINITY)}
    /// The model of a request: its front end, its drives kept until each ground state's folded rest is
    /// within `allowance` (ground.rs `Ground::with`), the focused beams, the carriers' bases and the
    /// tables.
    pub fn with(request:&Request,allowance:f64)->Result<Model,String>{Model::with_order(request,allowance,false)}
    /// The same with its clusters of ground states folded at fourth order where `fourth` (ground.rs,
    /// # Fourth order).
    pub fn with_order(request:&Request,allowance:f64,fourth:bool)->Result<Model,String>{Model::bounded(request,allowance,fourth,MOST_TABLE_BYTES,MOST_TABLE_NODES)}
    /// The same within the given work bounds on its tables (`MOST_TABLE_BYTES`, `MOST_TABLE_NODES`).
    pub(crate) fn bounded(request:&Request,allowance:f64,fourth:bool,most_bytes:usize,most_nodes:usize)->Result<Model,String>{
        let ground=Ground::with(request,allowance)?;
        let geometry:Vec<Geometry>=ground.beams.iter().map(|b|Geometry{direction:b.direction,wave_number:b.wave_number_per_um,waist:b.waist_um}).collect();
        let trap_beam=request.trap_beam;
        if trap_beam>=geometry.len()||geometry[trap_beam].waist.is_none(){return Err("光镊（trapBeamId 所指的光束）需要束腰：3D 运动计算聚焦在原子处的光镊中的单个原子".into());}
        let hbar_over_mass=HBAR/request.mass_kg*1e6;
        let trap_unit=ground.units.iter().position(|u|u.beams.contains(&trap_beam)).unwrap();let trap_beams=trap_beams(&ground,&geometry,trap_beam);
        // Each carrier's couplings: an orthonormal basis and the beams' coefficients on it.
        let mut carriers=vec![];let mut bases:Vec<Vec<Vec<Complex>>>=vec![];
        for c in &ground.carriers{
            let couplings:Vec<&Dense>=c.beams.iter().map(|&l|&ground.couplings[l]).collect();
            let(mut basis,mut mixing):(Vec<Dense>,Vec<Vec<Complex>>)=(vec![],vec![]);
            for(j,v)in couplings.iter().enumerate(){
                let mut u=(*v).clone();let mut g=vec![Complex::ZERO;couplings.len()];g[j]=Complex::ONE;
                // Twice, for orthogonality to round-off.
                for _ in 0..2{for(k,b)in basis.iter().enumerate(){
                    let x=frobenius_inner(b,&u);u.add_scaled(-x,b);for(gi,mi)in g.iter_mut().zip(&mixing[k]){*gi-=x**mi;}
                }}
                let norm=frobenius_inner(&u,&u).re.sqrt();let scale=frobenius_inner(v,v).re.sqrt();
                if norm<=1e-10*scale{continue;}
                basis.push(u.scaled(Complex::new(1./norm,0.)));mixing.push(g.into_iter().map(|x|x*(1./norm)).collect());
            }
            if basis.len()>MOST_RANK{return Err(format!("A carrier's couplings span {} matrices; the dipole operator has three components",basis.len()));}
            let coefficients:Vec<[Complex;MOST_RANK]>=couplings.iter().map(|v|std::array::from_fn(|k|if k<basis.len(){frobenius_inner(&basis[k],v)}else{Complex::ZERO})).collect();
            // The beams are on the span to round-off.
            for(v,c)in couplings.iter().zip(&coefficients){
                let mut d=(*v).clone();for(k,b)in basis.iter().enumerate(){d.add_scaled(-c[k],b);}
                let(e,s)=(frobenius_inner(&d,&d).re.sqrt(),frobenius_inner(v,v).re.sqrt());
                if e>1e-9*s{return Err(format!("A beam's coupling is off its carrier's span by {:.1e} of its norm",e/s));}
            }
            carriers.push(Carrier{beams:c.beams.clone(),rank:basis.len(),coefficients,wave_number:c.wave_number});bases.push(mixing);
        }
        if carriers.len()>MOST_CARRIERS||geometry.len()>MOST_BEAMS{return Err(format!("3D 运动最多支持 {MOST_CARRIERS} 个近共振频率、{MOST_BEAMS} 束光"));}
        if ground.ground_states.len()!=GROUND{return Err(format!("The ground manifold has {} states, not {GROUND}",ground.ground_states.len()));}
        if ground.clusters.iter().any(|c|c.excited.is_empty()&&c.states.len()>MOST_STATES){return Err("A cluster of ground states has more than eight states".into());}
        let trap_light=trap_light(&ground,&geometry,&trap_beams);
        let mut model=Model{ground,geometry,trap_beam,trap_beams,trap_light,focused:vec![trap_unit],profiles:vec![],carriers,hbar_over_mass,mass_kg:request.mass_kg,
            layouts:vec![],tables:Tables{nodes:1,data:vec![],bounds:vec![],reach:[0.;3],work:0.},table_error:0.,table_limit:None,table_stalled:false,dressing:0.,undressed:0.,fourth_dropped:0.,resonance:0.,reach:vec![[0.;3]],
            wide:std::arch::is_x86_feature_detected!("avx2")&&std::arch::is_x86_feature_detected!("fma"),doppler:true,fourth};
        model.choose_profiles(request)?;
        model.layouts=model.layouts(&bases);model.fourth_dropped=model.layouts.iter().filter_map(|l|l.fourth.as_ref()).map(|f|f.dropped).sum();
        let stake=model.stake(request);model.tabulate(&bases,most_bytes,most_nodes,stake)?;
        model.lindblad()?;
        Ok(model)
    }

    /// What a unit of booked population error costs the run (ns), the tables' stake (`tabulate`): the run's
    /// trajectories go as the inverse square of the room its estimates leave the statistics, so that an error
    /// e of the room R costs it 2e/R of itself at first order -- here at the most trajectories a run takes
    /// (`MOST_TRAJECTORIES`), each the pilot's first steps over the run (the light's pattern, solve.rs `run`)
    /// at an advance's counted work in the model's dearest cluster (solve.rs `advance_work`), against the
    /// whole population target. A run that takes fewer pays for at most a doubling beyond its own optimum,
    /// each cutting the error about 2³-fold at 2^D times the work of the last.
    fn stake(&self,request:&Request)->f64{
        let thermal=(KT_PER_UK*request.temperature_uk*self.hbar_over_mass).sqrt();
        let pattern=self.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max)*thermal;
        let rate=if pattern>0.{pattern}else{self.reference_state(request).map_or(0.,|s|self.bare_trap(s).frequencies.iter().copied().fold(0.,f64::max))};
        let steps=(self.ground.duration*rate/crate::solve::first_phase()).max(1.);
        let advance=(0..self.layouts.len()).map(|k|crate::solve::advance_work(&self.work(k))).fold(0.,f64::max);
        2.*MOST_TRAJECTORIES as f64*steps*advance/POPULATION_TARGET
    }

    /// The varying units other than the trap beam's (ground.rs `Unit`): each one's profile is judged
    /// over the atom's reach, the region of the trap within ln(1/budget) k_B T of its bottom (the
    /// harmonic trap's thermal ellipsoid holding all but the budget of a Boltzmann ensemble), T the
    /// request's -- at its ends along the principal axes, and inside it for interfering beams
    /// (ground.rs `reach_points`): its intensity changes there by ε from the focus's at worst, and its
    /// shifts O_u + F_u, taken at the focus, err by ε ‖O_u + F_u‖ against the nearest excited pole and
    /// the generator's scale. Within the budget the unit keeps its beams' exact profiles in their
    /// fields (near beams) and its shifts at the focus; beyond, an intensity axis (at most
    /// `MOST_FOCUSED` in all, the largest errors first).
    fn choose_profiles(&mut self,request:&Request)->Result<(),String>{
        let g=&self.ground;let reference=self.reference_state(request)?;
        let trap=self.bare_trap(reference);
        let reach:[f64;3]=std::array::from_fn(|a|{let w=trap.frequencies[a];
            if w>0.{(2.*BUDGET.recip().ln()*KT_PER_UK*request.temperature_uk*self.hbar_over_mass).sqrt()/w}else{0.}});
        self.reach.extend(ground::reach_points(&trap.axes,reach,g.units.iter().any(|u|u.beams.len()>1)));
        let mut errors:Vec<(usize,f64,f64)>=vec![];
        for(u,unit)in g.units.iter().enumerate(){
            if u==self.focused[0]||!unit.varies(&g.beams){continue;}
            let fall=self.reach.iter().map(|&r|(self.unit_intensity(u,r)-unit.focus).abs()).fold(0_f64,f64::max);
            let mut shift=g.outside[u].clone();shift.add_scaled(Complex::ONE,&g.folds[u]);
            let size=shift.spectral_norm();
            let weight=size*(1./g.pole.max(f64::MIN_POSITIVE)).max(1./g.generator_scale.max(f64::MIN_POSITIVE));
            errors.push((u,fall*weight,weight));
        }
        errors.sort_by(|a,b|b.1.total_cmp(&a.1));
        for(u,error,weight)in errors{
            if error>BUDGET&&self.focused.len()<MOST_FOCUSED{self.focused.push(u);}else{self.profiles.push((u,error,weight));}
        }
        Ok(())
    }
    /// The profile error at r: over the units taken at the focus, the largest intensity changed there
    /// times its weight.
    pub fn profile_error(&self,r:[f64;3])->f64{
        self.profiles.iter().map(|&(u,_,weight)|(self.unit_intensity(u,r)-self.ground.units[u].focus).abs()*weight).fold(0.,f64::max)
    }
    /// A unit's intensity at r (ground.rs `Unit`), from the beams' geometry.
    pub fn unit_intensity(&self,u:usize,r:[f64;3])->f64{
        let unit=&self.ground.units[u];unit.beams.iter().zip(&unit.weights).fold(Complex::ZERO,|s,(&l,&w)|s+self.geometry[l].field(r).0*w).norm_sqr()/unit.norm
    }

    /// The reference state: the request's largest initial population, a ground state (its position
    /// in `Ground::ground_states`).
    pub fn reference_state(&self,request:&Request)->Result<usize,String>{
        let rho=&request.initial_density;
        self.ground.ground_states.iter().enumerate().max_by(|a,b|rho[a.1*STATES+a.1].re.total_cmp(&rho[b.1*STATES+b.1].re)).map(|x|x.0).ok_or("No ground state".into())
    }
    /// The trap a ground state sees in H(r) alone (its energy, the folds and outside shifts at r):
    /// its Hessian at the focus by central differences over a thousandth of the trap's waist.
    fn bare_trap(&self,state:usize)->Trap{
        let g=&self.ground;let i=g.ground_states[state];let h=1e-3*self.geometry[self.trap_beam].waist.unwrap();
        let u=|r:[f64;3]|g.hamiltonian_at(r)[(i,i)].re;
        self.trap_of(&u,h)
    }
    /// The trap of a potential u(r) (rad/us): its Hessian at the focus, eigen-decomposed.
    fn trap_of(&self,u:&dyn Fn([f64;3])->f64,h:f64)->Trap{
        let e=|a:usize|{let mut v=[0.;3];v[a]=h;v};let add=|x:[f64;3],y:[f64;3]|std::array::from_fn(|i|x[i]+y[i]);let neg=|x:[f64;3]|x.map(|v|-v);
        let u0=u([0.;3]);
        let hessian:[[f64;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|if a==b{(u(e(a))+u(neg(e(a)))-2.*u0)/(h*h)}
            else{(u(add(e(a),e(b)))-u(add(e(a),neg(e(b))))-u(add(neg(e(a)),e(b)))+u(add(neg(e(a)),neg(e(b)))))/(4.*h*h)}));
        let(values,axes)=symmetric_eigen(hessian);
        let w=|x:f64|(x*self.hbar_over_mass).max(0.).sqrt();
        Trap{axes,frequencies:values.map(w),lab:std::array::from_fn(|a|w(hessian[a][a])),lattice:self.lattice()}
    }
    /// The trap's standing wave, where the trap beam's unit is it and its reflection.
    fn lattice(&self)->Option<Lattice>{
        let unit=&self.ground.units[self.focused[0]];if unit.beams.len()!=2||!unit.beams.iter().all(|l|self.trap_beams.contains(l)){return None;}
        let(a,b)=(&self.geometry[unit.beams[0]],&self.geometry[unit.beams[1]]);let waist=a.waist?;
        if (0..3).map(|i|a.direction[i]*b.direction[i]).sum::<f64>()>0.{return None;}
        let z=unit.weights[0]*unit.weights[1].conj();
        Some(Lattice{axis:a.direction,wave_number:a.wave_number,rayleigh:a.wave_number*waist*waist/2.,phase:0.5*z.im.atan2(z.re)})
    }
    /// The trap of the model's tables for a cluster's state (its potential with the interference left
    /// out, `potential`).
    pub fn trap(&self,cluster:usize,x:usize)->Trap{
        let h=1e-3*self.geometry[self.trap_beam].waist.unwrap();self.trap_of(&|r|self.potential(cluster,x,r),h)
    }

    /// Each cluster's table layout: h0, its groups split by residual beat, its carriers' photon
    /// tables (on its ground states) and its decay cross terms; and its explicit states' terms
    /// without a table, the kept drives (U_k at the focus) and the decay between two of them.
    fn layouts(&self,mixing:&[Vec<Vec<Complex>>])->Vec<Layout>{
        let g=&self.ground;let(ng,ne)=(g.ground_states.len(),g.excited_states.len());
        // Per carrier its basis U_k at the focus (ne × ng), the kept drives' entries.
        let basis:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|mixing[c].iter().map(|g_k|{let mut m=Dense::zeros(ne,ng);
            for(j,&l)in self.carriers[c].beams.iter().enumerate(){m.add_scaled(g_k[j],&g.couplings[l]);}m}).collect()).collect();
        g.clusters.iter().enumerate().map(|(k,cl)|{
            let(n,n0)=(cl.dimension(),cl.states.len());let mut size=n*n;let h0=0;let mut groups=vec![];
            for gr in &cl.groups{
                let mut residuals:Vec<f64>=gr.entries.iter().map(|e|e.2).collect();residuals.sort_by(f64::total_cmp);residuals.dedup();
                for residual in residuals{
                    let hermitian=gr.absorbed==gr.emitted&&residual==0.;
                    let pairs=field_pairs(hermitian,self.carriers[gr.emitted].rank,self.carriers[gr.absorbed].rank);
                    let mut entries:Vec<(usize,usize)>=gr.entries.iter().filter(|x|x.2==residual).map(|x|(x.0,x.1)).collect();
                    entries.sort();entries.dedup();
                    let square=pairs.len()*entries.len();groups.push(GroupLayout{absorbed:gr.absorbed,emitted:gr.emitted,residual,hermitian,pairs,entries,s:size,q:size+square,decay:false});size+=2*square;
                }
            }
            // The folded photons a class joins without a kept coupling (ground.rs `FoldedCross`): their decay,
            // in groups of their own per carrier pair and beat.
            let mut keyed:Vec<(usize,usize,f64,Vec<(usize,usize)>)>=vec![];
            for x in &cl.folded_crosses{match keyed.iter_mut().find(|k|k.0==x.absorbed&&k.1==x.emitted&&k.2==x.residual){
                Some(k)=>k.3.push((x.row,x.col)),None=>keyed.push((x.absorbed,x.emitted,x.residual,vec![(x.row,x.col)]))}}
            for(absorbed,emitted,residual,mut entries)in keyed{
                entries.sort();entries.dedup();let hermitian=absorbed==emitted&&residual==0.;
                let pairs=field_pairs(hermitian,self.carriers[emitted].rank,self.carriers[absorbed].rank);
                let square=pairs.len()*entries.len();groups.push(GroupLayout{absorbed,emitted,residual,hermitian,pairs,entries,s:size,q:size+square,decay:true});size+=2*square;
            }
            let carriers=self.carriers.iter().map(|c|{let jump=3*c.rank*ng*n0;let l=CarrierLayout{j:size,j2:size+jump,t:size+2*jump};size+=2*jump+c.rank*ne*n0;l}).collect();
            // The explicit states' decay, −(i/2) Σ_φ J_φ†J_φ on their rows: between two of them, and
            // between one and a ground state whose folded photons share its class and landing state.
            let(mut decays,mut crosses):(Vec<DecayLayout>,Vec<CrossLayout>)=(vec![],vec![]);
            for class in &cl.classes{for &(i,to,wi)in &class.direct{
                for &(j,to2,wj)in &class.direct{if to2!=to{continue;}let residual=wj-wi;
                    match decays.iter_mut().find(|x|x.row==i&&x.col==j&&x.residual==residual){Some(x)=>x.tos.push(to),None=>decays.push(DecayLayout{row:i,col:j,residual,tos:vec![to],at:0})}
                }
                for &(p,wp)in &class.photons{if p.to!=to{continue;}
                    let(col,residual)=(cl.position(p.from).unwrap(),wp-wi);
                    match crosses.iter_mut().find(|x|x.row==i&&x.col==col&&x.carrier==p.carrier&&x.residual==residual){
                        Some(x)=>x.tos.push(to),None=>crosses.push(CrossLayout{row:i,col,carrier:p.carrier,residual,tos:vec![to],at:0})}
                }
            }}
            for x in crosses.iter_mut(){x.at=size;size+=self.carriers[x.carrier].rank;}
            for x in decays.iter_mut(){x.at=size;size+=1;}
            // The dressing's drives: from a ground state of the cluster through a folded excited state
            // H couples to an explicit one, per carrier that drives it there.
            let folded_static=|a:usize,b:usize|g.dressed[a].contains(&b);
            let mut effective=vec![];
            for(p,&e)in cl.excited.iter().enumerate(){
                let reached:Vec<usize>=(0..ne).filter(|&f|!cl.excited.contains(&f)&&folded_static(e,f)).collect();
                if reached.is_empty(){continue;}
                for(x,&b)in cl.states.iter().enumerate(){for(c,carrier)in self.carriers.iter().enumerate(){
                    if !reached.iter().any(|&f|carrier.beams.iter().any(|&l|g.folded[l][(f,b)]!=Complex::ZERO)){continue;}
                    let(m,mx)=(&cl.combinations[n0+p],&cl.combinations[x]);
                    let residual=g.beat(0.,&(0..m.len()).map(|i|m[i]-mx[i]-i32::from(i==c)).collect::<Vec<i32>>());
                    effective.push(EffectiveLayout{row:n0+p,col:x,carrier:c,residual,at:size});size+=carrier.rank;
                }}
            }
            let emission=size;size+=3*ng*cl.excited.len();
            let drives:Vec<DriveLayout>=cl.drives.iter().map(|&(row,col,c,residual)|DriveLayout{row,col,carrier:c,residual,
                u:std::array::from_fn(|k|if k<self.carriers[c].rank{basis[c][k][(cl.excited[row-n0],cl.states[col])]}else{Complex::ZERO})}).collect();
            // The fourth order: a cluster of ground states, on the carriers whose folded drives reach it.
            let mut fourth=(self.fourth&&cl.excited.is_empty()).then(||self.fourth_layout(k,&mut size));
            // The beats of every term's residual, a residual static within `same_frame`'s tolerance at the
            // cluster's largest frame and detuning (the round-off of the frames' differences); the fourth order's
            // kept entries beat at θ_a − θ_b + ν (`FourthLayout::residuals`).
            let largest=cl.frames.iter().chain(&cl.excited_frames).map(|x|x.abs()).fold(0.,f64::max)+g.carriers.iter().map(|c|c.detuning.abs()).fold(0.,f64::max);
            let quartic:Vec<f64>=fourth.as_ref().map_or(vec![],|f|f.residuals(&cl.frames));
            let residuals=groups.iter().map(|x|x.residual).chain(drives.iter().map(|x|x.residual)).chain(crosses.iter().map(|x|x.residual))
                .chain(decays.iter().map(|x|x.residual)).chain(effective.iter().map(|x|x.residual)).chain(quartic);
            let beats=Beats::new(residuals,64.*f64::EPSILON*2.*largest);
            if let Some(f)=fourth.as_mut(){f.place(&cl.frames,&beats);}
            Layout{n,n0,size,h0,groups,carriers,crosses,drives,decays,effective,emission,fourth,beats}
        }).collect()
    }

    /// Every unit's intensity with the focused ones at s, the others at the focus.
    pub fn unit_intensities(&self,s:&[f64])->Vec<f64>{
        let mut intensities=self.ground.focus_intensities();for(i,&u)in self.focused.iter().enumerate(){intensities[u]=s[i];}intensities
    }
    /// H at the focused units' intensities s, the others at the focus.
    pub fn hamiltonian_at(&self,s:&[f64])->Dense{self.ground.hamiltonian_with(&self.unit_intensities(s))}

    /// Every cluster's tables at the intensities s, and its bounds there (`Node`).
    fn node(&self,s:&[f64],mixing:&[Vec<Vec<Complex>>])->Result<Node,String>{
        let g=&self.ground;let intensities=self.unit_intensities(s);let e=g.eliminate(&intensities)?;let(ne,ng)=(g.excited_states.len(),g.ground_states.len());
        // Per carrier: U^f_k (the folded basis), T_k, T2_k (ne × ng).
        let combine=|c:usize,matrices:&[Dense]|->Vec<Dense>{mixing[c].iter().map(|g_k|{let mut m=Dense::zeros(ne,ng);
            for(j,&l)in self.carriers[c].beams.iter().enumerate(){m.add_scaled(g_k[j],&matrices[l]);}m}).collect()};
        let u:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&g.folded)).collect();
        let t:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&e.responses)).collect();
        let t2:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&e.derivatives)).collect();
        let jump=|m:&Dense|->[Dense;3]{std::array::from_fn(|a|g.emission[a].product(m))};
        let(j,j2):(Vec<Vec<[Dense;3]>>,Vec<Vec<[Dense;3]>>)=(t.iter().map(|ts|ts.iter().map(jump).collect()).collect(),t2.iter().map(|ts|ts.iter().map(jump).collect()).collect());
        let mut out=vec![];
        for(cluster,(cl,layout))in g.clusters.iter().zip(&self.layouts).enumerate(){
            let(n,n0,states)=(layout.n,layout.n0,&cl.states);let mut node=vec![Complex::ZERO;layout.size];
            // h0: the energies less the frames, the kept static couplings (their residual zero) and
            // the folded static couplings' van Vleck shift, of the ground and the explicit states.
            for x in 0..n0{node[layout.h0+x*n+x]=Complex::new(e.energies[states[x]]-cl.frames[x],0.);}
            for(p,&f)in cl.excited.iter().enumerate(){node[layout.h0+(n0+p)*n+n0+p]=Complex::new(e.excited_energies[f]-cl.excited_frames[p],0.);}
            for &(x,y)in &cl.statics{node[layout.h0+x*n+y]+=e.ground[(states[x],states[y])];}
            for &(x,y,f)in &cl.static_paths{let(a,b)=(states[x],states[y]);
                node[layout.h0+x*n+y]+=e.ground[(a,f)]*e.ground[(f,b)]*(0.5*(1./(e.energies[a]-e.energies[f])+1./(e.energies[b]-e.energies[f])));}
            for &(x,y)in &cl.excited_statics{node[layout.h0+x*n+y]+=e.excited[(cl.excited[x-n0],cl.excited[y-n0])];}
            for &(x,y,f)in &cl.excited_paths{let(a,b)=(cl.excited[x-n0],cl.excited[y-n0]);let ee=&e.excited_energies;
                node[layout.h0+x*n+y]+=e.excited[(a,f)]*e.excited[(f,b)]*(0.5*(1./(ee[a]-ee[f])+1./(ee[b]-ee[f])));}
            // The kept groups: S and Q on their entries; the decay groups: Q, the photons' product over the landing
            // states their class shares.
            for gl in layout.groups.iter().filter(|g|g.decay){
                let(c,d)=(gl.absorbed,gl.emitted);
                for(p,&(k1,k0))in gl.pairs.iter().enumerate(){let base=p*gl.entries.len();
                    for(at,&(x,y))in gl.entries.iter().enumerate(){
                        let tos=&cl.folded_crosses.iter().find(|i|i.row==x&&i.col==y&&i.absorbed==c&&i.emitted==d&&i.residual==gl.residual).unwrap().tos;let(a,b)=(states[x],states[y]);
                        node[gl.q+base+at]=(0..3).fold(Complex::ZERO,|s,ax|tos.iter().fold(s,|s,&to|s+j[d][k1][ax][(to,a)].conj()*j[c][k0][ax][(to,b)]));
                    }
                }
            }
            for gl in layout.groups.iter().filter(|g|!g.decay){
                let(c,d)=(gl.absorbed,gl.emitted);
                let entries:Vec<(usize,usize)>=cl.groups.iter().find(|x|x.absorbed==c&&x.emitted==d).unwrap().entries.iter().filter(|x|x.2==gl.residual).map(|x|(x.0,x.1)).collect();
                for(p,&(k1,k0))in gl.pairs.iter().enumerate(){
                    let base=p*gl.entries.len();
                    for &(x,y)in &entries{
                        if x>=n0{
                            // Two explicit states: their folded drives' van Vleck shift (c = d, no Q).
                            let(ea,eb,delta)=(cl.excited[x-n0],cl.excited[y-n0],g.carriers[c].detuning);let mut sv=Complex::ZERO;
                            for b in 0..ng{sv-=u[c][k0][(ea,b)]*u[c][k1][(eb,b)].conj()*(0.5*(1./(e.excited_energies[ea]-delta-e.energies[b])+1./(e.excited_energies[eb]-delta-e.energies[b])));}
                            node[gl.s+base+gl.entries.binary_search(&(x,y)).unwrap()]=sv;continue;
                        }
                        let(a,b)=(states[x],states[y]);let(mut sv,mut qv)=(Complex::ZERO,Complex::ZERO);
                        for i in 0..ne{sv+=(u[d][k1][(i,a)].conj()*t[c][k0][(i,b)]+t[d][k1][(i,a)].conj()*u[c][k0][(i,b)])*0.5;}
                        let mut dt=vec![Complex::ZERO;ne];
                        for i in 0..ne{for i2 in 0..ne{dt[i]+=g.decay[(i,i2)]*t[c][k0][(i2,b)];}}
                        for i in 0..ne{qv+=t[d][k1][(i,a)].conj()*dt[i];}
                        let j=gl.entries.binary_search(&(x,y)).unwrap();node[gl.s+base+j]=sv;node[gl.q+base+j]=qv;
                    }
                }
            }
            for(c,cl2)in layout.carriers.iter().enumerate(){
                for kk in 0..self.carriers[c].rank{for ax in 0..3{for row in 0..ng{for y in 0..n0{
                    let at=((kk*3+ax)*ng+row)*n0+y;node[cl2.j+at]=j[c][kk][ax][(row,states[y])];node[cl2.j2+at]=j2[c][kk][ax][(row,states[y])];
                }}}}
                for kk in 0..self.carriers[c].rank{for i in 0..ne{for y in 0..n0{node[cl2.t+(kk*ne+i)*n0+y]=t[c][kk][(i,states[y])];}}}
            }
            // The explicit states: their dressing x (ground.rs `Ground::dressing`) and dressed photons
            // ĉ_a[to] = C_a[to,e] + Σ_f C_a[to,f] x_f; the decays and cross terms of those; the drives
            // the dressing brings, Y_k = ½(−H_eF T_k[F,b] + x_F† U^f_k[F,b]).
            if !cl.excited.is_empty(){
                let flags:Vec<bool>=(0..ne).map(|f|cl.excited.contains(&f)).collect();
                let dressings:Vec<Vec<Complex>>=cl.excited.iter().map(|&f|g.dressing(&e,f,&flags)).collect::<Result<_,_>>()?;
                let photon:Vec<[Vec<Complex>;3]>=cl.excited.iter().zip(&dressings).map(|(&f,x)|std::array::from_fn(|a|(0..ng).map(|to|
                    g.emission[a][(to,f)]+(0..ne).fold(Complex::ZERO,|s,q|s+g.emission[a][(to,q)]*x[q])).collect())).collect();
                for(p,ph)in photon.iter().enumerate(){for a in 0..3{for to in 0..ng{node[layout.emission+(p*3+a)*ng+to]=ph[a][to];}}}
                for x in &layout.decays{let(i,j)=(x.row-n0,x.col-n0);
                    node[x.at]=x.tos.iter().fold(Complex::ZERO,|s,&to|s+(0..3).fold(Complex::ZERO,|s,a|s+photon[i][a][to].conj()*photon[j][a][to]))*Complex::new(0.,-0.5);}
                for x in &layout.crosses{let i=x.row-n0;
                    let w:Vec<Complex>=(0..ne).map(|q|x.tos.iter().fold(Complex::ZERO,|s,&to|s+(0..3).fold(Complex::ZERO,|s,a|s+photon[i][a][to].conj()*g.emission[a][(to,q)]))).collect();
                    for kk in 0..self.carriers[x.carrier].rank{node[x.at+kk]=(0..ne).fold(Complex::ZERO,|s,q|s+w[q]*t[x.carrier][kk][(q,states[x.col])]);}
                }
                for x in &layout.effective{let(p,b)=(x.row-n0,states[x.col]);let f=cl.excited[p];let d=&dressings[p];
                    for kk in 0..self.carriers[x.carrier].rank{
                        node[x.at+kk]=(0..ne).filter(|&q|!flags[q]).fold(Complex::ZERO,|s,q|s-e.excited[(f,q)]*t[x.carrier][kk][(q,b)]+d[q].conj()*u[x.carrier][kk][(q,b)])*0.5;
                    }
                }
            }
            if let Some(f)=&layout.fourth{self.fourth_node(&e,cluster,f,&u,&t,&mut node)?;}
            out.push(node);
        }
        // The elimination's excited weight per ground state, every near beam at its peak (`resonance`).
        let resonance=(0..ng).map(|b|self.carriers.iter().flat_map(|c|c.beams.iter()).map(|&l|(0..ne).map(|i|e.responses[l][(i,b)].norm_sqr()).sum::<f64>()).sum::<f64>()).fold(0.,f64::max);
        let(dressing,undressed)=g.dressing_bounds(&e,&intensities)?;
        Ok((out,[dressing,resonance,undressed]))
    }

    /// A cluster of ground states' fourth-order layout (`FourthLayout`), its tables from `size` on: H^(4)'s
    /// entries kept where the second order would keep a coupling at their beat Ω (ground.rs `fourth_keeps`);
    /// another one's mixing m = b/(|Ω| − l0) booked, b = S l0 the size of H^(4).
    fn fourth_layout(&self,k:usize,size:&mut usize)->FourthLayout{
        let g=&self.ground;let cl=&g.clusters[k];let(n,ng)=(cl.states.len(),g.ground_states.len());let unmixed=|c:usize|g.unmixed.get(c).is_some_and(|x|x[k]);let r=|c:usize|self.carriers[c].rank;let delta=|c:usize|g.carriers[c].detuning;
        let(on,b)=g.fourth_carriers(cl);let l0=g.generator_scale;
        let fields:Vec<(usize,usize)>=on.iter().flat_map(|&c|(0..r(c)).map(move|k|(c,k))).collect();
        let pairs:Vec<(usize,usize)>=(0..fields.len()).flat_map(|i|(i..fields.len()).map(move|j|(i,j))).collect();
        let round_off=64.*f64::EPSILON*2.*(cl.frames.iter().map(|x|x.abs()).fold(0.,f64::max)+g.carriers.iter().map(|c|c.detuning.abs()).fold(0.,f64::max));
        let mut left:Vec<(usize,usize,f64)>=vec![];
        let mut keep=|nu:f64|->Vec<bool>{(0..n*n).map(|i|{
            let(x,y)=(i/n,i%n);let omega=cl.frames[x]-cl.frames[y]+nu;
            if g.fourth_keeps(cl,x,y,omega,b){return true;}
            let(a,c)=(x.min(y),x.max(y));if !left.iter().any(|&(p,q,w)|p==a&&q==c&&(w-omega.abs()).abs()<=round_off){left.push((a,c,omega.abs()));}
            false}).collect()};
        let mut quartic=vec![];
        for &(p1,p2)in &pairs{for &(q1,q2)in &pairs{
            let nu=g.frequency(&[fields[p1].0,fields[p2].0],&[fields[q1].0,fields[q2].0]);let k=keep(nu);
            if k.iter().any(|&x|x){quartic.push(QuarticSlab{conj:(p1,p2),plain:(q1,q2),nu,keep:k,part:vec![],at:*size});*size+=n*n;}
        }}
        let dropped=left.iter().fold(0.,|s,&(_,_,w)|s+crate::ground::mixing(w,b,l0).powi(2));
        // The metric's and Π's slabs per carrier pair, the metric on the coherent entries (the readout's).
        let group=|c:usize,s:usize|cl.coherent.iter().position(|gr|gr.iter().any(|x|x.0.carrier==c&&x.0.state==s));
        let mut slow=vec![];
        for &c1 in &on{for &c2 in &on{
            let coherent=(0..n*n).map(|i|{let(x,y)=(cl.states[i/n],cl.states[i%n]);let gx=group(c1,x);gx.is_some()&&gx==group(c2,y)}).collect();
            let square=r(c1)*r(c2)*n*n;slow.push(SlowSlab{carriers:(c1,c2),nu:delta(c1)-delta(c2),coherent,rr:*size,yw:*size+square});*size+=2*square;
        }}
        // The jumps' W3 at each carrier's frequency, through the triples (c1, c2, c4) at it, each middle factor
        // less what the dressing holds (`Middle`).
        let(mut jumps,mut p)=(vec![],Vec::<(usize,usize,Middle,usize)>::new());
        for &c in &on{for &c1 in &on{
            let triples:Vec<(usize,usize)>=on.iter().flat_map(|&c2|on.iter().map(move|&c4|(c2,c4))).filter(|&(c2,c4)|crate::ground::same_frame(g.frequency(&[c1,c4],&[c2]),delta(c))).collect();
            if triples.is_empty(){continue;}
            let pairs=triples.into_iter().map(|(c2,c4)|{let kind=Middle::of(c1,c2,c4,unmixed(c1));
                p.iter().position(|x|x.0==c2&&x.1==c4&&x.2==kind).unwrap_or_else(||{p.push((c2,c4,kind,*size));*size+=r(c2)*r(c4)*ng*n;p.len()-1})}).collect();
            let slab=JumpSlab{photon:c,inner:c1,pairs,diagonal:unmixed(c),at:*size};*size+=r(c1)*3*ng*ng*n*slab.sources(n);jumps.push(slab);
        }}
        // Where two photon channels' jumps meet: the landing states their photons share a class in (ground.rs `Class`)
        // -- a coherent pair's every one, a folded cross's (`FoldedCross`) --, as the jumps make their products.
        let nc=g.carriers.len();let mut landing=vec![0u32;nc*n*nc*n];
        for class in &cl.classes{for &(x,_)in &class.photons{for &(y,_)in &class.photons{if x.to!=y.to{continue;}
            let(a,b)=(cl.position(x.from).unwrap(),cl.position(y.from).unwrap());landing[((x.carrier*n+a)*nc+y.carrier)*n+b]|=1<<x.to;}}}
        FourthLayout{carriers:on,fields,quartic,slow,jumps,p,landing,dropped}
    }
    /// A cluster of ground states' fourth-order tables at a node (`FourthLayout`), from the elimination e
    /// there and the carriers' folded bases u and responses t (ne × ng per field index). With G_b(ω) the
    /// fourth order's resolvents (ground.rs `fourth_column`: at a carrier's frequency its block's dressed eigen-energy
    /// terms, from each source place to every column), the fields f3, f1, f2, f4 (carriers c3, c1, c2, c4), ω = δ_c1 − δ_c2 + δ_c4, every ground
    /// state b′ between and the middle factor M_f2f4 = U_f2†T_f4 less what the dressing holds (`Middle`): the van
    /// Vleck form's U_f3[:,a]† G_b(ω) T_f1[:,b′] M_f2f4[b′,b] and −½[X_f3f1, U_f2†T_f4] (X = T†T on the cluster's
    /// coherent entries), its Hermitian half on the slab of the conjugated pair {f3, f2} and the plain {f1, f4}
    /// and the adjoint's on the swapped one; the jumps' fourth order of Σ L̃†L̃ -- −(C T_f3)[to,a]* (C G_b(ω) T_f1
    /// M_f2f4)[to,b] at a carrier's frequency ω and its adjoint, −½{X, T†C†C T} --, over the landing states the two
    /// photons share a class in (`FourthLayout::landing`), times −i/2, on the same slabs; R_c1†R_c2 on the
    /// coherent entries and YW_c1c2 = −U_c1[:,a]† G_a(δ_c1) T_c2[:,b]; and the jumps' tables.
    fn fourth_node(&self,e:&ground::Elimination,k:usize,f:&FourthLayout,u:&[Vec<Dense>],t:&[Vec<Dense>],node:&mut [Complex])->Result<(),String>{
        let g=&self.ground;let cl=&g.clusters[k];let(n,ng,ne,nc)=(cl.states.len(),g.ground_states.len(),g.excited_states.len(),g.carriers.len());let st=&cl.states;let fl=&f.fields;let nf=fl.len();
        let delta=|c:usize|g.carriers[c].detuning;let field=|c:usize,k:usize|fl.iter().position(|&x|x==(c,k)).unwrap();
        let photon=|w:f64|f.carriers.iter().any(|&c|crate::ground::same_frame(delta(c),w));
        // The triples' frequencies, each once, and G_b(ω) T_f[:,b′] for the cluster's b and every b′.
        let mut omegas:Vec<f64>=vec![];
        for &c1 in &f.carriers{for &c2 in &f.carriers{for &c4 in &f.carriers{let w=g.frequency(&[c1,c4],&[c2]);if !omegas.iter().any(|&x|crate::ground::same_frame(x,w)){omegas.push(w);}}}}
        let omega_of=|w:f64|omegas.iter().position(|&x|crate::ground::same_frame(x,w)).unwrap();
        // G(ω) on T_f[:,b′] from each source place b to every place x (ground.rs `fourth_column`: at a photon's carrier its
        // block's eigen-energy terms, P_E[b,x]; else the bare G_b(ω) on x = b alone).
        let gt:Vec<Vec<Vec<Vec<Vec<Vec<Complex>>>>>>=omegas.iter().map(|&w|(0..n).map(|b|fl.iter().map(|&(c,kk)|(0..ng).map(|bp|
            g.fourth_column(e,k,b,w,&(0..ne).map(|i|t[c][kk][(i,bp)]).collect::<Vec<_>>(),false)).collect::<Result<Vec<_>,_>>()).collect::<Result<Vec<_>,_>>()).collect::<Result<Vec<_>,_>>())
            .collect::<Result<Vec<_>,_>>()?;// [ω][b][f][b′][x] → ne
        // The photons' amplitudes on every axis and landing state (axis-major, 3 ng): C T_f[:,s], and at a carrier's
        // frequency C G_b(ω) T_f[:,b′].
        let emit=|v:&[Complex]|->Vec<Complex>{(0..3*ng).map(|q|{let(ax,to)=(q/ng,q%ng);(0..ne).fold(Complex::ZERO,|s,i|s+g.emission[ax][(to,i)]*v[i])}).collect()};
        let ct:Vec<Vec<Vec<Complex>>>=fl.iter().map(|&(c,k)|(0..ng).map(|s|emit(&(0..ne).map(|i|t[c][k][(i,s)]).collect::<Vec<_>>())).collect()).collect();// [f][s]
        let cgt:Vec<Vec<Vec<Vec<Vec<Vec<Complex>>>>>>=omegas.iter().enumerate().map(|(w,&om)|if photon(om){
            (0..n).map(|b|(0..nf).map(|fi|(0..ng).map(|bp|(0..n).map(|x|emit(&gt[w][b][fi][bp][x])).collect()).collect()).collect()).collect()}else{vec![]}).collect();// [ω][b][f][b′][x]
        let full=|x:&Dense,y:&Dense|->Vec<Complex>{let mut m=vec![Complex::ZERO;ng*ng];for i in 0..ng{for j in 0..ng{m[i*ng+j]=(0..ne).fold(Complex::ZERO,|s,q|s+x[(q,i)].conj()*y[(q,j)]);}}m};
        let ut:Vec<Vec<Vec<Complex>>>=fl.iter().map(|&(c,k)|fl.iter().map(|&(c2,k2)|full(&u[c][k],&t[c2][k2])).collect()).collect();
        let tt:Vec<Vec<Vec<Complex>>>=fl.iter().map(|&(c,k)|fl.iter().map(|&(c2,k2)|full(&t[c][k],&t[c2][k2])).collect()).collect();
        // Two photon channels' product in the decay: over the landing states their photons share a class in, as the
        // jumps make it (`FourthLayout::landing`).
        let shared=|ca:usize,a:usize,cb:usize,b:usize|f.landing[((ca*n+a)*nc+cb)*n+b];
        let meet=|mask:u32,x:&[Complex],y:&[Complex]|->Complex{(0..3*ng).filter(|q|(mask>>(q%ng))&1==1).fold(Complex::ZERO,|s,q|s+x[q].conj()*y[q])};
        let tdt:Vec<Vec<Vec<Complex>>>=(0..nf).map(|i|(0..nf).map(|j|(0..n*n).map(|q|{let(a,b)=(q/n,q%n);
            meet(shared(fl[i].0,a,fl[j].0,b),&ct[i][st[a]],&ct[j][st[b]])}).collect()).collect()).collect();// [f][f′][a n + b]
        // X on the cluster: T_f†T_f′ on the coherent entries of the carriers' pair.
        let coherent=|c1:usize,c2:usize|&f.slow.iter().find(|s|s.carriers==(c1,c2)).unwrap().coherent;
        let xs:Vec<Vec<Vec<Complex>>>=(0..nf).map(|i|(0..nf).map(|j|{let mask=coherent(fl[i].0,fl[j].0);
            (0..n*n).map(|q|if mask[q]{tt[i][j][st[q/n]*ng+st[q%n]]}else{Complex::ZERO}).collect()}).collect()).collect();
        // The middle factor of a triple (c1, c2, c4) for the fields f2, f4 on column b, over b′ (`Middle`).
        let middle=|kind:Middle,f2:usize,f4:usize,b:usize|->Vec<Complex>{let s=st[b];let(c2,c4)=(fl[f2].0,fl[f4].0);
            let mut m:Vec<Complex>=(0..ng).map(|bp|ut[f2][f4][bp*ng+s]).collect();
            match kind{Middle::Whole=>{},
                Middle::Vias=>{for via in &g.vias[c4][s]{if via.carrier==c2{m[via.state]=Complex::ZERO;}}},
                // The field's Hermitian part of c2's light shift, (V_c2†R_c2)[b′,b], that c1's block holds, out: b's own (an unmixed
                // block's), or between the cluster's states; its anti-Hermitian part stays.
                Middle::Shift=>{m[s]=(ut[f2][f4][s*ng+s]-ut[f4][f2][s*ng+s].conj())*0.5;},
                Middle::Shifts=>{for &bp in st{m[bp]=(ut[f2][f4][bp*ng+s]-ut[f4][f2][s*ng+bp].conj())*0.5;}}}
            m};
        // A(ω)[f3][f1][((x n + a) n + b) ng + b′] = U_f3[:,a]† (G(ω) T_f1[:,b′] from place b)[x].
        let a_t:Vec<Vec<Vec<Vec<Complex>>>>=(0..omegas.len()).map(|w|(0..nf).map(|f3|{let(c3,k3)=fl[f3];(0..nf).map(|f1|{let mut m=vec![Complex::ZERO;n*n*n*ng];
            for x in 0..n{for a in 0..n{for b in 0..n{for bp in 0..ng{let v=&gt[w][b][f1][bp][x];m[((x*n+a)*n+b)*ng+bp]=(0..ne).fold(Complex::ZERO,|s,i|s+u[c3][k3][(i,st[a])].conj()*v[i]);}}}}
            m}).collect()}).collect()).collect();
        // The slabs by their pairs' indices (i ≤ j of nf fields).
        let pair=|i:usize,j:usize|{let(i,j)=(i.min(j),i.max(j));i*nf-i*(i+1)/2+j};let np=nf*(nf+1)/2;
        let mut index=vec![usize::MAX;np*np];for(s,q)in f.quartic.iter().enumerate(){index[pair(q.conj.0,q.conj.1)*np+pair(q.plain.0,q.plain.1)]=s;}
        let mut acc:Vec<Vec<Complex>>=vec![vec![Complex::ZERO;n*n];f.quartic.len()];
        let adjoint=|x:&[Complex]|->Vec<Complex>{(0..n*n).map(|q|x[(q%n)*n+q/n].conj()).collect()};
        let add=|acc:&mut Vec<Vec<Complex>>,conj:usize,plain:usize,x:&[Complex],z:Complex|{let s=index[conj*np+plain];if s!=usize::MAX{for q in 0..n*n{acc[s][q]+=x[q]*z;}}};
        let(mut x,mut l,mut m)=(vec![Complex::ZERO;n*n],vec![Complex::ZERO;n*n],vec![Complex::ZERO;n*n]);
        for &c1 in &f.carriers{for &c2 in &f.carriers{for &c4 in &f.carriers{
            let w=omega_of(g.frequency(&[c1,c4],&[c2]));let jumps=photon(omegas[w]);let kind=Middle::of(c1,c2,c4,g.unmixed.get(c1).is_some_and(|x|x[k]));
            // The photon's carrier at the triple's frequency (the jumps' W3^c).
            let carrier=f.carriers.iter().copied().find(|&c|crate::ground::same_frame(delta(c),omegas[w]));
            for f1 in (0..nf).filter(|&i|fl[i].0==c1){for f2 in (0..nf).filter(|&i|fl[i].0==c2){for f4 in (0..nf).filter(|&i|fl[i].0==c4){
                let p24=&ut[f2][f4];let middles:Vec<Vec<Complex>>=(0..n).map(|b|middle(kind,f2,f4,b)).collect();
                // C G(ω) T_f1 M on column x: this triple's W3 photons, from every source place b.
                let photons:Vec<Vec<Complex>>=if jumps{(0..n).map(|x|{let mut v=vec![Complex::ZERO;3*ng];
                    for b in 0..n{for(bp,&z)in middles[b].iter().enumerate(){if z==Complex::ZERO{continue;}for(o,&y)in v.iter_mut().zip(&cgt[w][b][f1][bp][x]){*o+=y*z;}}}v}).collect()}else{vec![]};
                for f3 in 0..nf{
                    let(conj,plain)=(pair(f3,f2),pair(f1,f4));
                    if index[conj*np+plain]==usize::MAX&&index[plain*np+conj]==usize::MAX{continue;}
                    for a in 0..n{for b in 0..n{
                        // Entry (a, b): U_f3[:,a]† W3[:,b], W3's column b from every source place s.
                        let mut z=Complex::ZERO;
                        for(src,mid)in middles.iter().enumerate(){let at=&a_t[w][f3][f1][((b*n+a)*n+src)*ng..][..ng];for(bp,&y)in mid.iter().enumerate(){z+=at[bp]*y;}}
                        let(mut zx,mut zl)=(Complex::ZERO,Complex::ZERO);
                        for q in 0..n{
                            z-=(xs[f3][f1][a*n+q]*p24[st[q]*ng+st[b]]-p24[st[a]*ng+st[q]]*xs[f3][f1][q*n+b])*0.5;
                            zx+=xs[f3][f1][a*n+q]*tdt[f2][f4][q*n+b]+tdt[f3][f1][a*n+q]*xs[f2][f4][q*n+b];
                        }
                        if let Some(c)=carrier.filter(|_|jumps){zl=meet(shared(fl[f3].0,a,c,b),&ct[f3][st[a]],&photons[b]);}
                        x[a*n+b]=z;m[a*n+b]=zx;l[a*n+b]=zl;
                    }}
                    // The Hermitian half: ½X here, ½X† on the swapped slab; −(i/2) of Σ L̃†L̃'s −½{X, T†DT} here.
                    add(&mut acc,conj,plain,&x,Complex::new(0.5,0.));add(&mut acc,plain,conj,&adjoint(&x),Complex::new(0.5,0.));
                    add(&mut acc,conj,plain,&m,Complex::new(0.,0.25));
                    // −(i/2) of Σ L̃†L̃'s −R†D W3 here and −W3†D R on the swapped slab.
                    if jumps{add(&mut acc,conj,plain,&l,Complex::new(0.,0.5));add(&mut acc,plain,conj,&adjoint(&l),Complex::new(0.,0.5));}
                }
            }}}
        }}}
        for(q,v)in f.quartic.iter().zip(&acc){for i in 0..n*n{if q.keep[i]{node[q.at+i]=v[i];}}}
        // The metric's and Π's slabs: YW_c1c2[a,x] = −Σ_b′ Σ_E P_E[a,b′] U_c1[:,b′]† G_b′(z_E) T_c2[:,x], c1's block's terms
        // (ground.rs `fourth_column`, its rows by the projector's transpose).
        for sl in &f.slow{let(c1,c2)=sl.carriers;let r2=self.carriers[c2].rank;
            for k1 in 0..self.carriers[c1].rank{for k2 in 0..r2{let(f1,f2)=(field(c1,k1),field(c2,k2));let base=(k1*r2+k2)*n*n;
                for a in 0..n{for b in 0..n{if sl.coherent[a*n+b]{node[sl.rr+base+a*n+b]=tt[f1][f2][st[a]*ng+st[b]];}}}
                for x in 0..n{let col:Vec<Complex>=(0..ne).map(|i|t[c2][k2][(i,st[x])]).collect();
                    for p in 0..n{for(a,img)in g.fourth_column(e,k,p,delta(c1),&col,true)?.iter().enumerate(){
                        node[sl.yw+base+a*n+x]-=(0..ne).fold(Complex::ZERO,|s,i|s+u[c1][k1][(i,st[p])].conj()*img[i]);}}}
            }}
        }
        // The jumps' tables: C G(δ_c) T_f[:,b′] from each source place b to each column y (a diagonal slab's y from y alone), and the
        // middle factors.
        for js in &f.jumps{let w=omega_of(delta(js.photon));let s=js.sources(n);
            for kk in 0..self.carriers[js.inner].rank{let fi=field(js.inner,kk);for ax in 0..3{for row in 0..ng{for bp in 0..ng{for y in 0..n{for b in 0..s{
                node[js.at+((((kk*3+ax)*ng+row)*ng+bp)*n+y)*s+b]=cgt[w][if js.diagonal{y}else{b}][fi][bp][y][ax*ng+row];
            }}}}}}
        }
        for &(c2,c4,kind,at)in &f.p{let r4=self.carriers[c4].rank;for k2 in 0..self.carriers[c2].rank{for k4 in 0..r4{
            let(f2,f4)=(field(c2,k2),field(c4,k4));let base=at+(k2*r4+k4)*ng*n;
            for b in 0..n{let m=middle(kind,f2,f4,b);for bp in 0..ng{node[base+bp*n+b]=m[bp];}}
        }}}
        Ok(())
    }
    /// Node i's intensities on a grid of n nodes per focused unit (node order: the first unit's
    /// fastest).
    fn grid_point(&self,n:usize,i:usize)->[f64;MOST_FOCUSED]{
        std::array::from_fn(|f|if f<self.focused.len(){((i/n.pow(f as u32))%n) as f64/(n-1).max(1) as f64}else{0.})
    }
    /// The tables of a grid of n nodes per focused unit: the nodes it shares with the grid of
    /// (n + 1)/2 before it (`coarse`: every other node along each axis, at the same intensities to the
    /// last bit) taken from that grid's tables, the others computed, `NODE_BATCH` at a time written
    /// into the tables; and the elimination's bound over the nodes inside the reach's intensities --
    /// each focused unit's from its smallest over the reach (less one node, the interval holding it)
    /// to 1.
    fn grid_tables(&self,n:usize,mixing:&[Vec<Vec<Complex>>],coarse:Option<&Tables>)->Result<Tables,String>{
        let(d,m)=(self.focused.len(),n.div_ceil(2));let count=n.pow(d as u32);
        let mut data:Vec<Vec<f64>>=self.layouts.iter().map(|l|Vec::with_capacity(2*count*l.size)).collect();let mut bounds=Vec::with_capacity(count);
        // A node of the coarse grid it shares, or computed (with its counted work: a node is one thread's).
        enum Made{Shared(usize),Computed(Node,f64)}let mut work=0.;
        for first in (0..count).step_by(NODE_BATCH){
            let batch:Vec<Made>=(first..(first+NODE_BATCH).min(count)).into_par_iter().map(|i|{
                let digits:Vec<usize>=(0..d).map(|f|(i/n.pow(f as u32))%n).collect();
                match coarse{
                    Some(_)if digits.iter().all(|x|x%2==0)=>Ok(Made::Shared(digits.iter().enumerate().map(|(f,x)|x/2*m.pow(f as u32)).sum())),
                    _=>{let before=crate::linear::counted();self.node(&self.grid_point(n,i)[..d],mixing).map(|x|Made::Computed(x,crate::linear::counted()-before))}
                }
            }).collect::<Result<_,_>>()?;
            for made in batch{match made{
                Made::Shared(j)=>{let c=coarse.unwrap();for(k,l)in self.layouts.iter().enumerate(){data[k].extend_from_slice(&c.data[k][2*j*l.size..2*(j+1)*l.size]);}bounds.push(c.bounds[j]);}
                Made::Computed(node,w)=>{for(k,t)in node.0.iter().enumerate(){data[k].extend(t.iter().flat_map(|z|[z.re,z.im]));}bounds.push(node.1);work+=w;}
            }}
        }
        let inside=self.inside(n);
        let reach=std::array::from_fn(|k|inside.iter().map(|&i|bounds[i][k]).fold(0.,f64::max));
        Ok(Tables{nodes:n,data,bounds,reach,work})
    }
    /// The nodes of a grid of n per focused unit inside the reach's intensities: each focused unit's from its
    /// smallest over the reach (less one node, the interval holding it) to 1.
    fn inside(&self,n:usize)->Vec<usize>{
        let low:Vec<f64>=self.focused.iter().map(|&u|self.reach.iter().map(|&r|self.unit_intensity(u,r).min(1.)).fold(1.,f64::min)-1./(n-1).max(1) as f64).collect();
        (0..n.pow(self.focused.len() as u32)).filter(|&i|self.grid_point(n,i).iter().zip(&low).all(|(x,l)|x>=l)).collect()
    }
    /// The coarse grid's (`self.tables`) largest interpolation error at a finer grid's nodes where the atoms go,
    /// relative to the scale of each cluster's table kind there: the nodes inside the reach's intensities (`inside`),
    /// as the elimination's bounds are taken (`grid_tables`). Beyond them -- a Gaussian beam's edge, where its
    /// intensity falls to 0 and the function bends its sharpest (the 3D PGC's error 2e-4 at 257 nodes there, 8e-10
    /// inside; F9's 7.5e-4 at 513 nodes per axis, 1.4e-12 inside) -- an atom is far beyond its thermal
    /// ellipsoid, and the estimates over the reach do not take it either.
    fn interpolation_error(&self,fine:&Tables)->f64{
        let(ng,ne)=(self.ground.ground_states.len(),self.ground.excited_states.len());let d=self.focused.len();
        let inside=self.inside(fine.nodes);let mut worst=0_f64;
        for(b,layout)in self.layouts.iter().enumerate(){
            let n=layout.n;let mut kinds:Vec<(usize,usize)>=vec![(layout.h0,layout.h0+n*n)];
            for gl in &layout.groups{let square=gl.pairs.len()*gl.entries.len();kinds.extend([(gl.s,gl.s+square),(gl.q,gl.q+square)]);}
            for(c,cl)in layout.carriers.iter().enumerate(){let r=self.carriers[c].rank;let(jump,response)=(3*r*ng*layout.n0,r*ne*layout.n0);
                kinds.extend([(cl.j,cl.j+jump),(cl.j2,cl.j2+jump),(cl.t,cl.t+response)]);}
            // The explicit states' tables, each kind one block (its entries' common scale: a decay
            // between two states with orthogonal photons is zero to rounding).
            let block=|slots:Vec<(usize,usize)>|slots.iter().map(|x|x.0).min().zip(slots.iter().map(|x|x.0+x.1).max());
            kinds.extend(block(layout.crosses.iter().map(|x|(x.at,self.carriers[x.carrier].rank)).collect()));
            kinds.extend(block(layout.decays.iter().map(|x|(x.at,1)).collect()));
            kinds.extend(block(layout.effective.iter().map(|x|(x.at,self.carriers[x.carrier].rank)).collect()));
            if layout.n>layout.n0{kinds.push((layout.emission,layout.emission+3*ng*(layout.n-layout.n0)));}
            if let Some(f)=&layout.fourth{
                let r=|c:usize|self.carriers[c].rank;let sq=n*n;
                // H^(4)'s slabs one kind (one scale), the metric's and Π's per pair, the jumps' tables.
                if let(Some(a),Some(b))=(f.quartic.first(),f.quartic.last()){kinds.push((a.at,b.at+sq));}
                for sl in &f.slow{let square=r(sl.carriers.0)*r(sl.carriers.1)*sq;kinds.push((sl.rr,sl.rr+square));kinds.push((sl.yw,sl.yw+square));}
                for js in &f.jumps{kinds.push((js.at,js.at+r(js.inner)*3*ng*ng*n*js.sources(n)));}
                for &(c2,c4,_,at)in &f.p{kinds.push((at,at+r(c2)*r(c4)*ng*n));}
            }
            // Per kind its scale and the largest error, the fine grid's nodes on the threads.
            for(start,end)in kinds{
                let exact=|i:usize|&fine.data[b][2*(i*layout.size+start)..2*(i*layout.size+end)];
                let modulus=|x:&[f64]|x.chunks_exact(2).map(|z|z[0].hypot(z[1])).fold(0_f64,f64::max);
                let scale=inside.par_iter().map(|&i|modulus(exact(i))).reduce(||0.,f64::max);
                if scale==0.{continue;}
                let e=inside.par_iter().map_init(||vec![Complex::ZERO;end-start],|out,&i|{
                    self.interpolate(b,&self.stencil(&self.grid_point(fine.nodes,i)[..d]),None,start,out);
                    out.iter().zip(exact(i).chunks_exact(2)).fold(0_f64,|m,(x,y)|m.max((x.re-y[0]).hypot(x.im-y[1])))
                }).reduce(||0.,f64::max)/scale;
                worst=worst.max(e);
            }
        }
        worst
    }
    /// The tables on grids of FIRST_NODES, then doubled (each grid's nodes computed once), the finer grid
    /// kept with the coarser one's measured error against it (`interpolation_error`, an upper estimate of
    /// its own) -- once refined at least, so that the error is measured --, and doubled again while that
    /// pays: while the next grid's counted work -- its new nodes' eliminations at the last grid's counted
    /// work per node (linear.rs `counted`) and its error's measure, `MATRIX_NS` a complex multiply-add -- is
    /// below what its smaller error saves the run at `stake` per unit of booked error (`stake`). Its error
    /// is predicted by the last doubling's rate, and by the cubic interpolation's order before one is seen,
    /// 2⁻³ a doubling (`cubic`): a doubling that did not shrink the error predicts no saving, and stops it
    /// (`table_stalled`: the tabulated function is not smooth on the grids' scale where the atoms go -- a
    /// folded drive's pole, PGCP_10's every-drive-folded model 0.8 to 1.3 from 9 to 32769 nodes there, which
    /// no grid resolves). A grid beyond the work bounds (`MOST_TABLE_BYTES`, `MOST_TABLE_NODES`) is not made
    /// whatever it saves (`table_limit`). Its error is booked (the ledger's), the room it takes from the
    /// statistics like any estimate's (solve.rs).
    fn tabulate(&mut self,mixing:&[Vec<Vec<Complex>>],most_bytes:usize,most_nodes:usize,stake:f64)->Result<(),String>{
        let d=self.focused.len() as u32;
        let entries:usize=self.layouts.iter().map(|l|l.size).sum();
        let within=|n:usize,coarse:usize|n.pow(d)<=most_nodes&&16*entries*(n.pow(d)+coarse.pow(d))<=most_bytes;
        if !within(2*FIRST_NODES-1,FIRST_NODES){return Err(format!("The elimination's tables take {entries} complex entries per node: beyond the work bounds"));}
        self.tables=self.grid_tables(FIRST_NODES,mixing,None)?;(self.dressing,self.undressed)=(self.tables.reach[0],self.tables.reach[2]);
        // The counted work per computed node, and the errors measured so far.
        let(mut per_node,mut errors)=(self.tables.work/FIRST_NODES.pow(d) as f64,vec![]);
        loop{
            let(coarse,n)=(self.tables.nodes,2*self.tables.nodes-1);
            if let Some(&error)=errors.last(){
                let rate:f64=if errors.len()>1{error/errors[errors.len()-2]}else{0.125};
                if !(rate<1.){self.table_stalled=true;self.table_limit=Some(coarse);break;}
                let work=MATRIX_NS*((n.pow(d)-coarse.pow(d)) as f64*per_node+(n.pow(d)*entries*4_usize.pow(d)) as f64);
                if !(work<stake*error*(1.-rate)){break;}
            }
            if !within(n,coarse){self.table_limit=Some(coarse);break;}
            let fine=self.grid_tables(n,mixing,Some(&self.tables))?;let worst=self.interpolation_error(&fine);
            per_node=fine.work/(n.pow(d)-coarse.pow(d)) as f64;
            (self.dressing,self.undressed)=(self.dressing.max(fine.reach[0]),self.undressed.max(fine.reach[2]));
            self.tables=fine;self.table_error=worst;errors.push(worst);
            if worst==0.{break;}
        }
        self.resonance=self.tables.reach[1];
        Ok(())
    }

    /// The stencil at intensities s (one per focused beam).
    pub fn stencil(&self,s:&[f64])->Stencil{
        let n=self.tables.nodes;let mut out=Stencil{dimensions:self.focused.len(),first:[0;MOST_FOCUSED],weights:[[1.,0.,0.,0.];MOST_FOCUSED],derivatives:[[0.;4];MOST_FOCUSED],stride:[0;MOST_FOCUSED]};
        for f in 0..out.dimensions{let(first,w,d)=cubic(n,s[f]);out.first[f]=first;out.weights[f]=w;out.derivatives[f]=d;out.stride[f]=n.pow(f as u32);}
        out
    }

    /// out = the table slice [start, start + out.len()) of a cluster, interpolated (with the weights'
    /// derivative in s_f when `derivative` names one).
    pub fn interpolate(&self,cluster:usize,stencil:&Stencil,derivative:Option<usize>,start:usize,out:&mut [Complex]){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let len=out.len();out.fill(Complex::ZERO);
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let w=match derivative{None=>w,Some(f)=>dw[f]};let base=node*size+2*start;
            for(o,x)in out.iter_mut().zip(data[base..base+2*len].chunks_exact(2)){o.re+=w*x[0];o.im+=w*x[1];}
        }
    }
    /// The value and the derivatives in every s_f of a slice of `len` values (2 × its entries), in
    /// one pass over the stencil's nodes; the one-beam stencil unrolled.
    #[inline(always)]
    fn gather(&self,cluster:usize,stencil:&Stencil,start:usize,len:usize,value:&mut [f64;MOST_W],derivatives:&mut [[f64;MOST_W];MOST_FOCUSED]){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let base=2*start;let d=stencil.dimensions;
        if d==1{
            let(w,dw)=(stencil.weights[0],stencil.derivatives[0]);let at=stencil.first[0]*size+base;
            let(a,b,c,e)=(&data[at..at+len],&data[at+size..at+size+len],&data[at+2*size..at+2*size+len],&data[at+3*size..at+3*size+len]);
            for j in 0..len{value[j]=w[0]*a[j]+w[1]*b[j]+w[2]*c[j]+w[3]*e[j];derivatives[0][j]=dw[0]*a[j]+dw[1]*b[j]+dw[2]*c[j]+dw[3]*e[j];}
            return;
        }
        value[..len].fill(0.);for g in 0..d{derivatives[g][..len].fill(0.);}
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let at=node*size+base;let x=&data[at..at+len];
            for j in 0..len{value[j]+=w*x[j];}for g in 0..d{for j in 0..len{derivatives[g][j]+=dw[g]*x[j];}}
        }
    }
    /// The value alone (the stencil with the trap beam off).
    #[inline(always)]
    fn gather_value(&self,cluster:usize,stencil:&Stencil,start:usize,len:usize,value:&mut [f64;MOST_W]){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let base=2*start;
        value[..len].fill(0.);
        for i in 0..stencil.count(){let(node,w,_)=stencil.node(i);if w==0.{continue;}let at=node*size+base;for(o,x)in value[..len].iter_mut().zip(&data[at..at+len]){*o+=w*x;}}
    }

    /// Everything at (r, κ) but the internal state: per focused unit its intensity s = |F|²/N
    /// (F = Σ_l a_l γ_l, ground.rs `Unit`), its gradient 2 Re(F* ∇F)/N and its intensity with the
    /// trap's light off (`trap_beams`).
    pub fn point(&self,r:[f64;3],kappa:[f64;3])->Point{
        let(mut s,mut off)=([0.;MOST_FOCUSED],[0.;MOST_FOCUSED]);let mut gradient=[[0.;3];MOST_FOCUSED];
        for(i,&u)in self.focused.iter().enumerate(){
            let unit=&self.ground.units[u];let(mut field,mut slope,mut rest)=(Complex::ZERO,[Complex::ZERO;3],Complex::ZERO);
            for(&l,&w)in unit.beams.iter().zip(&unit.weights){let(a,g)=self.geometry[l].field(r);let x=a*w;field+=x;for d in 0..3{slope[d]+=g[d]*w;}if !self.trap_beams.contains(&l){rest+=x;}}
            s[i]=(field.norm_sqr()/unit.norm).min(1.);off[i]=(rest.norm_sqr()/unit.norm).min(1.);gradient[i]=slope.map(|x|2.*(field.conj()*x).re/unit.norm);
        }
        let mut fields=[CarrierField::default();MOST_CARRIERS];
        for(c,f)in self.carriers.iter().zip(fields.iter_mut()){
            for(j,&l)in c.beams.iter().enumerate(){
                let(a,g)=self.geometry[l].field(r);let w=self.kinetic_shift(l,kappa);
                for k in 0..c.rank{let x=c.coefficients[j][k];let ax=a*x;f.e[k]+=ax;f.doppler[k]+=ax*w;for d in 0..3{f.grad[k][d]+=g[d]*x;}}
            }
        }
        let d=self.focused.len();let stencil=self.stencil(&s[..d]);
        Point{r,kappa,stencil,far:self.stencil(&off[..d]),intensity_gradient:gradient,fields}
    }

    /// w_l = (ħ/m) k_l·κ + ħ|k_l|²/2m (rad/us): the kinetic energy absorbing from beam l gives.
    pub fn kinetic_shift(&self,l:usize,kappa:[f64;3])->f64{
        if !self.doppler{return 0.;}
        let g=&self.geometry[l];let k=g.wave_vector();self.hbar_over_mass*(k[0]*kappa[0]+k[1]*kappa[1]+k[2]*kappa[2]+0.5*g.wave_number*g.wave_number)
    }
    /// The first order in the Doppler shift's expansion parameter at κ: the largest (w_l/σ_c)² over the
    /// near beams, w_l the kinetic shift and σ_c its carrier's nearest excited pole (ground.rs).
    pub fn doppler_ratio(&self,kappa:[f64;3])->f64{
        self.carriers.iter().zip(&self.ground.poles).flat_map(|(c,&pole)|c.beams.iter().map(move|&l|(l,pole))).map(|(l,pole)|(self.kinetic_shift(l,kappa)/pole).powi(2)).fold(0.,f64::max)
    }

    /// A cluster's dimension.
    pub fn dimension(&self,cluster:usize)->usize{self.layouts[cluster].n}
    /// A cluster's work per call in complex multiply-adds, from its tables' layout (the step choice's
    /// cost, solve.rs): the generator (`local`: per pair its S with the slopes and its Q gathered on the
    /// stencil's nodes, added -- twice in a Hermitian group -- and kept for the force; h0 likewise), the
    /// force (`force`: the products once per group, a form per pair and slope, h0's slopes), a jump's
    /// photons (`photons`: per carrier, rank and axis J and J2 on the stencil and their sum) and the
    /// picked class's absorbed light (`Photons::absorption`: per photon and rank J and J2 along the
    /// polarization and their slopes on the stencil), its classes' mean; at fourth order H^(4)'s slabs
    /// in the generator and W3, the metric and Π in a jump.
    pub fn work(&self,cluster:usize)->Work{
        let layout=&self.layouts[cluster];let d=self.focused.len() as f64;let nodes=(1usize<<(2*self.focused.len())) as f64;
        let(n2,ng)=((layout.n*layout.n) as f64,self.ground.ground_states.len() as f64);
        let(mut generator,mut force)=(nodes*(1.+d)*n2+n2,d*n2);
        for gl in &layout.groups{
            let(e,pairs)=(gl.entries.len() as f64,gl.pairs.len() as f64);let added=if gl.hermitian{4.}else{2.};
            generator+=pairs*(nodes*(2.+d)+added+1.+d)*e;force+=e+pairs*(1.+d)*e;
        }
        let mut photons=self.carriers.iter().map(|c|c.rank as f64*3.*ng*layout.n as f64*(2.*nodes+1.)).sum();
        // The fourth order: H^(4)'s slabs per generator; per jump W3's tables and forms, the metric on the
        // state and Π after the jump (`FourthLayout`).
        if let Some(f)=&layout.fourth{
            let r=|c:usize|self.carriers[c].rank as f64;let n=layout.n as f64;
            generator+=f.quartic.len() as f64*nodes*n2+n2;
            let pairs:f64=f.slow.iter().map(|s|r(s.carriers.0)*r(s.carriers.1)).sum();
            photons+=f.jumps.iter().map(|j|r(j.inner)*3.*ng*ng*n*j.sources(layout.n) as f64*(nodes+1.)).sum::<f64>()+f.p.iter().map(|&(c2,c4,_,_)|r(c2)*r(c4)*ng*n*nodes).sum::<f64>()+3.*pairs*nodes*n2;
        }
        let classes=&self.ground.clusters[cluster].classes;
        let absorption=classes.iter().map(|c|c.photons.iter().map(|(ph,_)|self.carriers[ph.carrier].rank as f64*(6.+nodes*(3.+d))).sum::<f64>()).sum::<f64>()/classes.len().max(1) as f64;
        Work{generator,force,photons,absorption,pairs:layout.groups.iter().map(|g|g.pairs.len()).sum()}
    }
    /// The number of clusters.
    pub fn clusters(&self)->usize{self.layouts.len()}
    /// Whether a cluster has explicit excited states.
    pub fn mixed(&self,cluster:usize)->bool{self.layouts[cluster].n>self.layouts[cluster].n0}
    /// Whether a cluster's generator beats: a term keeps a residual frequency in its frame.
    pub fn beating(&self,cluster:usize)->bool{!self.layouts[cluster].beats.omegas.is_empty()}
    /// A cluster's beats (rad/us), ascending and closed under negation, and their common period (us)
    /// where they are harmonics of one frequency.
    pub fn beats(&self,cluster:usize)->(&[f64],Option<f64>){let b=&self.layouts[cluster].beats;(&b.omegas,b.period)}
    /// A cluster's number of ground states (the first of its basis).
    pub fn ground_dimension(&self,cluster:usize)->usize{self.layouts[cluster].n0}
    /// The part of a cluster's components a residual goes into (`components`): 0 the static one, 1 + j beat j's.
    pub(crate) fn part(&self,cluster:usize,residual:f64)->usize{self.layouts[cluster].beats.of(residual).map_or(0,|j|j+1)}
    /// A cluster's terms of its explicit states for the GPU's components (gpu/program.rs `Term`), in
    /// `components`' order: per kept drive (kind 0), decay cross term (1), decay (2) and dressing's drive
    /// (3) its row, column, carrier, part, its mirror's part (the residual's negative), its tables' place
    /// and a drive's U_k; and where the explicit states' dressed photons lie (`photon`).
    pub(crate) fn exact_terms(&self,cluster:usize)->(Vec<[usize;7]>,Vec<[Complex;MOST_RANK]>,usize){
        let l=&self.layouts[cluster];let part=|w:f64|self.part(cluster,w);let(mut terms,mut u)=(vec![],vec![]);
        for d in &l.drives{terms.push([0,d.row,d.col,d.carrier,part(d.residual),part(-d.residual),0]);u.push(d.u);}
        for c in &l.crosses{terms.push([1,c.row,c.col,c.carrier,part(c.residual),part(-c.residual),c.at]);u.push([Complex::ZERO;MOST_RANK]);}
        for d in &l.decays{terms.push([2,d.row,d.col,0,part(d.residual),part(-d.residual),d.at]);u.push([Complex::ZERO;MOST_RANK]);}
        for e in &l.effective{terms.push([3,e.row,e.col,e.carrier,part(e.residual),part(-e.residual),e.at]);u.push([Complex::ZERO;MOST_RANK]);}
        (terms,u,l.emission)
    }

    /// The table slice [start, start + value.len()) of a cluster at a stencil, and with `slopes` its
    /// derivatives in every s_f, in one pass over the stencil's nodes.
    fn gather_dyn(&self,cluster:usize,stencil:&Stencil,start:usize,value:&mut [Complex],mut slopes:Option<&mut [Vec<Complex>]>){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let len=value.len();let d=stencil.dimensions;
        value.fill(Complex::ZERO);if let Some(s)=slopes.as_deref_mut(){for g in 0..d{s[g][..len].fill(Complex::ZERO);}}
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let base=node*size+2*start;let x=&data[base..base+2*len];
            for(o,z)in value.iter_mut().zip(x.chunks_exact(2)){o.re+=w*z[0];o.im+=w*z[1];}
            if let Some(s)=slopes.as_deref_mut(){for g in 0..d{for(o,z)in s[g].iter_mut().zip(x.chunks_exact(2)){o.re+=dw[g]*z[0];o.im+=dw[g]*z[1];}}}
        }
    }
    /// A cluster's generator at a point and time t on its n states (`Generator`): a static cluster of
    /// ground states' through `local`, every other one's through its beats' components (`components`).
    pub fn generator(&self,cluster:usize,p:&Point,t:f64)->Generator{
        if !self.mixed(cluster)&&!self.beating(cluster){
            let local=self.local(cluster,p,t);let n=self.dimension(cluster);
            return Generator{n,hamiltonian:local.hamiltonian[..n*n].to_vec(),scalar:local.scalar,gradient:self.gradients(cluster,&local)};
        }
        self.components(cluster,p).at(t)
    }
    /// A cluster's generator at a point split by its beats (`Components`): its tables' terms, the kept
    /// drives, the decay cross terms and the explicit states' decay (module documentation), and a cluster
    /// of ground states' fourth order, each into the part of its residual -- the static one, or its beat's.
    pub fn components(&self,cluster:usize,p:&Point)->Components{
        let layout=&self.layouts[cluster];let(n,d)=(layout.n,self.focused.len());let nn=n*n;let beats=&layout.beats;let parts=1+beats.omegas.len();
        // The part of a residual: 0 the static one, 1 + j beat j's.
        let part=|w:f64|beats.of(w).map_or(0,|j|j+1);
        let zeros=||vec![Complex::ZERO;nn];
        let mut gradient:Vec<[Vec<Complex>;3]>=(0..parts).map(|_|std::array::from_fn(|_|zeros())).collect();
        let(mut m,mut gamma):(Vec<Vec<Complex>>,Vec<Vec<Complex>>)=((0..parts).map(|_|zeros()).collect(),(0..parts).map(|_|zeros()).collect());
        let mut buf=zeros();let mut slope=vec![zeros();d];
        // acc += c X, and with `adjoint` c X + (c X)† (a Hermitian group's pair k1 < k standing for both);
        // a group's tables hold its entries alone (`scatter`).
        let add=|c:Complex,x:&[Complex],acc:&mut [Complex],adjoint:bool|{for a in 0..n{for b in 0..n{acc[a*n+b]+=c*x[a*n+b];if adjoint{acc[a*n+b]+=(c*x[b*n+a]).conj();}}}};
        let scatter=|c:Complex,x:&[Complex],entries:&[(usize,usize)],acc:&mut [Complex],adjoint:bool|{for(j,&(a,b))in entries.iter().enumerate(){acc[a*n+b]+=c*x[j];if adjoint{acc[b*n+a]+=(c*x[j]).conj();}}};
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let q=part(gl.residual);let count=gl.entries.len();
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                let at=pi*count;let ee=e.e[k1].conj()*f.e[k];let both=gl.hermitian&&k1<k;
                self.gather_dyn(cluster,&p.stencil,gl.s+at,&mut buf[..count],Some(&mut slope));
                scatter(ee,&buf[..count],&gl.entries,&mut m[q],both);
                // ∇H_herm ∋ −∇(Φ) S − Φ Σ_f ∂_f S ∇s_f.
                for a in 0..3{
                    let c=e.grad[k1][a].conj()*f.e[k]+e.e[k1].conj()*f.grad[k][a];scatter(-c,&buf[..count],&gl.entries,&mut gradient[q][a],both);
                    for g in 0..d{scatter(-ee*p.intensity_gradient[g][a],&slope[g][..count],&gl.entries,&mut gradient[q][a],both);}
                }
                self.gather_dyn(cluster,&p.stencil,gl.q+at,&mut buf[..count],None);scatter(ee,&buf[..count],&gl.entries,&mut gamma[q],both);
            }
        }
        self.gather_dyn(cluster,&p.stencil,layout.h0,&mut buf,Some(&mut slope));
        for a in 0..3{for g in 0..d{let c=Complex::new(p.intensity_gradient[g][a],0.);add(c,&slope[g],&mut gradient[0][a],false);}}
        // The static part less its scalar (its Hermitian part's mean diagonal, a phase); a beat's diagonal,
        // a phase that beats, stays in its part.
        let scalar=(0..n).map(|a|buf[a*n+a].re-m[0][a*n+a].re).sum::<f64>()/n as f64;
        let mut h:Vec<Vec<Complex>>=(0..parts).map(|q|(0..nn).map(|i|(if q==0{buf[i]}else{Complex::ZERO})-m[q][i]-gamma[q][i]*Complex::new(0.,0.5)).collect()).collect();
        for a in 0..n{h[0][a*n+a]-=Complex::new(scalar,0.);}
        // The fourth order (a cluster of ground states): each slab's kept entries in their beats' parts
        // (`fourth_parts`; the force keeps the second order's).
        self.fourth_parts(cluster,p,&mut h);
        // The kept drives, Σ_k E_k U_k e^{iΩ̃t} and its adjoint; their gradients the stimulated force.
        for dr in &layout.drives{
            let fl=&p.fields[dr.carrier];let(q,mirror)=(part(dr.residual),part(-dr.residual));let rank=self.carriers[dr.carrier].rank;
            let v=(0..rank).fold(Complex::ZERO,|s,k|s+fl.e[k]*dr.u[k]);h[q][dr.row*n+dr.col]+=v;h[mirror][dr.col*n+dr.row]+=v.conj();
            for a in 0..3{let g=(0..rank).fold(Complex::ZERO,|s,k|s+fl.grad[k][a]*dr.u[k]);gradient[q][a][dr.row*n+dr.col]+=g;gradient[mirror][a][dr.col*n+dr.row]+=g.conj();}
        }
        // The decay cross terms, (i/2) e^{iΩ̃t} Σ_k E_k X_k and its mirror; the explicit states' decay.
        let mut x=[Complex::ZERO;MOST_RANK];
        for cr in &layout.crosses{
            let rank=self.carriers[cr.carrier].rank;self.gather_dyn(cluster,&p.stencil,cr.at,&mut x[..rank],None);
            let z=(0..rank).fold(Complex::ZERO,|s,k|s+p.fields[cr.carrier].e[k]*x[k]);
            h[part(cr.residual)][cr.row*n+cr.col]+=z*Complex::new(0.,0.5);h[part(-cr.residual)][cr.col*n+cr.row]+=z.conj()*Complex::new(0.,0.5);
        }
        let mut z1=[Complex::ZERO];
        for x in &layout.decays{self.gather_dyn(cluster,&p.stencil,x.at,&mut z1,None);h[part(x.residual)][x.row*n+x.col]+=z1[0];}
        // The drives the dressing brings: Σ_k E_k Y_k e^{iΩ̃t} and its adjoint, and their gradients.
        let(mut y,mut dy)=([Complex::ZERO;MOST_RANK],vec![vec![Complex::ZERO;MOST_RANK];d]);
        for x in &layout.effective{
            let rank=self.carriers[x.carrier].rank;let fl=&p.fields[x.carrier];let(q,mirror)=(part(x.residual),part(-x.residual));
            self.gather_dyn(cluster,&p.stencil,x.at,&mut y[..rank],Some(&mut dy));
            let v=(0..rank).fold(Complex::ZERO,|s,k|s+fl.e[k]*y[k]);h[q][x.row*n+x.col]+=v;h[mirror][x.col*n+x.row]+=v.conj();
            for a in 0..3{
                let mut g=(0..rank).fold(Complex::ZERO,|s,k|s+fl.grad[k][a]*y[k]);
                for f in 0..d{g+=(0..rank).fold(Complex::ZERO,|s,k|s+fl.e[k]*dy[f][k])*p.intensity_gradient[f][a];}
                gradient[q][a][x.row*n+x.col]+=g;gradient[mirror][a][x.col*n+x.row]+=g.conj();
            }
        }
        // The gradients' Hermitian parts (the force reads Re⟨ψ|·|ψ⟩ alone): the static one's, and per pair of
        // beats ±Ω, (G_Ω + G_{−Ω}†)/2 at Ω and its adjoint at −Ω, so that G(t) is Hermitian at every t.
        for g in gradient[0].iter_mut(){for a in 0..n{for b in a..n{let z=(g[a*n+b]+g[b*n+a].conj())*0.5;g[a*n+b]=z;g[b*n+a]=z.conj();}}}
        let half=beats.omegas.len()/2;
        for i in 0..half{
            let(up,down)=(1+half+i,half-i);let(low,high)=gradient.split_at_mut(up);
            for x in 0..3{let(u,v)=(&mut high[0][x],&mut low[down][x]);
                for a in 0..n{for b in 0..n{let z=(u[a*n+b]+v[b*n+a].conj())*0.5;u[a*n+b]=z;v[b*n+a]=z.conj();}}}
        }
        let mut gradient=gradient.into_iter();let mut h=h.into_iter();
        let statics=Generator{n,hamiltonian:h.next().unwrap(),scalar,gradient:gradient.next().unwrap()};
        let beats_out=beats.omegas.iter().zip(h.zip(gradient)).map(|(&omega,(hamiltonian,gradient))|Beat{omega,hamiltonian,gradient}).collect();
        Components{statics,beats:beats_out,period:beats.period}
    }

    /// A cluster's generator at a point and time t (`Local`): a cluster of ground states' (one with
    /// explicit excited states has `generator`).
    pub fn local(&self,cluster:usize,p:&Point,t:f64)->Local{
        assert!(!self.mixed(cluster),"a cluster with explicit excited states has no Local");
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>if self.wide{unsafe{self.local_wide::<$n,{2*$n*$n}>(cluster,p,t)}}else{self.local_body::<$n,{2*$n*$n}>(cluster,p,t)},)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        let mut out=sized!(1 2 3 4 5 6 7 8);
        // The fourth order: H^(4) in the cluster's frames (the force keeps the second order's: T's
        // error stays the fold's S, the ledger's).
        if let Some(h)=self.fourth_hamiltonian(cluster,p,t){let n=self.dimension(cluster);for i in 0..n*n{out.hamiltonian[i]+=h[i];}}
        out
    }
    /// A cluster's fourth-order part of H_eff at a point and time (n × n, row-major), None without its
    /// tables: over its slabs the monomial E*_p E*_p′ E_q E_q′ in its beat's phase e^{iνt} times the slab, one
    /// pass over the stencil's nodes, each entry (a, b) then in the frames' phase e^{i(θ_a − θ_b)t} (the
    /// entries left out are zero in the tables).
    pub fn fourth_hamiltonian(&self,cluster:usize,p:&Point,t:f64)->Option<[Complex;MOST_STATES*MOST_STATES]>{
        let f=self.layouts[cluster].fourth.as_ref()?;let n=self.dimension(cluster);let cl=&self.ground.clusters[cluster];
        let e=|i:usize|{let(c,k)=f.fields[i];p.fields[c].e[k]};let mut rotation=Rotation::new(t);
        let mut terms:Vec<(usize,Complex)>=Vec::with_capacity(f.quartic.len());
        for q in &f.quartic{let m=e(q.conj.0).conj()*e(q.conj.1).conj()*e(q.plain.0)*e(q.plain.1);if m!=Complex::ZERO{terms.push((q.at,m*rotation.at(q.nu)));}}
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let mut h=[Complex::ZERO;MOST_STATES*MOST_STATES];let nn=n*n;
        for i in 0..p.stencil.count(){
            let(node,w,_)=p.stencil.node(i);let row=&data[node*size..(node+1)*size];
            for &(at,m)in &terms{let c=m*w;let x=&row[2*at..2*(at+nn)];
                for(o,z)in h[..nn].iter_mut().zip(x.chunks_exact(2)){o.re+=c.re*z[0]-c.im*z[1];o.im+=c.re*z[1]+c.im*z[0];}}
        }
        for a in 0..n{for b in 0..n{if a!=b{h[a*n+b]=h[a*n+b]*Complex::cis((cl.frames[a]-cl.frames[b])*t);}}}
        Some(h)
    }
    /// A cluster's H^(4) at a point added into its beats' parts (`components`): each slab's monomial times its
    /// kept entries, each into its part (`QuarticSlab::part`).
    fn fourth_parts(&self,cluster:usize,p:&Point,h:&mut [Vec<Complex>]){
        let Some(f)=self.layouts[cluster].fourth.as_ref()else{return};let nn=self.dimension(cluster).pow(2);
        let e=|i:usize|{let(c,k)=f.fields[i];p.fields[c].e[k]};let mut slab=vec![Complex::ZERO;nn];
        for q in &f.quartic{
            let m=e(q.conj.0).conj()*e(q.conj.1).conj()*e(q.plain.0)*e(q.plain.1);if m==Complex::ZERO{continue;}
            self.interpolate(cluster,&p.stencil,None,q.at,&mut slab);
            for i in 0..nn{if q.keep[i]{h[q.part[i]][i]+=m*slab[i];}}
        }
    }
    /// out += a carrier pair's slab at `at` at a point and time (n × n): Σ E^c1*_k1 E^c2_k2 M_k1k2 in its
    /// beat's phase e^{iνt} and the frames' e^{i(θ_a − θ_b)t}.
    fn fourth_pair(&self,cluster:usize,p:&Point,t:f64,sl:&SlowSlab,at:usize,out:&mut [Complex]){
        let n=self.dimension(cluster);let cl=&self.ground.clusters[cluster];let(c1,c2)=sl.carriers;let(r1,r2)=(self.carriers[c1].rank,self.carriers[c2].rank);
        let(e1,e2)=(&p.fields[c1].e,&p.fields[c2].e);let beat=Complex::cis(sl.nu*t);let mut buf=[Complex::ZERO;MOST_STATES*MOST_STATES];
        let phases:Vec<Complex>=(0..n*n).map(|i|Complex::cis((cl.frames[i/n]-cl.frames[i%n])*t)).collect();
        for k1 in 0..r1{for k2 in 0..r2{let m=e1[k1].conj()*e2[k2]*beat;if m==Complex::ZERO{continue;}
            self.interpolate(cluster,&p.stencil,None,at+(k1*r2+k2)*n*n,&mut buf[..n*n]);
            for i in 0..n*n{out[i]+=m*buf[i]*phases[i];}}}
    }
    /// The metric's coherent part X = Σ R_c1†R_c2 on a cluster's coherent entries at a point and time (n × n,
    /// in its frames; the readout's), None without the fourth order's tables.
    pub fn coherent_metric(&self,cluster:usize,p:&Point,t:f64)->Option<Vec<Complex>>{
        let f=self.layouts[cluster].fourth.as_ref()?;let n=self.dimension(cluster);let mut x=vec![Complex::ZERO;n*n];
        for sl in &f.slow{self.fourth_pair(cluster,p,t,sl,sl.rr,&mut x);}
        Some(x)
    }
    /// After a jump into a cluster with fourth-order tables (and at t = 0), the bare state's slow part
    /// Π = 1 + ½X + YW at the point (n × n, module documentation): ψ ← Π f, its weight A = ‖Π f‖² the
    /// threshold's divisor (engine.rs). None without the tables.
    pub fn slow_part(&self,cluster:usize,p:&Point,t:f64)->Option<Vec<Complex>>{
        let f=self.layouts[cluster].fourth.as_ref()?;let n=self.dimension(cluster);let(mut x,mut yw)=(vec![Complex::ZERO;n*n],vec![Complex::ZERO;n*n]);
        for sl in &f.slow{self.fourth_pair(cluster,p,t,sl,sl.rr,&mut x);self.fourth_pair(cluster,p,t,sl,sl.yw,&mut yw);}
        Some((0..n*n).map(|i|x[i]*0.5+yw[i]+if i%(n+1)==0{Complex::ONE}else{Complex::ZERO}).collect())
    }
    /// The fourth order's W3 part of a cluster's photons at a point: per carrier c and axis −C W3^c (ng × n,
    /// row-major), W3^c[:,b] = Σ over the triples (c1, c2, c4) at δ_c of G_b(δ_c) Σ_b′ R_c1[:,b′] (V_c2†R_c4)[b′,b]
    /// (the metric's part acts on the state, `photons`); None without the tables.
    fn fourth_photons(&self,cluster:usize,p:&Point)->Option<Vec<Vec<Complex>>>{
        let f=self.layouts[cluster].fourth.as_ref()?;let(n,ng)=(self.dimension(cluster),self.ground.ground_states.len());let r=|c:usize|self.carriers[c].rank;
        let form=|d:usize,e2:usize,at:usize|->Vec<Complex>{let(ed,ee)=(&p.fields[d].e,&p.fields[e2].e);
            let mut m=vec![Complex::ZERO;ng*n];let mut buf=vec![Complex::ZERO;ng*n];
            for k3 in 0..r(d){for k4 in 0..r(e2){let w=ed[k3].conj()*ee[k4];if w==Complex::ZERO{continue;}self.interpolate(cluster,&p.stencil,None,at+(k3*r(e2)+k4)*ng*n,&mut buf);for i in 0..ng*n{m[i]+=w*buf[i];}}}
            m};
        let forms:Vec<Vec<Complex>>=f.p.iter().map(|&(c2,c4,_,at)|form(c2,c4,at)).collect();
        let mut out=vec![vec![Complex::ZERO;3*ng*n];self.carriers.len()];let mut buf=vec![Complex::ZERO;3*ng*ng*n*n];
        for js in &f.jumps{
            let mut weight=vec![Complex::ZERO;ng*n];
            for &i in &js.pairs{for q in 0..ng*n{weight[q]+=forms[i][q];}}
            let ee=&p.fields[js.inner].e;
            let s=js.sources(n);let size=3*ng*ng*n*s;
            for k in 0..r(js.inner){if ee[k]==Complex::ZERO{continue;}
                self.interpolate(cluster,&p.stencil,None,js.at+k*size,&mut buf[..size]);
                // Column y of W3^c: its sources' places b (a diagonal slab's y alone), each through its middle factor (bp, b).
                for ax in 0..3{for row in 0..ng{for y in 0..n{
                    let z=(0..ng).fold(Complex::ZERO,|z,bp|{let x=&buf[(((ax*ng+row)*ng+bp)*n+y)*s..][..s];
                        if js.diagonal{z+x[0]*weight[bp*n+y]}else{(0..n).fold(z,|z,b|z+x[b]*weight[bp*n+b])}});
                    out[js.photon][(ax*ng+row)*n+y]-=ee[k]*z;
                }}}
            }
        }
        Some(out)
    }
    #[target_feature(enable="avx2,fma")]
    unsafe fn local_wide<const N:usize,const W:usize>(&self,cluster:usize,p:&Point,t:f64)->Local{self.local_body::<N,W>(cluster,p,t)}
    #[inline(always)]
    fn local_body<const N:usize,const W:usize>(&self,cluster:usize,p:&Point,t:f64)->Local{
        let layout=&self.layouts[cluster];let d=self.focused.len();
        let pairs:usize=layout.groups.iter().map(|g|g.pairs.len()).sum();let values:usize=layout.groups.iter().map(|g|g.pairs.len()*2*g.entries.len()).sum();
        let mut out=Local{hamiltonian:[Complex::ZERO;MOST_STATES*MOST_STATES],scalar:0.,values:Vec::with_capacity((1+d)*values),coefficients:Vec::with_capacity(pairs),
            h0:[[0.;MOST_W];MOST_FOCUSED],intensity_gradient:p.intensity_gradient};
        let mut buf=[0.;MOST_W];let mut slope=[[0.;MOST_W];MOST_FOCUSED];
        let(mut m,mut gamma)=([0.;MOST_W],[0.;MOST_W]);
        // The pair (k, k1)'s tables are (k1, k)'s adjoints in a Hermitian group: c X + (c X)† for k1 < k
        // (all of c X added first). The force reads only Re⟨ψ|·|ψ⟩, so its coefficients take 2 c for
        // the two. A group's tables hold its entries alone: a group of every entry is added as
        // matrices, the others entry by entry.
        let adjoint=|c:Complex,x:&[f64;MOST_W],acc:&mut [f64;MOST_W]|{for a in 0..N{for b in 0..N{let(i,j)=(2*(a*N+b),2*(b*N+a));let(u,v)=(x[j],-x[j+1]);acc[i]+=c.re*u+c.im*v;acc[i+1]+=c.re*v-c.im*u;}}};
        let scatter=|c:Complex,x:&[f64;MOST_W],entries:&[(usize,usize)],mirror:bool,acc:&mut [f64;MOST_W]|{
            for(j,&(a,b))in entries.iter().enumerate(){let w=c*Complex::new(x[2*j],x[2*j+1]);let i=2*(a*N+b);acc[i]+=w.re;acc[i+1]+=w.im;}
            if mirror{for(j,&(a,b))in entries.iter().enumerate(){let w=c*Complex::new(x[2*j],x[2*j+1]);let i=2*(b*N+a);acc[i]+=w.re;acc[i+1]-=w.im;}}
        };
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let phase=Complex::cis(gl.residual*t);let count=gl.entries.len();
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                let at=pi*count;let ee=phase*e.e[k1].conj()*f.e[k];let(both,mirror)=if gl.hermitian&&k1<k{(2.,true)}else{(1.,false)};
                // ∇H_herm ∋ −∇(Φ) S − Φ Σ_f ∂_f S ∇s_f: S, its slopes and the coefficients kept.
                let c=|a:usize|phase*(e.grad[k1][a].conj()*f.e[k]+e.e[k1].conj()*f.grad[k][a])*(-both);
                out.coefficients.push([c(0),c(1),c(2),ee*(-both)]);let len=2*count;
                if count==N*N{
                    self.gather(cluster,&p.stencil,gl.s+at,W,&mut buf,&mut slope);
                    caxpy::<W>(ee,&buf,&mut m);if mirror{adjoint(ee,&buf,&mut m);}
                }else{
                    self.gather(cluster,&p.stencil,gl.s+at,len,&mut buf,&mut slope);
                    scatter(ee,&buf,&gl.entries,mirror,&mut m);
                }
                out.values.extend_from_slice(&buf[..len]);for g in 0..d{out.values.extend_from_slice(&slope[g][..len]);}
                if count==N*N{
                    self.gather_value(cluster,&p.stencil,gl.q+at,W,&mut buf);
                    caxpy::<W>(ee,&buf,&mut gamma);if mirror{adjoint(ee,&buf,&mut gamma);}
                }else{
                    self.gather_value(cluster,&p.stencil,gl.q+at,len,&mut buf);
                    scatter(ee,&buf,&gl.entries,mirror,&mut gamma);
                }
            }
        }
        self.gather(cluster,&p.stencil,layout.h0,W,&mut buf,&mut out.h0);
        let at=|v:&[f64;MOST_W],a:usize,b:usize|Complex::new(v[2*(a*N+b)],v[2*(a*N+b)+1]);
        let mut scalar=0.;for a in 0..N{scalar+=buf[2*(a*N+a)]-m[2*(a*N+a)];}scalar/=N as f64;out.scalar=scalar;
        for a in 0..N{for b in 0..N{
            let herm=at(&buf,a,b)-at(&m,a,b);
            out.hamiltonian[a*N+b]=herm-at(&gamma,a,b)*Complex::new(0.,0.5)-if a==b{Complex::new(scalar,0.)}else{Complex::ZERO};
        }}
        out
    }

    /// −∇Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩ (rad/us per um): the Ehrenfest force on ħκ. Its gradient's quadratic
    /// form, Σ_pairs Re(−∇Φ ⟨ψ|S|ψ⟩ − Φ Σ_f ∇s_f ⟨ψ|∂_f S|ψ⟩) + Σ_f ∇s_f ⟨ψ|∂_f h0|ψ⟩, from the forms of
    /// each pair's tables on its entries (`Local`).
    pub fn force(&self,cluster:usize,local:&Local,psi:&[Complex])->[f64;3]{
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>self.force_sized::<$n>(cluster,local,psi),)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        sized!(1 2 3 4 5 6 7 8)
    }
    fn force_sized<const N:usize>(&self,cluster:usize,local:&Local,psi:&[Complex])->[f64;3]{
        let layout=&self.layouts[cluster];let d=self.focused.len();let norm:f64=psi[..N].iter().map(|z|z.norm_sqr()).sum();
        // Per group the products ψ_a* ψ_b on its entries (row-major for a group of every entry); a
        // pair's ⟨ψ|X|ψ⟩ is then their sum weighted by its table X there.
        let mut products=[Complex::ZERO;MOST_STATES*MOST_STATES];
        let mut g=[0.;3];let(mut at,mut pair)=(0,0);
        for gl in &layout.groups{
            let count=gl.entries.len();let len=2*count;
            if count==N*N{for a in 0..N{for b in 0..N{products[a*N+b]=psi[a].conj()*psi[b];}}}
            else{for(j,&(a,b))in gl.entries.iter().enumerate(){products[j]=psi[a].conj()*psi[b];}}
            let form=|x:&[f64]|->Complex{let mut s=Complex::ZERO;for j in 0..count{s+=products[j]*Complex::new(x[2*j],x[2*j+1]);}s};
            for _ in &gl.pairs{
                let[c0,c1,c2,b]=local.coefficients[pair];let sigma=form(&local.values[at..at+len]);
                g[0]+=(c0*sigma).re;g[1]+=(c1*sigma).re;g[2]+=(c2*sigma).re;
                for f in 0..d{let tau=b*form(&local.values[at+(1+f)*len..at+(2+f)*len]);for a in 0..3{g[a]+=tau.re*local.intensity_gradient[f][a];}}
                at+=(1+d)*len;pair+=1;
            }
        }
        for f in 0..d{let eta=quadratic::<N>(&local.h0[f],psi).re;for a in 0..3{g[a]+=local.intensity_gradient[f][a]*eta;}}
        std::array::from_fn(|a|-g[a]/norm)
    }
    /// A cluster of ground states' gradients of the Hermitian part per axis as matrices (n × n,
    /// Hermitian, row-major) from its `Local`'s pair forms -- `force`'s quadratic form, each pair's S and
    /// slopes on its entries with its coefficients, h0's slopes --, for the exact step's `Generator`.
    pub fn gradients(&self,cluster:usize,local:&Local)->[Vec<Complex>;3]{
        let layout=&self.layouts[cluster];let(n,d)=(layout.n,self.focused.len());
        let mut g:[Vec<Complex>;3]=std::array::from_fn(|_|vec![Complex::ZERO;n*n]);let(mut at,mut pair)=(0,0);
        for gl in &layout.groups{
            let len=2*gl.entries.len();
            for _ in &gl.pairs{
                let[c0,c1,c2,b]=local.coefficients[pair];let c=[c0,c1,c2];
                let value=|start:usize,j:usize|Complex::new(local.values[start+2*j],local.values[start+2*j+1]);
                for(j,&(x,y))in gl.entries.iter().enumerate(){
                    let s=value(at,j);
                    for a in 0..3{let mut z=c[a]*s;for f in 0..d{z+=b*value(at+(1+f)*len,j)*local.intensity_gradient[f][a];}g[a][x*n+y]+=z;}
                }
                at+=(1+d)*len;pair+=1;
            }
        }
        for a in 0..3{for f in 0..d{let w=local.intensity_gradient[f][a];for k in 0..n*n{g[a][k]+=Complex::new(local.h0[f][2*k],local.h0[f][2*k+1])*w;}}}
        // Their Hermitian parts: the force reads Re⟨ψ|·|ψ⟩ alone.
        for m in g.iter_mut(){for x in 0..n{for y in x..n{let z=(m[x*n+y]+m[y*n+x].conj())*0.5;m[x*n+y]=z;m[y*n+x]=z.conj();}}}
        g
    }
    /// The trap's potential of ψ at a point (rad/us): Re⟨ψ|H_eff|ψ⟩ less the same with the trap's light
    /// off (`trap_beams`), per ⟨ψ|ψ⟩ -- (h0 − M) at the point's stencil and fields less at its far
    /// stencil with the trap beams' fields taken out of their units' and carriers': the tweezer's
    /// whole light shift whether it is folded (in h0) or a near carrier (in M, through its field) -- a
    /// tweezer alone is always one, the light's frame rotating at it.
    pub fn trap_potential(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64)->f64{
        if self.mixed(cluster){return self.trap_mixed(cluster,p,psi,t,false);}
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>self.trap_sized::<$n,{2*$n*$n}>(cluster,p,psi,t,false),)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        sized!(1 2 3 4 5 6 7 8)
    }
    /// An upper bound of it: h0's share, and the light shift's change at most twice its bound.
    pub fn trap_bound(&self,cluster:usize,p:&Point,psi:&[Complex])->f64{
        if self.mixed(cluster){return self.trap_mixed(cluster,p,psi,0.,true);}
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>self.trap_sized::<$n,{2*$n*$n}>(cluster,p,psi,0.,true),)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        sized!(1 2 3 4 5 6 7 8)
    }
    fn trap_sized<const N:usize,const W:usize>(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64,bound:bool)->f64{
        let layout=&self.layouts[cluster];let(mut here,mut far,mut difference)=([0.;MOST_W],[0.;MOST_W],[0.;MOST_W]);
        self.gather_value(cluster,&p.stencil,layout.h0,W,&mut here);self.gather_value(cluster,&p.far,layout.h0,W,&mut far);
        for j in 0..W{difference[j]=here[j]-far[j];}
        let norm:f64=psi[..N].iter().map(|z|z.norm_sqr()).sum();
        if bound{return quadratic::<N>(&difference,psi).re/norm+2.*self.ground.light;}
        let off=self.fields_off(p);
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let(f0,e0)=(&off[gl.absorbed],&off[gl.emitted]);let phase=Complex::cis(gl.residual*t);
            let count=gl.entries.len();
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                // −(Φ S(s) − Φ_off S(far)) on the group's entries, the pair (k1, k) standing for (k, k1) in a
                // Hermitian group.
                let both=if gl.hermitian&&k1<k{2.}else{1.};let at=gl.s+pi*count;
                self.gather_value(cluster,&p.stencil,at,2*count,&mut here);self.gather_value(cluster,&p.far,at,2*count,&mut far);
                let(c,c0)=(-(e.e[k1].conj()*f.e[k]*both*phase),e0.e[k1].conj()*f0.e[k]*both*phase);
                for(j,&(a,b))in gl.entries.iter().enumerate(){let w=c*Complex::new(here[2*j],here[2*j+1])+c0*Complex::new(far[2*j],far[2*j+1]);let i=2*(a*N+b);difference[i]+=w.re;difference[i+1]+=w.im;}
            }
        }
        quadratic::<N>(&difference,psi).re/norm
    }

    /// An explicit state's dressed photon ĉ_a[to] at a stencil (the p-th explicit state of a cluster).
    pub fn photon(&self,cluster:usize,stencil:&Stencil,p:usize,a:usize,to:usize)->Complex{
        let mut z=[Complex::ZERO];let ng=self.ground.ground_states.len();self.gather_dyn(cluster,stencil,self.layouts[cluster].emission+(p*3+a)*ng+to,&mut z,None);z[0]
    }
    /// The carriers' fields at a point with the trap's light off (`trap_potential`): each one's less
    /// its trap beams' parts a_l(r) c_lk.
    fn fields_off(&self,p:&Point)->[CarrierField;MOST_CARRIERS]{
        let mut off=p.fields;
        for(c,carrier)in self.carriers.iter().enumerate(){
            for(j,&l)in carrier.beams.iter().enumerate(){
                if !self.trap_beams.contains(&l){continue;}
                let a=self.geometry[l].field(p.r).0;for k in 0..carrier.rank{off[c].e[k]-=a*carrier.coefficients[j][k];}
            }
        }
        off
    }
    /// `trap_sized` on any number of states.
    fn trap_mixed(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64,bound:bool)->f64{
        let layout=&self.layouts[cluster];let n=layout.n;let nn=n*n;let(mut here,mut far)=(vec![Complex::ZERO;nn],vec![Complex::ZERO;nn]);
        self.gather_dyn(cluster,&p.stencil,layout.h0,&mut here,None);self.gather_dyn(cluster,&p.far,layout.h0,&mut far,None);
        let mut difference:Vec<Complex>=here.iter().zip(&far).map(|(a,b)|*a-*b).collect();
        let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        let form=|x:&[Complex]|->f64{let mut s=Complex::ZERO;for a in 0..n{for b in 0..n{s+=psi[a].conj()*x[a*n+b]*psi[b];}}s.re};
        if bound{return form(&difference)/norm+2.*self.ground.light;}
        let off=self.fields_off(p);
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let(f0,e0)=(&off[gl.absorbed],&off[gl.emitted]);let count=gl.entries.len();let phase=Complex::cis(gl.residual*t);
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                let both=if gl.hermitian&&k1<k{2.}else{1.};let at=gl.s+pi*count;
                self.gather_dyn(cluster,&p.stencil,at,&mut here[..count],None);self.gather_dyn(cluster,&p.far,at,&mut far[..count],None);
                let(c,c0)=(-(e.e[k1].conj()*f.e[k]*both*phase),e0.e[k1].conj()*f0.e[k]*both*phase);
                for(j,&(a,b))in gl.entries.iter().enumerate(){difference[a*n+b]+=c*here[j]+c0*far[j];}
            }
        }
        form(&difference)/norm
    }

    /// The photon amplitudes at a point of every class of a cluster's jumps (`Photons`), with the
    /// exact responses in a step beyond the first order's allowance (engine.rs): a folded photon's
    /// C_x A_c ψ = −C_x R_c ψ, an explicit state's C_x ψ_e.
    pub fn photons<'a>(&'a self,cluster:usize,p:&Point,psi:&[Complex],t:f64,exact:Option<Exact>)->Photons<'a>{self.photons_to(cluster,p,psi,t,exact,true)}
    /// `photons`, with the fourth order's corrections where `fourth` (else the second order's alone).
    fn photons_to<'a>(&'a self,cluster:usize,p:&Point,psi:&[Complex],t:f64,exact:Option<Exact>,fourth:bool)->Photons<'a>{
        let g=&self.ground;let cl=&g.clusters[cluster];let layout=&self.layouts[cluster];let(n,n0)=(layout.n,layout.n0);let ng=g.ground_states.len();let size=ng*n0;
        // Per carrier, rank and axis the jump tables J_k and J2_k at the point (ng × n0), and per carrier
        // and axis M = Σ_k E_k J_k − E^D_k J2_k, or the exact responses' C_a Σ_l a_l R_l(w_l).
        let ranks:usize=self.carriers.iter().map(|c|c.rank).sum();
        let(mut jumps,mut doppler)=(vec![Complex::ZERO;3*ranks*size],vec![Complex::ZERO;3*ranks*size]);let mut matrices=vec![Complex::ZERO;3*self.carriers.len()*size];
        let mut at=0;
        for(c,cl2)in layout.carriers.iter().enumerate(){
            let fl=&p.fields[c];
            for k in 0..self.carriers[c].rank{for a in 0..3{
                let offset=(k*3+a)*size;
                self.interpolate(cluster,&p.stencil,None,cl2.j+offset,&mut jumps[at..at+size]);self.interpolate(cluster,&p.stencil,None,cl2.j2+offset,&mut doppler[at..at+size]);
                if exact.is_none(){let m=&mut matrices[(c*3+a)*size..(c*3+a+1)*size];for i in 0..size{m[i]+=fl.e[k]*jumps[at+i]-fl.doppler[k]*doppler[at+i];}}
                at+=size;
            }}
            if let Some(exact)=&exact{
                for &l in &self.carriers[c].beams{let amplitude=self.geometry[l].field(p.r).0;
                    for a in 0..3{let m=g.emission[a].product(&exact.responses[l]);let out=&mut matrices[(c*3+a)*size..(c*3+a+1)*size];
                        for row in 0..ng{for y in 0..n0{out[row*n0+y]+=m[(row,cl.states[y])]*amplitude;}}}}
            }
        }
        // The fourth order (the Doppler's first order stays the second's): the photons' W3 part on ψ and their
        // second order on ψ − ½Xψ, X the metric's coherent part (`coherent_metric`) -- the photons of the fourth
        // order's carriers (ground.rs `fourth_carriers`), whose products the tables' decay holds.
        let w3=if fourth{self.fourth_photons(cluster,p)}else{None};
        let bends=|c:usize|layout.fourth.as_ref().is_some_and(|f|f.carriers.contains(&c));
        let bent:Option<[Complex;MOST_STATES]>=if !fourth{None}else{self.coherent_metric(cluster,p,t)}.map(|x|std::array::from_fn(|a|if a<n0{psi[a]-(0..n0).fold(Complex::ZERO,|s,b|s+x[a*n0+b]*psi[b])*0.5}else{Complex::ZERO}));
        // The classes: each photon's entry into its target state's row from its source's column.
        let amplitude=cl.classes.iter().map(|class|{let target=&g.clusters[class.target];
            let mut u=[[Complex::ZERO;MOST_STATES];3];let mut rotation=Rotation::new(t);
            for &(ph,w)in &class.photons{
                let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());let phase=rotation.at(w);
                let state=match &bent{Some(b)if bends(ph.carrier)=>b[col],_=>psi[col]};
                for a in 0..3{u[a][row]+=phase*matrices[(ph.carrier*3+a)*size+ph.to*n0+col]*state;}
                if let Some(w3)=&w3{for a in 0..3{u[a][row]+=phase*w3[ph.carrier][a*size+ph.to*n0+col]*psi[col];}}
            }
            for &(i,to,w)in &class.direct{
                let row=target.position(to).unwrap();let phase=Complex::cis(w*t)*psi[i];
                for a in 0..3{u[a][row]-=phase*self.photon(cluster,&p.stencil,i-n0,a,to);}
            }
            u}).collect();
        let mut state=[Complex::ZERO;MOST_MIXED];state[..n].copy_from_slice(&psi[..n]);
        Photons{model:self,cluster,point:*p,psi:state,time:t,jumps,doppler,exact,amplitude}
    }

    /// A cluster's generator against its jumps at a point and time: D = i(H − H†) − Σ L†L over its classes'
    /// photons, which a Lindblad generator keeps positive semidefinite -- its no-jump decay at least its
    /// jumps' rate in every state -- at rest: the photons' first order in the Doppler and recoil shifts,
    /// which the decay leaves out, aside (its own estimate). At fourth order Σ L̃†L̃ to that order, the decay's
    /// (ground.rs, # Fourth order): L̃ = L + δ, δ the fourth order's corrections, less δ†δ, the sixth order
    /// within the fold's charge. D's smallest eigenvalue, Σ L†L's largest and D's round-off scale, the largest of
    /// i(H − H†)'s and Σ L†L's eigenvalues and of H's entries with the scalar the generator leaves out as a phase: the
    /// tables' entries carry it and their rounding with it, while i(H − H†) cancels it (rad/us). (Without it a lone
    /// state, its H nothing but the rest of its entry's rounding, 3e-20 of 70 rad/us under F9's crossed far beams, read
    /// as a gain beyond its own round-off.)
    pub fn decay_balance(&self,cluster:usize,p:&Point,t:f64)->(f64,f64,f64){let b=self.balance(cluster,p,t);(b.low,b.rate,b.scale)}
    /// `decay_balance` with D and the jumps' whole Σ L̃†L̃ (`Balance`).
    fn balance(&self,cluster:usize,p:&Point,t:f64)->Balance{
        let mut p=*p;for f in p.fields.iter_mut(){f.doppler=[Complex::ZERO;MOST_RANK];}let p=&p;
        let n=self.dimension(cluster);let generator=self.generator(cluster,p,t);let h=generator.hamiltonian;let largest=h.iter().map(|z|z.abs()).fold(0.,f64::max)+generator.scalar.abs();
        let decay=Dense::from_fn(n,n,|a,b|(h[a*n+b]-h[b*n+a].conj())*Complex::I);
        // Column b: the photons of the basis state b.
        let columns=|fourth:bool|->Vec<Vec<[[Complex;MOST_STATES];3]>>{(0..n).map(|b|{let mut psi=[Complex::ZERO;MOST_MIXED];psi[b]=Complex::ONE;self.photons_to(cluster,p,&psi,t,None,fourth).amplitude}).collect()};
        let(full,second)=(columns(true),self.layouts[cluster].fourth.as_ref().map(|_|columns(false)));
        let(mut jumps,mut whole)=(Dense::zeros(n,n),Dense::zeros(n,n));
        // A photon lands in its target's ground states.
        for(c,class)in self.ground.clusters[cluster].classes.iter().enumerate(){let nt=self.ground_dimension(class.target);
            for x in 0..3{for a in 0..n{for b in 0..n{for i in 0..nt{
                let z=full[a][c][x][i].conj()*full[b][c][x][i];jumps[(a,b)]+=z;whole[(a,b)]+=z;
                if let Some(l)=&second{let(da,db)=(full[a][c][x][i]-l[a][c][x][i],full[b][c][x][i]-l[b][c][x][i]);jumps[(a,b)]-=da.conj()*db;}
            }}}}}
        let mut d=decay.clone();d.add_scaled(-Complex::ONE,&jumps);
        let extreme=|m:&Dense|hermitian_eigenvalues(m).into_iter().fold((f64::INFINITY,f64::NEG_INFINITY),|(lo,hi),x|(lo.min(x),hi.max(x)));
        let((low,_),(_,rate),(g0,g1))=(extreme(&d),extreme(&jumps),extreme(&decay));
        Balance{d,jumps:whole,low,rate,scale:rate.abs().max(g0.abs()).max(g1.abs()).max(largest)}
    }
    /// A Lindblad generator where the atoms go, or why not (`balance` at rest, at the focus and the reach's
    /// points -- `reach`, the ends of the trap's principal axes --, at t = 0 and at a generic instant `GENERIC`,
    /// the beats' phases), the tables at their node nearest each point with the point's fields: there they are
    /// the elimination's own, while between nodes D carries their interpolation's error too, the tables' booked
    /// one. The eliminated generator is a Lindblad one by construction at either order (ground.rs: the decay its
    /// jumps' product, the folded photons' cross terms and the fourth order's alike, every frame and beat an exact
    /// combination of the carriers' detunings), so D = i(H − H†) − Σ L̃†L̃ is zero to round-off at a node: below
    /// its round-off it gains -- a decay no jump carries, the norm growing and the jumps stopping (the validation's
    /// F10 before the near light's dressing; b1/109's fourth order before the folded crosses met its W3, 2.4e-3
    /// rad/us) --, a defect of the model, not an approximation the ledger could take: it is refused
    /// (`NOT_LINDBLAD`), with its relative error, the largest ε with −D ≤ ε (Σ L̃†L̃ + 1/T) beyond D's round-off
    /// (`excess`), and the fold's ladder (solve.rs) passes over it -- the solve stops where it is the ladder's first
    /// rung, every drive folded. Between nodes the interpolation's D the solve meets along its pilot's states is
    /// booked (solve.rs `gain`).
    fn lindblad(&self)->Result<(),String>{
        let last=(self.tables.nodes-1).max(1) as f64;
        for &r in &self.reach{
            let mut p=self.point(r,[0.;3]);
            let s:Vec<f64>=(0..p.stencil.dimensions).map(|f|{let w=&p.stencil.weights[f];
                let k=(0..4).max_by(|&a,&b|w[a].abs().total_cmp(&w[b].abs())).unwrap();(p.stencil.first[f]+k) as f64/last}).collect();
            p.stencil=self.stencil(&s);
            for t in [0.,GENERIC]{for k in 0..self.clusters(){
                let b=self.balance(k,&p,t);let round=64.*f64::EPSILON*self.dimension(k) as f64*b.scale;
                if -b.low<=round{continue;}
                let floor=if self.ground.duration>0.{1./self.ground.duration}else{b.rate.max(round)};
                let error=excess(&b.d,&b.jumps,round,floor);
                return Err(format!("{NOT_LINDBLAD}：子空间 {k} 在 r = ({:.2}, {:.2}, {:.2}) μm、t = {t:.3} μs 处，无跳跃的衰减低于跳跃率（i(H − H†) − ΣL†L 的最小本征值 {:.2e} rad/μs，超出舍入误差 {round:.1e} rad/μs；最大跳跃率 {:.2e} rad/μs，相对误差 {:.2e}）——这部分衰减没有跳跃承载，轨迹的范数会增长",
                    r[0],r[1],r[2],b.low,b.rate,error));
            }}
        }
        Ok(())
    }
    /// The 32 populations (request order) of a cluster's state ψ at a point and time: the bare
    /// ground ones, ρ − ½{Y, ρ}, and the excited ones, Σ |Σ_coherent e^{−iε̃t} (R ψ)_i|² and ρ_ee of
    /// the explicit ones, for ρ = ψψ†/‖ψ‖² (ground.rs's readout).
    pub fn populations(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64)->[f64;STATES]{self.readout(cluster,p,psi,t).0}
    /// The populations and the folded excited population among them (the elimination's η²).
    pub fn readout(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64)->([f64;STATES],f64){
        let g=&self.ground;let cl=&g.clusters[cluster];let layout=&self.layouts[cluster];let(n,n0,ne)=(layout.n,layout.n0,g.excited_states.len());
        let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        let mut out=[0.;STATES];let mut y=[Complex::ZERO;MOST_STATES];let mut folded=0.;
        // Per carrier its response at the point, R_c = Σ_k E_k T_k (ne × n0).
        let responses=self.responses(cluster,p);
        for group in &cl.coherent{
            let mut e=vec![Complex::ZERO;ne];
            for &(x,w)in group{let col=cl.position(x.state).unwrap();let phase=Complex::cis(-w*t);for i in 0..ne{e[i]+=phase*responses[x.carrier][i*n0+col]*psi[col];}}
            for &(x,w)in group{let col=cl.position(x.state).unwrap();let phase=Complex::cis(w*t);
                for i in 0..ne{y[col]+=phase*responses[x.carrier][i*n0+col].conj()*e[i];}}
            for(i,&state)in g.excited_states.iter().enumerate(){let q=e[i].norm_sqr()/norm;out[state]+=q;folded+=q;}
        }
        for(x,&b)in cl.states.iter().enumerate(){out[g.ground_states[b]]=(psi[x].norm_sqr()-(psi[x].conj()*y[x]).re)/norm;}
        for(i,&e)in cl.excited.iter().enumerate(){out[g.excited_states[e]]+=psi[n0+i].norm_sqr()/norm;}
        (out,folded)
    }
    /// Per carrier its folded response at a point, R_c = Σ_k E_k T_k (ne × the cluster's n0 ground
    /// states, row-major).
    pub fn responses(&self,cluster:usize,p:&Point)->Vec<Vec<Complex>>{
        let layout=&self.layouts[cluster];let(n0,ne)=(layout.n0,self.ground.excited_states.len());
        layout.carriers.iter().enumerate().map(|(c,cl2)|{
            let mut rc=vec![Complex::ZERO;ne*n0];let mut buf=vec![Complex::ZERO;ne*n0];
            for k in 0..self.carriers[c].rank{self.interpolate(cluster,&p.stencil,None,cl2.t+k*ne*n0,&mut buf);for(o,x)in rc.iter_mut().zip(&buf){*o+=p.fields[c].e[k]**x;}}
            rc}).collect()
    }

    /// The post-jump transient's operators of a cluster at a point and time (engine.rs `transient`):
    /// the folded amplitude of its ground states, X = Σ e^{−iε̃t} A_c|b⟩ per coherent group (label),
    /// split by the groups of nearly degenerate excited states (`Ground::excited_groups`), which ring
    /// apart; per piece and target cluster its photon's S_a = P_B C_a D_g^{−1/2} X_g (rows: the target's
    /// ground states, columns: the cluster's); and Y = Σ X†X, the readout's admixture.
    pub fn transient(&self,cluster:usize,p:&Point,t:f64)->Transient{
        let g=&self.ground;let cl=&g.clusters[cluster];let(n0,ne,ng)=(self.ground_dimension(cluster),g.excited_states.len(),g.ground_states.len());
        let responses=self.responses(cluster,p);
        let mut y=vec![Complex::ZERO;n0*n0];let mut pieces=vec![];
        for(k,group)in cl.coherent.iter().enumerate(){
            // X (ne × n0): the group's channels' folded amplitudes.
            let mut x=vec![Complex::ZERO;ne*n0];
            for &(ch,w)in group{let col=cl.position(ch.state).unwrap();let phase=-Complex::cis(-w*t);for i in 0..ne{x[i*n0+col]+=phase*responses[ch.carrier][i*n0+col];}}
            for a in 0..n0{for b in 0..n0{y[a*n0+b]+=(0..ne).fold(Complex::ZERO,|s,i|s+x[i*n0+a].conj()*x[i*n0+b]);}}
            for(e,levels)in g.excited_groups.iter().enumerate(){
                if levels.iter().all(|&i|(0..n0).all(|b|x[i*n0+b]==Complex::ZERO)){continue;}
                // The piece's photon per axis on every ground state, then per target cluster.
                let s:[Vec<Complex>;3]=std::array::from_fn(|a|{let mut m=vec![Complex::ZERO;ng*n0];
                    for to in 0..ng{for &i in levels{let c=g.emission[a][(to,i)]*(1./g.decay[(i,i)].re.sqrt());if c==Complex::ZERO{continue;}
                        for b in 0..n0{m[to*n0+b]+=c*x[i*n0+b];}}}m});
                let mut targets:Vec<usize>=(0..ng).filter(|&to|(0..3).any(|a|s[a][to*n0..(to+1)*n0].iter().any(|z|*z!=Complex::ZERO))).map(|to|g.cluster[to]).collect();
                targets.sort();targets.dedup();
                for target in targets{let rows=&g.clusters[target].states;
                    let ops=std::array::from_fn(|a|rows.iter().flat_map(|&to|s[a][to*n0..(to+1)*n0].iter().copied()).collect());
                    pieces.push(Piece{coherent:k,excited:e,target,ops,wave_number:self.carriers[group[0].0.carrier].wave_number});}
            }
        }
        Transient{n0,y,pieces}
    }
    /// The photon of a transient's piece from the ground state ψ′ (`transient`), for the kicks of a
    /// moving atom: its amplitude u_a on the target's ground states, the amplitude's gradient
    /// [axis a][lab axis] through the fields and the intensities, and each beam's part of it.
    pub fn transient_photon(&self,cluster:usize,p:&Point,t:f64,piece:&Piece,psi:&[Complex])->([[Complex;MOST_STATES];3],[[[Complex;MOST_STATES];3];3],Vec<(usize,[[Complex;MOST_STATES];3])>){
        let g=&self.ground;let cl=&g.clusters[cluster];let layout=&self.layouts[cluster];let(n0,ne)=(layout.n0,g.excited_states.len());
        let group=&cl.coherent[piece.coherent];let levels=&g.excited_groups[piece.excited];let target=&g.clusters[piece.target];
        // The piece's folded amplitude on its excited states from a carrier's response matrices M_k (rank × ne × n0) weighted by w_k.
        let photon=|amplitudes:&dyn Fn(usize,usize)->Complex|->[[Complex;MOST_STATES];3]{
            let mut u=[[Complex::ZERO;MOST_STATES];3];
            for(row,&to)in target.states.iter().enumerate(){for a in 0..3{for &i in levels{
                let c=g.emission[a][(to,i)]*(1./g.decay[(i,i)].re.sqrt());if c!=Complex::ZERO{u[a][row]+=c*amplitudes(i,0);}}}}
            u};
        // T_k (and ∂_f T_k) at the point, per carrier and rank.
        let table=|c:usize,k:usize,f:Option<usize>|->Vec<Complex>{let mut buf=vec![Complex::ZERO;ne*n0];self.interpolate(cluster,&p.stencil,f,layout.carriers[c].t+k*ne*n0,&mut buf);buf};
        let tables:Vec<Vec<Vec<Complex>>>=(0..self.carriers.len()).map(|c|(0..self.carriers[c].rank).map(|k|table(c,k,None)).collect()).collect();
        // Σ e^{−iε̃t} Σ_k w_k M_k[i, b] ψ_b over the group's channels (the jumps' sign, ground.rs), M the
        // tables or their slopes.
        let folded=|weights:&dyn Fn(usize,usize)->Complex,matrices:&[Vec<Vec<Complex>>],i:usize|->Complex{
            let mut z=Complex::ZERO;
            for &(ch,w)in group{let col=cl.position(ch.state).unwrap();let phase=Complex::cis(-w*t)*psi[col];
                for k in 0..self.carriers[ch.carrier].rank{let wk=weights(ch.carrier,k);if wk!=Complex::ZERO{z+=phase*wk*matrices[ch.carrier][k][i*n0+col];}}}
            z};
        let u=photon(&|i,_|folded(&|c,k|p.fields[c].e[k],&tables,i));
        let mut du=[[[Complex::ZERO;MOST_STATES];3];3];
        let slopes:Vec<Vec<Vec<Vec<Complex>>>>=(0..self.focused.len()).map(|f|(0..self.carriers.len()).map(|c|(0..self.carriers[c].rank).map(|k|table(c,k,Some(f))).collect()).collect()).collect();
        for x in 0..3{
            let field=photon(&|i,_|folded(&|c,k|p.fields[c].grad[k][x],&tables,i));
            for a in 0..3{for r in 0..MOST_STATES{du[a][x][r]+=field[a][r];}}
            for(f,slope)in slopes.iter().enumerate(){
                let part=photon(&|i,_|folded(&|c,k|p.fields[c].e[k]*p.intensity_gradient[f][x],slope,i));
                for a in 0..3{for r in 0..MOST_STATES{du[a][x][r]+=part[a][r];}}
            }
        }
        let beams:Vec<(usize,[[Complex;MOST_STATES];3])>={
            let mut carriers:Vec<usize>=group.iter().map(|x|x.0.carrier).collect();carriers.sort();carriers.dedup();
            carriers.iter().flat_map(|&c|self.carriers[c].beams.iter().enumerate().map(move|(j,&l)|(c,j,l))).map(|(c,j,l)|{
                let amplitude=self.geometry[l].field(p.r).0;let coefficients=self.carriers[c].coefficients[j];
                (l,photon(&|i,_|folded(&|d,k|if d==c{amplitude*coefficients[k]}else{Complex::ZERO},&tables,i)))}).collect()
        };
        (u,du,beams)
    }

    /// A cluster state's potential at r relative to the focus (rad/us): the diagonal of h0 (the units'
    /// intensities there, a standing wave's in it) and of every near beam's own light shift,
    /// |a_l(r)|² Σ c_lk′* c_lk S_k′k -- the near beams' interference (their lattice) left out. A tweezer
    /// folded at second order is in h0; one that is a near carrier (the only light): the trap is the
    /// same.
    pub fn potential(&self,cluster:usize,x:usize,r:[f64;3])->f64{
        self.level(cluster,x,&|u|self.unit_intensity(u,r),&|l|self.geometry[l].intensity(r))-self.level(cluster,x,&|u|self.ground.units[u].focus,&|l|self.geometry[l].intensity([0.;3]))
    }
    /// A cluster state's trap depth (rad/us): the level its potential tends to far from every
    /// focused beam -- the Gaussian beams' amplitudes zero, the plane waves' as at the focus --,
    /// relative to the focus; positive for a state the trap holds.
    pub fn depth(&self,cluster:usize,x:usize)->f64{
        let far=|l:usize|if self.geometry[l].waist.is_some(){Complex::ZERO}else{self.geometry[l].field([0.;3]).0};
        let units=|u:usize|{let unit=&self.ground.units[u];unit.beams.iter().zip(&unit.weights).fold(Complex::ZERO,|s,(&l,&w)|s+far(l)*w).norm_sqr()/unit.norm};
        self.level(cluster,x,&units,&|l|far(l).norm_sqr())-self.level(cluster,x,&|u|self.ground.units[u].focus,&|l|self.geometry[l].intensity([0.;3]))
    }
    /// A cluster state's potential in the trap's light at r (rad/us): `level` with every beam at r less the same with
    /// the trap's light off (`trap_light`), the near light off in both -- the shifts of the trap's beams and of every
    /// far focused beam there, zero where those beams' fields are, −D_T at the focus (ensemble.rs: the initial
    /// ensemble's energies). The near light comes on at t = 0: its fold's shift, whose denominator a focused trap's
    /// shift can close on a shell about the focus (PGCP_10's probe in its all-folded model: 137 k_BT there against the
    /// focus's 34), is no part of what holds the atoms before it. Far from a resonance, with a trap beam alone, the
    /// loss's trap potential (`trap_potential`) without the near light, and the trap's potential relative to its far
    /// level (`potential`, `depth`).
    pub fn trap_level(&self,cluster:usize,x:usize,r:[f64;3])->f64{
        let off=|u:usize|{let unit=&self.ground.units[u];
            unit.beams.iter().zip(&unit.weights).filter(|(l,_)|!self.trap_light.contains(l)).fold(Complex::ZERO,|s,(&l,&w)|s+self.geometry[l].field(r).0*w).norm_sqr()/unit.norm};
        let beams=|l:usize|if self.trap_light.contains(&l){self.geometry[l].intensity(r)}else{0.};
        self.level(cluster,x,&|u|self.unit_intensity(u,r),&beams)-self.level(cluster,x,&off,&|_|0.)
    }
    /// The envelope's modes (ensemble.rs): one beam of each mode -- waist, wave number and axis either way, the same
    /// intensity profile -- of the trap's light in the focused units, the trap beam's first.
    pub fn trap_modes(&self)->Vec<usize>{
        let same=|a:&Geometry,b:&Geometry|{let c:f64=(0..3).map(|i|a.direction[i]*b.direction[i]).sum();a.waist==b.waist&&a.wave_number==b.wave_number&&(c.abs()-1.).abs()<=1e-12};
        let mut modes=vec![self.trap_beam];
        for &u in &self.focused{for &l in &self.ground.units[u].beams{
            if self.trap_light.contains(&l)&&!modes.iter().any(|&m|same(&self.geometry[m],&self.geometry[l])){modes.push(l);}}}
        modes
    }
    /// The trap's largest depth per unit of its light's intensity (`trap_level`), relative to `depth`, its depth at
    /// the focus: a bound of c = sup −V_T/(D_T L) over the intensities of the units of the trap's light in (0, 1], L
    /// their mean weighted by each one's share of the depth at the focus, every other focused unit's in [0, 1]. Every
    /// point's intensities lie in that set and a
    /// unit's intensity lies below the largest of its beams' profiles A_l(r), so −V_T(r) ≤ c D_T L ≤ c D_T max_l A_l(r)
    /// everywhere: the initial ensemble's envelope (ensemble.rs).
    ///
    /// Proven, not sampled: on each cell of the tables' grid in the units' intensities `level` is a polynomial -- `cubic`'s weights in
    /// each unit's intensity, times its s where the trap's own near beams scale with it: quartic in those, cubic in the
    /// others' --, −V_T too, zero where the trap's light is off. A polynomial lies below the largest of its Bernstein
    /// coefficients on the cell and L is linear, so −V_T ≤ R L on the cell wherever each coefficient of −V_T is at most
    /// R times L at its control point: the cell's R, from −V_T at the cell's equally spaced points (to the evaluations'
    /// round-off). A trap with no unit of its own (its shift linear in its intensity) takes L = 1. An error where a
    /// unit of the trap's light holds a beam not of it: its interference with the trap's field falls off as √A, and
    /// the trap's bound states fill an unbounded phase space.
    pub fn trap_slope(&self,cluster:usize,x:usize,depth:f64)->Result<f64,String>{
        let g=&self.ground;let trapped=|l:&usize|self.trap_light.contains(l);
        let traps:Vec<usize>=self.focused.iter().copied().filter(|&u|g.units[u].beams.iter().any(trapped)).collect();let traps=&traps[..];
        if traps.iter().any(|&u|!g.units[u].beams.iter().all(trapped)){return Err("光镊与同频率、同偏振的其他光束相干叠加：其干涉项随光镊强度的平方根衰减，束缚的初始热分布无界".into());}
        let others:Vec<usize>=self.focused.iter().copied().filter(|u|!traps.contains(u)).collect();let others=&others[..];
        let t=traps.len();
        // −V_T D_T at intensities q (the trap's units', then the other units').
        let at=|q:&[f64]|->f64{
            let units=|on:bool|move|u:usize|match traps.iter().position(|&v|v==u){Some(i)=>if on{q[i]}else{0.},None=>others.iter().position(|&v|v==u).map_or(0.,|i|q[t+i])};
            let beams=|on:bool|move|l:usize|if on&&trapped(&l){traps.iter().position(|&u|g.units[u].beams.contains(&l)).map_or(1.,|i|q[i])}else{0.};
            self.level(cluster,x,&units(false),&beams(false))-self.level(cluster,x,&units(true),&beams(true))};
        // L's weights: each unit's share of the depth at the focus (none for a repulsive one).
        let focus:Vec<f64>=traps.iter().chain(others).map(|&u|self.unit_intensity(u,[0.;3]).min(1.)).collect();
        let shares:Vec<f64>=(0..t).map(|i|{let mut q=focus.clone();q[i]=0.;(at(&focus)-at(&q)).max(0.)}).collect();let whole:f64=shares.iter().sum();
        let weights:Vec<f64>=shares.iter().map(|s|if whole>0.{s/whole}else{1./t as f64}).collect();
        // The dimensions: the trap's units' (quartic), the other units' (cubic). Each one's equally spaced points and
        // the inverse of Bernstein's basis there (Gauss–Jordan).
        let degrees:Vec<usize>=(0..t+others.len()).map(|i|if i<t{4}else{3}).collect();
        let inverse=|m:usize|->Vec<Vec<f64>>{
            let binomial=|j:usize|(0..j).fold(1.,|b,i|b*(m-i) as f64/(i+1) as f64);
            let mut a:Vec<Vec<f64>>=(0..=m).map(|i|{let t=i as f64/m as f64;(0..=m).map(|j|binomial(j)*t.powi(j as i32)*(1.-t).powi((m-j) as i32)).chain((0..=m).map(|k|if k==i{1.}else{0.})).collect()}).collect();
            for c in 0..=m{let p=(c..=m).max_by(|&x,&y|a[x][c].abs().total_cmp(&a[y][c].abs())).unwrap();a.swap(c,p);let v=a[c][c];for e in a[c].iter_mut(){*e/=v;}
                for r in 0..=m{if r!=c{let f=a[r][c];let row=a[c].clone();for(e,q)in a[r].iter_mut().zip(row){*e-=f*q;}}}}
            a.into_iter().map(|row|row[m+1..].to_vec()).collect()};
        let inverses:Vec<Vec<Vec<f64>>>=degrees.iter().map(|&m|inverse(m)).collect();
        let n=self.tables.nodes;let h=1./(n-1) as f64;let cells=n-1;let d=degrees.len();
        let sizes:Vec<usize>=degrees.iter().map(|m|m+1).collect();let size:usize=sizes.iter().product();
        let bounds:Vec<f64>=(0..cells.pow(d as u32)).into_par_iter().map(|c|{
            let corner=|i:usize|((c/cells.pow(i as u32))%cells) as f64*h;
            // The cell's points: index p ↦ per dimension j_i of its degree m_i, at corner + h j_i/m_i.
            let digit=|p:usize,i:usize|(p/sizes[..i].iter().product::<usize>())%sizes[i];
            let place=|p:usize,i:usize|corner(i)+h*digit(p,i) as f64/degrees[i] as f64;
            let mut v:Vec<f64>=(0..size).map(|p|{let q:Vec<f64>=(0..d).map(|i|place(p,i)).collect();at(&q)}).collect();
            for i in 0..d{let stride:usize=sizes[..i].iter().product();
                for p in 0..size{if digit(p,i)!=0{continue;}
                    let line:Vec<f64>=(0..sizes[i]).map(|k|v[p+k*stride]).collect();
                    for k in 0..sizes[i]{v[p+k*stride]=inverses[i][k].iter().zip(&line).map(|(a,b)|a*b).sum();}}}
            // The cell's R: the largest coefficient over L at its control point (none where L is 0: the trap's light off,
            // where −V_T is 0); L = 1 throughout without a unit.
            (0..size).filter_map(|p|{let l:f64=if t>0{(0..t).map(|i|weights[i]*place(p,i)).sum()}else{1.};(l>0.).then(||v[p]/l)}).fold(f64::NEG_INFINITY,f64::max)
        }).collect();
        Ok(bounds.into_iter().fold(f64::NEG_INFINITY,f64::max)/depth)
    }
    /// The diagonal of h0 and of every near beam's own light shift for a cluster state, the units at
    /// the intensities `units(u)` and the near beams at `beams(l)` (`potential`): the groups' S at the
    /// state's diagonal entry (none where it is not one of a group's).
    fn level(&self,cluster:usize,x:usize,units:&dyn Fn(usize)->f64,beams:&dyn Fn(usize)->f64)->f64{
        let layout=&self.layouts[cluster];let n=layout.n;
        let s:Vec<f64>=self.focused.iter().map(|&u|units(u).min(1.)).collect();let stencil=self.stencil(&s);
        let mut entry=[Complex::ZERO];self.interpolate(cluster,&stencil,None,layout.h0+x*n+x,&mut entry);let mut u=entry[0].re;
        for gl in layout.groups.iter().filter(|g|g.hermitian){
            let Ok(diagonal)=gl.entries.binary_search(&(x,x)) else{continue};
            let carrier=&self.carriers[gl.absorbed];let mut value=[Complex::ZERO];
            for(j,&l)in carrier.beams.iter().enumerate(){
                let at=beams(l);if at==0.{continue;}let coefficient=&carrier.coefficients[j];
                for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                    self.interpolate(cluster,&stencil,None,gl.s+pi*gl.entries.len()+diagonal,&mut value);
                    u-=at*(coefficient[k1].conj()*coefficient[k]*value[0]).re*if k1<k{2.}else{1.};
                }
            }
        }
        u
    }
}

/// The exact responses of every near beam at a point for a segment beyond the first order's
/// allowance (engine.rs): R_l(w_l) at the point's elimination.
#[derive(Clone,Debug)]
pub struct Exact{pub responses:Vec<Dense>}
impl Exact{
    /// The exact responses at a point and momentum.
    pub fn new(model:&Model,r:[f64;3],kappa:[f64;3])->Result<Exact,String>{
        let g=&model.ground;let e=g.eliminate(&model.intensities_at_point(r))?;
        let mut responses=vec![Dense::zeros(g.excited_states.len(),g.ground_states.len());g.beams.len()];
        for c in &g.carriers{for &l in &c.beams{responses[l]=g.response(&e,l,model.kinetic_shift(l,kappa))?;}}
        Ok(Exact{responses})
    }
}
impl Model{
    /// Every unit's intensity at a point as the tables take it: the focused ones there, the others at
    /// the focus.
    pub fn intensities_at_point(&self,r:[f64;3])->Vec<f64>{
        let s:Vec<f64>=self.focused.iter().map(|&u|self.unit_intensity(u,r).min(1.)).collect();self.unit_intensities(&s)
    }
    /// H at a point: the focused units at their intensities there, the others at the focus.
    pub fn hamiltonian_at_point(&self,r:[f64;3])->Dense{self.ground.hamiltonian_with(&self.intensities_at_point(r))}
}

/// A cluster's post-jump transient operators (`Model::transient`): Y (n0 × n0) and the pieces.
#[derive(Clone,Debug)]
pub struct Transient{pub n0:usize,pub y:Vec<Complex>,pub pieces:Vec<Piece>}
/// A piece of the transient: its coherent group and excited group, its target cluster, its photon's
/// operators per axis (the target's ground states × the cluster's, row-major) and wave number.
#[derive(Clone,Debug)]
pub struct Piece{pub coherent:usize,pub excited:usize,pub target:usize,pub ops:[Vec<Complex>;3],pub wave_number:f64}
/// A photon's absorbed light by its parts (`Photons::absorption`, engine.rs # Absorption): per near beam of its class's
/// carriers its folded share and its amplitude's projection on the photon, and its explicit states' part -- its share and
/// the beams' weights by their drives into the explicit states.
#[derive(Clone,Debug,Default)]
pub struct Absorption{pub beams:Vec<(usize,f64,f64)>,pub explicit:Option<(f64,Vec<(usize,f64)>)>}
/// The photon amplitudes of every class of a cluster's jumps at a point: per class and axis the
/// target cluster's amplitude u_a; for a class drawn, its gradient and its absorbed light's parts
/// (`Photons::absorption`) from the jump tables at the point, kept.
pub struct Photons<'a>{
    model:&'a Model,cluster:usize,point:Point,psi:[Complex;MOST_MIXED],time:f64,
    /// Per carrier, rank and axis (carrier-major): J_k and J2_k at the point, ng × n0.
    jumps:Vec<Complex>,doppler:Vec<Complex>,
    exact:Option<Exact>,
    pub amplitude:Vec<[[Complex;MOST_STATES];3]>,
}
impl Photons<'_>{
    /// Where carrier c's rank k and axis a's tables start in `jumps`.
    fn offset(&self,c:usize,k:usize,a:usize)->usize{
        let m=self.model;let size=m.ground.ground_states.len()*m.ground_dimension(self.cluster);
        (m.carriers[..c].iter().map(|x|x.rank).sum::<usize>()*3+k*3+a)*size
    }
    /// A class's absorbed light along a photon's polarization e (`Engine::jump`), in one pass over its
    /// photons: per carrier of the class and target row, A_k = Σ e^{iω̃t} ψ[from] Σ_a e_a J_k[a][to, from],
    /// B_k the same of J2_k and ∂_f A_k the slopes of A_k in each s_f (the axes' entries combined at
    /// each node of the stencil). Of them, per row the gradient of the class's amplitude along e,
    /// Σ_k ∇E_k A_k + E_k Σ_f ∂_f A_k ∇s_f (the Doppler term's own gradient left out); and the parts
    /// (`Absorption`) of the photon f along e: per beam l of the class's carriers its share ‖f_l‖² and projection
    /// Re⟨f|f_l⟩/⟨f|f⟩, f_l = a_l Σ_k c_lk (A_k − w_l B_k) per row -- beyond the first order its exact response's a_l Σ
    /// e^{iω̃t} ψ[from] Σ_a e_a (C_a R_l)[to, from] --, and the explicit states' part.
    pub fn absorption(&self,class:usize,e:[f64;3],f:&[Complex;MOST_STATES])->([[Complex;3];MOST_STATES],Absorption){
        let m=self.model;let g=&m.ground;let cl=&g.clusters[self.cluster];let class=&cl.classes[class];let target=&g.clusters[class.target];
        let(n0,nt,p,d)=(m.ground_dimension(self.cluster),target.states.len(),&self.point,m.focused.len());
        let(layout,size)=(&m.layouts[self.cluster],g.ground_states.len()*n0);let(data,slice)=(&m.tables.data[self.cluster],2*layout.size);
        let carriers:Vec<usize>={let mut v:Vec<usize>=class.photons.iter().map(|x|x.0.carrier).collect();v.sort();v.dedup();v};
        // Per carrier of the class and row, per rank: A, B and ∂_f A.
        let mut sums=vec![[[Complex::ZERO;2+MOST_FOCUSED];MOST_RANK];carriers.len()*MOST_STATES];let mut rotation=Rotation::new(self.time);
        for &(ph,w)in &class.photons{
            let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());let phase=rotation.at(w)*self.psi[col];
            let sum=&mut sums[carriers.binary_search(&ph.carrier).unwrap()*MOST_STATES+row];
            for k in 0..m.carriers[ph.carrier].rank{
                let(mut a,mut b)=(Complex::ZERO,Complex::ZERO);let mut slope=[Complex::ZERO;MOST_FOCUSED];
                for x in 0..3{let at=self.offset(ph.carrier,k,x)+ph.to*n0+col;a+=self.jumps[at]*e[x];b+=self.doppler[at]*e[x];}
                if d>0{
                    let starts:[usize;3]=std::array::from_fn(|x|2*(layout.carriers[ph.carrier].j+(k*3+x)*size+ph.to*n0+col));
                    for i in 0..p.stencil.count(){
                        let(node,_,dw)=p.stencil.node(i);let base=node*slice;
                        let v=(0..3).fold(Complex::ZERO,|v,x|v+Complex::new(data[base+starts[x]],data[base+starts[x]+1])*e[x]);
                        for f in 0..d{slope[f]+=v*dw[f];}
                    }
                }
                sum[k][0]+=phase*a;sum[k][1]+=phase*b;for f in 0..d{sum[k][2+f]+=phase*slope[f];}
            }
        }
        let mut gradient=[[Complex::ZERO;3];MOST_STATES];
        for(ci,&c)in carriers.iter().enumerate(){
            let fl=&p.fields[c];
            for(row,out)in gradient[..nt].iter_mut().enumerate(){
                let sum=&sums[ci*MOST_STATES+row];
                for x in 0..3{for k in 0..m.carriers[c].rank{
                    out[x]+=fl.grad[k][x]*sum[k][0];for f in 0..d{out[x]+=fl.e[k]*sum[k][2+f]*p.intensity_gradient[f][x];}
                }}
            }
        }
        let size:f64=f[..nt].iter().map(|z|z.norm_sqr()).sum();let on_f=|row:usize,z:Complex|if size>0.{(f[row].conj()*z).re/size}else{0.};
        let beams:Vec<(usize,f64,f64)>=carriers.iter().enumerate().flat_map(|(ci,&c)|m.carriers[c].beams.iter().enumerate().map(move|(j,&l)|(ci,c,j,l))).map(|(ci,c,j,l)|{
            let(amplitude,w)=(m.geometry[l].field(p.r).0,m.kinetic_shift(l,p.kappa));
            let rows:[Complex;MOST_STATES]=match &self.exact{
                Some(exact)=>{
                    let product:[Dense;3]=std::array::from_fn(|a|g.emission[a].product(&exact.responses[l]));let mut rows=[Complex::ZERO;MOST_STATES];let mut rotation=Rotation::new(self.time);
                    for &(ph,w)in class.photons.iter().filter(|x|x.0.carrier==c){
                        let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());
                        let z=(0..3).fold(Complex::ZERO,|z,a|z+product[a][(ph.to,cl.states[col])]*e[a]);rows[row]+=rotation.at(w)*self.psi[col]*amplitude*z;
                    }
                    rows
                }
                None=>std::array::from_fn(|row|{let sum=&sums[ci*MOST_STATES+row];
                    (0..m.carriers[c].rank).fold(Complex::ZERO,|z,k|z+m.carriers[c].coefficients[j][k]*(sum[k][0]-sum[k][1]*w))*amplitude}),
            };
            (l,rows[..nt].iter().map(|z|z.norm_sqr()).sum(),(0..nt).map(|row|on_f(row,rows[row])).sum())}).collect();
        // The explicit states' part of the photon: its weight along e, and the beams' weights: each explicit state's weight
        // in the photon shared by the drives into it, Σ |a_l Σ_k c_lk U_k[e,b] ψ_b|² -- a photon of that state.
        let explicit=(!class.direct.is_empty()).then(||{
            let dimension=m.dimension(self.cluster);let mut parts=vec![[Complex::ZERO;MOST_STATES];dimension];
            for &(i,to,w)in &class.direct{let row=target.position(to).unwrap();let phase=Complex::cis(w*self.time)*self.psi[i];
                for a in 0..3{parts[i][row]+=phase*m.photon(self.cluster,&p.stencil,i-n0,a,to)*e[a];}}
            let total:f64=(0..nt).map(|row|parts.iter().fold(Complex::ZERO,|s,v|s+v[row]).norm_sqr()).sum();
            let drive=|dr:&DriveLayout,j:usize,l:usize|{let carrier=&m.carriers[dr.carrier];((0..carrier.rank).fold(Complex::ZERO,|s,k|s+carrier.coefficients[j][k]*dr.u[k])*m.geometry[l].field(p.r).0*self.psi[dr.col]).norm_sqr()};
            let mut into=vec![0.;dimension];
            for dr in &m.layouts[self.cluster].drives{for(j,&l)in m.carriers[dr.carrier].beams.iter().enumerate(){into[dr.row]+=drive(dr,j,l);}}
            let mut drives:Vec<(usize,f64)>=vec![];
            for dr in &m.layouts[self.cluster].drives{
                let weight:f64=parts[dr.row][..nt].iter().map(|z|z.norm_sqr()).sum();
                for(j,&l)in m.carriers[dr.carrier].beams.iter().enumerate(){
                    let d=if into[dr.row]>0.{weight*drive(dr,j,l)/into[dr.row]}else{0.};
                    match drives.iter_mut().find(|x|x.0==l){Some(x)=>x.1+=d,None=>drives.push((l,d))}
                }
            }
            (total,drives)});
        (gradient,Absorption{beams,explicit})
    }
}

#[cfg(test)]
mod tests{
    use super::{cubic,Beats,Model,MOST_TABLE_BYTES,MOST_TABLE_NODES,NOT_LINDBLAD};
    use std::f64::consts::TAU;

    /// The interpolant (`cubic`) is C¹ -- its derivative, the force the tables give, the same from either side of every
    /// node, the grid's ends included --, reproduces a quadratic exactly, and converges at third order: a smooth
    /// function's largest error falls about eightfold per doubling of the grid.
    #[test]
    fn the_interpolant_is_c1_and_third_order(){
        let at=|n:usize,s:f64,f:&dyn Fn(f64)->f64|{let(first,w,d)=cubic(n,s);let h=1./(n-1) as f64;
            (0..4).fold((0.,0.),|(v,g),k|{let y=f((first+k) as f64*h);(v+w[k]*y,g+d[k]*y)})};
        let quadratic=|x:f64|3.*x*x-2.*x+0.5;
        for n in [4,5,9,17]{let h=1./(n-1) as f64;
            for i in 0..=200{let s=i as f64/200.;let(v,g)=at(n,s,&quadratic);assert!((v-quadratic(s)).abs()<1e-12&&(g-(6.*s-2.)).abs()<1e-10,"{n} {s}: {v} {g}");}
            for node in 1..n-1{let s=node as f64*h;let f=|x:f64|(7.*x).sin()+x*x*x;
                let(left,right)=(at(n,s-1e-12,&f).1,at(n,s+1e-12,&f).1);assert!((left-right).abs()<1e-8*(1.+left.abs()),"{n} node {node}: {left} {right}");}
        }
        let worst=|n:usize|(0..=1000).map(|i|{let s=i as f64/1000.;(at(n,s,&|x:f64|(5.*x).sin()).0-(5.*s).sin()).abs()}).fold(0.,f64::max);
        let(a,b,c)=(worst(33),worst(65),worst(129));assert!(a/b>6.&&b/c>6.&&a/b<10.&&b/c<10.,"{a} {b} {c}");
    }

    /// The tables stop refining at the work bounds (F9): the tweezer PGC's (its tweezer one intensity
    /// axis) within nine nodes, or within the bytes of the grids of nine and five, keep the grid of
    /// nine, its error measured against the five before it and booked -- where the unbounded tables
    /// refine on, the next grid paying for itself, to a smaller one --; a bound below the first
    /// refinement leaves no measured error, and is refused.
    #[test]
    fn the_tables_stop_at_the_work_bounds_with_their_error_measured(){
        let f:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/tweezer_pgc.json")).unwrap();
        let r=crate::ground::tests::request(&f["request"]);
        let free=Model::new(&r).unwrap();
        assert!(free.table_limit.is_none()&&!free.table_stalled&&free.tables.nodes>9,"{:?} at {} nodes: {:e}",free.table_limit,free.tables.nodes,free.table_error);
        let entries:usize=free.layouts.iter().map(|l|l.size).sum();
        for(bytes,nodes)in [(MOST_TABLE_BYTES,9),(16*entries*(9+5),MOST_TABLE_NODES)]{
            let m=Model::bounded(&r,f64::INFINITY,false,bytes,nodes).unwrap();
            assert!(m.table_limit==Some(9)&&m.tables.nodes==9&&m.table_error>free.table_error,"{:?} at {} nodes: {:e}",m.table_limit,m.tables.nodes,m.table_error);
        }
        assert!(Model::bounded(&r,f64::INFINITY,false,MOST_TABLE_BYTES,5).is_err());
    }
    /// A generator whose no-jump decay misses its jumps' rate beyond round-off is refused (`Model::lindblad`): the
    /// tweezer PGC's model, a Lindblad one to round-off, with one kept group's decay Q a millionth smaller at every
    /// node of its tables -- a decay no jump carries, a millionth of that group's rate.
    #[test]
    fn a_generator_whose_decay_misses_its_jumps_is_refused(){
        let f:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/tweezer_pgc.json")).unwrap();
        let r=crate::ground::tests::request(&f["request"]);
        let mut model=Model::new(&r).unwrap();assert!(model.lindblad().is_ok());
        let(k,q,square)=model.layouts.iter().enumerate().find_map(|(k,l)|l.groups.iter().find(|g|!g.decay).map(|g|(k,g.q,g.pairs.len()*g.entries.len()))).unwrap();
        let size=model.layouts[k].size;
        for node in 0..model.tables.data[k].len()/(2*size){for i in q..q+square{for c in 0..2{model.tables.data[k][2*(node*size+i)+c]*=1.-1e-6;}}}
        let e=model.lindblad().expect_err("a gain of a millionth of a rate");assert!(e.starts_with(NOT_LINDBLAD),"{e}");
    }
    /// Residuals within the round-off of zero are static and of each other one beat; the beats come in
    /// pairs ±Ω; harmonics of one frequency have its period, incommensurate ones none.
    #[test]
    fn the_beats_merge_the_round_off_and_find_the_period(){
        let w=228.702;let round_off=1e-10;
        let b=Beats::new([0.,-5.7e-14,w,w*(1.+1e-15),-w,2.*w,-2.*w+3e-13,5.*w,0.].into_iter(),round_off);
        assert_eq!(b.omegas.len(),6);
        for(x,y)in b.omegas.iter().zip([-5.*w,-2.*w,-w,w,2.*w,5.*w]){assert!((x-y).abs()<=round_off,"{:?}",b.omegas);}
        assert!((b.period.unwrap()-TAU/w).abs()<=1e-12*TAU/w,"{:?}",b.period);
        assert_eq!(b.of(1e-13),None);assert_eq!(b.of(-2.*w),Some(1));assert_eq!(b.of(5.*w*(1.-1e-16)),Some(5));
        // Incommensurate beats are harmonics only of a fundamental below their ratio's resolution in the
        // round-off: a period far too long for a step's cache (engine.rs).
        assert!(Beats::new([1.,2f64.sqrt()].into_iter(),1e-12).period.map_or(true,|t|TAU/t<1e-4));
        assert!((Beats::new([3.,4.5,-7.5].into_iter(),1e-12).period.unwrap()-TAU/1.5).abs()<=1e-12);
        assert!(Beats::new([0.,1e-11].into_iter(),1e-10).omegas.is_empty());
    }
}
