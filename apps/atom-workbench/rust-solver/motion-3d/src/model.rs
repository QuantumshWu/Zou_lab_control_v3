//! The 3D model: the eliminated generator of `ground` on each cluster at any point r and momentum
//! ħκ, from tables over the focused beams' intensities and the near carriers' fields at r.
//!
//! # Where the generator depends on r
//! H(r) = H0 + Σ_l (|a_l(r)|² − 1) O_l + Σ_l |a_l(r)|² F_l (ground.rs) depends on r only through
//! the beams' intensities |a_l(r)|², and so does the elimination: the energies, K and every near
//! beam's responses. The near beams enter through their amplitudes a_l(r) (phase and profile, a
//! Gaussian beam's exact), linearly per carrier. The tables hold the elimination on a uniform grid
//! of the focused beams' intensities s ∈ [0, 1]^D, cubic in each s_f, the grid doubled until the
//! interpolation at the new nodes agrees with the exact elimination to the tables' share of the
//! budget (`TABLE_SHARE`) of every table's scale. The focused beams are the trap beam and every
//! Gaussian beam whose profile matters (`Model::new`): one whose intensity changes over the atom's
//! reach by more than the budget leaves -- its shifts O_l + F_l against the excited poles -- gets an
//! axis; the others' shifts are taken at their peak.
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
//! is −∇Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩, with the kept drives' gradients Σ_k ∇E_k U_k the stimulated force.
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
//! # The folded pairs' van Vleck shift
//! A folded pair's path (c, b) → (d, a) → (f, b̂) shifts H_eff by W̃ D U, U = −[V_d† G V_c]_ab and
//! W̃ = −[V_f† G V_d]_b̂a at the cluster's path energy (ground.rs). Through a carrier d of one beam l,
//! E^d E^d† = |a_l|² c_l c_l† is its intensity (1, or s_l on its axis): the shift is quadratic in
//! E^f* E^c, Σ_qk E^f_q* E^c_k |a_l|² D W̄_q Ū_k with Ū_k = −V_l[:,a]† G U^c_k[:,b] and
//! W̄_q = −U^f_q[:,b̂]† G V_l[:,a], and the tables of the pair (c, f) take it -- the Hermitian
//! group's split into its Hermitian and anti-Hermitian halves. Through a carrier of several beams it
//! is quartic in their interfering fields; it is left out with its bound in the ledger
//! (`Model::quartic`), as a fold at first order.
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
//! The trap beam's focus is the origin. Its trap is each state's potential with the beams'
//! interference left out (`Model::potential`); its Hessian at the focus gives the trap's principal
//! axes and frequencies (`Trap`). An atom is held while its mechanical energy, ħ²κ²/2m plus its
//! potential relative to far from the trap beam, is negative: the potential the engine's force
//! derives from, Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩ of its current internal state, less the same with the trap beam
//! off (`Local::trap`).
use crate::ground::{self,Ground};
use crate::ledger::BUDGET;
use crate::linear::{Dense,Lu};
use crate::stats::KT_PER_UK;
use crate::{Complex,Request,STATES};
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;
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
/// The tables' interpolation error at the refined nodes, relative to each table's scale: a numerical
/// discretization, held to the a-priori share of the budget.
pub const TABLE_SHARE:f64=A_PRIORI_SHARE*BUDGET;
/// The coarsest grid of each intensity axis (nodes), the fewest a cubic stencil takes doubled once.
const FIRST_NODES:usize=5;

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
/// its field pairs (k′ ≤ k for a Hermitian one) and where its S and Q lie (n × n each per pair).
#[derive(Clone,Debug)]
pub(crate) struct GroupLayout{pub(crate) absorbed:usize,pub(crate) emitted:usize,pub(crate) residual:f64,pub(crate) hermitian:bool,pub(crate) pairs:Vec<(usize,usize)>,pub(crate) s:usize,pub(crate) q:usize}
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
    decays:Vec<DecayLayout>,effective:Vec<EffectiveLayout>,emission:usize,beats:Beats}
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
/// Hermitian part (a phase), and the gradient of its Hermitian part per axis, n × n, re/im
/// interleaved.
#[derive(Clone,Copy,Debug)]
pub struct Local{pub hamiltonian:[Complex;MOST_STATES*MOST_STATES],pub scalar:f64,pub gradient:[[f64;MOST_W];3]}
/// A cluster's generator at a point and a time on any number of states, row-major n × n: H_eff
/// less `scalar` (the scalar part of its Hermitian part) and the gradient of its Hermitian part per
/// axis -- the explicit excited states' clusters', and any cluster's in the exact step (engine.rs).
#[derive(Clone,Debug)]
pub struct Generator{pub n:usize,pub hamiltonian:Vec<Complex>,pub scalar:f64,pub gradient:[Vec<Complex>;3]}
impl Generator{
    /// A cluster of ground states' generator (`Local`) on its n states, its gradients' Hermitian parts
    /// (the force reads Re⟨ψ|∇H_eff|ψ⟩ alone).
    pub fn of(local:&Local,n:usize)->Generator{
        let entry=|a:usize,k:usize|Complex::new(local.gradient[a][2*k],local.gradient[a][2*k+1]);
        Generator{n,hamiltonian:local.hamiltonian[..n*n].to_vec(),scalar:local.scalar,
            gradient:std::array::from_fn(|a|(0..n*n).map(|k|(entry(a,k)+entry(a,(k%n)*n+k/n).conj())*0.5).collect())}
    }
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

/// One node of the tables: every cluster's tables there, the quartic shifts' bound and the
/// elimination's (`Model::resonance`).
type Node=(Vec<Vec<Complex>>,f64,f64);

/// The tables on one grid, and the elimination's bound over its nodes inside the reach (`Model::resonance`).
#[derive(Clone,Debug)]
pub(crate) struct Tables{pub(crate) nodes:usize,pub(crate) data:Vec<Vec<f64>>,resonance:f64}

/// The trap of the trap beam: its principal axes (unit rows on the lab axes) and frequencies
/// (rad/us) at the focus for a reference state, and the frequencies per lab axis.
#[derive(Clone,Copy,Debug)]
pub struct Trap{pub axes:[[f64;3];3],pub frequencies:[f64;3],pub lab:[f64;3]}

/// The 3D model of a request.
#[derive(Clone,Debug)]
pub struct Model{
    pub ground:Ground,
    pub geometry:Vec<Geometry>,
    /// The trap beam, and the focused beams (indices into the beams), one intensity axis each, the
    /// trap beam's first.
    pub trap_beam:usize,
    pub focused:Vec<usize>,
    /// Each Gaussian beam taken at its peak: the beam, its estimated error over the reach and its
    /// shifts' weight, the error per unit of intensity lost (`choose_profiles`).
    pub profiles:Vec<(usize,f64,f64)>,
    pub carriers:Vec<Carrier>,
    /// ħ/m (um²/us).
    pub hbar_over_mass:f64,
    pub mass_kg:f64,
    pub(crate) layouts:Vec<Layout>,
    pub(crate) tables:Tables,
    /// The nodes per intensity axis and the interpolation's measured largest relative error.
    pub table_error:f64,
    /// The quartic van Vleck shifts left out: their bound relative to the eliminated generator's scale.
    pub quartic:f64,
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
    /// The kinetic shift of absorption (`kinetic_shift`), always on in a solve; off only where a
    /// validation compares with the recoil-heating formula, which has no Doppler friction.
    pub doppler:bool,
}

fn frobenius_inner(a:&Dense,b:&Dense)->Complex{a.entries.iter().zip(&b.entries).fold(Complex::ZERO,|s,(x,y)|s+x.conj()**y)}

/// Cubic Lagrange weights on four consecutive nodes of a uniform grid of `nodes` points on [0, 1],
/// and their derivatives in s: the stencil's first node and the weights.
fn cubic(nodes:usize,s:f64)->(usize,[f64;4],[f64;4]){
    let h=1./(nodes-1) as f64;let x=s.clamp(0.,1.)/h;
    let first=((x.floor() as isize)-1).clamp(0,nodes as isize-4) as usize;let t=x-first as f64;
    let w=[-(t-1.)*(t-2.)*(t-3.)/6.,t*(t-2.)*(t-3.)/2.,-t*(t-1.)*(t-3.)/2.,t*(t-1.)*(t-2.)/6.];
    let d=[-((t-2.)*(t-3.)+(t-1.)*(t-3.)+(t-1.)*(t-2.))/6.,((t-2.)*(t-3.)+t*(t-3.)+t*(t-2.))/2.,
        -((t-1.)*(t-3.)+t*(t-3.)+t*(t-1.))/2.,((t-1.)*(t-2.)+t*(t-2.)+t*(t-1.))/6.];
    (first,w,d.map(|x|x/h))
}

/// The pairs k′ ≤ k of a Hermitian group of rank r, or every (k′, k) of ranks (r′, r).
fn field_pairs(hermitian:bool,emitted:usize,absorbed:usize)->Vec<(usize,usize)>{
    if hermitian{(0..absorbed).flat_map(|k1|(k1..absorbed).map(move|k|(k1,k))).collect()}else{(0..emitted).flat_map(|k1|(0..absorbed).map(move|k|(k1,k))).collect()}
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
    pub fn with(request:&Request,allowance:f64)->Result<Model,String>{
        let ground=Ground::with(request,allowance)?;
        let geometry:Vec<Geometry>=ground.beams.iter().map(|b|Geometry{direction:b.direction,wave_number:b.wave_number_per_um,waist:b.waist_um}).collect();
        let trap_beam=request.trap_beam;
        if trap_beam>=geometry.len()||geometry[trap_beam].waist.is_none(){return Err("光镊（trapBeamId 所指的光束）需要束腰：3D 运动计算聚焦在原子处的光镊中的单个原子".into());}
        let hbar_over_mass=HBAR/request.mass_kg*1e6;
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
        let mut model=Model{ground,geometry,trap_beam,focused:vec![trap_beam],profiles:vec![],carriers,hbar_over_mass,mass_kg:request.mass_kg,
            layouts:vec![],tables:Tables{nodes:1,data:vec![],resonance:0.},table_error:0.,quartic:0.,resonance:0.,reach:vec![[0.;3]],
            wide:std::arch::is_x86_feature_detected!("avx2")&&std::arch::is_x86_feature_detected!("fma"),doppler:true};
        model.choose_profiles(request)?;
        model.layouts=model.layouts(&bases);
        model.tabulate(&bases)?;
        Ok(model)
    }

    /// The Gaussian beams other than the trap beam: each one's profile is judged over the atom's
    /// reach, the region of the trap within ln(1/budget) k_B T of its bottom (the harmonic trap's
    /// thermal ellipsoid holding all but the budget of a Boltzmann ensemble), T the request's: its
    /// intensity falls there to 1 − ε at worst, and its shifts O_l + F_l, taken at the peak, err
    /// by ε ‖O_l + F_l‖ against the nearest excited pole and the generator's scale. Within the budget
    /// the beam keeps its exact profile in its field (a near beam) and its shifts at the peak; beyond,
    /// an intensity axis (at most `MOST_FOCUSED` in all, the largest errors first).
    fn choose_profiles(&mut self,request:&Request)->Result<(),String>{
        let g=&self.ground;let reference=self.reference_state(request)?;
        let trap=self.bare_trap(reference);
        let reach:[f64;3]=std::array::from_fn(|a|{let w=trap.frequencies[a];
            if w>0.{(2.*BUDGET.recip().ln()*KT_PER_UK*request.temperature_uk*self.hbar_over_mass).sqrt()/w}else{0.}});
        for a in 0..3{for sign in [-1.,1.]{self.reach.push(trap.axes[a].map(|x|x*sign*reach[a]));}}
        let mut errors:Vec<(usize,f64,f64)>=vec![];
        for l in 0..self.geometry.len(){
            if l==self.trap_beam||self.geometry[l].waist.is_none(){continue;}
            let fall=(0..3).flat_map(|a|[-1.,1.].map(move|s|(a,s))).map(|(a,s)|{let r=trap.axes[a].map(|x|x*s*reach[a]);1.-self.geometry[l].intensity(r)}).fold(0_f64,f64::max).max(0.);
            let mut shift=g.outside[l].clone();shift.add_scaled(Complex::ONE,&g.folds[l]);
            let size=shift.spectral_norm();
            let weight=size*(1./g.pole.max(f64::MIN_POSITIVE)).max(1./g.generator_scale.max(f64::MIN_POSITIVE));
            errors.push((l,fall*weight,weight));
        }
        errors.sort_by(|a,b|b.1.total_cmp(&a.1));
        for(l,error,weight)in errors{
            if error>BUDGET&&self.focused.len()<MOST_FOCUSED{self.focused.push(l);}else{self.profiles.push((l,error,weight));}
        }
        Ok(())
    }
    /// The profile error at r: over the Gaussian beams taken at their peak, the largest intensity
    /// lost there times its weight.
    pub fn profile_error(&self,r:[f64;3])->f64{
        self.profiles.iter().map(|&(l,_,weight)|(1.-self.geometry[l].intensity(r)).max(0.)*weight).fold(0.,f64::max)
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
        Trap{axes,frequencies:values.map(w),lab:std::array::from_fn(|a|w(hessian[a][a]))}
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
        g.clusters.iter().map(|cl|{
            let(n,n0)=(cl.dimension(),cl.states.len());let mut size=n*n;let h0=0;let mut groups=vec![];
            for gr in &cl.groups{
                let mut residuals:Vec<f64>=gr.entries.iter().map(|e|e.2).collect();residuals.sort_by(f64::total_cmp);residuals.dedup();
                for residual in residuals{
                    let hermitian=gr.absorbed==gr.emitted&&residual==0.;
                    let pairs=field_pairs(hermitian,self.carriers[gr.emitted].rank,self.carriers[gr.absorbed].rank);
                    let square=pairs.len()*n*n;groups.push(GroupLayout{absorbed:gr.absorbed,emitted:gr.emitted,residual,hermitian,pairs,s:size,q:size+square});size+=2*square;
                }
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
                    let delta=g.carriers[c].detuning;let w=cl.excited_frames[p]-cl.frames[x]-delta;let residual=if crate::ground::same_frame(cl.excited_frames[p],cl.frames[x]+delta){0.}else{w};
                    effective.push(EffectiveLayout{row:n0+p,col:x,carrier:c,residual,at:size});size+=carrier.rank;
                }}
            }
            let emission=size;size+=3*ng*cl.excited.len();
            let drives:Vec<DriveLayout>=cl.drives.iter().map(|&(row,col,c,residual)|DriveLayout{row,col,carrier:c,residual,
                u:std::array::from_fn(|k|if k<self.carriers[c].rank{basis[c][k][(cl.excited[row-n0],cl.states[col])]}else{Complex::ZERO})}).collect();
            // The beats of every term's residual, a residual static within `same_frame`'s tolerance at the
            // cluster's largest frame and detuning (the round-off of the frames' differences).
            let largest=cl.frames.iter().chain(&cl.excited_frames).map(|x|x.abs()).fold(0.,f64::max)+g.carriers.iter().map(|c|c.detuning.abs()).fold(0.,f64::max);
            let residuals=groups.iter().map(|x|x.residual).chain(drives.iter().map(|x|x.residual)).chain(crosses.iter().map(|x|x.residual))
                .chain(decays.iter().map(|x|x.residual)).chain(effective.iter().map(|x|x.residual));
            let beats=Beats::new(residuals,64.*f64::EPSILON*2.*largest);
            Layout{n,n0,size,h0,groups,carriers,crosses,drives,decays,effective,emission,beats}
        }).collect()
    }

    /// H at the focused beams' intensities s, the others at their peak.
    pub fn hamiltonian_at(&self,s:&[f64])->Dense{
        let mut intensities=vec![1.;self.geometry.len()];for(i,&l)in self.focused.iter().enumerate(){intensities[l]=s[i];}
        self.ground.hamiltonian_with(&intensities)
    }

    /// Every cluster's tables at the intensities s, and the quartic shifts' bound there.
    fn node(&self,s:&[f64],mixing:&[Vec<Vec<Complex>>])->Result<Node,String>{
        let g=&self.ground;let e=g.eliminate(&self.hamiltonian_at(s))?;let(ne,ng)=(g.excited_states.len(),g.ground_states.len());
        // Per carrier: U^f_k (the folded basis), T_k, T2_k (ne × ng).
        let combine=|c:usize,matrices:&[Dense]|->Vec<Dense>{mixing[c].iter().map(|g_k|{let mut m=Dense::zeros(ne,ng);
            for(j,&l)in self.carriers[c].beams.iter().enumerate(){m.add_scaled(g_k[j],&matrices[l]);}m}).collect()};
        let u:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&g.folded)).collect();
        let t:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&e.responses)).collect();
        let t2:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&e.derivatives)).collect();
        let jump=|m:&Dense|->[Dense;3]{std::array::from_fn(|a|g.emission[a].product(m))};
        let(j,j2):(Vec<Vec<[Dense;3]>>,Vec<Vec<[Dense;3]>>)=(t.iter().map(|ts|ts.iter().map(jump).collect()).collect(),t2.iter().map(|ts|ts.iter().map(jump).collect()).collect());
        let mut quartic=0_f64;
        let mut out=vec![];
        for(k,(cl,layout))in g.clusters.iter().zip(&self.layouts).enumerate(){
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
            // The kept groups: S and Q on their entries.
            for gl in &layout.groups{
                let(c,d)=(gl.absorbed,gl.emitted);
                let entries:Vec<(usize,usize)>=cl.groups.iter().find(|x|x.absorbed==c&&x.emitted==d).unwrap().entries.iter().filter(|x|x.2==gl.residual).map(|x|(x.0,x.1)).collect();
                for(p,&(k1,k0))in gl.pairs.iter().enumerate(){
                    let base=p*n*n;
                    for &(x,y)in &entries{
                        if x>=n0{
                            // Two explicit states: their folded drives' van Vleck shift (c = d, no Q).
                            let(ea,eb,delta)=(cl.excited[x-n0],cl.excited[y-n0],g.carriers[c].detuning);let mut sv=Complex::ZERO;
                            for b in 0..ng{sv-=u[c][k0][(ea,b)]*u[c][k1][(eb,b)].conj()*(0.5*(1./(e.excited_energies[ea]-delta-e.energies[b])+1./(e.excited_energies[eb]-delta-e.energies[b])));}
                            node[gl.s+base+x*n+y]=sv;continue;
                        }
                        let(a,b)=(states[x],states[y]);let(mut sv,mut qv)=(Complex::ZERO,Complex::ZERO);
                        for i in 0..ne{sv+=(u[d][k1][(i,a)].conj()*t[c][k0][(i,b)]+t[d][k1][(i,a)].conj()*u[c][k0][(i,b)])*0.5;}
                        let mut dt=vec![Complex::ZERO;ne];
                        for i in 0..ne{for i2 in 0..ne{dt[i]+=g.decay[(i,i2)]*t[c][k0][(i2,b)];}}
                        for i in 0..ne{qv+=t[d][k1][(i,a)].conj()*dt[i];}
                        node[gl.s+base+x*n+y]=sv;node[gl.q+base+x*n+y]=qv;
                    }
                }
            }
            // The folded pairs' van Vleck shift: through a carrier of one beam into the tables of the
            // pair's group; through several, its bound.
            if !cl.paths.is_empty(){
                let mut resolvents:Vec<Option<(Lu,f64)>>=vec![None;self.carriers.len()];let mut omitted=vec![0_f64;n*n];
                for p in &cl.paths{
                    let(c,d,f)=(p.from.carrier,p.via.carrier,p.back.carrier);
                    if resolvents[c].is_none(){let z=g.path_energy(&e,k,c);resolvents[c]=Some((ground::denominator(&e.excited_hamiltonian,-z)?,z));}
                    let(lu,z)=resolvents[c].as_ref().unwrap();let z=*z;
                    let denominator=1./(z-e.energies[p.via.state]-g.carriers[d].detuning);
                    let(x,y)=(cl.position(p.back.state).unwrap(),cl.position(p.from.state).unwrap());
                    let solve=|m:&Dense,col:usize|->Vec<Complex>{let mut v:Vec<Complex>=(0..ne).map(|i|m[(i,col)]).collect();lu.solve(&mut v);v};
                    let dot=|m:&Dense,col:usize,v:&[Complex]|->Complex{(0..ne).fold(Complex::ZERO,|s,i|s+m[(i,col)].conj()*v[i])};
                    if self.carriers[d].beams.len()==1{
                        let l=self.carriers[d].beams[0];
                        let intensity=self.focused.iter().position(|&x|x==l).map_or(1.,|i|s[i]);
                        let gv=solve(&g.folded[l],p.via.state);
                        let ubar:Vec<Complex>=(0..self.carriers[c].rank).map(|k0|-dot(&g.folded[l],p.via.state,&solve(&u[c][k0],p.from.state))).collect();
                        let wbar:Vec<Complex>=(0..self.carriers[f].rank).map(|q|-dot(&u[f][q],p.back.state,&gv)).collect();
                        // M_qk = |a_l|² D W̄_q Ū_k, into the group (c → f) at the path's residual.
                        let gl=layout.groups.iter().find(|x|x.absorbed==c&&x.emitted==f&&x.residual==p.residual).ok_or("A van Vleck path without its group")?;
                        let m=|q:usize,k0:usize|wbar[q]*ubar[k0]*(intensity*denominator);
                        for(pi,&(k1,k0))in gl.pairs.iter().enumerate(){
                            let(at,mirror)=(pi*n*n+x*n+y,pi*n*n+y*n+x);
                            if gl.hermitian{
                                // Its Hermitian and anti-Hermitian halves (H_eff = h0 − ΦS − (i/2)ΦQ), the pair
                                // (k1, k0) standing for (k0, k1) too: of A = Φ_qk M_qk |x⟩⟨y|, ½(A_qk + A_kq†) in S
                                // and i(A_qk − A_kq†) in Q.
                                let(a,b)=(m(k1,k0),m(k0,k1).conj());
                                node[gl.s+at]-=a*0.5;node[gl.s+mirror]-=b*0.5;
                                node[gl.q+at]+=a*Complex::new(0.,1.);node[gl.q+mirror]-=b*Complex::new(0.,1.);
                            }else{node[gl.s+at]-=m(k1,k0);}
                        }
                    }else{
                        // The bound of |W̃ D U| over the trap, every beam at most its peak.
                        let(uc,ud)=(&self.carriers[c].beams,&self.carriers[d].beams);let ub:f64=uc.iter().map(|&l|{let gv=solve(&g.folded[l],p.from.state);
                            ud.iter().map(|&k1|dot(&g.folded[k1],p.via.state,&gv).abs()).sum::<f64>()}).sum();
                        let wb:f64=ud.iter().map(|&k1|{let gv=solve(&g.folded[k1],p.via.state);
                            self.carriers[f].beams.iter().map(|&l|dot(&g.folded[l],p.back.state,&gv).abs()).sum::<f64>()}).sum();
                        omitted[x*n+y]+=ub*wb*denominator.abs();
                    }
                }
                quartic=quartic.max(omitted.iter().copied().fold(0.,f64::max));
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
            out.push(node);
        }
        // The elimination's excited weight per ground state, every near beam at its peak (`resonance`).
        let resonance=(0..ng).map(|b|self.carriers.iter().flat_map(|c|c.beams.iter()).map(|&l|(0..ne).map(|i|e.responses[l][(i,b)].norm_sqr()).sum::<f64>()).sum::<f64>()).fold(0.,f64::max);
        Ok((out,quartic,resonance))
    }

    /// The intensity points of a grid of n nodes per focused beam, in node order.
    fn grid(&self,n:usize)->Vec<Vec<f64>>{
        let d=self.focused.len();(0..n.pow(d as u32)).map(|i|(0..d).map(|f|((i/n.pow(f as u32))%n) as f64/(n-1).max(1) as f64).collect()).collect()
    }
    /// The nodes of a grid of n per focused beam: those it shares with the grid of (n + 1)/2 before
    /// it (`coarse`: every other node along each beam, at the same intensities to the last bit)
    /// taken from that grid, the others computed.
    fn nodes(&self,n:usize,mixing:&[Vec<Vec<Complex>>],coarse:Option<&[Node]>)->Result<Vec<Node>,String>{
        let(d,m)=(self.focused.len(),n.div_ceil(2));
        self.grid(n).par_iter().enumerate().map(|(i,s)|{
            let digits=(0..d).map(|f|(i/n.pow(f as u32))%n);
            match coarse{
                Some(c)if digits.clone().all(|x|x%2==0)=>Ok(c[digits.enumerate().map(|(f,x)|x/2*m.pow(f as u32)).sum::<usize>()].clone()),
                _=>self.node(s,mixing),
            }
        }).collect()
    }
    /// The tables of a grid of n nodes per focused beam, and the quartic shifts' bound over it.
    fn build(&self,n:usize,nodes:&[Node])->(Tables,f64){
        let quartic=nodes.iter().map(|x|x.1).fold(0.,f64::max);
        // The bound over the nodes inside the reach's intensities: each focused beam's from its
        // smallest over the reach (less one node, the interval holding it) to 1.
        let low:Vec<f64>=self.focused.iter().map(|&l|self.reach.iter().map(|&r|self.geometry[l].intensity(r).min(1.)).fold(1.,f64::min)-1./(n-1).max(1) as f64).collect();
        let resonance=self.grid(n).iter().zip(nodes).filter(|(s,_)|s.iter().zip(&low).all(|(x,l)|x>=l)).map(|(_,x)|x.2).fold(0.,f64::max);
        // data[cluster]: node-major, layout.size per node.
        (Tables{nodes:n,data:(0..self.layouts.len()).map(|k|nodes.iter().flat_map(|node|node.0[k].iter().flat_map(|z|[z.re,z.im])).collect()).collect(),resonance},quartic)
    }
    /// The coarse grid's (`self.tables`) largest interpolation error at a finer grid's nodes, relative
    /// to the scale of each cluster's table kind.
    fn interpolation_error(&self,fine:&Tables)->f64{
        let(ng,ne)=(self.ground.ground_states.len(),self.ground.excited_states.len());let points=self.grid(fine.nodes);let mut worst=0_f64;
        for(b,layout)in self.layouts.iter().enumerate(){
            let n=layout.n;let mut kinds:Vec<(usize,usize)>=vec![(layout.h0,layout.h0+n*n)];
            for gl in &layout.groups{let square=gl.pairs.len()*n*n;kinds.extend([(gl.s,gl.s+square),(gl.q,gl.q+square)]);}
            for(c,cl)in layout.carriers.iter().enumerate(){let r=self.carriers[c].rank;let(jump,response)=(3*r*ng*layout.n0,r*ne*layout.n0);
                kinds.extend([(cl.j,cl.j+jump),(cl.j2,cl.j2+jump),(cl.t,cl.t+response)]);}
            // The explicit states' tables, each kind one block (its entries' common scale: a decay
            // between two states with orthogonal photons is zero to rounding).
            let block=|slots:Vec<(usize,usize)>|slots.iter().map(|x|x.0).min().zip(slots.iter().map(|x|x.0+x.1).max());
            kinds.extend(block(layout.crosses.iter().map(|x|(x.at,self.carriers[x.carrier].rank)).collect()));
            kinds.extend(block(layout.decays.iter().map(|x|(x.at,1)).collect()));
            kinds.extend(block(layout.effective.iter().map(|x|(x.at,self.carriers[x.carrier].rank)).collect()));
            if layout.n>layout.n0{kinds.push((layout.emission,layout.emission+3*ng*(layout.n-layout.n0)));}
            for(start,end)in kinds{
                let exact=|i:usize|&fine.data[b][2*(i*layout.size+start)..2*(i*layout.size+end)];
                let modulus=|x:&[f64]|x.chunks_exact(2).map(|z|z[0].hypot(z[1])).fold(0_f64,f64::max);
                let scale=(0..points.len()).map(|i|modulus(exact(i))).fold(0_f64,f64::max);
                if scale==0.{continue;}
                let mut out=vec![Complex::ZERO;end-start];
                for(i,s)in points.iter().enumerate(){
                    self.interpolate(b,&self.stencil(s),None,start,&mut out);
                    let e=out.iter().zip(exact(i).chunks_exact(2)).fold(0_f64,|m,(x,y)|m.max((x.re-y[0]).hypot(x.im-y[1])))/scale;
                    worst=worst.max(e);
                }
            }
        }
        worst
    }
    /// The tables on grids of FIRST_NODES, then doubled (each grid's nodes computed once), until the
    /// cubic interpolation from the coarser grid meets TABLE_SHARE at the finer grid's nodes; the
    /// finer grid is kept.
    fn tabulate(&mut self,mixing:&[Vec<Vec<Complex>>])->Result<(),String>{
        let mut nodes=self.nodes(FIRST_NODES,mixing,None)?;
        let(tables,quartic)=self.build(FIRST_NODES,&nodes);self.tables=tables;self.quartic=quartic/self.ground.generator_scale.max(f64::MIN_POSITIVE);
        loop{
            let n=2*self.tables.nodes-1;nodes=self.nodes(n,mixing,Some(&nodes))?;let(fine,quartic)=self.build(n,&nodes);
            let worst=self.interpolation_error(&fine);
            self.tables=fine;self.table_error=worst;self.quartic=self.quartic.max(quartic/self.ground.generator_scale.max(f64::MIN_POSITIVE));
            if worst<=TABLE_SHARE{self.resonance=self.tables.resonance;return Ok(());}
            if n>4097{return Err(format!("The elimination's tables over the focused beams' intensities do not converge (relative error {worst:.1e} at {n} nodes)"));}
        }
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
    /// The value and the derivatives in every s_f of a slice of W values (2 × its entries), in one
    /// pass over the stencil's nodes; the one-beam stencil unrolled.
    #[inline(always)]
    fn gather<const W:usize>(&self,cluster:usize,stencil:&Stencil,start:usize,value:&mut [f64;MOST_W],derivatives:&mut [[f64;MOST_W];MOST_FOCUSED]){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let base=2*start;let d=stencil.dimensions;
        if d==1{
            let(w,dw)=(stencil.weights[0],stencil.derivatives[0]);let at=stencil.first[0]*size+base;
            let(a,b,c,e)=(&data[at..at+W],&data[at+size..at+size+W],&data[at+2*size..at+2*size+W],&data[at+3*size..at+3*size+W]);
            for j in 0..W{value[j]=w[0]*a[j]+w[1]*b[j]+w[2]*c[j]+w[3]*e[j];derivatives[0][j]=dw[0]*a[j]+dw[1]*b[j]+dw[2]*c[j]+dw[3]*e[j];}
            return;
        }
        value[..W].fill(0.);for g in 0..d{derivatives[g][..W].fill(0.);}
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let at=node*size+base;let x=&data[at..at+W];
            for j in 0..W{value[j]+=w*x[j];}for g in 0..d{for j in 0..W{derivatives[g][j]+=dw[g]*x[j];}}
        }
    }
    /// The value alone (the stencil with the trap beam off).
    #[inline(always)]
    fn gather_value<const W:usize>(&self,cluster:usize,stencil:&Stencil,start:usize,value:&mut [f64;MOST_W]){
        let size=2*self.layouts[cluster].size;let data=&self.tables.data[cluster];let base=2*start;
        value[..W].fill(0.);
        for i in 0..stencil.count(){let(node,w,_)=stencil.node(i);if w==0.{continue;}let at=node*size+base;for(o,x)in value[..W].iter_mut().zip(&data[at..at+W]){*o+=w*x;}}
    }

    /// Everything at (r, κ) but the internal state.
    pub fn point(&self,r:[f64;3],kappa:[f64;3])->Point{
        let mut s=[0.;MOST_FOCUSED];let mut gradient=[[0.;3];MOST_FOCUSED];
        for(i,&l)in self.focused.iter().enumerate(){let(a,g)=self.geometry[l].field(r);s[i]=a.norm_sqr().min(1.);gradient[i]=g.map(|x|2.*(a.conj()*x).re);}
        let mut fields=[CarrierField::default();MOST_CARRIERS];
        for(c,f)in self.carriers.iter().zip(fields.iter_mut()){
            for(j,&l)in c.beams.iter().enumerate(){
                let(a,g)=self.geometry[l].field(r);let w=self.kinetic_shift(l,kappa);
                for k in 0..c.rank{let x=c.coefficients[j][k];let ax=a*x;f.e[k]+=ax;f.doppler[k]+=ax*w;for d in 0..3{f.grad[k][d]+=g[d]*x;}}
            }
        }
        let d=self.focused.len();let stencil=self.stencil(&s[..d]);let mut off=s;off[0]=0.;
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
        if !self.mixed(cluster)&&!self.beating(cluster){return Generator::of(&self.local(cluster,p,t),self.dimension(cluster));}
        self.components(cluster,p).at(t)
    }
    /// A cluster's generator at a point split by its beats (`Components`): its tables' terms, the kept
    /// drives, the decay cross terms and the explicit states' decay (module documentation), each into the
    /// part of its residual -- the static one, or its beat's.
    pub fn components(&self,cluster:usize,p:&Point)->Components{
        let layout=&self.layouts[cluster];let(n,d)=(layout.n,self.focused.len());let nn=n*n;let beats=&layout.beats;let parts=1+beats.omegas.len();
        // The part of a residual: 0 the static one, 1 + j beat j's.
        let part=|w:f64|beats.of(w).map_or(0,|j|j+1);
        let zeros=||vec![Complex::ZERO;nn];
        let mut gradient:Vec<[Vec<Complex>;3]>=(0..parts).map(|_|std::array::from_fn(|_|zeros())).collect();
        let(mut m,mut gamma):(Vec<Vec<Complex>>,Vec<Vec<Complex>>)=((0..parts).map(|_|zeros()).collect(),(0..parts).map(|_|zeros()).collect());
        let mut buf=zeros();let mut slope=vec![zeros();d];
        // acc += c X, and with `adjoint` c X + (c X)† (a Hermitian group's pair k1 < k standing for both).
        let add=|c:Complex,x:&[Complex],acc:&mut [Complex],adjoint:bool|{for a in 0..n{for b in 0..n{acc[a*n+b]+=c*x[a*n+b];if adjoint{acc[a*n+b]+=(c*x[b*n+a]).conj();}}}};
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let q=part(gl.residual);
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                let at=pi*nn;let ee=e.e[k1].conj()*f.e[k];let both=gl.hermitian&&k1<k;
                self.gather_dyn(cluster,&p.stencil,gl.s+at,&mut buf,Some(&mut slope));
                add(ee,&buf,&mut m[q],both);
                // ∇H_herm ∋ −∇(Φ) S − Φ Σ_f ∂_f S ∇s_f.
                for a in 0..3{
                    let c=e.grad[k1][a].conj()*f.e[k]+e.e[k1].conj()*f.grad[k][a];add(-c,&buf,&mut gradient[q][a],both);
                    for g in 0..d{add(-ee*p.intensity_gradient[g][a],&slope[g],&mut gradient[q][a],both);}
                }
                self.gather_dyn(cluster,&p.stencil,gl.q+at,&mut buf,None);add(ee,&buf,&mut gamma[q],both);
            }
        }
        self.gather_dyn(cluster,&p.stencil,layout.h0,&mut buf,Some(&mut slope));
        for a in 0..3{for g in 0..d{let c=Complex::new(p.intensity_gradient[g][a],0.);add(c,&slope[g],&mut gradient[0][a],false);}}
        // The static part less its scalar (its Hermitian part's mean diagonal, a phase); a beat's diagonal,
        // a phase that beats, stays in its part.
        let scalar=(0..n).map(|a|buf[a*n+a].re-m[0][a*n+a].re).sum::<f64>()/n as f64;
        let mut h:Vec<Vec<Complex>>=(0..parts).map(|q|(0..nn).map(|i|(if q==0{buf[i]}else{Complex::ZERO})-m[q][i]-gamma[q][i]*Complex::new(0.,0.5)).collect()).collect();
        for a in 0..n{h[0][a*n+a]-=Complex::new(scalar,0.);}
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
        sized!(1 2 3 4 5 6 7 8)
    }
    #[target_feature(enable="avx2,fma")]
    unsafe fn local_wide<const N:usize,const W:usize>(&self,cluster:usize,p:&Point,t:f64)->Local{self.local_body::<N,W>(cluster,p,t)}
    #[inline(always)]
    fn local_body<const N:usize,const W:usize>(&self,cluster:usize,p:&Point,t:f64)->Local{
        let layout=&self.layouts[cluster];let d=self.focused.len();
        let mut out=Local{hamiltonian:[Complex::ZERO;MOST_STATES*MOST_STATES],scalar:0.,gradient:[[0.;MOST_W];3]};
        let mut buf=[0.;MOST_W];let mut slope=[[0.;MOST_W];MOST_FOCUSED];
        let(mut m,mut gamma)=([0.;MOST_W],[0.;MOST_W]);
        // The pair (k, k1)'s tables are (k1, k)'s adjoints in a Hermitian group: c X + (c X)† for k1 < k.
        // The force reads only Re⟨ψ|·|ψ⟩, so its matrices take 2 c X for the two.
        let adjoint=|c:Complex,x:&[f64;MOST_W],acc:&mut [f64;MOST_W]|{for a in 0..N{for b in 0..N{let(i,j)=(2*(a*N+b),2*(b*N+a));let(u,v)=(x[j],-x[j+1]);acc[i]+=c.re*u+c.im*v;acc[i+1]+=c.re*v-c.im*u;}}};
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);let phase=Complex::cis(gl.residual*t);
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                let at=pi*N*N;let ee=phase*e.e[k1].conj()*f.e[k];let both=if gl.hermitian&&k1<k{2.}else{1.};
                self.gather::<W>(cluster,&p.stencil,gl.s+at,&mut buf,&mut slope);
                caxpy::<W>(ee,&buf,&mut m);if gl.hermitian&&k1<k{adjoint(ee,&buf,&mut m);}
                // ∇H_herm ∋ −∇(Φ) S − Φ Σ_f ∂_f S ∇s_f.
                for a in 0..3{
                    let mut c=phase*(e.grad[k1][a].conj()*f.e[k]+e.e[k1].conj()*f.grad[k][a]);
                    caxpy::<W>(c*(-both),&buf,&mut out.gradient[a]);
                    for g in 0..d{c=ee*(-both*p.intensity_gradient[g][a]);caxpy::<W>(c,&slope[g],&mut out.gradient[a]);}
                }
                self.gather_value::<W>(cluster,&p.stencil,gl.q+at,&mut buf);
                caxpy::<W>(ee,&buf,&mut gamma);if gl.hermitian&&k1<k{adjoint(ee,&buf,&mut gamma);}
            }
        }
        self.gather::<W>(cluster,&p.stencil,layout.h0,&mut buf,&mut slope);
        for a in 0..3{for g in 0..d{let c=Complex::new(p.intensity_gradient[g][a],0.);caxpy::<W>(c,&slope[g],&mut out.gradient[a]);}}
        let at=|v:&[f64;MOST_W],a:usize,b:usize|Complex::new(v[2*(a*N+b)],v[2*(a*N+b)+1]);
        let mut scalar=0.;for a in 0..N{scalar+=buf[2*(a*N+a)]-m[2*(a*N+a)];}scalar/=N as f64;out.scalar=scalar;
        for a in 0..N{for b in 0..N{
            let herm=at(&buf,a,b)-at(&m,a,b);
            out.hamiltonian[a*N+b]=herm-at(&gamma,a,b)*Complex::new(0.,0.5)-if a==b{Complex::new(scalar,0.)}else{Complex::ZERO};
        }}
        out
    }

    /// −∇Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩ (rad/us per um): the Ehrenfest force on ħκ.
    pub fn force(&self,cluster:usize,local:&Local,psi:&[Complex])->[f64;3]{
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(cluster){$($n=>force_sized::<$n>(local,psi),)*_=>unreachable!("a cluster has 1 to 8 states")}}}
        sized!(1 2 3 4 5 6 7 8)
    }
    /// The trap's potential of ψ at a point (rad/us): Re⟨ψ|H_eff|ψ⟩ less the same with the trap beam
    /// off, per ⟨ψ|ψ⟩ -- (h0 − M) at the point's stencil less at its far one, with the same fields.
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
        self.gather_value::<W>(cluster,&p.stencil,layout.h0,&mut here);self.gather_value::<W>(cluster,&p.far,layout.h0,&mut far);
        for j in 0..W{difference[j]=here[j]-far[j];}
        let norm:f64=psi[..N].iter().map(|z|z.norm_sqr()).sum();
        if bound{return quadratic::<N>(&difference,psi).re/norm+2.*self.ground.light;}
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                // −Φ (S(s) − S(far)), the pair (k1, k) standing for (k, k1) in a Hermitian group.
                let ee=e.e[k1].conj()*f.e[k]*if gl.hermitian&&k1<k{2.}else{1.};let at=gl.s+pi*N*N;
                self.gather_value::<W>(cluster,&p.stencil,at,&mut here);self.gather_value::<W>(cluster,&p.far,at,&mut far);
                for j in 0..W{here[j]-=far[j];}caxpy::<W>(-ee*Complex::cis(gl.residual*t),&here,&mut difference);
            }
        }
        quadratic::<N>(&difference,psi).re/norm
    }

    /// An explicit state's dressed photon ĉ_a[to] at a stencil (the p-th explicit state of a cluster).
    pub fn photon(&self,cluster:usize,stencil:&Stencil,p:usize,a:usize,to:usize)->Complex{
        let mut z=[Complex::ZERO];let ng=self.ground.ground_states.len();self.gather_dyn(cluster,stencil,self.layouts[cluster].emission+(p*3+a)*ng+to,&mut z,None);z[0]
    }
    /// `trap_sized` on any number of states.
    fn trap_mixed(&self,cluster:usize,p:&Point,psi:&[Complex],t:f64,bound:bool)->f64{
        let layout=&self.layouts[cluster];let n=layout.n;let nn=n*n;let(mut here,mut far)=(vec![Complex::ZERO;nn],vec![Complex::ZERO;nn]);
        self.gather_dyn(cluster,&p.stencil,layout.h0,&mut here,None);self.gather_dyn(cluster,&p.far,layout.h0,&mut far,None);
        let mut difference:Vec<Complex>=here.iter().zip(&far).map(|(a,b)|*a-*b).collect();
        let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum();
        let form=|x:&[Complex]|->f64{let mut s=Complex::ZERO;for a in 0..n{for b in 0..n{s+=psi[a].conj()*x[a*n+b]*psi[b];}}s.re};
        if bound{return form(&difference)/norm+2.*self.ground.light;}
        for gl in &layout.groups{
            let(f,e)=(&p.fields[gl.absorbed],&p.fields[gl.emitted]);
            for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                let ee=e.e[k1].conj()*f.e[k]*if gl.hermitian&&k1<k{2.}else{1.};let at=gl.s+pi*nn;
                self.gather_dyn(cluster,&p.stencil,at,&mut here,None);self.gather_dyn(cluster,&p.far,at,&mut far,None);
                let c=-ee*Complex::cis(gl.residual*t);for i in 0..nn{difference[i]+=c*(here[i]-far[i]);}
            }
        }
        form(&difference)/norm
    }

    /// The photon amplitudes at a point of every class of a cluster's jumps (`Photons`), with the
    /// exact responses in a step beyond the first order's allowance (engine.rs): a folded photon's
    /// C_x A_c ψ = −C_x R_c ψ, an explicit state's C_x ψ_e.
    pub fn photons<'a>(&'a self,cluster:usize,p:&Point,psi:&[Complex],t:f64,exact:Option<Exact>)->Photons<'a>{
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
        // The classes: each photon's entry into its target state's row from its source's column.
        let amplitude=cl.classes.iter().map(|class|{let target=&g.clusters[class.target];
            let mut u=[[Complex::ZERO;MOST_STATES];3];
            for &(ph,w)in &class.photons{
                let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());let phase=Complex::cis(w*t);
                for a in 0..3{u[a][row]+=phase*matrices[(ph.carrier*3+a)*size+ph.to*n0+col]*psi[col];}
            }
            for &(i,to,w)in &class.direct{
                let row=target.position(to).unwrap();let phase=Complex::cis(w*t)*psi[i];
                for a in 0..3{u[a][row]-=phase*self.photon(cluster,&p.stencil,i-n0,a,to);}
            }
            u}).collect();
        let mut state=[Complex::ZERO;MOST_MIXED];state[..n].copy_from_slice(&psi[..n]);
        Photons{model:self,cluster,point:*p,psi:state,time:t,jumps,doppler,exact,amplitude}
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

    /// A cluster state's potential at r relative to the focus (rad/us): the diagonal of h0 and of
    /// every beam's own light shift, |a_l(r)|² Σ c_lk′* c_lk S_k′k -- the beams' interference (their
    /// lattice) left out. A tweezer folded at second order is in h0; one that is a near carrier
    /// (the only light): the trap is the same.
    pub fn potential(&self,cluster:usize,x:usize,r:[f64;3])->f64{
        let at=|r:[f64;3]|->f64{
            let layout=&self.layouts[cluster];let n=layout.n;let n2=n*n;
            let s:Vec<f64>=self.focused.iter().map(|&l|self.geometry[l].intensity(r).min(1.)).collect();let stencil=self.stencil(&s);
            let mut buf=vec![Complex::ZERO;n2];self.interpolate(cluster,&stencil,None,layout.h0,&mut buf);let mut u=buf[x*n+x].re;
            for gl in layout.groups.iter().filter(|g|g.hermitian){
                let carrier=&self.carriers[gl.absorbed];
                for(j,&l)in carrier.beams.iter().enumerate(){
                    let intensity=self.geometry[l].intensity(r);let coefficient=&carrier.coefficients[j];
                    for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                        self.interpolate(cluster,&stencil,None,gl.s+pi*n2,&mut buf);
                        u-=intensity*(coefficient[k1].conj()*coefficient[k]*buf[x*n+x]).re*if k1<k{2.}else{1.};
                    }
                }
            }
            u
        };
        at(r)-at([0.;3])
    }
}

/// The exact responses of every near beam at a point for a segment beyond the first order's
/// allowance (engine.rs): R_l(w_l) at the point's elimination.
#[derive(Clone,Debug)]
pub struct Exact{pub responses:Vec<Dense>}
impl Exact{
    /// The exact responses at a point and momentum.
    pub fn new(model:&Model,r:[f64;3],kappa:[f64;3])->Result<Exact,String>{
        let g=&model.ground;let e=g.eliminate(&model.hamiltonian_at_point(r))?;
        let mut responses=vec![Dense::zeros(g.excited_states.len(),g.ground_states.len());g.beams.len()];
        for c in &g.carriers{for &l in &c.beams{responses[l]=g.response(&e,l,model.kinetic_shift(l,kappa))?;}}
        Ok(Exact{responses})
    }
}
impl Model{
    /// H at a point: the focused beams at their intensities there, the others at their peak.
    pub fn hamiltonian_at_point(&self,r:[f64;3])->Dense{
        let s:Vec<f64>=self.focused.iter().map(|&l|self.geometry[l].intensity(r).min(1.)).collect();self.hamiltonian_at(&s)
    }
}

fn force_sized<const N:usize>(local:&Local,psi:&[Complex])->[f64;3]{
    let norm:f64=psi[..N].iter().map(|z|z.norm_sqr()).sum();
    std::array::from_fn(|a|-quadratic::<N>(&local.gradient[a],psi).re/norm)
}

/// A cluster's post-jump transient operators (`Model::transient`): Y (n0 × n0) and the pieces.
#[derive(Clone,Debug)]
pub struct Transient{pub n0:usize,pub y:Vec<Complex>,pub pieces:Vec<Piece>}
/// A piece of the transient: its coherent group and excited group, its target cluster, its photon's
/// operators per axis (the target's ground states × the cluster's, row-major) and wave number.
#[derive(Clone,Debug)]
pub struct Piece{pub coherent:usize,pub excited:usize,pub target:usize,pub ops:[Vec<Complex>;3],pub wave_number:f64}

/// The photon amplitudes of every class of a cluster's jumps at a point: per class and axis the
/// target cluster's amplitude u_a; for a class drawn, its gradient and its beams' shares
/// (`Photons::gradient`, `Photons::shares`) from the jump tables at the point, kept.
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
    /// ∇u_a [a][axis] of a class: the folded photons' dependence on r through the fields and the
    /// intensities (the Doppler term's own gradient left out); an explicit state's photon has none.
    pub fn gradient(&self,class:usize)->[[[Complex;MOST_STATES];3];3]{
        let m=self.model;let g=&m.ground;let cl=&g.clusters[self.cluster];let class=&cl.classes[class];let target=&g.clusters[class.target];
        let(n0,p)=(m.ground_dimension(self.cluster),&self.point);let cl2=&m.layouts[self.cluster].carriers;let size=g.ground_states.len()*n0;
        let mut out=[[[Complex::ZERO;MOST_STATES];3];3];let mut entry=[Complex::ZERO];
        for &(ph,w)in &class.photons{
            let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());let phase=Complex::cis(w*self.time)*self.psi[col];let fl=&p.fields[ph.carrier];
            for k in 0..m.carriers[ph.carrier].rank{for a in 0..3{
                let j=self.jumps[self.offset(ph.carrier,k,a)+ph.to*n0+col];
                for x in 0..3{out[a][x][row]+=phase*fl.grad[k][x]*j;}
                for f in 0..m.focused.len(){
                    m.interpolate(self.cluster,&p.stencil,Some(f),cl2[ph.carrier].j+(k*3+a)*size+ph.to*n0+col,&mut entry);
                    for x in 0..3{out[a][x][row]+=phase*fl.e[k]*entry[0]*p.intensity_gradient[f][x];}
                }
            }}
        }
        out
    }
    /// Each beam's share of a class's photon of polarization ε: its folded part
    /// −a_l Σ_k c_lk (J_k − w_l J2_k) ψ (its exact response's in a step beyond the first order), the
    /// weight ‖Σ_a ε_a v_a‖²; and of the explicit states' part, its weight shared by the beams in
    /// proportion to their drives' amplitude into the explicit states, Σ |a_l Σ_k c_lk U_k[e,b] ψ_b|².
    pub fn shares(&self,class:usize,epsilon:&[f64;3])->Vec<(usize,f64)>{
        let m=self.model;let g=&m.ground;let cl=&g.clusters[self.cluster];let class=&cl.classes[class];let target=&g.clusters[class.target];
        let(n0,p)=(m.ground_dimension(self.cluster),&self.point);let nt=target.states.len();
        let weight=|v:&[[Complex;MOST_STATES];3]|->f64{(0..nt).map(|i|(0..3).fold(Complex::ZERO,|s,a|s+v[a][i]*epsilon[a]).norm_sqr()).sum()};
        let carriers:Vec<usize>={let mut v:Vec<usize>=class.photons.iter().map(|x|x.0.carrier).collect();v.sort();v.dedup();v};
        let mut out:Vec<(usize,f64)>=carriers.iter().flat_map(|&c|m.carriers[c].beams.iter().enumerate().map(move|(j,&l)|(c,j,l))).map(|(c,j,l)|{
            let(amplitude,w)=(m.geometry[l].field(p.r).0,m.kinetic_shift(l,p.kappa));let mut v=[[Complex::ZERO;MOST_STATES];3];
            let exact:Option<[Dense;3]>=self.exact.as_ref().map(|e|std::array::from_fn(|a|g.emission[a].product(&e.responses[l])));
            for &(ph,phase)in class.photons.iter().filter(|x|x.0.carrier==c){
                let(row,col)=(target.position(ph.to).unwrap(),cl.position(ph.from).unwrap());let phase=-Complex::cis(phase*self.time)*self.psi[col]*amplitude;
                for a in 0..3{
                    let z=match &exact{Some(e)=>e[a][(ph.to,cl.states[col])],None=>(0..m.carriers[c].rank).fold(Complex::ZERO,|s,k|{let at=self.offset(c,k,a)+ph.to*n0+col;
                        s+m.carriers[c].coefficients[j][k]*(self.jumps[at]-self.doppler[at]*w)})};
                    v[a][row]+=phase*z;
                }
            }
            (l,weight(&v))}).collect();
        if !class.direct.is_empty(){
            let mut v=[[Complex::ZERO;MOST_STATES];3];
            for &(i,to,w)in &class.direct{let row=target.position(to).unwrap();let phase=Complex::cis(w*self.time)*self.psi[i];
                for a in 0..3{v[a][row]+=phase*m.photon(self.cluster,&p.stencil,i-n0,a,to);}}
            let total=weight(&v);
            // The drives' amplitudes into the explicit states, per beam.
            let mut drives:Vec<(usize,f64)>=vec![];
            for dr in &m.layouts[self.cluster].drives{
                let carrier=&m.carriers[dr.carrier];
                for(j,&l)in carrier.beams.iter().enumerate(){
                    let z=(0..carrier.rank).fold(Complex::ZERO,|s,k|s+carrier.coefficients[j][k]*dr.u[k])*m.geometry[l].field(p.r).0*self.psi[dr.col];
                    match drives.iter_mut().find(|x|x.0==l){Some(x)=>x.1+=z.norm_sqr(),None=>drives.push((l,z.norm_sqr()))}
                }
            }
            let sum:f64=drives.iter().map(|x|x.1).sum();
            if sum>0.{for(l,d)in drives{let share=total*d/sum;match out.iter_mut().find(|x|x.0==l){Some(x)=>x.1+=share,None=>out.push((l,share))}}}
        }
        out
    }
}

#[cfg(test)]
mod tests{
    use super::Beats;
    use std::f64::consts::TAU;
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
