//! The fold's step by a decomposition kept on the far coordinates, the near light in linear combinations, the energies and
//! the Doppler shifts on the diagonal and the energy's self-consistency by a series (the first road's minimal prototype,
//! BRIEF_PROTO.md; not the run's, the forward alone). The same object as `Fold::at` on the same choice of nodes, with every
//! excited row of a block that solves any row solved (a centred excited row there costs a mode, not a pass); `Fold::at`
//! is the truth of its tests. `Fold::at` and `Fold::step` are not touched.
//!
//! # The far coordinates
//! Every frequency's field at a point on its basis, ε_k = u_k†F (field.rs `Frequency::components`). The outside forms are
//! Σ_kk' ε_k*ε_k' N_kk' (N_kk' = Σ_pq u_kp* u_k'q O_pq), and a centred copy's dressing of a block's excited rows,
//! Ĥ_SC R₀ Ĥ_CS, is bilinear in its emitting component's coefficients, ε (co-rotating) or ε* (counter-rotating): terms of
//! the same pairs (`Pair`), R₀ held at the decomposition's reference. So K on a block's solved rows is, anywhere,
//! K_ref + Σ_t (c_t − c_t,ref) N_t with c_t = ε_k*ε_k' -- exactly, but for R₀'s own change (read).
//!
//! # What is kept (`Cache`, per block, by the far coordinates)
//! A block with solved rows (`Near`): the Schur form V T V† of K_ref + o part by part (unitary, fold.rs `Factored`; the
//! normal parts diagonal); per term the diagonal of V†N_tV and per row and part the norm of the rest; per beam its source in
//! the basis Ỹ_l = V†X_l; per source vector Y_a = V†U_a; per centred copy vector R₀U_a†V (the copies' rows' response);
//! ‖R₀‖. A block whose rows are all centred (`Far`): its response T = R₀B is its field coefficients times kept vectors, so
//! its compression is the pair forms U_a†R₀U_b and U_a†R₀†R₀U_b in e_a*e_b.
//!
//! # A step
//! 1. The light at the point: the beams' amplitudes, the fields' ε, the pairs' coefficients, the components' coefficients
//!    e and beam factors f_l, every beam's Doppler shift w_l, Ĥ_PP.
//! 2. The diagonal moved to first order, κ_i = T_ii − o + Σ_t Re δc_t (V†N_tV)_ii; the rest of the change is read.
//! 3. Ṽ_l = f_l Ỹ_l, beam by beam; Ṽ = Σ_a e_a Y_a.
//! 4. The far blocks' pair forms into H, A and M.
//! 5. The series: Z_l solves Z_l H − (T + w_l − o) Z_l = Ṽ_l with H = Ĥ_PP + Σ Ṽ†Z the Bloch generator. A sweep is
//!    Z_l ← (c − w_l − T)⁻¹(Ṽ_l − Z_l Δ), c the columns' energies on the diagonal (each column at its own, exact; a common centre for all columns, the
//!    brief's z₀, was measured and dropped: on an S2Q set its series took 6–11 sweeps against two, on R4 F=2 its reading
//!    fell short), Δ = H − diag(c) the
//!    series' part, H made again from the sweep's Z: each sweep adds the next order of Δ and of the self-consistency at
//!    once. Its change, column by column over every row of every block (each mode with its own column: no norm of the
//!    whole), reads it: the next change is the last times their ratio, what the rest add the last times r/(1 − r). Two
//!    sweeps at the least (a ratio is read from two changes), then until the rest reads within the share.
//! 6. The readings of what is held: the reuse's (fold.rs # One decomposition along the path, row by row, on the response
//!    the step uses, Z = Σ_l Z_l, at the beams' mean Doppler shift, with the beams' spread as a remainder; R₀'s own change
//!    with it) and the far blocks' (‖R₀‖ times the distance of the energies and the forms from the reference). Beyond the
//!    share the nearest kept decomposition, then a new one (fold.rs `Path`'s rule).
//! 7. The copies' rows T_C = R₀Ĥ_CS VZ: their own part of the compression (z₀T_C†T_C, T_C†T_C; their dressing is the kept
//!    decomposition's) only where it reads beyond the share (‖T_C‖, a copy row's absolute unit). The compression
//!    A = H + Σ(Ṽ†Z)† + Z†(T + δκ − o)Z (+ z₀T_C†T_C) + z₀ M_far, M = Σ Z†Z (+ T_C†T_C) + M_far (less one);
//!    S⁻¹ = (1 + M)^{−1/2} to the rounding (`inverse_root`), G = S⁻¹AS⁻¹. Its anti-Hermitian part is −(i/2)Σε†Dε by
//!    construction: Z†TZ's is Z†V†DVZ (T the Schur form of K_ref, whose anti-Hermitian part is −D/2; V†DV is not stored
//!    apart), the far forms' T†DT (R₀ − R₀† = −iR₀†DR₀).
//!
//! # The nodes (DESIGN §4.2)
//! The integrator's outer nodes s = ±τ/2 (`Fold::prototype_nodes`): the decomposition, H, S and every split's response per
//! unit amplitude held over the step, the field turned -- Z_l(s) = (f_l(s)/f_l) Z_l, Ṽ(s) = Σ_a e_a(s) Y_a, Ĥ_PP(s) and
//! the far forms at s exactly --, A(s) by the same compression, G(s) = S⁻¹A(s)S⁻¹.
//!
//! # The force (DESIGN §1.3)
//! −∇ Σ_j Re tr(Ḡ_j†G_j) over the step's positions moved together, by the adjoint of the step as it was evaluated
//! (`Fold::prototype_force`): G = S⁻¹AS⁻¹; S⁻¹'s Fréchet adjoint, S X̄ + X̄ S = −S⁻¹N̄S⁻¹ by the fixed point the fold's
//! existence makes a contraction; the compressions; every sweep, the last first, its triangular solves adjoint on the
//! same kept Schur form; to the light's coefficients (the splits' factors, the components', the pairs') contracted with
//! the beams' analytic gradients. K's change is taken whole on the decomposition's parts (V†N_tV, not its diagonal alone:
//! the reuse's first order moves the diagonal, the gradient is all of K's). R₀ and z₀ are held as the step holds them;
//! their own change along the axis bounds what that leaves (`Force::held`, ~10⁻⁶ of the force on the scenes).
//!
//! # The jumps (DESIGN §1.3, §2.6)
//! L_q = C_q ε S⁻¹ of the same responses at every position (`Fold::prototype_jumps`): ε = VZ of a near block (Z(s) at a
//! node), R₀Σ_a e_a U_a of a far one; i(G − G†) = Σ L†L to the rounding, S the middle's at the nodes too. The photons a
//! state emits per step (`Fold::prototype_photons`): u = C_q V Z S⁻¹ψ and its whole gradient, the step's tangent along each
//! axis by the same sweeps (the beams' fields, and the responses' own change through Ĥ_PP, K̃, Ṽ and H) and S⁻¹'s: the
//! responses' own change is the excited states' differential light shift along r (the tweezer's), a tenth of the
//! diffusion on R4 F=2 and nearly all of it on PGC3D F=1, where the one repump beam's field alone gives none.
//!
//! The copies a block solves exactly (the P layer's Woodbury), the explicit states and the static sources are not here
//! (`Fold::prototype_workspace` refuses them): none of the prototype's two scenes has them. Their copies are the tweezer's,
//! whose dressing (the excited states' tweezer shift, ~10Γ) no series about the excited response holds: it is the kept
//! decomposition's.
use super::*;
use crate::field::Amplitude;

/// A pair of one frequency's field coefficients (frequency w, k, k'): its coefficient ε_k*ε_k'.
type Pair = (usize, usize, usize);

/// The step's phases, as BRIEF_PROTO.md splits its time.
pub(crate) const PHASES: [&str; 10] = ["place", "refresh", "Ṽ", "diagonal", "far", "series", "reading", "copies", "compression", "normal"];
const PLACE: usize = 0;
const REFRESH: usize = 1;
const SOURCES: usize = 2;
const DIAGONAL: usize = 3;
const FAR: usize = 4;
const SERIES: usize = 5;
const READING: usize = 6;
const COPIES: usize = 7;
const COMPRESSION: usize = 8;
const NORMAL: usize = 9;

/// The processor's time stamp (its ticks; the measurement converts them by the wall clock).
fn ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

/// A block with solved rows: its layout (rows s, centred copies nc, its offset; the P places of its sources, as fold.rs
/// `lay` takes its centre; its beam splits (component, beam); its source vectors (component, k); per copy component its
/// centred rows, and its copy vectors (component, k, group); its terms) and the step's buffers.
#[derive(Clone)]
struct Near {
    o: usize,
    s: usize,
    nc: usize,
    value: f64,
    centre: Vec<usize>,
    splits: Vec<(usize, usize)>,
    vectors: Vec<(usize, usize)>,
    groups: Vec<(usize, Vec<usize>)>,
    copies: Vec<(usize, usize, usize)>,
    terms: Vec<usize>,
    // The step's.
    delta: Vec<C64>,
    kappa: Vec<C64>,
    off: Vec<f64>,
    rho: Vec<f64>,
    shifts: Vec<f64>,
    vt: Vec<C64>,
    vtot: Vec<C64>,
    z: Vec<C64>,
    ztot: Vec<C64>,
    zold: Vec<C64>,
    rhs: Vec<C64>,
    bloch: Vec<C64>,
    tc: Vec<C64>,
    y: Vec<C64>,
    err: Vec<f64>,
    norms: Vec<f64>,
    active: Vec<bool>,
    held: Vec<C64>,
    q: Vec<f64>,
    reading: f64,
    // Each sweep's, for the force (`Fold::prototype_force`): per split its diagonal's inverse and its response, and the
    // response summed over the splits.
    invs: Vec<C64>,
    zs: Vec<C64>,
    ztots: Vec<C64>,
    // The force's: per split the response's cotangent and the held rows' (to the first sweep), the adjoint's solution
    // Λ, the response summed's cotangent, the cotangents of K̃ (s × s, its parts' entries) and of Ṽ, the splits' factors'.
    zbar: Vec<C64>,
    bucket: Vec<C64>,
    lam: Vec<C64>,
    ztbar: Vec<C64>,
    kbar: Vec<C64>,
    vbar: Vec<C64>,
    fbar: Vec<C64>,
    // At the step's outer nodes (`Fold::prototype_nodes`): per node the splits' factors' ratios to the middle's, the
    // response Z(s), Ṽ(s), the copies' rows.
    ratio: [Vec<C64>; 2],
    znode: [Vec<C64>; 2],
    vnode: [Vec<C64>; 2],
    tcnode: [Vec<C64>; 2],
    copied_in: bool,
    // The exact copies (module documentation, # The exact copies): their rows nd (laid after the solved excited ones, in
    // their own coordinates), per row its (state, component), their coupling vectors (component, k); the step's K̂_ED
    // (ne × nd), K̂_DE (nd × ne), K̂_DD (nd × nd, less o); per split their response (nd × n) and summed, each sweep's
    // (with per split and column the solve's P_j = A_j⁻¹K̂_ED (nd × ne) and C_j⁻¹ (nd × nd)); the force's and the nodes'.
    nd: usize,
    dstates: Vec<(usize, usize)>,
    exact: Vec<(usize, usize)>,
    ked: Vec<C64>,
    kde: Vec<C64>,
    kdd: Vec<C64>,
    zd: Vec<C64>,
    zdtot: Vec<C64>,
    zds: Vec<C64>,
    zdtots: Vec<C64>,
    pjs: Vec<C64>,
    cinvs: Vec<C64>,
    zdbar: Vec<C64>,
    zdtbar: Vec<C64>,
    kedbar: Vec<C64>,
    kdebar: Vec<C64>,
    kddbar: Vec<C64>,
    zdnode: [Vec<C64>; 2],
    kednode: [Vec<C64>; 2],
    kdenode: [Vec<C64>; 2],
}

/// The light at a place (module documentation, 1): the beams' amplitudes with their gradients, each frequency's components
/// ε_k on its basis, the pairs' coefficients, per component its beams' factors f_l and Doppler shifts w_l and its
/// coefficients e_k, Ĥ_PP.
#[derive(Clone)]
struct Placed {
    amplitudes: Vec<Amplitude>,
    eps: Vec<[C64; 3]>,
    coef: Vec<C64>,
    f: Vec<Vec<C64>>,
    w: Vec<Vec<f64>>,
    e: Vec<Vec<C64>>,
    hpp: Vec<C64>,
}

/// A step's outer node (`Fold::prototype_nodes`): its light, A(s) and G(s) = S⁻¹A(s)S⁻¹ (S the middle's), per far block
/// its forms there.
struct Node {
    at: Placed,
    a: Mat,
    generator: Mat,
    hf: Vec<Vec<C64>>,
    mf: Vec<Vec<C64>>,
}

/// A block whose rows are all centred: its rows m, its centre's P places, its source vectors (component, k), its beam
/// splits (for the Doppler shifts its centring reads), its terms; the step's.
struct Far {
    o: usize,
    m: usize,
    centre: Vec<usize>,
    vectors: Vec<(usize, usize)>,
    splits: Vec<(usize, usize)>,
    terms: Vec<usize>,
    delta: Vec<C64>,
    products: Vec<C64>,
    hf: Vec<C64>,
    mf: Vec<C64>,
    reading: f64,
    centring: f64,
}

/// A near block's decomposition (module documentation, # What is kept).
struct NearBasis {
    at: Vec<C64>,
    weights: Vec<f64>,
    f: Factored,
    part: Vec<usize>,
    diag: Vec<C64>,
    norms: Vec<f64>,
    dressed: Vec<f64>,
    copied: Vec<f64>,
    beams: Vec<Vec<C64>>,
    vectors: Vec<Vec<C64>>,
    copies: Vec<Vec<C64>>,
    copied_rows: Vec<f64>,
    z0: f64,
    r0: f64,
    /// Per term V†N_tV on the parts' entries (s × s, zero across parts: the force's contraction) and the norm of its
    /// dressing's part; the emission on the block's rows C_qV (GROUND × s, the jumps and the photons).
    within: Vec<Vec<C64>>,
    dress: Vec<f64>,
    emit: [Vec<C64>; 3],
    /// The exact copies: per coupling vector (component g, k) V†U_k on (E rows, D rows of g) (ne × nd); what the
    /// decomposition's dressing adds to K̂_ED and K̂_DE at the reference beyond the bare drives (held); K̂_DD at the
    /// reference (less o) and per term its form on the D rows (nd × nd).
    couple: Vec<Vec<C64>>,
    ked0: Vec<C64>,
    kde0: Vec<C64>,
    kdd0: Vec<C64>,
    ddterms: Vec<Vec<C64>>,
}

/// A far block's pair forms (module documentation, # What is kept): per pair of source vectors (a, b) U_a†R₀U_b and
/// U_a†R₀†R₀U_b (n × n); its R₀ (the tests' response) and ‖R₀‖; the forms' size on its rows per term (R₀'s change).
struct FarBasis {
    at: Vec<C64>,
    weights: Vec<f64>,
    f: Vec<Vec<C64>>,
    g: Vec<Vec<C64>>,
    copied: Vec<f64>,
    resolvent: Mat,
    z0: f64,
    r0: f64,
    /// Per source vector and emission axis C_qR₀U_a (GROUND × n): the jumps and the photons.
    emitted: Vec<[Vec<C64>; 3]>,
}

/// An atom's kept decompositions, per block (near and far apart): the kept ones and the one it stands on; the
/// decompositions made and the kept ones taken up again.
#[derive(Default)]
pub(crate) struct Cache {
    near: Vec<Vec<NearBasis>>,
    far: Vec<Vec<FarBasis>>,
    at_near: Vec<Option<usize>>,
    at_far: Vec<Option<usize>>,
    pub(crate) made: usize,
    pub(crate) looked: usize,
}

/// A step's result: A, the metric less one, S⁻¹, G (n × n); the sweeps; the readings (rate units): the series' rest,
/// the largest of the near blocks' reuse, of the far blocks' held forms against `Fold::at` and of their centring against
/// the exact energies; the last changes' largest ratio; the work (complex multiply-adds) and the ticks by phase
/// (`PHASES`), and per sweep.
pub(crate) struct Step {
    pub(crate) a: Mat,
    pub(crate) metric: Mat,
    pub(crate) normal: Mat,
    pub(crate) generator: Mat,
    pub(crate) sweeps: usize,
    pub(crate) series: f64,
    pub(crate) reuse: f64,
    pub(crate) far: f64,
    pub(crate) centring: f64,
    pub(crate) copies: f64,
    pub(crate) ratio: f64,
    pub(crate) made: usize,
    pub(crate) rounds: usize,
    pub(crate) work: [f64; 10],
    pub(crate) ticks: [u64; 10],
    pub(crate) sweep_work: Vec<f64>,
    pub(crate) sweep_ticks: Vec<u64>,
}

/// The prototype's layout and buffers, once per fold (`Fold::prototype_workspace`).
pub(crate) struct Workspace {
    n: usize,
    pairs: Vec<Pair>,
    /// Per pair its entries on Ĥ_PP (flat index, value).
    pp: Vec<Vec<(usize, C64)>>,
    /// Ĥ_PP less the forms (the ground states of one frame, less the frame).
    base: Vec<C64>,
    near: Vec<Near>,
    far: Vec<Far>,
    /// The middle's light, and per frequency, beam and basis vector u_k†E_l (the components' gradients).
    mid: Placed,
    proj: Vec<Vec<C64>>,
    /// Each sweep's column energies and series' part Δ (the force's).
    cs: Vec<C64>,
    deltas: Vec<C64>,
    /// The step's outer nodes, their offsets along the path (µs) and whether they were made at this step.
    nodes: Vec<Node>,
    node_times: [f64; 2],
    noded: bool,
    hbase: Vec<C64>,
    h: Vec<C64>,
    delta: Vec<C64>,
    c: Vec<C64>,
    change: Vec<f64>,
    previous: Vec<f64>,
    whole: Vec<f64>,
    eigen: Mat,
    vectors: Mat,
    temp: Mat,
    renew: Vec<(bool, usize)>,
    order: Vec<(f64, usize, usize)>,
    step: Step,
}

thread_local! {
    static SPARE: std::cell::RefCell<Vec<Vec<C64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A vector of the thread's spares (assemble.rs `Held`'s rule for the prototype's temporaries: a step's force and photons
/// take the vectors the step before left, and ask the heap for nothing).
struct Buf(Vec<C64>);

impl Buf {
    /// `len` zeros.
    fn zeros(len: usize) -> Buf {
        let mut v = SPARE.try_with(|x| x.borrow_mut().pop()).ok().flatten().unwrap_or_default();
        v.clear();
        v.resize(len, C64::ZERO);
        Buf(v)
    }
    /// A copy of `x`.
    fn copy(x: &[C64]) -> Buf {
        let mut b = Buf::zeros(0);
        b.0.extend_from_slice(x);
        b
    }
}

impl std::ops::Deref for Buf {
    type Target = Vec<C64>;
    fn deref(&self) -> &Vec<C64> {
        &self.0
    }
}

impl std::ops::DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut Vec<C64> {
        &mut self.0
    }
}

impl<'a> IntoIterator for &'a Buf {
    type Item = &'a C64;
    type IntoIter = std::slice::Iter<'a, C64>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl AsRef<[C64]> for Buf {
    fn as_ref(&self) -> &[C64] {
        &self.0
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.0);
        let _ = SPARE.try_with(|x| x.borrow_mut().push(v));
    }
}

/// 1/z by one real division (z away from zero and overflow: the series' denominators).
#[inline]
fn reciprocal(z: C64) -> C64 {
    let d = 1.0 / z.norm_sqr();
    C64::new(z.re * d, -z.im * d)
}

/// V†MV of a block's solved rows (Schur parts).
fn rotate(f: &Factored, m: &Mat, work: &mut f64) -> Mat {
    f.to(&f.to(m, work).adjoint(), work).adjoint()
}

/// The rate's inverse: the relative amplitude error ρ of a reading r = 2ρ + ρ².
fn unrate(r: f64) -> f64 {
    (1.0 + r).sqrt() - 1.0
}

/// The series' order, mode by mode (module documentation, 5): after the first sweep the next one moves a Schur part P of a
/// block by at most q_P a_j in column j -- the next change is the diagonal times the first sweep's residual against the
/// Bloch generator H¹ it gives, −Z¹(H¹ − c⁰), so row by row against column by column (Cauchy–Schwarz) and beam by beam,
/// q_P = Σ_l (Σ_{i∈P} max_j |c⁰_j − w_l − κ_i|⁻² ‖Z¹_l,i‖²)^½ and a_j = ‖(H¹ − c⁰)_:,j‖. The parts of the smallest next order
/// stay at the first sweep's while what they leave together, (Σ q_P²)^½ max_j a_j/‖Z_j‖, reads within the share (all
/// blocks' parts on one share); the others go on. A part held keeps its rows' part of the Bloch generator (`held`).
/// Returns what the held parts leave (ρ). The excited rows far from the column energies (a centred tier before) are these.
#[allow(clippy::too_many_arguments)]
fn freeze(near: &mut [Near], cache: &Cache, h: &[C64], c: &[C64], whole: &[f64], order: &mut Vec<(f64, usize, usize)>, share: f64, n: usize, work: &mut f64) -> f64 {
    let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
    let mut most = 0.0_f64;
    for j in (0..n).filter(|&j| whole[j] > floor) {
        let a2: f64 = (0..n).map(|i| (h[i * n + j] - if i == j { c[j] } else { C64::ZERO }).norm_sqr()).sum();
        most = most.max((a2 / whole[j]).sqrt());
    }
    order.clear();
    for (x, nb) in near.iter_mut().enumerate() {
        let basis = &cache.near[x][cache.at_near[x].unwrap()];
        let s = nb.s;
        // The exact copies couple every part of their block through the P layer: the reading bounds a part's own next
        // order, not the copies' through it, so such a block goes on whole.
        if nb.nd > 0 {
            continue;
        }
        for (pi, (idx, at, _)) in basis.f.parts.iter().enumerate() {
            let mut q = 0.0;
            for (sp, &wl) in nb.shifts.iter().enumerate() {
                let mut sum = 0.0;
                for i in *at..at + idx.len() {
                    let gap = (0..n).map(|j| (c[j] - C64::real(wl) - nb.kappa[i]).norm_sqr()).fold(f64::INFINITY, f64::min);
                    let row: f64 = nb.z[sp * s * n + i * n..sp * s * n + i * n + n].iter().map(|z| z.norm_sqr()).sum();
                    sum += row / gap;
                }
                q += sum.sqrt();
            }
            nb.q[pi] = q;
            order.push((q, x, pi));
        }
        *work += (nb.shifts.len() * s * n) as f64 / 2.0;
    }
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let limit = unrate(share);
    let mut sum = 0.0;
    for &(q, x, pi) in order.iter() {
        if (sum + q * q).sqrt() * most > limit {
            break;
        }
        sum += q * q;
        let nb = &mut near[x];
        let (idx, at, _) = &cache.near[x][cache.at_near[x].unwrap()].f.parts[pi];
        for i in *at..at + idx.len() {
            nb.active[i] = false;
            for p in 0..n {
                let v = nb.vtot[i * n + p].conj();
                if v != C64::ZERO {
                    crate::linalg::mul_add(false, &mut nb.held[p * n..p * n + n], v, &nb.ztot[i * n..i * n + n]);
                }
            }
        }
        *work += (idx.len() * n * n) as f64;
    }
    sum.sqrt() * most
}

/// A near block's copies' rows T_C = Σ_a e_a* (R₀U_a†V) Z into `nb.tc` (module documentation, 7).
fn copy_rows(nb: &mut Near, basis: &NearBasis, e: &[Vec<C64>], n: usize, work: &mut f64) {
    let Near { copies, groups, ztot, tc, s, .. } = nb;
    copies_of(copies, groups, basis, e, ztot, tc, *s, n, work);
}

/// The copies' rows T_C = Σ_a e_a* (R₀U_a†V) Z of a response Z (s × n) into `out` (nc × n).
#[allow(clippy::too_many_arguments)]
fn copies_of(copies: &[(usize, usize, usize)], groups: &[(usize, Vec<usize>)], basis: &NearBasis, e: &[Vec<C64>], z: &[C64], out: &mut [C64], s: usize, n: usize, work: &mut f64) {
    out.fill(C64::ZERO);
    for (cv, &(g, k, gi)) in copies.iter().enumerate() {
        let ek = e[g][k].conj();
        let yc = &basis.copies[cv];
        let rows = &groups[gi].1;
        for (a2, &row) in rows.iter().enumerate() {
            let out = &mut out[row * n..row * n + n];
            for i in 0..s {
                let y = ek * yc[a2 * s + i];
                if y == C64::ZERO {
                    continue;
                }
                crate::linalg::mul_add(false, out, y, &z[i * n..i * n + n]);
            }
        }
        *work += (rows.len() * s * n) as f64;
    }
}

/// The adjoint of `copies_of`: given T̄_C (nc × n), Z̄ += Σ_a e_a (R₀U_a†V)† T̄_C and ḡ_e += (Σ T̄_C* ∘ (R₀U_a†V)Z)* per copy
/// vector (Re ḡ de).
#[allow(clippy::too_many_arguments)]
fn copies_back(copies: &[(usize, usize, usize)], groups: &[(usize, Vec<usize>)], basis: &NearBasis, e: &[Vec<C64>], z: &[C64], tbar: &[C64], zbar: &mut [C64], ge: &mut [Buf], s: usize, n: usize, work: &mut f64) {
    for (cv, &(g, k, gi)) in copies.iter().enumerate() {
        let ek = e[g][k];
        let yc = &basis.copies[cv];
        let rows = &groups[gi].1;
        let mut inner = C64::ZERO;
        for (a2, &row) in rows.iter().enumerate() {
            let tb = &tbar[row * n..row * n + n];
            for i in 0..s {
                let y = yc[a2 * s + i];
                if y == C64::ZERO {
                    continue;
                }
                let zi = &z[i * n..i * n + n];
                inner += y * tb.iter().zip(zi).map(|(t, x)| t.conj() * *x).sum::<C64>();
                crate::linalg::mul_add(false, &mut zbar[i * n..i * n + n], ek * y.conj(), tb);
            }
        }
        ge[g][k] += inner.conj();
        *work += (2 * rows.len() * s * n) as f64;
    }
}

/// The near block's compression of a response Z (s × n) with Ṽ (s × n) into `a` (n × n): Ṽ†Z + Z†Ṽ + Z†K̂Z, K̂ the kept
/// Schur form with the step's diagonal κ (module documentation, 7); `y` a buffer (s × n).
#[allow(clippy::too_many_arguments)]
fn compress(kappa: &[C64], t: &Mat, ends: &[usize], z: &[C64], v: &[C64], y: &mut [C64], a: &mut [C64], s: usize, n: usize, work: &mut f64) {
    for i in 0..s {
        for j in 0..n {
            y[i * n + j] = kappa[i] * z[i * n + j];
        }
        for l in i + 1..ends[i] {
            crate::linalg::mul_add(false, &mut y[i * n..i * n + n], t[(i, l)], &z[l * n..l * n + n]);
        }
        *work += ((ends[i] - i) * n) as f64;
    }
    for k in 0..s {
        for p in 0..n {
            let zp = z[k * n + p].conj();
            let vp = v[k * n + p].conj();
            if zp != C64::ZERO {
                crate::linalg::mul_add(false, &mut a[p * n..p * n + n], zp, &y[k * n..k * n + n]);
                crate::linalg::mul_add(false, &mut a[p * n..p * n + n], zp, &v[k * n..k * n + n]);
            }
            if vp != C64::ZERO {
                crate::linalg::mul_add(false, &mut a[p * n..p * n + n], vp, &z[k * n..k * n + n]);
            }
        }
    }
    *work += (3 * s * n * n) as f64;
}

/// The adjoint of `compress` for a cotangent Ā (n × n) of A: Z̄ += Ṽ(Ā + Ā†) + K̂ZĀ† + K̂†ZĀ, Ṽ̄ += Z(Ā + Ā†), K̃̄ += ZĀZ†
/// on the parts' entries (`part` per row), `y` = K̂Z (as `compress` left it), `x` a buffer (s × n).
#[allow(clippy::too_many_arguments)]
fn compress_back(kappa: &[C64], t: &Mat, ends: &[usize], part: &[usize], z: &[C64], v: &[C64], y: &[C64], abar: &[C64], zbar: &mut [C64], vbar: &mut [C64], kbar: &mut [C64], x: &mut [C64], s: usize, n: usize, work: &mut f64) {
    // S = Ā + Ā†.
    let sym: Vec<C64> = (0..n * n).map(|i| abar[i] + abar[(i % n) * n + i / n].conj()).collect();
    // Z̄ += Ṽ S + Y Ā† ; Ṽ̄ += Z S; x = Z Ā.
    for i in 0..s {
        for p in 0..n {
            let (vp, zp, yp) = (v[i * n + p], z[i * n + p], y[i * n + p]);
            for j in 0..n {
                let sp = sym[p * n + j];
                zbar[i * n + j] += vp * sp + yp * abar[j * n + p].conj();
                vbar[i * n + j] += zp * sp;
                if p == 0 {
                    x[i * n + j] = C64::ZERO;
                }
                x[i * n + j] += zp * abar[p * n + j];
            }
        }
    }
    // Z̄ += K̂† (Z Ā): K̂† = conj(κ) on the diagonal and the strict lower part T†.
    for i in 0..s {
        for j in 0..n {
            zbar[i * n + j] += kappa[i].conj() * x[i * n + j];
        }
        for l in i + 1..ends[i] {
            let c = t[(i, l)].conj();
            for j in 0..n {
                zbar[l * n + j] += c * x[i * n + j];
            }
        }
    }
    // K̃̄ += (Z Ā) Z† on the parts.
    for i in 0..s {
        for m in (0..s).filter(|&m| part[m] == part[i]) {
            kbar[i * s + m] += (0..n).map(|j| x[i * n + j] * z[m * n + j].conj()).sum::<C64>();
        }
    }
    *work += (4 * s * n * n + s * s * n) as f64;
}

/// The exact copies' K̂_ED = K̂_ED,0 + Σ_x e_x Y_x, K̂_DE = K̂_DE,0 + Σ_x e_x* Y_x† (Y_x the coupling vectors) at the light e.
fn copy_couplings(nb: &Near, basis: &NearBasis, e: &[Vec<C64>], ked: &mut [C64], kde: &mut [C64]) {
    let (s, nd) = (nb.s, nb.nd);
    ked.copy_from_slice(&basis.ked0);
    kde.copy_from_slice(&basis.kde0);
    for (x, &(g, k)) in nb.exact.iter().enumerate() {
        let ek = e[g][k];
        for i in 0..s {
            for d in 0..nd {
                let y = basis.couple[x][i * nd + d];
                ked[i * nd + d] += ek * y;
                kde[d * s + i] += (ek * y).conj();
            }
        }
    }
}

/// The exact copies' part of a block's solve for one split (module documentation, # The exact copies): with the excited
/// rows' z_E⁰ = A_j⁻¹ r_E already in `ze` (every column j, A_j = c_j − w − K̂_EE triangular in the kept basis, its inverse
/// diagonal `inv`), per column P_j = A_j⁻¹K̂_ED, C_j = (c_j − w) − K̂_DD − K̂_DE P_j, z_D = C_j⁻¹(r_D + K̂_DE z_E⁰),
/// z_E = z_E⁰ + P_j z_D -- the whole block's solve, by Woodbury. With `stored` (P_j, C_j⁻¹ of the forward) they are
/// taken as they are (a tangent's solve on the same matrices); otherwise made and kept there. False where a C_j is
/// singular.
#[allow(clippy::too_many_arguments)]
fn copies_solve(s: usize, nd: usize, ked: &[C64], kde: &[C64], kdd: &[C64], basis: &NearBasis, c: &[C64], w: f64, inv: &[C64], rd: &[C64], ze: &mut [C64], zd: &mut [C64], pj: &mut [C64], cinv: &mut [C64], stored: bool, n: usize, work: &mut f64) -> bool {
    let (t, ends) = (&basis.f.t, &basis.f.ends);
    let mut p = vec![C64::ZERO; s];
    for j in 0..n {
        let (pj, cinv) = (&mut pj[j * nd * s..(j + 1) * nd * s], &mut cinv[j * nd * nd..(j + 1) * nd * nd]);
        if !stored {
            for d in 0..nd {
                for i in (0..s).rev() {
                    let mut acc = ked[i * nd + d];
                    for l in i + 1..ends[i] {
                        acc += t[(i, l)] * p[l];
                    }
                    p[i] = acc * inv[i * n + j];
                }
                pj[d * s..(d + 1) * s].copy_from_slice(&p);
            }
            let cm = Mat::from_fn(nd, nd, |a, b| {
                let own = if a == b { c[j] - C64::real(w) } else { C64::ZERO } - kdd[a * nd + b];
                own - (0..s).map(|i| kde[a * s + i] * pj[b * s + i]).sum::<C64>()
            });
            let Some(ci) = inverse(&cm, work) else { return false };
            cinv.copy_from_slice(&ci.data);
            *work += (nd * (s * s / 2 + s) + nd * nd * s) as f64;
        }
        let rhs: Vec<C64> = (0..nd).map(|a| rd[a * n + j] + (0..s).map(|i| kde[a * s + i] * ze[i * n + j]).sum::<C64>()).collect();
        for a in 0..nd {
            let x: C64 = (0..nd).map(|b| cinv[a * nd + b] * rhs[b]).sum();
            zd[a * n + j] = x;
            for i in 0..s {
                ze[i * n + j] += pj[a * s + i] * x;
            }
        }
        *work += (2 * nd * s + nd * nd) as f64;
    }
    true
}

/// The adjoint of `copies_solve` with the excited rows' adjoint solve after it: given the cotangents Z̄_E (`ze`, ne × n)
/// and Z̄_D (`zd`, nd × n) of one split's solve, Λ_D = C_j^{−†}(Z̄_D + P_j†Z̄_E) per column into `lamd`, and Z̄_E += K̂_DE†Λ_D
/// (in place: the excited rows' adjoint triangular solve then gives Λ_E).
#[allow(clippy::too_many_arguments)]
fn copies_back_solve(s: usize, nd: usize, kde: &[C64], ze: &mut [C64], zd: &[C64], pj: &[C64], cinv: &[C64], lamd: &mut [C64], n: usize, work: &mut f64) {
    for j in 0..n {
        let (pj, cinv) = (&pj[j * nd * s..(j + 1) * nd * s], &cinv[j * nd * nd..(j + 1) * nd * nd]);
        let rhs: Vec<C64> = (0..nd).map(|a| zd[a * n + j] + (0..s).map(|i| pj[a * s + i].conj() * ze[i * n + j]).sum::<C64>()).collect();
        for a in 0..nd {
            lamd[a * n + j] = (0..nd).map(|b| cinv[b * nd + a].conj() * rhs[b]).sum();
        }
        for i in 0..s {
            ze[i * n + j] += (0..nd).map(|a| kde[a * s + i].conj() * lamd[a * n + j]).sum::<C64>();
        }
        *work += (2 * nd * s + nd * nd) as f64;
    }
}

/// The exact copies' part of a compression: Z_E†K̂_ED Z_D + Z_D†K̂_DE Z_E + Z_D†K̂_DD Z_D into `a`, Z_D†Z_D into `metric`.
#[allow(clippy::too_many_arguments)]
fn copies_compress(s: usize, nd: usize, ked: &[C64], kde: &[C64], kdd: &[C64], ze: &[C64], zd: &[C64], a: &mut [C64], metric: Option<&mut [C64]>, n: usize, work: &mut f64) {
    // y_E = K̂_ED Z_D (s × n), y_D = K̂_DE Z_E + K̂_DD Z_D (nd × n).
    let mut ye = vec![C64::ZERO; s * n];
    let mut yd = vec![C64::ZERO; nd * n];
    for i in 0..s {
        for d in 0..nd {
            crate::linalg::mul_add(false, &mut ye[i * n..i * n + n], ked[i * nd + d], &zd[d * n..d * n + n]);
        }
    }
    for d in 0..nd {
        for i in 0..s {
            crate::linalg::mul_add(false, &mut yd[d * n..d * n + n], kde[d * s + i], &ze[i * n..i * n + n]);
        }
        for e in 0..nd {
            crate::linalg::mul_add(false, &mut yd[d * n..d * n + n], kdd[d * nd + e], &zd[e * n..e * n + n]);
        }
    }
    for i in 0..s {
        for p in 0..n {
            let zp = ze[i * n + p].conj();
            if zp != C64::ZERO {
                crate::linalg::mul_add(false, &mut a[p * n..p * n + n], zp, &ye[i * n..i * n + n]);
            }
        }
    }
    let mut metric = metric;
    for d in 0..nd {
        for p in 0..n {
            let zp = zd[d * n + p].conj();
            crate::linalg::mul_add(false, &mut a[p * n..p * n + n], zp, &yd[d * n..d * n + n]);
            if let Some(m) = metric.as_deref_mut() {
                crate::linalg::mul_add(false, &mut m[p * n..p * n + n], zp, &zd[d * n..d * n + n]);
            }
        }
    }
    *work += (2 * s * nd * n + nd * nd * n + (s + 2 * nd) * n * n) as f64;
}

/// The adjoint of `copies_compress` for the cotangent Ā of A (and X̄ of the metric, `xbar`, where given): Z̄_E, Z̄_D, and
/// K̂_ED's, K̂_DE's, K̂_DD's cotangents (Z_E Ā Z_D†, Z_D Ā Z_E†, Z_D Ā Z_D†).
#[allow(clippy::too_many_arguments)]
fn copies_compress_back(s: usize, nd: usize, ked: &[C64], kde: &[C64], kdd: &[C64], ze: &[C64], zd: &[C64], abar: &[C64], xbar: Option<&[C64]>, zebar: &mut [C64], zdbar: &mut [C64], kedbar: &mut [C64], kdebar: &mut [C64], kddbar: &mut [C64], n: usize, work: &mut f64) {
    let adj: Vec<C64> = (0..n * n).map(|i| abar[(i % n) * n + i / n].conj()).collect();
    let prod = |z: &[C64], m: &[C64], rows: usize| -> Vec<C64> {
        let mut out = vec![C64::ZERO; rows * n];
        for i in 0..rows {
            for p in 0..n {
                let x = z[i * n + p];
                if x != C64::ZERO {
                    crate::linalg::mul_add(false, &mut out[i * n..i * n + n], x, &m[p * n..p * n + n]);
                }
            }
        }
        out
    };
    // Z Ā and Z Ā† of both.
    let (zea, zead, zda, zdad) = (prod(ze, abar, s), prod(ze, &adj, s), prod(zd, abar, nd), prod(zd, &adj, nd));
    for i in 0..s {
        for d in 0..nd {
            // Z̄_E += K̂_ED (Z_D Ā†) + K̂_DE† (Z_D Ā).
            crate::linalg::mul_add(false, &mut zebar[i * n..i * n + n], ked[i * nd + d], &zdad[d * n..d * n + n]);
            crate::linalg::mul_add(false, &mut zebar[i * n..i * n + n], kde[d * s + i].conj(), &zda[d * n..d * n + n]);
            // Z̄_D += K̂_DE (Z_E Ā†) + K̂_ED† (Z_E Ā).
            crate::linalg::mul_add(false, &mut zdbar[d * n..d * n + n], kde[d * s + i], &zead[i * n..i * n + n]);
            crate::linalg::mul_add(false, &mut zdbar[d * n..d * n + n], ked[i * nd + d].conj(), &zea[i * n..i * n + n]);
            kedbar[i * nd + d] += (0..n).map(|j| zea[i * n + j] * zd[d * n + j].conj()).sum::<C64>();
            kdebar[d * s + i] += (0..n).map(|j| zda[d * n + j] * ze[i * n + j].conj()).sum::<C64>();
        }
    }
    for d in 0..nd {
        for e in 0..nd {
            crate::linalg::mul_add(false, &mut zdbar[d * n..d * n + n], kdd[d * nd + e], &zdad[e * n..e * n + n]);
            crate::linalg::mul_add(false, &mut zdbar[d * n..d * n + n], kdd[e * nd + d].conj(), &zda[e * n..e * n + n]);
            kddbar[d * nd + e] += (0..n).map(|j| zda[d * n + j] * zd[e * n + j].conj()).sum::<C64>();
        }
        if let Some(x) = xbar {
            for p in 0..n {
                let zp = zd[d * n + p] * 2.0;
                crate::linalg::mul_add(false, &mut zdbar[d * n..d * n + n], zp, &x[p * n..p * n + n]);
            }
        }
    }
    *work += (4 * (s + nd) * n * n + 6 * s * nd * n + 3 * nd * nd * n) as f64;
}

/// The light's cotangents of the exact copies' couplings: K̂_ED's and K̂_DE's onto the coupling vectors' coefficients
/// (ḡ_e, Re ḡ de), K̂_DD's onto the pairs' (k_t).
fn copies_light(nb: &Near, basis: &NearBasis, kedbar: &[C64], kdebar: &[C64], kddbar: &[C64], ge: &mut [Buf], kt: Option<&mut [C64]>) {
    let (s, nd) = (nb.s, nb.nd);
    for (x, &(g, k)) in nb.exact.iter().enumerate() {
        let y = &basis.couple[x];
        let mut v = C64::ZERO;
        for i in 0..s {
            for d in 0..nd {
                v += kedbar[i * nd + d].conj() * y[i * nd + d] + kdebar[d * s + i] * y[i * nd + d];
            }
        }
        ge[g][k] += v;
    }
    if let Some(kt) = kt {
        for (ti, &tt) in nb.terms.iter().enumerate() {
            kt[tt] += kddbar.iter().zip(&basis.ddterms[ti]).map(|(a, b)| a.conj() * *b).sum::<C64>();
        }
    }
}

/// S's Lyapunov map solved, S Y + Y S = C (C Hermitian, S = (1 + X)^{1/2} = (1 + X)S⁻¹), into `y`, with `e` = S/σ − 1 made
/// here: the adjoint's and the tangent's one solve of S⁻¹'s Fréchet derivative (`inverse_root_back`, the photons'). With
/// σ = (1 + √(1 + ‖X‖_F))/2 the fixed point Y ← (C/σ − EY − YE)/2 contracts by ‖E‖ ≤ (s_max − 1)/(s_max + 1) < 1 for any
/// metric (S's eigenvalues in [1, √(1 + ‖X‖_F)]), to the rounding as `inverse_root` stops: each iterate Hermitian,
/// YE = (EY)†, one product a step. False where a step does not shrink the last one's move (the rounding's).
#[allow(clippy::too_many_arguments)]
fn lyapunov(x: &[C64], normal: &[C64], c: &[C64], n: usize, y: &mut [C64], e: &mut [C64], p: &mut [C64], work: &mut f64) -> bool {
    let sigma = 0.5 * (1.0 + (1.0 + x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()).sqrt());
    // E = (X N + N)/σ − 1.
    multiply(x, normal, e, n);
    for i in 0..n * n {
        e[i] = (e[i] + normal[i]) / sigma - if i % (n + 1) == 0 { C64::ONE } else { C64::ZERO };
    }
    for (o, b) in y.iter_mut().zip(c) {
        *o = *b * (0.5 / sigma);
    }
    *work += (n * n * n) as f64;
    let floor = crate::units::rounding(4 * n);
    let mut last = f64::INFINITY;
    loop {
        multiply(e, y, p, n);
        let (mut moved, mut size) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            for j in 0..n {
                let next = (c[i * n + j] / sigma - p[i * n + j] - p[j * n + i].conj()) * 0.5;
                moved += (next - y[i * n + j]).norm_sqr();
                size += next.norm_sqr();
                y[i * n + j] = next;
            }
        }
        *work += (n * n * n) as f64;
        let (moved, size) = (moved.sqrt(), size.sqrt());
        if moved <= floor * size || size == 0.0 {
            return true;
        }
        if moved >= last {
            return false;
        }
        last = moved;
    }
}

/// The adjoint of S⁻¹ = (1 + X)^{−1/2} (`inverse_root`): given N̄ on N = S⁻¹, X̄ on X into `out`. With S² = 1 + X,
/// dN = −N dS N and S dS + dS S = dX, so S̄ = −N N̄_h N (N̄_h its Hermitian part: N and dN are Hermitian) and X̄ solves
/// S X̄ + X̄ S = S̄ (the map is self-adjoint; `lyapunov`).
#[allow(clippy::too_many_arguments)]
fn inverse_root_back(x: &[C64], normal: &[C64], nbar: &[C64], n: usize, out: &mut [C64], p: &mut [C64], q: &mut [C64], r: &mut [C64], work: &mut f64) -> bool {
    for i in 0..n {
        for j in 0..n {
            r[i * n + j] = (nbar[i * n + j] + nbar[j * n + i].conj()) * 0.5;
        }
    }
    multiply(normal, r, p, n);
    multiply(p, normal, q, n);
    q.iter_mut().for_each(|z| *z = -*z);
    *work += (2 * n * n * n) as f64;
    lyapunov(x, normal, q, n, out, r, p, work)
}

/// out = a b (n × n, row by row; linalg.rs `fused_product`).
fn multiply(a: &[C64], b: &[C64], out: &mut [C64], n: usize) {
    crate::linalg::fused_product(a, b, (n, n, n), out);
}

/// S⁻¹ = (1 + X)^{−1/2} of the metric less one X (Hermitian, X ≥ 0, n × n) into `y`, by the Newton–Schulz iteration
/// Y ← Y(3 − (1 + X)Y²)/2 from its series' second order Y₀ = 1 − X/2 + 3X²/8, to the rounding -- the one algorithm for it.
/// Every iterate is a polynomial in X, so the iteration is scalar on each eigenvalue x ≥ 0 of X: with u = y√(1 + x) it is
/// u ← u(3 − u²)/2, which goes to 1 from every u₀ in (0, √3) (on (0, 1] it rises to 1; (1, √3) it maps into (0, 1)), and
/// u₀(x) = (1 − x/2 + 3x²/8)√(1 + x) lies in [1, 1.24] for 0 ≤ x ≤ 1 (u₀ reaches √3 only at x ≈ 1.5). The metric holds the
/// exact copies' populations beside the excitation (each below a half of the place's own, DESIGN §3.1), so X's largest
/// eigenvalue may pass one where its excitation does not: there (‖X‖_F > 1, a bound of it) the start is the scalar
/// (1 + ‖X‖_F)^{−1/2}, every u₀ in (0, 1]. From either start it converges, quadratically near one: a step's error e goes
/// to −(3/2)e² − e³/2 (Y = M^{−1/2}(1 + e)), and a step that moves Y by m leaves (3/2)m² -- it stops where that is within
/// γ_{4n} of Y's size (the next move's reading). False where a step does not shrink the last one's move (no fold there:
/// the caller returns none). Products alone (3n³ a step).
fn inverse_root(x: &[C64], n: usize, y: &mut [C64], p: &mut [C64], q: &mut [C64], r: &mut [C64], work: &mut f64) -> bool {
    let size = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if size <= 1.0 {
        multiply(x, x, p, n);
        for i in 0..n {
            for j in 0..n {
                y[i * n + j] = if i == j { C64::ONE } else { C64::ZERO } - x[i * n + j] * 0.5 + p[i * n + j] * 0.375;
            }
        }
        *work += (n * n * n) as f64;
    } else {
        let scale = 1.0 / (1.0 + size).sqrt();
        for i in 0..n * n {
            y[i] = if i % (n + 1) == 0 { C64::real(scale) } else { C64::ZERO };
        }
    }
    let floor = crate::units::rounding(4 * n);
    let mut last = f64::INFINITY;
    loop {
        multiply(y, y, p, n);
        multiply(x, p, q, n);
        for (a, b) in q.iter_mut().zip(p.iter()) {
            *a += *b;
        }
        multiply(y, q, r, n);
        // The move and Y in the Frobenius norm (at least the 2-norm the error's recursion holds in).
        let (mut moved, mut size) = (0.0_f64, 0.0_f64);
        for (a, b) in y.iter_mut().zip(r.iter()) {
            let next = *a * 1.5 - *b * 0.5;
            moved += (next - *a).norm_sqr();
            size += next.norm_sqr();
            *a = next;
        }
        let (moved, size) = (moved.sqrt(), size.sqrt());
        *work += (3 * n * n * n) as f64;
        if 1.5 * moved * moved <= floor * size {
            return true;
        }
        if moved >= last {
            return false;
        }
        last = moved;
    }
}

impl Fold {
    /// The prototype's layout for this fold (module documentation). Refused where a block holds what the prototype does
    /// not: explicit states, a static source, an exact copy, centred excited rows beside solved ones (the choice's to make
    /// solved), or a copy emitting into P.
    pub(crate) fn prototype_workspace(&self) -> Result<Workspace, String> {
        let n = self.size();
        let ng = self.ground.len();
        if n != ng {
            return Err("原型不含显式态".into());
        }
        let mut pairs: Vec<Pair> = vec![];
        for (w, f) in self.light.frequencies.iter().enumerate() {
            for k in 0..f.rank() {
                for kk in 0..f.rank() {
                    pairs.push((w, k, kk));
                }
            }
        }
        // A pair's contraction of a frequency's forms: Σ_pq u_kp* u_k'q O_pq on the entries `keep` takes, by `place`.
        let contract = |pair: Pair, keep: &dyn Fn(usize, usize) -> Option<(usize, usize)>, rows: usize, cols: usize| -> Mat {
            let (w, k, kk) = pair;
            let u = &self.light.frequencies[w].basis;
            let mut m = Mat::zeros(rows, cols);
            for (p, q, entries) in &self.entries[w] {
                let cu = u[k][*p].conj() * u[kk][*q];
                for &(i, j, v) in entries {
                    if let Some((a, b)) = keep(i, j) {
                        m[(a, b)] += cu * v;
                    }
                }
            }
            m
        };
        let pp: Vec<Vec<(usize, C64)>> = pairs
            .iter()
            .map(|&pair| {
                let m = contract(
                    pair,
                    &|i, j| {
                        let (p, q) = (self.ground.iter().position(|&g| g == i)?, self.ground.iter().position(|&g| g == j)?);
                        (self.frames[p] == self.frames[q]).then_some((p, q))
                    },
                    n,
                    n,
                );
                (0..n * n).filter(|&x| m.data[x] != C64::ZERO).map(|x| (x, m.data[x])).collect()
            })
            .collect();
        let mut base = vec![C64::ZERO; n * n];
        for p in 0..n {
            for q in (0..n).filter(|&q| self.frames[q] == self.frames[p]) {
                base[p * n + q] = self.base_ground[(self.ground[p], self.ground[q])] - if p == q { C64::real(self.frame_values[p]) } else { C64::ZERO };
            }
        }
        let (mut near, mut far) = (vec![], vec![]);
        for (o, b) in self.blocks.iter().enumerate() {
            let m = b.rows.len();
            if m == 0 {
                continue;
            }
            if !b.explicit.is_empty() {
                return Err("原型不含显式态".into());
            }
            let s = b.solved;
            // The exact copies are the solved rows after the excited ones (`Fold::build`'s order).
            let ne = b.rows[..s].iter().filter(|r| matches!(r, Row::Excited(_))).count();
            if b.rows[..ne].iter().any(|r| matches!(r, Row::Copy { .. })) {
                return Err(format!("块 {o}：精确拷贝不在激发行之后"));
            }
            let dstates: Vec<(usize, usize)> = b.rows[ne..s].iter().map(|r| if let Row::Copy { state, component } = *r { (state, component) } else { unreachable!() }).collect();
            let centre: Vec<usize> = b.sources.iter().map(|x| x.0).collect();
            let mut splits = vec![];
            let mut vectors = vec![];
            for (c, comp) in self.components.iter().enumerate() {
                if b.sources.iter().any(|x| x.1 == c) {
                    splits.extend((0..comp.beams.len()).map(|j| (c, j)));
                    vectors.extend((0..comp.basis.len()).map(|k| (c, k)));
                }
            }
            // The static split: its sources on the copies (a ground state at a copy's frame), none here.
            for r in &b.rows {
                if let Row::Copy { component, .. } = *r {
                    let omega = b.offset.plus(-1, &self.components[component].beat);
                    if (0..ng).any(|p| self.frames[p] == omega) {
                        return Err(format!("块 {o}：原型不含落在 P 帧上的拷贝"));
                    }
                }
            }
            let terms_of = |keep: &dyn Fn(usize, usize) -> Option<(usize, usize)>, size: usize| -> Vec<usize> {
                (0..pairs.len()).filter(|&t| contract(pairs[t], keep, size, size).frobenius() > 0.0).collect()
            };
            let row_of = |x: usize| b.rows.iter().position(|r| *r == Row::Excited(x));
            if s > 0 {
                if b.rows[s..].iter().any(|r| matches!(r, Row::Excited(_))) {
                    return Err(format!("块 {o}：解的块里有居中的激发行（先把它们改为解）"));
                }
                let nc = m - s;
                let mut groups: Vec<(usize, Vec<usize>)> = vec![];
                for (a, r) in b.rows[s..].iter().enumerate() {
                    if let Row::Copy { component, .. } = *r {
                        match groups.iter_mut().find(|g| g.0 == component) {
                            Some(g) => g.1.push(a),
                            None => groups.push((component, vec![a])),
                        }
                    }
                }
                let copies: Vec<(usize, usize, usize)> = groups.iter().enumerate().flat_map(|(gi, (g, _))| (0..self.components[*g].basis.len()).map(move |k| (*g, k, gi))).collect();
                // A term: a pair whose forms reach the solved rows, or whose frequency a copy component's is (its dressing).
                let mut terms = terms_of(&|i, j| if i >= GROUND && j >= GROUND { Some((row_of(i - GROUND).filter(|&a| a < s)?, row_of(j - GROUND).filter(|&a| a < s)?)) } else { None }, s);
                for (t, &(w, _, _)) in pairs.iter().enumerate() {
                    if groups.iter().any(|(g, _)| self.components[*g].part.frequency == w) && !terms.contains(&t) {
                        terms.push(t);
                    }
                }
                // The exact copies' own forms (H_GG among the copies of one component).
                for (t, &pair) in pairs.iter().enumerate() {
                    let dd = contract(pair, &|i, j| (dstates.iter().any(|d| d.0 == i) && dstates.iter().any(|d| d.0 == j)).then_some((0, 0)), 1, 1);
                    if dd.frobenius() > 0.0 && !terms.contains(&t) {
                        terms.push(t);
                    }
                }
                terms.sort();
                let nd = s - ne;
                let mut exact: Vec<(usize, usize)> = vec![];
                for &(_, g) in &dstates {
                    for k in 0..self.components[g].basis.len() {
                        if !exact.contains(&(g, k)) {
                            exact.push((g, k));
                        }
                    }
                }
                let s = ne;
                let z = |x: usize| vec![C64::ZERO; x];
                let sp = splits.len();
                near.push(Near {
                    nd,
                    dstates: dstates.clone(),
                    exact,
                    ked: z(s * nd),
                    kde: z(nd * s),
                    kdd: z(nd * nd),
                    zd: z(sp * nd * n),
                    zdtot: z(nd * n),
                    zds: vec![],
                    zdtots: vec![],
                    pjs: vec![],
                    cinvs: vec![],
                    zdbar: z(sp * nd * n),
                    zdtbar: z(nd * n),
                    kedbar: z(s * nd),
                    kdebar: z(nd * s),
                    kddbar: z(nd * nd),
                    zdnode: [z(nd * n), z(nd * n)],
                    kednode: [z(s * nd), z(s * nd)],
                    kdenode: [z(nd * s), z(nd * s)],
                    o,
                    s,
                    nc,
                    value: b.value,
                    centre,
                    terms,
                    delta: vec![],
                    kappa: z(s),
                    off: vec![0.0; s * s],
                    rho: vec![0.0; s],
                    shifts: vec![0.0; splits.len()],
                    vt: z(splits.len() * s * n),
                    vtot: z(s * n),
                    z: z(splits.len() * s * n),
                    ztot: z(s * n),
                    zold: z(s * n),
                    rhs: z(s * n),
                    bloch: z(n * n),
                    tc: z(nc * n),
                    y: z(s * n),
                    err: vec![0.0; s * n],
                    norms: vec![0.0; s + 1],
                    active: vec![true; s],
                    held: z(n * n),
                    q: vec![0.0; s + 1],
                    reading: 0.0,
                    invs: vec![],
                    zs: vec![],
                    ztots: vec![],
                    zbar: z(splits.len() * s * n),
                    bucket: z(splits.len() * s * n),
                    lam: z(s * n),
                    ztbar: z(s * n),
                    kbar: z(s * s),
                    vbar: z(s * n),
                    fbar: z(splits.len()),
                    ratio: [z(splits.len()), z(splits.len())],
                    znode: [z(s * n), z(s * n)],
                    vnode: [z(s * n), z(s * n)],
                    tcnode: [z(nc * n), z(nc * n)],
                    copied_in: false,
                    splits,
                    vectors,
                    groups,
                    copies,
                });
            } else {
                if b.rows.iter().any(|r| matches!(r, Row::Copy { .. })) {
                    return Err(format!("块 {o}：全居中的块里有拷贝"));
                }
                let terms = terms_of(&|i, j| if i >= GROUND && j >= GROUND { Some((row_of(i - GROUND)?, row_of(j - GROUND)?)) } else { None }, m);
                let v = vectors.len();
                far.push(Far { o, m, centre, splits, terms, delta: vec![], products: vec![C64::ZERO; v * v], hf: vec![C64::ZERO; n * n], mf: vec![C64::ZERO; n * n], reading: 0.0, centring: 0.0, vectors });
            }
        }
        for x in near.iter_mut() {
            x.delta = vec![C64::ZERO; x.terms.len()];
        }
        for x in far.iter_mut() {
            x.delta = vec![C64::ZERO; x.terms.len()];
        }
        let z = |r: usize, c: usize| Mat::zeros(r, c);
        let near_parts: usize = near.iter().map(|x| x.s).sum();
        let placed = Placed {
            amplitudes: vec![Amplitude { value: C64::ZERO, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] }; self.light.beams.len()],
            eps: vec![[C64::ZERO; 3]; self.light.frequencies.len()],
            coef: vec![C64::ZERO; pairs.len()],
            f: self.components.iter().map(|c| vec![C64::ZERO; c.beams.len()]).collect(),
            w: self.components.iter().map(|c| vec![0.0; c.beams.len()]).collect(),
            e: self.components.iter().map(|c| vec![C64::ZERO; c.basis.len()]).collect(),
            hpp: vec![C64::ZERO; n * n],
        };
        let proj = self.light.frequencies.iter().map(|f| f.fields.iter().flat_map(|e| f.basis.iter().map(move |u| (0..3).map(|q| u[q].conj() * e[q]).sum())).collect()).collect();
        let node = || Node { at: placed.clone(), a: z(n, n), generator: z(n, n), hf: far.iter().map(|_| vec![C64::ZERO; n * n]).collect(), mf: far.iter().map(|_| vec![C64::ZERO; n * n]).collect() };
        let nodes = vec![node(), node()];
        Ok(Workspace {
            n,
            pp,
            base,
            near,
            far,
            mid: placed,
            proj,
            cs: vec![],
            deltas: vec![],
            nodes,
            node_times: [0.0; 2],
            noded: false,
            pairs,
            hbase: vec![C64::ZERO; n * n],
            h: vec![C64::ZERO; n * n],
            delta: vec![C64::ZERO; n * n],
            c: vec![C64::ZERO; n],
            change: vec![0.0; n],
            previous: vec![0.0; n],
            whole: vec![0.0; n],
            eigen: z(n, n),
            vectors: z(n, n),
            temp: z(n, n),
            renew: vec![],
            order: Vec::with_capacity(near_parts),
            step: Step {
                a: z(n, n),
                metric: z(n, n),
                normal: z(n, n),
                generator: z(n, n),
                sweeps: 0,
                series: 0.0,
                reuse: 0.0,
                far: 0.0,
                centring: 0.0,
                copies: 0.0,
                ratio: 0.0,
                made: 0,
                rounds: 0,
                work: [0.0; 10],
                ticks: [0; 10],
                sweep_work: vec![],
                sweep_ticks: vec![],
            },
        })
    }

    /// The light at a place (module documentation, 1), the beams' gradients with it (the force's).
    #[allow(clippy::too_many_arguments)]
    fn proto_place(&self, r: [f64; 3], kappa: [f64; 3], at: &mut Placed, pairs: &[Pair], pp: &[Vec<(usize, C64)>], base: &[C64], proj: &[Vec<C64>], work: &mut f64) {
        for (a, b) in at.amplitudes.iter_mut().zip(&self.light.beams) {
            *a = b.amplitude_to(r, 1);
        }
        for ((eps, f), pr) in at.eps.iter_mut().zip(&self.light.frequencies).zip(proj) {
            let rank = f.rank();
            *eps = [C64::ZERO; 3];
            for (j, &l) in f.beams.iter().enumerate() {
                let a = at.amplitudes[l].value;
                for k in 0..rank {
                    eps[k] += a * pr[j * rank + k];
                }
            }
            *work += (f.beams.len() * rank) as f64;
        }
        for (x, &(w, k, kk)) in at.coef.iter_mut().zip(pairs) {
            *x = at.eps[w][k].conj() * at.eps[w][kk];
        }
        *work += pairs.len() as f64;
        for (c, comp) in self.components.iter().enumerate() {
            for (j, &l) in comp.beams.iter().enumerate() {
                at.f[c][j] = comp.factor(at.amplitudes[l].value);
                let q = &comp.kicks[j];
                at.w[c][j] = self.hbar_over_mass * (q.iter().zip(&kappa).map(|(q, k)| q * k).sum::<f64>() + 0.5 * q.iter().map(|q| q * q).sum::<f64>());
            }
            // e_k = ε_k co-rotating, ε_k* counter-rotating (fold.rs `Component::gamma`).
            let eps = &at.eps[comp.part.frequency];
            for k in 0..comp.basis.len() {
                at.e[c][k] = if comp.part.counter { eps[k].conj() } else { eps[k] };
            }
        }
        at.hpp.copy_from_slice(base);
        for (entries, &x) in pp.iter().zip(&at.coef) {
            for &(i, v) in entries {
                at.hpp[i] += x * v;
            }
            *work += entries.len() as f64;
        }
    }

    /// The light's cotangents at a place onto its beams' amplitudes, contracted with their gradients, per lab axis: given
    /// per component and beam ḡ_f (Re ḡ df of the factor f_l), per component and basis vector ḡ_e (of e_k) and per pair
    /// k_t (of c_t), Σ Re(ḡ ∂_a x) over them all (`Fold::prototype_force`).
    fn contracted<V: AsRef<[C64]>>(&self, at: &Placed, proj: &[Vec<C64>], pairs: &[Pair], gf: &[V], ge: &[V], kt: &[C64]) -> [f64; 3] {
        // Onto ε_k: e_k = ε_k or ε_k*; c_t = ε_k*ε_k'.
        let mut geps: Vec<[C64; 3]> = vec![[C64::ZERO; 3]; self.light.frequencies.len()];
        for (c, comp) in self.components.iter().enumerate() {
            for (k, g) in ge[c].as_ref().iter().enumerate() {
                geps[comp.part.frequency][k] += if comp.part.counter { g.conj() } else { *g };
            }
        }
        for (&(w, k, kk), &g) in pairs.iter().zip(kt) {
            if g == C64::ZERO {
                continue;
            }
            geps[w][k] += (g * at.eps[w][kk]).conj();
            geps[w][kk] += g * at.eps[w][k].conj();
        }
        // Onto the beams: ε_k = Σ_j a_l u_k†E_l, f_l = a_l or a_l*.
        let mut ga = vec![C64::ZERO; self.light.beams.len()];
        for (w, f) in self.light.frequencies.iter().enumerate() {
            let rank = f.rank();
            for (j, &l) in f.beams.iter().enumerate() {
                ga[l] += (0..rank).map(|k| geps[w][k] * proj[w][j * rank + k]).sum::<C64>();
            }
        }
        for (c, comp) in self.components.iter().enumerate() {
            for (j, &l) in comp.beams.iter().enumerate() {
                let g = gf[c].as_ref()[j];
                ga[l] += if comp.part.counter { g.conj() } else { g };
            }
        }
        std::array::from_fn(|a| ga.iter().zip(&at.amplitudes).map(|(g, x)| (*g * x.gradient[a]).re).sum())
    }

    /// A near block's decomposition at the point (module documentation, # What is kept), from the fold's own matrices
    /// there (`Fold::site`, `lay`, `effective`).
    fn near_basis(&self, nb: &Near, ws: &Workspace, site: &Site, work: &mut f64) -> Option<NearBasis> {
        let (n, s, nc, nd) = (ws.n, nb.s, nb.nc, nb.nd);
        let b = &self.blocks[nb.o];
        let laid = self.lay(nb.o, site, work);
        let e = self.effective(nb.o, &laid, work)?;
        // The excited rows decomposed; the exact copies after them in their own coordinates (P layer).
        let (er, dr): (Vec<usize>, Vec<usize>) = ((0..s).collect(), (s..s + nd).collect());
        let f = Factored::of(&e.k.block(&er, &er), work)?;
        let mut part = vec![0; s];
        for (pi, (idx, at, _)) in f.parts.iter().enumerate() {
            for a in 0..idx.len() {
                part[at + a] = pi;
            }
        }
        let parts = f.parts.len().max(1);
        let rows: Vec<usize> = b.rows[..s].iter().map(|r| if let Row::Excited(x) = *r { x } else { unreachable!() }).collect();
        let copy_state = |a: usize| if let Row::Copy { state, .. } = b.rows[s + nd + a] { state } else { unreachable!() };
        let r0 = &e.resolvent;
        // Per copy component: U_k on (S, C_g) and R₀ on (C_g, C_g).
        let u_sc = |g: usize, k: usize, cg: &[usize]| Mat::from_fn(s, cg.len(), |a, c| self.components[g].basis[k][(rows[a], copy_state(cg[c]))]);
        let mut terms = vec![];
        for &t in &nb.terms {
            let (w, k, kk) = ws.pairs[t];
            let u = &self.light.frequencies[w].basis;
            let mut form = Mat::zeros(s, s);
            let mut cc = Mat::zeros(nc, nc);
            for (p, q, entries) in &self.entries[w] {
                let cu = u[k][*p].conj() * u[kk][*q];
                for &(i, j, v) in entries {
                    if i >= GROUND && j >= GROUND {
                        if let (Some(a), Some(c)) = (rows.iter().position(|&x| x == i - GROUND), rows.iter().position(|&x| x == j - GROUND)) {
                            form[(a, c)] += cu * v;
                        }
                    } else if i < GROUND && j < GROUND {
                        // The copies' own rows (one component's copies among themselves).
                        for (_, cg) in &nb.groups {
                            for &x in cg {
                                for &y in cg {
                                    if copy_state(x) == i && copy_state(y) == j {
                                        cc[(x, y)] += cu * v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // The dressing: co-rotating e_k e_kk* = ε_k ε_kk* (the pair (w, kk, k)), counter-rotating ε_k* ε_kk (the pair (w, k, kk)).
            let mut dressing = Mat::zeros(s, s);
            for (g, cg) in &nb.groups {
                let comp = &self.components[*g];
                if comp.part.frequency != w {
                    continue;
                }
                let rg = r0.block(cg, cg);
                let (k1, k2) = if comp.part.counter { (k, kk) } else { (kk, k) };
                let x = u_sc(*g, k1, cg).product(&rg).product(&u_sc(*g, k2, cg).adjoint());
                plus(&mut dressing, &x);
            }
            let mut whole = form.clone();
            plus(&mut whole, &dressing);
            let tw = rotate(&f, &whole, work);
            let td = rotate(&f, &dressing, work);
            terms.push((whole.frobenius(), tw, td, cc.frobenius()));
        }
        let at: Vec<C64> = nb.terms.iter().map(|&t| ws.mid.coef[t]).collect();
        let within: Vec<Vec<C64>> = terms
            .iter()
            .map(|(_, tw, _, _)| Mat::from_fn(s, s, |i, j| if part[i] == part[j] { tw[(i, j)] } else { C64::ZERO }).data)
            .collect();
        let dress: Vec<f64> = terms.iter().map(|x| x.2.frobenius()).collect();
        let es: Vec<usize> = (0..GROUND).collect();
        let excited: Vec<usize> = rows.clone();
        let emit: [Vec<C64>; 3] = std::array::from_fn(|q| {
            let c = Mat::from_fn(GROUND, s, |g, a| self.emission[q][(es[g], excited[a])]);
            f.to(&c.adjoint(), work).adjoint().data
        });
        let weights = terms.iter().map(|x| x.0).collect();
        let mut diag = vec![C64::ZERO; terms.len() * s];
        let mut norms = vec![0.0; terms.len() * s * parts];
        let mut dressed = vec![0.0; terms.len() * s];
        for (t, (_, tw, td, _)) in terms.iter().enumerate() {
            for i in 0..s {
                diag[t * s + i] = tw[(i, i)];
                for j in (0..s).filter(|&j| j != i) {
                    norms[(t * s + i) * parts + part[j]] += tw[(i, j)].norm_sqr();
                }
                dressed[t * s + i] = (0..s).map(|j| td[(i, j)].norm_sqr()).sum::<f64>().sqrt();
            }
        }
        norms.iter_mut().for_each(|x| *x = x.sqrt());
        let copied = terms.iter().map(|x| x.3).collect();
        // The sources: per split Ỹ_l = V† Σ_k γ_jk U_k (source columns), per vector Y_a = V†U_a.
        let mut source = |c: usize, mat: &dyn Fn(usize, usize) -> C64| -> Vec<C64> {
            let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
            let m = Mat::from_fn(s, n, |a, p| if places.contains(&p) { mat(rows[a], self.ground[p]) } else { C64::ZERO });
            f.to(&m, work).data
        };
        let beams = nb.splits.iter().map(|&(c, j)| source(c, &|x, t| self.components[c].couplings[j][(x, t)])).collect();
        let vectors = nb.vectors.iter().map(|&(c, k)| source(c, &|x, t| self.components[c].basis[k][(x, t)])).collect();
        // The copies' rows: per copy vector R₀(C_g, C_g) U_k(S, C_g)† V.
        let vfull = f.from(&Mat::identity(s), work);
        let copies: Vec<Vec<C64>> = nb.copies.iter().map(|&(g, k, gi)| r0.block(&nb.groups[gi].1, &nb.groups[gi].1).product(&u_sc(g, k, &nb.groups[gi].1).adjoint()).product(&vfull).data).collect();
        let copied_rows = copies.iter().map(|y: &Vec<C64>| y.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()).collect();
        // The exact copies: their coupling vectors V†U_k(E, D of g), K̂_ED and K̂_DE at the reference less their bare drives
        // (the dressing through the centred rows, held), K̂_DD, its forms per term.
        let couple: Vec<Vec<C64>> = nb
            .exact
            .iter()
            .map(|&(g, k)| f.to(&Mat::from_fn(s, nd, |a, d| if nb.dstates[d].1 == g { self.components[g].basis[k][(rows[a], nb.dstates[d].0)] } else { C64::ZERO }), work).data)
            .collect();
        let mut ked0 = f.to(&e.k.block(&er, &dr), work).data;
        let mut kde0 = f.to(&e.k.block(&dr, &er).adjoint(), work).adjoint().data;
        for (x, &(g, k)) in nb.exact.iter().enumerate() {
            let ek = ws.mid.e[g][k];
            for i in 0..s {
                for d in 0..nd {
                    ked0[i * nd + d] -= ek * couple[x][i * nd + d];
                    kde0[d * s + i] -= ek.conj() * couple[x][i * nd + d].conj();
                }
            }
        }
        let kdd0 = Mat::from_fn(nd, nd, |a, c| e.k[(s + a, s + c)] - if a == c { C64::real(b.value) } else { C64::ZERO }).data;
        let ddterms: Vec<Vec<C64>> = nb
            .terms
            .iter()
            .map(|&t| {
                let (w, k, kk) = ws.pairs[t];
                let u = &self.light.frequencies[w].basis;
                let mut m = vec![C64::ZERO; nd * nd];
                for (p, q, entries) in &self.entries[w] {
                    let cu = u[k][*p].conj() * u[kk][*q];
                    for &(i, j, v) in entries.iter().filter(|x| x.0 < GROUND && x.1 < GROUND) {
                        for a in 0..nd {
                            for c in 0..nd {
                                if nb.dstates[a].0 == i && nb.dstates[c].0 == j && nb.dstates[a].1 == nb.dstates[c].1 {
                                    m[a * nd + c] += cu * v;
                                }
                            }
                        }
                    }
                }
                m
            })
            .collect();
        Some(NearBasis { at, weights, f, part, diag, norms, dressed, copied, beams, vectors, copies, copied_rows, z0: laid.z0, r0: r0.frobenius(), within, dress, emit, couple, ked0, kde0, kdd0, ddterms })
    }

    /// A far block's pair forms at the point (module documentation, # What is kept).
    fn far_basis(&self, fb: &Far, ws: &Workspace, site: &Site, work: &mut f64) -> Option<FarBasis> {
        let n = ws.n;
        let b = &self.blocks[fb.o];
        let laid = self.lay(fb.o, site, work);
        let e = self.effective(fb.o, &laid, work)?;
        let r0 = e.resolvent;
        let rows: Vec<usize> = b.rows.iter().map(|r| if let Row::Excited(x) = *r { x } else { unreachable!() }).collect();
        let u: Vec<Mat> = fb
            .vectors
            .iter()
            .map(|&(c, k)| {
                let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
                Mat::from_fn(fb.m, n, |a, p| if places.contains(&p) { self.components[c].basis[k][(rows[a], self.ground[p])] } else { C64::ZERO })
            })
            .collect();
        let ru: Vec<Mat> = u.iter().map(|x| r0.product(x)).collect();
        let es: Vec<usize> = (0..GROUND).collect();
        let emitted: Vec<[Vec<C64>; 3]> = ru.iter().map(|x| std::array::from_fn(|q| self.emission[q].block(&es, &rows).product(x).data)).collect();
        let (mut f, mut g) = (vec![], vec![]);
        for a in 0..u.len() {
            for bb in 0..u.len() {
                f.push(u[a].adjoint_product(&ru[bb]).data);
                g.push(ru[a].adjoint_product(&ru[bb]).data);
            }
        }
        *work += (u.len() * fb.m * fb.m * n + 2 * u.len() * u.len() * fb.m * n * n) as f64;
        let copied = fb
            .terms
            .iter()
            .map(|&t| {
                let (w, k, kk) = ws.pairs[t];
                let uu = &self.light.frequencies[w].basis;
                let mut m = Mat::zeros(fb.m, fb.m);
                for (p, q, entries) in &self.entries[w] {
                    let cu = uu[k][*p].conj() * uu[kk][*q];
                    for &(i, j, v) in entries.iter().filter(|x| x.0 >= GROUND && x.1 >= GROUND) {
                        if let (Some(a), Some(c)) = (rows.iter().position(|&x| x == i - GROUND), rows.iter().position(|&x| x == j - GROUND)) {
                            m[(a, c)] += cu * v;
                        }
                    }
                }
                m.frobenius()
            })
            .collect::<Vec<f64>>();
        Some(FarBasis { at: fb.terms.iter().map(|&t| ws.mid.coef[t]).collect(), weights: copied.clone(), f, g, copied, z0: laid.z0, r0: r0.frobenius(), resolvent: r0, emitted })
    }

    /// The prototype's step at a point (r, κ) on an atom's kept decompositions (module documentation): A, M, S⁻¹, G in the
    /// workspace's `Step`. None where a decomposition or a centring is singular.
    pub(crate) fn prototype<'w>(&self, r: [f64; 3], kappa: [f64; 3], cache: &mut Cache, ws: &'w mut Workspace) -> Option<&'w Step> {
        let n = ws.n;
        let mut work = [0.0_f64; 10];
        let mut tick = [0u64; 10];
        ws.step.sweep_work.clear();
        ws.step.sweep_ticks.clear();
        let mut clock = ticks();
        let lap = |k: usize, clock: &mut u64, tick: &mut [u64; 10]| {
            let now = ticks();
            tick[k] += now - *clock;
            *clock = now;
        };
        {
            let Workspace { mid, pairs, pp, base, proj, .. } = &mut *ws;
            self.proto_place(r, kappa, mid, pairs, pp, base, proj, &mut work[PLACE]);
        }
        ws.noded = false;
        lap(PLACE, &mut clock, &mut tick);
        let share = crate::model::elim::share();
        if cache.at_near.len() != ws.near.len() {
            cache.near = (0..ws.near.len()).map(|_| vec![]).collect();
            cache.at_near = vec![None; ws.near.len()];
            cache.far = (0..ws.far.len()).map(|_| vec![]).collect();
            cache.at_far = vec![None; ws.far.len()];
        }
        ws.renew.clear();
        let made_before = cache.made;
        let mut round = 0;
        let mut site: Option<Site> = None;
        let (sweeps, series, ratio, rounds) = loop {
            let made = cache.made;
            // The bases (module documentation, 6): the atom's own; beyond the share the nearest kept, then a new one.
            for kind in [false, true] {
                let count = if kind { ws.far.len() } else { ws.near.len() };
                for x in 0..count {
                    let tries = ws.renew.iter().filter(|y| **y == (kind, x)).count();
                    let standing = if kind { cache.at_far[x] } else { cache.at_near[x] };
                    if let (Some(_), 0) = (standing, tries) {
                        continue;
                    }
                    let (terms, kept): (&[usize], Vec<(&[C64], &[f64])>) = if kind {
                        (&ws.far[x].terms, cache.far[x].iter().map(|b| (&b.at[..], &b.weights[..])).collect())
                    } else {
                        (&ws.near[x].terms, cache.near[x].iter().map(|b| (&b.at[..], &b.weights[..])).collect())
                    };
                    let nearest = |but: Option<usize>, work: &mut f64| -> Option<usize> {
                        let mut best: Option<(usize, f64)> = None;
                        for (i, (at, weights)) in kept.iter().enumerate().filter(|(i, _)| Some(*i) != but) {
                            let d: f64 = terms.iter().enumerate().map(|(t, &p)| (ws.mid.coef[p] - at[t]).abs() * weights[t]).sum();
                            *work += terms.len() as f64;
                            if best.is_none_or(|b| d < b.1) {
                                best = Some((i, d));
                            }
                        }
                        best.map(|b| b.0)
                    };
                    let chosen = match (standing, tries) {
                        (Some(at), 0) => Some(at),
                        (None, 0) => nearest(None, &mut work[REFRESH]),
                        (_, 1) => nearest(standing, &mut work[REFRESH]),
                        _ => None,
                    };
                    if chosen.is_some() && chosen != standing {
                        cache.looked += 1;
                    }
                    let at = match chosen {
                        Some(at) => at,
                        None => {
                            if site.is_none() {
                                site = Some(self.site(r, kappa, &mut work[REFRESH]));
                            }
                            let site = site.as_ref().unwrap();
                            if kind {
                                let basis = self.far_basis(&ws.far[x], ws, site, &mut work[REFRESH])?;
                                cache.far[x].push(basis);
                                cache.far[x].len() - 1
                            } else {
                                let basis = self.near_basis(&ws.near[x], ws, site, &mut work[REFRESH])?;
                                cache.near[x].push(basis);
                                cache.near[x].len() - 1
                            }
                        }
                    };
                    if chosen.is_none() {
                        cache.made += 1;
                    }
                    if kind {
                        cache.at_far[x] = Some(at);
                    } else {
                        cache.at_near[x] = Some(at);
                    }
                }
            }
            lap(REFRESH, &mut clock, &mut tick);
            // The near blocks: the diagonal to first order and what it leaves; their sources (module documentation, 2, 3).
            let Workspace { near, far, mid, hbase, h, delta, c, change, previous, whole, step, order, cs, deltas, .. } = &mut *ws;
            let Placed { coef, f, w, e, hpp, .. } = &*mid;
            for (x, nb) in near.iter_mut().enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let fa = &basis.f;
                let parts = fa.parts.len().max(1);
                let z0: f64 = nb.centre.iter().map(|&p| hpp[p * n + p].re).sum::<f64>() / nb.centre.len().max(1) as f64;
                for i in 0..s {
                    nb.kappa[i] = fa.t[(i, i)] - C64::real(nb.value);
                    for g in 0..parts {
                        nb.off[i * parts + g] = if g == basis.part[i] { fa.dropped[g] } else { 0.0 };
                    }
                    nb.rho[i] = 0.0;
                }
                let mut dm = (z0 - basis.z0).abs();
                for (t, &p) in nb.terms.iter().enumerate() {
                    let d = coef[p] - basis.at[t];
                    nb.delta[t] = d;
                    if d == C64::ZERO {
                        continue;
                    }
                    let size = d.abs();
                    dm += size * basis.copied[t];
                    let (dg, nm) = (&basis.diag[t * s..(t + 1) * s], &basis.norms[t * s * parts..(t + 1) * s * parts]);
                    for i in 0..s {
                        nb.kappa[i].re += (d * dg[i]).re;
                    }
                    for (o, x) in nb.off[..s * parts].iter_mut().zip(nm) {
                        *o += size * x;
                    }
                    work[DIAGONAL] += (s + s * parts) as f64;
                }
                // The exact copies' couplings and K̂_DD at the point (exact in the light, the dressing's held).
                if nb.nd > 0 {
                    let (mut ked, mut kde) = (std::mem::take(&mut nb.ked), std::mem::take(&mut nb.kde));
                    copy_couplings(nb, basis, e, &mut ked, &mut kde);
                    (nb.ked, nb.kde) = (ked, kde);
                    nb.kdd.copy_from_slice(&basis.kdd0);
                    for (t, &p) in nb.terms.iter().enumerate() {
                        let d = coef[p] - basis.at[t];
                        if d != C64::ZERO {
                            crate::linalg::mul_add(false, &mut nb.kdd, d, &basis.ddterms[t]);
                        }
                    }
                    work[DIAGONAL] += (2 * nb.exact.len() * s * nb.nd + nb.terms.len() * nb.nd * nb.nd) as f64;
                }
                // R₀'s own change: δR₀ ≤ ‖R₀‖²‖δM‖/(1 − ‖R₀‖‖δM‖) through the dressing, ρ_i = ε_R Σ_t |c_t| ‖(V†N_t,dress V)_i‖.
                if nb.nc > 0 {
                    let x = basis.r0 * dm;
                    let eps_r = if x < 1.0 { x / (1.0 - x) } else { f64::INFINITY };
                    for (t, &p) in nb.terms.iter().enumerate() {
                        let size = coef[p].abs();
                        for i in 0..s {
                            nb.rho[i] += eps_r * size * basis.dressed[t * s + i];
                        }
                    }
                    work[DIAGONAL] += (nb.terms.len() * s) as f64 / 4.0;
                }
            }
            lap(DIAGONAL, &mut clock, &mut tick);
            for (x, nb) in near.iter_mut().enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                for (sp, &(cc, j)) in nb.splits.iter().enumerate() {
                    nb.shifts[sp] = w[cc][j];
                    let fl = f[cc][j];
                    for (v, y) in nb.vt[sp * s * n..(sp + 1) * s * n].iter_mut().zip(&basis.beams[sp]) {
                        *v = fl * *y;
                    }
                }
                nb.vtot.fill(C64::ZERO);
                for (a, &(cc, k)) in nb.vectors.iter().enumerate() {
                    let ek = e[cc][k];
                    for (v, y) in nb.vtot.iter_mut().zip(&basis.vectors[a]) {
                        *v += ek * *y;
                    }
                }
                work[SOURCES] += ((nb.splits.len() + nb.vectors.len()) * s * n) as f64;
            }
            lap(SOURCES, &mut clock, &mut tick);
            // The far blocks' pair forms (module documentation, 4): H_far = Σ e_a*e_b U_a†R₀U_b, M_far likewise.
            hbase.copy_from_slice(hpp);
            for (x, fb) in far.iter_mut().enumerate() {
                let basis = &cache.far[x][cache.at_far[x].unwrap()];
                let v = fb.vectors.len();
                for a in 0..v {
                    for b in 0..v {
                        let (ca, ka) = fb.vectors[a];
                        let (cb, kb) = fb.vectors[b];
                        fb.products[a * v + b] = e[ca][ka].conj() * e[cb][kb];
                    }
                }
                fb.hf.fill(C64::ZERO);
                fb.mf.fill(C64::ZERO);
                for (ab, &x) in fb.products.iter().enumerate() {
                    for ((hf, mf), (ff, gg)) in fb.hf.iter_mut().zip(fb.mf.iter_mut()).zip(basis.f[ab].iter().zip(&basis.g[ab])) {
                        *hf += x * *ff;
                        *mf += x * *gg;
                    }
                }
                for (h, x) in hbase.iter_mut().zip(&fb.hf) {
                    *h += *x;
                }
                for (t, &p) in fb.terms.iter().enumerate() {
                    fb.delta[t] = coef[p] - basis.at[t];
                }
                work[FAR] += (v * v + 2 * v * v * n * n + n * n) as f64;
            }
            lap(FAR, &mut clock, &mut tick);
            // The series (module documentation, 5).
            h.copy_from_slice(hbase);
            for nb in near.iter_mut() {
                nb.ztot.fill(C64::ZERO);
                nb.active.fill(true);
                nb.held.fill(C64::ZERO);
                nb.invs.clear();
                nb.zs.clear();
                nb.ztots.clear();
                nb.zd.fill(C64::ZERO);
                nb.zdtot.fill(C64::ZERO);
                nb.zds.clear();
                nb.zdtots.clear();
                nb.pjs.clear();
                nb.cinvs.clear();
            }
            cs.clear();
            deltas.clear();
            previous.fill(0.0);
            let mut sweep = 0;
            let mut frozen = 0.0_f64;
            let (series, ratio) = loop {
                let (t0, w0) = (ticks(), work[SERIES]);
                // The columns' energies (H's diagonal, each column at its own) and the series' part of H (off it).
                for j in 0..n {
                    c[j] = h[j * n + j];
                }
                delta.copy_from_slice(h);
                for j in 0..n {
                    delta[j * n + j] = C64::ZERO;
                }
                cs.extend_from_slice(c);
                deltas.extend_from_slice(delta);
                change.fill(0.0);
                whole.fill(0.0);
                for (x, nb) in near.iter_mut().enumerate() {
                    let basis = &cache.near[x][cache.at_near[x].unwrap()];
                    let s = nb.s;
                    let (t, ends) = (&basis.f.t, &basis.f.ends);
                    // Only the rows still going on move: the held ones keep their response.
                    nb.zold.copy_from_slice(&nb.ztot);
                    let block = nb.splits.len() * s * n;
                    let at = nb.invs.len();
                    nb.invs.resize(at + block, C64::ZERO);
                    let nd = nb.nd;
                    let (pat, cat) = (nb.pjs.len(), nb.cinvs.len());
                    nb.pjs.resize(pat + nb.splits.len() * n * nd * s, C64::ZERO);
                    nb.cinvs.resize(cat + nb.splits.len() * n * nd * nd, C64::ZERO);
                    nb.zdtot.fill(C64::ZERO);
                    for i in (0..s).filter(|&i| nb.active[i]) {
                        nb.ztot[i * n..i * n + n].fill(C64::ZERO);
                    }
                    for sp in 0..nb.splits.len() {
                        let wl = nb.shifts[sp];
                        let zl = &mut nb.z[sp * s * n..(sp + 1) * s * n];
                        let rhs = &mut nb.rhs;
                        rhs.copy_from_slice(&nb.vt[sp * s * n..(sp + 1) * s * n]);
                        if sweep > 0 {
                            for i in (0..s).filter(|&i| nb.active[i]) {
                                for k in 0..n {
                                    let zik = zl[i * n + k];
                                    if zik == C64::ZERO {
                                        continue;
                                    }
                                    crate::linalg::mul_add(true, &mut rhs[i * n..i * n + n], zik, &delta[k * n..k * n + n]);
                                }
                            }
                            work[SERIES] += (nb.active.iter().filter(|&&x| x).count() * n * n) as f64;
                        }
                        // The diagonal: (c_j − w_l − κ_i)⁻¹, each column at its own energy.
                        for i in (0..s).filter(|&i| nb.active[i]) {
                            for j in 0..n {
                                nb.invs[at + sp * s * n + i * n + j] = reciprocal(c[j] - C64::real(wl) - nb.kappa[i]);
                            }
                        }
                        work[SERIES] += (s * n) as f64;
                        for i in (0..s).rev().filter(|&i| nb.active[i]) {
                            for l in i + 1..ends[i] {
                                crate::linalg::mul_add(false, &mut rhs[i * n..i * n + n], t[(i, l)], &zl[l * n..l * n + n]);
                            }
                            work[SERIES] += ((ends[i] - i - 1) * n + n) as f64;
                            for j in 0..n {
                                zl[i * n + j] = rhs[i * n + j] * nb.invs[at + sp * s * n + i * n + j];
                            }
                        }
                        // The exact copies by Woodbury (`copies_solve`), their own source the series' part alone.
                        if nd > 0 {
                            let mut rd = vec![C64::ZERO; nd * n];
                            let zdl = &mut nb.zd[sp * nd * n..(sp + 1) * nd * n];
                            if sweep > 0 {
                                for a in 0..nd {
                                    for k in 0..n {
                                        let x = zdl[a * n + k];
                                        if x != C64::ZERO {
                                            crate::linalg::mul_add(true, &mut rd[a * n..a * n + n], x, &delta[k * n..k * n + n]);
                                        }
                                    }
                                }
                            }
                            let (pj, cinv) = (&mut nb.pjs[pat + sp * n * nd * s..pat + (sp + 1) * n * nd * s], &mut nb.cinvs[cat + sp * n * nd * nd..cat + (sp + 1) * n * nd * nd]);
                            let inv = &nb.invs[at + sp * s * n..at + (sp + 1) * s * n];
                            let mut zdn = vec![C64::ZERO; nd * n];
                            let ok = copies_solve(s, nd, &nb.ked, &nb.kde, &nb.kdd, basis, c, wl, inv, &rd, zl, &mut zdn, pj, cinv, false, n, &mut work[SERIES]);
                            if !ok {
                                return None;
                            }
                            zdl.copy_from_slice(&zdn);
                            for (a, b) in nb.zdtot.iter_mut().zip(&zdn) {
                                *a += *b;
                            }
                        }
                        for i in (0..s).filter(|&i| nb.active[i]) {
                            for (a, b) in nb.ztot[i * n..i * n + n].iter_mut().zip(&zl[i * n..i * n + n]) {
                                *a += *b;
                            }
                        }
                    }
                    nb.zs.extend_from_slice(&nb.z);
                    nb.ztots.extend_from_slice(&nb.ztot);
                    nb.zds.extend_from_slice(&nb.zd);
                    nb.zdtots.extend_from_slice(&nb.zdtot);
                    // The change and the whole response, column by column; the Bloch generator's part Ṽ†Z.
                    for i in 0..s {
                        for j in 0..n {
                            let zn = nb.ztot[i * n + j];
                            if nb.active[i] {
                                change[j] += (zn - nb.zold[i * n + j]).norm_sqr();
                            }
                            whole[j] += zn.norm_sqr();
                        }
                    }
                    nb.bloch.copy_from_slice(&nb.held);
                    for i in (0..s).filter(|&i| nb.active[i]) {
                        for p in 0..n {
                            let v = nb.vtot[i * n + p].conj();
                            if v == C64::ZERO {
                                continue;
                            }
                            crate::linalg::mul_add(false, &mut nb.bloch[p * n..p * n + n], v, &nb.ztot[i * n..i * n + n]);
                        }
                    }
                    work[SERIES] += (n * s * n + nb.splits.len() * s * n) as f64;
                }
                h.copy_from_slice(hbase);
                for nb in near.iter() {
                    for (a, b) in h.iter_mut().zip(&nb.bloch) {
                        *a += *b;
                    }
                }
                for fb in far.iter() {
                    for j in 0..n {
                        whole[j] += fb.mf[j * n + j].re;
                    }
                }
                sweep += 1;
                if sweep == 1 {
                    frozen = freeze(near, cache, h, c, whole, order, share, n, &mut work[SERIES]);
                }
                // The rest: the last change times r/(1 − r), r the ratio of the last two changes (none after the first).
                let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
                let (mut rest, mut most) = (0.0_f64, 0.0_f64);
                for j in (0..n).filter(|&j| whole[j] > floor) {
                    let ratio = if previous[j] > 0.0 { (change[j] / previous[j]).sqrt() } else { f64::INFINITY };
                    most = most.max(ratio);
                    let x = if ratio < 1.0 { change[j].sqrt() * ratio / (1.0 - ratio) } else { f64::INFINITY };
                    rest = rest.max(rate(x / whole[j].sqrt()));
                }
                previous.copy_from_slice(change);
                step.sweep_ticks.push(ticks() - t0);
                step.sweep_work.push(work[SERIES] - w0);
                // Two sweeps at the least; then within the share, or no longer converging (a ratio at or above one after
                // the second change: the self-consistency or the series does not exist here).
                if sweep >= 2 && (rest <= share || (sweep >= 3 && most >= 1.0)) {
                    break (rate(unrate(rest) + frozen), most);
                }
            };
            lap(SERIES, &mut clock, &mut tick);
            // The readings of what is held (module documentation, 6), at the columns' energies H_jj. The reuse's first-order
            // error of the response the step uses, Z = Σ_l Z_l: Σ_l R_l δK̃ Z_l = R̄ δK̃ Z + Σ_l (R_l − R̄) δK̃ Z_l, R̄ at the
            // beams' mean Doppler shift w̄ -- row by row, row i of δK̃Z at most Σ_P off_iP ‖Z_P∖i‖ (+ ρ_i‖Z‖), the beams'
            // rest at most W|R̄|²/(1 − W|R̄|) (W the largest |w_l − w̄|) times (Σ_P off_iP + ρ_i) Σ_l ‖Z_l‖. Read on the
            // beams one by one against their sum, the reading would grow where the beams' responses cancel (a standing
            // wave's node) though the error cancels with them.
            for (x, nb) in near.iter_mut().enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let parts = basis.f.parts.len().max(1);
                let beams = nb.splits.len();
                let mean = nb.shifts.iter().sum::<f64>() / beams.max(1) as f64;
                let spread = nb.shifts.iter().map(|w| (w - mean).abs()).fold(0.0, f64::max);
                for j in 0..n {
                    nb.norms[..parts].fill(0.0);
                    for i in 0..s {
                        nb.norms[basis.part[i]] += nb.ztot[i * n + j].norm_sqr();
                    }
                    let total: f64 = nb.norms[..parts].iter().sum();
                    let mut each = 0.0;
                    for sp in 0..beams {
                        each += (0..s).map(|i| nb.z[sp * s * n + i * n + j].norm_sqr()).sum::<f64>().sqrt();
                    }
                    for i in 0..s {
                        let own = basis.part[i];
                        let mut e = nb.rho[i] * total.sqrt();
                        let mut all = nb.rho[i];
                        for g in 0..parts {
                            let off = nb.off[i * parts + g];
                            all += off;
                            if off > 0.0 {
                                e += off * if g == own { (nb.norms[g] - nb.ztot[i * n + j].norm_sqr()).max(0.0) } else { nb.norms[g] }.sqrt();
                            }
                        }
                        let gap = (h[j * n + j] - C64::real(mean) - nb.kappa[i]).norm_sqr().sqrt();
                        let rest = if spread < gap { all * each * spread / (gap * (gap - spread)) } else { f64::INFINITY };
                        nb.err[i * n + j] = if e > 0.0 || rest > 0.0 { e / gap + rest } else { 0.0 };
                    }
                }
                work[READING] += (s * n * (beams + parts + 2)) as f64 / 4.0;
                let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
                nb.reading = (0..n).filter(|&j| whole[j] > floor).map(|j| rate(((0..s).map(|i| nb.err[i * n + j].powi(2)).sum::<f64>() / whole[j]).sqrt())).fold(0.0, f64::max);
            }
            for (x, fb) in far.iter_mut().enumerate() {
                let basis = &cache.far[x][cache.at_far[x].unwrap()];
                let z0: f64 = fb.centre.iter().map(|&p| hpp[p * n + p].re).sum::<f64>() / fb.centre.len().max(1) as f64;
                let held = (z0 - basis.z0).abs() + fb.delta.iter().zip(&basis.copied).map(|(d, c)| d.abs() * c).sum::<f64>();
                let energies = (0..n).map(|j| (h[j * n + j] - C64::real(z0)).abs()).fold(0.0, f64::max) + fb.splits.iter().map(|&(cc, j)| w[cc][j].abs()).fold(0.0, f64::max);
                let read = |d: f64| {
                    let x = basis.r0 * d;
                    if x < 1.0 { rate(x / (1.0 - x)) } else { f64::INFINITY }
                };
                fb.reading = read(held);
                fb.centring = read(energies);
                work[READING] += fb.terms.len() as f64 / 4.0;
            }
            lap(READING, &mut clock, &mut tick);
            let mut renew = vec![];
            for (x, nb) in near.iter().enumerate() {
                if nb.reading > share {
                    renew.push((false, x));
                }
            }
            for (x, fb) in far.iter().enumerate() {
                if fb.reading > share {
                    renew.push((true, x));
                }
            }
            if renew.is_empty() || round == 2 || cache.made > made {
                break (sweep, series, ratio, round + 1);
            }
            ws.renew.extend(renew);
            round += 1;
        };
        // The copies' rows and the compression (module documentation, 7).
        let Workspace { near, far, h, step, eigen, vectors, temp, mid, .. } = &mut *ws;
        let e = &mid.e;
        let (a, metric) = (&mut step.a, &mut step.metric);
        a.data.copy_from_slice(h);
        metric.data.fill(C64::ZERO);
        let mut copies_read = 0.0_f64;
        for (x, nb) in near.iter_mut().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            nb.tc.fill(C64::ZERO);
            nb.copied_in = false;
            if nb.nc == 0 {
                continue;
            }
            // Their rows' own part of the compression (their metric T_C†T_C and z₀T_C†T_C; their dressing is the kept
            // decomposition's) left out where it reads within the share: ‖T_C,j‖ ≤ Σ_a |e_a| ‖R₀U_a†V‖ ‖Z_j‖, a copy row's
            // error absolute (a population's amplitude per unit P amplitude, DESIGN §1.5).
            let size: f64 = nb.copies.iter().zip(&basis.copied_rows).map(|(&(g, k, _), y)| e[g][k].abs() * y).sum();
            let most = (0..n).map(|j| (0..nb.s).map(|i| nb.ztot[i * n + j].norm_sqr()).sum::<f64>()).fold(0.0, f64::max).sqrt();
            let read = rate(size * most);
            copies_read = copies_read.max(read);
            work[COPIES] += (nb.copies.len() + nb.s * n / 4) as f64;
            if read <= share {
                continue;
            }
            nb.copied_in = true;
            copy_rows(nb, basis, e, n, &mut work[COPIES]);
            for k in 0..nb.nc {
                for i in 0..n {
                    let ti = nb.tc[k * n + i].conj();
                    if ti == C64::ZERO {
                        continue;
                    }
                    for j in 0..n {
                        let tkj = nb.tc[k * n + j];
                        metric.data[i * n + j] += ti * tkj;
                        a.data[i * n + j] += ti * tkj * basis.z0;
                    }
                }
            }
            work[COPIES] += (nb.nc * n * n) as f64;
        }
        lap(COPIES, &mut clock, &mut tick);
        for (x, nb) in near.iter_mut().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let s = nb.s;
            let (t, ends) = (&basis.f.t, &basis.f.ends);
            // (Ṽ†Z)†, and Y = (T + δκ − o)Z with κ on the diagonal.
            for p in 0..n {
                for j in 0..n {
                    a.data[p * n + j] += nb.bloch[j * n + p].conj();
                }
            }
            for i in 0..s {
                for j in 0..n {
                    nb.y[i * n + j] = nb.kappa[i] * nb.ztot[i * n + j];
                }
                for l in i + 1..ends[i] {
                    let tij = t[(i, l)];
                    for j in 0..n {
                        nb.y[i * n + j] += tij * nb.ztot[l * n + j];
                    }
                }
                work[COMPRESSION] += ((ends[i] - i) * n) as f64;
            }
            for k in 0..s {
                for i in 0..n {
                    let zi = nb.ztot[k * n + i].conj();
                    if zi == C64::ZERO {
                        continue;
                    }
                    crate::linalg::mul_add(false, &mut a.data[i * n..i * n + n], zi, &nb.y[k * n..k * n + n]);
                    crate::linalg::mul_add(false, &mut metric.data[i * n..i * n + n], zi, &nb.ztot[k * n..k * n + n]);
                }
            }
            work[COMPRESSION] += (2 * s * n * n + n * n) as f64;
            if nb.nd > 0 {
                copies_compress(s, nb.nd, &nb.ked, &nb.kde, &nb.kdd, &nb.ztot, &nb.zdtot, &mut a.data, Some(&mut metric.data), n, &mut work[COMPRESSION]);
            }
        }
        for (x, fb) in far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            for ((am, mm), mf) in a.data.iter_mut().zip(metric.data.iter_mut()).zip(&fb.mf) {
                *am += *mf * basis.z0;
                *mm += *mf;
            }
            work[COMPRESSION] += (n * n) as f64;
        }
        lap(COMPRESSION, &mut clock, &mut tick);
        // S⁻¹ = (1 + X)^{−1/2} (`inverse_root`: it converges wherever the fold exists), G = S⁻¹AS⁻¹.
        if !inverse_root(&metric.data, n, &mut step.normal.data, &mut eigen.data, &mut vectors.data, &mut temp.data, &mut work[NORMAL]) {
            return None;
        }
        multiply(&step.normal.data, &step.a.data, &mut temp.data, n);
        multiply(&temp.data, &step.normal.data, &mut step.generator.data, n);
        work[NORMAL] += (2 * n * n * n) as f64;
        lap(NORMAL, &mut clock, &mut tick);
        step.sweeps = sweeps;
        step.series = series;
        step.ratio = ratio;
        step.reuse = near.iter().map(|x| x.reading).fold(0.0, f64::max);
        step.far = far.iter().map(|x| x.reading).fold(0.0, f64::max);
        step.centring = far.iter().map(|x| x.centring).fold(0.0, f64::max);
        step.copies = copies_read;
        step.made = cache.made - made_before;
        step.rounds = rounds;
        step.work = work;
        step.ticks = tick;
        Some(&ws.step)
    }
}

/// The force at a step (`Fold::prototype_force`): per lab axis −∂/∂r_a of Σ_j Re tr(Ḡ_j†G_j) over the step's positions
/// (the middle and the outer nodes where made), all moved together; per axis the bound on what the kept R₀ and z₀ leave of
/// it (their own change along the axis: module documentation, # The force); the work (complex multiply-adds).
pub(crate) struct Force {
    pub(crate) force: [f64; 3],
    pub(crate) held: [f64; 3],
    pub(crate) work: f64,
}

/// The absorbed light's moments at a step's middle (`Fold::prototype_photons`; photon.rs # The absorbed light's momentum):
/// the rate W, the force the jumps do not carry and the diffusion 2D, per unit norm of the state; the work.
pub(crate) struct Photons {
    pub(crate) rate: f64,
    pub(crate) force: [f64; 3],
    pub(crate) diffusion: [[f64; 3]; 3],
    pub(crate) work: f64,
}

impl Fold {
    /// Per frequency and basis vector its component's gradient at a place, ∂_a ε_k = Σ_l ∂_a a_l u_k†E_l.
    fn component_gradients(&self, at: &Placed, proj: &[Vec<C64>]) -> Vec<Vec<[C64; 3]>> {
        self.light
            .frequencies
            .iter()
            .enumerate()
            .map(|(w, f)| {
                let rank = f.rank();
                (0..rank).map(|k| std::array::from_fn(|a| f.beams.iter().enumerate().map(|(j, &l)| at.amplitudes[l].gradient[a] * proj[w][j * rank + k]).sum())).collect()
            })
            .collect()
    }

    /// Per pair its coefficient's gradient at a place, ∂_a c_t = ∂_a ε_k* ε_k' + ε_k* ∂_a ε_k'.
    fn pair_gradients(&self, at: &Placed, proj: &[Vec<C64>], pairs: &[Pair]) -> Vec<[C64; 3]> {
        let deps = self.component_gradients(at, proj);
        pairs.iter().map(|&(w, k, kk)| std::array::from_fn(|a| deps[w][k][a].conj() * at.eps[w][kk] + at.eps[w][k].conj() * deps[w][kk][a])).collect()
    }

    /// A block's centre z₀'s gradient (the mean of its sources' Ĥ_PP diagonal), from the pairs' coefficients' gradients.
    fn centre_gradient(&self, centre: &[usize], pp: &[Vec<(usize, C64)>], dc: &[[C64; 3]], n: usize) -> [f64; 3] {
        let mut g = [0.0; 3];
        for (t, entries) in pp.iter().enumerate() {
            for &(i, v) in entries {
                if i % (n + 1) == 0 && centre.contains(&(i / n)) {
                    for a in 0..3 {
                        g[a] += (dc[t][a] * v).re / centre.len().max(1) as f64;
                    }
                }
            }
        }
        g
    }

    /// The step's outer nodes (module documentation, # The nodes; DESIGN §4.2): at the offsets `times` (µs) from the middle
    /// along the path of velocity v (µm/µs), the light there; every split's response the middle's times its factor's ratio
    /// there to the middle's (held per unit amplitude: the field turned; the decomposition, H, S and the responses per beam
    /// held over the step), Ṽ(s), the far forms and Ĥ_PP there; A(s), and G(s) = S⁻¹A(s)S⁻¹ with the middle's S. After
    /// `prototype` at the middle, on its cache. Returns the work.
    pub(crate) fn prototype_nodes(&self, r: [f64; 3], kappa: [f64; 3], velocity: [f64; 3], times: [f64; 2], cache: &Cache, ws: &mut Workspace) -> f64 {
        let n = ws.n;
        let mut work = 0.0;
        ws.node_times = times;
        ws.noded = true;
        let Workspace { mid, nodes, near, far, pairs, pp, base, proj, step, temp, .. } = &mut *ws;
        let (mid, pairs, pp, base, proj, step) = (&*mid, &*pairs, &*pp, &*base, &*proj, &*step);
        for (j, node) in nodes.iter_mut().enumerate() {
            let rj: [f64; 3] = std::array::from_fn(|a| r[a] + velocity[a] * times[j]);
            self.proto_place(rj, kappa, &mut node.at, pairs, pp, base, proj, &mut work);
            let at = &node.at;
            let a = &mut node.a.data;
            a.copy_from_slice(&at.hpp);
            for (x, fb) in far.iter().enumerate() {
                let basis = &cache.far[x][cache.at_far[x].unwrap()];
                let v = fb.vectors.len();
                let (hf, mf) = (&mut node.hf[x], &mut node.mf[x]);
                hf.fill(C64::ZERO);
                mf.fill(C64::ZERO);
                for ia in 0..v {
                    for ib in 0..v {
                        let ((ca, ka), (cb, kb)) = (fb.vectors[ia], fb.vectors[ib]);
                        let p = at.e[ca][ka].conj() * at.e[cb][kb];
                        crate::linalg::mul_add(false, hf, p, &basis.f[ia * v + ib]);
                        crate::linalg::mul_add(false, mf, p, &basis.g[ia * v + ib]);
                    }
                }
                for i in 0..n * n {
                    a[i] += hf[i] + mf[i] * basis.z0;
                }
                work += (2 * v * v * n * n + n * n) as f64;
            }
            for (x, nb) in near.iter_mut().enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                for (sp, &(cc, jj)) in nb.splits.iter().enumerate() {
                    nb.ratio[j][sp] = at.f[cc][jj] / mid.f[cc][jj];
                }
                let zn = &mut nb.znode[j];
                zn.fill(C64::ZERO);
                for sp in 0..nb.splits.len() {
                    crate::linalg::mul_add(false, zn, nb.ratio[j][sp], &nb.z[sp * s * n..(sp + 1) * s * n]);
                }
                let vn = &mut nb.vnode[j];
                vn.fill(C64::ZERO);
                for (ia, &(cc, k)) in nb.vectors.iter().enumerate() {
                    crate::linalg::mul_add(false, vn, at.e[cc][k], &basis.vectors[ia]);
                }
                work += ((nb.splits.len() + nb.vectors.len()) * s * n) as f64;
                compress(&nb.kappa, &basis.f.t, &basis.f.ends, &nb.znode[j], &nb.vnode[j], &mut nb.rhs, a, s, n, &mut work);
                if nb.nd > 0 {
                    let nd = nb.nd;
                    let zd = &mut nb.zdnode[j];
                    zd.fill(C64::ZERO);
                    for sp in 0..nb.splits.len() {
                        crate::linalg::mul_add(false, zd, nb.ratio[j][sp], &nb.zd[sp * nd * n..(sp + 1) * nd * n]);
                    }
                    let (mut ked, mut kde) = (std::mem::take(&mut nb.kednode[j]), std::mem::take(&mut nb.kdenode[j]));
                    copy_couplings(nb, basis, &at.e, &mut ked, &mut kde);
                    (nb.kednode[j], nb.kdenode[j]) = (ked, kde);
                    copies_compress(s, nd, &nb.kednode[j], &nb.kdenode[j], &nb.kdd, &nb.znode[j], &nb.zdnode[j], a, None, n, &mut work);
                }
                if nb.copied_in {
                    copies_of(&nb.copies, &nb.groups, basis, &at.e, &nb.znode[j], &mut nb.tcnode[j], s, n, &mut work);
                    let tc = &nb.tcnode[j];
                    for k in 0..nb.nc {
                        for i in 0..n {
                            let ti = tc[k * n + i].conj() * basis.z0;
                            if ti != C64::ZERO {
                                crate::linalg::mul_add(false, &mut a[i * n..i * n + n], ti, &tc[k * n..k * n + n]);
                            }
                        }
                    }
                    work += (nb.nc * n * n) as f64;
                }
            }
            multiply(&step.normal.data, &node.a.data, &mut temp.data, n);
            multiply(&temp.data, &step.normal.data, &mut node.generator.data, n);
            work += (2 * n * n * n) as f64;
        }
        work
    }

    /// The force by the adjoint (module documentation, # The force): −∂/∂r_a of Σ_j Re tr(Ḡ_j†G_j), `bars` the cotangents of
    /// the middle's G and of the outer nodes' (where `prototype_nodes` made them; ignored otherwise), every position moved
    /// together. Back through G = S⁻¹AS⁻¹, S⁻¹ (`inverse_root_back`), the compressions (`compress_back`, the copies' rows),
    /// the nodes' responses (their factors' ratios), and each sweep of the series, the last first -- its triangular solves
    /// adjoint (the same kept Schur form, forward), its Δ and column energies to the Bloch generator before it, the held
    /// parts' rows to the first sweep --, to the light's coefficients at every position (`contracted`): the splits'
    /// factors, the components' coefficients, the pairs' (Ĥ_PP, the far forms, and K̃ through the kept decomposition: K̃̄
    /// on its parts' entries contracted with V†N_tV whole -- the reuse moves the diagonal alone, the gradient takes all of
    /// K's change). The kept R₀ and z₀ (the centred copies' and the far blocks') are held: what that leaves is bounded per
    /// axis (`Force::held`). After `prototype` (and `prototype_nodes`) on the same cache.
    pub(crate) fn prototype_force(&self, cache: &Cache, ws: &mut Workspace, bars: [&Mat; 3]) -> Force {
        let n = ws.n;
        let nn = n * n;
        let mut work = 0.0;
        let Workspace { near, far, mid, nodes, noded, pairs, pp, proj, deltas, step, eigen, vectors, temp, .. } = &mut *ws;
        let (mid, nodes, pairs, pp, proj, deltas, step) = (&*mid, &*nodes, &*pairs, &*pp, &*proj, &*deltas, &*step);
        let positions = if *noded { 3 } else { 1 };
        let sweeps = step.sweeps;
        let normal = &step.normal.data;
        // Ā_j = S⁻¹Ḡ_jS⁻¹; N̄ = Σ_j Ḡ_j S⁻¹ A_j† + A_j† S⁻¹ Ḡ_j.
        let mut abar: Vec<Buf> = vec![];
        let mut nbar = Buf::zeros(nn);
        let mut adj = Buf::zeros(nn);
        for j in 0..positions {
            let g = &bars[j].data;
            let aj = if j == 0 { &step.a.data } else { &nodes[j - 1].a.data };
            multiply(normal, g, &mut eigen.data, n);
            multiply(&eigen.data, normal, &mut temp.data, n);
            abar.push(Buf::copy(&temp.data));
            for p in 0..n {
                for q in 0..n {
                    adj[p * n + q] = aj[q * n + p].conj();
                }
            }
            multiply(g, normal, &mut eigen.data, n);
            multiply(&eigen.data, &adj, &mut temp.data, n);
            nbar.iter_mut().zip(&temp.data).for_each(|(o, x)| *o += *x);
            multiply(&adj, normal, &mut eigen.data, n);
            multiply(&eigen.data, g, &mut temp.data, n);
            nbar.iter_mut().zip(&temp.data).for_each(|(o, x)| *o += *x);
            work += (6 * n * n * n) as f64;
        }
        let mut xbar = Buf::zeros(nn);
        inverse_root_back(&step.metric.data, normal, &nbar, n, &mut xbar, &mut eigen.data, &mut temp.data, &mut vectors.data, &mut work);
        // The light's cotangents per position: the splits' factors, the components' coefficients, the pairs' coefficients.
        let mut gf: Vec<Vec<Buf>> = (0..positions).map(|_| self.components.iter().map(|c| Buf::zeros(c.beams.len())).collect()).collect();
        let mut ge: Vec<Vec<Buf>> = (0..positions).map(|_| self.components.iter().map(|c| Buf::zeros(c.basis.len())).collect()).collect();
        let mut kt: Vec<Buf> = (0..positions).map(|_| Buf::zeros(pairs.len())).collect();
        // A far block's forms Σ p_ab F_ab, Σ p_ab G_ab (p_ab = e_a*e_b) for their cotangents H̄, M̄.
        let far_back = |x: usize, fb: &Far, e: &[Vec<C64>], hbar: &[C64], mbar: &[C64], ge: &mut [Buf], work: &mut f64| {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            let v = fb.vectors.len();
            for ia in 0..v {
                for ib in 0..v {
                    let ((ca, ka), (cb, kb)) = (fb.vectors[ia], fb.vectors[ib]);
                    let (f, g) = (&basis.f[ia * v + ib], &basis.g[ia * v + ib]);
                    let pbar: C64 = (0..f.len()).map(|i| hbar[i].conj() * f[i] + mbar[i].conj() * g[i]).sum();
                    // Re(p̄ dp): ḡ_a += (p̄ e_b)*, ḡ_b += p̄ e_a*.
                    ge[ca][ka] += (pbar * e[cb][kb]).conj();
                    ge[cb][kb] += pbar * e[ca][ka].conj();
                }
            }
            *work += (2 * v * v * hbar.len()) as f64;
        };
        let pp_back = |hbar: &[C64], kt: &mut [C64], work: &mut f64| {
            for (t, entries) in pp.iter().enumerate() {
                kt[t] += entries.iter().map(|&(i, v)| hbar[i].conj() * v).sum::<C64>();
                *work += entries.len() as f64;
            }
        };
        // The nodes' Ĥ_PP and far forms (their metric at z₀).
        for j in 1..positions {
            pp_back(&abar[j], &mut kt[j], &mut work);
            for (x, fb) in far.iter().enumerate() {
                let z0 = cache.far[x][cache.at_far[x].unwrap()].z0;
                let mut mbar = Buf::zeros(nn);
                mbar.iter_mut().zip(abar[j].iter()).for_each(|(m, z)| *m = *z * z0);
                far_back(x, fb, &nodes[j - 1].at.e, &abar[j], &mbar, &mut ge[j], &mut work);
            }
        }
        // The near blocks' compressions: the middle's (its Bloch part Ṽ†Z the series' last H's, taken here), the nodes'.
        for (x, nb) in near.iter_mut().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let s = nb.s;
            let splits = nb.splits.len();
            let (t, ends) = (&basis.f.t, &basis.f.ends);
            for b in [&mut nb.ztbar, &mut nb.vbar, &mut nb.kbar, &mut nb.zbar, &mut nb.bucket, &mut nb.fbar, &mut nb.zdbar, &mut nb.zdtbar, &mut nb.kedbar, &mut nb.kdebar, &mut nb.kddbar] {
                b.fill(C64::ZERO);
            }
            let mut x_buf = Buf::zeros(s * n);
            compress_back(&nb.kappa, t, ends, &basis.part, &nb.ztot, &nb.vtot, &nb.y, &abar[0], &mut nb.ztbar, &mut nb.vbar, &mut nb.kbar, &mut x_buf, s, n, &mut work);
            // The metric: Z̄ += 2 Z X̄.
            for i in 0..s {
                for p in 0..n {
                    let zp = nb.ztot[i * n + p] * 2.0;
                    if zp != C64::ZERO {
                        crate::linalg::mul_add(false, &mut nb.ztbar[i * n..i * n + n], zp, &xbar[p * n..p * n + n]);
                    }
                }
            }
            work += (s * n * n) as f64;
            if nb.nd > 0 {
                copies_compress_back(s, nb.nd, &nb.ked, &nb.kde, &nb.kdd, &nb.ztot, &nb.zdtot, &abar[0], Some(&xbar), &mut nb.ztbar, &mut nb.zdtbar, &mut nb.kedbar, &mut nb.kdebar, &mut nb.kddbar, n, &mut work);
            }
            if nb.copied_in {
                // T̄_C = T_C (z₀(Ā + Ā†) + 2X̄).
                let w: Vec<C64> = (0..nn).map(|i| (abar[0][i] + abar[0][(i % n) * n + i / n].conj()) * basis.z0 + xbar[i] * 2.0).collect();
                let mut tbar = Buf::zeros(nb.nc * n);
                for k in 0..nb.nc {
                    for p in 0..n {
                        crate::linalg::mul_add(false, &mut tbar[k * n..k * n + n], nb.tc[k * n + p], &w[p * n..p * n + n]);
                    }
                }
                copies_back(&nb.copies, &nb.groups, basis, &mid.e, &nb.ztot, &tbar, &mut nb.ztbar, &mut ge[0], s, n, &mut work);
            }
            for j in 1..positions {
                let (mut zb, mut vb, mut y) = (Buf::zeros(s * n), Buf::zeros(s * n), Buf::zeros(s * n));
                let (zn, vn) = (&nb.znode[j - 1], &nb.vnode[j - 1]);
                // K̂Z(s) (`compress` left the last node's alone).
                for i in 0..s {
                    for c in 0..n {
                        y[i * n + c] = nb.kappa[i] * zn[i * n + c];
                    }
                    for l in i + 1..ends[i] {
                        crate::linalg::mul_add(false, &mut y[i * n..i * n + n], t[(i, l)], &zn[l * n..l * n + n]);
                    }
                }
                compress_back(&nb.kappa, t, ends, &basis.part, zn, vn, &y, &abar[j], &mut zb, &mut vb, &mut nb.kbar, &mut x_buf, s, n, &mut work);
                if nb.copied_in {
                    let w: Vec<C64> = (0..nn).map(|i| (abar[j][i] + abar[j][(i % n) * n + i / n].conj()) * basis.z0).collect();
                    let tc = &nb.tcnode[j - 1];
                    let mut tbar = Buf::zeros(nb.nc * n);
                    for k in 0..nb.nc {
                        for p in 0..n {
                            crate::linalg::mul_add(false, &mut tbar[k * n..k * n + n], tc[k * n + p], &w[p * n..p * n + n]);
                        }
                    }
                    copies_back(&nb.copies, &nb.groups, basis, &nodes[j - 1].at.e, zn, &tbar, &mut zb, &mut ge[j], s, n, &mut work);
                }
                for (ia, &(cc, k)) in nb.vectors.iter().enumerate() {
                    ge[j][cc][k] += vb.iter().zip(&basis.vectors[ia]).map(|(a, b)| a.conj() * *b).sum::<C64>();
                }
                let nd = nb.nd;
                let mut zdb = Buf::zeros(nd * n);
                if nd > 0 {
                    let (mut kedb, mut kdeb) = (Buf::zeros(s * nd), Buf::zeros(nd * s));
                    copies_compress_back(s, nd, &nb.kednode[j - 1], &nb.kdenode[j - 1], &nb.kdd, zn, &nb.zdnode[j - 1], &abar[j], None, &mut zb, &mut zdb, &mut kedb, &mut kdeb, &mut nb.kddbar, n, &mut work);
                    copies_light(nb, basis, &kedb, &kdeb, &[], &mut ge[j], None);
                }
                for (sp, &(cc, jj)) in nb.splits.iter().enumerate() {
                    let zl = &nb.z[sp * s * n..(sp + 1) * s * n];
                    let rho = nb.ratio[j - 1][sp];
                    crate::linalg::mul_add(false, &mut nb.zbar[sp * s * n..(sp + 1) * s * n], rho.conj(), &zb);
                    // ρ = f(s)/f(m): Re(ḡ dρ) → ḡ_f(s) += ḡ/f(m), ḡ_f(m) −= ḡ ρ/f(m).
                    let mut g: C64 = zb.iter().zip(zl).map(|(a, b)| a.conj() * *b).sum();
                    if nd > 0 {
                        let zdl = &nb.zd[sp * nd * n..(sp + 1) * nd * n];
                        g += zdb.iter().zip(zdl).map(|(a, b)| a.conj() * *b).sum::<C64>();
                        crate::linalg::mul_add(false, &mut nb.zdbar[sp * nd * n..(sp + 1) * nd * n], rho.conj(), &zdb);
                    }
                    let fm = mid.f[cc][jj];
                    gf[j][cc][jj] += g / fm;
                    gf[0][cc][jj] -= g * rho / fm;
                }
                work += ((nb.vectors.len() + 2 * splits) * s * n + s * s * n / 2) as f64;
            }
        }
        // The series, the last sweep first: h̄ the cotangent of the Bloch generator the sweep was solved at.
        let mut hb = abar[0].clone();
        let mut hbase_bar = Buf::zeros(nn);
        let mut cbar = Buf::zeros(n);
        let mut dbar = Buf::zeros(nn);
        let mut acc = Buf::zeros(n);
        for k in (0..sweeps).rev() {
            // h^{k+1} = hbase + Σ Ṽ†Z^k (the last sweep's Bloch part taken with the compression's).
            hbase_bar.iter_mut().zip(&hb).for_each(|(a, b)| *a += *b);
            cbar.fill(C64::ZERO);
            dbar.fill(C64::ZERO);
            let delta = &deltas[k * nn..(k + 1) * nn];
            for (x, nb) in near.iter_mut().enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let splits = nb.splits.len();
                let (t, ends) = (&basis.f.t, &basis.f.ends);
                if k + 1 < sweeps {
                    let ztot = &nb.ztots[k * s * n..(k + 1) * s * n];
                    for i in 0..s {
                        for p in 0..n {
                            let v = nb.vtot[i * n + p];
                            if v != C64::ZERO {
                                crate::linalg::mul_add(false, &mut nb.ztbar[i * n..i * n + n], v, &hb[p * n..p * n + n]);
                            }
                            nb.vbar[i * n + p] += (0..n).map(|j| ztot[i * n + j] * hb[p * n + j].conj()).sum::<C64>();
                        }
                    }
                    work += (2 * s * n * n) as f64;
                }
                let nd = nb.nd;
                for sp in 0..splits {
                    let at = (k * splits + sp) * s * n;
                    let (zk, inv) = (&nb.zs[at..at + s * n], &nb.invs[at..at + s * n]);
                    let lam = &mut nb.lam;
                    let zbl = &mut nb.zbar[sp * s * n..(sp + 1) * s * n];
                    let bucket = &mut nb.bucket[sp * s * n..(sp + 1) * s * n];
                    // The excited rows' cotangent of this sweep's Z_l (the held rows' to the first sweep), and the exact
                    // copies' Λ_D with it (`copies_back_solve`: K̂_DE†Λ_D onto the excited rows' cotangent).
                    let mut zeb = Buf::zeros(s * n);
                    (0..s * n).for_each(|x| zeb[x] = nb.ztbar[x] + zbl[x] + if k == 0 { bucket[x] } else { C64::ZERO });
                    let mut lamd = Buf::zeros(nd * n);
                    if nd > 0 {
                        let dat = (k * splits + sp) * nd * n;
                        let mut zdb = Buf::zeros(nd * n);
                        (0..nd * n).for_each(|x| zdb[x] = nb.zdtbar[x] + nb.zdbar[sp * nd * n + x]);
                        let (pat, cat) = ((k * splits + sp) * n * nd * s, (k * splits + sp) * n * nd * nd);
                        copies_back_solve(s, nd, &nb.kde, &mut zeb, &zdb, &nb.pjs[pat..pat + n * nd * s], &nb.cinvs[cat..cat + n * nd * nd], &mut lamd, n, &mut work);
                        let (zdk, zek) = (&nb.zds[dat..dat + nd * n], zk);
                        for a in 0..nd {
                            for j in 0..n {
                                cbar[j] -= lamd[a * n + j] * zdk[a * n + j].conj();
                            }
                            for e2 in 0..nd {
                                nb.kddbar[a * nd + e2] += (0..n).map(|j| lamd[a * n + j] * zdk[e2 * n + j].conj()).sum::<C64>();
                            }
                            for i in 0..s {
                                nb.kdebar[a * s + i] += (0..n).map(|j| lamd[a * n + j] * zek[i * n + j].conj()).sum::<C64>();
                            }
                        }
                        work += (nd * n * (2 + nd + s)) as f64;
                    }
                    // Λ: the adjoint triangular solve, the rows ascending; the held rows' cotangents to the first sweep.
                    for i in 0..s {
                        acc.fill(C64::ZERO);
                        for m in (0..i).filter(|&m| ends[m] > i) {
                            let cm = t[(m, i)].conj();
                            if cm != C64::ZERO {
                                crate::linalg::mul_add(false, &mut acc, cm, &lam[m * n..m * n + n]);
                                work += n as f64;
                            }
                        }
                        if k == 0 || nb.active[i] {
                            for j in 0..n {
                                lam[i * n + j] = (acc[j] + zeb[i * n + j]) * inv[i * n + j].conj();
                            }
                        } else {
                            for j in 0..n {
                                bucket[i * n + j] += acc[j] + zeb[i * n + j];
                                lam[i * n + j] = C64::ZERO;
                            }
                        }
                    }
                    if nd > 0 {
                        let dat = (k * splits + sp) * nd * n;
                        let zdk = &nb.zds[dat..dat + nd * n];
                        for i in 0..s {
                            for a in 0..nd {
                                nb.kedbar[i * nd + a] += (0..n).map(|j| lam[i * n + j] * zdk[a * n + j].conj()).sum::<C64>();
                            }
                        }
                        let zdbl = &mut nb.zdbar[sp * nd * n..(sp + 1) * nd * n];
                        zdbl.fill(C64::ZERO);
                        if k > 0 {
                            let prev = &nb.zds[dat - splits * nd * n..dat - splits * nd * n + nd * n];
                            for a in 0..nd {
                                for m in 0..n {
                                    zdbl[a * n + m] = -(0..n).map(|j| lamd[a * n + j] * delta[m * n + j].conj()).sum::<C64>();
                                    let pm = prev[a * n + m].conj();
                                    if pm != C64::ZERO {
                                        crate::linalg::mul_add(true, &mut dbar[m * n..m * n + n], pm, &lamd[a * n..a * n + n]);
                                    }
                                }
                            }
                        }
                        work += (s * nd * n + 2 * nd * n * n) as f64;
                    }
                    // Ṽ_l = f_l Ỹ_l; the column energies; K̃ on the parts; Z^{k−1} and Δ^k.
                    nb.fbar[sp] += lam.iter().zip(&basis.beams[sp]).map(|(a, b)| a.conj() * *b).sum::<C64>();
                    for i in (0..s).filter(|&i| k == 0 || nb.active[i]) {
                        for j in 0..n {
                            cbar[j] -= lam[i * n + j] * zk[i * n + j].conj();
                        }
                        for m in (0..s).filter(|&m| basis.part[m] == basis.part[i]) {
                            nb.kbar[i * s + m] += (0..n).map(|j| lam[i * n + j] * zk[m * n + j].conj()).sum::<C64>();
                        }
                    }
                    zbl.fill(C64::ZERO);
                    if k > 0 {
                        let prev = &nb.zs[at - splits * s * n..at - splits * s * n + s * n];
                        for i in (0..s).filter(|&i| nb.active[i]) {
                            for m in 0..n {
                                zbl[i * n + m] = -(0..n).map(|j| lam[i * n + j] * delta[m * n + j].conj()).sum::<C64>();
                                let pm = prev[i * n + m].conj();
                                if pm != C64::ZERO {
                                    crate::linalg::mul_add(true, &mut dbar[m * n..m * n + n], pm, &lam[i * n..i * n + n]);
                                }
                            }
                        }
                    }
                    work += (s * n * (3 + 2 * n)) as f64;
                }
                nb.ztbar.fill(C64::ZERO);
                nb.zdtbar.fill(C64::ZERO);
            }
            for i in 0..n {
                for j in 0..n {
                    hb[i * n + j] = if i == j { cbar[i] } else { dbar[i * n + j] };
                }
            }
        }
        hbase_bar.iter_mut().zip(&hb).for_each(|(a, b)| *a += *b);
        // The middle's light: Ĥ_PP, the far forms (H̄ the base's, M̄ = z₀Ā + X̄), the splits, Ṽ, K̃.
        pp_back(&hbase_bar, &mut kt[0], &mut work);
        for (x, fb) in far.iter().enumerate() {
            let z0 = cache.far[x][cache.at_far[x].unwrap()].z0;
            let mut mbar = Buf::zeros(nn);
            (0..nn).for_each(|i| mbar[i] = abar[0][i] * z0 + xbar[i]);
            far_back(x, fb, &mid.e, &hbase_bar, &mbar, &mut ge[0], &mut work);
        }
        let mut held = [0.0_f64; 3];
        let dc = self.pair_gradients(mid, proj, pairs);
        let norm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for (x, nb) in near.iter().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let s = nb.s;
            for (sp, &(cc, jj)) in nb.splits.iter().enumerate() {
                gf[0][cc][jj] += nb.fbar[sp];
            }
            for (ia, &(cc, k)) in nb.vectors.iter().enumerate() {
                ge[0][cc][k] += nb.vbar.iter().zip(&basis.vectors[ia]).map(|(a, b)| a.conj() * *b).sum::<C64>();
            }
            for (ti, &tt) in nb.terms.iter().enumerate() {
                let w = &basis.within[ti];
                kt[0][tt] += (0..s * s).filter(|&i| w[i] != C64::ZERO).map(|i| nb.kbar[i].conj() * w[i]).sum::<C64>();
            }
            if nb.nd > 0 {
                copies_light(nb, basis, &nb.kedbar, &nb.kdebar, &nb.kddbar, &mut ge[0], Some(&mut kt[0]));
            }
            work += ((nb.splits.len() + nb.vectors.len()) * s * n + nb.terms.len() * s * s / 2) as f64;
            // What the kept R₀ (at its z₀) leaves: δK = Ĥ_SC δR₀ Ĥ_CS, ‖δR₀‖ ≤ ‖R₀‖²‖δ(Ĥ_CC − z₀)‖.
            if nb.nc > 0 {
                let dressing: f64 = nb.terms.iter().enumerate().map(|(ti, &tt)| mid.coef[tt].abs() * basis.dress[ti]).sum();
                let z0g = self.centre_gradient(&nb.centre, pp, &dc, n);
                for a in 0..3 {
                    let change = z0g[a].abs() + nb.terms.iter().enumerate().map(|(ti, &tt)| dc[tt][a].abs() * basis.copied[ti]).sum::<f64>();
                    held[a] += basis.r0 * change * norm(&nb.kbar) * dressing;
                }
            }
        }
        let (hb_norm, ab_norm, xb_norm) = (norm(&hbase_bar), abar.iter().map(|a| norm(a)).sum::<f64>(), norm(&xbar));
        for (x, fb) in far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            let z0g = self.centre_gradient(&fb.centre, pp, &dc, n);
            let (hf, mf) = (norm(&fb.hf), norm(&fb.mf));
            for a in 0..3 {
                let change = z0g[a].abs() + fb.terms.iter().enumerate().map(|(ti, &tt)| dc[tt][a].abs() * basis.copied[ti]).sum::<f64>();
                held[a] += basis.r0 * change * (hb_norm * hf + 2.0 * (ab_norm * basis.z0.abs() + xb_norm) * mf) + z0g[a].abs() * mf * ab_norm;
            }
        }
        let mut force = [0.0; 3];
        for j in 0..positions {
            let at = if j == 0 { &*mid } else { &nodes[j - 1].at };
            let g = self.contracted(at, proj, pairs, &gf[j], &ge[j], &kt[j]);
            for a in 0..3 {
                force[a] -= g[a];
            }
            work += (self.light.beams.len() * 3 + pairs.len() * 2) as f64;
        }
        Force { force, held, work }
    }

    /// The absorbed light's moments of the state ψ at the step's middle (photon.rs # The absorbed light's momentum, its
    /// moments' formula): per block (its photons one frequency) and landing set (`sets`, the ground states' partition) a
    /// class, u_q = C_q ε S⁻¹ψ -- ε = VZ of a near block, R₀Σ_a e_a U_a of a far one -- with its whole gradient ∇u, the
    /// step's tangent along each lab axis: every split's response's change by the same sweeps as the step (its field's,
    /// and its own through Ĥ_PP, K̃ on the decomposition's parts, Ṽ and the Bloch generators), and S⁻¹'s (S dS + dS S = dX
    /// by the fixed point of `inverse_root_back`). The jumps carry each class's mean up to one absorbed photon
    /// (`Photon::kick`'s rule, k_c the class's largest beam wave number): the rest is the force; 2D = Σ_c M2_c − W_c b_c b_cᵀ
    /// of the kicks. After `prototype` on the same cache.
    pub(crate) fn prototype_photons(&self, cache: &Cache, ws: &Workspace, psi: &[C64], sets: &[Vec<usize>]) -> Photons {
        let n = ws.n;
        let nn = n * n;
        let mut work = 0.0;
        let (mid, step, pairs, pp, proj) = (&ws.mid, &ws.step, &ws.pairs, &ws.pp, &ws.proj);
        let normal = &step.normal.data;
        let sweeps = step.sweeps;
        // The light's tangents: per component and vector ∂e, per pair ∂c, per split ∂f.
        let deps = self.component_gradients(mid, proj);
        let de: Vec<Vec<[C64; 3]>> = self.components.iter().map(|c| deps[c.part.frequency].iter().map(|d| if c.part.counter { d.map(|z| z.conj()) } else { *d }).collect()).collect();
        let dc = self.pair_gradients(mid, proj, pairs);
        // Ĥ_PP's and the far forms' (and the far metric's) tangents.
        let mut dhbase: Vec<Buf> = (0..3).map(|_| Buf::zeros(nn)).collect();
        let mut dmetric: Vec<Buf> = (0..3).map(|_| Buf::zeros(nn)).collect();
        for (t, entries) in pp.iter().enumerate() {
            for a in 0..3 {
                for &(i, v) in entries {
                    dhbase[a][i] += dc[t][a] * v;
                }
            }
            work += (3 * entries.len()) as f64;
        }
        for (x, fb) in ws.far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            let v = fb.vectors.len();
            for ia in 0..v {
                for ib in 0..v {
                    let ((ca, ka), (cb, kb)) = (fb.vectors[ia], fb.vectors[ib]);
                    for a in 0..3 {
                        let dp = de[ca][ka][a].conj() * mid.e[cb][kb] + mid.e[ca][ka].conj() * de[cb][kb][a];
                        crate::linalg::mul_add(false, &mut dhbase[a], dp, &basis.f[ia * v + ib]);
                        crate::linalg::mul_add(false, &mut dmetric[a], dp, &basis.g[ia * v + ib]);
                    }
                }
            }
            work += (6 * v * v * nn) as f64;
        }
        // Per near block and axis: ∂K̃ on the parts, ∂Ṽ; per split and axis the response's tangent (the sweeps').
        struct Tangent {
            dk: [Buf; 3],
            dv: [Buf; 3],
            dz: Vec<[Buf; 3]>,
            dztot: [Buf; 3],
            // The exact copies': ∂K̂_ED, ∂K̂_DE, ∂K̂_DD, per split their response's tangent and summed.
            dked: [Buf; 3],
            dkde: [Buf; 3],
            dkdd: [Buf; 3],
            ddz: Vec<[Buf; 3]>,
            ddztot: [Buf; 3],
        }
        let mut tangents: Vec<Tangent> = ws
            .near
            .iter()
            .enumerate()
            .map(|(x, nb)| {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let dk = std::array::from_fn(|a| {
                    let mut m = Buf::zeros(s * s);
                    for (ti, &tt) in nb.terms.iter().enumerate() {
                        crate::linalg::mul_add(false, &mut m, dc[tt][a], &basis.within[ti]);
                    }
                    m
                });
                let dv = std::array::from_fn(|a| {
                    let mut m = Buf::zeros(s * n);
                    for (ia, &(cc, k)) in nb.vectors.iter().enumerate() {
                        crate::linalg::mul_add(false, &mut m, de[cc][k][a], &basis.vectors[ia]);
                    }
                    m
                });
                work += (3 * (nb.terms.len() * s * s + nb.vectors.len() * s * n)) as f64;
                let nd = nb.nd;
                let (mut dked, mut dkde, mut dkdd): ([Buf; 3], [Buf; 3], [Buf; 3]) = (std::array::from_fn(|_| Buf::zeros(s * nd)), std::array::from_fn(|_| Buf::zeros(nd * s)), std::array::from_fn(|_| Buf::zeros(nd * nd)));
                for a in 0..3 {
                    for (xx, &(g, k)) in nb.exact.iter().enumerate() {
                        let d = de[g][k][a];
                        for i in 0..s {
                            for e2 in 0..nd {
                                let y = basis.couple[xx][i * nd + e2];
                                dked[a][i * nd + e2] += d * y;
                                dkde[a][e2 * s + i] += (d * y).conj();
                            }
                        }
                    }
                    for (ti, &tt) in nb.terms.iter().enumerate() {
                        if nd > 0 {
                            crate::linalg::mul_add(false, &mut dkdd[a], dc[tt][a], &basis.ddterms[ti]);
                        }
                    }
                }
                Tangent {
                    dk,
                    dv,
                    dz: (0..nb.splits.len()).map(|_| std::array::from_fn(|_| Buf::zeros(s * n))).collect(),
                    dztot: std::array::from_fn(|_| Buf::zeros(s * n)),
                    dked,
                    dkde,
                    dkdd,
                    ddz: (0..nb.splits.len()).map(|_| std::array::from_fn(|_| Buf::zeros(nd * n))).collect(),
                    ddztot: std::array::from_fn(|_| Buf::zeros(nd * n)),
                }
            })
            .collect();
        let mut dh: Vec<Buf> = dhbase.iter().map(|x| Buf::copy(x)).collect();
        let mut rhs = vec![];
        for k in 0..sweeps {
            let delta = &ws.deltas[k * nn..(k + 1) * nn];
            for (x, (nb, tg)) in ws.near.iter().zip(tangents.iter_mut()).enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let splits = nb.splits.len();
                let (t, ends) = (&basis.f.t, &basis.f.ends);
                rhs.resize(s * n, C64::ZERO);
                for (sp, &(cc, jj)) in nb.splits.iter().enumerate() {
                    let at = (k * splits + sp) * s * n;
                    let (zk, inv) = (&ws.near[x].zs[at..at + s * n], &nb.invs[at..at + s * n]);
                    let l = self.components[cc].beams[jj];
                    let y = &basis.beams[sp];
                    for a in 0..3 {
                        let df = self.components[cc].factor(mid.amplitudes[l].gradient[a]);
                        let dz = &mut tg.dz[sp][a];
                        for i in (0..s).filter(|&i| k == 0 || nb.active[i]) {
                            for j in 0..n {
                                rhs[i * n + j] = df * y[i * n + j] - zk[i * n + j] * dh[a][j * n + j];
                            }
                            // ∂K̃ Z on the parts.
                            for m in (0..s).filter(|&m| basis.part[m] == basis.part[i]) {
                                let km = tg.dk[a][i * s + m];
                                if km != C64::ZERO {
                                    crate::linalg::mul_add(false, &mut rhs[i * n..i * n + n], km, &zk[m * n..m * n + n]);
                                }
                            }
                            if k > 0 {
                                // −(∂Z^{k−1} Δ + Z^{k−1} ∂Δ).
                                let prev = &ws.near[x].zs[at - splits * s * n..at - splits * s * n + s * n];
                                for m in 0..n {
                                    let (dzm, zm) = (dz[i * n + m], prev[i * n + m]);
                                    if dzm != C64::ZERO {
                                        crate::linalg::mul_add(true, &mut rhs[i * n..i * n + n], dzm, &delta[m * n..m * n + n]);
                                    }
                                    if zm != C64::ZERO {
                                        for j in (0..n).filter(|&j| j != m) {
                                            rhs[i * n + j] -= zm * dh[a][m * n + j];
                                        }
                                    }
                                }
                            }
                        }
                        let nd = nb.nd;
                        let dat = (k * splits + sp) * nd * n;
                        if nd > 0 {
                            let zdk = &ws.near[x].zds[dat..dat + nd * n];
                            for i in 0..s {
                                for e2 in 0..nd {
                                    let y = tg.dked[a][i * nd + e2];
                                    if y != C64::ZERO {
                                        crate::linalg::mul_add(false, &mut rhs[i * n..i * n + n], y, &zdk[e2 * n..e2 * n + n]);
                                    }
                                }
                            }
                        }
                        for i in (0..s).rev().filter(|&i| k == 0 || nb.active[i]) {
                            for l2 in i + 1..ends[i] {
                                crate::linalg::mul_add(false, &mut rhs[i * n..i * n + n], t[(i, l2)], &dz[l2 * n..l2 * n + n]);
                            }
                            for j in 0..n {
                                dz[i * n + j] = rhs[i * n + j] * inv[i * n + j];
                            }
                        }
                        if nd > 0 {
                            // The copies' rows: −(∂Z_D^{k−1}Δ + Z_D^{k−1}∂Δ) + ∂K̂_DE Z_E + ∂K̂_DD Z_D − Z_D∘∂c, by Woodbury
                            // on the forward's P_j and C_j⁻¹.
                            let zdk = &ws.near[x].zds[dat..dat + nd * n];
                            let ddz = &mut tg.ddz[sp][a];
                            let mut rd = Buf::zeros(nd * n);
                            for e2 in 0..nd {
                                for j in 0..n {
                                    rd[e2 * n + j] = -zdk[e2 * n + j] * dh[a][j * n + j];
                                }
                                for i in 0..s {
                                    let y = tg.dkde[a][e2 * s + i];
                                    if y != C64::ZERO {
                                        crate::linalg::mul_add(false, &mut rd[e2 * n..e2 * n + n], y, &zk[i * n..i * n + n]);
                                    }
                                }
                                for f2 in 0..nd {
                                    crate::linalg::mul_add(false, &mut rd[e2 * n..e2 * n + n], tg.dkdd[a][e2 * nd + f2], &zdk[f2 * n..f2 * n + n]);
                                }
                                if k > 0 {
                                    let prev = &ws.near[x].zds[dat - splits * nd * n..dat - splits * nd * n + nd * n];
                                    for m in 0..n {
                                        let (dzm, zm) = (ddz[e2 * n + m], prev[e2 * n + m]);
                                        if dzm != C64::ZERO {
                                            crate::linalg::mul_add(true, &mut rd[e2 * n..e2 * n + n], dzm, &delta[m * n..m * n + n]);
                                        }
                                        if zm != C64::ZERO {
                                            for j in (0..n).filter(|&j| j != m) {
                                                rd[e2 * n + j] -= zm * dh[a][m * n + j];
                                            }
                                        }
                                    }
                                }
                            }
                            let (pat, cat) = ((k * splits + sp) * n * nd * s, (k * splits + sp) * n * nd * nd);
                            let mut pj = Buf::copy(&nb.pjs[pat..pat + n * nd * s]);
                            let mut cinv = Buf::copy(&nb.cinvs[cat..cat + n * nd * nd]);
                            let mut out = Buf::zeros(nd * n);
                            copies_solve(s, nd, &nb.ked, &nb.kde, &nb.kdd, basis, &[], 0.0, inv, &rd, dz, &mut out, &mut pj, &mut cinv, true, n, &mut work);
                            ddz.copy_from_slice(&out);
                            work += (nd * n * (s + nd + 2 * n)) as f64;
                        }
                    }
                    work += (3 * s * n * (2 + 2 * n + s / 2)) as f64;
                }
                for a in 0..3 {
                    tg.dztot[a].fill(C64::ZERO);
                    tg.ddztot[a].fill(C64::ZERO);
                    for sp in 0..splits {
                        tg.dztot[a].iter_mut().zip(&tg.dz[sp][a]).for_each(|(o, z)| *o += *z);
                        tg.ddztot[a].iter_mut().zip(&tg.ddz[sp][a]).for_each(|(o, z)| *o += *z);
                    }
                }
            }
            // ∂H^{k+1} = ∂Ĥ_base + Σ (∂Ṽ†Z + Ṽ†∂Z).
            for a in 0..3 {
                dh[a].copy_from_slice(&dhbase[a]);
                for (x, (nb, tg)) in ws.near.iter().zip(&tangents).enumerate() {
                    let s = nb.s;
                    let ztot = &ws.near[x].ztots[k * s * n..(k + 1) * s * n];
                    for i in 0..s {
                        for p in 0..n {
                            let (dvp, vp) = (tg.dv[a][i * n + p].conj(), nb.vtot[i * n + p].conj());
                            if dvp != C64::ZERO {
                                crate::linalg::mul_add(false, &mut dh[a][p * n..p * n + n], dvp, &ztot[i * n..i * n + n]);
                            }
                            if vp != C64::ZERO {
                                crate::linalg::mul_add(false, &mut dh[a][p * n..p * n + n], vp, &tg.dztot[a][i * n..i * n + n]);
                            }
                        }
                    }
                    work += (2 * s * n * n) as f64;
                }
            }
        }
        // ∂X = Σ ∂Z†Z + Z†∂Z (+ the copies' rows', the far metric's); S⁻¹'s tangent dN = −N dS N, S dS + dS S = ∂X.
        let zeta: Vec<C64> = (0..n).map(|i| (0..n).map(|j| normal[i * n + j] * psi[j]).sum()).collect();
        let mut dzeta: Vec<Vec<C64>> = vec![];
        let (mut e_buf, mut p_buf, mut q_buf) = (Buf::zeros(nn), Buf::zeros(nn), Buf::zeros(nn));
        for a in 0..3 {
            let mut dx = Buf::copy(&dmetric[a]);
            for (x, (nb, tg)) in ws.near.iter().zip(&tangents).enumerate() {
                let s = nb.s;
                for i in 0..s {
                    for p in 0..n {
                        let (dzp, zp) = (tg.dztot[a][i * n + p].conj(), nb.ztot[i * n + p].conj());
                        crate::linalg::mul_add(false, &mut dx[p * n..p * n + n], dzp, &nb.ztot[i * n..i * n + n]);
                        crate::linalg::mul_add(false, &mut dx[p * n..p * n + n], zp, &tg.dztot[a][i * n..i * n + n]);
                    }
                }
                work += (2 * s * n * n) as f64;
                for d in 0..nb.nd {
                    for p in 0..n {
                        let (dzp, zp) = (tg.ddztot[a][d * n + p].conj(), nb.zdtot[d * n + p].conj());
                        crate::linalg::mul_add(false, &mut dx[p * n..p * n + n], dzp, &nb.zdtot[d * n..d * n + n]);
                        crate::linalg::mul_add(false, &mut dx[p * n..p * n + n], zp, &tg.ddztot[a][d * n..d * n + n]);
                    }
                }
                if nb.copied_in {
                    let basis = &cache.near[x][cache.at_near[x].unwrap()];
                    let (mut dtc, mut extra) = (Buf::zeros(nb.nc * n), Buf::zeros(nb.nc * n));
                    copies_of(&nb.copies, &nb.groups, basis, &mid.e, &tg.dztot[a], &mut dtc, s, n, &mut work);
                    let de_a: Vec<Vec<C64>> = de.iter().map(|c| c.iter().map(|d| d[a]).collect()).collect();
                    copies_of(&nb.copies, &nb.groups, basis, &de_a, &nb.ztot, &mut extra, s, n, &mut work);
                    dtc.iter_mut().zip(&extra).for_each(|(o, z)| *o += *z);
                    for c in 0..nb.nc {
                        for p in 0..n {
                            crate::linalg::mul_add(false, &mut dx[p * n..p * n + n], dtc[c * n + p].conj(), &nb.tc[c * n..c * n + n]);
                            crate::linalg::mul_add(false, &mut dx[p * n..p * n + n], nb.tc[c * n + p].conj(), &dtc[c * n..c * n + n]);
                        }
                    }
                }
            }
            // S dS + dS S = ∂X (`lyapunov`).
            let mut ds = Buf::zeros(nn);
            lyapunov(&step.metric.data, normal, &dx, n, &mut ds, &mut e_buf, &mut p_buf, &mut work);
            // dζ = −N dS ζ.
            for i in 0..n {
                q_buf[i] = (0..n).map(|j| ds[i * n + j] * zeta[j]).sum();
            }
            dzeta.push((0..n).map(|i| -(0..n).map(|j| normal[i * n + j] * q_buf[j]).sum::<C64>()).collect());
            work += (2 * n * n) as f64;
        }
        // The classes' moments.
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let mut out = Photons { rate: 0.0, force: [0.0; 3], diffusion: [[0.0; 3]; 3], work: 0.0 };
        let class = |u: &[C64], du: &[[C64; 3]], rows: &[usize], absorbed: f64, out: &mut Photons| {
            let (mut w, mut mean, mut second) = (0.0, [0.0; 3], [[0.0; 3]; 3]);
            for &b in rows {
                for q in 0..3 {
                    let (x, d) = (u[q * GROUND + b], &du[q * GROUND + b]);
                    w += x.norm_sqr();
                    for a in 0..3 {
                        mean[a] += (x.conj() * d[a]).im;
                        for c in 0..3 {
                            second[a][c] += (d[a].conj() * d[c]).re;
                        }
                    }
                }
            }
            let square = crate::linalg::dot3(mean, mean);
            let bound = absorbed * w;
            let kick = if square > bound * bound { mean.map(|x| absorbed * x / square.sqrt()) } else if w > 0.0 { mean.map(|x| x / w) } else { [0.0; 3] };
            out.rate += w;
            for a in 0..3 {
                out.force[a] += mean[a] - w * kick[a];
                for c in 0..3 {
                    out.diffusion[a][c] += second[a][c] - w * kick[a] * kick[c];
                }
            }
        };
        let (mut u, mut du) = (Buf::zeros(3 * GROUND), vec![[C64::ZERO; 3]; 3 * GROUND]);
        for (x, (nb, tg)) in ws.near.iter().zip(&tangents).enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let s = nb.s;
            let mut absorbed = 0.0_f64;
            for &(cc, jj) in &nb.splits {
                absorbed = absorbed.max(self.light.beams[self.components[cc].beams[jj]].wave_number);
            }
            let y: Vec<C64> = (0..s).map(|i| (0..n).map(|j| nb.ztot[i * n + j] * zeta[j]).sum()).collect();
            let dy: Vec<Vec<C64>> = (0..3).map(|a| (0..s).map(|i| (0..n).map(|j| tg.dztot[a][i * n + j] * zeta[j] + nb.ztot[i * n + j] * dzeta[a][j]).sum()).collect()).collect();
            for q in 0..3 {
                let e = &basis.emit[q];
                for b in 0..GROUND {
                    let row = &e[b * s..b * s + s];
                    u[q * GROUND + b] = row.iter().zip(&y).map(|(p, z)| *p * *z).sum();
                    du[q * GROUND + b] = std::array::from_fn(|a| row.iter().zip(&dy[a]).map(|(p, z)| *p * *z).sum());
                }
            }
            work += (7 * s * n + 12 * GROUND * s) as f64;
            for set in sets {
                class(&u, &du, set, absorbed, &mut out);
            }
            work += (3 * GROUND * 13) as f64;
        }
        for (x, fb) in ws.far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            u.fill(C64::ZERO);
            du.fill([C64::ZERO; 3]);
            let mut absorbed = 0.0_f64;
            for &(cc, _) in &fb.splits {
                for &l in &self.components[cc].beams {
                    absorbed = absorbed.max(self.light.beams[l].wave_number);
                }
            }
            for (ia, &(cc, k)) in fb.vectors.iter().enumerate() {
                let e = mid.e[cc][k];
                for q in 0..3 {
                    let m = &basis.emitted[ia][q];
                    for b in 0..GROUND {
                        let row = &m[b * n..b * n + n];
                        let v: C64 = row.iter().zip(&zeta).map(|(p, z)| *p * *z).sum();
                        u[q * GROUND + b] += e * v;
                        for a in 0..3 {
                            du[q * GROUND + b][a] += de[cc][k][a] * v + e * row.iter().zip(&dzeta[a]).map(|(p, z)| *p * *z).sum::<C64>();
                        }
                    }
                }
                work += (12 * GROUND * n) as f64;
            }
            for set in sets {
                class(&u, &du, set, absorbed, &mut out);
            }
            work += (3 * GROUND * 13) as f64;
        }
        out.rate /= norm;
        out.force = out.force.map(|x| x / norm);
        out.diffusion = out.diffusion.map(|r| r.map(|x| x / norm));
        out.work = work;
        out
    }

    /// The jumps at the step's middle (`node` None) or at an outer node: per block C_q ε S⁻¹ (GROUND × n, q the emission's
    /// axis), ε the block's excited amplitude there -- a near block's VZ (Z = Σ_l ρ_l Z_l at a node), a far block's
    /// R₀Σ_a e_a U_a --, S the middle's: i(G − G†) = Σ L†L to the rounding at every position (module documentation, # The
    /// jumps).
    pub(crate) fn prototype_jumps(&self, cache: &Cache, ws: &Workspace, node: Option<usize>) -> Vec<[Mat; 3]> {
        let n = ws.n;
        let normal = &ws.step.normal;
        let e = match node {
            None => &ws.mid.e,
            Some(j) => &ws.nodes[j].at.e,
        };
        let mut out = vec![];
        for (x, nb) in ws.near.iter().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let z = match node {
                None => &nb.ztot,
                Some(j) => &nb.znode[j],
            };
            let zm = Mat { rows: nb.s, cols: n, data: z.clone() };
            out.push(std::array::from_fn(|q| Mat { rows: GROUND, cols: nb.s, data: basis.emit[q].clone() }.product(&zm).product(normal)));
        }
        for (x, fb) in ws.far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            out.push(std::array::from_fn(|q| {
                let mut m = Mat::zeros(GROUND, n);
                for (ia, &(cc, k)) in fb.vectors.iter().enumerate() {
                    m.add_scaled(e[cc][k], &Mat { rows: GROUND, cols: n, data: basis.emitted[ia][q].clone() });
                }
                m.product(normal)
            }));
        }
        out
    }
}

impl Workspace {
    /// Per block of the jumps (`Fold::prototype_jumps`' order: the near blocks, then the far ones) the photon's absorbed
    /// wave number k_c its class's kick is bounded by: its sources' beams' largest.
    pub(crate) fn absorbed(&self, fold: &Fold) -> Vec<f64> {
        let of = |splits: &[(usize, usize)]| splits.iter().map(|&(c, j)| fold.light.beams[fold.components[c].beams[j]].wave_number).fold(0.0, f64::max);
        self.near.iter().map(|x| of(&x.splits)).chain(self.far.iter().map(|x| of(&x.splits))).collect()
    }

    /// An outer node's G(s) (`Fold::prototype_nodes`).
    pub(crate) fn node_generator(&self, j: usize) -> &Mat {
        &self.nodes[j].generator
    }

    /// The step's discrete choices (what the force's difference holds fixed): its sweeps, and per near block its rows
    /// going on after the first sweep and whether its copies' rows are in.
    pub(crate) fn choices(&self) -> (usize, Vec<(Vec<bool>, bool)>) {
        (self.step.sweeps, self.near.iter().map(|x| (x.active.clone(), x.copied_in)).collect())
    }

    /// Per block of the fold its response on its Q rows at the last step, x coordinates, in the block's row order (the
    /// tests' comparison with `Fold::at`): a near block's solved rows VZ and its copies' T_C, a far block's R₀B; None for
    /// the blocks the prototype does not hold (no rows).
    pub(crate) fn responses(&self, fold: &Fold, cache: &Cache) -> Vec<Option<Mat>> {
        let n = self.n;
        let mut out: Vec<Option<Mat>> = (0..fold.blocks.len()).map(|_| None).collect();
        let mut w = 0.0;
        for (x, nb) in self.near.iter().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let z = Mat { rows: nb.s, cols: n, data: nb.ztot.clone() };
            let ts = basis.f.from(&z, &mut w);
            let mut rows = Near { tc: vec![C64::ZERO; nb.nc * n], ..nb.clone() };
            copy_rows(&mut rows, basis, &self.mid.e, n, &mut w);
            let (s, nd) = (nb.s, nb.nd);
            out[nb.o] = Some(Mat::from_fn(s + nd + nb.nc, n, |i, j| if i < s { ts[(i, j)] } else if i < s + nd { nb.zdtot[(i - s) * n + j] } else { rows.tc[(i - s - nd) * n + j] }));
        }
        for (x, fb) in self.far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            let b = &fold.blocks[fb.o];
            let rows: Vec<usize> = b.rows.iter().map(|r| if let Row::Excited(x) = *r { x } else { unreachable!() }).collect();
            let mut src = Mat::zeros(fb.m, n);
            for &(c, k) in &fb.vectors {
                let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
                for a in 0..fb.m {
                    for &p in &places {
                        src[(a, p)] += self.mid.e[c][k] * fold.components[c].basis[k][(rows[a], fold.ground[p])];
                    }
                }
            }
            out[fb.o] = Some(basis.resolvent.product(&src));
        }
        out
    }

    /// The blocks the prototype holds: (near: (block, rows s, copies nc, beam splits, terms), far: (block, rows, terms)).
    #[allow(clippy::type_complexity)]
    pub(crate) fn layout(&self) -> (Vec<(usize, usize, usize, usize, usize)>, Vec<(usize, usize, usize)>) {
        (self.near.iter().map(|x| (x.o, x.s, x.nc, x.splits.len(), x.terms.len())).collect(), self.far.iter().map(|x| (x.o, x.m, x.terms.len())).collect())
    }
}
