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
//! A block with solved rows (`Near`): the Schur form V T V† of K_ref + o on its kept levels' rows (# The levels kept and
//! left out; unitary, fold.rs `Factored`: what couples them is one part; the normal parts diagonal); every level's data for
//! the omitted ones' bound; per term the diagonal of V†N_tV and per row and part the norm of the rest; per beam its source in
//! the basis Ỹ_l = V†X_l; per source vector Y_a = V†U_a; per centred copy vector R₀U_a†V (the copies' rows' response);
//! ‖R₀‖. A block whose rows are all centred (`Far`): its response T = R₀B is its field coefficients times kept vectors, so
//! its compression is the pair forms U_a†R₀U_b and U_a†R₀†R₀U_b in e_a*e_b.
//!
//! # A step
//! 1. The light at the point: the beams' amplitudes, the fields' ε, the pairs' coefficients, the components' coefficients
//!    e and beam factors f_l, every beam's Doppler shift w_l, Ĥ_PP.
//! 2. The diagonal moved to first order, κ_i = T_ii − o + Σ_t Re δc_t (V†N_tV)_ii; the rest of the change is read (on the
//!    kept rows; the omitted levels' by their bound).
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
//! # The beams side by side
//! The splits' operators differ by a scalar shift alone, (c_j − w_l − T)⁻¹: nothing of a split's solve is another's, but
//! every product with what they share -- Δ, ∂Δ, ∂K̃ -- and every row of the triangular solves is one for all of them. So
//! a block's splits are laid row by row, (i, l, j) at i·Ln + l·n + j (`Near::z`, `NearBasis::beams`), and the step's
//! tangent's three axes, three sources of the same operators, with them, (i, a, l, j) at i·3Ln + a·Ln + l·n + j
//! (`Tangent`): a sweep is one product on all of them and one triangular solve of rows Ln (3Ln) long -- the same formula,
//! its loops over all splits and axes at once.
//!
//! # The levels kept and left out (DESIGN §1.1, §1.8; BRIEF_PROTO4 items 1, 2)
//! A near block's excited levels are kept or left out; the kept ones are one Schur form (whatever couples them is in the
//! decomposition, no series across them), the omitted ones are neither decomposed nor solved. Their coupling R to the kept
//! rows enters their reading alone (`bound_levels`), a bound read after the first sweep without solving them: per level q,
//! lane l and column j, with g = 1/dist(c_j − w_l, W(K_qq − o)) -- W the numerical range, its Gershgorin rectangle at the
//! reference widened by Σ_t |δc_t| ‖N_t,qq‖ and R₀'s change: ‖(z − K)⁻¹‖ ≤ 1/dist(z, W(K)) for any K, normal or not --,
//! ‖z_q‖ ≤ g (‖W_q‖ + ‖R_qK‖ ‖Z_K‖ + Σ_p∈Q ‖R_qp‖ x_p + Σ_i |Δ_ij| x_i) (the last two by their largest, a contraction), and
//! what its absence leaves of the kept rows, ‖(c − w − K_KK)⁻¹‖ ‖R_Kq‖ ‖z_q‖ with ‖(z − D − N)⁻¹‖ ≤ 1/(δ − ν) (δ the Schur
//! diagonal's distance, ν the strict upper part's norm with what the decomposition and the reuse leave). The reading is
//! `Fold::at`'s omission reading of a node -- the response's share of the column and the potential it carries -- on these
//! bounds, against the kept rows' response (at most the whole column's: the reading only grows). Beyond the share on these
//! lane-by-lane bounds, the lanes summed first (the resolvent at their mean Doppler shift on the exact whole source and
//! reference coupling Σ_a e_a U_a,q + K_qK,ref V Z -- a standing wave's node cancels there --, the spread a remainder), the
//! smaller kept. A block with nothing kept and no explicit state is idle: no response, no work but its bound. Beyond the
//! share levels
//! join, the largest first, and the block is decomposed again at the point; where a decomposition is made its kept levels
//! leave while the bounds read within. At an atom's first step nothing is kept (a block with exact copies keeps every
//! level: they couple all of them through the P layer).
//!
//! # The nodes (DESIGN §4.2)
//! The integrator's outer nodes s = ±τ/2 (`Fold::prototype_nodes`): the decomposition, H, S and every split's response per
//! unit amplitude held over the step, the field turned -- Z_l(s) = (f_l(s)/f_l) Z_l, Ṽ(s) = Σ_a e_a(s) Y_a, Ĥ_PP(s) and
//! the far forms at s exactly --, A(s) by the same compression, G(s) = S⁻¹A(s)S⁻¹.
//!
//! # The force and the step's one derivative (DESIGN §1.3)
//! The step's tangent along the three lab axes (`Fold::prototype_tangent`), by the same sweeps as the step: every split's
//! response's change -- its field's and its own through Ĥ_PP, K̃ on the decomposition's parts (all of K's change, not the
//! diagonal the reuse moves), Ṽ and the Bloch generators, the exact copies' with it -- and S⁻¹'s (S dS + dS S = ∂X,
//! `lyapunov`, a contraction for any metric). The photons take it whole (∇u); the force −∇ Σ_j Re tr(Ḡ_j†G_j) over the
//! step's positions moved together contracts it with the top's cotangents alone (`Fold::prototype_force`: Ā = S⁻¹ḠS⁻¹,
//! X̄ by one `lyapunov`, the compressions' Z̄, Ṽ̄, K̄): one derivative of the step for both, exact (against the difference
//! of the evaluated step to 10⁻⁷). R₀ and z₀ are held as the step holds them; their relative change per µm is read
//! (`Force::held`).
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

/// Entries of Ĥ_PP linear in a component's coefficient (`Workspace::lin`): (component, vector, at e_k, at e_k*).
type Lin = Vec<(usize, usize, Vec<(usize, C64)>, Vec<(usize, C64)>)>;

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
    /// The lanes: the beam splits, then the static one where the block holds explicit states (their columns' source, no
    /// Doppler shift: DESIGN §1.3); the explicit states' P places.
    lanes: usize,
    xs: Vec<usize>,
    /// The step's emission Ê (n × s: Ṽ† on the ground rows, E_X on the explicit ones), the explicit states' dressing through
    /// the centred copies on their rows (n × n), the exact copies' static source (nd × n) and emission into P (n × nd).
    emit: Vec<C64>,
    stat: Vec<C64>,
    hx: Vec<C64>,
    b0d: Vec<C64>,
    exd: Vec<C64>,
    /// The same at the step's outer nodes, with their pairs' coefficients less the decomposition's.
    pnode: [Parts; 2],
    o: usize,
    /// The kept rows of the basis it stands on (`fit`), of all its excited rows `all`; per excited level its rows (of
    /// `all`); the kept levels the next decomposition is to hold (`omit`'s, where they change).
    s: usize,
    all: usize,
    levels: Vec<Vec<usize>>,
    wanted: Option<Vec<usize>>,
    /// Whether the basis it stands on was made at this step (`omit` reads whether kept levels may leave).
    fresh: bool,
    /// The step's R₀ change factor ε_R (the reuse's and the omission bound's).
    epsr: f64,
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
    /// Per row the explicit source's dressing's R₀ change (an absolute source error on the explicit columns).
    rhox: Vec<f64>,
    shifts: Vec<f64>,
    vt: Vec<C64>,
    vtot: Vec<C64>,
    z: Vec<C64>,
    ztot: Vec<C64>,
    zold: Vec<C64>,
    rhs: Vec<C64>,
    prod: Vec<C64>,
    bloch: Vec<C64>,
    tc: Vec<C64>,
    y: Vec<C64>,
    err: Vec<f64>,
    norms: Vec<f64>,
    active: Vec<bool>,
    q: Vec<f64>,
    reading: f64,
    // Each sweep's, for the force (`Fold::prototype_force`): per split its diagonal's inverse and its response, and the
    // response summed over the splits.
    invs: Vec<C64>,
    zs: Vec<C64>,
    ztots: Vec<C64>,
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
    zdnode: [Vec<C64>; 2],
    kednode: [Vec<C64>; 2],
    kdenode: [Vec<C64>; 2],
}

impl Near {
    /// Nothing kept and no explicit state: no response, only its levels' bound (`bound_levels`).
    fn idle(&self) -> bool {
        self.s == 0 && self.xs.is_empty()
    }

    /// The step's buffers for a basis of `s` kept rows (of `all`: within the capacity the workspace gave them).
    fn fit(&mut self, s: usize, n: usize) {
        let (ln, nd) = (self.lanes * n, self.nd);
        self.s = s;
        self.kappa.resize(s, C64::ZERO);
        self.off.resize(s * s.max(1), 0.0);
        self.rho.resize(s, 0.0);
        self.rhox.resize(s, 0.0);
        for v in [&mut self.vt, &mut self.z, &mut self.rhs, &mut self.prod] {
            v.resize(s * ln, C64::ZERO);
        }
        let [z0, z1] = &mut self.znode;
        let [v0, v1] = &mut self.vnode;
        for v in [&mut self.vtot, &mut self.ztot, &mut self.zold, &mut self.y, &mut self.stat, z0, z1, v0, v1] {
            v.resize(s * n, C64::ZERO);
        }
        self.emit.resize(n * s, C64::ZERO);
        for p in self.pnode.iter_mut() {
            p.emit.resize(n * s, C64::ZERO);
            p.stat.resize(s * n, C64::ZERO);
        }
        let [e0, e1] = &mut self.kednode;
        let [d0, d1] = &mut self.kdenode;
        for v in [&mut self.ked, &mut self.kde, e0, e1, d0, d1] {
            v.resize(s * nd, C64::ZERO);
        }
        self.err.resize(s * n, 0.0);
        self.norms.resize(s + 1, 0.0);
        self.q.resize(s + 1, 0.0);
        self.active.resize(s, true);
    }
}

/// A near block's explicit parts at an outer node (`explicit_parts`): its pairs' coefficients less the decomposition's,
/// the emission, the static source, the explicit states' dressing, the exact copies' static source and emission.
#[derive(Clone)]
struct Parts {
    delta: Vec<C64>,
    emit: Vec<C64>,
    stat: Vec<C64>,
    hx: Vec<C64>,
    b0d: Vec<C64>,
    exd: Vec<C64>,
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
    /// Whether the point leaves the block out (`omit`).
    omitted: bool,
}

/// A near block's decomposition (module documentation, # What is kept).
struct NearBasis {
    at: Vec<C64>,
    weights: Vec<f64>,
    /// The kept levels (of `Near::levels`) and their rows (of the block's excited rows, ascending): the rows of `f`.
    kept: Vec<usize>,
    rows: Vec<usize>,
    f: Factored,
    /// ‖the Schur form's strict upper part‖_F with what it dropped (the kept part's departure from normality, `omit`'s
    /// resolvent bound).
    nu: f64,
    /// Every level's data for the omission bound (`omit`), L levels, at the reference: per level the Gershgorin intervals of
    /// K_qq − o's Hermitian part and of its decay i(K_qq − K_qq†) (a_lo, a_hi, b_lo, b_hi); per term ‖N_t,qq‖_F (T × L);
    /// ‖K_qp‖_F of the reference and per term ‖N_t,qp‖_F ((T + 1) × L × L); per term the dressing's rows' norm on the
    /// level (T × L, R₀'s change); per level the beams' sources' and the source vectors' column norms (L × splits × n,
    /// L × v × n); the static source's and the explicit rows' emission's column norms ((T + 1) × L × n each).
    gersh: Vec<[f64; 4]>,
    lterm: Vec<f64>,
    lpair: Vec<f64>,
    ldress: Vec<f64>,
    lbeam: Vec<f64>,
    lvec: Vec<f64>,
    lstat: Vec<f64>,
    lemit: Vec<f64>,
    /// Per level its rows' source vectors U_a (v × m_q × n) and static source at the reference (m_q × n), and where it is
    /// omitted its reference coupling to the kept rows in the basis, K_qK,ref V (m_q × s): `bound_levels`' tight form.
    lvecs: Vec<Vec<C64>>,
    lstat0: Vec<Vec<C64>>,
    rqk: Vec<Vec<C64>>,
    part: Vec<usize>,
    diag: Vec<C64>,
    norms: Vec<f64>,
    dressed: Vec<f64>,
    copied: Vec<f64>,
    /// Per row the beams' sources side by side, Ỹ_l = V†X_l on row i at i·Ln + l·n (s × Ln: the splits' rows laid
    /// together, module documentation # The beams side by side).
    beams: Vec<C64>,
    vectors: Vec<Vec<C64>>,
    copies: Vec<Vec<C64>>,
    copied_rows: Vec<f64>,
    z0: f64,
    r0: f64,
    /// Per term V†N_tV on the parts' entries (s × s, zero across parts: the tangent's ∂K̃); the emission on the block's rows
    /// C_qV (GROUND × s, the jumps and the photons).
    within: Vec<C64>,
    emit: [Vec<C64>; 3],
    /// The exact copies: per coupling vector (component g, k) V†U_k on (E rows, D rows of g) (ne × nd); what the
    /// decomposition's dressing adds to K̂_ED and K̂_DE at the reference beyond the bare drives (held); K̂_DD at the
    /// reference (less o) and per term its form on the D rows (nd × nd).
    couple: Vec<Vec<C64>>,
    ked0: Vec<C64>,
    kde0: Vec<C64>,
    kdd0: Vec<C64>,
    ddterms: Vec<Vec<C64>>,
    /// The explicit states (module documentation, # The explicit states), at the reference and per term (T stacked): the
    /// static source on the solved rows V†B_SX (s × n, their columns), the emission E_X V (n × s, their rows), their dressing
    /// through the centred copies among themselves (n × n); per row and term the norm of the source's dressing (its R₀'s
    /// change read). The exact copies': their static source and emission less the bare drives (held; nd × n, n × nd), per
    /// coupling vector (g, k) U_k on (X, D of g) (n × nd). Per copy vector R₀(C_g)U_k(X, C_g)† (|C_g| × n, the centred
    /// copies' static rows). The emission's columns C_q on the explicit states (GROUND × n).
    stat: Vec<C64>,
    sterms: Vec<C64>,
    xemit0: Vec<C64>,
    eterms: Vec<C64>,
    hx0: Vec<C64>,
    hterms: Vec<C64>,
    dressedx: Vec<f64>,
    sd0: Vec<C64>,
    xd0: Vec<C64>,
    xcouple: Vec<Vec<C64>>,
    copies_x: Vec<Vec<C64>>,
    xemitc: [Vec<C64>; 3],
}

/// A far block's pair forms (module documentation, # What is kept): per pair of source vectors (a, b) U_a†R₀U_b and
/// U_a†R₀†R₀U_b (n × n), stacked (v² × n², row a·v + b: the forms in e_a*e_b one product); its R₀ (the tests' response) and ‖R₀‖; the forms' size on its rows per term (R₀'s change).
struct FarBasis {
    at: Vec<C64>,
    weights: Vec<f64>,
    f: Vec<C64>,
    g: Vec<C64>,
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
    /// Of `made`, those made because the kept levels changed (`omit`'s).
    pub(crate) grown: usize,
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
    /// What the nodes left out at the point leave (rate units), and their count (near parts, far blocks).
    pub(crate) omitted: f64,
    pub(crate) left: [usize; 2],
    pub(crate) ratio: f64,
    pub(crate) made: usize,
    /// Of `made`, those for a change of the kept levels.
    pub(crate) grown: usize,
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
    /// Ĥ_PP less the forms (the ground states of one frame, an explicit offset's states, less the frame).
    base: Vec<C64>,
    /// The drives between ground and explicit places, per (component, vector): its entries at e_k and at e_k*.
    lin: Lin,
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
    /// The step's tangent (`Fold::prototype_tangent`), its ticks and work by phase (`TANGENT`).
    tangents: Option<Tangents>,
    pub(crate) tangent_ticks: [u64; 6],
    pub(crate) tangent_work: [f64; 6],
    /// The force's ticks by phase: its top (Ā, X̄), the light's terms (Ĥ_PP, the far forms), the compressions' adjoints,
    /// the responses' tangents at the positions, what R₀ and z₀ leave.
    pub(crate) force_ticks: [u64; 5],
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
/// blocks' parts on one share); the others go on.
/// Returns what the held parts leave (ρ). The budget is the share's ρ less what the levels left out leave (`omit`): one
/// share for both.
#[allow(clippy::too_many_arguments)]
fn freeze(near: &mut [Near], cache: &Cache, h: &[C64], c: &[C64], whole: &[f64], order: &mut Vec<(f64, usize, usize)>, budget: f64, n: usize, work: &mut f64) -> f64 {
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
        let ln = nb.lanes * n;
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
                    let row: f64 = nb.z[i * ln + sp * n..i * ln + sp * n + n].iter().map(|z| z.norm_sqr()).sum();
                    sum += row / gap;
                }
                q += sum.sqrt();
            }
            nb.q[pi] = q;
            order.push((q, x, pi));
        }
        *work += (nb.shifts.len() * s * n) as f64;
    }
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut sum = 0.0;
    for &(q, x, pi) in order.iter() {
        if (sum + q * q).sqrt() * most > budget {
            break;
        }
        sum += q * q;
        let nb = &mut near[x];
        let (idx, at, _) = &cache.near[x][cache.at_near[x].unwrap()].f.parts[pi];
        for i in *at..at + idx.len() {
            nb.active[i] = false;
        }
    }
    sum.sqrt() * most
}

/// The distance of z from the rectangle [a_lo − r, a_hi + r] × [−b_hi/2 − r, −b_lo/2 + r] (a level's numerical range
/// widened by r, `bound_levels`), zero inside.
fn outside(z: C64, g: &[f64; 4], r: f64) -> f64 {
    let x = (g[0] - r - z.re).max(z.re - g[1] - r).max(0.0);
    let y = (-0.5 * g[3] - r - z.im).max(z.im + 0.5 * g[2] - r).max(0.0);
    x.hypot(y)
}

/// One block's omitted levels bounded (module documentation, # The levels kept and left out): per level q of `omitted` and
/// column j the bound N_q,j on its response (`sizes`, L × n), δ_q,j on what its absence leaves of the kept rows' (`backs`),
/// and its potential's numerator (`shifts`), at the Bloch generator H (its diagonal the column energies c, the rest Δ).
/// Lane by lane first; with `tight` also on the lanes summed -- the resolvent at the lanes' mean Doppler shift w̄ on the whole
/// source and the reference's coupling to the kept response, Σ_a e_a U_a,q + R_qK,ref V Z (exact vectors: a standing wave's
/// node cancels in them), the lanes' spread a remainder (D_l − D̄ = (w_l − w̄) D_l D̄) --, the smaller kept. Where a bound
/// does not exist (a column energy on a level's widened numerical range, or the omitted levels' own coupling and Δ's not
/// contracting) the level that holds it the most, its largest γ (or β's that do not exist).
#[allow(clippy::too_many_arguments)]
fn bound_levels(nb: &Near, basis: &NearBasis, omitted: &[bool], at: &Placed, h: &[C64], tight: bool, sizes: &mut [f64], backs: &mut [f64], shifts: &mut [f64], n: usize, work: &mut f64) -> Result<(), usize> {
    let nl = nb.levels.len();
    let terms = nb.terms.len();
    let (s, lanes) = (nb.s, nb.lanes);
    let ln = lanes * n;
    sizes[..nl * n].fill(0.0);
    backs[..nl * n].fill(0.0);
    shifts[..nl * n].fill(0.0);
    let q_of: Vec<usize> = (0..nl).filter(|&q| omitted[q]).collect();
    if q_of.is_empty() {
        return Ok(());
    }
    let c: Vec<C64> = (0..n).map(|j| h[j * n + j]).collect();
    // The level pairs' coupling and each level's numerical range's widening at the point: the reference's and the terms'
    // |δc_t| (exact in the light), R₀'s change ε_R Σ_t |c_t| on the dressing's rows.
    let mut pair = basis.lpair[..nl * nl].to_vec();
    let (mut radius, mut held) = (vec![0.0_f64; nl], vec![0.0_f64; nl]);
    for (t, &p) in nb.terms.iter().enumerate() {
        let d = nb.delta[t].abs();
        for q in 0..nl {
            radius[q] += d * basis.lterm[t * nl + q];
            held[q] += nb.epsr * at.coef[p].abs() * basis.ldress[t * nl + q];
        }
        if d > 0.0 {
            for (a, y) in pair.iter_mut().zip(&basis.lpair[(t + 1) * nl * nl..(t + 2) * nl * nl]) {
                *a += d * y;
            }
        }
    }
    for q in 0..nl {
        radius[q] += held[q];
        for p in 0..nl {
            pair[q * nl + p] += held[q];
        }
    }
    *work += (terms * nl * (nl + 2)) as f64;
    // The kept rows' response per lane and column, and the kept part's resolvent bound per lane and column: the Schur form's
    // diagonal's least distance δ less its departure from normality ν (the strict upper part, what it dropped, the reuse's
    // off-diagonal change and R₀'s): ‖(z − D − N)⁻¹‖ ≤ Σ_k ‖N‖^k/δ^{k+1} = 1/(δ − ν).
    let mut zk = vec![0.0_f64; ln];
    for i in 0..s {
        for (u, z) in zk.iter_mut().zip(&nb.z[i * ln..(i + 1) * ln]) {
            *u += z.norm_sqr();
        }
    }
    zk.iter_mut().for_each(|u| *u = u.sqrt());
    let parts = basis.f.parts.len().max(1);
    let nu = basis.nu + (0..s).map(|i| nb.off[i * parts..(i + 1) * parts].iter().sum::<f64>() + nb.rho[i]).fold(0.0, f64::max);
    let kept_resolvent = |z: C64| -> f64 {
        if s == 0 {
            return 0.0;
        }
        let d = (0..s).map(|i| (z - nb.kappa[i]).abs()).fold(f64::INFINITY, f64::min);
        if d > nu { 1.0 / (d - nu) } else { f64::INFINITY }
    };
    let mut gk = vec![0.0_f64; ln];
    for (sp, &wl) in nb.shifts.iter().enumerate() {
        for j in 0..n {
            gk[sp * n + j] = kept_resolvent(c[j] - C64::real(wl));
        }
    }
    let ek: Vec<f64> = (0..n).map(|j| nb.emit[j * s..(j + 1) * s].iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()).collect();
    let dcol: Vec<f64> = (0..n).map(|j| (0..n).filter(|&i| i != j).map(|i| h[i * n + j].abs()).sum()).collect();
    *work += (s * ln + ln * s / 4 + n * s + n * n / 4) as f64;
    // Per omitted level q, lane l and column j: x ≤ g (src + ‖R_qK‖‖Z_K‖ + Σ_p∈Q ‖R_qp‖ x_p + Σ_i |Δ_ij| x_i), g = 1/dist(z,
    // W(K_qq − o)) the numerical range's bound (any matrix, normal or not); the omitted levels' own coupling and Δ's by
    // their largest, x ≤ β + γ X with X = max β/(1 − γ).
    let nq = q_of.len();
    let (mut beta, mut gam, mut gl, mut src) = (vec![0.0_f64; nq * ln], vec![0.0_f64; nq * ln], vec![0.0_f64; nq * ln], vec![0.0_f64; nq * ln]);
    let mut rqk = vec![0.0_f64; nq];
    for (k, &q) in q_of.iter().enumerate() {
        rqk[k] = (0..nl).filter(|&p| !omitted[p]).map(|p| pair[q * nl + p].powi(2)).sum::<f64>().sqrt();
        let rqq: f64 = q_of.iter().filter(|&&p| p != q).map(|&p| pair[q * nl + p]).sum();
        for sp in 0..lanes {
            let beam = sp < nb.splits.len();
            let (wl, fl) = if beam { (nb.shifts[sp], at.f[nb.splits[sp].0][nb.splits[sp].1].abs()) } else { (0.0, 0.0) };
            for j in 0..n {
                let y = if beam {
                    fl * basis.lbeam[(q * nb.splits.len() + sp) * n + j]
                } else {
                    let mut v = basis.lstat[q * n + j];
                    for t in 0..terms {
                        v += nb.delta[t].abs() * basis.lstat[((t + 1) * nl + q) * n + j];
                    }
                    v + held[q]
                };
                // The denominator (c_j − w_l) − (K_qq − o): z against the numerical range of K_qq − o.
                let d = outside(c[j] - C64::real(wl), &basis.gersh[q], radius[q]);
                let g = if d > 0.0 { 1.0 / d } else { f64::INFINITY };
                let at = k * ln + sp * n + j;
                (gl[at], src[at]) = (g, y);
                beta[at] = g * (y + rqk[k] * zk[sp * n + j]);
                gam[at] = g * (rqq + dcol[j]);
            }
        }
        *work += (lanes * n * 3) as f64;
    }
    let mut x = vec![0.0_f64; nq * ln];
    for sp in 0..lanes {
        let (mut gmax, mut bmax) = (0.0_f64, 0.0_f64);
        for k in 0..nq {
            for j in 0..n {
                gmax = gmax.max(gam[k * ln + sp * n + j]);
                bmax = bmax.max(beta[k * ln + sp * n + j]);
            }
        }
        if bmax == 0.0 {
            continue;
        }
        if !(gmax < 1.0 && bmax.is_finite()) {
            let worst = |k: usize| (0..n).map(|j| if beta[k * ln + sp * n + j].is_finite() { gam[k * ln + sp * n + j] } else { f64::INFINITY }).fold(0.0, f64::max);
            let k = (0..nq).max_by(|&a, &b| worst(a).total_cmp(&worst(b))).unwrap();
            return Err(q_of[k]);
        }
        let top = bmax / (1.0 - gmax);
        for k in 0..nq {
            for j in 0..n {
                x[k * ln + sp * n + j] = beta[k * ln + sp * n + j] + gam[k * ln + sp * n + j] * top;
            }
        }
    }
    let mut rkq = vec![0.0_f64; nq];
    for (k, &q) in q_of.iter().enumerate() {
        rkq[k] = (0..nl).filter(|&p| !omitted[p]).map(|p| pair[p * nl + q].powi(2)).sum::<f64>().sqrt();
        for sp in 0..lanes {
            for j in 0..n {
                let v = x[k * ln + sp * n + j];
                sizes[q * n + j] += v;
                if rkq[k] > 0.0 && v > 0.0 {
                    backs[q * n + j] += gk[sp * n + j] * rkq[k] * v;
                }
            }
        }
    }
    if tight {
        // The lanes summed: Z_q,j = D̄ Σ_l r_l + Σ_l (D_l − D̄) r_l, ‖D_l − D̄‖ ≤ |w_l − w̄| g_l ḡ; Σ_l r_l = Σ_a e_a U_a,q (+ the
        // static source) + R_qK Z_tot + Σ_p R_qp z_p − (z_q Δ)_j, its first two exact at the reference, their change by norms.
        let mean = nb.shifts.iter().sum::<f64>() / lanes.max(1) as f64;
        let zt: Vec<f64> = (0..n).map(|j| (0..s).map(|i| nb.ztot[i * n + j].norm_sqr()).sum::<f64>().sqrt()).collect();
        let whole: Vec<f64> = sizes[..nl * n].to_vec();
        for (k, &q) in q_of.iter().enumerate() {
            // A kept level read as if omitted has no reference coupling stored: its lane bound stands.
            if s > 0 && basis.rqk[q].is_empty() {
                continue;
            }
            let m = nb.levels[q].len();
            let mut v = vec![C64::ZERO; m * n];
            for (a, &(cc, kk)) in nb.vectors.iter().enumerate() {
                crate::linalg::mul_add(false, &mut v, at.e[cc][kk], &basis.lvecs[q][a * m * n..(a + 1) * m * n]);
            }
            if !nb.xs.is_empty() {
                v.iter_mut().zip(&basis.lstat0[q]).for_each(|(a, b)| *a += *b);
            }
            if s > 0 && !basis.rqk[q].is_empty() {
                let mut o = Buf::zeros(m * n);
                fused_product_into(&basis.rqk[q], &nb.ztot, (m, s, n), &mut o);
                v.iter_mut().zip(o.iter()).for_each(|(a, b)| *a += *b);
            }
            let dr = (0..nl).filter(|&p| !omitted[p]).map(|p| (pair[q * nl + p] - basis.lpair[q * nl + p]).powi(2)).sum::<f64>().sqrt();
            *work += ((nb.vectors.len() + s) * m * n) as f64;
            for j in 0..n {
                let d = outside(c[j] - C64::real(mean), &basis.gersh[q], radius[q]);
                let gb = if d > 0.0 { 1.0 / d } else { f64::INFINITY };
                let vj = (0..m).map(|i| v[i * n + j].norm_sqr()).sum::<f64>().sqrt();
                let mut stat_d = 0.0;
                if !nb.xs.is_empty() && nb.xs.contains(&j) {
                    stat_d = held[q];
                    for t in 0..terms {
                        stat_d += nb.delta[t].abs() * basis.lstat[((t + 1) * nl + q) * n + j];
                    }
                }
                let (mut spread, mut most) = (0.0, 0.0_f64);
                for sp in 0..lanes {
                    let wl = if sp < nb.splits.len() { nb.shifts[sp] } else { 0.0 };
                    let a = (wl - mean).abs() * gl[k * ln + sp * n + j];
                    spread += a * (src[k * ln + sp * n + j] + rqk[k] * zk[sp * n + j]);
                    most = most.max(a);
                }
                let coupled: f64 = q_of.iter().filter(|&&p| p != q).map(|&p| pair[q * nl + p] * whole[p * n + j]).sum::<f64>() + (0..n).filter(|&i| i != j).map(|i| h[i * n + j].abs() * whole[q * n + i]).sum::<f64>();
                let new = gb * (vj + dr * zt[j] + stat_d + spread + (1.0 + most) * coupled);
                if new < sizes[q * n + j] {
                    sizes[q * n + j] = new;
                    if rkq[k] > 0.0 {
                        // What its absence leaves of the kept rows, likewise: the kept resolvent at the mean on the whole, the
                        // lanes' spread on each lane's bound.
                        let gkb = kept_resolvent(c[j] - C64::real(mean));
                        let lane: f64 = (0..lanes).map(|sp| {
                            let wl = if sp < nb.splits.len() { nb.shifts[sp] } else { 0.0 };
                            (wl - mean).abs() * gk[sp * n + j] * x[k * ln + sp * n + j]
                        }).sum();
                        backs[q * n + j] = backs[q * n + j].min(gkb * rkq[k] * (new + lane));
                    }
                }
            }
            *work += (n * (lanes + nq + n)) as f64;
        }
    }
    // The potential's numerator: |Re(Ê_Q Z_Q)_jj| and its kept rows' change, ‖Ê_q,j‖ N_q,j + ‖Ê_K,j‖ δ_q,j; Ê on a ground
    // column Ṽ† (‖Ṽ_q,j‖ ≤ Σ_a |e_a| ‖Y_a,q,j‖), on an explicit one its emission row (reference and terms).
    for &q in &q_of {
        for j in 0..n {
            let eq = if nb.xs.contains(&j) {
                let mut v = basis.lemit[q * n + j];
                for t in 0..terms {
                    v += nb.delta[t].abs() * basis.lemit[((t + 1) * nl + q) * n + j];
                }
                v + held[q]
            } else {
                nb.vectors.iter().enumerate().map(|(a, &(cc, k))| at.e[cc][k].abs() * basis.lvec[(q * nb.vectors.len() + a) * n + j]).sum()
            };
            shifts[q * n + j] = eq * sizes[q * n + j] + if backs[q * n + j] > 0.0 { ek[j] * backs[q * n + j] } else { 0.0 };
        }
        *work += (n * nb.vectors.len()) as f64;
    }
    Ok(())
}

/// A block's levels' bounds at a point (`bound_levels`): per level whether it is left out, N, δ and the potential's
/// numerator (L × n each).
type Bounds = (Vec<bool>, Vec<f64>, Vec<f64>, Vec<f64>);

/// The omitted levels' reading with the columns' denominators `den` (module documentation, # The levels left out): per
/// column (√(Σ N²) + Σ δ)²/W_j and pot(2 Σ shifts) over every block's omitted levels, the largest (ρ).
fn read_omitted(per: &[Bounds], den: &[f64], thermal: f64, n: usize) -> f64 {
    let pot = |dv: f64| ((1.0 + dv / (2.0 * thermal)).sqrt() - 1.0).powi(2);
    let total: f64 = den.iter().sum::<f64>() + per.iter().map(|p| (0..p.0.len()).filter(|&q| p.0[q]).map(|q| p.1[q * n..(q + 1) * n].iter().map(|x| x * x).sum::<f64>()).sum::<f64>()).sum::<f64>();
    let floor = (crate::units::rounding(n) * total.sqrt()).powi(2);
    let mut most = 0.0_f64;
    for j in 0..n {
        let (mut sq, mut back, mut dv) = (0.0, 0.0, 0.0);
        for (om, sz, bk, sh) in per {
            for q in (0..om.len()).filter(|&q| om[q]) {
                sq += sz[q * n + j].powi(2);
                back += bk[q * n + j];
                dv += sh[q * n + j];
            }
        }
        let size = (sq.sqrt() + back).powi(2);
        if size + den[j] <= floor {
            continue;
        }
        let w = if size > 0.0 { size / den[j] } else { 0.0 };
        most = most.max(w.max(pot(2.0 * dv)));
    }
    most.sqrt()
}

/// The levels and far blocks the point leaves out (module documentation, # The levels left out), read after the first sweep
/// without solving the omitted levels: their bounds (`bound_levels`) read together (`read_omitted`) against each column's
/// whole excited response W_j -- the kept rows' and the far blocks', at most the whole column's, so the reading only
/// grows. Within the share: the far blocks of the smallest reading go too while the whole stays within (their M_F,jj and
/// H_F,jj exact), out of H. Beyond it: levels join, the largest first (its bound's share of its column, the joined ones'
/// bounds counted in the columns as their responses), until the bounds read within; their blocks' `wanted` say the kept
/// levels the next decomposition holds. Returns the ρ left out and whether levels joined.
#[allow(clippy::too_many_arguments)]
fn omit(near: &mut [Near], far: &mut [Far], cache: &Cache, at: &Placed, hbase: &mut [C64], h: &mut [C64], whole: &mut [f64], tally: &mut [usize; 2], share: f64, thermal: f64, n: usize, work: &mut f64) -> (f64, bool) {
    let pot = |dv: f64| ((1.0 + dv / (2.0 * thermal)).sqrt() - 1.0).powi(2);
    let limit = unrate(share);
    let mut per: Vec<Bounds> = vec![];
    let mut grow = false;
    for (x, nb) in near.iter_mut().enumerate() {
        let basis = &cache.near[x][cache.at_near[x].unwrap()];
        let nl = nb.levels.len();
        let omitted: Vec<bool> = (0..nl).map(|q| !basis.kept.contains(&q)).collect();
        let (mut sz, mut bk, mut sh) = (vec![0.0; nl * n], vec![0.0; nl * n], vec![0.0; nl * n]);
        if let Err(q) = bound_levels(nb, basis, &omitted, at, h, false, &mut sz, &mut bk, &mut sh, n, work) {
            // No bound: the level that holds it the most joins.
            let mut kept = basis.kept.clone();
            kept.push(q);
            kept.sort_unstable();
            nb.wanted = Some(kept);
            grow = true;
        }
        per.push((omitted, sz, bk, sh));
    }
    if grow {
        return (f64::INFINITY, true);
    }
    let mut rho = read_omitted(&per, whole, thermal, n);
    *work += (n * per.iter().map(|p| p.0.len()).sum::<usize>()) as f64;
    // Beyond the share on the lanes' bounds: the lanes summed (`bound_levels`' tight form) before any level joins.
    if rho > limit {
        for (x, nb) in near.iter().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let (om, sz, bk, sh) = &mut per[x];
            let _ = bound_levels(nb, basis, om, at, h, true, sz, bk, sh, n, work);
        }
        rho = read_omitted(&per, whole, thermal, n);
    }
    if rho > limit {
        let mut den = whole.to_vec();
        while rho > limit {
            let mut best: Option<(f64, usize, usize)> = None;
            for (x, (om, sz, ..)) in per.iter().enumerate() {
                for q in (0..om.len()).filter(|&q| om[q]) {
                    let key = (0..n).map(|j| sz[q * n + j].powi(2) / (den[j] + sz[q * n + j].powi(2)).max(f64::MIN_POSITIVE)).fold(0.0, f64::max);
                    if best.is_none_or(|b| key > b.0) {
                        best = Some((key, x, q));
                    }
                }
            }
            let Some((_, x, q)) = best else { break };
            for j in 0..n {
                den[j] += per[x].1[q * n + j].powi(2);
            }
            per[x].0[q] = false;
            rho = read_omitted(&per, &den, thermal, n);
        }
        for (x, (nb, (om, ..))) in near.iter_mut().zip(&per).enumerate() {
            let kept: Vec<usize> = (0..om.len()).filter(|&q| !om[q]).collect();
            if kept != cache.near[x][cache.at_near[x].unwrap()].kept {
                nb.wanted = Some(kept);
            }
        }
        return (rho, true);
    }
    // Where a decomposition was made at this step its kept levels may leave, the smallest first, while the bounds (each with
    // the whole kept response, its own in it, and the column less its bound) read within: a new one without them.
    let mut den = whole.to_vec();
    let mut left = false;
    for (x, nb) in near.iter_mut().enumerate() {
        let basis = &cache.near[x][cache.at_near[x].unwrap()];
        if !nb.fresh || nb.nd > 0 {
            continue;
        }
        let nl = nb.levels.len();
        let mut units: Vec<(f64, usize)> = vec![];
        for &q in &basis.kept {
            let mut om = per[x].0.clone();
            om[q] = true;
            let (mut sz, mut bk, mut sh) = (vec![0.0; nl * n], vec![0.0; nl * n], vec![0.0; nl * n]);
            if bound_levels(nb, basis, &om, at, h, false, &mut sz, &mut bk, &mut sh, n, work).is_ok() {
                units.push(((0..n).map(|j| (sz[q * n + j] + bk[q * n + j]).powi(2) / den[j].max(f64::MIN_POSITIVE)).fold(0.0, f64::max), q));
            }
        }
        units.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for (_, q) in units {
            let mut om = per[x].0.clone();
            om[q] = true;
            let (mut sz, mut bk, mut sh) = (vec![0.0; nl * n], vec![0.0; nl * n], vec![0.0; nl * n]);
            if bound_levels(nb, basis, &om, at, h, false, &mut sz, &mut bk, &mut sh, n, work).is_err() {
                continue;
            }
            let d: Vec<f64> = (0..n).map(|j| (den[j] - sz[q * n + j].powi(2)).max(0.0)).collect();
            let mut trial = per.clone();
            trial[x] = (om, sz, bk, sh);
            if read_omitted(&trial, &d, thermal, n) <= limit {
                per = trial;
                den = d;
                left = true;
            }
        }
        if left {
            let kept: Vec<usize> = (0..nl).filter(|&q| !per[x].0[q]).collect();
            if kept != basis.kept {
                nb.wanted = Some(kept);
            }
        }
    }
    if left && near.iter().any(|nb| nb.wanted.is_some()) {
        return (read_omitted(&per, &den, thermal, n), true);
    }
    tally[0] += per.iter().map(|p| p.0.iter().filter(|&&o| o).count()).sum::<usize>();
    // The far blocks of the smallest reading go too while the whole reads within: each its exact M_F,jj and H_F,jj.
    let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
    let mut units: Vec<(f64, usize)> = far.iter().enumerate().map(|(x, fb)| ((0..n).filter(|&j| whole[j] > floor).map(|j| (fb.mf[j * n + j].re / whole[j]).max(pot(2.0 * fb.hf[j * n + j].re.abs()))).fold(0.0, f64::max), x)).collect();
    units.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    for (w, x) in units {
        if (rho * rho + w).sqrt() > limit {
            break;
        }
        rho = (rho * rho + w).sqrt();
        tally[1] += 1;
        let fb = &mut far[x];
        fb.omitted = true;
        for ((a, b), f) in hbase.iter_mut().zip(h.iter_mut()).zip(&fb.hf) {
            *a -= *f;
            *b -= *f;
        }
        for j in 0..n {
            whole[j] -= fb.mf[j * n + j].re;
        }
    }
    whole.iter_mut().for_each(|w| *w = w.max(0.0));
    (rho, false)
}

/// A near block's copies' rows T_C = Σ_a e_a* (R₀U_a†V Z + R₀U_a(X, C)†) into `nb.tc` (module documentation, 7).
fn copy_rows(nb: &mut Near, basis: &NearBasis, e: &[Vec<C64>], n: usize, work: &mut f64) {
    let Near { copies, groups, ztot, tc, s, .. } = nb;
    copies_of(copies, groups, basis, e, ztot, true, tc, *s, n, work);
}

/// The copies' rows T_C = Σ_a e_a* (R₀U_a†V) Z of a response Z (s × n) into `out` (nc × n), with the explicit columns'
/// static rows Σ_a e_a* R₀U_a(X, C)† where `stat` (T_C's whole; without, its part linear in Z).
#[allow(clippy::too_many_arguments)]
fn copies_of(copies: &[(usize, usize, usize)], groups: &[(usize, Vec<usize>)], basis: &NearBasis, e: &[Vec<C64>], z: &[C64], stat: bool, out: &mut [C64], s: usize, n: usize, work: &mut f64) {
    out.fill(C64::ZERO);
    for (cv, &(g, k, gi)) in copies.iter().enumerate() {
        let ek = e[g][k].conj();
        let yc = &basis.copies[cv];
        let rows = &groups[gi].1;
        for (a2, &row) in rows.iter().enumerate() {
            let out = &mut out[row * n..row * n + n];
            if stat {
                crate::linalg::mul_add(false, out, ek, &basis.copies_x[cv][a2 * n..a2 * n + n]);
            }
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
            inner += tb.iter().zip(&basis.copies_x[cv][a2 * n..a2 * n + n]).map(|(t, x)| t.conj() * *x).sum::<C64>();
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

/// The near block's compression of a response Z (s × n) into `a` (n × n): Ê Z (where the emission Ê, n × s, is given: at
/// the middle it is the Bloch generator's, in H) + Z†(K̂Z + B), B the source (s × n: Ṽ and the static lane's), K̂ the kept
/// Schur form with the step's diagonal κ (module documentation, 7); `y` = K̂Z on return (s × n).
#[allow(clippy::too_many_arguments)]
fn compress(kappa: &[C64], t: &Mat, ends: &[usize], z: &[C64], emit: Option<&[C64]>, src: &[C64], y: &mut [C64], a: &mut [C64], s: usize, n: usize, work: &mut f64) {
    // K̂Z = (T + δκ − o)Z.
    y[..s * n].fill(C64::ZERO);
    for i in 0..s {
        for j in 0..n {
            y[i * n + j] += kappa[i] * z[i * n + j];
        }
        for l in i + 1..ends[i] {
            crate::linalg::mul_add(false, &mut y[i * n..i * n + n], t[(i, l)], &z[l * n..l * n + n]);
        }
        *work += ((ends[i] - i) * n) as f64;
    }
    let mut o = Buf::zeros(n * n);
    if let Some(emit) = emit {
        fused_product_into(emit, z, (n, s, n), &mut o);
        a.iter_mut().zip(o.iter()).for_each(|(x, y)| *x += *y);
        *work += (s * n * n) as f64;
    }
    let mut zh = Buf::zeros(n * s);
    adjoint_into(z, s, n, &mut zh);
    let mut w = Buf::copy(&y[..s * n]);
    w.iter_mut().zip(src).for_each(|(x, y)| *x += *y);
    fused_product_into(&zh, &w, (n, s, n), &mut o);
    a.iter_mut().zip(o.iter()).for_each(|(x, y)| *x += *y);
    *work += (s * n * n) as f64;
}

/// The exact copies' explicit parts of a compression: Ê_D Z_D + Z_D†B_D into `a` (Ê_D n × nd, B_D nd × n).
fn explicit_copies(exd: &[C64], b0d: &[C64], zd: &[C64], a: &mut [C64], nd: usize, n: usize, work: &mut f64) {
    let mut o = Buf::zeros(n * n);
    fused_product_into(exd, zd, (n, nd, n), &mut o);
    a.iter_mut().zip(o.iter()).for_each(|(m, x)| *m += *x);
    let mut zdh = Buf::zeros(n * nd);
    adjoint_into(zd, nd, n, &mut zdh);
    fused_product_into(&zdh, b0d, (n, nd, n), &mut o);
    a.iter_mut().zip(o.iter()).for_each(|(m, x)| *m += *x);
    *work += (2 * n * nd * n) as f64;
}

/// A near block's explicit parts at a light (module documentation, # The explicit states), `delta` the pairs' coefficients
/// less the decomposition's: the emission Ê (Ṽ† on the ground rows, E_X,ref + Σ_t δc_t E_t on the explicit ones), and where
/// the block holds explicit states the static lane's source Ỹ₀ = Ỹ₀,ref + Σ_t δc_t S_t (`stat`), their dressing
/// hx = hx_ref + Σ_t δc_t H_t, the exact copies' static source (B_D,ref + Σ_x e_x* U_x(X, D)†) and emission into P
/// (Ê_D,ref + Σ_x e_x U_x(X, D)), the bare drives exact, the dressing through the centred copies of the pairs.
#[allow(clippy::too_many_arguments)]
fn explicit_parts(xs: &[usize], exact: &[(usize, usize)], basis: &NearBasis, e: &[Vec<C64>], delta: &[C64], vtot: &[C64], stat: &mut [C64], emit: &mut [C64], hx: &mut [C64], b0d: &mut [C64], exd: &mut [C64], s: usize, nd: usize, n: usize, work: &mut f64) {
    adjoint_into(vtot, s, n, emit);
    *work += (s * n) as f64 / 4.0;
    if xs.is_empty() {
        return;
    }
    let t = delta.len();
    let mut o = Buf::zeros(s * n);
    fused_product_into(delta, &basis.sterms, (1, t, s * n), stat);
    stat.iter_mut().zip(&basis.stat).for_each(|(a, b)| *a += *b);
    fused_product_into(delta, &basis.eterms, (1, t, n * s), &mut o);
    emit.iter_mut().zip(o.iter().zip(&basis.xemit0)).for_each(|(a, (b, c))| *a += *b + *c);
    fused_product_into(delta, &basis.hterms, (1, t, n * n), hx);
    hx.iter_mut().zip(&basis.hx0).for_each(|(a, b)| *a += *b);
    *work += (t * (2 * s * n + n * n)) as f64;
    if nd > 0 {
        b0d.copy_from_slice(&basis.sd0);
        exd.copy_from_slice(&basis.xd0);
        for (x, &(g, k)) in exact.iter().enumerate() {
            let ek = e[g][k];
            for p in 0..n {
                for d in 0..nd {
                    let y = basis.xcouple[x][p * nd + d];
                    if y != C64::ZERO {
                        exd[p * nd + d] += ek * y;
                        b0d[d * n + p] += (ek * y).conj();
                    }
                }
            }
        }
        *work += (exact.len() * n * nd) as f64;
    }
}

/// The adjoint of `compress` (with the emission) for a cotangent Ā (n × n) of A = ÊZ + Z†(K̂Z + B): Z̄ += Ê†Ā + (K̂Z + B)Ā†
/// + K̂†ZĀ, B̄ += ZĀ, Ê̄ += ĀZ† (n × s), K̃̄ += ZĀZ† on the parts' entries (`parts`, the decomposition's), `y` = K̂Z (as
/// `compress` left it), `x` a buffer (s × n): products (`fused_product`) but for K̂†, triangular.
#[allow(clippy::too_many_arguments)]
fn compress_back(kappa: &[C64], t: &Mat, ends: &[usize], z: &[C64], emit: &[C64], src: &[C64], y: &[C64], abar: &[C64], zbar: &mut [C64], srcbar: &mut [C64], emitbar: &mut [C64], kbar: &mut [C64], x: &mut [C64], s: usize, n: usize, work: &mut f64) {
    let mut adj = Buf::zeros(n * n);
    adjoint_into(abar, n, n, &mut adj);
    // Z̄ += Ê†Ā + (Y + B)Ā†; B̄ += ZĀ (= x); Ê̄ += ĀZ†.
    let mut eh = Buf::zeros(s * n);
    adjoint_into(emit, n, s, &mut eh);
    let mut o = Buf::zeros(s * n);
    fused_product_into(&eh, abar, (s, n, n), &mut o);
    zbar.iter_mut().zip(o.iter()).for_each(|(a, b)| *a += *b);
    let mut w = Buf::copy(&y[..s * n]);
    w.iter_mut().zip(src).for_each(|(a, b)| *a += *b);
    fused_product_into(&w, &adj, (s, n, n), &mut o);
    zbar.iter_mut().zip(o.iter()).for_each(|(a, b)| *a += *b);
    fused_product_into(z, abar, (s, n, n), x);
    srcbar.iter_mut().zip(x.iter()).for_each(|(a, b)| *a += *b);
    let mut zh = Buf::zeros(n * s);
    adjoint_into(z, s, n, &mut zh);
    let mut q = Buf::zeros(n * s);
    fused_product_into(abar, &zh, (n, n, s), &mut q);
    emitbar.iter_mut().zip(q.iter()).for_each(|(a, b)| *a += *b);
    // Z̄ += K̂† (Z Ā): K̂† = conj(κ) on the diagonal and the strict lower part T†.
    for i in 0..s {
        for j in 0..n {
            zbar[i * n + j] += kappa[i].conj() * x[i * n + j];
        }
        for l in i + 1..ends[i] {
            crate::linalg::mul_add(false, &mut zbar[l * n..l * n + n], t[(i, l)].conj(), &x[i * n..i * n + n]);
        }
    }
    // K̃̄ += (Z Ā) Z† (∂K̃ is on the parts' entries).
    fused_product_into(x, &zh, (s, n, s), kbar);
    *work += (s * s * n + 4 * s * n * n) as f64;
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

/// The columns `at..at + width` of the `rows` rows of stride `stride` in `x`, as a rows × width matrix (a split's or an
/// axis's own of the rows laid side by side).
fn gather(x: &[C64], rows: usize, stride: usize, at: usize, width: usize) -> Buf {
    let mut out = Buf::zeros(rows * width);
    for i in 0..rows {
        out[i * width..(i + 1) * width].copy_from_slice(&x[i * stride + at..i * stride + at + width]);
    }
    out
}

/// `gather`'s inverse: the rows × width matrix `x` into the columns `at..at + width` of `out`'s rows of stride `stride`.
fn scatter(x: &[C64], out: &mut [C64], rows: usize, stride: usize, at: usize, width: usize) {
    for i in 0..rows {
        out[i * stride + at..i * stride + at + width].copy_from_slice(&x[i * width..(i + 1) * width]);
    }
}

/// The conjugate transpose of the r × c matrix `a` into `out` (c × r).
fn adjoint_into(a: &[C64], r: usize, c: usize, out: &mut [C64]) {
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j].conj();
        }
    }
}

/// out = a b (r × k, k × c; linalg.rs `fused_product`).
fn fused_product_into(a: &[C64], b: &[C64], dims: (usize, usize, usize), out: &mut [C64]) {
    crate::linalg::fused_product(a, b, dims, out);
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
        // A state's P place: a ground state's, or an explicit excited state's (ng + its index).
        let place = |i: usize| -> Option<usize> {
            if i < GROUND {
                self.ground.iter().position(|&g| g == i)
            } else {
                self.explicit.iter().position(|&x| x == i - GROUND).map(|j| ng + j)
            }
        };
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
                        let (p, q) = (place(i)?, place(j)?);
                        ((i < GROUND) == (j < GROUND) && self.frames[p] == self.frames[q]).then_some((p, q))
                    },
                    n,
                    n,
                );
                (0..n * n).filter(|&x| m.data[x] != C64::ZERO).map(|x| (x, m.data[x])).collect()
            })
            .collect();
        // Ĥ_PP less the forms: the ground states of one frame among themselves, an explicit offset's states among themselves
        // (𝒦 with its decay), less the frame.
        let mut base = vec![C64::ZERO; n * n];
        for p in 0..n {
            for q in (0..n).filter(|&q| self.frames[q] == self.frames[p] && (q < ng) == (p < ng)) {
                let h = if p < ng { self.base_ground[(self.ground[p], self.ground[q])] } else { self.base_excited[(self.explicit[p - ng], self.explicit[q - ng])] };
                base[p * n + q] = h - if p == q { C64::real(self.frame_values[p]) } else { C64::ZERO };
            }
        }
        // The drives between a ground place and an explicit one at its component's offset, linear in the component's
        // coefficients: per (component, basis vector) its entries of Ĥ_PP at e_k (the explicit row) and at e_k* (its column).
        let mut lin: Lin = vec![];
        for (g, comp) in self.components.iter().enumerate() {
            for k in 0..comp.basis.len() {
                let (mut direct, mut conj) = (vec![], vec![]);
                for px in ng..n {
                    let x = self.explicit[px - ng];
                    for q in (0..ng).filter(|&q| self.frames[q].plus(1, &comp.beat) == self.frames[px]) {
                        let v = comp.basis[k][(x, self.ground[q])];
                        if v != C64::ZERO {
                            direct.push((px * n + q, v));
                            conj.push((q * n + px, v.conj()));
                        }
                    }
                }
                if !direct.is_empty() {
                    lin.push((g, k, direct, conj));
                }
            }
        }
        let (mut near, mut far) = (vec![], vec![]);
        for (o, b) in self.blocks.iter().enumerate() {
            let m = b.rows.len();
            if m == 0 {
                if !b.explicit.is_empty() {
                    return Err(format!("块 {o}：显式态的块没有 Q 行"));
                }
                continue;
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
            if s == 0 && !b.explicit.is_empty() {
                return Err(format!("块 {o}：原型不含全居中块里的显式态"));
            }
            // The block's explicit states' P places; a state's row among the solved rows and the explicit ones (the terms').
            let xs: Vec<usize> = b.explicit.clone();
            let ext_of = |x: usize| row_of(x).filter(|&a| a < s).or_else(|| xs.iter().position(|&p| self.explicit[p - ng] == x).map(|j| m + j));
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
                let mut terms = terms_of(&|i, j| if i >= GROUND && j >= GROUND { Some((ext_of(i - GROUND)?, ext_of(j - GROUND)?)) } else { None }, m + xs.len());
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
                // The block's excited levels, each with its rows.
                let mut levels: Vec<(usize, Vec<usize>)> = vec![];
                for (a, r) in b.rows[..ne].iter().enumerate() {
                    if let Row::Excited(x) = *r {
                        let lv = self.excited_level[x];
                        match levels.iter_mut().find(|y| y.0 == lv) {
                            Some(y) => y.1.push(a),
                            None => levels.push((lv, vec![a])),
                        }
                    }
                }
                let levels: Vec<Vec<usize>> = levels.into_iter().map(|x| x.1).collect();
                let z = |x: usize| vec![C64::ZERO; x];
                let lanes = splits.len() + usize::from(!xs.is_empty());
                let sp = lanes;
                near.push(Near {
                    lanes,
                    pnode: std::array::from_fn(|_| Parts { delta: z(terms.len()), emit: z(n * s), stat: z(s * n), hx: z(n * n), b0d: z(nd * n), exd: z(n * nd) }),
                    emit: z(n * s),
                    stat: z(s * n),
                    hx: z(n * n),
                    b0d: z(nd * n),
                    exd: z(n * nd),
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
                    zdnode: [z(nd * n), z(nd * n)],
                    kednode: [z(s * nd), z(s * nd)],
                    kdenode: [z(nd * s), z(nd * s)],
                    o,
                    s,
                    all: ne,
                    levels,
                    wanted: None,
                    fresh: false,
                    epsr: 0.0,
                    nc,
                    value: b.value,
                    centre,
                    terms,
                    delta: vec![],
                    kappa: z(s),
                    off: vec![0.0; s * s],
                    rho: vec![0.0; s],
                    rhox: vec![0.0; s],
                    shifts: vec![0.0; lanes],
                    vt: z(lanes * s * n),
                    vtot: z(s * n),
                    z: z(lanes * s * n),
                    ztot: z(s * n),
                    zold: z(s * n),
                    rhs: z(lanes * s * n),
                    prod: z(lanes * s * n),
                    bloch: z(n * n),
                    tc: z(nc * n),
                    y: z(s * n),
                    err: vec![0.0; s * n],
                    norms: vec![0.0; s + 1],
                    active: vec![true; s],
                    q: vec![0.0; s + 1],
                    reading: 0.0,
                    invs: vec![],
                    zs: vec![],
                    ztots: vec![],
                    ratio: [z(lanes), z(lanes)],
                    znode: [z(s * n), z(s * n)],
                    vnode: [z(s * n), z(s * n)],
                    tcnode: [z(nc * n), z(nc * n)],
                    copied_in: false,
                    splits,
                    xs,
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
                far.push(Far { o, m, centre, splits, terms, delta: vec![], products: vec![C64::ZERO; v * v], hf: vec![C64::ZERO; n * n], mf: vec![C64::ZERO; n * n], reading: 0.0, centring: 0.0, omitted: false, vectors });
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
            lin,
            near,
            far,
            mid: placed,
            proj,
            cs: vec![],
            deltas: vec![],
            nodes,
            node_times: [0.0; 2],
            noded: false,
            tangents: None,
            tangent_ticks: [0; 6],
            tangent_work: [0.0; 6],
            force_ticks: [0; 5],
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
                omitted: 0.0,
                left: [0; 2],
                ratio: 0.0,
                made: 0,
                grown: 0,
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
    fn proto_place(&self, r: [f64; 3], kappa: [f64; 3], at: &mut Placed, pairs: &[Pair], pp: &[Vec<(usize, C64)>], base: &[C64], lin: &Lin, proj: &[Vec<C64>], work: &mut f64) {
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
        for (g, k, direct, conj) in lin {
            let e = at.e[*g][*k];
            for &(i, v) in direct {
                at.hpp[i] += e * v;
            }
            for &(i, v) in conj {
                at.hpp[i] += e.conj() * v;
            }
            *work += (2 * direct.len()) as f64;
        }
    }

    /// A near block's decomposition at the point (module documentation, # What is kept): the kept levels' rows merged into
    /// one Schur form (`Factored::of` on K_KK: whatever couples them is in it), and every level's data for the omitted
    /// ones' bound (`bound_levels`), from the fold's own matrices there (`Fold::lay`, `Fold::effective`).
    #[allow(clippy::too_many_arguments)]
    fn near_basis(&self, nb: &Near, ws: &Workspace, laid: &Laid, e: &Effective, kept: &[usize], work: &mut f64) -> Option<NearBasis> {
        let (n, ne, nc, nd) = (ws.n, nb.all, nb.nc, nb.nd);
        let b = &self.blocks[nb.o];
        let nl = nb.levels.len();
        let mut kr: Vec<usize> = kept.iter().flat_map(|&q| nb.levels[q].iter().copied()).collect();
        kr.sort_unstable();
        let s = kr.len();
        let mut level = vec![0; ne];
        for (q, rows) in nb.levels.iter().enumerate() {
            for &i in rows {
                level[i] = q;
            }
        }
        // The excited rows; the exact copies after them in their own coordinates (P layer).
        let (er, dr): (Vec<usize>, Vec<usize>) = ((0..ne).collect(), (ne..ne + nd).collect());
        let kall = e.k.block(&er, &er);
        let f = Factored::of(&kall.block(&kr, &kr), work)?;
        let mut part = vec![0; s];
        for (pi, (idx, at, _)) in f.parts.iter().enumerate() {
            for a in 0..idx.len() {
                part[at + a] = pi;
            }
        }
        let parts = f.parts.len().max(1);
        let nu = ((0..s).flat_map(|i| (i + 1..s).map(move |j| (i, j))).map(|(i, j)| f.t[(i, j)].norm_sqr()).sum::<f64>().sqrt()) + f.dropped.iter().sum::<f64>();
        let rows: Vec<usize> = b.rows[..ne].iter().map(|r| if let Row::Excited(x) = *r { x } else { unreachable!() }).collect();
        let copy_state = |a: usize| if let Row::Copy { state, .. } = b.rows[ne + nd + a] { state } else { unreachable!() };
        let r0 = &e.resolvent;
        // Per copy component: U_k on (excited rows, C_g) and R₀ on (C_g, C_g); the block's explicit states and U_k on (X, C_g).
        let u_ec = |g: usize, k: usize, cg: &[usize]| Mat::from_fn(ne, cg.len(), |a, c| self.components[g].basis[k][(rows[a], copy_state(cg[c]))]);
        let ng = self.ground.len();
        let nx = nb.xs.len();
        let xall: Vec<usize> = (0..nx).collect();
        let xstates: Vec<usize> = nb.xs.iter().map(|&p| self.explicit[p - ng]).collect();
        let u_xc = |g: usize, k: usize, cg: &[usize]| Mat::from_fn(nx, cg.len(), |a, c| self.components[g].basis[k][(xstates[a], copy_state(cg[c]))]);
        // Each excited state's row, and per ground state the centred copies of it (row, group).
        let mut row_of = [usize::MAX; EXCITED];
        for (a, &x) in rows.iter().enumerate() {
            row_of[x] = a;
        }
        let mut copies_of_state: Vec<Vec<(usize, usize)>> = vec![vec![]; GROUND];
        for (gi, (_, cg)) in nb.groups.iter().enumerate() {
            for &x in cg {
                copies_of_state[copy_state(x)].push((x, gi));
            }
        }
        // A matrix's norm on a level's rows and another's columns (Frobenius).
        let lnorm = |m: &Mat, rq: &[usize], cq: &[usize]| rq.iter().flat_map(|&i| cq.iter().map(move |&j| (i, j))).map(|(i, j)| m[(i, j)].norm_sqr()).sum::<f64>().sqrt();
        let xcol = |jx: usize| nb.xs[jx];
        let terms_n = nb.terms.len();
        let (mut lterm, mut lpair, mut ldress) = (vec![0.0; terms_n * nl], vec![0.0; (terms_n + 1) * nl * nl], vec![0.0; terms_n * nl]);
        let (mut lstat, mut lemit) = (vec![0.0; (terms_n + 1) * nl * n], vec![0.0; (terms_n + 1) * nl * n]);
        let mut terms = vec![];
        for (ti, &t) in nb.terms.iter().enumerate() {
            let (w, k, kk) = ws.pairs[t];
            let u = &self.light.frequencies[w].basis;
            let mut form = Mat::zeros(ne, ne);
            let (mut form_sx, mut form_xs) = (Mat::zeros(ne, nx), Mat::zeros(nx, ne));
            let mut cc = Mat::zeros(nc, nc);
            for (p, q, entries) in &self.entries[w] {
                let cu = u[k][*p].conj() * u[kk][*q];
                for &(i, j, v) in entries {
                    if i >= GROUND && j >= GROUND {
                        let (a, c) = (row_of[i - GROUND], row_of[j - GROUND]);
                        let (xa, xc) = (xstates.iter().position(|&x| x == i - GROUND), xstates.iter().position(|&x| x == j - GROUND));
                        if a != usize::MAX && c != usize::MAX {
                            form[(a, c)] += cu * v;
                        } else if let (true, Some(xc)) = (a != usize::MAX, xc) {
                            form_sx[(a, xc)] += cu * v;
                        } else if let (Some(xa), true) = (xa, c != usize::MAX) {
                            form_xs[(xa, c)] += cu * v;
                        }
                    } else if i < GROUND && j < GROUND {
                        // The copies' own rows (one component's copies among themselves).
                        for &(x, gx) in &copies_of_state[i] {
                            for &(y, gy) in &copies_of_state[j] {
                                if gx == gy {
                                    cc[(x, y)] += cu * v;
                                }
                            }
                        }
                    }
                }
            }
            // The dressing: co-rotating e_k e_kk* = ε_k ε_kk* (the pair (w, kk, k)), counter-rotating ε_k* ε_kk (the pair (w, k, kk)).
            let mut dressing = Mat::zeros(ne, ne);
            let (mut dress_sx, mut dress_xs, mut dress_xx) = (Mat::zeros(ne, nx), Mat::zeros(nx, ne), Mat::zeros(nx, nx));
            for (g, cg) in &nb.groups {
                let comp = &self.components[*g];
                if comp.part.frequency != w {
                    continue;
                }
                let rg = r0.block(cg, cg);
                let (k1, k2) = if comp.part.counter { (k, kk) } else { (kk, k) };
                let (us1, us2) = (u_ec(*g, k1, cg), u_ec(*g, k2, cg));
                let x = us1.product(&rg).product(&us2.adjoint());
                plus(&mut dressing, &x);
                if nx > 0 {
                    let (ux1, ux2) = (u_xc(*g, k1, cg), u_xc(*g, k2, cg));
                    plus(&mut dress_sx, &us1.product(&rg).product(&ux2.adjoint()));
                    plus(&mut dress_xs, &ux1.product(&rg).product(&us2.adjoint()));
                    plus(&mut dress_xx, &ux1.product(&rg).product(&ux2.adjoint()));
                }
            }
            let mut whole = form.clone();
            plus(&mut whole, &dressing);
            let tw = rotate(&f, &whole.block(&kr, &kr), work);
            // The dressing's rows' norms alone are read (the reuse's R₀ change): ‖(V†DV)_i‖ = ‖(V†D)_i‖, V unitary.
            let td = f.to(&dressing.block(&kr, &kr), work);
            // The explicit columns' source V†(𝒦_SX + dressing), the explicit rows' emission (𝒦_XS + dressing)V, their own
            // dressing (the forms among them are Ĥ_PP's); the source's dressing's rows (its R₀'s change read).
            let (mut sx, mut xsm) = (form_sx, form_xs);
            plus(&mut sx, &dress_sx);
            plus(&mut xsm, &dress_xs);
            let st = f.to(&sx.block(&kr, &xall), work);
            let et = f.to(&xsm.block(&xall, &kr).adjoint(), work).adjoint();
            let tdx = f.to(&dress_sx.block(&kr, &xall), work);
            // Every level's: its own block's, its pairs', its dressing's rows' (with the static source's), its static source's
            // and explicit emission's columns.
            for (q, rq) in nb.levels.iter().enumerate() {
                lterm[ti * nl + q] = lnorm(&whole, rq, rq);
                for (p, rp) in nb.levels.iter().enumerate().filter(|&(p, _)| p != q) {
                    lpair[((ti + 1) * nl + q) * nl + p] = lnorm(&whole, rq, rp);
                }
                ldress[ti * nl + q] = (lnorm(&dressing, rq, &er).powi(2) + lnorm(&dress_sx, rq, &xall).powi(2)).sqrt();
                for jx in 0..nx {
                    lstat[((ti + 1) * nl + q) * n + xcol(jx)] = lnorm(&sx, rq, &[jx]);
                    lemit[((ti + 1) * nl + q) * n + xcol(jx)] = lnorm(&xsm.adjoint(), rq, &[jx]);
                }
            }
            terms.push((whole.frobenius(), tw, td, cc.frobenius(), st, et, dress_xx, tdx));
        }
        // The reference's: the level pairs' coupling and each level's Gershgorin intervals of K_qq − o's Hermitian part and
        // of its decay.
        let mut gersh = vec![[0.0; 4]; nl];
        for (q, rq) in nb.levels.iter().enumerate() {
            for (p, rp) in nb.levels.iter().enumerate().filter(|&(p, _)| p != q) {
                lpair[q * nl + p] = lnorm(&kall, rq, rp);
            }
            let (mut a, mut d) = ([f64::INFINITY, f64::NEG_INFINITY], [f64::INFINITY, f64::NEG_INFINITY]);
            for &i in rq {
                let (mut ra, mut rd) = (0.0, 0.0);
                for &j in rq.iter().filter(|&&j| j != i) {
                    let (x, y) = (kall[(i, j)], kall[(j, i)].conj());
                    ra += ((x + y) * 0.5).abs();
                    rd += ((x - y) * C64::I).abs();
                }
                let (ai, di) = (kall[(i, i)].re - b.value, -2.0 * kall[(i, i)].im);
                a = [a[0].min(ai - ra), a[1].max(ai + ra)];
                d = [d[0].min(di - rd), d[1].max(di + rd)];
            }
            gersh[q] = [a[0], a[1], d[0], d[1]];
        }
        *work += (ne * ne * (terms_n + 1)) as f64;
        let at: Vec<C64> = nb.terms.iter().map(|&t| ws.mid.coef[t]).collect();
        let within: Vec<C64> = terms.iter().flat_map(|x| Mat::from_fn(s, s, |i, j| if part[i] == part[j] { x.1[(i, j)] } else { C64::ZERO }).data).collect();
        let sterms: Vec<C64> = terms.iter().flat_map(|x| Mat::from_fn(s, n, |i, p| nb.xs.iter().position(|&q| q == p).map_or(C64::ZERO, |jx| x.4[(i, jx)])).data).collect();
        let eterms: Vec<C64> = terms.iter().flat_map(|x| Mat::from_fn(n, s, |p, i| nb.xs.iter().position(|&q| q == p).map_or(C64::ZERO, |jx| x.5[(jx, i)])).data).collect();
        let hterms: Vec<C64> = terms
            .iter()
            .flat_map(|x| {
                let mut m = Mat::zeros(n, n);
                for a in 0..nx {
                    for c in 0..nx {
                        m[(xcol(a), xcol(c))] = x.6[(a, c)];
                    }
                }
                m.data
            })
            .collect();
        let dressedx: Vec<f64> = terms.iter().flat_map(|x| (0..s).map(|i| (0..nx).map(|c| x.7[(i, c)].norm_sqr()).sum::<f64>().sqrt()).collect::<Vec<f64>>()).collect();
        let es: Vec<usize> = (0..GROUND).collect();
        let excited: Vec<usize> = kr.iter().map(|&a| rows[a]).collect();
        let emit: [Vec<C64>; 3] = std::array::from_fn(|q| {
            let c = Mat::from_fn(GROUND, s, |g, a| self.emission[q][(es[g], excited[a])]);
            f.to(&c.adjoint(), work).adjoint().data
        });
        let weights = terms.iter().map(|x| x.0).collect();
        let mut diag = vec![C64::ZERO; terms.len() * s];
        let mut norms = vec![0.0; terms.len() * s * parts];
        let mut dressed = vec![0.0; terms.len() * s];
        for (t, (_, tw, td, ..)) in terms.iter().enumerate() {
            for i in 0..s {
                diag[t * s + i] = tw[(i, i)];
                for j in (0..s).filter(|&j| j != i && part[j] == part[i]) {
                    norms[(t * s + i) * parts + part[j]] += tw[(i, j)].norm_sqr();
                }
                dressed[t * s + i] = (0..s).map(|j| td[(i, j)].norm_sqr()).sum::<f64>().sqrt();
            }
        }
        norms.iter_mut().for_each(|x| *x = x.sqrt());
        let copied = terms.iter().map(|x| x.3).collect();
        // The sources on every excited row: per split Σ_k γ_jk U_k (source columns), per vector U_a; the kept rows' in the
        // basis, Ỹ_l = V†X_l and Y_a = V†U_a; every level's column norms.
        let source = |c: usize, mat: &dyn Fn(usize, usize) -> C64| -> Mat {
            let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
            Mat::from_fn(ne, n, |a, p| if places.contains(&p) { mat(rows[a], self.ground[p]) } else { C64::ZERO })
        };
        let all: Vec<usize> = (0..n).collect();
        let col_norms = |m: &Mat, out: &mut [f64], at: usize| {
            for (q, rq) in nb.levels.iter().enumerate() {
                for j in 0..n {
                    out[at + q * n + j] = lnorm(m, rq, &[j]);
                }
            }
        };
        let nsp = nb.splits.len();
        let nv = nb.vectors.len();
        let (mut lbeam, mut lvec) = (vec![0.0; nl * nsp * n], vec![0.0; nl * nv * n]);
        let per: Vec<Vec<C64>> = nb
            .splits
            .iter()
            .enumerate()
            .map(|(sp, &(c, j))| {
                let m = source(c, &|x, t| self.components[c].couplings[j][(x, t)]);
                for (q, rq) in nb.levels.iter().enumerate() {
                    for p in 0..n {
                        lbeam[(q * nsp + sp) * n + p] = lnorm(&m, rq, &[p]);
                    }
                }
                f.to(&m.block(&kr, &all), work).data
            })
            .collect();
        let ln = per.len() * n;
        let mut beams = vec![C64::ZERO; s * ln];
        for (sp, y) in per.iter().enumerate() {
            for i in 0..s {
                beams[i * ln + sp * n..i * ln + sp * n + n].copy_from_slice(&y[i * n..i * n + n]);
            }
        }
        let mut lvecs: Vec<Vec<C64>> = vec![vec![]; nl];
        let vectors = nb
            .vectors
            .iter()
            .enumerate()
            .map(|(a, &(c, k))| {
                let m = source(c, &|x, t| self.components[c].basis[k][(x, t)]);
                for (q, rq) in nb.levels.iter().enumerate() {
                    for p in 0..n {
                        lvec[(q * nv + a) * n + p] = lnorm(&m, rq, &[p]);
                    }
                    lvecs[q].extend(m.block(rq, &all).data);
                }
                f.to(&m.block(&kr, &all), work).data
            })
            .collect();
        // The copies' rows: per copy vector R₀(C_g, C_g) U_k(S, C_g)† V.
        let vfull = f.from(&Mat::identity(s), work);
        let copies: Vec<Vec<C64>> = nb.copies.iter().map(|&(g, k, gi)| r0.block(&nb.groups[gi].1, &nb.groups[gi].1).product(&u_ec(g, k, &nb.groups[gi].1).block(&kr, &(0..nb.groups[gi].1.len()).collect::<Vec<_>>()).adjoint()).product(&vfull).data).collect();
        let copied_rows = copies.iter().map(|y: &Vec<C64>| y.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()).collect();
        // The exact copies (their block keeps every level): their coupling vectors V†U_k(E, D of g), K̂_ED and K̂_DE at the
        // reference less their bare drives (the dressing through the centred rows, held), K̂_DD, its forms per term.
        let couple: Vec<Vec<C64>> = nb
            .exact
            .iter()
            .map(|&(g, k)| f.to(&Mat::from_fn(s, nd, |a, d| if nb.dstates[d].1 == g { self.components[g].basis[k][(rows[kr[a]], nb.dstates[d].0)] } else { C64::ZERO }), work).data)
            .collect();
        let mut ked0 = f.to(&e.k.block(&kr, &dr), work).data;
        let mut kde0 = f.to(&e.k.block(&dr, &kr).adjoint(), work).adjoint().data;
        for (x, &(g, k)) in nb.exact.iter().enumerate() {
            let ek = ws.mid.e[g][k];
            for i in 0..s {
                for d in 0..nd {
                    ked0[i * nd + d] -= ek * couple[x][i * nd + d];
                    kde0[d * s + i] -= ek.conj() * couple[x][i * nd + d].conj();
                }
            }
        }
        let kdd0 = Mat::from_fn(nd, nd, |a, c| e.k[(ne + a, ne + c)] - if a == c { C64::real(b.value) } else { C64::ZERO }).data;
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
        // The explicit states' references at the point: the static source on the excited rows (`effective`'s, its dressing in
        // it), the emission (𝒦_XS + Ĥ_XC R₀ Ĥ_CS)V and the dressing Ĥ_XC R₀ B_C,X among themselves; the exact copies' static
        // source and emission less their bare drives (held, as K̂_ED's dressing); the centred copies' static rows.
        let cr: Vec<usize> = (ne + nd..b.rows.len()).collect();
        let source = e.source.clone().unwrap_or_else(|| Mat::zeros(ne + nd, n));
        let src_e = source.block(&er, &all);
        col_norms(&src_e, &mut lstat, 0);
        let lstat0: Vec<Vec<C64>> = nb.levels.iter().map(|rq| src_e.block(rq, &all).data).collect();
        let rqk: Vec<Vec<C64>> = nb.levels.iter().enumerate().map(|(q, rq)| if kept.contains(&q) || s == 0 { vec![] } else { kall.block(rq, &kr).product(&vfull).data }).collect();
        let stat = f.to(&src_e.block(&kr, &all), work).data;
        let mut ex = laid.emit.block(&nb.xs, &er);
        let mut exd = laid.emit.block(&nb.xs, &dr);
        let mut hx = Mat::zeros(nx, nx);
        if nc > 0 && nx > 0 {
            let xr = laid.emit.block(&nb.xs, &cr).product(r0);
            plus(&mut ex, &xr.product(&laid.inner.block(&cr, &er)));
            plus(&mut exd, &xr.product(&laid.inner.block(&cr, &dr)));
            hx = xr.product(&laid.absorb[0].block(&cr, &nb.xs));
        }
        for (q, rq) in nb.levels.iter().enumerate() {
            for jx in 0..nx {
                lemit[q * n + xcol(jx)] = lnorm(&ex.adjoint(), rq, &[jx]);
            }
        }
        let ext = f.to(&ex.block(&xall, &kr).adjoint(), work).adjoint();
        let (mut xemit0, mut hx0) = (vec![C64::ZERO; n * s], vec![C64::ZERO; n * n]);
        for a in 0..nx {
            for i in 0..s {
                xemit0[xcol(a) * s + i] = ext[(a, i)];
            }
            for c in 0..nx {
                hx0[xcol(a) * n + xcol(c)] = hx[(a, c)];
            }
        }
        let xcouple: Vec<Vec<C64>> = nb
            .exact
            .iter()
            .map(|&(g, k)| Mat::from_fn(n, nd, |p, d| nb.xs.iter().position(|&q| q == p).filter(|_| nb.dstates[d].1 == g).map_or(C64::ZERO, |jx| self.components[g].basis[k][(xstates[jx], nb.dstates[d].0)])).data)
            .collect();
        let (mut sd0, mut xd0) = (source.block(&dr, &all).data, vec![C64::ZERO; n * nd]);
        for a in 0..nx {
            for d in 0..nd {
                xd0[xcol(a) * nd + d] = exd[(a, d)];
            }
        }
        for (x, &(g, k)) in nb.exact.iter().enumerate() {
            let ek = ws.mid.e[g][k];
            for p in 0..n {
                for d in 0..nd {
                    let y = xcouple[x][p * nd + d];
                    xd0[p * nd + d] -= ek * y;
                    sd0[d * n + p] -= (ek * y).conj();
                }
            }
        }
        let copies_x: Vec<Vec<C64>> = nb
            .copies
            .iter()
            .map(|&(g, k, gi)| {
                let cg = &nb.groups[gi].1;
                let m = r0.block(cg, cg).product(&u_xc(g, k, cg).adjoint());
                Mat::from_fn(cg.len(), n, |a, p| nb.xs.iter().position(|&q| q == p).map_or(C64::ZERO, |jx| m[(a, jx)])).data
            })
            .collect();
        let xemitc: [Vec<C64>; 3] = std::array::from_fn(|q| Mat::from_fn(GROUND, n, |g, p| nb.xs.iter().position(|&x| x == p).map_or(C64::ZERO, |jx| self.emission[q][(g, xstates[jx])])).data);
        let _ = level;
        Some(NearBasis {
            at,
            weights,
            kept: kept.to_vec(),
            rows: kr,
            f,
            nu,
            gersh,
            lterm,
            lpair,
            ldress,
            lbeam,
            lvec,
            lstat,
            lemit,
            lvecs,
            lstat0,
            rqk,
            part,
            diag,
            norms,
            dressed,
            copied,
            beams,
            vectors,
            copies,
            copied_rows,
            z0: laid.z0,
            r0: r0.frobenius(),
            within,
            emit,
            couple,
            ked0,
            kde0,
            kdd0,
            ddterms,
            stat,
            sterms,
            xemit0,
            eterms,
            hx0,
            hterms,
            dressedx,
            sd0,
            xd0,
            xcouple,
            copies_x,
            xemitc,
        })
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
                f.extend(u[a].adjoint_product(&ru[bb]).data);
                g.extend(ru[a].adjoint_product(&ru[bb]).data);
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
            let Workspace { mid, pairs, pp, base, lin, proj, .. } = &mut *ws;
            self.proto_place(r, kappa, mid, pairs, pp, base, lin, proj, &mut work[PLACE]);
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
        for nb in ws.near.iter_mut() {
            nb.fresh = false;
        }
        let made_before = cache.made;
        let grown_before = cache.grown;
        let mut round = 0;
        let mut refreshes = 0;
        let mut site: Option<Site> = None;
        // Per near block its matrices at the point, once for every decomposition made here (`Fold::lay`, `Fold::effective`).
        let mut here: Vec<Option<(Laid, Effective)>> = (0..ws.near.len()).map(|_| None).collect();
        let (sweeps, series, ratio, rounds, omitted, tally) = loop {
            let made = cache.made;
            // The bases (module documentation, 6): the atom's own; beyond the share the nearest kept, then a new one. A near
            // block's kept levels: those `omit` wants, else its standing basis's; at an atom's first step none (a block with
            // exact copies keeps every level: the copies couple them all through the P layer).
            for kind in [false, true] {
                let count = if kind { ws.far.len() } else { ws.near.len() };
                for x in 0..count {
                    let tries = ws.renew.iter().filter(|y| **y == (kind, x)).count();
                    let standing = if kind { cache.at_far[x] } else { cache.at_near[x] };
                    let want: Vec<usize> = if kind {
                        vec![]
                    } else {
                        let nb = &mut ws.near[x];
                        match (nb.wanted.take(), standing) {
                            (Some(w), _) => w,
                            (None, Some(at)) => cache.near[x][at].kept.clone(),
                            (None, None) => (0..if nb.nd > 0 { nb.levels.len() } else { 0 }).collect(),
                        }
                    };
                    let same = standing.map(|at| kind || cache.near[x][at].kept == want);
                    if let (Some(at), 0, Some(true)) = (standing, tries, same) {
                        if !kind {
                            let s = cache.near[x][at].rows.len();
                            ws.near[x].fit(s, n);
                        }
                        continue;
                    }
                    let (terms, kept): (&[usize], Vec<(usize, &[C64], &[f64])>) = if kind {
                        (&ws.far[x].terms, cache.far[x].iter().enumerate().map(|(i, b)| (i, &b.at[..], &b.weights[..])).collect())
                    } else {
                        (&ws.near[x].terms, cache.near[x].iter().enumerate().filter(|(_, b)| b.kept == want).map(|(i, b)| (i, &b.at[..], &b.weights[..])).collect())
                    };
                    let nearest = |but: Option<usize>, work: &mut f64| -> Option<usize> {
                        let mut best: Option<(usize, f64)> = None;
                        for &(i, at, weights) in kept.iter().filter(|(i, ..)| Some(*i) != but) {
                            let d: f64 = terms.iter().enumerate().map(|(t, &p)| (ws.mid.coef[p] - at[t]).abs() * weights[t]).sum();
                            *work += terms.len() as f64;
                            if best.is_none_or(|b| d < b.1) {
                                best = Some((i, d));
                            }
                        }
                        best.map(|b| b.0)
                    };
                    let chosen = match tries {
                        0 => nearest(None, &mut work[REFRESH]),
                        1 => nearest(standing, &mut work[REFRESH]),
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
                                if here[x].is_none() {
                                    let laid = self.lay(ws.near[x].o, site, &mut work[REFRESH]);
                                    let e = self.effective(ws.near[x].o, &laid, &mut work[REFRESH])?;
                                    here[x] = Some((laid, e));
                                }
                                let (laid, e) = here[x].as_ref().unwrap();
                                let basis = self.near_basis(&ws.near[x], ws, laid, e, &want, &mut work[REFRESH])?;
                                cache.near[x].push(basis);
                                cache.near[x].len() - 1
                            }
                        }
                    };
                    if chosen.is_none() {
                        if !kind {
                            ws.near[x].fresh = true;
                        }
                        cache.made += 1;
                        if same == Some(false) {
                            cache.grown += 1;
                        }
                    }
                    if kind {
                        cache.at_far[x] = Some(at);
                    } else {
                        cache.at_near[x] = Some(at);
                        let s = cache.near[x][at].rows.len();
                        ws.near[x].fit(s, n);
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
                    nb.rhox[i] = 0.0;
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
                nb.epsr = 0.0;
                if nb.nc > 0 {
                    let x = basis.r0 * dm;
                    let eps_r = if x < 1.0 { x / (1.0 - x) } else { f64::INFINITY };
                    nb.epsr = eps_r;
                    for (t, &p) in nb.terms.iter().enumerate() {
                        let size = coef[p].abs();
                        for i in 0..s {
                            nb.rho[i] += eps_r * size * basis.dressed[t * s + i];
                            nb.rhox[i] += eps_r * size * basis.dressedx[t * s + i];
                        }
                    }
                    work[DIAGONAL] += (nb.terms.len() * s) as f64 / 4.0;
                }
            }
            lap(DIAGONAL, &mut clock, &mut tick);
            for (x, nb) in near.iter_mut().enumerate() {
                if nb.idle() {
                    continue;
                }
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let (ln, bn) = (nb.lanes * n, nb.splits.len() * n);
                for (sp, &(cc, j)) in nb.splits.iter().enumerate() {
                    nb.shifts[sp] = w[cc][j];
                    let fl = f[cc][j];
                    for i in 0..s {
                        let (at, from) = (i * ln + sp * n, i * bn + sp * n);
                        for (v, y) in nb.vt[at..at + n].iter_mut().zip(&basis.beams[from..from + n]) {
                            *v = fl * *y;
                        }
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
                let Near { xs, exact, nd, vtot, stat, emit, hx, b0d, exd, delta, .. } = &mut *nb;
                explicit_parts(xs, exact, basis, e, delta, vtot, stat, emit, hx, b0d, exd, s, *nd, n, &mut work[SOURCES]);
                if !nb.xs.is_empty() {
                    scatter(&nb.stat, &mut nb.vt, s, ln, (nb.lanes - 1) * n, n);
                }
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
                fused_product_into(&fb.products, &basis.f, (1, v * v, n * n), &mut fb.hf);
                fused_product_into(&fb.products, &basis.g, (1, v * v, n * n), &mut fb.mf);
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
                nb.bloch.fill(C64::ZERO);
                nb.ztot.fill(C64::ZERO);
                nb.active.fill(true);
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
            let (mut frozen, mut omitted) = (0.0_f64, 0.0_f64);
            let mut tally = [0usize; 2];
            for fb in far.iter_mut() {
                fb.omitted = false;
            }
            let outcome = loop {
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
                    if nb.idle() {
                        continue;
                    }
                    let basis = &cache.near[x][cache.at_near[x].unwrap()];
                    let s = nb.s;
                    let (t, ends) = (&basis.f.t, &basis.f.ends);
                    // Only the rows still going on move: the held ones keep their response. Every split at once, row by row
                    // (module documentation, # The beams side by side).
                    nb.zold.copy_from_slice(&nb.ztot);
                    let splits = nb.lanes;
                    let ln = splits * n;
                    let block = s * ln;
                    let at = nb.invs.len();
                    nb.invs.resize(at + block, C64::ZERO);
                    let nd = nb.nd;
                    let (pat, cat) = (nb.pjs.len(), nb.cinvs.len());
                    nb.pjs.resize(pat + splits * n * nd * s, C64::ZERO);
                    nb.cinvs.resize(cat + splits * n * nd * nd, C64::ZERO);
                    nb.zdtot.fill(C64::ZERO);
                    let rhs = &mut nb.rhs;
                    rhs.copy_from_slice(&nb.vt);
                    if sweep > 0 {
                        // −Z^{k−1}Δ.
                        fused_product_into(&nb.z, delta, (s * splits, n, n), &mut nb.prod);
                        for i in (0..s).filter(|&i| nb.active[i]) {
                            for (r, p) in rhs[i * ln..(i + 1) * ln].iter_mut().zip(&nb.prod[i * ln..(i + 1) * ln]) {
                                *r -= *p;
                            }
                        }
                        work[SERIES] += (s * ln * n) as f64;
                    }
                    // The diagonal: (c_j − w_l − κ_i)⁻¹, each column at its own energy.
                    for i in (0..s).filter(|&i| nb.active[i]) {
                        for (sp, &wl) in nb.shifts.iter().enumerate() {
                            let row = &mut nb.invs[at + i * ln + sp * n..at + i * ln + sp * n + n];
                            for (j, v) in row.iter_mut().enumerate() {
                                *v = reciprocal(c[j] - C64::real(wl) - nb.kappa[i]);
                            }
                        }
                    }
                    work[SERIES] += block as f64;
                    let zl = &mut nb.z;
                    for i in (0..s).rev().filter(|&i| nb.active[i]) {
                        let (head, tail) = zl.split_at_mut((i + 1) * ln);
                        let r = &mut rhs[i * ln..(i + 1) * ln];
                        for l in i + 1..ends[i] {
                            crate::linalg::mul_add(false, r, t[(i, l)], &tail[(l - i - 1) * ln..(l - i) * ln]);
                        }
                        work[SERIES] += ((ends[i] - i - 1) * ln + ln) as f64;
                        for ((z, x), v) in head[i * ln..].iter_mut().zip(r.iter()).zip(&nb.invs[at + i * ln..at + (i + 1) * ln]) {
                            *z = *x * *v;
                        }
                    }
                    // The exact copies by Woodbury (`copies_solve`), their own source the series' part alone, split by split.
                    if nd > 0 {
                        for sp in 0..splits {
                            let wl = nb.shifts[sp];
                            // The static lane's own source on the copies' rows (the explicit columns').
                            let mut rd = if sp == nb.splits.len() { nb.b0d.clone() } else { vec![C64::ZERO; nd * n] };
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
                            let mut zle = gather(&nb.z, s, ln, sp * n, n);
                            let inv = gather(&nb.invs[at..at + block], s, ln, sp * n, n);
                            let mut zdn = vec![C64::ZERO; nd * n];
                            let ok = copies_solve(s, nd, &nb.ked, &nb.kde, &nb.kdd, basis, c, wl, &inv, &rd, &mut zle, &mut zdn, pj, cinv, false, n, &mut work[SERIES]);
                            if !ok {
                                return None;
                            }
                            scatter(&zle, &mut nb.z, s, ln, sp * n, n);
                            zdl.copy_from_slice(&zdn);
                            for (a, b) in nb.zdtot.iter_mut().zip(&zdn) {
                                *a += *b;
                            }
                        }
                    }
                    for i in (0..s).filter(|&i| nb.active[i]) {
                        let zt = &mut nb.ztot[i * n..i * n + n];
                        zt.fill(C64::ZERO);
                        for sp in 0..splits {
                            for (a, b) in zt.iter_mut().zip(&nb.z[i * ln + sp * n..i * ln + sp * n + n]) {
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
                    // The block's part of the Bloch generator: Ê Z (+ the copies' emission Ê_D Z_D, the explicit states'
                    // dressing).
                    fused_product_into(&nb.emit, &nb.ztot, (n, s, n), &mut nb.bloch);
                    if !nb.xs.is_empty() {
                        nb.bloch.iter_mut().zip(&nb.hx).for_each(|(b, x)| *b += *x);
                        if nd > 0 {
                            let mut o = Buf::zeros(n * n);
                            fused_product_into(&nb.exd, &nb.zdtot, (n, nd, n), &mut o);
                            nb.bloch.iter_mut().zip(o.iter()).for_each(|(b, x)| *b += *x);
                        }
                    }
                    work[SERIES] += (n * s * n + splits * s * n) as f64;
                }
                for fb in far.iter().filter(|fb| !fb.omitted) {
                    for j in 0..n {
                        whole[j] += fb.mf[j * n + j].re;
                    }
                }
                // An explicit state's column: its excited amplitude ε holds its own unit row (DESIGN §1.3, ε = E + T), the
                // rate's unit of its column.
                for w in whole[self.ground.len()..].iter_mut() {
                    *w += 1.0;
                }
                h.copy_from_slice(hbase);
                for nb in near.iter() {
                    for (a, b) in h.iter_mut().zip(&nb.bloch) {
                        *a += *b;
                    }
                }
                // After the first sweep the levels and far blocks the point leaves out (`omit`, read on bounds): beyond the
                // share levels join, a new round on new decompositions.
                if sweep == 0 {
                    let (rho, grow) = omit(near, far, cache, mid, hbase, h, whole, &mut tally, share, self.thermal, n, &mut work[SERIES]);
                    if grow {
                        break None;
                    }
                    omitted = rho;
                }
                sweep += 1;
                if sweep == 1 {
                    frozen = freeze(near, cache, h, c, whole, order, unrate(share) - omitted, n, &mut work[SERIES]);
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
                    break Some((rate(unrate(rest) + frozen), most, omitted));
                }
            };
            lap(SERIES, &mut clock, &mut tick);
            let Some((series, ratio, omitted)) = outcome else {
                round += 1;
                if round > 3 + ws.near.iter().map(|x| x.levels.len()).sum::<usize>() {
                    return None;
                }
                continue;
            };
            // The readings of what is held (module documentation, 6), at the columns' energies H_jj. The reuse's first-order
            // error of the response the step uses, Z = Σ_l Z_l: Σ_l R_l δK̃ Z_l = R̄ δK̃ Z + Σ_l (R_l − R̄) δK̃ Z_l, R̄ at the
            // beams' mean Doppler shift w̄ -- row by row, row i of δK̃Z at most Σ_P off_iP ‖Z_P∖i‖ (+ ρ_i‖Z‖), the beams'
            // rest at most W|R̄|²/(1 − W|R̄|) (W the largest |w_l − w̄|) times (Σ_P off_iP + ρ_i) Σ_l ‖Z_l‖. Read on the
            // beams one by one against their sum, the reading would grow where the beams' responses cancel (a standing
            // wave's node) though the error cancels with them.
            for (x, nb) in near.iter_mut().enumerate() {
                nb.reading = 0.0;
                if nb.idle() {
                    continue;
                }
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let parts = basis.f.parts.len().max(1);
                let beams = nb.lanes;
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
                        each += (0..s).map(|i| nb.z[i * beams * n + sp * n + j].norm_sqr()).sum::<f64>().sqrt();
                    }
                    for i in 0..s {
                        let own = basis.part[i];
                        let mut e = nb.rho[i] * total.sqrt() + if nb.xs.contains(&j) { nb.rhox[i] } else { 0.0 };
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
                if fb.omitted {
                    (fb.reading, fb.centring) = (0.0, 0.0);
                    continue;
                }
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
            if renew.is_empty() || refreshes == 2 || cache.made > made {
                break (sweep, series, ratio, round + 1, omitted, tally);
            }
            ws.renew.extend(renew);
            round += 1;
            refreshes += 1;
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
            if nb.idle() {
                continue;
            }
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
            if nb.idle() {
                continue;
            }
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let s = nb.s;
            // Z†(Ṽ + Ỹ₀) + Z†(T + δκ − o)Z (the emission's part Ê Z is the Bloch generator's, in H), the metric Z†Z.
            let mut src = Buf::copy(&nb.vtot);
            if !nb.xs.is_empty() {
                src.iter_mut().zip(&nb.stat).for_each(|(a, b)| *a += *b);
            }
            compress(&nb.kappa, &basis.f.t, &basis.f.ends, &nb.ztot, None, &src, &mut nb.y, &mut a.data, s, n, &mut work[COMPRESSION]);
            let mut zh = Buf::zeros(n * s);
            adjoint_into(&nb.ztot, s, n, &mut zh);
            let mut o = Buf::zeros(n * n);
            fused_product_into(&zh, &nb.ztot, (n, s, n), &mut o);
            metric.data.iter_mut().zip(o.iter()).for_each(|(m, x)| *m += *x);
            work[COMPRESSION] += (s * n * n) as f64;
            if nb.nd > 0 {
                copies_compress(s, nb.nd, &nb.ked, &nb.kde, &nb.kdd, &nb.ztot, &nb.zdtot, &mut a.data, Some(&mut metric.data), n, &mut work[COMPRESSION]);
                if !nb.xs.is_empty() {
                    // Ê_D Z_D is the Bloch generator's (in H): Z_D†B_D alone.
                    let mut zdh = Buf::zeros(n * nb.nd);
                    adjoint_into(&nb.zdtot, nb.nd, n, &mut zdh);
                    fused_product_into(&zdh, &nb.b0d, (n, nb.nd, n), &mut o);
                    a.data.iter_mut().zip(o.iter()).for_each(|(m, x)| *m += *x);
                }
            }
        }
        for (x, fb) in far.iter().enumerate().filter(|(_, fb)| !fb.omitted) {
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
        step.omitted = rate(omitted);
        step.left = tally;
        step.made = cache.made - made_before;
        step.grown = cache.grown - grown_before;
        step.rounds = rounds;
        step.work = work;
        step.ticks = tick;
        Some(&ws.step)
    }
}

/// One near block's tangent along the lab axes (`Fold::prototype_tangent`), the three axes side by side in every row
/// (module documentation, # The beams side by side): ∂K̃ on the parts' entries ((a, i, l) at a·s² + i·s + l), ∂Ṽ
/// ((i, a, p) at i·3n + a·n + p), the emission's ((p, a, i) at (3p + a)·s + i), the explicit states' dressing's ((p, a, q)
/// at p·3n + a·n + q), the exact copies' explicit source's and emission's (axis by axis), the sources' tangent (∂f_l Ỹ_l,
/// the static lane's Σ_t ∂c_t S_t) and the lanes' responses' tangent at the last sweep ((i, a, l, j) at i·3Ln + a·Ln + l·n + j)
/// and summed over the splits ((i, a, j) at i·3n + a·n + j); the exact copies' ∂K̂ and theirs, the centred copies' rows'
/// (axis by axis).
struct Tangent {
    dk: Buf,
    demit: Buf,
    dhx: Buf,
    dexd: [Buf; 3],
    db0d: [Buf; 3],
    src: Buf,
    dz: Buf,
    dztot: Buf,
    // The exact copies': ∂K̂_ED, ∂K̂_DE, ∂K̂_DD, per split their response's tangent and summed.
    dked: [Buf; 3],
    dkde: [Buf; 3],
    dkdd: [Buf; 3],
    ddz: Vec<[Buf; 3]>,
    ddztot: [Buf; 3],
    dtc: [Buf; 3],
}

/// The tangent's phases (`Fold::prototype_tangent`; the measurement's split of its time): the light's gradients and the
/// terms' (∂Ĥ_PP, the far forms', ∂K̃, ∂Ṽ), the sources (∂f_l Ỹ_l − Z ∂c and ∂K̃ Z), the series' products (∂Z Δ, Z ∂Δ),
/// the triangular solves (with the exact copies'), the Bloch generators' tangent, the metric's and S's.
pub(crate) const TANGENT: [&str; 6] = ["light", "sources", "products", "solve", "bloch", "metric"];

/// Σ_i Σ_c x_ic* y_i,at+c over `rows` rows of `width` in x and of stride `stride` in y: a contraction with one axis's or
/// one split's own of the rows laid side by side.
fn inner(x: &[C64], y: &[C64], rows: usize, width: usize, stride: usize, at: usize) -> C64 {
    let mut sum = C64::ZERO;
    for i in 0..rows {
        for (a, b) in x[i * width..(i + 1) * width].iter().zip(&y[i * stride + at..i * stride + at + width]) {
            sum += a.conj() * *b;
        }
    }
    sum
}

/// The step's tangent (`Fold::prototype_tangent`): per near block its own (`Tangent`), per lab axis the metric's ∂X and
/// S's ∂S, the components' coefficients' gradients ∂e and the pairs' ∂c at the middle.
pub(crate) struct Tangents {
    blocks: Vec<Tangent>,
    dx: Vec<Buf>,
    ds: Vec<Buf>,
    de: Vec<Vec<[C64; 3]>>,
    dc: Vec<[C64; 3]>,
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
        let Workspace { mid, nodes, near, far, pairs, pp, base, lin, proj, step, temp, .. } = &mut *ws;
        let (mid, pairs, pp, base, lin, proj, step) = (&*mid, &*pairs, &*pp, &*base, &*lin, &*proj, &*step);
        for (j, node) in nodes.iter_mut().enumerate() {
            let rj: [f64; 3] = std::array::from_fn(|a| r[a] + velocity[a] * times[j]);
            self.proto_place(rj, kappa, &mut node.at, pairs, pp, base, lin, proj, &mut work);
            let at = &node.at;
            let a = &mut node.a.data;
            a.copy_from_slice(&at.hpp);
            for (x, fb) in far.iter().enumerate().filter(|(_, fb)| !fb.omitted) {
                let basis = &cache.far[x][cache.at_far[x].unwrap()];
                let v = fb.vectors.len();
                let (hf, mf) = (&mut node.hf[x], &mut node.mf[x]);
                let mut p = Buf::zeros(v * v);
                for ia in 0..v {
                    for ib in 0..v {
                        let ((ca, ka), (cb, kb)) = (fb.vectors[ia], fb.vectors[ib]);
                        p[ia * v + ib] = at.e[ca][ka].conj() * at.e[cb][kb];
                    }
                }
                fused_product_into(&p, &basis.f, (1, v * v, n * n), hf);
                fused_product_into(&p, &basis.g, (1, v * v, n * n), mf);
                for i in 0..n * n {
                    a[i] += hf[i] + mf[i] * basis.z0;
                }
                work += (2 * v * v * n * n + n * n) as f64;
            }
            for (x, nb) in near.iter_mut().enumerate() {
                if nb.idle() {
                    continue;
                }
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                // The beams' lanes turn with their factors; the static lane is held (its ratio one).
                for (sp, &(cc, jj)) in nb.splits.iter().enumerate() {
                    nb.ratio[j][sp] = at.f[cc][jj] / mid.f[cc][jj];
                }
                if nb.lanes > nb.splits.len() {
                    nb.ratio[j][nb.lanes - 1] = C64::ONE;
                }
                let zn = &mut nb.znode[j];
                zn.fill(C64::ZERO);
                let ln = nb.lanes * n;
                for i in 0..s {
                    for sp in 0..nb.lanes {
                        crate::linalg::mul_add(false, &mut zn[i * n..i * n + n], nb.ratio[j][sp], &nb.z[i * ln + sp * n..i * ln + sp * n + n]);
                    }
                }
                let vn = &mut nb.vnode[j];
                vn.fill(C64::ZERO);
                for (ia, &(cc, k)) in nb.vectors.iter().enumerate() {
                    crate::linalg::mul_add(false, vn, at.e[cc][k], &basis.vectors[ia]);
                }
                work += ((nb.lanes + nb.vectors.len()) * s * n) as f64;
                {
                    let Near { xs, exact, nd, vnode, pnode, terms, .. } = &mut *nb;
                    let pn = &mut pnode[j];
                    for (t, &p) in terms.iter().enumerate() {
                        pn.delta[t] = at.coef[p] - basis.at[t];
                    }
                    explicit_parts(xs, exact, basis, &at.e, &pn.delta, &vnode[j], &mut pn.stat, &mut pn.emit, &mut pn.hx, &mut pn.b0d, &mut pn.exd, s, *nd, n, &mut work);
                }
                let mut src = Buf::copy(&nb.vnode[j]);
                if !nb.xs.is_empty() {
                    src.iter_mut().zip(&nb.pnode[j].stat).for_each(|(u, y)| *u += *y);
                    a.iter_mut().zip(&nb.pnode[j].hx).for_each(|(u, y)| *u += *y);
                }
                compress(&nb.kappa, &basis.f.t, &basis.f.ends, &nb.znode[j], Some(&nb.pnode[j].emit), &src, &mut nb.rhs, a, s, n, &mut work);
                if nb.nd > 0 {
                    let nd = nb.nd;
                    let zd = &mut nb.zdnode[j];
                    zd.fill(C64::ZERO);
                    for sp in 0..nb.lanes {
                        crate::linalg::mul_add(false, zd, nb.ratio[j][sp], &nb.zd[sp * nd * n..(sp + 1) * nd * n]);
                    }
                    let (mut ked, mut kde) = (std::mem::take(&mut nb.kednode[j]), std::mem::take(&mut nb.kdenode[j]));
                    copy_couplings(nb, basis, &at.e, &mut ked, &mut kde);
                    (nb.kednode[j], nb.kdenode[j]) = (ked, kde);
                    copies_compress(s, nd, &nb.kednode[j], &nb.kdenode[j], &nb.kdd, &nb.znode[j], &nb.zdnode[j], a, None, n, &mut work);
                    if !nb.xs.is_empty() {
                        explicit_copies(&nb.pnode[j].exd, &nb.pnode[j].b0d, &nb.zdnode[j], a, nd, n, &mut work);
                    }
                }
                if nb.copied_in {
                    copies_of(&nb.copies, &nb.groups, basis, &at.e, &nb.znode[j], true, &mut nb.tcnode[j], s, n, &mut work);
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

    /// The force (module documentation, # The force): −∂/∂r_a of Σ_j Re tr(Ḡ_j†G_j), `bars` the cotangents of the middle's
    /// G and of the outer nodes' (where `prototype_nodes` made them; ignored otherwise), every position moved together --
    /// the step's derivative, the same tangent the photons take (`Fold::prototype_tangent`), contracted with the cotangents
    /// of its top: G = S⁻¹AS⁻¹ (Ā = S⁻¹ḠS⁻¹, and S⁻¹'s to X̄ by `lyapunov`: Re⟨X̄, ∂X⟩), and per position the compression's
    /// (`compress_back`: Re⟨Z̄, ∂Z⟩ + Re⟨Ṽ̄, ∂Ṽ⟩ + Re⟨K̄, ∂K̃⟩, the exact copies' and the centred copies' rows with them),
    /// Ĥ_PP's and the far forms'. A node's Z(s) = Σ_l ρ_l Z_l moves with its factors' ratios ρ_l = f_l(s)/f_l and the
    /// responses' own tangent. The kept R₀ and z₀ are held as the step holds them; their relative change per µm along each
    /// axis (`Force::held`, ‖R₀‖·|∂(Ĥ − z₀)|, ~10⁻⁷ on the scenes) is what that leaves of the force, relatively. After
    /// `prototype`, `prototype_nodes` and `prototype_tangent` on the same cache.
    pub(crate) fn prototype_force(&self, cache: &Cache, ws: &mut Workspace, bars: [&Mat; 3]) -> Force {
        let n = ws.n;
        let nn = n * n;
        let mut work = 0.0;
        let Workspace { near, far, mid, nodes, noded, pairs, pp, lin, proj, step, eigen, vectors, temp, tangents, .. } = &mut *ws;
        let (mid, nodes, pairs, pp, proj, step) = (&*mid, &*nodes, &*pairs, &*pp, &*proj, &*step);
        let tg = tangents.as_ref().expect("the step's tangent");
        let positions = if *noded { 3 } else { 1 };
        let normal = &step.normal.data;
        // Ā_j = S⁻¹Ḡ_jS⁻¹; N̄ = Σ_j Ḡ_j S⁻¹ A_j† + A_j† S⁻¹ Ḡ_j; X̄ (`inverse_root_back`).
        let mut abar: Vec<Buf> = vec![];
        let mut nbar = Buf::zeros(nn);
        let mut adj = Buf::zeros(nn);
        for j in 0..positions {
            let g = &bars[j].data;
            let aj = if j == 0 { &step.a.data } else { &nodes[j - 1].a.data };
            multiply(normal, g, &mut eigen.data, n);
            multiply(&eigen.data, normal, &mut temp.data, n);
            abar.push(Buf::copy(&temp.data));
            adjoint_into(aj, n, n, &mut adj);
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
        let mut ft = [0u64; 5];
        let mut clock = ticks();
        let lap = |k: usize, clock: &mut u64, ft: &mut [u64; 5]| {
            let now = ticks();
            ft[k] += now - *clock;
            *clock = now;
        };
        lap(0, &mut clock, &mut ft);
        // f'_a: the metric's part first.
        let re = |a: &[C64], b: &[C64]| a.iter().zip(b).map(|(x, y)| (x.conj() * *y).re).sum::<f64>();
        let im = |a: &[C64], b: &[C64]| a.iter().zip(b).map(|(x, y)| (x.conj() * *y).im).sum::<f64>();
        let mut fp: [f64; 3] = std::array::from_fn(|a| re(&xbar, &tg.dx[a]));
        work += (3 * nn) as f64;
        for j in 0..positions {
            let at = if j == 0 { mid } else { &nodes[j - 1].at };
            let ab = &abar[j];
            // The pairs' gradients and the components' at the position.
            let dc = if j == 0 { None } else { Some(self.pair_gradients(at, proj, pairs)) };
            let dc = dc.as_deref().unwrap_or(&tg.dc);
            let deps = if j == 0 { None } else { Some(self.component_gradients(at, proj)) };
            let de_at = |c: usize, k: usize| -> [C64; 3] {
                match &deps {
                    None => tg.de[c][k],
                    Some(d) => {
                        let comp = &self.components[c];
                        let x = d[comp.part.frequency][k];
                        if comp.part.counter { x.map(|z| z.conj()) } else { x }
                    }
                }
            };
            // Ĥ_PP's: Re⟨Ā, Σ_t ∂c_t pp_t⟩.
            for (t, entries) in pp.iter().enumerate() {
                let k: C64 = entries.iter().map(|&(i, v)| ab[i].conj() * v).sum();
                for a in 0..3 {
                    fp[a] += (k * dc[t][a]).re;
                }
                work += entries.len() as f64;
            }
            // The drives between ground and explicit places: Re⟨Ā, ∂e lin + ∂e* lin'⟩.
            for (g, kk, direct, conj) in lin.iter() {
                let kd: C64 = direct.iter().map(|&(i, v)| ab[i].conj() * v).sum();
                let kc: C64 = conj.iter().map(|&(i, v)| ab[i].conj() * v).sum();
                let d = de_at(*g, *kk);
                for a in 0..3 {
                    fp[a] += (kd * d[a] + kc * d[a].conj()).re;
                }
                work += (2 * direct.len()) as f64;
            }
            // The far forms: Re⟨Ā, Σ ∂p_ab F_ab⟩ + z₀ Re⟨Ā, Σ ∂p_ab G_ab⟩ (the metric's in X̄ already).
            for (x, fb) in far.iter().enumerate().filter(|(_, fb)| !fb.omitted) {
                let basis = &cache.far[x][cache.at_far[x].unwrap()];
                let v = fb.vectors.len();
                for ia in 0..v {
                    for ib in 0..v {
                        let ((ca, ka), (cb, kb)) = (fb.vectors[ia], fb.vectors[ib]);
                        let (f, g) = (&basis.f[(ia * v + ib) * nn..(ia * v + ib + 1) * nn], &basis.g[(ia * v + ib) * nn..(ia * v + ib + 1) * nn]);
                        let k: C64 = (0..nn).map(|i| ab[i].conj() * (f[i] + g[i] * basis.z0)).sum();
                        let (da, db) = (de_at(ca, ka), de_at(cb, kb));
                        for a in 0..3 {
                            fp[a] += (k * (da[a].conj() * at.e[cb][kb] + at.e[ca][ka].conj() * db[a])).re;
                        }
                    }
                }
                work += (v * v * nn) as f64;
            }
            lap(1, &mut clock, &mut ft);
            // The near blocks.
            for (x, nb) in near.iter_mut().enumerate() {
                if nb.idle() {
                    continue;
                }
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let t = &tg.blocks[x];
                let (s, nd, splits) = (nb.s, nb.nd, nb.lanes);
                let (z, v): (&[C64], &[C64]) = if j == 0 { (&nb.ztot, &nb.vtot) } else { (&nb.znode[j - 1], &nb.vnode[j - 1]) };
                // The position's emission, static source, explicit dressing, the exact copies' explicit parts.
                let (emit, stat, exd, b0d): (&[C64], &[C64], &[C64], &[C64]) = if j == 0 { (&nb.emit, &nb.stat, &nb.exd, &nb.b0d) } else { (&nb.pnode[j - 1].emit, &nb.pnode[j - 1].stat, &nb.pnode[j - 1].exd, &nb.pnode[j - 1].b0d) };
                let explicit = !nb.xs.is_empty();
                let mut src = Buf::copy(v);
                if explicit {
                    src.iter_mut().zip(stat).for_each(|(u, y)| *u += *y);
                }
                let mut y = Buf::zeros(s * n);
                for i in 0..s {
                    for c in 0..n {
                        y[i * n + c] += nb.kappa[i] * z[i * n + c];
                    }
                    for l in i + 1..basis.f.ends[i] {
                        crate::linalg::mul_add(false, &mut y[i * n..i * n + n], basis.f.t[(i, l)], &z[l * n..l * n + n]);
                    }
                }
                let (mut zb, mut vb, mut eb, mut kb, mut xb) = (Buf::zeros(s * n), Buf::zeros(s * n), Buf::zeros(n * s), Buf::zeros(s * s), Buf::zeros(s * n));
                compress_back(&nb.kappa, &basis.f.t, &basis.f.ends, z, emit, &src, &y, ab, &mut zb, &mut vb, &mut eb, &mut kb, &mut xb, s, n, &mut work);
                // The explicit parts' terms: Re⟨B̄, S_t⟩ + Re⟨Ê̄, E_t⟩ + Re⟨Ā, H_t⟩ times ∂c_t.
                if explicit {
                    for (ti, &tt) in nb.terms.iter().enumerate() {
                        let g = inner(&vb, &basis.sterms[ti * s * n..(ti + 1) * s * n], s, n, n, 0) + inner(&eb, &basis.eterms[ti * n * s..(ti + 1) * n * s], n, s, s, 0) + inner(ab, &basis.hterms[ti * nn..(ti + 1) * nn], n, n, n, 0);
                        for a in 0..3 {
                            fp[a] += (g * dc[tt][a]).re;
                        }
                    }
                    work += (nb.terms.len() * (2 * s * n + nn)) as f64;
                }
                // Ṽ is in both the source and the emission (Ê = Ṽ† on the ground rows): its cotangent B̄ + Ê̄†.
                for i in 0..s {
                    for p in 0..n {
                        vb[i * n + p] += eb[p * s + i].conj();
                    }
                }
                // The exact copies' rows.
                let (mut zdb, mut kedb, mut kdeb, mut kddb) = (Buf::zeros(nd * n), Buf::zeros(s * nd), Buf::zeros(nd * s), Buf::zeros(nd * nd));
                if nd > 0 {
                    let (zd, ked, kde): (&[C64], &[C64], &[C64]) = if j == 0 { (&nb.zdtot, &nb.ked, &nb.kde) } else { (&nb.zdnode[j - 1], &nb.kednode[j - 1], &nb.kdenode[j - 1]) };
                    copies_compress_back(s, nd, ked, kde, &nb.kdd, z, zd, ab, None, &mut zb, &mut zdb, &mut kedb, &mut kdeb, &mut kddb, n, &mut work);
                    // Ê_D Z_D + Z_D†B_D: Z̄_D += Ê_D†Ā + B_D Ā†, and Re⟨ĀZ_D†, ∂Ê_D⟩ + Re⟨Z_DĀ, ∂B_D⟩ on each coupling vector.
                    if explicit {
                        let mut adj = Buf::zeros(nn);
                        adjoint_into(ab, n, n, &mut adj);
                        let mut eh = Buf::zeros(nd * n);
                        adjoint_into(exd, n, nd, &mut eh);
                        let mut o = Buf::zeros(nd * n);
                        fused_product_into(&eh, ab, (nd, n, n), &mut o);
                        zdb.iter_mut().zip(o.iter()).for_each(|(u, y)| *u += *y);
                        fused_product_into(b0d, &adj, (nd, n, n), &mut o);
                        zdb.iter_mut().zip(o.iter()).for_each(|(u, y)| *u += *y);
                        let mut zdh = Buf::zeros(n * nd);
                        adjoint_into(zd, nd, n, &mut zdh);
                        let (mut exb, mut bdb) = (Buf::zeros(n * nd), Buf::zeros(nd * n));
                        fused_product_into(ab, &zdh, (n, n, nd), &mut exb);
                        fused_product_into(zd, ab, (nd, n, n), &mut bdb);
                        for (xx, &(g, k)) in nb.exact.iter().enumerate() {
                            let yv = &basis.xcouple[xx];
                            let (mut ge, mut gb) = (C64::ZERO, C64::ZERO);
                            for p in 0..n {
                                for d2 in 0..nd {
                                    ge += exb[p * nd + d2].conj() * yv[p * nd + d2];
                                    gb += bdb[d2 * n + p].conj() * yv[p * nd + d2].conj();
                                }
                            }
                            let d = de_at(g, k);
                            for a in 0..3 {
                                fp[a] += (ge * d[a] + gb * d[a].conj()).re;
                            }
                        }
                        work += (4 * nd * n * n) as f64;
                    }
                }
                // The centred copies' rows: T̄_C = z₀ T_C (Ā + Ā†), onto Z and e.
                let mut tcb = Buf::zeros(nb.nc * n);
                if nb.copied_in {
                    let tc: &[C64] = if j == 0 { &nb.tc } else { &nb.tcnode[j - 1] };
                    let mut w = Buf::zeros(nn);
                    (0..nn).for_each(|i| w[i] = (ab[i] + ab[(i % n) * n + i / n].conj()) * basis.z0);
                    for k in 0..nb.nc {
                        for p in 0..n {
                            crate::linalg::mul_add(false, &mut tcb[k * n..k * n + n], tc[k * n + p], &w[p * n..p * n + n]);
                        }
                    }
                }
                // ∂Ṽ = Σ ∂e_a Y_a; ∂K̃ (the middle's, held at the nodes) on the parts; the exact copies' ∂K̂.
                for (ia, &(cc, k)) in nb.vectors.iter().enumerate() {
                    let g: C64 = vb.iter().zip(&basis.vectors[ia]).map(|(a, b)| a.conj() * *b).sum();
                    let d = de_at(cc, k);
                    for a in 0..3 {
                        fp[a] += (g * d[a]).re;
                    }
                }
                for a in 0..3 {
                    fp[a] += inner(&kb, &t.dk[a * s * s..(a + 1) * s * s], s, s, s, 0).re;
                    if nd > 0 {
                        fp[a] += re(&kddb, &t.dkdd[a]);
                    }
                }
                if nd > 0 {
                    for (xx, &(g, k)) in nb.exact.iter().enumerate() {
                        let yv = &basis.couple[xx];
                        let mut gk = C64::ZERO;
                        for i in 0..s {
                            for d2 in 0..nd {
                                gk += kedb[i * nd + d2].conj() * yv[i * nd + d2] + kdeb[d2 * s + i] * yv[i * nd + d2];
                            }
                        }
                        let d = de_at(g, k);
                        for a in 0..3 {
                            fp[a] += (gk * d[a]).re;
                        }
                    }
                }
                lap(2, &mut clock, &mut ft);
                // ∂Z (and ∂Z_D, ∂T_C): the middle's tangent; at a node Σ_l (∂ρ_l Z_l + ρ_l ∂Z_l).
                if j == 0 {
                    for a in 0..3 {
                        fp[a] += inner(&zb, &t.dztot, s, n, 3 * n, a * n).re + re(&zdb, &t.ddztot[a]);
                        if nb.copied_in {
                            fp[a] += re(&tcb, &t.dtc[a]);
                        }
                    }
                    work += (3 * (s + nd) * n) as f64;
                } else {
                    // The centred copies' rows at a node: T_C(s) of Z(s) and e(s): onto Z(s) and e(s) by `copies_back`.
                    if nb.copied_in {
                        let mut ge: Vec<Buf> = self.components.iter().map(|c| Buf::zeros(c.basis.len())).collect();
                        copies_back(&nb.copies, &nb.groups, basis, &at.e, z, &tcb, &mut zb, &mut ge, s, n, &mut work);
                        for (c, g) in ge.iter().enumerate() {
                            for (k, gk) in g.iter().enumerate() {
                                let d = de_at(c, k);
                                for a in 0..3 {
                                    fp[a] += (*gk * d[a]).re;
                                }
                            }
                        }
                    }
                    let (ln, m) = (splits * n, 3 * splits * n);
                    // The static lane, held over the step (its ratio one).
                    if explicit {
                        let sp = splits - 1;
                        for a in 0..3 {
                            fp[a] += inner(&zb, &t.dz, s, n, m, a * ln + sp * n).re;
                            if nd > 0 {
                                fp[a] += re(&zdb, &t.ddz[sp][a]);
                            }
                        }
                    }
                    for (sp, &(cc, jj)) in nb.splits.iter().enumerate() {
                        let l = self.components[cc].beams[jj];
                        let rho = nb.ratio[j - 1][sp];
                        let fm = mid.f[cc][jj];
                        let mut g: C64 = inner(&zb, &nb.z, s, n, ln, sp * n);
                        if nd > 0 {
                            g += zdb.iter().zip(&nb.zd[sp * nd * n..(sp + 1) * nd * n]).map(|(a, b)| a.conj() * *b).sum::<C64>();
                        }
                        for a in 0..3 {
                            let dfs = self.components[cc].factor(at.amplitudes[l].gradient[a]);
                            let dfm = self.components[cc].factor(mid.amplitudes[l].gradient[a]);
                            let drho = (dfs - rho * dfm) / fm;
                            let dzl = inner(&zb, &t.dz, s, n, m, a * ln + sp * n);
                            fp[a] += (g * drho).re + rho.re * dzl.re - rho.im * dzl.im;
                            if nd > 0 {
                                fp[a] += rho.re * re(&zdb, &t.ddz[sp][a]) - rho.im * im(&zdb, &t.ddz[sp][a]);
                            }
                        }
                    }
                    work += (3 * splits * (s + nd) * n) as f64;
                }
                lap(3, &mut clock, &mut ft);
            }
        }
        // What the kept R₀ and z₀ leave: their relative change per µm along each axis. A block's centre z₀'s gradient: the
        // mean of its sources' Ĥ_PP diagonal's, of the pairs' coefficients' gradients.
        let mut diagonal = vec![[0.0_f64; 3]; n];
        for (t, e) in pp.iter().enumerate() {
            for &(i, v) in e.iter().filter(|x| x.0 % (n + 1) == 0) {
                for a in 0..3 {
                    diagonal[i / n][a] += (tg.dc[t][a] * v).re;
                }
            }
        }
        let centre_gradient = |centre: &[usize]| -> [f64; 3] { std::array::from_fn(|a| centre.iter().map(|&p| diagonal[p][a]).sum::<f64>() / centre.len().max(1) as f64) };
        let mut held = [0.0_f64; 3];
        for (x, nb) in near.iter().enumerate() {
            if nb.nc == 0 {
                continue;
            }
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let z0g = centre_gradient(&nb.centre);
            for a in 0..3 {
                let change = z0g[a].abs() + nb.terms.iter().enumerate().map(|(ti, &tt)| tg.dc[tt][a].abs() * basis.copied[ti]).sum::<f64>();
                held[a] = held[a].max(basis.r0 * change);
            }
        }
        for (x, fb) in far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            let z0g = centre_gradient(&fb.centre);
            for a in 0..3 {
                let change = z0g[a].abs() + fb.terms.iter().enumerate().map(|(ti, &tt)| tg.dc[tt][a].abs() * basis.copied[ti]).sum::<f64>();
                held[a] = held[a].max(basis.r0 * change);
            }
        }
        lap(4, &mut clock, &mut ft);
        ws.force_ticks = ft;
        Force { force: fp.map(|x| -x), held, work }
    }

    /// The step's tangent along the three lab axes (module documentation, # The force): every split's response's change by the
    /// same sweeps as the step -- its field's (the split's factor) and its own through Ĥ_PP, K̃ on the decomposition's parts,
    /// Ṽ and the Bloch generators, the exact copies' with it --, the metric's ∂X and S's ∂S (S dS + dS S = ∂X, `lyapunov`):
    /// the force's and the photons' one derivative of the step (`Fold::prototype_force`, `Fold::prototype_photons`). The
    /// three axes and every split in one pass, row by row (module documentation, # The beams side by side): the axes are
    /// three sources of one operator, the splits one operator each that differs from the others' by a scalar shift alone.
    /// After `prototype` on the same cache; returns the work, its ticks and work by phase in the workspace (`TANGENT`).
    pub(crate) fn prototype_tangent(&self, cache: &Cache, ws: &mut Workspace) -> f64 {
        let n = ws.n;
        let nn = n * n;
        let n3 = 3 * n;
        let mut work = [0.0_f64; 6];
        let mut tick = [0u64; 6];
        let mut clock = ticks();
        let lap = |k: usize, clock: &mut u64, tick: &mut [u64; 6]| {
            let now = ticks();
            tick[k] += now - *clock;
            *clock = now;
        };
        let (mid, step, pairs, pp, proj) = (&ws.mid, &ws.step, &ws.pairs, &ws.pp, &ws.proj);
        let normal = &step.normal.data;
        let sweeps = step.sweeps;
        // The light's tangents: per component and vector ∂e, per pair ∂c.
        let deps = self.component_gradients(mid, proj);
        let de: Vec<Vec<[C64; 3]>> = self.components.iter().map(|c| deps[c.part.frequency].iter().map(|d| if c.part.counter { d.map(|z| z.conj()) } else { *d }).collect()).collect();
        let dc = self.pair_gradients(mid, proj, pairs);
        // Ĥ_PP's and the far forms' (and the far metric's) tangents, the axes side by side: (p, a, q) at p·3n + a·n + q.
        let (mut dhbase, mut dmetric) = (Buf::zeros(n * n3), Buf::zeros(n * n3));
        for (t, entries) in pp.iter().enumerate() {
            for &(i, v) in entries {
                let (p, q) = (i / n, i % n);
                for a in 0..3 {
                    dhbase[p * n3 + a * n + q] += dc[t][a] * v;
                }
            }
            work[0] += (3 * entries.len()) as f64;
        }
        for (g, k, direct, conj) in &ws.lin {
            for a in 0..3 {
                let d = de[*g][*k][a];
                for &(i, v) in direct {
                    dhbase[(i / n) * n3 + a * n + i % n] += d * v;
                }
                for &(i, v) in conj {
                    dhbase[(i / n) * n3 + a * n + i % n] += d.conj() * v;
                }
            }
            work[0] += (6 * direct.len()) as f64;
        }
        for (x, fb) in ws.far.iter().enumerate().filter(|(_, fb)| !fb.omitted) {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            let v = fb.vectors.len();
            // The pairs' gradients (3 × v²) on the stacked forms (v² × n²), one product each.
            let mut dp = Buf::zeros(3 * v * v);
            for ia in 0..v {
                for ib in 0..v {
                    let ((ca, ka), (cb, kb)) = (fb.vectors[ia], fb.vectors[ib]);
                    for a in 0..3 {
                        dp[a * v * v + ia * v + ib] = de[ca][ka][a].conj() * mid.e[cb][kb] + mid.e[ca][ka].conj() * de[cb][kb][a];
                    }
                }
            }
            let mut o = Buf::zeros(3 * nn);
            for (forms, out) in [(&basis.f, &mut dhbase), (&basis.g, &mut dmetric)] {
                fused_product_into(&dp, forms, (3, v * v, nn), &mut o);
                for a in 0..3 {
                    for p in 0..n {
                        for (u, y) in out[p * n3 + a * n..p * n3 + a * n + n].iter_mut().zip(&o[a * nn + p * n..a * nn + p * n + n]) {
                            *u += *y;
                        }
                    }
                }
            }
            work[0] += (6 * v * v * nn) as f64;
        }
        // Per near block: ∂K̃ on the parts, ∂Ṽ, the exact copies' ∂K̂ (the axes side by side).
        let mut tangents: Vec<Tangent> = ws
            .near
            .iter()
            .enumerate()
            .map(|(x, nb)| {
                if nb.idle() {
                    let e = || Buf::zeros(0);
                    let e3 = || std::array::from_fn(|_| Buf::zeros(0));
                    return Tangent { dk: e(), demit: e(), dhx: e(), dexd: e3(), db0d: e3(), src: e(), dz: e(), dztot: e(), dked: e3(), dkde: e3(), dkdd: e3(), ddz: vec![], ddztot: e3(), dtc: e3() };
                }
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let s3 = 3 * s;
                // ∂K̃ = Σ_t ∂c_t V†N_tV on the parts' entries alone: the axes' (3 × T) on the terms' (T × s²), one product.
                let terms = nb.terms.len();
                let mut dct = Buf::zeros(3 * terms);
                for (ti, &tt) in nb.terms.iter().enumerate() {
                    for a in 0..3 {
                        dct[a * terms + ti] = dc[tt][a];
                    }
                }
                // ∂K̃ on the kept parts' entries.
                let mut dk = Buf::zeros(s * s3);
                for t in 0..terms {
                    let w = &basis.within[t * s * s..(t + 1) * s * s];
                    for a in 0..3 {
                        let d = dct[a * terms + t];
                        if d == C64::ZERO {
                            continue;
                        }
                        for (idx, at, _) in basis.f.parts.iter() {
                            for i in *at..at + idx.len() {
                                crate::linalg::mul_add(false, &mut dk[a * s * s + i * s + at..a * s * s + i * s + at + idx.len()], d, &w[i * s + at..i * s + at + idx.len()]);
                            }
                        }
                    }
                }
                let mut dv = Buf::zeros(s * n3);
                for (ia, &(cc, k)) in nb.vectors.iter().enumerate() {
                    let y = &basis.vectors[ia];
                    for i in 0..s {
                        for a in 0..3 {
                            crate::linalg::mul_add(false, &mut dv[i * n3 + a * n..i * n3 + a * n + n], de[cc][k][a], &y[i * n..i * n + n]);
                        }
                    }
                }
                work[0] += (3 * (terms * s * s + nb.vectors.len() * s * n)) as f64;
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
                // The emission's tangent (rows (p, a), 3n × s): ∂Ṽ† on the ground rows, Σ_t ∂c_t E_t on the explicit ones; the
                // static lane's source Σ_t ∂c_t S_t, the explicit states' dressing Σ_t ∂c_t H_t (n × 3n), the exact copies'
                // static source and emission, axis by axis.
                let lanes = nb.lanes;
                let mut demit = Buf::zeros(3 * n * s);
                for i in 0..s {
                    for a in 0..3 {
                        for p in 0..n {
                            demit[(p * 3 + a) * s + i] = dv[i * n3 + a * n + p].conj();
                        }
                    }
                }
                let mut src = Buf::zeros(s * 3 * lanes * n);
                let mut dhx = Buf::zeros(n * n3);
                let (mut dexd, mut db0d): ([Buf; 3], [Buf; 3]) = (std::array::from_fn(|_| Buf::zeros(n * nd)), std::array::from_fn(|_| Buf::zeros(nd * n)));
                if !nb.xs.is_empty() {
                    let (ln, m) = (lanes * n, 3 * lanes * n);
                    let mut o = Buf::zeros(3 * s * n);
                    fused_product_into(&dct, &basis.sterms, (3, terms, s * n), &mut o);
                    for a in 0..3 {
                        for i in 0..s {
                            src[i * m + a * ln + (lanes - 1) * n..i * m + a * ln + lanes * n].copy_from_slice(&o[a * s * n + i * n..a * s * n + i * n + n]);
                        }
                    }
                    fused_product_into(&dct, &basis.eterms, (3, terms, n * s), &mut o);
                    for a in 0..3 {
                        for p in 0..n {
                            for i in 0..s {
                                demit[(p * 3 + a) * s + i] += o[a * n * s + p * s + i];
                            }
                        }
                    }
                    let mut o = Buf::zeros(3 * n * n);
                    fused_product_into(&dct, &basis.hterms, (3, terms, n * n), &mut o);
                    for a in 0..3 {
                        for p in 0..n {
                            dhx[p * n3 + a * n..p * n3 + a * n + n].copy_from_slice(&o[a * n * n + p * n..a * n * n + p * n + n]);
                        }
                    }
                    for (xx, &(g, k)) in nb.exact.iter().enumerate() {
                        for a in 0..3 {
                            let d = de[g][k][a];
                            for p in 0..n {
                                for e2 in 0..nd {
                                    let y = basis.xcouple[xx][p * nd + e2];
                                    dexd[a][p * nd + e2] += d * y;
                                    db0d[a][e2 * n + p] += (d * y).conj();
                                }
                            }
                        }
                    }
                    work[0] += (3 * terms * (2 * s * n + n * n)) as f64;
                }
                Tangent {
                    dk,
                    demit,
                    dhx,
                    dexd,
                    db0d,
                    src,
                    dz: Buf::zeros(s * 3 * lanes * n),
                    dztot: Buf::zeros(s * n3),
                    dked,
                    dkde,
                    dkdd,
                    ddz: (0..if nd > 0 { lanes } else { 0 }).map(|_| std::array::from_fn(|_| Buf::zeros(nd * n))).collect(),
                    ddztot: std::array::from_fn(|_| Buf::zeros(nd * n)),
                    dtc: std::array::from_fn(|_| Buf::zeros(nb.nc * n)),
                }
            })
            .collect();
        lap(0, &mut clock, &mut tick);
        // The sweeps: ∂H (n × 3n) the Bloch generator's tangent the sweep was solved at, its diagonal ∂c and the rest ∂Δ.
        let mut dh = Buf::copy(&dhbase);
        let mut dd = Buf::zeros(nn);
        for k in 0..sweeps {
            let delta = &ws.deltas[k * nn..(k + 1) * nn];
            for (x, (nb, tg)) in ws.near.iter().zip(tangents.iter_mut()).enumerate() {
                if nb.idle() {
                    continue;
                }
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let splits = nb.lanes;
                let (ln, bn) = (splits * n, nb.splits.len() * n);
                let m = 3 * ln;
                let (t, ends) = (&basis.f.t, &basis.f.ends);
                let at = k * s * ln;
                let (zk, inv) = (&nb.zs[at..at + s * ln], &nb.invs[at..at + s * ln]);
                // The sources: ∂f_l Ỹ_l − Z_l ∂c, every axis and split of a row side by side.
                let df: Vec<[C64; 3]> = nb
                    .splits
                    .iter()
                    .map(|&(cc, jj)| {
                        let comp = &self.components[cc];
                        let l = comp.beams[jj];
                        std::array::from_fn(|a| comp.factor(mid.amplitudes[l].gradient[a]))
                    })
                    .collect();
                if k == 0 {
                    for i in 0..s {
                        for a in 0..3 {
                            for (sp, dfl) in df.iter().enumerate() {
                                let (o, src) = (i * m + a * ln + sp * n, i * bn + sp * n);
                                for (u, y) in tg.src[o..o + n].iter_mut().zip(&basis.beams[src..src + n]) {
                                    *u = dfl[a] * *y;
                                }
                            }
                        }
                    }
                    work[1] += (s * m) as f64;
                }
                let mut rhs = Buf::copy(&tg.src);
                let dhd: [Vec<C64>; 3] = std::array::from_fn(|a| (0..n).map(|j| dh[j * n3 + a * n + j]).collect());
                for i in 0..s {
                    for (a, d) in dhd.iter().enumerate() {
                        for sp in 0..splits {
                            let (o, src) = (i * m + a * ln + sp * n, i * ln + sp * n);
                            for ((u, z), c) in rhs[o..o + n].iter_mut().zip(&zk[src..src + n]).zip(d) {
                                *u -= *z * *c;
                            }
                        }
                    }
                }
                work[1] += (s * m) as f64;
                // ∂K̃ Z on the parts: per part its rows (i, a) of ∂K̃ (3p × p) on Z_P (p × Ln), the result rhs's rows there.
                for (idx, pat, _) in basis.f.parts.iter() {
                    let p = idx.len();
                    let mut mp = Buf::zeros(3 * p * p);
                    for ii in 0..p {
                        for a in 0..3 {
                            let from = a * s * s + (pat + ii) * s + pat;
                            mp[(ii * 3 + a) * p..(ii * 3 + a) * p + p].copy_from_slice(&tg.dk[from..from + p]);
                        }
                    }
                    let mut o = Buf::zeros(3 * p * ln);
                    fused_product_into(&mp, &zk[pat * ln..(pat + p) * ln], (3 * p, p, ln), &mut o);
                    for (r, y) in rhs[pat * m..(pat + p) * m].iter_mut().zip(o.iter()) {
                        *r += *y;
                    }
                    work[1] += (3 * p * p * ln) as f64;
                }
                lap(1, &mut clock, &mut tick);
                // −(∂Z^{k−1}Δ + Z^{k−1}∂Δ): the first on every axis and split at once, the second axis by axis.
                if k > 0 {
                    let h = s;
                    let prev = &nb.zs[at - s * ln..at];
                    let mut p1 = Buf::zeros(h * m);
                    fused_product_into(&tg.dz, delta, (3 * h * splits, n, n), &mut p1);
                    let rows = &mut rhs[..];
                    rows.iter_mut().zip(p1.iter()).for_each(|(r, y)| *r -= *y);
                    work[2] += (h * m * n) as f64;
                    let mut p2 = Buf::zeros(h * ln);
                    for a in 0..3 {
                        for r in 0..n {
                            for c2 in 0..n {
                                dd[r * n + c2] = if r == c2 { C64::ZERO } else { dh[r * n3 + a * n + c2] };
                            }
                        }
                        fused_product_into(prev, &dd, (h * splits, n, n), &mut p2);
                        for i in 0..h {
                            for (r, y) in rows[i * m + a * ln..i * m + a * ln + ln].iter_mut().zip(&p2[i * ln..(i + 1) * ln]) {
                                *r -= *y;
                            }
                        }
                    }
                    work[2] += (h * m * n) as f64;
                }
                lap(2, &mut clock, &mut tick);
                let nd = nb.nd;
                if nd > 0 {
                    for sp in 0..splits {
                        let dat = (k * splits + sp) * nd * n;
                        let zdk = &nb.zds[dat..dat + nd * n];
                        for i in 0..s {
                            for a in 0..3 {
                                for e2 in 0..nd {
                                    let y = tg.dked[a][i * nd + e2];
                                    if y != C64::ZERO {
                                        crate::linalg::mul_add(false, &mut rhs[i * m + a * ln + sp * n..i * m + a * ln + sp * n + n], y, &zdk[e2 * n..e2 * n + n]);
                                    }
                                }
                            }
                        }
                    }
                }
                // The triangular solves, every axis and split of a row at once: (c_j − w_l − T)⁻¹ the forward's.
                {
                    let dz = &mut tg.dz;
                    for i in (0..s).rev().filter(|&i| k == 0 || nb.active[i]) {
                        let (head, tail) = dz.split_at_mut((i + 1) * m);
                        let r = &mut rhs[i * m..(i + 1) * m];
                        for l2 in i + 1..ends[i] {
                            crate::linalg::mul_add(false, r, t[(i, l2)], &tail[(l2 - i - 1) * m..(l2 - i) * m]);
                        }
                        let v = &inv[i * ln..(i + 1) * ln];
                        for a in 0..3 {
                            for ((z, y), w) in head[i * m + a * ln..i * m + a * ln + ln].iter_mut().zip(&r[a * ln..a * ln + ln]).zip(v) {
                                *z = *y * *w;
                            }
                        }
                        work[3] += ((ends[i] - i) * m) as f64;
                    }
                }
                if nd > 0 {
                    // The copies' rows: −(∂Z_D^{k−1}Δ + Z_D^{k−1}∂Δ) + ∂K̂_DE Z_E + ∂K̂_DD Z_D − Z_D∘∂c, by Woodbury on the
                    // forward's P_j and C_j⁻¹, split by split and axis by axis.
                    for sp in 0..splits {
                        let dat = (k * splits + sp) * nd * n;
                        let zdk = &nb.zds[dat..dat + nd * n];
                        let zke = gather(zk, s, ln, sp * n, n);
                        let invl = gather(inv, s, ln, sp * n, n);
                        let (pat, cat) = ((k * splits + sp) * n * nd * s, (k * splits + sp) * n * nd * nd);
                        for a in 0..3 {
                            // The static lane's own source on the copies' rows, its tangent.
                            let mut rd = if sp == nb.splits.len() { Buf::copy(&tg.db0d[a]) } else { Buf::zeros(nd * n) };
                            for e2 in 0..nd {
                                for j in 0..n {
                                    rd[e2 * n + j] -= zdk[e2 * n + j] * dh[j * n3 + a * n + j];
                                }
                                for i in 0..s {
                                    let y = tg.dkde[a][e2 * s + i];
                                    if y != C64::ZERO {
                                        crate::linalg::mul_add(false, &mut rd[e2 * n..e2 * n + n], y, &zke[i * n..i * n + n]);
                                    }
                                }
                                for f2 in 0..nd {
                                    crate::linalg::mul_add(false, &mut rd[e2 * n..e2 * n + n], tg.dkdd[a][e2 * nd + f2], &zdk[f2 * n..f2 * n + n]);
                                }
                                if k > 0 {
                                    let prev = &nb.zds[dat - splits * nd * n..dat - splits * nd * n + nd * n];
                                    for mm in 0..n {
                                        let (dzm, zm) = (tg.ddz[sp][a][e2 * n + mm], prev[e2 * n + mm]);
                                        if dzm != C64::ZERO {
                                            crate::linalg::mul_add(true, &mut rd[e2 * n..e2 * n + n], dzm, &delta[mm * n..mm * n + n]);
                                        }
                                        if zm != C64::ZERO {
                                            for j in (0..n).filter(|&j| j != mm) {
                                                rd[e2 * n + j] -= zm * dh[mm * n3 + a * n + j];
                                            }
                                        }
                                    }
                                }
                            }
                            let mut pj = Buf::copy(&nb.pjs[pat..pat + n * nd * s]);
                            let mut cinv = Buf::copy(&nb.cinvs[cat..cat + n * nd * nd]);
                            let mut out = Buf::zeros(nd * n);
                            let mut dze = gather(&tg.dz, s, m, a * ln + sp * n, n);
                            copies_solve(s, nd, &nb.ked, &nb.kde, &nb.kdd, basis, &[], 0.0, &invl, &rd, &mut dze, &mut out, &mut pj, &mut cinv, true, n, &mut work[3]);
                            scatter(&dze, &mut tg.dz, s, m, a * ln + sp * n, n);
                            tg.ddz[sp][a].copy_from_slice(&out);
                            work[3] += (nd * n * (s + nd + 2 * n)) as f64;
                        }
                    }
                }
                lap(3, &mut clock, &mut tick);
                tg.dztot.fill(C64::ZERO);
                for i in 0..s {
                    for a in 0..3 {
                        let o = &mut tg.dztot[i * n3 + a * n..i * n3 + a * n + n];
                        for sp in 0..splits {
                            for (u, z) in o.iter_mut().zip(&tg.dz[i * m + a * ln + sp * n..i * m + a * ln + sp * n + n]) {
                                *u += *z;
                            }
                        }
                    }
                }
                for a in 0..3 {
                    tg.ddztot[a].fill(C64::ZERO);
                    for d in &tg.ddz {
                        tg.ddztot[a].iter_mut().zip(&d[a]).for_each(|(o, z)| *o += *z);
                    }
                }
                work[4] += (s * m) as f64 / 4.0;
                lap(4, &mut clock, &mut tick);
            }
            // ∂H^{k+1} = ∂Ĥ_base + Σ (∂Ṽ†Z + Ṽ†∂Z): Ṽ† on ∂Z (n × s on s × 3n), and ∂Ṽ†'s rows (p, a) on Z (3n × s on s × n).
            // ∂H^{k+1} = ∂Ĥ_base + Σ (∂Ê Z + Ê ∂Z (+ ∂Ê_D Z_D + Ê_D ∂Z_D + ∂hx)): Ê on ∂Z (n × s on s × 3n), and ∂Ê's rows
            // (p, a) on Z (3n × s on s × n).
            dh.copy_from_slice(&dhbase);
            for (nb, tg) in ws.near.iter().zip(&tangents) {
                if nb.idle() {
                    continue;
                }
                let (s, nd) = (nb.s, nb.nd);
                let ztot = &nb.ztots[k * s * n..(k + 1) * s * n];
                let mut o = Buf::zeros(n * n3);
                fused_product_into(&nb.emit, &tg.dztot, (n, s, n3), &mut o);
                dh.iter_mut().zip(o.iter()).for_each(|(h, y)| *h += *y);
                fused_product_into(&tg.demit, ztot, (n3, s, n), &mut o);
                dh.iter_mut().zip(o.iter()).for_each(|(h, y)| *h += *y);
                work[4] += (2 * s * n * n3) as f64;
                if !nb.xs.is_empty() {
                    dh.iter_mut().zip(tg.dhx.iter()).for_each(|(h, y)| *h += *y);
                    if nd > 0 {
                        let zdk = &nb.zdtots[k * nd * n..(k + 1) * nd * n];
                        let mut q = Buf::zeros(n * n);
                        for a in 0..3 {
                            fused_product_into(&tg.dexd[a], zdk, (n, nd, n), &mut o);
                            fused_product_into(&nb.exd, &tg.ddztot[a], (n, nd, n), &mut q);
                            for p in 0..n {
                                for j in 0..n {
                                    dh[p * n3 + a * n + j] += o[p * n + j] + q[p * n + j];
                                }
                            }
                        }
                        work[4] += (6 * n * nd * n) as f64;
                    }
                }
            }
            lap(4, &mut clock, &mut tick);
        }
        // ∂X = Σ ∂Z†Z + Z†∂Z (+ the copies' rows', the far metric's), the axes side by side as ∂H; S⁻¹'s tangent
        // dN = −N dS N, S dS + dS S = ∂X, axis by axis.
        let mut dx = Buf::copy(&dmetric);
        for (x, (nb, tg)) in ws.near.iter().zip(tangents.iter_mut()).enumerate() {
            if nb.idle() {
                continue;
            }
            let s = nb.s;
            let mut zh = Buf::zeros(n * s);
            adjoint_into(&nb.ztot, s, n, &mut zh);
            let mut o = Buf::zeros(n * n3);
            fused_product_into(&zh, &tg.dztot, (n, s, n3), &mut o);
            dx.iter_mut().zip(o.iter()).for_each(|(h, y)| *h += *y);
            let mut dzh = Buf::zeros(n3 * s);
            for i in 0..s {
                for a in 0..3 {
                    for p in 0..n {
                        dzh[(p * 3 + a) * s + i] = tg.dztot[i * n3 + a * n + p].conj();
                    }
                }
            }
            fused_product_into(&dzh, &nb.ztot, (n3, s, n), &mut o);
            dx.iter_mut().zip(o.iter()).for_each(|(h, y)| *h += *y);
            work[5] += (2 * s * n * n3) as f64;
            for a in 0..3 {
                for d in 0..nb.nd {
                    for p in 0..n {
                        let (dzp, zp) = (tg.ddztot[a][d * n + p].conj(), nb.zdtot[d * n + p].conj());
                        crate::linalg::mul_add(false, &mut dx[p * n3 + a * n..p * n3 + a * n + n], dzp, &nb.zdtot[d * n..d * n + n]);
                        crate::linalg::mul_add(false, &mut dx[p * n3 + a * n..p * n3 + a * n + n], zp, &tg.ddztot[a][d * n..d * n + n]);
                    }
                }
                if nb.copied_in {
                    let basis = &cache.near[x][cache.at_near[x].unwrap()];
                    let (mut dtc, mut extra) = (Buf::zeros(nb.nc * n), Buf::zeros(nb.nc * n));
                    copies_of(&nb.copies, &nb.groups, basis, &mid.e, &gather(&tg.dztot, s, n3, a * n, n), false, &mut dtc, s, n, &mut work[5]);
                    let de_a: Vec<Vec<C64>> = de.iter().map(|c| c.iter().map(|d| d[a]).collect()).collect();
                    copies_of(&nb.copies, &nb.groups, basis, &de_a, &nb.ztot, true, &mut extra, s, n, &mut work[5]);
                    dtc.iter_mut().zip(extra.iter()).for_each(|(o, z)| *o += *z);
                    tg.dtc[a].copy_from_slice(&dtc);
                    for c in 0..nb.nc {
                        for p in 0..n {
                            crate::linalg::mul_add(false, &mut dx[p * n3 + a * n..p * n3 + a * n + n], dtc[c * n + p].conj(), &nb.tc[c * n..c * n + n]);
                            crate::linalg::mul_add(false, &mut dx[p * n3 + a * n..p * n3 + a * n + n], nb.tc[c * n + p].conj(), &dtc[c * n..c * n + n]);
                        }
                    }
                }
            }
        }
        let (mut e_buf, mut p_buf) = (Buf::zeros(nn), Buf::zeros(nn));
        let (mut dxs, mut dss): (Vec<Buf>, Vec<Buf>) = (vec![], vec![]);
        for a in 0..3 {
            let dxa = gather(&dx, n, n3, a * n, n);
            let mut ds = Buf::zeros(nn);
            lyapunov(&step.metric.data, normal, &dxa, n, &mut ds, &mut e_buf, &mut p_buf, &mut work[5]);
            dxs.push(dxa);
            dss.push(ds);
        }
        lap(5, &mut clock, &mut tick);
        ws.tangents = Some(Tangents { blocks: tangents, dx: dxs, ds: dss, de, dc });
        ws.tangent_ticks = tick;
        ws.tangent_work = work;
        work.iter().sum()
    }

    /// The absorbed light's moments of the state ψ at the step's middle (photon.rs # The absorbed light's momentum, its
    /// moments' formula): per block (its photons one frequency) and landing set (`sets`, the ground states' partition) a
    /// class, u_q = C_q ε S⁻¹ψ -- ε = VZ of a near block, R₀Σ_a e_a U_a of a far one -- with its whole gradient ∇u of the
    /// step's tangent (`Fold::prototype_tangent`: ∂Z, ∂e, and S⁻¹'s ∂(S⁻¹ψ) = −S⁻¹ ∂S S⁻¹ψ). The jumps carry each class's
    /// mean up to one absorbed photon (`Photon::kick`'s rule, k_c the class's largest beam wave number): the rest is the
    /// force; 2D = Σ_c M2_c − W_c b_c b_cᵀ of the kicks. After `prototype_tangent`.
    pub(crate) fn prototype_photons(&self, cache: &Cache, ws: &Workspace, psi: &[C64], sets: &[Vec<usize>]) -> Photons {
        let n = ws.n;
        let mut work = 0.0;
        let mid = &ws.mid;
        let normal = &ws.step.normal.data;
        let tg = ws.tangents.as_ref().expect("the step's tangent");
        let (tangents, de) = (&tg.blocks, &tg.de);
        // ζ = S⁻¹ψ, ∂ζ = −S⁻¹ ∂S ζ.
        let zeta: Vec<C64> = (0..n).map(|i| (0..n).map(|j| normal[i * n + j] * psi[j]).sum()).collect();
        let dzeta: Vec<Vec<C64>> = (0..3)
            .map(|a| {
                let q: Vec<C64> = (0..n).map(|i| (0..n).map(|j| tg.ds[a][i * n + j] * zeta[j]).sum()).collect();
                (0..n).map(|i| -(0..n).map(|j| normal[i * n + j] * q[j]).sum::<C64>()).collect()
            })
            .collect();
        work += (7 * n * n) as f64;
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
        for (x, (nb, tg)) in ws.near.iter().zip(tangents.iter()).enumerate() {
            if nb.idle() {
                continue;
            }
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let s = nb.s;
            let mut absorbed = 0.0_f64;
            for &(cc, jj) in &nb.splits {
                absorbed = absorbed.max(self.light.beams[self.components[cc].beams[jj]].wave_number);
            }
            let y: Vec<C64> = (0..s).map(|i| (0..n).map(|j| nb.ztot[i * n + j] * zeta[j]).sum()).collect();
            let dy: Vec<Vec<C64>> = (0..3).map(|a| (0..s).map(|i| (0..n).map(|j| tg.dztot[i * 3 * n + a * n + j] * zeta[j] + nb.ztot[i * n + j] * dzeta[a][j]).sum()).collect()).collect();
            for q in 0..3 {
                let e = &basis.emit[q];
                let ex = &basis.xemitc[q];
                for b in 0..GROUND {
                    let row = &e[b * s..b * s + s];
                    u[q * GROUND + b] = row.iter().zip(&y).map(|(p, z)| *p * *z).sum();
                    du[q * GROUND + b] = std::array::from_fn(|a| row.iter().zip(&dy[a]).map(|(p, z)| *p * *z).sum());
                    // The explicit states' own rows of ε (units): C_q on them, of ζ and ∂ζ.
                    for &p in &nb.xs {
                        let c = ex[b * n + p];
                        u[q * GROUND + b] += c * zeta[p];
                        for a in 0..3 {
                            du[q * GROUND + b][a] += c * dzeta[a][p];
                        }
                    }
                }
            }
            work += (7 * s * n + 12 * GROUND * s) as f64;
            for set in sets {
                class(&u, &du, set, absorbed, &mut out);
            }
            work += (3 * GROUND * 13) as f64;
        }
        for (x, fb) in ws.far.iter().enumerate().filter(|(_, fb)| !fb.omitted) {
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
            out.push(std::array::from_fn(|q| {
                let mut m = Mat { rows: GROUND, cols: nb.s, data: basis.emit[q].clone() }.product(&zm);
                // The explicit states' own rows of ε: C_q on them.
                m.add_scaled(C64::ONE, &Mat { rows: GROUND, cols: n, data: basis.xemitc[q].clone() });
                m.product(normal)
            }));
        }
        for (x, fb) in ws.far.iter().enumerate() {
            if fb.omitted {
                out.push(std::array::from_fn(|_| Mat::zeros(GROUND, n)));
                continue;
            }
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

    /// Per near block at the last step: its kept rows, lanes and terms.
    pub(crate) fn kept_rows(&self) -> Vec<(usize, usize, usize)> {
        self.near.iter().map(|x| (x.s, x.lanes, x.terms.len())).collect()
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
            let (ne, nd) = (nb.all, nb.nd);
            // The kept rows' response on their rows of the block, the omitted levels' zero.
            let mut place = vec![usize::MAX; ne];
            for (a, &i) in basis.rows.iter().enumerate() {
                place[i] = a;
            }
            out[nb.o] = Some(Mat::from_fn(ne + nd + nb.nc, n, |i, j| {
                if i < ne {
                    if place[i] == usize::MAX { C64::ZERO } else { ts[(place[i], j)] }
                } else if i < ne + nd {
                    nb.zdtot[(i - ne) * n + j]
                } else {
                    rows.tc[(i - ne - nd) * n + j]
                }
            }));
        }
        for (x, fb) in self.far.iter().enumerate() {
            if fb.omitted {
                out[fb.o] = Some(Mat::zeros(fb.m, n));
                continue;
            }
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
