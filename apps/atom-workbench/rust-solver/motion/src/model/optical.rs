//! The optical sector's momentum noise (the momentum diffusion "Q"; design: refs/diffusion/DESIGN.md, claude/motion-v2-diffdesign).
//!
//! # The split by the model's own structure
//! The exact first-order correlation of the force at a point is D_x = Re Tr[𝓕 𝓡 X̄], 𝓡 = (−𝓛)⁻¹ on the traceless
//! densities, X̄ = 𝓕[ρ̄] − F̄ρ̄; a trajectory carries its Ehrenfest mean's own fluctuation D_E, and what it drops of a state
//! ψ is within(ψ) = Re Tr[𝓕 𝓡 X_ψ], X_ψ = 𝓕[ψψ†] − ⟨F⟩ψψ† (D_x = D_E + E[within]). The densities split by the model's
//! structure into Q -- every entry with an excited index, and the ground trace -- and P, the ground manifold's traceless
//! part (its population differences and its coherences): the slow optical pumping a folded cluster keeps. 𝓡_Q =
//! Q(−Q𝓛Q)⁻¹Q relaxes at about Γ/2 or faster: every Q direction holds an excited index or the excited trace (measured
//! 19.06–19.7/µs on three models; that it holds in general is the design's hypothesis). Its part of a state's correlation,
//! within_Q(ψ) = Re Tr[𝓕 𝓡_Q X_ψ], is local: the trajectory takes it as white noise of that rate, and its mean over the
//! trajectories is Re Tr[𝓕 𝓡_Q E[X_ψ]] by linearity, whatever the state's own fast dynamics. The rest passes through the
//! ground manifold's own state, which the trajectory keeps (the Ehrenfest error of the ground manifold's superpositions, a
//! model limitation the ledger names). A folded cluster has no Q direction a state of it enters: its fold holds that part
//! as its photons' local second moment (photon.rs).
//!
//! # The ground trace: the state's own cluster's
//! Q takes a state's ground trace as the identity on the ground states of the cluster it lies in, over their number, d =
//! 1_g/g: Q X = E X + (Tr_g X) d, a projector whatever X's other blocks (P = 1 − Q keeps every ground block's traceless part
//! and every other cluster's ground populations). The excited states' decay is read as landing where their own cluster's
//! drives take it up again -- a cycling transition's photon does, and a leak's is the rare event P carries. The split is
//! exact in the mean for any projector (above); this one leaves the least to P where it was compared: on R2_RECOIL's
//! kept probe (a two-state cluster beside seven other ground states) the trajectory's D_E + E[within_Q] reads the folded
//! drive's absorbed diffusion to 3·10⁻⁴, where the trace spread over every ground state of the model read it to 8·10⁻³ --
//! the mean ground force over states the atom does not visit (tests, `a_kept_and_a_folded_drive_diffuse_alike`). And a
//! cluster's K is its own: no other cluster at the point enters.
//!
//! # K_Q
//! within_Q(ψ) = Re Tr[K X_ψ] for the Hermitian K = 𝓡_Q†[F] (one per lab axis), F = −G the force of the generator's
//! Hermitian part (the Ehrenfest force the trajectory carries; 𝓕 = ½{F, ·}): within_Q,ab(ψ) = Re⟨K_a ψ|F_b ψ⟩/‖ψ‖² −
//! ⟨K_a⟩⟨F_b⟩. K is defined up to the identity (X_ψ is traceless); it is taken with no ground-ground entries.
//!
//! In the coordinates y_E of the entries with an excited index (a Hermitian y by its excited block's upper triangle and
//! its excited-ground block), a traceless Q density is y_E − τ d, τ = Tr y_E the excited trace, and −Q𝓛Q on it is
//!
//! ```text
//!   A′ y = A y − (τ/g) b,    A y = E[𝓛 y],   b = E[𝓛 1_g],
//! ```
//! E the entries with an excited index: the jumps land in ground states and take no part (only H, the generator with its
//! decay, enters), and the ground trace's equation is the trace's, which 𝓛 keeps. The border is of rank one -- a scalar
//! Schur complement (Sherman–Morrison). within_Q = Tr[F′ Y] with F′ = F less the mean φ̄ of its diagonal over the
//! cluster's ground states (the identity is free: F′ pairs with Y's E part alone), so K = −W⁻¹ A′⁻ᵀ W f′ in the coordinates
//! (W the trace form's weights: one on the excited block's diagonal, two on a pair of entries' real and imaginary parts):
//!
//! ```text
//!   K = −W⁻¹ A⁻ᵀ (W f + μ s),   μ = ζ/g − φ̄,   ζ = (u·W f − φ̄ u·s)/(1 − u·s/g),
//! ```
//! u = A⁻¹ b, s the excited trace's coordinates. Every vector A⁻¹ enters is paired with W f or s, u·v = b·A⁻ᵀv, and
//! Aᵀ W p = W E[𝓛†P] for the Hermitian P of coordinates p (the trace form's adjoint, 𝓛†P = i(H†P − PH)). In matrices:
//!
//! ```text
//!   K = −(P + μ S),   E[𝓛†P] = F_E,   E[𝓛†S] = 1_e,   u·W f = Tr[B P],   u·s = Tr[B S],
//! ```
//! P and S Hermitian with no ground-ground block, B the matrix of b (B_eg = −iH_eg, its excited block none).
//!
//! # The solve: the excited-ground block eliminated
//! By blocks (g ground, e excited), E[𝓛†P] = R is
//!
//! ```text
//!   eg:  H_ee† P_eg − P_eg H_gg = −i R_eg + P_ee H_eg,
//!   ee:  i(H_ee† P_ee − P_ee H_ee) + i(Z† − Z) = R_ee,   Z = P_eg H_ge.
//! ```
//! The first is a Sylvester equation 𝒮(P_eg) = H_ee†P_eg − P_eg H_gg = C whose spectrum, conj(λ_e) − λ_g over the blocks'
//! eigenvalues, lies the excited states' decay over two or more off the real axis: in the blocks' complex Schur forms
//! H_ee = U_e T U_e†, H_gg = U_g S U_g† it is a triangular substitution, row k of X from X_k A_k = C_k − Σ_{l<k} conj(T_lk)
//! X_l with A_k = conj(T_kk) − S upper triangular, e g (e + g)/2 complex multiply-adds (Bartels and Stewart; unitary
//! rotations within the blocks keep the entries with an excited index, the traces and the trace form, so the whole solve
//! is made in the rotated blocks and K rotated back). Its solution put into the second leaves a real linear system on the
//! Hermitian P_ee alone, e² unknowns: ℳ(P) = i(T†P − P T) + i(Z† − Z), Z = L(P) = 𝒮⁻¹(P C) B, C = H̃_eg and B = H̃_ge the
//! couplings in the rotated blocks. The whole sector's factorization, m = e² + 2eg reals and m³/3 multiply-adds, is the
//! eliminated block's: S3's cluster of five ground and six excited states factors 36 reals for 96.
//!
//! ℳ is made at once, not basis matrix by basis matrix. L is complex-linear, and of E_ij C -- one row, i, holding C's row j
//! -- the substitution starts at row i and carries C's row j through the same matrices whatever j: X_k = C_j W_ik for
//! k ≥ i (no row above i), W_ii A_i = 1, W_ik A_k = −Σ_{l=i}^{k−1} conj(T_lk) W_il. So with Y_ik = C W_ik (e × g, every j
//! at once: Y_ik A_k = δ_ik C − Σ_l conj(T_lk) Y_il) and V_ik = Y_ik B (e × e), L(E_ij) holds V_ik's row j in its row k for
//! every k ≥ i: the e(e + 1)/2 pairs i ≤ k make every column of ℳ, some e³g/6 + e²g²/4 + e³g/2 complex multiply-adds (S3:
//! 6.4 thousand, where the basis matrices one by one took a substitution and three products of e² g each, 35 thousand).
//! The Hermitian basis takes E_ii, E_ij + E_ji and i(E_ij − E_ji) of them, and the T part is a row and a column of T. ℳ
//! is factored (`Lu`, by panels) and its sides solved up to four at once.
//!
//! The sides in the rotated blocks: of the forces' F̃ = U†FU (U = U_g ⊕ U_e) its excited-excited and excited-ground
//! blocks, and of its ground block the trace alone (the mean r̄, unitarily invariant); along the path R = F̃₁ +
//! i(H̃₁†K̃ − K̃H̃₁) by the blocks of K̃ -- two, it has no ground-ground block -- and of H̃₁. K is rotated back once, K₀ and
//! its first order summed.
//!
//! # The workspace
//! A trajectory holds one sector (step.rs `Scratch`): its memory is made once, as large as the largest cluster it meets
//! asks, and every make writes what it reads.
//!
//! # The count
//! A make's real multiply-adds are one formula of the cluster's shape (`Sector::cost`: its e excited and g ground states,
//! whether it beats) and of its blocks' Schur forms, whose QR iteration converges as the matrices ask: the step counts
//! the make by it, and a price (step.rs `step_work`) by the same formula with the two Schur forms alone made -- no
//! solve.

use crate::linalg::{fused_product, mul_add, Mat, C64};
use crate::decide::COMPLEX;

/// A real square matrix's LU factorization with partial pivoting, P T = L U, T row by row, in its place: L unit lower,
/// U upper (Higham, Theorem 9.3: backward stable, the factors those of T + ΔT, |ΔT| ≤ γ_m |L||U|). By panels of `PANEL`
/// columns (LAPACK's right-looking getrf): a panel is eliminated with the rows exchanged whole -- the pivots are the
/// unblocked elimination's, every column it takes is up to date --, its rows of U right of it by its unit lower
/// triangle, and the rest of the matrix takes the panel's columns of L times those rows at once: four multiply-adds
/// for every number the trailing block reads and writes, which the elimination row by row does once -- it is held by
/// its memory's bandwidth, not by the arithmetic. The solves go four right-hand sides at once, one vector a row.
#[derive(Default)]
struct Lu {
    m: usize,
    lu: Vec<f64>,
    pivots: Vec<usize>,
}

/// The columns a panel of `Lu` eliminates at once: the vectors of four reals the trailing update reads per number.
const PANEL: usize = 4;

impl Lu {
    /// The m × m matrix in `lu` (row by row) factored in its place; false where a pivot is zero or no number.
    fn factor(&mut self, m: usize) -> bool {
        self.m = m;
        self.pivots.resize(m, 0);
        let (t, pivots) = (&mut self.lu[..m * m], &mut self.pivots);
        for k0 in (0..m).step_by(PANEL) {
            let k1 = (k0 + PANEL).min(m);
            // The panel: its columns eliminated below the diagonal, the rows exchanged whole.
            for k in k0..k1 {
                let p = (k..m).max_by(|&x, &y| t[x * m + k].abs().total_cmp(&t[y * m + k].abs())).unwrap_or(k);
                pivots[k] = p;
                if p != k {
                    let (above, below) = t.split_at_mut(p * m);
                    above[k * m..(k + 1) * m].swap_with_slice(&mut below[..m]);
                }
                let (above, below) = t.split_at_mut((k + 1) * m);
                let (pivot, upper) = (above[k * m + k], &above[k * m + k + 1..k * m + k1]);
                if !(pivot != 0.0 && pivot.is_finite()) {
                    return false;
                }
                for row in below.chunks_exact_mut(m) {
                    let f = row[k] / pivot;
                    row[k] = f;
                    for (x, u) in row[k + 1..k1].iter_mut().zip(upper) {
                        *x = (-f).mul_add(*u, *x);
                    }
                }
            }
            // The panel's rows of U right of it: its unit lower triangle's forward substitution.
            for k in k0 + 1..k1 {
                let (above, row) = t.split_at_mut(k * m);
                let (multipliers, right) = row[k0..m].split_at_mut(k - k0);
                for (j, l) in (k0..k).zip(multipliers.iter()) {
                    for (x, u) in right[k1 - k..].iter_mut().zip(&above[j * m + k1..(j + 1) * m]) {
                        *x = (-l).mul_add(*u, *x);
                    }
                }
            }
            // The trailing block less the panel's L times its rows of U.
            let (above, below) = t.split_at_mut(k1 * m);
            let upper: [&[f64]; PANEL] = std::array::from_fn(|j| if k0 + j < k1 { &above[(k0 + j) * m + k1..(k0 + j + 1) * m] } else { &[] });
            for row in below.chunks_exact_mut(m) {
                let (multipliers, right) = row[k0..].split_at_mut(k1 - k0);
                if k1 - k0 == PANEL {
                    let [l0, l1, l2, l3] = [multipliers[0], multipliers[1], multipliers[2], multipliers[3]];
                    for ((((x, u0), u1), u2), u3) in right.iter_mut().zip(upper[0]).zip(upper[1]).zip(upper[2]).zip(upper[3]) {
                        *x = (-l3).mul_add(*u3, (-l2).mul_add(*u2, (-l1).mul_add(*u1, (-l0).mul_add(*u0, *x))));
                    }
                } else {
                    for (l, u) in multipliers.iter().zip(&upper) {
                        for (x, u) in right.iter_mut().zip(u.iter()) {
                            *x = (-l).mul_add(*u, *x);
                        }
                    }
                }
            }
        }
        true
    }
    /// T X = B in place, B of four columns row by row.
    fn solve(&self, b: &mut [[f64; 4]]) {
        let (m, f) = (self.m, &self.lu);
        for (k, &p) in self.pivots[..m].iter().enumerate() {
            b.swap(k, p);
        }
        for i in 0..m {
            let (above, row) = b.split_at_mut(i);
            let row = &mut row[0];
            for (l, other) in f[i * m..i * m + i].iter().zip(above.iter()) {
                for (z, o) in row.iter_mut().zip(other) {
                    *z = (-l).mul_add(*o, *z);
                }
            }
        }
        for i in (0..m).rev() {
            let (rest, below) = b.split_at_mut(i + 1);
            let row = &mut rest[i];
            for (u, other) in f[i * m + i + 1..(i + 1) * m].iter().zip(below.iter()) {
                for (z, o) in row.iter_mut().zip(other) {
                    *z = (-u).mul_add(*o, *z);
                }
            }
            let d = f[i * m + i];
            for z in row.iter_mut() {
                *z /= d;
            }
        }
    }
}

/// The r × c block of an n × n matrix (row by row) at its rows from `r0` and its columns from `c0`, into `out`.
fn block(m: &[C64], n: usize, (r0, r): (usize, usize), (c0, c): (usize, usize), out: &mut [C64]) {
    for (o, row) in out[..r * c].chunks_exact_mut(c).zip(m[r0 * n..].chunks_exact(n)) {
        o.copy_from_slice(&row[c0..c0 + c]);
    }
}

/// X A_k⁻¹ of a row X in its place (module documentation, # The solve): A_k = conj(T_kk) − S, S upper triangular (g × g),
/// by the reciprocals `inverse` of A_k's diagonal, 1/(conj(T_kk) − S_qq).
fn right(inverse: &[C64], s: &[C64], g: usize, row: &mut [C64]) {
    for q in 0..g {
        let mut v = row[q];
        for p in 0..q {
            v += row[p] * s[p * g + q];
        }
        row[q] = v * inverse[q];
    }
}

/// 𝒮⁻¹ in its place: T†X − X S = C of an e × g C (module documentation, # The solve), its rows downwards; `inverse` the
/// reciprocals 1/(conj(T_kk) − S_qq), row k by row.
fn substitute(t: &[C64], s: &[C64], inverse: &[C64], (e, g): (usize, usize), x: &mut [C64]) {
    for k in 0..e {
        let (above, row) = x.split_at_mut(k * g);
        let row = &mut row[..g];
        for (l, xl) in above.chunks_exact(g).enumerate() {
            mul_add(true, row, t[l * e + k].conj(), xl);
        }
        right(&inverse[k * g..(k + 1) * g], s, g, row);
    }
}

/// The places of the Hermitian e × e matrices' real coordinates, in order: per i its diagonal, then per pair i < j the
/// real and the imaginary part of the entry (i, j).
fn places(e: usize) -> impl Iterator<Item = (usize, usize, u8)> {
    (0..e).flat_map(move |i| std::iter::once((i, i, 0)).chain((i + 1..e).flat_map(move |j| [(i, j, 1), (i, j, 2)])))
}

/// One cluster's part of the optical sector at a point (module documentation, # K_Q, # The solve), made in a
/// workspace (# The workspace): from its generator H (with its decay) and the gradient G_a of its Hermitian part (n × n,
/// row by row), its first `ground` states ground, and with their slopes along the path K's first order.
#[derive(Default)]
pub(crate) struct Sector {
    n: usize,
    g: usize,
    e: usize,
    /// Whether K was made: a cluster with excited and ground states, its blocks' Schur forms converged, ℳ regular.
    made: bool,
    /// The Schur forms T (e × e) and S (g × g), upper triangular, and their unitary factors U and U† (row by row);
    /// the Schur form's work vectors.
    t: Vec<C64>,
    s: Vec<C64>,
    /// The substitutions' reciprocals 1/(conj(T_kk) − S_qq) (e × g).
    inverse: Vec<C64>,
    ue: Vec<C64>,
    ue_h: Vec<C64>,
    ug: Vec<C64>,
    ug_h: Vec<C64>,
    q: Vec<C64>,
    v: Vec<C64>,
    sums: Vec<C64>,
    /// The rotated couplings C = H̃_eg (e × g) and B = H̃_ge (g × e).
    c: Vec<C64>,
    b: Vec<C64>,
    /// Y_ik of one i (k ≥ i, e × g each) and V_ik of every i ≤ k (e × e, at i e + k).
    y: Vec<C64>,
    vv: Vec<C64>,
    /// ℳ's factorization, and its right-hand sides.
    lu: Lu,
    rhs: Vec<[f64; 4]>,
    /// Up to four sides (R_ee e², R_eg e g, the ground diagonal's mean) and their solutions (P_ee, P_eg, Tr[B P]),
    /// with X₀ = 𝒮⁻¹(−iR_eg).
    sides: [Vec<C64>; 2],
    means: [f64; 4],
    solved: [Vec<C64>; 2],
    paired: [f64; 4],
    x0: Vec<C64>,
    /// S's blocks and Tr[B S].
    s_ee: Vec<C64>,
    s_eg: Vec<C64>,
    returned: f64,
    /// K's rotated blocks per axis (3 e², 3 e g), and H̃₁'s four blocks (ee, eg, ge, gg).
    k_ee: Vec<C64>,
    k_eg: Vec<C64>,
    h1: Vec<C64>,
    /// Work matrices (n² each).
    z: Vec<C64>,
    tmp: [Vec<C64>; 2],
    /// K_a in the cluster's own basis (n × n, row by row, Hermitian, no ground-ground block).
    k: [Vec<C64>; 3],
    /// The real multiply-adds the last make took.
    pub(crate) work: f64,
}

/// Why a sector made no K (`Sector::make`): a block's Schur form not made, ℳ singular (a zero pivot of its
/// factorization), or a number of K that is no finite number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unmade {
    Schur,
    Singular,
    Unfinite,
}

impl Sector {
    /// K at a point (module documentation, # K_Q, # The solve), and with `slope` -- the generator's and the gradient's
    /// slopes along the path at the point, dH/ds and dG_a/ds -- its first order along the path (# Along the path; step.rs,
    /// # The optical sector's noise): K + 𝓡_Q†𝓡_Q†(dF/ds + (d𝓛/ds)†K). Where it makes none, why (`Unmade`): a block's
    /// Schur form not made -- its QR iteration did not converge, or the cluster has no excited or no ground state (no
    /// trajectory lies in the latter: it starts in a ground state, and every jump lands in one) --, ℳ singular, or no
    /// number came out. None of them is to happen: every Q direction relaxes at about Γ/2 or faster (module
    /// documentation, # The split by the model's own structure), so ℳ is regular, and a step that meets one is refused
    /// (step.rs `Refused::Optical`) -- its noise is not dropped.
    pub(crate) fn make(&mut self, h: &Mat, gradient: &[Vec<C64>; 3], ground: usize, slope: Option<(&Mat, &[Vec<C64>; 3])>) -> Result<(), Unmade> {
        let n = h.rows;
        let (g, e) = (ground, n - ground);
        (self.n, self.g, self.e, self.made, self.work) = (n, g, e, false, 0.0);
        let schur = self.schur(h, ground).ok_or(Unmade::Schur)?;
        for k in 0..e {
            for q in 0..g {
                self.inverse[k * g + q] = (self.t[k * e + k].conj() - self.s[q * g + q]).recip();
            }
        }
        // The rotated couplings C = U_e† H_eg U_g and B = U_g† H_ge U_e.
        let [one, two] = &mut self.tmp;
        block(&h.data, n, (g, e), (0, g), one);
        fused_product(&self.ue_h, one, (e, e, g), two);
        fused_product(two, &self.ug, (e, g, g), &mut self.c);
        block(&h.data, n, (0, g), (g, e), one);
        fused_product(&self.ug_h, one, (g, g, e), two);
        fused_product(two, &self.ue, (g, e, e), &mut self.b);
        // ℳ, and its factorization.
        self.system();
        if !self.lu.factor(e * e) {
            return Err(Unmade::Singular);
        }
        // K₀: the three axes' F = −G, by blocks, and beside them S: E[𝓛†S] = 1_e, solved at once; S and Tr[B S] first
        // into their places, which the resolve reads.
        for a in 0..3 {
            let f: &[C64] = &gradient[a];
            self.side(a, f, -1.0);
        }
        let (ee, eg) = (e * e, e * g);
        self.sides[0][3 * ee..4 * ee].fill(C64::ZERO);
        self.sides[1][3 * eg..4 * eg].fill(C64::ZERO);
        for i in 0..e {
            self.sides[0][3 * ee + i * e + i] = C64::ONE;
        }
        self.solve(4);
        self.s_ee[..ee].copy_from_slice(&self.solved[0][3 * ee..4 * ee]);
        self.s_eg[..eg].copy_from_slice(&self.solved[1][3 * eg..4 * eg]);
        self.returned = self.paired[3];
        self.resolve(3, true);
        // Along the path: R = F̃₁ + i(H̃₁†K̃ − K̃H̃₁), then 𝓡_Q† twice.
        if let Some((h1, g1)) = slope {
            self.rotate_slope(&h1.data);
            for a in 0..3 {
                let f: &[C64] = &g1[a];
                self.side(a, f, -1.0);
                self.turned(a);
            }
            for _ in 0..2 {
                self.solve(3);
                self.resolve(3, false);
                self.means[..3].fill(0.0);
            }
            // K̃ += the outer resolve's.
            let (ee, eg) = (e * e, e * g);
            for a in 0..3 {
                for (k, d) in self.k_ee[a * ee..(a + 1) * ee].iter_mut().zip(&self.sides[0][a * ee..(a + 1) * ee]) {
                    *k += *d;
                }
                for (k, d) in self.k_eg[a * eg..(a + 1) * eg].iter_mut().zip(&self.sides[1][a * eg..(a + 1) * eg]) {
                    *k += *d;
                }
            }
        }
        // Rotated back, K_ee = U_e K̃_ee U_e† and K_eg = U_e K̃_eg U_g†.
        self.back();
        if !(self.returned.is_finite() && self.k.iter().all(|k| k[..n * n].iter().all(|z| z.is_finite()))) {
            return Err(Unmade::Unfinite);
        }
        self.work = Sector::cost(e, g, slope.is_some(), schur);
        self.made = true;
        Ok(())
    }

    /// The blocks' Schur forms H_ee = U_e T U_e† and H_gg = U_g S U_g† of a generator with `ground` ground states, in the
    /// workspace, and their complex multiply-adds; None where a QR iteration does not converge (or the cluster has no
    /// excited or no ground state).
    pub(crate) fn schur(&mut self, h: &Mat, ground: usize) -> Option<f64> {
        let n = h.rows;
        let (g, e) = (ground, n - ground);
        (self.n, self.g, self.e, self.made) = (n, g, e, false);
        if e == 0 || g == 0 {
            return None;
        }
        self.fit();
        let mut complex = 0.0;
        for (size, at) in [(e, g), (g, 0)] {
            let (t, u, u_h) = if at == g { (&mut self.t, &mut self.ue, &mut self.ue_h) } else { (&mut self.s, &mut self.ug, &mut self.ug_h) };
            block(&h.data, n, (at, size), (at, size), t);
            let (work, _) = crate::linalg::schur(&mut t[..size * size], &mut self.q[..size * size], &mut self.v[..size], &mut self.sums[..size], size)?;
            complex += work as f64;
            // Q column by column is U† row by row once conjugated, and U row by row once transposed.
            for i in 0..size {
                for k in 0..size {
                    let z = self.q[k * size + i];
                    u[i * size + k] = z;
                    u_h[k * size + i] = z.conj();
                }
            }
        }
        Some(complex)
    }

    /// The real multiply-adds of a make (module documentation, # The count) on e excited and g ground states, with the
    /// first order along the path (`sloped`), its Schur forms taking `schur` complex multiply-adds: the couplings'
    /// rotations and the substitutions' reciprocals; ℳ -- Y_ik's substitutions and sums and V_ik's products over the pairs i ≤ k, the columns' sums of V's
    /// rows and their m places -- and its factorization; the sides -- S's, the three forces' rotations, and along the
    /// path H̃₁'s four blocks, the slopes' rotations and i(H̃₁†K̃ − K̃H̃₁) --, each side's two substitutions and two
    /// products, its sum Tr[B P], and ℳ's solve; K rotated back.
    pub(crate) fn cost(e: usize, g: usize, sloped: bool, schur: f64) -> f64 {
        let (m, ee, eg) = (e * e, e * e, e * g);
        let pairs: usize = (0..e).map(|i| (i..e).map(|k| (k - i) * eg + e * g * (g + 1) / 2 + ee * g).sum::<usize>()).sum();
        let columns: usize = places(e).map(|(i, j, kind)| if kind == 0 { (e - i) * e } else { (2 * e - i - j) * e } + m).sum();
        let rotation = 2 * e * ee + e * eg + eg * g;
        let sides = if sloped { 10 } else { 4 };
        let side = 2 * (e * g * (e - 1) / 2 + e * g * (g + 1) / 2) + 2 * ee * g + eg;
        let slope = if sloped { 2 * (e + g) * (ee + g * g) + 3 * (rotation + 2 * ee * (e + g) + eg * (2 * e + g) + eg) } else { 0 };
        let complex = schur + (eg + 2 * (ee * g + e * g * g) + pairs + columns + 3 * rotation + slope + sides * side + 3 * rotation) as f64;
        COMPLEX * complex + (m * m * m) as f64 / 3.0 + (sides * 2 * m * m) as f64
    }

    /// K_a (n × n, row by row, Hermitian; module documentation, # K_Q) of the last make, None where it made none.
    pub(crate) fn operators(&self) -> Option<&[Vec<C64>; 3]> {
        self.made.then_some(&self.k)
    }

    /// The workspace for the cluster of the make.
    fn fit(&mut self) {
        let (n, g, e) = (self.n, self.g, self.e);
        let size = |v: &mut Vec<C64>, len: usize| {
            if v.len() < len {
                v.resize(len, C64::ZERO);
            }
        };
        let ([s0, s1], [v0, v1], [t0, t1]) = (&mut self.sides, &mut self.solved, &mut self.tmp);
        for (v, len) in [
            (&mut self.t, e * e),
            (&mut self.s, g * g),
            (&mut self.inverse, e * g),
            (&mut self.ue, e * e),
            (&mut self.ue_h, e * e),
            (&mut self.ug, g * g),
            (&mut self.ug_h, g * g),
            (&mut self.q, n * n),
            (&mut self.v, n),
            (&mut self.sums, n),
            (&mut self.c, e * g),
            (&mut self.b, e * g),
            (&mut self.y, e * e * g),
            (&mut self.vv, e * e * e * e),
            (s0, 4 * e * e),
            (s1, 4 * e * g),
            (v0, 4 * e * e),
            (v1, 4 * e * g),
            (&mut self.x0, 4 * e * g),
            (&mut self.s_ee, e * e),
            (&mut self.s_eg, e * g),
            (&mut self.k_ee, 3 * e * e),
            (&mut self.k_eg, 3 * e * g),
            (&mut self.h1, n * n),
            (&mut self.z, n * n),
            (t0, n * n),
            (t1, n * n),
        ] {
            size(v, len);
        }
        for k in &mut self.k {
            k.resize(n * n, C64::ZERO);
        }
        if self.lu.lu.len() < e * e * e * e {
            self.lu.lu.resize(e * e * e * e, 0.0);
        }
        if self.rhs.len() < e * e {
            self.rhs.resize(e * e, [0.0; 4]);
        }
    }

    /// ℳ into the factorization's place (module documentation, # The solve: V_ik for every i ≤ k, then its columns on the
    /// Hermitian basis).
    fn system(&mut self) {
        let (e, g) = (self.e, self.g);
        let (eg, ee, m) = (e * g, e * e, e * e);
        for i in 0..e {
            for k in i..e {
                let (done, rest) = self.y.split_at_mut((k - i) * eg);
                let yk = &mut rest[..eg];
                if k == i {
                    yk.copy_from_slice(&self.c[..eg]);
                } else {
                    yk.fill(C64::ZERO);
                    for (l, yl) in (i..k).zip(done.chunks_exact(eg)) {
                        mul_add(false, yk, -self.t[l * e + k].conj(), yl);
                    }
                }
                for row in yk.chunks_exact_mut(g) {
                    right(&self.inverse[k * g..(k + 1) * g], &self.s, g, row);
                }
                fused_product(yk, &self.b, (e, g, e), &mut self.vv[(i * e + k) * ee..(i * e + k + 1) * ee]);
            }
        }
        // The columns: Z = L(P) of the basis matrix, then i(T†P − PT) + i(Z† − Z) on the coordinates.
        let (t, vv, z, lu) = (&self.t, &self.vv, &mut self.z[..ee], &mut self.lu.lu);
        let add = |z: &mut [C64], i: usize, j: usize, f: C64| {
            for k in i..e {
                mul_add(false, &mut z[k * e..(k + 1) * e], f, &vv[(i * e + k) * ee + j * e..(i * e + k) * ee + (j + 1) * e]);
            }
        };
        // i(T†E_ab − E_ab T) at (r, c): conj(T_ar) where c = b, less T_bc where r = a (T upper triangular).
        let part = |a: usize, b: usize, r: usize, c: usize| -> C64 {
            let left = if c == b && a <= r { t[a * e + r].conj() } else { C64::ZERO };
            let right = if r == a && b <= c { t[b * e + c] } else { C64::ZERO };
            left - right
        };
        for (column, (i, j, kind)) in places(e).enumerate() {
            z.fill(C64::ZERO);
            match kind {
                0 => add(z, i, i, C64::ONE),
                1 => {
                    add(z, i, j, C64::ONE);
                    add(z, j, i, C64::ONE);
                }
                _ => {
                    add(z, i, j, C64::I);
                    add(z, j, i, -C64::I);
                }
            }
            // The coordinates in order (`places`): per r its diagonal, then per c > r the real and imaginary part.
            let mut row = 0;
            for r in 0..e {
                for c in r..e {
                    let p = match kind {
                        0 => part(i, i, r, c),
                        1 => part(i, j, r, c) + part(j, i, r, c),
                        _ => C64::I * (part(i, j, r, c) - part(j, i, r, c)),
                    };
                    let q = C64::I * (p + z[c * e + r].conj() - z[r * e + c]);
                    lu[row * m + column] = q.re;
                    if c > r {
                        lu[(row + 1) * m + column] = q.im;
                        row += 1;
                    }
                    row += 1;
                }
            }
        }
    }

    /// Side `a` of the forces' kind from an n × n matrix M (row by row) times `sign`: (U_e† M_ee U_e, U_e† M_eg U_g)
    /// and the mean of M's ground diagonal.
    fn side(&mut self, a: usize, m: &[C64], sign: f64) {
        let (n, g, e) = (self.n, self.g, self.e);
        let (ee, eg) = (e * e, e * g);
        let [one, two] = &mut self.tmp;
        block(m, n, (g, e), (g, e), one);
        fused_product(&self.ue_h, one, (e, e, e), two);
        fused_product(two, &self.ue, (e, e, e), &mut self.sides[0][a * ee..(a + 1) * ee]);
        block(m, n, (g, e), (0, g), one);
        fused_product(&self.ue_h, one, (e, e, g), two);
        fused_product(two, &self.ug, (e, g, g), &mut self.sides[1][a * eg..(a + 1) * eg]);
        if sign != 1.0 {
            let [s0, s1] = &mut self.sides;
            for z in s0[a * ee..(a + 1) * ee].iter_mut().chain(&mut s1[a * eg..(a + 1) * eg]) {
                *z = *z * sign;
            }
        }
        self.means[a] = sign * (0..g).map(|i| m[i * n + i].re).sum::<f64>() / g as f64;
    }

    /// H̃₁'s four blocks (ee, eg, ge, gg, one after another) of the slope dH/ds (n × n, row by row).
    fn rotate_slope(&mut self, h1: &[C64]) {
        let (n, g, e) = (self.n, self.g, self.e);
        let (ee, eg) = (e * e, e * g);
        let [one, two] = &mut self.tmp;
        let (hee, rest) = self.h1.split_at_mut(ee);
        let (heg, rest) = rest.split_at_mut(eg);
        let (hge, hgg) = rest.split_at_mut(eg);
        for (out, (rows, ru, cols, cu)) in [hee, heg, hge, hgg].into_iter().zip([
            ((g, e), &self.ue_h, (g, e), &self.ue),
            ((g, e), &self.ue_h, (0, g), &self.ug),
            ((0, g), &self.ug_h, (g, e), &self.ue),
            ((0, g), &self.ug_h, (0, g), &self.ug),
        ]) {
            block(h1, n, rows, cols, one);
            fused_product(ru, one, (rows.1, rows.1, cols.1), two);
            fused_product(two, cu, (rows.1, cols.1, cols.1), out);
        }
    }

    /// Onto side `a`: i(H̃₁†K̃_a − K̃_a H̃₁) by blocks (module documentation, # The solve), K̃ with no ground-ground
    /// block -- ee: (H₁_ge)†K̃_ge + H₁_ee†K̃_ee − K̃_eg H₁_ge − K̃_ee H₁_ee, eg: H₁_ee†K̃_eg − K̃_eg H₁_gg − K̃_ee H₁_eg,
    /// and of the ground block's trace i(Tr[H₁_eg†K̃_eg] − c.c.) = −2 Im Tr[H₁_eg†K̃_eg].
    fn turned(&mut self, a: usize) {
        let (g, e) = (self.g, self.e);
        let (ee, eg) = (e * e, e * g);
        let (hee, rest) = self.h1.split_at(ee);
        let (heg, rest) = rest.split_at(eg);
        let (hge, hgg) = rest.split_at(eg);
        let (kee, keg) = (&self.k_ee[a * ee..(a + 1) * ee], &self.k_eg[a * eg..(a + 1) * eg]);
        let [s0, s1] = &mut self.sides;
        let (see, seg) = (&mut s0[a * ee..(a + 1) * ee], &mut s1[a * eg..(a + 1) * eg]);
        // ee: Σ_q conj(H₁_ge[q, r]) conj(K̃_eg[c, q]) + Σ_p conj(H₁_ee[p, r]) K̃_ee[p, c] − Σ_q K̃_eg[r, q] H₁_ge[q, c]
        // − Σ_p K̃_ee[r, p] H₁_ee[p, c], times i.
        for r in 0..e {
            for c in 0..e {
                let mut v = C64::ZERO;
                for q in 0..g {
                    v += (hge[q * e + r] * keg[c * g + q]).conj() - keg[r * g + q] * hge[q * e + c];
                }
                for p in 0..e {
                    v += hee[p * e + r].conj() * kee[p * e + c] - kee[r * e + p] * hee[p * e + c];
                }
                see[r * e + c] += C64::I * v;
            }
        }
        // eg: Σ_p conj(H₁_ee[p, r]) K̃_eg[p, c] − Σ_q K̃_eg[r, q] H₁_gg[q, c] − Σ_p K̃_ee[r, p] H₁_eg[p, c], times i.
        for r in 0..e {
            for c in 0..g {
                let mut v = C64::ZERO;
                for p in 0..e {
                    v += hee[p * e + r].conj() * keg[p * g + c] - kee[r * e + p] * heg[p * g + c];
                }
                for q in 0..g {
                    v -= keg[r * g + q] * hgg[q * g + c];
                }
                seg[r * g + c] += C64::I * v;
            }
        }
        let trace: C64 = heg.iter().zip(keg).map(|(h, k)| h.conj() * *k).sum();
        self.means[a] += -2.0 * trace.im / g as f64;
    }

    /// E[𝓛†P] = R of the first `count` sides (module documentation, # The solve): P's blocks and Tr[B P] = 2 Re Σ
    /// (−iC) conj(P_eg) -- R_ee − i(Z₀† − Z₀) of Z₀ = X₀ B, X₀ = 𝒮⁻¹(−iR_eg), on ℳ's factorization, and P_eg = X₀ +
    /// 𝒮⁻¹(P_ee C).
    fn solve(&mut self, count: usize) {
        let (g, e) = (self.g, self.e);
        let (ee, eg, m) = (e * e, e * g, e * e);
        let (t, s, inverse, b, c) = (&self.t, &self.s, &self.inverse, &self.b, &self.c);
        for side in 0..count {
            let x0 = &mut self.x0[side * eg..(side + 1) * eg];
            for (x, r) in x0.iter_mut().zip(&self.sides[1][side * eg..(side + 1) * eg]) {
                *x = C64::new(r.im, -r.re);
            }
            substitute(t, s, inverse, (e, g), x0);
            fused_product(x0, b, (e, g, e), &mut self.z);
            let r_ee = &self.sides[0][side * ee..(side + 1) * ee];
            let mut row = 0;
            for r in 0..e {
                for cc in r..e {
                    let v = r_ee[r * e + cc] - C64::I * (self.z[cc * e + r].conj() - self.z[r * e + cc]);
                    self.rhs[row][side] = v.re;
                    if cc > r {
                        self.rhs[row + 1][side] = v.im;
                        row += 1;
                    }
                    row += 1;
                }
            }
        }
        for row in &mut self.rhs[..m] {
            row[count..].fill(0.0);
        }
        self.lu.solve(&mut self.rhs[..m]);
        let [solved_ee, solved_eg] = &mut self.solved;
        for side in 0..count {
            let p = &mut solved_ee[side * ee..(side + 1) * ee];
            let mut row = 0;
            for i in 0..e {
                p[i * e + i] = C64::real(self.rhs[row][side]);
                row += 1;
                for j in i + 1..e {
                    let z = C64::new(self.rhs[row][side], self.rhs[row + 1][side]);
                    (p[i * e + j], p[j * e + i]) = (z, z.conj());
                    row += 2;
                }
            }
            let x = &mut solved_eg[side * eg..(side + 1) * eg];
            fused_product(p, c, (e, e, g), x);
            substitute(t, s, inverse, (e, g), x);
            for (x, x0) in x.iter_mut().zip(&self.x0[side * eg..(side + 1) * eg]) {
                *x += *x0;
            }
            self.paired[side] = 2.0 * c.iter().zip(x.iter()).map(|(c, x)| (C64::new(c.im, -c.re) * x.conj()).re).sum::<f64>();
        }
    }

    /// 𝓡_Q†R = −(P + μS) of the first `count` sides solved (module documentation, # K_Q), their means r̄: μ = ζ/g − r̄,
    /// ζ = (Tr[B P] − r̄ Tr[B S])/(1 − Tr[B S]/g) -- into K̃'s blocks (`into_k`), or as the next sides.
    fn resolve(&mut self, count: usize, into_k: bool) {
        let (g, e) = (self.g, self.e);
        let (ee, eg, gf) = (e * e, e * g, g as f64);
        for side in 0..count {
            let mean = self.means[side];
            let zeta = (self.paired[side] - mean * self.returned) / (1.0 - self.returned / gf);
            let mu = zeta / gf - mean;
            let (to_ee, to_eg) = if into_k { (&mut self.k_ee, &mut self.k_eg) } else { let [a, b] = &mut self.sides; (a, b) };
            for (k, (p, s)) in to_ee[side * ee..(side + 1) * ee].iter_mut().zip(self.solved[0][side * ee..(side + 1) * ee].iter().zip(&self.s_ee[..ee])) {
                *k = -(*p + *s * mu);
            }
            for (k, (p, s)) in to_eg[side * eg..(side + 1) * eg].iter_mut().zip(self.solved[1][side * eg..(side + 1) * eg].iter().zip(&self.s_eg[..eg])) {
                *k = -(*p + *s * mu);
            }
        }
    }

    /// K_a in the cluster's basis from K̃'s blocks: K_ee = U_e K̃_ee U_e†, K_eg = U_e K̃_eg U_g†, K_ge = K_eg†, no
    /// ground-ground block.
    fn back(&mut self) {
        let (n, g, e) = (self.n, self.g, self.e);
        let (ee, eg) = (e * e, e * g);
        let [one, two] = &mut self.tmp;
        for a in 0..3 {
            let k = &mut self.k[a][..n * n];
            k.fill(C64::ZERO);
            fused_product(&self.ue, &self.k_ee[a * ee..(a + 1) * ee], (e, e, e), one);
            fused_product(one, &self.ue_h, (e, e, e), two);
            for (i, row) in two[..ee].chunks_exact(e).enumerate() {
                k[(g + i) * n + g..(g + i) * n + n].copy_from_slice(row);
            }
            fused_product(&self.ue, &self.k_eg[a * eg..(a + 1) * eg], (e, e, g), one);
            fused_product(one, &self.ug_h, (e, g, g), two);
            for (i, row) in two[..eg].chunks_exact(g).enumerate() {
                for (j, z) in row.iter().enumerate() {
                    k[(g + i) * n + j] = *z;
                    k[j * n + g + i] = z.conj();
                }
            }
        }
    }

    /// The counted complex multiply-adds of one `within` on n states (step.rs): K_a x and Ĝ_b x (n² each), and the
    /// inner products.
    pub(crate) fn forms(n: usize) -> f64 {
        (6 * n * n + 13 * n) as f64
    }
}

#[cfg(test)]
mod tests;
