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
//! H_ee = U_e T_e U_e†, H_gg = U_g T_g U_g† it is a triangular substitution, e g (e + g)/2 complex multiply-adds (Bartels
//! and Stewart; unitary rotations within the blocks keep the entries with an excited index, the traces and the trace
//! form, so the whole solve is made in the rotated blocks and K rotated back). Its solution put into the second leaves a
//! real linear system on the Hermitian P_ee alone, e² unknowns: ℳ(P) = i(T_e†P − P T_e) + i(Z† − Z), Z = 𝒮⁻¹(P H̃_eg)
//! H̃_ge (H̃ the generator in the rotated blocks), built on its e² basis matrices -- a substitution and a product of
//! e² g each -- and factored (`Lu`), its four right-hand sides -- the three axes and 1_e -- at once. The whole sector's
//! factorization, m = e² + 2eg reals and m³/3 multiply-adds, is the eliminated block's: S3's cluster of five ground and
//! six excited states factors 36 reals for 96.

use crate::linalg::{Mat, C64};

/// A real square matrix's LU factorization with partial pivoting, P T = L U, T row by row, in its place: L unit lower,
/// U upper (Higham, Theorem 9.3: backward stable, the factors those of T + ΔT, |ΔT| ≤ γ_m |L||U|). By panels of `PANEL`
/// columns (LAPACK's right-looking getrf): a panel is eliminated with the rows exchanged whole -- the pivots are the
/// unblocked elimination's, every column it takes is up to date --, its rows of U right of it by its unit lower
/// triangle, and the rest of the matrix takes the panel's columns of L times those rows at once: four multiply-adds
/// for every number the trailing block reads and writes, which the elimination row by row does once -- it is held by
/// its memory's bandwidth, not by the arithmetic. The solves go four right-hand sides at once, one vector a row.
struct Lu {
    m: usize,
    lu: Vec<f64>,
    pivots: Vec<usize>,
}

/// The columns a panel of `Lu` eliminates at once: the vectors of four reals the trailing update reads per number.
const PANEL: usize = 4;

impl Lu {
    /// None where a pivot is zero or no number.
    fn new(mut t: Vec<f64>, m: usize) -> Option<Lu> {
        let mut pivots = vec![0; m];
        for k0 in (0..m).step_by(PANEL) {
            let k1 = (k0 + PANEL).min(m);
            // The panel: its columns eliminated below the diagonal, the rows exchanged whole.
            for k in k0..k1 {
                let p = (k..m).max_by(|&x, &y| t[x * m + k].abs().total_cmp(&t[y * m + k].abs()))?;
                pivots[k] = p;
                if p != k {
                    let (above, below) = t.split_at_mut(p * m);
                    above[k * m..(k + 1) * m].swap_with_slice(&mut below[..m]);
                }
                let (above, below) = t.split_at_mut((k + 1) * m);
                let (pivot, upper) = (above[k * m + k], &above[k * m + k + 1..k * m + k1]);
                if !(pivot != 0.0 && pivot.is_finite()) {
                    return None;
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
        Some(Lu { m, lu: t, pivots })
    }
    /// T X = B in place, B of four columns row by row.
    fn solve(&self, b: &mut [[f64; 4]]) {
        let (m, f) = (self.m, &self.lu);
        for (k, &p) in self.pivots.iter().enumerate() {
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

/// A complex matrix of `rows` × `cols`, row by row: a block of the cluster's matrices and the products the solve makes.
#[derive(Clone)]
struct Block {
    rows: usize,
    cols: usize,
    data: Vec<C64>,
}

impl Block {
    fn zeros(rows: usize, cols: usize) -> Block {
        Block { rows, cols, data: vec![C64::ZERO; rows * cols] }
    }
    /// The block of an n × n matrix (row by row) at the rows `r` and the columns `c`.
    fn of(m: &[C64], n: usize, r: std::ops::Range<usize>, c: std::ops::Range<usize>) -> Block {
        let (rows, cols) = (r.len(), c.len());
        Block { rows, cols, data: r.flat_map(|i| c.clone().map(move |j| m[i * n + j])).collect() }
    }
    fn at(&self, i: usize, j: usize) -> C64 {
        self.data[i * self.cols + j]
    }
    /// self · other.
    fn times(&self, other: &Block) -> Block {
        let mut out = Block::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let x = self.at(i, k);
                if x != C64::ZERO {
                    for (o, y) in out.data[i * other.cols..(i + 1) * other.cols].iter_mut().zip(&other.data[k * other.cols..(k + 1) * other.cols]) {
                        *o += x * *y;
                    }
                }
            }
        }
        out
    }
    fn adjoint(&self) -> Block {
        Block { rows: self.cols, cols: self.rows, data: (0..self.cols).flat_map(|j| (0..self.rows).map(move |i| self.at(i, j).conj())).collect() }
    }
    /// A Schur form's unitary factor Q (column by column, as `linalg::schur_form` gives it) as a block.
    fn unitary(q: &[C64], n: usize) -> Block {
        Block { rows: n, cols: n, data: (0..n).flat_map(|i| (0..n).map(move |j| q[j * n + i])).collect() }
    }
}

/// The Hermitian e × e matrices' real coordinates: per place i its diagonal, then per pair i < j the real and the
/// imaginary part of the entry (i, j).
fn hermitian_places(e: usize) -> Vec<(usize, usize, u8)> {
    let mut places = vec![];
    for i in 0..e {
        places.push((i, i, 0));
        for j in i + 1..e {
            places.extend([(i, j, 1), (i, j, 2)]);
        }
    }
    places
}

/// One cluster's part of the optical sector at a point (module documentation, # K_Q, # The solve, # Along the path): from
/// its generator H (with its decay) and the gradient G_a of its Hermitian part (n × n, row by row) there, its first
/// `ground` states ground.
pub(crate) struct Sector {
    /// The elimination and K's blocks (none for a cluster without excited or without ground states).
    solver: Option<Solver>,
    /// The real multiply-adds that made it.
    pub(crate) work: f64,
}

/// The excited-ground block's elimination at a point (module documentation, # The solve), in the blocks' Schur bases
/// H_ee = U_e T_e U_e†, H_gg = U_g T_g U_g†: the rotated couplings H̃_eg and H̃_ge, the Hermitian e × e coordinates and
/// ℳ's factorization; S's blocks with Tr[B S]; and K's blocks per axis, all rotated.
struct Solver {
    n: usize,
    g: usize,
    e: usize,
    ue: Block,
    ug: Block,
    te: Block,
    tg: Block,
    coupling: Block,
    back: Block,
    places: Vec<(usize, usize, u8)>,
    lu: Lu,
    s: (Block, Block),
    returned: f64,
    k: [(Block, Block); 3],
}

impl Solver {
    /// 𝒮⁻¹: T_e†X − X T_g = C by substitution, rows and columns upwards.
    fn sylvester(&self, c: &Block) -> Block {
        let (e, g) = (self.e, self.g);
        let mut x = Block::zeros(e, g);
        for i in 0..e {
            for j in 0..g {
                let mut v = c.at(i, j);
                for k in 0..i {
                    v -= self.te.at(k, i).conj() * x.at(k, j);
                }
                for k in 0..j {
                    v += x.at(i, k) * self.tg.at(k, j);
                }
                x.data[i * g + j] = v / (self.te.at(i, i).conj() - self.tg.at(j, j));
            }
        }
        x
    }
    /// i(Z† − Z) of Z = X H̃_ge, X an excited-ground block.
    fn coupled(&self, x: &Block) -> Block {
        let (z, e) = (x.times(&self.back), self.e);
        let mut out = Block::zeros(e, e);
        for i in 0..e {
            for j in 0..e {
                out.data[i * e + j] = C64::new(0.0, 1.0) * (z.at(j, i).conj() - z.at(i, j));
            }
        }
        out
    }
    /// E[𝓛†P] = R for at most four sides, each by its rotated blocks (R_ee, R_eg): P's blocks and Tr[B P] = 2 Re Σ
    /// (−iH̃_eg) P_eg* -- R_ee − i(Z₀† − Z₀) of Z₀ = X₀ H̃_ge, X₀ = 𝒮⁻¹(−iR_eg), on ℳ's factorization, and P_eg = X₀ +
    /// 𝒮⁻¹(P_ee H̃_eg).
    fn solve(&self, sides: &[(&Block, &Block)]) -> Vec<(Block, Block, f64)> {
        let (e, places) = (self.e, &self.places);
        let starts: Vec<Block> = sides.iter().map(|(_, eg)| self.sylvester(&Block { rows: eg.rows, cols: eg.cols, data: eg.data.iter().map(|z| C64::new(0.0, -1.0) * *z).collect() })).collect();
        let mut rhs = vec![[0.0; 4]; places.len()];
        for (r, ((ee, _), x0)) in sides.iter().zip(&starts).enumerate() {
            let z0 = self.coupled(x0);
            for (row, &(i, j, kind)) in places.iter().enumerate() {
                let v = ee.at(i, j) - z0.at(i, j);
                rhs[row][r] = if kind == 2 { v.im } else { v.re };
            }
        }
        self.lu.solve(&mut rhs);
        starts
            .into_iter()
            .enumerate()
            .map(|(r, x0)| {
                let mut p = Block::zeros(e, e);
                for (row, &(i, j, kind)) in places.iter().enumerate() {
                    let v = rhs[row][r];
                    match kind {
                        0 => p.data[i * e + i] = C64::real(v),
                        1 => {
                            p.data[i * e + j].re += v;
                            p.data[j * e + i].re += v;
                        }
                        _ => {
                            p.data[i * e + j].im += v;
                            p.data[j * e + i].im -= v;
                        }
                    }
                }
                let mut x = self.sylvester(&p.times(&self.coupling));
                for (x, x0) in x.data.iter_mut().zip(&x0.data) {
                    *x += *x0;
                }
                let paired = 2.0 * self.coupling.data.iter().zip(&x.data).map(|(c, x)| (C64::new(0.0, -1.0) * *c * x.conj()).re).sum::<f64>();
                (p, x, paired)
            })
            .collect()
    }
    /// 𝓡_Q†R = −(P + μS) of at most four Hermitian R (module documentation, # K_Q), each by its rotated blocks and the
    /// mean r̄ of its ground diagonal: μ = ζ/g − r̄, ζ = (Tr[B P] − r̄ Tr[B S])/(1 − Tr[B S]/g).
    fn resolve(&self, sides: &[(&Block, &Block, f64)]) -> Vec<(Block, Block)> {
        let count = self.g as f64;
        let solved = self.solve(&sides.iter().map(|(ee, eg, _)| (*ee, *eg)).collect::<Vec<_>>());
        solved
            .into_iter()
            .zip(sides)
            .map(|((p_ee, p_eg, paired), (_, _, mean))| {
                let zeta = (paired - mean * self.returned) / (1.0 - self.returned / count);
                let mu = zeta / count - mean;
                let mix = |p: &Block, s: &Block| Block { rows: p.rows, cols: p.cols, data: p.data.iter().zip(&s.data).map(|(p, s)| -(*p + *s * mu)).collect() };
                (mix(&p_ee, &self.s.0), mix(&p_eg, &self.s.1))
            })
            .collect()
    }
    /// U†MU of an n × n matrix (row by row), U = U_g ⊕ U_e.
    fn rotated(&self, m: &[C64]) -> Block {
        let (n, g) = (self.n, self.g);
        let mut u = Block::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                u.data[i * n + j] = if i < g && j < g { self.ug.at(i, j) } else if i >= g && j >= g { self.ue.at(i - g, j - g) } else { C64::ZERO };
            }
        }
        u.adjoint().times(&Block { rows: n, cols: n, data: m.to_vec() }).times(&u)
    }
    /// The n × n rotated matrix of K's blocks (Hermitian, no ground-ground block).
    fn whole(&self, k: &(Block, Block)) -> Block {
        let (n, g, e) = (self.n, self.g, self.e);
        let mut out = Block::zeros(n, n);
        for i in 0..e {
            for j in 0..e {
                out.data[(g + i) * n + g + j] = k.0.at(i, j);
            }
            for j in 0..g {
                out.data[(g + i) * n + j] = k.1.at(i, j);
                out.data[j * n + g + i] = k.1.at(i, j).conj();
            }
        }
        out
    }
    /// A rotated n × n matrix's excited blocks (ee, eg) and the mean of its ground diagonal.
    fn sides(&self, m: &Block) -> (Block, Block, f64) {
        let (n, g) = (self.n, self.g);
        let mean = (0..g).map(|i| m.at(i, i).re).sum::<f64>() / g as f64;
        (Block::of(&m.data, n, g..n, g..n), Block::of(&m.data, n, g..n, 0..g), mean)
    }
}

impl Sector {
    /// The cluster's part (module documentation, # The solve): the Schur forms, ℳ's factorization, S, and K at the point.
    /// None where a Schur form does not converge or the excited block's system is singular (no number came out).
    pub(crate) fn new(h: &Mat, gradient: &[Vec<C64>; 3], ground: usize) -> Option<Sector> {
        let n = h.rows;
        let (g, e) = (ground, n - ground);
        if e == 0 || g == 0 {
            return Some(Sector { solver: None, work: 0.0 });
        }
        let (ge, ee) = (0..g, g..n);
        let block = |m: &[C64], r: std::ops::Range<usize>, c: std::ops::Range<usize>| Block::of(m, n, r, c);
        let schur = |b: &Block| -> Option<(Block, Block, f64)> {
            let (t, q, work) = crate::linalg::schur_form(&Mat { rows: b.rows, cols: b.cols, data: b.data.clone() })?;
            Some((Block { rows: b.rows, cols: b.cols, data: t }, Block::unitary(&q, b.rows), work))
        };
        let (te, ue, we) = schur(&block(&h.data, ee.clone(), ee.clone()))?;
        let (tg, ug, wg) = schur(&block(&h.data, ge.clone(), ge.clone()))?;
        let (ue_, ug_) = (ue.adjoint(), ug.adjoint());
        let coupling = ue_.times(&block(&h.data, ee.clone(), ge.clone())).times(&ug);
        let back = ug_.times(&block(&h.data, ge.clone(), ee.clone())).times(&ue);
        let places = hermitian_places(e);
        let m = places.len();
        let mut solver = Solver { n, g, e, ue, ug, te, tg, coupling, back, places, lu: Lu { m: 0, lu: vec![], pivots: vec![] }, s: (Block::zeros(e, e), Block::zeros(e, g)), returned: 0.0, k: std::array::from_fn(|_| (Block::zeros(e, e), Block::zeros(e, g))) };
        // ℳ on the Hermitian basis matrices, read on the same coordinates (row by row: output place, input place).
        let te_ = solver.te.adjoint();
        let mut system = vec![0.0; m * m];
        for (column, &(i, j, part)) in solver.places.iter().enumerate() {
            let mut p = Block::zeros(e, e);
            match part {
                0 => p.data[i * e + i] = C64::ONE,
                1 => (p.data[i * e + j], p.data[j * e + i]) = (C64::ONE, C64::ONE),
                _ => (p.data[i * e + j], p.data[j * e + i]) = (C64::new(0.0, 1.0), C64::new(0.0, -1.0)),
            }
            let mut out = solver.coupled(&solver.sylvester(&p.times(&solver.coupling)));
            let (tp, pt) = (te_.times(&p), p.times(&solver.te));
            for (o, (a, b)) in out.data.iter_mut().zip(tp.data.iter().zip(&pt.data)) {
                *o += C64::new(0.0, 1.0) * (*a - *b);
            }
            for (row, &(r, c, kind)) in solver.places.iter().enumerate() {
                let z = out.at(r, c);
                system[row * m + column] = if kind == 2 { z.im } else { z.re };
            }
        }
        solver.lu = Lu::new(system, m)?;
        // S: E[𝓛†S] = 1_e; then K_a = 𝓡_Q†F_a of the three axes' F = −G.
        let mut one = Block::zeros(e, e);
        for i in 0..e {
            one.data[i * e + i] = C64::ONE;
        }
        let (s_ee, s_eg, returned) = solver.solve(&[(&one, &Block::zeros(e, g))]).pop()?;
        (solver.s, solver.returned) = ((s_ee, s_eg), returned);
        let forces: Vec<(Block, Block, f64)> = (0..3).map(|a| solver.sides(&solver.rotated(&gradient[a].iter().map(|z| -*z).collect::<Vec<_>>()))).collect();
        let k = solver.resolve(&forces.iter().map(|(ee, eg, mean)| (ee, eg, *mean)).collect::<Vec<_>>());
        if !(returned.is_finite() && k.iter().all(|(ee, eg)| ee.data.iter().chain(&eg.data).all(|z| z.is_finite()))) {
            return None;
        }
        solver.k = [k[0].clone(), k[1].clone(), k[2].clone()];
        // Complex multiply-adds (four reals each): the Schur forms, the couplings' rotation, ℳ's basis solves (a Sylvester
        // substitution and the product with H̃_ge each) and four solves (`Sector::solves`), the forces' rotations; and ℳ's
        // factorization in reals.
        let (ef, gf, mf, nf) = (e as f64, g as f64, m as f64, n as f64);
        let (syl, product) = (ef * gf * (ef + gf) / 2.0, ef * ef * gf);
        let complex = we + wg + 2.0 * (ef * ef * gf + ef * gf * gf) + mf * (syl + product + 4.0 * ef) + 3.0 * 2.0 * nf * nf * nf;
        let work = crate::decide::COMPLEX * complex + mf * mf * mf / 3.0 + Sector::solves(e, g, 4);
        Some(Sector { solver: Some(solver), work })
    }

    /// The real multiply-adds of `count` sides' solves (`Solver::resolve`): their Sylvester substitutions and products
    /// (two each), ℳ's solve, and their μ.
    fn solves(e: usize, g: usize, count: usize) -> f64 {
        let (ef, gf, mf) = (e as f64, g as f64, (e * e) as f64);
        count as f64 * (crate::decide::COMPLEX * (ef * gf * (ef + gf) + 2.0 * ef * ef * gf + 2.0 * ef * gf) + 2.0 * mf * mf)
    }

    /// K_a (n × n, row by row, Hermitian; module documentation, # K_Q), and with the generator's and the gradient's
    /// slopes along the path at the point (`slope`: dH/ds, dG_a/ds) its first order along the path (module
    /// documentation, # Along the path): K + 𝓡_Q†𝓡_Q†(dF/ds + (d𝓛/ds)†K), with the real multiply-adds that took. None for
    /// a cluster without excited or without ground states (no trajectory lies in the latter: it starts in a ground state,
    /// and every jump lands in one).
    pub(crate) fn operators(&self, slope: Option<(&Mat, &[Vec<C64>; 3])>) -> Option<([Vec<C64>; 3], f64)> {
        let solver = self.solver.as_ref()?;
        let (n, e, g) = (solver.n, solver.e, solver.g);
        let mut k = solver.k.clone();
        let mut work = 0.0;
        if let Some((h1, g1)) = slope {
            // R_a = dF_a/ds + i(H₁†K_a − K_a H₁), rotated; 𝓡_Q† of it twice.
            let h1 = solver.rotated(&h1.data);
            let h1_ = h1.adjoint();
            let sides: Vec<(Block, Block, f64)> = (0..3)
                .map(|a| {
                    let (whole, mut r) = (solver.whole(&k[a]), solver.rotated(&g1[a].iter().map(|z| -*z).collect::<Vec<_>>()));
                    let (left, right) = (h1_.times(&whole), whole.times(&h1));
                    for (r, (l, q)) in r.data.iter_mut().zip(left.data.iter().zip(&right.data)) {
                        *r += C64::new(0.0, 1.0) * (*l - *q);
                    }
                    solver.sides(&r)
                })
                .collect();
            let inner = solver.resolve(&sides.iter().map(|(ee, eg, mean)| (ee, eg, *mean)).collect::<Vec<_>>());
            let outer = solver.resolve(&inner.iter().map(|(ee, eg)| (ee, eg, 0.0)).collect::<Vec<_>>());
            for (k, (ee, eg)) in k.iter_mut().zip(outer) {
                for (z, d) in k.0.data.iter_mut().zip(&ee.data) {
                    *z += *d;
                }
                for (z, d) in k.1.data.iter_mut().zip(&eg.data) {
                    *z += *d;
                }
            }
            // The rotations (H₁ and the three dG/ds), the three R's products, and the two resolves of three sides.
            let nf = n as f64;
            work = crate::decide::COMPLEX * (4.0 * 2.0 * nf * nf * nf + 3.0 * 2.0 * nf * nf * nf) + 2.0 * Sector::solves(e, g, 3);
        }
        // Rotated back: K_ee = U_e K̃_ee U_e†, K_eg = U_e K̃_eg U_g†.
        let (ue_, ug_) = (solver.ue.adjoint(), solver.ug.adjoint());
        let out = std::array::from_fn(|a| {
            let (k_ee, k_eg) = (solver.ue.times(&k[a].0).times(&ue_), solver.ue.times(&k[a].1).times(&ug_));
            let mut out = vec![C64::ZERO; n * n];
            for i in 0..e {
                for j in 0..e {
                    out[(g + i) * n + g + j] = k_ee.at(i, j);
                }
                for j in 0..g {
                    out[(g + i) * n + j] = k_eg.at(i, j);
                    out[j * n + g + i] = k_eg.at(i, j).conj();
                }
            }
            out
        });
        let ef = e as f64;
        Some((out, work + crate::decide::COMPLEX * 3.0 * (2.0 * ef * ef * ef + ef * ef * g as f64 + ef * g as f64 * g as f64)))
    }
}

#[cfg(test)]
mod tests;
