//! The optical sector's momentum noise (the momentum diffusion "Q"; design: refs/diffusion/DESIGN.md, claude/motion-v2-diffdesign).
//!
//! # The split by the model's own structure
//! The exact first-order correlation of the force at a point is D_x = Re Tr[𝓕 𝓡 X̄], 𝓡 = (−𝓛)⁻¹ on the traceless
//! densities, X̄ = 𝓕[ρ̄] − F̄ρ̄; a trajectory carries its Ehrenfest mean's own fluctuation D_E, and what it drops of a state
//! ψ is within(ψ) = Re Tr[𝓕 𝓡 X_ψ], X_ψ = 𝓕[ψψ†] − ⟨F⟩ψψ† (D_x = D_E + E[within]). The densities split by the model's
//! structure: P, the ground manifold's traceless part (its populations less their mean over every ground state of the
//! model, and its coherences within each cluster) -- the slow optical pumping a folded cluster keeps --, and Q = 1 − P:
//! every entry with an excited index, and the ground block's trace, the identity on the ground states over their number.
//! 𝓡_Q = Q(−Q𝓛Q)⁻¹Q relaxes at about Γ/2 or faster: every Q direction holds an excited index or the excited trace
//! (measured 19.06–19.7/µs on three models; that it holds in general is the design's hypothesis). Its part of a state's
//! correlation, within_Q(ψ) = Re Tr[𝓕 𝓡_Q X_ψ], is local: the trajectory takes it as white noise of that rate, and its
//! mean over the trajectories is Re Tr[𝓕 𝓡_Q E[X_ψ]] by linearity, whatever the state's own fast dynamics. The rest
//! passes through the ground manifold's own state, which the trajectory keeps (the Ehrenfest error of the ground
//! manifold's superpositions, a model limitation the ledger names). A folded cluster has no Q direction a state of it
//! enters: its fold holds that part as its photons' local second moment (photon.rs).
//!
//! # K_Q
//! within_Q(ψ) = Re Tr[K X_ψ] for the Hermitian K = 𝓡_Q†[F] (one per lab axis), F = −G the force of the generator's
//! Hermitian part (the Ehrenfest force the trajectory carries; 𝓕 = ½{F, ·}): within_Q,ab(ψ) = Re⟨K_a ψ|F_b ψ⟩/‖ψ‖² −
//! ⟨K_a⟩⟨F_b⟩. K is defined up to the identity (X_ψ is traceless); it is taken with no ground-ground entries.
//!
//! In the coordinates y_E of the entries with an excited index (a Hermitian y by its excited block's upper triangle and
//! its excited-ground block, `Coordinates`), a traceless Q density is y_E − (τ/N) 1_g, τ = Tr y_E the excited trace and
//! N the model's ground states, and −Q𝓛Q on it is
//!
//! ```text
//!   A′ y = A y − (τ/N) b,    A y = E[𝓛 y],   b = E[𝓛 1_g],
//! ```
//! E the entries with an excited index: the jumps land in ground states and take no part (only H, the generator with its
//! decay, enters), and the ground trace's equation is the trace's, which 𝓛 keeps. A is block-diagonal over the clusters;
//! they are joined by the one direction 1_g alone, a rank-one border -- a scalar Schur complement (Sherman–Morrison).
//! within_Q = Tr[F′ Y] with F′ = F less the mean of its diagonal over the model's ground states φ̄ (the identity is free:
//! F′ pairs with Y's E part alone), so K = −W⁻¹ A′⁻ᵀ W f′ in the coordinates (W the trace form's weights: one on the
//! excited block's diagonal, two on a pair of entries' real and imaginary parts), and
//!
//! ```text
//!   K_c = −W⁻¹ A_c⁻ᵀ (W f_c + μ s_c),   μ = ζ/N − φ̄,   ζ = (Σ_c u_c·W f_c − φ̄ Σ_c u_c·s_c)/(1 − Σ_c u_c·s_c/N),
//! ```
//! u_c = A_c⁻¹ b_c, s the excited trace's coordinates: per cluster one LU of its excited sector (dimension n_e² + 2 n_e n_g
//! in reals), one solve for u and four transposed ones (three axes and s); a cluster without excited states gives its
//! ground states' force to φ̄ and nothing else.

use crate::linalg::{Mat, C64};

/// A Hermitian matrix's coordinates on the entries with an excited index of a cluster of n states, the first g of them
/// ground: per excited place i and place j ≥ i of the excited block, its diagonal (j = i) or the real and imaginary parts
/// of y_ij; then per excited place i and ground place j those of y_ij (y_ji is their conjugate).
struct Coordinates {
    n: usize,
    /// Per coordinate: the entry (i, j) and what of it -- 0 its real diagonal, 1 the real part, 2 the imaginary part.
    entries: Vec<(usize, usize, u8)>,
}

impl Coordinates {
    fn new(n: usize, g: usize) -> Coordinates {
        let mut entries = vec![];
        for i in g..n {
            entries.push((i, i, 0));
            for j in i + 1..n {
                entries.extend([(i, j, 1), (i, j, 2)]);
            }
        }
        for i in g..n {
            for j in 0..g {
                entries.extend([(i, j, 1), (i, j, 2)]);
            }
        }
        Coordinates { n, entries }
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    /// The trace form's weight of a coordinate: Tr[P Y] = Σ_c w_c p_c y_c for Hermitian P, Y on these entries.
    fn weight(&self, c: usize) -> f64 {
        if self.entries[c].2 == 0 { 1.0 } else { 2.0 }
    }
    /// A Hermitian matrix's coordinates (its entries (i, j) as listed).
    fn of(&self, m: &[C64]) -> Vec<f64> {
        self.entries
            .iter()
            .map(|&(i, j, part)| {
                let z = m[i * self.n + j];
                if part == 2 { z.im } else { z.re }
            })
            .collect()
    }
    /// The Hermitian matrix of coordinates (zero off these entries).
    fn matrix(&self, y: &[f64]) -> Vec<C64> {
        let n = self.n;
        let mut m = vec![C64::ZERO; n * n];
        for (&(i, j, part), &v) in self.entries.iter().zip(y) {
            match part {
                0 => m[i * n + i] = C64::real(v),
                1 => {
                    m[i * n + j].re += v;
                    m[j * n + i].re += v;
                }
                _ => {
                    m[i * n + j].im += v;
                    m[j * n + i].im -= v;
                }
            }
        }
        m
    }
}

/// −i(H Y − Y H†) of a Hermitian Y given by its nonzero entries (`entries`: (i, j, y_ij)), into `out` (n × n).
fn coherent(h: &Mat, entries: &[(usize, usize, C64)], out: &mut [C64]) {
    let n = h.rows;
    out.fill(C64::ZERO);
    for &(p, q, y) in entries {
        // (H Y)[x][q] += H[x][p] y_pq; (Y H†)[p][x] += y_pq conj(H[x][q]).
        for x in 0..n {
            out[x * n + q] += C64::new(0.0, -1.0) * h[(x, p)] * y;
            out[p * n + x] += C64::new(0.0, 1.0) * y * h[(x, q)].conj();
        }
    }
}

/// A real square matrix's LU factorization with partial pivoting (row by row), for its solves and its transpose's.
struct Lu {
    m: usize,
    lu: Vec<f64>,
    pivots: Vec<usize>,
}

impl Lu {
    /// None where a pivot is zero or no number.
    fn new(mut a: Vec<f64>, m: usize) -> Option<Lu> {
        let mut pivots = vec![0; m];
        for k in 0..m {
            let p = (k..m).max_by(|&x, &y| a[x * m + k].abs().total_cmp(&a[y * m + k].abs()))?;
            pivots[k] = p;
            if p != k {
                for c in 0..m {
                    a.swap(k * m + c, p * m + c);
                }
            }
            let pivot = a[k * m + k];
            if !(pivot != 0.0 && pivot.is_finite()) {
                return None;
            }
            for r in k + 1..m {
                let f = a[r * m + k] / pivot;
                a[r * m + k] = f;
                if f != 0.0 {
                    for c in k + 1..m {
                        a[r * m + c] -= f * a[k * m + c];
                    }
                }
            }
        }
        Some(Lu { m, lu: a, pivots })
    }
    /// A x = b in place.
    fn solve(&self, b: &mut [f64]) {
        let (m, a) = (self.m, &self.lu);
        for k in 0..m {
            b.swap(k, self.pivots[k]);
        }
        for r in 0..m {
            let s: f64 = (0..r).map(|c| a[r * m + c] * b[c]).sum();
            b[r] -= s;
        }
        for r in (0..m).rev() {
            let s: f64 = (r + 1..m).map(|c| a[r * m + c] * b[c]).sum();
            b[r] = (b[r] - s) / a[r * m + r];
        }
    }
    /// Aᵀ x = b in place: (P L U)ᵀ with P the row exchanges, Uᵀ then Lᵀ then the exchanges backwards.
    fn solve_transposed(&self, b: &mut [f64]) {
        let (m, a) = (self.m, &self.lu);
        for r in 0..m {
            let s: f64 = (0..r).map(|c| a[c * m + r] * b[c]).sum();
            b[r] = (b[r] - s) / a[r * m + r];
        }
        for r in (0..m).rev() {
            let s: f64 = (r + 1..m).map(|c| a[c * m + r] * b[c]).sum();
            b[r] -= s;
        }
        for k in (0..m).rev() {
            b.swap(k, self.pivots[k]);
        }
    }
}

/// One cluster's part of the optical sector at a point (module documentation, # K_Q): from its generator H (with its
/// decay) and the gradient G_a of its Hermitian part (n × n, row by row), its first `ground` states ground.
pub(crate) struct Sector {
    coordinates: Coordinates,
    /// The factorization of A (None for a cluster without excited states).
    lu: Option<Lu>,
    /// W f_a, f the coordinates of F = −G on the excited entries; s the excited trace's.
    forces: [Vec<f64>; 3],
    trace: Vec<f64>,
    /// u·W f_a, u·s (u = A⁻¹ b), and Σ_g F_a,gg over the cluster's ground states.
    pub(crate) paired: [f64; 3],
    pub(crate) returned: f64,
    pub(crate) ground_force: [f64; 3],
    /// The real multiply-adds that made it.
    pub(crate) work: f64,
}

impl Sector {
    /// The cluster's part (module documentation, # K_Q). None where A is singular (no number came out).
    pub(crate) fn new(h: &Mat, gradient: &[Vec<C64>; 3], ground: usize) -> Option<Sector> {
        let n = h.rows;
        let coordinates = Coordinates::new(n, ground);
        let m = coordinates.len();
        let ground_force = std::array::from_fn(|a| -(0..ground).map(|i| gradient[a][i * n + i].re).sum::<f64>());
        if m == 0 {
            return Some(Sector { coordinates, lu: None, forces: Default::default(), trace: vec![], paired: [0.0; 3], returned: 0.0, ground_force, work: 0.0 });
        }
        // A's columns: 𝓛 on each coordinate's Hermitian basis matrix, read on the coordinates.
        let mut a = vec![0.0; m * m];
        let mut z = vec![C64::ZERO; n * n];
        for (c, &(i, j, part)) in coordinates.entries.iter().enumerate() {
            let basis: &[(usize, usize, C64)] = &match part {
                0 => [(i, i, C64::ONE), (i, i, C64::ZERO)],
                1 => [(i, j, C64::ONE), (j, i, C64::ONE)],
                _ => [(i, j, C64::new(0.0, 1.0)), (j, i, C64::new(0.0, -1.0))],
            };
            coherent(h, basis, &mut z);
            for (r, v) in coordinates.of(&z).into_iter().enumerate() {
                a[r * m + c] = v;
            }
        }
        // b = E[𝓛 1_g].
        let ones: Vec<(usize, usize, C64)> = (0..ground).map(|i| (i, i, C64::ONE)).collect();
        coherent(h, &ones, &mut z);
        let mut u = coordinates.of(&z);
        let lu = Lu::new(a, m)?;
        lu.solve(&mut u);
        let weights: Vec<f64> = (0..m).map(|c| coordinates.weight(c)).collect();
        let trace: Vec<f64> = coordinates.entries.iter().map(|e| if e.2 == 0 { 1.0 } else { 0.0 }).collect();
        let forces: [Vec<f64>; 3] = std::array::from_fn(|a| coordinates.of(&gradient[a]).iter().zip(&weights).map(|(g, w)| -g * w).collect());
        let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>();
        let paired = std::array::from_fn(|a| dot(&u, &forces[a]));
        let returned = dot(&u, &trace);
        if !(paired.iter().all(|x| x.is_finite()) && returned.is_finite()) {
            return None;
        }
        // The columns (n² each, 2 entries a column), the factorization (m³/3) and its solve (m²).
        let work = (m * 2 * n * 8 + m * m * m / 3 + m * m) as f64;
        Some(Sector { coordinates, lu: Some(lu), forces, trace, paired, returned, ground_force, work })
    }

    /// A cluster without excited states: its ground states' force Σ_g F_a,gg alone.
    pub(crate) fn folded(ground_force: [f64; 3]) -> Sector {
        Sector { coordinates: Coordinates { n: 0, entries: vec![] }, lu: None, forces: Default::default(), trace: vec![], paired: [0.0; 3], returned: 0.0, ground_force, work: 0.0 }
    }

    /// K_a of this cluster (n × n, row by row, Hermitian; module documentation, # K_Q) given every cluster's part at the
    /// point (this one among them) and the model's ground states `ground`; with the real multiply-adds of its four
    /// transposed solves. None for a cluster without excited states.
    pub(crate) fn operators(&self, all: &[&Sector], ground: usize) -> Option<([Vec<C64>; 3], f64)> {
        let lu = self.lu.as_ref()?;
        let n = ground as f64;
        let returned: f64 = all.iter().map(|s| s.returned).sum();
        let mean: [f64; 3] = std::array::from_fn(|a| all.iter().map(|s| s.ground_force[a]).sum::<f64>() / n);
        let mut s = self.trace.clone();
        lu.solve_transposed(&mut s);
        let k = std::array::from_fn(|a| {
            let zeta = (all.iter().map(|s| s.paired[a]).sum::<f64>() - mean[a] * returned) / (1.0 - returned / n);
            let mu = zeta / n - mean[a];
            let mut z = self.forces[a].clone();
            lu.solve_transposed(&mut z);
            let y: Vec<f64> = z.iter().zip(&s).enumerate().map(|(c, (z, s))| -(z + mu * s) / self.coordinates.weight(c)).collect();
            self.coordinates.matrix(&y)
        });
        let m = self.coordinates.len();
        Some((k, (4 * m * m) as f64))
    }
}

/// The 3 × 3 rate within_Q,ab of a state x (unnormalized; module documentation, # K_Q) from K_a and G_b on it:
/// −Re⟨K_a x|G_b x⟩/‖x‖² + ⟨K_a⟩⟨G_b⟩ (F = −G). The reference form, read by the tests; the step sums the same over its
/// quadrature (step.rs `Within`).
#[cfg(test)]
pub(crate) fn within(k: &[Vec<C64>; 3], g: &[Vec<C64>; 3], x: &[C64]) -> [[f64; 3]; 3] {
    let n = x.len();
    let apply = |m: &[C64]| -> Vec<C64> { (0..n).map(|i| (0..n).map(|j| m[i * n + j] * x[j]).sum()).collect() };
    let norm = crate::linalg::norm_sqr(x);
    let kx: [Vec<C64>; 3] = std::array::from_fn(|a| apply(&k[a]));
    let gx: [Vec<C64>; 3] = std::array::from_fn(|a| apply(&g[a]));
    let mean = |v: &[C64]| crate::linalg::inner(x, v).re / norm;
    std::array::from_fn(|a| std::array::from_fn(|b| -crate::linalg::inner(&kx[a], &gx[b]).re / norm + mean(&kx[a]) * mean(&gx[b])))
}

#[cfg(test)]
mod tests;
