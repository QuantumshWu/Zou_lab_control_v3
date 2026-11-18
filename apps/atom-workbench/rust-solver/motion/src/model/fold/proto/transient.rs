//! # The transient (landport.status.md; pland.status # 第三轮, 第四轮（续）; transient.status, transient2.status)
//! A jump's bare state f lands on the fold's slow span along its fast one (`Workspace::taken`): f = Φa + F̂β, Φ = [1; T]
//! the slow span, F̂ = [−T_L†; 1] the fast one, a = N⁻¹f, β = −Ta. On B = [Φ F̂] the generator is block diagonal,
//! H B = B diag(H_s, K_f): H_s the series' Bloch generator (the engine's own: what the step propagates), K_f = K − V T_L†
//! the fast part's own (V the light's source, Q ← P). The exact state is Ψ(τ) = Φe^{−iH_sτ}a + F̂e^{−iK_fτ}β; the engine
//! follows the slow part. Everything the fast part does over its ~1/Γ -- its time, its photons and where they land, its
//! momentum -- is of one object, the landing's modes (`Transient`): the slow r_j = Φq_j (H_s q_j = λ_j q_j) and the fast
//! r_k = F̂q_k (K_f q_k = λ_k q_k), the switched state's coefficients c (f = Σ c_j r_j) and their Gram G_ij = r_i†r_j. Its
//! integral X_tr = ∫(ΨΨ† − Φe^{−iH_sτ}aa†e^{iH_s†τ}Φ†)dτ = Σ_{(i,j) not both slow} c_j c_i* r_j r_i†/(i(λ_j − λ_i*)).
//!
//! The rows, as the fold holds them (pland.status # 第三轮 item 1):
//! - the near blocks' kept rows in their Schur bases, their lanes summed (T = Σ_l Z_l, T_L likewise; V = Σ_l Ṽ_l), K their
//!   Schur form with the step's diagonal κ, each block in its own frame (its offset's: the node space, where the generator is
//!   static), at no Doppler shift (the atom's own frame at rest; a moving atom's lanes turn its slow amplitude against the
//!   fast one at w_l, a relative kv/Γ of the transient: transient.status, # Assumptions);
//! - a far block's rows at its own order, the order its response is held at (proto.rs # The far coordinates: R₀ at its
//!   centre): its fast part β_f = −R₀Ba keeps its direction β̂ = β_f/‖β_f‖ -- one row, K = z₀ − β̂†R₀⁻¹β̂, V = β̂†B, T and
//!   T_L the rows β̂†R₀B, β̂†R₀†B --, exact for a one-row block (a two-level atom off its line), its block's turning within
//!   the decay left out (the far block's own centring order). R₀ is held as the step holds it.
//! The copies' rows (ground states at another offset, no decay of their own: no transient) are folded into P as the series
//! folds them: they are in the slow modes' Gram (the forward's whole metric 1 + M) and in no fast row.
//!
//! Two nodes of one physical state at two offsets (a Floquet copy) are two rows: the quadratic forms on the modes are the
//! node space's (dragchk.status 第二项 (B1), Q⁰): their cross parts turn at the offsets' difference and are taken averaged,
//! as the representation takes them (FOLDAPI §2: none of the scenes keeps such a pair as a tone).
//!
//! # The clock (pland.status # 第四轮（续）; dragchk.status 第二项)
//! The jump clock is the exact survival N(τ) = ‖Ψ(τ)‖² from the bare state, the slow part's and the fast part's: the jump
//! is where N reaches the trajectory's threshold θ. The step follows the slow part alone, its norm N_s the engine's ‖ψ‖²;
//! so the landing solves N(τ*) = θ on its modes and hands the step the threshold θ′ = N_s(τ*) for this cycle -- the step
//! then jumps at τ* (to its own accuracy: the motion's change of the transient is of the first order within it), with no
//! dead time and no norm of its own (`Workspace::clock`). θ′ = θ to the rounding where τ* lies past the transient.
//! The photon at τ* is the whole state's: L_qΨ(τ*), the slow part's the step's own (its state at the jump), the fast
//! part's owed to that jump by the landing (`Clock::owed`: its amplitudes at τ*, every block at its offset's phase) -- the
//! transient's photons land where its rows ring, each at its time (dragchk.status.md section 5: drawn on the slow
//! pattern, tilt0's at-once photons put its F = 2 populations off). Where the slow part is dark until τ* (θ′ its whole
//! norm), that jump is at once (step.rs `jump`). The jump operators enter through the bare state alone: f = P_set L_q(t*)ψ,
//! every block's at its exact phase (dragchk.status 第二项).
//!
//! # The kick (pland.status # 第三轮, # 第四轮（续） §1)
//! The cycle's momentum conditional on its length τ is the photon's at the jump with the tangent from the landing, E[R | τ]
//! = Im⟨Ψ(τ)|D|∂Ψ(τ)⟩/p(τ); on the modes g p = −Re Σ_jk d_j H′_jk c_k E_jk(τ) (`Transient::conditional`), H′ = ℓ†∂H_h r per
//! lab axis (`Transient::mode_forces`; ∂H_h the bare gradient from the step's tangent, a far block's R₀ held as the step
//! holds it). The step gives the slow part as it goes (the continuous force over τ, the photon's mean at the jump): g_s,
//! the same sums with the slow modes' d and p and the long-time E -- F_cτ + b̄ + K_∞ for one slow mode. The landing books
//! the rest at once, one number per axis: K_∞ + δg(τ*), δg = g − g_s at the τ* the clock knows, K_∞ = Im(ψ₀†∂ψ₀)/‖ψ₀‖² the
//! landed slow state's own gradient (∂a = κ_sf − N⁻¹T_L†(∂T a), κ_sf = N⁻¹∂T_L†β by the left Riccati equation, its
//! generator N H_s N⁻¹). The jump that takes the owed photon draws its class by the whole state's rates, so its kick's
//! mean over the classes is moved to the slow photon's, g_s's (run.rs `Middle::transient`); at once the landing books
//! g(τ*) whole.
use super::*;
use crate::linalg::Eigen;

/// A far block the step holds (module documentation): its index, its source B at the middle's light (m × n), its R₀
/// (m × m), its responses T_f = R₀B and T_L,f = R₀†B (m × n: the slow modes' whole); and of the landing, its fast part's
/// direction β̂ = β_f/‖β_f‖ and size ‖β_f‖, its row's K and the row's place among the fast rows -- none where the fast
/// part is zero.
struct FarRow {
    x: usize,
    b: Mat,
    r0: Mat,
    tf: Mat,
    tlf: Mat,
    hat: Vec<C64>,
    size: f64,
    k: C64,
    at: Option<usize>,
}

/// The landing's transient on its modes (module documentation): the P places n, the fast rows -- the near blocks' kept rows
/// (per block its index, first row and rows) then one per far block whose fast part is not zero (`FarRow`) --; T, T_L and V on the
/// rows (rows × n); the two eigendecompositions (H_s's, K_f's); the eigenvalues (the slow n first), the coefficients c of
/// the switched state, the Gram G of the modes; the slow part's P amplitudes a; the count of its making (complex
/// multiply-adds).
pub(crate) struct Transient {
    n: usize,
    blocks: Vec<(usize, usize, usize)>,
    far: Vec<FarRow>,
    t: Mat,
    tl: Mat,
    slow: Eigen,
    quick: Eigen,
    values: Vec<C64>,
    c: Vec<C64>,
    gram: Mat,
    a: Vec<C64>,
    pub(crate) work: f64,
}

/// A far block's source B on its rows at a light (its sources' components' coefficients `e`): B_ap = Σ e_ck U_ck(row a,
/// ground p) over its vectors (c, k) and the places p that source c.
pub(super) fn far_source(fold: &Fold, fb: &Far, n: usize, e: &dyn Fn(usize, usize) -> C64) -> Mat {
    let b = &fold.blocks[fb.o];
    let rows: Vec<usize> = b.rows.iter().map(|r| if let Row::Excited(x) = *r { x } else { unreachable!("a far block's rows are excited") }).collect();
    let mut src = Mat::zeros(fb.m, n);
    for &(c, k) in &fb.vectors {
        let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
        for (a, &row) in rows.iter().enumerate() {
            for &p in &places {
                src[(a, p)] += e(c, k) * fold.components[c].basis[k][(row, fold.ground[p])];
            }
        }
    }
    src
}

/// x†y.
fn dot(x: &[C64], y: &[C64]) -> C64 {
    x.iter().zip(y).map(|(a, b)| a.conj() * *b).sum()
}

/// β̂†M (n) of a block's m × n matrix M.
fn lift(m: &Mat, hat: &[C64]) -> Vec<C64> {
    (0..m.cols).map(|j| (0..hat.len()).map(|r| hat[r].conj() * m[(r, j)]).sum()).collect()
}

/// M placed into `out` at (r0, c0).
fn put(out: &mut Mat, m: &Mat, r0: usize, c0: usize) {
    for x in 0..m.rows {
        for y in 0..m.cols {
            out[(r0 + x, c0 + y)] = m[(x, y)];
        }
    }
}

impl Workspace {
    /// The transient of a landing whose slow part's P amplitudes are `a` (a = N⁻¹f, `Workspace::taken`), at the last
    /// forward (its left response made): its modes (module documentation; `Transient`). None where an eigendecomposition
    /// does not exist (its QR iteration not converging, its eigenvectors singular): the fold's one exit, as a landing that
    /// does not exist.
    pub(crate) fn transient(&self, fold: &Fold, cache: &Cache, a: &[C64]) -> Option<Transient> {
        let n = self.n;
        let mut work = 0.0;
        let mut blocks = vec![];
        let mut at = 0;
        for (x, nb) in self.near.iter().enumerate() {
            if nb.idle() || nb.s == 0 {
                continue;
            }
            blocks.push((x, at, nb.s));
            at += nb.s;
        }
        let s = at;
        // The far blocks: per far block held, its source at the middle's light, T_f = R₀B, T_L,f = R₀†B, β_f = −T_f a and
        // w_f = −Ba; a row where β_f is not zero.
        let mut far = vec![];
        let mut rows = s;
        for (x, fb) in self.far.iter().enumerate().filter(|(_, fb)| !fb.omitted) {
            let basis = cache.far_at(x);
            let b = far_source(fold, fb, n, &|c, k| self.mid.e[c][k]);
            let r0 = basis.resolvent.clone();
            let (tf, tlf) = (r0.product(&b), r0.adjoint().product(&b));
            let ba = b.apply(a);
            let beta: Vec<C64> = tf.apply(a).into_iter().map(|z| -z).collect();
            let size = crate::linalg::norm_sqr(&beta).sqrt();
            work += (2 * fb.m * fb.m * n + 3 * fb.m * n) as f64;
            // K = z₀ − β̂†R₀⁻¹β̂, R₀⁻¹β_f = w_f = −Ba.
            let (k, at) = if size > 0.0 { (C64::real(basis.z0) + dot(&beta, &ba) / (size * size), Some(rows)) } else { (C64::ZERO, None) };
            rows += usize::from(at.is_some());
            let hat = beta.iter().map(|z| if size > 0.0 { *z / size } else { C64::ZERO }).collect();
            far.push(FarRow { x, b, r0, tf, tlf, hat, size, k, at });
        }
        let (mut t, mut tl, mut v, mut k) = (Mat::zeros(rows, n), Mat::zeros(rows, n), Mat::zeros(rows, n), Mat::zeros(rows, rows));
        for &(x, o, sb) in &blocks {
            let nb = &self.near[x];
            let basis = cache.near_at(x);
            let ln = nb.lanes * n;
            for i in 0..sb {
                k[(o + i, o + i)] = nb.kappa[i];
                for l in i + 1..basis.f.ends[i] {
                    k[(o + i, o + l)] = basis.f.t[(i, l)];
                }
                for j in 0..n {
                    v[(o + i, j)] = (0..nb.lanes).map(|l| nb.vt[i * ln + l * n + j]).sum();
                    t[(o + i, j)] = nb.ztot[i * n + j];
                    tl[(o + i, j)] = nb.left.ztot[i * n + j];
                }
            }
        }
        for row in &far {
            let Some(r) = row.at else { continue };
            let (tz, lz, vz) = (lift(&row.tf, &row.hat), lift(&row.tlf, &row.hat), lift(&row.b, &row.hat));
            for j in 0..n {
                (t[(r, j)], tl[(r, j)], v[(r, j)]) = (tz[j], lz[j], vz[j]);
            }
            k[(r, r)] = row.k;
        }
        // K_f = K − V T_L†.
        let mut kf = v.product(&tl.adjoint()).scaled(-C64::ONE);
        kf.add_scaled(C64::ONE, &k);
        let hs = Mat { rows: n, cols: n, data: self.h.clone() };
        let (mut slow, mut quick) = (Eigen::default(), Eigen::default());
        if !(slow.decompose(&hs) && (rows == 0 || quick.decompose(&kf))) {
            return None;
        }
        work += slow.work + if rows > 0 { quick.work } else { 0.0 } + (rows * rows * n) as f64;
        // The coefficients: c_s = Q_s⁻¹a, c_f = Q_f⁻¹β, β = −Ta on the near rows and ‖β_f‖ on a far row (along β̂).
        let mut values = slow.values.clone();
        let mut c = slow.inverse.apply(a);
        if rows > 0 {
            values.extend(quick.values.iter().copied());
            let ta = t.apply(a);
            let beta: Vec<C64> = (0..s).map(|r| -ta[r]).chain(far.iter().filter(|row| row.at.is_some()).map(|row| C64::real(row.size))).collect();
            c.extend(quick.inverse.apply(&beta));
        }
        // The Gram: Q_s†(1 + M)Q_s, Q_s†(T − T_L)†Q_f, Q_f†(1 + T_L T_L†)Q_f.
        let d = n + rows;
        let mut metric = self.step.metric.clone();
        for p in 0..n {
            metric[(p, p)] += C64::ONE;
        }
        let mut gram = Mat::zeros(d, d);
        put(&mut gram, &slow.vectors.adjoint_product(&metric.product(&slow.vectors)), 0, 0);
        if rows > 0 {
            let mut diff = t.clone();
            diff.add_scaled(-C64::ONE, &tl);
            let sf = slow.vectors.adjoint_product(&diff.adjoint().product(&quick.vectors));
            let mut ff = tl.product(&tl.adjoint());
            for r in 0..rows {
                ff[(r, r)] += C64::ONE;
            }
            put(&mut gram, &sf, 0, n);
            put(&mut gram, &sf.adjoint(), n, 0);
            put(&mut gram, &quick.vectors.adjoint_product(&ff.product(&quick.vectors)), n, n);
        }
        work += (2 * n * n * n + 2 * n * rows * (n + rows) + 2 * rows * rows * (n + rows)) as f64;
        Some(Transient { n, blocks, far, t, tl, slow, quick, values, c, gram, a: a.to_vec(), work })
    }
}

/// The landing's clock (module documentation, # The clock, # The kick): the threshold θ′ the slow part's norm meets where
/// the exact survival meets the trajectory's θ at τ* (θ itself where it never does); the photon the fast part owes that
/// jump -- its amplitudes u_q on the ground states at the jump's time (rows (b, q) at 3b + q, every block at its offset's
/// phase, `Fold::prototype_photons`' layout) --, none where it is within the rounding of the slow photon's there; whether
/// that jump is at once (the slow part dark until τ*: θ′ its whole norm); the kick the landing books (1/µm per lab axis);
/// and its count (complex multiply-adds).
#[derive(Clone, Debug)]
pub(crate) struct Clock {
    pub(crate) threshold: f64,
    pub(crate) owed: Option<Vec<C64>>,
    pub(crate) once: bool,
    pub(crate) kick: [f64; 3],
    pub(crate) work: f64,
}

/// The gradient of the generator's Hermitian part on the transient's rows per lab axis: the bare ∂H on P ⊕ near (n + S
/// square, P first: Ĥ_PP's pairs and drives, the explicit states' dressing, the source and the emission on the kept rows,
/// ∂K̃ on them -- the step's tangent's, `Fold::prototype_tangent`), and per far block its source's ∂B (m × n; its rows'
/// own ∂K none: R₀ is held as the step holds it).
struct ForceParts {
    near: [Mat; 3],
    far: Vec<[Mat; 3]>,
}

impl Transient {
    /// The modes' count.
    fn size(&self) -> usize {
        self.values.len()
    }

    /// At τ from the landing: the exact survival N(τ) = Σ_ij c_i*c_j e^{i(λ_i*−λ_j)τ} G_ij, the slow pairs' N_s(τ) (the
    /// engine's ‖ψ‖²), and their photon rates p = −N′, p_s = −N_s′ (r_i†D r_j = i(λ_j − λ_i*)G_ij on the modes).
    fn survival(&self, tau: f64) -> [f64; 4] {
        let (n, d) = (self.n, self.size());
        let v: Vec<C64> = self.values.iter().zip(&self.c).map(|(l, c)| *c * (C64::new(0.0, -tau) * *l).exp()).collect();
        let mut out = [0.0; 4];
        for i in 0..d {
            for j in 0..d {
                let z = v[i].conj() * self.gram[(i, j)] * v[j];
                let r = (z * C64::I * (self.values[j] - self.values[i].conj())).re;
                out[0] += z.re;
                out[2] += r;
                if i < n && j < n {
                    out[1] += z.re;
                    out[3] += r;
                }
            }
        }
        out
    }

    /// τ* where the exact survival meets θ (N(0) = 1 ≥ θ, N falling): the bracket doubled from the fastest decay's time
    /// until N is below θ -- none where every decaying mode has gone (e^{−700} of itself) with N still at θ or above --,
    /// then Newton on ln N within it, halved where a step leaves it, to the rounding of τ or of ln N. With the survival's
    /// evaluations.
    fn crossing(&self, theta: f64) -> (Option<f64>, usize) {
        let decays = self.values.iter().map(|l| -2.0 * l.im).filter(|g| *g > 0.0);
        let (fastest, slowest) = decays.fold((0.0_f64, f64::INFINITY), |(a, b), g| (a.max(g), b.min(g)));
        if !(fastest > 0.0) {
            return (None, 0);
        }
        let (mut lo, mut hi, mut evaluations) = (0.0, 1.0 / fastest, 1);
        while self.survival(hi)[0] >= theta {
            if hi > 1400.0 / slowest {
                return (None, evaluations);
            }
            (lo, hi, evaluations) = (hi, 2.0 * hi, evaluations + 1);
        }
        let target = theta.ln();
        let mut tau = 0.5 * (lo + hi);
        for _ in 0..200 {
            let [all, _, rate, _] = self.survival(tau);
            evaluations += 1;
            let f = all.ln() - target;
            if f.abs() <= 4.0 * f64::EPSILON * target.abs().max(1.0) || hi - lo <= 4.0 * f64::EPSILON * hi {
                break;
            }
            if f > 0.0 {
                lo = tau;
            } else {
                hi = tau;
            }
            let step = tau + f * all / rate;
            tau = if rate > 0.0 && step > lo && step < hi { step } else { 0.5 * (lo + hi) };
        }
        (Some(tau), evaluations)
    }

    /// The fast part's coordinates on the rows at τ from the landing, z(τ) = Q_f e^{−iΛ_fτ} c_f.
    fn fast(&self, tau: f64) -> Vec<C64> {
        let n = self.n;
        let turned: Vec<C64> = (n..self.size()).map(|k| self.c[k] * (C64::new(0.0, -tau) * self.values[k]).exp()).collect();
        self.quick.vectors.apply(&turned)
    }

    /// The mean waiting time from the landing, ∫₀^∞ N(τ)dτ = Σ_ij c_i*c_j G_ij/(i(λ_j − λ_i*)), N(τ) = ‖Ψ(τ)‖² the exact
    /// survival on the modes (µs; infinite where a mode does not decay): the exact clock's.
    pub(crate) fn wait(&self) -> f64 {
        let d = self.size();
        let mut sum = C64::ZERO;
        for i in 0..d {
            for j in 0..d {
                sum += self.c[i].conj() * self.c[j] * self.gram[(i, j)] / (C64::I * (self.values[j] - self.values[i].conj()));
            }
        }
        sum.re
    }
}

impl Workspace {
    /// The photon the fast part z (its coordinates on the transient's rows) emits at the time t: u_q = Σ_o e^{−iot} C_q of
    /// its excited amplitude on each block -- a near block's kept rows by its emission on them (C_qV, `NearBasis::emit`),
    /// its explicit states' own rows (`NearBasis::xemitc`) on the fast part's P part −T_L†z, a far block's row along β̂
    /// (C_qβ_f = −Σ_a e_a C_qR₀U_a a, `FarBasis::emitted`) --, laid as `Fold::prototype_photons` lays the photons (rows
    /// (b, q) at 3b + q, 3 GROUND).
    fn fast_photon(&self, fold: &Fold, cache: &Cache, tr: &Transient, z: &[C64], t: f64) -> Vec<C64> {
        let n = self.n;
        let g3 = 3 * GROUND;
        let mut u = vec![C64::ZERO; g3];
        let p = tr.tl.adjoint().apply(z);
        for (x, nb) in self.near.iter().enumerate().filter(|(_, nb)| !nb.idle()) {
            let basis = cache.near_at(x);
            let phase = C64::cis(-fold.blocks[nb.o].value * t);
            if let Some(&(_, o, s)) = tr.blocks.iter().find(|b| b.0 == x) {
                for (r, out) in u.iter_mut().enumerate() {
                    *out += phase * (0..s).map(|i| basis.emit[r * s + i] * z[o + i]).sum::<C64>();
                }
            }
            if !nb.xs.is_empty() {
                for (r, out) in u.iter_mut().enumerate() {
                    *out -= phase * (0..n).map(|j| basis.xemitc[r * n + j] * p[j]).sum::<C64>();
                }
            }
        }
        for row in &tr.far {
            let Some(r) = row.at else { continue };
            let fb = &self.far[row.x];
            let basis = cache.far_at(row.x);
            let v = fb.vectors.len();
            let phase = C64::cis(-fold.blocks[fb.o].value * t) * (-z[r] / row.size);
            for (ia, &(cc, kk)) in fb.vectors.iter().enumerate() {
                let e = phase * self.mid.e[cc][kk];
                for b in 0..GROUND {
                    for q in 0..3 {
                        let from = ((b * v + ia) * 3 + q) * n;
                        u[3 * b + q] += e * (0..n).map(|j| basis.emitted[from + j] * tr.a[j]).sum::<C64>();
                    }
                }
            }
        }
        u
    }

    /// The landing's clock and kick (module documentation, # The clock, # The kick; `Clock`) of a landing whose slow part
    /// is ψ₀ = S a (`psi`, the engine's state; a = N⁻¹f, `Workspace::taken`), against the trajectory's threshold θ, the
    /// landing at the time t; at the last forward, its left response and the step's tangent made. None where the landing's
    /// modes do not exist (`Workspace::transient`).
    pub(crate) fn clock(&self, fold: &Fold, cache: &Cache, psi: &[C64], a: &[C64], theta: f64, t: f64) -> Option<Clock> {
        assert!(self.tangent, "the step's tangent");
        let _p = crate::prof::scope(crate::prof::TR_MODES);
        let tr = self.transient(fold, cache, a)?;
        drop(_p);
        let (n, d) = (self.n, tr.size());
        let _p = crate::prof::scope(crate::prof::TR_CROSSING);
        let (crossed, evaluations) = tr.crossing(theta);
        drop(_p);
        let _p = crate::prof::scope(crate::prof::TR_FORCES);
        let forces = fold.transient_forces(self, &tr);
        drop(_p);
        let _p = crate::prof::scope(crate::prof::TR_MODE_FORCES);
        let hprime = tr.mode_forces(self, &forces);
        drop(_p);
        let mut work = tr.work + (evaluations * 2 * d * d) as f64 + 3.0 * (6 * d * d * d) as f64;
        // K_∞ = Im(ψ₀†∂ψ₀)/‖ψ₀‖², ∂ψ₀ = ∂S a + S∂a, S∂a = S⁻¹(1 + M)∂a, ∂a = κ_sf − N⁻¹T_L†(∂T a), κ_sf on the slow modes
        // Σ_k H′_jk c_k/(λ_j − λ_k) over the fast k.
        let _p = crate::prof::scope(crate::prof::TR_LEFT);
        let left = self.left_tangent(cache, &tr, &forces, a);
        drop(_p);
        let _p = crate::prof::scope(crate::prof::TR_KICK);
        let mut metric = self.step.metric.clone();
        for k in 0..n {
            metric[(k, k)] += C64::ONE;
        }
        let size = crate::linalg::norm_sqr(psi);
        let mut kick = [0.0; 3];
        for (ax, hp) in hprime.iter().enumerate() {
            let kappa: Vec<C64> = (0..n).map(|j| (n..d).map(|k| hp[(j, k)] * tr.c[k] / (tr.values[j] - tr.values[k])).sum()).collect();
            let mut da = tr.slow.vectors.apply(&kappa);
            let mut y = left[ax].clone();
            crate::linalg::solve(&self.landing.lu, &self.landing.piv, n, &mut y, 1);
            da.iter_mut().zip(&y).for_each(|(u, v)| *u -= *v);
            let ds = Mat { rows: n, cols: n, data: self.ds[ax].clone() };
            let sda = self.step.normal.apply(&metric.apply(&da));
            let dpsi: Vec<C64> = ds.apply(a).iter().zip(&sda).map(|(u, v)| *u + *v).collect();
            kick[ax] = dot(psi, &dpsi).im / size;
        }
        work += (3 * (n * (d + 2 * n) + 2 * n * n)) as f64;
        drop(_p);
        let Some(wait) = crossed else { return Some(Clock { threshold: theta, owed: None, once: false, kick, work }) };
        let [_, slow, _, rate] = tr.survival(wait);
        let z = tr.fast(wait);
        let _p = crate::prof::scope(crate::prof::TR_FAST_PHOTON);
        let u = self.fast_photon(fold, cache, &tr, &z, t + wait);
        drop(_p);
        // The fast part's photon where it is beyond the rounding of the slow photon's (its hazard p_s at τ*).
        let floor = crate::units::rounding(d).powi(2) * rate.max(0.0);
        let owed = (crate::linalg::norm_sqr(&u) > floor).then_some(u);
        // The cycle's conditional mean g(τ*) and its slow part's g_s(τ*) (# The kick): the landing books K_∞ + g − g_s; at
        // once (the slow part dark until τ*: no step, no slow photon) g alone.
        let once = slow >= size;
        let _p = crate::prof::scope(crate::prof::TR_CONDITIONAL);
        let (whole, part) = tr.conditional(&hprime, wait);
        drop(_p);
        work += (2 * d * d + 3 * GROUND * d + 9 * d * d) as f64;
        for ax in 0..3 {
            kick[ax] = if once { whole[ax] } else if let Some(part) = part { kick[ax] + whole[ax] - part[ax] } else { kick[ax] };
        }
        Some(Clock { threshold: slow, owed, once, kick, work })
    }

    /// T_L†(∂T a) per lab axis (n each) over every row the metric holds (`Fold::prototype_left`'s N): a near block's kept
    /// rows, its exact copies' and its centred copies' where the metric holds them, a far block's R₀∂B a (R₀ held) against
    /// its R₀†B. After the step's tangent.
    fn left_tangent(&self, cache: &Cache, tr: &Transient, forces: &ForceParts, a: &[C64]) -> [Vec<C64>; 3] {
        let n = self.n;
        let apply = |m: &[C64], rows: usize, v: &[C64]| -> Vec<C64> { (0..rows).map(|r| m[r * n..(r + 1) * n].iter().zip(v).map(|(x, y)| *x * *y).sum()).collect() };
        let back = |m: &[C64], y: &[C64], out: &mut [C64]| {
            for (r, yr) in y.iter().enumerate() {
                for (j, o) in out.iter_mut().enumerate() {
                    *o += m[r * n + j].conj() * *yr;
                }
            }
        };
        let mut out: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; n]);
        for (x, nb) in self.near.iter().enumerate().filter(|(_, nb)| !nb.idle()) {
            let (s, nd, nc) = (nb.s, nb.nd, nb.nc);
            for (ax, o) in out.iter_mut().enumerate() {
                let dt: Vec<C64> = (0..s).map(|r| (0..n).map(|j| nb.tg.dztot[r * 3 * n + ax * n + j] * a[j]).sum()).collect();
                back(&nb.left.ztot, &dt, o);
                if nd > 0 {
                    back(&nb.left.zdtot, &apply(&nb.tg.ddztot[ax], nd, a), o);
                }
                if nb.copied() {
                    back(&nb.left.tc, &apply(&nb.tg.dtc[ax], nc, a), o);
                }
            }
            let _ = cache.near_at(x);
        }
        for (row, db) in tr.far.iter().zip(&forces.far) {
            for (ax, o) in out.iter_mut().enumerate() {
                let onto = row.tlf.adjoint().apply(&row.r0.apply(&db[ax].apply(a)));
                o.iter_mut().zip(onto).for_each(|(u, v)| *u += v);
            }
        }
        out
    }
}

impl Transient {
    /// The cycle's conditional mean momentum given its length τ (pland.status # 第三轮, # 第四轮（续） §1; module
    /// documentation, # The kick), per lab axis: g(τ) = −Re Σ_jk d_j H′_jk c_k E_jk(τ)/p(τ), d_j = Σ_i c_i* e^{iλ_i*τ}
    /// i(λ_j − λ_i*) G_ij, E_jk = ∫₀^τ e^{−iλ_j(τ−s)}e^{−iλ_k s}ds; and its slow part's g_s(τ), the same sums with the slow
    /// modes' d and p and the long-time E ((s,s) whole, (s,f) its e^{−iλ_jτ} term, (f,s) its e^{−iλ_kτ} term, (f,f) none)
    /// -- none where the slow part emits nothing at τ.
    fn conditional(&self, hprime: &[Mat; 3], tau: f64) -> ([f64; 3], Option<[f64; 3]>) {
        let (n, d) = (self.n, self.size());
        let i = C64::I;
        let lam = &self.values;
        let e: Vec<C64> = lam.iter().map(|l| (C64::new(0.0, -tau) * *l).exp()).collect();
        let dvec = |upto: usize| -> Vec<C64> { (0..d).map(|j| (0..upto).map(|q| (self.c[q] * e[q]).conj() * i * (lam[j] - lam[q].conj()) * self.gram[(q, j)]).sum()).collect() };
        let (dall, dslow) = (dvec(d), dvec(n));
        let p_all = (0..d).map(|j| dall[j] * self.c[j] * e[j]).sum::<C64>().re;
        let p_slow = (0..n).map(|j| dslow[j] * self.c[j] * e[j]).sum::<C64>().re;
        let mut whole = [0.0; 3];
        let mut part = [0.0; 3];
        // The kernels of every mode pair, the same on the three axes: made once.
        let kernels: Vec<(C64, C64)> = (0..d)
            .flat_map(|j| (0..d).map(move |k| (j, k)))
            .map(|(j, k)| {
                let full = between(lam[j], lam[k], e[j], e[k], tau);
                let long = match (j < n, k < n) {
                    (true, true) => full,
                    (true, false) => -e[j] / (i * (lam[j] - lam[k])),
                    (false, true) => e[k] / (i * (lam[j] - lam[k])),
                    (false, false) => C64::ZERO,
                };
                (full, long)
            })
            .collect();
        for (ax, hp) in hprime.iter().enumerate() {
            let (mut w, mut h) = (C64::ZERO, C64::ZERO);
            for j in 0..d {
                for k in 0..d {
                    let weight = hp[(j, k)] * self.c[k];
                    let (full, long) = kernels[j * d + k];
                    w += dall[j] * weight * full;
                    h += dslow[j] * weight * long;
                }
            }
            whole[ax] = if p_all > 0.0 { -w.re / p_all } else { 0.0 };
            part[ax] = -h.re / p_slow;
        }
        (whole, (p_slow > 0.0).then_some(part))
    }

    /// H′ = ℓ†∂H_h r per lab axis on the modes (module documentation, # The kick): Ĥ′ = B⁻¹∂H_h B by blocks, B⁻¹ =
    /// [N⁻¹Φ_L†; F_L†], F_L†v = v_Q − T N⁻¹Φ_L†v -- A = Φ_L†∂HΦ (the far blocks whole), Bm = Φ_L†∂H F̂, Cm and Dm the fast
    /// rows' of ∂HΦ and ∂H F̂; Ĥ′_ss = N⁻¹A, Ĥ′_sf = N⁻¹Bm, Ĥ′_fs = Cm − T N⁻¹A, Ĥ′_ff = Dm − T N⁻¹Bm -- turned onto the
    /// eigenvectors (Q⁻¹Ĥ′Q).
    fn mode_forces(&self, ws: &Workspace, forces: &ForceParts) -> [Mat; 3] {
        let (n, d) = (self.n, self.size());
        let s: usize = self.blocks.iter().map(|b| b.2).sum();
        let rows = d - n;
        let solve_n = |m: &Mat| -> Mat {
            let mut out = m.clone();
            crate::linalg::solve(&ws.landing.lu, &ws.landing.piv, n, &mut out.data, m.cols);
            out
        };
        let (tn, tln) = (Mat::from_fn(s, n, |x, y| self.t[(x, y)]), Mat::from_fn(s, n, |x, y| self.tl[(x, y)]));
        let tlh = self.tl.adjoint();
        std::array::from_fn(|ax| {
            let h = &forces.near[ax];
            let block = |r0: usize, c0: usize, nr: usize, nc: usize| Mat::from_fn(nr, nc, |x, y| h[(r0 + x, c0 + y)]);
            let (hpp, hpn, hnp, hnn) = (block(0, 0, n, n), block(0, n, n, s), block(n, 0, s, n), block(n, n, s, s));
            // (Φ_L†∂H) on P: A_P = h_PP + T_L,near†h_NP + Σ_f T_L,f†∂B_f; on the near rows Bq = h_PN + T_L,near†h_NN.
            let (mut ap, mut bq) = (hpp, hpn);
            if s > 0 {
                ap.add_scaled(C64::ONE, &tln.adjoint_product(&hnp));
                bq.add_scaled(C64::ONE, &tln.adjoint_product(&hnn));
            }
            let mut amat = Mat::zeros(n, n);
            for (row, db) in self.far.iter().zip(&forces.far) {
                ap.add_scaled(C64::ONE, &row.tlf.adjoint_product(&db[ax]));
                amat.add_scaled(C64::ONE, &db[ax].adjoint().product(&row.tf));
            }
            amat.add_scaled(C64::ONE, &ap);
            if s > 0 {
                amat.add_scaled(C64::ONE, &bq.product(&tn));
            }
            // Bm = A_P(−T_L†) + [Bq | ∂B_f†β̂].
            let mut bm = ap.product(&tlh).scaled(-C64::ONE);
            for x in 0..n {
                for y in 0..s {
                    bm[(x, y)] += bq[(x, y)];
                }
            }
            // Cm: a near row's h_NP + h_NN T, a far row's β̂†∂B. Dm: a near row's h_NP(−T_L†) + [h_NN | 0], a far row's
            // β̂†∂B(−T_L†).
            let (mut cm, mut dm) = (Mat::zeros(rows, n), Mat::zeros(rows, rows));
            if s > 0 {
                let mut cn = hnp.clone();
                cn.add_scaled(C64::ONE, &hnn.product(&tn));
                put(&mut cm, &cn, 0, 0);
                put(&mut dm, &hnp.product(&tlh).scaled(-C64::ONE), 0, 0);
                for x in 0..s {
                    for y in 0..s {
                        dm[(x, y)] += hnn[(x, y)];
                    }
                }
            }
            for (row, db) in self.far.iter().zip(&forces.far) {
                let Some(r) = row.at else { continue };
                let onto = db[ax].adjoint().apply(&row.hat);
                for j in 0..n {
                    bm[(j, r)] += onto[j];
                }
                let lifted = lift(&db[ax], &row.hat);
                for j in 0..n {
                    cm[(r, j)] = lifted[j];
                }
                for y in 0..rows {
                    dm[(r, y)] = -(0..n).map(|j| lifted[j] * tlh[(j, y)]).sum::<C64>();
                }
            }
            let (hss, hsf) = (solve_n(&amat), solve_n(&bm));
            let mut out = Mat::zeros(d, d);
            put(&mut out, &self.slow.inverse.product(&hss).product(&self.slow.vectors), 0, 0);
            if rows > 0 {
                let (mut hfs, mut hff) = (cm, dm);
                hfs.add_scaled(-C64::ONE, &self.t.product(&hss));
                hff.add_scaled(-C64::ONE, &self.t.product(&hsf));
                put(&mut out, &self.slow.inverse.product(&hsf).product(&self.quick.vectors), 0, n);
                put(&mut out, &self.quick.inverse.product(&hfs).product(&self.slow.vectors), n, 0);
                put(&mut out, &self.quick.inverse.product(&hff).product(&self.quick.vectors), n, n);
            }
            out
        })
    }
}

/// φ₁(z) = (e^z − 1)/z, one at zero: by its series where |z| < 1/2.
fn phi1(z: C64) -> C64 {
    if z.abs() >= 0.5 {
        return (z.exp() - C64::ONE) / z;
    }
    let (mut term, mut sum) = (C64::ONE, C64::ONE);
    for m in 2..20 {
        term = term * z / m as f64;
        sum += term;
    }
    sum
}

/// E_jk(τ) = ∫₀^τ e^{−iλ_j(τ−s)}e^{−iλ_k s}ds = (e_k − e_j)/(i(λ_j − λ_k)), e = e^{−iλτ}: τ e_k φ₁(−i(λ_j − λ_k)τ) where
/// the two are close (τ e_j at λ_j = λ_k).
fn between(lj: C64, lk: C64, ej: C64, ek: C64, tau: f64) -> C64 {
    let z = C64::new(0.0, -tau) * (lj - lk);
    if z.abs() < 0.5 {
        ek * phi1(z) * tau
    } else {
        (ek - ej) / (C64::I * (lj - lk))
    }
}

impl Fold {
    /// The bare ∂H along the lab axis `a` on P ⊕ near (P first; not yet its Hermitian part; `ForceParts`): on P Ĥ_PP's
    /// pairs (∂c_t) and drives between ground and explicit places (∂e, ∂e*), the explicit states' dressing (its terms'
    /// ∂c_t); on a kept row its source (the ground columns' Σ_a ∂e_a Y_a, the explicit columns' static source), the
    /// emission into P from it (∂Ê) and ∂K̃ among the rows -- the step's tangent's (`Fold::prototype_tangent`).
    fn near_force(&self, ws: &Workspace, tr: &Transient, a: usize) -> Mat {
        let n = ws.n;
        let n3 = 3 * n;
        let s: usize = tr.blocks.iter().map(|b| b.2).sum();
        let mid = &ws.mid;
        let mut h = Mat::zeros(n + s, n + s);
        for (t, entries) in ws.pp.iter().enumerate() {
            for &(i, v) in entries {
                h[(i / n, i % n)] += v * mid.dc[t][a];
            }
        }
        for (g, k, direct, conj) in ws.lin.iter() {
            let d = mid.de[*g][*k][a];
            for &(i, v) in direct {
                h[(i / n, i % n)] += v * d;
            }
            for &(i, v) in conj {
                h[(i / n, i % n)] += v * d.conj();
            }
        }
        for nb in ws.near.iter().filter(|nb| !nb.idle() && !nb.xs.is_empty()) {
            for p in 0..n {
                for q in 0..n {
                    h[(p, q)] += nb.tg.dhx[p * n3 + a * n + q];
                }
            }
        }
        for &(x, o, sb) in &tr.blocks {
            let nb = &ws.near[x];
            let (ln, m) = (nb.lanes * n, 3 * nb.lanes * n);
            for i in 0..sb {
                for p in 0..n {
                    let emit = nb.tg.demit[(p * 3 + a) * sb + i];
                    h[(n + o + i, p)] += if nb.xcol[p] { nb.tg.src[i * m + a * ln + (nb.lanes - 1) * n + p] } else { emit.conj() };
                    h[(p, n + o + i)] += emit;
                }
                for l in 0..sb {
                    h[(n + o + i, n + o + l)] += nb.tg.dk[a * sb * sb + i * sb + l];
                }
            }
        }
        h
    }

    /// The gradient of the generator's Hermitian part on the transient's rows (`ForceParts`): the bare ∂H on P ⊕ near
    /// (`near_force`), and per far block its source's ∂B = Σ ∂e_a U_a. After the step's tangent.
    fn transient_forces(&self, ws: &Workspace, tr: &Transient) -> ForceParts {
        let n = ws.n;
        ForceParts {
            near: std::array::from_fn(|a| self.near_force(ws, tr, a).hermitian_part()),
            far: tr.far.iter().map(|row| std::array::from_fn(|a| far_source(self, &ws.far[row.x], n, &|c, k| ws.mid.de[c][k][a]))).collect(),
        }
    }
}


/// For the tests: the modes' photons laid by offset, the whole cycle's next landing and the photon at a time.
#[cfg(test)]
impl Transient {
    /// The emission of every mode's excited part per offset of the fold (rows (b, q) at 3b + q, 3 GROUND × d each): a near
    /// block's kept rows by C_qV and its explicit states' own rows on the mode's P part, a far block's by C_qR₀B on the slow
    /// modes' whole response and on a fast mode's row along β̂. A slow mode's P part q_j and rows Tq_j, a fast mode's P part
    /// −T_L†q_k and rows q_k. Per offset (its value) its matrix: one offset's photons coherent, two offsets' turning apart.
    fn emissions(&self, fold: &Fold, ws: &Workspace, cache: &Cache) -> Vec<(f64, Mat)> {
        let (n, d) = (self.n, self.size());
        let g3 = 3 * GROUND;
        let rows = d - n;
        let mut p = Mat::zeros(n, d);
        let mut q = Mat::zeros(rows, d);
        for j in 0..n {
            for i in 0..n {
                p[(i, j)] = self.slow.vectors[(i, j)];
            }
            let tq = self.t.apply(&(0..n).map(|i| self.slow.vectors[(i, j)]).collect::<Vec<_>>());
            for r in 0..rows {
                q[(r, j)] = tq[r];
            }
        }
        if rows > 0 {
            let back = self.tl.adjoint().product(&self.quick.vectors);
            for k in 0..rows {
                for i in 0..n {
                    p[(i, n + k)] = -back[(i, k)];
                }
                for r in 0..rows {
                    q[(r, n + k)] = self.quick.vectors[(r, k)];
                }
            }
        }
        let mut out: Vec<(f64, Mat)> = vec![];
        let mut onto = |value: f64, m: Mat| match out.iter_mut().find(|(v, _)| *v == value) {
            Some((_, sum)) => sum.add_scaled(C64::ONE, &m),
            None => out.push((value, m)),
        };
        for (x, nb) in ws.near.iter().enumerate().filter(|(_, nb)| !nb.idle()) {
            let basis = cache.near_at(x);
            let mut m = Mat::zeros(g3, d);
            if let Some(&(_, o, s)) = self.blocks.iter().find(|b| b.0 == x) {
                let emit = Mat { rows: g3, cols: s, data: basis.emit[..g3 * s].to_vec() };
                m.add_scaled(C64::ONE, &emit.product(&Mat::from_fn(s, d, |i, j| q[(o + i, j)])));
            }
            if !nb.xs.is_empty() {
                let xe = Mat { rows: g3, cols: n, data: basis.xemitc[..g3 * n].to_vec() };
                m.add_scaled(C64::ONE, &xe.product(&p));
            }
            onto(fold.blocks[nb.o].value, m);
        }
        for (x, fb) in ws.far.iter().enumerate().filter(|(_, fb)| !fb.omitted) {
            let basis = cache.far_at(x);
            let v = fb.vectors.len();
            let mut crb = Mat::zeros(g3, n);
            for (ia, &(cc, kk)) in fb.vectors.iter().enumerate() {
                let e = ws.mid.e[cc][kk];
                for (r, z) in crb.data.chunks_exact_mut(n).enumerate() {
                    let (b, qq) = (r / 3, r % 3);
                    let row = &basis.emitted[((b * v + ia) * 3 + qq) * n..((b * v + ia) * 3 + qq + 1) * n];
                    z.iter_mut().zip(row).for_each(|(z, y)| *z += e * *y);
                }
            }
            let mut m = crb.product(&Mat::from_fn(n, d, |i, j| if j < n { self.slow.vectors[(i, j)] } else { C64::ZERO }));
            if let Some((r, size)) = self.far.iter().find(|row| row.x == x).and_then(|row| row.at.map(|r| (r, row.size))) {
                let ca: Vec<C64> = crb.apply(&self.a).into_iter().map(|z| -z / size).collect();
                for k in 0..rows {
                    let z = q[(r, n + k)];
                    for (i, w) in ca.iter().enumerate() {
                        m[(i, n + k)] += *w * z;
                    }
                }
            }
            onto(fold.blocks[fb.o].value, m);
        }
        out
    }

    /// The whole cycle's next landing from the landing, every ground state's rows (b, q) at 3b + q: ∫₀^∞ Σ_q C_qΨ(τ)Ψ(τ)†C_q†
    /// dτ per offset (`emissions`) -- the exact clock's, its photon at τ* the whole state's (`Clock::owed` with the slow
    /// state's own).
    pub(crate) fn next_landing(&self, fold: &Fold, ws: &Workspace, cache: &Cache) -> Mat {
        let d = self.size();
        let cm = Mat::from_fn(d, d, |j, i| self.c[j] * self.c[i].conj() / (C64::I * (self.values[j] - self.values[i].conj())));
        let mut out = Mat::zeros(3 * GROUND, 3 * GROUND);
        for (_, e) in &self.emissions(fold, ws, cache) {
            out.add_scaled(C64::ONE, &e.product(&cm).product(&e.adjoint()));
        }
        out
    }

    /// The photon of the modes at τ from the landing, at the time t (every offset at its phase): the slow modes' where
    /// `slow`, the fast's where `fast` (`emissions` on c e^{−iΛτ}).
    pub(crate) fn photon_at(&self, fold: &Fold, ws: &Workspace, cache: &Cache, tau: f64, t: f64, slow: bool, fast: bool) -> Vec<C64> {
        let n = self.n;
        let w: Vec<C64> = (0..self.size()).map(|j| if (j < n && slow) || (j >= n && fast) { self.c[j] * (C64::new(0.0, -tau) * self.values[j]).exp() } else { C64::ZERO }).collect();
        let mut u = vec![C64::ZERO; 3 * GROUND];
        for (value, e) in &self.emissions(fold, ws, cache) {
            let phase = C64::cis(-value * t);
            for (o, x) in u.iter_mut().zip(e.apply(&w)) {
                *o += phase * x;
            }
        }
        u
    }

    /// The modes' eigenvalues (the slow first) and the switched state's coefficients on them.
    pub(crate) fn modes(&self) -> (&[C64], &[C64], usize) {
        (&self.values, &self.c, self.n)
    }

    /// τ* where the exact survival meets θ (`crossing`).
    pub(crate) fn crossing_at(&self, theta: f64) -> Option<f64> {
        self.crossing(theta).0
    }

    /// The fast part's photon at τ by the clock's own (`Workspace::fast_photon`).
    pub(crate) fn fast_photon_at(&self, fold: &Fold, ws: &Workspace, cache: &Cache, tau: f64, t: f64) -> Vec<C64> {
        ws.fast_photon(fold, cache, self, &self.fast(tau), t)
    }
}
