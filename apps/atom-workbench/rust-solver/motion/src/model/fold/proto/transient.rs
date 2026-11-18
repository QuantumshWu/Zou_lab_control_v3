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
use super::*;
use crate::linalg::Eigen;

/// A far block on the transient's rows (module documentation): its index, its source B at the middle's light (m × n), its
/// R₀ (m × m); and of the landing, its fast part's direction β̂ = β_f/‖β_f‖ and size ‖β_f‖, its row's K.
struct FarRow {
    x: usize,
    b: Mat,
    r0: Mat,
    hat: Vec<C64>,
    size: f64,
    k: C64,
}

/// The landing's transient on its modes (module documentation): the P places n, the fast rows -- the near blocks' kept rows
/// (per block its index, first row and rows) then one per far block whose fast part is not zero --; T, T_L and V on the
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
        // The far rows: per far block held, its source at the middle's light, T_f = R₀B, β_f = −T_f a, w_f = −Ba.
        let mut far = vec![];
        for (x, fb) in self.far.iter().enumerate().filter(|(_, fb)| !fb.omitted) {
            let basis = cache.far_at(x);
            let b = far_source(fold, fb, n, &|c, k| self.mid.e[c][k]);
            let r0 = basis.resolvent.clone();
            let ba = b.apply(a);
            let beta: Vec<C64> = r0.apply(&ba).into_iter().map(|z| -z).collect();
            let size = crate::linalg::norm_sqr(&beta).sqrt();
            work += (fb.m * n + fb.m * fb.m) as f64;
            if !(size > 0.0) {
                continue;
            }
            // K = z₀ − β̂†R₀⁻¹β̂, R₀⁻¹β_f = w_f = −Ba.
            let k = C64::real(basis.z0) + dot(&beta, &ba) / (size * size);
            far.push(FarRow { x, b, r0, hat: beta.iter().map(|z| *z / size).collect(), size, k });
        }
        let rows = s + far.len();
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
        for (r, row) in far.iter().enumerate() {
            let tf = row.r0.product(&row.b);
            let tlf = row.r0.adjoint().product(&row.b);
            let (tz, lz, vz) = (lift(&tf, &row.hat), lift(&tlf, &row.hat), lift(&row.b, &row.hat));
            for j in 0..n {
                (t[(s + r, j)], tl[(s + r, j)], v[(s + r, j)]) = (tz[j], lz[j], vz[j]);
            }
            k[(s + r, s + r)] = row.k;
            work += (2 * row.b.rows * row.b.rows * n + 3 * row.b.rows * n) as f64;
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
            let beta: Vec<C64> = (0..s).map(|r| -ta[r]).chain(far.iter().map(|row| C64::real(row.size))).collect();
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

/// The landing's clock (module documentation, # The clock): the threshold θ′ the slow part's norm meets where the exact
/// survival meets the trajectory's θ at τ* (θ itself where it never does), the photon the fast part owes that jump -- its amplitudes u_q on the ground states at the jump's time (rows (b, q) at 3b + q, every block at
/// its offset's phase, `Fold::prototype_photons`' layout) --, none where it is within the rounding of the slow photon's
/// there; and its count (complex multiply-adds).
#[derive(Clone, Debug)]
pub(crate) struct Clock {
    pub(crate) threshold: f64,
    pub(crate) owed: Option<Vec<C64>>,
    pub(crate) work: f64,
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
        let s: usize = tr.blocks.iter().map(|b| b.2).sum();
        for (r, row) in tr.far.iter().enumerate() {
            let fb = &self.far[row.x];
            let basis = cache.far_at(row.x);
            let v = fb.vectors.len();
            let phase = C64::cis(-fold.blocks[fb.o].value * t) * (-z[s + r] / row.size);
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

    /// The landing's clock (module documentation, # The clock; `Clock`) of a landing whose slow part's P amplitudes are `a`
    /// (a = N⁻¹f, `Workspace::taken`), against the trajectory's threshold θ, the landing at the time t; at the last
    /// forward, its left response made. None where the landing's modes do not exist (`Workspace::transient`).
    pub(crate) fn clock(&self, fold: &Fold, cache: &Cache, a: &[C64], theta: f64, t: f64) -> Option<Clock> {
        let tr = self.transient(fold, cache, a)?;
        let (crossed, evaluations) = tr.crossing(theta);
        let d = tr.size();
        let mut work = tr.work + (evaluations * 2 * d * d) as f64;
        let Some(wait) = crossed else { return Some(Clock { threshold: theta, owed: None, work }) };
        let [_, slow, _, rate] = tr.survival(wait);
        let z = tr.fast(wait);
        let u = self.fast_photon(fold, cache, &tr, &z, t + wait);
        work += (2 * d * d + 3 * GROUND * d) as f64;
        // The fast part's photon where it is beyond the rounding of the slow photon's (its hazard p_s at τ*).
        let floor = crate::units::rounding(d).powi(2) * rate.max(0.0);
        let owed = (crate::linalg::norm_sqr(&u) > floor).then_some(u);
        Some(Clock { threshold: slow, owed, work })
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
        let s: usize = self.blocks.iter().map(|b| b.2).sum();
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
            if let Some(r) = self.far.iter().position(|row| row.x == x) {
                let ca: Vec<C64> = crb.apply(&self.a).into_iter().map(|z| -z / self.far[r].size).collect();
                for k in 0..rows {
                    let z = q[(s + r, n + k)];
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

    /// The fast part's photon at τ by the clock's own (`Workspace::fast_photon`).
    pub(crate) fn fast_photon_at(&self, fold: &Fold, ws: &Workspace, cache: &Cache, tau: f64, t: f64) -> Vec<C64> {
        ws.fast_photon(fold, cache, self, &self.fast(tau), t)
    }
}
