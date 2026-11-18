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
        Some(Transient { n, blocks, far, t, tl, slow, quick, values, c, gram, work })
    }
}

impl Transient {
    /// The modes' count.
    fn size(&self) -> usize {
        self.values.len()
    }

    /// The emission of every mode's excited part per offset of the fold (laid as the photons are: rows (b, q) at 3b + q,
    /// 3 GROUND × d each, `Fold::prototype_photons`): a near block's kept rows by its emission on them (C_qV, `NearBasis::
    /// emit`) and its explicit states' own rows (`NearBasis::xemitc`) on the mode's P part, a far block's by C_qR₀B on the
    /// slow modes' whole response and on a fast mode's row along β̂ (C_qβ_f/‖β_f‖ times its coordinate). A slow mode's P
    /// part q_j and rows Tq_j (the far block's R₀Bq_j whole), a fast mode's P part −T_L†q_k and rows q_k. Per offset (its
    /// value, exact as the blocks hold it) its matrix: the photons of one offset are one Floquet node's, coherent; two
    /// offsets' turn apart (module documentation).
    fn emissions(&self, fold: &Fold, ws: &Workspace, cache: &Cache) -> Vec<(f64, Mat)> {
        let (n, d) = (self.n, self.size());
        let g3 = 3 * GROUND;
        let rows = d - n;
        // Every mode's P part (n × d) and its rows (rows × d).
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
        let first: Vec<usize> = (0..ws.near.len()).map(|x| self.blocks.iter().find(|b| b.0 == x).map_or(usize::MAX, |b| b.1)).collect();
        for (x, nb) in ws.near.iter().enumerate().filter(|(_, nb)| !nb.idle()) {
            let basis = cache.near_at(x);
            let mut m = Mat::zeros(g3, d);
            if nb.s > 0 {
                let o = first[x];
                let emit = Mat { rows: g3, cols: nb.s, data: basis.emit[..g3 * nb.s].to_vec() };
                m.add_scaled(C64::ONE, &emit.product(&Mat::from_fn(nb.s, d, |i, j| q[(o + i, j)])));
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
            // C_qR₀B = Σ_a e_a C_qR₀U_a on P (3 GROUND × n).
            let mut crb = Mat::zeros(g3, n);
            for (ia, &(cc, kk)) in fb.vectors.iter().enumerate() {
                let e = ws.mid.e[cc][kk];
                for b in 0..GROUND {
                    for qq in 0..3 {
                        let row = &basis.emitted[((b * v + ia) * 3 + qq) * n..((b * v + ia) * 3 + qq + 1) * n];
                        for (z, y) in crb.data[(3 * b + qq) * n..(3 * b + qq + 1) * n].iter_mut().zip(row) {
                            *z += e * *y;
                        }
                    }
                }
            }
            let mut m = Mat::zeros(g3, d);
            // The slow modes' whole response R₀Bq_j.
            m.add_scaled(C64::ONE, &crb.product(&Mat::from_fn(n, d, |i, j| if j < n { self.slow.vectors[(i, j)] } else { C64::ZERO })));
            // A fast mode's coordinate on this block's row, along β̂ = −R₀Ba/‖β_f‖: C_qβ̂ = −C_qR₀Ba/‖β_f‖.
            if let Some(r) = self.far.iter().position(|row| row.x == x) {
                let row = &self.far[r];
                let ca: Vec<C64> = crb.apply(&self.slow_part()).into_iter().map(|z| -z / row.size).collect();
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

    /// The mean waiting time from the landing, ∫₀^∞ N(τ)dτ = Σ_ij c_i*c_j G_ij/(i(λ_j − λ_i*)), N(τ) = ‖Ψ(τ)‖² the exact
    /// survival on the modes (µs; infinite where a mode does not decay).
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

    /// The slow part's P amplitudes a = Q_s c_s.
    fn slow_part(&self) -> Vec<C64> {
        self.slow.vectors.apply(&self.c[..self.n])
    }

    /// The transient's landing density on each landing set (`sets`), over its states and the emission's axes (3|set| square,
    /// rows (b, q) of the set at 3b + q): Δ = Σ_q C_q X_tr C_q† per offset (module documentation), its trace the transient's
    /// net emission (1 − c of a unit bare state's landing whose norm is c).
    pub(crate) fn landing_density(&self, fold: &Fold, ws: &Workspace, cache: &Cache, sets: &[Vec<usize>]) -> Vec<Mat> {
        let (n, d) = (self.n, self.size());
        // C_ji = c_j c_i*/(i(λ_j − λ_i*)), the slow pairs none.
        let cm = Mat::from_fn(d, d, |j, i| if j < n && i < n { C64::ZERO } else { self.c[j] * self.c[i].conj() / (C64::I * (self.values[j] - self.values[i].conj())) });
        let emissions = self.emissions(fold, ws, cache);
        sets.iter()
            .map(|set| {
                let m = 3 * set.len();
                let mut out = Mat::zeros(m, m);
                for (_, e) in &emissions {
                    let a = Mat::from_fn(m, d, |r, j| e[(3 * set[r / 3] + r % 3, j)]);
                    out.add_scaled(C64::ONE, &a.product(&cm).product(&a.adjoint()));
                }
                out
            })
            .collect()
    }
}
