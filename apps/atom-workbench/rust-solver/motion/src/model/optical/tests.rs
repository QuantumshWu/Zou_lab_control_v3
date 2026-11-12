use super::*;
use crate::linalg::{inner, Lu as Complex, Mat};

/// A small generator for the tests: deterministic pseudo-random numbers in [−1, 1).
struct Draw(u64);
impl Draw {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn complex(&mut self) -> C64 {
        C64::new(self.next(), self.next())
    }
    fn hermitian(&mut self, n: usize, scale: f64) -> Vec<C64> {
        let mut m = vec![C64::ZERO; n * n];
        for i in 0..n {
            m[i * n + i] = C64::real(scale * self.next());
            for j in i + 1..n {
                let z = self.complex() * scale;
                m[i * n + j] = z;
                m[j * n + i] = z.conj();
            }
        }
        m
    }
}

/// A model of clusters (each its ground states first) for the dense reference: per cluster H (with its decay) and G_a,
/// and the jumps J_q (landing cluster, source cluster, matrix landing × source), the generator's decay their product.
struct Dense {
    sizes: Vec<(usize, usize)>,
    h: Vec<Mat>,
    g: Vec<[Vec<C64>; 3]>,
    jumps: Vec<(usize, usize, Vec<C64>)>,
}

impl Dense {
    /// Random clusters `sizes` (n, ground): Hermitian parts and gradients at random, the excited states' decay into
    /// every cluster's ground states by random jumps, and a folded cluster's own random jumps between its ground states.
    fn random(sizes: &[(usize, usize)], seed: u64) -> Dense {
        let mut d = Draw(seed);
        let ground: Vec<(usize, usize)> = sizes.iter().enumerate().flat_map(|(c, &(_, g))| (0..g).map(move |i| (c, i))).collect();
        let mut jumps = vec![];
        for (c, &(n, g)) in sizes.iter().enumerate() {
            for (landing, &(m, _)) in sizes.iter().enumerate() {
                for _ in 0..2 {
                    let mut j = vec![C64::ZERO; m * n];
                    for &(lc, b) in ground.iter().filter(|x| x.0 == landing) {
                        let _ = lc;
                        for s in 0..n {
                            // From excited states to every ground state; within a folded cluster between its ground states.
                            if s >= g || n == g {
                                j[b * n + s] = d.complex() * if n == g { 0.1 } else { 0.7 };
                            }
                        }
                    }
                    jumps.push((landing, c, j));
                }
            }
        }
        let mut h = vec![];
        let mut gs = vec![];
        for (c, &(n, _)) in sizes.iter().enumerate() {
            let herm = d.hermitian(n, 3.0);
            let mut m = Mat::zeros(n, n);
            for i in 0..n * n {
                m.data[i] = herm[i];
            }
            // −(i/2) Σ J†J of the jumps from this cluster.
            for (landing, source, j) in jumps.iter().filter(|x| x.1 == c) {
                let rows = sizes[*landing].0;
                let _ = source;
                for x in 0..n {
                    for y in 0..n {
                        let z: C64 = (0..rows).map(|b| j[b * n + x].conj() * j[b * n + y]).sum();
                        m.data[x * n + y] += C64::new(0.0, -0.5) * z;
                    }
                }
            }
            h.push(m);
            gs.push(std::array::from_fn(|_| d.hermitian(n, 2.0)));
        }
        Dense { sizes: sizes.to_vec(), h, g: gs, jumps }
    }

    /// The places of the blocks' entries in vec(ρ): per cluster its n² entries, row by row.
    fn offsets(&self) -> Vec<usize> {
        let mut out = vec![0];
        for &(n, _) in &self.sizes {
            out.push(out.last().unwrap() + n * n);
        }
        out
    }

    /// Re Tr[F_a Y], Y = 𝓡_Q X, of a traceless Hermitian X in cluster `at`, by the projector's definition on the whole vec
    /// space (module documentation, # The ground trace: Q X = E X + (Tr_g X) d, d the identity on `at`'s ground states over
    /// their number; P = 1 − Q), solved with the trace's column: (−Q𝓛Q + P + d⊗tr) Y = Q X -- Y in Q's range and
    /// traceless, since 𝓛 and Q keep the trace, P makes none, and P d = 0.
    fn reference(&self, at: usize, x: &[C64], a: usize) -> f64 {
        let offsets = self.offsets();
        let (off, d) = (&offsets, *offsets.last().unwrap());
        // 𝓛 as a matrix on vec(ρ).
        let mut l = Mat::zeros(d, d);
        for (c, &(n, _)) in self.sizes.iter().enumerate() {
            let h = &self.h[c];
            for p in 0..n {
                for q in 0..n {
                    let col = off[c] + p * n + q;
                    for k in 0..n {
                        // −i(H E_pq − E_pq H†): row (k, q) += −i H[k][p]; row (p, k) += i conj(H[k][q]).
                        l[(off[c] + k * n + q, col)] += C64::new(0.0, -1.0) * h[(k, p)];
                        l[(off[c] + p * n + k, col)] += C64::new(0.0, 1.0) * h[(k, q)].conj();
                    }
                }
            }
        }
        for (landing, source, j) in &self.jumps {
            let (m, n) = (self.sizes[*landing].0, self.sizes[*source].0);
            for p in 0..n {
                for q in 0..n {
                    for b in 0..m {
                        for b2 in 0..m {
                            l[(off[*landing] + b * m + b2, off[*source] + p * n + q)] += j[b * n + p] * j[b2 * n + q].conj();
                        }
                    }
                }
            }
        }
        // Q: the entries with an excited index as they are, every ground population onto d, the ground coherences none.
        let (n_at, g_at) = self.sizes[at];
        let dvec: Vec<usize> = (0..g_at).map(|i| off[at] + i * n_at + i).collect();
        let mut q = Mat::zeros(d, d);
        for (c, &(n, g)) in self.sizes.iter().enumerate() {
            for i in 0..n {
                for k in 0..n {
                    let j = off[c] + i * n + k;
                    if i >= g || k >= g {
                        q[(j, j)] = C64::ONE;
                    } else if i == k {
                        for &t in &dvec {
                            q[(t, j)] = C64::real(1.0 / g_at as f64);
                        }
                    }
                }
            }
        }
        let mut p = Mat::identity(d);
        p.add_scaled(-C64::ONE, &q);
        let qlq = q.product(&l).product(&q);
        let mut m = p.clone();
        m.add_scaled(-C64::ONE, &qlq);
        let trace: Vec<usize> = self.sizes.iter().enumerate().flat_map(|(c, &(n, _))| (0..n).map(move |i| off[c] + i * n + i)).collect();
        for &r in &dvec {
            for &cc in &trace {
                m[(r, cc)] += C64::real(1.0 / g_at as f64);
            }
        }
        let mut xv = vec![C64::ZERO; d];
        xv[off[at]..off[at + 1]].copy_from_slice(x);
        let mut y = q.apply(&xv);
        Complex::new(&m).expect("the bordered system").solve(&mut y);
        // Re Tr[F_a Y], F = −G.
        let mut out = 0.0;
        for (c, &(n, _)) in self.sizes.iter().enumerate() {
            for i in 0..n {
                for k in 0..n {
                    out -= (self.g[c][a][k * n + i] * y[off[c] + i * n + k]).re;
                }
            }
        }
        out
    }
}

/// K_Q of a cluster's sector (module documentation, # K_Q) against the projector's definition on the whole space
/// (`Dense::reference`): Tr[K_a X] = Re Tr[F_a 𝓡_Q X] for random traceless Hermitian X in an explicit cluster, beside a
/// folded cluster and a second explicit one, every excited state decaying into every cluster -- the jumps take no part,
/// and no other cluster enters.
#[test]
fn the_optical_operator_is_the_q_resolvent_s_adjoint() {
    for (sizes, seed) in [(vec![(2, 1)], 1), (vec![(5, 2), (3, 3)], 2), (vec![(4, 2), (3, 3), (3, 1)], 3), (vec![(6, 3), (2, 2)], 4)] {
        let dense = Dense::random(&sizes, seed);
        let sectors: Vec<Sector> = sizes.iter().enumerate().map(|(c, &(_, g))| Sector::new(&dense.h[c], &dense.g[c], g).expect("a sector")).collect();
        let mut draw = Draw(seed + 100);
        for (c, &(n, _)) in sizes.iter().enumerate().filter(|(_, s)| s.0 > s.1) {
            let (k, _) = sectors[c].operators(None).expect("an explicit cluster's K");
            for _ in 0..3 {
                let mut x = draw.hermitian(n, 1.0);
                let t: C64 = (0..n).map(|i| x[i * n + i]).sum::<C64>() * (1.0 / n as f64);
                for i in 0..n {
                    x[i * n + i] -= t;
                }
                for a in 0..3 {
                    let ours: f64 = (0..n * n).map(|p| (k[a][p] * x[(p % n) * n + p / n]).re).sum();
                    let exact = dense.reference(c, &x, a);
                    assert!((ours - exact).abs() <= 1e-10 * (1.0 + exact.abs()), "sizes {sizes:?}, cluster {c}, axis {a}: Tr[K X] {ours} against {exact}");
                }
            }
            for a in 0..3 {
                let size: f64 = (0..n).map(|i| (0..n).map(|j| (k[a][i * n + j] - k[a][j * n + i].conj()).abs()).sum::<f64>()).sum();
                assert!(size <= 1e-12 * (1.0 + k[a].iter().map(|z| z.abs()).sum::<f64>()), "K is Hermitian");
            }
        }
    }
}

/// The two-level atom at rest (Γ = k = 1; module documentation, # The split): P is empty, so the per-state optical noise
/// is the whole of what the trajectory drops, E[within_Q] = D_x − D_E, D_E the Ehrenfest mean's own fluctuation. Every
/// jump lands in g: the unravelling is a renewal process, its stationary means are integrals over the no-jump path from
/// g (survival S(t), the age's density S/E[τ]); D_E = ½ E[(Y − F̄τ)²]/E[τ] of the cycles' Y(τ) = ∫₀^τ ⟨F⟩dt; D_x =
/// Re Tr[F 𝓡 X̄] by the regression theorem. Standing (Ω₀ cos z at z = 0.3, the gradient −Ω₀ sin z) and running (Ω₀ e^{iz},
/// gradient iΩ₀ e^{iz}) waves, δ = 0 and −3, Ω₀ = 2, 5, 10.
#[test]
fn the_two_level_identity() {
    for running in [false, true] {
        for delta in [0.0, -3.0] {
            for om in [2.0, 5.0, 10.0] {
                let z: f64 = 0.3;
                let (rabi, slope) = if running { (C64::cis(z) * om, C64::new(0.0, om) * C64::cis(z)) } else { (C64::real(om * z.cos()), C64::real(-om * z.sin())) };
                // H = −δ|e⟩⟨e| + ½(Ω|e⟩⟨g| + h.c.) − (i/2)|e⟩⟨e| (g = 0, e = 1); G = ∂H_h, F = −G.
                let mut h = Mat::zeros(2, 2);
                h[(1, 1)] = C64::new(-delta, -0.5);
                h[(1, 0)] = rabi * 0.5;
                h[(0, 1)] = rabi.conj() * 0.5;
                let g: [Vec<C64>; 3] = std::array::from_fn(|a| if a == 2 { vec![C64::ZERO, slope.conj() * 0.5, slope * 0.5, C64::ZERO] } else { vec![C64::ZERO; 4] });
                let sector = Sector::new(&h, &g, 1).expect("the sector");
                let (k, _) = sector.operators(None).expect("K");
                // The no-jump path from g by RK4, its survival, ⟨F⟩ and within_Q, to where the survival is round-off.
                let force = |x: &[C64]| -> f64 { -(inner(x, &[g[2][0] * x[0] + g[2][1] * x[1], g[2][2] * x[0] + g[2][3] * x[1]])).re / crate::linalg::norm_sqr(x) };
                let deriv = |x: &[C64; 2]| -> [C64; 2] { [C64::new(0.0, -1.0) * (h[(0, 0)] * x[0] + h[(0, 1)] * x[1]), C64::new(0.0, -1.0) * (h[(1, 0)] * x[0] + h[(1, 1)] * x[1])] };
                let dt = 1e-3;
                let (mut x, mut t) = ([C64::ONE, C64::ZERO], 0.0);
                let (mut s, mut f, mut w) = (vec![], vec![], vec![]);
                loop {
                    let norm = x[0].norm_sqr() + x[1].norm_sqr();
                    s.push(norm);
                    f.push(force(&x));
                    w.push(crate::step::within(&k, &g, &x)[2][2]);
                    if norm < 1e-14 || t > 400.0 {
                        break;
                    }
                    let k1 = deriv(&x);
                    let x2 = [x[0] + k1[0] * (0.5 * dt), x[1] + k1[1] * (0.5 * dt)];
                    let k2 = deriv(&x2);
                    let x3 = [x[0] + k2[0] * (0.5 * dt), x[1] + k2[1] * (0.5 * dt)];
                    let k3 = deriv(&x3);
                    let x4 = [x[0] + k3[0] * dt, x[1] + k3[1] * dt];
                    let k4 = deriv(&x4);
                    for i in 0..2 {
                        x[i] += (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0);
                    }
                    t += dt;
                }
                // Trapezoids: E[τ] = ∫S, E[within] = ∫S w/E[τ]; the cycles' Y(τ) and their density −S′.
                let trap = |v: &dyn Fn(usize) -> f64| (1..s.len()).map(|i| 0.5 * dt * (v(i) + v(i - 1))).sum::<f64>();
                let tau = trap(&|i| s[i]);
                let within = trap(&|i| s[i] * w[i]) / tau;
                let mut y = vec![0.0; s.len()];
                for i in 1..s.len() {
                    y[i] = y[i - 1] + 0.5 * dt * (f[i] + f[i - 1]);
                }
                let density = |i: usize| -(s[(i + 1).min(s.len() - 1)] - s[i.saturating_sub(1)]) / (dt * ((i + 1).min(s.len() - 1) - i.saturating_sub(1)) as f64);
                let mean = trap(&|i| density(i) * y[i]) / tau;
                let d_e = 0.5 * trap(&|i| density(i) * (y[i] - mean * i as f64 * dt).powi(2)) / tau;
                // D_x by the dense regression: 𝓛 on vec(ρ) (gg, ge, eg, ee) with J = |g⟩⟨e|.
                let dense = Dense { sizes: vec![(2, 1)], h: vec![h.clone()], g: vec![g.clone()], jumps: vec![(0, 0, vec![C64::ZERO, C64::ONE, C64::ZERO, C64::ZERO])] };
                let d_x = {
                    // The steady state: 𝓛ρ = 0 with Tr ρ = 1, by the dense reference's own matrices.
                    let mut l = Mat::zeros(4, 4);
                    for p in 0..2 {
                        for q in 0..2 {
                            for kk in 0..2 {
                                l[(kk * 2 + q, p * 2 + q)] += C64::new(0.0, -1.0) * h[(kk, p)];
                                l[(p * 2 + kk, p * 2 + q)] += C64::new(0.0, 1.0) * h[(kk, q)].conj();
                            }
                        }
                    }
                    l[(0, 3)] += C64::ONE;
                    let mut a = l.clone();
                    for c in 0..4 {
                        a[(0, c)] = if c == 0 || c == 3 { C64::ONE } else { C64::ZERO };
                    }
                    let mut rho = vec![C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO];
                    Complex::new(&a).expect("steady").solve(&mut rho);
                    let fm: Vec<C64> = g[2].iter().map(|z| -*z).collect();
                    let mean_f = (0..4).map(|p| (fm[p] * rho[(p % 2) * 2 + p / 2]).re).sum::<f64>();
                    // X̄ = ½{F, ρ} − F̄ρ.
                    let mut xbar = vec![C64::ZERO; 4];
                    for i in 0..2 {
                        for j in 0..2 {
                            xbar[i * 2 + j] = (0..2).map(|m| (fm[i * 2 + m] * rho[m * 2 + j] + rho[i * 2 + m] * fm[m * 2 + j]) * 0.5).sum::<C64>() - rho[i * 2 + j] * mean_f;
                        }
                    }
                    dense.reference(0, &xbar, 2)
                };
                // Q is everything here: Tr[K X̄] is D_x itself, and E[within] = D_x − D_E.
                assert!((within - (d_x - d_e)).abs() <= 2e-4 * d_x.abs().max(1e-3), "running {running}, δ {delta}, Ω₀ {om}: E[within] {within:.6} against D_x − D_E = {d_x:.6} − {d_e:.6}");
            }
        }
    }
}

/// The stationary means of a cluster's renewal process (every jump lands in its first state; Γ-free units of the model's
/// rad/µs): the no-jump path x(t) = e^{−iHt}|0⟩ by RK4 at the step `dt` to where its survival is round-off, E[τ] = ∫S,
/// E[within_Q,ee] = ∫S w/E[τ], the Ehrenfest force's D_E,ee = ½E[(Y − F̄τ)²]/E[τ] of the cycles' Y(τ) = ∫₀^τ ⟨F_e⟩ (F = −G
/// along the unit vector e), and the excited share E[Σ_excited |x|²/‖x‖²]. (`the_two_level_identity` checks it.)
fn renewal(h: &Mat, g: &[Vec<C64>; 3], k: &[Vec<C64>; 3], ground: usize, e: [f64; 3], dt: f64) -> (f64, f64, f64) {
    let n = h.rows;
    let along = |m: &[Vec<C64>; 3]| -> Vec<C64> { (0..n * n).map(|p| m[0][p] * e[0] + m[1][p] * e[1] + m[2][p] * e[2]).collect() };
    let (ge, ke) = (along(g), along(k));
    let form = |m: &[C64], x: &[C64]| -> C64 { (0..n).map(|i| x[i].conj() * (0..n).map(|j| m[i * n + j] * x[j]).sum::<C64>()).sum() };
    let deriv = |x: &[C64]| -> Vec<C64> { (0..n).map(|i| C64::new(0.0, -1.0) * (0..n).map(|j| h[(i, j)] * x[j]).sum::<C64>()).collect() };
    let mut x = vec![C64::ZERO; n];
    x[0] = C64::ONE;
    let (mut s, mut f, mut w, mut p) = (vec![], vec![], vec![], vec![]);
    for step in 0.. {
        let norm = crate::linalg::norm_sqr(&x);
        s.push(norm);
        let (gm, km) = (form(&ge, &x).re / norm, form(&ke, &x).re / norm);
        f.push(-gm);
        // within_Q,ee = −Re⟨K x|G x⟩/‖x‖² + ⟨K⟩⟨G⟩ along e.
        let kx: Vec<C64> = (0..n).map(|i| (0..n).map(|j| ke[i * n + j] * x[j]).sum()).collect();
        let gx: Vec<C64> = (0..n).map(|i| (0..n).map(|j| ge[i * n + j] * x[j]).sum()).collect();
        w.push(-inner(&kx, &gx).re / norm + km * gm);
        p.push(x[ground..].iter().map(|z| z.norm_sqr()).sum::<f64>() / norm);
        if norm < 1e-14 || step > 50_000_000 {
            break;
        }
        let k1 = deriv(&x);
        let x2: Vec<C64> = (0..n).map(|i| x[i] + k1[i] * (0.5 * dt)).collect();
        let k2 = deriv(&x2);
        let x3: Vec<C64> = (0..n).map(|i| x[i] + k2[i] * (0.5 * dt)).collect();
        let k3 = deriv(&x3);
        let x4: Vec<C64> = (0..n).map(|i| x[i] + k3[i] * dt).collect();
        let k4 = deriv(&x4);
        for i in 0..n {
            x[i] += (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0);
        }
    }
    let trap = |v: &dyn Fn(usize) -> f64| (1..s.len()).map(|i| 0.5 * dt * (v(i) + v(i - 1))).sum::<f64>();
    let tau = trap(&|i| s[i]);
    let mut y = vec![0.0; s.len()];
    for i in 1..s.len() {
        y[i] = y[i - 1] + 0.5 * dt * (f[i] + f[i - 1]);
    }
    let last = s.len() - 1;
    let density = |i: usize| -(s[(i + 1).min(last)] - s[i.saturating_sub(1)]) / (dt * ((i + 1).min(last) - i.saturating_sub(1)) as f64);
    let mean = trap(&|i| density(i) * y[i]) / tau;
    let d_e = 0.5 * trap(&|i| density(i) * (y[i] - mean * i as f64 * dt).powi(2)) / tau;
    (trap(&|i| s[i] * w[i]) / tau, d_e, trap(&|i| s[i] * p[i]) / tau)
}

/// The fold gate (DESIGN §3.3): a drive kept explicit and the same drive folded at the atom give the atom one
/// absorbed-momentum diffusion where the fold holds. Kept (the cycling |2,2⟩ ↔ |3′,3′⟩, every jump back in |2,2⟩): what the
/// trajectory carries of its Ehrenfest force, D_E, and the optical sector's noise it draws, E[within_Q], each by the renewal
/// integrals of the cluster's no-jump path, with the generator, its gradient and K_Q of the model's own middle at rest
/// (`Middle::optical`, every other cluster at the point beside it); folded: half the absorbed light's second moment's rate
/// of the photons of |2,2⟩ (`Photon::moments`), along the probe's wave vector -- R2_RECOIL's running probe 100Γ off and
/// R3_DOPPLER's standing wave at Γ/2, s ≈ 0.02. They differ by the transition's saturation, which the fold leaves out --
/// the dipole fluctuations' (1 + s)⁻³ is 3s = 6 p_e of the excited share p_e at first order -- and by the recoil shift the
/// fold's response holds and the generator at rest does not: within 8 p_e and the shift's ratio to the half-width.
#[test]
fn a_kept_and_a_folded_drive_diffuse_alike() {
    use crate::field::Light;
    use crate::model::{Component, Model, Point, Unit};
    use crate::step::Middle as _;
    let scene = |name: &str| {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
        crate::request::Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap()
    };
    for (name, id) in [("R2_RECOIL", "probe"), ("R3_DOPPLER", "cool-plus")] {
        let request = scene(name);
        let light = Light::new(&request);
        let beam = request.beams.iter().position(|b| b.id == id).unwrap();
        let probe = light.frequencies.iter().position(|f| f.beams.contains(&beam)).unwrap();
        let bare = Model::new(&request, &light, &[]).unwrap();
        let lifted: Vec<Unit> = bare.bare_units(Component { frequency: probe, counter: false });
        let (b, e) = (7, 31);
        let folded = Model::new(&request, &light, &lifted).unwrap();
        let kept = Model::new(&request, &light, &lifted.iter().cloned().chain([Unit::Drive { component: Component { frequency: probe, counter: false }, excited: e, ground: b }]).collect::<Vec<_>>()).unwrap();
        let c = kept.cluster_of(b);
        assert_eq!(kept.clusters()[c].states, vec![b, e]);
        assert!((0..crate::request::GROUND).filter(|&x| x != b).all(|x| kept.emission.iter().all(|m| m[(x, e)] == C64::ZERO)), "{name}: |3′,3′⟩ decays to |2,2⟩ alone");
        let k = light.beams[beam].wave_vector();
        let axis = k.map(|x| x / crate::linalg::norm3(k));
        let half = 0.5 * kept.decay[(e - crate::request::GROUND, e - crate::request::GROUND)].re;
        for r in [[0.1, 0.2, -0.3], [-0.4, 0.05, 0.25]] {
            let at = Point { r, kappa: [0.0; 3], t: 0.37 };
            let middle = kept.assemble(c, r, at.t, &kept.running());
            let path = middle.path([0.0; 3], [0.0; 3]);
            let optical = middle.optical(&path, 0.1).expect("an explicit cluster");
            let mut h = Mat::zeros(2, 2);
            middle.generator(&path, 0.0, &mut h);
            let mut g: [Vec<C64>; 3] = Default::default();
            middle.gradient(&path, 0.0, &mut g);
            let dt = 0.02 / h.max_abs().max(half);
            let (within, d_e, excited) = renewal(&h, &g, &optical.k, 1, axis, dt);
            let photons = folded.photons(folded.cluster_of(b), &at, &[C64::ONE]);
            let second: f64 = photons.iter().map(|ph| ph.moments().second).map(|m| (0..3).map(|i| (0..3).map(|j| axis[i] * m[i][j] * axis[j]).sum::<f64>()).sum::<f64>()).sum();
            // The kept model's own folded classes from |2,2⟩ (its other drives, folded in both), at its ground share.
            let rest: f64 = kept.photons(c, &at, &[C64::ONE, C64::ZERO]).iter().map(|ph| ph.moments().second).map(|m| (0..3).map(|i| (0..3).map(|j| axis[i] * m[i][j] * axis[j]).sum::<f64>()).sum::<f64>()).sum();
            let (ours, fold) = (d_e + within + 0.5 * rest * (1.0 - excited), 0.5 * second);
            eprintln!("  the kept model's folded classes {:.3e}", 0.5 * rest);
            let recoil = kept.hbar_over_mass() * 0.5 * crate::linalg::dot3(k, k) / half;
            let within_tolerance = 8.0 * excited + recoil;
            eprintln!("{name} at {r:?}: kept D_E {d_e:.6e} + E[within_Q] {within:.6e} = {ours:.6e} against the fold's {fold:.6e} ({:+.2e} relative; excited share {excited:.2e}, recoil {recoil:.1e})", ours / fold - 1.0);
            assert!((ours / fold - 1.0).abs() <= within_tolerance, "{name}: kept {ours} folded {fold}");
        }
    }
}
