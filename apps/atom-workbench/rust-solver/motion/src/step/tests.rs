//! The integrator against exact and analytic references on a test model of its trait (`Dynamics`): a two-level
//! atom (ground g, excited e of width Γ) in plane or standing waves of one frequency, in a quadratic or Gaussian trap that
//! holds both states alike; or the ground state alone, with or without a decay of its own.
//!
//! The model in the rotating frame, the coupling e ← g of each beam l c_l(r) = (Ω_l/2) e^{i(k_l·r + φ_l)}:
//! H = [[V, Σ c_l*], [Σ c_l, V − Δ − iΓ/2]], G = ∇H_h. Its Doppler frame K_g = 0, K_e = k₀ (the first beam's): the first
//! beam's coupling is static on the path, every other beam's a beat of wave vector k_l − k₀ (and its adjoint). The photon:
//! one class, u_a = √(Γ/3) ψ_e per lab axis (isotropic), landing in g; the jumps' kicks are the emission's −k n alone, of
//! a chosen |k| (zero where a test reads the Ehrenfest force).

use crate::adapt::{self, Tolerances};
use crate::linalg::{C64, Mat};
use crate::model::{Beat, Cluster, Generator, Part, Photon, Point, Quadratic, Recoil, STATES};
use crate::step::{self, Dynamics, Trajectory};

#[derive(Clone, Copy, Debug)]
struct Beam {
    k: [f64; 3],
    rabi: f64,
    phase: f64,
}

#[derive(Clone, Copy, Debug)]
enum Trap {
    /// V = ½ rᵀKr + g·r.
    Quadratic { hessian: [[f64; 3]; 3], gradient: [f64; 3] },
    /// V = −U exp(−Σ_a 2 r_a²/w_a²).
    Gaussian { depth: f64, waists: [f64; 3] },
}

impl Trap {
    fn expansion(&self, r: [f64; 3]) -> Quadratic {
        match *self {
            Trap::Quadratic { hessian, gradient } => {
                let kr: [f64; 3] = std::array::from_fn(|a| (0..3).map(|b| hessian[a][b] * r[b]).sum());
                let value = 0.5 * (0..3).map(|a| r[a] * kr[a]).sum::<f64>() + (0..3).map(|a| gradient[a] * r[a]).sum::<f64>();
                Quadratic { value, gradient: std::array::from_fn(|a| kr[a] + gradient[a]), hessian }
            }
            Trap::Gaussian { depth, waists } => {
                let q: f64 = (0..3).map(|a| 2.0 * r[a] * r[a] / (waists[a] * waists[a])).sum();
                let e = depth * (-q).exp();
                let dq: [f64; 3] = std::array::from_fn(|a| 4.0 * r[a] / (waists[a] * waists[a]));
                Quadratic {
                    value: -e,
                    gradient: std::array::from_fn(|a| e * dq[a]),
                    hessian: std::array::from_fn(|a| {
                        std::array::from_fn(|b| e * ((if a == b { 4.0 / (waists[a] * waists[a]) } else { 0.0 }) - dq[a] * dq[b]))
                    }),
                }
            }
        }
    }
}

struct Atom {
    hm: f64,
    trap: Trap,
    beams: Vec<Beam>,
    detuning: f64,
    /// Γ of e, or with no beams the ground state's own decay.
    gamma: f64,
    /// |k| of the emitted photon (its recoil −k n).
    emission: f64,
    /// An unbound atom beyond this distance from the origin is lost.
    region: f64,
    cluster: Cluster,
}

impl Atom {
    fn new(hm: f64, trap: Trap, beams: Vec<Beam>, detuning: f64, gamma: f64) -> Atom {
        let explicit = !beams.is_empty();
        let cluster = Cluster {
            states: if explicit { vec![0, 8] } else { vec![0] },
            ground: 1,
            frames: if explicit { vec![Beat(vec![0]), Beat(vec![1])] } else { vec![Beat(vec![0])] },
            waves: if explicit { vec![[0.0; 3], beams[0].k] } else { vec![[0.0; 3]] },
            waves_exact: if explicit { vec![Beat(vec![]), Beat(vec![1])] } else { vec![Beat(vec![])] },
        };
        Atom { hm, trap, beams, detuning, gamma, emission: 0.0, region: f64::INFINITY, cluster }
    }
    fn n(&self) -> usize {
        self.cluster.states.len()
    }
    fn coupling(&self, l: usize, r: [f64; 3]) -> C64 {
        let b = self.beams[l];
        C64::cis(b.k[0] * r[0] + b.k[1] * r[1] + b.k[2] * r[2] + b.phase) * (0.5 * b.rabi)
    }
}

fn zeros(n: usize) -> [Mat; 3] {
    [Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n)]
}

impl Dynamics for Atom {
    fn hbar_over_mass(&self) -> f64 {
        self.hm
    }
    fn reference(&self, r: [f64; 3]) -> Quadratic {
        self.trap.expansion(r)
    }
    fn cluster(&self, _: usize) -> &Cluster {
        &self.cluster
    }
    fn cluster_of(&self, _: usize) -> usize {
        0
    }
    fn generator(&self, cluster: usize, at: &Point) -> Generator {
        let (n, v) = (self.n(), self.trap.expansion(at.r));
        let mut m = Mat::zeros(n, n);
        let mut g = zeros(n);
        for s in 0..n {
            m[(s, s)] = C64::real(v.value);
            for a in 0..3 {
                g[a][(s, s)] = C64::real(v.gradient[a]);
            }
        }
        let mut parts = Vec::new();
        if n == 1 {
            m[(0, 0)] += C64::new(0.0, -0.5 * self.gamma);
        } else {
            m[(1, 1)] += C64::new(-self.detuning, -0.5 * self.gamma);
            for l in 0..self.beams.len() {
                let (c, k) = (self.coupling(l, at.r), self.beams[l].k);
                let grad: [C64; 3] = std::array::from_fn(|a| C64::new(0.0, k[a]) * c);
                if l == 0 {
                    m[(1, 0)] += c;
                    m[(0, 1)] += c.conj();
                    for a in 0..3 {
                        g[a][(1, 0)] += grad[a];
                        g[a][(0, 1)] += grad[a].conj();
                    }
                } else {
                    let q: [f64; 3] = std::array::from_fn(|a| k[a] - self.beams[0].k[a]);
                    let (mut up, mut down) = (Mat::zeros(n, n), Mat::zeros(n, n));
                    let (mut gu, mut gd) = (zeros(n), zeros(n));
                    up[(1, 0)] = c;
                    down[(0, 1)] = c.conj();
                    for a in 0..3 {
                        gu[a][(1, 0)] = grad[a];
                        gd[a][(0, 1)] = grad[a].conj();
                    }
                    parts.push(Part { beat: Beat(vec![0]), omega: 0.0, wave: q, generator: up, force: gu });
                    parts.push(Part { beat: Beat(vec![0]), omega: 0.0, wave: q.map(|x| -x), generator: down, force: gd });
                }
            }
        }
        parts.insert(0, Part { beat: Beat(vec![]), omega: 0.0, wave: [0.0; 3], generator: m, force: g });
        Generator { cluster, at: *at, parts }
    }
    fn photons(&self, _: usize, _: &Point, psi: &[C64]) -> Vec<Photon> {
        let n = self.n();
        let source = psi[n - 1] * (self.gamma / 3.0).sqrt();
        let mut u = vec![C64::ZERO; n];
        u[0] = source;
        let none = || vec![C64::ZERO; n];
        vec![Photon {
            landing: 0,
            wave_number: self.emission,
            amplitudes: [u.clone(), u.clone(), u],
            gradients: std::array::from_fn(|_| [none(), none(), none()]),
            explicit: Vec::new(),
        }]
    }
    fn recoil(&self, _: &Photon, _: &[f64; 3]) -> Recoil {
        Recoil { mean: [0.0; 3], second: [[0.0; 3]; 3] }
    }
    fn energy(&self, _: usize, at: &Point, _: &[C64]) -> f64 {
        0.5 * self.hm * at.kappa.iter().map(|k| k * k).sum::<f64>() + self.trap.expansion(at.r).value
    }
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        at.r.iter().map(|x| x * x).sum::<f64>().sqrt() > self.region && self.energy(cluster, at, psi) > 0.0
    }
    fn readout(&self, _: usize, _: &Point, psi: &[C64]) -> [f64; STATES] {
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let mut rows = [0.0; STATES];
        for (s, &state) in self.cluster.states.iter().enumerate() {
            rows[state] = psi[s].norm_sqr() / norm;
        }
        rows
    }
    fn assembly_work(&self) -> f64 {
        (self.n() * self.n()) as f64
    }
}

/// ħ/m of Rb87 (µm²/µs).
const RB: f64 = 7.3072e-4;
/// The D2 line's wave number (rad/µm) and width Γ (1/µs).
const K: f64 = 8.0528;
const GAMMA: f64 = 38.117;

/// The largest |a − b| over two vectors relative to the largest |b|.
fn relative(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().map(|x| x.abs()).fold(0.0, f64::max);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale
}

/// A trajectory at the ground state at (r, κ) whose threshold no step reaches (a jump-free propagation).
fn held(model: &Atom, r: [f64; 3], kappa: [f64; 3]) -> Trajectory {
    let mut t = Trajectory::new(model, 0, r, kappa, 7, 0);
    t.threshold = f64::MIN_POSITIVE;
    t
}

/// RK4 of ṙ = (ħ/m)κ, κ̇ = −∇V on a fine grid: the dense reference of a conservative flow.
fn rk4(trap: &Trap, hm: f64, r: [f64; 3], kappa: [f64; 3], tau: f64, steps: usize) -> ([f64; 3], [f64; 3]) {
    let h = tau / steps as f64;
    let f = |y: [f64; 6]| -> [f64; 6] {
        let g = trap.expansion([y[0], y[1], y[2]]).gradient;
        [hm * y[3], hm * y[4], hm * y[5], -g[0], -g[1], -g[2]]
    };
    let mut y = [r[0], r[1], r[2], kappa[0], kappa[1], kappa[2]];
    for _ in 0..steps {
        let k1 = f(y);
        let k2 = f(std::array::from_fn(|i| y[i] + 0.5 * h * k1[i]));
        let k3 = f(std::array::from_fn(|i| y[i] + 0.5 * h * k2[i]));
        let k4 = f(std::array::from_fn(|i| y[i] + h * k3[i]));
        y = std::array::from_fn(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
    }
    ([y[0], y[1], y[2]], [y[3], y[4], y[5]])
}

/// A tilted quadratic potential with a stiff axis (a 100 kHz trap's curvature), a hyperbolic one and a flat one, and a
/// constant force: its flow is the reference's own, so one step of several trap periods is its exact flow (the kick
/// vanishes, V_ref being V's expansion at the middle), against RK4 at a 10⁻⁴ µs step.
#[test]
fn a_step_is_the_exact_flow_of_a_quadratic_potential() {
    let (c, s) = (0.6_f64, 0.8_f64);
    let axes = [[c, s, 0.0], [-s * 0.6, c * 0.6, 0.8], [s * 0.8, -c * 0.8, 0.6]];
    let curvature = [543.0, -100.0, 0.0];
    let hessian: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| (0..3).map(|i| curvature[i] * axes[i][a] * axes[i][b]).sum()));
    let trap = Trap::Quadratic { hessian, gradient: [1.0, -2.0, 3.0] };
    let model = Atom::new(RB, trap, vec![], 0.0, 0.0);
    let (r, kappa) = ([0.3, -0.2, 0.1], [40.0, 25.0, -60.0]);
    let tau = 20.0;
    let s = step::step(&model, &held(&model, r, kappa), tau).unwrap();
    let (re, ke) = rk4(&trap, RB, r, kappa, tau, 200_000);
    let (dr, dk) = (relative(&s.end.r, &re), relative(&s.end.kappa, &ke));
    assert!(dr < 1e-10 && dk < 1e-10, "r {:?} vs {re:?} ({dr:e}); κ {:?} vs {ke:?} ({dk:e})", s.end.r, s.end.kappa);
}

/// The step is its own inverse (SPEC §2.1): a hot atom in a Gaussian tweezer (anharmonic: the reference is the middle's
/// expansion, a fixed point), one step of τ and one of −τ from its end return to the start to round-off.
#[test]
fn a_step_and_its_reverse_return_to_the_start() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.2, 3.0] };
    let model = Atom::new(RB, trap, vec![], 0.0, 0.0);
    let (r, kappa) = ([0.35, -0.2, 0.9], [60.0, -45.0, 30.0]);
    for tau in [0.5, 2.0, 5.0] {
        let start = held(&model, r, kappa);
        let there = step::step(&model, &start, tau).unwrap();
        let back = step::step(&model, &there.end, -tau).unwrap();
        let (dr, dk) = (relative(&back.end.r, &r), relative(&back.end.kappa, &kappa));
        assert!(dr < 1e-13 && dk < 1e-13, "τ {tau}: r {:?} ({dr:e}), κ {:?} ({dk:e})", back.end.r, back.end.kappa);
    }
}

/// The tolerances a test asks: per µs of an energy (each axis's and E_s) and of the rows.
fn tolerances(energy: f64, population: f64) -> Tolerances {
    Tolerances { axes: energy, energy, population }
}

/// e^{−iHτ}ψ of a static generator (`linalg::exp`): the at-rest reference.
fn evolve(h: &Mat, psi: &[C64], tau: f64) -> Vec<C64> {
    crate::linalg::exp(h, C64::new(0.0, -tau)).apply(psi)
}

/// Galilei covariance (propagation.md §1.2): an atom moving uniformly through a running wave is an atom at rest at the
/// detuning less k·v, v the step's path velocity (the mean of its momenta before and after the kick). A mass so large
/// that the kicks barely move the velocity; one step of the moving atom's internal state against e^{−iH′τ} at rest:
/// populations and the norm, gauge-invariant, to round-off.
#[test]
fn a_moving_atom_is_an_atom_at_rest_at_the_shifted_detuning() {
    let hm = 1e-9;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let (rabi, detuning) = (18.85, -10.0);
    let beam = Beam { k: [0.0, 0.3 * K, 0.95 * K], rabi, phase: 0.4 };
    let model = Atom::new(hm, flat, vec![beam], detuning, GAMMA);
    for v in [[0.0, 0.0, 0.0], [0.2, -0.4, 0.6], [0.0, 0.0, -1.5]] {
        let kappa = v.map(|x| x / hm);
        let tau = 0.3;
        let s = step::step(&model, &held(&model, [0.1, 0.2, 0.3], kappa), tau).unwrap();
        // The path's velocity: the mean of the momenta before and after the kick (flat trap: κ_h = κ₀).
        let kv: f64 = (0..3).map(|a| beam.k[a] * hm * 0.5 * (kappa[a] + s.segment.kicked.1[a])).sum();
        let mut rest = Mat::zeros(2, 2);
        rest[(1, 0)] = C64::real(0.5 * rabi);
        rest[(0, 1)] = C64::real(0.5 * rabi);
        rest[(1, 1)] = C64::new(-(detuning - kv), -0.5 * GAMMA);
        let reference = evolve(&rest, &[C64::ONE, C64::ZERO], tau);
        let moving: Vec<f64> = s.end.psi.iter().map(|z| z.norm_sqr()).collect();
        let at_rest: Vec<f64> = reference.iter().map(|z| z.norm_sqr()).collect();
        assert!(relative(&moving, &at_rest) < 1e-12, "v {v:?}: {moving:?} vs {at_rest:?}");
    }
}

/// RK4 of iψ̇ = H(t)ψ over [0, τ] in `steps` steps.
fn rk4_state(h: &dyn Fn(f64) -> Mat, psi: &[C64], tau: f64, steps: usize) -> Vec<C64> {
    let dt = tau / steps as f64;
    let derivative = |time: f64, y: &[C64]| -> Vec<C64> { h(time).apply(y).into_iter().map(|z| C64::new(z.im, -z.re)).collect() };
    let add = |a: &[C64], b: &[C64], s: f64| -> Vec<C64> { a.iter().zip(b).map(|(x, y)| *x + *y * s).collect() };
    let mut y = psi.to_vec();
    for i in 0..steps {
        let t0 = i as f64 * dt;
        let k1 = derivative(t0, &y);
        let k2 = derivative(t0 + 0.5 * dt, &add(&y, &k1, 0.5 * dt));
        let k3 = derivative(t0 + 0.5 * dt, &add(&y, &k2, 0.5 * dt));
        let k4 = derivative(t0 + dt, &add(&y, &k3, dt));
        y = (0..y.len()).map(|a| y[a] + (k1[a] + k2[a] * 2.0 + k3[a] * 2.0 + k4[a]) * (dt / 6.0)).collect();
    }
    y
}

/// The same in a standing wave: the moving atom sees the two beams at Δ ∓ k·v, a beat at 2k·v the path's CF4 grid
/// resolves; against RK4 of the at-rest atom's beating generator at a 10⁻⁵ µs step, within the grid's fourth order at a
/// fine spacing.
#[test]
fn a_moving_atom_in_a_standing_wave_is_an_atom_at_rest_with_two_detunings() {
    let hm = 1e-9;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let (rabi, detuning) = (12.0, -15.0);
    let beams = vec![Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.7 }];
    let model = Atom::new(hm, flat, beams.clone(), detuning, GAMMA);
    let (v, z0, tau) = (0.8, 0.13, 0.5);
    let mut t = held(&model, [0.0, 0.0, z0], [0.0, 0.0, v / hm]);
    t.grid = 1e-4;
    let s = step::step(&model, &t, tau).unwrap();
    // At rest, the lab generator along z(t) = z₀ + v t.
    let h = |time: f64| -> Mat {
        let z = z0 + v * time;
        let c: C64 = beams.iter().map(|b| C64::cis(b.k[2] * z + b.phase) * (0.5 * b.rabi)).sum();
        let mut m = Mat::zeros(2, 2);
        m[(1, 0)] = c;
        m[(0, 1)] = c.conj();
        m[(1, 1)] = C64::new(-detuning, -0.5 * GAMMA);
        m
    };
    let y = rk4_state(&h, &[C64::ONE, C64::ZERO], tau, 50_000);
    let moving: Vec<f64> = s.end.psi.iter().map(|z| z.norm_sqr()).collect();
    let at_rest: Vec<f64> = y.iter().map(|z| z.norm_sqr()).collect();
    assert!(relative(&moving, &at_rest) < 1e-6, "{moving:?} vs {at_rest:?}");
}

/// The steady excited population of a two-level atom at the detuning δ: (Ω²/4)/(δ² + Γ²/4 + Ω²/2).
fn excited(rabi: f64, delta: f64) -> f64 {
    0.25 * rabi * rabi / (delta * delta + 0.25 * GAMMA * GAMMA + 0.5 * rabi * rabi)
}

/// The mean force (1/µm per µs) over trajectories of a uniformly moving atom (a mass the kicks do not move: ħ/m of 10⁻⁹
/// µm²/µs, its κ's round-off still far below the tolerances), from `from`
/// to `to` µs, the jumps without recoil (the Ehrenfest force alone, whose mean the unraveling makes the master
/// equation's): per lab axis the mean and its standard error.
fn mean_force(model: &Atom, r: [f64; 3], v: [f64; 3], trajectories: u64, from: f64, to: f64, tol: &Tolerances) -> ([f64; 3], [f64; 3]) {
    let mut forces = Vec::new();
    for i in 0..trajectories {
        let mut t = Trajectory::new(model, 0, r, v.map(|x| x / model.hm), 11, i);
        adapt::advance(model, tol, &mut t, from, &mut |_| {});
        let start = t.kappa;
        adapt::advance(model, tol, &mut t, to, &mut |_| {});
        forces.push(std::array::from_fn::<f64, 3, _>(|a| (t.kappa[a] - start[a]) / (to - from)));
    }
    let n = forces.len() as f64;
    let mean: [f64; 3] = std::array::from_fn(|a| forces.iter().map(|f| f[a]).sum::<f64>() / n);
    let error = std::array::from_fn(|a| (forces.iter().map(|f| (f[a] - mean[a]).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt());
    (mean, error)
}

/// The radiation pressure of a running wave on a moving atom: kΓρ_ee(Δ − k·v) along k (the master equation's steady
/// state), at several velocities through the Doppler profile, within three standard errors and 2 %.
#[test]
fn a_running_wave_s_force_is_the_bloch_equations_at_the_doppler_shifted_detuning() {
    let hm = 1e-9;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let (rabi, detuning) = (15.0, -12.0);
    let model = Atom::new(hm, flat, vec![Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }], detuning, GAMMA);
    let tol = tolerances(f64::INFINITY, 1e-6);
    for v in [-3.0, -1.5, 0.0, 1.5, 3.0] {
        let expected = K * GAMMA * excited(rabi, detuning - K * v);
        let (f, e) = mean_force(&model, [0.0; 3], [0.0, 0.0, v], 48, 2.0, 22.0, &tol);
        let miss = (f[2] - expected).abs();
        eprintln!("v {v}: F_z {} ± {} against {expected}", f[2], e[2]);
        assert!(miss <= 3.0 * e[2] && miss <= 0.02 * expected, "v {v}: F_z {} ± {} against {expected}", f[2], e[2]);
    }
}

/// The master equation of a two-level atom moving uniformly through a standing wave, z = z₀ + vt: RK4 at 10⁻³ µs, the
/// force −Re⟨G⟩ = −2Re(G_eg ρ_ge) averaged over [from, to] (the dense reference of the trajectories' mean).
fn bloch_force(beams: &[Beam], detuning: f64, z0: f64, v: f64, from: f64, to: f64) -> f64 {
    let coupling = |z: f64| -> (C64, C64) {
        let c: C64 = beams.iter().map(|b| C64::cis(b.k[2] * z + b.phase) * (0.5 * b.rabi)).sum();
        let g: C64 = beams.iter().map(|b| C64::new(0.0, b.k[2]) * C64::cis(b.k[2] * z + b.phase) * (0.5 * b.rabi)).sum();
        (c, g)
    };
    // ρ = [ρ_gg, ρ_ee, ρ_eg]; ρ_ge = ρ_eg*. ρ̇ = −i(Hρ − ρH†) + Γρ_ee|g⟩⟨g|, H = [[0, c*], [c, −Δ − iΓ/2]].
    let derivative = |time: f64, r: [C64; 3]| -> [C64; 3] {
        let (c, _) = coupling(z0 + v * time);
        let (gg, ee, eg) = (r[0], r[1], r[2]);
        let ge = eg.conj();
        let i = C64::I;
        let dgg = -i * (c.conj() * eg - ge * c) + C64::real(GAMMA) * ee;
        let dee = -i * (c * ge - eg * c.conj()) - C64::real(GAMMA) * ee;
        let deg = -i * (c * gg + C64::new(-detuning, -0.5 * GAMMA) * eg - ee * c);
        [dgg, dee, deg]
    };
    let dt = 1e-3;
    let mut r = [C64::ONE, C64::ZERO, C64::ZERO];
    let (mut sum, mut count) = (0.0, 0.0);
    let steps = (to / dt).round() as usize;
    for i in 0..steps {
        let t0 = i as f64 * dt;
        let add = |a: [C64; 3], b: [C64; 3], s: f64| -> [C64; 3] { std::array::from_fn(|k| a[k] + b[k] * s) };
        let k1 = derivative(t0, r);
        let k2 = derivative(t0 + 0.5 * dt, add(r, k1, 0.5 * dt));
        let k3 = derivative(t0 + 0.5 * dt, add(r, k2, 0.5 * dt));
        let k4 = derivative(t0 + dt, add(r, k3, dt));
        r = std::array::from_fn(|k| r[k] + (k1[k] + k2[k] * 2.0 + k3[k] * 2.0 + k4[k]) * (dt / 6.0));
        if t0 + dt > from {
            let (_, g) = coupling(z0 + v * (t0 + dt));
            sum += -2.0 * (g * r[2].conj()).re;
            count += 1.0;
        }
    }
    sum / count
}

/// The force on an atom moving through a standing wave (the velocity-dependent force of Doppler cooling, and its
/// dipole part averaged over the wave), against the master equation along the same uniform path: within three standard
/// errors and 3 % of the running wave's force, at several velocities.
#[test]
fn a_standing_wave_s_force_is_the_bloch_equations_along_the_path() {
    let hm = 1e-9;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let (rabi, detuning) = (10.0, -15.0);
    let beams = vec![Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.0 }];
    let model = Atom::new(hm, flat, beams.clone(), detuning, GAMMA);
    let scale = K * GAMMA * excited(rabi, detuning);
    for v in [0.3, 1.0, 2.0] {
        // The momentum's error 0.5/µm per µs: on the moving axis an energy's error of v δκ.
        let tol = tolerances(0.5 * v, 1e-4);
        let expected = bloch_force(&beams, detuning, 0.0, v, 3.0, 33.0);
        let (f, e) = mean_force(&model, [0.0; 3], [0.0, 0.0, v], 48, 3.0, 33.0, &tol);
        let miss = (f[2] - expected).abs();
        eprintln!("v {v}: F_z {} ± {} against {expected} (scale {scale})", f[2], e[2]);
        assert!(miss <= 3.0 * e[2] && miss <= 0.03 * scale, "v {v}: F_z {} ± {} against {expected} (scale {scale})", f[2], e[2]);
    }
}

/// The waiting times (SPEC §1.6): a ground state of constant hazard γ, ‖ψ‖² = e^{−γt} between jumps, jumps at the roots
/// ln(1/θ_j)/γ of the thresholds its streams draw (θ₀ from jump 0's, θ_j the first draw of jump j's). Over thousands of
/// jumps the count is the thresholds' and the final norm theirs to round-off: every root found to round-off. Their
/// waiting times' mean and variance those of the exponential law within four standard errors.
#[test]
fn the_jumps_are_at_the_roots_of_the_norm() {
    let gamma = 2.0;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let model = Atom::new(RB, flat, vec![], 0.0, gamma);
    let mut t = Trajectory::new(&model, 0, [0.0; 3], [0.0; 3], 3, 5);
    let end = 2000.0;
    adapt::advance(&model, &tolerances(1.0, 1e-6), &mut t, end, &mut |_| {});
    // The jumps the thresholds put before `end`, and the last one's time.
    let threshold = |j: u32| crate::random::Stream::jump(3, 5, j).open_uniform();
    let (mut last, mut j, mut waits) = (0.0, 0, Vec::new());
    loop {
        let wait = (1.0 / threshold(j)).ln() / gamma;
        if last + wait > end {
            break;
        }
        waits.push(wait);
        last += wait;
        j += 1;
    }
    assert_eq!(t.jumps, j, "the jumps made against the thresholds' count");
    let norm: f64 = t.psi.iter().map(|z| z.norm_sqr()).sum();
    let expected = (-gamma * (end - last)).exp();
    assert!((norm / expected - 1.0).abs() < 1e-9, "‖ψ‖² {norm} against {expected} after the last jump at {last}");
    let n = waits.len() as f64;
    let mean = waits.iter().sum::<f64>() / n;
    let variance = waits.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / (n - 1.0);
    assert!((mean * gamma - 1.0).abs() < 4.0 / n.sqrt(), "mean {mean} against {}", 1.0 / gamma);
    assert!((variance * gamma * gamma - 1.0).abs() < 4.0 * (8.0 / n).sqrt(), "variance {variance} against {}", 1.0 / (gamma * gamma));
}

/// R1's mechanism (SPEC §2.2): a conservative anharmonic trap's energy over a long run. A hot atom in a Gaussian tweezer for
/// some 50 radial periods. At fixed steps the splitting, symplectic and its own inverse, keeps the energy's error bounded:
/// its last tenth's mean within its first tenth's spread (no drift). At adaptive steps the step's choice, made from the
/// last step's error, is not a symmetric function of each step, and the energy drifts: measured 2-5 % of the run's
/// allotted error s·ε, a smaller share at tighter tolerances -- within the booked bound, so the plain controller stays
/// (SPEC §2.2's symmetric step choice is the remedy were it beyond). Asserted: within the allotment.
#[test]
fn a_conservative_trap_keeps_its_energy() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.0, 3.0] };
    let model = Atom::new(RB, trap, vec![], 0.0, 0.0);
    let start = Trajectory::new(&model, 0, [0.3, -0.2, 0.8], [20.0, 15.0, -10.0], 1, 0);
    let e0 = model.energy(0, &start.point(), &start.psi);
    let (run, outputs, rate) = (5000.0, 100, 1e-6);
    let tol = Tolerances { axes: rate, energy: rate, population: 1.0 };
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let mut t = start.clone();
    let mut adaptive = Vec::new();
    for k in 1..=outputs {
        let to = run * k as f64 / outputs as f64;
        adapt::advance(&model, &tol, &mut t, to, &mut |_| {});
        // Every advance lands on its output time exactly: an ulp short, the rest is an interval of the time's round-off
        // whose doubling reads only rounding (a trajectory once ran into the step cap there).
        assert_eq!(t.time, to, "advanced to {} for the output at {to}", t.time);
        adaptive.push(model.energy(0, &t.point(), &t.psi) - e0);
    }
    let largest = adaptive.iter().map(|e| e.abs()).fold(0.0, f64::max);
    let drift = mean(&adaptive[outputs - outputs / 10..]) - mean(&adaptive[..outputs / 10]);
    eprintln!("adaptive: {} steps, largest |ΔE| {largest:.3e}, drift {drift:.3e}, allotment {:.3e}", t.steps, rate * run);
    assert!(largest <= rate * run, "largest |ΔE| {largest:e} beyond the run's allotment {:e}", rate * run);
    let n = (run / (3.0 * run / t.steps as f64)).round() as usize;
    let tau = run / n as f64;
    let mut f = start;
    let mut fixed = Vec::new();
    for i in 1..=n {
        f = step::step(&model, &f, tau).unwrap().end;
        if i % (n / outputs) == 0 {
            fixed.push(model.energy(0, &f.point(), &f.psi) - e0);
        }
    }
    let k = fixed.len() / 10;
    let (first, last) = (mean(&fixed[..k]), mean(&fixed[fixed.len() - k..]));
    let spread = (fixed[..k].iter().map(|e| (e - first).powi(2)).sum::<f64>() / (k - 1) as f64).sqrt();
    eprintln!("fixed τ {tau:.4}: first tenth {first:.3e} ± {spread:.3e}, last tenth {last:.3e}");
    assert!((last - first).abs() <= 4.0 * spread, "fixed steps drift {:e} against the spread {spread:e}", last - first);
}

/// The step-doubling estimate is the scheme's local error, O(τ³): a two-level atom in a standing wave in a Gaussian
/// tweezer, its internal state relaxed (a bare state's first 1/Γ after a jump is a transient no power of τ describes), the
/// difference of one step and two halves falls eightfold per halving of τ -- between 7 and 10, the next order's room --
/// on the momentum and the position, from Γτ = 15 (the multirate regime: the internal state exact along the path) to 2.
/// Without light the momentum's falls eightfold and the position's sixteenfold (the reference's flow exact to its
/// curvature).
#[test]
fn the_doubling_estimate_falls_as_tau_cubed() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.0, 3.0] };
    let beams = vec![Beam { k: [0.0, 0.0, K], rabi: 6.0, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi: 6.0, phase: 0.3 }];
    for (beams, gamma, orders) in [(beams, GAMMA, [8.0, 8.0]), (vec![], 0.0, [8.0, 16.0])] {
        let model = Atom::new(RB, trap, beams, -40.0, gamma);
        let mut start = held(&model, [0.2, -0.1, 0.4], [30.0, -20.0, 50.0]);
        start.grid = 2.5e-4;
        start = step::step(&model, &start, 0.5).unwrap().end;
        let difference = |tau: f64| -> [f64; 2] {
            let full = step::step(&model, &start, tau).unwrap();
            let first = step::step(&model, &start, 0.5 * tau).unwrap();
            let second = step::step(&model, &first.end, 0.5 * tau).unwrap();
            let dk = (0..3).map(|a| (full.end.kappa[a] - second.end.kappa[a]).abs()).fold(0.0, f64::max);
            let dr = (0..3).map(|a| (full.end.r[a] - second.end.r[a]).abs()).fold(0.0, f64::max);
            [dk, dr]
        };
        let d: Vec<[f64; 2]> = [0.4, 0.2, 0.1, 0.05].iter().map(|&tau| difference(tau)).collect();
        eprintln!("Γ {gamma}: doubling differences (κ, r) at τ 0.4 … 0.05: {d:?}");
        for i in 0..3 {
            for q in 0..2 {
                let ratio = d[i][q] / d[i + 1][q];
                assert!(ratio > 0.875 * orders[q] && ratio < 1.25 * orders[q], "Γ {gamma}: ratio {ratio} of {} at the {i}th halving", ["κ", "r"][q]);
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// The step's own pieces.

/// The Stumpff functions are their series Σ_j (−x)^j/(k + 2j)! on both signs, near zero and far from it, to round-off
/// of their scale.
#[test]
fn the_stumpff_functions_are_their_series() {
    for x in [0.0, 1e-12, -1e-12, 1e-3, -1e-3, 0.7, -0.7, 4.0, -4.0, 30.0, -30.0] {
        let series = |k: usize| -> f64 {
            let (mut term, mut sum) = (1.0 / (1..=k).map(|i| i as f64).product::<f64>(), 0.0);
            for j in 0..200 {
                sum += term;
                term *= -x / (((k + 2 * j + 1) * (k + 2 * j + 2)) as f64);
            }
            sum
        };
        let (c0, c1, c2) = super::stumpff(x);
        for (k, c) in [c0, c1, c2].into_iter().enumerate() {
            let s = series(k);
            assert!((c - s).abs() <= 1e-13 * s.abs().max(1.0), "c{k}({x}) {c} against {s}");
        }
    }
}

/// Gauss–Legendre at its derived panel phase integrates e^{iνs} over [0, T] to round-off (`quadrature`), from a phase of
/// 10⁻³ to 10³ and with a decay, against (e^{iνT} − 1)/(iν).
#[test]
fn the_quadrature_integrates_a_bandwidth_to_round_off() {
    for (omega, decay, tau) in [(1.0, 0.0, 1e-3), (3.0, 0.0, 1.0), (40.0, 0.0, 0.7), (900.0, 0.0, 1.1), (60.0, 19.0, 0.3)] {
        let w = C64::new(omega, decay);
        let sum: C64 = super::quadrature(0.0, tau, w.abs()).map(|(s, weight)| (C64::I * w * s).exp() * weight).sum();
        let exact = ((C64::I * w * tau).exp() - C64::ONE) / (C64::I * w);
        assert!((sum - exact).abs() <= 1e-14 * tau, "ω {omega}, γ {decay}, τ {tau}: {sum:?} against {exact:?} on {} nodes", super::nodes(tau, w.abs()));
    }
}

/// A step's path moments (`Segment::moments`): ∫κ dt and ∫κ² dt over sub-intervals of a long step through a Gaussian
/// tweezer beyond its inflection (a hyperbolic axis in the middle's expansion) -- before the middle, across it, after it --
/// against dense trapezoids of the path's κ (`Segment::at`).
#[test]
fn a_segment_s_moments_are_its_path_s_integrals() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.2, 3.0] };
    let model = Atom::new(RB, trap, vec![], 0.0, 0.0);
    let start = held(&model, [0.3, -0.2, 0.5], [40.0, 25.0, -60.0]);
    let tau = 3.0;
    let step = step::step(&model, &start, tau).unwrap();
    let seg = &step.segment;
    for (from, to) in [(0.2, 1.1), (1.0, 2.6), (1.9, 2.95), (0.0, 3.0)] {
        let (one, two) = seg.moments(from, to);
        let n = 200_000;
        let h = (to - from) / n as f64;
        let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
        for i in 0..=n {
            let w = if i == 0 || i == n { 0.5 * h } else { h };
            let k = seg.at(from + i as f64 * h).1;
            for x in 0..3 {
                a[x] += w * k[x];
                b[x] += w * k[x] * k[x];
            }
        }
        assert!(relative(&one, &a) < 1e-8 && relative(&two, &b) < 1e-8, "[{from}, {to}]: {one:?} vs {a:?}, {two:?} vs {b:?}");
    }
}

/// With light the step is its own inverse up to its path momentum's predictor-corrector, of an order beyond the scheme's:
/// a two-level atom in a far-detuned standing wave (Γ = 0, its state following the light's dressing along the path), one
/// step and its reverse return to the start by far less than the step's doubling error, and the miss falls faster than τ³.
#[test]
fn a_step_with_light_and_its_reverse_miss_beyond_the_scheme_s_order() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.0, 3.0] };
    let beams = vec![Beam { k: [0.0, 0.0, K], rabi: 30.0, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi: 30.0, phase: 0.2 }];
    let model = Atom::new(RB, trap, beams, -300.0, 0.0);
    let mut start = held(&model, [0.1, -0.1, 0.3], [30.0, -20.0, 60.0]);
    start.grid = 1e-3;
    let mut misses = Vec::new();
    for tau in [0.2, 0.1, 0.05] {
        let there = step::step(&model, &start, tau).unwrap();
        let back = step::step(&model, &there.end, -tau).unwrap();
        let first = step::step(&model, &start, 0.5 * tau).unwrap();
        let second = step::step(&model, &first.end, 0.5 * tau).unwrap();
        let miss = (0..3).map(|a| (back.end.kappa[a] - start.kappa[a]).abs()).fold(0.0, f64::max);
        let doubling = (0..3).map(|a| (there.end.kappa[a] - second.end.kappa[a]).abs()).fold(0.0, f64::max);
        eprintln!("τ {tau}: forth and back misses κ by {miss:e}, the doubling differs by {doubling:e}");
        assert!(miss < 0.1 * doubling, "τ {tau}: miss {miss:e} against the doubling's {doubling:e}");
        misses.push(miss);
    }
    assert!(misses[0] / misses[2] > 16.0, "the miss falls as {misses:?}");
}

// ---------------------------------------------------------------------------------------------------------------------
// The model's far light (`model::Run`) on the R1 scenes: a conservative trap.

/// A reference scene (refs/scenes, the app's native request) through the one parser.
fn scene(name: &str) -> crate::request::Request {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    crate::request::Request::from_native(&line).unwrap()
}

/// R1 (contract §6): a conservative far-light trap holds its atoms' energies. The scene's model (every unit folded: the far
/// light alone), its run (`model::Run`) from the reference state, sixteen atoms of the scene's temperature in the trap's
/// harmonic approximation (Gaussian positions and momenta of k_BT along the principal axes), each advanced through the
/// scene's outputs by the step control at the tolerances SPEC §2.2 gives a run (s·ε of the thermal scales over T_run).
/// Between jumps every atom's E_s stays within its run's allotted error -- a far-detuned tweezer still scatters, its
/// folded far light's photons (SPEC §1.2), each a recoil of (ħ/m)κ·k ≈ 0.5 rad/µs at 30 µK, physics the energy is
/// re-read after --, none is lost, and the ensemble's kinetic energy keeps its time-averaged level (T constant): its
/// second half's mean within the first half's statistics.
fn conservative_trap(name: &str) {
    use crate::units::{KT_PER_UK, POPULATION_TARGET, STEP_SHARE, TEMPERATURE_TARGET};
    let request = scene(name);
    let light = crate::field::Light::new(&request);
    let model = crate::model::Model::new(&request, &light, &[]).expect("the far light folds");
    let state = (0..crate::model::GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let mut p = [0.0; crate::model::GROUND];
    p[state] = 1.0;
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let run = crate::model::Run::new(&model, p);
    let hm = run.hbar_over_mass();
    let kt = KT_PER_UK * request.temperature_uk;
    let duration = *request.times_us.last().unwrap();
    let thermal = (kt / hm).sqrt();
    let tol = Tolerances {
        axes: STEP_SHARE * TEMPERATURE_TARGET * kt / duration,
        energy: STEP_SHARE * POPULATION_TARGET * kt / duration,
        population: STEP_SHARE * POPULATION_TARGET / duration,
    };
    let atoms = 16;
    let clock = std::time::Instant::now();
    let (mut kinetic, mut steps, mut worst, mut jumps) = (vec![0.0; request.times_us.len()], 0, 0.0_f64, 0);
    for i in 0..atoms {
        let mut stream = crate::random::Stream::new(17, crate::random::Purpose::Initial, i, 0);
        let (mut r, mut kappa) = (trap.centre, [0.0; 3]);
        for axis in 0..3 {
            let (x, k) = (stream.normal() * (kt * hm).sqrt() / trap.frequencies[axis], stream.normal() * thermal);
            for a in 0..3 {
                r[a] += x * trap.axes[axis][a];
                kappa[a] += k * trap.axes[axis][a];
            }
        }
        let mut t = Trajectory::new(&run, state, r, kappa, 17, i);
        let mut e0 = run.energy(t.cluster, &t.point(), &t.psi);
        for (k, &time) in request.times_us.iter().enumerate().skip(1) {
            let before = t.jumps;
            adapt::advance(&run, &tol, &mut t, time, &mut |_| {});
            assert!(t.lost.is_none(), "{name}: atom {i} lost at {:?}", t.lost);
            let e = run.energy(t.cluster, &t.point(), &t.psi);
            if t.jumps == before {
                worst = worst.max((e - e0).abs() / (tol.energy * duration));
            } else {
                e0 = e;
            }
            kinetic[k] += 0.5 * hm * t.kappa.iter().map(|x| x * x).sum::<f64>() / atoms as f64;
        }
        steps += t.steps;
        jumps += t.jumps;
    }
    let half = kinetic.len() / 2;
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let (first, second) = (mean(&kinetic[1..half]), mean(&kinetic[half..]));
    let spread = (kinetic[1..half].iter().map(|e| (e - first).powi(2)).sum::<f64>() / (half - 2) as f64).sqrt();
    eprintln!(
        "{name}: {atoms} atoms × {duration} µs, {:.2} s, {steps} steps, {jumps} jumps, worst |ΔE_s| between jumps {worst:.3} of the run's allotment; kinetic {first:.4} then {second:.4} (± {spread:.4}), k_BT/2 per axis {:.4}",
        clock.elapsed().as_secs_f64(),
        0.5 * kt
    );
    assert!(worst <= 1.0, "{name}: an atom's E_s moved {worst} of its run's allotment");
    assert!((second - first).abs() <= 4.0 * spread, "{name}: kinetic energy {first} then {second} (spread {spread})");
}

#[test]
fn r1_a_lone_tweezer_keeps_its_atoms_energies() {
    conservative_trap("R1_TWEEZER");
}

#[test]
fn r1_a_retro_reflected_lattice_keeps_its_atoms_energies() {
    conservative_trap("R1_RETRO");
}

/// The step control's guard against a non-asymptotic first doubling (`adapt::level`): R2_RECOIL with its probe's drives
/// from ground 4 into D2 kept (their excited states explicit), the solve's product trajectory 500 of seed 0 at its first
/// output, τ unbounded at entry. (The solve's repro kept the probe's drives into D1 too, 7 THz off: a generator of
/// ‖H‖₁ ≈ 4.5·10⁷ rad/µs whose impulse quadrature takes some 10⁸ nodes per µs -- a stiffness `step::step_work` prices,
/// not a fault of the control.) Before the bound the first doubling over the whole interval read errors of 10³⁰ of the
/// tolerance, the error model asked a step of 10⁻¹³ µs, and there the doubling read the energy's round-off: rejected
/// again and again, the trajectory never reached 2 µs. Now it does, and every accepted step lies above the round-off's
/// scale, the step at which u|E_s| reaches the energy's tolerance.
#[test]
fn a_non_asymptotic_first_doubling_does_not_collapse_the_step() {
    use crate::model::{Component, Model, Run, Unit, GROUND};
    use crate::units::{KT_PER_UK, POPULATION_TARGET, STEP_SHARE, TEMPERATURE_TARGET, UNIT_ROUNDOFF};
    let request = scene("R2_RECOIL");
    let light = crate::field::Light::new(&request);
    let probe = (0..light.frequencies.len()).find(|&f| light.frequencies[f].beams.iter().any(|&l| request.beams[l].id == "probe")).unwrap();
    let near = Component { frequency: probe, counter: false };
    let mut kept = vec![Unit::Far { component: near }];
    kept.extend([16, 18, 22, 28].map(|excited| Unit::Drive { carrier: near, excited, ground: 4 }));
    let model = Model::new(&request, &light, &kept).unwrap();
    let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let mut p = [0.0; GROUND];
    p[state] = 1.0;
    let run = Run::new(&model, p);
    let (kt, duration) = (KT_PER_UK * request.temperature_uk, *request.times_us.last().unwrap());
    let tol = Tolerances {
        axes: STEP_SHARE * TEMPERATURE_TARGET * kt / duration,
        energy: STEP_SHARE * POPULATION_TARGET * kt / duration,
        population: STEP_SHARE * POPULATION_TARGET / duration,
    };
    let r = [0.022570157873194684, 0.03557204701765085, 0.34007333797674333];
    let kappa = [31.75598642152794, -145.69989434161738, -59.92757803634648];
    let mut t = Trajectory::new(&run, 4, r, kappa, 0, 500);
    let energy = run.energy(t.cluster, &t.point(), &t.psi).abs();
    let round_off = UNIT_ROUNDOFF * energy / tol.energy;
    let mut shortest = f64::INFINITY;
    adapt::advance(&run, &tol, &mut t, request.times_us[1], &mut |s| shortest = shortest.min(s.tau.abs()));
    eprintln!("R2 trajectory 500: {} steps, {} jumps, shortest step {shortest:.3e} µs against the round-off's {round_off:.3e}", t.steps, t.jumps);
    assert!(t.lost.is_some() || t.time == request.times_us[1], "stopped at {}", t.time);
    assert!(shortest > round_off, "a step of {shortest:e} µs at the round-off's scale {round_off:e}");
}
