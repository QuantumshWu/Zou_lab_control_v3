//! The integrator against exact and analytic references on a test model of its trait (`Dynamics`): a two-level
//! atom (ground g, excited e of width Γ) in plane or standing waves of one frequency, in a quadratic or Gaussian trap that
//! holds both states alike; or the ground state alone, with or without a decay of its own.
//!
//! The model in the rotating frame, the coupling e ← g of each beam l c_l(r) = (Ω_l/2) e^{i(k_l·r + φ_l)}:
//! H = [[V, Σ c_l*], [Σ c_l, V − Δ − iΓ/2]], G = ∇H_h. Its Doppler frame K_g = 0, K_e = k₀ (the first beam's): the first
//! beam's coupling is static on the path, every other beam's a beat of wave vector k_l − k₀ (and its adjoint). The photon:
//! one class, u_a = √(Γ/3) ψ_e per lab axis (isotropic), landing in g; the jumps' kicks are the emission's −k n alone, of
//! a chosen |k| (zero where a test reads the Ehrenfest force).
//!
//! The tests' models give their generator at a point by its parts (`Parts`): matrices M_j with G_j, each a plane wave
//! along a path, Ĥ(s) = Σ_j M_j e^{iω_j s} + diag(K_s·v), ω_j = Ω_j + q_j·v.

use crate::adapt::{self, Reader};
use crate::linalg::{dot3, C64, Mat};
use crate::model::{Beat, Cluster, Photon, Point, Quadratic, Recoil, STATES};
use crate::step::{self, Dynamics, Middle, Trajectory};

/// One part of a test model's generator at a point: its beat Ω (rad/µs), its wave vector q less the Doppler frames'
/// (1/µm), M and G = ∇M_h per lab axis there.
struct Part {
    omega: f64,
    wave: [f64; 3],
    generator: Mat,
    force: [Mat; 3],
}

/// A test model's generator at a point (`step::Middle`): its parts and its states' Doppler frames K_s.
struct Parts {
    waves: Vec<[f64; 3]>,
    parts: Vec<Part>,
}

impl Parts {
    /// tr(G ρ) of a matrix on the states and a ρ laid out row by row.
    fn trace(&self, g: &Mat, rho: &[C64]) -> C64 {
        let n = self.waves.len();
        (0..n * n).map(|i| g.data[i] * rho[(i % n) * n + i / n]).sum()
    }
}

impl Middle for Parts {
    type Path = [f64; 3];
    fn states(&self) -> usize {
        self.waves.len()
    }
    fn still(&self) -> bool {
        self.parts.iter().all(|p| p.omega == 0.0 && p.wave == [0.0; 3])
    }
    fn path(&self, _: [f64; 3], velocity: [f64; 3]) -> [f64; 3] {
        velocity
    }
    fn gauge(&self, velocity: &[f64; 3], rates: &mut [f64]) {
        for (rate, k) in rates.iter_mut().zip(&self.waves) {
            *rate = dot3(*k, *velocity);
        }
    }
    fn generator(&self, velocity: &[f64; 3], s: f64, h: &mut Mat) {
        h.data.fill(C64::ZERO);
        for p in &self.parts {
            h.add_scaled(C64::cis((p.omega + dot3(p.wave, *velocity)) * s), &p.generator);
        }
        for (x, k) in self.waves.iter().enumerate() {
            h[(x, x)] += C64::real(dot3(*k, *velocity));
        }
    }
    fn force(&self, velocity: &[f64; 3], s: f64, rho: &[C64]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for p in &self.parts {
            let phase = C64::cis((p.omega + dot3(p.wave, *velocity)) * s);
            for a in 0..3 {
                out[a] += (phase * self.trace(&p.force[a], rho)).re;
            }
        }
        out
    }
    fn tones(&self, _: &[f64; 3], rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        for p in &self.parts {
            each(p.omega, std::array::from_fn(|a| self.trace(&p.force[a], rho).re));
        }
    }
    fn work(&self) -> [f64; 3] {
        let size = (self.parts.len() * self.waves.len() * self.waves.len()) as f64;
        [size, 3.0 * size, 3.0 * size]
    }
}

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
    type Middle<'a> = Parts;
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
    fn middle(&self, _: usize, r: [f64; 3], _: f64) -> Parts {
        let (n, v) = (self.n(), self.trap.expansion(r));
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
                let (c, k) = (self.coupling(l, r), self.beams[l].k);
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
                    parts.push(Part { omega: 0.0, wave: q, generator: up, force: gu });
                    parts.push(Part { omega: 0.0, wave: q.map(|x| -x), generator: down, force: gd });
                }
            }
        }
        parts.insert(0, Part { omega: 0.0, wave: [0.0; 3], generator: m, force: g });
        Parts { waves: self.cluster.waves.clone(), parts }
    }
    fn photons(&self, _: usize, _: &Point, psi: &[C64]) -> Vec<Photon> {
        let n = self.n();
        let source = psi[n - 1] * (self.gamma / 3.0).sqrt();
        let mut u = vec![C64::ZERO; n];
        u[0] = source;
        let none = || vec![C64::ZERO; n];
        vec![Photon {
            class: 0,
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
    fn photon_work(&self) -> f64 {
        self.n() as f64
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

/// The middle's fixed point is refused where its map is no contraction (`middle`; the step control then halves τ): an
/// atom that flies into a tweezer of 50 nm waists within a step of several of its periods. The expansion at the free
/// flight's middle, inside the trap, is a harmonic well that the start lies four waists outside of; its flow reaches a
/// point far from that middle, whose own expansion is flat and sends the reach back: the moves do not shrink. Such a
/// step was once taken with a middle that is none (the rounds ended where the move stopped shrinking); a tenth of it,
/// where the map contracts, is taken.
#[test]
fn a_step_whose_middle_does_not_contract_is_refused() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [0.05, 0.05, 0.05] };
    let model = Atom::new(RB, trap, vec![], 0.0, 0.0);
    let start = held(&model, [-0.2, 0.0, 0.0], [60.0, 0.0, 0.0]);
    assert!(matches!(step::step(&model, &start, 9.0), Err(step::Refused::Reference)), "{:?}", step::step(&model, &start, 9.0).map(|s| s.end.r));
    assert!(step::step(&model, &start, 0.9).is_ok());
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

/// e^{−iHτ}ψ of a static generator (the tests' dense reference, `linalg::tests::exp`): the at-rest reference.
fn evolve(h: &Mat, psi: &[C64], tau: f64) -> Vec<C64> {
    crate::linalg::tests::exp(h, C64::new(0.0, -tau)).apply(psi)
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
/// µm²/µs) at the step τ, from `from` to `to` µs, the jumps without recoil (the Ehrenfest force alone, whose mean the
/// unraveling makes the master equation's): per lab axis the mean and its standard error.
fn mean_force(model: &Atom, r: [f64; 3], v: [f64; 3], trajectories: u64, from: f64, to: f64, tau: f64) -> ([f64; 3], [f64; 3]) {
    let mut forces = Vec::new();
    for i in 0..trajectories {
        let mut t = Trajectory::new(model, 0, r, v.map(|x| x / model.hm), 11, i);
        adapt::advance(model, tau, &mut t, None, from, &mut |_| {}).unwrap();
        let start = t.kappa;
        adapt::advance(model, tau, &mut t, None, to, &mut |_| {}).unwrap();
        forces.push(std::array::from_fn::<f64, 3, _>(|a| (t.kappa[a] - start[a]) / (to - from)));
    }
    let n = forces.len() as f64;
    let mean: [f64; 3] = std::array::from_fn(|a| forces.iter().map(|f| f[a]).sum::<f64>() / n);
    let error = std::array::from_fn(|a| (forces.iter().map(|f| (f[a] - mean[a]).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt());
    (mean, error)
}

/// The radiation pressure of a running wave on a moving atom: kΓρ_ee(Δ − k·v) along k (the master equation's steady
/// state), at several velocities through the Doppler profile, within three standard errors and 2 %. The path is static
/// in the atom's frame and the internal state exact along it at any step: steps of 1 µs, some 40/Γ.
#[test]
fn a_running_wave_s_force_is_the_bloch_equations_at_the_doppler_shifted_detuning() {
    let hm = 1e-9;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let (rabi, detuning) = (15.0, -12.0);
    let model = Atom::new(hm, flat, vec![Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }], detuning, GAMMA);
    for v in [-3.0, -1.5, 0.0, 1.5, 3.0] {
        let expected = K * GAMMA * excited(rabi, detuning - K * v);
        let (f, e) = mean_force(&model, [0.0; 3], [0.0, 0.0, v], 48, 2.0, 22.0, 1.0);
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
/// errors and 3 % of the running wave's force, at several velocities. The path beats at 2kv, up to 32 rad/µs here: at
/// steps of 0.02 µs the grid's spacing, a quarter of the step, turns the beat by 0.16 rad at most.
#[test]
fn a_standing_wave_s_force_is_the_bloch_equations_along_the_path() {
    let hm = 1e-9;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let (rabi, detuning) = (10.0, -15.0);
    let beams = vec![Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.0 }];
    let model = Atom::new(hm, flat, beams.clone(), detuning, GAMMA);
    let scale = K * GAMMA * excited(rabi, detuning);
    for v in [0.3, 1.0, 2.0] {
        let expected = bloch_force(&beams, detuning, 0.0, v, 3.0, 33.0);
        let (f, e) = mean_force(&model, [0.0; 3], [0.0, 0.0, v], 48, 3.0, 33.0, 0.02);
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
    adapt::advance(&model, 10.0, &mut t, None, end, &mut |_| {}).unwrap();
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

/// A read trajectory's jumps and hazard (adapt.rs, # The read): a ground state of constant hazard γ in no potential.
/// The hazard's integral is γ t; a constant hazard is integrated exactly by any step, so its local error is none; no
/// force acts between the jumps, so the kinetic energy the jumps put in per lab axis is the trajectory's own gain of it;
/// one internal state has no error of the rows, and its jumps' operator never annihilates the carried state. Each to the
/// round-off of the steps' norms: a norm's ratio and its logarithm are some eight operations a step.
#[test]
fn a_read_trajectory_holds_its_jumps_input_and_its_hazard() {
    let gamma = 2.0;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let model = Atom::new(RB, flat, vec![], 0.0, gamma);
    let start = Trajectory::new(&model, 0, [0.0; 3], [3.0, -2.0, 1.0], 3, 5);
    let (mut t, mut reader, end) = (start.clone(), Reader::new([5.0; 3]), 50.0);
    adapt::advance(&model, 1.0, &mut t, Some(&mut reader), end, &mut |_| {}).unwrap();
    let drift = reader.drift;
    assert!(t.jumps > 50 && drift.unread == 0.0 && drift.rows.iter().all(|x| *x == 0.0), "{} jumps, {drift:?}", t.jumps);
    let round_off = crate::units::rounding(8 * t.steps as usize);
    assert!((drift.mark.hazard - gamma * end).abs() <= round_off * gamma * end, "the hazard's integral {} against {}", drift.mark.hazard, gamma * end);
    assert!(drift.mark.hazard_error.abs() <= round_off * gamma * end, "the hazard's local error {}", drift.mark.hazard_error);
    for a in 0..3 {
        let gain = 0.5 * RB * (t.kappa[a] * t.kappa[a] - start.kappa[a] * start.kappa[a]);
        assert!((drift.mark.input[a] - gain).abs() <= round_off * gain.abs().max(drift.mark.input[a].abs()), "axis {a}: the jumps put in {} against the gain {gain}", drift.mark.input[a]);
    }
}

/// A jump is a linear map of the internal state (`step::jump`'s carried state; adapt.rs, # The read): on a pumping
/// scene's model (R5_PUMP), from every ground state in turn, a carried state that is the trajectory's own takes every
/// jump to the trajectory's own state after it, and so does twice it -- the class's operator found by its place, the
/// polarization the trajectory drew, the same renormalization --, to the bit. Between the jumps a short step moves the
/// state on. The jumps land in more than one cluster: more than one class's operator is met.
#[test]
fn a_carried_state_takes_the_trajectory_s_jumps() {
    let request = scene("R5_PUMP");
    let light = crate::field::Light::new(&request);
    let model = crate::model::Model::new(&request, &light, &[]).expect("every unit folds");
    let chief = (0..crate::model::GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let mut p = [0.0; crate::model::GROUND];
    p[chief] = 1.0;
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let run = crate::model::Run::new(&model, p);
    let (tau, jumps) = (0.25, 3);
    let mut landed = std::collections::BTreeSet::new();
    for state in 0..crate::model::GROUND {
        let mut t = Trajectory::new(&run, state, [trap.centre[0] + 0.05, trap.centre[1] - 0.03, trap.centre[2] + 0.1], [1.0, -2.0, 0.5], 11, state as u64);
        t.grid = tau / 4.0;
        for _ in 0..jumps {
            let (mut own, mut twice) = (Some(t.psi.clone()), Some(t.psi.iter().map(|z| *z * 2.0).collect::<Vec<C64>>()));
            let mut twin = t.clone();
            if step::jump(&run, &mut t, &mut own).is_err() {
                // A state with no photon to emit (the pumping's dark state) makes no jump.
                break;
            }
            step::jump(&run, &mut twin, &mut twice).unwrap();
            assert_eq!(own.as_ref(), Some(&t.psi), "the trajectory's own state, carried, from state {state} after jump {}", t.jumps);
            assert_eq!(twice.as_ref(), Some(&t.psi), "twice the trajectory's state, carried, from state {state} after jump {}", t.jumps);
            landed.insert(t.cluster);
            let next = step::step(&run, &t, tau).unwrap().end;
            (t.r, t.kappa, t.psi, t.time) = (next.r, next.kappa, next.psi, next.time);
        }
    }
    assert!(landed.len() > 1, "the jumps landed in the clusters {landed:?} only");
}

/// R1's mechanism (SPEC §2.2): a conservative anharmonic trap's energy over a long run at one constant step. A hot atom
/// in a Gaussian tweezer (radial ω = 0.13 rad/µs) for some hundred radial periods at ω τ = 0.8: the splitting, symplectic
/// and its own inverse, keeps the energy's error bounded -- its last tenth's mean within its first tenth's spread (no
/// drift) --, and every advance lands on its output time exactly.
#[test]
fn a_conservative_trap_keeps_its_energy() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.0, 3.0] };
    let model = Atom::new(RB, trap, vec![], 0.0, 0.0);
    let mut t = Trajectory::new(&model, 0, [0.3, -0.2, 0.8], [20.0, 15.0, -10.0], 1, 0);
    let e0 = model.energy(0, &t.point(), &t.psi);
    let (run, outputs) = (5000.0, 800);
    let tau = run / outputs as f64;
    let mut errors = Vec::new();
    for k in 1..=outputs {
        let to = run * k as f64 / outputs as f64;
        adapt::advance(&model, tau, &mut t, None, to, &mut |_| {}).unwrap();
        // Every advance lands on its output time exactly: an ulp short, the rest is an interval of the time's round-off.
        assert_eq!(t.time, to, "advanced to {} for the output at {to}", t.time);
        errors.push(model.energy(0, &t.point(), &t.psi) - e0);
    }
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let k = errors.len() / 10;
    let (first, last) = (mean(&errors[..k]), mean(&errors[errors.len() - k..]));
    let spread = (errors[..k].iter().map(|e| (e - first).powi(2)).sum::<f64>() / (k - 1) as f64).sqrt();
    eprintln!("τ {tau}: {} steps, first tenth {first:.3e} ± {spread:.3e}, last tenth {last:.3e}", t.steps);
    assert!((last - first).abs() <= 4.0 * spread, "the energy drifts {:e} against the spread {spread:e}", last - first);
}

/// A read trajectory's loss energy error is the sum of its steps' doubling differences (adapt.rs, # The read): a hot
/// atom in a Gaussian tweezer advanced through four steps with a reader, against the same four steps made by hand -- each
/// the step less its two halves from the same state, 4/3 of the loss energy's difference (second order) --, to the sums'
/// round-off; the read leaves the trajectory where the unread one goes; with one internal state there is no error of
/// the rows, no jump puts energy in, and the hazard is none.
#[test]
fn a_read_trajectory_accumulates_its_steps_doubling_differences() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.0, 3.0] };
    let model = Atom::new(RB, trap, vec![], 0.0, 0.0);
    let start = held(&model, [0.3, -0.2, 0.8], [20.0, 15.0, -10.0]);
    let (tau, steps) = (5.0, 4);
    let (mut read, mut plain, mut reader) = (start.clone(), start.clone(), Reader::new([2.0 * tau; 3]));
    adapt::advance(&model, tau, &mut read, Some(&mut reader), tau * steps as f64, &mut |_| {}).unwrap();
    let drift = reader.drift;
    adapt::advance(&model, tau, &mut plain, None, tau * steps as f64, &mut |_| {}).unwrap();
    assert_eq!((read.r, read.kappa), (plain.r, plain.kappa));
    let (mut by_hand, mut at) = (0.0, start);
    for _ in 0..steps {
        let whole = step::step(&model, &at, tau).unwrap();
        let first = step::step(&model, &at, 0.5 * tau).unwrap();
        let second = step::step(&model, &first.end, 0.5 * tau).unwrap();
        let energy = |t: &Trajectory| model.energy(0, &t.point(), &t.psi);
        by_hand += 4.0 / 3.0 * (energy(&whole.end) - energy(&second.end));
        at = whole.end;
    }
    assert!((drift.energy - by_hand).abs() <= 1e-12 * by_hand.abs(), "{} against {by_hand}", drift.energy);
    assert!(drift.energy.abs() > 1e-6 && drift.rows.iter().all(|x| *x == 0.0) && drift.unread == 0.0, "{drift:?}");
    assert!(drift.mark.input == [0.0; 3], "{drift:?}");
    assert!(drift.mark.level.iter().all(|&e| e > 0.0) && drift.mark.error.iter().any(|&e| e != 0.0), "{drift:?}");
}

/// The step-doubling estimate is the scheme's local error, O(τ³): a two-level atom in a standing wave in a Gaussian
/// tweezer, its internal state relaxed (a bare state's first 1/Γ after a jump is a transient no power of τ describes), the
/// difference of one step and two halves falls eightfold per halving of τ -- its order, the binary logarithm of the
/// fall, rounds to 3 --
/// on the momentum and the position, from Γτ = 15 (the multirate regime: the internal state exact along the path) to 2.
/// Without light the momentum's falls eightfold and the position's sixteenfold, order 4 (the reference's flow exact to its
/// curvature).
#[test]
fn the_doubling_estimate_falls_as_tau_cubed() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.0, 3.0] };
    let beams = vec![Beam { k: [0.0, 0.0, K], rabi: 6.0, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi: 6.0, phase: 0.3 }];
    for (beams, gamma, orders) in [(beams, GAMMA, [3.0, 3.0]), (vec![], 0.0, [3.0, 4.0])] {
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
                assert!((ratio.log2() - orders[q]).abs() < 0.5, "Γ {gamma}: ratio {ratio} of {} at the {i}th halving", ["κ", "r"][q]);
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// The step's own pieces.

/// The Stumpff functions are their series Σ_j (−x)^j/(k + 2j)! on both signs, near zero and far from it, to round-off
/// of their scale.
/// The kick's fluctuation (`fluctuation`) has the covariance M − b bᵀ of the recoil's moments (`Photon::recoil`) and
/// none in a null direction of it. A photon absorbed from a plane wave of wave vector k has ∇f = ik f: M = k kᵀ = b bᵀ,
/// as computed to the rounding of the two sums, and the kick is exactly zero (the root of that rounding is ~1e-8 of the
/// recoil). An envelope's gradient α beside it, ∇f = (ik + α)f, makes C = α αᵀ: one column, the kick along α and of its
/// length whatever the signs -- also where α's first component is its least by far, which a factor taken in the axes'
/// order turns into a spurious second column --, to the entries' rounding γ_N S over the pivot's root. An explicit
/// source's ∇ln σ adds its own directions (SPEC §1.6). Over the signs E[ξ ξᵀ] = C, to the same rounding.
#[test]
fn the_kick_s_fluctuation_has_its_covariance_and_none_in_a_null_direction() {
    let mut stream = crate::random::Stream::new(53, crate::random::Purpose::Initial, 0, 0);
    let (k, n) = ([3.1, -4.2, 5.9], 5);
    let epsilon = [0.6, 0.0, 0.8];
    let amplitudes: [Vec<C64>; 3] = std::array::from_fn(|_| (0..n).map(|_| C64::new(stream.normal(), stream.normal())).collect());
    let source = ([vec![C64::new(0.3, 0.1); n], vec![C64::ZERO; n], vec![C64::new(-0.2, 0.4); n]], [C64::new(0.05, 7.0), C64::new(-0.3, 1.0), C64::new(0.0, -2.0)]);
    for (alpha, explicit) in [([0.0; 3], vec![]), ([0.21, -0.13, 0.08], vec![]), ([1e-5, 0.2, 0.0], vec![]), ([0.21, -0.13, 0.08], vec![source])] {
        let gradients: [[Vec<C64>; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|a| amplitudes[a].iter().map(|u| *u * C64::new(alpha[i], k[i])).collect()));
        let links = 6 * n + explicit.len() + 9;
        let photon = Photon { class: 0, landing: 0, wave_number: 0.0, amplitudes: amplitudes.clone(), gradients, explicit };
        let recoil = photon.recoil(&epsilon);
        let exact: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| recoil.second[i][j] - recoil.mean[i] * recoil.mean[j]));
        let rounded = crate::units::rounding(links) * (0..3).map(|j| recoil.second[j][j] + recoil.mean[j] * recoil.mean[j]).fold(0.0, f64::max);
        let length = alpha.iter().map(|a| a * a).sum::<f64>().sqrt();
        let mut covariance = [[0.0; 3]; 3];
        for pattern in 0..8 {
            let signs: [f64; 3] = std::array::from_fn(|j| if pattern >> j & 1 == 0 { 1.0 } else { -1.0 });
            let xi = super::fluctuation(&recoil.second, &recoil.mean, links, signs);
            if length == 0.0 {
                assert_eq!(xi, [0.0; 3], "a plane wave's absorption fluctuates (the moments' rounding {:e})", exact[0][0]);
            } else if photon.explicit.is_empty() {
                // One column: ±α, each component to the entries' rounding over the pivot's root (at least α's largest
                // component, over √3 of its length).
                let side = if dot3(xi, alpha) > 0.0 { 1.0 } else { -1.0 };
                for i in 0..3 {
                    assert!((xi[i] - side * alpha[i]).abs() <= 2.0 * 3.0_f64.sqrt() * rounded / length, "α {alpha:?}, signs {signs:?}: the kick {xi:?}");
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    covariance[i][j] += xi[i] * xi[j] / 8.0;
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                // The factor's own backward error and the columns left out: the pivots' bound, 18 γ_N S at most.
                assert!((covariance[i][j] - exact[i][j]).abs() <= 2.0 * 18.0 * rounded, "α {alpha:?}: C[{i}][{j}] {} against {}", covariance[i][j], exact[i][j]);
            }
        }
    }
}

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
    // κ on the step's path at the time t: the flow from the start before the middle, from the kicked middle after it.
    let path = |t: f64| if t < 0.5 * tau { seg.flow.advance(seg.from, t).1 } else { seg.flow.advance(seg.kicked, t - 0.5 * tau).1 };
    for (from, to) in [(0.2, 1.1), (1.0, 2.6), (1.9, 2.95), (0.0, 3.0)] {
        let (one, two) = seg.moments(from, to);
        let n = 200_000;
        let h = (to - from) / n as f64;
        let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
        for i in 0..=n {
            let w = if i == 0 || i == n { 0.5 * h } else { h };
            let k = path(from + i as f64 * h);
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
/// step and its reverse return to the start by less than the step's doubling error (by seven orders less here), and
/// over two halvings of τ the miss falls faster than τ³ does.
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
        assert!(miss < doubling, "τ {tau}: miss {miss:e} against the doubling's {doubling:e}");
        misses.push(miss);
    }
    // Over the two halvings τ³ falls by 4³.
    assert!(misses[0] / misses[2] > 64.0, "the miss falls as {misses:?}");
}

// ---------------------------------------------------------------------------------------------------------------------
// The model's far light (`model::Run`) on the R1 scenes: a conservative trap.

/// A scene's request through the one parser: a reference scene by name (refs/scenes, the app's native request), or any
/// native request file by its path.
fn scene(name: &str) -> crate::request::Request {
    let path = if name.ends_with(".json") { std::path::PathBuf::from(name) } else { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json")) };
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    crate::request::Request::from_native(&line).unwrap()
}

/// R1 (contract §6): a conservative far-light trap holds its atoms' energies. The scene's model (every unit folded: the far
/// light alone), its run (`model::Run`) from the reference state, sixteen atoms of the scene's temperature in the trap's
/// harmonic approximation (Gaussian positions and momenta of k_BT along the principal axes), each advanced through the
/// scene's outputs at one constant step, the ladder's second rung (SPEC §2.2: half the largest power-of-two fraction of
/// the output interval below the trap's resonance ω τ = π), every step read (adapt.rs `Drift`). None is lost; between
/// jumps -- a far-detuned tweezer still scatters its folded far light's photons (SPEC §1.2), each a recoil the energy is
/// re-read after -- the atoms' mean E_s moves by less than the step's share of k_BT, s·ε_T; the accumulated read of E_s
/// stays within it too; and the ensemble's kinetic energy keeps its time-averaged level (T constant): its second half's
/// mean within the first half's statistics.
fn conservative_trap(name: &str) {
    use crate::units::{KT_PER_UK, STEP_SHARE, TEMPERATURE_TARGET};
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
    let thermal = (kt / hm).sqrt();
    let fastest = trap.frequencies.iter().copied().fold(0.0, f64::max);
    let mut tau = 0.5 * (request.times_us[1] - request.times_us[0]);
    while fastest * 2.0 * tau >= std::f64::consts::PI {
        tau *= 0.5;
    }
    let atoms = 16;
    let clock = std::time::Instant::now();
    let (mut kinetic, mut steps, mut moved, mut read, mut jumps) = (vec![0.0; request.times_us.len()], 0, 0.0, 0.0, 0);
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
        let mut reader = Reader::new(trap.lab.map(|w| std::f64::consts::TAU / w));
        let mut e0 = run.energy(t.cluster, &t.point(), &t.psi);
        for (k, &time) in request.times_us.iter().enumerate().skip(1) {
            let before = t.jumps;
            adapt::advance(&run, tau, &mut t, Some(&mut reader), time, &mut |_| {}).unwrap();
            assert!(t.lost.is_none(), "{name}: atom {i} lost at {:?}", t.lost);
            let e = run.energy(t.cluster, &t.point(), &t.psi);
            if t.jumps == before {
                moved += (e - e0) / (kt * (atoms as usize * (request.times_us.len() - 1)) as f64);
            } else {
                e0 = e;
            }
            kinetic[k] += 0.5 * hm * t.kappa.iter().map(|x| x * x).sum::<f64>() / atoms as f64;
        }
        read += reader.drift.energy / (kt * atoms as f64);
        steps += t.steps;
        jumps += t.jumps;
    }
    let half = kinetic.len() / 2;
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let (first, second) = (mean(&kinetic[1..half]), mean(&kinetic[half..]));
    let spread = (kinetic[1..half].iter().map(|e| (e - first).powi(2)).sum::<f64>() / (half - 2) as f64).sqrt();
    let share = STEP_SHARE * TEMPERATURE_TARGET;
    eprintln!(
        "{name}: {atoms} atoms × {} µs at τ {tau}, {:.2} s, {steps} step evaluations, {jumps} jumps; E_s moved {moved:+.5} k_BT between jumps on average, read {read:+.5} at the end (share {share:.4}); kinetic {first:.4} then {second:.4} (± {spread:.4}), k_BT/2 per axis {:.4}",
        request.times_us.last().unwrap(),
        clock.elapsed().as_secs_f64(),
        0.5 * kt
    );
    assert!(moved.abs() <= share, "{name}: the atoms' E_s moved {moved} k_BT between jumps");
    assert!(read.abs() <= share, "{name}: the accumulated read of E_s {read} k_BT");
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

// ---------------------------------------------------------------------------------------------------------------------
// The step's cost.

/// A cluster of n ground states at a near-resonant light's sizes in a Gaussian tweezer (the cost harness's):
/// H(r) = V(r) + L(r) − (i/2)Γ(r) + Σ_j [X_j(r) + X_j(r)†], every entry a ← b a plane wave -- L_ab e^{i(K_a − K_b)·r}, static
/// in the cluster's Doppler frames K_s, and X_j,ab e^{i(q_j + K_a − K_b)·r}, on a path a beat of wave vector q_j, its
/// adjoint −q_j's --, G = ∇H_h. V is the tweezer's light shift, on every state alike (the reference); L is Hermitian, Γ
/// positive semidefinite.
struct Synthetic {
    hm: f64,
    trap: Trap,
    cluster: Cluster,
    still: Mat,
    decay: Mat,
    beats: Vec<([f64; 3], Mat)>,
}

impl Synthetic {
    /// n states in a tweezer `depth` deep (rad/µs): light-shift differences and couplings of 1 to 30 rad/µs (log-uniform
    /// moduli, uniform phases), decay rates of 0.01 to 5 /µs (Γ's eigenvalues, log-uniform, on a random orthonormal
    /// basis), the Doppler frames K_s = (s − (n − 1)/2)·`frame`, and a pair of beat parts per wave vector of `waves`,
    /// the pairs sharing the same sizes.
    fn new(n: usize, depth: f64, frame: [f64; 3], waves: &[[f64; 3]]) -> Synthetic {
        let mut s = crate::random::Stream::new(29, crate::random::Purpose::Initial, n as u64, waves.len() as u32);
        let mut entry = |scale: f64| C64::cis(std::f64::consts::TAU * s.uniform()) * (30.0_f64.powf(s.uniform()) * scale);
        let full = Mat::from_fn(n, n, |_, _| entry(1.0));
        let still = Mat::from_fn(n, n, |a, b| if a == b { C64::real(full[(a, a)].re) } else if a < b { full[(a, b)] } else { full[(b, a)].conj() });
        let basis = crate::linalg::hermitian_eigen(&Mat::from_fn(n, n, |_, _| entry(1.0))).vectors;
        let beats = waves.iter().map(|&q| (q, Mat::from_fn(n, n, |_, _| entry(1.0 / waves.len() as f64)))).collect();
        let rates: Vec<C64> = (0..n).map(|_| C64::real(0.01 * 500.0_f64.powf(s.uniform()))).collect();
        let decay = basis.product(&Mat::diagonal(&rates)).product(&basis.adjoint());
        let cluster = Cluster {
            states: (0..n).collect(),
            ground: n,
            frames: vec![Beat(vec![0]); n],
            waves: (0..n).map(|x| frame.map(|k| (x as f64 - 0.5 * (n - 1) as f64) * k)).collect(),
            waves_exact: vec![Beat(vec![]); n],
        };
        Synthetic { hm: RB, trap: Trap::Gaussian { depth, waists: [1.0, 1.0, 3.0] }, cluster, still, decay, beats }
    }
}

impl Dynamics for Synthetic {
    type Middle<'a> = Parts;
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
    fn middle(&self, _: usize, r: [f64; 3], _: f64) -> Parts {
        let (n, v, k) = (self.cluster.states.len(), self.trap.expansion(r), &self.cluster.waves);
        // An entry a ← b of wave vector q: its own q + K_a − K_b, and its phase at the point.
        let wave = |a: usize, b: usize, q: [f64; 3]| -> ([f64; 3], C64) {
            let w: [f64; 3] = std::array::from_fn(|i| q[i] + k[a][i] - k[b][i]);
            (w, C64::cis(w[0] * r[0] + w[1] * r[1] + w[2] * r[2]))
        };
        let (mut m, mut g) = (Mat::zeros(n, n), zeros(n));
        for a in 0..n {
            for b in 0..n {
                let (w, phase) = wave(a, b, [0.0; 3]);
                let h = self.still[(a, b)] * phase;
                m[(a, b)] = h + C64::new(0.0, -0.5) * self.decay[(a, b)] * phase;
                for i in 0..3 {
                    g[i][(a, b)] = C64::new(0.0, w[i]) * h;
                }
            }
            m[(a, a)] += C64::real(v.value);
            for i in 0..3 {
                g[i][(a, a)] += C64::real(v.gradient[i]);
            }
        }
        let mut parts = vec![Part { omega: 0.0, wave: [0.0; 3], generator: m, force: g }];
        for (q, x) in &self.beats {
            let (mut up, mut down, mut gu, mut gd) = (Mat::zeros(n, n), Mat::zeros(n, n), zeros(n), zeros(n));
            for a in 0..n {
                for b in 0..n {
                    let (w, phase) = wave(a, b, *q);
                    let h = x[(a, b)] * phase;
                    up[(a, b)] = h;
                    down[(b, a)] = h.conj();
                    for i in 0..3 {
                        gu[i][(a, b)] = C64::new(0.0, w[i]) * h;
                        gd[i][(b, a)] = (C64::new(0.0, w[i]) * h).conj();
                    }
                }
            }
            parts.push(Part { omega: 0.0, wave: *q, generator: up, force: gu });
            parts.push(Part { omega: 0.0, wave: q.map(|x| -x), generator: down, force: gd });
        }
        Parts { waves: k.clone(), parts }
    }
    fn photons(&self, _: usize, _: &Point, _: &[C64]) -> Vec<Photon> {
        unreachable!("the cost harness takes no jump")
    }
    fn recoil(&self, _: &Photon, _: &[f64; 3]) -> Recoil {
        unreachable!("the cost harness takes no jump")
    }
    fn energy(&self, _: usize, at: &Point, _: &[C64]) -> f64 {
        0.5 * self.hm * at.kappa.iter().map(|k| k * k).sum::<f64>() + self.trap.expansion(at.r).value
    }
    fn lost(&self, _: usize, _: &Point, _: &[C64]) -> bool {
        false
    }
    fn readout(&self, _: usize, _: &Point, psi: &[C64]) -> [f64; STATES] {
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let mut rows = [0.0; STATES];
        for (s, z) in psi.iter().enumerate() {
            rows[s] = z.norm_sqr() / norm;
        }
        rows
    }
    fn assembly_work(&self) -> f64 {
        0.0
    }
    fn photon_work(&self) -> f64 {
        0.0
    }
}

/// A model that counts what a step asks of it: its reference expansions and its middles, and of the middles the
/// generators, the forces and the tones (`Probe`).
struct Counted<'a, D: Dynamics> {
    model: &'a D,
    calls: std::cell::Cell<[u64; 5]>,
}

/// A middle that counts its node calls into its model's counts.
struct Probe<'a, M: Middle> {
    middle: M,
    calls: &'a std::cell::Cell<[u64; 5]>,
}

impl<M: Middle> Probe<'_, M> {
    fn count(&self, which: usize) {
        let mut calls = self.calls.get();
        calls[which] += 1;
        self.calls.set(calls);
    }
}

impl<M: Middle> Middle for Probe<'_, M> {
    type Path = M::Path;
    fn states(&self) -> usize {
        self.middle.states()
    }
    fn still(&self) -> bool {
        self.middle.still()
    }
    fn path(&self, kappa: [f64; 3], velocity: [f64; 3]) -> M::Path {
        self.middle.path(kappa, velocity)
    }
    fn gauge(&self, path: &M::Path, rates: &mut [f64]) {
        self.middle.gauge(path, rates)
    }
    fn generator(&self, path: &M::Path, s: f64, h: &mut Mat) {
        self.count(2);
        self.middle.generator(path, s, h)
    }
    fn force(&self, path: &M::Path, s: f64, rho: &[C64]) -> [f64; 3] {
        self.count(3);
        self.middle.force(path, s, rho)
    }
    fn tones(&self, path: &M::Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        self.count(4);
        self.middle.tones(path, rho, each)
    }
    fn work(&self) -> [f64; 3] {
        self.middle.work()
    }
}

impl<D: Dynamics> Dynamics for Counted<'_, D> {
    type Middle<'b>
        = Probe<'b, D::Middle<'b>>
    where
        Self: 'b;
    fn hbar_over_mass(&self) -> f64 {
        self.model.hbar_over_mass()
    }
    fn reference(&self, r: [f64; 3]) -> Quadratic {
        let mut calls = self.calls.get();
        calls[0] += 1;
        self.calls.set(calls);
        self.model.reference(r)
    }
    fn cluster(&self, cluster: usize) -> &Cluster {
        self.model.cluster(cluster)
    }
    fn cluster_of(&self, ground: usize) -> usize {
        self.model.cluster_of(ground)
    }
    fn middle(&self, cluster: usize, r: [f64; 3], t: f64) -> Self::Middle<'_> {
        let mut calls = self.calls.get();
        calls[1] += 1;
        self.calls.set(calls);
        Probe { middle: self.model.middle(cluster, r, t), calls: &self.calls }
    }
    fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon> {
        self.model.photons(cluster, at, psi)
    }
    fn recoil(&self, photon: &Photon, polarization: &[f64; 3]) -> Recoil {
        self.model.recoil(photon, polarization)
    }
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        self.model.energy(cluster, at, psi)
    }
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.model.lost(cluster, at, psi)
    }
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.model.readout(cluster, at, psi)
    }
    fn assembly_work(&self) -> f64 {
        self.model.assembly_work()
    }
    fn photon_work(&self) -> f64 {
        self.model.photon_work()
    }
}

/// µs per item of each of the `passes` over `items` items, on this thread: every pass timed in batches of as many
/// repetitions as take the clock a millisecond, the batches taken in turn over 24 rounds, the least mean of each -- the
/// machine's other work only adds, and taken in turn it weighs on every pass alike.
fn microseconds<const N: usize>(items: usize, mut passes: [&mut dyn FnMut() -> f64; N]) -> [f64; N] {
    let mut batch = |pass: usize, repetitions: usize| {
        let clock = std::time::Instant::now();
        for _ in 0..repetitions {
            std::hint::black_box(passes[pass]());
        }
        clock.elapsed().as_secs_f64()
    };
    let repetitions: [usize; N] = std::array::from_fn(|pass| {
        let mut repetitions = 1;
        while batch(pass, repetitions) < 1e-3 {
            repetitions *= 2;
        }
        repetitions
    });
    let mut least = [f64::INFINITY; N];
    for _ in 0..24 {
        for pass in 0..N {
            least[pass] = least[pass].min(batch(pass, repetitions[pass]) / repetitions[pass] as f64);
        }
    }
    least.map(|t| 1e6 * t / items as f64)
}

/// The harness's yardstick: 1024 complex multiply-adds (an 8 × 8 matrix on a vector, sixteen times). Timed in turn with a
/// case's passes it tells the machine's pace while they ran -- another core or a lower clock under the machine's other
/// work scales every pass alike, by a factor of two and more from one run to the next --, by which runs compare.
fn yardstick() -> f64 {
    let matrix: [C64; 64] = std::array::from_fn(|k| C64::cis(0.37 * k as f64) * 0.125);
    let mut x: [C64; 8] = std::array::from_fn(|k| C64::cis(1.3 * k as f64));
    for _ in 0..16 {
        let m = std::hint::black_box(&matrix);
        x = std::array::from_fn(|i| m[8 * i..8 * i + 8].iter().zip(&x).map(|(a, b)| *a * *b).sum());
    }
    x[0].re
}

/// One case of the cost harness: µs per step of τ over the case's start states (held: no step reaches their thresholds),
/// split into the model's part -- its reference expansions, its middle, and at the middle's nodes its generators, forces
/// and tones, each timed alone and counted as the step calls them -- and the step's own, the rest. Then the step's
/// phases, the model's calls within them: the middle's fixed point less its reference expansions; the two propagations,
/// each on its path with its kick (the second also with the threshold reached halfway down the step's fall of the norm:
/// its root). Returns the two propagations' times.
fn cost<D: Dynamics>(name: &str, model: &D, starts: &[Trajectory], tau: f64) -> [f64; 2] {
    let (count, half, hm) = (starts.len(), 0.5 * tau, model.hbar_over_mass());
    let counted = Counted { model, calls: std::cell::Cell::new([0; 5]) };
    let steps: Vec<step::Step> = starts.iter().map(|s| step::step(&counted, s, tau).unwrap()).collect();
    let calls = counted.calls.get().map(|c| c as f64 / count as f64);
    // Each start's step, piece by piece as `step::step` makes it.
    struct Pieces<M: Middle> {
        at: ([f64; 3], f64),
        middle: M,
        reference: [f64; 3],
        kappas: [[f64; 3]; 2],
        velocities: [[f64; 3]; 2],
        crossed: Trajectory,
        rho: Vec<C64>,
    }
    let pieces: Vec<Pieces<D::Middle<'_>>> = starts
        .iter()
        .zip(&steps)
        .map(|(from, taken)| {
            let (flow, (r_h, kappa_h)) = super::middle(model, from.r, from.kappa, half).unwrap();
            let reference = flow.gradient_at(r_h);
            let middle = model.middle(from.cluster, r_h, from.time + half);
            let mut scratch = super::Scratch::new(from.psi.len());
            let held = super::shear(&super::Along::new(&middle, kappa_h, [0.0; 3], tau), scratch.carve(0).held, &from.psi, tau);
            let predicted: [f64; 3] = std::array::from_fn(|a| kappa_h[a] + 0.5 * (held[a] + tau * reference[a]));
            let (_, kick) = super::propagate(&mut scratch, &middle, from, reference, kappa_h, predicted.map(|k| hm * k), tau, None);
            let corrected: [f64; 3] = std::array::from_fn(|a| kappa_h[a] + 0.5 * kick[a]);
            let norm = |psi: &[C64]| psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
            let mut crossed = from.clone();
            crossed.threshold = (norm(&from.psi) * norm(&taken.end.psi)).sqrt();
            let mut rho = vec![C64::ZERO; from.psi.len() * from.psi.len()];
            super::density(&from.psi, &mut rho);
            Pieces { at: (r_h, from.time + half), middle, reference, kappas: [kappa_h, corrected], velocities: [predicted.map(|k| hm * k), corrected.map(|k| hm * k)], crossed, rho }
        })
        .collect();
    let propagation = |which: usize, crossing: bool| {
        pieces
            .iter()
            .zip(starts)
            .map(|(p, s)| {
                let (from, threshold) = if crossing { (&p.crossed, Some(p.crossed.threshold)) } else { (s, (which == 1).then_some(s.threshold)) };
                super::propagate(&mut super::Scratch::new(from.psi.len()), &p.middle, from, p.reference, p.kappas[which], p.velocities[which], tau, threshold).1[0]
            })
            .sum::<f64>()
    };
    // The corrector's path of every piece, and a node a fifth of the step off the middle.
    let paths: Vec<_> = pieces.iter().map(|p| p.middle.path(p.kappas[1], p.velocities[1])).collect();
    let n = starts[0].psi.len();
    let mut h = Mat::zeros(n, n);
    let [whole, reference, point, generator, force, tones, middle, first, second, root, pace] = microseconds(
        count,
        [
            &mut || starts.iter().map(|s| step::step(model, s, tau).unwrap().end.kappa[0]).sum::<f64>(),
            &mut || pieces.iter().map(|p| model.reference(p.at.0).value).sum::<f64>(),
            &mut || pieces.iter().zip(starts).map(|(p, s)| model.middle(s.cluster, p.at.0, p.at.1).states() as f64).sum::<f64>(),
            &mut || {
                pieces.iter().zip(&paths).for_each(|(p, path)| p.middle.generator(path, 0.2 * tau, &mut h));
                h.data[0].re
            },
            &mut || pieces.iter().zip(&paths).map(|(p, path)| p.middle.force(path, 0.2 * tau, &p.rho)[0]).sum::<f64>(),
            &mut || {
                let mut sum = 0.0;
                pieces.iter().zip(&paths).for_each(|(p, path)| p.middle.tones(path, &p.rho, &mut |_, f| sum += f[0]));
                sum
            },
            &mut || starts.iter().map(|s| super::middle(model, s.r, s.kappa, half).unwrap().1 .0[0]).sum::<f64>(),
            &mut || propagation(0, false),
            &mut || propagation(1, false),
            &mut || propagation(1, true),
            &mut yardstick,
        ],
    );
    let crossings = pieces
        .iter()
        .filter(|p| super::propagate(&mut super::Scratch::new(p.crossed.psi.len()), &p.middle, &p.crossed, p.reference, p.kappas[1], p.velocities[1], tau, Some(p.crossed.threshold)).0.crossing.is_some())
        .count();
    let nodes = calls[2] * generator + calls[3] * force + calls[4] * tones;
    let own = whole - calls[0] * reference - calls[1] * point - nodes;
    let fixed = middle - calls[0] * reference;
    let work = steps.iter().map(|s| s.work).sum::<f64>() / count as f64;
    eprintln!(
        "{name:<22} n {n} τ {tau:<6} | step {whole:8.2} = model {:8.2} (reference {reference:5.2} × {:4.1} + middle {point:7.2} × {:.0} + generator {generator:6.2} × {:4.1} + force {force:6.2} × {:4.1} + tones {tones:6.2} × {:3.1}) + own {own:8.2} | fixed point {fixed:6.2}  first {first:8.2}  second {second:8.2}  (root {root:8.2}, {crossings}/{count}) | counted {work:9.0} | pace {:.3} ns",
        whole - own,
        calls[0],
        calls[1],
        calls[2],
        calls[3],
        calls[4],
        pace * count as f64 / 1.024
    );
    [first, second]
}

/// Atoms of a cluster's cost case: `count` atoms at the tweezer's thermal sizes (positions of 0.1 µm, speeds of
/// 0.05 µm/µs in uniform directions), each a random state of the cluster, held.
fn atoms<D: Dynamics>(model: &D, count: u64, grid: f64) -> Vec<Trajectory> {
    (0..count)
        .map(|i| {
            let mut s = crate::random::Stream::new(31, crate::random::Purpose::Initial, i, 0);
            let r: [f64; 3] = std::array::from_fn(|_| 0.1 * s.normal());
            let kappa = s.direction().map(|x| 0.05 * x / model.hbar_over_mass());
            let mut t = Trajectory::new(model, 0, r, kappa, 7, i);
            let psi: Vec<C64> = t.psi.iter().map(|_| C64::new(s.normal(), s.normal())).collect();
            let norm = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            t.psi = psi.iter().map(|z| *z / norm).collect();
            t.threshold = f64::MIN_POSITIVE;
            t.grid = grid;
            t
        })
        .collect()
}

/// An R1 scene's cost case: the scene's far-light model (`model::Run`) from its reference state, sixteen atoms of the
/// scene's temperature in the trap's harmonic approximation (as `conservative_trap`), held, one step of τ each on a
/// grid of `panels` panels.
fn scene_cost(name: &str, tau: f64, panels: usize) -> [f64; 2] {
    let mut times = [0.0; 2];
    scene_atoms(name, tau / (2 * panels) as f64, &mut |run, starts| times = cost(&format!("{name}/{panels}"), run, starts, tau));
    times
}

/// An R1 scene's far-light model and its atoms on the grid's spacing `grid`, to `each`.
fn scene_atoms(name: &str, grid: f64, each: &mut dyn FnMut(&crate::model::Run, &[Trajectory])) {
    let request = scene(name);
    let light = crate::field::Light::new(&request);
    let model = crate::model::Model::new(&request, &light, &[]).expect("the far light folds");
    let state = (0..crate::model::GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let mut p = [0.0; crate::model::GROUND];
    p[state] = 1.0;
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let run = crate::model::Run::new(&model, p);
    let kt = crate::units::KT_PER_UK * request.temperature_uk;
    let starts: Vec<Trajectory> = (0..16)
        .map(|i| {
            let mut stream = crate::random::Stream::new(17, crate::random::Purpose::Initial, i, 0);
            let (r, kappa) = thermal(&trap, run.hbar_over_mass(), kt, &mut stream);
            let mut t = Trajectory::new(&run, state, r, kappa, 17, i);
            t.threshold = f64::MIN_POSITIVE;
            t.grid = grid;
            t
        })
        .collect();
    each(&run, &starts)
}

/// An atom's place and momentum at k_BT in a trap's harmonic approximation: Gaussian along its principal axes.
fn thermal(trap: &crate::model::Trap, hm: f64, kt: f64, stream: &mut crate::random::Stream) -> ([f64; 3], [f64; 3]) {
    let (mut r, mut kappa) = (trap.centre, [0.0; 3]);
    for axis in 0..3 {
        let (x, k) = (stream.normal() * (kt * hm).sqrt() / trap.frequencies[axis], stream.normal() * (kt / hm).sqrt());
        for a in 0..3 {
            r[a] += x * trap.axes[axis][a];
            kappa[a] += k * trap.axes[axis][a];
        }
    }
    (r, kappa)
}

/// A scene's step on its own model (run by name, alone: `SCENE=<a reference scene's name or a native request's path>
/// TAU=<µs> … a_scene_s_step_cost -- --ignored --nocapture`): the scene's model as the solve closes it (solve.rs
/// `attempt`: every unit whose fold does not exist over the reach kept, until none fails) with its near tables
/// (`Model::tabulate`), and per cluster a cost case (`cost`): sixteen atoms of the scene's temperature in the trap's
/// harmonic approximation, each a random state of the cluster, held, one step of τ each on the control's grid of two
/// panels (adapt.rs).
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn a_scene_s_step_cost() {
    use crate::model::{Model, Run, Unit, GROUND};
    let name = std::env::var("SCENE").expect("SCENE: a reference scene's name or a request file's path");
    let tau: f64 = std::env::var("TAU").expect("TAU: the step (µs)").parse().expect("TAU: a number");
    let request = scene(&name);
    let light = crate::field::Light::new(&request);
    let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let mut p = [0.0; GROUND];
    p[state] = 1.0;
    let started = std::time::Instant::now();
    let mut kept: Vec<Unit> = vec![];
    let (model, trap, reach) = loop {
        let model = Model::new(&request, &light, &kept).unwrap();
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, request.temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        let units = model.units();
        let failing: Vec<Unit> = units.iter().zip(model.gates(&units, &reach)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            break (model, trap, reach);
        }
        kept.extend(failing);
    };
    let closure = started.elapsed();
    let started = std::time::Instant::now();
    let tables = model.tabulate(&reach, 1 << 20);
    eprintln!("{name}: closure {closure:.2?}, {} units kept, tabulate {:.2?}, {} tables", kept.len(), started.elapsed(), tables.tables.len());
    let run = Run::tabulated(&model, &tables, p);
    let kt = crate::units::KT_PER_UK * request.temperature_uk;
    let panels = 2;
    for (cluster, c) in model.clusters().iter().enumerate() {
        let starts: Vec<Trajectory> = (0..16)
            .map(|i| {
                let mut stream = crate::random::Stream::new(17, crate::random::Purpose::Initial, i, cluster as u32);
                let (r, kappa) = thermal(&trap, run.hbar_over_mass(), kt, &mut stream);
                let mut t = Trajectory::new(&run, c.states[0], r, kappa, 17, i);
                let psi: Vec<C64> = t.psi.iter().map(|_| C64::new(stream.normal(), stream.normal())).collect();
                let norm = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                t.psi = psi.iter().map(|z| *z / norm).collect();
                t.threshold = f64::MIN_POSITIVE;
                t.grid = tau / (2 * panels) as f64;
                t
            })
            .collect();
        cost(&format!("{}+{} of {}", c.ground, c.states.len() - c.ground, std::path::Path::new(&name).file_stem().unwrap().to_string_lossy()), &run, &starts, tau);
    }
}


/// What a propagation on a grid of one panel spends where (µs, a start's): the model's generator at the spacings' nodes
/// and their combinations into the exponentials' matrices, the exponentials on the vector (their products of n²), and
/// the rest -- the kick's densities and the model's forms of them. On the predictor's path at the first start.
fn split<D: Dynamics>(name: &str, model: &D, starts: &[Trajectory], tau: f64) {
    let from = &starts[0];
    let n = from.psi.len();
    let hm = model.hbar_over_mass();
    let (_, (r_h, kappa_h)) = super::middle(model, from.r, from.kappa, 0.5 * tau).unwrap();
    let middle = model.middle(from.cluster, r_h, from.time + 0.5 * tau);
    let path = super::Along::new(&middle, kappa_h, kappa_h.map(|k| hm * k), tau);
    let h = 0.5 * tau;
    let (mut whole, mut combining, mut acting) = (super::Scratch::new(n), super::Scratch::new(n), super::Scratch::new(n));
    // One spacing's two generators and their two combinations as `cf4` makes them, the last one kept for the exponentials.
    let combine = |scratch: &mut super::Scratch, s: f64| {
        let c = scratch.carve(2);
        let r = 3.0_f64.sqrt() / 6.0;
        path.middle.generator(&path.path, s + (0.5 - r) * h, &mut c.generators[0]);
        path.middle.generator(&path.path, s + (0.5 + r) * h, &mut c.generators[1]);
        let mut sum = 0.0;
        for [w1, w2] in [[0.25 + r, 0.25 - r], [0.25 - r, 0.25 + r]] {
            for (z, (one, two)) in c.matrix.data.iter_mut().zip(c.generators[0].data.iter().zip(&c.generators[1].data)) {
                *z = *one * w1 + *two * w2;
            }
            sum += c.matrix.data[0].re;
        }
        sum
    };
    combine(&mut acting, -h);
    let kept = acting.matrix.clone();
    let mut state: Vec<C64> = from.psi.clone();
    let mut plan = crate::linalg::Chebyshev { parts: 0, terms: 0 };
    let [propagation, combinations, exponentials, pace] = microseconds(
        1,
        [
            &mut || super::beating(&path, &mut whole, &from.psi, tau, 1, None).impulse[0],
            &mut || combine(&mut combining, -h) + combine(&mut combining, 0.0),
            &mut || {
                for _ in 0..4 {
                    state.copy_from_slice(&from.psi);
                    acting.matrix.data.copy_from_slice(&kept.data);
                    let c = acting.carve(2);
                    plan = crate::linalg::exp_apply(c.matrix, C64::new(0.0, -h), &mut state, c.work);
                }
                state[0].re
            },
            &mut yardstick,
        ],
    );
    eprintln!(
        "{name:<22} n {n:>2} | a propagation {propagation:8.2} = generators and combinations {combinations:8.2} + exponentials {exponentials:8.2} ({} × {} products each) + kick and rest {:8.2} | pace {:.3} ns",
        plan.parts,
        plan.terms,
        propagation - combinations - exponentials,
        pace / 1.024
    );
}

/// The step's cost (run by name, alone: `the_step_s_cost -- --ignored --nocapture`): µs per `step::step` and per phase,
/// single-threaded, on the R1 scenes' far light (a static path and a standing wave's beating one), on a two-level atom
/// with its excited state explicit in a standing wave in a tweezer, and on clusters of 1, 3, 5 and 8 ground states at a
/// near-resonant light's sizes (`Synthetic`) on a static path (Doppler frames), on a standing wave's beating one (a pair
/// of beat parts) and on six beams' (nine pairs). The beating paths on grids of one panel and of four: the difference
/// over their six CF4 spacings is a spacing's cost. Then clusters of 5 to 32 states with 19 and 201 parts (dense), each
/// with what a propagation spends where (`split`).
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn the_step_s_cost() {
    let spacing = |name: &str, one: [f64; 2], four: [f64; 2]| {
        eprintln!("{name:<22} a CF4 spacing: first {:7.2}  second {:7.2} µs", (four[0] - one[0]) / 6.0, (four[1] - one[1]) / 6.0)
    };
    scene_cost("R1_TWEEZER", 2.5, 1);
    spacing("R1_RETRO", scene_cost("R1_RETRO", 0.3125, 1), scene_cost("R1_RETRO", 0.3125, 4));
    let tau = 0.5;
    let grids = [(1, f64::INFINITY), (4, tau / 8.0)];
    let standing = vec![Beam { k: [0.0, 0.0, K], rabi: 10.0, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi: 10.0, phase: 0.3 }];
    let two = Atom::new(RB, Trap::Gaussian { depth: 1000.0, waists: [1.0, 1.0, 3.0] }, standing, -0.5 * GAMMA, GAMMA);
    let [one, four] = grids.map(|(panels, grid)| cost(&format!("two-level/{panels}"), &two, &atoms(&two, 8, grid), tau));
    spacing("two-level", one, four);
    let six = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0], [0.0, 1.0, -1.0]]
        .map(|q| q.map(|x| x * K));
    for (n, depth) in [(1, 300.0), (3, 300.0), (5, 1000.0), (8, 3000.0)] {
        let still = Synthetic::new(n, depth, [0.0, 0.6 * K, 0.8 * K], &[]);
        cost(&format!("static {n}"), &still, &atoms(&still, 8, f64::INFINITY), tau);
        for (label, waves) in [("standing", &six[2..3]), ("six beams", &six[..])] {
            let model = Synthetic::new(n, depth, [0.0; 3], waves);
            let [one, four] = grids.map(|(panels, grid)| cost(&format!("{label} {n}/{panels}"), &model, &atoms(&model, 8, grid), tau));
            spacing(&format!("{label} {n}"), one, four);
        }
    }
    // Larger clusters and many parts: 9 pairs of beats (19 parts) and 100 pairs (201 parts), one panel.
    let mut directions = crate::random::Stream::new(37, crate::random::Purpose::Initial, 0, 0);
    let hundred: Vec<[f64; 3]> = (0..100).map(|_| directions.direction().map(|x| 2.0 * K * x)).collect();
    for n in [5, 8, 16, 32] {
        for waves in [&six[..], &hundred[..]] {
            let model = Synthetic::new(n, 1000.0, [0.0; 3], waves);
            let starts = atoms(&model, 4, f64::INFINITY);
            cost(&format!("{} parts {n}/1", 1 + 2 * waves.len()), &model, &starts, tau);
            split(&format!("{} parts {n}/1", 1 + 2 * waves.len()), &model, &starts, tau);
        }
    }
}
