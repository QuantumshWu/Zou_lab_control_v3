//! The integrator against exact and analytic references on a test model of its trait (`Dynamics`): a two-level
//! atom (ground g, excited e of width Γ) in plane or standing waves of one frequency, in a quadratic or Gaussian trap that
//! holds both states alike; or the ground state alone, with or without a decay of its own.
//!
//! The model in the rotating frame, the coupling e ← g of each beam l c_l(r) = (Ω_l/2) e^{i(k_l·r + φ_l)}:
//! H = [[V, Σ c_l*], [Σ c_l, V − Δ − iΓ/2]], G = ∇H_h. Its Doppler frame K_g = 0, K_e = k₀ (the first beam's): the first
//! beam's coupling is static on the path, every other beam's a beat of wave vector k_l − k₀ (and its adjoint). The photon:
//! one class, u_a = √(Γ/3) ψ_e per lab axis (isotropic), landing in g; the jumps' kicks are the emission's −k n alone, of
//! a chosen |k| (zero where a test reads the Ehrenfest force). The ground state alone may scatter a folded light
//! (`Atom::folded`): one class of the jump amplitude f(r) = Σ_l c_l e^{ik_l·r} on it, u_a = f ψ/√3 per lab axis with its
//! gradient, its decay |f(r)|² -- the absorbed light's momentum by the engine's own rule (`Photon::moments`).
//!
//! The tests' models give their generator at a point by its parts (`Parts`): matrices M_j with G_j, each a plane wave
//! along a path, Ĥ(s) = Σ_j M_j e^{iω_j s} + diag(K_s·v), ω_j = Ω_j + q_j·v.

use crate::adapt;
use crate::linalg::{dot3, C64, Mat};
use crate::model::{Beat, Cluster, Photon, Point, Quadratic, Scattering, STATES};
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
    /// The folded light the ground state alone scatters: per plane wave its amplitude c_l (1/√µs) and wave vector.
    folded: Vec<(C64, [f64; 3])>,
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
        Atom { hm, trap, beams, detuning, gamma, emission: 0.0, region: f64::INFINITY, folded: vec![], cluster }
    }
    /// The folded light's jump amplitude f = Σ_l c_l e^{ik_l·r} at a point and its gradient.
    fn field(&self, r: [f64; 3]) -> (C64, [C64; 3]) {
        let (mut f, mut df) = (C64::ZERO, [C64::ZERO; 3]);
        for (c, k) in &self.folded {
            let term = *c * C64::cis(dot3(*k, r));
            f += term;
            for a in 0..3 {
                df[a] += term * C64::new(0.0, k[a]);
            }
        }
        (f, df)
    }
    /// The folded light's one class of photons of the ground state's amplitude, with its gradients.
    fn made(&self, r: [f64; 3], psi: &[C64]) -> Vec<Photon> {
        let ((f, df), third) = (self.field(r), (1.0_f64 / 3.0).sqrt());
        let u = vec![f * psi[0] * third];
        let gradients = std::array::from_fn(|i| std::array::from_fn(|_| vec![df[i] * psi[0] * third]));
        let absorbed = self.folded.iter().map(|(_, k)| crate::linalg::norm3(*k)).fold(0.0, f64::max);
        vec![Photon { class: 0, landing: 0, wave_number: self.emission, absorbed, amplitudes: [u.clone(), u.clone(), u], gradients, explicit: [[0.0; 3]; 3] }]
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
            m[(0, 0)] += C64::new(0.0, -0.5 * (self.gamma + self.field(r).0.norm_sqr()));
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
    fn photons(&self, _: usize, at: &Point, psi: &[C64]) -> Vec<Photon> {
        if !self.folded.is_empty() {
            return self.made(at.r, psi);
        }
        let n = self.n();
        let source = psi[n - 1] * (self.gamma / 3.0).sqrt();
        let mut u = vec![C64::ZERO; n];
        u[0] = source;
        let none = || vec![C64::ZERO; n];
        vec![Photon {
            class: 0,
            landing: 0,
            wave_number: self.emission,
            absorbed: 0.0,
            amplitudes: [u.clone(), u.clone(), u],
            gradients: std::array::from_fn(|_| [none(), none(), none()]),
            explicit: [[0.0; 3]; 3],
        }]
    }
    fn kick(&self, _: usize, at: &Point, psi: &[C64], _: &Photon) -> [f64; 3] {
        if self.folded.is_empty() { [0.0; 3] } else { self.made(at.r, psi)[0].kick() }
    }
    fn scattering(&self, _: usize, at: &Point, psi: &[C64], _: &[f64]) -> Scattering {
        if self.folded.is_empty() { Scattering::NONE } else { Scattering::of(&self.made(at.r, psi), psi.iter().map(|z| z.norm_sqr()).sum()) }
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
    let s = step::step(&model, &held(&model, r, kappa), tau, &mut step::Scratch::default()).unwrap();
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
    assert!(matches!(step::step(&model, &start, 9.0, &mut step::Scratch::default()), Err(step::Refused::Reference)), "{:?}", step::step(&model, &start, 9.0, &mut step::Scratch::default()).map(|s| s.end.r));
    assert!(step::step(&model, &start, 0.9, &mut step::Scratch::default()).is_ok());
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
        let there = step::step(&model, &start, tau, &mut step::Scratch::default()).unwrap();
        let back = step::step(&model, &there.end, -tau, &mut step::Scratch::default()).unwrap();
        let (dr, dk) = (relative(&back.end.r, &r), relative(&back.end.kappa, &kappa));
        assert!(dr < 1e-13 && dk < 1e-13, "τ {tau}: r {:?} ({dr:e}), κ {:?} ({dk:e})", back.end.r, back.end.kappa);
    }
}

/// e^{−iHτ}ψ of a static generator (the tests' dense reference, `linalg::tests::exp`): the at-rest reference.
fn evolve(h: &Mat, psi: &[C64], tau: f64) -> Vec<C64> {
    crate::linalg::tests::exp(h, C64::new(0.0, -tau)).apply(psi)
}

/// Galilei covariance (propagation.md §1.2): an atom moving uniformly through a running wave is an atom at rest at the
/// detuning less k·v, v the step's path velocity -- predicted from the start state, which is the atom's own velocity
/// here: the bare ground state in a flat trap is held by no force. One step of the moving atom's internal state against
/// e^{−iH′τ} at rest: populations and the norm, gauge-invariant, to round-off.
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
        let s = step::step(&model, &held(&model, [0.1, 0.2, 0.3], kappa), tau, &mut step::Scratch::default()).unwrap();
        // The path's velocity: the start's (flat trap: κ_h = κ₀; the ground state's held kick is none).
        let kv = dot3(beam.k, v);
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

/// The same in a standing wave: the moving atom sees the two beams at Δ ∓ k·v, a beat at 2k·v that steps of 1/250 of
/// its turn resolve; against RK4 of the at-rest atom's beating generator at a 10⁻⁵ µs step.
#[test]
fn a_moving_atom_in_a_standing_wave_is_an_atom_at_rest_with_two_detunings() {
    let hm = 1e-9;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let (rabi, detuning) = (12.0, -15.0);
    let beams = vec![Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.7 }];
    let model = Atom::new(hm, flat, beams.clone(), detuning, GAMMA);
    let (v, z0, tau) = (0.8, 0.13, 0.5);
    let (mut t, steps) = (held(&model, [0.0, 0.0, z0], [0.0, 0.0, v / hm]), 250);
    for _ in 0..steps {
        t = step::step(&model, &t, tau / steps as f64, &mut step::Scratch::default()).unwrap().end;
    }
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
    let moving: Vec<f64> = t.psi.iter().map(|z| z.norm_sqr()).collect();
    let at_rest: Vec<f64> = y.iter().map(|z| z.norm_sqr()).collect();
    assert!(relative(&moving, &at_rest) < 1e-6, "{moving:?} vs {at_rest:?}");
}

/// The light of beams along z at a height: the coupling c(z) = Σ_l (Ω_l/2) e^{i(k_l z + φ_l)} and its gradient ∂_z c.
fn light(beams: &[Beam], z: f64) -> (C64, C64) {
    let (mut c, mut g) = (C64::ZERO, C64::ZERO);
    for b in beams {
        let term = C64::cis(b.k[2] * z + b.phase) * (0.5 * b.rabi);
        c += term;
        g += term * C64::new(0.0, b.k[2]);
    }
    (c, g)
}

/// The force along z on a two-level atom's state (a, e) in that light, −⟨ψ|∂_zH|ψ⟩/⟨ψ|ψ⟩ = −2 Re(e* ∂_z c a)/‖ψ‖².
fn pull(beams: &[Beam], z: f64, a: C64, e: C64) -> f64 {
    -2.0 * (e.conj() * light(beams, z).1 * a).re / (a.norm_sqr() + e.norm_sqr())
}

/// A two-level atom on the uniform path z₀ + vt through beams along z, sub-stepped: RK4 in `steps` steps of the lab's
/// iψ̇ = H(z(t))ψ, H = [[0, c*], [c, −Δ − iΓ/2]], with the impulse I = −∫⟨ψ|∂_zH|ψ⟩/⟨ψ|ψ⟩dt as a third unknown -- less,
/// where a state is `held` (the state and its time), that state's force held in the Doppler frame along the path, its
/// excited amplitude turned by e^{ik₀v(t − t_m)}: what the state's change along the path adds to a step's kick (step.rs,
/// # The kick). ψ(τ) and I: the dense reference of a step's state and kick.
fn substepped(beams: &[Beam], detuning: f64, gamma: f64, z0: f64, v: f64, psi: [C64; 2], tau: f64, steps: usize, held: Option<([C64; 2], f64)>) -> ([C64; 2], f64) {
    let derivative = |time: f64, y: &[C64; 3]| -> [C64; 3] {
        let z = z0 + v * time;
        let c = light(beams, z).0;
        let (a, e) = (y[0], y[1]);
        let (da, de) = (c.conj() * e, c * a + C64::new(-detuning, -0.5 * gamma) * e);
        let kept = held.map_or(0.0, |(x, at)| pull(beams, z, x[0], x[1] * C64::cis(beams[0].k[2] * v * (time - at))));
        [C64::new(da.im, -da.re), C64::new(de.im, -de.re), C64::real(pull(beams, z, a, e) - kept)]
    };
    let dt = tau / steps as f64;
    let add = |a: &[C64; 3], b: &[C64; 3], s: f64| -> [C64; 3] { std::array::from_fn(|k| a[k] + b[k] * s) };
    let mut y = [psi[0], psi[1], C64::ZERO];
    for i in 0..steps {
        let t0 = i as f64 * dt;
        let k1 = derivative(t0, &y);
        let k2 = derivative(t0 + 0.5 * dt, &add(&y, &k1, 0.5 * dt));
        let k3 = derivative(t0 + 0.5 * dt, &add(&y, &k2, 0.5 * dt));
        let k4 = derivative(t0 + dt, &add(&y, &k3, dt));
        y = std::array::from_fn(|k| y[k] + (k1[k] + k2[k] * 2.0 + k3[k] * 2.0 + k4[k]) * (dt / 6.0));
    }
    ([y[0], y[1]], y[2].re)
}

/// The same atom's step as the step defines it (step.rs, # The kick), sub-stepped in an even number of `steps`: ψ(τ),
/// and the kick in its decomposition about the state at the step's middle x_m -- that state's force at the fixed middle
/// point over the step (the shear: these beams do not beat in time), and what the state's change along the path adds.
/// A step's propagation errs against this by what it leaves of the path's change alone; against the path's own kick
/// (`substepped`) the step differs besides by its splitting, the shear's O(τ³).
fn decomposed(beams: &[Beam], detuning: f64, gamma: f64, z0: f64, v: f64, psi: [C64; 2], tau: f64, steps: usize) -> ([C64; 2], f64) {
    let (middle, _) = substepped(beams, detuning, gamma, z0, v, psi, 0.5 * tau, steps / 2, None);
    let (end, change) = substepped(beams, detuning, gamma, z0, v, psi, tau, steps, Some((middle, 0.5 * tau)));
    (end, change + tau * pull(beams, z0 + 0.5 * v * tau, middle[0], middle[1]))
}

/// One step of a two-level atom from the bare ground state at z₀, at the velocity v along z, against its sub-stepped
/// reference (`substepped`, 2000 sub-steps a radian of Δτ and no fewer than 20 000): the largest error of ψ's amplitudes
/// (lab), and the kick's error, relative to the larger of the reference's kick and of the kick of the light's dressed
/// state held over the step (its dipole force's: the bare state's kick may pass through zero). The mass is one the
/// kick does not move (ħ/m = 10⁻⁹ µm²/µs), so the path is the reference's.
fn step_errors(beams: &[Beam], detuning: f64, gamma: f64, z0: f64, v: f64, tau: f64) -> [f64; 2] {
    let hm = 1e-9;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let model = Atom::new(hm, flat, beams.to_vec(), detuning, gamma);
    let start = held(&model, [0.0, 0.0, z0], [0.0, 0.0, v / hm]);
    let s = step::step(&model, &start, tau, &mut step::Scratch::default()).unwrap();
    let steps = ((2000.0 * detuning.abs() * tau) as usize).max(20_000);
    let (psi, kick) = substepped(beams, detuning, gamma, z0, v, [C64::ONE, C64::ZERO], tau, steps, None);
    let state = (0..2).map(|a| (s.end.psi[a] - psi[a]).abs()).fold(0.0, f64::max);
    // The dressed state's dipole kick at the step's middle: −τ ∂_z(|c|²/Δ)-like, its size k|c|²τ/|Δ̃| at most.
    let size = beams.iter().map(|b| 0.5 * b.rabi).sum::<f64>().powi(2) * K * tau / detuning.hypot(0.5 * gamma);
    let impulse = (s.end.kappa[2] - start.kappa[2] - kick).abs() / kick.abs().max(size);
    [state, impulse]
}

/// The step integrates its generator's own frequencies exactly, whatever the detuning turns over the step (step.rs, # The
/// kick; spectral.rs): a two-level atom in a standing wave whose light shift turns its dressed state by 3 rad a step,
/// from the bare ground state -- the state after a jump, whose excited amplitude rings at the detuning --, with and
/// without the decay, at Δτ = 1, 10, 50, 200.
/// - At rest the generator is one matrix over the step: the state is the sub-stepped reference's to its rounding, and
///   so is the kick without the decay. With it the kick is a quotient by the norm, whose own ringing -- Γ/|Δ| times
///   the excited share, the bare state's 4U/|Δ| at most -- is beyond the bandwidth the quadrature resolves to round-off
///   (step.rs, # The kick): the kick errs by a part of that ringing, 10⁻⁵ of the dipole kick at the most here (at Δτ =
///   200) and 2·10⁻⁷ at Δτ = 50; held to the ringing itself and to 10⁻⁴.
/// - Moving, the light changes along the path by k·v τ: the errors are of the scheme's order, the second in that
///   change, and at one light shift they do not grow with Δτ. The state's is what a hop leaves out (spectral.rs: the
///   second Magnus term), the light shift of the change itself over a hop, at most half of (k v τ)²/12 of the light
///   shift's turn Uτ over the step -- held here to all of it, and falling threefold at least per halving of the
///   velocity. The kick's against the path's is the splitting's own beside it: the middle state's force is read at the
///   fixed r_h, (k v τ)²/6 of it off its mean along the path.
#[test]
fn a_step_integrates_its_internal_frequencies_whatever_the_detuning() {
    let (tau, shift) = (0.0625, 3.0);
    let mut failures = Vec::new();
    for gamma in [0.0, GAMMA] {
        for turns in [1.0_f64, 10.0, 50.0, 200.0] {
            let detuning: f64 = -turns / tau;
            let rabi = (shift / tau * detuning.abs()).sqrt();
            let beams = [Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.3 }];
            let rest = step_errors(&beams, detuning, gamma, 0.11, 0.0, tau);
            let speeds = [0.4, 0.2, 0.1];
            let moving: Vec<[f64; 2]> = speeds.iter().map(|&v| step_errors(&beams, detuning, gamma, 0.11, v, tau)).collect();
            eprintln!("Γ {gamma:6.3}, Δτ {turns:5}: at rest ψ {:.2e}, kick {:.2e}; moving at 0.4, 0.2, 0.1 µm/µs ψ {:.2e} {:.2e} {:.2e}, kick {:.2e} {:.2e} {:.2e}", rest[0], rest[1], moving[0][0], moving[1][0], moving[2][0], moving[0][1], moving[1][1], moving[2][1]);
            let ringing = (gamma / detuning.abs() * 4.0 * shift / tau / detuning.abs()).min(1e-4);
            if !(rest[0] < 1e-11 && rest[1] < if gamma == 0.0 { 1e-11 } else { ringing }) {
                failures.push(format!("Γ {gamma}, Δτ {turns}: at rest {rest:?}"));
            }
            for q in 0..2 {
                let within = (0..3).all(|i| moving[i][q] < (K * speeds[i] * tau).powi(2) * (shift / 12.0 + [0.0, 1.0 / 6.0][q]));
                let falls = q == 1 || (0..2).all(|i| moving[i + 1][q] < moving[i][q] / 3.0);
                if !(within && falls) {
                    failures.push(format!("Γ {gamma}, Δτ {turns}: moving, {} {:?}", ["ψ", "kick"][q], [moving[0][q], moving[1][q], moving[2][q]]));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The step's order in what the path changes (step.rs, # The kick; spectral.rs): the same atom from the bare ground
/// state at 0.4 µm/µs (k·v = 3.2 rad/µs), over 0.0625 µs in one, two and four steps, with and without the decay, at
/// detunings that turn 1, 10, 50 and 200 rad over that span (the light shift 3 rad over it at each). A mass the kick
/// does not move: the steps' paths are the reference's, and a step is its propagation. The state against the path's
/// sub-stepped reference, and the kick against the reference of the step's own decomposition (`decomposed`): what
/// the propagation leaves of the path's change, without the splitting.
/// - Where the step's nodes resolve the pair (it turns less than a radian between two of them: Δτ = 1 in two and in
///   four steps) the state is hopped, and both fall as the fourth order, sixteen a halving: a hop's second Magnus term
///   is of the fifth order in the step, and so is the three nodes' representation of the change (Simpson's rule's).
/// - Where they do not (Δτ = 1 in one step, Δτ = 10 and more in any) the state follows the change, by the following
///   X = D/ω and its rate's corrections, to the second order. Its error is within the square of the following at the
///   step's end, (k·v h V/ω)² with V the light's largest coupling and ω the dressed pair's frequency -- held to that
///   here, and measured an order and more below it --, so it falls with the step and with the detuning. (The hops
///   through an unresolved pair that this replaces left the light shift of the change, (k·v τ)²/24 of the light shift's
///   turn at any detuning: 3·10⁻³ a step here, where the following leaves 2·10⁻⁵ at Δτ = 50 and 2·10⁻⁶ at 200.)
#[test]
fn a_step_is_of_the_fourth_order_in_the_path_s_change_where_its_hops_resolve_the_generator() {
    let (span, shift, v, z0, hm) = (0.0625, 3.0, 0.4, 0.11, 1e-9);
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let mut failures = Vec::new();
    for gamma in [0.0, GAMMA] {
        for turns in [1.0_f64, 10.0, 50.0, 200.0] {
            let detuning: f64 = -turns / span;
            let rabi = (shift / span * detuning.abs()).sqrt();
            let beams = [Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.3 }];
            let model = Atom::new(hm, flat, beams.to_vec(), detuning, gamma);
            // The dressed state's dipole kick over the span (`step_errors`): the kick's scale.
            let size = beams.iter().map(|b| 0.5 * b.rabi).sum::<f64>().powi(2) * K * span / detuning.hypot(0.5 * gamma);
            let errors: Vec<[f64; 2]> = [1, 2, 4]
                .into_iter()
                .map(|steps: usize| {
                    let tau = span / steps as f64;
                    let subs = 2 * ((1000.0 * turns) as usize).max(10_000) / steps;
                    let start = held(&model, [0.0, 0.0, z0], [0.0, 0.0, v / hm]);
                    let (mut t, mut exact, mut kick) = (start.clone(), [C64::ONE, C64::ZERO], 0.0);
                    for i in 0..steps {
                        let (end, part) = decomposed(&beams, detuning, gamma, z0 + v * i as f64 * tau, v, exact, tau, subs);
                        (exact, kick) = (end, kick + part);
                        t = step::step(&model, &t, tau, &mut step::Scratch::default()).unwrap().end;
                    }
                    let state = (0..2).map(|a| (t.psi[a] - exact[a]).abs()).fold(0.0, f64::max);
                    [state, (t.kappa[2] - start.kappa[2] - kick).abs() / kick.abs().max(size)]
                })
                .collect();
            let falls: Vec<[f64; 2]> = (0..2).map(|i| std::array::from_fn(|q| errors[i][q] / errors[i + 1][q])).collect();
            eprintln!(
                "Γ {gamma:6.3}, Δτ {turns:5}: in 1, 2, 4 steps ψ {:.2e} {:.2e} {:.2e} (falls {:.1}, {:.1}), kick {:.2e} {:.2e} {:.2e} (falls {:.1}, {:.1})",
                errors[0][0], errors[1][0], errors[2][0], falls[0][0], falls[1][0], errors[0][1], errors[1][1], errors[2][1], falls[0][1], falls[1][1]
            );
            // The pair's frequency at the light's largest coupling V, and the following's size at the end of a step
            // of the span's n-th part: k·v h V/ω.
            let coupling = beams.iter().map(|b| 0.5 * b.rabi).sum::<f64>();
            let omega = detuning.hypot(2.0 * coupling);
            let following = |steps: usize| K * v * 0.5 * span / steps as f64 * coupling / omega;
            // Resolved by the step's nodes: less than a radian between two of them, at the pair's largest frequency.
            let resolved = |steps: usize| omega * 0.5 * span / (steps as f64) < 1.0;
            let hopped = (0..2).all(|i| !resolved(1 << i) || (0..2).all(|q| falls[i][q] > 12.0));
            let followed = (0..3).all(|i| resolved(1 << i) || errors[i][0] < following(1 << i).powi(2));
            if !(hopped && followed) {
                failures.push(format!("Γ {gamma}, Δτ {turns}: {errors:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Where a step's norm reaches a threshold (step.rs, # Jumps): the same atom with the decay, from the bare ground
/// state, at Δτ = 1, 10, 50 and 200, at rest and at 0.4 µm/µs, the threshold the sub-stepped reference's ‖ψ‖² at 0.37
/// of the step -- between the step's nodes, where the state is the nodes' quintic and the following. The step tried
/// for it (`Midway::attempt`, no slack) proposes that time: its error, as the norm's logarithm there (the time's error
/// times the norm's rate), is the reference's own accuracy at rest -- the state is exact at any time of the step --
/// and within the square of the following where the atom moves (`a_step_is_of_the_fourth_order…`), as the state is.
/// And without the decay the norm stays one within the same.
#[test]
fn a_step_s_norm_reaches_a_threshold_where_the_path_s_does_whatever_the_detuning() {
    let (tau, shift, hm, z0) = (0.0625, 3.0, 1e-9, 0.11);
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let norm = |psi: &[C64]| psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
    let mut failures = Vec::new();
    for turns in [1.0_f64, 10.0, 50.0, 200.0] {
        let detuning: f64 = -turns / tau;
        let rabi = (shift / tau * detuning.abs()).sqrt();
        let beams = [Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.3 }];
        let coupling = beams.iter().map(|b| 0.5 * b.rabi).sum::<f64>();
        for v in [0.0, 0.4] {
            let following = K * v * 0.5 * tau * coupling / detuning.hypot(2.0 * coupling);
            let steps = ((2000.0 * detuning.abs() * tau) as usize).max(20_000);
            let at = 0.37 * tau;
            let reference = |time: f64| norm(&substepped(&beams, detuning, GAMMA, z0, v, [C64::ONE, C64::ZERO], time, steps, None).0);
            let (theta, rate) = (reference(at), (reference(0.99 * at) / reference(1.01 * at)).ln() / (0.02 * at));
            let model = Atom::new(hm, flat, beams.to_vec(), detuning, GAMMA);
            let mut start = held(&model, [0.0, 0.0, z0], [0.0, 0.0, v / hm]);
            start.threshold = theta;
            let tried = step::Midway::new(&model, &start, tau).unwrap().attempt(&start, &step::Brownian::STILL, 0.0, 0.0, &mut step::Scratch::default()).unwrap();
            let step::Attempt::Proposes { length, .. } = tried else { panic!("Δτ {turns}, v {v}: no crossing within the step") };
            let error = (length - at) * rate;
            // Without the decay: the norm after the step.
            let free = Atom::new(hm, flat, beams.to_vec(), detuning, 0.0);
            let kept = norm(&step::step(&free, &held(&free, [0.0, 0.0, z0], [0.0, 0.0, v / hm]), tau, &mut step::Scratch::default()).unwrap().end.psi) - 1.0;
            eprintln!("Δτ {turns:5}, v {v}: the threshold {theta:.6} is reached at {length:.9} µs against {at:.9} (ln‖ψ‖² off by {error:+.2e}, its rate {rate:.3}/µs); without the decay ‖ψ‖² − 1 = {kept:+.2e}");
            let within = if v == 0.0 { 1e-9 } else { following * following };
            if !(error.abs() < within && kept.abs() < within) {
                failures.push(format!("Δτ {turns}, v {v}: ln‖ψ‖² off by {error:e}, ‖ψ‖² − 1 = {kept:e} (within {within:e})"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The step's doubling reads what the step leaves (adapt.rs, # The read): the same atom moving at 0.2 µm/µs, the decay
/// on, at Δτ = 50 and 200 -- the detuning unresolved by the step and by its halves alike. The step's error of the
/// state's fall of ln‖ψ‖² (the hazard, which times the jumps) against the sub-stepped reference is the doubling's
/// estimate of it, 4/3 of the step less its two halves, within a half of itself -- an error of 6·10⁻⁷ and of 5·10⁻⁸
/// of the hazard's integral here, a thousand times the sub-stepped reference's own (10⁻⁹ is asked of it).
#[test]
fn the_doubling_reads_what_a_step_leaves_of_the_path_s_change() {
    let (hm, tau, shift, v, z0) = (1e-9, 0.0625, 3.0, 0.2, 0.11);
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    for turns in [50.0_f64, 200.0] {
        let detuning: f64 = -turns / tau;
        let rabi = (shift / tau * detuning.abs()).sqrt();
        let beams = [Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.3 }];
        let model = Atom::new(hm, flat, beams.to_vec(), detuning, GAMMA);
        let start = held(&model, [0.0, 0.0, z0], [0.0, 0.0, v / hm]);
        let take = |from: &Trajectory, length: f64| step::step(&model, from, length, &mut step::Scratch::default()).unwrap().end;
        let (whole, halves) = (take(&start, tau), take(&take(&start, 0.5 * tau), 0.5 * tau));
        let (exact, _) = substepped(&beams, detuning, GAMMA, z0, v, [C64::ONE, C64::ZERO], tau, (2000.0 * turns) as usize, None);
        let fall = |psi: &[C64]| -psi.iter().map(|z| z.norm_sqr()).sum::<f64>().ln();
        let (error, read) = (fall(&whole.psi) - fall(&exact), 4.0 / 3.0 * (fall(&whole.psi) - fall(&halves.psi)));
        eprintln!("Δτ {turns}: the hazard's integral {:.5}, the step's error of it {error:+.3e}, the doubling reads {read:+.3e}", fall(&exact));
        assert!((read - error).abs() < 0.5 * error.abs() && error.abs() > 1e-9, "Δτ {turns}: error {error:e}, read {read:e}");
    }
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
        let mut path = adapt::Path::new(t.seed, t.index, tau, 0);
        adapt::advance(model, tau, &mut t, &mut step::Scratch::default(), &mut path, from, &mut |_| {}).unwrap();
        let start = t.kappa;
        adapt::advance(model, tau, &mut t, &mut step::Scratch::default(), &mut path, to, &mut |_| {}).unwrap();
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
/// ln(1/θ_j)/γ of the thresholds its streams draw (θ₀ from jump 0's, θ_j the first draw of jump j's). A constant
/// hazard's norm is one curve along a step of any length, so the step tried at the last step's rate ends at the
/// threshold to round-off and is the step to the jump: over thousands of jumps the count is the thresholds' and the
/// final norm theirs to round-off. Their waiting times' mean and variance those of the exponential law within four
/// standard errors.
#[test]
fn the_jumps_are_at_the_roots_of_the_norm() {
    let gamma = 2.0;
    let flat = Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] };
    let model = Atom::new(RB, flat, vec![], 0.0, gamma);
    let mut t = Trajectory::new(&model, 0, [0.0; 3], [0.0; 3], 3, 5);
    let end = 2000.0;
    adapt::advance(&model, 10.0, &mut t, &mut step::Scratch::default(), &mut adapt::Path::new(3, 5, 10.0, 0), end, &mut |_| {}).unwrap();
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

/// The jumps' times (adapt.rs, # The jumps' times): a two-level atom about the antinode of a resonant standing wave
/// along z, in a trap whose period the nominal step does not resolve (ω τ = 3), jumping some times a nominal step --
/// the hazard follows the atom through the wave, so a step's norm is its own curve. Every step `advance` takes, taken
/// again from the same state as the flow between jumps (`step::step`), ends with ‖ψ‖² at the threshold -- within the
/// tolerance of its length at twice its own mean rate -- exactly where a jump follows it, and above it elsewhere:
/// replayed so, a jump after each such step, the trajectory is `advance`'s own to the bit. The steps are the jumps' own:
/// far shorter than the nominal one.
#[test]
fn a_jump_is_made_where_its_own_step_reaches_the_threshold() {
    let (omega, tau, end) = (0.1, 30.0, 900.0);
    let mut hessian = [[0.0; 3]; 3];
    hessian[2][2] = omega * omega / RB;
    let wave = |k: f64| Beam { k: [0.0, 0.0, k], rabi: 1.7, phase: 0.0 };
    let model = Atom::new(RB, Trap::Quadratic { hessian, gradient: [0.0; 3] }, vec![wave(K), wave(-K)], 0.0, GAMMA);
    let start = Trajectory::new(&model, 0, [0.0, 0.0, 0.05], [0.0, 0.0, 8.0], 3, 5);
    let (mut t, mut steps) = (start.clone(), Vec::new());
    let mut path = adapt::Path::new(t.seed, t.index, tau, 0);
    adapt::advance(&model, tau, &mut t, &mut step::Scratch::default(), &mut path, end, &mut |s| steps.push(s.tau)).unwrap();
    let norm = |psi: &[C64]| psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
    let (mut twin, mut scratch, mut longest) = (start.clone(), step::Scratch::default(), 0.0_f64);
    for &length in &steps {
        let taken = step::step(&model, &twin, length, &mut scratch).unwrap().end;
        let fall = (norm(&twin.psi) / norm(&taken.psi)).ln();
        (twin.r, twin.kappa, twin.psi, twin.time) = (taken.r, taken.kappa, taken.psi, taken.time);
        let above = (norm(&twin.psi) / twin.threshold).ln();
        if above.abs() <= 2.0 * crate::units::JUMP_TIMING * fall {
            step::jump(&model, &mut twin).unwrap();
            longest = longest.max(length);
        } else {
            assert!(above > 0.0, "a step of {length} µs to {} ends {above} below its threshold in ln‖ψ‖² with no jump", twin.time);
        }
    }
    assert!(t.jumps > 100 && steps.len() as f64 * tau > 3.0 * end && longest < tau, "{} jumps in {} steps, the longest to a jump {longest} µs", t.jumps, steps.len());
    assert_eq!((twin.jumps, twin.r, twin.kappa, &twin.psi), (t.jumps, t.r, t.kappa, &t.psi), "the replay against the advance");
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
    let mut path = adapt::Path::new(t.seed, t.index, tau, 0);
    for k in 1..=outputs {
        let to = run * k as f64 / outputs as f64;
        adapt::advance(&model, tau, &mut t, &mut step::Scratch::default(), &mut path, to, &mut |_| {}).unwrap();
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
        start = step::step(&model, &start, 0.5, &mut step::Scratch::default()).unwrap().end;
        let difference = |tau: f64| -> [f64; 2] {
            let full = step::step(&model, &start, tau, &mut step::Scratch::default()).unwrap();
            let first = step::step(&model, &start, 0.5 * tau, &mut step::Scratch::default()).unwrap();
            let second = step::step(&model, &first.end, 0.5 * tau, &mut step::Scratch::default()).unwrap();
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
/// A diffusion's factor (`factor`) has its rate, Λ Λᵀ = 2D to the entries' rounding, and no column in a null direction
/// of it. A class absorbing from a plane wave of wave vector k has ∇u = ik u: its jumps carry k and leave nothing -- the
/// force and 2D are the rounding of their sums, and the factor is exactly zero (the root of that rounding would be
/// ~1e-8 of the recoil a step). An envelope's gradient α beside it, ∇u = (ik + α)u, leaves 2D = W α αᵀ: one column, along
/// α -- also where α's first component is its least by far, which a factor taken in the axes' order turns into a spurious
/// second column.
#[test]
fn a_diffusion_s_factor_has_its_rate_and_no_column_in_a_null_direction() {
    let mut stream = crate::random::Stream::new(53, crate::random::Purpose::Initial, 0, 0);
    let (k, n) = ([3.1, -4.2, 5.9], 5);
    let amplitudes: [Vec<C64>; 3] = std::array::from_fn(|_| (0..n).map(|_| C64::new(stream.normal(), stream.normal())).collect());
    for alpha in [[0.0; 3], [0.21, -0.13, 0.08], [1e-5, 0.2, 0.0]] {
        let gradients: [[Vec<C64>; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|a| amplitudes[a].iter().map(|u| *u * C64::new(alpha[i], k[i])).collect()));
        let photon = Photon { class: 0, landing: 0, wave_number: 0.0, absorbed: crate::linalg::norm3(k), amplitudes: amplitudes.clone(), gradients, explicit: [[0.0; 3]; 3] };
        let (weight, scattered) = (photon.weight(), Scattering::of(std::slice::from_ref(&photon), 1.0));
        let rounded = crate::units::rounding(scattered.links) * scattered.size;
        // The jumps carry k: the force they leave is the rounding of the mean's sum.
        let kick = photon.kick();
        for i in 0..3 {
            assert!((kick[i] - k[i]).abs() <= 8.0 * f64::EPSILON * crate::linalg::norm3(k), "α {alpha:?}: the jump carries {kick:?}");
            assert!(scattered.force[i].abs() <= rounded, "α {alpha:?}: the force left {:?}", scattered.force);
        }
        let l = super::factor(&scattered.diffusion, scattered.size, scattered.links);
        if alpha == [0.0; 3] {
            assert_eq!(l, [[0.0; 3]; 3], "a plane wave's absorption diffuses (its rate's rounding {:e})", scattered.diffusion[0][0]);
        }
        for i in 0..3 {
            assert_eq!((l[i][1], l[i][2]), (0.0, 0.0), "α {alpha:?}: a second column {l:?}");
            for j in 0..3 {
                // The factor's own backward error and the columns left out: the pivots' bound, 18 γ_N S at most.
                let made: f64 = (0..3).map(|r| l[i][r] * l[j][r]).sum();
                assert!((made - weight * alpha[i] * alpha[j]).abs() <= 2.0 * 18.0 * rounded, "α {alpha:?}: (Λ Λᵀ)[{i}][{j}] {made} against {}", weight * alpha[i] * alpha[j]);
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
    let step = step::step(&model, &start, tau, &mut step::Scratch::default()).unwrap();
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

/// With light the step is its own inverse up to its path momentum's prediction (step.rs, # The path's momentum: the two
/// directions' predictions differ by the held kick's trapezoid against the kick), of an order beyond the scheme's: a
/// two-level atom in a far-detuned standing wave (Γ = 0), one step and its reverse return to the start by less than the
/// step's doubling error. From the light's dressed state at the start, which follows the dressing along the path -- the
/// step resolves its change --, the miss falls over two halvings of τ faster than τ³ does. From the bare ground state,
/// whose admixture turns at the detuning, 15 to 60 rad a step, the two predictions meet the turning at different
/// phases: the miss stays below the doubling's difference, and no order is asked of it.
#[test]
fn a_step_with_light_and_its_reverse_miss_beyond_the_scheme_s_order() {
    let trap = Trap::Gaussian { depth: 6.0, waists: [1.0, 1.0, 3.0] };
    let beams = vec![Beam { k: [0.0, 0.0, K], rabi: 30.0, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi: 30.0, phase: 0.2 }];
    let model = Atom::new(RB, trap, beams, -300.0, 0.0);
    let bare = held(&model, [0.1, -0.1, 0.3], [30.0, -20.0, 60.0]);
    // The dressed state at the start: the eigenvector of [[0, c*], [c, −Δ]] that the ground state goes over into.
    let (c, d) = (model.coupling(0, bare.r) + model.coupling(1, bare.r), -model.detuning);
    let lambda = 0.5 * (d - (d * d + 4.0 * c.norm_sqr()).sqrt());
    let mut dressed = bare.clone();
    dressed.psi = [C64::ONE, c / (lambda - d)].into_iter().collect();
    for (name, start) in [("dressed", dressed), ("bare", bare)] {
        let mut misses = Vec::new();
        for tau in [0.2, 0.1, 0.05] {
            let there = step::step(&model, &start, tau, &mut step::Scratch::default()).unwrap();
            let back = step::step(&model, &there.end, -tau, &mut step::Scratch::default()).unwrap();
            let first = step::step(&model, &start, 0.5 * tau, &mut step::Scratch::default()).unwrap();
            let second = step::step(&model, &first.end, 0.5 * tau, &mut step::Scratch::default()).unwrap();
            let miss = (0..3).map(|a| (back.end.kappa[a] - start.kappa[a]).abs()).fold(0.0, f64::max);
            let doubling = (0..3).map(|a| (there.end.kappa[a] - second.end.kappa[a]).abs()).fold(0.0, f64::max);
            eprintln!("{name}, τ {tau}: forth and back misses κ by {miss:e}, the doubling differs by {doubling:e}");
            assert!(miss < doubling, "{name}, τ {tau}: miss {miss:e} against the doubling's {doubling:e}");
            misses.push(miss);
        }
        // Over the two halvings τ³ falls by 4³.
        assert!(name == "bare" || misses[0] / misses[2] > 64.0, "{name}: the miss falls as {misses:?}");
    }
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
/// the output interval below the trap's resonance ω τ = π). None is lost; between
/// jumps -- a far-detuned tweezer still scatters its folded far light's photons (SPEC §1.2), each a recoil the energy is
/// re-read after -- the atoms' mean E_s moves by less than the step's share of k_BT, s·ε_T; and the ensemble's kinetic
/// energy keeps its time-averaged level (T constant): its second half's mean within the first half's statistics.
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
    let (mut kinetic, mut steps, mut moved, mut jumps) = (vec![0.0; request.times_us.len()], 0, 0.0, 0);
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
        let mut path = adapt::Path::new(17, i, request.times_us[1] - request.times_us[0], ((request.times_us[1] - request.times_us[0]) / tau).log2().round() as u32);
        let mut e0 = run.energy(t.cluster, &t.point(), &t.psi);
        for (k, &time) in request.times_us.iter().enumerate().skip(1) {
            let before = t.jumps;
            adapt::advance(&run, tau, &mut t, &mut step::Scratch::default(), &mut path, time, &mut |_| {}).unwrap();
            assert!(t.lost.is_none(), "{name}: atom {i} lost at {:?}", t.lost);
            let e = run.energy(t.cluster, &t.point(), &t.psi);
            if t.jumps == before {
                moved += (e - e0) / (kt * (atoms as usize * (request.times_us.len() - 1)) as f64);
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
    let share = STEP_SHARE * TEMPERATURE_TARGET;
    eprintln!(
        "{name}: {atoms} atoms × {} µs at τ {tau}, {:.2} s, {steps} step evaluations, {jumps} jumps; E_s moved {moved:+.5} k_BT between jumps on average (share {share:.4}); kinetic {first:.4} then {second:.4} (± {spread:.4}), k_BT/2 per axis {:.4}",
        request.times_us.last().unwrap(),
        clock.elapsed().as_secs_f64(),
        0.5 * kt
    );
    assert!(moved.abs() <= share, "{name}: the atoms' E_s moved {moved} k_BT between jumps");
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
    fn kick(&self, _: usize, _: &Point, _: &[C64], _: &Photon) -> [f64; 3] {
        unreachable!("the cost harness takes no jump")
    }
    fn scattering(&self, _: usize, _: &Point, _: &[C64], _: &[f64]) -> Scattering {
        Scattering::NONE
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
    fn kick(&self, cluster: usize, at: &Point, psi: &[C64], photon: &Photon) -> [f64; 3] {
        self.model.kick(cluster, at, psi, photon)
    }
    fn scattering(&self, cluster: usize, at: &Point, psi: &[C64], populations: &[f64]) -> Scattering {
        self.model.scattering(cluster, at, psi, populations)
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

/// A scratch fitted to n states (the harness takes the step's parts apart).
fn fitted(n: usize) -> super::Scratch {
    let mut scratch = super::Scratch::default();
    scratch.fit(n);
    scratch
}

/// A threshold watched within the step alone: a crossing anywhere before the step's end ends the propagation there.
fn watched(theta: f64) -> super::Watch {
    super::Watch { theta, slack: 0.0, reach: 0.0 }
}

/// One case of the cost harness: µs per step of τ over the case's start states (held: no step reaches their thresholds),
/// split into the model's part -- its reference expansions, its middle, and at the middle's nodes its generators, forces
/// and tones, each timed alone and counted as the step calls them -- and the step's own, the rest. Then the step's
/// phases, the model's calls within them: the middle's fixed point less its reference expansions; the propagation on
/// its path with its impulse (also with the threshold reached halfway down the step's fall of the norm: its root).
/// The timed steps and propagations share one scratch, as a trajectory's steps do (adapt.rs). Returns the
/// propagation's time.
fn cost<D: Dynamics>(name: &str, model: &D, starts: &[Trajectory], tau: f64) -> f64 {
    let (count, half, hm) = (starts.len(), 0.5 * tau, model.hbar_over_mass());
    let kept = std::cell::RefCell::new(fitted(starts[0].psi.len()));
    let counted = Counted { model, calls: std::cell::Cell::new([0; 5]) };
    let steps: Vec<step::Step> = starts.iter().map(|s| step::step(&counted, s, tau, &mut step::Scratch::default()).unwrap()).collect();
    let calls = counted.calls.get().map(|c| c as f64 / count as f64);
    // Each start's step, piece by piece as `step::step` makes it.
    struct Pieces<M: Middle> {
        at: ([f64; 3], f64),
        middle: M,
        kappa: [f64; 3],
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
            let kappa = super::predicted(&mut fitted(from.psi.len()), &middle, &from.psi, kappa_h, reference, tau);
            let norm = |psi: &[C64]| psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
            let mut crossed = from.clone();
            crossed.threshold = (norm(&from.psi) * norm(&taken.end.psi)).sqrt();
            let mut rho = vec![C64::ZERO; from.psi.len() * from.psi.len()];
            super::density(&from.psi, &mut rho);
            Pieces { at: (r_h, from.time + half), middle, kappa, crossed, rho }
        })
        .collect();
    // A piece's propagation, watched for the start's own threshold (never reached) or for the one it crosses.
    let propagation = |crossing: bool| {
        pieces
            .iter()
            .zip(starts)
            .map(|(p, s)| {
                let from = if crossing { &p.crossed } else { s };
                match super::propagate(&mut kept.borrow_mut(), &p.middle, &from.psi, p.kappa, hm, tau, Some(watched(from.threshold))) {
                    Ok(made) => made.impulse[0],
                    Err(crossed) => crossed.at,
                }
            })
            .sum::<f64>()
    };
    // The path of every piece, and a node a fifth of the step off the middle.
    let paths: Vec<_> = pieces.iter().map(|p| p.middle.path(p.kappa, p.kappa.map(|k| hm * k))).collect();
    let n = starts[0].psi.len();
    let mut h = Mat::zeros(n, n);
    let populations: Vec<Vec<f64>> = starts.iter().map(|s| s.psi.iter().map(|z| z.norm_sqr()).collect()).collect();
    let [whole, reference, point, generator, force, tones, middle, propagated, root, scattering, pace] = microseconds(
        count,
        [
            &mut || starts.iter().map(|s| step::step(model, s, tau, &mut kept.borrow_mut()).unwrap().end.kappa[0]).sum::<f64>(),
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
            &mut || propagation(false),
            &mut || propagation(true),
            &mut || pieces.iter().zip(starts).zip(&populations).map(|((p, s), populations)| model.scattering(s.cluster, &Point { r: p.at.0, kappa: p.kappa, t: p.at.1 }, &s.psi, populations).force[0]).sum::<f64>(),
            &mut yardstick,
        ],
    );
    let crossings = pieces
        .iter()
        .filter(|p| super::propagate(&mut fitted(p.crossed.psi.len()), &p.middle, &p.crossed.psi, p.kappa, hm, tau, Some(watched(p.crossed.threshold))).is_err())
        .count();
    let nodes = calls[2] * generator + calls[3] * force + calls[4] * tones;
    let own = whole - calls[0] * reference - calls[1] * point - nodes;
    let fixed = middle - calls[0] * reference;
    let work = steps.iter().map(|s| s.work).sum::<f64>() / count as f64;
    // What the propagations took: the picture's hops and the quadrature's nodes at its bandwidth, the starts' means.
    let [hops, nodes, turned] = pieces.iter().zip(starts).fold([0.0; 3], |sums, (p, s)| {
        let mut scratch = fitted(s.psi.len());
        let _ = super::propagate(&mut scratch, &p.middle, &s.psi, p.kappa, hm, tau, None);
        let picture = &scratch.picture;
        let found = [picture.hops as f64, super::nodes(tau, picture.bandwidth) as f64, picture.bandwidth * tau];
        std::array::from_fn(|i| sums[i] + found[i] / count as f64)
    });
    eprintln!(
        "{name:<22} n {n} τ {tau:<6} | step {whole:8.2} = model {:8.2} (reference {reference:5.2} × {:4.1} + middle {point:7.2} × {:.0} + generator {generator:6.2} × {:4.1} + force {force:6.2} × {:4.1} + tones {tones:6.2} × {:3.1}) + own {own:8.2} (scattering {scattering:6.2}) | fixed point {fixed:6.2}  propagation {propagated:8.2}  (root {root:8.2}, {crossings}/{count}; {hops:.1} hops, {nodes:.1} nodes at the bandwidth × τ {turned:.2}) | counted {work:9.0} | pace {:.3} ns",
        whole - own,
        calls[0],
        calls[1],
        calls[2],
        calls[3],
        calls[4],
        pace * count as f64 / 1.024
    );
    propagated
}

/// Atoms of a cluster's cost case: `count` atoms at the tweezer's thermal sizes (positions of 0.1 µm, speeds of
/// 0.05 µm/µs in uniform directions), each a random state of the cluster, held.
fn atoms<D: Dynamics>(model: &D, count: u64) -> Vec<Trajectory> {
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
            t
        })
        .collect()
}

/// An R1 scene's cost case: the scene's far-light model (`model::Run`) from its reference state, sixteen atoms of the
/// scene's temperature in the trap's harmonic approximation (as `conservative_trap`), held, one step of τ each.
fn scene_cost(name: &str, tau: f64) -> f64 {
    let mut time = 0.0;
    scene_atoms(name, &mut |run, starts| time = cost(name, run, starts, tau));
    time
}

/// An R1 scene's far-light model and its atoms, to `each`.
fn scene_atoms(name: &str, each: &mut dyn FnMut(&crate::model::Run, &[Trajectory])) {
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
/// (`Model::tabulate`); what the ensemble's draw meets (the prepared well's depth against its envelope's top); and per
/// cluster what the propagation meets (the generator at the trap's centre, at rest, less its trace, its fastest tone,
/// each times τ, and the bounds of its Hermitian and decay parts) and a cost case (`cost`): sixteen atoms of the scene's
/// temperature in the trap's harmonic approximation, each a random state of the cluster, held, one step of τ each.
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
    // What the ensemble's draw meets (ensemble.rs): the prepared well's depth at its bottom against its envelope's top.
    eprintln!(
        "{name}: the prepared well at {:?} is {:.3} k_BT deep, its envelope's top {:.3} k_BT; trap frequencies {:?} rad/µs",
        trap.centre,
        -model.potential(trap.centre, &p, &model.prepared()) / kt,
        model.envelope(&p).peak() / kt,
        trap.frequencies
    );
    for (cluster, c) in model.clusters().iter().enumerate() {
        // What the propagation meets: the generator at the trap's centre, at rest, less its trace, and its fastest tone.
        {
            let (middle, n) = (run.middle(cluster, trap.centre, 0.0), c.states.len());
            let path = middle.path([0.0; 3], [0.0; 3]);
            let mut h = Mat::zeros(n, n);
            middle.generator(&path, 0.0, &mut h);
            let mean = (0..n).map(|x| h[(x, x)]).sum::<C64>() / n as f64;
            let norm = (0..n * n).map(|i| (h.data[i] - if i / n == i % n { mean } else { C64::ZERO }).norm_sqr()).sum::<f64>().sqrt();
            let rho: Vec<C64> = (0..n * n).map(|_| C64::real(1.0 / n as f64)).collect();
            let mut fastest = 0.0_f64;
            middle.tones(&path, &rho, &mut |omega, _| fastest = fastest.max(omega.abs()));
            eprintln!(
                "{name} cluster {cluster} ({}+{}, {}): its generator less its trace {norm:.3e} rad/µs, times τ {:.3e}; its fastest tone {fastest:.3e} rad/µs, times τ {:.3e}",
                c.ground,
                n - c.ground,
                if middle.still() { "still" } else { "beating" },
                norm * tau,
                fastest * tau
            );
            let crate::linalg::Bounds { rho, gamma, .. } = crate::linalg::bounds(&h, &mut vec![C64::ZERO; n]);
            eprintln!("{name} cluster {cluster}: its generator's bounds ρ {rho:.3e}, γ {gamma:.3e} rad/µs, times τ {:.3e}, {:.3e}", rho * tau, gamma * tau);
        }
        let starts: Vec<Trajectory> = (0..16)
            .map(|i| {
                let mut stream = crate::random::Stream::new(17, crate::random::Purpose::Initial, i, cluster as u32);
                let (r, kappa) = thermal(&trap, run.hbar_over_mass(), kt, &mut stream);
                let mut t = Trajectory::new(&run, c.states[0], r, kappa, 17, i);
                let psi: Vec<C64> = t.psi.iter().map(|_| C64::new(stream.normal(), stream.normal())).collect();
                let norm = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                t.psi = psi.iter().map(|z| *z / norm).collect();
                t.threshold = f64::MIN_POSITIVE;
                t
            })
            .collect();
        cost(&format!("{}+{} of {}", c.ground, c.states.len() - c.ground, std::path::Path::new(&name).file_stem().unwrap().to_string_lossy()), &run, &starts, tau);
    }
}

/// The step's cost (run by name, alone: `the_step_s_cost -- --ignored --nocapture`): µs per `step::step` and per phase,
/// single-threaded, on the R1 scenes' far light (a static path and a standing wave's beating one), on a two-level atom
/// with its excited state explicit in a standing wave in a tweezer -- near resonance, and twenty linewidths off it at
/// the steps 0.0625 and 0.5 µs --, and on clusters of 1, 3, 5 and 8 ground states at a near-resonant light's sizes
/// (`Synthetic`) on a static path (Doppler frames), on a standing wave's beating one (a pair of beat parts) and on six
/// beams' (nine pairs). Then clusters of 5 to 32 states with 19 and 201 parts (dense).
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn the_step_s_cost() {
    scene_cost("R1_TWEEZER", 2.5);
    scene_cost("R1_RETRO", 0.3125);
    let tau = 0.5;
    let standing = |rabi: f64| vec![Beam { k: [0.0, 0.0, K], rabi, phase: 0.0 }, Beam { k: [0.0, 0.0, -K], rabi, phase: 0.3 }];
    let tweezer = Trap::Gaussian { depth: 1000.0, waists: [1.0, 1.0, 3.0] };
    let two = Atom::new(RB, tweezer, standing(10.0), -0.5 * GAMMA, GAMMA);
    cost("two-level", &two, &atoms(&two, 8), tau);
    let far = Atom::new(RB, tweezer, standing(187.0), -20.0 * GAMMA, GAMMA);
    for tau in [0.0625, 0.5] {
        cost("two-level, 20 Γ", &far, &atoms(&far, 8), tau);
    }
    let six = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0], [0.0, 1.0, -1.0]]
        .map(|q| q.map(|x| x * K));
    for (n, depth) in [(1, 300.0), (3, 300.0), (5, 1000.0), (8, 3000.0)] {
        let still = Synthetic::new(n, depth, [0.0, 0.6 * K, 0.8 * K], &[]);
        cost(&format!("static {n}"), &still, &atoms(&still, 8), tau);
        for (label, waves) in [("standing", &six[2..3]), ("six beams", &six[..])] {
            let model = Synthetic::new(n, depth, [0.0; 3], waves);
            cost(&format!("{label} {n}"), &model, &atoms(&model, 8), tau);
        }
    }
    // Larger clusters and many parts: 9 pairs of beats (19 parts) and 100 pairs (201 parts).
    let mut directions = crate::random::Stream::new(37, crate::random::Purpose::Initial, 0, 0);
    let hundred: Vec<[f64; 3]> = (0..100).map(|_| directions.direction().map(|x| 2.0 * K * x)).collect();
    for n in [5, 8, 16, 32] {
        for waves in [&six[..], &hundred[..]] {
            let model = Synthetic::new(n, 1000.0, [0.0; 3], waves);
            cost(&format!("{} parts {n}", 1 + 2 * waves.len()), &model, &atoms(&model, 4), tau);
        }
    }
}

/// The ground state alone scattering a folded light of plane waves (amplitude, wave vector) in free flight (`Atom::folded`).
fn scatterer(hm: f64, folded: Vec<(C64, [f64; 3])>) -> Atom {
    let mut model = Atom::new(hm, Trap::Quadratic { hessian: [[0.0; 3]; 3], gradient: [0.0; 3] }, vec![], 0.0, 0.0);
    model.folded = folded;
    model
}

/// The interval of a statistic's standard error at the tests' confidence over `looks` looks (estimate.rs).
fn within(error: f64, looks: f64) -> f64 {
    crate::estimate::Spread { error, groups: f64::INFINITY }.interval_over(looks)
}

/// SPEC §1.6 in a travelling wave, f = c e^{ik·r}: the class's conditional mean is the wave's k, one photon's, so every
/// jump carries exactly k and the steps nothing -- no force is left to them and no diffusion (the factor's null
/// direction). Over a run the momentum is k times the jumps made, to the sums' rounding, whatever the Brownian path; the
/// jumps come at the rate c², so the absorbed momentum's mean and second moment per unit time are k c² and k kᵀ c².
#[test]
fn a_travelling_wave_s_jumps_carry_its_photons_and_its_steps_nothing() {
    let (k, c) = ([0.0, 0.0, 8.0], 0.5);
    let model = scatterer(4.6e-3, vec![(C64::real(c), k)]);
    let (end, mut jumps) = (512.0, 0);
    for index in 0..8 {
        let start = [0.3, -0.2, 1.1];
        let mut t = Trajectory::new(&model, 0, [0.1, 0.2, 0.3], start, 17, index);
        let mut path = adapt::Path::new(t.seed, t.index, 1.0, 0);
        adapt::advance(&model, 1.0, &mut t, &mut step::Scratch::default(), &mut path, end, &mut |_| {}).unwrap();
        assert!(t.jumps > 0 && t.taken > 0, "trajectory {index}: {} jumps over {} steps", t.jumps, t.taken);
        for a in 0..3 {
            let carried = t.jumps as f64 * k[a];
            assert!((t.kappa[a] - start[a] - carried).abs() <= crate::units::rounding(2 * t.jumps as usize + 4) * (start[a].abs() + carried.abs()), "trajectory {index}, axis {a}: κ moved by {} over {} jumps of {}", t.kappa[a] - start[a], t.jumps, k[a]);
        }
        jumps += t.jumps;
    }
    // The jumps' count is Poisson's at the rate c²: its mean over the eight runs within its standard error.
    let expected = 8.0 * c * c * end;
    assert!((jumps as f64 - expected).abs() <= within(expected.sqrt(), 1.0), "{jumps} jumps against {expected}");
}

/// SPEC §1.6 in a standing wave, f = c cos kz, on an atom too heavy to move (ħ/m → 0), at a node, between and at an
/// antinode. The class's mean is none (f is real), so its jumps carry nothing, and the steps carry the second moment's
/// rate 2D_zz = k² c² sin² kz as a diffusion: at the node, where there is no jump at all, k² c² -- the dipole force's
/// fluctuations (Gordon & Ashkin 1980) --, at the antinode nothing. Over unit times the momentum's increments along the
/// wave have the mean 0 and the variance 2D_zz within their standard errors, a normal's fourth moment (the old rule's
/// kicks, the second moment over the jumps' rate, had none), and across the wave nothing moves.
#[test]
fn a_standing_wave_s_light_diffuses_the_momentum_at_its_closed_form() {
    let (k, c, n) = (8.0, 0.5, 4096);
    let model = scatterer(1e-12, vec![(C64::real(0.5 * c), [0.0, 0.0, k]), (C64::real(0.5 * c), [0.0, 0.0, -k])]);
    for (name, phase) in [("node", std::f64::consts::FRAC_PI_2), ("between", std::f64::consts::FRAC_PI_4), ("antinode", 0.0_f64)] {
        let rate = k * k * c * c * phase.sin().powi(2);
        let mut t = Trajectory::new(&model, 0, [0.1, -0.2, phase / k], [0.0; 3], 23, 5);
        let (mut scratch, mut path) = (step::Scratch::default(), adapt::Path::new(23, 5, 1.0, 0));
        let (mut sums, mut last) = ([0.0; 4], t.kappa);
        for i in 0..n {
            adapt::advance(&model, 1.0, &mut t, &mut scratch, &mut path, (i + 1) as f64, &mut |_| {}).unwrap();
            let d = t.kappa[2] - last[2];
            assert_eq!((t.kappa[0], t.kappa[1]), (0.0, 0.0), "{name}: the momentum across the wave");
            for (p, sum) in sums.iter_mut().enumerate() {
                *sum += d.powi(p as i32 + 1);
            }
            last = t.kappa;
        }
        let [mean, second, _, fourth] = sums.map(|s| s / n as f64);
        eprintln!("{name}: {} jumps; the increments' mean {mean:.3e}, second moment {second:.4} (2D = {rate:.4}), fourth {fourth:.3}", t.jumps);
        if phase == 0.0 {
            assert_eq!(sums, [0.0; 4], "the antinode's atom is not moved");
            assert!(t.jumps > 0, "the antinode scatters");
            continue;
        }
        assert_eq!(t.jumps == 0, name == "node", "{name}: {} jumps", t.jumps);
        // A normal's moments of the variance 2D: the mean's error √(2D/n), the second's 2D √(2/n), the fourth's
        // (2D)² √(96/n).
        let looks = 3.0;
        assert!(mean.abs() <= within((rate / n as f64).sqrt(), looks), "{name}: the mean {mean}");
        assert!((second - rate).abs() <= within(rate * (2.0 / n as f64).sqrt(), looks), "{name}: the second moment {second} against {rate}");
        assert!((fourth - 3.0 * rate * rate).abs() <= within(rate * rate * (96.0 / n as f64).sqrt(), looks), "{name}: the fourth moment {fourth} against {}", 3.0 * rate * rate);
    }
}

/// Every jump's kick is one photon's at most (SPEC §1.6), also where the light's phase winds faster than any of its
/// waves: f = c (cos kz + iε sin kz), two unequal counter-propagating waves, has the conditional mean b_z = kε/(cos² kz +
/// ε² sin² kz) -- kε at the antinode and k/ε at the node, a hundred photons' at ε = 0.01, where the old rule's jump took
/// it whole. At every point the jump carries b up to k, and with what the step carries the two moments per unit time are
/// the closed forms: the mean k c² ε (the two waves' pressures' difference, the same everywhere) and the second moment
/// k² c² (sin² kz + ε² cos² kz); the diffusion left is not negative.
#[test]
fn a_jump_s_kick_is_one_photon_s_at_most() {
    let (k, c, epsilon) = (8.0, 0.5, 0.01);
    let mut model = scatterer(4.6e-3, vec![(C64::real(0.5 * c * (1.0 + epsilon)), [0.0, 0.0, k]), (C64::real(0.5 * c * (1.0 - epsilon)), [0.0, 0.0, -k])]);
    model.emission = 7.5;
    let mut carried_whole = 0;
    for i in 0..=64 {
        // From the antinode to the node, the last points within 1e-6 of it.
        let phase = if i <= 56 { std::f64::consts::FRAC_PI_2 * i as f64 / 57.0 } else { std::f64::consts::FRAC_PI_2 - 10f64.powi(56 - i) };
        let (r, psi) = ([0.0, 0.0, phase / k], [C64::new(0.6, -0.3)]);
        let at = Point { r, kappa: [0.0; 3], t: 0.0 };
        let (norm, photon) = (psi[0].norm_sqr(), &model.made(r, &psi)[0]);
        let (weight, b) = (photon.weight() / norm, k * epsilon / (phase.cos().powi(2) + (epsilon * phase.sin()).powi(2)));
        let kick = model.kick(0, &at, &psi, photon);
        let expected = b.min(k);
        assert!((kick[2] - expected).abs() <= 1e-9 * k && kick[0] == 0.0 && kick[1] == 0.0, "kz = {phase}: the jump carries {kick:?} of b = {b}");
        carried_whole += (b <= k) as usize;
        let scattered = model.scattering(0, &at, &psi, &[1.0]);
        let (mean, second) = (scattered.force[2] + weight * kick[2], scattered.diffusion[2][2] + weight * kick[2] * kick[2]);
        let closed = (k * c * c * epsilon, k * k * c * c * (phase.sin().powi(2) + (epsilon * phase.cos()).powi(2)));
        assert!((mean - closed.0).abs() <= 1e-9 * closed.0, "kz = {phase}: the mean per unit time {mean} against {}", closed.0);
        assert!((second - closed.1).abs() <= 1e-9 * k * k * c * c, "kz = {phase}: the second moment per unit time {second} against {}", closed.1);
        assert!(scattered.diffusion[2][2] >= -crate::units::rounding(scattered.links) * scattered.size, "kz = {phase}: 2D_zz = {}", scattered.diffusion[2][2]);
        // The jump itself: the emitted photon's recoil and the carried mean, k_φ + k at most.
        let mut t = Trajectory::new(&model, 0, r, [0.0; 3], 29, i as u64);
        step::jump(&model, &mut t).unwrap();
        let size = crate::linalg::norm3(t.kappa);
        assert!(size <= (model.emission + k) * (1.0 + 1e-12), "kz = {phase}: a jump's kick of {size} beyond {} + {k}", model.emission);
    }
    assert!(carried_whole > 8 && carried_whole < 65, "points on both sides of b = k: {carried_whole}");
}

/// A stretch's part of the Brownian path and its two pieces' (`Brownian::split`), cut at the middle and off it (a tenth,
/// two thirds): the increment is the sum of the pieces' and the first moment about the middle theirs joined
/// (`Brownian::joined`), exactly (to the rounding of the sums of deviates of the stretch's own scale, τ^{1/2} and
/// τ^{3/2}, a few standard deviations each); the pieces' parts have the law of a stretch's of their own lengths -- the
/// variances s and s³/12, nothing between the four --; and a step and its two halves on the same path end together.
/// An atom in free flight at a standing wave's node, where the light diffuses and makes no jump: the two ends differ by
/// the diffusion's change along the step alone, a hundredth of the (ħ/m)(τ/4)|Λ(ΔW₂ − ΔW₁)| they would be apart with
/// the increment alone.
#[test]
fn a_stretch_s_brownian_part_is_its_pieces_joined_and_they_end_together() {
    let (tau, n) = (0.8, 4096);
    let mut stream = crate::random::Stream::brownian(31, 2, 0, 0);
    let mut three = move || [stream.normals(), stream.normals(), stream.normals()];
    for share in [0.5, 0.1, 2.0 / 3.0] {
        let cut = share * tau;
        // The pieces' covariances, each over its own scale: ΔW₁², ΔW₂², ΔZ₁², ΔZ₂², and the six products between them.
        let (mut squares, mut products) = ([0.0; 4], [0.0; 6]);
        for _ in 0..n {
            let deviates = three();
            let whole = step::Brownian::over(tau, &[deviates[0], deviates[1]]);
            let [first, second] = whole.split(tau, cut, &three());
            let joined = first.joined(cut, &second, tau - cut);
            for a in 0..3 {
                assert!((joined.increment[a] - whole.increment[a]).abs() <= 64.0 * f64::EPSILON * tau.sqrt(), "the increment and its pieces'");
                assert!((joined.moment[a] - whole.moment[a]).abs() <= 64.0 * f64::EPSILON * tau.powf(1.5), "the first moment and its pieces'");
            }
            let scale = |s: f64| (s.sqrt(), (s.powi(3) / 12.0).sqrt());
            let ((w1, z1), (w2, z2)) = (scale(cut), scale(tau - cut));
            let x = [first.increment[0] / w1, second.increment[0] / w2, first.moment[0] / z1, second.moment[0] / z2];
            let mut pair = 0;
            for i in 0..4 {
                squares[i] += x[i] * x[i] / n as f64;
                for j in i + 1..4 {
                    products[pair] += x[i] * x[j] / n as f64;
                    pair += 1;
                }
            }
        }
        let looks = 30.0;
        for s in squares {
            assert!((s - 1.0).abs() <= within((2.0 / n as f64).sqrt(), looks), "cut at {share}: a piece's variance over its own: {s}");
        }
        for p in products {
            assert!(p.abs() <= within((1.0 / n as f64).sqrt(), looks), "cut at {share}: two of the pieces' deviates' covariance: {p}");
        }
    }
    // A step and its two halves on the same path.
    let (k, c) = (8.0, 0.5);
    let model = scatterer(4.6e-3, vec![(C64::real(0.5 * c), [0.0, 0.0, k]), (C64::real(0.5 * c), [0.0, 0.0, -k])]);
    let start = Trajectory::new(&model, 0, [0.0, 0.0, std::f64::consts::FRAC_PI_2 / k], [0.2, -0.1, 0.3], 37, 0);
    let mut scratch = step::Scratch::default();
    let deviates = three();
    let whole = step::Brownian::over(tau, &[deviates[0], deviates[1]]);
    let [first, second] = whole.split(tau, 0.5 * tau, &three());
    let taken = step::Midway::new(&model, &start, tau).unwrap().step(&start, &whole, None, &mut scratch).unwrap().end;
    let half = step::Midway::new(&model, &start, 0.5 * tau).unwrap().step(&start, &first, None, &mut scratch).unwrap().end;
    let halves = step::Midway::new(&model, &half, 0.5 * tau).unwrap().step(&half, &second, None, &mut scratch).unwrap().end;
    let lever = model.hm * 0.25 * tau * k * c * (second.increment[2] - first.increment[2]).abs();
    assert!(lever > 0.0 && (taken.kappa[2] - start.kappa[2]).abs() > 0.0, "the path moves the atom");
    eprintln!("a step and its halves end {:.2e} µm apart along the wave; with the increment alone {lever:.2e}", (taken.r[2] - halves.r[2]).abs());
    assert!((taken.r[2] - halves.r[2]).abs() <= 1e-2 * lever, "the ends {} and {} against the lever {lever}", taken.r[2], halves.r[2]);
    assert!((taken.kappa[2] - halves.kappa[2]).abs() <= 1e-2 * (taken.kappa[2] - start.kappa[2]).abs(), "the momenta {} and {}", taken.kappa[2], halves.kappa[2]);
}

/// The energy a diffusion puts into a trapped atom is (ħ/m) tr(2D) τ/2 a step in the mean at any ω τ (`spreading`): a
/// white force's work in a harmonic trap does not know the trap. An atom held at a standing wave's node -- no jump, and
/// 2D_zz = k² c² wherever its small orbit takes it -- in a trap of ω τ = 2.5 along the wave: over 1024 trajectories from
/// rest, the mean energy after 64 steps is the diffusion's input within its standard error (an energy of the
/// exponential law: its own size over √1024). With the free flight's lever the potential energy would gain (ω τ)²/12 =
/// 0.52 of it more.
#[test]
fn a_diffusion_s_energy_input_does_not_know_the_trap() {
    let (k, c, tau, steps, runs) = (8.0, 0.5, 1.0, 64, 1024);
    let (hm, omega) = (1e-9, 2.5);
    let node = std::f64::consts::FRAC_PI_2 / k;
    let mut hessian = [[0.0; 3]; 3];
    hessian[2][2] = omega * omega / hm;
    let mut model = Atom::new(hm, Trap::Quadratic { hessian, gradient: [0.0, 0.0, -hessian[2][2] * node] }, vec![], 0.0, 0.0);
    model.folded = vec![(C64::real(0.5 * c), [0.0, 0.0, k]), (C64::real(0.5 * c), [0.0, 0.0, -k])];
    let energy = |t: &Trajectory| 0.5 * hm * t.kappa[2] * t.kappa[2] + 0.5 * hessian[2][2] * (t.r[2] - node).powi(2);
    let mut mean = 0.0;
    for index in 0..runs {
        let mut t = Trajectory::new(&model, 0, [0.0, 0.0, node], [0.0; 3], 41, index);
        let mut path = adapt::Path::new(t.seed, t.index, tau, 0);
        adapt::advance(&model, tau, &mut t, &mut step::Scratch::default(), &mut path, tau * steps as f64, &mut |_| {}).unwrap();
        assert_eq!((t.jumps, t.taken), (0, steps), "trajectory {index}: a node scatters nothing, and every step is whole");
        assert!((k * (t.r[2] - node)).abs() < 1e-3, "trajectory {index} stays at the node: {}", t.r[2] - node);
        mean += energy(&t) / runs as f64;
    }
    let input = 0.5 * hm * k * k * c * c * tau * steps as f64;
    eprintln!("the mean energy {mean:.4e} against the diffusion's input {input:.4e}");
    assert!((mean - input).abs() <= within(input / (runs as f64).sqrt(), 1.0), "the mean energy {mean} against the diffusion's input {input}");
}
