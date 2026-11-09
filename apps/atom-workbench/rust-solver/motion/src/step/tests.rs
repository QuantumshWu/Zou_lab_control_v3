//! The integrator against exact and analytic references on a test model of its trait (`step::Dynamics`): a two-level
//! atom (ground g, excited e of width Γ) in plane or standing waves of one frequency, in a quadratic or Gaussian trap that
//! holds both states alike; or the ground state alone, with or without a decay of its own.
//!
//! The model in the rotating frame, the coupling e ← g of each beam l c_l(r) = (Ω_l/2) e^{i(k_l·r + φ_l)}:
//! H = [[V, Σ c_l*], [Σ c_l, V − Δ − iΓ/2]], G = ∇H_h. Its Doppler frame K_g = 0, K_e = k₀ (the first beam's): the first
//! beam's coupling is static on the path, every other beam's a beat of wave vector k_l − k₀ (and its adjoint). The photon:
//! one class, u_a = √(Γ/3) ψ_e per lab axis (isotropic), landing in g; the jumps' kicks are the emission's −k n alone, of
//! a chosen |k| (zero where a test reads the Ehrenfest force).

use atom_motion::adapt::{self, Tolerances};
use atom_motion::linalg::{C64, Mat};
use atom_motion::model::{Beat, Cluster, Generator, Part, Photon, Point, Quadratic, Recoil, STATES};
use atom_motion::step::{self, Dynamics, Trajectory};

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
