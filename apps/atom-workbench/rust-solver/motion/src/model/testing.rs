//! What the model's tests share (compiled with the tests alone): the reference scenes' requests, the units that fold a
//! light's drives at the atom, a scene's model closed as the solve closes it, a plan's near forms from its table or the
//! exact elimination, the generator's side of the hazard's identity, and normal deviates.
use super::assemble::{NearValues, Plan};
use super::tabulate::NearTable;
use super::{Component, Model, Reach, Trap, Unit, GROUND};
use crate::field::{Amplitude, Light};
use crate::linalg::{inner, norm_sqr, Mat, C64};
use crate::random::Stream;
use crate::request::Request;

/// A scene's request through the one parser: a reference scene by name (refs/scenes, the app's native request), or any
/// native request file by its path.
pub(crate) fn scene(name: &str) -> Request {
    let path = if name.ends_with(".json") { std::path::PathBuf::from(name) } else { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json")) };
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}

/// The units whose keeping folds every drive of the co-rotating components of the frequencies `of` with the levels at the
/// atom (`Unit::Bare`: kept, a drive's fold takes the levels there).
pub(crate) fn at_the_atom(request: &Request, light: &Light, of: &[usize]) -> Vec<Unit> {
    let bare = Model::new(request, light, &[]).unwrap();
    of.iter().flat_map(|&frequency| bare.bare_units(Component { frequency, counter: false })).collect()
}

/// The rate at which a generator H lowers a state's norm, ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖²: by the decay's construction (photon.rs)
/// the hazard, Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² of the photons. The anti-Hermitian part i(H − H†) is formed entry by entry before
/// it meets ψ, so that the Hermitian part's scale (the states' energies) does not round the rate.
pub(crate) fn decay_rate(h: &Mat, psi: &[C64]) -> f64 {
    let decay = Mat::from_fn(h.rows, h.cols, |x, y| C64::I * (h[(x, y)] - h[(y, x)].conj()));
    inner(psi, &decay.apply(psi)).re / norm_sqr(psi)
}

/// The steps of the difference quotients (`converged`): 0.1 µm halved seven times, down to 8·10⁻⁴ µm.
pub(crate) fn steps() -> Vec<f64> {
    (0..8).map(|k| 0.1 / f64::powi(2.0, k)).collect()
}

/// A derivative of order `order` (1 or 2) from its difference quotients at the steps h₀/2^k (`steps`), the step chosen by
/// their convergence: the truncation falls as h², the values' rounding ε grows as ε/h^order, and what the values hold
/// besides (their rounding is not smooth) jitters the quotients at any step. Each quotient's error is the largest
/// difference from the quotients within two steps of it, and no less than ε/h^order at its step, ε read off the finest
/// two quotients (rounding's alone there); the estimate is the quotient of the least error, with that error. (Two
/// neighbours alone agree by chance.)
pub(crate) fn converged(quotients: &[f64], order: i32) -> (f64, f64) {
    let hs = steps();
    let n = quotients.len();
    let rounding = (quotients[n - 1] - quotients[n - 2]).abs() * hs[n - 1].powi(order);
    (2..n - 2)
        .map(|k| (quotients[k], (k - 2..=k + 2).map(|j| (quotients[j] - quotients[k]).abs()).fold(rounding / hs[k].powi(order), crate::linalg::larger)))
        .fold((quotients[0], f64::INFINITY), |best, x| if x.1 < best.1 { x } else { best })
}

/// Whether a derivative `jet` is its values' difference (`converged`: estimate and error) on its own scale `scale`: within
/// four times the difference's error and 10⁻⁸ of that scale -- a gradient's part its own size, a Hessian's diagonal entry
/// its own, an off-diagonal one the geometric mean of its two axes' (`own`).
pub(crate) fn agrees(jet: f64, (estimate, error): (f64, f64), scale: f64) -> bool {
    (jet - estimate).abs() <= 4.0 * error + 1e-8 * crate::linalg::largest(scale, [estimate.abs(), jet.abs()])
}

/// The own scale of a Hessian's entry (a, c) (`agrees`): its own size, and for a ≠ c the geometric mean of the two axes'
/// curvatures -- the scale a symmetric form's off-diagonal entry is a part of.
pub(crate) fn own(hessian: &dyn Fn(usize, usize) -> f64, a: usize, c: usize) -> f64 {
    crate::linalg::larger(hessian(a, c).abs(), (hessian(a, a) * hessian(c, c)).abs().sqrt())
}

/// A standard normal deviate from two uniform ones (Box–Muller).
pub(crate) fn normal(stream: &mut Stream) -> f64 {
    let (u, v) = (stream.uniform(), stream.uniform());
    (-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

impl Model {
    /// The units that fold a component's drives with the bare levels (`Unit::Bare`), whatever their level in this model:
    /// kept, each of them is folded with the levels at the atom.
    pub(crate) fn bare_units(&self, component: Component) -> Vec<Unit> {
        self.driven.iter().filter(|d| d.0 == component).map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).collect()
    }

    /// A scene's model as the old solve closed it (`closed`: the prior at the atoms of the reach of `temperature_uk`,
    /// the ground state `state` the initial state) from the units `kept`: the model, the units it keeps, its prepared trap
    /// and its reach for that state.
    pub(crate) fn closed_at(request: &Request, light: &Light, mut kept: Vec<Unit>, state: usize, temperature_uk: f64) -> (Model, Vec<Unit>, Trap, Reach) {
        let model = closed(request, light, &mut kept, &[(state, 1.0)], temperature_uk).expect("the closed model");
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        (model, kept, trap, reach)
    }

    /// A plan's near forms at a point, the sections asked each to its order (0 or 1): from its table where one is given and
    /// reads there (tabulate.rs `Model::near_tabulated`), else the exact elimination's to the largest order (assemble.rs
    /// `Model::near_exact`) -- as the run's middles take them.
    pub(crate) fn near_values(&self, plan: &Plan, amplitudes: &[Amplitude], sections: &[(std::ops::Range<usize>, usize)], table: Option<&NearTable>) -> NearValues {
        table.and_then(|table| self.near_tabulated(table, amplitudes, sections)).unwrap_or_else(|| self.near_exact(plan, amplitudes, sections.iter().map(|s| s.1).max().unwrap_or(0)))
    }
}

/// The old model of the units `kept` and of every unit whose fold does not exist where the reach's atoms are, added to
/// `kept` (the old solve's model prior, which the solve on the fold at the atom no longer takes: the old model's tests
/// alone close their models so): the units that fail kept, and the model they make read again, until none fails; and
/// the fold's series constants at the same atoms (elim.rs `Series`), the excited block that stops them made explicit, or
/// the request refused (elim.rs `Model::stop_message`).
pub(crate) fn closed(request: &Request, light: &Light, kept: &mut Vec<Unit>, initial: &[(usize, f64)], temperature: f64) -> Result<Model, String> {
    loop {
        let mut model = Model::new(request, light, kept)?;
        let points = atoms(&model, request, light, initial, temperature)?;
        let more: Vec<Unit> = model.prior(&points).into_iter().filter(|u| !kept.contains(u)).collect();
        if more.is_empty() {
            model.referenced(&points);
            if model.series.exists() {
                return Ok(model);
            }
            match model.series_block().filter(|unit| !kept.contains(unit)) {
                Some(unit) => kept.push(unit),
                None => return Err(model.stop_message()),
            }
            continue;
        }
        kept.extend(more);
    }
}

/// The atoms an old model's prior reads its gates at (`closed`): the draws of the thermal ensemble at `temperature` (µK)
/// in the initial components' wells of the old model's prepared potential (`Prepared`), the pilot's numbers and the
/// run's seed, each its place and its velocity v = (ħ/m)κ.
pub(crate) fn atoms(model: &Model, request: &Request, light: &Light, initial: &[(usize, f64)], temperature: f64) -> Result<Vec<([f64; 3], [f64; 3])>, String> {
    use rayon::prelude::*;
    let hm = crate::units::hbar_over_mass(request.mass_kg);
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(model, light, std::array::from_fn(|b| if b == s { 1.0 } else { 0.0 }))).collect::<Result<_, _>>()?;
    let ensemble = crate::ensemble::Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn crate::ensemble::Well)).collect::<Vec<_>>(), temperature, hm)?;
    let first = crate::units::MOST_TRAJECTORIES;
    Ok((first..first + crate::solve::pilot_size())
        .into_par_iter()
        .map(|index| {
            let start = ensemble.draw(crate::solve::SEED, index);
            (start.r, start.kappa.map(|k| hm * k))
        })
        .collect())
}

/// One initial component's preparation well of an old model (ensemble.rs `Well`): its prepared potential, trap and
/// envelope.
struct Prepared<'a> {
    model: &'a Model,
    light: &'a Light,
    populations: [f64; GROUND],
    lights: super::Lights,
    trap: Trap,
    envelope: super::Envelope,
}
impl<'a> Prepared<'a> {
    fn new(model: &'a Model, light: &'a Light, populations: [f64; GROUND]) -> Result<Prepared<'a>, String> {
        Ok(Prepared { trap: model.trap(&populations, &model.prepared())?, envelope: model.envelope(&populations), lights: model.prepared(), model, light, populations })
    }
}
impl crate::ensemble::Well for Prepared<'_> {
    fn potential(&self, r: [f64; 3]) -> (f64, [f64; 3]) {
        let q = self.model.expansion(r, &self.populations, &self.lights);
        (q.value, q.gradient)
    }
    fn trap(&self) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
        (self.trap.centre, self.trap.axes, self.trap.frequencies)
    }
    fn envelope(&self) -> (Vec<crate::field::Geometry>, f64) {
        (self.envelope.beams.iter().map(|&l| self.light.beams[l]).collect(), self.envelope.peak())
    }
}
