//! The request (SPEC §9; contract §1.4): a single Rb87 atom in an optical tweezer, its internal state and its motion in
//! 3D -- the scene's rb87-full/2 model with its beams kept apart, their geometry on the lab axes, each beam's field at the
//! focus, the outside forms of every frequency, and the spontaneous emission resolved on the lab axes, as the
//! TypeScript model writes it (src/full-master-model.ts `MotionModel`, the only owner of atomic physics). The native
//! adapter parses the JSON into these types (every field of the motion block required, `electricField` and
//! `outsideForms` included: no request from before them is served); `Request::validate` holds the checks that need
//! the request alone (contract §2.4).
//!
//! At the position r (µm from the focus, the origin) the Hamiltonian in the model's frames is (rad/µs)
//!
//! ```text
//! H(r, t) = H0 + Σ_ω Σ_i (x_{ω,i}(r) − x_{ω,i}(0)) O_{ω,i}
//!              + Σ_l [ a_l(r) X_l e^{−iΔ_l t} + a_l(r)* R_l e^{+i(Δ_l + 2ν_frame) t} + h.c. ]
//! ```
//!
//! with a_l(r) the beam's amplitude relative to its focus value (field.rs), X_l and R_l its co- and counter-rotating
//! couplings at the focus, linear in its field E_l and E_l*, x_{ω,i} the coherence coordinates of the frequency's summed
//! field (field.rs) and O_{ω,i} its outside form: the shift through the states outside the model and the Rb⁺ core, of
//! which H0 already holds the focus value. Δ_l = 2π Σ_k frequency_index_l[k] frequencies_mhz[k] is the beam's
//! detuning from the excited states' frame. Spontaneous emission is Σ_a D[C_a] over the lowering dipole's lab-axis
//! components (`emission`).

use crate::linalg::{C64, Mat};

/// The internal states: the 32 Rb87 states of the model basis.
pub const STATES: usize = 32;
/// The ground manifold's states (5S₁/₂, F = 1 and 2).
pub const GROUND: usize = 8;

/// A state's manifold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Manifold {
    Ground,
    D1,
    D2,
}

/// A state's label `manifold-F-m` read (src/physics.ts: `${manifold}-${F}-${m}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    pub manifold: Manifold,
    pub f: u8,
    pub m: i8,
}

impl Level {
    /// The label read, or why it is not `manifold-F-m` with a manifold ground, D1 or D2.
    pub fn parse(label: &str) -> Result<Level, String> {
        let mut parts = label.splitn(3, '-');
        let manifold = match parts.next() {
            Some("ground") => Some(Manifold::Ground),
            Some("D1") => Some(Manifold::D1),
            Some("D2") => Some(Manifold::D2),
            _ => None,
        };
        match (manifold, parts.next().and_then(|f| f.parse().ok()), parts.next().and_then(|m| m.parse().ok())) {
            (Some(manifold), Some(f), Some(m)) => Ok(Level { manifold, f, m }),
            _ => Err(format!("The state label {label} is not manifold-F-m")),
        }
    }
    pub fn is_ground(&self) -> bool {
        self.manifold == Manifold::Ground
    }
}

/// One laser beam. Beams are kept apart: two counter-propagating beams of one frequency are two beams, each with its
/// own spatial phase; the beams of one frequency are one coherent field (SPEC §1.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Beam {
    /// The laser's name as the scene shows it (messages).
    pub name: String,
    /// The direction of propagation n on the lab axes x, y, z (the order of `Request::emission`); its length is
    /// read as a direction only (field.rs normalizes it).
    pub direction: [f64; 3],
    /// |k| = 2πν/c (rad/µm).
    pub wave_number: f64,
    /// A Gaussian beam focused at the origin: its waist, the 1/e² intensity radius at the focus (µm). None: a plane
    /// wave.
    pub waist: Option<f64>,
    /// The beam's frequency on the request's frequency axes (`Request::frequencies_mhz`): exact integer coordinates,
    /// so that equal frequencies are equal and no beat is rounded.
    pub frequency_index: Vec<i32>,
    /// X, the co-rotating excited ← ground coupling at the focus (rad/µs, 32 × 32, rows excited, columns ground).
    pub coupling: Mat,
    /// R, the counter-rotating coupling at the focus (rad/µs, 32 × 32, excited ← ground).
    pub counter_coupling: Mat,
    /// E_l, the beam's field at the focus (V/m) on the spherical components q = −1, 0, +1 of the quantization axis,
    /// its polarization and initial phase in it: X is linear in E_l, R in E_l* (SPEC §1.2).
    pub field: [C64; 3],
}

/// The outside shift of the beams of one frequency as a form in their summed field E (rad/µs/(V/m)²; src/spectroscopy.ts
/// `outsideFieldForm`): O(E) = Σ_i x_i(E) O_i with x = (|E_q|² for q = −1, 0, +1, then per pair p < q in the order
/// (−1, 0), (−1, +1), (0, +1) Re(E_p* E_q), Im(E_p* E_q)) -- field.rs `Coherence`. Each O_i is Hermitian.
#[derive(Clone, Debug, PartialEq)]
pub struct OutsideForm {
    pub frequency_index: Vec<i32>,
    pub matrices: [Mat; 9],
}

/// A motion solve: the model, the initial state and the output times.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// The model basis: 32 labels `manifold-F-m`, the eigenstates of H0.
    pub states: Vec<String>,
    /// H0 in the frames (rad/µs, 32 × 32): the magnetic field and the beams' shifts through the states outside the
    /// model at the focus, without the couplings; diagonal in the model basis.
    pub hamiltonian: Mat,
    /// The frequency each state's frame rotates at (MHz): 0 for the ground states, ν_frame for the excited ones.
    pub frame_frequencies_mhz: Vec<f64>,
    /// The frequency axes (MHz) the beams' `frequency_index` counts on.
    pub frequencies_mhz: Vec<f64>,
    pub beams: Vec<Beam>,
    /// One outside form per distinct frequency of the beams.
    pub outside_forms: Vec<OutsideForm>,
    /// The trap beam (an index into `beams`): the tweezer, a Gaussian beam whose focus is the origin.
    pub trap_beam: usize,
    /// The lowering dipole's components on the lab axes x, y, z (1/√µs, 32 × 32, ground ← excited): Σ_a D[C_a] is
    /// the model's spontaneous emission.
    pub emission: [Mat; 3],
    pub mass_kg: f64,
    /// The initial motion's temperature (µK).
    pub temperature_uk: f64,
    /// The initial internal density (32 × 32), diagonal in the model basis.
    pub initial_density: Mat,
    /// The output times (µs): from 0, evenly spaced.
    pub times_us: Vec<f64>,
}

/// Why a tweezer without a waist is refused (contract §2.4; shown in the app).
pub const NO_WAIST: &str = "光镊（trapBeamId 所指的光束）需要束腰：3D 运动计算聚焦在原子处的光镊中的单个原子";

impl Request {
    /// The states' levels, or why a label is not one.
    pub fn levels(&self) -> Result<Vec<Level>, String> {
        self.states.iter().map(|s| Level::parse(s)).collect()
    }

    /// The checks that need the request alone (contract §2.4), in the order a request is read: the states, H0, the
    /// frames, the beams and their forms, the tweezer, the emission, the atom, the initial state, the times. The
    /// messages the app shows are Chinese as today; the others name the field. The model's own checks (the couplings
    /// against the fields, the trap, the folds) are the model's.
    pub fn validate(&self) -> Result<(), String> {
        let square = |m: &Mat, what: &str| -> Result<(), String> {
            if m.rows != STATES || m.cols != STATES {
                return Err(format!("{what} must be a 32 x 32 matrix"));
            }
            if m.data.iter().any(|z| !z.is_finite()) {
                return Err(format!("{what} has a nonfinite entry"));
            }
            Ok(())
        };
        if self.states.len() != STATES {
            return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());
        }
        let ground: Vec<bool> = self.levels()?.iter().map(Level::is_ground).collect();
        let grounds = ground.iter().filter(|&&g| g).count();
        if grounds != GROUND {
            return Err(format!("The ground manifold has {grounds} states, not {GROUND}"));
        }
        // An operator of only (row, column) entries the rule allows: excited ← ground or ground ← excited.
        let only = |m: &Mat, allowed: &dyn Fn(usize, usize) -> bool| (0..STATES).all(|i| (0..STATES).all(|j| allowed(i, j) || m[(i, j)] == C64::ZERO));
        square(&self.hamiltonian, "H0")?;
        if !only(&self.hamiltonian, &|i, j| i == j) || (0..STATES).any(|i| self.hamiltonian[(i, i)].im != 0.0) {
            return Err("H0 must be diagonal and real in the model basis".into());
        }
        if self.frame_frequencies_mhz.len() != STATES || !self.frame_frequencies_mhz.iter().all(|x| x.is_finite()) {
            return Err("The motion block needs the 32 states' finite frame frequencies".into());
        }
        if !self.frequencies_mhz.iter().all(|x| x.is_finite()) {
            return Err("The frequency axes must be finite".into());
        }
        let axes = self.frequencies_mhz.len();
        for b in &self.beams {
            let norm = crate::linalg::norm3(b.direction);
            if !(norm.is_finite() && norm > 0.0 && b.wave_number.is_finite() && b.wave_number > 0.0)
                || b.waist.is_some_and(|w| !(w.is_finite() && w > 0.0))
                || b.frequency_index.len() != axes
            {
                return Err(format!(
                    "Beam {} needs a finite direction, a positive wave number, a positive waist if focused and an index on the {axes} frequency axes",
                    b.name
                ));
            }
            square(&b.coupling, &format!("Beam {}'s coupling", b.name))?;
            square(&b.counter_coupling, &format!("Beam {}'s counter-rotating coupling", b.name))?;
            let excitation = |i: usize, j: usize| !ground[i] && ground[j];
            if !only(&b.coupling, &excitation) || !only(&b.counter_coupling, &excitation) {
                return Err(format!("Beam {}'s couplings must be excited ← ground", b.name));
            }
            if b.field.iter().any(|z| !z.is_finite()) {
                return Err(format!("Beam {}'s electric field has a nonfinite entry", b.name));
            }
        }
        for f in &self.outside_forms {
            if f.frequency_index.len() != axes {
                return Err(format!("An outside form needs an index on the {axes} frequency axes"));
            }
            for m in &f.matrices {
                square(m, "An outside form's matrix")?;
            }
            if !self.beams.iter().any(|b| b.frequency_index == f.frequency_index) {
                return Err("An outside form's frequency has no beam".into());
            }
        }
        for b in &self.beams {
            let forms = self.outside_forms.iter().filter(|f| f.frequency_index == b.frequency_index).count();
            if forms != 1 {
                return Err(format!("Beam {}'s frequency has {forms} outside forms, not one", b.name));
            }
        }
        if self.trap_beam >= self.beams.len() {
            return Err("The motion block's trap beam names none of its beams".into());
        }
        if self.beams[self.trap_beam].waist.is_none() {
            return Err(NO_WAIST.into());
        }
        for (a, m) in self.emission.iter().enumerate() {
            square(m, &format!("Emission operator {a}"))?;
            if !only(m, &|i, j| ground[i] && !ground[j]) {
                return Err("The emission operators must be ground ← excited".into());
            }
        }
        if !(self.mass_kg.is_finite() && self.mass_kg > 0.0 && self.temperature_uk.is_finite() && self.temperature_uk > 0.0) {
            return Err("The motion block needs a positive temperature and mass".into());
        }
        square(&self.initial_density, "The initial density")?;
        if !only(&self.initial_density, &|i, j| i == j) {
            return Err("The initial density must be diagonal in the model basis".into());
        }
        if !((0..STATES).filter(|&i| ground[i]).map(|i| self.initial_density[(i, i)].re).sum::<f64>() > 0.0) {
            return Err("The initial density has no ground population".into());
        }
        self.check_times()
    }

    /// The output times: at least two, the first at 0, increasing and evenly spaced -- t_i = i·t_last/(n − 1) to the
    /// rounding of the grid's own construction and of this check (each t_i two operations from exact, i·Δ three:
    /// γ_6 of t_last, `units::rounding`).
    fn check_times(&self) -> Result<(), String> {
        let t = &self.times_us;
        if t.len() < 2 || t[0] != 0.0 {
            return Err("A 3D motion solve needs at least two times, the first at 0".into());
        }
        let last = t[t.len() - 1];
        let interval = last / (t.len() - 1) as f64;
        if !(last.is_finite() && interval > 0.0)
            || t.iter().enumerate().any(|(i, &x)| !((x - i as f64 * interval).abs() <= crate::units::rounding(6) * last))
        {
            return Err("A 3D motion solve needs evenly spaced times".into());
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A valid request of the contract's shape: the 32 states ground-F-m, D1-F-m, D2-F-m in the app's order; a
    /// diagonal H0; one tweezer (Gaussian, π-polarized) and a plane wave of another frequency; their forms; emission
    /// ground ← excited; the initial state ground-2-0; 101 times over 10 µs. Its numbers are placeholders of the right
    /// shape, not physics: the request's checks read shapes and structure.
    pub(crate) fn sample() -> Request {
        let mut states = vec![];
        for (manifold, fs) in [("ground", [1, 2]), ("D1", [1, 2])] {
            for f in fs {
                for m in -(f as i32)..=f as i32 {
                    states.push(format!("{manifold}-{f}-{m}"));
                }
            }
        }
        for f in 0..=3 {
            for m in -(f as i32)..=f as i32 {
                states.push(format!("D2-{f}-{m}"));
            }
        }
        let ground: Vec<bool> = states.iter().map(|s| s.starts_with("ground")).collect();
        let zero = || Mat::zeros(STATES, STATES);
        let coupling = |scale: f64| Mat::from_fn(STATES, STATES, |i, j| if !ground[i] && ground[j] { C64::real(scale * (1 + i + j) as f64) } else { C64::ZERO });
        let beam = |name: &str, waist: Option<f64>, index: i32, field: [C64; 3]| Beam {
            name: name.into(),
            direction: [0.0, 0.0, 1.0],
            wave_number: 7.37,
            waist,
            frequency_index: vec![index],
            coupling: coupling(0.01),
            counter_coupling: coupling(0.001),
            field,
        };
        let form = |index: i32| OutsideForm { frequency_index: vec![index], matrices: std::array::from_fn(|_| zero()) };
        let mut density = zero();
        let start = states.iter().position(|s| s == "ground-2-0").unwrap();
        density[(start, start)] = C64::ONE;
        Request {
            hamiltonian: Mat::diagonal(&(0..STATES).map(|i| C64::real(i as f64)).collect::<Vec<_>>()),
            frame_frequencies_mhz: ground.iter().map(|&g| if g { 0.0 } else { 3.8e8 }).collect(),
            frequencies_mhz: vec![1.0],
            beams: vec![
                beam("光镊", Some(0.8), -4, [C64::ZERO, C64::real(1e5), C64::ZERO]),
                beam("探测光", None, 0, [C64::real(10.0), C64::ZERO, C64::ZERO]),
            ],
            outside_forms: vec![form(-4), form(0)],
            trap_beam: 0,
            emission: std::array::from_fn(|_| Mat::from_fn(STATES, STATES, |i, j| if ground[i] && !ground[j] { C64::real(0.1) } else { C64::ZERO })),
            mass_kg: 1.4432e-25,
            temperature_uk: 30.0,
            initial_density: density,
            times_us: (0..101).map(|i| 10.0 * i as f64 / 100.0).collect(),
            states,
        }
    }

    /// The sample passes; each structural fault of contract §2.4 is refused with its message: a tweezer without a
    /// waist (the app's Chinese text), a coupling off excited ← ground, a beam frequency without its outside form, a
    /// non-diagonal H0 or initial density, no ground population, uneven or misplaced times, a bad label.
    #[test]
    fn a_request_is_refused_for_each_fault_of_its_own() {
        let ok = sample();
        assert_eq!(ok.validate(), Ok(()));
        assert_eq!(ok.levels().unwrap()[3], Level { manifold: Manifold::Ground, f: 2, m: -2 });
        let refused = |change: &dyn Fn(&mut Request), says: &str| {
            let mut r = sample();
            change(&mut r);
            let e = r.validate().expect_err(says);
            assert!(e.contains(says), "{e}");
        };
        refused(&|r| r.beams[0].waist = None, NO_WAIST);
        refused(&|r| r.beams[1].coupling[(0, 1)] = C64::real(1.0), "excited ← ground");
        refused(&|r| r.outside_forms.pop().map(|_| ()).unwrap(), "outside forms, not one");
        refused(&|r| r.outside_forms.push(r.outside_forms[0].clone()), "outside forms, not one");
        refused(&|r| r.hamiltonian[(0, 1)] = C64::real(1e-30), "H0 must be diagonal");
        refused(&|r| r.initial_density[(4, 5)] = C64::real(1e-30), "diagonal in the model basis");
        refused(
            &|r| {
                r.initial_density = Mat::zeros(STATES, STATES);
                r.initial_density[(20, 20)] = C64::ONE;
            },
            "no ground population",
        );
        refused(&|r| r.times_us[50] += 1e-6, "evenly spaced");
        refused(&|r| r.times_us[0] = 1e-3, "the first at 0");
        refused(&|r| r.states[0] = "ground-x-0".into(), "not manifold-F-m");
        refused(&|r| r.beams[1].field[2] = C64::new(f64::NAN, 0.0), "electric field");
        refused(&|r| r.emission[1][(20, 0)] = C64::real(1.0), "ground ← excited");
    }
}
