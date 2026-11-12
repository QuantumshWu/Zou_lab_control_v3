//! The request (SPEC §9; contract §1.3-§1.4): a single Rb87 atom in an optical tweezer, its internal state and its
//! motion in 3D -- the scene's rb87-full/2 model with its beams kept apart, their geometry on the lab axes, each beam's
//! field at the focus, the outside forms of every frequency, and the spontaneous emission resolved on the lab axes, as
//! the TypeScript model writes it (src/full-master-model.ts `MotionModel`, the only owner of atomic physics).
//!
//! `Request::from_native` is the one mapping from the native JSONL request to these types: the options, the model and
//! its motion block, every field of the motion block required (the contract's `MotionModel`; those the engine does not
//! read -- the magnetic field, a beam's intensity and its own outside shift, all of them in H0, the couplings and the
//! outside forms already -- for their presence alone), unknown fields refused in the motion block, its beams, their field
//! components and its outside forms. It holds the checks of the JSON layer: types, presence, unknown fields, and the
//! shapes without which no request is built. `Request::validate` holds those of the request itself (contract §2.4), and
//! `from_native` ends with it. The app translates some of the English messages word for word (src/solver-words.ts: the
//! mean, the duration, the samples) and shows every other one as it is written.
//!
//! The model basis holds the ground manifold first: state s is a ground state exactly when s < `GROUND` (`validate`
//! checks the labels against it once; everything else in the crate goes by the index).
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
use crate::units::{rounding, LONGEST_TIME_US, MOST_SAMPLES};
use serde_json::{Map, Value};

/// The internal states: the 32 Rb87 states of the model basis.
pub const STATES: usize = 32;
/// The ground manifold's states (5S₁/₂, F = 1 and 2).
pub const GROUND: usize = 8;

/// Why a motion block along an axis or in a plane is refused (contract §2.4; shown in the app).
pub const ONLY_3D: &str = "运动只按三维计算（半经典）：不计算沿单个轴或在平面内的运动";
/// Why a tweezer without a waist is refused (contract §2.4; shown in the app).
pub const NO_WAIST: &str = "光镊需要束腰（按功率与束腰或按数值孔径给出的聚焦光束）：运动计算的是聚焦在原子处的光镊中的单个原子";
/// Why an initial state with population in an excited state is refused (shown in the app): the initial ensemble is each
/// ground state's atoms in its own prepared well (SPEC §2.5), and no prepared well describes an excited state's motion.
pub const ON_GROUND: &str = "运动计算的初态需在基态上：初态在激发态上有布居";
/// Why a motion request with the mean is refused (translated by the app).
const TRAJECTORY_ALONE: &str = "A motion request is a trajectory alone: meanOnly and includeMean are false";
/// Why the atom's description is refused.
const ATOM: &str = "The motion block needs a positive temperature and mass";
/// Why a model is not the full one.
const FULL_MODEL: &str = "rb87-full/2 requires the complete 32-state Rb87 model";
/// The initial density in the messages of its matrix's checks.
const INITIAL_DENSITY: &str = "The initial density";

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
    /// The beam's id in the scene (the motion block's `trapBeamId` names one; the English messages name it).
    pub id: String,
    /// The laser's name as the scene shows it (the Chinese messages).
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

// ---------------------------------------------------------------------------------------------------------------------
// Reading the native line's JSON: types, presence, unknown fields.

/// The motion block's fields, a beam's, an outside form's and a field component's (contract §1.4): each of these objects
/// refuses any other field.
const MOTION_FIELDS: [&str; 10] =
    ["axis", "trapBeamId", "temperatureUK", "massKg", "fieldVectorG", "hamiltonian", "frameFrequenciesMHz", "beams", "outsideForms", "emission"];
const BEAM_FIELDS: [&str; 11] = [
    "id",
    "name",
    "direction",
    "waveNumberPerUm",
    "waistUm",
    "intensityMWcm2",
    "frequencyIndex",
    "coupling",
    "counterCoupling",
    "outsideShift",
    "electricField",
];
const FORM_FIELDS: [&str; 2] = ["frequencyIndex", "matrices"];
const AMPLITUDE_FIELDS: [&str; 2] = ["re", "im"];

fn invalid(why: impl std::fmt::Display) -> String {
    format!("Invalid request: {why}")
}
/// A JSON object; with `known`, one of those fields alone.
fn object<'a>(v: &'a Value, what: &str, known: Option<&[&str]>) -> Result<&'a Map<String, Value>, String> {
    let o = v.as_object().ok_or_else(|| invalid(format!("{what} must be an object")))?;
    if let Some(k) = known.and_then(|known| o.keys().find(|k| !known.contains(&k.as_str()))) {
        return Err(invalid(format!("unknown field `{k}` in {what}")));
    }
    Ok(o)
}
/// A field that must be present; null counts as absent.
fn get<'a>(o: &'a Map<String, Value>, key: &str, what: &str) -> Result<&'a Value, String> {
    o.get(key).filter(|v| !v.is_null()).ok_or_else(|| invalid(format!("missing field `{key}` in {what}")))
}
fn number(v: &Value, what: &str) -> Result<f64, String> {
    v.as_f64().ok_or_else(|| invalid(format!("{what} must be a number")))
}
fn array<'a>(v: &'a Value, what: &str) -> Result<&'a Vec<Value>, String> {
    v.as_array().ok_or_else(|| invalid(format!("{what} must be an array")))
}
fn numbers(v: &Value, what: &str) -> Result<Vec<f64>, String> {
    array(v, what)?.iter().map(|x| number(x, what)).collect()
}
fn triple(v: &Value, what: &str) -> Result<[f64; 3], String> {
    numbers(v, what)?.try_into().map_err(|_| invalid(format!("{what} must have three entries")))
}
fn text<'a>(v: &'a Value, what: &str) -> Result<&'a str, String> {
    v.as_str().ok_or_else(|| invalid(format!("{what} must be a string")))
}
fn integers(v: &Value, what: &str) -> Result<Vec<i32>, String> {
    array(v, what)?
        .iter()
        .map(|x| x.as_i64().and_then(|i| i32::try_from(i).ok()).ok_or_else(|| invalid(format!("{what} must hold integers"))))
        .collect()
}
fn count(v: &Value, what: &str) -> Result<usize, String> {
    v.as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| invalid(format!("{what} must be a nonnegative integer")))
}
/// A 32 × 32 matrix from its real and imaginary parts, each a row-major array of 32 × 32 numbers.
fn matrix(re: &Value, im: &Value, what: &str) -> Result<Mat, String> {
    let (re, im) = (numbers(re, what)?, numbers(im, what)?);
    if re.len() != STATES * STATES || im.len() != STATES * STATES {
        return Err(format!("{what} must be a 32 x 32 row-major matrix"));
    }
    Ok(Mat { rows: STATES, cols: STATES, data: re.iter().zip(&im).map(|(&a, &b)| C64::new(a, b)).collect() })
}
/// An operator: the matrix of an object's `real` and `imag` (nothing else of the object is read).
fn operator(v: &Value, what: &str) -> Result<Mat, String> {
    let o = object(v, what, None)?;
    matrix(get(o, "real", what)?, get(o, "imag", what)?, what)
}

impl Request {
    /// The request of a native JSONL line (contract §1.1-§1.4): `{options, model: {dimension, states, initialReal,
    /// initialImag, frequencyCoordinates, motion, …}, …}`. The options: a trajectory alone (`includeMean`
    /// and `meanOnly` false; `includeMean` absent is true, as the native protocol reads it), `durationUs` in
    /// (0, `LONGEST_TIME_US`], `samples` in [2, `MOST_SAMPLES`], `device` "gpu" or "cpu" if given (either is computed on
    /// the CPU). The times are linspace(0, durationUs, samples), t_i = durationUs·i/(samples − 1), the population
    /// core's grid. The initial density is `initialReal` + i `initialImag`, both required (motion does not read
    /// `initialIndex`, contract §1.4). Fields of the line, the options and the model that motion does not read are left
    /// to the native worker.
    pub fn from_native(line: &Value) -> Result<Request, String> {
        let line = object(line, "the request", None)?;
        let empty = Map::new();
        let options = match line.get("options") {
            Some(v) => object(v, "options", None)?,
            None => &empty,
        };
        let flag = |key: &str, absent: bool| match options.get(key) {
            None => Ok(absent),
            Some(v) => v.as_bool().ok_or_else(|| invalid(format!("options.{key} must be a boolean"))),
        };
        if flag("meanOnly", false)? || flag("includeMean", true)? {
            return Err(TRAJECTORY_ALONE.into());
        }
        if let Some(device) = options.get("device") {
            if !matches!(device.as_str(), Some("gpu") | Some("cpu")) {
                return Err(invalid("options.device must be \"gpu\" or \"cpu\""));
            }
        }
        let duration = number(get(options, "durationUs", "options")?, "options.durationUs")?;
        if !(duration.is_finite() && duration > 0.0) {
            return Err("The trajectory duration must be a positive finite time".into());
        }
        if duration > LONGEST_TIME_US {
            return Err(format!("The trajectory duration is limited to {} s (requested {} s)", LONGEST_TIME_US / 1e6, duration / 1e6));
        }
        let samples = count(get(options, "samples", "options")?, "options.samples")?;
        if !(2..=MOST_SAMPLES).contains(&samples) {
            return Err(format!("A trajectory has 2 to {MOST_SAMPLES} samples (requested {samples})"));
        }
        let model = object(get(line, "model", "the request")?, "the model", None)?;
        let states: Vec<String> = match model.get("states") {
            Some(v) => array(v, "model.states")?.iter().map(|s| text(s, "model.states").map(str::to_owned)).collect::<Result<_, _>>()?,
            None => vec![],
        };
        if count(get(model, "dimension", "the model")?, "model.dimension")? != STATES {
            return Err(FULL_MODEL.into());
        }
        let frequencies_mhz = match model.get("frequencyCoordinates").filter(|v| !v.is_null()) {
            Some(c) => match object(c, "model.frequencyCoordinates", None)?.get("frequenciesMHz").filter(|v| !v.is_null()) {
                Some(f) => numbers(f, "model.frequencyCoordinates.frequenciesMHz")?,
                None => vec![],
            },
            None => vec![],
        };
        let motion = object(get(model, "motion", "the model")?, "the motion block", Some(&MOTION_FIELDS))?;
        let field = |key: &str| get(motion, key, "the motion block");
        if text(field("axis")?, "motion.axis")? != "3d" {
            return Err(ONLY_3D.into());
        }
        let (temperature_uk, mass_kg) = (number(field("temperatureUK")?, "motion.temperatureUK")?, number(field("massKg")?, "motion.massKg")?);
        // The contract's field that the engine does not read (H0 holds the magnetic field): present, no more.
        field("fieldVectorG")?;
        let frame_frequencies_mhz = numbers(field("frameFrequenciesMHz")?, "motion.frameFrequenciesMHz")?;
        let beams = array(field("beams")?, "motion.beams")?
            .iter()
            .map(|b| {
                let o = object(b, "a beam", Some(&BEAM_FIELDS))?;
                let id = text(get(o, "id", "a beam")?, "a beam's id")?.to_owned();
                let what = format!("beam {id}");
                let at = |key: &str| get(o, key, &what);
                // The contract's fields that the engine does not read (the couplings and the field hold the intensity, the
                // outside forms every beam's outside shift): present, no more.
                at("intensityMWcm2")?;
                at("outsideShift")?;
                let field = array(at("electricField")?, "a beam's electricField")?
                    .iter()
                    .map(|z| {
                        let z = object(z, "an electric field component", Some(&AMPLITUDE_FIELDS))?;
                        Ok(C64::new(number(get(z, "re", "a component")?, "re")?, number(get(z, "im", "a component")?, "im")?))
                    })
                    .collect::<Result<Vec<C64>, String>>()?
                    .try_into()
                    .map_err(|_| invalid(format!("{what}'s electricField must have three components")))?;
                Ok(Beam {
                    name: text(at("name")?, "a beam's name")?.to_owned(),
                    direction: triple(at("direction")?, "a beam's direction")?,
                    wave_number: number(at("waveNumberPerUm")?, "a beam's waveNumberPerUm")?,
                    waist: o.get("waistUm").filter(|v| !v.is_null()).map(|w| number(w, "a beam's waistUm")).transpose()?,
                    frequency_index: integers(at("frequencyIndex")?, "a beam's frequencyIndex")?,
                    coupling: operator(at("coupling")?, "A beam's coupling")?,
                    counter_coupling: operator(at("counterCoupling")?, "A beam's counter-rotating coupling")?,
                    field,
                    id,
                })
            })
            .collect::<Result<Vec<Beam>, String>>()?;
        let outside_forms = array(field("outsideForms")?, "motion.outsideForms")?
            .iter()
            .map(|f| {
                let o = object(f, "an outside form", Some(&FORM_FIELDS))?;
                let frequency_index = integers(get(o, "frequencyIndex", "an outside form")?, "an outside form's frequencyIndex")?;
                let matrices = array(get(o, "matrices", "an outside form")?, "an outside form's matrices")?
                    .iter()
                    .map(|m| operator(m, "An outside form's matrix"))
                    .collect::<Result<Vec<Mat>, String>>()?;
                Ok(OutsideForm { frequency_index, matrices: matrices.try_into().map_err(|_| "An outside form has nine matrices".to_string())? })
            })
            .collect::<Result<Vec<OutsideForm>, String>>()?;
        let trap_id = text(field("trapBeamId")?, "motion.trapBeamId")?;
        let trap_beam = beams.iter().position(|b| b.id == trap_id).ok_or_else(|| format!("The motion block's trapBeamId {trap_id} names none of its beams"))?;
        let emission: Vec<Mat> = array(field("emission")?, "motion.emission")?.iter().map(|m| operator(m, "An emission operator")).collect::<Result<_, _>>()?;
        let emission: [Mat; 3] = emission.try_into().map_err(|_| "The motion block has three emission operators, one per lab axis".to_string())?;
        let initial_density = matrix(get(model, "initialReal", "the model")?, get(model, "initialImag", "the model")?, INITIAL_DENSITY)?;
        let request = Request {
            states,
            hamiltonian: operator(field("hamiltonian")?, "The motion block's H0")?,
            frame_frequencies_mhz,
            frequencies_mhz,
            beams,
            outside_forms,
            trap_beam,
            emission,
            mass_kg,
            temperature_uk,
            initial_density,
            times_us: (0..samples).map(|i| duration * i as f64 / (samples - 1) as f64).collect(),
        };
        request.validate()?;
        Ok(request)
    }

    /// The states' levels, or why a label is not one.
    pub fn levels(&self) -> Result<Vec<Level>, String> {
        self.states.iter().map(|s| Level::parse(s)).collect()
    }

    /// The checks of the request itself (contract §2.4), in the order a request is read: the states, H0, the
    /// frames, the beams and their forms, the tweezer, the emission, the atom, the initial state, the times. The
    /// model's own checks (the couplings against the fields, the trap, the folds) are the model's.
    pub fn validate(&self) -> Result<(), String> {
        let square = |m: &Mat, what: &str| -> Result<(), String> {
            if m.rows != STATES || m.cols != STATES {
                return Err(format!("{what} must be a 32 x 32 row-major matrix"));
            }
            if m.data.iter().any(|z| !z.is_finite()) {
                return Err(format!("{what} has a nonfinite entry"));
            }
            Ok(())
        };
        if self.states.len() != STATES {
            return Err(FULL_MODEL.into());
        }
        // The basis holds the ground manifold first (module documentation): from here on a ground state is an index.
        if let Some(s) = self.levels()?.iter().enumerate().position(|(s, level)| level.is_ground() != (s < GROUND)) {
            return Err(format!("The model basis holds its {GROUND} ground states first and no other: {} is out of place", self.states[s]));
        }
        // An operator of only (row, column) entries the rule allows: excited ← ground or ground ← excited.
        let only = |m: &Mat, allowed: &dyn Fn(usize, usize) -> bool| (0..STATES).all(|i| (0..STATES).all(|j| allowed(i, j) || m[(i, j)] == C64::ZERO));
        square(&self.hamiltonian, "The motion block's H0")?;
        if !only(&self.hamiltonian, &|i, j| i == j) || (0..STATES).any(|i| self.hamiltonian[(i, i)].im != 0.0) {
            return Err("H0 must be diagonal and real in the model basis".into());
        }
        if self.frame_frequencies_mhz.len() != STATES || !self.frame_frequencies_mhz.iter().all(|x| x.is_finite()) {
            return Err("The motion block needs the 32 states' frame frequencies".into());
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
                return Err(format!("Beam {} needs a finite direction, a positive wave number, a positive waist if focused and an index on the {axes} frequency axes", b.id));
            }
            square(&b.coupling, "A beam's coupling")?;
            square(&b.counter_coupling, "A beam's counter-rotating coupling")?;
            let excitation = |i: usize, j: usize| i >= GROUND && j < GROUND;
            if !only(&b.coupling, &excitation) || !only(&b.counter_coupling, &excitation) {
                return Err(format!("Beam {}'s couplings must be excited ← ground", b.id));
            }
            if b.field.iter().any(|z| !z.is_finite()) {
                return Err(format!("Beam {}'s electric field has a nonfinite entry", b.id));
            }
            if b.field.iter().all(|&z| z == C64::ZERO) {
                return Err(format!("Beam {}'s electric field is zero: the app sends only the beams that shine", b.id));
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
                return Err(format!("Beam {}'s frequency has {forms} outside forms, not one", b.id));
            }
        }
        if self.trap_beam >= self.beams.len() {
            return Err("The motion block's trap beam names none of its beams".into());
        }
        if self.beams[self.trap_beam].waist.is_none() {
            return Err(NO_WAIST.into());
        }
        for m in &self.emission {
            square(m, "An emission operator")?;
            if !only(m, &|i, j| i < GROUND && j >= GROUND) {
                return Err("The emission operators must be ground ← excited".into());
            }
        }
        if !(self.mass_kg.is_finite() && self.mass_kg > 0.0 && self.temperature_uk.is_finite() && self.temperature_uk > 0.0) {
            return Err(ATOM.into());
        }
        square(&self.initial_density, INITIAL_DENSITY)?;
        self.check_density()?;
        self.check_times()
    }

    /// The initial density (a finite 32 × 32 matrix): its trace one and Hermitian, positive semidefinite, diagonal in the
    /// model basis, on the ground states (contract §2.4). A density's entries are assembled from at most 32 terms of size
    /// at most one, so its trace and its Hermiticity hold to γ₃₂ (`units::rounding`); a diagonal density's eigenvalues are
    /// its diagonal, nonnegative to the same rounding; and to the same rounding an excited state's entry is no
    /// population -- what a normalisation's round-off leaves there is not refused, a population is (`ON_GROUND`). The
    /// ground states then hold all of the trace, so some ground state is populated.
    fn check_density(&self) -> Result<(), String> {
        let rho = &self.initial_density;
        let within = rounding(STATES);
        let hermitian = (0..STATES).all(|i| (0..STATES).all(|j| (rho[(i, j)] - rho[(j, i)].conj()).abs() <= within));
        if !((rho.trace() - C64::ONE).abs() <= within && hermitian) {
            return Err("Initial density fails the trace/Hermitian checks".into());
        }
        if (0..STATES).any(|i| (0..STATES).any(|j| i != j && rho[(i, j)] != C64::ZERO)) {
            return Err("The initial density must be diagonal in the model basis".into());
        }
        if (0..STATES).any(|i| rho[(i, i)].re < -within) {
            return Err("Initial density is not positive semidefinite".into());
        }
        if (GROUND..STATES).any(|s| rho[(s, s)].re > within) {
            return Err(ON_GROUND.into());
        }
        Ok(())
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
        if !(last.is_finite() && interval > 0.0) || t.iter().enumerate().any(|(i, &x)| !((x - i as f64 * interval).abs() <= rounding(6) * last)) {
            return Err("A 3D motion solve needs evenly spaced times".into());
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

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
        let beam = |id: &str, name: &str, waist: Option<f64>, index: i32, field: [C64; 3]| Beam {
            id: id.into(),
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
                beam("tweezer", "光镊", Some(0.8), -4, [C64::ZERO, C64::real(1e5), C64::ZERO]),
                beam("probe", "探测光", None, 0, [C64::real(10.0), C64::ZERO, C64::ZERO]),
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

    /// The native line of a request, as the app writes it (contract §1.1): the options of a trajectory alone, the model
    /// with the fields motion does not read, and the motion block.
    pub(crate) fn native(r: &Request) -> Value {
        let operator = |m: &Mat| json!({"real": m.data.iter().map(|z| z.re).collect::<Vec<_>>(), "imag": m.data.iter().map(|z| z.im).collect::<Vec<_>>()});
        let beams: Vec<Value> = r
            .beams
            .iter()
            .map(|b| {
                json!({"id": b.id, "name": b.name, "direction": b.direction, "waveNumberPerUm": b.wave_number, "waistUm": b.waist,
                    "intensityMWcm2": 1.0, "frequencyIndex": b.frequency_index, "coupling": operator(&b.coupling),
                    "counterCoupling": operator(&b.counter_coupling), "outsideShift": operator(&Mat::zeros(STATES, STATES)),
                    "electricField": b.field.iter().map(|z| json!({"re": z.re, "im": z.im})).collect::<Vec<_>>()})
            })
            .collect();
        let forms: Vec<Value> =
            r.outside_forms.iter().map(|f| json!({"frequencyIndex": f.frequency_index, "matrices": f.matrices.iter().map(operator).collect::<Vec<_>>()})).collect();
        let samples = r.times_us.len();
        json!({"id": 7, "modelSchema": "rb87-full/2", "progress": true,
            "options": {"durationUs": r.times_us[samples - 1], "samples": samples, "populationTolerance": 0.01, "includeMean": false,
                "meanOnly": false, "meanHorizonUs": 1e6, "device": "gpu"},
            "model": {"dimension": STATES, "states": r.states, "hReal": vec![0.0; STATES * STATES], "hImag": vec![0.0; STATES * STATES],
                "collapse": [], "initialIndex": 0, "initialReal": r.initial_density.data.iter().map(|z| z.re).collect::<Vec<_>>(),
                "initialImag": r.initial_density.data.iter().map(|z| z.im).collect::<Vec<_>>(), "drives": [],
                "frequencyCoordinates": {"frequenciesMHz": r.frequencies_mhz, "frequencies": [], "driveIndices": [], "frameIndices": [], "frameSectors": []},
                "motion": {"axis": "3d", "trapBeamId": r.beams[r.trap_beam].id, "temperatureUK": r.temperature_uk, "massKg": r.mass_kg,
                    "fieldVectorG": [0.0, 0.0, 1.5], "hamiltonian": operator(&r.hamiltonian), "frameFrequenciesMHz": r.frame_frequencies_mhz,
                    "beams": beams, "outsideForms": forms, "emission": r.emission.iter().map(operator).collect::<Vec<_>>()}}})
    }

    /// The sample passes; each structural fault of contract §2.4 is refused with its message: a tweezer without a
    /// waist (the app's Chinese text), a coupling off excited ← ground, a beam frequency without its outside form, a
    /// non-diagonal H0 or initial density, an initial state with an excited state's population (the app's Chinese text;
    /// a round-off there is none), uneven or misplaced times, a bad label, a ground state out of the basis's first eight.
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
        refused(&|r| r.initial_density[(4, 5)] = C64::real(0.1), "trace/Hermitian");
        refused(
            &|r| {
                r.initial_density[(4, 5)] = C64::real(1e-30);
                r.initial_density[(5, 4)] = C64::real(1e-30);
            },
            "diagonal in the model basis",
        );
        refused(
            &|r| {
                r.initial_density = Mat::zeros(STATES, STATES);
                r.initial_density[(20, 20)] = C64::ONE;
            },
            ON_GROUND,
        );
        refused(
            &|r| {
                let start = r.states.iter().position(|s| s == "ground-2-0").unwrap();
                r.initial_density[(start, start)] = C64::real(0.75);
                r.initial_density[(20, 20)] = C64::real(0.25);
            },
            ON_GROUND,
        );
        let mut rounded = sample();
        rounded.initial_density[(20, 20)] = C64::real(crate::units::UNIT_ROUNDOFF);
        assert_eq!(rounded.validate(), Ok(()));
        refused(
            &|r| {
                r.initial_density[(0, 0)] = C64::real(-0.25);
                r.initial_density[(1, 1)] = C64::real(0.25);
            },
            "not positive semidefinite",
        );
        refused(&|r| r.initial_density[(0, 0)] = C64::real(0.5), "trace/Hermitian");
        refused(&|r| r.times_us[50] += 1e-6, "evenly spaced");
        refused(&|r| r.times_us[0] = 1e-3, "the first at 0");
        refused(&|r| r.states[0] = "ground-x-0".into(), "not manifold-F-m");
        refused(&|r| r.states.swap(7, 8), "ground states first and no other: D1-1--1 is out of place");
        refused(&|r| r.states[3] = "D2-0-0".into(), "ground states first and no other: D2-0-0 is out of place");
        refused(&|r| r.states.pop().map(|_| ()).unwrap(), FULL_MODEL);
        refused(&|r| r.initial_density[(3, 3)] = C64::new(f64::NAN, 0.0), "The initial density has a nonfinite entry");
        refused(&|r| r.initial_density = Mat::zeros(STATES, STATES - 1), "The initial density must be a 32 x 32 row-major matrix");
        refused(&|r| r.beams[1].field[2] = C64::new(f64::NAN, 0.0), "electric field");
        refused(&|r| r.beams[1].field = [C64::ZERO; 3], "electric field is zero");
        refused(&|r| r.emission[1][(20, 0)] = C64::real(1.0), "ground ← excited");
        refused(&|r| r.mass_kg = 0.0, ATOM);
        refused(&|r| r.beams[1].wave_number = -1.0, "Beam probe needs a finite direction");
    }

    /// The native line maps onto the request it was written from, every field (the times linspace(0, durationUs,
    /// samples) to the last bit, the population core's grid), whatever the fields hold that the engine does not read; and
    /// each check of the JSON layer refuses its fault with its message: the mean, the axis (the app's Chinese text), an
    /// unknown field at each level of the motion block, a missing field -- read or not --, the trap beam's id, the
    /// emission's count, the 32-state model, the duration and the samples (the texts the app translates), the device,
    /// the forms' shape, the initial density's presence and form.
    #[test]
    fn a_native_line_maps_onto_its_request_and_its_faults_are_refused() {
        let r = sample();
        assert_eq!(Request::from_native(&native(&r)), Ok(r.clone()));
        let mut unread = native(&r);
        unread["model"]["initialIndex"] = json!("none");
        unread["model"]["motion"]["fieldVectorG"] = json!("in H0");
        unread["model"]["motion"]["beams"][1]["intensityMWcm2"] = json!(-1);
        unread["model"]["motion"]["beams"][1]["outsideShift"] = json!({"real": [], "imag": []});
        assert_eq!(Request::from_native(&unread), Ok(r.clone()));
        let refused = |change: &dyn Fn(&mut Value), says: &str| {
            let mut line = native(&sample());
            change(&mut line);
            let e = Request::from_native(&line).expect_err(says);
            assert!(e.contains(says), "{says}: {e}");
        };
        refused(&|l| l["options"]["includeMean"] = json!(true), TRAJECTORY_ALONE);
        refused(&|l| drop(l["options"].as_object_mut().unwrap().remove("includeMean")), TRAJECTORY_ALONE);
        refused(&|l| l["options"]["meanOnly"] = json!(true), TRAJECTORY_ALONE);
        refused(&|l| l["model"]["motion"]["axis"] = json!("1d"), ONLY_3D);
        refused(&|l| l["model"]["motion"]["spin"] = json!(1), "unknown field `spin` in the motion block");
        refused(&|l| l["model"]["motion"]["beams"][0]["polarization"] = json!(1), "unknown field `polarization` in a beam");
        refused(&|l| l["model"]["motion"]["beams"][1]["electricField"][0]["phase"] = json!(0), "unknown field `phase`");
        refused(&|l| l["model"]["motion"]["outsideForms"][0]["scale"] = json!(1), "unknown field `scale` in an outside form");
        refused(&|l| drop(l["model"]["motion"]["beams"][1].as_object_mut().unwrap().remove("electricField")), "missing field `electricField` in beam probe");
        refused(&|l| drop(l["model"]["motion"].as_object_mut().unwrap().remove("outsideForms")), "missing field `outsideForms`");
        refused(&|l| drop(l["model"]["motion"].as_object_mut().unwrap().remove("fieldVectorG")), "missing field `fieldVectorG` in the motion block");
        refused(&|l| drop(l["model"]["motion"]["beams"][1].as_object_mut().unwrap().remove("intensityMWcm2")), "missing field `intensityMWcm2` in beam probe");
        refused(&|l| drop(l["model"]["motion"]["beams"][0].as_object_mut().unwrap().remove("outsideShift")), "missing field `outsideShift` in beam tweezer");
        refused(&|l| l["model"]["motion"]["trapBeamId"] = json!("lattice"), "trapBeamId lattice names none of its beams");
        refused(&|l| drop(l["model"]["motion"]["emission"].as_array_mut().unwrap().pop()), "three emission operators");
        refused(&|l| l["model"]["dimension"] = json!(31), FULL_MODEL);
        refused(&|l| drop(l["model"]["states"].as_array_mut().unwrap().pop()), FULL_MODEL);
        refused(&|l| l["options"]["durationUs"] = json!(0), "The trajectory duration must be a positive finite time");
        refused(&|l| l["options"]["durationUs"] = json!(1.5e7), "The trajectory duration is limited to 10 s (requested 15 s)");
        refused(&|l| l["options"]["samples"] = json!(1), "A trajectory has 2 to 2000 samples (requested 1)");
        refused(&|l| l["options"]["samples"] = json!(2001), "A trajectory has 2 to 2000 samples (requested 2001)");
        refused(&|l| l["options"]["device"] = json!("tpu"), "options.device");
        refused(&|l| l["model"]["motion"]["outsideForms"][0]["frequencyIndex"] = json!([-4, 0]), "An outside form needs an index on the 1 frequency axes");
        refused(&|l| drop(l["model"]["motion"]["outsideForms"][0]["matrices"].as_array_mut().unwrap().pop()), "An outside form has nine matrices");
        refused(&|l| drop(l["model"]["motion"]["hamiltonian"]["real"].as_array_mut().unwrap().pop()), "The motion block's H0 must be a 32 x 32 row-major matrix");
        refused(&|l| drop(l["model"].as_object_mut().unwrap().remove("initialImag")), "missing field `initialImag` in the model");
        refused(&|l| drop(l["model"].as_object_mut().unwrap().remove("initialReal")), "missing field `initialReal` in the model");
        refused(&|l| drop(l["model"]["initialReal"].as_array_mut().unwrap().pop()), "The initial density must be a 32 x 32 row-major matrix");
        refused(&|l| l["model"]["initialReal"][0] = json!(0.5), "trace/Hermitian");
        refused(&|l| l["model"]["motion"]["temperatureUK"] = json!(-30), ATOM);
    }
}
