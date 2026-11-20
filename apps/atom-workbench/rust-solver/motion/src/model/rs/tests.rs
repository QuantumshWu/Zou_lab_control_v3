//! The fold against td_s.hilbert_rs (scratchpad v3c\rsx\export.py: the prototype's components at the six scenes' two
//! points, its RS model on the selected P, and the components at r ± δ along the axis), read from the directory named by
//! the environment variable `V3C_RSX` (the exports are data of the check, not of the crate).

use super::*;
use crate::scene::Component;

fn matrix(v: &serde_json::Value) -> Mat {
    let shape = v["shape"].as_array().expect("shape");
    let (rows, cols) = (shape[0].as_u64().unwrap() as usize, shape[1].as_u64().unwrap() as usize);
    let mut m = Mat::zeros(rows, cols);
    for e in v["e"].as_array().expect("entries") {
        let e = e.as_array().unwrap();
        m[(e[0].as_u64().unwrap() as usize, e[1].as_u64().unwrap() as usize)] = C64::new(e[2].as_f64().unwrap(), e[3].as_f64().unwrap());
    }
    m
}

fn key(v: &serde_json::Value) -> Key {
    v.as_array().expect("key").iter().map(|k| k.as_i64().unwrap() as i32).collect()
}

/// The prototype's components at a point as the scene's: the static part and each component, gradients along z.
fn components(point: &serde_json::Value, length: usize) -> Components {
    let along = |g: Mat| [Mat::zeros(g.rows, g.cols), Mat::zeros(g.rows, g.cols), g];
    let mut parts = vec![Component { key: vec![0; length], frequency: 0.0, value: matrix(&point["static"]), gradient: along(matrix(&point["static_grad"])) }];
    for c in point["comps"].as_array().unwrap() {
        parts.push(Component { key: key(&c["key"]), frequency: c["f"].as_f64().unwrap(), value: matrix(&c["value"]), gradient: along(matrix(&c["grad"])) });
    }
    Components { parts }
}

/// The largest entry difference of two matrices and the larger of their largest entries.
fn apart(a: &Mat, b: &Mat) -> (f64, f64) {
    let d = a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max);
    (d, a.max_abs().max(b.max_abs()))
}

/// The exports, each scene and point: (name, document).
fn exports() -> Vec<(String, serde_json::Value)> {
    let dir = std::env::var("V3C_RSX").expect("V3C_RSX: the directory of the prototype's exports (scratchpad v3c\\rsx)");
    let mut found: Vec<_> = std::fs::read_dir(&dir).expect("the exports").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    found.sort();
    found.into_iter().map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap())).collect()
}

/// The model of a document's point `which` (0: r, 1: r + δu, 2: r − δu, 3: r + 2δu, 4: r − 2δu): its source folded on P with no fast element
/// (the prototype's hilbert_rs folds Q alone).
fn model(doc: &serde_json::Value, which: usize) -> ModelAt {
    let labels: Vec<String> = doc["labels"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    let keep: Vec<usize> = doc["keep"].as_array().unwrap().iter().map(|k| k.as_u64().unwrap() as usize).collect();
    let length = doc["out"]["H"][0]["key"].as_array().unwrap().len();
    let emission: Vec<Mat> = doc["emission"].as_array().unwrap().iter().map(matrix).collect();
    let jumps = Jumps { emission: [emission[0].clone(), emission[1].clone(), emission[2].clone()], wave_number: matrix(&doc["wave"]) };
    let source = ModelAt::source(&components(&doc["points"][which], length), &jumps, &labels);
    let sectors = Sectors { states: vec![[0; 3]; STATES], generators: vec![[0; 3]; length] };
    fold(&source, &keep, &sectors, MODEL)
}

/// The fold is hilbert_rs to rounding (RUSTPLAN C1): on the six scenes' selected P at both points (centre: the tweezer
/// full; edge: none), every component of H_eff and its gradient along the axis, of the jumps on the channels e₁, e₂, u
/// and their gradients, of the photons' forms, and the lift M_c, each against the prototype's of the same key, relative
/// to the quantity's largest entry over every key. The decomposition's route against the prototype's column-by-column
/// inverse differs by κ(V) u (V the folded block's eigenvectors): measured up to 3.8·10⁻¹¹ (GMD1C, its 24 folded excited
/// states); the bound read is 10⁻¹⁰.
#[test]
fn the_fold_is_hilbert_rs_to_rounding() {
    let mut worst = 0.0_f64;
    for (name, doc) in exports() {
        let at = model(&doc, 0);
        let hilbert = at.hilbert([0.0, 0.0, 1.0]);
        let out = &doc["out"];
        // Per quantity its largest entry difference and its largest entry over every key (a key whose gradient is
        // nothing but rounding is read against the quantity's scale).
        let (mut differences, mut sizes) = ([0.0_f64; 5], [0.0_f64; 5]);
        let mut check = |slot: usize, rust: Option<&Mat>, python: &Mat| {
            let zero = Mat::zeros(python.rows, python.cols);
            let (d, size) = apart(rust.unwrap_or(&zero), python);
            (differences[slot], sizes[slot]) = (differences[slot].max(d), sizes[slot].max(size));
        };
        let keys_h: Vec<Key> = out["H"].as_array().unwrap().iter().map(|h| key(&h["key"])).collect();
        for h in out["H"].as_array().unwrap() {
            let found = hilbert.hamiltonian.iter().find(|p| p.key == key(&h["key"]));
            check(0, found.map(|p| &p.value), &matrix(&h["value"]));
            check(1, found.map(|p| &p.gradient), &matrix(&h["grad"]));
        }
        for p in &hilbert.hamiltonian {
            assert!(keys_h.contains(&p.key) || p.value.max_abs() == 0.0, "{name}: a key the prototype has not: {:?}", p.key);
        }
        for (channel, python) in hilbert.jumps.iter().zip(out["C"].as_array().unwrap()) {
            for c in python.as_array().unwrap() {
                let found = channel.iter().find(|p| p.key == key(&c["key"]));
                check(2, found.map(|p| &p.value), &matrix(&c["value"]));
                check(2, found.map(|p| &p.gradient), &matrix(&c["grad"]));
            }
        }
        for (channel, python) in hilbert.kicks.iter().zip(out["C"].as_array().unwrap()) {
            for c in python.as_array().unwrap() {
                check(3, channel.iter().find(|p| p.key == key(&c["key"])).map(|p| &p.value), &matrix(&c["kick"]));
            }
        }
        for l in out["lift"].as_array().unwrap() {
            check(4, at.lift.iter().find(|p| p.key == key(&l["key"])).map(|p| &p.value), &matrix(&l["value"]));
        }
        let errors: Vec<f64> = differences.iter().zip(&sizes).map(|(d, s)| d / s.max(f64::MIN_POSITIVE)).collect();
        eprintln!("{name}: P {} H {:.1e} ∂H {:.1e} L {:.1e} K {:.1e} M {:.1e}", at.states.len(), errors[0], errors[1], errors[2], errors[3], errors[4]);
        worst = errors.iter().fold(worst, |w, e| w.max(*e));
    }
    assert!(worst <= 1e-10, "{worst:e}");
}

/// The adjoint force against the finite difference (RUSTPLAN C1): on a random density matrix of P, −Re tr(∂_z H_eff ρ) +
/// Σ_ch Im tr(ρ L_ch†∂_z L_ch) of the fold's gradients against the same with the derivatives taken by Richardson's
/// central differences of the folds at z ± δ and z ± 2δ, (4D(δ) − D(2δ))/3, D(h) = (X(z + h) − X(z − h))/2h (δ = 2·10⁻³
/// µm: the remainder (2kδ)⁴/30 ~ 2·10⁻⁹ of a beam's phase; the rounding u|H|/δ, |H| ~ 4·10⁴ rad/µs of the bare ground
/// splitting, ~ 5·10⁻⁹ rad/µs/µm -- at δ = 10⁻⁵ µm it was 4·10⁻⁷, above the smallest forces' 10⁻⁷), within 10⁻⁷ of the
/// force and the differences' rounding floor (measured: 0.9–2.4·10⁻⁹ relative; GMD1C's edge, its force 0.043, 2.3·10⁻⁷ at
/// 1.0·10⁻⁸ rad/µs/µm, within its floor).
#[test]
fn the_adjoint_force_is_the_finite_difference() {
    let mut stream = crate::random::Stream::new(41, crate::random::Purpose::Initial, 0, 0);
    let mut worst = 0.0_f64;
    for (name, doc) in exports() {
        let delta = doc["delta"].as_f64().unwrap();
        let [at, up, down, up2, down2] = [0, 1, 2, 3, 4].map(|w| model(&doc, w));
        let n = at.states.len();
        // Richardson's derivative of a matrix read off each fold.
        let derivative = |read: &dyn Fn(&ModelAt) -> Mat| {
            let (mut d, mut d2) = (read(&up), read(&up2));
            d.add_scaled(C64::real(-1.0), &read(&down));
            d2.add_scaled(C64::real(-1.0), &read(&down2));
            let mut r = d.scaled(C64::real(4.0 / (3.0 * 2.0 * delta)));
            r.add_scaled(C64::real(-1.0 / (3.0 * 4.0 * delta)), &d2);
            r
        };
        let a = Mat::from_fn(n, n, |_, _| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5));
        let mut rho = a.product(&a.adjoint());
        let t = rho.trace();
        rho = rho.scaled(C64::real(1.0) / t);
        let trace = |x: &Mat| -> C64 { (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| x[(i, j)] * rho[(j, i)]).sum() };
        let adjoint = at.force(&rho.data)[2];
        let mut difference = -trace(&derivative(&|m: &ModelAt| summed(&m.effective, None, n))).re;
        for channel in &at.jumps {
            let l = summed(&channel.parts, None, n);
            let pick = |m: &ModelAt| m.jumps.iter().find(|c| c.emission == channel.emission && c.fast == channel.fast).map_or(Mat::zeros(n, n), |c| summed(&c.parts, None, n));
            difference += trace(&l.adjoint_product(&derivative(&pick))).im;
        }
        // The differences' rounding: each entry of a fold rounds by u|H|, a trace with ρ sums n² of them within
        // n u max|H| (Σ|ρ_ab| ≤ n tr ρ), and Richardson's four differences weigh them by (4/3)/δ + (1/6)/δ.
        let largest = at.effective.iter().map(|p| p.value.max_abs()).fold(0.0, f64::max);
        let floor = 1.5 * n as f64 * crate::units::UNIT_ROUNDOFF * largest / delta;
        let error = (adjoint - difference).abs() / (1e-7 * adjoint.abs() + floor);
        eprintln!("{name}: force {adjoint:.10e}, finite difference {difference:.10e}, relative {:.1e} (rounding floor {floor:.1e})", (adjoint - difference).abs() / adjoint.abs());
        worst = worst.max(error);
    }
    assert!(worst <= 1.0, "{worst:e} of 1e-7 of the force and the rounding floor");
}
