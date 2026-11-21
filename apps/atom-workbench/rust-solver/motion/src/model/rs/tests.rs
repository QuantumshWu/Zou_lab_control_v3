//! The fold against td_s.hilbert_rs (scratchpad v3c\rsx\export.py: the prototype's components at the six scenes' two
//! points, its RS model on the selected P, and the components at r ± δ along the axis), read from the directory named by
//! the environment variable `V3C_RSX`; and `Rs::at` on the six scenes' own requests (scratchpad reqprec\req, the
//! directory named by `V3C_REQ`) against the fold with every block decomposed at the point. The data are the check's,
//! not the crate's.

use super::*;
use crate::floquet::Frame;
use crate::request::Request;
use crate::linalg;


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

/// The prototype's model at a point as a fold's source: each component (the static part, then each key) its value and
/// its gradient along z as two fixed matrices (factors: one, and zero with the gradient (0, 0, 1)), each nonzero key both
/// ways; the Cartesian emission's channels with their photons' forms (the export's wave numbers); the generator the
/// static part − (i/2) Σ C†C. Returns the library, the source and its values.
fn source(doc: &serde_json::Value, point: &serde_json::Value, length: usize) -> (Vec<Fixed>, Source, Values) {
    let mut library: Vec<Fixed> = Vec::new();
    let mut put = |m: &Mat| {
        library.push(fixed(m));
        library.len() - 1
    };
    let mut hamiltonian = Vec::new();
    let mut frequencies = Vec::new();
    let mut piece = |key: Key, frequency: f64, value: &serde_json::Value, grad: &serde_json::Value| {
        let terms = vec![Term::one(0, put(&matrix(value))), Term::one(1, put(&matrix(grad)))];
        let nonzero = key.iter().any(|k| *k != 0);
        let p = Piece { key, terms };
        let adjoint = nonzero.then(|| Piece { key: p.key.iter().map(|k| -k).collect(), terms: p.terms.iter().map(|t| Term { adjoint: true, ..t.clone() }).collect() });
        hamiltonian.push(p);
        frequencies.push(frequency);
        if let Some(a) = adjoint {
            hamiltonian.push(a);
            frequencies.push(-frequency);
        }
    };
    piece(vec![0; length], 0.0, &point["static"], &point["static_grad"]);
    for c in point["comps"].as_array().unwrap() {
        piece(key(&c["key"]), c["f"].as_f64().unwrap(), &c["value"], &c["grad"]);
    }
    let emission_matrices: Vec<Mat> = doc["emission"].as_array().unwrap().iter().map(matrix).collect();
    let jumps = Jumps { emission: [emission_matrices[0].clone(), emission_matrices[1].clone(), emission_matrices[2].clone()], wave_number: matrix(&doc["wave"]) };
    let (c, k, gamma) = emission(&jumps);
    let mut put = |m: &Mat| {
        library.push(fixed(m));
        library.len() - 1
    };
    let (parts, kicks) = (c.each_ref().map(&mut put), k.each_ref().map(&mut put));
    let decay = put(&gamma);
    let mut generator = hamiltonian[0].terms.clone();
    generator.push(Term { factor: 0, adjoint: false, scale: C64::new(0.0, -0.5), matrices: vec![(decay, 0)] });
    let jumps = emission_channels(length, 0, parts, kicks);
    frequencies.extend(std::iter::repeat_n(0.0, 6));
    let source = Source { hamiltonian, jumps, generator, varying: vec![false, true], coefficients: 1 };
    let values = Values { factors: vec![(C64::ONE, [C64::ZERO; 3]), (C64::ZERO, [C64::ZERO, C64::ZERO, C64::ONE])], coefficients: vec![1.0], frequencies };
    (library, source, values)
}

/// The largest entry difference of two matrices and the larger of their largest entries.
fn apart(a: &Mat, b: &Mat) -> (f64, f64) {
    let d = a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).abs()).fold(0.0, crate::linalg::larger);
    (d, crate::linalg::larger(a.max_abs(), b.max_abs()))
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
fn model(doc: &serde_json::Value, which: usize) -> (ModelAt, Now) {
    let labels: Vec<String> = doc["labels"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    let mut keep: Vec<usize> = doc["keep"].as_array().unwrap().iter().map(|k| k.as_u64().unwrap() as usize).collect();
    keep.sort_unstable();
    let length = doc["out"]["H"][0]["key"].as_array().unwrap().len();
    let (library, source, values) = source(doc, &doc["points"][which], length);
    let sectors = Sectors { states: vec![[0; 2]; STATES], generators: vec![[0; 2]; length] };
    let all: Vec<usize> = (0..STATES).collect();
    let folding = Folding::new(&library, &source, &keep, &sectors, MODEL, &levels(&labels, &all));
    let local = folding.at(&values);
    (local.components(), local.now(0.0))
}

/// The sum of a model operator's components at the frame's time zero, or of their gradients along r_i.
fn summed(parts: &[ModelPart], gradient: Option<usize>, size: usize) -> Mat {
    let mut sum = Mat::zeros(size, size);
    for p in parts {
        sum.add_scaled(C64::ONE, gradient.map_or(&p.value, |i| &p.gradient[i]));
    }
    sum
}

/// A dense model's generator: H's static part − (i/2) Σ_ch Σ_c L†L, with its gradient.
fn generator_of(model: &ModelAt) -> ModelPart {
    let mut g = model.still().clone();
    let (gamma, changes) = decay(&model.jumps, model.states.len());
    g.value.add_scaled(C64::new(0.0, -0.5), &gamma);
    for (d, c) in g.gradient.iter_mut().zip(&changes) {
        d.add_scaled(C64::new(0.0, -0.5), c);
    }
    g
}

/// The fold is hilbert_rs to rounding (RUSTPLAN C1): on the six scenes' selected P at both points (centre: the tweezer
/// full; edge: none), every component of H_eff and its gradient along the axis, of the jumps on the channels e₁ = x,
/// e₂ = y, u = z (the analysis axis z) and their gradients, of the photons' forms (the prototype's weighted by √w, w =
/// 2/5, 2/5, 1/5), and the lift M_c, each against the prototype's of the same key, relative to the quantity's largest
/// entry over every key. The decomposition's route against the prototype's column-by-column inverse differs by κ(V) u
/// (V the folded block's eigenvectors): measured up to 3.8·10⁻¹¹ (GMD1C, its 24 folded excited states); the bound read
/// is 10⁻¹⁰.
#[test]
fn the_fold_is_hilbert_rs_to_rounding() {
    let mut worst = 0.0_f64;
    for (name, doc) in exports() {
        let at = model(&doc, 0).0;
        let out = &doc["out"];
        // Per quantity its largest entry difference and its largest entry over every key (a key whose gradient is
        // nothing but rounding is read against the quantity's scale).
        let (mut differences, mut sizes) = ([0.0_f64; 5], [0.0_f64; 5]);
        let mut check = |slot: usize, rust: Option<&Mat>, python: &Mat| {
            let zero = Mat::zeros(python.rows, python.cols);
            let (d, size) = apart(rust.unwrap_or(&zero), python);
            (differences[slot], sizes[slot]) = (crate::linalg::larger(differences[slot], d), crate::linalg::larger(sizes[slot], size));
        };
        let keys_h: Vec<Key> = out["H"].as_array().unwrap().iter().map(|h| key(&h["key"])).collect();
        for h in out["H"].as_array().unwrap() {
            let found = at.effective.iter().find(|p| p.key == key(&h["key"]));
            check(0, found.map(|p| &p.value), &matrix(&h["value"]));
            check(1, found.map(|p| &p.gradient[2]), &matrix(&h["grad"]));
        }
        for p in &at.effective {
            assert!(keys_h.contains(&p.key) || p.value.max_abs() == 0.0, "{name}: a key the prototype has not: {:?}", p.key);
        }
        for (q, (python, weight)) in out["C"].as_array().unwrap().iter().zip([0.4_f64, 0.4, 0.2]).enumerate() {
            let channel = at.jumps.iter().find(|c| c.emission == q).expect("the emission's channel");
            for c in python.as_array().unwrap() {
                let found = channel.parts.iter().find(|p| p.key == key(&c["key"]));
                check(2, found.map(|p| &p.value), &matrix(&c["value"]));
                check(2, found.map(|p| &p.gradient[2]), &matrix(&c["grad"]));
                let kick = channel.kicks.iter().find(|p| p.key == key(&c["key"])).map(|p| p.value.scaled(C64::real(weight.sqrt())));
                check(3, kick.as_ref(), &matrix(&c["kick"]));
            }
        }
        for l in out["lift"].as_array().unwrap() {
            check(4, at.lift.iter().find(|p| p.key == key(&l["key"])).map(|p| &p.value), &matrix(&l["value"]));
        }
        let errors: Vec<f64> = differences.iter().zip(&sizes).map(|(d, s)| d / crate::linalg::larger(*s, f64::MIN_POSITIVE)).collect();
        eprintln!("{name}: P {} H {:.1e} ∂H {:.1e} L {:.1e} K {:.1e} M {:.1e}", at.states.len(), errors[0], errors[1], errors[2], errors[3], errors[4]);
        worst = errors.iter().fold(worst, |w, e| crate::linalg::larger(w, *e));
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
        let (at, now) = model(&doc, 0);
        let [up, down, up2, down2] = [1, 2, 3, 4].map(|w| model(&doc, w).0);
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
        let adjoint = now.force(&rho.data)[2];
        let mut difference = -trace(&derivative(&|m: &ModelAt| summed(&m.effective, None, n))).re;
        for channel in &at.jumps {
            let l = summed(&channel.parts, None, n);
            let pick = |m: &ModelAt| m.jumps.iter().find(|c| c.emission == channel.emission && c.fast == channel.fast).map_or(Mat::zeros(n, n), |c| summed(&c.parts, None, n));
            difference += trace(&l.adjoint_product(&derivative(&pick))).im;
        }
        // The differences' rounding: each entry of a fold rounds by u|H|, a trace with ρ sums n² of them within
        // n u max|H| (Σ|ρ_ab| ≤ n tr ρ), and Richardson's four differences weigh them by (4/3)/δ + (1/6)/δ.
        let largest = at.effective.iter().map(|p| p.value.max_abs()).fold(0.0, crate::linalg::larger);
        let floor = 1.5 * n as f64 * crate::units::UNIT_ROUNDOFF * largest / delta;
        let error = (adjoint - difference).abs() / (1e-7 * adjoint.abs() + floor);
        eprintln!("{name}: force {adjoint:.10e}, finite difference {difference:.10e}, relative {:.1e} (rounding floor {floor:.1e})", (adjoint - difference).abs() / adjoint.abs());
        worst = crate::linalg::larger(worst, error);
    }
    assert!(worst <= 1.0, "{worst:e} of 1e-7 of the force and the rounding floor");
}

// ---------------------------------------------------------------------------------------------------------------------
// Rs::at on the six scenes' requests.

/// The six scenes' own requests (V3C_REQ: scratchpad reqprec\req, PYEXP2's) with their selected P (pyexp2.status.md A,
/// the same as PYSEL's: the levels added to the ground states).
fn scenes() -> Vec<(&'static str, Scene, Vec<usize>)> {
    let dir = std::path::PathBuf::from(std::env::var("V3C_REQ").expect("V3C_REQ: the directory of the six scenes' requests (scratchpad reqprec\\req)"));
    let chosen: [(&str, &str, &[&str]); 6] = [
        ("tilt0", "tilt0.json", &["D2-3", "D2-2"]),
        ("tilt2", "tilt2.json", &["D2-3", "D2-2"]),
        ("R4", "R4_LIN_4.json", &["D2-2", "D2-3", "D2-1"]),
        ("GMD1C", "GMD1C_1ms.json", &[]),
        ("PGC3D", "PGC3D_1ms.json", &["D2-2", "D2-3"]),
        ("TET", "TET_1ms.json", &["D2-2", "D2-3"]),
    ];
    chosen
        .into_iter()
        .map(|(name, file, levels)| {
            let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap();
            line["options"]["device"] = serde_json::json!("cpu");
            // These requests (the ones PYSEL's exports were made of) predate the outside forms' transitions, which the
            // scene does not read.
            for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
                form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
            }
            let request = Request::from_native(&line).unwrap();
            let scene = Scene::new(&request).unwrap();
            let keep: Vec<usize> = (0..STATES)
                .filter(|&s| {
                    let label = &scene.request.states[s];
                    label.starts_with("ground") || levels.iter().any(|l| label.starts_with(&format!("{l}-")))
                })
                .collect();
            (name, scene, keep)
        })
        .collect()
}

/// The frame's sectors at the checkpoint's centre (r = 0, the tweezer full, no Doppler shift).
fn sectors(scene: &Scene) -> Sectors {
    let zero = vec![0.0; scene.request.beams.len()];
    let centre = scene.components([0.0; 3], &zero, 1.0);
    Frame::new(scene, &crate::floquet::Hilbert::full(scene, &centre), &zero).sectors()
}

/// Model parts by key, accumulated element by element (the exact reference's sums).
#[derive(Default)]
struct Exact(BTreeMap<Key, ModelPart>);

impl Exact {
    /// Adds scale·x into the part of `key` (size × size), x's entry (i, j) -- or, `adjoint`, x's (j, i) conjugated -- at
    /// (rows[i], cols[j]), where `keep(rows[i], cols[j])` holds.
    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, key: &Key, frequency: f64, size: usize, x: (&Mat, [&Mat; 3]), adjoint: bool, rows: &[usize], cols: &[usize], scale: f64, keep: impl Fn(usize, usize) -> bool) {
        let part = self.0.entry(key.clone()).or_insert_with(|| ModelPart { key: key.clone(), frequency, value: Mat::zeros(size, size), gradient: zeros(size, size) });
        let read = |m: &Mat, i: usize, j: usize| if adjoint { m[(j, i)].conj() } else { m[(i, j)] };
        for (i, &b) in rows.iter().enumerate() {
            for (j, &a) in cols.iter().enumerate() {
                if keep(b, a) {
                    part.value[(b, a)] += read(x.0, i, j) * scale;
                    for (g, d) in part.gradient.iter_mut().zip(x.1) {
                        g[(b, a)] += read(d, i, j) * scale;
                    }
                }
            }
        }
    }

    fn parts(self) -> Vec<ModelPart> {
        self.0.into_values().filter(|p| nonzero(&p.value) || p.gradient.iter().any(nonzero)).collect()
    }
}

/// The fold with every block decomposed at the point and every column's resolvent its spectrum (64c973e8's fold, the
/// formula of the module documentation without the series): the check's exact reference.
fn exact(source: &ModelAt, keep: &[usize], sectors: &Sectors, axes: [bool; 2]) -> ModelAt {
    let place = |state: usize| source.states.iter().position(|&s| s == state).unwrap();
    let mut kept: Vec<usize> = keep.iter().map(|&s| place(s)).collect();
    kept.sort_unstable();
    let size = source.states.len();
    let others: Vec<usize> = (0..size).filter(|p| kept.binary_search(p).is_err()).collect();
    let n = kept.len();
    let fast = |key: &Key, to: usize, from: usize| -> [i64; 2] { Sectors::element(sectors.key(key, axes), sectors.state(source.states[to], axes), sectors.state(source.states[from], axes)) };
    let still = source.still();
    let mut g0 = still.value.clone();
    let mut dg0 = still.gradient.clone();
    for channel in &source.jumps {
        for p in &channel.parts {
            g0.add_scaled(C64::new(0.0, -0.5), &p.value.adjoint_product(&p.value));
            for (d, dl) in dg0.iter_mut().zip(&p.gradient) {
                d.add_scaled(C64::new(0.0, -0.5), &dl.adjoint_product(&p.value));
                d.add_scaled(C64::new(0.0, -0.5), &p.value.adjoint_product(dl));
            }
        }
    }
    let mut groups: BTreeMap<[i64; 2], Vec<usize>> = BTreeMap::new();
    for &p in &kept {
        groups.entry(sectors.state(source.states[p], axes)).or_default().push(p);
    }
    let blocks: Vec<(Vec<usize>, linalg::Eigen, [Mat; 3])> = std::iter::once(others.clone())
        .filter(|rows| !rows.is_empty())
        .chain(groups.into_values())
        .map(|rows| {
            let mut eigen = linalg::Eigen::default();
            assert!(eigen.decompose(&g0.block(&rows, &rows)));
            let changes = dg0.each_ref().map(|d| d.block(&rows, &rows));
            (rows, eigen, changes)
        })
        .collect();
    let resolve = |eigen: &linalg::Eigen, z: &[f64], y: &Mat, columns: &[bool]| {
        let mut w = eigen.inverse.product(y);
        for k in 0..w.rows {
            for a in 0..w.cols {
                w[(k, a)] = if columns[a] { w[(k, a)] / (C64::real(z[a]) - eigen.values[k]) } else { C64::ZERO };
            }
        }
        eigen.vectors.product(&w)
    };
    let energies: Vec<f64> = kept.iter().map(|&a| still.value[(a, a)].re).collect();
    let slopes: [Vec<f64>; 3] = std::array::from_fn(|i| kept.iter().map(|&a| still.gradient[i][(a, a)].re).collect());
    let mut amplitudes: Vec<(usize, Mat, [Mat; 3])> = Vec::new();
    for (index, c) in source.effective.iter().enumerate() {
        let mut m = Mat::zeros(size, n);
        let mut dm = zeros(size, n);
        for (rows, eigen, changes) in &blocks {
            let columns: Vec<bool> = kept.iter().map(|&a| others.binary_search(&rows[0]).is_ok() || fast(&c.key, rows[0], a) != [0; 2]).collect();
            let y = c.value.block(rows, &kept);
            let moved = c.gradient.each_ref().map(|d| d.block(rows, &kept));
            let z: Vec<f64> = energies.iter().map(|e| e + c.frequency).collect();
            let amplitude = resolve(eigen, &z, &y, &columns);
            for (i, d) in dm.iter_mut().enumerate() {
                let mut x = moved[i].clone();
                x.add_scaled(C64::ONE, &changes[i].product(&amplitude));
                for k in 0..x.rows {
                    for a in 0..n {
                        x[(k, a)] -= amplitude[(k, a)] * slopes[i][a];
                    }
                }
                let change = resolve(eigen, &z, &x, &columns);
                for (k, &row) in rows.iter().enumerate() {
                    for a in 0..n {
                        d[(row, a)] = change[(k, a)];
                    }
                }
            }
            for (k, &row) in rows.iter().enumerate() {
                for a in 0..n {
                    m[(row, a)] = amplitude[(k, a)];
                }
            }
        }
        amplitudes.push((index, m, dm));
    }
    let all_kept: Vec<usize> = (0..n).collect();
    let all: Vec<usize> = (0..size).collect();
    let mut effective = Exact::default();
    for c in &source.effective {
        let (block, moved) = (c.value.block(&kept, &kept), c.gradient.each_ref().map(|d| d.block(&kept, &kept)));
        effective.add(&c.key, c.frequency, n, (&block, moved.each_ref()), false, &all_kept, &all_kept, 1.0, |b, a| fast(&c.key, kept[b], kept[a]) == [0; 2]);
    }
    for c2 in &source.effective {
        let (out, moved) = (c2.value.block(&kept, &all), c2.gradient.each_ref().map(|d| d.block(&kept, &all)));
        for (index, m, dm) in &amplitudes {
            let c = &source.effective[*index];
            let (key, frequency) = (sum_keys(&c.key, &c2.key), c.frequency + c2.frequency);
            let a = out.product(m);
            let da: [Mat; 3] = std::array::from_fn(|i| {
                let mut d = moved[i].product(m);
                d.add_scaled(C64::ONE, &out.product(&dm[i]));
                d
            });
            let slow = |b: usize, x: usize| fast(&key, kept[b], kept[x]) == [0; 2];
            effective.add(&key, frequency, n, (&a, da.each_ref()), false, &all_kept, &all_kept, 0.5, slow);
            effective.add(&negated(&key), -frequency, n, (&a, da.each_ref()), true, &all_kept, &all_kept, 0.5, |b, x| slow(x, b));
        }
    }
    let mut out: BTreeMap<(usize, [i64; 2]), Exact> = BTreeMap::new();
    for channel in &source.jumps {
        for p in &channel.parts {
            let mut add = |key: &Key, frequency: f64, value: &Mat, gradient: [&Mat; 3]| {
                let mut indices: Vec<[i64; 2]> = (0..n).flat_map(|b| (0..n).map(move |a| (b, a))).filter(|&(b, a)| value[(b, a)] != C64::ZERO || gradient.iter().any(|d| d[(b, a)] != C64::ZERO)).map(|(b, a)| fast(key, kept[b], kept[a])).collect();
                indices.sort_unstable();
                indices.dedup();
                for index in indices {
                    out.entry((channel.emission, index)).or_default().add(key, frequency, n, (value, gradient), false, &all_kept, &all_kept, 1.0, |b, a| fast(key, kept[b], kept[a]) == index);
                }
            };
            let (direct, moved) = (p.value.block(&kept, &kept), p.gradient.each_ref().map(|d| d.block(&kept, &kept)));
            add(&p.key, p.frequency, &direct, moved.each_ref());
            let (from, moved) = (p.value.block(&kept, &all), p.gradient.each_ref().map(|d| d.block(&kept, &all)));
            for (index, m, dm) in &amplitudes {
                let c = &source.effective[*index];
                let gradient: [Mat; 3] = std::array::from_fn(|i| {
                    let mut d = moved[i].product(m);
                    d.add_scaled(C64::ONE, &from.product(&dm[i]));
                    d
                });
                add(&sum_keys(&p.key, &c.key), p.frequency + c.frequency, &from.product(m), gradient.each_ref());
            }
        }
    }
    let jumps = out.into_iter().map(|((emission, fast), parts)| Channel { emission, fast, parts: parts.parts(), kicks: Vec::new() }).filter(|c| !c.parts.is_empty()).collect();
    ModelAt { states: kept.iter().map(|&p| source.states[p]).collect(), folded: others.iter().map(|&p| source.states[p]).collect(), groups: source.groups.clone(), effective: effective.parts(), jumps, lift: Vec::new() }
}

/// The largest entry difference of two models' H_eff and jumps (by key and channel), each relative to its quantity's
/// largest entry: (H, ∇H, L, ∇L).
fn differ(a: &ModelAt, b: &ModelAt) -> [f64; 4] {
    let mut d = [0.0_f64; 4];
    let mut scale = [0.0_f64; 4];
    let mut take = |slot: usize, x: Option<&Mat>, y: Option<&Mat>| {
        let (x, y) = (x.cloned().unwrap_or_else(|| Mat::zeros(a.states.len(), a.states.len())), y.cloned().unwrap_or_else(|| Mat::zeros(a.states.len(), a.states.len())));
        let (e, s) = apart(&x, &y);
        d[slot] = crate::linalg::larger(d[slot], e);
        scale[slot] = crate::linalg::larger(scale[slot], s);
    };
    let keys: std::collections::BTreeSet<Key> = a.effective.iter().chain(&b.effective).map(|p| p.key.clone()).collect();
    for key in keys {
        let (x, y) = (a.effective.iter().find(|p| p.key == key), b.effective.iter().find(|p| p.key == key));
        take(0, x.map(|p| &p.value), y.map(|p| &p.value));
        for i in 0..3 {
            take(1, x.map(|p| &p.gradient[i]), y.map(|p| &p.gradient[i]));
        }
    }
    let channels: std::collections::BTreeSet<(usize, [i64; 2])> = a.jumps.iter().chain(&b.jumps).map(|c| (c.emission, c.fast)).collect();
    for (emission, fast) in channels {
        let pick = |m: &ModelAt| m.jumps.iter().find(|c| c.emission == emission && c.fast == fast).map(|c| c.parts.clone()).unwrap_or_default();
        let (x, y) = (pick(a), pick(b));
        let keys: std::collections::BTreeSet<Key> = x.iter().chain(&y).map(|p| p.key.clone()).collect();
        for key in keys {
            let (p, q) = (x.iter().find(|p| p.key == key), y.iter().find(|p| p.key == key));
            take(2, p.map(|p| &p.value), q.map(|p| &p.value));
            for i in 0..3 {
                take(3, p.map(|p| &p.gradient[i]), q.map(|p| &p.gradient[i]));
            }
        }
    }
    std::array::from_fn(|i| d[i] / crate::linalg::larger(scale[i], f64::MIN_POSITIVE))
}

/// The points of the check: the centre and the edge of the checkpoint (r = 0, the tweezer full and dark, no Doppler
/// shift) and a moving atom off the centre (r = (0.13, −0.21, 0.37) µm, κ = (50, −80, 120)/µm, the tweezer full).
fn points(scene: &Scene) -> [(&'static str, Components); 3] {
    let zero = vec![0.0; scene.request.beams.len()];
    [
        ("centre", scene.components([0.0; 3], &zero, 1.0)),
        ("edge", scene.components([0.0; 3], &zero, 0.0)),
        ("moving", scene.components_at([0.13, -0.21, 0.37], [50.0, -80.0, 120.0])),
    ]
}

/// `Rs::at` is the fold with every block decomposed at the point to rounding (RUSTPLAN correction "模型只定一次": "Rs::at
/// 对逐点完整折叠到舍入"): on the six scenes' P at the checkpoint's centre and edge and at a moving point off the centre,
/// H_eff, the jumps and their gradients, each relative to its quantity's largest entry. The sideband parts' Neumann
/// series is summed to the rounding of each column; the decompositions' rounding is κ(V)u (measured on hilbert_rs up to
/// 3.8·10⁻¹¹); the bound read is 10⁻¹⁰.
#[test]
fn rs_at_is_the_exact_fold_to_rounding() {
    let mut worst = 0.0_f64;
    for (name, scene, keep) in scenes() {
        let rs = Rs::new(&scene, &keep, sectors(&scene));
        for (point, components) in points(&scene) {
            let model = rs.at(&scene, &components);
            let (reference, generator) = rs.tweezer.reference(&scene, &components.point);
            let made = generator_of(&reference);
            let (g, scale) = apart(&generator.value, &made.value);
            assert!(g <= 64.0 * UNIT_ROUNDOFF * scale, "{name} {point}: the generator made once against the point's: {g:e} of {scale:e}");
            let truth = exact(&reference, &rs.keep, &rs.sectors, MODEL);
            let errors = differ(&model, &truth);
            eprintln!("{name} {point}: P {} H {:.1e} ∇H {:.1e} L {:.1e} ∇L {:.1e} ({} parts, {} channels)", model.states.len(), errors[0], errors[1], errors[2], errors[3], model.effective.len(), model.jumps.len());
            worst = errors.iter().fold(worst, |w, e| crate::linalg::larger(w, *e));
        }
    }
    assert!(worst <= 1e-10, "{worst:e}");
}

/// The model keeps slow elements alone, by the frame's own rule (V3E's the_selected_model_is_slow_in_its_frame): on the
/// six scenes' P at the checkpoint's centre and edge and at the moving point, every nonzero element b ← a of every H_eff
/// component at key c has the frame vector c − θ_b + θ_a (floquet/frame.rs `element`, the assembly's) with no fast part
/// (n_φ = n_tw = n_ν = 0, `Frame::split`); every element of a channel has the channel's fast index (n_φ, n_tw), and the
/// channel's elements one n_ν (they beat together). The frame is the checkpoint's centre's (no Doppler shift).
#[test]
fn the_model_is_slow_in_its_frame() {
    let mut wrong = 0;
    for (name, scene, keep) in scenes() {
        let zero = vec![0.0; scene.request.beams.len()];
        let centre = scene.components([0.0; 3], &zero, 1.0);
        let frame = Frame::new(&scene, &crate::floquet::Hilbert::full(&scene, &centre), &zero);
        let rs = Rs::new(&scene, &keep, frame.sectors());
        for (point, components) in points(&scene) {
            let model = rs.at(&scene, &components);
            let n = model.states.len();
            let vector = |key: &Key, b: usize, a: usize| -> [i64; 3] {
                let (tb, ta) = (&frame.theta[model.states[b]], &frame.theta[model.states[a]]);
                let v: crate::floquet::frame::FrameKey = (0..tb.len()).map(|i| crate::floquet::frame::element(key.get(i).map_or(0, |&k| i64::from(k)), tb[i], ta[i])).collect();
                frame.split(&v).0
            };
            let entries = |p: &ModelPart| -> Vec<(usize, usize)> { (0..n).flat_map(|b| (0..n).map(move |a| (b, a))).filter(|&(b, a)| p.value[(b, a)] != C64::ZERO || p.gradient.iter().any(|g| g[(b, a)] != C64::ZERO)).collect() };
            let (mut kept, mut fast) = (0, 0);
            for p in &model.effective {
                for (b, a) in entries(p) {
                    kept += 1;
                    fast += usize::from(vector(&p.key, b, a) != [0; 3]);
                }
            }
            let (mut elements, mut astray) = (0, 0);
            for c in &model.jumps {
                let mut carrier = None;
                for p in &c.parts {
                    for (b, a) in entries(p) {
                        let v = vector(&p.key, b, a);
                        elements += 1;
                        astray += usize::from([v[0], v[1]] != c.fast || *carrier.get_or_insert(v[2]) != v[2]);
                    }
                }
            }
            eprintln!("{name} {point}: H_eff {kept} elements kept, {fast} fast; {} channels, {elements} elements, {astray} astray", model.jumps.len());
            wrong += fast + astray;
        }
    }
    assert_eq!(wrong, 0);
}

/// The value at a time is the components at their phases (module documentation, # Two contractions): on the six scenes'
/// P at the checkpoint's centre and edge and at the moving point, at t = 0 and two times off it (0.37 and −1.9 µs: every
/// slow component's phase turned), H_eff(t) and its gradient, each channel's L(t) and its gradient, the photons' forms and
/// the lift M(t), against Σ_c X⁽ᶜ⁾ e^{−if_c t} of the same fold's components (`Local::components`). The two are one set
/// of sums taken in two orders, so they differ by rounding alone: per component, the phase -- e^{−if_c t}e^{−if_c′ t}
/// against e^{−i(f_c + f_c′)t}, the frequency's own rounding u|f_c t| (a channel's absolute frequency reaches the
/// tweezer's 10⁸ rad/µs) and the cis's, a few u: within 64u (1 + |f_c t|) max|X⁽ᶜ⁾| per component, summed over the
/// quantity's components --, and the products in V's basis, pair by pair against class sum by class sum: κ(V)u relative
/// to the quantity's largest entry (the decomposition's rounding, `rs_at_is_the_exact_fold_to_rounding`'s 10⁻¹⁰). The
/// bound read is their sum (measured: within the phase term alone but for PGC3D's and TET's moving channels, 1.6·10⁻¹² of
/// their largest entry).
#[test]
fn now_is_the_components_at_their_phases() {
    let mut worst = 0.0_f64;
    for (name, scene, keep) in scenes() {
        let rs = Rs::new(&scene, &keep, sectors(&scene));
        for (point, components) in points(&scene) {
            let local = rs.local(&scene, &components.point);
            let model = local.components();
            let n = model.states.len();
            for t in [0.0, 0.37, -1.9] {
                let now = local.now(t);
                let kicks = local.kicks(t);
                // Σ_c X⁽ᶜ⁾ e^{−if_c t} and its rounding bound.
                let at = |parts: &[ModelPart], gradient: Option<usize>, rows: usize, cols: usize| -> (Mat, f64) {
                    let (mut sum, mut bound) = (Mat::zeros(rows, cols), 0.0);
                    for p in parts {
                        let x = gradient.map_or(&p.value, |i| &p.gradient[i]);
                        sum.add_scaled(C64::cis(-p.frequency * t), x);
                        bound += 64.0 * UNIT_ROUNDOFF * (1.0 + (p.frequency * t).abs()) * x.max_abs();
                    }
                    (sum, bound)
                };
                let mut ratios = [0.0_f64; 5];
                let mut compare = |slot: usize, x: &Mat, (y, bound): (Mat, f64), scale: f64| ratios[slot] = crate::linalg::larger(ratios[slot], apart(x, &y).0 / crate::linalg::larger(bound + 1e-10 * scale, f64::MIN_POSITIVE));
                compare(0, &now.hamiltonian, at(&model.effective, None, n, n), now.hamiltonian.max_abs());
                let scale = now.gradient.iter().map(Mat::max_abs).fold(0.0, crate::linalg::larger);
                for i in 0..3 {
                    compare(1, &now.gradient[i], at(&model.effective, Some(i), n, n), scale);
                }
                let (lscale, dscale, kscale) = (
                    now.jumps.iter().map(|c| c.value.max_abs()).fold(0.0, crate::linalg::larger),
                    now.jumps.iter().flat_map(|c| c.gradient.iter().map(Mat::max_abs)).fold(0.0, crate::linalg::larger),
                    kicks.iter().map(|c| c.value.max_abs()).fold(0.0, crate::linalg::larger),
                );
                let zero = Mat::zeros(n, n);
                let channels: std::collections::BTreeSet<(usize, [i64; 2])> = model.jumps.iter().map(|c| (c.emission, c.fast)).chain(now.jumps.iter().map(|c| (c.emission, c.fast))).collect();
                for (emission, fast) in channels {
                    let found = model.jumps.iter().find(|c| c.emission == emission && c.fast == fast);
                    let mine = now.jumps.iter().find(|c| c.emission == emission && c.fast == fast);
                    let kick = kicks.iter().find(|c| c.emission == emission && c.fast == fast);
                    let none: &[ModelPart] = &[];
                    let (parts, forms) = found.map_or((none, none), |c| (&c.parts[..], &c.kicks[..]));
                    compare(2, mine.map_or(&zero, |c| &c.value), at(parts, None, n, n), lscale);
                    for i in 0..3 {
                        compare(2, mine.map_or(&zero, |c| &c.gradient[i]), at(parts, Some(i), n, n), dscale);
                    }
                    compare(3, kick.map_or(&zero, |c| &c.value), at(forms, None, n, n), kscale);
                }
                compare(4, &now.lift, at(&model.lift, None, model.folded.len(), n), now.lift.max_abs());
                eprintln!("{name} {point} t {t}: of the bound H {:.1e} ∇H {:.1e} L {:.1e} K {:.1e} M {:.1e} ({} channels)", ratios[0], ratios[1], ratios[2], ratios[3], ratios[4], now.jumps.len());
                worst = ratios.iter().fold(worst, |w, e| crate::linalg::larger(w, *e));
            }
        }
    }
    assert!(worst <= 1.0, "{worst:e} of the rounding bound");
}

/// The time per point (one thread) of the trajectory's work -- the terms at the point (`Scene::point_at`), the fold there
/// (`Rs::local`) and the value at a time (`Local::now`) -- and of the checkpoint's components (`Local::components`):
/// the means of repeated calls, at the checkpoint's centre and edge and the moving point.
#[test]
#[ignore]
fn rs_at_cost() {
    let time = |f: &mut dyn FnMut()| {
        let start = std::time::Instant::now();
        let mut calls = 0;
        while start.elapsed().as_secs_f64() < 0.5 {
            f();
            calls += 1;
        }
        start.elapsed().as_secs_f64() / calls as f64 * 1e6
    };
    for (name, scene, keep) in scenes() {
        let rs = Rs::new(&scene, &keep, sectors(&scene));
        for (point, components) in points(&scene) {
            let terms = time(&mut || {
                std::hint::black_box(scene.point_at([0.13, -0.21, 0.37], [50.0, -80.0, 120.0]));
            });
            let fold = time(&mut || {
                std::hint::black_box(rs.local(&scene, &components.point));
            });
            let local = rs.local(&scene, &components.point);
            let now = time(&mut || {
                std::hint::black_box(local.now(0.37));
            });
            let by = time(&mut || {
                std::hint::black_box(local.components());
            });
            let dense = time(&mut || {
                std::hint::black_box(scene.components_at([0.13, -0.21, 0.37], [50.0, -80.0, 120.0]));
            });
            let sizes = rs.plan().blocks();
            eprintln!("{name} {point}: point_at {terms:.1} + Rs::local {fold:.1} + now {now:.1} = {:.1} µs per point; components {by:.1} µs; (components_at {dense:.1} µs); blocks {sizes:?}", terms + fold + now);
        }
    }
}

/// The full reference is the prototype's to rounding: PYEXP2's construction -- pyexp2/scene_p.py `build` (each
/// scene's own request, reqprec/req; the tweezer's dissipative fold; scene_x's H0 convention) and pysel/ps.py
/// `hilbert_full3` --, its input by the engine's rule since 3a (model/classify.rs `input`: each coupling entry of the
/// base, every beam's X_l, the tweezer's X_tw and R_tw and every outside-form product zero where its first-order
/// amplitude is below ε at the term's frequency at rest; the emission by its share of its state's decay, classify.rs
/// `emission_input`), as JSON in the directory
/// `V3C_FULL` (made by motion3d_status/rewrite/r1clean_ref/export_rule.py, the files kept beside it). On the six
/// scenes at the checkpoint's centre (r = 0, the tweezer full) and edge (dark), no Doppler shift: H's components summed
/// by the prototype's keys (its beams grouped by frequency, then the carrier and the frame's α) and their gradients
/// along x, y, z; the emission's channels e₁, e₂, u (= x, y, z) and the tweezer's six (co- and counter-rotating,
/// n_tw = ±1), their gradients; each relative to its quantity's largest entry over every key: to 10⁻¹². (Until
/// 2026-11-20 the JSON was v3c2/full, converted from PYSEL's pysel/export -- other requests: the tilt scenes from
/// tilt2's by scene_x `angle_request`, PGC3D in another geometry, its couplings' signs and the emission's axes apart, ∇H
/// and L 1.6 off --; PYEXP2's own-request export2 without the input rule leaves ∇H up to 5.8·10⁻¹⁰ (GMD1C): the
/// request's residues the rule takes, such as cool-plus-x's −9.9·10⁻⁹ coupling to D2-3-0 at a 7 THz detuning, and the
/// outside-form products' entries below it, which in the exact sum cancel the y-gradient of ground-1-1 ← ground-2-0.)
#[test]
fn the_reference_is_pysel_to_rounding() {
    let dir = std::path::PathBuf::from(std::env::var("V3C_FULL").expect("V3C_FULL: the prototype's full references as JSON (motion3d_status/rewrite/r1clean_ref)"));
    let mut worst = 0.0_f64;
    for (name, scene, _) in scenes() {
        let beams = &scene.request.beams;
        let tw = scene.tweezer();
        // The prototype's groups: the non-tweezer beams by frequency, in the order of their first beam.
        let mut indices: Vec<&Vec<i32>> = Vec::new();
        for (l, b) in beams.iter().enumerate() {
            if l != tw && !indices.contains(&&b.frequency_index) {
                indices.push(&b.frequency_index);
            }
        }
        let groups = indices.len();
        let carrier = scene.generators.len() - 1;
        let map = |key: &Key| -> Vec<i64> {
            assert_eq!(key[tw], 0, "a tweezer key left in the reference's H");
            let mut out = vec![0_i64; groups + 2];
            for (l, b) in beams.iter().enumerate().filter(|(l, _)| *l != tw) {
                out[indices.iter().position(|x| **x == b.frequency_index).unwrap()] += i64::from(key[l]);
            }
            out[groups] = i64::from(key[carrier]);
            out
        };
        let zero = vec![0.0; beams.len()];
        for (point, scale) in [("centre", 1.0), ("edge", 0.0)] {
            let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{}_{point}_full.json", name.to_lowercase()))).unwrap()).unwrap();
            let reference = reference(&scene, &scene.components([0.0; 3], &zero, scale));
            let mut summed: BTreeMap<Vec<i64>, (Mat, [Mat; 3])> = BTreeMap::new();
            for p in &reference.effective {
                let entry = summed.entry(map(&p.key)).or_insert_with(|| (Mat::zeros(STATES, STATES), zeros(STATES, STATES)));
                entry.0.add_scaled(C64::ONE, &p.value);
                for (g, d) in entry.1.iter_mut().zip(&p.gradient) {
                    g.add_scaled(C64::ONE, d);
                }
            }
            let (mut differences, mut sizes) = ([0.0_f64; 4], [0.0_f64; 4]);
            let mut check = |slot: usize, rust: Option<&Mat>, python: &Mat| {
                let zero = Mat::zeros(python.rows, python.cols);
                let (d, size) = apart(rust.unwrap_or(&zero), python);
                (differences[slot], sizes[slot]) = (crate::linalg::larger(differences[slot], d), crate::linalg::larger(sizes[slot], size));
            };
            let keys: Vec<Vec<i64>> = doc["H"].as_array().unwrap().iter().map(|h| h["key"].as_array().unwrap().iter().map(|k| k.as_i64().unwrap()).collect()).collect();
            for (h, key) in doc["H"].as_array().unwrap().iter().zip(&keys) {
                let found = summed.get(key);
                check(0, found.map(|x| &x.0), &matrix(&h["value"]));
                for a in 0..3 {
                    check(1, found.map(|x| &x.1[a]), &matrix(&h["grad"][a]));
                }
            }
            for (key, (value, _)) in &summed {
                assert!(keys.contains(key) || value.max_abs() == 0.0, "{name} {point}: a key the prototype has not: {key:?}");
            }
            let channels = doc["C"].as_array().unwrap();
            for c in channels {
                let label = c["name"].as_str().unwrap();
                let axis = ["e1", "e2", "u"].iter().position(|a| label.ends_with(a)).unwrap();
                let fast = if label.starts_with("tw-co") { [0, 1] } else if label.starts_with("tw-cr") { [0, -1] } else { [0, 0] };
                let found = reference.jumps.iter().find(|ch| ch.emission == axis && ch.fast == fast).map(|ch| (summed_parts(&ch.parts, None), [0, 1, 2].map(|i| summed_parts(&ch.parts, Some(i)))));
                check(2, found.as_ref().map(|x| &x.0), &matrix(&c["value"]));
                for a in 0..3 {
                    check(3, found.as_ref().map(|x| &x.1[a]), &matrix(&c["grad"][a]));
                }
            }
            assert_eq!(reference.jumps.len(), channels.len(), "{name} {point}: the channels");
            let errors: Vec<f64> = differences.iter().zip(&sizes).map(|(d, s)| d / crate::linalg::larger(*s, f64::MIN_POSITIVE)).collect();
            eprintln!("{name} {point}: H {:.1e} ∇H {:.1e} L {:.1e} ∇L {:.1e} ({} channels)", errors[0], errors[1], errors[2], errors[3], channels.len());
            worst = errors.iter().fold(worst, |w, e| crate::linalg::larger(w, *e));
        }
    }
    assert!(worst <= 1e-12, "{worst:e}");
}

fn summed_parts(parts: &[ModelPart], gradient: Option<usize>) -> Mat {
    summed(parts, gradient, STATES)
}



/// The tweezer's sidebands are its photons (`Rs::new`): on R1_TWEEZER and R1_SIGMA_P (lone tweezers, the frame's
/// reference: no axis of their own, scene.rs `tweezer_axis`) and R4_LIN_4 (one of its own), the model on the ground states
/// at the trap's centre has every channel's parts at one count of tweezer photons, the channel's n_tw, and the tweezer's
/// co-rotating (n_tw = 1) and counter-rotating (n_tw = −1) channels apart: their interference, 2ν (R1) or 2(ω_tw + ν)
/// apart, is not in the model. Every number finite and both channels lit: on the lone tweezers the co-rotating coupling
/// is static (f = 0), and the unit fold resolves it on the blocks and columns it holds alone (`unit_fold`; a ground
/// state's own block there is singular, and its 0·∞ made the channel NaN, which the largest entry does not show).
#[test]
fn the_tweezer_s_channels_are_its_photons() {
    for name in ["R1_TWEEZER", "R1_SIGMA_P", "R4_LIN_4"] {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let scene = Scene::new(&Request::from_native(&line).unwrap()).unwrap();
        let zero = vec![0.0; scene.request.beams.len()];
        let centre = scene.components([0.0; 3], &zero, 1.0);
        let frame = Frame::new(&scene, &crate::floquet::Hilbert::full(&scene, &centre), &zero);
        let rs = Rs::new(&scene, &(0..crate::request::GROUND).collect::<Vec<_>>(), frame.sectors());
        let model = rs.at(&scene, &centre);
        let tw = scene.tweezer();
        let mut counts = std::collections::BTreeSet::new();
        let finite = |m: &Mat| m.data.iter().all(|z| z.re.is_finite() && z.im.is_finite());
        assert!(model.effective.iter().all(|p| finite(&p.value) && p.gradient.iter().all(finite)), "{name}: H_eff finite");
        let mut lit = std::collections::BTreeMap::new();
        for c in &model.jumps {
            for p in &c.parts {
                assert_eq!(i64::from(p.key[tw]), c.fast[1], "{name}: channel {} {:?}, part at {:?}", c.emission, c.fast, p.key);
                assert!(finite(&p.value) && p.gradient.iter().all(finite), "{name}: channel {} {:?} finite", c.emission, c.fast);
                let size: &mut f64 = lit.entry(c.fast[1]).or_insert(0.0);
                *size = crate::linalg::larger(*size, p.value.max_abs());
            }
            counts.insert(c.fast[1]);
        }
        println!("{name}: the channels' largest entries by n_tw {lit:?}");
        assert!(lit.get(&1).is_some_and(|&x| x > 0.0) && lit.get(&-1).is_some_and(|&x| x > 0.0), "{name}: both tweezer channels lit");
        let frame_count = frame.fast_index(&scene.unit(tw))[1];
        println!("{name}: {} channels, their tweezer photons {counts:?}; the frame's count of the tweezer's own key {frame_count}", model.jumps.len());
        assert!(counts.contains(&1) && counts.contains(&-1), "{name}: the co- and counter-rotating channels");
    }
}

/// R6_RAMAN_CO and R6_RAMAN_COUNTER (a Raman pair on D1, ramanB 6834.59 MHz below ramanA, the frame's reference): the
/// frame's ground levels apart by the pair's difference (the Raman coupling static: floquet/frame.rs `Raman`), φ =
/// 42943 rad/µs; at the two checkpoint points the full reference certified with |ρ| ≤ 1 and the model on all 32 states
/// against it, both solved as the selection solves them (populations by level, R, F⃗ over max(|F⃗|, kR)), within 10⁻⁶ --
/// the fold's own fourth order in the fast (n_φ) elements, measured 2·10⁻⁸, 7·10⁻⁹, 2.8·10⁻⁷ --; the selection passes.
/// With both ground levels in one frame (until 2026-11-20) φ was 0, the Raman elements were taken as fast (2·10⁻⁴,
/// F⃗ 1.2·10⁻² on COUNTER), the reference's max|ρ| was 1.4·10⁸ at the centre and singular at the edge.
#[test]
fn the_raman_scenes_frame_and_model() {
    use crate::floquet::{assemble, readouts, select, Liouville};
    for name in ["R6_RAMAN_CO", "R6_RAMAN_COUNTER"] {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let scene = Scene::new(&Request::from_native(&line).unwrap()).unwrap();
        let points = crate::select::points(&scene);
        let all: Vec<usize> = (0..STATES).collect();
        let mut worst = [0.0_f64; 3];
        for (p, point) in points.iter().enumerate() {
            let reference = reference(&scene, point);
            let zero = vec![0.0; scene.request.beams.len()];
            let frame = Frame::new(&scene, &reference.hilbert(), &zero);
            let grounds: Vec<(String, Vec<i64>)> = [0usize, 3].iter().map(|&s| (scene.request.states[s].clone(), frame.theta[s].clone())).collect();
            let solve = |model: &ModelAt| {
                let liouville = Liouville::new(&frame, &model.hilbert());
                let system = assemble(&liouville, &select(&liouville));
                let solved = crate::krylov::solve(&system).expect("consistent");
                let biggest = solved.rho.iter().map(|z| z.abs()).fold(0.0, crate::linalg::larger);
                (readouts(&liouville, &system, &solved.rho), solved.route, solved.certified, biggest)
            };
            let (r, route, certified, biggest) = solve(&reference);
            let model = Rs::new(&scene, &all, frame.sectors()).at(&scene, point);
            let (m, mroute, mcertified, _) = solve(&model);
            let pops = r.levels.iter().zip(&m.levels).map(|(a, b)| (a - b).abs()).fold(0.0, crate::linalg::larger);
            let rate = (r.rate - m.rate).abs() / crate::linalg::larger(r.rate.abs(), f64::MIN_POSITIVE);
            let scale = crate::linalg::larger(r.force.iter().map(|x| x * x).sum::<f64>().sqrt(), scene.generators.last().unwrap().frequency / crate::units::SPEED_OF_LIGHT * r.rate);
            let force = (0..3).map(|i| (r.force[i] - m.force[i]).powi(2)).sum::<f64>().sqrt() / scale;
            println!(
                "{name} point {p}: grounds {grounds:?}, φ {:.4} rad/µs; reference {route:?} certified {certified} max|ρ| {biggest:.2e}, R {:.4e}; P 32 {mroute:?} {mcertified}: populations {pops:.2e}, R {rate:.2e}, F {force:.2e}",
                frame.fast[0], r.rate
            );
            for (w, x) in worst.iter_mut().zip([pops, rate, force]) {
                *w = crate::linalg::larger(*w, x);
            }
            assert!((frame.fast[0] - 42943.0).abs() < 1.0, "{name}: φ {}", frame.fast[0]);
            assert!(certified && mcertified && biggest <= 1.0 + 1e-9, "{name} point {p}: the solves");
        }
        let chosen = crate::select::choose(&scene);
        println!("{name}: P 32 against the reference at worst {worst:?}; the selection {:?}", chosen.as_ref().map(|s| s.keep.len()));
        assert!(worst.iter().all(|&w| w <= 1e-6), "{name}: P 32 {worst:?}");
        assert!(chosen.is_ok(), "{name}: {chosen:?}");
    }
}

/// The fold's input read by the first-order amplitude rule (model/classify.rs `input`), said per scene: the entries left out of
/// each coupling of H's terms at rest, their largest amplitude a (below ε), and the largest of those kept below
/// u·max|x| of their own matrix (what the matrix's own rounding would have left out, kept as amplitudes).
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn the_fold_s_input_by_its_amplitudes() {
    let dir = std::path::PathBuf::from(std::env::var("V3C_REQ").expect("V3C_REQ"));
    for file in ["tilt0.json", "tilt2.json", "R4_LIN_4.json", "GMD1C_1ms.json", "PGC3D_1ms.json", "TET_1ms.json"] {
        let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap();
        line["options"]["device"] = serde_json::json!("cpu");
        for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
            form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
        }
        let request = crate::request::Request::from_native(&line).unwrap();
        let scene = Scene::new(&request).unwrap();
        let (energy, width) = levels_of(&scene);
        let mut frequency = vec![0.0; scene.matrices()];
        for t in scene.terms(&scene.doppler([0.0; 3])) {
            frequency[t.matrix] = t.frequency;
        }
        let (mut dropped, mut most_dropped, mut kept_small, mut most_kept_small) = (0, 0.0_f64, 0, 0.0_f64);
        for i in 0..scene.matrices() {
            let m = scene.matrix(i);
            let top = m.max_abs();
            for a in 0..STATES {
                for b in 0..STATES {
                    let z = m[(a, b)];
                    if z == C64::ZERO {
                        continue;
                    }
                    let amplitude = z.abs() / C64::new(energy[a] - energy[b] - frequency[i], 0.5 * (width[a] + width[b])).abs();
                    if amplitude < f64::EPSILON {
                        dropped += 1;
                        most_dropped = crate::linalg::larger(most_dropped, amplitude);
                    } else if z.abs() < UNIT_ROUNDOFF * top {
                        kept_small += 1;
                        most_kept_small = crate::linalg::larger(most_kept_small, amplitude);
                    }
                }
            }
        }
        eprintln!("AMPLITUDES {file}: {dropped} entries left out (largest a {most_dropped:.2e}); {kept_small} kept below u·max of their matrix (largest a {most_kept_small:.2e})");
    }
}

#[path = "zeros.rs"]
mod zeros;
