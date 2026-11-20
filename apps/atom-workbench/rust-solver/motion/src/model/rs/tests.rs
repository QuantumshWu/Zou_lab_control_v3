//! The fold against td_s.hilbert_rs (scratchpad v3c\rsx\export.py: the prototype's components at the six scenes' two
//! points, its RS model on the selected P, and the components at r ± δ along the axis), read from the directory named by
//! the environment variable `V3C_RSX`; and `Rs::at` on the six scenes' requests (scratchpad final\req, the directory
//! named by `V3C_REQ`) against the fold with every block decomposed at the point. The data are the check's, not the
//! crate's.

use super::*;
use crate::floquet::Frame;
use crate::request::Request;
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
    Components { parts, tweezer: (C64::ZERO, [C64::ZERO; 3]) }
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
    let source = ModelAt::source(&components(&doc["points"][which], length), &jumps, &labels, |_| false);
    let sectors = Sectors { states: vec![[0; 2]; STATES], generators: vec![[0; 2]; length] };
    fold(&source, &source.generator(), &keep, &sectors, MODEL)
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
        let at = model(&doc, 0);
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

// ---------------------------------------------------------------------------------------------------------------------
// Rs::at on the six scenes' requests.

/// The six scenes' requests (V3C_REQ: scratchpad final\req) with their selected P (pysel.status.md A: the levels added to
/// the ground states).
fn scenes() -> Vec<(&'static str, Scene, Vec<usize>)> {
    let dir = std::path::PathBuf::from(std::env::var("V3C_REQ").expect("V3C_REQ: the directory of the six scenes' requests (scratchpad final\\req)"));
    let chosen: [(&str, &str, &[&str]); 6] = [
        ("tilt0", "tilt0.json", &["D2-3", "D2-2"]),
        ("tilt2", "tilt2.json", &["D2-3", "D2-2"]),
        ("R4", "../../traj/req/R4_LIN_4.json", &["D2-2", "D2-3", "D2-1"]),
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

/// The fold with every block decomposed at the point and every column's resolvent its spectrum (64c973e8's fold, the
/// formula of the module documentation without the series): the check's exact reference.
fn exact(source: &ModelAt, keep: &[usize], sectors: &Sectors, axes: [bool; 2]) -> ModelAt {
    let place = |state: usize| source.states.iter().position(|&s| s == state).unwrap();
    let mut kept: Vec<usize> = keep.iter().map(|&s| place(s)).collect();
    kept.sort_unstable();
    let size = source.states.len();
    let others: Vec<usize> = (0..size).filter(|p| kept.binary_search(p).is_err()).collect();
    let n = kept.len();
    let fast = |key: &Key, to: usize, from: usize| -> [i64; 2] { plus(sectors.key(key, axes), [sectors.state(source.states[to], axes)[0] - sectors.state(source.states[from], axes)[0], sectors.state(source.states[to], axes)[1] - sectors.state(source.states[from], axes)[1]]) };
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
    let mut effective = Sums::default();
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
    let mut out: BTreeMap<(usize, [i64; 2]), Sums> = BTreeMap::new();
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
        d[slot] = d[slot].max(e);
        scale[slot] = scale[slot].max(s);
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
    std::array::from_fn(|i| d[i] / scale[i].max(f64::MIN_POSITIVE))
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
            let (reference, generator) = rs.tweezer.reference(&scene, &components);
            let made = reference.generator();
            let (g, scale) = apart(&generator.value, &made.value);
            assert!(g <= 64.0 * UNIT_ROUNDOFF * scale, "{name} {point}: the generator made once against the point's: {g:e} of {scale:e}");
            let truth = exact(&reference, &rs.keep, &rs.sectors, MODEL);
            let errors = differ(&model, &truth);
            eprintln!("{name} {point}: P {} H {:.1e} ∇H {:.1e} L {:.1e} ∇L {:.1e} ({} parts, {} channels)", model.states.len(), errors[0], errors[1], errors[2], errors[3], model.effective.len(), model.jumps.len());
            worst = errors.iter().fold(worst, |w, e| w.max(*e));
        }
    }
    assert!(worst <= 1e-10, "{worst:e}");
}

/// The time of `Rs::at` per point (one thread), at the moving point of the check: the mean of repeated calls.
#[test]
#[ignore]
fn rs_at_cost() {
    for (name, scene, keep) in scenes() {
        let rs = Rs::new(&scene, &keep, sectors(&scene));
        for (point, components) in points(&scene) {
            let start = std::time::Instant::now();
            let mut calls = 0;
            while start.elapsed().as_secs_f64() < 1.0 {
                std::hint::black_box(rs.at(&scene, &components));
                calls += 1;
            }
            let each = start.elapsed().as_secs_f64() / calls as f64;
            let build = std::time::Instant::now();
            let _ = std::hint::black_box(scene.components_at([0.13, -0.21, 0.37], [50.0, -80.0, 120.0]));
            eprintln!("{name} {point}: Rs::at {:.1} µs per point ({calls} calls); components_at {:.1} µs", each * 1e6, build.elapsed().as_secs_f64() * 1e6);
        }
    }
}

/// The full reference is PYSEL's (scene_p.build with the tweezer's dissipative fold, scene_x's H0 convention; its
/// exports pysel\export\*_full_rs.npz as JSON by scratchpad v3c2\tojson.py in the directory `V3C_FULL`) to rounding, on
/// the six scenes at the checkpoint's centre (r = 0, the tweezer full) and edge (dark), no Doppler shift: H's components
/// summed by the prototype's keys (its beams grouped by frequency, then the carrier and the frame's α) and their
/// gradients along x, y, z; the emission's channels e₁, e₂, u (= x, y, z) and the tweezer's six (co- and
/// counter-rotating, n_tw = ±1), their gradients; each relative to its quantity's largest entry over every key: to
/// 10⁻¹² (measured ≤ 1.0·10⁻¹⁵ on R4, GMD1C, PGC3D, TET). The tilt scenes are read against the differences of their
/// inputs, 10⁻⁷: PYSEL makes both from the tilt2 request (scene_x `angle_request`, the probe turned to 0° or 2° with its
/// direction recomputed), while these requests store the direction to eight digits (0.0348995 for sin 2° =
/// 0.034899497: ∇H 1.7·10⁻⁸) and tilt0's own emission differs from tilt2's by 1.1·10⁻⁷ absolute (L 2.2·10⁻⁸).
#[test]
fn the_reference_is_pysel_to_rounding() {
    let dir = std::path::PathBuf::from(std::env::var("V3C_FULL").expect("V3C_FULL: PYSEL's full references as JSON (scratchpad v3c2/full)"));
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
                (differences[slot], sizes[slot]) = (differences[slot].max(d), sizes[slot].max(size));
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
            let errors: Vec<f64> = differences.iter().zip(&sizes).map(|(d, s)| d / s.max(f64::MIN_POSITIVE)).collect();
            eprintln!("{name} {point}: H {:.1e} ∇H {:.1e} L {:.1e} ∇L {:.1e} ({} channels)", errors[0], errors[1], errors[2], errors[3], channels.len());
            let bound = if name.starts_with("tilt") { 1e-7 } else { 1e-12 };
            worst = errors.iter().fold(worst, |w, e| w.max(e / bound));
        }
    }
    assert!(worst <= 1.0, "{worst:e} of the bound");
}

fn summed_parts(parts: &[ModelPart], gradient: Option<usize>) -> Mat {
    summed(parts, gradient, STATES)
}
