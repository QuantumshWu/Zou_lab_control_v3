//! The path errors' checks and measurements (paths.rs), each ignored and run by name with the scenes' requests
//! (V3C_REQ: the scratchpad's reqprec\req):
//! - `the_exponential_is_exact`: the Taylor action against the decomposed exponential of the sector's Liouvillian, and
//!   the Liouvillian's trace kept, on two scenes' models;
//! - `paths_replay` (PATHS_*): the models' readings along given paths (ENGFOLD's and PATHGAP's exported paths and
//!   windows) by a scheme, to a JSON file for the comparison with the prototypes;
//! - `paths_memory`: τ_mem and L of the models at the trap's centre at rest;
//! - `paths_pilot`: the pilot's paths handed out by the solve (solve.rs `walks`), and the path errors along them;
//! - `paths_cost`: the work of a model a path's µs, one thread, the least of five;
//! - `paths_check`: the scheme's check against its halves on a path piece (`check`);
//! - `paths_a_family_is_its_members`: a family's series are its members' runs alone (`Form::along_family`).
use super::*;
use crate::request::Request;
use crate::scene::Scene;
use crate::select::Selection;

fn var(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("{k}"))
}

/// A scene's request by its name (V3C_REQ; the outside forms' transitions filled in, as model/selected/tests.rs reads them).
fn request_of(name: &str) -> Request {
    let dir = std::path::PathBuf::from(var("V3C_REQ"));
    let file = match name {
        "tilt0" => "tilt0.json",
        "tilt2" => "tilt2.json",
        "R4" => "R4_LIN_4.json",
        "GMD1C" => "GMD1C_1ms.json",
        "PGC3D" => "PGC3D_1ms.json",
        "TET" => "TET_1ms.json",
        other => panic!("unknown scene {other}"),
    };
    let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap();
    line["options"]["device"] = serde_json::json!("cpu");
    for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
        form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
    }
    Request::from_native(&line).unwrap()
}

/// A model's states by its name: G (the full-ground model, the chain's first; P its old name), full (the 32 states),
/// c<n> (the selection's chain's candidate of n states), w<n> (the ground states and the whole levels of the largest
/// structural amplitudes, n states).
fn keep_of(spec: &str, selection: &Selection, labels: &[String]) -> Vec<usize> {
    match spec {
        "G" | "P" => selection.chain[0].clone(),
        "full" => (0..STATES).collect(),
        s if s.starts_with('c') => {
            let n: usize = s[1..].parse().unwrap();
            selection.chain.iter().find(|c| c.len() == n).unwrap_or_else(|| panic!("no candidate of {n} on the chain")).clone()
        }
        s if s.starts_with('w') => {
            let n: usize = s[1..].parse().unwrap();
            let mut levels = crate::model::rs::levels(labels, &(GROUND..STATES).collect::<Vec<_>>());
            let top = |l: &Vec<usize>| l.iter().map(|&e| selection.amplitudes[e]).fold(0.0, larger);
            levels.sort_by(|a, b| top(b).total_cmp(&top(a)));
            let mut keep: Vec<usize> = (0..GROUND).collect();
            for level in levels {
                if keep.len() >= n {
                    break;
                }
                keep.extend(level);
            }
            keep.sort_unstable();
            assert_eq!(keep.len(), n, "no whole-level model of {n} states");
            keep
        }
        other => panic!("unknown model {other}"),
    }
}

/// The scene's models by their names (`keep_of`), each the run's model on its P (solve.rs `run_of`'s, its reference state
/// the request's chief).
fn models(request: &Request, specs: &[String]) -> Vec<Selected> {
    let scene = Scene::new(request).unwrap();
    let selection = crate::select::choose(&scene).unwrap();
    let reference = crate::solve::alone(crate::solve::chief(request));
    specs
        .iter()
        .map(|s| {
            let keep = keep_of(s, &selection, &request.states);
            Selected::new(Scene::new(request).unwrap(), Selection { keep, ..selection.clone() }, reference).unwrap()
        })
        .collect()
}

/// The trap's centre of a model (the reported trap of the chief state).
fn centre(model: &Selected, request: &Request) -> V3 {
    crate::solve::reported(model, &crate::solve::alone(crate::solve::chief(request))).unwrap().centre
}

/// ENGFOLD's exported paths (`<scene>_paths.json`: per path its step ends as samples) as walks from `state`.
fn walks_of(file: &str, state: usize) -> Vec<Walk> {
    let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
    doc["paths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Walk { index: p["index"].as_u64().unwrap(), start: vec![(state, C64::ONE)], ends: serde_json::from_value(p["samples"].clone()).unwrap(), lost: false })
        .collect()
}

/// The Taylor action is the exponential (module documentation, # The segments): on GMD1C's P and its full model and on
/// R4's at their traps' centres, for a sector vector of every entry and h = 1/4 and 1 µs, e^{hL}v against R e^{hΛ} R⁻¹ v
/// of the dense sector Liouvillian's decomposition within 1e-11 of ‖v‖ (the decomposition's own error, κ(R) u ‖hL‖,
/// is the larger); and the trace kept: Σ_a (L e_j)_aa = 0 for every column to the rounding of its entries.
#[test]
#[ignore]
fn the_exponential_is_exact() {
    for (scene, specs) in [("GMD1C", vec!["P".to_string(), "full".to_string()]), ("R4", vec!["P".to_string(), "full".to_string()])] {
        let request = request_of(scene);
        for model in models(&request, &specs) {
            let form = Form::new(&model, centre(&model, &request));
            let l = form.liouvillian(centre(&model, &request), [0.0; 3], 0.0);
            let d = form.dim;
            // The trace: the diagonal places of the sector's pairs (X, X).
            let diagonal: Vec<usize> = (0..form.n).filter_map(|a| form.place(a, a)).collect();
            for j in 0..d {
                let (sum, scale) = diagonal.iter().fold((C64::ZERO, 0.0), |(s, m), &i| (s + l[(i, j)], m + l[(i, j)].abs()));
                assert!(sum.abs() <= 1e-12 * larger(scale, 1.0), "{scene} P{}: column {j} changes the trace by {sum:?} of {scale}", form.n);
            }
            let mut eigen = Eigen::default();
            assert!(eigen.decompose(&l));
            let v: Vec<C64> = (0..d).map(|i| C64::new((0.37 * i as f64).sin(), (0.91 * i as f64 + 0.2).cos())).collect();
            let norm = v.iter().map(|z| z.abs()).fold(0.0, larger);
            let node = model.node(centre(&model, &request), [0.0; 3], false);
            let amps = form.amps(&node);
            for h in [0.25, 1.0] {
                let mut x = v.clone();
                let mut counts = Counts::default();
                form.exp(&[(&amps, 1.0, 0.0)], h, &mut x, &mut Work::new(&form), &mut counts);
                // R e^{hΛ} R⁻¹ v.
                let w: Vec<C64> = (0..d).map(|i| (0..d).map(|j| eigen.inverse[(i, j)] * v[j]).sum::<C64>() * (eigen.values[i] * h).exp()).collect();
                let y: Vec<C64> = (0..d).map(|i| (0..d).map(|j| eigen.vectors[(i, j)] * w[j]).sum()).collect();
                let err = x.iter().zip(&y).map(|(a, b)| (*a - *b).abs()).fold(0.0, larger);
                eprintln!("{scene} P{} sector {d}: h {h}: |e^hL v − R e^hΛ R⁻¹ v| {err:.2e} of {norm:.2} ({} products), κ(R) {:.1e}", form.n, counts.products, eigen.condition);
                assert!(err <= 1e-11 * norm.max(1.0) + 1e-14 * eigen.condition * norm, "{scene} P{}: h {h}: {err:e}", form.n);
            }
        }
    }
}

/// The scheme of PATHS_SCHEME: "exact:<h>:<read>" (ENGFOLD's), "sparse:<nodes>[:<turn>[:<read>]]", or
/// "mid[:<tol>[:<read>]]" / "cf4[:<tol>[:<read>]]" (`Scheme::Steps`), "exponential:<substeps a µs>[:<pieces a step>[:<read>[:<the
/// opening's substeps a turn>]]]".
fn scheme_of(text: &str) -> Scheme {
    let p: Vec<&str> = text.split(':').collect();
    match p[0] {
        "exact" => Scheme::Exact { h: p[1].parse().unwrap(), read: p[2].parse().unwrap() },
        "sparse" => Scheme::Sparse {
            nodes: p[1].parse().unwrap(),
            turn: p.get(2).map_or(std::f64::consts::TAU, |x| x.parse().unwrap()),
            read: p.get(3).map_or(1, |x| x.parse().unwrap()),
        },
        "exponential" => {
            Scheme::Exponential {
                cutoff: f64::INFINITY,
                turns: p.get(4).map_or(1.0, |x| x.parse().unwrap()),
                rate: p[1].parse().unwrap(),
                pieces: p.get(2).map_or(1, |x| x.parse().unwrap()),
                read: p.get(3).map_or(1, |x| x.parse().unwrap()),
            }
        }
        "mid" | "cf4" => Scheme::Steps {
            gauss: p[0] == "cf4",
            tol: p.get(1).map_or(0.0, |x| x.parse().unwrap()),
            read: p.get(2).map_or(1, |x| x.parse().unwrap()),
        },
        other => panic!("unknown scheme {other}"),
    }
}

fn reading_json(r: &Reading) -> serde_json::Value {
    serde_json::json!([r.t, r.rows.to_vec(), r.rates, r.force, r.total, r.trace, r.r, r.kappa])
}

/// The models' readings along given paths (module documentation): PATHS_SCENE, PATHS_FILE (ENGFOLD's
/// `<scene>_paths.json`), PATHS_WINDOWS (its windows file: [path, t0, t1, h, read] each; a window's start state the
/// state PATHS_STATE, default 7), PATHS_MODELS (`keep_of`'s names), PATHS_SCHEME (`scheme_of`; "engfold" takes each
/// window's own h and read as `Scheme::Exact`), PATHS_OUT (the JSON written: per model per window its readings
/// [t, rows, rates, force, total, trace, r, κ], the classes, the counts). On the pool, one job a model and window.
#[test]
#[ignore]
fn paths_replay() {
    let scene = var("PATHS_SCENE");
    let specs: Vec<String> = var("PATHS_MODELS").split(',').map(String::from).collect();
    let state: usize = std::env::var("PATHS_STATE").map_or(7, |s| s.parse().unwrap());
    let walks = walks_of(&var("PATHS_FILE"), state);
    // The windows [path, t0, t1, h, read]; a window's start state the state PATHS_STATE, or its earlier run's (PATHS_RESUME).
    let windows: Vec<(usize, f64, f64, f64, usize)> = serde_json::from_str(&std::fs::read_to_string(var("PATHS_WINDOWS")).unwrap()).unwrap();
    let scheme_text = var("PATHS_SCHEME");
    let clock = std::time::Instant::now();
    let request = request_of(&scene);
    let runs = models(&request, &specs);
    let at = centre(&runs[0], &request);
    let forms: Vec<Form> = runs.iter().map(|m| Form::new(m, at)).collect();
    let made = clock.elapsed().as_secs_f64();
    let mut jobs: Vec<(usize, usize)> = (0..forms.len()).flat_map(|m| (0..windows.len()).map(move |w| (m, w))).collect();
    jobs.sort_by(|a, b| forms[b.0].dim.cmp(&forms[a.0].dim));
    // PATHS_RESUME: an earlier run's output; a window goes on from its window of the same model and path that ends at
    // this one's start (that one's end state), the chain's origin carried.
    let resume: Option<serde_json::Value> = std::env::var("PATHS_RESUME").ok().map(|f| serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap());
    let earlier = |spec: &str, p: usize, t0: f64| -> Option<(Vec<C64>, f64)> {
        let r = resume.as_ref()?;
        let w = r[spec]["windows"].as_array()?.iter().find(|w| w["path"].as_u64() == Some(p as u64) && w["t1"].as_f64() == Some(t0))?;
        let state = w["state"].as_array()?.iter().map(|z| C64::new(z[0].as_f64().unwrap(), z[1].as_f64().unwrap())).collect();
        Some((state, w["origin"].as_f64().unwrap()))
    };
    let origins: Vec<Vec<f64>> = specs.iter().map(|spec| windows.iter().map(|w| earlier(spec, w.0, w.1).map_or(w.1, |e| e.1)).collect()).collect();
    let done: Vec<((usize, usize), Series)> = jobs
        .par_iter()
        .map(|&(m, w)| {
            let (p, t0, t1, h, read) = windows[w];
            let scheme = if scheme_text == "engfold" { Scheme::Exact { h, read } } else { scheme_of(&scheme_text) };
            let s = match earlier(&specs[m], p, t0) {
                Some((state, _)) => forms[m].along_from(&walks[p], state, t0, t1, scheme),
                None => forms[m].along(&walks[p], &[(state, C64::ONE)], t0, t1, scheme),
            };
            ((m, w), s.unwrap())
        })
        .collect();
    let mut out = serde_json::Map::new();
    for (m, spec) in specs.iter().enumerate() {
        let mut total = Counts::default();
        let mut per = Vec::new();
        for ((mm, w), s) in &done {
            if *mm != m {
                continue;
            }
            total.add(&s.counts);
            per.push(serde_json::json!({"window": w, "path": windows[*w].0, "t0": windows[*w].1, "t1": windows[*w].2, "origin": origins[m][*w],
                "readings": s.readings.iter().map(reading_json).collect::<Vec<_>>(), "state": s.state.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>()}));
        }
        let (blocks, pairs) = forms[m].blocks();
        eprintln!(
            "{scene} {spec} (P{}, sector {} = blocks {blocks:?}, {pairs} pairs): {} windows; nodes {} ({:.1} s), exponentials {} with {} products ({:.1} s), segments {}, readouts {} with {} calls ({:.1} s); a path's µs {:.4} s, {:.0} products",
            forms[m].n, forms[m].dim, per.len(), total.nodes, total.node_seconds, total.exponentials, total.products, total.propagation_seconds, total.segments, total.readouts, total.readout_calls, total.readout_seconds, total.seconds() / total.time, total.products as f64 / total.time
        );
        out.insert(spec.clone(), serde_json::json!({"keep": runs[m].rs.keep, "classes": done.iter().find(|x| x.0 .0 == m).unwrap().1.classes, "windows": per,
            "counts": {"nodes": total.nodes, "exponentials": total.exponentials, "products": total.products, "segments": total.segments, "readouts": total.readouts,
            "calls": total.readout_calls, "node_s": total.node_seconds, "prop_s": total.propagation_seconds, "read_s": total.readout_seconds, "time": total.time}}));
    }
    // The pairs' direct errors (`compare`): PATHS_PAIRS "a-b,..." (candidate − reference), per start time of the
    // windows over the paths starting there, the windows PATHS_EDGES (µs from the start, increasing).
    if let Some(pairs) = std::env::var("PATHS_PAIRS").ok().filter(|p| p.contains('-') && p.len() > 1) {
        let edges: Vec<f64> = var("PATHS_EDGES").split(',').map(|x| x.parse().unwrap()).collect();
        let mut starts: Vec<f64> = windows.iter().map(|w| w.1).collect();
        starts.sort_by(f64::total_cmp);
        starts.dedup();
        let mut stats = serde_json::Map::new();
        for pair in pairs.split(',') {
            let (a, b) = pair.split_once('-').unwrap();
            let (ma, mb) = (specs.iter().position(|s| s == a).unwrap(), specs.iter().position(|s| s == b).unwrap());
            for &t0 in &starts {
                let of = |m: usize| -> Vec<Series> {
                    let mut x: Vec<(usize, &Series)> = done.iter().filter(|((mm, w), _)| *mm == m && windows[*w].1 == t0).map(|((_, w), s)| (windows[*w].0, s)).collect();
                    x.sort_by_key(|y| y.0);
                    x.into_iter().map(|y| y.1.clone()).collect()
                };
                let wins: Vec<(f64, f64)> = edges.windows(2).map(|e| (t0 + e[0], t0 + e[1])).collect();
                let result = compare(&of(ma), &of(mb), &wins, &request.states, runs[0].hbar_over_mass());
                for w in &result {
                    eprintln!("{pair} {}", summary(w, &request.states));
                }
                stats.insert(format!("{pair}@{t0}"), serde_json::Value::Array(result.iter().map(window_json).collect()));
            }
        }
        out.insert("stats".into(), serde_json::Value::Object(stats));
    }
    std::fs::write(var("PATHS_OUT"), serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("{scene}: models {specs:?} made in {made:.1} s; all in {:.1} s", clock.elapsed().as_secs_f64());
}

/// A window's errors in a line (ENGFOLD's an2.py's quantities): the ground F=1 sum, the worst row, the paths' worst rows,
/// the total rate's relative difference, the force's rms over the force's, the drive.
fn summary(w: &Window, labels: &[String]) -> String {
    let f1 = w.levels.iter().find(|l| l.0 == "ground-1").map_or(Mean::default(), |l| l.1);
    let (worst, m) = w.worst;
    let per = w.worst_per_path.iter().sum::<f64>() / w.paths as f64;
    format!(
        "({:.0},{:.0}] {} paths: F=1 {:+.5} ± {:.5}; worst row {} {:+.5} ± {:.5}; per path worst mean {per:.5} max {:.5}; rate rel {:+.4} ± {:.4}; force rms rel {:.4} ± {:.4}; energy rate {:.2e} ± {:.1e} /µs²",
        w.from, w.to, w.paths, f1.value, f1.error, labels[worst], m.value, m.error, w.worst_per_path.iter().copied().fold(0.0, larger), w.rate.value, w.rate.error, w.force_rms.value, w.force_rms.error, w.energy_rate.iter().map(|m| m.value).sum::<f64>(), w.energy_rate.iter().map(|m| m.error * m.error).sum::<f64>().sqrt()
    )
}

fn mean_json(m: &Mean) -> serde_json::Value {
    serde_json::json!([m.value, m.error])
}

fn window_json(w: &Window) -> serde_json::Value {
    serde_json::json!({
        "from": w.from, "to": w.to, "paths": w.paths, "rows": w.rows.iter().map(mean_json).collect::<Vec<_>>(),
        "levels": w.levels.iter().map(|(n, m)| serde_json::json!([n, m.value, m.error])).collect::<Vec<_>>(),
        "worst": [w.worst.0, w.worst.1.value, w.worst.1.error], "worst_per_path": w.worst_per_path,
        "rates": w.rates.iter().map(|(f, m, r)| serde_json::json!([f, m.value, m.error, r])).collect::<Vec<_>>(),
        "rate": mean_json(&w.rate), "force": w.force.iter().map(mean_json).collect::<Vec<_>>(), "force_rms": mean_json(&w.force_rms), "heating": w.heating.iter().map(mean_json).collect::<Vec<_>>(), "energy_rate": w.energy_rate.iter().map(mean_json).collect::<Vec<_>>(),
    })
}

/// τ_mem and L (module documentation, # The memory) of PATHS_MODELS on PATHS_SCENE at the trap's centre at rest, for a
/// run of the request's length; the eigenvalues' count and seconds.
#[test]
#[ignore]
fn paths_memory() {
    let scene = var("PATHS_SCENE");
    let specs: Vec<String> = var("PATHS_MODELS").split(',').map(String::from).collect();
    let request = request_of(&scene);
    let run = request.times_us.iter().copied().fold(0.0, larger);
    for (spec, model) in specs.iter().zip(models(&request, &specs)) {
        let at = centre(&model, &request);
        let form = Form::new(&model, at);
        let clock = std::time::Instant::now();
        let m = memory(&form, at, [0.0; 3], 0.0).expect("the sector's Liouvillian decomposes");
        eprintln!("{scene} {spec} (P{}, sector {}): slowest nonzero decay {:.4e} /µs ({} rounding zeros), next {:.3e} {:.3e} {:.3e}: τ_mem {:.4} µs, L {:.1} µs of {run} µs ({:.2} s)", form.n, form.dim, m.rate, m.zeros, m.rates[0], m.rates[1], m.rates[2], m.time, m.length(run), clock.elapsed().as_secs_f64());
    }
}

/// The pilot's paths through the solve (solve.rs `walks`, its hand-out) and the path errors along them: PATHS_SCENE with
/// its outputs cut at PATHS_END µs, PATHS_COUNT paths; PATHS_MODELS (the first the reference), PATHS_NODES (the
/// scheme's parts a step); the windows the octaves of τ_mem below L down to the step. Each path's step ends increase
/// from t = 0 and hold the trajectory's own initial state; the errors per window printed.
#[test]
#[ignore]
fn paths_pilot() {
    let scene = var("PATHS_SCENE");
    let end: f64 = var("PATHS_END").parse().unwrap();
    let count: usize = var("PATHS_COUNT").parse().unwrap();
    let specs: Vec<String> = var("PATHS_MODELS").split(',').map(String::from).collect();
    let nodes: usize = std::env::var("PATHS_NODES").map_or(4, |s| s.parse().unwrap());
    let mut request = request_of(&scene);
    let mut times: Vec<f64> = request.times_us.iter().copied().filter(|&t| t < end).collect();
    times.push(end);
    request.times_us = times;
    let clock = std::time::Instant::now();
    WALKS.with(|w| *w.borrow_mut() = Some((count, end, vec![])));
    crate::solve::solve(&request, &mut |_| {}).unwrap();
    let walks = WALKS.with(|w| w.borrow_mut().take().unwrap().2);
    let solved = clock.elapsed().as_secs_f64();
    assert_eq!(walks.len(), count, "the pilot hands out {count} paths");
    for w in &walks {
        assert!(w.ends[0][0] == 0.0 && w.ends.windows(2).all(|x| x[1][0] > x[0][0]), "path {}: its step ends increase from 0", w.index);
        let norm: f64 = w.start.iter().map(|x| x.1.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-12 && w.start.iter().all(|x| x.0 < GROUND), "path {}: its start a normalized ground state", w.index);
        let steps: Vec<f64> = w.ends.windows(2).map(|x| x[1][0] - x[0][0]).collect();
        eprintln!("path {}: {} steps to {:.2} µs (step {:.4} … {:.4} µs), start {:?}", w.index, steps.len(), w.end(), steps.iter().copied().fold(f64::INFINITY, f64::min), steps.iter().copied().fold(0.0, larger), w.start);
    }
    let runs = models(&request, &specs);
    let at = centre(&runs[0], &request);
    let forms: Vec<Form> = runs.iter().map(|m| Form::new(m, at)).collect();
    let mem = memory(forms.iter().min_by_key(|f| f.dim).unwrap(), at, [0.0; 3], 0.0).unwrap();
    let length = mem.length(end);
    let step = walks[0].ends[1][0] - walks[0].ends[0][0];
    let windows = super::windows(length, mem.time, step);
    assert!(windows[0].0 == 0.0 && windows.last().unwrap().1 == length && windows.windows(2).all(|w| w[0].1 == w[1].0), "the windows tile [0, L]");
    let taken: Vec<Option<usize>> = request.times_us.iter().map(|&t| window_of(&windows, t)).collect();
    assert!(request.times_us.iter().zip(&taken).all(|(&t, w)| w.is_some() == (t > 0.0)), "every output after t = 0 takes a window");
    eprintln!("{scene}: solved in {solved:.1} s; τ_mem {:.3} µs (the smallest model's), L {length:.1} µs; windows {windows:?}; the outputs' windows {taken:?}", mem.time);
    let candidates: Vec<&Form> = forms[1..].iter().collect();
    let clock = std::time::Instant::now();
    let errors = path_errors(&forms[0], &candidates, &walks, &|w: &Walk| w.start.clone(), 0.0, length, &windows, Scheme::sparse(nodes)).unwrap();
    let labels = &request.states;
    for (c, spec) in specs[1..].iter().enumerate() {
        for w in &errors.windows[c] {
            let (worst, m) = w.worst;
            let mean_worst = w.worst_per_path.iter().sum::<f64>() / w.paths as f64;
            eprintln!(
                "{spec} − {}: ({:.1}, {:.1}] {} paths: worst row {} {:+.5} ± {:.5} (per path mean {mean_worst:.5}, max {:.5}); rate {:+.4} ± {:.4}; force rms {:.4} ± {:.4}; energy rate {:.2e}",
                specs[0], w.from, w.to, w.paths, labels[worst], m.value, m.error, w.worst_per_path.iter().copied().fold(0.0, larger), w.rate.value, w.rate.error, w.force_rms.value, w.force_rms.error, w.energy_rate.iter().map(|m| m.value).sum::<f64>()
            );
        }
    }
    for (m, spec) in specs.iter().enumerate() {
        let mut total = Counts::default();
        errors.series[m].iter().for_each(|s| total.add(&s.counts));
        eprintln!("{spec} (sector {}): a path's µs {:.4} s ({:.0} products, {:.1} nodes, {:.1} readout calls)", forms[m].dim, total.seconds() / total.time, total.products as f64 / total.time, total.nodes as f64 / total.time, total.readout_calls as f64 / total.time);
    }
    eprintln!("{scene}: path errors in {:.1} s", clock.elapsed().as_secs_f64());
}

/// The work of PATHS_MODELS a path's µs (one thread; the test run alone, pinned by its caller): path PATHS_PATH of
/// PATHS_FILE over [PATHS_FROM, PATHS_TO] by PATHS_SCHEME, five times, the least seconds and the counts.
#[test]
#[ignore]
fn paths_cost() {
    let scene = var("PATHS_SCENE");
    let specs: Vec<String> = var("PATHS_MODELS").split(',').map(String::from).collect();
    let walks = walks_of(&var("PATHS_FILE"), 7);
    let path: usize = var("PATHS_PATH").parse().unwrap();
    let (from, to): (f64, f64) = (var("PATHS_FROM").parse().unwrap(), var("PATHS_TO").parse().unwrap());
    let scheme = scheme_of(&var("PATHS_SCHEME"));
    let request = request_of(&scene);
    let runs = models(&request, &specs);
    let at = centre(&runs[0], &request);
    for (spec, model) in specs.iter().zip(&runs) {
        let form = Form::new(model, at);
        let node = model.node(at, [0.0; 3], false);
        let amps = form.amps(&node);
        let active = (0..form.h_beats.len()).filter(|&k| amps.h[k * form.h_slot..(k + 1) * form.h_slot].iter().any(|z| *z != C64::ZERO)).count();
        eprintln!("{spec}: H slots {} ({active} with a sector entry at the centre), slot {} entries; jumps {} channels, {} elements, R {} × {} by {} beat groups, {} terms; width {:.0} rad/µs", form.h_beats.len(), form.h_slot, 3 * node.classes.len(), form.elements, form.rows.len(), form.cols.len(), form.beats_r.len(), form.terms.len(), amps.width);
        let mut best: Option<Counts> = None;
        for _ in 0..5 {
            let s = form.along(&walks[path], &[(7, C64::ONE)], from, to, scheme).unwrap();
            if best.is_none_or(|b| s.counts.seconds() < b.seconds()) {
                best = Some(s.counts);
            }
        }
        let c = best.unwrap();
        let us = c.time;
        eprintln!(
            "{scene} {spec} (P{}, sector {}): a path's µs {:.4} s = nodes {:.4} + propagation {:.4} + readouts {:.4}; a µs: {:.1} nodes, {:.0} segments, {:.0} exponentials, {:.0} products, {:.1} readouts with {:.1} calls; {:.2} µs a product",
            form.n, form.dim, c.seconds() / us, c.node_seconds / us, c.propagation_seconds / us, c.readout_seconds / us, c.nodes as f64 / us, c.segments as f64 / us, c.exponentials as f64 / us, c.products as f64 / us, c.readouts as f64 / us, c.readout_calls as f64 / us, 1e6 * c.propagation_seconds / c.products as f64
        );
    }
}

/// The scheme's check (`check`; module documentation, # The segments): PATHS_MODELS "candidate,reference" on path
/// PATHS_PATH of PATHS_FILE over [PATHS_FROM, PATHS_TO] at `Scheme::sparse(PATHS_NODES)` (or PATHS_SCHEME) against its halves; the
/// largest change of the difference's row, rate and force; and that a turn past 2π is refused.
#[test]
#[ignore]
fn paths_check() {
    let scene = var("PATHS_SCENE");
    let specs: Vec<String> = var("PATHS_MODELS").split(',').map(String::from).collect();
    let walks = walks_of(&var("PATHS_FILE"), 7);
    let path: usize = var("PATHS_PATH").parse().unwrap();
    let (from, to): (f64, f64) = (var("PATHS_FROM").parse().unwrap(), var("PATHS_TO").parse().unwrap());
    let nodes: usize = var("PATHS_NODES").parse().unwrap();
    let request = request_of(&scene);
    let runs = models(&request, &specs);
    let at = centre(&runs[0], &request);
    let (c, r) = (Form::new(&runs[0], at), Form::new(&runs[1], at));
    let start = [(7, C64::ONE)];
    assert!(check(&c, &r, &walks[path], &start, from, to, Scheme::Sparse { nodes, turn: 4.0 * std::f64::consts::PI, read: 1 }).is_none(), "a turn past 2π is refused");
    let clock = std::time::Instant::now();
    let scheme = std::env::var("PATHS_SCHEME").map_or(Scheme::sparse(nodes), |x| scheme_of(&x));
    let [row, rate, force] = check(&c, &r, &walks[path], &start, from, to, scheme).unwrap().unwrap();
    eprintln!("{scene} {} − {}, path {path} on [{from}, {to}] µs, {nodes} nodes a step against {} and half the turn: the difference changes by row {row:.2e}, rate {rate:.2e} (of the rate), force {force:.2e} (of the rms force); {:.1} s", specs[0], specs[1], 2 * nodes, clock.elapsed().as_secs_f64());
}

/// A family's series are its members' runs alone, to the bit (module documentation, # The nodes): on TET's and R4's 32
/// states along a slow line through the trap's centre, a setting and its refinements -- the opening's substeps, the
/// rate, the pieces and a partner's rate doubled or raised alone, at one and at two pieces a step --, every reading's
/// time, rows and rates, the opening, its bound, the condition and the end state the same as each run's alone; its
/// nodes and products fewer.
#[test]
#[ignore]
fn paths_a_family_is_its_members() {
    for scene in ["TET", "R4"] {
        let request = request_of(scene);
        let model = models(&request, &["full".to_string()]).remove(0);
        let c = centre(&model, &request);
        let form = Form::new(&model, c);
        let (speed, step, hm) = (0.03, 0.5, model.hbar_over_mass());
        let ends = (0..=24).map(|k| {
            let t = k as f64 * step;
            [t, c[0] + speed * t, c[1], c[2], speed / hm, 0.0, 0.0]
        });
        let walk = Walk { index: 0, start: vec![(7, C64::ONE)], ends: ends.collect(), lost: false };
        let at = |turns: f64, rate: f64, pieces: usize| Scheme::Exponential { turns, rate, pieces, cutoff: f64::INFINITY, read: 1 };
        for pieces in [1, 2] {
            let members = [at(1.0, 8.0, pieces), at(2.0, 8.0, pieces), at(1.0, 16.0, pieces), at(1.0, 8.0, 2 * pieces), at(1.0, 9.0, pieces)];
            let family = form.along_family(&walk, &walk.start, 0.0, walk.end(), &members).unwrap();
            let alone: Vec<Series> = members.iter().map(|&s| form.along(&walk, &walk.start, 0.0, walk.end(), s).unwrap()).collect();
            let same = |a: &Series, b: &Series| a.readings.len() == b.readings.len() && a.readings.iter().zip(&b.readings).all(|(x, y)| x.t == y.t && x.rows == y.rows && x.rates == y.rates) && a.state == b.state;
            let sum = |x: &[Series], f: &dyn Fn(&Counts) -> u64| x.iter().map(|s| f(&s.counts)).sum::<u64>();
            eprintln!(
                "{scene} P{} pieces {pieces}: the family's nodes {} and products {}; its members' alone {} and {}",
                form.n,
                sum(&family, &|c| c.nodes),
                sum(&family, &|c| c.products),
                sum(&alone, &|c| c.nodes),
                sum(&alone, &|c| c.products)
            );
            for (k, (f, a)) in family.iter().zip(&alone).enumerate() {
                assert!(same(f, a), "{scene} pieces {pieces}: member {k}'s series is not its run alone");
                assert_eq!((f.counts.opening, f.counts.booked, f.counts.condition), (a.counts.opening, a.counts.booked, a.counts.condition), "{scene} pieces {pieces}: member {k}'s opening");
            }
            assert!(!same(&family[0], &family[2]), "{scene} pieces {pieces}: the rate changes nothing");
            assert!(sum(&family, &|c| c.nodes) < sum(&alone, &|c| c.nodes) && sum(&family, &|c| c.products) < sum(&alone, &|c| c.products));
        }
        assert!(form.along_family(&walk, &walk.start, 0.0, walk.end(), &[at(1.0, 8.0, 3), at(1.0, 8.0, 2)]).is_err(), "pieces not powers of two of one another refused");
    }
}
