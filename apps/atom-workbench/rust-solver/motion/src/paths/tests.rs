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
        .map(|p| Walk { index: p["index"].as_u64().unwrap(), start: vec![(state, C64::ONE)], ends: serde_json::from_value(p["samples"].clone()).unwrap(), lost: false, weight: 1.0 })
        .collect()
}

/// The scheme of PATHS_SCHEME: "exponential:<substeps a µs>[:<pieces a step>[:<read>[:<the opening's substeps a
/// turn>]]]".
fn scheme_of(text: &str) -> Scheme {
    let p: Vec<&str> = text.split(':').collect();
    assert_eq!(p[0], "exponential", "the paths' one scheme");
    Scheme::Exponential {
        cutoff: f64::INFINITY,
        turns: p.get(4).map_or(1.0, |x| x.parse().unwrap()),
        rate: p[1].parse().unwrap(),
        pieces: p.get(2).map_or(1, |x| x.parse().unwrap()),
        read: p.get(3).map_or(1, |x| x.parse().unwrap()),
    }
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
        eprintln!("{scene} {spec} (P{}, sector {}): slowest nonzero decay {:.4e} /µs ({} rounding zeros), next {:.3e} {:.3e} {:.3e}: τ_mem {:.4} µs, L {:.1} µs of {run} µs ({:.2} s)", form.n, form.dim, m.rate, m.zeros, m.rates[0], m.rates[1], m.rates[2], m.time, run.min(5.0 * m.time), clock.elapsed().as_secs_f64());
    }
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
        let walk = Walk { index: 0, start: vec![(7, C64::ONE)], ends: ends.collect(), lost: false, weight: 1.0 };
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
