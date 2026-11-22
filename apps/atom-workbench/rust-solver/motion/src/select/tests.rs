//! The selection's checks: the chain by hand and on every reference scene, the middle point, the accounts' bounds by
//! hand; and, run by name, the whole selection on the scenes' requests (`select_on_the_scenes`).
use super::*;
use crate::paths::Mean;

/// The chain by hand (module documentation, # The candidates): the ground states first, the excited states added in
/// decreasing amplitude, a pair of equal amplitude (to the rounding) at once -- the pair GROUND + 2, GROUND + 3 one
/// candidate --, the states of no amplitude last, all 32 at the end; every candidate holds the one before it.
#[test]
fn select_the_chain_by_hand() {
    let mut a = [0.0_f64; STATES];
    a[GROUND] = 0.3;
    a[GROUND + 2] = 0.1;
    a[GROUND + 3] = 0.1 * (1.0 + f64::EPSILON);
    a[GROUND + 5] = 0.02;
    let chain = chain(&a);
    assert_eq!(chain[0], (0..GROUND).collect::<Vec<_>>());
    assert_eq!(chain[1], (0..GROUND).chain([GROUND]).collect::<Vec<_>>());
    assert_eq!(chain[2], (0..GROUND).chain([GROUND, GROUND + 2, GROUND + 3]).collect::<Vec<_>>());
    assert_eq!(chain[3], (0..GROUND).chain([GROUND, GROUND + 2, GROUND + 3, GROUND + 5]).collect::<Vec<_>>());
    assert_eq!(chain.len(), 5, "the states of no amplitude at once");
    assert_eq!(*chain.last().unwrap(), (0..STATES).collect::<Vec<_>>());
    assert!(chain.windows(2).all(|w| w[0].iter().all(|s| w[1].contains(s))), "nested");
}

/// The middle point (`middle`), on two reference requests (refs/scenes): R10_NODE, the trap at a node of the standing
/// wave whose nodal plane holds the tweezer's axis -- the middle leaves the plane (n̂ across the axis, |n̂·d| at
/// rounding), the tweezer's intensity there half its peak to rounding, and the light other than the tweezer's excites
/// the atom there (`excitation` ≥ ε², an amplitude's square at the rounding) where one Rayleigh range along the axis it
/// does not; R1_TWEEZER, the tweezer alone (M = 0) -- the middle on the axis at the Rayleigh range, z_R d, to rounding.
#[test]
fn select_the_middle_point_leaves_a_node() {
    let half = |scene: &Scene, r: V3| scene.geometry[scene.tweezer()].amplitude_to(r, 0).value.norm_sqr();
    let node = Scene::new(&crate::model::testing::scene("R10_NODE")).unwrap();
    let r = middle(&node);
    let d = node.geometry[node.tweezer()].direction;
    let along = (0..3).map(|i| r[i] * d[i]).sum::<f64>() / crate::linalg::norm3(r);
    assert!(along.abs() <= 8.0 * f64::EPSILON, "R10_NODE: the middle along the axis by {along:e}");
    assert!((half(&node, r) - 0.5).abs() <= 8.0 * f64::EPSILON, "R10_NODE: the tweezer's intensity {}", half(&node, r));
    let zero = vec![0.0; node.request.beams.len()];
    let tw = node.tweezer();
    let even = [1.0 / GROUND as f64; GROUND];
    let other = |point: &Components| excitation(&node, &rs::reference(&node, point), point, &|l| l != tw, &even).iter().fold(0.0, |m: f64, &a| larger(m, a));
    let at_middle = other(&node.components(r, &zero, 1.0));
    let on_axis = other(&node.components(d.map(|x| x * node.geometry[tw].rayleigh().unwrap()), &zero, 1.0));
    let floor = f64::EPSILON * f64::EPSILON;
    assert!(at_middle >= floor && on_axis < floor, "R10_NODE: the other light's excitation {at_middle:e} at the middle, {on_axis:e} on the axis");
    let alone = Scene::new(&crate::model::testing::scene("R1_TWEEZER")).unwrap();
    let tweezer = alone.geometry[alone.tweezer()];
    let r = middle(&alone);
    let z = tweezer.rayleigh().unwrap();
    assert!((0..3).all(|i| (r[i] - z * tweezer.direction[i]).abs() <= 8.0 * f64::EPSILON * z), "R1_TWEEZER: {r:?} for z_R {z} along {:?}", tweezer.direction);
}

/// A window of every row's error `row` (no spread over the paths).
fn window(from: f64, to: f64, row: f64) -> Window {
    Window { from, to, paths: 4, rows: vec![Mean { value: row, error: 0.0 }; STATES], ..Window::default() }
}

/// The direct part by hand (module documentation, # The direct part): the start's windows to L = 100 µs, a late segment
/// over [900, 1000] µs; at a time in the start's windows its own window's worst row, past L the larger of the start's last
/// and the late's last, past the late segment's start the late's last; the calibration beside it; none at t = 0.
#[test]
fn select_the_direct_part_by_hand() {
    let judged = Judged {
        keep: vec![],
        windows: vec![window(0.0, 10.0, 4e-3), window(10.0, 100.0, 1e-3)],
        late: vec![window(900.0, 910.0, 9e-3), window(910.0, 1000.0, 2e-3)],
        calibration: Calibration { rows: 1e-5, rate: 0.0 },
        region: Region::NONE,
        seconds: 0.0,
        products: 0,
    };
    let windows = vec![(0.0, 10.0), (10.0, 100.0)];
    let at = |t: f64| direct_at(&judged, &windows, t);
    assert_eq!(at(0.0), 0.0);
    assert!((at(5.0) - 4.01e-3).abs() < 1e-15 && (at(50.0) - 1.01e-3).abs() < 1e-15, "{} {}", at(5.0), at(50.0));
    assert!((at(500.0) - 2.01e-3).abs() < 1e-15 && (at(950.0) - 2.01e-3).abs() < 1e-15, "{} {}", at(500.0), at(950.0));
}

/// The flag's question by hand (`decide`): read at the end (|r| > z·se), yes; its upper end within the allowance, no;
/// else not told; a number that is none tells nothing.
#[test]
fn select_the_memory_flag_by_hand() {
    assert_eq!(decide(3.0, 1.0, 100.0), Some(true));
    assert_eq!(decide(0.5, 1.0, 3.0), Some(false));
    assert_eq!(decide(0.5, 1.0, 2.0), None);
    assert_eq!(decide(f64::NAN, 1.0, 2.0), None);
}

/// The selection's chain on every reference scene (the coordinator's check before the merge; ignored; refs/scenes/*.json
/// but its *.scene.json app scenes and the manifest, read as the model's tests read them, model/testing.rs `scene`) and
/// the six scenes' own requests (`V3D_REQ`): on every one the chain is G first and the 32 states last, nested, every
/// candidate holding the ground states, and G's model is made (the run the scout and the pilot are made on). A scene
/// whose chain or model fails, errs or panics is a failure, all of them said at the end. Each scene's chain (its
/// candidates' sizes), the largest structural amplitude folded in G, and the time. (Until 2026-11-22 the test took the
/// three-point choice, its 1 % gate and its descent, which are gone: the choice is made on the pilot's paths,
/// `select_on_the_scenes`.)
#[test]
#[ignore]
fn choose_on_every_reference_scene() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| f.ends_with(".json") && !f.ends_with(".scene.json") && f != "manifest.json")
        .collect();
    files.sort();
    let mut all: Vec<(String, Box<dyn Fn() -> crate::request::Request>)> = files
        .iter()
        .map(|f| {
            let path = dir.join(f).to_string_lossy().into_owned();
            (f.trim_end_matches(".json").to_string(), Box::new(move || crate::model::testing::scene(&path)) as Box<dyn Fn() -> crate::request::Request>)
        })
        .collect();
    for (name, request) in requests() {
        all.push((format!("{name} (reqprec)"), Box::new(move || request.clone())));
    }
    let clock = Instant::now();
    let mut failed: Vec<String> = Vec::new();
    for (name, read) in &all {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<String, String> {
            let started = Instant::now();
            let request = read();
            let scene = Scene::new(&request)?;
            let selection = choose(&scene)?;
            let c = &selection.chain;
            let ground: Vec<usize> = (0..GROUND).collect();
            if c[0] != ground || c.last().map(|x| x.len()) != Some(STATES) {
                return Err(format!("the chain's ends {:?} … {:?}", c[0], c.last()));
            }
            if !c.windows(2).all(|w| w[1].len() > w[0].len() && w[0].iter().all(|s| w[1].contains(s))) {
                return Err("the chain is not nested".into());
            }
            let run = Selected::new(Scene::new(&request)?, selection.clone(), crate::solve::alone(crate::solve::chief(&request)))?;
            let folded = selection.amplitudes.iter().copied().fold(0.0, larger);
            Ok(format!("chain {} ({}), G's model P{} made; the largest amplitude G folds {folded:.3e}; {:.2} s", c.len(), c.iter().map(|x| x.len().to_string()).collect::<Vec<_>>().join(" "), run.rs.keep.len(), started.elapsed().as_secs_f64()))
        }));
        match outcome {
            Ok(Ok(said)) => eprintln!("{name}: {said}"),
            Ok(Err(why)) => {
                eprintln!("{name}: refused: {why}");
                failed.push(format!("{name}: {why}"));
            }
            Err(payload) => {
                let why = payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                eprintln!("{name}: panicked: {why}");
                failed.push(format!("{name}: panicked: {why}"));
            }
        }
    }
    eprintln!("{} scenes in {:.1} s; {} failed", all.len(), clock.elapsed().as_secs_f64(), failed.len());
    assert!(failed.is_empty(), "{failed:#?}");
}

/// The six scenes' requests (the directory `V3D_REQ`: scratchpad reqprec/req; `PSTATE_SCENE` one of them), each with
/// its name in the prototype's files.
pub(crate) fn requests() -> Vec<(&'static str, crate::request::Request)> {
    let dir = std::path::PathBuf::from(std::env::var("V3D_REQ").expect("V3D_REQ: the six scenes' requests (scratchpad reqprec/req)"));
    [("tilt0.json", "tilt0"), ("tilt2.json", "tilt2"), ("R4_LIN_4.json", "r4"), ("GMD1C_1ms.json", "gmd1c"), ("PGC3D_1ms.json", "pgc3d"), ("TET_1ms.json", "tet")]
        .into_iter()
        .filter(|(_, name)| std::env::var("PSTATE_SCENE").map_or(true, |only| only == *name))
        .map(|(file, name)| (name, read_request(&dir.join(file))))
        .collect()
}

/// A native request file as the tests read the scenes' (the outside forms' transitions filled in: these requests
/// predate them, and the scene does not read them).
fn read_request(p: &std::path::Path) -> crate::request::Request {
    let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap();
    line["options"]["device"] = serde_json::json!("cpu");
    if let Some(forms) = line["model"]["motion"]["outsideForms"].as_array_mut() {
        for form in forms {
            form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
        }
    }
    crate::request::Request::from_native(&line).unwrap()
}

/// The whole selection (module documentation; solve.rs # The selection) on the scenes named by `SELECT_SCENES` (comma-
/// separated: the six's names in `requests`, or a native request's path, e.g. R5_PUMP's and S1's), one after another:
/// G's pilot, the paths, the response, the candidates judged and weighed, P -- its report (every candidate's parts, work
/// and fate; the stages' times), and with `SELECT_OUT` a directory, each scene's choice to <scene>.json there: P, the
/// parts at the response's times, the response's Λ_T, the pairs. `SELECT_ALSO`: candidates beside the chain judged for
/// comparison (`ALSO`), ';'-separated, each the excited states' labels it keeps, ','-separated (the ground states always).
#[test]
#[ignore]
fn select_on_the_scenes() {
    let names = std::env::var("SELECT_SCENES").expect("SELECT_SCENES: the scenes, comma-separated");
    let out = std::env::var("SELECT_OUT").ok().map(std::path::PathBuf::from);
    let six = std::env::var("V3D_REQ").ok().map(std::path::PathBuf::from);
    for name in names.split(',') {
        let request = match (name, &six) {
            ("tilt0" | "tilt2" | "r4" | "gmd1c" | "pgc3d" | "tet", Some(dir)) => {
                let file = match name {
                    "tilt0" => "tilt0.json",
                    "tilt2" => "tilt2.json",
                    "r4" => "R4_LIN_4.json",
                    "gmd1c" => "GMD1C_1ms.json",
                    "pgc3d" => "PGC3D_1ms.json",
                    _ => "TET_1ms.json",
                };
                read_request(&dir.join(file))
            }
            _ => read_request(std::path::Path::new(name)),
        };
        let also: Vec<Vec<usize>> = std::env::var("SELECT_ALSO").map_or(vec![], |a| {
            a.split(';')
                .filter(|g| !g.is_empty())
                .map(|g| {
                    let mut keep: Vec<usize> = (0..GROUND).collect();
                    keep.extend(g.split(',').map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
                    keep.sort_unstable();
                    keep
                })
                .collect()
        });
        ALSO.with(|a| *a.borrow_mut() = also);
        let clock = Instant::now();
        let run = crate::solve::chosen(&request, crate::solve::alone(crate::solve::chief(&request))).unwrap();
        let choice = run.report.choice.clone().expect("the chosen run holds its choice");
        eprintln!("{name}: P{} chosen in {:.1} s", run.rs.keep.len(), clock.elapsed().as_secs_f64());
        for line in &choice.report {
            eprintln!("  {line}");
        }
        if let Some(dir) = &out {
            let times = &choice.response.times;
            let parts: Vec<serde_json::Value> = times.iter().map(|&t| serde_json::json!({"t": t, "direct": choice.at(t)})).collect();
            let doc = serde_json::json!({
                "scene": name, "keep": choice.keep, "labels": choice.keep.iter().map(|&s| request.states[s].clone()).collect::<Vec<_>>(),
                "memory": choice.memory, "length": choice.length, "lambda_t": choice.response.memory(*request.times_us.last().unwrap()),
                "collapsed": choice.response.decided, "response_groups": choice.response.groups, "region": [choice.judged.region.intensity, choice.judged.region.error, choice.judged.region.largest],
                "calibration": [choice.judged.calibration.rows, choice.judged.calibration.rate],
                "response": {"times": choice.response.times, "temperature": choice.response.temperature, "errors": choice.response.errors},
                "parts": parts, "report": choice.report,
            });
            let stem = name.rsplit(['/', '\\']).next().unwrap().trim_end_matches(".json").to_string();
            std::fs::write(dir.join(format!("{stem}.json")), serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        }
    }
}
