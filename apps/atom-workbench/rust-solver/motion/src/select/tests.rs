//! The selection's checks: the chain by hand and on every reference scene, the middle point, the accounts' bounds by
//! hand; and, run by name, the whole selection on the scenes' requests (`select_on_the_scenes`) and where every atom is
//! lost (`select_where_every_atom_is_lost`, R9_PUSH).
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
    let at = |t: f64| direct_at(&judged, &windows, t).expect("every window read");
    assert_eq!(at(0.0), 0.0);
    assert!((at(5.0) - 4.01e-3).abs() < 1e-15 && (at(50.0) - 1.01e-3).abs() < 1e-15, "{} {}", at(5.0), at(50.0));
    assert!((at(500.0) - 2.01e-3).abs() < 1e-15 && (at(950.0) - 2.01e-3).abs() < 1e-15, "{} {}", at(500.0), at(950.0));
}

/// The direct part at the paths' own statistics by hand (module documentation, # The direct part, # How many paths): a
/// window of four paths reads |mean| ± q·se, q Student's t at three degrees of freedom at the run's confidence (95.4 %
/// two-sided: 3.31, where the normal's z is 2), its lower end none below zero; eight paths take seven degrees (2.43);
/// one path's spread is unknown: its upper end without bound, its lower nothing; the calibration beside both ends.
#[test]
fn select_the_direct_part_at_the_paths_statistics() {
    let spread = |paths: usize, value: f64, error: f64| Window { from: 0.0, to: 10.0, paths, rows: vec![Mean { value, error }; STATES], ..Window::default() };
    let at = |w: Window, upper: bool| {
        let judged = Judged { keep: vec![], windows: vec![w], late: vec![], calibration: Calibration { rows: 1e-5, rate: 0.0 }, region: Region::NONE, seconds: 0.0, products: 0 };
        direct_bound_at(&judged, &[(0.0, 10.0)], 5.0, upper).expect("the window read")
    };
    let q = |freedom: u64| crate::estimate::quantile(crate::estimate::tail_among(1.0), Some(freedom));
    assert!((q(3) - 3.31).abs() < 0.01 && (q(7) - 2.43).abs() < 0.01, "{} {}", q(3), q(7));
    assert!((at(spread(4, -2e-3, 1e-3), true) - (2e-3 + q(3) * 1e-3 + 1e-5)).abs() < 1e-15);
    assert!((at(spread(4, -2e-3, 1e-3), false) - 1e-5).abs() < 1e-15, "the lower end none below zero");
    assert!((at(spread(4, -5e-3, 1e-3), false) - (5e-3 - q(3) * 1e-3 + 1e-5)).abs() < 1e-15);
    assert!((at(spread(8, 5e-3, 1e-3), true) - (5e-3 + q(7) * 1e-3 + 1e-5)).abs() < 1e-15);
    assert_eq!(at(spread(1, 5e-3, f64::INFINITY), true), f64::INFINITY);
    assert_eq!(at(spread(1, 5e-3, f64::INFINITY), false), 1e-5);
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

/// The ladder's top by hand (`Settled::unmet`): parts within the tolerance book nothing more; parts beyond it -- the
/// setting stopped at `FINEST` -- book their whole sum.
#[test]
fn select_the_ladder_top_by_hand() {
    let settled = |errors: [f64; KNOBS], trace: f64| Settled { setting: FINEST, opening: 1.0, errors, trace, runs: vec![FINEST], pairs: [(0, 0); KNOBS], tried: 1, products: 0, seconds: 0.0 };
    assert_eq!(settled([0.25 * TOLERANCE, 0.5 * TOLERANCE, 0.125 * TOLERANCE, 0.0], 0.0).unmet(), None);
    assert_eq!(settled([TOLERANCE, 0.5 * TOLERANCE, 0.0, 0.0], 0.0).unmet(), Some(1.5 * TOLERANCE));
    // The trace's part decides with the knobs' (beyond the tolerance together) and is booked on its own (`Paths::trace`).
    assert_eq!(settled([0.25 * TOLERANCE, 0.25 * TOLERANCE, 0.0, 0.0], 0.75 * TOLERANCE).unmet(), Some(0.5 * TOLERANCE));
    assert_eq!(settled([0.25 * TOLERANCE, 0.25 * TOLERANCE, 0.0, 0.0], 0.25 * TOLERANCE).unmet(), None);
}

/// The refinements by hand (`refinements`), a step of 0.5 µs (a piece's floor: 2 substeps a µs a piece a step): off the
/// floor the pieces' part is read against the setting itself; where twice the pieces raise the run's rate to its floor
/// at twice the setting's, against the twice-substepped run; where they raise it to less than twice, against the
/// setting's pieces at the raised rate, one run more.
#[test]
fn select_the_refinements_by_hand() {
    let own = |s: Setting| Setting { rate: larger(s.rate, s.pieces as f64 / 0.5), ..s };
    let at = |o: f64, r: f64, p: usize| Setting { turns: o, rate: r, pieces: p, cutoff: 50.0 };
    let (runs, pairs) = refinements(at(32.0, 16.0, 2), &own);
    assert_eq!((runs, pairs), (vec![at(32.0, 16.0, 2), at(64.0, 16.0, 2), at(32.0, 32.0, 2), at(32.0, 16.0, 4)], [(0, 1), (0, 2), (0, 3), (0, 0)]));
    let (runs, pairs) = refinements(at(32.0, 8.0, 4), &own);
    assert_eq!((runs, pairs), (vec![at(32.0, 8.0, 4), at(64.0, 8.0, 4), at(32.0, 16.0, 4), at(32.0, 16.0, 8)], [(0, 1), (0, 2), (2, 3), (0, 0)]));
    let (runs, pairs) = refinements(at(1.0, 10.0, 4), &own);
    assert_eq!((runs, pairs), (vec![at(1.0, 10.0, 4), at(2.0, 10.0, 4), at(1.0, 20.0, 4), at(1.0, 16.0, 8), at(1.0, 16.0, 4)], [(0, 1), (0, 2), (4, 3), (0, 0)]));
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

/// The whole selection where every atom is lost (refs/scenes R9_PUSH: the pushing beam has pushed every atom out by
/// 45 µs, the paths' four walks by 28 µs): it ends with a choice, complete -- its W a number --, and no line of its
/// report says a number that is none. (On dca75e10 every candidate's price and the 32 states' own account read no
/// number there, and the choice panicked: "the anchor is complete".) A few seconds; a gate's.
#[test]
#[ignore]
fn select_where_every_atom_is_lost() {
    let request = crate::model::testing::scene("R9_PUSH");
    let clock = Instant::now();
    let run = crate::solve::chosen(&request, crate::solve::alone(crate::solve::chief(&request))).expect("R9_PUSH's choice");
    let choice = run.report.choice.clone().expect("the chosen run holds its choice");
    for line in &choice.report {
        eprintln!("  {line}");
    }
    eprintln!("R9_PUSH: P{} chosen in {:.1} s", run.rs.keep.len(), clock.elapsed().as_secs_f64());
    let chosen = choice.report.iter().find(|l| l.contains("（选中）")).expect("the chosen candidate's line");
    assert!(chosen.contains("W ") && !chosen.contains("W —") && !chosen.contains("inf"), "{chosen}");
    assert!(choice.report.iter().all(|l| !l.contains("NaN")), "{:#?}", choice.report);
    assert_eq!(choice.keep, run.rs.keep);
}

/// The whole selection where G's fold runs away at every start (refs/scenes SAT_3: σ⁺ on resonance with the cycling
/// line, s = 3, D2-3-3's first-order excitation 1.5 -- its fold's series does not converge, and the folded ground state
/// scatters beyond every state of the atom, step.rs # The domain): G's pilot takes no step, the attempt is made again on
/// the structural chain's next model (solve.rs `piloted`), its paths go past their starts, and the choice holds D2-3-3
/// explicit -- P9 or above --, complete (its W a number), no line of its report saying a number that is none. (On
/// d6b78fad: "no path of the pilot's goes past its start", 0 steps and 279 refused.) Seconds; a gate's.
#[test]
#[ignore]
fn select_where_g_runs_away_at_every_start() {
    let request = crate::model::testing::scene("SAT_3");
    let clock = Instant::now();
    let run = crate::solve::chosen(&request, crate::solve::alone(crate::solve::chief(&request))).expect("SAT_3's choice");
    let choice = run.report.choice.clone().expect("the chosen run holds its choice");
    for line in &choice.report {
        eprintln!("  {line}");
    }
    eprintln!("SAT_3: P{} chosen in {:.1} s", run.rs.keep.len(), clock.elapsed().as_secs_f64());
    let excited = request.states.iter().position(|s| s == "D2-3-3").expect("D2-3-3 among the states");
    assert!(choice.keep.contains(&excited) && choice.keep.len() > GROUND, "P{} {:?}", choice.keep.len(), choice.keep);
    // The pilot on a model above G, and its paths read.
    let pilot = choice.report.iter().find(|l| l.starts_with("先导立定于")).expect("the pilot's line");
    let model: usize = pilot.split("先导的模型 P").nth(1).and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next()).and_then(|n| n.parse().ok()).expect("the pilot's model");
    assert!(model > GROUND, "{pilot}");
    assert!(choice.report.iter().any(|l| l.starts_with("路径：") && !l.starts_with("路径：0 ")), "{:#?}", choice.report);
    let chosen = choice.report.iter().find(|l| l.contains("（选中）")).expect("the chosen candidate's line");
    assert!(chosen.contains("W ") && !chosen.contains("W —") && !chosen.contains("inf"), "{chosen}");
    assert!(choice.report.iter().all(|l| !l.contains("NaN")), "{:#?}", choice.report);
    assert_eq!(choice.keep, run.rs.keep);
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

/// A scene's request by its name: one of the six's in `six` (V3D_REQ), or a native request's path.
fn scene_request(name: &str, six: Option<&std::path::Path>) -> crate::request::Request {
    match (name, six) {
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
    }
}

/// The whole selection (module documentation; solve.rs # The selection) on the scenes named by `SELECT_SCENES` (comma-
/// separated: the six's names in `requests`, or a native request's path, e.g. R5_PUMP's and S1's), one after another:
/// G's pilot, the paths, the candidates judged and weighed, P -- its report (every candidate's parts, work
/// and fate; the stages' times), and with `SELECT_OUT` a directory, each scene's choice to <scene>.json there: P, the
/// direct part at the windows' ends, the region, the calibration. `SELECT_ALSO`: candidates beside the chain judged for
/// comparison (`ALSO`), ';'-separated, each the excited states' labels it keeps, ','-separated (the ground states always).
/// `SELECT_SOLVE` set: the whole solve instead (the product's own lines, its memory flag among them).
#[test]
#[ignore]
fn select_on_the_scenes() {
    let names = std::env::var("SELECT_SCENES").expect("SELECT_SCENES: the scenes, comma-separated");
    let out = std::env::var("SELECT_OUT").ok().map(std::path::PathBuf::from);
    let six = std::env::var("V3D_REQ").ok().map(std::path::PathBuf::from);
    for name in names.split(',') {
        let request = scene_request(name, six.as_deref());
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
        // `SELECT_SOLVE`: the whole solve instead, its product and its flags (the run's own lines on stderr).
        if std::env::var("SELECT_SOLVE").is_ok() {
            let solved = crate::solve::solve(&request, &mut |_| {});
            eprintln!("{name}: solved in {:.1} s: {}", clock.elapsed().as_secs_f64(), if solved.is_ok() { "ok" } else { "an error" });
            continue;
        }
        let run = crate::solve::chosen(&request, crate::solve::alone(crate::solve::chief(&request))).unwrap();
        let choice = run.report.choice.clone().expect("the chosen run holds its choice");
        eprintln!("{name}: P{} chosen in {:.1} s", run.rs.keep.len(), clock.elapsed().as_secs_f64());
        for line in &choice.report {
            eprintln!("  {line}");
        }
        if let Some(dir) = &out {
            let parts: Vec<serde_json::Value> = choice.windows.iter().map(|&(_, t)| serde_json::json!({"t": t, "direct": choice.at(t)})).collect();
            let doc = serde_json::json!({
                "scene": name, "keep": choice.keep, "labels": choice.keep.iter().map(|&s| request.states[s].clone()).collect::<Vec<_>>(),
                "memory": choice.memory, "length": choice.length, "region": [choice.judged.region.intensity, choice.judged.region.error, choice.judged.region.largest],
                "calibration": [choice.judged.calibration.rows, choice.judged.calibration.rate],
                "parts": parts, "report": choice.report,
            });
            let stem = name.rsplit(['/', '\\']).next().unwrap().trim_end_matches(".json").to_string();
            std::fs::write(dir.join(format!("{stem}.json")), serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        }
    }
}

/// The paths' work, measured (module documentation, # The paths): on the scene `SELECT_SCENES` (one), G's pilot's
/// walks -- read from `SELECT_WALKS` where that file is, else the solve run up to the selection's stage (stopped there)
/// and its walks written to it --; the reference's `Paths` to a run of `SELECT_UPTO` µs at the most, each job's work and
/// the pool's threads; and Lawson at the substeps a µs `SELECT_RATES` (comma-separated, increasing): on the
/// calibration's piece the reference's rows against the converged scheme's and the calibration (`calibrate`) of G,
/// along every walk's first `SELECT_SPAN` µs the reference's rows against the last rate's, and the seconds.
#[test]
#[ignore]
fn select_paths_cost() {
    use crate::paths::{Form, Scheme, Series, Walk};
    let name = std::env::var("SELECT_SCENES").expect("SELECT_SCENES: one scene");
    let six = std::env::var("V3D_REQ").ok().map(std::path::PathBuf::from);
    let request = scene_request(&name, six.as_deref());
    let file = std::path::PathBuf::from(std::env::var("SELECT_WALKS").expect("SELECT_WALKS: the walks' file"));
    let length = request.times_us.last().copied().unwrap();
    let reference = crate::solve::alone(crate::solve::chief(&request));
    if std::env::var("SELECT_SHAPE").is_ok() {
        let g = crate::solve::run_of(&request, reference).unwrap();
        let centre = crate::solve::reported(&g, &reference).unwrap().centre;
        let full = Selected::new(crate::scene::Scene::new(&request).unwrap(), Selection { keep: (0..STATES).collect(), ..g.report.clone() }, reference).unwrap();
        eprintln!("{name}: reference {}", Form::new(&full, centre).shape());
        eprintln!("{name}: G {}", Form::new(&g, centre).shape());
        return;
    }
    let walks: Vec<Walk> = if file.exists() {
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        doc["walks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| Walk {
                index: w["index"].as_u64().unwrap(),
                start: w["start"].as_array().unwrap().iter().map(|x| (x[0].as_u64().unwrap() as usize, C64::new(x[1].as_f64().unwrap(), x[2].as_f64().unwrap()))).collect(),
                ends: serde_json::from_value(w["ends"].clone()).unwrap(),
            })
            .collect()
    } else {
        let clock = Instant::now();
        crate::paths::WALKS.with(|w| *w.borrow_mut() = Some((WALKS, length, vec![])));
        let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::solve::solve(&request, &mut |m| {
                if let crate::solve::Message::Stage(s) = m {
                    assert!(s.step != crate::solve::Step::Select, "stopped at the selection");
                }
            })
        }));
        assert!(stopped.is_err(), "the solve stopped at the selection");
        let walks = crate::paths::WALKS.with(|w| w.borrow_mut().take().unwrap().2);
        let doc = serde_json::json!({"walks": walks.iter().map(|w| serde_json::json!({
            "index": w.index, "start": w.start.iter().map(|(s, z)| serde_json::json!([s, z.re, z.im])).collect::<Vec<_>>(), "ends": w.ends,
        })).collect::<Vec<_>>()});
        std::fs::write(&file, serde_json::to_string(&doc).unwrap()).unwrap();
        eprintln!("{name}: the pilot's walks in {:.1} s, written", clock.elapsed().as_secs_f64());
        walks
    };
    let g = crate::solve::run_of(&request, reference).unwrap();
    let centre = crate::solve::reported(&g, &reference).unwrap().centre;
    let form = Form::new(&g, centre);
    let memory = walks.iter().filter_map(|w| crate::paths::memory(&form, [w.ends[0][1], w.ends[0][2], w.ends[0][3]], [w.ends[0][4], w.ends[0][5], w.ends[0][6]], w.ends[0][0])).map(|m| m.time).fold(0.0, larger);
    let tau = walks[0].ends[1][0] - walks[0].ends[0][0];
    let upto: f64 = std::env::var("SELECT_UPTO").map_or(length, |x| x.parse().unwrap());
    let run = length.min(upto);
    eprintln!("{name}: {} walks to {:?} µs, τ {tau:.4} µs, τ_mem {memory:.3} µs, run {length} µs (measured to {run}); pool {} threads", walks.len(), walks.iter().map(|w| w.end()).collect::<Vec<_>>(), rayon::current_num_threads());
    let full = Selected::new(crate::scene::Scene::new(&request).unwrap(), Selection { keep: (0..STATES).collect(), ..g.report.clone() }, reference).unwrap();
    let light = g.light();
    if let Ok(probe) = std::env::var("SELECT_PROBE") {
        // Lawson on walk `SELECT_WALK` (0)'s first `probe` µs at the settings `SELECT_GRID` ("rate/pieces", comma-
        // separated, the finest last), the reference and G: seconds and counts; the reference's rows and G's fold
        // error against the finest's; `SELECT_SPARSE` set, the converged scheme too.
        let probe: f64 = probe.parse().unwrap();
        let k: usize = std::env::var("SELECT_WALK").map_or(0, |x| x.parse().unwrap());
        // `SELECT_STILL`: the walk held at its first place at rest (the same knots).
        let still: Walk = {
            let w = &walks[k];
            let e0 = w.ends[0];
            Walk { index: w.index, start: w.start.clone(), ends: w.ends.iter().map(|e| [e[0], e0[1], e0[2], e0[3], 0.0, 0.0, 0.0]).collect() }
        };
        let w = if std::env::var("SELECT_STILL").is_ok() { &still } else { &walks[k] };
        let to = probe.min(w.end());
        // A setting "the opening's substeps a turn/the run's a µs/pieces"; a rate "sN" is N substeps a piece (N × pieces /
        // τ a µs).
        let grid: Vec<Setting> = std::env::var("SELECT_GRID").map_or(vec![COARSEST], |x| {
            x.split(',')
                .map(|g| {
                    let p: Vec<&str> = g.split('/').collect();
                    let pieces: usize = p[2].parse().unwrap();
                    let rate = match p[1].strip_prefix('s') { Some(n) => n.parse::<f64>().unwrap() * pieces as f64 / tau, None => p[1].parse().unwrap() };
                    Setting { turns: p[0].parse().unwrap(), rate, pieces, cutoff: p.get(3).map_or(f64::INFINITY, |x| x.parse().unwrap()) }
                })
                .collect()
        });
        let (fr, fg) = (Form::new(&full, centre), Form::new(&g, centre));
        eprintln!("walk {k} to {to} µs: reference {}; G {}", fr.shape(), fg.shape());
        let mut schemes: Vec<Scheme> = grid.iter().map(|&s| scheme(s, 1)).collect();
        // `SELECT_SPARSE` = 2: the converged scheme at twice its nodes and half its turn first (its own check), compared
        // against it.
        if std::env::var("SELECT_SPARSE").is_ok_and(|x| x == "2") {
            schemes.push(Scheme::Sparse { nodes: 8, turn: std::f64::consts::PI, read: 1 });
        }
        if std::env::var("SELECT_SPARSE").is_ok() {
            schemes.push(Scheme::sparse(4));
        }
        // `SELECT_FROM`: every scheme from the converged scheme's state at the first knot at or past it (the start's
        // transient left behind).
        let from: f64 = std::env::var("SELECT_FROM").map_or(0.0, |x| { let x: f64 = x.parse().unwrap(); w.ends.iter().map(|e| e[0]).find(|&t| t >= x).unwrap() });
        let states = (from > 0.0).then(|| (fr.along(w, &w.start, 0.0, from, Scheme::sparse(4)).unwrap().state, fg.along(w, &w.start, 0.0, from, Scheme::sparse(4)).unwrap().state));
        let runs: Vec<(Series, Series)> = schemes
            .iter()
            .map(|&scheme| {
                match &states {
                    Some((x, y)) => (fr.along_from(w, x.clone(), from, to, scheme).unwrap(), fg.along_from(w, y.clone(), from, to, scheme).unwrap()),
                    None => (fr.along(w, &w.start, 0.0, to, scheme).unwrap(), fg.along(w, &w.start, 0.0, to, scheme).unwrap()),
                }
            })
            .collect();
        let finest = grid.len() - 1;
        let at = |s: &Series, t: f64| s.readings.iter().find(|r| r.t == t).map(|r| r.rows);
        let sparse = (runs.len() > grid.len()).then(|| runs.last().unwrap());
        let against = |a: &Series, b: &Series, fa: &Series, fb: &Series| -> (f64, f64) {
            let (mut rows, mut fold) = (0.0_f64, 0.0_f64);
            for x in &a.readings {
                let (Some(y), Some(u), Some(v)) = (at(fa, x.t), at(b, x.t), at(fb, x.t)) else { continue };
                for q in 0..STATES {
                    rows = rows.max((x.rows[q] - y[q]).abs());
                    fold = fold.max(((u[q] - x.rows[q]) - (v[q] - y[q])).abs());
                }
            }
            (rows, fold)
        };
        for (scheme, (a, b)) in schemes.iter().zip(&runs) {
            if let Some((sa, sb)) = sparse {
                let (rows, fold) = against(a, b, sa, sb);
                // Where the reference's rows differ most, and at the first readings.
                let mut worst = (0.0_f64, 0.0_f64, 0usize);
                for x in &a.readings {
                    let Some(y) = at(sa, x.t) else { continue };
                    for q in 0..STATES {
                        if (x.rows[q] - y[q]).abs() > worst.0 {
                            worst = ((x.rows[q] - y[q]).abs(), x.t, q);
                        }
                    }
                }
                let first: Vec<String> = a.readings.iter().take(std::env::var("SELECT_FIRST").map_or(6, |x| x.parse().unwrap())).filter_map(|x| at(sa, x.t).map(|y| format!("{:.3}:{:.1e}", x.t, (0..STATES).map(|q| (x.rows[q] - y[q]).abs()).fold(0.0, f64::max)))).collect();
                eprintln!("  {scheme:?} vs the converged scheme: reference's rows {rows:.2e} (at {:.3} µs, row {}), G's fold error {fold:.2e}; condition {:.1e}; first readings {first:?}", worst.1, worst.2, a.counts.condition);
            }
            let (fa, fb) = &runs[finest];
            let (mut rows, mut fold) = (0.0_f64, 0.0_f64);
            for x in &a.readings {
                let (Some(y), Some(u), Some(v)) = (at(fa, x.t), at(b, x.t), at(fb, x.t)) else { continue };
                for q in 0..STATES {
                    rows = rows.max((x.rows[q] - y[q]).abs());
                    fold = fold.max(((u[q] - x.rows[q]) - (v[q] - y[q])).abs());
                }
            }
            let (c, d) = (&a.counts, &b.counts);
            eprintln!(
                "  {scheme:?}: reference {:.3} s (nodes {:.3} + propagation {:.3} + readouts {:.3}; {} nodes, {} segments, {} products), G {:.3} s; vs the finest: reference's rows {rows:.2e}, G's fold error {fold:.2e}",
                c.seconds(), c.node_seconds, c.propagation_seconds, c.readout_seconds, c.nodes, c.segments, c.products, d.seconds()
            );
        }
        return;
    }
    let paths = Paths::new(&full, centre, walks.clone(), memory, run, tau, light).unwrap();
    let say = |what: &str, s: &Series| {
        let c = &s.counts;
        eprintln!(
            "  {what}: {:.1} µs, {:.2} s (nodes {:.2} + propagation {:.2} + readouts {:.2}); {} nodes, {} segments, {} products, {} readouts",
            c.time, c.seconds(), c.node_seconds, c.propagation_seconds, c.readout_seconds, c.nodes, c.segments, c.products, c.readouts
        );
    };
    let jobs: f64 = paths.series.iter().chain(paths.late.iter().flat_map(|l| l.2.iter())).map(|s| s.counts.seconds()).sum();
    eprintln!("{name}: Paths L {:.1} µs, {} windows, late {:?}: {:.1} s wall, {jobs:.1} s over the walks' jobs; settled {:?}", paths.length, paths.windows.len(), paths.late_from(), paths.seconds, paths.settled);
    for (k, s) in paths.series.iter().enumerate() {
        say(&format!("walk {k} start"), s);
    }
    for (k, s) in paths.late.iter().flat_map(|l| l.2.iter()).enumerate() {
        say(&format!("walk {k} late"), s);
    }
    let clock = Instant::now();
    let judged = judge(&paths, &g, centre).unwrap();
    eprintln!("{name}: G judged in {:.1} s ({:.1} s wall): calibration {:?}", judged.seconds, clock.elapsed().as_secs_f64(), judged.calibration);
}
