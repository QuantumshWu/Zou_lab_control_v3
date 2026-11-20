//! The selection's checks: the gate's measures and the two stages by hand; on the six scenes' requests the base
//! against the prototype's whole-level selection (`choose_is_pysel`) and P at the three points (`choose_s_p_passes_at_the_three_points`);
//! the cost.
use super::*;

fn readouts(levels: &[f64], rate: f64, force: [f64; 3]) -> Readouts {
    Readouts { levels: levels.to_vec(), rate, force, ..Default::default() }
}

/// The gate's three measures, by hand: populations 0.004 absolute, R 0.02 relative, F⃗ |(0.3, 0.4, 0)| = 0.5 over
/// max(|F⃗| = 10, kR = 2·4 = 8) = 0.05; at a balanced point F⃗ = 0 the scale is kR = 8: |(0, 0, 0.08)| / 8 = 0.01. A
/// measure that is not a number is the worst, and passes no gate.
#[test]
fn select_errors_by_hand() {
    let reference = readouts(&[0.5, 0.5], 4.0, [6.0, 8.0, 0.0]);
    let model = readouts(&[0.504, 0.496], 4.08, [6.3, 8.4, 0.0]);
    let e = errors(&model, &reference, 2.0, Route::Krylov);
    assert!((e.populations - 0.004).abs() < 1e-15 && (e.rate - 0.02).abs() < 1e-15 && (e.force - 0.05).abs() < 1e-15);
    let balanced = readouts(&[0.5, 0.5], 4.0, [0.0; 3]);
    let e = errors(&readouts(&[0.5, 0.5], 4.0, [0.0, 0.0, 0.08]), &balanced, 2.0, Route::Krylov);
    assert!((e.force - 0.01).abs() < 1e-15);
    let lost = errors(&readouts(&[0.5, 0.5], f64::NAN, [0.0; 3]), &balanced, 2.0, Route::Krylov);
    assert!(lost.worst().is_nan());
    let candidate = Candidate { keep: vec![0], centre: e.clone(), edge: lost, middle: e };
    assert!(candidate.worst().is_nan() && !candidate.within(1.0) && !candidate.passes());
}

/// The base takes the smallest candidate passing at the three points, though a larger one passes too and a smaller one
/// passes at two points only; the report keeps the candidates up to it, all whole-level ones, P the base. None passing
/// is an error.
#[test]
fn select_decide_takes_the_smallest_passing() {
    let candidates = vec![vec![0], vec![0, 1], vec![0, 1, 2]];
    // R's relative error per (point, candidate): candidate 0 fails at the edge, 1 and 2 pass at every point
    let error = |p: usize, c: usize| [[0.001, 0.005, 0.0], [0.03, 0.009, 0.0], [0.002, 0.004, 0.0]][p][c];
    let out = decide(&candidates, 1.0, |p, c| {
        Ok((readouts(&[1.0], 1.0 + c.map_or(0.0, |c| error(p, c)), [1.0, 0.0, 0.0]), Route::Krylov))
    })
    .unwrap();
    assert_eq!(out.keep, vec![0, 1]);
    assert_eq!((out.candidates.len(), out.whole), (2, 2));
    assert!(!out.candidates[0].passes() && out.candidates[1].passes());
    assert_eq!(out.base().keep, out.keep);
    let none = decide(&candidates, 1.0, |_, c| {
        Ok((readouts(&[1.0], 1.0 + c.map_or(0.0, |_| 0.5), [1.0, 0.0, 0.0]), Route::Krylov))
    });
    assert!(none.is_err());
}

/// A point without light is not judged (module documentation, # Points without light): with the edge dark its solve
/// is never asked (the closure would panic), its errors read zero with no route, its reference none, the report says
/// so; the base the first candidate passing at the centre and the middle (candidate 0 fails there, R 3 %). Every point
/// dark: the first candidate, the ground states, with nothing solved.
#[test]
fn select_a_point_without_light_is_not_judged() {
    let candidates = vec![vec![0], vec![0, 1]];
    let error = |c: usize| [0.03, 0.005][c];
    let out = decide_on(&candidates, [true, false, true], 1.0, |p, c| {
        assert_ne!(p, 1, "the dark edge is solved");
        Ok((readouts(&[1.0], 1.0 + c.map_or(0.0, error), [1.0, 0.0, 0.0]), Route::Krylov))
    })
    .unwrap();
    assert_eq!(out.keep, vec![0, 1]);
    assert!(out.candidates.iter().all(|c| c.edge == PointError::unlit() && c.centre.route == Some(Route::Krylov) && c.middle.route == Some(Route::Krylov)));
    assert!(out.reference[0].is_some() && out.reference[1].is_none() && out.reference[2].is_some());
    assert_eq!(out.notes(), vec!["边缘点无光、无唯一稳态，不判".to_string()]);
    let dark = decide_on(&candidates, [false; POINTS], 1.0, |_, _| -> Result<(Readouts, Route), String> { panic!("a dark point is solved") }).unwrap();
    assert_eq!((dark.keep.clone(), dark.whole), (vec![0], 1));
    assert_eq!(dark.notes().len(), POINTS);
    let refined = refine(dark, 1.0, &|_: usize, _: &[usize]| -> Result<(Readouts, Route), String> { panic!("a dark point is solved") }).unwrap();
    assert_eq!(refined.keep, vec![0]);
}

/// The middle point (module documentation, # The middle point's direction and distance), on two reference requests
/// (refs/scenes): R10_NODE, the trap at a node of the standing wave whose nodal plane holds the tweezer's axis -- the
/// middle leaves the plane (n̂ across the axis, |n̂·d| at rounding), the tweezer's intensity there half its peak to
/// rounding, and the light other than the tweezer's reaches it (`reach` ≥ ε) where one Rayleigh range along the axis
/// it does not; R1_TWEEZER, the tweezer alone (M = 0) --
/// the middle on the axis at the Rayleigh range, z_R d, to rounding.
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
    let other = |point: &Components| reach(&node, &rs::reference(&node, point), point, &|l| l != tw).iter().fold(0.0, |m: f64, &a| m.max(a));
    let at_middle = other(&node.components(r, &zero, 1.0));
    let on_axis = other(&node.components(d.map(|x| x * node.geometry[tw].rayleigh().unwrap()), &zero, 1.0));
    assert!(at_middle >= f64::EPSILON && on_axis < f64::EPSILON, "R10_NODE: the other light's reach {at_middle:e} at the middle, {on_axis:e} on the axis");
    eprintln!("R10_NODE: the middle {r:?}, the other light's largest first-order amplitude {at_middle:.3e} there, {on_axis:.3e} on the axis");
    let alone = Scene::new(&crate::model::testing::scene("R1_TWEEZER")).unwrap();
    let tweezer = alone.geometry[alone.tweezer()];
    let r = middle(&alone);
    let z = tweezer.rayleigh().unwrap();
    assert!((0..3).all(|i| (r[i] - z * tweezer.direction[i]).abs() <= 8.0 * f64::EPSILON * z), "R1_TWEEZER: {r:?} for z_R {z} along {:?}", tweezer.direction);
}

/// The leave-one-out by hand (module documentation, # The decision, 2): the ground states and four excited states
/// GROUND..GROUND + 4 the base; a set's R error (every point alike) 0.001 plus each missing state's own (0.001, 0.0015,
/// 0.0025, 0.02) plus 0.008 where the first two are both missing (the errors not additive: a pair that cancels).
/// - e_i = 0.002, 0.0025, 0.0035, 0.021: the base less i, not the increment;
/// - at 1 %: 0.001 + 0.002 + 0.0025 + 0.0035 = 0.009 ≤ 1 %, the fourth over; the three removed fail (0.014), the
///   largest e_i (the third) put back, the two fail (0.0115), the second put back, the first alone passes (0.002): P
///   the base less the first, three checks after the two whole levels, the last P's;
/// - at 5 %: all four removed (0.03), the set passes (0.035): P the ground states;
/// - at 0.25 %: the first already over (0.003): nothing removed, no check, P the base; below the base's 0.1 % an error.
#[test]
fn select_leave_one_out_by_hand() {
    let own = [0.001, 0.0015, 0.0025, 0.02];
    let base: Vec<usize> = (0..GROUND + 4).collect();
    let error = |keep: &[usize]| -> f64 {
        let missing = |s: usize| !keep.contains(&s);
        let pair = if missing(GROUND) && missing(GROUND + 1) { 0.008 } else { 0.0 };
        0.001 + (0..4).filter(|&i| missing(GROUND + i)).map(|i| own[i]).sum::<f64>() + pair
    };
    let solve = |_: usize, keep: &[usize]| -> Result<(Readouts, Route), String> {
        Ok((readouts(&[1.0], 1.0 + error(keep), [1.0, 0.0, 0.0]), Route::Krylov))
    };
    let ground: Vec<usize> = (0..GROUND).collect();
    let candidates = vec![ground.clone(), base.clone()];
    let first = decide(&candidates, 1.0, |p, c| match c {
        None => Ok((readouts(&[1.0], 1.0, [1.0, 0.0, 0.0]), Route::Krylov)),
        Some(c) => solve(p, &candidates[c]),
    })
    .unwrap();
    assert_eq!(first.keep, base);
    let out = refine(first, 1.0, &solve).unwrap();
    let e: Vec<f64> = out.left_out.iter().map(|l| l.e()).collect();
    assert_eq!(out.left_out.iter().map(|l| l.state).collect::<Vec<_>>(), (GROUND..GROUND + 4).collect::<Vec<_>>());
    assert!(e.iter().zip([0.002, 0.0025, 0.0035, 0.021]).all(|(a, b)| (a - b).abs() < 1e-15), "{e:?}");
    let less = |gone: &[usize]| -> Vec<usize> { base.iter().copied().filter(|s| !gone.contains(s)).collect() };
    assert_eq!(out.keep, less(&[GROUND]));
    assert_eq!(out.removed, vec![GROUND]);
    assert!((out.predicted - 0.009).abs() < 1e-15);
    let checked: Vec<Vec<usize>> = out.candidates[out.whole..].iter().map(|c| c.keep.clone()).collect();
    assert_eq!(checked, vec![less(&[GROUND, GROUND + 1, GROUND + 2]), less(&[GROUND, GROUND + 1]), less(&[GROUND])]);
    let passing: Vec<bool> = out.candidates[out.whole..].iter().map(|c| c.passes()).collect();
    assert_eq!(passing, vec![false, false, true]);
    assert_eq!(out.candidates.last().unwrap().keep, out.keep);
    let reference = out.reference.clone();
    let wide = descend(out.base(), &out.left_out, 0.05, &reference, 1.0, &solve).unwrap();
    assert_eq!((wide.keep, wide.checks.len()), (ground.clone(), 1));
    assert!((wide.predicted - 0.03).abs() < 1e-15);
    let narrow = descend(out.base(), &out.left_out, 0.0025, &reference, 1.0, &solve).unwrap();
    assert_eq!((narrow.keep, narrow.removed.len(), narrow.checks.len()), (base.clone(), 0, 0));
    assert!(descend(out.base(), &out.left_out, 0.0005, &reference, 1.0, &solve).is_err());
}

/// A level's name in the prototype's readouts (td_s: "g1", "g2", "e<line><F′>"), of its first state's label
/// ("ground-F-m", "D<line>-F′-m").
fn name(label: &str) -> String {
    let mut parts = label.splitn(3, '-');
    let (head, f) = (parts.next().unwrap(), parts.next().unwrap());
    if head == "ground" { format!("g{f}") } else { format!("e{}{f}", &head[1..]) }
}

/// The six scenes (`requests`).
fn six() -> Vec<(&'static str, Scene)> {
    requests().into_iter().map(|(name, request)| (name, Scene::new(&request).unwrap())).collect()
}

/// The six scenes' requests (the directory `V3D_REQ`: scratchpad reqprec/req; `PSTATE_SCENE` one of them), each with
/// its name in the prototype's files.
fn requests() -> Vec<(&'static str, crate::request::Request)> {
    let dir = std::path::PathBuf::from(std::env::var("V3D_REQ").expect("V3D_REQ: the six scenes' requests (scratchpad reqprec/req)"));
    [("tilt0.json", "tilt0"), ("tilt2.json", "tilt2"), ("R4_LIN_4.json", "r4"), ("GMD1C_1ms.json", "gmd1c"), ("PGC3D_1ms.json", "pgc3d"), ("TET_1ms.json", "tet")]
        .into_iter()
        .filter(|(_, name)| std::env::var("PSTATE_SCENE").map_or(true, |only| only == *name))
        .map(|(file, name)| {
            let p = dir.join(file);
            let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap();
            line["options"]["device"] = serde_json::json!("cpu");
            // These requests predate the outside forms' transitions, which the scene does not read.
            for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
                form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
            }
            (name, crate::request::Request::from_native(&line).unwrap())
        })
        .collect()
}

/// On the centre and the edge (`two_point_choice`: the prototype has those two points, not the middle), the base
/// (# The decision, 1) is the prototype's whole-level selection (PYEXP2, scratchpad pyexp2/sel_<scene>.json: PYSEL's
/// procedure on each scene's own request, reqprec/req, its exports pysel/export2) on the six scenes' requests
/// (`V3D_REQ`, the selections' `V3D_SEL`): the levels' ranking and the base the same; the full reference's readouts at
/// the two points -- F⃗, R, the levels' populations, each as the gate reads it -- the prototype's to the solves' accuracy;
/// and per whole-level candidate the gate's three errors, which differ by what the two models differ by (the prototype
/// folds the model's fast elements in Liouville space, fold2; this one in Hilbert space by RS, model/rs.rs: the same
/// second order, apart in the next, v3c2.status.md §2.3), said.
///
/// Measured: the references' readouts within 3.4·10⁻¹³ on five scenes, the same element sets (GMD1C 4634 and 4428 as
/// the prototype's); GMD1C's F⃗ 2.0·10⁻⁷ of its scale max(|F⃗|, kR) ≈ 6·10⁻⁴ (its force near balance), all of it the
/// prototype's F_y: −1.2·10⁻¹⁰ at the centre where the scene's symmetry has none (here 5.7·10⁻¹⁵) -- the prototype's
/// solve there agrees with its own direct LU to 10⁻¹¹ only (pyexp2.status.md, its condition). The bound read is 10⁻⁶
/// of the gate's scale: four orders below the gate, the decisions the same where no error sits within it of 1 %.
#[test]
fn choose_is_pysel() {
    let sel = std::path::PathBuf::from(std::env::var("V3D_SEL").expect("V3D_SEL: the prototype's selections (scratchpad pyexp2)"));
    let json = |p: std::path::PathBuf| -> serde_json::Value { serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap() };
    let mut worst = [0.0_f64; 2];
    for (scene_name, scene) in six() {
        let python = json(sel.join(format!("sel_{scene_name}.json")));
        let labels = &scene.request.states;
        // The ranking on the two points.
        let points = points(&scene);
        let references: Vec<ModelAt> = points.iter().map(|p| rs::reference(&scene, p)).collect();
        let ours: Vec<String> = ranking(&scene, &references[..2], &points[..2]).iter().map(|l| name(&labels[l[0]])).collect();
        let theirs: Vec<String> = python["ranking"].as_array().unwrap().iter().map(|r| format!("e{}{}", &r[0].as_str().unwrap()[1..], r[1])).collect();
        assert_eq!(ours, theirs, "{scene_name}: the ranking");
        // The base on the two points.
        let selection = two_point_choice(&scene).unwrap();
        let p = python["P"].as_u64().unwrap() as usize;
        eprintln!("{scene_name}: the base {} (prototype {p}), {} whole-level candidates, P {}, {:.2} s", selection.base().keep.len(), selection.whole, selection.keep.len(), selection.seconds);
        assert_eq!(selection.base().keep.len(), p, "{scene_name}: the base");
        // The full reference's readouts.
        let groups = levels(labels, &(0..STATES).collect::<Vec<_>>());
        for (i, point) in ["centre", "edge"].into_iter().enumerate() {
            let (ours, route) = selection.reference[i].as_ref().expect("the six scenes' centre and edge are lit");
            let theirs = &python["full"][point];
            let f: Vec<f64> = theirs["F"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            let r = theirs["R"].as_f64().unwrap();
            let scale = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt().max(selection_k(&scene) * r);
            let df = (0..3).map(|a| (ours.force[a] - f[a]).powi(2)).sum::<f64>().sqrt() / scale;
            let dr = (ours.rate - r).abs() / r;
            let dp = groups.iter().zip(&ours.levels).map(|(g, x)| (x - theirs["pops"][name(&labels[g[0]])].as_f64().unwrap()).abs()).fold(0.0, f64::max);
            eprintln!("  {point} reference ({route:?}): F⃗ {df:.1e} R {dr:.1e} populations {dp:.1e}");
            worst[0] = worst[0].max(df.max(dr).max(dp));
            assert!(df.max(dr).max(dp) <= 1e-6, "{scene_name} {point}: the reference's readouts");
        }
        // Each whole-level candidate's gate errors.
        for (c, step) in selection.candidates[..selection.whole].iter().zip(python["steps"].as_array().unwrap()) {
            assert_eq!(c.keep.len() as u64, step["P"].as_u64().unwrap(), "{scene_name}: the candidates' sizes");
            for (point, ours) in [("centre", &c.centre), ("edge", &c.edge)] {
                let e = &step["err"][point];
                let (f, r, pop) = (e["F"].as_f64().unwrap(), e["R"].as_f64().unwrap(), e["P"].as_f64().unwrap());
                eprintln!(
                    "  P {:2} {point}: F⃗ {:.4e} ({:.4e}) R {:.4e} ({:.4e}) populations {:.4e} ({:.4e}) -- ours (prototype's); {:?}",
                    c.keep.len(), ours.force, f, ours.rate, r, ours.populations, pop, ours.route
                );
                worst[1] = worst[1].max(((ours.force - f).abs()).max((ours.rate - r).abs()).max((ours.populations - pop).abs()));
            }
        }
    }
    eprintln!("largest: the reference's readouts {:.1e}; the gate's errors apart by {:.1e}", worst[0], worst[1]);
}

/// A candidate's errors for the report: per point populations, R, F⃗.
fn said(c: &Candidate) -> String {
    let e = |p: &PointError| format!("{:.2e} {:.2e} {:.2e}", p.populations, p.rate, p.force);
    format!("centre {} | edge {} | middle {}", e(&c.centre), e(&c.edge), e(&c.middle))
}

fn as_json(c: &Candidate) -> serde_json::Value {
    let e = |p: &PointError| serde_json::json!({"populations": p.populations, "rate": p.rate, "force": p.force, "route": format!("{:?}", p.route)});
    serde_json::json!({"keep": c.keep, "worst": c.worst(), "centre": e(&c.centre), "edge": e(&c.edge), "middle": e(&c.middle)})
}

/// The selection on the centre and the edge alone -- the procedure before the middle point (the user's 2026-11-20) --:
/// the ranking over those two, the middle not judged. For the prototype's comparison and for the three points'.
fn two_point_choice(scene: &Scene) -> Result<Selection, String> {
    let three = ThreePoints::new(scene);
    let mut keep: Vec<usize> = (0..GROUND).collect();
    let mut candidates = vec![keep.clone()];
    for level in ranking(scene, &three.references[..2], &three.points[..2]) {
        keep.extend(level);
        keep.sort_unstable();
        candidates.push(keep.clone());
    }
    let base = decide_on(&candidates, [three.lit[0], three.lit[1], false], three.k, |p, c| match c {
        None => three.reference(p),
        Some(c) => three.model(p, &candidates[c]),
    })?;
    refine(base, three.k, &|p: usize, keep: &[usize]| three.model(p, keep))
}

/// The two points beside the three (the user's change of 2026-11-20), for a scene and its three-point selection: the
/// two-point P and its time; why the three points' P differs -- the ranking (the levels in order over two points and
/// over three) or the middle point (the two-point P's errors there: populations, R, F⃗).
fn beside(scene: &Scene, selection: Option<&Selection>) -> String {
    let three = ThreePoints::new(scene);
    let labels = &scene.request.states;
    let order = |points: usize| -> Vec<String> {
        ranking(scene, &three.references[..points], &three.points[..points]).iter().map(|l| name(&labels[l[0]])).collect()
    };
    let (two, three_ranked) = (order(2), order(POINTS));
    let ranked = if two == three_ranked { "the ranking the same".to_string() } else { format!("the ranking {two:?} -> {three_ranked:?}") };
    match two_point_choice(scene) {
        Err(why) => format!("two points: refused ({why}); {ranked}"),
        Ok(t) => {
            let middle = if !three.lit[2] {
                "dark".to_string()
            } else {
                match (three.reference(2), three.model(2, &t.keep)) {
                    (Ok(r), Ok(m)) => {
                        let e = errors(&m.0, &r.0, three.k, m.1);
                        format!("{:.2e} {:.2e} {:.2e}{}", e.populations, e.rate, e.force, if e.worst() <= GATE { "" } else { " (over)" })
                    }
                    (Err(w), _) | (_, Err(w)) => format!("unsolved: {w}"),
                }
            };
            let three_points = selection.map_or("refused".to_string(), |s| format!("base P {} -> P {} in {:.2} s", s.base().keep.len(), s.keep.len(), s.seconds));
            format!("two points: base P {} -> P {} in {:.2} s, that P at the middle {middle}; three points: {three_points}; {ranked}", t.base().keep.len(), t.keep.len(), t.seconds)
        }
    }
}

/// PSTATE's acceptance on the six scenes' requests (`V3D_REQ`; `PSTATE_SCENE` one of them): `choose`'s P passes the
/// gate at the three points -- the one assertion. Reported: the two points beside (`beside`); the base, every e_i, the
/// removal, its prediction and its measured errors, the checks, the time by stages beside one check of the base and
/// the budget (one check
/// times one more than the explicit excited states); for the user's choice of a gate (the code's stays `GATE`), P at
/// 2 % and 5 % -- the whole-level base at that gate with its own leave-one-out and descent, and the descent from the
/// 1 % base (PFINE predict.py's). With `PSTATE_OUT` a directory, each scene's numbers to <scene>.json there.
#[test]
fn choose_s_p_passes_at_the_three_points() {
    let out = std::env::var("PSTATE_OUT").ok().map(std::path::PathBuf::from);
    for (scene_name, scene) in six() {
        let labels = &scene.request.states;
        let named = |states: &[usize]| states.iter().map(|&s| labels[s].clone()).collect::<Vec<_>>();
        let selection = choose(&scene).unwrap();
        let p = selection.candidates.last().unwrap();
        assert!(p.keep == selection.keep && p.passes(), "{scene_name}: P {} does not pass at the three points: {}", selection.keep.len(), said(p));
        eprintln!("{scene_name}: {}", beside(&scene, Some(&selection)));
        let base = selection.base();
        let two = ThreePoints::new(&scene);
        let model = |p: usize, keep: &[usize]| two.model(p, keep);
        let clock = Instant::now();
        let _ = pair(&base.keep, two.lit, &model).unwrap();
        let check = clock.elapsed().as_secs_f64();
        let excited = selection.left_out.len();
        let mut order: Vec<&LeftOut> = selection.left_out.iter().collect();
        order.sort_by(|a, b| a.e().total_cmp(&b.e()));
        eprintln!(
            "{scene_name}: base P {} ({} explicit excited); P {} removing {:?}: predicted {:.3e}, measured {:.3e} ({}); {} checks",
            base.keep.len(), excited, selection.keep.len(), named(&selection.removed), selection.predicted, p.worst(), said(p), selection.candidates.len() - selection.whole
        );
        eprintln!("  base: worst {:.3e} ({})", base.worst(), said(base));
        for l in &order {
            eprintln!("  e({}) = {:.3e} ({})", labels[l.state], l.e(), said(&l.candidate));
        }
        for c in &selection.candidates[selection.whole..] {
            eprintln!("  check P {}: worst {:.3e} ({}){}", c.keep.len(), c.worst(), said(c), if c.passes() { " passes" } else { "" });
        }
        eprintln!(
            "  time: choose {:.3} s = base {:.3} + leave-one-out {:.3} + checks {:.3}; one check of the base {:.3} s, the budget of the stage after the base {:.3} s ({} + 1 checks)",
            selection.seconds, selection.stages[0], selection.stages[1], selection.stages[2], check, check * (1 + excited) as f64, excited
        );
        let mut gates = Vec::new();
        for gate in [0.02, 0.05] {
            let whole = selection.candidates[..selection.whole].iter().find(|c| c.within(gate)).expect("the base passes a wider gate");
            let own = if whole.keep == base.keep { selection.left_out.clone() } else { leave_one_out(whole, &selection.reference, two.k, &model).unwrap() };
            let at = descend(whole, &own, gate, &selection.reference, two.k, &model).unwrap();
            let from = descend(base, &selection.left_out, gate, &selection.reference, two.k, &model).unwrap();
            // P's measured worst: its check's, or its base's where nothing was removed.
            let measured = |d: &Descent, from: &Candidate| d.checks.last().filter(|c| c.keep == d.keep).map_or(from.worst(), |c| c.worst());
            eprintln!(
                "  gate {:.0} %: the whole-level base {} -> P {} (removing {:?}, predicted {:.3e}, measured {:.3e}, {} checks); from the 1 % base -> P {} (removing {:?}, predicted {:.3e}, measured {:.3e}, {} checks)",
                100.0 * gate, whole.keep.len(), at.keep.len(), named(&at.removed), at.predicted, measured(&at, whole), at.checks.len(),
                from.keep.len(), named(&from.removed), from.predicted, measured(&from, base), from.checks.len()
            );
            gates.push(serde_json::json!({
                "gate": gate, "whole_base": whole.keep.len(), "P": at.keep.len(), "removed": named(&at.removed), "predicted": at.predicted, "measured": measured(&at, whole), "checks": at.checks.len(),
                "from_base": {"P": from.keep.len(), "removed": named(&from.removed), "predicted": from.predicted, "measured": measured(&from, base), "checks": from.checks.len()},
            }));
        }
        if let Some(dir) = &out {
            let record = serde_json::json!({
                "scene": scene_name, "labels": labels, "base": as_json(base), "base_labels": named(&base.keep),
                "whole": selection.candidates[..selection.whole].iter().map(as_json).collect::<Vec<_>>(),
                "left_out": selection.left_out.iter().map(|l| serde_json::json!({"state": labels[l.state], "e": l.e(), "candidate": as_json(&l.candidate)})).collect::<Vec<_>>(),
                "removed": named(&selection.removed), "predicted": selection.predicted, "P": selection.keep.len(), "P_labels": named(&selection.keep),
                "checks": selection.candidates[selection.whole..].iter().map(as_json).collect::<Vec<_>>(),
                "seconds": selection.seconds, "stages": selection.stages, "one_check": check, "gates": gates,
            });
            std::fs::write(dir.join(format!("{scene_name}.json")), serde_json::to_string_pretty(&record).unwrap()).unwrap();
        }
    }
}

/// The time of a checkpoint's stages per point (one thread for each stage, the solve's own threads aside): the
/// reference, the frame, a candidate's model (`Rs::new` + `Rs::at`), and of each the Liouvillian, the elements, the
/// assembly and the solve; and `choose`'s, the least of five, by its stages. On the six scenes' requests (`V3D_REQ`).
#[test]
#[ignore]
fn choose_cost() {
    for (file, scene) in six() {
        let points = points(&scene);
        let zero = vec![0.0; scene.request.beams.len()];
        let references: Vec<ModelAt> = points.iter().map(|p| rs::reference(&scene, p)).collect();
        let order = ranking(&scene, &references, &points);
        for (i, p) in points.iter().enumerate() {
            let clock = Instant::now();
            let reference = rs::reference(&scene, p);
            let made = clock.elapsed().as_secs_f64();
            let clock = Instant::now();
            let frame = Frame::new(&scene, &reference.hilbert(), &zero);
            let framed = clock.elapsed().as_secs_f64();
            let keep: Vec<usize> = (0..GROUND).chain(order.iter().take(2).flatten().copied()).collect();
            let clock = Instant::now();
            let model = Rs::new(&scene, &keep, frame.sectors()).at(&scene, p);
            let modelled = clock.elapsed().as_secs_f64();
            for (what, m, built) in [("reference", &reference, made), ("model", &model, modelled)] {
                let clock = Instant::now();
                let liouville = Liouville::new(&frame, &m.hilbert());
                let l = clock.elapsed().as_secs_f64();
                let clock = Instant::now();
                let elements = select(&liouville);
                let e = clock.elapsed().as_secs_f64();
                let clock = Instant::now();
                let system = assemble(&liouville, &elements);
                let a = clock.elapsed().as_secs_f64();
                let clock = Instant::now();
                let solved = crate::krylov::solve(&system).unwrap();
                let k = clock.elapsed().as_secs_f64();
                eprintln!("{file} point {i} {what} (n {}, {} elements): made {built:.3} frame {framed:.3} | Liouville {l:.3} elements {e:.3} assembly {a:.3} solve {k:.3} s ({:?})", m.states.len(), elements.order.len(), solved.route);
            }
        }
        let runs: Vec<Selection> = (0..5).map(|_| choose(&scene).unwrap()).collect();
        let least = |f: &dyn Fn(&Selection) -> f64| runs.iter().map(f).fold(f64::INFINITY, f64::min);
        let s = &runs[0];
        eprintln!(
            "{file}: choose {:.3} s (the least of five; base {:.3}, leave-one-out {:.3}, checks {:.3}): base P {}, {} left out one at a time, P {}, {} checks",
            least(&|s| s.seconds), least(&|s| s.stages[0]), least(&|s| s.stages[1]), least(&|s| s.stages[2]), s.base().keep.len(), s.left_out.len(), s.keep.len(), s.candidates.len() - s.whole
        );
    }
}

/// Off the two points (PSTATE's risk, a measurement; `V3D_REQ`, `PSTATE_SCENE` one scene): the gate's measures of the
/// base and of P against the full reference where the gate does not look -- the static points moved along each axis by
/// λ/8 and λ/4 (λ the carrier's wavelength: the light's pattern there, its polarization and intensity, the mirror
/// states' cancellations broken), the tweezer as the scene has it there and off; the focus at a quarter, half and three
/// quarters of the tweezer's intensity; at rest. Each place's model on that
/// place's own frame, as the gate's own points. (At the thermal momentum along z the full reference of R4 at the centre
/// outgrew 8 GB; tilt0 and tilt2 there: pstate.status.md.)
#[test]
#[ignore]
fn off_the_two_points() {
    for (scene_name, scene) in six() {
        let selection = choose(&scene).unwrap();
        let k = selection_k(&scene);
        let lambda = 2.0 * std::f64::consts::PI / k;
        let mut places: Vec<(String, [f64; 3], f64)> = Vec::new();
        for (a, axis) in ["x", "y", "z"].into_iter().enumerate() {
            for (f, part) in [(0.125, "λ/8"), (0.25, "λ/4")] {
                let mut r = [0.0; 3];
                r[a] = f * lambda;
                places.push((format!("{part} {axis}, tweezer"), r, 1.0));
                places.push((format!("{part} {axis}, dark"), r, 0.0));
            }
        }
        // Between the two points: the focus at a quarter, half and three quarters of the tweezer's intensity (its
        // amplitude the square root), where its light shift moves the levels part way.
        for part in [0.25_f64, 0.5, 0.75] {
            places.push((format!("tweezer at {part} intensity"), [0.0; 3], part.sqrt()));
        }
        let base = selection.base().keep.clone();
        let rows: Vec<(PointError, PointError)> = places
            .par_iter()
            .map(|(_, r, tweezer)| {
                // At rest as the gate's points (`points`): no Doppler shift.
                let doppler = vec![0.0; scene.request.beams.len()];
                let components = scene.components(*r, &doppler, *tweezer);
                let reference = rs::reference(&scene, &components);
                let frame = Frame::new(&scene, &reference.hilbert(), &doppler);
                let full = solved(&frame, &reference).unwrap();
                let model = |keep: &[usize]| {
                    let (m, route) = solved(&frame, &Rs::new(&scene, keep, frame.sectors()).at(&scene, &components)).unwrap();
                    errors(&m, &full.0, k, route)
                };
                (model(&base), model(&selection.keep))
            })
            .collect();
        let e = |p: &PointError| format!("{:.2e} {:.2e} {:.2e}", p.populations, p.rate, p.force);
        eprintln!("{scene_name}: off the two points, populations R F⃗ -- the base P {} | P {}", base.len(), selection.keep.len());
        let mut most = [[0.0_f64; 3]; 2];
        for ((place, ..), (b, p)) in places.iter().zip(&rows) {
            eprintln!("  {place:<20} {} | {}", e(b), e(p));
            for (m, x) in most.iter_mut().zip([b, p]) {
                *m = [m[0].max(x.populations), m[1].max(x.rate), m[2].max(x.force)];
            }
        }
        eprintln!("  {:<20} {:.2e} {:.2e} {:.2e} | {:.2e} {:.2e} {:.2e}", "largest", most[0][0], most[0][1], most[0][2], most[1][0], most[1][1], most[1][2]);
    }
}

/// One initial component's preparation well, as the solve's (solve.rs `Prepared`, private there; the same public parts):
/// the run's prepared potential on the component's ground populations, its trap by the one descent from the trap
/// light's shortest length (solve.rs `trap_of` with the trap light lit), the trap light's beams and the envelope's peak.
struct Well<'a> {
    light: &'a crate::field::Light,
    potential: crate::model::selected::Potential<'a>,
    trap: crate::model::Trap,
}

impl<'a> Well<'a> {
    fn new(run: &'a crate::model::selected::Selected, populations: [f64; GROUND]) -> Well<'a> {
        let (light, potential) = (run.light(), run.prepared(&populations));
        let k = light.frequencies[light.trap].beams.iter().map(|&l| light.beams[l].wave_number).fold(0., f64::max);
        let trap = crate::model::potential::descend(1. / k, crate::step::Dynamics::hbar_over_mass(run), &|r| light.outside(r), &|r| potential.rounded(r)).expect("the prepared trap");
        Well { light, potential, trap }
    }
}

impl crate::ensemble::Well for Well<'_> {
    fn potential(&self, r: [f64; 3]) -> (f64, [f64; 3]) {
        let q = self.potential.at(r);
        (q.value, q.gradient)
    }
    fn trap(&self) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
        (self.trap.centre, self.trap.axes, self.trap.frequencies)
    }
    fn envelope(&self) -> (Vec<crate::field::Geometry>, f64) {
        (self.light.frequencies[self.light.trap].beams.iter().map(|&l| self.light.beams[l]).collect(), self.potential.peak())
    }
}

/// The run's domain, as the solve's (solve.rs `Widest`, private there): a hazard above the fastest excited state's decay
/// is a runaway.
struct Widest(f64);

impl crate::step::Domain for Widest {
    fn runaway(&self, hazard: f64) -> bool {
        hazard > self.0
    }
}

/// One trajectory's end (`the_first_checkpoint_in_pairs`): F=1's and F=2's populations and the kinetic temperature
/// where the atom is held, whether it is lost, its jumps, its start; and F=1's and F=2's populations at every output
/// it is held at (none after).
#[derive(Clone, Debug)]
struct End {
    populations: Option<[f64; 2]>,
    lost: bool,
    temperature: Option<f64>,
    jumps: u32,
    start: [f64; 3],
    outputs: Vec<Option<[f64; 2]>>,
}

impl End {
    /// F=1's and F=2's mean over the outputs from `from` (µs) on, where it is held at every one of them.
    fn mean_from(&self, times: &[f64], from: f64) -> Option<[f64; 2]> {
        let held: Vec<[f64; 2]> = times.iter().zip(&self.outputs).filter(|(t, _)| **t >= from).map(|(_, x)| *x).collect::<Option<_>>()?;
        let n = held.len() as f64;
        Some([held.iter().map(|x| x[0]).sum::<f64>() / n, held.iter().map(|x| x[1]).sum::<f64>() / n])
    }
}

/// The first checkpoint in pairs (the coordinator's check of PSTATE's P, a measurement; `V3D_REQ`, `PSTATE_SCENE` one
/// scene, `PSTATE_TAU` the step in µs -- the first result's --): the first result's trajectories -- the scout's, the
/// first member of each pilot group (solve.rs: group numbers from MOST_TRAJECTORIES / GROUP, index = number·GROUP) --
/// each on its own streams by its index (solve.rs `SEED`: its start by `Ensemble::draw`, its jumps, its Brownian path),
/// run on the whole-level base's model and on P's (`Selected::new` on the selection with that P: the run as the solve
/// makes it, which reads P alone), each advanced by the run's integrator (adapt.rs `advance`) at the same step to the
/// last output. Per trajectory at the end: F=1's and F=2's populations of the held atom, lost or not, its kinetic
/// temperature (ħ/m)|κ|²/(3k_B); and F=1's and F=2's means over the request's outputs (after t = 0, and over the
/// second half) -- the same trajectories read at every output, as the result reads them; per quantity the pairs'
/// differences (P − base), their mean and standard error; and the first output where a pair's F=1 parts by 0.05.
#[test]
#[ignore]
fn the_first_checkpoint_in_pairs() {
    use crate::step::Dynamics;
    let tau: f64 = std::env::var("PSTATE_TAU").expect("PSTATE_TAU: the first result's step (µs)").parse().unwrap();
    for (scene_name, request) in requests() {
        let scene = Scene::new(&request).unwrap();
        let mut selection = choose(&scene).unwrap();
        let whole = Selection { keep: selection.base().keep.clone(), candidates: selection.candidates[..selection.whole].to_vec(), ..selection.clone() };
        // `PSTATE_REMOVE` (labels, comma-separated): the second run on the base less these instead of on P (which removed
        // state does it).
        if let Ok(gone) = std::env::var("PSTATE_REMOVE") {
            let gone: Vec<&str> = gone.split(',').collect();
            selection.keep = whole.keep.iter().copied().filter(|&s| !gone.contains(&request.states[s].as_str())).collect();
            eprintln!("{scene_name}: the second run on the base less {gone:?}");
        }
        let reference = crate::solve::alone(crate::solve::chief(&request));
        let runs: Vec<crate::model::selected::Selected> =
            [whole, selection].into_iter().map(|s| crate::model::selected::Selected::new(Scene::new(&request).unwrap(), s, reference).unwrap()).collect();
        let hm = crate::units::hbar_over_mass(request.mass_kg);
        let initial = crate::solve::components(&request);
        let decay = crate::model::far::decay(&request);
        let domain = Widest((0..decay.rows).map(|x| decay[(x, x)].re).fold(0., f64::max));
        let end = *request.times_us.last().unwrap();
        let group = crate::ensemble::GROUP;
        let first = crate::units::MOST_TRAJECTORIES / group;
        let indices: Vec<u64> = (first..first + crate::solve::pilot_size() / group).map(|g| g * group).collect();
        let manifold = |f: &str| -> Vec<usize> { (0..STATES).filter(|&s| request.states[s].starts_with(&format!("ground-{f}-"))).collect() };
        let (one, two) = (manifold("1"), manifold("2"));
        let clock = Instant::now();
        let ends: Vec<Vec<End>> = runs
            .iter()
            .map(|run| {
                let wells: Vec<Well> = initial.iter().map(|&(s, _)| Well::new(run, crate::solve::alone(s))).collect();
                let components: Vec<(usize, f64, &dyn crate::ensemble::Well)> = initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn crate::ensemble::Well)).collect();
                let ensemble = crate::ensemble::Ensemble::new(&components, request.temperature_uk, hm).unwrap();
                let starts: Vec<(u64, crate::ensemble::Start)> = indices.iter().map(|&index| (index, ensemble.draw(crate::solve::SEED, index))).collect();
                starts
                    .into_par_iter()
                    .map(|(index, start)| {
                        let mut t = crate::step::Trajectory::new(run, start.state, start.r, start.kappa, crate::solve::SEED, index);
                        let mut scratch = crate::step::Scratch::default();
                        let mut path = crate::adapt::Path::new(crate::solve::SEED, index, 2. * tau, 1);
                        let mut outputs: Vec<Option<[f64; 2]>> = vec![None; request.times_us.len()];
                        let mut read = |k: usize, t: &crate::step::Trajectory, shown: Option<&[f64; STATES]>| {
                            let r = shown.copied().unwrap_or_else(|| run.readout(t.cluster, &t.point(), &t.psi));
                            outputs[k] = Some([one.iter().map(|&s| r[s]).sum::<f64>(), two.iter().map(|&s| r[s]).sum::<f64>()]);
                        };
                        let advanced = crate::adapt::advance(run, &domain, tau, &mut t, &mut scratch, &mut path, end, end, &request.times_us, &mut read, &mut |_, _| {});
                        let lost = t.lost.is_some();
                        let held = advanced.is_ok() && !lost;
                        let rows = held.then(|| run.readout(t.cluster, &t.point(), &t.psi));
                        End {
                            populations: rows.map(|r| [one.iter().map(|&s| r[s]).sum::<f64>(), two.iter().map(|&s| r[s]).sum::<f64>()]),
                            lost,
                            temperature: held.then(|| hm * t.kappa.iter().map(|k| k * k).sum::<f64>() / (3. * crate::units::KT_PER_UK)),
                            jumps: t.jumps,
                            start: start.r,
                            outputs,
                        }
                    })
                    .collect()
            })
            .collect();
        eprintln!(
            "{scene_name}: {} trajectories (the scout's) at τ {tau} µs to {end} µs, the base P {} and P {}, {:.1} s",
            indices.len(), runs[0].rs.keep.len(), runs[1].rs.keep.len(), clock.elapsed().as_secs_f64()
        );
        let said = |x: &End| match x.populations {
            Some(q) => format!("F1 {:.4} F2 {:.4} T {:7.3} µK jumps {:5}", q[0], q[1], x.temperature.unwrap(), x.jumps),
            None => format!("{} jumps {:5}", if x.lost { "lost" } else { "failed" }, x.jumps),
        };
        for (i, (b, p)) in ends[0].iter().zip(&ends[1]).enumerate() {
            eprintln!("  #{}: base {} | P {}", indices[i], said(b), said(p));
        }
        let apart = ends[0].iter().zip(&ends[1]).map(|(b, p)| (0..3).map(|a| (b.start[a] - p.start[a]).abs()).fold(0., f64::max)).fold(0., f64::max);
        eprintln!("  the pairs' starts apart by at most {apart:.2e} µm");
        let times = &request.times_us;
        let parted: Vec<String> = ends[0]
            .iter()
            .zip(&ends[1])
            .map(|(b, p)| {
                let k = b.outputs.iter().zip(&p.outputs).position(|(x, y)| matches!((x, y), (Some(x), Some(y)) if (x[0] - y[0]).abs() > 0.05));
                k.map_or("never".into(), |k| format!("{:.0}", times[k]))
            })
            .collect();
        eprintln!("  the pairs' F=1 first apart by 0.05 at (µs): {}", parted.join(" "));
        let quantities: [(&str, &dyn Fn(&End) -> Option<f64>); 9] = [
            ("F=1 at the end", &|x| x.populations.map(|p| p[0])),
            ("F=2 at the end", &|x| x.populations.map(|p| p[1])),
            ("F=1 over the outputs", &|x| x.mean_from(times, times[1]).map(|p| p[0])),
            ("F=2 over the outputs", &|x| x.mean_from(times, times[1]).map(|p| p[1])),
            ("F=1 over the second half", &|x| x.mean_from(times, 0.5 * end).map(|p| p[0])),
            ("F=2 over the second half", &|x| x.mean_from(times, 0.5 * end).map(|p| p[1])),
            ("lost", &|x| Some(f64::from(u8::from(x.lost)))),
            ("T at the end (µK)", &|x| x.temperature),
            ("jumps", &|x| Some(f64::from(x.jumps))),
        ];
        for (what, q) in quantities {
            let d: Vec<(f64, f64)> = ends[0].iter().zip(&ends[1]).filter_map(|(b, p)| Some((q(b)?, q(p)?))).collect();
            let n = d.len() as f64;
            let (mb, mp) = (d.iter().map(|x| x.0).sum::<f64>() / n, d.iter().map(|x| x.1).sum::<f64>() / n);
            let md = mp - mb;
            let se = (d.iter().map(|x| (x.1 - x.0 - md).powi(2)).sum::<f64>() / (n - 1.) / n).sqrt();
            eprintln!("  {what:<24} base {mb:.4} P {mp:.4}: P − base {md:+.4} ± {se:.4} (standard error of {} pairs; {:.1}σ)", d.len(), if se > 0. { md / se } else { 0. });
        }
    }
}

/// `choose` on every reference scene (the coordinator's integration check before the merge; ignored; the two points
/// beside the three, `beside`): each request of
/// the reference set (refs/scenes/*.json, not its *.scene.json app scenes nor the manifest; read as the model's tests
/// read it, model/testing.rs `scene`) and the six scenes' own requests (`V3D_REQ`). The one assertion: on every one,
/// `choose` gives a P that passes the gate at the three points (`choose_s_p_passes_at_the_three_points`'s); a scene whose
/// `choose` fails, errs or panics is a failure, all of them said at the end. Each scene's base P → P, the states left
/// out, the predicted and the measured error, the checks, `choose`'s wall clock.
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
    // Each scene's request, read where it is chosen (a request that does not read is a failure of its own).
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
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(crate::request::Request, Selection), String> {
            let request = read();
            let scene = Scene::new(&request)?;
            let selection = choose(&scene);
            eprintln!("{name}: {}", beside(&scene, selection.as_ref().ok()));
            Ok((request, selection?))
        }));
        let (request, selection) = match outcome {
            Ok(Ok(s)) => s,
            Ok(Err(why)) => {
                eprintln!("{name}: choose refused: {why}");
                failed.push(format!("{name}: {why}"));
                continue;
            }
            Err(payload) => {
                let why = payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                eprintln!("{name}: choose panicked: {why}");
                failed.push(format!("{name}: panicked: {why}"));
                continue;
            }
        };
        let p = selection.candidates.last().unwrap();
        let passes = p.keep == selection.keep && p.passes();
        let removed: Vec<&str> = selection.removed.iter().map(|&s| request.states[s].as_str()).collect();
        eprintln!(
            "{name}: base P {} -> P {} removing {removed:?}; predicted {:.3e}, measured {:.3e} ({}); {} checks; choose {:.2} s{}",
            selection.base().keep.len(), selection.keep.len(), selection.predicted, p.worst(), said(p), selection.candidates.len() - selection.whole, selection.seconds,
            if passes { "" } else { " -- DOES NOT PASS" }
        );
        if !passes {
            failed.push(format!("{name}: P {} does not pass at the three points: {}", selection.keep.len(), said(p)));
        }
    }
    eprintln!("{} scenes in {:.1} s; {} failed", all.len(), clock.elapsed().as_secs_f64(), failed.len());
    assert!(failed.is_empty(), "{failed:#?}");
}

/// Why no whole-level candidate passes (a diagnosis, ignored; `PSTATE_REFS` reference scenes by name, comma-separated):
/// per scene the frame (its fast frequencies, every state's and generator's fast sector) and at the three points the full
/// reference's readouts -- the levels, R and its channels, F⃗, the force's scale kR -- and every whole-level
/// candidate's, the largest (all 32 kept) included, each with the gate's three measures and the numbers they divide;
/// and `choose`'s outcome.
#[test]
#[ignore]
fn why_no_candidate_passes() {
    let names = std::env::var("PSTATE_REFS").expect("PSTATE_REFS: reference scenes, comma-separated");
    for name in names.split(',') {
        let request = crate::model::testing::scene(name);
        let scene = Scene::new(&request).unwrap();
        let two = ThreePoints::new(&scene);
        let labels = &request.states;
        match choose(&scene) {
            Ok(s) => eprintln!("{name}: choose: base P {} -> P {}, measured {:.3e}; {:?}", s.base().keep.len(), s.keep.len(), s.candidates.last().unwrap().worst(), s.notes()),
            Err(why) => eprintln!("{name}: choose refused: {why}"),
        }
        eprintln!("{name}: generators (rad/µs) {:?}; tweezer beam {}", scene.generators.iter().map(|g| g.frequency).collect::<Vec<_>>(), request.trap_beam);
        for (p, point) in ["centre", "edge", "middle"].into_iter().enumerate() {
            let frame = &two.frames[p];
            let sectors = frame.sectors();
            eprintln!("  {point}: frame fast {:?}; generators' sectors {:?}", frame.fast, sectors.generators);
            let mut by: Vec<(String, [i64; 2])> = Vec::new();
            for (s, sector) in sectors.states.iter().enumerate() {
                let level = labels[s].rsplit_once('-').map_or(labels[s].clone(), |(l, _)| l.trim_end_matches('-').to_string());
                if !by.iter().any(|(l, x)| *l == level && x == sector) {
                    by.push((level, *sector));
                }
            }
            eprintln!("  {point}: the levels' sectors {by:?}");
            let most = reach(&scene, &two.references[p], &two.points[p], &|_| true).iter().fold(0.0, |m: f64, &a| m.max(a));
            eprintln!("  {point}: light reaches the atom: {} (the largest first-order amplitude {most:.3e})", two.lit[p]);
            let (reference, route) = match two.reference(p) {
                Ok(r) => r,
                Err(why) => {
                    eprintln!("  {point} reference: {why}");
                    continue;
                }
            };
            let scale = |r: &Readouts| (r.force.iter().map(|f| f * f).sum::<f64>().sqrt(), two.k * r.rate);
            let (f, kr) = scale(&reference);
            eprintln!(
                "  {point} reference ({route:?}): levels {:?}; R {:.6e} (channels {:?}); F⃗ {:?} |F⃗| {f:.3e} kR {kr:.3e}",
                reference.levels.iter().map(|x| format!("{x:.4e}")).collect::<Vec<_>>(), reference.rate, reference.rates.iter().map(|x| format!("{x:.3e}")).collect::<Vec<_>>(), reference.force
            );
            // The reference's solve itself: its elements, route, certificate, whether ρ is a number.
            {
                let liouville = Liouville::new(frame, &two.references[p].hilbert());
                let elements = select(&liouville);
                let system = assemble(&liouville, &elements);
                let solved = crate::krylov::solve(&system).expect("the reference's solve (its error said above)");
                let finite = solved.rho.iter().filter(|z| z.re.is_finite() && z.im.is_finite()).count();
                let largest = solved.rho.iter().map(|z| z.abs()).fold(0., f64::max);
                eprintln!(
                    "  {point} reference's solve: {} elements, {:?}, certified {}, ρ finite {finite} of {}, max |ρ| {largest:.3e}",
                    elements.order.len(), solved.route, solved.certified, solved.rho.len()
                );
                for a in &solved.attempts {
                    eprintln!(
                        "    attempt {:?}: unfactorised {:?}; run {}",
                        a.route, a.unfactorised, a.run.as_ref().map_or("none".into(), |r| format!("{} iterations, ζ {:.3e}, η {:.3e}, certified {}, stop {:?}", r.iterations, r.zeta, r.eta, r.certified, r.stop))
                    );
                }
                let ground = |f: &str| (0..STATES).find(|&s| labels[s].starts_with(&format!("ground-{f}-"))).unwrap();
                eprintln!("  {point}: θ of ground-1 {:?}, of ground-2 {:?}; the frame's values {:?}; the request's axes (MHz) {:?}", frame.theta[ground("1")], frame.theta[ground("2")], frame.values, request.frequencies_mhz);
            }
            for keep in &two.candidates {
                let (model, route) = two.model(p, keep).unwrap();
                let e = errors(&model, &reference, two.k, route);
                eprintln!(
                    "    P {:2}: populations {:.3e} R {:.3e} F⃗ {:.3e}{} (80b9782e's worst, NaN dropped by f64::max: {:.3e}) | levels {:?}; R {:.6e}; F⃗ {:?}",
                    keep.len(), e.populations, e.rate, e.force, if e.worst() <= GATE { "" } else { " (over)" }, e.populations.max(e.rate).max(e.force),
                    model.levels.iter().map(|x| format!("{x:.4e}")).collect::<Vec<_>>(), model.rate, model.force
                );
            }
            // All 32 kept on the frame's sectors with the hyperfine coordinate n_φ taken out (a diagnosis: what the
            // model folds beyond the reference at P = 32 is K's fast elements by n_φ; the reference folds by n_tw alone).
            let mut plain = frame.sectors();
            plain.states.iter_mut().for_each(|s| s[0] = 0);
            plain.generators.iter_mut().for_each(|g| g[0] = 0);
            let all: Vec<usize> = (0..STATES).collect();
            let (model, route) = solved(frame, &Rs::new(&scene, &all, plain).at(&scene, &two.points[p])).unwrap();
            let e = errors(&model, &reference, two.k, route);
            eprintln!("    P 32 without n_φ: populations {:.3e} R {:.3e} F⃗ {:.3e} | R {:.6e}; F⃗ {:?}", e.populations, e.rate, e.force, model.rate, model.force);
        }
    }
}

fn selection_k(scene: &Scene) -> f64 {
    scene.generators.last().unwrap().frequency / SPEED_OF_LIGHT
}
