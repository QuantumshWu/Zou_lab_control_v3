//! STEP's prototype measurement for the Doppler series' strict tail (BRIEF_IMPL line STEP item 8; motion3d_status/rewrite/
//! study/PLAN_UNIFIED.md §6 item 6). Ignored; run by name, one at a time:
//! `<test exe> model::selected::tests::doppler::<name> --exact --include-ignored --nocapture --test-threads 1`.
//! - `doppler_census`: per scene of DOP_SCENE (comma separated, each `name=model@tau`; default PGC3D, TET, tilt0, tilt2
//!   with their ground states and R4 with D2-2-1) along a walk of DOP_STEPS (24) steps from a chief-state atom at the
//!   thermal speed of the request's temperature times DOP_SPEED (1; 3 and 10 a hotter atom), at each step's end:
//!   the fold's amplitudes' Doppler census (fold.rs `Local::doppler`: per term of M̂ its ratio ρ = |δ_c|/|z − λ_k| and
//!   its size) -- the largest ρ, and per order J the tail's strict bound Σ |M̂| ρ^{J+1}/(1 − ρ) over Σ |M̂| (relative to
//!   the amplitudes' whole) and its largest over the terms; the pieces, those that move (δ ≠ 0), their distinct shifts
//!   (the beams); and the model's split by its exact blocks with the largest F = 1 ↔ F = 2 element of the middle's
//!   generator per beat against its largest element, and that beat's frequency. Said per scene: the walk's largest
//!   and its mean.
use super::*;
use crate::step::{Brownian, Dynamics, Midway, Scratch, Trajectory};

/// A scene's request by name (the six scenes' native requests in V3C_REQ, else a reference scene).
fn request_of(name: &str) -> Request {
    let files = [("PGC3D", "PGC3D_1ms.json"), ("TET", "TET_1ms.json"), ("R4", "R4_LIN_4.json"), ("GMD1C", "GMD1C_1ms.json"), ("tilt0", "tilt0.json"), ("tilt2", "tilt2.json")];
    match files.iter().find(|(n, _)| *n == name) {
        Some((_, file)) => {
            let dir = std::path::PathBuf::from(std::env::var("V3C_REQ").expect("V3C_REQ"));
            let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap();
            line["options"]["device"] = serde_json::json!("cpu");
            for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
                form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
            }
            Request::from_native(&line).unwrap()
        }
        None => crate::model::testing::scene(name),
    }
}

struct Everywhere;
impl crate::step::Domain for Everywhere {
    fn runaway(&self, _: f64) -> bool {
        false
    }
}

#[test]
#[ignore]
fn doppler_census() {
    let list = std::env::var("DOP_SCENE").unwrap_or_else(|_| "PGC3D=ground@0.4741346189956611,TET=ground@0.4741346189956611,tilt0=ground@0.326,tilt2=ground@0.326,R4=+D2-2-1@0.23696682464454977".into());
    let steps: usize = std::env::var("DOP_STEPS").ok().map_or(24, |s| s.parse().unwrap());
    let hot: f64 = std::env::var("DOP_SPEED").ok().map_or(1.0, |s| s.parse().unwrap());
    for entry in list.split(',') {
        let (name, rest) = entry.split_once('=').unwrap();
        let (spec, tau) = rest.split_once('@').unwrap();
        let tau: f64 = tau.parse().unwrap();
        let request = request_of(name);
        let mut keep: Vec<usize> = (0..GROUND).collect();
        if let Some(labels) = spec.strip_prefix('+') {
            keep.extend(labels.split('+').map(|l| request.states.iter().position(|s| s == l).unwrap()));
        }
        keep.sort_unstable();
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let chief = crate::solve::chief(&request);
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep, ..selection }, crate::solve::alone(chief)).unwrap();
        let speed = hot * (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let mut t = Trajectory::new(&run, chief, [0.1, 0.2, 0.3], [speed, -0.7 * speed, 1.2 * speed], 7, 203);
        let plan = run.rs.plan();
        let (mut point, mut local) = (crate::scene::Point::default(), Local::default());
        let orders = 6;
        // The walk's largest and summed: ρ, the tail's relative bound per order and its largest term's.
        let (mut most, mut tails, mut worst, mut nodes) = (0.0_f64, vec![0.0_f64; orders], vec![0.0_f64; orders], 0usize);
        let (mut pieces, mut moving, mut shifts) = (0usize, 0usize, 0usize);
        let mut scratch = Scratch::default();
        for _ in 0..steps {
            // The fold at rest at the point (the pieces' frequencies there), then at the atom's momentum.
            run.scene.point_into(t.r, [0.0; 3], &mut point);
            run.rs.evaluate(&run.scene, &point, &mut local, Making::node(false));
            let rest = local.frequencies().to_vec();
            run.scene.point_into(t.r, t.kappa, &mut point);
            run.rs.evaluate(&run.scene, &point, &mut local, Making::node(false));
            let census = local.doppler(plan, &rest);
            let (mut mass, mut tail) = (0.0, vec![0.0; orders]);
            let mut distinct: Vec<f64> = Vec::new();
            for (_, _, delta, terms) in &census {
                if *delta > 0.0 && !distinct.iter().any(|d| (d - delta).abs() <= 1e-12 * delta) {
                    distinct.push(*delta);
                }
                for &(rho, size) in terms {
                    most = most.max(rho);
                    mass += size;
                    for (j, (t, w)) in tail.iter_mut().zip(worst.iter_mut()).enumerate() {
                        let bound = if rho < 1.0 { rho.powi(j as i32 + 1) / (1.0 - rho) } else { f64::INFINITY };
                        *t += size * bound;
                        *w = w.max(bound);
                    }
                }
            }
            for (s, t) in tails.iter_mut().zip(&tail) {
                *s += t / mass;
            }
            pieces = census.len();
            moving = census.iter().filter(|c| c.2 > 0.0).count();
            shifts = shifts.max(distinct.len());
            nodes += 1;
            let next = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Everywhere));
            match next {
                Ok(step) => t = step.end,
                Err(e) => {
                    eprintln!("DOP {name}: refused {e:?}");
                    break;
                }
            }
        }
        let k = nodes.max(1) as f64;
        let said: Vec<String> = (0..orders).map(|j| format!("J={j}: {:.2e} (term {:.2e})", tails[j] / k, worst[j])).collect();
        eprintln!("DOP {name} {spec} at κ {speed:.3} /µm, v {:.4} µm/µs ({hot}×thermal): {nodes} nodes, {pieces} amplitude pieces ({moving} moving, {shifts} shifts); ρ largest {most:.3e}; the tail's bound relative to M̂ (mean over the nodes; its largest term) {}", run.hbar_over_mass() * speed, said.join(", "));
        // The F = 1 ↔ F = 2 elements: the model's exact blocks, and the middle's generator's cross entries per beat.
        let middle = run.middle(0, t.r, t.time);
        let path = crate::step::Middle::path(&middle, t.kappa, t.kappa.map(|x| run.hbar_over_mass() * x));
        let node = middle.node(&path, 0.0);
        let n = run.rs.keep.len();
        let one: Vec<usize> = (0..n).filter(|&a| request.states[run.rs.keep[a]].starts_with("ground-1-")).collect();
        let two: Vec<usize> = (0..n).filter(|&a| request.states[run.rs.keep[a]].starts_with("ground-2-")).collect();
        let largest = node.generator.iter().map(|(_, m)| m.data.iter().map(|z| z.abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
        let cross: Vec<String> = node
            .generator
            .iter()
            .filter_map(|(beat, m)| {
                let c = one.iter().flat_map(|&a| two.iter().map(move |&b| (a, b))).map(|(a, b)| m.data[a * n + b].abs().max(m.data[b * n + a].abs())).fold(0.0, f64::max);
                (c > 0.0).then(|| format!("beat {beat} Ω {:.4e} rad/µs: {:.3e} of the largest", run.read(|b| b.value[*beat]), c / largest))
            })
            .collect();
        // The static beat's diagonal in the frame: the least gap between an F = 1 and an F = 2 state, and the largest
        // static cross element over it (the mixing a deletion would leave, to first order).
        let g0 = &node.generator.iter().find(|(beat, _)| *beat == 0).unwrap().1;
        let gap = one.iter().flat_map(|&a| two.iter().map(move |&b| (a, b))).map(|(a, b)| (g0.data[a * n + a] - g0.data[b * n + b]).abs()).fold(f64::INFINITY, f64::min);
        let coupling = one.iter().flat_map(|&a| two.iter().map(move |&b| (a, b))).map(|(a, b)| g0.data[a * n + b].abs().max(g0.data[b * n + a].abs())).fold(0.0, f64::max);
        eprintln!("DOP-SPLIT {name} {spec}: exact blocks {:?}; F=1↔F=2 elements of the generator {}; static: the largest {coupling:.3e} rad/µs over the least diagonal gap {gap:.3e} rad/µs, ratio {:.3e}", run.blocks, if cross.is_empty() { "none".to_string() } else { cross.join("; ") }, coupling / gap);
    }
}
