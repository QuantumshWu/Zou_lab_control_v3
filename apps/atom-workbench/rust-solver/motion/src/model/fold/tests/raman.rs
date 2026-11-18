//! Raman pairs on the run's fold (BRIEF_RAMAN): where a near-resonant two-photon coupling of two ground states is kept.
use super::super::proto::Cache;
use super::*;

/// Diagnostic (RAMAN_REQ a native request's path): the components, the ground levels, the rule's Raman edges of mixing
/// beyond 1 with their mismatch, the rule's sets and frames, and per run set its blocks' offsets and sources, the static
/// G's ground entries at the focus and its tones.
#[test]
#[ignore]
fn raman_diagnostic() {
    let path = std::env::var("RAMAN_REQ").expect("RAMAN_REQ");
    let req = crate::model::testing::scene(&path);
    let share = crate::model::elim::share();
    let labels = req.levels().unwrap();
    let light = Light::new(&req);
    let axes = Axes::new(&req).unwrap();
    let comps = components(&req, &light, &axes).unwrap();
    for (i, c) in comps.iter().enumerate() {
        let names: Vec<&str> = c.beams.iter().map(|&l| req.beams[l].name.as_str()).collect();
        println!("component {i}: {:?} beat {:?} nu {:.5} beams {names:?}", c.part, c.beat, c.nu);
    }
    let e: Vec<f64> = (0..STATES).map(|s| req.hamiltonian[(s, s)].re).collect();
    for s in 0..GROUND {
        println!("ground {s}: F={} m={} E={:.5}", labels[s].f, labels[s].m, e[s]);
    }
    let d = decay(&req);
    let peak: Vec<Mat> = comps.iter().map(|c| Mat::from_fn(EXCITED, GROUND, |x, g| C64::real(c.couplings.iter().map(|m| m[(x, g)].abs()).sum()))).collect();
    for (f, cf) in comps.iter().enumerate() {
        for (g, cg) in comps.iter().enumerate() {
            for s in 0..GROUND {
                for t in (0..GROUND).filter(|&t| t != s) {
                    let a: f64 = (0..EXCITED).map(|x| peak[g][(x, t)].re * peak[f][(x, s)].re / C64::new(e[s] + cf.nu - e[GROUND + x], 0.5 * d[(x, x)].re).abs()).sum();
                    let delta = e[t] - e[s] - (cf.nu - cg.nu);
                    if a > 0.0 && a / delta.abs() > 1.0 {
                        println!("edge {t} <- {s} by {f} then {g}: a {a:.4} Δ {delta:.5} m {:.3e} unfoldable {}", a / delta.abs(), unfoldable(a / delta.abs(), share));
                    }
                }
            }
        }
    }
    let choice = Choice::of(&req, share).unwrap();
    println!("rule's sets {:?}", choice.sets);
    for s in 0..GROUND {
        println!("frame {s}: {:?} = {:.5}", choice.frames[s], axes.omega(&choice.frames[s]));
    }
    // The averaged terms folded (run.rs # The averaged terms folded): their second order against the static generator at
    // the focus, per ground set's cluster.
    let run = super::super::run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
    for k in 0..run.sets_len() {
        use crate::step::{Dynamics, Middle as _};
        let middle = run.middle(k, [0.0; 3], 0.0, &crate::step::Held::default(), false);
        let path = middle.path([0.0; 3], [0.0; 3]);
        let (folded, whole, beats) = middle.folded(&path);
        println!("cluster {k}: the averaged terms' second order {folded:.3e} against the generator's {whole:.3e} ({:.2e} of it; {beats} averaged beats)", folded / whole.max(1e-300));
    }
    let base = Choice::base(&req, vec![], vec![]).unwrap();
    for set in Fold::sets(&req, &base).unwrap() {
        let fold = Fold::new(&req, &base, set[0]).unwrap();
        println!("set {set:?}: {} blocks; frames {:?}; cross decays kept {:?}", fold.blocks.len(), fold.frame_values, (0..fold.blocks.len()).flat_map(|i| (i + 1..fold.blocks.len()).map(move |j| (i, j))).filter(|&(i, j)| fold.beating(i, j)).collect::<Vec<_>>());
        for i in 0..fold.blocks.len() {
            for j in i + 1..fold.blocks.len() {
                if fold.crossing[i * fold.blocks.len() + j] {
                    println!("  cross decay {i} {j}: beat {:.3} {}", fold.blocks[i].value - fold.blocks[j].value, if fold.beating(i, j) { "tone" } else { "averaged" });
                }
            }
        }
        for b in &fold.blocks {
            let sources: Vec<(usize, usize)> = b.sources.iter().map(|&(p, c)| (fold.ground[p], c)).collect();
            println!("  block {:?} = {:.5}: sources (state, component) {sources:?}, rows {} solved {}", b.offset, b.value, b.rows.len(), b.solved);
        }
        if fold.blocks.is_empty() {
            continue;
        }
        let Ok(mut ws) = fold.prototype_workspace() else {
            println!("  no prototype workspace");
            continue;
        };
        let mut cache = Cache::default();
        let Some(step) = fold.prototype_within([0.0; 3], [0.0; 3], fold.thermal, &mut cache, &mut ws, true) else {
            println!("  no fold at the focus");
            continue;
        };
        let g = step.generator.clone();
        let n = fold.ground.len();
        for p in 0..n {
            let row: Vec<String> = (0..n).map(|q| format!("{:+.4}{:+.4}i", g[(p, q)].re, g[(p, q)].im)).collect();
            println!("  G[{}] {}", fold.ground[p], row.join(" "));
        }
        for (m, (beat, value)) in ws.tones().enumerate() {
            let t = ws.tone(m, 0);
            let largest = t.data.iter().map(|z| z.abs()).fold(0.0, f64::max);
            println!("  tone {m}: {beat:?} = {value:.5}, largest entry {largest:.4e}");
        }
    }
}
