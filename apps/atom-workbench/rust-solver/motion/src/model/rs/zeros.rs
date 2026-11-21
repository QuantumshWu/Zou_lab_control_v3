//! What the zero tier takes out of the fixed matrices the engine makes (a measurement; ignored, run by name): per scene
//! of the six (V3C_REQ) and per kind -- the photon forms, the decay, the spherical channels, the tweezer's unit fold's
//! pairs, channels and scattering -- the entries other than zero that its computation's rounding reads as zeros
//! (|x| ≤ γ_K S, `Scaled::zeroed`) and those the rule until 2026-11-21 did (below u·max of their matrix), each with
//! the largest such |x| over its matrix's largest: what the two rules differ by.
use super::*;

#[test]
#[ignore]
fn the_zero_tier_on_the_fixed_matrices() {
    for (name, scene, _) in scenes() {
        let jumps = scene.jumps();
        let raw: [Scaled; 3] = jumps.emission.each_ref().map(Scaled::input);
        let c: [Scaled; 3] = crate::model::classify::emission_input(&jumps.emission).each_ref().map(Scaled::input);
        let k: [Scaled; 3] = c.each_ref().map(|x| {
            let value = Mat::from_fn(STATES, STATES, |i, j| x.value[(i, j)] * jumps.wave_number[(i, j)].re);
            Scaled { scale: value.data.iter().map(|z| z.abs()).collect(), value, terms: 1 }
        });
        let gamma = decay_scaled(&c);
        let unit = unit_fold(&scene, &c, &k, &gamma.zeroed());
        let basis = channel_basis(&c);
        let mut kinds: Vec<(&str, Vec<Scaled>)> = vec![
            ("emission (the input side's rule against u·max)", raw.to_vec()),
            ("photon forms", k.to_vec()),
            ("decay", vec![gamma.clone()]),
            ("spherical emission", basis.iter().map(|row| combined_scaled(row, [&c[0], &c[1], &c[2]])).collect()),
        ];
        kinds.push(("unit fold pairs", unit.shift.values().flat_map(|v| v.iter().map(|x| x.2.clone())).collect()));
        kinds.push(("unit fold channels", unit.channels.values().flat_map(|(p, kk)| p.values().chain(kk.values()).flat_map(|v| v.iter().map(|x| x.2.clone()))).collect()));
        kinds.push(("unit fold scattering", unit.scattering.iter().map(|x| x.2.clone()).collect()));
        for (kind, list) in kinds {
            let (mut new, mut old, mut both, mut most_new, mut most_old) = (0usize, 0usize, 0usize, 0.0_f64, 0.0_f64);
            for m in &list {
                let top = m.value.max_abs();
                let z = if kind.starts_with("emission") { crate::model::classify::emission_input(&std::array::from_fn(|a| raw[a].value.clone()))[list.iter().position(|x| std::ptr::eq(x, m)).unwrap()].clone() } else { m.zeroed() };
                for (x, y) in m.value.data.iter().zip(&z.data) {
                    if *x == C64::ZERO {
                        continue;
                    }
                    let by_new = *y == C64::ZERO;
                    let by_old = x.abs() < crate::units::UNIT_ROUNDOFF * top;
                    match (by_new, by_old) {
                        (true, true) => both += 1,
                        (true, false) => {
                            new += 1;
                            most_new = crate::linalg::larger(most_new, x.abs() / top);
                        }
                        (false, true) => {
                            old += 1;
                            most_old = crate::linalg::larger(most_old, x.abs() / top);
                        }
                        _ => {}
                    }
                }
            }
            eprintln!("ZEROS {name} {kind} ({} matrices): zero by both {both}; by γ_K S alone {new} (largest {most_new:.2e} of its matrix's largest); by u·max alone, now kept, {old} (largest {most_old:.2e})", list.len());
        }
    }
}
