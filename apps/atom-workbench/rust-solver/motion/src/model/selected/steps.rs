//! A step's time on the six scenes (a measurement; BRIEF_JUDGE item 4: what the model's evaluation saves per step).
use super::*;

/// Per scene of the six (V3C_REQ; V3E_SCENE one of them) its run as the solve makes it, an atom of `starts` at the
/// request's temperature warmed V3E_WARM steps (5), then V3E_STEPS steps (20) of its PROF1 step (GMD1C, PGC3D, TET
/// 0.474 µs; R4 0.237; tilt0, tilt2 0.326) with one noise, V3E_PASSES passes (5) from that one state, the least; said
/// per step (µs) with the jumps. One thread; the caller pins it.
#[test]
#[ignore]
fn judge_steps_on_the_six_scenes() {
    let (steps, passes, warm) = (env("V3E_STEPS", 20.0) as usize, env("V3E_PASSES", 5.0) as usize, env("V3E_WARM", 5.0) as usize);
    for (name, request, _) in scenes() {
        let tau = match name {
            "R4" => 0.237,
            "tilt0" | "tilt2" => 0.326,
            _ => 0.474,
        };
        let run = made(&request, true);
        let start = starts(&run, request.temperature_uk, 1);
        let (warmed, _) = through(&run, &start, tau, warm);
        let t = &warmed[0];
        let from = vec![(t.r, t.kappa, t.psi.iter().copied().collect::<Vec<C64>>(), t.threshold)];
        let (mut least, mut jumps) = (f64::INFINITY, 0);
        for _ in 0..passes {
            let (ends, seconds) = through(&run, &from, tau, steps);
            least = smaller(least, seconds);
            jumps = ends[0].jumps;
        }
        eprintln!("JUDGESTEPS {name}: P {}, τ {tau}: {:.1} µs a step (least of {passes} passes of {steps} steps), {jumps} jumps", run.rs.keep.len(), 1e6 * least / steps as f64);
    }
}
