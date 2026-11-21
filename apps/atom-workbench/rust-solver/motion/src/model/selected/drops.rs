//! The drop tier's checks (model/classify.rs, # The drop tier; BRIEF_JUDGE item 2): its bound holds over the reach, and
//! what a step saves.
use super::*;

/// tilt2's request (V3C_REQ; the six scenes' one whose node holds beats but the static one).
fn tilt2() -> Request {
    scenes().into_iter().find(|s| s.0 == "tilt2").expect("tilt2 among the six scenes (V3E_SCENE unset or tilt2)").1
}

/// The largest singular value of a matrix from below: ‖G v‖/‖v‖ after power iterations on G†G (any v gives a lower
/// bound of ‖G‖₂).
fn norm_below(g: &Mat) -> f64 {
    let n = g.cols;
    let mut v = vec![C64::ONE; n];
    let mut best = 0.0_f64;
    for _ in 0..60 {
        let w: Vec<C64> = (0..g.rows).map(|i| (0..n).map(|j| g[(i, j)] * v[j]).sum()).collect();
        let (nv, nw) = (norm_sqr(&v).sqrt(), norm_sqr(&w).sqrt());
        if nv == 0.0 || nw == 0.0 {
            break;
        }
        best = larger(best, nw / nv);
        v = (0..n).map(|j| (0..g.rows).map(|i| g[(i, j)].conj() * w[i]).sum::<C64>() * (1.0 / nw)).collect();
    }
    best
}

/// The drop tier's bound holds over the reach (# The drop tier): tilt2's model with nothing dropped (its candidate
/// pairs those of the judged run), at points of the trap's region, along and across the tweezer's axis to well beyond
/// its intensity's 1/20 and far off it (s → 0), at rest and moving with κ_max(s) = √(2U₀s/(ħ/m)) along each lab axis
/// and the diagonal: each pair's two slots' ‖G_k‖₂ + ‖G_−k‖₂ (from below) at most its Y; said per pair beside its Y,
/// its H_eff part's and decay's largest, and the run's drop (the pairs dropped, Y, the run's D). tilt2's judged run
/// drops ±3ν, ±4ν, ±5ν, ±6ν and keeps ±ν, ±2ν.
#[test]
fn the_dropped_beats_are_within_their_bound() {
    let request = tilt2();
    let (run, whole) = (made_with(&request, &Drops::Judged), made_with(&request, &Drops::Only(Vec::new())));
    let tw = &whole.scene.geometry[whole.scene.tweezer()];
    let (w, zr) = (tw.waist.unwrap(), tw.rayleigh().unwrap());
    let (across, other) = crate::linalg::perpendicular(tw.direction);
    let depth = whole.scene.trap_potential([0.0; 3], &whole.depth).iter().map(|q| -q.value).fold(0.0, larger);
    let mut places: Vec<V3> = vec![[0.0; 3]];
    for t in [0.3, 1.0, 2.0, 5.0, 20.0] {
        for sign in [-1.0, 1.0] {
            places.push(tw.direction.map(|d| sign * t * zr * d));
        }
    }
    for t in [0.2, 0.5, 1.0, 1.5, 3.0] {
        places.push(across.map(|d| t * w * d));
        places.push(other.map(|d| -t * w * d));
    }
    places.push(std::array::from_fn(|a| 0.7 * w * across[a] + 0.4 * zr * tw.direction[a] - 0.3 * w * other[a]));
    let mut points: Vec<(V3, V3)> = Vec::new();
    for r in places {
        let s = tw.amplitude(r).value.norm_sqr();
        let k = (2.0 * depth * s / whole.hbar_over_mass).sqrt();
        let d = k / 3.0_f64.sqrt();
        for kappa in [[0.0; 3], [k, 0.0, 0.0], [-k, 0.0, 0.0], [0.0, k, 0.0], [0.0, -k, 0.0], [0.0, 0.0, k], [0.0, 0.0, -k], [d, d, d]] {
            points.push((r, kappa));
        }
    }
    let pairs = &whole.judgment().pairs;
    assert!(!pairs.is_empty(), "tilt2's node holds beats but the static one");
    let values = whole.components().0;
    let mut most = vec![(0.0_f64, 0.0_f64, 0.0_f64); pairs.len()];
    for &(r, kappa) in &points {
        let node = whole.node(r, kappa, false);
        for (p, (slots, _)) in pairs.iter().enumerate() {
            let beats: Vec<usize> = slots.iter().map(|&k| whole.judgment().tiers[k].0).collect();
            let g: Vec<&Mat> = beats.iter().map(|b| &node.generator.iter().find(|x| x.0 == *b).expect("the beat in the node").1).collect();
            let total: f64 = g.iter().map(|m| norm_below(m)).sum();
            let (h, d) = if g.len() == 2 {
                let n = g[0].rows;
                let h = Mat::from_fn(n, n, |a, b| (g[0][(a, b)] + g[1][(b, a)].conj()) * 0.5);
                let d = Mat::from_fn(n, n, |a, b| (g[0][(a, b)] - g[1][(b, a)].conj()) * C64::I);
                (h.max_abs(), d.max_abs())
            } else {
                (g[0].max_abs(), 0.0)
            };
            most[p] = (larger(most[p].0, total), larger(most[p].1, h), larger(most[p].2, d));
        }
    }
    for (p, (slots, y)) in pairs.iter().enumerate() {
        let omega = values[whole.judgment().tiers[slots[0]].0].abs();
        let tier = run.judgment().tiers[slots[0]].1;
        eprintln!(
            "JUDGE tilt2 pair |Ω| {omega:.6} rad/µs ({} slots): Y {y:.3e} /µs; over {} points ‖G‖ ≤ {:.3e} (Y over it {:.2}), H_eff largest {:.3e}, decay largest {:.3e}; {tier:?}",
            slots.len(),
            points.len(),
            most[p].0,
            y / most[p].0,
            most[p].1,
            most[p].2
        );
        assert!(most[p].0 <= *y, "tilt2's pair at {omega} rad/µs: ‖G‖ {} above its bound {y}", most[p].0);
    }
    let length = request.times_us.last().copied().unwrap_or(0.0);
    eprintln!("JUDGE tilt2 run: Y {:.3e} /µs, D ≤ {:.3e} over {length} µs, blocks {:?}", run.dropped(), 0.5 * (2.0 * run.dropped() * length).exp_m1(), run.blocks().iter().map(Vec::len).collect::<Vec<_>>());
    let nu = pairs.iter().map(|p| values[whole.judgment().tiers[p.0[0]].0].abs()).fold(f64::INFINITY, smaller);
    let mut dropped: Vec<i64> = run.judgment().pairs.iter().filter(|p| run.judgment().tiers[p.0[0]].1 == Tier::Drop).map(|p| (values_of(&run)[run.judgment().tiers[p.0[0]].0].abs() / nu).round() as i64).collect();
    dropped.sort_unstable();
    assert_eq!(dropped, [3, 4, 5, 6], "tilt2 drops ±3ν … ±6ν");
}

/// A run's beats' values.
fn values_of(run: &Selected) -> Vec<f64> {
    run.components().0
}

/// What a step saves by the drop (a measurement; ignored, run by name, one thread, pinned by the caller): tilt2's model
/// three ways -- nothing dropped (six tones), its ±5ν and ±6ν dropped (the four tones the ε·max rule left before the
/// judgment), the judged drop (two tones) --, from one state (an atom of `starts` at the request's temperature, V3E_WARM
/// steps (5) on the judged model), V3E_STEPS steps (20) of τ = V3E_TAU (0.326 µs) with one noise, V3E_PASSES passes (5)
/// in turn over the three, each one's least; said per step (µs) with the jumps and the nodes made.
#[test]
#[ignore]
fn judge_step_time() {
    let (tau, steps, passes, warm) = (env("V3E_TAU", 0.326), env("V3E_STEPS", 20.0) as usize, env("V3E_PASSES", 5.0) as usize, env("V3E_WARM", 5.0) as usize);
    let request = tilt2();
    let whole = made_with(&request, &Drops::Only(Vec::new()));
    let values = values_of(&whole);
    let omegas: Vec<f64> = whole.judgment().pairs.iter().map(|p| values[whole.judgment().tiers[p.0[0]].0].abs()).collect();
    let nu = omegas.iter().copied().fold(f64::INFINITY, smaller);
    let high: Vec<f64> = omegas.iter().copied().filter(|v| (v / nu).round() >= 5.0).collect();
    let runs = [("none", whole), ("before (±5ν, ±6ν dropped)", made_with(&request, &Drops::Only(high))), ("judged", made_with(&request, &Drops::Judged))];
    let start = starts(&runs[2].1, request.temperature_uk, 1);
    let (warmed, _) = through(&runs[2].1, &start, tau, warm);
    let t = &warmed[0];
    let from = vec![(t.r, t.kappa, t.psi.iter().copied().collect::<Vec<C64>>(), t.threshold)];
    let mut least = [f64::INFINITY; 3];
    let mut jumps = [0u32; 3];
    for _ in 0..passes {
        for (i, (_, run)) in runs.iter().enumerate() {
            let (ends, seconds) = through(run, &from, tau, steps);
            least[i] = smaller(least[i], seconds);
            jumps[i] = ends[0].jumps;
        }
    }
    for (i, (name, run)) in runs.iter().enumerate() {
        let tones = run.candidates.len();
        eprintln!("JUDGESTEP tilt2 {name}: {tones} tones, Y {:.3e} /µs: {:.1} µs a step (least of {passes} passes of {steps} steps of τ {tau}), {} jumps", run.dropped(), 1e6 * least[i] / steps as f64, jumps[i]);
    }
    eprintln!("JUDGESTEP tilt2: judged against before {:+.1} %, against none {:+.1} %", 100.0 * (least[2] / least[1] - 1.0), 100.0 * (least[2] / least[0] - 1.0));
}
