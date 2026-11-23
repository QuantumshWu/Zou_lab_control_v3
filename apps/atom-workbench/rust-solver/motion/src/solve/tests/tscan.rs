//! T2STEP2's measurement (ignored, by name): a forced model's step error against the step -- the same trajectories (the
//! same numbers: the same initial atoms, jumps' draws and Brownian path) run at the steps τ_k = top/2^k, k = 0..levels,
//! every level on the same groups, read at the coarsest level's step ends. Per level its loss at some outputs; per pair
//! of adjacent levels (τ, 2τ) the loss's difference Q(2τ) − Q(τ) at those times with its jackknife, and the check's own
//! read of the pair (`read_as_it_is`, its loss and T looks over the share's square); per level its loss against the
//! finest level's -- what the step error is, where the finest stands for the converged run.
//!
//! T2S2_SCENE a refs/scenes name (R9_PUSH); T2S2_EXCITED the explicit excited states beside the ground states (labels,
//! comma separated; empty: G); T2S2_TOP the coarsest step (µs; the ladder's first rung times four where not given),
//! T2S2_LEVELS the levels (8); T2S2_GROUPS the groups of eight (32); T2S2_UNTIL the last time printed (50 µs);
//! T2S2_EVERY the outputs' spacing printed (2 µs).
use super::super::*;

fn var<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|x| x.parse().ok()).unwrap_or(default)
}

/// A reading's whole and its jackknife's standard error over the groups.
fn with_error(r: &crate::analysis::Reading) -> (f64, f64) {
    let g = r.leave.len() as f64;
    let mean = r.leave.iter().sum::<f64>() / g;
    let var = r.leave.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() * (g - 1.) / g;
    (r.whole, var.sqrt())
}

/// STUDY B: the loss's step-error read over windows of the run instead of the whole run -- per time its unbiased mean
/// product of two groups' values (estimate.rs `between`, the same estimator), each window's trapezoid mean over its
/// times with its delete-one-group jackknife --, the run cut into K equal windows (K = 1: the whole run, the check's
/// own read), and the worst window over the share's square b, its error and its span: what a windowed read of the same
/// pairs says where the error is confined to a stretch of the run.
fn windowed(times: &[f64], fine: &[&Part], coarse: &[&Part], scales: &Scales, tau: f64, double: f64) {
    let index = 1 + crate::request::STATES + 3;
    let (f, c) = (reads_of(fine), reads_of(coarse));
    let weights = crate::estimate::trapezoid(times);
    // Per time k ≥ 1: its weight, the groups' pseudo-values' sum, and the pseudo-values.
    let mut per: Vec<(f64, f64, Vec<f64>)> = Vec::new();
    let mut at: Vec<f64> = Vec::new();
    for k in 1..times.len() {
        let r = &crate::estimate::readings(&f, &c, k, scales)[index];
        if !r.whole.is_finite() || r.leave.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let g = r.leave.len() as f64;
        let x: Vec<f64> = r.leave.iter().map(|l| g * r.whole - (g - 1.) * l).collect();
        per.push((weights[k], x.iter().sum(), x));
        at.push(times[k]);
    }
    let Some(first) = per.first() else { return };
    let g = first.2.len();
    let gf = g as f64;
    let b = (crate::units::STEP_SHARE * POPULATION_TARGET).powi(2);
    let end = times.last().copied().unwrap_or(0.);
    let mut line = Vec::new();
    for parts in [1usize, 4, 16, 64] {
        let mut worst: Option<(f64, f64, f64, f64)> = None;
        for w in 0..parts {
            let (from, to) = (end * w as f64 / parts as f64, end * (w + 1) as f64 / parts as f64);
            let (mut whole, mut weight, mut without) = (0., 0., vec![0.; g]);
            for ((wk, s, x), &t) in per.iter().zip(&at) {
                if t <= from || t > to {
                    continue;
                }
                let q: f64 = x.iter().map(|v| v * v).sum();
                whole += wk * (s * s - q);
                weight += wk;
                for (o, xi) in without.iter_mut().zip(x) {
                    *o += wk * ((s - xi).powi(2) - (q - xi * xi));
                }
            }
            if weight <= 0. {
                continue;
            }
            let value = whole / (weight * gf * (gf - 1.));
            let loo: Vec<f64> = without.iter().map(|o| o / (weight * (gf - 1.) * (gf - 2.))).collect();
            let mean = loo.iter().sum::<f64>() / gf;
            let se = ((gf - 1.) / gf * loo.iter().map(|v| (v - mean).powi(2)).sum::<f64>()).sqrt();
            if worst.is_none_or(|x| value / b > x.0) {
                worst = Some((value / b, se / b, from, to));
            }
        }
        if let Some((v, s, from, to)) = worst {
            line.push(format!("K={parts}: {v:+.3} ± {s:.3} b over ({from:.0}, {to:.0}]"));
        }
    }
    println!("T2S2 windowed ({tau:.5}, {double:.5}) loss, worst window: {}", line.join("; "));
}

#[test]
#[ignore]
fn t2step2_scan() {
    let name: String = var("T2S2_SCENE", "R9_PUSH".to_string());
    let excited: Vec<String> = var("T2S2_EXCITED", String::new()).split(',').filter(|x| !x.is_empty()).map(str::to_string).collect();
    // STUDY B: a request file of the app's native line (as step/tests/t2step.rs `scenes` reads the six scenes) where
    // T2S2_SCENE is a path; G forced (the ground states alone) where no excited state is listed.
    let request = if name.ends_with(".json") {
        let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&name).unwrap()).unwrap();
        line["options"]["device"] = serde_json::json!("cpu");
        if let Some(forms) = line["model"]["motion"]["outsideForms"].as_array_mut() {
            for form in forms {
                form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
            }
        }
        Request::from_native(&line).unwrap()
    } else {
        crate::model::testing::scene(&name)
    };
    let reference = alone(chief(&request));
    let run = if excited.is_empty() {
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        Selected::new(scene, crate::select::Selection { keep: (0..GROUND).collect(), ..selection }, reference).unwrap()
    } else {
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let mut keep: Vec<usize> = (0..GROUND).collect();
        keep.extend(excited.iter().map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
        keep.sort_unstable();
        Selected::new(scene, crate::select::Selection { keep, ..selection }, reference).unwrap()
    };
    // As `attempt` sets an attempt up (as `r2ladder_companions` does).
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(&request);
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&run, alone(s)).unwrap()).collect();
    let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
    let lead = initial.iter().position(|&(s, _)| s == chief(&request)).unwrap();
    let trap = &wells[lead].trap;
    let report = reported(&run, &alone(initial[lead].0)).unwrap_or_else(|_| trap.clone());
    let scales = Scales { hbar_over_mass: hm, frequencies: report.lab };
    let plan = Plan::new(&request.times_us, report.lab);
    let kinetic = kinetic_uk(trap, request.temperature_uk).iter().sum::<f64>() / 3.;
    let (frequency, end) = (trap.frequencies.iter().copied().fold(0., crate::linalg::larger), plan.end);
    let kick = kick_curvature(&run, trap);
    let first_rung = Ladder { run: end, prior: prior(run.pattern(), pattern_wave_number(run.light()), hm, kinetic, &scales), frequency, kick }.first().unwrap();
    let top: f64 = var("T2S2_TOP", 4. * first_rung);
    let top = end / (2. * (end / (2. * top)).round());
    let levels: usize = var("T2S2_LEVELS", 8);
    let groups: u64 = var("T2S2_GROUPS", 32);
    let (until, every): (f64, f64) = (var("T2S2_UNTIL", 50.), var("T2S2_EVERY", 2.));
    let domain = Domain { widest: Widest::of(&request) };
    let mut plan = Plan { cell: top, ..plan };
    plan.read_at(grid(end, 0.5 * top));
    let first = MOST_TRAJECTORIES / GROUP;
    let numbers: Vec<u64> = (first..first + groups).collect();
    let mut silent = |_: Message| {};
    let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0, stage: None };
    let taus: Vec<f64> = (0..levels).map(|k| top / f64::powi(2., k as i32)).collect();
    println!(
        "T2S2 {name} P{} (excited {:?}): first rung {first_rung:.5} µs, the trap's fastest {frequency:.4} rad/µs, the run {end} µs; {} trajectories a level; steps {:?}",
        run.rs.keep.len(),
        excited,
        groups * GROUP,
        taus
    );
    // T2S2_KEEP: each trajectory's end kept (`Part::ends`: its number and the time it is held), its loss time against
    // the finest level's of the same number read below.
    let keep = std::env::var("T2S2_KEEP").is_ok();
    let mut parts: Vec<Vec<Part>> = Vec::new();
    for &tau in &taus {
        let clock = Instant::now();
        let level = level(&run, &ensemble, &domain, &plan, tau, &numbers, 0..GROUP, keep, &mut messages, &mut |_, _| {});
        let counts = counts_of(&level.iter().collect::<Vec<_>>());
        println!("T2S2 level {tau:.5} µs: {:.1} s, halved time share {:.3e}", clock.elapsed().as_secs_f64(), counts.halved.time / counts.time.max(1e-300));
        parts.push(level);
    }
    let refs: Vec<Vec<&Part>> = parts.iter().map(|l| l.iter().collect()).collect();
    // The loss per level at the outputs printed.
    let estimates: Vec<Vec<Estimate>> = refs.iter().map(|l| pooled(l, &scales).0).collect();
    let shown: Vec<usize> = (0..request.times_us.len()).filter(|&k| request.times_us[k] <= until && (request.times_us[k] / every).fract().abs() < 1e-9).collect();
    for &k in &shown {
        let line: Vec<String> = estimates.iter().map(|e| format!("{:.4}", e[k].loss)).collect();
        println!("T2S2 L t {:6.1}: {}", request.times_us[k], line.join(" "));
    }
    // The held atoms' mean velocity and temperature along z, per level (m/s, µK).
    for &k in &shown {
        let line: Vec<String> = estimates.iter().map(|e| e[k].motion.map_or("-".to_string(), |m| format!("{:.4}/{:.0}", m.mean_velocity[2], m.temperature[2]))).collect();
        println!("T2S2 vz/Tz t {:6.1}: {}", request.times_us[k], line.join(" "));
    }
    if keep {
        // Each level's trajectories' held times against the finest's of the same number: the mean shift, its error, the
        // share within 0.1 µs (pairs that kept together up to the loss).
        let held = |level: &[Part]| -> std::collections::HashMap<u64, f64> { level.iter().flat_map(|p| p.ends.iter().map(|&(i, h, _)| (i, h))).collect() };
        let finest = held(&parts[levels - 1]);
        for (j, level) in parts.iter().enumerate().take(levels - 1) {
            let d: Vec<f64> = held(level).iter().filter_map(|(i, &h)| finest.get(i).map(|&f| h - f)).filter(|x| x.is_finite() && x.abs() < 100.).collect();
            let n = d.len() as f64;
            let mean = d.iter().sum::<f64>() / n;
            let sd = (d.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.)).sqrt();
            let close = d.iter().filter(|x| x.abs() < 0.1).count() as f64 / n;
            let mut sorted = d.clone();
            sorted.sort_by(f64::total_cmp);
            println!("T2S2 held {:.5} against {:.5}: {} pairs, mean shift {:+.4} ± {:.4} µs, median {:+.4}, within 0.1 µs {:.3}", taus[j], taus[levels - 1], d.len(), mean, sd / n.sqrt(), sorted[d.len() / 2], close);
        }
    }
    // Per pair of adjacent levels, and per level against the finest, the loss's difference at the read times printed.
    let times = &plan.reads;
    let at: Vec<usize> = (0..times.len()).filter(|&k| times[k] > 0. && times[k] <= until).collect();
    let loss_d = |fine: &[&Part], coarse: &[&Part], k: usize| -> (f64, f64) {
        let r = crate::estimate::readings(&reads_of(fine), &reads_of(coarse), k, &scales);
        // The step error of the loss, d/(2^p − 1) (estimate.rs `step_error`), after the loss, the rows and T's three
        // axes; d itself.
        let (e, s) = with_error(&r[1 + crate::request::STATES + 3]);
        let share = crate::decide::richardson(0.5, ORDER as f64) - 1.;
        (e / share, s / share)
    };
    for j in 0..levels - 1 {
        let (coarse, fine) = (&refs[j], &refs[j + 1]);
        let read = read_as_it_is(times, plan.window, fine, coarse, &scales);
        let looks = judged(&read);
        let b = looks[0].1;
        let t = looks.last().unwrap();
        println!(
            "T2S2 pair ({:.5}, {:.5}): the check's read as it is: loss {:+.4} ± {:.4} b, T {:+.4} ± {:.4} b",
            taus[j + 1],
            taus[j],
            read.of(Quantity::Loss).iter().map(|r| r.value).fold(f64::NEG_INFINITY, f64::max) / b,
            read.of(Quantity::Loss).iter().map(|r| r.spread.error).fold(0., f64::max) / b,
            t.0.value / t.1,
            t.0.spread.error / t.1
        );
        let line: Vec<String> = at.iter().map(|&k| { let (d, s) = loss_d(fine, coarse, k); format!("{:.0}:{:+.4}±{:.4}", times[k], d, s) }).collect();
        println!("T2S2 pair ({:.5}, {:.5}) loss Q(2τ) − Q(τ): {}", taus[j + 1], taus[j], line.join(" "));
        windowed(times, fine, coarse, &scales, taus[j + 1], taus[j]);
    }
    let finest = &refs[levels - 1];
    for j in 0..levels - 1 {
        let line: Vec<String> = at.iter().map(|&k| { let (d, s) = loss_d(finest, &refs[j], k); format!("{:.0}:{:+.4}±{:.4}", times[k], d, s) }).collect();
        println!("T2S2 against {:.5}: {:.5} loss Q − Q_finest: {}", taus[levels - 1], taus[j], line.join(" "));
    }
    // STUDY B (T2S2_COMPARE): every level's readings against the finest's at the read times nearest to the multiples of
    // T2S2_CMP_EVERY (µs): the loss, T per lab axis and their mean (the total T), the rows of the populated states;
    // each level's value with its jackknife error, the difference with the paired jackknife (the same groups), and the
    // targets -- 0.05 for the loss and the rows, T's allowance √((0.1 T)² + T_sc²) at the finest level's T.
    if std::env::var("T2S2_COMPARE").is_ok() {
        let every: f64 = var("T2S2_CMP_EVERY", 100.);
        let none: Vec<&Group> = Vec::new();
        let mut picks: Vec<usize> = Vec::new();
        let mut t = every;
        while t <= end + 1e-9 {
            if let Some(k) = (1..times.len()).min_by(|&a, &b| (times[a] - t).abs().total_cmp(&(times[b] - t).abs())) {
                picks.push(k);
            }
            t += every;
        }
        let tx = 1 + crate::request::STATES;
        let total = |r: &[crate::analysis::Reading]| crate::analysis::Reading {
            whole: (r[tx].whole + r[tx + 1].whole + r[tx + 2].whole) / 3.,
            leave: (0..r[tx].leave.len()).map(|i| (r[tx].leave[i] + r[tx + 1].leave[i] + r[tx + 2].leave[i]) / 3.).collect(),
        };
        let minus = |a: &crate::analysis::Reading, b: &crate::analysis::Reading| crate::analysis::Reading { whole: a.whole - b.whole, leave: a.leave.iter().zip(&b.leave).map(|(x, y)| x - y).collect() };
        let fin = reads_of(finest);
        let scattered = fin.iter().fold([0.; 2], |a, g| [a[0] + g.scattered[0], a[1] + g.scattered[1]]);
        let scale = crate::estimate::semiclassical(scattered, &scales);
        for &k in &picks {
            let rf = crate::estimate::readings(&fin, &none, k, &scales);
            let tf = total(&rf);
            let target_t = crate::estimate::allowance(tf.whole, scale) * tf.whole;
            let shown: Vec<usize> = (0..crate::request::STATES).filter(|&s| rf[1 + s].whole > 0.02).collect();
            let (lf, lfe) = with_error(&rf[0]);
            let (tfw, tfe) = with_error(&tf);
            println!(
                "T2S2 CMP t {:.1}: finest {:.5}: L {:.4} ± {:.4}, T x/y/z {:.2}/{:.2}/{:.2} total {:.2} ± {:.2} µK (target ± {:.2} µK, T_sc {:.2}); rows {}",
                times[k],
                taus[levels - 1],
                lf,
                lfe,
                rf[tx].whole,
                rf[tx + 1].whole,
                rf[tx + 2].whole,
                tfw,
                tfe,
                target_t,
                scale,
                shown.iter().map(|&s| { let (v, e) = with_error(&rf[1 + s]); format!("{} {v:.4} ± {e:.4}", request.states[s]) }).collect::<Vec<_>>().join(", ")
            );
            for j in 0..levels - 1 {
                let rj = crate::estimate::readings(&reads_of(&refs[j]), &none, k, &scales);
                let (dl, dle) = with_error(&minus(&rj[0], &rf[0]));
                let axes: Vec<String> = (0..3).map(|a| { let (d, e) = with_error(&minus(&rj[tx + a], &rf[tx + a])); format!("{d:+.2}±{e:.2}") }).collect();
                let (dt, dte) = with_error(&minus(&total(&rj), &tf));
                let rows: Vec<String> = shown.iter().map(|&s| { let (d, e) = with_error(&minus(&rj[1 + s], &rf[1 + s])); format!("{} {d:+.4}±{e:.4}", request.states[s]) }).collect();
                println!("T2S2 CMP t {:.1}: {:.5} − finest: L {dl:+.4}±{dle:.4}, T x/y/z {} total {dt:+.2}±{dte:.2} µK; rows {}", times[k], taus[j], axes.join("/"), rows.join(", "));
            }
        }
    }
}
