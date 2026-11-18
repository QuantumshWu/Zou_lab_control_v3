//! The onefold integrator's friction on a dragged atom (BRIEF_FOLDAPI_B item 1; dragchk.status.md): a heavy atom pulled
//! at a fixed velocity along z through tilt0's light, stepped by `step::Midway`, its mean force read off its momentum's
//! change over a window -- at two steps, to see whether the friction has a first-order error in τ (V3's frozen middle
//! had one: s(τ) ≈ 22 + 242τ, v3i.status.md).
use super::*;
use crate::step::{Brownian, Domain, Dynamics, Midway, Scratch, Trajectory};
use rayon::prelude::*;

/// The time bins of the F = 1 population's relaxation from the start (`a_dragged_atom_s_mean_force_on_the_onefold_step`).
const BINS: usize = 40;
/// A trajectory's row: [force z, kick z, jumps z, flow z, jumps, F = 1 population sum, step ends, time], the ground
/// populations, the readout rows, the jumps by F (4), the explicit steps, the bins (2 each), then the landings' ground
/// populations by the source F (2 × GROUND, `LANDED`) and the F = 2 sets' coherences ψ_aψ_b* (`PAIRS`, re and im).
const LANDED: usize = 8 + GROUND + STATES + 5 + 2 * BINS;
const COHERENT: usize = LANDED + 2 * GROUND;
const PAIRS: [(usize, usize); 4] = [(3, 5), (5, 7), (3, 7), (4, 6)];
const PHOTONS: usize = COHERENT + 2 * PAIRS.len();
const ROW: usize = PHOTONS + 1;

struct Anywhere;
impl Domain for Anywhere {
    fn runaway(&self, _: f64) -> bool {
        false
    }
}

/// An f64 from the environment, or its default.
fn env(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().map(|s| s.parse().unwrap()).unwrap_or(default)
}

/// A list of f64 from the environment (comma-separated), or its default.
fn list(key: &str, default: &[f64]) -> Vec<f64> {
    std::env::var(key).ok().map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect()).unwrap_or_else(|| default.to_vec())
}

/// Diagnostic (DRAG_REQ a native request's path, tilt0's): the mass times DRAG_MASS (1), the velocity held by setting the
/// momentum back after every step where DRAG_HOLD (1; else the mass's alone holds it), the trap beam's waist DRAG_WAIST
/// (µm; 1e5: its light the focus's along the drag, as obe's master equation takes it; 0 the request's own), the starts'
/// spread over the period times DRAG_SPREAD (1); per step τ of DRAG_TAUS and velocity v of DRAG_VS (µm/µs = m/s) along z, DRAG_COUNT
/// trajectories from the request's chief ground state at z spread over the pattern's period, decisions at the ends of
/// cells of DRAG_CELL µs (the same times at every τ), no Brownian draw (`Brownian::STILL`: the diffusion's mean is none);
/// over DRAG_WINDOW µs after DRAG_BURN µs: the mean force (κ's change over the time, ħ/(µm·µs)) with its standard error,
/// its parts (the kick, the jumps, the flow), the jumps' rate and the F = 1 population at the step ends.
#[test]
#[ignore]
fn a_dragged_atom_s_mean_force_on_the_onefold_step() {
    let path = std::env::var("DRAG_REQ").expect("DRAG_REQ");
    let mut req = crate::model::testing::scene(&path);
    zeroed(&mut req);
    full_h0(&mut req);
    req.mass_kg *= env("DRAG_MASS", 1.0);
    let waist = env("DRAG_WAIST", 1e5);
    if waist > 0.0 {
        let t = req.trap_beam;
        req.beams[t].waist = Some(waist);
    }
    let taus = list("DRAG_TAUS", &[0.3256, 0.0814]);
    let vs = list("DRAG_VS", &[-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]);
    let count = env("DRAG_COUNT", 64.0) as u64;
    let (burn, window, cell) = (env("DRAG_BURN", 40.0), env("DRAG_WINDOW", 100.0), env("DRAG_CELL", 0.6512));
    let seed = env("DRAG_SEED", 20261117.0) as u64;
    // Held: the momentum set back to the drag's after every step (an outside force that holds the velocity), the force
    // the steps' momentum changes; at the request's own mass the atom's mechanical energy -- the fold's readings' unit
    // (run.rs `mechanical`) -- is a moving atom's, not a heavy one's (at 10⁴ times the mass and 0.1 m/s it is 6.9·10⁴
    // rad/µs, and the fold read within its share what it should not).
    let hold = env("DRAG_HOLD", 1.0) > 0.0;
    // Pinned: the position set back to the drag's path (z₀ + vt) after every step too, so that a held atom at a trap's
    // focus stays there (unpinned, the light force walks it off until the trap's pull answers it) -- and the atom is
    // the master equation's, infinitely heavy at exactly v: unpinned, the held atom moves between the momentum's resets
    // with what the step gave it (±2 mm/s while a stretched state cycles, ±6 mm/s after a photon) and keeps the
    // displacement; on tilt0 that moved F(0) by +4.2 % ± 0.8 % at every τ from 0.16 to 0.65 µs, at 100 times the mass by
    // −0.1 % (readings.status.md R2). The comparison with obe_fix pins.
    let pin = env("DRAG_PIN", 0.0) > 0.0;
    let initial = crate::solve::components(&req);
    let chief = initial.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    let reference: [f64; GROUND] = std::array::from_fn(|s| if s == chief { 1.0 } else { 0.0 });
    let run = the_run(&req, reference);
    // DRAG_BUDGET: the run's evaluation within this times the share (its representation the start's, decided above).
    super::super::proto::BUDGET_SCALE.store(env("DRAG_BUDGET", 1.0).to_bits(), std::sync::atomic::Ordering::Relaxed);
    let hm = run.hbar_over_mass();
    let labels = req.levels().unwrap();
    // The σ⁺σ⁻ pattern's period along z: the two counter-propagating beams' wave numbers.
    let kz: f64 = req.beams.iter().filter(|b| b.direction[2].abs() > 0.5).map(|b| b.wave_number).sum();
    let period = 2.0 * std::f64::consts::PI / kz;
    println!(
        "drag: mass x{}, held {hold}, trap waist {:?}, chief {chief}, period {period:.5} µm, count {count}, burn {burn}, window {window}, cell {cell}, ATOM_PROTO_DECIDE {:?}, pinned {pin}",
        env("DRAG_MASS", 1.0),
        req.beams[req.trap_beam].waist,
        std::env::var("ATOM_PROTO_DECIDE").unwrap_or_default()
    );
    for &tau in &taus {
        for &v in &vs {
            let clock = std::time::Instant::now();
            // Per trajectory: [force z, kick z, jumps z, flow z, jumps, F = 1 population sum, step ends, time].
            let rows: Vec<[f64; ROW]> = (0..count)
                .into_par_iter()
                .map(|i| {
                    let z0 = env("DRAG_SPREAD", 1.0) * period * (i as f64 + 0.5) / count as f64;
                    let mut t = Trajectory::new(&run, chief, [0.0, 0.0, z0], [0.0, 0.0, v / hm], seed, i);
                    let mut scratch = Scratch::default();
                    let mut row = [0.0; ROW];
                    let readout = env("DRAG_READOUT", 0.0) > 0.0;
                    let (mut k0, mut t0, mut j0, mut on) = (0.0, 0.0, 0u32, false);
                    // The momentum the steps made (held: summed before each reset).
                    let mut made = 0.0;
                    while t.time < burn + window - 1e-9 {
                        let from = t.time;
                        let crosses = ((from + tau) / cell + 1e-9).floor() > (from / cell + 1e-9).floor();
                        let step = Midway::new(&run, &t, tau, crosses.then_some(cell)).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Anywhere)).expect("a step");
                        let m = step.momentum();
                        // The jumps' transfers between F (the made states after each jump, from the step's start cluster).
                        if on {
                            let fof = |c: usize| run.cluster(c).states.iter().filter(|&&x| x < GROUND).map(|&x| labels[x].f).next().unwrap();
                            let mut from = fof(t.cluster);
                            for made in step.made.iter().filter(|x| x.kick.is_some()) {
                                let to = fof(made.state.cluster);
                                row[8 + GROUND + STATES + 2 * (from as usize - 1) + (to as usize - 1)] += 1.0;
                                // The landed state's ground populations, by the source F.
                                let lc = run.cluster(made.state.cluster);
                                let ln: f64 = made.state.psi.iter().map(|z| z.norm_sqr()).sum();
                                for (&st, z) in lc.states.iter().zip(made.state.psi.iter()) {
                                    if st < GROUND {
                                        row[LANDED + GROUND * (from as usize - 1) + st] += z.norm_sqr() / ln;
                                    }
                                }
                                from = to;
                            }
                        }
                        t = step.end;
                        if on && readout {
                            let rows = run.readout(t.cluster, &t.point(), &t.psi, &t.held);
                            for (k, x) in rows.iter().enumerate() {
                                row[8 + GROUND + k] += x;
                            }
                        }
                        if on {
                            made += m.whole[2];
                            row[1] += m.kick[2];
                            row[2] += m.jumps[2];
                            row[3] += m.flow[2];
                            let c = run.cluster(t.cluster);
                            let norm: f64 = t.psi.iter().map(|z| z.norm_sqr()).sum();
                            let f1: f64 = c.states.iter().zip(t.psi.iter()).filter(|(&s, _)| s < GROUND && labels[s].f == 1).map(|(_, z)| z.norm_sqr()).sum();
                            row[5] += f1 / norm;
                            row[6] += 1.0;
                            if c.states.iter().any(|&s| s >= GROUND) {
                                row[8 + GROUND + STATES + 4] += 1.0;
                            }
                            for (&st, z) in c.states.iter().zip(t.psi.iter()) {
                                if st < GROUND {
                                    row[8 + st] += z.norm_sqr() / norm;
                                }
                            }
                            for (k, &(a, b)) in PAIRS.iter().enumerate() {
                                if let (Some(i), Some(j)) = (c.states.iter().position(|&x| x == a), c.states.iter().position(|&x| x == b)) {
                                    let z = t.psi[i] * t.psi[j].conj() / norm;
                                    row[COHERENT + 2 * k] += z.re;
                                    row[COHERENT + 2 * k + 1] += z.im;
                                }
                            }
                        }
                        // The F = 1 population over time from the start (the burn's relaxation), in BINS bins.
                        {
                            let c = run.cluster(t.cluster);
                            let norm: f64 = t.psi.iter().map(|z| z.norm_sqr()).sum();
                            let f1: f64 = c.states.iter().zip(t.psi.iter()).filter(|(&s, _)| s < GROUND && labels[s].f == 1).map(|(_, z)| z.norm_sqr()).sum();
                            let b = ((t.time / (burn + window) * BINS as f64) as usize).min(BINS - 1);
                            row[8 + GROUND + STATES + 5 + 2 * b] += f1 / norm;
                            row[8 + GROUND + STATES + 5 + 2 * b + 1] += 1.0;
                        }
                        if hold {
                            t.kappa = [0.0, 0.0, v / hm];
                        }
                        if pin {
                            t.r = [0.0, 0.0, z0 + v * t.time];
                        }
                        if !on && t.time >= burn - 1e-9 {
                            (k0, t0, j0, on) = (t.kappa[2], t.time, t.jumps, true);
                            row[PHOTONS] = -t.scattered[0];
                        }
                    }
                    let span = t.time - t0;
                    row[0] = if hold { made / span } else { (t.kappa[2] - k0) / span };
                    row[1] /= span;
                    row[2] /= span;
                    row[3] /= span;
                    row[4] = (t.jumps - j0) as f64 / span;
                    row[PHOTONS] = (row[PHOTONS] + t.scattered[0]) / span;
                    row[7] = span;
                    row
                })
                .collect();
            let n = rows.len() as f64;
            let mean = |k: usize| rows.iter().map(|r| r[k]).sum::<f64>() / n;
            let se = |k: usize| {
                let m = mean(k);
                (rows.iter().map(|r| (r[k] - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
            };
            let f1 = rows.iter().map(|r| r[5]).sum::<f64>() / rows.iter().map(|r| r[6]).sum::<f64>();
            let ends = rows.iter().map(|r| r[6]).sum::<f64>();
            let pops: Vec<String> = (0..GROUND).map(|g| format!("{}{:+}:{:.4}", labels[g].f, labels[g].m, rows.iter().map(|r| r[8 + g]).sum::<f64>() / ends)).collect();
            println!("  ground populations at the step ends (F m: p): {}", pops.join(" "));
            let span: f64 = rows.iter().map(|r| r[7]).sum();
            let jumps: Vec<f64> = (0..4).map(|k| rows.iter().map(|r| r[8 + GROUND + STATES + k]).sum::<f64>() / span).collect();
            println!("  jumps per us by F: 1->1 {:.5} 1->2 {:.5} 2->1 {:.5} 2->2 {:.5}", jumps[0], jumps[1], jumps[2], jumps[3]);
            println!("  photons per us (every photon, a landing's at-once ones too): {:.5}; jumps per us (landings) {:.5}", mean(PHOTONS), mean(4));
            println!("  step ends on a cluster with explicit excited states: {:.5}", rows.iter().map(|r| r[8 + GROUND + STATES + 4]).sum::<f64>() / ends);
            let bins: Vec<String> = (0..BINS)
                .map(|b| {
                    let (x, c) = rows.iter().fold((0.0, 0.0), |a, r| (a.0 + r[8 + GROUND + STATES + 5 + 2 * b], a.1 + r[8 + GROUND + STATES + 5 + 2 * b + 1]));
                    format!("{:.0}:{:.4}", (b as f64 + 0.5) * (burn + window) / BINS as f64, x / c)
                })
                .collect();
            println!("  F = 1 population by time from the start (µs: p): {}", bins.join(" "));
            for from in 0..2 {
                let land: Vec<String> = (0..GROUND).map(|g| format!("{}{:+}:{:.5}", labels[g].f, labels[g].m, rows.iter().map(|r| r[LANDED + GROUND * from + g]).sum::<f64>() / span)).collect();
                println!("  landings per us from F={} by landed ground state: {}", from + 1, land.join(" "));
            }
            let coh: Vec<String> = PAIRS
                .iter()
                .enumerate()
                .map(|(k, &(a, b))| {
                    let (re, im) = (rows.iter().map(|r| r[COHERENT + 2 * k]).sum::<f64>() / ends, rows.iter().map(|r| r[COHERENT + 2 * k + 1]).sum::<f64>() / ends);
                    format!("{}{:+}|{}{:+}: {re:+.5}{im:+.5}i", labels[a].f, labels[a].m, labels[b].f, labels[b].m)
                })
                .collect();
            println!("  F = 2 coherences at the step ends: {}", coh.join("  "));
            if rows.iter().any(|r| r[8 + GROUND] != 0.0 || r[8 + GROUND + 3] != 0.0) {
                let read = |k: usize| rows.iter().map(|r| r[8 + GROUND + k]).sum::<f64>() / ends;
                let g: Vec<String> = (0..GROUND).map(|k| format!("{}{:+}:{:.4}", labels[k].f, labels[k].m, read(k))).collect();
                let mut ex = std::collections::BTreeMap::new();
                for k in GROUND..STATES {
                    *ex.entry(format!("{:?}F'{}", labels[k].manifold, labels[k].f)).or_insert(0.0) += read(k);
                }
                println!("  readout rows: ground {} | excited {:?}", g.join(" "), ex);
            }
            println!(
                "tau {tau:.4} v {v:+.3}: F {:+.4} ± {:.4} | kick {:+.4} ± {:.4} jumps {:+.4} ± {:.4} flow {:+.2e} | jumps/µs {:.4} | F=1 {:.4} | 2kvτ {:.3} | {:.1} s",
                mean(0),
                se(0),
                mean(1),
                se(1),
                mean(2),
                se(2),
                mean(3),
                mean(4),
                f1,
                kz * v * tau,
                clock.elapsed().as_secs_f64()
            );
        }
    }
}

/// The zeroed lines (DRAG_ZERO, comma-separated F′ of D2): every beam's couplings into those excited levels set to zero --
/// the attribution of a transfer rate to its F′ by deleting it.
fn zeroed(req: &mut crate::request::Request) {
    let Ok(list) = std::env::var("DRAG_ZERO") else { return };
    let fs: Vec<u8> = list.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect();
    let labels = req.levels().unwrap();
    for b in req.beams.iter_mut() {
        for (e, l) in labels.iter().enumerate() {
            if l.manifold == crate::request::Manifold::D2 && fs.contains(&l.f) {
                for g in 0..GROUND {
                    b.coupling[(e, g)] = C64::ZERO;
                    b.counter_coupling[(e, g)] = C64::ZERO;
                }
            }
        }
    }
}

/// Diagnostic (BRIEF dragchk third item): the engine's photon classes of each ground basis state of tilt0 (DRAG_REQ) at the
/// real tweezer's focus at rest, its own set's cluster, a new atom's representation -- per class its rate and the set it
/// lands in; summed, the state's rate into F = 1 and into F = 2 (its depumping or repumping and its cycling). DRAG_ZERO
/// deletes D2 lines.
#[test]
#[ignore]
fn tilt0_s_transfer_rates_by_ground_state_at_the_focus() {
    use crate::step::Middle as _;
    let path = std::env::var("DRAG_REQ").expect("DRAG_REQ");
    let mut req = crate::model::testing::scene(&path);
    zeroed(&mut req);
    full_h0(&mut req);
    let initial = crate::solve::components(&req);
    let chief = initial.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    let reference: [f64; GROUND] = std::array::from_fn(|s| if s == chief { 1.0 } else { 0.0 });
    let run = the_run(&req, reference);
    let labels = req.levels().unwrap();
    println!("zeroed D2 F' {:?}", std::env::var("DRAG_ZERO").unwrap_or_default());
    for s in 0..GROUND {
        let cluster = run.cluster_of(s);
        let states = run.cluster(cluster).states.clone();
        let mut psi = vec![C64::ZERO; states.len()];
        psi[states.iter().position(|&x| x == s).unwrap()] = C64::ONE;
        let middle = run.middle(cluster, [0.0; 3], 0.0, &crate::step::Held::default(), false);
        let p = middle.path([0.0; 3], [0.0; 3]);
        let mut rates: Vec<f64> = vec![];
        let _ = middle.emitted(&p, 0.0, &psi, &mut |w: &[f64]| {
            rates = w.to_vec();
            None
        });
        let (mut to1, mut to2) = (0.0, 0.0);
        let mut parts = vec![];
        for k in 0..rates.len() {
            if rates[k] <= 0.0 {
                continue;
            }
            let e = middle.emitted(&p, 0.0, &psi, &mut |_: &[f64]| Some(k)).expect("the class");
            let land = run.cluster(e.landing).states.iter().filter(|&&x| x < GROUND).map(|&x| labels[x].f).next().unwrap();
            if land == 1 { to1 += rates[k] } else { to2 += rates[k] }
            let ms: Vec<i8> = run.cluster(e.landing).states.iter().filter(|&&x| x < GROUND).map(|&x| labels[x].m).collect();
            parts.push(format!("{:.5}->F{land}{ms:?}", rates[k]));
        }
        println!("F={} m={:+}: cluster states {:?}; to F=1 {:.6} /us, to F=2 {:.6} /us; classes {}", labels[s].f, labels[s].m, states, to1, to2, parts.join(" "));
        if env("DRAG_READOUT", 0.0) > 0.0 {
            let pt = crate::model::Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 };
            let rows = run.readout(cluster, &pt, &psi, &crate::step::Held::default());
            let ex: Vec<String> = (GROUND..STATES).filter(|&x| rows[x] > 1e-7).map(|x| format!("{:?}{}{:+}:{:.6e}", labels[x].manifold, labels[x].f, labels[x].m, rows[x])).collect();
            let gr: Vec<String> = (0..GROUND).filter(|&x| rows[x] > 1e-9).map(|x| format!("{}{:+}:{:.6}", labels[x].f, labels[x].m, rows[x])).collect();
            println!("  readout ground {} | excited {}", gr.join(" "), ex.join(" "));
        }
    }
}

/// Diagnostic (dragchk section 4): tilt0's (DRAG_REQ) run choice (`Choice::base`) per ground set -- every block's offset,
/// sources, rows (excited levels solved or centred, copies decomposed or centred) and omitted copies -- and the readings at
/// the focus (omission, centring, order, total) beside the share: which copy or level each F = 2 block leaves out.
#[test]
#[ignore]
fn tilt0_s_choice_and_readings_at_the_focus() {
    let path = std::env::var("DRAG_REQ").expect("DRAG_REQ");
    let mut req = crate::model::testing::scene(&path);
    full_h0(&mut req);
    let labels = req.levels().unwrap();
    let name = |s: usize| format!("{:?}{}{:+}", labels[s].manifold, labels[s].f, labels[s].m);
    let share = crate::model::elim::share();
    let choice = the_choice(&req);
    println!("share {share:.5}");
    for (key, st) in &choice.copies {
        if *st != Status::Decomposed {
            println!("copy key (absorbed {:?}, source {}, emitting {:?}, F={}) -> {:?}", key.0, name(key.1), key.2, key.3, st);
        }
    }
    for (key, st) in &choice.excited {
        println!("excited key (absorbed {:?}, source {}, level {}) -> {:?}", key.0, name(key.1), key.2, st);
    }
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).expect("the fold");
        println!("== set {:?}", set.iter().map(|&s| name(s)).collect::<Vec<_>>());
        for (c, comp) in fold.components.iter().enumerate() {
            println!("  component {c}: {:?} nu {:.3} beams {:?}", comp.part, comp.nu, comp.beams.iter().map(|&l| req.beams[l].id.clone()).collect::<Vec<_>>());
        }
        let lvl = |level: usize| (0..EXCITED).find(|&x| fold.excited_level[x] == level).map(|x| format!("{:?}F'{}", labels[GROUND + x].manifold, labels[GROUND + x].f)).unwrap_or_default();
        for (o, b) in fold.blocks.iter().enumerate() {
            let rows: Vec<String> = b
                .rows
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let tag = if i < b.solved { "" } else { "~" };
                    match *r {
                        Row::Excited(x) => format!("{tag}{}", name(GROUND + x)),
                        Row::Copy { state, component } => format!("{tag}copy({},c{component})", name(state)),
                    }
                })
                .collect();
            let om: Vec<String> = b.omitted.iter().map(|&(t, g)| format!("{}c{g}", name(t))).collect();
            println!(
                "  block {o}: value {:.3} sources {:?} explicit {:?}\n    rows {}\n    omitted {}",
                b.value,
                b.sources.iter().map(|&(p, c)| format!("{}<-c{c}", name(fold.ground[p]))).collect::<Vec<_>>(),
                b.explicit,
                rows.join(" "),
                om.join(" ")
            );
        }
        let Some(f) = fold.read_at([0.0; 3]) else {
            println!("  (no fold at the focus)");
            continue;
        };
        // The static generator G at the focus (normalized coordinates, the set's frames) and the metric M.
        let show = |m: &Mat| (0..m.rows).map(|i| (0..m.cols).map(|j| format!("{:+.5}{:+.5}i", m[(i, j)].re, m[(i, j)].im)).collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join(" ; ");
        println!("  generator G: {}", show(&f.generator));
        println!("  metric M: {}", show(&f.metric));
        println!("  frames {:?}", fold.frame_values);
        for r in &f.readings {
            let node = match r.node {
                Node::Excited(level) => format!("excited {}", lvl(level)),
                Node::Copy { level, component } => format!("copy F={level} c{component}"),
            };
            println!(
                "  reading block {} {node}: omission {:.3e} (rate {:.3e}) centring {:.3e} (rate {:.3e}) order {:.2e} total {:.2e}",
                r.block,
                r.omission,
                2.0 * r.omission.sqrt() + r.omission,
                r.centring,
                2.0 * r.centring.sqrt() + r.centring,
                r.order,
                r.total()
            );
        }
    }
}

/// The run of the diagnostics: the base choice (`Onefold::new`'s), where DRAG_KEEP_FREQ (comma-separated frequency indices
/// of the request's light) names frequencies, every copy those frequencies' components emit into that the base choice
/// leaves out (Omitted) centred instead -- in the near blocks (a far block holds no copy) -- a measurement of what the omission costs, on the test's own run (the production
/// choice unchanged).
fn the_run(req: &crate::request::Request, reference: [f64; GROUND]) -> run::Onefold {
    // DRAG_EXPLICIT ("level,frequency,ground;..."): the representation those explicit nodes give (co-rotating components),
    // no decision -- a measurement of a representation by hand.
    if let Some(keys) = explicit_keys() {
        println!("DRAG_EXPLICIT {keys:?}");
        return run::Onefold::of(req, &Choice::base(req, keys, vec![]).expect("the choice"), reference).expect("the run");
    }
    if std::env::var("DRAG_KEEP_FREQ").is_err() {
        // The production run: the start's decision (decide.rs) on the base choice.
        return run::Onefold::new(req, reference).expect("the run");
    }
    run::Onefold::of(req, &the_choice(req), reference).expect("the run")
}

/// DRAG_EXPLICIT's keys ("level,frequency,ground;...", co-rotating components), if set.
fn explicit_keys() -> Option<Vec<(usize, crate::model::Component, usize)>> {
    let list = std::env::var("DRAG_EXPLICIT").ok()?;
    Some(
        list.split(';')
            .filter(|x| !x.trim().is_empty())
            .map(|x| {
                let v: Vec<usize> = x.split(',').map(|y| y.trim().parse().unwrap()).collect();
                (v[0], crate::model::Component { frequency: v[1], counter: false }, v[2])
            })
            .collect(),
    )
}

/// The diagnostics' choice: the base choice, with DRAG_KEEP_FREQ's omitted copies of the near blocks centred (`the_run`).
fn the_choice(req: &crate::request::Request) -> Choice {
    let mut choice = Choice::base(req, vec![], vec![]).expect("the choice");
    if let Ok(list) = std::env::var("DRAG_KEEP_FREQ") {
        let fs: Vec<usize> = list.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect();
        let mut keys = vec![];
        for set in Fold::sets(req, &choice).unwrap() {
            let fold = Fold::new(req, &choice, set[0]).expect("the fold");
            for b in fold.blocks.iter().filter(|b| !b.far()) {
                for &(t, g) in b.omitted.iter().filter(|&&(_, g)| fs.contains(&fold.components[g].part.frequency)) {
                    for &(p, a) in &b.sources {
                        let key = (fold.components[a].part, fold.ground[p], fold.components[g].part, fold.ground_level[t]);
                        if !keys.contains(&key) {
                            keys.push(key);
                        }
                    }
                }
            }
        }
        println!("DRAG_KEEP_FREQ {fs:?}: {} omitted copy keys of near blocks centred", keys.len());
        for key in keys {
            choice.copies.push((key, Status::Centred));
        }
    }
    choice
}

/// DRAG_H0_FULL (1): the request's H0 given every outside form at the focus whole, its off-diagonal too (the request holds
/// its diagonal alone, request.rs `hamiltonian`; the engine takes H0 − O(0) + O(r), so that with it the light's
/// Hamiltonian at r is the bare levels' plus O(r) whole) -- on the test's request, a measurement of what the
/// off-diagonal at the focus does (dragchk.status.md section 4 (b)); the production request and engine unchanged.
fn full_h0(req: &mut crate::request::Request) {
    if env("DRAG_H0_FULL", 0.0) <= 0.0 {
        return;
    }
    let light = crate::field::Light::new(req);
    let focus = light.amplitudes([0.0; 3]);
    let mut added = 0.0_f64;
    for f in &light.frequencies {
        let e0 = f.field(&focus).value;
        let form = crate::model::far::outside_form(&req.outside_forms[f.form].matrices);
        for p in 0..3 {
            for q in 0..3 {
                let w = e0[p].conj() * e0[q];
                if w == C64::ZERO {
                    continue;
                }
                for i in 0..STATES {
                    for j in 0..STATES {
                        if i != j {
                            let x = w * form[p][q][(i, j)];
                            req.hamiltonian[(i, j)] += x;
                            added = added.max(x.abs());
                        }
                    }
                }
            }
        }
    }
    println!("DRAG_H0_FULL: the outside forms' off-diagonal at the focus put into H0 (largest entry added {added:.4} rad/µs)");
}

/// Diagnostic (DRAG_REQ a request's path): the start's decided representation (`Onefold::new`'s choice) set by set -- the
/// ground and explicit places with their frames (rad/µs), the drives that are tones, and each block's offset, sources,
/// explicit places and rows.
#[test]
#[ignore]
fn the_decided_folds_laid_out() {
    let path = std::env::var("DRAG_REQ").expect("DRAG_REQ");
    let req = crate::model::testing::scene(&path);
    let labels = req.levels().unwrap();
    let name = |s: usize| format!("{:?}{}{:+}", labels[s].manifold, labels[s].f, labels[s].m);
    let choice = match explicit_keys() {
        Some(keys) => Choice::base(&req, keys, vec![]).expect("the choice"),
        None => {
            let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).expect("the run");
            run.choice().clone()
        }
    };
    println!("explicit keys {:?}", choice.explicit);
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).expect("the fold");
        let places: Vec<String> = (0..fold.size()).map(|p| format!("{}@{:.3}", name(if p < fold.ground.len() { fold.ground[p] } else { GROUND + fold.explicit[p - fold.ground.len()] }), fold.frame_values[p])).collect();
        println!("== set {:?}: places {}", set.iter().map(|&s| name(s)).collect::<Vec<_>>(), places.join(" "));
        println!("  driven (state, offset): {:?}", fold.driven.iter().map(|(x, o)| format!("{}@{:.3}", name(GROUND + x), fold.blocks.iter().find(|b| b.offset == *o).map_or(f64::NAN, |b| b.value))).collect::<Vec<_>>());
        for (o, b) in fold.blocks.iter().enumerate().filter(|(_, b)| !b.far()) {
            let rows: Vec<String> = b
                .rows
                .iter()
                .map(|r| match *r {
                    Row::Excited(x) => name(GROUND + x),
                    Row::Copy { state, component } => format!("copy({},c{component})", name(state)),
                })
                .collect();
            println!(
                "  block {o}: value {:.3} sources {:?} explicit {:?} rows {}",
                b.value,
                b.sources.iter().map(|&(p, c)| format!("{}<-{:?}", name(fold.ground[p]), fold.components[c].part)).collect::<Vec<_>>(),
                b.explicit.iter().map(|&p| name(GROUND + fold.explicit[p - fold.ground.len()])).collect::<Vec<_>>(),
                rows.join(" ")
            );
        }
    }
}

/// Diagnostic (DRAG_REQ; DRAG_EXPLICIT or the decided representation; DRAG_SET a ground state of the set, 3): the static
/// generator G = S⁻¹AS⁻¹ and the metric M of that set's fold at the focus at rest, place by place (rad/µs; the frames
/// added back on the diagonal), and its tones' G_m largest entries.
#[test]
#[ignore]
fn the_set_s_generator_at_the_focus() {
    let path = std::env::var("DRAG_REQ").expect("DRAG_REQ");
    let req = crate::model::testing::scene(&path);
    let labels = req.levels().unwrap();
    let name = |s: usize| format!("{:?}{}{:+}", labels[s].manifold, labels[s].f, labels[s].m);
    let choice = match explicit_keys() {
        Some(keys) => Choice::base(&req, keys, vec![]).expect("the choice"),
        None => {
            let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).expect("the run");
            run.choice().clone()
        }
    };
    let s0 = env("DRAG_SET", 3.0) as usize;
    let set = Fold::sets(&req, &choice).unwrap().into_iter().find(|s| s.contains(&s0)).unwrap();
    let fold = Fold::new(&req, &choice, set[0]).expect("the fold");
    let mut ws = fold.prototype_workspace().expect("the workspace");
    let mut cache = super::super::proto::Cache::default();
    let step = fold.prototype([0.0; 3], [0.0; 3], fold.thermal, &mut cache, &mut ws).expect("the fold at the focus");
    let n = fold.size();
    let place = |p: usize| name(if p < fold.ground.len() { fold.ground[p] } else { GROUND + fold.explicit[p - fold.ground.len()] });
    println!("set {:?}: {} places", set.iter().map(|&s| name(s)).collect::<Vec<_>>(), n);
    for p in 0..n {
        let row: Vec<String> = (0..n)
            .filter(|&q| step.generator[(p, q)].abs() > 1e-6 || p == q)
            .map(|q| {
                let g = step.generator[(p, q)] + if p == q { C64::real(fold.frame_values[p]) } else { C64::ZERO };
                format!("{}:{:+.4}{:+.4}i", place(q), g.re, g.im)
            })
            .collect();
        println!("  G {} @{:.3}: {}", place(p), fold.frame_values[p], row.join(" "));
    }
    for p in 0..n {
        let row: Vec<String> = (0..n).filter(|&q| (step.metric[(p, q)] - if p == q { C64::ONE } else { C64::ZERO }).abs() > 1e-7).map(|q| format!("{}:{:+.3e}{:+.3e}i", place(q), step.metric[(p, q)].re - if p == q { 1.0 } else { 0.0 }, step.metric[(p, q)].im)).collect();
        println!("  M-1 {}: {}", place(p), row.join(" "));
    }
    for (m, g) in step.tones.iter().enumerate() {
        let big = (0..n * n).map(|x| (x, g[(x / n, x % n)].abs())).fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        println!("  tone {m}: largest {:.4} at {} <- {}", big.1, place(big.0 / n), place(big.0 % n));
    }
}

/// Diagnostic (landport.status.md, the landing's cost): tilt0's (DRAG_REQ) production run, each ground set's cluster at the
/// focus at rest, every ground state's bare landing on its middle (`Middle::landing`: the left response, the transient's
/// modes, the clock and its kick), DRAG_COUNT (200) times each after one to warm: the mean µs a landing.
#[test]
#[ignore]
fn a_landing_s_cost_at_the_focus() {
    use crate::step::Middle as _;
    let path = std::env::var("DRAG_REQ").expect("DRAG_REQ");
    let mut req = crate::model::testing::scene(&path);
    full_h0(&mut req);
    let initial = crate::solve::components(&req);
    let chief = initial.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    let reference: [f64; GROUND] = std::array::from_fn(|s| if s == chief { 1.0 } else { 0.0 });
    let run = the_run(&req, reference);
    let count = env("DRAG_COUNT", 200.0) as usize;
    let mut seen = vec![];
    for s in 0..GROUND {
        let cluster = run.cluster_of(s);
        if seen.contains(&cluster) {
            continue;
        }
        seen.push(cluster);
        let states = run.cluster(cluster).states.clone();
        let middle = run.middle(cluster, [0.0; 3], 0.0, &crate::step::Held::default(), false);
        let p = middle.path([0.0; 3], [0.0; 3]);
        let grounds: Vec<usize> = states.iter().enumerate().filter(|(_, &x)| x < GROUND).map(|(i, _)| i).collect();
        let mut times = vec![];
        for &g in &grounds {
            let f: Vec<C64> = (0..states.len()).map(|i| if i == g { C64::ONE } else { C64::ZERO }).collect();
            let _ = middle.landing(&p, &f, 0.5, 0.0);
            let clock = std::time::Instant::now();
            for _ in 0..count {
                std::hint::black_box(middle.landing(&p, &f, 0.5, 0.0));
            }
            times.push(clock.elapsed().as_secs_f64() / count as f64 * 1e6);
        }
        println!("cluster {cluster} ({} states, {} ground): a landing {:.1?} µs", states.len(), grounds.len(), times);
    }
}
