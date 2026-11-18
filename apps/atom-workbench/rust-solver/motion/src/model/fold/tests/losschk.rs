//! Diagnostics of tilt0's loss (losschk.status.md): an atom held in the tweezer where its intensity is s of the focus's,
//! its velocity and its position held, the mean force, the photons, the F = 1 population and the momentum diffusion.
use super::*;
use crate::step::{Brownian, Domain, Dynamics, Midway, Scratch, Trajectory};
use rayon::prelude::*;

struct Anywhere;
impl Domain for Anywhere {
    fn runaway(&self, _: f64) -> bool {
        false
    }
}

fn env(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().map(|s| s.parse().unwrap()).unwrap_or(default)
}

fn list(key: &str, default: &[f64]) -> Vec<f64> {
    std::env::var(key).ok().map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect()).unwrap_or_else(|| default.to_vec())
}

/// Diagnostic (LOSS_REQ, tilt0's request): per step τ of LOSS_TAUS, trap intensity s of LOSS_S (the atom at y =
/// w√(−ln s / 2) off the tweezer's axis, w its waist: its intensity there s of the focus's) and velocity v of LOSS_VS
/// (m/s, along z), LOSS_COUNT atoms over the σ⁺σ⁻ pattern's period, the velocity and the position held after every step,
/// the Brownian draws of the run (LOSS_BROWN 1; 0: none); LOSS_BURN µs, then over LOSS_WINDOW µs: the mean force per axis
/// (ħ/(µm·µs)) with its standard error, the photons per µs, the F = 1 population (step ends), and the momentum diffusion
/// per axis D_aa = Var(ΔP over a block)/(2 T_block) of each atom's blocks of LOSS_BLOCK µs about its own mean (the force
/// a pinned atom feels its own: no spread over z enters). LOSS_REP: "new" the run's (`Onefold::new`, the start's
/// decision), "base" nothing explicit (`Choice::base`); a step that is refused ends its atom (counted).
#[test]
#[ignore]
fn tilt0_held_at_trap_intensities() {
    let mut req = crate::model::testing::scene(&std::env::var("LOSS_REQ").expect("LOSS_REQ"));
    // LOSS_WAIST: the trap beam's waist widened (its force on a pinned atom off its axis gone; s = 1 only).
    if let Some(w) = std::env::var("LOSS_WAIST").ok().map(|x| x.parse::<f64>().unwrap()) {
        let t = req.trap_beam;
        req.beams[t].waist = Some(w);
    }
    let taus = list("LOSS_TAUS", &[0.3256]);
    let ss = list("LOSS_S", &[1.0, 0.5, 0.2, 0.05, 0.01]);
    let vs = list("LOSS_VS", &[0.0]);
    let count = env("LOSS_COUNT", 64.0) as u64;
    let (burn, window, block, cell) = (env("LOSS_BURN", 40.0), env("LOSS_WINDOW", 100.0), env("LOSS_BLOCK", 10.0), env("LOSS_CELL", 0.6512));
    let brown = env("LOSS_BROWN", 1.0) > 0.0;
    let seed = env("LOSS_SEED", 20261118.0) as u64;
    let initial = crate::solve::components(&req);
    let chief = initial.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    let reference: [f64; GROUND] = std::array::from_fn(|s| if s == chief { 1.0 } else { 0.0 });
    let rep = std::env::var("LOSS_REP").unwrap_or_else(|_| "new".into());
    let run = if rep == "base" { run::Onefold::of(&req, &Choice::base(&req, vec![]).unwrap(), reference).unwrap() } else { run::Onefold::new(&req, reference).unwrap() };
    // LOSS_WAIST set: the trap's intensity scaled to s in the request itself (`scaled`), the atom on the axis, the run on
    // the request's own decision (its explicit nodes).
    let explicit = run.explicit();
    let light = crate::field::Light::new(&req);
    let scaling = std::env::var("LOSS_WAIST").is_ok();
    let hm = run.hbar_over_mass();
    let labels = req.levels().unwrap();
    let waist = req.beams[req.trap_beam].waist.unwrap();
    let kz: f64 = req.beams.iter().filter(|b| b.direction[2].abs() > 0.5).map(|b| b.wave_number).sum();
    let period = 2.0 * std::f64::consts::PI / kz;
    println!("loss: rep {rep}, chief {chief}, waist {waist} µm, period {period:.5} µm, count {count}, burn {burn}, window {window}, block {block}, brownian {brown}");
    for &tau in &taus {
        for &s in &ss {
            let y = if scaling { 0.0 } else { waist * (-(s.ln()) / 2.0).max(0.0).sqrt() };
            let held = scaling.then(|| {
                let r = scaled(&req, &light, s);
                run::Onefold::of(&r, &Choice::base(&r, if rep == "base" { vec![] } else { explicit.clone() }).unwrap(), reference).unwrap()
            });
            let run = held.as_ref().unwrap_or(&run);
            for &v in &vs {
                let clock = std::time::Instant::now();
                // Per atom: [P x, y, z over the window, photons, F = 1 sum, ends, D x, y, z (block variances), failed, time].
                let rows: Vec<[f64; 14]> = (0..count)
                    .into_par_iter()
                    .map(|i| {
                        let z0 = period * (i as f64 + 0.5) / count as f64;
                        let kappa = [0.0, 0.0, v / hm];
                        let mut t = Trajectory::new(run, chief, [0.0, y, z0], kappa, seed, i);
                        let mut scratch = Scratch::default();
                        let mut row = [0.0; 14];
                        let (mut k, mut jumps0) = (0u32, None);
                        let (mut blocks, mut current, mut edge): (Vec<[f64; 3]>, [f64; 3], f64) = (vec![], [0.0; 3], burn + block);
                        while t.time < burn + window - 1e-9 {
                            let from = t.time;
                            let crosses = ((from + tau) / cell + 1e-9).floor() > (from / cell + 1e-9).floor();
                            let mut g = crate::random::Stream::brownian(seed, i, k, 0);
                            let deviates = [g.normals(), g.normals()];
                            let b = if brown { Brownian::over(tau, &deviates) } else { Brownian::STILL };
                            k += 1;
                            let Ok(step) = Midway::new(run, &t, tau, crosses.then_some(cell)).and_then(|m| m.attempt(&t, &b, &[], &mut scratch, &Anywhere)) else {
                                row[9] = 1.0;
                                break;
                            };
                            let m = step.momentum();
                            if jumps0.is_none() && step.end.time > burn + 1e-9 {
                                jumps0 = Some(t.scattered[0]);
                            }
                            t = step.end;
                            if t.time > burn + 1e-9 {
                                for a in 0..3 {
                                    row[a] += m.whole[a];
                                    current[a] += m.whole[a];
                                }
                                row[11] += m.flow[2];
                                row[12] += m.kick[2];
                                row[13] += m.jumps[2];
                                let c = run.cluster(t.cluster);
                                let norm: f64 = t.psi.iter().map(|z| z.norm_sqr()).sum();
                                row[4] += c.states.iter().zip(t.psi.iter()).filter(|(&st, _)| st < GROUND && labels[st].f == 1).map(|(_, z)| z.norm_sqr()).sum::<f64>() / norm;
                                row[5] += 1.0;
                                row[10] = t.time - burn;
                                if t.time >= edge - 1e-9 {
                                    blocks.push(current);
                                    current = [0.0; 3];
                                    edge += block;
                                }
                            }
                            t.kappa = kappa;
                            t.r = [0.0, y, z0 + v * t.time];
                        }
                        row[3] = t.scattered[0] - jumps0.unwrap_or(t.scattered[0]);
                        let nb = blocks.len() as f64;
                        if nb > 1.0 {
                            for a in 0..3 {
                                let mean = blocks.iter().map(|x| x[a]).sum::<f64>() / nb;
                                row[6 + a] = blocks.iter().map(|x| (x[a] - mean).powi(2)).sum::<f64>() / (nb - 1.0) / (2.0 * block);
                            }
                        }
                        row
                    })
                    .collect();
                let ok: Vec<&[f64; 14]> = rows.iter().filter(|r| r[9] == 0.0 && r[10] > 0.0).collect();
                let n = ok.len() as f64;
                let force = |a: usize| -> (f64, f64) {
                    let f: Vec<f64> = ok.iter().map(|r| r[a] / r[10]).collect();
                    let m = f.iter().sum::<f64>() / n;
                    (m, (f.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt())
                };
                let (fx, fy, fz) = (force(0), force(1), force(2));
                let span: f64 = ok.iter().map(|r| r[10]).sum();
                let photons = ok.iter().map(|r| r[3]).sum::<f64>() / span;
                let f1 = ok.iter().map(|r| r[4]).sum::<f64>() / ok.iter().map(|r| r[5]).sum::<f64>();
                let d: Vec<f64> = (0..3).map(|a| ok.iter().map(|r| r[6 + a]).sum::<f64>() / n).collect();
                let part = |k: usize| ok.iter().map(|r| r[k]).sum::<f64>() / span;
                println!("  z parts: flow {:+.4}, kick {:+.4}, jumps {:+.4}", part(11), part(12), part(13));
                println!(
                    "τ {tau} s {s} (y {y:.3} µm) v {v}: F [{:+.4} ± {:.4}, {:+.4} ± {:.4}, {:+.4} ± {:.4}] ħ/(µm·µs); photons {photons:.4}/µs; F=1 {f1:.4}; D [{:.3}, {:.3}, {:.3}] ħ²/(µm²·µs); failed {} of {count}; {:.1} s",
                    fx.0,
                    fx.1,
                    fy.0,
                    fy.1,
                    fz.0,
                    fz.1,
                    d[0],
                    d[1],
                    d[2],
                    rows.iter().filter(|r| r[9] > 0.0).count(),
                    clock.elapsed().as_secs_f64()
                );
            }
        }
    }
}

/// The request with its trap light's intensity s of its own (`run::lit`'s rule: the fields and couplings by √s, the
/// outside shift H0 holds at the focus by s).
fn scaled(req: &Request, light: &crate::field::Light, s: f64) -> Request {
    let mut out = req.clone();
    let f = &light.frequencies[light.trap];
    let focus: [C64; 3] = std::array::from_fn(|q| f.beams.iter().map(|&l| req.beams[l].field[q]).sum());
    for (x, o) in crate::request::coordinates(&focus).iter().zip(&req.outside_forms[f.form].matrices) {
        out.hamiltonian.add_scaled(C64::real(-(1.0 - s) * x), o);
    }
    let a = C64::real(s.sqrt());
    for &l in &f.beams {
        let beam = &mut out.beams[l];
        beam.field = beam.field.map(|z| z * a);
        beam.coupling = beam.coupling.scaled(a);
        beam.counter_coupling = beam.counter_coupling.scaled(a);
    }
    out
}

/// Diagnostic (LOSS_REQ): LOSS_FREE atoms drawn thermal in the reference's harmonic well at the request's temperature
/// (positions and momenta Gaussian), free -- no hold, no pin --, the run's Brownian draws, steps of LOSS_TAUS' first, the
/// decision cells LOSS_CELL, to LOSS_TIME µs; per 100 µs: the fraction outside the trap's region (`Light::outside`, the
/// loss's test's first half) and the retained atoms' mean (ħ/m)κ²/2 per axis and V_ref (rad/µs); per atom that leaves, the
/// time it leaves and the time its (ħ/m)κ²/2 + V_ref first rose above zero, and the trap intensity there. LOSS_REP as
/// above.
#[test]
#[ignore]
fn tilt0_free_atoms_leave_the_trap() {
    let req = crate::model::testing::scene(&std::env::var("LOSS_REQ").expect("LOSS_REQ"));
    let tau = list("LOSS_TAUS", &[0.3256])[0];
    let (count, end, cell) = (env("LOSS_FREE", 64.0) as u64, env("LOSS_TIME", 1000.0), env("LOSS_CELL", 0.6512));
    let seed = env("LOSS_SEED", 20261118.0) as u64;
    let initial = crate::solve::components(&req);
    let chief = initial.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    let reference: [f64; GROUND] = std::array::from_fn(|s| if s == chief { 1.0 } else { 0.0 });
    let rep = std::env::var("LOSS_REP").unwrap_or_else(|_| "new".into());
    let run = if rep == "base" { run::Onefold::of(&req, &Choice::base(&req, vec![]).unwrap(), reference).unwrap() } else { run::Onefold::new(&req, reference).unwrap() };
    let hm = run.hbar_over_mass();
    let at0 = run.reference([0.0; 3]);
    let kt = crate::units::KT_PER_UK * req.temperature_uk;
    let sigma_r: Vec<f64> = (0..3).map(|a| (kt / (hm * at0.hessian[a][a] / hm * hm)).sqrt() * 0.0 + (kt / at0.hessian[a][a]).sqrt()).collect();
    let sigma_k = (kt / hm).sqrt();
    let trap = &run.light().frequencies[run.light().trap];
    let intensity = |r: [f64; 3]| trap.beams.iter().map(|&l| run.light().beams[l].amplitude_to(r, 0).value.norm_sqr()).fold(0.0, f64::max);
    println!("free: rep {rep}, kT {kt:.3} rad/µs, σ_r {sigma_r:?} µm, σ_κ {sigma_k:.3}/µm, τ {tau}, to {end} µs");
    let bins = (end / 100.0).ceil() as usize;
    // Per atom: per bin [outside, kinetic x, y, z, V_ref], and (time out, time E > 0, intensity there).
    let rows: Vec<(Vec<[f64; 5]>, [f64; 3])> = (0..count)
        .into_par_iter()
        .map(|i| {
            let mut g = crate::random::Stream::brownian(seed, i, u32::MAX, 7);
            let (a, b) = (g.normals(), g.normals());
            let r0: [f64; 3] = std::array::from_fn(|x| sigma_r[x] * a[x]);
            let k0: [f64; 3] = std::array::from_fn(|x| sigma_k * b[x]);
            let mut t = Trajectory::new(&run, chief, r0, k0, seed, i);
            let mut scratch = Scratch::default();
            let mut out = vec![[0.0; 5]; bins];
            let mut fate = [f64::NAN; 3];
            let mut k = 0u32;
            while t.time < end - 1e-9 {
                let crosses = ((t.time + tau) / cell + 1e-9).floor() > (t.time / cell + 1e-9).floor();
                let mut gg = crate::random::Stream::brownian(seed, i, k, 0);
                let deviates = [gg.normals(), gg.normals()];
                k += 1;
                let Ok(step) = Midway::new(&run, &t, tau, crosses.then_some(cell)).and_then(|m| m.attempt(&t, &Brownian::over(tau, &deviates), &[], &mut scratch, &Anywhere)) else { break };
                t = step.end;
                let e = 0.5 * hm * t.kappa.iter().map(|x| x * x).sum::<f64>() + run.reference(t.r).value;
                if fate[1].is_nan() && e > 0.0 {
                    fate[1] = t.time;
                    fate[2] = intensity(t.r);
                }
                if run.light().outside(t.r) {
                    fate[0] = t.time;
                    break;
                }
                let bin = ((t.time / 100.0) as usize).min(bins - 1);
                if ((t.time - tau) / 100.0) as usize != bin || k == 1 {
                    for x in 0..3 {
                        out[bin][1 + x] = 0.5 * hm * t.kappa[x] * t.kappa[x];
                    }
                    out[bin][4] = run.reference(t.r).value;
                    out[bin][0] = 1.0;
                }
            }
            (out, fate)
        })
        .collect();
    for bin in 0..bins {
        let alive: Vec<&[f64; 5]> = rows.iter().map(|r| &r.0[bin]).filter(|x| x[0] > 0.0).collect();
        let n = alive.len().max(1) as f64;
        let gone = rows.iter().filter(|r| r.1[0] <= 100.0 * (bin + 1) as f64).count();
        println!(
            "to {} µs: left {gone} of {count}; the rest's (ħ/m)κ²/2 [{:.3}, {:.3}, {:.3}], V_ref {:.3} rad/µs",
            100 * (bin + 1),
            alive.iter().map(|x| x[1]).sum::<f64>() / n,
            alive.iter().map(|x| x[2]).sum::<f64>() / n,
            alive.iter().map(|x| x[3]).sum::<f64>() / n,
            alive.iter().map(|x| x[4]).sum::<f64>() / n
        );
    }
    let mut fates: Vec<[f64; 3]> = rows.iter().map(|r| r.1).filter(|f| !f[0].is_nan()).collect();
    fates.sort_by(|a, b| a[0].total_cmp(&b[0]));
    for f in fates.iter().take(20) {
        println!("  out at {:.1} µs; E > 0 from {:.1} µs at intensity {:.3}", f[0], f[1], f[2]);
    }
}
