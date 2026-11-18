//! The readings against what they read (readings.status.md): each reading of a fold an estimate of what its approximation
//! moves of the fold's results, not a bound.
use super::*;

/// Numbers in e-notation, three digits.
fn e3(x: &[f64]) -> String {
    x.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>().join(" ")
}

/// The two-level atom of landport.status.md (Δ = −4Γ, s = 10, 30, 60; nothing explicit, at the focus at rest): the
/// decision's reading against what the run propagates, G's slow mode -- its rate and its energy against the exact
/// two-level no-jump root [[0, g], [g, −Δ − iΓ/2]] (`moved_modes`' units) -- at least the error, at most three times it.
/// The landing's slow span (H_s, the series' Bloch generator: proto/transient.rs) errs by 2.4–4.8 % on the rate where
/// the reading says 1.2–1.9 % (said, not asserted): H_s's eigenvalue errs at first order in the series' rest, amplified
/// by the shift over the rate (≈ 2|Δ|/Γ), where G's -- the compression's, stationary -- does not; the clock hands the
/// step θ′ = N_s(τ*), so H_s's rate cancels beyond the transient and the run's waiting time is G's (readings.status.md).
#[test]
fn a_two_level_atom_s_reading_reads_its_generator_s_slow_mode() {
    for (delta, s) in [(-4.0 * GAMMA, 10.0), (-4.0 * GAMMA, 30.0), (-4.0 * GAMMA, 60.0)] {
        let req = two_level(delta, s);
        let g = 0.5 * GAMMA * (s / 2.0_f64).sqrt();
        let k = C64::new(-delta, -0.5 * GAMMA);
        let root = (k * k + C64::real(4.0 * g * g)).sqrt();
        let exact = [(k + root) * 0.5, (k - root) * 0.5].into_iter().min_by(|x, y| x.abs().total_cmp(&y.abs())).unwrap();
        let fold = Fold::new(&req, &Choice::base(&req, vec![], vec![]).unwrap(), 0).unwrap();
        let mut ws = fold.prototype_workspace().unwrap();
        let mut cache = super::super::proto::Cache::default();
        let reading = fold.prototype([0.0; 3], [0.0; 3], fold.thermal, &mut cache, &mut ws).expect("the fold").reading;
        let held = fold.prototype_without([0.0; 3], [0.0; 3], fold.thermal, &cache, &mut ws, super::super::proto::Without::Nothing, 0).expect("the held fold");
        let (moved, rate_part, _) = moved_modes(&held, &Mat::from_fn(1, 1, |_, _| exact), fold.thermal);
        let (_, a) = ws.taken(&fold, &cache, &[C64::ONE], &[], false).expect("the landing");
        let span = ws.transient(&fold, &cache, &a).expect("the landing's modes").modes().0[0].im / exact.im - 1.0;
        println!("Δ {:+.0}Γ s {s}: the reading {reading:.3e}; G's slow mode against the exact {moved:.3e} (its rate {rate_part:+.3e}); the landing's slow span's rate {span:+.3e} (H_s, not the run's)", delta / GAMMA);
        assert!(reading >= moved && reading <= 3.0 * moved, "the reading an estimate of G's slow mode's error: {reading:.3e} against {moved:.3e}");
    }
}

/// What a generator `a` differs from `b` by in the readings' unit (rate units, proto.rs # The decisions): the rate
/// channel -- the largest relative change of any state's photon rate, ψ†δΓψ/ψ†Γψ (Γ = i(G − G†), b's), over the states
/// b's Γ does not hold dark -- and the potential channel -- the largest change of any state's energy, ‖δ(G + G†)/2‖,
/// against twice the atom's energy --, summed as the readings sum them: rate(√(unrate(rate)² + unrate(potential)²)).
pub(super) fn moved(a: &Mat, b: &Mat, energy: f64) -> (f64, f64, f64) {
    let n = a.rows;
    let decay = |g: &Mat| Mat::from_fn(n, n, |i, j| C64::I * (g[(i, j)] - g[(j, i)].conj()));
    let hermitian = |g: &Mat| Mat::from_fn(n, n, |i, j| (g[(i, j)] + g[(j, i)].conj()) * 0.5);
    let (da, db) = (decay(a), decay(b));
    let e = crate::linalg::hermitian_eigen(&db);
    let top = e.values.iter().fold(0.0_f64, |m, v| m.max(*v));
    let lit: Vec<usize> = (0..n).filter(|&k| e.values[k] > crate::units::rounding(n).sqrt() * top).collect();
    let mut d = da.clone();
    d.add_scaled(-C64::ONE, &db);
    let vd = e.vectors.adjoint_product(&d).product(&e.vectors);
    let c = Mat::from_fn(lit.len(), lit.len(), |i, j| vd[(lit[i], lit[j])] / (e.values[lit[i]] * e.values[lit[j]]).sqrt());
    let rate_change = if lit.is_empty() { 0.0 } else { crate::linalg::hermitian_eigen(&c).values.iter().fold(0.0_f64, |m, v| m.max(v.abs())) };
    let mut dh = hermitian(a);
    dh.add_scaled(-C64::ONE, &hermitian(b));
    let potential = crate::linalg::hermitian_eigen(&dh).values.iter().fold(0.0_f64, |m, v| m.max(v.abs())) / (2.0 * energy);
    let un = |r: f64| (1.0 + r).sqrt() - 1.0;
    (rate((un(rate_change).powi(2) + un(potential).powi(2)).sqrt()), rate_change, potential)
}

/// What a generator `a` differs from `b` by on its modes (rate units): per eigenvalue of b its nearest of a's, the relative
/// change of its decay rate −2 Im λ (the modes b's rates do not hold dark) and the change of its energy Re λ against twice
/// the atom's energy, summed as the readings sum them; the largest over the modes, and its two channels.
pub(super) fn moved_modes(a: &Mat, b: &Mat, energy: f64) -> (f64, f64, f64) {
    let (Some(ea), Some(eb)) = (crate::linalg::eigen(a), crate::linalg::eigen(b)) else { return (f64::NAN, f64::NAN, f64::NAN) };
    let top = eb.values.iter().fold(0.0_f64, |m, v| m.max(-v.im));
    let un = |r: f64| (1.0 + r).sqrt() - 1.0;
    let mut used = vec![false; ea.values.len()];
    let (mut worst, mut wr, mut wp) = (0.0_f64, 0.0, 0.0);
    let mut order: Vec<usize> = (0..eb.values.len()).collect();
    order.sort_by(|&x, &y| eb.values[y].im.total_cmp(&eb.values[x].im).reverse());
    for k in order {
        let lb = eb.values[k];
        let Some(m) = (0..ea.values.len()).filter(|&m| !used[m]).min_by(|&x, &y| (ea.values[x] - lb).abs().total_cmp(&(ea.values[y] - lb).abs())) else { continue };
        used[m] = true;
        let d = ea.values[m] - lb;
        let r = if -lb.im > crate::units::rounding(a.rows).sqrt() * top { (d.im / lb.im).abs() } else { 0.0 };
        let p = d.re.abs() / (2.0 * energy);
        let total = rate((un(r).powi(2) + un(p).powi(2)).sqrt());
        if total > worst {
            (worst, wr, wp) = (total, r, p);
        }
    }
    (worst, wr, wp)
}

/// A random small system for the readings (`every_reading_is_an_estimate_of_what_it_reads`): one to three ground states
/// (beats to ±25 rad/µs), one to three excited ones (±1.5Γ about the line, each landing on one of the ground states), one
/// or two laser frequencies each of one or two beams (along ±z or x: standing waves, Doppler shifts), detunings near the
/// line (|Δ| < 1.5Γ) or off it (to ±8Γ), couplings 0.05–1.5Γ at random phases; in a third of them an excited level
/// explicit. Its point within a wavelength of the focus, its velocity to 0.4 m/s.
fn random_model(u: &mut dyn FnMut() -> f64) -> (Request, Request, Vec<(usize, Part, usize)>, [f64; 3], [f64; 3]) {
    let (grounds, exciteds, ee) = ([0usize, 4, 6], [17usize, 13, 9], 1000.0);
    let g = 1 + (u() * 3.0) as usize;
    let e = 1 + (u() * 3.0) as usize;
    let ground: Vec<(usize, f64)> = (0..g).map(|i| (grounds[i], if i == 0 { 0.0 } else { 50.0 * u() - 25.0 })).collect();
    let excited: Vec<(usize, f64, f64, usize)> = (0..e).map(|i| (exciteds[i], ee + 3.0 * GAMMA * (u() - 0.5), GAMMA, grounds[(u() * g as f64) as usize])).collect();
    let frequencies = 1 + (u() * 2.0) as usize;
    let (mut rays, mut factors, mut dusts) = (vec![], vec![], vec![]);
    for axis in 0..frequencies {
        let delta = if u() < 0.4 { 3.0 * GAMMA * (u() - 0.5) } else { (if u() < 0.5 { -1.0 } else { 1.0 }) * GAMMA * (1.5 + 6.5 * u()) };
        let scale = GAMMA * (0.05 + 1.45 * u() * u());
        let mut couplings = vec![];
        for x in 0..e {
            for y in 0..g {
                if u() < 0.7 {
                    couplings.push((exciteds[x], grounds[y], C64::cis(6.283 * u()) * (scale * (0.3 + 0.7 * u()))));
                }
            }
        }
        if couplings.is_empty() {
            couplings.push((exciteds[0], grounds[0], C64::real(scale)));
        }
        // Dust (tilt2's dressed excited basis: a forbidden coupling 1e-7 … 1e-45 of the light, not zero) on every pair the
        // light does not couple, in half the frequencies.
        let dusty = u() < 0.5;
        let mut dust = vec![];
        for x in 0..e {
            for y in 0..g {
                let level = 10f64.powf(-7.0 - 38.0 * u());
                if dusty && !couplings.iter().any(|c| c.0 == exciteds[x] && c.1 == grounds[y]) {
                    dust.push((exciteds[x], grounds[y], C64::cis(6.283 * u()) * (scale * level)));
                }
            }
        }
        let beams = 1 + (u() * 2.0) as usize;
        let first = if u() < 0.7 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        for b in 0..beams {
            let direction = if b == 0 { first } else if u() < 0.6 { first.map(|x| -x) } else { [0.0, 1.0, 0.0] };
            // One frequency's beams share its field's polarization: another beam's couplings the first's times one factor.
            let factor = if b == 0 { C64::ONE } else { C64::cis(6.283 * u()) * (0.5 + u()) };
            rays.push(Ray { axis, nu: ee + delta, direction, couplings: couplings.iter().map(|&(x, y, v)| (x, y, v * factor)).collect() });
            dusts.push(dust.iter().map(|&(x, y, v)| (x, y, v * factor)).collect::<Vec<_>>());
            factors.push(factor);
        }
    }
    let make = |rays: &[Ray]| {
        let mut req = system(&ground, &excited, rays, frequencies, false);
        for (beam, f) in req.beams.iter_mut().zip(&factors) {
            beam.field[1] *= *f;
        }
        idle_decay(&mut req, GAMMA);
        req.validate().unwrap();
        req
    };
    let clean = make(&rays);
    let dusty: Vec<Ray> = rays.iter().zip(&dusts).map(|(r, d)| Ray { axis: r.axis, nu: r.nu, direction: r.direction, couplings: r.couplings.iter().chain(d.iter()).copied().collect() }).collect();
    let req = make(&dusty);
    let mut explicit = vec![];
    if u() < 1.0 / 3.0 {
        let fold = Fold::new(&req, &folded(&req, lab(&req)), 0).unwrap();
        if let Some(ray) = rays.iter().find(|r| r.couplings.iter().any(|c| c.0 == exciteds[0])) {
            let y = ray.couplings.iter().find(|c| c.0 == exciteds[0]).unwrap().1;
            explicit.push((fold.excited_level[exciteds[0]], Part { frequency: ray.axis, counter: false }, y));
        }
    }
    let r: [f64; 3] = std::array::from_fn(|_| 0.8 * u() - 0.4);
    let speed = 0.4 * u();
    let (ct, phi) = (2.0 * u() - 1.0, 6.283 * u());
    let st = (1.0 - ct * ct).sqrt();
    let v = [speed * st * phi.cos(), speed * st * phi.sin(), speed * ct];
    (req, clean, explicit, r, v)
}

/// A choice with block o's centred rows of `fold` decomposed (its centred copies, or a far block's excited rows), every
/// block's where o is none: the same representation with that centring taken out.
fn decomposed(choice: &Choice, fold: &Fold, o: Option<usize>) -> Choice {
    let mut out = choice.clone();
    for (_, b) in fold.blocks.iter().enumerate().filter(|(k, _)| o.is_none_or(|o| o == *k)) {
        for &(p, c) in &b.sources {
            let (part, ground) = (fold.components[c].part, fold.ground[p]);
            for x in out.excited.iter_mut().filter(|x| x.1 == Status::Centred && x.0 .0 == part && x.0 .1 == ground) {
                x.1 = Status::Decomposed;
            }
            for r in &b.rows[b.solved..] {
                if let Row::Copy { state, component } = *r {
                    let key = (part, ground, fold.components[component].part, fold.ground_level[state]);
                    out.copies.retain(|x| x.0 != key);
                    out.copies.push((key, Status::Decomposed));
                }
            }
            // The copies the block leaves out stay out (a far block holds none: its rows decomposed would take them).
            for &(state, component) in &b.omitted {
                let key = (part, ground, fold.components[component].part, fold.ground_level[state]);
                out.copies.retain(|x| x.0 != key);
                out.copies.push((key, Status::Omitted));
            }
        }
    }
    out
}

/// The sweeps where a test takes a series part out (to its rounding wherever its ratio is below one).
const SWEEPS: usize = 200;

/// Every reading is an estimate of what its approximation moves (readings.status.md; the user's rule of 2026-11-17: a
/// reading is the error's estimate, not a loose bound): on random small systems (`random_model`), at a decision's point
/// and at a point near it on the decision's held decompositions, each account the decision reads (proto.rs
/// `Fold::accounts`) against what taking its approximation out moves the held forward's G (`moved`: the photon rates
/// and the energies, the readings' unit) -- a series part swept on, a block made again at the point, a level joined, a far
/// block taken in, a block's centred rows decomposed. Per account the reading over the exact; said by kind.
#[test]
fn every_reading_is_an_estimate_of_what_it_reads() {
    use super::super::proto::{Cache, Item, Without};
    let mut state = 20261118u64;
    let mut u = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let trials: usize = std::env::var("READINGS_TRIALS").ok().map_or(60, |s| s.parse().unwrap());
    // (kind, reading, exact, rate part, potential part, what, on the modes)
    let mut rows: Vec<(String, f64, f64, f64, f64, String, f64)> = vec![];
    let tiny = 1e-9;
    let mut seen: std::collections::BTreeMap<String, usize> = Default::default();
    let (mut decided, mut nowhere) = (0usize, 0usize);
    let mut dust_most = 0.0_f64;
    for trial in 0..trials {
        let (req, clean, explicit, r1, v) = random_model(&mut u);
        // READINGS_CLEAN: the same systems without their dust (readings.status.md R2: which low readings the dust makes).
        let req = if std::env::var("READINGS_CLEAN").is_ok() { clean.clone() } else { req };
        let reach = [0.003, 0.03, 0.1][(u() * 3.0) as usize];
        let d: [f64; 3] = std::array::from_fn(|_| reach * (2.0 * u() - 1.0));
        let r2: [f64; 3] = std::array::from_fn(|a| r1[a] + d[a]);
        let Ok(choice) = Choice::base(&req, explicit.clone(), vec![]) else { continue };
        let Ok(sets) = Fold::sets(&req, &choice) else { continue };
        for set in sets {
            let Ok(fold) = Fold::new(&req, &choice, set[0]) else { continue };
            if fold.blocks.is_empty() {
                continue;
            }
            let Ok(mut ws) = fold.prototype_workspace() else { continue };
            let kappa = v.map(|x| x / fold.hbar_over_mass);
            let energy = fold.thermal;
            let mut first = Cache::default();
            for (point, r) in [(0usize, r1), (1, r2)] {
                let mut cache = if point == 0 { Cache::default() } else { first.clone() };
                let Some(step) = fold.prototype(r, kappa, energy, &mut cache, &mut ws) else {
                    nowhere += 1;
                    break;
                };
                decided += 1;
                let whole = step.reading;
                if point == 0 {
                    first = cache.clone();
                    // The dust's part of the reading: the same set without it, at the same point.
                    if let Some(rc) = Choice::base(&clean, explicit.clone(), vec![]).ok().and_then(|c| Fold::new(&clean, &c, set[0]).ok()).and_then(|f| {
                        let mut w = f.prototype_workspace().ok()?;
                        f.prototype(r, kappa, energy, &mut Cache::default(), &mut w).map(|s| s.reading)
                    }) {
                        if whole.is_finite() && rc.is_finite() && whole.max(rc) > tiny {
                            dust_most = dust_most.max((whole - rc).abs() / whole.max(rc));
                        }
                    }
                }
                let accounts = ws.accounts().to_vec();
                let Some(held) = fold.prototype_without(r, kappa, energy, &cache, &mut ws, Without::Nothing, 0) else { break };
                let what = format!("trial {trial} set {set:?} point {point} (n {}, explicit {}, |v| {:.2})", fold.size(), fold.explicit.len(), v.iter().map(|x| x * x).sum::<f64>().sqrt());
                // The variant of block o (every block where none) with its centred rows decomposed, everything else taken
                // out alike (a far block `held_far` of this fold kept as the decision holds it): this fold's G and its.
                let variant = |o: Option<usize>, ws: &mut super::super::proto::Workspace, held_far: Option<usize>| -> Option<(Mat, Mat)> {
                    let other = decomposed(&choice, &fold, o);
                    let f2 = Fold::new(&req, &other, set[0]).ok()?;
                    let mut w2 = f2.prototype_workspace().ok()?;
                    let mut c2 = Cache::default();
                    f2.prototype(r, kappa, energy, &mut c2, &mut w2)?;
                    let exact = f2.prototype_without(r, kappa, energy, &c2, &mut w2, Without::All(None), SWEEPS)?;
                    let mine = fold.prototype_without(r, kappa, energy, &cache, ws, Without::All(held_far), SWEEPS)?;
                    Some((mine, exact))
                };
                for a in accounts.iter().filter(|a| a.on) {
                    *seen.entry(format!("{:?}", a.item).split(' ').next().unwrap().to_string()).or_insert(0usize) += 1;
                    let reading = a.squares.iter().map(|x| rate(x.sqrt())).fold(0.0, f64::max);
                    let (kind, pair) = match a.item {
                        Item::Series { block, part } => ("series", fold.prototype_without(r, kappa, energy, &cache, &mut ws, Without::Part(block, part), SWEEPS).map(|g| (held.clone(), g))),
                        Item::Reuse { block } => ("reuse", fold.prototype_without(r, kappa, energy, &cache, &mut ws, Without::Remade(block), 0).map(|g| (held.clone(), g))),
                        Item::Omitted { block, level } if point == 0 => ("omitted", fold.prototype_without(r, kappa, energy, &cache, &mut ws, Without::Level(block, level), 0).map(|g| (held.clone(), g))),
                        Item::Left { block } => ("left", fold.prototype_without(r, kappa, energy, &cache, &mut ws, Without::In(block), 0).map(|g| (held.clone(), g))),
                        Item::Centred { block, far: true } => ("centred far", variant(Some(ws.block_of(block, true)), &mut ws, Some(block))),
                        Item::Centred { block, far: false } if point == 0 => ("centred near", variant(Some(ws.block_of(block, false)), &mut ws, None)),
                        _ => continue,
                    };
                    let Some((mine, exact)) = pair else { continue };
                    let (moved, rate_part, potential) = moved(&mine, &exact, energy);
                    let modes = moved_modes(&mine, &exact, energy).0;
                    if reading > tiny || moved > tiny {
                        rows.push((format!("{kind}{}", if point == 1 { " (held)" } else { "" }), reading, moved, rate_part, potential, format!("{what} {:?}", a.item), modes));
                    }
                }
                if std::env::var("READINGS_ONLY").ok().is_some_and(|x| x.parse::<usize>().unwrap() == trial) {
                    println!("== {what}: whole reading {whole:.3e}");
                    for (o, b) in fold.blocks.iter().enumerate() {
                        println!("   block {o} at {:+.3} (far {}): solved {} of rows {:?}; sources {:?}; omitted copies {:?}; kept {:?}", b.value, b.far(), b.solved, b.rows, b.sources, b.omitted, b.kept);
                    }
                    println!("   explicit {:?}; driven {:?}; turning {:?}", fold.explicit, fold.driven, fold.turning);
                    for a in &accounts {
                        println!("   {:?} on {}: {}", a.item, a.on, e3(&a.squares));
                    }
                    for a in &accounts {
                        if let Item::Series { block, part } = a.item {
                            for sw in [1, 2, 3, 4, 6, 10, 20, 50, 200] {
                                if let Some(g) = fold.prototype_without(r, kappa, energy, &cache, &mut ws, Without::Part(block, part), sw) {
                                    println!("     part ({block}, {part}) at {sw} sweeps: against the held {:?}", moved(&held, &g, energy));
                                }
                            }
                        }
                    }
                    for sw in [1, 2, 3, 4, 6, 10, 20, 50, 200] {
                        if let Some(g) = fold.prototype_without(r, kappa, energy, &cache, &mut ws, Without::All(None), sw) {
                            println!("     all at {sw} sweeps: against the held {:?}", moved(&held, &g, energy));
                        }
                    }
                    let all = fold.prototype_without(r, kappa, energy, &cache, &mut ws, Without::All(None), SWEEPS);
                    if let (Some(all), Some((_, exact))) = (all, variant(None, &mut ws, None)) {
                        println!("   held against all out {:?}; all out against every centring decomposed {:?}", moved(&held, &all, energy), moved(&all, &exact, energy));
                        println!("   held G {:?}", held.data);
                        println!("   exact G {:?}", exact.data);
                    }
                }
                // The whole: every approximation out, every centred row decomposed.
                if let Some((_, exact)) = variant(None, &mut ws, None) {
                    let (moved, rate_part, potential) = moved(&held, &exact, energy);
                    let modes = moved_modes(&held, &exact, energy).0;
                    if whole > tiny || moved > tiny {
                        rows.push((format!("whole{}", if point == 1 { " (held)" } else { "" }), whole, moved, rate_part, potential, what.clone(), modes));
                    }
                }
            }
        }
    }
    println!("accounts seen (on): {seen:?}; {decided} decisions, {nowhere} where the fold does not exist; the dust moves a whole reading by at most {dust_most:.2e} of it");
    let mut failing: Vec<String> = vec![];
    let mut kinds: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    kinds.sort();
    kinds.dedup();
    for kind in &kinds {
        let mut ratios: Vec<(f64, usize)> = rows.iter().enumerate().filter(|(_, r)| &r.0 == kind).map(|(i, r)| (if r.2 > 0.0 { r.1 / r.2 } else { f64::INFINITY }, i)).collect();
        ratios.sort_by(|a, b| a.0.total_cmp(&b.0));
        let median = ratios[ratios.len() / 2].0;
        let mut on_modes: Vec<f64> = rows.iter().filter(|r| &r.0 == kind && (r.6 > tiny || r.1 > tiny)).map(|r| if r.6 > 0.0 { r.1 / r.6 } else { f64::INFINITY }).collect();
        on_modes.sort_by(|a, b| a.total_cmp(b));
        if on_modes.is_empty() {
            println!("{kind}: {} readings, none beyond the rounding on the modes", ratios.len());
            continue;
        }
        println!("{kind}: {} readings, reading over exact min {:.3} median {:.3} max {:.3e}; on the modes min {:.3} median {:.3}", ratios.len(), ratios[0].0, median, ratios[ratios.len() - 1].0, on_modes[0], on_modes[on_modes.len() / 2]);
        for &(q, i) in ratios.iter().take(4) {
            let r = &rows[i];
            println!("    {q:.3}: reading {:.3e}, exact {:.3e} (rate {:.3e}, potential {:.3e}; modes {:.3e}) -- {}", r.1, r.2, r.3, r.4, r.6, r.5);
        }
        for &(q, i) in ratios.iter().rev().take(2) {
            let r = &rows[i];
            println!("    {q:.3e}: reading {:.3e}, exact {:.3e} (rate {:.3e}, potential {:.3e}; modes {:.3e}) -- {}", r.1, r.2, r.3, r.4, r.6, r.5);
        }
        // On the modes (the slow modes' rates and energies: what the dynamics run on), every reading at least the error it
        // reads, to the rounding, and at most three times it at the median: an estimate, not a bound.
        let low = on_modes.iter().copied().filter(|&q| q < 1.0 - 1e-9).count();
        if low > 0 || on_modes[on_modes.len() / 2] > 3.0 {
            failing.push(format!("{kind}: {low} of {} below the error, median {:.2}", on_modes.len(), on_modes[on_modes.len() / 2]));
        }
    }
    assert!(failing.is_empty(), "readings that are no estimate of what they read: {failing:?}");
}

/// Diagnostic (readings.status.md R2, item 3; DECIDE_REQ a scene's path, tilt2's): the start's decided representation
/// (`Onefold::new`), per set at the points R2_POINTS ("s,vx,vy,vz;...", m/s; the phases zero) the decision's forward --
/// its whole reading against the share -- and each account read beyond a tenth of the share; for every near block's
/// centred copies (`Item::Centred`, near) the error its reading reads, measured as the property test measures it
/// (`every_reading_is_an_estimate_of_what_it_reads`): every approximation out on the held decompositions, against the
/// same with that block's centred rows decomposed, on G's modes.
#[test]
#[ignore]
fn a_scene_s_centred_copies_read_against_their_error() {
    use super::super::decide::Local;
    use super::super::proto::{Cache, Item, Without};
    let req = crate::model::testing::scene(&std::env::var("DECIDE_REQ").expect("DECIDE_REQ"));
    let light = crate::field::Light::new(&req);
    let initial = crate::solve::components(&req);
    let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).unwrap();
    let reference = crate::solve::alone(initial[chief].0);
    let hbar_over_mass = crate::units::hbar_over_mass(req.mass_kg);
    let states = super::super::run::Reference::states(&req, &light).unwrap();
    let (floor, _) = super::super::run::Onefold::scales(&light, hbar_over_mass, &super::super::run::Reference::of(&states, &reference));
    // R2_EXPLICIT ("level,frequency,ground;...", co-rotating): that representation (the tests' fixture's tiers at the focus)
    // instead of the start's decision.
    let choice = match std::env::var("R2_EXPLICIT") {
        Ok(list) => {
            let keys = list
                .split(';')
                .filter(|x| !x.trim().is_empty())
                .map(|x| {
                    let v: Vec<usize> = x.split(',').map(|y| y.trim().parse().unwrap()).collect();
                    (v[0], crate::model::Component { frequency: v[1], counter: false }, v[2])
                })
                .collect();
            Choice::base(&req, keys, vec![]).expect("the choice")
        }
        Err(_) => super::super::run::Onefold::new(&req, reference).expect("the run").choice().clone(),
    };
    let share = crate::model::elim::share();
    let points: Vec<[f64; 4]> = std::env::var("R2_POINTS")
        .unwrap_or_else(|_| "1,0,0,0;0.5,0,0,0.2;0.1,0.2,0,0.1".into())
        .split(';')
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            let v: Vec<f64> = x.split(',').map(|y| y.trim().parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3]]
        })
        .collect();
    println!("floor {floor:.4} rad/µs, share {share:.4}");
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold0 = Fold::new(&req, &choice, set[0]).expect("the fold");
        if fold0.blocks.is_empty() {
            continue;
        }
        for p in &points {
            let local = Local { s: p[0], phases: vec![0.0; light.beams.len()], kappa: [p[1], p[2], p[3]].map(|v| v / hbar_over_mass) };
            let mut fold = fold0.clone();
            fold.local = local.factors(&light);
            let mut ws = fold.prototype_workspace().unwrap();
            let mut cache = Cache::default();
            let Some(step) = fold.prototype([0.0; 3], local.kappa, floor, &mut cache, &mut ws) else {
                println!("set {set:?} point {p:?}: no fold");
                continue;
            };
            println!("set {set:?} point s {} v {:?}: whole reading {:.3e} ({} the share)", p[0], &p[1..], step.reading, if step.reading > share { "beyond" } else { "within" });
            let accounts = ws.accounts().to_vec();
            let Some(mine) = fold.prototype_without([0.0; 3], local.kappa, floor, &cache, &mut ws, Without::All(None), SWEEPS) else { continue };
            for a in accounts.iter().filter(|a| a.on) {
                let reading = a.squares.iter().map(|x| rate(x.sqrt())).fold(0.0, f64::max);
                if reading < 0.1 * share {
                    continue;
                }
                let mut said = format!("  {:?}: reading {reading:.3e}", a.item);
                if let Item::Centred { block, far: false } = a.item {
                    let o = ws.block_of(block, false);
                    let other = decomposed(&choice, &fold, Some(o));
                    let exact = Fold::new(&req, &other, set[0]).ok().and_then(|mut f2| {
                        f2.local = fold.local.clone();
                        let mut w2 = f2.prototype_workspace().ok()?;
                        let mut c2 = Cache::default();
                        f2.prototype([0.0; 3], local.kappa, floor, &mut c2, &mut w2)?;
                        f2.prototype_without([0.0; 3], local.kappa, floor, &c2, &mut w2, Without::All(None), SWEEPS)
                    });
                    match exact {
                        Some(exact) => {
                            let (modes, rate_part, potential) = moved_modes(&mine, &exact, floor);
                            let (any, ..) = moved(&mine, &exact, floor);
                            let b = &fold.blocks[o];
                            let centred: Vec<String> = b.rows[b.solved..].iter().map(|r| format!("{r:?}")).collect();
                            said += &format!(" -- block {o} at {:+.3} (centred rows {centred:?}): the error on the modes {modes:.3e} (rate {rate_part:.3e}, energy {potential:.3e}), on any state {any:.3e}; reading over error {:.2}", b.value, reading / modes);
                        }
                        None => said += " -- its centred rows decomposed: no fold",
                    }
                }
                println!("{said}");
            }
        }
    }
}
