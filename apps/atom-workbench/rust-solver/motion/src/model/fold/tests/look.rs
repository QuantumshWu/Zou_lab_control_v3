//! The look over a cell's tests (proto/look.rs; unify2 §4.2; BRIEF_P_LOOK): a two-level atom whose fold ceases to exist
//! inside a cell, seen from the cell's start, against the exact root along the cell and the toy's numbers
//! (`lookahead.py`); and a multilevel set's look against the forward at every point it saw -- what holding the off-diagonal
//! part costs.
use super::super::proto::look::Look;
use super::super::proto::Cache;
use super::*;
use crate::field::Amplitude;

/// The two-level fold's P-like root at a point with lanes (sources V_l, poles p_l = κ_e + w_l): the roots of λ = Σ_l
/// N_l/(λ − p_l), N_l = (Σ_k V_k)* V_l -- of the polynomial λ Π(λ − p_l) − Σ_l N_l Π_{k≠l}(λ − p_k), by its companion's
/// eigenvalues --, the one of least excitation x = |Σ_l V_l/(λ − p_l)|² (the toy's smaller |λ| at one lane); its x and
/// its ratio r = |Σ_l N_l/(λ − p_l)²|.
fn p_like(v: &[C64], p: &[C64]) -> (C64, f64, f64) {
    roots(v, p).into_iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap()
}

/// Every root of `p_like`'s polynomial with its x and r.
fn roots(v: &[C64], p: &[C64]) -> Vec<(C64, f64, f64)> {
    let total: C64 = v.iter().copied().sum();
    let num: Vec<C64> = v.iter().map(|x| total.conj() * *x).collect();
    let times = |a: &[C64], root: C64| -> Vec<C64> {
        let mut out = vec![C64::ZERO; a.len() + 1];
        for (k, &c) in a.iter().enumerate() {
            out[k + 1] += c;
            out[k] -= c * root;
        }
        out
    };
    let mut poly = p.iter().fold(vec![C64::ZERO, C64::ONE], |q, &x| times(&q, x));
    for (l, &nl) in num.iter().enumerate() {
        let others = p.iter().enumerate().filter(|(k, _)| *k != l).fold(vec![C64::ONE], |q, (_, &x)| times(&q, x));
        for (k, c) in others.iter().enumerate() {
            poly[k] -= nl * *c;
        }
    }
    let d = poly.len() - 1;
    let companion = Mat::from_fn(d, d, |i, j| if j == d - 1 { -poly[i] / poly[d] } else if i == j + 1 { C64::ONE } else { C64::ZERO });
    let roots = crate::linalg::eigen(&companion).expect("the companion's roots").values;
    roots
        .iter()
        .map(|&l| {
            let z: C64 = v.iter().zip(p).map(|(a, b)| *a / (l - *b)).sum();
            let r: C64 = num.iter().zip(p).map(|(a, b)| *a / ((l - *b) * (l - *b))).sum();
            (l, z.norm_sqr(), r.abs())
        })
        .collect()
}

/// The largest of f over [0, cell] from its values on an even grid: about the grid's largest, golden-section search to the
/// rounding of the time (an exceptional point's kink is where x and r peak: the grid alone misses it).
fn refined(f: &dyn Fn(f64) -> f64, cell: f64, grid: &[f64]) -> f64 {
    let m = grid.len() - 1;
    let k = (0..=m).max_by(|&a, &b| grid[a].total_cmp(&grid[b])).unwrap();
    let h = cell / m as f64;
    let (mut a, mut b) = ((k.max(1) - 1) as f64 * h, ((k + 1).min(m)) as f64 * h);
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut best = grid[k];
    while b - a > f64::EPSILON * cell {
        let (u, v) = (b - golden * (b - a), a + golden * (b - a));
        let (fu, fv) = (f(u), f(v));
        best = best.max(fu).max(fv);
        if fu >= fv {
            b = v;
        } else {
            a = u;
        }
    }
    best
}

/// What a run of the look along a straight path read (`look_along`).
#[derive(Default, Debug)]
struct Tally {
    cells: usize,
    /// Cells whose fold does not exist at their start (the decision sees it at its own point: no look).
    at_point: usize,
    samples: Vec<usize>,
    points: usize,
    /// The cells' largest excitation: the look's relative error, the most the cell's two ends miss of it (both against
    /// the exact root on 400 points), and the cells whose exact largest is above ½ that the look reads below it.
    worst: f64,
    ends: f64,
    missed: usize,
    /// The cells' largest ratio: the most the look reads below the exact one relatively; the cells that cease to exist
    /// inside (the exact ratio at one, within the allowed error, somewhere) and of them those the look sees (a point with
    /// no fold, or its ratio at one within the error); the cells that exist throughout that the look reads as ceasing.
    under: f64,
    ceasing: usize,
    seen: usize,
    alarms: usize,
    /// The look at its decision against the exact root (its column energy's and excitation's largest relative
    /// differences: one column, the look is the fold), and the forward's (its series' rest); the forward's last sweep's
    /// diagonal reproduced (`Look::identity`).
    root: [f64; 2],
    forward: [f64; 2],
    identity: f64,
    work: f64,
    /// The worst cell's: its start, the exact largest x and ratio, the look's, its points; the worst decision's roots.
    detail: String,
    detail_root: String,
}

/// The look from every cell's start along a straight path (r₀ + v t ẑ, κ = v/(ħ/m) ẑ: a heavy atom's) of a two-level
/// request's ground state 0, against the exact root (`p_like`) of the light there: the drive's beams (index, coupling
/// per unit amplitude), the excited line relative to the laser `line` at the beams' amplitudes, its width Γ.
#[allow(clippy::too_many_arguments)]
fn look_along(req: &Request, drive: &[(usize, f64)], line: &dyn Fn(&[Amplitude]) -> f64, gamma: f64, start: [f64; 3], v: f64, cell: f64, cells: usize, allowed: f64) -> Tally {
    // The excited node solved, not centred: the tiers the focus gives (`Choice::base`) can centre it where the line is
    // pushed far from the drive there, and a centring is its own reading, not the look's (the decision's remedy).
    let mut choice = Choice::base(req, vec![]).unwrap();
    choice.excited.iter_mut().for_each(|x| x.1 = Status::Decomposed);
    let fold = Fold::new(req, &choice, 0).unwrap();
    let mut ws = fold.prototype_workspace().unwrap();
    let mut cache = Cache::default();
    let mut look: Look = fold.look(&ws);
    let hm = fold.hbar_over_mass;
    let kappa = [0.0, 0.0, v / hm];
    let truth = |r: [f64; 3]| -> (C64, f64, f64) {
        let a = fold.light.amplitudes(r);
        let delta = line(&a);
        let (vs, ps): (Vec<C64>, Vec<C64>) = drive
            .iter()
            .map(|&(l, g)| {
                let k = fold.light.beams[l].wave_vector();
                let w = hm * ((0..3).map(|q| k[q] * kappa[q]).sum::<f64>() + 0.5 * (0..3).map(|q| k[q] * k[q]).sum::<f64>());
                (a[l].value * g, C64::new(delta + w, -0.5 * gamma))
            })
            .unzip();
        p_like(&vs, &ps)
    };
    let root_list = |r: [f64; 3]| -> Vec<(C64, f64, f64)> {
        let a = fold.light.amplitudes(r);
        let delta = line(&a);
        let (vs, ps): (Vec<C64>, Vec<C64>) = drive
            .iter()
            .map(|&(l, g)| {
                let k = fold.light.beams[l].wave_vector();
                let w = hm * ((0..3).map(|q| k[q] * kappa[q]).sum::<f64>() + 0.5 * (0..3).map(|q| k[q] * k[q]).sum::<f64>());
                (a[l].value * g, C64::new(delta + w, -0.5 * gamma))
            })
            .unzip();
        roots(&vs, &ps)
    };
    let mut tally = Tally { cells, ..Tally::default() };
    for c in 0..cells {
        let rd = [start[0], start[1], start[2] + v * cell * c as f64];
        let path = move |t: f64| ([rd[0], rd[1], rd[2] + v * t], kappa);
        let fine: Vec<(f64, f64)> = (0..400).map(|k| truth(path(cell * k as f64 / 399.0).0)).map(|x| (x.1, x.2)).collect();
        let exact = refined(&|t| truth(path(t).0).1, cell, &fine.iter().map(|x| x.0).collect::<Vec<_>>());
        let ratio = refined(&|t| truth(path(t).0).2, cell, &fine.iter().map(|x| x.1).collect::<Vec<_>>());
        tally.ends = tally.ends.max(exact - fine[0].0.max(fine[399].0));
        if fold.prototype(rd, kappa, &mut cache, &mut ws).is_none() {
            tally.at_point += 1;
            continue;
        }
        let (lambda, x, _) = truth(rd);
        let (h, whole) = ws.columns(fold.ground.len())[0];
        tally.forward = [tally.forward[0].max((h - lambda).abs() / lambda.abs()), tally.forward[1].max((whole - x).abs() / x)];
        fold.prototype_look(&path, cell, &cache, &ws, &mut look);
        let first = look.seen[0].columns[0];
        if (first.c - lambda).abs() / lambda.abs() > tally.root[0] {
            let shifts: Vec<Vec<f64>> = look.seen[0].near.iter().map(|x| x.shifts.clone()).collect();
            let kappas: Vec<Vec<C64>> = look.seen[0].near.iter().map(|x| x.kappa.clone()).collect();
            let doppler: Vec<f64> = drive.iter().map(|&(l, _)| { let k = fold.light.beams[l].wave_vector(); hm * ((0..3).map(|q| k[q] * kappa[q]).sum::<f64>() + 0.5 * (0..3).map(|q| k[q] * k[q]).sum::<f64>()) }).collect();
            tally.detail_root = format!("at z {:.3}: the look's c {:.6?} x {:.6} r {:.6}, the forward's {h:.6?} {whole:.6}; the roots {:.6?}; shifts {shifts:?} kappa {kappas:?} the truth's Doppler {doppler:?}", rd[2], first.c, first.x, first.r, root_list(rd));
        }
        tally.root = [tally.root[0].max((first.c - lambda).abs() / lambda.abs()), tally.root[1].max((first.x - x).abs() / x)];
        tally.identity = tally.identity.max(look.identity);
        tally.samples.push(look.samples);
        tally.points += look.seen.len();
        tally.work += look.work;
        let predicted = look.largest(|s| s.columns[0].x);
        let read = look.largest(|s| s.columns[0].r);
        let ceases = look.ceases().is_some();
        if ratio >= 1.0 - allowed {
            tally.ceasing += 1;
            tally.seen += usize::from(ceases || read >= 1.0 - allowed);
        } else {
            tally.alarms += usize::from(ceases);
        }
        if ceases {
            continue;
        }
        let error = (predicted - exact).abs() / exact;
        if error > tally.worst {
            let points: Vec<String> = look.seen.iter().map(|s| format!("{:.3}{} x {:.4} r {:.4}{}", s.t, if s.sample { "" } else { "*" }, s.columns[0].x, s.columns[0].r, if s.columns[0].exists { "" } else { " (none)" })).collect();
            let curve: Vec<String> = (0..=20).map(|k| format!("{:.2}", fine[k * 399 / 20].0)).collect();
            tally.detail = format!("cell from z {:.3}: exact x {exact:.4} r {ratio:.4}, the look's x {predicted:.4} r {read:.4}; {points:?}; the exact x every D/20 {curve:?}", rd[2]);
        }
        tally.worst = tally.worst.max(error);
        tally.missed += usize::from(exact > 0.5 && predicted <= 0.5);
        tally.under = tally.under.max((ratio - read) / ratio);
    }
    tally.samples.sort();
    tally.samples.dedup();
    tally
}

/// The look at a cell's start sees where the fold ceases to exist inside the cell (unify2 §4.2, §2.3; BRIEF_P_LOOK): two
/// two-level atoms at the speeds of plan4 R-b (0.10–0.24 µm/µs), the decision at every cell's start, the look over the
/// cell from there; the truth the exact P-like root (`p_like`) on 400 points of the cell, its largest refined about the
/// grid's (`refined`: an exceptional point's kink). The excited node solved (`look_along`).
///
/// - R-b's scene (BRIEF_STEP4B (a)): a resonant standing wave (s = 0.3 a beam, 1.2 at the antinodes), eight paths of
///   ten cells of 0.4 µs. 45 of 80 cells have no fold at their start (the decision sees it at its own point); of the
///   others 13 cease inside, all seen from their start, none read ceasing that does not; a cell's two ends miss its
///   largest excitation by up to 0.33.
/// - The toy's (`lookahead.py`): a line pushed through a drive by a tweezer's excited-state shift (120 rad/µs at its
///   focus, waist 1.2 µm, an outside form; the laser 26 rad/µs above the line beside it), Γ = 38, Ω₀ = 30 or 20 a running
///   wave across the path (no Doppler shift: the toy itself), Ω₀ = 30 a standing wave along it (its beams' Doppler shifts
///   ±kv, which the toy left out), twelve cells from 0.4 µm at 0.24 or 0.10 µm/µs, cells of 0.4 or 0.8 µs. The cells'
///   largest excitation: the running waves within 3.2·10⁻⁴ (the toy's 4·10⁻³–3.2·10⁻²: the resonance's critical point
///   is where the bare distance is zero, exactly, for one lane), the standing wave 7.6·10⁻⁴ at 0.4 µs and 4.8·10⁻² at
///   0.8 (the toy's 1.0·10⁻² and 0.128; the root turns fast where no input has a critical point, the Hermite peak takes
///   it); a cell's ends miss up to 0.65.
///
/// Asserted: the cells' largest excitation within the toy's errors -- 1.3·10⁻² at D = 0.4 µs, 0.13 at 0.8 (unify2 §4.2;
/// the toy's worst cells, its standing wave at 0.8) --; no cell above ½ read below it; the cells' largest ratio not read
/// lower by more than the same; every cell that ceases to exist inside (its exact ratio at one within the same) seen,
/// none that exists throughout read ceasing; at the decisions the look the exact root to the rounding (one column: the
/// fold itself), the forward it within its series' rest (twice the share), the look's diagonal at the forward's last
/// sweep the forward's to the rounding.
#[test]
fn a_cell_s_lookahead_sees_the_loss_of_existence() {
    let share = crate::model::elim::share();
    let allowed = |cell: f64| if cell <= 0.4 { 1.3e-2 } else { 0.13 };
    let check = |what: &str, cell: f64, t: &Tally| {
        println!(
            "{what}: samples {:?}, {} points over {} cells ({} not existing at their start); the cells' largest x: the look's error {:.2e}, the ends miss {:.3}, missed {}; the ratio read low by {:.2e}; ceasing inside {} (seen {}), false alarms {}; at the decisions the look against the root {:.1e}, {:.1e} (the forward's {:.1e}, {:.1e}), its identity {:.1e}; work {:.0} a cell\n   the worst {}\n   the worst decision {}",
            t.samples, t.points, t.cells, t.at_point, t.worst, t.ends, t.missed, t.under, t.ceasing, t.seen, t.alarms, t.root[0], t.root[1], t.forward[0], t.forward[1], t.identity, t.work / (t.cells - t.at_point).max(1) as f64, t.detail, t.detail_root
        );
        assert!(t.worst <= allowed(cell), "{what}: the cells' largest excitation within the toy's error");
        assert_eq!(t.missed, 0, "{what}: no cell above ½ read below it");
        assert!(t.under <= allowed(cell), "{what}: the cells' largest ratio not read low");
        assert_eq!(t.seen, t.ceasing, "{what}: every cell that ceases to exist inside seen from its start");
        assert_eq!(t.alarms, 0, "{what}: no cell that exists throughout read as ceasing");
        assert!(t.root[0] <= 1e-9 && t.root[1] <= 1e-9, "{what}: the look at its decision is the exact root (one column: the fold)");
        assert!(t.forward[0] <= 2.0 * share && t.forward[1] <= 2.0 * share, "{what}: the forward the exact root within its series' rest");
        assert!(t.identity <= 1e-12, "{what}: the look's diagonal at the forward's last sweep is the forward's");
    };
    // R-b's scene.
    {
        let (ee, s, cell) = (1000.0, 0.3, 0.4);
        let g = 0.5 * GAMMA * (s / 2.0_f64).sqrt();
        let rays = [1.0, -1.0].map(|d| Ray { axis: 0, nu: ee, direction: [0.0, 0.0, d], couplings: vec![(17, 0, C64::real(g))] });
        let mut req = system(&[(0, 0.0)], &[(17, ee, GAMMA, 0)], &rays, 1, true);
        idle_decay(&mut req, GAMMA);
        req.validate().unwrap();
        let mut all = Tally::default();
        for i in 0..8 {
            let t = look_along(&req, &[(0, g), (1, g)], &|_| 0.0, GAMMA, [0.0, 0.0, 0.05 * i as f64], 0.1 + 0.02 * i as f64, cell, 10, allowed(cell));
            all.cells += t.cells;
            all.at_point += t.at_point;
            all.samples.extend(t.samples);
            all.points += t.points;
            let worse = t.worst >= all.worst;
            all.worst = all.worst.max(t.worst);
            all.ends = all.ends.max(t.ends);
            all.missed += t.missed;
            all.under = all.under.max(t.under);
            all.ceasing += t.ceasing;
            all.seen += t.seen;
            all.alarms += t.alarms;
            if t.root[0] >= all.root[0] {
                all.detail_root = t.detail_root.clone();
            }
            all.root = [all.root[0].max(t.root[0]), all.root[1].max(t.root[1])];
            all.forward = [all.forward[0].max(t.forward[0]), all.forward[1].max(t.forward[1])];
            all.identity = all.identity.max(t.identity);
            all.work += t.work;
            if worse {
                all.detail = t.detail;
            }
        }
        all.samples.sort();
        all.samples.dedup();
        check("R-b, a resonant standing wave at 0.10–0.24 µm/µs", cell, &all);
    }
    // The toy's.
    let (gamma, d0, shift, ee) = (38.0, -26.0, 120.0, 1000.0);
    for (omega, standing) in [(30.0, false), (20.0, false), (30.0, true)] {
        let tweezer = Ray { axis: 0, nu: -1e5, direction: [0.0, 1.0, 0.0], couplings: vec![] };
        let mut rays = vec![tweezer];
        let beams: Vec<(usize, f64)> = if standing {
            rays.extend([1.0, -1.0].map(|d| Ray { axis: 1, nu: ee - d0, direction: [0.0, 0.0, d], couplings: vec![(17, 0, C64::real(0.25 * omega))] }));
            vec![(1, 0.25 * omega), (2, 0.25 * omega)]
        } else {
            rays.push(Ray { axis: 1, nu: ee - d0, direction: [1.0, 0.0, 0.0], couplings: vec![(17, 0, C64::real(0.5 * omega))] });
            vec![(1, 0.5 * omega)]
        };
        // H0 holds the outside form at the focus: the line there ee + 120, beside the tweezer ee.
        let mut req = system(&[(0, 0.0)], &[(17, ee + shift, gamma, 0)], &rays, 2, true);
        req.beams[0].waist = Some(1.2);
        let f = req.outside_forms.iter().position(|f| f.frequency_index == vec![1, 0]).unwrap();
        let focus = req.beams[0].field[1].norm_sqr();
        req.outside_forms[f].matrices[1][(GROUND + 17, GROUND + 17)] = C64::real(shift / focus);
        idle_decay(&mut req, GAMMA);
        req.validate().unwrap();
        let line = |a: &[Amplitude]| d0 + shift * a[0].value.norm_sqr();
        for (v, cell) in [(0.24, 0.4), (0.10, 0.4), (0.24, 0.8)] {
            let t = look_along(&req, &beams, &line, gamma, [0.0, 0.0, 0.4], v, cell, 12, allowed(cell));
            check(&format!("the toy's, Ω₀ {omega} {} at {v} µm/µs, cells of {cell} µs", if standing { "standing" } else { "running" }), cell, &t);
        }
    }
}

/// The look on multilevel sets holds their off-diagonal part (unify2 §4.2: what several columns hold besides the
/// diagonal; BRIEF_P_LOOK): R4_LIN_4 (its atom 10⁶ times heavier: straight paths) at the focus's choice (`focus_choice`),
/// every set of two columns or more, a straight path at 0.24 µm/µs along (1, 2, 3)/√14 through the lattice, six cells of
/// 0.4 µs, the decision at every cell's start and the look from there. The truth: the forward made at every point the
/// look saw (its own decomposition there, the series to its share) and on 41 points a cell. Printed per set: the held
/// part's size ‖O‖/‖V + O‖ at the decisions; the look's excitations against the forward's at the look's own points (its
/// inputs exact there: what holding O costs) and the cells' largest per column against the 41 points; its largest ratio
/// against the forward's measured series ratio; the profiles' worst (§2.7: the series at two passes, the centrings, the
/// nearest distance, the chirp, the reuse's drift); the look's work against the decision forward's. Asserted: at its
/// points and as the cells' largest, the look's excitations within the share of the forward's (the forward's own
/// accuracy: no decision moves within it), each relative to its column's whole response (an explicit column's unit with
/// it). Said, not asserted: the series' ratio of a set whose explicit states P drives is the static couplings' (Δ over
/// the rows' distances), which the diagonal's pass 0 does not read -- R4_LIN_4's F = 1 set with its D2 F′ = 2 explicit:
/// the forward's 0.216 against the look's 0.002 (plook.status.md, U3).
#[test]
fn a_multilevel_cell_s_lookahead_holds_its_off_diagonal_part() {
    let share = crate::model::elim::share();
    let mut req = crate::model::testing::scene("R4_LIN_4");
    req.mass_kg *= 1e6;
    let choice = focus_choice(&req);
    let (cell, cells, speed) = (0.4, 6, 0.24);
    let direction = [1.0, 2.0, 3.0].map(|x: f64| x / 14f64.sqrt());
    let start = [0.05, -0.1, 0.08];
    let mut checked = 0;
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        let n = fold.size();
        if fold.blocks.is_empty() || n < 2 {
            continue;
        }
        let (Ok(mut ws), Ok(mut truth)) = (fold.prototype_workspace(), fold.prototype_workspace()) else { continue };
        let mut look = fold.look(&ws);
        let (ground, hm) = (fold.ground.len(), fold.hbar_over_mass);
        let kappa = direction.map(|x| x * speed / hm);
        let (mut cache, mut exact) = (Cache::default(), Cache::default());
        let (mut held, mut points, mut largest, mut ratios, mut absent, mut missing) = (0.0_f64, 0.0_f64, 0.0_f64, [0.0_f64; 2], 0, 0);
        let (mut series, mut centred, mut nearest, mut chirp, mut drift, mut middle) = (0.0_f64, 0.0_f64, f64::INFINITY, 0.0_f64, 0.0_f64, 0.0_f64);
        let (mut work, mut forward, mut samples) = (0.0, 0.0, vec![]);
        for c in 0..cells {
            let rd: [f64; 3] = std::array::from_fn(|a| start[a] + direction[a] * speed * cell * c as f64);
            let path = move |t: f64| -> ([f64; 3], [f64; 3]) { (std::array::from_fn(|a| rd[a] + direction[a] * speed * t), kappa) };
            let Some(step) = fold.prototype(rd, kappa, &mut cache, &mut ws) else {
                absent += 1;
                continue;
            };
            forward += step.work.iter().sum::<f64>();
            fold.prototype_look(&path, cell, &cache, &ws, &mut look);
            work += look.work;
            samples.push(look.samples);
            held = held.max(look.off_diagonal());
            // The look against the forward at its own points.
            let mut most = vec![0.0_f64; n];
            for s in &look.seen {
                let (r, k) = path(s.t);
                let Some(step) = fold.prototype(r, k, &mut exact, &mut truth) else {
                    missing += usize::from(look.exists());
                    continue;
                };
                ratios = [ratios[0].max(step.ratio), ratios[1].max(s.columns.iter().map(|x| x.r).fold(0.0, f64::max))];
                // Relative to the column's whole response (the readings' unit): an explicit column's own unit with it.
                for (j, &(_, x)) in truth.columns(ground).iter().enumerate() {
                    most[j] = most[j].max(x);
                    let unit = if j >= ground { 1.0 } else { 0.0 };
                    if s.columns[j].exists && x + unit > 0.0 {
                        points = points.max((s.columns[j].x - x).abs() / (x + unit));
                    }
                }
            }
            // The cells' largest per column against 41 points of the forward.
            for k in 0..=40 {
                let (r, kk) = path(cell * k as f64 / 40.0);
                if fold.prototype(r, kk, &mut exact, &mut truth).is_some() {
                    for (j, &(_, x)) in truth.columns(ground).iter().enumerate() {
                        most[j] = most[j].max(x);
                    }
                }
            }
            for (j, &m) in most.iter().enumerate() {
                let unit = if j >= ground { 1.0 } else { 0.0 };
                if m + unit > 0.0 {
                    largest = largest.max((look.largest(|s| s.columns[j].x) - m).abs() / (m + unit));
                }
            }
            // The profiles (§2.7).
            series = series.max(look.largest(|s| (0..n).map(|j| s.series(j, 2)).fold(0.0, f64::max)));
            for k in 0..look.seen.len() {
                for j in 0..n {
                    for b in 0..look.seen[k].near.len() {
                        centred = centred.max(look.centred(k, b, j));
                    }
                    for x in 0..look.seen[k].far.len() {
                        centred = centred.max(look.centred_far(k, x, j));
                    }
                }
            }
            for s in &look.seen {
                for (b, inputs) in s.near.iter().enumerate().filter(|x| !x.1.kappa.is_empty()) {
                    for j in 0..n {
                        nearest = nearest.min(inputs.nearest[j]).min(s.distance(b, j, inputs.kappa[0]));
                    }
                }
                chirp = chirp.max(s.chirp.iter().copied().fold(0.0, f64::max));
                drift = drift.max(s.near.iter().map(|x| x.drift).sum::<f64>());
            }
            middle = middle.max(look.at(0.5 * cell, |s| s.columns.iter().map(|x| x.x).fold(0.0, f64::max)));
        }
        samples.sort();
        samples.dedup();
        let decided = cells - absent;
        println!(
            "R4_LIN_4 set {set:?} (n {n}): {decided} of {cells} decisions (samples {samples:?}); the held part ‖O‖/‖V + O‖ {held:.2e}; the look's x against the forward's at its points {points:.2e}, the cells' largest {largest:.2e}; the largest ratio: the forward's {:.3}, the look's {:.3}; points with no forward where the look exists {missing}; profiles: series at two passes {series:.2e}, centring {centred:.2e}, nearest distance {nearest:.2} rad/µs, chirp {chirp:.2e}, drift {drift:.2e}, x at the cells' middle {middle:.3}; work a cell: the look {:.0}, the decision's forward {:.0}",
            ratios[0], ratios[1], work / decided.max(1) as f64, forward / decided.max(1) as f64
        );
        assert!(points <= share, "set {set:?}: the look's excitations at its points within the share of the forward's");
        assert!(largest <= share, "set {set:?}: the cells' largest excitations within the share");
        checked += 1;
    }
    assert!(checked > 0, "a multilevel set");
}
