//! The fold's named tests (DESIGN §8, step 1: ① – ⑩). Small systems are real 32-state requests with the idle states
//! uncoupled (onefold 7); against a Floquet problem the mass is made 10⁶ times larger and κ = 0, so that no Doppler shift
//! enters either side; the exact problem is the truncated Floquet lattice of the request itself, its offsets keyed by
//! integer coordinates on the components, diagonalized whole.
use super::*;
use crate::request::tests::sample;
use crate::request::{Beam, OutsideForm};
use std::collections::{HashMap, VecDeque};

const GAMMA: f64 = 38.1;

/// The rule's choice of a request (`Choice::of`) at the model's share.
fn rule(request: &Request) -> Choice {
    Choice::of(request, super::super::elim::share()).unwrap()
}

/// Every node folded, every copy centred, at the frames `frames`, the rule's sets and parts.
fn folded(request: &Request, frames: Vec<Beat>) -> Choice {
    Choice { frames, ..rule(request) }
}

/// The fold at a point on a fresh path (its decompositions made there), its jumps and its readings.
fn fold_at(fold: &Fold, r: [f64; 3], kappa: [f64; 3]) -> Option<Folded> {
    let mut f = fold.at(r, kappa, &mut Path::default())?;
    fold.jumps(&mut f);
    fold.read(&mut f, 0.0);
    Some(f)
}

/// `sylvester` with T_K's own diagonal.
fn syl(tk: &Mat, ends: &[usize], th: &Mat, w: f64, b: &Mat, work: &mut f64) -> Mat {
    let diag: Vec<C64> = (0..tk.rows).map(|i| tk[(i, i)]).collect();
    sylvester(tk, &diag, ends, th, w, b, work)
}

/// A beam of a synthetic system: its frequency axis, ν (rad/µs, its detuning from the excited frame), its direction, its
/// couplings (excited state less GROUND, ground state, value).
struct Ray {
    axis: usize,
    nu: f64,
    direction: [f64; 3],
    couplings: Vec<(usize, usize, C64)>,
}

/// A synthetic request: the listed ground states (state, energy) and excited states (state less GROUND, energy, Γ, landing
/// state) of the 32, the others idle at zero; the beams (the first the trap, a Gaussian of waist 10⁴ µm: a plane wave about
/// the focus), each frequency's field along π, no counter-rotating coupling; `axes` frequency axes (beyond the beams' ones a
/// gauge's); the mass 10⁶ times Rb87's where `heavy`.
fn system(ground: &[(usize, f64)], excited: &[(usize, f64, f64, usize)], rays: &[Ray], axes: usize, heavy: bool) -> Request {
    let mut r = sample();
    r.frequencies_mhz = vec![0.0; axes];
    for b in rays {
        r.frequencies_mhz[b.axis] = b.nu / crate::units::RAD_PER_US_PER_MHZ;
    }
    let used = rays.iter().map(|b| b.axis + 1).max().unwrap();
    r.frequencies_mhz.iter_mut().skip(used).enumerate().for_each(|(i, x)| *x = 1.2345 + i as f64);
    r.hamiltonian = Mat::zeros(STATES, STATES);
    for &(s, e) in ground {
        r.hamiltonian[(s, s)] = C64::real(e);
    }
    for &(x, e, _, _) in excited {
        r.hamiltonian[(GROUND + x, GROUND + x)] = C64::real(e);
    }
    r.emission = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
    for &(x, _, gamma, land) in excited {
        r.emission[x % 3][(land, GROUND + x)] = C64::real(gamma.sqrt());
    }
    r.beams = rays
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mut coupling = Mat::zeros(STATES, STATES);
            for &(x, g, v) in &b.couplings {
                coupling[(GROUND + x, g)] = v;
            }
            let mut index = vec![0; axes];
            index[b.axis] = 1;
            Beam {
                id: format!("b{i}"),
                name: format!("光{i}"),
                direction: b.direction,
                wave_number: 8.05,
                waist: (i == 0).then_some(1e4),
                frequency_index: index,
                coupling,
                counter_coupling: Mat::zeros(STATES, STATES),
                field: [C64::ZERO, C64::real(1e3), C64::ZERO],
            }
        })
        .collect();
    let mut indices: Vec<Vec<i32>> = r.beams.iter().map(|b| b.frequency_index.clone()).collect();
    indices.sort();
    indices.dedup();
    r.outside_forms = indices.into_iter().map(|frequency_index| OutsideForm { frequency_index, matrices: std::array::from_fn(|_| Mat::zeros(STATES, STATES)), transitions: vec![] }).collect();
    r.trap_beam = 0;
    if heavy {
        r.mass_kg *= 1e6;
    }
    r.validate().unwrap();
    r
}

/// The lab frames: every ground state at zero.
fn lab(request: &Request) -> Vec<Beat> {
    vec![Beat::whole(vec![0; request.frequencies_mhz.len() + 1]); GROUND]
}

/// A request's components' parts, in the fold's order.
fn parts(request: &Request) -> Vec<Part> {
    let light = Light::new(request);
    let axes = Axes::new(request).unwrap();
    components(request, &light, &axes).unwrap().iter().map(|c| c.part).collect()
}

/// Every copy at one status (`Choice::copies`): from every ground state and component, by every component, of both F.
fn all_copies(parts: &[Part], status: Status) -> Vec<((Part, usize, Part, u8), Status)> {
    let mut out = vec![];
    for &a in parts {
        for s in 0..GROUND {
            for &e in parts {
                for f in [1, 2] {
                    out.push(((a, s, e, f), status));
                }
            }
        }
    }
    out
}

/// The exact Floquet problem of a fold's request at r, truncated (module documentation): from the ground states at offset
/// zero, every hop of a component (absorbing to an excited state at +ν_c, emitting to a ground state at −ν_c) and every
/// static coupling (H_GG, 𝒦), to `depth` hops of the components `near` -- a hop of another makes a leaf, which joins the
/// nodes there are and makes no new ones. Its matrix: H − offset on the diagonal, the hops' couplings off it.
struct Lattice {
    nodes: Vec<(usize, Vec<i64>)>,
    m: Mat,
}

fn lattice(fold: &Fold, r: [f64; 3], depth: usize, near: &dyn Fn(usize) -> bool) -> Lattice {
    // The explicit states at their offsets seed it too (an idle one no hop reaches): in the lab frames, the offset of the
    // component that makes it explicit.
    let mut w = 0.0;
    let site = fold.site(r, [0.0; 3], &mut w);
    let nc = fold.components.len();
    let value = |k: &[i64]| k.iter().zip(&fold.components).map(|(&n, c)| n as f64 * c.nu).sum::<f64>();
    // A state's neighbours: (state, the component's hop, the coupling to it from the state).
    let neighbours = |s: usize| -> Vec<(usize, Option<(usize, i64)>, C64)> {
        let mut out = vec![];
        if s < GROUND {
            for t in (0..GROUND).filter(|&t| t != s && site.ground[(t, s)] != C64::ZERO) {
                out.push((t, None, site.ground[(t, s)]));
            }
            for (c, v) in site.drives.iter().enumerate() {
                for x in (0..EXCITED).filter(|&x| v[(x, s)] != C64::ZERO) {
                    out.push((GROUND + x, Some((c, 1)), v[(x, s)]));
                }
            }
        } else {
            let x = s - GROUND;
            for y in (0..EXCITED).filter(|&y| y != x && site.excited[(y, x)] != C64::ZERO) {
                out.push((GROUND + y, None, site.excited[(y, x)]));
            }
            for (c, v) in site.drives.iter().enumerate() {
                for g in (0..GROUND).filter(|&g| v[(x, g)] != C64::ZERO) {
                    out.push((g, Some((c, -1)), v[(x, g)].conj()));
                }
            }
        }
        out
    };
    let mut index: HashMap<(usize, Vec<i64>), usize> = HashMap::new();
    let mut nodes: Vec<(usize, Vec<i64>)> = vec![];
    let mut cost: Vec<(usize, bool)> = vec![];
    let mut queue = VecDeque::new();
    for s in 0..GROUND {
        index.insert((s, vec![0; nc]), nodes.len());
        nodes.push((s, vec![0; nc]));
        cost.push((0, false));
        queue.push_back(s);
    }
    for (j, &x) in fold.explicit.iter().enumerate() {
        let offset = fold.frame_values[fold.ground.len() + j];
        let c = (0..nc).min_by(|&a, &b| (fold.components[a].nu - offset).abs().total_cmp(&(fold.components[b].nu - offset).abs())).unwrap();
        let mut key = vec![0; nc];
        key[c] = 1;
        if index.contains_key(&(GROUND + x, key.clone())) {
            continue;
        }
        index.insert((GROUND + x, key.clone()), nodes.len());
        nodes.push((GROUND + x, key));
        cost.push((1, false));
        queue.push_back(nodes.len() - 1);
    }
    while let Some(i) = queue.pop_front() {
        let (s, k) = nodes[i].clone();
        let (here, leaf) = cost[i];
        if leaf {
            continue;
        }
        for (t, hop, _) in neighbours(s) {
            let mut key = k.clone();
            let (there, to_leaf) = match hop {
                None => (here, false),
                Some((c, d)) => {
                    key[c] += d;
                    if near(c) {
                        (here + 1, false)
                    } else {
                        (here, true)
                    }
                }
            };
            if index.contains_key(&(t, key.clone())) || there > depth {
                continue;
            }
            index.insert((t, key.clone()), nodes.len());
            nodes.push((t, key));
            cost.push((there, to_leaf));
            queue.push_back(nodes.len() - 1);
        }
    }
    let n = nodes.len();
    let mut m = Mat::zeros(n, n);
    for (i, (s, k)) in nodes.iter().enumerate() {
        m[(i, i)] = if *s < GROUND { site.ground[(*s, *s)] } else { site.excited[(s - GROUND, s - GROUND)] } - C64::real(value(k));
        for (t, hop, v) in neighbours(*s) {
            let mut key = k.clone();
            if let Some((c, d)) = hop {
                key[c] += d;
            }
            if let Some(&j) = index.get(&(t, key)) {
                m[(j, i)] += v;
            }
        }
    }
    Lattice { nodes, m }
}

/// The `count` eigenvalues of a matrix with the largest weight on the nodes `p0`, with their weights.
fn p_like(m: &Mat, p0: &[usize], count: usize) -> Vec<(C64, f64)> {
    let e = crate::linalg::eigen(m).expect("the lattice's eigendecomposition");
    let mut w: Vec<(C64, f64)> = (0..m.rows)
        .map(|k| {
            let all: f64 = (0..m.rows).map(|i| e.vectors[(i, k)].norm_sqr()).sum();
            (e.values[k], p0.iter().map(|&i| e.vectors[(i, k)].norm_sqr()).sum::<f64>() / all)
        })
        .collect();
    w.sort_by(|a, b| b.1.total_cmp(&a.1));
    w.truncate(count);
    w
}

/// The exact lattice's P-like eigenvalues: of the ground states at offset zero and each explicit state at its offset.
fn exact(fold: &Fold, l: &Lattice) -> Vec<(C64, f64)> {
    let mut p0: Vec<usize> = (0..GROUND).collect();
    for (j, &x) in fold.explicit.iter().enumerate() {
        let offset = fold.frame_values[fold.ground.len() + j];
        let value = |k: &[i64]| k.iter().zip(&fold.components).map(|(&n, c)| n as f64 * c.nu).sum::<f64>();
        if let Some(i) = l.nodes.iter().position(|(s, k)| *s == GROUND + x && (value(k) - offset).abs() <= 1e-9 * (1.0 + offset.abs())) {
            p0.push(i);
        }
    }
    let count = p0.len();
    p_like(&l.m, &p0, count)
}

/// The fold's own P-Floquet problem: G − offset on the diagonal, each tone e^{iΩt}G_Ω a hop to the offset less Ω, to `depth`
/// hops; its eigenvalues of the largest weight at offset zero.
fn p_floquet(f: &Folded, depth: usize) -> Vec<(C64, f64)> {
    let n = f.n;
    let mut keys: Vec<(f64, usize)> = vec![(0.0, 0)];
    let mut at = 0;
    while at < keys.len() {
        let (v, level) = keys[at];
        if level < depth {
            for t in &f.tones {
                let value = v - t.value;
                if !keys.iter().any(|(u, _)| (u - value).abs() <= 1e-9 * (1.0 + value.abs())) {
                    keys.push((value, level + 1));
                }
            }
        }
        at += 1;
    }
    let size = keys.len() * n;
    let mut m = Mat::zeros(size, size);
    for (a, &(va, _)) in keys.iter().enumerate() {
        for p in 0..n {
            for q in 0..n {
                m[(a * n + p, a * n + q)] = f.generator[(p, q)];
            }
            m[(a * n + p, a * n + p)] -= C64::real(va);
        }
        for t in &f.tones {
            let target = va - t.value;
            if let Some(b) = keys.iter().position(|(v, _)| (v - target).abs() <= 1e-9 * (1.0 + target.abs())) {
                for p in 0..n {
                    for q in 0..n {
                        m[(b * n + p, a * n + q)] += t.generator[(p, q)];
                    }
                }
            }
        }
    }
    p_like(&m, &(0..n).collect::<Vec<_>>(), n)
}

/// Per eigenvalue of the first set its distance to the nearest of the second.
fn apart(a: &[(C64, f64)], b: &[(C64, f64)]) -> Vec<f64> {
    a.iter().map(|(x, _)| b.iter().map(|(y, _)| (*x - *y).abs()).fold(f64::INFINITY, f64::min)).collect()
}

/// The readings' ρ² summed over the nodes, of the kinds `kinds` picks.
fn read(f: &Folded, kinds: &dyn Fn(&Reading) -> f64) -> f64 {
    f.readings.iter().map(kinds).sum()
}

/// What the readings of the kinds `kinds` allow an eigenvalue to move (module documentation of `Reading`): an excited
/// node's ρ² is a relative error of the excited amplitude, on the fold's part of the generator, rate(ρ)·‖H⁽¹⁾ − Ĥ_PP‖_F; a
/// copy's is an absolute error δc of its amplitude per unit P amplitude, which the couplings that make and take it -- its
/// emitting component's, v_g its largest -- carry into the generator to first order twice, 2 v_g δc, summed over the copies.
fn eigenvalue_bound(f: &Folded, kinds: &dyn Fn(&Reading) -> f64) -> f64 {
    let excited: f64 = f.readings.iter().filter(|r| matches!(r.node, Node::Excited(_))).map(kinds).sum();
    let copies: f64 = f.readings.iter().filter_map(|r| if let Node::Copy { component, .. } = r.node { Some(2.0 * f.site.drives[component].max_abs() * kinds(r).sqrt()) } else { None }).sum();
    rate(excited.sqrt()) * fold_part(f) + copies
}

/// The fold's part of the generator: its last Bloch generator less Ĥ_PP, ‖H − Ĥ_PP‖_F.
fn fold_part(f: &Folded) -> f64 {
    let mut d = f.h[2].clone();
    d.add_scaled(-C64::ONE, &f.hpp);
    d.frobenius()
}

/// A two-level root of [[a, g], [g, d]]: the one nearer a.
fn root(a: C64, d: C64, g: f64) -> C64 {
    let half = (a - d) * 0.5;
    let disc = (half * half + C64::real(g * g)).sqrt();
    let (x, y) = ((a + d) * 0.5 + disc, (a + d) * 0.5 - disc);
    if (x - a).abs() < (y - a).abs() {
        x
    } else {
        y
    }
}

/// ① A two-level atom (Γ = 38.1 rad/µs, Δ = −4Γ) at saturations s = 2Ω²/Γ² from 0.1 to 10: its dressed ground state's
/// eigenvalue, the root of [[0, Ω/2], [Ω/2, −Δ − iΓ/2]] -- its light shift and its rate to all orders in Ω -- within the
/// fold's read next pass (rate units, of the eigenvalue). And moving (κ along the beam, kv = ±Γ/2): its rate is the
/// analytic one at Δ − w, w = (ħ/m)(kκ + k²/2) -- the excited state raised by w --, not at Δ + w; red-detuned, an atom moving
/// against the beam scatters more (cooling).
#[test]
fn a_two_level_atom_s_shift_and_rate_are_the_analytic_ones_and_doppler_takes_minus_w() {
    let delta = -4.0 * GAMMA;
    let exact = |delta: f64, g: f64| root(C64::ZERO, C64::new(-delta, -0.5 * GAMMA), g);
    for heavy in [true, false] {
        for s in [0.1, 1.0, 3.0, 10.0] {
            let g = 0.5 * GAMMA * (s / 2.0_f64).sqrt();
            let req = system(&[(0, 0.0)], &[(17, 0.0, GAMMA, 0)], &[Ray { axis: 0, nu: delta, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(g))] }], 1, heavy);
            let fold = Fold::new(&req, &folded(&req, lab(&req)), 0).unwrap();
            if heavy {
                let f = fold_at(&fold, [0.0; 3], [0.0; 3]).unwrap();
                let (lambda, exact) = (f.generator[(0, 0)], exact(delta, g));
                let bound = rate(read(&f, &|r| r.next).sqrt()) * exact.abs() + 1e-12 * exact.abs();
                println!("s {s:5.1}: fold {lambda:.9?} exact {exact:.9?} apart {:.2e} bound {bound:.2e} (x {:.4}, ratio {:.4})", (lambda - exact).abs(), f.excitation, f.ratio);
                assert!((lambda - exact).abs() <= bound, "s {s}: {lambda:?} against {exact:?}");
                continue;
            }
            let hm = crate::units::hbar_over_mass(req.mass_kg);
            let k = 8.05;
            let mut rates = vec![];
            for sign in [1.0, -1.0] {
                let kappa = sign * 0.5 * GAMMA / (hm * k);
                let w = hm * (k * kappa + 0.5 * k * k);
                let f = fold_at(&fold, [0.0; 3], [0.0, 0.0, kappa]).unwrap();
                let ours = -2.0 * f.generator[(0, 0)].im;
                let (minus, other) = (-2.0 * exact(delta - w, g).im, -2.0 * exact(delta + w, g).im);
                let bound = rate(read(&f, &|r| r.next).sqrt()) * minus + 1e-12 * minus;
                println!("s {s:5.1}, kv {:+.2}: rate {ours:.6} against Δ − w {minus:.6} and Δ + w {other:.6}", sign * 0.5 * GAMMA);
                assert!((ours - minus).abs() <= bound && (ours - other).abs() > 10.0 * bound, "s {s}: the Doppler shift's sign");
                rates.push(ours);
            }
            assert!(rates[1] > rates[0], "red detuning: the atom moving against the beam scatters more");
        }
    }
}

/// The Λ of lam.py in this code's convention (onefold 7: ν₁ = 0, ν₂ = +Ω, E_e = 3072, E_g2 = d₀ − Ω; Γ = 38.1): g1 = ground
/// state 0, g2 = ground state 4, e = excited state 17; beam 1 couples g1 by g11 and g2 by g21, beam 2 g1 by g12 and g2 by g22.
fn lambda_system(g11: f64, g12: f64, g21: f64, g22: f64, d0: f64, om: f64) -> Request {
    let legs = |a: f64, b: f64| -> Vec<(usize, usize, C64)> { [(17, 0, a), (17, 4, b)].into_iter().filter(|x| x.2 != 0.0).map(|(x, g, v)| (x, g, C64::real(v))).collect() };
    system(
        &[(0, 0.0), (4, d0 - om)],
        &[(17, 3072.0, GAMMA, 0)],
        &[Ray { axis: 0, nu: 0.0, direction: [0.0, 0.0, 1.0], couplings: legs(g11, g21) }, Ray { axis: 1, nu: om, direction: [0.0, 0.0, 1.0], couplings: legs(g12, g22) }],
        2,
        true,
    )
}

/// The eigenvalue of a fold's static generator with the largest weight on P place 0 (g1).
fn g1_like(f: &Folded) -> C64 {
    let e = crate::linalg::eigen(&f.generator).unwrap();
    let k = (0..f.n).max_by(|&a, &b| e.vectors[(0, a)].norm_sqr().total_cmp(&e.vectors[(0, b)].norm_sqr())).unwrap();
    e.values[k]
}

/// ② lam.py's five rows (three closed ladders, two open with cross legs) against the exact Floquet lattice (to 24 hops):
/// with the copies folded in the block solve -- in the lab frames and in the mixing tree's (the Raman frame) -- the g1-like
/// eigenvalue's shift and rate are the exact ones within the read next pass and truncation, the Raman static shift and the
/// rate's interference 2U₂/δ in them; with the copies left out (the replica alone, the old run) the closed ladders' rate
/// misses by about 2U₂/δ.
#[test]
fn the_lambda_rows_are_the_exact_floquet_ones_with_the_copies_folded() {
    let rows = [(60.0, 0.0, 0.0, 50.0, 8.0, 400.0), (60.0, 0.0, 0.0, 50.0, 3.0, 400.0), (90.0, 0.0, 0.0, 80.0, 6.0, 400.0), (60.0, 18.0, 15.0, 50.0, 8.0, 400.0), (60.0, 60.0, 50.0, 50.0, 8.0, 60.0)];
    for (g11, g12, g21, g22, d0, om) in rows {
        let req = lambda_system(g11, g12, g21, g22, d0, om);
        let ps = parts(&req);
        let base = Fold::new(&req, &folded(&req, lab(&req)), 0).unwrap();
        let l = lattice(&base, [0.0; 3], 24, &|_| true);
        let g1 = p_like(&l.m, &[0], 1)[0].0;
        let u2 = g22 * g22 / -3072.0;
        let delta = d0 + u2 - g11 * g11 / -3072.0;
        println!("== row ({g11}, {g12}, {g21}, {g22}, d0 {d0}, Ω {om}): exact {g1:.6?}, rate {:.6e}, 2U₂/δ {:+.4}", -2.0 * g1.im, 2.0 * u2 / delta);
        for (name, frames, status) in [("lab, decomposed", lab(&req), Status::Decomposed), ("mixing, decomposed", rule(&req).frames, Status::Decomposed), ("lab, copies left out", lab(&req), Status::Omitted)] {
            let fold = Fold::new(&req, &Choice { frames, explicit: vec![], copies: all_copies(&ps, status), ..rule(&req) }, 0).unwrap();
            let f = fold_at(&fold, [0.0; 3], [0.0; 3]).unwrap();
            let v = g1_like(&f);
            let (shift, rr) = ((v.re - g1.re) / g1.re, (v.im - g1.im) / g1.im);
            let bound = rate(read(&f, &|r| r.next + r.truncation).sqrt()) * fold_part(&f) / v.abs();
            println!("   {name:22}: {v:.6?}  shift {shift:+.3e}  rate {rr:+.3e}  read {bound:.2e} (x {:.4}, ratio {:.4}, fold part {:.3})", f.excitation, f.ratio, fold_part(&f));
            if std::env::var("FOLD_READINGS").is_ok() {
                for r in &f.readings {
                    println!("      block {} {:?}: next {:.2e} truncation {:.2e} order {:.2e} sideband {:.2e} mixing {:.2e}", r.block, r.node, r.next, r.truncation, r.order, r.sideband, r.mixing);
                }
            }
            if status == Status::Omitted {
                // The copies' omission is read (their dressing, the evaluation order's reading of a left-out copy).
                let read_out = rate(read(&f, &|r| r.next + r.truncation + r.order).sqrt()) * fold_part(&f) / v.abs();
                println!("   {:22}  the omission read: {read_out:.2e}", "");
                assert!(shift.abs() <= read_out && rr.abs() <= read_out, "the left-out copies are read");
                if g12 == 0.0 {
                    assert!(rr.abs() > 0.5 * (2.0 * u2 / delta).abs(), "without the copies the rate misses its interference");
                }
            } else {
                assert!(shift.abs() <= bound + 1e-9 && rr.abs() <= bound + 1e-9, "{name}: the fold against the exact lattice");
            }
        }
    }
}

/// A random small system (as defn's sambe_tests): two or three ground states (energies within ±20 rad/µs), one to three
/// excited ones (±20, Γ = 38.1, each landing in one of them), two beams of frequencies ±(80 … 580) rad/µs with couplings of
/// 5 … 40 rad/µs at random phases (each with probability 0.6); the copies decomposed; in a third of the trials the first
/// excited state's level explicit at a component that drives it.
fn random_system(state: &mut u64, trial: usize) -> (Request, Choice) {
    let mut u = || {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let grounds = [0usize, 4, 6];
    let exciteds = [17usize, 13, 9];
    let g = 2 + (u() * 2.0) as usize;
    let e = 1 + (u() * 3.0) as usize;
    let ground: Vec<(usize, f64)> = (0..g).map(|i| (grounds[i], 40.0 * u() - 20.0)).collect();
    let excited: Vec<(usize, f64, f64, usize)> = (0..e).map(|i| (exciteds[i], 40.0 * u() - 20.0, GAMMA, grounds[(u() * g as f64) as usize])).collect();
    let rays: Vec<Ray> = (0..2)
        .map(|axis| {
            let nu = (if u() < 0.5 { -1.0 } else { 1.0 }) * (80.0 + 500.0 * u());
            let mut couplings = vec![];
            for x in 0..e {
                for y in 0..g {
                    if u() < 0.6 {
                        couplings.push((exciteds[x], grounds[y], C64::cis(6.283 * u()) * (5.0 + 35.0 * u())));
                    }
                }
            }
            if couplings.is_empty() {
                couplings.push((exciteds[0], grounds[0], C64::real(20.0)));
            }
            Ray { axis, nu, direction: [0.0, 0.0, 1.0], couplings }
        })
        .collect();
    let req = system(&ground, &excited, &rays, 2, true);
    let ps = parts(&req);
    let mut choice = Choice { frames: lab(&req), explicit: vec![], copies: all_copies(&ps, Status::Decomposed), ..rule(&req) };
    if trial % 3 == 0 {
        if let Some((a, y)) = (0..rays.len()).flat_map(|a| (0..g).map(move |y| (a, y))).find(|&(a, y)| rays[a].couplings.iter().any(|c| c.0 == exciteds[0] && c.1 == grounds[y])) {
            let fold = Fold::new(&req, &folded(&req, lab(&req)), 0).unwrap();
            choice.explicit.push((fold.excited_level[exciteds[0]], Part { frequency: a, counter: false }, grounds[y]));
        }
    }
    (req, choice)
}

/// R4_LIN_4 (the mass 10⁶ times larger where `heavy`) with its D2 F′ = 2 explicit at the repump's offset from F = 1, its
/// near copies (of the cooling and the repump, co-rotating) decomposed and the others centred.
fn r4(heavy: bool) -> (Request, Choice, Part, Part) {
    let mut req = crate::model::testing::scene("R4_LIN_4");
    if heavy {
        req.mass_kg *= 1e6;
    }
    let light = Light::new(&req);
    let frequency_of = |id: &str| light.frequencies.iter().position(|f| f.beams.iter().any(|&l| req.beams[l].id == id)).unwrap();
    let (cool, repump) = (Part { frequency: frequency_of("cool-plus"), counter: false }, Part { frequency: frequency_of("repump-plus"), counter: false });
    let fold = Fold::new(&req, &folded(&req, lab(&req)), 0).unwrap();
    let level = fold.levels.iter().position(|l| *l == (Manifold::D2, 2)).unwrap();
    let ps = parts(&req);
    let near = |p: &Part| *p == cool || *p == repump;
    let mut copies = vec![];
    for &a in &ps {
        for s in 0..GROUND {
            for &e in &ps {
                for f in [1, 2] {
                    copies.push(((a, s, e, f), if near(&a) && near(&e) { Status::Decomposed } else { Status::Centred }));
                }
            }
        }
    }
    let choice = Choice { frames: lab(&req), explicit: vec![(level, repump, 0)], copies, ..rule(&req) };
    (req, choice, cool, repump)
}

/// ③ The decay is the jumps' products to the rounding (DESIGN §2.6), explicit rows with them: on 20 random systems and at a
/// point of R4_LIN_4 (its D2 F′ = 2 explicit at the repump's offset), i(G − G†) = Σ_o Σ_q J_oq†J_oq within 10⁻¹² of their
/// size.
#[test]
fn the_decay_is_the_jumps_products_to_rounding_with_explicit_rows() {
    let check = |f: &Folded, what: &str| {
        let n = f.n;
        let mut gap = Mat::from_fn(n, n, |i, j| C64::I * (f.generator[(i, j)] - f.generator[(j, i)].conj()));
        let mut products = Mat::zeros(n, n);
        for j in &f.jumps {
            for q in j {
                products.add_scaled(C64::ONE, &q.adjoint_product(q));
            }
        }
        let scale = f.generator.frobenius().max(products.frobenius());
        gap.add_scaled(-C64::ONE, &products);
        let gap = if scale > 0.0 { gap.max_abs() / scale } else { gap.max_abs() };
        assert!(gap <= 1e-12, "{what}: the decay against the jumps' products, {gap:.2e} of the scale");
        gap
    };
    let mut state = 3u64;
    let mut worst = 0.0_f64;
    for trial in 0..20 {
        let (req, choice) = random_system(&mut state, trial);
        for set in Fold::sets(&req, &choice).unwrap() {
            let f = fold_at(&Fold::new(&req, &choice, set[0]).unwrap(), [0.0; 3], [0.0; 3]).unwrap();
            worst = worst.max(check(&f, &format!("trial {trial}, set {set:?}")));
        }
    }
    let (req, choice, ..) = r4(false);
    for set in Fold::sets(&req, &choice).unwrap() {
        let f = fold_at(&Fold::new(&req, &choice, set[0]).unwrap(), [0.1, -0.2, 0.3], [0.0; 3]).unwrap();
        worst = worst.max(check(&f, &format!("R4_LIN_4, set {set:?}")));
    }
    println!("the decay against the jumps' products: worst {worst:.2e} of the scale");
}

/// ④ R4_LIN_4 at a point against its 32-state Floquet problem (onefold 7: near hops of the cooling and the repump to depth
/// three, every other component's a leaf): the 13 P-like eigenvalues -- the explicit F′ = 2's dressed ones with them, its own
/// decay and its couplings K_Qx in the one formula, the copies folded -- each within the read next pass, truncation and
/// evaluation order of the fold's part.
#[test]
fn r4_lin_4_at_a_point_is_its_32_state_floquet() {
    let (req, choice, cool, repump) = r4(true);
    println!("R4_LIN_4's state sets: {:?}", Fold::sets(&req, &choice).unwrap());
    for state in [0, 4] {
    let fold = Fold::new(&req, &choice, state).unwrap();
    let at = [0.1, -0.2, 0.3];
    let f = fold_at(&fold, at, [0.0; 3]).unwrap();
    let near = |c: usize| fold.components[c].part == cool || fold.components[c].part == repump;
    let l = lattice(&fold, at, 3, &near);
    let ex = exact(&fold, &l);
    let ours = p_floquet(&f, 2);
    let gaps = apart(&ours, &ex);
    let bound = eigenvalue_bound(&f, &|r| r.next + r.truncation + r.order);
    println!("R4_LIN_4, the set of {:?}: {} lattice nodes, {} blocks, n = {}, x {:.4}, ratio {:.4}; largest gap {:.3e} rad/µs, bound {bound:.3e}; {:.0} complex multiply-adds", fold.ground, l.nodes.len(), fold.blocks.len(), f.n, f.excitation, f.ratio, gaps.iter().fold(0.0_f64, |a, &b| a.max(b)), f.work);
    for ((v, w), g) in ours.iter().zip(&gaps) {
        println!("   {v:.5?} (P weight {w:.3}): {g:.3e}");
    }
    let mut readings: Vec<&Reading> = f.readings.iter().collect();
    readings.sort_by(|a, b| (b.next + b.truncation + b.order).total_cmp(&(a.next + a.truncation + a.order)));
    for r in readings.iter().take(4) {
        println!("   block {} ({:.1} rad/µs) {:?}: next {:.2e} truncation {:.2e} order {:.2e} mixing {:.2e}", r.block, fold.blocks[r.block].value, r.node, r.next, r.truncation, r.order, r.mixing);
    }
    assert!(gaps.iter().all(|&g| g <= bound + 1e-9), "the fold against the 32-state Floquet problem");
    }
}

/// ④'s reference against itself one hop deeper (not a test: the convergence of the 32-state Floquet problem it is compared
/// with): per set the P-like eigenvalues at depth three and four, their largest change, and the fold's gaps to each.
#[test]
#[ignore]
fn r4_lin_4_s_lattice_one_hop_deeper() {
    let (req, choice, cool, repump) = r4(true);
    for state in [0, 4] {
        let fold = Fold::new(&req, &choice, state).unwrap();
        let point = [0.1, -0.2, 0.3];
        let f = fold_at(&fold, point, [0.0; 3]).unwrap();
        let near = |c: usize| fold.components[c].part == cool || fold.components[c].part == repump;
        let ours = p_floquet(&f, 2);
        let largest = |v: Vec<f64>| v.iter().fold(0.0_f64, |a, &b| a.max(b));
        let (l3, l4) = (lattice(&fold, point, 3, &near), lattice(&fold, point, 4, &near));
        let (e3, e4) = (exact(&fold, &l3), exact(&fold, &l4));
        println!("the set of {:?}: {} and {} nodes; depth 3 to 4 moves the P-like eigenvalues by {:.3e} rad/µs; the fold's largest gap {:.3e} to depth 3, {:.3e} to depth 4", fold.ground, l3.nodes.len(), l4.nodes.len(), largest(apart(&e4, &e3)), largest(apart(&ours, &e3)), largest(apart(&ours, &e4)));
    }
}

/// The exact copies after the excited block (fold.rs, # The exact copies): on random systems with their copies exact, each
/// block's column solve (c − K)z = r and its adjoint against a dense LU of the whole block as the decomposition holds it
/// (the couplings between the lines left out), at energies about the block's: equal to the rounding times the block's
/// condition.
#[test]
fn the_exact_copies_after_the_excited_block_are_the_whole_block_s_solve() {
    let mut state = 77u64;
    let (mut checked, mut worst) = (0, 0.0_f64);
    for trial in 0..20 {
        let (req, choice) = random_system(&mut state, trial);
        for set in Fold::sets(&req, &choice).unwrap() {
            let fold = Fold::new(&req, &choice, set[0]).unwrap();
            let Some(f) = fold_at(&fold, [0.0; 3], [0.0; 3]) else { continue };
            for (o, e) in f.eliminated.iter().enumerate() {
                if e.exact == 0 {
                    continue;
                }
                let b = &fold.blocks[o];
                let m = e.solved;
                // K in the basis, the whole block's: V†K V with the lines' couplings left out (as decomposed and read).
                let mut w = 0.0;
                let l = &f.laid[o];
                let eff = fold.effective(o, l, &mut w).unwrap();
                let mut k = eff.k.clone();
                for i in 0..m {
                    k[(i, i)] -= C64::real(b.value);
                }
                let kt = e.to(&e.to(&k, &mut w).adjoint(), &mut w).adjoint();
                for (s, c) in [(0, C64::new(3.0, 0.5)), (1, C64::new(-40.0, -2.0)), (2, C64::new(0.7, 9.0))] {
                    let r: Vec<C64> = (0..m).map(|i| C64::new(1.0 + i as f64 + s as f64, 0.5 - 0.1 * i as f64)).collect();
                    for adjoint in [false, true] {
                        let ours = e.column(c, &r, adjoint, &mut w);
                        let a = Mat::from_fn(m, m, |i, j| if i == j { c } else { C64::ZERO } - if adjoint { kt[(j, i)].conj() } else { kt[(i, j)] });
                        let mut x = r.clone();
                        crate::linalg::Lu::new(&a).unwrap().solve(&mut x);
                        let gap = ours.iter().zip(&x).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max) / x.iter().map(|q| q.abs()).fold(0.0, f64::max);
                        worst = worst.max(gap);
                        checked += 1;
                    }
                }
            }
        }
    }
    println!("{checked} exact-copy solves against the whole block's LU: worst {worst:.2e} relative");
    assert!(checked >= 20 && worst <= 1e-10, "{checked} solves, worst {worst:.2e}");
}

/// The tweezer's trap from the fold (replan, guard 5): R4_LIN_4's request with its trap light alone (the other beams'
/// couplings and outside forms zero, as the prepared trap takes it), the fold's potential of the reference state,
/// V(r) = Re G_ss about the old prepared trap's centre: its curvature along each lab axis (ω = √(V″ ħ/m)) and its depth
/// against 10 µm off, against the old engine's prepared trap (`Model::closed_at`, itself judged against the references'
/// trap frequencies by trap_tests) within 1 %; and in the whole scene the tweezer's blocks are not left out: their
/// omission reads beyond the share in the potential's unit though their rates' is nothing.
#[test]
fn the_tweezer_s_trap_is_the_prepared_one_and_its_blocks_stay() {
    use crate::model::Model;
    let full = crate::model::testing::scene("R4_LIN_4");
    let mut req = full.clone();
    let trap_index = req.beams[req.trap_beam].frequency_index.clone();
    for beam in req.beams.iter_mut().filter(|b| b.frequency_index != trap_index) {
        beam.coupling = Mat::zeros(STATES, STATES);
        beam.counter_coupling = Mat::zeros(STATES, STATES);
    }
    for form in req.outside_forms.iter_mut().filter(|f| f.frequency_index != trap_index) {
        form.matrices = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
    }
    let chief = (0..GROUND).max_by(|&a, &b| full.initial_density[(a, a)].re.total_cmp(&full.initial_density[(b, b)].re)).unwrap();
    let (model, _, trap, _) = Model::closed_at(&full, &Light::new(&full), vec![], chief, full.temperature_uk);
    let mut p = [0.0; GROUND];
    p[chief] = 1.0;
    let far = [trap.centre[0], trap.centre[1], trap.centre[2] + 10.0];
    let depth_old = model.potential(far, &p, &model.prepared()) - model.potential(trap.centre, &p, &model.prepared());
    let fold = Fold::new(&req, &rule(&req), chief).unwrap();
    let place = fold.ground.iter().position(|&s| s == chief).unwrap();
    let v = |r: [f64; 3]| fold_at(&fold, r, [0.0; 3]).unwrap().generator[(place, place)].re;
    let c = trap.centre;
    let h = 0.01;
    let mut worst = 0.0_f64;
    for a in 0..3 {
        let (mut plus, mut minus) = (c, c);
        plus[a] += h;
        minus[a] -= h;
        let curvature = (v(plus) - 2.0 * v(c) + v(minus)) / (h * h);
        let omega = (curvature * fold.hbar_over_mass).max(0.0).sqrt();
        println!("axis {a}: the fold's ω {omega:.6} rad/µs, the prepared trap's {:.6} ({:+.2e} relative)", trap.lab[a], omega / trap.lab[a] - 1.0);
        worst = worst.max((omega / trap.lab[a] - 1.0).abs());
    }
    let depth = v(far) - v(c);
    println!("depth: the fold's {depth:.6} rad/µs, the prepared trap's {depth_old:.6} ({:+.2e} relative)", depth / depth_old - 1.0);
    // The tweezer's blocks in the whole scene: their omission reading in the potential's unit.
    let fold = Fold::new(&full, &rule(&full), chief).unwrap();
    let f = fold_at(&fold, c, [0.0; 3]).unwrap();
    let share = super::super::elim::share();
    let tweezer: Vec<usize> = (0..fold.blocks.len()).filter(|&o| fold.blocks[o].sources.iter().any(|&(_, k)| fold.components[k].beams.contains(&full.trap_beam))).collect();
    let read: f64 = f.readings.iter().filter(|r| tweezer.contains(&r.block) && matches!(r.node, Node::Excited(_))).map(|r| r.omission).sum();
    println!("the tweezer's blocks {tweezer:?}: their omission read {:.3e} in rate units (the share {share:.4})", rate(read.sqrt()));
    assert!(worst <= 1e-2 && (depth / depth_old - 1.0).abs() <= 1e-2, "the fold's trap against the prepared one");
    assert!(rate(read.sqrt()) > share, "the tweezer's blocks stay");
}

/// ⑤ Sixty random small systems (`random_system`): the fold's P-Floquet eigenvalues against the dense lattice's (its depth
/// grown from 4 by 2 until they stay within 10⁻¹⁰), each within the read next pass and truncation of the fold's part; the
/// gap over that bound printed -- below one, and how far.
#[test]
fn sixty_random_systems_are_their_dense_sambe_within_the_read_truncation() {
    let mut state = 20261115u64;
    let (mut compared, mut absent, mut ratios) = (0, 0, vec![]);
    for trial in 0..60 {
        let (req, choice) = random_system(&mut state, trial);
        for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        if fold.blocks.is_empty() {
            continue;
        }
        let Some(f) = fold_at(&fold, [0.0; 3], [0.0; 3]) else { continue };
        if !(f.excitation < 1.0 && f.ratio < 1.0) {
            absent += 1;
            continue;
        }
        let mut depth = 4;
        // The reference itself: a lattice whose eigenvectors are singular (an exactly degenerate, defective spectrum) is
        // no reference -- counted beside the systems without a fold.
        let l0 = lattice(&fold, [0.0; 3], depth, &|_| true);
        if crate::linalg::eigen(&l0.m).is_none() {
            println!("   trial {trial}: the dense lattice ({} nodes) does not decompose: no reference", l0.nodes.len());
            absent += 1;
            continue;
        }
        let mut ex = exact(&fold, &l0);
        loop {
            let l = lattice(&fold, [0.0; 3], depth + 2, &|_| true);
            if crate::linalg::eigen(&l.m).is_none() {
                println!("   trial {trial}: the lattice at depth {} ({} nodes) does not decompose; depth {depth} taken", depth + 2, l.nodes.len());
                break;
            }
            let next = exact(&fold, &l);
            let change = apart(&next, &ex).iter().fold(0.0_f64, |a, &b| a.max(b));
            ex = next;
            depth += 2;
            if change <= 1e-10 * (1.0 + ex.iter().map(|x| x.0.abs()).fold(0.0, f64::max)) || depth >= 12 {
                break;
            }
        }
        let ours = p_floquet(&f, 2);
        let gap = apart(&ours, &ex).iter().fold(0.0_f64, |a, &b| a.max(b));
        let bound = eigenvalue_bound(&f, &|r| r.next + r.truncation) + 1e-9 * fold_part(&f);
        ratios.push(gap / bound);
        compared += 1;
        if gap > bound {
            for ((v, w), g) in ours.iter().zip(apart(&ours, &ex)) {
                println!("   trial {trial}: {v:.5?} (P weight {w:.3}): {g:.3e}");
            }
            for r in &f.readings {
                println!("   block {} ({:.1}) {:?}: next {:.2e} truncation {:.2e} order {:.2e} mixing {:.2e} centring {:.2e} omission {:.2e}", r.block, fold.blocks[r.block].value, r.node, r.next, r.truncation, r.order, r.mixing, r.centring, r.omission);
            }
            println!("   exact: {:?}", ex.iter().map(|x| x.0).collect::<Vec<_>>());
        }
        assert!(gap <= bound, "trial {trial}: the fold {gap:.3e} from the dense lattice, the read bound {bound:.3e} (x {:.3}, ratio {:.3})", f.excitation, f.ratio);
        }
    }
    ratios.sort_by(f64::total_cmp);
    println!("{compared} systems against the dense lattice, {absent} without a fold or a reference; gap over bound: median {:.3e}, largest {:.3e}", ratios[ratios.len() / 2], ratios[ratios.len() - 1]);
    assert!(compared >= 40, "{compared} compared");
}

/// ⑥ The closed Λ (g11 = 60, g22 = 50, Ω = 400; copies decomposed, lab frames) swept in its two-photon detuning from 40 to 0
/// rad/µs: the readings' total moves without a cliff (a step of 0.5 rad/µs changes it by less than (d/(d − ½))⁶, the
/// steepest power its parts make -- the next pass's δT, three resolvent factors of the copy at d, squared --, for d ≥ 2, at
/// the same number of passes; where a pass is added, the switch is within the share),
/// and passes the share before the fold stops existing (x ≥ 1, or the passes' ratio ≥ 1).
#[test]
fn a_lambda_swept_into_resonance_reads_beyond_the_share_before_its_fold_fails() {
    let share = super::super::elim::share();
    let (mut beyond, mut gone, mut last): (Option<f64>, Option<f64>, Option<(f64, f64, usize)>) = (None, None, None);
    for step in 0..=80 {
        let d0 = 40.0 - 0.5 * step as f64;
        let req = lambda_system(60.0, 0.0, 0.0, 50.0, d0, 400.0);
        let ps = parts(&req);
        // Both ground states in P whatever the rule makes of them (the reading, not the rule, is tested).
        let fold = Fold::new(&req, &Choice { frames: lab(&req), explicit: vec![], copies: all_copies(&ps, Status::Decomposed), sets: vec![vec![0, 4]], ..rule(&req) }, 0).unwrap();
        let f = fold_at(&fold, [0.0; 3], [0.0; 3]);
        let Some(f) = f.filter(|f| f.excitation < 1.0 && f.ratio < 1.0) else {
            gone.get_or_insert(d0);
            continue;
        };
        let total = f.readings.iter().map(|r| r.total()).fold(0.0_f64, f64::max);
        if total > share {
            beyond.get_or_insert(d0);
        }
        if let Some((d, t, passes)) = last {
            if d >= 2.0 && passes == f.passes {
                let limit = (d / (d - 0.5)).powi(6);
                assert!(total <= limit * t && t <= limit * total, "d₀ {d0}: the reading {total:.3e} after {t:.3e} (limit ×{limit:.2})");
            }
            if passes != f.passes {
                // A pass more (the next one's reading beyond the share): the switch moves the result within the share.
                println!("d₀ {d0}: {passes} to {} passes, the reading {t:.3e} to {total:.3e}", f.passes);
                assert!(t.min(total) <= share, "d₀ {d0}: a switch of the passes beyond the share");
            }
        }
        last = Some((d0, total, f.passes));
    }
    println!("beyond the share at d₀ = {beyond:?} rad/µs; the fold gone at {gone:?}");
    let (beyond, gone) = (beyond.expect("the reading passes the share"), gone.unwrap_or(-1.0));
    assert!(beyond > gone, "the reading passes the share before the fold stops existing");
}

/// ⑦ The adjoint's force (DESIGN §1.3) against the high-precision difference of the contracted generator: at a point of
/// R4_LIN_4 with explicit, decomposed, centred and kept nodes (one kept copy: F = 1 by the repump in the cooling's block from
/// ground state 4), f(r) = Re tr(ρ G(r)) + Σ_k Re tr(B_k†G_k(r)) for a random state's ρ and random tone cotangents B_k;
/// −∂f/∂r_a by the difference quotients' convergence (testing.rs `converged`), the adjoint within it -- the reference pass's
/// derivative and the centred copies' with it.
#[test]
fn the_adjoint_force_is_the_difference_of_the_contracted_generator() {
    let (req, mut choice, cool, repump) = r4(false);
    choice.copies.retain(|x| x.0 != (cool, 4, repump, 1));
    choice.copies.push(((cool, 4, repump, 1), Status::Kept));
    // As chosen, and with every D1 level centred (excited rows centred beside solved ones: every split's source dressed
    // through them).
    let levels = Fold::new(&req, &choice, 4).unwrap().levels.clone();
    let mut centred = choice.clone();
    for &a in &parts(&req) {
        for s in 0..GROUND {
            for (l, level) in levels.iter().enumerate() {
                if level.0 == Manifold::D1 {
                    centred.excited.push(((a, s, l), Status::Centred));
                }
            }
        }
    }
    for (what, choice) in [("as chosen", choice), ("every D1 level centred", centred)] {
        println!("{what}:");
        let fold = Fold::new(&req, &choice, 4).unwrap();
        let at = [0.3, -0.2, 0.5];
        let f0 = fold_at(&fold, at, [0.0; 3]).unwrap();
        let n = f0.n;
        let mut seed = 11u64;
        let mut u = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let psi: Vec<C64> = (0..n).map(|_| C64::new(u(), u())).collect();
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let rho = Mat::from_fn(n, n, |i, j| psi[i] * psi[j].conj() / norm);
        let cot: Vec<Mat> = f0.tones.iter().map(|_| Mat::from_fn(n, n, |_, _| C64::new(u(), u()) * 0.1)).collect();
        let value = |r: [f64; 3]| -> f64 {
            let f = fold_at(&fold, r, [0.0; 3]).unwrap();
            assert_eq!(f.tones.len(), cot.len());
            let tr = |b: &Mat, g: &Mat| (0..b.data.len()).map(|i| (b.data[i].conj() * g.data[i]).re).sum::<f64>();
            tr(&rho, &f.generator) + f.tones.iter().zip(&cot).map(|(t, b)| tr(b, &t.generator)).sum::<f64>()
        };
        let (force, work) = fold.force(&f0, &rho, &cot);
        println!("{} tones, {} blocks; the force {force:?} ({work:.0} complex multiply-adds; the fold's own {:.0})", f0.tones.len(), fold.blocks.len(), f0.work);
        let scale = force.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        for a in 0..3 {
            let quotients: Vec<f64> = crate::model::testing::steps()
                .iter()
                .map(|&h| {
                    let (mut p, mut m) = (at, at);
                    p[a] += h;
                    m[a] -= h;
                    -(value(p) - value(m)) / (2.0 * h)
                })
                .collect();
            let estimate = crate::model::testing::converged(&quotients, 1);
            println!("   ∂{a}: adjoint {:.10e}, difference {:.10e} ± {:.1e}", force[a], estimate.0, estimate.1);
            assert!(crate::model::testing::agrees(force[a], estimate, scale), "{what}, axis {a}: the adjoint against the difference");
        }

    }
}

/// ⑧ A frame is a gauge: the open Λ with cross legs (its copy of g2 by beam 2 in beam 1's block kept, a tone) and a second
/// set (ground state 6 driven by a third beam into excited state 9), each set's frames shifted by its own random multiple of
/// a gauge axis, the idle states' by a third: the static generator less the frames' shift, every tone (by its beat,
/// unchanged), and every block's jumps (by its offset, shifted with its set) are the lab frames' ones, to the rounding.
#[test]
fn a_frame_shift_leaves_the_generator_and_the_jumps_as_they_are() {
    let rays = [
        Ray { axis: 0, nu: 0.0, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(60.0)), (17, 4, C64::real(15.0))] },
        Ray { axis: 1, nu: 400.0, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(18.0)), (17, 4, C64::real(50.0))] },
        Ray { axis: 2, nu: -150.0, direction: [0.0, 1.0, 0.0], couplings: vec![(9, 6, C64::real(30.0))] },
    ];
    let req = system(&[(0, 0.0), (4, 8.0 - 400.0), (6, 5.0)], &[(17, 3072.0, GAMMA, 0), (9, 0.0, GAMMA, 6)], &rays, 4, true);
    let ps = parts(&req);
    let kept = (ps[0], 0, ps[1], 2);
    let mut copies = all_copies(&ps, Status::Decomposed);
    copies.retain(|x| x.0 != kept);
    copies.push((kept, Status::Kept));
    let base = Fold::new(&req, &Choice { frames: lab(&req), explicit: vec![], copies: copies.clone(), ..rule(&req) }, 0).unwrap();
    let at = [0.2, 0.1, -0.3];
    let f = fold_at(&base, at, [0.0; 3]).unwrap();
    assert!(!f.tones.is_empty(), "the kept copy's tones");
    let scale = f.generator.frobenius();
    let mut seed = 5u64;
    for _ in 0..3 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let shift = |k: u64| {
            let mut n = vec![0i64; req.frequencies_mhz.len() + 1];
            n[3] = ((seed >> (8 * k)) % 97) as i64 - 48;
            Beat::whole(n)
        };
        let set = |s: usize| if s == 0 || s == 4 { 0 } else if s == 6 { 1 } else { 2 };
        let frames: Vec<Beat> = (0..GROUND).map(|s| shift(set(s))).collect();
        let fold = Fold::new(&req, &Choice { frames: frames.clone(), explicit: vec![], copies: copies.clone(), ..rule(&req) }, 0).unwrap();
        let g = fold_at(&fold, at, [0.0; 3]).unwrap();
        for p in 0..f.n {
            for q in 0..f.n {
                let expected = f.generator[(p, q)] - if p == q { C64::real(fold.frame_values[p]) } else { C64::ZERO };
                assert!((g.generator[(p, q)] - expected).abs() <= 1e-12 * (scale + fold.frame_values[p].abs()), "({p}, {q}): the static generator");
            }
        }
        assert_eq!(f.tones.len(), g.tones.len(), "the tones");
        for t in &f.tones {
            let u = g.tones.iter().find(|u| u.beat == t.beat).expect("each tone at its beat");
            let mut d = u.generator.clone();
            d.add_scaled(-C64::ONE, &t.generator);
            assert!(d.max_abs() <= 1e-12 * scale, "a tone");
        }
        for (b, j) in base.blocks.iter().zip(&f.jumps) {
            let source = b.sources.first().map_or(0, |s| base.ground[s.0]);
            let target = b.offset.plus(1, &frames[source]);
            let k = fold.blocks.iter().position(|c| c.offset == target).expect("each block at its shifted offset");
            for q in 0..3 {
                let mut d = g.jumps[k][q].clone();
                d.add_scaled(-C64::ONE, &j[q]);
                assert!(d.max_abs() <= 1e-12 * (1.0 + j[q].max_abs()), "a block's jumps");
            }
        }
    }
}

/// ⑨ The Sylvester back substitution (onefold 3): on random matrices, Z with Z T_H − (T_K + w) Z = B is per eigenvalue
/// (λ_p − w − K)⁻¹ B r_p to the rounding times H's eigenvectors' condition; through the exceptional point of H(s) = [[1, 1],
/// [s, 1]] at s = 0 its solution moves continuously (its difference quotients bounded) while the eigenvectors' condition
/// grows without bound. And the read next pass against the self-consistent solution of the same equation (the passes
/// repeated to their fixed point), both in the readings' units: the actual error within 2(1 − ratio)⁻¹ of the reading (a
/// reading above the error is conservative; how far above, printed).
#[test]
fn the_sylvester_back_substitution_is_the_eigenvalue_one_and_continuous_through_an_exceptional_point() {
    let mut seed = 7u64;
    let mut u = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut w = 0.0;
    for _ in 0..10 {
        let (n, m) = (5, 4);
        let h = Mat::from_fn(n, n, |_, _| C64::new(u(), u()));
        let k = Mat::from_fn(m, m, |i, j| C64::new(u(), u()) + if i == j { C64::new(3.0, -1.0) } else { C64::ZERO });
        let b = Mat::from_fn(m, n, |_, _| C64::new(u(), u()));
        let shift = 0.3;
        let (th, tk) = (Triangle::of(&h, &mut w).unwrap(), Triangle::of(&k, &mut w).unwrap());
        let z = syl(&tk.t, &[m; 4], &th.t, shift, &tk.u.adjoint().product(&b).product(&th.u), &mut w);
        let t = tk.u.product(&z).product(&th.u.adjoint());
        let e = crate::linalg::eigen(&h).unwrap();
        for p in 0..n {
            let mut col: Vec<C64> = (0..m).map(|i| (0..n).map(|j| b[(i, j)] * e.vectors[(j, p)]).sum()).collect();
            let a = Mat::from_fn(m, m, |i, j| if i == j { e.values[p] - C64::real(shift) } else { C64::ZERO } - k[(i, j)]);
            crate::linalg::Lu::new(&a).unwrap().solve(&mut col);
            let ours: Vec<C64> = (0..m).map(|i| (0..n).map(|j| t[(i, j)] * e.vectors[(j, p)]).sum()).collect();
            let gap: f64 = ours.iter().zip(&col).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max);
            assert!(gap <= 1e-12 * e.condition * t.frobenius(), "eigenvalue {p}: {gap:.2e} (condition {:.1})", e.condition);
        }
    }
    // Through the exceptional point.
    let k = Mat::from_fn(2, 2, |i, j| if i == j { C64::new(4.0, -0.5) } else { C64::ZERO });
    let b = Mat::from_fn(2, 2, |i, j| C64::real(1.0 + i as f64 + 2.0 * j as f64));
    let solve = |s: f64, w: &mut f64| {
        let h = Mat::from_fn(2, 2, |i, j| C64::real(if i == j || i == 0 { 1.0 } else { s }));
        let (th, tk) = (Triangle::of(&h, w).unwrap(), Triangle::of(&k, w).unwrap());
        let z = syl(&tk.t, &[2, 2], &th.t, 0.0, &tk.u.adjoint().product(&b).product(&th.u), w);
        (tk.u.product(&z).product(&th.u.adjoint()), crate::linalg::eigen(&h).map_or(f64::INFINITY, |e| e.condition))
    };
    let mut last: Option<(f64, Mat)> = None;
    let mut worst = 0.0_f64;
    for step in -10i32..=10 {
        let s = 1e-4 * step as f64;
        let (t, condition) = solve(s, &mut w);
        if let Some((s0, t0)) = &last {
            let mut d = t.clone();
            d.add_scaled(-C64::ONE, t0);
            worst = worst.max(d.frobenius() / (s - s0));
        }
        if step.abs() <= 1 {
            println!("s {s:+.1e}: H's eigenvectors' condition {condition:.2e}");
        }
        last = Some((s, t));
    }
    println!("through the exceptional point: the largest difference quotient {worst:.3}");
    assert!(worst <= 10.0, "the solution moves continuously through the exceptional point");
    // The read next pass against the exact self-consistent solution of the fold's own lattice.
    let mut state = 99u64;
    let mut checked = 0;
    let mut over: Vec<f64> = vec![];
    for trial in 0..20 {
        let (req, choice) = random_system(&mut state, trial);
        let fold = Fold::new(&req, &choice, 0).unwrap();
        let f = fold_at(&fold, [0.0; 3], [0.0; 3]).unwrap();
        if !(f.excitation < 0.5 && f.ratio < 0.5) {
            continue;
        }
        let n = f.n;
        // The fixed point of the same equation: the passes repeated (each the Sylvester solve at the Bloch generator of the
        // last) until the generator stays within its rounding.
        let mut h = f.h[1].clone();
        let mut w = 0.0;
        let fixed = loop {
            let stage = fold.solve(&f.site, &f.laid, &f.eliminated, &h, false, &mut w).unwrap();
            let next = fold.bloch(&f.hpp, &f.laid, &stage, &mut w);
            let mut d = next.clone();
            d.add_scaled(-C64::ONE, &h);
            h = next;
            if d.frobenius() <= 1e-14 * h.frobenius() {
                break stage;
            }
        };
        // Per node the largest over H's Schur columns of the error's part in the readings' units (an excited row's relative
        // to the column's whole excited response, a copy's absolute), summed over the nodes: as the readings sum.
        let stage = f.stages.last().unwrap();
        let u_h = &stage.triangle.u;
        let ours: Vec<Mat> = stage.responses.iter().map(|t| t.product(u_h)).collect();
        let theirs: Vec<Mat> = fixed.responses.iter().map(|t| t.product(u_h)).collect();
        let floor = (crate::units::rounding(n) * ours.iter().map(|t| t.frobenius()).sum::<f64>()).powi(2);
        let whole: Vec<f64> = (0..n)
            .map(|j| fold.blocks.iter().zip(&ours).map(|(b, t)| (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))).map(|i| t[(i, j)].norm_sqr()).sum::<f64>()).sum())
            .map(|x: f64| if x > floor { x } else { 0.0 })
            .collect();
        let mut parts: Vec<((usize, Node), f64)> = vec![];
        for (o, b) in fold.blocks.iter().enumerate() {
            for j in (0..n).filter(|&j| whole[j] > 0.0) {
                let mut here: Vec<(Node, f64)> = vec![];
                for i in 0..b.rows.len() {
                    let (nd, per) = match b.rows[i] {
                        Row::Excited(x) => (Node::Excited(fold.excited_level[x]), whole[j]),
                        Row::Copy { state, component } => (Node::Copy { level: fold.ground_level[state], component }, 1.0),
                    };
                    let e = (ours[o][(i, j)] - theirs[o][(i, j)]).norm_sqr() / per;
                    match here.iter_mut().find(|x| x.0 == nd) {
                        Some(x) => x.1 += e,
                        None => here.push((nd, e)),
                    }
                }
                for (nd, e) in here {
                    match parts.iter_mut().find(|x| x.0 == (o, nd)) {
                        Some(x) => x.1 = x.1.max(e),
                        None => parts.push(((o, nd), e)),
                    }
                }
            }
        }
        let actual: f64 = parts.iter().map(|x| x.1).sum();
        if actual <= 1e-24 {
            continue;
        }
        let (a, r) = (actual.sqrt(), read(&f, &|r| r.next).sqrt());
        println!("trial {trial}: the actual error {a:.3e}, the read next pass {r:.3e} (ratio {:.3})", f.ratio);
        assert!(a <= 2.0 * r / (1.0 - f.ratio), "trial {trial}: the read next pass below the self-consistent solution's error");
        over.push(r / a);
        checked += 1;
    }
    over.sort_by(f64::total_cmp);
    println!("the read next pass over the actual error: median {:.3}, largest {:.3}", over[over.len() / 2], over[over.len() - 1]);
    assert!(checked >= 5, "{checked} systems checked");
}

/// ⑩ The tweezer's shift of the excited state, summed to all orders (DESIGN §1.4): a two-level atom (Γ = 38.1) under a weak
/// cooling beam (Δ_c = −228 rad/µs) and a far red tweezer (ν_t = −10⁵ rad/µs) whose ground light shift U is −0.9 |Δ_c|: the
/// excited state's shift −U makes ΣR ≈ −0.9 on the cooling's replica. The reference is the system's own Floquet lattice (its
/// depth grown from 4 by 2 until it stays within 10⁻¹⁰): the fold's eigenvalue within its read bound (next pass,
/// truncation, evaluation order). The dressed two-level formula, the root of [[U, g_c], [g_c, −Δ_c − U − iΓ/2]] (U the
/// tweezer's two-level root), is not exact: it takes the tweezer's dressing of the ground and the excited state as the two
/// shifts U and −U and leaves out the dressed states' mixing, |g_t/ν_t|² ≈ 2·10⁻³ of each, which the cooling's coupling
/// and the rate carry at that order -- printed beside, not asserted. The first order in Σ (what a tone would follow)
/// misses the rate by far.
#[test]
fn the_tweezer_s_excited_shift_is_summed_to_all_orders() {
    let (dc, nt, gc) = (-228.0, -1e5, 3.0);
    let gt = (0.9 * 228.0 * 1e5_f64).sqrt();
    let req = system(
        &[(0, 0.0)],
        &[(17, 0.0, GAMMA, 0)],
        &[Ray { axis: 0, nu: dc, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(gc))] }, Ray { axis: 1, nu: nt, direction: [1.0, 0.0, 0.0], couplings: vec![(17, 0, C64::real(gt))] }],
        2,
        true,
    );
    let fold = Fold::new(&req, &folded(&req, lab(&req)), 0).unwrap();
    let f = fold_at(&fold, [0.0; 3], [0.0; 3]).unwrap();
    let ours = f.generator[(0, 0)];
    let mut depth = 4;
    let mut ex = exact(&fold, &lattice(&fold, [0.0; 3], depth, &|_| true));
    loop {
        let next = exact(&fold, &lattice(&fold, [0.0; 3], depth + 2, &|_| true));
        let change = apart(&next, &ex).iter().fold(0.0_f64, |a, &b| a.max(b));
        ex = next;
        depth += 2;
        if change <= 1e-10 * (1.0 + ex.iter().map(|x| x.0.abs()).fold(0.0, f64::max)) || depth >= 12 {
            break;
        }
    }
    let exact = ex.iter().map(|x| x.0).min_by(|a, b| (*a - ours).abs().total_cmp(&(*b - ours).abs())).unwrap();
    let u = root(C64::ZERO, C64::new(-nt, -0.5 * GAMMA), gt);
    let analytic = root(u, C64::new(-dc, -0.5 * GAMMA) - u, gc);
    let sigma_r = -u.re / dc;
    let first = {
        let r = C64::ONE / C64::new(dc, 0.5 * GAMMA);
        let t = r * gc * (C64::ONE + (-u) * r);
        GAMMA * t.norm_sqr()
    };
    let rate = |l: C64| -2.0 * (l.im - u.im);
    let (r_ours, r_exact, r_analytic) = (rate(ours), rate(exact), rate(analytic));
    let bound = eigenvalue_bound(&f, &|r| r.next + r.truncation + r.order);
    let mixing = (gt / nt).powi(2);
    println!("U {u:.4?}, ΣR {sigma_r:+.3}, |g_t/ν_t|² {mixing:.2e} (lattice depth {depth})");
    println!("the rate: the fold's {r_ours:.6e}, the lattice's {r_exact:.6e} ({:+.2e} relative), the dressed two-level's {r_analytic:.6e} ({:+.2e}), first order in Σ {first:.6e}", r_ours / r_exact - 1.0, r_analytic / r_exact - 1.0);
    println!("the eigenvalue: the fold {:.3e} from the lattice's, the read bound {bound:.3e}; the formula {:.3e} from it", (ours - exact).abs(), (analytic - exact).abs());
    assert!((sigma_r + 0.9).abs() < 0.05, "ΣR ≈ −0.9");
    assert!((ours - exact).abs() <= bound + 1e-9 * exact.abs(), "summed to all orders, within the read bound");
    assert!((first - r_exact).abs() > 0.5 * r_exact, "the first order misses it");
}

/// The rise and the fall (DESIGN §11.1) on a random system's excited node: rising a normalized state's x = S⁻¹y and falling
/// it back gives y and δp = 0 to the rounding; an amplitude off T_X x by d falls with δp = ‖d‖² less what the fold's span
/// takes of it (between zero and ‖d‖²).
#[test]
fn a_node_rises_and_falls_back_exactly_and_reads_what_it_drops() {
    let mut state = 41u64;
    let (req, choice) = random_system(&mut state, 1);
    let fold = Fold::new(&req, &choice, 0).unwrap();
    let f = fold_at(&fold, [0.0; 3], [0.0; 3]).unwrap();
    let (o, rows) = (0..fold.blocks.len())
        .find_map(|o| {
            let level = fold.blocks[o].rows.iter().find_map(|r| if let Row::Excited(x) = r { Some(fold.excited_level[*x]) } else { None })?;
            Some((o, fold.node_rows(o, Node::Excited(level))))
        })
        .unwrap();
    let y: Vec<C64> = (0..f.n).map(|i| C64::new(1.0 + i as f64, 0.5 - i as f64) / (f.n as f64 * 3.0)).collect();
    let x = f.normal.apply(&y);
    let risen = f.rise(o, &rows, &x);
    let (back, dropped) = f.fall(&x, &[(o, rows.clone(), risen.clone())]);
    let gap: f64 = back.iter().zip(&y).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max);
    println!("rise and fall: {gap:.2e} from y, δp {dropped:.2e}");
    assert!(gap <= 1e-13 && dropped.abs() <= 1e-13, "the fall of an exact rise");
    let off: Vec<C64> = risen.iter().map(|a| *a + C64::new(0.01, -0.02)).collect();
    let (_, dropped) = f.fall(&x, &[(o, rows.clone(), off)]);
    let size = rows.len() as f64 * (0.01f64 * 0.01 + 0.02 * 0.02);
    println!("an amplitude off by ‖d‖² = {size:.2e}: δp {dropped:.3e}");
    assert!(dropped > 0.0 && dropped <= size * (1.0 + 1e-9), "the fast part dropped");
}


/// The work of a fold along a path in a scene (not a test of the fold: the step's count for the gate, DESIGN §8; FOLD_SCENE a
/// reference scene's name or a native request's path). The rule's sets, frames and parts (`Choice::of`); every copy
/// decomposed (exact: a centring can be singular, a copy at the centre itself), then each at the cheapest evaluation its
/// readings allow within the share (left out, centred, decomposed: M18), to its fixed point, at the focus; then per set one
/// path of 100 steps of 30 nm through the focus on one `Path`: per step the complex multiply-adds of the fold and of the
/// force, the decompositions made, the largest reuse reading; the readings' once.
#[test]
#[ignore]
fn a_fold_s_work_at_a_scene() {
    let name = std::env::var("FOLD_SCENE").expect("FOLD_SCENE");
    let req = crate::model::testing::scene(&name);
    let share = super::super::elim::share();
    let choice = tiers(&req, &name);
    let unit = 1.0 / 3.0_f64.sqrt();
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        if fold.blocks.is_empty() {
            continue;
        }
        // FOLD_DETAIL: per block at the focus, its copies' emitting light and the couplings between the lines its
        // decomposition leaves out -- the bare frame matrix's and the centred dressing's.
        if std::env::var("FOLD_DETAIL").is_ok() {
            let mut w = 0.0;
            let site = fold.site([0.0; 3], [0.0; 3], &mut w);
            for (o, b) in fold.blocks.iter().enumerate() {
                let l = fold.lay(o, &site, &mut w);
                let Some(e) = fold.effective(o, &l, &mut w) else { continue };
                let mut by: Vec<String> = vec![];
                for r in &b.rows {
                    if let Row::Copy { state, component } = *r {
                        let comp = &fold.components[component];
                        let names: Vec<&str> = comp.beams.iter().map(|&x| req.beams[x].name.as_str()).collect();
                        let tag = format!("{:?} {names:?} {}", comp.part, if b.rows[..b.solved].contains(r) { "decomposed" } else { "centred" });
                        if !by.contains(&tag) {
                            by.push(tag);
                        }
                        let _ = state;
                    }
                }
                let _ = (&l, &e);
                let sources: Vec<String> = b.sources.iter().map(|&(p, c)| format!("{}<-{:?}", fold.ground[p], fold.components[c].part)).take(2).collect();
                println!("   block {o} ({:.1} rad/µs; sources {sources:?}…): rows {}/{}, copies {by:?}", b.value, b.rows.len(), b.solved);
            }
        }
        for thermal in [false, true] {
        let mut path = Path::default();
        let (mut works, mut forces, mut remade, mut worst) = (vec![], vec![], vec![], 0.0_f64);
        let mut phases: Vec<(&'static str, f64)> = vec![];
        let mut passes = [0usize; 4];
        // A moving atom: 0.1 m/s along (1, 2, 3)/√14, every beam its own Doppler shift.
        let speed = 0.1 / fold.hbar_over_mass;
        let kappa = [speed / 14f64.sqrt(), 2.0 * speed / 14f64.sqrt(), 3.0 * speed / 14f64.sqrt()];
        let started = std::time::Instant::now();
        let steps = 100;
        let mut clocks = [0.0_f64; 2];
        let mut last = None;
        for step in 0..steps {
            // Through the focus (±1.5 µm along (1, 1, 1)/√3, 30 nm a step), or a thermal atom about it: 0.1 µm (30 µK's
            // spread) on each axis at periods of 15, 17 and 60 steps.
            let s = -1.5 + 0.03 * step as f64;
            let phase = |period: f64| (2.0 * std::f64::consts::PI * step as f64 / period).sin();
            let r = if thermal { [0.1 * phase(15.0), 0.1 * phase(17.0), 0.1 * phase(60.0)] } else { [s * unit; 3] };
            let made = path.made;
            let clock = std::time::Instant::now();
            let f = fold.at(r, kappa, &mut path).unwrap();
            let folded = clock.elapsed().as_secs_f64();
            passes[f.passes] += 1;
            let rho = Mat::from_fn(f.n, f.n, |i, j| if i == j { C64::real(1.0 / f.n as f64) } else { C64::ZERO });
            let clock = std::time::Instant::now();
            let (_, force) = fold.force(&f, &rho, &[]);
            let forced = clock.elapsed().as_secs_f64();
            if step > 0 {
                clocks[0] += folded / (steps - 1) as f64;
                clocks[1] += forced / (steps - 1) as f64;
            }
            works.push(f.work);
            forces.push(force);
            if step > 0 {
                for &(name, x) in &f.phases {
                    match phases.iter_mut().find(|p| p.0 == name) {
                        Some(p) => p.1 += x / (steps - 1) as f64,
                        None => phases.push((name, x / (steps - 1) as f64)),
                    }
                }
            }
            worst = f.reuse.iter().fold(worst, |a, &b| a.max(b));
            if path.made > made {
                remade.push(step);
            }
            last = Some(f);
        }
        let elapsed = started.elapsed().as_secs_f64() / steps as f64;
        let mut f = last.unwrap();
        fold.read(&mut f, 0.0);
        let rest = |v: &[f64]| v[1..].iter().sum::<f64>() / (v.len() - 1) as f64;
        let rows: Vec<String> = fold
            .blocks
            .iter()
            .zip(&f.laid)
            .map(|(b, l)| format!("{}/{}/{}", b.rows.len(), b.solved, l.shifts.len() - 1))
            .collect();
        println!(
            "{name} set {:?}: n {}, {} blocks (rows/solved/beams {rows:?}), {} bases, x {:.4}, ratio {:.4}; made {} (at steps {remade:?}), the largest reuse reading {worst:.2e} (share {share:.4}); the fold's work: first {:.0}, then {:.0} a step (largest {:.0}); the force's {:.0} a step; the readings' {:.0} once; {:.2} ms a step with the force",
            fold.ground,
            f.n,
            fold.blocks.len(),
            path.bases.len(),
            f.excitation,
            f.ratio,
            path.made,
            works[0],
            rest(&works),
            works.iter().fold(0.0_f64, |a, &b| a.max(b)),
            rest(&forces),
            f.reading_work,
            1e3 * elapsed
        );
        println!("   a step by phase: {}; passes one/two/three at {}/{}/{} steps", phases.iter().map(|(n, x)| format!("{n} {x:.0}")).collect::<Vec<_>>().join(", "), passes[1], passes[2], passes[3]);
        println!("   wall clock a step (single thread, after the first): the fold {:.1} µs, the force {:.1} µs", 1e6 * clocks[0], 1e6 * clocks[1]);
        println!("   {} path: {} decompositions made in {steps} steps (at {remade:?}), the reuse's largest reading {worst:.2e}, the decompositions' eigenvectors' largest condition {:.3}", if thermal { "a thermal atom's" } else { "the focus's" }, path.made, path.condition);
        }
        // The cache shared by the atoms (Path): the focus's path twice (the second on what the first kept), and 88 thermal
        // atoms (0.1 µm about the focus on each axis, random amplitudes, phases and periods), each 100 steps, atom after atom.
        let mut path = Path::default();
        for pass in 0..2 {
            let before = path.made;
            for step in 0..100 {
                let s = -1.5 + 0.03 * step as f64;
                fold.at([s * unit; 3], kappa_of(&fold), &mut path).unwrap();
            }
            println!("   shared, the focus's path, pass {pass}: {} decompositions made, {} kept in all, {} taken up again", path.made - before, path.bases_len(), path.looked);
        }
        let mut path = Path::default();
        let mut seed = 1u64;
        let mut u = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let started = std::time::Instant::now();
        let mut after10 = 0;
        for atom in 0..88 {
            let shape: Vec<(f64, f64, f64)> = (0..3).map(|a| (0.2 * u(), 2.0 * std::f64::consts::PI * u(), [15.0, 17.0, 60.0][a] * (0.8 + 0.4 * u()))).collect();
            path.current.clear();
            for step in 0..100 {
                let r: [f64; 3] = std::array::from_fn(|a| shape[a].0 * (2.0 * std::f64::consts::PI * step as f64 / shape[a].2 + shape[a].1).sin());
                fold.at(r, kappa_of(&fold), &mut path).unwrap();
            }
            if atom == 9 {
                after10 = path.made;
            }
        }
        let each = started.elapsed().as_secs_f64() / 8800.0;
        println!("   shared, 88 thermal atoms × 100 steps: {} decompositions made in all ({} by the first 10 atoms), {} taken up again; {:.1} µs a step (the fold alone)", path.made, after10, path.looked, 1e6 * each);
    }
}

/// A scene's evaluation tiers at its focus (the measurements' choice, M18): the rule's sets, frames and parts
/// (`Choice::of`); every copy decomposed (exact: a centring can be singular, a copy at the centre itself), then each at the
/// cheapest evaluation its readings allow within the share (left out, centred, decomposed), and the excited parts whose
/// omission reads within it left out, to the fixed point.
fn tiers(req: &Request, name: &str) -> Choice {
    let share = super::super::elim::share();
    let ps = parts(req);
    let mut choice = Choice { copies: all_copies(&ps, Status::Decomposed), ..rule(req) };
    println!("{name}: the rule's ground sets {:?}", choice.sets);
    let at = [0.0; 3];
    for set in Fold::sets(req, &choice).unwrap() {
        for round in 0..4 {
            let fold = Fold::new(req, &choice, set[0]).unwrap();
            if fold.blocks.is_empty() {
                break;
            }
            let f = fold_at(&fold, at, [0.0; 3]).unwrap();
            let mut changed = false;
            let mut statuses: Vec<String> = vec![];
            for r in &f.readings {
                let Node::Copy { level, component } = r.node else { continue };
                let b = &fold.blocks[r.block];
                let Some(&(s, a)) = b.sources.first() else { continue };
                let key = (fold.components[a].part, fold.ground[s], fold.components[component].part, level);
                let status = choice.copies.iter().rev().find(|x| x.0 == key).map_or(Status::Centred, |x| x.1);
                // The cheapest evaluation whose reading is within the share (M18): left out, centred, decomposed.
                let next = if rate(r.omission.sqrt()) <= share { Status::Omitted } else if rate(r.centring.sqrt()) <= share { Status::Centred } else { Status::Decomposed };
                if next != Status::Omitted {
                    statuses.push(format!("block {} F={level} by {:?}: {next:?}", r.block, fold.components[component].part));
                }
                if next != status {
                    for &(s, a) in &b.sources {
                        let key = (fold.components[a].part, fold.ground[s], fold.components[component].part, level);
                        choice.copies.retain(|x| x.0 != key);
                        choice.copies.push((key, next));
                    }
                    changed = true;
                }
            }
            // The excited nodes, the same rule: a block leaves out the nodes whose omission readings (the rates' and the
            // potential's, the larger) sum within the share, the smallest first; of the others it centres those whose
            // centring reads within it; the rest it solves.
            for (o, b) in fold.blocks.iter().enumerate() {
                let mut nodes: Vec<(usize, f64, f64)> = f.readings.iter().filter(|r| r.block == o).filter_map(|r| if let Node::Excited(level) = r.node { Some((level, r.omission, r.centring)) } else { None }).collect();
                nodes.sort_by(|a, b| a.1.total_cmp(&b.1));
                let mut sum = 0.0;
                for (level, omission, centring) in nodes {
                    let next = if rate((sum + omission).sqrt()) <= share {
                        sum += omission;
                        Status::Omitted
                    } else if rate(centring.sqrt()) <= share {
                        Status::Centred
                    } else {
                        Status::Decomposed
                    };
                    let Some(&(s0, a0)) = b.sources.first() else { continue };
                    let now = choice.excited.iter().rev().find(|x| x.0 == (fold.components[a0].part, fold.ground[s0], level)).map_or(Status::Decomposed, |x| x.1);
                    if next != now {
                        for &(s, a) in &b.sources {
                            let key = (fold.components[a].part, fold.ground[s], level);
                            choice.excited.retain(|x| x.0 != key);
                            choice.excited.push((key, next));
                        }
                        statuses.push(format!("block {o}: excited level {level} {next:?} (omission {omission:.1e}, centring {centring:.1e})"));
                        changed = true;
                    }
                }
            }
            println!("set {set:?} round {round}: {statuses:?}");
            if !changed {
                break;
            }
        }
    }
    choice
}

/// A moving atom's wave vector (0.1 m/s along (1, 2, 3)/√14).
fn kappa_of(fold: &Fold) -> [f64; 3] {
    let speed = 0.1 / fold.hbar_over_mass;
    [speed / 14f64.sqrt(), 2.0 * speed / 14f64.sqrt(), 3.0 * speed / 14f64.sqrt()]
}

/// The step against the fold at a point (B2, stage 1; step.rs): on a fresh cache -- its decompositions made at the point,
/// as `Fold::at` makes its own -- the step's A, metric, S⁻¹ and G are the fold's to the rounding, its passes the same; and
/// its decay is its own jumps' products, C_q ε_o S⁻¹ of its own responses, to the rounding (DESIGN §2.6). On the 20 random
/// systems of ③ (every copy exact, an explicit level in every third), on R4_LIN_4 with its D2 F′ = 2 explicit and its near
/// copies exact, the others centred (a static source of the explicit state's, and of H_GG at a copy's frame), and on
/// R4_LIN_4 by the rule (every copy centred); at rest and moving.
#[test]
fn the_step_is_the_fold_at_a_point_to_rounding() {
    use super::step::{Cache, Workspace};
    let decay = |fold: &Fold, ws: &Workspace, g: &Mat, normal: &Mat| -> f64 {
        let n = fold.size();
        let es: Vec<usize> = (0..GROUND).collect();
        let mut products = Mat::zeros(n, n);
        for (o, b) in fold.blocks.iter().enumerate() {
            let e = fold.amplitude(b, ws.response(o));
            let rows = b.excited(fold);
            for q in 0..3 {
                let j = fold.emission[q].block(&es, &rows).product(&e).product(normal);
                products.add_scaled(C64::ONE, &j.adjoint_product(&j));
            }
        }
        let mut gap = Mat::from_fn(n, n, |i, j| C64::I * (g[(i, j)] - g[(j, i)].conj()));
        let scale = g.frobenius().max(products.frobenius()).max(f64::MIN_POSITIVE);
        gap.add_scaled(-C64::ONE, &products);
        gap.max_abs() / scale
    };
    let (mut worst, mut worst_decay, mut compared) = (0.0_f64, 0.0_f64, 0);
    let mut check = |fold: &Fold, r: [f64; 3], kappa: [f64; 3], what: &str| {
        if fold.blocks.is_empty() {
            return;
        }
        let Some(f) = fold.at(r, kappa, &mut Path::default()) else { return };
        let mut ws = fold.workspace();
        let s = fold.step(r, kappa, &mut Cache::default(), &mut ws).expect("the step where the fold is").clone_parts();
        let scale = f.generator.frobenius().max(f.a.frobenius()).max(f64::MIN_POSITIVE);
        let gap = |a: &Mat, b: &Mat| a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max);
        let x = [gap(&s.0, &f.a) / scale, gap(&s.1, &f.metric) / f.metric.frobenius().max(f64::MIN_POSITIVE), gap(&s.2, &f.normal), gap(&s.3, &f.generator) / scale];
        let d = decay(fold, &ws, &s.3, &s.2);
        println!("{what}: n {}, passes {} / {}; A {:.1e}, metric {:.1e}, S⁻¹ {:.1e}, G {:.1e} (of their scales); the decay against its jumps {d:.1e}", f.n, s.4, f.passes, x[0], x[1], x[2], x[3]);
        assert_eq!(s.4, f.passes, "{what}: the passes");
        // To 1e-10 of their scales: one Schur form holds both lines, 4.4e7 rad/µs apart, so its eigenvalues are ε‖K‖ ~ 3e-8
        // rad/µs exact, and the two evaluations' different roundings of K move T by that over Γ.
        assert!(x.iter().all(|&g| g <= 1e-10), "{what}: the step against the fold");
        assert!(d <= 1e-12, "{what}: the decay against the step's jumps");
        worst = x.iter().fold(worst, |a, &b| a.max(b));
        worst_decay = worst_decay.max(d);
        compared += 1;
    };
    let mut state = 3u64;
    for trial in 0..20 {
        let (req, choice) = random_system(&mut state, trial);
        for set in Fold::sets(&req, &choice).unwrap() {
            let fold = Fold::new(&req, &choice, set[0]).unwrap();
            check(&fold, [0.0; 3], [0.0; 3], &format!("trial {trial}, set {set:?}"));
            check(&fold, [0.3, -0.1, 0.2], [2.0, -1.0, 3.0], &format!("trial {trial}, set {set:?}, moving"));
        }
    }
    let (req, choice, ..) = r4(false);
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        check(&fold, [0.1, -0.2, 0.3], [0.0; 3], &format!("R4_LIN_4 explicit, set {set:?}"));
        check(&fold, [0.1, -0.2, 0.3], kappa_of(&fold), &format!("R4_LIN_4 explicit, set {set:?}, moving"));
    }
    let req = crate::model::testing::scene("R4_LIN_4");
    let choice = rule(&req);
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        check(&fold, [0.0; 3], kappa_of(&fold), &format!("R4_LIN_4 by the rule, set {set:?}"));
        check(&fold, [0.4, 0.3, -0.5], kappa_of(&fold), &format!("R4_LIN_4 by the rule, set {set:?}, off the focus"));
    }
    // Excited nodes centred: the tweezer's blocks whole (no solved rows; the beams' sources on centred rows), and every D1
    // level in every block (excited rows solved and centred in one block, coupled by 𝒦).
    let light = Light::new(&req);
    let ps = parts(&req);
    let levels = Fold::new(&req, &choice, 0).unwrap().levels.clone();
    let count = levels.len();
    let tweezer: Vec<Part> = ps.iter().copied().filter(|p| light.frequencies[p.frequency].beams.contains(&req.trap_beam)).collect();
    let every = |parts: &[Part], keep: &dyn Fn(usize) -> bool| -> Vec<((Part, usize, usize), Status)> {
        parts.iter().flat_map(|&a| (0..GROUND).flat_map(move |s| (0..count).map(move |l| (a, s, l)))).filter(|x| keep(x.2)).map(|x| (x, Status::Centred)).collect()
    };
    for (name, excited) in [("the tweezer's blocks centred", every(&tweezer, &|_| true)), ("every D1 level centred", every(&ps, &|l| levels[l].0 == Manifold::D1))] {
        let c = Choice { excited, ..choice.clone() };
        for set in Fold::sets(&req, &c).unwrap() {
            let fold = Fold::new(&req, &c, set[0]).unwrap();
            check(&fold, [0.4, 0.3, -0.5], kappa_of(&fold), &format!("R4_LIN_4, {name}, set {set:?}"));
        }
    }
    println!("{compared} points: the step against the fold, worst {worst:.2e}; the decay against the step's jumps, worst {worst_decay:.2e}");
    assert!(compared >= 44);
}

/// A run of the step over atoms' paths (`a_step_s_work_at_a_scene`): per step after each atom's first the mean work, its
/// phases and the wall clock; the decompositions made and taken up; the passes, and the passes' work on two and on three;
/// the largest reuse reading.
struct Run {
    work: f64,
    wall: f64,
    phases: Vec<(&'static str, f64)>,
    made: usize,
    looked: usize,
    passes: [usize; 4],
    third: f64,
    second: f64,
    reading: f64,
}

/// Atoms 0.2 µm about the focus, 100 steps each (random amplitudes, phases and periods about 15, 17 and 60 steps).
fn thermal_paths(atoms: usize) -> Vec<Vec<[f64; 3]>> {
    let mut seed = 1u64;
    let mut u = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..atoms)
        .map(|_| {
            let shape: Vec<(f64, f64, f64)> = (0..3).map(|a| (0.2 * u(), 2.0 * std::f64::consts::PI * u(), [15.0, 17.0, 60.0][a] * (0.8 + 0.4 * u()))).collect();
            (0..100).map(|step| std::array::from_fn(|a| shape[a].0 * (2.0 * std::f64::consts::PI * step as f64 / shape[a].2 + shape[a].1).sin())).collect()
        })
        .collect()
}

/// The step over the paths at 0.1 m/s, one cache per atom; `fresh` a new cache every step.
fn run_step(fold: &Fold, paths: &[Vec<[f64; 3]>], fresh: bool) -> Run {
    use super::step::Cache;
    let kappa = kappa_of(fold);
    let mut ws = fold.workspace();
    let mut cache = Cache::default();
    let (mut work, mut wall, mut count, mut made, mut looked, mut passes, mut reading) = (0.0, 0.0, 0usize, 0, 0, [0usize; 4], 0.0_f64);
    let mut phases: Vec<(&'static str, f64)> = vec![];
    let (mut third, mut second) = ((0.0, 0usize), (0.0, 0usize));
    for path in paths {
        made += cache.made;
        looked += cache.looked;
        cache = Cache::default();
        for r in path.iter() {
            if fresh {
                made += cache.made;
                cache = Cache::default();
            }
            let clock = std::time::Instant::now();
            let s = fold.step(*r, kappa, &mut cache, &mut ws).expect("the step");
            let t = clock.elapsed().as_secs_f64();
            passes[s.passes] += 1;
            reading = s.reuse.iter().fold(reading, |a, &b| a.max(b));
            if std::env::var("STEP_FIRST").is_ok() && s.first.iter().any(|&x| x > super::super::elim::share()) {
                println!("      beyond on the first round: {:?}", s.first.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>());
            }
            let p3 = s.phases[3].1;
            if s.passes == 3 {
                third = (third.0 + p3, third.1 + 1);
            } else {
                second = (second.0 + p3, second.1 + 1);
            }
            {
                work += s.work;
                wall += t;
                count += 1;
                for &(n, x) in &s.phases {
                    match phases.iter_mut().find(|p| p.0 == n) {
                        Some(p) => p.1 += x,
                        None => phases.push((n, x)),
                    }
                }
            }
        }
    }
    made += cache.made;
    looked += cache.looked;
    let c = count as f64;
    phases.iter_mut().for_each(|p| p.1 /= c);
    Run { work: work / c, wall: wall / c, phases, made, looked, passes, third: third.0 / third.1.max(1) as f64, second: second.0 / second.1.max(1) as f64, reading }
}

fn show_run(what: &str, x: &Run) {
    println!(
        "   {what}: {:.0} a step ({}), {:.1} µs a step, {:.2} ns a multiply-add; made {}, taken up {}; passes two/three {}/{} (the passes' work {:.0} on two, {:.0} on three); the largest reuse reading {:.2e}",
        x.work,
        x.phases.iter().map(|(n, v)| format!("{n} {v:.0}")).collect::<Vec<_>>().join(", "),
        1e6 * x.wall,
        1e9 * x.wall / x.work,
        x.made,
        x.looked,
        x.passes[2],
        x.passes[3],
        x.second,
        x.third,
        x.reading
    );
}

/// The step's work along thermal paths in a scene and its devices with and without (not a test: B2's measurement; STEP_SCENE
/// a reference scene's name or a native request's path; STEP_ATOMS the atoms, 20 by default; run it single-threaded). The
/// scene's tiers (`tiers`); per set, atoms 0.2 µm about the focus at 0.1 m/s, 100 steps each, one cache per atom
/// (`run_step`; every step counted, the atom's first with its decompositions). Then on the same paths each device taken
/// away alone: no reuse (a fresh cache every step), no tiers (every node at its most exact: every copy exact, every excited
/// node solved); and `Fold::at` on a path per atom (the object's B1 evaluation: its wall clock per counted multiply-add
/// against the step's, the fixed workspace's).
#[test]
#[ignore]
fn a_step_s_work_at_a_scene() {
    let name = std::env::var("STEP_SCENE").expect("STEP_SCENE");
    let atoms: usize = std::env::var("STEP_ATOMS").map_or(20, |x| x.parse().unwrap());
    let req = crate::model::testing::scene(&name);
    let share = super::super::elim::share();
    let choice = tiers(&req, &name);
    let ps = parts(&req);
    let paths = thermal_paths(atoms);
    let exact = Choice { copies: all_copies(&ps, Status::Decomposed), excited: vec![], ..choice.clone() };
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        if fold.blocks.is_empty() {
            continue;
        }
        let rows: Vec<String> = fold.blocks.iter().map(|b| format!("{}/{}", b.rows.len(), b.solved)).collect();
        println!("{name} set {set:?}: n {}, {} blocks (rows/solved {rows:?}), {atoms} atoms × 100 steps, share {share:.4}", fold.size(), fold.blocks.len());
        show_run("B2 (the step)", &run_step(&fold, &paths, false));
        if std::env::var("STEP_ONLY").is_err() {
            show_run("no reuse (a fresh cache every step)", &run_step(&fold, &paths, true));
            let all = Fold::new(&req, &exact, set[0]).unwrap();
            show_run("no tiers (every copy exact, every excited node solved)", &run_step(&all, &paths, false));
            let kappa = kappa_of(&fold);
            let (mut work, mut wall, mut count, mut made) = (0.0, 0.0, 0usize, 0);
            for p in &paths {
                let mut path = Path::default();
                for r in p {
                    let clock = std::time::Instant::now();
                    let f = fold.at(*r, kappa, &mut path).unwrap();
                    wall += clock.elapsed().as_secs_f64();
                    work += f.work;
                    count += 1;
                }
                made += path.made;
            }
            let c = count as f64;
            println!("   Fold::at (B1, allocating): {:.0} a step, {:.1} µs a step, {:.2} ns a multiply-add; made {}", work / c, 1e6 * wall / c, 1e9 * wall / work, made);
        }
    }
}
