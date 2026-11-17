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

/// The left response is the right one of the adjoint (proto.rs # The landing; landing.status 2): on sixty random systems
/// (`random_system`), the prototype's N = 1 + T_L†T at the focus against the Sambe lattice's slow spans to the fold's
/// order, N⁻¹ = R_P Y_P -- its P-like modes' right eigenvectors' P rows and left ones' P columns --, within the readings
/// of both responses (T's and T_L's relative errors, ρ = unrate of each, on T_L†T: (ρ + ρ_L) tr(M − 1)); and with no
/// decay N = M (T_L = T, one iteration) to the rounding.
#[test]
fn the_left_response_is_the_right_one_of_the_adjoint() {
    let share = crate::model::elim::share();
    let mut state = 20261116u64;
    let (mut compared, mut ratios, mut hermitian) = (0, vec![], 0.0_f64);
    for trial in 0..60 {
        let (req, choice) = random_system(&mut state, trial);
        let mut still = req.clone();
        still.emission = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
        for (request, decays) in [(&req, true), (&still, false)] {
            for set in Fold::sets(request, &choice).unwrap() {
                let fold = Fold::new(request, &choice, set[0]).unwrap();
                if fold.blocks.is_empty() {
                    continue;
                }
                let Ok(mut ws) = fold.prototype_workspace() else { continue };
                let mut cache = super::proto::Cache::default();
                let Some(step) = fold.prototype([0.0; 3], [0.0; 3], &mut cache, &mut ws) else { continue };
                let (x, most) = (step.metric.clone(), step.readings().most());
                if !fold.prototype_left(&cache, &mut ws) {
                    continue;
                }
                let n = fold.size();
                let ours = ws.left_metric();
                if !decays {
                    let gap = (0..n * n).map(|i| (ours.data[i] - x.data[i] - if i % (n + 1) == 0 { C64::ONE } else { C64::ZERO }).abs()).fold(0.0, f64::max);
                    hermitian = hermitian.max(gap / (1.0 + x.frobenius()));
                    continue;
                }
                // The lattice to the fold's own order (M6: its second; beyond it the truncation's reading, which the
                // prototype's responses hold alike on both sides), so that what is compared is the series.
                let l = lattice(&fold, [0.0; 3], 2, &|_| true);
                let Some(e) = crate::linalg::eigen(&l.m) else { continue };
                // The P places' nodes: the ground states at offset zero, each explicit state at its offset (`exact`).
                let value = |k: &[i64]| k.iter().zip(&fold.components).map(|(&m, c)| m as f64 * c.nu).sum::<f64>();
                let p0: Vec<usize> = (0..n)
                    .map(|p| {
                        let (state, offset) = if p < fold.ground.len() { (fold.ground[p], 0.0) } else { (GROUND + fold.explicit[p - fold.ground.len()], fold.frame_values[p]) };
                        l.nodes.iter().position(|(s, k)| *s == state && (value(k) - offset).abs() <= 1e-9 * (1.0 + offset.abs())).expect("a P place's node")
                    })
                    .collect();
                let size = l.m.rows;
                let weight = |k: usize| p0.iter().map(|&i| e.vectors[(i, k)].norm_sqr()).sum::<f64>() / (0..size).map(|i| e.vectors[(i, k)].norm_sqr()).sum::<f64>();
                let mut modes: Vec<usize> = (0..size).collect();
                modes.sort_by(|&a, &b| weight(b).total_cmp(&weight(a)));
                modes.truncate(n);
                // Y = R⁻¹ (its rows the left eigenvectors), N⁻¹ = R_P Y_P, N.
                let mut lu = e.vectors.data.clone();
                let mut piv = vec![0; size];
                if !crate::linalg::factor(&mut lu, &mut piv, size) {
                    continue;
                }
                let mut y: Vec<C64> = (0..size * size).map(|i| if i % (size + 1) == 0 { C64::ONE } else { C64::ZERO }).collect();
                crate::linalg::solve(&lu, &piv, size, &mut y, size);
                let mut inv: Vec<C64> = (0..n * n).map(|i| (0..n).map(|k| e.vectors[(p0[i / n], modes[k])] * y[modes[k] * size + p0[i % n]]).sum()).collect();
                let mut piv = vec![0; n];
                let mut lattice_n: Vec<C64> = (0..n * n).map(|i| if i % (n + 1) == 0 { C64::ONE } else { C64::ZERO }).collect();
                if !crate::linalg::factor(&mut inv, &mut piv, n) {
                    continue;
                }
                crate::linalg::solve(&inv, &piv, n, &mut lattice_n, n);
                let gap = (0..n * n).map(|i| (ours.data[i] - lattice_n[i]).norm_sqr()).sum::<f64>().sqrt();
                let trace: f64 = (0..n).map(|p| x[(p, p)].re).sum();
                let bound = (unrate(most) + unrate(share)) * trace + 1e-9 * (1.0 + trace);
                ratios.push(gap / bound);
                compared += 1;
                if gap > bound {
                    println!("   trial {trial}: N {:.5?}, the lattice's {:.5?}; readings {most:.2e}, tr(M − 1) {trace:.3e}", (0..n).map(|p| ours.data[p * n + p]).collect::<Vec<_>>(), (0..n).map(|p| lattice_n[p * n + p]).collect::<Vec<_>>());
                }
                assert!(gap <= bound, "trial {trial}: N {gap:.3e} from the lattice's, the readings' bound {bound:.3e}");
            }
        }
    }
    ratios.sort_by(f64::total_cmp);
    println!("{compared} systems' N against the dense lattice: gap over bound median {:.3e}, largest {:.3e}; with no decay N − M at most {hermitian:.1e} (relative)", ratios[ratios.len() / 2], ratios[ratios.len() - 1]);
    assert!(compared >= 40, "{compared} compared");
    assert!(hermitian <= 1e-13, "with no decay N = M to the rounding");
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
    tiers_with(req, name, vec![])
}

/// `tiers` with the explicit levels `explicit` (`Choice::explicit`) kept in P (`Choice::tiered`).
fn tiers_with(req: &Request, name: &str, explicit: Vec<(usize, Part, usize)>) -> Choice {
    let choice = Choice::tiered(req, explicit).unwrap();
    println!("{name}: the rule's ground sets {:?}", choice.sets);
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
        // The floor, derived: one Schur form holds both lines (4.4e7 rad/µs apart in R4_LIN_4), so its eigenvalues are only
        // ε‖K‖ exact (‖K‖ ~ 1.3e8 rad/µs: ~3e-8 rad/µs); the two evaluations round K differently, which moves a resonant
        // response by that over its distance Γ/2 -- relative ε‖K‖/(Γ/2); and 1e-12 below it (the passes' and the exact
        // copies' own conditioning, the per-line era's worst 2.7e-13).
        let k_max = f.laid.iter().flat_map(|l| (0..l.inner.rows).map(move |i| l.inner[(i, i)].abs())).fold(0.0_f64, f64::max);
        let gamma = (0..EXCITED).map(|x| -2.0 * fold.base_excited[(x, x)].im).filter(|&g| g > 0.0).fold(f64::INFINITY, f64::min);
        let floor = (f64::EPSILON * k_max / (0.5 * gamma)).max(1e-12);
        assert!(x.iter().all(|&g| g <= floor), "{what}: the step against the fold beyond the floor {floor:.1e}");
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

/// `thermal_paths` at a run's step τ (µs): the same orbits (amplitudes ≤ 0.2 µm, periods 15–60 steps of 0.65 µs, i.e. trap
/// frequencies 26–100 kHz and speeds up to 0.13 m/s, a 30 µK atom's in the tweezer), each step advancing them by τ, so the
/// speed is the scene's whatever its τ.
fn thermal_paths_at(atoms: usize, tau: f64) -> Vec<Vec<[f64; 3]>> {
    let mut seed = 1u64;
    let mut u = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..atoms)
        .map(|_| {
            let shape: Vec<(f64, f64, f64)> = (0..3).map(|a| (0.2 * u(), 2.0 * std::f64::consts::PI * u(), [15.0, 17.0, 60.0][a] * (0.8 + 0.4 * u()))).collect();
            (0..100).map(|step| std::array::from_fn(|a| shape[a].0 * (2.0 * std::f64::consts::PI * step as f64 * tau / 0.65 / shape[a].2 + shape[a].1).sin())).collect()
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

/// The prototype's choice (proto.rs) for the tests: the explicit nodes the run held fixed at the focus before step3c
/// (`focus_choice`), so that the tests keep reaching the explicit sets they reached.
fn proto_choice(req: &Request, name: &str) -> Choice {
    let choice = focus_choice(req);
    println!("{name}: the rule's ground sets {:?}", choice.sets);
    choice
}

/// A fixture (the run's choice before BRIEF_STEP3C item 1, the run's own now made per point, run.rs # The nodes risen and
/// fallen): the base choice (`Choice::base`) with, to the fixed point, the excited nodes whose reading at the focus is
/// beyond the share kept in P, and where the fold does not exist there the one that reads the most.
fn focus_choice(request: &Request) -> Choice {
    let share = crate::model::elim::share();
    let mut explicit: Vec<(usize, Part, usize)> = vec![];
    loop {
        let choice = Choice::base(request, explicit.clone()).unwrap();
        let mut kept = vec![];
        for set in Fold::sets(request, &choice).unwrap() {
            let fold = Fold::new(request, &choice, set[0]).unwrap();
            if fold.blocks.is_empty() {
                continue;
            }
            let Some(f) = fold.read_at([0.0; 3]) else { continue };
            let exists = f.excitation < 1.0 && f.ratio < 1.0;
            let mut nodes: Vec<(usize, usize, f64)> = f.readings.iter().filter_map(|r| if let Node::Excited(level) = r.node { Some((r.block, level, r.total())) } else { None }).collect();
            nodes.sort_by(|a, b| b.2.total_cmp(&a.2));
            let beyond: Vec<(usize, usize, f64)> = nodes.iter().copied().filter(|x| x.2 > share).collect();
            let chosen = if !beyond.is_empty() { beyond } else if !exists { nodes.into_iter().take(1).collect() } else { vec![] };
            for (o, level, _) in chosen {
                let Some(&(p, c)) = fold.blocks[o].sources.first() else { continue };
                let key = (level, fold.components[c].part, fold.ground[p]);
                if !explicit.contains(&key) && !kept.contains(&key) {
                    kept.push(key);
                }
            }
        }
        if kept.is_empty() {
            return choice;
        }
        explicit.extend(kept);
    }
}

/// The rate's inverse: the relative amplitude error ρ of a reading r = 2ρ + ρ².
fn unrate(r: f64) -> f64 {
    (1.0 + r).sqrt() - 1.0
}

/// One point of the prototype against `Fold::at` there (its decompositions made at the point): the largest column's
/// difference of the excited responses (relative to the column's whole, as ρ), what the readings allow it -- the
/// prototype's series, reuse and held far forms and `Fold::at`'s own next pass, each as ρ, summed --, G's difference (of
/// its scale), the copies' largest absolute difference (a population's amplitude), the prototype's own decay against its
/// own jumps, its sweeps and the decompositions it made. Of the tones (proto.rs # The tones): their A_m's largest
/// difference from `Fold::at`'s (of the static A's scale), their G_m's (of their own scale), their decay
/// i(G_m − G_{−m}†) (of G's scale: no jump of their own); and the metric's beating part `Fold::at` leaves out (DESIGN M8),
/// X_cross = Σ_{o ≠ o′} ε_o′†ε_o on the excited rows two blocks share -- its largest entry, and that times the mixing
/// c_pq = √w(|G_m,pq|/|ν_m + G_pp − G_qq|) of the tone at the same entry.
struct Against {
    diff: f64,
    allowed: f64,
    parts: [f64; 5],
    g: f64,
    copies: f64,
    decay: f64,
    sweeps: usize,
    made: usize,
    tones: usize,
    tone_a: f64,
    tone_g: f64,
    tone_decay: f64,
    beating: [f64; 2],
}

fn proto_against(fold: &Fold, ws: &mut super::proto::Workspace, cache: &mut super::proto::Cache, r: [f64; 3], kappa: [f64; 3]) -> Against {
    let f = fold.at(r, kappa, &mut Path::default()).expect("the fold at the point");
    let mut w = 0.0;
    let last = f.stages.last().unwrap();
    let at_next = last.next(&f.eliminated, &f.h[1], &f.h[2], &mut w);
    let step = fold.prototype(r, kappa, cache, ws).expect("the prototype at the point");
    let (generator, normal, a) = (step.generator.clone(), step.normal.clone(), step.a.clone());
    let allowed = unrate(step.series) + unrate(step.reuse) + unrate(step.far) + unrate(step.omitted) + unrate(at_next) + 1e-10;
    let (sweeps, made, series, reuse, far, omitted) = (step.sweeps, step.made, step.series, step.reuse, step.far, step.omitted);
    let responses = ws.responses(fold, cache, super::proto::Side::Right);
    let n = f.n;
    let (mut diff, mut whole, mut copies) = (vec![0.0; n], vec![0.0; n], 0.0_f64);
    for (o, b) in fold.blocks.iter().enumerate() {
        let Some(mine) = &responses[o] else { continue };
        let theirs = &last.responses[o];
        for i in 0..b.rows.len() {
            for j in 0..n {
                let d = (mine[(i, j)] - theirs[(i, j)]).norm_sqr();
                match b.rows[i] {
                    Row::Excited(_) => {
                        diff[j] += d;
                        whole[j] += theirs[(i, j)].norm_sqr();
                    }
                    Row::Copy { .. } => copies = copies.max(d.sqrt()),
                }
            }
        }
    }
    // An explicit column's excited amplitude holds its own unit row (ε = E + T): the rate's unit of its column.
    for w in whole[fold.ground.len()..].iter_mut() {
        *w += 1.0;
    }
    let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
    let rho = (0..n).filter(|&j| whole[j] > floor).map(|j| (diff[j] / whole[j]).sqrt()).fold(0.0, f64::max);
    let g = generator.data.iter().zip(&f.generator.data).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max) / f.generator.frobenius();
    // The prototype's decay against its own jumps (DESIGN §2.6): i(G − G†) = Σ_q (C_q ε S⁻¹)†(C_q ε S⁻¹).
    let es: Vec<usize> = (0..GROUND).collect();
    let mut products = Mat::zeros(n, n);
    for (o, b) in fold.blocks.iter().enumerate() {
        let Some(t) = &responses[o] else { continue };
        let e = fold.amplitude(b, t);
        let rows = b.excited(fold);
        for q in 0..3 {
            let j = fold.emission[q].block(&es, &rows).product(&e).product(&normal);
            products.add_scaled(C64::ONE, &j.adjoint_product(&j));
        }
    }
    let mut gap = Mat::from_fn(n, n, |i, j| C64::I * (generator[(i, j)] - generator[(j, i)].conj()));
    let scale = generator.frobenius().max(products.frobenius());
    gap.add_scaled(-C64::ONE, &products);
    // The tones against `Fold::at`'s (none there at a beat: zero), and their decay, pair by pair.
    let beats: Vec<(Beat, f64)> = ws.tones().map(|(b, v)| (b.clone(), v)).collect();
    let (mut tone_a, mut tone_g, mut tone_decay) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (m, (beat, _)) in beats.iter().enumerate() {
        let theirs = f.tones.iter().find(|t| t.beat == *beat);
        let mine = Mat { rows: n, cols: n, data: ws.tone_matrix(m).to_vec() };
        let (ta, tg) = theirs.map_or((Mat::zeros(n, n), Mat::zeros(n, n)), |t| (t.a.clone(), t.generator.clone()));
        let g_m = ws.tone(m, 0);
        tone_a = tone_a.max((0..n * n).map(|i| (mine.data[i] - ta.data[i]).abs()).fold(0.0, f64::max) / a.frobenius());
        let size = g_m.max_abs().max(tg.max_abs());
        if size > 0.0 {
            tone_g = tone_g.max((0..n * n).map(|i| (g_m.data[i] - tg.data[i]).abs()).fold(0.0, f64::max) / size);
        }
        if let Some(k) = beats.iter().position(|x| x.0 == beat.negated()) {
            let partner = ws.tone(k, 0);
            tone_decay = tone_decay.max((0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (C64::I * (g_m[(i, j)] - partner[(j, i)].conj())).abs()).fold(0.0, f64::max) / scale);
        }
    }
    // The metric's beating part (DESIGN M8) on `Fold::at`'s amplitudes, and with the tones' mixing at its entries.
    let mut cross = Mat::zeros(n, n);
    for (o, bo) in fold.blocks.iter().enumerate() {
        let ro = bo.excited(fold);
        for (q, bq) in fold.blocks.iter().enumerate().filter(|&(q, _)| q != o) {
            let rq = bq.excited(fold);
            for (x, &e) in ro.iter().enumerate() {
                if let Some(y) = rq.iter().position(|&d| d == e) {
                    for p in 0..n {
                        for c in 0..n {
                            cross[(p, c)] += f.amplitudes[q][(y, p)].conj() * f.amplitudes[o][(x, c)];
                        }
                    }
                }
            }
        }
    }
    let mixing = |p: usize, c: usize| -> f64 {
        f.tones
            .iter()
            .map(|t| {
                let delta = (t.value + f.generator[(p, p)].re - f.generator[(c, c)].re).abs();
                admixture(t.generator[(p, c)].abs() / delta).sqrt()
            })
            .fold(0.0, f64::max)
    };
    let largest = cross.max_abs();
    let mixed = (0..n).flat_map(|p| (0..n).map(move |c| (p, c))).filter(|(p, c)| p != c).map(|(p, c)| cross[(p, c)].abs() * mixing(p, c)).fold(0.0, f64::max);
    Against { diff: rho, allowed, parts: [unrate(series), unrate(reuse), unrate(far), unrate(at_next), unrate(omitted)], g, copies, decay: gap.max_abs() / scale, sweeps, made, tones: beats.len(), tone_a, tone_g, tone_decay, beating: [largest, mixed] }
}

/// The prototype against the fold at a point (proto.rs; BRIEF_PROTO.md gate 2), on a scene (`PROTO_SCENE`, a request's
/// path) with its choice (`proto_choice`), each split: twenty random points about the focus (±0.3 µm, 0.1 m/s in a random
/// direction), each on a fresh cache, and a thermal atom's 100 steps on one cache. At every point the excited responses'
/// difference within what the readings allow (`proto_against`), the decay the jumps' products; the sweeps' shares said.
#[test]
#[ignore = "a check on a scene's request: run by name with PROTO_SCENE"]
fn the_prototype_is_the_fold_within_its_readings() {
    use super::proto::Cache;
    let name = std::env::var("PROTO_SCENE").expect("PROTO_SCENE");
    let req = crate::model::testing::scene(&name);
    let choice = proto_choice(&req, &name);
    let mut seed = 7u64;
    let mut u = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut checked = 0;
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        if fold.blocks.is_empty() {
            continue;
        }
        let mut ws = match fold.prototype_workspace() {
            Ok(w) => w,
            Err(e) => {
                println!("{name} set {set:?}: not the prototype's ({e})");
                continue;
            }
        };
        let (near, far) = ws.layout();
        println!("{name} set {set:?}: n {}, near blocks (block, rows, copies, beams, terms) {near:?}, far blocks (block, rows, terms) {far:?}", fold.size());
        let speed = 0.1 / fold.hbar_over_mass;
        {
            let split = "";
            let (mut worst, mut worst_g, mut worst_copy, mut worst_decay) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let (mut tones, mut tone_a, mut tone_g, mut tone_decay, mut beating) = (0, 0.0_f64, 0.0_f64, 0.0_f64, [0.0_f64; 2]);
            let mut sweeps = [0usize; 64];
            let (mut diffs, mut parts) = (0.0_f64, [0.0_f64; 5]);
            let mut check = |x: Against, what: &str, worst: &mut f64| {
                diffs = diffs.max(x.diff);
                for (a, b) in parts.iter_mut().zip(x.parts) {
                    *a = a.max(b);
                }
                assert!(x.diff <= x.allowed, "{what}: the responses' difference {:.3e} beyond what the readings allow {:.3e} (series, reuse, far, Fold::at's next, left out: {:?})", x.diff, x.allowed, x.parts);
                assert!(x.decay <= 1e-12, "{what}: the decay against the jumps {:.1e}", x.decay);
                assert!(x.tone_a <= 1e-12 && x.tone_decay <= 1e-12, "{what}: the tones' A against Fold::at's {:.1e}, their decay {:.1e}", x.tone_a, x.tone_decay);
                assert!(x.tone_g <= x.allowed, "{what}: the tones' G against Fold::at's {:.3e} beyond what the readings allow {:.3e}", x.tone_g, x.allowed);
                tones = x.tones;
                (tone_a, tone_g, tone_decay) = (tone_a.max(x.tone_a), tone_g.max(x.tone_g), tone_decay.max(x.tone_decay));
                beating = [beating[0].max(x.beating[0]), beating[1].max(x.beating[1])];
                *worst = worst.max(x.diff / x.allowed);
                worst_g = worst_g.max(x.g);
                worst_copy = worst_copy.max(x.copies);
                worst_decay = worst_decay.max(x.decay);
                sweeps[x.sweeps.min(63)] += 1;
                x.made
            };
            for p in 0..20 {
                let r: [f64; 3] = std::array::from_fn(|_| 0.6 * u() - 0.3);
                let (ct, phi) = (2.0 * u() - 1.0, 2.0 * std::f64::consts::PI * u());
                let st = (1.0 - ct * ct).sqrt();
                let kappa = [speed * st * phi.cos(), speed * st * phi.sin(), speed * ct];
                check(proto_against(&fold, &mut ws, &mut Cache::default(), r, kappa), &format!("{split}point {p}"), &mut worst);
            }
            let mut along = 0.0_f64;
            let (mut made, mut cache) = (0, Cache::default());
            let kappa = kappa_of(&fold);
            for (k, r) in thermal_paths(1)[0].iter().enumerate() {
                made += check(proto_against(&fold, &mut ws, &mut cache, *r, kappa), &format!("{split}path step {k}"), &mut along);
            }
            let total: usize = sweeps.iter().sum();
            let shares: Vec<String> = sweeps.iter().enumerate().filter(|x| *x.1 > 0).map(|(k, c)| format!("{k}: {:.1}%", 100.0 * *c as f64 / total as f64)).collect();
            println!("   {split}20 points {worst:.6e} of the allowance; the path (100 steps, {made} decompositions made) {along:.6e}; G {worst_g:.6e}, copies {worst_copy:.1e}, decay {worst_decay:.1e}; sweeps {shares:?}; the largest ρ: difference {diffs:.6e}, the series' rest {:.2e}, reuse {:.2e}, the far forms held {:.2e}, Fold::at's next {:.2e}, the nodes left out {:.2e}", parts[0], parts[1], parts[2], parts[3], parts[4]);
            println!("   {split}{tones} tones: A against Fold::at's {tone_a:.1e} (of A's scale), G_m {tone_g:.3e} (of their own), their decay {tone_decay:.1e}; the metric's beating part (M8): its largest entry {:.3e}, times the tones' mixing {:.3e}", beating[0], beating[1]);
            checked += 1;
        }
    }
    assert!(checked > 0);
}


/// The prototype's step along thermal paths (not a test: BRIEF_PROTO.md gate 1; `PROTO_SCENE` a request's path,
/// `PROTO_ATOMS` the atoms, 10 by default; run it single-threaded): per set and split, one cache per atom, 100 steps an
/// atom (`thermal_paths`, 0.1 m/s), timed twice (the first pass warms the machine), the second reported. Per step the wall
/// clock and its phases (the time stamps' ticks, scaled by the wall clock), the work, the sweeps, the decompositions made,
/// the readings; and the rest of the step's model: the outer nodes at ±τ/2 (`PROTO_TAU`, 0.65 µs by default), the force
/// of the three positions (one density for all) and the photons of one state.
#[test]
#[ignore = "a measurement: run by name with PROTO_SCENE"]
fn a_prototype_s_step_at_a_scene() {
    use super::proto::{Cache, PHASES, TANGENT};
    let name = std::env::var("PROTO_SCENE").expect("PROTO_SCENE");
    let atoms: usize = std::env::var("PROTO_ATOMS").map_or(10, |x| x.parse().unwrap());
    let req = crate::model::testing::scene(&name);
    let choice = proto_choice(&req, &name);
    let tau: f64 = std::env::var("PROTO_TAU").map_or(0.65, |x| x.parse().unwrap());
    let paths = thermal_paths_at(atoms, tau);
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        if fold.blocks.is_empty() {
            continue;
        }
        let mut ws = match fold.prototype_workspace() {
            Ok(w) => w,
            Err(e) => {
                println!("{name} set {set:?}: not the prototype's ({e})");
                continue;
            }
        };
        let (near, far) = ws.layout();
        println!("{name} set {set:?}: n {}, near blocks (block, rows, copies, beams, terms) {near:?}, far blocks (block, rows, terms) {far:?}; {atoms} atoms × 100 steps", fold.size());
        let kappa = kappa_of(&fold);
        let velocity = kappa.map(|k| k * fold.hbar_over_mass);
        let sets = Fold::sets(&req, &choice).unwrap();
        let n = fold.size();
        let psi: Vec<C64> = (0..n).map(|k| C64::cis(0.7 * k as f64) * (1.0 + 0.1 * k as f64)).collect();
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let rho = Mat::from_fn(n, n, |i, j| psi[i] * psi[j].conj() / norm);
        // The machine's pace while the passes run (the old harness's yardstick, step/tests.rs: 1024 complex multiply-adds),
        // timed in turn with them; each split's best pass (the least mean wall clock of its steps of one round) reported.
        let yardstick = || {
            let matrix: [C64; 64] = std::array::from_fn(|k| C64::cis(0.37 * k as f64) * 0.125);
            let mut x: [C64; 8] = std::array::from_fn(|k| C64::cis(1.3 * k as f64));
            for _ in 0..16 {
                let m = std::hint::black_box(&matrix);
                x = std::array::from_fn(|i| m[8 * i..8 * i + 8].iter().zip(&x).map(|(a, b)| *a * *b).sum());
            }
            x[0].re
        };
        let passes: usize = std::env::var("PROTO_PASSES").map_or(6, |x| x.parse().unwrap());
        {
            let split = "";
            let mut best: Option<(f64, String)> = None;
            let mut each_pass = vec![];
            for pass in 0..passes {
                let clock = std::time::Instant::now();
                let mut sink = 0.0;
                for _ in 0..1000 {
                    sink += yardstick();
                }
                let pace = clock.elapsed().as_secs_f64() / 1000.0 * 1e9 / 1024.0;
                std::hint::black_box(sink);
                // Per step: wall, ticks by phase, work by phase, sweeps, made; per sweep index its ticks and work.
                let (mut wall, mut ticks, mut work, mut rest) = (vec![], vec![], vec![], vec![]);
                let mut tangent: Vec<([u64; 6], [f64; 6], f64, [u64; 5])> = vec![];
                let (mut sweeps, mut made, mut rounds) = (vec![], vec![], vec![]);
                let (mut sweep_ticks, mut sweep_work, mut sweep_count) = (vec![0u64; 64], vec![0.0; 64], vec![0usize; 64]);
                let mut most = [0.0_f64; 6];
                let (mut left, mut most_omitted, mut grown) = ([0usize; 2], 0.0_f64, 0usize);
                // The decompositions at an atom's first step and after it (count, kept-level changes, ticks), and per step
                // the work the path's δK̃ would add entering the series (BRIEF_PROTO4 item 3 (a)): T s² to form it, per
                // sweep s²Ln in the series and 3s²Ln in the tangent, s²n in the compression, the two nodes' and the force's.
                let (mut first, mut later, mut later_grown, mut first_ticks, mut later_ticks) = (0usize, 0usize, 0usize, 0u64, 0u64);
                let mut series_a = vec![];
                for path in &paths {
                    let mut cache = Cache::default();
                    for (k, r) in path.iter().enumerate() {
                        let clock = std::time::Instant::now();
                        let s = fold.prototype(*r, kappa, &mut cache, &mut ws).expect("the prototype's step");
                        wall.push(clock.elapsed().as_secs_f64());
                        if k == 0 {
                            first += s.made;
                            first_ticks += s.ticks[1];
                        } else {
                            later += s.made;
                            later_grown += s.grown;
                            later_ticks += s.ticks[1];
                        }
                        let sw = s.sweeps;
                        left[0] += s.left[0];
                        left[1] += s.left[1];
                        grown += s.grown;
                        most_omitted = most_omitted.max(s.omitted);
                        let s = (s.ticks, s.work, s.sweeps, s.made, s.rounds, s.sweep_ticks.clone(), s.sweep_work.clone(), [s.series, s.reuse, s.far, s.ring, s.ratio, s.copies]);
                        series_a.push(ws.kept_rows().iter().map(|&(sk, l, t)| (t * sk * sk + sw * 4 * sk * sk * l * n + 12 * sk * sk * n) as f64).sum::<f64>());
                        // The rest of the step's model: the outer nodes, the force (the three positions' cotangents), the
                        // photons of the state.
                        let clock = std::time::Instant::now();
                        let node_work = fold.prototype_nodes(*r, kappa, velocity, [-0.5 * tau, 0.5 * tau], &cache, &mut ws);
                        let t_nodes = clock.elapsed().as_secs_f64();
                        let clock = std::time::Instant::now();
                        let tangent_work = fold.prototype_tangent(&cache, &mut ws);
                        let t_tangent = clock.elapsed().as_secs_f64();
                        let bars: Vec<super::proto::Bar> = (0..3).map(|position| super::proto::Bar { channel: 0, position, tone: None, matrix: &rho.data }).collect();
                        let force = fold.prototype_force(&cache, &mut ws, &bars, 1);
                        tangent.push((ws.tangent_ticks, ws.tangent_work, t_tangent, ws.force_ticks));
                        let t_force = clock.elapsed().as_secs_f64();
                        let clock = std::time::Instant::now();
                        let photons = fold.prototype_photons(&cache, &mut ws, &psi, &sets, super::proto::At::Middle, 0.0, super::proto::Moments::Every);
                        let t_photons = clock.elapsed().as_secs_f64();
                        rest.push([t_nodes, t_force, t_photons, node_work, tangent_work + force.work, photons.work]);
                        ticks.push(s.0);
                        work.push(s.1);
                        sweeps.push(s.2);
                        made.push(s.3);
                        rounds.push(s.4);
                        for (k, (t, w)) in s.5.iter().zip(&s.6).enumerate() {
                            sweep_ticks[k.min(63)] += t;
                            sweep_work[k.min(63)] += w;
                            sweep_count[k.min(63)] += 1;
                        }
                        for (m, x) in most.iter_mut().zip(s.7) {
                            *m = m.max(x);
                        }
                    }
                }
                if pass == 0 {
                    continue;
                }
                let mut out = String::new();
                let steps = wall.len();
                let all_ticks: u64 = ticks.iter().map(|t| t.iter().sum::<u64>()).sum();
                let ns_tick = 1e9 * wall.iter().sum::<f64>() / all_ticks as f64;
                let calm: Vec<usize> = (0..steps).filter(|&i| made[i] == 0 && rounds[i] == 1).collect();
                let every: Vec<usize> = (0..steps).collect();
                let mean = |idx: &[usize], f: &dyn Fn(usize) -> f64| idx.iter().map(|&i| f(i)).sum::<f64>() / idx.len().max(1) as f64;
                let us = |idx: &[usize]| 1e6 * mean(idx, &|i| wall[i]);
                let phase_us = |idx: &[usize], k: usize| 1e-3 * ns_tick * mean(idx, &|i| ticks[i][k] as f64);
                let phase_work = |idx: &[usize], k: usize| mean(idx, &|i| work[i][k]);
                let total_work = |idx: &[usize]| mean(idx, &|i| work[i].iter().sum::<f64>());
                let mut hist = vec![0usize; 64];
                sweeps.iter().for_each(|&k| hist[k.min(63)] += 1);
                let shares: Vec<String> = hist.iter().enumerate().filter(|x| *x.1 > 0).map(|(k, c)| format!("{k}: {:.1}%", 100.0 * *c as f64 / steps as f64)).collect();
                out += &format!(
                    "   {split}(pace {pace:.3} ns a yardstick multiply-add): {:.2} µs a step ({:.2} µs on the {} steps of one round that made nothing; {} steps made a decomposition or took up a kept one, {} made in all, {grown} of them for a change of the kept levels), {:.0} multiply-adds a step ({:.0} on those), {:.2} ns each; sweeps {shares:?}; the largest readings: series {:.1e}, reuse {:.1e}, the far forms held (their centring with them) {:.1e}, the ring {:.1e}, the sweeps' ratio {:.2}, the copies' rows left out {:.1e}",
                    us(&every),
                    us(&calm),
                    calm.len(),
                    steps - calm.len(),
                    made.iter().sum::<usize>(),
                    total_work(&every),
                    total_work(&calm),
                    1e3 * us(&calm) / total_work(&calm),
                    most[0],
                    most[1],
                    most[2],
                    most[3],
                    most[4],
                    most[5]
                );
                out += &format!("; the nodes left out: {:.2} near levels and {:.2} far blocks a step, their largest reading {most_omitted:.1e}", left[0] as f64 / steps as f64, left[1] as f64 / steps as f64);
                out += &format!(
                    "; decompositions at the atoms' first steps {first} ({:.1} µs each), after them {later} ({later_grown} for the kept levels; {:.1} µs each, {:.2} µs a step); the path's δK̃ in the series would add {:.0} multiply-adds a step",
                    1e-3 * ns_tick * first_ticks as f64 / first.max(1) as f64,
                    1e-3 * ns_tick * later_ticks as f64 / later.max(1) as f64,
                    1e-3 * ns_tick * later_ticks as f64 / steps as f64,
                    series_a.iter().sum::<f64>() / steps as f64
                );
                out += "
";
                out += &format!("      by phase (µs / multiply-adds, the steps of one round; all steps' µs in brackets): {}
", (0..10).map(|k| format!("{} {:.3}/{:.0} [{:.3}]", PHASES[k], phase_us(&calm, k), phase_work(&calm, k), phase_us(&every, k))).collect::<Vec<_>>().join(", "));
                out += &format!(
                    "      by sweep (µs / multiply-adds a sweep; the share of steps reaching it): {}
",
                    (0..64).filter(|&k| sweep_count[k] > 0).map(|k| format!("{}: {:.3}/{:.0} ({:.0}%)", k + 1, 1e-3 * ns_tick * sweep_ticks[k] as f64 / sweep_count[k] as f64, sweep_work[k] / sweep_count[k] as f64, 100.0 * sweep_count[k] as f64 / steps as f64)).collect::<Vec<_>>().join(", ")
                );
                let part = |idx: &[usize], k: usize| mean(idx, &|i| rest[i][k]);
                let (nodes_us, force_us, photons_us) = (1e6 * part(&calm, 0), 1e6 * part(&calm, 1), 1e6 * part(&calm, 2));
                out += &format!(
                    "      the rest of the step (µs / multiply-adds, the steps of one round): nodes {nodes_us:.2}/{:.0}, tangent and force {force_us:.2}/{:.0}, photons {photons_us:.2}/{:.0}; the step's model (forward + nodes + force + photons) {:.2} µs ({:.2} µs with every step's refresh); the decompositions' cost an atom (its first step's and its refreshes'): {:.2} ms",
                    part(&calm, 3),
                    part(&calm, 4),
                    part(&calm, 5),
                    us(&calm) + nodes_us + force_us + photons_us,
                    us(&every) + 1e6 * (part(&every, 0) + part(&every, 1) + part(&every, 2)),
                    1e-3 * phase_us(&every, 1) * steps as f64 / paths.len() as f64
                );
                let tangent_us = 1e6 * mean(&calm, &|i| tangent[i].2);
                out += &format!(
                    "      the tangent {tangent_us:.2} µs, the force's contraction {:.2} µs (top, light, compressions, responses, held: {}); the tangent by phase (µs / multiply-adds): {}
",
                    force_us - tangent_us,
                    (0..5).map(|k| format!("{:.2}", 1e-3 * ns_tick * mean(&calm, &|i| tangent[i].3[k] as f64))).collect::<Vec<_>>().join(", "),
                    (0..6).map(|k| format!("{} {:.2}/{:.0}", TANGENT[k], 1e-3 * ns_tick * mean(&calm, &|i| tangent[i].0[k] as f64), mean(&calm, &|i| tangent[i].1[k]))).collect::<Vec<_>>().join(", ")
                );
                let whole = us(&calm) + nodes_us + force_us + photons_us;
                each_pass.push(whole);
                if best.as_ref().is_none_or(|b| whole < b.0) {
                    best = Some((whole, out));
                }
            }
            each_pass.sort_by(f64::total_cmp);
            {
                let mut cache = Cache::default();
                for r in &paths[0] {
                    fold.prototype(*r, kappa, &mut cache, &mut ws).expect("the prototype's step");
                }
                println!("      ∂K̃'s entries at the rounding in the kept basis (rows, its pattern's blocks, the zeros' share), the first atom's last step: {:?}", ws.dk_zeros(&cache));
            }
            println!("{}", best.unwrap().1);
            println!("      the passes' µs a step (steps of one round), sorted: {:?}", each_pass.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>());
        }
    }
}

/// The decay of a prototype step's generator against its own jumps at a position (DESIGN §2.6): the largest entry of
/// i(G − G†) − Σ L†L over the scale of G and Σ L†L.
fn proto_decay(generator: &Mat, jumps: &[[Mat; 3]]) -> f64 {
    let n = generator.rows;
    let mut products = Mat::zeros(n, n);
    for js in jumps {
        for j in js {
            products.add_scaled(C64::ONE, &j.adjoint_product(j));
        }
    }
    let mut gap = Mat::from_fn(n, n, |i, j| C64::I * (generator[(i, j)] - generator[(j, i)].conj()));
    let scale = generator.frobenius().max(products.frobenius());
    gap.add_scaled(-C64::ONE, &products);
    gap.max_abs() / scale
}

/// A Λ whose two ground states turn against each other (BRIEF_WIRE2 item 1's tone): |0⟩ at 0 and |4⟩ at 100 rad/µs, one
/// excited state (17) at 1000 rad/µs decaying at Γ to |0⟩, laser A (the trap's axis, a plane wave) on |0⟩ at 0.5 rad/µs and
/// laser B on |4⟩ at 0.4 rad/µs a standing wave along z, both on resonance: the Raman coupling is resonant, so |0⟩ and |4⟩
/// are one set, in frames 100 rad/µs apart (ν_A − ν_B), and B's outside form between them, 0.8 rad/µs times its intensity
/// over a beam's, an entry of H_GG at their frames' beat -- a tone at ±100 rad/µs whose envelope is B's standing wave.
fn toned_lambda() -> Request {
    let (e4, ee) = (100.0, 1000.0);
    let rays = [
        Ray { axis: 0, nu: ee, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(0.5))] },
        Ray { axis: 1, nu: ee - e4, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 4, C64::real(0.4))] },
        Ray { axis: 1, nu: ee - e4, direction: [0.0, 0.0, -1.0], couplings: vec![(17, 4, C64::real(0.4))] },
    ];
    let mut req = system(&[(0, 0.0), (4, e4)], &[(17, ee, GAMMA, 0)], &rays, 2, false);
    // |E_π|² of one beam is 10⁶ (`system`): the form's entry 0.8·10⁻⁶ gives 0.8 rad/µs a beam's intensity.
    let b = req.outside_forms.iter().position(|f| f.frequency_index == vec![0, 1]).unwrap();
    req.outside_forms[b].matrices[1][(0, 4)] = C64::real(0.8e-6);
    req.outside_forms[b].matrices[1][(4, 0)] = C64::real(0.8e-6);
    req.validate().unwrap();
    req
}

/// The prototype's force (proto.rs `Fold::prototype_force`, BRIEF_PROTO2 item 1; BRIEF_WIRE2 item 1) against the
/// high-precision difference of its contracted generator, every position moved by δ: on channel 0
/// f(δ) = Σ_j Re tr(Ḡ_j†G_j) + Σ_m,j Re tr(X_m,j G_m(j)) over the step's middle and outer nodes (τ = 0.65 µs, the path at
/// 0.1 m/s), random Hermitian cotangents Ḡ_j and random complex X_m,j of every tone; on channel 1 + m the complex
/// tr(Ḡ′ G_m(h)) of each tone at the outer node at h (a random density Ḡ′: a tone's complex amplitude). Each evaluation
/// on a fresh cache (its decompositions made at the point, where the reuse leaves nothing), its discrete choices (sweeps,
/// held rows, copies' rows) those of the centre -- a point where a difference meets another choice is skipped. Within the
/// difference's error and what the kept R₀ and z₀ leave (`Force::held`). On PGC3D_1ms (refs), R4_LIN_4, a Λ with a tone
/// (`toned_lambda`) and unify's Λ (`lambda_request`: its excited level in P, a block of no Q rows) by their prototype
/// choices (`proto_choice`), every set the prototype holds, three points each; the decay is its jumps' products at all
/// three positions (`proto_decay`).
#[test]
fn the_prototype_s_force_is_the_difference_of_its_generator() {
    use super::proto::{Bar, Cache};
    let pgc = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("PGC3D_1ms.json").to_string_lossy().into_owned();
    let (mut checked, mut toned) = (0, 0);
    for (name, req) in [(pgc.clone(), crate::model::testing::scene(&pgc)), ("R4_LIN_4".to_string(), crate::model::testing::scene("R4_LIN_4")), ("a Λ with a tone".to_string(), toned_lambda()), ("a Λ with its excited level in P".to_string(), lambda_request(3.0, 1.5, 0.0))] {
        let choice = proto_choice(&req, &name);
        let mut seed = 23u64;
        let mut u = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for set in Fold::sets(&req, &choice).unwrap() {
            let fold = Fold::new(&req, &choice, set[0]).unwrap();
            if fold.blocks.is_empty() {
                continue;
            }
            let mut ws = match fold.prototype_workspace() {
                Ok(w) => w,
                Err(e) => {
                    println!("{name} set {set:?}: not the prototype's ({e})");
                    continue;
                }
            };
            let n = fold.size();
            let tones = ws.tones().count();
            let speed = 0.1 / fold.hbar_over_mass;
            let times = [-0.325, 0.325];
            for point in 0..3 {
                let r0: [f64; 3] = std::array::from_fn(|_| 0.6 * u() - 0.3);
                let (ct, phi) = (2.0 * u() - 1.0, 2.0 * std::f64::consts::PI * u());
                let st = (1.0 - ct * ct).sqrt();
                let kappa = [speed * st * phi.cos(), speed * st * phi.sin(), speed * ct];
                let velocity = kappa.map(|k| k * fold.hbar_over_mass);
                let mut density = || {
                    let psi: Vec<C64> = (0..n).map(|_| C64::new(u() - 0.5, u() - 0.5)).collect();
                    let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
                    Mat::from_fn(n, n, |i, j| psi[i] * psi[j].conj() / norm)
                };
                let bars: Vec<Mat> = (0..3).map(|_| density()).collect();
                let ends: Vec<Mat> = (0..tones).map(|_| density()).collect();
                let xs: Vec<Mat> = (0..3 * tones).map(|_| Mat::from_fn(n, n, |_, _| C64::new(u() - 0.5, u() - 0.5))).collect();
                // tr(X G) = Σ_ij X_ij G_ji.
                let tr = |x: &Mat, g: &Mat| (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| x[(i, j)] * g[(j, i)]).sum::<C64>();
                // Channel 0 and the tones' channels at the point moved by d.
                let value = |d: [f64; 3], ws: &mut super::proto::Workspace| -> (Vec<C64>, (usize, Vec<(Vec<bool>, bool)>)) {
                    let r: [f64; 3] = std::array::from_fn(|a| r0[a] + d[a]);
                    let mut cache = Cache::default();
                    let g0 = fold.prototype(r, kappa, &mut cache, ws).expect("the prototype").generator.clone();
                    fold.prototype_nodes(r, kappa, velocity, times, &cache, ws);
                    let real = |b: &Mat, g: &Mat| (0..b.data.len()).map(|i| (b.data[i].conj() * g.data[i]).re).sum::<f64>();
                    let mut f0 = real(&bars[0], &g0) + real(&bars[1], ws.node_generator(0)) + real(&bars[2], ws.node_generator(1));
                    for m in 0..tones {
                        for j in 0..3 {
                            f0 += tr(&xs[3 * m + j], ws.tone(m, j)).re;
                        }
                    }
                    let mut out = vec![C64::real(f0)];
                    out.extend((0..tones).map(|m| tr(&ends[m], ws.tone(m, 2))));
                    (out, ws.choices())
                };
                let (_, centre) = value([0.0; 3], &mut ws);
                let mut cache = Cache::default();
                let g0 = fold.prototype(r0, kappa, &mut cache, &mut ws).unwrap().generator.clone();
                fold.prototype_nodes(r0, kappa, velocity, times, &cache, &mut ws);
                let decay: Vec<f64> = [None, Some(0), Some(1)]
                    .iter()
                    .map(|&j| {
                        let g = match j {
                            None => g0.clone(),
                            Some(j) => ws.node_generator(j).clone(),
                        };
                        proto_decay(&g, &fold.prototype_jumps(&cache, &ws, j.map_or(super::proto::At::Middle, super::proto::At::Node)))
                    })
                    .collect();
                fold.prototype_tangent(&cache, &mut ws);
                let mut contraction: Vec<Bar> = (0..3).map(|j| Bar { channel: 0, position: j, tone: None, matrix: &bars[j].data }).collect();
                contraction.extend((0..3 * tones).map(|k| Bar { channel: 0, position: k % 3, tone: Some(k / 3), matrix: &xs[k].data }));
                contraction.extend((0..tones).map(|m| Bar { channel: 1 + m, position: 2, tone: Some(m), matrix: &ends[m].data }));
                let force = fold.prototype_force(&cache, &mut ws, &contraction, 1 + tones);
                let scale = force.forms[0].iter().fold(0.0_f64, |m, x| m.max(x.re.abs()));
                println!("{name} set {set:?} point {point}: the forms {:?} (held {:?}, {:.0} multiply-adds, {tones} tones); decay against the jumps {:?}", force.forms[0].map(|z| z.re), force.held, force.work, decay.iter().map(|d| format!("{d:.1e}")).collect::<Vec<_>>());
                assert!(decay.iter().all(|&d| d <= 1e-12), "the decay against the jumps {decay:?}");
                let mut skipped = false;
                'axes: for a in 0..3 {
                    // Per channel the real part, then the tones' imaginary parts.
                    let mut quotients: Vec<Vec<f64>> = vec![vec![]; 1 + 2 * tones];
                    for h in crate::model::testing::steps() {
                        let (mut p, mut m) = ([0.0; 3], [0.0; 3]);
                        p[a] = h;
                        m[a] = -h;
                        let ((fp, cp), (fm, cm)) = (value(p, &mut ws), value(m, &mut ws));
                        if cp != centre || cm != centre {
                            skipped = true;
                            break 'axes;
                        }
                        for c in 0..1 + tones {
                            let q = (fp[c] - fm[c]) / (2.0 * h);
                            quotients[c].push(q.re);
                            if c > 0 {
                                quotients[tones + c].push(q.im);
                            }
                        }
                    }
                    for (c, quotients) in quotients.iter().enumerate() {
                        let form = if c <= tones { force.forms[c][a].re } else { force.forms[c - tones][a].im };
                        let estimate = crate::model::testing::converged(quotients, 1);
                        // The finest three quotients extrapolated twice (their error h², then h⁴): the limit to the rounding
                        // the coarse steps' estimate (`converged`) leaves where the light turns fast along the axis.
                        let q = &quotients[quotients.len() - 3..];
                        let (r1, r0) = (q[2] + (q[2] - q[1]) / 3.0, q[1] + (q[1] - q[0]) / 3.0);
                        let limit = r1 + (r1 - r0) / 15.0;
                        let size = if c == 0 { scale } else { force.forms[if c <= tones { c } else { c - tones }].iter().fold(0.0_f64, |m, z| m.max(z.abs())) };
                        println!("   ∂{a} channel {c}: the forms {form:.10e}, difference {:.10e} ± {:.1e} (extrapolated {limit:.10e}, ± {:.1e}), held {:.1e}", estimate.0, estimate.1, (limit - r1).abs(), force.held[a]);
                        // What the kept R₀ and z₀ leave: their relative change along the axis (`Force::held`) of the scale.
                        let held = force.held[a] * size;
                        assert!(crate::model::testing::agrees(form, (estimate.0, estimate.1 + held / 4.0), size), "{name} set {set:?} point {point} axis {a} channel {c}: the forms against the difference");
                        assert!((form - limit).abs() <= 4.0 * (limit - r1).abs() + held + 1e-8 * size, "{name} set {set:?} point {point} axis {a} channel {c}: the forms against the extrapolated difference");
                    }
                }
                if skipped {
                    println!("   another choice within the differences' steps: skipped");
                }
                checked += usize::from(!skipped);
                toned += usize::from(!skipped && tones > 0);
            }
        }
    }
    assert!(checked >= 4 && toned >= 1, "{checked} points checked, {toned} with tones");
}

/// The gradient's matrices (proto.rs `Fold::prototype_gradient`, the step's derivative in its forward mode; BRIEF_STEP3
/// item 3) are the force's (`Fold::prototype_force`, its reverse mode): at the step's middle and at its two outer nodes,
/// Re tr(Ĝ_a ρ) of a random density is its static bar's form, and of a tone's bar cρ its tone's matrices at c, to the
/// rounding of the two orders of summation (of the matrices' size, or of the generator's times the light's wave number
/// where the light's gradient vanishes there: a uniform light's). Every set of PGC3D_1ms (refs), a Λ with a tone, unify's
/// Λ with its excited level in P, R4_LIN_4 (its F = 1 set explicit), R6_RAMAN_CO, R6_RAMAN_COUNTER, R5_PUMP and
/// R1_TWEEZER, by their prototype choices, two moving points a set. (No set of these -- nor of GMD1C, tilt2 or S2Q -- takes
/// an exact copy or a centred copy's rows in at its points: those terms mirror the tangent's and the force's own.)
#[test]
fn the_gradient_s_matrices_are_the_force_s_forms() {
    use super::proto::{Bar, Cache};
    let pgc = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("PGC3D_1ms.json").to_string_lossy().into_owned();
    let (mut checked, mut toned, mut exact, mut copied, mut explicit, mut worst) = (0, 0, 0, 0, 0, 0.0_f64);
    let scenes = ["R4_LIN_4", "R6_RAMAN_CO", "R6_RAMAN_COUNTER", "R5_PUMP", "R1_TWEEZER"].map(|x| (x.to_string(), crate::model::testing::scene(x)));
    for (name, req) in [(pgc.clone(), crate::model::testing::scene(&pgc)), ("a Λ with a tone".to_string(), toned_lambda()), ("a Λ with its excited level in P".to_string(), lambda_request(3.0, 1.5, 0.0))].into_iter().chain(scenes) {
        let choice = proto_choice(&req, &name);
        let mut seed = 29u64;
        let mut u = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for set in Fold::sets(&req, &choice).unwrap() {
            let fold = Fold::new(&req, &choice, set[0]).unwrap();
            if fold.blocks.is_empty() {
                continue;
            }
            let mut ws = match fold.prototype_workspace() {
                Ok(w) => w,
                Err(e) => {
                    println!("{name} set {set:?}: not the prototype's ({e})");
                    continue;
                }
            };
            let n = fold.size();
            let tones = ws.tones().count();
            let speed = 0.1 / fold.hbar_over_mass;
            let k = fold.components.iter().flat_map(|c| c.kicks.iter()).map(|q| (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()).fold(0.0, f64::max);
            for _ in 0..2 {
                let r0: [f64; 3] = std::array::from_fn(|_| 0.6 * u() - 0.3);
                let (ct, phi) = (2.0 * u() - 1.0, 2.0 * std::f64::consts::PI * u());
                let st = (1.0 - ct * ct).sqrt();
                let kappa = [speed * st * phi.cos(), speed * st * phi.sin(), speed * ct];
                let velocity = kappa.map(|k| k * fold.hbar_over_mass);
                let mut cache = Cache::default();
                let middle = fold.prototype(r0, kappa, &mut cache, &mut ws).expect("the prototype").generator.clone();
                fold.prototype_nodes(r0, kappa, velocity, [-0.325, 0.325], &cache, &mut ws);
                fold.prototype_tangent(&cache, &mut ws);
                exact += usize::from(ws.exact_rows() > 0);
                copied += usize::from(ws.choices().1.iter().any(|x| x.1));
                explicit += usize::from(n > fold.ground.len());
                for j in 0..3 {
                    let psi: Vec<C64> = (0..n).map(|_| C64::new(u() - 0.5, u() - 0.5)).collect();
                    let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
                    let rho: Vec<C64> = (0..n * n).map(|i| psi[i / n] * psi[i % n].conj() / norm).collect();
                    // Re tr(Ĝ ρ) = Σ_pq Re(Ĝ_pq ρ_qp).
                    let contracted = |g: &[Vec<C64>; 3]| -> [f64; 3] { std::array::from_fn(|a| (0..n * n).map(|i| (g[a][i] * rho[(i % n) * n + i / n]).re).sum()) };
                    let frobenius = |m: &[C64]| m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                    let held = if j == 0 { frobenius(&middle.data) } else { frobenius(&ws.node_generator(j - 1).data) };
                    let toned = (0..tones).map(|m| frobenius(&ws.tone(m, j).data)).fold(0.0, f64::max);
                    let size = |g: &[Vec<C64>; 3]| -> f64 { g.iter().map(|m| frobenius(m)).fold(0.0, f64::max).max(k * held.max(toned)) };
                    let mut matrices: [Vec<C64>; 3] = Default::default();
                    fold.prototype_gradient(&cache, &mut ws, j, true, &[], &mut matrices);
                    let forms = fold.prototype_force(&cache, &mut ws, &[Bar { channel: 0, position: j, tone: None, matrix: &rho }], 1).forms[0];
                    let (ours, scale) = (contracted(&matrices), size(&matrices));
                    let mut miss: f64 = (0..3).map(|a| (ours[a] - forms[a].re).abs()).fold(0.0, f64::max) / scale.max(f64::MIN_POSITIVE);
                    for m in 0..tones {
                        let c = C64::cis(2.0 * std::f64::consts::PI * u());
                        let x: Vec<C64> = rho.iter().map(|z| *z * c).collect();
                        let mut coefficients = vec![C64::ZERO; tones];
                        coefficients[m] = c;
                        let mut toned_matrices: [Vec<C64>; 3] = Default::default();
                        fold.prototype_gradient(&cache, &mut ws, j, false, &coefficients, &mut toned_matrices);
                        let forms = fold.prototype_force(&cache, &mut ws, &[Bar { channel: 0, position: j, tone: Some(m), matrix: &x }], 1).forms[0];
                        let (ours, scale) = (contracted(&toned_matrices), size(&toned_matrices));
                        miss = miss.max((0..3).map(|a| (ours[a] - forms[a].re).abs()).fold(0.0, f64::max) / scale.max(f64::MIN_POSITIVE));
                    }
                    println!("{name} set {set:?} position {j}: Re tr(Ĝρ) {:+.6e} {:+.6e} {:+.6e}, the forms {:+.6e} {:+.6e} {:+.6e}: {miss:.1e} of ‖Ĝ‖ {scale:.2e} ({tones} tones)", ours[0], ours[1], ours[2], forms[0].re, forms[1].re, forms[2].re);
                    assert!(miss <= 1e-12, "{name} set {set:?} position {j}: the matrices against the forms, {miss:.2e} of their size");
                    worst = worst.max(miss);
                }
                checked += 1;
                toned += usize::from(tones > 0);
            }
        }
    }
    println!("{checked} points, {toned} with tones, {explicit} with explicit states, {exact} with exact copies, {copied} with centred copies' rows in: the largest miss {worst:.1e}");
    assert!(checked >= 8 && toned >= 1 && explicit >= 2, "{checked} points checked, {toned} with tones, {explicit} explicit");
}

/// Steps of R4_LIN_4's run from each of its ground sets, moving (BRIEF_STEP3 item 3): a step on a cluster that holds
/// explicit states (its F = 1 set's, its node risen where its fold does not exist: BRIEF_STEP3C item 1) makes the optical
/// sector's K of the gradient's matrices with their slope along the path -- not refused (`Refused::Optical`), the
/// covariance booked and a number; a step on one that holds none books none.
#[test]
fn an_explicit_set_s_step_makes_its_optical_sector() {
    struct Anywhere;
    impl crate::step::Domain for Anywhere {
        fn runaway(&self, _: f64) -> bool {
            false
        }
    }
    let req = crate::model::testing::scene("R4_LIN_4");
    let run = super::run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).expect("the run");
    let speed = 0.05 / crate::units::hbar_over_mass(req.mass_kg);
    let mut scratch = crate::step::Scratch::default();
    let mut explicit = 0;
    for set in 0..2 {
        let first = crate::step::Dynamics::cluster(&run, set).states[0];
        let mut t = crate::step::Trajectory::new(&run, first, [0.05, -0.03, 0.02], [speed, 0.4 * speed, -0.7 * speed], 7, set as u64);
        // The steps that stay on their cluster with no jump: each books its covariance where the cluster holds explicit
        // states, none where not.
        let mut stayed = 0;
        for _ in 0..8 {
            let (before, start, jumps) = (t.booked, t.cluster, t.jumps);
            let middle = crate::step::Midway::new(&run, &t, 0.126).expect("the reference's fixed point");
            t = middle.attempt(&t, &crate::step::Brownian::STILL, &[], &mut scratch, &Anywhere).expect("a step, its optical sector made").end;
            assert!(t.booked[..2].iter().all(|x| x.is_finite()) && t.pending.iter().flatten().all(|x| x.is_finite()), "the optical sector's covariance a number");
            let cluster = crate::step::Dynamics::cluster(&run, start);
            let holds = cluster.states.len() > cluster.ground;
            if t.cluster == start && t.jumps == jumps {
                assert_eq!(t.booked[1] != before[1], holds, "the optical sector where the cluster holds explicit states");
                stayed += 1;
                explicit += usize::from(holds);
            }
        }
        let cluster = crate::step::Dynamics::cluster(&run, t.cluster);
        println!("from set {set}: {stayed} steps on their cluster with no jump; at the end on {} states ({} ground); booked over 8 steps ε_Q tr C {:.3e}, tr C {:.3e}", cluster.states.len(), cluster.ground, t.booked[0], t.booked[1]);
    }
    assert!(explicit >= 1, "a step on R4_LIN_4's F = 1 set with its node risen");
}

/// The prototype's photons (proto.rs `Fold::prototype_photons`, BRIEF_PROTO2 item 2) of a random state at the step's middle:
/// their rate is the state's hazard ⟨ψ|i(G − G†)|ψ⟩/‖ψ‖² to the rounding (the jumps' amplitudes u = Lψ are the decay's);
/// a class's jump's amplitudes as they leave them (`Workspace::jumped`, a jump's at its place) are the operators' on the
/// state (`prototype_jumps`) to the rounding; and their moments, of the step's tangent, are those of the amplitudes' difference -- ∇u by the central difference of
/// u = L(r)ψ, each on a fresh cache (`prototype_jumps`; h = 10⁻⁴ µm), its responses' own change and S⁻¹'s with the
/// beams', its discrete choices those of the centre (else skipped) -- within the difference's error and what the kept R₀
/// leaves (10⁻⁴ of the force's and the diffusion's scale). On PGC3D_1ms (refs), R4_LIN_4 and unify's Λ (`lambda_request`:
/// a block of no Q rows, its photons its explicit row's) by their prototype choices, three points a set.
#[test]
fn the_prototype_s_photons_are_its_hazard_and_their_gradients_said() {
    use super::proto::Cache;
    let pgc = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("PGC3D_1ms.json").to_string_lossy().into_owned();
    let mut checked = 0;
    for (name, req) in [(pgc.clone(), crate::model::testing::scene(&pgc)), ("R4_LIN_4".to_string(), crate::model::testing::scene("R4_LIN_4")), ("a Λ with its excited level in P".to_string(), lambda_request(3.0, 1.5, 0.0))] {
        let choice = proto_choice(&req, &name);
        let sets = Fold::sets(&req, &choice).unwrap();
        let mut seed = 31u64;
        let mut u = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for set in &sets {
            let fold = Fold::new(&req, &choice, set[0]).unwrap();
            if fold.blocks.is_empty() {
                continue;
            }
            let Ok(mut ws) = fold.prototype_workspace() else { continue };
            let n = fold.size();
            let speed = 0.1 / fold.hbar_over_mass;
            for point in 0..3 {
                let r0: [f64; 3] = std::array::from_fn(|_| 0.6 * u() - 0.3);
                let (ct, phi) = (2.0 * u() - 1.0, 2.0 * std::f64::consts::PI * u());
                let st = (1.0 - ct * ct).sqrt();
                let kappa = [speed * st * phi.cos(), speed * st * phi.sin(), speed * ct];
                let psi: Vec<C64> = (0..n).map(|_| C64::new(u() - 0.5, u() - 0.5)).collect();
                let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
                // Per class (block, landing set) and emission axis, u on the set's rows.
                let amplitudes = |r: [f64; 3], ws: &mut super::proto::Workspace| -> (Vec<Vec<C64>>, (usize, Vec<(Vec<bool>, bool)>)) {
                    let mut cache = Cache::default();
                    fold.prototype(r, kappa, &mut cache, ws).expect("the prototype");
                    let mut out = vec![];
                    for js in fold.prototype_jumps(&cache, ws, super::proto::At::Middle) {
                        for set in &sets {
                            out.push((0..3).flat_map(|q| set.iter().map(move |&b| (q, b))).map(|(q, b)| (0..n).map(|j| js[q][(b, j)] * psi[j]).sum()).collect());
                        }
                    }
                    (out, ws.choices())
                };
                let mut cache = Cache::default();
                let g = fold.prototype(r0, kappa, &mut cache, &mut ws).unwrap().generator.clone();
                fold.prototype_tangent(&cache, &mut ws);
                let photons = fold.prototype_photons(&cache, &mut ws, &psi, &sets, super::proto::At::Middle, 0.0, super::proto::Moments::Every);
                let hazard: f64 = (0..n).map(|i| (0..n).map(|j| psi[i].conj() * C64::I * (g[(i, j)] - g[(j, i)].conj()) * psi[j]).sum::<C64>().re).sum::<f64>() / norm;
                assert!((photons.rate - hazard).abs() <= 1e-12 * hazard.abs().max(1e-300), "{name} set {set:?} point {point}: the photons' rate {} against the hazard {hazard}", photons.rate);
                // A class's jump's amplitudes as the photons leave them (`Workspace::jumped`), on its set's rows.
                let jumped: Vec<(usize, Vec<C64>)> = photons.classes.iter().map(|c| (c.landing, (0..3).flat_map(|q| sets[c.landing].iter().map(move |&b| (q, b))).map(|(q, b)| ws.jumped(c.slot, b, q)).collect())).collect();
                // A jump's order -- every class's rate, then the drawn class's moments alone (BRIEF_STEP3 item 6) -- is the
                // same formula: each class drawn, its rates, kick and amplitudes those of every class's moments.
                for (c, whole) in photons.classes.iter().enumerate() {
                    let drawn = fold.prototype_photons(&cache, &mut ws, &psi, &sets, super::proto::At::Middle, 0.0, super::proto::Moments::Drawn(&mut |_| Some(c)));
                    let one = &drawn.classes[c];
                    let rates: Vec<f64> = drawn.classes.iter().map(|x| x.rate).collect();
                    assert!(rates == photons.classes.iter().map(|x| x.rate).collect::<Vec<f64>>() && drawn.drawn == Some(c), "{name} set {set:?} point {point}: the rates before the draw");
                    let size = whole.kick.iter().map(|x| x.abs()).fold(0.0, f64::max);
                    let apart = (0..3).map(|a| (one.kick[a] - whole.kick[a]).abs()).fold(0.0, f64::max);
                    assert!(apart <= 1e-14 * size, "{name} set {set:?} point {point}: class {c}'s kick drawn {:?} against every class's {:?}", one.kick, whole.kick);
                    let amplitudes: Vec<C64> = (0..3).flat_map(|q| sets[one.landing].iter().map(move |&b| (q, b))).map(|(q, b)| ws.jumped(one.slot, b, q)).collect();
                    assert!(amplitudes == jumped[c].1, "{name} set {set:?} point {point}: class {c}'s amplitudes drawn");
                    assert!(drawn.work < photons.work || photons.classes.len() == 1, "{name} set {set:?} point {point}: a drawn class's work {} against every class's {}", drawn.work, photons.work);
                }
                // The whole gradient by the difference (h = 1e-4 µm, central).
                let h = 1e-4;
                let (u0, centre) = amplitudes(r0, &mut ws);
                // ... are the operators' on the state (every block a coherence group of its own here): each class's those of
                // one block landing in its set, to the rounding.
                let scale = u0.iter().flatten().map(|z| z.abs()).fold(0.0, f64::max);
                for (landing, a) in &jumped {
                    let apart = u0.iter().skip(*landing).step_by(sets.len()).map(|x| x.iter().zip(a).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max)).fold(f64::INFINITY, f64::min);
                    assert!(apart <= 1e-13 * scale, "{name} set {set:?} point {point}: a class's jump's amplitudes {apart:e} from the operators' (scale {scale:e})");
                }
                let mut other = false;
                let du: Vec<[Vec<C64>; 3]> = {
                    let mut per: Vec<[Vec<C64>; 3]> = u0.iter().map(|x| std::array::from_fn(|_| vec![C64::ZERO; x.len()])).collect();
                    for a in 0..3 {
                        let (mut p, mut m) = (r0, r0);
                        p[a] += h;
                        m[a] -= h;
                        let ((up, cp), (um, cm)) = (amplitudes(p, &mut ws), amplitudes(m, &mut ws));
                        other |= cp != centre || cm != centre;
                        for c in 0..u0.len() {
                            for i in 0..u0[c].len() {
                                per[c][a][i] = (up[c][i] - um[c][i]) / (2.0 * h);
                            }
                        }
                    }
                    per
                };
                if other {
                    println!("{name} set {set:?} point {point}: another choice within the difference's step: skipped");
                    continue;
                }
                // The moments by photon.rs's rule, k_c the class's block's (`Workspace::absorbed`).
                let kcs = ws.absorbed(&fold);
                let (mut force, mut diffusion) = ([0.0; 3], [[0.0; 3]; 3]);
                for (c, x) in u0.iter().enumerate() {
                    let kc = kcs[c / sets.len()];
                    let w: f64 = x.iter().map(|z| z.norm_sqr()).sum();
                    let mean: [f64; 3] = std::array::from_fn(|a| x.iter().zip(&du[c][a]).map(|(z, d)| (z.conj() * *d).im).sum());
                    let square = crate::linalg::dot3(mean, mean);
                    let kick = if square > (kc * w).powi(2) { mean.map(|m| kc * m / square.sqrt()) } else if w > 0.0 { mean.map(|m| m / w) } else { [0.0; 3] };
                    for a in 0..3 {
                        force[a] += (mean[a] - w * kick[a]) / norm;
                        for b in 0..3 {
                            let second: f64 = du[c][a].iter().zip(&du[c][b]).map(|(p, q)| (p.conj() * *q).re).sum();
                            diffusion[a][b] += (second - w * kick[a] * kick[b]) / norm;
                        }
                    }
                }
                let trace = |d: &[[f64; 3]; 3]| d[0][0] + d[1][1] + d[2][2];
                let fsize = force.iter().map(|x| x * x).sum::<f64>().sqrt().max(photons.force.iter().map(|x| x * x).sum::<f64>().sqrt());
                let fgap = (0..3).map(|a| (photons.force[a] - force[a]).powi(2)).sum::<f64>().sqrt();
                let dgap = (0..3).flat_map(|a| (0..3).map(move |b| (a, b))).map(|(a, b)| (photons.diffusion[a][b] - diffusion[a][b]).abs()).fold(0.0, f64::max);
                println!(
                    "{name} set {set:?} point {point}: rate {:.4e} (hazard {hazard:.4e}); the force the jumps leave {:?} against the whole gradient's {:?} ({:.1e} of its size); 2D's trace {:.4e} against {:.4e}, its largest entry apart {:.1e} of the trace; {:.0} multiply-adds",
                    photons.rate,
                    photons.force.map(|x| format!("{x:.3e}")),
                    force.map(|x| format!("{x:.3e}")),
                    fgap / fsize.max(1e-300),
                    trace(&photons.diffusion),
                    trace(&diffusion),
                    dgap / trace(&diffusion).abs(),
                    photons.work
                );
                assert!(fgap <= 1e-4 * fsize.max(trace(&diffusion).abs().sqrt() * photons.rate.sqrt()) && dgap <= 1e-4 * trace(&diffusion).abs(), "{name} set {set:?} point {point}: the moments against the difference's");
                checked += 1;
            }
        }
    }
    assert!(checked >= 6);
}

/// The cross-level series (coordinator's note on the parts, 2026-11-16; not a test, a measurement: `PROTO_SCENE`): per set
/// and near block of the prototype's choice, the block's K on its solved excited rows split by excited level, K₀ (within
/// each level) and R = K − K₀ (across levels), at the focus (the tweezer's centre) and twenty random points (±0.3 µm,
/// 0.1 m/s): per column j and lane (beam w_l, or the static one) D₀ = (c_j + o − w − K₀)⁻¹ at `Fold::at`'s column energy
/// c_j = H_jj; the operator's size ‖D₀R‖_F, and on the exact response column T_j the first and second orders the series
/// would add, ‖D₀RT_j‖/‖T_j‖ and ‖(D₀R)²T_j‖/‖T_j‖, the largest over columns and lanes; and the level pair carrying the
/// most of R (‖D₀_P R_PQ‖_F).
#[test]
#[ignore = "a measurement: run by name with PROTO_SCENE"]
fn the_levels_cross_series() {
    let name = std::env::var("PROTO_SCENE").expect("PROTO_SCENE");
    let req = crate::model::testing::scene(&name);
    let choice = proto_choice(&req, &name);
    let mut seed = 11u64;
    let mut u = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for set in Fold::sets(&req, &choice).unwrap() {
        let fold = Fold::new(&req, &choice, set[0]).unwrap();
        if fold.blocks.is_empty() {
            continue;
        }
        let n = fold.size();
        let speed = 0.1 / fold.hbar_over_mass;
        let mut points: Vec<([f64; 3], [f64; 3])> = vec![([0.0; 3], [0.0; 3])];
        for _ in 0..20 {
            let r: [f64; 3] = std::array::from_fn(|_| 0.6 * u() - 0.3);
            let (ct, phi) = (2.0 * u() - 1.0, 2.0 * std::f64::consts::PI * u());
            let st = (1.0 - ct * ct).sqrt();
            points.push((r, [speed * st * phi.cos(), speed * st * phi.sin(), speed * ct]));
        }
        for (o, b) in fold.blocks.iter().enumerate().filter(|(_, b)| b.solved > 0) {
            let ne = b.rows[..b.solved].iter().filter(|r| matches!(r, Row::Excited(_))).count();
            if ne == 0 {
                continue;
            }
            let lv: Vec<usize> = b.rows[..ne].iter().map(|r| if let Row::Excited(x) = *r { fold.excited_level[x] } else { unreachable!() }).collect();
            let mut levels = lv.clone();
            levels.sort();
            levels.dedup();
            let (mut op, mut first, mut second) = (0.0_f64, 0.0_f64, 0.0_f64);
            let mut pairs: Vec<((usize, usize), f64)> = vec![];
            for (k, &(r, kappa)) in points.iter().enumerate() {
                let Some(f) = fold.at(r, kappa, &mut Path::default()) else { continue };
                let mut w = 0.0;
                let site = fold.site(r, kappa, &mut w);
                let laid = fold.lay(o, &site, &mut w);
                let Some(e) = fold.effective(o, &laid, &mut w) else { continue };
                let er: Vec<usize> = (0..ne).collect();
                let kk = e.k.block(&er, &er);
                let k0 = Mat::from_fn(ne, ne, |i, j| if lv[i] == lv[j] { kk[(i, j)] } else { C64::ZERO });
                let mut rr = kk.clone();
                rr.add_scaled(-C64::ONE, &k0);
                let t = &f.stages.last().unwrap().responses[o];
                let mut lanes: Vec<f64> = laid.shifts.clone();
                if b.explicit.is_empty() {
                    lanes.remove(0);
                }
                for j in 0..n {
                    let col = Mat::from_fn(ne, 1, |i, _| t[(i, j)]);
                    let size = col.frobenius();
                    if size == 0.0 {
                        continue;
                    }
                    for &wl in &lanes {
                        let z = f.h[2][(j, j)] + C64::real(b.value - wl);
                        let Some(d0) = inverse(&Mat::from_fn(ne, ne, |a, c| if a == c { z } else { C64::ZERO } - k0[(a, c)]), &mut w) else { continue };
                        let dr = d0.product(&rr);
                        op = op.max(dr.frobenius());
                        let one = dr.product(&col);
                        first = first.max(one.frobenius() / size);
                        second = second.max(dr.product(&one).frobenius() / size);
                        for &p in &levels {
                            for &q in levels.iter().filter(|&&q| q != p) {
                                let x: f64 = (0..ne).filter(|&i| lv[i] == p).flat_map(|i| (0..ne).filter(|&c| lv[c] == q).map(move |c| (i, c))).map(|(i, c)| dr[(i, c)].norm_sqr()).sum::<f64>().sqrt();
                                match pairs.iter_mut().find(|y| y.0 == (p, q)) {
                                    Some(y) => y.1 = y.1.max(x),
                                    None => pairs.push(((p, q), x)),
                                }
                            }
                        }
                    }
                }
                if k == 0 {
                    println!("{name} set {set:?} block {o} ({ne} excited rows, levels {levels:?}) at the focus: ‖D₀R‖ {op:.2e}, first order {first:.2e}, second {second:.2e}");
                }
            }
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            let top: Vec<String> = pairs.iter().take(4).map(|((p, q), x)| format!("{:?}←{:?} {x:.1e}", fold.levels[*p], fold.levels[*q])).collect();
            println!("{name} set {set:?} block {o}: over the focus and 20 points, ‖D₀R‖ {op:.2e}, first order {first:.2e}, second {second:.2e}; the pairs (‖D₀_P R_PQ‖) {top:?}");
        }
    }
}

/// The run on the fold at the atom (run.rs) against the exact master equation of a Λ (unify part 2's): |1⟩ at 0 and |2⟩ at
/// 100 rad/µs, one excited state decaying at Γ = 38 rad/µs, three quarters to |1⟩ and a quarter to |2⟩ (on two axes),
/// laser A on |1⟩ (`a` rad/µs) and B on |2⟩ (`b`), both on resonance. Where `form` is not zero B is a standing wave along z
/// (two beams of `b` each) with an outside form `form` between |1⟩ and |2⟩ (per unit |E_π|², a beam's 10⁶): H_GG's entry
/// c(r) = form (|E_B(r)|² − |E_B(0)|²) (H0 holds the forms at the focus), in their frames 100 rad/µs apart a tone at
/// ±100 rad/µs (run.rs # The tones), and the atom at rest at kz = π/4, where B's intensity is half its focus's. The two
/// ground states are one set (their Raman coupling resonant), both lasers reach one block, its jumps are one operator, and
/// a jump lands on both states at their frames' phases (run.rs # The jumps' phase): 400 trajectories from |1⟩ (the atom
/// 10⁶ times Rb's mass, at rest), steps of 0.25 µs through the integrator and its jumps, read at `time` µs. Returns the run's
/// mean |1⟩ share, its standard error, the exact master equation's (RK4 in the lasers' frame, the beams' amplitudes at the
/// atom) with its own error (against the steps halved), and the dark state's |b|²/(|a|² + |b|²) there.
/// The Λ's request (`lambda_against_master`): A of `a` rad/µs, B of `b` (two counter-propagating beams where `form` is not
/// zero, with the form between |1⟩ and |2⟩).
fn lambda_request(a: f64, b: f64, form: f64) -> Request {
    let (e2, gamma) = (100.0, 38.0);
    let (to1, to2) = (0.75 * gamma, 0.25 * gamma);
    // The excited state at 1000 rad/µs in its frame, the lasers on resonance there (the idle excited states at zero, no
    // laser's energy on them).
    let ee = 1000.0;
    let mut rays = vec![Ray { axis: 0, nu: ee, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(a))] }, Ray { axis: 1, nu: ee - e2, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 4, C64::real(b))] }];
    if form != 0.0 {
        rays.push(Ray { axis: 1, nu: ee - e2, direction: [0.0, 0.0, -1.0], couplings: vec![(17, 4, C64::real(b))] });
    }
    let mut req = system(&[(0, 0.0), (4, e2)], &[(17, ee, gamma, 0)], &rays, 2, true);
    req.emission = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
    req.emission[0][(0, GROUND + 17)] = C64::real(to1.sqrt());
    req.emission[1][(4, GROUND + 17)] = C64::real(to2.sqrt());
    if form != 0.0 {
        let f = req.outside_forms.iter().position(|f| f.frequency_index == vec![0, 1]).unwrap();
        req.outside_forms[f].matrices[1][(0, 4)] = C64::real(form);
        req.outside_forms[f].matrices[1][(4, 0)] = C64::real(form);
    }
    req.validate().unwrap();
    req
}

fn lambda_against_master(a: f64, b: f64, form: f64, time: f64) -> (f64, f64, f64, f64, f64, usize) {
    use crate::step::{Dynamics, Midway, Trajectory};
    let (e2, gamma): (f64, f64) = (100.0, 38.0);
    let (to1, to2) = (0.75 * gamma, 0.25 * gamma);
    let mut req = lambda_request(a, b, form);
    // Every excited state decays, as in every request the app makes: the idle ones (no laser on them) at Γ each by a
    // channel of its own (`idle_decay`) -- never populated, they leave the Λ's dynamics as they are, and an explicit set's
    // optical sector relaxes in every direction (step.rs # The optical sector's noise: a direction that does not relax
    // makes no K, and the step is refused).
    idle_decay(&mut req, gamma);
    let rays = req.beams.len();
    let k = req.beams[1].wave_number;
    let at = if form != 0.0 { [0.0, 0.0, std::f64::consts::FRAC_PI_4 / k] } else { [0.0; 3] };
    let mut p = [0.0; GROUND];
    p[0] = 1.0;
    let run = run::Onefold::new(&req, p).expect("the run");
    assert_eq!(run.cluster_of(0), run.cluster_of(4), "|1⟩ and |2⟩ one set");
    // The couplings at the atom: each beam's amplitude there, and the form's entry less its value at the focus.
    let light = Light::new(&req);
    let amplitude = |l: usize, r: [f64; 3]| light.beams[l].amplitude_to(r, 1).value;
    let (ae, be) = (amplitude(0, at) * a, (1..rays).map(|l| amplitude(l, at)).sum::<C64>() * b);
    let intensity = |r: [f64; 3]| 1e6 * (1..rays).map(|l| amplitude(l, r)).sum::<C64>().norm_sqr();
    let c = form * (intensity(at) - intensity([0.0; 3]));
    // The exact three-level master equation in the lasers' frame, from |1⟩⟨1|, at 60 µs (RK4): |2⟩'s frame 100 rad/µs
    // above |1⟩'s, so c turns as e^{∓i100t}.
    let h3 = |t: f64| {
        let turn = C64::cis(-e2 * t) * c;
        Mat::from_fn(3, 3, |i, j| match (i, j) {
            (2, 0) => ae,
            (0, 2) => ae.conj(),
            (2, 1) => be,
            (1, 2) => be.conj(),
            (0, 1) => turn,
            (1, 0) => turn.conj(),
            _ => C64::ZERO,
        })
    };
    let c3 = [Mat::from_fn(3, 3, |i, j| if (i, j) == (0, 2) { C64::real(to1.sqrt()) } else { C64::ZERO }), Mat::from_fn(3, 3, |i, j| if (i, j) == (1, 2) { C64::real(to2.sqrt()) } else { C64::ZERO })];
    let lindblad = |t: f64, rho: &Mat| -> Mat {
        let h = h3(t);
        let mut out = h.product(rho);
        out.add_scaled(-C64::ONE, &rho.product(&h));
        let mut out = out.scaled(C64::new(0.0, -1.0));
        for j in &c3 {
            out.add_scaled(C64::ONE, &j.product(rho).product(&j.adjoint()));
            let jj = j.adjoint_product(j);
            out.add_scaled(C64::real(-0.5), &jj.product(rho));
            out.add_scaled(C64::real(-0.5), &rho.product(&jj));
        }
        out
    };
    let share = |dt: f64| {
        let mut rho = Mat::zeros(3, 3);
        rho[(0, 0)] = C64::ONE;
        for k in 0..(time / dt).round() as usize {
            let t = k as f64 * dt;
            let k1 = lindblad(t, &rho);
            let mut x = rho.clone();
            x.add_scaled(C64::real(0.5 * dt), &k1);
            let k2 = lindblad(t + 0.5 * dt, &x);
            let mut x = rho.clone();
            x.add_scaled(C64::real(0.5 * dt), &k2);
            let k3 = lindblad(t + 0.5 * dt, &x);
            let mut x = rho.clone();
            x.add_scaled(C64::real(dt), &k3);
            let k4 = lindblad(t + dt, &x);
            for (k, w) in [(&k1, 1.0), (&k2, 2.0), (&k3, 2.0), (&k4, 1.0)] {
                rho.add_scaled(C64::real(dt * w / 6.0), k);
            }
        }
        rho[(0, 0)].re / (rho[(0, 0)].re + rho[(1, 1)].re)
    };
    let (exact, finer) = (share(2e-4), share(1e-4));
    struct Anywhere;
    impl crate::step::Domain for Anywhere {
        fn runaway(&self, _: f64) -> bool {
            false
        }
    }
    let (tau, count) = (0.25, 400);
    let steps = (time / tau).round() as usize;
    let mut shares = vec![];
    let mut scratch = crate::step::Scratch::default();
    let (mut explicit, mut switches, mut dropped) = (0, 0, 0.0);
    for i in 0..count {
        let mut t = Trajectory::new(&run, 0, at, [0.0; 3], 11, i);
        for _ in 0..steps {
            let before = t.cluster;
            t = Midway::new(&run, &t, tau).unwrap().attempt(&t, &crate::step::Brownian::STILL, &[], &mut scratch, &Anywhere).unwrap().end;
            let c = run.cluster(t.cluster);
            explicit += usize::from(c.states.len() > c.ground);
            switches += usize::from(t.cluster != before && run.cluster(before).ground == c.ground);
        }
        dropped += t.dropped;
        let rows = run.readout(t.cluster, &t.point(), &t.psi, &t.held);
        shares.push(rows[0] / (rows[0] + rows[4]));
    }
    println!("the run's steps' ends: {explicit} of {} explicit, {switches} switches, the switches' booked rings {dropped:.3e} over the trajectories", count as usize * steps);
    let mean = shares.iter().sum::<f64>() / count as f64;
    let error = (shares.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count * (count - 1)) as f64).sqrt();
    (mean, error, finer, (exact - finer).abs(), be.norm_sqr() / (ae.norm_sqr() + be.norm_sqr()), explicit)
}

/// The run's Λ (`lambda_against_master`) without the coupling, the lasers at unify's (3 and 1.5 rad/µs): its fold with
/// nothing explicit holds it at every step (BRIEF_LANDIMPL: its ring 1.6·10⁻³, no node rises). The jumps' coherence makes
/// the dark state b|1⟩ − a|2⟩ (coherent population trapping), |1⟩'s share 0.2, where the rate equations, each laser's
/// photons apart, give 0.43: at 60 µs the master equation is there to 10⁻¹⁰ and every trajectory of the run in the fold's
/// dark mode, within the share (its series' rest: 0.1996), and on the way, at 10 µs, their mean |1⟩ shares agree within
/// three standard errors.
#[test]
fn the_run_s_lambda_is_the_master_equation_s_dark_state() {
    for time in [10.0, 60.0] {
        let (mean, error, exact, own, dark, explicit) = lambda_against_master(3.0, 1.5, 0.0, time);
        assert_eq!(explicit, 0, "the Λ's fold rings within the share: no node rises");
        println!("|1⟩'s share at {time} µs: the run {mean:.6} ± {error:.1e} over 400 trajectories, the exact master equation {exact:.6} ± {own:.1e} (the dark state's {dark:.4}; the rate equations' 0.43)");
        if time < 60.0 {
            assert!((mean - exact).abs() <= 3.0 * error + own, "the run's |1⟩ share {mean} ± {error} against the master equation's {exact} ± {own}");
        } else {
            assert!((mean - dark).abs() <= crate::model::elim::share() && error <= 1e-10 && (exact - dark).abs() <= 1e-10, "at {time} µs the run's {mean} ± {error} and the master equation's {exact} against the dark state's {dark}");
        }
    }
}

/// The run's Λ with a tone (BRIEF_WIRE2 item 1; `lambda_against_master`): A as there, B a standing wave of two beams of
/// 0.4 rad/µs (its fold the rule's at the focus, where B is 0.8 rad/µs), and a form
/// whose coupling between |1⟩ and |2⟩ at the atom is c = 10 rad/µs -- in their frames a tone at ±100 rad/µs (νh = 12.5 at
/// the steps of 0.25 µs: a tone of the integrator), its light shift c²/100 = 1 rad/µs on the two-photon resonance, beyond
/// the dark state's width (a² + b²)/Γ ≈ 0.07 rad/µs: the dark state is lifted -- |1⟩'s share moves from its 0.12 -- and the
/// run follows the exact master equation within three standard errors and the elimination's (a/(Γ/2))².
#[test]
fn the_run_s_lambda_with_a_tone_is_the_master_equation_s() {
    let (a, gamma) = (1.5, 38.0);
    // B's intensity at the atom half its focus's, 2·10⁶ less: c = −2·10⁶ form.
    let (mean, error, exact, own, dark, _) = lambda_against_master(a, 0.4, -5e-6, 60.0);
    let elimination = (a / (0.5 * gamma)).powi(2);
    println!("with the tone: |1⟩'s share at 60 µs: the run {mean:.4} ± {error:.4} over 400 trajectories, the exact master equation {exact:.4} ± {own:.1e} (the dark state's {dark:.4}; the elimination's (a/(Γ/2))² {elimination:.4})");
    assert!((exact - dark).abs() > 3.0 * error + elimination + own, "the tone moves the share beyond what the run is held to: {exact} against the dark state's {dark}");
    assert!((mean - exact).abs() <= 3.0 * error + elimination + own, "the run's |1⟩ share {mean} ± {error} against the master equation's {exact}");
}

/// The jumps' coherence (run.rs # The jumps' coherence; BRIEF_WIRE3 item 3): |1⟩ at 0 (|2⟩ at 100 rad/µs, lit by
/// nothing), an excited state e at 1000 rad/µs in its frame (Γ = 38 rad/µs: 1 to |1⟩, the rest to |2⟩), A on |1⟩ → e
/// (10 rad/µs) on its resonance, and a second excited level e′ (D1's F′ = 1, e being D2's) 10 rad/µs above e decaying
/// to |1⟩ at 25 rad/µs on e's axis, driven from |1⟩ by a laser C (10 rad/µs) on its resonance. Both excited levels in P,
/// each at its offset (a tiered choice), the offsets 100 rad/µs apart; D_ee′ = 10 rad/µs, so P holds a tone between them
/// (its mixing |D_ee′|/2 over their 100 rad/µs, 0.05, beyond the share; at 10 rad/µs apart the copies of the ground state
/// the two drives make at ±10 rad/µs mix at 1 and the fold does not exist -- its series' ratio 58 --, BRIEF_STEP3C item 1):
/// their blocks are one coherence group. At the step's three positions and at times across the beat, for random states
/// of the set: the photons' rate at the jump's place (`Middle::emitted`'s classes, the light turned there) is the hazard
/// ψ†i(G − G†)ψ/ψ†ψ of the generator at that node (`Middle::generator`) to the rounding: the decay is the jumps' product at
/// every time, its beat between the offsets with it. (With the jumps per block and the tone's decay in G they differed by
/// the cross decay, a fraction of the hazard.) And the beating decay is the integrator's candidates' (`Middle::beats`,
/// `Middle::envelopes`): the generator less every candidate at its phase decays alike at every time across the beat, at
/// each of the step's three positions, to the rounding.
#[test]
fn a_set_with_explicit_states_at_two_offsets_decays_by_its_jumps() {
    use crate::step::{Dynamics, Middle};
    let (gamma, ee, e2) = (38.0, 1000.0, 100.0);
    let rays = [
        Ray { axis: 0, nu: ee, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(10.0))] },
        Ray { axis: 1, nu: ee + 100.0, direction: [1.0, 0.0, 0.0], couplings: vec![(0, 0, C64::real(10.0))] },
    ];
    let mut req = system(&[(0, 0.0), (4, e2)], &[(17, ee, gamma, 0), (0, ee + 100.0, gamma, 0)], &rays, 2, true);
    req.emission = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
    req.emission[0][(0, GROUND + 17)] = C64::real(2.0);
    req.emission[1][(4, GROUND + 17)] = C64::real((gamma - 4.0).sqrt());
    req.emission[0][(0, GROUND)] = C64::real(25.0_f64.sqrt());
    req.validate().unwrap();
    let mut p = [0.0; GROUND];
    p[0] = 1.0;
    // Both levels in P at their own offsets (the rule at the focus would want each at both: through D_ee′ each is the
    // other's block's Q row, beyond the share where the tone is -- a fold keeps a level at one offset).
    let level = |x: usize| Fold::new(&req, &Choice::tiered(&req, vec![]).unwrap(), 0).unwrap().excited_level[x];
    let part = |frequency: usize| crate::model::Component { frequency, counter: false };
    let choice = Choice::tiered(&req, vec![(level(17), part(0), 0), (level(0), part(1), 0)]).expect("the choice");
    let run = super::run::Onefold::of(&req, &choice, p).expect("the run");
    let fold = Fold::new(&req, &choice, 0).unwrap();
    let joined: Vec<usize> = (0..fold.blocks.len()).filter(|&b| fold.groups[b] != b).collect();
    assert!(!joined.is_empty(), "the explicit levels' blocks one coherence group: {:?}", fold.groups);
    let set = run.cluster_of(0);
    let cluster = run.cluster(set);
    let n = cluster.states.len();
    println!("the set's states {:?}, their frames {:?}", cluster.states, cluster.frames);
    assert!(cluster.states.contains(&GROUND) && cluster.states.contains(&(GROUND + 17)), "both excited levels in P: {:?}", cluster.states);
    let (h, mut worst, mut beat) = (0.1, 0.0_f64, 0.0_f64);
    let mut stream = crate::random::Stream::new(41, crate::random::Purpose::Initial, 0, 0);
    // The hazard of one state at the middle across the beat: its beating part, the cross decay.
    let probe: Vec<C64> = (0..n).map(|i| C64::new(1.0 + i as f64, 0.3 * i as f64)).collect();
    let hazards: Vec<f64> = [0.0, 0.07, 0.19, 0.31]
        .iter()
        .map(|&t| {
            let middle = run.middle(set, [0.3, 0.0, 0.0], t, &crate::step::Held::default(), false);
            let mut rate = 0.0;
            let _ = middle.emitted(&middle.path([0.0; 3], [2.0, 0.0, 0.0]), 0.0, &probe, &mut |rates: &[f64]| {
                rate = rates.iter().sum();
                None
            });
            rate
        })
        .collect();
    let swing = (hazards.iter().copied().fold(f64::NEG_INFINITY, f64::max) - hazards.iter().copied().fold(f64::INFINITY, f64::min)) / hazards[0];
    println!("one state's hazard across the beat {hazards:.4?}: its swing {swing:.3} of it, the cross decay");
    assert!(swing > 1e-3, "the cross decay beats: {swing}");
    for t in [0.0, 0.07, 0.19, 0.31] {
        let middle = run.middle(set, [0.3, 0.0, 0.0], t, &crate::step::Held::default(), false);
        let path = middle.path([0.0; 3], [2.0, 0.0, 0.0]);
        for s in [-h, 0.0, h] {
            let mut g = Mat::zeros(n, n);
            middle.generator(&path, s, &mut g);
            for _ in 0..4 {
                let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.normal(), stream.normal())).collect();
                let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
                let gpsi: Vec<C64> = (0..n).map(|i| (0..n).map(|j| g[(i, j)] * psi[j]).sum()).collect();
                let gdag: Vec<C64> = (0..n).map(|i| (0..n).map(|j| g[(j, i)].conj() * psi[j]).sum()).collect();
                let hazard = (0..n).map(|i| (psi[i].conj() * C64::I * (gpsi[i] - gdag[i])).re).sum::<f64>() / norm;
                let mut rate = f64::NAN;
                let _ = middle.emitted(&path, s, &psi, &mut |rates: &[f64]| {
                    rate = rates.iter().sum();
                    None
                });
                worst = worst.max((rate - hazard).abs() / hazard.abs());
                beat = beat.max(hazard);
            }
        }
    }
    println!("the photons' rate against the generator's hazard: within {worst:.2e} of it (hazards up to {beat:.3e})");
    assert!(worst < 1e-12, "the photons' rate against the hazard: {worst:e}");
    // The decay's beat is the integrator's candidate's (run.rs # The jumps' coherence; spectral.rs # Which beats are tones):
    // the generator less every candidate at its phase -- the static group the integrator keeps where all are tones --
    // decays alike at every time across the beat, to the rounding, where the generator's own decay swings.
    let (mut kept, mut swung, mut size) = (0.0_f64, 0.0_f64, 0.0_f64);
    for s in [-h, 0.0, h] {
        let decays: Vec<[Mat; 2]> = [0.0, 0.07, 0.19]
            .iter()
            .map(|&t| {
                let middle = run.middle(set, [0.3, 0.0, 0.0], t, &crate::step::Held::default(), false);
                let path = middle.path([0.0; 3], [2.0, 0.0, 0.0]);
                let (mut g, mut plus, mut minus) = (Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n));
                middle.generator(&path, s, &mut g);
                let whole = g.clone();
                let (mut beats, mut keys) = (vec![], vec![]);
                middle.beats(&path, &mut beats, &mut keys);
                for (m, &nu) in beats.iter().enumerate() {
                    middle.envelopes(&path, m, s, &mut plus, &mut minus);
                    g.add_scaled(-C64::cis(nu * s), &plus);
                    g.add_scaled(-C64::cis(-nu * s), &minus);
                }
                [whole, g].map(|g| Mat::from_fn(n, n, |a, b| C64::I * (g[(a, b)] - g[(b, a)].conj())))
            })
            .collect();
        for d in &decays[1..] {
            kept = kept.max(d[1].data.iter().zip(&decays[0][1].data).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max));
            swung = swung.max(d[0].data.iter().zip(&decays[0][0].data).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max));
        }
        size = size.max(decays[0][0].data.iter().map(|z| z.abs()).fold(0.0, f64::max));
    }
    println!("less its candidates the generator's decay moves across the beat by {kept:.2e}, its own by {swung:.3e} (its entries up to {size:.3e})");
    assert!(kept <= crate::units::rounding(4 * n + 10) * size && swung > 1e3 * kept, "the decay less the candidates moves by {kept:e}, the generator's by {swung:e}");
}

/// The readout (DESIGN §1.3; proto.rs `Workspace::populations`, run.rs `Dynamics::readout`): at random points and states of
/// every set of PGC3D_1ms and R4_LIN_4 (the rule's choice at the focus; R4's F = 1 set holds explicit excited states), the
/// fold's populations sum to ψ†ψ to the rounding (the metric's terms, x = S⁻¹ψ), an explicit state's is its own |x_X|²
/// with what other offsets' blocks hold of it as their Q rows (at most the Q rows' whole), and the run's rows sum to one.
#[test]
fn the_readout_is_the_fold_s_populations_and_they_sum_to_the_norm() {
    use crate::step::Dynamics;
    use super::proto::Cache;
    let pgc = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("PGC3D_1ms.json").to_string_lossy().into_owned();
    let mut seed = 7u64;
    let mut u = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut explicit_checked = 0;
    for name in [pgc, "R4_LIN_4".to_string()] {
        let req = crate::model::testing::scene(&name);
        let choice = focus_choice(&req);
        let mut p = [0.0; GROUND];
        p[0] = 1.0;
        let run = super::run::Onefold::of(&req, &choice, p).unwrap();
        for set in Fold::sets(&req, &choice).unwrap() {
            let fold = Fold::new(&req, &choice, set[0]).unwrap();
            if fold.blocks.is_empty() {
                continue;
            }
            let mut ws = fold.prototype_workspace().unwrap();
            let n = fold.size();
            for point in 0..3 {
                let r: [f64; 3] = std::array::from_fn(|_| 0.6 * u() - 0.3);
                let kappa: [f64; 3] = std::array::from_fn(|_| (0.2 * u() - 0.1) / fold.hbar_over_mass);
                let psi: Vec<C64> = (0..n).map(|_| C64::new(u() - 0.5, u() - 0.5)).collect();
                let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
                let mut cache = Cache::default();
                fold.prototype(r, kappa, &mut cache, &mut ws).unwrap();
                let mut out = [0.0; STATES];
                let sum = ws.populations(&fold, &cache, &psi, &mut out);
                assert!((sum - norm).abs() <= 1e-12 * norm && (out.iter().sum::<f64>() - sum).abs() <= 1e-13 * norm, "{name} set {set:?} point {point}: the populations' sum {sum} against ψ†ψ {norm}");
                let x = ws.normal().product(&Mat { rows: n, cols: 1, data: psi.clone() });
                let rows: f64 = sum - x.data.iter().map(|z| z.norm_sqr()).sum::<f64>();
                for (j, &e) in fold.explicit.iter().enumerate() {
                    let own = x.data[fold.ground.len() + j].norm_sqr();
                    let more = out[GROUND + e] - own;
                    assert!(more >= -1e-13 * norm && more <= rows + 1e-13 * norm, "{name} set {set:?} point {point}: explicit state {e}'s population {} against |x|² {own} (the Q rows' {rows})", out[GROUND + e]);
                    explicit_checked += 1;
                }
                let at = crate::model::Point { r, kappa, t: 0.0 };
                let rows = run.readout(run.cluster_of(set[0]), &at, &psi, &crate::step::Held::default());
                assert!((rows.iter().sum::<f64>() - 1.0).abs() <= 1e-13, "{name} set {set:?} point {point}: the rows' sum {}", rows.iter().sum::<f64>());
            }
        }
    }
    assert!(explicit_checked > 0, "an explicit state checked");
}

/// The readout of a two-level atom at rest in a running wave (BRIEF_STEP3 item 2): the excited state folded, the readout's
/// excited population of the ground state is the slow mode's of the 2 × 2 generator the explicit choice holds there,
/// |T|²/(1 + |T|²) with (1, T) its eigenvector of the eigenvalue nearest the ground state's, within the share the fold's
/// readings hold it to -- the fold's conditional state, as the run carries it between jumps; beside it (said, not held)
/// the optical Bloch equations' steady state ρ_ee, an ensemble's, with the transient after each jump that the fold's
/// landing leaves out.
#[test]
fn a_two_level_atom_s_readout_is_its_slow_mode_s_population() {
    use crate::step::{Dynamics, Middle};
    let (gamma, ee) = (38.0, 1000.0);
    for (a, delta) in [(2.0, -60.0), (6.0, -60.0), (12.0, -60.0), (6.0, -20.0)] {
        let rays = [Ray { axis: 0, nu: ee + delta, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(a))] }];
        let req = system(&[(0, 0.0)], &[(17, ee, gamma, 0)], &rays, 1, true);
        let mut p = [0.0; GROUND];
        p[0] = 1.0;
        let level = Fold::new(&req, &Choice::tiered(&req, vec![]).unwrap(), 0).unwrap().excited_level[17];
        let kept = Choice::tiered(&req, vec![(level, crate::model::Component { frequency: 0, counter: false }, 0)]).unwrap();
        let (explicit, folded) = (super::run::Onefold::of(&req, &kept, p).unwrap(), super::run::Onefold::of(&req, &Choice::tiered(&req, vec![]).unwrap(), p).unwrap());
        let at = crate::model::Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 };
        let set = explicit.cluster_of(0);
        let states = &explicit.cluster(set).states;
        let i = states.iter().position(|&x| x == GROUND + 17).expect("the excited state explicit");
        let middle = explicit.middle(set, at.r, at.t, &crate::step::Held::default(), false);
        let mut whole = Mat::zeros(states.len(), states.len());
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut whole);
        // The level's other states are not driven: the 2 × 2 of the ground state and the driven one.
        let g = Mat::from_fn(2, 2, |r, c| whole[([0, i][r], [0, i][c])]);
        // The slow mode: the eigenvalue of [[g00, g01], [g10, g11]] nearest g00, its eigenvector (1, (λ − g00)/g01).
        let (tr, det) = (g[(0, 0)] + g[(1, 1)], g[(0, 0)] * g[(1, 1)] - g[(0, 1)] * g[(1, 0)]);
        let root = (tr * tr * 0.25 - det).sqrt();
        let lambda = [tr * 0.5 + root, tr * 0.5 - root].into_iter().min_by(|x, y| (*x - g[(0, 0)]).abs().total_cmp(&(*y - g[(0, 0)]).abs())).unwrap();
        let t = (lambda - g[(0, 0)]) / g[(0, 1)];
        let slow = t.norm_sqr() / (1.0 + t.norm_sqr());
        // The optical Bloch equations' steady state of the same generator: Ω/2 = |g10|, Δ, Γ.
        let (half, detuning, decay) = (g[(1, 0)].abs(), g[(1, 1)].re - g[(0, 0)].re, -2.0 * g[(1, 1)].im);
        let bloch = half * half / (detuning * detuning + 0.25 * decay * decay + 2.0 * half * half);
        let rows = folded.readout(folded.cluster_of(0), &at, &[C64::ONE], &crate::step::Held::default());
        println!("a {a}, Δ {delta}: Ω/2 {half:.3}, Δ {detuning:.2}, Γ {decay:.2} rad/µs: the readout's excited population {:.6e}, the slow mode's {slow:.6e} ({:+.2e} of it), the Bloch equations' ρ_ee {bloch:.6e} ({:+.2e})", rows[GROUND + 17], rows[GROUND + 17] / slow - 1.0, rows[GROUND + 17] / bloch - 1.0);
        assert!((rows.iter().sum::<f64>() - 1.0).abs() <= 1e-13, "the rows' sum");
        assert!((rows[GROUND + 17] - slow).abs() <= crate::model::elim::share() * slow, "the readout {} against the slow mode's {slow}", rows[GROUND + 17]);
    }
}


/// A two-level atom's request at rest (heavy): its excited state at 1000 rad/µs, its laser at the detuning `delta` with the
/// saturation s = 2Ω²/Γ² (the coupling Ω/2 = (Γ/2)√(s/2)); its idle excited states decaying (`idle_decay`).
fn two_level(delta: f64, s: f64) -> Request {
    let (ee, g) = (1000.0, 0.5 * GAMMA * (s / 2.0_f64).sqrt());
    let mut req = system(&[(0, 0.0)], &[(17, ee, GAMMA, 0)], &[Ray { axis: 0, nu: ee + delta, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(g))] }], 1, true);
    idle_decay(&mut req, GAMMA);
    req.validate().unwrap();
    req
}

/// The idle excited states of a synthetic request made to decay at Γ, as every request the app makes has them (an
/// explicit level's optical sector relaxes in every direction, step.rs # The optical sector's noise), each by a channel of
/// its own: its emission's column over the channels -- a ground state and a polarization -- orthogonal to every other
/// excited state's, so that the decay Σ_q C_q†C_q holds Γ on each idle state and nothing between any two (BRIEF_STEP3E
/// item 2: through one shared channel the idle states made one mode decaying at their sum and the rest none -- modes no
/// relaxation reaches, the master equation's steady state not one). Gram–Schmidt over the channels against the live
/// states' columns: a channel no other state uses first (the latest ground state's, the last polarization), then of the
/// others the one the columns leave the most of.
fn idle_decay(req: &mut Request, gamma: f64) {
    let channels: Vec<(usize, usize)> = (0..GROUND).rev().flat_map(|b| (0..3).map(move |q| (b, q))).collect();
    let column = |req: &Request, e: usize| -> Vec<C64> { channels.iter().map(|&(b, q)| req.emission[q][(b, GROUND + e)]).collect() };
    let (live, idle): (Vec<usize>, Vec<usize>) = (0..EXCITED).partition(|&e| column(req, e).iter().any(|z| *z != C64::ZERO));
    // A vector less its parts along an orthonormal basis, and its norm.
    let less = |mut v: Vec<C64>, basis: &[Vec<C64>]| -> (Vec<C64>, f64) {
        for u in basis {
            let p: C64 = u.iter().zip(&v).map(|(a, b)| a.conj() * *b).sum();
            v.iter_mut().zip(u).for_each(|(x, y)| *x -= p * *y);
        }
        let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        (v, norm)
    };
    let mut basis: Vec<Vec<C64>> = vec![];
    for e in live {
        let (v, norm) = less(column(req, e), &basis);
        if norm > 0.0 {
            basis.push(v.iter().map(|z| *z / norm).collect());
        }
    }
    for e in idle {
        let unit = |k: usize| -> Vec<C64> { (0..channels.len()).map(|j| if j == k { C64::ONE } else { C64::ZERO }).collect() };
        let (v, norm) = (0..channels.len()).map(|k| less(unit(k), &basis)).max_by(|a, b| a.1.total_cmp(&b.1)).expect("a channel");
        assert!(norm > 0.0, "a channel left for every excited state");
        let v: Vec<C64> = v.iter().map(|z| *z / norm).collect();
        for (&(b, q), z) in channels.iter().zip(&v) {
            req.emission[q][(b, GROUND + e)] = *z * gamma.sqrt();
        }
        basis.push(v);
    }
}

/// A run's two-level clusters (BRIEF_STEP3C item 1): the ground set's with nothing explicit and with its excited node risen.
fn two_level_clusters(run: &run::Onefold, req: &Request) -> (usize, usize) {
    let fold = Fold::new(req, &Choice::base(req, vec![]).unwrap(), 0).unwrap();
    let key = Choice::candidates(&fold)[0].0;
    (0, run.clusters.cluster_with(0, vec![key]).expect("the explicit cluster"))
}

/// A jump's landing is a fall with no explicit amplitude (BRIEF_STEP3C item 1; DESIGN §11.1, run.rs # The nodes risen and
/// fallen: one formula, `Workspace::taken`; proto.rs # The landing): a two-level atom with the light on, the state (f, 0) of
/// its explicit cluster -- nothing in Q there, S = 1 -- taken onto its folded cluster at the same middle is S N⁻¹f, the
/// landing of the bare state f there, to the rounding, and what it books is its ring (proto.rs `Workspace::ring`). A state
/// that follows T -- the folded state ψ risen, (x, T_X x) -- falls back to ψ and books nothing, to the rounding.
#[test]
fn a_two_level_atom_s_landing_is_its_fall_with_no_excited_amplitude() {
    use crate::step::{Dynamics, Middle as _};
    for (delta, s) in [(-GAMMA, 0.1), (-GAMMA, 1.0), (-0.5 * GAMMA, 10.0)] {
        let req = two_level(delta, s);
        let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
        let (folded, explicit) = two_level_clusters(&run, &req);
        let (r, t) = ([0.1, 0.0, 0.0], 0.3);
        let (low, high) = (run.middle(folded, r, t, &crate::step::Held::default(), false), run.middle(explicit, r, t, &crate::step::Held::default(), false));
        let (path, path2) = (low.path([0.0; 3], [0.0; 3]), high.path([0.0; 3], [0.0; 3]));
        let n = run.cluster(explicit).states.len();
        low.generator(&path, 0.0, &mut Mat::zeros(1, 1));
        high.generator(&path2, 0.0, &mut Mat::zeros(n, n));
        let f = C64::new(0.6, 0.8);
        let crate::step::Landing::On(landed) = low.landing(&path, &[f]) else { panic!("the landing exists") };
        let bare: Vec<C64> = (0..n).map(|i| if i == 0 { f } else { C64::ZERO }).collect();
        let (fallen, dropped, _) = high.take_onto(&bare, folded, &low).expect("the fall");
        let apart = (landed[0] - fallen[0]).abs();
        println!("Δ {delta:+.1} s {s}: the landing {:?} and the fall at x_X = 0 {:?} apart {apart:.1e}; its norm's change {:.6}, its ring {dropped:.3e}", landed[0], fallen[0], 1.0 - landed[0].norm_sqr());
        assert!(apart <= 1e-14, "the fall at no explicit amplitude is the landing");
        assert!(dropped > 0.0, "it books its ring");
        let (risen, up, _) = low.take_onto(&[f], explicit, &high).expect("the rise");
        let (back, down, _) = high.take_onto(&risen, folded, &low).expect("the fall back");
        println!("   the folded state risen ({:.4?}) and fallen back: {:.1e} from it, dropped {up:.1e} and {down:.1e}", risen, (back[0] - f).abs());
        assert!((back[0] - f).abs() <= 1e-14 && up.abs() <= 1e-14 && down.abs() <= 1e-14, "a state that follows T falls back to itself");
    }
}

/// A decaying matrix M's flow by its eigenvectors (every eigenvalue below the real axis): of a vector v its coefficients
/// c = V⁻¹v, e^{−iMt}v = Σ_k c_k v_k e^{−iλ_k t}, and ∫₀^∞ e^{−iMt}vv†e^{iM†t}dt = Σ_kl c_k c̄_l v_k v_l† (−i)/(λ_k − λ̄_l).
struct Flow {
    e: crate::linalg::Eigen,
}

impl Flow {
    fn new(m: &Mat) -> Option<Flow> {
        crate::linalg::eigen(m).map(|e| Flow { e })
    }

    fn at(&self, v: &[C64], t: f64) -> Vec<C64> {
        let c = self.e.inverse.apply(v);
        let size = v.len();
        (0..size).map(|i| (0..size).map(|k| self.e.vectors[(i, k)] * c[k] * (C64::new(0.0, -t) * self.e.values[k]).exp()).sum()).collect()
    }

    fn integral(&self, v: &[C64]) -> Mat {
        let c = self.e.inverse.apply(v);
        let size = v.len();
        let mut w = Mat::zeros(size, size);
        for k in 0..size {
            for l in 0..size {
                w[(k, l)] = c[k] * c[l].conj() * C64::new(0.0, -1.0) / (self.e.values[k] - self.e.values[l].conj());
            }
        }
        let vw = self.e.vectors.product(&w);
        vw.product(&self.e.vectors.adjoint())
    }
}

/// Random systems land within their ring (proto.rs # The ring; landing.status 6, 9): one laser frequency on 2–3 ground and
/// 2–3 excited states (detunings within ±2Γ, couplings 0.05–0.6 Γ at random phases, each excited state decaying at Γ to a
/// ground state of its own choosing), the fold at the focus at rest. Per landing state the exact statistics -- the dense
/// lattice's flow from the bare state (`lattice`, `Flow`): the mean waiting time ∫‖ψ‖²dt and the next landing's
/// distribution, each offset's jumps apart -- against the fold's: its spectral landing ψ₀ = S N⁻¹f on the jump clock
/// (c = ‖ψ₀‖² below one: the bare state's photon at once with 1 − c; above: the dead time to ‖e^{−iGt}ψ₀‖² = 1), its own
/// flow e^{−iGt}, its jumps L_q (`Fold::prototype_jumps`). Where its readings are within the share (ρ, the run's test: the
/// ring's and the series' among them), so is every landing's waiting time's error and the total variation of its next
/// landing; what they err by over the ring's and the series' readings together is said (a leading-order estimate, not a
/// bound: no ratio is asserted).
#[test]
fn random_systems_land_within_their_ring() {
    use super::proto::{At, Cache};
    let share = crate::model::elim::share();
    let mut state = 20261116u64;
    let mut u = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let (grounds, exciteds, ee) = ([0usize, 4, 6], [17usize, 13, 9], 1000.0);
    let (mut within, mut compared, mut lines) = (0, 0, vec![]);
    for trial in 0..40 {
        let (g, e) = (2 + (u() * 2.0) as usize, 2 + (u() * 2.0) as usize);
        let scale = 0.05 + 0.55 * u();
        let ground: Vec<(usize, f64)> = (0..g).map(|i| (grounds[i], 0.3 * GAMMA * (2.0 * u() - 1.0))).collect();
        let excited: Vec<(usize, f64, f64, usize)> = (0..e).map(|i| (exciteds[i], ee + 2.0 * GAMMA * (2.0 * u() - 1.0), GAMMA, grounds[(u() * g as f64) as usize])).collect();
        let mut couplings = vec![];
        for x in 0..e {
            for y in 0..g {
                couplings.push((exciteds[x], grounds[y], C64::cis(6.283 * u()) * (scale * GAMMA * (0.5 + u()))));
            }
        }
        let req = system(&ground, &excited, &[Ray { axis: 0, nu: ee, direction: [0.0, 0.0, 1.0], couplings }], 1, true);
        let choice = Choice::base(&req, vec![]).unwrap();
        for set in Fold::sets(&req, &choice).unwrap() {
            let fold = Fold::new(&req, &choice, set[0]).unwrap();
            if fold.blocks.is_empty() || !fold.explicit.is_empty() {
                continue;
            }
            let Ok(mut ws) = fold.prototype_workspace() else { continue };
            let mut cache = Cache::default();
            let Some(step) = fold.prototype([0.0; 3], [0.0; 3], &mut cache, &mut ws) else { continue };
            let (readings, generator) = (step.readings(), step.generator.clone());
            let n = fold.size();
            let jumps = fold.prototype_jumps(&cache, &ws, At::Middle);
            let (near, far) = ws.layout();
            let blocks: Vec<usize> = near.iter().map(|x| x.0).chain(far.iter().map(|x| x.0)).collect();
            // The fold's jumps by offset: a group of one offset's blocks coherent, the offsets apart.
            let mut offsets: Vec<f64> = blocks.iter().map(|&o| fold.blocks[o].value).collect();
            offsets.sort_by(f64::total_cmp);
            offsets.dedup();
            let landed = |psi: &[C64], out: &mut [f64; GROUND], weight: f64| {
                for &offset in &offsets {
                    for q in 0..3 {
                        let mut u = [C64::ZERO; GROUND];
                        for (k, &o) in blocks.iter().enumerate().filter(|(_, o)| fold.blocks[**o].value == offset) {
                            let _ = o;
                            for (b, z) in u.iter_mut().enumerate() {
                                *z += (0..n).map(|p| jumps[k][q][(b, p)] * psi[p]).sum::<C64>();
                            }
                        }
                        for b in 0..GROUND {
                            out[b] += weight * u[b].norm_sqr();
                        }
                    }
                }
            };
            let Some(fold_flow) = Flow::new(&generator) else { continue };
            // The exact lattice, its nodes connected to the set's ground states (the lattice seeds every ground state of the
            // model; the others' nodes are no part of the flow): its flow, its excited nodes by offset.
            let whole = lattice(&fold, [0.0; 3], 4, &|_| true);
            let mut kept: Vec<usize> = whole.nodes.iter().enumerate().filter(|(_, (s, key))| fold.ground.contains(s) && key.iter().all(|&x| x == 0)).map(|x| x.0).collect();
            let mut at = 0;
            while at < kept.len() {
                let i = kept[at];
                for j in 0..whole.m.rows {
                    if (whole.m[(j, i)] != C64::ZERO || whole.m[(i, j)] != C64::ZERO) && !kept.contains(&j) {
                        kept.push(j);
                    }
                }
                at += 1;
            }
            let l = Lattice { nodes: kept.iter().map(|&i| whole.nodes[i].clone()).collect(), m: Mat::from_fn(kept.len(), kept.len(), |a, b| whole.m[(kept[a], kept[b])]) };
            let Some(exact_flow) = Flow::new(&l.m) else { continue };
            let size = l.m.rows;
            let mut keys: Vec<Vec<i64>> = l.nodes.iter().filter(|(s, _)| *s >= GROUND).map(|x| x.1.clone()).collect();
            keys.sort();
            keys.dedup();
            let (mut worst_tau, mut worst_tv) = (0.0_f64, 0.0_f64);
            for k in 0..fold.ground.len() {
                // Exact.
                let at = l.nodes.iter().position(|(s, key)| *s == fold.ground[k] && key.iter().all(|&x| x == 0)).expect("the ground state's node");
                let f: Vec<C64> = (0..size).map(|i| if i == at { C64::ONE } else { C64::ZERO }).collect();
                let rho = exact_flow.integral(&f);
                let tau_exact: f64 = (0..size).map(|i| rho[(i, i)].re).sum();
                let mut next_exact = [0.0; GROUND];
                for key in &keys {
                    let nodes: Vec<usize> = (0..size).filter(|&i| l.nodes[i].0 >= GROUND && l.nodes[i].1 == *key).collect();
                    for q in 0..3 {
                        for (b, out) in next_exact.iter_mut().enumerate() {
                            let w: Vec<C64> = nodes.iter().map(|&i| req.emission[q][(b, l.nodes[i].0)]).collect();
                            let mut x = C64::ZERO;
                            for (a, &i) in nodes.iter().enumerate() {
                                for (c, &j) in nodes.iter().enumerate() {
                                    x += w[a].conj() * rho[(i, j)] * w[c];
                                }
                            }
                            *out += x.re;
                        }
                    }
                }
                // The fold on its jump clock.
                let bare: Vec<C64> = (0..n).map(|p| if p == k { C64::ONE } else { C64::ZERO }).collect();
                let Some((psi0, _)) = ws.taken(&fold, &cache, &bare, &[], false) else { continue };
                let c = crate::linalg::norm_sqr(&psi0);
                let (mut tau, mut next) = (0.0, [0.0; GROUND]);
                let start = if c <= 1.0 {
                    let mut prompt = [0.0; GROUND];
                    landed(&bare, &mut prompt, 1.0);
                    let total: f64 = prompt.iter().sum();
                    for (o, x) in next.iter_mut().zip(prompt) {
                        *o += (1.0 - c) * x / total;
                    }
                    psi0
                } else {
                    let norm = |t: f64| crate::linalg::norm_sqr(&fold_flow.at(&psi0, t)) - 1.0;
                    let mut hi = 1.0 / GAMMA;
                    while norm(hi) > 0.0 {
                        hi *= 2.0;
                    }
                    let mut lo = 0.0;
                    for _ in 0..80 {
                        let mid = 0.5 * (lo + hi);
                        if norm(mid) > 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    tau += lo;
                    fold_flow.at(&psi0, lo)
                };
                let rho = fold_flow.integral(&start);
                tau += (0..n).map(|p| rho[(p, p)].re).sum::<f64>();
                for &offset in &offsets {
                    for q in 0..3 {
                        let lq = Mat::from_fn(GROUND, n, |b, p| blocks.iter().enumerate().filter(|(_, o)| fold.blocks[**o].value == offset).map(|(x, _)| jumps[x][q][(b, p)]).sum());
                        let m = lq.product(&rho).product(&lq.adjoint());
                        for (b, o) in next.iter_mut().enumerate() {
                            *o += m[(b, b)].re;
                        }
                    }
                }
                let total_exact: f64 = next_exact.iter().sum();
                let total: f64 = next.iter().sum();
                worst_tau = worst_tau.max((tau / tau_exact - 1.0).abs());
                worst_tv = worst_tv.max(0.5 * (0..GROUND).map(|b| (next[b] / total - next_exact[b] / total_exact).abs()).sum::<f64>());
            }
            let read = readings.ring + readings.series;
            compared += 1;
            lines.push(format!("   trial {trial} (ground {g}, excited {e}, couplings ×{scale:.2}Γ): the ring {:.2e}, the series {:.2e}; the worst landing's waiting time {worst_tau:.2e}, next landing's total variation {worst_tv:.2e}; actual over read {:.2}", readings.ring, readings.series, worst_tau.max(worst_tv) / read));
            if readings.most() <= share {
                within += 1;
                assert!(worst_tau <= share && worst_tv <= share, "trial {trial}: within its readings ({:.2e}) the landings err by {worst_tau:.2e} (waiting) and {worst_tv:.2e} (next landing)", readings.most());
            }
        }
    }
    for l in &lines {
        println!("{l}");
    }
    println!("{compared} folds compared, {within} within their readings");
    assert!(within >= 10, "{within} within");
}

/// The loss energy where one light set's fold ceases to exist (BRIEF_LANDIMPL item 4; run.rs # The rest of the model,
/// `Clusters::difference`): a two-level atom in a far trap light (Δ_t = −60Γ, Ω_t = 6Γ) and a cooling light 0.1Γ below
/// its line, at rest at the focus, the cooling's saturation s = 0.8 and 0.9 -- with every light the fold exists at both
/// (the trap moves the line), with the cooling alone it ceases between them. There its node rises in both light sets:
/// V_run at s = 0.9 is the trap's own shift, the run's of the trap light alone, within the share of it; risen in the
/// cooling's set alone (before), V_run dropped by that set's whole cooling shift there (−6.2 rad/µs against V_run's
/// −4.3); now by the trap's change of the cooling's shift through the risen node alone (said).
#[test]
fn the_loss_energy_stands_on_one_set_of_risen_nodes_for_both_light_sets() {
    use crate::model::Point;
    use crate::step::Dynamics;
    let ee = 1000.0;
    let rays = |s: f64, cooling: bool| -> Vec<Ray> {
        let trap = Ray { axis: 0, nu: ee - 60.0 * GAMMA, direction: [1.0, 0.0, 0.0], couplings: vec![(17, 0, C64::real(3.0 * GAMMA))] };
        let cool = Ray { axis: 1, nu: ee - 0.1 * GAMMA, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(0.5 * GAMMA * (s / 2.0_f64).sqrt()))] };
        if cooling { vec![trap, cool] } else { vec![trap] }
    };
    let energy = |s: f64, cooling: bool| -> (f64, bool) {
        let mut req = system(&[(0, 0.0)], &[(17, ee, GAMMA, 0)], &rays(s, cooling), if cooling { 2 } else { 1 }, true);
        idle_decay(&mut req, GAMMA);
        req.validate().unwrap();
        let mut p = [0.0; GROUND];
        p[0] = 1.0;
        let run = super::run::Onefold::new(&req, p).expect("the run");
        let light = Light::new(&req);
        let off = super::run::lit(&req, &light, |w| w != light.trap);
        let fold = Fold::new(&off, &Choice::base(&off, vec![]).unwrap(), 0).unwrap();
        let exists = fold.blocks.is_empty() || fold.prototype([0.0; 3], [0.0; 3], &mut super::proto::Cache::default(), &mut fold.prototype_workspace().unwrap()).is_some();
        let set = run.cluster_of(0);
        let psi: Vec<C64> = run.cluster(set).states.iter().map(|&x| if x == 0 { C64::ONE } else { C64::ZERO }).collect();
        (run.energy(set, &Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, &psi), exists)
    };
    let (trap, _) = energy(0.0, false);
    let ((below, held), (above, gone)) = (energy(0.8, true), energy(0.9, true));
    println!("V_run at s = 0.8 {below:.4} rad/µs (the cooling's fold {}), at s = 0.9 {above:.4} ({}); the trap light's alone {trap:.4}: the step across {:.4}", if held { "exists" } else { "does not" }, if gone { "exists" } else { "does not" }, above - below);
    assert!(held && !gone, "the cooling's fold ceases between s = 0.8 and 0.9");
    assert!((above - trap).abs() <= crate::model::elim::share() * trap.abs(), "risen in both, V_run is the trap's own shift");
}

/// A risen node's cluster is a fold the prototype holds wherever the base choice makes it (BRIEF_LANDIMPL item 3; fold.rs
/// `Choice::base`, `Block::far`): on sixty random systems, every candidate node of every set risen alone on the base
/// choice of its set (`Choice::base_of`), no block is left far -- its rows all centred -- that holds an explicit state or a
/// copy row, what a far form does not hold (GMD1C's (1, f2, 0) was refused so, every such block's rows centred around
/// the risen node's) -- and none refused for a copy at a P place's frame whose static coupling to it folds (BRIEF_COPIES
/// item 4; 5 of 235 were).
#[test]
fn a_risen_node_s_cluster_holds_no_far_block_a_far_form_does_not_hold() {
    let mut state = 20261117u64;
    let (mut made, mut refused) = (0, std::collections::BTreeMap::<String, usize>::new());
    for trial in 0..60 {
        let (req, _) = random_system(&mut state, trial);
        let Ok(base) = Choice::base(&req, vec![]) else { continue };
        for set in Fold::sets(&req, &base).unwrap() {
            let Ok(fold) = Fold::new(&req, &base, set[0]) else { continue };
            for (key, _, _) in Choice::candidates(&fold) {
                let Ok(choice) = Choice::base_of(&req, vec![key], &|x| x == &set[..]) else { continue };
                let Ok(risen) = Fold::new(&req, &choice, set[0]) else { continue };
                for b in &risen.blocks {
                    let centred = b.solved == 0 && !b.rows.is_empty();
                    assert!(!centred || b.far(), "trial {trial}, node {key:?}: an all-centred block with an explicit state or a copy");
                }
                match risen.prototype_workspace() {
                    Ok(_) => made += 1,
                    Err(why) => *refused.entry(why.split('：').nth(1).unwrap_or(&why).to_string()).or_default() += 1,
                }
            }
        }
    }
    println!("{made} risen clusters held by the prototype; refused: {refused:?}");
    assert!(refused.is_empty(), "every risen cluster held: no all-centred block with an explicit state or a copy, no copy at a P frame refused where its static coupling folds (BRIEF_COPIES item 4: 5 of 235 were)");
}

/// A two-level atom's ring reads what its landing leaves (proto.rs # The ring; landing.status 6): the ring's reading at the
/// focus, resonant at s = 0.1–0.4, 6.8·10⁻³, 3.0·10⁻², 7.9·10⁻², 0.174, and at Δ = −4Γ, s = 10, 30, 60, 4.6·10⁻³,
/// 2.45·10⁻², 5.5·10⁻² -- the leading order of the waiting time's error on the exact slow span -- each within a tenth.
#[test]
fn a_two_level_atom_s_ring_reads_its_landing() {
    for (delta, s, expected) in [(0.0, 0.1, 6.8e-3), (0.0, 0.2, 3.0e-2), (0.0, 0.3, 7.9e-2), (0.0, 0.4, 0.174), (-4.0 * GAMMA, 10.0, 4.6e-3), (-4.0 * GAMMA, 30.0, 2.45e-2), (-4.0 * GAMMA, 60.0, 5.5e-2)] {
        let req = two_level(delta, s);
        let fold = Fold::new(&req, &Choice::base(&req, vec![]).unwrap(), 0).unwrap();
        let mut ws = fold.prototype_workspace().unwrap();
        let mut cache = super::proto::Cache::default();
        let readings = fold.prototype([0.0; 3], [0.0; 3], &mut cache, &mut ws).expect("the fold at the focus").readings();
        println!("Δ {delta:+.1} s {s}: the ring {:.3e} against {expected:.2e} (the series' {:.1e})", readings.ring, readings.series);
        assert!((readings.ring - expected).abs() <= 0.1 * expected, "the ring reads the landing's error");
    }
}

/// A jump's landing on the fold carries the slow mode's survival back to the jump (proto.rs # The landing; landing.status
/// 3): a resonant two-level atom at rest, ‖S N⁻¹f‖² = (1 + q)/(2q²), q = √(1 − 2s) -- 1.18402, 1.47883, 2.04057, 3.61803
/// at s = 0.1–0.4 --, within twice the share of c − 1 (the series' rest on T and on T_L); off resonance (Δ = −4Γ, s = 2)
/// 0.98611. The orthogonal projection's (1 + q)/2 said beside.
#[test]
fn a_landing_is_the_slow_mode_s_survival_extrapolated_to_the_jump() {
    use crate::step::{Dynamics, Middle as _};
    let share = crate::model::elim::share();
    for (delta, s, expected) in [(0.0, 0.1, 1.18402), (0.0, 0.2, 1.47883), (0.0, 0.3, 2.04057), (0.0, 0.4, 3.61803), (-4.0 * GAMMA, 2.0, 0.98611)] {
        let req = two_level(delta, s);
        let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
        let low = run.middle(0, [0.0; 3], 0.0, &crate::step::Held::default(), false);
        let path = low.path([0.0; 3], [0.0; 3]);
        low.generator(&path, 0.0, &mut Mat::zeros(1, 1));
        let crate::step::Landing::On(landed) = low.landing(&path, &[C64::ONE]) else { panic!("the landing exists") };
        let c = landed[0].norm_sqr();
        let q = (1.0 - 2.0 * s).max(0.0).sqrt();
        println!("Δ {delta:+.1} s {s}: ‖S N⁻¹f‖² {c:.5} against {expected:.5} (the orthogonal (1 + q)/2 at resonance {:.5})", 0.5 * (1.0 + q));
        assert!((c - expected).abs() <= 2.0 * share * (expected - 1.0).abs(), "the landing's norm the slow mode's survival at the jump");
    }
}

/// A resonant two-level atom at rest swept over saturations s = 0.1–0.6 (BRIEF_STEP3C item 1, BRIEF_LANDIMPL; DESIGN §11,
/// M18, M15): its fold's series' ratio goes to one about s = 0.55 (proto.rs: none at and beyond it), and its ring reads
/// beyond the share from s ≈ 0.24 (proto.rs # The ring; 3.0·10⁻² at s = 0.2, 7.9·10⁻² at 0.3): at s = 0.1 and 0.2 the
/// fold stays (no step's end explicit), from s = 0.3 the node rises; at every s every step's middle exists (no step
/// refused), and the run's photons per µs are the master equation's Γρ_ee = (Γ/2)s/(1 + s) within four of their standard
/// errors and the share -- continuous in s, whichever side a step stands on.
#[test]
fn a_resonant_two_level_atom_rises_before_its_fold_ceases_to_exist() {
    use crate::step::{Dynamics, Midway, Trajectory};
    struct Anywhere;
    impl crate::step::Domain for Anywhere {
        fn runaway(&self, _: f64) -> bool {
            false
        }
    }
    let share = crate::model::elim::share();
    let (tau, steps, count) = (0.1, 100, 12);
    for s in [0.1, 0.2, 0.3, 0.5, 0.55, 0.6] {
        let req = two_level(0.0, s);
        let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
        let mut scratch = crate::step::Scratch::default();
        let (mut jumps, mut explicit, mut dropped) = (0u64, 0usize, 0.0);
        for i in 0..count {
            let mut t = Trajectory::new(&run, 0, [0.0; 3], [0.0; 3], 5, i);
            for _ in 0..steps {
                t = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &crate::step::Brownian::STILL, &[], &mut scratch, &Anywhere)).expect("every step's middle exists").end;
                let c = run.cluster(t.cluster);
                explicit += usize::from(c.states.len() > c.ground);
            }
            jumps += t.scattered[0] as u64;
            dropped += t.dropped;
        }
        let time = (count as usize * steps) as f64 * tau;
        let (rate, error, master) = (jumps as f64 / time, (jumps as f64).sqrt() / time, 0.5 * GAMMA * s / (1.0 + s));
        println!("s {s:.2}: {rate:.4} ± {error:.4} photons/µs against the master equation's {master:.4}; {explicit} of {} steps' ends explicit; rings booked {dropped:.2e}", count as usize * steps);
        assert!((rate - master).abs() <= 4.0 * error + share * master, "s {s}: the photons' rate {rate} ± {error} against {master}");
        assert!(if s < 0.25 { explicit == 0 } else { explicit > 0 }, "s {s}: the fold stays where its ring is within the share, the node rises where not");
    }
}

/// A jump's landing that does not exist rises and lands (BRIEF_COPIES item 1; step.rs `land`): a laser half a Γ below an
/// excited state of ground state 5 (s = 0.2) that decays to ground state 0, which its own laser holds at resonance at
/// s = 0.6 -- there the fold of 0's set with nothing explicit does not exist (its series' ratio at one) --: the photons of
/// 5's set land in 0's, where the landing was no number (and the jump's loop, the bare state's photon and its landing
/// again, went round on it). The landing rises the node that holds the most of the bare state (the rule of a
/// trajectory's start) and lands there: at a point, and inside the steps of a run of the atom, every state a number and
/// no step refused.
#[test]
fn a_landing_that_does_not_exist_rises_and_lands() {
    use crate::step::{Dynamics, Landing, Midway, Trajectory};
    struct Anywhere;
    impl crate::step::Domain for Anywhere {
        fn runaway(&self, _: f64) -> bool {
            false
        }
    }
    let ee = 1000.0;
    let g = |s: f64| C64::real(0.5 * GAMMA * (s / 2.0_f64).sqrt());
    let rays = [
        Ray { axis: 0, nu: ee - 0.5 * GAMMA, direction: [0.0, 0.0, 1.0], couplings: vec![(12, 5, g(0.2))] },
        Ray { axis: 1, nu: ee, direction: [1.0, 0.0, 0.0], couplings: vec![(17, 0, g(0.6))] },
    ];
    let mut req = system(&[(0, 0.0), (5, 0.0)], &[(12, ee, GAMMA, 0), (17, ee, GAMMA, 0)], &rays, 2, true);
    idle_decay(&mut req, GAMMA);
    req.validate().unwrap();
    let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
    let (folded, explicit) = two_level_clusters(&run, &req);
    assert_eq!(run.cluster_of(0), folded, "0's set its own");
    let at = crate::model::Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 };
    let held = crate::step::Held::default();
    let unit = |n: usize| -> Vec<C64> { (0..n).map(|i| if i == 0 { C64::ONE } else { C64::ZERO }).collect() };
    assert_eq!(run.landed(folded, &at, None, &unit(1), &held), Landing::Risen(explicit), "no landing on 0's fold with nothing explicit: its rise");
    let Landing::On(landed) = run.landed(explicit, &at, None, &unit(run.cluster(explicit).states.len()), &held) else { panic!("the risen cluster's landing") };
    println!("the landing on the risen cluster: {:.4?}", &landed[..2]);
    // A jump at the point from 5's set.
    let mut t = Trajectory::new(&run, 5, at.r, at.kappa, 3, 0);
    let from = t.cluster;
    crate::step::jump(&run, &mut t, None).expect("the jump");
    println!("a jump from cluster {from} at the point lands in {} ({} states), its norm {:.4}", t.cluster, run.cluster(t.cluster).states.len(), crate::linalg::norm_sqr(&t.psi));
    assert_eq!(t.cluster, explicit, "it lands on the risen cluster");
    assert!(t.psi.iter().all(|z| z.is_finite()) && crate::linalg::norm_sqr(&t.psi) > 0.0, "a number");
    // Inside the steps: a trajectory's photon from 5's set to 0's lands there, risen.
    let (tau, steps) = (0.1, 100);
    let mut scratch = crate::step::Scratch::default();
    let mut moved = 0;
    for i in 0..12 {
        let mut t = Trajectory::new(&run, 5, at.r, at.kappa, 7, i);
        for _ in 0..steps {
            t = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &crate::step::Brownian::STILL, &[], &mut scratch, &Anywhere)).expect("no step refused").end;
            assert!(t.psi.iter().all(|z| z.is_finite()), "every state a number");
        }
        moved += usize::from(run.cluster(t.cluster).states.contains(&0));
    }
    println!("{moved} of 12 trajectories went from 5's set to 0's inside their steps");
    assert!(moved > 0, "jumps land in 0's set inside the steps");
}

/// The fall's criterion with its balance (BRIEF_STEP3C item 1, BRIEF_LANDIMPL item 2; DESIGN §11.4, run.rs # The nodes risen
/// and fallen): a two-level atom's decisions (at Δ = −4Γ, s = 2: its fold within the share, its ring too) forced to flip --
/// the explicit side priced dearer from the explicit cluster, cheaper from the folded one (the integrator's price is the
/// decision's argument) --, the explicit state each time (1, 0): a fast part, its norm's change δp = 1 − ‖ψ′‖² the
/// landing's (+0.014), what the fall books its ring E (proto.rs `Workspace::ring`, ≈ 2.5·10⁻⁴). Where an interval's share
/// pays for a flip, (share − ρ)γΔt ≥ E, the node falls at every other decision; where it does not, the booked E stops the
/// falls; and over all of it what the falls booked stays within the share of the time, ΣE ≤ Σ share·γΔt + the largest E.
#[test]
fn a_flipping_node_s_booked_drops_stop_its_falls_where_the_share_does_not_pay_them() {
    use crate::step::{Dynamics, Middle as _, Offer};
    let share = crate::model::elim::share();
    let req = two_level(-4.0 * GAMMA, 2.0);
    let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
    let (folded, explicit) = two_level_clusters(&run, &req);
    let (r, t) = ([0.0; 3], 0.0);
    let fewer = |o: &Offer| 1e9 * o.states as f64;
    let more = |o: &Offer| 1e9 / o.states as f64;
    // The folded side's slow mode's decay rate γ; one fall of (1, 0) where the share pays for anything, its ring.
    let low = run.middle(folded, r, t, &crate::step::Held::default(), false);
    let path = low.path([0.0; 3], [0.0; 3]);
    let mut g = Mat::zeros(1, 1);
    low.generator(&path, 0.0, &mut g);
    let gamma = -2.0 * g[(0, 0)].im;
    let n = run.cluster(explicit).states.len();
    let bare: Vec<C64> = (0..n).map(|i| if i == 0 { C64::ONE } else { C64::ZERO }).collect();
    let probe = run.middle(explicit, r, t, &crate::step::Held::default(), true);
    let fell = probe.decide(&probe.path([0.0; 3], [0.0; 3]), &bare, 1e3 / gamma, 0.0, &fewer).switched.expect("a fall the share pays for");
    let (ring, norm) = (fell.dropped, 1.0 - crate::linalg::norm_sqr(&fell.psi));
    println!("the fall of (1, 0): its norm's change δp {norm:+.4e}, its ring {ring:.3e}");
    assert!(ring > 0.0 && ring < norm.abs(), "a fall books its ring, not its norm");
    let (mut owed, mut booked, mut paid, mut largest) = (0.0, 0.0, 0.0, 0.0_f64);
    for (phase, interval) in [(0, 20.0 * ring / (share * gamma)), (1, 0.5 * ring / (share * gamma))] {
        let mut falls = 0;
        for _ in 0..20 {
            let high = run.middle(explicit, r, t, &crate::step::Held::default(), true);
            let path = high.path([0.0; 3], [0.0; 3]);
            let fell = high.decide(&path, &bare, interval, owed, &fewer);
            owed = fell.owed;
            let Some(down) = fell.switched else { continue };
            owed = (owed + down.dropped).max(0.0);
            assert_eq!(down.cluster, folded, "the fall to the folded cluster");
            falls += 1;
            booked += down.dropped;
            largest = largest.max(down.dropped);
            let low = run.middle(folded, r, t, &crate::step::Held::default(), true);
            let path = low.path([0.0; 3], [0.0; 3]);
            let rose = low.decide(&path, &down.psi, interval, owed, &more);
            owed = (rose.owed + rose.switched.as_ref().map_or(0.0, |x| x.dropped)).max(0.0);
            paid += share * gamma * 2.0 * interval;
            assert_eq!(rose.switched.map(|x| x.cluster), Some(explicit), "the rise back");
        }
        println!("Δt {interval:.3e} µs (the share's pay {:.3e} an interval, the ring {ring:.3e}): {falls} falls of 20; booked {booked:.3e}, the share's {paid:.3e}", share * gamma * interval);
        assert!(if phase == 0 { falls == 20 } else { falls == 0 }, "falls where the share pays for them, none where not");
    }
    assert!(booked <= paid + largest, "the booked rings within the share of the time");
}

/// A fall right after a landing on the explicit side books its ring, not its norm (BRIEF_LANDIMPL item 2; landing.status 4,
/// 8): a resonant two-level atom at s = 0.2, its explicit state from (1, 0) after t_d = 0, 1/Γ, 4/Γ of no jump, taken onto
/// its fold: the norm's change δp = 1 − ‖ψ′‖² is large and negative (−0.48, −0.33, −0.10: the fast part's antibunching, a
/// dead time on the clock), while the clock's mean waiting time from there errs by +2.9 %, +3.0 %, +1.7 % against the
/// exact (the explicit generator's) on the exact slow span -- what the fall's ring reads (2.9·10⁻², 3.0·10⁻², 1.6·10⁻²),
/// within its 15 % and the share (its leading order; T's error within the share, the ring quadratic in it) --; the fold's
/// own slow part, c = ‖ψ′‖², is the exact one's within the series' share (2·share·|c − 1|, as the landing's).
#[test]
fn a_fall_right_after_a_landing_books_its_ring_not_its_norm() {
    use crate::step::{Dynamics, Middle as _};
    let req = two_level(0.0, 0.2);
    let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
    let (folded, explicit) = two_level_clusters(&run, &req);
    let (low, high) = (run.middle(folded, [0.0; 3], 0.0, &crate::step::Held::default(), false), run.middle(explicit, [0.0; 3], 0.0, &crate::step::Held::default(), false));
    let (path, path2) = (low.path([0.0; 3], [0.0; 3]), high.path([0.0; 3], [0.0; 3]));
    let n = run.cluster(explicit).states.len();
    let (mut g, mut gx) = (Mat::zeros(1, 1), Mat::zeros(n, n));
    low.generator(&path, 0.0, &mut g);
    high.generator(&path2, 0.0, &mut gx);
    let gamma = -2.0 * g[(0, 0)].im;
    // The explicit side's two coupled states (the ground state and the excited one the light drives) and its generator
    // there; by its eigenvectors e^{−iGt}ψ and the exact mean waiting time ∫‖e^{−iGt}ψ‖²dt.
    let coupled: Vec<usize> = (0..n).filter(|&k| k == 0 || gx[(k, 0)] != C64::ZERO || gx[(0, k)] != C64::ZERO).collect();
    assert_eq!(coupled.len(), 2, "a two-level atom");
    let two = Mat::from_fn(2, 2, |i, j| gx[(coupled[i], coupled[j])]);
    let e = crate::linalg::eigen(&two).expect("its eigenvectors");
    let det = e.vectors[(0, 0)] * e.vectors[(1, 1)] - e.vectors[(0, 1)] * e.vectors[(1, 0)];
    let on = |psi: [C64; 2]| [(e.vectors[(1, 1)] * psi[0] - e.vectors[(0, 1)] * psi[1]) / det, (e.vectors[(0, 0)] * psi[1] - e.vectors[(1, 0)] * psi[0]) / det];
    let evolved = |psi: [C64; 2], t: f64| -> [C64; 2] {
        let c = on(psi);
        std::array::from_fn(|i| (0..2).map(|k| e.vectors[(i, k)] * c[k] * (C64::new(0.0, -t) * e.values[k]).exp()).sum())
    };
    let waiting = |psi: [C64; 2]| -> f64 {
        let c = on(psi);
        let mut sum = C64::ZERO;
        for k in 0..2 {
            for l in 0..2 {
                let overlap: C64 = (0..2).map(|i| e.vectors[(i, k)].conj() * e.vectors[(i, l)]).sum();
                sum += c[k].conj() * c[l] * overlap * C64::new(0.0, 1.0) / (e.values[k].conj() - e.values[l]);
            }
        }
        sum.re
    };
    // The slow mode: the eigenvector of the larger P weight.
    let weight = |k: usize| e.vectors[(0, k)].norm_sqr() / (e.vectors[(0, k)].norm_sqr() + e.vectors[(1, k)].norm_sqr());
    let slow = if weight(0) > weight(1) { 0 } else { 1 };
    let clock = |c: f64, gamma: f64| if c > 1.0 { (1.0 + c.ln()) / gamma } else { c / gamma };
    let share = crate::model::elim::share();
    for (td, expected) in [(0.0, 2.9e-2), (1.0 / GAMMA, 3.0e-2), (4.0 / GAMMA, 1.6e-2)] {
        let psi = evolved([C64::ONE, C64::ZERO], td);
        let scale = 1.0 / (psi[0].norm_sqr() + psi[1].norm_sqr()).sqrt();
        let psi = psi.map(|z| z * scale);
        let exact = waiting(psi);
        // Its slow part along the fast span on the exact slow span: the slow eigenvector's coefficient.
        let alpha = on(psi)[slow];
        let c_exact = alpha.norm_sqr() * (e.vectors[(0, slow)].norm_sqr() + e.vectors[(1, slow)].norm_sqr());
        let error = clock(c_exact, -2.0 * e.values[slow].im) / exact - 1.0;
        let mut state = vec![C64::ZERO; n];
        (state[coupled[0]], state[coupled[1]]) = (psi[0], psi[1]);
        let (fallen, ring, _) = high.take_onto(&state, folded, &low).expect("the fall");
        let c = fallen[0].norm_sqr();
        println!("t_d {:.2}/Γ: δp {:+.4} (on the exact span {:+.4}), the clock's mean waiting on the exact span {:+.2} % from the exact, on the fold's {:+.2} %; the fall's ring {ring:.3e} (derived {expected:.1e})", td * GAMMA, 1.0 - c, 1.0 - c_exact, 100.0 * error, 100.0 * (clock(c, gamma) / exact - 1.0));
        assert!((error - ring).abs() <= (0.15 + share) * ring, "the fall's ring reads the clock's error (its leading order, on T within the share)");
        assert!((ring - expected).abs() <= 0.15 * expected, "the ring as derived");
        assert!((c - c_exact).abs() <= 2.0 * share * (c_exact - 1.0).abs(), "the fold's slow part the exact span's within the series' share");
    }
}

/// R4_LIN_4's F = 1 set along thermal paths (BRIEF_STEP3C item 1): its fold does not exist at the focus (its series'
/// ratio 1.1 there; before, the run held its F′ = 2 node explicit for the whole run), and along the paths its ratio goes
/// up to 1.87; on the run's per-point decisions every step's middle exists -- no step is refused, its node rising before
/// -- and the steps stand on both sides (said).
#[test]
fn r4_lin_4_s_thermal_paths_meet_no_point_where_their_fold_does_not_exist() {
    use crate::step::{Dynamics, Midway, Trajectory};
    struct Anywhere;
    impl crate::step::Domain for Anywhere {
        fn runaway(&self, _: f64) -> bool {
            false
        }
    }
    let req = crate::model::testing::scene("R4_LIN_4");
    let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
    let hm = crate::units::hbar_over_mass(req.mass_kg);
    let thermal = (crate::units::KT_PER_UK * req.temperature_uk / hm).sqrt();
    let mut scratch = crate::step::Scratch::default();
    let (mut steps, mut explicit, mut switched, mut dropped) = (0, 0, 0, 0.0);
    let first = run.cluster(0).states[0];
    for i in 0..8u64 {
        let phase = 2.0 * std::f64::consts::PI * i as f64 / 8.0;
        let kappa = [thermal * phase.cos(), thermal * (2.0 * phase).sin(), 0.5 * thermal * phase.sin()];
        let mut t = Trajectory::new(&run, first, [0.2 * phase.sin(), 0.1 * phase.cos(), 0.0], kappa, 3, i);
        for _ in 0..60 {
            let before = t.cluster;
            t = Midway::new(&run, &t, 0.126).and_then(|m| m.attempt(&t, &crate::step::Brownian::STILL, &[], &mut scratch, &Anywhere)).expect("every step's middle exists").end;
            let c = run.cluster(t.cluster);
            steps += 1;
            explicit += usize::from(c.states.len() > c.ground);
            switched += usize::from(t.cluster != before && run.cluster(before).ground == c.ground && run.cluster(before).states[0] == c.states[0]);
        }
        dropped += t.dropped;
    }
    println!("R4_LIN_4: {steps} steps from F = 1, {explicit} steps' ends on an explicit cluster, {switched} switches, δp booked {dropped:.2e}");
}

/// The run is one object the trajectories' threads share (run.rs # The state sets; BRIEF_STEP4A 4b): R4_LIN_4's sixteen
/// thermal paths from every ground state in turn, whose nodes rise and fall (clusters made where first asked for, in whatever order the
/// threads ask), 60 steps each -- one after another on one run, and all at once on the threads of another: every
/// trajectory's end the same to the bit, its cluster by its states (a cluster's number is its run's handle).
#[test]
fn a_run_s_trajectories_are_the_same_to_the_bit_on_one_thread_and_on_many() {
    use crate::step::{Dynamics, Midway, Trajectory};
    use rayon::prelude::*;
    struct Anywhere;
    impl crate::step::Domain for Anywhere {
        fn runaway(&self, _: f64) -> bool {
            false
        }
    }
    let req = crate::model::testing::scene("R4_LIN_4");
    let thermal = (crate::units::KT_PER_UK * req.temperature_uk / crate::units::hbar_over_mass(req.mass_kg)).sqrt();
    // A trajectory's path: the clusters it took (their numbers), and every number's bits -- each step's cluster's states,
    // and at its end its place, momentum, time, clock, books and state.
    let end = |run: &run::Onefold, i: u64| -> (Vec<usize>, Vec<u64>) {
        let mut scratch = crate::step::Scratch::default();
        let phase = 2.0 * std::f64::consts::PI * i as f64 / 16.0;
        let kappa = [thermal * phase.cos(), thermal * (2.0 * phase).sin(), 0.5 * thermal * phase.sin()];
        let mut t = Trajectory::new(run, i as usize % GROUND, [0.2 * phase.sin(), 0.1 * phase.cos(), 0.0], kappa, 3, i);
        let (mut bits, mut visited) = (vec![], vec![t.cluster]);
        for _ in 0..60 {
            t = Midway::new(run, &t, 0.126).and_then(|m| m.attempt(&t, &crate::step::Brownian::STILL, &[], &mut scratch, &Anywhere)).expect("a step").end;
            bits.extend(run.cluster(t.cluster).states.iter().map(|&s| s as u64));
            visited.push(t.cluster);
        }
        visited.sort_unstable();
        visited.dedup();
        bits.extend(t.r.iter().chain(&t.kappa).chain([&t.time, &t.threshold, &t.dropped, &t.owed, &t.work]).map(|x| x.to_bits()));
        bits.extend(t.psi.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]));
        bits.extend([u64::from(t.jumps), t.lost.map_or(0, f64::to_bits)]);
        (visited, bits)
    };
    let reference = [1.0 / GROUND as f64; GROUND];
    let one = run::Onefold::new(&req, reference).unwrap();
    let serial: Vec<(Vec<usize>, Vec<u64>)> = (0..16).map(|i| end(&one, i)).collect();
    let many = run::Onefold::new(&req, reference).unwrap();
    let parallel: Vec<(Vec<usize>, Vec<u64>)> = (0..16).into_par_iter().map(|i| end(&many, i)).collect();
    println!("the clusters each path took: one thread {:?}; {} threads {:?}", serial.iter().map(|x| &x.0).collect::<Vec<_>>(), rayon::current_num_threads(), parallel.iter().map(|x| &x.0).collect::<Vec<_>>());
    for (i, (a, b)) in serial.iter().zip(&parallel).enumerate() {
        assert_eq!(a.1, b.1, "trajectory {i}");
    }
}




/// The splitting's reference and the loss energy on the fold (model/fold/run.rs, # The rest of the model; BRIEF_STEP3C
/// item 3): on R4_LIN_4, R1_TWEEZER and R1_RETRO, at the prepared trap's centre, a thermal offset and 10 µm along each
/// axis, the reference -- the trap light's held forms at the focus -- against the old model's prepared potential of the
/// reference state (the bare fold's far forms, `Model::expansion`): value, gradient and Hessian within the share of their
/// size; and the loss energy's potential at rest, V_run(r; p) of the folds with every light and with every light but the
/// trap's, against the same difference of `Fold::at`'s generators (the prototype's truth) within the share of the
/// reference's depth. The old model's running potential is said beside: its bare folds hold none of what the trap light
/// does to the other lights' shifts (model/potential.rs, # An averaged light), the fold at the atom holds it.
#[test]
fn the_reference_and_the_loss_energy_are_the_trap_light_s_on_the_fold() {
    use crate::model::{Model, Point};
    use crate::step::Dynamics;
    let share = super::super::elim::share();
    for name in ["R4_LIN_4", "R1_TWEEZER", "R1_RETRO"] {
        let request = crate::model::testing::scene(name);
        let light = Light::new(&request);
        let chief = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
        let mut p = [0.0; GROUND];
        p[chief] = 1.0;
        let old = Model::new(&request, &light, &[]).unwrap();
        let trap = old.trap(&p, &old.prepared()).unwrap();
        let run = super::run::Onefold::new(&request, p).expect("the run");
        let depth = -old.potential(trap.centre, &p, &old.prepared());
        let c = trap.centre;
        let points = [c, [c[0] + 0.3, c[1] - 0.2, c[2] + 0.5], [c[0] + 10.0, c[1], c[2]], [c[0], c[1] + 10.0, c[2]], [c[0], c[1], c[2] + 10.0]];
        let size = |q: &crate::model::Quadratic| (q.value.abs(), q.gradient.iter().map(|x| x * x).sum::<f64>().sqrt(), q.hessian.iter().flatten().map(|x| x * x).sum::<f64>().sqrt());
        let set = run.cluster_of(chief);
        let psi: Vec<C64> = run.cluster(set).states.iter().map(|&s| if s == chief { C64::ONE } else { C64::ZERO }).collect();
        for r in points {
            let (new, was) = (run.reference(r), old.expansion(r, &p, &old.prepared()));
            let apart = crate::model::Quadratic { value: new.value - was.value, gradient: std::array::from_fn(|a| new.gradient[a] - was.gradient[a]), hessian: std::array::from_fn(|a| std::array::from_fn(|b| new.hessian[a][b] - was.hessian[a][b])) };
            let (d, s) = (size(&apart), size(&was));
            let energy = run.energy(set, &Point { r, kappa: [0.0; 3], t: 0.0 }, &psi);
            let running = old.potential(r, &p, &old.running());
            // V_run of `Fold::at` on the folds with every light and with every light but the trap's (each ground set's
            // with nothing explicit; the chief state's diagonal in the lab).
            let diagonal = |request: &Request| -> f64 {
                let choice = Choice::base(request, vec![]).unwrap();
                let fold = Fold::new(request, &choice, chief).unwrap();
                let x = fold.ground.iter().position(|&g| g == chief).unwrap();
                fold_at(&fold, r, [0.0; 3]).expect("the fold at the point").generator[(x, x)].re + fold.frame_values[x]
            };
            let at = diagonal(&request) - diagonal(&super::run::lit(&request, &light, |w| w != light.trap));
            println!("{name} at {r:.2?}: the reference {:.6e} (old {:.6e}), apart: value {:.2e}, gradient {:.2e}, Hessian {:.2e} of {:.2e}, {:.2e}, {:.2e}; V_run {energy:.6e} (Fold::at's {at:.6e}; the old model's {running:.6e}; depth {depth:.4e})", new.value, was.value, d.0, d.1, d.2, s.0, s.1, s.2);
            assert!(d.0 <= share * s.0.max(1e-300) + 1e-12 * depth && d.1 <= share * s.1 + 1e-12 * depth && d.2 <= share * s.2 + 1e-12 * depth, "{name}: the reference against the prepared potential");
            assert!((energy - at).abs() <= share * depth, "{name}: the loss energy against Fold::at's");
        }
    }
}

/// The loss energy where a light set's fold does not exist (BRIEF_STEP3E item 3; run.rs # The rest of the model): a
/// resonant two-level atom at rest at the focus, s = 0.5 and 0.6 -- its fold with nothing explicit exists at the first
/// and not at the second (its series' ratio beyond one, BRIEF_STEP3C) --: the light set's cluster rises there as a
/// trajectory's start rises it (`Clusters::rise`), and the energy is a number, not none. On resonance the light shifts
/// nothing: at both it is zero within the share of the light's scale (Ω/2)²/(Γ/2).
#[test]
fn the_loss_energy_rises_where_a_light_set_s_fold_does_not_exist() {
    use crate::model::Point;
    use crate::step::Dynamics;
    for s in [0.5, 0.6] {
        let req = two_level(0.0, s);
        let mut p = [0.0; GROUND];
        p[0] = 1.0;
        let run = super::run::Onefold::new(&req, p).expect("the run");
        let fold = Fold::new(&req, &Choice::base(&req, vec![]).unwrap(), 0).unwrap();
        let exists = fold.prototype([0.0; 3], [0.0; 3], &mut super::proto::Cache::default(), &mut fold.prototype_workspace().unwrap()).is_some();
        let set = run.cluster_of(0);
        let psi: Vec<C64> = run.cluster(set).states.iter().map(|&x| if x == 0 { C64::ONE } else { C64::ZERO }).collect();
        let energy = run.energy(set, &Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, &psi);
        let scale = 0.5 * GAMMA * s / 2.0;
        println!("s {s}: the fold with nothing explicit {} at the focus; the loss energy at rest {energy:.3e} (the light's scale {scale:.2})", if exists { "exists" } else { "does not exist" });
        assert_eq!(exists, s < 0.55, "the fold at s = {s}");
        assert!(energy.abs() <= crate::model::elim::share() * scale, "s {s}: the loss energy {energy}");
    }
}

/// A request whose idle excited states lie at a laser's energy (BRIEF_STEP3C item 3): a two-level atom four linewidths
/// below its line, the 31 other excited states idle at zero -- where the laser's energy is -- and undecaying. The run is made (no model of
/// every bare drive, whose denominators vanish there, is built), its reference and loss energy are finite, and atoms
/// step.
#[test]
fn a_request_whose_idle_excited_states_lie_at_a_laser_s_energy_is_run() {
    use crate::model::Point;
    use crate::step::{Dynamics, Midway, Trajectory};
    struct Anywhere;
    impl crate::step::Domain for Anywhere {
        fn runaway(&self, _: f64) -> bool {
            false
        }
    }
    let req = system(&[(0, 0.0)], &[(17, 4.0 * GAMMA, GAMMA, 0)], &[Ray { axis: 0, nu: 0.0, direction: [0.0, 0.0, 1.0], couplings: vec![(17, 0, C64::real(0.5 * GAMMA))] }], 1, true);
    let mut p = [0.0; GROUND];
    p[0] = 1.0;
    let run = super::run::Onefold::new(&req, p).expect("the run of idle states at the laser's energy");
    let r = [0.3, -0.2, 0.1];
    let q = run.reference(r);
    let set = run.cluster_of(0);
    let psi: Vec<C64> = run.cluster(set).states.iter().map(|&s| if s == 0 { C64::ONE } else { C64::ZERO }).collect();
    let energy = run.energy(set, &Point { r, kappa: [0.0; 3], t: 0.0 }, &psi);
    println!("the reference {q:?}; the loss energy at rest {energy:.6e}");
    assert!(q.value.is_finite() && q.gradient.iter().chain(q.hessian.iter().flatten()).all(|x| x.is_finite()) && energy.is_finite(), "finite");
    let mut t = Trajectory::new(&run, 0, r, [0.0; 3], 17, 0);
    for _ in 0..20 {
        t = Midway::new(&run, &t, 0.05).and_then(|m| m.attempt(&t, &crate::step::Brownian::STILL, &[], &mut crate::step::Scratch::default(), &Anywhere)).expect("a step").end;
    }
    println!("after 20 steps: κ {:?}, ‖ψ‖² {:.6}", t.kappa, crate::linalg::norm_sqr(&t.psi));
    assert!(t.kappa.iter().all(|x| x.is_finite()) && t.psi.iter().all(|z| z.re.is_finite() && z.im.is_finite()), "the atom steps");
}

/// The force's tone part node by node (model/fold/run.rs `Middle::tone_force`; BRIEF_STEP3D item 3): on the Λ with a
/// tone (`lambda_request`: B two counter-propagating beams, the form between |1⟩ and |2⟩ beating), an atom moving along
/// B at k·v h = 0.6, random Hermitian sums σ_j = re_j + i im_j at the step's three nodes: the candidate's forms on them are
/// the central differences of its envelopes there, Σ_j Re tr(G₊(s_j) σ_j + G₋(s_j) σ_j†) at r ± δ along each lab axis --
/// one formula with the envelopes the propagation and the hazard take (`Middle::envelopes`).
#[test]
fn a_tone_s_forms_node_by_node_are_its_envelopes_differences() {
    use crate::step::{Dynamics, Held, Middle};
    let req = lambda_request(1.5, 0.4, -5e-6);
    let mut p = [0.0; GROUND];
    p[0] = 1.0;
    let run = run::Onefold::new(&req, p).expect("the run");
    let set = run.cluster_of(0);
    let (k, h, t) = (req.beams[1].wave_number, 0.05, 0.37);
    let velocity = [0.0, 0.0, 0.6 / (k * h)];
    let r = [0.2, -0.1, 0.3];
    let middle = |r: [f64; 3]| run.middle(set, r, t, &Held::default(), false);
    let at = middle(r);
    let n = at.states();
    let path = at.path(velocity.map(|v| v / run.hbar_over_mass()), velocity);
    let (mut beats, mut keys) = (vec![], vec![]);
    at.beats(&path, &mut beats, &mut keys);
    assert!(!beats.is_empty(), "the Λ's tone a candidate");
    let mut state = 20261116u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut hermitian = || -> Vec<C64> {
        let mut m = vec![C64::ZERO; n * n];
        for a in 0..n {
            for b in a..n {
                let z = if a == b { C64::real(next()) } else { C64::new(next(), next()) };
                (m[a * n + b], m[b * n + a]) = (z, z.conj());
            }
        }
        m
    };
    let nodes = [-h, 0.0, h];
    let sums: Vec<(Vec<C64>, Vec<C64>)> = nodes.iter().map(|_| (hermitian(), hermitian())).collect();
    let given: Vec<(f64, &[C64], &[C64])> = nodes.iter().zip(&sums).map(|(&s, (re, im))| (s, &re[..], &im[..])).collect();
    // Its envelopes at the nodes on the sums, Σ_j Re tr(G₊ σ_j + G₋ σ_j†), at a point.
    let value = |r: [f64; 3], m: usize| -> f64 {
        let other = middle(r);
        let path = other.path(velocity.map(|v| v / run.hbar_over_mass()), velocity);
        let mut out = 0.0;
        for &(s, re, im) in &given {
            let (mut plus, mut minus) = (Mat::zeros(n, n), Mat::zeros(n, n));
            other.envelopes(&path, m, s, &mut plus, &mut minus);
            for a in 0..n {
                for b in 0..n {
                    let sigma = re[b * n + a] + C64::I * im[b * n + a];
                    let dagger = re[b * n + a] - C64::I * im[b * n + a];
                    out += (plus[(a, b)] * sigma + minus[(a, b)] * dagger).re;
                }
            }
        }
        out
    };
    let delta = 1e-4;
    let mut worst = 0.0_f64;
    for m in 0..beats.len() {
        let form = at.tone_force(&path, m, &given);
        let difference: [f64; 3] = std::array::from_fn(|a| {
            let (mut plus, mut minus) = (r, r);
            plus[a] += delta;
            minus[a] -= delta;
            (value(plus, m) - value(minus, m)) / (2.0 * delta)
        });
        let size = difference.iter().map(|x| x.abs()).fold(1e-300, f64::max);
        let apart = (0..3).map(|a| (form[a] - difference[a]).abs()).fold(0.0, f64::max) / size;
        println!("candidate {m} (ν {:.4} rad/µs): the forms node by node {form:?}, the envelopes' differences {difference:?} ({apart:.2e} of their size)", beats[m]);
        worst = worst.max(apart);
    }
    assert!(worst <= 1e-6, "a tone's forms node by node are its envelopes' differences: {worst:.2e}");
}

/// A held atom's mean force on the run (heavy: it does not move): `count` trajectories from the ground state 0 at r,
/// each `steps` steps of τ, every kick the run makes (the steps' forms, the scattered light's force, the jumps');
/// the mean of κ's change per µs and its standard error (1/µm/µs).
fn held_mean_force(run: &run::Onefold, r: [f64; 3], count: u64, steps: usize, tau: f64) -> ([f64; 3], [f64; 3]) {
    use crate::step::{Midway, Trajectory};
    struct Anywhere;
    impl crate::step::Domain for Anywhere {
        fn runaway(&self, _: f64) -> bool {
            false
        }
    }
    let mut scratch = crate::step::Scratch::default();
    let time = steps as f64 * tau;
    let changes: Vec<[f64; 3]> = (0..count)
        .map(|i| {
            let mut t = Trajectory::new(run, 0, r, [0.0; 3], 11, i);
            for _ in 0..steps {
                t = Midway::new(run, &t, tau).and_then(|m| m.attempt(&t, &crate::step::Brownian::STILL, &[], &mut scratch, &Anywhere)).expect("every step's middle exists").end;
            }
            t.kappa.map(|k| k / time)
        })
        .collect();
    let mean: [f64; 3] = std::array::from_fn(|a| changes.iter().map(|c| c[a]).sum::<f64>() / count as f64);
    let error = std::array::from_fn(|a| (changes.iter().map(|c| (c[a] - mean[a]).powi(2)).sum::<f64>() / ((count - 1) * count) as f64).sqrt());
    (mean, error)
}

/// A two-level atom held in a running wave (BRIEF_STEP3C test table): its mean force is the master equation's radiation
/// pressure, k Γ ρ_ee with ρ_ee = (s/2)/(1 + s + (2Δ/Γ)²), along the wave and nothing across it -- on resonance at s = 0.3
/// and at Δ = −2Γ, s = 2 (each within four standard errors and the share).
#[test]
fn a_two_level_atom_s_mean_force_in_a_running_wave_is_the_master_equation_s() {
    let share = crate::model::elim::share();
    for (delta, s) in [(0.0, 0.3), (-2.0 * GAMMA, 2.0)] {
        let req = two_level(delta, s);
        let k = req.beams[0].wave_number;
        let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
        let (mean, error) = held_mean_force(&run, [0.0; 3], 48, 200, 0.1);
        let master = k * GAMMA * 0.5 * s / (1.0 + s + (2.0 * delta / GAMMA).powi(2));
        println!("Δ {:.1}Γ, s {s}: the run's mean force {mean:.3?} ± {error:.3?} 1/µm/µs against the master equation's {master:.3} along the wave", delta / GAMMA);
        assert!((mean[2] - master).abs() <= 4.0 * error[2] + share * master, "Δ {delta}, s {s}: along the wave {} ± {} against {master}", mean[2], error[2]);
        assert!((0..2).all(|a| mean[a].abs() <= 4.0 * error[a] + share * master), "nothing across the wave");
    }
}

/// A two-level atom held in a standing wave (BRIEF_STEP3C test table): two counter-propagating beams of one frequency,
/// the coupling 2g cos kz, s(z) = 4 s₁ cos² kz; at kz = π/8, red of the line (Δ = −2Γ, s₁ = 0.5) and on it (Δ = 0,
/// s₁ = 0.1), the mean force along the wave is the master equation's dipole force −(Δ/2) s′(z)/(1 + s(z) + (2Δ/Γ)²) --
/// the two beams' pressures cancel -- within four standard errors and the share of its size.
#[test]
fn a_two_level_atom_s_dipole_force_in_a_standing_wave_is_the_master_equation_s() {
    let share = crate::model::elim::share();
    for (delta, s1) in [(-2.0 * GAMMA, 0.5), (0.0, 0.1)] {
        let (ee, g) = (1000.0, 0.5 * GAMMA * (s1 / 2.0_f64).sqrt());
        let rays = [1.0, -1.0].map(|d| Ray { axis: 0, nu: ee + delta, direction: [0.0, 0.0, d], couplings: vec![(17, 0, C64::real(g))] });
        let mut req = system(&[(0, 0.0)], &[(17, ee, GAMMA, 0)], &rays, 1, true);
        idle_decay(&mut req, GAMMA);
        req.validate().unwrap();
        let k = req.beams[0].wave_number;
        let z = std::f64::consts::FRAC_PI_8 / k;
        let run = run::Onefold::new(&req, [1.0 / GROUND as f64; GROUND]).unwrap();
        let (mean, error) = held_mean_force(&run, [0.0, 0.0, z], 48, 200, 0.1);
        let (s, slope) = (4.0 * s1 * (k * z).cos().powi(2), -4.0 * s1 * k * (2.0 * k * z).sin());
        let master = -0.5 * delta * slope / (1.0 + s + (2.0 * delta / GAMMA).powi(2));
        let scale = master.abs().max(k * GAMMA * 0.5 * s / (1.0 + s + (2.0 * delta / GAMMA).powi(2)));
        println!("Δ {:.1}Γ, s₁ {s1}: the run's mean force {mean:.3?} ± {error:.3?} 1/µm/µs against the master equation's dipole force {master:.3}", delta / GAMMA);
        assert!((mean[2] - master).abs() <= 4.0 * error[2] + share * scale, "Δ {delta}, s₁ {s1}: along the wave {} ± {} against {master}", mean[2], error[2]);
    }
}

/// A scene's risen clusters hold their copies at a P place's frame (BRIEF_COPIES item 4; proto.rs `prototype_workspace`): of
/// the request in SCENE, every candidate node of every set risen alone on its set's base choice (`Choice::base_of`), each
/// fold whose copy rows hold one at a P place's frame -- tilt2 F=2's three were refused so with level 5 risen, the copy of a
/// P state (3, 4, 5) at another P state's frame -- is held by the prototype where its static coupling to that place can be
/// folded (P leaves the same coupling out of its entries), and each such coupling's mixing is said.
#[test]
#[ignore = "a check on a scene's request: run by name with SCENE"]
fn a_scene_s_risen_clusters_hold_their_copies_at_a_p_frame() {
    let name = std::env::var("SCENE").expect("SCENE: a request file's path");
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&name).unwrap()).unwrap();
    let request = crate::request::Request::from_native(&line).unwrap();
    let base = Choice::base(&request, vec![]).unwrap();
    let (mut held, mut refused) = (0, 0);
    for set in Fold::sets(&request, &base).unwrap().iter().filter(|s| s.len() > 1) {
        let fold = Fold::new(&request, &base, set[0]).unwrap();
        for (key, _, _) in Choice::candidates(&fold) {
            let Ok(choice) = Choice::base_of(&request, vec![key], &|s| s == &set[..]) else { continue };
            let Ok(risen) = Fold::new(&request, &choice, set[0]) else { continue };
            let ng = risen.ground.len();
            let mut on: Vec<(usize, usize)> = vec![];
            for b in &risen.blocks {
                for r in &b.rows {
                    if let Row::Copy { state, component } = *r {
                        let omega = b.offset.plus(-1, &risen.components[component].beat);
                        on.extend((0..ng).filter(|&p| risen.frames[p] == omega).map(|p| (state, p)));
                    }
                }
            }
            if on.is_empty() {
                continue;
            }
            match risen.prototype_workspace() {
                Ok(_) => held += 1,
                Err(why) => {
                    refused += 1;
                    println!("node {key:?} refused: {why}");
                }
            }
            println!("node {key:?}: copies at a P frame {:?}, their static couplings foldable {}", on, on.iter().all(|&(t, p)| !risen.ground_tone[t * GROUND + risen.ground[p]]));
        }
    }
    println!("{held} risen clusters with copies at a P frame held, {refused} refused");
    assert!(held > 0 && refused == 0, "every one held");
}
