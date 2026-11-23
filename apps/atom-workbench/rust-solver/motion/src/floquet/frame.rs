//! The frame and the slow lattice (td_s.choose_frame, mdlat.Prim / choose_frame_x / PerM / hermite).
//!
//! Frequencies are exact integer vectors over B = (the scene's generators, α): a Hilbert component's key with α = 0; the
//! frame θ_a = α m_a + β_level(a) an integer vector per state; a Liouville component's vector their difference.
//!
//! # Primitives
//! Every frequency is an exact rational vector over the primitive generators: the request's frequency axes Φ_i
//! (independent by the request's construction) and the Doppler classes δ_c. A beam's Doppler shift w_l = r_l δ_c, r_l
//! rational: two shifts of one laser (wave number and waist) are in one class iff their ratio is p/q to the rounding level
//! of their evaluation, |q x − p y| ≤ γ₈(|q| A_x + |p| A_y), with max(|p|, |q|) below the Dirichlet bound γ₈^(−1/2)
//! (every real ratio has a relation at that size, so a match there carries no information); shifts of different lasers
//! are independent (a ratio of optical frequencies). A zero shift is no class. A_x is taken as |x| (the shifts' terms
//! are not passed; the two-point check's points have no shift).
//!
//! # The frame
//! θ is an exact change of variables, any choice is exact; the choice (choose_frame_x) only keeps the strong couplings
//! static: the couplings ranked by their first-order amplitude |V| / |E_e − E_g − f − iΓ_e/2|, α = (f_a − f_b)/(m_a − m_b)
//! from pairs of the strong ones on one ground level, β per level from the strongest coupling into it. The constants of
//! that choice (`STRONG`, `STATIC`, `BAND`, `TIE`, `DIGITS`) decide nothing but which exact frame is used (td_s's, kept
//! so that the elements are labelled as the prototype's).
//!
//! A ground level's strongest coupling may be of the second order: two beams l, m through one excited state e couple
//! the ground states s ← s′ by W = ½ V_es* V_es′ (1/Δ_s + 1/Δ_s′) (the Raman coupling, RS's pair) at the two-photon
//! detuning δ = (E_s′ + f_m) − (E_s + f_l), its amplitude |W| / |δ|. Where it is above the first-order amplitude of the
//! coupling that would set the level, the level is set from it -- θ_s − θ_s′ = e_m − e_l, so that the Raman coupling is
//! static (`Raman`). A Raman pair far from its excited states (R6_RAMAN_CO, R6_RAMAN_COUNTER: 15 GHz on D1, 0.1 MHz
//! from the two-photon resonance) has first-order amplitudes ~10⁻³ and a two-photon one ~10⁻¹: by the first order alone
//! both ground levels took the frame of the stronger beam, the same vector, φ = 0 (2026-11-20, pstate.status.md 4.5).
//! In the cooling scenes the resonant first-order couplings stay the strongest and the frame is the same.
//!
//! The lattice's split (# The lattice) reads an element's frame vector c − θ_b + θ_a as its frequency, which is its
//! detuning f_c − (E_b − E_a) only where every state's diagonal in the frame, D_a = E_a − θ_a, is slow. A level set
//! from a laser of another line is not: R5_PUMP's F = 1 set from its D2 repump and F = 2 from its D1 pump are φ =
//! 4.48·10⁷ rad/µs apart (the D1–D2 axis), and F = 1's diagonal is 0.999 φ -- its static elements to F = 2 (the pump
//! and repump's Raman pair) were slow in the frame and 7 THz off resonance, joining the two levels at that bandwidth
//! (R9_HEAT the same at −4.63 φ). So each state's θ is then moved, exactly, to the lattice point nearest its energy
//! along φ (`settle`): by the whole number k_a = round((D_a − D_r)/φ) of φ's vector, D_r the first ground state's
//! diagonal (no energy zero is the request's to set: a common offset moves nothing). φ is the levels' as chosen. An
//! element's fast part is then that of its detuning -- the split's own rounding of a frequency on the lattice, read
//! of the frequency the frame does not choose. Where every diagonal is within half a φ of the reference's, nothing moves.
//!
//! # The lattice
//! A vector's fast part n = (n_φ, n_tw, n_ν) is split off exactly (`Frame::split`): n_ν the carrier axis' coefficient,
//! n_tw the tweezer's own axis', n_φ = round(f/φ), φ = β_g2 − β_g1 (the ground hyperfine in the frame). The slow
//! remainders of every Liouville component and readout row generate a Z-module; its Hermite basis b gives the slow
//! coordinates m (`Lattice`). A lattice key (m, n) has the frequency m·b + n·(φ, Φ_tw, Φ_ν). The split is the
//! product's (the fold's sectors, `Frame::sectors`); the lattice is the tests' static reference's (floquet/mod.rs).

use crate::linalg::larger;
use std::cmp::Ordering;

use super::Hilbert;
use crate::linalg::{C64, Mat};
use crate::request::{Level, Manifold, STATES};
use crate::scene::{Key, Scene};

/// An exact frequency vector over B = (generators, α).
pub type FrameKey = Vec<i64>;

/// γ₈ = 8u/(1 − 8u): the relative rounding of an 8-term dot product's evaluation (Higham's γ_n), u the unit roundoff.
const GAMMA: f64 = 8.0 * crate::units::UNIT_ROUNDOFF / (1.0 - 8.0 * crate::units::UNIT_ROUNDOFF);
/// The frame's choice (module documentation, # The frame): couplings above this share of the strongest pair up for α.
const STRONG: f64 = 0.05;
/// A residual frequency (rad/µs) at or below this counts as static in a candidate frame's score.
const STATIC: f64 = 1e-6;
/// The couplings above this share of the strongest set a candidate's band.
const BAND: f64 = 1e-6;
/// Scores within this share of the strongest coupling tie.
const TIE: f64 = 1e-9;
/// α candidates equal to this many decimals are one.
const DIGITS: usize = 10;

/// A second-order coupling between two ground levels' states through one excited state (`Frame::new`): |W|/|δ|, the
/// levels of s (`to`) and s′ (`from`), the beams l (into s) and m (into s′), their frequencies, m_s − m_s′.
struct Raman {
    amplitude: f64,
    to: Lev,
    from: Lev,
    l: usize,
    m: usize,
    f: (f64, f64),
    dm: i64,
}

/// An exact rational number, normalized (den > 0, gcd 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ratio {
    pub num: i64,
    pub den: i64,
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// ⌊a/b⌋ (Python's //).
#[cfg(test)]
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

impl Ratio {
    pub const ZERO: Ratio = Ratio { num: 0, den: 1 };
    pub fn new(num: i64, den: i64) -> Ratio {
        let g = gcd(num, den).max(1) * den.signum();
        Ratio { num: num / g, den: den / g }
    }
    pub fn int(n: i64) -> Ratio {
        Ratio { num: n, den: 1 }
    }
    fn add(self, o: Ratio) -> Ratio {
        Ratio::new(self.num * o.den + o.num * self.den, self.den * o.den)
    }
    fn sub(self, o: Ratio) -> Ratio {
        self.add(Ratio { num: -o.num, den: o.den })
    }
    fn mul(self, o: Ratio) -> Ratio {
        Ratio::new(self.num * o.num, self.den * o.den)
    }
    fn value(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl PartialOrd for Ratio {
    fn partial_cmp(&self, o: &Ratio) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ratio {
    fn cmp(&self, o: &Ratio) -> Ordering {
        (self.num as i128 * o.den as i128).cmp(&(o.num as i128 * self.den as i128))
    }
}

/// Σ x_i correctly rounded (Shewchuk's partials with the half-way correction, as Python's math.fsum): the value of an
/// exact vector, free of the large axes' cancellation.
pub fn exact_sum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in xs {
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    let Some(mut n) = partials.len().checked_sub(1) else {
        return 0.0;
    };
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}

/// The continued-fraction convergents (p, q), q > 0, of x below the bound.
fn convergents(x: f64, bound: i64) -> Vec<(i64, i64)> {
    let a0 = x.floor();
    let (mut p0, mut q0, mut p1, mut q1) = (1i64, 0i64, a0 as i64, 1i64);
    let mut out = vec![(p1, q1)];
    let mut r = x - a0;
    while r > 0.0 && q1 < bound {
        let y = 1.0 / r;
        let a = y.floor();
        r = y - a;
        let a = a as i64;
        (p0, q0, p1, q1) = (p1, q1, a.saturating_mul(p1).saturating_add(p0), a.saturating_mul(q1).saturating_add(q0));
        if q1 >= bound {
            break;
        }
        out.push((p1, q1));
    }
    out
}

/// x = (p/q) y to the rounding level of their evaluations (A_x, A_y their absolute sums), max(|p|, q) below the bound.
fn relation(x: f64, ax: f64, y: f64, ay: f64) -> Option<Ratio> {
    let bound = (1.0 / GAMMA.sqrt()) as i64;
    for (p, q) in convergents(x / y, bound) {
        if p.abs().max(q) >= bound {
            return None;
        }
        if (q as f64 * x - p as f64 * y).abs() <= GAMMA * (q as f64 * ax + p.abs() as f64 * ay) {
            return Some(Ratio::new(p, q));
        }
    }
    None
}

/// A level of the frame: a ground hyperfine level, or an excited manifold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Lev {
    Ground(u8),
    Excited(u8),
}

fn lev(level: &Level) -> Lev {
    match level.manifold {
        Manifold::Ground => Lev::Ground(level.f),
        Manifold::D1 => Lev::Excited(1),
        Manifold::D2 => Lev::Excited(2),
    }
}

/// The primitive generators: the request's axes, then the Doppler classes.
#[derive(Clone, Debug, PartialEq)]
struct Primitives {
    axes: usize,
    /// Each generator's frequency index (request axes) and its Doppler class with its ratio.
    generators: Vec<(Vec<i32>, Option<(usize, Ratio)>)>,
    /// The primitives' values (rad/µs): Φ_i, then δ_c.
    values: Vec<f64>,
}

impl Primitives {
    /// u over the generators → its primitive vector: Σ_g u_g (frequency index) − Σ_g u_g r_g δ_c(g).
    fn of(&self, u: &[Ratio]) -> Vec<Ratio> {
        let mut out = vec![Ratio::ZERO; self.values.len()];
        for (c, (index, class)) in u.iter().zip(&self.generators) {
            if c.num == 0 {
                continue;
            }
            for (i, &x) in index.iter().enumerate() {
                if x != 0 {
                    out[i] = out[i].add(c.mul(Ratio::int(x as i64)));
                }
            }
            if let Some((k, r)) = class {
                out[self.axes + k] = out[self.axes + k].sub(c.mul(*r));
            }
        }
        out
    }
    /// A primitive vector's value, correctly rounded.
    fn value(&self, p: &[Ratio]) -> f64 {
        exact_sum(p.iter().zip(&self.values).filter(|(c, _)| c.num != 0).map(|(c, v)| c.value() * v))
    }
}

/// The frame: θ_a per state and the exact data that split a vector into its fast part and its slow remainder.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// θ_a over B, one per state of the 32.
    pub theta: Vec<FrameKey>,
    /// The float value of each coordinate of B (rad/µs): the generators' rest-frame frequencies, then α.
    pub values: Vec<f64>,
    /// (φ, Φ_tw, Φ_ν) (rad/µs).
    pub fast: [f64; 3],
    primitives: Primitives,
    /// α as a rational vector over the generators.
    alpha: Vec<Ratio>,
    /// φ's primitive vector (None with fewer than two ground levels in the frame).
    phi: Option<Vec<Ratio>>,
    /// The tweezer's own axis (scene.rs `tweezer_axis`: one no other generator uses) and its frequency index; none for a
    /// lone tweezer at the frame's carrier.
    tweezer: Option<(usize, Vec<i32>)>,
}

impl Frame {
    /// The frame at a point: the reference there (the 32 states, the tweezer folded: `Hilbert::full`) and the Doppler shifts
    /// it was made with (one per beam). The frame's choice reads the explicit couplings of the beams other than the
    /// tweezer and the static energies.
    pub fn new(scene: &Scene, reference: &Hilbert, doppler: &[f64]) -> Frame {
        assert_eq!(reference.states.len(), STATES, "the frame is chosen on the reference");
        let request = &scene.request;
        let levels = request.levels().expect("validated labels");
        let phi_axes: Vec<f64> = request.frequencies_mhz.iter().map(|f| crate::units::RAD_PER_US_PER_MHZ * f).collect();
        let axes = phi_axes.len();
        let generators = &scene.generators;
        let count = generators.len();
        // The primitives: the Doppler classes per laser.
        let mut classes: Vec<(f64, f64, (u64, Option<u64>))> = Vec::new();
        let mut of_generator = Vec::new();
        for gen in generators {
            let Some(l) = gen.beam else {
                of_generator.push((gen.frequency_index.clone(), None));
                continue;
            };
            let b = &request.beams[l];
            let (x, laser) = (doppler[l], (b.wave_number.to_bits(), b.waist.map(f64::to_bits)));
            if x == 0.0 {
                of_generator.push((gen.frequency_index.clone(), None));
                continue;
            }
            let found = classes.iter().enumerate().find_map(|(c, &(y, ay, l))| if l == laser { relation(x, x.abs(), y, ay).map(|r| (c, r)) } else { None });
            let class = found.unwrap_or_else(|| {
                classes.push((x, x.abs(), laser));
                (classes.len() - 1, Ratio::int(1))
            });
            of_generator.push((gen.frequency_index.clone(), Some(class)));
        }
        let primitives = Primitives { axes, generators: of_generator, values: phi_axes.iter().copied().chain(classes.iter().map(|c| c.0)).collect() };
        // The couplings of the beams other than the tweezer in groups of one frequency (one primitive vector), each group
        // under its first beam's generator, in the prototype's order (its components sorted by key: the later group
        // first), entries row-major.
        let fixed = reference.hamiltonian.iter().find(|p| p.key.iter().all(|&k| k == 0)).expect("a static part");
        let energy: Vec<f64> = (0..STATES).map(|s| fixed.value[(s, s)].re).collect();
        let decay = scene.jumps.decay();
        let gamma: Vec<f64> = (0..STATES).map(|e| decay[(e, e)].re).collect();
        let unit = |g: usize| -> Vec<Ratio> { (0..count).map(|h| Ratio::int((h == g) as i64)).collect() };
        let mut groups: Vec<(Vec<Ratio>, usize, f64, Mat)> = Vec::new();
        let values: Vec<f64> = generators.iter().map(|g| g.frequency - g.beam.map_or(0.0, |l| doppler[l])).collect();
        for part in &reference.hamiltonian {
            if part.key.iter().filter(|&&k| k != 0).count() != 1 {
                continue;
            }
            let Some(g) = part.key.iter().position(|&k| k == 1) else { continue };
            if generators[g].beam.is_none_or(|l| l == request.trap_beam) {
                continue;
            }
            let p = primitives.of(&unit(g));
            match groups.iter_mut().find(|q| q.0 == p) {
                Some(group) => group.3.add_scaled(C64::ONE, &part.value),
                None => groups.push((p, g, values[g], part.value.clone())),
            }
        }
        // (amplitude, f, Δm, ground level, excited level, generator), and per term its excited and ground states, V and
        // its denominator.
        let mut terms: Vec<(f64, f64, i64, Lev, Lev, usize)> = Vec::new();
        let mut legs: Vec<(usize, usize, C64, C64)> = Vec::new();
        for (_, g, f, v) in groups.iter().rev() {
            for e in 0..STATES {
                for s in 0..STATES {
                    if v[(e, s)] == C64::ZERO || levels[e].is_ground() || !levels[s].is_ground() {
                        continue;
                    }
                    let den = C64::new(energy[e] - energy[s] - f, -0.5 * gamma[e]);
                    let amp = v[(e, s)].abs() / den.abs();
                    terms.push((amp, *f, levels[e].m as i64 - levels[s].m as i64, lev(&levels[s]), lev(&levels[e]), *g));
                    legs.push((e, s, v[(e, s)], den));
                }
            }
        }
        // The second-order couplings between ground states of two levels through one excited state (module
        // documentation, # The frame): per pair of legs (amplitude, s's level, s′'s level, l, m, f_l, f_m, m_s − m_s′).
        let mut raman: Vec<Raman> = Vec::new();
        for (i, &(e, s, vs, ds)) in legs.iter().enumerate() {
            for (j, &(e2, s2, vs2, ds2)) in legs.iter().enumerate() {
                if e2 != e || terms[i].3 == terms[j].3 {
                    continue;
                }
                let w = vs.conj() * vs2 * (ds.recip() + ds2.recip()) * 0.5;
                let delta = (energy[s2] + terms[j].1) - (energy[s] + terms[i].1);
                if w == C64::ZERO {
                    continue;
                }
                raman.push(Raman {
                    amplitude: w.abs() / delta.abs(),
                    to: terms[i].3,
                    from: terms[j].3,
                    l: terms[i].5,
                    m: terms[j].5,
                    f: (terms[i].1, terms[j].1),
                    dm: levels[s].m as i64 - levels[s2].m as i64,
                });
            }
        }
        let mut order: Vec<usize> = (0..terms.len()).collect();
        order.sort_by(|&a, &b| terms[b].0.partial_cmp(&terms[a].0).expect("finite amplitudes"));
        let terms: Vec<(f64, f64, i64, Lev, Lev, usize)> = order.iter().map(|&i| terms[i]).collect();
        let strongest = terms.first().map_or(0.0, |t| t.0);
        let strong: Vec<_> = terms.iter().filter(|t| t.0 > STRONG * strongest).collect();
        // α candidates: (rounded value, value, vector over the generators), the first of each rounded value.
        let mut candidates: Vec<(f64, f64, Vec<Ratio>)> = vec![(0.0, 0.0, vec![Ratio::ZERO; count])];
        for (i, a) in strong.iter().enumerate() {
            for b in &strong[i + 1..] {
                if a.3 == b.3 && a.2 != b.2 && a.1 != b.1 {
                    let dd = a.2 - b.2;
                    let (ua, ub) = (unit(a.5), unit(b.5));
                    let vector: Vec<Ratio> = ua.iter().zip(&ub).map(|(x, y)| Ratio::new(x.sub(*y).num, dd)).collect();
                    let value = primitives.value(&primitives.of(&vector));
                    let rounded: f64 = format!("{:.*}", DIGITS, value).parse().expect("a number");
                    if !candidates.iter().any(|c| c.0 == rounded) {
                        candidates.push((rounded, value, vector));
                    }
                }
            }
        }
        candidates.sort_by(|a, b| a.0.abs().partial_cmp(&b.0.abs()).expect("finite"));
        let size = count + 1;
        let betas = |alpha: f64| {
            let mut bet: Vec<(Lev, f64, FrameKey)> = Vec::new();
            for &(amp, f, dm, lg, le, g) in &terms {
                if !bet.iter().any(|b| b.0 == le) {
                    bet.push((le, 0.0, vec![0; size]));
                }
                if bet.iter().any(|b| b.0 == lg) {
                    continue;
                }
                // The level's strongest coupling: this one, or a Raman one to a ground level set before it.
                let second = raman
                    .iter()
                    .filter(|r| r.from == lg && r.amplitude > amp && bet.iter().any(|b| b.0 == r.to))
                    .max_by(|a, b| a.amplitude.partial_cmp(&b.amplitude).expect("finite amplitudes"));
                match second {
                    Some(r) => {
                        let (bt, mut v) = bet.iter().find(|b| b.0 == r.to).map(|b| (b.1, b.2.clone())).expect("set");
                        v[r.l] += 1;
                        v[r.m] -= 1;
                        v[count] += r.dm;
                        bet.push((lg, bt + r.f.0 - r.f.1 + alpha * r.dm as f64, v));
                    }
                    None => {
                        let (be, mut v) = bet.iter().find(|b| b.0 == le).map(|b| (b.1, b.2.clone())).expect("set");
                        v[g] -= 1;
                        v[count] += dm;
                        bet.push((lg, be - f + alpha * dm as f64, v));
                    }
                }
            }
            bet
        };
        let mut best: Option<(f64, usize, Vec<(Lev, f64, FrameKey)>, f64)> = None;
        for (c, (_, alpha, _)) in candidates.iter().enumerate() {
            let bet = betas(*alpha);
            let at = |l: Lev| bet.iter().find(|b| b.0 == l).map_or(0.0, |b| b.1);
            let residual: Vec<(f64, f64)> = terms.iter().map(|&(amp, f, dm, lg, le, _)| (amp, f - alpha * dm as f64 - (at(le) - at(lg)))).collect();
            // A term that is not a number would be left out of the score and the band, and the frame chosen without it.
            assert!(residual.iter().all(|r| !r.0.is_nan() && !r.1.is_nan()), "the frame's terms are not numbers");
            let score = residual.iter().filter(|r| r.1.abs() > STATIC).map(|r| r.0).fold(0.0, larger);
            let band = residual.iter().filter(|r| r.0 > BAND * strongest).map(|r| r.1.abs()).fold(0.0, larger);
            let better = match &best {
                None => true,
                Some((s, _, _, b)) => score < s - TIE * strongest || ((score - s).abs() <= TIE * strongest && band < b - TIE),
            };
            if better {
                best = Some((score, c, bet, band));
            }
        }
        let (_, chosen, bet, _) = best.expect("the zero candidate");
        let (_, alpha_value, alpha) = candidates.swap_remove(chosen);
        let theta = levels
            .iter()
            .map(|l| {
                let mut v = bet.iter().find(|b| b.0 == lev(l)).map_or_else(|| vec![0; size], |b| b.2.clone());
                v[count] += l.m as i64;
                v
            })
            .collect();
        let mut values = values;
        values.push(alpha_value);
        let mut grounds: Vec<&(Lev, f64, FrameKey)> = bet.iter().filter(|b| matches!(b.0, Lev::Ground(_))).collect();
        grounds.sort_by_key(|b| b.0);
        let tw = &request.beams[request.trap_beam].frequency_index;
        let tweezer = crate::scene::tweezer_axis(request).map(|i| (i, tw.clone()));
        let tweezer_frequency = tweezer.as_ref().map_or(0.0, |(_, idx)| idx.iter().zip(&phi_axes).map(|(&i, p)| i as f64 * p).sum());
        let mut frame = Frame { theta, values, fast: [0.0, tweezer_frequency, phi_axes[axes - 1]], primitives, alpha, phi: None, tweezer };
        // φ: the two ground levels' frames apart, where they are (an exact vector: two levels in one frame have no φ,
        // n_φ ≡ 0, and `split` reads none).
        if let [low, high] = grounds[..] {
            let v: FrameKey = high.2.iter().zip(&low.2).map(|(a, b)| a - b).collect();
            let p = frame.primitive(&v);
            if p.iter().any(|x| x.num != 0) {
                frame.fast[0] = v.iter().zip(&frame.values).map(|(&k, w)| k as f64 * w).sum();
                assert!(frame.fast[0].is_finite() && frame.fast[0] != 0.0, "the ground levels' frames apart by a nonzero frequency");
                frame.phi = Some(p);
                frame.settle(request, &v);
            }
        }
        frame
    }

    /// Each state's θ moved to the lattice point nearest its energy along φ (module documentation, # The frame): by
    /// k_a = round((D_a − D_r)/φ) times φ's vector `v`, D_a = E_a − θ_a its diagonal in the frame (E the request's static
    /// energies), D_r the first ground state's. An exact change of variables; φ itself is not moved.
    fn settle(&mut self, request: &crate::request::Request, v: &FrameKey) {
        let value = |t: &FrameKey, values: &[f64]| t.iter().zip(values).map(|(&k, w)| k as f64 * w).sum::<f64>();
        let diagonal: Vec<f64> = (0..STATES).map(|s| request.hamiltonian[(s, s)].re - value(&self.theta[s], &self.values)).collect();
        let reference = diagonal[0];
        for (theta, d) in self.theta.iter_mut().zip(&diagonal) {
            let k = ((d - reference) / self.fast[0]).round_ties_even();
            // A diagonal that is not a number moves no frame and reaches the model as it is (its energy no number).
            if k.is_finite() && k != 0.0 {
                for (t, x) in theta.iter_mut().zip(v) {
                    *t += k as i64 * x;
                }
            }
        }
    }

    /// A vector over B as a primitive vector (linear: what the frame reads of a vector, `split` and the lattice).
    pub(crate) fn primitive(&self, v: &FrameKey) -> Vec<Ratio> {
        let count = self.alpha.len();
        let u: Vec<Ratio> = (0..count).map(|g| Ratio::int(v[g]).add(Ratio::int(v[count]).mul(self.alpha[g]))).collect();
        self.primitives.of(&u)
    }

    /// A vector's fast part n = (n_φ, n_tw, n_ν) and its slow remainder (a primitive vector), exact (PerM.split).
    pub fn split(&self, v: &FrameKey) -> ([i64; 3], Vec<Ratio>) {
        let mut p = self.primitive(v);
        let carrier = self.primitives.axes - 1;
        assert_eq!(p[carrier].den, 1, "a whole number of carriers");
        let n_nu = p[carrier].num;
        p[carrier] = Ratio::ZERO;
        let mut n_tw = 0;
        if let Some((axis, index)) = &self.tweezer {
            let q = Ratio::new(p[*axis].num, p[*axis].den * index[*axis] as i64);
            assert_eq!(q.den, 1, "a whole number of tweezer frequencies");
            n_tw = q.num;
            for (i, &x) in index.iter().enumerate() {
                p[i] = p[i].sub(Ratio::int(n_tw * x as i64));
            }
        }
        let mut n_phi = 0;
        if let Some(phi) = &self.phi {
            n_phi = (self.primitives.value(&p) / self.fast[0]).round_ties_even() as i64;
            for (x, y) in p.iter_mut().zip(phi) {
                *x = x.sub(Ratio::int(n_phi).mul(*y));
            }
        }
        ([n_phi, n_tw, n_nu], p)
    }

    /// A Hilbert component's key (α = 0): its fast index n = (n_φ, n_tw, n_ν). The component is fast iff n ≠ 0: its
    /// frequency lies a whole number of the ground hyperfine φ (n_φ) or of the tweezer's or the carrier's frequency
    /// (n_tw, n_ν) off the slow lattice.
    pub fn fast_index(&self, key: &Key) -> [i64; 3] {
        let mut v: FrameKey = key.iter().map(|&k| k as i64).collect();
        v.push(0);
        self.split(&v).0
    }

    /// The model's sectors (model/rs.rs `fold`): the fast part (n_φ, n_tw) of each state's θ_a and of each generator. An
    /// element b ← a of a component with key c has the fast index `Sectors::element`(Σ_l c_l generators_l, states_b,
    /// states_a) = Σ_l c_l generators_l − states_b + states_a, the fast part of its frame vector (`element`). The carrier's
    /// n_ν is not carried: under the RWA only the tweezer's counter-rotating coupling holds the carrier, with n_tw ≠ 0.
    pub fn sectors(&self) -> Sectors {
        let pick = |n: [i64; 3]| [n[0], n[1]];
        let count = self.alpha.len();
        Sectors {
            states: self.theta.iter().map(|t| pick(self.split(t).0)).collect(),
            generators: (0..count).map(|g| pick(self.fast_index(&(0..count).map(|h| (h == g) as i32).collect()))).collect(),
        }
    }

    /// The value of a primitive vector (rad/µs), correctly rounded.
    #[cfg(test)]
    fn value(&self, p: &[Ratio]) -> f64 {
        self.primitives.value(p)
    }
}

/// The fast parts (n_φ, n_tw) of the 32 states' frames and of the generators (`Frame::sectors`; the reference's, the
/// tweezer's alone, model/rs.rs).
#[derive(Clone, Debug, PartialEq)]
pub struct Sectors {
    pub states: Vec<[i64; 2]>,
    pub generators: Vec<[i64; 2]>,
}

/// One coordinate of the frame vector of the element b ← a (row b, column a) of a component at the vector c: c − θ_b +
/// θ_a. In the frame ψ_a = e^{−iθ_a t} ψ̃_a, so H_ba e^{−ict} ψ_a enters i dψ̃_b/dt as H_ba e^{−i(c − θ_b + θ_a)t} ψ̃_a.
/// The one definition: the assembly's on whole vectors (floquet/assemble.rs `to_frame`), the fold's on their fast parts
/// (`Sectors::element`).
pub fn element(c: i64, row: i64, col: i64) -> i64 {
    c - row + col
}

impl Sectors {
    /// The fast index of the element b ← a of a component whose key's fast part is `key`, `row` and `col` the fast parts
    /// of b and a (`element` on the fast parts).
    pub fn element(key: [i64; 2], row: [i64; 2], col: [i64; 2]) -> [i64; 2] {
        std::array::from_fn(|i| element(key[i], row[i], col[i]))
    }
}

/// The slow lattice: the Hermite basis of the slow remainders.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    /// b_i (rad/µs): the slow basis' frequencies; M = b.len().
    pub slow: Vec<f64>,
    /// (φ, Φ_tw, Φ_ν) (rad/µs).
    pub fast: [f64; 3],
    /// The common denominator L of the remainders, the basis rows (integers over the primitives, × L) and their pivots.
    scale: i64,
    basis: Vec<Vec<i64>>,
    pivots: Vec<usize>,
}

/// Integer row reduction to echelon form (Euclid per pivot column): the nonzero rows, a Z-basis of the module the rows
/// generate, and their pivot columns (mdlat.hermite).
#[cfg(test)]
fn hermite(rows: &[Vec<i64>]) -> (Vec<Vec<i64>>, Vec<usize>) {
    let ncol = rows.first().map_or(0, Vec::len);
    let mut rest: Vec<Vec<i64>> = rows.iter().filter(|r| r.iter().any(|&x| x != 0)).cloned().collect();
    let (mut basis, mut pivots) = (Vec::new(), Vec::new());
    for c in 0..ncol {
        let (mut nz, zero): (Vec<Vec<i64>>, Vec<Vec<i64>>) = rest.into_iter().partition(|r| r[c] != 0);
        rest = zero;
        while nz.len() > 1 {
            nz.sort_by_key(|r| r[c].abs());
            let p = nz[0].clone();
            let mut next = Vec::new();
            for r in &nz[1..] {
                let k = floor_div(r[c], p[c]);
                let r2: Vec<i64> = r.iter().zip(&p).map(|(x, y)| x - k * y).collect();
                if r2[c] != 0 {
                    next.push(r2);
                } else if r2.iter().any(|&x| x != 0) {
                    rest.push(r2);
                }
            }
            nz = std::iter::once(p).chain(next).collect();
        }
        if let Some(mut p) = nz.pop() {
            if p[c] < 0 {
                p.iter_mut().for_each(|x| *x = -*x);
            }
            basis.push(p);
            pivots.push(c);
        }
        rest.retain(|r| r.iter().any(|&x| x != 0));
    }
    (basis, pivots)
}

#[cfg(test)]
impl Lattice {
    /// The lattice generated by the slow remainders of `vectors` (every Liouville component's and readout row's).
    pub fn new(frame: &Frame, vectors: &[FrameKey]) -> Lattice {
        let mut slows: Vec<Vec<Ratio>> = vectors.iter().map(|v| frame.split(v).1).collect();
        slows.sort();
        slows.dedup();
        let scale = slows.iter().flatten().fold(1i64, |l, c| l / gcd(l, c.den) * c.den);
        let rows: Vec<Vec<i64>> = slows.iter().map(|s| s.iter().map(|c| c.num * (scale / c.den)).collect()).collect();
        let (basis, pivots) = hermite(&rows);
        let slow = basis.iter().map(|row| frame.value(&row.iter().map(|&x| Ratio::new(x, scale)).collect::<Vec<_>>())).collect();
        Lattice { slow, fast: frame.fast, scale, basis, pivots }
    }

    /// A frame vector's lattice key (m, n), exact.
    pub fn key(&self, frame: &Frame, vector: &FrameKey) -> (Vec<i64>, [i64; 3]) {
        let (n, s) = frame.split(vector);
        let mut r: Vec<i64> = s.iter().map(|c| c.num * (self.scale / c.den)).collect();
        let mut m = Vec::with_capacity(self.basis.len());
        for (row, &c) in self.basis.iter().zip(&self.pivots) {
            assert_eq!(r[c] % row[c], 0, "a lattice vector");
            let k = r[c] / row[c];
            m.push(k);
            r.iter_mut().zip(row).for_each(|(x, y)| *x -= k * y);
        }
        assert!(r.iter().all(|&x| x == 0), "a lattice vector");
        (m, n)
    }

    /// The frequency m·b + n·(φ, Φ_tw, Φ_ν) (rad/µs).
    pub fn frequency(&self, m: &[i64], n: &[i64; 3]) -> f64 {
        m.iter().zip(&self.slow).map(|(&k, b)| k as f64 * b).sum::<f64>() + n.iter().zip(&self.fast).map(|(&k, f)| k as f64 * f).sum::<f64>()
    }
}
