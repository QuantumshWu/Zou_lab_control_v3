//! The elimination with the levels at the atom (SPEC §1.4): second-order Reiter–Sørensen with Brillouin–Wigner dressing and
//! first-order Doppler, for the drives folded at that level (model.rs `Level::Atom`), evaluated exactly at a point with its
//! first derivatives -- the definition the tables interpolate (SPEC §2.3).
//!
//! # The response
//! A carrier c (`Carrier`: a co-rotating component with a drive not bare) drives V_c(r) = Σ_l a_l(r) X_l e^{−iδ_c t}; V^f here
//! is the part of it folded with the levels at the atom -- its bare drives' part is a constant (far.rs), its explicit drives
//! are the generator's. The excited amplitude it drives from a ground state b of the cluster follows adiabatically,
//! −R[:,b] x_b, with (internal.md 1.2)
//!
//! ```text
//! R_c[:,b] = (K + Σ_b(z) − z)⁻¹ V_c^f[:,b]      z = E_b + δ_c + Σ_{d≠c} ΔE_b^d      K = H_ee − (i/2) D
//! ΔE_b^d = −Re[V_d^f[:,b]† (K − E_b − δ_d)⁻¹ V_d^f[:,b]]          Σ_b(z) = Σ_{d, a∉C} V_d^f[:,a] V_d^f[:,a]† / (z − Ẽ_a − δ_d)
//! ```
//!
//! H the constant forms' Hamiltonian at the point (far.rs: the base and every lit frequency's static form; a carrier's own
//! frequency's with the interference its dressing keeps, below: `Model::far_hamiltonian`), E_b its ground diagonal,
//! Ẽ_a = E_a + Σ_d ΔE_a^d. The other carriers dress b's energy (Brillouin–Wigner: in the denominator, to all orders); the folded
//! channels to the ground states of other clusters (a ∉ C) are the self-energy of the excited states (a strong beam's Autler–Townes
//! dressing of a probe through a folded Raman pair); the channels to the cluster's own states are its dynamics', none of Σ.
//! A kept drive's channel is in Σ as a folded one's is, with its explicit coupling for V_d^f: an excited state another
//! drive V makes explicit is dressed by it wherever b's channel is another Floquet state than that drive's own
//! (`Model::apart`: a in another cluster, or the two a residual beat apart in one cluster's frames) -- b's response through
//! it is U/(β̃ − |V|²/d_R), d_R = z − Ẽ_a − δ_d the two-photon distance of the Raman pair (far.rs, # A fold onto an
//! explicit state). Without it the folded channel's shift is wrong by |V|²/(β d_R) and its rate by twice that (three
//! levels, V = 2Γ, β = 27.4Γ, on resonance: −0.53 % and −1.06 %; with it 1.6e-4), without a bound at d_R → 0.
//! Within a carrier the dressing takes T̄_d, the beams' coherence kept within a polarization unit (beams of parallel fields:
//! a standing wave's intensity pattern) and averaged between units (SPEC §1.4): V_d V_d† → Σ_u v_u v_u†, v_u = Σ_{l∈u} a_l X^f_l,
//! and of its frequency's constant form in H the beam pairs of one unit alone -- everything its light does to the
//! denominators, its bare drives' shifts of the levels too, is one dressing; the units' interference is the dressing's
//! error, measured at the atom. Per beam the response is T_l, R_c = Σ_l a_l T_l.
//!
//! # First-order Doppler
//! Absorbing beam l raises the excited state's energy by w_l = (ħ/m)(k_l·κ + |k_l|²/2); to first order T_l → T_l − w_l T2_l,
//! T2 = (K + Σ − z)⁻¹ T (internal.md 1.4): in the generator and the photons alike. Its error is the second order,
//! (w_l ‖T2_l‖/‖T_l‖)² measured at the atom (units.rs); no clamp, no switch (SPEC §1.4).
//!
//! # The fold's terms: one contraction
//! A folded drive leaves the ground states two things, both made of its response (`shift`, `jump`): with an emitted leg
//! (coupling V', response R') and an absorbed one (V, R), the ground block −½(V'†R + R'†V) -- the light shifts, the Raman and
//! Zeeman couplings of the light, the beats of two components --, and the photons' jumps C_a R, whose own product is the decay
//! −(i/2)Σ J†J (photon.rs). A response is (K − z)⁻¹V with the levels it is taken at: the bare ones (K₀ = H0_ee − (i/2)D,
//! z = E_b + ν: a constant, far.rs), or those at the atom (`Model::eliminate`: the far light's H(r), the other carriers'
//! dressing, the folded channels' self-energy -- the bare folds' results are its inputs, so they are made first). One
//! definition for both: what differs is the response handed in.
//!
//! # The generator's terms at the atom (assemble.rs)
//! Per pair of frequencies (ω', ω) and of their beams (l', l): a_l'* a_l S_l'l on the cluster's ground states, S_l'l the
//! contraction of the two beams' legs -- each its frequency's every folded coupling, bare ones too, with its response at the
//! atom (none where the frequency is no carrier): what the bare drives' constants leave of the fold (far.rs) --, at the beat
//! δ_ω' − δ_ω and wave vector k_l − k_l'; an entry whose residual beat in the frames is not zero only where its unit
//! (`Unit::Term`) is kept. Per beam the jumps C_a T_l. The explicit states' energies take the van Vleck shift of the drives
//! folded here, ½ Σ_b (V^f[e,b] V[f,b]*/(E_e − E_b − δ_c) + V[e,b] V^f[f,b]*/(E_f − E_b − δ_c)), each drive's own denominator
//! against the other's folded coupling V. The explicit drives enter as they are: a_l X_l[e,b] e^{−iδ_c t} and its adjoint.
//!
//! # Derivatives
//! Everything above is a resolvent of the constant forms' H(r) and the fields: ∂_i T = −(K + Σ − z)⁻¹ ∂_i(K + Σ − z) T, with ∂_i K, ∂_i E_b
//! from the constant forms' exact derivatives (far.rs), ∂_i ΔE from the dressing's own resolvent, ∂_i Σ from the fields' and the
//! energies' -- the force's exact definition at the point.

use super::assemble::{NearValues, Plan};
use super::reach::{Frame, Region};
use super::units::{gate, Flow, Measured};
use super::{Beat, Component, Gate, Level, Lights, Model, Unit, UnitError};
use crate::field::Amplitude;
use crate::linalg::{dot3, Lu, Mat, C64};
use crate::request::{GROUND, STATES};
use super::far::EXCITED;
use super::jet;
use super::own::{Band, Form, Own};
use std::sync::OnceLock;

/// One leg of a second-order term (module documentation, # The fold's terms): a coupling's column from a ground state,
/// U[GROUND + e, `state`], and its response's, R[e, `place`] = ((K − z)⁻¹U)[e, `state`] -- a constant with the bare levels
/// (far.rs), or the elimination's at the atom (`Model::eliminate`, or its table).
#[derive(Clone, Copy)]
pub(crate) struct Leg<'a> {
    pub(crate) coupling: &'a Mat,
    pub(crate) state: usize,
    pub(crate) response: &'a Mat,
    pub(crate) place: usize,
}

/// −½ (U†R)[state, place] of a coupling's column and a response's: the half of a ground block's entry whose response is the
/// absorbed leg's.
pub(crate) fn half(coupling: &Mat, state: usize, response: &Mat, place: usize) -> C64 {
    (0..EXCITED).map(|e| coupling[(GROUND + e, state)].conj() * response[(e, place)]).sum::<C64>() * -0.5
}

/// The ground block's entry of an emitted leg and an absorbed one, −½(V'†R + R'†V) (the one definition, module documentation):
/// the absorbed response against the emitted coupling, and the emitted response against the absorbed coupling, conjugated.
pub(crate) fn shift(emitted: Leg, absorbed: Leg) -> C64 {
    half(emitted.coupling, emitted.state, absorbed.response, absorbed.place) + half(absorbed.coupling, absorbed.state, emitted.response, emitted.place).conj()
}

/// A jump's entry (C_a R)[b, place] of a response's column (the one definition).
pub(crate) fn jump(emission: &Mat, b: usize, response: &Mat, place: usize) -> C64 {
    (0..EXCITED).map(|e| emission[(b, GROUND + e)] * response[(e, place)]).sum()
}

/// A carrier: a co-rotating component with a drive not bare (derived from the drives' levels, model.rs `Level`; never listed).
/// Its beams' couplings; of its drives folded with the levels at the atom the couplings per beam and on its field's basis
/// -- the responses' sources --; and its polarization units (beams of parallel fields).
#[derive(Clone, Debug)]
pub(crate) struct Carrier {
    pub(crate) frequency: usize,
    /// δ_c exactly and its value (rad/µs).
    pub(crate) beat: Beat,
    pub(crate) detuning: f64,
    /// Its beams (request indices), their couplings X_l and those of the drives folded at the atom X^f_l (STATES × STATES), their
    /// units.
    pub(crate) beams: Vec<usize>,
    pub(crate) couplings: Vec<Mat>,
    pub(crate) folded: Vec<Mat>,
    /// The same on its field's basis (field.rs), U^f_k (STATES × STATES): X^f_l = Σ_k γ_lk U^f_k, the responses' sources.
    pub(crate) basis: Vec<Mat>,
    pub(crate) units: Vec<usize>,
    /// Its explicit drives' couplings per beam (STATES × STATES), and per ground state whether it has one from it: the
    /// kept channels of the excited states' self-energy (module documentation, # The response).
    pub(crate) explicit: Vec<Mat>,
    pub(crate) kept: [bool; GROUND],
}

impl Carrier {
    /// The carriers of a model: every co-rotating component with a drive folded at the atom or explicit. Units: beams whose
    /// focus fields are parallel (field.rs `parallel`).
    pub(crate) fn all(model: &Model) -> Vec<Carrier> {
        let mut out = vec![];
        for (frequency, tone) in model.tones.iter().enumerate() {
            let component = Component { frequency, counter: false };
            if !model.driven.iter().any(|&(c, e, g)| c == component && model.level(c, e, g) != Level::Bare) {
                continue;
            }
            let f = &model.light.frequencies[frequency];
            let at_level = |m: &Mat, level: Level| Mat::from_fn(STATES, STATES, |e, g| if e >= GROUND && g < GROUND && model.level(component, e, g) == level { m[(e, g)] } else { C64::ZERO });
            let at_the_atom = |m: &Mat| at_level(m, Level::Atom);
            let explicit: Vec<Mat> = tone.couplings[0].iter().map(|m| at_level(m, Level::Explicit)).collect();
            let mut units: Vec<usize> = vec![];
            for (j, e) in f.fields.iter().enumerate() {
                let unit = (0..j).find(|&i| crate::field::parallel(&f.fields[i], e)).map_or(j, |i| units[i]);
                units.push(unit);
            }
            out.push(Carrier {
                frequency,
                beat: tone.beat.clone(),
                detuning: tone.detuning,
                beams: f.beams.clone(),
                couplings: tone.couplings[0].clone(),
                folded: tone.couplings[0].iter().map(at_the_atom).collect(),
                basis: tone.basis[0].iter().map(at_the_atom).collect(),
                units,
                kept: std::array::from_fn(|g| explicit.iter().any(|m| (GROUND..STATES).any(|e| m[(e, g)] != C64::ZERO))),
                explicit,
            });
        }
        out
    }
}

/// The elimination's responses at one constant Hamiltonian and the carriers' amplitudes (`Model::eliminate`): per carrier (empty where unlit), basis
/// vector k of its field and jet part, T_k = (K + Σ − z)⁻¹ U^f_k and T2_k = (K + Σ − z)⁻¹ T_k on the cluster's ground places
/// (EXCITED × n_g) -- a beam's are T_l = Σ_k γ_lk T_k, the responses being linear in the coupling --; and the complex multiply-adds
/// that made them.
pub(crate) struct Responses {
    pub(crate) first: Vec<Vec<Vec<Mat>>>,
    pub(crate) second: Vec<Vec<Vec<Mat>>>,
    pub(crate) work: f64,
}

impl Model {
    /// Per frequency, row-major over its beams, whether a pair of them enters the levels the eliminations take (module
    /// documentation, # The response): a pair the lights hold (model.rs `Lights::pair`), and of a carrier's frequency one
    /// whose interference its dressing keeps -- two beams of one polarization unit, or any two where its dressing is coherent.
    pub(crate) fn level_pairs(&self, lights: &Lights, coherent: &[bool]) -> Vec<Vec<bool>> {
        (0..self.light.frequencies.len()).map(|w| (0..self.light.frequencies[w].beams.len().pow(2)).map(|x| self.level_pair(lights, coherent, w, x)).collect()).collect()
    }

    /// Whether pair x of frequency w's beams (row-major) enters the levels (`level_pairs`).
    fn level_pair(&self, lights: &Lights, coherent: &[bool], w: usize, x: usize) -> bool {
        let n = self.light.frequencies[w].beams.len();
        let carrier = self.carrier_of(w).filter(|&d| !coherent[d]).map(|d| &self.carriers[d]);
        lights.pair(w, x / n, x % n) && carrier.is_none_or(|cc| cc.units[x / n] == cc.units[x % n])
    }

    /// The levels at the atom as the eliminations take them: H(r) on all the states, its jet to `order` -- the base and every
    /// lit frequency's static form, a carrier's frequency's as its dressing takes its light (`level_pairs`).
    pub(crate) fn far_hamiltonian(&self, amplitudes: &[Amplitude], lights: &Lights, order: usize, coherent: &[bool]) -> Vec<Mat> {
        let mut h: Vec<Mat> = (0..jet::parts(order)).map(|p| if p == 0 { self.base.clone() } else { Mat::zeros(STATES, STATES) }).collect();
        let pairs = self.level_pairs(lights, coherent);
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            let n = f.beams.len();
            for (x, _) in pairs[w].iter().enumerate().filter(|of| *of.1) {
                let c = jet::of(&super::far::pair(&amplitudes[f.beams[x / n]], &amplitudes[f.beams[x % n]]), order);
                jet::add_scaled(&mut h, &c, &tone.pairs[x]);
            }
        }
        h
    }

    /// The same levels' energies, H(r)'s diagonal to `order`, into `e` ([part][state]: its caller's vectors, kept from one
    /// point to the next and written whole here), and their complex multiply-adds: the base's and each beam pair's
    /// coefficient times its form's diagonal, STATES per part.
    pub(crate) fn far_energies(&self, amplitudes: &[Amplitude], lights: &Lights, order: usize, coherent: &[bool], e: &mut Vec<Vec<f64>>) -> f64 {
        let np = jet::parts(order);
        e.resize_with(np, Vec::new);
        for (p, part) in e.iter_mut().enumerate() {
            part.clear();
            part.extend((0..STATES).map(|s| if p == 0 { self.base[(s, s)].re } else { 0.0 }));
        }
        let mut work = 0.0;
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            let n = f.beams.len();
            for x in (0..n * n).filter(|&x| self.level_pair(lights, coherent, w, x)) {
                let c = jet::whole(&super::far::pair(&amplitudes[f.beams[x / n]], &amplitudes[f.beams[x % n]]));
                for (part, &z) in e.iter_mut().zip(&c) {
                    for (s, v) in part.iter_mut().enumerate() {
                        *v += (z * tone.pairs[x][(s, s)]).re;
                    }
                }
                work += (np * STATES) as f64;
            }
        }
        work
    }

    /// The complex multiply-adds of `far_hamiltonian`: each of its beam pairs' form added to every part of the jet, STATES² each.
    pub(crate) fn far_work(&self, lights: &Lights, order: usize, coherent: &[bool]) -> f64 {
        let pairs: usize = self.level_pairs(lights, coherent).iter().map(|of| of.iter().filter(|pair| **pair).count()).sum();
        (pairs * jet::parts(order) * STATES * STATES) as f64
    }

    /// Per carrier whether its dressing takes its coherent field (`Unit::Dressing` kept) rather than its units' T̄.
    pub(crate) fn coherent_dressing(&self) -> Vec<bool> {
        (0..self.carriers.len()).map(|d| self.dressed_coherently(d)).collect()
    }

    /// Whether one carrier's dressing takes its coherent field (`coherent_dressing`, carrier by carrier: what a caller
    /// that keeps its vector fills it from).
    pub(crate) fn dressed_coherently(&self, carrier: usize) -> bool {
        self.kept.contains(&Unit::Dressing { carrier: Component { frequency: self.carriers[carrier].frequency, counter: false } })
    }

    /// A carrier's beam groups whose interference its dressing keeps: its units (beams of parallel fields), all its beams where its
    /// dressing is coherent, each beam alone where its light is averaged.
    fn dressing_groups(&self, carrier: &Carrier, lights: &Lights, coherent: bool) -> Vec<Vec<usize>> {
        let n = carrier.beams.len();
        if lights.0[carrier.frequency] == super::Lit::Averaged {
            return (0..n).map(|l| vec![l]).collect();
        }
        if coherent {
            return vec![(0..n).collect()];
        }
        let mut units = carrier.units.clone();
        units.sort();
        units.dedup();
        units.iter().map(|&u| (0..n).filter(|&l| carrier.units[l] == u).collect()).collect()
    }

    /// The near light's elimination at a far Hamiltonian `h` (its jet) and the near beams' amplitudes (module documentation): per
    /// lit carrier, basis vector of its field and cluster ground place its responses T_k and T2_k (`Responses`) -- every carrier's
    /// dressing of the ground states, the self-energy of the folded channels to the other clusters' ground states, one factorization
    /// per ground place. Of the ground places `places` takes: a place's responses are its own factorization's, whatever
    /// the others' (one not taken is left zero).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eliminate(&self, cluster: usize, h: &[Mat], amplitudes: &[Amplitude], lights: &Lights, order: usize, coherent: &[bool], places: &dyn Fn(usize) -> bool) -> Responses {
        let np = jet::parts(order);
        let c = &self.clusters[cluster];
        let g = c.ground;
        let mut work = 0.0;
        let excited: Vec<usize> = (GROUND..STATES).collect();
        let k: Vec<Mat> = h
            .iter()
            .enumerate()
            .map(|(p, m)| {
                let mut x = m.block(&excited, &excited);
                if p == 0 {
                    x.add_scaled(C64::new(0.0, -0.5), &self.decay);
                }
                x
            })
            .collect();
        let energy = |b: usize| -> Vec<f64> { h.iter().map(|m| m[(b, b)].re).collect() };
        // A matrix jet less a real jet on its diagonal.
        let shifted = |m: &[Mat], z: &[f64]| -> Vec<Mat> {
            m.iter()
                .zip(z)
                .map(|(x, &d)| {
                    let mut y = x.clone();
                    for e in 0..EXCITED {
                        y[(e, e)] -= C64::real(d);
                    }
                    y
                })
                .collect()
        };
        let constant = |v: Vec<C64>| -> Vec<Vec<C64>> { (0..np).map(|p| if p == 0 { v.clone() } else { vec![C64::ZERO; v.len()] }).collect() };
        let column = |m: &Mat, b: usize| -> Vec<C64> { (0..EXCITED).map(|x| m[(GROUND + x, b)]).collect() };
        let lit: Vec<bool> = self.carriers.iter().map(|x| lights.on(x.frequency)).collect();
        let coefficient = |l: usize| jet::of(&amplitudes[l], order);
        // The dressing ΔE_b^d of every ground state by every lit carrier, its jet.
        let mut shift = vec![vec![vec![0.0_f64; np]; GROUND]; self.carriers.len()];
        for (d, carrier) in self.carriers.iter().enumerate() {
            if !lit[d] {
                continue;
            }
            let groups = self.dressing_groups(carrier, lights, coherent[d]);
            for b in 0..GROUND {
                let mut z = energy(b);
                z[0] += carrier.detuning;
                let m = shifted(&k, &z);
                let lu = Lu::new(&m[0]).expect("K − z regular: the decay keeps K's spectrum off the real axis");
                let ys: Vec<Vec<Vec<C64>>> = (0..carrier.beams.len()).map(|l| jet::solve(&lu, &m, &constant(column(&carrier.folded[l], b)))).collect();
                work += jet::factor_work(EXCITED) + carrier.beams.len() as f64 * jet::solve_work(EXCITED, np);
                for group in &groups {
                    for &j in group {
                        for &i in group {
                            work += (EXCITED * np) as f64 + 2.0 * jet::product_work(np);
                            let pair = jet::of(&super::far::pair(&amplitudes[carrier.beams[j]], &amplitudes[carrier.beams[i]]), order);
                            let xj = column(&carrier.folded[j], b);
                            let form: Vec<C64> = ys[i].iter().map(|y| crate::linalg::inner(&xj, y)).collect();
                            for (s, x) in shift[d][b].iter_mut().zip(jet::product(&pair, &form)) {
                                *s -= x.re;
                            }
                        }
                    }
                }
            }
        }
        let dressed = |a: usize, except: Option<usize>| -> Vec<f64> {
            let mut e = energy(a);
            for (d, s) in shift.iter().enumerate() {
                if Some(d) != except {
                    for (x, y) in e.iter_mut().zip(&s[a]) {
                        *x += y;
                    }
                }
            }
            e
        };
        let (mut first, mut second) = (vec![], vec![]);
        for (ci, carrier) in self.carriers.iter().enumerate() {
            let n = carrier.basis.len();
            if !lit[ci] {
                first.push(vec![]);
                second.push(vec![]);
                continue;
            }
            let zeros = || -> Vec<Vec<Mat>> { (0..n).map(|_| (0..np).map(|_| Mat::zeros(EXCITED, g)).collect()).collect() };
            let (mut out, mut out2) = (zeros(), zeros());
            for (pb, &b) in c.states[..g].iter().enumerate().filter(|of| places(of.0)) {
                let mut z = dressed(b, Some(ci));
                z[0] += carrier.detuning;
                let mut m = shifted(&k, &z);
                // The self-energy of the folded channels to the other clusters' ground states, and of the kept ones that are
                // another Floquet state than this response's (module documentation, # The response).
                for (d, other) in self.carriers.iter().enumerate() {
                    if !lit[d] {
                        continue;
                    }
                    for a in 0..GROUND {
                        let (folded, explicit) = (self.membership[a] != cluster, other.kept[a] && self.apart(b, &carrier.beat, a, &other.beat));
                        if !folded && !explicit {
                            continue;
                        }
                        let mut den: Vec<f64> = z.iter().zip(dressed(a, None)).map(|(x, y)| x - y).collect();
                        den[0] -= other.detuning;
                        let inverse = jet::reciprocal(&den);
                        for group in self.dressing_groups(other, lights, coherent[d]) {
                            let mut v: Vec<Vec<C64>> = vec![vec![C64::ZERO; EXCITED]; np];
                            work += (group.len() * EXCITED * np) as f64;
                            for &l in &group {
                                let a_l = coefficient(other.beams[l]);
                                for x in 0..EXCITED {
                                    let xv = if folded { other.folded[l][(GROUND + x, a)] } else { C64::ZERO } + if explicit { other.explicit[l][(GROUND + x, a)] } else { C64::ZERO };
                                    if xv != C64::ZERO {
                                        for p in 0..np {
                                            v[p][x] += a_l[p] * xv;
                                        }
                                    }
                                }
                            }
                            if v[0].iter().all(|z| *z == C64::ZERO) {
                                continue;
                            }
                            work += (EXCITED * EXCITED) as f64 * (2.0 * jet::product_work(np) + np as f64);
                            // The excited states the channel reaches: the others' rows and columns of v v† are zero.
                            let reached: Vec<usize> = (0..EXCITED).filter(|&x| v.iter().any(|part| part[x] != C64::ZERO)).collect();
                            let weight: [C64; jet::WHOLE] = std::array::from_fn(|p| inverse.get(p).map_or(C64::ZERO, |&r| C64::real(r)));
                            let of = |x: usize, conjugate: bool| -> [C64; jet::WHOLE] { std::array::from_fn(|p| v.get(p).map_or(C64::ZERO, |part| if conjugate { part[x].conj() } else { part[x] })) };
                            for &x in &reached {
                                let vx = of(x, false);
                                for &y in &reached {
                                    let (mut outer, mut term) = ([C64::ZERO; jet::WHOLE], [C64::ZERO; jet::WHOLE]);
                                    jet::multiply(&vx[..np], &of(y, true)[..np], &mut outer[..np]);
                                    jet::multiply(&outer[..np], &weight[..np], &mut term[..np]);
                                    for p in 0..np {
                                        m[p][(x, y)] += term[p];
                                    }
                                }
                            }
                        }
                    }
                }
                let lu = Lu::new(&m[0]).expect("K + Σ − z regular: the decay keeps it off the real axis");
                work += jet::factor_work(EXCITED);
                for l in 0..n {
                    let x = column(&carrier.basis[l], b);
                    if x.iter().all(|v| *v == C64::ZERO) {
                        continue;
                    }
                    work += 2.0 * jet::solve_work(EXCITED, np);
                    let t = jet::solve(&lu, &m, &constant(x));
                    let t2 = jet::solve(&lu, &m, &t);
                    for p in 0..np {
                        for e in 0..EXCITED {
                            out[l][p][(e, pb)] = t[p][e];
                            out2[l][p][(e, pb)] = t2[p][e];
                        }
                    }
                }
            }
            first.push(out);
            second.push(out2);
        }
        Responses { first, second, work }
    }

    /// Whether the channel of a component at the exact frequency `to` into ground state a is another Floquet state than
    /// the one a component at `from` drives ground state b from (module documentation, # The response): the two states in
    /// different clusters -- no frame joins them --, or in one cluster's frames a residual beat θ_b + ν − θ_a − ν' that is
    /// not exactly zero apart. Where it is zero the channel is the cluster's own dynamics.
    pub(crate) fn apart(&self, b: usize, from: &Beat, a: usize, to: &Beat) -> bool {
        let c = &self.clusters[self.membership[b]];
        match (c.states.iter().position(|&s| s == b), c.states.iter().position(|&s| s == a)) {
            (Some(pb), Some(pa)) => c.frames[pb].plus(1, from) != c.frames[pa].plus(1, to),
            _ => true,
        }
    }

    /// The legs of the Raman pairs that a component's drive e ← b makes folded onto e (far.rs, # A fold onto an explicit
    /// state): e's kept drives (`legs`) whose channel is another Floquet state than the drive's own (`apart`) -- a fact
    /// of the model's clusters and frames, the same at every state, made once for the errors read at thousands
    /// (`pairs_made`).
    pub(crate) fn pair_legs(&self, component: Component, excited: usize, ground: usize) -> &[(usize, usize)] {
        &self.pairs[((2 * component.frequency + component.counter as usize) * EXCITED + excited - GROUND) * GROUND + ground]
    }

    /// Every drive's pairs' legs (`pair_legs`), by component (a frequency's co-rotating one, then its counter-rotating
    /// one), excited state and ground state.
    pub(crate) fn pairs_made(&self) -> Vec<Vec<(usize, usize)>> {
        let components = (0..self.light.frequencies.len()).flat_map(|frequency| [false, true].map(|counter| Component { frequency, counter }));
        components
            .flat_map(|component| {
                let from = self.frequency(component);
                (0..EXCITED * GROUND).map(move |at| (from.clone(), at / GROUND, at % GROUND))
            })
            .map(|(from, x, ground)| self.legs[x].iter().copied().filter(|&(d, a)| self.apart(ground, &from, a, &self.carriers[d].beat)).collect())
            .collect()
    }

    /// Per excited state (less GROUND) its kept drives: (carrier, ground state) of each explicit drive into it -- the
    /// legs of the Raman pairs a drive folded onto it makes (far.rs, # A fold onto an explicit state).
    pub(crate) fn legs_made(&self) -> Vec<Vec<(usize, usize)>> {
        (GROUND..STATES)
            .map(|e| {
                let kept = self.carriers.iter().enumerate().flat_map(|(d, cc)| (0..GROUND).filter(move |&a| cc.explicit.iter().any(|m| m[(e, a)] != C64::ZERO)).map(move |a| (d, a)));
                kept.collect()
            })
            .collect()
    }

    /// A frequency's carrier, where its co-rotating component is one.
    pub(crate) fn carrier_of(&self, frequency: usize) -> Option<usize> {
        self.carriers.iter().position(|c| c.frequency == frequency)
    }

    /// Whether a coupling of two places of a cluster (x ← y at the base beat `beat`) is in the cluster's generator: at a
    /// residual beat of exactly zero in the frames (free), or as a kept unit (`Unit::Term`) -- one rule for every second-order
    /// coupling, whatever light makes it.
    pub(crate) fn term_included(&self, cluster: usize, x: usize, y: usize, beat: &Beat) -> bool {
        let c = &self.clusters[cluster];
        beat.plus(1, &c.frames[x]).plus(-1, &c.frames[y]).is_zero() || self.kept.contains(&term_unit(c.states[x], c.states[y], beat))
    }

    /// The light-shift bound per beam pair of the trap frequency's drives folded with the levels at the atom (potential.rs
    /// `Model::envelope`), row-major over its beams: |[X^f_l'† G X^f_l]_p| with ‖G‖ ≤ 2/Γ_min (a resolvent of an excited block
    /// whose decay is at least Γ_min/2 off the real axis); zero where it has none.
    pub(crate) fn near_envelope(&self, p: &[f64; GROUND]) -> Vec<f64> {
        let n = self.light.frequencies[self.light.trap].beams.len();
        let Some(cc) = self.carriers.iter().find(|c| c.frequency == self.light.trap) else { return vec![0.0; n * n] };
        let gamma = self.least_decay();
        (0..n * n)
            .map(|x| {
                let (j, i) = (x / n, x % n);
                (0..GROUND)
                    .filter(|&b| p[b] != 0.0)
                    .map(|b| {
                        let (a, c) = ((0..EXCITED).map(|e| cc.folded[j][(GROUND + e, b)].norm_sqr()).sum::<f64>().sqrt(), (0..EXCITED).map(|e| cc.folded[i][(GROUND + e, b)].norm_sqr()).sum::<f64>().sqrt());
                        p[b] * a * c * 2.0 / gamma
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    /// Per carrier (None where unlit) its folded response on a plan's cluster at a point and momentum, R = Σ_l a_l (T_l − w_l T2_l)
    /// = Σ_k (e_k T_k − e^w_k T2_k) on its field's basis (EXCITED × the ground places), e^w_k = Σ_l w_l a_l γ_lk its Doppler-weighted
    /// field (module documentation, # First-order Doppler), from the near forms' responses.
    pub(crate) fn carrier_responses(&self, plan: &Plan, near: &NearValues, amplitudes: &[Amplitude], kappa: [f64; 3]) -> Vec<Option<Mat>> {
        let g = self.clusters[plan.cluster].ground;
        let start = plan.near.responding().start;
        let mut out: Vec<Option<Mat>> = self.carriers.iter().map(|cc| plan.lights.on(cc.frequency).then(|| Mat::zeros(EXCITED, g))).collect();
        self.responses_each(plan, |i| near.data[start + i], amplitudes, kappa, &mut vec![], |c, e, y, z| out[c].as_mut().expect("a lit carrier's response")[(e, y)] += z);
        out
    }

    /// The terms of the carriers' folded responses (`carrier_responses`), one a response of the plan's near forms, each to
    /// `add` as it is made -- its carrier, its excited state (less GROUND), its ground place and its value
    /// e_k T_k − e^w_k T2_k --, from the forms' responding section `near` (its entry i: the T's, then the T2's). `fields` is
    /// its caller's vector, kept from one point to the next and written whole here before it is read: per carrier and
    /// basis vector its field's component, plain and Doppler-weighted, as many a carrier as the widest basis holds.
    pub(crate) fn responses_each(&self, plan: &Plan, near: impl Fn(usize) -> C64, amplitudes: &[Amplitude], kappa: [f64; 3], fields: &mut Vec<(C64, C64)>, mut add: impl FnMut(usize, usize, usize, C64)) {
        let count = plan.near.responses.len();
        let widest = self.carriers.iter().map(|cc| cc.basis.len()).max().unwrap_or(0);
        fields.clear();
        for cc in &self.carriers {
            let f = &self.light.frequencies[cc.frequency];
            fields.extend((0..widest).map(|k| {
                f.beams.iter().zip(&f.coefficients).filter(|_| k < cc.basis.len()).fold((C64::ZERO, C64::ZERO), |(plain, weighted), (&l, c)| {
                    let z = amplitudes[l].value * c[k];
                    (plain + z, weighted + z * self.doppler_shift(l, kappa))
                })
            }));
        }
        for (i, &[c, k, e, y]) in plan.near.responses.iter().enumerate() {
            let (plain, weighted) = fields[c * widest + k];
            add(c, e, y, plain * near(i) - weighted * near(count + i));
        }
    }

    /// The first-order Doppler term's relative size on each ground place y of a plan's cluster (its error's square root,
    /// units.rs): max_l |w_l| ‖T2_l[:, y]‖ / max_l ‖T_l[:, y]‖ over the carrier's beams, T_l = Σ_k γ_lk T_k.
    pub(crate) fn doppler_sizes(&self, plan: &Plan, near: &NearValues, carrier: usize, kappa: [f64; 3]) -> Vec<f64> {
        let f = &self.light.frequencies[self.carriers[carrier].frequency];
        let g = self.clusters[plan.cluster].ground;
        let (start, count) = (plan.near.responding().start, plan.near.responses.len());
        let (mut first, mut second) = (vec![0.0_f64; g], vec![0.0_f64; g]);
        for (&l, c) in f.beams.iter().zip(&f.coefficients) {
            let (mut t, mut t2) = (vec![[C64::ZERO; EXCITED]; g], vec![[C64::ZERO; EXCITED]; g]);
            for (i, &[of, k, e, y]) in plan.near.responses.iter().enumerate() {
                if of == carrier {
                    t[y][e] += c[k] * near.data[start + i];
                    t2[y][e] += c[k] * near.data[start + count + i];
                }
            }
            let shift = self.doppler_shift(l, kappa).abs();
            for y in 0..g {
                first[y] = first[y].max(crate::linalg::norm_sqr(&t[y]).sqrt());
                second[y] = second[y].max(shift * crate::linalg::norm_sqr(&t2[y]).sqrt());
            }
        }
        first.iter().zip(&second).map(|(&t, &t2)| if t > 0.0 { t2 / t } else { 0.0 }).collect()
    }

    /// A carrier's dressing's error at a sample: every carrier's folded response there with its coherent field in the dressing, R'
    /// -- the exact elimination with that carrier's dressing the other way, the one elimination a unit's error takes of its own --
    /// against the model's with its units' T̄, R, on the sample's ground amplitudes ψ_g -- the other carriers' through its shift of
    /// their denominators, its own through its channels' self-energy and the dressed energies of the states they lead to --:
    /// δ = max_c ‖(R'_c − R_c)ψ_g‖/‖R_c ψ_g‖, the relative error of the rates and light shifts it moves of the sample's state, zero
    /// where nothing drives it; on the populations and on T δ P_g, P_g the sample's ground population (SPEC §1.5's map of a
    /// relative rate error, as a drive's S_b p_b).
    pub(crate) fn dressing_error(&self, carrier: Component, at: &Measured) -> UnitError {
        let mut toggled = self.coherent_dressing();
        let Some(d) = self.carrier_of(carrier.frequency) else { return UnitError::NONE };
        toggled[d] = !toggled[d];
        let other = self.carrier_responses(&at.plan, &self.near_exact(&at.plan, &at.amplitudes, 0, &toggled), &at.amplitudes, at.sample.at.kappa);
        let ground = &at.sample.psi[..self.clusters[at.sample.cluster].ground];
        let mut delta = 0.0_f64;
        for (r, s) in at.responses.iter().zip(&other) {
            let (Some(r), Some(s)) = (r, s) else { continue };
            let (r, s) = (r.apply(ground), s.apply(ground));
            let size: f64 = r.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if size > 0.0 {
                let diff: f64 = r.iter().zip(&s).map(|(&x, &y)| (y - x).norm_sqr()).sum::<f64>().sqrt();
                delta = delta.max(diff / size);
            }
        }
        let weight: f64 = at.populations[..GROUND].iter().sum();
        UnitError { parameter: delta, population: delta * weight, temperature: delta * weight, ..UnitError::NONE }
    }

}

/// A drive folded with the levels at the atom, at a sample (`Model::drive_parameters`): its own saturation s; its
/// denominator's modulus; its sweep a; its Doppler order's (w‖T2‖/‖T‖)²; its ground state's population; of that state
/// under the drive's carrier (units.rs `Fold`) the excited share S_b, and how far beyond it the drive reads the level's
/// own second-order term, L_e/S_b where that is above one (`Model::drive_parameters`); and its response R_c[e, b].
pub(crate) struct Folded {
    pub(crate) share: f64,
    pub(crate) distance: f64,
    pub(crate) sweep: f64,
    pub(crate) doppler: f64,
    pub(crate) population: f64,
    pub(crate) total: f64,
    pub(crate) reads: f64,
    pub(crate) response: C64,
}

impl Folded {
    /// The drive's part of what its fold's order leaves of its ground state (units.rs, # The fold's order): its line errs
    /// by M^k -- M = max(S_b, L_e), the larger of the state's excited share and of twice its level's own second-order
    /// term across the drive's distance, both s on one line; k = 1 at the second order, 2 at the fourth -- on the
    /// drive's part s/S_b of the state's excitation: s M^k/S_b = s S_b^{k−1} (M/S_b)^k.
    pub(crate) fn order(&self, k: i32) -> f64 {
        self.share * self.total.powi(k - 1) * self.reads.powi(k)
    }
}

/// The difference E_x − E_y of two levels of the constant forms over a region (`Model::gap`; the gates' distances), in its own
/// form (own.rs): of every beam pair, the levels themselves, and of the pairs the eliminations' levels hold (module
/// documentation, # The response: a carrier's frequency's form with its polarization units' interference averaged) -- the
/// levels the eliminations take, whose difference a drive's denominator at the atom and a self-energy's channel read. Its
/// gradient's form is made where a rate asks for it (`Model::gap_slope`): most differences a cell is asked for are
/// distances alone (a resolvent's, a self-energy's channel), and the Hessians the gradient's first order is made of were
/// most of what a cell's distances cost.
///
/// Of the difference itself, never of each level: a shift the two levels share moves no distance and no rate. (Read as each
/// level's own slack, the tweezer's scalar light shift -- the same on every Zeeman sublevel of a ground manifold, 236 rad/µs a
/// micrometre across its waist -- moved the 13 rad/µs between the user's F=1 m=0 and m=−1 by twice that: the self-energy's
/// two-photon distance between them read zero on every cell wider than 0.03 µm across the beam, and the gates' proofs over
/// the lin⊥lin scene's reach never ended. The difference's weights w = [P_l'l]_xx − [P_l'l]_yy hold nothing of a shift the
/// two share, at every order: its value, gradient and Hessian at the centre are the two levels' differences, its bounds
/// the beam pairs' bounds weighted by |w|.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gap {
    pub(crate) all: Own,
    pub(crate) taken: Own,
}

/// The constant forms' diagonal over a region (`Model::gap`): per state x, of the beam pairs the eliminations' levels hold and
/// of the others, Σ_p [P_p]_xx c_p's value and gradient at the centre, made once per region where a gap is first asked; and
/// its Hessian (on a cell alone), made where a gap's gradient is first asked.
pub(crate) struct Diagonal {
    sums: Vec<[PairSum; 2]>,
    curvatures: OnceLock<Vec<[[[C64; 3]; 3]; 2]>>,
}

/// A sum of beam pairs' coefficients at a cell's centre: its value and gradient.
#[derive(Clone, Copy)]
struct PairSum {
    value: C64,
    gradient: [C64; 3],
}

impl PairSum {
    const ZERO: PairSum = PairSum { value: C64::ZERO, gradient: [C64::ZERO; 3] };
}

/// A ground level's shift over a region with its sign (own.rs, # Bands): by its drives folded at the atom (units.rs
/// `Model::near_level`) or by a carrier's dressing (`Model::dressing_shifts`) -- a band, its smooth part with the parameters it
/// is not computed at (the drives' Doppler shifts, the excited states' couplings and self-energy) at their centre and how
/// far they move it; the smooth part's gradient (own.rs `Form`); and `swing`, how far the parameters move that gradient.
/// What a rate reads of a level is its gradient, of two levels' difference the gradient of the difference (`Shift::sub`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shift {
    pub(crate) band: Band,
    pub(crate) slope: Form<[f64; 3]>,
    pub(crate) swing: Own,
}

impl Shift {
    pub(crate) fn none() -> Shift {
        Shift { band: Band::exact(Own::constant(0.0)), slope: Form::zero(), swing: Own::constant(0.0) }
    }

    pub(crate) fn add(&self, o: &Shift) -> Shift {
        Shift { band: self.band.add(o.band), slope: self.slope.add(o.slope), swing: self.swing.add(o.swing) }
    }

    pub(crate) fn sub(&self, o: &Shift) -> Shift {
        Shift { band: self.band.sub(o.band), slope: self.slope.sub(o.slope), swing: self.swing.add(o.swing) }
    }

    /// The level's difference with another's: a gap (`Gap`) with this shift, less the other's -- of the dressed levels, one
    /// band and one gradient.
    pub(crate) fn of(level: Own, slope: Form<[f64; 3]>) -> Shift {
        Shift { band: Band::exact(level), slope, swing: Own::constant(0.0) }
    }

    /// The bound of its gradient's norm over the region and the parameters: |∇mid| + swing.
    pub(crate) fn steepness(&self) -> Own {
        self.slope.norm().add(self.swing)
    }
}

/// A carrier's coherent couplings over a region (reach.rs `Region`), each in its own form (own.rs). With u_s = Σ_l a_l X^f_l[:, s]
/// the excited amplitudes its field drives from ground state s -- its beams coherent at the request's phases (SPEC §1.2), what
/// the atom feels at r --, per ground state |u_s[e]| per excited state, ‖u_s‖, and ‖u_s‖ per polarization unit (its beams
/// alone) (reach.rs `Frame`). On a point, the values there.
pub(crate) struct Couplings {
    /// [s][e − GROUND].
    pub(crate) entry: [[Own; EXCITED]; GROUND],
    /// [s].
    pub(crate) column: [Own; GROUND],
    /// [unit][s], over the carrier's distinct polarization units.
    pub(crate) units: Vec<[Own; GROUND]>,
}

impl Model {
    /// The difference E_x − E_y − `at` of two levels of the constant forms over a region (`Gap`), every light on: Re Σ w c over
    /// the beam pairs' coefficients c = a_l'* a_l, w = [P_l'l]_xx − [P_l'l]_yy -- over every pair, and over the pairs the
    /// eliminations' levels hold (`Forms::taken`) --, in its own form: its value and gradient at the centre the two levels'
    /// (`Diagonal`), its remainder Σ_p |w_p| `curve`_p, its range Σ_p |w_p| `radius`_p about its value (reach.rs
    /// `Combination`). The bare levels' difference less `at` is one constant, taken first: a transition's 10⁹ rad/µs less
    /// its light's frequency is its detuning, and what the light does at a place is added to that -- added to the
    /// transition instead, each place's sum was rounded apart from the next one's by 10⁻⁷ rad/µs. Between a level and
    /// itself, the constant.
    pub(crate) fn gap(&self, x: usize, y: usize, region: &Region, at: f64) -> Gap {
        let constant = (self.base[(x, x)].re - at) - self.base[(y, y)].re;
        if x == y {
            let level = Own::constant(constant);
            return Gap { all: level, taken: level };
        }
        let made = region.gaps.lock().expect("the gaps' lock").get(&(x, y)).copied();
        let made = made.unwrap_or_else(|| {
            let made = self.gap_made(x, y, region);
            region.gaps.lock().expect("the gaps' lock").insert((x, y), made);
            made
        });
        Gap { all: made.all.plus(constant), taken: made.taken.plus(constant) }
    }

    /// Per part of a difference of two levels' weights (`gap`, `gap_slope`) -- the pairs the eliminations' levels hold, the
    /// others -- the sum over its beam pairs of |w_p| times their bounds `bound` reads.
    fn weighed<const N: usize>(&self, x: usize, y: usize, region: &Region, bound: impl Fn(&super::reach::PairSpan) -> [f64; N]) -> [[f64; N]; 2] {
        let taken = region.forms.taken(self);
        let mut sums = [[0.0; N]; 2];
        for (frequency, pair, w) in region.forms.diagonals(self) {
            let size = (w[x] - w[y]).abs();
            if size > 0.0 {
                let (sum, of) = (&mut sums[!taken[*frequency][*pair] as usize], bound(&region.pairs[*frequency][*pair]));
                for (total, part) in sum.iter_mut().zip(of) {
                    *total += size * part;
                }
            }
        }
        sums
    }

    /// The spatial part of a gap (`gap`), of the region's diagonal sums and the beam pairs' bounds.
    fn gap_made(&self, x: usize, y: usize, region: &Region) -> Gap {
        let diagonal = region.diagonal.get_or_init(|| self.diagonal(region));
        // Per part the weighted bounds: curve, radius.
        let bounds = self.weighed(x, y, region, |pair| [pair.curve, pair.radius]);
        let level = |of: &[usize]| -> Own {
            let (mut value, mut gradient, mut bound) = (0.0, [0.0; 3], [0.0; 2]);
            for &k in of {
                let (a, b) = (&diagonal.sums[x][k], &diagonal.sums[y][k]);
                value += (a.value - b.value).re;
                for i in 0..3 {
                    gradient[i] += (a.gradient[i] - b.gradient[i]).re;
                }
                bound = [bound[0] + bounds[k][0], bound[1] + bounds[k][1]];
            }
            Own::new(value, gradient, bound[0], region.half).within(value - bound[1], value + bound[1])
        };
        Gap { taken: level(&[0]), all: level(&[0, 1]) }
    }

    /// The gradient ∇(E_x − E_y) over a region in its own form (own.rs `Form`), of the levels the eliminations take
    /// (`taken`) or of every pair: its value and second derivatives at the centre the two levels' (`Diagonal`), its
    /// remainder Σ_p |w_p| `rest`_p, moving by Σ_p |w_p| `veer`_p, at most Σ_p |w_p| `slope`_p (reach.rs `Region::gradient`);
    /// on the whole space at most the last. Made once per region and difference where a rate first asks for it; nothing
    /// between a level and itself.
    pub(crate) fn gap_slope(&self, x: usize, y: usize, region: &Region, taken: bool) -> Form<[f64; 3]> {
        if x == y {
            return Form::zero();
        }
        let made = region.slopes.lock().expect("the slopes' lock").get(&(x, y)).copied();
        let made = made.unwrap_or_else(|| {
            let made = self.gap_slope_made(x, y, region);
            region.slopes.lock().expect("the slopes' lock").insert((x, y), made);
            made
        });
        made[!taken as usize]
    }

    /// A difference's gradients (`gap_slope`), [taken, all].
    fn gap_slope_made(&self, x: usize, y: usize, region: &Region) -> [Form<[f64; 3]>; 2] {
        let diagonal = region.diagonal.get_or_init(|| self.diagonal(region));
        let bounds = self.weighed(x, y, region, |pair| [pair.rest, pair.veer, pair.slope]);
        if !region.rho.is_finite() {
            return [bounds[0][2], bounds[0][2] + bounds[1][2]].map(Form::whole);
        }
        let curvatures = diagonal.curvatures.get_or_init(|| self.curvatures(region));
        let half = region.half;
        let slope = |of: &[usize]| -> Form<[f64; 3]> {
            let (mut value, mut hessian, mut bound) = ([0.0; 3], [[0.0; 3]; 3], [0.0; 3]);
            for &k in of {
                let (a, b) = (&diagonal.sums[x][k], &diagonal.sums[y][k]);
                for i in 0..3 {
                    value[i] += (a.gradient[i] - b.gradient[i]).re;
                    for j in 0..3 {
                        hessian[i][j] += (curvatures[x][k][i][j] - curvatures[y][k][i][j]).re;
                    }
                }
                bound = std::array::from_fn(|i| bound[i] + bounds[k][i]);
            }
            let [rest, veer, most] = bound;
            Form { value, first: std::array::from_fn(|i| hessian[i].map(|z| z * half[i])), rest, moved: veer, most }
        };
        [slope(&[0]), slope(&[0, 1])]
    }

    /// The region's diagonal sums per state (`Diagonal`): their values and gradients.
    fn diagonal(&self, region: &Region) -> Diagonal {
        let taken = region.forms.taken(self);
        let mut sums = vec![[PairSum::ZERO; 2]; STATES];
        for (frequency, pair, w) in region.forms.diagonals(self) {
            let p = &region.pairs[*frequency][*pair];
            let k = !taken[*frequency][*pair] as usize;
            for (s, &x) in w.iter().enumerate().filter(|of| *of.1 != C64::ZERO) {
                let sum = &mut sums[s][k];
                sum.value += p.centre * x;
                for i in 0..3 {
                    sum.gradient[i] += p.gradient[i] * x;
                }
            }
        }
        Diagonal { sums, curvatures: OnceLock::new() }
    }

    /// The diagonal sums' Hessians per state (`Diagonal`), on a cell; nothing on a point.
    fn curvatures(&self, region: &Region) -> Vec<[[[C64; 3]; 3]; 2]> {
        let taken = region.forms.taken(self);
        let mut out = vec![[[[C64::ZERO; 3]; 3]; 2]; STATES];
        if region.rho > 0.0 {
            for (frequency, pair, w) in region.forms.diagonals(self) {
                let p = &region.pairs[*frequency][*pair];
                let k = !taken[*frequency][*pair] as usize;
                for (s, &x) in w.iter().enumerate().filter(|of| *of.1 != C64::ZERO) {
                    for i in 0..3 {
                        for j in 0..3 {
                            out[s][k][i][j] += p.hessian[i][j] * x;
                        }
                    }
                }
            }
        }
        out
    }

    /// The Gershgorin radii over a region of the constant forms' couplings among the excited states (per state; nothing for a
    /// ground one), made once per region, every light on: an excited state's row over the states it is coupled to somewhere
    /// (the others' terms are zero), Σ_y |H_xy|, each entry's modulus in its own form (reach.rs `Region::modulus`) and the
    /// row their sum: the moduli's own first orders add with their signs, two entries that move apart moving the row less.
    pub(crate) fn radii<'r>(&self, region: &'r Region) -> &'r [Own; STATES] {
        region.radii.get_or_init(|| {
            let forms = &region.forms;
            let mut radius = [Own::constant(0.0); STATES];
            for (x, mixed) in (GROUND..STATES).zip(forms.mixed(self)) {
                for &(y, base) in mixed {
                    radius[x] = radius[x].add(region.modulus(forms.entry(self, x, y).iter().map(|&(frequency, pair, v, size)| (region.pairs[frequency][pair], v, size)), base));
                }
            }
            radius
        })
    }

    /// The carriers' coherent couplings over a region (`Couplings`), made once per region, their folded couplings from the region's
    /// `Forms`: of a beam its entries that are not zero (the others add nothing), an excited state no beam reaches nothing. Each
    /// norm of its own beams in its own frame (reach.rs `Frame`): a polarization unit's, the whole column's, an entry's.
    pub(crate) fn couplings<'r>(&self, region: &'r Region) -> &'r [Couplings] {
        region.couplings.get_or_init(|| {
            let none = Own::constant(0.0);
            self.carriers
                .iter()
                .zip(region.forms.folded(self))
                .map(|(cc, folded)| {
                    let mut entry = [[none; EXCITED]; GROUND];
                    let mut column = [none; GROUND];
                    let mut per_unit = vec![[none; GROUND]; folded.units.len()];
                    for s in 0..GROUND {
                        // The norm of a set of the carrier's beams (its place among them, its column's entries), ‖X^f_l[:, s]‖ each.
                        let norm = |beams: &mut dyn Iterator<Item = &(usize, Vec<(usize, C64, f64)>)>| -> Own {
                            let beams: Vec<&(usize, Vec<(usize, C64, f64)>)> = beams.collect();
                            let mut sum = Frame::<EXCITED>::of(region, beams.iter().map(|(l, _)| (cc.beams[*l], folded.columns[s][*l])));
                            for (l, entries) in beams {
                                sum.add(region, cc.beams[*l], folded.columns[s][*l], entries.iter().map(|&(e, v, _)| (e, v)));
                            }
                            sum.norm(region)
                        };
                        for (ui, beams) in folded.beams[s].iter().enumerate() {
                            per_unit[ui][s] = norm(&mut beams.iter());
                        }
                        column[s] = norm(&mut folded.beams[s].iter().flatten());
                        for &e in &folded.reached[s] {
                            let beams = folded.beams[s].iter().flatten().filter_map(|(l, entries)| entries.iter().find(|x| x.0 == e).map(|x| (cc.beams[*l], x.1, x.2)));
                            let beams: Vec<(usize, C64, f64)> = beams.collect();
                            entry[s][e] = region.coherent_own(&beams);
                        }
                    }
                    Couplings { entry, column, units: per_unit }
                })
                .collect()
        })
    }

    /// The least decay rate of the excited states, Γ_min: every folded response's resolvent is at most 2/Γ_min.
    pub(crate) fn least_decay(&self) -> f64 {
        (0..EXCITED).map(|x| self.decay[(x, x)].re).fold(f64::INFINITY, f64::min)
    }

    /// ‖(K − z)⁻¹‖ over a region, z = E_b + δ_d (carrier d's resolvent from ground state b in its dressing), its bound
    /// (`resolvent_own`).
    pub(crate) fn resolvent(&self, d: usize, b: usize, region: &Region) -> f64 {
        self.resolvent_own(d, b, region).upper()
    }

    /// ‖(K − z)⁻¹‖ over a region in its own form: the smaller of the dissipation's 2/Γ_min (Im⟨x|K − z|x⟩ ≤ −(Γ_min/2)‖x‖²)
    /// and, where every excited row of K − z is diagonally dominant, Varah's 1/m, m = min_e (|E_e − z − iΓ_e/2| − r_e), r_e
    /// the far couplings' Gershgorin radius (`radii`), E_e − E_b of the levels the eliminations take (`gap`). Where m ≤ 0
    /// Varah's bound says nothing and the dissipation's is the bound, so the smaller of the two is 1/max(m, Γ_min/2)
    /// everywhere -- one quotient of a quantity at least Γ_min/2, its own form at every place. (As the minimum of Varah's
    /// and the dissipation's, a cell on which m reached nothing read Varah's as [0, ∞] and the minimum as [0, 2/Γ_min], the
    /// constant 2/Γ_min of every point of the cell with a remainder of its whole size: every dressing's band through it was
    /// as wide as the dressing at every scale, and S2Q's self-energy and the drives' distances that read it never tended
    /// to their values.)
    pub(crate) fn resolvent_own(&self, d: usize, b: usize, region: &Region) -> Own {
        let made = region.resolvents.get_or_init(|| {
            let radii = self.radii(region);
            let dissipated = 0.5 * region.forms.least_decay(self);
            let of = |delta: f64, b: usize| -> Own {
                let mut least: Option<Own> = None;
                for x in 0..EXCITED {
                    let apart = self.gap(GROUND + x, b, region, delta).taken.abs().hypot(0.5 * self.decay[(x, x)].re).sub(radii[GROUND + x]);
                    least = Some(least.map_or(apart, |least| least.min(apart)));
                }
                least.expect("an atom has excited states").max(Own::constant(dissipated)).recip()
            };
            self.carriers.iter().map(|carrier| std::array::from_fn(|b| of(carrier.detuning, b))).collect()
        });
        made[d][b]
    }

    /// The carriers' dressings ΔE_b^d of the ground states over a region, [carrier][state], made once per region, each in its
    /// own form: the bound of |ΔE_b^d| = |Re Σ_units u_b† (K − z)⁻¹ u_b| (T̄ keeps each polarization unit's beams coherent),
    /// Σ over the units of ‖u_b‖² (`Couplings`) times the smaller of the resolvent's bound (`resolvent_own`) and 1/Γ_min. The
    /// second is the real part's, half the dissipation's bound of the whole: with y = (K − z)⁻¹u, y†(K − z)y = a − ib,
    /// a = y†(H − z)y real and b = y†(D/2)y ≥ (Γ_min/2)‖y‖², the form is u†(K − z)⁻¹u = a + ib, and ‖u‖ = ‖(K − z)y‖ ≥
    /// |a − ib|/‖y‖ gives ‖u‖² ≥ (a² + b²)Γ_min/(2b) ≥ Γ_min |a| -- a two-level atom's light shift |u|²Δ/(Δ² + Γ²/4) at most
    /// |u|²/Γ, at Δ = Γ/2. (With the whole's 2/Γ_min, the user's repump's dressing of F=1 m=−1 read 12.8 rad/µs against the
    /// 13.2 rad/µs Zeeman splitting from m=0, its shift about 1: the self-energy's two-photon distance between them was
    /// within 0.4 rad/µs of nothing at every point.)
    pub(crate) fn dressings<'r>(&self, region: &'r Region) -> &'r [[Own; GROUND]] {
        region.dressings.get_or_init(|| {
            let couplings = self.couplings(region);
            let shift = Own::constant(1.0 / region.forms.least_decay(self));
            (0..self.carriers.len())
                .map(|d| {
                    std::array::from_fn(|b| {
                        let square = couplings[d].units.iter().fold(Own::constant(0.0), |sum, u| sum.add(u[b].square()));
                        square.mul(self.resolvent_own(d, b, region).min(shift)).within(0.0, f64::INFINITY)
                    })
                })
                .collect()
        })
    }

    /// The carriers' dressings ΔE_b^d of the ground states over a region with their signs, [carrier][state], made once per
    /// region (`Shift`): ΔE = −Re Σ_units u†(K − z)⁻¹u, z = E_b + δ_d. With K − z = D + C, D its diagonal -- E_e − z − iΓ_e/2
    /// of the levels the eliminations take (`Gap`) -- and C the far couplings among the excited states, u†(D + C)⁻¹u = u†D⁻¹u −
    /// u†D⁻¹C(D + C)⁻¹u: the first Σ_e |u_e|²/(Δ_e − iΓ_e/2), whose real part is each excited state's dispersive shape φ
    /// (own.rs `Own::dispersion`) times each unit's |u_e|² -- the band's smooth part, its gradient Σ (∇|u_e|² φ(Δ_e) + |u_e|²
    /// φ'(Δ_e) ∇Δ_e) of the intensity's own forms and the gap's (reach.rs `Region::intensity`, `Gap`) --, the second at most
    /// ‖u‖² ‖C‖ ‖(K − z)⁻¹‖/min_e |Δ_e − iΓ_e/2|, ‖C‖ at most the largest Gershgorin radius (`radii`) and the resolvent's
    /// bound `resolvent_own`: the band's radius; its ends each within the dressing's bound (`dressings`).
    /// A two-photon distance moves by the difference of its two levels' dressings (units.rs `Model::across`): two Zeeman
    /// sublevels the same light dresses alike stay as far apart as they were, where the two dressings' bounds added -- 12.8
    /// rad/µs on the user's F=1 m=−1 -- closed the 13.2 rad/µs between them.
    pub(crate) fn dressing_shifts<'r>(&self, region: &'r Region) -> &'r [[Shift; GROUND]] {
        region.shifted.get_or_init(|| {
            let (couplings, dressings, radii) = (self.couplings(region), self.dressings(region), self.radii(region));
            let folded = region.forms.folded(self);
            let none = Own::constant(0.0);
            let coupled = (GROUND..STATES).map(|x| radii[x]).fold(none, Own::max);
            (0..self.carriers.len())
                .map(|d| {
                    let (cc, of) = (&self.carriers[d], &folded[d]);
                    std::array::from_fn(|b| {
                        // Per excited state the dispersive shape at its detuning and its slope there, the detuning's gradient,
                        // and |Δ_e − iΓ_e/2|.
                        let lines: Vec<(usize, Own, Own, Form<[f64; 3]>, Own)> = of.reached[b]
                            .iter()
                            .map(|&e| {
                                let gamma = 0.5 * self.decay[(e, e)].re;
                                let gap = self.gap(GROUND + e, b, region, cc.detuning);
                                (e, gap.taken.dispersion(gamma), gap.taken.dispersion_slope(gamma), self.gap_slope(GROUND + e, b, region, true), gap.taken.abs().hypot(gamma))
                            })
                            .collect();
                        let (mut shift, mut norms) = (Shift::none(), none);
                        for (beams, size) in of.beams[b].iter().zip(&couplings[d].units) {
                            for (e, shape, bend, slope, _) in &lines {
                                let entries: Vec<(usize, C64, f64)> = beams.iter().flat_map(|(l, entries)| entries.iter().filter(|x| x.0 == *e).map(|&(_, v, size)| (cc.beams[*l], v, size))).collect();
                                if entries.is_empty() {
                                    continue;
                                }
                                let (square, square_slope) = region.intensity(self, cc.frequency, &entries);
                                shift.band.mid = shift.band.mid.sub(square.mul(*shape));
                                shift.slope = shift.slope.sub(square_slope.by(shape).add(slope.by(&square.mul(*bend))));
                            }
                            norms = norms.add(size[b].square());
                        }
                        shift.band.rad = match lines.iter().map(|x| x.4).reduce(Own::min) {
                            Some(least) if norms.upper() > 0.0 => norms.mul(coupled).mul(self.resolvent_own(d, b, region)).div(least).within(0.0, f64::INFINITY),
                            _ => none,
                        };
                        shift.band = shift.band.within(dressings[d][b]);
                        shift
                    })
                })
                .collect()
        })
    }

    /// ∇ Σ_pairs c_l'l [P_l'l]_xy over a region, the far light's static forms' entry (x, y) (reach.rs `Region::gradient`;
    /// rad/µs/µm).
    pub(crate) fn entry_slope(&self, x: usize, y: usize, region: &Region) -> Form<[C64; 3]> {
        region.gradient(region.forms.entry(self, x, y).iter().map(|&(frequency, pair, w, size)| (region.pairs[frequency][pair], w, size)))
    }

    /// The bound of the force on a cluster's state over a region (rad/µs/µm), made once per region and cluster, in its own form:
    /// −⟨ψ|∇H_h|ψ⟩ at most the gradient's norm, at most the largest over its states of their rows' sums -- a state's own level
    /// with its shift by its drives folded at the atom, one gradient (units.rs `Model::near_level`), and the far light's
    /// |∇H_xy| to the cluster's other states (`entry_slope`) --, and the scattering's mean recoil k_max Γ_max/2 (the jumps'
    /// mean force at most).
    pub(crate) fn force_over(&self, cluster: usize, region: &Region) -> Own {
        *region.forces[cluster].get_or_init(|| {
            let states = &self.clusters[cluster].states;
            let k_max = self.carriers.iter().flat_map(|c| c.beams.iter().map(|&l| self.light.beams[l].wave_number)).fold(0.0, f64::max);
            let mut most: Option<Own> = None;
            for &x in states {
                let own = Shift::of(Own::constant(0.0), self.entry_slope(x, x, region).re()).add(&self.near_level(x, region)).steepness();
                let row = states.iter().filter(|&&y| y != x).fold(own, |row, &y| row.add(self.entry_slope(x, y, region).norm()));
                most = Some(most.map_or(row, |most| most.max(row)));
            }
            let gamma_max = (0..EXCITED).map(|x| self.decay[(x, x)].re).fold(0.0, f64::max);
            most.unwrap_or(Own::constant(0.0)).plus(k_max * gamma_max / 2.0)
        })
    }

    /// Per component (2 frequency + counter) and pair of an excited state and a ground one ((excited − GROUND) GROUND +
    /// ground) the request's beam of the component's largest coupling between them, of equal ones the last: the beam whose
    /// photon a drive's motion is read with (`drive_motion`), the same at every sample.
    pub(crate) fn strongest_made(&self) -> Vec<Vec<usize>> {
        let of = |frequency: usize, side: usize, excited: usize, ground: usize| -> usize {
            let beams = &self.light.frequencies[frequency].beams;
            beams.iter().zip(&self.tones[frequency].couplings[side]).max_by(|x, y| x.1[(excited, ground)].abs().total_cmp(&y.1[(excited, ground)].abs())).map(|(&l, _)| l).expect("a frequency has a beam")
        };
        (0..2 * self.tones.len()).map(|c| (GROUND..STATES).flat_map(|e| (0..GROUND).map(move |g| (e, g))).map(|(e, g)| of(c / 2, c % 2, e, g)).collect()).collect()
    }

    /// A drive's motion at a sample (SPEC §1.5; units.rs `Measured`): how far its two levels are there from the bare ones,
    /// δΔ = (E_e − E_b)(r) − (E_e − E_b)₀ + w + Σ_ee, with its strongest beam's Doppler shift w = (ħ/m)(q·κ + q²/2) at the
    /// sample's momentum (module documentation, # First-order Doppler; q = k_l, a counter-rotating component's −k_l) and
    /// its excited state's self-energy by the kept drives into it, Σ_ee = Σ |V[e, a]|²/(E_b + ν − E_a − δ_d) over those
    /// whose channel is another Floquet state (# The response: what a bare fold's constant response leaves out of the
    /// Raman pair, its series' first order with the rest); and the rate |v·∇(E_e − E_b) + q·v̇| at which its detuning
    /// moves, v̇ the force on the state at the point over m.
    pub(crate) fn drive_motion(&self, component: Component, excited: usize, ground: usize, at: &Measured) -> (f64, f64) {
        let side = component.counter as usize;
        let strongest = self.strongest[2 * component.frequency + side][(excited - GROUND) * GROUND + ground];
        let q = self.light.beams[strongest].wave_vector().map(|k| if component.counter { -k } else { k });
        let h = &at.energies;
        let v: [f64; 3] = at.sample.at.kappa.map(|k| k * self.hbar_over_mass);
        let mut moved = (h[0][excited] - self.energies[excited]) - (h[0][ground] - self.energies[ground]) + self.hbar_over_mass * (dot3(q, at.sample.at.kappa) + 0.5 * dot3(q, q));
        let nu = self.tones[component.frequency].nu[side];
        for &(d, a) in self.pair_legs(component, excited, ground) {
            moved += self.kept_at(at, d, excited, a).norm_sqr() / (h[0][ground] + nu - h[0][a] - self.carriers[d].detuning);
        }
        let rate: f64 = (0..3).map(|i| v[i] * (h[1 + i][excited] - h[1 + i][ground])).sum::<f64>() + self.hbar_over_mass * dot3(q, at.force);
        (moved, rate.abs())
    }

    /// A drive folded with the levels at the atom, its small parameters at a sample (`Folded`; SPEC §1.5; units.rs
    /// `Measured`): its own saturation s = 2|R_c[e,b]|² at the point (R = Σ_l a_l T̃_l), its denominator's modulus |ΔE|
    /// there (`drive_motion`) and its sweep a = |v·∇(E_e − E_b) + k·v̇|/|ΔE|² (the one gate, units.rs `gate`), its Doppler
    /// order's (w‖T2‖/‖T‖)², its ground state's population p_b and what its carrier's fold makes of that state
    /// (`Measured::folds`), with the response R_c[e,b] itself; None where the sample's cluster does not hold its ground
    /// state or the model no such fold.
    pub(crate) fn drive_parameters(&self, unit: &Unit, at: &Measured) -> Option<Folded> {
        let Unit::Drive { component, excited, ground } = *unit else { return None };
        let c = &self.clusters[at.sample.cluster];
        let pb = c.states[..c.ground].iter().position(|&s| s == ground)?;
        let ci = self.carrier_of(component.frequency).filter(|_| !component.counter)?;
        let response = at.responses[ci].as_ref().map_or(C64::ZERO, |m| m[(excited - GROUND, pb)]);
        let (distance, rate) = self.drive_distance(ci, excited, ground, at);
        let doppler = at.doppler[ci].get_or_init(|| self.doppler_sizes(&at.plan, &at.near, ci, at.sample.at.kappa))[pb];
        // L_e/S_b as a ratio of distances: the level's own term across this drive's, over the drives' sizes in it each
        // across its own -- 2|N_bb|/|ΔE_e| over S_b where the responses are V/ΔE, and one on one line whatever the
        // response holds beside (its first-order Doppler term). At one distance it is |Σ V*R|/Σ |V R|, at most one.
        let fold = &at.folds[ci][pb];
        let across = *fold.across.get_or_init(|| (0..EXCITED).filter(|&x| fold.terms[x] > 0.0).map(|x| fold.terms[x] / self.drive_distance(ci, GROUND + x, ground, at).0).sum());
        let reads = if across > 0.0 { (fold.level.abs() / distance / across).max(1.0) } else { 1.0 };
        // Its sweep as the gate reads it (units.rs `Gate::resolved`): nothing where its whole effect is below what an output resolves.
        let share = 2.0 * response.norm_sqr();
        Some(Folded { share, distance, sweep: Gate { saturation: share, sweep: gate(0.0, distance, rate).sweep }.resolved().sweep, doppler: doppler * doppler, population: at.populations[ground], total: fold.share, reads, response })
    }

    /// A drive's denominator at a sample, its modulus |E_e − E_b − δ_c + δΔ − iΓ_e/2| with the levels' motion there, and the
    /// rate at which its detuning moves (`drive_motion`), of a carrier's co-rotating drive e ← b.
    fn drive_distance(&self, carrier: usize, excited: usize, ground: usize, at: &Measured) -> (f64, f64) {
        let (moved, rate) = self.drive_motion(Component { frequency: self.carriers[carrier].frequency, counter: false }, excited, ground, at);
        (C64::new(self.energies[excited] - self.energies[ground] - self.carriers[carrier].detuning + moved, 0.5 * self.decay[(excited - GROUND, excited - GROUND)].re).abs(), rate)
    }

    /// The error at a sample of a drive folded with the levels at the atom (SPEC §1.5; units.rs, # The fold's order): its
    /// motion's part on T, ((w‖T2‖/‖T‖)² + a) p_b, and its order's (`drive_parameters`). The order's error is its ground
    /// state's and not the drive's alone (units.rs, # The fold's order): the drive's line errs by M^k, M the larger of the
    /// state's excited share S_b = 2 Σ_e |R[e, b]|² and of twice its level's own second-order term across the drive's
    /// distance, 2|N_bb|/|ΔE| -- on one line both are s --, on the drive's part s/S_b of the state's excitation
    /// (`Folded::order`). Its line's rate off by M^k moves the populations by what its flows move (units.rs `Model::moved`,
    /// # A rate's error on the populations) and its folded excited population s/2 p_b by M^k of it, on the populations
    /// and on T, where its cluster's fold has no other order -- and the second order's s M/S_b p_b on T; where it has,
    /// nothing at the second order (the order is the cluster's own unit's, `Unit::Second`), and at the fourth what it
    /// leaves: `ORDER_RATES` M² of the flows and the excited population, and (`ORDER_FORCE` s M²/S_b + the cluster's ground
    /// levels' spread over the drive's distance times s) p_b on T. A state's drives at one distance add up to S_b p_b and
    /// c S_b² p_b on T. With the drive's response there, where the cluster holds it (`pair_error` reads it).
    pub(crate) fn drive_error(&self, unit: &Unit, at: &Measured) -> (UnitError, Option<C64>) {
        let Some(drive) = self.drive_parameters(unit, at) else { return (UnitError::NONE, None) };
        let (s, cluster) = (drive.share, at.sample.cluster);
        let Unit::Drive { component, excited, ground } = *unit else { unreachable!("a drive's parameters are a drive's") };
        let m = drive.total * drive.reads;
        // What its line's order moves: its flows' part and its excited population's.
        let moved = |line: f64| self.moved(at, Flow::Atom(component, excited, ground)) + 0.5 * s * line * drive.population;
        let flow = self.moved(at, Flow::Atom(component, excited, ground));
        let (populations, force) = match (self.fourth[cluster], self.fold_frequency(cluster)) {
            (Some(_), _) => {
                let levels = self.clusters[cluster].states.iter().map(|&b| at.energies[0][b]);
                let spread = levels.clone().fold(f64::NEG_INFINITY, f64::max) - levels.fold(f64::INFINITY, f64::min);
                (moved(super::units::ORDER_RATES * m * m), (super::units::ORDER_FORCE * drive.order(2) + spread / drive.distance * s) * drive.population)
            }
            (None, Some(_)) => (0.0, 0.0),
            (None, None) => (moved(m), drive.order(1) * drive.population),
        };
        let flow = if populations > 0.0 { flow } else { 0.0 };
        (UnitError { parameter: s, population: populations, temperature: force + populations + (drive.doppler + drive.sweep) * drive.population, moved: flow, into: 0 }, Some(drive.response))
    }

    /// A carrier's explicit coupling e ← a at a point, Σ_l a_l X_l[e, a].
    pub(crate) fn kept_coupling(&self, carrier: usize, e: usize, a: usize, amplitudes: &[Amplitude]) -> C64 {
        let cc = &self.carriers[carrier];
        cc.beams.iter().zip(&cc.explicit).map(|(&l, m)| amplitudes[l].value * m[(e, a)]).sum()
    }

    /// A kept drive's coupling at a sample's point (`kept_coupling` of one of the model's legs, `Model::legs`): every
    /// leg's made once for the sample -- each drive folded onto an explicit state asks its pairs' legs' (units.rs
    /// `Measured`).
    pub(crate) fn kept_at(&self, at: &Measured, carrier: usize, e: usize, a: usize) -> C64 {
        let place = |carrier: usize, x: usize, a: usize| (carrier * EXCITED + x) * GROUND + a;
        let made = at.kept.get_or_init(|| {
            let mut made = vec![C64::ZERO; self.carriers.len() * EXCITED * GROUND];
            for (x, legs) in self.legs.iter().enumerate() {
                for &(d, s) in legs {
                    made[place(d, x, s)] = self.kept_coupling(d, GROUND + x, s, &at.amplitudes);
                }
            }
            made
        });
        made[place(carrier, e - GROUND, a)]
    }

    /// A ground level's shift by its kept drives at a sample's point, at most: each one's coupling v, and v² across
    /// its own denominator. Every ground state's made once for the sample.
    pub(crate) fn lifts_at(&self, at: &Measured, s: usize) -> f64 {
        let h = &at.energies[0];
        at.lifts.get_or_init(|| {
            std::array::from_fn(|s| {
                let kept = self.legs.iter().enumerate().flat_map(|(x, of)| of.iter().filter(move |leg| leg.1 == s).map(move |leg| (GROUND + x, leg.0)));
                kept.map(|(x, d)| {
                    let v = self.kept_at(at, d, x, s).abs();
                    v.min(v * v / C64::new(h[x] - h[s] - self.carriers[d].detuning, 0.5 * self.decay[(x - GROUND, x - GROUND)].re).abs())
                })
                .sum()
            })
        })[s]
    }

    /// The error at a sample of the Raman pairs that a drive U (e ← b at ν) folded onto an explicit state e makes with
    /// the kept drives V_a into e (far.rs, # A fold onto an explicit state; SPEC §1.5): the pair's next order, as each of
    /// its two states reads it, `units::PAIR` m².
    /// - The sample in e's cluster: the cluster's state holds b's dressed state by m² = |U|² p_e/d_R² -- the excited
    ///   amplitude times U across the two-photon distance d_R = z_e − E_b − ν of the ground states that read e, the
    ///   nearest of them. The self-energy is |U|²/(d_R + λ) at the reading state's own eigenvalue λ = −|V|²/ε̃, ε̃ =
    ///   ε − iΓ/2, of which the generator's |U|²/d_R is the first term: the kept line's response −|V|²/(ε̃ + Σ) is off by
    ///   |U V|² |V|²/(ε̃³ d_R²), its rate by m² (3ε² − Γ²/4)/(ε² + Γ²/4) of itself and its shift by m² (ε² − 3Γ²/4)/(ε² +
    ///   Γ²/4), both at most 3 m² (`units::PAIR`), booked on the populations and on T (three levels, V = 0.5Γ, ε = Γ,
    ///   U = 0.2Γ: the rate within it wherever the gate holds, units.rs' test). The distance is what is left of |d_R|
    ///   by the kept drives' own shifts of the two ground levels (`kept_lift`: λ, as the gate takes it, units.rs
    ///   `Model::pair_gate`). The shift is read at the cluster's mean (model.rs
    ///   `Model::read`): a ground state `off` from it reads |U|² off/d_R² too much or too little of e's level, a
    ///   relative error of the kept lines' response of that over their nearest denominator |ε̃| -- 2 of it times e's
    ///   share of the state's rates, on the populations and T, and one of it times the kept drives' share of its
    ///   light shifts, on T, as a bare fold's denominator's first order is booked (units.rs `Model::bare_error`).
    /// - The sample in b's cluster: b's response R[e, b] continues through each kept drive into a's dressed state,
    ///   m² = |R[e, b]|² |V_a|²/d_a², d_a = E_b + ν − E_a − δ_d, by the same bound `units::PAIR` m² p_b (the response's
    ///   own value, with the levels at the atom, has the pair in its self-energy; with the bare ones, in the first
    ///   order of its denominator, `drive_motion`).
    pub(crate) fn pair_error(&self, component: Component, excited: usize, ground: usize, response: Option<C64>, at: &Measured) -> UnitError {
        let side = component.counter as usize;
        let (nu, h) = (self.tones[component.frequency].nu[side], &at.energies[0]);
        let legs = self.pair_legs(component, excited, ground);
        let mut error = UnitError::NONE;
        if legs.is_empty() {
            return error;
        }
        let p = &at.populations;
        let lift = |s: usize| self.lifts_at(at, s);
        if let Some(r) = response {
            for &(d, a) in legs {
                let apart = (h[ground] + nu - h[a] - self.carriers[d].detuning).abs() - lift(ground) - lift(a);
                let s = super::units::PAIR * 0.5 * gate((r * self.kept_at(at, d, excited, a)).abs(), apart.max(0.0), 0.0).saturation;
                error.parameter = error.parameter.max(s);
                error.population += s * p[ground];
                error.temperature += s * p[ground];
            }
        }
        if let Some((s, epsilon, kept)) = self.pair_reading(component, excited, ground, at) {
            let held: f64 = p[..GROUND].iter().sum();
            let share = |of: f64, all: f64| if all > 0.0 { of / all } else { 0.0 };
            let (rates, shifts) = (2.0 * epsilon * share(p[excited], at.rates) * held, epsilon * share(kept, at.shifts) * held);
            let moved = self.moved(at, Flow::Kept(component, excited, ground));
            error.parameter = error.parameter.max(s);
            error.population += s + moved;
            error.temperature += s + rates + shifts + moved;
            error.moved += moved;
        }
        error
    }

    /// The reading of a drive U (e ← b at ν) folded onto an explicit state e by the kept lines through e, at a sample in
    /// e's cluster (`pair_error`): the pair's mixing `units::PAIR` m², m² = |U|² p_e/d_R² over the nearest two-photon
    /// distance; the relative error ε of the kept lines' response, |U|² off/d_R² over their nearest denominator; and the
    /// kept couplings' energy in the state. None off e's cluster or where e is empty.
    pub(crate) fn pair_reading(&self, component: Component, excited: usize, ground: usize, at: &Measured) -> Option<(f64, f64, f64)> {
        let side = component.counter as usize;
        let (nu, h, p) = (self.tones[component.frequency].nu[side], &at.energies[0], &at.populations);
        let lift = |s: usize| self.lifts_at(at, s);
        let c = &self.clusters[at.sample.cluster];
        let pe = c.states.iter().position(|&s| s == excited).filter(|_| c.ground > 0 && p[excited] > 0.0)?;
        let beams = &self.light.frequencies[component.frequency].beams;
        let u: C64 = beams.iter().zip(&self.tones[component.frequency].couplings[side]).map(|(&l, x)| if component.counter { at.amplitudes[l].value.conj() } else { at.amplitudes[l].value } * x[(excited, ground)]).sum();
        let frame = |place: usize| self.axes.omega(&c.frames[place]);
        let read = self.read(c, pe, h, true);
        let each = |pa: usize| frame(pe) - frame(pa) + h[c.states[pa]];
        let nearest = (0..c.ground).map(|pa| ((each(pa) - h[ground] - nu).abs() - lift(c.states[pa]) - lift(ground)).max(0.0)).fold(f64::INFINITY, f64::min);
        let off = (0..c.ground).map(|pa| (each(pa) - read).abs()).fold(0.0, f64::max);
        let s = super::units::PAIR * 0.5 * gate(u.abs() * p[excited].sqrt(), nearest, 0.0).saturation;
        // The kept lines through e: their nearest denominator, and their coupling's energy in the state.
        let gamma = self.decay[(excited - GROUND, excited - GROUND)].re;
        let own = (0..c.ground).map(|pa| C64::new(h[excited] - each(pa), 0.5 * gamma).abs()).fold(f64::INFINITY, f64::min);
        let kept: f64 = (0..c.ground).map(|pa| self.legs[excited - GROUND].iter().filter(|leg| leg.1 == c.states[pa]).map(|&(d, a)| self.kept_at(at, d, excited, a)).sum::<C64>().abs() * (p[excited] * p[c.states[pa]]).sqrt()).sum();
        Some((s, gate(0.0, nearest, u.norm_sqr() * off).sweep / own, kept))
    }
}

/// A coupling of two states at a beat in canonical form (`Unit::Term`): a ← b at the beat and its adjoint, b ← a at the
/// opposite beat, are one unit, named a < b, or for a = b by the beat whose first coordinate that is not zero is positive.
pub(crate) fn term_unit(a: usize, b: usize, beat: &Beat) -> Unit {
    if a < b || (a == b && beat.0.iter().find(|n| **n != 0).is_none_or(|n| *n > 0)) {
        Unit::Term { a, b, beat: beat.clone() }
    } else {
        Unit::Term { a: b, b: a, beat: beat.negated() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
    use crate::model::{Point, Sample};
    use crate::request::Request;

    fn scene(name: &str) -> Request {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        Request::from_native(&line).unwrap()
    }

    /// |∫₀^∞ e^{−x + iax²/2} dx|² by composite Simpson on [0, 40] (the integrand's modulus e^{−x}, its tail below e^{−40}).
    fn crossing(a: f64) -> f64 {
        let (n, top) = (400_000, 40.0);
        let h = top / n as f64;
        let f = |x: f64| C64::cis(a * x * x / 2.0) * (-x).exp();
        let mut sum = f(0.0) + f(top);
        for i in 1..n {
            sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        (sum * (h / 3.0)).norm_sqr()
    }

    /// A two-level atom swept linearly through resonance (δ(t) = δ̇t, a weak drive, decay Γ): the exact excited amplitude relative to
    /// the instantaneous fold is ∫₀^∞e^{−x+iax²/2}dx at a = |δ̇|/(Γ/2)² (internal.md 1B.1) -- its population 0.71 of the fold's at
    /// a = ½, the gate's edge, and 0.39 at a = 1.75; the photons of one passage are πΩ²/(2|δ̇|) = 2π|V|²/|δ̇| exactly at weak drive,
    /// which the fold's rate Γ|V|²/(δ² + Γ²/4) integrates to whatever a (the rate itself is the two-level test's, tests/model_near.rs).
    /// R2_RECOIL's σ⁺ probe and |2,2⟩ → |3′,3′⟩, its excited state moved onto resonance at r₀ on the tweezer's flank, an atom crossing
    /// r₀ along x at the speed that makes a = ½ through the tweezer's differential shift: the measured a is |v·∂Δ + k·v̇|/(Γ/2)² there,
    /// and the gate's provable sweep bound over a reach that holds that point and speed is above it.
    #[test]
    fn a_swept_resonance_s_measured_sweep_and_its_gate_are_the_exact_crossing_s() {
        assert!((crossing(0.5) - 0.71).abs() < 0.005, "{}", crossing(0.5));
        assert!((crossing(1.75) - 0.39).abs() < 0.005, "{}", crossing(1.75));
        let mut request = scene("R2_RECOIL");
        let light = Light::new(&request);
        let beam = request.beams.iter().position(|b| b.id == "probe").unwrap();
        let probe = light.frequencies.iter().position(|f| f.beams.contains(&beam)).unwrap();
        let kept = Model::new(&request, &light, &[]).unwrap().bare_units(Component { frequency: probe, counter: false });
        let (b, e) = (7, 31);
        let r0 = [0.4, 0.0, 0.0];
        let detuning = |model: &Model, r: [f64; 3]| {
            let h = model.far_hamiltonian(&model.light.amplitudes(r), &model.running(), 1, &model.coherent_dressing());
            let cc = &model.carriers[0];
            (h[0][(e, e)].re - h[0][(b, b)].re - cc.detuning, h[1][(e, e)].re - h[1][(b, b)].re)
        };
        for _ in 0..3 {
            let model = Model::new(&request, &light, &kept).unwrap();
            let (delta, _) = detuning(&model, r0);
            request.hamiltonian[(e, e)] -= C64::real(delta);
        }
        let model = Model::new(&request, &light, &kept).unwrap();
        let (delta, slope) = detuning(&model, r0);
        let gamma = model.decay[(e - GROUND, e - GROUND)].re;
        assert!(delta.abs() < 1e-6 * gamma, "on resonance at r₀: Δ = {delta}");
        let step = 1e-4;
        let difference = (detuning(&model, [r0[0] + step, 0.0, 0.0]).0 - detuning(&model, [r0[0] - step, 0.0, 0.0]).0) / (2.0 * step);
        assert!((difference - slope).abs() <= 1e-6 * slope.abs(), "∂Δ {slope} against its difference {difference}");
        let speed = 0.5 * gamma * gamma / 4.0 / slope.abs();
        let cluster = model.cluster_of(b);
        let at = Point { r: r0, kappa: [speed / model.hbar_over_mass, 0.0, 0.0], t: 0.0 };
        let sample = Sample { cluster, at, psi: vec![C64::ONE] };
        let unit = Unit::Drive { component: Component { frequency: probe, counter: false }, excited: e, ground: b };
        let measured = model.measured(&sample, None);
        let sweep = model.drive_parameters(&unit, &measured).unwrap().sweep;
        let force = measured.force;
        let k = model.light.beams[model.carriers[0].beams[0]].wave_vector();
        let shift = model.hbar_over_mass * (dot3(k, at.kappa) + 0.5 * dot3(k, k));
        let expected = (speed * slope + model.hbar_over_mass * dot3(k, force)).abs() / (shift * shift + gamma * gamma / 4.0);
        assert!((sweep - expected).abs() <= 1e-9 * expected, "the measured a {sweep} against |v·∂Δ + k·v̇|/(w² + (Γ/2)²) = {expected}");
        assert!((sweep - 0.5).abs() < 0.05, "a near ½: {sweep}");
        // The gate over a reach that holds r₀ at that speed.
        let p = {
            let mut p = [0.0; GROUND];
            p[b] = 1.0;
            p
        };
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let mut temperature = 1.0;
        let reach = loop {
            let reach = model.reach(&trap, &p, temperature, 2e-3);
            if reach.speed >= speed && model.potential(r0, &p, &model.prepared()) <= reach.minimum + reach.energy {
                break reach;
            }
            temperature *= 2.0;
        };
        let gate = model.gates(&[unit], &reach)[0];
        assert!(gate.sweep >= sweep * (1.0 - 1e-9), "the gate's bound {} below the measured a {sweep}", gate.sweep);
    }
}
