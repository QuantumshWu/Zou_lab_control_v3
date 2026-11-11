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
use super::photon::Origin;
use super::reach::{distance, Region, Weights};
use super::units::{gate, Measured};
use super::{Beat, Component, Level, Lights, Model, Point, Unit, UnitError};
use crate::field::Amplitude;
use crate::linalg::{dot3, Lu, Mat, C64};
use crate::request::{GROUND, STATES};
use super::far::EXCITED;
use super::jet;

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
        (0..self.light.frequencies.len())
            .map(|w| {
                let n = self.light.frequencies[w].beams.len();
                let carrier = self.carrier_of(w).filter(|&d| !coherent[d]).map(|d| &self.carriers[d]);
                (0..n * n).map(|x| lights.pair(w, x / n, x % n) && carrier.is_none_or(|cc| cc.units[x / n] == cc.units[x % n])).collect()
            })
            .collect()
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

    /// The same levels' energies, H(r)'s diagonal to `order` ([part][state]), and their complex multiply-adds: the base's and
    /// each beam pair's coefficient times its form's diagonal, STATES per part.
    pub(crate) fn far_energies(&self, amplitudes: &[Amplitude], lights: &Lights, order: usize, coherent: &[bool]) -> (Vec<Vec<f64>>, f64) {
        let np = jet::parts(order);
        let mut e: Vec<Vec<f64>> = (0..np).map(|p| (0..STATES).map(|s| if p == 0 { self.base[(s, s)].re } else { 0.0 }).collect()).collect();
        let mut work = 0.0;
        let pairs = self.level_pairs(lights, coherent);
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            let n = f.beams.len();
            for (x, _) in pairs[w].iter().enumerate().filter(|of| *of.1) {
                let c = jet::of(&super::far::pair(&amplitudes[f.beams[x / n]], &amplitudes[f.beams[x % n]]), order);
                for (part, &z) in e.iter_mut().zip(&c) {
                    for (s, v) in part.iter_mut().enumerate() {
                        *v += (z * tone.pairs[x][(s, s)]).re;
                    }
                }
                work += (np * STATES) as f64;
            }
        }
        (e, work)
    }

    /// The complex multiply-adds of `far_hamiltonian`: each of its beam pairs' form added to every part of the jet, STATES² each.
    pub(crate) fn far_work(&self, lights: &Lights, order: usize, coherent: &[bool]) -> f64 {
        let pairs: usize = self.level_pairs(lights, coherent).iter().map(|of| of.iter().filter(|pair| **pair).count()).sum();
        (pairs * jet::parts(order) * STATES * STATES) as f64
    }

    /// Per carrier whether its dressing takes its coherent field (`Unit::Dressing` kept) rather than its units' T̄.
    pub(crate) fn coherent_dressing(&self) -> Vec<bool> {
        self.carriers.iter().map(|c| self.kept.contains(&Unit::Dressing { carrier: Component { frequency: c.frequency, counter: false } })).collect()
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
    /// per ground place.
    pub(crate) fn eliminate(&self, cluster: usize, h: &[Mat], amplitudes: &[Amplitude], lights: &Lights, order: usize, coherent: &[bool]) -> Responses {
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
            for (pb, &b) in c.states[..g].iter().enumerate() {
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
        let (start, count) = (plan.near.responding().start, plan.near.responses.len());
        let mut out: Vec<Option<Mat>> = self.carriers.iter().map(|cc| plan.lights.on(cc.frequency).then(|| Mat::zeros(EXCITED, g))).collect();
        let fields: Vec<Vec<(C64, C64)>> = self
            .carriers
            .iter()
            .map(|cc| {
                let f = &self.light.frequencies[cc.frequency];
                (0..cc.basis.len())
                    .map(|k| {
                        f.beams.iter().zip(&f.coefficients).fold((C64::ZERO, C64::ZERO), |(plain, weighted), (&l, c)| {
                            let z = amplitudes[l].value * c[k];
                            (plain + z, weighted + z * self.doppler_shift(l, kappa))
                        })
                    })
                    .collect()
            })
            .collect();
        for (i, &[c, k, e, y]) in plan.near.responses.iter().enumerate() {
            let (plain, weighted) = fields[c][k];
            out[c].as_mut().expect("a lit carrier's response")[(e, y)] += plain * near.data[start + i] - weighted * near.data[start + count + i];
        }
        out
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

    /// The channels of the drives folded at the atom on a cluster at a point (readout.rs): per lit carrier its response there
    /// (`carrier_responses`), the forms from their table where one serves.
    pub(crate) fn near_channels(&self, cluster: usize, at: &Point, amplitudes: &[Amplitude], lights: &Lights, tables: Option<&super::tabulate::Tables>) -> Vec<super::readout::Channels> {
        let plan = self.plan(cluster, lights);
        if plan.near.len() == 0 {
            return vec![];
        }
        let near = self.near_values(&plan, amplitudes, &[(plan.near.responding(), 0)], tables.and_then(|t| t.of(cluster, lights)));
        self.carrier_responses(&plan, &near, amplitudes, at.kappa)
            .into_iter()
            .zip(&self.carriers)
            .filter_map(|(r, cc)| r.map(|response| super::readout::Channels { component: Component { frequency: cc.frequency, counter: false }, nu: cc.beat.clone(), response }))
            .collect()
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
        let Some(d) = self.carrier_of(carrier.frequency) else { return UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 } };
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
        UnitError { parameter: delta, population: delta * weight, temperature: delta * weight }
    }

    /// The bound of the differential shift of two states by the drives folded at the atom over a region (what a coupling's
    /// distance loses, units.rs # What a folded drive does to its level): each state's own shift's bound (units.rs
    /// `Model::near_shift`).
    pub(crate) fn near_spread(&self, a: usize, b: usize, region: &Region) -> f64 {
        self.near_shift(a, region) + self.near_shift(b, region)
    }
}

/// The constant forms' energies over a region (the gates' distances): per state its value at the centre and the bound of how
/// far it moves within the cell, per excited state the Gershgorin radius of the forms' couplings among the excited states,
/// and per state the bound of how far the level the eliminations take (module documentation, # The response: a carrier's
/// frequency's form with its polarization units' interference averaged) lies from the level itself -- the beam pairs
/// that average leaves out, their sum at the centre and how far each moves.
pub(crate) struct CellEnergies {
    pub(crate) centre: [f64; STATES],
    pub(crate) slack: [f64; STATES],
    pub(crate) radius: [f64; STATES],
    pub(crate) averaged: [f64; STATES],
}

/// A carrier's coherent couplings over a region (reach.rs `Region`). With u_s = Σ_l a_l X^f_l[:, s] the excited amplitudes its
/// field drives from ground state s -- its beams coherent at the request's phases (SPEC §1.2), what the atom feels at r --, per
/// ground state the bounds over the region of |u_s[e]| per excited state, of ‖u_s‖, and of ‖u_s‖ per polarization unit (its beams
/// alone): each a sum of its beams over the cell, to first and to second order from the centre's (reach.rs `Region::summed`), at
/// most every beam at its largest amplitude and in phase; on a point, the value there.
pub(crate) struct Couplings {
    /// [s][e − GROUND].
    pub(crate) entry: [[f64; EXCITED]; GROUND],
    /// [s].
    pub(crate) column: [f64; GROUND],
    /// [unit][s], over the carrier's distinct polarization units.
    pub(crate) units: Vec<[f64; GROUND]>,
}

impl Model {
    /// The far light's energies over a region (`CellEnergies`), made once per region, every light on: each beam pair's
    /// coefficient a_l'* a_l within its centre value ± its radius (reach.rs `PairSpan`), the forms' entries from the region's
    /// `Forms` -- an excited state's Gershgorin row over the states it is coupled to somewhere (the others' terms are zero).
    pub(crate) fn energies<'r>(&self, region: &'r Region) -> &'r CellEnergies {
        region.energies.get_or_init(|| {
            let forms = &region.forms;
            let mut centre: [f64; STATES] = std::array::from_fn(|s| self.base[(s, s)].re);
            let mut slack = [0.0; STATES];
            let (mut left, mut averaged) = ([0.0; STATES], [0.0; STATES]);
            for ((levels, pairs), taken) in forms.levels(self).iter().zip(&region.pairs).zip(forms.taken(self)) {
                for ((p, pair), taken) in levels.iter().zip(pairs).zip(taken) {
                    for s in 0..STATES {
                        centre[s] += (pair.centre * p[s].0).re;
                        slack[s] += pair.radius * p[s].1;
                        if !taken {
                            left[s] += (pair.centre * p[s].0).re;
                            averaged[s] += pair.radius * p[s].1;
                        }
                    }
                }
            }
            for s in 0..STATES {
                averaged[s] += left[s].abs();
            }
            let mut radius = [0.0; STATES];
            for (x, mixed) in (GROUND..STATES).zip(forms.mixed(self)) {
                for &(y, base) in mixed {
                    let (mut off, mut moved) = (base, 0.0);
                    for &(frequency, pair, v, size) in forms.entry(self, x, y) {
                        let pair = &region.pairs[frequency][pair];
                        off += pair.centre * v;
                        moved += pair.radius * size;
                    }
                    radius[x] += off.abs() + moved;
                }
            }
            Box::new(CellEnergies { centre, slack, radius, averaged })
        })
    }

    /// The carriers' coherent couplings over a region (`Couplings`), made once per region, their folded couplings from the region's
    /// `Forms`: of a beam its entries that are not zero (the others add nothing), an excited state no beam reaches bounded by zero.
    pub(crate) fn couplings<'r>(&self, region: &'r Region) -> &'r [Couplings] {
        region.couplings.get_or_init(|| {
            let spans = &region.spans;
            // A sum Σ_{l ∈ beams} a_l x_l over the excited states as its bound over the region reads it (reach.rs
            // `Region::summed`): its value and slopes at the centre, and how far its beams move from theirs, how far they bow
            // and their phase-blind sum -- [moved, bowed, blind], of the whole vector and per excited state.
            #[derive(Clone, Copy)]
            struct Sum {
                u: [C64; EXCITED],
                slopes: [[C64; EXCITED]; 3],
                whole: [f64; 3],
                each: [[f64; 3]; EXCITED],
            }
            let nothing = Sum { u: [C64::ZERO; EXCITED], slopes: [[C64::ZERO; EXCITED]; 3], whole: [0.0; 3], each: [[0.0; 3]; EXCITED] };
            let bounded = region.rho.is_finite();
            let norm_over = |sum: &Sum| region.summed(&sum.u, [&sum.slopes[0], &sum.slopes[1], &sum.slopes[2]], sum.whole[0], sum.whole[1], sum.whole[2]);
            self.carriers
                .iter()
                .zip(region.forms.folded(self))
                .map(|(cc, folded)| {
                    let mut entry = [[0.0; EXCITED]; GROUND];
                    let mut column = [0.0; GROUND];
                    let mut per_unit = vec![[0.0; GROUND]; folded.units.len()];
                    for s in 0..GROUND {
                        // Per polarization unit, and in all.
                        let mut all = nothing;
                        for (ui, beams) in folded.beams[s].iter().enumerate() {
                            let mut unit = nothing;
                            for (l, entries) in beams {
                                let span = &spans[cc.beams[*l]];
                                let over = if bounded { [span.moved(), span.bowed(), span.most] } else { [0.0, 0.0, span.most] };
                                for sum in [&mut unit, &mut all] {
                                    for &(e, v, size) in entries {
                                        sum.u[e] += span.centre * v;
                                        for a in 0..3 {
                                            sum.slopes[a][e] += span.gradient[a] * v;
                                            sum.each[e][a] += size * over[a];
                                        }
                                    }
                                    for a in 0..3 {
                                        sum.whole[a] += folded.columns[s][*l] * over[a];
                                    }
                                }
                            }
                            per_unit[ui][s] = norm_over(&unit);
                        }
                        for &e in &folded.reached[s] {
                            let slopes: [&[C64]; 3] = std::array::from_fn(|a| std::slice::from_ref(&all.slopes[a][e]));
                            entry[s][e] = region.summed(&[all.u[e]], slopes, all.each[e][0], all.each[e][1], all.each[e][2]);
                        }
                        column[s] = norm_over(&all);
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

    /// The bound over a region of ‖(K − z)⁻¹‖, z = E_b + δ_d (carrier d's resolvent from ground state b in its dressing): the
    /// dissipation's 2/Γ_min (Im⟨x|K − z|x⟩ ≤ −(Γ_min/2)‖x‖²), and where every excited row of K − z is diagonally dominant, Varah's
    /// 1/min_e (|E_e − z − iΓ_e/2| − r_e), r_e the far couplings' Gershgorin radius (`CellEnergies`).
    pub(crate) fn resolvent(&self, d: usize, b: usize, region: &Region) -> f64 {
        let made = region.resolvents.get_or_init(|| {
            let energies = self.energies(region);
            let dissipation = 2.0 / region.forms.least_decay(self);
            let of = |delta: f64, b: usize| {
                let mut least = f64::INFINITY;
                for e in GROUND..STATES {
                    let slack = energies.slack[e] + energies.slack[b] + energies.averaged[e] + energies.averaged[b];
                    let (lo, hi) = (energies.centre[e] - energies.centre[b] - delta - slack, energies.centre[e] - energies.centre[b] - delta + slack);
                    let (x, gamma) = (distance(lo, hi), self.decay[(e - GROUND, e - GROUND)].re);
                    least = least.min((x * x + gamma * gamma / 4.0).sqrt() - energies.radius[e]);
                }
                if least > 0.0 {
                    dissipation.min(1.0 / least)
                } else {
                    dissipation
                }
            };
            self.carriers.iter().map(|carrier| std::array::from_fn(|b| of(carrier.detuning, b))).collect()
        });
        made[d][b]
    }

    /// The carriers' dressings ΔE_b^d of the ground states over a region, [carrier][state], made once per region: Σ over the
    /// carrier's polarization units of ‖u_b‖² (`Couplings`; T̄ keeps each unit's beams coherent) times its resolvent's bound
    /// (`resolvent`).
    pub(crate) fn dressings<'r>(&self, region: &'r Region) -> &'r [[f64; GROUND]] {
        region.dressings.get_or_init(|| {
            let couplings = self.couplings(region);
            (0..self.carriers.len()).map(|d| std::array::from_fn(|b| couplings[d].units.iter().map(|u| u[b] * u[b]).sum::<f64>() * self.resolvent(d, b, region))).collect()
        })
    }

    /// The bound over a region of |∇ Σ_pairs c_l'l w(P_l'l)|, the far light's static forms each weighted by `w` (an entry, a
    /// diagonal's difference; reach.rs `Weights`): its value at the centre plus Σ |w| max|∇c_l'l − ∇c_l'l(c)|, at most
    /// Σ |w| max|∇c_l'l| (reach.rs `PairSpan`); on a point, its value there (rad/µs/µm).
    pub(crate) fn far_gradient(&self, region: &Region, w: &Weights) -> f64 {
        let blind: f64 = w.iter().map(|&(frequency, pair, _, size)| size * region.pairs[frequency][pair].slope).sum();
        if !region.rho.is_finite() {
            return blind;
        }
        let (mut g, mut veer) = ([C64::ZERO; 3], 0.0);
        for &(frequency, pair, x, size) in w {
            let pair = &region.pairs[frequency][pair];
            for i in 0..3 {
                g[i] += pair.gradient[i] * x;
            }
            veer += size * pair.veer;
        }
        (g.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() + veer).min(blind)
    }

    /// The bound of |∇(E_x − E_y)| of the far light's diagonal over a region (rad/µs/µm, `far_gradient`).
    pub(crate) fn energy_slope_over(&self, x: usize, y: usize, region: &Region) -> f64 {
        self.far_gradient(region, region.forms.difference(self, x, y))
    }

    /// The bound over a region of |∇| of a carrier's light shift of ground state b (module documentation, # Derivatives): over its
    /// beam pairs -- all of them, or those within one polarization unit (its dressing, T̄) -- Σ max|∇(a_l'* a_l)| ‖X^f_l'[:,b]‖
    /// ‖X^f_l[:,b]‖ ‖R‖, plus ‖u_b‖² L_b ‖R‖², the resolvent's derivative at most the slope L_b of the levels the column reaches (b's
    /// and its excited states', `energy_slope_over`) times its square (‖u_b‖ in all, or per unit, `Couplings`); ‖R‖ the dressing's
    /// resolvent bound (`resolvent`), or for the shift in the generator one over the least distance of the carrier's drives of b
    /// folded at the atom, each as its own gate reads it over the region -- its self-energy, the other carriers' dressing and
    /// the Doppler shift in it (units.rs `Model::least_across`, # What a folded drive does to its level).
    pub(crate) fn shift_slope(&self, carrier: usize, b: usize, region: &Region, dressing: bool) -> f64 {
        let cc = &self.carriers[carrier];
        let couplings = &self.couplings(region)[carrier];
        let folded = &region.forms.folded(self)[carrier];
        let columns = &folded.columns[b];
        if columns.iter().all(|&c| c == 0.0) {
            return 0.0;
        }
        let slopes = self.level_slopes(b, region);
        let levels = folded.reached[b].iter().map(|&e| slopes[e]).fold(0.0, f64::max);
        let resolvent = if dressing { self.resolvent(carrier, b, region) } else { self.least_across(carrier, b, region).map_or(0.0, |distance| 1.0 / distance) };
        let n = cc.beams.len();
        let pairs = &region.pairs[cc.frequency];
        let mut out = 0.0;
        for j in 0..n {
            for i in 0..n {
                let size = columns[j] * columns[i];
                if size > 0.0 && (!dressing || cc.units[j] == cc.units[i]) {
                    out += pairs[j * n + i].slope * size * resolvent;
                }
            }
        }
        let square = if dressing { couplings.units.iter().map(|u| u[b] * u[b]).sum::<f64>() } else { couplings.column[b] * couplings.column[b] };
        out + square * levels * resolvent * resolvent
    }

    /// The bound of the force on a cluster's state over a region (rad/µs/µm), made once per region and cluster: −⟨ψ|∇H_h|ψ⟩ at most
    /// the gradient's norm, at most the largest over its states of their rows' sums -- the far light's |∇H_xy| (`far_gradient`)
    /// and each lit carrier's light shift's (`shift_slope`) --, and the scattering's mean recoil k_max Γ_max/2 (the jumps' mean
    /// force at most).
    pub(crate) fn force_over(&self, cluster: usize, region: &Region) -> f64 {
        *region.forces[cluster].get_or_init(|| {
            let states = &self.clusters[cluster].states;
            let k_max = self.carriers.iter().flat_map(|c| c.beams.iter().map(|&l| self.light.beams[l].wave_number)).fold(0.0, f64::max);
            let mut most = 0.0_f64;
            for &x in states {
                let mut row = 0.0;
                for &y in states {
                    row += self.far_gradient(region, region.forms.entry(self, x, y));
                }
                row += self.near_slope(x, region);
                most = most.max(row);
            }
            let gamma_max = (0..EXCITED).map(|x| self.decay[(x, x)].re).fold(0.0, f64::max);
            most + k_max * gamma_max / 2.0
        })
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
        let beams = &self.light.frequencies[component.frequency].beams;
        let strongest = beams.iter().zip(&self.tones[component.frequency].couplings[side]).max_by(|x, y| x.1[(excited, ground)].abs().total_cmp(&y.1[(excited, ground)].abs())).map(|(&l, _)| l).expect("a frequency has a beam");
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

    /// A drive folded with the levels at the atom, its small parameters at a sample (SPEC §1.5; units.rs `Measured`): its own
    /// saturation s = 2|R_c[e,b]|² at the point (R = Σ_l a_l T̃_l), its sweep a = |v·∇(E_e − E_b) + k·v̇|/|ΔE|² (ΔE its
    /// denominator there, `drive_motion`; the one gate, units.rs `gate`), its Doppler order's (w‖T2‖/‖T‖)², and its ground
    /// state's population p_b, with the response R_c[e,b] itself; None where the sample's cluster does not hold its ground
    /// state or the model no such fold.
    pub(crate) fn drive_parameters(&self, unit: &Unit, at: &Measured) -> Option<(f64, f64, f64, f64, C64)> {
        let Unit::Drive { component, excited, ground } = *unit else { return None };
        let c = &self.clusters[at.sample.cluster];
        let pb = c.states[..c.ground].iter().position(|&s| s == ground)?;
        let ci = self.carrier_of(component.frequency).filter(|_| !component.counter)?;
        let r = at.responses[ci].as_ref().map_or(C64::ZERO, |m| m[(excited - GROUND, pb)]);
        let s = 2.0 * r.norm_sqr();
        let (moved, rate) = self.drive_motion(component, excited, ground, at);
        let own = C64::new(self.energies[excited] - self.energies[ground] - self.carriers[ci].detuning + moved, 0.5 * self.decay[(excited - GROUND, excited - GROUND)].re).abs();
        let doppler = at.doppler[ci].get_or_init(|| self.doppler_sizes(&at.plan, &at.near, ci, at.sample.at.kappa))[pb];
        Some((s, gate(0.0, own, rate).sweep, doppler * doppler, at.populations[ground], r))
    }

    /// The error at a sample of a drive folded with the levels at the atom (SPEC §1.5; units.rs, # The fold's order): its
    /// motion's part on T, ((w‖T2‖/‖T‖)² + a) p_b, and its order's (`drive_parameters`) -- the second order's s p_b on
    /// the populations and on T, where its cluster's fold has no other order; where it has, nothing at the second order
    /// (the order is the cluster's own unit's, `Unit::Second`), and at the fourth what it leaves: `ORDER_RATES` s² p_b on
    /// the populations and (`ORDER_FORCE` s² + the cluster's ground levels' spread over the drive's distance times s)
    /// p_b on T. With the drive's response there, where the cluster holds it (`pair_error` reads it).
    pub(crate) fn drive_error(&self, unit: &Unit, at: &Measured) -> (UnitError, Option<C64>) {
        let Some((s, sweep, doppler, p, response)) = self.drive_parameters(unit, at) else { return (UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 }, None) };
        let cluster = at.sample.cluster;
        let (rates, force) = match (self.fourth[cluster], self.fold_frequency(cluster)) {
            (Some(_), _) => {
                let Unit::Drive { component, excited, ground } = *unit else { unreachable!("a drive's parameters are a drive's") };
                let (moved, _) = self.drive_motion(component, excited, ground, at);
                let own = C64::new(self.energies[excited] - self.energies[ground] - self.tones[component.frequency].detuning + moved, 0.5 * self.decay[(excited - GROUND, excited - GROUND)].re).abs();
                let levels = self.clusters[cluster].states.iter().map(|&b| at.energies[0][b]);
                let spread = levels.clone().fold(f64::NEG_INFINITY, f64::max) - levels.fold(f64::INFINITY, f64::min);
                (super::units::ORDER_RATES * s * s, super::units::ORDER_FORCE * s * s + spread / own * s)
            }
            (None, Some(_)) => (0.0, 0.0),
            (None, None) => (s, s),
        };
        (UnitError { parameter: s, population: rates * p, temperature: (force + doppler + sweep) * p }, Some(response))
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
        let mut error = UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
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
        let c = &self.clusters[at.sample.cluster];
        if let Some(pe) = c.states.iter().position(|&s| s == excited).filter(|_| c.ground > 0 && p[excited] > 0.0) {
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
            let epsilon = gate(0.0, nearest, u.norm_sqr() * off).sweep / own;
            let held: f64 = p[..GROUND].iter().sum();
            let share = |of: f64, all: f64| if all > 0.0 { of / all } else { 0.0 };
            let (rates, shifts) = (2.0 * epsilon * share(p[excited], at.rates) * held, epsilon * share(kept, at.shifts) * held);
            error.parameter = error.parameter.max(s);
            error.population += s + rates;
            error.temperature += s + rates + shifts;
        }
        error
    }

    /// An explicit state's part of a class's second moment's rate (photon.rs, # The absorbed light's momentum): its photons'
    /// rate into the class -- its decay's `decay` = Γ_{e→class} = Σ |C_a[b, e]|² over the class's entries, times its
    /// population `population` in ψ's norm (the state's own |ψ_e|², or a step's mean) -- times the second moment a photon
    /// takes from the response its kept drives would make folded at the atom, T/S: per carrier c the response summed as a
    /// folded jump sums it -- over the carrier's beams l and the cluster's ground amplitudes, coherently --, σ_c = Σ_b ψ_b
    /// Σ_l a_l X_l[e,b]/D_bl, ∇σ_c = Σ_b ψ_b Σ_l (∇a_l − a_l ∇δ_b/D_bl) X_l[e,b]/D_bl, D_bl = δ_b + w_l − iΓ_e/2, δ_b = E_e − E_b
    /// − δ_c at the atom and w_l the beam's Doppler shift at the atom's momentum (module documentation, # First-order
    /// Doppler); S = Σ_c |σ_c|² and T = Σ_c Re[∇σ_c*⊗∇σ_c] (two carriers beat against each other and share the photon by
    /// |σ_c|²). SPEC §1.6, one rule: a running wave's photon has k⊗k, a standing wave's k⊗k tan², none at its antinode, and
    /// towards its node the rate times it stays the dipole force's fluctuations.
    ///
    /// The state's population counts as far as the response's: the rate is Γ_{e→class} min(|ψ_e|², S). An excited
    /// population beyond what the light makes here was not absorbed here -- it came with the atom or with a transient --, and
    /// T/S, the moment of a photon absorbed here, is not its; where the response vanishes (a node) and the population does
    /// not, the quotient has no bound. So the rate times the moment is at most Γ_{e→class} T, the folded drives' own value,
    /// finite everywhere, and no quotient is taken by a response that is nothing. With the complex multiply-adds that made
    /// it (`h` the far light's energies to first order, `Model::far_energies`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn explicit_moment(&self, cluster: usize, amplitudes: &[Amplitude], h: &[Vec<f64>], kappa: [f64; 3], source: usize, decay: f64, population: f64, psi: &[C64]) -> ([[f64; 3]; 3], f64) {
        let mut out = [[0.0; 3]; 3];
        let Origin::Explicit { state: e } = self.sources[source].origin else { return (out, 0.0) };
        let c = &self.clusters[cluster];
        if !(decay > 0.0 && population > 0.0) {
            return (out, 0.0);
        }
        let gamma = self.decay[(e - GROUND, e - GROUND)].re;
        let (mut response, mut terms) = (0.0, 0);
        for cc in &self.carriers {
            let (mut sigma, mut gradient) = (C64::ZERO, [C64::ZERO; 3]);
            for (pb, &b) in c.states[..c.ground].iter().enumerate() {
                if psi[pb] == C64::ZERO || self.level(Component { frequency: cc.frequency, counter: false }, e, b) != Level::Explicit {
                    continue;
                }
                for (l, &bl) in cc.beams.iter().enumerate() {
                    let x = cc.couplings[l][(e, b)];
                    if x == C64::ZERO {
                        continue;
                    }
                    let a = &amplitudes[bl];
                    let d = C64::new(h[0][e] - h[0][b] - cc.detuning + self.doppler_shift(bl, kappa), -gamma / 2.0);
                    let response = x * psi[pb] / d;
                    sigma += a.value * response;
                    for i in 0..3 {
                        gradient[i] += (a.gradient[i] - a.value * (h[1 + i][e] - h[1 + i][b]) / d) * response;
                    }
                    terms += 1;
                }
            }
            response += sigma.norm_sqr();
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j] += (gradient[i].conj() * gradient[j]).re;
                }
            }
        }
        // Γ_{e→class} min(population, S)/S, the quotient taken only by an S that is not below a population above zero.
        let factor = if population <= response { decay * population / response } else { decay };
        for row in out.iter_mut() {
            for x in row.iter_mut() {
                *x *= factor;
            }
        }
        // Each term's response and its gradient's three parts; each carrier's products.
        (out, (10 * terms) as f64 + (10 * self.carriers.len()) as f64)
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
    use crate::model::Sample;
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
        let (_, sweep, _, _, _) = model.drive_parameters(&unit, &measured).unwrap();
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
