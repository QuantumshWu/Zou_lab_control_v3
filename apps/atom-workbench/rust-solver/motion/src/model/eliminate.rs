//! The near light's elimination (SPEC §1.4): second-order Reiter–Sørensen with Brillouin–Wigner dressing and first-order
//! Doppler, for the co-rotating components the units keep near (the near carriers), evaluated exactly at a point with its
//! first derivatives -- the definition the tables interpolate (SPEC §2.3).
//!
//! # The response
//! A near carrier c (`Carrier`) drives V_c(r) = Σ_l a_l(r) X_l e^{−iδ_c t}; its folded part V^f zeroes the kept drives' entries
//! (kept as units, or for free: a drive inside a cluster whose residual beat is exactly zero). The excited amplitude it drives from
//! a ground state b of the cluster follows adiabatically, −R[:,b] x_b, with (internal.md 1.2)
//!
//! ```text
//! R_c[:,b] = (K + Σ_b(z) − z)⁻¹ V_c^f[:,b]      z = E_b + δ_c + Σ_{d≠c} ΔE_b^d      K = H_ee − (i/2) D
//! ΔE_b^d = −Re[V_d^f[:,b]† (K − E_b − δ_d)⁻¹ V_d^f[:,b]]          Σ_b(z) = Σ_{d, a∉C} V_d^f[:,a] V_d^f[:,a]† / (z − Ẽ_a − δ_d)
//! ```
//!
//! H the far light's Hamiltonian at the point (far.rs: the base and every lit frequency's static form), E_b its ground diagonal,
//! Ẽ_a = E_a + Σ_d ΔE_a^d. The other carriers dress b's energy (Brillouin–Wigner: in the denominator, to all orders); the folded
//! channels to the ground states of other clusters (a ∉ C) are the self-energy of the excited states (a strong beam's Autler–Townes
//! dressing of a probe through a folded Raman pair); the channels to the cluster's own states are its dynamics', none of Σ.
//! Within a carrier the dressing takes T̄_d, the beams' coherence kept within a polarization unit (beams of parallel fields:
//! a standing wave's intensity pattern) and averaged between units (SPEC §1.4): V_d V_d† → Σ_u v_u v_u†, v_u = Σ_{l∈u} a_l X^f_l;
//! the units' interference is the dressing's error, measured at the atom. Per beam the response is T_l, R_c = Σ_l a_l T_l.
//!
//! # First-order Doppler
//! Absorbing beam l raises the excited state's energy by w_l = (ħ/m)(k_l·κ + |k_l|²/2); to first order T_l → T_l − w_l T2_l,
//! T2 = (K + Σ − z)⁻¹ T (internal.md 1.4): in the generator and the photons alike. Its error is the second order,
//! (w_l ‖T2_l‖/‖T_l‖)² measured at the atom (units.rs); no clamp, no switch (SPEC §1.4).
//!
//! # The generator's near terms (assemble.rs)
//! Per pair of carriers (c', c) and of their beams (l', l): a_l'* a_l S_l'l on the cluster's ground states, S_l'l =
//! −½(X^f_l'† T_l + T_l'† X^f_l), at the beat δ_c' − δ_c and wave vector k_l − k_l' -- the light shifts, the Raman and Zeeman
//! couplings of the light, the beats of two carriers; an entry whose residual beat in the frames is not zero only where its unit
//! (`Unit::Term`) is kept. The decay −(i/2)Σ J†J is the photons' (photon.rs): per beam the jumps C_a T_l. The explicit states'
//! energies take the folded drives' van Vleck shift ½ Σ_b V^f[e,b] V^f[f,b]* (1/(E_e − E_b − δ_c) + 1/(E_f − E_b − δ_c)). The kept
//! drives enter as they are: a_l X_l[e,b] e^{−iδ_c t} and its adjoint.
//!
//! # Derivatives
//! Everything above is a resolvent of the far light's H(r) and the fields: ∂_i T = −(K + Σ − z)⁻¹ ∂_i(K + Σ − z) T, with ∂_i K, ∂_i E_b
//! from the far light's exact derivatives (far.rs), ∂_i ΔE from the dressing's own resolvent, ∂_i Σ from the fields' and the
//! energies' -- the force's exact definition at the point.

use super::assemble::{NearValues, Plan};
use super::cluster::Edge;
use super::photon::Origin;
use super::reach::{distance, Region};
use super::units::{mixing, swept};
use super::{Beat, Component, Lights, Model, Point, Sample, Unit, UnitError};
use crate::field::Amplitude;
use crate::linalg::{dot3, Lu, Mat, C64};
use crate::request::{GROUND, STATES};
use super::far::EXCITED;
use super::jet;

/// A near carrier: a co-rotating component the units keep near, its beams' couplings and folded couplings, its folded couplings
/// on its field's basis, and its polarization units (beams of parallel fields).
#[derive(Clone, Debug)]
pub(crate) struct Carrier {
    pub(crate) frequency: usize,
    /// δ_c exactly and its value (rad/µs).
    pub(crate) beat: Beat,
    pub(crate) detuning: f64,
    /// Its beams (request indices), their couplings X_l and folded couplings X^f_l (STATES × STATES), their units.
    pub(crate) beams: Vec<usize>,
    pub(crate) couplings: Vec<Mat>,
    pub(crate) folded: Vec<Mat>,
    /// Its folded couplings on its field's basis (field.rs), U^f_k (STATES × STATES): X^f_l = Σ_k γ_lk U^f_k, the responses' sources.
    pub(crate) basis: Vec<Mat>,
    pub(crate) units: Vec<usize>,
}

impl Carrier {
    /// The near carriers of a model: every co-rotating component not folded far (its kept drives' entries are zeroed in X^f
    /// once the clusters are known, `Model::new`). Units: beams whose focus fields are parallel (field.rs `parallel`).
    pub(crate) fn all(model: &Model) -> Vec<Carrier> {
        let mut out = vec![];
        for frequency in 0..model.light.frequencies.len() {
            if model.folded(Component { frequency, counter: false }) {
                continue;
            }
            let f = &model.light.frequencies[frequency];
            let tone = &model.tones[frequency];
            let couplings = tone.couplings[0].clone();
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
                folded: couplings.clone(),
                couplings,
                basis: tone.basis[0].clone(),
                units,
            });
        }
        out
    }
}

/// The elimination's responses at one far light and near amplitudes (`Model::eliminate`): per carrier (empty where unlit), basis
/// vector k of its field and jet part, T_k = (K + Σ − z)⁻¹ U^f_k and T2_k = (K + Σ − z)⁻¹ T_k on the cluster's ground places
/// (EXCITED × n_g) -- a beam's are T_l = Σ_k γ_lk T_k, the responses being linear in the coupling --; and the complex multiply-adds
/// that made them.
pub(crate) struct Responses {
    pub(crate) first: Vec<Vec<Vec<Mat>>>,
    pub(crate) second: Vec<Vec<Vec<Mat>>>,
    pub(crate) work: f64,
}

impl Model {
    /// The far light's Hamiltonian H(r) on all the states, its jet to `order`: the base and every lit frequency's static form.
    pub(crate) fn far_hamiltonian(&self, amplitudes: &[Amplitude], lights: &Lights, order: usize) -> Vec<Mat> {
        let mut h: Vec<Mat> = (0..jet::parts(order)).map(|p| if p == 0 { self.base.clone() } else { Mat::zeros(STATES, STATES) }).collect();
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            let n = f.beams.len();
            for j in 0..n {
                for i in 0..n {
                    if lights.pair(w, j, i) {
                        let c = jet::of(&super::far::pair(&amplitudes[f.beams[j]], &amplitudes[f.beams[i]]), order);
                        jet::add_scaled(&mut h, &c, &tone.pairs[j * n + i]);
                    }
                }
            }
        }
        h
    }

    /// The far light's energies, H(r)'s diagonal to `order` ([part][state]), and their complex multiply-adds: the base's and each
    /// lit beam pair's coefficient times its form's diagonal, STATES per part.
    pub(crate) fn far_energies(&self, amplitudes: &[Amplitude], lights: &Lights, order: usize) -> (Vec<Vec<f64>>, f64) {
        let np = jet::parts(order);
        let mut e: Vec<Vec<f64>> = (0..np).map(|p| (0..STATES).map(|s| if p == 0 { self.base[(s, s)].re } else { 0.0 }).collect()).collect();
        let mut work = 0.0;
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            let n = f.beams.len();
            for x in 0..n * n {
                if lights.pair(w, x / n, x % n) {
                    let c = jet::of(&super::far::pair(&amplitudes[f.beams[x / n]], &amplitudes[f.beams[x % n]]), order);
                    for (part, &z) in e.iter_mut().zip(&c) {
                        for (s, v) in part.iter_mut().enumerate() {
                            *v += (z * tone.pairs[x][(s, s)]).re;
                        }
                    }
                    work += (np * STATES) as f64;
                }
            }
        }
        (e, work)
    }

    /// The complex multiply-adds of `far_hamiltonian`: each lit beam pair's form added to every part of the jet, STATES² each.
    pub(crate) fn far_work(&self, lights: &Lights, order: usize) -> f64 {
        let pairs: usize = self.light.frequencies.iter().enumerate().map(|(w, f)| (0..f.beams.len().pow(2)).filter(|&x| lights.pair(w, x / f.beams.len(), x % f.beams.len())).count()).sum();
        (pairs * jet::parts(order) * STATES * STATES) as f64
    }

    /// Whether a carrier's drive e ← b is kept: as a unit, or for free inside a cluster at a residual beat of exactly zero.
    pub(crate) fn drive_kept(&self, carrier: usize, e: usize, b: usize) -> bool {
        let c = &self.carriers[carrier];
        let component = Component { frequency: c.frequency, counter: false };
        if self.kept.contains(&Unit::Drive { carrier: component, excited: e, ground: b }) {
            return true;
        }
        self.clusters.iter().any(|k| {
            let (pe, pb) = (k.states.iter().position(|&s| s == e), k.states.iter().position(|&s| s == b));
            matches!((pe, pb), (Some(pe), Some(pb)) if k.frames[pe].plus(-1, &k.frames[pb]) == c.beat)
        })
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
        let outside: Vec<usize> = (0..GROUND).filter(|&a| self.membership[a] != cluster).collect();
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
                // The self-energy of the folded channels to the other clusters' ground states.
                for (d, other) in self.carriers.iter().enumerate() {
                    if !lit[d] {
                        continue;
                    }
                    for &a in &outside {
                        let mut den: Vec<f64> = z.iter().zip(dressed(a, None)).map(|(x, y)| x - y).collect();
                        den[0] -= other.detuning;
                        let inverse = jet::reciprocal(&den);
                        for group in self.dressing_groups(other, lights, coherent[d]) {
                            let mut v: Vec<Vec<C64>> = vec![vec![C64::ZERO; EXCITED]; np];
                            work += (group.len() * EXCITED * np) as f64;
                            for &l in &group {
                                let a_l = coefficient(other.beams[l]);
                                for x in 0..EXCITED {
                                    let xv = other.folded[l][(GROUND + x, a)];
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
                            for x in 0..EXCITED {
                                for y in 0..EXCITED {
                                    let vx: Vec<C64> = v.iter().map(|q| q[x]).collect();
                                    let vy: Vec<C64> = v.iter().map(|q| q[y].conj()).collect();
                                    let outer = jet::product(&vx, &vy);
                                    let term = jet::product(&outer, &inverse.iter().map(|&r| C64::real(r)).collect::<Vec<_>>());
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

    /// A carrier's index from its frequency.
    pub(crate) fn carrier_index(&self, frequency: usize) -> usize {
        self.carriers.iter().position(|c| c.frequency == frequency).expect("a near carrier")
    }

    /// Whether a near coupling of two ground places (x ← y, absorbing `from`, emitting `to`, base beat `beat`) enters the
    /// cluster's generator: at a residual beat of exactly zero in the frames (free), or as a kept unit.
    pub(crate) fn term_included(&self, cluster: usize, x: usize, y: usize, from: Component, to: Component, beat: &Beat) -> bool {
        let c = &self.clusters[cluster];
        if beat.plus(1, &c.frames[x]).plus(-1, &c.frames[y]).is_zero() {
            return true;
        }
        self.kept.contains(&term_unit(c.states[x], c.states[y], Some(from), Some(to)))
    }

    /// The near trap light's light-shift bound per beam pair of the trap frequency (potential.rs `Model::envelope`), row-major
    /// over its beams: where the trap's co-rotating component is near, |[X^f_l'† G X^f_l]_p| with ‖G‖ ≤ 2/Γ_min (a resolvent of an
    /// excited block whose decay is at least Γ_min/2 off the real axis) -- the far fold's bound on the folded near drive --, and
    /// zero where it is folded far.
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

    /// The first-order Doppler term's relative size on a ground place y of a plan's cluster (its error's square root, units.rs):
    /// max_l |w_l| ‖T2_l[:, y]‖ / max_l ‖T_l[:, y]‖ over the carrier's beams, T_l = Σ_k γ_lk T_k.
    pub(crate) fn doppler_size(&self, plan: &Plan, near: &NearValues, carrier: usize, y: usize, kappa: [f64; 3]) -> f64 {
        let f = &self.light.frequencies[self.carriers[carrier].frequency];
        let (start, count) = (plan.near.responding().start, plan.near.responses.len());
        let (mut first, mut second) = (0.0_f64, 0.0_f64);
        for (&l, c) in f.beams.iter().zip(&f.coefficients) {
            let (mut t, mut t2) = ([C64::ZERO; EXCITED], [C64::ZERO; EXCITED]);
            for (i, &[of, k, e, place]) in plan.near.responses.iter().enumerate() {
                if of == carrier && place == y {
                    t[e] += c[k] * near.data[start + i];
                    t2[e] += c[k] * near.data[start + count + i];
                }
            }
            first = first.max(crate::linalg::norm_sqr(&t).sqrt());
            second = second.max(self.doppler_shift(l, kappa).abs() * crate::linalg::norm_sqr(&t2).sqrt());
        }
        if first > 0.0 { second / first } else { 0.0 }
    }

    /// The near carriers' folded channels on a cluster at a point (readout.rs): per lit carrier its response there
    /// (`carrier_responses`), the near forms from their table where one serves.
    pub(crate) fn near_channels(&self, cluster: usize, at: &Point, amplitudes: &[Amplitude], lights: &Lights, tables: Option<&super::tabulate::Tables>) -> Vec<super::readout::Channels> {
        let plan = self.plan(cluster, lights);
        if plan.near.len() == 0 {
            return vec![];
        }
        let near = self.near_values(&plan, amplitudes, &[(plan.near.responding(), 0)], tables.and_then(|t| t.of(cluster, lights)));
        self.carrier_responses(&plan, &near, amplitudes, at.kappa)
            .into_iter()
            .zip(&self.carriers)
            .filter_map(|(r, cc)| r.map(|response| super::readout::Channels { nu: cc.beat.clone(), response }))
            .collect()
    }

    /// The near carriers' units: their drives, and the couplings of two ground states (or one state's beat of two carriers) they
    /// make at second order (`Unit::Term`, canonical: a < b, or a = b with from < to).
    pub(crate) fn near_units(&self) -> Vec<Unit> {
        let mut out = vec![];
        for cc in &self.carriers {
            let component = Component { frequency: cc.frequency, counter: false };
            for e in GROUND..STATES {
                for b in 0..GROUND {
                    if cc.couplings.iter().any(|x| x[(e, b)] != C64::ZERO) {
                        out.push(Unit::Drive { carrier: component, excited: e, ground: b });
                    }
                }
            }
        }
        // A carrier's dressing: where its T̄ averages an interference (several polarization units) and enters an elimination --
        // another carrier's denominators, or its own channels' self-energy to the ground states of another cluster (module
        // documentation, # The response).
        for cc in &self.carriers {
            let mut units = cc.units.clone();
            units.sort();
            units.dedup();
            if units.len() > 1 && (self.carriers.len() > 1 || self.clusters.len() > 1) {
                out.push(Unit::Dressing { carrier: Component { frequency: cc.frequency, counter: false } });
            }
        }
        for cp in &self.carriers {
            for cc in &self.carriers {
                let (from, to) = (Component { frequency: cc.frequency, counter: false }, Component { frequency: cp.frequency, counter: false });
                for a in 0..GROUND {
                    for b in 0..GROUND {
                        if a == b && from == to {
                            continue;
                        }
                        let coupled = (GROUND..STATES).any(|e| cp.couplings.iter().any(|x| x[(e, a)] != C64::ZERO) && cc.couplings.iter().any(|x| x[(e, b)] != C64::ZERO));
                        let unit = term_unit(a, b, Some(from), Some(to));
                        if coupled && !out.contains(&unit) {
                            out.push(unit);
                        }
                    }
                }
            }
        }
        out
    }

    /// A kept near unit's edge: a drive joins its states at θ_e − θ_b = δ_c, its strongest beam's wave vector; a coupling of two
    /// ground states at θ_a − θ_b = δ_from − δ_to, its strongest beam pair's k_l − k_l'.
    pub(crate) fn near_edge(&self, unit: &Unit) -> Option<Edge> {
        match unit {
            Unit::Drive { carrier, excited, ground } => {
                let cc = &self.carriers[self.carrier_index(carrier.frequency)];
                let (best, strength) = cc.beams.iter().zip(&cc.couplings).fold((cc.beams[0], 0.0_f64), |(best, s), (&l, x)| {
                    let v = x[(*excited, *ground)].abs();
                    (if v > s { l } else { best }, s.max(v))
                });
                Some(Edge { a: *excited, b: *ground, beat: cc.beat.clone(), wave: self.beam_wave(best), strength })
            }
            Unit::Term { a, b, from: Some(from), to: Some(to) } => {
                let (cc, cp) = (&self.carriers[self.carrier_index(from.frequency)], &self.carriers[self.carrier_index(to.frequency)]);
                let mut best = (0.0_f64, self.no_wave());
                for (&li, xi) in cc.beams.iter().zip(&cc.couplings) {
                    for (&lj, xj) in cp.beams.iter().zip(&cp.couplings) {
                        let size: f64 = (GROUND..STATES).map(|e| xj[(e, *a)].abs() * xi[(e, *b)].abs()).sum();
                        if size > best.0 {
                            best = (size, self.pair_wave(lj, li));
                        }
                    }
                }
                Some(Edge { a: *a, b: *b, beat: cc.beat.plus(-1, &cp.beat), wave: best.1, strength: best.0 })
            }
            _ => None,
        }
    }

    /// A near unit's error at a sample.
    pub(crate) fn near_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        match unit {
            Unit::Coherence { .. } => self.coherence_error(unit, sample),
            Unit::Drive { .. } => self.drive_error(unit, sample),
            Unit::Term { .. } => self.term_error(unit, sample),
            Unit::Dressing { carrier } => self.dressing_error(*carrier, sample),
            _ => unreachable!("the far units' errors are units.rs'"),
        }
    }

    /// A carrier's dressing's error at a sample: every carrier's folded response there with its coherent field in the dressing, R',
    /// against the model's with its units' T̄, R, on the sample's ground amplitudes ψ_g -- the other carriers' through its shift of
    /// their denominators, its own through its channels' self-energy and the dressed energies of the states they lead to --:
    /// δ = max_c ‖(R'_c − R_c)ψ_g‖/‖R_c ψ_g‖, the relative error of the rates and light shifts it moves of the sample's state, zero
    /// where nothing drives it; on the populations and on T δ P_g, P_g the sample's ground population (SPEC §1.5's map of a
    /// relative rate error, as a drive's S_b p_b).
    pub(crate) fn dressing_error(&self, carrier: Component, sample: &Sample) -> UnitError {
        let lights = self.running();
        let amplitudes = self.light.amplitudes(sample.at.r);
        let plan = self.plan(sample.cluster, &lights);
        let mut toggled = self.coherent_dressing();
        let d = self.carrier_index(carrier.frequency);
        toggled[d] = !toggled[d];
        let responses = |coherent: &[bool]| self.carrier_responses(&plan, &self.near_exact(&plan, &amplitudes, 0, coherent), &amplitudes, sample.at.kappa);
        let (model, other) = (responses(&self.coherent_dressing()), responses(&toggled));
        let g = self.clusters[sample.cluster].ground;
        let ground = &sample.psi[..g];
        let mut delta = 0.0_f64;
        for (r, s) in model.iter().zip(&other) {
            let (Some(r), Some(s)) = (r, s) else { continue };
            let (r, s) = (r.apply(ground), s.apply(ground));
            let size: f64 = r.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if size > 0.0 {
                let diff: f64 = r.iter().zip(&s).map(|(&x, &y)| (y - x).norm_sqr()).sum::<f64>().sqrt();
                delta = delta.max(diff / size);
            }
        }
        let total: f64 = sample.psi.iter().map(|z| z.norm_sqr()).sum();
        let weight = ground.iter().map(|z| z.norm_sqr()).sum::<f64>() / total;
        UnitError { parameter: delta, population: delta * weight, temperature: delta * weight }
    }

    /// The bound of the near light's differential shift of two states over a region (the static gates' separations): per lit
    /// carrier (‖u_a‖² + ‖u_b‖²) 2/Γ_min, u_s its coherent field's couplings of ground state s (`Couplings`), the resolvent's norm
    /// at most 2/Γ_min.
    pub(crate) fn near_spread(&self, a: usize, b: usize, region: &Region) -> f64 {
        let column = |c: &Couplings, s: usize| if s < GROUND { c.column[s] } else { 0.0 };
        self.couplings(region).iter().map(|c| column(c, a).powi(2) + column(c, b).powi(2)).sum::<f64>() * 2.0 / self.least_decay()
    }
}

/// The far light's energies over a region (the near gates' separations): per state its value at the centre and the bound of how
/// far it moves within the cell, and per excited state the Gershgorin radius of the far couplings among the excited states.
pub(crate) struct CellEnergies {
    pub(crate) centre: Vec<f64>,
    pub(crate) slack: Vec<f64>,
    pub(crate) radius: Vec<f64>,
}

/// A carrier's coherent couplings over a region (reach.rs `Region`). With u_s = Σ_l a_l X^f_l[:, s] the excited amplitudes its
/// field drives from ground state s -- its beams coherent at the request's phases (SPEC §1.2), what the atom feels at r --, per
/// ground state the bounds over the region of |u_s[e]| per excited state, of ‖u_s‖, and of ‖u_s‖ per polarization unit (its beams
/// alone): the value at the centre plus the cell's radius times the Lipschitz bound Σ_l max|∇a_l| |X^f_l| (|∇a_l| ≤ k_l max|a_l| +
/// max|∇b_l|, reach.rs `Span`), at most every beam at its largest amplitude and in phase; on a point, the value there.
pub(crate) struct Couplings {
    /// [s][e − GROUND].
    pub(crate) entry: Vec<Vec<f64>>,
    /// [s].
    pub(crate) column: Vec<f64>,
    /// [unit][s], over the carrier's distinct polarization units.
    pub(crate) units: Vec<Vec<f64>>,
}

impl Model {
    /// The far light's energies over a region (`CellEnergies`), made once per region, every light on: each beam pair's
    /// coefficient a_l'* a_l within its centre value ± its radius (reach.rs `PairSpan`).
    pub(crate) fn energies<'r>(&self, region: &'r Region) -> &'r CellEnergies {
        region.energies.get_or_init(|| {
            let mut centre: Vec<f64> = (0..STATES).map(|s| self.base[(s, s)].re).collect();
            let mut slack = vec![0.0; STATES];
            let mut off = Mat::from_fn(STATES, STATES, |x, y| if x == y { C64::ZERO } else { self.base[(x, y)] });
            let mut off_slack = Mat::zeros(STATES, STATES);
            for (tone, pairs) in self.tones.iter().zip(&region.pairs) {
                for (p, pair) in tone.pairs.iter().zip(pairs) {
                    let (c0, rad) = (pair.centre, pair.radius);
                    for s in 0..STATES {
                        centre[s] += (c0 * p[(s, s)]).re;
                        slack[s] += rad * p[(s, s)].abs();
                    }
                    for x in GROUND..STATES {
                        for y in GROUND..STATES {
                            if x != y && p[(x, y)] != C64::ZERO {
                                off[(x, y)] += c0 * p[(x, y)];
                                off_slack[(x, y)] += C64::real(rad * p[(x, y)].abs());
                            }
                        }
                    }
                }
            }
            let radius = (0..STATES)
                .map(|x| if x < GROUND { 0.0 } else { (GROUND..STATES).filter(|&y| y != x).map(|y| off[(x, y)].abs() + off_slack[(x, y)].re).sum() })
                .collect();
            CellEnergies { centre, slack, radius }
        })
    }

    /// The carriers' coherent couplings over a region (`Couplings`), made once per region.
    pub(crate) fn couplings<'r>(&self, region: &'r Region) -> &'r [Couplings] {
        region.couplings.get_or_init(|| {
            let spans = &region.spans;
            // The bound over the region of |Σ_{l ∈ beams} a_l x_l| from its centre value, its Lipschitz bound Σ |x_l| max|∇a_l| and
            // its phase-blind bound Σ |x_l| max|a_l| (an unbounded region: the last).
            let bound = |centre: f64, lipschitz: f64, blind: f64| if region.rho.is_finite() { (centre + region.rho * lipschitz).min(blind) } else { blind };
            self.carriers
                .iter()
                .map(|cc| {
                    let mut units: Vec<usize> = cc.units.clone();
                    units.sort();
                    units.dedup();
                    let mut entry = vec![vec![0.0; EXCITED]; GROUND];
                    let mut column = vec![0.0; GROUND];
                    let mut per_unit = vec![vec![0.0; GROUND]; units.len()];
                    for s in 0..GROUND {
                        // Per polarization unit and in all: the centre amplitudes, and per excited state the Lipschitz and blind sums.
                        let mut all = (vec![C64::ZERO; EXCITED], vec![0.0; EXCITED], vec![0.0; EXCITED], 0.0, 0.0);
                        for (ui, &unit) in units.iter().enumerate() {
                            let (mut u, mut lipschitz, mut blind) = (vec![C64::ZERO; EXCITED], 0.0, 0.0);
                            for (l, x) in cc.folded.iter().enumerate().filter(|(l, _)| cc.units[*l] == unit) {
                                let span = &spans[cc.beams[l]];
                                let norm = (0..EXCITED).map(|e| x[(GROUND + e, s)].norm_sqr()).sum::<f64>().sqrt();
                                if norm == 0.0 {
                                    continue;
                                }
                                for e in 0..EXCITED {
                                    let v = x[(GROUND + e, s)];
                                    u[e] += span.centre * v;
                                    all.0[e] += span.centre * v;
                                    all.1[e] += v.abs() * span.lipschitz();
                                    all.2[e] += v.abs() * span.most;
                                }
                                lipschitz += norm * span.lipschitz();
                                blind += norm * span.most;
                            }
                            per_unit[ui][s] = bound(u.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(), lipschitz, blind);
                            all.3 += lipschitz;
                            all.4 += blind;
                        }
                        for e in 0..EXCITED {
                            entry[s][e] = bound(all.0[e].abs(), all.1[e], all.2[e]);
                        }
                        column[s] = bound(all.0.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(), all.3, all.4);
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
        let energies = self.energies(region);
        let delta = self.carriers[d].detuning;
        let mut least = f64::INFINITY;
        for e in GROUND..STATES {
            let lo = energies.centre[e] - energies.slack[e] - energies.centre[b] - energies.slack[b] - delta;
            let hi = energies.centre[e] + energies.slack[e] - energies.centre[b] + energies.slack[b] - delta;
            let (x, gamma) = (distance(lo, hi), self.decay[(e - GROUND, e - GROUND)].re);
            least = least.min((x * x + gamma * gamma / 4.0).sqrt() - energies.radius[e]);
        }
        let dissipation = 2.0 / self.least_decay();
        if least > 0.0 {
            dissipation.min(1.0 / least)
        } else {
            dissipation
        }
    }

    /// The carriers' dressings ΔE_b^d of the ground states over a region, [carrier][state], made once per region: Σ over the
    /// carrier's polarization units of ‖u_b‖² (`Couplings`; T̄ keeps each unit's beams coherent) times its resolvent's bound
    /// (`resolvent`).
    fn dressings<'r>(&self, region: &'r Region) -> &'r [Vec<f64>] {
        region.dressings.get_or_init(|| {
            let couplings = self.couplings(region);
            (0..self.carriers.len()).map(|d| (0..GROUND).map(|b| couplings[d].units.iter().map(|u| u[b] * u[b]).sum::<f64>() * self.resolvent(d, b, region)).collect()).collect()
        })
    }

    /// The bound over a region of |∇ Σ_pairs c_l'l w(P_l'l)|, the far light's static forms each weighted by `w` (an entry, a
    /// diagonal's difference): its value at the centre plus the cell's radius times Σ |w| max|∇∇c_l'l|, at most Σ |w| max|∇c_l'l|
    /// (reach.rs `PairSpan`); on a point, its value there (rad/µs/µm).
    pub(crate) fn far_gradient(&self, region: &Region, w: &dyn Fn(&Mat) -> C64) -> f64 {
        let (mut g, mut curvature, mut blind) = ([C64::ZERO; 3], 0.0, 0.0);
        for (tone, pairs) in self.tones.iter().zip(&region.pairs) {
            for (p, pair) in tone.pairs.iter().zip(pairs) {
                let x = w(p);
                if x == C64::ZERO {
                    continue;
                }
                for i in 0..3 {
                    g[i] += pair.gradient[i] * x;
                }
                curvature += x.abs() * pair.hessian;
                blind += x.abs() * pair.slope;
            }
        }
        if !region.rho.is_finite() {
            return blind;
        }
        (g.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() + region.rho * curvature).min(blind)
    }

    /// The bound of |∇(E_x − E_y)| of the far light's diagonal over a region (rad/µs/µm, `far_gradient`).
    pub(crate) fn energy_slope_over(&self, x: usize, y: usize, region: &Region) -> f64 {
        self.far_gradient(region, &|p: &Mat| p[(x, x)] - if y == x { C64::ZERO } else { p[(y, y)] })
    }

    /// The bound over a region of |∇| of a carrier's light shift of ground state b (module documentation, # Derivatives): over its
    /// beam pairs -- all of them, or those within one polarization unit (its dressing, T̄) -- Σ max|∇(a_l'* a_l)| ‖X^f_l'[:,b]‖
    /// ‖X^f_l[:,b]‖ ‖R‖, plus ‖u_b‖² L_b ‖R‖², the resolvent's derivative at most the slope L_b of the levels the column reaches (b's
    /// and its excited states', `energy_slope_over`) times its square (‖u_b‖ in all, or per unit, `Couplings`); ‖R‖ the dressing's
    /// resolvent bound (`resolvent`), or the dissipation's 2/Γ_min for the shift in the generator (its self-energy and dressing
    /// unbounded here).
    pub(crate) fn shift_slope(&self, carrier: usize, b: usize, region: &Region, dressing: bool) -> f64 {
        let cc = &self.carriers[carrier];
        let couplings = &self.couplings(region)[carrier];
        let columns: Vec<f64> = cc.folded.iter().map(|m| (GROUND..STATES).map(|e| m[(e, b)].norm_sqr()).sum::<f64>().sqrt()).collect();
        if columns.iter().all(|&c| c == 0.0) {
            return 0.0;
        }
        let levels = (GROUND..STATES)
            .filter(|&e| cc.folded.iter().any(|m| m[(e, b)] != C64::ZERO))
            .map(|e| self.energy_slope_over(e, b, region))
            .fold(0.0, f64::max);
        let resolvent = if dressing { self.resolvent(carrier, b, region) } else { 2.0 / self.least_decay() };
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
                    row += self.far_gradient(region, &|p: &Mat| p[(x, y)]);
                }
                if x < GROUND {
                    row += (0..self.carriers.len()).map(|c| self.shift_slope(c, x, region, false)).sum::<f64>();
                }
                most = most.max(row);
            }
            let gamma_max = (0..EXCITED).map(|x| self.decay[(x, x)].re).fold(0.0, f64::max);
            most + k_max * gamma_max / 2.0
        })
    }

    /// A ground state's folded drives' saturation and sweep bounds over a region (module documentation of units.rs), made once per
    /// region and ground state: per lit carrier and excited state, v_e = |Σ_l a_l X^f_l[e,b]| (`Couplings`) over |ΔE_e| ≥
    /// √(d² + Γ_e²/4), d the distance of E_e − z_b − k·v from zero over the region and its speeds less the excited states'
    /// Gershgorin radius and the self-energy's bound; S = 2 Σ v_e²/|ΔE_e|², a = max (v L + k a_max)/|ΔE_e|², L the slope of
    /// E_e − E_b and of the other carriers' dressings of b.
    pub(crate) fn drive_bound(&self, b: usize, cluster: usize, region: &Region) -> (f64, f64) {
        *region.drives[b].get_or_init(|| {
            let speed = region.speed;
            let energies = self.energies(region);
            let couplings = self.couplings(region);
            let dressings = self.dressings(region);
            let outside: Vec<usize> = (0..GROUND).filter(|&a| self.membership[a] != cluster).collect();
            let dress: Vec<f64> = (0..GROUND).map(|a| dressings.iter().map(|d| d[a]).sum()).collect();
            let acceleration = self.hbar_over_mass * self.force_over(cluster, region);
            let (mut s, mut a) = (0.0_f64, 0.0_f64);
            for (ci, carrier) in self.carriers.iter().enumerate() {
                let others: f64 = dressings.iter().enumerate().filter(|&(d, _)| d != ci).map(|(_, d)| d[b]).sum();
                let others_slope: f64 = (0..self.carriers.len()).filter(|&d| d != ci).map(|d| self.shift_slope(d, b, region, true)).sum();
                let (z_lo, z_hi) = (energies.centre[b] - energies.slack[b] + carrier.detuning - others, energies.centre[b] + energies.slack[b] + carrier.detuning + others);
                let k = carrier.beams.iter().map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max);
                // The self-energy's bound per excited state: its Gershgorin row, Σ_{d, a∉C} |u_a[e]| ‖u_a‖/|z − Ẽ_a − δ_d|, u_a the
                // folded channel's coupling (only the states a channel reaches move).
                let mut sigma = vec![0.0; STATES];
                for (d, cd) in self.carriers.iter().zip(couplings) {
                    for &o in &outside {
                        let (lo, hi) = (z_lo - energies.centre[o] - energies.slack[o] - dress[o] - d.detuning, z_hi - energies.centre[o] + energies.slack[o] + dress[o] - d.detuning);
                        let den = distance(lo, hi);
                        let size = cd.column[o];
                        if size == 0.0 {
                            continue;
                        }
                        for e in GROUND..STATES {
                            let reach = cd.entry[o][e - GROUND];
                            if reach > 0.0 {
                                sigma[e] += reach * size / den;
                            }
                        }
                    }
                }
                for e in GROUND..STATES {
                    let v = couplings[ci].entry[b][e - GROUND];
                    if v == 0.0 {
                        continue;
                    }
                    let (lo, hi) = (energies.centre[e] - energies.slack[e] - z_hi - k * speed, energies.centre[e] + energies.slack[e] - z_lo + k * speed);
                    let d = (distance(lo, hi) - energies.radius[e] - sigma[e]).max(0.0);
                    let gamma = self.decay[(e - GROUND, e - GROUND)].re;
                    let width = d * d + gamma * gamma / 4.0;
                    s += 2.0 * v * v / width;
                    let slope = self.energy_slope_over(e, b, region) + others_slope;
                    a = a.max((speed * slope + k * acceleration) / width);
                }
            }
            (s, a)
        })
    }

    /// A folded drive's small parameters at a sample (SPEC §1.5): its own saturation s = 2|R_c[e,b]|² at the point (R = Σ_l a_l T̃_l),
    /// its sweep a = |v·∇(E_e − E_b) + k·v̇|/|ΔE|² (ΔE = E_e − E_b − δ_c − k·v − iΓ_e/2, k its strongest beam's, v̇ the force on the
    /// state at the point over m, model.rs `Generator::force`), its Doppler order's (w‖T2‖/‖T‖)², and its ground state's
    /// population p_b; None where the sample's cluster does not hold its ground state.
    pub(crate) fn drive_parameters(&self, unit: &Unit, sample: &Sample) -> Option<(f64, f64, f64, f64)> {
        let Unit::Drive { carrier, excited, ground } = unit else { unreachable!() };
        let c = &self.clusters[sample.cluster];
        let pb = c.states[..c.ground].iter().position(|&s| s == *ground)?;
        let p = self.state_populations(sample)[*ground];
        let lights = self.running();
        let amplitudes = self.light.amplitudes(sample.at.r);
        let plan = self.plan(sample.cluster, &lights);
        let near = self.near_exact(&plan, &amplitudes, 0, &self.coherent_dressing());
        let ci = self.carrier_index(carrier.frequency);
        let cc = &self.carriers[ci];
        let r = self.carrier_responses(&plan, &near, &amplitudes, sample.at.kappa)[ci].as_ref().map_or(C64::ZERO, |m| m[(excited - GROUND, pb)]);
        let s = 2.0 * r.norm_sqr();
        let h = &self.far_energies(&amplitudes, &lights, 1).0;
        let v: [f64; 3] = sample.at.kappa.map(|k| k * self.hbar_over_mass);
        let force = self.assemble(sample.cluster, &sample.at, &lights).force(&sample.psi, &|_| 1.0);
        let strongest = cc.beams.iter().zip(&cc.couplings).max_by(|x, y| x.1[(*excited, *ground)].abs().total_cmp(&y.1[(*excited, *ground)].abs())).map(|(&l, _)| l).unwrap();
        let k = self.light.beams[strongest].wave_vector();
        let delta = h[0][*excited] - h[0][*ground] - cc.detuning - dot3(k, v);
        let rate: f64 = (0..3).map(|i| v[i] * (h[1 + i][*excited] - h[1 + i][*ground])).sum::<f64>() + self.hbar_over_mass * dot3(k, force);
        let gamma = self.decay[(excited - GROUND, excited - GROUND)].re;
        let sweep = rate.abs() / (delta * delta + gamma * gamma / 4.0);
        Some((s, sweep, self.doppler_size(&plan, &near, ci, pb, sample.at.kappa).powi(2), p))
    }

    /// A folded drive's error at a sample (SPEC §1.5): on the populations s p_b, on T (√2 s + (w‖T2‖/‖T‖)² + a) p_b
    /// (`drive_parameters`).
    pub(crate) fn drive_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        match self.drive_parameters(unit, sample) {
            None => UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 },
            Some((s, sweep, doppler, p)) => UnitError { parameter: s, population: s * p, temperature: (std::f64::consts::SQRT_2 * s + doppler + sweep) * p },
        }
    }

    /// A near coupling's (`Unit::Term`) bounds over a region, its saturation S = 2m² and its sweep a: |X_ab| ≤ ½ Σ_e |u_to,a[e]|
    /// |u_from,b[e]| (1/σ_eb + 1/σ'_ea + Γ_e/(σ_eb σ'_ea)), the carriers' coherent couplings (`Couplings`) and σ the drives'
    /// denominators' bounds; its beat Ω = E_a − E_b + δ_to − δ_from over the region, less the Doppler shift of its beam pairs' wave
    /// vectors at the region's speeds (at most 2k v) and the near light's differential shift; and `swept` (units.rs) of
    /// v |∇(E_a − E_b)| + 2k a_max over that separation -- the two states' far energies and near shifts (one state's beat of two
    /// carriers moves with neither), the acceleration in their clusters. On a point, its measured values there. Made once per
    /// region (reach.rs `Region::term`).
    pub(crate) fn term_bound(&self, a: usize, b: usize, from: Component, to: Component, region: &Region) -> (f64, f64) {
        region.term((a, b, Some(from), Some(to)), || self.term_bound_made(a, b, from, to, region))
    }
    fn term_bound_made(&self, a: usize, b: usize, from: Component, to: Component, region: &Region) -> (f64, f64) {
        let energies = self.energies(region);
        let couplings = self.couplings(region);
        let (fi, ti) = (self.carrier_index(from.frequency), self.carrier_index(to.frequency));
        let (cf, ct) = (&self.carriers[fi], &self.carriers[ti]);
        let sigma = |c: &Carrier, g: usize, e: usize| -> f64 {
            let (lo, hi) = (energies.centre[e] - energies.slack[e] - energies.centre[g] - energies.slack[g] - c.detuning, energies.centre[e] + energies.slack[e] - energies.centre[g] + energies.slack[g] - c.detuning);
            let d = (distance(lo, hi) - energies.radius[e]).max(0.0);
            let gamma = self.decay[(e - GROUND, e - GROUND)].re;
            (d * d + gamma * gamma / 4.0).sqrt()
        };
        let mut size = 0.0;
        for e in GROUND..STATES {
            let (vf, vt) = (couplings[fi].entry[b][e - GROUND], couplings[ti].entry[a][e - GROUND]);
            if vf == 0.0 || vt == 0.0 {
                continue;
            }
            let (s1, s2) = (sigma(cf, b, e), sigma(ct, a, e));
            let gamma = self.decay[(e - GROUND, e - GROUND)].re;
            size += 0.5 * vt * vf * (1.0 / s1 + 1.0 / s2 + gamma / (s1 * s2));
        }
        let beat = self.omega(&ct.beat.plus(-1, &cf.beat));
        let (lo, hi) = (energies.centre[a] - energies.slack[a] - energies.centre[b] - energies.slack[b] + beat, energies.centre[a] + energies.slack[a] - energies.centre[b] + energies.slack[b] + beat);
        let k = cf.beams.iter().chain(&ct.beams).map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max);
        if size == 0.0 {
            return (0.0, 0.0);
        }
        let separation = distance(lo, hi) - 2.0 * k * region.speed - self.near_spread(a, b, region);
        let m = mixing(size, separation, 0.0);
        let slope = if a == b { 0.0 } else { self.energy_slope_over(a, b, region) + self.near_slope(a, region) + self.near_slope(b, region) };
        let acceleration = self.acceleration(region, &|c| c.states.contains(&a) || c.states.contains(&b));
        (2.0 * m * m, swept(region.speed * slope + 2.0 * k * acceleration, separation))
    }

    /// A near coupling's error at a sample: m² (p_a + p_b) with m its magnitude over its beat at the point and the sample's
    /// velocity (the beat moved by the strongest beam pair's q·v): its bound on the point (`term_bound`).
    pub(crate) fn term_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        let Unit::Term { a, b, from: Some(from), to: Some(to) } = unit else { unreachable!() };
        let speed = self.hbar_over_mass * crate::linalg::norm3(sample.at.kappa);
        let m2 = 0.5 * self.term_bound(*a, *b, *from, *to, &self.point_region(sample.at.r, speed)).0;
        let p = self.state_populations(sample);
        UnitError { parameter: m2, population: m2 * (p[*a] + p[*b]), temperature: m2 * (p[*a] + p[*b]) }
    }

    /// The explicit sources' recoil terms of a photon (photon.rs): the explicit state e's photon amplitudes `own`, shared among its
    /// kept drives (carrier c, beam l, ground b of the cluster) by |a_l X_l[e,b] ψ_b|², each with its response's log-gradient
    /// ∇ln σ = ∇ln a_l − ∇δ/(δ − iΓ_e/2), δ = E_e − E_b − δ_c + w_l with the beam's Doppler shift at the atom's momentum (module
    /// documentation, # First-order Doppler) (SPEC §1.6: the explicit state's photon takes the second moment its drives' responses
    /// give it, as the same transition folded would); with the complex multiply-adds that made them (`h` the far light's energies
    /// to first order, `Model::far_energies`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn explicit_recoil(&self, cluster: usize, amplitudes: &[Amplitude], h: &[Vec<f64>], kappa: [f64; 3], source: usize, own: [Vec<C64>; 3], psi: &[C64]) -> (Vec<([Vec<C64>; 3], [C64; 3])>, f64) {
        let Origin::Explicit { state: e } = self.sources[source].origin else { return (vec![], 0.0) };
        let c = &self.clusters[cluster];
        let gamma = self.decay[(e - GROUND, e - GROUND)].re;
        let mut drives = vec![];
        for (ci, cc) in self.carriers.iter().enumerate() {
            for (l, &bl) in cc.beams.iter().enumerate() {
                for (pb, &b) in c.states[..c.ground].iter().enumerate() {
                    if cc.couplings[l][(e, b)] == C64::ZERO || !self.drive_kept(ci, e, b) {
                        continue;
                    }
                    let a = &amplitudes[bl];
                    let weight = (a.value * cc.couplings[l][(e, b)] * psi[pb]).norm_sqr();
                    if weight == 0.0 {
                        continue;
                    }
                    let delta = h[0][e] - h[0][b] - cc.detuning + self.doppler_shift(bl, kappa);
                    let g: [C64; 3] = std::array::from_fn(|i| a.gradient[i] / a.value - C64::real(h[1 + i][e] - h[1 + i][b]) / C64::new(delta, -gamma / 2.0));
                    drives.push((weight, g));
                }
            }
        }
        let total: f64 = drives.iter().map(|d| d.0).sum();
        // Each drive's weight and log-gradient, its share of the amplitudes on every axis.
        let work = drives.len() as f64 * (5.0 + (3 * own[0].len()) as f64);
        let shares = drives
            .into_iter()
            .map(|(w, g)| {
                let share = (w / total).sqrt();
                (own.clone().map(|v| v.into_iter().map(|z| z * share).collect()), g)
            })
            .collect();
        (shares, work)
    }
}

/// A coupling of two ground states in canonical form (`Unit::Term`): the pair (a ← b absorbing `from`, emitting `to`) and its
/// adjoint (b ← a absorbing `to`, emitting `from`) are one unit, named a < b, or for a = b with from < to.
pub(crate) fn term_unit(a: usize, b: usize, from: Option<Component>, to: Option<Component>) -> Unit {
    if a < b || (a == b && from <= to) {
        Unit::Term { a, b, from, to }
    } else {
        Unit::Term { a: b, b: a, from: to, to: from }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
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
        let kept = [Unit::Far { component: Component { frequency: probe, counter: false } }];
        let (b, e) = (7, 31);
        let r0 = [0.4, 0.0, 0.0];
        let detuning = |model: &Model, r: [f64; 3]| {
            let h = model.far_hamiltonian(&model.light.amplitudes(r), &model.running(), 1);
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
        let unit = Unit::Drive { carrier: Component { frequency: probe, counter: false }, excited: e, ground: b };
        let (_, sweep, _, _) = model.drive_parameters(&unit, &sample).unwrap();
        let force = model.assemble(sample.cluster, &sample.at, &model.running()).force(&sample.psi, &|_| 1.0);
        let k = model.light.beams[model.carriers[0].beams[0]].wave_vector();
        let expected = (speed * slope + model.hbar_over_mass * dot3(k, force)).abs() / (gamma * gamma / 4.0);
        assert!((sweep - expected).abs() <= 1e-9 * expected, "the measured a {sweep} against |v·∂Δ + k·v̇|/(Γ/2)² = {expected}");
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
