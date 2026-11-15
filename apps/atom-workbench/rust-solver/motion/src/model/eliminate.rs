//! The elimination with the levels at the atom (SPEC §1.4): the one fold (elim.rs `fold`; ELIMINATION §2–§3) with
//! first-order Doppler, for the drives folded at that level (model.rs `Level::Atom`), evaluated exactly at a point with its
//! first derivatives -- the definition the tables interpolate (SPEC §2.3).
//!
//! # The response
//! A carrier c (`Carrier`: a co-rotating component with a drive not bare) drives V_c(r) = Σ_l a_l(r) X_l e^{−iδ_c t}; V^f here
//! is the part of it folded with the levels at the atom -- its bare drives' part is a constant (far.rs), its explicit drives
//! are the generator's. The excited amplitude it drives from a ground state b of the cluster follows adiabatically,
//! −R[:,b] x_b, R the wave operator of the cluster's fold at the point (`Model::eliminate`): P the cluster's states in its
//! frames, every lit carrier a class whose beams are one coherent field at the atom (ELIMINATION v2 2), the levels the far
//! light's H(r) (`Model::far_hamiltonian`: the base and every lit frequency's static form), and the response
//! self-consistent at its replica -- the other carriers' shifts of the levels and the folded channels' self-energy are in
//! it to all orders (ELIMINATION §2–§3). An excited state a kept drive makes explicit is one of P's: a drive folded onto it
//! is a Raman pair's leg, read across the two-photon distance (far.rs, # A fold onto an explicit state; model.rs
//! `Model::settled`). Per beam the response is T_l, R_c = Σ_l a_l T_l.
//!
//! # First-order Doppler
//! Absorbing beam l raises the excited state's energy by w_l = (ħ/m)(k_l·κ + |k_l|²/2); to first order T_l → T_l − w_l T2_l,
//! T2 the response's next order in the excited levels' energy (internal.md 1.4): in the generator and the photons alike.
//! Its error, the second order, is in criterion one's remainder over the frequency range the moving atom spans (elim.rs
//! `errors`, Δ = max|Ω| + 2 k_max v); no clamp, no switch (SPEC §1.4).
//!
//! # The fold's terms: one contraction
//! A folded drive leaves the ground states two things, both made of its response (`shift`, and the response itself): with an emitted leg
//! (coupling V', response R') and an absorbed one (V, R), the ground block −½(V'†R + R'†V) -- the light shifts, the Raman and
//! Zeeman couplings of the light, the beats of two components --, and the excited amplitude R itself, whose photons C_a R
//! and whose own product −(i/2) R†D R, the decay, are one object (assemble.rs, # The excited amplitude; photon.rs). A response is (K − z)⁻¹V with the levels it is taken at: the bare ones (K₀ = H0_ee − (i/2)D,
//! z = E_b + ν: a constant, far.rs), or those at the atom (`Model::eliminate`: the far light's H(r), the other carriers'
//! dressing, the folded channels' self-energy -- the bare folds' results are its inputs, so they are made first). One
//! definition for both: what differs is the response handed in.
//!
//! # The generator's terms at the atom (assemble.rs)
//! Per pair of frequencies (ω', ω) and of their beams (l', l): a_l'* a_l S_l'l on the cluster's ground states, S_l'l the
//! contraction of the two beams' legs -- each its frequency's every folded coupling, bare ones too, with its response at the
//! atom (none where the frequency is no carrier): what the bare drives' constants leave of the fold (far.rs) --, at the beat
//! δ_ω' − δ_ω and wave vector k_l − k_l'; an entry whose residual beat in the frames is not zero only where its unit
//! (`Unit::Term`) is kept. Per beam the responses T_l, the excited amplitude's (assemble.rs, # The excited amplitude). The explicit states' energies take the van Vleck shift of the drives
//! folded here, ½ Σ_b (V^f[e,b] V[f,b]*/(E_e − E_b − δ_c) + V[e,b] V^f[f,b]*/(E_f − E_b − δ_c)), each drive's own denominator
//! against the other's folded coupling V. The explicit drives enter as they are: a_l X_l[e,b] e^{−iδ_c t} and its adjoint.
//!
//! # Derivatives
//! The elimination's derivatives in the position are central differences of the same function along its inputs' own Taylor
//! path (`Model::eliminate`): exact but for the difference's O(δ²) and its rounding -- the force's definition at the point.

use super::assemble::Plan;
use super::{Beat, Component, Level, Lights, Model, Unit};
use crate::field::Amplitude;
use crate::linalg::{Mat, C64};
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

/// A carrier: a co-rotating component with a drive not bare (derived from the drives' levels, model.rs `Level`; never listed).
/// Its beams' couplings; of its drives folded with the levels at the atom the couplings per beam and on its field's basis
/// -- the responses' sources --; and its polarization units (beams of parallel fields).
#[derive(Clone, Debug)]
pub(crate) struct Carrier {
    pub(crate) frequency: usize,
    /// δ_c exactly and its value (rad/µs).
    pub(crate) beat: Beat,
    pub(crate) detuning: f64,
    /// Its beams (request indices), the couplings of its drives folded at the atom X^f_l (STATES × STATES).
    pub(crate) beams: Vec<usize>,
    pub(crate) folded: Vec<Mat>,
    /// The same on its field's basis (field.rs), U^f_k (STATES × STATES): X^f_l = Σ_k γ_lk U^f_k, the responses' sources.
    pub(crate) basis: Vec<Mat>,
    /// Its explicit drives' couplings per beam (STATES × STATES).
    pub(crate) explicit: Vec<Mat>,
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
            out.push(Carrier {
                frequency,
                beat: tone.beat.clone(),
                detuning: tone.detuning,
                beams: f.beams.clone(),
                folded: tone.couplings[0].iter().map(at_the_atom).collect(),
                basis: tone.basis[0].iter().map(at_the_atom).collect(),
                explicit,
            });
        }
        out
    }
}

/// The elimination's responses on a cluster at a point (`Model::eliminate`: the one elimination's series about the
/// reference, elim.rs `Reference`), jets in the position: per carrier (empty where unlit), basis vector k of its field
/// and jet part, R_k = −R₀ U_k and R2_k = R₀² U_k on every place of the cluster (EXCITED × n) -- the order-0 response
/// and the first-order Doppler partner, R₀(y)² U_k on place y's column; a beam's Σ_k γ_lk of them; the sign the
/// contraction's (`shift`) --; where the evaluation takes a first order (the series' or the normalization's) and the
/// cluster's places have more than one reference class (`Model::references`), the products R₀(z_a)R₀(z_y) U_k of a
/// class's reference and the place's on place y's column, [carrier][k][class][part] (elim.rs `Reference::mixed`, at
/// each class's first place); per jet part the reference Z₀ itself, its every place's z_y; and the complex
/// multiply-adds that made them.
pub(crate) struct Responses {
    pub(crate) first: Vec<Vec<Vec<Mat>>>,
    pub(crate) second: Vec<Vec<Vec<Mat>>>,
    pub(crate) mixed: Vec<Vec<Vec<Vec<Mat>>>>,
    pub(crate) reference: Vec<Vec<f64>>,
    pub(crate) work: f64,
}

impl Model {
    /// Per frequency, row-major over its beams, whether a pair of them enters the levels the eliminations take (module
    /// documentation, # The response): a pair the lights hold (model.rs `Lights::pair`), and of a carrier's frequency one
    /// whose interference its dressing keeps -- two beams of one polarization unit, or any two where its dressing is coherent.
    pub(crate) fn level_pairs(&self, lights: &Lights) -> Vec<Vec<bool>> {
        (0..self.light.frequencies.len()).map(|w| (0..self.light.frequencies[w].beams.len().pow(2)).map(|x| self.level_pair(lights, w, x)).collect()).collect()
    }

    /// Whether pair x of frequency w's beams (row-major) enters the levels the elimination's reference is taken at
    /// (`level_pairs`): a pair the lights hold of the far light -- a frequency that is no carrier. A carrier's own static
    /// form (its bare drives' shifts and its outside form) is bilinear in the near field: the generator's constant
    /// (assemble.rs), and on the excited block what the reference leaves out, booked in the series' ratio (elim.rs
    /// `Model::series_at`).
    fn level_pair(&self, lights: &Lights, w: usize, x: usize) -> bool {
        let n = self.light.frequencies[w].beams.len();
        lights.pair(w, x / n, x % n) && self.carrier_of(w).is_none()
    }

    /// The levels at the atom as the eliminations take them: H(r) on all the states, its jet to `order` -- the base and every
    /// lit frequency's static form, a carrier's frequency's as its dressing takes its light (`level_pairs`).
    pub(crate) fn far_hamiltonian(&self, amplitudes: &[Amplitude], lights: &Lights, order: usize) -> Vec<Mat> {
        let mut h: Vec<Mat> = (0..jet::parts(order)).map(|p| if p == 0 { self.base.clone() } else { Mat::zeros(STATES, STATES) }).collect();
        let pairs = self.level_pairs(lights);
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
    pub(crate) fn far_energies(&self, amplitudes: &[Amplitude], lights: &Lights, order: usize, e: &mut Vec<Vec<f64>>) -> f64 {
        let np = jet::parts(order);
        e.resize_with(np, Vec::new);
        for (p, part) in e.iter_mut().enumerate() {
            part.clear();
            part.extend((0..STATES).map(|s| if p == 0 { self.base[(s, s)].re } else { 0.0 }));
        }
        let mut work = 0.0;
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            let n = f.beams.len();
            for x in (0..n * n).filter(|&x| self.level_pair(lights, w, x)) {
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
    pub(crate) fn far_work(&self, lights: &Lights, order: usize) -> f64 {
        let pairs: usize = self.level_pairs(lights).iter().map(|of| of.iter().filter(|pair| **pair).count()).sum();
        (pairs * jet::parts(order) * STATES * STATES) as f64
    }

    /// The near light's elimination on a cluster at a far Hamiltonian `h` (its jet: the constant forms' H at the point) and the
    /// beams' amplitudes (their jets), jets to `order`: the one elimination (elim.rs `fold`) at the point -- P the
    /// cluster's states in its frames, every lit carrier a class whose beams are coherent at the point, the carriers'
    /// drives folded at their replicas or explicit where the frames make them static --, its responses on the carriers'
    /// field bases, normalized (`Responses`) -- with the products of two places' resolvents each reference class's first
    /// order and the landing read (`Responses::mixed`) where `mixes`: where the plan lays the forms they make
    /// (assemble.rs `Plan::mixes`). Its derivatives in the position, to `order`, are the exact ones by the
    /// resolvent's identity (elim.rs `Reference::respond`) from the far Hamiltonian's jet `h`: the reference z₀'s and
    /// the excited block's. (Central differences of the same function, as they were, read the forms' rounding -- that of
    /// their terms, not of their small value -- times 1/δ²: on R4_LIN_1's cluster of three ground states the potential's
    /// curvature along the tweezer's axis +0.166 rad/µs/µm² where it is −3.5·10⁻⁴, its weak axis's own 0.033.)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eliminate(&self, cluster: usize, h: &[Mat], lights: &Lights, order: usize, mixes: bool) -> Responses {
        use super::elim;
        let parts = jet::parts(order);
        let c = &self.clusters[cluster];
        let (n, g) = (c.states.len(), c.ground);
        let grounds: Vec<usize> = c.states[..g].to_vec();
        let excited: Vec<usize> = (GROUND..STATES).collect();
        let levels = elim::Levels {
            ground: h[0].block(&grounds, &grounds),
            excited: h[0].block(&excited, &excited),
            decay: self.decay.clone(),
        };
        // The reference Z₀ = diag(z_y): each place's level in its frame with the far light alone (`h`'s diagonal, no
        // near field in it) and its kind's constant centre c of the near light's shift in the cluster, a ground place's
        // or an explicit place's (elim.rs `Series::centre_of`); its jet the far levels' (the frames and c are constants).
        let frames: Vec<f64> = c.frames.iter().map(|f| self.omega(f)).collect();
        let reference: Vec<Vec<f64>> = (0..parts).map(|p| (0..n).map(|y| h[p][(c.states[y], c.states[y])].re + if p == 0 { self.series.centre_of(cluster, y >= c.ground) - frames[y] } else { 0.0 }).collect()).collect();
        let slopes: Vec<(Vec<f64>, Mat)> = (1..parts).map(|p| (reference[p].clone(), h[p].block(&excited, &excited))).collect();
        let classes = self.references(cluster, lights);
        let firsts: Vec<usize> = (0..classes.iter().max().map_or(0, |m| m + 1)).map(|a| classes.iter().position(|&x| x == a).expect("a class's place")).collect();
        let lit: Vec<usize> = (0..self.carriers.len()).filter(|&ci| lights.on(self.carriers[ci].frequency)).collect();
        let nus: Vec<f64> = lit.iter().map(|&ci| self.carriers[ci].detuning).collect();
        let sizes: Vec<f64> = c.frames.iter().map(|f| self.axes.size(f)).collect();
        let slow = elim::Slow { explicit: c.states[g..].iter().map(|&s| s - GROUND).collect(), frames, sizes, groups: vec![0; c.states.len()] };
        let series = elim::Reference::new(&levels, &nus, &slow, &reference[0]);
        let (mut first, mut second, mut mixed) = (vec![vec![]; self.carriers.len()], vec![vec![]; self.carriers.len()], vec![vec![]; self.carriers.len()]);
        let mut work = 0.0;
        for (j, &ci) in lit.iter().enumerate() {
            for u in &self.carriers[ci].basis {
                let coupling = u.block(&excited, &grounds);
                let mut orders = series.respond(j, &coupling, n, 1, &slopes).into_iter();
                first[ci].push(orders.next().expect("order 0").into_iter().map(|m| m.scaled(-C64::ONE)).collect());
                second[ci].push(orders.next().expect("order 1"));
                work += series.work(1, parts);
                if mixes {
                    mixed[ci].push(series.mixed(j, &coupling, n, &slopes, &firsts));
                    work += series.work(1, parts) * firsts.len() as f64;
                }
            }
        }
        Responses { first, second, mixed, reference, work }
    }

    /// A frequency's carrier, where its co-rotating component is one.
    pub(crate) fn carrier_of(&self, frequency: usize) -> Option<usize> {
        self.carriers.iter().position(|c| c.frequency == frequency)
    }

    /// The reference classes of a cluster's places under the lights `lights` (elim.rs, # The definition and the route it is
    /// evaluated on; `eliminate`'s Z₀): per place the class of the places whose reference z_y = H_far[y, y] − θ_y + c is
    /// its own at every point whatever the far field -- the base levels in their frames one, and every far beam pair's
    /// form (`level_pairs`) one on their diagonals --, each within its terms' rounding (elim.rs `equal`), the classes in
    /// the order of their first places. One resolvent a class: the series' first order on column s is of s's class's
    /// reference and of each other place's, R₀(z_a)R₀(z_t) (`Responses::mixed`), the Doppler partner where they are one.
    pub(crate) fn references(&self, cluster: usize, lights: &Lights) -> Vec<usize> {
        let c = &self.clusters[cluster];
        let pairs = self.level_pairs(lights);
        let level = |y: usize| {
            let s = c.states[y];
            (self.base[(s, s)].re - self.omega(&c.frames[y]), self.base[(s, s)].abs() + self.axes.size(&c.frames[y]))
        };
        let one = |x: usize, y: usize| {
            let ((a, sa), (b, sb)) = (level(x), level(y));
            let (sx, sy) = (c.states[x], c.states[y]);
            super::elim::equal(a - b, sa + sb)
                && pairs.iter().enumerate().all(|(w, of)| {
                    of.iter().enumerate().filter(|x| *x.1).all(|(x, _)| {
                        let form = &self.tones[w].pairs[x];
                        let (u, v) = (form[(sx, sx)], form[(sy, sy)]);
                        super::elim::equal((u - v).abs(), u.abs() + v.abs())
                    })
                })
        };
        let mut classes: Vec<usize> = vec![];
        let mut firsts: Vec<usize> = vec![];
        for y in 0..c.states.len() {
            match firsts.iter().position(|&x| one(x, y)) {
                Some(a) => classes.push(a),
                None => {
                    classes.push(firsts.len());
                    firsts.push(y);
                }
            }
        }
        classes
    }

    /// The orders a cluster's evaluation takes under the lights `lights` (elim.rs `Orders`): the model's (`Series`), the
    /// series' first order only where the cluster's own series converges (elim.rs `Model::first_order_holds`, ELIMINATION
    /// v12) -- a level of each cluster: elsewhere the first order is no part of the run's evaluation and no unit of the
    /// cluster's, and what it would take away of the evaluation's error is the next orders', the nearest block's keep
    /// (elim.rs `Model::readings`); `Unit::Series` is a candidate where some cluster holds it (elim.rs `Model::candidates`).
    pub(crate) fn orders_of(&self, cluster: usize) -> super::elim::Orders {
        let orders = self.series.orders();
        if orders.series >= 1 && !self.first_order_holds(cluster) { super::elim::Orders { series: 0, ..orders } } else { orders }
    }

    /// Whether a coupling of two places of a cluster (x ← y at the base beat `beat`) is in the cluster's generator: at a
    /// residual beat of exactly zero in the frames (free), or as a kept unit (`Unit::Term`) -- one rule for every second-order
    /// coupling, whatever light makes it.
    pub(crate) fn term_included(&self, cluster: usize, x: usize, y: usize, beat: &Beat) -> bool {
        let c = &self.clusters[cluster];
        beat.plus(1, &c.frames[x]).plus(-1, &c.frames[y]).is_zero() || self.kept.contains(&term_unit(c.states[x], c.states[y], beat))
    }

    /// The light-shift bound per beam pair of the trap frequency's drives folded with the levels at the atom (potential.rs
    /// `Model::envelope`), row-major over its beams: |[X^f_l'† G X^f_l]_p| ≤ Σ_b p_b Σ_e |X^f_l' eb| |X^f_l eb| / m_e, line by
    /// line, m_e = |E_e − E_b − ν − iΓ_e/2| with the spread taken off its detuning (`Model::spread`: what the light moves
    /// the levels by, anywhere) -- the fold's series 1/(d_e + δ_e) at most 1/(|d_e| − |δ_e|), and at least Γ_e/2 off the
    /// axis --; the excited lines' own couplings by the light enter at the next order, the spread over a distance. Zero
    /// where the trap light's drives are not folded at the atom. A far-detuned trap's drive is bounded so by its bare
    /// fold's shift times 1 + O(spread/detuning), as tight as the bare fold's own form. (With the dissipation's 2/Γ_min the
    /// user's 852 nm tweezer's drive F=2 m=2 → F′=3 m=3 kept at the atom bounded −V_prep by 4.2·10⁹ rad/µs against its
    /// 392.6, and the initial ensemble's rejection sampling, whose proposals spread as that bound's a^{5/2}, accepted none
    /// in 140 s -- R2_RECOIL's second model; with one resolvent norm for every line, the D1 lines' 25 THz distance taken
    /// for the D2 lines' 32 THz too, 498.1, and 216 proposals a draw against the bare model's 2.6.)
    pub(crate) fn near_envelope(&self, p: &[f64; GROUND]) -> Vec<f64> {
        let n = self.light.frequencies[self.light.trap].beams.len();
        let Some(cc) = self.carriers.iter().find(|c| c.frequency == self.light.trap) else { return vec![0.0; n * n] };
        let (nu, spread) = (self.tones[self.light.trap].nu[0], self.spread());
        let line = |e: usize, b: usize| ((self.energies[GROUND + e] - self.energies[b] - nu).abs() - spread).max(0.0).hypot(0.5 * self.decay[(e, e)].re);
        (0..n * n)
            .map(|x| {
                let (j, i) = (x / n, x % n);
                (0..GROUND)
                    .filter(|&b| p[b] != 0.0)
                    .map(|b| p[b] * (0..EXCITED).map(|e| cc.folded[j][(GROUND + e, b)].abs() * cc.folded[i][(GROUND + e, b)].abs() / line(e, b)).sum::<f64>())
                    .sum::<f64>()
            })
            .collect()
    }

    /// The terms of the carriers' folded responses, one a response of the plan's near forms, each to
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

}

impl Model {

    /// The largest wave number of the carriers' beams, k_max (1/µm): the light's pattern an atom crosses -- the recoil of
    /// the scattering's photons at most, and the phase a step advances along it (solve.rs `prior`). None
    /// carried, zero.
    pub(crate) fn carrier_wave_number(&self) -> f64 {
        self.carriers.iter().flat_map(|c| c.beams.iter().map(|&l| self.light.beams[l].wave_number)).fold(0.0, f64::max)
    }

}

/// A coupling of two states at a beat in canonical form (`Unit::Term`): a ← b at the beat and its adjoint, b ← a at the
/// opposite beat, are one unit, named a < b, or for a = b by the beat whose first coordinate that is not zero is positive.
pub(crate) fn term_unit(a: usize, b: usize, beat: &Beat) -> Unit {
    if a < b || (a == b && beat.sign() >= 0) {
        Unit::Term { a, b, beat: beat.clone() }
    } else {
        Unit::Term { a: b, b: a, beat: beat.negated() }
    }
}
