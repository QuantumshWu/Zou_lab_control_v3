//! What a model's clusters and terms are made of beside the elimination (elim.rs): its beats and its frequencies'
//! constant forms (far.rs `constant`), the kept couplings' edges (cluster.rs), whether a coupling is in a generator as it
//! is. A counter-rotating drive is only ever folded with the bare levels (model.rs `Model::new`: a keep of one is refused,
//! and so is a request where that fold does not exist, the model's domain); a co-rotating leg against a counter-rotating
//! one is no term of the model (far.rs, # Two frequencies' constant form).

use super::cluster::Edge;
use super::{Beat, Model, Unit, Wave};
use crate::linalg::Mat;
use crate::request::GROUND;
use std::collections::HashMap;
use std::sync::Arc;

/// Two frequencies' constant form (far.rs `constant`) on their fields' bases.
pub(crate) struct Cross {
    pub(crate) forms: Vec<Mat>,
}

impl Model {
    /// The spread of what moves the blocks' spectra, a bound over all space (rad/µs): the constant forms' coherent peaks
    /// and the carriers' couplings at theirs; made once (`Model::new`). The photon coherences' distance takes it (photon.rs).
    pub(crate) fn spread(&self) -> f64 {
        self.spread
    }
    pub(crate) fn spread_made(&self) -> f64 {
        let constant: f64 = self.tones.iter().map(|t| t.pairs.iter().map(|m| m.frobenius()).sum::<f64>()).sum();
        let carriers: f64 = self.carriers.iter().map(|c| self.tones[c.frequency].couplings[0].iter().map(|m| m.frobenius()).sum::<f64>()).sum();
        constant + carriers
    }

    /// The model's beats: the distinct δ_ω − δ_ω' of its frequencies, each with its pairs (ω, ω') -- the first is zero,
    /// every frequency's with itself.
    pub(crate) fn beats_made(&self) -> Vec<(Beat, Vec<(usize, usize)>)> {
        let mut out: Vec<(Beat, Vec<(usize, usize)>)> = vec![];
        for to in 0..self.tones.len() {
            for from in 0..self.tones.len() {
                let beat = self.tones[to].beat.plus(-1, &self.tones[from].beat);
                match out.iter_mut().find(|b| b.0 == beat) {
                    Some(found) => found.1.push((to, from)),
                    None => out.push((beat, vec![(to, from)])),
                }
            }
        }
        out
    }

    /// Two frequencies' constant form (`Cross`), made once: a frequency's own is its static form.
    pub(crate) fn constants(&self, to: usize, from: usize) -> Arc<Cross> {
        let mut made = self.crosses.lock().expect("the constant forms' lock");
        made.entry((to, from))
            .or_insert_with(|| {
                let (t, f) = (&self.tones[to], &self.tones[from]);
                let forms = if to == from { t.forms.clone() } else { super::far::constant(t, f) };
                Arc::new(Cross { forms })
            })
            .clone()
    }

    /// Whether a state is an explicit excited state of this model.
    pub(crate) fn explicit(&self, s: usize) -> bool {
        s >= GROUND && self.clusters.iter().any(|c| c.states.contains(&s))
    }

    /// Whether the coupling a ← b at a beat is in this model's generators as it is (SPEC §1.4), nothing of it folded: the
    /// two states in one cluster, at a residual beat of exactly zero or as a kept unit (eliminate.rs `Model::term_included`).
    pub(crate) fn in_generator(&self, a: usize, b: usize, beat: &Beat) -> bool {
        self.clusters.iter().enumerate().any(|(k, c)| {
            let place = |s: usize| c.states.iter().position(|&x| x == s);
            matches!((place(a), place(b)), (Some(x), Some(y)) if self.term_included(k, x, y, beat))
        })
    }

    /// The kept units' edges (cluster.rs): a kept drive at the atom joins its two states at θ_e − θ_g = δ_ω, its strongest
    /// beam's wave vector; a kept coupling a ← b at the beat β joins them at θ_a − θ_b = −β, its strongest beam pair's
    /// k_absorbed − k_emitted, its strength the peak of its entry with every drive bare, Σ_pairs |C_ji[a, b]| (`bare`: the
    /// frequencies with every drive folded with the bare levels -- one measure for every coupling, whatever the drives'
    /// levels turn out; the tree is a choice of cost, cluster.rs). A unit of nothing in this request makes none.
    pub(crate) fn edges(&self, bare: &[super::far::Tone]) -> Vec<Edge> {
        let mut out = vec![];
        let mut crosses: HashMap<(usize, usize), Vec<Mat>> = HashMap::new();
        for unit in &self.kept {
            match unit {
                Unit::Drive { component, excited, ground } if !component.counter && self.driven.contains(&(*component, *excited, *ground)) => {
                    let (f, tone) = (&self.light.frequencies[component.frequency], &self.tones[component.frequency]);
                    let (best, strength) = f.beams.iter().zip(&tone.couplings[0]).fold((f.beams[0], 0.0_f64), |(best, s), (&l, x)| {
                        let v = x[(*excited, *ground)].abs();
                        (if v > s { l } else { best }, s.max(v))
                    });
                    out.push(Edge { a: *excited, b: *ground, beat: tone.beat.clone(), wave: self.beam_wave(best), strength });
                }
                Unit::Term { a, b, beat } => {
                    let (mut best, mut strength) = ((0.0_f64, self.no_wave()), 0.0);
                    for &(to, from) in self.beats.iter().find(|x| x.0 == *beat).map_or(&[][..], |x| &x.1) {
                        let pairs = crosses.entry((to, from)).or_insert_with(|| {
                            if to == from { bare[to].pairs.clone() } else { super::far::per_pair(&bare[to], &bare[from], &super::far::constant(&bare[to], &bare[from])) }
                        });
                        let (bt, bf) = (&self.light.frequencies[to].beams, &self.light.frequencies[from].beams);
                        for (x, m) in pairs.iter().enumerate() {
                            let size = m[(*a, *b)].abs();
                            strength += size;
                            if size > best.0 {
                                best = (size, self.pair_wave(bt[x / bf.len()], bf[x % bf.len()]));
                            }
                        }
                    }
                    if strength > 0.0 {
                        out.push(Edge { a: *a, b: *b, beat: beat.negated(), wave: best.1, strength });
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// The exact wave vector of a beam pair's term, absorbing `absorbed` and emitting `emitted`: k_absorbed − k_emitted.
    pub(crate) fn pair_wave(&self, emitted: usize, absorbed: usize) -> Wave {
        self.beam_wave(absorbed).plus(-1, &self.beam_wave(emitted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
    use crate::linalg::C64;
    use crate::model::Level;
    use crate::request::{Request, STATES};
    use super::super::far::EXCITED;

    /// Every number of a model's forms is one, whatever it keeps -- over random scenes and random kept sets. A drive folded
    /// with the bare levels onto an explicit state whose kept leg's ground state is exactly degenerate with its own (Zeeman
    /// sublevels at B = 0) is criterion two's link at δ = 0: the model takes it explicit (model.rs `Model::settled`), never
    /// folds it at a two-photon distance of nothing (far.rs `second_order` asserts so). The scenes are R4_LIN_2's atom and lights -- a tweezer 32 THz red of
    /// D2, a lin⊥lin cooling pair, a repump, B = 0 --: first as the reproduction of the counter-rotating refusal kept it
    /// (ground state 3's co-rotating drives folded at the atom), then drawn: every beam but the tweezer along a random lab
    /// axis, every beam's polarization uniform over the transverse ones (its field and couplings the dipole operator's on
    /// the unit fields x̂, ŷ of the cooling pair and ẑ of the tweezer -- the repump's per V/m the cooling's, checked), its
    /// intensity log-uniform over 0.1 to 10 of the scene's (the tweezer's 0.5 to 2), the near light's detuning from the
    /// atom within ±30 Γ, the repump's offset within ±30 Γ, the repump present or not, and a random kept set (of the drawn
    /// scene's units -- each co-rotating drive one level up, `Unit::Bare` or `Unit::Drive`, and the keep rule's candidates
    /// (elim.rs `Model::candidates`) -- each with a probability of 0.02 to 0.3, in one or two rounds).
    #[test]
    fn every_model_s_forms_are_numbers_whatever_it_keeps() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R4_LIN_2.json");
        let base = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let id = |name: &str| base.beams.iter().position(|b| b.id == name).unwrap();
        // A beam's field and couplings per V/m of its field along its own polarization (−ẑ: the tweezer's field is −ẑ).
        let per_field = |l: usize, sign: f64| -> ([C64; 3], Mat, Mat) {
            let b = &base.beams[l];
            let size = sign * b.field.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            let (mut x, mut r) = (Mat::zeros(STATES, STATES), Mat::zeros(STATES, STATES));
            x.add_scaled(C64::real(1.0 / size), &b.coupling);
            r.add_scaled(C64::real(1.0 / size), &b.counter_coupling);
            (b.field.map(|z| z * (1.0 / size)), x, r)
        };
        let axes = [per_field(id("cool-plus"), 1.0), per_field(id("cool-minus"), 1.0), per_field(id("tweezer"), -1.0)];
        for (i, a) in axes.iter().enumerate() {
            for (j, b) in axes.iter().enumerate() {
                let dot: C64 = a.0.iter().zip(&b.0).map(|(x, y)| x.conj() * *y).sum();
                assert!((dot - C64::real(if i == j { 1.0 } else { 0.0 })).abs() < 1e-12, "the unit fields x̂, ŷ, ẑ: ⟨{i}|{j}⟩ = {dot:?}");
            }
        }
        let repump = per_field(id("repump-plus"), 1.0);
        let apart = |a: &Mat, b: &Mat| a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max);
        assert!((0..3).all(|q| (repump.0[q] - axes[0].0[q]).abs() < 1e-12), "the repump is x-polarized");
        assert!(apart(&repump.1, &axes[0].1) <= 1e-9 * axes[0].1.frobenius() && apart(&repump.2, &axes[0].2) <= 1e-9 * axes[0].2.frobenius(), "one dipole operator: the repump's couplings per V/m are the cooling's");
        let gamma = {
            let model = Model::new(&base, &Light::new(&base), &[]).unwrap();
            (0..EXCITED).map(|x| model.decay[(x, x)].re).fold(0.0, f64::max)
        };
        let repump_index = base.beams[id("repump-plus")].frequency_index.clone();
        let tweezer = base.beams[base.trap_beam].id.clone();
        // A model's units one level up: each co-rotating drive's next level, and the keep rule's candidates.
        let units = |model: &Model| -> Vec<Unit> {
            let raised = model.driven.iter().filter(|d| !d.0.counter).filter_map(|&(component, excited, ground)| match model.level(component, excited, ground) {
                Level::Bare => Some(Unit::Bare { component, excited, ground }),
                Level::Atom => Some(Unit::Drive { component, excited, ground }),
                Level::Explicit => None,
            });
            raised.chain(model.candidates()).collect()
        };
        let mut stream = crate::random::Stream::new(7, crate::random::Purpose::Initial, 0, 0);
        let state = (0..GROUND).max_by(|&a, &b| base.initial_density[(a, a)].re.total_cmp(&base.initial_density[(b, b)].re)).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
        let (mut unmade, mut joined, mut kept_all) = (0, 0, 0);
        for s in 0..49 {
            let mut request = base.clone();
            let mut kept: Vec<Unit> = vec![];
            if s == 0 {
                // The scene as the reproduction of the counter-rotating refusal kept it: ground state 3's co-rotating
                // drives folded at the atom.
                let model = Model::new(&request, &Light::new(&request), &[]).unwrap();
                kept = model.driven.iter().filter(|d| !d.0.counter && d.2 == 3).map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).collect();
            } else {
                let log_uniform = |lo: f64, hi: f64, u: f64| (lo.ln() + (hi / lo).ln() * u).exp();
                for b in request.beams.iter_mut() {
                    let trap = b.id == tweezer;
                    let along = if trap { 0 } else { (3.0 * stream.uniform()) as usize % 3 };
                    if !trap {
                        let sign = if stream.uniform() < 0.5 { -1.0 } else { 1.0 };
                        b.direction = std::array::from_fn(|i| if i == along { sign } else { 0.0 });
                    }
                    let (z1, z2) = (stream.normals(), stream.normals());
                    let (first, second) = (C64::new(z1[0], z1[1]), C64::new(z1[2], z2[0]));
                    let norm = (first.norm_sqr() + second.norm_sqr()).sqrt();
                    let across: Vec<usize> = (0..3).filter(|&i| i != along).collect();
                    let mut polarization = [C64::ZERO; 3];
                    polarization[across[0]] = first / norm;
                    polarization[across[1]] = second / norm;
                    let (lo, hi) = if trap { (0.5, 2.0) } else { (0.1, 10.0) };
                    let size = b.field.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() * log_uniform(lo, hi, stream.uniform()).sqrt();
                    b.field = std::array::from_fn(|q| (0..3).map(|c| polarization[c] * axes[c].0[q]).sum::<C64>() * size);
                    b.coupling = Mat::zeros(STATES, STATES);
                    b.counter_coupling = Mat::zeros(STATES, STATES);
                    for c in 0..3 {
                        b.coupling.add_scaled(polarization[c] * size, &axes[c].1);
                        b.counter_coupling.add_scaled(polarization[c].conj() * size, &axes[c].2);
                    }
                }
                let detuning = 30.0 * gamma * (2.0 * stream.uniform() - 1.0);
                for e in GROUND..STATES {
                    request.hamiltonian[(e, e)] -= C64::real(detuning);
                }
                request.frequencies_mhz[0] += 30.0 * gamma / (2.0 * std::f64::consts::PI) * (2.0 * stream.uniform() - 1.0);
                if stream.uniform() < 0.5 {
                    request.beams.retain(|b| b.frequency_index != repump_index);
                    request.outside_forms.retain(|f| f.frequency_index != repump_index);
                    request.trap_beam = request.beams.iter().position(|b| b.id == tweezer).unwrap();
                }
                // The reach's temperature, drawn as the scenes were first drawn (the stream's draws are the scenes').
                let _temperature = 10.0 + 70.0 * stream.uniform();
                let p = [0.02, 0.1, 0.3][(3.0 * stream.uniform()) as usize % 3];
                let light = Light::new(&request);
                for _ in 0..1 + usize::from(stream.uniform() < 0.5) {
                    let model = Model::new(&request, &light, &kept).unwrap();
                    let more: Vec<Unit> = units(&model).into_iter().filter(|_| stream.uniform() < p).collect();
                    kept.extend(more);
                }
            }
            let model = Model::new(&request, &Light::new(&request), &kept).unwrap();
            if model.tones.iter().any(|t| t.pairs.iter().chain(&t.forms).any(|m| m.data.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())))) {
                unmade += 1;
            }
            joined += model.kept.iter().filter(|u| !kept.contains(u)).count();
            kept_all += kept.len();
            // Where the scene has a trap, the nine draws of the states first read about its centre: the next scene is
            // the one first drawn.
            if model.trap(&reference, &model.prepared()).is_ok() {
                (0..9).for_each(|_| _ = stream.uniform());
            }
        }
        println!("49 scenes, {kept_all} units kept: {joined} drives taken explicit at no two-photon distance, {unmade} models with forms that are no number");
        assert!(unmade == 0, "{unmade} models with forms that are no number");
        assert!(joined > 0, "no drive taken explicit: the draw reaches no degenerate pair");
    }
}
