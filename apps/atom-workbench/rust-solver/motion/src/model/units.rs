//! What a model's clusters and terms are made of beside the elimination (elim.rs): its beats and its frequencies'
//! constant forms (far.rs `constant`), the kept couplings' edges (cluster.rs), whether a coupling is in a generator as it
//! is. A counter-rotating drive is only ever folded with the bare levels (model.rs `Model::new`: a keep of one is refused,
//! and so is a request where that fold does not exist, the model's domain); a co-rotating leg against a counter-rotating
//! one is no term of the model (far.rs, # Two frequencies' constant form).

use super::cluster::Edge;
use super::{Beat, Component, Model, Unit, Wave};
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

    /// The units that fold a component's drives with the bare levels (`Unit::Bare`), whatever their level in this model:
    /// kept, each of them is folded with the levels at the atom.
    pub fn bare_units(&self, component: Component) -> Vec<Unit> {
        self.driven.iter().filter(|d| d.0 == component).map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).collect()
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
    use crate::model::{Point, Sample};
    use crate::request::Request;

    /// A reference scene's model with the units the released line kept for it (`released_kept`), and its trap's centre at
    /// the reference state.
    fn closed(scene: &str) -> (Model, [f64; 3]) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{scene}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let state = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
        let model = Model::new(&request, &light, &released_kept(scene)).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
        let centre = model.trap(&reference, &model.prepared()).unwrap().centre;
        (model, centre)
    }

    /// The units the released line's closure (v2026.11.13.2, 991fa7b0: every unit whose existence gate failed over the
    /// reach at the request's temperature kept, until none failed; for E3_SPSM its second model, the first one's failing
    /// folds kept) kept for the scenes these tests take their models from: recorded, the model the tests' statements are
    /// about -- which folds are taken is no longer proven but measured in a run (solve.rs, # The model).
    fn released_kept(scene: &str) -> Vec<Unit> {
        match scene {
            "R4_LIN_1" => vec![
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 25, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 25, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 26, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 26, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 28, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 28, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 30, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 30, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 31, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 31, ground: 7 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 20, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 20, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 21, ground: 1 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 22, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 22, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 23, ground: 1 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 24, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 24, ground: 2 },
                Unit::Term { a: 0, b: 2, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 3, b: 5, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 3, b: 7, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 4, b: 6, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 5, b: 7, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 20, ground: 0 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 21, ground: 1 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 23, ground: 1 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 24, ground: 2 },
            ],
            "R4_LIN_2" => vec![
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 21, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 25, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 25, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 26, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 26, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 28, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 28, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 30, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 30, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 31, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 31, ground: 7 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 20, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 20, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 21, ground: 1 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 22, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 22, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 23, ground: 1 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 24, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 24, ground: 2 },
                Unit::Term { a: 0, b: 2, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 3, b: 5, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 3, b: 7, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 4, b: 6, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 5, b: 7, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 20, ground: 0 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 21, ground: 1 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 23, ground: 1 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 24, ground: 2 },
            ],
            "R4_LIN_4" => vec![
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 21, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 21, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 21, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 22, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 22, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 23, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 23, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 23, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 25, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 26, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 26, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 28, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 28, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 30, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 30, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 31, ground: 7 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 20, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 20, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 21, ground: 1 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 22, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 22, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 23, ground: 1 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 24, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 24, ground: 2 },
                Unit::Term { a: 0, b: 2, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 3, b: 5, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 3, b: 7, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 4, b: 6, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 5, b: 7, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 20, ground: 0 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 21, ground: 1 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 23, ground: 1 },
                Unit::Drive { component: Component { frequency: 2, counter: false }, excited: 24, ground: 2 },
            ],
            "E3_SPSM" => vec![
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 25, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 25, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 25, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 26, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 26, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 27, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 28, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 28, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 29, ground: 7 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 30, ground: 4 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 30, ground: 6 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 31, ground: 3 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 31, ground: 5 },
                Unit::Bare { component: Component { frequency: 0, counter: false }, excited: 31, ground: 7 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 17, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 17, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 19, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 19, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 20, ground: 1 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 21, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 21, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 22, ground: 1 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 23, ground: 0 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 23, ground: 2 },
                Unit::Bare { component: Component { frequency: 2, counter: false }, excited: 24, ground: 1 },
                Unit::Term { a: 0, b: 2, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 3, b: 7, beat: Beat::whole(vec![0, 0, 0, 0]) },
                Unit::Term { a: 4, b: 6, beat: Beat::whole(vec![0, 0, 0, 0]) },
            ],
            _ => panic!("no recorded kept set for {scene}"),
        }
    }

    /// Every number of a model's forms is one, whatever it keeps -- over random scenes and random kept sets. A drive folded
    /// onto an explicit state whose kept leg's ground state is exactly degenerate with its own (Zeeman sublevels at B = 0)
    /// is a resonant coherent coupling: the model joins it (model.rs `Model::settled`, `Model::joined`), never folds it at a
    /// two-photon distance of nothing (far.rs `second_order` asserts so; before, scene 19 left one so: a NaN row of its
    /// constant form, which `Own::max` then dropped). The scenes are R4_LIN_2's atom and lights -- a tweezer 32 THz red of
    /// D2, a lin⊥lin cooling pair, a repump, B = 0 --: first as the reproduction of the counter-rotating refusal kept it
    /// (ground state 3's co-rotating drives folded at the atom), then drawn: every beam but the tweezer along a random lab
    /// axis, every beam's polarization uniform over the transverse ones (its field and couplings the dipole operator's on
    /// the unit fields x̂, ŷ of the cooling pair and ẑ of the tweezer -- the repump's per V/m the cooling's, checked), its
    /// intensity log-uniform over 0.1 to 10 of the scene's (the tweezer's 0.5 to 2), the near light's detuning from the
    /// atom within ±30 Γ, the repump's offset within ±30 Γ, the repump present or not, and a random kept set (the drawn
    /// scene's units but the counter-rotating drives, each with a probability of 0.02 to 0.3, in one or two rounds).
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
        let counter = |u: &Unit| matches!(u, Unit::Bare { component, .. } | Unit::Drive { component, .. } if !component.keepable());
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
                kept = model.units().into_iter().filter(|u| matches!(u, Unit::Bare { component, ground: 3, .. } if !component.counter)).collect();
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
                    let more: Vec<Unit> = model.units().into_iter().filter(|u| !counter(u)).filter(|_| stream.uniform() < p).collect();
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
        println!("49 scenes, {kept_all} units kept: {joined} drives joined by their Raman pairs, {unmade} models with forms that are no number");
        assert!(unmade == 0, "{unmade} models with forms that are no number");
        assert!(joined > 0, "no drive joined: the draw reaches no degenerate Raman pair");
    }

    /// The sweep is a relative error of the fold's own effect (module documentation, # One rule; `Gate::resolved`). The
    /// exact response of a linear sweep is the instantaneous fold times I(a) = ∫₀^∞ e^{−x + iax²/2} dx (internal.md
    /// 1B.1), of modulus at most one at any a -- so the fold is off by at most twice its own amplitude: |I(a) − 1| ≤ 2
    /// (Simpson on [0, 40], the step resolving the phase's largest rate a·40). At a = ½ the response holds 0.71 of the
    /// fold's population, at a = 1.75 0.39 (the gate's edge, internal.md). A coupling whose whole effect, 2S, is below the
    /// least resolved share has its sweep scaled down with it -- the user's lin⊥lin pair, S = 9e-38 at a sweep of 0.83,
    /// has none --; one above it keeps its sweep, and a fold of a share at the target's scale with a sweep beyond ½ is
    /// kept. Continuous across the resolution (a cut-off there left a cell's bound at 8.4 over its centre's 0).
    #[test]
    fn a_fold_s_sweep_counts_as_far_as_its_own_effect_is_seen() {
        let response = |a: f64| -> C64 {
            let (end, n) = (40.0_f64, (2.0 * (1.0 + a * 40.0) * 40.0 * 20.0).ceil() as usize * 2);
            let h = end / n as f64;
            let f = |x: f64| C64::new(-x, 0.5 * a * x * x).exp();
            let inner: C64 = (1..n).map(|i| f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 }).sum();
            (f(0.0) + inner + f(end)) * (h / 3.0)
        };
        for a in [0.0, 0.1, 0.5, 1.0, 1.75, 4.0, 20.0] {
            let i = response(a);
            assert!(i.abs() <= 1.0 + 1e-9 && (i - C64::ONE).abs() <= 2.0, "a = {a}: I = {i:?}");
        }
        assert!((response(0.5).norm_sqr() - 0.71).abs() < 0.01 && (response(1.75).norm_sqr() - 0.39).abs() < 0.01, "{} {}", response(0.5).norm_sqr(), response(1.75).norm_sqr());
        let unseen = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        assert!(Gate { saturation: 9e-38, sweep: 0.83 }.resolved().exists());
        assert!(Gate { saturation: 0.25 * unseen, sweep: 0.9 }.resolved().exists());
        assert!(!Gate { saturation: unseen, sweep: 0.6 }.resolved().exists());
        assert!(!Gate { saturation: 0.1, sweep: 0.6 }.resolved().exists());
        assert_eq!(Gate { saturation: 0.1, sweep: 0.3 }.resolved().sweep, 0.3);
        // Continuous across the resolution: just below and just above 2S = unseen the sweep is the same to the rounding.
        let (below, above) = (Gate { saturation: 0.5 * unseen * (1.0 - 1e-12), sweep: 8.4 }.resolved().sweep, Gate { saturation: 0.5 * unseen * (1.0 + 1e-12), sweep: 8.4 }.resolved().sweep);
        assert!((above - below).abs() <= 1e-10, "{below} {above}");
    }

    /// Three levels in a scene's atom: every coupling of R6_RAMAN_CO's request removed but g –V→ e of one frequency and
    /// b –U→ e of the other, no outside form, the levels set so that the kept drive is ε from e and the folded one β
    /// (in Γ_e). The request; the two frequencies; Γ_e.
    fn three_levels(v: f64, epsilon: f64, u: f64, beta: f64) -> (Request, [usize; 2], f64) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R6_RAMAN_CO.json");
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let (g, b, e) = (THREE[0], THREE[1], THREE[2]);
        let of: Vec<usize> = (0..light.frequencies.len()).filter(|&w| w != light.trap).collect();
        assert!(of.len() == 2 && of.iter().all(|&w| light.frequencies[w].beams.len() == 1), "two lights of one beam each beside the trap's");
        let nu: Vec<f64> = { let model = Model::new(&request, &light, &[]).unwrap(); of.iter().map(|&w| model.tones[w].nu[0]).collect() };
        let gamma: f64 = request.emission.iter().map(|c| (0..GROUND).map(|s| c[(s, e)].norm_sqr()).sum::<f64>()).sum();
        for beam in request.beams.iter_mut() {
            beam.coupling = Mat::zeros(STATES, STATES);
            beam.counter_coupling = Mat::zeros(STATES, STATES);
        }
        request.beams[light.frequencies[of[0]].beams[0]].coupling[(e, g)] = C64::real(v * gamma);
        request.beams[light.frequencies[of[1]].beams[0]].coupling[(e, b)] = C64::real(u * gamma);
        for form in request.outside_forms.iter_mut() {
            form.matrices = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
        }
        request.hamiltonian[(g, g)] = C64::ZERO;
        request.hamiltonian[(e, e)] = C64::real(epsilon * gamma + nu[0]);
        request.hamiltonian[(b, b)] = C64::real(epsilon * gamma + nu[0] - nu[1] - beta * gamma);
        (request, [of[0], of[1]], gamma)
    }
    /// `three_levels`' states: g, b, e.
    const THREE: [usize; 3] = [3, 0, GROUND];

    /// A cluster's no-jump generator at the focus, at rest; its eigenvalue whose vector is mostly the place `of`, and
    /// that vector.
    fn eigenvalue(model: &Model, cluster: usize, of: usize) -> (C64, Vec<C64>) {
        use crate::step::Middle as _;
        let middle = model.assemble(cluster, [0.0; 3], 0.0, &model.running());
        let n = model.clusters[cluster].states.len();
        let mut h = Mat::zeros(n, n);
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
        like(&h, of)
    }
    fn like(h: &Mat, of: usize) -> (C64, Vec<C64>) {
        let decomposed = crate::linalg::eigen(h).unwrap();
        let k = (0..h.rows).max_by(|&x, &y| decomposed.vectors[(of, x)].norm_sqr().total_cmp(&decomposed.vectors[(of, y)].norm_sqr())).unwrap();
        (decomposed.values[k], (0..h.rows).map(|x| decomposed.vectors[(x, k)]).collect())
    }

    /// A drive folded onto an excited state that another drive makes explicit is a Raman pair (far.rs, # A fold onto
    /// an explicit state), against the exact three levels g –V→ e ←U– b (`three_levels`; the exact no-jump spectrum of
    /// the 3 × 3 generator: the kept cluster's own 2 × 2 without U, b at the two-photon distance d_R = β − ε from g,
    /// U between e and b). V = 0.5 Γ, ε = Γ, U = 0.2 Γ, β from 10 Γ through the Raman resonance at β = ε to 0.02 Γ, the
    /// folded drive with the bare levels and with those at the atom:
    /// - where the drive's gate holds at the point (`Model::folds`: its own and the pair's), the kept line's rate
    ///   −2 Im λ_g and b's rate −2 Im λ_b are the exact ones within the drive's booked error (`Model::errors`, at the
    ///   state itself; the populations' error over the state's ground population) -- b's, where its fold's order is
    ///   its cluster's unit (`Unit::Second`), with that unit's s times the 3/2 the second order's slow decay is off by
    ///   (SPEC §1.5; the photons' rate, which s is booked for, by s). Measured: the kept line's rate +1.7e-4 at 10 Γ
    ///   (booked 2.4e-4), +1.6e-2 at 2 Γ (2.8e-2), +1.2e-2 at 0.02 Γ (3.5e-2), where the one-photon real denominator
    ///   read 82 % low;
    /// - it holds far from the resonance and fails around it (bare: from 1.5 Γ down, its own gate's too -- b's
    ///   undressed response is 19 % off at 2 Γ --; at the atom: from 1.2 Γ to 0.1 Γ): the fold is refused there, not
    ///   evaluated; at the Raman resonance itself (β = ε, the two-photon distance nothing) the model joins b to the kept
    ///   cluster (`Model::joined`): the pair is a resonant coherent coupling, which no fold holds anywhere.
    /// And b's response through an e that a strong drive dresses (V = 2 Γ on resonance, β = 27.4 Γ): with the levels at
    /// the atom its rate is the exact one to 1e-3 (the undressed response's: −1.06 %, within its booked error).
    #[test]
    fn a_drive_folded_onto_an_explicit_state_is_a_raman_pair() {
        let (g, b, e) = (THREE[0], THREE[1], THREE[2]);
        let case = |v: f64, epsilon: f64, u: f64, beta: f64, atom: bool| -> (bool, [f64; 2], [f64; 2]) {
            let (request, of, gamma) = three_levels(v, epsilon, u, beta);
            let light = Light::new(&request);
            let (kept, folded) = (Component { frequency: of[0], counter: false }, Component { frequency: of[1], counter: false });
            let explicit = Unit::Drive { component: kept, excited: e, ground: g };
            let bare = Unit::Bare { component: folded, excited: e, ground: b };
            let model = Model::new(&request, &light, &if atom { vec![explicit.clone(), bare.clone()] } else { vec![explicit.clone()] }).unwrap();
            let unit = if atom { Unit::Drive { component: folded, excited: e, ground: b } } else { bare };
            if model.level(folded, e, b) == Level::Explicit {
                // The pair cannot be folded anywhere: the model joins b to the kept cluster (`Model::joined`) -- refused,
                // not evaluated, as where its gate fails at the point.
                assert_eq!(model.membership[b], model.membership[g], "β = {beta}: a joined drive's ground state is the cluster's");
                return (false, [0.0; 2], [0.0; 2]);
            }
            assert_eq!(model.level(folded, e, b), if atom { Level::Atom } else { Level::Bare });
            let (cg, cb) = (model.membership[g], model.membership[b]);
            assert_eq!((model.clusters[cg].states.clone(), model.clusters[cb].states.clone()), (vec![g, e], vec![b]));
            // The exact three levels: the kept cluster's own generator without the folded light, and b.
            let alone = {
                let (without, _, _) = three_levels(v, epsilon, 0.0, beta);
                let model = Model::new(&without, &Light::new(&without), &[explicit]).unwrap();
                use crate::step::Middle as _;
                let middle = model.assemble(model.membership[g], [0.0; 3], 0.0, &model.running());
                let mut h = Mat::zeros(2, 2);
                middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                h
            };
            let exact = Mat::from_fn(3, 3, |x, y| match (x, y) {
                (2, 2) => alone[(0, 0)] - C64::real((beta - epsilon) * gamma),
                (1, 2) | (2, 1) => C64::real(u * gamma),
                (x, y) if x < 2 && y < 2 => alone[(x, y)],
                _ => C64::ZERO,
            });
            let region = model.point_region([0.0; 3], 0.0);
            let gate = model.folds(b, 0, &region);
            let exists = if atom { gate[0] < EXISTS && gate[1] < EXISTS } else { gate[2] < EXISTS && gate[3] < EXISTS };
            let mut errors = [0.0; 2];
            let mut booked = [0.0; 2];
            if !exists {
                return (exists, errors, booked);
            }
            for (i, (cluster, place, level)) in [(cg, 0, 0), (cb, 0, 2)].into_iter().enumerate() {
                let (lambda, psi) = eigenvalue(&model, cluster, place);
                let ground = psi[0].norm_sqr() / psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
                errors[i] = lambda.im / like(&exact, level).0.im - 1.0;
                let sample = Sample { cluster, at: Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, psi };
                let of = model.errors(&[unit.clone(), Unit::Second { ground: b }], &[sample]);
                booked[i] = (of[0][0].population + ORDER_FORCE * of[1][0].population) / ground;
            }
            (exists, errors, booked)
        };
        for atom in [false, true] {
            let mut refused = vec![];
            for beta in [10.0, 6.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.5, 0.1, 0.02] {
                let (exists, errors, booked) = case(0.5, 1.0, 0.2, beta, atom);
                println!("atom {atom} β {beta:5.2}: exists {exists}; kept line's rate {:+.2e} (booked {:.2e}), b's rate {:+.2e} (booked {:.2e})", errors[0], booked[0], errors[1], booked[1]);
                if !exists {
                    refused.push(beta);
                    continue;
                }
                for i in 0..2 {
                    assert!(errors[i].abs() <= booked[i], "atom {atom}, β = {beta} Γ, rate {i}: {:+.3e} of the exact one, booked {:.3e}", errors[i], booked[i]);
                }
            }
            assert!([1.2, 1.0, 0.8].iter().all(|beta| refused.contains(beta)), "atom {atom}: around the Raman resonance the fold is refused: {refused:?}");
            let (request, of, _) = three_levels(0.5, 1.0, 0.2, 1.0);
            let joined = Model::new(&request, &Light::new(&request), &[Unit::Drive { component: Component { frequency: of[0], counter: false }, excited: e, ground: g }]).unwrap();
            assert!(joined.level(Component { frequency: of[1], counter: false }, e, b) == Level::Explicit && joined.membership[b] == joined.membership[g], "at the Raman resonance the pair joins b to the kept cluster");
            let far: &[f64] = if atom { &[10.0, 6.0, 3.0, 2.0, 0.02] } else { &[10.0, 6.0, 3.0, 2.0] };
            assert!(far.iter().all(|beta| !refused.contains(beta)), "atom {atom}: far from the Raman resonance the fold exists: {refused:?}");
        }
        let (exists, errors, _) = case(2.0, 0.0, 0.2, 27.4, true);
        assert!(exists && errors[1].abs() < 1e-3, "b's rate through the dressed e: {:+.2e} of the exact one", errors[1]);
        let (_, errors, booked) = case(2.0, 0.0, 0.2, 27.4, false);
        assert!(errors[1].abs() <= booked[1] && errors[1] < -5e-3, "with the bare levels {:+.2e}, booked {:.2e}", errors[1], booked[1]);
    }

    /// What a folded drive does to its ground level is within its own gate's bounds (module documentation, # What a
    /// folded drive does to its level): on the lin-perp-lin scene at 160 mW/cm² a beam, closed, at points across a fringe
    /// of the cooling light and off the trap's axis, the shift of every ground state by its drives folded at the atom --
    /// the elimination's own, −Re Σ_e V[e, g]* R[e, g] over the carriers -- is within `near_level` on the point, and its
    /// slope along the fringes and across them (a central difference) within its gradient's bound (`Shift::steepness`). The bounds are the drives'
    /// own: the shift's is below a twentieth of what its far drives' couplings would be across Γ/2, a resonant drive's
    /// distance, with its resonant ones' across their own Γ/2 (the repump's F=1 → D2 F'=2 m=0, folded at the atom once
    /// the D2 sublevels' couplings within their width are no resonance, # A coupling of two states).
    #[test]
    fn a_folded_drive_s_shift_and_its_slope_are_within_their_bounds() {
        let (model, centre) = closed("R4_LIN_4");
        let shift = |cluster: usize, place: usize, r: [f64; 3]| -> f64 {
            let c = &model.clusters[cluster];
            let psi: Vec<C64> = (0..c.states.len()).map(|x| if x == place { C64::ONE } else { C64::ZERO }).collect();
            let sample = Sample { cluster, at: Point { r, kappa: [0.0; 3], t: 0.0 }, psi };
            let at = model.measured(&sample, None);
            let g = c.states[place];
            let mut out = 0.0;
            for (cc, response) in model.carriers.iter().zip(&at.responses) {
                let Some(response) = response else { continue };
                for e in 0..EXCITED {
                    let v: C64 = cc.beams.iter().zip(&cc.folded).map(|(&l, m)| at.amplitudes[l].value * m[(GROUND + e, g)]).sum();
                    out -= (v.conj() * response[(e, place)]).re;
                }
            }
            out
        };
        let (mut largest, mut steepest, mut tightest) = (0.0_f64, 0.0_f64, f64::INFINITY);
        for offset in [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.0, 0.0, 0.1], [0.0, 0.0, 0.17], [0.0, 0.0, 0.25], [0.3, 0.0, 0.07], [0.0, 0.4, 0.31]] {
            let r: [f64; 3] = std::array::from_fn(|i| centre[i] + offset[i]);
            let region = model.point_region(r, 0.0);
            for (cluster, c) in model.clusters.iter().enumerate() {
                for (place, &g) in c.states[..c.ground].iter().enumerate() {
                    let (here, level) = (shift(cluster, place, r), model.near_level(g, &region));
                    let (lower, upper) = (level.band.lower(), level.band.upper());
                    let bound = lower.abs().max(upper.abs());
                    assert!(here >= lower - 1e-9 * bound && here <= upper + 1e-9 * bound, "the shift {here} of ground state {g} at {r:?} beyond its range [{lower}, {upper}]");
                    // Per drive folded at the atom its coupling across Γ/2: the far ones' (their bare distance beyond twenty
                    // half-widths) and the resonant ones'.
                    let (mut far, mut resonant) = (0.0, 0.0);
                    for (of, d) in model.across(g, &region).iter().zip(&region.forms.drives(&model)[g]).filter(|of| of.1.level == Level::Atom) {
                        let across = of.size.value * of.size.value / (0.5 * d.gamma);
                        if d.bare >= 10.0 * d.gamma { far += across } else { resonant += across }
                    }
                    assert!(bound <= far / 20.0 + resonant, "the bound {bound} of ground state {g} against its couplings across Γ/2, far {far} and resonant {resonant}");
                    largest = largest.max(here.abs());
                    if here.abs() > 1.0 {
                        tightest = tightest.min(bound / here.abs());
                    }
                    let step = 1e-4;
                    for axis in [0, 2] {
                        let moved = |sign: f64| {
                            let mut at = r;
                            at[axis] += sign * step;
                            shift(cluster, place, at)
                        };
                        let slope = (moved(1.0) - moved(-1.0)) / (2.0 * step);
                        let bound = model.near_level(g, &region).steepness().value;
                        assert!(slope.abs() <= bound * (1.0 + 1e-6) + 1e-6, "the slope {slope} of ground state {g}'s shift along axis {axis} at {r:?} beyond its bound {bound}");
                        steepest = steepest.max(slope.abs());
                    }
                }
            }
        }
        eprintln!("the largest shift {largest:.2} rad/µs, the steepest slope {steepest:.1} rad/µs/µm, the least bound over its shift {tightest:.2}");
        assert!(largest > 20.0 && steepest > 100.0, "the points read a light shift: {largest}, {steepest}");
    }

    /// A fold's error on a rate is booked by what the rate moves (module documentation, # A rate's error on the
    /// populations): the rate agent's e3 (rate.status.md §4) -- S3's tweezer and repump, a σ+σ− probe pair at 1 mW 20 MHz
    /// red of the free F=2 → F'=3 line, mass ×100 at 1 µK: an atom at rest at the trap's centre --, whose probe drives F=2 →
    /// F'=2, F'=1 the released closure's second model (`released_kept`: the first one's failing folds at the atom, no
    /// explicit state) leaves folded with the bare levels, 247 MHz from their lines where the trap puts them at 200.
    /// The rate equations of the ground states at the point -- each one's jumps into the others, its photons' classes on
    /// their landing states (model.rs `Photon`) -- have their stationary populations; of the closure's model and of the same
    /// model with the probe's bare drives at the atom they differ by the bare folds' error on the populations, which the
    /// probe's bare units book at the stationary state (each ground state's units' errors weighed by its population): the
    /// booking holds the largest row's difference and the F=1 share's. (Booked by the drives' share of the photons, about a
    /// percent of them, the engine read 1.5 % on the populations where they were 7 % off.)
    #[test]
    fn a_fold_s_rate_error_is_booked_by_the_population_it_moves() {
        let request = request_of("E3_SPSM");
        let light = Light::new(&request);
        let probe = (0..light.frequencies.len()).find(|&f| f != light.trap && light.frequencies[f].beams.len() == 2).expect("the probe pair");
        let model = &Model::new(&request, &light, &released_kept("E3_SPSM")).unwrap();
        let bare = model.bare_units(Component { frequency: probe, counter: false });
        let units = model.units();
        let folded: Vec<usize> = (0..units.len()).filter(|&u| bare.contains(&units[u])).collect();
        let at = model.rekept(&[model.kept.clone(), bare].concat()).unwrap();
        let explicit: Vec<usize> = model.clusters.iter().chain(&at.clusters).flat_map(|c| c.states[c.ground..].to_vec()).collect();
        assert!(!folded.is_empty() && explicit.is_empty(), "the probe's bare drives ({}), no explicit state: {explicit:?}; kept {:?}", folded.len(), model.kept);
        let state = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
        let centre = model.trap(&reference, &model.prepared()).unwrap().centre;
        let point = Point { r: centre, kappa: [0.0; 3], t: 0.0 };
        let pure = |m: &Model, g: usize| -> Sample {
            let cluster = m.membership[g];
            Sample { cluster, at: point, psi: m.clusters[cluster].states.iter().map(|&s| if s == g { C64::ONE } else { C64::ZERO }).collect() }
        };
        // The stationary populations of a model's ground states' rate equations at the point.
        let stationary = |m: &Model| -> [f64; GROUND] {
            let mut q = [[0.0; GROUND]; GROUND];
            for g in 0..GROUND {
                let sample = pure(m, g);
                for photon in m.photons(sample.cluster, &point, &sample.psi) {
                    let landing = &m.clusters[photon.landing].states;
                    for a in 0..3 {
                        for (j, z) in photon.amplitudes[a].iter().enumerate() {
                            if landing[j] != g {
                                q[g][landing[j]] += z.norm_sqr();
                            }
                        }
                    }
                }
            }
            // π Q = 0 with Σ π = 1, by elimination: the generator's columns, the last equation the normalization.
            let mut a = [[0.0; GROUND + 1]; GROUND];
            for b in 0..GROUND {
                for g in 0..GROUND {
                    a[b][g] = if b == g { -q[g].iter().sum::<f64>() } else { q[g][b] };
                }
            }
            a[GROUND - 1] = [1.0; GROUND + 1];
            for i in 0..GROUND {
                let pivot = (i..GROUND).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs())).unwrap();
                a.swap(i, pivot);
                for k in 0..GROUND {
                    if k != i {
                        let f = a[k][i] / a[i][i];
                        for j in i..=GROUND {
                            a[k][j] -= f * a[i][j];
                        }
                    }
                }
            }
            std::array::from_fn(|g| a[g][GROUND] / a[g][g])
        };
        let (bare, exact) = (stationary(model), stationary(&at));
        let largest = (0..GROUND).map(|g| (bare[g] - exact[g]).abs()).fold(0.0, f64::max);
        let f1 = (bare[..3].iter().sum::<f64>() - exact[..3].iter().sum::<f64>()).abs();
        let booked: f64 = (0..GROUND)
            .map(|g| {
                let errors = model.errors_at(&units, &pure(model, g), None);
                bare[g] * folded.iter().map(|&u| errors[u].population).sum::<f64>()
            })
            .sum();
        // As the run books it: each state's flows against the populations, the lesser of their two sides (`destined`).
        let mut p = [0.0; STATES];
        p[..GROUND].copy_from_slice(&bare);
        let run: f64 = (0..GROUND)
            .map(|g| {
                let errors = model.errors_at(&units, &pure(model, g), None);
                bare[g] * folded.iter().map(|&u| errors[u].population - errors[u].moved * (1.0 - destined(errors[u].into, &p))).sum::<f64>()
            })
            .sum();
        eprintln!("e3 at rest: F=1 {:.4} with the bare folds, {:.4} at the atom; the largest row off by {largest:.4}; booked {booked:.4}, as the run books it {run:.4}", bare[..3].iter().sum::<f64>(), exact[..3].iter().sum::<f64>());
        assert!(f1 > 0.02, "the bare folds move the F=1 share: {f1}");
        assert!(booked >= largest && booked >= f1, "booked {booked} against the rows' {largest} and the F=1 share's {f1}");
        assert!(run >= largest, "as the run books it, {run}, against the rows' {largest}");
    }

    /// The gate of a drive folded at the atom says that its first-order Doppler series converges (module documentation,
    /// # Levels). R2_RECOIL's σ⁺ probe, its drives folded at the atom, made weak and near its line: its beams' couplings
    /// scaled to |V| = Γ/10 on the atoms' |2,2⟩ → |3′,3′⟩, that excited level moved to put the drive Γ from its line at
    /// the trap's centre (D = |Γ − iΓ/2|). |V| below Γ/4: its share across the moving atom's actual distance -- never
    /// below Γ/2 -- is below ½ at any speed, and so is the tail it is the one share of: the gate before. At the speed
    /// where the photon's Doppler shift is 1.5 D (the series in w/D diverges) the gate, the larger of the tail and the
    /// series' one gate of |w| across D, is beyond ½; at rest it is the tail.
    #[test]
    fn a_drive_at_the_atom_fails_where_its_doppler_series_diverges() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R2_RECOIL.json");
        let load = || Request::from_native(&serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()).unwrap();
        let at_the_atom = |request: &Request| -> Model {
            let light = Light::new(request);
            let bare = Model::new(request, &light, &[]).unwrap();
            let kept: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).collect();
            Model::new(request, &light, &kept).unwrap()
        };
        // The strongest drive at the atom of the ground state g at the trap's centre: the centre, its place among g's
        // drives, its component, excited state and width, its size and its detuning there.
        let request = load();
        let g = (0..GROUND).find(|&s| request.initial_density[(s, s)].re == 1.0).unwrap();
        let strongest = |model: &Model| -> ([f64; 3], usize, (usize, usize, f64), Across) {
            let centre = model.trap(&std::array::from_fn(|s| if s == g { 1.0 } else { 0.0 }), &model.prepared()).unwrap().centre;
            let region = model.point_region(centre, 0.0);
            let drives = &region.forms.drives(model)[g];
            let place = (0..drives.len()).filter(|&i| drives[i].level == Level::Atom).max_by(|&a, &b| model.across(g, &region)[a].size.value.total_cmp(&model.across(g, &region)[b].size.value)).expect("a drive folded at the atom");
            (centre, place, (drives[place].component, drives[place].excited, drives[place].gamma), model.across(g, &region)[place])
        };
        let (_, _, (component, excited, gamma), of) = strongest(&at_the_atom(&request));
        let mut weak = load();
        let light = Light::new(&weak);
        for &l in &light.frequencies[component / 2].beams {
            let beam = &mut weak.beams[l];
            for z in beam.coupling.data.iter_mut().chain(beam.counter_coupling.data.iter_mut()) {
                *z = *z * (0.1 * gamma / of.size.value);
            }
        }
        weak.hamiltonian[(GROUND + excited, GROUND + excited)] += C64::real(gamma - of.centre.value);
        let model = at_the_atom(&weak);
        let (centre, place, (component, _, gamma), of) = strongest(&model);
        let group = model.point_region(centre, 0.0).forms.drives(&model)[g][place].group;
        let (size, distance, wave) = (of.size.value, of.distance_at_rest(gamma).value, model.wave_of(component));
        let speed = 1.5 * distance / wave;
        let (rest, moving) = (model.point_region(centre, 0.0), model.point_region(centre, speed));
        let w = wave * speed + 0.5 * model.hbar_over_mass * wave * wave;
        let [gate, _, _, _, tail] = model.folds(g, group, &moving);
        let share = 2.0 * (size / model.across(g, &moving)[place].distance.value).powi(2);
        eprintln!("|V| {:.3} Γ, D {:.3} Γ, k {wave:.3}/µm; at {speed:.2} µm/µs |w/D| {:.2}: its share {share:.3}, the tail {tail:.3}, its gate {gate:.3}", size / gamma, distance / gamma, w / distance);
        assert!(size < 0.25 * gamma && w / distance > 1.0, "|V| {size} against Γ {gamma}, |w/D| {}", w / distance);
        assert!(share < 0.5 && tail < 0.5, "before: its share {share}, the tail {tail}");
        assert!(gate >= 0.5, "its gate {gate} with the Doppler series diverging");
        let [still, _, _, _, still_tail] = model.folds(g, group, &rest);
        eprintln!("at rest: its gate {still:.4}, the tail {still_tail:.4}");
        assert!(still < 0.5 && still_tail < 0.5 && still >= still_tail, "at rest: its gate {still}, the tail {still_tail}");
    }

    /// The slow states of a cluster's generator at a point, the atom at rest: its eigenvalues nearest the real axis, as
    /// many as it has ground states, by their real parts.
    fn slow(model: &Model, cluster: usize, r: [f64; 3]) -> Vec<C64> {
        use crate::step::Middle as _;
        let c = &model.clusters[cluster];
        let middle = model.assemble(cluster, r, 0.0, &model.running());
        let mut h = Mat::zeros(c.states.len(), c.states.len());
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
        let mut values = crate::linalg::eigen(&h).expect("a cluster's generator has its eigenvalues").values;
        values.sort_by(|a, b| b.im.total_cmp(&a.im));
        values.truncate(c.ground);
        values.sort_by(|a, b| a.re.total_cmp(&b.re));
        values
    }

    /// The exact elimination's slow states at a point, the atom at rest: of the no-jump generator of a cluster's ground
    /// states with every excited state -- the cluster's own generator without its drives folded at the atom, those
    /// drives' couplings V, and the excited block K − δ the eliminations take there --, the eigenvalues nearest the real
    /// axis, by their real parts.
    fn eliminated(model: &Model, cluster: usize, carrier: usize, r: [f64; 3]) -> Vec<C64> {
        let n = model.clusters[cluster].ground;
        let mut values = crate::linalg::eigen(&no_jump(model, cluster, carrier, r)).expect("the generator has its eigenvalues").values;
        values.sort_by(|a, b| b.im.total_cmp(&a.im));
        values.truncate(n);
        values.sort_by(|a, b| a.re.total_cmp(&b.re));
        values
    }

    /// The no-jump generator of a cluster's ground states with every excited state at a point, the atom at rest
    /// (`eliminated`): the ground states first.
    fn no_jump(model: &Model, cluster: usize, carrier: usize, r: [f64; 3]) -> Mat {
        use crate::step::Middle as _;
        let (c, cc) = (&model.clusters[cluster], &model.carriers[carrier]);
        let n = c.ground;
        assert!(c.frames.iter().all(|frame| *frame == c.frames[0]), "the cluster's ground states share a frame");
        let amplitudes = model.light.amplitudes(r);
        let middle = model.middle_of(&model.plan(cluster, &model.running()), 0.0, &amplitudes, None);
        let mut ground = Mat::zeros(n, n);
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut ground);
        let h = &model.far_hamiltonian(&amplitudes, &model.running(), 0, &model.coherent_dressing())[0];
        let frame = model.omega(&c.frames[0]) + cc.detuning;
        Mat::from_fn(n + EXCITED, n + EXCITED, |x, y| match (x < n, y < n) {
            (true, true) => ground[(x, y)],
            (false, true) => cc.beams.iter().zip(&cc.folded).map(|(&l, m)| amplitudes[l].value * m[(GROUND + x - n, c.states[y])]).sum(),
            (true, false) => cc.beams.iter().zip(&cc.folded).map(|(&l, m)| amplitudes[l].value * m[(GROUND + y - n, c.states[x])]).sum::<C64>().conj(),
            (false, false) => h[(GROUND + x - n, GROUND + y - n)] - model.decay[(x - n, y - n)] * C64::new(0.0, 0.5) - if x == y { C64::real(frame) } else { C64::ZERO },
        })
    }

    /// Where a jump lands, against the exact elimination (model.rs `Model::landed`; assemble.rs, # The fourth order): the
    /// lin-perp-lin scenes, the cluster F=2 m = −2, 0, 2 with its D2 drives at the atom at the fourth order. A bare
    /// ground state f is the slow subspace's part of it and a fast part that decays at once: exactly, the part is
    /// P (f, 0) with P the spectral projector of the no-jump generator onto its three slow eigenvectors, the prompt
    /// photon's probability 1 − ‖P (f, 0)‖²/‖f‖², and the slow state's coordinates (1 + ½X) times the part's ground
    /// amplitudes. The model's p and Π f are these to the next order, s² -- p itself is about half the landing state's
    /// share --, and the norm is kept on average by construction: ‖Π f‖² + p ‖f‖² = ‖f‖².
    #[test]
    fn a_jump_s_landing_is_the_exact_elimination_s_slow_part() {
        for scene in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
            let (closed, centre) = closed(scene);
            let states = closed.clusters[closed.membership[3]].states.clone();
            let cooling = Component { frequency: closed.driven.iter().find(|d| d.2 == 3 && closed.level(d.0, d.1, d.2) == Level::Atom).expect("a drive at the atom").0.frequency, counter: false };
            let line = closed.driven.iter().filter(|&&(c, e, g)| c == cooling && states.contains(&g) && closed.bare_distance(c, e, g) < 5000.0);
            let more: Vec<Unit> = line.map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).chain([Unit::Dressing { carrier: cooling }, Unit::Second { ground: 3 }]).collect();
            let fourth = closed.rekept(&[closed.kept.clone(), more].concat()).unwrap();
            let (cluster, carrier) = (fourth.membership[3], fourth.carrier_of(cooling.frequency).unwrap());
            let plan = fourth.plan(cluster, &fourth.running());
            assert!(plan.fourth.is_some(), "{scene}: the cluster's fold is the fourth order's");
            let (mut s, mut worst, mut turned) = (0.0_f64, 0.0_f64, 0.0_f64);
            for offset in [0.0, 0.05, 0.1, 0.17, 0.26] {
                let r = [centre[0], centre[1], centre[2] + offset];
                let at = Point { r, kappa: [0.0; 3], t: 0.0 };
                let exact = crate::linalg::eigen(&no_jump(&fourth, cluster, carrier, r)).unwrap();
                let mut order: Vec<usize> = (0..exact.values.len()).collect();
                order.sort_by(|&a, &b| exact.values[b].im.total_cmp(&exact.values[a].im));
                for f in [[C64::ONE, C64::ZERO, C64::ZERO], [C64::ZERO, C64::ONE, C64::ZERO], [C64::real(0.6), C64::new(0.0, 0.5), C64::real(-0.4)]] {
                    let size = crate::linalg::norm_sqr(&f);
                    // P (f, 0) = Σ_slow r_i (R⁻¹ (f, 0))_i.
                    let mut part = vec![C64::ZERO; 3 + EXCITED];
                    for &i in &order[..3] {
                        let weight: C64 = (0..3).map(|y| exact.inverse[(i, y)] * f[y]).sum();
                        for (x, z) in part.iter_mut().enumerate() {
                            *z += exact.vectors[(x, i)] * weight;
                        }
                    }
                    let prompt = 1.0 - crate::linalg::norm_sqr(&part) / size;
                    let slow = fourth.landed(&plan, None, &at, &f);
                    let p = 1.0 - crate::linalg::norm_sqr(&slow) / size;
                    worst = worst.max((p - prompt).abs());
                    s = s.max(2.0 * prompt);
                    // The slow state's direction: Π f against the exact part's ground amplitudes (to their common first
                    // order they differ by the metric's half, of the order of p itself; the angle between them is s²).
                    let overlap: C64 = (0..3).map(|x| slow[x].conj() * part[x]).sum();
                    let ground: f64 = part[..3].iter().map(|z| z.norm_sqr()).sum();
                    turned = turned.max(1.0 - overlap.norm_sqr() / (crate::linalg::norm_sqr(&slow) * ground));
                }
            }
            eprintln!("{scene}: the largest prompt photon's probability {:.4}; the model's differs by at most {worst:.2e}; the slow state turned by at most {turned:.2e}", 0.5 * s);
            assert!(worst <= 1.5 * s * s, "{scene}: p within the next order: {worst} for s = {s}");
            assert!(turned <= s * s, "{scene}: the slow state within the next order: {turned} for s = {s}");
        }
    }

    /// The fold's orders against the exact elimination, the atom frozen (module documentation, # The fold's order;
    /// assemble.rs, # The fourth order): the lin-perp-lin scenes at 40, 80 and 160 mW/cm² a beam, 20 linewidths red, on
    /// ⁸⁷Rb's real F′ = 1, 2, 3 in the tweezer. The cluster F=2 m = −2, 0, 2 with every cooling drive into F′=3 folded at
    /// the atom, at the second order and at the fourth, against the slow eigenvalues of the same ground states with all
    /// the excited states (`eliminated`; the dressing coherent, so that the levels are the same in both), at points across
    /// a fringe. Of the three slow states, with s the largest share 2 Σ_e |R[e, g]|² met: the potentials err by s/2 of the
    /// light's largest shift there at the second order and by s²/2 at the fourth; the force on them (the potentials'
    /// slope along the fringes, a central difference), relative to the steepest potential met, by s and by 3/2 s²; their
    /// decay rates by 3/2 s and by the next order. And the fourth order's generator is a Lindblad one as the second's: a
    /// state's norm falls at its photons' rate.
    #[test]
    fn the_fourth_order_s_potentials_forces_and_rates_against_the_exact_elimination() {
        for scene in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
            let (closed, centre) = closed(scene);
            let states = closed.clusters[closed.membership[3]].states.clone();
            assert_eq!(states, [3, 5, 7], "the cluster of F=2 m = −2, 0, 2");
            // Every cooling drive from the cluster into D2 -- F′ = 1, 2, 3 -- at the atom (the closure leaves the weaker
            // bare), the dressing coherent.
            let cooling = Component { frequency: closed.driven.iter().find(|d| d.2 == 3 && closed.level(d.0, d.1, d.2) == Level::Atom).expect("a drive at the atom").0.frequency, counter: false };
            let line = closed.driven.iter().filter(|&&(c, e, g)| c == cooling && states.contains(&g) && closed.bare_distance(c, e, g) < 5000.0);
            let more: Vec<Unit> = line.map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).chain([Unit::Dressing { carrier: cooling }]).collect();
            let second = closed.rekept(&[closed.kept.clone(), more].concat()).unwrap();
            let fourth = second.rekept(&[second.kept.clone(), vec![Unit::Second { ground: 3 }]].concat()).unwrap();
            let (cluster, carrier) = (second.membership[3], second.carrier_of(cooling.frequency).unwrap());
            assert_eq!(second.fold_frequency(cluster), Some(cooling.frequency));
            assert_eq!((second.fourth[cluster], fourth.fourth[fourth.membership[3]]), (None, Some(cooling.frequency)));
            assert!(second.units().contains(&Unit::Second { ground: 3 }) && !fourth.units().contains(&Unit::Second { ground: 3 }));
            let (mut s, mut largest, mut steepest) = (0.0_f64, 0.0_f64, 0.0_f64);
            let (mut potential, mut force, mut rate) = ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
            let step = 1e-4;
            for offset in [0.0, 0.03, 0.07, 0.1, 0.15, 0.2, 0.26, 0.33] {
                let r = [centre[0], centre[1], centre[2] + offset];
                let read = |values: &dyn Fn(f64) -> Vec<C64>| -> (Vec<C64>, Vec<f64>) {
                    let (up, down) = (values(step), values(-step));
                    (values(0.0), up.iter().zip(&down).map(|(a, b)| (a.re - b.re) / (2.0 * step)).collect())
                };
                let (exact, pull) = read(&|by| eliminated(&second, cluster, carrier, [r[0], r[1], r[2] + by]));
                let level = {
                    use crate::step::Middle as _;
                    let middle = second.middle_of(&second.plan(cluster, &second.running()), 0.0, &second.light.amplitudes(r), None);
                    let mut h = Mat::zeros(3, 3);
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                    crate::linalg::eigen(&h).unwrap().values
                };
                let mean = level.iter().map(|z| z.re).sum::<f64>() / 3.0;
                let shift = exact.iter().map(|z| (z.re - mean).abs()).fold(0.0, f64::max);
                let decay = exact.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
                largest = largest.max(shift);
                steepest = pull.iter().map(|f| f.abs()).fold(steepest, f64::max);
                for (order, model) in [&second, &fourth].into_iter().enumerate() {
                    let (values, slopes) = read(&|by| slow(model, model.membership[3], [r[0], r[1], r[2] + by]));
                    for i in 0..3 {
                        potential[order] = potential[order].max((values[i].re - exact[i].re).abs() / shift);
                        force[order] = force[order].max((slopes[i] - pull[i]).abs());
                        rate[order] = rate[order].max((values[i].im - exact[i].im).abs() / decay);
                    }
                }
                // The largest share at the point, and the fourth order's hazard identity on a state of the cluster.
                let psi: Vec<C64> = (0..3).map(|x| C64::cis(1.3 * x as f64) * (0.2 + 0.3 * x as f64)).collect();
                let here = Point { r, kappa: [0.0; 3], t: 0.0 };
                let sample = Sample { cluster, at: here, psi: psi.clone() };
                let at = second.measured(&sample, None);
                let response = at.responses[carrier].as_ref().unwrap();
                s = (0..3).map(|y| 2.0 * (0..EXCITED).map(|x| response[(x, y)].norm_sqr()).sum::<f64>()).fold(s, f64::max);
                let plan = fourth.plan(fourth.membership[3], &fourth.running());
                let photons: f64 = fourth.photons_of(&plan, None, &here, &psi, crate::model::photon::Gradients::None).0.iter().map(|p| p.amplitudes.iter().map(|u| crate::linalg::norm_sqr(u)).sum::<f64>()).sum();
                let mut h = Mat::zeros(3, 3);
                {
                    use crate::step::Middle as _;
                    let middle = fourth.assemble(fourth.membership[3], r, 0.0, &fourth.running());
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                }
                let falls: f64 = (0..3).flat_map(|x| (0..3).map(move |y| (x, y))).map(|(x, y)| (psi[x].conj() * C64::I * (h[(x, y)] - h[(y, x)].conj()) * psi[y]).re).sum();
                // The potential is the same generator's (V_L is the light set's less the same without the trap light): a
                // state's potential differs between the orders by the diagonals of what the fourth adds to the two
                // generators, and its gradient is its own slope.
                let dark = crate::model::Lights((0..fourth.light.frequencies.len()).map(|w| if w == fourth.light.trap { crate::model::Lit::Off } else { crate::model::Lit::Coherent }).collect());
                let diagonal = |model: &Model, lights: &crate::model::Lights, place: usize| -> f64 {
                    use crate::step::Middle as _;
                    let middle = model.assemble(model.membership[3], r, 0.0, lights);
                    let mut g = Mat::zeros(3, 3);
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut g);
                    g[(place, place)].re
                };
                for (place, &state) in states.iter().enumerate() {
                    let alone: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
                    let of = |model: &Model, by: f64| model.potential([r[0], r[1], r[2] + by], &alone, &model.running());
                    let lit = fourth.running();
                    let more = diagonal(&fourth, &lit, place) - diagonal(&second, &lit, place) - (diagonal(&fourth, &dark, place) - diagonal(&second, &dark, place));
                    let added = of(&fourth, 0.0) - of(&second, 0.0);
                    assert!((added - more).abs() <= 1e-9 * largest.max(1.0), "{scene}: the potential's fourth order {added} against the generator's {more}");
                    let slope = fourth.expand(r, &alone, &fourth.running(), 1, None).quadratic.gradient[2];
                    let difference = (of(&fourth, step) - of(&fourth, -step)) / (2.0 * step);
                    // (The central difference's own error: (2k step)²/6 of the slope, 4e-7.)
                    assert!((slope - difference).abs() <= 2e-6 * slope.abs() + 1e-6, "{scene}: the potential's gradient {slope} against its slope {difference}");
                }
                // To the product of the fourth order and the first-order Doppler term (here the recoil shift, 3e-5 of the
                // detuning, times s), which the photons' product holds and the middle's decay, made at rest, does not.
                assert!((falls - photons).abs() <= 1e-5 * photons, "{scene}: the norm falls at {falls}, the photons' rate is {photons}");
            }
            let force = force.map(|f| f / steepest);
            eprintln!("{scene}: s = {s:.4}, the largest shift {largest:.2} rad/µs: potentials {:.2e} (second), {:.2e} (fourth); forces {:.2e}, {:.2e}; decay rates {:.2e}, {:.2e}", potential[0], potential[1], force[0], force[1], rate[0], rate[1]);
            assert!((0.4 * s..0.6 * s).contains(&potential[0]) && potential[1] <= 0.6 * s * s, "{scene}: the potentials err by s/2 and by s²/2: {potential:?} for s = {s}");
            assert!((0.5 * s..1.0 * s).contains(&force[0]) && force[1] <= 1.5 * s * s, "{scene}: the forces err by s and by 3/2 s² at most: {force:?} for s = {s}");
            assert!((1.2 * s..1.8 * s).contains(&rate[0]) && rate[1] <= 2.5 * s * s, "{scene}: the decay rates err by 3/2 s and by the next order: {rate:?} for s = {s}");
        }
    }

    /// A lin-perp-lin scene's cycling point: where the cooling light at the trap's axis is σ⁻ alone, F=2 m=−2 is driven
    /// into F′=3 m′=−3 and nowhere else, and that state decays to it alone -- a two-level atom. The scene's model with
    /// every cooling drive of m=−2's cluster at the atom and the dressing kept, at the second order and at the fourth;
    /// the point; and of the no-jump generator there the drive's share s = 2|V|²/(Δ² + Γ²/4), the two-level atom's
    /// steady photon rate Γ |V|²/(Δ² + Γ²/4 + 2|V|²) with the ground state's own bare scattering, and the line's
    /// (Γ/2Δ)².
    fn cycling(scene: &str) -> (Model, Model, Point, [f64; 3]) {
        let (closed, centre) = closed(scene);
        let states = closed.clusters[closed.membership[3]].states.clone();
        let cooling = Component { frequency: closed.driven.iter().find(|d| d.2 == 3 && closed.level(d.0, d.1, d.2) == Level::Atom).expect("a drive at the atom").0.frequency, counter: false };
        let line = closed.driven.iter().filter(|&&(c, e, g)| c == cooling && states.contains(&g) && closed.bare_distance(c, e, g) < 5000.0);
        let more: Vec<Unit> = line.map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).chain([Unit::Dressing { carrier: cooling }]).collect();
        let second = closed.rekept(&[closed.kept.clone(), more].concat()).unwrap();
        let fourth = second.rekept(&[second.kept.clone(), vec![Unit::Second { ground: 3 }]].concat()).unwrap();
        let (cluster, carrier) = (second.membership[3], second.carrier_of(cooling.frequency).unwrap());
        assert_eq!(second.clusters[cluster].states[0], 3, "the cluster's first place is F=2 m=−2");
        // The point where the light is σ⁻ alone: where m=−2's coupling is the largest (its σ⁺ part is nothing there).
        // The excited state it drives (in the request's basis a combination of states), its detuning and its width.
        let column = |z: f64| -> (f64, Mat) {
            let g = no_jump(&second, cluster, carrier, [centre[0], centre[1], z]);
            ((0..EXCITED).map(|e| g[(3 + e, 0)].norm_sqr()).sum(), g)
        };
        let grid: Vec<f64> = (0..400).map(|i| centre[2] + 0.4 * i as f64 / 400.0).collect();
        let best = grid.iter().copied().max_by(|a, b| column(*a).0.total_cmp(&column(*b).0)).unwrap();
        let (mut lo, mut hi) = (best - 0.001, best + 0.001);
        for _ in 0..60 {
            let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if column(a).0 > column(b).0 { hi = b } else { lo = a }
        }
        let z = 0.5 * (lo + hi);
        let (v2, g) = column(z);
        let bright: Vec<C64> = (0..EXCITED).map(|e| g[(3 + e, 0)] * (1.0 / v2.sqrt())).collect();
        let level: C64 = (0..EXCITED * EXCITED).map(|i| bright[i / EXCITED].conj() * g[(3 + i / EXCITED, 3 + i % EXCITED)] * bright[i % EXCITED]).sum();
        let (gamma, delta) = (-2.0 * level.im, level.re - g[(0, 0)].re);
        let s = 2.0 * v2 / (delta * delta + 0.25 * gamma * gamma);
        let exact = gamma * v2 / (delta * delta + 0.25 * gamma * gamma + 2.0 * v2) - 2.0 * g[(0, 0)].im;
        let at = Point { r: [centre[0], centre[1], z], kappa: [0.0; 3], t: 0.0 };
        (second, fourth, at, [s, exact, 0.25 * gamma * gamma / (delta * delta)])
    }

    /// A cycling state's photons per unit time, against the two-level atom's (module documentation, # The fold's order;
    /// `cycling`). With the drive's share s: the second-order fold's rate is too high by s; the fourth order's jumps
    /// alone are too low by s/2; with the landing -- every jump back into the bare state followed by one more photon
    /// with the probability p, and so on: 1/(1 − p) photons a jump, these lines being far from resonance -- the rate is
    /// right to the next order: low by s², to the order after that, within `ORDER_RATES` s².
    #[test]
    fn a_cycling_state_s_photon_rate_with_the_landing_is_the_two_level_atom_s() {
        for scene in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
            let (second, fourth, at, [s, exact, width]) = cycling(scene);
            let psi = [C64::ONE, C64::ZERO, C64::ZERO];
            let rate = |model: &Model| -> f64 {
                let plan = model.plan(model.membership[3], &model.running());
                model.photons_of(&plan, None, &at, &psi, crate::model::photon::Gradients::None).0.iter().map(|p| p.amplitudes.iter().map(|u| crate::linalg::norm_sqr(u)).sum::<f64>()).sum()
            };
            let plan = fourth.plan(fourth.membership[3], &fourth.running());
            let prompt = 1.0 - crate::linalg::norm_sqr(&fourth.landed(&plan, None, &at, &psi));
            assert!(prompt > 0.0, "{scene}: far from its line the landing's norm is below one: p = {prompt}");
            let errors = [rate(&second) / exact - 1.0, rate(&fourth) / exact - 1.0, rate(&fourth) / (1.0 - prompt) / exact - 1.0];
            // What the comparison cannot resolve: the second order's rate is the two-level atom's times (1 + s) but for
            // the line's width over its detuning, squared.
            let floor = (errors[0] - s).abs();
            let rest = (errors[2] - (errors[0] - s)) / (s * s);
            eprintln!("{scene}: s = {s:.4}, p = {prompt:.4}: the photon rate against the two-level atom's -- second order {:+.3e} (s {:+.1e}), fourth {:+.3e}, fourth with the landing {:+.3e} ({rest:+.3} s²)", errors[0], errors[0] - s, errors[1], errors[2]);
            assert!(floor <= width, "{scene}: the second order's rate is too high by s: {} for s = {s}", errors[0]);
            assert!((errors[1] + 0.5 * s).abs() <= ORDER_RATES * s * s + floor, "{scene}: the fourth order's jumps alone too low by s/2: {} for s = {s}", errors[1]);
            assert!((rest + 1.0).abs() <= s && rest.abs() <= ORDER_RATES, "{scene}: with the landing the rate is low by s², to the order after: {} for s = {s}", errors[2]);
        }
    }

    /// The landing's norm against the threshold (model.rs `Model::landed`, step.rs `jump`) is the two-level atom's rate
    /// to the fourth order on and off resonance, by one rule. Two exact levels (`three_levels` with no second light:
    /// g –V→ e alone, the drive folded at the atom at the fourth order, the atom at κ = −k/2 where its Doppler shift is
    /// nothing), Δ from 0 to 6 Γ on both sides, s = 2|V|²/(Δ² + Γ²/4) = 0.03, 0.06, 0.12. With the slow state's rate γ off
    /// the generator and the landing's norm n = ‖Π g‖², the waiting time after a jump is P(T > t) = min(1, n e^{−γt}), its
    /// mean (max(0, ln n) + min(1, n))/γ, against the exact steady rate Γ|V|²/(Δ² + Γ²/4 + 2|V|²):
    /// - n = 1 − p with p = (s/2)(Δ² − 3Γ²/4)/(Δ² + Γ²/4) to its order: above one within 0.866 Γ of the line (1 + 3s/2 on
    ///   resonance), below one beyond;
    /// - the rate is the exact one within `ORDER_RATES` s² everywhere (+1.07 s² on resonance at s = 0.03, −0.99 s² at
    ///   6 Γ); with the norm clamped at one it was 1.52 s high on resonance;
    /// - `jump` itself on resonance: every jump emits one photon and no more, and a trajectory that lands back in g goes
    ///   on with the norm n, above one.
    #[test]
    fn the_landing_s_norm_against_the_threshold_is_the_two_level_atom_s_rate_on_and_off_resonance() {
        use crate::step::{self, Dynamics, Trajectory};
        let (g, e) = (THREE[0], THREE[2]);
        for s in [0.03, 0.06, 0.12] {
            for delta in [0.0, 0.25, -0.25, 0.5, 0.8, 1.0, -1.0, 2.0, -6.0] {
                let v = (0.5 * s * (delta * delta + 0.25_f64)).sqrt();
                let (request, of, gamma) = three_levels(v, delta, 0.0, 0.0);
                let light = Light::new(&request);
                let component = Component { frequency: of[0], counter: false };
                let model = Model::new(&request, &light, &[Unit::Bare { component, excited: e, ground: g }, Unit::Second { ground: g }]).unwrap();
                let cluster = model.membership[g];
                assert!(model.clusters[cluster].states == [g] && model.fourth[cluster].is_some(), "g alone, its fold at the fourth order");
                let beam = &light.beams[light.frequencies[of[0]].beams[0]];
                let k = beam.wave_vector();
                let at = Point { r: [0.0; 3], kappa: k.map(|x| -0.5 * x), t: 0.0 };
                let rate = {
                    use crate::step::Middle as _;
                    let middle = model.assemble(cluster, at.r, 0.0, &model.running());
                    let mut h = Mat::zeros(1, 1);
                    middle.generator(&middle.path(at.kappa, [0.0; 3]), 0.0, &mut h);
                    -2.0 * h[(0, 0)].im
                };
                let plan = model.plan(cluster, &model.running());
                let n = crate::linalg::norm_sqr(&model.landed(&plan, None, &at, &[C64::ONE]));
                let p = 0.5 * s * (delta * delta - 0.75) / (delta * delta + 0.25);
                assert!((1.0 - n - p).abs() <= 1.5 * s * s, "s = {s}, Δ = {delta} Γ: the landing's norm {n} against 1 − p, p = {p}");
                assert!((n > 1.0) == (delta.abs() < 0.75_f64.sqrt()), "s = {s}, Δ = {delta} Γ: the norm {n} is above one within 0.866 Γ of the line");
                let (vr, d) = (v * gamma, delta * gamma);
                let exact = gamma * vr * vr / (d * d + 0.25 * gamma * gamma + 2.0 * vr * vr);
                let mean = (n.ln().max(0.0) + n.min(1.0)) / rate;
                let error = 1.0 / (mean * exact) - 1.0;
                println!("s = {s}, Δ = {delta:+.2} Γ: n − 1 = {:+.3} s, the rate {:+.3} s² of the exact one (the norm clamped at one: {:+.3} s)", (n - 1.0) / s, error / (s * s), (rate / n.min(1.0) / exact - 1.0) / s);
                assert!(error.abs() <= ORDER_RATES * s * s, "s = {s}, Δ = {delta} Γ: the rate {error:+.3e} of the exact one, s² = {:.3e}", s * s);
                if delta == 0.0 {
                    assert!(((rate / exact - 1.0) / s - 1.5).abs() < 0.1, "on resonance the clamped norm's rate is 1.5 s high");
                    let mut reference = [0.0; GROUND];
                    reference[g] = 1.0;
                    let run = crate::model::Run::new(&model, reference);
                    let mut back = 0;
                    for index in 0..200 {
                        let mut t = Trajectory::new(&run, g, at.r, at.kappa, 11, index);
                        step::jump(&run, &mut t).unwrap();
                        assert_eq!((t.work / run.photon_work()).round(), 1.0, "one photon a jump where the landing's norm is above one");
                        let norm = crate::linalg::norm_sqr(&t.psi);
                        if t.cluster == cluster {
                            back += 1;
                            assert!(norm > 1.0 && (norm - n).abs() <= 0.2 * (n - 1.0), "the trajectory goes on with the landing's norm: {norm} against {n}");
                        } else {
                            assert!((norm - 1.0).abs() <= 1e-12, "a second-order landing is the bare state: {norm}");
                        }
                    }
                    assert!(back > 0, "some jumps land back in g");
                }
            }
        }
    }

    /// One ground state driven by one frequency to several excited states (`three_levels`' atom, no second light): g
    /// –V_i→ e_i, each line (its coupling, its detuning, in Γ) one drive folded at the atom at the fourth order. The
    /// request; the frequency; the excited states with their own widths (the scene's, equal to its eight digits); Γ.
    fn fan(lines: &[(f64, f64)]) -> (Request, usize, Vec<(usize, f64)>, f64) {
        let (mut request, of, gamma) = three_levels(0.0, 0.0, 0.0, 0.0);
        let light = Light::new(&request);
        let nu = Model::new(&request, &light, &[]).unwrap().tones[of[0]].nu[0];
        let width = |e: usize| -> f64 { request.emission.iter().map(|c| (0..GROUND).map(|s| c[(s, e)].norm_sqr()).sum::<f64>()).sum() };
        let excited: Vec<(usize, f64)> = (0..lines.len()).map(|i| (GROUND + i, width(GROUND + i))).collect();
        for (&(e, width), &(v, delta)) in excited.iter().zip(lines) {
            assert!((width - gamma).abs() <= 1e-6 * gamma, "the excited states share one width: {width} against {gamma}");
            request.beams[light.frequencies[of[0]].beams[0]].coupling[(e, THREE[0])] = C64::real(v * gamma);
            request.hamiltonian[(e, e)] = C64::real(delta * gamma + nu);
        }
        (request, of[0], excited, gamma)
    }

    /// What the fourth order leaves of a ground state's several drives (module documentation, # The fold's order), against
    /// the exact atom: g driven by one frequency to one, two and three excited states (`fan`; each line its share s =
    /// 2|V|²/(Δ² + Γ²/4) and its detuning in Γ), every drive folded at the atom at the fourth order, the atom where its
    /// Doppler shift is nothing. The rate as the two-level test above reads it -- the mean waiting time after a jump
    /// with the slow state's rate and the landing's norm against the threshold -- against the exact mean waiting time
    /// from g, ∫ ‖e^{−iHt} g‖² dt of the no-jump generator of g with its excited states; the potential, the generator's
    /// real part against the exact slow eigenvalue's, over the light's shift Σ |V|²/|Δ − iΓ/2|, of which the force's
    /// error is three times (a sixth order in the field against the second).
    /// - At one detuning the drives are one line of the share Σs: the rate's remainder is +0.87 to +1.07 (Σs)² on
    ///   resonance and −0.98 to −0.99 (Σs)² at 6 Γ whatever the number of drives and their shares, within `ORDER_RATES`
    ///   (Σs)² -- and 1.7 to 2.6 times `ORDER_RATES` Σs² (the drives booked one by one) with two and three of them.
    /// - At different distances (Σs)² is no bound: shares of 0.04 at 3 Γ and at 30 Γ leave −7.2 (Σs)², the near line
    ///   reading the far one's light shift over its own distance (L = 5.4 Σs).
    /// - Everywhere the remainder is within what the drives' units book (`Model::drive_error`, p_g = 1): the rate's
    ///   within the populations' (1.05 to 4.3 of it where every parameter is small; 26 of it where the far line moves
    ///   the near one by 1.8 of its distance, no small parameter: 0.61 booked), three times the potential's within T's.
    /// - The same atom with its fold at the second order (`Unit::Second` not kept: its jumps' rate Σ Γ |R|² against the
    ///   same waiting time, twice the potential's error for the force's): at one detuning the rate is high by Σs to the
    ///   widths' eight digits, and booked so; at different distances by more -- 2.72 Σs with shares of 0.04 at 3 Γ and
    ///   at 30 Γ, 3.10 Σs with 0.01 at 1 Γ and 0.05 at 40 Γ, 1.19 Σs at 20, 64 and 90 Γ --, within what the cluster's
    ///   unit books (`Model::order_error`: 3.21, 5.83, 1.26 Σs); the force's stays within Σs (0.94 Σs at most).
    #[test]
    fn a_ground_state_s_drives_leave_the_fourth_order_one_line_s_remainder() {
        let g = THREE[0];
        let together: Vec<Vec<(f64, f64)>> = [0.0, 0.5, -1.0, -6.0]
            .into_iter()
            .flat_map(|delta| [vec![(0.03, delta)], vec![(0.03, delta); 2], vec![(0.03, delta); 3], vec![(0.06, delta); 3], vec![(0.06, delta), (0.02, delta), (0.01, delta)]])
            .collect();
        let decade = vec![(0.04, -3.0), (0.04, -30.0)];
        let apart = [
            vec![(0.04, -3.0), (0.04, -3.5)],
            vec![(0.04, -3.0), (0.04, -6.0)],
            decade.clone(),
            vec![(0.04, -3.0), (0.04, 5.0)],
            vec![(0.04, 0.0), (0.04, -3.0)],
            vec![(0.01, -1.0), (0.05, -40.0)],
            vec![(0.06, -20.0), (0.006, -64.0), (0.002, -90.0)],
            vec![(0.03, -0.3), (0.03, 0.0), (0.03, 0.3)],
        ];
        for (index, case) in together.iter().chain(&apart).enumerate() {
            let lines: Vec<(f64, f64)> = case.iter().map(|&(s, delta)| ((0.5 * s * (delta * delta + 0.25_f64)).sqrt(), delta)).collect();
            let (request, of, excited, gamma) = fan(&lines);
            let light = Light::new(&request);
            let component = Component { frequency: of, counter: false };
            let kept: Vec<Unit> = excited.iter().map(|&(e, _)| Unit::Bare { component, excited: e, ground: g }).chain([Unit::Second { ground: g }]).collect();
            let model = Model::new(&request, &light, &kept).unwrap();
            let cluster = model.membership[g];
            assert!(model.clusters[cluster].states == [g] && model.fourth[cluster].is_some(), "g alone, its fold at the fourth order");
            let k = light.beams[light.frequencies[of].beams[0]].wave_vector();
            let at = Point { r: [0.0; 3], kappa: k.map(|x| -0.5 * x), t: 0.0 };
            let h = {
                use crate::step::Middle as _;
                let middle = model.assemble(cluster, at.r, 0.0, &model.running());
                let mut h = Mat::zeros(1, 1);
                middle.generator(&middle.path(at.kappa, [0.0; 3]), 0.0, &mut h);
                h[(0, 0)]
            };
            let plan = model.plan(cluster, &model.running());
            let n = crate::linalg::norm_sqr(&model.landed(&plan, None, &at, &[C64::ONE]));
            let mean = (n.ln().max(0.0) + n.min(1.0)) / (-2.0 * h.im);
            // The exact no-jump generator of g and its excited states; its mean waiting time from g, M[g, g] of
            // i(H†M − MH) = −1 (M = ∫ e^{iH†t} e^{−iHt} dt); its slow eigenvalue.
            let m = lines.len() + 1;
            let exact = Mat::from_fn(m, m, |x, y| match (x, y) {
                (0, 0) => C64::ZERO,
                (0, y) => C64::real(lines[y - 1].0 * gamma),
                (x, 0) => C64::real(lines[x - 1].0 * gamma),
                (x, y) if x == y => C64::new(lines[x - 1].1 * gamma, -0.5 * excited[x - 1].1),
                _ => C64::ZERO,
            });
            let lyapunov = Mat::from_fn(m * m, m * m, |row, column| {
                let ((x, y), (a, b)) = ((row / m, row % m), (column / m, column % m));
                let mut z = C64::ZERO;
                if b == y {
                    z += C64::I * exact[(a, x)].conj();
                }
                if a == x {
                    z -= C64::I * exact[(b, y)];
                }
                z
            });
            let mut waiting: Vec<C64> = (0..m * m).map(|i| if i / m == i % m { C64::real(-1.0) } else { C64::ZERO }).collect();
            crate::linalg::Lu::new(&lyapunov).unwrap().solve(&mut waiting);
            let slow = crate::linalg::eigen(&exact).unwrap().values.into_iter().max_by(|a, b| a.im.total_cmp(&b.im)).unwrap();
            let rate = waiting[0].re / mean - 1.0;
            let shift: f64 = lines.iter().map(|&(v, delta)| v * v * gamma / (delta * delta + 0.25_f64).sqrt()).sum();
            let potential = (h.re - slow.re).abs() / shift;
            let (sum, squares): (f64, f64) = (case.iter().map(|c| c.0).sum(), case.iter().map(|c| c.0 * c.0).sum());
            let units: Vec<Unit> = excited.iter().map(|&(e, _)| Unit::Drive { component, excited: e, ground: g }).collect();
            let sample = Sample { cluster, at, psi: vec![C64::ONE] };
            let errors = model.errors(&units, &[sample]);
            let booked: [f64; 2] = [errors.iter().map(|e| e[0].population).sum(), errors.iter().map(|e| e[0].temperature).sum()];
            for (error, &(share, _)) in errors.iter().zip(case) {
                assert!((error[0].parameter - share).abs() <= 1e-6 * share, "{case:?}: a drive's share at the atom {} against {share}", error[0].parameter);
            }
            println!(
                "{case:?}: the rate {rate:+.3e} = {:+.3} (Σs)² = {:+.3} Σs², booked {:.3e}; three times the potential {:.3e} = {:.3} (Σs)², booked on T {:.3e}",
                rate / (sum * sum),
                rate / squares,
                booked[0],
                3.0 * potential,
                3.0 * potential / (sum * sum),
                booked[1]
            );
            assert!(rate.abs() <= booked[0], "{case:?}: the rate's remainder {rate:+.3e} beyond the drives' booked {:.3e}", booked[0]);
            assert!(3.0 * potential <= booked[1], "{case:?}: three times the potential's remainder {:.3e} beyond the drives' booked {:.3e} on T", 3.0 * potential, booked[1]);
            if index < together.len() {
                assert!(rate.abs() <= ORDER_RATES * sum * sum, "{case:?}: at one detuning the remainder is one line's of the share Σs: {:+.3} (Σs)²", rate / (sum * sum));
                let ends = case[0].1 == 0.0 || case[0].1 == -6.0;
                assert!(case.len() == 1 || !ends || rate.abs() > 1.5 * ORDER_RATES * squares, "{case:?}: and not the drives' one by one: {:+.3} Σs²", rate / squares);
            }
            if *case == decade {
                assert!(rate.abs() > 5.0 * ORDER_RATES * sum * sum, "{case:?}: a decade apart (Σs)² is no bound: {:+.3} (Σs)²", rate / (sum * sum));
            }
            // The second order's own (no `Unit::Second` kept): its jumps' rate against the exact waiting time, twice the
            // potential's error, and what the cluster's unit books of them.
            let second = Model::new(&request, &light, &kept[..kept.len() - 1]).unwrap();
            let lower = {
                use crate::step::Middle as _;
                let middle = second.assemble(second.membership[g], at.r, 0.0, &second.running());
                let mut h = Mat::zeros(1, 1);
                middle.generator(&middle.path(at.kappa, [0.0; 3]), 0.0, &mut h);
                h[(0, 0)]
            };
            let state = Sample { cluster: second.membership[g], at, psi: vec![C64::ONE] };
            let order = second.errors(&[Unit::Second { ground: g }], &[state])[0][0];
            let (rate, force) = (waiting[0].re * (-2.0 * lower.im) - 1.0, 2.0 * (lower.re - slow.re).abs() / shift);
            println!("    its second order: the rate {rate:+.3e} = {:+.3} Σs, booked {:.3e}; twice the potential {force:.3e} = {:.3} Σs, booked on T {:.3e}", rate / sum, order.population, force / sum, order.temperature);
            // (The excited states' widths are one to the scene's eight digits: so are the one line's statements.)
            let exactly = 1.0 + 1e-6;
            assert!(rate.abs() <= order.population * exactly, "{case:?}: the second order's rate {rate:+.3e} beyond its booked {:.3e}", order.population);
            assert!(force <= order.temperature * exactly, "{case:?}: the second order's force {force:.3e} beyond its booked {:.3e} on T", order.temperature);
            if index < together.len() {
                assert!((rate - sum).abs() <= 1e-6 * sum && (order.population - sum).abs() <= 1e-6 * sum, "{case:?}: at one detuning the second order is high by Σs exactly, and booked so: {rate:e}, {:e}", order.population);
            }
            if *case == decade {
                assert!(rate > 2.0 * sum, "{case:?}: a decade apart Σs is no bound of the second order: {:+.3} Σs", rate / sum);
            }
        }
    }

    /// A jump at the fourth order (step.rs `jump`, the landing), on the cycling state at rest (`cycling`): of single jumps
    /// with their own streams, the photons a jump -- each one's counted work -- are 1/(1 − p) on average, within four of
    /// their mean's deviations (a geometric number's, √p/(1 − p) a jump).
    #[test]
    fn a_jump_at_the_fourth_order_draws_its_landing_s_photons() {
        use crate::step::{self, Dynamics, Trajectory};
        for scene in ["R4_LIN_1", "R4_LIN_4"] {
            let (_, fourth, at, _) = cycling(scene);
            let mut reference = [0.0; GROUND];
            reference[3] = 1.0;
            let run = crate::model::Run::new(&fourth, reference);
            let prompt = 1.0 - crate::linalg::norm_sqr(&run.landed(fourth.membership[3], &at, &[C64::ONE, C64::ZERO, C64::ZERO]));
            let trials = 4000u64;
            let mut photons = 0.0;
            for index in 0..trials {
                let mut t = Trajectory::new(&run, 3, at.r, [0.0; 3], 7, index);
                step::jump(&run, &mut t).unwrap();
                photons += (t.work / run.photon_work()).round();
            }
            let (mean, expected) = (photons / trials as f64, 1.0 / (1.0 - prompt));
            let deviation = prompt.sqrt() / (1.0 - prompt) / (trials as f64).sqrt();
            eprintln!("{scene}: p = {prompt:.4}: {mean:.4} photons a jump of {trials} jumps, 1/(1 − p) = {expected:.4} ± {deviation:.4}");
            assert!((mean - expected).abs() <= 4.0 * deviation, "{scene}: {mean} photons a jump against {expected} ± {deviation}");
        }
    }

    /// A middle made of given near forms carries the fourth order's part made of the same forms (assemble.rs
    /// `Model::middle_of`): on the lin-perp-lin scene with the cluster of F=2 m=−2 at the fourth order, at points along a
    /// fringe and off the axis, under the running lights and without the trap's, its generator on a state is the point's
    /// own middle's (the exact elimination's forms both) to their rounding, and differs from the second order's by the
    /// fourth order's size. So a table's error read against the exact elimination (tabulate.rs `Model::table_reads`) is
    /// the table's: of a grid with no axis -- the forms without the trap's light do not vary -- nothing, as it is at the
    /// second order; of one with axes, its interpolation's.
    #[test]
    fn a_middle_of_given_forms_carries_the_fourth_order() {
        let (closed, centre) = closed("R4_LIN_2");
        let second = closed.rekept(&closed.kept).unwrap();
        let fourth = closed.rekept(&[closed.kept.clone(), vec![Unit::Second { ground: 3 }]].concat()).unwrap();
        let cluster = fourth.membership[3];
        assert!(fourth.fourth[cluster].is_some() && second.fourth.iter().all(Option::is_none));
        let running = fourth.running();
        let mut untrapped = running.clone();
        untrapped.0[fourth.light.trap] = crate::model::Lit::Off;
        let n = fourth.clusters[cluster].states.len();
        let psi: Vec<C64> = (0..n).map(|x| C64::new(1.0 / (1.0 + x as f64), 0.3 * x as f64)).collect();
        let mut samples = vec![];
        for offset in [[0.0, 0.0, 0.0], [0.0, 0.0, 0.07], [0.3, 0.0, 0.19], [0.0, 0.4, 0.31]] {
            let r: [f64; 3] = std::array::from_fn(|i| centre[i] + offset[i]);
            let sample = Sample { cluster, at: Point { r, kappa: [3.0, -2.0, 5.0], t: 0.0 }, psi: psi.clone() };
            for lights in [&running, &untrapped] {
                let amplitudes = fourth.light.amplitudes(r);
                let plan = fourth.plan(cluster, lights);
                let own = fourth.action(&sample, &fourth.middle(&plan, r, 0.0, None));
                let given = fourth.action(&sample, &fourth.middle_of(&plan, 0.0, &amplitudes, Some(&fourth.near_exact(&plan, &amplitudes, 1, &fourth.coherent_dressing()))));
                // The generator reads the forms' values alone, the fourth order's part too: of the forms to order 0 it
                // is the same, bit for bit.
                assert_eq!(given, fourth.action(&sample, &fourth.middle_of(&plan, 0.0, &amplitudes, Some(&fourth.near_exact(&plan, &amplitudes, 0, &fourth.coherent_dressing())))), "at {r:?}: the generator of the forms' values alone");
                let lower = second.action(&sample, &second.middle(&second.plan(cluster, lights), r, 0.0, None));
                let size = |x: &[C64]| crate::linalg::norm_sqr(x).sqrt();
                let moved = |x: &[C64], y: &[C64]| size(&x.iter().zip(y).map(|(a, b)| *a - *b).collect::<Vec<_>>());
                assert!(moved(&own, &given) <= 1e-12 * size(&own), "at {r:?}: a middle of given forms {} from the point's own of {}", moved(&own, &given), size(&own));
                assert!(moved(&own, &lower) > 1e-4 * size(&own), "at {r:?}: the fourth order's part is in the middle: {} of {}", moved(&own, &lower), size(&own));
            }
            samples.push(sample);
        }
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == 7 { 1.0 } else { 0.0 });
        let trap = fourth.trap(&reference, &fourth.prepared()).unwrap();
        let reach = fourth.reach(&trap, &reference, 30.0, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        let tables = fourth.tabulate(&reach, 1_000_000);
        let reads = fourth.table_reads(&tables, &samples);
        let mut read = 0;
        for (table, of) in tables.tables.iter().zip(&reads) {
            if table.layout.cluster != cluster {
                continue;
            }
            let whole = of.whole.iter().map(|e| e.population.max(e.temperature)).fold(0.0, f64::max);
            eprintln!("a table of {} axes: its whole error at the samples {whole:.2e}", table.axes());
            assert!(whole <= if table.axes() == 0 { 1e-12 } else { 1e-5 }, "a table of {} axes errs by {whole}", table.axes());
            read += 1;
        }
        assert_eq!(read, 2, "the cluster's tables: under the running lights, and without the trap's");
    }

    /// A request by its name: a reference scene's (refs/scenes/<name>.json), or one exported as the app sends it for the
    /// gates' tests (refs/requests/<name>.json: the sanity scenes S2, S2Q and S3, refs/sanity.mjs; PGC3D_1ms, the regress
    /// scene PGC3D for 1 ms, refs/regress.mjs).
    fn request_of(name: &str) -> Request {
        let refs = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs");
        let path = [refs.join("scenes"), refs.join("requests")].map(|d| d.join(format!("{name}.json"))).into_iter().find(|p| p.exists()).expect("the scene's request");
        Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap()
    }

    /// What the model's part of a step costs at the second order and at the fourth (measured; `SCENE`, default R4_LIN_1):
    /// per cluster of the closed model whose fold has a fourth order, from the two models' tables, one middle at a
    /// thermal point, its path, sixteen generators, eight forces, three tones' forms and the state's scattering -- a
    /// product step's calls --, in µs, with the middle's and the scattering's share; and the tables' reals a node.
    #[test]
    #[ignore]
    fn the_fourth_order_s_cost_a_step() {
        use crate::model::Run;
        use crate::step::Middle as _;
        let name = std::env::var("SCENE").unwrap_or("R4_LIN_1".into());
        let (closed, centre) = closed(&name);
        let grounds: Vec<usize> = (0..closed.clusters.len()).filter(|&c| closed.fold_frequency(c).is_some()).map(|c| closed.clusters[c].states[0]).collect();
        let fourth = closed.rekept(&[closed.kept.clone(), grounds.iter().map(|&ground| Unit::Second { ground }).collect()].concat()).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == grounds[0] { 1.0 } else { 0.0 });
        for (order, model) in [("second", &closed), ("fourth", &fourth)] {
            let trap = model.trap(&reference, &model.prepared()).unwrap();
            let reach = model.reach(&trap, &reference, 30.0, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
            let started = std::time::Instant::now();
            let tables = model.tabulate(&reach, 1 << 20);
            let built = started.elapsed();
            let run = Run::tabulated(model, &tables, reference);
            for &ground in &grounds {
                let cluster = model.membership[ground];
                let n = model.clusters[cluster].states.len();
                let psi: Vec<C64> = (0..n).map(|x| C64::cis(0.7 * x as f64) * (1.0 / n as f64).sqrt()).collect();
                let rho: Vec<C64> = (0..n * n).map(|i| psi[i / n] * psi[i % n].conj()).collect();
                let (rounds, mut h, mut sink) = (20_000, Mat::zeros(n, n), 0.0);
                let mut times = [std::time::Duration::ZERO; 7];
                let mut counts = [0.0; 4];
                for round in 0..rounds {
                    let r = [centre[0] + 0.01, centre[1], centre[2] + 0.39 * (round % 97) as f64 / 97.0];
                    let at = Point { r, kappa: [0.3, -0.2, 2.5], t: 0.0 };
                    let from = std::time::Instant::now();
                    let middle = run.middle(cluster, r, 0.0);
                    times[0] += from.elapsed();
                    let path = middle.path(at.kappa, at.kappa.map(|k| k * model.hbar_over_mass));
                    let part = std::time::Instant::now();
                    for node in 0..16 {
                        middle.generator(&path, 0.004 * (node as f64 - 7.5), &mut h);
                        sink += h.data[0].re;
                    }
                    times[3] += part.elapsed();
                    let part = std::time::Instant::now();
                    for node in 0..8 {
                        sink += middle.force(&path, 0.008 * (node as f64 - 3.5), &rho)[2];
                    }
                    times[4] += part.elapsed();
                    middle.tones(&path, &rho, &mut |_, f| sink += f[2]);
                    times[1] += from.elapsed();
                    let work = middle.work();
                    counts = [middle.work, 16.0 * work[0], 8.0 * work[1], 0.0];
                    drop(path);
                    drop(middle);
                    let scattering = std::time::Instant::now();
                    let plan = model.plan(cluster, &model.running());
                    let (scattered, counted) = model.scattering_of(&plan, tables.of(cluster, &model.running()), &at, &psi);
                    sink += scattered.force[2];
                    counts[3] = counted;
                    times[2] += scattering.elapsed();
                    let energy = std::time::Instant::now();
                    sink += run.model.energy_with(cluster, &at, &psi, Some(&tables));
                    times[5] += energy.elapsed();
                }
                let each = |t: std::time::Duration| t.as_secs_f64() * 1e6 / rounds as f64;
                let table = tables.of(cluster, &model.running()).map_or(0, |t| t.table.width());
                let ns = |time: std::time::Duration, count: f64| each(time) * 1e3 / count;
                eprintln!(
                    "{name} {order}: cluster of ground state {ground}: ns a counted multiply-add -- middle {:.2} ({:.0}), generators {:.2} ({:.0}), forces {:.2} ({:.0}), scattering {:.2} ({:.0}); an energy {:.2} µs",
                    ns(times[0], counts[0]), counts[0], ns(times[3], counts[1]), counts[1], ns(times[4], counts[2]), counts[2], ns(times[2], counts[3]), counts[3], each(times[5])
                );
                eprintln!("{name} {order}: cluster of ground state {ground} ({n} states): the model's part of a step {:.2} µs (its middle {:.2}, its scattering {:.2}); table {table} reals a node, {} tables in {built:.2?} [{sink:.3e}]", each(times[1]) + each(times[2]), each(times[0]), each(times[2]), tables.tables.len());
            }
        }
    }
}
