//! The price of a representation as its sizes' formula (unify2 §3.3; BRIEF_P_PRICE): what every counter of the prototype
//! adds (complex multiply-adds, as proto.rs counts them; the run's real ones are four times these, `decide::COMPLEX`), read
//! off the sizes of the representation and of its evaluation -- no forward made, no count kept. Each function below is
//! one of proto.rs's counted functions, its terms in the order they are counted there, so that the two read alike line by
//! line; `the_price_formula_is_the_counted_work` holds them equal item by item on the scenes.
//!
//! # The sizes (`Sizes`)
//! A representation is its key (the set's P places, n) and its evaluation: per near block (`NearSize`) its kept rows by
//! Schur part s_P with each part's triangle's reach Σ_{i∈P} (end_i − i) (s_P(s_P + 1)/2 where its Schur form is dense: a
//! candidate not yet decomposed is priced so, `Span::dense`) and the sweeps the evaluation gives the part (proto.rs
//! `Evaluation`: one is the first sweep's order), its lanes L, beam splits, source vectors, terms (and of them those
//! whose coefficient moved off the decomposition's), explicit states, exact copies n_d with their coupling vectors,
//! centred copies' rows n_c by copy vector (always in the compression, unify2 §2.2), its levels' rows and which are left
//! out; per far block (`FarSize`) its source vectors v, rows, splits, terms, and whether the evaluation leaves it out;
//! the light's layout (`Layout`: the frequencies' beams × rank, the pairs, Ĥ_PP's form entries, the drives' entries, each
//! tone's entries, the fold's own forms and drives); the series' sweeps (its parts' most); the coherence groups and the
//! landing sets.
//!
//! # What is not a size (`Route`)
//! Two counts are iterations to the rounding, set by a norm, not by a size: S⁻¹'s Newton–Schulz passes (by ‖X‖_F,
//! `Route::newton_schulz`: the scalar iteration on the largest eigenvalue it bounds) and the Lyapunov map's passes on each
//! axis (by its contraction ‖E‖ ≤ (s − 1)/(s + 1), s = √(1 + ‖X‖_F): `Route::lyapunov`). They are the formula's inputs,
//! predicted from the metric's norm; the test lists what the forward took. The left series sweeps each row as the right's
//! (proto.rs `left_within`, unify2 U1): a size, the right's sweeps. A Schur form's QR iterations are iterations likewise
//! (`schur_count`: two an eigenvalue; a block's kept rows' and each of its levels', `NearBasis::lfact`). The integrator's picture (its decomposition's route, following, hops) is the integrator's (step.rs
//! `Price::made`, `Price::stops`), given to `Sizes::step` as it is: its count::* terms are step.rs's own formula
//! (`Price::stepped`).
//!
//! # A step, a cell, the once-only
//! A step (`Sizes::step`, real multiply-adds): the middle's held forward (`forward`), the integrator's stretch with the
//! model's counts at its nodes (`costs`: the outer nodes, the tangent and the force of the set's bars) and the light
//! scattered at the middle (`photons`), and per expected jump its place, its drawn photons and its landing (`jump`). A
//! cell (`cell`): D/τ steps and what is made once in it -- a block decomposed again (`decomposition`, `distant`, with the
//! light at the point once, `site`), a switch to another representation (`switch`: its decision's forward on
//! decompositions made at the point, the state taken onto it, its held forms).
//!
//! # How the decision takes it (U3; unify2 §3.4)
//! Every option of the convex-hull walk is a pair (Δρ², Δ the cell's price): the price of the representation the option
//! leaves, `cell(...)` on its `Sizes`, less the standing one's -- the standing `Sizes` read off the decision's forward
//! (`Workspace::sizes`), the option's made from it by its sizes alone: a part's sweep more or less (`Span::sweeps`, and
//! `sweeps` their most); a level solved or left out (its rows into or out of `parts` -- the parts it couples to merged,
//! priced dense -- and `omitted`, with `decomposition` once); a far block left out or not (`omitted`); a refresh
//! (`decomposition` once, the sizes the same); a node risen or fallen (the target cluster's layout and sizes, `switch`
//! once). No forward is made for a price, and none is kept. A held forward reads nothing (unify2 U1: the readings are a
//! decision's, `Fold::accounts`), so the formula of a forward holding the evaluation has no reading in it.
use super::*;

/// A Schur part of a near block's kept rows: its rows s_P, its triangle's reach Σ_{i∈P} (end_i − i) (the rows' entries
/// to the right of the diagonal and the diagonal: K̂Z's and the triangular solve's), and the sweeps the evaluation gives
/// it (proto.rs `Evaluation`, `Near::upto`: its series part's; one is the first sweep's order).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Span {
    pub(crate) rows: usize,
    pub(crate) reach: usize,
    pub(crate) sweeps: usize,
}

impl Span {
}

/// A near block's sizes (proto.rs `Near` on its standing decomposition).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NearSize {
    pub(crate) parts: Vec<Span>,
    pub(crate) lanes: usize,
    pub(crate) splits: usize,
    pub(crate) vectors: usize,
    pub(crate) terms: usize,
    /// Of the terms, those whose coefficient at the point differs from the decomposition's (the diagonal's first order).
    pub(crate) moved: usize,
    pub(crate) explicit: usize,
    pub(crate) nd: usize,
    pub(crate) exact: usize,
    /// The centred copies' rows n_c (always in the compression where the block has a response, unify2 §2.2), and per
    /// copy vector its group's rows.
    pub(crate) nc: usize,
    pub(crate) copies: Vec<usize>,
    /// Per level its rows, whether the point leaves it out; every excited row (`all`).
    pub(crate) levels: Vec<usize>,
    pub(crate) omitted: Vec<bool>,
    pub(crate) all: usize,
    /// Per level its K_qq − o's Schur parts' rows (`NearBasis::lfact`; none where it is singular) and its first row in the
    /// stacked reference coupling (`NearBasis::rat`): what a decision reads of a level it leaves out (`Sizes::reading`).
    pub(crate) level_parts: Vec<Vec<usize>>,
    pub(crate) rat: Vec<usize>,
    /// Its coherence group (fold.rs `Fold::groups`).
    pub(crate) group: usize,
}

impl NearSize {
    /// Its kept rows s.
    pub(crate) fn s(&self) -> usize {
        self.parts.iter().map(|p| p.rows).sum()
    }

    /// Nothing kept and no explicit state (proto.rs `Near::idle`).
    pub(crate) fn idle(&self) -> bool {
        self.s() == 0 && self.explicit == 0
    }

    /// Its triangle's reach over every part.
    fn reach(&self) -> usize {
        self.parts.iter().map(|p| p.reach).sum()
    }

    /// Its triangle's reach over the parts a right sweep k solves (those whose sweeps go beyond k, `Near::upto`).
    fn reach_at(&self, k: usize) -> usize {
        self.parts.iter().filter(|p| k < p.sweeps).map(|p| p.reach).sum()
    }

    /// Its centred copies' rows in the compression (proto.rs `Near::copied`).
    fn copied(&self) -> bool {
        self.nc > 0 && !self.idle()
    }

    /// Σ over the copy vectors of their groups' rows (`copies_of`: rows · s · n each).
    fn copy_rows(&self) -> usize {
        self.copies.iter().sum()
    }
}

/// A far block's sizes (proto.rs `Far`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FarSize {
    pub(crate) v: usize,
    pub(crate) m: usize,
    pub(crate) splits: usize,
    pub(crate) terms: usize,
    pub(crate) omitted: bool,
    pub(crate) group: usize,
}

/// The light's layout on a set's P places (proto.rs `Workspace`): Σ_f beams_f · rank_f, the pairs, Ĥ_PP's form entries
/// over the pairs, the drives' entries at e_k and at e_k*, per tone its forms' entries; and the fold's own (`Fold::site`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Layout {
    pub(crate) beams: usize,
    pub(crate) pairs: usize,
    pub(crate) pp: usize,
    pub(crate) direct: usize,
    pub(crate) conj: usize,
    pub(crate) tones: Vec<usize>,
    /// The fold's own light at a point (fold.rs `Fold::site`): its forms' entries, its components' beams.
    pub(crate) entries: usize,
    pub(crate) drives: usize,
}

/// A representation's sizes (module documentation, # The sizes).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sizes {
    pub(crate) n: usize,
    /// The set's ground places.
    pub(crate) ground: usize,
    pub(crate) layout: Layout,
    pub(crate) near: Vec<NearSize>,
    pub(crate) far: Vec<FarSize>,
    pub(crate) sweeps: usize,
    /// The landing sets' sizes (the ground states' partition, the photons' classes).
    pub(crate) sets: Vec<usize>,
}

/// The counts that are iterations to the rounding (module documentation, # What is not a size): S⁻¹'s Newton–Schulz
/// passes and whether it starts from the series (‖X‖_F ≤ 1), the Lyapunov map's passes per axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Route {
    pub(crate) started: bool,
    pub(crate) normal: usize,
    pub(crate) lyapunov: [usize; 3],
}

/// A bar of a contraction (proto.rs `Bar`, its matrix aside): its channel, position, tone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cotangent {
    pub(crate) channel: usize,
    pub(crate) position: usize,
    pub(crate) tone: Option<usize>,
}

/// The photons asked (proto.rs `Moments`): every class's moments at the middle, or a jump's at its place -- its class's
/// group and landing set's size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Asked {
    Every,
    Drawn { group: usize, set: usize },
}

impl Route {
    /// The route predicted from the metric less one's Frobenius norm on n places (module documentation, # What is not a
    /// size).
    pub(crate) fn predicted(x: f64, n: usize) -> Route {
        let (started, normal) = Route::newton_schulz(x, n);
        let each = Route::lyapunov(x, n);
        Route { started, normal, lyapunov: [each; 3] }
    }

    /// `inverse_root`'s start and passes on the largest eigenvalue ‖X‖_F bounds: from the series' second order where ‖X‖_F
    /// ≤ 1 (its product counted), else from the scalar (1 + ‖X‖_F)^{−1/2}; per pass y ← y(3 − (1 + x)y²)/2, until 1.5 m² ≤
    /// γ_{4n} ‖Y‖_F (m the pass's move), ‖Y‖_F read as √n y.
    pub(crate) fn newton_schulz(x: f64, n: usize) -> (bool, usize) {
        let started = x <= 1.0;
        let mut y = if started { 1.0 - 0.5 * x + 0.375 * x * x } else { 1.0 / (1.0 + x).sqrt() };
        let floor = crate::units::rounding(4 * n);
        let (mut passes, mut last) = (0, f64::INFINITY);
        loop {
            let next = y * (3.0 - (1.0 + x) * y * y) / 2.0;
            let moved = (next - y).abs() * (n as f64).sqrt();
            passes += 1;
            y = next;
            if 1.5 * moved * moved <= floor * (n as f64).sqrt() * y.abs() || !(moved < last) {
                return (started, passes);
            }
            last = moved;
        }
    }

    /// `lyapunov`'s passes: its fixed point contracts by ‖E‖ ≤ (s − 1)/(s + 1), s = √(1 + ‖X‖_F), its moves from the first
    /// (at most ‖E‖ the size) until within γ_{4n} of the size; one where X is zero (E = 0: the first pass moves nothing).
    pub(crate) fn lyapunov(x: f64, n: usize) -> usize {
        let s = (1.0 + x).sqrt();
        let q = (s - 1.0) / (s + 1.0);
        let floor = crate::units::rounding(4 * n);
        if !(q > 0.0) {
            return 1;
        }
        (floor.ln() / q.ln()).ceil().max(1.0) as usize
    }
}

impl Layout {
    /// The light at a place (proto.rs `light_at`): ε_k and its gradients, per frequency its beams × rank, four times.
    pub(crate) fn light(&self) -> f64 {
        (4 * self.beams) as f64
    }

    /// Ĥ_PP at a place (`generator_at`): the pairs' coefficients, the forms' entries, the drives', the tones', the pairs'
    /// gradients.
    pub(crate) fn generator(&self) -> f64 {
        (self.pairs + self.pp + 2 * self.direct + self.tones.iter().sum::<usize>() + 6 * self.pairs) as f64
    }

    /// The place (`proto_place`).
    pub(crate) fn place(&self) -> f64 {
        self.light() + self.generator()
    }
}

impl Sizes {
    fn active(&self) -> impl Iterator<Item = &NearSize> {
        self.near.iter().filter(|b| !b.idle())
    }

    fn lit(&self) -> impl Iterator<Item = &FarSize> {
        self.far.iter().filter(|f| !f.omitted)
    }

    /// The exact copies' Woodbury on one split (`copies_factor` and `copies_apply`, per column).
    fn woodbury(b: &NearSize, n: usize) -> f64 {
        let (s, nd) = (b.s(), b.nd);
        n as f64 * ((nd * (s * s / 2 + s) + nd * nd * s) as f64 + (nd * nd * nd) as f64 * 4.0 / 3.0) + (n * (2 * nd * s + nd * nd)) as f64
    }

    /// One sweep k of a near block in an orientation (proto.rs `sweep_block`).
    fn sweep_block(b: &NearSize, n: usize, k: usize, side: Side) -> f64 {
        let (s, l) = (b.s(), b.lanes);
        let ln = l * n;
        let mut w = 0.0;
        if k > 0 {
            w += (s * ln * n) as f64;
        }
        w += (s * ln) as f64;
        let reach = if side == Side::Right { b.reach_at(k) } else { b.reach() };
        w += (reach * ln) as f64;
        if b.nd > 0 {
            w += l as f64 * Sizes::woodbury(b, n);
        }
        w + (n * s * n + l * s * n) as f64
    }

    /// One sweep k over the blocks (`sweep_all`).
    pub(crate) fn sweep(&self, k: usize, side: Side) -> f64 {
        self.active().map(|b| Sizes::sweep_block(b, self.n, k, side)).sum()
    }

    /// The left response at the last forward (`left_within`): its inputs, its sweeps (the right's), N = 1 + T_L†T and its
    /// factors.
    pub(crate) fn left(&self) -> f64 {
        let n = self.n;
        let nn = n * n;
        let mut w: f64 = self.active().map(|b| ((b.nd * (b.s() + n)) * 2 + b.nd * b.nd + nn) as f64 / 4.0).sum();
        w += (0..self.sweeps).map(|k| self.sweep(k, Side::Left)).sum::<f64>();
        for b in self.active() {
            w += ((b.s() + b.nd) * nn) as f64;
            if b.copied() {
                w += (b.copy_rows() * b.s() * n + b.nc * nn) as f64;
            }
        }
        w += self.lit().map(|f| (f.v * f.v * nn) as f64).sum::<f64>();
        w + (n * nn) as f64 / 3.0
    }

    /// The exact copies' compression (`copies_compress`).
    fn copies_compress(b: &NearSize, n: usize) -> f64 {
        let (s, nd) = (b.s(), b.nd);
        (2 * s * nd * n + nd * nd * n + (s + 2 * nd) * n * n) as f64
    }

    /// The step's two outer nodes (`prototype_nodes`).
    pub(crate) fn nodes(&self) -> f64 {
        let n = self.n;
        let nn = n * n;
        let mut one = self.layout.place();
        one += self.lit().map(|f| (2 * f.v * f.v * nn + nn) as f64).sum::<f64>();
        for b in self.active() {
            let (s, nd) = (b.s(), b.nd);
            one += ((b.lanes + b.vectors) * s * n) as f64;
            if b.explicit > 0 {
                one += (b.terms * (2 * s * n + nn)) as f64;
                if nd > 0 {
                    one += (b.exact * n * nd) as f64;
                }
            }
            one += (b.reach() * n + 2 * s * nn) as f64;
            if nd > 0 {
                one += Sizes::copies_compress(b, n);
                if b.explicit > 0 {
                    one += (2 * n * nd * n) as f64;
                }
            }
            if b.copied() {
                one += (b.copy_rows() * s * n + b.nc * nn) as f64;
            }
        }
        one += (2 * n * nn * (1 + self.layout.tones.len())) as f64;
        2.0 * one
    }

    /// The step's tangent (`prototype_tangent`), by its phases (proto.rs `TANGENT`).
    pub(crate) fn tangent(&self, route: &Route) -> [f64; 6] {
        let (n, layout) = (self.n, &self.layout);
        let (nn, n3) = (n * n, 3 * n);
        let mut w = [0.0; 6];
        w[0] = (3 * layout.pp + 6 * layout.direct) as f64 + self.lit().map(|f| (6 * f.v * f.v * nn) as f64).sum::<f64>();
        for b in self.active() {
            let s = b.s();
            w[0] += (3 * (b.terms * s * s + b.vectors * s * n)) as f64;
            if b.explicit > 0 {
                w[0] += (3 * b.terms * (2 * s * n + nn)) as f64;
            }
        }
        for k in 0..self.sweeps {
            for b in self.active() {
                let (s, nd, l) = (b.s(), b.nd, b.lanes);
                let (ln, m) = (l * n, 3 * l * n);
                if k == 0 {
                    w[1] += (s * m) as f64;
                }
                w[1] += (s * m) as f64;
                w[1] += b.parts.iter().map(|p| (3 * p.rows * p.rows * ln) as f64).sum::<f64>();
                if k > 0 {
                    w[2] += (2 * s * m * n) as f64;
                }
                w[3] += (b.reach_at(k) * m) as f64;
                if nd > 0 {
                    w[3] += (l * 3) as f64 * ((n * (2 * nd * s + nd * nd)) as f64 + (nd * n * (s + nd + 2 * n)) as f64);
                }
                w[4] += (s * m) as f64 / 4.0;
            }
            for b in self.active() {
                w[4] += (2 * b.s() * n * n3) as f64;
                if b.explicit > 0 && b.nd > 0 {
                    w[4] += (6 * n * b.nd * n) as f64;
                }
            }
        }
        for b in self.active() {
            w[5] += (2 * b.s() * n * n3) as f64;
            if b.copied() {
                w[5] += (3 * 2 * b.copy_rows() * b.s() * n) as f64;
            }
        }
        w[5] += route.lyapunov.iter().map(|&p| ((1 + p) * n * nn) as f64).sum::<f64>();
        w
    }

    /// The forms of the step's derivative on the bars (`prototype_force`).
    pub(crate) fn force(&self, bars: &[Cotangent]) -> f64 {
        let (n, layout) = (self.n, &self.layout);
        let (nn, n3) = (n * n, n * n * n);
        let mut w = 0.0;
        for b in bars {
            if let Some(m) = b.tone {
                w += (layout.tones[m] + 7 * n3 + 3 * nn) as f64;
            }
        }
        let statics: Vec<&Cotangent> = bars.iter().filter(|b| b.tone.is_none()).collect();
        let channels = bars.iter().map(|b| b.channel + 1).max().unwrap_or(0);
        let count = |c: usize| statics.iter().filter(|b| b.channel == c).count();
        for c in 0..channels {
            let k = count(c);
            if k > 0 {
                w += (3 * n3 * k + 2 * n3 + 3 * nn) as f64;
            }
        }
        let total: usize = self.active().map(|b| b.s()).sum();
        for bar in &statics {
            w += (layout.pp + 2 * layout.direct) as f64 + self.lit().map(|f| (f.v * f.v * nn) as f64).sum::<f64>();
            w += (2 * total * nn) as f64;
            for b in self.active() {
                let (s, nd) = (b.s(), b.nd);
                w += (s * n) as f64;
                if b.explicit > 0 {
                    w += (b.terms * (2 * s * n + nn)) as f64;
                }
                if nd > 0 {
                    w += (2 * (s + nd) * nn + 6 * s * nd * n + 3 * nd * nd * n) as f64;
                    if b.explicit > 0 {
                        w += (2 * nd * nn) as f64;
                    }
                }
                if bar.position == 0 {
                    w += (3 * (s + nd) * n) as f64;
                } else {
                    if b.copied() {
                        w += (2 * b.copy_rows() * s * n) as f64;
                    }
                    w += (3 * b.lanes * (s + nd) * n) as f64;
                }
            }
        }
        for b in self.active() {
            let s = b.s();
            for c in (0..channels).filter(|&c| count(c) > 0) {
                w += (s * s * count(c) * n + 3 * s * s) as f64;
            }
        }
        w
    }

    /// The coherence groups the photons' first pass makes slots of: the active near blocks' and the lit far blocks', each
    /// group once.
    fn slots(&self) -> Vec<usize> {
        let mut out: Vec<usize> = vec![];
        for g in self.active().map(|b| b.group).chain(self.lit().map(|f| f.group)) {
            if !out.contains(&g) {
                out.push(g);
            }
        }
        out
    }

    /// The absorbed light's moments (`prototype_photons`): every class's at the middle, or a jump's drawn class's at its
    /// place (`placed`: the responses' tangent turned there).
    pub(crate) fn photons(&self, asked: Asked, placed: bool) -> f64 {
        let n = self.n;
        let nn = n * n;
        let g3 = 3 * GROUND;
        let mut w = nn as f64;
        let slots = self.slots().len();
        let width = if asked == Asked::Every { 4 } else { 1 };
        if asked == Asked::Every {
            w += (6 * nn) as f64;
        }
        for b in self.active() {
            let s = b.s();
            w += (width * s * n) as f64;
            if asked == Asked::Every {
                w += (3 * s * n) as f64;
            }
            w += (g3 * s * width) as f64;
            if b.explicit > 0 {
                w += (g3 * n * width) as f64;
            }
            w += (g3 * (width + 1)) as f64;
        }
        for f in self.lit() {
            w += (3 * f.v * GROUND * n * width + g3 * (width + 1)) as f64;
        }
        w += (slots * g3) as f64;
        match asked {
            Asked::Every => w += (slots * self.sets.iter().map(|&set| 13 * 3 * set).sum::<usize>()) as f64,
            Asked::Drawn { group, set } => {
                // The second pass: ∂u of the drawn class's group's blocks on its landing set's rows.
                w += (6 * nn) as f64;
                let rows = 3 * set;
                for b in self.active().filter(|b| b.group == group) {
                    let s = b.s();
                    if placed {
                        w += (3 * b.lanes * s * n) as f64;
                    }
                    w += (3 * s * n + 3 * s * n + rows * s * 3) as f64;
                    if b.explicit > 0 {
                        w += (rows * n * 3) as f64;
                    }
                    w += (rows * 4) as f64;
                }
                for f in self.lit().filter(|f| f.group == group) {
                    w += (f.v * rows * n * 3 + rows * 4) as f64;
                }
                w += (13 * rows) as f64;
            }
        }
        w
    }

    /// A jump's place (`prototype_place`): the light there and the responses turned to it.
    pub(crate) fn place(&self) -> f64 {
        self.layout.light() + self.active().map(|b| (b.lanes * b.s() * self.n) as f64).sum::<f64>()
    }

    /// The model's counts at a node as the run's set gives them to the integrator (run.rs `Set::new`, `step::Middle::work`):
    /// a node's generator (the two outer nodes' a third each with the middle's), a node's forms ((tangent + force)/3) and
    /// the middle's held forms (tangent + force) -- the force of the three positions' static bars and every tone's three.
    pub(crate) fn costs(&self, route: &Route) -> [f64; 3] {
        let mut bars: Vec<Cotangent> = (0..3).map(|position| Cotangent { channel: 0, position, tone: None }).collect();
        bars.extend((0..self.layout.tones.len()).flat_map(|m| (0..3).map(move |position| Cotangent { channel: 0, position, tone: Some(m) })));
        let forms = self.tangent(route).iter().sum::<f64>() + self.force(&bars);
        [self.nodes() / 3.0, forms / 3.0, forms]
    }

    /// An integrator's candidate tone's counts (run.rs `Candidate::work`): its envelopes -- its members' tones at the three
    /// nodes, each cross decay's product there, `cross` its outer products of n² and its sum into the envelope over the
    /// candidate's cross decays (run.rs `Middle::crossed`, `Fold::decay_between`) -- and its forms (its members' bars at
    /// the three nodes, of tone `tone`).
    pub(crate) fn candidate(&self, members: usize, cross: usize, tone: usize) -> [f64; 2] {
        let n = self.n;
        let one: Vec<Cotangent> = (0..members).flat_map(|_| (0..3).map(|position| Cotangent { channel: 0, position, tone: Some(tone) })).collect();
        let forms = if one.is_empty() { 0.0 } else { self.force(&one) };
        [(3 * one.len() * n * n + 3 * cross * n * n) as f64, forms]
    }


}

impl Workspace {
    /// The route predicted for the last forward's metric (`Route::predicted`).
    pub(crate) fn route(&self) -> Route {
        Route::predicted(self.step.metric.data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(), self.n)
    }

    /// The sizes of the representation the last forward stood on and of its evaluation (module documentation, # The
    /// sizes): its decompositions' parts in `cache`, each part's sweeps (`Near::upto`, within the forward's), the far
    /// blocks it left out; `sets` the landing sets (the ground states' partition).
    pub(crate) fn sizes(&self, fold: &Fold, cache: &Cache, sets: &[Vec<usize>]) -> Sizes {
        let layout = Layout {
            beams: fold.light.frequencies.iter().map(|f| f.beams.len() * f.rank()).sum(),
            pairs: self.pairs.len(),
            pp: self.pp.iter().map(Vec::len).sum(),
            direct: self.lin.iter().map(|x| x.2.len()).sum(),
            conj: self.lin.iter().map(|x| x.3.len()).sum(),
            tones: self.rotating.iter().map(|r| r.pp.iter().map(|x| x.1.len()).sum()).collect(),
            entries: fold.entries.iter().flat_map(|w| w.iter().map(|x| x.2.len())).sum(),
            drives: fold.components.iter().map(|c| c.beams.len()).sum(),
        };
        let near = self
            .near
            .iter()
            .enumerate()
            .map(|(x, nb)| {
                let nl = nb.levels.len();
                let basis = cache.near.get(x).and_then(Option::as_deref);
                let parts = basis.map_or(vec![], |b| b.f.parts.iter().map(|(idx, at, _)| Span { rows: idx.len(), reach: (*at..at + idx.len()).map(|i| b.f.ends[i] - i).sum(), sweeps: nb.upto[*at].min(self.step.sweeps) }).collect());
                NearSize {
                    parts,
                    lanes: nb.lanes,
                    splits: nb.splits.len(),
                    vectors: nb.vectors.len(),
                    terms: nb.terms.len(),
                    moved: nb.delta.iter().filter(|d| **d != C64::ZERO).count(),
                    explicit: nb.xs.len(),
                    nd: nb.nd,
                    exact: nb.exact.len(),
                    nc: nb.nc,
                    copies: nb.copies.iter().map(|&(_, _, gi)| nb.groups[gi].1.len()).collect(),
                    levels: nb.levels.iter().map(Vec::len).collect(),
                    omitted: basis.map_or(vec![true; nl], |b| (0..nl).map(|q| !b.kept.contains(&q)).collect()),
                    all: nb.all,
                    level_parts: basis.map_or(vec![vec![]; nl], |b| b.lfact.iter().map(|f| f.as_ref().map_or(vec![], |f| f.parts.iter().map(|p| p.0.len()).collect())).collect()),
                    rat: basis.map_or(vec![0; nl], |b| b.rat.clone()),
                    group: 0,
                }
            })
            .collect();
        let far = self.far.iter().map(|fb| FarSize { v: fb.vectors.len(), m: fb.m, splits: fb.splits.len(), terms: fb.terms.len(), omitted: fb.omitted, group: 0 }).collect();
        Sizes { n: self.n, ground: fold.ground.len(), layout, near, far, sweeps: self.step.sweeps, sets: sets.iter().map(Vec::len).collect() }
    }
}
