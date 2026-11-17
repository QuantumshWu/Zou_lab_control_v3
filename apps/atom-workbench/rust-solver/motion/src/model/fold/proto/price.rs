//! The price of a representation as its sizes' formula (unify2 §3.3; BRIEF_P_PRICE): what every counter of the prototype
//! adds (complex multiply-adds, as proto.rs counts them; the run's real ones are four times these, `decide::COMPLEX`), read
//! off the sizes of the representation and of its evaluation -- no forward made, no count kept. Each function below is
//! one of proto.rs's counted functions, its terms in the order they are counted there, so that the two read alike line by
//! line; `the_price_formula_is_the_counted_work` holds them equal item by item on the scenes.
//!
//! # The sizes (`Sizes`)
//! A representation is its key (the set's P places, n) and its evaluation: per near block (`NearSize`) its kept rows by
//! Schur part s_P with each part's triangle's reach Σ_{i∈P} (end_i − i) (s_P(s_P + 1)/2 where its Schur form is dense: a
//! candidate not yet decomposed is priced so, `Span::dense`) and whether the evaluation holds the part at the first sweep's
//! order, its lanes L, beam splits, source vectors, terms (and of them those whose coefficient moved off the
//! decomposition's), explicit states, exact copies n_d with their coupling vectors, centred copies' rows n_c by copy
//! vector (and whether they are in the compression), its levels' rows and which are left out; per far block (`FarSize`)
//! its source vectors v, rows, splits, terms, and whether the point leaves it out; the light's layout (`Layout`: the
//! frequencies' beams × rank, the pairs, Ĥ_PP's form entries, the drives' entries, each tone's entries, the fold's own
//! forms and drives); the series' sweeps; the coherence groups and the landing sets.
//!
//! # What is not a size (`Route`)
//! Three counts are iterations to the rounding, set by a norm, not by a size: S⁻¹'s Newton–Schulz passes (by ‖X‖_F,
//! `Route::newton_schulz`: the scalar iteration on the largest eigenvalue it bounds), the Lyapunov map's passes on each
//! axis (by its contraction ‖E‖ ≤ (s − 1)/(s + 1), s = √(1 + ‖X‖_F): `Route::lyapunov`) and the left series' sweeps (the
//! right's, at least two: with no decay the two are one iteration). They are the formula's inputs, predicted from the
//! metric's norm; the test lists what the forward took. A Schur form's QR iterations likewise (`schur_count`: two an
//! eigenvalue). The integrator's picture (its decomposition's route, following, hops) is the integrator's (step.rs
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
//! (`Workspace::sizes`), the option's made from it by its sizes alone: a sweep more or less (`sweeps`); a part held or
//! solved (`Span::frozen`); a level solved or left out (its rows into or out of `parts` -- the parts it couples to merged,
//! priced dense -- and `omitted`, with `decomposition` once); centred copies' rows in or out (`copied`); a far block left
//! out or not (`omitted`); a refresh (`decomposition` once, the sizes the same); a node risen or fallen (the target
//! cluster's layout and sizes, `switch` once). No forward is made for a price, and none is kept.
#![cfg_attr(not(test), allow(dead_code))] // Until the decision takes it (U3).
use super::*;
use crate::decide::COMPLEX;

/// A Schur part of a near block's kept rows: its rows s_P, its triangle's reach Σ_{i∈P} (end_i − i) (the rows' entries
/// to the right of the diagonal and the diagonal: K̂Z's and the triangular solve's), and whether the evaluation holds it
/// at the first sweep's order (proto.rs `freeze`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Span {
    pub(crate) rows: usize,
    pub(crate) reach: usize,
    pub(crate) frozen: bool,
}

impl Span {
    /// A part not yet decomposed, priced dense (its triangle full: s_P(s_P + 1)/2), not frozen.
    pub(crate) fn dense(rows: usize) -> Span {
        Span { rows, reach: rows * (rows + 1) / 2, frozen: false }
    }
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
    /// The centred copies' rows n_c, and per copy vector its group's rows; whether they are in the compression.
    pub(crate) nc: usize,
    pub(crate) copies: Vec<usize>,
    pub(crate) copied: bool,
    /// Per level its rows, whether the point leaves it out; every excited row (`all`).
    pub(crate) levels: Vec<usize>,
    pub(crate) omitted: Vec<bool>,
    pub(crate) all: usize,
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

    /// Its parts' count as the counters take it (at least one).
    fn parts_n(&self) -> usize {
        self.parts.len().max(1)
    }

    /// Its triangle's reach over every part.
    fn reach(&self) -> usize {
        self.parts.iter().map(|p| p.reach).sum()
    }

    /// Its triangle's reach over the parts a right sweep k solves (every one at the first; then those not frozen).
    fn reach_at(&self, k: usize) -> usize {
        self.parts.iter().filter(|p| k == 0 || !p.frozen).map(|p| p.reach).sum()
    }

    /// Σ over the copy vectors of their groups' rows (`copies_of`: rows · s · n each).
    fn copy_rows(&self) -> usize {
        self.copies.iter().sum()
    }

    /// The omitted levels: their count and their rows.
    fn left_out(&self) -> (usize, usize) {
        self.levels.iter().zip(&self.omitted).filter(|x| *x.1).fold((0, 0), |(q, r), (rows, _)| (q + 1, r + rows))
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
    /// The set's ground places (the ring's columns).
    pub(crate) ground: usize,
    pub(crate) layout: Layout,
    pub(crate) near: Vec<NearSize>,
    pub(crate) far: Vec<FarSize>,
    pub(crate) sweeps: usize,
    /// The landing sets' sizes (the ground states' partition, the photons' classes).
    pub(crate) sets: Vec<usize>,
}

/// The counts that are iterations to the rounding (module documentation, # What is not a size): S⁻¹'s Newton–Schulz
/// passes and whether it starts from the series (‖X‖_F ≤ 1), the Lyapunov map's passes per axis, the left series' sweeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Route {
    pub(crate) started: bool,
    pub(crate) normal: usize,
    pub(crate) lyapunov: [usize; 3],
    pub(crate) left: usize,
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
    /// The route predicted from the metric less one's Frobenius norm on n places and the right's sweeps (module
    /// documentation, # What is not a size).
    pub(crate) fn predicted(x: f64, n: usize, sweeps: usize) -> Route {
        let (started, normal) = Route::newton_schulz(x, n);
        let each = Route::lyapunov(x, n);
        Route { started, normal, lyapunov: [each; 3], left: sweeps.max(2) }
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

    /// A block's omitted levels bounded (`bound_levels`, where it bounds them).
    fn bound_levels(b: &NearSize, n: usize) -> f64 {
        let (nq, stacked) = b.left_out();
        if nq == 0 {
            return 0.0;
        }
        let (nl, terms, s, lanes) = (b.levels.len(), b.terms, b.s(), b.lanes);
        let ln = lanes * n;
        let mut w = (terms * nl * (nl + 2)) as f64;
        w += (s * ln + ln * s / 4 + n * s + n * n / 4) as f64;
        w += (nq * lanes * n * 3) as f64;
        w += ((b.vectors * b.all + stacked * s) * n) as f64;
        w += (nq * n * (nq + n)) as f64 / 4.0;
        w += (nq * n * lanes) as f64;
        w + (nq * n * b.vectors) as f64
    }

    /// The levels and far blocks the point leaves out, read on what the evaluation holds (`omit` holding), and the parts the
    /// first sweep holds (`freeze`).
    pub(crate) fn bounds(&self) -> f64 {
        let n = self.n;
        let levels: usize = self.near.iter().map(|b| b.levels.len()).sum();
        let omitted: f64 = self.near.iter().map(|b| Sizes::bound_levels(b, n)).sum::<f64>() + (n * levels) as f64;
        let frozen: usize = self.near.iter().filter(|b| b.nd == 0).map(|b| b.lanes * b.s() * n).sum();
        omitted + frozen as f64
    }

    /// A forward holding the representation and its evaluation (`forward_within` with `hold`, its decompositions
    /// standing), by phase (proto.rs `PHASES`).
    pub(crate) fn forward(&self, route: &Route) -> [f64; 10] {
        let n = self.n;
        let nn = n * n;
        let mut w = [0.0; 10];
        w[PLACE] = self.layout.place();
        for b in &self.near {
            let s = b.s();
            w[DIAGONAL] += (b.moved * (s + s * b.parts_n())) as f64;
            if b.nd > 0 {
                w[DIAGONAL] += (2 * b.exact * s * b.nd + b.terms * b.nd * b.nd) as f64;
            }
            if b.nc > 0 {
                w[DIAGONAL] += (b.terms * s) as f64 / 4.0;
            }
        }
        for b in self.active() {
            let s = b.s();
            w[SOURCES] += ((b.splits + b.vectors) * s * n) as f64;
            if b.explicit > 0 {
                w[SOURCES] += (b.terms * (2 * s * n + nn)) as f64;
                if b.nd > 0 {
                    w[SOURCES] += (b.exact * n * b.nd) as f64;
                }
            }
        }
        for f in &self.far {
            w[FAR] += (f.v * f.v + 2 * f.v * f.v * nn + nn) as f64;
        }
        w[SERIES] = (0..self.sweeps).map(|k| self.sweep(k, Side::Right)).sum::<f64>() + self.bounds();
        for b in self.active() {
            w[READING] += (b.s() * n * (b.lanes + b.parts_n() + 2)) as f64 / 4.0;
        }
        for f in self.lit() {
            w[READING] += (f.terms + n) as f64 / 4.0;
        }
        w[READING] += self.near.iter().map(|b| b.s() * b.lanes).sum::<usize>() as f64 * self.ground as f64 / 4.0;
        for b in self.active().filter(|b| b.nc > 0) {
            let s = b.s();
            w[COPIES] += (b.copies.len() + s * n / 4) as f64;
            if b.copied {
                w[COPIES] += (b.copy_rows() * s * n + b.nc * nn) as f64;
            }
        }
        for b in self.active() {
            let (s, nd) = (b.s(), b.nd);
            w[COMPRESSION] += (b.reach() * n + 2 * s * nn) as f64;
            if nd > 0 {
                w[COMPRESSION] += (2 * s * nd * n + nd * nd * n + (s + 2 * nd) * nn) as f64;
            }
        }
        w[COMPRESSION] += (self.lit().count() * nn) as f64;
        w[NORMAL] = self.normal(route) + (2 * n * nn * (1 + self.layout.tones.len())) as f64;
        w
    }

    /// S⁻¹ by Newton–Schulz (`inverse_root`) on the route's passes.
    pub(crate) fn normal(&self, route: &Route) -> f64 {
        let n3 = self.n * self.n * self.n;
        (if route.started { n3 } else { 0 } + 3 * n3 * route.normal) as f64
    }

    /// The left response at the last forward (`left_within`): its inputs, its sweeps, N = 1 + T_L†T and its factors.
    pub(crate) fn left(&self, route: &Route) -> f64 {
        let n = self.n;
        let nn = n * n;
        let mut w: f64 = self.active().map(|b| ((b.nd * (b.s() + n)) * 2 + b.nd * b.nd + nn) as f64 / 4.0).sum();
        w += (0..route.left).map(|k| self.sweep(k, Side::Left)).sum::<f64>();
        for b in self.active() {
            w += ((b.s() + b.nd) * nn) as f64;
            if b.copied {
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
            if b.copied {
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
            if b.copied {
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
                    if b.copied {
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

    /// The generator's gradient's matrices at a position (`prototype_gradient`): `whole` the static G's, `tones` the tones
    /// whose coefficient is not zero.
    pub(crate) fn gradient(&self, whole: bool, tones: &[usize]) -> f64 {
        let (n, layout) = (self.n, &self.layout);
        let nn = n * n;
        let mut w = 0.0;
        if whole {
            w += (3 * layout.pp + 3 * (layout.direct + layout.conj)) as f64;
            w += self.lit().map(|f| (3 * f.v * f.v * nn) as f64).sum::<f64>();
            for b in self.active() {
                let (s, nd, l) = (b.s(), b.nd, b.lanes);
                let sn = s * n;
                w += (b.reach() * n) as f64;
                w += (6 * l * sn) as f64;
                for _ in 0..3 {
                    w += (b.reach() * n) as f64;
                    if b.explicit > 0 {
                        w += (b.terms * (2 * sn + nn)) as f64;
                    }
                    w += (s * s * n + b.vectors * sn + 4 * s * nn) as f64;
                    if nd > 0 {
                        w += (b.exact * s * nd + 4 * s * nd * n + 3 * nd * nd * n + 2 * (s + nd) * nn + l * nd * n) as f64;
                        if b.explicit > 0 {
                            w += (b.exact * n * nd + 4 * nd * nn) as f64;
                        }
                    }
                    if b.copied {
                        w += (2 * b.copy_rows() * s * n + 2 * b.nc * nn + nn) as f64;
                    }
                }
            }
        }
        w += tones.iter().map(|&m| (3 * layout.tones[m]) as f64).sum::<f64>();
        w + (18 * n * nn) as f64
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
    /// nodes, each cross decay's three products L†L there -- and its forms (its members' bars at the three nodes, of tone
    /// `tone`).
    pub(crate) fn candidate(&self, members: usize, cross: usize, tone: usize) -> [f64; 2] {
        let n = self.n;
        let one: Vec<Cotangent> = (0..members).flat_map(|_| (0..3).map(|position| Cotangent { channel: 0, position, tone: Some(tone) })).collect();
        let forms = if one.is_empty() { 0.0 } else { self.force(&one) };
        [(3 * one.len() * n * n + 9 * cross * (GROUND + 1) * n * n) as f64, forms]
    }

    /// A step's price (real multiply-adds): the middle's held forward (`Middle::made`), the integrator's stretch with the
    /// model's counts at its nodes (step.rs `Price::stepped`: its count::* -- the propagation, the kick's sums and forms,
    /// the survival's hazard, the optical sector, the end's forms --, its picture as the integrator counted it: `picture`),
    /// the light scattered at the middle (`Price::scattering`), and `jumps` jumps' photons, places and landings.
    pub(crate) fn step(&self, route: &Route, mut picture: crate::step::Price, jumps: f64) -> f64 {
        let forward: f64 = self.forward(route).iter().sum();
        picture.middle = COMPLEX * forward;
        picture.scattering = COMPLEX * self.photons(Asked::Every, false);
        picture.work(self.costs(route)) + jumps * self.jump(route)
    }

    /// A jump's price (real multiply-adds): its place, its drawn class's photons there (the dearest class's: the sizes do not
    /// say which a jump draws), its landing's left response.
    pub(crate) fn jump(&self, route: &Route) -> f64 {
        let dearest = self.slots().into_iter().flat_map(|group| self.sets.iter().map(move |&set| Asked::Drawn { group, set })).map(|asked| self.photons(asked, true)).fold(0.0, f64::max);
        COMPLEX * (self.place() + dearest + self.left(route))
    }

    /// A cell's price (real multiply-adds, unify2 §3.3): its steps at the step's price, D/τ of them, and the once-only
    /// `once` (`decomposition`, `switch`).
    pub(crate) fn cell(&self, route: &Route, picture: crate::step::Price, jumps: f64, steps: f64, once: f64) -> f64 {
        steps * self.step(route, picture, jumps) + once
    }

    /// The once-only price of a switch to this representation (real multiply-adds): its decision's forward at the point
    /// on decompositions made there (every block's: `decomposition`, `distant`), every part's sweeps to the decision's, the
    /// state taken onto it (its left response) and its held forms -- the middle's tangent and the force of its static bar
    /// and its tones' bars at the middle (run.rs `Middle::decide`).
    pub(crate) fn switch(&self, route: &Route) -> f64 {
        let forward: f64 = self.forward(route).iter().sum();
        let made: f64 = self.site() + self.near.iter().map(|b| self.decomposition(b)).sum::<f64>() + self.far.iter().map(|f| self.distant(f)).sum::<f64>();
        let mut bars = vec![Cotangent { channel: 0, position: 0, tone: None }];
        bars.extend((0..self.layout.tones.len()).map(|m| Cotangent { channel: 1 + m, position: 0, tone: Some(m) }));
        let held = self.tangent(route).iter().sum::<f64>() + self.force(&bars);
        COMPLEX * (forward + made + self.left(route) + held)
    }

    /// A near block's decomposition at a point (complex multiply-adds): the fold's own matrices at the block
    /// (`effective`), its kept rows' Schur form part by part (`schur_count`) and every product on its basis (`basis`). The
    /// light at the point the decompositions share is `site`'s.
    pub(crate) fn decomposition(&self, b: &NearSize) -> f64 {
        self.effective(b) + b.parts.iter().map(|p| schur_count(p.rows)).sum::<f64>() + self.basis(b)
    }

    /// A near block's basis, its products (`near_basis` less its Schur forms; fold.rs `Factored::to`, `from`: Σ_P s_P² a
    /// column): per term the rotated form (two of s columns), its dressing's rows (s), the explicit states' source,
    /// emission and dressed source (n_x each); the level pairs' and the forms' norms (ne² a term and the reference's); the
    /// emission's three axes (GROUND each); per split its source and per vector its own (n each); the basis itself (s);
    /// per exact coupling vector its own and K̂_ED, K̂_DE (n_d each); the static source (n); the explicit emission (n_x).
    pub(crate) fn basis(&self, b: &NearSize) -> f64 {
        let (n, s, nx) = (self.n, b.s(), b.explicit);
        let squares: usize = b.parts.iter().map(|p| p.rows * p.rows).sum();
        let per = |cols: usize| (squares * cols) as f64;
        let terms = b.terms as f64 * (3.0 * per(s) + 3.0 * per(nx));
        terms + (b.all * b.all * (b.terms + 1)) as f64 + 3.0 * per(GROUND) + (b.splits + b.vectors) as f64 * per(n) + per(s) + (b.exact + 2) as f64 * per(b.nd) + per(n) + per(nx)
    }

    /// The fold's own matrices at a near block (fold.rs `Fold::lay`, `effective`; the second assembly's -- unify2 §5.4:
    /// U4 replaces it --): its rows' frame matrix and sources (m² + m n per split and the static one, m = ne + n_d + n_c),
    /// the centred copies' resolvent (n_c × n_c inverted) and their dressing of the solved rows (ne + n_d of them): K's,
    /// the static source's.
    pub(crate) fn effective(&self, b: &NearSize) -> f64 {
        let (n, c, solved) = (self.n, b.nc, b.all + b.nd);
        let m = solved + c;
        let laid = (m * m + m * n * (1 + b.splits)) as f64;
        if c == 0 {
            return laid;
        }
        laid + (c * c * c) as f64 * 4.0 / 3.0 + (solved * c * c + solved * c * solved + solved * c * n) as f64
    }

    /// A far block's pair forms at a point (`far_basis`): its rows' frame matrix and sources (fold.rs `lay`: m² + m n per
    /// split and the static one), their resolvent (`effective`: m × m inverted), R₀U_a, its emission and left response,
    /// and per pair U_a†R₀U_b with its two companions.
    pub(crate) fn distant(&self, f: &FarSize) -> f64 {
        let (n, m) = (self.n, f.m);
        (m * m + m * n * (1 + f.splits)) as f64 + (m * m * m) as f64 * 4.0 / 3.0 + (2 * f.v * m * m * n + 3 * f.v * f.v * m * n * n) as f64
    }

    /// The light at a point for the decompositions made there (fold.rs `Fold::site`): its forms' entries, each component's
    /// drive (EXCITED × GROUND a beam).
    pub(crate) fn site(&self) -> f64 {
        (self.layout.entries + self.layout.drives * EXCITED * GROUND) as f64
    }
}

/// The Schur form of an m × m matrix (linalg.rs `schur`): the Hessenberg reduction, then the QR iteration -- an
/// iteration's rotations per row of its window (four products an entry of each pair rotated: H's rows from the bulge on,
/// its columns above it, Q's columns) -- at two iterations an eigenvalue (Golub & Van Loan, Matrix Computations §7.5:
/// the Francis step's usual count; an iteration count, not a size: listed by the test against the QR's own).
pub(crate) fn schur_count(m: usize) -> f64 {
    let hessenberg: usize = (0..m.saturating_sub(2)).map(|k| (2 + 6 * m) * (m - k - 1)).sum();
    let mut qr = 0usize;
    for hi in (2..=m).rev() {
        let iteration: usize = (0..hi - 1).map(|k| 4 * ((m - k.saturating_sub(1)) + (k + 3).min(hi) + m)).sum();
        qr += 2 * iteration;
    }
    (hessenberg + qr) as f64
}

impl Workspace {
    /// The sizes of the representation the last forward stood on and of its evaluation (module documentation, # The
    /// sizes): its decompositions' parts in `cache`, the parts its sweeps held, the far blocks it left out, the centred
    /// copies' rows it took in; `sets` the landing sets (the ground states' partition).
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
                let basis = cache.at_near.get(x).copied().flatten().map(|at| &cache.near[x][at]);
                let parts = basis.map_or(vec![], |b| b.f.parts.iter().map(|(idx, at, _)| Span { rows: idx.len(), reach: (*at..at + idx.len()).map(|i| b.f.ends[i] - i).sum(), frozen: !nb.active[*at] }).collect());
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
                    copied: nb.copied_in,
                    levels: nb.levels.iter().map(Vec::len).collect(),
                    omitted: basis.map_or(vec![true; nl], |b| (0..nl).map(|q| !b.kept.contains(&q)).collect()),
                    all: nb.all,
                    group: fold.groups[nb.o],
                }
            })
            .collect();
        let far = self.far.iter().map(|fb| FarSize { v: fb.vectors.len(), m: fb.m, splits: fb.splits.len(), terms: fb.terms.len(), omitted: fb.omitted, group: fold.groups[fb.o] }).collect();
        Sizes { n: self.n, ground: fold.ground.len(), layout, near, far, sweeps: self.step.sweeps, sets: sets.iter().map(Vec::len).collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{At, Bar, Cache, Moments, PHASES, TANGENT};
    use super::*;

    /// Within the rounding of a sum of counts (each an integer or a quarter of one).
    fn same(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
    }

    /// What the points checked held (the formula's terms each reached), and the iterations against their prediction.
    #[derive(Default)]
    struct Tally {
        points: usize,
        explicit: usize,
        exact: usize,
        copied: usize,
        frozen: usize,
        omitted: usize,
        left_out: usize,
        tones: usize,
        multipart: usize,
        sparse: usize,
        normal: Vec<i64>,
        decompositions: Vec<(f64, f64, f64, f64)>,
        lyapunov: Vec<i64>,
        left: Vec<i64>,
    }

    /// A decision's forward at r0 and three forwards holding it along the path (`the_price_formula_is_the_counted_work`),
    /// each held one's counts against the formula item by item.
    fn check(fold: &Fold, sets: &[Vec<usize>], label: &str, tally: &mut Tally, verbose: bool) {
        let mut ws = match fold.prototype_workspace() {
            Ok(w) => w,
            Err(e) => {
                println!("{label}: not the prototype's ({e})");
                return;
            }
        };
        let n = fold.size();
        let speed = 0.1 / fold.hbar_over_mass;
        let kappa = [speed / 14f64.sqrt(), 2.0 * speed / 14f64.sqrt(), 3.0 * speed / 14f64.sqrt()];
        let velocity = kappa.map(|k| k * fold.hbar_over_mass);
        let mut cache = Cache::default();
        let r0 = [0.05, -0.03, 0.02];
        let Some(decided) = fold.prototype(r0, kappa, &mut cache, &mut ws).map(|s| (s.work, s.made, s.rounds, s.sweeps)) else {
            println!("{label}: the fold does not exist at the decision's point");
            return;
        };
        // The decision itself, listed: its decompositions (their QR and the fold's own matrices), its rounds.
        let sizes = ws.sizes(fold, &cache, sets);
        if verbose {
            let x = ws.step.metric.data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            let formula = sizes.forward(&Route::predicted(x, n, sizes.sweeps));
            let made: f64 = sizes.near.iter().map(|b| sizes.decomposition(b)).sum::<f64>() + sizes.far.iter().map(|f| sizes.distant(f)).sum::<f64>();
            println!(
                "{label}: near (s, parts, L, nd, nc, explicit) {:?}, far (v, m, left out) {:?}; the decision's forward ({} decompositions made, {} rounds, {} sweeps): counted {:.0} (its decompositions' {:.0}), the formula holding it {:.0}, every block decomposed once {made:.0} (QR at two iterations an eigenvalue; the fold's own matrices apart)",
                sizes.near.iter().map(|b| (b.s(), b.parts.len(), b.lanes, b.nd, b.nc, b.explicit)).collect::<Vec<_>>(),
                sizes.far.iter().map(|f| (f.v, f.m, f.omitted)).collect::<Vec<_>>(),
                decided.1,
                decided.2,
                decided.3,
                decided.0.iter().sum::<f64>(),
                decided.0[REFRESH],
                formula.iter().sum::<f64>() - formula[REFRESH]
            );
        }
        // Three points holding the decision's evaluation, and a fourth holding one it did not make: every centred copy's
        // rows in the compression, every far block left out, every part of a block without exact copies held at the first
        // sweep's order (the formula prices any evaluation; the scenes' decisions reach none of these at their points).
        for k in 1..=4 {
            let r: [f64; 3] = std::array::from_fn(|a| r0[a] + k as f64 * [0.02, 0.01, -0.015][a]);
            let at = format!("{label} point {k}{}", if k == 4 { " (an evaluation forced)" } else { "" });
            if k == 4 {
                let e = &mut cache.evaluation;
                e.copied.iter_mut().zip(&ws.near).for_each(|(c, nb)| *c = nb.nc > 0);
                e.far.iter_mut().for_each(|f| *f = true);
                e.active.iter_mut().zip(&ws.near).filter(|(_, nb)| nb.nd == 0).for_each(|(a, _)| a.iter_mut().for_each(|x| *x = false));
            }
            let Some(counted) = fold.prototype_within(r, kappa, &mut cache, &mut ws, f64::INFINITY, true).map(|s| s.work) else {
                println!("{at}: does not exist on what it holds");
                continue;
            };
            let sizes = ws.sizes(fold, &cache, sets);
            let x = ws.step.metric.data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            let predicted = Route::predicted(x, n, sizes.sweeps);
            let n3 = (n * n * n) as f64;
            // The forward, phase by phase; S⁻¹'s passes read off its phase's rest (whole counts of 3n³).
            let formula = sizes.forward(&predicted);
            for p in (0..10).filter(|&p| p != NORMAL) {
                assert!(same(counted[p], formula[p]), "{at}: the forward's {} counted {} against the formula's {}", PHASES[p], counted[p], formula[p]);
            }
            let passes = (counted[NORMAL] - (formula[NORMAL] - sizes.normal(&predicted)) - if predicted.started { n3 } else { 0.0 }) / (3.0 * n3);
            assert!(passes >= 1.0 && same(passes, passes.round()), "{at}: S⁻¹'s count {} is no whole number of passes ({passes})", counted[NORMAL]);
            let route = Route { normal: passes.round() as usize, ..predicted };
            assert!(same(counted[NORMAL], sizes.forward(&route)[NORMAL]));
            // The outer nodes.
            let nodes = fold.prototype_nodes(r, kappa, velocity, [-0.325, 0.325], &cache, &mut ws);
            assert!(same(nodes, sizes.nodes()), "{at}: the nodes counted {nodes} against {}", sizes.nodes());
            // The tangent, phase by phase; the Lyapunov passes read off its last phase (whole counts of n³).
            fold.prototype_tangent(&cache, &mut ws);
            let tangent = ws.tangent_work;
            let mine = sizes.tangent(&predicted);
            for p in 0..5 {
                assert!(same(tangent[p], mine[p]), "{at}: the tangent's {} counted {} against {}", TANGENT[p], tangent[p], mine[p]);
            }
            let lyapunov = (tangent[5] - mine[5]) / n3 + predicted.lyapunov.iter().sum::<usize>() as f64;
            assert!(same(lyapunov, lyapunov.round()), "{at}: the metric's tangent's count {} is no whole number of Lyapunov passes", tangent[5]);
            let lyapunov = lyapunov.round() as usize;
            let route = Route { lyapunov: [lyapunov, 0, 0], ..route };
            assert!(same(tangent[5], sizes.tangent(&route)[5]));
            // The force of the set's bars (run.rs `Set::new`): the three positions' static bars and every tone's; and a
            // candidate tone's forms (its members' bars, `Candidate::work`).
            let zero = vec![C64::ZERO; n * n];
            let tones = ws.rotating.len();
            let mut bars: Vec<Bar> = (0..3).map(|position| Bar { channel: 0, position, tone: None, matrix: &zero }).collect();
            bars.extend((0..tones).flat_map(|m| (0..3).map(move |position| (m, position))).map(|(m, position)| Bar { channel: 0, position, tone: Some(m), matrix: &zero }));
            let shape: Vec<Cotangent> = bars.iter().map(|b| Cotangent { channel: b.channel, position: b.position, tone: b.tone }).collect();
            let force = fold.prototype_force(&cache, &mut ws, &bars, 1).work;
            assert!(same(force, sizes.force(&shape)), "{at}: the force counted {force} against {}", sizes.force(&shape));
            for members in (1..=2).filter(|_| tones > 0) {
                let one: Vec<Bar> = (0..members).flat_map(|_| (0..3).map(|position| Bar { channel: 0, position, tone: Some(0), matrix: &zero })).collect();
                let forms = fold.prototype_force(&cache, &mut ws, &one, 1).work;
                assert!(same(forms, sizes.candidate(members, 0, 0)[1]), "{at}: a candidate's forms counted {forms} against {}", sizes.candidate(members, 0, 0)[1]);
            }
            // The photons: every class's at the middle; a jump's place, and every class drawn there.
            let psi: Vec<C64> = (0..n).map(|k| C64::cis(0.7 * k as f64) * (1.0 + 0.1 * k as f64)).collect();
            let every = fold.prototype_photons(&cache, &mut ws, &psi, sets, At::Middle, 0.0, Moments::Every);
            assert!(same(every.work, sizes.photons(Asked::Every, false)), "{at}: the photons counted {} against {}", every.work, sizes.photons(Asked::Every, false));
            let classes = every.classes.len();
            let place = fold.prototype_place(r, kappa, velocity, 0.1, &mut ws);
            assert!(same(place, sizes.place()), "{at}: the place counted {place} against {}", sizes.place());
            let slots = sizes.slots();
            for c in 0..classes {
                let mut draw = |rates: &[f64]| (c < rates.len()).then_some(c);
                let drawn = fold.prototype_photons(&cache, &mut ws, &psi, sets, At::Place, 0.0, Moments::Drawn(&mut draw));
                let class = &drawn.classes[c];
                let asked = Asked::Drawn { group: slots[class.slot], set: sets[class.landing].len() };
                assert!(same(drawn.work, sizes.photons(asked, true)), "{at}: class {c}'s photons counted {} against {}", drawn.work, sizes.photons(asked, true));
            }
            // The left response: its sweeps read off its count (the formula at each count of sweeps).
            let mut left = None;
            if fold.prototype_left(&cache, &mut ws) {
                let work = ws.left_work();
                let sweeps = (2..200).find(|&l| same(work, sizes.left(&Route { left: l, ..route }))).unwrap_or_else(|| panic!("{at}: the left response's count {work} at no count of sweeps (the formula at the predicted {}: {})", predicted.left, sizes.left(&predicted)));
                left = Some(sweeps);
            }
            // The gradient's matrices at the middle, where the set holds explicit states (the optical sector's).
            if !fold.explicit.is_empty() {
                let mut out: [Vec<C64>; 3] = Default::default();
                let coefficients = vec![C64::ONE; tones];
                let counted = fold.prototype_gradient(&cache, &mut ws, 0, true, &coefficients, &mut out);
                let formula = sizes.gradient(true, &(0..tones).collect::<Vec<_>>());
                assert!(same(counted, formula), "{at}: the gradient counted {counted} against {formula}");
            }
            // The step's model and once-only parts at these sizes (real multiply-adds; the integrator's picture none here).
            let picture = crate::step::Price { states: n, still: false, tau: 0.65, bandwidth: 0.0, nodes: 0, traces: 0, made: 0.0, stops: 0.0, sector: None, tones: 0, toned: [0.0; 2], centred: true, ended: true, middle: 0.0, scattering: 0.0 };
            let step = sizes.step(&route, picture, 0.0);
            let dense: Vec<Span> = sizes.near.iter().flat_map(|b| b.parts.iter().map(|p| Span::dense(p.rows))).collect();
            let reach: usize = sizes.near.iter().flat_map(|b| b.parts.iter().map(|p| p.reach)).sum();
            let full: usize = dense.iter().map(|p| p.reach).sum();
            if verbose {
                println!(
                    "{at}: equal; ‖X‖_F {x:.2e}: S⁻¹ {} passes (predicted {}), Lyapunov {lyapunov} over the axes (predicted {}), left sweeps {left:?} (predicted {}); the Schur forms' reach {reach} of {full} dense; a step (its model, the picture aside) {step:.0}, its node counts {:?}, a jump {:.0}, a switch to it {:.0}, a cell of two steps with every block decomposed {:.0} (real multiply-adds)",
                    route.normal,
                    predicted.normal,
                    3 * predicted.lyapunov[0],
                    predicted.left,
                    sizes.costs(&route).map(|c| (COMPLEX * c).round()),
                    sizes.jump(&route),
                    sizes.switch(&route),
                    sizes.cell(&route, picture, 0.0, 2.0, COMPLEX * sizes.near.iter().map(|b| sizes.decomposition(b)).sum::<f64>())
                );
            }
            tally.points += 1;
            tally.explicit += usize::from(!fold.explicit.is_empty());
            tally.exact += usize::from(sizes.near.iter().any(|b| b.nd > 0));
            tally.copied += usize::from(sizes.near.iter().any(|b| b.copied));
            tally.frozen += usize::from(sizes.near.iter().any(|b| b.parts.iter().any(|p| p.frozen)));
            tally.omitted += usize::from(sizes.far.iter().any(|f| f.omitted));
            tally.left_out += usize::from(sizes.near.iter().any(|b| !b.idle() && b.omitted.iter().any(|&o| o)));
            tally.tones += usize::from(tones > 0);
            tally.multipart += usize::from(sizes.near.iter().any(|b| b.parts.len() > 1));
            tally.sparse += usize::from(reach < full);
            // The decompositions the representation stands on, made again at the point (`near_basis`, `far_basis`): the
            // prototype's products the formula's to the rounding, the Schur forms' QR listed against their estimate, the
            // fold's own matrices at the point (`site`, `lay`, `effective`) listed apart.
            if k == 3 {
                let mut site_work = 0.0;
                let site = fold.site(r, kappa, &mut site_work);
                assert!(same(site_work, sizes.site()), "{at}: the light at the point counted {site_work} against {}", sizes.site());
                let (mut own, mut products, mut qr, mut estimate) = (site_work, 0.0, 0.0, 0.0);
                for (x, nb) in ws.near.iter().enumerate() {
                    let basis = &cache.near[x][cache.at_near[x].unwrap()];
                    let (mut laid_work, mut basis_work, mut schur_work) = (0.0, 0.0, 0.0);
                    let laid = fold.lay(nb.o, &site, &mut laid_work);
                    let Some(e) = fold.effective(nb.o, &laid, &mut laid_work) else { continue };
                    if fold.near_basis(nb, &ws, &laid, &e, &basis.kept, &mut basis_work).is_none() {
                        continue;
                    }
                    let er: Vec<usize> = (0..nb.all).collect();
                    Factored::of(&e.k.block(&er, &er).block(&basis.rows, &basis.rows), &mut schur_work).unwrap();
                    let b = &sizes.near[x];
                    assert!(same(laid_work, sizes.effective(b)), "{at}: block {x}'s own matrices counted {laid_work} against {}", sizes.effective(b));
                    assert!(same(basis_work - schur_work, sizes.basis(b)), "{at}: block {x}'s basis counted {basis_work} (its Schur forms {schur_work}) against the formula's {}", sizes.basis(b));
                    let mine: f64 = b.parts.iter().map(|p| schur_count(p.rows)).sum();
                    (own, products, qr, estimate) = (own + laid_work, products + sizes.basis(b), qr + schur_work, estimate + mine);
                }
                for (x, fb) in ws.far.iter().enumerate() {
                    let mut basis_work = 0.0;
                    if fold.far_basis(fb, &ws, &site, &mut basis_work).is_none() {
                        continue;
                    }
                    let formula = sizes.distant(&sizes.far[x]);
                    assert!(same(basis_work, formula), "{at}: far block {x}'s forms counted {basis_work} against the formula's {formula}");
                    products += formula;
                }
                if verbose {
                    println!("{at}: every block decomposed again at the point: the light there, the fold's own matrices and the products {:.0} (equal), the Schur forms' QR {qr:.0} against the estimate {estimate:.0}", own + products);
                }
                tally.decompositions.push((products, qr, estimate, own));
            }
            tally.normal.push(route.normal as i64 - predicted.normal as i64);
            tally.lyapunov.push(lyapunov as i64 - 3 * predicted.lyapunov[0] as i64);
            if let Some(l) = left {
                tally.left.push(l as i64 - predicted.left as i64);
            }
        }
    }

    /// The price formula is the counted work (BRIEF_P_PRICE item 2; unify2 §3.3, §9 U2): on the five scenes (PGC3D_1ms,
    /// GMD1C_1ms where its request is at hand, tilt2 = PGC1D_TILT_2, R4_LIN_4, S2Q), every set the prototype holds by the
    /// tests' choice (`proto_choice`: the explicit nodes the focus reads beyond the share), a decision's forward at a point
    /// (an atom moving at 0.1 m/s) and three forwards holding it at points along the path; and, for the terms no set of the
    /// scenes reaches (exact copies, centred copies' rows in the compression), the 60 random systems (`random_system`:
    /// their copies decomposed, an explicit level in every third). At each held point, the sizes read off the
    /// representation and its evaluation (`Workspace::sizes`), and the formula against what the prototype counted for the
    /// same evaluation, item by item: the forward by phase, the outer nodes, the tangent by phase, the force of the set's
    /// bars (run.rs `Set::new`) and a candidate tone's, the photons of every class at the middle, a jump's place and every
    /// class's drawn photons there, the left response, and the gradient's matrices where the set holds explicit states.
    /// Equal to the rounding but for the iterations (module documentation, # What is not a size): S⁻¹'s Newton–Schulz
    /// passes, the Lyapunov passes and the left series' sweeps -- what each took against `Route::predicted`, listed, the
    /// rest of the count equal. The decision's forward (its decompositions, its sweeps to the share, its rounds), the
    /// decompositions' QR and the Schur forms' reach against dense are listed. `PRICE_SCENE` runs one scene alone.
    #[test]
    fn the_price_formula_is_the_counted_work() {
        let refs = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs");
        let path = |dir: &str, file: &str| refs.join(dir).join(file).to_string_lossy().into_owned();
        let gmd = r"C:\Users\eadri\AppData\Local\Temp\claude\c0815e347c54866c\8308d66c-699b-4483-ab9b-7305d93a9f7e\scratchpad\final\req\GMD1C_1ms.json".to_string();
        let only = std::env::var("PRICE_SCENE").ok();
        let scenes = [
            ("PGC3D_1ms", path("requests", "PGC3D_1ms.json")),
            ("GMD1C_1ms", gmd),
            ("tilt2", path("requests", "PGC1D_TILT_2.json")),
            ("R4_LIN_4", path("scenes", "R4_LIN_4.json")),
            ("S2Q", path("requests", "S2Q.json")),
        ];
        let mut tally = Tally::default();
        for (name, file) in scenes {
            if only.as_ref().is_some_and(|o| o != name) {
                continue;
            }
            if !std::path::Path::new(&file).exists() {
                println!("{name}: its request is not at hand ({file}), skipped");
                continue;
            }
            let clock = std::time::Instant::now();
            let req = crate::model::testing::scene(&file);
            let choice = super::super::super::tests::proto_choice(&req, name);
            let sets = Fold::sets(&req, &choice).unwrap();
            for set in &sets {
                let fold = Fold::new(&req, &choice, set[0]).unwrap();
                if !fold.blocks.is_empty() {
                    check(&fold, &sets, &format!("{name} set {set:?} (n {}, {} explicit)", fold.size(), fold.explicit.len()), &mut tally, true);
                }
            }
            println!("{name}: {:.1?}", clock.elapsed());
        }
        let scenes = tally.points;
        if only.is_none() {
            // A Λ whose two ground states beat (a tone of Ĥ_PP, its envelope a standing wave).
            let req = super::super::super::tests::toned_lambda();
            let choice = super::super::super::tests::proto_choice(&req, "a Λ with a tone");
            let sets = Fold::sets(&req, &choice).unwrap();
            for set in &sets {
                let fold = Fold::new(&req, &choice, set[0]).unwrap();
                if !fold.blocks.is_empty() {
                    check(&fold, &sets, &format!("a Λ with a tone, set {set:?}"), &mut tally, true);
                }
            }
            let clock = std::time::Instant::now();
            let mut state = 20261117u64;
            for trial in 0..60 {
                let (req, choice) = super::super::super::tests::random_system(&mut state, trial);
                let sets = Fold::sets(&req, &choice).unwrap();
                for set in &sets {
                    let fold = Fold::new(&req, &choice, set[0]).unwrap();
                    if !fold.blocks.is_empty() {
                        check(&fold, &sets, &format!("random system {trial} set {set:?}"), &mut tally, false);
                    }
                }
            }
            println!("the random systems: {} points, {:.1?}", tally.points - scenes, clock.elapsed());
        }
        let worst = |x: &[i64]| (x.iter().copied().min().unwrap_or(0), x.iter().copied().max().unwrap_or(0));
        println!(
            "{} points equal item by item ({scenes} on the scenes): {} with explicit states, {} with exact copies, {} with centred copies' rows in, {} with parts held at the first order, {} with far blocks left out, {} with levels left out of a solved block, {} with tones, {} with several Schur parts, {} with a Schur form's zeros; the iterations' misses of their prediction (least, most): S⁻¹ {:?}, Lyapunov over the axes {:?}, left {:?}",
            tally.points,
            tally.explicit,
            tally.exact,
            tally.copied,
            tally.frozen,
            tally.omitted,
            tally.left_out,
            tally.tones,
            tally.multipart,
            tally.sparse,
            worst(&tally.normal),
            worst(&tally.lyapunov),
            worst(&tally.left)
        );
        let sum = |f: &dyn Fn(&(f64, f64, f64, f64)) -> f64| tally.decompositions.iter().map(f).sum::<f64>();
        println!(
            "the decompositions made again ({} representations): the products {:.3e} and the fold's own matrices with the light at the point {:.3e} (equal), the Schur forms' QR {:.3e} against the estimate {:.3e} (complex multiply-adds in all)",
            tally.decompositions.len(),
            sum(&|d| d.0),
            sum(&|d| d.3),
            sum(&|d| d.1),
            sum(&|d| d.2)
        );
        assert!(scenes > 0);
    }
}
