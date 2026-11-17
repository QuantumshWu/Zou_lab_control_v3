//! The look over a cell (unify2 §4.2, §2.3, §2.7; BRIEF_P_LOOK): at a decision, on the forward the decision made at its
//! point, what the fold's existence and its readings become along the cell to the next decision -- from the bare inputs
//! along the reference flow's path and each column's pass 0 at points, without a forward there.
//!
//! # What is smooth along a cell
//! The bare inputs: every beam's amplitude at the path's point (its plane wave turned and its envelope, field.rs -- as the
//! integrator's outer nodes take the light), the lines' places (a near block's kept rows' diagonal moved by the forms to
//! first order, `moved`: the forms are bilinear in the field) and every lane's Doppler shift w_l from the path's momentum.
//! The folded column energy is not: near x → 1 its P-like root turns sharply, and on an exceptional line it changes branch
//! (loss.status 2). So the inputs are what is interpolated, and the fold is solved at points (unify2 §4.2, the toy
//! `lookahead.py`: the folded distance interpolated erred by 0.3–1.1 of the cell's largest excitation, the bare inputs'
//! critical points with the fold solved there by 1.3·10⁻² at D = 0.4 µs).
//!
//! # The samples
//! m = max(p + 1, ⌈ν_max D/π⌉ + 1) points evenly over the cell [0, D], made odd, p = 2 the panels' degree
//! (`Quadratic::POINTS` = p + 1 = 3; unify2 §10 item 4). Why p + 1 is the least, derived: what the look must see is an
//! extremum of a bare input inside the cell -- the line's distance |d(t)| at its least, the pattern's intensity at its
//! most, where a column's excitation x ∝ |N|²/|d|² peaks between two samples. An interpolant of degree one through two
//! samples is monotone between them: its extremum is always an end, and a peak inside the cell is never seen (measured,
//! the toy `lookahead.py`: the cell's ends alone missed 13–64 % of the cell's largest x). Degree two is the least whose
//! interpolant has an extremum inside, at −b/2c (`Quadratic::critical`), and its p + 1 = 3 samples are the coarse step's
//! own three points (its ends and its middle: the integrator's quadratic through the step's nodes, the same order). Degree
//! two is also enough: along a cell the bare inputs are smooth on the step's scale (the step resolves them), so their
//! cubic and higher terms are the step's own error order. Beyond that, no fewer than the light pattern's Nyquist count,
//! its fastest beat along the path ν_max = max |(k_l − k_l′)·v| over two beams of one frequency at the cell's three points
//! -- the only beat a column's inputs carry: a beam's own phase is its Doppler shift, exact in w_l, and two frequencies
//! reach a column at two offsets, never one. π is the Nyquist spacing, two samples a period of the beat (the spectral
//! routes' bound's origin). Every three consecutive samples are a panel (odd m: the panels tile the cell). No tuned
//! number.
//!
//! # The critical points
//! Every bare input is one quadratic over a panel (through its three samples): per column j, near block, kept row i and
//! lane l the bare distance d⁰ = a_j − w_l − κ_i (a_j the column's bare energy, # The column's pass 0, with the off-diagonal
//! part held) and the numerator N = Ē_ij V_lij (the emission's conjugate times the lane's source: gauge-free, it turns with
//! the pattern's beats alone), and per row the source's intensity |E_ij|² (the pattern). Their zeros (of Re d⁰) and their
//! extrema within the panel are where a column's excitation can peak between two samples. Of them, per column and panel,
//! the one whose own term |N/d⁰²| the interpolants read largest, where it is larger than at the panel's three samples, is
//! a critical point; and so is the largest of the column's solved excitation between the panel's samples, by its values
//! and its slopes there (a quintic Hermite; the slope the fold's own one step on, the forward difference of least error)
//! -- near an exceptional point the root turns fast where no input has a critical point (a standing wave's lanes at their
//! own Doppler shifts: the excitation rose 0.52 → 0.66 in a fifth of the cell, its quadratic through the samples read
//! 0.62; measured, the tests). The light is placed there exactly, as at a sample (the toy's: its critical points' inputs
//! exact), and the column solved from the point before.
//!
//! # The column's pass 0 at a point
//! c = a_j + Σ_b Σ_i Σ_l N_lij/(c − w_l − κ_i) -- the series' pass 0 with each column at its own energy (DESIGN §1.5,
//! H⁽⁰⁾ = diag), the diagonal of the block's Sylvester equation; a_j = H⁰_jj, Ĥ_PP's diagonal with the far blocks' held
//! forms and the explicit states' dressing (the forward's H base). Solved by scalar Newton from the point before (the first
//! from the decision's own column energy) until its step is within what the residual's rounding alone makes (Higham's γ of
//! its terms, over the slope), in at most the mantissa's rounds (an iteration that halves its distance a round at least:
//! step.rs `ROUNDS`); where that root is not the fold's -- no ratio below one: past an exceptional point the continuation
//! is the excited-like branch -- again from the bare energy, the series' own start (the fold at a point is the root its
//! series reaches from there; one column, two levels: at most one root has r < 1, r₁r₂ = 1). Of the solution: the
//! excitation x_j = Σ_b Σ_i |Σ_l V_lij/(c − w_l − κ_i)|² with the far blocks' held responses M_jj (the forward's `whole`),
//! the ratio r_j = |Σ′(c)| = |Σ N/d²| -- one column and one lane, r = x (unify2 §2.3) -- and whether the fold exists there:
//! the iteration converged and r < 1.
//!
//! # What is held: the off-diagonal part
//! The forward's last sweep solved, lane by lane and row by row, (c_j − w_l − κ_i) Z_lij = V_lij + O_lij at its column
//! energies c (H⁽ᴷ⁻¹⁾'s diagonal): O the source the rest of the equation adds to the lane's own -- the other columns'
//! −(ZΔ)_ij (the Raman part), a Schur form's triangular coupling Σ_m T_im Z_lmj, the exact copies' Woodbury rows (a row
//! whose part's sweeps ended before the last, `Near::upto`: what its own last sweep added, at that sweep's column
//! energies). The look holds O as the sweep made it,
//! O = Z(c − w − κ) − V at the decision, and the exact copies' emission into the explicit states (Ê_D Z_D)_jj on their
//! columns' bare energy; it solves the diagonal again on the sources V + O. At the decision its Σ_j at the sweep's
//! energies is the forward's H_jj to the rounding (`Look::identity`), and its root the series' fixed point (neither its
//! last sweep's -- the series' rest is not held -- nor the order of a part's fewer sweeps). One column: O = 0, the look is the
//! fold itself; several: what moves of O along the cell is the error (measured: the tests).
//!
//! # The profiles (unify2 §2.7)
//! Every account's numerator (a coupling times a response) is the decision's; what moves along the cell is a distance or
//! an excitation, each a quantity here at every point seen (`Look::largest`, `Look::at` between two points):
//! - the series of a part at p passes: r^p/(1 − r) (`Seen::series`);
//! - a centring: x/(1 − x) of its factor, ‖R₀‖ times the column energy's distance from the centre with the lanes' spread and
//!   the forms' change (proto.rs `centring`; `Look::centred`, `Look::centred_far`);
//! - a node left out: 1/|d| at its held energy (`Seen::distance`);
//! - the ring: |z|² = |numerator|²/|d|², a block's part of the excitation (`Inputs::part`);
//! - the chirp (M7): |z_ij|·|ḋ_ij|/|d_ij|² relative to the column's response, ḋ from its panel's quadratic (`Seen::chirp`);
//! - the reuse: Σ_t |δc_t| ‖Ñ_t‖, the field's drift from the decomposition (`Inputs::drift`; it grows from zero after a
//!   refresh);
//! - the third order, the cross-set coherence, the interference out of a group and the tones' sidebands: the decision's.
//!
//! The look counts its work (complex multiply-adds, `Look::work`): the price's (unify2 §3.3).
//!
//! # What it does not read
//! - The series' ratio where P's own off-diagonal sets it: an explicit state's drive Δ between P columns, against the
//!   rows' distances, Δ/|d| -- the diagonal's ratio does not see it (R4_LIN_4's F = 1 set with its D2 F′ = 2 explicit: the
//!   forward's 0.216, the look's 0.002; its F = 2 set, ground columns alone: 0.057 against 0.054). U3's existence there
//!   is the decision's measured ratio, its profile to be had from Δ at the points (Ĥ_PP's off-diagonal is placed with
//!   the light) over the rows' distances (plook.status.md).
//! - The exact copies' and the centred copies' own distances: their rows are in O, held.
use super::*;

/// The rounds of a column's Newton iteration (# The column's pass 0): step.rs `ROUNDS`, the mantissa's bits.
const ROUNDS: usize = f64::MANTISSA_DIGITS as usize;

/// A near block's kept rows' energies at a point whose pairs' coefficients are `coef` (proto.rs module documentation, 2):
/// κ_i = T_ii − o + Σ_t Re δc_t (V†N_tV)_ii, δc_t = c_t − c_t,ref into `delta`. The forward's and the look's.
pub(super) fn moved(basis: &NearBasis, value: f64, terms: &[usize], coef: &[C64], kappa: &mut [C64], delta: &mut [C64], work: &mut f64) {
    let s = kappa.len();
    for (i, k) in kappa.iter_mut().enumerate() {
        *k = basis.f.t[(i, i)] - C64::real(value);
    }
    for (t, &p) in terms.iter().enumerate() {
        let d = coef[p] - basis.at[t];
        delta[t] = d;
        if d == C64::ZERO {
            continue;
        }
        for (k, g) in kappa.iter_mut().zip(&basis.diag[t * s..(t + 1) * s]) {
            k.re += (d * *g).re;
        }
        *work += s as f64;
    }
}

/// A near block's sources at a point whose light is (f, w, e) (proto.rs module documentation, 3): per beam split
/// Ṽ_l = f_l Ỹ_l and its Doppler shift, Ṽ = Σ_a e_a Y_a, the explicit states' parts (`explicit_parts`) and the static
/// lane. The forward's and the look's.
#[allow(clippy::too_many_arguments)]
pub(super) fn sourced(splits: &[(usize, usize)], vectors: &[(usize, usize)], xs: &[usize], exact: &[(usize, usize)], lanes: usize, nd: usize, basis: &NearBasis, f: &[Vec<C64>], w: &[Vec<f64>], e: &[Vec<C64>], delta: &[C64], shifts: &mut [f64], vt: &mut [C64], vtot: &mut [C64], stat: &mut [C64], emit: &mut [C64], hx: &mut [C64], b0d: &mut [C64], exd: &mut [C64], s: usize, n: usize, work: &mut f64) {
    let (ln, bn) = (lanes * n, splits.len() * n);
    for (sp, &(cc, j)) in splits.iter().enumerate() {
        shifts[sp] = w[cc][j];
        let fl = f[cc][j];
        for i in 0..s {
            let (at, from) = (i * ln + sp * n, i * bn + sp * n);
            for (v, y) in vt[at..at + n].iter_mut().zip(&basis.beams[from..from + n]) {
                *v = fl * *y;
            }
        }
    }
    vtot.fill(C64::ZERO);
    for (a, &(cc, k)) in vectors.iter().enumerate() {
        let ek = e[cc][k];
        for (v, y) in vtot.iter_mut().zip(&basis.vectors[a]) {
            *v += ek * *y;
        }
    }
    *work += ((splits.len() + vectors.len()) * s * n) as f64;
    explicit_parts(xs, exact, basis, e, delta, vtot, stat, emit, hx, b0d, exd, s, nd, n, work);
    if !xs.is_empty() {
        scatter(stat, vt, s, ln, (lanes - 1) * n, n);
    }
}

/// The column energies of the sweep that solved row i of a near block last: the sweep its part's sweeps reach
/// (`Near::upto`) of those the forward made (`Workspace::cs`, n a sweep) -- at a decision every row's last, the forward's.
fn solved<'a>(nb: &Near, cs: &'a [C64], i: usize, n: usize) -> &'a [C64] {
    let k = nb.upto[i].min(cs.len() / n) - 1;
    &cs[k * n..(k + 1) * n]
}

/// The samples of a cell of width D whose pattern's fastest beat is ν (# The samples): max(p + 1, ⌈νD/π⌉ + 1), odd -- p + 1
/// a panel's points (`Quadratic::POINTS`), the least that sees an extremum inside the cell.
pub(crate) fn samples(beat: f64, cell: f64) -> usize {
    let m = ((beat * cell / std::f64::consts::PI).ceil() as usize + 1).max(Quadratic::POINTS);
    m + 1 - m % 2
}

/// A quadratic over a panel, u ∈ [−1, 1] from its first sample to its third: q(u) = a + b u + c u².
#[derive(Clone, Copy)]
struct Quadratic {
    a: C64,
    b: C64,
    c: C64,
}

impl Quadratic {
    /// Its points, degree two plus one: the least degree whose interpolant has an extremum between its ends (# The samples).
    const POINTS: usize = 3;

    /// Through (−1, y0), (0, y1), (1, y2).
    fn of(y0: C64, y1: C64, y2: C64) -> Quadratic {
        Quadratic { a: y1, b: (y2 - y0) * 0.5, c: (y0 + y2) * 0.5 - y1 }
    }
    fn at(&self, u: f64) -> C64 {
        self.a + (self.b + self.c * u) * u
    }
    /// dq/du.
    fn slope(&self, u: f64) -> C64 {
        self.b + self.c * (2.0 * u)
    }
    /// The zeros of its real part within the panel and its real part's extremum inside it.
    fn critical(&self, out: &mut Vec<f64>) {
        let (a, b, c) = (self.a.re, self.b.re, self.c.re);
        if c == 0.0 {
            if b != 0.0 {
                out.push(-a / b);
            }
        } else {
            out.push(-b / (2.0 * c));
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let q = -0.5 * (b + disc.sqrt().copysign(b));
                out.push(q / c);
                if q != 0.0 {
                    out.push(a / q);
                }
            }
        }
        out.retain(|u| u.abs() < 1.0);
    }
}

/// A column's pass 0 at a point (# The column's pass 0): its energy c, its excitation x, its ratio r = |Σ′(c)|, whether the
/// fold exists there (converged, r < 1).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Column {
    pub(crate) c: C64,
    pub(crate) x: f64,
    pub(crate) r: f64,
    pub(crate) exists: bool,
}

/// A near block's bare inputs at a point (# The critical points) and what its column's pass 0 reads on it: the lanes'
/// Doppler shifts (the static lane's zero), the kept rows' energies κ_i, the lanes' sources V_lij and numerators N_lij =
/// Ē_ij V_lij (each laid as the block's lanes, i·Ln + l·n + j), the sources' intensity |E_ij|² (i·n + j); the reuse's
/// drift Σ_t |δc_t| ‖Ñ_t‖ (the parts' off-diagonal norms on the decomposition) and the centred copies' forms' change
/// Σ_t |δc_t| ‖N_t,CC‖ (`centring`'s); per column its part of the excitation and its nearest distance min |c − w_l − κ_i|.
#[derive(Clone, Default)]
pub(crate) struct Inputs {
    pub(crate) shifts: Vec<f64>,
    pub(crate) kappa: Vec<C64>,
    vt: Vec<C64>,
    num: Vec<C64>,
    intensity: Vec<f64>,
    pub(crate) drift: f64,
    pub(crate) held: f64,
    pub(crate) part: Vec<f64>,
    pub(crate) nearest: Vec<f64>,
}

/// A point the look saw (`Look::seen`): its time from the decision (µs), whether a sample, per column its bare energy a_j
/// (H⁰_jj) and its far blocks' held response M_jj, per near block its inputs, per far block its forms' change and lanes'
/// shifts (the static lane last, as the forward's reading takes them); per column its pass 0 and its chirp (M7).
#[derive(Clone, Default)]
pub(crate) struct Seen {
    pub(crate) t: f64,
    pub(crate) sample: bool,
    bare: Vec<C64>,
    far_m: Vec<f64>,
    pub(crate) near: Vec<Inputs>,
    pub(crate) far: Vec<(f64, Vec<f64>)>,
    pub(crate) columns: Vec<Column>,
    pub(crate) chirp: Vec<f64>,
    /// At a sample, per column the excitation's slope along the path dx/dt (# The critical points).
    slope: Vec<f64>,
}

impl Seen {
    /// The series of a part at `passes` passes on column j, r^p/(1 − r) (§2.7): infinite where the fold does not exist.
    pub(crate) fn series(&self, j: usize, passes: i32) -> f64 {
        let x = self.columns[j];
        if x.exists { x.r.powi(passes) / (1.0 - x.r) } else { f64::INFINITY }
    }
    /// A node's distance from column j through near block b's lanes, min_l |c_j − w_l − κ| (§2.7: a node left out at its
    /// held energy κ).
    pub(crate) fn distance(&self, b: usize, j: usize, kappa: C64) -> f64 {
        let c = self.columns[j].c;
        self.near[b].shifts.iter().map(|&w| (c - C64::real(w) - kappa).abs()).fold(f64::INFINITY, f64::min)
    }
    /// The far blocks' held response on column j, M_jj (§2.7: a far block left out reads its response).
    pub(crate) fn far_response(&self, j: usize) -> f64 {
        self.far_m.get(j).copied().unwrap_or(0.0)
    }
}

/// The look over a cell (module documentation): the cell's width, the pattern's beat and the samples; the points seen, by
/// time; what is held (# What is held): per near block the off-diagonal source O (laid as its lanes), per column the
/// exact copies' emission into it; the largest relative difference of the forward's H_jj and the look's diagonal at the
/// forward's last sweep (`identity`: the rounding); its work. Its buffers -- the light at a point, per near block its
/// sources' temporaries -- made once for a workspace (`Fold::look`).
pub(crate) struct Look {
    pub(crate) cell: f64,
    pub(crate) beat: f64,
    pub(crate) samples: usize,
    pub(crate) seen: Vec<Seen>,
    held: Vec<Vec<C64>>,
    offset: Vec<C64>,
    pub(crate) identity: f64,
    pub(crate) work: f64,
    /// Per near block (‖R₀‖, z₀) where it centres copies; per far block its own (‖R₀‖, z₀).
    centres: Vec<Option<(f64, f64)>>,
    far_centres: Vec<(f64, f64)>,
    at: Placed,
    temp: Vec<Temps>,
}

/// A near block's sources' temporaries at a point (`sourced`).
#[derive(Default)]
struct Temps {
    delta: Vec<C64>,
    vtot: Vec<C64>,
    stat: Vec<C64>,
    emit: Vec<C64>,
    hx: Vec<C64>,
    b0d: Vec<C64>,
    exd: Vec<C64>,
}

impl Look {
    /// Whether the fold exists at every point seen, every column.
    pub(crate) fn exists(&self) -> bool {
        self.seen.iter().all(|s| s.columns.iter().all(|c| c.exists))
    }
    /// The first point where a column's fold does not exist: its time and the column.
    pub(crate) fn ceases(&self) -> Option<(f64, usize)> {
        self.seen.iter().find_map(|s| s.columns.iter().position(|c| !c.exists).map(|j| (s.t, j)))
    }
    /// A quantity's largest over the points seen (one that is no number is infinite).
    pub(crate) fn largest(&self, of: impl Fn(&Seen) -> f64) -> f64 {
        self.seen.iter().map(|s| of(s)).map(|x| if x.is_nan() { f64::INFINITY } else { x }).fold(0.0, f64::max)
    }
    /// A quantity at the time t of the cell, linear between the two points seen about it (the per-step booking, §2.7).
    pub(crate) fn at(&self, t: f64, of: impl Fn(&Seen) -> f64) -> f64 {
        let k = self.seen.partition_point(|s| s.t <= t).clamp(1, self.seen.len() - 1);
        let (a, b) = (&self.seen[k - 1], &self.seen[k]);
        let u = if b.t > a.t { ((t - a.t) / (b.t - a.t)).clamp(0.0, 1.0) } else { 0.0 };
        of(a) * (1.0 - u) + of(b) * u
    }
    /// The held off-diagonal part's size at the decision (# What is held): ‖O‖ over the sources ‖V + O‖, all near blocks.
    pub(crate) fn off_diagonal(&self) -> f64 {
        let (mut held, mut whole) = (0.0, 0.0);
        for (o, inp) in self.held.iter().zip(&self.seen[0].near) {
            held += o.iter().map(|z| z.norm_sqr()).sum::<f64>();
            whole += inp.vt.iter().map(|z| z.norm_sqr()).sum::<f64>();
        }
        if whole > 0.0 { (held / whole).sqrt() } else { 0.0 }
    }
    /// A near block's centring of column j at point k, x/(1 − x) (`centring`); zero where it centres no copy.
    pub(crate) fn centred(&self, k: usize, b: usize, j: usize) -> f64 {
        self.centred_at(&self.seen[k], b, j)
    }
    /// The same at a point seen.
    pub(crate) fn centred_at(&self, s: &Seen, b: usize, j: usize) -> f64 {
        self.centres[b].map_or(0.0, |(r0, z0)| centring(r0, z0, s.near[b].held, s.columns[j].c, &s.near[b].shifts))
    }
    /// A far block's centring of column j at point k (`centring`, as the forward reads it); zero where it is left out.
    pub(crate) fn centred_far(&self, k: usize, x: usize, j: usize) -> f64 {
        self.centred_far_at(&self.seen[k], x, j)
    }
    /// The same at a point seen.
    pub(crate) fn centred_far_at(&self, s: &Seen, x: usize, j: usize) -> f64 {
        let (r0, z0) = self.far_centres[x];
        if s.far[x].1.is_empty() { 0.0 } else { centring(r0, z0, s.far[x].0, s.columns[j].c, &s.far[x].1) }
    }
}

impl Fold {
    /// A look's buffers for a workspace (`prototype_look`).
    pub(crate) fn look(&self, ws: &Workspace) -> Look {
        Look {
            cell: 0.0,
            beat: 0.0,
            samples: 0,
            seen: vec![],
            held: vec![],
            offset: vec![],
            identity: 0.0,
            work: 0.0,
            centres: vec![],
            far_centres: vec![],
            at: ws.mid.clone(),
            temp: ws.near.iter().map(|_| Temps::default()).collect(),
        }
    }

    /// The light pattern's fastest beat along a path through the points (r, κ) (# The samples): max |(k_l − k_l′)·v|
    /// over two beams of one frequency, v = (ħ/m)κ.
    fn beat(&self, points: &[([f64; 3], [f64; 3])]) -> f64 {
        let mut most = 0.0_f64;
        for f in &self.light.frequencies {
            for (a, &l) in f.beams.iter().enumerate() {
                for &m in &f.beams[a + 1..] {
                    let (kl, km) = (self.light.beams[l].wave_vector(), self.light.beams[m].wave_vector());
                    for (_, kappa) in points {
                        let v = (0..3).map(|q| (kl[q] - km[q]) * self.hbar_over_mass * kappa[q]).sum::<f64>();
                        most = most.max(v.abs());
                    }
                }
            }
        }
        most
    }

    /// The look over a cell of width `cell` from a decision (module documentation): `path(t)` is the reference flow's
    /// (r, κ) at the time t ∈ [0, cell] after the decision, path(0) the point of the forward just made on `cache` in `ws`
    /// (`prototype`, the decision's: its last sweep is what the look holds the off-diagonal part of). Into `look`;
    /// nothing of the workspace changes.
    pub(crate) fn prototype_look(&self, path: &dyn Fn(f64) -> ([f64; 3], [f64; 3]), cell: f64, cache: &Cache, ws: &Workspace, look: &mut Look) {
        let n = ws.n;
        look.work = 0.0;
        look.cell = cell;
        look.beat = self.beat(&[path(0.0), path(0.5 * cell), path(cell)]);
        look.samples = samples(look.beat, cell);
        look.centres = (0..ws.near.len()).map(|x| cache.near[x].as_deref().filter(|_| ws.near[x].nc > 0).map(|b| (b.r0, b.z0))).collect();
        look.far_centres = (0..ws.far.len()).map(|x| cache.far[x].as_deref().map_or((0.0, 0.0), |b| (b.r0, b.z0))).collect();
        // What is held (# What is held): per near block the source its last sweep added, O = Z(c − w − κ) − V at the
        // sweep's column energies, and per column the exact copies' emission into the explicit states.
        look.held = ws
            .near
            .iter()
            .map(|nb| {
                let ln = nb.lanes * n;
                let rows = if nb.idle() { 0 } else { nb.s };
                let mut o = vec![C64::ZERO; rows * ln];
                for (i, &kappa) in nb.kappa[..rows].iter().enumerate() {
                    // A row was solved last at the sweep its part's sweeps reach, at that sweep's column energies.
                    let c = solved(nb, &ws.cs, i, n);
                    for (l, &w) in nb.shifts.iter().enumerate() {
                        for j in 0..n {
                            let k = i * ln + l * n + j;
                            o[k] = nb.z[k] * (c[j] - C64::real(w) - kappa) - nb.vt[k];
                        }
                    }
                }
                o
            })
            .collect();
        look.offset = vec![C64::ZERO; n];
        for nb in ws.near.iter().filter(|nb| !nb.idle() && !nb.xs.is_empty() && nb.nd > 0) {
            for (j, o) in look.offset.iter_mut().enumerate() {
                *o += (0..nb.nd).map(|d| nb.exd[j * nb.nd + d] * nb.zdtot[d * n + j]).sum::<C64>();
            }
        }
        look.work += (ws.near.iter().map(|nb| nb.s * nb.lanes).sum::<usize>() * n) as f64;
        let m = look.samples;
        look.seen.clear();
        for k in 0..m {
            let t = cell * k as f64 / (m - 1) as f64;
            let seen = self.seen_at(t, path(t), true, cache, ws, look);
            look.seen.push(seen);
        }
        // The identity at the decision: the forward's last sweep's diagonal, H_jj = a_j + (Ê_D Z_D)_jj + Σ_j(c_j).
        // Each row at the column energies of the sweep that solved it last (its part's sweeps, `Near::upto`).
        look.identity = (0..n)
            .map(|j| {
                let (mut sigma, mut size) = (C64::ZERO, 0.0);
                for (nb, inp) in ws.near.iter().zip(&look.seen[0].near).filter(|(nb, _)| !nb.idle()) {
                    let ln = nb.lanes * n;
                    for (i, &kappa) in inp.kappa.iter().enumerate() {
                        let c = solved(nb, &ws.cs, i, n)[j];
                        for (l, &w) in inp.shifts.iter().enumerate() {
                            let q = inp.num[i * ln + l * n + j] / (c - C64::real(w) - kappa);
                            sigma += q;
                            size += q.abs();
                        }
                    }
                }
                let h = ws.h[j * n + j];
                (look.seen[0].bare[j] + look.offset[j] + sigma - h).abs() / (h.abs() + size)
            })
            .fold(0.0, f64::max);
        // Every column's pass 0 at the samples, in time, from the decision's own column energy on; the critical points
        // (# The critical points), placed exactly, each solved from the point before it.
        let mut from: Vec<C64> = (0..n).map(|j| ws.h[j * n + j]).collect();
        solve(&mut look.seen, &look.offset, &mut from, ws, n, &mut look.work);
        // The excitation's slope at every sample, by the fold one forward difference on: ε = √u h, the step of the least
        // error of a difference quotient (its truncation ε x″/2 against its rounding u x/ε, x varying on the samples'
        // scale h).
        // A cell of no width (a decision's point alone, `Fold::worst`): its samples the point's, nothing between them.
        if !(cell > 0.0) {
            look.seen.truncate(1);
            look.seen[0].chirp = vec![0.0; n];
            return;
        }
        let epsilon = crate::units::UNIT_ROUNDOFF.sqrt() * cell / (m - 1) as f64;
        for k in 0..m {
            let t = look.seen[k].t + epsilon;
            let mut ahead = vec![self.seen_at(t, path(t), false, cache, ws, look)];
            let mut from: Vec<C64> = look.seen[k].columns.iter().map(|x| x.c).collect();
            solve(&mut ahead, &look.offset, &mut from, ws, n, &mut look.work);
            look.seen[k].slope = (0..n).map(|j| (ahead[0].columns[j].x - look.seen[k].columns[j].x) / epsilon).collect();
        }
        let times = critical(look, ws, n);
        for t in times {
            let seen = self.seen_at(t, path(t), false, cache, ws, look);
            look.seen.push(seen);
        }
        look.seen.sort_by(|a, b| a.t.total_cmp(&b.t));
        let Look { seen, offset, work, .. } = &mut *look;
        let placed: Vec<usize> = (1..seen.len()).filter(|&k| !seen[k].sample).collect();
        for k in placed {
            let (before, rest) = seen.split_at_mut(k);
            let mut from: Vec<C64> = before[k - 1].columns.iter().map(|x| x.c).collect();
            solve(&mut rest[..1], offset, &mut from, ws, n, work);
        }
        chirp(look, ws, n);
    }

    /// A point's bare inputs (# The critical points): the light placed at (r, κ) (`proto_place`, the forward's), per near
    /// block its rows' energies and sources (`moved`, `sourced`) with the held source O (# What is held), per far block
    /// its held forms' diagonal.
    fn seen_at(&self, t: f64, (r, kappa): ([f64; 3], [f64; 3]), sample: bool, cache: &Cache, ws: &Workspace, look: &mut Look) -> Seen {
        let n = ws.n;
        let mut work = 0.0;
        self.proto_place(r, kappa, &mut look.at, &ws.pairs, &ws.pp, &ws.base, &ws.lin, &ws.rotating, &ws.proj, &mut work);
        let Placed { coef, f, w, e, hpp, .. } = &look.at;
        let mut seen = Seen { t, sample, bare: (0..n).map(|j| hpp[j * n + j]).collect(), far_m: vec![0.0; n], ..Seen::default() };
        for (x, (nb, tp)) in ws.near.iter().zip(look.temp.iter_mut()).enumerate() {
            let mut inputs = Inputs::default();
            if nb.idle() {
                seen.near.push(inputs);
                continue;
            }
            let basis = cache.near_at(x);
            let (s, lanes, nd) = (nb.s, nb.lanes, nb.nd);
            let ln = lanes * n;
            inputs.kappa = vec![C64::ZERO; s];
            inputs.shifts = vec![0.0; lanes];
            inputs.vt = vec![C64::ZERO; s * ln];
            let Temps { delta, vtot, stat, emit, hx, b0d, exd } = tp;
            delta.resize(nb.terms.len(), C64::ZERO);
            for (v, len) in [(&mut *vtot, s * n), (&mut *stat, s * n), (&mut *emit, s * n), (&mut *hx, n * n), (&mut *b0d, nd * n), (&mut *exd, n * nd)] {
                v.clear();
                v.resize(len, C64::ZERO);
            }
            moved(basis, nb.value, &nb.terms, coef, &mut inputs.kappa, delta, &mut work);
            sourced(&nb.splits, &nb.vectors, &nb.xs, &nb.exact, lanes, nd, basis, f, w, e, delta, &mut inputs.shifts, &mut inputs.vt, vtot, stat, emit, hx, b0d, exd, s, n, &mut work);
            for (v, o) in inputs.vt.iter_mut().zip(&look.held[x]) {
                *v += *o;
            }
            // The numerators and the intensities; the explicit states' dressing on their columns' bare energy.
            inputs.num = vec![C64::ZERO; s * ln];
            inputs.intensity = vec![0.0; s * n];
            for i in 0..s {
                for j in 0..n {
                    let ej = emit[i * n + j];
                    inputs.intensity[i * n + j] = ej.norm_sqr();
                    for l in 0..lanes {
                        inputs.num[i * ln + l * n + j] = ej.conj() * inputs.vt[i * ln + l * n + j];
                    }
                }
            }
            work += (s * ln) as f64;
            if !nb.xs.is_empty() {
                for j in 0..n {
                    seen.bare[j] += hx[j * n + j];
                }
            }
            // The reuse's drift and the centred copies' forms' change.
            let parts = basis.f.parts.len().max(1);
            for (t, d) in delta.iter().enumerate().filter(|(_, d)| **d != C64::ZERO) {
                let off = basis.norms[t * s * parts..(t + 1) * s * parts].iter().map(|x| x * x).sum::<f64>().sqrt();
                inputs.drift += d.abs() * off;
                inputs.held += d.abs() * basis.copied[t];
            }
            work += (nb.terms.len() * s * parts) as f64 / 4.0;
            seen.near.push(inputs);
        }
        for (x, fb) in ws.far.iter().enumerate() {
            if fb.omitted {
                seen.far.push((0.0, vec![]));
                continue;
            }
            let basis = cache.far_at(x);
            let v = fb.vectors.len();
            for a in 0..v {
                for b in 0..v {
                    let (ca, ka) = fb.vectors[a];
                    let (cb, kb) = fb.vectors[b];
                    let p = e[ca][ka].conj() * e[cb][kb];
                    let at = (a * v + b) * n * n;
                    for j in 0..n {
                        seen.bare[j] += p * basis.f[at + j * n + j];
                        seen.far_m[j] += (p * basis.g[at + j * n + j]).re;
                    }
                }
            }
            let held = fb.terms.iter().zip(&basis.at).zip(&basis.copied).map(|((&p, a), c)| (coef[p] - *a).abs() * c).sum::<f64>();
            let shifts: Vec<f64> = fb.splits.iter().map(|&(cc, j)| w[cc][j]).chain([0.0]).collect();
            seen.far.push((held, shifts));
            work += (2 * v * v * n + fb.terms.len()) as f64;
        }
        look.work += work;
        seen
    }
}

/// Every column's pass 0 at the points `seen`, in their order, each from the root before it (`from`: a column whose fold
/// does not exist at a point keeps the last root it had).
fn solve(seen: &mut [Seen], offset: &[C64], from: &mut [C64], ws: &Workspace, n: usize, work: &mut f64) {
    for s in seen.iter_mut() {
        s.columns.resize(n, Column::default());
        for j in 0..n {
            let a = s.bare[j] + offset[j];
            let column = pass0(s, ws, j, a, from[j], n, work);
            if column.exists {
                from[j] = column.c;
            }
            s.columns[j] = column;
        }
    }
}

/// Column j's Σ_j(c) = Σ_b Σ_i Σ_l N_lij/(c − w_l − κ_i) at a point, its derivative's opposite Σ N/d², the size of its
/// terms Σ |N/d| and their count.
fn sum(s: &Seen, ws: &Workspace, j: usize, c: C64, n: usize, work: &mut f64) -> (C64, C64, f64, usize) {
    let (mut sigma, mut slope, mut size, mut count) = (C64::ZERO, C64::ZERO, 0.0, 0);
    for (nb, inp) in ws.near.iter().zip(&s.near).filter(|(nb, _)| !nb.idle()) {
        let ln = nb.lanes * n;
        for (i, &kappa) in inp.kappa.iter().enumerate() {
            for (l, &w) in inp.shifts.iter().enumerate() {
                let num = inp.num[i * ln + l * n + j];
                if num == C64::ZERO {
                    continue;
                }
                let inv = (c - C64::real(w) - kappa).recip();
                let q = num * inv;
                sigma += q;
                slope += q * inv;
                size += q.abs();
                count += 1;
            }
        }
    }
    *work += count as f64;
    (sigma, slope, size, count)
}

/// Column j's excitation at a point at the energy c, Σ_b Σ_i |Σ_l V_lij/(c − w_l − κ_i)|² with the far blocks' held M_jj.
fn response(s: &Seen, ws: &Workspace, j: usize, c: C64, n: usize) -> f64 {
    let mut x = s.far_m[j];
    for (nb, inp) in ws.near.iter().zip(&s.near).filter(|(nb, _)| !nb.idle()) {
        let ln = nb.lanes * n;
        for (i, &kappa) in inp.kappa.iter().enumerate() {
            let mut z = C64::ZERO;
            for (l, &w) in inp.shifts.iter().enumerate() {
                let v = inp.vt[i * ln + l * n + j];
                if v == C64::ZERO {
                    continue;
                }
                z += v / (c - C64::real(w) - kappa);
            }
            x += z.norm_sqr();
        }
    }
    x
}

/// Column j's pass 0 at a point (# The column's pass 0): Newton on c − a − Σ(c) from `start`; the blocks' parts and
/// nearest distances into the point's inputs.
fn pass0(s: &mut Seen, ws: &Workspace, j: usize, a: C64, start: C64, n: usize, work: &mut f64) -> Column {
    // Newton from c: its root, whether it converged -- a step within what the residual's rounding alone makes, γ of its
    // terms times their sizes over the slope --, and its ratio.
    let newton = |mut c: C64, work: &mut f64| -> (C64, bool, f64) {
        for _ in 0..ROUNDS {
            let (sigma, slope, size, count) = sum(s, ws, j, c, n, work);
            let step = (c - a - sigma) / (C64::ONE + slope);
            if !step.is_finite() {
                return (c, false, f64::INFINITY);
            }
            c -= step;
            if step.abs() * (C64::ONE + slope).abs() <= crate::units::rounding(count + 2) * (c.abs() + a.abs() + size) {
                let (_, slope, ..) = sum(s, ws, j, c, n, work);
                return (c, true, slope.abs());
            }
        }
        (c, false, f64::INFINITY)
    };
    // From the point before; where that root is not the fold's (no ratio below one: past an exceptional point the
    // continuation is the excited-like branch), from the series' own start, the bare energy -- the fold at a point is the
    // root its series reaches from there.
    let (mut c, mut converged, mut r) = newton(start, work);
    if !(converged && r < 1.0) {
        let again = newton(a, work);
        if again.1 && again.2 < 1.0 {
            (c, converged, r) = again;
        }
    }
    let x = response(s, ws, j, c, n);
    // Per near block its part of the excitation and its nearest distance.
    for (nb, inp) in ws.near.iter().zip(s.near.iter_mut()) {
        if inp.part.len() != n {
            inp.part = vec![0.0; n];
            inp.nearest = vec![f64::INFINITY; n];
        }
        if nb.idle() {
            continue;
        }
        let ln = nb.lanes * n;
        let (mut part, mut nearest) = (0.0, f64::INFINITY);
        for (i, &kappa) in inp.kappa.iter().enumerate() {
            let mut z = C64::ZERO;
            for (l, &w) in inp.shifts.iter().enumerate() {
                let v = inp.vt[i * ln + l * n + j];
                if v == C64::ZERO {
                    continue;
                }
                let d = c - C64::real(w) - kappa;
                z += v / d;
                nearest = nearest.min(d.abs());
            }
            part += z.norm_sqr();
        }
        inp.part[j] = part;
        inp.nearest[j] = nearest;
    }
    let exists = converged && r < 1.0;
    Column { c, x, r, exists }
}

/// The critical points (# The critical points): per panel and column, of the bare inputs' quadratics' zeros and extrema
/// the one whose own term reads largest, where it is larger than at the panel's samples; their times.
fn critical(look: &Look, ws: &Workspace, n: usize) -> Vec<f64> {
    let m = look.seen.len();
    let h = look.cell / (m - 1) as f64;
    let mut times: Vec<f64> = vec![];
    let mut us: Vec<f64> = vec![];
    let mut lanes: Vec<(Quadratic, Quadratic)> = vec![];
    for p in 0..(m - 1) / 2 {
        let panel = [&look.seen[2 * p], &look.seen[2 * p + 1], &look.seen[2 * p + 2]];
        for j in 0..n {
            let a = |s: &Seen| s.bare[j] + look.offset[j];
            let mut best: Option<(f64, f64)> = None;
            for (b, nb) in ws.near.iter().enumerate().filter(|(_, nb)| !nb.idle()) {
                let ins = panel.map(|s| &s.near[b]);
                let ln = nb.lanes * n;
                for i in 0..nb.s {
                    let mut row = [C64::ZERO; 3];
                    lanes.clear();
                    for l in 0..nb.lanes {
                        let k = i * ln + l * n + j;
                        let nums = ins.map(|x| x.num[k]);
                        if nums.iter().all(|x| *x == C64::ZERO) {
                            continue;
                        }
                        let d: [C64; 3] = std::array::from_fn(|q| a(panel[q]) - C64::real(ins[q].shifts[l]) - ins[q].kappa[i]);
                        let (qd, qn) = (Quadratic::of(d[0], d[1], d[2]), Quadratic::of(nums[0], nums[1], nums[2]));
                        let own = |u: f64| (qn.at(u) / (qd.at(u) * qd.at(u))).abs();
                        let edge = [-1.0, 0.0, 1.0].map(own).into_iter().fold(0.0, f64::max);
                        us.clear();
                        qd.critical(&mut us);
                        for &u in &us {
                            let v = own(u);
                            if v > edge && best.is_none_or(|x| v > x.1) {
                                best = Some((u, v));
                            }
                        }
                        for (q, r) in row.iter_mut().enumerate() {
                            *r += nums[q] / (d[q] * d[q]);
                        }
                        lanes.push((qd, qn));
                    }
                    if lanes.is_empty() {
                        continue;
                    }
                    // The row's intensity's extremum, its lanes' terms summed there.
                    let y = ins.map(|x| C64::real(x.intensity[i * n + j]));
                    let qi = Quadratic::of(y[0], y[1], y[2]);
                    if qi.c.re != 0.0 {
                        let u = -qi.b.re / (2.0 * qi.c.re);
                        if u.abs() < 1.0 {
                            let v = lanes.iter().map(|(qd, qn)| qn.at(u) / (qd.at(u) * qd.at(u))).sum::<C64>().abs();
                            let edge = row.iter().map(|r| r.abs()).fold(0.0, f64::max);
                            if v > edge && best.is_none_or(|x| v > x.1) {
                                best = Some((u, v));
                            }
                        }
                    }
                }
            }
            if let Some((u, _)) = best {
                times.push(panel[1].t + u * h);
            }
            // The solved excitation's own largest between the samples, by its values and slopes there (quintic Hermite):
            // where the fold's root turns fast, the inputs' critical points are not the excitation's.
            if panel.iter().all(|s| s.columns[j].exists) {
                let (x, d) = (panel.map(|s| s.columns[j].x), panel.map(|s| s.slope[j] * h));
                if let Some(u) = hermite_peak(x, d) {
                    times.push(panel[1].t + u * h);
                }
            }
        }
    }
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

/// The interior largest of the quintic through (−1, 0, 1) with values x and slopes d there, where it is above the three
/// values: among the real roots of its derivative (a quartic, by its companion's eigenvalues) within the panel.
fn hermite_peak(x: [f64; 3], d: [f64; 3]) -> Option<f64> {
    let (even, odd) = (0.5 * (x[2] + x[0]) - x[1], 0.5 * (x[2] - x[0]) - d[1]);
    let (c4, c5) = (0.5 * (0.5 * (d[2] - d[0]) - 2.0 * even), 0.5 * (0.5 * (d[2] + d[0]) - d[1] - 3.0 * odd));
    let c = [x[1], d[1], even - c4, odd - c5, c4, c5];
    let q = |u: f64| c.iter().rev().fold(0.0, |a, &k| a * u + k);
    // q′ = Σ_k k c_k u^(k−1), its leading coefficient the last one not zero.
    let slope: Vec<f64> = (1..6).map(|k| k as f64 * c[k]).collect();
    let degree = slope.iter().rposition(|&k| k != 0.0)?;
    if degree == 0 {
        return None;
    }
    let companion = Mat::from_fn(degree, degree, |i, j| if j == degree - 1 { C64::real(-slope[i] / slope[degree]) } else if i == j + 1 { C64::ONE } else { C64::ZERO });
    let mut eigen = crate::linalg::Eigen::default();
    if !eigen.decompose(&companion) {
        return None;
    }
    let roots = &eigen.values;
    let top = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    roots
        .iter()
        .filter(|u| u.im.abs() <= crate::units::rounding(degree) * u.abs().max(1.0) && u.re.abs() < 1.0)
        .map(|u| u.re)
        .filter(|&u| q(u) > top)
        .max_by(|a, b| q(*a).total_cmp(&q(*b)))
}

/// The chirp's reading (M7, unify2 §2.2) at every point: ‖Σ_l z_lij ḋ_lij/d_lij‖ over the rows relative to the column's
/// response, ḋ the slope of the distance d = c_j − w_l − κ_i along its panel's quadratic through the samples; infinite
/// where a sample of the panel has no fold.
fn chirp(look: &mut Look, ws: &Workspace, n: usize) {
    let samples: Vec<usize> = (0..look.seen.len()).filter(|&k| look.seen[k].sample).collect();
    let panels = (samples.len() - 1) / 2;
    let h = look.cell / (samples.len() - 1) as f64;
    let mut out: Vec<Vec<f64>> = vec![];
    for s in &look.seen {
        let p = ((s.t / (2.0 * h)).floor() as usize).min(panels - 1);
        let panel = [&look.seen[samples[2 * p]], &look.seen[samples[2 * p + 1]], &look.seen[samples[2 * p + 2]]];
        let u = (s.t - panel[1].t) / h;
        let mut row = vec![0.0; n];
        for (j, x) in row.iter_mut().enumerate() {
            if !panel.iter().all(|q| q.columns[j].exists) || !s.columns[j].exists {
                *x = f64::INFINITY;
                continue;
            }
            let c = s.columns[j].c;
            let (mut err, mut whole) = (0.0, 0.0);
            for (b, nb) in ws.near.iter().enumerate().filter(|(_, nb)| !nb.idle()) {
                let ln = nb.lanes * n;
                for i in 0..nb.s {
                    let (mut e, mut z) = (C64::ZERO, C64::ZERO);
                    for l in 0..nb.lanes {
                        let v = s.near[b].vt[i * ln + l * n + j];
                        if v == C64::ZERO {
                            continue;
                        }
                        let d = c - C64::real(s.near[b].shifts[l]) - s.near[b].kappa[i];
                        let ds = panel.map(|q| q.columns[j].c - C64::real(q.near[b].shifts[l]) - q.near[b].kappa[i]);
                        let slope = Quadratic::of(ds[0], ds[1], ds[2]).slope(u) / h;
                        let zl = v / d;
                        z += zl;
                        e += zl * slope / d;
                    }
                    err += e.norm_sqr();
                    whole += z.norm_sqr();
                }
            }
            *x = if whole > 0.0 { (err / whole).sqrt() } else { 0.0 };
        }
        out.push(row);
    }
    for (s, row) in look.seen.iter_mut().zip(out) {
        s.chirp = row;
    }
}

#[cfg(test)]
impl Workspace {
    /// The last forward's column excitations (its `whole`, the explicit columns' unit taken out) and Bloch generator's
    /// diagonal: the tests' truth of the look.
    pub(crate) fn columns(&self, ground: usize) -> Vec<(C64, f64)> {
        let n = self.n;
        (0..n).map(|j| (self.h[j * n + j], self.whole[j] - if j >= ground { 1.0 } else { 0.0 })).collect()
    }
}
