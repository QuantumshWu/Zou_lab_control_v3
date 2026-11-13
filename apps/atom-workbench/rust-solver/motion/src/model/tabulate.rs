//! The near light's tables (SPEC §2.3): per cluster and light set the elimination's nonlinear part -- the cluster's near forms
//! (assemble.rs `NearLayout`: the shifts S and their Doppler partners, the jumps, the explicit states' van Vleck shifts, the
//! responses) -- tabulated over the coordinates it depends on, the linear part (the field's components, the Doppler shifts) exact.
//! The light sets the integrator evaluates: every light on (the generator, the photons, the readout), and every light but the
//! trap's (V_run's subtraction, potential.rs).
//!
//! # Coordinates
//! The elimination at a point depends on the point through the levels' H (eliminate.rs `Model::far_hamiltonian`) and through
//! the carriers' dressings and self-energies. A frequency that is no carrier enters H by its coherence T_ω, a function of its
//! chart (field.rs # The chart: intensity, polarization angles, relative phases). A carrier enters both by its dressing: with
//! its units' T̄ (eliminate.rs) on each polarization unit's intensity I_u = |f_u|²/N_u, f_u = Σ_{l∈u} a_l κ_l the unit's field
//! along its polarization ε_u (E_l = κ_l ε_u), N_u = (Σ_{l∈u} |κ_l|)² so that I_u ∈ [0, 1] (|a_l| ≤ 1: a computed I_u above 1 is
//! its rounding, and is 1) -- its frequency's static form in H too, Σ_u I_u N_u F(ε_u) --; with its coherent dressing
//! (`Unit::Dressing` kept) on its chart. Every coordinate on its whole range: every value is some field's, no atom meets an
//! edge, and a table is never asked off its axes (crate::table refuses it).
//!
//! # A node
//! At coordinates s the elimination is the one in a light whose fields have them: each lit frequency's field from its chart
//! (field.rs `Frequency::chart_field`), its beams' amplitudes the least-norm ones that give it, a_l = Σ_k γ_lk* e_k/σ_k² (the
//! static forms depend on the field alone, far.rs); each unit's field on its first beam, a = √(I_u N_u)/κ₁. The coordinates that
//! are no axes take their values at the focus.
//!
//! # A node's reals
//! The near forms, their real and imaginary parts in the layout's order: what the photons, the readout, the potential and the
//! refinement read. And after them the middle's own (assemble.rs, # Dense terms), where the generator's dense terms and the
//! forms' take them: the Hermitian forms (`Plan::lay`) and the decay's sums of the node's jumps' products (`Plan::decay`). A
//! middle takes that part in one pass over the stencil's rows, each real straight to its place (`Model::near_point`), and
//! multiplies no jumps. The Hermitian part is linear in the forms: its interpolant is the interpolated forms' own layout,
//! number for number. The decay is quadratic in the jumps: its interpolant is not the product of the interpolated jumps --
//! which the photons take (photon.rs) -- but differs from it by the two interpolations' errors, both third order in the
//! spacing and both in the table's error (below). And last the photons' own: the near jumps as the matrices a state's
//! photons sum (assemble.rs, # The jumps' matrices; `Plan::lay_jumps`), linear in the jumps -- their interpolant, and
//! its slopes along the axes, are the interpolated jumps' own, number for number, each real in its place for the
//! photons' sums (`Model::near_jumps`).
//!
//! # Axes
//! A coordinate's range over the atoms' reach is its focus value ± the proven bound of its distance from it there, to the
//! least first spacing of the coordinates (reach.rs `Model::extents_over`; `Model::coordinate_moved`): an intensity's from
//! its beam pairs' interference (a plane wave's own intensity, the same everywhere, does not move); an angle's from the
//! bounds of its basis components' squared moduli, which it rises and falls with; a phase's from the box of its two
//! components' product, any phase where the box holds zero. A coordinate that does not move over the reach is its focus
//! value at every atom: it is no axis, and leaves nothing out. (An angle's or a phase's range was its whole range: S3's
//! cooling pair -- two plane waves of equal amplitudes and orthogonal fields -- has its polarization angle at π/4 at every
//! atom, and with its dressing coherent the angle was an axis by its weight over [0, π/2] while it moved nothing: the forms
//! 4.0 → 11.9 µs a point, for 1e-18 of the responses.)
//!
//! A coordinate's rate is how many pole half-widths a unit of it moves the cluster's lines: the largest change of E_e − E_b
//! (e ← b a carrier's coupling from the cluster's ground states, Γ_e/2 its half-width) over its first grid's nodes, the other
//! coordinates at the focus -- per unit of an intensity, whose shift is linear in it; an angle's or a phase's the amplitude
//! of a trigonometric polynomial of degree two or one in it (the field's components are of degree one in each, the levels
//! quadratic in the field), whose slope is at most that degree times the amplitude (Bernstein) --; for a unit, its
//! dressing at full intensity over Γ_min/2 (its resolvent's Varah bound, eliminate.rs) and the same change of E_e − E_b by
//! its static form at full intensity. Its weight is its rate times the range it moves across over the reach. The rate sees
//! the static forms and a unit's dressing, not a coherent dressing's or the self-energy's part of the lines (eliminate.rs):
//! it orders the coordinates and scales their poles (# Refinement); it decides nothing alone.
//!
//! What a table leaves out is held to one criterion, the tables' shares (# Refinement), whether it lies between an axis' nodes
//! or in a coordinate that is no axis: twice the responses' error within the shares, the potential's within T's of k_BT_r.
//! Every coordinate that moves over the reach is asked whether the table needs it, by a look (`Model::look`): the table that
//! holds it at the focus against the elimination with the coordinate moved, at the table's nodes over the reach, by the
//! refinement's own two measures and shares. (A weight within half the shares held a coordinate at the focus unasked, as a
//! bound -- but a coherent dressing moves the lines through the self-energy and the other carriers' z as well: S3's cooling
//! pair's relative phase weighed 4.6e-4 by its static form and moved the repump's responses by 0.19 without the trap's light,
//! 8e-3 with it; held, the table's error at the atoms was 2.1 times its share, and the solve's reads halved an axis that was
//! not its cause up to the work bound, # Refinement.) Twice:
//! - before it is made an axis, the heaviest coordinate first: the table of the axes taken so far, each on its first grid,
//!   against the coordinate where the atoms take it: the ends of its range over the reach and its first nodes between
//!   them (a node beyond the range read an intensity that moved 2e-3 over the reach at 1/6 of its range away, and took it
//!   as an axis for 1e-6 of the responses). Beyond the shares it is taken as an axis -- as
//!   many as keep one interpolation cheaper than the elimination it replaces and the first grid within the work bound --,
//!   within them it is held: a coordinate that moves nothing costs these few nodes and never multiplies the table's;
//! - after the refinement: every coordinate so held, at the one value its first look read worst, against every node of the
//!   refined table over the reach. Beyond the shares there it is an axis after all, and the table is made again with it.
//! What the looks say, and what they do not. The first sees the others' first nodes alone: a response's change by a shift δ
//! of its line is δ/|D|, largest where an axis carries the line through resonance, and between that axis' first nodes the
//! coordinate's part is unseen. The last is at every node of the table kept -- where the refinement itself reads -- but at
//! one value of the coordinate: for one that shifts the lines in proportion to its move (every intensity) the error
//! |δ|/|D + δ| rises with |δ| as long as it is below one, so the value farthest over the reach, the first look's worst, is
//! the worst at every node; of an angle or a phase the first look's order need not hold at the other nodes. Neither looks
//! between the nodes, as the refinement does not. What the coordinates held leave is in the table's whole error, read at the
//! atoms (`Model::table_reads`): that read is the measure, the looks decide the axes. None where no coordinate is needed:
//! the forms at the focus alone. (With the cut at the population target, as it was, a coordinate could leave the responses
//! off by 0.05 where an axis was refined to 0.004: a probe on resonance in a tweezer that moves its line by 0.04 half-widths
//! over its whole intensity read 2.1e-2 at the edge of a reach of 120 µK and 8.2e-3 on average across it, beyond the share
//! 7.8e-3, with no axis to halve; the test below.)
//!
//! # At a point
//! The axes' coordinates and their gradients at the point (field.rs `Frequency::chart`; ∇I_u = 2 Re(f_u* ∇f_u)/N_u), the table's
//! interpolant of the forms asked and, to first order, its derivatives along the axes, ∇v = Σ_f ∂v/∂s_f ∇s_f.
//!
//! # The table's error
//! A table's values are ṽ(s) = v(s) + e(s), the elimination's and the interpolation's error, C¹ in the coordinates. The
//! generator they give has the force of its own Hermitian part (the interpolant's own derivative), and its decay is the
//! interpolant of the jumps' product: the elimination's decay to the same order as every other value, and a Lindblad one with
//! the interpolated jumps to that order -- at a node exactly. So the table stands for the model whose generator is the
//! interpolated one, and its error on the outputs is that generator's distance from the elimination's (both reads below take
//! the generator the steps take, the decay's interpolated sums in it): the rates and the couplings of a state moved by the relative δ =
//! ‖(Ñ − N)ψ‖/‖Nψ‖ (N the near light's part of the generator at the atom, its Doppler partners at the atom's momentum) -- the
//! populations by at most δ P_g, and T, which the near light sets through its rates, its friction and the depth of its shifts
//! (all of them N's), by the relative δ --; and the atom's potential moved by ΔV = ⟨ψ|H̃_h − H_h|ψ⟩, a conservative change that
//! moves a kinetic energy by at most itself: T by at most |ΔV|/k_BT_r relative (T_r the reach's temperature). The force's own
//! relative error does not enter beyond these: it is the gradient of the small ΔV, whatever its slope. `Model::table_reads`
//! measures δ and ΔV at trajectory states against the exact elimination (the ledger's form: the populations absolute, T relative).
//!
//! # Refinement
//! A response is a pole in its line's detuning, and an axis that moves the line across it within the atoms' reach is the generic
//! case of near-resonant light in a trap; the first grid resolves nothing of it. A table is therefore refined before it is used,
//! by its own measured error between its nodes. Halving an axis' spacing (crate::table `Axis::halved`) puts the new nodes at
//! the middles of its intervals: there the table before the halving is compared with the elimination itself -- the new nodes'
//! own values, so the check costs nothing beyond the finer table. The middle is not where the interpolant is worst: its
//! leading error (h³f‴/6) u(1 − u)(1 − 2u) vanishes there and is largest at u ≈ 0.21 and 0.79 (crate::table, # The
//! interpolant), so the check reads the next order, h⁴f⁗, and is no bound of the table before the halving; what the table
//! kept errs by is read at the atoms (`Model::table_reads`). The comparison is made over every interval the reach meets, the
//! axis's range over the reach (# Axes).
//! Two measures, per node the largest: the responses' relative error, max over the lit carriers of (‖ΔT‖ + w‖ΔT2‖)/‖T‖ (Frobenius
//! norms over the carrier's basis vectors and the cluster's ground places; w the largest Doppler shift of the carrier's beams at
//! the reach's speed), of which the rates take twice (they are quadratic in the responses); and the bound of the potential's
//! error, the bound of ‖H̃_h − H_h‖ by its rows: Σ over one carrier's shifts and couplings S_k'k[x, y] of |ΔS| (|ΔD| w with it,
//! for either factor) times the two field components' peaks Σ_l |γ_lk| (every beam at its peak and in phase), the largest over the
//! ground places x. An axis is halved until the table before the halving is within the tables' shares -- twice the
//! responses' error within the populations' and T's, the potential's within T's of k_BT_r --, and the finer table is kept:
//! its own error is then below the verified one by the interpolation's orders. An error that is no number (a node on a pole
//! of a real denominator, outside the reach but in a stencil that serves it) is unread, +∞: never within a share.
//!
//! How far: the function's structure and the shares, never the run. A pole a distance ρ = 1/λ from the real axis of the
//! coordinate (λ its rate, # Axes: the line moves a half-width in 1/λ) is unresolved while the spacing h is above ρ, and the
//! read is the peak the grid misses: it falls as h, by 2 a halving, or by less or not at all as the pole's place among the
//! nodes turns. From h ≈ ρ on the read falls faster, 3 to 6 a halving across the transition, and then at the interpolant's
//! order -- inside an interval the middle errs by 7h⁴f⁗/384 (the slopes' (h²/6)f‴ and the cubic's own fourth order), 16 a
//! halving; at a bounded axis's end intervals by h³f‴/16 (the one-sided slope), 8. (1/(x + i) on a uniform grid, the read's
//! fall a halving at h/ρ = 8, 2, 1, 1/2, 1/4, 1/8: 2.0, 2.4, 3.4, 6.1, 10.5, 14.0; S3's tweezer intensity, its lines moved
//! 17.4 half-widths a unit by its rate: 2.2, 3.2, 5.0, 6.8 from h = 1/6 to 1/48; R4_LIN_2's, 10.8 a unit: 2.7, 4.0, 5.8 --
//! the rate, of the static forms alone, puts the pole about twice too far.) So an axis is halved while its read is beyond
//! the shares and either its grid is unresolved by its rate (λh > 1) or the read fell by more than the factor 2 of a peak
//! the grid misses (`halving_on`): an error being resolved goes on, at whatever pace the transition takes; one that falls
//! by 2 or less on a resolved grid -- a kink, an unread node, a read of no order -- ends the axis there, and what it leaves
//! stands as `table_reads` reads it at the atoms. Past its rate's resolution every halving at least halves the read, so an
//! axis takes at most ⌈log₂ λh₀⌉⁺ halvings and then ⌈log₂ r⌉ (r its read there over the shares): the nodes are fixed by
//! the lines' rates, the reads and the shares alone -- the work bound (solve.rs: the pilot's middles) is only a ceiling no
//! grid passes, met where a table of many axes needs a verifying halving of each (every one doubles the grid: R4_LIN_4
//! with both carriers' dressings coherent asks 193 x 13 x 13 x 16 nodes of 1760 reals, 7 GB, of one table; at the ceiling
//! its error is read at the atoms and booked). (The order's own factor, 2^VALUE_ORDER a halving, holds only past the
//! transition: asked of every halving, or of the halvings a read's order predicts, it ended R4_LIN_2's tweezer axis at
//! 97 nodes with its read 1.44 times its share, where 193 read it within.) The solve's work bound decides which
//! coordinates are axes (their first grid, `first_table`); of how far they are refined it is the ceiling alone. (Refined until the
//! next grid passed the work bound -- the pilot's middles, pilot_size (end/τ + end/2τ) --, a table's nodes grew with the
//! run's length over its step: S3 with its cooling dressing coherent reached that bound through the solve's reads,
//! 1.7 million nodes at τ 0.079 µs, 8.9 GB; S3c at 100 ms would have reached 22 million.)
//!
//! The solve's reads at the atoms (solve.rs) refine by the same rule (`Model::halved_by`), each read there of the table's
//! own grid against the read of the grid before -- and only while the axes' reads can explain what is beyond: the table's
//! whole error less every axis's own read beyond the shares is a coordinate held at the focus, which no halving lowers.
//!
//! The tables' shares of the targets are the resolution of the largest ensemble the solve runs (SPEC §3, N = 2¹⁴ trajectories at
//! the confidence z): an absolute z/(2√N) on what is shown (a share p's standard error is at most 1/(2√N)), a relative z√(2/N) on
//! T (a variance's, normal theory). An error within them is below what any run can read, so no run is limited by its tables.
//!
//! # Work
//! A point costs the table's interpolation of the reals asked, 4^D (1 + D) each on D axes (4^D for a value alone), and the
//! coordinates (each frequency's field and chart from its beams); the chain rule's 3 D a real is not counted. A node's own
//! cost -- its elimination and its forms -- is counted nowhere: the tables are made once a model.

use super::assemble::{sized, Middle, NearValues, Plan, Site};
use super::reach::Region;
use super::{Lights, Lit, Model, Reach, Sample, UnitError};
use crate::field::{range, Amplitude, Chart};
use crate::linalg::{Mat, C64};
use crate::request::{GROUND, STATES};
use crate::table::{Axis, Table};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

/// A coordinate a cluster's near elimination depends on (module documentation, # Coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coordinate {
    /// A lit frequency's chart coordinate (field.rs # The chart): its intensity, a polarization angle or a relative phase.
    Chart { frequency: usize, index: usize },
    /// A near carrier's polarization unit's intensity I_u (its dressing's T̄).
    Unit { carrier: usize, unit: usize },
}

/// A near carrier's polarization unit (module documentation, # Coordinates): its beams (positions among the carrier's), their
/// κ_l along its polarization, and N_u.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Polarization {
    beams: Vec<usize>,
    kappa: Vec<C64>,
    norm: f64,
}

/// What one cluster's near table for one light set tabulates (module documentation): its coordinates, the axes first, their
/// values at the focus and their rates (# Axes), its carriers' units, and the cluster's plan under the lights (its near forms).
#[derive(Clone, Debug)]
pub struct Layout {
    pub cluster: usize,
    pub lights: Lights,
    pub coordinates: Vec<Coordinate>,
    focus: Vec<f64>,
    rates: Vec<f64>,
    polarizations: Vec<Vec<Polarization>>,
    plan: Arc<Plan>,
}

/// A near table: its layout and the table of its nodes' reals (module documentation, # A node's reals).
#[derive(Clone, Debug)]
pub struct NearTable {
    pub layout: Layout,
    pub table: Table,
}

impl NearTable {
    /// Its axes: the first coordinates.
    pub fn axes(&self) -> usize {
        self.table.axes().len()
    }
}

impl Layout {
    /// A node's reals (module documentation, # A node's reals).
    fn width(&self) -> usize {
        2 * self.plan.near.len() + self.plan.generator_start.len() + self.plan.form_start.len() + self.plan.jumps_start.len()
    }
}

/// The near tables of a model (module documentation), and k_BT_r of the reach they were made for (rad/µs).
#[derive(Clone, Debug)]
pub struct Tables {
    pub tables: Vec<NearTable>,
    pub thermal: f64,
}

/// The tables' shares of the targets (module documentation, # Refinement): on what is shown, absolute, and on T, relative.
pub(crate) fn shares() -> (f64, f64) {
    let (z, n) = (crate::units::CONFIDENCE, crate::units::MOST_TRAJECTORIES as f64);
    (z / (2.0 * n.sqrt()), z * (2.0 / n).sqrt())
}

/// Whether an axis is halved again (module documentation, # Refinement): its read `over` (its error over the shares) of a
/// grid at spacing h is beyond the shares, and the grid does not yet resolve its lines' poles (λh > 1, λ the coordinate's
/// rate) or the read fell by more than an unresolved peak's factor 2 from the read of the grid before, `before` (none
/// before the axis's first). An unread error (+∞) falls by nothing.
pub(crate) fn halving_on(over: f64, spacing: f64, rate: f64, before: Option<f64>) -> bool {
    over > 1.0 && (rate * spacing > 1.0 || before.is_none_or(|read| read > 2.0 * over))
}

/// A table's error reads at samples (`Model::table_reads`): per axis the read of its interpolation error, and the table's whole
/// error against the exact elimination (its coordinates held at the focus moving, its interpolation), each as the error it
/// makes on the outputs through the sample's generator (module documentation, # The table's error: δ in `parameter`, δ P_g on
/// the populations, the larger of δ P_g and |ΔV|/k_BT_r on T); zero on the samples of other clusters.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRead {
    pub axes: Vec<Vec<UnitError>>,
    pub whole: Vec<UnitError>,
}

/// A cluster's first table for a light set (`Model::first_table`): the table, its axes each on its first grid; their
/// ranges over the atoms' reach; and the coordinates its first looks held at the focus, each with the value it was read
/// worst at (module documentation, # Axes).
struct First {
    table: NearTable,
    ranges: Vec<(f64, f64)>,
    held: Vec<(Coordinate, f64)>,
}

/// The counted complex multiply-adds of a table's interpolation of `reals` reals on `axes` axes (module documentation,
/// # Work): 4^D nodes for each value and, to first order, its D slopes.
pub(crate) fn read_work(axes: usize, order: usize, reals: usize) -> f64 {
    (4usize.pow(axes as u32) * (1 + if order >= 1 { axes } else { 0 }) * reals) as f64 / crate::decide::COMPLEX
}

/// A coordinate by what it is of, the same in every model of a request: its frequency, whether it is a carrier's
/// polarization unit, and its index (a chart's, or the unit's).
pub(crate) type Key = (usize, bool, usize);

/// The zero amplitude.
fn dark() -> Amplitude {
    Amplitude { value: C64::ZERO, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] }
}

impl Tables {
    /// The table of a cluster for a light set, where there is one.
    pub(crate) fn of(&self, cluster: usize, lights: &Lights) -> Option<&NearTable> {
        self.tables.iter().find(|t| t.layout.cluster == cluster && t.layout.lights == *lights)
    }
}

impl Model {
    /// A near carrier's polarization units (module documentation, # Coordinates).
    fn polarizations(&self, carrier: usize) -> Vec<Polarization> {
        let cc = &self.carriers[carrier];
        let f = &self.light.frequencies[cc.frequency];
        let mut units = cc.units.clone();
        units.sort();
        units.dedup();
        units
            .iter()
            .map(|&u| {
                let beams: Vec<usize> = (0..cc.beams.len()).filter(|&l| cc.units[l] == u).collect();
                let first = &f.fields[beams[0]];
                let size = first.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                let kappa: Vec<C64> = beams.iter().map(|&l| (0..3).map(|q| first[q].conj() * f.fields[l][q]).sum::<C64>() / size).collect();
                let norm = kappa.iter().map(|k| k.abs()).sum::<f64>().powi(2);
                Polarization { beams, kappa, norm }
            })
            .collect()
    }

    /// Every coordinate an elimination under a light set depends on (module documentation, # Coordinates): the chart of every
    /// lit frequency that enters by its coherent field, and the units of each lit carrier whose dressing takes its units' T̄.
    fn coordinates(&self, lights: &Lights, polarizations: &[Vec<Polarization>]) -> Vec<Coordinate> {
        let mut out = vec![];
        for (frequency, f) in self.light.frequencies.iter().enumerate() {
            if lights.on(frequency) && self.carrier_of(frequency).is_none_or(|d| polarizations[d].is_empty()) {
                out.extend((0..f.chart_dimension()).map(|index| Coordinate::Chart { frequency, index }));
            }
        }
        for (carrier, cc) in self.carriers.iter().enumerate() {
            if lights.on(cc.frequency) {
                out.extend((0..polarizations[carrier].len()).map(|unit| Coordinate::Unit { carrier, unit }));
            }
        }
        out
    }

    /// A coordinate's key (`Key`).
    fn key(&self, coordinate: Coordinate) -> Key {
        match coordinate {
            Coordinate::Chart { frequency, index } => (frequency, false, index),
            Coordinate::Unit { carrier, unit } => (self.carriers[carrier].frequency, true, unit),
        }
    }

    /// A table's axes and every coordinate it was made over (their keys), its axes' number, and the counted work of their
    /// coordinates at a point (`coordinates_work`): what a pricing reads of the tables a run has (solve.rs `Pricing`).
    pub(crate) fn table_axes(&self, table: &NearTable) -> (Vec<Key>, Vec<Key>, (usize, f64)) {
        let (d, layout) = (table.axes(), &table.layout);
        let keys = |c: &[Coordinate]| c.iter().map(|&c| self.key(c)).collect::<Vec<_>>();
        (keys(&layout.coordinates[..d]), keys(&layout.coordinates), (d, self.coordinates_work(&layout.coordinates[..d], &layout.polarizations)))
    }

    /// The axes a cluster's table under the running lights would take (module documentation, # Axes), priced without
    /// making it (solve.rs `Pricing`): their number and the counted work of their coordinates at a point. Of the
    /// coordinates the cluster's elimination depends on, those of the tables a run has that took them as axes (`axes`,
    /// where they are coordinates still), and those new to the cluster -- that no such table looked at (`known`) --
    /// whose weight there (`first_table`: its rate, `rates`, times the range it moves across over the reach, `moves` by
    /// its key, `reach_ranges` on the coordinate's own, cached by the caller) is beyond the tables' shares: a coordinate
    /// that moves the cluster's lines by less than the shares of a half-width over the reach moves the responses by
    /// less than a look reads beyond (`over`: twice the responses' error within the shares). The rate is the levels'
    /// alone (# Axes: a coherent dressing's part of the lines is in the look, which eliminates). At most `MOST_AXES`;
    /// none where the cluster has no near form.
    pub(crate) fn priced_axes(&self, cluster: usize, reach: &Reach, axes: &[Key], known: &[Key], moves: &dyn Fn(Key, &dyn Fn() -> f64) -> f64) -> (usize, f64) {
        let lights = self.running();
        if self.plan(cluster, &lights).near.len() == 0 {
            return (0, 0.0);
        }
        let coherent = self.coherent_dressing();
        let polarizations: Vec<Vec<Polarization>> = (0..self.carriers.len()).map(|d| if coherent[d] { vec![] } else { self.polarizations(d) }).collect();
        let all = self.coordinates(&lights, &polarizations);
        let mut focus = vec![0.0; all.len()];
        self.coordinates_at(&all, &polarizations, &self.light.amplitudes([0.0; 3]), &mut focus, &mut vec![[0.0; 3]; all.len()]);
        let rates = std::cell::OnceCell::new();
        let (shown, temperature) = shares();
        let taken: Vec<Coordinate> = (0..all.len())
            .filter(|&i| {
                let key = self.key(all[i]);
                axes.contains(&key)
                    || (!known.contains(&key) && {
                        let range = moves(key, &|| {
                            let range = self.reach_ranges(&all[i..=i], &polarizations, &focus[i..=i], reach)[0];
                            range.1 - range.0
                        });
                        range > 0.0 && 2.0 * rates.get_or_init(|| self.rates(cluster, &lights, &all, &polarizations, &focus))[i] * range > shown.min(temperature)
                    })
            })
            .map(|i| all[i])
            .take(crate::table::MOST_AXES)
            .collect();
        (taken.len(), self.coordinates_work(&taken, &polarizations))
    }

    /// Coordinates' values at a point and their gradients (module documentation, # At a point) into `values` and `gradients`,
    /// and the complex multiply-adds.
    fn coordinates_at(&self, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>], amplitudes: &[Amplitude], values: &mut [f64], gradients: &mut [[f64; 3]]) -> f64 {
        // The chart of the frequency last read: a frequency's coordinates take one chart.
        let mut held: Option<(usize, Chart)> = None;
        for (i, c) in coordinates.iter().enumerate() {
            match *c {
                Coordinate::Chart { frequency, index } => {
                    let chart = match held {
                        Some((of, chart)) if of == frequency => chart,
                        _ => {
                            let f = &self.light.frequencies[frequency];
                            f.chart(&f.field(amplitudes))
                        }
                    };
                    held = Some((frequency, chart));
                    values[i] = chart.value[index];
                    gradients[i] = chart.gradient[index];
                }
                Coordinate::Unit { carrier, unit } => {
                    let p = &polarizations[carrier][unit];
                    let cc = &self.carriers[carrier];
                    let (mut f, mut df) = (C64::ZERO, [C64::ZERO; 3]);
                    for (&l, &k) in p.beams.iter().zip(&p.kappa) {
                        let a = &amplitudes[cc.beams[l]];
                        f += a.value * k;
                        for i in 0..3 {
                            df[i] += a.gradient[i] * k;
                        }
                    }
                    values[i] = (f.norm_sqr() / p.norm).min(1.0);
                    gradients[i] = std::array::from_fn(|a| 2.0 * (f.conj() * df[a]).re / p.norm);
                }
            }
        }
        self.coordinates_work(coordinates, polarizations)
    }

    /// The counted complex multiply-adds of coordinates' values at a point (`coordinates_at`): a chart's field and its
    /// derivatives from each beam (13 parts of 3 components) and the chart from the components, once a frequency; a
    /// unit's field from its beams with its gradient, and its intensity.
    fn coordinates_work(&self, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>]) -> f64 {
        let mut charts = vec![];
        coordinates
            .iter()
            .map(|c| match *c {
                Coordinate::Chart { frequency, .. } if !charts.contains(&frequency) => {
                    charts.push(frequency);
                    let f = &self.light.frequencies[frequency];
                    (f.beams.len() * 39 + 6 * f.rank() * f.rank()) as f64
                }
                Coordinate::Chart { .. } => 0.0,
                Coordinate::Unit { carrier, unit } => (4 * polarizations[carrier][unit].beams.len() + 4) as f64,
            })
            .sum()
    }

    /// The light at coordinate values x (module documentation, # A node): each lit frequency's beams' amplitudes from its chart,
    /// a carrier's from its units' intensities (a carrier with a coherent dressing: from its chart).
    fn light_at(&self, lights: &Lights, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>], x: &[f64]) -> Vec<Amplitude> {
        let mut near = vec![dark(); self.light.beams.len()];
        for (w, f) in self.light.frequencies.iter().enumerate() {
            if !lights.on(w) {
                continue;
            }
            let s: Vec<f64> = (0..f.chart_dimension())
                .map(|index| coordinates.iter().position(|c| *c == Coordinate::Chart { frequency: w, index }).map_or(0.0, |at| x[at]))
                .collect();
            let e = f.chart_field(&s);
            let components: Vec<C64> = f.basis.iter().map(|u| (0..3).map(|q| u[q].conj() * e[q]).sum()).collect();
            for (j, &l) in f.beams.iter().enumerate() {
                near[l].value = components.iter().enumerate().map(|(k, &e)| f.dual(j, k) * e).sum();
            }
        }
        for (d, cc) in self.carriers.iter().enumerate() {
            if !lights.on(cc.frequency) || polarizations[d].is_empty() {
                continue;
            }
            for &l in &cc.beams {
                near[l] = dark();
            }
            for (u, p) in polarizations[d].iter().enumerate() {
                if let Some(at) = coordinates.iter().position(|c| *c == Coordinate::Unit { carrier: d, unit: u }) {
                    near[cc.beams[p.beams[0]]].value = C64::real((x[at].max(0.0) * p.norm).sqrt()) / p.kappa[0];
                }
            }
        }
        near
    }

    /// The near forms at coordinate values x into a node's reals (module documentation, # A node's reals).
    fn node(&self, layout: &Layout, x: &[f64], out: &mut [f64]) {
        let (lights, plan) = (&layout.lights, &layout.plan);
        let near = self.light_at(lights, &layout.coordinates, &layout.polarizations, x);
        let coherent = self.coherent_dressing();
        let h = self.far_hamiltonian(&near, lights, 0, &coherent);
        let r = self.eliminate(layout.cluster, &h, &near, lights, 0, &coherent, &|_| true);
        let energies = vec![(0..STATES).map(|s| h[0][(s, s)].re).collect::<Vec<f64>>()];
        let mut values = vec![C64::ZERO; plan.near.len()];
        self.forms(plan, &r, &energies, &mut values, &|_| true);
        let (forms, point) = out.split_at_mut(2 * values.len());
        for (pair, z) in forms.chunks_mut(2).zip(&values) {
            pair.copy_from_slice(&[z.re, z.im]);
        }
        point.fill(0.0);
        let (generator, rest) = point.split_at_mut(plan.generator_start.len());
        let (forms, jumps) = rest.split_at_mut(plan.form_start.len());
        plan.lay(|at| values[at], Some(generator), forms);
        let len = values.len();
        plan.decay(|at| if at < len { values[at] + plan.bare(at) } else { plan.constants[at - len] }, &mut vec![], generator);
        plan.lay_jumps(|at| values[at], jumps);
    }

    /// A near table's grid with axis f's spacing halved (crate::table `Axis::halved`), its nodes reused: the refinement's one step.
    pub fn halved(&self, table: &NearTable, f: usize) -> Table {
        table.table.refine(Model::finer(table, f), &|s: &[f64], out: &mut [f64]| self.node_at(&table.layout, s, out))
    }

    /// A table's axes with axis f's spacing halved.
    fn finer(table: &NearTable, f: usize) -> Vec<Axis> {
        table.table.axes().iter().enumerate().map(|(g, a)| a.halved(if g == f { 1 } else { 0 })).collect()
    }

    /// The node at the axes' values s, the other coordinates at their focus values.
    fn node_at(&self, layout: &Layout, s: &[f64], out: &mut [f64]) {
        let mut x = layout.focus.clone();
        x[..s.len()].copy_from_slice(s);
        self.node(layout, &x, out);
    }

    /// Axis f of a table halved by the read at the atoms (solve.rs; module documentation, # Refinement): where `over`, its
    /// interpolation error along f over the tables' shares as the caller reads it on the table's grid, asks it
    /// (`halving_on`) against the caller's read of the grid before, `before` (None before the first), which takes this
    /// read, and the halved grid within the work bound `most`. None where not: the axis is refined no further.
    pub fn halved_by(&self, table: &NearTable, f: usize, over: f64, before: &mut Option<f64>, most: usize) -> Option<Table> {
        let on = halving_on(over, table.table.axes()[f].spacing(), table.layout.rates[f], *before);
        *before = Some(over);
        (on && Table::nodes(&Model::finer(table, f)) <= most).then(|| self.halved(table, f))
    }

    /// The near tables for the atoms' reach (module documentation, # Axes, # Refinement): per cluster the running lights'
    /// and, where the trap is lit and a carrier stays lit without it, those lights less the trap's -- each light set under
    /// which the cluster has near forms --, refined over the reach; then the coordinates its first table's looks held are
    /// looked at once more on the refined one (module documentation, # Axes), and the table is made again with one that it
    /// needs after all. A coordinate the axes' rules refuse (`first_table`: the cost of an interpolation, the work bound
    /// `most` on the first grid, `table::MOST_AXES`) is held at its focus value like the light ones, whatever its weight:
    /// the table's whole error at the atoms reads it (`table_reads`). `most` decides which coordinates are axes, and is the
    /// ceiling of a refinement its function's structure stops (# Refinement). A table asks nothing of another: they are made
    /// side by side on the pool, and kept in the order of their clusters and light sets.
    pub fn tabulate(&self, reach: &Reach, most: usize) -> Tables {
        let running = self.running();
        let mut untrapped = running.clone();
        untrapped.0[self.light.trap] = Lit::Off;
        let sets: Vec<(usize, &Lights)> = (0..self.clusters.len())
            .flat_map(|cluster| [&running, &untrapped].map(|lights| (cluster, lights)))
            .filter(|&(_, lights)| !(lights == &untrapped && !running.on(self.light.trap)))
            .collect();
        let tables: Vec<Option<NearTable>> = sets
            .par_iter()
            .map(|&(cluster, lights)| {
                let mut needed = vec![];
                while let Some(First { table, ranges, held }) = self.first_table(cluster, lights, reach, most, &needed) {
                    let table = self.refined(table, &ranges, reach, most);
                    match held.iter().find(|(coordinate, worst)| self.look(&table, &ranges, *coordinate, &[*worst], reach).0 > 1.0) {
                        Some((coordinate, _)) => needed.push(*coordinate),
                        None => return Some(table),
                    }
                }
                None
            })
            .collect();
        Tables { tables: tables.into_iter().flatten().collect(), thermal: crate::units::KT_PER_UK * reach.temperature_uk }
    }

    /// The ranges of coordinates over the atoms' reach (module documentation, # Axes): each its focus value ± the proven bound
    /// of its distance from it (`coordinate_moved`), to the least of their first spacings, on its axis's range -- a phase's on its whole
    /// period where that interval would cross the period's seam.
    fn reach_ranges(&self, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>], focus: &[f64], reach: &Reach) -> Vec<(f64, f64)> {
        if coordinates.is_empty() {
            return vec![];
        }
        let quantities: Vec<Box<dyn Fn(&Region) -> f64 + Sync + '_>> = coordinates.iter().zip(focus).map(|(&c, &at)| Box::new(move |r: &Region| self.coordinate_moved(c, polarizations, at, r)) as Box<dyn Fn(&Region) -> f64 + Sync + '_>).collect();
        let refs: Vec<&(dyn Fn(&Region) -> f64 + Sync)> = quantities.iter().map(|q| q.as_ref()).collect();
        let resolution = coordinates.iter().map(|&c| first_axis(c).spacing()).fold(f64::INFINITY, f64::min);
        let moved = self.extents_over(reach, resolution, &refs);
        coordinates
            .iter()
            .zip(focus)
            .zip(moved)
            .map(|((&c, &at), moved)| {
                let axis = first_axis(c);
                if axis.periodic && (at - moved < axis.lo || at + moved > axis.hi) {
                    (axis.lo, axis.hi)
                } else {
                    ((at - moved).max(axis.lo), (at + moved).min(axis.hi))
                }
            })
            .collect()
    }

    /// A table's error by the two measures (`node_error`: the responses' relative error, the bound of the potential's) over
    /// the tables' shares (module documentation, # Refinement), the larger of the two: twice the responses' over the smaller
    /// of what is shown's and T's, the potential's over T's of k_BT_r. Within the shares: at most one.
    fn over(&self, (responses, potential): (f64, f64), reach: &Reach) -> f64 {
        let (shown, temperature) = shares();
        (2.0 * responses / shown.min(temperature)).max(potential / (temperature * crate::units::KT_PER_UK * reach.temperature_uk))
    }

    /// A table refined over the atoms' reach, its axes' ranges there given (module documentation, # Refinement): each axis in
    /// turn halved, each halving reading the grid before it at the new nodes, while that read asks it (`halving_on`) and the
    /// halved grid is within the work bound `most`.
    fn refined(&self, mut table: NearTable, ranges: &[(f64, f64)], reach: &Reach, most: usize) -> NearTable {
        let d = table.axes();
        let (mut ended, mut before) = (vec![false; d], vec![None; d]);
        while let Some(f) = ended.iter().position(|e| !e) {
            if Table::nodes(&Model::finer(&table, f)) > most {
                ended[f] = true;
                continue;
            }
            let finer = self.halved(&table, f);
            let over = self.over(self.halving_error(&table.layout, &table.table, &finer, f, ranges, reach), reach);
            ended[f] = !halving_on(over, table.table.axes()[f].spacing(), table.layout.rates[f], before[f]);
            before[f] = Some(over);
            table.table = finer;
        }
        table
    }

    /// A look at a coordinate a table holds at its focus value (module documentation, # Axes): the table's nodes that the
    /// reach's ranges of its axes meet (a node's own spacing about it), each against the elimination there with the
    /// coordinate at each of `values`, by the refinement's two measures. The largest error over the tables' shares (`over`)
    /// and the value it was read at (the focus value where every error is nothing).
    fn look(&self, table: &NearTable, ranges: &[(f64, f64)], coordinate: Coordinate, values: &[f64], reach: &Reach) -> (f64, f64) {
        let (layout, grid) = (&table.layout, &table.table);
        let place = layout.coordinates.iter().position(|c| *c == coordinate).expect("a coordinate of the table's layout");
        let shift = self.reach_shifts(reach);
        let places = |node: usize| -> Vec<f64> { grid.node(node).0.iter().zip(grid.axes()).map(|(&i, a)| a.node(i)).collect() };
        let met: Vec<usize> = (0..Table::nodes(grid.axes())).filter(|&node| places(node).iter().zip(grid.axes()).zip(ranges).all(|((s, a), range)| s - a.spacing() <= range.1 && s + a.spacing() >= range.0)).collect();
        let looks: Vec<(usize, f64)> = met.iter().flat_map(|&node| values.iter().map(move |&value| (node, value))).collect();
        looks
            .par_iter()
            .map(|&(node, value)| {
                let mut x = layout.focus.clone();
                let s = places(node);
                x[..s.len()].copy_from_slice(&s);
                x[place] = value;
                let mut moved = vec![0.0; grid.width()];
                self.node(layout, &x, &mut moved);
                (self.over(self.node_error(layout, &shift, &moved, grid.node(node).1), reach), value)
            })
            .reduce(|| (0.0, layout.focus[place]), |worst, read| if read.0 > worst.0 { read } else { worst })
    }

    /// Per frequency the largest Doppler shift of its beams at the reach's speed (the weight of a T2's error beside its T's,
    /// `node_error`).
    fn reach_shifts(&self, reach: &Reach) -> Vec<f64> {
        self.light.frequencies.iter().map(|f| f.beams.iter().map(|&l| self.doppler_shift(l, [0.0; 3]) + self.light.beams[l].wave_number * reach.speed).fold(0.0, f64::max)).collect()
    }

    /// A node's reals `other` against the elimination's `exact` (module documentation, # Refinement): the responses' relative
    /// error, the largest over the lit carriers, and the bound of the potential's error (rad/µs), the largest over the ground
    /// places; +∞ where one is no number. `shift`: `reach_shifts`.
    fn node_error(&self, layout: &Layout, shift: &[f64], exact: &[f64], other: &[f64]) -> (f64, f64) {
        let plan = &layout.plan;
        let (start, count, pairs) = (plan.near.responding().start, plan.near.responses.len(), plan.near.pairs.len());
        // A field component's peak.
        let peak = |frequency: usize, k: usize| self.light.frequencies[frequency].coefficients.iter().map(|c| c[k].abs()).sum::<f64>();
        let complex = |row: &[f64], at: usize| C64::new(row[2 * at], row[2 * at + 1]);
        let read = |x: f64| if x.is_nan() { f64::INFINITY } else { x };
        let error = |at: usize| (complex(other, at) - complex(exact, at)).norm_sqr();
        let (mut responses, mut potential) = (0.0_f64, 0.0_f64);
        for (c, cc) in self.carriers.iter().enumerate() {
            let w = shift[cc.frequency];
            let sum = |of: &dyn Fn(usize) -> f64| (0..count).filter(|&i| plan.near.responses[i][0] == c).map(of).sum::<f64>().sqrt();
            let size = sum(&|i| complex(exact, start + i).norm_sqr());
            if !(size == 0.0) {
                responses = responses.max(read((sum(&|i| error(start + i)) + w * sum(&|i| error(start + count + i))) / size));
            }
        }
        for x in 0..self.clusters[layout.cluster].ground {
            let moved: f64 = plan
                .near
                .pairs
                .iter()
                .enumerate()
                .filter(|(_, p)| p[0] == p[2] && p[4] == x)
                .map(|(i, p)| peak(p[0], p[1]) * peak(p[2], p[3]) * (error(i).sqrt() + 2.0 * shift[p[0]] * error(pairs + i).sqrt()))
                .sum();
            potential = potential.max(read(moved));
        }
        (responses, potential)
    }

    /// The error of a table against the one a halving of axis f finer, at the finer's new nodes over the reach's ranges (module
    /// documentation, # Refinement): the responses' relative error and the bound of the potential's error (`node_error`), each
    /// the largest over the nodes.
    fn halving_error(&self, layout: &Layout, coarse: &Table, fine: &Table, f: usize, ranges: &[(f64, f64)], reach: &Reach) -> (f64, f64) {
        let shift = self.reach_shifts(reach);
        let mut interpolated = vec![0.0; fine.width()];
        let mut error = (0.0_f64, 0.0_f64);
        for node in 0..Table::nodes(fine.axes()) {
            let (at, exact) = fine.node(node);
            if at[f] % 2 == 0 {
                continue;
            }
            // A new node: the middle of an interval of the coarser grid along f, a node of it along the other axes; within the
            // reach where the intervals it lies in or bounds meet the ranges.
            let s: Vec<f64> = at.iter().zip(fine.axes()).map(|(&i, a)| a.node(i)).collect();
            let within = (0..s.len()).all(|g| {
                let h = coarse.axes()[g].spacing() * if g == f { 0.5 } else { 1.0 };
                s[g] - h <= ranges[g].1 && s[g] + h >= ranges[g].0
            });
            if !within {
                continue;
            }
            coarse.eval(&s, &mut interpolated, None);
            let (responses, potential) = self.node_error(layout, &shift, exact, &interpolated);
            error = (error.0.max(responses), error.1.max(potential));
        }
        error
    }

    /// A cluster's first table for a light set (module documentation, # Axes): its axes the coordinates its first looks
    /// find it needs, and those `needed` (a last look's, taken unasked), each on its first grid; None where it has no near
    /// form under it, or where its grid exceeds `most` nodes.
    fn first_table(&self, cluster: usize, lights: &Lights, reach: &Reach, most: usize, needed: &[Coordinate]) -> Option<First> {
        let plan = self.plan(cluster, lights);
        if plan.near.len() == 0 {
            return None;
        }
        let coherent = self.coherent_dressing();
        let polarizations: Vec<Vec<Polarization>> = (0..self.carriers.len()).map(|d| if coherent[d] { vec![] } else { self.polarizations(d) }).collect();
        let all = self.coordinates(lights, &polarizations);
        let focus_amplitudes = self.light.amplitudes([0.0; 3]);
        let mut focus = vec![0.0; all.len()];
        self.coordinates_at(&all, &polarizations, &focus_amplitudes, &mut focus, &mut vec![[0.0; 3]; all.len()]);
        let rates = self.rates(cluster, lights, &all, &polarizations, &focus);
        // Every coordinate's range over the reach, and its weight there: its rate times the range it moves across.
        let over_reach = self.reach_ranges(&all, &polarizations, &focus, reach);
        let weights: Vec<f64> = rates.iter().zip(&over_reach).map(|(rate, (lo, hi))| rate * (hi - lo)).collect();
        let mut order: Vec<usize> = (0..all.len()).collect();
        order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
        // The table of given axes, each on its first grid, the other coordinates at the focus.
        let width = 2 * plan.near.len();
        let exact = self.near_exact(&plan, &focus_amplitudes, 1, &coherent).work * crate::decide::COMPLEX;
        let made = |axes: &[usize]| -> Option<NearTable> {
            let picked: Vec<usize> = axes.iter().copied().chain(order.iter().copied().filter(|c| !axes.contains(c))).collect();
            let layout = Layout {
                cluster,
                lights: lights.clone(),
                coordinates: picked.iter().map(|&c| all[c]).collect(),
                focus: picked.iter().map(|&c| focus[c]).collect(),
                rates: picked.iter().map(|&c| rates[c]).collect(),
                polarizations: polarizations.clone(),
                plan: plan.clone(),
            };
            let grid = axes.iter().map(|&c| first_axis(all[c])).collect();
            let table = Table::build(grid, layout.width(), most, &|s: &[f64], out: &mut [f64]| self.node_at(&layout, s, out)).ok()?;
            Some(NearTable { layout, table })
        };
        // The axes: of the coordinates that move over the reach, the heaviest first, those the table so far needs by its look
        // at the coordinate's first nodes over the reach -- as many as keep an interpolation cheaper than the elimination and
        // the first grid within the bound. One that does not move there is its focus value at every atom: exact, no axis.
        let (mut axes, mut ranges, mut held) = (vec![], vec![], vec![]);
        let mut table = made(&axes)?;
        for &c in &order {
            let range = over_reach[c];
            if range.0 == range.1 {
                continue;
            }
            let d = axes.len() + 1;
            let first: Vec<Axis> = axes.iter().chain(std::iter::once(&c)).map(|&a| first_axis(all[a])).collect();
            if d > crate::table::MOST_AXES || (4usize.pow(d as u32) * (1 + d) * width) as f64 >= exact || Table::nodes(&first) > most {
                break;
            }
            let axis = first[d - 1];
            let take = needed.contains(&all[c]) || {
                // Where the atoms take it: the ends of its range over the reach and its first nodes between them.
                let values: Vec<f64> = [range.0, range.1].into_iter().chain((0..axis.nodes).map(|i| axis.node(i)).filter(|v| range.0 < *v && *v < range.1)).collect();
                let (error, worst) = self.look(&table, &ranges, all[c], &values, reach);
                if error <= 1.0 {
                    held.push((all[c], worst));
                }
                error > 1.0
            };
            if take {
                axes.push(c);
                ranges.push(range);
                table = made(&axes)?;
            }
        }
        Some(First { table, ranges, held })
    }

    /// The coordinates' rates for a cluster under a light set (module documentation, # Axes): how many half-widths a unit of
    /// each moves the cluster's lines.
    fn rates(&self, cluster: usize, lights: &Lights, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>], focus: &[f64]) -> Vec<f64> {
        let c = &self.clusters[cluster];
        // The cluster's lines e ← b and their half-widths.
        let lines: Vec<(usize, usize, f64)> = c.states[..c.ground]
            .iter()
            .flat_map(|&b| {
                (GROUND..STATES)
                    .filter(move |&e| self.carriers.iter().any(|cc| lights.on(cc.frequency) && cc.folded.iter().any(|x| x[(e, b)] != C64::ZERO)))
                    .map(move |e| (e, b, self.decay[(e - GROUND, e - GROUND)].re / 2.0))
            })
            .collect();
        let coherent = self.coherent_dressing();
        let energies = |x: &[f64]| -> Vec<f64> {
            let near = self.light_at(lights, coordinates, polarizations, x);
            let mut energies = vec![];
            self.far_energies(&near, lights, 0, &coherent, &mut energies);
            energies.swap_remove(0)
        };
        let at_focus = energies(focus);
        // The largest change of a line's E_e − E_b from the focus, in its half-widths.
        let moved = |e: &[f64]| lines.iter().map(|&(a, b, half)| ((e[a] - e[b]) - (at_focus[a] - at_focus[b])).abs() / half).fold(0.0, f64::max);
        let region = self.point_region([0.0; 3], 0.0);
        coordinates
            .iter()
            .enumerate()
            .map(|(i, coordinate)| match *coordinate {
                Coordinate::Chart { index, .. } => {
                    // The largest change over the nodes: an intensity's per unit intensity (its shift linear in it); an angle's
                    // or a phase's the amplitude of a trigonometric polynomial of degree two or one in it, whose slope is at
                    // most its degree times that (Bernstein).
                    let axis = first_axis(*coordinate);
                    let (mut most, mut span) = (0.0_f64, 0.0_f64);
                    for n in 0..axis.nodes {
                        let mut x = focus.to_vec();
                        x[i] = axis.node(n);
                        span = span.max((x[i] - focus[i]).abs());
                        most = most.max(moved(&energies(&x)));
                    }
                    match index {
                        0 if span > 0.0 => most / span,
                        0 => 0.0,
                        _ if index % 2 == 1 => 2.0 * most,
                        _ => most,
                    }
                }
                Coordinate::Unit { carrier, unit } => {
                    let p = &polarizations[carrier][unit];
                    let x = &self.carriers[carrier].folded[p.beams[0]];
                    let k = p.kappa[0].norm_sqr();
                    let dressing = (0..GROUND)
                        .map(|b| p.norm * (GROUND..STATES).map(|e| x[(e, b)].norm_sqr()).sum::<f64>() / k * self.resolvent(carrier, b, &region))
                        .fold(0.0, f64::max)
                        / (self.least_decay() / 2.0);
                    // Its static form's part, linear in its intensity: from none to all of it.
                    let (mut dark, mut full) = (focus.to_vec(), focus.to_vec());
                    (dark[i], full[i]) = (0.0, 1.0);
                    let (none, all) = (energies(&dark), energies(&full));
                    dressing + lines.iter().map(|&(a, b, half)| ((all[a] - all[b]) - (none[a] - none[b])).abs() / half).fold(0.0, f64::max)
                }
            })
            .collect()
    }

    /// The bound over a region of a coordinate's distance from its focus value `at` (module documentation, # Axes): an
    /// intensity's (`intensity_moved`); an angle's and a phase's from the frequency's field components on its basis, e_k =
    /// Σ_l a_l γ_lk, whose products e_k'* e_k are sums of beam pairs' coefficients as an intensity is (reach.rs
    /// `Region::combined`). An angle α_i = atan2(√Σ_{j≥i}|e_{o_j}|², |e_{o_{i−1}}|) grows with its numerator and falls with
    /// its denominator, so it lies between the angles of their bounds' ends; a phase φ_i = arg(e_{o_i} e_{o₀}*) lies within
    /// the arguments of its product's box (the bounds of its real and imaginary parts) where the box holds no zero, and
    /// is any phase where it does. Two beams of fixed amplitudes and orthogonal fields keep their angle everywhere -- it
    /// weighs nothing, on any cell --, and their relative phase turns through the period over half a wavelength.
    fn coordinate_moved(&self, c: Coordinate, polarizations: &[Vec<Polarization>], at: f64, region: &Region) -> f64 {
        use std::f64::consts::PI;
        let Coordinate::Chart { frequency, index } = c else { return self.intensity_moved(c, polarizations, at, region) };
        if index == 0 {
            return self.intensity_moved(c, polarizations, at, region);
        }
        let f = &self.light.frequencies[frequency];
        let (n, o, gamma) = (f.beams.len(), &f.order, &f.coefficients);
        // The bounds of Re Σ_{j,i} a_j* a_i w(j, i) over the region.
        let bounds = |w: &dyn Fn(usize, usize) -> C64| -> (f64, f64) {
            let real = region
                .combined((0..n * n).map(|p| (p, w(p / n, p % n))).filter(|(_, weight)| *weight != C64::ZERO).map(|(p, weight)| (region.pairs[frequency][p], weight, weight.abs())))
                .real();
            (real.lower(), real.upper())
        };
        if index % 2 == 1 {
            let i = (index + 1) / 2;
            let squared = |set: &[usize]| {
                let (lo, hi) = bounds(&|j, l| set.iter().map(|&k| gamma[j][k].conj() * gamma[l][k]).sum());
                (lo.max(0.0).sqrt(), hi.max(0.0).sqrt())
            };
            let ((top_lo, top_hi), (bottom_lo, bottom_hi)) = (squared(&o[i..]), squared(&o[i - 1..i]));
            let (lo, hi) = (top_lo.atan2(bottom_hi), top_hi.atan2(bottom_lo));
            return (hi - at).max(at - lo).max(0.0);
        }
        let (k, pivot) = (o[index / 2], o[0]);
        let (re, im) = (bounds(&|j, l| gamma[j][pivot].conj() * gamma[l][k]), bounds(&|j, l| gamma[j][pivot].conj() * gamma[l][k] * C64::new(0.0, -1.0)));
        if re.0 <= 0.0 && re.1 >= 0.0 && im.0 <= 0.0 && im.1 >= 0.0 {
            return PI;
        }
        // The box's arguments about its centre's: an arc shorter than π, spanned by its corners.
        let wrap = |a: f64| (a + PI).rem_euclid(2.0 * PI) - PI;
        let centre = C64::new((re.0 + re.1) / 2.0, (im.0 + im.1) / 2.0).arg();
        let corners = [(re.0, im.0), (re.0, im.1), (re.1, im.0), (re.1, im.1)].map(|(x, y)| wrap(C64::new(x, y).arg() - centre));
        let (lo, hi) = (corners.iter().copied().fold(f64::INFINITY, f64::min), corners.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        let antipode = wrap(at + PI - centre);
        if lo <= antipode && antipode <= hi {
            PI
        } else {
            wrap(centre + lo - at).abs().max(wrap(centre + hi - at).abs())
        }
    }

    /// The bound over a region of an intensity coordinate's distance from its focus value `at` (module documentation, # Axes). The
    /// intensity is a field's squared modulus over N, the field F = Σ_l a_l c_l (a frequency's, c_l = E_l; a unit's, c_l = κ_l):
    /// |F|² = Σ_l'l a_l'* a_l c_l'†c_l, a sum of beam pairs' coefficients (reach.rs `PairSpan`) -- so |F|² lies within how far
    /// that sum moves of its own form (reach.rs `Region::combined`) of its centre value, at least 0 and at most every beam at
    /// its largest amplitude and in phase; on a point it is the value. A pair of beams whose fields are
    /// orthogonal is no term of it, and a plane wave's pair with itself does not move: light whose intensity is the same
    /// everywhere weighs nothing, on any cell. (Bounded by the field's modulus within the region's radius times Σ_l |c_l|
    /// (|k_l − k₀| max|b_l| + max|∇b_l|) of its centre value, the intensity of two counter-propagating plane waves of
    /// orthogonal polarizations -- a constant -- was taken to move by 2kρ: with the lin-perp-lin scene's cooling light on its
    /// chart no cell of the reach was small enough, the proof had cut 8 million cells in 43 s and went on.)
    fn intensity_moved(&self, c: Coordinate, polarizations: &[Vec<Polarization>], at: f64, region: &Region) -> f64 {
        // The coordinate's field: its frequency, its beams' places among the frequency's with their c_l, and N.
        let (frequency, beams, norm): (usize, Vec<(usize, [C64; 3])>, f64) = match c {
            Coordinate::Chart { frequency, .. } => {
                let f = &self.light.frequencies[frequency];
                (frequency, f.fields.iter().copied().enumerate().collect(), f.norm)
            }
            Coordinate::Unit { carrier, unit } => {
                let p = &polarizations[carrier][unit];
                (self.carriers[carrier].frequency, p.beams.iter().zip(&p.kappa).map(|(&l, &k)| (l, [k, C64::ZERO, C64::ZERO])).collect(), p.norm)
            }
        };
        let f = &self.light.frequencies[frequency];
        let size = |c: &[C64; 3]| c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let blind: f64 = beams.iter().map(|(l, c)| region.spans[f.beams[*l]].most * size(c)).sum();
        let intensity = region.combined(
            beams
                .iter()
                .flat_map(|(j, to)| beams.iter().map(move |(i, from)| (j * f.beams.len() + i, (0..3).map(|q| to[q].conj() * from[q]).sum::<C64>())))
                .filter(|(_, weight)| *weight != C64::ZERO)
                .map(|(pair, weight)| (region.pairs[frequency][pair], weight, weight.abs())),
        );
        let intensity = intensity.real();
        let (lo, hi) = (intensity.lower().max(0.0), intensity.upper().min(blind * blind));
        (hi / norm - at).abs().max((at - lo / norm).abs())
    }

    /// The near forms at a point from a table (module documentation, # At a point), the sections asked each to its order (0 or
    /// 1): their values and, to first order, their slopes along the axes with the coordinates' gradients.
    pub(crate) fn near_tabulated(&self, table: &NearTable, amplitudes: &[Amplitude], sections: &[(Range<usize>, usize)]) -> NearValues {
        let d = table.axes();
        let (mut x, mut gradients) = (vec![0.0; d], vec![[0.0; 3]; d]);
        let mut work = self.coordinates_at(&table.layout.coordinates[..d], &table.layout.polarizations, amplitudes, &mut x, &mut gradients);
        let len = sections.iter().map(|s| s.0.end).max().unwrap_or(0);
        let order = sections.iter().map(|s| s.1).max().unwrap_or(0);
        let mut data = vec![C64::ZERO; (1 + if order >= 1 { d } else { 0 }) * len];
        let place = table.table.place(&x);
        for (entries, order) in sections {
            let w = 2 * entries.len();
            let (mut value, mut slope) = (vec![0.0; w], vec![0.0; if *order >= 1 { d * w } else { 0 }]);
            table.table.read(&place, 2 * entries.start, None, &mut value, if *order >= 1 { Some(&mut slope) } else { None });
            for (i, e) in entries.clone().enumerate() {
                data[e] = C64::new(value[2 * i], value[2 * i + 1]);
                if *order >= 1 {
                    for f in 0..d {
                        data[(1 + f) * len + e] = C64::new(slope[f * w + 2 * i], slope[f * w + 2 * i + 1]);
                    }
                }
            }
            // The interpolation (4^D nodes: the value and, to first order, D slopes of each real), in complex multiply-adds.
            work += read_work(d, *order, w);
        }
        NearValues { len, order, data, directions: Some(gradients), work }
    }

    /// A site's point on a table (module documentation, # At a point), from the site's beams' amplitudes: the axes'
    /// coordinates there, their gradients -- the directions the forms' and the jumps' slopes are along --, and the
    /// point's place on the table's grid with the coordinates' counted work (`Site::place`). Every read of the table
    /// at the point takes that place (`near_point`, `near_jumps`, `near_responses`; assemble.rs `Model::quartic`): a
    /// site is placed once, after its beams and before its first read, whatever is read of the table there.
    pub(crate) fn placed(&self, table: &NearTable, point: &mut Site) {
        let d = table.axes();
        sized(&mut point.coordinates, d);
        sized(&mut point.directions, d);
        let work = self.coordinates_at(&table.layout.coordinates[..d], &table.layout.polarizations, &point.beams, &mut point.coordinates, &mut point.directions);
        point.place = Some((table.table.place(&point.coordinates), work));
    }

    /// A point's jumps' matrices from a table (assemble.rs, # The jumps' matrices), to an order (0 or 1), into the
    /// vectors of a site placed there (`placed`), and their counted work: the matrices -- the table's interpolant of
    /// them onto their constant part -- and to first order their slopes per axis, in one pass over the stencil's rows.
    /// Counted with the coordinates as the jumps' own interpolation is, each jump once (assemble.rs, # Work).
    pub(crate) fn near_jumps(&self, table: &NearTable, order: usize, point: &mut Site) -> f64 {
        let (d, plan) = (table.axes(), &table.layout.plan);
        let slopes = if order >= 1 { d } else { 0 };
        let (place, work) = point.place.expect("a site placed on its table");
        let (first, width) = (2 * plan.near.len() + plan.generator_start.len() + plan.form_start.len(), plan.jumps_start.len());
        sized(&mut point.jumps, width);
        sized(&mut point.jump_slopes, slopes * width);
        table.table.read(&place, first, Some(&plan.jumps_start), &mut point.jumps, (order >= 1).then_some(&mut point.jump_slopes[..]));
        work + read_work(d, order, 2 * plan.near.jumping().len())
    }

    /// A point's responses from a table (the near forms' responding section, to order 0: what the readout reads,
    /// readout.rs), into the vectors of a site placed there (`placed`): the section's reals, a response's real part
    /// and its imaginary part in the forms' order.
    pub(crate) fn near_responses(&self, table: &NearTable, point: &mut Site) {
        let entries = table.layout.plan.near.responding();
        let (place, _) = point.place.expect("a site placed on its table");
        sized(&mut point.responses, 2 * entries.len());
        table.table.read(&place, 2 * entries.start, None, &mut point.responses, None);
    }

    /// A middle's near terms' values at a point from a table (module documentation, # At a point; assemble.rs `Site`),
    /// into a site placed there (`placed`), and their counted work: the table's interpolant of the generator's values
    /// -- the Hermitian forms and the decay's sums of the jumps' products, as the nodes hold them -- and of the forms'
    /// with their slopes along the axes, each straight to its place onto its constant part, in one pass over the
    /// stencil's rows. Counted with the coordinates as the forms' interpolation is (the Hermitian ones to first order,
    /// the jumps to order 0): the count is the definition's, not the layout's (assemble.rs, # Work).
    pub(crate) fn near_point(&self, table: &NearTable, point: &mut Site) -> f64 {
        let (d, plan) = (table.axes(), &table.layout.plan);
        let (place, work) = point.place.expect("a site placed on its table");
        let (first, generator, forms) = (2 * plan.near.len(), plan.generator_start.len(), plan.form_start.len());
        sized(&mut point.generator, generator);
        sized(&mut point.forms, forms);
        sized(&mut point.slopes, d * forms);
        table.table.read(&place, first, Some(&plan.generator_start), &mut point.generator, None);
        table.table.read(&place, first + generator, Some(&plan.form_start), &mut point.forms, Some(&mut point.slopes));
        work + plan.near_read(d)
    }

    /// The tables' error reads at samples (module documentation; SPEC §2.3, §5.4): per table, per axis the read of its
    /// interpolation error (crate::table `Table::axis_error`) and the whole error (the exact elimination against the table's), each
    /// as the error it makes of the sample's generator (`form_error`).
    pub fn table_reads(&self, tables: &Tables, samples: &[Sample]) -> Vec<TableRead> {
        let zero = UnitError::NONE;
        tables
            .tables
            .iter()
            .map(|table| {
                let (d, plan) = (table.axes(), &table.layout.plan);
                let (w, generator) = (table.table.width(), 2 * plan.near.len()..2 * plan.near.len() + plan.generator_start.len());
                let mut axes = vec![vec![zero; samples.len()]; d];
                let mut whole = vec![zero; samples.len()];
                for (k, sample) in samples.iter().enumerate() {
                    if sample.cluster != table.layout.cluster {
                        continue;
                    }
                    let amplitudes = self.light.amplitudes(sample.at.r);
                    let mut x = vec![0.0; d];
                    self.coordinates_at(&table.layout.coordinates[..d], &table.layout.polarizations, &amplitudes, &mut x, &mut vec![[0.0; 3]; d]);
                    // The generator the steps take from the table, on the sample's state; and with no near light.
                    let tabulated = self.middle(plan, sample.at.r, sample.at.t, Some(table));
                    let (far, ours) = (self.action(sample, &self.middle_of(plan, sample.at.t, &amplitudes, None)), self.action(sample, &tabulated));
                    for (f, read) in axes.iter_mut().enumerate() {
                        // The axis' read of the generator's values, the decay's with the Hermitian ones.
                        let (mut dv, mut ds) = (vec![0.0; w], vec![0.0; d * w]);
                        table.table.axis_error(f, &x, &mut dv, &mut ds);
                        read[k] = self.form_error(sample, &far, &ours, &self.action(sample, &tabulated.moved(&dv[generator.clone()])), tables.thermal);
                    }
                    // The exact elimination's forms to order 0: the generator on a state reads their values alone.
                    let exact = self.middle_of(plan, sample.at.t, &amplitudes, Some(&self.near_exact(plan, &amplitudes, 0, &self.coherent_dressing())));
                    whole[k] = self.form_error(sample, &far, &self.action(sample, &exact), &ours, tables.thermal);
                }
                TableRead { axes, whole }
            })
            .collect()
    }

    /// A middle's generator at a sample's momentum, at rest on the middle, on the sample's state.
    pub(crate) fn action(&self, sample: &Sample, middle: &Middle) -> Vec<C64> {
        let n = self.clusters[sample.cluster].states.len();
        let mut total = Mat::zeros(n, n);
        crate::step::Middle::generator(middle, &crate::step::Middle::path(middle, sample.at.kappa, [0.0; 3]), 0.0, &mut total);
        total.apply(&sample.psi)
    }

    /// The error on the outputs of a sample from an error of its generator's near part (module documentation, # The table's
    /// error), the generator of `second` against that of `first` -- each on the sample's state (`action`), `far` the one with
    /// no near light: with N the near light's part of the generator at the sample's point and momentum, δ = ‖(N′ − N)ψ‖/‖Nψ‖,
    /// and ΔV = ⟨ψ|H′_h − H_h|ψ⟩/‖ψ‖²; `thermal` is k_BT_r.
    pub(crate) fn form_error(&self, sample: &Sample, far: &[C64], first: &[C64], second: &[C64], thermal: f64) -> UnitError {
        let moved: Vec<C64> = second.iter().zip(first).map(|(a, b)| *a - *b).collect();
        let size = first.iter().zip(far).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>().sqrt();
        let norm = crate::linalg::norm_sqr(&sample.psi);
        // An error that is no number is unread: +∞ (module documentation, # Refinement).
        let read = |x: f64| if x.is_nan() { f64::INFINITY } else { x };
        let delta = if size == 0.0 { 0.0 } else { read(crate::linalg::norm_sqr(&moved).sqrt() / size) };
        let potential = read(crate::linalg::inner(&sample.psi, &moved).re.abs() / norm);
        let g = self.clusters[sample.cluster].ground;
        let weight = crate::linalg::norm_sqr(&sample.psi[..g]) / norm;
        let shown = if weight == 0.0 { 0.0 } else { delta * weight };
        UnitError { parameter: delta, population: shown, temperature: shown.max(potential / thermal), ..UnitError::NONE }
    }
}

/// A coordinate's first grid (crate::table `Axis::first`) on its whole range (field.rs `range`; a unit's intensity [0, 1]).
fn first_axis(c: Coordinate) -> Axis {
    match c {
        Coordinate::Chart { index, .. } => {
            let (lo, hi, periodic) = range(index);
            Axis::first(lo, hi, periodic)
        }
        Coordinate::Unit { .. } => Axis::first(0.0, 1.0, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
    use crate::model::{Component, Point, Unit};
    use crate::request::Request;

    fn scene(name: &str) -> Request {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        Request::from_native(&line).unwrap()
    }

    /// The reach of a model's atoms in the trap of ground state `b` at a temperature, as the solve makes it.
    fn reach_of(model: &Model, b: usize, temperature: f64) -> (crate::model::Trap, Reach) {
        let p: [f64; GROUND] = std::array::from_fn(|s| if s == b { 1.0 } else { 0.0 });
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, temperature, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        (trap, reach)
    }

    /// One criterion for what a table leaves out (module documentation, # Axes), on the two sides of it.
    ///
    /// A coordinate that is needed. R2_RECOIL's σ⁺ probe on |2,2⟩ → |3′,3′⟩, folded at the atom and on resonance at the
    /// focus, in a tweezer whose shift of |3′,3′⟩ is the ground state's but for 0.04 half-widths of the line (an outside
    /// form's entry): the tweezer's intensity weighs 0.0407 at most, between half the tables' shares (0.0039) and the
    /// population target. Over a reach of 120 µK it is the table's axis, and the table's error as the ledger reads it
    /// (`table_reads`) is within the shares from the trap's axis to the reach's edge. (Held at the focus, as the cut at
    /// the population target had it, the read was 2.1e-2 at the edge and 8.2e-3 on average: beyond the share 7.8e-3 of a
    /// table that had no axis to halve.)
    ///
    /// A coordinate that is not. R3_DOPPLER's cooling pair is one polarization unit, its intensity a standing wave's
    /// from none to all over the reach, and its dressing at full intensity is 0.045 half-widths: by its weight a
    /// coordinate to ask. The pair is the model's one carrier and its line the cycling one, so its dressing is in no
    /// denominator of its own response: the first look, the unit at its seven first nodes on the tweezer's intensity's
    /// first nodes, reads 6e-7 of the shares and holds it; the last, at every node of the refined table (193), 7e-7; and
    /// the scene's table is the one of the tweezer's intensity alone. Named needed, the unit is an axis unasked.
    #[test]
    fn a_coordinate_is_an_axis_where_the_table_without_it_is_beyond_its_shares() {
        let (request, light, kept, b) = nearly_magic(0.04);
        let model = Model::new(&request, &light, &kept).unwrap();
        let (trap, reach) = reach_of(&model, b, 4.0 * request.temperature_uk);
        let tables = model.tabulate(&reach, 1 << 16);
        let cluster = model.cluster_of(b);
        let index = tables.tables.iter().position(|t| t.layout.cluster == cluster && t.layout.lights == model.running()).unwrap();
        let table = &tables.tables[index];
        assert!(table.axes() == 1 && table.layout.coordinates[0] == (Coordinate::Chart { frequency: light.trap, index: 0 }), "the tweezer's intensity is the axis: {:?}", &table.layout.coordinates[..table.axes()]);
        let reads = &model.table_reads(&tables, &along(&model, &trap, &reach, b, cluster))[index];
        let largest = reads.whole.iter().map(|r| r.population.max(r.temperature)).fold(0.0, f64::max);
        let (shown, temperature) = shares();
        eprintln!("the probe in the nearly magic tweezer: {} nodes; across the reach the table's largest read {largest:.2e}", Table::nodes(table.table.axes()));
        assert!(largest <= shown.min(temperature), "the table's read {largest:.3e} over the reach beyond its share {shown:.3e}");

        // The cooling pair's own unit: an axis by its weight, one the table does not need.
        let request = scene("R3_DOPPLER");
        let light = Light::new(&request);
        let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let mut kept: Vec<Unit> = vec![];
        let (model, reach) = loop {
            let model = Model::new(&request, &light, &kept).unwrap();
            let (_, reach) = reach_of(&model, state, request.temperature_uk);
            let units = model.units();
            let failing: Vec<Unit> = units.iter().zip(model.gates(&units, &reach)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
            if failing.is_empty() {
                break (model, reach);
            }
            kept.extend(failing);
        };
        let (cluster, lights, most) = (model.cluster_of(state), model.running(), 1 << 16);
        let First { table: first, ranges, held } = model.first_table(cluster, &lights, &reach, most, &[]).unwrap();
        assert!(first.axes() == 1 && matches!(first.layout.coordinates[0], Coordinate::Chart { index: 0, .. }), "the first table's axis is the tweezer's intensity: {:?}", &first.layout.coordinates[..first.axes()]);
        assert!(held.len() == 1 && matches!(held[0].0, Coordinate::Unit { .. }), "the pair's unit is asked and held: {held:?}");
        let (unit, worst) = held[0];
        let (axis, grid) = (first_axis(unit), Table::nodes(first.table.axes()));
        let values: Vec<f64> = (0..axis.nodes).map(|i| axis.node(i)).collect();
        let early = model.look(&first, &ranges, unit, &values, &reach).0;
        let refined = model.refined(first, &ranges, &reach, most);
        let last = model.look(&refined, &ranges, unit, &[worst], &reach).0;
        eprintln!("R3_DOPPLER: the pair's unit over the tweezer's {grid} first nodes reads {early:.1e} of the shares; at its worst value {worst} over the refined table's {} nodes, {last:.1e}", Table::nodes(refined.table.axes()));
        assert!(early <= 1.0 && last <= 1.0, "the table does not need the unit: {early:e}, {last:e} of the shares");
        let tables = model.tabulate(&reach, most);
        let table = tables.tables.iter().find(|t| t.layout.cluster == cluster && t.layout.lights == lights).unwrap();
        assert!(table.axes() == 1 && table.layout.coordinates[0] == refined.layout.coordinates[0] && table.table.axes()[0].nodes == refined.table.axes()[0].nodes, "the table kept is the tweezer's intensity's: {:?}", &table.layout.coordinates[..table.axes()]);
        // A coordinate a last look finds needed is taken unasked.
        assert_eq!(model.first_table(cluster, &lights, &reach, most, &[unit]).unwrap().table.axes(), 2, "the unit, needed, is the first table's second axis");
    }

    /// R2_RECOIL with its σ⁺ probe on |2,2⟩ → |3′,3′⟩ folded at the atom and on resonance at the focus, in a tweezer whose shift
    /// of |3′,3′⟩ is the ground state's but for `half_widths` half-widths of the line (an outside form's entry): the request,
    /// its light, the probe's drives kept at the atom, and the ground state.
    fn nearly_magic(half_widths: f64) -> (Request, Light, Vec<Unit>, usize) {
        let (b, e) = (7, 31);
        let mut request = scene("R2_RECOIL");
        let light = Light::new(&request);
        let beam = request.beams.iter().position(|b| b.id == "probe").unwrap();
        let probe = light.frequencies.iter().position(|f| f.beams.contains(&beam)).unwrap();
        let kept = Model::new(&request, &light, &[]).unwrap().bare_units(Component { frequency: probe, counter: false });
        // The line's detuning at the focus, what the tweezer's light moves it by there, and its width.
        let line = |request: &Request| -> (f64, f64, f64) {
            let model = Model::new(request, &light, &kept).unwrap();
            let (coherent, amplitudes) = (model.coherent_dressing(), model.light.amplitudes([0.0; 3]));
            let (lit, mut unlit) = (model.running(), model.running());
            unlit.0[model.light.trap] = Lit::Off;
            let of = |lights: &Lights| {
                let h = &model.far_hamiltonian(&amplitudes, lights, 0, &coherent)[0];
                h[(e, e)].re - h[(b, b)].re
            };
            (of(&lit) - model.carriers[model.carrier_of(probe).unwrap()].detuning, of(&lit) - of(&unlit), model.decay[(e - GROUND, e - GROUND)].re)
        };
        let (_, moved, gamma) = line(&request);
        let tweezer = &light.frequencies[light.trap];
        let field: f64 = (0..3).map(|q| tweezer.fields.iter().map(|f| f[q]).sum::<C64>().norm_sqr()).sum();
        let wanted = half_widths * 0.5 * gamma * moved.signum();
        for q in 0..3 {
            request.outside_forms[tweezer.form].matrices[q][(e, e)] += C64::real((wanted - moved) / field);
        }
        for _ in 0..3 {
            let (detuning, _, _) = line(&request);
            request.hamiltonian[(e, e)] -= C64::real(detuning);
        }
        let (detuning, moved, _) = line(&request);
        assert!(detuning.abs() <= 1e-9 * gamma && (moved.abs() / (0.5 * gamma) - half_widths).abs() < 0.05 * half_widths, "on resonance at the focus, moved by {half_widths} half-widths: {detuning}, {moved}");
        (request, light, kept, b)
    }

    /// Samples of a model's ground state `b` on its cluster at rest along x from the trap's centre to the reach's edge.
    fn along(model: &Model, trap: &crate::model::Trap, reach: &Reach, b: usize, cluster: usize) -> Vec<Sample> {
        let (p, prepared) = (std::array::from_fn(|s| if s == b { 1.0 } else { 0.0 }), model.prepared());
        let edge = (1..).map(|i| 0.01 * i as f64).find(|&rho| model.potential([trap.centre[0] + rho, trap.centre[1], trap.centre[2]], &p, &prepared) > reach.minimum + reach.energy).unwrap();
        (0..40).map(|i| Sample { cluster, at: Point { r: [trap.centre[0] + edge * (i as f64 + 0.5) / 40.0, trap.centre[1], trap.centre[2]], kappa: [0.0; 3], t: 0.0 }, psi: vec![C64::ONE] }).collect()
    }

    /// The refinement's rule on one axis (module documentation, # Refinement), with the table's own interpolant and its read
    /// at the new nodes against a share of 1e-3 relative. A line's pole 1/(λ(s − s₀) + i) on [0, 1], λ half-widths a unit:
    /// at its rate (17.4, S3's tweezer, and 300, unresolved for six halvings), and at a rate twice too small (the static
    /// forms' rate of R4_LIN_2's tweezer): read within the share. A kink |s − s₀| + 1, whose rate (one) calls it smooth:
    /// its read does not fall by more than 2 at a halving, and the axis ends within a few halvings, beyond the share --
    /// where a bound of the run's had let it halve on.
    #[test]
    fn an_axis_is_refined_while_its_reads_fall_faster_than_an_unresolved_peak_s() {
        let first = Axis::first(0.0, 1.0, false);
        // The halvings and the last read over the share.
        let refine = |f: &(dyn Fn(f64) -> C64 + Sync), rate: f64| -> (u32, f64) {
            let share = 1e-3;
            let value = |s: &[f64], out: &mut [f64]| {
                let z = f(s[0]);
                out.copy_from_slice(&[z.re, z.im]);
            };
            let mut table = Table::build(vec![first], 2, usize::MAX, &value).unwrap();
            let mut before = None;
            loop {
                let axis = table.axes()[0];
                let read = (0..axis.nodes - 1)
                    .map(|i| {
                        let s = 0.5 * (axis.node(i) + axis.node(i + 1));
                        let mut v = [0.0; 2];
                        table.eval(&[s], &mut v, None);
                        (C64::new(v[0], v[1]) - f(s)).abs() / f(s).abs()
                    })
                    .fold(0.0, f64::max);
                let over = read / share;
                table = table.refine(vec![axis.halved(1)], &value);
                if !halving_on(over, axis.spacing(), rate, before) {
                    return (first.halvings_to(&table.axes()[0]).unwrap(), over);
                }
                before = Some(over);
            }
        };
        for (pole, rate) in [(17.4, 17.4), (300.0, 300.0), (34.8, 17.4)] {
            let (halvings, over) = refine(&|s: f64| C64::ONE / C64::new(pole * (s - 0.37), 1.0), rate);
            eprintln!("a pole of rate {pole} taken at {rate}: {halvings} halvings, read {over:.2} of the share");
            assert!(over <= 1.0, "the pole of rate {pole} at {rate} read within the share: {over}");
        }
        let (halvings, over) = refine(&|s: f64| C64::real((s - 0.37).abs() + 1.0), 1.0);
        eprintln!("a kink: {halvings} halvings, read {over:.2} of the share");
        assert!(over > 1.0 && halvings <= 3, "the kink ends within a few halvings, beyond the share: {halvings}, {over}");
    }

    /// A table's nodes are its function's, not the run's (module documentation, # Refinement). The nearly magic probe of the
    /// test above with its line moved 30 half-widths over the tweezer's whole intensity, over a reach of 120 µK: the line
    /// crosses resonance on the intensity's axis, and the refinement goes on to a grid whose read is within the shares (1537
    /// nodes). Tabulated under the work bound of a run and of one twice as long -- the bound of the pilot's middles, here 2^12
    /// and 2^13 nodes -- the tables are the same, node for node and value for value, and the table's error at the atoms
    /// across the reach is within its shares; the ceiling of a run eight times as short, 2^9 nodes, ends the axis below it.
    #[test]
    fn a_table_s_nodes_do_not_grow_with_the_run() {
        let (request, light, kept, b) = nearly_magic(30.0);
        let model = Model::new(&request, &light, &kept).unwrap();
        let (trap, reach) = reach_of(&model, b, 4.0 * request.temperature_uk);
        let (short, long) = (model.tabulate(&reach, 1 << 12), model.tabulate(&reach, 1 << 13));
        assert_eq!(short.tables.len(), long.tables.len());
        for (a, z) in short.tables.iter().zip(&long.tables) {
            assert_eq!((a.layout.coordinates.clone(), a.table.axes()), (z.layout.coordinates.clone(), z.table.axes()), "the same axes and grids");
            assert!((0..Table::nodes(a.table.axes())).all(|n| a.table.node(n).1 == z.table.node(n).1), "the same values at every node");
        }
        let cluster = model.cluster_of(b);
        let index = short.tables.iter().position(|t| t.layout.cluster == cluster && t.layout.lights == model.running()).unwrap();
        let nodes = Table::nodes(short.tables[index].table.axes());
        let samples = along(&model, &trap, &reach, b, cluster);
        let largest = model.table_reads(&short, &samples)[index].whole.iter().map(|r| r.population.max(r.temperature)).fold(0.0, f64::max);
        let (shown, temperature) = shares();
        eprintln!("the line moved 30 half-widths: {nodes} nodes under either bound; the table's largest read across the reach {largest:.2e}");
        let ceiling = model.tabulate(&reach, 1 << 9);
        assert!(nodes > 1 << 9 && Table::nodes(ceiling.tables[index].table.axes()) <= 1 << 9, "the refined grid {nodes} beyond the ceiling 2^9, which ends it below");
        assert!(largest <= shown.min(temperature), "the table's read {largest:.3e} across the reach within its share {:.3e}", shown.min(temperature));
    }
}
