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
//! A coordinate's a-priori weight is how far it moves the cluster's lines over the atoms' reach, in pole half-widths: for a chart
//! coordinate the largest change of E_e − E_b (e ← b a carrier's coupling from the cluster's ground states) over its first grid's
//! nodes, the other coordinates at the focus, over Γ_e/2; for a unit, its dressing at full intensity over Γ_min/2 (its resolvent's
//! Varah bound, eliminate.rs) and the same change of E_e − E_b by its static form at full intensity. An intensity's shift is linear in it, so its weight is that per unit intensity times the bound of
//! its distance from its focus value over the reach (reach.rs `Model::bounds_over`): a plane wave's intensity, the same everywhere,
//! weighs nothing; an angle or a phase weighs its whole range. A response is a pole in E_e − E_b, its relative change at most the
//! weight.
//!
//! What a table leaves out is held to one criterion, the tables' shares (# Refinement), whether it lies between an axis' nodes
//! or in a coordinate that is no axis: twice the responses' error within the shares, the potential's within T's of k_BT_r. A
//! coordinate whose weight is within half the shares leaves the responses within them a priori, and is held at the focus.
//!
//! The weight is a bound: the line it is taken across may be far from resonance, or not meet the coordinate at all (a lone
//! carrier's own unit: its dressing is in no denominator of its own response). So a heavier coordinate is asked whether the
//! table needs it, by a look (`Model::look`): the table that holds it at the focus against the elimination with the coordinate
//! moved, at the table's nodes over the reach, by the refinement's own two measures and shares. Twice:
//! - before it is made an axis, the heaviest coordinate first: the table of the axes taken so far, each on its first grid,
//!   against the coordinate at each of its own first nodes over the reach. Beyond the shares it is taken as an axis -- as
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
//! kept errs by is read at the atoms (`Model::table_reads`). The comparison is made over every interval the
//! reach meets: an intensity's range over the reach is its focus value ± the proven bound of its distance from it, to the axis'
//! first spacing (reach.rs `Model::extents_over`); an angle's or a phase's is its whole range.
//! Two measures, per node the largest: the responses' relative error, max over the lit carriers of (‖ΔT‖ + w‖ΔT2‖)/‖T‖ (Frobenius
//! norms over the carrier's basis vectors and the cluster's ground places; w the largest Doppler shift of the carrier's beams at
//! the reach's speed), of which the rates take twice (they are quadratic in the responses); and the bound of the potential's
//! error, the bound of ‖H̃_h − H_h‖ by its rows: Σ over one carrier's shifts and couplings S_k'k[x, y] of |ΔS| (|ΔD| w with it,
//! for either factor) times the two field components' peaks Σ_l |γ_lk| (every beam at its peak and in phase), the largest over the
//! ground places x. An axis is halved until the
//! table before the halving is within the tables' shares -- twice the responses' error within the populations' and T's, the
//! potential's within T's of k_BT_r --, and the finer table is kept: its own error is then below the verified one by the
//! interpolation's orders. Where the next grid would pass the work bound the refinement stops, and the error stands as
//! `table_reads` reads it at the atoms. An error that is no number (a node on a pole of a real denominator, outside the reach
//! but in a stencil that serves it) is unread, +∞: never within a share.
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
/// values at the focus, the range each was verified over where the reach bounds it, its carriers' units, and the cluster's
/// plan under the lights (its near forms).
///
/// The range (`verified`; module documentation, # Axes, # Refinement) of an intensity whose weight per unit intensity is
/// within half the tables' shares is its whole range (None): it is held at the focus whatever it is. Of a heavier one, the
/// reach is what makes the table hold, over its range over the reach -- its focus value ± the proven bound of its
/// distance from it (`reach_range`): an axis is refined over the intervals that meet that range of its grid before the
/// last halving (the table before it read against the elimination at the new nodes there); a coordinate held
/// after a look is looked at its first grid's nodes within a spacing of it (the error rises with an intensity's move,
/// so between them too); one the axes' bounds leave held unlooked has its error read at the atoms there; and one held
/// because its weight over the reach, per unit intensity times that distance, is within half the shares holds wherever
/// the distance keeps it so, within half the shares over its weight per unit intensity of the focus value. Angles and
/// phases are held or refined over their whole ranges (None).
#[derive(Clone, Debug)]
pub struct Layout {
    pub cluster: usize,
    pub lights: Lights,
    pub coordinates: Vec<Coordinate>,
    focus: Vec<f64>,
    verified: Vec<Option<(f64, f64)>>,
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

    /// Coordinates' values at a point and their gradients (module documentation, # At a point) into `values` and `gradients`,
    /// and the complex multiply-adds.
    fn coordinates_at(&self, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>], amplitudes: &[Amplitude], values: &mut [f64], gradients: &mut [[f64; 3]]) -> f64 {
        let mut work = 0.0;
        // The chart of the frequency last read: a frequency's coordinates take one chart.
        let mut held: Option<(usize, Chart)> = None;
        for (i, c) in coordinates.iter().enumerate() {
            match *c {
                Coordinate::Chart { frequency, index } => {
                    let chart = match held {
                        Some((of, chart)) if of == frequency => chart,
                        _ => {
                            let f = &self.light.frequencies[frequency];
                            // The field and its derivatives from each beam (13 parts of 3 components), its chart from the
                            // components: counted once a frequency.
                            if !coordinates[..i].iter().any(|c| matches!(c, Coordinate::Chart { frequency: other, .. } if *other == frequency)) {
                                work += (f.beams.len() * 39 + 6 * f.rank() * f.rank()) as f64;
                            }
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
                    work += (4 * p.beams.len() + 4) as f64;
                    values[i] = (f.norm_sqr() / p.norm).min(1.0);
                    gradients[i] = std::array::from_fn(|a| 2.0 * (f.conj() * df[a]).re / p.norm);
                }
            }
        }
        work
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
    /// None where the finer grid is beyond `most` nodes.
    pub fn halved(&self, table: &NearTable, f: usize, most: usize) -> Option<Table> {
        let grid: Vec<Axis> = table.table.axes().iter().enumerate().map(|(g, a)| a.halved(if g == f { 1 } else { 0 })).collect();
        (Table::nodes(&grid) <= most).then(|| self.build(&table.layout, grid, Some(&table.table), most)).flatten()
    }

    /// A table's grid filled (its coordinates' axes the first, the others at their focus values); refined from `from` where given
    /// (its axes nested in these). None beyond `most` new nodes.
    fn build(&self, layout: &Layout, grid: Vec<Axis>, from: Option<&Table>, most: usize) -> Option<Table> {
        let d = grid.len();
        let f = |s: &[f64], out: &mut [f64]| {
            let mut x = layout.focus.clone();
            x[..d].copy_from_slice(s);
            self.node(layout, &x, out);
        };
        match from {
            None => Table::build(grid, layout.width(), most, &f).ok(),
            Some(t) => t.refine(grid, most, &f).ok(),
        }
    }

    /// The near tables for the atoms' reach (module documentation, # Axes, # Refinement), each grid of at most `most` nodes: per
    /// cluster the running lights' and, where the trap is lit and a carrier stays lit without it, those lights less the trap's --
    /// each light set under which the cluster has near forms --, refined over the reach; then the coordinates its first
    /// table's looks held are looked at once more on the refined one (module documentation, # Axes), and the table is made
    /// again with one that it needs after all. A coordinate the axes' rules refuse (`first_table`: the cost of an
    /// interpolation, the node bound, `table::MOST_AXES`) is held at its focus value like the light ones, whatever its
    /// weight: the table's whole error at the atoms reads it (`table_reads`). A table asks nothing of another: they are
    /// made side by side on the pool, and kept in the order of their clusters and light sets.
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
                    let mut table = self.refined(table, &ranges, reach, most);
                    // An axis is verified over the intervals that meet its range of its grid before the last halving,
                    // where the table before it was read against the elimination at the new nodes (# Refinement): of
                    // twice the spacing.
                    for (f, axis) in table.table.axes().to_vec().into_iter().enumerate() {
                        table.layout.verified[f] = table.layout.verified[f].map(|(lo, hi)| {
                            let h = 2.0 * axis.spacing();
                            let (below, above) = (((lo - axis.lo) / h).floor(), ((hi - axis.lo) / h).ceil());
                            ((axis.lo + below * h).max(axis.lo).min(lo), (axis.lo + above * h).min(axis.hi).max(hi))
                        });
                    }
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

    /// The range of a table's coordinate (an axis or one held) over the atoms' reach (module documentation, # Refinement).
    fn reach_range(&self, layout: &Layout, f: usize, reach: &Reach) -> (f64, f64) {
        let (coordinate, at) = (layout.coordinates[f], layout.focus[f]);
        let axis = first_axis(coordinate);
        if !intensity(coordinate) {
            return (axis.lo, axis.hi);
        }
        let quantity = |r: &Region| self.intensity_moved(coordinate, &layout.polarizations, at, r);
        let moved = self.extents_over(reach, axis.spacing(), &[&quantity])[0];
        ((at - moved).max(axis.lo), (at + moved).min(axis.hi))
    }

    /// A table's error by the two measures (`node_error`: the responses' relative error, the bound of the potential's) over
    /// the tables' shares (module documentation, # Refinement), the larger of the two: twice the responses' over the smaller
    /// of what is shown's and T's, the potential's over T's of k_BT_r. Within the shares: at most one.
    fn over(&self, (responses, potential): (f64, f64), reach: &Reach) -> f64 {
        let (shown, temperature) = shares();
        (2.0 * responses / shown.min(temperature)).max(potential / (temperature * crate::units::KT_PER_UK * reach.temperature_uk))
    }

    /// A table refined over the atoms' reach, its axes' ranges there given (module documentation, # Refinement).
    fn refined(&self, mut table: NearTable, ranges: &[(f64, f64)], reach: &Reach, most: usize) -> NearTable {
        let mut verified = vec![false; table.axes()];
        while let Some(f) = verified.iter().position(|v| !v) {
            let Some(finer) = self.halved(&table, f, most) else { break };
            verified[f] = self.over(self.halving_error(&table.layout, &table.table, &finer, f, ranges, reach), reach) <= 1.0;
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
        let mut weights = self.weights(cluster, lights, &all, &polarizations, &focus);
        // The weight a coordinate held at the focus may have: the responses' error the refinement leaves of an axis, half
        // the tables' shares (module documentation, # Axes). An intensity's weight per unit intensity times the bound of its
        // distance from the focus value over the reach.
        let (shown, temperature) = shares();
        let target = shown.min(temperature) / 2.0;
        let intensities: Vec<usize> = (0..all.len()).filter(|&c| weights[c] >= target && intensity(all[c])).collect();
        let moved = {
            let polarizations = &polarizations;
            let quantities: Vec<Box<dyn Fn(&Region) -> f64 + '_>> = intensities
                .iter()
                .map(|&c| {
                    let (coordinate, at, per) = (all[c], focus[c], weights[c]);
                    Box::new(move |r: &Region| per * self.intensity_moved(coordinate, polarizations, at, r)) as Box<dyn Fn(&Region) -> f64 + '_>
                })
                .collect();
            let refs: Vec<&dyn Fn(&Region) -> f64> = quantities.iter().map(|q| q.as_ref() as &dyn Fn(&Region) -> f64).collect();
            self.bounds_over(reach, target, target, &refs)
        };
        // Per coordinate the range it was verified over where the reach bounds it (`Layout::verified`): of one held by its
        // weight over the reach, the distance from the focus value that keeps its weight within the target.
        let mut verified: Vec<Option<(f64, f64)>> = vec![None; all.len()];
        for (&c, w) in intensities.iter().zip(moved) {
            if w < target {
                let span = target / weights[c];
                verified[c] = Some((focus[c] - span, focus[c] + span));
            }
            weights[c] = w;
        }
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
                verified: vec![None; picked.len()],
                polarizations: polarizations.clone(),
                plan: plan.clone(),
            };
            let table = self.build(&layout, axes.iter().map(|&c| first_axis(all[c])).collect(), None, most)?;
            Some(NearTable { layout, table })
        };
        // The axes: of the coordinates at or above that weight, the heaviest first, those the table so far needs by its look at
        // the coordinate's first nodes over the reach -- as many as keep an interpolation cheaper than the elimination and the
        // first grid within the bound.
        let (mut axes, mut ranges, mut held) = (vec![], vec![], vec![]);
        let mut table = made(&axes)?;
        for &c in &order {
            if weights[c] < target {
                break;
            }
            let d = axes.len() + 1;
            let first: Vec<Axis> = axes.iter().chain(std::iter::once(&c)).map(|&a| first_axis(all[a])).collect();
            if d > crate::table::MOST_AXES || (4usize.pow(d as u32) * (1 + d) * width) as f64 >= exact || Table::nodes(&first) > most {
                break;
            }
            let place = table.layout.coordinates.iter().position(|x| *x == all[c]).expect("a coordinate of the table's layout");
            let (range, axis) = (self.reach_range(&table.layout, place, reach), first[d - 1]);
            let values: Vec<f64> = (0..axis.nodes).map(|i| axis.node(i)).filter(|v| v - axis.spacing() <= range.1 && v + axis.spacing() >= range.0).collect();
            if intensity(all[c]) {
                verified[c] = Some(range);
            }
            let take = needed.contains(&all[c]) || {
                let (error, worst) = self.look(&table, &ranges, all[c], &values, reach);
                if error <= 1.0 {
                    held.push((all[c], worst));
                    // Held after a look: over the values looked at, its first grid's nodes within a spacing of the range.
                    if let (true, Some(&least), Some(&most)) = (intensity(all[c]), values.first(), values.last()) {
                        verified[c] = Some((least, most));
                    }
                }
                error > 1.0
            };
            if take {
                axes.push(c);
                ranges.push(range);
                table = made(&axes)?;
            }
        }
        // The intensities the axes' bounds left held unlooked: their range over the reach.
        for &c in &intensities {
            if verified[c].is_none() {
                let place = table.layout.coordinates.iter().position(|x| *x == all[c]).expect("a coordinate of the table's layout");
                verified[c] = Some(self.reach_range(&table.layout, place, reach));
            }
        }
        table.layout.verified = table.layout.coordinates.iter().map(|x| verified[all.iter().position(|y| y == x).expect("a coordinate of the light set")]).collect();
        Some(First { table, ranges, held })
    }

    /// The coordinates' a-priori weights for a cluster under a light set (module documentation, # Axes).
    fn weights(&self, cluster: usize, lights: &Lights, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>], focus: &[f64]) -> Vec<f64> {
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
                    // The largest change over the nodes; an intensity's per unit intensity (its shift linear in it).
                    let axis = first_axis(*coordinate);
                    let (mut most, mut span) = (0.0_f64, 0.0_f64);
                    for n in 0..axis.nodes {
                        let mut x = focus.to_vec();
                        x[i] = axis.node(n);
                        span = span.max((x[i] - focus[i]).abs());
                        most = most.max(moved(&energies(&x)));
                    }
                    if index == 0 && span > 0.0 {
                        most / span
                    } else {
                        most
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

    /// The bound over a region of an intensity coordinate's distance from its focus value `at` (module documentation, # Axes). The
    /// intensity is a field's squared modulus over N, the field F = Σ_l a_l c_l (a frequency's, c_l = E_l; a unit's, c_l = κ_l):
    /// |F|² = Σ_l'l a_l'* a_l c_l'†c_l, each beam pair's coefficient within its radius of its centre value over the region
    /// (reach.rs `PairSpan`) -- so |F|² lies within Σ |c_l'†c_l| times the pairs' radii of its centre value, at least 0 and at
    /// most every beam at its largest amplitude and in phase; on a point it is the value. A pair of beams whose fields are
    /// orthogonal is no term of it, and a plane wave's pair with itself does not move: light whose intensity is the same
    /// everywhere weighs nothing, on any cell. (Bounded by the field's modulus within the region's radius times Σ_l |c_l|
    /// (|k_l − k₀| max|b_l| + max|∇b_l|) of its centre value, the intensity of two counter-propagating plane waves of
    /// orthogonal polarizations -- a constant -- was taken to move by 2kρ: with the lin-perp-lin scene's cooling light on its
    /// chart no cell of the reach was small enough, the proof had cut 8 million cells in 43 s and went on.)
    fn intensity_moved(&self, c: Coordinate, polarizations: &[Vec<Polarization>], at: f64, region: &Region) -> f64 {
        let (lo, hi) = self.intensity_over(c, polarizations, region);
        (hi - at).abs().max((at - lo).abs())
    }

    /// The bounds over a region of an intensity coordinate (`intensity_moved`): its least and its largest value there.
    fn intensity_over(&self, c: Coordinate, polarizations: &[Vec<Polarization>], region: &Region) -> (f64, f64) {
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
        let (mut centre, mut radius) = (0.0, 0.0);
        for (j, to) in &beams {
            for (i, from) in &beams {
                let weight: C64 = (0..3).map(|q| to[q].conj() * from[q]).sum();
                let pair = &region.pairs[frequency][j * f.beams.len() + i];
                centre += (pair.centre * weight).re;
                radius += pair.radius * weight.abs();
            }
        }
        let (lo, hi) = ((centre - radius).max(0.0), (centre + radius).min(blind * blind));
        (lo / norm, hi / norm)
    }

    /// Whether the tables hold over a region as they were verified over the reach they were made for (module
    /// documentation, # Axes, # Refinement): every intensity coordinate whose verification the reach bounded
    /// (`Layout::verified`) keeps within its range over the region (`intensity_over`).
    pub(crate) fn within(&self, tables: &Tables, region: &Region) -> bool {
        tables.tables.iter().all(|t| {
            let layout = &t.layout;
            layout.coordinates.iter().zip(&layout.verified).all(|(&c, range)| {
                range.is_none_or(|(lo, hi)| {
                    let (least, most) = self.intensity_over(c, &layout.polarizations, region);
                    lo <= least && most <= hi
                })
            })
        })
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
            work += (4usize.pow(d as u32) * (1 + if *order >= 1 { d } else { 0 }) * w) as f64 / crate::decide::COMPLEX;
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
        work + (4usize.pow(d as u32) * (1 + slopes) * 2 * plan.near.jumping().len()) as f64 / crate::decide::COMPLEX
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
        let (place, mut work) = point.place.expect("a site placed on its table");
        let (first, generator, forms) = (2 * plan.near.len(), plan.generator_start.len(), plan.form_start.len());
        sized(&mut point.generator, generator);
        sized(&mut point.forms, forms);
        sized(&mut point.slopes, d * forms);
        table.table.read(&place, first, Some(&plan.generator_start), &mut point.generator, None);
        table.table.read(&place, first + generator, Some(&plan.form_start), &mut point.forms, Some(&mut point.slopes));
        for (entries, order) in [(plan.near.hermitian(), 1), (plan.near.jumping(), 0)] {
            work += (4usize.pow(d as u32) * (1 + if order >= 1 { d } else { 0 }) * 2 * entries.len()) as f64 / crate::decide::COMPLEX;
        }
        work
    }

    /// The tables' error reads at samples (module documentation; SPEC §2.3, §5.4): per table, per axis the read of its
    /// interpolation error (crate::table `Table::axis_error`) and the whole error (the exact elimination against the table's), each
    /// as the error it makes of the sample's generator (`form_error`).
    pub fn table_reads(&self, tables: &Tables, samples: &[Sample]) -> Vec<TableRead> {
        let zero = UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
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
        UnitError { parameter: delta, population: shown, temperature: shown.max(potential / thermal) }
    }
}

/// Whether a coordinate is an intensity: a chart's first, or a unit's.
fn intensity(c: Coordinate) -> bool {
    matches!(c, Coordinate::Chart { index: 0, .. } | Coordinate::Unit { .. })
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
        let wanted = 0.04 * 0.5 * gamma * moved.signum();
        for q in 0..3 {
            request.outside_forms[tweezer.form].matrices[q][(e, e)] += C64::real((wanted - moved) / field);
        }
        for _ in 0..3 {
            let (detuning, _, _) = line(&request);
            request.hamiltonian[(e, e)] -= C64::real(detuning);
        }
        let (detuning, moved, _) = line(&request);
        assert!(detuning.abs() <= 1e-9 * gamma && (moved.abs() / (0.5 * gamma) - 0.04).abs() < 0.002, "on resonance at the focus, moved by 0.04 half-widths: {detuning}, {moved}");
        let model = Model::new(&request, &light, &kept).unwrap();
        let (trap, reach) = reach_of(&model, b, 4.0 * request.temperature_uk);
        let tables = model.tabulate(&reach, 1 << 16);
        let cluster = model.cluster_of(b);
        let index = tables.tables.iter().position(|t| t.layout.cluster == cluster && t.layout.lights == model.running()).unwrap();
        let table = &tables.tables[index];
        assert!(table.axes() == 1 && table.layout.coordinates[0] == (Coordinate::Chart { frequency: light.trap, index: 0 }), "the tweezer's intensity is the axis: {:?}", &table.layout.coordinates[..table.axes()]);
        let (p, prepared) = (std::array::from_fn(|s| if s == b { 1.0 } else { 0.0 }), model.prepared());
        let edge = (1..).map(|i| 0.01 * i as f64).find(|&rho| model.potential([trap.centre[0] + rho, trap.centre[1], trap.centre[2]], &p, &prepared) > reach.minimum + reach.energy).unwrap();
        let samples: Vec<Sample> = (0..40).map(|i| Sample { cluster, at: Point { r: [trap.centre[0] + edge * (i as f64 + 0.5) / 40.0, trap.centre[1], trap.centre[2]], kappa: [0.0; 3], t: 0.0 }, psi: vec![C64::ONE] }).collect();
        let reads = &model.table_reads(&tables, &samples)[index];
        let largest = reads.whole.iter().map(|r| r.population.max(r.temperature)).fold(0.0, f64::max);
        let (shown, temperature) = shares();
        eprintln!("the probe in the nearly magic tweezer: {} nodes; over {edge:.2} µm the table's largest read {largest:.2e}", Table::nodes(table.table.axes()));
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
}
