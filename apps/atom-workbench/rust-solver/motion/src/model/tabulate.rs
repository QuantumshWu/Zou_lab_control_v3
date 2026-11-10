//! The near light's tables (SPEC §2.3): per cluster and light set the elimination's nonlinear part -- the cluster's near forms
//! (assemble.rs `NearLayout`: the shifts S and their Doppler partners, the jumps, the explicit states' van Vleck shifts, the
//! responses) -- tabulated over the coordinates it depends on, the linear part (the field's components, the Doppler shifts) exact.
//! The light sets the integrator evaluates: every light on (the generator, the photons, the readout), and every light but the
//! trap's (V_run's subtraction, potential.rs).
//!
//! # Coordinates
//! The elimination at a point depends on the point through the far light's H, linear in each lit frequency's coherence T_ω and
//! so a function of its chart (field.rs # The chart: intensity, polarization angles, relative phases), and through the near
//! carriers' dressings and self-energies: with the units' T̄ (eliminate.rs) on each polarization unit's intensity
//! I_u = |f_u|²/N_u, f_u = Σ_{l∈u} a_l κ_l the unit's field along its polarization ε_u (E_l = κ_l ε_u), N_u = (Σ_{l∈u} |κ_l|)² so
//! that I_u ∈ [0, 1] (|a_l| ≤ 1: a computed I_u above 1 is its rounding, and is 1); with a carrier's coherent dressing
//! (`Unit::Dressing` kept) on its chart. Every coordinate on its whole range: every value is some field's, no atom meets an
//! edge, and a table is never asked off its axes (crate::table refuses it).
//!
//! # A node
//! At coordinates s the elimination is the one in a light whose fields have them: each lit frequency's field from its chart
//! (field.rs `Frequency::chart_field`), its beams' amplitudes the least-norm ones that give it, a_l = Σ_k γ_lk* e_k/σ_k² (the
//! far forms depend on the field alone, far.rs); each unit's field on its first beam, a = √(I_u N_u)/κ₁. The coordinates that
//! are no axes take their values at the focus.
//!
//! # Axes
//! A coordinate's a-priori weight is how far it moves the cluster's lines over the atoms' reach, in pole half-widths: for a chart
//! coordinate the largest change of E_e − E_b (e ← b a carrier's coupling from the cluster's ground states) over its first grid's
//! nodes, the other coordinates at the focus, over Γ_e/2; for a unit, its dressing at full intensity over Γ_min/2 (its resolvent's
//! Varah bound, eliminate.rs). An intensity's shift is linear in it, so its weight is that per unit intensity times the bound of
//! its distance from its focus value over the reach (reach.rs `Model::bounds_over`): a plane wave's intensity, the same everywhere,
//! weighs nothing; an angle or a phase weighs its whole range. A response is a pole in E_e − E_b, its relative change at most the
//! weight: a coordinate whose weight is below the population target leaves the responses within it a priori. The axes are the
//! heaviest coordinates above it, as many as keep one interpolation cheaper than the elimination it replaces and the first grid
//! within the work bound -- none where no coordinate reaches the target: the forms at the focus alone --; the others are held at
//! the focus, the table's whole error read at the atoms (`Model::table_reads`).
//!
//! # At a point
//! The axes' coordinates and their gradients at the point (field.rs `Frequency::chart`; ∇I_u = 2 Re(f_u* ∇f_u)/N_u), the table's
//! interpolant of the forms asked and, to first order, its derivatives along the axes, ∇v = Σ_f ∂v/∂s_f ∇s_f.
//!
//! # The table's error
//! A table's forms are ṽ(s) = v(s) + e(s), the elimination's and the interpolation's error, C¹ in the coordinates. The generator
//! they give is a generator in its own right: its force is the gradient of its own Hermitian part (the interpolant's own
//! derivative) and its decay its own jumps' product. So the table stands for the model whose forms are ṽ, and its error on the
//! outputs is that model's distance from the elimination's: the rates and the couplings of a state moved by the relative δ =
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
//! by its own measured error where an interpolant is worst, between its nodes. Halving an axis' spacing (crate::table
//! `Axis::halved`) puts the new nodes at the middles of its intervals: there the table before the halving is compared with the
//! elimination itself -- the new nodes' own values, so the check costs nothing beyond the finer table --, over every interval the
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
//! A node costs its elimination (eliminate.rs `Responses::work`) and its forms; a point the table's interpolation of the reals
//! asked, 4^D (1 + D) each on D axes (4^D for a value alone), its chain rule (3 D each) and the coordinates (each frequency's
//! field and chart from its beams).

use super::assemble::{NearValues, Plan};
use super::reach::Region;
use super::{Lights, Lit, Model, Reach, Sample, UnitError};
use crate::field::{range, Amplitude, Chart};
use crate::linalg::{norm3, Mat, C64};
use crate::request::{GROUND, STATES};
use crate::table::{Axis, Table};
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
/// a-priori weights and their values at the focus, its carriers' units, and the cluster's plan under the lights (its near forms).
#[derive(Clone, Debug)]
pub struct Layout {
    pub cluster: usize,
    pub lights: Lights,
    pub coordinates: Vec<Coordinate>,
    pub weights: Vec<f64>,
    focus: Vec<f64>,
    polarizations: Vec<Vec<Polarization>>,
    plan: Arc<Plan>,
}

/// A near table: its layout and the table of the near forms (real and imaginary parts, in the layout's order).
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

/// The near tables of a model (module documentation), the complex multiply-adds their nodes took, and k_BT_r of the reach they
/// were made for (rad/µs).
#[derive(Clone, Debug)]
pub struct Tables {
    pub tables: Vec<NearTable>,
    pub work: f64,
    pub thermal: f64,
}

/// The tables' shares of the targets (module documentation, # Refinement): on what is shown, absolute, and on T, relative.
fn shares() -> (f64, f64) {
    let (z, n) = (crate::units::CONFIDENCE, crate::units::MOST_TRAJECTORIES as f64);
    (z / (2.0 * n.sqrt()), z * (2.0 / n).sqrt())
}

/// A table's error reads at samples (`Model::table_reads`): per axis the read of its interpolation error, and the table's whole
/// error against the exact elimination (its coordinates held at the focus moving, its interpolation), each as the error it
/// makes on the outputs through the sample's generator (module documentation, # The table's error: δ in `parameter`, δ P_g on
/// the populations, the larger of δ P_g and |ΔV|/k_BT_r on T); zero on the samples of other clusters.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRead {
    pub table: usize,
    pub axes: Vec<Vec<UnitError>>,
    pub whole: Vec<UnitError>,
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

    /// Every coordinate an elimination under a light set depends on (module documentation, # Coordinates): every lit frequency's
    /// chart, and the units of each lit carrier whose dressing takes its units' T̄.
    fn coordinates(&self, lights: &Lights, polarizations: &[Vec<Polarization>]) -> Vec<Coordinate> {
        let mut out = vec![];
        for (frequency, f) in self.light.frequencies.iter().enumerate() {
            if lights.on(frequency) {
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

    /// Coordinates' values at a point and their gradients (module documentation, # At a point), and the complex multiply-adds.
    fn coordinates_at(&self, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>], amplitudes: &[Amplitude]) -> (Vec<f64>, Vec<[f64; 3]>, f64) {
        let mut charts: Vec<Option<Chart>> = vec![None; self.light.frequencies.len()];
        let (mut values, mut gradients, mut work) = (vec![], vec![], 0.0);
        for c in coordinates {
            match *c {
                Coordinate::Chart { frequency, index } => {
                    let chart = charts[frequency].get_or_insert_with(|| {
                        let f = &self.light.frequencies[frequency];
                        // The field and its derivatives from each beam (13 parts of 3 components), its chart from the components.
                        work += (f.beams.len() * 39 + 6 * f.rank() * f.rank()) as f64;
                        f.chart(&f.field(amplitudes))
                    });
                    values.push(chart.value[index]);
                    gradients.push(chart.gradient[index]);
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
                    values.push((f.norm_sqr() / p.norm).min(1.0));
                    gradients.push(std::array::from_fn(|i| 2.0 * (f.conj() * df[i]).re / p.norm));
                }
            }
        }
        (values, gradients, work)
    }

    /// The light at coordinate values x (module documentation, # A node): the far beams' amplitudes from each lit frequency's chart,
    /// the near beams' from their units' intensities (a carrier with a coherent dressing: from its chart).
    fn light_at(&self, lights: &Lights, coordinates: &[Coordinate], polarizations: &[Vec<Polarization>], x: &[f64]) -> (Vec<Amplitude>, Vec<Amplitude>) {
        let mut far = vec![dark(); self.light.beams.len()];
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
                far[l].value = components.iter().enumerate().map(|(k, &e)| f.dual(j, k) * e).sum();
            }
        }
        let mut near = far.clone();
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
        (far, near)
    }

    /// The near forms at coordinate values x into a node's reals, and the complex multiply-adds that made them.
    fn node(&self, layout: &Layout, x: &[f64], out: &mut [f64]) -> f64 {
        let lights = &layout.lights;
        let (far, near) = self.light_at(lights, &layout.coordinates, &layout.polarizations, x);
        let h = self.far_hamiltonian(&far, lights, 0);
        let r = self.eliminate(layout.cluster, &h, &near, lights, 0, &self.coherent_dressing());
        let energies = vec![(0..STATES).map(|s| h[0][(s, s)].re).collect::<Vec<f64>>()];
        let mut values = vec![C64::ZERO; layout.plan.near.len()];
        let work = self.forms(&layout.plan, &r, &energies, &mut values);
        for (pair, z) in out.chunks_mut(2).zip(&values) {
            pair.copy_from_slice(&[z.re, z.im]);
        }
        r.work + self.far_work(lights, 0) + work
    }

    /// The complex multiply-adds of one node (the same at every node: the elimination's loops do not depend on the values).
    fn node_work(&self, layout: &Layout) -> f64 {
        let mut out = vec![0.0; 2 * layout.plan.near.len()];
        self.node(layout, &layout.focus, &mut out)
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
            None => Table::build(grid, 2 * layout.plan.near.len(), most, &f).ok(),
            Some(t) => t.refine(grid, most, &f).ok(),
        }
    }

    /// The near tables for the atoms' reach (module documentation, # Axes, # Refinement), each grid of at most `most` nodes: per
    /// cluster the running lights' and, where the trap is lit and a carrier stays lit without it, those lights less the trap's --
    /// each light set under which the cluster has near forms --, refined over the reach. A cluster whose first grid exceeds the
    /// bound keeps the exact elimination.
    pub fn tabulate(&self, reach: &Reach, most: usize) -> Tables {
        let running = self.running();
        let mut untrapped = running.clone();
        untrapped.0[self.light.trap] = Lit::Off;
        let mut out = Tables { tables: vec![], work: 0.0, thermal: crate::units::KT_PER_UK * reach.temperature_uk };
        for cluster in 0..self.clusters.len() {
            for lights in [&running, &untrapped] {
                if lights == &untrapped && !running.on(self.light.trap) {
                    continue;
                }
                if let Some(first) = self.first_table(cluster, lights, reach, most) {
                    let nodes = Table::nodes(first.table.axes());
                    let (table, more) = self.refined(first, reach, most);
                    out.work += nodes as f64 * self.node_work(&table.layout) + more;
                    out.tables.push(table);
                }
            }
        }
        out
    }

    /// The range of a table's axis over the atoms' reach (module documentation, # Refinement).
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

    /// A table refined over the atoms' reach (module documentation, # Refinement), and the complex multiply-adds of its new
    /// nodes.
    fn refined(&self, mut table: NearTable, reach: &Reach, most: usize) -> (NearTable, f64) {
        let d = table.axes();
        let ranges: Vec<(f64, f64)> = (0..d).map(|f| self.reach_range(&table.layout, f, reach)).collect();
        let (shown, temperature) = shares();
        let thermal = crate::units::KT_PER_UK * reach.temperature_uk;
        let node = self.node_work(&table.layout);
        let mut work = 0.0;
        let mut verified = vec![false; d];
        while let Some(f) = verified.iter().position(|v| !v) {
            let Some(finer) = self.halved(&table, f, most) else { break };
            work += table.table.new_nodes(finer.axes()) as f64 * node;
            let (responses, potential) = self.halving_error(&table.layout, &table.table, &finer, f, &ranges, reach);
            verified[f] = 2.0 * responses <= shown.min(temperature) && potential <= temperature * thermal;
            table.table = finer;
        }
        (table, work)
    }

    /// The error of a table against the one a halving of axis f finer, at the finer's new nodes over the reach's ranges (module
    /// documentation, # Refinement): the responses' relative error and the bound of the potential's error (rad/µs), each the
    /// largest over the nodes; +∞ where one is no number.
    fn halving_error(&self, layout: &Layout, coarse: &Table, fine: &Table, f: usize, ranges: &[(f64, f64)], reach: &Reach) -> (f64, f64) {
        let plan = &layout.plan;
        let (start, count, pairs) = (plan.near.responding().start, plan.near.responses.len(), plan.near.pairs.len());
        // Per carrier the largest Doppler shift of its beams at the reach's speed; a field component's peak.
        let shift: Vec<f64> = self.carriers.iter().map(|cc| cc.beams.iter().map(|&l| self.doppler_shift(l, [0.0; 3]) + self.light.beams[l].wave_number * reach.speed).fold(0.0, f64::max)).collect();
        let peak = |carrier: usize, k: usize| self.light.frequencies[self.carriers[carrier].frequency].coefficients.iter().map(|c| c[k].abs()).sum::<f64>();
        let ground = self.clusters[layout.cluster].ground;
        let complex = |row: &[f64], at: usize| C64::new(row[2 * at], row[2 * at + 1]);
        let mut interpolated = vec![0.0; fine.width()];
        let (mut responses, mut potential) = (0.0_f64, 0.0_f64);
        let read = |x: f64| if x.is_nan() { f64::INFINITY } else { x };
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
            let error = |at: usize| (complex(&interpolated, at) - complex(exact, at)).norm_sqr();
            for (c, w) in shift.iter().enumerate() {
                let sum = |of: &dyn Fn(usize) -> f64| (0..count).filter(|&i| plan.near.responses[i][0] == c).map(of).sum::<f64>().sqrt();
                let size = sum(&|i| complex(exact, start + i).norm_sqr());
                if !(size == 0.0) {
                    responses = responses.max(read((sum(&|i| error(start + i)) + w * sum(&|i| error(start + count + i))) / size));
                }
            }
            for x in 0..ground {
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
        }
        (responses, potential)
    }

    /// A cluster's first table for a light set (module documentation, # Axes); None where it has no near form under it, or where
    /// its grid exceeds `most` nodes.
    fn first_table(&self, cluster: usize, lights: &Lights, reach: &Reach, most: usize) -> Option<NearTable> {
        let plan = self.plan(cluster, lights);
        if plan.near.len() == 0 {
            return None;
        }
        let coherent = self.coherent_dressing();
        let polarizations: Vec<Vec<Polarization>> = (0..self.carriers.len()).map(|d| if coherent[d] { vec![] } else { self.polarizations(d) }).collect();
        let all = self.coordinates(lights, &polarizations);
        let focus_amplitudes = self.light.amplitudes([0.0; 3]);
        let (focus, _, _) = self.coordinates_at(&all, &polarizations, &focus_amplitudes);
        let mut weights = self.weights(cluster, lights, &all, &polarizations, &focus);
        // An intensity's weight per unit intensity times the bound of its distance from the focus value over the reach.
        let target = crate::units::POPULATION_TARGET;
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
            self.bounds_over(reach, target, &refs)
        };
        for (&c, w) in intensities.iter().zip(moved) {
            weights[c] = w;
        }
        let mut order: Vec<usize> = (0..all.len()).collect();
        order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
        // The axes: the heaviest coordinates at or above the population target, as many as keep an interpolation cheaper than the
        // elimination and the first grid within the bound.
        let width = 2 * plan.near.len();
        let exact = self.near_exact(&plan, &focus_amplitudes, 1, &coherent).work * crate::decide::COMPLEX;
        let mut axes: Vec<usize> = vec![];
        for &c in &order {
            if weights[c] < target {
                break;
            }
            let d = axes.len() + 1;
            let first: Vec<Axis> = axes.iter().chain(std::iter::once(&c)).map(|&a| first_axis(all[a])).collect();
            if (4usize.pow(d as u32) * (1 + d) * width) as f64 >= exact || Table::nodes(&first) > most {
                break;
            }
            axes.push(c);
        }
        let picked: Vec<usize> = axes.iter().copied().chain(order.iter().copied().filter(|c| !axes.contains(c))).collect();
        let grid: Vec<Axis> = axes.iter().map(|&c| first_axis(all[c])).collect();
        let layout = Layout {
            cluster,
            lights: lights.clone(),
            coordinates: picked.iter().map(|&c| all[c]).collect(),
            weights: picked.iter().map(|&c| weights[c]).collect(),
            focus: picked.iter().map(|&c| focus[c]).collect(),
            polarizations,
            plan,
        };
        let table = self.build(&layout, grid, None, most)?;
        Some(NearTable { layout, table })
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
        let energies = |x: &[f64]| -> Vec<f64> {
            let (far, _) = self.light_at(lights, coordinates, polarizations, x);
            self.far_energies(&far, lights, 0).0.swap_remove(0)
        };
        let at_focus = energies(focus);
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
                        let e = energies(&x);
                        for &(a, b, half) in &lines {
                            most = most.max(((e[a] - e[b]) - (at_focus[a] - at_focus[b])).abs() / half);
                        }
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
                    (0..GROUND)
                        .map(|b| p.norm * (GROUND..STATES).map(|e| x[(e, b)].norm_sqr()).sum::<f64>() / k * self.resolvent(carrier, b, &region))
                        .fold(0.0, f64::max)
                        / (self.least_decay() / 2.0)
                }
            })
            .collect()
    }

    /// The bound over a region of an intensity coordinate's distance from its focus value `at` (module documentation, # Axes). The
    /// intensity is a field's squared modulus, |F|²/N with F = Σ_l a_l c_l (a frequency's field, c_l = E_l; a unit's, c_l = κ_l), and a
    /// modulus does not see a phase common to its beams: with a_l = e^{ik_l·r} b_l (reach.rs `Span`), |F| = |Σ_l e^{i(k_l − k₀)·r} b_l c_l|
    /// for any k₀, so |∇|F|| ≤ Σ_l |c_l| (|k_l − k₀| max|b_l| + max|∇b_l|) -- the least over the beams' own wave vectors as k₀. (Bounded
    /// by Σ_l |c_l| max|∇a_l| instead, a plane wave's intensity, the same everywhere, was taken to move by 2kρ over a cell of radius
    /// ρ, and no cell of the reach was ever small enough for a weight of a hundred half-widths: R2_RECOIL's probe.) |F| lies within
    /// the region's radius times that of its centre value, at most every beam at its largest amplitude; on a point it is the value.
    fn intensity_moved(&self, c: Coordinate, polarizations: &[Vec<Polarization>], at: f64, region: &Region) -> f64 {
        // The beams of the coordinate's field with |c_l|, its centre value's modulus, and N.
        let (beams, centre, norm): (Vec<(&super::reach::Span, f64)>, f64, f64) = match c {
            Coordinate::Chart { frequency, .. } => {
                let f = &self.light.frequencies[frequency];
                let mut e = [C64::ZERO; 3];
                let mut beams = vec![];
                for (&l, field) in f.beams.iter().zip(&f.fields) {
                    let span = &region.spans[l];
                    for q in 0..3 {
                        e[q] += span.centre * field[q];
                    }
                    beams.push((span, field.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()));
                }
                (beams, e.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(), f.norm)
            }
            Coordinate::Unit { carrier, unit } => {
                let p = &polarizations[carrier][unit];
                let cc = &self.carriers[carrier];
                let f: C64 = p.beams.iter().zip(&p.kappa).map(|(&l, &k)| region.spans[cc.beams[l]].centre * k).sum();
                (p.beams.iter().zip(&p.kappa).map(|(&l, k)| (&region.spans[cc.beams[l]], k.abs())).collect(), f.abs(), p.norm)
            }
        };
        let blind: f64 = beams.iter().map(|(span, size)| span.most * size).sum();
        let slope = beams
            .iter()
            .map(|(first, _)| beams.iter().map(|(span, size)| size * (norm3(std::array::from_fn(|i| span.wave[i] - first.wave[i])) * span.most + span.slope)).sum::<f64>())
            .fold(f64::INFINITY, f64::min);
        let (lo, hi) = if region.rho.is_finite() { ((centre - region.rho * slope).max(0.0), (centre + region.rho * slope).min(blind)) } else { (0.0, blind) };
        (hi * hi / norm - at).abs().max((at - lo * lo / norm).abs())
    }

    /// The near forms at a point from a table (module documentation, # At a point), the sections asked each to its order (0 or
    /// 1): their values and, to first order, their slopes along the axes with the coordinates' gradients.
    pub(crate) fn near_tabulated(&self, table: &NearTable, amplitudes: &[Amplitude], sections: &[(Range<usize>, usize)]) -> NearValues {
        let d = table.axes();
        let (x, gradients, mut work) = self.coordinates_at(&table.layout.coordinates[..d], &table.layout.polarizations, amplitudes);
        let len = sections.iter().map(|s| s.0.end).max().unwrap_or(0);
        let order = sections.iter().map(|s| s.1).max().unwrap_or(0);
        let mut data = vec![C64::ZERO; (1 + if order >= 1 { d } else { 0 }) * len];
        for (entries, order) in sections {
            let w = 2 * entries.len();
            let (mut value, mut slope) = (vec![0.0; w], vec![0.0; if *order >= 1 { d * w } else { 0 }]);
            table.table.eval_range(&x, 2 * entries.start, &mut value, if *order >= 1 { Some(&mut slope) } else { None });
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

    /// The tables' error reads at samples (module documentation; SPEC §2.3, §5.4): per table, per axis the read of its
    /// interpolation error (crate::table `Table::axis_error`) and the whole error (the exact elimination against the table's), each
    /// as the error it makes of the sample's generator (`form_error`).
    pub fn table_reads(&self, tables: &Tables, samples: &[Sample]) -> Vec<TableRead> {
        let zero = UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
        tables
            .tables
            .iter()
            .enumerate()
            .map(|(index, table)| {
                let (d, plan) = (table.axes(), &table.layout.plan);
                let (len, w) = (plan.near.len(), table.table.width());
                let mut axes = vec![vec![zero; samples.len()]; d];
                let mut whole = vec![zero; samples.len()];
                for (k, sample) in samples.iter().enumerate() {
                    if sample.cluster != table.layout.cluster {
                        continue;
                    }
                    let amplitudes = self.light.amplitudes(sample.at.r);
                    let (x, _, _) = self.coordinates_at(&table.layout.coordinates[..d], &table.layout.polarizations, &amplitudes);
                    let tabulated = self.near_tabulated(table, &amplitudes, &[(0..len, 1)]);
                    for (f, read) in axes.iter_mut().enumerate() {
                        // The axis' read of the reals and of their slopes, on the table's forms.
                        let (mut dv, mut ds) = (vec![0.0; w], vec![0.0; d * w]);
                        table.table.axis_error(f, &x, &mut dv, &mut ds);
                        let mut moved = tabulated.clone();
                        for e in 0..len {
                            moved.data[e] += C64::new(dv[2 * e], dv[2 * e + 1]);
                            for g in 0..d {
                                moved.data[(1 + g) * len + e] += C64::new(ds[g * w + 2 * e], ds[g * w + 2 * e + 1]);
                            }
                        }
                        read[k] = self.form_error(sample, plan, &tabulated, &moved, &amplitudes, tables.thermal);
                    }
                    let exact = self.near_exact(plan, &amplitudes, 1, &self.coherent_dressing());
                    whole[k] = self.form_error(sample, plan, &exact, &tabulated, &amplitudes, tables.thermal);
                }
                TableRead { table: index, axes, whole }
            })
            .collect()
    }

    /// The error on the outputs of a sample from an error of its near forms (module documentation, # The table's error), the
    /// forms `other` against `reference`: with N the near light's part of the generator at the sample's point and momentum --
    /// every term a near form enters, summed over its parts with their phases there --, δ = ‖(N′ − N)ψ‖/‖Nψ‖, and ΔV = ⟨ψ|H′_h −
    /// H_h|ψ⟩/‖ψ‖²; `thermal` is k_BT_r.
    fn form_error(&self, sample: &Sample, plan: &Arc<Plan>, reference: &NearValues, other: &NearValues, amplitudes: &[Amplitude], thermal: f64) -> UnitError {
        let n = self.clusters[sample.cluster].states.len();
        let action = |near: Option<&NearValues>| -> Vec<C64> {
            let middle = self.middle_of(plan, sample.at.t, plan.amplitudes(amplitudes), near);
            let mut total = Mat::zeros(n, n);
            crate::step::Middle::generator(&middle, &crate::step::Middle::path(&middle, sample.at.kappa, [0.0; 3]), 0.0, &mut total);
            total.apply(&sample.psi)
        };
        let (far, first, second) = (action(None), action(Some(reference)), action(Some(other)));
        let moved: Vec<C64> = second.iter().zip(&first).map(|(a, b)| *a - *b).collect();
        let size = first.iter().zip(&far).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>().sqrt();
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
