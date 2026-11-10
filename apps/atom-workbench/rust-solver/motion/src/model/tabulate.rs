//! The near light's tables (SPEC §2.3): per cluster and light set the elimination's nonlinear part -- its responses T_l and T2_l
//! on the cluster's ground places (eliminate.rs `Responses`) -- tabulated over the coordinates it depends on, the linear part
//! exact. The light sets the integrator evaluates: every light on (the generator, the photons, the readout), and every light but
//! the trap's (V_run's subtraction, potential.rs).
//!
//! # Coordinates
//! The elimination at a point depends on the point through the far light's H, linear in each lit frequency's coherence T_ω and
//! so a function of its chart (field.rs # The chart: intensity, polarization angles, relative phases), and through the near
//! carriers' dressings and self-energies: with the units' T̄ (eliminate.rs) on each polarization unit's intensity
//! I_u = |f_u|²/N_u, f_u = Σ_{l∈u} a_l κ_l the unit's field along its polarization ε_u (E_l = κ_l ε_u), N_u = (Σ_{l∈u} |κ_l|)² so
//! that I_u ∈ [0, 1]; with a carrier's coherent dressing (`Unit::Dressing` kept) on its chart. Every coordinate on its whole
//! range: every value is some field's, and no atom meets an edge.
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
//! within the work bound; the others are held at the focus, the table's whole error read at the atoms (`Model::table_reads`).
//!
//! # At a point
//! The axes' coordinates and their gradients at the point (field.rs `Frequency::chart`; ∇I_u = 2 Re(f_u* ∇f_u)/N_u), the table's
//! interpolant and its derivatives along the axes, ∇T = Σ_f ∂T/∂s_f ∇s_f; then the Doppler term at the point's momentum and the
//! far energies exact (eliminate.rs `Model::doppler`).
//!
//! # Work
//! A node costs its elimination (eliminate.rs `Responses::work`); a point the table's interpolation, 4^D (1 + D) w real
//! multiply-adds on w reals per node and D axes (4^D w for a value alone), its chain rule (3 D w), the coordinates (each
//! frequency's field and chart from its beams) and the far energies.

use super::eliminate::{NearAt, Responses};
use super::far::EXCITED;
use super::reach::Region;
use super::{Lights, Lit, Model, Point, Reach, Sample, UnitError};
use crate::field::{range, Amplitude, Chart};
use crate::linalg::{Mat, C64};
use crate::request::{GROUND, STATES};
use crate::table::{Axis, Table};

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
/// a-priori weights and their values at the focus, and its carriers' units.
#[derive(Clone, Debug)]
pub struct Layout {
    pub cluster: usize,
    pub lights: Lights,
    pub coordinates: Vec<Coordinate>,
    pub weights: Vec<f64>,
    focus: Vec<f64>,
    polarizations: Vec<Vec<Polarization>>,
}

/// A near table: its layout and the table of the responses (per lit carrier, beam, ground place and excited state T and T2, real
/// and imaginary parts).
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

/// The near tables of a model (module documentation), and the complex multiply-adds their nodes took.
#[derive(Clone, Debug)]
pub struct Tables {
    pub tables: Vec<NearTable>,
    pub work: f64,
}

/// A table's error reads at samples (`Model::table_reads`): per axis the read of its interpolation error, and the table's whole
/// error against the exact elimination (its coordinates held at the focus moving, its interpolation), per sample as the error of
/// the sample's rates (a dressing's map, eliminate.rs `Model::dressing_error`: the relative error δ of its state's rates, δ P_g on
/// the populations, the larger of δ and its gradient's on T); zero on the samples of other clusters.
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

    /// Table `index` refined, its axis f's spacing halved `halvings[f]` times (crate::table `Axis::halved`), its nodes reused, the
    /// complex multiply-adds of its new nodes returned: Err(the new nodes) beyond `most`, nothing built.
    pub fn refine(&mut self, model: &Model, index: usize, halvings: &[u32], most: usize) -> Result<f64, usize> {
        let t = &self.tables[index];
        let grid: Vec<Axis> = t.table.axes().iter().zip(halvings).map(|(a, &k)| a.halved(k)).collect();
        let fresh = t.table.new_nodes(&grid);
        let table = model.build(&t.layout, grid, Some(&t.table), most).ok_or(fresh)?;
        let work = fresh as f64 * model.node_work(&t.layout);
        self.tables[index].table = table;
        self.work += work;
        Ok(work)
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
                    values.push(f.norm_sqr() / p.norm);
                    gradients.push(std::array::from_fn(|i| 2.0 * (f.conj() * df[i]).re / p.norm));
                }
            }
        }
        (values, gradients, work)
    }

    /// The reals per node of a cluster's table under a light set: per lit carrier, beam, ground place and excited state, T and T2's
    /// real and imaginary parts.
    fn table_width(&self, cluster: usize, lights: &Lights) -> usize {
        let g = self.clusters[cluster].ground;
        self.carriers.iter().filter(|c| lights.on(c.frequency)).map(|c| c.beams.len() * g * EXCITED * 4).sum()
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

    /// The responses at coordinate values x into a node's reals, and the elimination's complex multiply-adds.
    fn node(&self, layout: &Layout, x: &[f64], out: &mut [f64]) -> f64 {
        let lights = &layout.lights;
        let (far, near) = self.light_at(lights, &layout.coordinates, &layout.polarizations, x);
        let h = self.far_hamiltonian(&far, lights, 0);
        let r = self.eliminate(layout.cluster, &h, &near, lights, 0, &self.coherent_dressing());
        let mut at = 0;
        for (first, second) in r.first.iter().zip(&r.second) {
            for (t, t2) in first.iter().zip(second) {
                let (t, t2) = (&t[0], &t2[0]);
                for pb in 0..t.cols {
                    for e in 0..EXCITED {
                        out[at..at + 4].copy_from_slice(&[t[(e, pb)].re, t[(e, pb)].im, t2[(e, pb)].re, t2[(e, pb)].im]);
                        at += 4;
                    }
                }
            }
        }
        r.work + self.far_work(lights, 0)
    }

    /// The complex multiply-adds of one node's elimination (the same at every node: the elimination's loops do not depend on the
    /// values).
    fn node_work(&self, layout: &Layout) -> f64 {
        let mut out = vec![0.0; self.table_width(layout.cluster, &layout.lights)];
        self.node(layout, &layout.focus, &mut out)
    }

    /// A table's grid filled (its coordinates' axes the first, the others at their focus values); refined from `from` where given
    /// (its axes nested in these). None beyond `most` new nodes.
    fn build(&self, layout: &Layout, grid: Vec<Axis>, from: Option<&Table>, most: usize) -> Option<Table> {
        let d = grid.len();
        let width = self.table_width(layout.cluster, &layout.lights);
        let f = |s: &[f64], out: &mut [f64]| {
            let mut x = layout.focus.clone();
            x[..d].copy_from_slice(s);
            self.node(layout, &x, out);
        };
        match from {
            None => Table::build(grid, width, most, &f).ok(),
            Some(t) => t.refine(grid, most, &f).ok(),
        }
    }

    /// The near tables on their first grids for the atoms' reach (module documentation, # Axes), each grid of at most `most` nodes:
    /// per cluster the running lights' and, where the trap is lit and a carrier stays lit without it, those lights less the trap's; a
    /// cluster whose grid of its heaviest coordinate alone exceeds the bound, or with no coordinate at the target, keeps the exact
    /// elimination.
    pub fn tabulate(&self, reach: &Reach, most: usize) -> Tables {
        let running = self.running();
        let mut untrapped = running.clone();
        untrapped.0[self.light.trap] = Lit::Off;
        let mut out = Tables { tables: vec![], work: 0.0 };
        for cluster in 0..self.clusters.len() {
            for lights in [&running, &untrapped] {
                if !self.carriers.iter().any(|c| lights.on(c.frequency)) || (lights == &untrapped && !running.on(self.light.trap)) {
                    continue;
                }
                if let Some(t) = self.first_table(cluster, lights, reach, most) {
                    out.work += Table::nodes(t.table.axes()) as f64 * self.node_work(&t.layout);
                    out.tables.push(t);
                }
            }
        }
        out
    }

    /// A cluster's first table for a light set (module documentation, # Axes).
    fn first_table(&self, cluster: usize, lights: &Lights, reach: &Reach, most: usize) -> Option<NearTable> {
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
        let width = self.table_width(cluster, lights);
        let exact = self.near_at(cluster, &Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, &focus_amplitudes, lights, 1).work * crate::decide::COMPLEX;
        let mut axes: Vec<usize> = vec![];
        for &c in &order {
            if weights[c] < crate::units::POPULATION_TARGET {
                break;
            }
            let d = axes.len() + 1;
            let first: Vec<Axis> = axes.iter().chain(std::iter::once(&c)).map(|&a| first_axis(all[a])).collect();
            if (4usize.pow(d as u32) * (1 + d) * width) as f64 >= exact || Table::nodes(&first) > most {
                break;
            }
            axes.push(c);
        }
        if axes.is_empty() {
            return None;
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

    /// The bound over a region of an intensity coordinate's distance from its focus value `at` (module documentation, # Axes): the
    /// field's magnitude within the region's radius times Σ_l max|∇a_l| ‖E_l‖ of its centre value (a unit's |f_u| with |κ_l|), at most
    /// every beam at its largest amplitude; on a point, the distance.
    fn intensity_moved(&self, c: Coordinate, polarizations: &[Vec<Polarization>], at: f64, region: &Region) -> f64 {
        let (centre, lipschitz, blind, norm) = match c {
            Coordinate::Chart { frequency, .. } => {
                let f = &self.light.frequencies[frequency];
                let mut e = [C64::ZERO; 3];
                let (mut lipschitz, mut blind) = (0.0, 0.0);
                for (&l, field) in f.beams.iter().zip(&f.fields) {
                    let span = &region.spans[l];
                    for q in 0..3 {
                        e[q] += span.centre * field[q];
                    }
                    let size = field.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                    lipschitz += span.lipschitz() * size;
                    blind += span.most * size;
                }
                (e.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(), lipschitz, blind, f.norm)
            }
            Coordinate::Unit { carrier, unit } => {
                let p = &polarizations[carrier][unit];
                let cc = &self.carriers[carrier];
                let mut f = C64::ZERO;
                let (mut lipschitz, mut blind) = (0.0, 0.0);
                for (&l, &k) in p.beams.iter().zip(&p.kappa) {
                    let span = &region.spans[cc.beams[l]];
                    f += span.centre * k;
                    lipschitz += span.lipschitz() * k.abs();
                    blind += span.most * k.abs();
                }
                (f.abs(), lipschitz, blind, p.norm)
            }
        };
        let (lo, hi) = if region.rho.is_finite() { ((centre - region.rho * lipschitz).max(0.0), (centre + region.rho * lipschitz).min(blind)) } else { (0.0, blind) };
        (hi * hi / norm - at).abs().max((at - lo * lo / norm).abs())
    }

    /// The near light at a point on a cluster under a light set, to `order`: from its table where one serves them (to order 0 or 1),
    /// else the exact elimination (eliminate.rs `Model::near_at`).
    pub(crate) fn near(&self, cluster: usize, at: &Point, amplitudes: &[Amplitude], lights: &Lights, order: usize, tables: Option<&Tables>) -> NearAt {
        match tables.and_then(|t| t.of(cluster, lights)) {
            Some(table) if order <= 1 => self.near_tabulated(table, at, amplitudes, order),
            _ => self.near_at(cluster, at, amplitudes, lights, order),
        }
    }

    /// The near light at a point from a table, to order 0 or 1 (module documentation, # At a point).
    fn near_tabulated(&self, table: &NearTable, at: &Point, amplitudes: &[Amplitude], order: usize) -> NearAt {
        let d = table.axes();
        let (x, gradients, mut work) = self.coordinates_at(&table.layout.coordinates[..d], &table.layout.polarizations, amplitudes);
        let w = table.table.width();
        let mut value = vec![0.0; w];
        let mut slope = vec![0.0; if order >= 1 { d * w } else { 0 }];
        table.table.eval(&x, &mut value, if order >= 1 { Some(&mut slope) } else { None });
        // The interpolation (4^D nodes, the value and, to first order, D slopes of w reals each, then the chain rule), in complex
        // multiply-adds.
        let real = if order >= 1 { (4usize.pow(d as u32) * (1 + d) + 3 * d) * w } else { 4usize.pow(d as u32) * w };
        work += real as f64 / crate::decide::COMPLEX;
        let responses = self.table_responses(table, &value, &slope, &gradients, order);
        let (energies, far) = self.far_energies(amplitudes, &table.layout.lights, order);
        self.doppler(at, responses, energies, work + far)
    }

    /// The responses (jets to `order`) from a table's reals and their slopes along the axes.
    fn table_responses(&self, table: &NearTable, value: &[f64], slope: &[f64], gradients: &[[f64; 3]], order: usize) -> Responses {
        let (d, w, g) = (table.axes(), table.table.width(), self.clusters[table.layout.cluster].ground);
        let np = super::jet::parts(order);
        let (mut first, mut second) = (vec![], vec![]);
        let mut place = 0;
        for cc in &self.carriers {
            if !table.layout.lights.on(cc.frequency) {
                first.push(vec![]);
                second.push(vec![]);
                continue;
            }
            let (mut f1, mut f2) = (vec![], vec![]);
            for _ in &cc.beams {
                let mut t: Vec<Mat> = (0..np).map(|_| Mat::zeros(EXCITED, g)).collect();
                let mut t2: Vec<Mat> = (0..np).map(|_| Mat::zeros(EXCITED, g)).collect();
                for pb in 0..g {
                    for e in 0..EXCITED {
                        t[0][(e, pb)] = C64::new(value[place], value[place + 1]);
                        t2[0][(e, pb)] = C64::new(value[place + 2], value[place + 3]);
                        if order >= 1 {
                            for i in 0..3 {
                                let along = |k: usize| (0..d).map(|f| slope[f * w + place + k] * gradients[f][i]).sum::<f64>();
                                t[1 + i][(e, pb)] = C64::new(along(0), along(1));
                                t2[1 + i][(e, pb)] = C64::new(along(2), along(3));
                            }
                        }
                        place += 4;
                    }
                }
                f1.push(t);
                f2.push(t2);
            }
            first.push(f1);
            second.push(f2);
        }
        Responses { first, second, work: 0.0 }
    }

    /// The tables' error reads at samples (module documentation; SPEC §2.3, §5.4): per table, per axis the read of its
    /// interpolation error (crate::table `Table::axis_error`) and the whole error (the exact elimination against the table's), each
    /// as the relative error of the sample's state's rates (`TableRead`).
    pub fn table_reads(&self, tables: &Tables, samples: &[Sample]) -> Vec<TableRead> {
        let zero = UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
        tables
            .tables
            .iter()
            .enumerate()
            .map(|(index, table)| {
                let (d, w) = (table.axes(), table.table.width());
                let mut axes = vec![vec![zero; samples.len()]; d];
                let mut whole = vec![zero; samples.len()];
                for (k, sample) in samples.iter().enumerate() {
                    if sample.cluster != table.layout.cluster {
                        continue;
                    }
                    let amplitudes = self.light.amplitudes(sample.at.r);
                    let (x, gradients, _) = self.coordinates_at(&table.layout.coordinates[..d], &table.layout.polarizations, &amplitudes);
                    let (mut value, mut slope) = (vec![0.0; w], vec![0.0; d * w]);
                    table.table.eval(&x, &mut value, Some(&mut slope));
                    let tabulated = self.table_responses(table, &value, &slope, &gradients, 1);
                    for (f, read) in axes.iter_mut().enumerate() {
                        let (mut dv, mut ds) = (vec![0.0; w], vec![0.0; d * w]);
                        table.table.axis_error(f, &x, &mut dv, &mut ds);
                        read[k] = self.rate_error(sample, &table.layout.lights, &tabulated, &self.table_responses(table, &dv, &ds, &gradients, 1), &amplitudes);
                    }
                    let h = self.far_hamiltonian(&amplitudes, &table.layout.lights, 1);
                    let exact = self.eliminate(table.layout.cluster, &h, &amplitudes, &table.layout.lights, 1, &self.coherent_dressing());
                    let error = Responses { first: difference(&exact.first, &tabulated.first), second: difference(&exact.second, &tabulated.second), work: 0.0 };
                    whole[k] = self.rate_error(sample, &table.layout.lights, &tabulated, &error, &amplitudes);
                }
                TableRead { table: index, axes, whole }
            })
            .collect()
    }

    /// The relative error of a sample's rates from an error of the responses: per lit carrier its field's response on the sample's
    /// ground amplitudes, R ψ_g = Σ_l a_l T_l ψ_g, against the error's; δ the largest over the carriers, δ_∇ the same of the
    /// gradients (the force's); on the populations δ P_g, on T max(δ, δ_∇) P_g (`TableRead`).
    fn rate_error(&self, sample: &Sample, lights: &Lights, responses: &Responses, error: &Responses, amplitudes: &[Amplitude]) -> UnitError {
        let g = self.clusters[sample.cluster].ground;
        let ground = &sample.psi[..g];
        let (mut delta, mut slope) = (0.0_f64, 0.0_f64);
        for (ci, cc) in self.carriers.iter().enumerate() {
            if !lights.on(cc.frequency) || responses.first[ci].is_empty() {
                continue;
            }
            for part in 0..responses.first[ci][0].len() {
                let rates = |r: &Responses| -> Vec<C64> {
                    let mut m = Mat::zeros(EXCITED, g);
                    for (l, &b) in cc.beams.iter().enumerate() {
                        m.add_scaled(amplitudes[b].value, &r.first[ci][l][part]);
                    }
                    m.apply(ground)
                };
                let (r, e) = (rates(responses), rates(error));
                let size = crate::linalg::norm_sqr(&r).sqrt();
                if size > 0.0 {
                    let ratio = crate::linalg::norm_sqr(&e).sqrt() / size;
                    if part == 0 {
                        delta = delta.max(ratio);
                    } else {
                        slope = slope.max(ratio);
                    }
                }
            }
        }
        let total: f64 = sample.psi.iter().map(|z| z.norm_sqr()).sum();
        let weight = ground.iter().map(|z| z.norm_sqr()).sum::<f64>() / total;
        UnitError { parameter: delta, population: delta * weight, temperature: delta.max(slope) * weight }
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

/// a − b, per carrier, beam and part.
fn difference(a: &[Vec<Vec<Mat>>], b: &[Vec<Vec<Mat>>]) -> Vec<Vec<Vec<Mat>>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            x.iter()
                .zip(y)
                .map(|(p, q)| {
                    p.iter()
                        .zip(q)
                        .map(|(m, n)| {
                            let mut out = m.clone();
                            out.add_scaled(-C64::ONE, n);
                            out
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}
