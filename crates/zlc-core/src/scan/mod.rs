//! Immutable scan geometry, independent of devices, I/O and run scheduling.
//!
//! Independent groups form a Cartesian product; linked coordinates advance
//! together. Only each axis's authored vector is stored. Rows are addressed
//! by mixed-radix index, and dataset mappings retain the same factorization.

mod dataset;

use std::collections::{HashMap, HashSet};


pub type Result<T> = std::result::Result<T, String>;

/// Bound a materialized request, including a request for zero-column rows.
pub const MAX_ROW_VALUES: usize = 65_536;

#[derive(Clone, Debug, PartialEq)]
pub struct AxisSpec {
    pub port: String,
    /// Stable identity projected by the port owner, not the display label.
    pub identity: String,
    pub label: String,
    pub unit: String,
    pub values: Vec<f64>,
    pub coordinate_of: Option<String>,
    pub host_advanced: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanSpec {
    pub axes: Vec<AxisSpec>,
    pub scan_repeats: usize,
    pub run_repeats: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scope { Outer, Inner, All }

#[derive(Clone, Debug)]
struct Group {
    members: Vec<usize>,
    length: usize,
    /// Product of all following independent group lengths.
    stride: usize,
    host_advanced: bool,
}

#[derive(Clone, Debug)]
pub struct Plan {
    spec: PlanSpec,
    groups: Vec<Group>,
    axis_groups: Vec<usize>,
    axis_ids: Vec<String>,
    outer_axes: usize,
    outer_count: usize,
    inner_count: usize,
    point_count: usize,
    total_shots: usize,
}

fn multiply(left: usize, right: usize, what: &str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| format!("{what} overflows usize"))
}

fn text(value: &str, what: &str) -> Result<()> {
    if value.is_empty() || value.trim() != value { return Err(format!("{what} must be canonical nonempty text")); }
    Ok(())
}

fn group_key(axis: &AxisSpec) -> &str { axis.coordinate_of.as_deref().unwrap_or(&axis.port) }

/// Python floating-coordinate equality, including signed zero. NaN is refused
/// at admission, so the bit representation is otherwise a valid hash key.
fn number_key(value: f64) -> u64 { if value == 0.0 { 0 } else { value.to_bits() } }

impl Plan {
    pub fn new(mut spec: PlanSpec) -> Result<Self> {
        if spec.axes.is_empty() { return Err("a scan plan needs at least one axis".into()); }
        if spec.scan_repeats == 0 || spec.run_repeats == 0 { return Err("scan_repeats and run_repeats must be at least 1".into()); }
        let mut ports = HashMap::new();
        for (index, axis) in spec.axes.iter().enumerate() {
            text(&axis.port, "scan port")?;
            text(&axis.identity, "scan axis identity")?;
            text(&axis.label, "scan axis label")?;
            if axis.unit.trim() != axis.unit { return Err(format!("axis {:?} unit must be canonical text", axis.port)); }
            if ports.insert(axis.port.as_str(), index).is_some() { return Err(format!("port {:?} appears on more than one axis", axis.port)); }
            if axis.values.is_empty() { return Err(format!("axis {:?} has no values to play", axis.port)); }
            if axis.values.iter().any(|value| !value.is_finite()) { return Err(format!("axis {:?} contains a non-finite value", axis.port)); }
        }
        for axis in &spec.axes {
            if let Some(parent) = &axis.coordinate_of {
                let primary = ports.get(parent.as_str()).map(|index| &spec.axes[*index])
                    .ok_or_else(|| format!("linked axis {:?} names absent parent {parent:?}", axis.port))?;
                if parent == &axis.port || primary.coordinate_of.is_some() {
                    return Err(format!("linked axis {:?} must directly name an independent parent; cycles and linked parents are not allowed", axis.port));
                }
                if axis.values.len() != primary.values.len() { return Err(format!("linked axis {:?} and {parent:?} must have equal lengths", axis.port)); }
            }
        }
        let host_groups: HashSet<String> = spec.axes.iter().filter(|axis| axis.host_advanced).map(|axis| group_key(axis).to_owned()).collect();
        // Stable partition: a linked device/manual/API axis makes its whole
        // group host-advanced, preserving authored column order within each side.
        spec.axes.sort_by_key(|axis| !host_groups.contains(group_key(axis)));
        let outer_axes = spec.axes.iter().take_while(|axis| host_groups.contains(group_key(axis))).count();
        let mut group_indexes = HashMap::new();
        let mut groups: Vec<Group> = Vec::new();
        let mut axis_groups = Vec::with_capacity(spec.axes.len());
        for (index, axis) in spec.axes.iter().enumerate() {
            let key = group_key(axis);
            let group = match group_indexes.get(key) {
                Some(group) => *group,
                None => {
                    let group = groups.len();
                    group_indexes.insert(key.to_owned(), group);
                    groups.push(Group { members: Vec::new(), length: axis.values.len(), stride: 0, host_advanced: host_groups.contains(key) });
                    group
                }
            };
            groups[group].members.push(index);
            axis_groups.push(group);
        }
        let mut point_count = 1;
        for group in groups.iter_mut().rev() {
            group.stride = point_count;
            point_count = multiply(point_count, group.length, "scan point count")?;
        }
        let outer_count = groups.iter().filter(|group| group.host_advanced).try_fold(1, |count, group| multiply(count, group.length, "outer point count"))?;
        let inner_count = point_count / outer_count;
        let total_shots = multiply(multiply(point_count, spec.scan_repeats, "scan shot count")?, spec.run_repeats, "scan shot count")?;
        for group in groups.iter().filter(|group| group.members.len() > 1) {
            // An alternative coordinate must be one-to-one with its primary.
            // Replaying an identical tuple is valid; two different tuples
            // sharing any member's value cannot become a valid Dataset axis.
            // This is knowable before the first device move or pulse fire.
            let mut tuples = HashSet::new();
            let mut coordinates: Vec<HashSet<u64>> = group.members.iter().map(|_| HashSet::new()).collect();
            for position in 0..group.length {
                let tuple: Vec<u64> = group.members.iter().map(|member| number_key(spec.axes[*member].values[position])).collect();
                if !tuples.insert(tuple.clone()) { continue; }
                for ((seen, key), member) in coordinates.iter_mut().zip(tuple).zip(&group.members) {
                    if !seen.insert(key) {
                        return Err(format!("linked axis {:?} repeats a coordinate in different linked tuples; alternative coordinates must be unique", spec.axes[*member].port));
                    }
                }
            }
        }
        let mut ranks = HashMap::new();
        let mut taken = HashSet::new();
        let mut axis_ids = Vec::with_capacity(spec.axes.len());
        for axis in &spec.axes {
            let base = format!("scan.{}", axis.identity);
            let rank = ranks.entry(base.clone()).or_insert(0usize);
            *rank += 1;
            let id = if *rank == 1 { base } else { format!("{base}.{rank}") };
            if id == "scan.repeat" || id == "pulse.run" {
                return Err(format!("scan axis identity {id} is reserved for scan execution"));
            }
            if !taken.insert(id.clone()) { return Err(format!("scan axis identity collision: {id}")); }
            axis_ids.push(id);
        }
        Ok(Self { spec, groups, axis_groups, axis_ids, outer_axes, outer_count, inner_count, point_count, total_shots })
    }

    pub fn spec(&self) -> &PlanSpec { &self.spec }
    pub fn axes(&self) -> &[AxisSpec] { &self.spec.axes }
    pub fn axis_ids(&self) -> &[String] { &self.axis_ids }
    pub fn outer_axis_count(&self) -> usize { self.outer_axes }
    pub fn outer_count(&self) -> usize { self.outer_count }
    pub fn inner_count(&self) -> usize { self.inner_count }
    pub fn point_count(&self) -> usize { self.point_count }
    pub fn total_shots(&self) -> usize { self.total_shots }

    pub fn row_count(&self, scope: Scope) -> usize {
        match scope { Scope::Outer => self.outer_count, Scope::Inner => self.inner_count, Scope::All => self.point_count }
    }

    fn columns(&self, scope: Scope) -> std::ops::Range<usize> {
        match scope { Scope::Outer => 0..self.outer_axes, Scope::Inner => self.outer_axes..self.axes().len(), Scope::All => 0..self.axes().len() }
    }

    /// One row, with the last independent group advancing fastest.
    /// An empty Outer/Inner part has one row with zero columns.
    pub fn row(&self, scope: Scope, index: usize) -> Result<Vec<f64>> {
        if index >= self.row_count(scope) { return Err(format!("{scope:?} row {index} is outside the scan plan")); }
        let global = if scope == Scope::Outer { index * self.inner_count } else { index };
        Ok(self.columns(scope).map(|axis| {
            let group = &self.groups[self.axis_groups[axis]];
            self.spec.axes[axis].values[(global / group.stride) % group.length]
        }).collect())
    }

    /// A bounded page, never an implicit materialization of the whole product.
    pub fn rows(&self, scope: Scope, start: usize, count: usize) -> Result<Vec<Vec<f64>>> {
        let end = start.checked_add(count).ok_or("row range overflows usize")?;
        if end > self.row_count(scope) { return Err(format!("{scope:?} row range is outside the scan plan")); }
        let cells = multiply(count, self.columns(scope).len().max(1), "row page size")?;
        if cells > MAX_ROW_VALUES { return Err(format!("a row page may contain at most {MAX_ROW_VALUES} values")); }
        (start..end).map(|index| self.row(scope, index)).collect()
    }

    /// Record the board's quantized coordinates without expanding the outer
    /// product. The same requested value must always have the same effective
    /// value; a receipt depending on another axis cannot be represented by
    /// the declared independent plan and is refused explicitly.
    pub fn with_effective_inner(&self, rows: &[Vec<f64>]) -> Result<Self> {
        if rows.len() != self.inner_count { return Err(format!("effective inner receipt has {} rows, expected {}", rows.len(), self.inner_count)); }
        let width = self.axes().len() - self.outer_axes;
        let mut maps: Vec<HashMap<u64, f64>> = (0..width).map(|_| HashMap::new()).collect();
        for (row_index, row) in rows.iter().enumerate() {
            if row.len() != width { return Err(format!("effective inner row {row_index} has {} columns, expected {width}", row.len())); }
            for (column, effective) in row.iter().copied().enumerate() {
                if !effective.is_finite() { return Err(format!("effective inner row {row_index} contains a non-finite value")); }
                let axis = self.outer_axes + column;
                let group = &self.groups[self.axis_groups[axis]];
                let requested = self.spec.axes[axis].values[(row_index / group.stride) % group.length];
                if let Some(prior) = maps[column].insert(number_key(requested), effective) {
                    if prior != effective { return Err(format!("axis {:?}: requested {requested} has inconsistent effective values {prior} and {effective}", self.spec.axes[axis].port)); }
                }
            }
        }
        let mut spec = self.spec.clone();
        for (axis, effective) in spec.axes[self.outer_axes..].iter_mut().zip(maps) {
            for value in &mut axis.values { *value = effective[&number_key(*value)]; }
        }
        Self::new(spec)
    }

    /// Finite-store origin of one source event. Source Repeat is one shot;
    /// each scan row contributes every source Point cell.
    pub fn placement(&self, row: usize, scan_repeat: usize, run_repeat: usize, source_points: usize) -> Result<[usize; 2]> {
        if row >= self.point_count { return Err("plan row is outside the scan plan".into()); }
        if scan_repeat >= self.spec.scan_repeats { return Err("scan_repeat is outside the declared scan repeats".into()); }
        if run_repeat >= self.spec.run_repeats { return Err("run_repeat is outside the declared run repeats".into()); }
        if source_points == 0 { return Err("a source event needs at least one point".into()); }
        // Check the whole carrier, so even row zero cannot endorse an invalid layout.
        multiply(self.point_count, source_points, "scan Point carrier")?;
        Ok([scan_repeat * self.spec.run_repeats + run_repeat, row * source_points])
    }
}

#[cfg(test)]
mod tests;
