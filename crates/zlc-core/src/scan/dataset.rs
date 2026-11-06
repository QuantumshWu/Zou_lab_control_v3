use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::OwnedBuffer;
use crate::schema::{Axis, CodeBase, CodeMapping, Coordinates, DatasetSchema, Domain, DomainRole};

use super::{Plan, Result, multiply, number_key};

fn clone_base(base: &CodeBase) -> CodeBase {
    match base {
        CodeBase::Range { start, step, len } => CodeBase::Range { start: *start, step: *step, len: *len },
        CodeBase::Explicit(values) => CodeBase::Explicit(values.clone()),
    }
}

fn code_base(codes: Vec<u64>) -> CodeBase {
    if codes.iter().enumerate().all(|(index, value)| *value == index as u64) {
        CodeBase::Range { start: 0, step: 1, len: codes.len() }
    } else { CodeBase::Explicit(OwnedBuffer::from_vec(codes)) }
}

impl Plan {
    /// The plan's fixed schema over a one-shot source. Source Point and Cell
    /// axes are preserved; its singleton Repeat carrier is consumed. Code
    /// vectors scale with authored group lengths, never the Cartesian product.
    pub fn dataset_schema(&self, source: &Arc<DatasetSchema>) -> Result<Arc<DatasetSchema>> {
        if source.domain(DomainRole::Repeat).size() != 1 {
            return Err("a scan point's source must publish one shot per event; its Repeat carrier must have size 1".into());
        }
        let source_point = source.domain(DomainRole::Point);
        let source_cell = source.domain(DomainRole::Cell);
        let occupied: HashSet<&str> = source_point.axes().iter().chain(source_cell.axes()).map(|axis| axis.id()).collect();
        if occupied.contains("scan.repeat") || occupied.contains("pulse.run") {
            return Err("source Dataset axes collide with scan execution axes".into());
        }
        if let Some(id) = self.axis_ids.iter().find(|id| occupied.contains(id.as_str())) {
            return Err(format!("source Dataset axes collide with scan axes: {id}"));
        }
        if self.axis_ids.iter().any(|id| id == "scan.repeat" || id == "pulse.run") {
            return Err("scan axis ids collide with scan execution axes".into());
        }
        let point_count = multiply(self.point_count, source_point.size(), "scan Point carrier")?;
        let mut axes = source_point.axes().to_vec();
        let mut mappings = Vec::with_capacity(axes.len() + self.axes().len());
        for axis in &axes {
            let mapping = source_point.mapping(axis.id()).expect("source axis is mapped");
            mappings.push(CodeMapping::new(clone_base(mapping.base()), mapping.inner(),
                multiply(mapping.outer(), self.point_count, "source Point mapping")?)?);
        }

        let ports: HashMap<&str, usize> = self.axes().iter().enumerate().map(|(index, axis)| (axis.port.as_str(), index)).collect();
        let mut scan_axes: Vec<Option<Arc<Axis>>> = vec![None; self.axes().len()];
        let mut scan_mappings: Vec<Option<Arc<CodeMapping>>> = vec![None; self.axes().len()];
        for group in &self.groups {
            // Dedupe complete linked tuples, in first-seen order, just as the
            // flat Python schema did. Individual repeated linked coordinates
            // remain subject to the existing Axis uniqueness contract.
            let mut seen = HashMap::new();
            let mut domains = vec![Vec::new(); group.members.len()];
            let mut codes = Vec::with_capacity(group.length);
            for position in 0..group.length {
                let key: Vec<u64> = group.members.iter().map(|member| number_key(self.axes()[*member].values[position])).collect();
                let next = seen.len();
                let code = *seen.entry(key).or_insert_with(|| {
                    for (domain, member) in domains.iter_mut().zip(&group.members) { domain.push(self.axes()[*member].values[position]); }
                    next
                });
                codes.push(code as u64);
            }
            let inner = multiply(group.stride, source_point.size(), "scan coordinate mapping")?;
            let outer = self.point_count / (group.stride * group.length);
            let mapping = CodeMapping::new(code_base(codes), inner, outer)?;
            for (member, domain) in group.members.iter().zip(domains) {
                let axis = &self.axes()[*member];
                let coordinate_of = axis.coordinate_of.as_deref().map(|port| self.axis_ids[ports[port]].clone().into());
                scan_axes[*member] = Some(Axis::new(self.axis_ids[*member].clone().into(), axis.label.clone().into(), "scan-point".into(),
                    Coordinates::Numeric(OwnedBuffer::from_vec(domain).into()),
                    (!axis.unit.is_empty()).then(|| axis.unit.clone().into()), None, None, coordinate_of)?);
                scan_mappings[*member] = Some(mapping.clone());
            }
        }
        axes.extend(scan_axes.into_iter().map(|axis| axis.expect("every scan axis belongs to a group")));
        mappings.extend(scan_mappings.into_iter().map(|mapping| mapping.expect("every scan axis belongs to a group")));
        let point = Domain::new(vec![point_count], axes, mappings)?;

        // Integer coordinates are implicit: even a very large repeat count
        // needs only these two mappings, not coordinate vectors of that size.
        let scan = Axis::new("scan.repeat".into(), "repeat".into(), "repeat".into(),
            Coordinates::Index { origin: 0, len: self.spec.scan_repeats }, None, None, None, None)?;
        let shot = Axis::new("pulse.run".into(), "shots per point".into(), "repeat".into(),
            Coordinates::Index { origin: 0, len: self.spec.run_repeats }, None, None, None, None)?;
        let scan_mapping = CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: self.spec.scan_repeats }, self.spec.run_repeats, 1)?;
        let shot_mapping = CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: self.spec.run_repeats }, 1, self.spec.scan_repeats)?;
        let repeat = Domain::new(vec![multiply(self.spec.scan_repeats, self.spec.run_repeats, "scan Repeat carrier")?],
            vec![scan, shot], vec![scan_mapping, shot_mapping])?;
        DatasetSchema::new(repeat, point, source_cell.clone(), source.value().clone())
    }
}
