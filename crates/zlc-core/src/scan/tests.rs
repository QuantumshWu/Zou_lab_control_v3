use std::sync::Arc;

use crate::DType;
use crate::schema::{Axis, CodeBase, CodeMapping, Coordinates, DatasetSchema, Domain, DomainRole, ValidityContract, ValueSchema};

use super::*;

fn axis(port: &str, values: &[f64], host: bool) -> AxisSpec {
    AxisSpec { port: port.into(), identity: port.into(), label: port.into(), unit: "Hz".into(), values: values.to_vec(), coordinate_of: None, host_advanced: host }
}

fn plan(axes: Vec<AxisSpec>) -> Plan {
    Plan::new(PlanSpec { axes, scan_repeats: 2, run_repeats: 3 }).unwrap()
}

fn index_axis(id: &str, len: usize) -> Arc<Axis> {
    Axis::new(id.into(), id.into(), "generic".into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap()
}

fn source(points: usize, repeats: usize) -> Arc<DatasetSchema> {
    let repeat = Domain::dense(vec![repeats], vec![index_axis("source.repeat", repeats)]).unwrap();
    let point = Domain::dense(vec![points], vec![index_axis("frame", points)]).unwrap();
    let cell = Domain::dense(vec![2, 3], vec![index_axis("y", 2), index_axis("x", 3)]).unwrap();
    DatasetSchema::new(repeat, point, cell, ValueSchema::new(DType::U16, Some("count".into()), Some("image".into()),
        ValidityContract::Value).unwrap()).unwrap()
}

#[test]
fn cartesian_rows_scopes_and_placements_have_the_existing_order() {
    let plan = plan(vec![axis("board", &[1.0, 2.0, 3.0], false), axis("device", &[10.0, 20.0], true)]);
    assert_eq!(plan.axes().iter().map(|axis| axis.port.as_str()).collect::<Vec<_>>(), ["device", "board"]);
    assert_eq!((plan.outer_axis_count(), plan.outer_count(), plan.inner_count(), plan.point_count(), plan.total_shots()), (1, 2, 3, 6, 36));
    assert_eq!(plan.rows(Scope::All, 0, 6).unwrap(), vec![vec![10.0, 1.0], vec![10.0, 2.0], vec![10.0, 3.0], vec![20.0, 1.0], vec![20.0, 2.0], vec![20.0, 3.0]]);
    assert_eq!(plan.row(Scope::Outer, 1).unwrap(), [20.0]);
    assert_eq!(plan.row(Scope::Inner, 2).unwrap(), [3.0]);
    assert_eq!(plan.placement(5, 1, 2, 4).unwrap(), [5, 20]);
    assert!(plan.row(Scope::All, 6).is_err());
    assert!(plan.placement(0, 2, 0, 4).is_err());
    assert!(plan.placement(0, 0, 3, 4).is_err());
    assert!(plan.placement(6, 0, 0, 4).is_err());
    assert!(plan.placement(0, 0, 0, 0).is_err());
    assert!(plan.placement(0, 0, 0, usize::MAX).is_err());
}

#[test]
fn linked_host_members_move_the_whole_group_without_reordering_columns() {
    let mut linked = axis("device", &[10.0, 20.0], true);
    linked.coordinate_of = Some("board".into());
    let plan = plan(vec![axis("inner", &[7.0, 8.0], false), linked, axis("board", &[1.0, 2.0], false)]);
    assert_eq!(plan.axes().iter().map(|axis| axis.port.as_str()).collect::<Vec<_>>(), ["device", "board", "inner"]);
    assert_eq!(plan.outer_axis_count(), 2);
    assert_eq!(plan.row(Scope::Outer, 1).unwrap(), [20.0, 2.0]);
    assert_eq!(plan.row(Scope::All, 2).unwrap(), [20.0, 2.0, 7.0]);
    assert_eq!(plan.point_count(), 4);
    let schema = plan.dataset_schema(&source(2, 1)).unwrap();
    let point = schema.domain(DomainRole::Point);
    assert_eq!(point.axis("scan.device").unwrap().coordinate_of(), Some("scan.board"));
    assert!(Arc::ptr_eq(point.mapping("scan.device").unwrap(), point.mapping("scan.board").unwrap()));
}

#[test]
fn empty_scope_is_one_empty_row_not_an_empty_plan() {
    let host = plan(vec![axis("device", &[1.0, 2.0], true)]);
    assert_eq!(host.row(Scope::Inner, 0).unwrap(), Vec::<f64>::new());
    assert!(host.row(Scope::Inner, 1).is_err());
    assert_eq!(host.with_effective_inner(&[vec![]]).unwrap().spec(), host.spec());
    let board = plan(vec![axis("board", &[1.0, 2.0], false)]);
    assert_eq!(board.row(Scope::Outer, 0).unwrap(), Vec::<f64>::new());
    assert_eq!(board.rows(Scope::All, 2, 0).unwrap(), Vec::<Vec<f64>>::new());
}

#[test]
fn invalid_axes_links_and_arithmetic_are_refused() {
    let spec = |axes| PlanSpec { axes, scan_repeats: 1, run_repeats: 1 };
    assert!(Plan::new(spec(vec![])).is_err());
    assert!(Plan::new(spec(vec![axis("a", &[], false)])).is_err());
    assert!(Plan::new(spec(vec![axis("a", &[f64::NAN], false)])).is_err());
    assert!(Plan::new(spec(vec![axis("a", &[1.0], false), axis("a", &[2.0], true)])).is_err());
    let mut link = axis("b", &[10.0], false);
    link.coordinate_of = Some("a".into());
    assert!(Plan::new(spec(vec![link.clone()])).is_err());
    assert!(Plan::new(spec(vec![axis("a", &[1.0, 2.0], false), link.clone()])).is_err());
    let mut parent = axis("a", &[1.0], false);
    parent.coordinate_of = Some("b".into());
    assert!(Plan::new(spec(vec![parent, link])).is_err());
    let axes = (0..usize::BITS).map(|i| axis(&format!("a{i}"), &[0.0, 1.0], false)).collect();
    assert!(Plan::new(spec(axes)).unwrap_err().contains("overflows"));
    assert!(Plan::new(PlanSpec { axes: vec![axis("a", &[1.0, 2.0], false)], scan_repeats: usize::MAX, run_repeats: 1 }).is_err());
    assert!(Plan::new(PlanSpec { axes: vec![axis("a", &[1.0], false)], scan_repeats: 0, run_repeats: 1 }).is_err());
}

#[test]
fn board_effective_values_update_only_the_inner_vectors_and_must_factor() {
    let original = plan(vec![axis("host", &[5.0, 6.0], true), axis("a", &[1.0, 2.0, 1.0], false), axis("b", &[10.0, 20.0], false)]);
    let rows = vec![vec![1.1, 10.1], vec![1.1, 20.1], vec![2.1, 10.1], vec![2.1, 20.1], vec![1.1, 10.1], vec![1.1, 20.1]];
    let effective = original.with_effective_inner(&rows).unwrap();
    assert_eq!(effective.axes()[0], original.axes()[0]);
    assert_eq!(effective.axes()[1].values, [1.1, 2.1, 1.1]);
    assert_eq!(effective.row(Scope::All, 11).unwrap(), [6.0, 1.1, 20.1]);
    assert_eq!(original.axes()[1].values, [1.0, 2.0, 1.0]);
    let mut inconsistent = rows.clone();
    inconsistent[4][0] = 1.2;
    assert!(original.with_effective_inner(&inconsistent).unwrap_err().contains("inconsistent"));
    assert!(original.with_effective_inner(&rows[..5]).is_err());
    let mut nonfinite = rows.clone(); nonfinite[0][0] = f64::INFINITY;
    assert!(original.with_effective_inner(&nonfinite).is_err());
    let mut wrong_width = rows; wrong_width[0].pop();
    assert!(original.with_effective_inner(&wrong_width).is_err());
}

#[test]
fn schema_keeps_source_points_and_pixels_and_compacts_repeated_coordinates() {
    let plan = plan(vec![axis("a", &[2.0, 1.0, 2.0, -0.0, 0.0], true), axis("b", &[3.0, 4.0], false)]);
    let source = source(2, 1);
    let schema = plan.dataset_schema(&source).unwrap();
    assert!(Arc::ptr_eq(schema.domain(DomainRole::Cell), source.domain(DomainRole::Cell)));
    assert_eq!(schema.physical_shape(), [6, 20, 2, 3]);
    let repeat = schema.domain(DomainRole::Repeat);
    assert_eq!(repeat.axes().iter().map(|axis| axis.id()).collect::<Vec<_>>(), ["scan.repeat", "pulse.run"]);
    assert_eq!((0..6).map(|i| repeat.mapping("scan.repeat").unwrap().code(i).unwrap()).collect::<Vec<_>>(), [0, 0, 0, 1, 1, 1]);
    assert_eq!((0..6).map(|i| repeat.mapping("pulse.run").unwrap().code(i).unwrap()).collect::<Vec<_>>(), [0, 1, 2, 0, 1, 2]);
    let point = schema.domain(DomainRole::Point);
    assert_eq!(point.axes().iter().map(|axis| axis.id()).collect::<Vec<_>>(), ["frame", "scan.a", "scan.b"]);
    assert_eq!(point.axis("scan.a").unwrap().len(), 3);
    for row in 0..plan.point_count() {
        let expected = plan.row(Scope::All, row).unwrap();
        for frame in 0..2 {
            let flat = row * 2 + frame;
            assert_eq!(point.mapping("frame").unwrap().code(flat), Some(frame));
            for (id, value) in ["scan.a", "scan.b"].into_iter().zip(&expected) {
                let axis = point.axis(id).unwrap();
                assert_eq!(axis.coordinate(point.mapping(id).unwrap().code(flat).unwrap()), Some(crate::schema::Coordinate::Real(*value)));
            }
        }
    }
    assert!(matches!(point.mapping("scan.a").unwrap().base(), CodeBase::Explicit(values) if values.as_slice() == [0,1,0,2,2]));
    assert!(plan.dataset_schema(&self::source(2, 2)).is_err());
}

#[test]
fn linked_tuple_dedup_and_existing_coordinate_uniqueness_are_preserved() {
    let mut second = axis("b", &[4.0, 3.0, 4.0], false);
    second.coordinate_of = Some("a".into());
    let plan = plan(vec![axis("a", &[2.0, 1.0, 2.0], false), second.clone()]);
    let schema = plan.dataset_schema(&source(1, 1)).unwrap();
    let point = schema.domain(DomainRole::Point);
    assert_eq!(point.axis("scan.b").unwrap().len(), 2);
    assert_eq!(point.mapping("scan.b").unwrap().code(2), Some(0));
    // Two different linked tuples with the same primary coordinate still
    // violate Axis uniqueness, as in the existing Python data model.
    second.values = vec![4.0, 3.0, 5.0];
    let invalid = Plan::new(PlanSpec { axes: vec![axis("a", &[2.0, 1.0, 2.0], false), second], scan_repeats: 1, run_repeats: 1 });
    assert!(invalid.unwrap_err().contains("different linked tuples"));
    let mut second = axis("b", &[4.0, 4.0], false); second.coordinate_of = Some("a".into());
    assert!(Plan::new(PlanSpec { axes: vec![axis("a", &[1.0, 2.0], false), second], scan_repeats: 1, run_repeats: 1 }).is_err());
}

#[test]
fn stable_axis_identity_suffixes_and_collisions_are_checked() {
    let a = axis("a", &[1.0], false);
    let mut b = axis("b", &[2.0], false); b.identity = "a".into(); b.label = "renamed".into();
    let plan = plan(vec![a.clone(), b.clone()]);
    assert_eq!(plan.axis_ids(), ["scan.a", "scan.a.2"]);
    let mut collision = axis("c", &[3.0], false); collision.identity = "a.2".into();
    assert!(Plan::new(PlanSpec { axes: vec![a, b, collision], scan_repeats: 1, run_repeats: 1 }).is_err());
    let source = source(1, 1);
    let point = Domain::dense(vec![1], vec![index_axis("scan.a", 1)]).unwrap();
    let collision = DatasetSchema::new(source.domain(DomainRole::Repeat).clone(), point,
        source.domain(DomainRole::Cell).clone(), source.value().clone()).unwrap();
    assert!(plan.dataset_schema(&collision).is_err());
    assert!(Plan::new(PlanSpec { axes: vec![axis("repeat", &[1.0], false)], scan_repeats: 1, run_repeats: 1 })
        .unwrap_err().contains("reserved"));
}

#[test]
fn source_mapped_coordinates_are_tiled_without_expanding_their_base() {
    let source = source(1, 1);
    let frames = index_axis("frame", 2);
    let mapping = CodeMapping::new(CodeBase::Explicit(crate::OwnedBuffer::from_vec(vec![1, 0, 1])), 2, 3).unwrap();
    let point = Domain::new(vec![18], vec![frames], vec![mapping.clone()]).unwrap();
    let source = DatasetSchema::new(source.domain(DomainRole::Repeat).clone(), point,
        source.domain(DomainRole::Cell).clone(), source.value().clone()).unwrap();
    let plan = plan(vec![axis("a", &[1.0, 2.0], false)]);
    let schema = plan.dataset_schema(&source).unwrap();
    let extended = schema.domain(DomainRole::Point).mapping("frame").unwrap();
    assert_eq!((extended.base_len(), extended.inner(), extended.outer()), (3, 2, 6));
    for i in 0..36 { assert_eq!(extended.code(i), mapping.code(i % 18)); }
}

#[test]
fn billion_point_geometry_stays_factored_and_row_exports_are_bounded() {
    let values = (0..1000).map(|i| i as f64).collect::<Vec<_>>();
    let plan = plan(vec![axis("a", &values, true), axis("b", &values, false), axis("c", &values, false)]);
    assert_eq!(plan.point_count(), 1_000_000_000);
    assert_eq!(plan.axes().iter().map(|axis| axis.values.len()).sum::<usize>(), 3000);
    assert_eq!(plan.row(Scope::All, 999_999_999).unwrap(), [999.0, 999.0, 999.0]);
    assert!(plan.rows(Scope::All, 0, 1_000_000_000).is_err());
    let schema = plan.dataset_schema(&source(1, 1)).unwrap();
    for id in plan.axis_ids() {
        let mapping = schema.domain(DomainRole::Point).mapping(id).unwrap();
        assert_eq!(mapping.base_len(), 1000);
        assert!(matches!(mapping.base(), CodeBase::Range { .. }));
        assert_eq!(mapping.carrier_len(), 1_000_000_000);
    }
    let repeats = Plan::new(PlanSpec { axes: vec![axis("a", &[1.0], false)], scan_repeats: 100_000_000, run_repeats: 1 }).unwrap();
    let schema = repeats.dataset_schema(&source(1, 1)).unwrap();
    assert!(matches!(schema.domain(DomainRole::Repeat).axes()[0].coordinates(), Coordinates::Index { len: 100_000_000, .. }));
}
