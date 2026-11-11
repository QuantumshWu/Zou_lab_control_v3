//! The factorized plan reaches Pulse directly, with bounded native expansion.
use std::path::PathBuf;
use std::sync::Arc;
use zlc_core::scan::{AxisSpec, Plan, PlanSpec};
use zlc_pulse::board::Board;
use zlc_pulse::run::{ScanRows, load_sequence, read_document};
use zlc_pulse::scan_plan::{ScanPlanRequest, effective_inner, prepare_application, table_for_plan};
use zlc_pulse::wire::StreamerParams;

fn axis(port: &str, values: Vec<f64>, unit: &str, parent: Option<&str>, host: bool) -> AxisSpec {
    AxisSpec {port:port.into(),identity:port.into(),label:port.into(),unit:unit.into(),values,
        coordinate_of:parent.map(str::to_owned),host_advanced:host}
}
fn plan(axes: Vec<AxisSpec>, repeats: usize) -> Arc<Plan> {
    Arc::new(Plan::new(PlanSpec {axes,scan_repeats:repeats,run_repeats:1}).unwrap())
}
fn pulse() -> zlc_pulse::model::Sequence {
    read_document(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/pulses/imaging_template.json")).unwrap().1
}

#[test]
fn a_native_segment_preserves_inner_order_host_linked_slots_and_quantized_units() {
    let sequence=pulse();
    let parent="api:duration:short";
    let plan=plan(vec![axis(parent,vec![100.0,200.0],"us",None,true),
        axis("pulse:param:duration:long_before",vec![0.2,0.3],"ms",Some(parent),false),
        axis("pulse:param:duration:long_after",vec![400.003,500.007],"us",None,false)],1);
    let request=ScanPlanRequest {plan:plan.clone(),outer_index:1};
    let board=Board::virtual_board(None).unwrap();
    let table=table_for_plan(&sequence,&request,&board.description().geometry).unwrap().unwrap();
    assert_eq!(table.fields,["duration:long_after","duration:long_before"]);
    for (actual, expected) in table.rows.iter().flatten().zip([0.000400003,0.0003,0.000500007,0.0003]) {
        assert!((actual-expected).abs()<1e-15);
    }
    let receipt=load_sequence(&board,&sequence,&[("duration:short".into(),(200.0,"us".into()))],Some(&table)).unwrap();
    let played=receipt.get("scan").unwrap();
    let played=ScanRows {fields:played.get("fields").unwrap().as_array().unwrap().iter().map(|v|v.as_str().unwrap().into()).collect(),
        rows:played.get("rows").unwrap().as_array().unwrap().iter().map(|r|r.as_array().unwrap().iter().map(|v|v.as_f64().unwrap()).collect()).collect()};
    let effective=effective_inner(&sequence,&plan,Some(&played)).unwrap().unwrap();
    assert!((effective[0][0]-400.0).abs()<1e-9 && (effective[1][0]-500.0).abs()<1e-9);
    assert_eq!(board.applied().unwrap().unwrap().rows.len(),2);
    assert!(board.last_fire().is_none(),"the native LOAD must not fire");
    board.close().unwrap();
}

#[test]
fn expansion_is_refused_by_slots_count_cursor_and_memory_before_building_rows() {
    let sequence=pulse();
    let geometry=StreamerParams::default();
    let axes=||vec![axis("pulse:param:duration:long_before",vec![1.0;1000],"us",None,false),
        axis("pulse:param:duration:long_after",vec![2.0;1000],"us",None,false)];
    let request=ScanPlanRequest {plan:plan(axes(),1),outer_index:0};
    assert!(table_for_plan(&sequence,&request,&geometry).unwrap_err().message.contains("materialization"));
    let mut narrow=geometry.clone(); narrow.num_slots=1;
    assert!(table_for_plan(&sequence,&request,&narrow).unwrap_err().message.contains("at most 1 slots"));
    let request=ScanPlanRequest {plan:plan(vec![axis("pulse:param:duration:long_before",vec![1.0;100_000],"us",None,false),
        axis("pulse:param:duration:long_after",vec![2.0;100_000],"us",None,false)],1),outer_index:0};
    assert!(table_for_plan(&sequence,&request,&geometry).unwrap_err().message.contains("SCAN_COUNT"));
    let request=ScanPlanRequest {plan:plan(vec![axis("pulse:param:duration:long_before",vec![1.0,2.0],"us",None,false)],1usize<<32),outer_index:0};
    assert!(table_for_plan(&sequence,&request,&geometry).unwrap_err().message.contains("CURSOR"));
}

#[test]
fn an_api_only_plan_has_no_table_and_bad_bindings_are_refused() {
    let sequence=pulse(); let geometry=StreamerParams::default();
    let api=plan(vec![axis("api:duration:short",vec![1.0,2.0],"us",None,true)],2);
    assert!(table_for_plan(&sequence,&ScanPlanRequest {plan:api.clone(),outer_index:1},&geometry).unwrap().is_none());
    assert!(effective_inner(&sequence,&api,None).unwrap().is_none());
    assert!(table_for_plan(&sequence,&ScanPlanRequest {plan:api,outer_index:2},&geometry).is_err());
    let plan=plan(vec![axis("pulse:param:absent",vec![1.0],"us",None,false)],1);
    assert!(table_for_plan(&sequence,&ScanPlanRequest {plan,outer_index:0},&geometry).unwrap_err().message.contains("not a field"));
}

#[test]
fn quantized_linked_coordinates_are_validated_before_any_load_or_fire() {
    let sequence=pulse();
    let parent="pulse:param:duration:long_before";
    let request=ScanPlanRequest {plan:plan(vec![axis(parent,vec![200.001,200.007],"us",None,false),
        axis("pulse:param:duration:long_after",vec![400.0,500.0],"us",Some(parent),false)],1),outer_index:0};
    let error=prepare_application(&sequence,&request,&[],&StreamerParams::default()).err().unwrap();
    assert!(error.message.contains("alternative coordinates must be unique"),"{error}");
}
