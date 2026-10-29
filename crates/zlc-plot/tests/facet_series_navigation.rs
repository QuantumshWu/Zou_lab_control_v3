//! Cell-kind/group edits must preserve the open facet, independently of the
//! changed cell coordinate roles. The fixture models a 5 x 400 x 35 bool scan
//! with one pair facet, 20 frequencies, 20 dipoles and 35 reduced sites.
use std::sync::Arc;
use serde_json::json;
use zlc_core::{DType, OwnedBuffer};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::schema::{Axis, CodeBase, CodeMapping, Coordinates, DatasetSchema, Domain, ValueSchema, ValidityContract};
use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
use zlc_core::store::{DatasetStore, Placement, Snapshot};
use zlc_plot::configuration::{PlotKind, PlotSchema, Value, ValueMap};
use zlc_plot::configuration::resolve::{resolve_configuration, ConfigurationUpdates, Defaults, ResolveRequest};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::projection::{SchemaCache, Source};
use zlc_plot::projection::output::Payload;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{DataPreparation, EngineFitLayer, PlotInput, PlotSession, PointerAction, PointerEvent, SessionServices};

const PAIR: &str = "scan.pair";
const RF: &str = "scan.rf";
const DIPOLE: &str = "scan.dipole";

fn services() -> SessionServices {
    SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() }
}
fn axis(name: &str, count: usize, role: &str) -> Arc<Axis> {
    let unit=match name { RF=>Some("Hz".into()), DIPOLE=>Some("V".into()), _=>None };
    Axis::new(name.into(), name.into(), role.into(), Coordinates::Index { origin: 0, len: count }, unit, None, None, None).unwrap()
}
fn schema(pair_count: usize, repeats: usize) -> Arc<DatasetSchema> {
    scan_schema(pair_count,repeats,20)
}
fn scan_schema(pair_count: usize, repeats: usize, rf_count: usize) -> Arc<DatasetSchema> {
    let repeat = Domain::dense(vec![repeats], vec![axis("scan.repeat", repeats, "repeat")]).unwrap();
    let mapping = |len, inner, outer| CodeMapping::new(CodeBase::Range { start: 0, step: 1, len }, inner, outer).unwrap();
    let point = Domain::new(vec![pair_count * rf_count * 20], vec![axis(PAIR, pair_count, "scan-point"), axis(RF, rf_count, "scan-point"), axis(DIPOLE, 20, "scan-point")],
        vec![mapping(pair_count,rf_count*20,1), mapping(rf_count,20,pair_count), mapping(20,1,pair_count*rf_count)]).unwrap();
    let cell = Domain::dense(vec![35], vec![axis("site",35,"site")]).unwrap();
    DatasetSchema::new(repeat, point, cell, ValueSchema::new(DType::Bool,None,None,ValidityContract::Value).unwrap()).unwrap()
}
fn target(pair: &str, rf: &str, dipole: &str) -> ValueMap {
    Value::from_json(&json!({"semantic":{"kind":"facet_grid","cell_kind":"image",
        format!("fate:point:{pair}"):"facet",format!("fate:point:{rf}"):"x",format!("fate:point:{dipole}"):"y"},
        "facet_focus":0,"size":"2x2"})).as_map().unwrap().clone()
}
fn mounted(snapshot: Snapshot) -> PlotInput {
    let runtime=Runtime::new();
    let generation=runtime.begin("fixture",vec![Declaration { signal:"scan".into(),schema:snapshot.schema().clone(),storage:Storage::Snapshot }],json!({})).unwrap();
    runtime.commit(&generation,&Bundle { inputs:vec![Input::Dataset { signal:"scan".into(),ordinal:None,snapshot }],metadata:json!({}),parents:vec![] }).unwrap();
    let read=runtime.read_coherent_views(&[(generation,"scan".into(),None)],&mut Default::default()).pop().unwrap().unwrap();
    PlotInput::Dataset(Arc::new(Source::from_read(&read).unwrap()))
}
fn synthetic_input() -> PlotInput {
    let schema=schema(1,5);
    let values=(0..5).flat_map(|r| (0..400).flat_map(move |point| (0..35).map(move |site| r*35+site < (point%20+1)*8))).collect();
    let block=DataBlock::new(schema.clone(),OwnedBuffer::from_vec(values).into(),Validity::Uniform(true),None).unwrap();
    let snapshot=DatasetStore::new(schema).commit(vec![Placement::new([0,0],[5,400],block).unwrap()]).unwrap();
    mounted(snapshot)
}
fn open(input: &PlotInput, target: &ValueMap) -> PlotSession {
    let services=services();
    let candidate=resolve_configuration(ResolveRequest { schema:input.schema(),kind:Some(PlotKind::FacetGrid),target:Some(target),
        defaults:Some(services.defaults.clone()),fit:Some(services.fit.clone()),..Default::default() }).unwrap();
    let prepared=DataPreparation::new(candidate.clone(),None).prepare(input,None,&mut SchemaCache::default(),&services.settings(),&services.fit.engine).unwrap();
    let mut plot=PlotSession::from_prepared(prepared,candidate,1.0,"scan",services).unwrap();
    plot.finish_presentation(true,None);
    plot
}
fn edit(plot: &mut PlotSession, input: &PlotInput, name: &str, value: &str) {
    let candidate=plot.resolve_configuration(None,None,None,Some(ValueMap::from_pairs([(name,Value::text(value))])),None,true,None,None,None,None).unwrap();
    assert_eq!(candidate.facet_focus,Some(0),"{name} must retain the open pair cell");
    let prepared=plot.prepare_data(input,None,Some(candidate.clone())).unwrap();
    plot.present_prepared(prepared,Some(candidate),false).unwrap();
    plot.finish_presentation(true,None);
}
fn curve_group_edit_and_interact(input: PlotInput, target: ValueMap, dipole: &str, followup: Option<PlotInput>) {
    let mut plot=open(&input,&target);
    assert_eq!(plot.facet_focus(),Some(0));
    edit(&mut plot,&input,"cell_kind","curve");
    edit(&mut plot,&input,&format!("fate:point:{dipole}"),"group");
    assert_eq!(plot.facet_focus(),Some(0));
    let (x,y)=match &**plot.projection().payload() {
        Payload::Facet(facet) => {
            assert_eq!(facet.cells.len(),1);
            let Payload::Curve(curve)=&facet.cells[0].payload else {panic!("curve cell")};
            assert_eq!(curve.series.len(),20);
            (curve.series[10].x.display.to_f64()[5],curve.series[10].y.display.to_f64()[5])
        }, _=>panic!("facet grid"),
    };
    let axes=plot.raster_axes_snapshot().unwrap().into_iter().find(|axes|axes.role=="facet_cell"&&axes.cell_index==Some(0)).unwrap();
    let (nx,ny)=axes.display_to_normalized(x,y);
    let mut front=plot.rgba().unwrap();
    for (action,button,time) in [(PointerAction::Move,None,1.0),(PointerAction::Press,Some(1),2.0),(PointerAction::Release,Some(1),3.0)] {
        plot.raster_pointer_event(&PointerEvent {action,button,x:nx,y:ny,double:false,step:0.0,key:None,axes:Some(axes.clone()),time:Some(time)}).unwrap();
        if action==PointerAction::Press {continue;}
        let readout=plot.renderer().series_presentation().series_readout.expect("the grouped curve must hover and lock after editing");
        assert_eq!(readout.target.facet_index,Some(0));
        assert!(readout.target.key.iter().any(|part| &*part[1]==dipole));
        assert_eq!(&*readout.mode,if action==PointerAction::Move {"hover"} else {"locked"});
        let changed=plot.rgba().unwrap();
        assert!(front.pixels.as_slice()!=changed.pixels.as_slice(),"{action:?} must visibly highlight/label the series");
        front=changed;
    }
    assert!(plot.display_state().interaction.series_lock.is_some());
    let following=followup.as_ref().unwrap_or(&input);
    let candidate=plot.resolve_configuration(following.schema(),None,None,None,None,true,None,None,None,None).unwrap();
    let prepared=plot.prepare_data(following,None,Some(candidate.clone())).unwrap();
    plot.present_prepared(prepared,Some(candidate),true).unwrap();
    plot.finish_presentation(true,None);
    assert_eq!(plot.facet_focus(),Some(0),"a later frame keeps the open cell");
    assert!(plot.display_state().interaction.series_lock.is_some(),"a later frame keeps its locked series");
}

#[test]
fn editing_cell_kind_and_group_keeps_the_focused_scan_interactive() {
    curve_group_edit_and_interact(synthetic_input(),target(PAIR,RF,DIPOLE),DIPOLE,Some(synthetic_input()));
}

#[test]
fn facet_focus_tracks_facet_layout_and_explicit_overview_not_cell_axes() {
    let original=PlotSchema::new(schema(3,5));
    let base=resolve_configuration(ResolveRequest {schema:Some(original.clone()),kind:Some(PlotKind::FacetGrid),target:Some(&target(PAIR,RF,DIPOLE)),
        fit:Some(services().fit),configuration_updates:Some(ConfigurationUpdates {facet_focus:Some(Some(2)),..Default::default()}),..Default::default()}).unwrap();
    let change=|| ValueMap::from_pairs([("cell_kind",Value::text("curve"))]);
    let next=resolve_configuration(ResolveRequest {schema:Some(original.clone()),base:Some(&base),semantic_updates:Some(change()),..Default::default()}).unwrap();
    assert_eq!((next.facet_focus,next.focused_facet_index),(Some(2),Some(2)));
    let overview=resolve_configuration(ResolveRequest {schema:Some(original.clone()),base:Some(&base),semantic_updates:Some(change()),
        configuration_updates:Some(ConfigurationUpdates {facet_focus:Some(None),..Default::default()}),..Default::default()}).unwrap();
    assert_eq!(overview.facet_focus,None);
    let same_overview=resolve_configuration(ResolveRequest {schema:Some(original.clone()),base:Some(&overview),semantic_updates:Some(ValueMap::from_pairs([("cell_kind",Value::text("image"))])),..Default::default()}).unwrap();
    assert_eq!(same_overview.facet_focus,None);
    let changed_source=resolve_configuration(ResolveRequest {schema:Some(original.clone()),base:Some(&base),different_source:true,semantic_updates:Some(change()),..Default::default()}).unwrap();
    assert_eq!(changed_source.facet_focus,None);
    let changed_facet=resolve_configuration(ResolveRequest {schema:Some(original),base:Some(&base),semantic_updates:Some(ValueMap::from_pairs([
        ("fate:point:scan.pair",Value::text("reduce")),("fate:point:scan.dipole",Value::text("facet")),("cell_kind",Value::text("curve"))])),..Default::default()}).unwrap();
    assert_eq!(changed_facet.facet_focus,None);
    let smaller=resolve_configuration(ResolveRequest {schema:Some(PlotSchema::new(schema(1,5))),base:Some(&base),semantic_updates:Some(change()),..Default::default()}).unwrap();
    assert_eq!(smaller.facet_focus,None,"a changed facet carrier must not retain index 2");
    let more_repeats=resolve_configuration(ResolveRequest {schema:Some(PlotSchema::new(schema(3,7))),base:Some(&base),semantic_updates:Some(change()),..Default::default()}).unwrap();
    assert_eq!(more_repeats.facet_focus,Some(2),"unrelated Repeat growth does not change the Point facet");
    let more_rf=resolve_configuration(ResolveRequest {schema:Some(PlotSchema::new(scan_schema(3,5,21))),base:Some(&base),semantic_updates:Some(change()),..Default::default()}).unwrap();
    assert_eq!(more_rf.facet_focus,Some(2),"growth of another axis in the same carrier keeps facet identity");

    for changes in [ValueMap::from_pairs([("fate:point:scan.dipole",Value::text("x")),("fate:point:scan.rf",Value::text("y"))]),
        ValueMap::from_pairs([("fate:repeat:scan.repeat",Value::list([Value::text("scope-value"),Value::Int(2)]))]),
        ValueMap::from_pairs([("fate:cell_data:site",Value::list([Value::text("scope-value"),Value::Int(3)]))])] {
        let kept=resolve_configuration(ResolveRequest {schema:base.schema.clone(),base:Some(&base),semantic_updates:Some(changes),..Default::default()}).unwrap();
        assert_eq!(kept.facet_focus,Some(2),"cell axis/scope edits keep the pair facet");
    }
    let unit=resolve_configuration(ResolveRequest {schema:base.schema.clone(),base:Some(&base),parameter_updates:Some(ValueMap::from_pairs([("x_display_unit",Value::text("kHz"))])),..Default::default()}).unwrap();
    assert_eq!(unit.facet_focus,Some(2));
    let group=resolve_configuration(ResolveRequest {schema:next.schema.clone(),base:Some(&next),semantic_updates:Some(ValueMap::from_pairs([("fate:point:scan.dipole",Value::text("group"))])),..Default::default()}).unwrap();
    assert_eq!(group.facet_focus,Some(2));
    let reduction=resolve_configuration(ResolveRequest {schema:group.schema.clone(),base:Some(&group),semantic_updates:Some(ValueMap::from_pairs([("reduction",Value::text("sum"))])),
        parameter_updates:Some(ValueMap::from_pairs([("uncertainty",Value::Bool(false))])),..Default::default()}).unwrap();
    assert_eq!(reduction.facet_focus,Some(2));
    let histogram=resolve_configuration(ResolveRequest {schema:base.schema.clone(),base:Some(&base),semantic_updates:Some(ValueMap::from_pairs([("cell_kind",Value::text("histogram"))])),..Default::default()}).unwrap();
    let scoped_rf=resolve_configuration(ResolveRequest {schema:histogram.schema.clone(),base:Some(&histogram),semantic_updates:Some(ValueMap::from_pairs([
        ("fate:point:scan.rf",Value::list([Value::text("scope-value"),Value::Int(2)]))])),..Default::default()}).unwrap();
    assert_eq!(scoped_rf.facet_focus,Some(2),"Cartesian RF scope leaves all pair identities present");

    use zlc_plot::configuration::resolve::InteractionUpdates;
    let captured=resolve_configuration(ResolveRequest {schema:base.schema.clone(),base:Some(&base),presented:Some(&base),semantic_updates:Some(change()),
        interaction_updates:Some(InteractionUpdates {recipe:ValueMap::from_pairs([("facet_focus",Value::None)]),focused_facet_index:Some(Some(1)),..Default::default()}),..Default::default()}).unwrap();
    assert_eq!((captured.facet_focus,captured.focused_facet_index),(None,Some(1)),"captured facet navigation survives an unrelated cell-kind edit");
}

#[test]
fn facet_focus_clears_when_scope_or_coordinate_order_changes_cell_identity() {
    use zlc_core::schema::{Coordinate, DomainRole};
    let original=schema(3,5);
    // Pair and RF are correlated in this carrier. Scoping RF removes pair
    // cells, unlike the Cartesian RF scope checked above.
    let mapping=|| CodeMapping::new(CodeBase::Range {start:0,step:1,len:3},20,1).unwrap();
    let point=Domain::new(vec![60],vec![axis(PAIR,3,"scan-point"),axis(RF,3,"scan-point"),axis(DIPOLE,20,"scan-point")],
        vec![mapping(),mapping(),CodeMapping::new(CodeBase::Range {start:0,step:1,len:20},1,3).unwrap()]).unwrap();
    let correlated=PlotSchema::new(DatasetSchema::new(original.domain(DomainRole::Repeat).clone(),point,original.domain(DomainRole::Cell).clone(),original.value().clone()).unwrap());
    let base=resolve_configuration(ResolveRequest {schema:Some(correlated.clone()),kind:Some(PlotKind::FacetGrid),target:Some(&target(PAIR,RF,DIPOLE)),
        fit:Some(services().fit),..Default::default()}).unwrap();
    let histogram=resolve_configuration(ResolveRequest {schema:Some(correlated.clone()),base:Some(&base),semantic_updates:Some(ValueMap::from_pairs([("cell_kind",Value::text("histogram"))])),..Default::default()}).unwrap();
    let scoped=resolve_configuration(ResolveRequest {schema:Some(correlated),base:Some(&histogram),semantic_updates:Some(ValueMap::from_pairs([
        ("fate:point:scan.rf",Value::list([Value::text("scope-value"),Value::Int(2)]))])),..Default::default()}).unwrap();
    assert_eq!(scoped.facet_focus,None,"scope changed which pair cell index zero names");

    let point=original.domain(DomainRole::Point);
    let mut axes=point.axes().to_vec();
    axes[0]=Axis::new(PAIR.into(),PAIR.into(),"scan-point".into(),Coordinates::Values(Arc::from(vec![Coordinate::Integer(2),Coordinate::Integer(1),Coordinate::Integer(0)])),None,None,None,None).unwrap();
    let reordered=Domain::new(point.physical_shape().to_vec(),axes,point.axes().iter().map(|axis|point.mapping(axis.id()).unwrap().clone()).collect()).unwrap();
    let reordered=PlotSchema::new(DatasetSchema::new(original.domain(DomainRole::Repeat).clone(),reordered,original.domain(DomainRole::Cell).clone(),original.value().clone()).unwrap());
    let base=resolve_configuration(ResolveRequest {schema:Some(PlotSchema::new(original)),kind:Some(PlotKind::FacetGrid),target:Some(&target(PAIR,RF,DIPOLE)),fit:Some(services().fit),..Default::default()}).unwrap();
    let changed=resolve_configuration(ResolveRequest {schema:Some(reordered),base:Some(&base),..Default::default()}).unwrap();
    assert_eq!(changed.facet_focus,None,"the same ordinal must not silently select a different coordinate");
}

#[test]
#[ignore="requires ZLC_SCAN_HOVER_FIXTURE pointing to a local saved scan; never commits experimental data"]
fn saved_scan_cell_curve_and_dipole_group_remain_interactive() {
    let path=std::env::var("ZLC_SCAN_HOVER_FIXTURE").expect("saved scan fixture");
    let services=services();
    let opened=zlc_plot::figure::read_document(&path,services.fit.clone(),services.defaults.clone(),None).unwrap();
    let item=opened.items.into_iter().next().unwrap();
    let target=Value::from_json(&item.target).as_map().unwrap().clone();
    curve_group_edit_and_interact(mounted(item.plane),target,"scan.dac:period8:da_dipole",None);
}
