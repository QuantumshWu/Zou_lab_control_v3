use std::sync::Arc;
use zlc_core::{DType};
use zlc_core::schema::{Axis,Coordinates,DatasetSchema,Domain,ValueSchema,ValidityContract,CodeMapping,CodeBase};
use zlc_plot::configuration::{PlotSchema,PlotKind,Value,ValueMap,ResolveRequest,resolve_configuration,ResolvedConfiguration};
use zlc_plot::configuration::fit::EmptyFitLayer;
fn schema(axes:&[(&str,usize)]) -> Arc<PlotSchema> {
    let size:usize=axes.iter().map(|(_,n)|*n).product(); let shape=vec![size]; let codes=axes.iter().enumerate().map(|(i,(_,n))|CodeMapping::new(CodeBase::Range{start:0,step:1,len:*n},axes[i+1..].iter().map(|(_,n)|*n).product(),axes[..i].iter().map(|(_,n)|*n).product()).unwrap()).collect();
    let axes=axes.iter().map(|(id,n)| Axis::new((*id).into(),(*id).into(),"scan-point".into(),Coordinates::Index{origin:0,len:*n},None,None,None,None).unwrap()).collect();
    let empty=Domain::new(vec![1],vec![],vec![]).unwrap();
    let cell=Domain::dense(vec![1],vec![Axis::new("zlc_data.scalar".into(),"value".into(),"scalar".into(),Coordinates::Index{origin:0,len:1},None,None,None,None).unwrap()]).unwrap(); let value=ValueSchema::new(DType::F64,None,None,ValidityContract::Value).unwrap();
    PlotSchema::new(DatasetSchema::new(empty.clone(),Domain::new(shape,axes,codes).unwrap(),cell,value).unwrap())
}
fn fresh(s:Arc<PlotSchema>) -> Arc<ResolvedConfiguration> {
    let target=ValueMap::from_pairs([("semantic",Value::Map(ValueMap::from_pairs([("cell_kind",Value::text(""))])))]);
    resolve_configuration(ResolveRequest{schema:Some(s),kind:Some(PlotKind::FacetGrid),target:Some(&target),fit:Some(Arc::new(EmptyFitLayer)),..Default::default()}).unwrap()
}
fn show(c:&ResolvedConfiguration)->String { format!("{:?}",c.semantics.values()) }
#[test]
fn restart_rebases_automatic_axes_and_preserves_authored_choices(){
    let cases:Vec<(&str,Vec<(&str,usize)>,Vec<(&str,usize)>)>=vec![
        ("add_axis",vec![("A",3)],vec![("A",3),("B",4)]),
        ("remove_axis",vec![("A",3),("B",4)],vec![("A",3)]),
        ("replace_axis",vec![("A",3)],vec![("B",4)]),
        ("reorder_axes",vec![("A",3),("B",4)],vec![("B",4),("A",3)]),
        ("shrink_axis",vec![("A",3),("B",4)],vec![("A",3),("B",1)]), ("add_facet",vec![("A",3),("B",4)],vec![("A",3),("B",4),("C",5)]), ("remove_facet",vec![("A",3),("B",4),("C",5)],vec![("A",3),("B",4)])];
    for (name,before,after) in cases {
        let base=fresh(schema(&before)); let following=schema(&after);
        let expected=fresh(following.clone());
        let target=base.target().unwrap();
        let restored=resolve_configuration(ResolveRequest{schema:base.schema.clone(),target:Some(&target),kind:Some(PlotKind::FacetGrid),fit:Some(Arc::new(EmptyFitLayer)),..Default::default()}).unwrap();
        let restored_next=resolve_configuration(ResolveRequest{schema:Some(following.clone()),base:Some(&restored),..Default::default()}).unwrap();
        assert_eq!(show(&restored_next),show(&expected), "saved Auto {name}");
        let result=resolve_configuration(ResolveRequest{schema:Some(following),base:Some(&base),..Default::default()});
        assert_eq!(show(&result.unwrap()), show(&expected), "{name}");
    }
    let base=fresh(schema(&[("A",3),("B",4)]));
    let manual=resolve_configuration(ResolveRequest{schema:base.schema.clone(),base:Some(&base),semantic_updates:Some(ValueMap::from_pairs([("fate:point:A",Value::text("x")),("fate:point:B",Value::text("y"))])),..Default::default()}).unwrap();
    let after=resolve_configuration(ResolveRequest{schema:Some(schema(&[("B",4),("A",3)])),base:Some(&manual),..Default::default()});
    let after=after.unwrap();
    for field in ["fate:point:A", "fate:point:B"] { assert_eq!(after.semantics.values().get(field),manual.semantics.values().get(field)); }
    let base=fresh(schema(&[("A",3),("B",4),("C",5)]));
    let pin=Value::list([Value::text("scope-value"),Value::Int(2)]);
    let scoped=resolve_configuration(ResolveRequest{schema:base.schema.clone(),base:Some(&base),semantic_updates:Some(ValueMap::from_pairs([("fate:point:A",pin)])),..Default::default()}).unwrap();
    for (name,length) in [("scope_lost",2),("scope_kept",4)] {
        let after=resolve_configuration(ResolveRequest{schema:Some(schema(&[("A",length),("B",4),("C",5)])),base:Some(&scoped),..Default::default()});
        let after=after.unwrap();
        let fate=after.semantics.values().get("fate:point:A").cloned().unwrap();
        assert_eq!(fate, if name=="scope_lost" {Value::text("facet")} else {Value::list([Value::text("scope-value"),Value::Int(2)])});
    }
    let base=fresh(schema(&[("A",3),("B",4)]));
    let manual=resolve_configuration(ResolveRequest{schema:base.schema.clone(),base:Some(&base),semantic_updates:Some(ValueMap::from_pairs([("cell_kind",Value::text("image"))])),..Default::default()}).unwrap();
    let after=resolve_configuration(ResolveRequest{schema:Some(schema(&[("A",3)])),base:Some(&manual),..Default::default()});
    assert!(after.is_err(), "manual Image cannot become an automatic Curve");
    let image_schema=|axes:&[(&str,usize)]| {
        let s=schema(axes);let d=s.dataset();
        let cell=Domain::dense(vec![2,3],vec![
            Axis::new("img.y".into(),"y".into(),"spatial-y".into(),Coordinates::Index{origin:0,len:2},None,None,None,None).unwrap(),
            Axis::new("img.x".into(),"x".into(),"spatial-x".into(),Coordinates::Index{origin:0,len:3},None,None,None,None).unwrap()]).unwrap();
        PlotSchema::new(DatasetSchema::new(d.domain(zlc_core::schema::DomainRole::Repeat).clone(),d.domain(zlc_core::schema::DomainRole::Point).clone(),cell,d.value().clone()).unwrap())
    };
    for (label,axes) in [("image_reorder",vec![("B",4),("A",3)]),("image_replace",vec![("C",3),("B",4)])] {
        let base=fresh(image_schema(&[("A",3),("B",4)]));let next=image_schema(&axes);let expected=fresh(next.clone());
        let after=resolve_configuration(ResolveRequest{schema:Some(next),base:Some(&base),..Default::default()});
        assert_eq!(show(&after.unwrap()),show(&expected), "{label}");
    }}
