//! Scientific kernels are usable with typed values, without service messages.
use zlc_core::{AnyBuffer,DType,OwnedBuffer};
use zlc_core::scientific::{Tensor,binary,transform};
use zlc_core::scientific::groups::{Group,GroupConfig,GroupFeed};

#[test]
fn typed_accumulator_keeps_event_order_and_partial_group_uncertainty(){
    let mut group=Group::new(GroupConfig{operation:"mean".into(),domain:0,spanning:true,every:2,
        counts_r:vec![1],counts_p:vec![1],source_points:1,output_points:1,lines:None,steps:None}).unwrap();
    let feed=|ordinal,values|GroupFeed{data:Tensor::from_reals(values,vec![1,1,2]),flags:vec![true,true],sigma:None,
        rows:vec![0],points:vec![0],plans:vec![],ordinal};
    assert!(group.feed(feed(0,vec![1.0,2.0])).unwrap().is_empty());
    let chunks=group.feed(feed(1,vec![3.0,4.0])).unwrap();
    assert_eq!(chunks.len(),1);assert_eq!(chunks[0].rows,[0]);assert_eq!(chunks[0].columns,[0]);
    assert_eq!(chunks[0].planes["values"].reals().unwrap(),[2.0,3.0]);
    assert_eq!(chunks[0].planes["sigma"].reals().unwrap(),[1.0,1.0]);
    assert!(group.feed(feed(2,vec![5.0,6.0])).unwrap().is_empty());
    let partial=group.finish().unwrap();assert_eq!(partial[0].rows,[1]);
    assert_eq!(partial[0].planes["values"].reals().unwrap(),[5.0,6.0]);
    assert!(partial[0].planes["sigma"].reals().unwrap().iter().all(|v|v.is_nan()));
    assert_eq!(group.state().open_groups,0);
}

#[test]
fn typed_selection_and_integer_arithmetic_preserve_wide_bits(){
    let tensor=Tensor{values:OwnedBuffer::from_vec(vec![u64::MAX,9_007_199_254_740_993]).into(),shape:vec![2]};
    let selected=transform::selected(vec![Some(tensor)],&[transform::SelectionStep{axis:0,indexes:vec![1,0],drop:false}],false).unwrap();
    let selected=selected.into_iter().next().unwrap().unwrap();
    let output=binary::apply(binary::Request{left:binary::Input::Tensor(selected),right:binary::Input::Tensor(Tensor{
        values:OwnedBuffer::from_vec(vec![1u64,1]).into(),shape:vec![2]}),shape:vec![2],valid:None,
        operation:"+".into(),compute_dtype:DType::U64,dtype:DType::U64}).unwrap();
    let AnyBuffer::U64(values)=output.values.values else{panic!("integer result was widened")};
    assert_eq!(values.as_slice(),[9_007_199_254_740_994,0]);
    assert_eq!(output.valid.flags().unwrap(),[true,true]);
}
