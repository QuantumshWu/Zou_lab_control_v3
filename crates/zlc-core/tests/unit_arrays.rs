//! Regression values generated with NumPy 2.4's public unit-array contract.
use zlc_core::{AnyBuffer,Complex64,F16,OwnedBuffer};
use zlc_core::units::{Conversion,Unit,DEFAULT_UNITS};

#[test]
fn scaled_arrays_keep_float32_stage_rounding_and_integer_identity_ownership(){
    let input=OwnedBuffer::from_vec(vec![1.2345_f32,33.333,-44.44]).into();
    let output=DEFAULT_UNITS.resolve("ms").unwrap().convert_array_to(&input,&DEFAULT_UNITS.resolve("µs").unwrap()).unwrap();
    let AnyBuffer::F32(values)=output.values else{panic!("linear float32 conversion was widened")};
    assert_eq!(values.as_slice().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),[1150963713,1191326977,3341654016]);
    let input=OwnedBuffer::from_vec(vec![u64::MAX,9_007_199_254_740_993]).into();
    let unit=DEFAULT_UNITS.resolve("count").unwrap();let output=unit.convert_array_to(&input,&unit).unwrap();
    assert_eq!(input.as_bytes().as_ptr(),output.values.as_bytes().as_ptr());
}

#[test]
fn complex_scaling_rounds_before_nonlinear_widening_and_reports_discard(){
    let input=OwnedBuffer::from_vec(vec![Complex64{re:12.3,im:0.3}]).into();
    let conversion=Conversion::Prefixed{inner:Box::new(Conversion::Scaled(3.3)),factor:0.01};
    let output=conversion.apply_array(&input,true);
    assert!(output.discarded_imaginary);
    let AnyBuffer::F64(values)=output.values else{panic!("Prefixed inverse must widen")};
    assert_eq!(values.as_slice(),[372.72729873657227]);
    let negative=Conversion::VoltageIntoLoad{load_ohms:50.0,divisor:8.0}.apply_array(&OwnedBuffer::from_vec(vec![-0.04,0.0,0.04]).into(),false);
    let AnyBuffer::F64(values)=negative.values else{unreachable!()};
    assert!(values.as_slice()[0].is_nan());assert_eq!(values.as_slice()[1],0.0);assert_eq!(values.as_slice()[2],0.04*0.04/400.0);
    let incompatible=Unit::new("x","different",Conversion::Scaled(1.0),&[],&[],None).unwrap();
    assert!(incompatible.convert_array_to(&OwnedBuffer::from_vec(Vec::<f64>::new()).into(),&DEFAULT_UNITS.resolve("Hz").unwrap()).is_err());
}

#[test]
fn explicit_half_conversion_roundtrips_all_finite_storage_values(){
    for bits in 0u16..=u16::MAX{let value=F16::from_bits(bits);if value.to_f64().is_finite(){assert_eq!(F16::from_f64(value.to_f64()).to_bits(),bits);}}
    assert_eq!(F16::from_f64(1.0+2.0_f64.powi(-11)).to_bits(),0x3c00);
    assert_eq!(F16::from_f64(1.0+3.0*2.0_f64.powi(-11)).to_bits(),0x3c02);
}
