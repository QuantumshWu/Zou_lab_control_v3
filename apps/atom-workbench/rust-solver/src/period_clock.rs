//! Output-time bookkeeping for a complete period map. No phase is snapped.
//! Declared MHz axes are authoritative only when their integer reduction is
//! exactly representable; all other inputs retain the angular-period path.
use crate::{harmonics::FrequencyBasis,Model};

const MAX_COUNT:f64=4503599627370495.;
#[derive(Clone,Copy,Debug)]
pub(super) struct Clock{period:f64,mhz:Option<f64>}
impl Clock{
    pub(super) fn new(model:&Model)->Result<Self,String>{
        let basis=crate::harmonics::model_frequency_basis(model,16)?;
        if !basis.exact||basis.rounded_relations!=0||basis.frequencies.len()!=1||basis.frequencies[0]<=0.{
            return Err("Full period propagation requires one exact commensurate drive frequency; no carrier was rounded".into());
        }
        let mhz=model.frequency_coordinates.as_ref().and_then(|coordinates|coordinates.frequencies_mhz.as_deref())
            .and_then(|source|exact_mhz(source,&basis));
        let period=mhz.map_or_else(||std::f64::consts::TAU/basis.frequencies[0],|frequency|1./frequency);
        if !period.is_finite()||period<=0.{return Err("Invalid complete-period duration".into());}
        Ok(Self{period,mhz})
    }
    pub(super) fn period(self)->f64{self.period}
    pub(super) fn source(self)->&'static str{if self.mhz.is_some(){"exact-declared-MHz"}else{"angular-period"}}
    pub(super) fn split(self,time:f64)->Result<(u64,f64),String>{
        match self.mhz{Some(frequency)=>split_cycles(time,frequency,self.period),None=>split_period(time,self.period)}
    }
}

fn exact_mhz(source:&[f64],basis:&FrequencyBasis)->Option<f64>{
    let mapping=basis.source_to_basis.as_ref()?;
    if source.is_empty()||mapping.len()!=source.len()||mapping.iter().any(|row|row.len()!=1||row[0]<=0){return None;}
    // A quotient such as 100/3 is not an exactly representable base axis.
    // Require every source relation to reconstruct with zero FMA residual,
    // rather than accepting a close ratio or introducing a phase tolerance.
    source.iter().zip(mapping).find_map(|(&value,row)|{
        let frequency=value/row[0]as f64;
        if !frequency.is_finite()||frequency<=0.||!frequency.recip().is_finite()
            ||basis.frequencies[0]!=frequency*std::f64::consts::TAU{return None;}
        source.iter().zip(mapping).all(|(&source,row)|{
            let coefficient=row[0]as f64;
            frequency*coefficient==source&&frequency.mul_add(coefficient,-source)==0.
        }).then_some(frequency)
    })
}

fn two_sum(a:f64,b:f64)->(f64,f64){let sum=a+b;let v=sum-a;(sum,(a-(sum-v))+(b-v))}
fn split_cycles(time:f64,frequency:f64,period:f64)->Result<(u64,f64),String>{
    if !time.is_finite()||time<0.{return Err("Invalid full-period output time".into());}
    let hi=time*frequency;let lo=time.mul_add(frequency,-hi);
    if !hi.is_finite()||hi>MAX_COUNT+1.{return Err("Full period count exceeds exact integer range".into());}
    let mut count=hi.floor();let(mut fraction,mut tail)=two_sum(hi-count,lo);
    // Compare the double-double value, including the sign of a low part
    // hidden when the rounded high product lands exactly on an integer.
    if fraction<0.||(fraction==0.&&tail<0.){count-=1.;let(sum,error)=two_sum(fraction,1.);(fraction,tail)=two_sum(sum,error+tail);}
    if fraction>1.||(fraction==1.&&tail>=0.){count+=1.;let(sum,error)=two_sum(fraction,-1.);(fraction,tail)=two_sum(sum,error+tail);}
    if count<0.||count>MAX_COUNT{return Err("Full period count exceeds exact integer range".into());}
    // With no complete cycle, the requested time already IS the phase.
    // Avoid a multiply/divide roundtrip, including underflow of tiny t*f.
    if count==0.{return Ok((0,time));}
    // Divide the retained fractional double-double. In particular, a value
    // just below one remains associated with the previous integer period;
    // an unavoidable final f64 phase==period does not increment that count.
    let leading=fraction/frequency;
    let phase=leading+((-leading).mul_add(frequency,fraction)+tail)/frequency;
    if !phase.is_finite()||phase<0.||phase>period{return Err("Invalid full-period remainder".into());}
    Ok((count as u64,phase))
}

fn split_period(time:f64,period:f64)->Result<(u64,f64),String>{
    let quotient=(time/period).floor();
    if !quotient.is_finite()||quotient<0.||quotient>MAX_COUNT{return Err("Full period count exceeds exact integer range".into());}
    let mut count=quotient as u64;let mut phase=(-(count as f64)).mul_add(period,time);
    if phase<0.&&count>0{count-=1;phase+=period;}
    if phase>=period{count+=1;phase-=period;}
    if phase<0.||phase>period{return Err("Invalid full-period remainder".into());}Ok((count,phase))
}

#[cfg(test)]
mod tests{
    use super::*;
    fn basis(mapping:Option<Vec<Vec<i32>>>,frequency:f64)->FrequencyBasis{FrequencyBasis{
        frequencies:vec![frequency*std::f64::consts::TAU],drive_indices:vec![],reconstructed_frequencies:vec![],reconstruction_errors:vec![],
        exact_relations:0,rounded_relations:0,exact:true,search_exhausted:false,source_to_basis:mapping}}
    #[test]
    fn declared_clock_preserves_exact_integer_and_half_cycles_without_near_period_partials(){
        let clock=Clock{period:0.01,mhz:Some(100.)};let mut zeros=0;let mut halves=0;
        for i in 0..161{let(count,phase)=clock.split(20.*i as f64/160.).unwrap();
            assert_eq!(count,(25*i/2)as u64);if i%2==0{assert_eq!(phase,0.);zeros+=1;}else{assert_eq!(phase,0.005);halves+=1;}}
        assert_eq!((zeros,halves),(81,80));assert!(split_period(0.25,0.01).unwrap().1>0.009);
    }
    #[test]
    fn neighboring_times_and_hidden_negative_product_tail_are_not_snapped(){
        let clock=Clock{period:0.01,mhz:Some(100.)};
        let below=clock.split(0.25_f64.next_down()).unwrap();let exact=clock.split(0.25).unwrap();let above=clock.split(0.25_f64.next_up()).unwrap();
        assert_eq!((below.0,exact.0,above.0),(24,25,25));assert!(below.1>0.&&below.1<0.01);assert_eq!(exact.1,0.);assert!(above.1>0.);
        assert_eq!(0.3*10.,3.);assert!(0.3_f64.mul_add(10.,-3.)<0.);
        let hidden=split_cycles(0.3,10.,0.1).unwrap();assert_eq!(hidden.0,2);assert!(hidden.1>0.&&hidden.1<0.1);
    }
    #[test]
    fn source_mhz_requires_every_exact_integer_relation_and_keeps_legacy_fallback(){
        assert_eq!(exact_mhz(&[100.,300.],&basis(Some(vec![vec![1],vec![3]]),100.)),Some(100.));
        assert_eq!(exact_mhz(&[200.,300.],&basis(Some(vec![vec![2],vec![3]]),100.)),Some(100.));
        assert_eq!(exact_mhz(&[100.],&basis(None,100.)),None);
        assert_eq!(exact_mhz(&[100.,300_f64.next_up()],&basis(Some(vec![vec![1],vec![3]]),100.)),None);
        assert_eq!(exact_mhz(&[100.],&basis(Some(vec![vec![3]]),100./3.)),None);
        let legacy=Clock{period:0.01,mhz:None};assert_eq!(legacy.split(0.25).unwrap(),split_period(0.25,0.01).unwrap());
    }
    #[test]
    fn count_range_and_invalid_times_remain_checked(){
        let clock=Clock{period:1.,mhz:Some(1.)};assert_eq!(clock.split(MAX_COUNT).unwrap(),(MAX_COUNT as u64,0.));
        assert_eq!(clock.split((MAX_COUNT+1.).next_down()).unwrap().0,MAX_COUNT as u64);
        for time in [-1.,f64::NAN,f64::INFINITY,MAX_COUNT+1.]{assert!(clock.split(time).is_err());}
        assert_eq!(split_cycles(1e-300,1e-100,1e100).unwrap(),(0,1e-300));
    }
}
