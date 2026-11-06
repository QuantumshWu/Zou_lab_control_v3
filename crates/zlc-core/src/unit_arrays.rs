//! Typed bulk unit conversion. Each stage retains NumPy's established dtype
//! contract; nonlinear operations which explicitly widen keep doing so.
use super::{Conversion,Unit,UnitError};
use crate::{AnyBuffer,Complex64,Complex128,F16,OwnedBuffer};

pub struct ArrayConversion { pub values:AnyBuffer, pub discarded_imaginary:bool }

fn complex32(value:Complex64,factor:f32,inverse:bool)->Complex64 {
    if !inverse{return Complex64{re:value.re*factor-value.im*0.0,im:value.re*0.0+value.im*factor};}
    if factor==0.0{return Complex64{re:value.re/factor,im:value.im/factor};}
    let ratio=0.0/factor;let reciprocal=1.0/(factor+0.0*ratio);
    Complex64{re:(value.re+value.im*ratio)*reciprocal,im:(value.im-value.re*ratio)*reciprocal}
}
fn complex64(value:Complex128,factor:f64,inverse:bool)->Complex128 {
    if !inverse{return Complex128{re:value.re*factor-value.im*0.0,im:value.re*0.0+value.im*factor};}
    if factor==0.0{return Complex128{re:value.re/factor,im:value.im/factor};}
    let ratio=0.0/factor;let reciprocal=1.0/(factor+0.0*ratio);
    Complex128{re:(value.re+value.im*ratio)*reciprocal,im:(value.im-value.re*ratio)*reciprocal}
}

fn wide(values:&AnyBuffer)->ArrayConversion {
    let mut discarded=false;
    macro_rules! real{($v:expr)=>{$v.as_slice().iter().map(|v|*v as f64).collect::<Vec<_>>()};}
    let values=match values{
        AnyBuffer::Bool(v)=>v.as_slice().iter().map(|v|if *v{1.0}else{0.0}).collect(),
        AnyBuffer::I8(v)=>real!(v),AnyBuffer::U8(v)=>real!(v),AnyBuffer::I16(v)=>real!(v),AnyBuffer::U16(v)=>real!(v),
        AnyBuffer::I32(v)=>real!(v),AnyBuffer::U32(v)=>real!(v),AnyBuffer::I64(v)=>real!(v),AnyBuffer::U64(v)=>real!(v),
        AnyBuffer::F16(v)=>v.as_slice().iter().map(|v|v.to_f64()).collect(),AnyBuffer::F32(v)=>real!(v),AnyBuffer::F64(v)=>v.as_slice().to_vec(),
        AnyBuffer::C64(v)=>{discarded=true;v.as_slice().iter().map(|v|v.re as f64).collect()},
        AnyBuffer::C128(v)=>{discarded=true;v.as_slice().iter().map(|v|v.re).collect()},
    };
    ArrayConversion{values:OwnedBuffer::from_vec(values).into(),discarded_imaginary:discarded}
}

impl Conversion {
    /// A Python ArrayLike is explicitly cast to real float64 at the first
    /// step of these conversions. Frontends use this fact to preserve their
    /// existing coercion errors/warnings, without implementing unit arithmetic.
    pub fn initial_real_input(&self,inverse:bool)->bool {
        match self{Self::Scaled(_)=>false,Self::Decibel{..}=>inverse,
            Self::Offset(_)|Self::VoltageIntoLoad{..}=>true,
            Self::Prefixed{inner,..}=>if inverse{inner.initial_real_input(true)}else{true}}
    }

    pub fn apply_array(&self,input:&AnyBuffer,inverse:bool)->ArrayConversion {
        match self {
            Self::Prefixed{inner,factor}=>{
                if inverse{
                    let first=inner.apply_array(input,true);let mut real=wide(&first.values);
                    real.discarded_imaginary|=first.discarded_imaginary;
                    let mut answer=Self::Scaled(*factor).apply_array(&real.values,true);answer.discarded_imaginary|=real.discarded_imaginary;answer
                }else{
                    let real=wide(input);let scaled=Self::Scaled(*factor).apply_array(&real.values,false);
                    let mut answer=inner.apply_array(&scaled.values,false);answer.discarded_imaginary|=real.discarded_imaginary;answer
                }
            }
            Self::Scaled(factor)=>{
                let f=*factor;
                let values=match input{
                    AnyBuffer::F16(v)=>{let f=F16::from_f64(f).to_f64() as f32;OwnedBuffer::from_vec(v.as_slice().iter().map(|v|{
                        let x=v.to_f64() as f32;F16::from_f64(if inverse{(x/f) as f64}else{(x*f) as f64})}).collect::<Vec<_>>()).into()},
                    AnyBuffer::F32(v)=>{let f=f as f32;OwnedBuffer::from_vec(v.as_slice().iter().map(|v|if inverse{*v/f}else{*v*f}).collect::<Vec<_>>()).into()},
                    AnyBuffer::C64(v)=>OwnedBuffer::from_vec(v.as_slice().iter().map(|v|complex32(*v,f as f32,inverse)).collect::<Vec<_>>()).into(),
                    AnyBuffer::C128(v)=>OwnedBuffer::from_vec(v.as_slice().iter().map(|v|complex64(*v,f,inverse)).collect::<Vec<_>>()).into(),
                    _=>{let real=wide(input);let AnyBuffer::F64(v)=real.values else{unreachable!()};OwnedBuffer::from_vec(v.as_slice().iter().map(|v|if inverse{*v/f}else{*v*f}).collect::<Vec<_>>()).into()},
                };ArrayConversion{values,discarded_imaginary:false}
            }
            Self::Decibel{reference,factor} if !inverse=>{
                let (reference,factor)=(*reference,*factor);
                let values=match input{
                    AnyBuffer::F16(v)=>{let r=F16::from_f64(reference).to_f64() as f32;let f=F16::from_f64(factor).to_f64() as f32;
                        OwnedBuffer::from_vec(v.as_slice().iter().map(|v|{
                            let x=F16::from_f64((v.to_f64() as f32/f) as f64).to_f64() as f32;
                            let power=F16::from_f64(10_f32.powf(x) as f64).to_f64() as f32;
                            F16::from_f64((r*power) as f64)}).collect::<Vec<_>>()).into()},
                    AnyBuffer::F32(v)=>{let r=reference as f32;let f=factor as f32;OwnedBuffer::from_vec(v.as_slice().iter().map(|v|r*10_f32.powf(*v/f)).collect::<Vec<_>>()).into()},
                    AnyBuffer::C64(v)=>{let r=reference as f32;let f=factor as f32;OwnedBuffer::from_vec(v.as_slice().iter().map(|v|{
                        let exponent=complex32(*v,f,true);let(a,b)=(exponent.re,exponent.im);
                        let power=if b==0.0{Complex64{re:10_f32.powf(a),im:b}}else{
                            let amplitude=(a*10_f32.ln()).exp();let angle=b*10_f32.ln();Complex64{re:amplitude*angle.cos(),im:amplitude*angle.sin()}};
                        complex32(power,r,false)
                    }).collect::<Vec<_>>()).into()},
                    AnyBuffer::C128(v)=>OwnedBuffer::from_vec(v.as_slice().iter().map(|v|{
                        let exponent=complex64(*v,factor,true);let(a,b)=(exponent.re,exponent.im);
                        let power=if b==0.0{Complex128{re:10_f64.powf(a),im:b}}else{
                            let amplitude=(a*10_f64.ln()).exp();let angle=b*10_f64.ln();Complex128{re:amplitude*angle.cos(),im:amplitude*angle.sin()}};
                        complex64(power,reference,false)
                    }).collect::<Vec<_>>()).into(),
                    _=>{let real=wide(input);let AnyBuffer::F64(v)=real.values else{unreachable!()};OwnedBuffer::from_vec(v.as_slice().iter().map(|v|self.to_base(*v)).collect::<Vec<_>>()).into()},
                };ArrayConversion{values,discarded_imaginary:false}
            }
            _=>{
                let real=wide(input);let AnyBuffer::F64(v)=real.values else{unreachable!()};
                ArrayConversion{values:OwnedBuffer::from_vec(v.as_slice().iter().map(|v|if inverse{self.from_base(*v)}else{self.to_base(*v)}).collect::<Vec<_>>()).into(),discarded_imaginary:real.discarded_imaginary}
            }
        }
    }
}

impl Unit {
    pub fn convert_array_to(&self,values:&AnyBuffer,target:&Unit)->Result<ArrayConversion,UnitError>{
        // Use the same dimension refusal as scalar conversion, before a
        // possibly empty array could evade it.
        self.convert_value_to(0.0,target)?;
        if self==target{return Ok(ArrayConversion{values:values.clone(),discarded_imaginary:false});}
        let first=self.conversion.apply_array(values,false);let mut result=target.conversion.apply_array(&first.values,true);
        result.discarded_imaginary|=first.discarded_imaginary;Ok(result)
    }
}
