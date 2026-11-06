use serde_json::{Value,json};
use zlc_core::{AnyBuffer,DType,OwnedBuffer,F16};
use crate::Tensor;

fn dtype(name:&str)->Result<DType,String>{Ok(match name{
    "|b1"=>DType::Bool,"|u1"=>DType::U8,"|i1"=>DType::I8,"<u2"=>DType::U16,"<i2"=>DType::I16,
    "<u4"=>DType::U32,"<i4"=>DType::I32,"<u8"=>DType::U64,"<i8"=>DType::I64,"<f2"=>DType::F16,"<f4"=>DType::F32,"<f8"=>DType::F64,
    _=>return Err(format!("unsupported real dtype {name}")),})}

fn half(value:f64)->u16{F16::from_f64(value).to_bits()}

struct Argument{data:Option<Tensor>,float:Option<f64>,boolean:Option<bool>}
impl Argument{
    fn new(value:&Value,read:&dyn Fn(&Value)->Result<Tensor,String>,size:usize)->Result<Self,String>{
        if let Some(v)=value.as_bool(){return Ok(Self{data:None,float:None,boolean:Some(v)});}
        if let Some(v)=value.as_f64(){return Ok(Self{data:None,float:Some(v),boolean:None});}
        let data=read(value)?;data.validate()?;if data.values.len()!=size{return Err("binary operands must already have the planned shape".into());}
        Ok(Self{data:Some(data),float:None,boolean:None})
    }
    fn integer(&self,index:usize)->Result<i128,String>{
        if let Some(v)=self.boolean{return Ok(i128::from(v));}
        if self.float.is_some(){return Err("a floating scalar cannot enter an integer operation".into());}
        macro_rules! n{($v:expr)=>{Ok($v.as_slice()[index] as i128)}}
        match &self.data.as_ref().unwrap().values{AnyBuffer::Bool(v)=>Ok(i128::from(v.as_slice()[index])),AnyBuffer::I8(v)=>n!(v),AnyBuffer::U8(v)=>n!(v),
            AnyBuffer::I16(v)=>n!(v),AnyBuffer::U16(v)=>n!(v),AnyBuffer::I32(v)=>n!(v),AnyBuffer::U32(v)=>n!(v),AnyBuffer::I64(v)=>n!(v),AnyBuffer::U64(v)=>n!(v),_=>Err("integer operation received floats".into())}
    }
    fn floats(&self,size:usize,kind:DType)->Result<Vec<f64>,String>{
        let values=if let Some(v)=self.float{vec![v;size]}else if let Some(v)=self.boolean{vec![if v{1.0}else{0.0};size]}else{self.data.as_ref().unwrap().reals()?};
        Ok(values.into_iter().map(|v|match kind{DType::F32=>v as f32 as f64,DType::F16=>crate::tensor::half_to_f64(half(v)),_=>v}).collect())
    }
}

fn compared<T:PartialOrd+PartialEq>(op:&str,a:T,b:T)->Result<bool,String>{Ok(match op{"=="=>a==b,"!="=>a!=b,">"=>a>b,"<"=>a<b,">="=>a>=b,"<="=>a<=b,_=>return Err("unknown comparison".into())})}

pub(crate) fn dispatch(args:&Value,read:&dyn Fn(&Value)->Result<Tensor,String>,write:&dyn Fn(Tensor)->Result<Value,String>)->Result<Value,String>{
    let shape=crate::reduce::integers(&args["shape"],read)?;let size=crate::tensor::product(&shape)?;
    let compute=dtype(args["compute_dtype"].as_str().ok_or("compute dtype is required")?)?;
    let result=dtype(args["dtype"].as_str().ok_or("result dtype is required")?)?;
    let op=args["op"].as_str().ok_or("operator is required")?;
    let left=Argument::new(&args["left"],read,size)?;let right=Argument::new(&args["right"],read,size)?;
    let mut valid=if args["valid"].is_null(){vec![true;size]}else{read(&args["valid"])?.flags()?};
    if valid.len()!=size{return Err("binary validity has wrong size".into());}
    let integers=!matches!(compute,DType::F16|DType::F32|DType::F64);
    let values=if result==DType::Bool{
        let values=if integers{(0..size).map(|i|{let a=left.integer(i)?;let b=right.integer(i)?;match op{
            "and"=>Ok(a!=0&&b!=0),"or"=>Ok(a!=0||b!=0),"and not"=>Ok(a!=0&&b==0),_=>compared(op,a,b)}}).collect::<Result<Vec<_>,String>>()?}
        else{let a=left.floats(size,compute)?;let b=right.floats(size,compute)?;a.into_iter().zip(b).enumerate().map(|(i,(a,b))|{valid[i]&=a.is_finite()&&b.is_finite();compared(op,a,b)}).collect::<Result<Vec<_>,String>>()?};
        OwnedBuffer::from_vec(values).into()
    }else if integers{
        let mut values=Vec::with_capacity(size);
        for i in 0..size{let a=left.integer(i)?;let b=right.integer(i)?;values.push(match op{"+"=>a.wrapping_add(b),"-"=>a.wrapping_sub(b),"*"=>a.wrapping_mul(b),_=>return Err("integer operation must be +, - or *".into())});}
        macro_rules! cast{($t:ty)=>{OwnedBuffer::from_vec(values.iter().map(|v|*v as $t).collect::<Vec<_>>()).into()};}
        match result{DType::I8=>cast!(i8),DType::U8=>cast!(u8),DType::I16=>cast!(i16),DType::U16=>cast!(u16),DType::I32=>cast!(i32),DType::U32=>cast!(u32),DType::I64=>cast!(i64),DType::U64=>cast!(u64),_=>return Err("integer operation has a noninteger result".into())}
    }else{
        let a=left.floats(size,compute)?;let b=right.floats(size,compute)?;
        let mut values=Vec::with_capacity(size);
        for (i,(a,b)) in a.into_iter().zip(b).enumerate(){
            valid[i]&=a.is_finite()&&b.is_finite();
            let v=if matches!(compute,DType::F16|DType::F32){let a=a as f32;let b=b as f32;match op{"+"=>(a+b) as f64,"-"=>(a-b) as f64,"*"=>(a*b) as f64,"/"=>(a/b) as f64,_=>return Err("unknown arithmetic operator".into())}}
                else{match op{"+"=>a+b,"-"=>a-b,"*"=>a*b,"/"=>a/b,_=>return Err("unknown arithmetic operator".into())}};
            let v=match result{DType::F16=>crate::tensor::half_to_f64(half(v)),DType::F32=>v as f32 as f64,_=>v};valid[i]&=v.is_finite();values.push(v);
        }
        match result{DType::F16=>OwnedBuffer::from_vec(values.iter().map(|v|F16::from_bits(half(*v))).collect::<Vec<_>>()).into(),
            DType::F32=>OwnedBuffer::from_vec(values.iter().map(|v|*v as f32).collect::<Vec<_>>()).into(),DType::F64=>OwnedBuffer::from_vec(values).into(),_=>return Err("floating arithmetic has a nonfloating result".into())}
    };
    Ok(json!({"values":write(Tensor{values,shape:shape.clone()})?,"valid":write(Tensor::from_flags(valid,shape))?}))
}
