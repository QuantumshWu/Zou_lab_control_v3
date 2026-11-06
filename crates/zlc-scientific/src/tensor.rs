use zlc_core::{AnyBuffer, OwnedBuffer};

#[derive(Clone)]
pub struct Tensor { pub values: AnyBuffer, pub shape: Vec<usize> }

pub(crate) fn product(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |n, size| n.checked_mul(*size).ok_or("tensor shape overflows usize".into()))
}

pub fn half_to_f64(bits: u16) -> f64 { zlc_core::F16::from_bits(bits).to_f64() }

impl Tensor {
    pub fn validate(&self) -> Result<(), String> {
        if product(&self.shape)? != self.values.len() { return Err("tensor values do not match shape".into()); }
        Ok(())
    }
    pub fn reals(&self) -> Result<Vec<f64>, String> {
        self.validate()?;
        macro_rules! cast { ($v:expr) => { $v.as_slice().iter().map(|v| *v as f64).collect() }; }
        Ok(match &self.values {
            AnyBuffer::Bool(v) => v.as_slice().iter().map(|v| if *v { 1.0 } else { 0.0 }).collect(),
            AnyBuffer::U8(v) => cast!(v), AnyBuffer::I8(v) => cast!(v), AnyBuffer::U16(v) => cast!(v), AnyBuffer::I16(v) => cast!(v),
            AnyBuffer::U32(v) => cast!(v), AnyBuffer::I32(v) => cast!(v), AnyBuffer::U64(v) => cast!(v), AnyBuffer::I64(v) => cast!(v),
            AnyBuffer::F16(v) => v.as_slice().iter().map(|v| half_to_f64(v.to_bits())).collect(), AnyBuffer::F32(v) => cast!(v),
            AnyBuffer::F64(v) => v.as_slice().to_vec(), _ => return Err("scientific operations need real numeric tensors".into()),
        })
    }
    pub fn flags(&self) -> Result<Vec<bool>, String> {
        self.validate()?;
        match &self.values { AnyBuffer::Bool(v) => Ok(v.as_slice().to_vec()), _ => Err("validity must be a bool tensor".into()) }
    }
    pub fn from_reals(values: Vec<f64>, shape: Vec<usize>) -> Self { Self { values: OwnedBuffer::from_vec(values).into(), shape } }
    pub fn from_flags(values: Vec<bool>, shape: Vec<usize>) -> Self { Self { values: OwnedBuffer::from_vec(values).into(), shape } }
    pub fn from_counts(values: Vec<i64>, shape: Vec<usize>) -> Self { Self { values: OwnedBuffer::from_vec(values).into(), shape } }

    /// Selection preserves every stored bit and dtype, including u64 values
    /// which cannot be represented by a floating intermediate.
    pub fn gather(&self, indexes: &[usize], shape: Vec<usize>) -> Result<Self, String> {
        self.validate()?;
        if product(&shape)? != indexes.len() || indexes.iter().any(|index| *index >= self.values.len()) { return Err("selection indexes lie outside tensor".into()); }
        macro_rules! select { ($v:expr) => { OwnedBuffer::from_vec(indexes.iter().map(|i| $v.as_slice()[*i]).collect::<Vec<_>>()).into() }; }
        let values = match &self.values {
            AnyBuffer::Bool(v) => select!(v), AnyBuffer::U8(v) => select!(v), AnyBuffer::I8(v) => select!(v),
            AnyBuffer::U16(v) => select!(v), AnyBuffer::I16(v) => select!(v), AnyBuffer::U32(v) => select!(v), AnyBuffer::I32(v) => select!(v),
            AnyBuffer::U64(v) => select!(v), AnyBuffer::I64(v) => select!(v), AnyBuffer::F16(v) => select!(v), AnyBuffer::F32(v) => select!(v),
            AnyBuffer::F64(v) => select!(v), AnyBuffer::C64(v) => select!(v), AnyBuffer::C128(v) => select!(v),
        };
        Ok(Self { values, shape })
    }

    pub(crate) fn join(parts: &[&Tensor], shape: Vec<usize>) -> Result<Self,String> {
        let first=parts.first().ok_or("cannot join no tensors")?;
        if parts.iter().any(|p|p.values.dtype()!=first.values.dtype()) {return Err("joined tensor dtypes differ".into());}
        macro_rules! joined {($kind:ident)=>{{let mut values=Vec::new();for part in parts{if let AnyBuffer::$kind(v)=&part.values{values.extend_from_slice(v.as_slice());}}OwnedBuffer::from_vec(values).into()}};}
        let values=match &first.values{AnyBuffer::Bool(_)=>joined!(Bool),AnyBuffer::I64(_)=>joined!(I64),AnyBuffer::F64(_)=>joined!(F64),_=>return Err("unsupported statistic dtype".into())};
        let out=Self{values,shape};out.validate()?;Ok(out)
    }
}

impl crate::numeric::Widen for zlc_core::F16 {
    fn widen(self) -> f64 { half_to_f64(self.to_bits()) }
}
