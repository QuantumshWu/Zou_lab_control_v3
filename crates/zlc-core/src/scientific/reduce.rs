use std::collections::BTreeMap;
use crate::AnyBuffer;
use super::{Tensor, numeric, statistics};
use super::tensor::product;

#[derive(Clone)]
pub enum Step { Mapped { axis: usize, codes: Vec<usize>, size: usize }, Dense { axes: Vec<usize>, expand: bool, shape: Vec<usize>, kept_positions: Vec<usize> } }

fn dimensions(shape: &[usize], step: &Step) -> Result<Vec<usize>, String> {
    if shape.contains(&0) { return Err("a reduction cannot have an empty dimension".into()); }
    let mut out = shape.to_vec();
    match step {
        Step::Mapped { axis, codes, size } => {
            if *axis >= shape.len() || codes.len() != shape[*axis] || *size == 0 || codes.iter().any(|n| *n >= *size) {
                return Err("mapped reduction codes must cover the selected dimension".into());
            }
            out[*axis] = *size;
        }
        Step::Dense { axes, expand, .. } => {
            let mut seen = std::collections::HashSet::new();
            if axes.iter().any(|axis| *axis >= shape.len() || !seen.insert(*axis)) { return Err("dense reduction axes must be unique valid dimensions".into()); }
            out = shape.iter().enumerate().filter_map(|(i, n)| (!axes.contains(&i)).then_some(*n)).collect();
            if *expand { out.push(1); }
        }
    }
    product(&out)?;
    Ok(out)
}

fn mapped(index: usize, shape: &[usize], axis: usize, size: usize, codes: &[usize]) -> usize {
    let inner = shape[axis+1..].iter().product::<usize>();
    (index / (inner * shape[axis]) * size + codes[(index / inner) % shape[axis]]) * inner + index % inner
}

fn dense_index(mut index: usize, shape: &[usize], axes: &[usize]) -> usize {
    let mut out = 0;
    let mut stride = 1;
    for (axis, length) in shape.iter().enumerate().rev() {
        let position = index % length; index /= length;
        if !axes.contains(&axis) { out += position * stride; stride *= length; }
    }
    out
}

fn output_indexes(shape: &[usize], steps: &[Step]) -> Result<(Vec<usize>, Vec<usize>), String> {
    let mut current = shape.to_vec();
    let mut indexes: Vec<usize> = (0..product(shape)?).collect();
    for step in steps {
        let next = dimensions(&current, step)?;
        for index in &mut indexes { *index = match step {
            Step::Mapped { axis, codes, size } => mapped(*index, &current, *axis, *size, codes),
            Step::Dense { axes, .. } => dense_index(*index, &current, axes),
        }; }
        current = next;
    }
    Ok((current, indexes))
}

fn extreme(a: f64, b: f64, min: bool) -> f64 {
    if a.is_nan() || if min { a < b } else { a > b } { a } else { b }
}
fn flag(value: bool) -> f64 { if value { 1.0 } else { 0.0 } }

pub fn apply(mut values: Vec<f64>, shape: &[usize], steps: &[Step], operation: &str) -> Result<Vec<f64>, String> {
    let mut shape = shape.to_vec();
    for step in steps {
        let next = dimensions(&shape, step)?;
        values = match step {
            Step::Mapped { axis, codes, size } => {
                let fill = match operation { "min" => f64::INFINITY, "max" => f64::NEG_INFINITY, "all" => 1.0, _ => 0.0 };
                let mut out = vec![fill; product(&next)?];
                for (index, value) in values.into_iter().enumerate() {
                    let slot = &mut out[mapped(index, &shape, *axis, *size, codes)];
                    *slot = match operation { "min" => extreme(*slot, value, true), "max" => extreme(*slot, value, false),
                        "any" => flag(*slot != 0.0 || value != 0.0), "all" => flag(*slot != 0.0 && value != 0.0), _ => *slot + value };
                }
                out
            }
            Step::Dense { axes, .. } => match operation {
                "min" | "max" => numeric::extreme_reduce(&values, &shape, axes, None, None, numeric::Extremes::of('f', 8), operation == "min"),
                "any" | "all" => {
                    let mut out = vec![if operation == "all" { 1.0 } else { 0.0 }; product(&next)?];
                    for (i, v) in values.iter().enumerate() { let slot = &mut out[dense_index(i, &shape, axes)];
                        *slot = flag(if operation == "all" { *slot != 0.0 && *v != 0.0 } else { *slot != 0.0 || *v != 0.0 }); }
                    out
                }
                _ => numeric::add_reduce(&values, &shape, axes, None, false),
            },
        };
        shape = next;
    }
    Ok(values)
}

fn counts(flags: &[bool], shape: &[usize], steps: &[Step]) -> Result<Vec<i64>, String> {
    let mut values: Vec<i64> = flags.iter().map(|v| i64::from(*v)).collect();
    let mut shape = shape.to_vec();
    for step in steps {
        let next = dimensions(&shape, step)?;
        values = match step {
            Step::Mapped { axis, codes, size } => {
                let mut out = vec![0i64; product(&next)?];
                for (index, value) in values.into_iter().enumerate() { let slot = &mut out[mapped(index, &shape, *axis, *size, codes)];
                    *slot = slot.checked_add(value).ok_or("sample count overflows int64")?; }
                out
            }
            Step::Dense { axes, .. } => numeric::int_add_reduce(&values, &shape, axes, None, |v| v as i128).into_iter()
                .map(|v| i64::try_from(v).map_err(|_| "sample count overflows int64".to_owned())).collect::<Result<_,_>>()?,
        };
        shape = next;
    }
    Ok(values)
}

pub type Planes = BTreeMap<String, Tensor>;

pub fn measure(data: &Tensor, flags: &[bool], sigma: Option<&Tensor>, steps: &[Step], operation: &str) -> Result<Planes, String> {
    let values = data.reals()?;
    if flags.len() != values.len() { return Err("validity shape differs from values".into()); }
    let flags: Vec<bool> = flags.iter().zip(&values).map(|(v,x)| *v && x.is_finite()).collect();
    if !["sum", "mean", "std", "count", "min", "max", "any", "all"].contains(&operation) { return Err("unknown statistic".into()); }
    if matches!(operation, "any" | "all") && !matches!(data.values, AnyBuffer::Bool(_)) { return Err("any/all require boolean values".into()); }
    let (shape, indexes) = output_indexes(&data.shape, steps)?;
    let n = counts(&flags, &data.shape, steps)?;
    let mut part = Planes::new();
    if matches!(operation, "mean" | "std") {
        let mut first = vec![None; n.len()];
        for (i, valid) in flags.iter().enumerate() { if *valid && first[indexes[i]].is_none() { first[indexes[i]] = Some(i); } }
        let anchors: Vec<f64> = first.iter().map(|i| i.map_or(0.0, |i| values[i])).collect();
        let mut centered: Vec<f64> = values.iter().enumerate().map(|(i,v)| if flags[i] { *v-anchors[indexes[i]] } else { 0.0 }).collect();
        let sums = apply(centered.clone(), &data.shape, steps, "sum")?;
        let local: Vec<f64> = sums.iter().zip(&n).map(|(sum,n)| if *n>0 { *sum / *n as f64 } else { 0.0 }).collect();
        for (i, value) in centered.iter_mut().enumerate() { *value = if flags[i] { *value-local[indexes[i]] } else { 0.0 }; }
        let m2 = apply(centered.iter().map(|v| v*v).collect(), &data.shape, steps, "sum")?;
        part.insert("mean".into(), Tensor::from_reals(anchors.iter().zip(local).map(|(a,b)| *a+b).collect(), shape.clone()));
        part.insert("m2".into(), Tensor::from_reals(m2, shape.clone()));
        if operation == "mean" { if let Some(sigma) = sigma {
            if sigma.shape != data.shape { return Err("sigma shape differs from values".into()); }
            let squares = sigma.reals()?.iter().zip(&flags).map(|(v,valid)| if *valid { v*v } else { 0.0 }).collect();
            part.insert("stated".into(), Tensor::from_reals(apply(squares,&data.shape,steps,"sum")?,shape.clone()));
        } }
    } else if operation == "sum" {
        if matches!(data.values,AnyBuffer::Bool(_)) {
            let flags: Vec<bool> = flags.iter().zip(&values).map(|(v,x)| *v && *x!=0.0).collect();
            part.insert("total".into(), Tensor::from_counts(counts(&flags,&data.shape,steps)?,shape.clone()));
        } else {
            part.insert("total".into(),Tensor::from_reals(apply(values.iter().zip(&flags).map(|(v,ok)| if *ok {*v}else{0.0}).collect(),&data.shape,steps,"sum")?,shape.clone()));
        }
    } else if matches!(operation,"min"|"max") {
        let fill=if operation=="min"{f64::INFINITY}else{f64::NEG_INFINITY};
        part.insert("extreme".into(),Tensor::from_reals(apply(values.iter().zip(&flags).map(|(v,ok)| if *ok{*v}else{fill}).collect(),&data.shape,steps,operation)?,shape.clone()));
    } else if matches!(operation,"any"|"all") {
        let result=apply(values.iter().zip(&flags).map(|(v,ok)| if *ok{*v}else{flag(operation=="all")}).collect(),&data.shape,steps,operation)?;
        part.insert("flag".into(),Tensor::from_flags(result.iter().map(|v|*v!=0.0).collect(),shape.clone()));
    }
    part.insert("n".into(),Tensor::from_counts(n,shape));
    Ok(part)
}

pub fn result(mut part: Planes, operation: &str) -> Result<Planes,String> {
    let n=part.remove("n").ok_or("missing count")?;
    let counts=match &n.values { AnyBuffer::I64(v)=>v.as_slice().iter().map(|n|*n as u64).collect::<Vec<_>>(), _=>return Err("counts need int64".into()) };
    let values=match operation {
        "count"=>n.clone(), "sum"=>part.remove("total").ok_or("missing total")?, "min"|"max"=>part.remove("extreme").ok_or("missing extreme")?,
        "any"|"all"=>part.remove("flag").ok_or("missing flags")?,
        "std"=>Tensor::from_reals(part["m2"].reals()?.iter().zip(&counts).map(|(m,n)|if *n>0{(m / *n as f64).sqrt()}else{f64::NAN}).collect(),n.shape.clone()),
        "mean"=>Tensor::from_reals(part["mean"].reals()?.iter().zip(&counts).map(|(m,n)|if *n>0{*m}else{f64::NAN}).collect(),n.shape.clone()),
        _=>return Err("unknown statistic".into()),
    };
    let valid=Tensor::from_flags(counts.iter().enumerate().map(|(i,n)|*n>0&&match &values.values{
        AnyBuffer::F64(v)=>v.as_slice()[i].is_finite(),_=>true,
    }).collect(),n.shape.clone());
    let mut output=Planes::from([("values".into(),values),("valid".into(),valid)]);
    if operation=="mean" {
        let stated=part.get("stated").map(Tensor::reals).transpose()?;
        let sigma=statistics::sem_from_centred(&part["m2"].reals()?,&counts,stated.as_deref());
        output.insert("sigma".into(),Tensor::from_reals(sigma,n.shape.clone()));
        output.insert("dof".into(),Tensor::from_reals(counts.iter().map(|n|statistics::sem_dof(*n)).collect(),n.shape));
    }
    Ok(output)
}

/// One complete non-streaming statistic. The centred moments and the
/// sum/count estimate intentionally retain their established rounding order.
pub fn statistic(data:&Tensor,flags:&[bool],sigma:Option<&Tensor>,plans:&[Step],operation:&str,errors:bool)->Result<Planes,String>{
    let parts=measure(data,flags,sigma,plans,operation)?;
    let mut result=result(parts,operation)?;
    if operation=="mean" {
        let measured=measure(data,flags,None,plans,"sum")?;
        let sums=measured["total"].reals()?;let counts=measured["n"].reals()?;
        let values:Vec<f64>=sums.iter().zip(counts).map(|(s,n)|if n>0.0{*s/n}else{f64::NAN}).collect();
        let valid=result["valid"].flags()?.iter().zip(&values).map(|(v,x)|*v&&x.is_finite()).collect();
        result.insert("valid".into(),Tensor::from_flags(valid,result["values"].shape.clone()));
        result.insert("values".into(),Tensor::from_reals(values,result["values"].shape.clone()));
    }
    if !errors{result.remove("sigma");result.remove("dof");}
    Ok(result)
}
