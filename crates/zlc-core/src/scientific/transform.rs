//! Typed schema alignment and bit-preserving tensor selection.
use crate::schema::{DatasetSchema,DomainRole};
use super::Tensor;
use super::tensor::product;

#[derive(Clone)]
pub struct SelectionStep { pub axis:usize, pub indexes:Vec<usize>, pub drop:bool }
#[derive(Clone,Default)]
pub struct Alignment { pub rows:Option<Vec<usize>>, pub points:Option<Vec<usize>>, pub cells:Option<Vec<usize>> }

pub fn selected(planes:Vec<Option<Tensor>>,steps:&[SelectionStep],expand:bool)->Result<Vec<Option<Tensor>>,String>{
    let mut steps=steps.to_vec();steps.sort_by_key(|step|std::cmp::Reverse(step.axis));
    planes.into_iter().map(|plane|{
        let Some(mut tensor)=plane else{return Ok(None)};
        for SelectionStep{axis,indexes,drop} in &steps{
            if *axis>=tensor.shape.len()||indexes.is_empty()||indexes.iter().any(|i|*i>=tensor.shape[*axis])||(*drop&&indexes.len()!=1){return Err("invalid selection dimension or indexes".into());}
            let inner=product(&tensor.shape[axis+1..])?;let outer=product(&tensor.shape[..*axis])?;
            let mut shape=tensor.shape.clone();if *drop{shape.remove(*axis);}else{shape[*axis]=indexes.len();}
            let mut chosen=Vec::new();chosen.try_reserve_exact(product(&shape)?).map_err(|e|format!("selection is too large: {e}"))?;
            for row in 0..outer{for coordinate in indexes{for column in 0..inner{chosen.push((row*tensor.shape[*axis]+coordinate)*inner+column);}}}
            tensor=tensor.gather(&chosen,shape)?;
        }
        if expand{tensor.shape.push(1);}Ok(Some(tensor))
    }).collect()
}

pub fn aligned(planes:Vec<Option<Tensor>>,plan:Option<&Alignment>,shape:&[usize])->Result<Vec<Option<Tensor>>,String>{
    let count=product(shape)?;
    planes.into_iter().map(|plane|{
        let Some(tensor)=plane else{return Ok(None)};
        if tensor.shape.len()<2||shape.len()<2{return Err("alignment needs three-domain storage".into());}
        let rows=plan.and_then(|p|p.rows.as_deref());let points=plan.and_then(|p|p.points.as_deref());
        let cells=plan.and_then(|p|p.cells.as_deref()).unwrap_or(&tensor.shape[2..]);
        if shape.len()!=cells.len()+2{return Err("alignment physical shapes differ".into());}
        if rows.is_some_and(|v|v.len()!=shape[0])||points.is_some_and(|v|v.len()!=shape[1]){return Err("alignment row maps have wrong length".into());}
        if cells.iter().zip(&shape[2..]).any(|(a,b)|*a!=1&&a!=b){return Err("cell dimensions cannot broadcast".into());}
        let mut indexes=Vec::with_capacity(count);let cell_count=product(cells)?;
        if cell_count!=product(&tensor.shape[2..])?{return Err("aligned cell reshape changes its size".into());}
        if rows.is_some_and(|v|v.iter().any(|n|*n>=tensor.shape[0]))||points.is_some_and(|v|v.iter().any(|n|*n>=tensor.shape[1])){return Err("alignment row is outside input".into());}
        let target_cells=product(&shape[2..])?;
        for flat in 0..count{
            let r=flat/(shape[1]*target_cells);let p=flat/target_cells%shape[1];
            let mut rem=flat%target_cells;let mut c=0;let mut stride=1;
            for(target,own)in shape[2..].iter().zip(cells).rev(){let at=rem%target;rem/=target;c+=if *own==1{0}else{at*stride};stride*=own;}
            let r=rows.map_or(r,|v|v[r]);let p=points.map_or(p,|v|v[p]);indexes.push((r*tensor.shape[1]+p)*cell_count+c);
        }
        Ok(Some(tensor.gather(&indexes,shape.to_vec())?))
    }).collect()
}

pub fn masked(valid:&Tensor,mask:&Tensor,judged:&Tensor,negate:bool)->Result<Tensor,String>{
    let valid=valid.flags()?;let shape=mask.shape.clone();let mask=mask.flags()?;let judged=judged.flags()?;
    if valid.len()!=mask.len()||judged.len()!=mask.len(){return Err("where mask dimensions differ".into());}
    Ok(Tensor::from_flags(valid.iter().zip(mask).zip(judged).map(|((v,m),j)|*v&&j&&(m!=negate)).collect(),shape))
}

pub fn alignment(target:&DatasetSchema,other:&DatasetSchema)->Result<Option<Alignment>,String>{
    let roles=[DomainRole::Repeat,DomainRole::Point,DomainRole::Cell];
    if roles.iter().all(|role|target.domain(*role)==other.domain(*role)){return Ok(None);}
    let mut result=Alignment::default();
    for(d,role)in roles.into_iter().enumerate(){
        let wanted=target.domain(role);let given=other.domain(role);
        if wanted==given{continue;}
        let scalar=d==2&&given.size()==1&&given.axes().len()==1&&given.axes()[0].role()=="scalar";
        let axes=if scalar{&[][..]}else{given.axes()};
        if axes.iter().any(|axis|!wanted.axes().contains(axis)){return Err("an axis of one side is not on the other".into());}
        if d==2{
            let positions=axes.iter().map(|axis|wanted.axes().iter().position(|other|other==axis).unwrap()).collect::<Vec<_>>();
            if positions.windows(2).any(|v|v[0]>v[1]){return Err("the cell axes are in another order".into());}
            result.cells=Some(wanted.axes().iter().map(|axis|if axes.contains(axis){axis.len()}else{1}).collect());continue;
        }
        let rows=if axes.is_empty(){if given.size()!=1{return Err("unnamed rows cannot be matched".into());}vec![0;wanted.size()]}else{
            let key=|domain:&crate::schema::Domain,row|axes.iter().map(|axis|domain.mapping(axis.id()).unwrap().code(row).unwrap()).collect::<Vec<_>>();
            let mut positions=std::collections::HashMap::new();for row in 0..given.size(){positions.insert(key(given,row),row);}
            (0..wanted.size()).map(|row|positions.get(&key(wanted,row)).copied().ok_or("one side has no value for some coordinates of the other".to_owned())).collect::<Result<Vec<_>,_>>()?
        };
        if d==0{result.rows=Some(rows);}else{result.points=Some(rows);}
    }
    Ok(Some(result))
}
