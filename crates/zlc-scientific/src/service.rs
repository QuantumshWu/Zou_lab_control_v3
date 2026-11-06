use serde_json::{Value, json};
use crate::{Tensor, reduce};
use crate::tensor::product;

#[derive(Default)]
pub struct Service { groups:std::collections::HashMap<String,crate::groups::Group> }

fn encode(planes: reduce::Planes, write: &dyn Fn(Tensor)->Result<Value,String>) -> Result<Value,String> {
    Ok(Value::Object(planes.into_iter().map(|(name,tensor)| Ok((name,write(tensor)?))).collect::<Result<_,String>>()?))
}

impl Service {
    pub fn dispatch(&mut self, operation: &str, args: &Value,
        read: &dyn Fn(&Value) -> Result<Tensor, String>,
        write: &dyn Fn(Tensor) -> Result<Value, String>) -> Result<Value, String> {
        let operation=operation.strip_prefix("scientific.").unwrap_or(operation);
        match operation {
            "axis"|"pick_target"|"pick_plan"|"reduce_plan"=>crate::geometry::dispatch(operation,args,read,write),
            "binary"=>crate::binary::dispatch(args,read,write),
            "where"=>{
                let valid=read(&args["valid"])?.flags()?;let mask=read(&args["mask"])?;
                let judged=read(&args["judged"])?.flags()?;let shape=mask.shape.clone();let mask=mask.flags()?;
                if valid.len()!=mask.len()||judged.len()!=mask.len(){return Err("where mask dimensions differ".into());}
                let negate=args["negate"].as_bool().unwrap_or(false);
                let result=valid.iter().zip(mask).zip(judged).map(|((v,m),j)|*v&&j&&(m!=negate)).collect();
                Ok(json!({"valid":write(Tensor::from_flags(result,shape))?}))
            }
            "group.create"=>{
                let group=crate::groups::Group::new(args,read)?;
                // Handles survive in warm Python objects. They must not alias
                // a counter in a new run service or a restarted offline helper.
                let handle=uuid::Uuid::new_v4().to_string();self.groups.insert(handle.clone(),group);Ok(json!({"handle":handle}))
            }
            "group.feed"=>{
                let handle=args["handle"].as_str().ok_or("group handle must be text")?;
                self.groups.get_mut(handle).ok_or("group handle is not current")?.feed(args,read,write)
            }
            "group.finish"=>{
                let handle=args["handle"].as_str().ok_or("group handle must be text")?;
                self.groups.remove(handle).ok_or("group handle is not current")?.finish(write)
            }
            "group.drop"=>{if let Some(handle)=args["handle"].as_str(){self.groups.remove(handle);}Ok(json!({}))}
            "group.state"=>{let handle=args["handle"].as_str().ok_or("group handle must be text")?;Ok(self.groups.get(handle).ok_or("group handle is not current")?.state())}
            "reduce" | "measure" => {
                let data=read(&args["values"])?;
                let flags=if let Some(flag)=args["valid"].as_bool() { vec![flag;data.values.len()] }
                    else if args["valid"].is_null(){vec![true;data.values.len()]}else{read(&args["valid"])?.flags()?};
                let sigma=if args["sigma"].is_null(){None}else{Some(read(&args["sigma"])?) };
                let plans=reduce::steps(&args["plans"],read)?;
                let statistic=args["statistic"].as_str().ok_or("statistic must be text")?;
                let parts=reduce::measure(&data,&flags,sigma.as_ref(),&plans,statistic)?;
                if operation.ends_with("measure") { return encode(parts,write); }
                let mut result=reduce::result(parts,statistic)?;
                if statistic=="mean" {
                    let measured=reduce::measure(&data,&flags,None,&plans,"sum")?;
                    let sums=measured["total"].reals()?;
                    let counts=measured["n"].reals()?;
                    let values:Vec<f64>=sums.iter().zip(counts).map(|(s,n)|if n>0.0{*s/n}else{f64::NAN}).collect();
                    let valid=result["valid"].flags()?.iter().zip(&values).map(|(v,x)|*v&&x.is_finite()).collect();
                    result.insert("valid".into(),Tensor::from_flags(valid,result["values"].shape.clone()));
                    result.insert("values".into(),Tensor::from_reals(values,result["values"].shape.clone()));
                }
                if !args["errors"].as_bool().unwrap_or(false) {result.remove("sigma");result.remove("dof");}
                encode(result,write)
            }
            "select" => {
                let mut steps=args["steps"].as_array().ok_or("selection steps must be a list")?.iter().map(|step| {
                    Ok((reduce::integer(step,"axis")?,reduce::integers(&step["indexes"],read)?,step["drop"].as_bool().unwrap_or(false)))
                }).collect::<Result<Vec<_>,String>>()?;
                steps.sort_by_key(|(axis,_,_)|std::cmp::Reverse(*axis));
                let mut result=Vec::new();
                for plane in args["planes"].as_array().ok_or("selection planes must be a list")? {
                    if plane.is_null(){result.push(Value::Null);continue;}
                    let mut tensor=read(plane)?;
                    for (axis,indexes,drop) in &steps {
                        if *axis>=tensor.shape.len() || indexes.is_empty() || indexes.iter().any(|i|*i>=tensor.shape[*axis]) || (*drop&&indexes.len()!=1) {
                            return Err("invalid selection dimension or indexes".into());
                        }
                        let inner=product(&tensor.shape[axis+1..])?;
                        let outer=product(&tensor.shape[..*axis])?;
                        let mut shape=tensor.shape.clone();
                        if *drop{shape.remove(*axis);}else{shape[*axis]=indexes.len();}
                        let mut chosen=Vec::new();chosen.try_reserve_exact(product(&shape)?).map_err(|e|format!("selection is too large: {e}"))?;
                        for row in 0..outer{for coordinate in indexes{for column in 0..inner{chosen.push((row*tensor.shape[*axis]+coordinate)*inner+column);}}}
                        tensor=tensor.gather(&chosen,shape)?;
                    }
                    if args["expand"].as_bool().unwrap_or(false){tensor.shape.push(1);}
                    result.push(write(tensor)?);
                }
                Ok(json!({"planes":result}))
            }
            "alignment" => alignment(args,write),
            "align" => {
                let shape=reduce::integers(&args["shape"],read)?;
                let count=product(&shape)?;
                let mut result=Vec::new();
                for plane in args["planes"].as_array().ok_or("alignment planes must be a list")? {
                    if plane.is_null(){result.push(Value::Null);continue;}
                    let tensor=read(plane)?;
                    if tensor.shape.len()<2||shape.len()<2{return Err("alignment needs three-domain storage".into());}
                    let steps=&args["steps"];
                    let rows=if steps.is_null()||steps[0].is_null(){None}else{Some(reduce::integers(&steps[0],read)?)};
                    let points=if steps.is_null()||steps[1].is_null(){None}else{Some(reduce::integers(&steps[1],read)?)};
                    let cells=if steps.is_null()||steps[2].is_null(){tensor.shape[2..].to_vec()}else{reduce::integers(&steps[2],read)?};
                    if shape.len()!=cells.len()+2 {return Err("alignment physical shapes differ".into());}
                    if rows.as_ref().is_some_and(|v|v.len()!=shape[0])||points.as_ref().is_some_and(|v|v.len()!=shape[1]){return Err("alignment row maps have wrong length".into());}
                    if cells.iter().zip(&shape[2..]).any(|(a,b)|*a!=1&&a!=b){return Err("cell dimensions cannot broadcast".into());}
                    let mut indexes=Vec::with_capacity(count);
                    let cell_count=product(&cells)?;
                    if cell_count!=product(&tensor.shape[2..])?{return Err("aligned cell reshape changes its size".into());}
                    if rows.as_ref().is_some_and(|v|v.iter().any(|n|*n>=tensor.shape[0]))||points.as_ref().is_some_and(|v|v.iter().any(|n|*n>=tensor.shape[1])){return Err("alignment row is outside input".into());}
                    let target_cells=product(&shape[2..])?;
                    for flat in 0..count{
                        let r=flat/(shape[1]*target_cells);let p=flat/target_cells%shape[1];
                        let mut rem=flat%target_cells;let mut c=0;let mut stride=1;
                        for (target,own) in shape[2..].iter().zip(&cells).rev(){let at=rem%target;rem/=target;c+=if *own==1{0}else{at*stride};stride*=own;}
                        let r=rows.as_ref().map_or(r,|v|v[r]);let p=points.as_ref().map_or(p,|v|v[p]);
                        indexes.push((r*tensor.shape[1]+p)*cell_count+c);
                    }
                    result.push(write(tensor.gather(&indexes,shape.clone())?)?);
                }
                Ok(json!({"planes":result}))
            }
            _=>Err(format!("unknown scientific operation {operation}")),
        }
    }
}

fn alignment(args:&Value,write:&dyn Fn(Tensor)->Result<Value,String>)->Result<Value,String>{
    use zlc_core::schema::DomainRole;
    let parse=|name:&str|zlc_archive::tree::schema_from_tree(&zlc_archive::json::Json::from_serde(&args[name])).map_err(|e|e.to_string());
    let target=parse("target")?;let other=parse("other")?;
    let roles=[DomainRole::Repeat,DomainRole::Point,DomainRole::Cell];
    if roles.iter().all(|role|target.domain(*role)==other.domain(*role)){return Ok(json!({"steps":null}));}
    let mut steps=Vec::new();
    for (d,role) in roles.into_iter().enumerate(){
        let wanted=target.domain(role);let given=other.domain(role);
        if wanted==given{steps.push(Value::Null);continue;}
        let scalar=d==2&&given.size()==1&&given.axes().len()==1&&given.axes()[0].role()=="scalar";
        let axes=if scalar{&[][..]}else{given.axes()};
        if axes.iter().any(|axis|!wanted.axes().contains(axis)){return Err("an axis of one side is not on the other".into());}
        if d==2{
            let positions=axes.iter().map(|axis|wanted.axes().iter().position(|other|other==axis).unwrap()).collect::<Vec<_>>();
            if positions.windows(2).any(|v|v[0]>v[1]){return Err("the cell axes are in another order".into());}
            steps.push(json!(wanted.axes().iter().map(|axis|if axes.contains(axis){axis.len()}else{1}).collect::<Vec<_>>()));continue;
        }
        let rows=if axes.is_empty(){if given.size()!=1{return Err("unnamed rows cannot be matched".into());}vec![0i64;wanted.size()]}else{
            let key=|domain:&zlc_core::schema::Domain,row|axes.iter().map(|axis|domain.mapping(axis.id()).unwrap().code(row).unwrap()).collect::<Vec<_>>();
            let mut positions=std::collections::HashMap::new();
            for row in 0..given.size(){positions.insert(key(given,row),row);}
            (0..wanted.size()).map(|row|positions.get(&key(wanted,row)).map(|n|*n as i64).ok_or("one side has no value for some coordinates of the other".to_owned())).collect::<Result<Vec<_>,_>>()?
        };
        let len=rows.len();steps.push(write(Tensor::from_counts(rows,vec![len]))?);
    }
    Ok(json!({"steps":steps}))
}
