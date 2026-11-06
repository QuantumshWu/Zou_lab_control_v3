//! Scientific service protocol: codecs, dispatch and run-scoped handles live
//! here. Numerical plans and accumulator state use the typed core directly.
use std::sync::Arc;
use serde_json::{Value,json};
use zlc_core::{AnyBuffer,DType};
use zlc_core::schema::DatasetSchema;
use zlc_core::scientific::{Tensor,reduce,binary,geometry,transform};
use zlc_core::scientific::groups::{Group,GroupConfig,GroupFeed,GroupChunk};

type Result<T>=std::result::Result<T,String>;
type Reader<'a>=dyn Fn(&Value)->Result<Tensor>+'a;
type Writer<'a>=dyn Fn(Tensor)->Result<Value>+'a;

#[derive(Default)]
pub struct Service { groups:std::collections::HashMap<String,Group> }

fn integer(value:&Value,name:&str)->Result<usize>{
    value[name].as_u64().and_then(|n|usize::try_from(n).ok()).ok_or_else(||format!("{name} must be a nonnegative integer"))
}
fn integers(value:&Value,read:&Reader<'_>)->Result<Vec<usize>>{
    if let Some(values)=value.as_array(){return values.iter().map(|v|v.as_u64().and_then(|n|usize::try_from(n).ok()).ok_or("indices must be nonnegative integers".into())).collect();}
    let tensor=read(value)?;tensor.validate()?;
    match tensor.values{
        AnyBuffer::I64(v)=>v.as_slice().iter().map(|n|usize::try_from(*n).map_err(|_|"indices must be nonnegative integers".into())).collect(),
        AnyBuffer::U64(v)=>v.as_slice().iter().map(|n|usize::try_from(*n).map_err(|_|"index overflows usize".into())).collect(),
        _=>Err("indices require int64 or uint64".into()),
    }
}
fn index_array(values:Vec<usize>,write:&Writer<'_>)->Result<Value>{let n=values.len();write(Tensor::from_counts(values.into_iter().map(|n|n as i64).collect(),vec![n]))}
fn schema(value:&Value)->Result<Arc<DatasetSchema>>{zlc_archive::tree::schema_from_tree(&zlc_archive::json::Json::from_serde(value)).map_err(|e|e.to_string())}
fn schema_tree(value:&DatasetSchema)->Value{zlc_archive::tree::schema_tree(value).to_serde()}
fn planes(value:&Value,read:&Reader<'_>)->Result<Vec<Option<Tensor>>>{value.as_array().ok_or("planes must be a list")?.iter().map(|v|if v.is_null(){Ok(None)}else{read(v).map(Some)}).collect()}
fn encode_planes(planes:reduce::Planes,write:&Writer<'_>)->Result<Value>{Ok(Value::Object(planes.into_iter().map(|(name,t)|Ok((name,write(t)?))).collect::<Result<_>>()?))}
fn encode_optional(planes:Vec<Option<Tensor>>,write:&Writer<'_>)->Result<Value>{Ok(json!({"planes":planes.into_iter().map(|v|v.map_or(Ok(Value::Null),write)).collect::<Result<Vec<_>>>()?}))}
fn optional(value:&Value,read:&Reader<'_>)->Result<Option<Tensor>>{if value.is_null(){Ok(None)}else{read(value).map(Some)}}
fn flags(value:&Value,count:usize,read:&Reader<'_>)->Result<Vec<bool>>{if value.is_null(){Ok(vec![true;count])}else if let Some(v)=value.as_bool(){Ok(vec![v;count])}else{read(value)?.flags()}}
fn dtype(name:&str)->Result<DType>{
    let dtype=super::arrays::dtype(name).map_err(|_|format!("unsupported real dtype {name}"))?;
    if matches!(dtype,DType::C64|DType::C128){Err(format!("unsupported real dtype {name}"))}else{Ok(dtype)}
}

fn reduction_steps(value:&Value,read:&Reader<'_>)->Result<Vec<reduce::Step>>{
    value.as_array().ok_or("reduction plans must be a list")?.iter().map(|step|Ok(match step["kind"].as_str().unwrap_or(""){
        "mapped"=>reduce::Step::Mapped{axis:integer(step,"axis")?,codes:integers(&step["codes"],read)?,size:integer(step,"size")?},
        "dense"=>reduce::Step::Dense{axes:integers(&step["axes"],read)?,expand:step["expand"].as_bool().unwrap_or(false),shape:Vec::new(),kept_positions:Vec::new()},
        _=>return Err("unknown reduction plan kind".into()),
    })).collect()
}
fn encode_steps(steps:Vec<reduce::Step>,write:&Writer<'_>)->Result<Vec<Value>>{
    steps.into_iter().map(|step|Ok(match step{
        reduce::Step::Mapped{axis,codes,size}=>json!({"kind":"mapped","axis":axis,"codes":index_array(codes,write)?,"size":size}),
        reduce::Step::Dense{axes,expand,shape,kept_positions}=>json!({"kind":"dense","axes":axes,"expand":expand,"shape":shape,"kept_positions":kept_positions}),
    })).collect()
}
fn encode_chunks(chunks:Vec<GroupChunk>,write:&Writer<'_>)->Result<Value>{
    let chunks=chunks.into_iter().map(|chunk|{let mut planes=encode_planes(chunk.planes,write)?;planes["rows"]=json!(chunk.rows);planes["columns"]=json!(chunk.columns);Ok(planes)}).collect::<Result<Vec<_>>>()?;
    Ok(json!({"chunks":chunks}))
}

impl Service{
    pub fn dispatch(&mut self,operation:&str,args:&Value,read:&Reader<'_>,write:&Writer<'_>)->Result<Value>{
        let operation=operation.strip_prefix("scientific.").unwrap_or(operation);
        match operation{
            "axis"|"pick_target"|"pick_plan"|"reduce_plan"=>geometry_request(operation,args,read,write),
            "binary"=>{
                let input=|value:&Value|->Result<binary::Input>{if let Some(v)=value.as_bool(){Ok(binary::Input::Boolean(v))}else if let Some(v)=value.as_f64(){Ok(binary::Input::Float(v))}else{read(value).map(binary::Input::Tensor)}};
                let output=binary::apply(binary::Request{left:input(&args["left"] )?,right:input(&args["right"] )?,shape:integers(&args["shape"],read)?,
                    valid:optional(&args["valid"],read)?,operation:args["op"].as_str().ok_or("operator is required")?.into(),
                    dtype:dtype(args["dtype"].as_str().ok_or("result dtype is required")?)?,compute_dtype:dtype(args["compute_dtype"].as_str().ok_or("compute dtype is required")?)?})?;
                Ok(json!({"values":write(output.values)?,"valid":write(output.valid)?}))
            }
            "where"=>Ok(json!({"valid":write(transform::masked(&read(&args["valid"])?,&read(&args["mask"])?,&read(&args["judged"])?,args["negate"].as_bool().unwrap_or(false))?)?})),
            "reduce"|"measure"=>{
                let data=read(&args["values"])?;let flags=flags(&args["valid"],data.values.len(),read)?;let sigma=optional(&args["sigma"],read)?;
                let steps=reduction_steps(&args["plans"],read)?;let statistic=args["statistic"].as_str().ok_or("statistic must be text")?;
                let result=if operation=="measure"{reduce::measure(&data,&flags,sigma.as_ref(),&steps,statistic)?}
                    else{reduce::statistic(&data,&flags,sigma.as_ref(),&steps,statistic,args["errors"].as_bool().unwrap_or(false))?};
                encode_planes(result,write)
            }
            "select"=>{
                let steps=args["steps"].as_array().ok_or("selection steps must be a list")?.iter().map(|step|Ok(transform::SelectionStep{
                    axis:integer(step,"axis")?,indexes:integers(&step["indexes"],read)?,drop:step["drop"].as_bool().unwrap_or(false)})).collect::<Result<Vec<_>>>()?;
                encode_optional(transform::selected(planes(&args["planes"],read)?,&steps,args["expand"].as_bool().unwrap_or(false))?,write)
            }
            "alignment"=>{
                let target=schema(&args["target"])?;let other=schema(&args["other"])?;
                let result=transform::alignment(&target,&other)?;
                let steps=match result{None=>Value::Null,Some(p)=>json!([
                    p.rows.map_or(Ok(Value::Null),|v|index_array(v,write))?,p.points.map_or(Ok(Value::Null),|v|index_array(v,write))?,p.cells])};
                Ok(json!({"steps":steps}))
            }
            "align"=>{
                let steps=&args["steps"];let indexes=|v:&Value|if v.is_null(){Ok(None)}else{integers(v,read).map(Some)};
                let plan=if steps.is_null(){None}else{Some(transform::Alignment{rows:indexes(&steps[0])?,points:indexes(&steps[1])?,cells:indexes(&steps[2])?})};
                encode_optional(transform::aligned(planes(&args["planes"],read)?,plan.as_ref(),&integers(&args["shape"],read)?)?,write)
            }
            "group.create"=>{
                let optional_indexes=|v:&Value|if v.is_null(){Ok(None)}else{integers(v,read).map(Some)};
                let group=Group::new(GroupConfig{operation:args["statistic"].as_str().ok_or("statistic is required")?.into(),domain:integer(args,"domain")?,
                    spanning:args["spanning"].as_bool().unwrap_or(false),every:integer(args,"every")?,counts_r:integers(&args["counts_r"],read)?,counts_p:integers(&args["counts_p"],read)?,
                    source_points:integer(args,"source_points")?,output_points:integer(args,"output_points")?,lines:optional_indexes(&args["lines"] )?,steps:optional_indexes(&args["steps"] )?})?;
                let handle=uuid::Uuid::new_v4().to_string();self.groups.insert(handle.clone(),group);Ok(json!({"handle":handle}))
            }
            "group.feed"=>{
                let handle=args["handle"].as_str().ok_or("group handle must be text")?;
                let group=self.groups.get_mut(handle).ok_or("group handle is not current")?;
                let data=read(&args["values"])?;
                let flags=if let Some(v)=args["valid"].as_bool(){vec![v;data.values.len()]}else{read(&args["valid"])?.flags()?};
                let ordinal=if group.spans_events(){integer(args,"ordinal")?}else{0};
                let feed=GroupFeed{flags,data,sigma:optional(&args["sigma"],read)?,
                    rows:integers(&args["rows"],read)?,points:integers(&args["points"],read)?,plans:reduction_steps(&args["plans"],read)?,ordinal};
                encode_chunks(group.feed(feed)?,write)
            }
            "group.finish"=>{let handle=args["handle"].as_str().ok_or("group handle must be text")?;encode_chunks(self.groups.remove(handle).ok_or("group handle is not current")?.finish()?,write)},
            "group.drop"=>{if let Some(handle)=args["handle"].as_str(){self.groups.remove(handle);}Ok(json!({}))},
            "group.state"=>{let handle=args["handle"].as_str().ok_or("group handle must be text")?;let state=self.groups.get(handle).ok_or("group handle is not current")?.state();Ok(json!({"open_groups":state.open_groups,"held_bytes":state.held_bytes}))},
            _=>Err(format!("unknown scientific operation {operation}")),
        }
    }
}

fn geometry_request(operation:&str,args:&Value,read:&Reader<'_>,write:&Writer<'_>)->Result<Value>{
    let source=schema(&args["schema"])?;
    match operation{
        "axis"=>{let(d,a)=geometry::find_axis(&source,args["name"].as_str().unwrap_or(""),args["what"].as_str().unwrap_or("the value"),args["verb"].as_str().unwrap_or("reduce"))?;Ok(json!({"domain":d,"axis":a.id()}))},
        "pick_target"=>{let(d,axes)=geometry::pick_target(&source,args["name"].as_str().unwrap_or(""),args["what"].as_str().unwrap_or("the value"))?;Ok(json!({"domain":d,"axes":axes.iter().map(|a|a.id()).collect::<Vec<_>>()}))},
        "pick_plan"=>{
            let plan=geometry::pick_plan(&source,geometry::PickRequest{domain:integer(args,"domain")?,axis_ids:serde_json::from_value(args["axes"].clone()).map_err(|e|e.to_string())?,
                indexes:integers(&args["indexes"],read)?,scalar:args["scalar"].as_bool().unwrap_or(false),domain_pick:args["domain_pick"].as_bool().unwrap_or(false)})?;
            let steps=plan.steps.into_iter().map(|s|Ok(json!({"axis":s.axis,"indexes":index_array(s.indexes,write)?,"drop":s.drop}))).collect::<Result<Vec<_>>>()?;
            Ok(json!({"schema":schema_tree(&plan.schema),"take":{"steps":steps,"expand":plan.expand}}))
        }
        "reduce_plan"=>{
            let plan=geometry::reduction_plan(&source,geometry::ReductionRequest{operation:args["statistic"].as_str().ok_or("statistic must be text")?.into(),
                names:serde_json::from_value(args["names"].clone()).map_err(|e|e.to_string())?,by:args["by"].as_str().unwrap_or("").into(),every:integer(args,"every")?})?;
            Ok(json!({"schema":schema_tree(&plan.schema),"plans":encode_steps(plan.steps,write)?}))
        }
        _=>Err("unknown scientific geometry operation".into()),
    }
}

#[cfg(test)]
#[path="scientific_tests.rs"]
mod tests;
