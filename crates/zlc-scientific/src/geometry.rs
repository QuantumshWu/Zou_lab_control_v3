//! Canonical named-axis selection and reduction compilation. Frontends submit
//! authored names/indices; the native schema determines storage maps and types.
use std::collections::{BTreeSet,HashMap,HashSet};
use std::sync::Arc;
use serde_json::{Value,json};
use zlc_core::{DType,OwnedBuffer};
use zlc_core::schema::{Axis,CodeBase,CodeMapping,Coordinate,Coordinates,DatasetSchema,Domain,DomainRole,ValueSchema,ValidityContract};
use zlc_core::restrict::{Indices,Vector,Held,subset_axis,subset_mapped_domain};
use crate::{Tensor,reduce,numeric};

const ROLES:[DomainRole;3]=[DomainRole::Repeat,DomainRole::Point,DomainRole::Cell];
const WORDS:[&str;3]=["Repeat","Point","Cell"];
fn primary(axis:&Axis)->bool{axis.coordinate_of().is_none()&&axis.role()!="scalar"}
fn family(axis:&Axis,other:&Axis)->bool{other.id()==axis.id()||other.coordinate_of()==Some(axis.id())}
fn unit(schema:&DatasetSchema)->&str{schema.value().unit().unwrap_or("1")}
fn scalar()->Result<Arc<Domain>,String>{
    let axis=Axis::new("zlc_data.scalar".into(),"value".into(),"scalar".into(),Coordinates::Numeric(OwnedBuffer::from_vec(vec![0i64]).into()),None,None,None,None)?;
    Domain::dense(vec![1],vec![axis])
}
fn schema(domains:[Arc<Domain>;3],dtype:DType,unit:&str,name:Option<&str>)->Result<Arc<DatasetSchema>,String>{
    zlc_core::units::DEFAULT_UNITS.resolve(unit).map_err(|e|e.to_string())?;
    let cell=&domains[2];
    let validity=if cell.size()==1&&cell.axes().len()==1&&cell.axes()[0].role()=="scalar"{ValidityContract::Value}
        else{ValidityContract::Components(cell.axes().iter().map(|axis|axis.id().into()).collect())};
    let value=ValueSchema::new(dtype,Some(unit.into()),name.map(Arc::from),validity)?;
    DatasetSchema::new(domains[0].clone(),domains[1].clone(),domains[2].clone(),value)
}
fn axis(schema:&DatasetSchema,name:&str,what:&str,verb:&str)->Result<(usize,Arc<Axis>),String>{
    let found:Vec<_>=ROLES.iter().enumerate().flat_map(|(d,role)|schema.domain(*role).axes().iter().filter(|a|primary(a)&&a.name()==name).map(move|a|(d,a.clone()))).collect();
    if found.len()==1{return Ok(found[0].clone());}
    if found.len()>1{return Err(format!("two axes of {what} are named {name}; it cannot be {} by name",if verb=="pick"{"picked"}else{"reduced"}));}
    for role in ROLES{let domain=schema.domain(role);for a in domain.axes(){if a.name()==name{if let Some(parent)=a.coordinate_of(){
        let parent=domain.axis(parent).unwrap().name();return Err(format!("{name} is another coordinate of {parent}; {verb} {parent}"));
    }}}}
    let names=ROLES.iter().flat_map(|role|schema.domain(*role).axes()).filter(|a|primary(a)).map(|a|a.name()).collect::<Vec<_>>().join(", ");
    Err(format!("{what} has no axis '{name}'; its axes: {}",if names.is_empty(){"none"}else{&names}))
}
fn target(schema:&DatasetSchema,name:&str,what:&str)->Result<(usize,Vec<Arc<Axis>>),String>{
    let Some(d)=WORDS.iter().position(|w|*w==name)else{let(d,a)=axis(schema,name,what,"pick")?;return Ok((d,vec![a]));};
    if ROLES.iter().any(|role|schema.domain(*role).axes().iter().any(|a|a.name()==name)){return Err(format!("{name} names both a domain and an axis of {what}; it cannot be picked by name"));}
    let axes:Vec<_>=schema.domain(ROLES[d]).axes().iter().filter(|a|primary(a)).cloned().collect();
    if axes.len()==1{return Err(format!("{name} has one axis, {}; pick {}",axes[0].name(),axes[0].name()));}
    if axes.is_empty(){return Err(format!("{what} has no {name} axes"));}
    Ok((d,axes))
}
fn natural(codes:Vec<u64>)->Result<Arc<CodeMapping>,String>{
    let base=if codes.iter().enumerate().all(|(i,c)|i as u64==*c){CodeBase::Range{start:0,step:1,len:codes.len()}}else{CodeBase::Explicit(OwnedBuffer::from_vec(codes))};
    CodeMapping::new(base,1,1)
}
fn indexes(indices:Vec<usize>)->Indices{
    if !indices.is_empty()&&indices.iter().enumerate().all(|(i,v)|*v==indices[0]+i){Indices::Range(indices[0],indices[0]+indices.len())}else{Indices::List(indices)}
}
fn pick(source:&Arc<DatasetSchema>,args:&Value,read:&dyn Fn(&Value)->Result<Tensor,String>,write:&dyn Fn(Tensor)->Result<Value,String>)->Result<Value,String>{
    let d=reduce::integer(args,"domain")?;if d>2{return Err("unknown scientific domain".into());}
    let domain=source.domain(ROLES[d]);let ids:Vec<String>=serde_json::from_value(args["axes"].clone()).map_err(|e|e.to_string())?;
    let selected=ids.iter().map(|id|domain.axis(id).filter(|a|primary(a)).cloned().ok_or("pick must name primary axes in its domain".to_owned())).collect::<Result<Vec<_>,_>>()?;
    let domain_pick=args["domain_pick"].as_bool().unwrap_or(false);let drop=args["scalar"].as_bool().unwrap_or(false);
    if selected.is_empty()||(!domain_pick&&selected.len()!=1){return Err("pick must name its primary axis or whole domain".into());}
    let indices=reduce::integers(&args["indexes"],read)?;
    let limit=if domain_pick{domain.size()}else{selected[0].len()};
    if indices.is_empty()||indices.windows(2).any(|p|p[0]>=p[1])||indices.iter().any(|i|*i>=limit)||(drop&&indices.len()!=1){return Err("pick indices must be ascending, unique and inside the axis".into());}
    let remove:HashSet<&str>=selected.iter().flat_map(|a|domain.axes().iter().filter(move|b|family(a,b)).map(|a|a.id())).collect();
    let mut domains=std::array::from_fn(|i|source.domain(ROLES[i]).clone());let mut take=Vec::new();let mut expand=false;
    if d<2{
        let (rows,rank)=if domain_pick{(indices.clone(),None)}else{
            let a=&selected[0];let mut ranks=vec![None;a.len()];for(i,old)in indices.iter().enumerate(){ranks[*old]=Some(i);}
            let mapping=domain.mapping(a.id()).unwrap();let mut rows=(0..domain.size()).filter(|row|ranks[mapping.code(*row).unwrap()].is_some()).collect::<Vec<_>>();
            rows.sort_by_key(|row|ranks[mapping.code(*row).unwrap()].unwrap());(rows,Some(ranks))
        };
        if rows.is_empty(){return Err("EmptySelection: no stored rows".into());}
        let cut=subset_mapped_domain(domain,&indexes(rows.clone())).map_err(|e|e.to_string())?;
        let mut axes=Vec::new();let mut maps=Vec::new();
        for a in cut.axes(){
            if drop&&remove.contains(a.id()){continue;}
            if !drop&&!domain_pick&&remove.contains(a.id()){
                let original=domain.axis(a.id()).unwrap();axes.push(subset_axis(original,&indexes(indices.clone())).map_err(|e|e.to_string())?);
                let map=domain.mapping(selected[0].id()).unwrap();let rank=rank.as_ref().unwrap();
                maps.push(natural(rows.iter().map(|row|rank[map.code(*row).unwrap()].unwrap() as u64).collect())?);
            }else{axes.push(a.clone());maps.push(cut.mapping(a.id()).unwrap().clone());}
        }
        domains[d]=Domain::new(vec![rows.len()],axes,maps)?;
        take.push(json!({"axis":d,"indexes":write(Tensor::from_counts(rows.iter().map(|v|*v as i64).collect(),vec![rows.len()]))?,"drop":false}));
    }else{
        let mut subsets:HashMap<&str,Vec<usize>>=HashMap::new();
        if domain_pick{
            let mut matrix=Vec::new();for mut row in indices.iter().copied(){let mut line=vec![0;selected.len()];for(i,a)in selected.iter().enumerate().rev(){line[i]=row%a.len();row/=a.len();}matrix.push(line);}
            for(i,a)in selected.iter().enumerate(){subsets.insert(a.id(),matrix.iter().map(|r|r[i]).collect::<BTreeSet<_>>().into_iter().collect());}
            if !drop&&subsets.values().map(Vec::len).product::<usize>()!=indices.len(){return Err(format!("these Cell values do not fill a grid of {}; pick one, or pick each axis in a variable of its own",selected.iter().map(|a|a.name()).collect::<Vec<_>>().join(" × ")));}
        }else{subsets.insert(selected[0].id(),indices.clone());}
        let mut axes=Vec::new();
        for(i,a)in domain.axes().iter().enumerate(){
            if let Some(subset)=subsets.get(a.id()){
                take.push(json!({"axis":i+2,"indexes":write(Tensor::from_counts(subset.iter().map(|v|*v as i64).collect(),vec![subset.len()]))?,"drop":drop}));
                if !drop{axes.push(subset_axis(a,&indexes(subset.clone())).map_err(|e|e.to_string())?);}
            }else{axes.push(a.clone());}
        }
        expand=axes.is_empty();domains[d]=if expand{scalar()?}else{Domain::dense(axes.iter().map(|a|a.len()).collect(),axes)?};
    }
    let output=schema(domains,source.dtype(),unit(source),source.value().name())?;
    Ok(json!({"schema":zlc_archive::tree::schema_tree(&output).to_serde(),"take":{"steps":take,"expand":expand}}))
}

fn grouped(domain:&Domain,axis:&Axis,every:usize)->Result<HashMap<String,Arc<Axis>>,String>{
    let id=format!("{}/every-{every}",axis.id());let mut out=HashMap::new();
    for member in domain.axes().iter().filter(|a|family(axis,a)){
        let held=Held::of(member);let numeric=match &held{Held::Values(_)=>false,Held::Implicit{origin}=>*origin+member.len() as i128-1<=i64::MAX as i128,_=>true};
        let values=(0..member.len()).map(|i|if !numeric{i as f64}else{match member.coordinate(i).unwrap(){Coordinate::Integer(v)=>v as f64,Coordinate::Real(v)=>v,_=>i as f64}}).collect::<Vec<_>>();
        let means=values.chunks(every).map(|part|numeric::sum_f64(part)/part.len() as f64).collect::<Vec<_>>();
        let new=Axis::new(format!("{}/every-{every}",member.id()).into(),member.name().into(),member.role().into(),Vector::of_floats(means).into_coordinates(),
            member.unit().map(Arc::from),member.coordinate_frame().map(Arc::from),None,if member.id()==axis.id(){None}else{Some(id.clone().into())})
            .map_err(|_|format!("two groups of {every} {} read the same mean coordinate; choose another size",member.name()))?;
        out.insert(member.id().to_owned(),new);
    }Ok(out)
}
fn reduction(source:&Arc<DatasetSchema>,args:&Value,write:&dyn Fn(Tensor)->Result<Value,String>)->Result<Value,String>{
    let operation=args["statistic"].as_str().ok_or("statistic must be text")?;
    let names:Vec<String>=serde_json::from_value(args["names"].clone()).map_err(|e|e.to_string())?;
    let every=reduce::integer(args,"every")?;let by=args["by"].as_str().unwrap_or("");
    let boolean=source.dtype()==DType::Bool;
    if !(if boolean{&["sum","mean","count","any","all"][..]}else{&["sum","mean","count","std","min","max"][..]}).contains(&operation){return Err(if boolean{format!("{operation} needs numbers; for booleans use sum, mean, count, any or all")}else{format!("{operation} needs booleans")});}
    let mut found=Vec::new();let mut ids=HashSet::new();
    for name in names{let(d,a)=axis(source,&name,"the value","reduce")?;if !ids.insert(a.id().to_owned()){return Err(format!("{} is chosen twice",a.name()));}found.push((d,a));}
    let groups=if every>0{let(d,a)=found.iter().find(|(_,a)|a.name()==by).ok_or_else(||format!("{by} is not among the axes it reduces"))?;grouped(source.domain(ROLES[*d]),a,every)?}else{HashMap::new()};
    let removed:HashSet<String>=found.iter().flat_map(|(d,a)|source.domain(ROLES[*d]).axes().iter().filter(move|b|family(a,b)).map(|b|b.id().to_owned())).filter(|id|!groups.contains_key(id)).collect();
    let mut domains=std::array::from_fn(|i|source.domain(ROLES[i]).clone());let mut plans=Vec::new();
    for(d,role)in ROLES.into_iter().enumerate(){
        let domain=source.domain(role);
        if !domain.axes().iter().any(|a|removed.contains(a.id())||groups.contains_key(a.id())){continue;}
        let staying=domain.axes().iter().filter(|a|!removed.contains(a.id())).collect::<Vec<_>>();
        let kept=staying.iter().map(|a|groups.get(a.id()).cloned().unwrap_or_else(||(*a).clone())).collect::<Vec<_>>();
        if d<2{
            let mut codes=Vec::new();let mut inverse=Vec::with_capacity(domain.size());let mut seen=HashMap::new();
            for row in 0..domain.size(){
                let key=staying.iter().map(|a|domain.mapping(a.id()).unwrap().code(row).unwrap()/if groups.contains_key(a.id()){every}else{1}).collect::<Vec<_>>();
                let next=seen.len();let index=*seen.entry(key.clone()).or_insert_with(||{codes.push(key);next});inverse.push(index as i64);
            }
            let maps=(0..kept.len()).map(|column|natural(codes.iter().map(|row|row[column] as u64).collect())).collect::<Result<Vec<_>,_>>()?;
            domains[d]=Domain::new(vec![codes.len()],kept,maps)?;
            plans.push(json!({"kind":"mapped","axis":d,"codes":write(Tensor::from_counts(inverse,vec![domain.size()]))?,"size":domains[d].size()}));
        }else{
            let positions=domain.axes().iter().enumerate().filter_map(|(i,a)|removed.contains(a.id()).then_some(i+2)).collect::<Vec<_>>();
            let kept_positions=domain.axes().iter().enumerate().filter_map(|(i,a)|(!removed.contains(a.id())).then_some(i)).collect::<Vec<_>>();
            let mut shape=domain.physical_shape().to_vec();
            for(i,a)in domain.axes().iter().enumerate(){if let Some(group)=groups.get(a.id()){
                let inverse=(0..a.len()).map(|v|(v/every) as i64).collect();
                plans.push(json!({"kind":"mapped","axis":i+2,"codes":write(Tensor::from_counts(inverse,vec![a.len()]))?,"size":group.len()}));shape[i]=group.len();
            }}
            if !positions.is_empty(){plans.push(json!({"kind":"dense","axes":positions,"shape":shape,"kept_positions":kept_positions,"expand":kept_positions.is_empty()}));}
            domains[d]=if kept.is_empty(){scalar()?}else{Domain::dense(kept.iter().map(|a|a.len()).collect(),kept)?};
        }
    }
    let dtype=if operation=="count"||(boolean&&operation=="sum"){DType::I64}else if matches!(operation,"any"|"all"){DType::Bool}else{DType::F64};
    let unit=if boolean||operation=="count"{"1"}else{unit(source)};
    let output=schema(domains,dtype,unit,None)?;
    Ok(json!({"schema":zlc_archive::tree::schema_tree(&output).to_serde(),"plans":plans}))
}

pub(crate) fn dispatch(operation:&str,args:&Value,read:&dyn Fn(&Value)->Result<Tensor,String>,write:&dyn Fn(Tensor)->Result<Value,String>)->Result<Value,String>{
    let source=zlc_archive::tree::schema_from_tree(&zlc_archive::json::Json::from_serde(&args["schema"])).map_err(|e|e.to_string())?;
    match operation{
        "axis"=>{let(d,a)=axis(&source,args["name"].as_str().unwrap_or(""),args["what"].as_str().unwrap_or("the value"),args["verb"].as_str().unwrap_or("reduce"))?;Ok(json!({"domain":d,"axis":a.id()}))},
        "pick_target"=>{let(d,axes)=target(&source,args["name"].as_str().unwrap_or(""),args["what"].as_str().unwrap_or("the value"))?;Ok(json!({"domain":d,"axes":axes.iter().map(|a|a.id()).collect::<Vec<_>>()}))},
        "pick_plan"=>pick(&source,args,read,write),"reduce_plan"=>reduction(&source,args,write),_=>Err("unknown scientific geometry operation".into()),
    }
}
