//! Transport normalization around the core's parameter value rules. JSON
//! `$float` envelopes are decoded here, never interpreted by the core.
use serde_json::{Map,Value};
use zlc_core::parameters::{Field,typed};

fn normalize(field:&Field,value:&Value)->Value{
    if field.kind=="float"{if let Some(text)=value.get("$float").and_then(Value::as_str){return Value::String(text.to_owned());}}
    if field.kind=="numeric_tuple"{if let Some(items)=value.as_array(){return Value::Array(items.iter().map(|item|
        item.get("$float").and_then(Value::as_str).map_or_else(||item.clone(),|text|Value::String(text.to_owned()))).collect());}}
    if field.kind=="rows"{if let Some(rows)=value.as_array(){return Value::Array(rows.iter().map(|row|match row{
        Value::Object(values)=>Value::Object(values.iter().map(|(name,value)|(name.clone(),field.columns.iter().find(|f|f.name==*name).map_or_else(||value.clone(),|f|normalize(f,value)))).collect()),
        other=>other.clone(),
    }).collect());}}
    value.clone()
}
fn normalized(field:&Field)->Field{
    let mut rule=field.clone();rule.default=normalize(field,&field.default);rule.columns=field.columns.iter().map(normalized).collect();rule
}
fn restore_wire(field:&Field,projected:Value,original:&Value)->Value{
    if field.kind=="float"&&original.get("$float").and_then(Value::as_str)==projected.as_str()&&original.get("$float").is_some(){return original.clone();}
    if field.kind=="numeric_tuple"{if let(Value::Array(items),Some(before))=(&projected,original.as_array()){
        return Value::Array(items.iter().zip(before).map(|(item,original)|
            if original.get("$float").is_some()&&original.get("$float").and_then(Value::as_str)==item.as_str(){original.clone()}else{item.clone()}).collect());
    }}
    if field.kind=="rows"{if let(Value::Array(rows),Some(before))=(&projected,original.as_array()){
        return Value::Array(rows.iter().zip(before).map(|(row,original)|match(row,original){
            (Value::Object(row),Value::Object(original))=>Value::Object(row.iter().map(|(name,value)|{
                let restored=field.columns.iter().find(|f|f.name==*name).map_or_else(||value.clone(),|f|restore_wire(f,value.clone(),original.get(name).unwrap_or(&f.default)));
                (name.clone(),restored)
            }).collect()),_=>row.clone(),
        }).collect());
    }}
    projected
}

pub fn project(field:&Field,value:&Value,complete:bool)->Result<Value,typed::Error>{
    let projected=typed::project(&normalized(field),&normalize(field,value),complete)?;
    Ok(restore_wire(field,projected,value))
}
pub fn values(fields:&[Field],supplied:&Map<String,Value>,complete:bool)->Result<(Map<String,Value>,bool),typed::Error>{
    let rules:Vec<_>=fields.iter().map(normalized).collect();
    let normalized_values:Map<_,_>=supplied.iter().map(|(name,value)|(name.clone(),fields.iter().find(|f|f.name==*name).map_or_else(||value.clone(),|f|normalize(f,value)))).collect();
    let (mut output,complete)=typed::values(&rules,&normalized_values,complete)?;
    // Active non-finite numbers are rejected by typed::values. Only inactive
    // leaves can retain their input text, and those keep the original envelope.
    for field in fields{
        let original=supplied.get(&field.name).unwrap_or(&field.default);
        if let Some(value)=output.remove(&field.name){output.insert(field.name.clone(),restore_wire(field,value,original));}
    }
    Ok((output,complete))
}

#[cfg(test)]
mod tests{
    use super::*;use serde_json::json;
    #[test]
    fn wire_nonfinite_values_keep_type_errors_and_inactive_values(){
        let field:Field=serde_json::from_value(json!({"name":"level","type":"float","label":"Level"})).unwrap();
        let error=project(&field,&json!({"$float":"nan"}),true).unwrap_err();assert_eq!(error.kind,"ValueError");assert_eq!(error.message,"Level must be finite");
        let error=project(&field,&json!({"$float":"nonsense"}),true).unwrap_err();assert_eq!(error.kind,"TypeError");
        let numbers:Field=serde_json::from_value(json!({"name":"numbers","type":"numeric_tuple"})).unwrap();
        assert_eq!(project(&numbers,&json!([{"$float":"nan"}]),true).unwrap_err().kind,"ValueError");
        let fields:Vec<Field>=serde_json::from_value(json!([
            {"name":"enabled","type":"bool","default":false},
            {"name":"level","type":"float","enabled_when":["enabled",[true]]}
        ])).unwrap();
        let supplied=json!({"level":{"$float":"nan"}});
        assert_eq!(values(&fields,supplied.as_object().unwrap(),true).unwrap().0["level"],supplied["level"]);
        let enabled=json!({"enabled":"true","level":{"$float":"nan"}});
        assert_eq!(values(&fields,enabled.as_object().unwrap(),true).unwrap_err().kind,"ValueError");
    }
}
