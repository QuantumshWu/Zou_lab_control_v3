//! One typed projection rule set shared by native views and Python plugins.
use serde_json::{Map, Value, json};
use super::{Field, effective, enabled, py_float};

#[derive(Clone, Debug)]
pub struct Error { pub kind: &'static str, pub message: String }
fn value(message: impl Into<String>) -> Error { Error { kind:"ValueError", message:message.into() } }
fn wrong(message: impl Into<String>) -> Error { Error { kind:"TypeError", message:message.into() } }
impl std::fmt::Display for Error { fn fmt(&self, out:&mut std::fmt::Formatter<'_>)->std::fmt::Result { out.write_str(&self.message) } }
impl std::error::Error for Error {}
type Result<T> = std::result::Result<T,Error>;

fn integer(input:&Value,label:&str)->Result<Value> {
    match input {
        Value::Number(n) if n.is_i64() || n.is_u64() => Ok(input.clone()),
        Value::String(text) => {
            let text=text.trim(); let digits=text.strip_prefix(['+','-']).unwrap_or(text);
            if digits.is_empty() || !digits.bytes().all(|b|b.is_ascii_digit()) { return Err(wrong(format!("{label} must be an integer or decimal integer text"))); }
            if let Ok(n)=text.parse::<i64>() { return Ok(json!(n)); }
            text.parse::<u64>().map(Value::from).map_err(|_|value(format!("{label} is out of range")))
        }
        _=>Err(wrong(format!("{label} must be an integer or decimal integer text"))),
    }
}
fn real(input:&Value,label:&str)->Result<f64> {
    let number=match input { Value::Number(n)=>n.as_f64(), Value::String(text)=>text.trim().parse::<f64>().ok(),
        _=>None };
    let number=number.ok_or_else(||wrong(format!("{label} must be a finite number")))?;
    if !number.is_finite() { return Err(value(format!("{label} must be finite"))); } Ok(number)
}

pub fn project(field:&Field,input:&Value,complete:bool)->Result<Value> {
    let label=field.label();
    let answer=match (field.kind.as_str(),input) {
        (_,Value::Null)=>Value::Null,
        ("rows",Value::Array(rows))=>Value::Array(rows.iter().enumerate().map(|(i,row)| {
            let row=row.as_object().ok_or_else(||wrong(format!("{label} row {} must be a mapping of its columns",i+1)))?;
            values(&field.columns,row,complete).map(|(row,_)|Value::Object(row))
                .map_err(|mut e| { e.message=format!("{label} row {}: {}",i+1,e.message); e })
        }).collect::<Result<_>>()?),
        ("rows",_)=>return Err(wrong(format!("{label} must be a list of rows"))),
        ("text_tuple",Value::Array(items))=>Value::Array(items.iter().map(|item| {
            let text=item.as_str().ok_or_else(||wrong(format!("{label} must be a list of texts")))?.trim();
            if text.is_empty() { return Err(value(format!("{label} items must be non-empty text"))); } Ok(json!(text))
        }).collect::<Result<_>>()?),
        ("text_tuple",_)=>return Err(wrong(format!("{label} must be a list of texts"))),
        ("numeric_tuple",input)=> {
            let parts=match input { Value::Array(items)=>items.clone(), Value::String(text)=>text.replace(',', " ").split_whitespace().map(|p|json!(p)).collect(),
                _=>return Err(wrong(format!("{label} must be a numeric list"))) };
            if parts.is_empty() { return Err(value(format!("{label} needs at least one number"))); }
            Value::Array(parts.iter().map(|part| real(part,&format!("{label} items")).map(Value::from)).collect::<Result<_>>()?)
        }
        ("pair",input)=> {
            let parts=match input {Value::Array(items)=>items.clone(),Value::String(text)=>text.replace(',', " ").split_whitespace().map(|p|json!(p)).collect(),
                _=>return Err(value(format!("{label} needs two numbers, as y, x")))};
            if parts.len()!=2 { return Err(value(format!("{label} needs two numbers, as y, x"))); }
            Value::Array(parts.iter().map(|part|integer(part,&format!("{label} item"))).collect::<Result<_>>()?)
        }
        ("int",input)=>integer(input,label)?,
        ("float",input)=>json!(real(input,label)?),
        ("bool",Value::Bool(flag))=>json!(flag),
        ("bool",Value::String(text))=>match text.trim().to_ascii_lowercase().as_str() {
            "true"|"1"|"yes"|"on"=>json!(true), "false"|"0"|"no"|"off"=>json!(false), _=>return Err(wrong(format!("{label} must be true or false"))) },
        ("bool",_)=>return Err(wrong(format!("{label} must be true or false"))),
        ("choice",input)=>input.clone(),
        ("str"|"text"|"resource"|"folder",Value::String(text))=>json!(text),
        ("str"|"text"|"resource"|"folder",_)=>return Err(wrong(format!("{label} must be text"))),
        (kind,_)=>return Err(value(format!("field {:?} is declared as {kind:?}, which has no value rule",field.name))),
    };
    let answer=match (&answer,field.kind.as_str()) { (Value::Null,"str"|"text"|"folder")=>json!(""), (Value::Null,"rows"|"text_tuple")=>json!([]), _=>answer };
    if !field.choices.is_empty() && !answer.is_null() && !field.choices.iter().any(|choice|choice.value==answer) {
        return Err(value(format!("{label} must be one of {}",field.choices.iter().map(|choice|choice.value.to_string()).collect::<Vec<_>>().join(", "))));
    }
    if let Some(number)=answer.as_f64().filter(|_|answer.is_number()) {
        let unit=field.unit.as_ref().filter(|s|!s.is_empty()).map(|s|format!(" {s}")).unwrap_or_default();
        if let Some(min)=field.minimum.filter(|min|number<*min) {return Err(value(format!("{label} must be at least {}{unit}",py_float(min))));}
        if let Some(max)=field.maximum.filter(|max|number>*max) {return Err(value(format!("{label} must be at most {}{unit}",py_float(max))));}
    }
    Ok(answer)
}

pub fn values(fields:&[Field],supplied:&Map<String,Value>,complete:bool)->Result<(Map<String,Value>,bool)> {
    let unknown:Vec<_>=supplied.keys().filter(|key|!fields.iter().any(|field|field.name==**key)).collect();
    if !unknown.is_empty() { return Err(value(format!("unknown authoring fields: {unknown:?}"))); }
    let mut effective=effective(fields,supplied); let mut result=Map::new(); let mut whole=true;
    for field in fields {
        if !enabled(field,&effective) {result.insert(field.name.clone(),effective[&field.name].clone());continue;}
        let answer=project(field,&effective[&field.name],complete)?;
        if field.required && field.vacant(&answer) {
            if complete {return Err(value(if field.kind=="rows" {format!("{} must have at least one row",field.label())} else {format!("{} must be filled in",field.label())}));}
            whole=false;
        }
        effective.insert(field.name.clone(),answer.clone()); result.insert(field.name.clone(),answer);
    }
    Ok((result,whole))
}
