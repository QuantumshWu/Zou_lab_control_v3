//! JSON boundary for coarse capture evidence, backed by core typed rules.
//! These pure validations may run during Stop cleanup while the run still exists.
use serde_json::{Value,json};
use zlc_core::capture::{self,TerminalEvidence};

fn integer(args:&Value,name:&str)->Result<u64,String>{
    args[name].as_u64().ok_or_else(||format!("{name} must be a nonnegative integer"))
}
fn boolean(args:&Value,name:&str,default:Option<bool>)->Result<bool,String>{
    match args.get(name){
        Some(value)=>value.as_bool().ok_or_else(||format!("{name} must be boolean")),
        None=>default.ok_or_else(||format!("{name} must be boolean")),
    }
}
fn ordinals(args:&Value)->Result<Vec<u64>,String>{
    args["ordinals"].as_array().ok_or("ordinals must be an array")?.iter().map(|value|
        value.as_u64().ok_or_else(||"record ordinals must be nonnegative integers".to_owned())).collect()
}

pub fn dispatch(operation:&str,args:&Value)->Result<Value,String>{
    let group_size=if args.get("group_size").is_some(){integer(args,"group_size")?}else{1};
    match operation{
        "capture.batch"=>{
            let evidence=capture::validate_batch(integer(args,"expected_start")?,&ordinals(args)?,group_size).map_err(|error|error.to_string())?;
            Ok(json!({"next":evidence.next,"group_ends":evidence.group_ends}))
        }
        "capture.group"=>{
            capture::validate_group(integer(args,"expected_start")?,&ordinals(args)?,group_size).map_err(|error|error.to_string())?;
            Ok(json!({}))
        }
        "capture.terminal"=>{
            let proof=TerminalEvidence{produced_count:integer(args,"produced")?,source_stopped:boolean(args,"source_stopped",None)?,joined:boolean(args,"joined",None)?};
            let expected=if args["expected"].is_null(){None}else{Some(integer(args,"expected")?)};
            let evidence=capture::validate_terminal(proof,expected,boolean(args,"stopped",Some(false))?,group_size).map_err(|error|error.to_string())?;
            Ok(json!({"produced":evidence.produced,"complete_groups":evidence.complete_groups,"partial_records":evidence.partial_records}))
        }
        _=>Err(format!("unknown capture operation {operation:?}")),
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn batch_and_terminal_wire_values_match_typed_evidence(){
        assert_eq!(dispatch("capture.batch",&json!({"expected_start":2,"ordinals":[2,3,4,5,6],"group_size":3})).unwrap(),json!({"next":7,"group_ends":[2,5]}));
        let proof=json!({"produced":5,"source_stopped":true,"joined":true,"expected":4,"stopped":true,"group_size":2});
        assert_eq!(dispatch("capture.terminal",&proof).unwrap(),json!({"produced":5,"complete_groups":2,"partial_records":1}));
    }
    #[test]fn malformed_wire_values_cannot_be_coerced_into_capture_facts(){
        for args in [json!({"expected_start":0,"ordinals":[true]}),json!({"expected_start":0,"ordinals":[0.5]}),json!({"expected_start":0,"ordinals":[0],"group_size":0})]{assert!(dispatch("capture.batch",&args).is_err());}
        assert!(dispatch("capture.terminal",&json!({"produced":2,"source_stopped":"yes","joined":true})).is_err());
        assert!(dispatch("capture.group",&json!({"expected_start":0,"ordinals":[0,2],"group_size":2})).is_err());
    }
}
