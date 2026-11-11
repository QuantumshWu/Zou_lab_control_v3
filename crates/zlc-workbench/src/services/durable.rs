//! Shared offline durability router. Hosted Task publication uses the run-confined facade.
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use serde_json::{Value,json};
use zlc_archive::{ArchiveError,ErrorKind,durable::{self,StagedWrite}};
use zlc_core::scientific::Tensor;

#[derive(Default)]
pub struct Service{next:u64,stages:HashMap<u64,StagedWrite>}
fn error(error:ArchiveError,published:Option<&Path>)->Value{json!({"error":{"type":error.kind.python_name(),"message":error.message,"published":published}})}
impl Service{
    pub fn dispatch(&mut self,op:&str,p:&Value,read:&dyn Fn(&Value)->Result<Tensor,String>)->Value{
        if op=="durable.abort"{
            let handle=p["handle"].as_u64().unwrap_or(0);
            return if self.stages.remove(&handle).is_some(){json!({"ok":{}})}else{error(ArchiveError::value("unknown staged write"),None)};
        }
        if op=="durable.finalize"{
            let handle=p["handle"].as_u64().unwrap_or(0);
            let Some(mut stage)=self.stages.remove(&handle)else{return error(ArchiveError::value("unknown staged write"),None)};
            let result=if let Some(candidate)=p["candidate"].as_str(){stage.finish_exclusive(Path::new(candidate))}else{stage.finish(p["unique"].as_bool().unwrap_or(false))};
            return match result{
                Ok(path)=>json!({"ok":{"path":path,"published":true,"durable":true}}),
                Err(failure)=>{
                    let answer=error(failure.clone(),stage.published());
                    if failure.kind==ErrorKind::FileExists && p["candidate"].is_string(){self.stages.insert(handle,stage);}
                    answer
                }
            };
        }
        if op=="durable.stage" && self.stages.len()>=128{return error(ArchiveError::value("128 staged writes are already active"),None);}
        let Some(path)=p["path"].as_str()else{return error(ArchiveError::value("durable operation needs a path"),None)};
        let path=Path::new(path);
        match op{
            "durable.stage"=>match StagedWrite::with_temporary(path,p["temporary_prefix"].as_str(),p["temporary_suffix"].as_str()){
                Ok(stage)=>{self.next+=1;let answer=json!({"ok":{"handle":self.next,"path":stage.temporary(),"target":stage.destination()}});self.stages.insert(self.next,stage);answer}
                Err(failure)=>error(failure,None)
            },
            "durable.write_bytes"=>{
                let mut stage=match StagedWrite::new(path){Ok(stage)=>stage,Err(failure)=>return error(failure,None)};
                let result=(||{
                    let array=read(&p["bytes"]).map_err(ArchiveError::value)?;
                    if !matches!(&array.values,zlc_core::AnyBuffer::U8(_)){return Err(ArchiveError::type_error("durable bytes need uint8 dtype"));}
                    let mut file=std::fs::OpenOptions::new().write(true).truncate(true).open(stage.temporary()).map_err(|e|ArchiveError::os(&e,None))?;
                    file.write_all(array.values.as_bytes()).map_err(|e|ArchiveError::os(&e,None))?;drop(file);stage.finish(false)
                })();
                match result{Ok(path)=>json!({"ok":{"path":path}}),Err(failure)=>error(failure,stage.published())}
            },
            "durable.flush"=>match durable::flush_directory(path){Ok(())=>json!({"ok":{}}),Err(failure)=>error(failure,None)},
            _=>{
                let result=match op{"durable.mkdir"=>durable::durable_mkdir(path),"durable.makedirs"=>durable::durable_makedirs(path),
                    "durable.claim_directory"=>durable::claim_directory(path),"durable.resolve"=>durable::resolve(path),
                    _=>Err(ArchiveError::value(format!("unknown durable operation {op}")))};
                match result{Ok(path)=>json!({"ok":{"path":path}}),Err(failure)=>{
                    let published=(failure.kind==ErrorKind::DirectoryDurability && path.exists()).then_some(path);error(failure,published)
                }}
            }
        }
    }
}
