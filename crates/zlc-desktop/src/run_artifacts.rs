//! Run-contained publication and registration, sharing archive durability with offline tools.
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path,PathBuf};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use serde_json::{Value,json};
use zlc_archive::{ArchiveError,durable::{self,StagedWrite}};
use zlc_plugin_host::Lent;

#[derive(Default)]
struct RunState { stages:HashMap<String,Pending>, names:HashMap<String,PathBuf> }
struct Pending { write:StagedWrite, options:Value }
#[derive(Default)]
struct Slot {state:Mutex<RunState>,retired:AtomicBool}
pub(crate) struct Store {
    runs:Mutex<HashMap<String,Option<Arc<Slot>>>>,
    cleanup:zlc_plugin_host::workers::Workers,
}
impl Default for Store {
    fn default()->Self{Self{runs:Mutex::default(),cleanup:zlc_plugin_host::workers::Workers::new("zlc-artifact-cleanup")}}
}
struct Failure {kind:String,message:String,published:Option<PathBuf>,artifacts:Vec<Value>}
type Result<T>=std::result::Result<T,Failure>;
impl From<ArchiveError> for Failure {fn from(error:ArchiveError)->Self{Self{kind:error.kind.python_name().into(),message:error.message,published:None,artifacts:Vec::new()}}}
fn bad(message:impl Into<String>)->Failure{Failure{kind:"ValueError".into(),message:message.into(),published:None,artifacts:Vec::new()}}
fn io(error:std::io::Error)->Failure{Failure{kind:"OSError".into(),message:error.to_string(),published:None,artifacts:Vec::new()}}

pub(crate) fn confined(directory:&Path,path:&Path)->std::result::Result<PathBuf,String>{
    let root=durable::resolve(directory).map_err(|e|e.to_string())?;
    let target=durable::resolve(&if path.is_absolute(){path.to_path_buf()}else{root.join(path)}).map_err(|e|e.to_string())?;
    let relative=target.strip_prefix(&root).map_err(|_|format!("{} is outside the run directory",target.display()))?;
    if relative.as_os_str().is_empty(){return Err("an artifact must be a file below the run directory".into());}
    if cfg!(windows) && relative.components().any(|part|part.as_os_str().to_string_lossy().contains(':')){return Err("artifact names cannot address alternate data streams".into());}
    if relative.components().any(|c|matches!(c,std::path::Component::ParentDir)){return Err("an artifact path escapes its run directory".into());}
    if relative.components().count()==1 && matches!(relative.to_string_lossy().to_lowercase().as_str(),"start.json"|"run.json"){
        return Err("start.json and run.json are the run's own records".into());
    }
    Ok(target)
}

impl Store {
    fn slot(&self,run:&str)->Result<Arc<Slot>>{
        self.runs.lock().unwrap().entry(run.to_owned()).or_insert_with(||Some(Arc::new(Slot::default())))
            .clone().ok_or_else(||bad("the artifact run has ended"))
    }

    /// The UI only marks retirement and removes the registry entry. A running
    /// write retains its slot and can report publication evidence; waiting
    /// operations see the flag before touching files. Temporary cleanup waits
    /// for that owner on a background worker, never on the window's thread.
    pub fn retire(&self,run:&str){
        let slot={
            let mut runs=self.runs.lock().unwrap();
            let slot=runs.entry(run.to_owned()).or_default().take();
            if let Some(slot)=&slot{slot.retired.store(true,Ordering::Release);}
            slot
        };
        if let Some(slot)=slot{
            self.cleanup.run(move||{
                let discarded=std::mem::take(&mut *slot.state.lock().unwrap());
                drop(discarded); // file cleanup happens outside every registry lock
            });
        }
    }

    fn execute_request(&self,run:&str,directory:Option<&Path>,params:&Value,lent:Option<&Lent>)->Result<(Value,Vec<Value>)>{
        let slot=self.slot(run)?;
        let mut state=slot.state.lock().unwrap();
        if slot.retired.load(Ordering::Acquire){return Err(bad("the artifact run has ended"));}
        let root=directory.ok_or_else(||bad("this run has no run directory"))?;
        let root=durable::resolve(root)?;
        execute(&mut state,run,&root,params,lent)
    }

    /// Errors are data so published-but-not-durable carries its visible path
    /// through the ordinary plugin error wire without losing that distinction.
    pub fn answer(&self,run:&str,directory:Option<&Path>,params:&Value,lent:&Lent)->(Value,Vec<Value>){
        match self.execute_request(run,directory,params,Some(lent)){
            Ok((answer,artifacts))=>(json!({"ok":answer}),artifacts),
            Err(error)=>(json!({"error":{"type":error.kind,"message":error.message,"published":error.published}}),error.artifacts),
        }
    }
}

fn register(state:&mut RunState,run:&str,root:&Path,path:&Path,p:&Value)->Result<Value>{
    let path=confined(root,path).map_err(bad)?;
    let name=p["name"].as_str().filter(|s|!s.trim().is_empty()&&s.trim()==*s).ok_or_else(||bad("an artifact needs a canonical name"))?;
    let role=p["role"].as_str().filter(|s|!s.is_empty()).ok_or_else(||bad("an artifact needs a role"))?;
    if state.names.get(name).is_some_and(|prior|prior!=&path){return Err(bad(format!("artifact name {name:?} already names another path")));}
    let metadata=std::fs::metadata(&path).map_err(io)?;
    if !metadata.is_file(){return Err(bad("an artifact must be a regular file"));}
    let relative=path.strip_prefix(root).map_err(|_|bad("artifact escaped its run directory"))?;
    let mut artifact=json!({"name":name,"path":relative.to_string_lossy().replace('\\',"/"),"role":role,
        "contract_id":p["contract_id"].as_str().unwrap_or(""),"size_bytes":metadata.len(),"run":run});
    if let Some(lineage)=p.get("lineage"){artifact["lineage"]=lineage.clone();}
    state.names.insert(name.into(),path);
    Ok(artifact)
}

fn registration_options(state:&RunState,path:&Path,p:&Value)->Result<()>{
    if let Some(name)=p.get("name"){
        let name=name.as_str().filter(|name|!name.is_empty()&&name.trim()==*name).ok_or_else(||bad("an artifact needs a canonical name"))?;
        if p["role"].as_str().is_none_or(|role|role.is_empty()){return Err(bad("an artifact needs a role"));}
        if state.names.get(name).is_some_and(|prior|prior!=path){return Err(bad(format!("artifact name {name:?} already names another path")));}
        if p["unique"]==true && state.names.contains_key(name){return Err(bad(format!("unique artifact name {name:?} is already registered")));}
    }
    Ok(())
}

fn publication_failure(state:&mut RunState,run:&str,root:&Path,pending:&Pending,error:ArchiveError)->Failure{
    let mut failure=Failure::from(error);
    failure.published=pending.write.published().map(Path::to_path_buf);
    if let Some(path)=&failure.published{
        if pending.options["name"].is_string(){
            if let Ok(mut artifact)=register(state,run,root,path,&pending.options){
                artifact["durable"]=json!(false);artifact["publication_error"]=json!(failure.message);
                failure.artifacts.push(artifact);
            }
        }
    }
    failure
}

fn finish(state:&mut RunState,run:&str,root:&Path,mut pending:Pending)->Result<(Value,Vec<Value>)>{
    confined(root,pending.write.destination()).map_err(bad)?;
    registration_options(state,pending.write.destination(),&pending.options)?;
    let path=match pending.write.finish(pending.options["unique"].as_bool().unwrap_or(false)){
        Ok(path)=>path,Err(error)=>return Err(publication_failure(state,run,root,&pending,error))
    };
    registered_publication(state,run,root,path,&pending.options)
}

fn registered_publication(state:&mut RunState,run:&str,root:&Path,path:PathBuf,options:&Value)->Result<(Value,Vec<Value>)>{
    let artifacts=if options["name"].is_string(){vec![register(state,run,root,&path,options).map_err(|mut error|{error.published=Some(path.clone());error})?]}else{Vec::new()};
    Ok((json!({"path":path,"published":true,"durable":true,"artifact":artifacts.first()}),artifacts))
}

fn execute(state:&mut RunState,run:&str,root:&Path,p:&Value,lent:Option<&Lent>)->Result<(Value,Vec<Value>)>{
    let action=p["action"].as_str().unwrap_or("");
    if matches!(action,"finalize"|"abort"){
        let token=p["token"].as_str().ok_or_else(||bad("a staged artifact needs its token"))?;
        let mut pending=state.stages.remove(token).ok_or_else(||bad("the staged artifact token is not owned by this run"))?;
        if action=="abort"{drop(pending);return Ok((json!({"aborted":true}),Vec::new()));}
        if let Some(candidate)=p["candidate"].as_str(){
            let path=confined(root,Path::new(candidate)).map_err(bad)?;
            registration_options(state,&path,&pending.options)?;
            return match pending.write.finish_exclusive(&path){
                Ok(path)=>registered_publication(state,run,root,path,&pending.options),
                Err(error)=>{
                    let failure=publication_failure(state,run,root,&pending,error.clone());
                    if error.kind==zlc_archive::ErrorKind::FileExists{state.stages.insert(token.to_owned(),pending);}
                    Err(failure)
                }
            };
        }
        return finish(state,run,root,pending);
    }
    let wanted=p["path"].as_str().ok_or_else(||bad("an artifact needs a path"))?;
    if matches!(action,"flush"|"mkdir"|"makedirs"|"claim_directory"){
        let value=Path::new(wanted);
        let path=durable::resolve(&if value.is_absolute(){value.to_path_buf()}else{root.join(value)})?;
        if !path.starts_with(root){return Err(bad("directory is outside the run directory"));}
        if path!=root{confined(root,&path).map_err(bad)?;}
        let result=match action{
            "flush"=>durable::flush_directory(&path).map(|_|path.clone()),
            "mkdir"=>durable::durable_mkdir(&path),"makedirs"=>durable::durable_makedirs(&path),
            _=>durable::claim_directory(&path),
        };
        return match result{Ok(path)=>Ok((json!({"path":path}),Vec::new())),Err(error)=>{
            let mut failure=Failure::from(error.clone());if error.kind==zlc_archive::ErrorKind::DirectoryDurability && path.exists(){failure.published=Some(path);}Err(failure)
        }};
    }
    let path=confined(root,Path::new(wanted)).map_err(bad)?;
    if action=="register"{
        let artifact=register(state,run,root,&path,p)?;
        return Ok((artifact.clone(),vec![artifact]));
    }
    if !matches!(action,"stage"|"write_text"|"write_bytes"|"write_json"){return Err(bad(format!("unknown artifact action {action:?}")));}
    registration_options(state,&path,p)?;
    if action=="stage" && state.stages.len()>=128{return Err(bad("this run already owns 128 staged artifacts"));}
    durable::durable_makedirs(path.parent().unwrap())?;
    let pending=Pending{write:StagedWrite::with_temporary(&path,p["temporary_prefix"].as_str(),p["temporary_suffix"].as_str())?,options:p.clone()};
    if action=="stage"{
        let token=uuid::Uuid::new_v4().to_string();
        let answer=json!({"token":token,"path":pending.write.temporary(),"target":path});
        state.stages.insert(token,pending);return Ok((answer,Vec::new()));
    }
    let mut file=std::fs::OpenOptions::new().write(true).truncate(true).open(pending.write.temporary()).map_err(io)?;
    if action=="write_bytes"{
        let array=lent.ok_or_else(||bad("artifact byte block is missing"))?.array(&p["bytes"]).map_err(bad)?;
        if !matches!(&array.values,zlc_core::AnyBuffer::U8(_)){return Err(bad("artifact bytes must have uint8 dtype"));}
        file.write_all(array.values.as_bytes()).map_err(io)?;
    }else{
        let text=if action=="write_json"{serde_json::to_string_pretty(&p["value"]).map_err(|e|bad(e.to_string()))?+"\n"}
            else{p["text"].as_str().ok_or_else(||bad("artifact text must be a string"))?.to_owned()};
        file.write_all(text.as_bytes()).map_err(io)?;
    }
    drop(file);finish(state,run,root,pending)
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn retiring_a_busy_run_never_waits_for_its_io_or_blocks_other_runs(){
        use std::sync::mpsc;
        use std::time::Duration;
        let root=std::env::temp_dir().join(format!("zlc-artifact-parallel-{}",uuid::Uuid::new_v4()));std::fs::create_dir_all(&root).unwrap();
        let store=Arc::new(Store::default());
        let slot=store.slot("busy").ok().unwrap();
        let held=slot.state.lock().unwrap();
        let(done,finished)=mpsc::channel();
        let retiring={let store=store.clone();std::thread::spawn(move||{store.retire("busy");done.send(()).unwrap();})};
        let retired_promptly=finished.recv_timeout(Duration::from_secs(1)).is_ok();
        let(done,finished)=mpsc::channel();
        let other={let store=store.clone();let root=root.clone();std::thread::spawn(move||{
            let result=store.execute_request("other",Some(&root),&json!({"action":"write_text","path":"other.txt","text":"complete"}),None);
            done.send(result.is_ok()).unwrap();
        })};
        let other_completed=finished.recv_timeout(Duration::from_secs(1)).unwrap_or(false);
        drop(held);retiring.join().unwrap();other.join().unwrap();
        assert!(retired_promptly,"retirement waited for the per-run state lock");
        assert!(other_completed,"another run inherited the busy run's IO lock");
        assert!(store.slot("busy").is_err(),"retirement must leave a tombstone");
        assert_eq!(std::fs::read(root.join("other.txt")).unwrap(),b"complete");store.retire("other");std::fs::remove_dir_all(root).unwrap();
    }

    #[test]fn queued_work_rechecks_retirement_and_cleanup_runs_after_the_owner_releases(){
        use std::time::{Duration,Instant};
        let root=std::env::temp_dir().join(format!("zlc-artifact-retire-{}",uuid::Uuid::new_v4()));std::fs::create_dir_all(&root).unwrap();
        let store=Arc::new(Store::default());
        let(stage,_)=store.execute_request("run",Some(&root),&json!({"action":"stage","path":"unfinished.bin"}),None).ok().unwrap();
        let temporary=PathBuf::from(stage["path"].as_str().unwrap());
        let slot=store.slot("run").ok().unwrap();let held=slot.state.lock().unwrap();
        let waiting={let store=store.clone();let root=root.clone();std::thread::spawn(move||
            store.execute_request("run",Some(&root),&json!({"action":"write_text","path":"late.txt","text":"late"}),None)
        )};
        // Registry + this test own two Arcs. The third proves execute_request
        // acquired its slot and is waiting on the deliberately held state lock.
        let until=Instant::now()+Duration::from_secs(1);while Arc::strong_count(&slot)<3&&Instant::now()<until{std::thread::sleep(Duration::from_millis(1));}
        let queued=Arc::strong_count(&slot)>=3;
        store.retire("run");assert!(temporary.exists(),"cleanup raced the current owner");drop(held);
        assert!(waiting.join().unwrap().is_err());assert!(queued,"request did not reach its per-run wait");assert!(!root.join("late.txt").exists());
        let until=Instant::now()+Duration::from_secs(2);while temporary.exists()&&Instant::now()<until{std::thread::sleep(Duration::from_millis(2));}
        assert!(!temporary.exists(),"retired staging file was not cleaned in the background");std::fs::remove_dir_all(root).unwrap();
    }

    #[test]fn staged_writer_abort_and_run_ownership_preserve_registration(){
        let root=std::env::temp_dir().join(format!("zlc-artifact-stage-{}",uuid::Uuid::new_v4()));std::fs::create_dir_all(&root).unwrap();
        let mut state=RunState::default();
        let (stage,_)=execute(&mut state,"run-a",&root,&json!({"action":"stage","path":"nested/data.json","name":"data","role":"final","lineage":{"parent":"sample"}}),None).ok().unwrap();
        std::fs::write(stage["path"].as_str().unwrap(),b"complete").unwrap();
        assert!(execute(&mut RunState::default(),"run-b",&root,&json!({"action":"finalize","token":stage["token"]}),None).is_err());
        let(answer,artifacts)=execute(&mut state,"run-a",&root,&json!({"action":"finalize","token":stage["token"]}),None).ok().unwrap();
        assert_eq!(answer["durable"],true);assert_eq!(artifacts[0]["lineage"]["parent"],"sample");assert_eq!(artifacts[0]["run"],"run-a");
        let (stage,_)=execute(&mut state,"run-a",&root,&json!({"action":"stage","path":"discard.json"}),None).ok().unwrap();
        let temporary=PathBuf::from(stage["path"].as_str().unwrap());execute(&mut state,"run-a",&root,&json!({"action":"abort","token":stage["token"]}),None).ok().unwrap();
        assert!(!temporary.exists());assert!(!root.join("discard.json").exists());std::fs::remove_dir_all(root).unwrap();
    }

    #[test]fn resolved_paths_stay_inside_the_run_and_reserved_records_are_case_insensitive(){
        let root=std::env::temp_dir().join(format!("zlc-artifact-paths-{}",uuid::Uuid::new_v4()));std::fs::create_dir_all(&root).unwrap();
        assert!(confined(&root,Path::new("../outside.txt")).is_err());assert!(confined(&root,Path::new("RUN.JSON")).is_err());
        assert_eq!(confined(&root,Path::new("nested/value.json")).unwrap(),root.join("nested/value.json"));std::fs::remove_dir(&root).unwrap();
    }
}
