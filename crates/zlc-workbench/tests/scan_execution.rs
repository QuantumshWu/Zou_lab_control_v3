//! The scan application runs against generic run capabilities, with no plugin
//! transport or UI. Publications still use the real Runtime and immutable data.
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::time::Duration;
use serde_json::{Value, json};
use zlc_core::{DType, OwnedBuffer};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, DeliveryKind, DeliveryPolicy, Generation, GenerationId, Input, Publication, Read, Runtime, Start, Storage, Subscription};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValueSchema, ValidityContract};
use zlc_workbench::scan::execution::{self, Context, Resource};

#[derive(Default)]
struct State { resources: HashMap<String, Resource>, output: Option<(Generation, Arc<str>)> }
struct Run {
    runtime: Runtime, source: Generation, schema: Arc<DatasetSchema>, input: Subscription,
    state: Mutex<State>, active: AtomicBool,
}

impl Run {
    fn new() -> Self {
        let runtime=Runtime::new();
        let scalar=Domain::new(vec![1],vec![],vec![]).unwrap();
        let axis=Axis::new("source.pixel".into(),"pixel".into(),"coordinate".into(),Coordinates::Index{origin:0,len:2},None,None,None,None).unwrap();
        let schema=DatasetSchema::new(scalar.clone(),Domain::dense(vec![2],vec![axis]).unwrap(),scalar,
            ValueSchema::new(DType::F64,None,None,ValidityContract::Value).unwrap()).unwrap();
        let source=runtime.begin("camera",vec![Declaration{signal:"source".into(),schema:schema.clone(),storage:Storage::Latest}],json!({})).unwrap();
        let input=runtime.subscribe(&source,vec!["source".into()],DeliveryPolicy::Follow,Start::Current).unwrap();
        Self{runtime,source,schema,input,state:Mutex::default(),active:AtomicBool::new(true)}
    }
    fn publish(&self, ordinal:u64)->Arc<Publication>{
        let data=DataBlock::new(self.schema.clone(),OwnedBuffer::from_vec(vec![ordinal as f64;2]).into(),Validity::Uniform(true),None).unwrap();
        self.runtime.commit(&self.source,&Bundle{inputs:vec![Input::Event{signal:"source".into(),ordinal,data,origin:None,shot_time:None}],
            metadata:json!({"records":{"source":{"point":ordinal}}}),parents:vec![]}).unwrap().publications[0].clone()
    }
}

impl Context for Run {
    fn check_active(&self)->Result<(),String>{if self.active.load(Ordering::Acquire){Ok(())}else{Err("the scan run has ended".into())}}
    fn input_next(&self,timeout:Duration,terminal_errors:bool)->Result<Option<Read>,String>{
        self.check_active()?;
        match self.input.next(timeout).map_err(|error|error.to_string())? {
            Some(delivery) if matches!(delivery.kind,DeliveryKind::Data)=>Ok(Some(delivery.reads[0].clone())),
            Some(_) if terminal_errors=>Err("the source signal ended during collection".into()),
            _=>Ok(None),
        }
    }
    fn input_drain(&self,quiet:Duration)->Result<usize,String>{
        let mut count=0;
        while self.input_next(quiet,false)?.is_some(){count+=1;}
        Ok(count)
    }
    fn input_await_new_source(&self)->Result<(),String>{Ok(())}
    fn input_generation(&self)->Result<Option<GenerationId>,String>{Ok(Some(self.source.id()))}
    fn begin_output(&self,name:&str,schema:Arc<DatasetSchema>,record:Value)->Result<(Generation,Arc<str>),String>{
        self.check_active()?;
        if name!="scan"{return Err("output is not declared".into());}
        let mut state=self.state.lock().unwrap();
        if state.output.is_some(){return Err("outputs already began".into());}
        let signal:Arc<str>="scan".into();
        let generation=self.runtime.begin_consuming("scan",vec![Declaration{signal:signal.clone(),schema,storage:Storage::Finite}],record,
            "source".into(),Some(self.source.id())).map_err(|error|error.to_string())?;
        self.runtime.bind_consumer(&generation,self.input.ticket()).map_err(|error|error.to_string())?;
        state.output=Some((generation.clone(),signal.clone()));
        Ok((generation,signal))
    }
    fn commit(&self,generation:&Generation,bundle:&Bundle)->Result<(),String>{
        self.check_active()?;
        self.runtime.commit(generation,bundle).map(|_|()).map_err(|error|error.to_string())
    }
    fn resource(&self,key:&str)->Result<Option<Resource>,String>{Ok(self.state.lock().unwrap().resources.get(key).cloned())}
    fn install_resource(&self,key:&str,value:Arc<dyn Any+Send+Sync>)->Result<Resource,String>{
        self.check_active()?;
        let mut state=self.state.lock().unwrap();
        if state.output.is_some(){return Err("prepare before outputs begin".into());}
        Ok(state.resources.entry(key.into()).or_insert(value).clone())
    }
}

fn prepare(run:&Run)->String{
    execution::dispatch("run.scan.prepare",&json!({"spec":{"axes":[{"port":"pulse:param:x","identity":"x","label":"x","unit":"us",
        "values":[1.,2.,3.],"host_advanced":false}],"scan_repeats":2,"run_repeats":2}}),run).unwrap()["handle"].as_str().unwrap().into()
}
fn configure(run:&Run,handle:&str){
    execution::dispatch("run.scan.configure",&json!({"handle":handle,"output":"scan","record":{"experiment":"test"}}),run).unwrap();
    execution::dispatch("run.scan.arm",&json!({"handle":handle,"row_offset":0,"scan_repeat_base":0,"sweeps":2,"quiet_seconds":0,
        "actual_devices":{"rf":{"frequency":80.25}}}),run).unwrap();
}

#[test]
fn generic_context_places_same_data_and_parents_with_native_pulse_facts(){
    let run=Run::new();let handle=prepare(&run);
    execution::accept_pulse(&handle,Some(vec![vec![1.1],vec![2.1],vec![3.1]]),json!({"config":{},"program":{"rows":[[10],[20],[30]]}}),&run).unwrap();
    configure(&run,&handle);
    let parents:Vec<_>=(10..22).map(|ordinal|run.publish(ordinal)).collect();
    let answer=execution::dispatch("run.scan.collect",&json!({"handle":handle,"timeout_seconds":1.0,"max_events":1024}),&run).unwrap();
    assert_eq!(answer["written"],12);assert_eq!(answer["complete"],true);
    let output=run.state.lock().unwrap().output.clone().unwrap();
    let latest=run.runtime.read_latest(&output.0,&output.1).unwrap();
    let stored=latest.canonical().unwrap();
    assert_eq!((stored.written_cells(),stored.total_cells()),(24,24));
    for(index,(segment,parent))in stored.segments().zip(&parents).enumerate(){
        assert_eq!(segment.origin(),[index/6*2+index%2,index%6/2*2]);
        assert!(Arc::ptr_eq(segment.data(),parent.data().unwrap()));
    }
    assert_eq!(latest.key().ordinal,Some(21));
    assert_eq!(latest.event().parents[0].key,*parents[11].key());
    assert_eq!(latest.event().metadata["records"]["scan"]["actual_devices"]["rf"]["frequency"],80.25);
    assert_eq!(latest.run().metadata["device_snapshots"]["sequencer"]["program"]["rows"],json!([[10],[20],[30]]));
    assert_eq!(execution::plan(&handle,&run).unwrap().axes()[0].values,[1.1,2.1,3.1]);
}

#[test]
fn bad_handles_do_not_poison_state_but_lost_shots_do(){
    let run=Run::new();let handle=prepare(&run);configure(&run,&handle);
    assert!(execution::plan("old-run-handle",&run).unwrap_err().contains("not current"));
    assert!(execution::plan(&handle,&run).is_ok());
    run.publish(1);run.publish(3);
    let params=json!({"handle":handle,"timeout_seconds":1.0,"max_events":1024});
    let error=execution::dispatch("run.scan.collect",&params,&run).unwrap_err();
    assert!(error.contains("ordinal jumped"));
    assert_eq!(execution::dispatch("run.scan.collect",&params,&run).unwrap_err(),error);
    assert_eq!(execution::plan(&handle,&run).unwrap_err(),error);
}

#[test]
fn preparation_is_once_before_publication_and_resources_expire_with_run(){
    let run=Run::new();let handle=prepare(&run);
    let duplicate=execution::dispatch("run.scan.prepare",&json!({"spec":{}}),&run).unwrap_err();
    assert!(duplicate.contains("already prepared"));
    run.active.store(false,Ordering::Release);
    assert!(execution::plan(&handle,&run).unwrap_err().contains("ended"));
    let other=Run::new();let fresh=prepare(&other);assert_ne!(fresh,handle);
    assert!(execution::plan(&handle,&other).unwrap_err().contains("not current"));
    let published=Run::new();published.begin_output("scan",published.schema.clone(),json!({})).unwrap();
    let spec=json!({"axes":[{"port":"x","identity":"x","label":"x","unit":"","values":[1.],"host_advanced":false}],"scan_repeats":1,"run_repeats":1});
    assert!(execution::dispatch("run.scan.prepare",&json!({"spec":spec}),&published).unwrap_err().contains("before outputs"));
}

#[test]
fn the_plan_is_read_while_a_collect_waits_for_its_source(){
    let run=Run::new();let handle=prepare(&run);configure(&run,&handle);
    let collect=json!({"handle":handle,"timeout_seconds":1.0,"max_events":1024});
    std::thread::scope(|scope|{
        // No source event comes: the collect waits its whole slice for one.
        let waiting=scope.spawn(||execution::dispatch("run.scan.collect",&collect,&run));
        std::thread::sleep(Duration::from_millis(100));
        let started=std::time::Instant::now();
        assert_eq!(execution::plan(&handle,&run).unwrap().axes()[0].values,[1.,2.,3.]);
        execution::accept_pulse(&handle,None,json!({"config":{}}),&run).unwrap();
        assert!(started.elapsed()<Duration::from_millis(200),"the plan never waits for a scan operation");
        assert!(execution::dispatch("run.scan.rows",&json!({"handle":handle,"scope":"all","start":0,"count":1}),&run).unwrap_err().contains("another operation"));
        assert_eq!(waiting.join().unwrap().unwrap()["collected"],0);
    });
}
