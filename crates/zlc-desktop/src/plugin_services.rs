//! The application composes its existing domains behind the plugin host's
//! generic service boundary. No domain implementation is built into the host.
use std::sync::{Arc,Mutex};
use std::time::Duration;
use serde_json::Value;
use zlc_core::runtime::{Bundle,Generation,GenerationId,Read};
use zlc_core::schema::DatasetSchema;
use zlc_plugin_host::{CallError,ErrorKind,PluginHost,services::{Context,RunContext,Resource}};
use zlc_workbench::{services::Service,scan::execution};

struct ScanContext<'a>(&'a RunContext);
impl execution::Context for ScanContext<'_>{
    fn check_active(&self)->Result<(),String>{self.0.check_running().map_err(|e|e.message)}
    fn input_next(&self,t:Duration,terminal:bool)->Result<Option<Read>,String>{self.0.input_next(t,terminal).map_err(|e|e.message)}
    fn input_drain(&self,t:Duration)->Result<usize,String>{self.0.input_drain(t).map_err(|e|e.message)}
    fn input_await_new_source(&self)->Result<(),String>{self.0.input_await_new_source().map_err(|e|e.message)}
    fn input_generation(&self)->Result<Option<GenerationId>,String>{self.0.input_generation().map_err(|e|e.message)}
    fn begin_output(&self,n:&str,s:Arc<DatasetSchema>,r:Value)->Result<(Generation,Arc<str>),String>{self.0.begin_consuming_finite_output(n,s,r).map_err(|e|e.message)}
    fn commit(&self,g:&Generation,b:&Bundle)->Result<(),String>{self.0.commit(g,b).map_err(|e|e.message)}
    fn resource(&self,k:&str)->Result<Option<Resource>,String>{self.0.resource(k).map_err(|e|e.message)}
    fn install_resource(&self,k:&str,v:Resource)->Result<Resource,String>{self.0.install_resource(k,v,true).map_err(|e|e.message)}
}

fn invalid(message:String)->CallError{CallError::new(ErrorKind::Invalid,message)}
fn call(operation:&str,params:&Value,context:&Context)->Result<Value,CallError>{
    if operation.starts_with("durable."){return Err(CallError::new(ErrorKind::Refused,"run files use the artifact service"));}
    if operation.starts_with("run.scan."){
        let run=context.run()?;
        let result=execution::dispatch(operation,params,&ScanContext(run));
        run.check_running()?;
        return result.map_err(invalid);
    }
    let computation=operation.starts_with("scientific.");
    let check=||->Result<(),CallError>{
        if let Some(run)=context.optional_run(){if computation{run.check_computation()?;}else{run.check_active()?;}}Ok(())
    };
    check()?;
    let key="zlc.workbench.services";
    let service=match context.resource(key)?{
        Some(held)=>held,
        None=>context.install_resource(key,Arc::new(Mutex::new(Service::default())))?,
    }.downcast::<Mutex<Service>>().map_err(|_|invalid("application service state has the wrong type".into()))?;
    let read=|value:&Value|{check().map_err(|e|e.message)?;let a=context.read_array(value).map_err(|e|e.message)?;Ok(zlc_core::scientific::Tensor{values:a.values,shape:a.shape})};
    let write=|array:zlc_core::scientific::Tensor|{check().map_err(|e|e.message)?;context.write_array(&array.values,&array.shape).map_err(|e|e.message)};
    let result=service.lock().unwrap().dispatch(operation,params,&read,&write);
    check()?;result.map_err(invalid)
}

pub fn install(host:&PluginHost)->Result<(),String>{host.install_services(Arc::new(call)).map_err(|e|e.to_string())}
pub fn scan_plan(host:&PluginHost,run:&str,handle:&str)->Result<Arc<zlc_core::scan::Plan>,CallError>{
    let context=host.run_context(run)?;execution::plan(handle,&ScanContext(&context)).map_err(invalid)
}
pub fn accept_scan_pulse(host:&PluginHost,run:&str,handle:&str,rows:Option<Vec<Vec<f64>>>,applied:Value)->Result<(),CallError>{
    let context=host.run_context(run)?;execution::accept_pulse(handle,rows,applied,&ScanContext(&context)).map_err(invalid)
}

/// Offline SDK mode of the same engine and domain implementations. This starts
/// no window, device, interpreter or background host process.
pub fn offline()->Result<(),Box<dyn std::error::Error>>{
    use std::io::{Read as _,Write as _};
    use serde_json::json;
    const MAX:usize=256<<20;
    let(mut input,mut output)=(std::io::stdin().lock(),std::io::stdout().lock());
    let mut scopes=std::collections::HashMap::<String,Service>::new();
    loop{
        let mut head=[0u8;4];match input.read_exact(&mut head){Ok(())=>{},Err(e)if e.kind()==std::io::ErrorKind::UnexpectedEof=>return Ok(()),Err(e)=>return Err(e.into())}
        let size=u32::from_le_bytes(head)as usize;if size>MAX{return Err("service request is too large".into());}
        let mut bytes=vec![0;size];input.read_exact(&mut bytes)?;
        let result=(||->Result<Value,String>{
            let request:Value=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
            let operation=request["operation"].as_str().ok_or("operation is required")?;
            let scope=request["scope"].as_str().unwrap_or("");
            if operation=="scope.release"{scopes.remove(scope);return Ok(Value::Null);}
            scopes.entry(scope.into()).or_default().dispatch(operation,&request["params"],
                &zlc_workbench::services::arrays::decode,&zlc_workbench::services::arrays::encode)
        })();
        let response=match result{Ok(value)=>json!({"ok":value}),Err(error)=>json!({"error":{"kind":"invalid","message":error}})};
        let bytes=serde_json::to_vec(&response)?;if bytes.len()>MAX{return Err("service response is too large".into());}
        output.write_all(&(bytes.len()as u32).to_le_bytes())?;output.write_all(&bytes)?;output.flush()?;
    }
}
