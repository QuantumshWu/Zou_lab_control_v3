//! Application services use the host's existing run, input and array ownership.
//! The host does not select an application implementation or know its operations.
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::sync::atomic::Ordering;
use std::time::Duration;
use serde_json::{Value,json};
use zlc_core::AnyBuffer;
use zlc_core::runtime::{Bundle,Generation,GenerationId,Read};
use zlc_core::schema::DatasetSchema;
use crate::{CallError,ErrorKind,Inner,PluginHost,blocks::Lend,process::PluginProcess,wire};

pub type Resource = Arc<dyn Any+Send+Sync>;
pub(crate) type Resources = HashMap<String,Resource>;
pub type Handler = Arc<dyn Fn(&str,&Value,&Context)->Result<Value,CallError>+Send+Sync>;

#[derive(Clone)]
pub struct RunContext { inner:Weak<Inner>, instance:String, run:String }

impl RunContext {
    fn host(&self)->Result<Arc<Inner>,CallError>{self.inner.upgrade().ok_or_else(||CallError::lost("the host ended"))}
    fn params(&self)->Value{json!({"instance":self.instance,"run":self.run})}
    pub fn id(&self)->&str{&self.run}
    pub fn check_active(&self)->Result<(),CallError>{
        let host=self.host()?;let mut state=host.state.lock().unwrap();
        Inner::run_named(&mut state,&self.instance,&self.run).map(|_|())
    }
    pub fn check_running(&self)->Result<(),CallError>{
        let host=self.host()?;let mut state=host.state.lock().unwrap();
        let run=Inner::run_named(&mut state,&self.instance,&self.run)?;
        crate::native_input::check_stop(&run.stop.0)
    }
    pub fn check_computation(&self)->Result<(),CallError>{
        let host=self.host()?;let mut state=host.state.lock().unwrap();
        let run=Inner::run_named(&mut state,&self.instance,&self.run)?;
        if run.lifecycle.allows_terminal_work(){Ok(())}else{crate::native_input::check_stop(&run.stop.0)}
    }
    pub fn resource(&self,key:&str)->Result<Option<Resource>,CallError>{
        let host=self.host()?;let mut state=host.state.lock().unwrap();
        Ok(Inner::run_named(&mut state,&self.instance,&self.run)?.resources.get(key).cloned())
    }
    pub fn install_resource(&self,key:&str,value:Resource,unpublished:bool)->Result<Resource,CallError>{
        let host=self.host()?;let mut state=host.state.lock().unwrap();
        let run=Inner::run_named(&mut state,&self.instance,&self.run)?;
        if unpublished && run.generation.is_some(){return Err(CallError::new(ErrorKind::Refused,"the run has already declared its outputs"));}
        Ok(run.resources.entry(key.to_owned()).or_insert(value).clone())
    }
    fn input(&self)->Result<(Arc<Mutex<crate::native_input::CoreInput>>,Arc<std::sync::atomic::AtomicBool>),CallError>{
        let host=self.host()?;let mut state=host.state.lock().unwrap();
        let run=Inner::run_named(&mut state,&self.instance,&self.run)?;
        Ok((run.core_input.clone().ok_or_else(||CallError::new(ErrorKind::Invalid,"this run has no core-consumed exact input"))?,run.stop.0.clone()))
    }
    pub fn input_next(&self,timeout:Duration,terminal_errors:bool)->Result<Option<Read>,CallError>{
        let(input,stop)=self.input()?;let answer=input.lock().unwrap().next(timeout,&stop,terminal_errors)?;
        self.check_running()?;Ok(answer)
    }
    pub fn input_drain(&self,quiet:Duration)->Result<usize,CallError>{
        let(input,stop)=self.input()?;let answer=input.lock().unwrap().drain(quiet,&stop);answer
    }
    pub fn input_await_new_source(&self)->Result<(),CallError>{
        self.check_running()?;self.input()?.0.lock().unwrap().await_new_source();Ok(())
    }
    pub fn input_generation(&self)->Result<Option<GenerationId>,CallError>{
        let input=self.input()?.0;let result=input.lock().unwrap().generation();Ok(result)
    }
    /// Bind this exact-input subscription and begin one finite output.
    pub fn begin_consuming_finite_output(&self,name:&str,schema:Arc<DatasetSchema>,record:Value)->Result<(Generation,Arc<str>),CallError>{
        self.check_running()?;let host=self.host()?;
        let ticket=self.input()?.0.lock().unwrap().ticket();
        {let mut state=host.state.lock().unwrap();Inner::run_named(&mut state,&self.instance,&self.run)?.consumer=ticket;}
        let mut params=self.params();params["record"]=record;
        host.begin_outputs(&params,vec![(name.to_owned(),schema,true)])?;
        let mut state=host.state.lock().unwrap();let run=Inner::run_named(&mut state,&self.instance,&self.run)?;
        Ok((run.generation.clone().ok_or_else(||CallError::new(ErrorKind::Failed,"the output did not begin"))?,run.outputs[name].signal.clone()))
    }
    pub fn commit(&self,generation:&Generation,bundle:&Bundle)->Result<(),CallError>{
        let host=self.host()?;
        {let mut state=host.state.lock().unwrap();let run=Inner::run_named(&mut state,&self.instance,&self.run)?;
            if run.generation.as_ref().is_none_or(|own|own.id()!=generation.id()){
                return Err(CallError::new(ErrorKind::StaleRun,"the publication does not belong to this run"));
            }
        }
        host.runtime.commit(generation,bundle).map(|_|()).map_err(|error|CallError::new(ErrorKind::Failed,error.to_string()))
    }
}

/// A single service call's transport resources. Outputs are released on any
/// failed call; successful replies transfer their holds to the plugin.
pub struct Context {
    inner:Arc<Inner>,process:Arc<PluginProcess>,pid:u32,run:Option<RunContext>,lends:Vec<Arc<Lend>>,exports:Mutex<Vec<String>>,
}
impl Context {
    fn check_process(&self)->Result<(),CallError>{
        if self.process.alive.load(Ordering::Acquire){Ok(())}else{Err(CallError::lost("the plugin process ended"))}
    }
    pub fn run(&self)->Result<&RunContext,CallError>{self.run.as_ref().ok_or_else(||CallError::new(ErrorKind::Invalid,"this operation requires an active run"))}
    pub fn optional_run(&self)->Option<&RunContext>{self.run.as_ref()}
    pub fn read_array(&self,value:&Value)->Result<wire::Array,CallError>{wire::array(value,&self.lends).map_err(|e|CallError::new(ErrorKind::Invalid,e))}
    pub fn write_array(&self,values:&AnyBuffer,shape:&[usize])->Result<Value,CallError>{
        let value=wire::encode_array(values,shape,&self.inner.registry,self.pid).map_err(|e|CallError::new(ErrorKind::Invalid,e))?;
        if let Some(name)=value["$array"]["block"].as_str(){self.exports.lock().unwrap().push(name.to_owned());}Ok(value)
    }
    pub fn resource(&self,key:&str)->Result<Option<Resource>,CallError>{
        if let Some(run)=&self.run{return run.resource(key);}
        let resources=self.inner.process_resources.lock().unwrap();
        self.check_process()?;
        Ok(resources.get(&self.pid).and_then(|values|values.get(key)).cloned())
    }
    pub fn install_resource(&self,key:&str,value:Resource)->Result<Resource,CallError>{
        if let Some(run)=&self.run{return run.install_resource(key,value,false);}
        let mut resources=self.inner.process_resources.lock().unwrap();
        // Exit marks the process dead before removing this map. Check under
        // the same lock so a finishing worker cannot resurrect its resources.
        self.check_process()?;
        Ok(resources.entry(self.pid).or_default().entry(key.to_owned()).or_insert(value).clone())
    }
}
impl Drop for Context {fn drop(&mut self){self.inner.registry.released_by(self.pid,&self.exports.get_mut().unwrap());}}

impl PluginHost {
    /// The application installs domain services before any plugin is started.
    pub fn install_services(&self,handler:Handler)->Result<(),CallError>{
        if !self.inner.processes.lock().unwrap().is_empty(){return Err(CallError::new(ErrorKind::Refused,"install services before starting plugins"));}
        let mut held=self.inner.services.lock().unwrap();
        if held.is_some(){return Err(CallError::new(ErrorKind::Refused,"application services are already installed"));}
        *held=Some(handler);Ok(())
    }
    pub fn run_context(&self,id:&str)->Result<RunContext,CallError>{
        let state=self.inner.state.lock().unwrap();
        let instance=state.instances.iter().find(|(_,entry)|entry.run.as_ref().is_some_and(|run|run.id==id))
            .map(|(instance,_)|instance.clone()).ok_or_else(||CallError::new(ErrorKind::StaleRun,"the run has ended"))?;
        Ok(RunContext{inner:Arc::downgrade(&self.inner),instance,run:id.to_owned()})
    }
}

impl Inner {
    pub(crate) fn service_request(self:&Arc<Self>,process:&Arc<PluginProcess>,params:&Value,lends:Vec<Arc<Lend>>)->Result<(Value,Vec<String>),CallError>{
        let operation=params["operation"].as_str().ok_or_else(||CallError::new(ErrorKind::Invalid,"a service request needs an operation"))?;
        let handler=self.services.lock().unwrap().clone().ok_or_else(||CallError::new(ErrorKind::UnknownOp,"the application has not installed services"))?;
        let run=if let Some(id)=params["run"].as_str(){
            let instance=params["instance"].as_str().unwrap_or("");let mut state=self.state.lock().unwrap();
            if state.instances.get(instance).is_none_or(|entry|entry.serving.pid()!=process.pid()){
                return Err(CallError::new(ErrorKind::StaleRun,"the run belongs to another plugin process"));
            }
            Self::run_of(&mut state,params)?;
            Some(RunContext{inner:Arc::downgrade(self),instance:instance.to_owned(),run:id.to_owned()})
        }else{None};
        let context=Context{inner:self.clone(),process:process.clone(),pid:process.pid(),run,lends,exports:Mutex::default()};
        context.check_process()?;
        let answer=handler(operation,&params["params"],&context)?;
        context.check_process()?;
        if let Some(run)=&context.run{run.check_active()?;}
        let exports=std::mem::take(&mut *context.exports.lock().unwrap());Ok((answer,exports))
    }
}
