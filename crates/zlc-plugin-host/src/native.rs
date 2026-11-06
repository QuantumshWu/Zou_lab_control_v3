//! Native rule/kernel calls share one implementation with offline tools. Pure
//! authoring is available before an instance exists; mutable scientific handles
//! are owned by a run whenever the caller is executing a run.
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use serde_json::Value;
use crate::{CallError, ErrorKind, Inner, blocks::Lend, process::PluginProcess, wire};

impl Inner {
    pub(crate) fn native_request(&self, process: &Arc<PluginProcess>, params: &Value, lends: &[Arc<Lend>]) -> Result<(Value, Vec<String>), CallError> {
        let invalid = |message| CallError::new(ErrorKind::Invalid, message);
        let operation = params["operation"].as_str().ok_or_else(|| invalid("native request needs an operation".into()))?;
        if operation.starts_with("durable.") {
            return Err(CallError::new(ErrorKind::Refused, "run files use the run-scoped artifact service"));
        }
        let (service, stop, terminal) = if params["run"].is_string() {
            let mut state = self.state.lock().unwrap();
            let instance = params["instance"].as_str().unwrap_or("");
            if state.instances.get(instance).is_none_or(|entry| entry.serving.pid() != process.pid()) {
                return Err(CallError::new(ErrorKind::StaleRun, "native service belongs to another plugin process"));
            }
            let run = Self::run_of(&mut state, params)?;
            (run.native.clone(), Some(run.stop.0.clone()), run.lifecycle.allows_terminal_work())
        } else {
            (self.native_services.lock().unwrap().entry(process.pid())
                .or_insert_with(|| Arc::new(Mutex::new(zlc_native::Service::default()))).clone(), None, false)
        };
        let check = || -> Result<(), String> {
            if stop.is_some() && !self.current_run(params["instance"].as_str().unwrap_or(""), params["run"].as_str().unwrap_or("")) {
                return Err("the run has ended".into());
            }
            if !terminal && stop.as_ref().is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Acquire)) { Err("the run was stopped".into()) } else { Ok(()) }
        };
        check().map_err(|message| CallError::new(ErrorKind::Cancelled, message))?;
        let mut service = service.lock().unwrap();
        let blocks = RefCell::new(Vec::<String>::new());
        let read = |value: &Value| {
            check()?;
            let array = wire::array(value, lends)?;
            Ok(zlc_native::Tensor { values: array.values, shape: array.shape })
        };
        let write = |tensor: zlc_native::Tensor| {
            check()?;
            let value = wire::encode_array(&tensor.values, &tensor.shape, &self.registry, process.pid())?;
            if let Some(name) = value["$array"]["block"].as_str() { blocks.borrow_mut().push(name.to_owned()); }
            Ok(value)
        };
        let answer = service.dispatch(operation, &params["params"], &read, &write);
        let blocks = blocks.into_inner();
        if let Err(error) = check() {
            self.registry.released_by(process.pid(), &blocks);
            return Err(CallError::new(ErrorKind::Cancelled, error));
        }
        match answer {
            Ok(value) => Ok((value, blocks)),
            Err(error) => { self.registry.released_by(process.pid(), &blocks); Err(invalid(error)) }
        }
    }
}
