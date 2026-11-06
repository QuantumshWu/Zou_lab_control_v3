//! Core authority for the race between Stop, protected terminal work, and finish.
//! SDK cleanup executes on its run thread; it receives the outcome decided here.
use serde_json::{Value, json};
use crate::{CallError, ErrorKind, HostEvent, Inner};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Intent { Success, Error, Invalid }

#[derive(Debug)]
struct Terminal {
    token: String,
    explicit: bool,
    stop_accepted: bool,
    partial: bool,
    outcome: Option<&'static str>,
    exit_writer: bool,
}

#[derive(Debug, Default)]
pub(crate) struct RunLifecycle {
    stopped: bool,
    terminal: Option<Terminal>,
    finished: bool,
}

impl RunLifecycle {
    /// Caller holds the run registry lock and sets the shared cancellation flag
    /// only if this returns true. A terminal commit wins over every later Stop.
    pub fn allows_terminal_work(&self) -> bool { self.terminal.is_some() && !self.finished }

    pub fn request_stop(&mut self) -> bool {
        if self.terminal.is_some() || self.finished { return false; }
        self.stopped = true;
        true
    }

    pub fn begin(&mut self, accept_stop: bool, partial: bool, finalizing: bool, intent: Intent) -> Result<Value, CallError> {
        if self.finished { return Err(CallError::new(ErrorKind::StaleRun, "the run already finished")); }
        if self.terminal.is_none() {
            if self.stopped && !accept_stop {
                return Err(CallError::new(ErrorKind::Cancelled, "the run was stopped before its terminal commit"));
            }
            self.terminal = Some(Terminal { token: uuid::Uuid::new_v4().to_string(), explicit: !finalizing,
                stop_accepted: self.stopped && accept_stop, partial, outcome: None, exit_writer: false });
        }
        let terminal = self.terminal.as_mut().unwrap();
        if finalizing && terminal.outcome.is_none() {
            // An explicit seal protects its subsequent work. A failure there
            // remains a failure, even if a stopped prefix was accepted earlier.
            let stopped = terminal.stop_accepted && !terminal.explicit;
            let outcome = match intent {
                Intent::Invalid => "failed",
                Intent::Error if !stopped => "failed",
                _ if terminal.stop_accepted => "stopped",
                _ => "completed",
            };
            terminal.outcome = Some(outcome);
            terminal.exit_writer = intent == Intent::Error || (outcome == "stopped" && !terminal.explicit);
        }
        Ok(json!({"token": terminal.token, "stop_accepted": terminal.stop_accepted,
            "partial": terminal.partial, "outcome": terminal.outcome, "run_exit_writer": terminal.exit_writer}))
    }

    pub fn finish(&mut self, token: &str) -> Result<&'static str, CallError> {
        if self.finished { return Err(CallError::new(ErrorKind::StaleRun, "the run already finished")); }
        let terminal = self.terminal.as_ref().filter(|terminal| terminal.token == token)
            .ok_or_else(|| CallError::new(ErrorKind::StaleRun, "the terminal token does not belong to this run"))?;
        let outcome = terminal.outcome.ok_or_else(|| CallError::new(ErrorKind::Invalid, "finalize terminal cleanup before finishing the run"))?;
        self.finished = true;
        Ok(outcome)
    }
}

impl Inner {
    pub(crate) fn lifecycle_request(&self, pid: u32, op: &str, params: &Value) -> Result<Value, CallError> {
        let instance = params["instance"].as_str().unwrap_or("").to_owned();
        let id = params["run"].as_str().unwrap_or("").to_owned();
        let mut state = self.state.lock().unwrap();
        if state.instances.get(&instance).is_none_or(|entry| entry.serving.pid() != pid) {
            return Err(CallError::new(ErrorKind::StaleRun, "the run does not belong to this plugin process"));
        }
        let run = Self::run_of(&mut state, params)?;
        if op == "run.begin_terminal" {
            let flag = |name: &str| -> Result<bool, CallError> {
                params.get(name).map_or(Ok(false), |value| value.as_bool().ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{name} must be boolean"))))
            };
            let intent = match params["intent"].as_str().unwrap_or("success") {
                "success" => Intent::Success, "error" => Intent::Error, "invalid" => Intent::Invalid,
                _ => return Err(CallError::new(ErrorKind::Invalid, "terminal intent must be success, error or invalid")),
            };
            return run.lifecycle.begin(flag("accept_stop")?, flag("partial")?, flag("finalizing")?, intent);
        }
        if op != "run.finish" { return Err(CallError::new(ErrorKind::UnknownOp, "unknown run lifecycle operation")); }
        let token = params["token"].as_str().ok_or_else(|| CallError::new(ErrorKind::Invalid, "run.finish needs its terminal token"))?;
        let outcome = run.lifecycle.finish(token)?;
        drop(state);
        let result = params["result"].clone();
        let error = params.get("error").filter(|error| !error.is_null()).cloned();
        let generation = self.end_run(&instance, &id, outcome, &json!({"result": result, "error": error}));
        (self.events)(HostEvent::RunFinished { instance, run: id, outcome: outcome.to_owned(), result, error, generation });
        Ok(json!({"outcome": outcome}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_and_terminal_have_one_order_and_finish_needs_its_token() {
        let mut run = RunLifecycle::default();
        assert!(run.request_stop());
        assert_eq!(run.begin(false, false, false, Intent::Success).unwrap_err().kind, ErrorKind::Cancelled);
        let seal = run.begin(true, true, false, Intent::Success).unwrap();
        assert_eq!(seal["stop_accepted"], true);
        assert!(!run.request_stop());
        assert!(run.finish(seal["token"].as_str().unwrap()).is_err());
        let finalizing = run.begin(true, false, true, Intent::Success).unwrap();
        assert_eq!(finalizing["token"], seal["token"]);
        assert_eq!(finalizing["partial"], true);
        assert_eq!(finalizing["outcome"], "stopped");
        assert_eq!(finalizing["run_exit_writer"], false);
        assert!(run.finish("another-run").is_err());
        assert_eq!(run.finish(seal["token"].as_str().unwrap()).unwrap(), "stopped");
        assert!(run.finish(seal["token"].as_str().unwrap()).is_err());
    }

    #[test]
    fn explicit_seal_ignores_late_stop_and_keeps_terminal_failures() {
        let mut run = RunLifecycle::default();
        run.begin(false, false, false, Intent::Success).unwrap();
        assert!(!run.request_stop());
        let ending = run.begin(true, false, true, Intent::Error).unwrap();
        assert_eq!(ending["outcome"], "failed");
        assert_eq!(ending["run_exit_writer"], true);
    }

    #[test]
    fn ordinary_cancelled_work_stays_stopped_but_invalid_input_stays_failed() {
        for (intent, outcome, writer) in [(Intent::Success, "stopped", true), (Intent::Error, "stopped", true), (Intent::Invalid, "failed", false)] {
            let mut run = RunLifecycle::default();
            run.request_stop();
            let ending = run.begin(true, false, true, intent).unwrap();
            assert_eq!(ending["outcome"], outcome);
            assert_eq!(ending["run_exit_writer"], writer);
        }
    }

    #[test]
    fn concurrent_stop_and_seal_agree_with_shared_registry_order() {
        use std::sync::{Arc, Barrier, Mutex};
        for _ in 0..64 {
            let run = Arc::new(Mutex::new(RunLifecycle::default()));
            let gate = Arc::new(Barrier::new(2));
            let (other, ready) = (run.clone(), gate.clone());
            let worker = std::thread::spawn(move || { ready.wait(); other.lock().unwrap().request_stop() });
            gate.wait();
            let sealed = run.lock().unwrap().begin(false, false, false, Intent::Success);
            let stopped = worker.join().unwrap();
            assert_eq!(sealed.is_err(), stopped);
        }
    }
}
