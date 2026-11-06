//! Process-local acquisition policy. No SDK handles or borrowed image buffers cross this API.
use std::ffi::{CStr, CString, c_char};
use std::sync::Mutex;
use serde_json::{Value, json};

type Answer = Result<Value, (&'static str, String)>;
fn fail(kind: &'static str, message: impl Into<String>) -> (&'static str, String) { (kind, message.into()) }
fn integer(p: &Value, name: &str) -> Result<u64, (&'static str, String)> {
    p[name].as_u64().ok_or_else(|| fail("invalid", format!("{name} must be a nonnegative integer")))
}
fn flag(p: &Value, name: &str) -> bool { p[name].as_bool().unwrap_or(false) }

#[derive(Default)]
pub struct Ledger {
    armed: bool, accepting: bool, ready: bool,
    expected: Option<u64>, capacity: u64, produced: u64, pending: u64,
    failure: Option<String>, next_record: u64, group_pending: u64,
}

impl Ledger {
    fn failed(&self) -> Result<(), (&'static str, String)> {
        self.failure.as_ref().map_or(Ok(()), |error| Err(fail("failed", format!("failed while producing records: {error}"))))
    }
    fn record_failure(&mut self, message: String) {
        self.failure.get_or_insert(message);
        self.accepting = false; self.ready = true;
    }
    pub fn call(&mut self, op: &str, p: &Value) -> Answer {
        match op {
            "state" => Ok(json!({"armed": self.armed, "accepting": self.accepting, "ready": self.ready,
                "produced": self.produced, "pending": self.pending, "capacity": self.capacity, "failure": self.failure})),
            "arm" => {
                if self.armed { return Err(fail("failed", "is already armed")); }
                if flag(p,"worker_alive") { return Err(fail("failed", "is still finishing its previous capture")); }
                let capacity=integer(p,"capacity")?;
                if capacity==0 { return Err(fail("invalid", "buffer_record_count must be positive")); }
                let expected=if p["expected"].is_null() {None} else {Some(integer(p,"expected")?)};
                if expected==Some(0) {return Err(fail("invalid", "finite records must be positive"));}
                *self=Self {armed:true,accepting:true,capacity,expected,..Self::default()};
                Ok(json!({}))
            }
            "ready" => {self.ready=true;Ok(json!({}))}
            "check_ready" => {
                self.failed()?;
                if !self.armed {return Err(fail("failed","stopped before becoming ready"));}
                Ok(json!({"ready":self.ready}))
            }
            "push" => {
                if !self.accepting {return Ok(json!({"accepted":false}));}
                let ordinal=p["ordinal"].as_u64();
                if ordinal!=Some(self.produced) {
                    self.record_failure(format!("record ordinal {} is not the expected {} (observed_at_ns={})",p["ordinal"],self.produced,p["observed_at_ns"]));
                    return Ok(json!({"accepted":false}));
                }
                if self.pending>=self.capacity {
                    self.record_failure(format!("capture buffer overflow at record {} (capacity {}, observed_at_ns={}); no queued record was discarded",self.produced,self.capacity,p["observed_at_ns"]));
                    return Ok(json!({"accepted":false}));
                }
                self.produced=self.produced.checked_add(1).ok_or_else(||fail("failed","capture record count overflow"))?;
                self.pending+=1;
                if self.expected.is_some_and(|n|self.produced>=n) {self.accepting=false;}
                Ok(json!({"accepted":true}))
            }
            "fail" => {self.record_failure(p["message"].as_str().unwrap_or("producer failed").to_owned());Ok(json!({}))}
            "read" => {
                self.failed()?;
                let requested=integer(p,"count")?;
                if requested==0 {return Err(fail("invalid","n must be positive"));}
                let exact=flag(p,"exact");let wanted=if exact {requested}else{1};
                if self.pending<wanted && self.armed && self.accepting && !flag(p,"timed_out") {return Ok(json!({"wait":true}));}
                if exact && self.pending<requested {return Err(fail("timeout","did not deliver the requested exact records"));}
                let count=if exact {requested}else{requested.min(self.pending)};
                self.pending-=count;
                Ok(json!({"wait":false,"count":count}))
            }
            "begin_finish" => {self.accepting=false;self.ready=true;Ok(json!({"armed":self.armed,"produced":self.produced}))}
            "finish" => {
                if !flag(p,"joined") {return Err(fail("failed","did not stop producing within its join deadline"));}
                self.armed=false;self.accepting=false;self.ready=true;
                self.failed()?;Ok(json!({"produced":self.produced}))
            }
            "close" => {
                if flag(p,"worker_alive") {return Err(fail("failed","is still producing records"));}
                self.armed=false;self.accepting=false;self.ready=true;self.pending=0;self.failure=None;Ok(json!({}))
            }
            "sequence" => {
                let ordinals=p["ordinals"].as_array().ok_or_else(||fail("invalid","record ordinals must be an array"))?;
                let mut next=self.next_record;
                for ordinal in ordinals {
                    if ordinal.as_u64()!=Some(next) {return Err(fail("failed",format!("record sequence gap: expected {next}, received {ordinal}")));}
                    next=next.checked_add(1).ok_or_else(||fail("failed","record sequence overflow"))?;
                }
                self.next_record=next;Ok(json!({"next":next}))
            }
            "append_group" => {
                let size=integer(p,"size")?;
                if size==0 {return Err(fail("invalid","capture group size must be positive"));}
                self.group_pending=self.group_pending.checked_add(1).ok_or_else(||fail("failed","capture group count overflow"))?;
                if self.group_pending>size {return Err(fail("failed","capture group size changed during a cycle"));}
                let complete=self.group_pending==size;
                if complete {self.group_pending=0;}
                Ok(json!({"complete":complete}))
            }
            "reset_group"=>{self.group_pending=0;Ok(json!({}))}
            "group" => {
                let start=integer(p,"start")?;let size=integer(p,"size")?;
                if size==0 {return Err(fail("invalid","capture group size must be positive"));}
                let ordinals=p["ordinals"].as_array().ok_or_else(||fail("invalid","record ordinals must be an array"))?;
                if ordinals.len() as u64!=size || ordinals.iter().enumerate().any(|(i,x)|start.checked_add(i as u64)!=x.as_u64()) {
                    return Err(fail("failed",format!("cycle source ordinals are not contiguous: expected {size} from {start}, received {ordinals:?}")));
                }
                Ok(json!({}))
            }
            "terminal" => {
                if !flag(p,"source_stopped") || !flag(p,"joined") {return Err(fail("failed","terminal evidence is incomplete: the capture did not stop and join"));}
                let produced=integer(p,"produced")?;
                if !p["expected"].is_null() {
                    let expected=integer(p,"expected")?;
                    if produced<expected || (produced!=expected && !flag(p,"stopped")) {
                        return Err(fail("failed",format!("terminal count differs from completed cycles: expected {expected}, produced {produced} (a partial cycle may be present)")));
                    }
                }
                let size=p["group_size"].as_u64().unwrap_or(1);
                if size==0 {return Err(fail("invalid","capture group size must be positive"));}
                Ok(json!({"complete_groups":produced/size,"partial_records":produced%size,"produced":produced}))
            }
            _=>Err(fail("invalid",format!("unknown capture operation {op}"))),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn zlc_capture_new() -> *mut Mutex<Ledger> {Box::into_raw(Box::new(Mutex::new(Ledger::default())))}

/// # Safety
/// `handle` must be a live handle from zlc_capture_new, with no simultaneous free.
/// `request` must point to a NUL-terminated UTF-8 JSON request.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_capture_call(handle:*mut Mutex<Ledger>,request:*const c_char)->*mut c_char {
    let outcome=std::panic::catch_unwind(|| {
        if handle.is_null() || request.is_null() {return Err(fail("invalid","null capture ABI argument"));}
        let text=unsafe{CStr::from_ptr(request)}.to_str().map_err(|e|fail("invalid",e.to_string()))?;
        let value:Value=serde_json::from_str(text).map_err(|e|fail("invalid",e.to_string()))?;
        unsafe{&*handle}.lock().map_err(|_|fail("failed","capture ledger lock is poisoned"))?
            .call(value["op"].as_str().unwrap_or(""),&value)
    }).unwrap_or_else(|_|Err(fail("failed","capture native call panicked")));
    let answer=match outcome {Ok(value)=>json!({"ok":value}),Err((kind,message))=>json!({"error":{"kind":kind,"message":message}})};
    CString::new(answer.to_string()).unwrap().into_raw()
}
/// # Safety
/// `value` must be an unreleased result of zlc_capture_call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_capture_free_result(value:*mut c_char) {if !value.is_null(){drop(unsafe{CString::from_raw(value)});}}
/// # Safety
/// `handle` must be an unreleased handle with no in-flight calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zlc_capture_free(handle:*mut Mutex<Ledger>) {if !handle.is_null(){drop(unsafe{Box::from_raw(handle)});}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn bounded_queue_keeps_accepted_records_and_latches_first_failure(){
        let mut q=Ledger::default();q.call("arm",&json!({"capacity":1,"expected":null})).unwrap();
        assert_eq!(q.call("push",&json!({"ordinal":0})).unwrap()["accepted"],true);
        assert_eq!(q.call("push",&json!({"ordinal":1})).unwrap()["accepted"],false);
        assert!(q.call("read",&json!({"count":1})).is_err());assert_eq!(q.pending,1);assert_eq!(q.produced,1);
        q.call("fail",&json!({"message":"later"})).unwrap();assert!(q.failure.as_ref().unwrap().contains("overflow"));
    }
    #[test] fn exact_reads_wait_then_timeout_without_losing_records(){
        let mut q=Ledger::default();q.call("arm",&json!({"capacity":3,"expected":null})).unwrap();
        q.call("push",&json!({"ordinal":0})).unwrap();
        assert_eq!(q.call("read",&json!({"count":2,"exact":true})).unwrap()["wait"],true);
        assert_eq!(q.call("read",&json!({"count":2,"exact":true,"timed_out":true})).unwrap_err().0,"timeout");
        assert_eq!(q.call("read",&json!({"count":3,"exact":false})).unwrap()["count"],1);
    }
    #[test] fn finite_stop_join_and_rearm_are_distinct(){
        let mut q=Ledger::default();q.call("arm",&json!({"capacity":2,"expected":1})).unwrap();
        q.call("push",&json!({"ordinal":0})).unwrap();assert!(!q.accepting);assert!(q.armed);
        q.call("begin_finish",&json!({})).unwrap();assert!(q.call("finish",&json!({"joined":false})).is_err());assert!(q.armed);
        q.call("finish",&json!({"joined":true})).unwrap();assert_eq!(q.pending,1);
        assert_eq!(q.call("read",&json!({"count":1,"exact":true})).unwrap()["count"],1);
        q.call("arm",&json!({"capacity":2,"expected":null})).unwrap();assert_eq!(q.produced,0);
    }
    #[test] fn group_and_terminal_evidence_reject_gaps_and_incomplete_hardware_stop(){
        let mut q=Ledger::default();assert!(q.call("group",&json!({"start":4,"size":2,"ordinals":[4,6]})).is_err());
        q.call("group",&json!({"start":4,"size":2,"ordinals":[4,5]})).unwrap();
        assert!(q.call("terminal",&json!({"source_stopped":true,"joined":false,"produced":5})).is_err());
        assert!(q.call("terminal",&json!({"source_stopped":true,"joined":true,"produced":5,"expected":4})).is_err());
        let end=q.call("terminal",&json!({"source_stopped":true,"joined":true,"produced":5,"expected":4,"stopped":true,"group_size":2})).unwrap();
        assert_eq!(end["complete_groups"],2);assert_eq!(end["partial_records"],1);
    }
}
