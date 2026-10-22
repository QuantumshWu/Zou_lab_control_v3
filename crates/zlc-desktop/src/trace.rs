//! Opt-in bounded timing evidence. No scientific payloads or hot-path file I/O.
use std::{path::PathBuf, sync::{Mutex,OnceLock}, time::{Instant,SystemTime,UNIX_EPOCH}};
use serde_json::{Value,json};
use zlc_core::runtime::PublicationKey;

struct Recorder { path:PathBuf, events:Vec<Value>, dropped:usize }
static TRACE:OnceLock<Option<Mutex<Recorder>>> = OnceLock::new();
fn recorder() -> Option<&'static Mutex<Recorder>> {
    TRACE.get_or_init(||std::env::var_os("ZLC_V3_TRACE").map(|path|Mutex::new(Recorder {
        path:path.into(),events:Vec::new(),dropped:0}))).as_ref()
}
fn epoch_ns() -> u128 {SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()}
pub fn key(key:&PublicationKey) -> Value {
    json!({"generation":key.generation.as_uuid().to_string(),"signal":key.signal.as_ref(),"revision":key.revision,"ordinal":key.ordinal})
}
fn record(stage:&str,start:u128,duration:u128,facts:Value) {
    if let Some(recorder)=recorder() {
        let mut recorder=recorder.lock().unwrap_or_else(|error|error.into_inner());
        if recorder.events.len()>=100_000 {recorder.dropped+=1;return;}
        recorder.events.push(json!({"stage":stage,"start_ns":start,"duration_ns":duration,
            "thread":std::thread::current().name().unwrap_or("native-ui"),"facts":facts}));
    }
}
pub fn point(stage:&str,facts:impl FnOnce()->Value) {
    if recorder().is_some() {record(stage,epoch_ns(),0,facts());}
}
pub struct Span {stage:&'static str,start:u128,clock:Instant,facts:Value}
pub fn span(stage:&'static str,facts:impl FnOnce()->Value)->Option<Span> {
    recorder().map(|_|Span {stage,start:epoch_ns(),clock:Instant::now(),facts:facts()})
}
impl Drop for Span {fn drop(&mut self) {record(self.stage,self.start,self.clock.elapsed().as_nanos(),std::mem::take(&mut self.facts));}}
/// When this process was created (epoch ns): what came before main is part of a start.
pub fn process_created_ns() -> u128 {
    use windows_sys::Win32::{Foundation::FILETIME, System::Threading::{GetCurrentProcess, GetProcessTimes}};
    let mut times = [FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    let [created, exited, kernel, user] = &mut times;
    if unsafe { GetProcessTimes(GetCurrentProcess(), created, exited, kernel, user) } == 0 { return 0; }
    // 100 ns ticks since 1601 to ns since 1970.
    let ticks = (u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime);
    u128::from(ticks.saturating_sub(116_444_736_000_000_000)) * 100
}

pub fn flush() {
    let Some(recorder)=recorder() else{return;};
    let recorder=recorder.lock().unwrap_or_else(|error|error.into_inner());
    let result=(||->Result<(),Box<dyn std::error::Error>> {
        if let Some(parent)=recorder.path.parent() {std::fs::create_dir_all(parent)?;}
        // ZLC_RENDER_PROFILE adds the engine's phase totals: (name, ms, count).
        let phases=zlc_plot::render::profile::report();
        std::fs::write(&recorder.path,serde_json::to_vec(&json!({"events":recorder.events,"dropped":recorder.dropped,"phases":phases}))?)?;
        Ok(())
    })();
    if let Err(error)=result {eprintln!("Could not save native timing evidence: {error}");}
}
