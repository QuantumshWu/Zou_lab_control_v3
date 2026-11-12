//! One bounded wire writer per plugin. Expired unsent frames never reach an SDK.
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Condvar, Mutex, atomic::{AtomicBool, AtomicU8, Ordering}, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde_json::Value;
use crate::{CallError, ErrorKind, frame};

const QUEUED:u8=0;const WRITING:u8=1;const CANCELLED:u8=2;
const NORMAL_LIMIT:usize=64;const CONTROL_LIMIT:usize=64;
/// How long a frame may wait for the writer before it is given up.
const SEND_WITHIN:Duration=Duration::from_secs(10);
/// Who learns how a frame went: a sender waiting for it, or work the writer does itself
/// (a posted frame's caller has gone on; its failure is handled where it is known).
enum Done {Wait(mpsc::Sender<Result<(),CallError>>),Then(Box<dyn FnOnce(Result<(),CallError>)+Send>)}
impl Done {
    fn tell(self,result:Result<(),CallError>){match self{Done::Wait(reply)=>{let _=reply.send(result);}Done::Then(then)=>then(result)}}
}
struct Job {message:Value,until:Instant,state:Arc<AtomicU8>,done:Done}
#[derive(Default)]
struct Queue {normal:VecDeque<Job>,control:VecDeque<Job>,closed:bool}
#[derive(Default)]
struct Shared {queue:Mutex<Queue>,ready:Condvar,writing:AtomicBool,#[cfg(test)] gate:Mutex<()>}
pub(crate) struct Outgoing {shared:Arc<Shared>}
pub(crate) struct SendError {pub error:CallError,pub started:bool}

/// QPC is the cross-process clock behind Python time.monotonic_ns on Windows.
/// Instant is opaque, so translate its remaining budget once at admission.
#[cfg(windows)]
fn shared_monotonic_ns()->Option<u64>{
    #[link(name="kernel32")]
    unsafe extern "system" {fn QueryPerformanceCounter(value:*mut i64)->i32;fn QueryPerformanceFrequency(value:*mut i64)->i32;}
    let(mut ticks,mut frequency)=(0i64,0i64);
    if unsafe{QueryPerformanceFrequency(&mut frequency)}==0 || unsafe{QueryPerformanceCounter(&mut ticks)}==0 || ticks<0 || frequency<=0{return None;}
    u64::try_from((ticks as u128)*1_000_000_000/(frequency as u128)).ok()
}
#[cfg(not(windows))]
fn shared_monotonic_ns()->Option<u64>{None}

fn stamp_deadline(message:&mut Value,until:Instant){
    if message.get("deadline_ms").is_none(){return;}
    if let Some(now)=shared_monotonic_ns(){
        let remaining=until.saturating_duration_since(Instant::now()).as_nanos();
        message["deadline_monotonic_ns"]=serde_json::json!((now as u128+remaining).min(u64::MAX as u128) as u64);
    }
    if let Ok(now)=SystemTime::now().duration_since(UNIX_EPOCH){
        let remaining=until.saturating_duration_since(Instant::now()).as_nanos();
        message["deadline_unix_ns"]=serde_json::json!((now.as_nanos()+remaining).min(u64::MAX as u128) as u64);
    }
}

impl Outgoing {
    pub fn new(mut sink:impl Write+Send+'static)->Arc<Self>{
        let shared=Arc::new(Shared::default());let worker=shared.clone();
        std::thread::Builder::new().name("zlc-plugin-writer".into()).spawn(move||loop{
            let job={let mut queue=worker.queue.lock().unwrap();loop{
                if queue.closed{return;}
                if let Some(job)=queue.control.pop_front().or_else(||queue.normal.pop_front()){break job;}
                queue=worker.ready.wait(queue).unwrap();
            }};
            #[cfg(test)]
            let _gate=loop {
                if Instant::now()>=job.until {break None;}
                match worker.gate.try_lock(){Ok(gate)=>break Some(gate),Err(_)=>std::thread::sleep(Duration::from_millis(1))}
            };
            if Instant::now()>=job.until || job.state.compare_exchange(QUEUED,WRITING,Ordering::AcqRel,Ordering::Acquire).is_err(){
                job.state.store(CANCELLED,Ordering::Release);
                job.done.tell(Err(CallError::new(ErrorKind::Timeout,"request expired before its wire frame was written")));continue;
            }
            let mut message=job.message;
            if message.get("deadline_ms").is_some(){
                message["deadline_ms"]=serde_json::json!(job.until.saturating_duration_since(Instant::now()).as_millis() as u64);
            }
            worker.writing.store(true,Ordering::Release);
            let answer=frame::write(&mut sink,&message).map_err(|error|CallError::lost(format!("cannot write to the plugin: {error}")));
            worker.writing.store(false,Ordering::Release);
            job.done.tell(answer);
        }).expect("start plugin wire writer");
        Arc::new(Self{shared})
    }
    /// Queue a frame for the writer: its state, or why it was not queued.
    fn admit(&self,message:&Value,until:Instant,done:Done)->Result<Arc<AtomicU8>,(SendError,Done)>{
        if Instant::now()>=until{return Err((SendError{error:CallError::new(ErrorKind::Timeout,"request expired before send"),started:false},done));}
        let mut message=message.clone();
        stamp_deadline(&mut message,until);
        let state=Arc::new(AtomicU8::new(QUEUED));
        let control=message["t"]!="req" || matches!(message["op"].as_str(),Some("ping"|"exit"|"close"|"make_safe"))
            || (message["op"]=="command" && message["p"]["name"]=="stop")
            || (message["op"]=="spi" && matches!(message["p"]["method"].as_str(),Some("stop"|"make_safe")));
        let mut queue=self.shared.queue.lock().unwrap();
        if queue.closed{return Err((SendError{error:CallError::lost("the plugin writer has closed"),started:false},done));}
        let jobs=if control{&mut queue.control}else{&mut queue.normal};
        if jobs.len()>=if control{CONTROL_LIMIT}else{NORMAL_LIMIT}{return Err((SendError{error:CallError::new(ErrorKind::Refused,"the plugin wire queue is full"),started:false},done));}
        jobs.push_back(Job{message,until,state:state.clone(),done});
        self.shared.ready.notify_one();
        Ok(state)
    }
    pub fn send_until(&self,message:&Value,until:Instant)->Result<(),SendError>{
        let(reply,received)=mpsc::channel();
        let state=self.admit(message,until,Done::Wait(reply)).map_err(|(error,_)|error)?;
        match received.recv_timeout(until.saturating_duration_since(Instant::now())){
            Ok(Ok(()))=>Ok(()),
            Ok(Err(error))=>Err(SendError{error,started:state.load(Ordering::Acquire)==WRITING}),
            Err(error)=>{
                let started=state.compare_exchange(QUEUED,CANCELLED,Ordering::AcqRel,Ordering::Acquire).is_err() && state.load(Ordering::Acquire)==WRITING;
                Err(SendError{error:match error{
                    mpsc::RecvTimeoutError::Timeout=>CallError::new(ErrorKind::Timeout,"request deadline expired while waiting for the plugin writer"),
                    mpsc::RecvTimeoutError::Disconnected=>CallError::lost("the plugin wire writer ended"),
                },started})
            }
        }
    }
    pub fn send(&self,message:&Value)->Result<(),CallError>{self.send_until(message,Instant::now()+SEND_WITHIN).map_err(|error|error.error)}
    /// Queue a frame and go on: `done` learns how it went, on the writer's thread (or at once,
    /// here, when it could not be queued). It must not wait for this writer.
    pub fn post(&self,message:&Value,done:impl FnOnce(Result<(),CallError>)+Send+'static){
        if let Err((error,done))=self.admit(message,Instant::now()+SEND_WITHIN,Done::Then(Box::new(done))){done.tell(Err(error.error));}
    }
    #[cfg(test)]
    pub fn pause(&self)->std::sync::MutexGuard<'_,()>{self.shared.gate.lock().unwrap()}
    #[cfg(test)]
    pub fn writing(&self)->bool{self.shared.writing.load(Ordering::Acquire)}
    pub fn close(&self){
        let jobs:Vec<Job>={
            let mut queue=self.shared.queue.lock().unwrap();queue.closed=true;
            let control:Vec<Job>=queue.control.drain(..).collect();
            control.into_iter().chain(queue.normal.drain(..).collect::<Vec<_>>()).collect()
        };
        self.shared.ready.notify_all();
        // Told outside the queue: what a posted frame does on failure may post again.
        for job in jobs{job.state.store(CANCELLED,Ordering::Release);job.done.tell(Err(CallError::lost("the plugin writer closed")));}
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    struct Blocked {entered:mpsc::Sender<()>,release:mpsc::Receiver<()>,bytes:Arc<Mutex<Vec<u8>>>,first:bool}
    impl Write for Blocked{
        fn write(&mut self,bytes:&[u8])->std::io::Result<usize>{if self.first{self.first=false;let _=self.entered.send(());let _=self.release.recv();}self.bytes.lock().unwrap().extend_from_slice(bytes);Ok(bytes.len())}
        fn flush(&mut self)->std::io::Result<()>{Ok(())}
    }
    #[test]fn blocked_writer_does_not_extend_caller_deadline_or_send_expired_queue(){
        let(entered,inside)=mpsc::channel();let(release,wait)=mpsc::channel();let bytes=Arc::new(Mutex::new(Vec::new()));
        let writer=Outgoing::new(Blocked{entered,release:wait,bytes:bytes.clone(),first:true});
        let thread={let writer=writer.clone();std::thread::spawn(move||writer.send_until(&serde_json::json!({"t":"req","op":"first"}),Instant::now()+Duration::from_millis(100)))};
        inside.recv_timeout(Duration::from_secs(1)).unwrap();
        let started=Instant::now();let error=writer.send_until(&serde_json::json!({"t":"req","op":"expired"}),Instant::now()+Duration::from_millis(30)).err().unwrap();
        assert!(!error.started);assert!(started.elapsed()<Duration::from_secs(1));
        assert!(thread.join().unwrap().err().unwrap().started);
        release.send(()).unwrap();
        writer.send(&serde_json::json!({"t":"evt","op":"run.stop"})).unwrap();
        let text=String::from_utf8_lossy(&bytes.lock().unwrap()).to_string();assert!(text.contains("first"));assert!(text.contains("run.stop"));assert!(!text.contains("expired"));writer.close();
    }
    #[test]fn a_frame_completed_after_its_deadline_is_refused_by_the_real_python_sdk(){
        let Some(python)=std::env::var_os("ZLC_PLUGIN_PYTHON") else{return};
        let(entered,inside)=mpsc::channel();let(release,wait)=mpsc::channel();let bytes=Arc::new(Mutex::new(Vec::new()));
        let writer=Outgoing::new(Blocked{entered,release:wait,bytes:bytes.clone(),first:true});
        let sending={let writer=writer.clone();std::thread::spawn(move||writer.send_until(&serde_json::json!({
            "t":"req","id":1,"op":"spi","p":{"instance":"axis","method":"mutate"},"blocks":[],"deadline_ms":10_000
        }),Instant::now()+Duration::from_millis(500)))};
        inside.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(sending.join().unwrap().err().unwrap().started);
        release.send(()).unwrap();
        writer.send(&serde_json::json!({"t":"evt","op":"fence"})).unwrap();
        let frame=frame::read(&mut std::io::Cursor::new(bytes.lock().unwrap().clone())).unwrap().unwrap();
        assert!(frame["deadline_ms"].as_u64().unwrap()>0,"relative TTL was positive before the write blocked");
        if let Some(now)=shared_monotonic_ns(){assert!(frame["deadline_monotonic_ns"].as_u64().unwrap()<=now);}
        let script=r#"import json,sys,threading
sys.path.insert(0,sys.argv[1])
from zlc_plugin import host
request=json.loads(sys.stdin.read())
executed=[]; answers=[]; ready=threading.Event()
class Channel:
 def send(self,message):
  answers.append(message); ready.set()
process=host.Process(Channel())
instance=host.Instance(process,'axis'); process.instances['axis']=instance
host._HANDLERS['spi']=lambda instance,params: executed.append('late hardware write')
process._request(request)
assert ready.wait(2)
instance.shutdown()
print(json.dumps({'executed':executed,'answer':answers[0]}))
"#;
        let mut command=std::process::Command::new(python);
        command.arg("-c").arg(script).arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../science/src"))
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
        #[cfg(windows)]{use std::os::windows::process::CommandExt;command.creation_flags(0x0800_0000);}
        let mut child=command.spawn().unwrap();
        {let mut input=child.stdin.take().unwrap();input.write_all(frame.to_string().as_bytes()).unwrap();}
        let output=child.wait_with_output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
        let result:Value=serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["executed"],serde_json::json!([]));assert_eq!(result["answer"]["error"]["kind"],"timeout");
        writer.close();
    }

    #[test]fn a_posted_frame_returns_at_once_and_its_failure_is_handled_by_the_writer(){
        let(entered,inside)=mpsc::channel();let(release,wait)=mpsc::channel();let bytes=Arc::new(Mutex::new(Vec::new()));
        let writer=Outgoing::new(Blocked{entered,release:wait,bytes:bytes.clone(),first:true});
        let(told,heard)=mpsc::channel();
        let first=told.clone();
        writer.post(&serde_json::json!({"t":"res","id":1,"ok":1}),move|sent|{let _=first.send(sent.is_ok());});
        inside.recv_timeout(Duration::from_secs(1)).unwrap();
        // The writer is stuck inside a frame: posting another does not wait for it.
        let started=Instant::now();
        let second=told.clone();
        writer.post(&serde_json::json!({"t":"res","id":2,"ok":2}),move|sent|{let _=second.send(sent.is_ok());});
        assert!(started.elapsed()<Duration::from_millis(50),"a response never waits for the writer");
        release.send(()).unwrap();
        assert!(heard.recv_timeout(Duration::from_secs(1)).unwrap());
        assert!(heard.recv_timeout(Duration::from_secs(1)).unwrap());
        // Once closed, a posted frame is refused at once and its failure handler runs.
        writer.close();
        let refused=told.clone();
        writer.post(&serde_json::json!({"t":"res","id":3,"ok":3}),move|sent|{let _=refused.send(sent.is_ok());});
        assert!(!heard.recv_timeout(Duration::from_secs(1)).unwrap());
        let text=String::from_utf8_lossy(&bytes.lock().unwrap()).to_string();assert!(text.contains("\"id\":2"));
    }
}
