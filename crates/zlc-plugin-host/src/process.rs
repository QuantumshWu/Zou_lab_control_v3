//! One plugin host process: its stdio channel, pending requests and log.
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::process::Child;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
    TerminateProcess, WaitForMultipleObjects};
use zlc_core::journal::now_us;
use zlc_core::process::{Job, local_time};

use crate::blocks::Lend;
use crate::{CallError, ErrorKind, outgoing::Outgoing};

/// How long a terminated process may take to be gone. Windows ends it at once and
/// closes its handles; only an I/O request a driver does not let go of (a stuck USB
/// transfer) keeps it, and that is reported rather than waited out.
pub(crate) const KILL_WAIT: Duration = Duration::from_secs(2);

/// A message read from a plugin, with the holds its `blocks` gave the core.
pub(crate) struct Incoming { pub message: Value, pub lends: Vec<Arc<Lend>> }

pub(crate) struct PluginProcess {
    /// The interpreter's pid (the launcher's until the handshake names it).
    pid: AtomicU32,
    /// The interpreter, from the handshake on. A venv's python.exe is a launcher whose
    /// child is the interpreter that opens the devices: the process is gone only when
    /// this handle (and the launcher's) is signalled.
    interpreter: OnceLock<OwnedHandle>,
    /// How the core ended it, once it did (None: it ended on its own, or runs).
    ended_by: Mutex<Option<String>>,
    /// The process it is: the one plugin process (`plugins`) or an isolated type's (`type:<id>`).
    pub key: String,
    writer: Arc<Outgoing>,
    /// Each request's waiter, and whether its deadline watches the process.
    pending: Mutex<HashMap<u64, (mpsc::Sender<Incoming>, bool, Option<String>)>>,
    /// One read-only snapshot worker per device, including replies that became
    /// late. Repeated Starts cannot accumulate blocked pipe writers or SDK reads.
    snapshots: Mutex<BTreeSet<String>>,
    snapshot_done: Condvar,
    next: AtomicU64,
    pub alive: AtomicBool,
    /// Set when the core itself ends the process: its exit is then no failure.
    pub retiring: AtomicBool,
    child: Mutex<Child>,
    log: Mutex<Log>,
    pub instances: Mutex<BTreeSet<String>>,
    /// Requests of no instance's using it now (the catalogue, a hardware scan).
    pub users: AtomicUsize,
    /// Requests that passed their deadline unanswered, by id: since when, and the instance asked.
    overdue: Mutex<HashMap<u64, (Instant, Option<String>)>>,
    pub missed_pings: AtomicU64,
}

impl PluginProcess {
    pub fn new(key: String, mut child: Child) -> (Arc<Self>, std::process::ChildStdout, std::process::ChildStderr) {
        let writer = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let process = Arc::new(Self {
            pid: AtomicU32::new(child.id()), interpreter: OnceLock::new(), ended_by: Mutex::default(), key, writer: Outgoing::new(writer), pending: Mutex::default(), snapshots: Mutex::default(), snapshot_done: Condvar::new(), next: AtomicU64::new(1),
            alive: AtomicBool::new(true), retiring: AtomicBool::new(false), child: Mutex::new(child),
            log: Mutex::default(), instances: Mutex::default(), users: AtomicUsize::new(0), overdue: Mutex::default(), missed_pings: AtomicU64::new(0),
        });
        (process, stdout, stderr)
    }

    pub fn pid(&self) -> u32 { self.pid.load(Ordering::Acquire) }

    /// The interpreter the handshake named: held from now on, and only if it is one of
    /// `job`'s processes (a pid Windows reused for another program is not).
    pub fn adopt_interpreter(&self, pid: u32, job: &Job) -> Result<(), String> {
        let handle = unsafe { OpenProcess(PROCESS_SYNCHRONIZE | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle.is_null() { return Err(format!("the interpreter (pid {pid}) cannot be held: {}", std::io::Error::last_os_error())); }
        let handle = unsafe { OwnedHandle::from_raw_handle(handle as _) };
        if !job.holds(&handle) { return Err(format!("pid {pid}, named at the handshake, is not one of this host's processes")); }
        self.interpreter.set(handle).map_err(|_| "the interpreter was already named".to_owned())?;
        self.pid.store(pid, Ordering::Release);
        Ok(())
    }

    pub fn send(&self, message: &Value) -> Result<(), CallError> {
        self.writer.send(message)
    }

    /// Queue a message and go on: `failed` runs, on the writer's thread, only if it was not written.
    pub fn post(&self, message: &Value, failed: impl FnOnce(CallError) + Send + 'static) {
        self.writer.post(message, move |sent| if let Err(error) = sent { failed(error); });
    }

    pub fn event(&self, op: &str, params: Value) -> Result<(), CallError> {
        self.send(&json!({"t": "evt", "op": op, "p": params}))
    }

    pub fn request(&self, op: &str, params: Value, deadline: Duration) -> Result<Incoming, CallError> {
        self.request_with(op, params, Vec::new(), deadline)
    }

    /// A request lending `blocks` (already held for this process by the caller).
    pub fn request_with(&self, op: &str, params: Value, blocks: Vec<String>, deadline: Duration) -> Result<Incoming, CallError> {
        self.exchange(op, params, blocks, deadline, true, &mut || {})
    }

    /// Release argument holds only when writer admission proves no frame began.
    pub fn request_with_cleanup(&self, op: &str, params: Value, blocks: Vec<String>, deadline: Duration,
        mut on_unsent: impl FnMut()) -> Result<Incoming, CallError> {
        self.exchange(op, params, blocks, deadline, true, &mut on_unsent)
    }

    /// A request whose missed deadline says nothing about the process's health
    /// (a hardware scan waiting on a vendor call): its late answer is dropped.
    pub fn request_untracked(&self, op: &str, params: Value, deadline: Duration) -> Result<Incoming, CallError> {
        self.exchange(op, params, Vec::new(), deadline, false, &mut || {})
    }

    /// Start metadata has one total budget, including pipe contention. A frame
    /// already being written must finish in the background to keep the shared
    /// channel valid, but this caller never waits past its deadline or ends the owner.
    pub fn snapshot(self: &Arc<Self>, instance: &str, units: &Value, deadline: Duration) -> Result<Incoming, CallError> {
        let until = Instant::now() + deadline;
        if !self.alive.load(Ordering::Acquire) { return Err(CallError::lost("the plugin process has ended")); }
        {
            let mut snapshots = self.snapshots.lock().unwrap();
            while snapshots.contains(instance) {
                if !self.alive.load(Ordering::Acquire) { return Err(CallError::lost("the plugin process has ended")); }
                let left = until.saturating_duration_since(Instant::now());
                if left.is_zero() { return Err(CallError::new(ErrorKind::Timeout, format!("{instance}: the previous snapshot is still pending"))); }
                snapshots = self.snapshot_done.wait_timeout(snapshots, left).unwrap_or_else(|poisoned| poisoned.into_inner()).0;
            }
            snapshots.insert(instance.to_owned());
        }
        let (reply, answer) = mpsc::channel();
        let (process, name, units) = (self.clone(), instance.to_owned(), units.clone());
        let spawned = std::thread::Builder::new().name("zlc-plugin-snapshot".into()).spawn(move || {
            let mut identity = None;
            let result = (|| {
                if Instant::now() >= until { return Err(CallError::new(ErrorKind::Timeout, "snapshot expired before it could be sent")); }
                let id = process.next.fetch_add(1, Ordering::Relaxed);
                identity = Some(id);
                let (sender, received) = mpsc::channel();
                process.pending.lock().unwrap().insert(id, (sender, false, Some(name.clone())));
                let sent = process.writer.send_until(&json!({"t":"req", "id":id, "op":"snapshot", "p":{"instance":name, "plain":true, "units":units},
                    "blocks":[], "deadline_ms":until.saturating_duration_since(Instant::now()).as_millis() as u64}), until);
                if let Err(error) = sent {
                    if !error.started { return Err(error.error); }
                    // A partially written frame must finish without interleaving.
                    // The bounded caller has left; retain this one per-device
                    // snapshot until its real reply or process retirement.
                }
                // The read loop retires pending on the actual answer or process
                // exit, even if the bounded caller has already gone away.
                let incoming = received.recv().map_err(|_| CallError::lost("the plugin process ended during snapshot"))?;
                match incoming.message.get("error") {
                    Some(error) => Err(CallError::from_wire(error)),
                    None => Ok(incoming),
                }
            })();
            if let Some(id) = identity { process.pending.lock().unwrap().remove(&id); }
            process.snapshots.lock().unwrap().remove(&name);
            process.snapshot_done.notify_all();
            let _ = reply.send(result);
        });
        if let Err(error) = spawned {
            self.snapshots.lock().unwrap().remove(instance);
            self.snapshot_done.notify_all();
            return Err(CallError::new(ErrorKind::Failed, format!("could not start snapshot: {error}")));
        }
        answer.recv_timeout(until.saturating_duration_since(Instant::now())).unwrap_or_else(|error| Err(match error {
            mpsc::RecvTimeoutError::Timeout => CallError::new(ErrorKind::Timeout, format!("snapshot did not answer within {:.1} s", deadline.as_secs_f64())),
            mpsc::RecvTimeoutError::Disconnected => CallError::lost("the snapshot worker ended without an answer"),
        }))
    }

    /// A health ping: a late answer is counted by the watcher, not as an overdue request.
    pub fn ping(&self) -> bool { self.exchange("ping", json!({}), Vec::new(), Duration::from_secs(1), false, &mut || {}).is_ok() }

    fn exchange(&self, op: &str, params: Value, blocks: Vec<String>, deadline: Duration, track: bool, on_unsent: &mut dyn FnMut()) -> Result<Incoming, CallError> {
        let until = Instant::now() + deadline;
        if !self.alive.load(Ordering::Acquire) { on_unsent(); return Err(CallError::lost("the plugin process has ended")); }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let instance = params["instance"].as_str().map(str::to_owned);
        let (sender, answer) = mpsc::channel();
        {
            let mut pending = self.pending.lock().unwrap();
            let emergency = matches!(op, "ping" | "exit" | "close" | "make_safe")
                || (op == "command" && params["name"] == "stop")
                || (op == "spi" && matches!(params["method"].as_str(), Some("stop" | "make_safe")));
            let limit = if emergency { 320 } else { 256 };
            if pending.len() >= limit { drop(pending); on_unsent(); return Err(CallError::new(ErrorKind::Refused, "the plugin has too many pending calls")); }
            pending.insert(id, (sender, track, instance.clone()));
        }
        let message = json!({"t": "req", "id": id, "op": op, "p": params, "blocks": blocks,
                             "deadline_ms": deadline.as_millis() as u64});
        if let Err(failure) = self.writer.send_until(&message, until) {
            let mut pending = self.pending.lock().unwrap();
            if failure.started && track {
                if pending.contains_key(&id) { self.overdue.lock().unwrap().insert(id, (Instant::now(), instance)); }
            } else { pending.remove(&id); }
            drop(pending);
            if !failure.started { on_unsent(); }
            return Err(failure.error);
        }
        match answer.recv_timeout(until.saturating_duration_since(Instant::now())) {
            Ok(incoming) => match incoming.message.get("error") {
                Some(error) => Err(CallError::from_wire(error)),
                None => Ok(incoming),
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let mut pending = self.pending.lock().unwrap();
                if track {
                    if pending.contains_key(&id) { self.overdue.lock().unwrap().insert(id, (Instant::now(), instance)); }
                } else { pending.remove(&id); }
                Err(CallError::new(ErrorKind::Timeout, format!("{op} did not answer within {:.1} s", deadline.as_secs_f64())))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(CallError::lost(format!(
                "the plugin process ended during {op}: {}", self.tail(6).join(" | ")))),
        }
    }

    /// Deliver a response; a late answer to an overdue request is only dropped.
    pub fn resolve(&self, id: u64, incoming: Incoming) {
        let mut pending = self.pending.lock().unwrap();
        self.overdue.lock().unwrap().remove(&id);
        if let Some((sender, _, _)) = pending.remove(&id) { let _ = sender.send(incoming); }
    }

    /// A caller's timeout is not completion: keep counting the request until
    /// its reply arrives or the process ends, including declared concurrent SPI.
    pub fn requests_pending(&self, instance: &str) -> bool {
        let pending = self.pending.lock().unwrap().values().any(|(_, _, name)| name.as_deref() == Some(instance));
        pending || self.snapshots.lock().unwrap().contains(instance)
    }

    /// The instances (None: the process's own request) whose requests have stayed
    /// unanswered `grace` past their deadline, each told once.
    pub fn overdue(&self, grace: Duration) -> Vec<Option<String>> {
        let mut overdue = self.overdue.lock().unwrap();
        let late: Vec<u64> = overdue.iter().filter(|(_, (since, _))| since.elapsed() > grace).map(|(id, _)| *id).collect();
        late.into_iter().filter_map(|id| overdue.remove(&id)).map(|(_, instance)| instance).collect()
    }

    /// The channel ended: every waiter learns the process is gone.
    pub fn ended(&self) {
        self.alive.store(false, Ordering::Release);
        self.pending.lock().unwrap().clear();
        self.snapshot_done.notify_all();
        self.writer.close();
    }

    /// End the interpreter (not only its launcher) and the launcher, and wait for both
    /// to be gone; `why` is how the session file will say it ended. False when it is
    /// still there [`KILL_WAIT`] later.
    pub fn kill(&self, why: &str) -> bool {
        self.ended_as(why);
        if let Some(interpreter) = self.interpreter.get() { unsafe { TerminateProcess(interpreter.as_raw_handle() as HANDLE, 1) }; }
        let _ = self.child.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).kill();
        self.wait_exit(KILL_WAIT)
    }

    /// Whether the core is ending it (asked it to exit, or killed it).
    pub fn ending(&self) -> bool { self.ended_by.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_some() }

    /// Say how the core ends it (the latest word: a kill after an ask is a kill).
    pub fn ended_as(&self, how: &str) {
        *self.ended_by.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(how.to_owned());
    }

    pub fn ended_by(&self) -> Option<String> { self.ended_by.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone() }

    /// Wait until the process is gone: its launcher and, once named, its interpreter
    /// have exited. False if one is still running at the deadline.
    pub fn wait_exit(&self, deadline: Duration) -> bool {
        // The handles stay valid while `self` lives: the Child and the OwnedHandle are its own.
        let mut handles = vec![self.child.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).as_raw_handle() as HANDLE];
        if let Some(interpreter) = self.interpreter.get() { handles.push(interpreter.as_raw_handle() as HANDLE); }
        let ms = u32::try_from(deadline.as_millis()).unwrap_or(u32::MAX - 1);
        let waited = unsafe { WaitForMultipleObjects(handles.len() as u32, handles.as_ptr(), 1, ms) };
        waited < WAIT_OBJECT_0 + handles.len() as u32
    }

    pub fn exit_code(&self) -> Option<i32> {
        self.child.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).try_wait().ok().flatten().and_then(|status| status.code())
    }

    /// A line the core adds to this process's log (it reads as the plugin's own).
    pub fn note(&self, line: &str) { self.log.lock().unwrap().keep(line.to_owned()); }

    pub fn tail(&self, lines: usize) -> Vec<String> { self.log.lock().unwrap().tail(lines) }

    /// Read what the plugin writes to stderr into its log, each line also told to
    /// `echo` with its time (journal microseconds); a line repeating the one before is counted, not repeated.
    pub fn read_log(self: &Arc<Self>, stderr: std::process::ChildStderr, echo: impl Fn(u64, &str) + Send + 'static) {
        let process = self.clone();
        let _ = std::thread::Builder::new().name(format!("zlc-plugin-log-{}", self.key)).spawn(move || {
            let tell = |said: Vec<(u64, String)>| for (us, line) in said {
                echo(us, &line);
                process.log.lock().unwrap().keep(line);
            };
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                let said = process.log.lock().unwrap().fold.line(line, now_us());
                tell(said);
            }
            let said = process.log.lock().unwrap().fold.end();
            tell(said.into_iter().collect());
        });
    }
}

const LOG_LINES: usize = 400;

/// A plugin's last [`LOG_LINES`] lines, and the run of repeats being counted now.
#[derive(Default)]
struct Log { lines: VecDeque<String>, fold: Fold }

impl Log {
    fn keep(&mut self, line: String) {
        if self.lines.len() == LOG_LINES { self.lines.pop_front(); }
        self.lines.push_back(line);
    }

    fn tail(&self, lines: usize) -> Vec<String> {
        let mut tail: Vec<String> = self.lines.iter().skip(self.lines.len().saturating_sub(lines)).cloned().collect();
        tail.extend(self.fold.pending());
        tail
    }
}

/// Consecutive identical lines as one, with a count: the first is said at once, the
/// rest are counted and said as one line when a different line comes (or the stream
/// ends), as the journal's `device_repeated` says a repeated device call.
#[derive(Default)]
pub(crate) struct Fold { last: Option<String>, again: u64, last_us: u64 }

impl Fold {
    /// What `line`, written at `us`, makes said: the run of repeats it ends, then itself;
    /// nothing when it repeats the line before.
    pub fn line(&mut self, line: String, us: u64) -> Vec<(u64, String)> {
        if self.last.as_ref() == Some(&line) { self.again += 1; self.last_us = us; return Vec::new(); }
        let mut said: Vec<(u64, String)> = self.end().into_iter().collect();
        said.push((us, line.clone()));
        (self.last, self.last_us) = (Some(line), us);
        said
    }

    /// The run of repeats not said yet, if any, said now.
    pub fn end(&mut self) -> Option<(u64, String)> {
        let said = self.pending().map(|line| (self.last_us, line));
        self.again = 0;
        said
    }

    fn pending(&self) -> Option<String> {
        (self.again > 0).then(|| format!("(the line above {} more time{}, the last at {})", self.again,
            if self.again == 1 { "" } else { "s" }, local_time(self.last_us)))
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::{HostConfig, PluginHost};
    use zlc_core::{control::SafeActions, runtime::Runtime};

    fn host() -> Option<(PluginHost, Arc<PluginProcess>)> {
        let python = std::env::var_os("ZLC_PLUGIN_PYTHON")?;
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let host = PluginHost::new(HostConfig { python: python.into(),
            python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
            packages: vec!["zlc_test_plugins".into()], memory_limit_mib: Some(3072),
        }, Runtime::new(), SafeActions::new(), |_| {}).unwrap();
        crate::test_application::install(&host).unwrap();
        host.open("echo", "test.echo", json!({}), json!({})).unwrap();
        let process = host.inner.state.lock().unwrap().instances["echo"].serving.process().unwrap().clone();
        Some((host, process))
    }

    #[test]
    fn service_workers_release_failed_arrays_and_cannot_restore_exited_process_resources() {
        let Some((host, process)) = host() else { return };
        let names = Arc::new(Mutex::new(Vec::<String>::new()));
        let exported = names.clone();
        let (started, ready) = mpsc::channel();
        let (resume, resumed) = mpsc::channel();
        let resumed = Mutex::new(resumed);
        *host.inner.services.lock().unwrap() = Some(Arc::new(move |_, params, context| {
            if params["wait"] == true {
                context.install_resource("fixture", Arc::new(7_u64))?;
                started.send(()).unwrap();
                resumed.lock().unwrap().recv_timeout(Duration::from_secs(10)).unwrap();
                assert!(context.install_resource("late", Arc::new(8_u64)).is_err());
            }
            let values = zlc_core::OwnedBuffer::from_vec(vec![9_u8; crate::wire::INLINE_BYTES]);
            let value = context.write_array(&values.into(), &[crate::wire::INLINE_BYTES])?;
            exported.lock().unwrap().push(value["$array"]["block"].as_str().unwrap().to_owned());
            if params["wait"] == true { Ok(value) } else { Err(CallError::new(ErrorKind::Failed, "fixture failure")) }
        }));
        let result = host.inner.service_request(&process, &json!({"operation":"fixture", "params":{}}), Vec::new());
        assert_eq!(result.err().unwrap().kind, ErrorKind::Failed);
        assert_eq!(host.registry().holds(&names.lock().unwrap()[0]), 0, "failed handlers retain no output holds");
        let (inner, caller) = (host.inner.clone(), process.clone());
        let pending = std::thread::spawn(move || inner.service_request(&caller,
            &json!({"operation":"fixture", "params":{"wait":true}}), Vec::new()));
        ready.recv_timeout(Duration::from_secs(10)).unwrap();
        host.shutdown();
        assert!(!process.alive.load(Ordering::Acquire));
        resume.send(()).unwrap();
        assert!(pending.join().unwrap().is_err());
        assert!(!host.inner.process_resources.lock().unwrap().contains_key(&process.pid()));
        for name in names.lock().unwrap().iter() { assert_eq!(host.registry().holds(name), 0, "late output {name} leaked"); }
    }

    #[test]
    fn an_expired_unsent_request_releases_its_argument_holds() {
        let Some((host, process)) = host() else { return };
        let block=host.registry().export(b"owned argument").unwrap();
        host.registry().hold_for(process.pid(),block.name()).unwrap();
        assert_eq!(host.registry().holds(block.name()),2);
        let result=process.request_with_cleanup("spi",json!({"instance":"echo","method":"frame"}),vec![block.name().to_owned()],Duration::ZERO,
            || host.registry().released_by(process.pid(),&[block.name().to_owned()]));
        assert_eq!(result.err().unwrap().kind,ErrorKind::Timeout);
        assert_eq!(host.registry().holds(block.name()),1);
        assert!(!process.requests_pending("echo"));host.shutdown();
    }

    #[test]
    fn local_snapshot_expires_before_a_busy_pipe_is_written_and_leaves_no_worker() {
        let Some((host, process)) = host() else { return };
        host.open("other", "test.echo", json!({}), json!({})).unwrap();
        let held = process.writer.pause();
        for name in ["echo", "other"] {
            let started = Instant::now();
            assert_eq!(host.snapshot(name, &json!({}), Duration::from_millis(50)).unwrap_err().kind, ErrorKind::Timeout);
            assert!(started.elapsed() < Duration::from_millis(500));
        }
        let until = Instant::now() + Duration::from_secs(1);
        while !process.snapshots.lock().unwrap().is_empty() && Instant::now() < until { std::thread::sleep(Duration::from_millis(5)); }
        assert!(process.snapshots.lock().unwrap().is_empty(), "expired unsent metadata must not retain workers");
        assert!(!process.requests_pending("echo") && !process.requests_pending("other"), "nothing was sent after the deadlines");
        drop(held);
        for name in ["echo", "other"] { assert_eq!(host.snapshot(name, &json!({}), Duration::from_secs(2)).unwrap()["values"]["amplitude"], 0.0); }
        host.shutdown();
    }

    #[test]
    fn repeated_snapshots_cannot_queue_up_and_shutdown_retires_the_one_late_reply() {
        let Some((host, process)) = host() else { return };
        let sleeping = host.clone();
        let call = std::thread::spawn(move || sleeping.spi("echo", "sleep", json!([30]), json!({}), Duration::from_secs(40)));
        let until = Instant::now() + Duration::from_secs(2);
        while !process.requests_pending("echo") && Instant::now() < until { std::thread::sleep(Duration::from_millis(5)); }
        assert!(process.requests_pending("echo"));
        for _ in 0..3 { assert_eq!(host.snapshot("echo", &json!({}), Duration::from_millis(50)).unwrap_err().kind, ErrorKind::Timeout); }
        assert_eq!(process.snapshots.lock().unwrap().len(), 1);
        assert_eq!(process.pending.lock().unwrap().values().filter(|(_, tracked, name)| !tracked && name.as_deref() == Some("echo")).count(), 1);
        let started = Instant::now();
        host.shutdown();
        assert!(started.elapsed() < Duration::from_secs(20));
        assert!(call.join().unwrap().is_err());
        let until = Instant::now() + Duration::from_secs(2);
        while !process.snapshots.lock().unwrap().is_empty() && Instant::now() < until { std::thread::sleep(Duration::from_millis(5)); }
        assert!(process.snapshots.lock().unwrap().is_empty());
        assert!(process.pending.lock().unwrap().is_empty());
        assert!(!process.alive.load(Ordering::Acquire));
    }

    #[test]
    fn only_explicit_shutdown_ends_an_owner_with_a_snapshot_write_blocked_in_a_full_pipe() {
        use std::process::{Command, Stdio};
        let Some((host, _)) = host() else { return };
        // This process deliberately never reads stdin. No device or network
        // is involved; a large instance name fills its real anonymous pipe.
        let child = host.inner.job.spawn(Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 60"])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()), 0x0800_0000).unwrap();
        let (process, stdout, stderr) = PluginProcess::new("blocked-snapshot-test".into(), child);
        host.inner.processes.lock().unwrap().insert(process.pid(), process.clone());
        process.read_log(stderr, |_, _| {});
        let (inner, reader) = (Arc::downgrade(&host.inner), process.clone());
        let reading = std::thread::spawn(move || crate::read_loop(inner, reader, stdout));
        let instance = "snapshot-instance-".repeat(65536);
        let started = Instant::now();
        assert_eq!(process.snapshot(&instance, &json!({}), Duration::from_millis(200)).err().unwrap().kind, ErrorKind::Timeout);
        assert!(started.elapsed() < Duration::from_secs(1));
        // The writer thread takes the frame up when it is scheduled (late on a loaded machine), then holds the full pipe.
        let until = Instant::now() + Duration::from_secs(2);
        while !process.writer.writing() && Instant::now() < until { std::thread::sleep(Duration::from_millis(5)); }
        assert!(process.writer.writing(), "the frame write really owns the full pipe");
        assert!(!process.wait_exit(Duration::ZERO), "metadata timeout must leave its owner alive");
        assert_eq!(process.snapshots.lock().unwrap().len(), 1);
        let shutting = Instant::now();
        host.shutdown();
        assert!(shutting.elapsed() < Duration::from_secs(20), "shutdown must reach its kill bound without the writer lock");
        reading.join().unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        while !process.snapshots.lock().unwrap().is_empty() && Instant::now() < until { std::thread::sleep(Duration::from_millis(5)); }
        assert!(process.snapshots.lock().unwrap().is_empty());
        assert!(process.pending.lock().unwrap().is_empty());
        assert!(!process.alive.load(Ordering::Acquire));
    }

    /// (created, exited) of a process, in Windows' 100 ns ticks.
    fn times(process: &impl AsRawHandle) -> (u64, u64) {
        use windows_sys::Win32::Foundation::FILETIME;
        let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
        assert_ne!(unsafe { windows_sys::Win32::System::Threading::GetProcessTimes(process.as_raw_handle() as HANDLE,
            &mut created, &mut exited, &mut kernel, &mut user) }, 0);
        let ticks = |time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
        (ticks(created), ticks(exited))
    }

    #[test]
    fn a_successor_starts_only_after_the_interpreter_not_its_launcher_is_gone() {
        use std::process::{Command, Stdio};
        let Some((host, _)) = host() else { return };
        // A launcher that starts its "interpreter" and exits at once, as a venv's python.exe
        // would if it did not wait: the child outlives it by seconds.
        let launcher = host.inner.job.spawn(Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command",
            "$child = Start-Process -PassThru -WindowStyle Hidden powershell.exe -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 4'; [Console]::Out.WriteLine($child.Id)"])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()), 0x0800_0000).unwrap();
        let (process, stdout, _stderr) = PluginProcess::new("type:successor-test".into(), launcher);
        let mut said = String::new();
        BufReader::new(stdout).read_line(&mut said).unwrap();
        let interpreter: u32 = said.trim().parse().unwrap();
        process.adopt_interpreter(interpreter, &host.inner.job).expect("the launcher's child was born in the job");
        assert!(process.child.lock().unwrap().wait().is_ok(), "the launcher has exited");
        assert!(!process.wait_exit(Duration::ZERO), "its interpreter has not");
        process.retiring.store(true, Ordering::Release);
        host.inner.gate("type:successor-test").lock().unwrap().push(process.clone());
        let successor = host.inner.acquire("type:successor-test", None).unwrap();
        let (_, gone) = times(process.interpreter.get().unwrap());
        let (born, _) = times(&*successor.child.lock().unwrap());
        assert!(gone > 0 && born >= gone, "the successor ({born}) started after the interpreter was gone ({gone})");
        assert!(host.inner.job.holds(&*successor.child.lock().unwrap()) && host.inner.job.holds(successor.interpreter.get().unwrap()),
            "the successor's launcher and interpreter are in the job");
        assert!(host.inner.gate("type:successor-test").lock().unwrap().iter().all(|started| Arc::ptr_eq(started, &successor)));
        host.inner.done(&successor);
        assert!(successor.wait_exit(Duration::ZERO), "retired: gone, interpreter and launcher");
        assert_eq!(successor.ended_by().as_deref(), Some("exited when asked"));
        host.shutdown();
    }

    #[test]
    fn repeated_stderr_lines_are_said_once_with_their_count() {
        let mut fold = Fold::default();
        assert_eq!(fold.line("warning".into(), 1_000_000), vec![(1_000_000, "warning".to_owned())]);
        assert!(fold.line("warning".into(), 2_000_000).is_empty() && fold.line("warning".into(), 3_000_000).is_empty());
        let said = fold.line("done".into(), 4_000_000);
        assert_eq!(said.len(), 2);
        assert_eq!(said[0].0, 3_000_000, "the run is said at its last repeat's time");
        assert!(said[0].1.starts_with("(the line above 2 more times, the last at "), "{}", said[0].1);
        assert_eq!(said[1], (4_000_000, "done".to_owned()));
        assert!(fold.end().is_none(), "nothing pending after a different line");
        assert!(fold.line("done".into(), 5_000_000).is_empty());
        assert!(fold.end().unwrap().1.starts_with("(the line above 1 more time, "));
    }
}
