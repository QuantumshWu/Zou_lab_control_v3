//! One plugin host process: its stdio channel, pending requests and log.
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::os::windows::io::AsRawHandle;
use std::process::{Child, ChildStdin};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, SetInformationJobObject, TerminateJobObject};
use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess};

use crate::blocks::Lend;
use crate::{CallError, ErrorKind, frame};

/// A message read from a plugin, with the holds its `blocks` gave the core.
pub(crate) struct Incoming { pub message: Value, pub lends: Vec<Arc<Lend>> }

pub(crate) struct PluginProcess {
    /// The interpreter's pid (the launcher's until the handshake names it).
    pid: AtomicU32,
    /// The process it is: the one plugin process (`plugins`) or an isolated type's (`type:<id>`).
    pub key: String,
    writer: Mutex<Option<ChildStdin>>,
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
    log: Mutex<VecDeque<String>>,
    pub instances: Mutex<BTreeSet<String>>,
    /// Requests of no instance's using it now (the catalogue, a hardware scan).
    pub users: AtomicUsize,
    /// Requests that passed their deadline unanswered, by id: since when, and the instance asked.
    overdue: Mutex<HashMap<u64, (Instant, Option<String>)>>,
    pub missed_pings: AtomicU64,
}

const LOG_LINES: usize = 400;

impl PluginProcess {
    pub fn new(key: String, mut child: Child) -> (Arc<Self>, std::process::ChildStdout, std::process::ChildStderr) {
        let writer = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let process = Arc::new(Self {
            pid: AtomicU32::new(child.id()), key, writer: Mutex::new(Some(writer)), pending: Mutex::default(), snapshots: Mutex::default(), snapshot_done: Condvar::new(), next: AtomicU64::new(1),
            alive: AtomicBool::new(true), retiring: AtomicBool::new(false), child: Mutex::new(child),
            log: Mutex::default(), instances: Mutex::default(), users: AtomicUsize::new(0), overdue: Mutex::default(), missed_pings: AtomicU64::new(0),
        });
        (process, stdout, stderr)
    }

    pub fn raw_handle(&self) -> HANDLE { self.child.lock().unwrap().as_raw_handle() as HANDLE }

    pub fn pid(&self) -> u32 { self.pid.load(Ordering::Acquire) }
    pub fn set_pid(&self, pid: u32) { self.pid.store(pid, Ordering::Release); }

    pub fn send(&self, message: &Value) -> Result<(), CallError> {
        let mut writer = self.writer.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let sink = writer.as_mut().ok_or_else(|| CallError::lost("the plugin process has ended"))?;
        frame::write(sink, message).map_err(|error| CallError::lost(format!("cannot write to the plugin: {error}")))
    }

    pub fn event(&self, op: &str, params: Value) -> Result<(), CallError> {
        self.send(&json!({"t": "evt", "op": op, "p": params}))
    }

    pub fn request(&self, op: &str, params: Value, deadline: Duration) -> Result<Incoming, CallError> {
        self.request_with(op, params, Vec::new(), deadline)
    }

    /// A request lending `blocks` (already held for this process by the caller).
    pub fn request_with(&self, op: &str, params: Value, blocks: Vec<String>, deadline: Duration) -> Result<Incoming, CallError> {
        self.exchange(op, params, blocks, deadline, true)
    }

    /// A request whose missed deadline says nothing about the process's health
    /// (a hardware scan waiting on a vendor call): its late answer is dropped.
    pub fn request_untracked(&self, op: &str, params: Value, deadline: Duration) -> Result<Incoming, CallError> {
        self.exchange(op, params, Vec::new(), deadline, false)
    }

    /// Start metadata has one total budget, including pipe contention. A frame
    /// already being written must finish in the background to keep the shared
    /// channel valid, but this caller never waits past its deadline or ends the owner.
    pub fn snapshot(self: &Arc<Self>, instance: &str, deadline: Duration) -> Result<Incoming, CallError> {
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
        let (process, name) = (self.clone(), instance.to_owned());
        let spawned = std::thread::Builder::new().name("zlc-plugin-snapshot".into()).spawn(move || {
            let mut identity = None;
            let result = (|| {
                let mut writer = loop {
                    if Instant::now() >= until { return Err(CallError::new(ErrorKind::Timeout, "snapshot expired before it could be sent")); }
                    if !process.alive.load(Ordering::Acquire) { return Err(CallError::lost("the plugin process has ended")); }
                    match process.writer.try_lock() {
                        Ok(writer) => break writer,
                        Err(std::sync::TryLockError::Poisoned(poisoned)) => break poisoned.into_inner(),
                        Err(std::sync::TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(1)),
                    }
                };
                if Instant::now() >= until { return Err(CallError::new(ErrorKind::Timeout, "snapshot expired before it could be sent")); }
                let sink = writer.as_mut().ok_or_else(|| CallError::lost("the plugin process has ended"))?;
                let id = process.next.fetch_add(1, Ordering::Relaxed);
                identity = Some(id);
                let (sender, received) = mpsc::channel();
                process.pending.lock().unwrap().insert(id, (sender, false, Some(name.clone())));
                frame::write(sink, &json!({"t":"req", "id":id, "op":"snapshot", "p":{"instance":name, "plain":true},
                    "blocks":[], "deadline_ms":until.saturating_duration_since(Instant::now()).as_millis() as u64}))
                    .map_err(|error| CallError::lost(format!("cannot write snapshot to the plugin: {error}")))?;
                drop(writer);
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
    pub fn ping(&self) -> bool { self.exchange("ping", json!({}), Vec::new(), Duration::from_secs(1), false).is_ok() }

    fn exchange(&self, op: &str, params: Value, blocks: Vec<String>, deadline: Duration, track: bool) -> Result<Incoming, CallError> {
        if !self.alive.load(Ordering::Acquire) { return Err(CallError::lost("the plugin process has ended")); }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let instance = params["instance"].as_str().map(str::to_owned);
        let (sender, answer) = mpsc::channel();
        self.pending.lock().unwrap().insert(id, (sender, track, instance.clone()));
        let message = json!({"t": "req", "id": id, "op": op, "p": params, "blocks": blocks,
                             "deadline_ms": deadline.as_millis() as u64});
        if let Err(error) = self.send(&message) {
            self.pending.lock().unwrap().remove(&id);
            return Err(error);
        }
        match answer.recv_timeout(deadline) {
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
        self.writer.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
    }

    /// End the interpreter (not only its launcher), then the launcher.
    pub fn kill(&self) {
        let mut child = self.child.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.pid() != child.id() {
            unsafe {
                let handle = OpenProcess(PROCESS_TERMINATE, 0, self.pid());
                if !handle.is_null() { TerminateProcess(handle, 1); CloseHandle(handle); }
            }
        }
        let _ = child.kill();
    }

    /// Wait for the process to exit; false if it is still running at the deadline.
    pub fn wait_exit(&self, deadline: Duration) -> bool {
        let until = Instant::now() + deadline;
        loop {
            if let Ok(Some(_)) = self.child.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).try_wait() { return true; }
            if Instant::now() >= until { return false; }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn exit_code(&self) -> Option<i32> {
        self.child.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).try_wait().ok().flatten().and_then(|status| status.code())
    }

    /// A line the core adds to this process's log (it reads as the plugin's own).
    pub fn note(&self, line: &str) {
        let mut log = self.log.lock().unwrap();
        if log.len() == LOG_LINES { log.pop_front(); }
        log.push_back(line.to_owned());
    }

    pub fn tail(&self, lines: usize) -> Vec<String> {
        let log = self.log.lock().unwrap();
        log.iter().skip(log.len().saturating_sub(lines)).cloned().collect()
    }

    pub fn read_log(self: &Arc<Self>, stderr: std::process::ChildStderr, echo: impl Fn(&str) + Send + 'static) {
        let process = self.clone();
        let _ = std::thread::Builder::new().name(format!("zlc-plugin-log-{}", self.key)).spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                echo(&line);
                let mut log = process.log.lock().unwrap();
                if log.len() == LOG_LINES { log.pop_front(); }
                log.push_back(line);
            }
        });
    }
}

/// Every plugin process belongs to this job: none survives the core.
pub(crate) struct Job(usize);

impl Job {
    pub fn new(memory_limit_mib: Option<u64>) -> Result<Self, String> {
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() { return Err(format!("Cannot create the plugin job: {}", std::io::Error::last_os_error())); }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if let Some(limit) = memory_limit_mib {
            info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
            info.JobMemoryLimit = (limit as usize) << 20;
        }
        let ok = unsafe { SetInformationJobObject(handle, JobObjectExtendedLimitInformation,
            (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32) };
        if ok == 0 {
            let error = std::io::Error::last_os_error();
            unsafe { CloseHandle(handle) };
            return Err(format!("Cannot limit the plugin job: {error}"));
        }
        Ok(Self(handle as usize))
    }

    pub fn assign(&self, process: HANDLE) -> Result<(), String> {
        if unsafe { AssignProcessToJobObject(self.0 as HANDLE, process) } == 0 {
            return Err(format!("Cannot place the plugin in its job: {}", std::io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Explicit host shutdown only: descendants can retain inherited pipes
    /// after their parent exits, so the whole owned tree must obey the deadline.
    pub fn terminate(&self) -> Result<(), String> {
        if unsafe { TerminateJobObject(self.0 as HANDLE, 1) } == 0 { return Err(std::io::Error::last_os_error().to_string()); }
        Ok(())
    }
}

impl Drop for Job {
    fn drop(&mut self) { unsafe { CloseHandle(self.0 as HANDLE) }; }
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
        host.open("echo", "test.echo", json!({}), json!({})).unwrap();
        let process = host.inner.state.lock().unwrap().instances["echo"].serving.process().unwrap().clone();
        Some((host, process))
    }

    #[test]
    fn local_snapshot_expires_before_a_busy_pipe_is_written_and_leaves_no_worker() {
        let Some((host, process)) = host() else { return };
        host.open("other", "test.echo", json!({}), json!({})).unwrap();
        let held = process.writer.lock().unwrap();
        for name in ["echo", "other"] {
            let started = Instant::now();
            assert_eq!(host.snapshot(name, Duration::from_millis(50)).unwrap_err().kind, ErrorKind::Timeout);
            assert!(started.elapsed() < Duration::from_millis(500));
        }
        let until = Instant::now() + Duration::from_secs(1);
        while !process.snapshots.lock().unwrap().is_empty() && Instant::now() < until { std::thread::sleep(Duration::from_millis(5)); }
        assert!(process.snapshots.lock().unwrap().is_empty(), "expired unsent metadata must not retain workers");
        assert!(!process.requests_pending("echo") && !process.requests_pending("other"), "nothing was sent after the deadlines");
        drop(held);
        for name in ["echo", "other"] { assert_eq!(host.snapshot(name, Duration::from_secs(2)).unwrap()["values"]["amplitude"], 0.0); }
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
        for _ in 0..3 { assert_eq!(host.snapshot("echo", Duration::from_millis(50)).unwrap_err().kind, ErrorKind::Timeout); }
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
        use std::os::windows::process::CommandExt;
        use std::process::{Command, Stdio};
        let Some((host, _)) = host() else { return };
        // This process deliberately never reads stdin. No device or network
        // is involved; a large instance name fills its real anonymous pipe.
        let child = Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 60"])
            .creation_flags(0x0800_0000).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let (process, stdout, stderr) = PluginProcess::new("blocked-snapshot-test".into(), child);
        host.inner.job.assign(process.raw_handle()).unwrap();
        host.inner.processes.lock().unwrap().insert(process.pid(), process.clone());
        process.read_log(stderr, |_| {});
        let (inner, reader) = (Arc::downgrade(&host.inner), process.clone());
        let reading = std::thread::spawn(move || crate::read_loop(inner, reader, stdout));
        let instance = "snapshot-instance-".repeat(65536);
        let started = Instant::now();
        assert_eq!(process.snapshot(&instance, Duration::from_millis(200)).err().unwrap().kind, ErrorKind::Timeout);
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(process.writer.try_lock().is_err(), "the frame write really owns the full pipe");
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
}
