//! One plugin host process: its stdio channel, pending requests and log.
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::os::windows::io::AsRawHandle;
use std::process::{Child, ChildStdin};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, SetInformationJobObject};
use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess};

use crate::blocks::Lend;
use crate::{CallError, ErrorKind, frame};

/// A message read from a plugin, with the holds its `blocks` gave the core.
pub(crate) struct Incoming { pub message: Value, pub lends: Vec<Arc<Lend>> }

pub(crate) struct PluginProcess {
    /// The interpreter's pid (the launcher's until the handshake names it).
    pid: AtomicU32,
    /// The process group (`group:<name>`) or the type (`type:<id>`) it serves.
    pub key: String,
    writer: Mutex<Option<ChildStdin>>,
    pending: Mutex<HashMap<u64, mpsc::Sender<Incoming>>>,
    next: AtomicU64,
    pub alive: AtomicBool,
    /// Set when the core itself ends the process: its exit is then no failure.
    pub retiring: AtomicBool,
    child: Mutex<Child>,
    log: Mutex<VecDeque<String>>,
    pub instances: Mutex<BTreeSet<String>>,
    /// Requests that passed their deadline unanswered, by id.
    overdue: Mutex<HashMap<u64, Instant>>,
    pub missed_pings: AtomicU64,
}

const LOG_LINES: usize = 400;

impl PluginProcess {
    pub fn new(key: String, mut child: Child) -> (Arc<Self>, std::process::ChildStdout, std::process::ChildStderr) {
        let writer = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let process = Arc::new(Self {
            pid: AtomicU32::new(child.id()), key, writer: Mutex::new(Some(writer)), pending: Mutex::default(), next: AtomicU64::new(1),
            alive: AtomicBool::new(true), retiring: AtomicBool::new(false), child: Mutex::new(child),
            log: Mutex::default(), instances: Mutex::default(), overdue: Mutex::default(), missed_pings: AtomicU64::new(0),
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

    /// A health ping: a late answer is counted by the watcher, not as an overdue request.
    pub fn ping(&self) -> bool { self.exchange("ping", json!({}), Vec::new(), Duration::from_secs(1), false).is_ok() }

    fn exchange(&self, op: &str, params: Value, blocks: Vec<String>, deadline: Duration, track: bool) -> Result<Incoming, CallError> {
        if !self.alive.load(Ordering::Acquire) { return Err(CallError::lost("the plugin process has ended")); }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (sender, answer) = mpsc::channel();
        self.pending.lock().unwrap().insert(id, sender);
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
                if track { self.overdue.lock().unwrap().insert(id, Instant::now()); }
                else { self.pending.lock().unwrap().remove(&id); }
                Err(CallError::new(ErrorKind::Timeout, format!("{op} did not answer within {:.1} s", deadline.as_secs_f64())))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(CallError::lost(format!(
                "the plugin process ended during {op}: {}", self.tail(6).join(" | ")))),
        }
    }

    /// Deliver a response; a late answer to an overdue request is only dropped.
    pub fn resolve(&self, id: u64, incoming: Incoming) {
        self.overdue.lock().unwrap().remove(&id);
        if let Some(sender) = self.pending.lock().unwrap().remove(&id) { let _ = sender.send(incoming); }
    }

    /// Whether a request is in flight (its own deadline then watches the process).
    pub fn busy(&self) -> bool { !self.pending.lock().unwrap().is_empty() }

    /// The oldest unanswered overdue request's age.
    pub fn overdue_for(&self) -> Option<Duration> {
        self.overdue.lock().unwrap().values().map(Instant::elapsed).max()
    }

    /// The channel ended: every waiter learns the process is gone.
    pub fn ended(&self) {
        self.alive.store(false, Ordering::Release);
        self.pending.lock().unwrap().clear();
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
}

impl Drop for Job {
    fn drop(&mut self) { unsafe { CloseHandle(self.0 as HANDLE) }; }
}
