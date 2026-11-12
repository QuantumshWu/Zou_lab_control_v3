//! What the application costs the machine, said in the journal once every [`PERIOD`]: the processor time
//! and memory of each of its processes -- itself, the plugin processes and the pages' browser processes,
//! found as its descendants -- the machine's own load beside them, and how the application's event loop
//! spent the period.
//!
//! Reading a process it already knows costs microseconds: its handle is held. Finding the family means
//! listing every process of the machine, which costs milliseconds, so that is done only when the machine's
//! set of process ids has changed -- a family gains or loses a member no other way.
//!
//! The loop's use is counted, not probed: every turn is timed, so a period's line says how many turns
//! there were, how long they took together, and the longest one with its name. A stall cannot fall between
//! two samples, and no duration decides what counts as one.
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS};
use windows_sys::Win32::System::ProcessStatus::{K32EnumProcesses, K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::{GetProcessTimes, GetSystemTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

/// How often a line is said. Short against what it is read for -- memory that grows over minutes, a busy
/// stretch of a scan -- and long against its own cost: a sample is some hundred microseconds of a thread
/// of its own (tens of milliseconds the times the family is found again), and an hour is 720 lines.
pub const PERIOD: Duration = Duration::from_secs(5);

#[derive(Default)]
struct LoopUse { turns: u64, busy: Duration, longest: Duration, longest_name: Cow<'static, str>, running: Option<(Instant, Cow<'static, str>)> }

static LOOP: Mutex<LoopUse> = Mutex::new(LoopUse { turns: 0, busy: Duration::ZERO, longest: Duration::ZERO, longest_name: Cow::Borrowed(""), running: None });

fn loop_use() -> std::sync::MutexGuard<'static, LoopUse> { LOOP.lock().unwrap_or_else(|error| error.into_inner()) }

/// One turn of the event loop, from when it is made to when it is dropped.
pub struct Turn(());

/// A turn of the event loop begins; `name` says what it handles.
pub fn turn(name: Cow<'static, str>) -> Turn {
    loop_use().running = Some((Instant::now(), name));
    Turn(())
}

impl Drop for Turn {
    fn drop(&mut self) {
        let mut used = loop_use();
        let Some((started, name)) = used.running.take() else { return };
        let took = started.elapsed();
        used.turns += 1;
        used.busy += took;
        if took > used.longest { used.longest = took; used.longest_name = name; }
    }
}

fn milliseconds(duration: Duration) -> f64 { (duration.as_secs_f64() * 1e4).round() / 10.0 }

/// How the loop spent the time since this was last asked; a turn still running is said with how long it has run.
fn loop_spent() -> Value {
    let mut used = loop_use();
    let mut spent = json!({"turns": used.turns, "busy_ms": milliseconds(used.busy), "longest_ms": milliseconds(used.longest), "longest": used.longest_name});
    if let Some((started, name)) = &used.running { spent["running"] = json!(name); spent["running_ms"] = json!(milliseconds(started.elapsed())); }
    let running = used.running.take();
    *used = LoopUse { running, ..LoopUse::default() };
    spent
}

/// A process as it is now: when it was created and the processor time it has used (both in 100 ns), and its memory.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Measured { created: u64, cpu: u64, working: usize, private: usize }

fn ticks(time: &FILETIME) -> u64 { (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime) }

const NO_TIME: FILETIME = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };

/// A process held open to be read: the handle names the process itself, whatever becomes of its pid.
struct Held(HANDLE);

impl Held {
    fn open(pid: u32) -> Option<Self> {
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        (!process.is_null()).then_some(Self(process))
    }

    fn measured(&self) -> Option<Measured> {
        let (mut created, mut exited, mut kernel, mut user) = (NO_TIME, NO_TIME, NO_TIME, NO_TIME);
        let mut memory: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        memory.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        let read = unsafe {
            GetProcessTimes(self.0, &mut created, &mut exited, &mut kernel, &mut user) != 0
                && K32GetProcessMemoryInfo(self.0, (&mut memory as *mut PROCESS_MEMORY_COUNTERS_EX).cast(), memory.cb) != 0
        };
        read.then(|| Measured { created: ticks(&created), cpu: ticks(&kernel) + ticks(&user), working: memory.WorkingSetSize, private: memory.PrivateUsage })
    }
}

impl Drop for Held { fn drop(&mut self) { unsafe { CloseHandle(self.0) }; } }

fn measured(pid: u32) -> Option<Measured> { Held::open(pid)?.measured() }

/// The machine's process ids, in order; None when they cannot be read.
fn process_ids() -> Option<Vec<u32>> {
    let mut ids = vec![0u32; 1024];
    loop {
        let mut bytes = 0u32;
        if unsafe { K32EnumProcesses(ids.as_mut_ptr(), (ids.len() * size_of::<u32>()) as u32, &mut bytes) } == 0 { return None; }
        let count = bytes as usize / size_of::<u32>();
        // A full buffer may have been cut short.
        if count < ids.len() { ids.truncate(count); ids.sort_unstable(); return Some(ids); }
        ids.resize(ids.len() * 2, 0);
    }
}

/// Every process of the machine: its pid, its parent's and its program's name.
fn listed() -> Vec<(u32, u32, String)> {
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
    if snapshot == INVALID_HANDLE_VALUE { return Vec::new(); }
    let mut entry: PROCESSENTRY32W = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<PROCESSENTRY32W>() as u32;
    let mut all = Vec::new();
    let mut more = unsafe { Process32FirstW(snapshot, &mut entry) } != 0;
    while more {
        let name = &entry.szExeFile[..entry.szExeFile.iter().position(|unit| *unit == 0).unwrap_or(entry.szExeFile.len())];
        all.push((entry.th32ProcessID, entry.th32ParentProcessID, String::from_utf16_lossy(name)));
        more = unsafe { Process32NextW(snapshot, &mut entry) } != 0;
    }
    unsafe { CloseHandle(snapshot) };
    all
}

/// `root` and its descendants among `listed`, each with its measure. A parent's pid outlives the parent:
/// a process is a child only if it was created no earlier than the process whose pid it names.
fn family(listed: &[(u32, u32, String)], root: u32, mut measure: impl FnMut(u32) -> Option<Measured>) -> Vec<(u32, u32, String, Measured)> {
    let Some(own) = measure(root) else { return Vec::new() };
    let name = listed.iter().find(|(pid, ..)| *pid == root).map_or_else(String::new, |(_, _, name)| name.clone());
    let mut found = vec![(root, 0, name, own)];
    let mut next = 0;
    while next < found.len() {
        let (parent, born) = (found[next].0, found[next].3.created);
        next += 1;
        for (pid, of, name) in listed {
            if *of != parent || found.iter().any(|(known, ..)| known == pid) { continue; }
            if let Some(child) = measure(*pid).filter(|child| child.created >= born) { found.push((*pid, parent, name.clone(), child)); }
        }
    }
    found
}

fn mebibytes(bytes: usize) -> f64 { (bytes as f64 / 1048576.0 * 10.0).round() / 10.0 }

/// One of the family, held open; `cpu` is its processor time when it was last read.
struct Member { pid: u32, parent: u32, name: String, process: Held, created: u64, cpu: Option<u64> }

/// The family as last found and what was read last time, by which this time's is a rate.
#[derive(Default)]
struct Sampler { at: Option<Instant>, ids: Option<Vec<u32>>, family: Vec<Member>, machine: Option<(u64, u64)> }

impl Sampler {
    /// The machine's processors' busy share since last asked (0..1), and its free memory.
    fn machine(&mut self) -> Value {
        let (mut idle, mut kernel, mut user) = (NO_TIME, NO_TIME, NO_TIME);
        let mut said = json!({});
        if unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } != 0 {
            // The kernel's time counts its idle time.
            let now = (ticks(&idle), ticks(&kernel) + ticks(&user));
            if let Some((idle, all)) = self.machine.replace(now).map(|before| (now.0 - before.0, now.1 - before.1)).filter(|(_, all)| *all > 0) {
                said["cpu"] = json!(((1.0 - idle as f64 / all as f64) * 1000.0).round() / 1000.0);
            }
        }
        let mut memory: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        memory.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut memory) } != 0 { said["free_mib"] = json!((memory.ullAvailPhys >> 20)); said["memory_mib"] = json!((memory.ullTotalPhys >> 20)); }
        said
    }

    /// Find the family again, keeping what is known of the members that are still the same processes.
    fn find(&mut self, root: u32) {
        let mut known: HashMap<(u32, u64), Member> = std::mem::take(&mut self.family).into_iter().map(|member| ((member.pid, member.created), member)).collect();
        self.family = family(&listed(), root, measured).into_iter().filter_map(|(pid, parent, name, process)| {
            known.remove(&(pid, process.created)).or_else(|| Some(Member { pid, parent, name, process: Held::open(pid)?, created: process.created, cpu: None }))
        }).collect();
    }

    /// One line's fields. A process's `cpu` is the processors it used since the last line (1 is one whole
    /// processor), absent the first time it is read.
    fn sample(&mut self, root: u32) -> Value {
        let ids = process_ids();
        if ids.is_none() || ids != self.ids { self.find(root); self.ids = ids; }
        let now = Instant::now();
        let elapsed = self.at.replace(now).map(|before| (now - before).as_secs_f64() * 1e7);
        let processes: Vec<Value> = self.family.iter_mut().filter_map(|member| {
            let process = member.process.measured()?;
            let mut said = json!({"pid": member.pid, "exe": member.name, "working_mib": mebibytes(process.working), "private_mib": mebibytes(process.private)});
            if member.parent != 0 { said["parent"] = json!(member.parent); }
            if let (Some(elapsed), Some(before)) = (elapsed, member.cpu.replace(process.cpu)) {
                said["cpu"] = json!((process.cpu.saturating_sub(before) as f64 / elapsed * 1000.0).round() / 1000.0);
            }
            Some(said)
        }).collect();
        json!({"machine": self.machine(), "loop": loop_spent(), "processes": processes})
    }
}

/// Say the application's use of the machine in the journal from now on.
pub fn start() {
    let root = std::process::id();
    let cores = std::thread::available_parallelism().map_or(0, |cores| cores.get());
    zlc_core::journal::lifecycle("system", "machine", json!({"processors": cores, "resources_every_s": PERIOD.as_secs()}));
    let _ = std::thread::Builder::new().name("zlc-resources".into()).spawn(move || {
        let mut sampler = Sampler::default();
        loop {
            zlc_core::journal::note_fields("system", "resources", &sampler.sample(root).to_string());
            std::thread::sleep(PERIOD);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_process_s_family_is_its_descendants_born_after_it() {
        let listed: Vec<(u32, u32, String)> = [(10, 1, "app"), (20, 10, "plugin"), (21, 20, "interpreter"), (30, 10, "stale"), (40, 99, "other"), (50, 21, "worker")]
            .into_iter().map(|(pid, parent, name)| (pid, parent, name.to_owned())).collect();
        // 30 names 10 as its parent but is older than it: the pid it names was another process's.
        let born = |pid: u32| Some(Measured { created: if pid == 30 { 5 } else { u64::from(pid) * 10 }, cpu: 0, working: 0, private: 0 });
        let found: Vec<(u32, u32)> = family(&listed, 10, born).into_iter().map(|(pid, parent, ..)| (pid, parent)).collect();
        assert_eq!(found, [(10, 0), (20, 10), (21, 20), (50, 21)]);
        assert!(family(&listed, 10, |_| None).is_empty(), "a root that cannot be read has no family");
    }

    #[test]
    fn a_sample_names_this_process_its_memory_and_the_loop_s_turns() {
        let mut sampler = Sampler::default();
        let first = sampler.sample(std::process::id());
        let own = &first["processes"][0];
        assert_eq!(own["pid"], std::process::id());
        assert!(own["working_mib"].as_f64().unwrap() > 0.0 && own["private_mib"].as_f64().unwrap() > 0.0, "{own}");
        assert!(own.get("cpu").is_none(), "a rate needs two readings");
        assert!(first["machine"]["free_mib"].as_u64().unwrap() > 0);
        // Two turns, one of them long; then something to measure.
        loop_spent();
        drop(turn(Cow::Borrowed("short")));
        { let _long = turn(Cow::Borrowed("long")); std::thread::sleep(Duration::from_millis(30)); }
        let busy = Instant::now();
        while busy.elapsed() < Duration::from_millis(60) { std::hint::black_box(busy.elapsed()); }
        let started = Instant::now();
        let second = sampler.sample(std::process::id());
        let cost = started.elapsed();
        let (started, ids) = (Instant::now(), process_ids().unwrap());
        let listing_ids = started.elapsed();
        let started = Instant::now();
        sampler.find(std::process::id());
        let finding = started.elapsed();
        assert!(ids.contains(&std::process::id()) && sampler.family[0].cpu.is_some(), "found again, a member keeps what was read of it");
        let spent = &second["loop"];
        assert_eq!((&spent["turns"], &spent["longest"]), (&json!(2), &json!("long")));
        assert!(spent["longest_ms"].as_f64().unwrap() >= 30.0 && spent["busy_ms"].as_f64().unwrap() >= spent["longest_ms"].as_f64().unwrap(), "{spent}");
        assert!(second["processes"][0]["cpu"].as_f64().unwrap() > 0.0, "{}", second["processes"][0]);
        assert!(second["machine"]["cpu"].as_f64().is_some_and(|share| (0.0..=1.0).contains(&share)), "{}", second["machine"]);
        assert_eq!(loop_spent()["turns"], 0, "asked, the count starts again");
        println!("one sample: {cost:?} ({} ids listed in {listing_ids:?}; the family found again in {finding:?}), {} bytes, {} processes",
                 ids.len(), second.to_string().len(), second["processes"].as_array().unwrap().len());
    }
}
