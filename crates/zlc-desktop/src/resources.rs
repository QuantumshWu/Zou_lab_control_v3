//! What the application costs the machine, said in the journal once every [`PERIOD`]: the processor time
//! and memory of each of its processes -- itself, the plugin processes and the pages' browser processes,
//! found as its descendants -- the machine's own load beside them, who else held the machine when most
//! of its load was not the application's, and how the application's event loop spent the period.
//!
//! One call lists every process of the machine with its processor time and memory: the family and the
//! others are both read from that list, so a sample opens no process and the family is found anew each
//! time. No command line or path is read, only each program's file name.
//!
//! The loop's use is counted, not probed: every turn is timed, so a period's line says how many turns
//! there were, how long they took together, and the longest one with its name. A stall cannot fall between
//! two samples, and no duration decides what counts as one.
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use windows_sys::Wdk::System::SystemInformation::{NtQuerySystemInformation, SystemProcessInformation};
use windows_sys::Win32::Foundation::{FILETIME, STATUS_INFO_LENGTH_MISMATCH, UNICODE_STRING};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::GetSystemTimes;

/// How often a line is said. Short against what it is read for -- memory that grows over minutes, a busy
/// stretch of a scan -- and long against its own cost: a sample is some milliseconds of a thread of its
/// own, nearly all of them the kernel writing the machine's process list (it lists every thread too), and
/// an hour is 720 lines.
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

/// A process as the machine lists it: when it was created and the processor time it has used (both in
/// 100 ns), and its memory. A pid is reused, so a process is its pid with its creation time.
#[derive(Clone, Debug, PartialEq)]
struct Process { pid: u32, parent: u32, name: String, created: u64, cpu: u64, working: usize, private: usize }

impl Process { fn key(&self) -> (u32, u64) { (self.pid, self.created) } }

fn ticks(time: &FILETIME) -> u64 { (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime) }

const NO_TIME: FILETIME = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };

/// The head of an entry of the machine's process list (SYSTEM_PROCESS_INFORMATION, 64-bit) up to the last
/// field read here. The documented header calls the times `Reserved1` and the parent `Reserved2`.
#[repr(C)]
struct Entry {
    next: u32,
    _threads: u32,
    /// WorkingSetPrivateSize, HardFaultCount with NumberOfThreadsHighWatermark, CycleTime.
    _counts: [u64; 3],
    created: u64,
    user: u64,
    kernel: u64,
    name: UNICODE_STRING,
    _base_priority: i32,
    pid: usize,
    parent: usize,
    /// HandleCount with SessionId, UniqueProcessKey, PeakVirtualSize, VirtualSize, PageFaultCount, PeakWorkingSetSize.
    _sizes: [usize; 6],
    working: usize,
    /// The paged and the non-paged pools' quota, each at its peak and now.
    _quotas: [usize; 4],
    /// PagefileUsage: the private memory committed, what GetProcessMemoryInfo calls PrivateUsage.
    private: usize,
}

const _: () = assert!(std::mem::offset_of!(Entry, created) == 0x20 && std::mem::offset_of!(Entry, pid) == 0x50 && std::mem::offset_of!(Entry, private) == 0xB8);

/// Every process of the machine with its times and memory, read by one call into `buffer` (kept between
/// calls, grown when the list outgrows it); None when the list cannot be read.
fn listed(buffer: &mut Vec<u64>) -> Option<Vec<Process>> {
    loop {
        let mut needed = 0u32;
        let bytes = u32::try_from(size_of_val(buffer.as_slice())).ok()?;
        let status = unsafe { NtQuerySystemInformation(SystemProcessInformation, buffer.as_mut_ptr().cast(), bytes, &mut needed) };
        if status == STATUS_INFO_LENGTH_MISMATCH {
            // The list may grow between being measured and being read.
            buffer.resize((needed as usize).div_ceil(size_of::<u64>()).max(2 * buffer.len()), 0);
            continue;
        }
        if status < 0 { return None; }
        break;
    }
    let (base, mut at, mut all) = (buffer.as_ptr().cast::<u8>(), 0, Vec::new());
    loop {
        // The kernel wrote whole entries, each aligned for its fields, chained by their offsets.
        let entry = unsafe { &*base.add(at).cast::<Entry>() };
        // The idle process's time is the processors' idle time, which the machine's own times say.
        if entry.pid != 0 {
            let name = if entry.name.Buffer.is_null() { String::new() } else {
                String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(entry.name.Buffer, usize::from(entry.name.Length) / 2) })
            };
            all.push(Process { pid: entry.pid as u32, parent: entry.parent as u32, name, created: entry.created, cpu: entry.kernel + entry.user,
                               working: entry.working, private: entry.private });
        }
        if entry.next == 0 { return Some(all); }
        at += entry.next as usize;
    }
}

/// `root` and its descendants among `listed`, as indices into it, `root` first. A parent's pid outlives the
/// parent: a process is a child only if it was created no earlier than the process whose pid it names.
fn family(listed: &[Process], root: u32) -> Vec<usize> {
    let Some(own) = listed.iter().position(|process| process.pid == root) else { return Vec::new() };
    let mut found = vec![own];
    let mut next = 0;
    while next < found.len() {
        let (parent, born) = (listed[found[next]].pid, listed[found[next]].created);
        next += 1;
        for (index, process) in listed.iter().enumerate() {
            if process.parent == parent && process.created >= born && !found.contains(&index) { found.push(index); }
        }
    }
    found
}

/// The processor time `process` used since the listing `before` (100 ns): a process that listing did not
/// have was born since, so all its time is this period's.
fn used_since(process: &Process, before: &HashMap<(u32, u64), u64>) -> u64 {
    process.cpu.saturating_sub(before.get(&process.key()).copied().unwrap_or(0))
}

/// Who besides the application kept the machine busy, all times in 100 ns: `busy` is the machine's busy
/// processor time in the period, `ours` the application's, `used` every other process with its time.
/// When the busy time not ours is the larger part -- more than ours -- it is said: how much it is, how much
/// of it the other processes hold (the rest no process holds: interrupts, deferred procedure calls, and
/// processes that ended within the period), and from the busiest down the fewest of those processes that
/// together hold more than half of theirs.
fn others<'a>(busy: u64, ours: u64, mut used: Vec<(&'a Process, u64)>) -> Option<(u64, u64, Vec<(&'a Process, u64)>)> {
    let beyond = busy.saturating_sub(ours);
    if beyond <= ours { return None; }
    let held: u64 = used.iter().map(|(_, cpu)| cpu).sum();
    used.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.pid.cmp(&b.0.pid)));
    let mut named = 0;
    let count = used.iter().take_while(|(_, cpu)| { let more = 2 * named <= held && *cpu > 0; named += cpu; more }).count();
    used.truncate(count);
    Some((beyond, held, used))
}

fn mebibytes(bytes: usize) -> f64 { (bytes as f64 / 1048576.0 * 10.0).round() / 10.0 }

/// Processor time over a period, both in 100 ns, as processors: 1 is one whole processor.
fn processors(time: u64, period: f64) -> f64 { (time as f64 / period * 1000.0).round() / 1000.0 }

/// What was read last time, by which this time's is a rate.
#[derive(Default)]
struct Sampler { at: Option<Instant>, buffer: Vec<u64>, before: Option<HashMap<(u32, u64), u64>>, machine: Option<(u64, u64)> }

impl Sampler {
    /// The machine's processors' busy share since last asked (0..1) and its free memory; beside them its
    /// busy processor time since last asked (100 ns).
    fn machine(&mut self) -> (Value, Option<u64>) {
        let (mut idle, mut kernel, mut user) = (NO_TIME, NO_TIME, NO_TIME);
        let (mut said, mut busy) = (json!({}), None);
        if unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } != 0 {
            // The kernel's time counts its idle time.
            let now = (ticks(&idle), ticks(&kernel) + ticks(&user));
            if let Some((idle, all)) = self.machine.replace(now).map(|before| (now.0 - before.0, now.1 - before.1)).filter(|(_, all)| *all > 0) {
                said["cpu"] = json!(((1.0 - idle as f64 / all as f64) * 1000.0).round() / 1000.0);
                busy = Some(all.saturating_sub(idle));
            }
        }
        let mut memory: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        memory.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut memory) } != 0 { said["free_mib"] = json!((memory.ullAvailPhys >> 20)); said["memory_mib"] = json!((memory.ullTotalPhys >> 20)); }
        (said, busy)
    }

    /// One line's fields. A process's `cpu` is the processors it used since the last line (1 is one whole
    /// processor; a process born since then counts from its birth), absent on the first line; `others`
    /// is said as [`others`] decides, in the same unit.
    fn sample(&mut self, root: u32) -> Value {
        let listing = listed(&mut self.buffer);
        let (machine, busy) = self.machine();
        let now = Instant::now();
        let period = self.at.replace(now).map(|before| (now - before).as_secs_f64() * 1e7);
        let listed = listing.unwrap_or_default();
        let before = std::mem::replace(&mut self.before, (!listed.is_empty()).then(|| listed.iter().map(|process| (process.key(), process.cpu)).collect()));
        let used = |process: &Process| period.zip(before.as_ref()).map(|(period, before)| (used_since(process, before), period));
        let family = family(&listed, root);
        let mut ours = 0;
        let processes: Vec<Value> = family.iter().enumerate().map(|(member, &index)| {
            let process = &listed[index];
            let mut said = json!({"pid": process.pid, "exe": process.name, "working_mib": mebibytes(process.working), "private_mib": mebibytes(process.private)});
            if member > 0 { said["parent"] = json!(process.parent); }
            if let Some((cpu, period)) = used(process) { ours += cpu; said["cpu"] = json!(processors(cpu, period)); }
            said
        }).collect();
        let mut line = json!({"machine": machine, "loop": loop_spent(), "processes": processes});
        if let (Some(busy), Some(period), Some(_)) = (busy, period, &before) {
            let rest = listed.iter().enumerate().filter(|(index, _)| !family.contains(index)).filter_map(|(_, process)| Some((process, used(process)?.0))).collect();
            if let Some((beyond, held, named)) = others(busy, ours, rest) {
                let named: Vec<Value> = named.into_iter().map(|(process, cpu)| {
                    json!({"exe": process.name, "pid": process.pid, "cpu": processors(cpu, period), "private_mib": mebibytes(process.private)})
                }).collect();
                line["others"] = json!({"cpu": processors(beyond, period), "in_processes": processors(held, period), "processes": named});
            }
        }
        line
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

    fn process(pid: u32, parent: u32, created: u64, cpu: u64) -> Process {
        Process { pid, parent, name: format!("p{pid}.exe"), created, cpu, working: 0, private: 1 << 20 }
    }

    #[test]
    fn a_process_s_family_is_its_descendants_born_after_it() {
        // 30 names 10 as its parent but is older than it: the pid it names was another process's.
        let listed: Vec<Process> = [(10, 1), (20, 10), (21, 20), (30, 10), (40, 99), (50, 21)].into_iter()
            .map(|(pid, parent)| process(pid, parent, if pid == 30 { 5 } else { u64::from(pid) * 10 }, 0)).collect();
        let found: Vec<u32> = family(&listed, 10).into_iter().map(|index| listed[index].pid).collect();
        assert_eq!(found, [10, 20, 21, 50]);
        assert!(family(&listed, 11).is_empty(), "a root not listed has no family");
    }

    #[test]
    fn the_others_named_are_the_fewest_busiest_holding_most_of_their_time() {
        let [a, b, c, d, e] = [1, 2, 3, 4, 5].map(|pid| process(pid, 1, 0, 0));
        let said = |busy, ours, used| others(busy, ours, used).map(|(beyond, held, named)| (beyond, held, named.iter().map(|(process, cpu)| (process.pid, *cpu)).collect::<Vec<_>>()));
        // Most of the busy time ours: nothing said; half is not most.
        assert_eq!(said(150, 100, vec![(&a, 50)]), None);
        assert_eq!(said(200, 100, vec![(&a, 100)]), None);
        // One process holds more than half of the others' time: it alone is named.
        assert_eq!(said(1000, 10, vec![(&d, 50), (&a, 500), (&e, 0), (&b, 300), (&c, 100)]), Some((990, 950, vec![(1, 500)])));
        // Even shares: as many as it takes to pass half; exactly half is not more than half.
        assert_eq!(said(900, 0, vec![(&c, 300), (&b, 300), (&a, 300)]), Some((900, 900, vec![(1, 300), (2, 300)])));
        assert_eq!(said(800, 0, vec![(&a, 400), (&b, 200), (&c, 200)]), Some((800, 800, vec![(1, 400), (2, 200)])));
        // Busy time no process holds (interrupts) is said beside the processes' part of it.
        assert_eq!(said(1000, 0, vec![(&a, 60), (&b, 40)]), Some((1000, 100, vec![(1, 60)])));
        assert_eq!(said(500, 0, vec![(&a, 0)]), Some((500, 0, vec![])), "no process to name");
        // A period's time: since the last listing, or since its birth for a process born since -- a pid reused too.
        let before: HashMap<(u32, u64), u64> = [((7, 70), 1000)].into_iter().collect();
        assert_eq!(used_since(&process(7, 1, 70, 1600), &before), 600);
        assert_eq!(used_since(&process(7, 1, 90, 250), &before), 250);
    }

    #[test]
    fn a_sample_names_this_process_its_memory_and_the_loop_s_turns() {
        let mut sampler = Sampler::default();
        let first = sampler.sample(std::process::id());
        let own = &first["processes"][0];
        assert_eq!(own["pid"], std::process::id());
        assert!(own["exe"].as_str().is_some_and(|exe| exe.ends_with(".exe") && !exe.contains('\\')), "{own}");
        assert!(own["working_mib"].as_f64().unwrap() > 0.0 && own["private_mib"].as_f64().unwrap() > 0.0, "{own}");
        assert!(own.get("cpu").is_none() && first.get("others").is_none(), "a rate needs two readings");
        assert!(first["machine"]["free_mib"].as_u64().unwrap() > 0);
        // Two turns, one of them long; then something to measure.
        loop_spent();
        drop(turn(Cow::Borrowed("short")));
        { let _long = turn(Cow::Borrowed("long")); std::thread::sleep(Duration::from_millis(30)); }
        let busy = Instant::now();
        while busy.elapsed() < Duration::from_millis(60) { std::hint::black_box(busy.elapsed()); }
        let second = sampler.sample(std::process::id());
        let spent = &second["loop"];
        assert_eq!((&spent["turns"], &spent["longest"]), (&json!(2), &json!("long")));
        assert!(spent["longest_ms"].as_f64().unwrap() >= 30.0 && spent["busy_ms"].as_f64().unwrap() >= spent["longest_ms"].as_f64().unwrap(), "{spent}");
        assert!(second["processes"][0]["cpu"].as_f64().unwrap() > 0.0, "{}", second["processes"][0]);
        assert!(second["machine"]["cpu"].as_f64().is_some_and(|share| (0.0..=1.0).contains(&share)), "{}", second["machine"]);
        assert_eq!(loop_spent()["turns"], 0, "asked, the count starts again");
        // A line over an idle second of this process: the machine's other load, if it is the larger part.
        std::thread::sleep(Duration::from_secs(1));
        let idle = sampler.sample(std::process::id());
        if let Some(others) = idle.get("others") {
            for named in others["processes"].as_array().unwrap() {
                assert!(named["exe"].as_str().is_some_and(|exe| !exe.contains('\\')) && named["cpu"].as_f64().unwrap() >= 0.0, "{named}");
            }
        }
        println!("a line: {}", json!({"machine": idle["machine"], "others": idle["others"]}));
        // What a sample costs, over many.
        let (count, started) = (100u32, Instant::now());
        for _ in 0..count { std::hint::black_box(sampler.sample(std::process::id())); }
        let each = started.elapsed() / count;
        let started = Instant::now();
        for _ in 0..count { std::hint::black_box(listed(&mut sampler.buffer)); }
        let listing = started.elapsed() / count;
        println!("one sample: {each:?}, of which the list of {} processes {listing:?} (a {} KiB list); {} bytes",
                 listed(&mut sampler.buffer).unwrap().len(), size_of_val(sampler.buffer.as_slice()) / 1024, idle.to_string().len());
    }
}
