//! installation.exe's window (zlc-desktop --installer), Installation@Zou lab.
//! It shows what this PC has of what the experiment needs -- the program (this
//! checkout's release of the engine), the plugins' Python environment and
//! instrument links, the SLM GPU, the FPGA tools and a developer's toolchain --
//! and runs the steps that install, update, warm, build or program them.
//! Every check and every step is one of the checkout's tools\*.ps1, the ones
//! bin\scripts\*.bat run, started without a console
//! window; a step's output streams into the window's log and this run's log
//! file (.local\logs, diagnostics.rs). Nothing here needs Python: before the
//! environment exists the window says so and offers to make it.
use std::collections::{BTreeMap, VecDeque};
use std::io::BufRead;
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tao::event::{Event, StartCause, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop, EventLoopProxy};
use wry::WebContext;

use crate::windows::{Kind, Setup, Windows};
use crate::{AppEvent, frame, job};

/// What the background work tells the window's turn.
pub enum Report {
    /// A check's JSON line, or why there is none (Online: the newest release on GitHub).
    Checked(Check, Result<Value, String>),
    /// A line the running step wrote.
    Line(String),
    /// The running step ended, with its exit code.
    Ended(Option<i32>),
}

/// The checks: each one script's status mode, nothing changed. Online asks GitHub (a short
/// timeout) and holds nothing back: it is not one of ALL, the checks a step waits for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Check { Program, Science, Fpga, Tools, Online }

impl Check {
    const ALL: [Check; 4] = [Check::Program, Check::Science, Check::Fpga, Check::Tools];
    fn script(self) -> (&'static str, &'static [&'static str]) {
        match self {
            Check::Program => ("fetch_engine.ps1", &["-Status"]),
            Check::Online => ("fetch_engine.ps1", &["-Latest"]),
            Check::Science => ("install_science.ps1", &["-Status"]),
            Check::Fpga => ("fpga.ps1", &["-Action", "status"]),
            Check::Tools => ("dev_setup.ps1", &["-Status"]),
        }
    }
}

/// How a step's end is said: v2's words for the FPGA and the warmup (their .bat files').
#[derive(Clone, Copy, PartialEq)]
enum Outcome { Install, Warm, Fpga, Estimate }

/// A step, and the component (check box) it installs, if any.
#[derive(Clone)]
struct Step { title: String, script: &'static str, arguments: Vec<String>, outcome: Outcome, component: Option<&'static str> }

impl Step {
    fn new(title: &str, script: &'static str, arguments: &[&str], outcome: Outcome) -> Step {
        Step { title: title.into(), script, arguments: arguments.iter().map(|text| (*text).to_owned()).collect(), outcome, component: None }
    }
    /// The line its end leaves in the log, and whether it succeeded.
    fn ended(&self, code: Option<i32>, stopped: bool) -> (String, bool) {
        let title = &self.title;
        match (self.outcome, code) {
            _ if stopped => (format!("{title} was stopped."), false),
            (Outcome::Estimate, Some(0)) => ("ZLC resource estimate: within the configured budget. Build and timing still require verification.".into(), true),
            (Outcome::Estimate, Some(1)) => ("ZLC resource estimate: INSUFFICIENT -- see the OVER BUDGET lines above.".into(), false),
            (Outcome::Estimate, code) => (format!("ZLC resource estimate failed with code {} -- read the messages above.", code.unwrap_or(-1)), false),
            (Outcome::Fpga, Some(0)) => (format!("ZLC {title} completed successfully."), true),
            (Outcome::Fpga, code) => (format!("ZLC {title} failed with code {}. Read the messages above.", code.unwrap_or(-1)), false),
            (_, Some(0)) => (format!("{title}: done."), true),
            (Outcome::Warm, _) => ("SLM GPU warmup failed -- see the message above.".into(), false),
            (Outcome::Install, code) => (format!("{title} failed with code {}. Read the messages above.", code.unwrap_or(-1)), false),
        }
    }
}

/// The FPGA steps by the page's names, with v2 build_and_program.bat's words for each.
fn fpga_step(name: &str, force: bool) -> Option<Step> {
    let (title, action) = match name {
        "fpga_estimate" => return Some(Step::new("resource estimate", "fpga.ps1", &["-Action", "estimate"], Outcome::Estimate)),
        "fpga_build" => ("build", "build"),
        "fpga_program" => ("program", "program"),
        "fpga_flash" => ("program SPI flash (persists across power cycle)", "flash"),
        "fpga_diagnose" => ("hardware diagnose", "diagnose"),
        _ => return None,
    };
    let mut step = Step::new(title, "fpga.ps1", &["-Action", action], Outcome::Fpga);
    if force && action == "build" { step.arguments.push("-ForceBuild".into()); }
    Some(step)
}

/// `powershell -File tools\<script> <arguments>` with no console window, its output piped.
fn powershell(script: &str, arguments: &[String]) -> Command {
    let mut command = Command::new("powershell.exe");
    command.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]).arg(zlc_core::checkout::path("tools").join(script)).args(arguments)
        // fpga.ps1 runs the `fpga` command of the pulse server beside this engine (not a build of the sources).
        .env("ZLC_PULSE_SERVER", std::env::current_exe().map(|exe| exe.with_file_name("zlc-pulse-server.exe")).unwrap_or_default())
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    command
}

/// A line a step wrote: UTF-8 when it is that, else the console code page Windows PowerShell and
/// Windows itself write a pipe in (a localized system message).
fn decode(bytes: &[u8]) -> String { decode_in(bytes, unsafe { windows_sys::Win32::Globalization::GetOEMCP() }) }

fn decode_in(bytes: &[u8], page: u32) -> String {
    if let Ok(text) = std::str::from_utf8(bytes) { return text.to_owned(); }
    use windows_sys::Win32::Globalization::MultiByteToWideChar;
    let Ok(length) = i32::try_from(bytes.len()) else { return String::from_utf8_lossy(bytes).into_owned() };
    let size = unsafe { MultiByteToWideChar(page, 0, bytes.as_ptr(), length, std::ptr::null_mut(), 0) };
    if size <= 0 { return String::from_utf8_lossy(bytes).into_owned(); }
    let mut wide = vec![0u16; size as usize];
    let size = unsafe { MultiByteToWideChar(page, 0, bytes.as_ptr(), length, wide.as_mut_ptr(), size) };
    String::from_utf16_lossy(&wide[..size.max(0) as usize])
}

/// A check's answer: the last JSON line it printed, else why it printed none.
fn check(which: Check) -> Result<Value, String> {
    let (script, arguments) = which.script();
    let arguments: Vec<String> = arguments.iter().map(|text| (*text).to_owned()).collect();
    let child = powershell(script, &arguments).spawn().map_err(|error| format!("Cannot run tools\\{script}: {error}"))?;
    let _job = job::limit(&child, None);
    let output = child.wait_with_output().map_err(|error| error.to_string())?;
    let text = decode(&output.stdout);
    if let Some(answer) = text.lines().rev().map(str::trim).find(|line| line.starts_with('{') || line.starts_with('[')) {
        if let Ok(value) = serde_json::from_str(answer) { return Ok(value); }
    }
    let errors = decode(&output.stderr);
    let lines: Vec<&str> = text.lines().chain(errors.lines()).map(str::trim).filter(|line| !line.is_empty()).collect();
    Err(lines.iter().rev().find(|line| line.starts_with("ERROR:")).or(lines.last()).map(|line| line.trim_start_matches("ERROR:").trim().to_owned())
        .unwrap_or_else(|| format!("tools\\{script} answered nothing (code {:?})", output.status.code())))
}

/// The components a check box installs: the rows its step completes (its own first), and whether
/// every PC needs it (the others are optional).
const SELECTABLE: [(&str, &[&str], bool); 5] =
    [("program", &["program"], true), ("environment", &["environment", "links"], true), ("pylon", &["pylon"], false), ("slm_gpu", &["slm_gpu", "warm"], false), ("tools", &["tools"], false)];

/// Whether a component is missing or broken -- a row of it is missing, its last step failed, or
/// the check of one every PC needs failed: its check box starts checked, and Fill in missing
/// installs exactly these. An optional part that is not there is neither.
fn needed(rows: &[Value], id: &str) -> bool {
    let Some((_, part, required)) = SELECTABLE.iter().find(|(name, ..)| *name == id) else { return false };
    rows.iter().filter(|row| part.iter().any(|name| row["id"] == *name))
        .any(|row| match row["state"].as_str() { Some("missing") => true, Some("error") => *required || row["failed"] == true, _ => false })
}

pub struct Installer {
    proxy: EventLoopProxy<AppEvent>,
    checks: [Option<Result<Value, String>>; 4],
    checking: usize,
    online: Option<Result<Value, String>>,
    /// The check boxes the operator changed since a step of theirs last ended well: the others follow the status.
    picked: BTreeMap<&'static str, bool>,
    /// The components whose last step failed, with its end line: broken whatever their package
    /// metadata says, until a step of theirs ends well.
    failed: BTreeMap<&'static str, String>,
    plan_only: bool,
    force_build: bool,
    queue: VecDeque<Step>,
    /// The step under way and the job holding its processes (dropped: they end).
    running: Option<(Step, Option<job::Limited>)>,
    stopped: bool,
    /// The last step's end: its line and whether it succeeded.
    last: Option<(String, bool)>,
    log: Vec<String>,
    /// How many log lines the page holds, and when the next ones go to it.
    sent: usize,
    flush_at: Option<Instant>,
    /// Counts the closes asked while a step runs: the page asks the operator each time.
    close_asked: u64,
    log_file: Option<String>,
    /// The operator's steps include the program: once they all end well and the check finds the
    /// new engine waiting (this window runs from bin\engine), the window starts again by itself.
    restart_after: bool,
}

/// The log the page keeps; past it, the lines are in the log file alone.
const LOG_LINES: usize = 20_000;

impl Installer {
    fn new(proxy: EventLoopProxy<AppEvent>, log_file: Option<String>) -> Installer {
        Installer { proxy, checks: [None, None, None, None], checking: 0, online: None, picked: BTreeMap::new(), failed: BTreeMap::new(), plan_only: false, force_build: false, queue: VecDeque::new(),
                    running: None, stopped: false, last: None, log: Vec::new(), sent: 0, flush_at: None, close_asked: 0, log_file, restart_after: false }
    }

    /// Whether the window should start again now: the operator's program update ended well, every
    /// check has answered, and the new engine waits in bin\engine.next for this one to end.
    fn restart_due(&mut self) -> bool {
        let staged = matches!(self.answer(Check::Program), Some(Ok(program)) if program["relation"] == "staged");
        let done = self.running.is_none() && self.queue.is_empty() && self.checking == 0;
        if !(self.restart_after && done) { return false; }
        self.restart_after = false;
        staged && self.last.as_ref().is_some_and(|(_, ok)| *ok)
    }

    fn check_all(&mut self) {
        if self.checking > 0 || self.running.is_some() { return; }
        for which in Check::ALL.into_iter().chain([Check::Online]) {
            if which != Check::Online { self.checking += 1; }
            let proxy = self.proxy.clone();
            std::thread::spawn(move || { let _ = proxy.send_event(AppEvent::Installer(Report::Checked(which, check(which)))); });
        }
    }

    fn answer(&self, which: Check) -> Option<&Result<Value, String>> { self.checks[Check::ALL.iter().position(|item| *item == which).unwrap()].as_ref() }

    /// Every component: its card, words, state (ok, missing, optional, error, checking) and whether it is
    /// selected (as the operator checked it, else whether it is needed).
    fn components(&self) -> Vec<Value> {
        let mut rows = self.rows();
        let selected: Vec<Option<bool>> = rows.iter().map(|row| (row["selectable"] == true).then(|| {
            let id = row["id"].as_str().unwrap_or("");
            self.picked.get(id).copied().unwrap_or_else(|| needed(&rows, id))
        })).collect();
        for (row, selected) in rows.iter_mut().zip(selected) { if let Some(on) = selected { row["selected"] = json!(on); } }
        rows
    }

    /// Every component's row, from the checks' answers.
    fn rows(&self) -> Vec<Value> {
        let row = |id: &str, card: &str, title: &str, state: &str, detail: String| {
            let selectable = SELECTABLE.iter().any(|(name, ..)| *name == id);
            // What its state is called: a package is installed, Vivado found, a bitstream built.
            let words = match (id, state) {
                (_, "checking") => "Checking…", (_, "error") => "Error",
                ("warm", "ok") => "Warmed", ("warm", _) => "Not warmed",
                ("vivado", "ok") => "Found", ("vivado", _) => "Not found",
                ("board", _) => "Valid",
                ("bitstream", "ok") => "Built", ("bitstream", _) => "Not built",
                (_, "ok") => "Installed", (_, "missing") => "Missing", _ => "Not installed",
            };
            json!({"id": id, "card": card, "title": title, "state": state, "words": words, "detail": detail, "selectable": selectable, "selected": false})
        };
        let mut rows = Vec::new();
        // The program: bin\engine against the release this checkout names (bin\release.json).
        rows.push(match self.answer(Check::Program) {
            None => row("program", "program", "Zou lab control", "checking", "Checking…".into()),
            Some(Err(error)) => row("program", "program", "Zou lab control", "error", error.clone()),
            Some(Ok(program)) => {
                let named = |facts: &Value| format!("v{} (commit {})", facts["version"].as_str().unwrap_or("?"),
                                                    facts["commit"].as_str().unwrap_or("?").chars().take(7).collect::<String>());
                let (have, wanted) = (named(&program["installed"]), format!("v{}", program["expected"]["version"].as_str().unwrap_or("?")));
                let size = program["expected"]["asset"]["size"].as_f64().unwrap_or(0.0) / 1_048_576.0;
                let (mut state, mut words, mut detail) = match program["relation"].as_str().unwrap_or("") {
                    "match" => ("ok", "Installed", format!("{have}: this checkout's release")),
                    "built" => ("ok", "Built here", "Built on this PC from this checkout's sources".into()),
                    "staged" => ("optional", "Restart", format!("{wanted} is downloaded: it goes in place when installation.exe starts again")),
                    "missing" => ("missing", "Missing", format!("Not installed: Install downloads this checkout's release {wanted} ({size:.0} MB)")),
                    "older" => ("missing", "Update", format!("{have} is older than this checkout's release {wanted}: Update downloads it")),
                    "newer" => ("missing", "Newer", format!("{have} is newer than this checkout's release {wanted}: pull the repository (or Update to go back)")),
                    "unreleased" => ("optional", "Not released", "This checkout names no release (bin\\release.json): build_release makes one".into()),
                    _ => ("missing", "Update", format!("{have} is not this checkout's release {wanted}: Update downloads it")),
                };
                // GitHub ahead of this checkout: Update pulls it (a fast-forward) and then installs the
                // release it names; when it cannot pull, the words say why and the pull is the operator's.
                if let Some(Ok(latest)) = &self.online {
                    let (pull, release) = (&latest["pull"], latest["tag"].as_str().filter(|_| latest["newer"] == true));
                    let behind = pull["behind"].as_u64().unwrap_or(0);
                    let release = release.map(|tag| format!(", release {tag}")).unwrap_or_default();
                    if pull["pullable"] == true {
                        (state, words) = ("missing", "Update");
                        detail = format!("{have} · GitHub has {behind} newer commit(s){release}: Update pulls them and installs the program they name");
                    } else if behind > 0 || !release.is_empty() {
                        if state == "ok" { words = "Newer online"; }
                        let why = pull["why"].as_str().unwrap_or("pull the repository to get it");
                        detail.push_str(&format!(" · GitHub is ahead{release}: {why}"));
                    }
                }
                let mut shown = row("program", "program", "Zou lab control", state, detail);
                shown["words"] = json!(words);
                shown
            }
        });
        // The Python environment and what each part of it lacks.
        match self.answer(Check::Science) {
            None => for (id, card, title) in [("environment", "python", "Science environment"), ("links", "instruments", "Serial, VISA and USB links"),
                                              ("pylon", "instruments", "Basler pylon"), ("slm_gpu", "gpu", "CuPy and CUDA"), ("warm", "gpu", "Warmed on this GPU")] {
                rows.push(row(id, card, title, "checking", "Checking…".into()));
            },
            Some(Err(error)) => {
                rows.push(row("environment", "python", "Science environment", "error", error.clone()));
                rows.push(row("links", "instruments", "Serial, VISA and USB links", "missing", "Installed with the science environment".into()));
                rows.push(row("pylon", "instruments", "Basler pylon", "optional", "Needs the science environment".into()));
                rows.push(row("slm_gpu", "gpu", "CuPy and CUDA", "optional", "Needs the science environment".into()));
                rows.push(row("warm", "gpu", "Warmed on this GPU", "optional", "Needs CuPy and CUDA".into()));
            }
            Some(Ok(science)) => {
                let exists = science["exists"] == true;
                // What a part lacks is read in the environment, or (none yet) in the base Python a new
                // one would use; a new one with packages of its own has none of them yet.
                let usable = exists || science["inherit"] == true;
                let base = if exists { "" } else { ": in the base Python" };
                let text = |key: &str| science[key].as_str().unwrap_or("").to_owned();
                let unmet = |part: &str| -> Vec<String> { science["unmet"][part].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect() };
                let lacks = |parts: &[&str]| format!("Unmet: {}", parts.iter().flat_map(|part| unmet(part)).collect::<Vec<_>>().join("; "));
                // The environment row reads what its step installs, the instrument links with it:
                // its check box never stands checked beside "Installed".
                // No venv: a base Python with everything is the environment, as v2 runs on it (the plugins run there too).
                let complete = usable && unmet("core").is_empty() && unmet("devices").is_empty();
                let (state, detail) = if !exists && complete {
                    ("ok", format!("Python {} · uses the base Python {} (found through {})", text("version"), text("python"), text("source")))
                } else if !exists {
                    let reuse = if science["inherit"] == true { "; its numpy, scipy and Pillow are used" } else { "" };
                    ("missing", format!("Not installed: made from Python {} at {} (found through {}){reuse}", text("version"), text("python"), text("source")))
                } else if complete { ("ok", format!("Python {} · {}", text("version"), text("environment"))) }
                else { ("missing", lacks(&["core", "devices"])) };
                rows.push(row("environment", "python", "Science environment", state, detail));
                let links = "pyserial, PyVISA, PyVISA-py, pyusb, libusb-package";
                rows.push(if !usable { row("links", "instruments", "Serial, VISA and USB links", "missing", format!("{links}: installed with the science environment")) }
                          else if unmet("devices").is_empty() { row("links", "instruments", "Serial, VISA and USB links", "ok", format!("{links}{base}")) }
                          else { row("links", "instruments", "Serial, VISA and USB links", "missing", lacks(&["devices"])) });
                rows.push(if usable && unmet("pylon").is_empty() { row("pylon", "instruments", "Basler pylon", "ok", format!("pypylon{base}")) }
                          else { row("pylon", "instruments", "Basler pylon", "optional", "pypylon, about 270 MB: the PC with a Basler camera".into()) });
                let gpu = usable && unmet("slm-gpu").is_empty();
                rows.push(if gpu { row("slm_gpu", "gpu", "CuPy and CUDA", "ok", format!("CuPy and the CUDA runtime for the SLM solver{base}")) }
                          else { row("slm_gpu", "gpu", "CuPy and CUDA", "optional", "The PC that drives the SLM: CuPy and the CUDA runtime, then warmed".into()) });
                let warmed = zlc_core::checkout::local("slm_gpu_cache").join("analysis.json");
                let when = std::fs::metadata(&warmed).and_then(|meta| meta.modified()).ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok()).map(|age| age.as_millis() as u64);
                let mut warm = match (gpu, when) {
                    (true, Some(_)) => row("warm", "gpu", "Warmed on this GPU", "ok", warmed.display().to_string()),
                    (true, None) => row("warm", "gpu", "Warmed on this GPU", "missing", "Not yet: the first SLM preparation would measure it".into()),
                    (false, _) => row("warm", "gpu", "Warmed on this GPU", "optional", "After CuPy and CUDA".into()),
                };
                if let Some(when) = when.filter(|_| gpu) { warm["time"] = json!(when); }
                rows.push(warm);
            }
        }
        // The FPGA tools: the Vivado a build runs, the configured part, the bitstream.
        match self.answer(Check::Fpga) {
            None => for (id, title) in [("vivado", "Vivado"), ("board", "Board configuration"), ("bitstream", "Bitstream")] { rows.push(row(id, "fpga", title, "checking", "Checking…".into())); },
            Some(Err(error)) => rows.push(row("vivado", "fpga", "Vivado", "error", error.clone())),
            Some(Ok(fpga)) => {
                rows.push(match fpga["vivado"].as_str() {
                    Some(path) => row("vivado", "fpga", "Vivado", "ok", match fpga["version"].as_str() { Some(version) => format!("Vivado {version} · {path}"), None => path.into() }),
                    None => row("vivado", "fpga", "Vivado", "optional", "Not found: build, program, flash and diagnose need it (set ZLC_PS_VIVADO_BIN)".into()),
                });
                let config = fpga["config"].as_str().unwrap_or("");
                rows.push(match (fpga["part"].as_str(), fpga["config_error"].as_str()) {
                    (Some(part), _) => row("board", "fpga", "Board configuration", "ok", format!("{part} · {config}")),
                    (None, error) => row("board", "fpga", "Board configuration", "error", error.unwrap_or("The board configuration is refused").into()),
                });
                // Against the sources by the build's own receipt (tools\fpga.ps1): what Build would skip or redo.
                let built = fpga["bitstream_built"].as_str().unwrap_or("?");
                rows.push(match (fpga["bitstream"].as_str(), fpga["bitstream_current"].as_bool()) {
                    (None, _) => row("bitstream", "fpga", "Bitstream", "optional", "Not built on this PC".into()),
                    (Some(bit), Some(true)) => row("bitstream", "fpga", "Bitstream", "ok", format!("Matches the sources (built {built}) · {bit}")),
                    (Some(bit), Some(false)) => {
                        let mut stale = row("bitstream", "fpga", "Bitstream", "missing", format!("Older than the sources (built {built}): Build makes it again · {bit}"));
                        stale["words"] = json!("Out of date");
                        stale
                    }
                    (Some(bit), None) => row("bitstream", "fpga", "Bitstream", "ok", format!("Built {built}; not compared with the sources (no build receipt or Vivado here) · {bit}")),
                });
            }
        }
        // A development machine's toolchain (tools\dev_setup.ps1's steps).
        rows.push(match self.answer(Check::Tools) {
            None => row("tools", "tools", "Development toolchain", "checking", "Checking…".into()),
            Some(Err(error)) => row("tools", "tools", "Development toolchain", "error", error.clone()),
            Some(Ok(steps)) => {
                let steps = steps.as_array().cloned().unwrap_or_default();
                let missing: Vec<&str> = steps.iter().filter(|step| step["present"] != true).filter_map(|step| step["name"].as_str()).collect();
                if missing.is_empty() { row("tools", "tools", "Development toolchain", "ok", format!("All {} present: builds this checkout's sources", steps.len())) }
                else { row("tools", "tools", "Development toolchain", "optional", format!("Missing: {}", missing.join("; "))) }
            }
        });
        // A component whose last step failed says so on its first row not in place, else its own.
        for (id, line) in &self.failed {
            let Some((_, part, _)) = SELECTABLE.iter().find(|(name, ..)| name == id) else { continue };
            let at = rows.iter().position(|row| part.iter().any(|name| row["id"] == *name) && matches!(row["state"].as_str(), Some("missing" | "optional")))
                .or_else(|| rows.iter().position(|row| row["id"] == *id && row["state"] == "ok"));
            if let Some(row) = at.map(|at| &mut rows[at]) {
                for (key, value) in [("state", "error"), ("words", "Failed"), ("detail", line.as_str())] { row[key] = json!(value); }
                row["failed"] = json!(true);
            }
        }
        rows
    }

    /// Install selected (or, `missing_only`, what is missing or broken), in the order they build on.
    fn plan(&self, missing_only: bool) -> Vec<Step> {
        let rows = self.components();
        let wanted = |id: &str| if missing_only { needed(&rows, id) } else { rows.iter().any(|row| row["id"] == id && row["selected"] == true) };
        let dry = |arguments: &[&str]| -> Vec<String> { arguments.iter().map(|text| (*text).to_owned()).chain(self.plan_only.then(|| "-DryRun".to_owned())).collect() };
        let named = |title: &str| if self.plan_only { format!("{title} (plan only)") } else { title.to_owned() };
        let step = |component: &'static str, title: &str, script: &'static str, arguments: &[&str]|
            Step { title: named(title), script, arguments: dry(arguments), outcome: Outcome::Install, component: Some(component) };
        let mut steps = Vec::new();
        if wanted("program") {
            // A checkout behind GitHub is brought up first: the program step then reads the release it names.
            if matches!(&self.online, Some(Ok(latest)) if latest["pull"]["pullable"] == true) {
                steps.push(step("program", "Pull this checkout from GitHub", "fetch_engine.ps1", &["-Pull"]));
            }
            steps.push(step("program", "Program (this checkout's release)", "fetch_engine.ps1", &[]));
        }
        // Pylon and the GPU install into the environment (a venv, using the base's packages when it
        // can): one whose base lacks numpy, scipy or Pillow is made first.
        let usable = matches!(self.answer(Check::Science), Some(Ok(science)) if science["exists"] == true || science["inherit"] == true);
        if wanted("environment") || (!usable && (wanted("pylon") || wanted("slm_gpu"))) {
            steps.push(step("environment", "Science environment", "install_science.ps1", &["-Group", "runtime"]));
        }
        if wanted("pylon") { steps.push(step("pylon", "Basler pylon", "install_science.ps1", &["-Group", "pylon"])); }
        if wanted("slm_gpu") {
            // Installed but not yet warmed: the warmup alone (the install warms after installing).
            let installed = rows.iter().any(|row| row["id"] == "slm_gpu" && row["state"] == "ok");
            steps.push(if missing_only && installed { Step { component: Some("slm_gpu"), ..Step::new("SLM GPU warmup", "warm_slm_gpu.ps1", &[], Outcome::Warm) } }
                       else { step("slm_gpu", "SLM GPU (CuPy and CUDA, then warmed)", "install_science.ps1", &["-Group", "slm-gpu"]) });
        }
        if wanted("tools") { steps.push(step("tools", "Development toolchain", "dev_setup.ps1", &[])); }
        steps
    }

    fn state(&self) -> Value {
        let running = self.running.as_ref().map(|(step, _)| json!({"title": step.title, "stopping": self.stopped}));
        json!({"type": "state", "installation": {
            "components": self.components(), "checking": self.checking > 0, "running": running,
            // The engine this window runs from was replaced: it starts again to run the new one.
            "restart": matches!(self.answer(Check::Program), Some(Ok(program)) if program["relation"] == "staged"),
            "queue": self.queue.iter().map(|step| step.title.clone()).collect::<Vec<_>>(),
            "plan_only": self.plan_only, "force_build": self.force_build, "close_asked": self.close_asked, "log_file": self.log_file,
            "last": self.last.as_ref().map(|(text, ok)| json!({"text": text, "ok": ok})),
        }})
    }

    fn line(&mut self, text: String) {
        println!("{text}");
        if self.log.len() < LOG_LINES { self.log.push(text); }
        else if self.log.len() == LOG_LINES { self.log.push(format!("… the rest is in the log file{}", self.log_file.as_ref().map(|path| format!(" ({path})")).unwrap_or_default())); }
        self.flush_at.get_or_insert_with(|| Instant::now() + Duration::from_millis(50));
    }

    /// The lines the page does not hold yet.
    fn flush(&mut self, windows: &Windows) -> Result<(), String> {
        self.flush_at = None;
        if self.sent >= self.log.len() { return Ok(()); }
        let message = json!({"type": "state", "installation_log": {"from": self.sent, "lines": &self.log[self.sent..]}});
        self.sent = self.log.len();
        windows.send_to(&Kind::Installer, &message)
    }

    /// Start the next queued step (none when the queue is done: then everything is checked again).
    fn next(&mut self) {
        let Some(step) = self.queue.pop_front() else { self.stopped = false; self.check_all(); return; };
        self.line(String::new());
        self.line(format!("== {} ==", step.title));
        self.line(format!("tools\\{} {}", step.script, step.arguments.join(" ")).trim_end().to_owned());
        if self.plan_only && step.outcome == Outcome::Warm {
            self.line("Plan only: the warmup measures this GPU; nothing is run.".into());
            return self.next();
        }
        let child = match powershell(step.script, &step.arguments).spawn() {
            Ok(child) => child,
            Err(error) => {
                self.line(format!("Cannot run tools\\{}: {error}", step.script));
                self.last = Some((format!("{} did not start.", step.title), false));
                self.queue.clear();
                return self.next();
            }
        };
        let job = job::limit(&child, None);
        let mut child = child;
        for stream in [child.stdout.take().map(|out| Box::new(out) as Box<dyn std::io::Read + Send>),
                       child.stderr.take().map(|err| Box::new(err) as Box<dyn std::io::Read + Send>)].into_iter().flatten() {
            let proxy = self.proxy.clone();
            std::thread::spawn(move || {
                for line in std::io::BufReader::new(stream).split(b'\n').map_while(Result::ok) {
                    let text = decode(&line).trim_end().to_owned();
                    if proxy.send_event(AppEvent::Installer(Report::Line(text))).is_err() { return; }
                }
            });
        }
        let proxy = self.proxy.clone();
        std::thread::spawn(move || { let code = child.wait().ok().and_then(|status| status.code()); let _ = proxy.send_event(AppEvent::Installer(Report::Ended(code))); });
        self.running = Some((step, job));
    }

    fn start(&mut self, steps: Vec<Step>) -> Result<(), String> {
        if self.running.is_some() { return Err("A step is under way: wait for it or stop it".into()); }
        // What to do, and what a step's end is checked against, is the status: a check under way decides it.
        if self.checking > 0 { return Err("Checking this PC: wait for the check to finish".into()); }
        self.restart_after = !self.plan_only && steps.iter().any(|step| step.component == Some("program"));
        self.queue = steps.into();
        self.last = None;
        self.next();
        Ok(())
    }

    /// End the running step and what was queued after it (its processes end with their job).
    fn stop(&mut self) {
        self.queue.clear();
        if let Some((_, job)) = self.running.as_mut() { self.stopped = true; drop(job.take()); }
    }

    fn report(&mut self, report: Report) {
        match report {
            Report::Checked(Check::Online, answer) => self.online = Some(answer),
            Report::Checked(which, answer) => {
                self.checking = self.checking.saturating_sub(1);
                self.checks[Check::ALL.iter().position(|item| *item == which).unwrap()] = Some(answer);
            }
            Report::Line(text) => self.line(text),
            Report::Ended(code) => {
                let Some((step, _job)) = self.running.take() else { return };
                let (words, ok) = step.ended(code, self.stopped);
                // A component whose step ended well follows the status again; one whose step failed
                // or was stopped keeps the operator's pick and reads broken. A plan changed nothing.
                if let Some(id) = step.component.filter(|_| !self.plan_only) {
                    if ok { self.picked.remove(id); self.failed.remove(id); } else { self.failed.insert(id, words.clone()); }
                }
                self.line(words.clone());
                self.last = Some((words, ok));
                if !ok { self.queue.clear(); }
                self.stopped = false;
                self.next();
            }
        }
    }

    /// A command from the page; its answer is the state.
    fn command(&mut self, command: &Value) -> Result<(), String> {
        let flag = |name: &str| command[name].as_bool().ok_or_else(|| format!("{name} is true or false"));
        match command["action"].as_str().unwrap_or("") {
            "installation_check" => { if self.running.is_some() { return Err("A step is under way".into()); } self.check_all(); }
            "installation_select" => {
                let id = command["component"].as_str().unwrap_or("");
                let id = SELECTABLE.iter().map(|(name, ..)| *name).find(|name| *name == id).ok_or_else(|| format!("{id:?} is not a component to install"))?;
                self.picked.insert(id, flag("selected")?);
            }
            "installation_option" => match command["name"].as_str() {
                Some("plan_only") => self.plan_only = flag("on")?,
                Some("force_build") => self.force_build = flag("on")?,
                other => return Err(format!("No option {other:?}")),
            },
            "installation_install" => {
                let missing = command["missing"] == true;
                let steps = self.plan(missing);
                if steps.is_empty() && self.checking == 0 { return Err((if missing { "Nothing is missing or broken" } else { "Nothing is selected" }).into()); }
                self.start(steps)?;
            }
            "installation_step" => {
                let name = command["step"].as_str().unwrap_or("");
                let step = if name == "rewarm" { Some(Step::new("SLM GPU warmup", "warm_slm_gpu.ps1", &[], Outcome::Warm)) } else { fpga_step(name, self.force_build) };
                self.start(vec![step.ok_or_else(|| format!("No step {name:?}"))?])?;
            }
            "installation_stop" => self.stop(),
            other => return Err(format!("{other} is not a command of this window")),
        }
        Ok(())
    }
}

/// installation.exe: the one window, until it closes.
pub fn run(event_loop: EventLoop<AppEvent>, context: WebContext, setup: Setup, log_file: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let proxy = event_loop.create_proxy();
    let mut windows = Windows::new(context, setup, proxy.clone());
    windows.make(&Kind::Installer, &event_loop, true)?;
    let mut installer = Installer::new(proxy, log_file);
    installer.check_all();
    event_loop.run(move |event, _target, flow| {
        let mut request = Value::Null;
        let result = (|| -> Result<(), String> {
            match event {
                Event::NewEvents(StartCause::ResumeTimeReached { .. }) => installer.flush(&windows)?,
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    // A step under way is the operator's to stop: the page asks first.
                    if installer.running.is_none() { *flow = ControlFlow::Exit; return Ok(()); }
                    installer.close_asked += 1;
                    windows.send_to(&Kind::Installer, &installer.state())?;
                }
                Event::UserEvent(AppEvent::Installer(report)) => {
                    let ended = matches!(report, Report::Ended(_) | Report::Checked(..));
                    installer.report(report);
                    // A program update that left the new engine waiting: installation.exe again puts it in place.
                    if installer.restart_due() {
                        installer.line("The new program goes in place: installation.exe starts again.".into());
                        installer.flush(&windows)?;
                        Command::new(zlc_core::checkout::path("installation.exe")).spawn().map_err(|error| format!("Cannot start installation.exe: {error}"))?;
                        *flow = ControlFlow::Exit;
                        return Ok(());
                    }
                    // A step that writes without pause still shows its lines every 50 ms.
                    if ended || installer.flush_at.is_some_and(|at| Instant::now() >= at) { installer.flush(&windows)?; }
                    if ended { windows.send_to(&Kind::Installer, &installer.state())?; }
                }
                Event::UserEvent(AppEvent::Command { window: Some(kind), command }) => {
                    request = command["id"].clone();
                    match command["action"].as_str().unwrap_or("") {
                        "ready" => {
                            // The page is (re)loaded: it is given the state, its title bar and the whole log.
                            if let Some(window) = windows.get_mut(&kind) { window.ready = true; }
                            windows.page_loaded(&kind);
                            installer.sent = 0;
                            windows.send_to(&kind, &installer.state())?;
                            if let Some(caption) = windows.get(&kind).and_then(|shown| frame::caption(&shown.window)) { windows.send_to(&kind, &caption)?; }
                            installer.flush(&windows)?;
                        }
                        "installation_close" => { installer.stop(); *flow = ControlFlow::Exit; }
                        // installation.exe again: its launcher puts the downloaded engine in place once this one has ended.
                        "installation_restart" => {
                            Command::new(zlc_core::checkout::path("installation.exe")).spawn().map_err(|error| format!("Cannot start installation.exe: {error}"))?;
                            *flow = ControlFlow::Exit;
                        }
                        "frontend_error" => eprintln!("{}", json!({"type": "frontend-error", "error": command["error"]})),
                        "map_ready" | "frame_consumed" | "mappings_released" => {}
                        _ => {
                            let answer = installer.command(&command);
                            installer.flush(&windows)?;
                            let mut state = installer.state();
                            state["request"] = request.clone();
                            state["error"] = answer.err().map_or(Value::Null, Value::String);
                            windows.send_to(&kind, &state)?;
                        }
                    }
                }
                Event::UserEvent(AppEvent::InvalidMessage(error)) => return Err(format!("Invalid UI command: {error}")),
                Event::UserEvent(AppEvent::Title { window, title }) => if let Some(shown) = windows.get(&window) { shown.window.set_title(&title); },
                Event::UserEvent(AppEvent::Caption { window, message }) => windows.send_to(&window, &message)?,
                Event::UserEvent(AppEvent::Activate(window)) => windows.activate(&window),
                _ => {}
            }
            Ok(())
        })();
        if let Err(error) = result {
            eprintln!("{}", json!({"type": "desktop-error", "error": error}));
            let _ = windows.send_to(&Kind::Installer, &json!({"type": "state", "request": request, "error": error}));
        }
        if *flow != ControlFlow::Exit {
            *flow = match installer.flush_at { Some(at) => ControlFlow::WaitUntil(at), None => ControlFlow::Wait };
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_component_is_needed_when_a_row_of_it_is_missing_or_a_required_check_failed() {
        let rows = |states: &[(&str, &str)]| -> Vec<serde_json::Value> { states.iter().map(|(id, state)| serde_json::json!({"id": id, "state": state})).collect() };
        let lab = rows(&[("environment", "ok"), ("links", "ok"), ("pylon", "ok"), ("slm_gpu", "optional"), ("warm", "optional"), ("tools", "error")]);
        assert!(["environment", "pylon", "slm_gpu", "tools"].iter().all(|id| !super::needed(&lab, id)));
        let broken = rows(&[("environment", "ok"), ("links", "missing"), ("slm_gpu", "ok"), ("warm", "missing"), ("tools", "optional")]);
        assert!(super::needed(&broken, "environment") && super::needed(&broken, "slm_gpu") && !super::needed(&broken, "tools"));
        assert!(super::needed(&rows(&[("environment", "error")]), "environment"));
        // An optional component is broken once its own step failed.
        let mut pylon = rows(&[("pylon", "error")]);
        assert!(!super::needed(&pylon, "pylon"));
        pylon[0]["failed"] = serde_json::json!(true);
        assert!(super::needed(&pylon, "pylon"));
    }

    #[test]
    fn a_line_is_utf8_or_the_console_code_page() {
        assert_eq!(super::decode_in("ZLC 完成".as_bytes(), 936), "ZLC 完成");
        // cmd's "cannot find" on a Chinese Windows: GBK, the console's code page 936.
        assert_eq!(super::decode_in(&[0xD5, 0xD2, 0xB2, 0xBB, 0xB5, 0xBD], 936), "找不到");
    }
}
