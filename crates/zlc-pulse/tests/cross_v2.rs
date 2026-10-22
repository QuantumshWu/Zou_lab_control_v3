//! The remote protocol against v2 itself, both ways, on memory boards only:
//! this crate's client drives v2's `pulse_server --backend memory`, and v2's
//! `RemotePulseStreamer` drives `zlc-pulse-server --backend memory`.
//!
//! Env-gated: `ZLC_PULSE_V2` = the v2 checkout (its interpreter is `python`
//! on PATH, as v2 installs itself); `ZLC_PULSE_CAPPED` = `capped.py`, the
//! memory-capped launcher every Python process here runs under.
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use zlc_pulse::binding::resolve_api_parameters;
use zlc_pulse::codec::read_pulse_document;
use zlc_pulse::compile::compile_sequence;
use zlc_pulse::remote::RemotePulseStreamer;
use zlc_pulse::wire::STATUS_RUNNING;

fn environment() -> Option<(PathBuf, PathBuf)> {
    Some((PathBuf::from(std::env::var_os("ZLC_PULSE_V2")?), PathBuf::from(std::env::var_os("ZLC_PULSE_CAPPED")?)))
}

fn python(capped: &PathBuf, args: &[&str]) -> Command {
    let mut command = Command::new("python");
    command.arg(capped).args(["2000", "120", "python"]).args(args).env("PYTHONDONTWRITEBYTECODE", "1");
    command
}

struct Killed(Child);
impl Drop for Killed { fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); } }

fn free_port() -> u16 { std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port() }

#[test]
fn this_client_drives_a_v2_pulse_server() {
    let Some((v2, capped)) = environment() else { return };
    let port = free_port();
    let code = format!("import sys; sys.path.insert(0, r'{}'); import zou_lab_control; from zlc_pulse.remote import _main; \
                        sys.exit(_main(['--backend', 'memory', '--host', '127.0.0.1', '--port', '{port}']))", v2.display());
    let mut child = python(&capped, &["-u", "-c", &code]).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
    let stdout = child.stdout.take().unwrap();
    let server = Killed(child);
    let mut lines = BufReader::new(stdout).lines();
    loop {
        let line = lines.next().expect("the v2 server ended before READY").unwrap();
        if line.contains("ZLC READY") { break; }
    }
    std::thread::spawn(move || for _ in lines {});
    let client = RemotePulseStreamer::new("127.0.0.1", port, None, Duration::from_secs(5)).unwrap();
    client.open().unwrap();
    let description = client.describe().unwrap();
    assert_eq!(description.target.raw_lanes.len(), 69);
    let (sequence, _) = read_pulse_document(&v2.join("packages/zlc_atom/tests/pulses/imaging_template.json")).unwrap();
    let sequence = Arc::new(resolve_api_parameters(&sequence, None).unwrap());
    let program = Arc::new(compile_sequence(&sequence, &description.geometry, description.clock_hz).unwrap());
    client.load(program.clone(), Some(sequence.clone()), Vec::new()).unwrap();
    let applied = client.fire(2, 1).unwrap();
    assert_eq!(applied.run_repeats, 2);
    // v2's memory board plays forever: the wait answers "still pending".
    assert!(client.wait_done(Some(Duration::from_millis(300)), None).unwrap().is_none());
    let snapshot = client.snapshot().unwrap();
    assert_eq!(snapshot.get("firing").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(snapshot.get("applied_digest").and_then(|v| v.as_str()), Some(program.digest().as_str()), "v2 digests what this side compiled");
    assert_eq!(snapshot.get("status").and_then(|v| v.as_int()), Some(STATUS_RUNNING.into()));
    let held = client.applied().unwrap().expect("an application");
    assert_eq!(*held.program, *program, "the program survives the round trip through v2 unchanged");
    assert_eq!(held.source.as_deref(), Some(&*sequence));
    assert!(client.safe().unwrap().stable());
    client.close().unwrap();
    drop(server);
}

const V2_CLIENT: &str = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
import zou_lab_control
from zlc_pulse import compile_sequence, read_pulse_document, resolve_api_parameters
from zlc_pulse.remote import RemotePulseStreamer
client = RemotePulseStreamer("127.0.0.1", int(sys.argv[2]))
client.open()
board = client.describe()
sequence, _ = read_pulse_document(sys.argv[1] + "/packages/zlc_atom/tests/pulses/imaging_template.json")
sequence = resolve_api_parameters(sequence)
program = compile_sequence(sequence, board.geometry, board.clock_hz)
client.load(program, source=sequence)
applied = client.fire(run_repeats=3, scan_repeats=1)
report = client.wait_done(5.0)
held = client.applied()
state = client.snapshot()
safe = client.safe()
client.close()
print(json.dumps({"lanes": len(board.target.raw_lanes), "pins": len(board.target.package_pins), "layout": board.layout_fingerprint,
                  "run": applied.run_repeats, "status": report.status, "fault": report.fault, "command": report.command_id,
                  "same_program": held.program == program, "same_source": held.source == sequence, "digest": state["applied_digest"],
                  "expected_digest": program.digest, "safe": safe.stable}))
"#;

#[test]
fn a_v2_client_drives_this_server() {
    let Some((v2, capped)) = environment() else { return };
    let mut child = Command::new(env!("CARGO_BIN_EXE_zlc-pulse-server"))
        .args(["--backend", "memory", "--memory-auto-done", "--host", "127.0.0.1", "--port", "0", "--local-only"])
        .stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
    let stdout = child.stdout.take().unwrap();
    let server = Killed(child);
    let mut lines = BufReader::new(stdout).lines();
    let port: u16 = loop {
        let line = lines.next().expect("the server ended before listening").unwrap();
        if let Some(rest) = line.strip_prefix("{\"type\":\"pulse-server\",\"port\":") { break rest.trim_end_matches('}').parse().unwrap(); }
    };
    std::thread::spawn(move || for _ in lines {});
    let script = std::env::temp_dir().join(format!("zlc_pulse_v2_client_{}.py", std::process::id()));
    std::fs::write(&script, V2_CLIENT).unwrap();
    let output = python(&capped, &[script.to_str().unwrap(), v2.to_str().unwrap(), &port.to_string()]).output().unwrap();
    let _ = std::fs::remove_file(&script);
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "v2 client failed:\n{text}\n{}", String::from_utf8_lossy(&output.stderr));
    let answer = zlc_pulse::json::parse(text.lines().last().unwrap(), "v2 client").unwrap();
    assert_eq!(answer.get("lanes").and_then(|v| v.as_int()), Some(69));
    assert_eq!(answer.get("pins").and_then(|v| v.as_int()), Some(69));
    assert_eq!(answer.get("run").and_then(|v| v.as_int()), Some(3));
    assert_eq!(answer.get("status").and_then(|v| v.as_int()), Some(zlc_pulse::wire::STATUS_DONE.into()));
    assert_eq!(answer.get("fault").and_then(|v| v.as_str()), Some(""));
    assert_eq!(answer.get("same_program").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(answer.get("same_source").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(answer.get("digest"), answer.get("expected_digest"), "this server digests what v2 compiled");
    assert_eq!(answer.get("safe").and_then(|v| v.as_bool()), Some(true));
    drop(server);
}
