//! `zlc-pulse-server`'s one line on how it came up: ready with the transport it took and what its
//! probe tried, or the error it stopped on, typed as raised (a named UART port that cannot be opened
//! is `SerialPortUnusable`, never a quiet JTAG fallback). Memory boards and an absent port only.
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

use zlc_pulse::json::{Json, parse};

struct Killed(Child);
impl Drop for Killed { fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); } }

/// The board's build config, as deployment hands it to the server.
fn config() -> String { concat!(env!("CARGO_MANIFEST_DIR"), "/fpga/board_config/streamer_config.json").to_owned() }

/// The server's `{"type":"pulse-server",...}` line for these arguments.
fn said(arguments: &[&str]) -> (Json, Killed) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_zlc-pulse-server")).args(arguments).args(["--config", &config(), "--host", "127.0.0.1", "--port", "0", "--local-only"])
        .stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let stdout = child.stdout.take().unwrap();
    let server = Killed(child);
    for line in BufReader::new(stdout).lines() {
        let line = line.unwrap();
        if line.starts_with("{\"type\":\"pulse-server\"") { return (parse(&line, "ready line").unwrap(), server); }
    }
    panic!("the server ended without saying how it came up");
}

#[test]
fn the_ready_line_names_the_transport_taken() {
    let (ready, _server) = said(&["--backend", "memory", "--memory-auto-done"]);
    assert!(ready.get("port").and_then(Json::as_int).is_some_and(|port| port > 0), "{ready:?}");
    assert_eq!((ready.get("transport"), ready.get("attempts")), (Some(&Json::str("memory")), Some(&Json::Array(Vec::new()))));
}

#[test]
fn a_named_port_that_cannot_be_opened_stops_the_server_with_its_class() {
    let (said, mut server) = said(&["--backend", "auto", "--uart-port", "COM250"]);
    let error = said.get("error").and_then(Json::as_str).unwrap_or_default().to_owned();
    assert!(error.starts_with("SerialPortUnusable: COM250 could not be opened: not present (os error 2: "), "{error}");
    assert!(error.contains(" -- serial ports on this PC: ") || error.ends_with(" -- this PC has no serial port"), "{error}");
    assert_eq!(server.0.wait().unwrap().code(), Some(3), "it stops; Auto does not go on over JTAG");
}
