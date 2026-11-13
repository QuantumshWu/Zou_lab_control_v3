//! `zlc-pulse-server`: serve the board plugged into this machine over the pulse
//! RPC (the same protocol as v2's `pulse_server`, board_protocol 8).
//!
//! Startup order: the strict deployment config, the target generated from the
//! board manifest, the backend probe, the listening port (a second server for
//! this port is refused here), then open + first verified SAFE. Ctrl+C or a
//! console close drives the board SAFE before the process exits.
//!
//! Its stdout says how it came up, one JSON line the application reads: ready
//! `{"type":"pulse-server","port":N,"transport":"uart","uart_port":"COM5","attempts":[...]}`
//! (`transport` the one it took: `uart`, `jtag-axi` or `memory`; `attempts` each port its probe tried
//! and what that came to), or `{"type":"pulse-server","error":"<Type>: <message>"}` when it could not;
//! either with `serial_faults`, each serial port that would not open on the way (`OpenFailure::fact`).
//! Each change of the board's owner is a line of its own, in order:
//! `{"type":"pulse-owner","owner":"192.168.50.1:61656"}` (the owning client's address), or
//! `"owner":null` while nobody owns it -- so the application that started it knows when another
//! client took its board and when it was let go.
//!
//! `zlc-pulse-server fpga ...` is the FPGA build's command instead
//! (zlc_pulse::fpga::command, v2's `zlc fpga`).
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use zlc_pulse::axi::{JtagAxiRegisterTransport, default_vivado};
use zlc_pulse::device::PulseStreamer;
use zlc_pulse::manifest::target_from_config;
use zlc_pulse::remote::{DEFAULT_BIND_HOST, DEFAULT_PORT, PulseRemoteServer, server_log};
use zlc_pulse::json::Json;
use zlc_pulse::serial::{Attempt, ComPortLink, FaultWatch, OpenFailure, Probe, find_board_uart, list_com_ports};
use zlc_pulse::transport::{MemoryRegisterTransport, RegisterTransport};
use zlc_pulse::uart::UartRegisterTransport;
use zlc_pulse::wire::{load_streamer_config, require_streamer_config};
use zlc_pulse::{Error, Kind, Result};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn on_console(_event: u32) -> windows_sys::core::BOOL {
    INTERRUPTED.store(true, Ordering::SeqCst);
    1
}

struct Options { host: String, port: u16, backend: String, uart_port: Option<String>, uart_baud: Option<u64>, config: Option<PathBuf>,
                 check: bool, memory_auto_done: bool, local_only: bool, admission_stdin: bool }

fn options() -> std::result::Result<Options, String> {
    let mut options = Options { host: DEFAULT_BIND_HOST.into(), port: DEFAULT_PORT, backend: "auto".into(), uart_port: None, uart_baud: None,
                                config: None, check: false, memory_auto_done: false, local_only: false, admission_stdin: false };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--host" => options.host = value()?,
            "--port" => options.port = value()?.parse().map_err(|_| "--port must be a TCP port")?,
            "--backend" => options.backend = value()?,
            "--uart-port" => options.uart_port = Some(value()?),
            "--uart-baud" => options.uart_baud = Some(value()?.parse().map_err(|_| "--uart-baud must be an integer")?),
            "--config" => options.config = Some(PathBuf::from(value()?)),
            "--check-config" => options.check = true,
            "--memory-auto-done" => options.memory_auto_done = true,
            "--local-only" => options.local_only = true,
            // The application that starts this server says on stdin when peers are admitted (`admit` / `refuse`): its Remote switch.
            "--admission" if value()? == "stdin" => options.admission_stdin = true,
            "--help" | "-h" => return Err("usage: zlc-pulse-server [--host 0.0.0.0] [--port 18861] [--backend auto|uart|jtag-axi|memory] [--uart-port COM6] \
                                           [--uart-baud N] [--config streamer_config.json] [--check-config] [--local-only] [--admission stdin]\n       \
                                           zlc-pulse-server fpga --help   (the FPGA build's config check, geometry and estimate)".into()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if !["auto", "uart", "memory", "jtag-axi"].contains(&options.backend.as_str()) { return Err(format!("unsupported backend: {}", options.backend)); }
    Ok(options)
}

fn uart_transport(port: &str, baud: u64, timeout: Duration) -> Result<Arc<dyn RegisterTransport>> {
    Ok(Arc::new(UartRegisterTransport::new(Box::new(ComPortLink::new(port, baud)?), timeout)))
}

/// Vivado's working directory and its last reply per action (`ZLC_PULSE_AXI_STATE`,
/// else the checkout's `.local\vivado`).
fn axi_state_dir() -> PathBuf {
    std::env::var_os("ZLC_PULSE_AXI_STATE").map(PathBuf::from).unwrap_or_else(|| zlc_core::checkout::local("vivado"))
}

fn jtag_transport() -> Result<Arc<dyn RegisterTransport>> {
    server_log("BACKEND NOTE", None, "detail=\"resident vivado.exe ~1-2 GB; prefer UART on memory-constrained hosts\"");
    Ok(Arc::new(JtagAxiRegisterTransport::from_env(&axi_state_dir())?))
}

/// Why the server did not come up: the error type the application reads (a serial open's own,
/// `SerialPortUnusable` / `SerialPortBusy`, else the pulse error's Python class) and its words.
struct Failed { kind: String, message: String }

impl From<Error> for Failed { fn from(error: Error) -> Failed { Failed { kind: error.kind.python_name().into(), message: error.message } } }

/// The board's transport as it was taken: its kind, the UART port, and why (a fallback's reason).
struct Taken { kind: &'static str, uart_port: Option<String>, reason: String }

/// Each serial port that would not open while `watch` was held, as the application's line carries them.
fn serial_faults(watch: &FaultWatch) -> Json { Json::Array(watch.met().iter().map(OpenFailure::fact).collect()) }

fn run(options: Options, watch: &FaultWatch) -> std::result::Result<(), Failed> {
    let found = load_streamer_config(options.config.as_deref())?;
    let source = found.source.clone().ok_or_else(|| Error::new(Kind::Value,
        format!("deployment requires a valid streamer_config.json without fallback: {}", found.warnings.join("; "))))?;
    if !found.warnings.is_empty() { return Err(Error::new(Kind::Value, format!("deployment requires a valid streamer_config.json without fallback: {}", found.warnings.join("; "))).into()); }
    let config = require_streamer_config(&source)?;
    let target = target_from_config(&config, None, None)?;
    let baud = options.uart_baud.unwrap_or(config.uart_baud);
    if options.check {
        println!("backend={}\nuart_port={}\nuart_baud={baud}\nlisten_bind={}:{}\nconfig={}\nclock_hz={}\ngeometry={:?}\ntarget_ports={}",
                 options.backend, options.uart_port.as_deref().unwrap_or("auto-discover"), options.host, options.port, source.display(),
                 config.clock_hz, config.params, target.ports.len());
        if options.backend != "uart" && options.backend != "memory" {
            println!("vivado={}\naxi_state={}", default_vivado().display(), axi_state_dir().display());
        }
        println!("remote RPC: length-prefixed JSON, one owner, separate completion wait connection");
        return Ok(());
    }
    server_log("SERVER STARTING", None, &format!("requested_backend={} endpoint={}:{}", options.backend, options.host, options.port));
    let mut attempts: Vec<Attempt> = Vec::new();
    let (transport, taken): (Arc<dyn RegisterTransport>, Taken) = match options.backend.as_str() {
        "memory" => (Arc::new(MemoryRegisterTransport::new(&config.params, options.memory_auto_done, false)?),
                     Taken { kind: "memory", uart_port: None, reason: String::new() }),
        "jtag-axi" => {
            server_log("BACKEND RESOLVED", None, "requested=jtag-axi selected=jtag-axi reason=\"explicit jtag-axi backend; UART probe skipped\"");
            (jtag_transport()?, Taken { kind: "jtag-axi", uart_port: None, reason: "explicit jtag-axi backend".into() })
        }
        backend => {
            // Run by hand without --uart-port it probes every port; the application finds the port on Init and names it.
            let ports = options.uart_port.clone().map_or_else(list_com_ports, |port| vec![port]);
            let selected;
            (selected, attempts) = find_board_uart(&ports, baud, config.params, config.clock_hz, &target);
            let tried: Vec<String> = attempts.iter().map(ToString::to_string).collect();
            let failure = if tried.is_empty() { "no board UART detected".to_owned() } else { tried.join("; ") };
            match selected {
                Some(port) => {
                    server_log("BACKEND RESOLVED", None, &format!("requested={backend} selected=uart uart_port={port} attempts={:?}", tried.join("; ")));
                    (uart_transport(&port, baud, Duration::from_secs(5))?, Taken { kind: "uart", uart_port: Some(port), reason: String::new() })
                }
                None => {
                    // The port the application names that cannot be opened at all, or that another program
                    // holds, is not a board that failed to answer: neither backend goes on without it, and
                    // the application looks for the board again (its Init resolve).
                    if let (Some(_), [Attempt { probe: Probe::NotOpened(failure), .. }]) = (&options.uart_port, attempts.as_slice()) {
                        return Err(Failed { kind: failure.fault.error_type().into(), message: failure.to_string() });
                    }
                    if backend == "uart" { return Err(Error::new(Kind::Runtime, format!("explicit UART backend failed; no matching device: {failure}")).into()); }
                    // As v2: a board with no UART answering is reached through its JTAG cable.
                    let reason = format!("auto fallback to jtag-axi after UART probe: {failure}");
                    server_log("BACKEND RESOLVED", None, &format!("requested=auto selected=jtag-axi reason={reason:?}"));
                    (jtag_transport().map_err(|error| Error::new(error.kind, format!("no matching UART board ({failure}), and the JTAG-to-AXI fallback \
                                                                                      could not start: {}", error.message)))?,
                     Taken { kind: "jtag-axi", uart_port: None, reason })
                }
            }
        }
    };
    let streamer = PulseStreamer::new(transport, config.params, config.clock_hz, target.clone())?;
    // Peers (other machines) are admitted from the start, never, or when the application says.
    let mut server = PulseRemoteServer::bind(&format!("{}:{}", options.host, options.port), streamer.clone(), !options.local_only && !options.admission_stdin)?;
    server_log("HARDWARE CONNECTING", None, "action=open_deployed_streamer");
    let connected = streamer.open().and_then(|()| streamer.safe());
    match connected {
        Ok(readback) if readback.stable() => server_log("HARDWARE CONNECTED", None, &format!("geometry_handshake=True safe_readback=True status={}", readback.status)),
        Ok(readback) => { let _ = streamer.close(); return Err(Error::new(Kind::Runtime, format!("initial SAFE readback was not stable: status={}", readback.status)).into()); }
        Err(error) => { let _ = streamer.close(); return Err(error.into()); }
    }
    unsafe { windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(on_console), 1) };
    server.on_owner(|owner| println!("{}", Json::object(vec![("type", Json::str("pulse-owner")), ("owner", Json::opt_str(owner))]).compact_text()));
    server.start()?;
    server_log("READY", None, &format!("hardware=connected waiting_for_client=True same_computer=127.0.0.1:{} transport={}", server.address().port(), taken.kind));
    let mut ready = vec![("type", Json::str("pulse-server")), ("port", Json::Int(server.address().port().into())), ("transport", Json::str(taken.kind)),
                         ("attempts", Json::strings(&attempts.iter().map(ToString::to_string).collect::<Vec<_>>())), ("serial_faults", serial_faults(watch))];
    if let Some(port) = &taken.uart_port { ready.push(("uart_port", Json::str(port))); }
    if !taken.reason.is_empty() { ready.push(("reason", Json::str(&taken.reason))); }
    println!("{}", Json::object(ready).compact_text());
    // `admit` / `refuse` from the application that started this server; its
    // going away (stdin ends) ends the server, which drives the board SAFE.
    let (said, admission) = std::sync::mpsc::channel::<Option<String>>();
    if options.admission_stdin {
        std::thread::spawn(move || {
            for line in std::io::BufRead::lines(std::io::stdin().lock()) {
                let Ok(line) = line else { break };
                if said.send(Some(line.trim().to_owned())).is_err() { return; }
            }
            let _ = said.send(None);
        });
    }
    let mut reason = "console interrupt";
    while !INTERRUPTED.load(Ordering::SeqCst) {
        match admission.try_recv() {
            Ok(Some(word)) if word == "admit" || word == "refuse" => server.admit_peers(word == "admit"),
            Ok(Some(_)) => {}
            Ok(None) => { reason = "the application that started this server ended"; break; }
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
    server_log("SERVER STOPPING", None, &format!("reason={reason:?}"));
    server.close();
    streamer.close()?;
    server_log("SERVER CLOSED", None, "device_session=closed");
    Ok(())
}

fn main() {
    // `zlc-pulse-server fpga ...`: the board's build-time command (v2 `zlc fpga`), not the server.
    if std::env::args().nth(1).as_deref() == Some("fpga") { std::process::exit(zlc_pulse::fpga::command(std::env::args().skip(2))); }
    let options = match options() { Ok(options) => options, Err(message) => { eprintln!("{message}"); std::process::exit(2); } };
    let watch = FaultWatch::start();
    if let Err(Failed { kind, message }) = run(options, &watch) {
        eprintln!("ERROR before LISTENING: {kind}: {message}");
        println!("{}", Json::object(vec![("type", Json::str("pulse-server")), ("error", Json::str(format!("{kind}: {message}"))),
                                         ("serial_faults", serial_faults(&watch))]).compact_text());
        std::process::exit(3);
    }
}
