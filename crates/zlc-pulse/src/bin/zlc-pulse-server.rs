//! `zlc-pulse-server`: serve the board plugged into this machine over the pulse
//! RPC (the same protocol as v2's `pulse_server`, board_protocol 8).
//!
//! Startup order: the strict deployment config, the target generated from the
//! board manifest, the backend probe, the listening port (a second server for
//! this port is refused here), then open + first verified SAFE. Ctrl+C or a
//! console close drives the board SAFE before the process exits.
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
use zlc_pulse::serial::{ComPortLink, find_board_uart, list_com_ports};
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

fn run(options: Options) -> Result<()> {
    let found = load_streamer_config(options.config.as_deref())?;
    let source = found.source.clone().ok_or_else(|| Error::new(Kind::Value,
        format!("deployment requires a valid streamer_config.json without fallback: {}", found.warnings.join("; "))))?;
    if !found.warnings.is_empty() { return Err(Error::new(Kind::Value, format!("deployment requires a valid streamer_config.json without fallback: {}", found.warnings.join("; ")))); }
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
    let transport: Arc<dyn RegisterTransport> = match options.backend.as_str() {
        "memory" => Arc::new(MemoryRegisterTransport::new(&config.params, options.memory_auto_done, false)?),
        "jtag-axi" => {
            server_log("BACKEND RESOLVED", None, "requested=jtag-axi selected=jtag-axi reason=\"explicit jtag-axi backend; UART probe skipped\"");
            jtag_transport()?
        }
        backend => {
            // Run by hand without --uart-port it probes every port; the application finds the port on Init and names it.
            let ports = options.uart_port.clone().map_or_else(list_com_ports, |port| vec![port]);
            let (selected, attempts) = find_board_uart(&ports, baud, config.params, config.clock_hz, &target);
            let failure = if attempts.is_empty() { "no board UART detected".to_owned() } else { attempts.join("; ") };
            match selected {
                Some(port) => {
                    server_log("BACKEND RESOLVED", None, &format!("requested={backend} selected=uart uart_port={port} attempts={:?}", attempts.join("; ")));
                    uart_transport(&port, baud, Duration::from_secs(5))?
                }
                None if backend == "uart" => return Err(Error::new(Kind::Runtime, format!("explicit UART backend failed; no matching device: {failure}"))),
                // As v2: a board with no UART answering is reached through its JTAG cable.
                None => {
                    server_log("BACKEND RESOLVED", None, &format!("requested=auto selected=jtag-axi reason={:?}",
                                                                  format!("auto fallback to jtag-axi after UART probe: {failure}")));
                    jtag_transport().map_err(|error| Error::new(error.kind, format!("no matching UART board ({failure}), and the JTAG-to-AXI fallback \
                                                                                     could not start: {}", error.message)))?
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
        Ok(readback) => { let _ = streamer.close(); return Err(Error::new(Kind::Runtime, format!("initial SAFE readback was not stable: status={}", readback.status))); }
        Err(error) => { let _ = streamer.close(); return Err(error); }
    }
    unsafe { windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(on_console), 1) };
    server.start()?;
    server_log("READY", None, &format!("hardware=connected waiting_for_client=True same_computer=127.0.0.1:{}", server.address().port()));
    println!("{{\"type\":\"pulse-server\",\"port\":{}}}", server.address().port());
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
    if let Err(error) = run(options) {
        eprintln!("ERROR before LISTENING: {}: {}", error.kind.python_name(), error.message);
        std::process::exit(3);
    }
}
