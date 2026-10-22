//! The JTAG-to-AXI transport against a fake Vivado Tcl session in front of the
//! memory board twin. Vivado is never run and no board is touched.
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use zlc_pulse::axi::{FakeVivado, JtagAxiRegisterTransport};
use zlc_pulse::compile::compile_sequence;
use zlc_pulse::device::PulseStreamer;
use zlc_pulse::model::{AnalogStep, Binding, FieldRef, OutputDelay, Period, PortSpec, Sequence, SequenceParts, Target};
use zlc_pulse::num::Num;
use zlc_pulse::remote::{PulseRemoteServer, RemotePulseStreamer, set_log_sink};
use zlc_pulse::transport::{MemoryRegisterTransport, RegisterTransport};
use zlc_pulse::wire::{STATUS_DONE, STATUS_RUNNING, StreamerParams, ctrl, region_bases};

fn geometry() -> StreamerParams { StreamerParams { channel_count: 11, bus_count: 2, bus_width: 3, bank_size: 2, ..StreamerParams::default() } }

fn target() -> Arc<Target> {
    let lanes: Vec<String> = ["d0", "d1", "d2", "a0", "a1", "a2", "c0", "b0", "b1", "b2", "c1"].iter().map(|s| s.to_string()).collect();
    let s = |items: &[&str]| items.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let ports = vec![
        PortSpec::new("cooling", "digital", &s(&["d0"]), "cooling", None, None, None, None, None).unwrap(),
        PortSpec::new("probe", "digital", &s(&["d1"]), "", None, None, None, None, None).unwrap(),
        PortSpec::new("camera", "digital", &s(&["d2"]), "", None, None, None, None, None).unwrap(),
        PortSpec::new("dipole", "dac", &s(&["a0", "a1", "a2"]), "", Some(0), None, None, None, Some("clk0")).unwrap(),
        PortSpec::new("clk0", "clock", &s(&["c0"]), "", None, None, None, None, None).unwrap(),
        PortSpec::new("bias", "dac", &s(&["b0", "b1", "b2"]), "", Some(1), None, None, None, Some("clk1")).unwrap(),
        PortSpec::new("clk1", "clock", &s(&["c1"]), "", None, None, None, None, None).unwrap(),
    ];
    Target::new(&lanes, ports, Vec::new()).unwrap()
}

fn sequence(scan: bool) -> Sequence {
    let state = |bits: [u8; 3]| { let mut states = bits.to_vec(); states.extend([0; 8]); states };
    let periods = vec![
        Period::new("load", Num::Int(40), "ns", state([1, 0, 0]), vec![AnalogStep::new("dipole", "edge", Num::Int(2)).unwrap()], "", "period").unwrap(),
        Period::new("image", Num::Int(60), "ns", state([0, 1, 1]), vec![AnalogStep::new("bias", "ramp", Num::Int(-3)).unwrap()], "", "period").unwrap(),
        Period::new("rest", Num::Int(20), "ns", state([0, 0, 0]), vec![], "", "period").unwrap(),
    ];
    let mut parts = SequenceParts::new(target(), periods);
    parts.delays = vec![OutputDelay::new("probe", Num::Int(40), "ns").unwrap()];
    if scan { parts.bindings = vec![Binding::new(FieldRef::duration("load"), "ns", true, "default", "").unwrap()]; }
    Sequence::new(parts).unwrap()
}

fn state_dir(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("zlc-jtag-test-{}-{name}", std::process::id()))
}

fn jtag(board: &Arc<MemoryRegisterTransport>, name: &str) -> (Arc<FakeVivado>, Arc<JtagAxiRegisterTransport>) {
    let fake = Arc::new(FakeVivado::new(board.clone()));
    let transport = Arc::new(JtagAxiRegisterTransport::new(fake.clone(), &state_dir(name), None, "").unwrap());
    (fake, transport)
}

/// The same session on a board reached directly and on one behind the JTAG
/// transport: load, fire, a streamed scan with bank refills, SAFE.
fn session(transport: Arc<dyn RegisterTransport>, board: &Arc<MemoryRegisterTransport>) {
    let streamer = PulseStreamer::new(transport, geometry(), 50e6, target()).unwrap();
    streamer.open().unwrap();
    let plain = Arc::new(sequence(false));
    streamer.load(Arc::new(compile_sequence(&plain, &geometry(), 50e6).unwrap()), Some(plain), Vec::new()).unwrap();
    streamer.fire(2, 1).unwrap();
    board.publish_execution_readback(STATUS_DONE, 0).unwrap();
    assert_eq!(streamer.wait_done(Some(Duration::from_secs(5)), None).unwrap().expect("a report").status, STATUS_DONE);
    let scanned = Arc::new(sequence(true));
    let rows: Vec<Vec<u64>> = (1..=7).map(|ticks| vec![ticks]).collect();
    streamer.load(Arc::new(compile_sequence(&scanned, &geometry(), 50e6).unwrap()), Some(scanned), rows).unwrap();
    streamer.fire(1, 1).unwrap();
    for cursor in 1..7u32 {
        board.publish_execution_readback(STATUS_RUNNING, cursor).unwrap();
        // The observer polls every 50 ms on JTAG: wait for it to refill.
        let seen = Instant::now();
        while streamer.snapshot().cursor != Some(cursor) && seen.elapsed() < Duration::from_secs(5) { std::thread::sleep(Duration::from_millis(5)); }
        std::thread::sleep(Duration::from_millis(60));
    }
    board.publish_execution_readback(STATUS_DONE, 6).unwrap();
    assert_eq!(streamer.wait_done(Some(Duration::from_secs(5)), None).unwrap().expect("a report").cursor, Some(6));
    assert!(streamer.safe().unwrap().stable());
    streamer.close().unwrap();
}

#[test]
fn a_board_behind_jtag_ends_with_the_registers_of_one_reached_directly() {
    let direct = Arc::new(MemoryRegisterTransport::new(&geometry(), false, false).unwrap());
    session(direct.clone(), &direct);
    let behind = Arc::new(MemoryRegisterTransport::new(&geometry(), false, false).unwrap());
    let (fake, transport) = jtag(&behind, "parity");
    session(transport.clone(), &behind);
    let words = region_bases(&geometry()).total as u32;
    // The direct twin takes a command as a call; JTAG writes the COMMAND
    // register's rising edge, which leaves the last code standing there.
    let differing: Vec<u32> = (0..words).filter(|address| *address != ctrl::COMMAND && direct.word(*address) != behind.word(*address)).collect();
    assert!(differing.is_empty(), "registers differ at {differing:?}");
    assert_eq!(behind.word(ctrl::BANK0_CHUNK), 2);
    assert_eq!(fake.conduct().launches, 1, "one resident Vivado for the whole session");
    assert!(transport.describe_link().contains("fake Vivado"));
    // Writes travel as INCR bursts, never one transaction per word.
    let blocks = fake.conduct().blocks.clone();
    assert!(blocks.iter().any(|block| block.contains("-len 3 -type read") && block.contains("ZLCCOMMAND")));
    assert!(blocks.iter().any(|block| block.lines().any(|line| line.contains("-type write -burst INCR") && !line.contains(" -len 1 "))));
}

#[test]
fn a_stop_during_a_slow_read_keeps_the_session_for_the_safe_behind_it() {
    let board = Arc::new(MemoryRegisterTransport::new(&geometry(), false, false).unwrap());
    let (fake, transport) = jtag(&board, "stop");
    let streamer = PulseStreamer::new(transport.clone(), geometry(), 50e6, target()).unwrap();
    streamer.open().unwrap();
    let plain = Arc::new(sequence(false));
    streamer.load(Arc::new(compile_sequence(&plain, &geometry(), 50e6).unwrap()), Some(plain), Vec::new()).unwrap();
    streamer.fire(1, 1).unwrap();
    assert_eq!(board.status(), STATUS_RUNNING);
    // Every reply now takes 400 ms: the observer is caught inside a read.
    fake.conduct().latency = Duration::from_millis(400);
    std::thread::sleep(Duration::from_millis(150));
    let readback = streamer.safe().expect("SAFE travels on the same session");
    assert!(readback.stable());
    assert_eq!(board.status(), 0, "the board was driven SAFE");
    assert_eq!(fake.conduct().launches, 1, "the cancelled read did not end Vivado");
    fake.conduct().latency = Duration::ZERO;
    assert_eq!(transport.read_word(ctrl::LAYOUT_ID, None, None).unwrap(), board.word(ctrl::LAYOUT_ID), "replies stay in step");
    streamer.close().unwrap();
}

#[test]
fn a_wedged_vivado_times_out_ends_the_session_and_restarts_on_start() {
    let board = Arc::new(MemoryRegisterTransport::new(&geometry(), true, false).unwrap());
    let (fake, transport) = jtag(&board, "wedged");
    transport.start().unwrap();
    fake.conduct().hang_next = true;
    let error = transport.read_word(ctrl::STATUS, None, Some(Instant::now() + Duration::from_millis(400))).unwrap_err();
    assert!(error.message.contains("timed out waiting for ZLC_AXI_"), "{error}");
    let error = transport.read_word(ctrl::STATUS, None, None).unwrap_err();
    assert!(error.message.contains("call start()"), "{error}");
    transport.start().unwrap();
    assert_eq!(fake.conduct().launches, 2);
    assert_eq!(transport.read_word(ctrl::LAYOUT_ID, None, None).unwrap(), board.word(ctrl::LAYOUT_ID));
    // An interpreter that exits mid-session is the same: the next action is refused until start.
    fake.conduct().exit_next = true;
    assert!(transport.read_word(ctrl::STATUS, None, None).unwrap_err().message.contains("exited unexpectedly"));
    assert!(transport.read_word(ctrl::STATUS, None, None).is_err());
    transport.close().unwrap();
}

#[test]
fn a_hardware_fault_fails_its_action_and_two_targets_fail_the_start() {
    let board = Arc::new(MemoryRegisterTransport::new(&geometry(), true, false).unwrap());
    let (fake, transport) = jtag(&board, "fault");
    transport.start().unwrap();
    fake.conduct().fail_run = Some("Xicom AXI transaction failed".into());
    let error = transport.write_words(&[(ctrl::CLK_ENABLE, 1)], None, None).unwrap_err();
    assert_eq!(error.message, "persistent Vivado axi_write failed: ERROR: [Labtools 27-3176] Xicom AXI transaction failed");
    transport.write_words(&[(ctrl::CLK_ENABLE, 1)], None, None).expect("the session survives a failed transaction");
    assert_eq!(board.word(ctrl::CLK_ENABLE), 1);
    transport.close().unwrap();
    let (fake, transport) = jtag(&board, "targets");
    fake.conduct().targets = 2;
    let streamer = PulseStreamer::new(transport, geometry(), 50e6, target()).unwrap();
    let error = streamer.open().unwrap_err();
    assert!(error.message.contains("Expected exactly one Vivado hardware target"), "{error}");
}

#[test]
fn a_pulse_server_on_jtag_gives_its_client_the_operation_budget() {
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = lines.clone();
    set_log_sink(Some(Box::new(move |line| sink.lock().unwrap().push(line.to_owned()))));
    let board = Arc::new(MemoryRegisterTransport::new(&geometry(), true, false).unwrap());
    let (_fake, transport) = jtag(&board, "server");
    let streamer = PulseStreamer::new(transport, geometry(), 50e6, target()).unwrap();
    streamer.open().unwrap();
    let mut server = PulseRemoteServer::bind("127.0.0.1:0", streamer, false).unwrap();
    server.start().unwrap();
    let client = RemotePulseStreamer::new("127.0.0.1", server.address().port(), None, Duration::from_secs(2)).unwrap();
    client.open().unwrap();
    let description = client.describe().unwrap();
    let plain = Arc::new(sequence(false));
    client.load(Arc::new(compile_sequence(&plain, &description.geometry, description.clock_hz).unwrap()), Some(plain), Vec::new()).unwrap();
    client.fire(1, 1).unwrap();
    assert_eq!(client.wait_done(Some(Duration::from_secs(5)), None).unwrap().expect("a report").status, STATUS_DONE);
    assert!(client.safe().unwrap().stable());
    client.disconnect();
    server.close();
    set_log_sink(None);
    let lines = lines.lock().unwrap();
    assert!(lines.iter().any(|line| line.contains("RPC BUDGET") && line.contains("method=open")), "{lines:?}");
    assert!(lines.iter().any(|line| line.contains("RPC BUDGET") && line.contains("method=load")));
}
