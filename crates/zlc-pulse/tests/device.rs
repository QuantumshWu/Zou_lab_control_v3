//! The device, its transports and the remote protocol against the in-memory
//! board twin: the same twin v2 tests its device with. No hardware.
use std::sync::Arc;
use std::time::Duration;

use zlc_pulse::compile::compile_sequence;
use zlc_pulse::device::PulseStreamer;
use zlc_pulse::model::{AnalogStep, Binding, FieldRef, OutputDelay, Period, PortSpec, Sequence, SequenceParts, Target};
use zlc_pulse::num::Num;
use zlc_pulse::remote::{PulseRemoteServer, RemotePulseStreamer};
use zlc_pulse::transport::{MemoryRegisterTransport, RegisterTransport};
use zlc_pulse::uart::{LoopbackLink, UartRegisterTransport};
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

#[test]
fn load_fire_wait_and_safe_on_the_memory_twin() {
    let board = Arc::new(MemoryRegisterTransport::new(&geometry(), true, true).unwrap());
    let streamer = PulseStreamer::new(board.clone(), geometry(), 50e6, target()).unwrap();
    streamer.open().unwrap();
    let sequence = Arc::new(sequence(false));
    let program = Arc::new(compile_sequence(&sequence, &geometry(), 50e6).unwrap());
    streamer.load(program.clone(), Some(sequence.clone()), Vec::new()).unwrap();
    assert_eq!(board.word(ctrl::PROG_COUNT), 3);
    assert_eq!(streamer.snapshot().applied_digest, program.digest());
    let applied = streamer.fire(2, 1).unwrap();
    assert_eq!(applied.run_repeats, 2);
    assert_eq!(board.word(ctrl::RUN_REPEAT_COUNT), 2);
    let report = streamer.wait_done(Some(Duration::from_secs(2)), None).unwrap().expect("a report");
    assert_eq!(report.status, STATUS_DONE);
    assert!(report.fault().is_empty(), "{}", report.fault());
    // A second load of the same program writes no new image.
    let batches = board.write_batches().len();
    streamer.load(program, Some(sequence), Vec::new()).unwrap();
    assert!(board.write_batches()[batches..].iter().all(|batch| batch.iter().all(|(a, _)| *a == ctrl::COMMAND || *a == ctrl::COMMAND_ID)));
    assert!(streamer.safe().unwrap().stable());
    streamer.close().unwrap();
    assert!(!board.is_started());
}

#[test]
fn a_streamed_scan_refills_each_released_bank() {
    let board = Arc::new(MemoryRegisterTransport::new(&geometry(), false, false).unwrap());
    let streamer = PulseStreamer::new(board.clone(), geometry(), 50e6, target()).unwrap();
    streamer.open().unwrap();
    let sequence = Arc::new(sequence(true));
    let program = Arc::new(compile_sequence(&sequence, &geometry(), 50e6).unwrap());
    let rows: Vec<Vec<u64>> = (1..=7).map(|ticks| vec![ticks]).collect();
    streamer.load(program, Some(sequence), rows.clone()).unwrap();
    streamer.fire(1, 1).unwrap();
    let scan_base = region_bases(&geometry()).scan as u32;
    let bank_words = (geometry().bank_size * geometry().scan_words()) as u32;
    // Chunks 0 and 1 are resident at FIRE.
    assert_eq!(board.word(scan_base), 1);
    assert_eq!(board.word(scan_base + bank_words), 3);
    // The board plays on: entering chunk 1 releases bank 0 for chunk 2, and so on.
    for cursor in 1..7u32 {
        board.publish_execution_readback(STATUS_RUNNING, cursor).unwrap();
        std::thread::sleep(Duration::from_millis(30));
    }
    assert_eq!(board.word(ctrl::BANK0_CHUNK), 2);
    assert_eq!(board.word(ctrl::BANK1_CHUNK), 3);
    assert_eq!(board.word(scan_base), 5, "bank 0 holds chunk 2");
    assert_eq!(board.word(scan_base + bank_words), 7, "bank 1 holds chunk 3");
    board.publish_execution_readback(STATUS_DONE, 6).unwrap();
    let report = streamer.wait_done(Some(Duration::from_secs(2)), None).unwrap().expect("a report");
    assert_eq!(report.cursor, Some(6));
    streamer.close().unwrap();
}

#[test]
fn a_layout_mismatch_refuses_the_board() {
    let board = Arc::new(MemoryRegisterTransport::with_layout(0x5A00_0001, true, false));
    let streamer = PulseStreamer::new(board.clone(), geometry(), 50e6, target()).unwrap();
    let error = streamer.open().unwrap_err();
    assert!(error.message.starts_with("geometry/layout mismatch: device=0x5A000001"), "{error}");
    assert!(!board.is_started());
}

#[test]
fn the_uart_transport_recovers_lost_and_damaged_frames() {
    let board = Arc::new(MemoryRegisterTransport::new(&geometry(), true, false).unwrap());
    let mut link = LoopbackLink::new(board.clone());
    link.drop_replies.extend([1, 4]);
    link.damage_requests.extend([6]);
    let transport: Arc<dyn RegisterTransport> = Arc::new(UartRegisterTransport::new(Box::new(link), Duration::from_secs(2)));
    let streamer = PulseStreamer::new(transport.clone(), geometry(), 50e6, target()).unwrap();
    streamer.open().unwrap();
    let sequence = Arc::new(sequence(false));
    let program = Arc::new(compile_sequence(&sequence, &geometry(), 50e6).unwrap());
    streamer.load(program, Some(sequence), Vec::new()).unwrap();
    streamer.fire(1, 1).unwrap();
    let report = streamer.wait_done(Some(Duration::from_secs(5)), None).unwrap().expect("a report");
    assert_eq!(report.status, STATUS_DONE);
    assert!(transport.resends() >= 3, "{} resends", transport.resends());
    streamer.close().unwrap();
}

#[test]
fn a_remote_client_owns_the_board_until_a_newcomer_takes_it_safe() {
    let board = Arc::new(MemoryRegisterTransport::new(&geometry(), false, false).unwrap());
    let streamer = PulseStreamer::new(board.clone(), geometry(), 50e6, target()).unwrap();
    streamer.open().unwrap();
    let finish = |board: Arc<MemoryRegisterTransport>| std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        board.publish_execution_readback(STATUS_DONE, 0).unwrap();
    });
    let mut server = PulseRemoteServer::bind("127.0.0.1:0", streamer, false).unwrap();
    server.start().unwrap();
    let port = server.address().port();
    let client = RemotePulseStreamer::new("127.0.0.1", port, None, Duration::from_secs(2)).unwrap();
    client.open().unwrap();
    let description = client.describe().unwrap();
    assert_eq!(description.target.abi_fingerprint(), target().abi_fingerprint());
    let sequence = Arc::new(sequence(false));
    let program = Arc::new(compile_sequence(&sequence, &description.geometry, description.clock_hz).unwrap());
    client.load(program.clone(), Some(sequence), Vec::new()).unwrap();
    let applied = client.fire(3, 1).unwrap();
    finish(board.clone());
    assert_eq!(applied.run_repeats, 3);
    let report = client.wait_done(Some(Duration::from_secs(3)), None).unwrap().expect("a report");
    assert_eq!(report.status, STATUS_DONE);
    let held = client.applied().unwrap().expect("an application");
    assert_eq!(held.program.digest(), program.digest());
    assert_eq!(*held.program, *program);
    // A second client takes over only after SAFE; the first hears it lost the board.
    let newcomer = RemotePulseStreamer::new("127.0.0.1", port, None, Duration::from_secs(2)).unwrap();
    newcomer.open().unwrap();
    assert!(newcomer.safe().unwrap().stable());
    assert!(client.snapshot().is_err(), "the replaced client no longer owns the board");
    newcomer.load(program.clone(), None, Vec::new()).unwrap();
    newcomer.fire(0, 1).unwrap();
    assert_eq!(board.status(), STATUS_RUNNING);
    newcomer.disconnect();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(board.status(), 0, "a lost owner leaves the board SAFE");
    server.close();
}
