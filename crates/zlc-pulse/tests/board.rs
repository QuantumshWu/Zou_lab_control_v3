//! The virtual board (the device on its memory twin, played at wall time into
//! a world) and the run surface nodes use. The deployed board's manifest
//! ships with this crate (`fpga/`); no hardware.
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use zlc_pulse::binding::resolve_api_parameters;
use zlc_pulse::board::{Board, Fate, World, run_timeline};
use zlc_pulse::run::{ScanRows, describe_document, done_json, read_document, run_document};
use zlc_pulse::compile::compile_sequence;
use zlc_pulse::json::Json;
use zlc_pulse::transport::Event;

fn pulses() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("pulses") }

#[derive(Default)]
struct Recorder { runs: Mutex<Vec<Json>>, safes: Mutex<u32> }

impl World for Recorder {
    fn play_run(&self, timeline: &Json, stop: &Arc<Event>) -> zlc_pulse::Result<()> {
        self.runs.lock().unwrap().push(timeline.clone());
        // The world paces itself: it answers when the run has played.
        let ticks = timeline.get("duration_ticks").and_then(Json::as_int).unwrap() as f64;
        let clock = timeline.get("clock_hz").and_then(Json::as_f64).unwrap();
        stop.wait(Some(Duration::from_secs_f64(ticks / clock)));
        Ok(())
    }
    fn safe(&self) { *self.safes.lock().unwrap() += 1; }
}

#[test]
fn run_timelines_equal_the_worlds_v2_derivation() {
    // D_tools/golden/out_world_v2.timelines.json: v2's program read the way
    // v2's SimulationWorld.fire read it (D's v2_timeline.py), per test pulse.
    let Some(golden) = std::env::var_os("ZLC_PULSE_WORLD_GOLDEN") else { return };
    let golden = zlc_pulse::json::parse(&std::fs::read_to_string(golden).unwrap(), "golden").unwrap();
    let board = Board::virtual_board(None).unwrap();
    let description = board.description().clone();
    for (name, expected) in golden.as_object().unwrap() {
        let (_path, sequence) = read_document(&pulses().join(name)).unwrap();
        let resolved = resolve_api_parameters(&sequence, None).unwrap();
        let program = compile_sequence(&resolved, &description.geometry, description.clock_hz).unwrap();
        let ours = run_timeline(&program, None).unwrap();
        for key in ["clock_hz", "duration_ticks", "buses"] { assert_eq!(ours.get(key), expected.get(key), "{name}: {key}"); }
        for (channel, windows) in expected.get("windows").unwrap().as_object().unwrap() {
            assert_eq!(ours.get("windows").unwrap().get(channel), Some(windows), "{name}: {channel} windows");
        }
    }
    board.close().unwrap();
}

#[test]
fn the_virtual_board_plays_each_run_into_its_world_at_wall_time() {
    let board = Board::virtual_board(None).unwrap();
    let world = Arc::new(Recorder::default());
    board.set_world(Some(world.clone()));
    let document = pulses().join("imaging_template.json");
    let description = describe_document(&document).unwrap();
    let fields = description.get("api_fields").unwrap().as_array().unwrap();
    assert_eq!(fields[0].get("field_id").and_then(Json::as_str), Some("duration:long_before"));
    // 0.01 s for long_before (declared in s); the receipt says it in the field's unit.
    let started = Instant::now();
    let receipt = run_document(&board, &document, &[("duration:long_before".into(), (10.0, "ms".into()))], None, 2, 1).unwrap();
    assert_eq!(receipt.get("api_values").unwrap().get("duration:long_before").and_then(Json::as_f64), Some(0.01));
    let command_id = receipt.get("command_id").and_then(Json::as_int).unwrap() as u32;
    assert!(board.wait_done(Some(Duration::from_millis(1)), Some(command_id)).unwrap().is_none(), "still playing");
    let report = board.wait_done(Some(Duration::from_secs(5)), Some(command_id)).unwrap().expect("a report");
    let elapsed = started.elapsed().as_secs_f64();
    let answer = done_json(Some(&report));
    assert_eq!(answer.get("status").and_then(Json::as_str), Some("done"), "{answer:?}");
    let runs = world.runs.lock().unwrap().len();
    assert_eq!(runs, 2, "one timeline per run");
    let ticks = world.runs.lock().unwrap()[0].get("duration_ticks").and_then(Json::as_int).unwrap() as f64;
    assert!(elapsed >= 2.0 * ticks / 50e6, "two runs take their physical time: {elapsed} s");
    assert!(board.safe().unwrap().stable());

    // Run until Stop: SAFE ends the run being played and tells the world.
    let receipt = run_document(&board, &document, &[], None, 0, 1).unwrap();
    let command_id = receipt.get("command_id").and_then(Json::as_int).unwrap() as u32;
    std::thread::sleep(Duration::from_millis(150));
    let stopping = Instant::now();
    assert!(board.safe().unwrap().stable());
    assert!(stopping.elapsed() < Duration::from_millis(500), "SAFE reached the run being played");
    assert!(board.wait_done(Some(Duration::from_millis(50)), Some(command_id)).unwrap().is_none(), "a stopped run reports nothing");
    assert!(*world.safes.lock().unwrap() >= 2);
    let snapshot = board.snapshot().unwrap();
    assert_eq!(snapshot.get("firing").and_then(Json::as_bool), Some(false));
    board.close().unwrap();
}

fn command_id(receipt: &Json) -> u32 { receipt.get("command_id").and_then(Json::as_int).unwrap() as u32 }

fn refusal(result: zlc_pulse::Result<Option<zlc_pulse::device::DoneReport>>) -> String {
    match result { Err(error) => error.message, Ok(report) => panic!("expected a refusal, got {report:?}") }
}

#[test]
fn a_fire_reports_once_and_a_fire_that_cannot_report_is_refused() {
    let board = Board::virtual_board(None).unwrap();
    let document = pulses().join("imaging_template.json");
    assert!(refusal(board.wait_report(7, Some(Duration::from_millis(1)))).contains("this board has not fired it"));
    let first = command_id(&run_document(&board, &document, &[], None, 1, 1).unwrap());
    // Until its report is taken, the next FIRE is refused and says why.
    let busy = run_document(&board, &document, &[], None, 1, 1).unwrap_err().message;
    assert!(busy.contains(&format!("FIRE {first} has not reported yet")), "{busy}");
    assert!(board.wait_report(first, Some(Duration::from_millis(1))).unwrap().is_none(), "still playing");
    let report = board.wait_report(first, Some(Duration::from_secs(5))).unwrap().expect("its report");
    assert_eq!(done_json(Some(&report)).get("status").and_then(Json::as_str), Some("done"));
    assert_eq!(board.last_fire(), Some((first, Fate::Reported)));
    assert!(refusal(board.wait_report(first, Some(Duration::from_secs(5)))).contains("its report was already taken"));
    // Taking the report freed the board.
    let second = command_id(&run_document(&board, &document, &[], None, 0, 1).unwrap());
    assert_ne!(first, second);
    assert!(refusal(board.wait_report(first, None)).contains(&format!("FIRE {second} was sent since")));
    // A waiter blocked on a FIRE that Stop ends is answered at once, not at its timeout.
    let shared = Arc::new(board);
    let waiter = { let board = shared.clone(); std::thread::spawn(move || (Instant::now(), board.wait_report(second, Some(Duration::from_secs(20))))) };
    std::thread::sleep(Duration::from_millis(100));
    shared.safe().unwrap();
    let (started, result) = waiter.join().unwrap();
    assert!(started.elapsed() < Duration::from_secs(3), "the waiter was released by SAFE");
    assert!(refusal(result).contains(&format!("FIRE {second} can no longer report")));
    assert_eq!(shared.last_fire(), Some((second, Fate::Safed)));
    assert!(refusal(shared.wait_report(second, None)).contains("the board was made safe"));
    // After SAFE the board fires again.
    let third = command_id(&run_document(&shared, &document, &[], None, 1, 1).unwrap());
    assert!(shared.wait_report(third, Some(Duration::from_secs(5))).unwrap().is_some());
    shared.close().unwrap();
}

#[test]
fn a_run_plays_a_scan_table_and_its_receipt_says_what_plays_when() {
    let board = Board::virtual_board(None).unwrap();
    let world = Arc::new(Recorder::default());
    board.set_world(Some(world.clone()));
    let document = pulses().join("imaging_template.json");
    let description = describe_document(&document).unwrap();
    let field = description.get("api_fields").unwrap().as_array().unwrap()[1].clone();
    assert_eq!(field.get("field_id").and_then(Json::as_str), Some("duration:short"));
    for key in ["is_dac", "lo", "hi", "limit_lo", "limit_hi"] { assert!(field.get(key).is_some(), "api field fact {key}"); }
    assert!(description.get("scan_fields").unwrap().as_array().unwrap().is_empty(), "the template carries no scan slot");
    // The table's rows are said in the field's own unit (s) and quantized to the board's 20 ns tick.
    let table = ScanRows { fields: vec!["duration:short".into()], rows: vec![vec![0.002], vec![0.003_000_001]] };
    let receipt = run_document(&board, &document, &[], Some(&table), 1, 1).unwrap();
    let played = receipt.get("scan").unwrap();
    assert_eq!(played.get("rows").unwrap().as_array().unwrap()[1].as_array().unwrap()[0].as_f64(), Some(0.003));
    let timing = receipt.get("timing").unwrap();
    let short = timing.get("periods").unwrap().as_array().unwrap().iter().find(|p| p.get("period_id").and_then(Json::as_str) == Some("short")).unwrap();
    assert_eq!(short.get("duration_ticks").and_then(Json::as_int), Some(100_000), "the first row's 2 ms at 50 MHz");
    let id = command_id(&receipt);
    board.wait_report(id, Some(Duration::from_secs(5))).unwrap().expect("its report");
    let runs = world.runs.lock().unwrap();
    assert_eq!(runs.len(), 2, "one run per row");
    let ticks: Vec<i128> = runs.iter().map(|run| run.get("duration_ticks").and_then(Json::as_int).unwrap()).collect();
    assert_eq!(ticks[1] - ticks[0], 50_000, "the second row plays 1 ms longer");
    drop(runs);
    // A field the pulse does not have is refused before anything is sent.
    let wrong = ScanRows { fields: vec!["duration:nope".into()], rows: vec![vec![0.002]] };
    assert!(run_document(&board, &document, &[], Some(&wrong), 1, 1).unwrap_err().message.contains("is not a field of this pulse"));
    board.close().unwrap();
}
