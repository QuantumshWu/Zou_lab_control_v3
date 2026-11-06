//! Finite schedule queries keep table rows distinct from execution repeats.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use zlc_pulse::compile::{CompiledProgram, TargetBusAction};
use zlc_pulse::model::{AnalogMode, FieldKind, MAXIMUM_REPEAT_COUNT};
use zlc_pulse::schedule::{bus_action_ticks, run_duration_seconds, trigger_edge_ticks, trigger_windows_by_channel};

fn program() -> CompiledProgram {
    CompiledProgram {
        clock_hz: 50_000_000.0, target_abi_fingerprint: String::new(), geometry_fingerprint: 0,
        channels: vec!["gate".into(), "held".into()], durations: vec![2, 3, 5], duration_slots: vec![1, 0, 1],
        masks: vec![3, 2, 3], loops: vec![(0, 2, 5), (1, 1, 3)], duration_seconds: 0.0,
        slot_kinds: vec![FieldKind::Duration], bus_names: vec!["dac".into()],
        bus_actions: [0, 2].map(|row| TargetBusAction {
            row, bus_index: 0, bus_name: "dac".into(), mode: AnalogMode::Edge, value: 0, value_select: 0,
        }).to_vec(),
        bus_delays: vec![], channel_delays: vec![7, 11], clk_enable: 0,
        logical_digital_outputs: vec![("camera".into(), "gate".into())], bus_safe_values: vec![0],
    }
}

fn expanded(table: &[Vec<u64>], runs: u64, scans: u64) -> Vec<Vec<u64>> {
    (0..scans).flat_map(|_| table.iter().flat_map(|row| (0..runs).map(move |_| row.clone()))).collect()
}

/// The previous eager schedule, independently retained as a small-input oracle.
fn eager_schedule(program: &CompiledProgram, points: &[Vec<u64>], choose: Option<&dyn Fn(u128) -> u64>)
    -> (Vec<Vec<u128>>, Vec<Vec<u128>>) {
    let mut shapes = BTreeMap::new();
    for point in points {
        if !shapes.contains_key(point) {
            shapes.insert(point.clone(), (program.frame_visits(point, choose).unwrap(), program.frame_ticks(point).unwrap()));
        }
    }
    let mut channels = Vec::new();
    for (bit, delay) in program.channel_delays.iter().enumerate() {
        let (mut offset, mut high, mut edges) = (0, false, Vec::new());
        for point in points {
            let (visits, total) = &shapes[point];
            for &(row, tick) in visits {
                let level = program.masks[row] & (1 << bit) != 0;
                if high != level { edges.push(offset + tick + u128::from(*delay)); high = level; }
            }
            offset += total;
        }
        if high { edges.push(offset + u128::from(*delay)); }
        channels.push(edges);
    }
    let mut buses = vec![Vec::new(); program.bus_names.len()];
    let mut offset = 0;
    for point in points {
        let (visits, total) = &shapes[point];
        for &(row, tick) in visits {
            for action in program.bus_actions.iter().filter(|action| action.row as usize == row) {
                buses[action.bus_index as usize].push(offset + tick);
            }
        }
        offset += total;
    }
    (channels, buses)
}

#[test]
fn repeated_scans_match_the_eager_order_edges_windows_and_duration() {
    let program = program();
    program.validate().unwrap();
    let table = vec![vec![4], vec![2], vec![4]];
    for runs in [1, 2, 5] {
        for scans in [1, 2, 3] {
            let points = expanded(&table, runs, scans);
            let (edges, buses) = eager_schedule(&program, &points, None);
            let actual = trigger_edge_ticks(&program, &["camera", "held"], Some(&table), runs, scans, None).unwrap();
            assert_eq!(actual, vec![("camera".into(), edges[0].clone()), ("held".into(), edges[1].clone())]);
            assert_eq!(bus_action_ticks(&program, Some(&table), runs, scans, None).unwrap(), buses);
            let windows = trigger_windows_by_channel(&program, &["camera", "held"], Some(&table), runs, scans).unwrap();
            for (index, (_, actual)) in windows.iter().enumerate() {
                assert_eq!(actual, &edges[index].chunks_exact(2).map(|pair| (pair[0], pair[1])).collect::<Vec<_>>());
            }
            let ticks: u128 = points.iter().map(|point| program.frame_ticks(point).unwrap()).sum();
            assert_eq!(run_duration_seconds(&program, Some(&table), runs, scans).unwrap(), ticks as f64 / program.clock_hz);
        }
    }
}

#[test]
fn bounded_bracket_callbacks_keep_first_seen_point_order_and_are_not_repeated() {
    let program = program();
    let table = vec![vec![4], vec![2], vec![4]];
    let expected_calls = RefCell::new(Vec::new());
    let choose = |body| { expected_calls.borrow_mut().push(body); 1 + expected_calls.borrow().len() as u64 % 2 };
    let (edges, buses) = eager_schedule(&program, &expanded(&table, 3, 2), Some(&choose));
    let calls = RefCell::new(Vec::new());
    let choose = |body| { calls.borrow_mut().push(body); 1 + calls.borrow().len() as u64 % 2 };
    let actual = trigger_edge_ticks(&program, &["camera", "held"], Some(&table), 3, 2, Some(&choose)).unwrap();
    assert_eq!(actual.into_iter().map(|(_, ticks)| ticks).collect::<Vec<_>>(), edges);
    assert_eq!(*calls.borrow(), *expected_calls.borrow());
    calls.borrow_mut().clear();
    assert_eq!(bus_action_ticks(&program, Some(&table), 3, 2, Some(&choose)).unwrap(), buses);
    assert_eq!(*calls.borrow(), *expected_calls.borrow());
}

#[test]
fn maximal_repeat_duration_and_empty_channel_query_do_not_expand_execution_rows() {
    let mut program = program();
    program.loops.clear();
    let table = vec![vec![1], vec![2], vec![1]]; // One sweep: 5 + 7 + 5 ticks.
    let count = MAXIMUM_REPEAT_COUNT;
    let expected = (17 * u128::from(count) * u128::from(count)) as f64 / program.clock_hz;
    let started = Instant::now();
    assert_eq!(run_duration_seconds(&program, Some(&table), count, count).unwrap(), expected);
    assert!(trigger_edge_ticks(&program, &[], Some(&table), count, count, None).unwrap().is_empty());
    assert!(started.elapsed() < Duration::from_secs(2), "finite repeats were expanded");
}

fn long_program() -> CompiledProgram {
    let mut program = program();
    program.durations = vec![1 << 31];
    program.duration_slots = vec![0];
    program.slot_kinds.clear();
    program.masks = vec![0];
    program.loops = vec![(0, 0, 1 << 31); 3]; // One Pulse is 2^124 ticks, which fits u128.
    program.bus_actions.clear();
    program
}

#[test]
fn duration_keeps_big_integer_totals_but_schedule_offsets_still_refuse_overflow() {
    let mut program = long_program();
    program.validate().unwrap();
    assert_eq!(run_duration_seconds(&program, Some(&[vec![]]), 16, 16).unwrap(), 2f64.powi(132) / program.clock_hz);
    let skip_brackets = |_body| 0;
    for error in [
        trigger_edge_ticks(&program, &["gate"], None, 16, 1, Some(&skip_brackets)).unwrap_err(),
        bus_action_ticks(&program, None, 16, 1, Some(&skip_brackets)).unwrap_err(),
    ] {
        assert_eq!(error.message, "this run lasts longer than the schedule can count");
    }
    program.loops.push((0, 0, 16)); // One Pulse itself no longer fits, even though duration totals are big ints.
    assert_eq!(run_duration_seconds(&program, None, 1, 1).unwrap_err().message,
               "this program lasts longer than the schedule can count");
}

#[test]
fn validation_and_fixed_pulse_rules_are_unchanged() {
    let mut program = program();
    assert_eq!(run_duration_seconds(&program, None, 0, 0).unwrap_err().message,
               "run_repeats must be in the finite hardware range [1, 2^32-1]");
    assert_eq!(run_duration_seconds(&program, None, 1, 0).unwrap_err().message,
               "scan_repeats must be in the finite hardware range [1, 2^32-1]");
    assert_eq!(run_duration_seconds(&program, None, 1, 1).unwrap_err().message,
               "a slotted program requires an explicit value table");
    assert_eq!(run_duration_seconds(&program, Some(&[vec![]]), 1, 1).unwrap_err().message,
               "table width differs from the program slot count");
    assert_eq!(run_duration_seconds(&program, Some(&[]), 1, 1).unwrap_err().message,
               "table must contain at least one row");
    program.duration_slots.fill(0);
    program.slot_kinds.clear();
    assert_eq!(run_duration_seconds(&program, None, 1, 2).unwrap_err().message,
               "scan_repeats must be 1 when there is no scan table");
    assert_eq!(run_duration_seconds(&program, None, 7, 1).unwrap(),
               7.0 * program.frame_ticks(&[]).unwrap() as f64 / program.clock_hz);
}

#[test]
#[ignore = "manual duration-query microbenchmark; run with --ignored --nocapture"]
fn duration_repeat_scaling_benchmark() {
    let program = program();
    let table = vec![vec![4], vec![2], vec![4]];
    for repeats in [1, 1_000, MAXIMUM_REPEAT_COUNT] {
        let started = Instant::now();
        for _ in 0..5_000 {
            std::hint::black_box(run_duration_seconds(std::hint::black_box(&program), Some(&table), repeats, repeats).unwrap());
        }
        eprintln!("duration: 3 rows, run/scan repeats {repeats}, 5000 queries: {:?}", started.elapsed());
    }
}
