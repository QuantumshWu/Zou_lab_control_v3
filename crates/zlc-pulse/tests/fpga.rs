//! The FPGA build surface against v2's `zlc fpga`: the shipped geometry header
//! is what the config projects, and the estimate and both projections of the
//! shipped and an over-budget geometry are v2's output line for line.
use std::path::{Path, PathBuf};

use zlc_pulse::fpga::{capacity_report, emit_geom_tcl, emit_geometry_vh};
use zlc_pulse::wire::{fpga_root, load_streamer_config, require_streamer_config};

fn shipped() -> PathBuf { fpga_root().join("board_config").join("streamer_config.json") }

/// The shipped config with v2's capture edits: 16384 rows, 64/128-deep FIFOs, a 90.5% target.
fn over_budget() -> PathBuf {
    let mut text = std::fs::read_to_string(shipped()).unwrap();
    for (from, to) in [("\"max_rows\": 512,", "\"max_rows\": 16384,"), ("\"evt_fifo_depth\": 32,", "\"evt_fifo_depth\": 64,"),
                       ("\"bus_evt_fifo_depth\": 64,", "\"bus_evt_fifo_depth\": 128,"), ("\"target_pct\": 98.0,", "\"target_pct\": 90.5,")] {
        assert_eq!(text.matches(from).count(), 1, "{from}");
        text = text.replace(from, to);
    }
    let path = std::env::temp_dir().join(format!("zlc_fpga_over_{}.json", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}

/// The report without its `config:` line, which names the file it read.
fn report(config: &Path, part: Option<&str>) -> (String, bool) {
    let (text, ok) = capacity_report(&load_streamer_config(Some(config)).unwrap(), part).unwrap();
    (text.lines().filter(|line| !line.starts_with("  config:")).collect::<Vec<_>>().join("\n"), ok)
}

const TABLE_HEAD: &str = "  resource     used   budget    total    %use  verdict";
const FINAL: &str = "  final note: Vivado report_utilization after synthesis; this is a design-budget estimate.";

#[test]
fn the_checked_in_geometry_header_is_the_shipped_configs_projection() {
    let config = require_streamer_config(&shipped()).unwrap();
    let header = std::fs::read_to_string(fpga_root().join("pulse_streamer").join("zlc_geometry.vh")).unwrap().replace("\r\n", "\n");
    assert_eq!(emit_geometry_vh(&config.params, config.uart_baud).unwrap(), header);
    assert!(header.contains("`define ZLC_LAYOUT_FINGERPRINT  32'h5AD5A6A0"));
    assert_eq!(emit_geom_tcl(&config.params).unwrap(), "# AUTO-GENERATED from streamer_config.json by zlc_pulse.wire.emit_geom_tcl -- do not edit.\n\
        # BRAM-IP sizing vars for create_project.tcl (all derived from the config geometry).\n\
        set zlc_row_addr_width 9\nset zlc_row_portb_bits 128\nset zlc_bank_size 2048\nset zlc_scan_portb_bits 128\nset zlc_axi_bram_depth 32768\n");
}

#[test]
fn the_shipped_estimate_is_v2s() {
    let geometry = "  geometry:   channels=69 rows=512 loops=8x4deep bank_size=2048 slots=4 buses=4x10b evt_fifo=32 bus_evt_fifo=64";
    let expected = ["ZLC pulse-streamer resource estimate", "  part:       xc7a35tfgg484-2  (profile xc7a35t)", "  target:     98% of each resource",
        geometry, "", TABLE_HEAD,
        "  LUT         16204    20384    20800   77.9%  OK", "  FF          10936    40768    41600   26.3%  OK",
        "  DSP             8       88       90    8.9%  OK", "  RAMB36         23       49       50   46.0%  OK", "",
        "  RESULT: estimated resources are within 98% on xc7a35tfgg484-2; synthesis, routing and timing are not verified.", "", FINAL];
    assert_eq!(report(&shipped(), None), (expected.join("\n"), true));
    let expected = ["ZLC pulse-streamer resource estimate", "  part:       xc7a100tfgg484-2  (profile xc7a100t)", "  target:     98% of each resource",
        geometry, "", TABLE_HEAD,
        "  LUT         16204    62132    63400   25.6%  OK", "  FF          10936   124264   126800    8.6%  OK",
        "  DSP             8      235      240    3.3%  OK", "  RAMB36         23      132      135   17.0%  OK", "",
        "  RESULT: estimated resources are within 98% on xc7a100tfgg484-2; synthesis, routing and timing are not verified.", "", FINAL];
    assert_eq!(report(&shipped(), Some("xc7a100tfgg484-2")), (expected.join("\n"), true));
}

#[test]
fn an_over_budget_geometry_is_refused_as_v2_refused_it() {
    let path = over_budget();
    let expected = ["ZLC pulse-streamer resource estimate", "  part:       xc7a35tfgg484-2  (profile xc7a35t)", "  target:     90.5% of each resource",
        "  geometry:   channels=69 rows=16384 loops=8x4deep bank_size=2048 slots=4 buses=4x10b evt_fifo=64 bus_evt_fifo=128", "", TABLE_HEAD,
        "  LUT         17676    18824    20800   85.0%  OK", "  FF          10936    37648    41600   26.3%  OK",
        "  DSP             8       81       90    8.9%  OK", "  RAMB36         85       45       50  170.0%  OVER BUDGET", "",
        "  RESULT: INSUFFICIENT -- RAMB36 exceed 90.5% on xc7a35tfgg484-2.  Reduce the geometry in streamer_config.json or choose a larger part (see FPGA_PARTS).",
        "", FINAL];
    assert_eq!(report(&path, None), (expected.join("\n"), false));
    let config = load_streamer_config(Some(&path)).unwrap();
    let header = emit_geometry_vh(&config.params, config.uart_baud).unwrap();
    for line in ["`define ZLC_ROW_ADDR_WIDTH      14", "`define ZLC_EVT_FIFO_DEPTH      64", "`define ZLC_BUS_EVT_FIFO_DEPTH  128",
                 "`define ZLC_LAYOUT_FINGERPRINT  32'h5A7907CF"] {
        assert!(header.contains(line), "{line}");
    }
    assert!(emit_geom_tcl(&config.params).unwrap().ends_with("set zlc_row_addr_width 14\nset zlc_row_portb_bits 128\nset zlc_bank_size 2048\n\
                                                              set zlc_scan_portb_bits 128\nset zlc_axi_bram_depth 131072\n"));
    let _ = std::fs::remove_file(path);
}
