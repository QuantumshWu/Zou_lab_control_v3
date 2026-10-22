//! The board's build-time surface (v2 `zlc_pulse.fpga`): whether a geometry
//! fits an FPGA part, and the two projections of `streamer_config.json` the
//! Vivado build reads -- the RTL geometry header `zlc_geometry.vh` and the
//! BRAM-IP sizing Tcl `geom.tcl`. `zlc-pulse-server fpga` is its command line
//! ([`command`]), which `bin\scripts\build_and_program.bat` (installation.exe's FPGA steps) and
//! `bin\scripts\estimate_resources.bat` call.
use std::path::Path;

use zlc_core::pyfmt::format_g;

use crate::error::{Error, Kind, Result, value_error};
use crate::wire::{FROZEN_CLOCK_HZ, StreamerConfig, StreamerParams, build_fingerprint, build_ip_sizes, check_rtl_assumptions};

/// The resources of one FPGA family member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartProfile { pub name: &'static str, pub ramb36: u64, pub lut: u64, pub ff: u64, pub dsp: u64, pub distributed_ram_kib: u64 }

pub const FPGA_PARTS: [PartProfile; 5] = [
    PartProfile { name: "xc7a35t", ramb36: 50, lut: 20800, ff: 41600, dsp: 90, distributed_ram_kib: 400 },
    PartProfile { name: "xc7a50t", ramb36: 75, lut: 32600, ff: 65200, dsp: 120, distributed_ram_kib: 600 },
    PartProfile { name: "xc7a75t", ramb36: 105, lut: 47200, ff: 94400, dsp: 180, distributed_ram_kib: 892 },
    PartProfile { name: "xc7a100t", ramb36: 135, lut: 63400, ff: 126800, dsp: 240, distributed_ram_kib: 1188 },
    PartProfile { name: "xc7a200t", ramb36: 365, lut: 134600, ff: 269200, dsp: 740, distributed_ram_kib: 2888 },
];

/// The routed design outside the geometry-dependent schedulers (engine, top,
/// UART bridge and the JTAG-AXI IPs), calibrated on the 2026-09-21 routed
/// period-table build on the xc7a35t (16204 LUT, 10936 FF in all).
pub const ENGINE_LOGIC_LUTS: u64 = 14052;
pub const ENGINE_FF: u64 = 10936;
/// BRAM36-equivalent tiles outside the geometry memories (four RAMB36 and two
/// RAMB18: the JTAG-AXI master, its debug hub and the UART bridge).
const FIXED_RAMB36: u64 = 5;

/// The profile whose name the part string starts with (longest name first).
pub fn part_profile(part: &str) -> Result<&'static PartProfile> {
    let key = part.trim().to_lowercase();
    let mut parts: Vec<&'static PartProfile> = FPGA_PARTS.iter().collect();
    parts.sort_by_key(|profile| std::cmp::Reverse(profile.name.len()));
    parts.into_iter().find(|profile| key.starts_with(profile.name))
        .ok_or_else(|| Error::new(Kind::Key, format!("unknown FPGA part '{part}'; add it to FPGA_PARTS.")))
}

fn resource_target_pct(pct: f64) -> Result<f64> {
    if !pct.is_finite() || !(1.0..=100.0).contains(&pct) { return value_error("target_pct must be finite and from 1 through 100"); }
    Ok(pct)
}

/// One resource of an estimate: used against `target_pct` of the part's total.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis { pub used: u64, pub budget: u64, pub total: u64, pub pct: f64, pub ok: bool }

impl Axis {
    fn new(used: u64, total: u64, target_pct: f64) -> Axis {
        let budget = (total as f64 * target_pct / 100.0) as u64;
        let pct = if total > 0 { 100.0 * used as f64 / total as f64 } else { 0.0 };
        Axis { used, budget, total, pct, ok: used <= budget }
    }
}

/// The estimate of every resource, in the report's order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate { pub lut: Axis, pub ff: Axis, pub dsp: Axis, pub ramb36: Axis }

impl Estimate {
    pub fn axes(&self) -> [(&'static str, &Axis); 4] { [("LUT", &self.lut), ("FF", &self.ff), ("DSP", &self.dsp), ("RAMB36", &self.ramb36)] }
    pub fn ok(&self) -> bool { self.axes().iter().all(|(_, axis)| axis.ok) }
}

/// RAMB36 tiles of one `width x depth` block RAM: 36-bit width slices, 512-word
/// depth slices, two RAMB18 per tile.
fn ramb36(width_bits: u64, depth: u64) -> u64 { (width_bits.div_ceil(36) * depth.div_ceil(512)).div_ceil(2) }

/// Distributed-RAM LUTs of one FIFO: a RAM32M gives 6 data bits per 4 LUTs,
/// a RAM64M 3; deeper FIFOs take whole 64-deep banks.
fn fifo_ram_luts(depth: u64, width: u64) -> u64 {
    if depth <= 32 { 4 * width.div_ceil(6) } else { 4 * width.div_ceil(3) * depth.div_ceil(64) }
}

/// The resources a concrete geometry takes on `part` (v2 `estimate_resources`).
/// The TTL event scheduler is one 49-bit event FIFO plus ~20 LUTs of compare and
/// control per delay-capable channel; the DAC delay one FIFO of resolved action
/// descriptors per bus (the RTL's SEG_W: emit time, span, two values, two
/// step/remainder fields, two flags); DSP the two reciprocal products of each
/// bus's live divmod.
pub fn estimate_resources(params: &StreamerParams, part: &str, target_pct: f64) -> Result<Estimate> {
    check_rtl_assumptions(params)?;
    let profile = part_profile(part)?;
    let pct = resource_target_pct(target_pct)?;
    let ramb36_used = ramb36(params.row_portb_bits(), params.max_rows) + ramb36(params.slot_bits(), 2 * params.bank_size) + FIXED_RAMB36;
    let ttl_luts = params.num_delay_ch() * (20 + fifo_ram_luts(params.evt_fifo_depth.max(1), 49));
    let segment_bits = 48 + params.tick_width + 2 * params.bus_width + 2 * (params.bus_width + 1) + 2;
    let bus_luts = params.bus_count * (20 + fifo_ram_luts(params.bus_evt_fifo_depth.max(1), segment_bits));
    Ok(Estimate {
        lut: Axis::new(ENGINE_LOGIC_LUTS + ttl_luts + bus_luts, profile.lut, pct),
        ff: Axis::new(ENGINE_FF, profile.ff, pct),
        dsp: Axis::new(2 * params.bus_count, profile.dsp, pct),
        ramb36: Axis::new(ramb36_used, profile.ramb36, pct),
    })
}

/// The pass/fail table of `config`'s geometry on `part` (the config's own part
/// unless one is given), v2 `format_capacity_report`; `true` when it fits.
pub fn capacity_report(config: &StreamerConfig, part: Option<&str>) -> Result<(String, bool)> {
    let part = part.unwrap_or(&config.fpga_part);
    let estimate = estimate_resources(&config.params, part, config.target_pct)?;
    let profile = part_profile(part)?;
    let p = &config.params;
    let target = format_g(config.target_pct, 6);
    let source = config.source.as_ref().map(|path| path.display().to_string())
        .unwrap_or_else(|| "(built-in defaults -- no streamer_config.json found)".into());
    let mut lines = vec![
        "ZLC pulse-streamer resource estimate".to_owned(),
        format!("  config:     {source}"),
        format!("  part:       {part}  (profile {})", profile.name),
        format!("  target:     {target}% of each resource"),
        format!("  geometry:   channels={} rows={} loops={}x{}deep bank_size={} slots={} buses={}x{}b evt_fifo={} bus_evt_fifo={}",
                p.channel_count, p.max_rows, p.max_loops, p.loop_depth, p.bank_size, p.num_slots, p.bus_count, p.bus_width,
                p.evt_fifo_depth, p.bus_evt_fifo_depth),
        String::new(),
        format!("  {:<8} {:>8} {:>8} {:>8}  {:>6}  verdict", "resource", "used", "budget", "total", "%use"),
    ];
    for (label, axis) in estimate.axes() {
        lines.push(format!("  {label:<8} {:>8} {:>8} {:>8}  {:>5.1}%  {}", axis.used, axis.budget, axis.total, axis.pct,
                           if axis.ok { "OK" } else { "OVER BUDGET" }));
    }
    lines.push(String::new());
    if estimate.ok() {
        lines.push(format!("  RESULT: estimated resources are within {target}% on {part}; synthesis, routing and timing are not verified."));
    } else {
        let over: Vec<&str> = estimate.axes().iter().filter(|(_, axis)| !axis.ok).map(|(label, _)| *label).collect();
        lines.push(format!("  RESULT: INSUFFICIENT -- {} exceed {target}% on {part}.  Reduce the geometry in streamer_config.json \
                            or choose a larger part (see FPGA_PARTS).", over.join(", ")));
    }
    lines.extend(config.warnings.iter().map(|warning| format!("  note: {warning}")));
    lines.push(String::new());
    lines.push("  final note: Vivado report_utilization after synthesis; this is a design-budget estimate.".into());
    Ok((lines.join("\n"), estimate.ok()))
}

pub const GEOMETRY_VH_FILENAME: &str = "zlc_geometry.vh";

/// The Verilog header every RTL source and testbench `include`s: each geometry
/// parameter the RTL defaults to, derived here, and the layout fingerprint the
/// host checks on word 63.
pub fn emit_geometry_vh(params: &StreamerParams, uart_baud: u64) -> Result<String> {
    check_rtl_assumptions(params)?;
    if uart_baud == 0 { return value_error("uart_baud must be a positive integer"); }
    if uart_baud as f64 * 16.0 > FROZEN_CLOCK_HZ { return value_error("uart_baud requires more than one 16x oversampling tick per fabric clock"); }
    let p = params;
    let macros: [(&str, u64); 20] = [
        ("ZLC_CHANNEL_COUNT", p.channel_count), ("ZLC_NUM_SLOTS", p.num_slots), ("ZLC_SLOT_SEL_WIDTH", p.slot_sel_width()),
        ("ZLC_TICK_WIDTH", p.tick_width), ("ZLC_ROW_ADDR_WIDTH", p.row_addr_width()), ("ZLC_ROW_BITS", p.row_bits()),
        ("ZLC_ROW_WORDS", p.row_words()), ("ZLC_BANK_SIZE", p.bank_size), ("ZLC_SCAN_ADDR_WIDTH", p.scan_addr_width()),
        ("ZLC_BUS_COUNT", p.bus_count), ("ZLC_BUS_INDEX_WIDTH", p.bus_index_width()), ("ZLC_BUS_WIDTH", p.bus_width),
        ("ZLC_MAX_LOOPS", p.max_loops), ("ZLC_LOOP_INDEX_WIDTH", p.loop_index_width()), ("ZLC_LOOP_DEPTH", p.loop_depth),
        ("ZLC_EVT_FIFO_DEPTH", p.evt_fifo_depth), ("ZLC_BUS_EVT_FIFO_DEPTH", p.bus_evt_fifo_depth), ("ZLC_NUM_DELAY_CH", p.num_delay_ch()),
        ("ZLC_DELAY_CH_IDX_W", p.channel_bit_width()), ("ZLC_DELAY_REG_WORDS", p.delay_region_words),
    ];
    let width = macros.iter().map(|(name, _)| name.len()).max().unwrap_or(0) + 1;
    let mut lines: Vec<String> = [
        "// ==========================================================================",
        "// zlc_geometry.vh -- AUTO-GENERATED from fpga/board_config/streamer_config.json by",
        "//   zlc fpga --emit-geometry-vh <path>",
        "// DO NOT EDIT.  Every RTL geometry parameter (+ the LAYOUT_FINGERPRINT the host connect-",
        "// check verifies) defaults to a macro here, so editing the config + rebuilding propagates",
        "// to the bitstream and testbenches with no hand-carried literal.  Regenerated from the",
        "// active config by the frozen build flow; the deployment copy is pinned by the geometry",
        "// anchor test.",
        "// ==========================================================================",
        "`ifndef ZLC_GEOMETRY_VH",
        "`define ZLC_GEOMETRY_VH",
    ].into_iter().map(str::to_owned).collect();
    lines.extend(macros.iter().map(|(name, value)| format!("`define {name:<width$} {value}")));
    lines.push(format!("`define {:<width$} {uart_baud}", "ZLC_UART_BAUD"));
    lines.push(format!("`define {:<width$} 32'h{:08X}", "ZLC_LAYOUT_FINGERPRINT", build_fingerprint(params)?));
    lines.push("`endif // ZLC_GEOMETRY_VH".into());
    lines.push(String::new());
    Ok(lines.join("\n"))
}

/// The BRAM-IP sizing variables `create_project.tcl` sources (`ZLC_PS_GEOM_TCL`),
/// derived from the geometry so the IP depths and port widths follow it.
pub fn emit_geom_tcl(params: &StreamerParams) -> Result<String> {
    check_rtl_assumptions(params)?;
    let ip = build_ip_sizes(params);
    let size = |name: &str| ip.iter().find(|(key, _)| *key == name).map(|(_, value)| *value).expect("a build IP size");
    Ok(format!("# AUTO-GENERATED from streamer_config.json by zlc_pulse.wire.emit_geom_tcl -- do not edit.\n\
                # BRAM-IP sizing vars for create_project.tcl (all derived from the config geometry).\n\
                set zlc_row_addr_width {}\nset zlc_row_portb_bits {}\nset zlc_bank_size {}\nset zlc_scan_portb_bits {}\nset zlc_axi_bram_depth {}\n",
               params.row_addr_width(), size("row_portb_bits"), params.bank_size, size("scan_portb_bits"), size("axi_bram_depth")))
}

/// Write a generated text file in the checkout's own line endings (CRLF on
/// Windows, as the RTL tree is checked out).
pub fn write_generated(path: &Path, text: &str) -> Result<()> {
    let text = if cfg!(windows) { text.replace('\n', "\r\n") } else { text.to_owned() };
    std::fs::write(path, text).map_err(|error| Error::new(Kind::Io, format!("could not write {}: {error}", path.display())))
}

// ------------------------------------------------------------------ the command
const USAGE: &str = "usage: zlc-pulse-server fpga [--config streamer_config.json] [--part PART] [--emit-geom-tcl PATH] \
                     [--emit-geometry-vh PATH] [--require] [--vivado]";

/// `zlc-pulse-server fpga ...`: v2's `zlc fpga` command, with its arguments.
/// With no action it prints whether the configured part has the resources for
/// the configured geometry (0 when it does, 1 when it does not);
/// `--emit-geometry-vh` / `--emit-geom-tcl` write the two projections of the
/// config the Vivado build reads; `--require` checks the named config with
/// nothing forgiven and prints its part; `--vivado` prints the Vivado this
/// machine builds with (the pulse server's own rule). Any error answers 2.
pub fn command(args: impl Iterator<Item = String>) -> i32 {
    match run(args) {
        Ok(code) => code,
        Err(message) => { eprintln!("zlc-pulse-server fpga: {message}"); 2 }
    }
}

fn run(mut args: impl Iterator<Item = String>) -> std::result::Result<i32, String> {
    let (mut config, mut part, mut geom_tcl, mut geometry_vh, mut require) = (None, None, None, None, false);
    while let Some(arg) = args.next() {
        let (name, inline) = match arg.split_once('=') { Some((name, value)) if name.starts_with("--") => (name.to_owned(), Some(value.to_owned())), _ => (arg.clone(), None) };
        let mut value = || inline.clone().or_else(|| args.next()).filter(|value| !value.is_empty()).ok_or_else(|| format!("{name} needs a value"));
        match name.as_str() {
            "--config" => config = Some(std::path::PathBuf::from(value()?)),
            "--part" => part = Some(value()?),
            "--emit-geom-tcl" => geom_tcl = Some(std::path::PathBuf::from(value()?)),
            "--emit-geometry-vh" => geometry_vh = Some(std::path::PathBuf::from(value()?)),
            "--require" => require = true,
            "--vivado" => { println!("{}", crate::axi::default_vivado().display()); return Ok(0); }
            "--help" | "-h" => { println!("{USAGE}"); return Ok(0); }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if require {
        let path = config.ok_or("--require needs --config")?;
        let found = crate::wire::require_streamer_config(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        println!("{}", found.fpga_part);
        return Ok(0);
    }
    let found = crate::wire::load_streamer_config(config.as_deref()).map_err(|error| error.to_string())?;
    if let Some(path) = geom_tcl {
        write_generated(&path, &emit_geom_tcl(&found.params).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
        println!("wrote geometry tcl -> {}", path.display());
        return Ok(0);
    }
    if let Some(path) = geometry_vh {
        write_generated(&path, &emit_geometry_vh(&found.params, found.uart_baud).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
        println!("wrote geometry header -> {}", path.display());
        return Ok(0);
    }
    let (report, ok) = capacity_report(&found, part.as_deref()).map_err(|error| error.to_string())?;
    println!("{report}");
    Ok(if ok { 0 } else { 1 })
}
