//! The host-side AXI image and geometry contract of the frozen pulse-streamer
//! RTL: register map, the word-63 layout fingerprint, and packing a compiled
//! program and its scan rows into the words the board reads.
use std::path::{Path, PathBuf};

use crate::compile::CompiledProgram;
use crate::error::{Error, Kind, Result, value_error};
use crate::json::{self, Json};
use crate::loops::loop_nesting_depth;
use crate::model::{PortKind, Target};

/// Bumped with any register-structure change; folded into the fingerprint.
pub const LAYOUT_STRUCT_VERSION: u32 = 8;
pub const FROZEN_CLOCK_HZ: f64 = 50_000_000.0;
pub const DEFAULT_FPGA_PART: &str = "xc7a35tfgg484-2";
pub const DEFAULT_UART_BAUD: u64 = 460_800;
pub const DEFAULT_TARGET_PCT: f64 = 90.0;

pub const CMD_LOAD: u32 = 1 << 0;
pub const CMD_FIRE: u32 = 1 << 1;
pub const CMD_RESET: u32 = 1 << 2;
pub const CMD_SAFE: u32 = 1 << 3;

pub const STATUS_LOADED: u32 = 1 << 0;
pub const STATUS_RUNNING: u32 = 1 << 1;
pub const STATUS_DONE: u32 = 1 << 2;
pub const STATUS_ERROR: u32 = 1 << 3;
pub const STATUS_UNDERFLOW: u32 = 1 << 4;
pub const STATUS_LINK_ERROR: u32 = 1 << 5;

/// CTRL register words (word addresses).
pub mod ctrl {
    pub const COMMAND: u32 = 1;
    pub const STATUS: u32 = 2;
    pub const PROG_COUNT: u32 = 3;
    pub const SCAN_COUNT: u32 = 4;
    pub const SCAN_ENABLE: u32 = 5;
    pub const RUN_REPEAT_COUNT: u32 = 6;
    pub const LOOP_TABLE_COUNT: u32 = 7;
    pub const CURSOR: u32 = 15;
    pub const BANK_READY: u32 = 16;
    pub const BANK0_CHUNK: u32 = 17;
    pub const BANK1_CHUNK: u32 = 18;
    pub const SCAN_REPEAT_COUNT: u32 = 19;
    pub const CLK_ENABLE: u32 = 20;
    pub const COMMAND_ID: u32 = 22;
    pub const ACK_ID: u32 = 23;
    pub const ACK_STATUS: u32 = 24;
    pub const ACK_CURSOR: u32 = 25;
    pub const LAYOUT_ID: u32 = 63;
}
pub const CTRL_WORDS: u64 = 64;

/// The deployed geometry; every field but `ttl_delay_max_ticks` is folded into
/// the fingerprint, so a host packing for one geometry refuses a bitstream
/// built for another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamerParams {
    pub channel_count: u64,
    pub num_slots: u64,
    pub tick_width: u64,
    pub max_rows: u64,
    pub bank_size: u64,
    pub bus_count: u64,
    pub bus_width: u64,
    pub max_loops: u64,
    pub loop_depth: u64,
    pub ttl_delay_max_ticks: u64,
    pub evt_fifo_depth: u64,
    pub bus_evt_fifo_depth: u64,
    pub delay_region_words: u64,
}

impl Default for StreamerParams {
    /// The values of the shipped `streamer_config.json` (the qualified 35T build).
    fn default() -> Self {
        StreamerParams { channel_count: 69, num_slots: 4, tick_width: 32, max_rows: 512, bank_size: 2048, bus_count: 4, bus_width: 10,
                         max_loops: 8, loop_depth: 4, ttl_delay_max_ticks: (1 << 31) - 1, evt_fifo_depth: 32, bus_evt_fifo_depth: 64,
                         delay_region_words: 128 }
    }
}

/// The dataclass field names, in declaration order.
pub const PARAM_FIELDS: [&str; 13] = ["channel_count", "num_slots", "tick_width", "max_rows", "bank_size", "bus_count", "bus_width",
                                      "max_loops", "loop_depth", "ttl_delay_max_ticks", "evt_fifo_depth", "bus_evt_fifo_depth",
                                      "delay_region_words"];

fn pow2_at_least(value: u64) -> u64 { let mut n = 1; while n < value { n <<= 1; } n }
fn addr_width(depth: u64) -> u64 { (64 - pow2_at_least(depth.max(1)).leading_zeros() as u64).saturating_sub(1).max(1) }
fn ceil_div(a: u64, b: u64) -> u64 { a.div_ceil(b) }
fn is_pow2(value: u64) -> bool { value > 0 && value & (value - 1) == 0 }

impl StreamerParams {
    pub fn get(&self, name: &str) -> Option<u64> {
        Some(match name {
            "channel_count" => self.channel_count, "num_slots" => self.num_slots, "tick_width" => self.tick_width,
            "max_rows" => self.max_rows, "bank_size" => self.bank_size, "bus_count" => self.bus_count, "bus_width" => self.bus_width,
            "max_loops" => self.max_loops, "loop_depth" => self.loop_depth, "ttl_delay_max_ticks" => self.ttl_delay_max_ticks,
            "evt_fifo_depth" => self.evt_fifo_depth, "bus_evt_fifo_depth" => self.bus_evt_fifo_depth,
            "delay_region_words" => self.delay_region_words, _ => return None,
        })
    }

    pub fn set(&mut self, name: &str, value: u64) -> bool {
        let slot = match name {
            "channel_count" => &mut self.channel_count, "num_slots" => &mut self.num_slots, "tick_width" => &mut self.tick_width,
            "max_rows" => &mut self.max_rows, "bank_size" => &mut self.bank_size, "bus_count" => &mut self.bus_count,
            "bus_width" => &mut self.bus_width, "max_loops" => &mut self.max_loops, "loop_depth" => &mut self.loop_depth,
            "ttl_delay_max_ticks" => &mut self.ttl_delay_max_ticks, "evt_fifo_depth" => &mut self.evt_fifo_depth,
            "bus_evt_fifo_depth" => &mut self.bus_evt_fifo_depth, "delay_region_words" => &mut self.delay_region_words,
            _ => return false,
        };
        *slot = value;
        true
    }

    /// Leading channels that drive real TTL pins.
    pub fn num_delay_ch(&self) -> u64 { self.channel_count.saturating_sub(self.bus_count * (self.bus_width + 1)) }
    pub fn channel_bit_width(&self) -> u64 { addr_width(self.num_delay_ch().max(2)) }
    pub fn bus_index_width(&self) -> u64 { addr_width(self.bus_count.max(2)) }
    pub fn ctrl_scratch_base(&self) -> u64 { (u64::from(ctrl::CLK_ENABLE) + 1).max(u64::from(ctrl::ACK_CURSOR) + 1) }
    pub fn ctrl_scratch_words(&self) -> i64 { i64::from(ctrl::LAYOUT_ID) - self.ctrl_scratch_base() as i64 }
    /// Bits of a slot selector: `0` = literal, `k` = slot `k-1`.
    pub fn slot_sel_width(&self) -> u64 { addr_width(self.num_slots + 1) }
    pub fn slot_bits(&self) -> u64 { self.num_slots * self.tick_width }
    pub fn scan_words(&self) -> u64 { self.num_slots }
    /// One row's action for one DAC bus: mode (2), slot selector, value.
    pub fn bus_action_bits(&self) -> u64 { 2 + self.slot_sel_width() + self.bus_width }
    pub fn row_bits(&self) -> u64 { self.tick_width + self.slot_sel_width() + self.num_delay_ch() + self.bus_count * self.bus_action_bits() }
    pub fn row_words(&self) -> u64 { pow2_at_least(ceil_div(self.row_bits(), 32)) }
    pub fn row_portb_bits(&self) -> u64 { self.row_words() * 32 }
    pub fn row_addr_width(&self) -> u64 { addr_width(self.max_rows) }
    pub fn loop_index_width(&self) -> u64 { addr_width(self.max_loops.max(2)) }
    pub fn loop_words(&self) -> u64 { 2 }
    pub fn scan_addr_width(&self) -> u64 { addr_width(2 * self.bank_size) }
}

/// The 32-bit host/bitstream fingerprint on CTRL word 63: 'Z' in the high byte,
/// the low 24 bits a CRC-32 of the layout version and every geometry field.
pub fn build_fingerprint(params: &StreamerParams) -> Result<u32> {
    let mut names: Vec<&str> = PARAM_FIELDS.iter().copied().filter(|name| *name != "ttl_delay_max_ticks").collect();
    names.sort_unstable();
    let mut payload = b"ZLL".to_vec();
    payload.extend_from_slice(&LAYOUT_STRUCT_VERSION.to_le_bytes());
    for name in names {
        let value = i32::try_from(params.get(name).expect("a field"))
            .map_err(|_| Error::new(Kind::Value, "'i' format requires -2147483648 <= number <= 2147483647"))?;
        payload.extend_from_slice(&value.to_le_bytes());
    }
    Ok(0x5A00_0000 | (crc32fast::hash(&payload) & 0x00FF_FFFF))
}

/// Word-address bases of each AXI write region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionBases { pub ctrl: u64, pub rows: u64, pub scan: u64, pub r#loop: u64, pub delay: u64, pub total: u64 }

pub fn region_bases(p: &StreamerParams) -> RegionBases {
    let rows = CTRL_WORDS;
    let scan = rows + p.max_rows * p.row_words();
    let r#loop = scan + 2 * p.bank_size * p.scan_words();
    let delay = r#loop + p.max_loops * p.loop_words();
    RegionBases { ctrl: 0, rows, scan, r#loop, delay, total: delay + p.delay_region_words }
}

/// BRAM/IP sizes derived from the geometry.
pub fn build_ip_sizes(p: &StreamerParams) -> Vec<(&'static str, u64)> {
    vec![("row_portb_bits", p.row_portb_bits()), ("scan_portb_bits", p.slot_bits()), ("row_addr_width", p.row_addr_width()),
         ("bank_size", p.bank_size), ("row_porta_depth", p.max_rows * p.row_words()),
         ("scan_porta_depth", 2 * p.bank_size * (p.slot_bits() / 32)), ("axi_bram_depth", pow2_at_least(region_bases(p).total))]
}

/// Refuse geometries that would silently corrupt the shipped RTL contract.
pub fn check_rtl_assumptions(p: &StreamerParams) -> Result<()> {
    if p.num_delay_ch() < 1 || p.bus_count > 32 {
        return value_error("geometry requires TTL channels and at most 32 DAC clock-enable bits");
    }
    if !is_pow2(p.num_slots) {
        return value_error(format!("num_slots must be a power of two (got {}): the scan BRAM reads one whole slot vector per access, \
                                    and its wide read port must be a power-of-two multiple of the 32-bit host write port.", p.num_slots));
    }
    if p.ctrl_scratch_words() < 2 {
        return value_error(format!("CTRL register file has no scratch room: defined words reach {} but word {} holds the layout \
                                    fingerprint; grow CTRL_WORDS / the RTL ctrl_reg file in lock-step.", p.ctrl_scratch_base(), ctrl::LAYOUT_ID));
    }
    if !is_pow2(p.bank_size) {
        return value_error(format!("bank_size must be a power of two (got {}); scan_addr_of concatenates {{bank_bit, offset}} and would \
                                    alias the two banks otherwise.", p.bank_size));
    }
    if !is_pow2(p.max_rows) { return value_error(format!("max_rows must be a power of two (got {}); MAX_ROWS = 1 << ROW_ADDR_WIDTH.", p.max_rows)); }
    if p.max_rows >= 1 << 16 {
        return value_error(format!("max_rows must fit a 16-bit loop-table row field (got {}); one loop word packs its first and last row \
                                    into two 16-bit halves.", p.max_rows));
    }
    if p.max_loops < 1 || p.loop_depth < 1 || p.loop_depth > p.max_loops {
        return value_error(format!("max_loops ({}) and loop_depth ({}) must be at least one, with the depth no deeper than the table.",
                                   p.max_loops, p.loop_depth));
    }
    if !is_pow2(p.evt_fifo_depth) || !is_pow2(p.bus_evt_fifo_depth) {
        return value_error(format!("evt_fifo_depth ({}) and bus_evt_fifo_depth ({}) must each be a power of two: the engine's event-FIFO \
                                    ring pointers wrap at 2^clog2(depth) but the distributed-RAM array is exactly `depth` deep, so a non-pow2 \
                                    depth reaches indices depth..2^k-1 = out-of-bounds LUTRAM (silent X corruption of scheduled toggles).",
                                   p.evt_fifo_depth, p.bus_evt_fifo_depth));
    }
    if p.tick_width != 32 {
        return value_error(format!("tick_width must be 32 (got {}); the row duration, scan slots and CTRL words are 32b.", p.tick_width));
    }
    if p.ttl_delay_max_ticks >= 1 << 32 {
        return value_error(format!("ttl_delay_max_ticks ({}) must fit the 32-bit R_DELAY register field (0 <= cap < 2^32).", p.ttl_delay_max_ticks));
    }
    if p.num_delay_ch() + p.bus_count > p.delay_region_words {
        return value_error(format!("TTL count {} + bus_count {} exceeds the DELAY register region ({} words; one 32b word per TTL, then per bus).",
                                   p.num_delay_ch(), p.bus_count, p.delay_region_words));
    }
    Ok(())
}

fn checked(value: u128, width: u64, name: &str) -> Result<u128> {
    if width < 128 && value >> width != 0 {
        return value_error(format!("{name}={value} does not fit the unsigned {width}-bit wire field"));
    }
    Ok(value)
}

/// A bit vector as little-endian 32-bit words.
struct Bits(Vec<u32>);

impl Bits {
    fn put(&mut self, offset: u64, width: u64, value: u128) {
        for bit in 0..width.min(128) {
            if (value >> bit) & 1 == 1 {
                let at = offset + bit;
                self.0[(at / 32) as usize] |= 1 << (at % 32);
            }
        }
    }
}

/// One DAC bus action of a row as packed: `(mode, value, value_select)`.
pub type PackedAction = (crate::model::AnalogMode, u64, u64);

/// One period row as its image words, least-significant word first: duration,
/// its slot selector, the TTL mask, then one action per DAC bus.
pub fn pack_row(p: &StreamerParams, duration: u64, duration_slot: u64, mask: u128, actions: &[(u64, PackedAction)]) -> Result<Vec<u32>> {
    let duration = checked(u128::from(duration), p.tick_width, "row duration")?;
    if duration < 1 { return value_error("a row lasts at least one tick"); }
    let mut bits = Bits(vec![0; p.row_words() as usize]);
    bits.put(0, p.tick_width, duration);
    let mut offset = p.tick_width;
    let slot = checked(u128::from(duration_slot), p.slot_sel_width(), "row duration slot")?;
    if duration_slot > p.num_slots { return value_error(format!("row duration slot {duration_slot} exceeds num_slots {}", p.num_slots)); }
    bits.put(offset, p.slot_sel_width(), slot);
    offset += p.slot_sel_width();
    bits.put(offset, p.num_delay_ch(), checked(mask, p.num_delay_ch(), "TTL row mask")?);
    offset += p.num_delay_ch();
    for bus in 0..p.bus_count {
        if let Some((_, (mode, value, select))) = actions.iter().find(|(index, _)| *index == bus) {
            if *select > p.num_slots { return value_error(format!("bus {bus} value slot {select} exceeds num_slots {}", p.num_slots)); }
            let value = checked(u128::from(*value), p.bus_width, &format!("bus {bus} value"))?;
            let select = checked(u128::from(*select), p.slot_sel_width(), &format!("bus {bus} value slot"))?;
            let mode = match mode { crate::model::AnalogMode::Edge => 1, crate::model::AnalogMode::Ramp => 2 };
            bits.put(offset, p.bus_width, value);
            bits.put(offset + p.bus_width, p.slot_sel_width(), select);
            bits.put(offset + p.bus_width + p.slot_sel_width(), 2, mode);
        }
        offset += p.bus_action_bits();
    }
    Ok(bits.0)
}

/// `pack_program(program, params, target=)`: the final sparse AXI write image,
/// `(word address, value)` in address order.
pub fn pack_program(program: &CompiledProgram, p: &StreamerParams, target: &Target) -> Result<Vec<(u32, u32)>> {
    check_rtl_assumptions(p)?;
    if program.target_abi_fingerprint != target.abi_fingerprint() || program.channels != target.raw_lanes {
        return value_error("compiled target ABI/channels do not match the wire target");
    }
    let bases = region_bases(p);
    let rows = program.durations.len() as u64;
    if rows > p.max_rows { return value_error(format!("{rows} rows > max_rows {}.", p.max_rows)); }
    if program.slot_count() as u64 > p.num_slots {
        return value_error(format!("program slot count {} exceeds wire capacity {}", program.slot_count(), p.num_slots));
    }
    if program.loops.len() as u64 > p.max_loops { return value_error(format!("{} loops > max_loops {}.", program.loops.len(), p.max_loops)); }
    let depth = loop_nesting_depth(&program.loops);
    if depth as u64 > p.loop_depth { return value_error(format!("loops nest {depth} deep > loop_depth {}.", p.loop_depth)); }
    let mut by_row: Vec<Vec<(u64, PackedAction)>> = vec![Vec::new(); rows as usize];
    for action in &program.bus_actions {
        if action.bus_index >= p.bus_count {
            return value_error(format!("bus action index {} is outside the {}-bus wire geometry", action.bus_index, p.bus_count));
        }
        let entry = &mut by_row[action.row as usize];
        entry.retain(|(bus, _)| *bus != action.bus_index);
        entry.push((action.bus_index, (action.mode, action.value, action.value_select)));
    }
    let mut words: std::collections::BTreeMap<u64, u32> = std::collections::BTreeMap::new();
    words.insert(u64::from(ctrl::PROG_COUNT), rows as u32);
    words.insert(u64::from(ctrl::SCAN_COUNT), 0);
    words.insert(u64::from(ctrl::SCAN_ENABLE), 0);
    words.insert(u64::from(ctrl::RUN_REPEAT_COUNT), 1);
    words.insert(u64::from(ctrl::SCAN_REPEAT_COUNT), 1);
    words.insert(u64::from(ctrl::LOOP_TABLE_COUNT), program.loops.len() as u32);
    for row in 0..rows as usize {
        let packed = pack_row(p, program.durations[row], program.duration_slots[row], program.masks[row], &by_row[row])?;
        for (k, word) in packed.into_iter().enumerate() { words.insert(bases.rows + row as u64 * p.row_words() + k as u64, word); }
    }
    for (index, &(first, last, count)) in program.loops.iter().enumerate() {
        if !(first <= last && (last as u64) < rows) {
            return value_error(format!("loop {index} rows {first}..{last} lie outside the {rows}-row table"));
        }
        words.insert(bases.r#loop + index as u64 * p.loop_words(), (first as u32) | ((last as u32) << 16));
        let count = checked(u128::from(count), 32, &format!("loop {index} count"))?;
        if count < 1 { return value_error(format!("loop {index} count must be at least one")); }
        words.insert(bases.r#loop + index as u64 * p.loop_words() + 1, count as u32);
    }
    words.insert(u64::from(ctrl::BANK0_CHUNK), 0);
    words.insert(u64::from(ctrl::BANK1_CHUNK), 1);
    let limit_text = |cap: u64| format!("{:.1}", cap as f64 * 20e-9);
    for (channel, &delay) in program.channel_delays.iter().enumerate() {
        let channel = channel as u64;
        if channel >= p.channel_count {
            if delay != 0 { return value_error(format!("channel-delay bit {channel} is outside channel_count {}.", p.channel_count)); }
            continue;
        }
        if delay > p.ttl_delay_max_ticks {
            return value_error(format!("channel bit {channel} delay {delay} ticks is outside [0, {}] (~{} s at 20 ns/tick).",
                                       p.ttl_delay_max_ticks, limit_text(p.ttl_delay_max_ticks)));
        }
        if delay != 0 && channel >= p.num_delay_ch() {
            return value_error(format!(
                "channel bit {channel} has a non-zero delay ({delay} ticks) but is NOT delay-eligible: only channels 0..{} (real TTL outputs) \
                 carry a hardware delay; {}..{} are DAC bus-member / da_clk pins the RTL does not address as TTL delays.",
                p.num_delay_ch() - 1, p.num_delay_ch(), p.channel_count - 1));
        }
    }
    for channel in 0..p.num_delay_ch() {
        let delay = program.channel_delays.get(channel as usize).copied().unwrap_or(0);
        words.insert(bases.delay + channel, delay as u32);
    }
    let mut bus_delay = vec![0u64; p.bus_count as usize];
    for item in &program.bus_delays {
        if item.bus_index >= p.bus_count { return value_error(format!("bus delay bus_index {} is outside bus_count {}.", item.bus_index, p.bus_count)); }
        if item.delay_ticks > p.ttl_delay_max_ticks {
            return value_error(format!("bus {} delay {} ticks is outside [0, {}] (~{} s at 20 ns/tick).", item.bus_index, item.delay_ticks,
                                       p.ttl_delay_max_ticks, limit_text(p.ttl_delay_max_ticks)));
        }
        bus_delay[item.bus_index as usize] = item.delay_ticks;
    }
    for (bus, delay) in bus_delay.iter().enumerate() { words.insert(bases.delay + p.num_delay_ch() + bus as u64, *delay as u32); }
    let raw_clocks = checked(program.clk_enable, p.channel_count, "clock-enable mask")?;
    let (mut known, mut bus_clocks) = (0u128, 0u32);
    for port in target.ports.iter().filter(|port| port.kind == PortKind::Dac) {
        let Some(clock) = &port.latch_clock else { continue };
        let bus = port.bus_index.expect("a DAC");
        if bus >= p.bus_count { return value_error("target DAC bus exceeds the wire geometry"); }
        let lane = &target.port(clock).expect("validated clock").lanes[0];
        let bit = 1u128 << target.lane_index(lane).expect("a lane");
        known |= bit;
        if raw_clocks & bit != 0 { bus_clocks |= 1 << bus; }
    }
    if raw_clocks & !known != 0 { return value_error("clock-enable mask contains a lane that is not a DAC latch clock"); }
    words.insert(u64::from(ctrl::CLK_ENABLE), bus_clocks);
    Ok(words.into_iter().map(|(address, value)| (address as u32, value)).collect())
}

/// `pack_scan_rows(rows, geom, bank, chunk)`: one bank-sized chunk of slot rows
/// into a resident scan bank, `(word address, value)` in address order; empty
/// when the chunk is past the end of the table.
pub fn pack_scan_rows(rows: &[Vec<u64>], p: &StreamerParams, bank: u64, chunk: u64) -> Result<Vec<(u32, u32)>> {
    if bank > 1 { return value_error("scan bank must be 0 or 1"); }
    if rows.is_empty() { return value_error("scan rows must be non-empty"); }
    let first = chunk * p.bank_size;
    if first >= rows.len() as u64 { return Ok(Vec::new()); }
    let points = &rows[first as usize..(first + p.bank_size).min(rows.len() as u64) as usize];
    let slot_count = points[0].len();
    if points.iter().any(|row| row.len() != slot_count) { return value_error("scan rows must have equal widths"); }
    if slot_count as u64 > p.num_slots { return value_error("scan row has more slots than the wire geometry"); }
    let base = region_bases(p).scan + bank * p.bank_size * p.scan_words();
    let mut words = Vec::with_capacity(points.len() * p.num_slots as usize);
    for (offset, point) in points.iter().enumerate() {
        let row = base + offset as u64 * p.scan_words();
        for slot in 0..p.num_slots {
            let value = point.get(slot as usize).copied().unwrap_or(0);
            let value = checked(u128::from(value), p.tick_width, &format!("scan row {} slot {slot}", first + offset as u64))?;
            words.push(((row + slot) as u32, value as u32));
        }
    }
    Ok(words)
}

// ------------------------------------------------------------------ config file
/// The one geometry/board configuration (`fpga/board_config/streamer_config.json`).
#[derive(Clone, Debug)]
pub struct StreamerConfig {
    pub params: StreamerParams,
    pub fpga_part: String,
    pub clock_hz: f64,
    pub uart_baud: u64,
    pub target_pct: f64,
    pub source: Option<PathBuf>,
    pub warnings: Vec<String>,
    /// The parsed file, for the board manifest (`board.lanes`).
    pub tree: Json,
}

const CONFIG_TOP_LEVEL_FIELDS: [&str; 8] = ["_README", "_field_docs", "fpga_part", "clock_hz", "uart_baud", "target_pct", "params", "board"];

/// Where the deployed board configuration lives: `ZLC_PS_CONFIG`, else the
/// FPGA tree's `board_config/streamer_config.json`.
pub fn default_config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("ZLC_PS_CONFIG").filter(|value| !value.is_empty()) { return PathBuf::from(path); }
    fpga_root().join("board_config").join("streamer_config.json")
}

/// The FPGA tree the host validates against and the build reads: `board_config`
/// (streamer_config.json, board.xdc), the RTL and its Vivado scripts, in this
/// crate's `fpga/` of the checkout the program runs from (v2's build tree
/// that made the bitstream); `ZLC_PULSE_FPGA` points at another tree.
pub fn fpga_root() -> PathBuf {
    if let Some(path) = std::env::var_os("ZLC_PULSE_FPGA").filter(|value| !value.is_empty()) { return PathBuf::from(path); }
    zlc_core::checkout::path("crates/zlc-pulse/fpga")
}

/// `params_from_config(mapping)`: known fields over the defaults.
fn params_from(tree: Option<&Json>) -> Result<StreamerParams> {
    let mut params = StreamerParams::default();
    if let Some(Json::Object(items)) = tree {
        for (name, value) in items {
            if !PARAM_FIELDS.contains(&name.as_str()) { continue; }
            let Some(value) = value.as_int().and_then(|value| u64::try_from(value).ok()) else {
                return value_error(format!("{name} must be a non-negative integer"));
            };
            params.set(name, value);
        }
    }
    Ok(params)
}

/// `load_streamer_config(path)`: forgiving about what is missing (defaults and
/// warnings), never about what is ambiguous; the clock must be the frozen one.
pub fn load_streamer_config(path: Option<&Path>) -> Result<StreamerConfig> {
    let mut warnings = Vec::new();
    let candidate = path.map(Path::to_path_buf).unwrap_or_else(default_config_path);
    let (tree, source) = match std::fs::read_to_string(&candidate) {
        Ok(text) => match json::parse(&text, "streamer_config.json") {
            Ok(tree) => (tree, Some(candidate.clone())),
            Err(error) => { warnings.push(format!("could not read config {}: {error}", candidate.display())); (Json::Null, None) }
        },
        Err(error) if candidate.exists() => { warnings.push(format!("could not read config {}: {error}", candidate.display())); (Json::Null, None) }
        Err(_) => (Json::Null, None),
    };
    if source.is_none() { warnings.push("no streamer_config.json found; using built-in defaults.".into()); }
    let tree = if matches!(tree, Json::Object(_)) { tree } else {
        if source.is_some() { warnings.push("config root is not an object; using built-in defaults.".into()); }
        Json::Object(Vec::new())
    };
    let raw_params = tree.get("params");
    if source.is_some() {
        match raw_params {
            Some(Json::Object(items)) => {
                let mut missing: Vec<&str> = PARAM_FIELDS.iter().copied().filter(|name| !items.iter().any(|(key, _)| key == name)).collect();
                missing.sort_unstable();
                if !missing.is_empty() { warnings.push(format!("config params omit deployed fields: {}", missing.join(", "))); }
            }
            _ => warnings.push("config has no params object; using built-in defaults.".into()),
        }
        let missing: Vec<&str> = ["fpga_part", "clock_hz", "uart_baud", "target_pct"].into_iter().filter(|name| tree.get(name).is_none()).collect();
        if !missing.is_empty() { warnings.push(format!("config omits deployed fields: {}", missing.join(", "))); }
    }
    let params = params_from(raw_params.filter(|value| matches!(value, Json::Object(_)))).unwrap_or_else(|error| {
        warnings.push(format!("invalid params in config ({error}); using built-in defaults."));
        StreamerParams::default()
    });
    let clock_hz = match tree.get("clock_hz") { None => FROZEN_CLOCK_HZ, Some(value) => value.as_f64().ok_or_else(|| Error::new(Kind::Value, "clock_hz must be numeric"))? };
    if clock_hz != FROZEN_CLOCK_HZ { return value_error("clock_hz differs from the frozen RTL (5e+07 Hz)"); }
    let uart_baud = match tree.get("uart_baud") {
        None => DEFAULT_UART_BAUD,
        Some(value) => value.as_int().and_then(|value| u64::try_from(value).ok()).filter(|baud| *baud > 0)
            .ok_or_else(|| Error::new(Kind::Value, "uart_baud must be a positive integer"))?,
    };
    if uart_baud as f64 * 16.0 > clock_hz { return value_error("uart_baud requires more than one 16x oversampling tick per fabric clock"); }
    if let Err(error) = check_rtl_assumptions(&params) { warnings.push(format!("geometry violates a shipped-RTL assumption: {error}")); }
    Ok(StreamerConfig {
        params,
        fpga_part: tree.get("fpga_part").and_then(Json::as_str).unwrap_or(DEFAULT_FPGA_PART).to_owned(),
        clock_hz,
        uart_baud,
        target_pct: tree.get("target_pct").and_then(Json::as_f64).unwrap_or(DEFAULT_TARGET_PCT),
        source,
        warnings,
        tree,
    })
}

/// `require_streamer_config(path)`: one named file with nothing forgiven, the
/// door a server driving hardware goes through.
pub fn require_streamer_config(path: &Path) -> Result<StreamerConfig> {
    let text = std::fs::read_to_string(path).map_err(|error| Error::new(Kind::Io, format!("streamer config is missing: {} ({error})", path.display())))?;
    let tree = json::parse(&text, "streamer_config.json")?;
    let Json::Object(items) = &tree else { return value_error("streamer_config.json fields are not exact"); };
    let keys: std::collections::BTreeSet<&str> = items.iter().map(|(key, _)| key.as_str()).collect();
    if keys != CONFIG_TOP_LEVEL_FIELDS.into_iter().collect() { return value_error("streamer_config.json fields are not exact"); }
    let exact = tree.get("params").and_then(Json::as_object).is_some_and(|params| {
        params.iter().map(|(key, _)| key.as_str()).collect::<std::collections::BTreeSet<_>>() == PARAM_FIELDS.into_iter().collect()
    });
    if !exact { return value_error("streamer_config.json params fields are not exact"); }
    if !matches!(tree.get("board"), Some(Json::Object(_))) { return value_error("streamer_config.json board must be an object"); }
    if !tree.get("fpga_part").and_then(Json::as_str).is_some_and(|part| !part.trim().is_empty()) { return value_error("fpga_part must be non-empty text"); }
    let config = load_streamer_config(Some(path))?;
    if config.source.is_none() { return value_error("build config fell back from the requested file"); }
    if !config.warnings.is_empty() { return value_error(config.warnings.join("; ")); }
    Ok(config)
}
