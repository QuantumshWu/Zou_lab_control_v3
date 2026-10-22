//! The pulse target generated from the explicit board manifest
//! (`streamer_config.json` `board.lanes`), validated against both of its
//! projections: the XDC pin constraints and the RTL top's output assigns.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Kind, Result, repr, value_error};
use crate::json::Json;
use crate::model::{PortSpec, Target};
use crate::wire::{StreamerConfig, StreamerParams, fpga_root, load_streamer_config};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role { Digital, DacData, DacClock }

#[derive(Clone, Debug)]
struct Lane { index: u64, logical: String, rtl_port: String, pin: String, role: Role, bus: Option<u64>, bit: Option<u64> }

fn board_lanes(config: &StreamerConfig, params: &StreamerParams) -> Result<Vec<Lane>> {
    let path = config.source.as_ref().map(|path| path.display().to_string()).unwrap_or_default();
    let board = config.tree.get("board");
    let rows = board.and_then(|board| board.get("lanes"));
    if !board.and_then(|board| board.get("id")).and_then(Json::as_str).is_some_and(|id| !id.trim().is_empty()) {
        return value_error(format!("{path} board.id must be non-empty text"));
    }
    let Some(Json::Array(rows)) = rows else { return value_error(format!("{path} board.lanes must be a list")); };
    let common = ["index", "logical_signal", "rtl_port", "package_pin", "electrical_role"];
    let mut lanes = Vec::new();
    for (position, row) in rows.iter().enumerate() {
        let Json::Object(fields) = row else { return value_error(format!("board lane {position} must be an object")); };
        let role = match row.get("electrical_role").and_then(Json::as_str) {
            Some("digital") => Some(Role::Digital), Some("dac_data") => Some(Role::DacData), Some("dac_clock") => Some(Role::DacClock), _ => None,
        };
        let extra: &[&str] = match role { Some(Role::DacData) => &["bus_index", "bit_index"], Some(Role::DacClock) => &["bus_index"], _ => &[] };
        let keys: BTreeSet<&str> = fields.iter().map(|(key, _)| key.as_str()).collect();
        let expected: BTreeSet<&str> = common.iter().chain(extra).copied().collect();
        let Some(role) = role.filter(|_| keys == expected) else {
            return value_error(format!("board lane {position} has invalid fields or electrical_role"));
        };
        let Some(index) = row.get("index").and_then(Json::as_int) else { return value_error(format!("board lane {position} index must be an integer")); };
        let text = |name: &str| -> Result<String> {
            row.get(name).and_then(Json::as_str).filter(|value| !value.trim().is_empty()).map(str::to_owned)
                .ok_or_else(|| Error::new(Kind::Value, format!("board lane {index} {name} must be non-empty text")))
        };
        let (logical, rtl_port, pin) = (text("logical_signal")?, text("rtl_port")?, text("package_pin")?);
        let bus = row.get("bus_index").and_then(Json::as_int);
        let bit = row.get("bit_index").and_then(Json::as_int);
        if !extra.is_empty() && bus.is_none() { return value_error(format!("board lane {index} bus_index must be an integer")); }
        if role == Role::DacData && bit.is_none() { return value_error(format!("board lane {index} bit_index must be an integer")); }
        let unsigned = |value: Option<i128>| value.map(|value| value.max(0) as u64);
        lanes.push(Lane { index: index.max(0) as u64, logical, rtl_port, pin, role, bus: unsigned(bus), bit: unsigned(bit) });
    }
    lanes.sort_by_key(|lane| lane.index);
    if !lanes.iter().map(|lane| lane.index).eq(0..params.channel_count) {
        return value_error(format!("board lane indices must be exactly 0..{}", params.channel_count - 1));
    }
    if lanes.iter().any(|lane| (lane.role == Role::Digital) != (lane.index < params.num_delay_ch())) {
        return value_error("board digital lanes must be the leading TTL control domain");
    }
    for (label, values) in [("rtl_port", lanes.iter().map(|lane| &lane.rtl_port).collect::<Vec<_>>()),
                            ("package_pin", lanes.iter().map(|lane| &lane.pin).collect())] {
        if values.iter().collect::<BTreeSet<_>>().len() != values.len() { return value_error(format!("board lane {label} values must be unique")); }
    }
    for bus in 0..params.bus_count {
        let mut data: Vec<&Lane> = lanes.iter().filter(|lane| lane.role == Role::DacData && lane.bus == Some(bus)).collect();
        data.sort_by_key(|lane| lane.bit);
        let clocks = lanes.iter().filter(|lane| lane.role == Role::DacClock && lane.bus == Some(bus)).count();
        if !data.iter().map(|lane| lane.bit.unwrap_or(u64::MAX)).eq(0..params.bus_width) || clocks != 1 {
            return value_error(format!("board DAC bus {bus} must have bits 0..{} and one clock", params.bus_width - 1));
        }
        if data.iter().map(|lane| &lane.logical).collect::<BTreeSet<_>>().len() != 1 {
            return value_error(format!("board DAC bus {bus} data bits must share one logical signal"));
        }
    }
    if lanes.iter().any(|lane| lane.bus.is_some_and(|bus| bus >= params.bus_count)) {
        return value_error(format!("board bus_index must be in 0..{}", params.bus_count - 1));
    }
    Ok(lanes)
}

fn word_char(byte: u8) -> bool { byte.is_ascii_alphanumeric() || byte == b'_' }

/// `[A-Za-z_][A-Za-z0-9_]*(\[\d+\])?` at the start of `text`: its length.
fn port_name(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if !bytes.first().is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_') { return None; }
    let mut at = 1;
    while at < bytes.len() && word_char(bytes[at]) { at += 1; }
    if bytes.get(at) == Some(&b'[') {
        let digits = bytes[at + 1..].iter().take_while(|b| b.is_ascii_digit()).count();
        if digits > 0 && bytes.get(at + 1 + digits) == Some(&b']') { at += digits + 2; }
    }
    Some(at)
}

/// `[\s*get_ports\s+({ ... }|name)\s*]` at the start of `text`: the port.
fn get_ports(text: &str) -> Option<String> {
    let rest = text.strip_prefix('[')?.trim_start();
    if !rest.get(..9).is_some_and(|word| word.eq_ignore_ascii_case("get_ports")) { return None; }
    let after = &rest[9..];
    let rest = after.trim_start();
    if rest.len() == after.len() { return None; }
    let (port, rest) = if let Some(inner) = rest.strip_prefix('{') {
        let close = inner.find(['{', '}'])?;
        if inner.as_bytes()[close] != b'}' { return None; }
        let port = inner[..close].trim();
        if port.is_empty() { return None; }
        (port.to_owned(), &inner[close + 1..])
    } else {
        let length = port_name(rest)?;
        (rest[..length].to_owned(), &rest[length..])
    };
    rest.trim_start().starts_with(']').then_some(port)
}

fn system_port(port: &str) -> bool {
    let lower = port.to_ascii_lowercase();
    if ["clk", "reset", "start", "running", "done", "uart_rx", "uart_tx", "led"].contains(&lower.as_str()) { return true; }
    if let Some(digits) = lower.strip_prefix("gnd") { return digits.bytes().all(|b| b.is_ascii_digit()); }
    lower.strip_prefix("led[").and_then(|rest| rest.strip_suffix(']')).is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

fn xdc_ports(path: &Path) -> Result<BTreeMap<String, String>> {
    let text = std::fs::read(path).map_err(|error| Error::new(Kind::Io, format!("{}: {error}", path.display())))?;
    let text = String::from_utf8_lossy(&text);
    let mut result = BTreeMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        let upper = line.to_ascii_uppercase();
        let mut found = None;
        let mut search = 0;
        while let Some(offset) = upper[search..].find("PACKAGE_PIN") {
            let start = search + offset;
            search = start + 1;
            if start > 0 && word_char(line.as_bytes()[start - 1]) { continue; }
            let after = &line[start + 11..];
            let gap = after.len() - after.trim_start().len();
            if gap == 0 { continue; }
            let pin_text = &after[gap..];
            let pin_length = pin_text.find(|c: char| c.is_whitespace() || c == '}' || c == ']').unwrap_or(pin_text.len());
            if pin_length == 0 { continue; }
            let pin = &pin_text[..pin_length];
            let tail = &pin_text[pin_length..];
            if let Some(port) = tail.match_indices('[').find_map(|(at, _)| get_ports(&tail[at..])) { found = Some((pin.to_owned(), port)); break; }
        }
        let Some((pin, port)) = found else { continue };
        if system_port(&port) { continue; }
        if let Some(existing) = result.get(&port) {
            if *existing != pin { return value_error(format!("XDC port {port} has conflicting PACKAGE_PIN declarations")); }
        }
        result.insert(port, pin);
    }
    Ok(result)
}

fn strip_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while !rest.is_empty() {
        let line = rest.find("//");
        let block = rest.find("/*");
        match (line, block) {
            (Some(at), block) if block.is_none_or(|b| at < b) => {
                out.push_str(&rest[..at]);
                out.push(' ');
                rest = rest[at..].find('\n').map_or("", |end| &rest[at + end..]);
            }
            (_, Some(at)) => {
                out.push_str(&rest[..at]);
                match rest[at + 2..].find("*/") {
                    Some(end) => { out.push(' '); rest = &rest[at + 2 + end + 2..]; }
                    None => { out.push_str(&rest[at..]); rest = ""; }
                }
            }
            _ => { out.push_str(rest); rest = ""; }
        }
    }
    out
}

fn top_assignments(path: &Path) -> Result<BTreeMap<String, String>> {
    let text = std::fs::read(path).map_err(|error| Error::new(Kind::Io, format!("{}: {error}", path.display())))?;
    let source = strip_comments(&String::from_utf8_lossy(&text));
    let mut actual = BTreeMap::new();
    let mut search = 0;
    while let Some(offset) = source[search..].find("assign") {
        let start = search + offset;
        search = start + 1;
        if start > 0 && word_char(source.as_bytes()[start - 1]) { continue; }
        let after = &source[start + 6..];
        let rest = after.trim_start();
        if rest.len() == after.len() { continue; }
        let Some(length) = port_name(rest) else { continue };
        let port = &rest[..length];
        let Some(rest) = rest[length..].trim_start().strip_prefix('=') else { continue };
        let rest = rest.trim_start();
        let Some(name) = ["bus_out_final", "bus_clk_final", "out_final"].into_iter().find(|name| rest.starts_with(&format!("{name}["))) else { continue };
        let digits = rest[name.len() + 1..].bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 || rest.as_bytes().get(name.len() + 1 + digits) != Some(&b']') { continue; }
        let source_text = &rest[..name.len() + digits + 2];
        if !rest[source_text.len()..].trim_start().starts_with(';') { continue; }
        if actual.insert(port.to_owned(), source_text.to_owned()).is_some() {
            return value_error(format!("RTL top ({}) assigns {port} more than once", path.display()));
        }
        search = start + 6;
    }
    Ok(actual)
}

/// `pulse_target_from_xdc(path, config_path, top_path)`: the host target from
/// the explicit lane indices, both projections validated.
pub fn pulse_target_from_xdc(xdc: Option<&Path>, config_path: Option<&Path>, top: Option<&Path>) -> Result<Arc<Target>> {
    let config = load_streamer_config(config_path)?;
    if config.source.is_none() { return value_error("an explicit streamer_config.json board manifest is required"); }
    target_from_config(&config, xdc, top)
}

pub fn target_from_config(config: &StreamerConfig, xdc: Option<&Path>, top: Option<&Path>) -> Result<Arc<Target>> {
    let params = config.params;
    let lanes = board_lanes(config, &params)?;
    let xdc = xdc.map(Path::to_path_buf).unwrap_or_else(|| fpga_root().join("board_config").join("board.xdc"));
    let actual = xdc_ports(&xdc)?;
    let expected: BTreeMap<String, String> = lanes.iter().map(|lane| (lane.rtl_port.clone(), lane.pin.clone())).collect();
    if actual != expected {
        let ports: BTreeSet<&String> = actual.keys().chain(expected.keys()).collect();
        let details: Vec<String> = ports.into_iter().filter(|port| actual.get(*port) != expected.get(*port))
            .map(|port| format!("{port}: manifest={}, XDC={}", opt_repr(expected.get(port)), opt_repr(actual.get(port)))).collect();
        return value_error(format!("XDC differs from explicit board manifest ({}): {}", xdc.display(), details.join("; ")));
    }
    let top = top.map(Path::to_path_buf).unwrap_or_else(|| fpga_root().join("pulse_streamer").join("zlc_pulse_streamer_top.v"));
    let assigns = top_assignments(&top)?;
    let mut errors = Vec::new();
    for lane in &lanes {
        let expected = match lane.role {
            Role::DacData => format!("bus_out_final[{}]", lane.bus.unwrap_or(0) * params.bus_width + lane.bit.unwrap_or(0)),
            Role::DacClock => format!("bus_clk_final[{}]", lane.bus.unwrap_or(0)),
            Role::Digital => format!("out_final[{}]", lane.index),
        };
        if assigns.get(&lane.rtl_port) != Some(&expected) {
            errors.push(format!("{}: manifest={expected}, RTL={}", lane.rtl_port, opt_repr(assigns.get(&lane.rtl_port))));
        }
    }
    if !errors.is_empty() { return value_error(format!("RTL top differs from explicit board manifest ({}): {}", top.display(), errors.join("; "))); }
    target_of(&lanes, &params)
}

fn opt_repr(value: Option<&String>) -> String { value.map_or_else(|| "None".to_owned(), |text| repr(text)) }

fn lane_name(index: u64) -> String { format!("ch{index:02}") }

fn target_of(lanes: &[Lane], params: &StreamerParams) -> Result<Arc<Target>> {
    let mut ports = Vec::new();
    let mut emitted = BTreeSet::new();
    for lane in lanes {
        if lane.role == Role::DacData {
            let bus = lane.bus.unwrap_or(0);
            if !emitted.insert(bus) { continue; }
            let mut data: Vec<&Lane> = lanes.iter().filter(|item| item.role == Role::DacData && item.bus == Some(bus)).collect();
            data.sort_by_key(|item| item.bit);
            let clock = lanes.iter().find(|item| item.role == Role::DacClock && item.bus == Some(bus)).expect("validated clock");
            let names: Vec<String> = data.iter().map(|item| lane_name(item.index)).collect();
            ports.push(PortSpec::new(&lane.logical, "dac", &names, &lane.logical, Some(bus.into()), None, None, None, Some(&clock.logical))?);
        } else {
            let kind = if lane.role == Role::DacClock { "clock" } else { "digital" };
            ports.push(PortSpec::new(&lane.logical, kind, &[lane_name(lane.index)], &lane.logical, None, None, None, None, None)?);
        }
    }
    let _ = params;
    Target::new(&lanes.iter().map(|lane| lane_name(lane.index)).collect::<Vec<_>>(), ports,
                lanes.iter().map(|lane| (lane_name(lane.index), lane.pin.clone())).collect())
}
