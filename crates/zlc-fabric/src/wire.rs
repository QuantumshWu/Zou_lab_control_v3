//! Frames between cores and the handshake that opens a connection.
//!
//! A frame is a u32 little-endian JSON length, the JSON object, then the raw
//! bytes of every block it names in `attach` (`[{name, size}]`, in order).
//! `$array` references inside the JSON name the sender's blocks; the receiver
//! reads each block straight into a block of its own core and renames the
//! references. `lent` lists the blocks with the multiplicity the plugin
//! protocol counts holds by.
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use zlc_plugin_host::{Lend, Registry};
use zlc_core::ForeignBytes;

use crate::{Bench, FABRIC_VERSION};

pub const MAX_JSON: usize = 16 << 20;
/// The largest block one frame may carry (a camera frame is a few MB).
pub const MAX_BLOCK: usize = 1 << 30;
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Write one frame; the blocks' bytes follow the JSON without another copy.
pub fn write_frame(sink: &mut impl Write, message: &Value, blocks: &[Arc<Lend>]) -> io::Result<()> {
    let mut distinct: Vec<&Arc<Lend>> = Vec::new();
    for lend in blocks { if !distinct.iter().any(|seen| seen.name() == lend.name()) { distinct.push(lend); } }
    let mut message = message.clone();
    if !blocks.is_empty() {
        message["attach"] = Value::Array(distinct.iter().map(|lend| json!({"name": lend.name(), "size": lend.bytes().len()})).collect());
        message["lent"] = Value::Array(blocks.iter().map(|lend| json!(lend.name())).collect());
    }
    let payload = serde_json::to_vec(&message).map_err(io::Error::other)?;
    if payload.len() > MAX_JSON { return Err(io::Error::other(format!("a fabric message of {} bytes exceeds {MAX_JSON}", payload.len()))); }
    let mut head = Vec::with_capacity(4 + payload.len());
    head.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    head.extend_from_slice(&payload);
    sink.write_all(&head)?;
    for lend in distinct { sink.write_all(lend.bytes())?; }
    sink.flush()
}

/// The next frame (None at a clean end): its message, with `$array`s renamed
/// to this core's blocks, and one hold per `lent` listing.
pub fn read_frame(source: &mut impl Read, registry: &Arc<Registry>) -> io::Result<Option<(Value, Vec<Arc<Lend>>)>> {
    let mut header = [0u8; 4];
    let mut filled = 0;
    while filled < 4 {
        match source.read(&mut header[filled..])? {
            0 if filled == 0 => return Ok(None),
            0 => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the fabric stream ended inside a frame")),
            count => filled += count,
        }
    }
    let length = u32::from_le_bytes(header) as usize;
    if length > MAX_JSON { return Err(io::Error::other(format!("a fabric message of {length} bytes exceeds {MAX_JSON}"))); }
    let mut payload = vec![0u8; length];
    source.read_exact(&mut payload)?;
    let mut message: Value = serde_json::from_slice(&payload).map_err(io::Error::other)?;
    if !message.is_object() { return Err(io::Error::other("a fabric frame holds one JSON object")); }
    let mut local: HashMap<String, Arc<Lend>> = HashMap::new();
    for item in message.get("attach").and_then(Value::as_array).cloned().unwrap_or_default() {
        let (name, size) = (item["name"].as_str().unwrap_or("").to_owned(), item["size"].as_u64().unwrap_or(0) as usize);
        if size > MAX_BLOCK { return Err(io::Error::other(format!("a fabric block of {size} bytes exceeds {MAX_BLOCK}"))); }
        let lend = registry.export_with(size, |block| source.read_exact(block).map_err(|error| error.to_string())).map_err(io::Error::other)?;
        local.insert(name, Arc::new(lend));
    }
    rename(&mut message, &local);
    let lends = message.get("lent").and_then(Value::as_array).into_iter().flatten()
        .filter_map(|name| name.as_str().and_then(|name| local.get(name)).cloned()).collect();
    if let Some(object) = message.as_object_mut() { object.remove("attach"); object.remove("lent"); }
    Ok(Some((message, lends)))
}

/// Point every `$array` of a sender's block at the receiver's copy.
fn rename(value: &mut Value, local: &HashMap<String, Arc<Lend>>) {
    match value {
        Value::Object(object) => {
            if let Some(spec) = object.get_mut("$array").and_then(Value::as_object_mut) {
                if let Some(lend) = spec.get("block").and_then(Value::as_str).and_then(|name| local.get(name)) {
                    spec.insert("block".into(), json!(lend.name()));
                    spec.insert("size".into(), json!(lend.bytes().len()));
                }
                return;
            }
            for item in object.values_mut() { rename(item, local); }
        }
        Value::Array(items) => for item in items { rename(item, local); },
        _ => {}
    }
}

/// A frame of plain JSON (the handshake, beats, events).
pub fn send(stream: &mut TcpStream, message: &Value) -> io::Result<()> { write_frame(stream, message, &[]) }

fn version_skew(peer: &str, spoken: &Value) -> String {
    format!("{peer} speaks fabric version {spoken} and this machine speaks {FABRIC_VERSION}; update the older side and restart it")
}

/// Read one plain frame within the handshake's time.
fn expect(stream: &mut TcpStream, registry: &Arc<Registry>) -> Result<Value, String> {
    stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT)).map_err(|error| error.to_string())?;
    match read_frame(stream, registry) {
        Ok(Some((message, _))) => Ok(message),
        Ok(None) => Err("the peer closed the connection".into()),
        Err(error) if matches!(error.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock) => Err("the peer did not answer the handshake".into()),
        Err(error) => Err(error.to_string()),
    }
}

/// The using side: say its version and machine; the publisher's name (v2's fabric: no key).
pub fn client_handshake(stream: &mut TcpStream, bench: &Bench, registry: &Arc<Registry>) -> Result<String, String> {
    send(stream, &json!({"t": "hello", "fabric": FABRIC_VERSION, "client": bench.client()})).map_err(|error| error.to_string())?;
    let hello = expect(stream, registry)?;
    if hello["t"] == "refused" { return Err(hello["message"].as_str().unwrap_or("refused").to_owned()); }
    if hello["fabric"] != json!(FABRIC_VERSION) { return Err(version_skew("the publisher", &hello["fabric"])); }
    Ok(hello["server"].as_str().unwrap_or("").to_owned())
}

/// The publishing side: the client's name, once it speaks this fabric version.
pub fn server_handshake(stream: &mut TcpStream, bench: &Bench, registry: &Arc<Registry>) -> Result<String, String> {
    let hello = expect(stream, registry)?;
    if hello["t"] != "hello" { return Err("the peer did not start with hello".into()); }
    if hello["fabric"] != json!(FABRIC_VERSION) {
        let message = version_skew("the peer", &hello["fabric"]);
        let _ = send(stream, &json!({"t": "refused", "fabric": FABRIC_VERSION, "message": message}));
        return Err(message);
    }
    send(stream, &json!({"t": "hello", "fabric": FABRIC_VERSION, "server": bench.name})).map_err(|error| error.to_string())?;
    Ok(hello["client"].as_str().unwrap_or("peer").to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_follow_their_blocks() {
        let mut message = json!({"ok": {"frame": {"$array": {"block": "Local\\zlc.a.1.1", "size": 65536, "offset": 0}}, "n": [1, {"$inline": {}}]}});
        let registry = zlc_plugin_host::Registry::standalone("fabric-test");
        let lend = Arc::new(registry.export(&[1, 2, 3]).unwrap());
        let local: HashMap<String, Arc<Lend>> = [("Local\\zlc.a.1.1".to_owned(), lend.clone())].into();
        rename(&mut message, &local);
        assert_eq!(message["ok"]["frame"]["$array"]["block"], json!(lend.name()));
        assert_eq!(message["ok"]["frame"]["$array"]["size"], json!(65536));
    }
}
