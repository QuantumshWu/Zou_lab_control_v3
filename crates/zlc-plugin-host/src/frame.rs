//! Frames on a plugin's stdio (PLUGIN_PROTOCOL.md §4): a little-endian u32
//! byte length, then one UTF-8 JSON object.
use std::io::{self, Read, Write};

use serde_json::Value;

pub const MAX_FRAME: usize = 16 << 20;

pub fn write(sink: &mut impl Write, message: &Value) -> io::Result<()> {
    let payload = serde_json::to_vec(message).map_err(io::Error::other)?;
    if payload.len() > MAX_FRAME {
        return Err(io::Error::other(format!("frame of {} bytes exceeds {MAX_FRAME}", payload.len())));
    }
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload);
    sink.write_all(&frame)?;
    sink.flush()
}

/// The next message, or None at a clean end of stream.
pub fn read(source: &mut impl Read) -> io::Result<Option<Value>> {
    let mut header = [0u8; 4];
    let mut filled = 0;
    while filled < 4 {
        match source.read(&mut header[filled..])? {
            0 if filled == 0 => return Ok(None),
            0 => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "stream ended inside a frame header")),
            count => filled += count,
        }
    }
    let length = u32::from_le_bytes(header) as usize;
    if length > MAX_FRAME {
        return Err(io::Error::other(format!("incoming frame of {length} bytes exceeds {MAX_FRAME}")));
    }
    let mut payload = vec![0u8; length];
    source.read_exact(&mut payload)?;
    let message: Value = serde_json::from_slice(&payload).map_err(io::Error::other)?;
    if !message.is_object() { return Err(io::Error::other("a frame must hold one JSON object")); }
    Ok(Some(message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn frames_round_trip_and_a_short_stream_is_an_error() {
        let mut buffer = Vec::new();
        write(&mut buffer, &json!({"t": "req", "id": 1})).unwrap();
        write(&mut buffer, &json!({"t": "evt", "op": "release", "p": {"blocks": ["a"]}})).unwrap();
        let mut source = buffer.as_slice();
        assert_eq!(read(&mut source).unwrap().unwrap()["id"], 1);
        assert_eq!(read(&mut source).unwrap().unwrap()["p"]["blocks"][0], "a");
        assert!(read(&mut source).unwrap().is_none());
        let mut cut = &buffer[..6];
        assert!(read(&mut cut).is_err());
    }
}
