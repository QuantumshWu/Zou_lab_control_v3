//! Persistent framed helper for offline Python. Live plugins use these same
//! functions in their existing Rust host, without spawning a helper.
use std::io::{Read, Write};
use serde_json::{Value, json};

const MAX_FRAME: usize = 256 << 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut input, mut output) = (std::io::stdin().lock(), std::io::stdout().lock());
    let mut services = std::collections::HashMap::<String, zlc_native::Service>::new();
    loop {
        let mut header = [0u8;4];
        match input.read_exact(&mut header) {
            Ok(()) => {}, Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()), Err(e) => return Err(e.into()),
        }
        let size = u32::from_le_bytes(header) as usize;
        if size > MAX_FRAME { return Err("native request is too large".into()); }
        let mut bytes = vec![0;size]; input.read_exact(&mut bytes)?;
        let answer = match serde_json::from_slice::<Value>(&bytes) {
            Ok(request) => {
                let operation = request["operation"].as_str().unwrap_or("");
                let scope = request["scope"].as_str().unwrap_or("");
                if operation == "scope.release" { services.remove(scope); json!({"ok":null}) }
                else {
                    match services.entry(scope.into()).or_default().dispatch(operation, &request["params"], &zlc_native::arrays::decode, &zlc_native::arrays::encode) {
                        Ok(value) => json!({"ok":value}), Err(error) => json!({"error":{"kind":"invalid","message":error}}),
                    }
                }
            }
            Err(error) => json!({"error":{"kind":"invalid","message":error.to_string()}}),
        };
        let bytes = serde_json::to_vec(&answer)?;
        if bytes.len() > MAX_FRAME { return Err("native response is too large".into()); }
        output.write_all(&(bytes.len() as u32).to_le_bytes())?;
        output.write_all(&bytes)?;
        output.flush()?;
    }
}
