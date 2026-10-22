//! Persistent read-only WebView mappings; per-frame borrow is a separate lifetime.
use std::collections::BTreeMap;
use serde_json::{Value, json};
use webview2_com::Microsoft::Web::WebView2::Win32::{
    ICoreWebView2Environment12, ICoreWebView2_17, ICoreWebView2SharedBuffer,
    COREWEBVIEW2_SHARED_BUFFER_ACCESS_READ_ONLY,
};
use windows_core::{Interface, HSTRING};
use wry::{WebView, WebViewExtWindows};
use crate::science::Front;

struct Slot { buffer: ICoreWebView2SharedBuffer, ready: bool, borrowed: Option<u64>, closed: bool }
struct Mapping {
    id: u64, panel: String, width: u32, height: u32,
    slots: Vec<Slot>, pending: Option<(u64, Front)>, retiring: bool, release_sent: bool,
}

pub struct Pixels {
    environment: ICoreWebView2Environment12,
    core: ICoreWebView2_17,
    mappings: BTreeMap<u64, Mapping>,
    current: BTreeMap<String, u64>,
    replacement: BTreeMap<String, (u64, Front)>,
    next_mapping: u64,
    failed: Option<String>,
    /// Browsers watching (share.rs): told every state and every picture the window is.
    share: Option<crate::share::Share>,
    /// What the page holds of the state (delta.rs).
    sent: std::cell::RefCell<crate::delta::Sent>,
}

impl Pixels {
    pub fn new(webview: &WebView) -> Result<Self, String> {
        let environment = webview.environment().cast().map_err(|e| format!("Shared pixel environment: {e}"))?;
        let core = unsafe { webview.controller().CoreWebView2() }.and_then(|x| x.cast())
            .map_err(|e| format!("Shared pixel presentation: {e}"))?;
        Ok(Self { environment, core, mappings: BTreeMap::new(), current: BTreeMap::new(),
            replacement: BTreeMap::new(), next_mapping: 1, failed: None, share: None, sent: Default::default() })
    }

    pub fn set_share(&mut self, share: crate::share::Share) { self.share = Some(share); }

    /// A message that is not a state (frames, mappings, errors), as it is.
    pub fn send(&self, value: &Value) -> Result<(), String> {
        let _send = crate::trace::span("ui.message",||json!({"type":value["type"],"request":value["request"]}));
        let text = { let _serialize = crate::trace::span("ui.serialize", || json!({})); value.to_string() };
        self.post(&text)
    }

    /// A state message as the change since what this page was last told
    /// (delta.rs), of the sections it `reads`; nothing when that is nothing
    /// and the request is not its own (`owns`).
    pub fn send_state(&self, state: &Value, reads: impl Fn(&str) -> bool, owns: impl Fn(&Value) -> bool) -> Result<(), String> {
        let Some(object) = state.as_object().filter(|object| object.get("type").and_then(Value::as_str) == Some("state")) else { return self.send(state) };
        let text = { let _encode = crate::trace::span("ui.serialize", || json!({})); self.sent.borrow_mut().encode(object, reads, owns) };
        match text { Some(text) => self.post(&text), None => Ok(()) }
    }

    /// The page was (re)loaded: it holds no state and no mapping yet. The
    /// mappings the last page held went with it; its fronts come in new ones.
    pub fn page_loaded(&mut self) {
        self.sent.borrow_mut().reset();
        for mapping in std::mem::take(&mut self.mappings).into_values() {
            for slot in mapping.slots.into_iter().filter(|slot| !slot.closed) { let _ = unsafe { slot.buffer.Close() }; }
        }
        self.current.clear();
        self.replacement.clear();
    }

    /// A message already written as JSON. It goes as a string the page parses
    /// (transport.ts): posted as JSON, WebView2 parsed it once more on this,
    /// the window's, thread (~35 ms for a 500-period pulse's 1.3 MB).
    pub fn post(&self, text: &str) -> Result<(), String> {
        let _post = crate::trace::span("ui.post", || json!({"chars": text.len()}));
        unsafe { self.core.PostWebMessageAsString(&HSTRING::from(text)) }
            .map_err(|e| format!("UI message: {e}"))
    }

    pub fn submit(&mut self, panel: String, sequence: u64, front: Front) -> Result<(), String> {
        if let Some(share) = &self.share { share.publish_front(&panel, sequence, &front); }
        if let Some(error) = &self.failed { return Err(error.clone()); }
        let result = self.submit_inner(panel, sequence, front);
        if let Err(error) = &result { self.fail(error.clone()); }
        result
    }

    fn submit_inner(&mut self, panel: String, sequence: u64, front: Front) -> Result<(), String> {
        let existing = self.current.get(&panel).copied();
        let matching = existing.filter(|id| self.mappings.get(id)
            .is_some_and(|m| !m.retiring && m.width == front.width && m.height == front.height));
        let id = if let Some(id) = matching { id } else {
            if let Some(old) = existing {
                // Keep only the newest replacement front. A new geometry cannot
                // allocate another three mappings until the previous set is gone.
                self.replacement.insert(panel, (sequence, front));
                return self.retire(old);
            }
            if self.mappings.values().any(|mapping| mapping.retiring) {
                // Document changes obey the same credit as a resize: retain
                // only the desired document's fronts while old views drain.
                self.replacement.insert(panel, (sequence, front));
                return Ok(());
            }
            let id = self.next_mapping;
            self.next_mapping += 1;
            let bytes = u64::from(front.width) * u64::from(front.height) * 4;
            if bytes == 0 || bytes != front.pixels.byte_len() as u64 { return Err("Invalid CPU front extent".into()); }
            self.mappings.insert(id, Mapping { id, panel: panel.clone(), width: front.width, height: front.height,
                slots: Vec::new(), pending: None, retiring: false, release_sent: false });
            self.current.insert(panel.clone(), id);
            for index in 0..3 {
                let buffer = unsafe { self.environment.CreateSharedBuffer(bytes) }.map_err(|e| e.to_string())?;
                let metadata = HSTRING::from(json!({"type":"shared-rgba-map", "mapping":id,
                    "slot":index, "panel":panel, "width":front.width, "height":front.height,
                    "stride":front.width * 4, "byte_length":bytes}).to_string());
                self.mappings.get_mut(&id).unwrap().slots.push(Slot { buffer, ready: false, borrowed: None, closed: false });
                let slot = &self.mappings[&id].slots[index];
                unsafe { self.core.PostSharedBufferToScript(&slot.buffer, COREWEBVIEW2_SHARED_BUFFER_ACCESS_READ_ONLY, &metadata) }
                    .map_err(|e| e.to_string())?;
            }
            id
        };
        self.mappings.get_mut(&id).unwrap().pending = Some((sequence, front));
        self.flush(id)
    }

    fn flush(&mut self, id: u64) -> Result<(), String> {
        let mapping = self.mappings.get_mut(&id).ok_or("Pixel mapping no longer exists")?;
        if mapping.retiring {
            if !mapping.release_sent && mapping.slots.iter().all(|s| s.borrowed.is_none()) {
                mapping.release_sent = true;
                return self.send(&json!({"type":"release_mappings", "mapping":id}));
            }
            return Ok(());
        }
        let Some(slot_index) = mapping.slots.iter().position(|s| s.ready && s.borrowed.is_none()) else { return Ok(()); };
        let Some((sequence, front)) = mapping.pending.take() else { return Ok(()); };
        let slot = &mut mapping.slots[slot_index];
        let mut pointer = std::ptr::null_mut();
        unsafe { slot.buffer.Buffer(&mut pointer) }.map_err(|e| e.to_string())?;
        if pointer.is_null() { return Err("Shared pixel mapping is null".into()); }
        let copy_span = crate::trace::span("front.copy",||json!({"panel":mapping.panel,"sequence":sequence,"bytes":front.pixels.byte_len()}));
        // Only a free slot is written. The browser ACKs after putImageData,
        // never on notification receipt; that ACK ends this immutable borrow.
        unsafe { std::ptr::copy_nonoverlapping(front.pixels.as_slice().as_ptr(), pointer, front.pixels.byte_len()); }
        drop(copy_span);
        slot.borrowed = Some(sequence);
        let message = json!({"type":"frame", "mapping":mapping.id, "slot":slot_index,
            "panel":mapping.panel, "sequence":sequence, "width":mapping.width, "height":mapping.height});
        crate::trace::point("front.post",||message.clone());
        self.send(&message)
    }

    pub fn acknowledge(&mut self, command: &Value) -> Result<(), String> {
        if command["action"]=="frame_consumed" {crate::trace::point("front.ack",||command.clone());}
        if self.failed.is_some() { return Ok(()); }
        let result = self.acknowledge_inner(command);
        if let Err(error) = &result { self.fail(error.clone()); }
        result
    }

    fn acknowledge_inner(&mut self, command: &Value) -> Result<(), String> {
        let id = command["mapping"].as_u64().ok_or("Missing pixel mapping identity")?;
        let mapping = self.mappings.get_mut(&id).ok_or("Unknown pixel mapping")?;
        match command["action"].as_str().unwrap_or("") {
            "mappings_released" => {
                if !mapping.release_sent || mapping.slots.iter().any(|s| s.borrowed.is_some()) {
                    return Err("Pixel mappings released before pending frames were consumed".into());
                }
                let mapping = self.mappings.remove(&id).unwrap();
                for slot in mapping.slots { unsafe { slot.buffer.Close() }.map_err(|e| e.to_string())?; }
                self.current.remove(&mapping.panel);
                if !self.mappings.values().any(|mapping| mapping.retiring) {
                    let pending = std::mem::take(&mut self.replacement);
                    for (panel, (sequence, front)) in pending {
                        self.submit(panel, sequence, front)?;
                    }
                }
                return Ok(());
            }
            action => {
                let index = command["slot"].as_u64().ok_or("Missing pixel slot")? as usize;
                let slot = mapping.slots.get_mut(index).ok_or("Pixel slot out of bounds")?;
                match action {
                    "map_ready" if !slot.ready => slot.ready = true,
                    "frame_consumed" if slot.borrowed.is_some() && slot.borrowed == command["sequence"].as_u64() => slot.borrowed = None,
                    _ => return Err("Pixel acknowledgement does not match its outstanding lease".into()),
                }
            }
        }
        self.flush(id)
    }

    fn retire(&mut self, id: u64) -> Result<(), String> {
        if let Some(mapping) = self.mappings.get_mut(&id) {
            mapping.retiring = true;
            mapping.pending = None;
        }
        self.flush(id)
    }

    /// The window's page as it is on screen -- its controls and every
    /// presented plot front -- as PNG bytes (Chromium's own capture of what it
    /// composed). `done` runs on this thread once the page has answered.
    pub fn capture(&self, done: impl FnOnce(Result<Vec<u8>, String>) + 'static) -> Result<(), String> {
        let handler = webview2_com::CallDevToolsProtocolMethodCompletedHandler::create(Box::new(move |result, answer: String| {
            done(result.map_err(|error| format!("Window capture: {error}")).and_then(|()| {
                let answer: Value = serde_json::from_str(&answer).map_err(|error| format!("Window capture: {error}"))?;
                base64(answer["data"].as_str().ok_or("Window capture returned no image")?)
            }));
            Ok(())
        }));
        unsafe { self.core.CallDevToolsProtocolMethod(&HSTRING::from("Page.captureScreenshot"), &HSTRING::from(r#"{"format":"png"}"#), &handler) }
            .map_err(|error| format!("Window capture: {error}"))
    }

    pub fn retain_panels(&mut self, panels: &[String]) -> Result<(), String> {
        if let Some(share) = &self.share { share.retain(panels); }
        self.replacement.retain(|key, _| panels.contains(key));
        let obsolete: Vec<_> = self.current.iter().filter(|(key,_)| !panels.contains(key)).map(|(key,id)| (key.clone(),*id)).collect();
        for (_key,id) in obsolete { self.retire(id)?; }
        Ok(())
    }

    fn fail(&mut self, error: String) {
        if self.failed.is_some() { return; }
        self.failed = Some(error);
        self.replacement.clear();
        for mapping in self.mappings.values_mut() {
            mapping.pending = None;
            mapping.retiring = true;
            // A failed post cannot prove whether the receiver saw the frame.
            // Stop ALL writes; never clear its lease and reuse the memory.
            for slot in &mut mapping.slots {
                if let Err(error) = unsafe { slot.buffer.Close() } { eprintln!("Pixel failure cleanup: {error}"); }
                slot.closed = true;
            }
        }
        for id in self.mappings.keys() {
            if let Err(error) = self.send(&json!({"type":"release_mappings", "mapping":id})) {
                eprintln!("Cannot release failed browser mapping: {error}");
            }
        }
    }
}

/// Standard base64 (RFC 4648, padded) to bytes.
fn base64(text: &str) -> Result<Vec<u8>, String> {
    let value = |byte: u8| -> Result<u32, String> {
        Ok(match byte { b'A'..=b'Z' => byte - b'A', b'a'..=b'z' => byte - b'a' + 26, b'0'..=b'9' => byte - b'0' + 52, b'+' => 62, b'/' => 63,
                        _ => return Err("Window capture is not base64".into()) } as u32)
    };
    let bytes = text.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut word = 0u32;
        for (index, byte) in chunk.iter().enumerate() { word |= value(*byte)? << (18 - 6 * index); }
        out.extend_from_slice(&word.to_be_bytes()[1..chunk.len()]);
    }
    Ok(out)
}

/// A PNG's width and height (its IHDR).
pub fn png_size(png: &[u8]) -> Option<(u32, u32)> {
    let size = |at: usize| png.get(at..at + 4).map(|bytes| u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    (png.starts_with(b"\x89PNG")).then(|| Some((size(16)?, size(20)?))).flatten()
}

impl Drop for Pixels {
    fn drop(&mut self) {
        // Window destruction stops writes; Close releases only native mappings.
        // WebView-owned readonly mappings remain valid until its page is destroyed.
        for mapping in self.mappings.values() {
            for slot in &mapping.slots {
                if !slot.closed {
                    if let Err(error) = unsafe { slot.buffer.Close() } { eprintln!("Pixel cleanup: {error}"); }
                }
            }
        }
    }
}
