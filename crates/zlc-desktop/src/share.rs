//! Browser sharing (v2 `apps/share.py`): the window, read-only, in a browser
//! on another computer of the lab network.
//!
//! v2 captured its Qt windows into an H.264 WebRTC stream and injected the
//! viewer's mouse and keys back into them. v3's window is a page fed by
//! state messages and finished plot pixels, so a browser is served that same
//! page with the same messages: the state as a server-sent event stream (each
//! message carries only the sections that changed), every new plot picture as
//! a PNG of exactly its pixels. A shared viewer changes nothing -- the
//! instrument is driven at the instrument, or by MCP clients with their token
//! -- so no command is taken from a browser at all.
//!
//! Off by default (Start sharing in the Device Manager). Every browser waits
//! for the operator's Allow, as in v2; a reconnect is a new request. Only the
//! page, its assets, the event stream and the pictures are served, to Host
//! names of this machine (no DNS rebinding), never a file outside the UI.
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::science::Front;

/// Sections a viewer is not told: the operator's own request bookkeeping.
const PRIVATE: [&str; 5] = ["request", "result", "error", "busy", "message"];
const BEAT: Duration = Duration::from_secs(15);

struct Client { address: String, browser: String, approved: bool, token: String, sender: mpsc::Sender<Option<String>> }

struct Listening { port: u16, urls: Vec<String>, hosts: BTreeSet<String>, stop: Arc<AtomicBool>, wake: SocketAddr, thread: Option<std::thread::JoinHandle<()>> }

struct Inner {
    ui_root: PathBuf,
    notify: Box<dyn Fn() + Send + Sync>,
    /// The newest of every state section, and the latest picture of every panel (sequence, front, its PNG once asked).
    snapshot: Mutex<Map<String, Value>>,
    fronts: Mutex<BTreeMap<String, (u64, Front, Option<Arc<Vec<u8>>>)>>,
    clients: Mutex<BTreeMap<String, Client>>,
    listening: Mutex<Option<Listening>>,
    error: Mutex<Option<String>>,
    dpr: Mutex<f64>,
    active: AtomicBool,
    next: AtomicU64,
}

#[derive(Clone)]
pub struct Share(Arc<Inner>);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> { mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

fn secret() -> String { uuid::Uuid::new_v4().simple().to_string() }

/// This machine's IPv4 addresses a lab computer can reach (no loopback, no link-local).
fn local_addresses() -> BTreeSet<IpAddr> {
    let mut found = BTreeSet::new();
    // The route out (no packet is sent), and every address the machine's own name has.
    if let Ok(socket) = UdpSocket::bind(("0.0.0.0", 0)) {
        if socket.connect(("192.0.2.1", 9)).is_ok() { if let Ok(address) = socket.local_addr() { found.insert(address.ip()); } }
    }
    if let Ok(name) = std::env::var("COMPUTERNAME") {
        if let Ok(addresses) = (name.as_str(), 0).to_socket_addrs() { found.extend(addresses.map(|address| address.ip())); }
    }
    found.retain(|ip| matches!(ip, IpAddr::V4(v4) if !v4.is_loopback() && !v4.is_link_local() && !v4.is_unspecified()));
    found
}

impl Share {
    pub fn new(ui_root: PathBuf, notify: impl Fn() + Send + Sync + 'static) -> Share {
        Share(Arc::new(Inner { ui_root, notify: Box::new(notify), snapshot: Mutex::default(), fronts: Mutex::default(), clients: Mutex::default(),
            listening: Mutex::new(None), error: Mutex::new(None), dpr: Mutex::new(1.0), active: AtomicBool::new(false), next: AtomicU64::new(1) }))
    }

    /// The `share` state section (the Device Manager's Browser sharing card).
    pub fn state(&self) -> Value {
        let listening = lock(&self.0.listening);
        let clients: Vec<Value> = lock(&self.0.clients).iter().map(|(id, client)| json!({"id": id, "address": client.address,
            "browser": client.browser, "status": if client.approved { "connected" } else { "pending" }})).collect();
        json!({"active": listening.is_some(), "urls": listening.as_ref().map(|open| open.urls.clone()).unwrap_or_default(),
               "port": listening.as_ref().map(|open| open.port), "clients": clients, "error": *lock(&self.0.error)})
    }

    /// The operator's frames are this many device px per CSS px (fronts are drawn at it).
    pub fn set_dpr(&self, dpr: f64) { *lock(&self.0.dpr) = dpr; }

    /// Start sharing what the window shows now (`state`, and each panel's latest picture).
    pub fn start(&self, state: &Value, fronts: Vec<(String, u64, Front)>) -> Result<(), String> {
        if lock(&self.0.listening).is_some() { return Ok(()); }
        let bind = std::env::var("ZLC_SHARE_BIND").ok().filter(|text| !text.trim().is_empty()).unwrap_or_else(|| "0.0.0.0".into());
        // v2: port 80, else 18865 when 80 is taken or not ours to bind.
        let ports: Vec<u16> = match std::env::var("ZLC_SHARE_PORT").ok().and_then(|text| text.parse().ok()) { Some(port) => vec![port], None => vec![80, 18865, 0] };
        let mut failure = String::new();
        let listener = ports.iter().find_map(|port| TcpListener::bind((bind.as_str(), *port)).map_err(|error| failure = error.to_string()).ok());
        let Some(listener) = listener else {
            let error = format!("Sharing could not start: {failure}");
            *lock(&self.0.error) = Some(error.clone());
            return Err(error);
        };
        let mut wake = listener.local_addr().map_err(|error| error.to_string())?;
        if wake.ip().is_unspecified() { wake.set_ip(std::net::Ipv4Addr::LOCALHOST.into()); }
        let port = wake.port();
        let addresses: BTreeSet<IpAddr> = if bind == "0.0.0.0" { local_addresses() } else { bind.parse().into_iter().collect() };
        let suffix = if port == 80 { String::new() } else { format!(":{port}") };
        let mut urls: Vec<String> = addresses.iter().filter(|ip| !ip.is_loopback()).map(|ip| format!("http://{ip}{suffix}/zlc/")).collect();
        if urls.is_empty() { urls.push(format!("http://127.0.0.1{suffix}/zlc/")); }
        let mut hosts: BTreeSet<String> = addresses.iter().map(IpAddr::to_string).collect();
        hosts.extend(["localhost".to_owned(), "127.0.0.1".to_owned(), "[::1]".to_owned()]);
        if let Ok(name) = std::env::var("COMPUTERNAME") { hosts.insert(name.to_lowercase()); }
        {
            let mut snapshot = lock(&self.0.snapshot);
            snapshot.clear();
            for (key, value) in state.as_object().into_iter().flatten() { if !PRIVATE.contains(&key.as_str()) { snapshot.insert(key.clone(), value.clone()); } }
        }
        *lock(&self.0.fronts) = fronts.into_iter().map(|(panel, sequence, front)| (panel, (sequence, front, None))).collect();
        let stop = Arc::new(AtomicBool::new(false));
        let (serving, flag) = (self.clone(), stop.clone());
        let thread = std::thread::Builder::new().name("zlc-share-http".into()).spawn(move || {
            for stream in listener.incoming() {
                if flag.load(Ordering::Acquire) { break; }
                let Ok(stream) = stream else { continue };
                let share = serving.clone();
                let _ = std::thread::Builder::new().name("zlc-share-connection".into()).spawn(move || share.connection(stream));
            }
        }).map_err(|error| error.to_string())?;
        *lock(&self.0.error) = None;
        *lock(&self.0.listening) = Some(Listening { port, urls, hosts, stop, wake, thread: Some(thread) });
        self.0.active.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop sharing: every browser is told, nothing listens any more.
    pub fn stop(&self) {
        let Some(mut listening) = lock(&self.0.listening).take() else { return };
        self.0.active.store(false, Ordering::Release);
        listening.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect_timeout(&listening.wake, Duration::from_millis(200));
        if let Some(thread) = listening.thread.take() { let _ = thread.join(); }
        for (_, client) in std::mem::take(&mut *lock(&self.0.clients)) { close(&client, "Experiment sharing stopped."); }
        lock(&self.0.snapshot).clear();
        lock(&self.0.fronts).clear();
    }

    /// The operator's Allow / Reject (or Disconnect) for one browser.
    pub fn decide(&self, id: &str, allow: bool) -> Result<(), String> {
        let mut clients = lock(&self.0.clients);
        let client = clients.get_mut(id).ok_or_else(|| format!("No browser {id} is asking or connected"))?;
        if !allow {
            let client = clients.remove(id).expect("found");
            close(&client, if client.approved { "Disconnected by the experiment computer." } else { "Connection rejected by the experiment computer." });
            return Ok(());
        }
        if client.approved { return Ok(()); }
        client.approved = true;
        // Hello, the window as it stands, then every picture it shows.
        let hello = json!({"type": "hello", "client": id, "token": client.token, "dpr": *lock(&self.0.dpr)});
        let _ = client.sender.send(Some(hello.to_string()));
        let mut state = Value::Object(lock(&self.0.snapshot).clone());
        state["type"] = json!("state");
        let _ = client.sender.send(Some(state.to_string()));
        for (panel, (sequence, front, _)) in lock(&self.0.fronts).iter() {
            let _ = client.sender.send(Some(frame_notice(panel, *sequence, front).to_string()));
        }
        Ok(())
    }

    /// A state message the window was sent: the sections that changed go to every viewer.
    pub fn publish_state(&self, message: &Value) {
        if !self.0.active.load(Ordering::Acquire) { return; }
        let Some(object) = message.as_object() else { return };
        let mut changed = Map::new();
        {
            let mut snapshot = lock(&self.0.snapshot);
            for (key, value) in object {
                if key == "type" || PRIVATE.contains(&key.as_str()) || snapshot.get(key) == Some(value) { continue; }
                snapshot.insert(key.clone(), value.clone());
                changed.insert(key.clone(), value.clone());
            }
        }
        if changed.is_empty() { return; }
        changed.insert("type".into(), json!("state"));
        self.tell(&Value::Object(changed).to_string());
    }

    /// A panel's new picture: kept (its PNG made when a viewer asks) and announced.
    pub fn publish_front(&self, panel: &str, sequence: u64, front: &Front) {
        if !self.0.active.load(Ordering::Acquire) { return; }
        lock(&self.0.fronts).insert(panel.to_owned(), (sequence, front.clone(), None));
        self.tell(&frame_notice(panel, sequence, front).to_string());
    }

    /// Panels the window no longer shows are forgotten.
    pub fn retain(&self, panels: &[String]) { lock(&self.0.fronts).retain(|panel, _| panels.contains(panel)); }

    fn tell(&self, text: &str) {
        for client in lock(&self.0.clients).values().filter(|client| client.approved) { let _ = client.sender.send(Some(text.to_owned())); }
    }

    // ------------------------------------------------------------ HTTP
    fn connection(&self, stream: TcpStream) {
        let peer = stream.peer_addr().map(|address| address.ip().to_string()).unwrap_or_default();
        let Ok(mut writer) = stream.try_clone() else { return };
        let mut reader = BufReader::new(stream);
        let _ = reader.get_ref().set_read_timeout(Some(Duration::from_secs(30)));
        while let Some((method, target, headers)) = read_request(&mut reader) {
            let host = headers.get("host").map(|host| host.rsplit_once(':').filter(|(_, port)| port.bytes().all(|b| b.is_ascii_digit())).map_or(host.as_str(), |(name, _)| name).to_lowercase());
            let allowed = lock(&self.0.listening).as_ref().is_some_and(|open| host.as_ref().is_some_and(|host| open.hosts.contains(host)));
            if !allowed { let _ = respond(&mut writer, "421 Misdirected Request", "text/plain", b"not a name of this computer", &[]); continue; }
            if method != "GET" { let _ = respond(&mut writer, "405 Method Not Allowed", "text/plain", b"a shared view only reads", &[("Allow", "GET")]); continue; }
            let (path, query) = target.split_once('?').unwrap_or((target.as_str(), ""));
            let query: BTreeMap<&str, String> = query.split('&').filter_map(|pair| pair.split_once('=')).map(|(key, value)| (key, decode(value))).collect();
            let done = match path {
                "/zlc" => respond(&mut writer, "301 Moved Permanently", "text/plain", b"", &[("Location", "/zlc/")]).is_err(),
                "/zlc/" | "/zlc/index.html" => self.page(&mut writer).is_err(),
                "/zlc/events" => { self.events(writer, &peer, headers.get("user-agent").cloned().unwrap_or_default()); return; }
                _ if path.starts_with("/zlc/frame/") => self.picture(&mut writer, &decode(&path["/zlc/frame/".len()..]), &query).is_err(),
                _ if path.starts_with("/zlc/") => self.asset(&mut writer, &path["/zlc/".len()..]).is_err(),
                _ => respond(&mut writer, "404 Not Found", "text/plain", b"", &[]).is_err(),
            };
            if done { return; }
        }
    }

    /// The window's own page, marked shared and allowed to read this server.
    fn page(&self, writer: &mut TcpStream) -> std::io::Result<()> {
        let text = std::fs::read_to_string(self.0.ui_root.join("index.html")).unwrap_or_default()
            .replacen("<html lang=\"en\">", "<html lang=\"en\" data-share=\"1\">", 1)
            .replace("connect-src 'none'", "connect-src 'self'")
            .replace("img-src 'self' data:", "img-src 'self' data: blob:");
        respond(writer, "200 OK", "text/html; charset=utf-8", text.as_bytes(), &[])
    }

    fn asset(&self, writer: &mut TcpStream, relative: &str) -> std::io::Result<()> {
        let path = self.0.ui_root.join(relative).canonicalize();
        let root = self.0.ui_root.canonicalize();
        let file = match (path, root) { (Ok(path), Ok(root)) if path.starts_with(&root) && path.is_file() => path, _ => return respond(writer, "404 Not Found", "text/plain", b"", &[]) };
        let kind = match file.extension().and_then(|x| x.to_str()).unwrap_or("") {
            "js" => "text/javascript; charset=utf-8", "css" => "text/css; charset=utf-8", "ttf" => "font/ttf", "woff2" => "font/woff2",
            "png" => "image/png", "svg" => "image/svg+xml", _ => "application/octet-stream",
        };
        match std::fs::read(&file) { Ok(bytes) => respond(writer, "200 OK", kind, &bytes, &[]), Err(_) => respond(writer, "404 Not Found", "text/plain", b"", &[]) }
    }

    /// One panel's picture, for an approved viewer, exactly at the sequence it was told of.
    fn picture(&self, writer: &mut TcpStream, panel: &str, query: &BTreeMap<&str, String>) -> std::io::Result<()> {
        let token = query.get("token").cloned().unwrap_or_default();
        if !lock(&self.0.clients).values().any(|client| client.approved && client.token == token) {
            return respond(writer, "403 Forbidden", "text/plain", b"not an approved viewer", &[]);
        }
        let wanted: Option<u64> = query.get("sequence").and_then(|text| text.parse().ok());
        let ready = {
            let mut fronts = lock(&self.0.fronts);
            match fronts.get_mut(panel) {
                Some((sequence, front, png)) if Some(*sequence) == wanted => match png {
                    Some(png) => Ok(png.clone()),
                    None => Err((*sequence, front.clone())),
                },
                _ => return respond(writer, "410 Gone", "text/plain", b"a newer picture replaced it", &[]),
            }
        };
        // Encoded once per picture, off the lock, whichever viewer asks first.
        let png = match ready {
            Ok(png) => png,
            Err((sequence, front)) => {
                let png = Arc::new(crate::remote::png(&front).map_err(std::io::Error::other)?);
                if let Some(entry) = lock(&self.0.fronts).get_mut(panel).filter(|entry| entry.0 == sequence) { entry.2 = Some(png.clone()); }
                png
            }
        };
        respond(writer, "200 OK", "image/png", &png, &[])
    }

    /// The event stream of one browser: pending until the operator decides.
    fn events(&self, mut writer: TcpStream, peer: &str, browser: String) {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n";
        if writer.write_all(head.as_bytes()).is_err() { return; }
        let id = format!("{:x}{}", self.0.next.fetch_add(1, Ordering::Relaxed), &secret()[..8]);
        let (sender, receiver) = mpsc::channel();
        let pending = json!({"type": "pending", "message": "Waiting for approval on the experiment computer."});
        let _ = sender.send(Some(pending.to_string()));
        let browser: String = browser.chars().take(256).collect();
        lock(&self.0.clients).insert(id.clone(), Client { address: peer.to_owned(), browser, approved: false, token: secret(), sender });
        (self.0.notify)();
        loop {
            let line = match receiver.recv_timeout(BEAT) {
                Ok(Some(text)) => format!("data: {text}\n\n"),
                Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => ": beat\n\n".to_owned(),
            };
            if writer.write_all(line.as_bytes()).and_then(|()| writer.flush()).is_err() { break; }
        }
        let _ = writer.shutdown(std::net::Shutdown::Both);
        lock(&self.0.clients).remove(&id);
        (self.0.notify)();
    }
}

fn close(client: &Client, message: &str) {
    let _ = client.sender.send(Some(json!({"type": "closed", "message": message}).to_string()));
    let _ = client.sender.send(None);
}

fn frame_notice(panel: &str, sequence: u64, front: &Front) -> Value {
    json!({"type": "frame", "panel": panel, "sequence": sequence, "width": front.width, "height": front.height})
}

fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                match u8::from_str_radix(&text[index + 1..index + 3], 16) { Ok(byte) => { out.push(byte); index += 3; } Err(_) => { out.push(b'%'); index += 1; } }
            }
            b'+' => { out.push(b' '); index += 1; }
            byte => { out.push(byte); index += 1; }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn read_request(reader: &mut BufReader<TcpStream>) -> Option<(String, String, BTreeMap<String, String>)> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 || line.len() > 8192 { return None; }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next()?.to_owned(), parts.next()?.to_owned());
    let mut headers = BTreeMap::new();
    let mut total = 0;
    loop {
        let mut header = String::new();
        let read = reader.read_line(&mut header).ok()?;
        total += read;
        if read == 0 || total > 16 << 10 { return None; }
        let header = header.trim_end();
        if header.is_empty() { break; }
        if let Some((name, value)) = header.split_once(':') { headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned()); }
    }
    // A GET has no body; whatever one announces is read past.
    let length: usize = headers.get("content-length").and_then(|value| value.parse().ok()).unwrap_or(0);
    if length > 0 { let mut sink = vec![0; length.min(1 << 20)]; reader.read_exact(&mut sink).ok()?; }
    Some((method, target, headers))
}

fn respond(writer: &mut TcpStream, status: &str, kind: &str, body: &[u8], extra: &[(&str, &str)]) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\n\
        X-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n\
        Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; \
        connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'\r\n", body.len());
    for (name, value) in extra { head.push_str(&format!("{name}: {value}\r\n")); }
    head.push_str("\r\n");
    writer.write_all(head.as_bytes())?;
    writer.write_all(body)?;
    writer.flush()
}
