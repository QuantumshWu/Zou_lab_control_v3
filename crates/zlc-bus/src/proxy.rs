//! The one MCP entry point an AI client registers (stdio, `zlc-mcp`): it lists
//! the experiments running on this computer and forwards every other call to
//! the one it is attached to (v2 `zou_lab_control mcp` with `lab_connections`
//! and `lab_connect`). Each running instrument leaves a connection record
//! (`connection-<pid>-<port>.json`: url, token, pid, workspace, control) in
//! its own checkout's [`records_directory`] (`.local\mcp`); this process reads
//! those of every checkout with a running zlc-desktop.exe, and its own
//! ([`record_directories`]). Tokens never leave this process.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

/// Where this checkout's running instruments record their endpoints (and keep the MCP token): `<checkout>\.local\mcp`.
pub fn records_directory() -> PathBuf { zlc_core::checkout::local("mcp") }

/// The record folders of every checkout an instrument runs from on this
/// computer -- found from the running zlc-desktop.exe processes -- and this one's.
pub fn record_directories() -> Vec<PathBuf> {
    let mut folders = vec![records_directory()];
    for checkout in running_desktops().iter().filter_map(|exe| zlc_core::checkout::of(exe)) {
        let folder = checkout.join(zlc_core::checkout::LOCAL).join("mcp");
        if !folders.iter().any(|known| same_path(&known.display().to_string(), &folder.display().to_string())) { folders.push(folder); }
    }
    folders
}

/// The executables of the zlc-desktop.exe processes running on this computer.
#[cfg(windows)]
fn running_desktops() -> Vec<PathBuf> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS};
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW};
    let mut found = Vec::new();
    // SAFETY: the snapshot and process handles are checked and closed here; the buffers outlive the calls that fill them.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snapshot == INVALID_HANDLE_VALUE { return found; }
        let mut entry: PROCESSENTRY32W = std::mem::zeroed();
        entry.dwSize = size_of::<PROCESSENTRY32W>() as u32;
        let mut more = Process32FirstW(snapshot, &mut entry) != 0;
        while more {
            let name = &entry.szExeFile[..entry.szExeFile.iter().position(|unit| *unit == 0).unwrap_or(entry.szExeFile.len())];
            if String::from_utf16_lossy(name).eq_ignore_ascii_case("zlc-desktop.exe") {
                let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, entry.th32ProcessID);
                if !process.is_null() {
                    let mut path = vec![0u16; 32768];
                    let mut length = path.len() as u32;
                    if QueryFullProcessImageNameW(process, 0, path.as_mut_ptr(), &mut length) != 0 {
                        found.push(PathBuf::from(String::from_utf16_lossy(&path[..length as usize])));
                    }
                    CloseHandle(process);
                }
            }
            more = Process32NextW(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    found
}

#[cfg(not(windows))]
fn running_desktops() -> Vec<PathBuf> { Vec::new() }

/// One instrument's record, as it wrote it (`path`: the file).
#[derive(Clone, Debug, PartialEq)]
pub struct Record { pub file: String, pub path: PathBuf, pub url: String, pub token: String, pub pid: u64, pub workspace: String, pub control: bool }

impl Record {
    /// What a client may see: everything but the token.
    fn public(&self) -> Value {
        json!({"connection": self.file, "url": self.url, "pid": self.pid, "workspace": self.workspace, "control_enabled": self.control})
    }
}

/// The records in `directory`, loopback endpoints only.
pub fn records(directory: &Path) -> Vec<Record> {
    let mut found: Vec<Record> = std::fs::read_dir(directory).into_iter().flatten().flatten().filter_map(|entry| {
        let file = entry.file_name().to_string_lossy().into_owned();
        if !(file.starts_with("connection-") && file.ends_with(".json")) { return None; }
        let value: Value = serde_json::from_str(&std::fs::read_to_string(entry.path()).ok()?).ok()?;
        let url = value["url"].as_str()?.to_owned();
        if !url.starts_with("http://127.0.0.1:") || !url.ends_with("/mcp") { return None; }
        Some(Record { file, path: entry.path(), url, token: value["token"].as_str()?.to_owned(), pid: value["pid"].as_u64()?,
                      workspace: value["workspace"].as_str().unwrap_or_default().to_owned(), control: value["control_enabled"] != false })
    }).collect();
    found.sort_by(|left, right| left.file.cmp(&right.file));
    found
}

/// One JSON-RPC POST to an instrument: its new session id (if it gave one) and the reply.
fn post(record: &Record, session: Option<&str>, body: &Value, timeout: Duration) -> Result<(Option<String>, Value), String> {
    let address = record.url.trim_start_matches("http://").trim_end_matches("/mcp");
    let mut stream = TcpStream::connect(address).map_err(|error| format!("cannot reach {address}: {error}"))?;
    stream.set_read_timeout(Some(timeout)).map_err(|error| error.to_string())?;
    let text = body.to_string();
    let mut request = format!("POST /mcp HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\n\
        Accept: application/json, text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n", record.token, text.len());
    if let Some(session) = session { request.push_str(&format!("Mcp-Session-Id: {session}\r\n")); }
    request.push_str("\r\n");
    request.push_str(&text);
    stream.write_all(request.as_bytes()).map_err(|error| error.to_string())?;
    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader.read_line(&mut status).map_err(|error| format!("no answer from {address}: {error}"))?;
    let (mut length, mut new_session) = (0usize, None);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).map_err(|error| error.to_string())? == 0 { break; }
        let line = line.trim_end();
        if line.is_empty() { break; }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") { length = value.trim().parse().unwrap_or(0); }
            if name.eq_ignore_ascii_case("mcp-session-id") { new_session = Some(value.trim().to_owned()); }
        }
    }
    let mut payload = vec![0; length];
    reader.read_exact(&mut payload).map_err(|error| error.to_string())?;
    if !status.contains(" 200 ") && !status.contains(" 202 ") { return Err(format!("{address} answered {}: {}", status.trim(), String::from_utf8_lossy(&payload))); }
    Ok((new_session, if payload.is_empty() { Value::Null } else { serde_json::from_slice(&payload).map_err(|error| error.to_string())? }))
}

/// Whether the record's instrument answers as itself (its pid and workspace): Ok(false) when nothing
/// listens there any more or another process does; Err when it did not answer in time.
fn live(record: &Record) -> Result<bool, String> {
    let address = record.url.trim_start_matches("http://").trim_end_matches("/mcp");
    if TcpStream::connect(address).is_err() { return Ok(false); }
    let (_, reply) = post(record, None, &json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "health", "arguments": {}}}),
                          Duration::from_secs(3))?;
    let health = &reply["result"]["structuredContent"];
    Ok(health["pid"].as_u64() == Some(record.pid) && health["workspace"].as_str().is_some_and(|workspace| same_path(workspace, &record.workspace)))
}

fn same_path(left: &str, right: &str) -> bool {
    let normal = |path: &str| path.replace('\\', "/").trim_end_matches('/').to_lowercase();
    normal(left) == normal(right)
}

struct Attached { record: Record, session: String }

/// The proxy's state: the client's name, the attached instrument and the calls forwarded to it.
pub struct Proxy {
    /// The record folders to read, asked again at every listing (instruments start and end).
    directories: Box<dyn Fn() -> Vec<PathBuf> + Send + Sync>,
    client: Mutex<String>,
    attached: Mutex<Option<Attached>>,
    /// Counts attachments: the event relay of an earlier one stops.
    generation: Arc<AtomicU64>,
    forwarding: AtomicUsize,
    /// Writes one message to the client (stdout).
    notify: Arc<dyn Fn(&Value) + Send + Sync>,
}

const RELEASED: &str = "No operation was replayed; read the experiment's state before retrying a write.";

impl Proxy {
    pub fn new(directories: impl Fn() -> Vec<PathBuf> + Send + Sync + 'static, notify: Arc<dyn Fn(&Value) + Send + Sync>) -> Arc<Self> {
        Arc::new(Self { directories: Box::new(directories), client: Mutex::new("mcp-client".into()), attached: Mutex::new(None), generation: Arc::new(AtomicU64::new(0)),
                        forwarding: AtomicUsize::new(0), notify })
    }

    fn tools(&self) -> Vec<Value> {
        vec![
            json!({"name": "list_connections", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false},
                   "annotations": {"readOnlyHint": true},
                   "description": "The experiments running on this computer: connection, workspace, pid, control, whether each answers (live), and which one this connection is attached to. Never a token; starts nothing."}),
            json!({"name": "connect", "inputSchema": {"type": "object", "properties": {"connection": {"type": "string",
                       "description": "A connection from list_connections; omitted: the only live experiment."}}, "additionalProperties": false},
                   "description": "Attach this AI connection to one running experiment. Nothing is started or replayed; the tool list becomes that experiment's."}),
        ]
    }

    /// Every record with whether it answers. A record whose instrument has ended (nothing, or another
    /// process, listens at its address) is removed; one that is slow to answer stays, not live.
    fn connections(&self) -> Vec<(Record, Value)> {
        let attached = self.attached.lock().unwrap().as_ref().map(|attached| attached.record.file.clone());
        let mut found: Vec<Record> = (self.directories)().iter().flat_map(|folder| records(folder)).collect();
        found.sort_by(|left, right| left.file.cmp(&right.file));
        found.into_iter().filter_map(|record| {
            let mut row = record.public();
            match live(&record) {
                Ok(true) => row["live"] = json!(true),
                Ok(false) => { let _ = std::fs::remove_file(&record.path); return None; }
                Err(error) => { row["live"] = json!(false); row["error"] = json!(error); }
            }
            row["attached"] = json!(attached.as_deref() == Some(record.file.as_str()));
            Some((record, row))
        }).collect()
    }

    fn connect(&self, wanted: Option<&str>) -> Result<Value, String> {
        if self.forwarding.load(Ordering::Acquire) > 0 {
            return Err("Calls to the attached experiment are still running; wait for them (or stop_all there) before changing experiment".into());
        }
        let candidates: Vec<(Record, Value)> = self.connections().into_iter()
            .filter(|(record, row)| row["live"] == true && wanted.is_none_or(|file| record.file == file)).collect();
        let [(record, row)] = <[_; 1]>::try_from(candidates).map_err(|found: Vec<_>| match (wanted, found.len()) {
            (Some(file), _) => format!("No running experiment {file:?}; list_connections names them"),
            (None, 0) => "No experiment is running on this computer; start one, then call connect".into(),
            (None, _) => "Several experiments are running; choose one: connect {connection} with a name from list_connections".into(),
        })?;
        if self.attached.lock().unwrap().as_ref().is_some_and(|attached| attached.record == record) { return Ok(row); }
        let client = self.client.lock().unwrap().clone();
        let (session, reply) = post(&record, None, &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": client}, "capabilities": {}}}), Duration::from_secs(10))?;
        if reply.get("error").is_some() { return Err(format!("The experiment refused the connection: {}", reply["error"])); }
        let session = session.ok_or("The experiment gave no session")?;
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        *self.attached.lock().unwrap() = Some(Attached { record: record.clone(), session });
        self.relay_tool_changes(record, generation);
        (self.notify)(&json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}));
        Ok(row)
    }

    /// The attached instrument's tool list changes (devices opened, closed) reach the client while it stays attached.
    fn relay_tool_changes(&self, record: Record, generation: u64) {
        let (notify, current) = (self.notify.clone(), self.generation.clone());
        let _ = std::thread::Builder::new().name("zlc-mcp-events".into()).spawn(move || {
            let address = record.url.trim_start_matches("http://").trim_end_matches("/mcp").to_owned();
            let Ok(mut stream) = TcpStream::connect(&address) else { return };
            let request = format!("GET /mcp HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {}\r\nAccept: text/event-stream\r\n\r\n", record.token);
            if stream.write_all(request.as_bytes()).is_err() { return; }
            for line in BufReader::new(stream).lines() {
                let Ok(line) = line else { return };
                if current.load(Ordering::Acquire) != generation { return; }
                let Some(message) = line.strip_prefix("data: ").and_then(|data| serde_json::from_str::<Value>(data).ok()) else { continue };
                if message["method"] == "notifications/tools/list_changed" { notify(&message); }
            }
        });
    }

    fn forward(&self, method: &str, params: &Value) -> Result<Value, String> {
        let (record, session) = {
            let attached = self.attached.lock().unwrap();
            match attached.as_ref() {
                Some(attached) => (attached.record.clone(), attached.session.clone()),
                None => { drop(attached); self.connect(None)?; return self.forward(method, params); }
            }
        };
        // A wait answers when it is done: give it its own time and some.
        let seconds = params["arguments"]["timeout_seconds"].as_f64().filter(|_| matches!(params["name"].as_str(), Some("wait" | "wait_for" | "pulse_wait_done")))
            .map_or(900.0, |seconds| seconds.clamp(0.0, 1800.0) + 30.0);
        self.forwarding.fetch_add(1, Ordering::AcqRel);
        let result = post(&record, Some(&session), &json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}), Duration::from_secs_f64(seconds));
        self.forwarding.fetch_sub(1, Ordering::AcqRel);
        let (_, reply) = result.map_err(|error| format!("The experiment connection is closed ({error}); connect again, then read its state. {RELEASED}"))?;
        match reply.get("error") { Some(error) => Err(error["message"].as_str().unwrap_or("error").to_owned()), None => Ok(reply["result"].clone()) }
    }

    /// One client message to its reply (None for notifications).
    pub fn handle(&self, message: &Value) -> Option<Value> {
        let id = message.get("id").cloned();
        let method = message["method"].as_str().unwrap_or("");
        let params = message.get("params").cloned().unwrap_or(json!({}));
        let result: Result<Value, (i64, String)> = match method {
            "initialize" => {
                if let Some(name) = params["clientInfo"]["name"].as_str() { *self.client.lock().unwrap() = name.to_owned(); }
                Ok(json!({"protocolVersion": params["protocolVersion"].as_str().unwrap_or("2025-06-18"),
                          "capabilities": {"tools": {"listChanged": true}},
                          "serverInfo": {"name": "zlc-mcp", "version": env!("CARGO_PKG_VERSION")},
                          "instructions": format!("{} This connection reaches the experiments running on this computer: \
                              list_connections names them, connect attaches to one (with one running, the first call attaches by itself). \
                              If none is running yet, keep this connection and connect after starting one.", crate::catalog::instructions())}))
            }
            "ping" => Ok(json!({})),
            "tools/list" => {
                let mut tools = self.tools();
                if let Ok(remote) = self.forward("tools/list", &json!({})) { tools.extend(remote["tools"].as_array().cloned().unwrap_or_default()); }
                Ok(json!({"tools": tools}))
            }
            "tools/call" => Ok(self.call(&params)),
            "prompts/list" => Ok(self.forward("prompts/list", &json!({})).unwrap_or_else(|_| json!({"prompts": []}))),
            "prompts/get" => self.forward("prompts/get", &params).map_err(|error| (-32602, error)),
            _ if method.starts_with("notifications/") => return None,
            _ => Err((-32601, format!("method not found: {method}"))),
        };
        let id = id?;
        Some(match result {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err((code, text)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": text}}),
        })
    }

    fn call(&self, params: &Value) -> Value {
        let answer = |value: Value| json!({"content": [{"type": "text", "text": serde_json::to_string_pretty(&value).unwrap_or_default()}],
                                           "structuredContent": value, "isError": false});
        let failed = |text: String| json!({"content": [{"type": "text", "text": text}], "isError": true});
        match params["name"].as_str().unwrap_or("") {
            "list_connections" => answer(json!({"connections": self.connections().into_iter().map(|(_, row)| row).collect::<Vec<_>>()})),
            "connect" => match self.connect(params["arguments"]["connection"].as_str()) { Ok(row) => answer(row), Err(error) => failed(error) },
            _ => match self.forward("tools/call", params) { Ok(result) => result, Err(error) => failed(error) },
        }
    }
}

/// Serve the client on stdin/stdout until stdin ends; each call runs on its own thread, so a long wait never holds up Stop.
pub fn serve_stdio() -> std::io::Result<()> {
    let out = Arc::new(Mutex::new(std::io::stdout()));
    let writer = out.clone();
    let proxy = Proxy::new(record_directories, Arc::new(move |message: &Value| {
        let mut out = writer.lock().unwrap();
        let _ = writeln!(out, "{message}");
        let _ = out.flush();
    }));
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let message: Value = match serde_json::from_str(&line) {
            Ok(message) => message,
            Err(error) => { (proxy.notify)(&json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": error.to_string()}})); continue; }
        };
        let proxy = proxy.clone();
        std::thread::spawn(move || if let Some(reply) = proxy.handle(&message) { (proxy.notify)(&reply); });
    }
    Ok(())
}

/// The client registration for this binary (v2 `--client-config`).
pub fn client_config(client: &str) -> Result<String, String> {
    let command = std::env::current_exe().map_err(|error| error.to_string())?.display().to_string();
    match client {
        "claude" => Ok(serde_json::to_string_pretty(&json!({"mcpServers": {"zlc": {"type": "stdio", "command": command, "args": []}}})).unwrap()),
        "codex" => Ok(format!("[mcp_servers.zlc]\ncommand = {}\nargs = []\nrequired = true\nstartup_timeout_sec = 30\ntool_timeout_sec = 1860\n",
                              serde_json::to_string(&command).unwrap())),
        _ => Err("client is claude or codex".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::{self, Host, ToolOutput};
    use zlc_core::control::Origin;

    /// One experiment: it names its workspace in every reply.
    struct Bench(&'static str);
    impl Host for Bench {
        fn call(&self, origin: Origin, name: &str, _: Value) -> Result<ToolOutput, String> {
            Ok(ToolOutput { json: json!({"bench": self.0, "tool": name, "origin": origin.to_string()}), png: None })
        }
        fn identity(&self) -> Value { json!({"workspace": self.0}) }
    }

    fn record(directory: &Path, server: &mcp::Server, workspace: &str) -> String {
        let file = format!("connection-{}-{}.json", std::process::id(), server.address.port());
        std::fs::write(directory.join(&file), json!({"url": format!("http://{}/mcp", server.address), "token": server.token,
            "pid": std::process::id(), "workspace": workspace, "control_enabled": true}).to_string()).unwrap();
        file
    }

    fn call(proxy: &Proxy, name: &str, arguments: Value) -> Value {
        proxy.handle(&json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": name, "arguments": arguments}})).unwrap()["result"].clone()
    }

    #[test]
    fn one_entry_point_lists_and_attaches_to_each_experiment() {
        let directory = std::env::temp_dir().join(format!("zlc-mcp-proxy-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let (a, b) = (mcp::serve(Arc::new(Bench("C:/lab/a")), true, 0, "a".repeat(40)).unwrap(),
                      mcp::serve(Arc::new(Bench("C:/lab/b")), true, 0, "b".repeat(40)).unwrap());
        let (file_a, file_b) = (record(&directory, &a, "C:/lab/a"), record(&directory, &b, "C:/lab/b"));
        // An instrument that has ended: nothing listens at its address any more.
        let gone = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        std::fs::write(directory.join("connection-1-1.json"), json!({"url": format!("http://{gone}/mcp"), "token": "x", "pid": 1, "workspace": "C:/old"}).to_string()).unwrap();
        let notes = Arc::new(Mutex::new(Vec::new()));
        let seen = notes.clone();
        let folder = directory.clone();
        let proxy = Proxy::new(move || vec![folder.clone()], Arc::new(move |message: &Value| seen.lock().unwrap().push(message.clone())));
        proxy.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "assistant"}}}));
        let listed = call(&proxy, "list_connections", json!({}))["structuredContent"]["connections"].clone();
        assert_eq!(listed.as_array().unwrap().len(), 2, "{listed}");
        assert!(listed.as_array().unwrap().iter().all(|row| row["live"] == true && row.get("token").is_none()));
        assert!(!directory.join("connection-1-1.json").exists(), "the ended instrument's record is removed");
        // Two are running: a call needs a choice first.
        assert!(call(&proxy, "get_state", json!({}))["content"][0]["text"].as_str().unwrap().contains("Several experiments"));
        let attached = call(&proxy, "connect", json!({"connection": file_b}));
        assert_eq!(attached["structuredContent"]["workspace"], "C:/lab/b");
        assert!(notes.lock().unwrap().iter().any(|note| note["method"] == "notifications/tools/list_changed"));
        let state = call(&proxy, "get_state", json!({}))["structuredContent"].clone();
        assert_eq!((state["bench"].clone(), state["origin"].clone()), (json!("C:/lab/b"), json!("remote client 'assistant'")));
        let tools = proxy.handle(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})).unwrap()["result"]["tools"].clone();
        let names: Vec<&str> = tools.as_array().unwrap().iter().filter_map(|tool| tool["name"].as_str()).collect();
        assert!(names.starts_with(&["list_connections", "connect"]) && names.contains(&"get_state"));
        call(&proxy, "connect", json!({"connection": file_a}));
        assert_eq!(call(&proxy, "get_state", json!({}))["structuredContent"]["bench"], "C:/lab/a");
        drop(b);
        let after = call(&proxy, "list_connections", json!({}))["structuredContent"]["connections"].clone();
        assert_eq!(after.as_array().unwrap().iter().map(|row| row["workspace"].clone()).collect::<Vec<_>>(), vec![json!("C:/lab/a")]);
        assert!(client_config("claude").unwrap().contains("mcpServers") && client_config("codex").unwrap().contains("[mcp_servers.zlc]"));
        for file in [&file_a, &file_b] { let _ = std::fs::remove_file(directory.join(file)); }
        let _ = std::fs::remove_dir(&directory);
    }
}
