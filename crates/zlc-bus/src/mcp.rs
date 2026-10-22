//! The Model Context Protocol endpoint: the same catalogue the window uses,
//! served to AI clients over Streamable HTTP on the loopback interface only.
//!
//! Every tool call is validated against the command's schema and then run by
//! the instrument exactly as the window's command would be. Requests need the
//! bearer token the instrument wrote at start; browsers are refused by Origin.
//! With control off (v2 `allow_control=False`, `--mcp-read-only`) a client is
//! offered, and may call, only the tools that read.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::control::Origin;

use crate::catalog::{self, COMMANDS};

/// What a tool call returns: structured state as text, optionally a picture (PNG, or JPEG by its bytes).
#[derive(Debug, Default)]
pub struct ToolOutput { pub json: Value, pub png: Option<Vec<u8>> }

/// The instrument side of the endpoint.
pub trait Host: Send + Sync + 'static {
    /// Run one validated command for `origin`.
    fn call(&self, origin: Origin, name: &str, params: Value) -> Result<ToolOutput, String>;
    /// Commands open plugin instances add (`<instance>__<command>`, PLUGIN_PROTOCOL §8), as
    /// MCP tool rows (`annotations.readOnlyHint` says which only read); the instrument validates these itself.
    fn tools(&self) -> Vec<Value> { Vec::new() }
    /// Facts about the instrument that never wait for it (`health`, connection records): workspace, closing.
    fn identity(&self) -> Value { json!({}) }
}

/// One client's calls: validated, checked against control, and either composed here or run by the instrument.
#[derive(Clone)]
pub struct Caller { pub host: Arc<dyn Host>, pub control: bool, client: String, sessions: Arc<Sessions> }

impl Caller {
    /// A command as the client sent it.
    pub fn dispatch(&self, name: &str, arguments: Value) -> Result<ToolOutput, String> {
        match catalog::find(name) {
            Some(spec) => {
                spec.validate(&arguments).map_err(|error| format!("invalid arguments for {name}: {error}"))?;
                if !self.control && !spec.read_only { return Err(format!("{name} changes the instrument. {READ_ONLY}")); }
                if spec.endpoint { return crate::compose::run(self, name, arguments); }
            }
            None => match self.host.tools().into_iter().find(|tool| tool["name"] == name) {
                Some(tool) if !self.control && !reads(&tool) => return Err(format!("{name} changes the instrument. {READ_ONLY}")),
                Some(_) => {}
                None => return Err(format!("unknown tool {name:?}")),
            },
        }
        self.instrument(name, arguments)
    }

    /// Run a command on the instrument for this client.
    pub fn instrument(&self, name: &str, arguments: Value) -> Result<ToolOutput, String> {
        self.sessions.in_flight.fetch_add(1, Ordering::AcqRel);
        let result = self.host.call(Origin::Remote(self.client.clone()), name, arguments);
        self.sessions.in_flight.fetch_sub(1, Ordering::AcqRel);
        *self.sessions.answered.lock().unwrap() = Some(Instant::now());
        result
    }

    pub fn calls_in_flight(&self) -> u64 { self.sessions.in_flight.load(Ordering::Acquire) }
    pub fn seconds_since_answer(&self) -> Option<f64> { self.sessions.answered.lock().unwrap().map(|at| at.elapsed().as_secs_f64()) }
}

/// A recipe offered as an MCP prompt.
pub struct Recipe { pub name: &'static str, pub description: &'static str, pub text: &'static str }

pub static RECIPES: &[Recipe] = &[
    Recipe {
        name: "camera_roi_profile_fit",
        description: "Image a cloud, cut a region of interest, plot its profile and fit a Gaussian.",
        text: "1. list_signals and get_state. 2. start_camera {exposure_seconds: 0.1}; wait {produced: 2}. \
3. In the image panel, configure {panel, selectors: [{kind: \"area\", value: {x: [x0, x1], y: [y0, y1]}, facet_index: null}]} \
with bounds in pixels around the cloud (get_panel_image shows it). 4. list_signals again: the ROI publishes \"@logic/<panel>/roi_frame\". \
5. add_panel {signal: \"@logic/<panel>/roi_frame\", kind: \"curve\"}; configure it so one spatial axis is x and the rest reduce \
(see its semantic controls). 6. configure {panel, fit: {model: \"gaussian_offset\"}}. 7. get_state {panel}: fit.computed holds \
the parameters with uncertainties; get_panel_image to confirm the fit drawn on the data.",
    },
    Recipe {
        name: "shot_history_facets",
        description: "Watch the last N shots side by side and fit each.",
        text: "1. Have an ROI signal (see camera_roi_profile_fit). 2. add_panel {signal, kind: \"facet_grid\"}; configure \
{panel, binding: {window: 40}} so it reads 40 shots, with the shot axis as facet. 3. configure {panel, fit: {model, fit_all_facets: true}}. \
4. focus {panel, index} opens one cell; focus {panel, index: null} returns. get_state lists each cell's fit.",
    },
];

#[derive(Default)]
struct Sessions { clients: Mutex<BTreeMap<String, String>>, next: AtomicU64, streams: Mutex<Vec<TcpStream>>,
                  in_flight: AtomicU64, answered: Mutex<Option<Instant>> }

/// Tells clients holding a notification stream (`GET /mcp`) that the tool list changed.
#[derive(Clone)]
pub struct Notifier(Arc<Sessions>);

impl Notifier {
    pub fn tools_changed(&self) {
        let event = b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n";
        self.0.streams.lock().unwrap().retain_mut(|stream| stream.write_all(event).and_then(|()| stream.flush()).is_ok());
    }
}

/// A running endpoint; dropping it stops accepting connections.
pub struct Server { pub address: SocketAddr, pub token: String, pub control: bool, stop: Arc<AtomicBool>, worker: Option<std::thread::JoinHandle<()>>, sessions: Arc<Sessions> }

impl Server {
    pub fn notifier(&self) -> Notifier { Notifier(self.sessions.clone()) }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect_timeout(&self.address, Duration::from_millis(200));
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
    }
}

/// Serve on 127.0.0.1:`port` (0 picks a free port); `control`: whether clients may change anything.
pub fn serve(host: Arc<dyn Host>, control: bool, port: u16, token: String) -> std::io::Result<Server> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let address = listener.local_addr()?;
    let stop = Arc::new(AtomicBool::new(false));
    let sessions = Arc::new(Sessions::default());
    let (flag, secret, shared) = (stop.clone(), token.clone(), sessions.clone());
    let worker = std::thread::Builder::new().name("zlc-mcp".into()).spawn(move || {
        for stream in listener.incoming() {
            if flag.load(Ordering::Acquire) { break; }
            let Ok(stream) = stream else { continue };
            let (host, sessions, secret) = (host.clone(), shared.clone(), secret.clone());
            let _ = std::thread::Builder::new().name("zlc-mcp-connection".into()).spawn(move || {
                let _ = connection(stream, &host, control, &sessions, &secret);
            });
        }
    })?;
    Ok(Server { address, token, control, stop, worker: Some(worker), sessions })
}

struct Request { method: String, path: String, headers: BTreeMap<String, String>, body: Vec<u8> }

fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 { return Ok(None); }
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("").to_owned(), parts.next().unwrap_or("").to_owned());
    let mut headers = BTreeMap::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 { return Ok(None); }
        let header = header.trim_end();
        if header.is_empty() { break; }
        if let Some((name, value)) = header.split_once(':') { headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned()); }
    }
    let length: usize = headers.get("content-length").and_then(|value| value.parse().ok()).unwrap_or(0);
    if length > 16 << 20 { return Err(std::io::Error::other("request too large")); }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, headers, body }))
}

fn respond(stream: &mut TcpStream, status: &str, headers: &[(&str, String)], body: &[u8]) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\n", body.len());
    for (name, value) in headers { head.push_str(&format!("{name}: {value}\r\n")); }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn loopback_origin(origin: &str) -> bool {
    ["http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost"]
        .iter().any(|allowed| origin == *allowed || origin.strip_prefix(allowed).is_some_and(|rest| rest.starts_with(':')))
}

fn connection(stream: TcpStream, host: &Arc<dyn Host>, control: bool, sessions: &Arc<Sessions>, token: &str) -> std::io::Result<()> {
    let mut writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    while let Some(request) = read_request(&mut reader)? {
        if request.path.split('?').next() != Some("/mcp") {
            respond(&mut writer, "404 Not Found", &[], b"")?;
            continue;
        }
        if request.headers.get("origin").is_some_and(|origin| !loopback_origin(origin)) {
            respond(&mut writer, "403 Forbidden", &[], b"cross-origin requests are refused")?;
            continue;
        }
        if request.headers.get("authorization").map(String::as_str) != Some(&format!("Bearer {token}")) {
            respond(&mut writer, "401 Unauthorized", &[("WWW-Authenticate", "Bearer".into())], b"missing or wrong bearer token")?;
            continue;
        }
        match request.method.as_str() {
            "POST" => {}
            "DELETE" => {
                if let Some(session) = request.headers.get("mcp-session-id") { sessions.clients.lock().unwrap().remove(session); }
                respond(&mut writer, "200 OK", &[], b"")?;
                continue;
            }
            // The server's notification stream: kept open, written when the tool list changes.
            "GET" if request.headers.get("accept").is_some_and(|accept| accept.contains("text/event-stream")) => {
                writer.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n")?;
                writer.flush()?;
                sessions.streams.lock().unwrap().push(writer);
                return Ok(());
            }
            _ => { respond(&mut writer, "405 Method Not Allowed", &[("Allow", "GET, POST, DELETE".into())], b"")?; continue; }
        }
        let message: Value = match serde_json::from_slice(&request.body) {
            Ok(message) => message,
            Err(error) => {
                let body = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": error.to_string()}});
                respond(&mut writer, "400 Bad Request", &[("Content-Type", "application/json".into())], body.to_string().as_bytes())?;
                continue;
            }
        };
        let session = request.headers.get("mcp-session-id").cloned();
        let client = session.as_ref().and_then(|id| sessions.clients.lock().unwrap().get(id).cloned());
        let (reply, new_session) = handle(&message, client, host, control, sessions);
        let mut headers = vec![("Content-Type", "application/json".to_owned())];
        if let Some(id) = new_session { headers.push(("Mcp-Session-Id", id)); }
        match reply {
            Some(body) => respond(&mut writer, "200 OK", &headers, body.to_string().as_bytes())?,
            None => respond(&mut writer, "202 Accepted", &headers, b"")?,
        }
    }
    Ok(())
}

/// One JSON-RPC message (or batch) to its reply; notifications have none.
fn handle(message: &Value, client: Option<String>, host: &Arc<dyn Host>, control: bool, sessions: &Arc<Sessions>)
              -> (Option<Value>, Option<String>) {
    if let Some(batch) = message.as_array() {
        let mut new_session = None;
        let replies: Vec<Value> = batch.iter().filter_map(|item| {
            let (reply, session) = handle(item, client.clone(), host, control, sessions);
            new_session = new_session.take().or(session);
            reply
        }).collect();
        return ((!replies.is_empty()).then(|| Value::Array(replies)), new_session);
    }
    let id = message.get("id").cloned();
    let method = message["method"].as_str().unwrap_or("");
    let params = message.get("params").cloned().unwrap_or(json!({}));
    let mut new_session = None;
    let result: Result<Value, (i64, String)> = match method {
        "initialize" => {
            let name = params["clientInfo"]["name"].as_str().unwrap_or("mcp-client").to_owned();
            let session = format!("zlc-{}", sessions.next.fetch_add(1, Ordering::Relaxed) + 1);
            sessions.clients.lock().unwrap().insert(session.clone(), name);
            new_session = Some(session);
            let version = params["protocolVersion"].as_str().unwrap_or("2025-06-18");
            Ok(json!({"protocolVersion": version,
                      "capabilities": {"tools": {"listChanged": true}, "prompts": {"listChanged": false}},
                      "serverInfo": {"name": "zlc-v3", "version": env!("CARGO_PKG_VERSION")},
                      "instructions": if control { catalog::instructions() } else { format!("{READ_ONLY} {}", catalog::instructions()) }}))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": COMMANDS.iter().filter(|spec| control || spec.read_only).map(|spec| json!({
            "name": spec.name, "description": spec.summary, "inputSchema": spec.input_schema(),
            "annotations": {"readOnlyHint": spec.read_only}})).chain(host.tools().into_iter().filter(|tool| control || reads(tool))).collect::<Vec<_>>()})),
        "tools/call" => Ok(call_tool(&params, &Caller { host: host.clone(), control, client: client.unwrap_or_else(|| "mcp-client".into()),
                                                         sessions: sessions.clone() })),
        "prompts/list" => Ok(json!({"prompts": RECIPES.iter().map(|r| json!({"name": r.name, "description": r.description})).collect::<Vec<_>>()})),
        "prompts/get" => match RECIPES.iter().find(|r| Some(r.name) == params["name"].as_str()) {
            Some(recipe) => Ok(json!({"description": recipe.description,
                "messages": [{"role": "user", "content": {"type": "text", "text": recipe.text}}]})),
            None => Err((-32602, "unknown prompt".into())),
        },
        _ if method.starts_with("notifications/") => return (None, None),
        _ => Err((-32601, format!("method not found: {method}"))),
    };
    let id = match id { Some(id) => id, None => return (None, new_session) };
    let reply = match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, message)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
    };
    (Some(reply), new_session)
}

/// What a read-only endpoint tells its clients.
const READ_ONLY: &str = "This endpoint is read-only: the instrument was started with --mcp-read-only, so only the tools that read are offered.";

fn reads(tool: &Value) -> bool { tool["annotations"]["readOnlyHint"] == true }

fn call_tool(params: &Value, caller: &Caller) -> Value {
    let name = params["name"].as_str().unwrap_or("");
    let arguments = params.get("arguments").cloned().filter(|value| !value.is_null()).unwrap_or(json!({}));
    match caller.dispatch(name, arguments) {
        Ok(output) => {
            let mut content = vec![json!({"type": "text", "text": serde_json::to_string_pretty(&output.json).unwrap_or_default()})];
            if let Some(image) = output.png {
                let mime = if image.starts_with(&[0xFF, 0xD8, 0xFF]) { "image/jpeg" } else { "image/png" };
                content.push(json!({"type": "image", "mimeType": mime, "data": base64(&image)}));
            }
            json!({"content": content, "structuredContent": output.json, "isError": false})
        }
        Err(error) => json!({"content": [{"type": "text", "text": error}], "isError": true}),
    }
}

/// Standard base64 with padding.
pub fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for (index, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            text.push(if index <= chunk.len() { TABLE[(n >> shift & 63) as usize] as char } else { '=' });
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    impl Host for Echo {
        fn call(&self, origin: Origin, name: &str, params: Value) -> Result<ToolOutput, String> {
            Ok(ToolOutput { json: json!({"origin": origin.to_string(), "name": name, "params": params}),
                            png: (name == "get_panel_image").then(|| vec![137, 80, 78, 71]) })
        }
    }

    fn post(address: SocketAddr, token: &str, session: Option<&str>, body: &Value) -> (String, Value) {
        let mut stream = TcpStream::connect(address).unwrap();
        let body = body.to_string();
        let mut request = format!("POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {}\r\n", body.len());
        if let Some(session) = session { request.push_str(&format!("Mcp-Session-Id: {session}\r\n")); }
        request.push_str("\r\n");
        request.push_str(&body);
        stream.write_all(request.as_bytes()).unwrap();
        let mut reader = BufReader::new(stream);
        let mut status = String::new();
        reader.read_line(&mut status).unwrap();
        let mut length = 0; let mut session_id = String::new();
        loop {
            let mut line = String::new(); reader.read_line(&mut line).unwrap();
            let line = line.trim_end();
            if line.is_empty() { break; }
            if let Some((n, v)) = line.split_once(':') {
                if n.eq_ignore_ascii_case("content-length") { length = v.trim().parse().unwrap(); }
                if n.eq_ignore_ascii_case("mcp-session-id") { session_id = v.trim().to_owned(); }
            }
        }
        let mut payload = vec![0; length];
        reader.read_exact(&mut payload).unwrap();
        (format!("{} {session_id}", status.trim()), serde_json::from_slice(&payload).unwrap_or(Value::Null))
    }

    #[test]
    fn a_client_initializes_lists_and_calls() {
        let server = serve(Arc::new(Echo), true, 0, "secret".into()).unwrap();
        let (status, reply) = post(server.address, "secret", None, &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "assistant"}, "capabilities": {}}}));
        let session = status.split_whitespace().last().unwrap().to_owned();
        assert!(status.starts_with("HTTP/1.1 200") && session.starts_with("zlc-"), "{status}");
        assert!(reply["result"]["instructions"].as_str().unwrap().contains("get_state"));
        let (status, _) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        assert!(status.starts_with("HTTP/1.1 202"));
        let (_, tools) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}));
        assert_eq!(tools["result"]["tools"].as_array().unwrap().len(), COMMANDS.len());
        let (_, called) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "add_panel", "arguments": {"signal": "camera/frames", "kind": "image"}}}));
        assert_eq!(called["result"]["isError"], false);
        assert_eq!(called["result"]["structuredContent"]["origin"], "remote client 'assistant'");
        let (_, image) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "get_panel_image", "arguments": {"panel": "panel-1"}}}));
        assert_eq!(image["result"]["content"][1]["data"], base64(&[137, 80, 78, 71]));
        let (_, invalid) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "add_panel", "arguments": {"signal": "camera/frames"}}}));
        assert_eq!(invalid["result"]["isError"], true);
        let (_, stop) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "stop_all", "arguments": {}}}));
        assert_eq!(stop["result"]["isError"], false);
        let (status, _) = post(server.address, "wrong", Some(&session), &json!({"jsonrpc": "2.0", "id": 8, "method": "ping"}));
        assert!(status.starts_with("HTTP/1.1 401"));
    }

    #[test]
    fn with_control_off_a_client_only_reads() {
        let server = serve(Arc::new(Echo), false, 0, "secret".into()).unwrap();
        let (status, reply) = post(server.address, "secret", None, &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "assistant"}, "capabilities": {}}}));
        let session = status.split_whitespace().last().unwrap().to_owned();
        assert!(reply["result"]["instructions"].as_str().unwrap().contains("read-only"));
        let (_, tools) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}));
        let listed = tools["result"]["tools"].as_array().unwrap();
        assert_eq!(listed.len(), COMMANDS.iter().filter(|spec| spec.read_only).count());
        assert!(listed.iter().all(|tool| tool["annotations"]["readOnlyHint"] == true));
        let (_, state) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "get_state", "arguments": {}}}));
        assert_eq!(state["result"]["isError"], false);
        let (_, refused) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "start_camera", "arguments": {"exposure_seconds": 0.1}}}));
        assert_eq!(refused["result"]["isError"], true);
        assert!(refused["result"]["content"][0]["text"].as_str().unwrap().contains("--mcp-read-only"));
    }

    /// A node that runs after start_node and is running from the third state read on.
    struct Lab { reads: AtomicU64, started: AtomicBool }
    impl Host for Lab {
        fn call(&self, _: Origin, name: &str, params: Value) -> Result<ToolOutput, String> {
            match name {
                "start_node" if params["node"] == "missing" => Err("No logic node \"missing\"".into()),
                "start_node" => { self.started.store(true, Ordering::Release); Ok(ToolOutput { json: json!({"run": {"run": "camera#1"}}), png: None }) }
                "get_state" => {
                    let reads = self.reads.fetch_add(1, Ordering::AcqRel) + 1;
                    let phase = if self.started.load(Ordering::Acquire) && reads >= 3 { "running" } else { "idle" };
                    Ok(ToolOutput { json: json!({"logic": [{"node_id": "camera", "phase": phase}]}), png: None })
                }
                "list_plugins" => Ok(ToolOutput { json: json!({"types": [{"type_id": params["type_id"]}]}), png: None }),
                _ => Ok(ToolOutput { json: json!({"name": name, "params": params}), png: None }),
            }
        }
        fn identity(&self) -> Value { json!({"workspace": "C:/lab"}) }
    }

    fn call(address: SocketAddr, session: &str, id: u64, name: &str, arguments: Value) -> Value {
        post(address, "secret", Some(session), &json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}})).1["result"].clone()
    }

    fn session(address: SocketAddr) -> String {
        let (status, _) = post(address, "secret", None, &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "assistant"}, "capabilities": {}}}));
        status.split_whitespace().last().unwrap().to_owned()
    }

    #[test]
    fn the_endpoint_composes_guide_health_wait_for_and_batch() {
        let server = serve(Arc::new(Lab { reads: AtomicU64::new(0), started: AtomicBool::new(false) }), true, 0, "secret".into()).unwrap();
        let session = session(server.address);
        let overview = call(server.address, &session, 2, "guide", json!({}));
        assert!(overview["structuredContent"]["content"].as_str().unwrap().contains("node_id"));
        assert!(overview["structuredContent"]["topics"].as_array().unwrap().iter().any(|topic| topic == "lifetime"));
        let typed = call(server.address, &session, 3, "guide", json!({"topic": "type:occupancy"}));
        assert_eq!(typed["structuredContent"]["descriptor"]["types"][0]["type_id"], "occupancy");
        assert_eq!(call(server.address, &session, 4, "guide", json!({"topic": "recipe"}))["isError"], true);
        let health = call(server.address, &session, 5, "health", json!({}))["structuredContent"].clone();
        assert_eq!((health["service"].clone(), health["workspace"].clone(), health["control_enabled"].clone()), (json!("running"), json!("C:/lab"), json!(true)));
        // The wait sees the state change; a write cannot be waited on.
        let steps = call(server.address, &session, 6, "batch", json!({"calls": [
            {"tool": "start_node", "arguments": {"node": "camera"}, "expect": {}},
            {"tool": "wait_for", "arguments": {"tool": "get_state", "arguments": {"sections": ["logic"]}, "path": ["logic", 0, "phase"],
                                               "equals": "running", "timeout_seconds": 5}},
            {"tool": "fit", "arguments": {"panel": {"$result": [0, "run", "run"]}}},
        ]}))["structuredContent"].clone();
        assert_eq!(steps["completed"], true, "{steps}");
        assert_eq!(steps["results"][1]["result"]["matched"], true);
        assert_eq!(steps["results"][2]["result"]["params"]["panel"], "camera#1");
        assert_eq!(call(server.address, &session, 7, "wait_for", json!({"tool": "start_node", "path": ["x"], "equals": 1}))["isError"], true);
        // The first refusal stops the batch; what ran stays.
        let stopped = call(server.address, &session, 8, "batch", json!({"calls": [
            {"tool": "fit", "arguments": {"panel": "panel-1"}},
            {"tool": "start_node", "arguments": {"node": "missing"}},
            {"tool": "fit", "arguments": {"panel": "panel-2"}},
        ]}))["structuredContent"].clone();
        assert_eq!((stopped["completed"].clone(), stopped["stopped_at"].clone()), (json!(false), json!(1)), "{stopped}");
        assert!(stopped["results"][1]["error"].as_str().unwrap().contains("missing"));
        assert_eq!(stopped["results"].as_array().unwrap().len(), 2);
        let unmet = call(server.address, &session, 9, "batch", json!({"calls": [{"tool": "fit", "arguments": {"panel": "p"}, "expect": {"name": "configure"}}]}));
        assert_eq!(unmet["structuredContent"]["completed"], false);
        assert_eq!(call(server.address, &session, 10, "batch", json!({"calls": [{"tool": "batch", "arguments": {"calls": []}}]}))["isError"], true);
    }

    #[test]
    fn a_read_only_endpoint_offers_no_batch() {
        let server = serve(Arc::new(Lab { reads: AtomicU64::new(0), started: AtomicBool::new(false) }), false, 0, "secret".into()).unwrap();
        let session = session(server.address);
        let (_, tools) = post(server.address, "secret", Some(&session), &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}));
        let names: Vec<&str> = tools["result"]["tools"].as_array().unwrap().iter().map(|tool| tool["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"guide") && names.contains(&"wait_for") && names.contains(&"read_file") && !names.contains(&"batch"));
        assert_eq!(call(server.address, &session, 3, "batch", json!({"calls": [{"tool": "get_state"}]}))["isError"], true);
    }

    #[test]
    fn base64_matches_the_standard_alphabet() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }
}
