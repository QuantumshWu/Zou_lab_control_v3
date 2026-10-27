//! Remote (MCP/AI) clients on the instrument's one command queue.
//!
//! A tool call becomes the same command event the window sends, marked with
//! its origin; the window's validation, queue and completion
//! path then run unchanged, and the call is answered when the instrument
//! state reports that request settled. Queries are answered from the accepted
//! state, the runtime and the last accepted front of each panel.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use serde_json::{Value, json};
use zlc_bus::mcp::{self, Host, ToolOutput};
use zlc_core::control::Origin;
use zlc_core::runtime::Runtime;
use zlc_core::schema::{DatasetSchema, DomainRole};

use crate::science::Front;

/// First request id given to remote commands; the window counts up from 1.
pub const FIRST_REMOTE_REQUEST: u64 = 8_000_000_000_000_000;

pub type Reply = mpsc::Sender<Result<ToolOutput, String>>;

/// One tool call travelling into the event loop.
pub struct RemoteCall { pub origin: Origin, pub name: String, pub params: Value, pub reply: Reply }

/// What the event loop publishes for the endpoint: the tools open plugin instances
/// add (a change is announced to clients holding a notification stream), and
/// whether the instrument is closing.
#[derive(Default)]
pub struct ToolList { list: Mutex<Vec<Value>>, notifier: std::sync::OnceLock<mcp::Notifier>, closing: std::sync::atomic::AtomicBool }
pub type Tools = Arc<ToolList>;

impl ToolList {
    pub fn publish(&self, tools: Vec<Value>) {
        let changed = self.list.lock().map(|mut list| if *list != tools { *list = tools; true } else { false }).unwrap_or(false);
        if changed { if let Some(notifier) = self.notifier.get() { notifier.tools_changed(); } }
    }
    pub fn set_closing(&self) { self.closing.store(true, std::sync::atomic::Ordering::Release); }
}

/// The longest a wait may watch (v2 `lab_wait`).
pub const WAIT_SECONDS: f64 = 1800.0;

struct DesktopHost { send: Mutex<Box<dyn Fn(RemoteCall) -> Result<(), String> + Send>>, tools: Tools, root: PathBuf, runtime: Runtime }

impl Host for DesktopHost {
    fn tools(&self) -> Vec<Value> { self.tools.list.lock().map(|tools| tools.clone()).unwrap_or_default() }
    fn identity(&self) -> Value {
        let closing = self.tools.closing.load(std::sync::atomic::Ordering::Acquire);
        json!({"workspace": self.root, "service": if closing { "closing" } else { "running" }})
    }
    fn call(&self, origin: Origin, name: &str, params: Value) -> Result<ToolOutput, String> {
        // Files, signal pages and the plot tables are read here, never on the window's turn.
        match name {
            "list_files" => return crate::readers::list_files(&self.root, &params),
            "read_file" => return crate::readers::read_file(&self.root, &params),
            "read_signal" => return crate::readers::read_signal(&self.runtime, &params),
            "plot_catalog" => return crate::readers::plot_catalog(&params),
            _ => {}
        }
        let (reply, answer) = mpsc::channel();
        let limit = if name == "wait" {
            params["timeout_seconds"].as_f64().unwrap_or(30.0).clamp(0.0, WAIT_SECONDS) + 10.0
        } else { 900.0 };
        (self.send.lock().map_err(|_| "instrument is shutting down")?)(RemoteCall { origin, name: name.to_owned(), params, reply })?;
        answer.recv_timeout(Duration::from_secs_f64(limit))
            .map_err(|_| format!("the instrument did not answer {name} within {limit:.0} s"))?
    }
}

/// Start the loopback endpoint and record where it is, its token and whether
/// clients may change anything (`control`, v2 `control_enabled`).
pub fn start_endpoint(send: impl Fn(RemoteCall) -> Result<(), String> + Send + 'static, control: bool, tools: Tools, root: PathBuf, runtime: Runtime)
                      -> Result<(mcp::Server, PathBuf, PathBuf), String> {
    // The token and this instrument's connection record, in the checkout's .local\mcp.
    let directory = zlc_bus::proxy::records_directory();
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let record = directory.join("mcp.json");
    // The token is kept across launches so a registered client keeps working;
    // deleting the file revokes every client.
    let token = std::fs::read_to_string(&record).ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|value| value["token"].as_str().map(str::to_owned))
        .filter(|token| token.len() >= 32)
        .unwrap_or_else(|| format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple()));
    let port: u16 = std::env::var("ZLC_V3_MCP_PORT").ok().and_then(|text| text.parse().ok()).unwrap_or(8767);
    let workspace = root.display().to_string();
    let host: Arc<dyn Host> = Arc::new(DesktopHost { send: Mutex::new(Box::new(send)), tools: tools.clone(), root, runtime });
    let server = mcp::serve(host.clone(), control, port, token.clone())
        .or_else(|_| mcp::serve(host, control, 0, token.clone()))
        .map_err(|error| format!("MCP endpoint: {error}"))?;
    let _ = tools.notifier.set(server.notifier());
    let url = format!("http://{}/mcp", server.address);
    std::fs::write(&record, serde_json::to_string_pretty(&json!({"url": url, "token": token, "pid": std::process::id(), "control_enabled": control,
        "register": format!("claude mcp add --transport http zlc {url} --header \"Authorization: Bearer {token}\""),
        "register_any": "zlc-mcp --client-config claude (one entry point for every experiment running on this computer)"})).unwrap())
        .map_err(|error| error.to_string())?;
    // This instrument among the others running on this computer (zlc-mcp lists them); removed at exit.
    let connection = directory.join(format!("connection-{}-{}.json", std::process::id(), server.address.port()));
    std::fs::write(&connection, serde_json::to_string_pretty(&json!({"url": url, "token": token, "pid": std::process::id(),
        "workspace": workspace, "control_enabled": control})).unwrap()).map_err(|error| error.to_string())?;
    Ok((server, record, connection))
}

/// A remote client's relative file paths are the workspace's (v2: workspace-relative), as the workbench reads its own.
pub fn workspace_paths(params: &mut Value, root: &std::path::Path) {
    let Some(object) = params.as_object_mut() else { return };
    for key in ["path", "document"] {
        if let Some(Value::String(path)) = object.get_mut(key) {
            if !path.is_empty() && std::path::Path::new(path.as_str()).is_relative() { *path = root.join(&*path).display().to_string(); }
        }
    }
}

/// The command a tool call becomes, or None for calls answered elsewhere.
pub fn command_for(call: &RemoteCall, id: u64, accepted: &Value, root: &std::path::Path) -> Result<Value, String> {
    let mut command = call.params.clone();
    workspace_paths(&mut command, root);
    let object = command.as_object_mut().ok_or("tool arguments must be an object")?;
    if matches!(call.name.as_str(), "open" | "save_layout" | "load_layout" | "save_apparatus" | "load_apparatus" | "pulse_open")
        && object.get("path").is_none_or(Value::is_null) {
        return Err(format!("{} needs a path: a remote client has no file dialog", call.name));
    }
    if call.name == "configure" {
        if let Some(selectors) = object.get("selectors").cloned() {
            object.insert("selectors".into(), wire_selectors(&selectors, object["panel"].as_str().unwrap_or(""), accepted)?);
        }
    }
    object.insert("id".into(), json!(id));
    object.insert("action".into(), json!(call.name));
    object.insert("_origin".into(), json!(match &call.origin { Origin::Operator => "operator".to_owned(), Origin::Remote(name) => name.clone() }));
    Ok(command)
}

/// Remote selectors carry no revision; each kind's continues from the panel's.
fn wire_selectors(selectors: &Value, panel: &str, accepted: &Value) -> Result<Value, String> {
    let current = accepted["panels"].as_array().and_then(|rows| rows.iter().find(|row| row["id"] == panel))
        .map(|row| row["target"]["selectors"].clone()).unwrap_or(Value::Null);
    let mut revisions: BTreeMap<String, u64> = BTreeMap::new();
    for selector in current.as_array().into_iter().flatten() {
        if let (Some(kind), Some(revision)) = (selector["kind"].as_str(), selector["revision"].as_u64()) {
            let entry = revisions.entry(kind.to_owned()).or_default();
            *entry = (*entry).max(revision);
        }
    }
    let mut result = Vec::new();
    for selector in selectors.as_array().ok_or("selectors must be a list")? {
        let kind = selector["kind"].as_str().ok_or("selector kind")?;
        let revision = revisions.get(kind).copied().unwrap_or(0) + 1;
        result.push(json!({"kind": kind, "value": selector["value"], "facet_index": selector.get("facet_index").cloned().unwrap_or(Value::Null),
                           "revision": revision}));
    }
    Ok(Value::Array(result))
}

/// The origin a command event carries, removing the mark.
pub fn take_origin(command: &mut Value) -> Origin {
    match command.as_object_mut().and_then(|object| object.remove("_origin")) {
        Some(Value::String(name)) if name != "operator" => Origin::Remote(name),
        _ => Origin::Operator,
    }
}

/// A command's parameters as its schema sees them.
pub fn parameters(command: &Value) -> Value {
    let mut params = command.clone();
    if let Some(object) = params.as_object_mut() {
        object.retain(|key, _| key != "id" && key != "action" && !key.starts_with('_'));
    }
    params
}

/// A pending remote command and what it acts on.
/// `workbench`: the reply is the workbench's own result, not a panel summary.
/// `save`: a `save`, whose files already there are a refusal.
pub struct Pending { pub reply: Reply, pub panel: Option<String>, pub panels_before: BTreeSet<String>, pub workbench: bool, pub save: bool }

fn panel_ids(accepted: &Value) -> BTreeSet<String> {
    accepted["panels"].as_array().into_iter().flatten().filter_map(|row| row["id"].as_str().map(str::to_owned)).collect()
}

pub fn pending(call: &RemoteCall, accepted: &Value) -> Pending {
    Pending { reply: call.reply.clone(), panel: call.params["panel"].as_str().map(str::to_owned), panels_before: panel_ids(accepted),
        // get_fit answers with the fit table itself (v2 lab_results), as workbench commands answer with their result.
        workbench: zlc_workbench::Workbench::handles(&call.name) || call.name.contains("__") || call.name.starts_with("pulse_") || call.name == "get_fit",
        save: call.name == "save" }
}

/// Answer every remote command whose request the accepted state reports.
pub fn settle(accepted: &Value, pending: &mut BTreeMap<u64, Pending>) {
    let Some(request) = accepted["request"].as_u64() else { return };
    let Some(entry) = pending.remove(&request) else { return };
    if let Some(error) = accepted["error"].as_str() {
        let _ = entry.reply.send(Err(error.to_owned()));
        return;
    }
    // A save onto files already there wrote nothing (the window asks Overwrite or Cancel): refused, naming them.
    if let Some(existing) = accepted["saved"]["exists"].as_array().filter(|_| entry.save) {
        let files: Vec<&str> = existing.iter().filter_map(Value::as_str).collect();
        let one = files.len() == 1;
        let _ = entry.reply.send(Err(format!("Nothing was saved: {} already exist{}. Pass overwrite: true to write over {}.",
                                             files.join(" and "), if one { "s" } else { "" }, if one { "it" } else { "them" })));
        return;
    }
    if entry.workbench {
        let _ = entry.reply.send(Ok(ToolOutput { json: accepted["result"].clone(), png: None }));
        return;
    }
    let now = panel_ids(accepted);
    let added: Vec<String> = now.difference(&entry.panels_before).cloned().collect();
    let mut shown: BTreeSet<String> = added.iter().cloned().collect();
    shown.extend(entry.panel.into_iter().filter(|panel| now.contains(panel)));
    let panels: Vec<Value> = accepted["panels"].as_array().into_iter().flatten()
        .filter(|row| row["id"].as_str().is_some_and(|id| shown.contains(id))).cloned().collect();
    let mut reply = json!({"request": request, "acquisition": accepted["acquisition"], "panel_ids": now, "added_panels": added, "panels": panels,
                         "result": accepted["result"]});
    // A save answers with the files it wrote (the document's `saved`).
    if entry.save { reply["saved"] = accepted["saved"].clone(); }
    let _ = entry.reply.send(Ok(ToolOutput { json: reply, png: None }));
}

/// A remote wait for data, a newer frame or a node's run milestone (`run`: only that run of the node;
/// `past`: the node's ended run when the wait began while a new one was still to start).
pub struct Waiter { pub id: u64, pub produced: Option<u64>, pub panel: Option<(String, u64)>, pub node: Option<(String, String)>,
                    pub run: Option<String>, past: Option<Value>, pub reply: Reply }

fn frame_sequence(accepted: &Value, panel: &str) -> Option<u64> {
    accepted["panels"].as_array()?.iter().find(|row| row["id"] == panel)?["frame"]["sequence"].as_u64()
}

/// The node's run row, from the logic section of the accepted state.
fn node_run<'a>(accepted: &'a Value, node: &str) -> Option<&'a Value> {
    accepted["logic"].as_array()?.iter().find(|row| row["node_id"] == node).map(|row| &row["run"]).filter(|run| run.is_object())
}

/// The run a wait watches: the node's run, unless it is the one that had ended before the wait began.
fn watched<'a>(accepted: &'a Value, waiter: &Waiter) -> Option<&'a Value> {
    waiter.node.as_ref().and_then(|(node, _)| node_run(accepted, node)).filter(|run| waiter.past.as_ref().is_none_or(|past| run["run"] != *past))
}

fn terminal(run: &Value) -> bool { matches!(run["state"].as_str(), Some("completed" | "stopped" | "failed")) }

/// Whether the run has reached `until` itself (not merely ended first).
fn reached(run: &Value, until: &str) -> bool {
    match until { "ready" => run["ready"] == true, "request" => run["request"].is_object(), _ => terminal(run) }
}

/// Whether a node wait is answered now: reached, ended, or the run asks the operator (v2: a review
/// releases the wait so it can be answered).
fn settled(run: &Value, until: &str) -> bool { reached(run, until) || terminal(run) || run["request"].is_object() }

pub fn waiter(id: u64, call: &RemoteCall, accepted: &Value) -> Result<Waiter, String> {
    let panel = match call.params["panel"].as_str() {
        Some(panel) => {
            let after = call.params["after_sequence"].as_u64().or_else(|| frame_sequence(accepted, panel)).unwrap_or(0);
            Some((panel.to_owned(), after))
        }
        None => None,
    };
    let produced = call.params["produced"].as_u64();
    let node = call.params["node"].as_str().map(|node| (node.to_owned(), call.params["until"].as_str().unwrap_or("terminal").to_owned()));
    let run = call.params["run"].as_str().map(str::to_owned);
    let mut past = None;
    if let Some((node, _)) = &node {
        let row = accepted["logic"].as_array().and_then(|rows| rows.iter().find(|row| row["node_id"] == node.as_str()))
            .ok_or_else(|| format!("No logic node {node:?}"))?;
        if let Some(run) = &run {
            if node_run(accepted, node).is_none_or(|current| current["run"] != run.as_str()) {
                return Err(format!("{run} is not {node}'s current run; read get_state {{node}} for the run it has now"));
            }
        }
        // Started (or following) but its new run has not begun: the ended run shown is not the one waited for.
        if run.is_none() && matches!(row["phase"].as_str(), Some("waiting" | "following")) && row["run"].is_object() && terminal(&row["run"]) {
            past = Some(row["run"]["run"].clone());
        }
    } else if run.is_some() { return Err("run names one run of node; give node too".into()); }
    if panel.is_none() && produced.is_none() && node.is_none() { return Err("wait needs produced, panel and/or node".into()); }
    Ok(Waiter { id, produced, panel, node, run, past, reply: call.reply.clone() })
}

fn wait_report(accepted: &Value, waiter: &Waiter, timed_out: bool) -> ToolOutput {
    let run = watched(accepted, waiter);
    let current = waiter.run.as_ref().is_none_or(|named| run.is_some_and(|run| run["run"] == named.as_str()));
    let done = !timed_out && current && waiter.node.as_ref().is_none_or(|(_, until)| run.is_some_and(|run| reached(run, until)))
        && waiter.produced.is_none_or(|count| accepted["acquisition"]["produced"].as_u64().unwrap_or(0) >= count)
        && waiter.panel.as_ref().is_none_or(|(panel, after)| frame_sequence(accepted, panel).is_some_and(|now| now > *after));
    ToolOutput { json: json!({"timed_out": timed_out, "reached": done, "still_current": current, "acquisition": accepted["acquisition"],
        "frame_sequence": waiter.panel.as_ref().and_then(|(panel, _)| frame_sequence(accepted, panel)),
        "run": run.cloned()}), png: None }
}

/// Answer the waits the accepted state now satisfies (or all of them, expired).
pub fn check_waiters(accepted: &Value, waiters: &mut Vec<Waiter>, expired: Option<u64>) {
    waiters.retain(|waiter| {
        let produced_ok = waiter.produced.is_none_or(|count| accepted["acquisition"]["produced"].as_u64().unwrap_or(0) >= count);
        let run = watched(accepted, waiter);
        // Another run of the node replaced the one waited on.
        let replaced = waiter.run.as_ref().is_some_and(|named| run.is_none_or(|run| run["run"] != named.as_str()));
        let frame_ok = waiter.panel.as_ref().is_none_or(|(panel, after)| frame_sequence(accepted, panel).is_some_and(|now| now > *after))
            && waiter.node.as_ref().is_none_or(|(_, until)| run.is_some_and(|run| settled(run, until)));
        let stopped = matches!(accepted["acquisition"]["state"].as_str(), Some("stopped" | "error" | "idle")) && waiter.produced.is_some();
        if (produced_ok && frame_ok) || replaced {
            let _ = waiter.reply.send(Ok(wait_report(accepted, waiter, false)));
            false
        } else if expired == Some(waiter.id) || (stopped && !produced_ok) {
            let _ = waiter.reply.send(Ok(wait_report(accepted, waiter, expired == Some(waiter.id))));
            false
        } else { true }
    });
}

fn schema_json(schema: &DatasetSchema) -> Value {
    let domains = [("repeat", DomainRole::Repeat), ("point", DomainRole::Point), ("cell", DomainRole::Cell)];
    let mut result = serde_json::Map::new();
    for (name, role) in domains {
        result.insert(name.into(), Value::Array(schema.domain(role).axes().iter().map(|axis| json!({
            "id": axis.id(), "name": axis.name(), "role": axis.role(), "size": axis.len(), "unit": axis.unit()})).collect()));
    }
    result.insert("value".into(), json!({"unit": schema.value().unit(), "name": schema.value().name(),
                                          "dtype": format!("{:?}", schema.dtype()).to_lowercase()}));
    Value::Object(result)
}

/// The live signals panels can plot.
pub fn signals(runtime: &Runtime) -> Value {
    Value::Array(runtime.signals().into_iter().map(|source| json!({
        "signal": source.signal.as_ref(), "published": source.published, "finite": source.finite,
        "sealed": source.sealed, "history_capacity": source.history_capacity, "retained_events": source.retained_events,
        "schema": schema_json(&source.schema)})).collect())
}

/// One front as PNG, exactly its pixels.
pub fn png(front: &Front) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, front.width, front.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
        writer.write_image_data(front.pixels.as_slice()).map_err(|error| error.to_string())?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(save: bool, accepted: &Value) -> Result<Value, String> {
        let (reply, answers) = mpsc::channel();
        let mut pending = BTreeMap::from([(7, Pending { reply, panel: None, panels_before: BTreeSet::new(), workbench: false, save })]);
        settle(accepted, &mut pending);
        answers.recv().unwrap().map(|output| output.json)
    }

    /// A remote save onto files already there is refused, naming them and the flag; what it wrote is its reply.
    #[test]
    fn a_save_onto_existing_files_is_refused_naming_them() {
        let both = json!({"request": 7, "error": null, "panels": [], "saved": {"exists": ["C:\\d\\a.png", "C:\\d\\a.npz"]}});
        assert_eq!(answer(true, &both).unwrap_err(), "Nothing was saved: C:\\d\\a.png and C:\\d\\a.npz already exist. Pass overwrite: true to write over them.");
        let one = json!({"request": 7, "error": null, "panels": [], "saved": {"exists": ["C:\\d\\a.npz"]}});
        assert_eq!(answer(true, &one).unwrap_err(), "Nothing was saved: C:\\d\\a.npz already exists. Pass overwrite: true to write over it.");
        let written = json!({"request": 7, "error": null, "panels": [], "saved": ["C:\\d\\a.png", "C:\\d\\a.npz"]});
        assert_eq!(answer(true, &written).unwrap()["saved"], json!(["C:\\d\\a.png", "C:\\d\\a.npz"]));
        // Another command answered after a refused save is not that refusal.
        let reply = answer(false, &both).unwrap();
        assert!(reply.get("saved").is_none(), "{reply}");
    }
}
