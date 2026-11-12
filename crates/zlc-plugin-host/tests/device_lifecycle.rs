//! A device's life in the session journal, across a real plugin process (needs ZLC_PLUGIN_PYTHON):
//! its process's start, each open with where and how long, what the device says of itself, and a
//! failure whole, with the plugin's Python traceback.
mod support;

use std::path::PathBuf;

use serde_json::{Value, json};
use zlc_core::control::SafeActions;
use zlc_core::journal;
use zlc_core::runtime::Runtime;
use zlc_plugin_host::{HostConfig, PluginHost};

#[test]
fn an_open_is_kept_with_its_process_its_time_and_a_failure_whole() {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping the plugin process test");
        return;
    };
    let folder = std::env::temp_dir().join(format!("zlc-journal-lifecycle-{}", journal::now_us()));
    journal::open(&folder).unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
        packages: vec!["zlc_test_plugins".into()], memory_limit_mib: Some(3072) };
    let host = PluginHost::new(config, Runtime::new(), SafeActions::new(), |_| {}).unwrap();
    support::install(&host).unwrap();

    // What the device says of itself comes back with its open.
    let opened = host.open("echo", "test.echo", json!({"gain": 2.0}), json!({})).unwrap();
    assert_eq!(opened["device_info"], json!({"model": "echo", "serial": "gain-2.0"}), "a fact with no value is left out");
    let pid = host.instances().into_iter().find(|(name, ..)| name == "echo").unwrap().2;
    // A device whose open raises: its failure, traceback and all.
    let held = folder.join("held.txt");
    let failed = host.open("broken", "test.failed_init", json!({"path": held}), json!({})).unwrap_err();
    assert!(failed.traceback.contains("initialization verification failed"), "{failed:?}");
    host.close("echo").unwrap();

    let read = || -> Vec<Value> {
        let (session, _) = journal::sessions().unwrap();
        std::fs::read_to_string(&session).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect()
    };
    // The process's end is said by its reader once its output closes, a moment after it was retired.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let lines = loop {
        let lines = read();
        if lines.iter().any(|line| line["ev"] == "process_ended" && line["pid"] == pid) || std::time::Instant::now() > deadline { break lines; }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let said = |event: &str, instance: &str| lines.iter().find(|line| line["ev"] == event && line["instance"] == instance).cloned()
        .unwrap_or_else(|| panic!("no {event} of {instance} in {lines:#?}"));
    // The process the device opened in (discovery may have had one of its own before it).
    let started = lines.iter().find(|line| line["ev"] == "process_started" && line["pid"] == pid).expect("the process's start");
    assert_eq!(started["key"], "plugins");
    // It ended once nothing used it any more: asked to, not a crash.
    let ended = lines.iter().find(|line| line["ev"] == "process_ended" && line["pid"] == pid).expect("the process's end");
    assert_eq!(ended["expected"], true);

    let open = said("device_open", "echo");
    assert_eq!((&open["src"], &open["type_id"], &open["pid"], &open["outcome"], &open["error"]), (&json!("plugins"), &json!("test.echo"), &json!(pid), &json!("opened"), &Value::Null));
    assert!(open["ms"].as_f64().is_some_and(|ms| ms > 0.0), "how long it took: {open}");

    let broken = said("device_open", "broken");
    assert_eq!((&broken["outcome"], &broken["error"]["type"]), (&json!("failed"), &json!("builtins:RuntimeError")));
    let traceback = broken["error"]["traceback"].as_str().unwrap();
    assert!(traceback.starts_with("Traceback") && traceback.contains("initialization verification failed"), "the traceback whole: {traceback}");

    let closed = said("device_close", "echo");
    assert_eq!((&closed["pid"], &closed["outcome"]), (&json!(pid), &json!("closed")));
    // The same lines, in the same order, in the rolling journal.
    let rolling: Vec<Value> = journal::segments().iter().flat_map(|path| std::fs::read_to_string(path).unwrap().lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap()).collect::<Vec<_>>()).collect();
    assert!(lines.iter().all(|line| rolling.contains(line)), "every lifecycle line is in the hour's journal too");
    drop(host);
    std::fs::remove_dir_all(&folder).ok();
}
