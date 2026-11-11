//! The flight journal across a real plugin process (needs ZLC_PLUGIN_PYTHON): what a plugin says
//! reaches the core's journal stamped with the time the plugin observed it, beside the core's own
//! lines of the call that made it say so.
mod support;

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{Value, json};
use zlc_core::control::SafeActions;
use zlc_core::journal;
use zlc_core::runtime::Runtime;
use zlc_plugin_host::{HostConfig, PluginHost};

#[test]
fn a_plugins_line_is_filed_among_the_cores_own() {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping the plugin process test");
        return;
    };
    let folder = std::env::temp_dir().join(format!("zlc-journal-host-{}", journal::now_us()));
    journal::open(&folder).unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
        packages: vec!["zlc_test_plugins".into()], memory_limit_mib: Some(3072) };
    let host = PluginHost::new(config, Runtime::new(), SafeActions::new(), |_| {}).unwrap();
    support::install(&host).unwrap();
    host.open("echo", "test.echo", json!({"gain": 2.0}), json!({})).unwrap();
    let pid = host.instances().into_iter().find(|(name, ..)| name == "echo").unwrap().2;
    let before = journal::now_us();
    assert_eq!(host.spi("echo", "fire", json!([]), json!({}), Duration::from_secs(10)).unwrap().value, json!("fired"));
    let read = || -> Vec<Value> {
        journal::segments().iter().flat_map(|path| std::fs::read_to_string(path).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect::<Vec<Value>>()).collect()
    };
    // The plugin's line is an event of its own: it follows the answer on the same channel.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let said = loop {
        if let Some(line) = read().into_iter().find(|line| line["src"] == "test") { break line; }
        assert!(std::time::Instant::now() < deadline, "the plugin's line never reached the journal: {:?}", read());
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!((&said["ev"], &said["gain"], &said["pid"]), (&json!("fired"), &json!(2.0), &json!(pid)));
    let observed = said["us"].as_u64().unwrap();
    assert!(observed >= before && observed <= journal::now_us(), "stamped when the plugin observed it: {observed} not in [{before}, now]");
    drop(host);
    std::fs::remove_dir_all(&folder).ok();
}
