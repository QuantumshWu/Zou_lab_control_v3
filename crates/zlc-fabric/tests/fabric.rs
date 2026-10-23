//! Two cores in one process, each with its own plugin host and real plugin
//! processes (needs ZLC_PLUGIN_PYTHON = the science env's python.exe): one
//! publishes the test echo device, the other uses it through the fabric.
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json::json;
use zlc_core::AnyBuffer;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::Runtime;
use zlc_fabric::{Bench, Link, Publisher};
use zlc_plugin_host::{Endpoint, ErrorKind, HostConfig, HostEvent, Lend, PluginHost, RunRequest};

struct Core { host: PluginHost, runtime: Runtime, safe: SafeActions, events: mpsc::Receiver<HostEvent> }

fn core() -> Option<Core> {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping the fabric test");
        return None;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("../zlc-plugin-host/tests/plugins")],
        packages: vec!["zlc_test_plugins".into()], memory_limit_mib: Some(3072) };
    let (runtime, safe) = (Runtime::new(), SafeActions::new());
    let (sender, events) = mpsc::channel();
    let sender = Mutex::new(sender);
    let host = PluginHost::new(config, runtime.clone(), safe.clone(), move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    Some(Core { host, runtime, safe, events })
}

fn bench(name: &str) -> Bench { Bench { name: name.into(), ..Bench::new() } }

fn owner(values: &AnyBuffer) -> String {
    let (owner, _) = values.foreign().expect("borrowed from a block, not copied");
    owner.as_any().downcast_ref::<Lend>().expect("a block").name().to_owned()
}

/// Attach `instance` of the publisher at `port` into `core` as `local`.
fn attach(core: &Core, port: u16, instance: &str, local: &str, bench: Bench) -> Link {
    let (link, attached) = Link::open("127.0.0.1", port, instance, bench, core.host.registry().clone()).unwrap();
    assert_eq!(attached.descriptor.type_id, "test.echo");
    let (told, name) = (core.host.clone(), local.to_owned());
    link.on_event(move |op, params, lends| told.remote_event(&name, op, params, lends));
    assert!(attached.reads.iter().any(|name| name == "convert_tunable_value"));
    core.host.attach(local, attached.descriptor, Arc::new(link.clone()), attached.tunable, attached.reads).unwrap();
    let (lost, name) = (core.host.clone(), local.to_owned());
    link.on_failure(|_| {}, move |reason| lost.lost(&name, reason));
    link
}

fn failed(core: &Core, instance: &str, within: Duration) -> String {
    let until = Instant::now() + within;
    loop {
        match core.events.recv_timeout(until.saturating_duration_since(Instant::now())) {
            Ok(HostEvent::Failed { instances, reason }) if instances.iter().any(|name| name == instance) => return reason,
            Ok(_) => continue,
            Err(_) => panic!("{instance} was not lost within {within:?}"),
        }
    }
}

#[test]
fn a_published_device_serves_another_core_through_its_blocks_claims_and_safe_actions() {
    let (Some(a), Some(b)) = (core(), core()) else { return };
    a.host.open("echo", "test.echo", json!({"gain": 2.0}), json!({})).unwrap();
    let publisher = Publisher::start(a.host.clone(), bench("pc2"), "127.0.0.1", 0).unwrap();
    publisher.publish("echo", "Echo").unwrap();
    // Handed over: the publishing core no longer reaches it.
    assert_eq!(a.host.spi("echo", "frame", json!([2, 1]), json!({}), Duration::from_secs(5)).err().unwrap().kind, ErrorKind::Refused);

    let user = bench("pc1");
    let link = attach(&b, publisher.port(), "echo", "remote_echo", user.clone());
    // A frame lands in a block of the using core.
    let reply = b.host.spi("remote_echo", "frame", json!([256, 5]), json!({}), Duration::from_secs(10)).unwrap();
    let frame = reply.array(&reply.value).unwrap();
    let AnyBuffer::U16(values) = &frame.values else { panic!("dtype") };
    assert!(frame.shape == [256, 256] && values.as_slice().iter().all(|value| *value == 5));
    assert_eq!(b.host.registry().owner_of(&owner(&frame.values)).unwrap(), std::process::id());
    drop((frame, reply));
    assert_eq!(b.host.spi("remote_echo", "get:opened_with", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!({"gain": 2.0}));
    // Its commands are this core's catalogue entries.
    assert_eq!(b.host.command("remote_echo__frame_total", &json!({"size": 4})).unwrap().value, json!(32));
    // What it says for itself (a notice, its frame as a block) this core hears under its own name; the publisher does not.
    assert_eq!(b.host.command("remote_echo__tell", &json!({"size": 3})).unwrap().value, json!(3));
    let (name, value, lent) = loop {
        match b.events.recv_timeout(Duration::from_secs(10)).expect("the device's notice") {
            HostEvent::Notice { instance, name, value, lent } => { assert_eq!(instance, "remote_echo"); break (name, value, lent); }
            _ => {}
        }
    };
    let told = lent.array(&value["frame"]).unwrap();
    let AnyBuffer::U16(values) = &told.values else { panic!("dtype") };
    assert!(name == "told" && told.shape == [3, 3] && values.as_slice().iter().all(|value| *value == 7));
    drop(told);
    assert!(a.events.try_iter().all(|event| !matches!(event, HostEvent::Notice { .. })));
    // Its files are this core's: what it writes lands here, what it reads comes from here (a block past 16 KB).
    let file = std::env::temp_dir().join(format!("zlc-fabric-files-{}", std::process::id())).join("kept.txt");
    let text = "remote ".repeat(4000);
    assert_eq!(b.host.command("remote_echo__keep", &json!({"path": file, "text": text})).unwrap().value, json!(text.len()));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), text);
    assert_eq!(b.host.command("remote_echo__take", &json!({"path": file})).unwrap().value, json!(text));
    let _ = std::fs::remove_dir_all(file.parent().unwrap());

    // A node of this core runs on it: frames out, a frame back in, published here without another copy.
    b.host.open("counter", "counter", json!({}), json!({})).unwrap();
    let signal = "@logic/counter/frames";
    let run = b.host.start_run("counter", RunRequest { device_labels: Default::default(), values: json!({"count": 3, "size": 128}), devices: [("echo".to_owned(), "remote_echo".to_owned())].into(),
        shared: Default::default(), origin: Origin::Operator, resources: json!({}), source_signal: None, run_directory: None, outputs: [("frames".to_owned(), signal.to_owned())].into() }).unwrap();
    let (outcome, result, error) = loop {
        match b.events.recv_timeout(Duration::from_secs(60)).expect("run events") {
            HostEvent::Request { run: id, ticket, params, .. } if id == run => b.host.respond(ticket, Ok(json!({"pong": params["params"]["value"]}))),
            HostEvent::RunFinished { run: id, outcome, result, error, .. } if id == run => break (outcome, result, error),
            HostEvent::Failed { reason, .. } => panic!("lost: {reason}"),
            _ => {}
        }
    };
    assert_eq!(outcome, "completed", "{error:?}");
    assert_eq!(result, json!({"published": 3, "device_total": 3 * 128 * 128, "core": {"pong": 3}}));
    let source = b.runtime.signals().into_iter().find(|item| item.signal.as_ref() == signal).expect("published");
    let latest = b.runtime.read_latest(&source.generation, signal).unwrap();
    let AnyBuffer::U16(published) = latest.data().unwrap().values() else { panic!("dtype") };
    assert!(published.as_slice().iter().all(|value| *value == 3));
    assert_eq!(b.host.registry().owner_of(&owner(latest.data().unwrap().values())).unwrap(), std::process::id());
    assert_eq!(latest.run().metadata["opened_with"], json!({"gain": 2.0}));
    assert_eq!(latest.run().metadata["converted"], json!([2.0, 4.0]));
    assert!((latest.run().metadata["tuned_dbm"].as_f64().unwrap() - 0.969100130080564).abs() < 1e-10);
    assert_eq!(latest.run().metadata["restored_dbm"], 0.0);

    // Its outputs join this core's Stop all; their SAFE goes through the link.
    assert_eq!(b.safe.outputs(), ["remote_echo/beam"]);
    assert_eq!(b.host.spi("remote_echo", "fire", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!("fired"));
    assert!(b.safe.make_safe().is_empty());
    assert_eq!(b.host.spi("remote_echo", "made_safe", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!(["beam"]));

    // Another core finds it in use.
    let error = Link::open("127.0.0.1", publisher.port(), "echo", bench("pc3"), b.host.registry().clone()).err().unwrap();
    assert!(error.contains("in use by pc1#"), "{error}");
    let scanned = zlc_fabric::discover(&user, publisher.port(), &[format!("127.0.0.1:{}", publisher.port())], b.host.registry(), Duration::from_millis(500));
    // (The broadcast may reach the same publisher at another address: it is offered once.)
    assert_eq!(scanned.0.len(), 1, "{scanned:?}");
    assert_eq!((scanned.0[0]["parameters"]["instance_id"].as_str(), scanned.0[0]["parameters"]["port"].as_u64()), (Some("echo"), Some(u64::from(publisher.port()))));
    assert_eq!((scanned.0[0]["instance_id"].as_str(), scanned.0[0]["role"].as_str()), (Some("remote_echo"), Some("Echo")));

    // Withdrawn: lost there as a crashed plugin is, its outputs gone from that core; usable here again, left safe.
    publisher.withdraw("echo").unwrap();
    let reason = failed(&b, "remote_echo", Duration::from_secs(10));
    assert!(reason.contains("withdrew"), "{reason}");
    assert!(b.host.instances().iter().all(|(name, ..)| name != "remote_echo"));
    assert!(b.safe.outputs().is_empty());
    // Made safe by Stop all there and by the withdrawal.
    assert_eq!(a.host.spi("echo", "made_safe", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!(["beam", "beam"]));
    drop(link);
    drop(publisher);
    a.host.shutdown();
    b.host.shutdown();
}

#[test]
fn a_device_with_its_own_server_is_announced_as_the_client_type_another_machine_opens() {
    let (Some(a), Some(b)) = (core(), core()) else { return };
    a.host.open("served", "test.served", json!({"port": 18990}), json!({})).unwrap();
    let publisher = Publisher::start(a.host.clone(), bench("pc2"), "127.0.0.1", 0).unwrap();
    publisher.publish("served", "Served").unwrap();
    let admitted = || a.host.spi("served", "get:admitted", json!([]), json!({}), Duration::from_secs(5)).unwrap().value;
    assert_eq!(admitted(), json!([true]));
    // Scan hardware offers it as v2 did the local pulse board and SLM head: its client type, at the address found.
    let (found, notes) = zlc_fabric::discover(&bench("pc1"), publisher.port(), &[format!("127.0.0.1:{}", publisher.port())], b.host.registry(), Duration::from_millis(500));
    assert_eq!(found.len(), 1, "{found:?} {notes:?}");
    assert_eq!((found[0]["instance_id"].as_str(), found[0]["type_id"].as_str()), (Some("remote_served"), Some("test.echo")));
    assert_eq!(found[0]["parameters"], json!({"port": 18990, "host": "127.0.0.1"}));
    // A core using it as that client watches the announcement: the withdrawal reaches it as a remote.device's does.
    assert!(Link::watch("127.0.0.1", publisher.port(), "test.echo", &json!(1), bench("pc1"), b.host.registry().clone()).is_err(), "nothing there");
    let watch = Link::watch("127.0.0.1", publisher.port(), "test.echo", &json!(18990), bench("pc1"), b.host.registry().clone()).unwrap();
    let (sender, lost) = mpsc::channel();
    let sender = Mutex::new(sender);
    watch.on_failure(|_| {}, move |reason| { let _ = sender.lock().unwrap().send(reason.to_owned()); });
    publisher.withdraw("served").unwrap();
    assert_eq!(admitted(), json!([true, false]));
    let reason = lost.recv_timeout(Duration::from_secs(5)).expect("the watch is told");
    assert!(reason.contains("withdrew"), "{reason}");
    drop(publisher);
    a.host.shutdown();
    b.host.shutdown();
}

/// A TCP relay the test can cut (a network blip) or refuse (a dead machine).
struct Relay { port: u16, pairs: Arc<Mutex<Vec<(TcpStream, TcpStream)>>>, refusing: Arc<AtomicBool> }

impl Relay {
    fn new(target: u16) -> Relay {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (pairs, refusing) = (Arc::new(Mutex::new(Vec::new())), Arc::new(AtomicBool::new(false)));
        let (kept, refuse) = (pairs.clone(), refusing.clone());
        std::thread::spawn(move || for client in listener.incoming().flatten() {
            if refuse.load(Ordering::Acquire) { let _ = client.shutdown(Shutdown::Both); continue; }
            let Ok(server) = TcpStream::connect(("127.0.0.1", target)) else { continue };
            for (mut from, mut to) in [(client.try_clone().unwrap(), server.try_clone().unwrap()), (server.try_clone().unwrap(), client.try_clone().unwrap())] {
                std::thread::spawn(move || { let _ = std::io::copy(&mut from, &mut to); let _ = to.shutdown(Shutdown::Both); });
            }
            kept.lock().unwrap().push((client, server));
        });
        Relay { port, pairs, refusing }
    }
    fn cut(&self) { for (client, server) in self.pairs.lock().unwrap().drain(..) { let _ = client.shutdown(Shutdown::Both); let _ = server.shutdown(Shutdown::Both); } }
}

#[test]
fn a_broken_link_fails_what_was_in_flight_relinks_and_a_silent_publisher_loses_the_device() {
    let (Some(a), Some(b)) = (core(), core()) else { return };
    a.host.open("echo", "test.echo", json!({}), json!({})).unwrap();
    let publisher = Publisher::start(a.host.clone(), bench("pc2"), "127.0.0.1", 0).unwrap();
    publisher.publish("echo", "Echo").unwrap();
    let relay = Relay::new(publisher.port());
    let link = attach(&b, relay.port, "echo", "remote_echo", bench("pc1"));
    b.host.spi("remote_echo", "frame", json!([2, 1]), json!({}), Duration::from_secs(5)).unwrap();

    // A call in flight when the line breaks fails as lost; it is not sent again.
    let calling = b.host.clone();
    let slow = std::thread::spawn(move || calling.spi("remote_echo", "sleep", json!([1.5]), json!({}), Duration::from_secs(10)));
    std::thread::sleep(Duration::from_millis(300));
    relay.cut();
    assert_eq!(slow.join().unwrap().err().unwrap().kind, ErrorKind::Lost);
    // The link comes back by itself; the next call waits for it.
    let again = b.host.spi("remote_echo", "made_safe", json!([]), json!({}), Duration::from_secs(10)).unwrap();
    let safe = again.value.as_array().cloned().unwrap_or_default();
    assert!(!safe.is_empty() && safe.iter().all(|output| output == "beam"), "the publisher drove the abandoned output safe: {safe:?}");
    assert!(link.tail(20).iter().any(|line| line.starts_with("FABRIC RELINKED")), "{:?}", link.tail(20));

    // A publisher that stays unreachable loses the device after the relink window.
    relay.refusing.store(true, Ordering::Release);
    relay.cut();
    let started = Instant::now();
    let reason = failed(&b, "remote_echo", Duration::from_secs(30));
    assert!(reason.contains("did not come back"), "{reason}");
    assert!(started.elapsed() >= Duration::from_secs(9), "{:?}", started.elapsed());
    drop(link);
    drop(publisher);
    a.host.shutdown();
    b.host.shutdown();
}
