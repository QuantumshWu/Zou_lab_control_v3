//! Scan hardware's bench part: one broadcast finds the publishing cores, each
//! is asked what it publishes, and every device becomes a one-click
//! `remote.device` row -- no address typed.
use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_plugin_host::Registry;

use crate::wire::{client_handshake, read_frame, send};
use crate::{Bench, FABRIC_VERSION, PROBE};

/// How long the announcers that answered may take to list, all asked at once.
const LISTING_DEADLINE: Duration = Duration::from_secs(15);
/// A named peer resolved slower than this is named in the notes, with its seconds.
const SLOW_PEER: Duration = Duration::from_secs(1);

/// Every device the reachable publishers offer, as Device Manager rows
/// (`{instance_id: remote_<id>, role, parameters: {host, port, instance_id}}`),
/// and why each other publisher that answered is not listed.
pub fn discover(bench: &Bench, port: u16, peers: &[String], registry: &Arc<Registry>, wait: Duration) -> (Vec<Value>, Vec<String>) {
    let started = Instant::now();
    let mut notes = Vec::new();
    let mut announcers: BTreeMap<(IpAddr, u16), String> = BTreeMap::new();
    let socket = match UdpSocket::bind(("0.0.0.0", 0)) {
        Ok(socket) => socket,
        Err(error) => return (Vec::new(), vec![format!("the bench probe could not start: {error}")]),
    };
    let _ = socket.set_broadcast(true);
    let _ = socket.send_to(PROBE, ("255.255.255.255", port));
    for peer in peers {
        let began = Instant::now();
        let (host, peer_port) = match peer.rsplit_once(':').and_then(|(host, port)| port.parse::<u16>().ok().map(|port| (host, port))) {
            Some((host, port)) => (host.to_owned(), port), None => (peer.clone(), port) };
        match (host.as_str(), peer_port).to_socket_addrs().map(|mut addresses| addresses.find(SocketAddr::is_ipv4)) {
            Ok(Some(address)) => { let _ = socket.send_to(PROBE, address); }
            Ok(None) => notes.push(format!("the named peer {peer:?} has no IPv4 address")),
            Err(error) => notes.push(format!("the named peer {peer:?} was not probed: {error} ({:.1}s)", began.elapsed().as_secs_f64())),
        }
        if began.elapsed() > SLOW_PEER { notes.push(format!("the named peer {peer:?} took {:.1}s to resolve; write its address", began.elapsed().as_secs_f64())); }
    }
    let until = Instant::now() + wait;
    let mut buffer = [0u8; 2048];
    while Instant::now() < until {
        let _ = socket.set_read_timeout(Some(until.saturating_duration_since(Instant::now()).max(Duration::from_millis(1))));
        // A reset is Windows reporting an earlier probe to a peer where nothing listens.
        let Ok((size, sender)) = socket.recv_from(&mut buffer) else { continue };
        let Ok(answer) = serde_json::from_slice::<Value>(&buffer[..size]) else { continue };
        let (Some(their_port), spoken) = (answer["port"].as_u64().and_then(|port| u16::try_from(port).ok()), &answer["fabric"]) else { continue };
        if answer["session"].as_str() == Some(bench.session.as_str()) { continue; }
        if *spoken != json!(FABRIC_VERSION) {
            notes.push(format!("the announcer at {}:{their_port} speaks fabric version {spoken} and this machine speaks {FABRIC_VERSION}; \
                                update the older side and restart it", sender.ip()));
            continue;
        }
        announcers.insert((sender.ip(), their_port), answer["name"].as_str().unwrap_or("").to_owned());
    }
    // All asked at once: one that answered the probe and then stalls costs only itself.
    let answers: Arc<Mutex<BTreeMap<(IpAddr, u16), Result<Value, String>>>> = Arc::default();
    let deadline = started + LISTING_DEADLINE;
    for &(ip, their_port) in announcers.keys() {
        let (answers, bench, registry) = (answers.clone(), bench.clone(), registry.clone());
        let _ = std::thread::Builder::new().name("zlc-fabric-list".into()).spawn(move || {
            let answer = list(ip, their_port, &bench, &registry);
            answers.lock().unwrap_or_else(|p| p.into_inner()).insert((ip, their_port), answer);
        });
    }
    while Instant::now() < deadline && answers.lock().unwrap_or_else(|p| p.into_inner()).len() < announcers.len() {
        std::thread::sleep(Duration::from_millis(20));
    }
    let answers = answers.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let mut found = Vec::new();
    let mut offered: BTreeMap<String, String> = BTreeMap::new();
    // Lowest address first, so a name two publishers use is offered from the same one every scan.
    for (&(ip, their_port), name) in &announcers {
        let at = format!("{ip}:{their_port}");
        let devices = match answers.get(&(ip, their_port)) {
            None => { notes.push(format!("the announcer at {at} did not list within {} s", LISTING_DEADLINE.as_secs())); continue; }
            Some(Err(error)) => { notes.push(format!("the announcer at {at} ({name}) did not list: {error}")); continue; }
            Some(Ok(listing)) => listing["devices"].as_array().cloned().unwrap_or_default(),
        };
        for device in devices {
            let Some(instance) = device["instance_id"].as_str().filter(|id| !id.is_empty()) else { continue };
            if let Some(other) = offered.get(instance) {
                notes.push(format!("{instance:?} at {at} is not offered: the announcer at {other} publishes one by that name, or is the same machine reached at another address"));
                continue;
            }
            offered.insert(instance.to_owned(), at.clone());
            let local = format!("remote_{instance}");
            if device["own"] == true {
                // Its own server (v2: the local pulse board): its client type, dialled at the
                // address this scan reached the publisher at -- never one it guessed for itself.
                let mut parameters = device["parameters"].as_object().cloned().unwrap_or_default();
                parameters.insert("host".into(), json!(ip.to_string()));
                found.push(json!({"instance_id": local, "role": device["role"].as_str().filter(|role| !role.is_empty()).unwrap_or(instance),
                    "type_id": device["type_id"], "parameters": parameters}));
                continue;
            }
            found.push(json!({"instance_id": local, "role": device["role"].as_str().filter(|role| !role.is_empty()).unwrap_or(instance),
                "parameters": {"host": ip.to_string(), "port": their_port, "instance_id": instance},
                "in_use_by": device["in_use_by"]}));
        }
    }
    (found, notes)
}

/// One publisher's offer, on a connection of its own.
fn list(ip: IpAddr, port: u16, bench: &Bench, registry: &Arc<Registry>) -> Result<Value, String> {
    let mut stream = std::net::TcpStream::connect_timeout(&SocketAddr::new(ip, port), Duration::from_secs(5)).map_err(|error| error.to_string())?;
    client_handshake(&mut stream, bench, registry)?;
    send(&mut stream, &json!({"t": "req", "id": 1, "op": "list", "p": {}, "deadline_ms": 5000})).map_err(|error| error.to_string())?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    loop {
        match read_frame(&mut stream, registry).map_err(|error| error.to_string())? {
            Some((message, _)) if message["t"] == "res" => {
                if let Some(error) = message.get("error") { return Err(error["message"].as_str().unwrap_or("refused").to_owned()); }
                return Ok(message["ok"].clone());
            }
            Some(_) => continue,
            None => return Err("the publisher closed the connection".into()),
        }
    }
}
