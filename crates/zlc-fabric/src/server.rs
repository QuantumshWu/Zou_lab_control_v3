//! The publishing core: answer the probe, list what is published, hand one
//! device to one using core at a time and serve its instance ops.
use std::collections::{BTreeMap, HashMap};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_plugin_host::{CallError, ErrorKind, Forward, Lend, PluginHost, TypeDescriptor};

use crate::wire::{read_frame, send, server_handshake, write_frame};
use crate::{Bench, FABRIC_VERSION, PROBE};

/// Both sides say they are there this often...
pub const BEAT: Duration = Duration::from_secs(2);
/// ...and a peer silent this long is gone.
pub const SILENCE: Duration = Duration::from_secs(6);
const SAFE_DEADLINE: Duration = Duration::from_secs(2);
const LONGEST_REQUEST: Duration = Duration::from_secs(600);
/// How long a withdrawal waits for a request that reached the device before it.
const WITHDRAW_WAIT: Duration = Duration::from_secs(150);
const INSTANCE_OPS: [&str; 6] = ["spi", "fields", "tune", "values", "command", "make_safe"];

struct Offer { role: String, descriptor: Arc<TypeDescriptor>, attached: Option<(u64, String)> }

/// A device with a server of its own: what it is announced as, and the cores watching it.
struct Own { role: String, type_id: String, label: String, parameters: Value, watchers: Vec<u64> }

struct Shared {
    host: PluginHost,
    bench: Bench,
    port: u16,
    /// Where the accepting thread is woken to see `closing`.
    wake: SocketAddr,
    closing: AtomicBool,
    offers: Mutex<BTreeMap<String, Offer>>,
    /// Devices with a server of their own (the pulse board): announced, never served here.
    own: Mutex<BTreeMap<String, Own>>,
    in_flight: Mutex<HashMap<String, usize>>,
    settled: Condvar,
    /// Every authenticated connection's writer.
    peers: Mutex<HashMap<u64, Arc<Mutex<TcpStream>>>>,
    next: AtomicU64,
    /// Who else hears what the fabric did with a device (its Device Manager log).
    hook: Mutex<Option<Box<dyn Fn(&str, &str) + Send + Sync>>>,
}

/// One core's publications; dropping it withdraws everything and stops listening.
pub struct Publisher { shared: Arc<Shared>, threads: Vec<JoinHandle<()>> }

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> { mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

impl Publisher {
    /// Listen at `bind:port` (every interface: `0.0.0.0`; TCP for the devices, UDP for the probe).
    pub fn start(host: PluginHost, bench: Bench, bind: &str, port: u16) -> Result<Publisher, String> {
        let listener = TcpListener::bind((bind, port)).map_err(|error| format!("the bench fabric could not start: {error}"))?;
        let mut wake = listener.local_addr().map_err(|error| error.to_string())?;
        if wake.ip().is_unspecified() { wake.set_ip(std::net::Ipv4Addr::LOCALHOST.into()); }
        let port = wake.port();
        let probe = UdpSocket::bind((bind, port)).map_err(|error| format!("the bench fabric could not start: {error}"))?;
        let shared = Arc::new(Shared { host, bench, port, wake, closing: AtomicBool::new(false), offers: Mutex::default(), own: Mutex::default(),
            in_flight: Mutex::default(), settled: Condvar::new(), peers: Mutex::default(), next: AtomicU64::new(1), hook: Mutex::new(None) });
        let accepting = shared.clone();
        let tcp = std::thread::Builder::new().name("zlc-fabric-tcp".into()).spawn(move || {
            for stream in listener.incoming() {
                if accepting.closing.load(Ordering::Acquire) { break; }
                let Ok(stream) = stream else { continue };
                let (shared, id) = (accepting.clone(), accepting.next.fetch_add(1, Ordering::Relaxed));
                let _ = std::thread::Builder::new().name("zlc-fabric-peer".into()).spawn(move || connection(shared, stream, id));
            }
        }).map_err(|error| error.to_string())?;
        let answering = shared.clone();
        let udp = std::thread::Builder::new().name("zlc-fabric-udp".into()).spawn(move || answer_probes(answering, probe)).map_err(|error| error.to_string())?;
        Ok(Publisher { shared, threads: vec![tcp, udp] })
    }

    pub fn port(&self) -> u16 { self.shared.port }

    /// Every line the fabric writes in a device's log is also told to `hook` (instance, line).
    pub fn on_note(&self, hook: impl Fn(&str, &str) + Send + Sync + 'static) { *lock(&self.shared.hook) = Some(Box::new(hook)); }

    /// Offer one open device. Refused for a device a run holds or one that
    /// itself came from the fabric (a relay loop).
    pub fn publish(&self, instance: &str, role: &str) -> Result<(), String> {
        let descriptor = self.shared.host.descriptor(instance).ok_or_else(|| format!("{instance} is not open"))?;
        if !descriptor.announces.is_empty() {
            // A server of its own (v2 announce: a local SLM head): it admits other machines and
            // is announced as the client type they open there, as the pulse board is.
            let answer = self.shared.host.call(instance, "publish", json!({"published": true}), SAFE_DEADLINE).map_err(|error| error.message)?.value;
            let type_id = answer["type_id"].as_str().unwrap_or_default().to_owned();
            let label = self.shared.host.catalog().ok().and_then(|catalog| catalog.find(&type_id).map(|item| item.label.clone())).unwrap_or_else(|| type_id.clone());
            self.publish_own(instance, role, &type_id, &label, answer["parameters"].clone());
            return Ok(());
        }
        let (shared, name) = (Arc::downgrade(&self.shared), instance.to_owned());
        let forward: Forward = Arc::new(move |op, params, lends| if let Some(shared) = shared.upgrade() { shared.forward(&name, op, params, lends); });
        self.shared.host.lend_out(instance, forward).map_err(|error| error.message)?;
        lock(&self.shared.offers).insert(instance.to_owned(), Offer { role: role.to_owned(), descriptor, attached: None });
        self.shared.note(instance, &format!("FABRIC PUBLISH device={instance} port={}", self.shared.port));
        Ok(())
    }

    /// Take a device back: off offer, its user told, its requests in flight
    /// finished, its outputs safe if it was in use, then usable here again.
    pub fn withdraw(&self, instance: &str) -> Result<(), String> {
        if let Some(own) = lock(&self.shared.own).remove(instance) {
            // The cores watching it lose it now, as a remote.device's user does.
            for connection in own.watchers { self.shared.withdrawn(connection, instance); }
            // A plugin device's own server drops the other machines (the pulse board's host does its own).
            let refused = self.shared.host.descriptor(instance).filter(|descriptor| !descriptor.announces.is_empty())
                .and_then(|_| self.shared.host.call(instance, "publish", json!({"published": false}), SAFE_DEADLINE).err());
            self.shared.note(instance, &format!("FABRIC WITHDRAW device={instance}{}", refused.map(|error| format!(" error={:?}", error.message)).unwrap_or_default()));
            return Ok(());
        }
        let offer = lock(&self.shared.offers).remove(instance).ok_or_else(|| format!("{instance} is not published"))?;
        if let Some((connection, _)) = &offer.attached { self.shared.withdrawn(*connection, instance); }
        let mut busy = lock(&self.shared.in_flight);
        let until = Instant::now() + WITHDRAW_WAIT;
        while busy.get(instance).copied().unwrap_or(0) > 0 && Instant::now() < until {
            busy = self.shared.settled.wait_timeout(busy, until.saturating_duration_since(Instant::now())).unwrap_or_else(|p| p.into_inner()).0;
        }
        drop(busy);
        if let Some((_, client)) = &offer.attached { self.shared.make_safe(instance, &offer.descriptor, &format!("withdrawn from {client}")); }
        self.shared.host.take_back(instance);
        self.shared.note(instance, &format!("FABRIC WITHDRAW device={instance}"));
        Ok(())
    }

    /// Announce a device that serves its own protocol (v2: the local pulse board
    /// announced its client type): the using core dials `parameters` itself, at
    /// the address it reached this one at.
    pub fn publish_own(&self, instance: &str, role: &str, type_id: &str, label: &str, parameters: Value) {
        lock(&self.shared.own).insert(instance.to_owned(), Own { role: role.to_owned(), type_id: type_id.to_owned(), label: label.to_owned(), parameters, watchers: Vec::new() });
        self.shared.note(instance, &format!("FABRIC PUBLISH device={instance} as {type_id} (its own server)"));
    }

    /// What is on offer: instance, role and the core using it.
    pub fn offers(&self) -> Vec<Value> {
        let mut offers: Vec<Value> = lock(&self.shared.offers).iter().map(|(instance, offer)| json!({"instance": instance, "role": offer.role,
            "in_use_by": offer.attached.as_ref().map(|(_, client)| client.clone())})).collect();
        offers.extend(lock(&self.shared.own).iter().map(|(instance, own)| json!({"instance": instance, "role": own.role, "in_use_by": null})));
        offers
    }

    pub fn is_published(&self, instance: &str) -> bool { lock(&self.shared.offers).contains_key(instance) || lock(&self.shared.own).contains_key(instance) }

    fn close(&mut self) {
        if self.shared.closing.swap(true, Ordering::AcqRel) { return; }
        let instances: Vec<String> = lock(&self.shared.offers).keys().chain(lock(&self.shared.own).keys()).cloned().collect();
        for instance in instances { let _ = self.withdraw(&instance); }
        for writer in lock(&self.shared.peers).values() { let _ = lock(writer).shutdown(std::net::Shutdown::Both); }
        let _ = TcpStream::connect_timeout(&self.shared.wake, Duration::from_millis(200));
        for thread in self.threads.drain(..) { let _ = thread.join(); }
    }
}

impl Drop for Publisher {
    fn drop(&mut self) { self.close(); }
}

fn answer_probes(shared: Arc<Shared>, socket: UdpSocket) {
    let _ = socket.set_read_timeout(Some(Duration::from_millis(250)));
    let mut buffer = [0u8; 512];
    while !shared.closing.load(Ordering::Acquire) {
        // Windows reports an earlier answer's unreachable port to the next read; any
        // read error is one datagram's news, never the end of the responder.
        let Ok((size, sender)) = socket.recv_from(&mut buffer) else { continue };
        if &buffer[..size] != PROBE { continue; }
        let answer = json!({"fabric": FABRIC_VERSION, "port": shared.port, "name": shared.bench.name, "session": shared.bench.session});
        let _ = socket.send_to(answer.to_string().as_bytes(), sender);
    }
}

impl Shared {
    fn note(&self, instance: &str, line: &str) {
        self.host.note(instance, line);
        if let Some(hook) = lock(&self.hook).as_ref() { hook(instance, line); }
    }

    fn tell(&self, connection: u64, message: &Value) {
        let writer = lock(&self.peers).get(&connection).cloned();
        if let Some(writer) = writer { let _ = send(&mut lock(&writer), message); }
    }

    /// The core using or watching `instance` loses it: this computer took it back.
    fn withdrawn(&self, connection: u64, instance: &str) {
        self.tell(connection, &json!({"t": "evt", "op": "withdrawn", "p": {"instance": instance,
            "reason": format!("{} withdrew it from the bench fabric", self.bench.name)}}));
    }

    /// What a published device says for itself (its notices, its log) goes to
    /// the core using it, its arrays as blocks; nobody attached, nobody hears.
    fn forward(&self, instance: &str, op: &str, params: &Value, lends: Vec<Arc<Lend>>) {
        let connection = lock(&self.offers).get(instance).and_then(|offer| offer.attached.as_ref().map(|(connection, _)| *connection));
        let Some(writer) = connection.and_then(|connection| lock(&self.peers).get(&connection).cloned()) else { return };
        if write_frame(&mut *lock(&writer), &json!({"t": "evt", "op": op, "p": params}), &lends).is_err() {
            let _ = lock(&writer).shutdown(std::net::Shutdown::Both);
        }
    }

    /// Drive a published device's outputs safe (its user is gone or changed).
    fn make_safe(&self, instance: &str, descriptor: &TypeDescriptor, reason: &str) {
        for output in &descriptor.physical_outputs {
            let outcome = self.host.serve(instance, "make_safe", json!({"output": output.name, "plain": true}), Vec::new(), SAFE_DEADLINE);
            self.note(instance, &format!("FABRIC AUTO-SAFE device={instance} output={} reason={reason:?}{}", output.name,
                outcome.err().map(|error| format!(" error={:?}", error.to_string())).unwrap_or_default()));
        }
    }

    fn handle(&self, connection: u64, client: &str, message: &Value, lends: Vec<Arc<Lend>>) -> Result<(Value, Vec<Arc<Lend>>), CallError> {
        let op = message["op"].as_str().unwrap_or("");
        let params = message["p"].clone();
        let instance = params["instance"].as_str().unwrap_or("").to_owned();
        match op {
            "ping" => Ok((json!({}), Vec::new())),
            "list" => {
                let mut devices: Vec<Value> = lock(&self.offers).iter().map(|(instance, offer)| json!({
                    "instance_id": instance, "role": offer.role, "type_id": offer.descriptor.type_id, "label": offer.descriptor.label,
                    "family": offer.descriptor.family, "in_use_by": offer.attached.as_ref().map(|(_, holder)| holder.clone())})).collect();
                // A device with its own server is announced as its client type at its own port.
                devices.extend(lock(&self.own).iter().map(|(instance, own)| json!({
                    "instance_id": instance, "role": own.role, "type_id": own.type_id, "label": own.label, "own": true, "parameters": own.parameters})));
                Ok((json!({"publisher": self.bench.name, "devices": devices}), Vec::new()))
            }
            "attach" => self.attach(connection, client, &instance).map(|value| (value, Vec::new())),
            // A core using a device announced here (its client type at its own server) learns from
            // this connection when it is withdrawn, or that this computer is gone (silence).
            "watch" => {
                let mut own = lock(&self.own);
                let (instance, watched) = own.iter_mut().find(|(_, own)| own.type_id == params["type_id"].as_str().unwrap_or_default()
                        && own.parameters["port"] == params["port"])
                    .ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{} announces no {} at port {}", self.bench.name, params["type_id"], params["port"])))?;
                if !watched.watchers.contains(&connection) { watched.watchers.push(connection); }
                Ok((json!({"instance": instance, "publisher": self.bench.name}), Vec::new()))
            }
            "detach" => {
                let released = {
                    let mut offers = lock(&self.offers);
                    match offers.get_mut(&instance) {
                        Some(offer) if offer.attached.as_ref().is_some_and(|(held, _)| *held == connection) => { offer.attached = None; Some(offer.descriptor.clone()) }
                        _ => None,
                    }
                };
                if let Some(descriptor) = released {
                    self.make_safe(&instance, &descriptor, &format!("{client} closed it"));
                    self.note(&instance, &format!("FABRIC DETACH device={instance} client={client}"));
                }
                Ok((json!({}), Vec::new()))
            }
            op if INSTANCE_OPS.contains(&op) => {
                let deadline = message["deadline_ms"].as_u64().map(Duration::from_millis).unwrap_or(Duration::from_secs(120)).min(LONGEST_REQUEST);
                self.instance_op(connection, client, &instance, op, params, lends, deadline)
            }
            other => Err(CallError::new(ErrorKind::UnknownOp, format!("the bench fabric has no operation {other:?}"))),
        }
    }

    fn attach(&self, connection: u64, client: &str, instance: &str) -> Result<Value, CallError> {
        let (descriptor, role, takeover) = {
            let mut offers = lock(&self.offers);
            let offer = offers.get_mut(instance)
                .ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{} publishes no device {instance:?}", self.bench.name)))?;
            let takeover = match &offer.attached {
                Some((held, holder)) if *held != connection && holder != client =>
                    return Err(CallError::new(ErrorKind::Refused, format!("{instance} on {} is in use by {holder}", self.bench.name))),
                Some((held, _)) => *held != connection,
                None => false,
            };
            offer.attached = Some((connection, client.to_owned()));
            (offer.descriptor.clone(), offer.role.clone(), takeover)
        };
        // The same core again on a new connection: the old one broke, and what
        // the device drove then is not known to it -- safe before it goes on.
        if takeover { self.make_safe(instance, &descriptor, "its user reconnected"); }
        self.note(instance, &format!("FABRIC ATTACH device={instance} client={client}"));
        Ok(json!({"descriptor": serde_json::to_value(&*descriptor).unwrap_or(Value::Null), "role": role,
                  "tunable": self.host.tunable(instance), "publisher": self.bench.name}))
    }

    #[allow(clippy::too_many_arguments)]
    fn instance_op(&self, connection: u64, client: &str, instance: &str, op: &str, params: Value, lends: Vec<Arc<Lend>>, deadline: Duration)
                   -> Result<(Value, Vec<Arc<Lend>>), CallError> {
        {
            let offers = lock(&self.offers);
            let offer = offers.get(instance).ok_or_else(|| CallError::lost(format!("{instance} is no longer published on {}", self.bench.name)))?;
            if offer.attached.as_ref().is_none_or(|(held, _)| *held != connection) {
                return Err(CallError::new(ErrorKind::NotClaimed, format!("attach {instance} before using it")));
            }
        }
        *lock(&self.in_flight).entry(instance.to_owned()).or_default() += 1;
        let answer = self.host.serve(instance, op, params.clone(), lends, deadline);
        {
            let mut busy = lock(&self.in_flight);
            if let Some(count) = busy.get_mut(instance) { *count = count.saturating_sub(1); }
            self.settled.notify_all();
        }
        if op == "tune" || op == "command" {
            let what = if op == "tune" { format!("field={} value={}", params["field"], params["value"]) } else { format!("command={}", params["name"]) };
            self.note(instance, &format!("FABRIC {} device={instance} {what} client={client}{}", op.to_uppercase(),
                match &answer { Ok(reply) => format!(" effective={}", reply.value), Err(error) => format!(" refused={:?}", error.to_string()) }));
        }
        answer.map(|reply| (reply.value.clone(), reply.lends().to_vec()))
    }
}

/// One using core: handshake, then requests (each on its own thread, so a
/// long call does not hold the cancel behind it) until it closes or falls silent.
fn connection(shared: Arc<Shared>, mut stream: TcpStream, id: u64) {
    let _ = stream.set_nodelay(true);
    let peer = stream.peer_addr().map(|address| address.to_string()).unwrap_or_default();
    let client = match server_handshake(&mut stream, &shared.bench, shared.host.registry()) {
        Ok(client) => client,
        Err(error) => { eprintln!("{}", json!({"type": "fabric-refused", "peer": peer, "error": error})); return; }
    };
    let Ok(writer) = stream.try_clone().map(|stream| Arc::new(Mutex::new(stream))) else { return };
    lock(&shared.peers).insert(id, writer.clone());
    let (beating, beat_writer) = (shared.clone(), writer.clone());
    let _ = std::thread::Builder::new().name("zlc-fabric-beat".into()).spawn(move || {
        while lock(&beating.peers).contains_key(&id) && !beating.closing.load(Ordering::Acquire) {
            if send(&mut lock(&beat_writer), &json!({"t": "beat"})).is_err() { break; }
            std::thread::sleep(BEAT);
        }
    });
    let _ = stream.set_read_timeout(Some(SILENCE));
    let reason = loop {
        match read_frame(&mut stream, shared.host.registry()) {
            Ok(None) => break "the user closed the connection".to_owned(),
            Ok(Some((message, lends))) => {
                if message["t"] != "req" { continue; }
                let (shared, writer, client) = (shared.clone(), writer.clone(), client.clone());
                let _ = std::thread::Builder::new().name("zlc-fabric-request".into()).spawn(move || {
                    let reply = match shared.handle(id, &client, &message, lends) {
                        Ok((value, blocks)) => write_frame(&mut *lock(&writer), &json!({"t": "res", "id": message["id"], "ok": value}), &blocks),
                        Err(error) => send(&mut lock(&writer), &json!({"t": "res", "id": message["id"], "error": error.wire()})),
                    };
                    if reply.is_err() { let _ = lock(&writer).shutdown(std::net::Shutdown::Both); }
                });
            }
            Err(error) if matches!(error.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) =>
                break format!("the user fell silent for {} s", SILENCE.as_secs()),
            Err(error) => break format!("the connection failed: {error}"),
        }
    };
    lock(&shared.peers).remove(&id);
    for own in lock(&shared.own).values_mut() { own.watchers.retain(|watcher| *watcher != id); }
    let _ = lock(&writer).shutdown(std::net::Shutdown::Both);
    // What this user held stays on offer for its return; what it drove is made safe now.
    let released: Vec<(String, Arc<TypeDescriptor>)> = lock(&shared.offers).iter_mut()
        .filter(|(_, offer)| offer.attached.as_ref().is_some_and(|(held, _)| *held == id))
        .map(|(instance, offer)| { offer.attached = None; (instance.clone(), offer.descriptor.clone()) }).collect();
    for (instance, descriptor) in released {
        shared.note(&instance, &format!("FABRIC LOST client={client} device={instance} reason={reason:?}"));
        shared.make_safe(&instance, &descriptor, &reason);
    }
}
