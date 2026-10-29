//! The using core's end: one published device, reached as an [`Endpoint`] of
//! this core's plugin host. A broken connection fails the requests in flight
//! (none is sent again: it may have acted), then the link redials for
//! [`RELINK_WINDOW`] while new requests wait their deadline for it; past that
//! the instance is lost.
use std::collections::{HashMap, VecDeque};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, mpsc};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_plugin_host::{CallError, Endpoint, ErrorKind, Lend, Registry, Reply, TypeDescriptor};

use crate::server::{BEAT, SILENCE};
use crate::wire::{client_handshake, read_frame, send, write_frame};
use crate::Bench;

/// How long a broken link redials before its device is lost.
pub const RELINK_WINDOW: Duration = Duration::from_secs(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// A request is answered within its deadline plus this, or it timed out.
const ANSWER_GRACE: Duration = Duration::from_secs(5);
const LOG_LINES: usize = 400;

/// What the publisher says the device is.
pub struct Attached {
    pub descriptor: TypeDescriptor, pub role: String, pub tunable: bool,
    /// None identifies an older publisher that does not advertise snapshots.
    pub snapshot: Option<bool>, pub reads: Vec<String>, pub publisher: String,
}

type Answer = Result<(Value, Vec<Arc<Lend>>), CallError>;
type Hook = Box<dyn Fn(&str) + Send + Sync>;
/// What the device says for itself: the event's op, params and the blocks its arrays landed in.
type Told = Box<dyn Fn(&str, &Value, Vec<Arc<Lend>>) + Send + Sync>;

enum Status { Up, Relinking, Lost(String), Closed }

struct State { status: Status, writer: Option<Arc<Mutex<TcpStream>>>, generation: u64 }

struct Shared {
    host: String,
    port: u16,
    instance: String,
    /// What each connection asks first: `attach` (a device served here) or `watch` (one announced here).
    first: (&'static str, Value),
    bench: Bench,
    registry: Arc<Registry>,
    state: Mutex<State>,
    changed: Condvar,
    pending: Mutex<HashMap<u64, mpsc::Sender<Answer>>>,
    next: AtomicU64,
    log: Mutex<VecDeque<String>>,
    broke: Mutex<Option<Hook>>,
    lost: Mutex<Option<Hook>>,
    told: Mutex<Option<Told>>,
}

/// One published device of another core, attached.
#[derive(Clone)]
pub struct Link(Arc<Shared>);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> { mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

fn dial(host: &str, port: u16) -> Result<TcpStream, String> {
    let addresses: Vec<_> = (host, port).to_socket_addrs().map_err(|error| format!("{host}: {error}"))?.collect();
    let mut last = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, CONNECT_TIMEOUT) {
            Ok(stream) => { let _ = stream.set_nodelay(true); return Ok(stream); }
            Err(error) => last = Some(error),
        }
    }
    Err(match last {
        Some(error) if error.kind() == std::io::ErrorKind::ConnectionRefused =>
            format!("{host}:{port} is up but publishes nothing (no device there has Remote on)"),
        Some(error) => format!("could not reach {host}:{port}: {error} (a firewall on that machine may hold the port)"),
        None => format!("{host} has no address"),
    })
}

impl Link {
    fn new(host: &str, port: u16, instance: &str, first: (&'static str, Value), bench: Bench, registry: Arc<Registry>) -> Link {
        Link(Arc::new(Shared { host: host.trim().to_owned(), port, instance: instance.trim().to_owned(), first, bench, registry,
            state: Mutex::new(State { status: Status::Relinking, writer: None, generation: 0 }), changed: Condvar::new(),
            pending: Mutex::default(), next: AtomicU64::new(1), log: Mutex::default(), broke: Mutex::new(None), lost: Mutex::new(None), told: Mutex::new(None) }))
    }

    /// Dial `host:port` and attach its published `instance`.
    pub fn open(host: &str, port: u16, instance: &str, bench: Bench, registry: Arc<Registry>) -> Result<(Link, Attached), String> {
        let link = Link::new(host, port, instance, ("attach", json!({"instance": instance.trim()})), bench, registry);
        let attached = link.connect().and_then(|value| {
            let descriptor: TypeDescriptor = serde_json::from_value(value["descriptor"].clone()).map_err(|error| format!("the publisher's descriptor: {error}"))?;
            Ok(Attached { descriptor, role: value["role"].as_str().unwrap_or("").to_owned(), tunable: value["tunable"] == true,
                          snapshot: value["snapshot"].as_bool(),
                          reads: serde_json::from_value(value["reads"].clone()).map_err(|error| format!("the publisher's reads: {error}"))?,
                          publisher: value["publisher"].as_str().unwrap_or("").to_owned() })
        });
        match attached {
            Ok(attached) => { link.note(format!("FABRIC LINKED device={} at {}:{} publisher={}", link.0.instance, link.0.host, link.0.port, attached.publisher)); Ok((link, attached)) }
            Err(error) => { link.shut(Status::Closed); Err(error) }
        }
    }

    /// Dial `host:port` and watch what it announces as `type_id` at its own server's `server_port`:
    /// the link breaks, relinks and is lost exactly as a served device's does (withdrawn, silent).
    pub fn watch(host: &str, port: u16, type_id: &str, server_port: &Value, bench: Bench, registry: Arc<Registry>) -> Result<Link, String> {
        let link = Link::new(host, port, &format!("{type_id} at {}:{server_port}", host.trim()), ("watch", json!({"type_id": type_id, "port": server_port})), bench, registry);
        match link.connect() {
            Ok(answer) => { link.note(format!("FABRIC WATCHING {} publisher={}", answer["instance"], answer["publisher"])); Ok(link) }
            Err(error) => { link.shut(Status::Closed); Err(error) }
        }
    }

    /// What to do when the link breaks (it redials) and when it is lost for good.
    pub fn on_failure(&self, broke: impl Fn(&str) + Send + Sync + 'static, lost: impl Fn(&str) + Send + Sync + 'static) {
        *lock(&self.0.broke) = Some(Box::new(broke));
        *lock(&self.0.lost) = Some(Box::new(lost));
    }

    /// What the device says for itself on the publishing core (its `device.notice`
    /// and `device.log`), as the plugin protocol's events: this core hears them
    /// as from a device of its own.
    pub fn on_event(&self, told: impl Fn(&str, &Value, Vec<Arc<Lend>>) + Send + Sync + 'static) { *lock(&self.0.told) = Some(Box::new(told)); }

    /// Where it is reached, for the operator.
    pub fn address(&self) -> String { format!("{}:{}", self.0.host, self.0.port) }

    fn note(&self, line: String) {
        let mut log = lock(&self.0.log);
        if log.len() == LOG_LINES { log.pop_front(); }
        log.push_back(line);
    }

    /// One connection: handshake, its reader and beat, then its first request (attach or watch).
    fn connect(&self) -> Result<Value, String> {
        let mut stream = dial(&self.0.host, self.0.port)?;
        client_handshake(&mut stream, &self.0.bench, &self.0.registry)?;
        let writer = Arc::new(Mutex::new(stream.try_clone().map_err(|error| error.to_string())?));
        let generation = {
            let mut state = lock(&self.0.state);
            if matches!(state.status, Status::Closed | Status::Lost(_)) { return Err("the link was closed".into()); }
            state.generation += 1;
            state.writer = Some(writer.clone());
            state.generation
        };
        let reader = self.clone();
        std::thread::Builder::new().name("zlc-fabric-link".into()).spawn(move || reader.read(stream, generation)).map_err(|error| error.to_string())?;
        let (beating, beat_writer) = (self.clone(), writer.clone());
        let _ = std::thread::Builder::new().name("zlc-fabric-link-beat".into()).spawn(move || {
            while lock(&beating.0.state).generation == generation {
                if send(&mut lock(&beat_writer), &json!({"t": "beat"})).is_err() { break; }
                std::thread::sleep(BEAT);
            }
        });
        let (op, params) = &self.0.first;
        let answer = self.exchange(generation, op, params.clone(), Vec::new(), Duration::from_secs(10)).map(|(value, _)| value).map_err(|error| error.message);
        let mut state = lock(&self.0.state);
        match &answer {
            Ok(_) if state.generation == generation => { state.status = Status::Up; self.0.changed.notify_all(); }
            _ => {
                if state.generation == generation { state.writer = None; state.generation += 1; }
                let _ = lock(&writer).shutdown(std::net::Shutdown::Both);
            }
        }
        answer
    }

    /// Frames from the publisher until the connection ends.
    fn read(&self, mut stream: TcpStream, generation: u64) {
        let _ = stream.set_read_timeout(Some(SILENCE));
        let reason = loop {
            match read_frame(&mut stream, &self.0.registry) {
                Ok(None) => break format!("{} closed the connection", self.0.host),
                Ok(Some((message, lends))) => match message["t"].as_str() {
                    Some("res") => {
                        let answer = match message.get("error") { Some(error) => Err(CallError::from_wire(error)), None => Ok((message["ok"].clone(), lends)) };
                        if let Some(waiter) = message["id"].as_u64().and_then(|id| lock(&self.0.pending).remove(&id)) { let _ = waiter.send(answer); }
                    }
                    Some("evt") if message["op"] == "withdrawn" => {
                        let reason = message["p"]["reason"].as_str().unwrap_or("the publisher withdrew it").to_owned();
                        let _ = stream.shutdown(std::net::Shutdown::Both);
                        self.lose(&reason);
                        return;
                    }
                    Some("evt") => if let Some(told) = lock(&self.0.told).as_ref() { told(message["op"].as_str().unwrap_or(""), &message["p"], lends); },
                    _ => {}
                },
                Err(error) if matches!(error.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) =>
                    break format!("{} fell silent for {} s", self.0.host, SILENCE.as_secs()),
                Err(error) => break format!("the connection to {} failed: {error}", self.0.host),
            }
        };
        let _ = stream.shutdown(std::net::Shutdown::Both);
        self.broke(generation, &reason);
    }

    /// Requests in flight cannot be answered any more.
    fn fail_pending(&self, reason: &str) {
        for (_, waiter) in lock(&self.0.pending).drain() { let _ = waiter.send(Err(CallError::lost(reason.to_owned()))); }
    }

    fn broke(&self, generation: u64, reason: &str) {
        {
            let mut state = lock(&self.0.state);
            if state.generation != generation || !matches!(state.status, Status::Up | Status::Relinking) { return; }
            state.generation += 1;
            state.writer = None;
            state.status = Status::Relinking;
            self.0.changed.notify_all();
        }
        let reason = format!("the bench link for {} broke: {reason}", self.0.instance);
        self.fail_pending(&reason);
        self.note(format!("FABRIC BROKE device={} reason={reason:?}", self.0.instance));
        // Redialling first: anything the hook sets off through this link waits for it.
        let (relinking, cause) = (self.clone(), reason.clone());
        let _ = std::thread::Builder::new().name("zlc-fabric-relink".into()).spawn(move || relinking.relink(cause));
        if let Some(hook) = lock(&self.0.broke).as_ref() { hook(&reason); }
    }

    fn relink(&self, reason: String) {
        let until = Instant::now() + RELINK_WINDOW;
        let mut pause = Duration::from_millis(250);
        let mut last = reason;
        while Instant::now() < until {
            if !matches!(lock(&self.0.state).status, Status::Relinking) { return; }
            match self.connect() {
                Ok(_) => { self.note(format!("FABRIC RELINKED device={} at {}:{}", self.0.instance, self.0.host, self.0.port)); return; }
                Err(error) => last = error,
            }
            std::thread::sleep(pause.min(until.saturating_duration_since(Instant::now())));
            pause = (pause * 2).min(Duration::from_secs(2));
        }
        self.lose(&format!("the bench link for {} did not come back within {} s: {last}", self.0.instance, RELINK_WINDOW.as_secs()));
    }

    fn lose(&self, reason: &str) {
        if !self.shut(Status::Lost(reason.to_owned())) { return; }
        self.note(format!("FABRIC LOST device={} reason={reason:?}", self.0.instance));
        if let Some(hook) = lock(&self.0.lost).as_ref() { hook(reason); }
    }

    /// End the connection with `status`; false when it had ended already.
    fn shut(&self, status: Status) -> bool {
        let writer = {
            let mut state = lock(&self.0.state);
            if matches!(state.status, Status::Lost(_) | Status::Closed) { return false; }
            state.status = status;
            state.generation += 1;
            self.0.changed.notify_all();
            state.writer.take()
        };
        if let Some(writer) = writer { let _ = lock(&writer).shutdown(std::net::Shutdown::Both); }
        self.fail_pending(&format!("the bench link for {} is closed", self.0.instance));
        true
    }

    /// The connection to send on, waiting out a relink until `deadline`.
    fn current(&self, deadline: Instant) -> Result<(u64, Arc<Mutex<TcpStream>>), CallError> {
        let mut state = lock(&self.0.state);
        loop {
            match (&state.status, &state.writer) {
                (Status::Up, Some(writer)) => return Ok((state.generation, writer.clone())),
                (Status::Lost(reason), _) => return Err(CallError::lost(reason.clone())),
                (Status::Closed, _) => return Err(CallError::lost(format!("the bench link for {} is closed", self.0.instance))),
                _ => {}
            }
            let now = Instant::now();
            if now >= deadline { return Err(CallError::new(ErrorKind::Timeout, format!("the bench link for {} is down", self.0.instance))); }
            state = self.0.changed.wait_timeout(state, deadline - now).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    /// One request on connection `generation`.
    fn exchange(&self, generation: u64, op: &str, params: Value, lends: Vec<Arc<Lend>>, deadline: Duration) -> Answer {
        let writer = {
            let state = lock(&self.0.state);
            if state.generation != generation { return Err(CallError::lost("the connection changed")); }
            state.writer.clone().ok_or_else(|| CallError::lost("the connection is down"))?
        };
        self.send_on(generation, &writer, op, params, lends, deadline)
    }

    fn send_on(&self, generation: u64, writer: &Arc<Mutex<TcpStream>>, op: &str, params: Value, lends: Vec<Arc<Lend>>, deadline: Duration) -> Answer {
        struct Counted<'a> { stream: &'a mut TcpStream, bytes: usize }
        impl std::io::Write for Counted<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let count = std::io::Write::write(self.stream, bytes)?;
                self.bytes += count;
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> { std::io::Write::flush(self.stream) }
        }
        let until = Instant::now() + deadline;
        let snapshot = op == "snapshot";
        let timed_out = || CallError::new(ErrorKind::Timeout, format!("snapshot of {} did not answer within {:.1} s", self.0.instance, deadline.as_secs_f64()));
        // Start's metadata cannot wait behind another request's bulk write.
        // An expired snapshot is dropped; it does not cancel the owning link.
        let mut stream = if snapshot {
            loop {
                if Instant::now() >= until { return Err(timed_out()); }
                match writer.try_lock() {
                    Ok(stream) => break stream,
                    Err(std::sync::TryLockError::Poisoned(poisoned)) => break poisoned.into_inner(),
                    Err(std::sync::TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(1)),
                }
            }
        } else { lock(writer) };
        let previous_timeout = if snapshot {
            let previous = stream.write_timeout().map_err(|error| CallError::lost(error.to_string()))?;
            let remaining = until.saturating_duration_since(Instant::now());
            if remaining.is_zero() { return Err(timed_out()); }
            stream.set_write_timeout(Some(remaining)).map_err(|error| CallError::lost(error.to_string()))?;
            previous
        } else { None };
        let id = self.0.next.fetch_add(1, Ordering::Relaxed);
        let (sender, answer) = mpsc::channel();
        lock(&self.0.pending).insert(id, sender);
        let remaining = if snapshot { until.saturating_duration_since(Instant::now()) } else { deadline };
        let message = json!({"t": "req", "id": id, "op": op, "p": params, "deadline_ms": remaining.as_millis() as u64});
        let mut sink = Counted { stream: &mut stream, bytes: 0 };
        let mut written = write_frame(&mut sink, &message, &lends);
        let bytes_written = sink.bytes;
        if snapshot { if let Err(error) = stream.set_write_timeout(previous_timeout) { written = Err(error); } }
        drop(stream);
        if let Err(error) = written {
            lock(&self.0.pending).remove(&id);
            if snapshot && bytes_written == 0 && matches!(error.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) {
                return Err(timed_out());
            }
            // A partial frame cannot be discarded on a shared byte stream.
            // Only that real transport failure follows the normal recovery path.
            self.broke(generation, &format!("sending {op} failed: {error}"));
            return Err(CallError::lost(format!("the bench link for {} broke while sending {op}", self.0.instance)));
        }
        drop(lends);
        let wait = if snapshot { until.saturating_duration_since(Instant::now()) } else { deadline + ANSWER_GRACE };
        match answer.recv_timeout(wait) {
            Ok(answer) => answer,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                lock(&self.0.pending).remove(&id);
                if snapshot { return Err(timed_out()); }
                Err(CallError::new(ErrorKind::Timeout, format!("{} did not answer {op} for {} within {:.0} s", self.0.host, self.0.instance,
                                                               (deadline + ANSWER_GRACE).as_secs_f64())))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(CallError::lost(format!("the bench link for {} ended during {op}", self.0.instance))),
        }
    }
}

impl Endpoint for Link {
    fn request(&self, op: &str, mut params: Value, lends: Vec<Arc<Lend>>, deadline: Duration) -> Result<Reply, CallError> {
        params["instance"] = json!(self.0.instance);
        let until = Instant::now() + deadline;
        let (generation, writer) = self.current(until)?;
        let remaining = if op == "snapshot" { until.saturating_duration_since(Instant::now()) } else { deadline };
        self.send_on(generation, &writer, op, params, lends, remaining).map(|(value, lends)| Reply::new(value, lends))
    }

    fn close(&self) {
        let current = { let state = lock(&self.0.state); (matches!(state.status, Status::Up)).then(|| (state.generation, state.writer.clone())) };
        if let Some((generation, Some(writer))) = current {
            let _ = self.send_on(generation, &writer, "detach", json!({"instance": self.0.instance}), Vec::new(), Duration::from_secs(2));
        }
        self.shut(Status::Closed);
        self.note(format!("FABRIC CLOSED device={}", self.0.instance));
    }

    fn tail(&self, lines: usize) -> Vec<String> {
        let log = lock(&self.0.log);
        log.iter().skip(log.len().saturating_sub(lines)).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_deadline_includes_a_busy_writer_without_closing_or_sending_late() {
        use std::io::Read;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut peer, _) = listener.accept().unwrap();
        let writer = Arc::new(Mutex::new(stream));
        let link = Link::new("127.0.0.1", listener.local_addr().unwrap().port(), "camera",
            ("attach", json!({"instance": "camera"})), Bench::new(), Registry::standalone("snapshot-writer"));
        { let mut state = lock(&link.0.state); state.writer = Some(writer.clone()); state.status = Status::Up; }
        let held = lock(&writer);
        let started = Instant::now();
        assert_eq!(link.request("snapshot", json!({}), Vec::new(), Duration::from_millis(50)).err().unwrap().kind, ErrorKind::Timeout);
        assert!(started.elapsed() < Duration::from_millis(500));
        assert!(matches!(lock(&link.0.state).status, Status::Up));
        assert!(lock(&link.0.pending).is_empty());
        drop(held);
        peer.set_read_timeout(Some(Duration::from_millis(30))).unwrap();
        let error = peer.read(&mut [0u8; 1]).unwrap_err();
        assert!(matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut));
        link.shut(Status::Closed);
    }
}
