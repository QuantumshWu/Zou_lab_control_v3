//! This computer on the bench fabric (crates/zlc-fabric): the devices it
//! publishes (the Device Manager's Remote switch) and the devices of other
//! computers it opens as `remote.device` rows. No key: as v2's fabric, a
//! computer is reachable only while Remote is on for one of its devices.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use zlc_fabric::{Link, Publisher};
use zlc_plugin_host::{CallError, Endpoint, ErrorKind, PluginHost, Registry};

/// A fabric line for a device's log: (instance, line).
pub type Note = Arc<dyn Fn(&str, &str) + Send + Sync>;

struct Inner {
    bench: zlc_fabric::Bench,
    /// Shared out of the lock: a withdrawal waits on the device, the state never does.
    publisher: Mutex<Option<Arc<Publisher>>>,
    note: Note,
    /// Clients of another computer's own server, watched on the fabric: instance -> link.
    watches: Mutex<BTreeMap<String, Link>>,
}

#[derive(Clone)]
pub struct Bench(Arc<Inner>);

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> { mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

impl Bench {
    pub fn new(note: Note) -> Bench {
        Bench(Arc::new(Inner { bench: zlc_fabric::Bench::new(), publisher: Mutex::new(None), note, watches: Mutex::default() }))
    }

    fn bench(&self) -> zlc_fabric::Bench { self.0.bench.clone() }

    /// Open a `remote.device` row: dial, attach, and admit it into `host` as
    /// `instance`; its working point carries the publisher's type.
    pub fn open(&self, host: &PluginHost, instance: &str, parameters: &Value) -> Result<Value, CallError> {
        let invalid = |message: &str| CallError::new(ErrorKind::Invalid, message);
        let address = parameters["host"].as_str().map(str::trim).filter(|host| !host.is_empty()).ok_or_else(|| invalid("Fabric host is required"))?;
        let port = parameters["port"].as_u64().and_then(|port| u16::try_from(port).ok()).unwrap_or(zlc_fabric::DEFAULT_PORT);
        let published = parameters["instance_id"].as_str().map(str::trim).filter(|id| !id.is_empty()).ok_or_else(|| invalid("Published instance is required"))?;
        let (link, attached) = Link::open(address, port, published, self.bench(), host.registry().clone()).map_err(|error| CallError::new(ErrorKind::Failed, error))?;
        let point = json!({"tunable": attached.tunable, "descriptor": crate::workbench_host::descriptor(&attached.descriptor),
            "remote": {"host": address, "port": port, "instance_id": published, "publisher": attached.publisher, "role": attached.role,
                       "type_id": attached.descriptor.type_id, "label": attached.descriptor.label}});
        // What it says for itself (an SLM Editor's draft, its log) is this core's to hear, as a local device's.
        let (told, local) = (host.clone(), instance.to_owned());
        link.on_event(move |op, params, lends| told.remote_event(&local, op, params, lends));
        if let Err(error) = host.attach(instance, attached.descriptor, Arc::new(link.clone()), attached.tunable) {
            zlc_plugin_host::Endpoint::close(&link);
            return Err(error);
        }
        let (lost, name) = (host.clone(), instance.to_owned());
        let (noted, again) = (self.0.note.clone(), instance.to_owned());
        link.on_failure(move |reason| noted(&name, &format!("FABRIC BROKE {reason}")), move |reason| lost.lost(&again, reason));
        (self.0.note)(instance, &format!("FABRIC LINKED {} on {} ({}:{port})", published, attached.publisher, address));
        Ok(point)
    }

    /// Watch `instance`, a client of another computer's own server (`type_id`, announced
    /// there: the pulse board as sequencer.hardware, the SLM head as slm.hamamatsu_x15213):
    /// when a bench computer at its `host` announces that server, the fabric says when that
    /// computer withdraws it or goes away -- the link, beats and words a remote.device's are
    /// lost by -- and `lost` is told why. A server no bench computer announces is not watched.
    pub fn watch(&self, registry: Arc<Registry>, instance: &str, type_id: &str, parameters: &Value, lost: impl Fn(&str) + Send + Sync + 'static) {
        let Some(host) = parameters["host"].as_str().map(str::trim).filter(|host| !host.is_empty()) else { return };
        // Where a fabric listens on that host: a named peer there, else this bench's port.
        let mut ports: Vec<u16> = zlc_fabric::named_peers().iter().filter_map(|peer| match peer.rsplit_once(':') {
            Some((name, port)) => (name == host).then(|| port.parse().ok()).flatten(),
            None => (peer == host).then_some(zlc_fabric::DEFAULT_PORT),
        }).collect();
        ports.push(zlc_fabric::port());
        let Some(link) = ports.into_iter().find_map(|port| Link::watch(host, port, type_id, &parameters["port"], self.bench(), registry.clone()).ok()) else { return };
        let (noted, name) = (self.0.note.clone(), instance.to_owned());
        link.on_failure(move |reason| noted(&name, &format!("FABRIC BROKE {reason}")), lost);
        if let Some(old) = lock(&self.0.watches).insert(instance.to_owned(), link) { old.close(); }
    }

    /// The device closed here: nothing to watch for it any more.
    pub fn unwatch(&self, instance: &str) {
        let link = lock(&self.0.watches).remove(instance);
        if let Some(link) = link { link.close(); }
    }

    /// Remote on or off for an open device of this computer.
    pub fn publish(&self, host: &PluginHost, instance: &str, role: &str, published: bool) -> Result<Value, String> {
        if published {
            let open = self.publisher(host)?;
            open.publish(instance, role)?;
            Ok(json!({"port": open.port(), "name": self.bench().name}))
        } else {
            let open = lock(&self.0.publisher).clone().ok_or_else(|| format!("{instance} is not published"))?;
            open.withdraw(instance)?;
            self.stop_if_idle(&open);
            Ok(json!({}))
        }
    }

    /// Announce a device that has a server of its own (the pulse board on this
    /// computer): the other computers' scan offers `type_id` at `parameters`
    /// with this computer's address as `host`.
    pub fn publish_own(&self, host: &PluginHost, instance: &str, role: &str, type_id: &str, label: &str, parameters: Value) -> Result<Value, String> {
        let open = self.publisher(host)?;
        open.publish_own(instance, role, type_id, label, parameters);
        Ok(json!({"port": open.port(), "name": self.bench().name}))
    }

    /// The running publisher; listening starts with the first publication and stops with the last.
    fn publisher(&self, host: &PluginHost) -> Result<Arc<Publisher>, String> {
        let mut publisher = lock(&self.0.publisher);
        if let Some(open) = publisher.as_ref() { return Ok(open.clone()); }
        let started = Arc::new(Publisher::start(host.clone(), self.bench(), &zlc_fabric::bind_address(), zlc_fabric::port())?);
        let note = self.0.note.clone();
        started.on_note(move |instance, line| note(instance, line));
        *publisher = Some(started.clone());
        Ok(started)
    }

    /// Nothing on offer any more: stop listening (other machines find nothing here).
    fn stop_if_idle(&self, open: &Arc<Publisher>) {
        let mut publisher = lock(&self.0.publisher);
        if publisher.as_ref().is_some_and(|current| Arc::ptr_eq(current, open)) && open.offers().is_empty() { *publisher = None; }
    }

    /// Take a device back before it closes (Shutdown withdraws first, as v2's did).
    pub fn withdraw_if_published(&self, instance: &str) {
        let open = lock(&self.0.publisher).clone();
        if let Some(open) = open.filter(|open| open.is_published(instance)) {
            let _ = open.withdraw(instance);
            self.stop_if_idle(&open);
        }
    }

    /// Scan hardware's bench part: `{found, notes}` as the workbench reads a scan.
    pub fn discover(&self, host: &PluginHost) -> Value {
        let (found, notes) = zlc_fabric::discover(&self.bench(), zlc_fabric::port(), &zlc_fabric::named_peers(), host.registry(), Duration::from_secs(1));
        json!({"found": found, "notes": notes})
    }
}
