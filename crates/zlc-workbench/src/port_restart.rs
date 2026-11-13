//! A serial port its driver refused, and the operator's way out of it.
//!
//! A USB-serial adapter (CH340, CP210x, FTDI, PL2303) can get stuck: its COM port is listed and
//! Device Manager says it works, yet every open fails at CreateFile with ERROR_INVALID_FUNCTION or
//! ERROR_GEN_FAILURE -- the class the one shared serial-open path calls `driver refused`
//! (`zlc_pulse::serial::PortFault::Refused`, `zlc_atom.devices.SERIAL_FAULTS`). Restarting only the
//! device node that owns the port clears it. Each open or resolve of a row on this computer carries
//! the serial ports that would not open on its way ([`SerialFault`], `Event::SerialFaults`); a row
//! that met one refused offers `restart_serial_port`: the host restarts that node (Windows asks
//! the operator to allow it), and the row then opens again as Apply opens it.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::apparatus::{LiveState, Settling, say_device};
use crate::logic::in_use;
use crate::{Event, Host, Ticket, Workbench};

/// The fault class of a serial port whose driver refused the open (`zlc_pulse::serial::PortFault::Refused`'s
/// words, Python's `SERIAL_FAULTS[1]`): the port exists, and its device node is stuck.
pub const REFUSED: &str = "driver refused";

/// A serial port an open or a search met that would not open, as the shared serial-open path
/// classes it (`zlc_pulse::serial::OpenFailure::fact`, `zlc_plugin.wire.serial_fault`): the port,
/// its fault class's words, the OS error code.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerialFault {
    pub port: String,
    pub fault: String,
    #[serde(default)]
    pub code: Option<u32>,
}

impl SerialFault {
    /// The faults an answer or an error lists (`serial_faults`, PLUGIN_PROTOCOL §7), each that reads as one.
    pub fn listed(faults: Option<&Value>) -> Vec<SerialFault> {
        faults.and_then(Value::as_array).into_iter().flatten().filter_map(|fact| serde_json::from_value(fact.clone()).ok()).collect()
    }

    /// Its port as the core names one (`serial:COM3`): upper-case, without `\\.\`.
    fn port_name(&self) -> String { self.port.trim().trim_start_matches(r"\\.\").to_uppercase() }
}

/// What restarting the device node that owns a serial port came to ([`crate::Plugins::restart_port`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PortRestart {
    /// That node (its device instance id) restarted, and the port is there again.
    Restarted { node: String },
    /// The operator declined Windows' confirmation: nothing was restarted.
    Declined { node: String },
    /// No present device has that port any more: there is nothing to restart.
    Absent,
    /// The restart was tried and did not succeed: pnputil's words, or why it could not run.
    Failed { node: String, words: String },
}

/// The restart under way: of which port, since when (its row is its ticket's).
pub(crate) struct Restarting { pub port: String, pub asked: std::time::Instant }

impl Event {
    /// `event`, the outcome of an open or a resolve of `instance`, with the serial faults met on its
    /// way; `event` alone when it met none.
    pub fn met_serial_faults(instance: &str, faults: Vec<SerialFault>, event: Event) -> Event {
        if faults.is_empty() { event } else { Event::SerialFaults { instance: instance.to_owned(), faults, event: Box::new(event) } }
    }
}

impl Workbench {
    /// Serial faults an open or a resolve of row `id` met, kept with its open under way (said
    /// before that open's outcome is: a resolve's go on with the open of what it found).
    pub(crate) fn serial_faults_met(&mut self, id: &str, faults: Vec<SerialFault>) {
        let Some(live) = self.devices.get_mut(id).filter(|live| live.state == LiveState::Opening) else { return };
        for fault in faults { if !live.serial_faults.contains(&fault) { live.serial_faults.push(fault); } }
    }

    /// The ports row `id`'s open or search met refused by their driver, once each, while it stands
    /// failed or open: each is its to restart unless another row binds it (a search never opens a
    /// bound port; a fault on one, met beside this open in the same plugin process, is that row's).
    pub(crate) fn refused_ports(&self, id: &str) -> Vec<String> {
        let Some(live) = self.devices.get(id).filter(|live| matches!(live.state, LiveState::Failed | LiveState::Open) && !live.close_failed) else { return Vec::new() };
        let others: Vec<String> = self.bound_resources(Some(id)).into_iter().map(|(resource, _)| resource).collect();
        let mut ports: Vec<String> = Vec::new();
        for port in live.serial_faults.iter().filter(|fault| fault.fault == REFUSED).map(SerialFault::port_name) {
            if !ports.contains(&port) && !others.contains(&format!("serial:{port}")) { ports.push(port); }
        }
        ports
    }

    /// One restart at a time, and no Init or Apply beside it: the row it opens again is the restart's.
    pub(crate) fn not_restarting(&self) -> Result<(), String> {
        match &self.restarting {
            Some(restarting) => Err(format!("{}'s USB device is being restarted; try again once that is done", restarting.port)),
            None => Ok(()),
        }
    }

    /// `restart_serial_port`: the device node that owns `port` -- which row `id`'s open or search met
    /// refused by its driver -- is restarted on this computer (Windows asks the operator to allow
    /// it), and the row then opens again. Answered once the restart is done (`port_restarted`).
    pub(crate) fn restart_serial_port(&mut self, id: &str, port: &str, request: u64) -> Result<(), String> {
        self.not_restarting()?;
        if self.settling.is_some() { return Err("a device change is under way; restart the USB device once it is done".into()); }
        let port = port.trim().trim_start_matches(r"\\.\").to_uppercase();
        let role = self.device_role(id).to_owned();
        if !self.refused_ports(id).contains(&port) {
            return Err(format!("{role}: {port} is not a serial port its open or search found refused by its driver"));
        }
        // The row opens again once its port's device restarted: not while something outside the workbench drives it.
        if let Some(user) = self.device_use(id).outside { return Err(format!("{role}: {}", in_use(&[user]))); }
        self.restarting = Some(Restarting { port: port.clone(), asked: std::time::Instant::now() });
        let ticket = self.ticket(Ticket::Restart { device: id.to_owned(), port: port.clone(), request });
        self.log_device(id, &format!("restarting the USB device of {port}, which its driver refused (Windows asks to allow it)"));
        self.plugins.restart_port(ticket, &port);
        Ok(())
    }

    /// The restart asked for row `id`'s `port` answered: said in the journal, the row's log and the
    /// Devices line; restarted, the row opens again as Apply opens it (failed: opened; open: closed
    /// and opened, its runs stopped first), and the line says how that went once it settles.
    pub(crate) fn port_restarted(&mut self, id: &str, port: &str, request: u64, result: Result<Value, String>, host: &mut dyn Host) {
        let asked = self.restarting.take().map(|restarting| restarting.asked);
        let outcome = result.and_then(|answer| serde_json::from_value::<PortRestart>(answer).map_err(|error| format!("its answer: {error}")))
            .unwrap_or_else(|words| PortRestart::Failed { node: String::new(), words });
        let (said, severity) = match &outcome {
            PortRestart::Restarted { node } => (format!("{port}'s USB device ({node}) restarted"), "task"),
            PortRestart::Declined { .. } => (format!("{port}'s USB device was not restarted: the Windows confirmation was declined"), "warning"),
            PortRestart::Absent => (format!("{port} is no longer on this PC: nothing was restarted"), "warning"),
            PortRestart::Failed { words, .. } => (format!("{port}'s USB device could not be restarted: {words}"), "error"),
        };
        if let Some(live) = self.devices.get(id) {
            let mut facts = json!({"port": port, "ms": asked.map(|asked| asked.elapsed().as_secs_f64() * 1e3)});
            if let (Some(facts), Value::Object(outcome)) = (facts.as_object_mut(), json!(outcome)) { facts.extend(outcome); }
            say_device("device_port_restart", &live.entry, facts);
        }
        self.log_device(id, &said);
        self.replies.push((request, Ok(json!({"device": id, "port": port, "restart": outcome}))));
        if matches!(outcome, PortRestart::Restarted { .. }) && self.open_again(id, port, host) { return; }
        self.report = (said, severity);
    }

    /// Row `id` opens again after its port's device restarted, as Apply opens it: one that failed
    /// is opened, one that is open is closed and opened. False when it is neither (shut down or
    /// taken out of the apparatus meanwhile): nothing is opened.
    fn open_again(&mut self, id: &str, port: &str, host: &mut dyn Host) -> bool {
        let Some(entry) = self.apparatus.draft.iter().find(|entry| entry.instance_id == id).cloned() else { return false };
        let (retrying, reopening) = match self.devices.get(id).map(|live| (live.state, live.close_failed)) {
            Some((LiveState::Failed, false)) => (vec![id], Vec::new()),
            Some((LiveState::Open, false)) => (Vec::new(), vec![id]),
            _ => return false,
        };
        zlc_core::journal::lifecycle("workbench", "devices_apply", json!({"kind": "restart", "port": port, "retrying": retrying, "reopening": reopening}));
        if retrying.is_empty() {
            self.close_instances(vec![(id.to_owned(), Some(entry))], host);
        } else {
            self.devices.remove(id);
            self.open_entry(entry);
        }
        self.found.clear();
        self.settling = Some(Settling::Restart { device: id.to_owned(), port: port.to_owned() });
        self.settling_since = Some(std::time::Instant::now());
        self.settle();
        self.bench_changed();
        true
    }
}
