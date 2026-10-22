//! The bench fabric: one core publishes chosen devices, another core uses
//! them as if they were its own (v2's PC1 <-> PC2 device fabric, rebuilt).
//!
//! v2 had a Python announcer, one generic data plane for tunable knobs and a
//! hand-made server per remaining device (pulse server, SLM server). In v3
//! every device already speaks one protocol (PLUGIN_PROTOCOL), so the fabric
//! carries exactly that protocol between cores: a published device's SPI
//! calls, its tunable ops and its commands, with their arrays as blocks --
//! a camera frame travels as its bytes and lands in a block of the using
//! core, from which its node and Runtime read it without another copy.
//!
//! * Finding: one UDP probe (`zlc-device-fabric?`, port 18859, v2's) answers
//!   which cores publish; `ZLC_FABRIC_PEERS` names cores a broadcast cannot
//!   reach. Scan hardware lists their devices as `remote.device` rows.
//! * Admission, as v2's fabric: no key. A computer is reachable only while
//!   its operator has Remote on for a device (the publisher listens only
//!   then), on a trusted lab network; a connection says its fabric version
//!   and its machine, and either side refuses another version by name.
//! * Holding: a published device is handed over -- the publishing core
//!   refuses its own runs, commands and tunes on it until Remote is off --
//!   and one using core attaches it at a time. The publisher listens only
//!   while something is published (v2: other machines only when Remote is
//!   on) and answers only the instance ops; the using core drives the device
//!   as its own (its Stop all makes it safe through the link).
//! * Failing: both sides beat every 2 s and give a silent peer 6 s. The
//!   publisher drives an abandoned device's outputs safe. The user fails the
//!   requests in flight (never replays one) and redials for 10 s; a link that
//!   does not come back loses the instance as a crashed plugin is lost
//!   (v2's error state; Apply opens it again).
pub mod wire;
pub mod server;
pub mod link;
pub mod discover;

pub use discover::discover;
pub use link::{Attached, Link};
pub use server::Publisher;

/// The wire vocabulary; either side refuses a peer of another version by name.
/// (v2's fabric spoke 2: a v2 announcer is reported, not listed; 3 was a keyed handshake.)
pub const FABRIC_VERSION: u64 = 4;
pub const DEFAULT_PORT: u16 = 18859;
pub const PROBE: &[u8] = b"zlc-device-fabric?";
/// The using core's type for a device another core publishes.
pub const REMOTE_TYPE: &str = "remote.device";

/// This machine on the bench: its name, and this process's session.
#[derive(Clone, Debug)]
pub struct Bench { pub name: String, pub session: String }

impl Bench {
    /// This computer (`COMPUTERNAME`) in a fresh session.
    pub fn new() -> Bench {
        let name = std::env::var("COMPUTERNAME").ok().filter(|name| !name.trim().is_empty()).unwrap_or_else(|| "bench".into()).trim().to_lowercase();
        Bench { name, session: uuid::Uuid::new_v4().simple().to_string()[..8].to_owned() }
    }

    /// Who this core is to a publisher: its machine and this run of it.
    pub fn client(&self) -> String { format!("{}#{}", self.name, self.session) }
}

impl Default for Bench { fn default() -> Self { Self::new() } }

/// `ZLC_FABRIC_PEERS`: comma-separated `host[:port]` a broadcast cannot reach.
pub fn named_peers() -> Vec<String> {
    std::env::var("ZLC_FABRIC_PEERS").unwrap_or_default().split(',').map(str::trim).filter(|peer| !peer.is_empty()).map(str::to_owned).collect()
}

/// `ZLC_FABRIC_PORT`, else v2's 18859.
pub fn port() -> u16 { std::env::var("ZLC_FABRIC_PORT").ok().and_then(|text| text.parse().ok()).unwrap_or(DEFAULT_PORT) }

/// Where a publisher listens: `ZLC_FABRIC_BIND` (one interface, or 127.0.0.1 for a bench on one machine), else every interface.
pub fn bind_address() -> String { std::env::var("ZLC_FABRIC_BIND").ok().filter(|text| !text.trim().is_empty()).unwrap_or_else(|| "0.0.0.0".into()) }

/// The descriptor of the using core's `remote.device` rows (the workbench's form).
pub fn remote_type() -> serde_json::Value {
    serde_json::json!({
        "type_id": REMOTE_TYPE, "kind": "device", "family": "remote", "label": "Device on another bench computer",
        "description": "A device another ZLC computer publishes on the bench fabric (its Device Manager row's Remote switch); \
                        Scan hardware finds it. It works here as it does there.",
        "module": "zlc-fabric (native)", "capabilities": [], "physical_outputs": [], "commands": [],
        "parameters": [
            {"name": "host", "type": "str", "label": "Fabric host", "default": "", "required": true},
            {"name": "port", "type": "int", "label": "Fabric port", "default": DEFAULT_PORT, "minimum": 1, "maximum": 65535},
            {"name": "instance_id", "type": "str", "label": "Published instance", "default": "", "required": true},
        ],
        "addable": false, "queries": ["discover"], "identity": "fabric:{host}:{port}/{instance_id}"})
}
