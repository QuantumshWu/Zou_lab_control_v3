//! The apparatus: the Device Manager draft (`zlc.installation`) and the live
//! device instances it was applied to. A draft edit touches no device;
//! `apply_devices` opens what the draft adds or changes and closes what it
//! removes, after the runs that hold those devices have stopped.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use zlc_core::units::{DEFAULT_UNITS, NO_PREFIX, Number, format_quantity};

use crate::{Host, Ticket, Workbench, fields, text, workspace};

pub const FORMAT: &str = "zlc.installation";
const ENTRY_KEYS: [&str; 4] = ["instance_id", "role", "type_id", "parameters"];
/// Types in this process group observe the one simulated world.
const SIMULATION_GROUP: &str = "simulation";

#[derive(Clone, Debug, PartialEq)]
pub struct Entry { pub instance_id: String, pub role: String, pub type_id: String, pub parameters: Map<String, Value> }

impl Entry {
    fn tree(&self) -> Value { json!({"instance_id": self.instance_id, "role": self.role, "type_id": self.type_id, "parameters": self.parameters}) }
}

pub struct Apparatus {
    pub path: PathBuf,
    pub simulation: Map<String, Value>,
    pub draft: Vec<Entry>,
    /// The draft as last read or saved (dirty = differs).
    baseline: Value,
    pub discovered: Vec<Entry>,
}

impl Apparatus {
    pub fn empty(path: PathBuf) -> Self {
        let mut apparatus = Self { path, simulation: Map::new(), draft: Vec::new(), baseline: Value::Null, discovered: Vec::new() };
        apparatus.baseline = apparatus.tree();
        apparatus
    }
    pub fn tree(&self) -> Value {
        json!({"format": FORMAT, "simulation": self.simulation, "devices": self.draft.iter().map(Entry::tree).collect::<Vec<_>>()})
    }
    pub fn dirty(&self) -> bool { self.tree() != self.baseline }
    fn entry(&self, id: &str) -> Result<&Entry, String> {
        self.draft.iter().find(|entry| entry.instance_id == id).ok_or_else(|| format!("No device {id:?} in the apparatus"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveState { Opening, Open, Closing, Failed }

/// A device change in progress: v2's busy word while it runs, its line once it settles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Settling { Init, Apply, Close(String), Shutdown }

impl Settling {
    /// v2 DeviceManagerPresenter's word while the device worker runs.
    fn busy(&self) -> String {
        match self {
            Settling::Init => "initializing devices".into(),
            Settling::Apply => "applying device changes".into(),
            Settling::Close(id) => format!("closing {id}"),
            Settling::Shutdown => "shutting down devices".into(),
        }
    }
}

/// A hardware scan in flight: who asked, the types still to answer, what came back.
pub struct Scan { request: u64, asked: Vec<String>, pending: Vec<String>, found: Vec<Entry>, reasons: Vec<(String, Vec<String>)> }

/// One live instance: what it was opened with and what it reports.
pub struct Live {
    pub entry: Entry,
    pub state: LiveState,
    pub error: Option<String>,
    pub working_point: Value,
    pub fields: Option<Value>,
    /// Close once no run holds it; then open this entry, if any.
    pub close_pending: bool,
    pub reopen: Option<Entry>,
    /// The run whose claim the operator accepted tuning risk for.
    pub risk: Option<String>,
    /// Device Control: the spelling the operator reads a field in (field -> unit).
    pub units: BTreeMap<String, String>,
    /// The type it opened as, when that is not its row's (a device another
    /// computer publishes: its working point names the publisher's type).
    pub descriptor: Option<crate::TypeDescriptor>,
    /// Handed to another bench computer (Remote on): nothing here uses it meanwhile.
    pub published: bool,
}

const PUBLISHED: &str = "published on the bench fabric -- withdraw Remote to control it here";

/// One device reading in words (zlc_ui `_readable_value`): a number in the
/// spelling the operator picked for its row exactly as it stands, else in the
/// readable prefix of its own unit; anything else as itself.
fn reading_text(value: &Value, unit: &str, picked: bool) -> String {
    let number = match value {
        Value::Null => return "\u{2014}".into(),
        Value::Bool(flag) => return if *flag { "True" } else { "False" }.into(),
        Value::String(text) => return text.clone(),
        Value::Number(number) => match number.as_i64() { Some(integer) => Number::Int(integer.into()), None => Number::Float(number.as_f64().unwrap_or(f64::NAN)) },
        other => return other.to_string(),
    };
    let unit = if unit.trim().is_empty() { "1" } else { unit.trim() };
    DEFAULT_UNITS.resolve(unit).and_then(|resolved| format_quantity(number, &resolved, picked.then_some(NO_PREFIX)))
        .unwrap_or_else(|_| number.text())
}

/// The spellings a numeric field may be read in, for its unit picker (`[]`: none).
fn unit_choices(unit: &str) -> Vec<Value> {
    DEFAULT_UNITS.choice_tree(unit).into_iter()
        .flat_map(|(trunk, leaves)| leaves.into_iter().map(move |leaf| json!({"value": leaf, "label": leaf, "group": trunk})))
        .collect()
}

/// v2's simulation world schema, used when no simulated type declares its own.
fn simulation_fields() -> Vec<fields::Field> {
    serde_json::from_value(json!([
        // The virtual qCMOS sensor: the whole trap plane the full SLM addresses (zlc_atom simulation world).
        {"name": "image_shape_yx", "type": "pair", "label": "Image shape (y, x)", "default": [372, 460]},
        {"name": "grid_shape_yx", "type": "pair", "label": "Trap grid (y, x)", "default": [5, 7]},
        {"name": "seed", "type": "int", "label": "Seed", "default": 0},
        {"name": "world_profile", "type": "str", "label": "World profile", "default": ""},
    ])).expect("simulation schema")
}

/// zlc_ui.status.counted: "1 device", "2 devices".
fn counted(count: usize, noun: &str) -> String { format!("{count} {noun}{}", if count == 1 { "" } else { "s" }) }

pub(crate) fn file_name(path: &Path) -> String { path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default() }

/// v2 DeviceInstanceConfig `_text`: an id, name or type is non-empty text with no surrounding space.
fn entry_text<'a>(text: Option<&'a str>, key: &str) -> Result<&'a str, String> {
    match text {
        Some(text) if !text.trim().is_empty() && text == text.trim() => Ok(text),
        _ => Err(format!("device {key} must be non-empty text with no surrounding space")),
    }
}

/// v2's one rule for a device id: that text, without `/`. Nothing else is asked of
/// it; the MCP tool names of its commands are spelled from it (`tool_name`).
fn instance_id(id: Option<&str>) -> Result<&str, String> {
    let id = entry_text(id, "instance_id")?;
    if id.contains('/') { Err("device instance_id cannot contain '/'".into()) } else { Ok(id) }
}

/// The device family a plugin module belongs to: the folder under `devices/` (v2
/// UnavailableDeviceTypes.family), else the module itself.
fn module_family(module: &str) -> &str {
    let mut parts = module.split('.');
    parts.by_ref().find(|part| *part == "devices").and_then(|_| parts.next()).unwrap_or(module)
}

impl Workbench {
    fn device_type(&self, type_id: &str) -> Result<&crate::TypeDescriptor, String> {
        self.catalog.get(type_id).filter(|item| item.is_device()).ok_or_else(|| self.not_installed(&format!("No device type {type_id:?} is installed"), type_id))
    }

    /// The plugin families that did not load, one entry per reason with its families in catalogue order.
    fn unavailable_families(&self) -> Vec<(String, Vec<String>)> {
        let mut reasons: Vec<(String, Vec<String>)> = Vec::new();
        for (module, reason) in &self.catalog.errors {
            let family = module_family(module).to_owned();
            match reasons.iter_mut().find(|(known, _)| known == reason) {
                Some((_, families)) => if !families.contains(&family) { families.push(family) },
                None => reasons.push((reason.clone(), vec![family])),
            }
        }
        reasons
    }

    /// `words`, and why: the reason its family did not load (`camera.virtual` is of the camera
    /// family), else the one reason every plugin that did not load gave (no plugin environment).
    fn not_installed(&self, words: &str, type_id: &str) -> String {
        let family = type_id.split('.').next().unwrap_or(type_id);
        let unavailable = self.unavailable_families();
        let mut reasons: Vec<&str> = unavailable.iter().filter(|(_, families)| families.iter().any(|known| known == family)).map(|(reason, _)| reason.as_str()).collect();
        if reasons.is_empty() && unavailable.len() == 1 { reasons.push(&unavailable[0].0); }
        if reasons.is_empty() { words.to_owned() } else { format!("{words}: {}", reasons.join("; ")) }
    }

    /// The draft's devices whose type discovery does not have, each said as an apparatus read says it.
    pub(crate) fn uninstalled_draft(&self) -> Vec<String> {
        self.apparatus.draft.iter().filter(|entry| self.catalog.get(&entry.type_id).is_none())
            .map(|entry| self.not_installed(&format!("{}: device type {} is not installed", entry.role, entry.type_id), &entry.type_id)).collect()
    }

    /// Whether `id` already names a device (draft or live) or a node.
    pub(crate) fn instance_taken(&self, id: &str) -> bool { self.id_in_use(id).is_some() }

    /// Why `id` cannot name another device or node: devices and nodes never share an id,
    /// so the clash is said where the id is given, not at Init.
    pub(crate) fn id_in_use(&self, id: &str) -> Option<String> {
        let device = self.apparatus.draft.iter().find(|entry| entry.instance_id == id).map(|entry| entry.role.as_str())
            .or_else(|| self.devices.get(id).map(|live| live.entry.role.as_str()));
        match (device, self.node(id)) {
            (Some(role), _) => Some(format!("{id} is already used by device {role}")),
            (None, Some(node)) => Some(format!("{id} is already used by node {}", node.name)),
            (None, None) => None,
        }
    }

    /// v2 `_free_name`: the family, numbered from 2, that no device holds as its id or its name.
    fn free_device_id(&self, family: &str) -> String {
        (1..).map(|n: u32| if n == 1 { family.to_owned() } else { format!("{family}{n}") })
            .find(|id| !self.instance_taken(id) && !self.apparatus.draft.iter().any(|entry| entry.role == *id)).unwrap()
    }

    fn draft_parameters(&self, type_id: &str, parameters: &Map<String, Value>) -> Result<Map<String, Value>, String> {
        let schema = self.catalog.get(type_id).map(|item| item.parameters.clone()).unwrap_or_default();
        if self.catalog.get(type_id).is_none() { return Ok(parameters.clone()); }
        fields::project_values(&schema, parameters, false).map(|(values, _)| values)
    }

    pub(crate) fn add_device(&mut self, params: &Value) -> Result<Value, String> {
        let type_id = text(params, "type_id")?.to_owned();
        let family = self.device_type(&type_id)?.family.clone();
        let id = match params.get("instance_id").filter(|id| !id.is_null()) {
            Some(id) => {
                let id = instance_id(id.as_str())?;
                if let Some(clash) = self.id_in_use(id) { return Err(clash); }
                id.to_owned()
            }
            None => self.free_device_id(&family),
        };
        let role = params["role"].as_str().map(str::trim).filter(|role| !role.is_empty()).unwrap_or(&id).to_owned();
        if self.apparatus.draft.iter().any(|entry| entry.role == role) { return Err(format!("Another device is already called {role:?}")); }
        let supplied = params["parameters"].as_object().cloned().unwrap_or_default();
        let parameters = self.draft_parameters(&type_id, &supplied)?;
        self.report = (format!("added {role} ({type_id})"), "task");
        self.apparatus.draft.push(Entry { instance_id: id.clone(), role, type_id, parameters });
        Ok(json!({"device": id}))
    }

    pub(crate) fn remove_device(&mut self, id: &str) -> Result<Value, String> {
        // A loaded device the draft no longer holds is out already: Apply closes it.
        if !self.devices.contains_key(id) { self.apparatus.entry(id)?; }
        self.apparatus.draft.retain(|entry| entry.instance_id != id);
        self.report = (format!("removed {id}"), "task");
        Ok(json!({"device": id}))
    }

    pub(crate) fn edit_device(&mut self, params: &Value) -> Result<Value, String> {
        let id = text(params, "device")?;
        let mut entry = self.apparatus.entry(id)?.clone();
        if let Some(role) = params["role"].as_str() {
            let role = role.trim();
            if role.is_empty() { return Err("A device needs a name".into()); }
            entry.role = role.to_owned();
        }
        if let Some(type_id) = params["type_id"].as_str().filter(|type_id| *type_id != entry.type_id) {
            self.device_type(type_id)?;
            // A new type starts from its own defaults; nothing carries over.
            entry.type_id = type_id.to_owned();
            entry.parameters = self.draft_parameters(type_id, &Map::new())?;
        }
        if let Some(patch) = params["parameters"].as_object() {
            let mut merged = entry.parameters.clone();
            merged.extend(patch.clone());
            entry.parameters = self.draft_parameters(&entry.type_id, &merged)?;
        }
        let index = self.apparatus.draft.iter().position(|item| item.instance_id == id).unwrap();
        self.apparatus.draft[index] = entry;
        Ok(json!({"device": id}))
    }

    pub(crate) fn edit_simulation(&mut self, values: &Value) -> Result<Value, String> {
        let patch = values.as_object().ok_or("values must be an object")?;
        let mut merged = self.apparatus.simulation.clone();
        merged.extend(patch.clone());
        let (projected, _) = fields::project_values(&simulation_fields(), &merged, false)?;
        self.apparatus.simulation = projected;
        Ok(json!({"simulation": self.apparatus.simulation}))
    }

    /// v2 templates: every addable virtual type, or every addable hardware type, once.
    pub(crate) fn use_template(&mut self, template: &str) -> Result<Value, String> {
        // The plugin packages name their templates (v2 zlc_atom.install.templates); one that
        // needs a type this machine lacks is refused whole, as v2 did.
        let rows = self.catalog.templates.iter().find(|(name, _)| name == template).map(|(_, rows)| rows.clone())
            .ok_or_else(|| format!("Unknown template {template:?}; the installed ones are {}",
                self.catalog.templates.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ")))?;
        let missing: Vec<&str> = rows.iter().filter(|(_, type_id)| self.catalog.get(type_id).is_none_or(|item| !item.is_device())).map(|(_, type_id)| type_id.as_str()).collect();
        if !missing.is_empty() { return Err(format!("The {template} template needs device types this machine does not offer: {}", missing.join(", "))); }
        self.apparatus.draft.clear();
        for (instance_id, type_id) in &rows { self.add_device(&json!({"type_id": type_id, "instance_id": instance_id}))?; }
        self.report = (format!("new {template} apparatus draft"), "task");
        Ok(json!({"devices": self.apparatus.draft.iter().map(|entry| &entry.instance_id).collect::<Vec<_>>()}))
    }

    /// Scan hardware (v2 DeviceManagerPresenter.discover): every type with a scan is
    /// asked at once in its own plugin process; nothing found is opened or joins the
    /// draft. The reply comes when every type has answered (or missed its deadline).
    pub(crate) fn discover_devices(&mut self, request: u64) -> Result<Option<Value>, String> {
        self.not_scanning()?;
        let types: Vec<String> = self.catalog.devices().filter(|item| item.discoverable()).map(|item| item.type_id.clone()).collect();
        if types.is_empty() {
            let why: String = self.unavailable_families().iter().map(|(reason, families)| format!("; {}: {reason}", families.join(", "))).collect();
            return Err(format!("No installed device type can scan for hardware{why}"));
        }
        for type_id in &types {
            let ticket = self.ticket(Ticket::Discover { type_id: type_id.clone() });
            self.plugins.discover(ticket, type_id);
        }
        self.scan = Some(Scan { request, asked: types.clone(), pending: types, found: Vec::new(), reasons: Vec::new() });
        Ok(None)
    }

    /// v2 _still_scanning: a scan, or a family still answering past its deadline, holds Init, Apply and
    /// another scan, and says so by name.
    fn not_scanning(&self) -> Result<(), String> {
        let mut families = self.scan.as_ref().map(|scan| scan.pending.clone()).unwrap_or_default();
        for family in self.plugins.scanning() { if !families.contains(&family) { families.push(family); } }
        if families.is_empty() { Ok(()) } else { Err(format!("hardware scan still running: {}", families.join(", "))) }
    }

    /// One type's scan answered: its finds (with their type) and what it could not ask.
    pub(crate) fn discovered(&mut self, type_id: &str, result: Result<Value, String>) {
        // A find may be of another type than the scan that found it (a pulse board another
        // computer publishes is a sequencer.hardware row).
        let parameters = |kind: &str, found: &Map<String, Value>| self.draft_parameters(kind, found);
        let mut finds = Vec::new();
        let mut reasons = Vec::new();
        match result {
            Ok(answer) => {
                for item in answer["found"].as_array().into_iter().flatten() {
                    let instance_id = match instance_id(item["instance_id"].as_str()) {
                        Ok(id) => id.to_owned(),
                        Err(reason) => { reasons.push(reason); continue; }
                    };
                    let role = item["role"].as_str().map(str::trim).filter(|role| !role.is_empty()).unwrap_or(&instance_id).to_owned();
                    let kind = item["type_id"].as_str().filter(|kind| self.catalog.get(kind).is_some_and(|item| item.is_device())).unwrap_or(type_id);
                    match parameters(kind, item["parameters"].as_object().unwrap_or(&Map::new())) {
                        Ok(parameters) => finds.push(Entry { instance_id, role, type_id: kind.to_owned(), parameters }),
                        Err(error) => reasons.push(error),
                    }
                }
                reasons.extend(answer["notes"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_owned));
            }
            Err(error) => reasons.push(error),
        }
        let Some(scan) = self.scan.as_mut() else { return };
        scan.pending.retain(|pending| pending != type_id);
        scan.found.extend(finds);
        // One line per reason, naming every family that gave it (two VISA scans on one empty bus say it once).
        for reason in reasons {
            match scan.reasons.iter_mut().find(|(known, _)| *known == reason) {
                Some((_, families)) => families.push(type_id.to_owned()),
                None => scan.reasons.push((reason, vec![type_id.to_owned()])),
            }
        }
        if !scan.pending.is_empty() { return; }
        let mut scan = self.scan.take().expect("scan");
        // The families that did not load could not be asked: each says why (no plugin environment, a missing vendor runtime).
        for (reason, families) in self.unavailable_families() {
            match scan.reasons.iter_mut().find(|(known, _)| *known == reason) {
                Some((_, asked)) => for family in families { if !asked.contains(&family) { asked.push(family); } },
                None => scan.reasons.push((reason, families)),
            }
        }
        // Families in the catalogue's order, whichever answered first (v2 joined its threads in order).
        let rank = |family: &String| scan.asked.iter().position(|asked| asked == family).unwrap_or(usize::MAX);
        for (_, families) in &mut scan.reasons { families.sort_by_key(rank); }
        scan.reasons.sort_by_key(|(_, families)| families.first().map_or(usize::MAX, rank));
        scan.found.sort_by_key(|entry| rank(&entry.type_id));
        let count = scan.found.len();
        let mut message = format!("discovered {count} device{}", if count == 1 { "" } else { "s" });
        for (reason, families) in &scan.reasons { message.push_str(&format!("; {}: {reason}", families.join(", "))); }
        self.apparatus.discovered = scan.found;
        let severity = if scan.reasons.is_empty() { "task" } else { "warning" };
        // v2 discover's line on the status strip.
        self.report = (message.clone(), severity);
        self.replies.push((scan.request, Ok(json!({"found": count, "message": message, "severity": severity}))));
    }

    pub(crate) fn add_discovered(&mut self, id: &str) -> Result<Value, String> {
        let entry = self.apparatus.discovered.iter().find(|entry| entry.instance_id == id).cloned().ok_or_else(|| format!("No discovered device {id:?}"))?;
        // v2's words for a device already there; a node's id is said as Add device says it.
        // A loaded device the draft dropped comes back as itself (Apply keeps it open).
        if self.apparatus.draft.iter().any(|item| item.instance_id == entry.instance_id || item.role == entry.role) {
            return Err(format!("{} is already configured", entry.role));
        }
        if !self.devices.contains_key(&entry.instance_id) { if let Some(clash) = self.id_in_use(&entry.instance_id) { return Err(clash); } }
        self.apparatus.draft.push(entry);
        Ok(json!({"device": id}))
    }

    /// Replace the draft from a `zlc.installation` file, entry by entry: a device
    /// entry v2 would refuse is left out with its reason and every other one loads.
    /// An unknown type is kept (it fails to open with its reason); a parameter the
    /// type no longer has is dropped, and a value its field refuses is left at its
    /// default, each with a note. The Devices@ line says all of it (v2 `_report`).
    pub(crate) fn load_apparatus(&mut self, path: &Path) -> Result<Value, String> {
        let read = workspace::read_strict(path, &format!("apparatus {}", path.display())).and_then(|document| self.read_apparatus(&document, path));
        let (draft, simulation, skipped, notes) = match read {
            Ok(read) => read,
            // v2 load: a file that cannot be read leaves the draft as it was and says why.
            Err(error) => { self.report = (format!("cannot read {}: {error}", file_name(path)), "error"); return Err(error); }
        };
        let mut line = format!("{} from {}", counted(draft.len(), "device"), file_name(path));
        if !skipped.is_empty() { line += &format!(" · {} not loaded -- {}", skipped.len(), skipped.join("; ")); }
        if !notes.is_empty() { line += &format!(" · {}", notes.join("; ")); }
        self.report = (line, if skipped.is_empty() && notes.is_empty() { "task" } else { "warning" });
        self.notes.extend(skipped.iter().chain(&notes).cloned());
        self.apparatus.draft = draft;
        self.apparatus.simulation = simulation;
        self.apparatus.path = path.to_path_buf();
        self.apparatus.baseline = self.apparatus.tree();
        Ok(json!({"path": path, "devices": self.apparatus.draft.len(), "skipped": skipped, "notes": notes}))
    }

    /// The draft and simulation a file holds, the entries left out (with why) and the notes.
    fn read_apparatus(&self, document: &Value, path: &Path) -> Result<(Vec<Entry>, Map<String, Value>, Vec<String>, Vec<String>), String> {
        let object = document.as_object().ok_or("An apparatus file is a JSON object")?;
        if object.get("format").and_then(Value::as_str) != Some(FORMAT) { return Err(format!("{} is not a saved apparatus ({FORMAT})", path.display())); }
        let devices = object.get("devices").and_then(Value::as_array).ok_or("devices must be a list")?;
        let (mut draft, mut skipped, mut notes) = (Vec::new(), Vec::new(), Vec::new());
        for key in object.keys().filter(|key| !["format", "simulation", "devices"].contains(&key.as_str())) {
            notes.push(format!("{key} is not part of an apparatus file; it was dropped"));
        }
        for (index, item) in devices.iter().enumerate() {
            match self.read_entry(item, &draft, &mut notes) {
                Ok(entry) => draft.push(entry),
                Err(reason) => skipped.push(format!("device {}: {reason}", index + 1)),
            }
        }
        let saved = match object.get("simulation") {
            None => Map::new(),
            Some(Value::Object(saved)) => saved.clone(),
            Some(_) => { notes.push("simulation must be an object; the simulation settings are at their defaults".into()); Map::new() }
        };
        let (simulation, refusals) = fields::restore(&simulation_fields(), &saved)?;
        notes.extend(refusals.into_iter().map(|refusal| format!("simulation: {refusal}; it is at its default")));
        Ok((draft, simulation, skipped, notes))
    }

    /// One device entry as v2 `DeviceInstanceConfig.from_dict` reads it; Err says why it is left out.
    fn read_entry(&self, item: &Value, read: &[Entry], notes: &mut Vec<String>) -> Result<Entry, String> {
        let entry = item.as_object().ok_or("a device entry is an object")?;
        let missing: Vec<&str> = ENTRY_KEYS.into_iter().filter(|key| !entry.contains_key(*key)).collect();
        if !missing.is_empty() { return Err(format!("a device entry has {}; this one has no {}", ENTRY_KEYS.join(", "), missing.join(", "))); }
        let instance_id = instance_id(entry["instance_id"].as_str())?.to_owned();
        let role = entry_text(entry["role"].as_str(), "role")?.to_owned();
        let type_id = entry_text(entry["type_id"].as_str(), "type_id")?.to_owned();
        let saved = entry["parameters"].as_object().ok_or_else(|| format!("{role}: device parameters must be an object"))?;
        if let Some(other) = read.iter().find(|other| other.instance_id == instance_id) { return Err(format!("{instance_id} is already used by device {}", other.role)); }
        if let Some(node) = self.node(&instance_id) { return Err(format!("{instance_id} is already used by node {}", node.name)); }
        if read.iter().any(|other| other.role == role) { return Err(format!("another device is already called {role}")); }
        for key in entry.keys().filter(|key| !ENTRY_KEYS.contains(&key.as_str())) {
            notes.push(format!("{role}: {key} is not part of a device entry; it was dropped"));
        }
        let parameters = match self.catalog.get(&type_id) {
            Some(descriptor) => {
                let (values, refusals) = fields::restore(&descriptor.parameters, saved).map_err(|error| format!("{role}: {error}"))?;
                for gone in saved.keys().filter(|key| !descriptor.parameters.iter().any(|field| &field.name == *key)) {
                    notes.push(format!("{role}: {gone} is no longer a {type_id} setting; it was dropped"));
                }
                notes.extend(refusals.into_iter().map(|refusal| format!("{role}: {refusal}; it is at its default")));
                values
            }
            // Discovery has not answered yet: whether the type is installed is not known (refresh_catalog reads again).
            None if self.plugins.discovering() => saved.clone(),
            None => { notes.push(self.not_installed(&format!("{role}: device type {type_id} is not installed"), &type_id)); saved.clone() }
        };
        Ok(Entry { instance_id, role, type_id, parameters })
    }

    pub(crate) fn save_apparatus(&mut self, path: &Path) -> Result<Value, String> {
        let roles: BTreeSet<&str> = self.apparatus.draft.iter().map(|entry| entry.role.as_str()).collect();
        if roles.len() != self.apparatus.draft.len() { return Err("Two devices have the same name; rename one before saving".into()); }
        let tree = self.apparatus.tree();
        workspace::write_readable(path, &tree)?;
        self.apparatus.path = path.to_path_buf();
        self.apparatus.baseline = tree;
        self.report = (format!("saved {} to {}", counted(self.apparatus.draft.len(), "device"), file_name(path)), "task");
        Ok(json!({"path": path}))
    }

    fn simulated(&self, type_id: &str) -> bool {
        self.catalog.get(type_id).is_some_and(|item| item.process_group.as_deref() == Some(SIMULATION_GROUP))
    }

    fn open_entry(&mut self, entry: Entry) {
        let context = if self.simulated(&entry.type_id) { json!({"simulation": self.apparatus.simulation}) } else { json!({}) };
        let complete = self.catalog.get(&entry.type_id).map(|item| fields::project_values(&item.parameters, &entry.parameters, true));
        let id = entry.instance_id.clone();
        let failure = match (self.catalog.get(&entry.type_id), complete) {
            (None, _) => Some({
                let words = format!("{} is not installed", entry.type_id);
                let why = self.not_installed(&words, &entry.type_id);
                if why == words { format!("{words} (see the plugins that failed to load)") } else { why }
            }),
            // v2: the factory's projection raised it, and _outcome names its type.
            (_, Some(Err(error))) => Some(format!("ValueError: {error}")),
            _ => None,
        };
        let state = if failure.is_some() { LiveState::Failed } else { LiveState::Opening };
        if failure.is_none() { self.plugins.open(&id, &entry.type_id, &Value::Object(entry.parameters.clone()), &context); }
        self.devices.insert(id, Live { entry, state, error: failure, working_point: Value::Null, fields: None, close_pending: false, reopen: None,
            risk: None, units: BTreeMap::new(), descriptor: None, published: false });
    }

    /// The type a live instance is: the one it opened as, else its row's.
    pub(crate) fn live_type<'a>(&'a self, live: &'a Live) -> Option<&'a crate::TypeDescriptor> {
        live.descriptor.as_ref().or_else(|| self.catalog.get(&live.entry.type_id))
    }

    /// Why a published device cannot change here now (v2: publication is a claim).
    fn refuse_published(&self, ids: &[&str]) -> Result<(), String> {
        match ids.iter().find(|id| self.devices.get(**id).is_some_and(|live| live.published)) {
            Some(id) => Err(format!("{}: {PUBLISHED}", self.device_role(id))),
            None => Ok(()),
        }
    }

    /// Remote on/off (v2 Device Manager `toggle_remote`): hand an open device to
    /// another bench computer, or take it back. Its runs and controls here wait.
    pub(crate) fn publish_device(&mut self, id: &str, published: bool, request: u64) -> Result<(), String> {
        // v2 toggle_remote's refusals, in its words.
        if self.devices.values().all(|live| live.state != LiveState::Open) { return Err("initialize devices before publishing one".into()); }
        let live = self.devices.get(id).filter(|live| live.state == LiveState::Open && !live.close_pending)
            .ok_or_else(|| format!("no loaded device '{id}'"))?;
        if published {
            if live.published { return Err(format!("{id} is already published")); }
            if live.descriptor.is_some() { return Err(format!("{id}: this device already comes from the bench fabric -- publish it on the machine that owns it")); }
            let holders: Vec<String> = self.claims_on(id).into_iter().map(|claim| claim.node_name).collect();
            if !holders.is_empty() { return Err(format!("{}: not published -- {} uses it", live.entry.role, holders.join(", "))); }
        } else if !live.published { return Err(format!("{} is not published", live.entry.role)); }
        let role = live.entry.role.clone();
        let ticket = self.ticket(Ticket::Publish { device: id.to_owned(), published, request });
        if !published { self.report = (format!("{id}: withdrawing from the bench fabric"), "task"); }
        self.plugins.publish(ticket, id, &role, published);
        Ok(())
    }

    pub(crate) fn published(&mut self, id: &str, published: bool, request: u64, result: Result<Value, String>) {
        if let Ok(answer) = &result {
            if let Some(live) = self.devices.get_mut(id) {
                live.published = published;
                // Its Control here would go on looking editable over a Current nobody re-reads.
                live.fields = None;
                live.risk = None;
            }
            // v2 toggle_remote's lines.
            self.report = (if published { format!("{id}: published on the bench fabric (port {})", answer["port"]) }
                           else { format!("{id}: withdrawn from the bench fabric") }, "task");
        }
        self.replies.push((request, result.map(|answer| json!({"device": id, "published": published, "fabric": answer}))));
    }

    /// Init / Apply: reconcile the live instances to the draft.
    pub(crate) fn apply_devices(&mut self, host: &mut dyn Host) -> Result<Value, String> {
        // A scan still talking to a bus is not interrupted by opening what is on it (v2 _still_scanning).
        self.not_scanning()?;
        // Before discovery answers, every plugin device would fail as not installed.
        if self.plugins.discovering() { return Err("the device types are still being discovered; Init once they are in".into()); }
        // v2: Init starts a session; with one running it is an Apply.
        let kind = if self.devices.is_empty() { Settling::Init } else { Settling::Apply };
        let mut opened = Vec::new();
        let mut closing = Vec::new();
        let simulation_changed = self.applied_simulation.as_ref().is_some_and(|applied| *applied != self.apparatus.simulation);
        for live_id in self.devices.keys().cloned().collect::<Vec<_>>() {
            let wanted = self.apparatus.draft.iter().find(|entry| entry.instance_id == live_id).cloned();
            let live = &self.devices[&live_id];
            // A published device is the other computer's: no rebuild or close of it here.
            let changes = wanted.as_ref().is_none_or(|entry| entry.type_id != live.entry.type_id || entry.parameters != live.entry.parameters);
            if live.published && changes { return Err(format!("{}: {PUBLISHED}", live.entry.role)); }
            match wanted {
                None => closing.push((live_id, None)),
                Some(entry) if live.state == LiveState::Failed => { self.devices.remove(&live_id); opened.push(entry.instance_id.clone()); self.open_entry(entry); }
                Some(entry) if entry.type_id != live.entry.type_id || entry.parameters != live.entry.parameters
                    || (simulation_changed && self.simulated(&entry.type_id)) => closing.push((live_id, Some(entry))),
                Some(entry) => { if let Some(live) = self.devices.get_mut(&live_id) { live.entry.role = entry.role; } }
            }
        }
        for entry in self.apparatus.draft.clone() {
            if !self.devices.contains_key(&entry.instance_id) { opened.push(entry.instance_id.clone()); self.open_entry(entry); }
        }
        let stopped = self.close_instances(closing.clone(), host);
        self.applied_simulation = Some(self.apparatus.simulation.clone());
        self.settling = Some(kind);
        self.settle();
        Ok(json!({"opening": opened, "closing": closing.iter().map(|(id, _)| id).collect::<Vec<_>>(), "stopping": stopped}))
    }

    /// Close instances after the runs that claim them stop; returns the nodes stopped.
    fn close_instances(&mut self, closing: Vec<(String, Option<Entry>)>, host: &mut dyn Host) -> Vec<String> {
        let mut stopped = BTreeSet::new();
        for (id, reopen) in closing {
            for node in self.claiming_nodes(&id) { stopped.insert(node); }
            if let Some(live) = self.devices.get_mut(&id) { live.close_pending = true; live.reopen = reopen; }
        }
        for node in &stopped { let _ = self.stop_node(node, host); }
        self.close_released();
        stopped.into_iter().collect()
    }

    /// Close every pending instance no run holds any more.
    pub(crate) fn close_released(&mut self) {
        let ready: Vec<String> = self.devices.iter()
            .filter(|(id, live)| live.close_pending && live.state != LiveState::Closing && self.claiming_nodes(id).is_empty())
            .map(|(id, _)| id.clone()).collect();
        for id in ready {
            let live = self.devices.get_mut(&id).unwrap();
            live.close_pending = false;
            if live.state == LiveState::Failed {
                let reopen = live.reopen.take();
                self.devices.remove(&id);
                if let Some(entry) = reopen { self.open_entry(entry); }
            } else {
                live.state = LiveState::Closing;
                self.plugins.close(&id);
            }
        }
    }

    pub(crate) fn close_device(&mut self, id: &str, host: &mut dyn Host) -> Result<Value, String> {
        if !self.devices.contains_key(id) { return Err(format!("{id} is not open")); }
        self.refuse_published(&[id])?;
        let stopped = self.close_instances(vec![(id.to_owned(), None)], host);
        self.settling = Some(Settling::Close(id.to_owned()));
        self.settle();
        Ok(json!({"device": id, "stopping": stopped}))
    }

    pub(crate) fn shut_down_devices(&mut self, host: &mut dyn Host) -> Result<Value, String> {
        let nodes: Vec<String> = self.nodes.iter().map(|node| node.id.clone()).collect();
        for node in &nodes { let _ = self.stop_node(node, host); }
        let all: Vec<_> = self.devices.keys().map(|id| (id.clone(), None)).collect();
        self.close_instances(all, host);
        self.settling = Some(Settling::Shutdown);
        self.settle();
        Ok(json!({"closing": self.devices.keys().collect::<Vec<_>>()}))
    }

    pub(crate) fn device_opened(&mut self, id: &str, working_point: Value) -> bool {
        let Some(live) = self.devices.get_mut(id) else { return false };
        if live.state == LiveState::Opening { live.state = LiveState::Open; }
        live.error = None;
        // A device another computer publishes is the type it is there.
        live.descriptor = working_point.get("descriptor").and_then(|descriptor| serde_json::from_value(descriptor.clone()).ok());
        live.working_point = working_point;
        self.settle();
        true
    }

    /// The line an Init, Apply, Close or shut down leaves once nothing is opening
    /// or closing any more (v2 _session_ready, _reconcile_active and shutdown finished).
    pub(crate) fn settle(&mut self) {
        let Some(kind) = self.settling.clone() else { return };
        if self.devices.values().any(|live| matches!(live.state, LiveState::Opening | LiveState::Closing) || live.close_pending) { return; }
        self.settling = None;
        // In the apparatus's order, as v2's session lists them.
        let mut ordered: Vec<&Live> = self.apparatus.draft.iter().filter_map(|entry| self.devices.get(&entry.instance_id)).collect();
        ordered.extend(self.devices.values().filter(|live| !self.apparatus.draft.iter().any(|entry| entry.instance_id == live.entry.instance_id)));
        let open: Vec<&str> = ordered.iter().filter(|live| live.state == LiveState::Open).map(|live| live.entry.role.as_str()).collect();
        let failed: Vec<String> = ordered.iter().filter(|live| live.state == LiveState::Failed)
            .map(|live| format!("{}: {}", live.entry.instance_id, live.error.as_deref().unwrap_or("did not start"))).collect();
        let loaded = counted(open.len(), "device");
        let mut text = match &kind {
            Settling::Shutdown => { self.report = ("devices shut down".into(), "task"); return; }
            Settling::Init if open.is_empty() => format!("{loaded} initialized"),
            Settling::Init => format!("{loaded} initialized: {}", open.join(", ")),
            Settling::Apply => format!("device changes applied · {loaded} loaded"),
            Settling::Close(id) => format!("{id} closed · {loaded} loaded"),
        };
        // v2 _outcome: each failure's reason on the line.
        if !failed.is_empty() { text += &format!(" · {} failed -- {}", failed.len(), failed.join("; ")); }
        self.report = (text, if failed.is_empty() { "task" } else { "warning" });
    }

    pub(crate) fn device_failed(&mut self, id: &str, error: String) -> bool {
        let Some(live) = self.devices.get_mut(id) else { return false };
        // An open device lost (its process, or the computer that publishes it) is said at once;
        // one that failed to open is said by the Init or Apply it belongs to.
        if live.state == LiveState::Open { self.report = (format!("{} lost: {error}", live.entry.role), "error"); }
        live.state = LiveState::Failed;
        live.error = Some(error);
        live.fields = None;
        live.published = false;
        self.settle();
        true
    }

    pub(crate) fn device_closed(&mut self, id: &str, error: Option<String>) -> bool {
        let Some(live) = self.devices.remove(id) else { return false };
        if let Some(error) = error { self.notes.push(format!("{}: {error}", live.entry.role)); }
        if let Some(entry) = live.reopen { self.open_entry(entry); }
        self.settle();
        true
    }

    /// Device Control open/Refresh, or a unit picked for some fields (`units`:
    /// field -> spelling, "" = its own unit): the device reads every field,
    /// those in the spelling the operator reads them in.
    pub(crate) fn read_device(&mut self, id: &str, units: Option<&Map<String, Value>>, request: Option<u64>) -> Result<(), String> {
        self.refuse_published(&[id])?;
        let live = self.devices.get_mut(id).filter(|live| live.state == LiveState::Open).ok_or_else(|| format!("{id} is not open"))?;
        for (field, unit) in units.into_iter().flatten() {
            let unit = unit.as_str().ok_or("A unit is a spelling such as MHz")?.trim();
            if unit.is_empty() { live.units.remove(field); } else { live.units.insert(field.clone(), unit.to_owned()); }
        }
        let call = json!({"units": live.units});
        let ticket = self.ticket(Ticket::Fields { device: id.to_owned(), request });
        self.plugins.call(id, ticket, "fields", &call);
        Ok(())
    }

    /// Device Control Apply of one field, under the claims on that device.
    pub(crate) fn tune_device(&mut self, params: &Value, request: u64) -> Result<(), String> {
        let id = text(params, "device")?;
        let field = text(params, "field")?;
        let live = self.devices.get(id).filter(|live| live.state == LiveState::Open).ok_or_else(|| format!("{id} is not open"))?;
        let live_write = live.fields.as_ref().and_then(|fields| fields.as_array())
            .and_then(|fields| fields.iter().find(|item| item["field"]["name"] == field))
            .is_some_and(|item| item["live_write"] == true);
        self.tune_policy(id, field, live_write, live.risk.as_deref()).map_err(|reason| format!("{field}: {reason}"))?;
        let ticket = self.ticket(Ticket::Tune { device: id.to_owned(), request });
        let mut call = json!({"field": field, "value": params["value"]});
        if let Some(unit) = params["unit"].as_str() { call["unit"] = json!(unit); }
        self.plugins.call(id, ticket, "tune", &call);
        Ok(())
    }

    /// v2's Device Control field policy: may `field` of `id` be tuned now (else why not).
    fn tune_policy(&self, id: &str, field: &str, live_write: bool, risk: Option<&str>) -> Result<(), String> {
        if self.devices.get(id).is_some_and(|live| live.published) { return Err(PUBLISHED.into()); }
        let claims = self.claims_on(id);
        let protected: Vec<&str> = claims.iter().filter(|claim| claim.protected.as_ref().is_none_or(|fields| fields.iter().any(|name| name == field)))
            .map(|claim| claim.node_name.as_str()).collect();
        if !protected.is_empty() { return Err(format!("Protected by {}", protected.join(", "))); }
        if claims.is_empty() { return Ok(()); }
        if !live_write { return Err("Stop the active Logic; this field is not live-writable".into()); }
        if claims.iter().any(|claim| risk != Some(claim.run.as_str())) { return Err("Accept risk to edit this unclaimed live-safe field".into()); }
        Ok(())
    }

    /// Device Control rows (WORKBENCH_COMMANDS §3.3 `TunableField`): each tunable field the
    /// plugin read (`{field, current, live_write, device_limits, dependency_group}`) as a
    /// Control with its readback and whether it can be tuned now.
    /// Its readback and the device's own range are written as v2 wrote them
    /// (`current_text`, `limits_text`: format_quantity in the row's spelling), a
    /// number with more than one spelling offers them (`units`, re-read by
    /// `read_device {units}`), and `status` is v2's word for the row at rest.
    pub(crate) fn tunable_rows(&self, id: &str, live: &Live, read: &Value) -> Value {
        Value::Array(read.as_array().into_iter().flatten().filter_map(|item| {
            let field: fields::Field = serde_json::from_value(item["field"].clone()).ok()?;
            let live_write = item["live_write"] == true;
            let current = &item["current"];
            let values: Map<String, Value> = [(field.name.clone(), current.clone())].into_iter().collect();
            let mut row = fields::controls(std::slice::from_ref(&field), &values, "tunable", &[]).pop()?;
            let policy = self.tune_policy(id, &field.name, live_write, live.risk.as_deref());
            let unit = field.unit.as_deref().unwrap_or("");
            let picked = live.units.contains_key(&field.name);
            let limits = item["device_limits"].as_array().filter(|pair| pair.len() == 2)
                .map(|pair| format!("{} to {}", reading_text(&pair[0], unit, picked), reading_text(&pair[1], unit, picked))).unwrap_or_default();
            let outside = current.as_f64().is_some_and(|value| field.minimum.is_some_and(|low| value < low) || field.maximum.is_some_and(|high| value > high));
            let (status, severity) = if policy.is_err() { ("Protected", "warning") } else if outside { ("Stands outside its bounds", "warning") } else { ("Ready", "idle") };
            let numeric = matches!(field.kind.as_str(), "float" | "int") && field.choices.is_empty();
            for (key, value) in [("current", current.clone()), ("desired", current.clone()), ("live_write", json!(live_write)),
                ("device_limits", item["device_limits"].clone()), ("dependency_group", item["dependency_group"].clone()),
                ("current_text", json!(reading_text(current, unit, picked))), ("limits_text", json!(limits)),
                ("units", json!(if numeric { unit_choices(unit) } else { Vec::new() })),
                ("status", json!(status)), ("severity", json!(severity)),
                ("editable", json!(policy.is_ok())), ("reason", json!(policy.err().unwrap_or_default()))] { row[key] = value; }
            Some(row)
        }).collect())
    }

    /// Whether accepting risk would open anything: a claimed device with a field
    /// no claim protects that can be written live (v2 `risk_possible`).
    fn risk_possible(&self, id: &str, live: &Live) -> bool {
        let claims = self.claims_on(id);
        !claims.is_empty() && live.fields.as_ref().and_then(Value::as_array).is_some_and(|rows| rows.iter().any(|item| {
            let name = item["field"]["name"].as_str().unwrap_or("");
            item["live_write"] == true && !claims.iter().any(|claim| claim.protected.as_ref().is_none_or(|fields| fields.iter().any(|field| field == name)))
        }))
    }

    /// Device Control's title and policy line (v2 projection `owner_names`, `reason`, `risk_enabled`).
    fn control_policy(&self, id: &str, live: &Live) -> Value {
        let mut owners: Vec<String> = self.claims_on(id).into_iter().map(|claim| claim.node_name).collect();
        owners.dedup();
        json!({"owners": owners, "risk_enabled": self.risk_possible(id, live),
               "reason": if owners.is_empty() { "No active Logic uses this device" } else { "Risk acceptance applies only to unclaimed live-safe fields" }})
    }

    pub(crate) fn accept_device_risk(&mut self, id: &str, accepted: bool) -> Result<Value, String> {
        let run = self.claims_on(id).first().map(|claim| claim.run.clone());
        let live = self.devices.get(id).ok_or_else(|| format!("{id} is not open"))?;
        if accepted && run.is_some() && !self.risk_possible(id, live) {
            return Err("Every live-writable field of this device is protected by the run that holds it; there is no risk to accept".into());
        }
        let live = self.devices.get_mut(id).ok_or_else(|| format!("{id} is not open"))?;
        live.risk = if accepted { Some(run.ok_or("No run holds this device; there is no risk to accept")?) } else { None };
        Ok(json!({"device": id, "accepted": live.risk.is_some()}))
    }

    /// An open device's declared command for a name: its MCP tool name, or the windows'
    /// `<instance>__<command>` of its id as written. A name two devices could mean is none's.
    fn device_command(&self, name: &str) -> Option<(String, Value)> {
        let mut found = self.devices.iter().filter(|(_, live)| live.state == LiveState::Open).flat_map(|(id, live)| {
            let commands = self.live_type(live).map(|item| item.commands.as_slice()).unwrap_or_default();
            commands.iter().filter(move |command| {
                let declared = command["name"].as_str().unwrap_or("");
                name == zlc_core::control::tool_name(id, declared) || name.strip_prefix(id.as_str()).and_then(|rest| rest.strip_prefix("__")) == Some(declared)
            }).map(move |command| (id.clone(), command.clone()))
        });
        let first = found.next()?;
        found.next().is_none().then_some(first)
    }

    pub fn handles_command(&self, name: &str) -> bool { self.device_command(name).is_some() }

    /// Run a device command in its instance; the reply follows its answer.
    pub(crate) fn run_device_command(&mut self, name: &str, params: &Value, request: u64) -> Result<(), String> {
        let (instance, declared) = self.device_command(name).ok_or_else(|| format!("{name} is not a command of an open device"))?;
        self.refuse_published(&[instance.as_str()])?;
        // A command that drives the device waits for no run: a run holding it refuses it (v2 DeviceUseBusy).
        if declared["drives"] == true {
            let holders: Vec<String> = self.claims_on(&instance).into_iter().map(|claim| claim.node_name).collect();
            if !holders.is_empty() { return Err(format!("device is in use by {}", holders.join(", "))); }
        }
        let fields: Vec<fields::Field> = serde_json::from_value(declared["params"].clone()).unwrap_or_default();
        let supplied = params.as_object().cloned().unwrap_or_default();
        let (mut values, _) = fields::project_values(&fields, &supplied, true).map_err(|error| format!("{name}: {error}"))?;
        // The file a command names (its declared `file`) is the workspace's when relative (v2
        // lab_editor_action's workspace_path); the plugin host reads or writes it on this computer.
        if let Some(param) = declared["file"]["param"].as_str() {
            if let Some(path) = values.get(param).and_then(Value::as_str).filter(|path| !path.trim().is_empty()) {
                let resolved = self.workspace.resolve(path.trim()).display().to_string();
                values.insert(param.into(), json!(resolved));
            }
        }
        let ticket = self.ticket(Ticket::Command { request });
        self.plugins.call(&instance, ticket, "command", &json!({"name": declared["name"], "params": values}));
        Ok(())
    }

    /// Every open device's commands as MCP tool rows.
    pub fn plugin_tools(&self) -> Vec<Value> {
        self.devices.iter().filter(|(_, live)| live.state == LiveState::Open).flat_map(|(id, live)| {
            self.live_type(live).map(|item| item.commands.clone()).unwrap_or_default().into_iter().map(move |command| {
                let schema = if command["input_schema"].is_object() { command["input_schema"].clone() } else { json!({"type": "object"}) };
                json!({"name": zlc_core::control::tool_name(id, command["name"].as_str().unwrap_or("")), "description": command["summary"],
                       "inputSchema": schema, "annotations": {"readOnlyHint": command["read_only"] == true}})
            })
        }).collect()
    }

    pub(crate) fn apparatus_state(&self) -> Value {
        let pending = self.apparatus.draft.iter().any(|entry| self.devices.get(&entry.instance_id).is_none_or(|live| live.entry != *entry))
            || self.devices.keys().any(|id| self.apparatus.draft.iter().all(|entry| &entry.instance_id != id));
        let action = if self.devices.is_empty() { (!self.apparatus.draft.is_empty()).then_some("init") }
            else if pending { Some("apply") } else { Some("shut_down") };
        let busy = self.devices.values().filter(|live| matches!(live.state, LiveState::Opening | LiveState::Closing)).count();
        json!({
            "path": self.apparatus.path, "dirty": self.apparatus.tree() != self.apparatus.baseline,
            "simulation": self.apparatus.simulation,
            "types": self.catalog.devices().map(crate::TypeDescriptor::summary).collect::<Vec<_>>(),
            "unavailable": self.catalog.errors.iter().map(|(module, reason)| json!({"module": module, "reason": reason})).collect::<Vec<_>>(),
            "templates": self.catalog.templates.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>(),
            "discovered": self.apparatus.discovered.iter().map(Entry::tree).collect::<Vec<_>>(),
            "action": action,
            "report": {"text": self.report.0, "severity": self.report.1},
            // v2's busy words: the scan, else the device change under way.
            "busy": if self.scan.is_some() { Some("scanning hardware".to_owned()) }
                    else if self.plugins.discovering() { Some("discovering device types".to_owned()) }
                    else { self.settling.as_ref().map(Settling::busy).or_else(|| (busy > 0).then(|| "applying device changes".to_owned())) },
        })
    }

    pub(crate) fn device_rows(&self) -> Vec<Value> {
        let mut ids: Vec<&str> = self.apparatus.draft.iter().map(|entry| entry.instance_id.as_str()).collect();
        ids.extend(self.devices.keys().map(String::as_str).filter(|id| !self.apparatus.draft.iter().any(|entry| entry.instance_id == *id)));
        ids.into_iter().map(|id| {
            let draft = self.apparatus.draft.iter().find(|entry| entry.instance_id == id);
            let live = self.devices.get(id);
            let entry = draft.or(live.map(|live| &live.entry)).unwrap();
            let descriptor = self.catalog.get(&entry.type_id);
            let state = match live.map(|live| live.state) {
                None => "draft", Some(LiveState::Opening) => "opening", Some(LiveState::Open) => "open",
                Some(LiveState::Closing) => "closing", Some(LiveState::Failed) => "failed",
            };
            let pending = match (draft, live) {
                (Some(_), None) => Some("add"),
                (None, Some(_)) => Some("remove"),
                (Some(draft), Some(live)) if draft.type_id != live.entry.type_id || draft.parameters != live.entry.parameters => Some("change"),
                _ => None,
            };
            let claims: Vec<Value> = self.claims_on(id).into_iter().map(|claim| json!({"node_id": claim.node, "run": claim.run,
                "protected_fields": claim.protected})).collect();
            // What it is once open (a remote row: the publisher's type); its settings stay its row's.
            let kind = live.and_then(|live| live.descriptor.as_ref()).or(descriptor);
            json!({
                "instance_id": id, "role": entry.role, "type_id": entry.type_id,
                "family": kind.map(|item| item.family.as_str()).unwrap_or("Unavailable"),
                "label": kind.map(crate::TypeDescriptor::label).unwrap_or(&entry.type_id),
                "parameters": entry.parameters,
                "controls": descriptor.map(|item| fields::controls(&item.parameters, &entry.parameters, "device", &[])).unwrap_or_default(),
                "state": state, "pending": pending, "error": live.and_then(|live| live.error.clone()),
                "working_point": live.map(|live| live.working_point.clone()).unwrap_or(Value::Null),
                "claimed_by": claims,
                "physical_outputs": kind.map(|item| item.physical_outputs.iter().map(|output| json!({"name": output.name})).collect::<Vec<_>>()).unwrap_or_default(),
                "control_window": kind.map(|item| item.control_window.as_str()).unwrap_or(""),
                "commands": if state == "open" { kind.map(|item| item.commands.iter().filter_map(|command| command["name"].as_str())
                    .map(|name| zlc_core::control::tool_name(id, name)).collect::<Vec<_>>()).unwrap_or_default() } else { Vec::new() },
                "fields": live.and_then(|live| live.fields.as_ref().map(|read| self.tunable_rows(id, live, read))),
                "control": live.filter(|live| live.fields.is_some()).map(|live| self.control_policy(id, live)),
                "risk_accepted": live.is_some_and(|live| live.risk.is_some()),
                // Remote on (v2's switch); `remote`: where a device another computer publishes is.
                "published": live.is_some_and(|live| live.published),
                "publishable": live.is_some_and(|live| live.state == LiveState::Open && live.descriptor.is_none()),
                "remote": live.map(|live| live.working_point["remote"].clone()).unwrap_or(Value::Null),
            })
        }).collect()
    }

    /// Open instances providing `capability`, in operator order: (instance_id, role).
    pub(crate) fn device_options(&self, capability: &str) -> Vec<(String, String)> {
        let mut options: Vec<(String, String)> = self.apparatus.draft.iter().map(|entry| entry.instance_id.clone())
            .chain(self.devices.keys().cloned()).collect::<Vec<_>>().into_iter()
            .filter_map(|id| self.devices.get(&id).filter(|live| live.state == LiveState::Open)
                .filter(|live| self.live_type(live).is_some_and(|item| item.capabilities.iter().any(|token| token == capability)))
                .map(|live| (id.clone(), live.entry.role.clone())))
            .collect();
        let mut seen = BTreeSet::new();
        options.retain(|(id, _)| seen.insert(id.clone()));
        options
    }

    pub(crate) fn capability_of(&self, id: &str, capability: &str) -> bool {
        self.devices.get(id).and_then(|live| self.live_type(live))
            .is_some_and(|item| item.capabilities.iter().any(|token| token == capability))
    }

    /// Published on the bench fabric (no run here may claim it).
    pub(crate) fn device_published(&self, id: &str) -> bool { self.devices.get(id).is_some_and(|live| live.published) }

    pub(crate) fn device_open(&self, id: &str) -> bool { self.devices.get(id).is_some_and(|live| live.state == LiveState::Open && !live.close_pending) }
    pub(crate) fn device_role<'a>(&'a self, id: &'a str) -> &'a str {
        self.devices.get(id).map(|live| live.entry.role.as_str())
            .or_else(|| self.apparatus.draft.iter().find(|entry| entry.instance_id == id).map(|entry| entry.role.as_str())).unwrap_or(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readings_read_as_v2_wrote_them_and_spellings_group_by_family() {
        assert_eq!(reading_text(&json!(120000000.0), "Hz", false), "120 MHz");
        assert_eq!(reading_text(&json!(120000000.0), "Hz", true), "120000000 Hz");
        assert_eq!(reading_text(&json!(0.0001), "s", false), "100 µs");
        assert_eq!(reading_text(&json!(-3.02), "dBm", false), "-3.02 dBm");
        assert_eq!(reading_text(&json!(7), "", false), "7");
        assert_eq!(reading_text(&json!("idle"), "Hz", false), "idle");
        assert_eq!(reading_text(&Value::Null, "Hz", false), "\u{2014}");
        let hertz: Vec<String> = unit_choices("Hz").iter().map(|choice| choice["value"].as_str().unwrap().to_owned()).collect();
        assert_eq!(hertz, ["GHz", "MHz", "kHz", "Hz"]);
        let power = DEFAULT_UNITS.choice_tree("dBm");
        assert_eq!(power[0], ("dBm".to_owned(), vec!["dBm".to_owned()]));
        assert!(power.iter().any(|(trunk, leaves)| trunk == "W" && leaves.contains(&"mW".to_owned())));
        assert!(unit_choices("1").is_empty() && unit_choices("count").is_empty());
    }
}
