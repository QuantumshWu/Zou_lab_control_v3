//! Plugin type descriptors as the plugin process's describe reports them
//! (PLUGIN_PROTOCOL §2). The workbench reads them; it never branches on a
//! concrete type.
use serde::Deserialize;
use serde_json::{Value, json};

use crate::fields::Field;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct TypeDescriptor {
    pub type_id: String,
    /// `device` or `node`.
    pub kind: String,
    pub family: String,
    pub label: String,
    pub description: String,
    pub module: String,
    /// It stands in the simulated world: opened with the installation's simulation
    /// parameters, and opened again when they change (v2 world_config).
    pub simulation: bool,
    pub capabilities: Vec<String>,
    pub parameters: Vec<Field>,
    pub physical_outputs: Vec<PhysicalOutput>,
    pub commands: Vec<Value>,
    pub addable: Option<bool>,
    pub node: Option<NodeSpec>,
    /// What a device type answers without an open instance (`discover`: its hardware scan).
    pub queries: Vec<String>,
    /// The window its Control opens instead of the field form (`slm-editor`); empty for the form.
    pub control_window: String,
    /// The physical instrument a row of this type names, from its settings (v2's bound
    /// identity): `rigol-dg4000:{resource}`. Empty: every row is its own instrument.
    pub identity: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct PhysicalOutput { pub name: String }

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct NodeSpec {
    /// `measurement`, `task` or `processor`.
    pub node_kind: String,
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<OutputSpec>,
    pub devices: Vec<DeviceRequirement>,
    pub artifact_inputs: Vec<ArtifactInput>,
    pub artifact_outputs: Vec<Value>,
    pub previews: Vec<Preview>,
    pub reports_ready: bool,
    /// Draft queries the type answers in its own process (`declare_outputs`, ...).
    pub queries: Vec<String>,
    /// A field holding the id of the measurement node this one restarts.
    pub acquisition_input: Option<String>,
    pub acquisition_completion: Option<String>,
    /// Draft fields that name a file of a workspace collection (v2 WorkspaceResourceSpec).
    pub workspace_resources: Vec<WorkspaceResource>,
    /// Draft fields a plot selector writes (v2 SelectionMapping).
    pub selection_mappings: Vec<SelectionMapping>,
    /// Node types this one took over: a saved row naming one loads as this type, its values with these over them.
    pub replaces: std::collections::BTreeMap<String, serde_json::Map<String, Value>>,
    /// The draft queries' answers for the type's default draft (its catalogue entry): a node
    /// is drawn from them at once, its own process's answers replacing them when they land.
    pub answers: serde_json::Map<String, Value>,
}

/// A selector of `selector_kind` on a `plot_kind` panel writes `draft_fields`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct SelectionMapping { pub plot_kind: String, pub selector_kind: String, pub draft_fields: Vec<String> }

/// `field` names a file in `<workspace>/<directory>` with one of `suffixes`; the build
/// receives it resolved under `argument`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct WorkspaceResource { pub field: String, pub contract: String, pub directory: String, pub suffixes: Vec<String>, pub argument: String }

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct InputSpec {
    pub name: String,
    pub label: String,
    pub contract: Option<String>,
    pub delivery: String,
    pub select_bundle: bool,
    pub enabled_when: Option<(String, Vec<Value>)>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct OutputSpec { pub name: String, pub contract: Option<String>, pub index_by_source: bool }

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct DeviceRequirement {
    pub capability: String,
    pub argument: String,
    /// `None`: every tunable field of the device is frozen while claimed.
    pub protected_fields: Option<Vec<String>>,
    pub enabled_when: Option<(String, Vec<Value>)>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct ArtifactInput {
    pub name: String,
    pub label: String,
    pub contract_id: Option<String>,
    /// Its codec's file dialog filter (v2 `codec.file_filter`).
    pub file_filter: String,
    pub required: Option<bool>,
    pub frame_claims: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Preview {
    pub output: String,
    pub plot_kind: String,
    pub semantic: Value,
    /// A companion producer suffix: `@logic/<node>/<producer>/<output>`.
    pub producer: String,
}

impl TypeDescriptor {
    pub fn is_device(&self) -> bool { self.kind == "device" }
    pub fn node(&self) -> &NodeSpec {
        static EMPTY: std::sync::OnceLock<NodeSpec> = std::sync::OnceLock::new();
        self.node.as_ref().unwrap_or_else(|| EMPTY.get_or_init(NodeSpec::default))
    }
    pub fn label(&self) -> &str { if self.label.is_empty() { &self.type_id } else { &self.label } }
    /// A device type with a hardware scan (v2 `discover`).
    pub fn discoverable(&self) -> bool { self.is_device() && self.queries.iter().any(|query| query == "discover") }
    /// What views list: everything except the authoring schema.
    pub fn summary(&self) -> Value {
        let mut row = json!({"type_id": self.type_id, "kind": self.kind, "family": self.family, "label": self.label(),
            "description": self.description, "capabilities": self.capabilities,
            "physical_outputs": self.physical_outputs.iter().map(|output| json!({"name": output.name})).collect::<Vec<_>>(),
            "addable": self.addable.unwrap_or(true), "family_label": crate::logic::words(&self.family), "discoverable": self.discoverable()});
        if let Some(node) = &self.node { row["node_kind"] = json!(node.node_kind); }
        row
    }
}

/// The describe result: every type that imported, and why the others did not.
#[derive(Clone, Debug, Default)]
pub struct TypeCatalog {
    pub types: Vec<TypeDescriptor>,
    pub errors: Vec<(String, String)>,
    /// Apparatus starting points (Device Manager New…): name -> (instance id, type id) rows.
    pub templates: Vec<(String, Vec<(String, String)>)>,
}

impl TypeCatalog {
    /// `{"protocol": [2, 0], "types": [...], "errors": [{"module", "error"|"reason"}]}`.
    pub fn from_describe(document: &Value) -> Result<Self, String> {
        let mut types = document["types"].as_array().ok_or("describe result has no types list")?.iter()
            .map(|value| serde_json::from_value::<TypeDescriptor>(value.clone()).map_err(|error| format!("type descriptor: {error}")))
            .collect::<Result<Vec<_>, _>>()?;
        // v2 spells "no acquisition field" as "".
        for node in types.iter_mut().filter_map(|item| item.node.as_mut()) {
            if node.acquisition_input.as_deref() == Some("") { node.acquisition_input = None; }
        }
        let errors = document["errors"].as_array().into_iter().flatten().map(|error| (
            error["module"].as_str().unwrap_or("").to_owned(),
            error["error"].as_str().or_else(|| error["reason"].as_str()).or_else(|| error["message"].as_str()).unwrap_or("").to_owned(),
        )).collect();
        let templates = document["templates"].as_object().into_iter().flatten().map(|(name, rows)| (name.clone(),
            rows.as_array().into_iter().flatten().map(|row| (row["instance_id"].as_str().unwrap_or("").to_owned(), row["type_id"].as_str().unwrap_or("").to_owned())).collect()))
            .collect();
        Ok(Self { types, errors, templates })
    }
    pub fn get(&self, type_id: &str) -> Option<&TypeDescriptor> { self.types.iter().find(|item| item.type_id == type_id) }
    pub fn devices(&self) -> impl Iterator<Item = &TypeDescriptor> { self.types.iter().filter(|item| item.is_device()) }
    pub fn nodes(&self) -> impl Iterator<Item = &TypeDescriptor> { self.types.iter().filter(|item| !item.is_device()) }
}
