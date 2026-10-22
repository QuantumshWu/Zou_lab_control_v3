//! Plugin type descriptors as the plugin process's `describe` answers them (PLUGIN_PROTOCOL.md §2, §8).
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An output Stop all drives safe through the adapter's `make_safe(name)`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PhysicalOutput { pub name: String }

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CommandDescriptor {
    pub name: String,
    pub summary: String,
    /// Only reads: offered to a read-only MCP client too.
    #[serde(default)]
    pub read_only: bool,
    /// Commands the device itself: refused while a run claims it.
    #[serde(default)]
    pub drives: bool,
    pub params: Vec<Value>,
    pub input_schema: Value,
    /// A parameter naming a file on the core's computer, which the core reads or writes (files.rs).
    #[serde(default)]
    pub file: Option<FileParam>,
}

/// A command's file: `param` names it; `direction` is "read" (the core reads it and the plugin
/// gets its bytes) or "write" (the plugin gives the bytes and the core writes them).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FileParam { pub param: String, pub direction: String }

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TypeDescriptor {
    pub type_id: String,
    /// "device" or "node".
    pub kind: String,
    pub family: String,
    pub label: String,
    pub description: String,
    pub module: String,
    /// A device standing in the simulated world, opened with the installation's simulation parameters.
    #[serde(default)]
    pub simulation: bool,
    /// Why the type cannot share the one plugin process; empty: it shares it.
    #[serde(default)]
    pub isolated: String,
    pub capabilities: Vec<String>,
    /// Authoring fields (PLUGIN_PROTOCOL.md §2 `Field`), as data.
    pub parameters: Vec<Value>,
    pub physical_outputs: Vec<PhysicalOutput>,
    pub commands: Vec<CommandDescriptor>,
    pub queries: Vec<String>,
    /// Node inputs, outputs, device requirements and previews, as declared.
    #[serde(default)]
    pub node: Option<Value>,
    #[serde(default = "yes")]
    pub addable: bool,
    /// The window a device's Control opens instead of the field form ("slm-editor"); empty for the form.
    #[serde(default)]
    pub control_window: String,
    /// A device with a server of its own (SDK `announce`): the client type Remote on announces
    /// type another machine opens there (the instance op `publish`), as the pulse board's.
    #[serde(default)]
    pub announces: String,
    /// The physical instrument a row names, over its parameters (`rigol-dg4000:{resource}`).
    #[serde(default)]
    pub identity: String,
}

fn yes() -> bool { true }

impl TypeDescriptor {
    pub fn is_device(&self) -> bool { self.kind == "device" }

    /// The dataset input a node run uses: the first whose `enabled_when` the run's
    /// values (a parameter's default where not given) meet (v2 `dataset_inputs`).
    pub fn active_input(&self, values: &Value) -> Option<&Value> {
        let value = |name: &Value| name.as_str().and_then(|name| values.get(name)
            .or_else(|| self.parameters.iter().find(|field| field["name"] == name).map(|field| &field["default"]))).unwrap_or(&Value::Null);
        self.node.as_ref()?["inputs"].as_array()?.iter().find(|input| input["enabled_when"].as_array()
            .is_none_or(|rule| rule.get(1).and_then(Value::as_array).is_some_and(|offered| offered.contains(value(&rule[0])))))
    }
}

/// Whether an SPI method only reads: an attribute (`get:<name>`), the knobs
/// as data (`tunable_fields`, the `fields` op's rows), or one of the reads the
/// opened device says it implements (`reads`: its knobs' values and
/// provenance, its own unit conversions). A draft query may call only these.
pub fn method_reads(method: &str, reads: &[String]) -> bool {
    method.starts_with("get:") || method == "tunable_fields" || reads.iter().any(|read| read == method)
}

#[derive(Clone, Debug, Deserialize)]
pub struct DescribeError {
    pub module: String,
    pub error: String,
    pub traceback: String,
}

/// Everything discovery found, and every leaf that failed to load.
#[derive(Clone, Debug, Deserialize)]
pub struct Catalog {
    pub protocol: [u32; 2],
    pub types: Vec<TypeDescriptor>,
    pub errors: Vec<DescribeError>,
    /// Apparatus templates the packages offer: name -> [{instance_id, type_id}].
    #[serde(default)]
    pub templates: serde_json::Map<String, serde_json::Value>,
}

impl Catalog {
    pub fn find(&self, type_id: &str) -> Option<&TypeDescriptor> { self.types.iter().find(|item| item.type_id == type_id) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_and_knob_rows_are_the_reads() {
        let reads = ["convert_tunable_value".to_owned()];
        assert!(method_reads("get:timeout", &[]) && method_reads("tunable_fields", &[]));
        assert!(method_reads("convert_tunable_value", &reads) && !method_reads("convert_tunable_value", &[]));
        assert!(!method_reads("tune", &reads) && !method_reads("set_output", &reads));
        let descriptor: TypeDescriptor = serde_json::from_value(serde_json::json!({
            "type_id": "rf.x", "kind": "device", "family": "rf", "label": "x", "description": "", "module": "m",
            "capabilities": [], "parameters": [], "physical_outputs": [{"name": "rf"}],
            "commands": [{"name": "status", "summary": "s", "read_only": true, "params": [], "input_schema": {}}], "queries": []})).unwrap();
        assert_eq!(descriptor.physical_outputs[0].name, "rf");
        assert!(descriptor.commands[0].read_only);
    }
}
