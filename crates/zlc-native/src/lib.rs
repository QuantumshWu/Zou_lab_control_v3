//! The same native rules inside the application and in offline Python tools.
//! This router owns no instruments, UI, interpreter, or Runtime history.
pub mod arrays;
pub mod durable;
mod units;
use serde_json::{Value, json};
pub use zlc_scientific::Tensor;

#[derive(Default)]
pub struct Service { scientific: zlc_scientific::Service, durable: durable::Service }

impl Service {
    pub fn dispatch(&mut self, operation: &str, params: &Value,
        read: &dyn Fn(&Value) -> Result<Tensor, String>,
        write: &dyn Fn(Tensor) -> Result<Value, String>) -> Result<Value, String> {
        if operation.starts_with("scan.") { return zlc_scan::authoring::dispatch(operation, params); }
        if operation.starts_with("scientific.") { return self.scientific.dispatch(operation, params, read, write); }
        if operation.starts_with("units.") { return units::dispatch(operation, params, read, write); }
        if operation.starts_with("durable.") { return Ok(self.durable.dispatch(operation, params, read)); }
        if operation == "fields.project" {
            let fields: Vec<zlc_authoring::Field> = serde_json::from_value(params["fields"].clone()).map_err(|e|e.to_string())?;
            let values = params["values"].as_object().ok_or("field values must be an object")?;
            return Ok(match zlc_authoring::typed::values(&fields, values, params["complete"].as_bool().unwrap_or(false)) {
                Ok((values, complete)) => {
                    let enabled: serde_json::Map<String,Value> = fields.iter().map(|field|
                        (field.name.clone(), json!(zlc_authoring::enabled(field,&values)))).collect();
                    json!({"values": values, "complete": complete, "enabled":enabled})
                },
                Err(error) => json!({"error":{"type":error.kind,"message":error.message}}),
            });
        }
        if operation == "fields.value" {
            let field: zlc_authoring::Field = serde_json::from_value(params["field"].clone()).map_err(|e|e.to_string())?;
            return Ok(match zlc_authoring::typed::project(&field, &params["value"], params["complete"].as_bool().unwrap_or(false)) {
                Ok(value) => json!({"value":value}),
                Err(error) => json!({"error":{"type":error.kind,"message":error.message}}),
            });
        }
        Err(format!("unknown native operation {operation:?}"))
    }
}
