//! Application adapter around the common typed-array codec.
use serde_json::Value;
use zlc_core::scientific::Tensor;
pub use zlc_archive::array_wire::{dtype,dtype_name};
pub fn decode(value:&Value)->Result<Tensor,String>{let array=zlc_archive::array_wire::decode(value)?;Ok(Tensor{values:array.values,shape:array.shape})}
pub fn encode(value:Tensor)->Result<Value,String>{zlc_archive::array_wire::encode(&value.values,&value.shape)}
