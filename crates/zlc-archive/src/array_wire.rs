//! Typed inline-array codec shared by live plugin messages and offline services.
//! Shared-memory ownership remains the plugin host's responsibility.
use std::any::Any;
use std::sync::Arc;
use base64::Engine;
use serde_json::{Value, json};
use zlc_core::{AnyBuffer, DType, ForeignBytes};
pub struct Array { pub values: AnyBuffer, pub shape: Vec<usize> }

pub fn dtype(text: &str) -> Result<DType, String> {
    Ok(match text {
        "|b1" => DType::Bool, "|u1" => DType::U8, "|i1" => DType::I8, "<u2" => DType::U16, "<i2" => DType::I16,
        "<u4" => DType::U32, "<i4" => DType::I32, "<u8" => DType::U64, "<i8" => DType::I64, "<f2" => DType::F16,
        "<f4" => DType::F32, "<f8" => DType::F64, "<c8" => DType::C64, "<c16" => DType::C128,
        other => return Err(format!("Unsupported array dtype {other:?}")),
    })
}

pub fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "|b1", DType::U8 => "|u1", DType::I8 => "|i1", DType::U16 => "<u2", DType::I16 => "<i2",
        DType::U32 => "<u4", DType::I32 => "<i4", DType::U64 => "<u8", DType::I64 => "<i8", DType::F16 => "<f2",
        DType::F32 => "<f4", DType::F64 => "<f8", DType::C64 => "<c8", DType::C128 => "<c16",
    }
}

struct Inline(Vec<u128>, usize);
unsafe impl ForeignBytes for Inline {
    fn bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast(), self.1) } }
    fn as_any(&self) -> &dyn Any { self }
}

pub fn decode(reference: &Value) -> Result<Array, String> {
    let spec = reference.get("$inline").ok_or("expected an inline array")?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(spec["b64"].as_str().ok_or("inline array has no bytes")?)
        .map_err(|e| e.to_string())?;
    let shape: Vec<usize> = serde_json::from_value(spec["shape"].clone()).map_err(|e| e.to_string())?;
    let count = shape.iter().try_fold(1usize, |size, dim| size.checked_mul(*dim).ok_or("array shape overflows"))?;
    let kind = dtype(spec["dtype"].as_str().ok_or("array has no dtype")?)?;
    let mut words = vec![0u128; bytes.len().div_ceil(16)];
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast::<u8>(), bytes.len()); }
    let owner: Arc<dyn ForeignBytes> = Arc::new(Inline(words, bytes.len()));
    Ok(Array { values: AnyBuffer::from_foreign(kind, owner, 0, count)?, shape })
}

pub fn encode(values: &AnyBuffer, shape: &[usize]) -> Result<Value, String> {
    Ok(json!({"$inline": {"b64":base64::engine::general_purpose::STANDARD.encode(values.as_bytes()),
        "dtype":dtype_name(values.dtype()), "shape":shape}}))
}
