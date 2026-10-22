//! Arrays and datasets on the wire (PLUGIN_PROTOCOL.md §6), core side.
use std::any::Any;
use std::sync::Arc;

use base64::Engine;
use serde_json::{Value, json};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::schema::{DatasetSchema, DomainRole, ValidityContract};
use zlc_core::{AnyBuffer, DType, ForeignBytes};

use crate::blocks::{Lend, Registry};

/// Arrays smaller than this are sent inline (as the SDK does).
pub const INLINE_BYTES: usize = 16 << 10;

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

/// Inline bytes kept 16-byte aligned, so any dtype can borrow them.
struct Inline(Vec<u128>, usize);

unsafe impl ForeignBytes for Inline {
    fn bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast(), self.1) } }
    fn as_any(&self) -> &dyn Any { self }
}

/// One decoded array: its values (borrowed from a block when lent) and shape.
pub struct Array { pub values: AnyBuffer, pub shape: Vec<usize> }

fn shape_of(spec: &Value) -> Result<Vec<usize>, String> {
    spec["shape"].as_array().ok_or("Array has no shape")?.iter()
        .map(|item| item.as_u64().map(|value| value as usize).ok_or_else(|| "Array shape must be counts".to_string())).collect()
}

/// Decode `{"$array": ...}` (borrowing the message's lend) or `{"$inline": ...}`.
pub fn array(reference: &Value, lends: &[Arc<Lend>]) -> Result<Array, String> {
    if let Some(spec) = reference.get("$array") {
        let name = spec["block"].as_str().ok_or("Array reference has no block")?;
        let lend = lends.iter().find(|lend| lend.name() == name)
            .ok_or_else(|| format!("Block {name} was not lent by this message"))?;
        let kind = dtype(spec["dtype"].as_str().ok_or("Array has no dtype")?)?;
        let shape = shape_of(spec)?;
        let count = shape.iter().product::<usize>();
        let offset = spec["offset"].as_u64().ok_or("Array has no offset")? as usize;
        let owner: Arc<dyn ForeignBytes> = lend.clone();
        return Ok(Array { values: AnyBuffer::from_foreign(kind, owner, offset, count)?, shape });
    }
    if let Some(spec) = reference.get("$inline") {
        let bytes = base64::engine::general_purpose::STANDARD.decode(spec["b64"].as_str().ok_or("Inline array has no data")?)
            .map_err(|error| format!("Inline array: {error}"))?;
        let kind = dtype(spec["dtype"].as_str().ok_or("Array has no dtype")?)?;
        let shape = shape_of(spec)?;
        let count = shape.iter().product::<usize>();
        let mut words = vec![0u128; bytes.len().div_ceil(16)];
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast::<u8>(), bytes.len()) };
        let owner: Arc<dyn ForeignBytes> = Arc::new(Inline(words, bytes.len()));
        return Ok(Array { values: AnyBuffer::from_foreign(kind, owner, 0, count)?, shape });
    }
    Err(format!("Expected an array, got {reference}"))
}

/// Encode values for plugin `holder`: inline when small, the same block when the
/// core borrowed them from one, else a core block. Each `$array` is one hold.
pub fn encode_array(values: &AnyBuffer, shape: &[usize], registry: &Arc<Registry>, holder: u32) -> Result<Value, String> {
    let kind = dtype_name(values.dtype());
    let bytes = values.as_bytes();
    if bytes.len() < INLINE_BYTES {
        return Ok(json!({"$inline": {"b64": base64::engine::general_purpose::STANDARD.encode(bytes), "dtype": kind, "shape": shape}}));
    }
    let borrowed = values.foreign().and_then(|(owner, offset)| owner.as_any().downcast_ref::<Lend>().map(|lend| (lend, offset)));
    let (name, size, offset) = match borrowed {
        Some((lend, offset)) => {
            registry.hold_for(holder, lend.name())?;
            (lend.name().to_owned(), lend.bytes().len(), offset)
        }
        None => {
            let exported = registry.export(bytes)?;
            registry.hold_for(holder, exported.name())?;
            (exported.name().to_owned(), exported.bytes().len(), 0)
        }
    };
    Ok(json!({"$array": {"block": name, "size": size, "offset": offset, "dtype": kind, "shape": shape}}))
}

/// The compact validity layout: (Repeat, Point, *declared component axes).
fn mask_shape(schema: &DatasetSchema) -> Vec<usize> {
    let mut shape = vec![schema.domain(DomainRole::Repeat).size(), schema.domain(DomainRole::Point).size()];
    if let ValidityContract::Components(ids) = schema.value().validity() {
        let cell = schema.domain(DomainRole::Cell);
        for axis in cell.axes() {
            if ids.iter().any(|id| id.as_ref() == axis.id()) { shape.push(axis.len()); }
        }
    }
    shape
}

pub fn encode_block(block: &DataBlock, registry: &Arc<Registry>, holder: u32) -> Result<Value, String> {
    let shape = block.schema().physical_shape();
    let validity = match block.validity() {
        Validity::Uniform(value) => json!(value),
        Validity::Mask(mask) => encode_array(&AnyBuffer::from(mask.clone()), &mask_shape(block.schema()), registry, holder)?,
    };
    let sigma = match block.sigma() {
        Some(sigma) => encode_array(&AnyBuffer::from(sigma.clone()), shape, registry, holder)?,
        None => Value::Null,
    };
    Ok(json!({"values": encode_array(block.values(), shape, registry, holder)?, "validity": validity, "sigma": sigma}))
}

pub fn decode_block(data: &Value, schema: Arc<DatasetSchema>, lends: &[Arc<Lend>]) -> Result<Arc<DataBlock>, String> {
    let values = array(&data["values"], lends)?;
    if values.shape != schema.physical_shape() {
        return Err(format!("Values of shape {:?} do not match the declared {:?}", values.shape, schema.physical_shape()));
    }
    let validity = match &data["validity"] {
        Value::Bool(value) => Validity::Uniform(*value),
        Value::Null => Validity::Uniform(true),
        reference => match array(reference, lends)?.values {
            AnyBuffer::Bool(mask) => Validity::Mask(mask),
            _ => return Err("A validity mask must be bool".into()),
        },
    };
    let sigma = match &data["sigma"] {
        Value::Null => None,
        reference => match array(reference, lends)?.values {
            AnyBuffer::F64(sigma) => Some(sigma),
            _ => return Err("Sample sigma must be float64".into()),
        },
    };
    DataBlock::new(schema, values.values, validity, sigma)
}

/// One complete dataset with its identity (`zlc_plugin.figures.dataset_to_wire`).
pub struct Dataset { pub block: Arc<DataBlock>, pub block_id: String, pub revision: u64, pub generation: String }

pub fn decode_dataset(dataset: &Value, lends: &[Arc<Lend>]) -> Result<Dataset, String> {
    let text = |name: &str| dataset[name].as_str().map(str::to_owned).ok_or_else(|| format!("A dataset needs its {name}"));
    let block = decode_block(&dataset["data"], schema_from_tree(&dataset["schema"])?, lends)?;
    let revision = dataset["revision"].as_u64().ok_or("A dataset needs its revision")?;
    Ok(Dataset { block, block_id: text("block_id")?, revision, generation: text("generation")? })
}

/// The DatasetSchema tree (`zlc_data.codec`) as JSON, and back.
pub fn schema_tree(schema: &DatasetSchema) -> Value { zlc_archive::tree::schema_tree(schema).to_serde() }

pub fn schema_from_tree(tree: &Value) -> Result<Arc<DatasetSchema>, String> {
    zlc_archive::tree::schema_from_tree(&zlc_archive::json::Json::from_serde(tree)).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zlc_core::OwnedBuffer;

    #[test]
    fn inline_arrays_round_trip_with_their_dtype() {
        let registry = Registry::new("wire".into(), Box::new(|_, _| {}));
        let values = AnyBuffer::from(OwnedBuffer::from_vec(vec![1.5f64, -2.0, 3.25]));
        let encoded = encode_array(&values, &[3], &registry, 1).unwrap();
        let decoded = array(&encoded, &[]).unwrap();
        assert_eq!(decoded.shape, vec![3]);
        let AnyBuffer::F64(back) = decoded.values else { panic!("dtype changed") };
        assert_eq!(back.as_slice(), &[1.5, -2.0, 3.25]);
        // Large core values travel in a core block, held once for the receiver.
        let large = AnyBuffer::from(OwnedBuffer::from_vec(vec![5u8; INLINE_BYTES * 2]));
        let encoded = encode_array(&large, &[INLINE_BYTES * 2], &registry, 9).unwrap();
        let name = encoded["$array"]["block"].as_str().unwrap();
        assert_eq!(registry.holds(name), 1);
        registry.released_by(9, &[name.to_owned()]);
        assert_eq!(registry.holds(name), 0);
    }
}
