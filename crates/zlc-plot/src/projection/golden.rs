//! Reading golden suites (written by `tests/gen/goldens.py`) into native
//! inputs, and comparing a native payload with a recorded one bit for bit.
//! Used by the golden tests, the benches and integration tooling.

use std::sync::Arc;

use serde_json::Value;
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::schema::{Axis, CodeBase, CodeMapping, Coordinate, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::{AnyBuffer, DType as CoreDType, F16, OwnedBuffer};

use super::model::{Source, SourceData};
use super::nd::{Array, DType};
use super::output::GoldenArrays;
use super::plan::{HistogramProjection, parse_hex_f64};

/// One golden suite: the JSON document and the array blob it refers into.
pub struct Suite { pub document: Value, pub blob: Vec<u8>, pub compared: std::cell::Cell<(usize, usize)> }

impl Suite {
    /// A suite from its JSON document and array blob (already decompressed).
    pub fn from_bytes(document: &[u8], blob: Vec<u8>) -> Suite {
        Suite { document: serde_json::from_slice(document).expect("golden document"), blob, compared: std::cell::Cell::new((0, 0)) }
    }
    /// An uncompressed suite `<name>.json` + `<name>.bin`.
    pub fn read(path: &std::path::Path) -> Suite {
        Suite::from_bytes(&std::fs::read(path).expect("golden document"), std::fs::read(path.with_extension("bin")).expect("golden blob"))
    }
    pub fn bytes(&self, reference: &Value) -> (DType, Vec<usize>, &[u8]) {
        let a = &reference["a"];
        let dtype = DType::parse(a[0].as_str().unwrap()).unwrap();
        let shape: Vec<usize> = a[1].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let offset = a[2].as_u64().unwrap() as usize;
        let n: usize = shape.iter().product();
        (dtype, shape, &self.blob[offset..offset + n * dtype.itemsize()])
    }
    pub fn buffer(&self, reference: &Value) -> (AnyBuffer, Vec<usize>) {
        let (dtype, shape, bytes) = self.bytes(reference);
        macro_rules! typed { ($t:ty, $n:expr) => { OwnedBuffer::from_vec(bytes.chunks_exact($n).map(|c| <$t>::from_le_bytes(c.try_into().unwrap())).collect::<Vec<$t>>()).into() }; }
        let buffer: AnyBuffer = match dtype {
            DType::Bool => OwnedBuffer::from_vec(bytes.iter().map(|b| *b != 0).collect::<Vec<bool>>()).into(),
            DType::U8 => typed!(u8, 1), DType::I8 => typed!(i8, 1), DType::U16 => typed!(u16, 2), DType::I16 => typed!(i16, 2),
            DType::U32 => typed!(u32, 4), DType::I32 => typed!(i32, 4), DType::U64 => typed!(u64, 8), DType::I64 => typed!(i64, 8),
            DType::F32 => typed!(f32, 4), DType::F64 => typed!(f64, 8),
            DType::F16 => OwnedBuffer::from_vec(bytes.chunks_exact(2).map(|c| F16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect::<Vec<F16>>()).into(),
            DType::Object => unreachable!(),
        };
        (buffer, shape)
    }
}

pub fn text(value: &Value) -> Option<Arc<str>> { value.as_str().map(Arc::from) }

pub fn coordinate(value: &Value) -> Coordinate {
    match value {
        Value::Null => Coordinate::Missing,
        Value::String(s) => Coordinate::Text(s.as_str().into()),
        Value::Number(n) => n.as_i64().map(|i| Coordinate::Integer(i as i128)).or_else(|| n.as_u64().map(|u| Coordinate::Integer(u as i128))).unwrap(),
        Value::Object(o) if o.contains_key("f") => Coordinate::Real(parse_hex_f64(o["f"].as_str().unwrap()).unwrap()),
        Value::Object(o) => Coordinate::Integer(o["i"].as_str().unwrap().parse().unwrap()),
        _ => panic!("bad coordinate {value}"),
    }
}

pub fn axis(suite: &Suite, document: &Value) -> Arc<Axis> {
    let size = document["size"].as_u64().unwrap() as usize;
    let coordinates = match (&document["coordinates"], &document["origins"]) {
        (Value::Null, _) => Coordinates::Index { origin: document["index_origin"].as_i64().unwrap() as i128, len: size },
        (c, Value::Object(_)) => {
            let AnyBuffer::F64(offsets) = suite.buffer(&c["array"]).0 else { panic!("offsets") };
            let AnyBuffer::F64(origins) = suite.buffer(&document["origins"]).0 else { panic!("origins") };
            Coordinates::OriginsAndOffsets { origins, offsets }
        }
        (c, _) if c.get("array").is_some() => Coordinates::Numeric(suite.buffer(&c["array"]).0),
        (c, _) => Coordinates::Values(c["values"].as_array().unwrap().iter().map(coordinate).collect::<Vec<_>>().into()),
    };
    let labels = document["labels"].as_array().map(|l| l.iter().map(|v| Arc::from(v.as_str().unwrap())).collect());
    Axis::new(document["id"].as_str().unwrap().into(), document["name"].as_str().unwrap().into(), document["role"].as_str().unwrap().into(),
        coordinates, text(&document["unit"]), text(&document["frame"]), labels, text(&document["coordinate_of"])).unwrap()
}

pub fn domain(suite: &Suite, document: &Value) -> Arc<Domain> {
    let shape: Vec<usize> = document["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let axes: Vec<Arc<Axis>> = document["axes"].as_array().unwrap().iter().map(|a| axis(suite, a)).collect();
    match document["codes"].as_array() {
        None => Domain::dense(shape, axes).unwrap(),
        Some(codes) => {
            let mappings = codes.iter().map(|code| {
                let base = if let Some(range) = code["base"]["range"].as_array() {
                    CodeBase::Range { start: range[0].as_u64().unwrap(), step: range[1].as_i64().unwrap(), len: range[2].as_u64().unwrap() as usize }
                } else {
                    let AnyBuffer::I64(values) = suite.buffer(&code["base"]["explicit"]).0 else { panic!("codes") };
                    CodeBase::Explicit(OwnedBuffer::from_vec(values.as_slice().iter().map(|v| *v as u64).collect()))
                };
                CodeMapping::new(base, code["inner"].as_u64().unwrap() as usize, code["outer"].as_u64().unwrap() as usize).unwrap()
            }).collect();
            Domain::new(shape, axes, mappings).unwrap()
        }
    }
}

pub fn core_dtype(name: &str) -> CoreDType {
    match name { "bool" => CoreDType::Bool, "u8" => CoreDType::U8, "i8" => CoreDType::I8, "u16" => CoreDType::U16, "i16" => CoreDType::I16,
        "u32" => CoreDType::U32, "i32" => CoreDType::I32, "u64" => CoreDType::U64, "i64" => CoreDType::I64, "f16" => CoreDType::F16,
        "f32" => CoreDType::F32, "f64" => CoreDType::F64, other => panic!("dtype {other}") }
}

pub fn validity(suite: &Suite, value: &Value) -> Validity {
    match value {
        Value::Bool(v) => Validity::Uniform(*v),
        reference => { let AnyBuffer::Bool(mask) = suite.buffer(reference).0 else { panic!("mask") }; Validity::Mask(mask) }
    }
}

pub fn sigma(suite: &Suite, value: &Value) -> Option<OwnedBuffer<f64>> {
    if value.is_null() { return None; }
    let AnyBuffer::F64(values) = suite.buffer(value).0 else { panic!("sigma") };
    Some(values)
}

pub fn source(suite: &Suite, document: &Value) -> Source {
    let schema_doc = &document["schema"];
    let value = &schema_doc["value"];
    let contract = match value["components"].as_array() {
        None => ValidityContract::Value,
        Some(ids) => ValidityContract::Components(ids.iter().map(|v| Arc::from(v.as_str().unwrap())).collect()),
    };
    let value_schema = ValueSchema::new(core_dtype(value["dtype"].as_str().unwrap()), text(&value["unit"]), text(&value["name"]), contract).unwrap();
    let cell = domain(suite, &schema_doc["cell"]);
    let schema = DatasetSchema::new(domain(suite, &schema_doc["repeat"]), domain(suite, &schema_doc["point"]), cell.clone(), value_schema.clone()).unwrap();
    let block = &document["block"];
    let data = if block["kind"] == "values" {
        SourceData::Dense(DataBlock::new(schema.clone(), suite.buffer(&block["values"]).0, validity(suite, &block["validity"]), sigma(suite, &block["sigma"])).unwrap())
    } else {
        SourceData::Segments(block["segments"].as_array().unwrap().iter().map(|segment| {
            let origin = [segment["origin"][0].as_u64().unwrap() as usize, segment["origin"][1].as_u64().unwrap() as usize];
            let extent = [segment["shape"][0].as_u64().unwrap() as usize, segment["shape"][1].as_u64().unwrap() as usize];
            let carrier = |id: &str, len: usize| Domain::dense(vec![len], vec![Axis::new(id.into(), id.into(), "generic".into(),
                Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap()]).unwrap();
            let segment_schema = DatasetSchema::new(carrier("zlc.golden.segment-repeat", extent[0]), carrier("zlc.golden.segment-point", extent[1]),
                cell.clone(), value_schema.clone()).unwrap();
            let data = DataBlock::new(segment_schema, suite.buffer(&segment["values"]).0, validity(suite, &segment["validity"]), sigma(suite, &segment["sigma"])).unwrap();
            (origin, data)
        }).collect())
    };
    let reference = &document["ref"];
    Source {
        schema, data, block_id: reference["block_id"].as_str().unwrap().into(), generation: reference["generation"].as_str().unwrap().into(),
        revision: reference["revision"].as_u64().unwrap(),
        window: document["window"].as_array().map(|w| (w[0].as_i64().unwrap() as i128, w[1].as_i64().unwrap() as i128)),
    }
}

pub fn histogram_state(value: &Value) -> Option<HistogramProjection> {
    if value.is_null() { return None; }
    let float = |v: &Value| parse_hex_f64(v["f"].as_str().unwrap()).unwrap();
    Some(HistogramProjection {
        bin_count: value["bin_count"].as_u64().unwrap() as usize, requested_bins: value["requested_bins"].as_u64().unwrap() as usize,
        domain: (float(&value["domain"][0]), float(&value["domain"][1])), edges: value["edges"].as_array().unwrap().iter().map(float).collect(),
    })
}

pub fn histogram_json(state: &Option<HistogramProjection>) -> Value {
    use super::output::hex_bits;
    match state {
        None => Value::Null,
        Some(h) => serde_json::json!({"bin_count": h.bin_count, "requested_bins": h.requested_bins,
            "domain": [{"f": hex_bits(h.domain.0)}, {"f": hex_bits(h.domain.1)}], "edges": h.edges.iter().map(|e| serde_json::json!({"f": hex_bits(*e)})).collect::<Vec<_>>()}),
    }
}

/// Compare a Python golden value with a Rust one; arrays bit for bit.
pub fn compare(suite: &Suite, arrays: &GoldenArrays, expected: &Value, actual: &Value, path: &str, out: &mut Vec<String>) {
    if out.len() >= 6 { return; }
    if let (Some(_), Some(index)) = (expected.get("a").or(expected.get("objects")), actual.get("rust")) {
        let array: &Array = &arrays.0[index.as_u64().unwrap() as usize];
        if let Some(objects) = expected.get("objects") {
            let shape: Vec<usize> = expected["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            let got: Vec<Value> = (0..array.len()).map(|i| super::output::scalar_json(&array.scalar(i))).collect();
            if array.dtype() != DType::Object || shape != array.shape() || objects.as_array().unwrap() != &got {
                out.push(format!("{path}: object array {:?} vs {} {:?}", objects, array.dtype().name(), array.shape()));
            }
            return;
        }
        let (dtype, shape, bytes) = suite.bytes(expected);
        let (count, size) = suite.compared.get();
        suite.compared.set((count + 1, size + bytes.len()));
        if dtype != array.dtype() || shape != array.shape() {
            out.push(format!("{path}: array {}{:?} vs {}{:?}", dtype.name(), shape, array.dtype().name(), array.shape()));
        } else if bytes != array.bytes() {
            let size = dtype.itemsize();
            let index = (0..array.len()).find(|i| bytes[i * size..(i + 1) * size] != array.bytes()[i * size..(i + 1) * size]).unwrap();
            let differing = (0..array.len()).filter(|i| bytes[i * size..(i + 1) * size] != array.bytes()[i * size..(i + 1) * size]).count();
            let python = Array::from_buffer(suite.buffer(expected).0, shape.clone());
            out.push(format!("{path}: {differing}/{} elements differ, first [{index}] python {:?} rust {:?}", array.len(),
                python.get_f64(index), array.get_f64(index)));
        }
        return;
    }
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => {
            for key in e.keys().chain(a.keys().filter(|k| !e.contains_key(*k))) {
                match (e.get(key), a.get(key)) {
                    (Some(x), Some(y)) => compare(suite, arrays, x, y, &format!("{path}.{key}"), out),
                    _ => out.push(format!("{path}.{key}: present on one side only")),
                }
            }
        }
        (Value::Array(e), Value::Array(a)) => {
            if e.len() != a.len() { out.push(format!("{path}: length {} vs {}", e.len(), a.len())); return; }
            for (i, (x, y)) in e.iter().zip(a).enumerate() { compare(suite, arrays, x, y, &format!("{path}[{i}]"), out); }
        }
        (e, a) => if e != a { out.push(format!("{path}: python {} rust {}", truncate(e), truncate(a))); },
    }
}

pub fn truncate(value: &Value) -> String { let s = value.to_string(); if s.len() > 160 { format!("{}...", &s[..160]) } else { s } }
