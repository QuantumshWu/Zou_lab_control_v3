//! Serde helpers for the IR's wire form: every float is its IEEE-754 bit
//! pattern (a JSON integer), so a round trip through JSON is exact and NaN or
//! infinities need no special spelling. Arrays are `{dtype, shape, data}`
//! (inline) or `{dtype, shape, blob}` (a content-addressed file of the raw
//! little-endian C-order bytes under the directory set by [`set_blob_root`]).
use std::cell::RefCell;
use std::path::PathBuf;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

thread_local! {
    static BLOB_ROOT: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Where `blob` arrays are read from while deserializing on this thread.
pub fn set_blob_root(root: Option<PathBuf>) {
    BLOB_ROOT.with(|cell| *cell.borrow_mut() = root);
}

fn read_blob(digest: &str) -> Result<Vec<u8>, String> {
    BLOB_ROOT.with(|cell| {
        let root = cell.borrow();
        let root = root.as_ref().ok_or_else(|| format!("blob {digest} but no blob root is set"))?;
        let path = root.join(&digest[..2]).join(format!("{digest}.bin"));
        std::fs::read(&path).map_err(|error| format!("blob {}: {error}", path.display()))
    })
}

pub mod f {
    use super::*;
    pub fn serialize<S: Serializer>(value: &f64, s: S) -> Result<S::Ok, S::Error> { value.to_bits().serialize(s) }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> { Ok(f64::from_bits(u64::deserialize(d)?)) }
}

pub mod of {
    use super::*;
    pub fn serialize<S: Serializer>(value: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
        value.map(f64::to_bits).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
        Ok(Option::<u64>::deserialize(d)?.map(f64::from_bits))
    }
}

pub mod vf {
    use super::*;
    pub fn serialize<S: Serializer>(value: &[f64], s: S) -> Result<S::Ok, S::Error> {
        value.iter().map(|v| v.to_bits()).collect::<Vec<_>>().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        Ok(Vec::<u64>::deserialize(d)?.into_iter().map(f64::from_bits).collect())
    }
}

pub mod ovf {
    use super::*;
    pub fn serialize<S: Serializer>(value: &Option<Vec<f64>>, s: S) -> Result<S::Ok, S::Error> {
        value.as_ref().map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<f64>>, D::Error> {
        Ok(Option::<Vec<u64>>::deserialize(d)?.map(|v| v.into_iter().map(f64::from_bits).collect()))
    }
}

/// `[f64; N]` as N bit patterns.
pub mod fa {
    use super::*;
    pub fn serialize<S: Serializer, const N: usize>(value: &[f64; N], s: S) -> Result<S::Ok, S::Error> {
        value.iter().map(|v| v.to_bits()).collect::<Vec<_>>().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(d: D) -> Result<[f64; N], D::Error> {
        let raw = Vec::<u64>::deserialize(d)?;
        let values: Vec<f64> = raw.into_iter().map(f64::from_bits).collect();
        values.try_into().map_err(|v: Vec<f64>| D::Error::custom(format!("expected {N} floats, got {}", v.len())))
    }
}

pub mod ofa {
    use super::*;
    pub fn serialize<S: Serializer, const N: usize>(value: &Option<[f64; N]>, s: S) -> Result<S::Ok, S::Error> {
        value.map(|v| v.map(f64::to_bits).to_vec()).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(d: D) -> Result<Option<[f64; N]>, D::Error> {
        match Option::<Vec<u64>>::deserialize(d)? {
            None => Ok(None),
            Some(raw) => {
                let values: Vec<f64> = raw.into_iter().map(f64::from_bits).collect();
                values.try_into().map(Some).map_err(|v: Vec<f64>| D::Error::custom(format!("expected {N} floats, got {}", v.len())))
            }
        }
    }
}

/// The element type of an array on the wire (numpy's dtype).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    #[serde(rename = "f8")] F64,
    #[serde(rename = "f4")] F32,
    #[serde(rename = "u1")] U8,
    #[serde(rename = "u2")] U16,
    #[serde(rename = "u4")] U32,
    #[serde(rename = "u8")] U64,
    #[serde(rename = "i1")] I8,
    #[serde(rename = "i2")] I16,
    #[serde(rename = "i4")] I32,
    #[serde(rename = "i8")] I64,
    #[serde(rename = "b1")] Bool,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F64 | DType::U64 | DType::I64 => 8,
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::U16 | DType::I16 => 2,
            DType::U8 | DType::I8 | DType::Bool => 1,
        }
    }
}

/// An n-dimensional array, its elements kept as raw little-endian bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct NdArray { pub dtype: DType, pub shape: Vec<usize>, pub bytes: Vec<u8> }

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireArray {
    dtype: DType,
    shape: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    blob: Option<String>,
}

impl NdArray {
    pub fn len(&self) -> usize { self.shape.iter().product() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn from_f64(shape: Vec<usize>, values: &[f64]) -> Self {
        NdArray { dtype: DType::F64, shape, bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect() }
    }

    /// Every element as f64 (numpy's `astype(float)` for the integer and bool types).
    /// Element `k` (flat, C order) as f64, without materializing the array.
    #[inline]
    pub fn get_f64(&self, k: usize) -> f64 {
        let b = &self.bytes;
        match self.dtype {
            DType::F64 => f64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap()),
            DType::F32 => f64::from(f32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().unwrap())),
            DType::U8 | DType::Bool => f64::from(b[k]),
            DType::I8 => f64::from(b[k] as i8),
            DType::U16 => f64::from(u16::from_le_bytes(b[k * 2..k * 2 + 2].try_into().unwrap())),
            DType::I16 => f64::from(i16::from_le_bytes(b[k * 2..k * 2 + 2].try_into().unwrap())),
            DType::U32 => f64::from(u32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().unwrap())),
            DType::I32 => f64::from(i32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().unwrap())),
            DType::U64 => u64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap()) as f64,
            DType::I64 => i64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap()) as f64,
        }
    }

    pub fn to_f64(&self) -> Vec<f64> {
        let b = &self.bytes;
        match self.dtype {
            DType::F64 => b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
            DType::F32 => b.chunks_exact(4).map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap()))).collect(),
            DType::U8 | DType::Bool => b.iter().map(|&v| f64::from(v)).collect(),
            DType::I8 => b.iter().map(|&v| f64::from(v as i8)).collect(),
            DType::U16 => b.chunks_exact(2).map(|c| f64::from(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
            DType::I16 => b.chunks_exact(2).map(|c| f64::from(i16::from_le_bytes(c.try_into().unwrap()))).collect(),
            DType::U32 => b.chunks_exact(4).map(|c| f64::from(u32::from_le_bytes(c.try_into().unwrap()))).collect(),
            DType::I32 => b.chunks_exact(4).map(|c| f64::from(i32::from_le_bytes(c.try_into().unwrap()))).collect(),
            DType::U64 => b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            DType::I64 => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        }
    }

    pub fn to_bool(&self) -> Vec<bool> {
        match self.dtype {
            DType::Bool | DType::U8 | DType::I8 => self.bytes.iter().map(|&v| v != 0).collect(),
            _ => self.to_f64().into_iter().map(|v| v != 0.0).collect(),
        }
    }
}

impl Serialize for NdArray {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let size = self.dtype.size();
        let data = self.bytes.chunks_exact(size).map(|c| match (self.dtype, size) {
            (DType::F64, _) => u64::from_le_bytes(c.try_into().unwrap()),
            (DType::F32, _) => u64::from(u32::from_le_bytes(c.try_into().unwrap())),
            (DType::I8, _) => (c[0] as i8) as i64 as u64,
            (DType::I16, _) => i16::from_le_bytes(c.try_into().unwrap()) as i64 as u64,
            (DType::I32, _) => i32::from_le_bytes(c.try_into().unwrap()) as i64 as u64,
            (DType::I64, _) => i64::from_le_bytes(c.try_into().unwrap()) as u64,
            (_, 1) => u64::from(c[0]),
            (_, 2) => u64::from(u16::from_le_bytes(c.try_into().unwrap())),
            (_, 4) => u64::from(u32::from_le_bytes(c.try_into().unwrap())),
            _ => u64::from_le_bytes(c.try_into().unwrap()),
        }).collect();
        WireArray { dtype: self.dtype, shape: self.shape.clone(), data: Some(data), blob: None }.serialize(s)
    }
}

impl<'de> Deserialize<'de> for NdArray {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = WireArray::deserialize(d)?;
        let count: usize = wire.shape.iter().product();
        let size = wire.dtype.size();
        let bytes = match (wire.data, wire.blob) {
            (Some(data), None) => {
                if data.len() != count { return Err(D::Error::custom("inline array length differs from its shape")); }
                let mut bytes = Vec::with_capacity(count * size);
                for v in data {
                    // Signed integers arrive as JSON integers that may be negative;
                    // they were read as u64 two's complement by serde only if
                    // non-negative, so the extractor writes them non-negative.
                    bytes.extend_from_slice(&v.to_le_bytes()[..size]);
                }
                bytes
            }
            (None, Some(digest)) => read_blob(&digest).map_err(D::Error::custom)?,
            _ => return Err(D::Error::custom("array needs exactly one of data and blob")),
        };
        if bytes.len() != count * size {
            return Err(D::Error::custom(format!("array bytes {} differ from shape {:?} x {size}", bytes.len(), wire.shape)));
        }
        Ok(NdArray { dtype: wire.dtype, shape: wire.shape, bytes })
    }
}

/// Typed views of wire arrays for IR fields: each takes exactly the dtype and
/// shape the extractor writes (f8, u1 or b1; 1-d, (n, N) or (n, 3, 3)) and
/// fails the load on anything else, never converting or reshaping it.
pub mod arr {
    use super::*;

    /// `a` when its dtype is `dtype` and its shape passes `shape`.
    fn typed<E: serde::de::Error>(a: NdArray, dtype: DType, shape: impl Fn(&[usize]) -> bool, what: &str)
                                  -> Result<NdArray, E> {
        if a.dtype != dtype || !shape(&a.shape) {
            return Err(E::custom(format!("expected {what}, got {:?} {:?}", a.dtype, a.shape)));
        }
        Ok(a)
    }

    fn f64s_of<E: serde::de::Error>(a: NdArray) -> Result<Vec<f64>, E> {
        Ok(typed::<E>(a, DType::F64, |s| s.len() == 1, "a 1-d f8 array")?.to_f64())
    }

    fn u8s_of<E: serde::de::Error>(a: NdArray) -> Result<Vec<u8>, E> {
        Ok(typed::<E>(a, DType::U8, |s| s.len() == 1, "a 1-d u1 array")?.bytes)
    }

    fn rows_of<E: serde::de::Error, const N: usize>(a: NdArray) -> Result<Vec<[f64; N]>, E> {
        let a = typed::<E>(a, DType::F64, |s| s.len() == 2 && s[1] == N, &format!("an (n, {N}) f8 array"))?;
        Ok(a.to_f64().chunks_exact(N).map(|c| c.try_into().unwrap()).collect())
    }

    fn mat3s_of<E: serde::de::Error>(a: NdArray) -> Result<Vec<[[f64; 3]; 3]>, E> {
        let a = typed::<E>(a, DType::F64, |s| s.len() == 3 && s[1] == 3 && s[2] == 3, "an (n, 3, 3) f8 array")?;
        Ok(a.to_f64().chunks_exact(9).map(|c| [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]]).collect())
    }

    pub mod f64s {
        use super::*;
        pub fn serialize<S: Serializer>(value: &[f64], s: S) -> Result<S::Ok, S::Error> {
            NdArray::from_f64(vec![value.len()], value).serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
            f64s_of(NdArray::deserialize(d)?)
        }
    }

    pub mod of64s {
        use super::*;
        pub fn serialize<S: Serializer>(value: &Option<Vec<f64>>, s: S) -> Result<S::Ok, S::Error> {
            value.as_ref().map(|v| NdArray::from_f64(vec![v.len()], v)).serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<f64>>, D::Error> {
            Option::<NdArray>::deserialize(d)?.map(f64s_of).transpose()
        }
    }

    pub mod bools {
        use super::*;
        pub fn serialize<S: Serializer>(value: &[bool], s: S) -> Result<S::Ok, S::Error> {
            NdArray { dtype: DType::Bool, shape: vec![value.len()], bytes: value.iter().map(|&b| u8::from(b)).collect() }.serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<bool>, D::Error> {
            Ok(typed::<D::Error>(NdArray::deserialize(d)?, DType::Bool, |s| s.len() == 1, "a 1-d b1 array")?.to_bool())
        }
    }

    pub mod u8s {
        use super::*;
        pub fn serialize<S: Serializer>(value: &[u8], s: S) -> Result<S::Ok, S::Error> {
            NdArray { dtype: DType::U8, shape: vec![value.len()], bytes: value.to_vec() }.serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
            u8s_of(NdArray::deserialize(d)?)
        }
    }

    pub mod ou8s {
        use super::*;
        pub fn serialize<S: Serializer>(value: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
            value.as_ref().map(|v| NdArray { dtype: DType::U8, shape: vec![v.len()], bytes: v.clone() }).serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
            Option::<NdArray>::deserialize(d)?.map(u8s_of).transpose()
        }
    }

    /// Vec<[f64; N]> from an (n, N) array.
    pub mod rows {
        use super::*;
        pub fn serialize<S: Serializer, const N: usize>(value: &[[f64; N]], s: S) -> Result<S::Ok, S::Error> {
            let flat: Vec<f64> = value.iter().flatten().copied().collect();
            NdArray::from_f64(vec![value.len(), N], &flat).serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(d: D) -> Result<Vec<[f64; N]>, D::Error> {
            rows_of(NdArray::deserialize(d)?)
        }
    }

    pub mod orows {
        use super::*;
        pub fn serialize<S: Serializer, const N: usize>(value: &Option<Vec<[f64; N]>>, s: S) -> Result<S::Ok, S::Error> {
            value.as_ref().map(|v| {
                let flat: Vec<f64> = v.iter().flatten().copied().collect();
                NdArray::from_f64(vec![v.len(), N], &flat)
            }).serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(d: D) -> Result<Option<Vec<[f64; N]>>, D::Error> {
            Option::<NdArray>::deserialize(d)?.map(rows_of).transpose()
        }
    }

    /// Vec<[[f64; 3]; 3]> from an (n, 3, 3) array.
    pub mod mat3s {
        use super::*;
        pub fn serialize<S: Serializer>(value: &[[[f64; 3]; 3]], s: S) -> Result<S::Ok, S::Error> {
            let flat: Vec<f64> = value.iter().flatten().flatten().copied().collect();
            NdArray::from_f64(vec![value.len(), 3, 3], &flat).serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[[f64; 3]; 3]>, D::Error> {
            mat3s_of(NdArray::deserialize(d)?)
        }
    }

    pub mod omat3s {
        use super::*;
        pub fn serialize<S: Serializer>(value: &Option<Vec<[[f64; 3]; 3]>>, s: S) -> Result<S::Ok, S::Error> {
            value.as_ref().map(|v| {
                let flat: Vec<f64> = v.iter().flatten().flatten().copied().collect();
                NdArray::from_f64(vec![v.len(), 3, 3], &flat)
            }).serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<[[f64; 3]; 3]>>, D::Error> {
            Option::<NdArray>::deserialize(d)?.map(mat3s_of).transpose()
        }
    }
}
