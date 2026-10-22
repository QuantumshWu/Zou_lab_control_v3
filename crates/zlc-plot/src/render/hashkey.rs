//! A value's identity for the renderer's caches: its serde form fed to a
//! hasher as binary (no text formatting), so an IR of a few thousand nodes
//! keys in a fraction of a millisecond.
use std::hash::Hasher;

use serde::ser::{self, Serialize};

/// Two independent multiply-rotate lanes over the bytes, mixed at the end:
/// a cache key, not a defence against chosen collisions.
#[derive(Clone)]
pub struct Lanes { a: u64, b: u64, length: u64 }

impl Lanes {
    pub fn new() -> Lanes { Lanes { a: 0x243f_6a88_85a3_08d3, b: 0x1319_8a2e_0370_7344, length: 0 } }

    /// Both lanes, each finished on its own: a 128-bit key for caches whose
    /// hit replaces work outright (no comparison of the full value behind it).
    pub fn finish128(&self) -> u128 {
        let mix = |mut h: u64| {
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            h ^ (h >> 33)
        };
        (u128::from(mix(self.a ^ self.length)) << 64) | u128::from(mix(self.b.rotate_left(32) ^ self.length))
    }

    pub fn f64(&mut self, v: f64) { self.write_u64(v.to_bits()); }

    pub fn f64s(&mut self, values: &[f64]) {
        self.write_usize(values.len());
        for &v in values { self.write_u64(v.to_bits()); }
    }

    #[inline(always)]
    fn word(&mut self, w: u64) {
        self.a = (self.a ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
        self.b = (self.b.rotate_left(17) ^ w).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    }
}

impl Hasher for Lanes {
    fn write(&mut self, bytes: &[u8]) {
        self.length = self.length.wrapping_add(bytes.len() as u64);
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks { self.word(u64::from_le_bytes(chunk.try_into().unwrap())); }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.word(u64::from_le_bytes(last) ^ ((rest.len() as u64) << 56));
        }
    }

    fn write_u64(&mut self, w: u64) { self.length = self.length.wrapping_add(8); self.word(w); }

    fn finish(&self) -> u64 {
        let mut h = self.a ^ self.b.rotate_left(32) ^ self.length;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }
}

/// The 64-bit key of `value`'s serialized form.
pub fn hash_of<T: Serialize + ?Sized>(value: &T) -> u64 {
    let mut hasher = KeyHasher::new(None);
    value.serialize(&mut hasher).expect("hashable");
    hasher.lanes.finish()
}

/// The key of a figure IR as it draws: its node references stand for the
/// nodes they name (each node's hash over its own fields and children), so
/// the same picture keys alike however many nodes other artists made before
/// its own; nodes nothing in it references are no part of it.
pub fn figure_key(figure: &zlc_figure::ir::Figure) -> u64 { keyed(&node_hashes(figure), figure) }

/// `value`'s key, its node references standing for `nodes` (a figure's [`node_hashes`]).
pub fn keyed<T: Serialize + ?Sized>(nodes: &[u64], value: &T) -> u64 {
    let mut hasher = KeyHasher::new(Some(nodes));
    value.serialize(&mut hasher).expect("hashable");
    hasher.lanes.finish()
}

/// Every node's structural hash: its own fields, the nodes it names (by
/// their hashes) and the figure state it reads -- an axes' frame (positions,
/// aspect, view limits) or the figure's size and dpi -- so a node hash alone
/// says what the node evaluates to.
pub fn node_hashes(figure: &zlc_figure::ir::Figure) -> Vec<u64> {
    use zlc_figure::ir::{BboxNode, Node, TransformNode};
    let frames: Vec<u64> = figure.axes.iter().map(|a| {
        let mut k = Key::new();
        k.add(&a.axes_locator).add(&a.position_original).add(&a.position_active).add(&a.apply_aspect).add(&a.aspect)
            .add(&a.box_aspect).add(&a.adjustable).add(&a.anchor).add(&a.view_lim);
        k.finish()
    }).collect();
    let mut nodes: Vec<u64> = Vec::with_capacity(figure.nodes.len());
    for node in &figure.nodes {
        // A node names only earlier ones; a later index hashes as a plain number.
        let mut hasher = KeyHasher::new(Some(&nodes));
        node.serialize(&mut hasher).expect("hashable");
        match node {
            Node::Bbox(BboxNode::AxesPosition { axes } | BboxNode::ViewLim { axes }) => {
                hasher.lanes.write_u64(frames.get(*axes).copied().unwrap_or(u64::MAX));
            }
            Node::Bbox(BboxNode::FigureInches {}) | Node::Transform(TransformNode::DpiScale {}) => {
                hasher.lanes.write_u64(figure.dpi.to_bits());
                hasher.lanes.write_u64(figure.size_inches[0].to_bits());
                hasher.lanes.write_u64(figure.size_inches[1].to_bits());
            }
            _ => {}
        }
        let hash = hasher.lanes.finish();
        nodes.push(hash);
    }
    nodes
}

/// Feed several values to one key.
pub struct Key(KeyHasher<'static>);

impl Key {
    pub fn new() -> Key { Key(KeyHasher::new(None)) }
    pub fn add<T: Serialize + ?Sized>(&mut self, value: &T) -> &mut Key {
        value.serialize(&mut self.0).expect("hashable");
        self
    }
    pub fn finish(&self) -> u64 { self.0.lanes.finish() }
}

impl Default for Key {
    fn default() -> Key { Key::new() }
}

impl Default for Lanes {
    fn default() -> Lanes { Lanes::new() }
}

/// The serializer feeding [`Lanes`]. With a node table, every node
/// reference (an IR `u32` other than `version`) hashes as the node's own
/// structural hash, and the figure's `nodes` field is not hashed itself.
pub struct KeyHasher<'n> { lanes: Lanes, nodes: Option<&'n [u64]>, field: &'static str }

impl<'n> KeyHasher<'n> {
    fn new(nodes: Option<&'n [u64]>) -> KeyHasher<'n> { KeyHasher { lanes: Lanes::new(), nodes, field: "" } }
}

#[derive(Debug)]
pub struct Unhashable;

impl std::fmt::Display for Unhashable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("unhashable") }
}

impl std::error::Error for Unhashable {}

impl ser::Error for Unhashable {
    fn custom<T: std::fmt::Display>(_: T) -> Self { Unhashable }
}

macro_rules! bytes_of {
    ($($name:ident: $t:ty),*) => {
        $(fn $name(self, v: $t) -> Result<(), Unhashable> { self.lanes.write(&v.to_le_bytes()); Ok(()) })*
    };
}

impl ser::Serializer for &mut KeyHasher<'_> {
    type Ok = ();
    type Error = Unhashable;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    bytes_of!(serialize_i8: i8, serialize_i16: i16, serialize_i32: i32, serialize_i128: i128,
              serialize_u8: u8, serialize_u16: u16, serialize_u128: u128, serialize_f32: f32);

    fn serialize_u32(self, v: u32) -> Result<(), Unhashable> {
        match self.nodes {
            Some(nodes) if self.field != "version" && (v as usize) < nodes.len() => self.lanes.write_u64(nodes[v as usize]),
            _ => self.lanes.write(&v.to_le_bytes()),
        }
        Ok(())
    }

    fn serialize_i64(self, v: i64) -> Result<(), Unhashable> { self.lanes.write_u64(v as u64); Ok(()) }
    fn serialize_u64(self, v: u64) -> Result<(), Unhashable> { self.lanes.write_u64(v); Ok(()) }
    fn serialize_f64(self, v: f64) -> Result<(), Unhashable> { self.lanes.write_u64(v.to_bits()); Ok(()) }

    fn serialize_bool(self, v: bool) -> Result<(), Unhashable> { self.lanes.write_u8(u8::from(v)); Ok(()) }
    fn serialize_char(self, v: char) -> Result<(), Unhashable> { self.lanes.write_u32(v as u32); Ok(()) }
    fn serialize_str(self, v: &str) -> Result<(), Unhashable> { self.lanes.write_usize(v.len()); self.lanes.write(v.as_bytes()); Ok(()) }
    fn serialize_bytes(self, v: &[u8]) -> Result<(), Unhashable> { self.lanes.write_usize(v.len()); self.lanes.write(v); Ok(()) }
    fn serialize_none(self) -> Result<(), Unhashable> { self.lanes.write_u8(0); Ok(()) }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), Unhashable> { self.lanes.write_u8(1); value.serialize(self) }
    fn serialize_unit(self) -> Result<(), Unhashable> { self.lanes.write_u8(2); Ok(()) }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Unhashable> { self.lanes.write_u8(3); Ok(()) }
    fn serialize_unit_variant(self, _: &'static str, index: u32, _: &'static str) -> Result<(), Unhashable> {
        self.lanes.write_u32(index);
        Ok(())
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _: &'static str, value: &T) -> Result<(), Unhashable> { value.serialize(self) }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(self, _: &'static str, index: u32, _: &'static str, value: &T)
        -> Result<(), Unhashable> {
        self.lanes.write_u32(index);
        value.serialize(self)
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self, Unhashable> { self.lanes.write_usize(len.unwrap_or(usize::MAX)); Ok(self) }
    fn serialize_tuple(self, len: usize) -> Result<Self, Unhashable> { self.lanes.write_usize(len); Ok(self) }
    fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Self, Unhashable> { self.lanes.write_usize(len); Ok(self) }
    fn serialize_tuple_variant(self, _: &'static str, index: u32, _: &'static str, _: usize) -> Result<Self, Unhashable> {
        self.lanes.write_u32(index);
        Ok(self)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self, Unhashable> { self.lanes.write_usize(len.unwrap_or(usize::MAX)); Ok(self) }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, Unhashable> { Ok(self) }
    fn serialize_struct_variant(self, _: &'static str, index: u32, _: &'static str, _: usize) -> Result<Self, Unhashable> {
        self.lanes.write_u32(index);
        Ok(self)
    }
}

macro_rules! compound {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for &mut KeyHasher<'_> {
            type Ok = ();
            type Error = Unhashable;
            fn $method<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Unhashable> { value.serialize(&mut **self) }
            fn end(self) -> Result<(), Unhashable> { self.lanes.write_u8(0xff); Ok(()) }
        }
    };
}

compound!(SerializeSeq, serialize_element);
compound!(SerializeTuple, serialize_element);
compound!(SerializeTupleStruct, serialize_field);
compound!(SerializeTupleVariant, serialize_field);

impl ser::SerializeMap for &mut KeyHasher<'_> {
    type Ok = ();
    type Error = Unhashable;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Unhashable> { key.serialize(&mut **self) }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Unhashable> { value.serialize(&mut **self) }
    fn end(self) -> Result<(), Unhashable> { self.lanes.write_u8(0xfe); Ok(()) }
}

impl ser::SerializeStruct for &mut KeyHasher<'_> {
    type Ok = ();
    type Error = Unhashable;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<(), Unhashable> {
        if self.nodes.is_some() && key == "nodes" { return Ok(()); }
        self.field = key;
        self.lanes.write(key.as_bytes());
        value.serialize(&mut **self)
    }
    fn end(self) -> Result<(), Unhashable> { self.lanes.write_u8(0xfd); Ok(()) }
}

impl ser::SerializeStructVariant for &mut KeyHasher<'_> {
    type Ok = ();
    type Error = Unhashable;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<(), Unhashable> {
        self.lanes.write(key.as_bytes());
        value.serialize(&mut **self)
    }
    fn end(self) -> Result<(), Unhashable> { self.lanes.write_u8(0xfc); Ok(()) }
}
