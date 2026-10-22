//! Validated immutable scientific planes. Identity belongs to their publication.
use std::sync::Arc;
use crate::{AnyBuffer, DType, OwnedBuffer};
use crate::schema::DatasetSchema;

#[derive(Clone)]
pub enum Validity {
    Uniform(bool),
    Mask(OwnedBuffer<bool>),
}

pub struct DataBlock {
    schema: Arc<DatasetSchema>,
    values: AnyBuffer,
    validity: Validity,
    sigma: Option<OwnedBuffer<f64>>,
}

impl DataBlock {
    pub fn new(schema: Arc<DatasetSchema>, values: AnyBuffer, validity: Validity,
               sigma: Option<OwnedBuffer<f64>>) -> Result<Arc<Self>, String> {
        if values.dtype() != schema.dtype() || values.len() != schema.value_len() {
            return Err("Values do not match their declared dtype and physical shape".into());
        }
        if let Validity::Mask(mask) = &validity {
            if mask.len() != schema.component_mask_len() {
                return Err("Validity mask does not match the schema's named component layout".into());
            }
        }
        if let Some(sigma) = &sigma {
            if sigma.len() != values.len() || sigma.as_slice().iter().any(|v| v.is_finite() && *v < 0.0) {
                return Err("Sample sigma must match values and contain no finite negative errors".into());
            }
        }
        Ok(Arc::new(Self { schema, values, validity, sigma }))
    }

    pub fn schema(&self) -> &Arc<DatasetSchema> { &self.schema }
    /// Equal incoming metadata adopts the declaration's one canonical schema.
    /// Already validated immutable planes need neither another scan nor a copy.
    pub fn with_schema(self: &Arc<Self>, schema: Arc<DatasetSchema>) -> Result<Arc<Self>, String> {
        if Arc::ptr_eq(&self.schema,&schema) { return Ok(self.clone()); }
        if self.schema != schema { return Err("Dataset schema differs from its declaration".into()); }
        Ok(Arc::new(Self { schema, values: self.values.clone(), validity: self.validity.clone(), sigma: self.sigma.clone() }))
    }
    pub fn dtype(&self) -> DType { self.values.dtype() }
    pub fn values(&self) -> &AnyBuffer { &self.values }
    pub fn validity(&self) -> &Validity { &self.validity }
    pub fn sigma(&self) -> Option<&OwnedBuffer<f64>> { self.sigma.as_ref() }
    pub fn payload_bytes(&self) -> usize {
        self.values.byte_len() + self.sigma.as_ref().map_or(0, OwnedBuffer::byte_len)
            + match &self.validity { Validity::Uniform(_) => 0, Validity::Mask(mask) => mask.byte_len() }
    }
}
