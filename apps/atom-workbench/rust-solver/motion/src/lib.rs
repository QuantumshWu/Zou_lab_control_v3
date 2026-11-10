//! atom-motion: the 3D semiclassical motion solver (CPU). The specification is SPEC.md beside this crate;
//! every quantity and decision here maps to one section of it.

pub mod units;
pub mod linalg;
mod special;
pub mod random;
pub mod request;
pub mod field;
pub mod model;
pub mod table;
pub mod step;
pub mod adapt;
pub mod ensemble;
pub mod estimate;
pub mod transport;
pub mod decide;
pub mod ledger;
pub mod solve;
