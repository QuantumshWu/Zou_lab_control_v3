//! atom-motion: the 3D semiclassical motion solver (CPU). The specification is each module's own documentation;
//! SPEC.md beside this crate is its index, by the sections the code cites ("SPEC §x"), with the principles, what is
//! not done and the known limits.

pub mod units;
pub mod linalg;
mod special;
pub mod random;
pub mod request;
pub mod field;
pub mod scene;
pub mod floquet;
#[cfg(test)]
pub mod krylov;
pub mod select;
pub mod model;
pub mod spectral;
pub mod step;
pub mod adapt;
pub mod ensemble;
pub mod estimate;
pub mod decide;
pub mod ledger;
pub mod analysis;
pub mod paths;
pub mod solve;
pub mod prof;
