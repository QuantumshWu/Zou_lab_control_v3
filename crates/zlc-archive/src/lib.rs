//! Figure archives on the native data plane: the strict, pickle-free NPZ file
//! `zlc_data.figure_archive` writes and reads, byte for byte.
//!
//! * [`figure::write_figure_archive`] / [`figure::write_figure_file`] write one
//!   figure (named arrays and typed datasets beside one JSON metadata tree);
//!   [`figure::read_archive`] reads it back into the metadata, every member and
//!   native [`zlc_core::store::Snapshot`]s.
//! * [`manifest::ArchiveDataset`] is a dataset to write with the identity the
//!   scientific read had; [`manifest::ReadDataset`] is one read back.
//! * [`provenance::capture_figure_provenance`] is the lineage a saved figure
//!   carries; [`durable`] publishes files and folders crash-durably.
//! * [`json`] is Python's `json` (C scanner and encoder), [`npy`] numpy's NPY
//!   format, [`zip`] CPython's `zipfile`, [`deflate`] zlib 1.3.1 at level 1
//!   and [`inflate`] its inflate behind `zlib.decompressobj(-15)`.
//!
//! Every refusal is an [`ArchiveError`] carrying the Python exception class and
//! message the scientific reader raises for the same file.

pub mod deflate;
pub mod durable;
pub mod error;
pub mod figure;
pub mod inflate;
pub mod json;
pub mod manifest;
pub mod npdtype;
pub mod npy;
pub mod provenance;
pub mod pyliteral;
pub mod pystr;
pub mod tree;
pub mod zip;

pub use error::{ArchiveError, ErrorKind, Result};
pub use figure::{read_archive, write_figure_archive, write_figure_file, Entry, FigureArchive, Members};
pub use json::Json;
pub use manifest::{ArchiveDataset, DatasetRef, ReadDataset};
pub use npy::{NpyArray, NpyData, NpyType};
