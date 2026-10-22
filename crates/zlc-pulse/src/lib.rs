//! The pulse stack of the frozen period-table pulse streamer: model, binding
//! resolution, compile, board image, transports, device and the remote board
//! protocol. Bit-exact with v2's `zlc_pulse` (see `docs/PULSE_V3.md` and
//! the goldens in `research/pulse-golden`).
pub mod error;
pub mod exact;
pub mod json;
pub mod num;
pub mod units;
pub mod canonical;
pub mod model;
pub mod loops;
pub mod components;
pub mod compile;
pub mod wire;
pub mod codec;
pub mod binding;
pub mod scan;
pub mod schedule;
pub mod manifest;
pub mod transport;
pub mod uart;
pub mod axi;
pub mod serial;
pub mod device;
pub mod remote;
pub mod board;
pub mod run;
pub mod fpga;

pub use error::{Error, Kind, Result};
