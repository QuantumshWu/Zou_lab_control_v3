//! The command bus: one catalogue of schema-described commands that the
//! instrument window and remote AI clients (MCP) submit alike, and the MCP
//! endpoint that serves it: every client reads; a remote one changes things
//! only while the endpoint allows control (v2 `allow_control`).
pub mod catalog;
mod compose;
pub mod mcp;
pub mod proxy;

pub use zlc_core::control::Origin;
