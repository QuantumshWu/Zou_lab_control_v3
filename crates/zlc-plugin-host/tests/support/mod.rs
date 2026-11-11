#[allow(dead_code)] // The desktop's Pulse/offline entry points are not used by host fixtures.
#[path = "../../../zlc-desktop/src/plugin_services.rs"]
mod application;
pub use application::install;
