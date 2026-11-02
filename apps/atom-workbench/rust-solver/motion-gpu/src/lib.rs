//! The 2D radial motion engine: a single Rb87 atom in an optical tweezer moving in the plane through
//! the focus perpendicular to the tweezer, by quantum trajectories (Monte Carlo wavefunctions) on
//! an NVIDIA GPU. The physics front end is `atom_motion_core::ground`, evaluated on the grid on
//! the CPU in double precision; the trajectories run in single precision on the GPU through the
//! driver API alone (`driver`), with kernels compiled ahead of time to PTX (`ptx`).
pub mod driver;
pub mod ptx;
pub mod gpu;
