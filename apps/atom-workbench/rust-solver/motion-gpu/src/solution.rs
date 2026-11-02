//! What a 2D motion solve returns, at every output time of the request. Everything but the loss
//! describes the atom still held.

/// A distribution on a grid at some of the output times.
#[derive(Clone,Debug,Default,PartialEq)]
pub struct Distributions{
    /// The grid, strictly increasing (velocities in m/s or positions in um).
    pub grid:Vec<f64>,
    /// The output-time indices the distributions are at: increasing, the last time among them.
    pub samples:Vec<usize>,
    /// One distribution per listed time, probabilities on the grid summing to one.
    pub probabilities:Vec<Vec<f64>>,
}
/// The position distribution in the plane: on `grid` along both axes, row-major (y rows of x).
#[derive(Clone,Debug,Default,PartialEq)]
pub struct Map{
    pub grid:Vec<f64>,
    pub samples:Vec<usize>,
    pub probabilities:Vec<Vec<f64>>,
}

#[derive(Clone,Debug,Default,PartialEq)]
pub struct Solution{
    /// Per time, the 32 internal populations (model basis) averaged over the position.
    pub populations:Vec<Vec<f64>>,
    /// Per time, the largest standard error of a population or an F group.
    pub sample_errors:Vec<f64>,
    /// Per time and plane axis: T = m Var(v)/k_B (uK) and its standard error, the mean velocity
    /// (m/s) and the mean occupation n̄ of the harmonic levels of `trap_frequency_khz`.
    pub temperature_uk:Vec<[f64;2]>,
    pub temperature_errors_uk:Vec<[f64;2]>,
    pub mean_velocity_mps:Vec<[f64;2]>,
    pub mean_occupation:Vec<[f64;2]>,
    /// ω/2π (kHz) along each axis at the focus.
    pub trap_frequency_khz:[f64;2],
    /// Per time, the probability that the atom has been lost (has climbed half the trap's depth).
    pub loss_probability:Vec<f64>,
    /// Velocity marginals along each axis, and the position distribution in the plane.
    pub velocity:[Distributions;2],
    pub position:Map,
    /// The grids (points per axis) and, per output time, how many trajectories of the last batch
    /// each held.
    pub grids:Vec<usize>,
    pub census:Vec<Vec<usize>>,
    /// Wall seconds: launching the steps, waiting for the observations, moving trajectories between
    /// grids, building the grids' tables.
    pub seconds:[f64;4],
    /// The trajectories the estimates rest on, and the method in one sentence.
    pub trajectories:usize,
    pub method:String,
    /// Why the run stopped above its targets (none when it reached them).
    pub reason:Option<String>,
}
