//! The gate in front of the full estimate: the change of the outputs since
//! the previous checkpoint.
//!
//! The outputs of a checkpoint (trajectory, horizons, steady state) are the
//! pipeline's own (`ReducedView::outputs`). If they moved by delta since the
//! previous checkpoint, then the errors e_k, e_k-1 of the two satisfy
//! delta <= e_k + e_k-1, so delta > 4 eps means one of them exceeds 2 eps:
//! the full estimate is skipped (and delta returned) unless the previous
//! full estimate was itself within GATE_FACTOR eps. The first checkpoint
//! has nothing to compare with and is not estimated, except for a request
//! that propagates nothing: its steady state is compared with the
//! pipeline's own steady solve. The gate never accepts: a skipped
//! checkpoint reports delta > 4 eps (the first one infinity). Its cost is
//! one checkpoint late at worst, when the first checkpoint is already within
//! the tolerance or a jump from a poor checkpoint lands directly within it.
//! The basis columns enter the fast part at every checkpoint, open or not
//! (the projection hands their frame coordinates over once).
//!
//! A residual-norm bound is no gate: the fast components dominate the
//! residual's norm and would keep it closed; R1's dressed-block propagation
//! needs a d x d coordinate change per mode and Ritz mode, as much as the
//! full estimate.
use super::super::ledger::metric;

/// The full estimate runs when the outputs moved by at most this times the
/// tolerance, or when the previous full estimate was within it.
pub const GATE_FACTOR:f64=4.;

/// The reduced model's outputs: populations per grid time, per horizon, and
/// the steady state.
#[derive(Clone,Debug,Default)]
pub struct Outputs{pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,pub steady:Option<Vec<f64>>}

#[derive(Default)]
pub struct OutputGate{
    previous:Option<Outputs>,
    /// The latest full estimate's largest bound.
    pub last_estimate:Option<f64>,
}

impl OutputGate{
    /// The largest change of the outputs since the previous checkpoint (None
    /// at the first), and the outputs kept for the next.
    pub fn change(&mut self,outputs:Outputs,groups:&[Vec<usize>])->Option<f64>{
        let change=self.previous.as_ref().map(|p|{
            let pairs=p.trajectory.iter().zip(&outputs.trajectory).chain(p.horizons.iter().zip(&outputs.horizons)).chain(p.steady.iter().zip(&outputs.steady));
            pairs.map(|(a,b)|metric(a,b,groups)).fold(0.,f64::max)});
        self.previous=Some(outputs);
        change
    }
}
