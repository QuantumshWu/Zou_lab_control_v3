//! The approximation ledger of a 3D solve: every a-priori approximation -- one the statistics
//! cannot see, as they compare the method with itself -- with its estimated error, booked as measured
//! against the solve's target (solve.rs, # Shares): the estimates are
//! summed per output into its a-priori total (solve.rs `Apriori`), and combined with the step's bias and the statistics in quadrature into its total
//! (solve.rs, # Shares); an output whose total exceeds a target is flagged. An item shows its largest over the outputs and the ledger's `totals` the output
//! where the a-priori total takes most of its target, at most the items' sum: they need not meet at one output.
//!
//! # The far light's threshold
//! `BUDGET`, the population core's a-priori share of the population target, is still the per-item
//! threshold of the far components' folds (ground.rs `far_folds`, no rung of the ladder), the profiles'
//! intensity axes (model.rs) and the ringing's share. The couplings of the excited and the ground states fold
//! by existence and the ladder's allowance (ground.rs `pair_folds`, `Ground::rungs`), the reach's tail is its
//! own (ground.rs `OUTSIDE_REACH`).
use crate::solve::POPULATION_TARGET;
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;

/// The far light's per-item threshold (module documentation).
pub const BUDGET:f64=A_PRIORI_SHARE*POPULATION_TARGET;

/// One approximation: what it is, its estimated error and what the engine did with it (Chinese,
/// as the app shows them).
#[derive(Clone,Debug,PartialEq)]
pub struct Item{pub label:String,pub estimate:f64,pub action:String}

/// A target's a-priori total at the output where it takes most of the target, and the target there (T's its regime's, solve.rs `regime`).
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct Total{pub error:f64,pub target:f64}

/// The ledger of a solve: its items; its a-priori totals, the populations' and T's (`totals`); when some output times lie beyond a
/// target, why; and what is true of the request's setup whatever the solve does (`note`: the initial ensemble's, no error of the solve's and no warning).
#[derive(Clone,Debug,PartialEq)]
pub struct Ledger{pub items:Vec<Item>,pub totals:[Total;2],pub flagged:Option<String>,pub note:Option<String>}
impl Ledger{
    pub fn new()->Ledger{Ledger{items:vec![],totals:[Total::default();2],flagged:None,note:None}}
    pub fn add(&mut self,label:&str,estimate:f64,action:String){self.items.push(Item{label:label.into(),estimate,action});}
    /// The items' estimates summed, what the a-priori total starts from (solve.rs `Apriori`).
    pub(crate) fn sum(&self)->f64{self.items.iter().map(|i|i.estimate).sum()}
    /// What the estimates use of the population target: its a-priori total (`totals`).
    pub fn used(&self)->f64{self.totals[0].error}
}
impl Default for Ledger{fn default()->Ledger{Ledger::new()}}
