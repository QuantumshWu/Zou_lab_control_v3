//! The approximation ledger of a 3D solve: every a-priori approximation -- one the statistics
//! cannot see, as they compare the method with itself -- with its estimated error, booked as measured
//! against the solve's population target (solve.rs, # Shares), the ledger's budget: the estimates are
//! summed, and combined with the step's bias and the statistics in quadrature into each output's total
//! (solve.rs, # Shares); an output whose total exceeds a target is flagged.
//!
//! # The fold rule's threshold
//! `BUDGET`, the population core's a-priori share of the population target, is still the per-item
//! threshold of the couplings' and components' folds (ground.rs `classify`, the excited states' static
//! couplings), the profiles' intensity axes and the reach's thermal tail (model.rs): their move into the
//! fold candidates' cost comparison (solve.rs, # The fold's choice) is pending; on the scenes measured no
//! verdict lies between it and the target where a cost comparison would differ.
use crate::solve::POPULATION_TARGET;
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;

/// The fold rule's per-item threshold (module documentation).
pub const BUDGET:f64=A_PRIORI_SHARE*POPULATION_TARGET;

/// One approximation: what it is, its estimated error and what the engine did with it (Chinese,
/// as the app shows them).
#[derive(Clone,Debug,PartialEq)]
pub struct Item{pub label:String,pub estimate:f64,pub action:String}

/// The ledger of a solve: its items against the population target (`budget`); when some output times lie beyond a
/// target, why; and what is true of the request's setup whatever the solve does (`note`: the initial ensemble's, no error of the solve's and no warning).
#[derive(Clone,Debug,PartialEq)]
pub struct Ledger{pub budget:f64,pub items:Vec<Item>,pub flagged:Option<String>,pub note:Option<String>}
impl Ledger{
    pub fn new()->Ledger{Ledger{budget:crate::solve::POPULATION_TARGET,items:vec![],flagged:None,note:None}}
    pub fn add(&mut self,label:&str,estimate:f64,action:String){self.items.push(Item{label:label.into(),estimate,action});}
    /// The estimates' sum.
    pub fn used(&self)->f64{self.items.iter().map(|i|i.estimate).sum()}
    /// What is left of the budget.
    pub fn left(&self)->f64{(self.budget-self.used()).max(0.)}
}
impl Default for Ledger{fn default()->Ledger{Ledger::new()}}
