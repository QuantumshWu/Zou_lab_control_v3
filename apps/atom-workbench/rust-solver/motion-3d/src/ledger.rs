//! The approximation ledger of a 3D solve: every a-priori approximation -- one the statistics
//! cannot see, as they compare the method with itself -- with its estimated error, all judged by
//! one rule against one budget.
//!
//! # Budget
//! The budget is the a-priori share of the solve's own targets (solve.rs), as the population core
//! gives its a-priori rules their share of its tolerance: `A_PRIORI_SHARE` × the population target,
//! the stricter of the two -- every estimate is a relative error of the generator's rates and
//! energies, which bounds the populations' absolute error and T's relative error alike.
//!
//! # Rule
//! An approximation is taken where its estimate fits the budget; the estimates are summed and the
//! total is checked against it. What does not fit is made exact where the engine can (a term kept
//! explicitly, the exact response, a beam's profile tabulated); what cannot be made exact flags the
//! output times it concerns.
use crate::solve::POPULATION_TARGET;
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;

/// The budget of the a-priori approximations (`Ledger`).
pub const BUDGET:f64=A_PRIORI_SHARE*POPULATION_TARGET;

/// One approximation: what it is, its estimated error and what the engine did with it (Chinese,
/// as the app shows them).
#[derive(Clone,Debug,PartialEq)]
pub struct Item{pub label:String,pub estimate:f64,pub action:String}

/// The ledger of a solve: its items and, when some output times lie beyond the budget, why.
#[derive(Clone,Debug,PartialEq)]
pub struct Ledger{pub budget:f64,pub items:Vec<Item>,pub flagged:Option<String>}
impl Ledger{
    pub fn new()->Ledger{Ledger{budget:BUDGET,items:vec![],flagged:None}}
    pub fn add(&mut self,label:&str,estimate:f64,action:String){self.items.push(Item{label:label.into(),estimate,action});}
    /// The estimates' sum.
    pub fn used(&self)->f64{self.items.iter().map(|i|i.estimate).sum()}
    /// What is left of the budget.
    pub fn left(&self)->f64{(self.budget-self.used()).max(0.)}
}
impl Default for Ledger{fn default()->Ledger{Ledger::new()}}
