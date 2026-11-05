//! What a solve's work costs and whether spending it pays: one table of unit costs, one rule.
//!
//! # The rule
//! Every decision that spends work for a saving is `pays`: the spend, what it could save, and the uncertainty of that saving, all in the table's unit.
//! The uncertainty is the prediction's own where it has one, the need's relative error (solve.rs `need_error`) times the predicted cost. What each gate reads differs for a reason, stated at the gate:
//! a check already made is no longer a spend; a count of products has no error of its own (the half step); a decision made before any pilot has no need's error to read yet (the tables' refinement); and a read that guards an accuracy takes no margin against itself (`read_pays`).
//! Beside the rule stand what is no trade of work: the shares (a block taken unmixed, ground.rs `MIXING_SHARE`, an error the ledger books, decided before a model's layouts and its tables' nodes exist to cost it) and the first batch's size (solve.rs # Batches).
//!
//! # The unit
//! The nanosecond of one performance core of the Core Ultra 7 265H, the machine the table was measured on. No timing enters a solve: every cost is a count of work at the table's rates, so that a request gives
//! the same decisions on any machine and at any load, and only the table's ratios decide -- on either device: a device's own speed divides out of any choice between models on it.
//! What the kernel's work reads against these counts, a choice between the orders: their ratio here (PGC3D 1.88, TET 1.94, GMD1C 2.75, THREE 1.82, DEFAULT 1.00) against the kernel's as profiled on a laptop's RTX 5070, a wave of 576 (1.71, 1.83, 2.84, 2.03, 1.07), within 10 %; no device needs a table of its own.
//! A wave takes its width at no further cost, and that no choice reads, for a choice that read it would change with the device's width: where the fourth order saves trajectories beyond its work and the pilot's model fits a wave, a GPU pays up to that ratio for the saving a CPU makes.
//! No run of the gate reads differently at a width of 576: those the fourth order's promise passes take more than a wave at the second order (R5004, R5051) or lie past the targets (B1_128, the one that takes it).
//!
//! # The table
//! The advance's, the half step's: measured on that core (examples/bench_cost.rs) and fitted over DEFAULT, THREE, PGC3D, TET, GMD1C and PGCP_10, each part within 10–20 % of its measured time.
//! The jump's: timed against the advance in the same interleaved rounds on the loaded machine (each the least of 30 rounds × 3 passes), taken at the advance's cost: within 10 % but DEFAULT's (+21 %); a jump costs 1.5–2.1 times its photons alone.
//! The exact step's: against the step's CPU time on that core (the jump-free step at each of a candidate's states along its trajectories, as examples/multirate.rs `profile` times it): PGC3D's 10-state cluster
//! (spectral at the run's beats share), PGCP_10's 12-state and 5-state beating clusters and PGC3D's folded one take 1.2-3.3 times their counts on the loaded machine -- the counts' ratios within about ±30 %.
use crate::model::Work;

/// An advance without its propagation: per call (the point's fields, the flow, the kick), per multiply-add of the generator and the force (`Model::work`) and per table pair.
pub(crate) const ADVANCE_NS:f64=374.;pub(crate) const MULTIPLY_ADD_NS:f64=0.247;pub(crate) const PAIR_NS:f64=29.9;
/// A half step's series (engine.rs `propagate`): per term, and per complex multiply-add (n² a term).
pub(crate) const TERM_NS:f64=4.4;pub(crate) const COMPLEX_MULTIPLY_ADD_NS:f64=0.469;
/// A whole jump (`Engine::jump`): per call (its point, the class's and the polarization's draws, the dipole pattern, the kicks), per multiply-add of every class's photons and per multiply-add of the
/// picked class's absorbed light (`Model::work`).
pub(crate) const JUMP_NS:f64=970.;pub(crate) const PHOTON_MULTIPLY_ADD_NS:f64=0.594;pub(crate) const ABSORPTION_MULTIPLY_ADD_NS:f64=1.48;
/// The exact step (solve.rs `exact_step`): per n³ of the spectral path's eigendecomposition with its forms, in complex multiply-adds, and per complex exponential of a beat's phase (at the large arguments a run's times give).
pub(crate) const SPECTRAL_N3:f64=60.;pub(crate) const CIS_NS:f64=25.;

/// Whether spending `spend` pays: it does only where what it could save stands beyond the spend and the saving's uncertainty, all in the table's unit (module documentation).
#[inline]
pub(crate) fn pays(spend:f64,could_save:f64,uncertainty:f64)->bool{spend+uncertainty<could_save}

/// An advance's counted work in a cluster, its propagation aside.
pub(crate) fn advance_work(w:&Work)->f64{ADVANCE_NS+MULTIPLY_ADD_NS*(w.generator+w.force)+PAIR_NS*w.pairs as f64}
/// A cluster's jump.
pub(crate) fn jump_work(w:&Work)->f64{JUMP_NS+PHOTON_MULTIPLY_ADD_NS*w.photons+ABSORPTION_MULTIPLY_ADD_NS*w.absorption}

#[cfg(test)]
mod tests{
    use super::pays;
    /// A spend pays only for a saving beyond it and the saving's uncertainty; a spend made is none, and no saving is none.
    #[test]
    fn a_spend_pays_only_beyond_itself_and_the_uncertainty(){
        assert!(pays(1.,2.,0.)&&!pays(2.,2.,0.)&&!pays(1.,2.,1.)&&pays(1.,2.,0.5));
        assert!(pays(0.,1.,0.)&&!pays(0.,0.,0.)&&!pays(0.,-1.,0.));
        assert!(!pays(1.,f64::NAN,0.),"a saving that cannot be read pays nothing");
    }
}
