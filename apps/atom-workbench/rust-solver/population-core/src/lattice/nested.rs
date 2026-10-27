//! Validation scaffolding: the nested near-order check. Starting at k = 2,
//! the steady state is solved at k and k + 1, and k + 1 is accepted when
//! kappa (|p_k - p_k+1| + fold charges) <= POPULATION_TOLERANCE; the k + 1
//! populations are reported, so the check costs the order-k solve. The GCRO
//! continuation of the primal (an error equation on its Krylov space, 3-6
//! steps) replaces it; this module then goes into the tests.
use super::*;
use super::ledger::{self,KAPPA,POPULATION_TOLERANCE};
use super::preconditioner::{DenseDressed,DressedDiagnostics};
use super::steady::{self,OrderSolve,SteadyOptions};

/// First near order of the check; k = 2 suffices for the steady cases seen.
pub const START_ORDER:usize=2;
pub const MAXIMUM_ORDER:usize=6;
/// Largest near ball the check may reach.
pub const MAXIMUM_MODES:usize=1500;

pub struct NestedResult{
    pub solves:Vec<OrderSolve>,
    /// Index of the reported solve in `solves`.
    pub accepted:usize,
    /// |p_k - p_k+1| between the reported order and the one below.
    pub difference:Option<f64>,
    /// kappa (difference + fold charges), the reported error estimate.
    pub estimate:Option<f64>,
    pub dressed:DressedDiagnostics,
    pub dressed_ms:f64,
}

/// The steady state with the nested check, or at the given orders only
/// (validation: every listed order is solved and charged; the last is
/// reported).
pub fn solve(prepared:&LatticePrepared,groups:&[Vec<usize>],options:&SteadyOptions,orders:Option<&[usize]>)->Result<NestedResult,String>{
    let clock=Instant::now();let dressed=DenseDressed::new(prepared)?;let dressed_ms=clock.elapsed().as_secs_f64()*1e3;
    let fixed=orders.is_some();
    let list:Vec<usize>=orders.map(|o|o.to_vec()).unwrap_or_else(||(START_ORDER..=MAXIMUM_ORDER).collect());
    let mut solves:Vec<OrderSolve>=Vec::new();
    for(i,&k)in list.iter().enumerate(){
        if prepared.modes(k)?.len()>MAXIMUM_MODES{
            if fixed{break;}
            return Err(format!("Near-order check: order {k} exceeds {MAXIMUM_MODES} modes before the populations agreed"));
        }
        let charge=fixed||i>0;
        let solve=steady::solve_order(prepared,&dressed,k,groups,options,charge)?;
        if !fixed{if let Some(previous)=solves.last(){
            let difference=ledger::metric(&previous.populations,&solve.populations,groups);
            let charges=solve.charge.as_ref().map_or(0.,|c|c.charges.total());
            if KAPPA*(difference+charges)<=POPULATION_TOLERANCE{
                solves.push(solve);let accepted=solves.len()-1;
                return Ok(NestedResult{solves,accepted,difference:Some(difference),estimate:Some(KAPPA*(difference+charges)),dressed:dressed.diagnostics.clone(),dressed_ms});
            }
        }}
        solves.push(solve);
    }
    if !fixed||solves.is_empty(){return Err(format!("Near-order check did not reach the tolerance by order {MAXIMUM_ORDER}"));}
    let accepted=solves.len()-1;
    let difference=(accepted>0).then(||ledger::metric(&solves[accepted-1].populations,&solves[accepted].populations,groups));
    let charges=solves[accepted].charge.as_ref().map_or(0.,|c|c.charges.total());
    Ok(NestedResult{estimate:difference.map(|x|KAPPA*(x+charges)),solves,accepted,difference,dressed:dressed.diagnostics.clone(),dressed_ms})
}
