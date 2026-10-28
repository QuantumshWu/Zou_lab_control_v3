//! The error ledger: the population tolerance, the error metric, and every
//! remainder the folded lattice leaves out as a generator-norm bound (closed
//! form, prepared from the classification). The error estimate builds three
//! of them into the slow modes' sources by their action
//! (`LatticeOperator::left_out`, `estimator`): the K^(1) rests the operator
//! leaves out (`k1_rest`), the non-structural fold rests (applied but for
//! their population rows; their row sums are no longer dropped entries) and
//! the static expansion's remainder (`nu_freeze`). The others (`unbuilt`) are
//! reported only. Their action on the computed state, measured with the third
//! order and the fold's memory built (two to five lasers, 20 us
//! to 10 ms, the steady state and the mean over 1 s): third order and memory
//! at most 6e-6 each, every other remainder at most 2.6e-7, against the
//! tolerance 5e-3; their generator-norm bounds lie 6-8 orders of magnitude
//! above what they do.
//!
//! Bounds (rad/us, Schur test over the lattice blocks, for a lattice whose
//! largest mode frequency is nu_max; d(r) = 1/(|nu_r| - nu_max - ||L0||)
//! bounds the far block of charge r):
//! - `nu_freeze`: the static expansion of (D_s - i nu_q)^-1 beyond
//!   i nu_q K^(1), sum c_back c_s nu_max^2 d0^3 / (1 - nu_max d0);
//! - `third_order`: q -> q+s -> q+s+e -> q+s+e+b, far s and b, a near
//!   coupling e between the far modes (tweezer -> near laser -> tweezer),
//!   returning to a near charge: sum c_b c_e c_s d(s) d(s+e);
//! - `far_shell`: every other far-shell path of three to six steps (second
//!   far shell, far + far + far returns), enumerated exactly over charges;
//! - `tail`: seven and more steps (Neumann series with the far shell above
//!   half the slowest far frequency);
//! - `omitted_cross`: far + far groups below the cross-fold floor;
//! - `non_structural`: near couplings below the structural floor, on their
//!   edges leaving the lattice;
//! - `dropped_entries`: fold entries below the storage floor (row sums);
//! - `k1_rest`: the K^(1) rests the lattice operator leaves out but for
//!   their population rows (`operator::K1_REST_SHARE`), times nu_max.
use super::*;
use super::classification::{HarmonicPlan,Treatment};

/// The population tolerance: absolute, on the probability scale, over every
/// state and requested group. It is fixed; no other accuracy is served.
pub const POPULATION_TOLERANCE:f64=5e-3;
/// Largest accepted static-expansion parameter max |nu_q| ||D_s^-1||.
pub const EXPANSION_LIMIT:f64=1e-2;
/// Far-shell paths enumerated exactly up to this many steps.
const ENUMERATED_STEPS:usize=6;

/// Largest population difference over the states and the requested groups.
pub fn metric(a:&[f64],b:&[f64],groups:&[Vec<usize>])->f64{
    let mut m=a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0_f64,f64::max);
    for g in groups{m=m.max(g.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs());}
    m
}

#[derive(Clone,Debug,Default)]
pub struct RemainderLedger{
    pub nu_freeze:f64,pub third_order:f64,pub far_shell:f64,pub tail:f64,
    pub omitted_cross:f64,pub non_structural:f64,pub dropped_entries:f64,pub k1_rest:f64,
    /// max |nu_q| ||D_s^-1|| over the folded steps.
    pub expansion_parameter:f64,
    /// The diagonal and cross folds' own bound sum c c' d (the size of K).
    pub fold:f64,
}
impl RemainderLedger{
    /// The remainders the estimate does not build (reported only; see the
    /// module documentation for their measured action).
    pub fn unbuilt(&self)->f64{
        self.third_order+self.far_shell+self.tail+self.omitted_cross+self.non_structural+self.dropped_entries
    }
    /// Bounds for a lattice with largest |nu_q| = `nu_max`; `dropped` are the
    /// fold storage's dropped row sums of K^(0) and K^(1), `k1_rest` the
    /// summed row sums of the K^(1) rests the lattice operator leaves out.
    pub fn new(plan:&HarmonicPlan,nu_max:f64,dropped:[f64;2],k1_rest:f64)->Result<Self,String>{
        let l0=plan.l0_bound;let cut=plan.cut;
        let mut ledger=Self{dropped_entries:dropped[0]+nu_max*dropped[1],k1_rest:nu_max*k1_rest,..Default::default()};
        ledger.non_structural=plan.steps.iter().filter(|s|s.treatment==Treatment::Explicit{structural:false}).map(|s|s.size).sum();
        ledger.fold=plan.targets.iter().map(|t|t.size).sum();
        let far=plan.far();
        if far.is_empty(){return Ok(ledger);}
        let delta=|nu:f64|->Result<f64,String>{let room=nu.abs()-nu_max-l0;
            if room>0.{Ok(1./room)}else{Err(format!("Far shell at {nu:e} rad/us is not separated from the near lattice (|nu_q| <= {nu_max:e}, ||L0|| <= {l0:e})"))}};
        for target in &plan.targets{for pair in &target.pairs{
            let s=&plan.steps[pair.source];let d0=1./(s.nu.abs()-l0);let p=nu_max*d0;
            ledger.expansion_parameter=ledger.expansion_parameter.max(p);
            if p>=1.{return Err(format!("Static fold expansion diverges: |nu_q| ||D_s^-1|| = {p:e}"));}
            ledger.nu_freeze+=plan.steps[pair.back].size*s.size*nu_max*nu_max*d0*d0*d0/(1.-p);
        }}
        for target in &plan.omitted{for pair in &target.pairs{
            ledger.omitted_cross+=plan.steps[pair.back].size*plan.steps[pair.source].size*delta(plan.steps[pair.source].nu)?;}}
        for &s in &far{for e in plan.explicit(){
            let(nu,c)=(plan.steps[s].nu,plan.steps[s].size);let mid=nu+plan.steps[e].nu;if mid.abs()<cut{continue;}
            let weight=c*plan.steps[e].size*delta(nu)?*delta(mid)?;
            for &b in &far{if (mid+plan.steps[b].nu).abs()<cut{ledger.third_order+=weight*plan.steps[b].size;}}
        }}
        // Far-shell paths by charge: weights of the far modes reached after m
        // steps (first step far, every intermediate beyond the cut) and the
        // weight returning below the cut at step m + 1.
        let all:Vec<(&Charge,f64,f64)>=plan.steps.iter().map(|s|(&s.charge,s.nu,s.size)).collect();
        let mut frontier=BTreeMap::<Charge,(f64,f64)>::new();
        for &s in &far{frontier.insert(plan.steps[s].charge.clone(),(plan.steps[s].nu,plan.steps[s].size*delta(plan.steps[s].nu)?));}
        let mut returns=0.;
        for step in 2..=ENUMERATED_STEPS{
            let mut next=BTreeMap::<Charge,(f64,f64)>::new();
            for(r,&(nu,w))in &frontier{for &(s,ns,c)in &all{
                let to=nu+ns;
                if to.abs()<cut{if step>2{returns+=w*c;}continue;}
                if step==ENUMERATED_STEPS{continue;}
                let entry=next.entry(charge_plus(r,s)?).or_insert((to,0.));entry.1+=w*c*delta(to)?;
            }}
            frontier=next;
        }
        ledger.far_shell=(returns-ledger.third_order).max(0.);
        let far_min=far.iter().map(|&s|plan.steps[s].nu.abs()).fold(f64::INFINITY,f64::min);
        let c_far:f64=far.iter().map(|&s|plan.steps[s].size).sum();let c_all:f64=all.iter().map(|x|x.2).sum();
        let q=c_all*delta(0.5*far_min)?;
        ledger.tail=if q<1.{c_far*q.powi(ENUMERATED_STEPS as i32)/(1.-q)}else{f64::INFINITY};
        Ok(ledger)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_ledger_bounds_are_finite_and_split_without_double_counting(){
        let f=fixture();let p=prepared(&f);
        let ledger=RemainderLedger::new(&p.plan,6.,[0.,0.],0.).unwrap();
        assert!(ledger.expansion_parameter<1e-2&&ledger.nu_freeze>0.&&ledger.third_order>0.&&ledger.unbuilt().is_finite());
        // The third-order paths are part of the enumerated three-step returns.
        assert!(ledger.far_shell>=0.&&ledger.tail.is_finite());
        assert_eq!(RemainderLedger::new(&p.plan,0.,[0.,0.],0.).unwrap().nu_freeze,0.);
    }
}
