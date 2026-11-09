//! A read (solve.rs # Time step): an estimate and its standard error, the jackknife's arithmetic shared. Observable definitions and validity
//! checks stay with their callers.

/// The delete-one-group standard error. Product statistics call this on their existing valid
/// group domain, preserving their former IEEE behavior if a moment is non-finite.
/// A cloneable exact-size iterator avoids an allocation in each product observable.
pub(crate) fn jackknife_error(leave:impl ExactSizeIterator<Item=f64>+Clone)->f64{
    let n=leave.len() as f64;
    let mean=leave.clone().sum::<f64>()/n;
    ((n-1.)/n*leave.map(|x|(x-mean).powi(2)).sum::<f64>()).sqrt()
}

#[derive(Clone,Copy,Debug)]
pub(crate) struct Read{pub value:f64,pub error:f64}
impl Read{
    /// What authorizes a step (solve.rs # Time step): the magnitude and CONFIDENCE standard errors -- an estimate the read cannot tell
    /// from nothing stands at its errors.
    pub fn upper(&self)->f64{self.value.abs()+crate::solve::CONFIDENCE*self.error}
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn the_jackknife_s_error_is_the_groups_left_out_spread(){
        let ordinary=jackknife_error([1.,2.,4.].into_iter());
        assert!((ordinary-28_f64.sqrt()/3.).abs()<1e-15);
        assert!(jackknife_error([1.,f64::NAN].into_iter()).is_nan());
        assert_eq!(jackknife_error([2.,2.,2.].into_iter()),0.);
    }
}
