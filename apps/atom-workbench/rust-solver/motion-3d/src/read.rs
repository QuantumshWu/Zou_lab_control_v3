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
    /// A paired numerical read cannot certify a non-finite statistic or fewer than two groups.
    /// Its guarded domain is distinct from the product's legacy presentation semantics;
    /// both use the one arithmetic kernel above.
    pub fn jackknife(value:f64,leave:impl IntoIterator<Item=f64>)->Self{
        let leave:Vec<_>=leave.into_iter().collect();
        if !value.is_finite()||!crate::stats::reads_spread(leave.len() as f64)||leave.iter().any(|x|!x.is_finite()){
            return Self{value,error:f64::INFINITY};
        }
        Self{value,error:jackknife_error(leave.iter().copied())}
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn one_jackknife_kernel_keeps_the_product_and_guarded_domains_distinct(){
        let ordinary=jackknife_error([1.,2.,4.].into_iter());
        assert!((ordinary-28_f64.sqrt()/3.).abs()<1e-15);
        assert_eq!(Read::jackknife(7.,[1.,2.,4.]).error,ordinary);
        assert!(jackknife_error([1.,f64::NAN].into_iter()).is_nan());
        for r in [Read::jackknife(0.,[]),Read::jackknife(0.,[1.]),Read::jackknife(0.,[1.,f64::NAN]),Read::jackknife(f64::NAN,[1.,2.])]{assert!(r.error.is_infinite());}
        let constant=Read::jackknife(3.,[2.,2.,2.]);assert_eq!((constant.value,constant.error),(3.,0.));
    }
}
