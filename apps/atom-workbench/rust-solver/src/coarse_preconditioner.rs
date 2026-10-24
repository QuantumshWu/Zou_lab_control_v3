//! Optional two-level correction of a fixed linear preconditioner.
//!
//! This module does not remove states, matrix entries, or physical couplings.
//! The full outer solver must continue checking its ORIGINAL system residual.
//! The coarse space changes only the approximate inverse supplied to GMRES:
//!   z = M^-1 b; z <- z + Q (Q^T A Q)^-1 Q^T (b - A z).
//! `Q` spans numerically independent columns `M^-1 e_seed`. When explicitly
//! enabled, one full-operator residual sweep lifts each column into coupled
//! harmonic blocks without adding coarse coordinates. A singular or
//! poorly conditioned coarse operator is an error so the caller can use M^-1
//! without this optional correction.

use crate::lapack::Lapack;

pub(super) type BaseApply = Box<dyn Fn(&Lapack, &[f64]) -> Result<Vec<f64>, String>>;
type Sparse = Vec<Vec<(usize, f64)>>;

pub(super) struct Prepared {
    a: Sparse,
    base: BaseApply,
    // Column-major Q (n by rank) and Q^T (rank by n) for BLAS applications.
    q: Vec<f64>,
    qt: Vec<f64>,
    rank: usize,
    lu: Vec<f64>,
    pivots: Vec<i32>,
    condition: f64,
}

fn norm(x: &[f64]) -> f64 {
    let scale = x.iter().map(|v| v.abs()).fold(0., f64::max);
    if scale == 0. { return 0.; }
    scale * x.iter().map(|v| (v / scale).powi(2)).sum::<f64>().sqrt()
}

fn product(a: &[Vec<(usize, f64)>], x: &[f64]) -> Vec<f64> {
    let mut result = vec![0.; a.len()];
    for (j, column) in a.iter().enumerate() {
        for &(i, value) in column { result[i] += value * x[j]; }
    }
    result
}

impl Prepared {
    /// `a` and the callback must describe the same (unscaled) coordinates.
    /// The callback must be the fixed base inverse, not an RHS-dependent
    /// adaptive inner solve; the resulting preconditioner is then linear.
    pub fn new(
        lap: &Lapack,
        a: &[Vec<(usize, f64)>],
        base: BaseApply,
        seeds: &[usize],
    ) -> Result<Self, String> {
        let lift=std::env::var("ATOM_COARSE_LIFT").is_ok_and(|value|value=="1");
        Self::new_with_lift(lap,a,base,seeds,lift)
    }

    fn new_with_lift(
        lap: &Lapack,
        a: &[Vec<(usize, f64)>],
        base: BaseApply,
        seeds: &[usize],
        lift: bool,
    ) -> Result<Self, String> {
        let n = a.len();
        if n == 0 || n > i32::MAX as usize
            || a.iter().flatten().any(|&(row, value)| row >= n || !value.is_finite())
            || seeds.iter().any(|&index| index >= n) {
            return Err("Invalid coarse-preconditioner matrix or seeds".into());
        }
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut used = vec![false; n];
        let mut unit = vec![0.; n];
        let independent = (64. * f64::EPSILON * (n as f64).sqrt()).max(1e-12);
        for &seed in seeds {
            if used[seed] { continue; }
            used[seed] = true;
            unit[seed] = 1.;
            let mut column = base(lap, &unit)?;
            unit[seed] = 0.;
            if column.len() != n || column.iter().any(|v| !v.is_finite()) {
                return Err("Invalid base inverse column in coarse preconditioner".into());
            }
            if lift {
                // A harmonic-block diagonal M^-1 maps a central seed only to
                // the central block. One Richardson step supplies sideband
                // tails through EVERY original A entry, while keeping one
                // candidate per seed: q=q0+M^-1(e-Aq0).
                let mut residual:Vec<_>=product(a,&column).into_iter().map(|value|-value).collect();
                residual[seed]+=1.;
                if residual.iter().any(|v|!v.is_finite()) {
                    return Err("Non-finite coarse prolongation residual".into());
                }
                let correction=base(lap,&residual)?;
                if correction.len()!=n||correction.iter().any(|v|!v.is_finite()) {
                    return Err("Invalid base inverse in coarse prolongation".into());
                }
                for (value,delta) in column.iter_mut().zip(correction) {*value+=delta;}
                if column.iter().any(|v|!v.is_finite()) {
                    return Err("Non-finite lifted coarse column".into());
                }
            }
            let initial_norm = norm(&column);
            if initial_norm == 0. { continue; }
            if !initial_norm.is_finite() {
                return Err("Unbounded base inverse column in coarse preconditioner".into());
            }
            // Normalize first: dependence is relative to each seed's response,
            // not to the arbitrary magnitude of a diagonal scaling in M^-1.
            for value in &mut column { *value /= initial_norm; }
            for _ in 0..2 {
                for previous in &basis {
                    let projection: f64 = previous.iter().zip(&column).map(|(a,b)| a*b).sum();
                    for (value, &direction) in column.iter_mut().zip(previous) {
                        *value -= projection * direction;
                    }
                }
            }
            let remaining = norm(&column);
            if remaining <= independent { continue; }
            for value in &mut column { *value /= remaining; }
            basis.push(column);
        }
        let rank = basis.len();
        if rank == 0 { return Err("No independent coarse-preconditioner columns".into()); }
        let mut qt = vec![0.; n * rank];
        let mut aq = Vec::with_capacity(n * rank);
        for (j, column) in basis.iter().enumerate() {
            for (i, &value) in column.iter().enumerate() { qt[j+i*rank] = value; }
            aq.extend(product(a, column));
        }
        let e = lap.multiply(&qt, &aq, rank, rank, n);
        if e.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite coarse-preconditioner operator".into());
        }
        let norm_one = e.chunks_exact(rank).map(|column| column.iter().map(|v|v.abs()).sum::<f64>()).fold(0.,f64::max);
        let (lu, pivots) = lap.factor(&e, rank)
            .map_err(|error| format!("Coarse-preconditioner factorization: {error}"))?;
        let condition = lap.condition(&lu, rank, norm_one)?;
        if !condition.is_finite() || condition > 1e13 {
            return Err(format!("Ill-conditioned coarse preconditioner: {condition:e}"));
        }
        let q = basis.into_iter().flatten().collect();
        Ok(Self { a: a.to_vec(), base, q, qt, rank, lu, pivots, condition })
    }

    pub fn rank(&self) -> usize { self.rank }
    pub fn condition(&self) -> f64 { self.condition }

    pub fn apply(&self, lap: &Lapack, b: &[f64]) -> Result<Vec<f64>, String> {
        let n = self.a.len();
        if b.len() != n || b.iter().any(|v|!v.is_finite()) {
            return Err("Invalid coarse-preconditioner right-hand side".into());
        }
        let mut z = (self.base)(lap, b)?;
        if z.len() != n || z.iter().any(|v|!v.is_finite()) {
            return Err("Invalid base inverse result in coarse preconditioner".into());
        }
        let az = product(&self.a, &z);
        let residual: Vec<_> = b.iter().zip(az).map(|(b,az)| b-az).collect();
        let projected = lap.multiply(&self.qt, &residual, self.rank, 1, n);
        let coefficients = lap.solve_factored(&self.lu, &self.pivots, &projected, self.rank)?;
        let correction = lap.multiply(&self.q, &coefficients, n, 1, self.rank);
        for (value, delta) in z.iter_mut().zip(correction) { *value += delta; }
        if z.iter().any(|v|!v.is_finite()) {
            return Err("Non-finite coarse-preconditioner correction".into());
        }
        Ok(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse(dense: &[f64], n: usize) -> Sparse {
        (0..n).map(|j|(0..n).filter_map(|i|(dense[i+j*n]!=0.).then_some((i,dense[i+j*n]))).collect()).collect()
    }

    #[test]
    fn coarse_correction_resolves_bad_slow_block_and_outer_solve_keeps_every_coupling() {
        let lap = Lapack::load().unwrap();
        let n = 12;
        let mut dense = vec![0.; n*n];
        for j in 0..n {
            for i in 0..n {
                dense[i+j*n] = if i==j { if i<4 { 1. } else { 100.+i as f64 } }
                    else if i<4 && j<4 { -0.32 }
                    else { 0.00004*((i+2*j+1) as f64).sin() };
            }
        }
        let a = sparse(&dense,n);
        let before = a.clone();
        let diagonal: Vec<_> = (0..n).map(|i|dense[i+i*n]).collect();
        let make_base = || {let diagonal=diagonal.clone();Box::new(move|_:&Lapack,b:&[f64]|Ok(b.iter().zip(&diagonal).map(|(b,d)|b/d).collect())) as BaseApply};
        let prepared = Prepared::new_with_lift(&lap,&a,make_base(),&[0,1,2,3],false).unwrap();
        assert_eq!(prepared.rank(),4);
        let b: Vec<_> = (0..n).map(|i|if i<4 {1.} else {(i as f64).cos()}).collect();
        let initial = make_base()(&lap,&b).unwrap();
        let corrected = prepared.apply(&lap,&b).unwrap();
        let relative_residual = |x:&[f64]|norm(&b.iter().zip(product(&a,x)).map(|(b,ax)|b-ax).collect::<Vec<_>>())/norm(&b);
        assert!(relative_residual(&initial)>0.5);
        assert!(relative_residual(&corrected)<0.01);
        // The correction annihilates the coarse residual. Fast/coarse feedback
        // remains in A and is completed by the unchanged outer full-system solve.
        let coarse_residual=lap.multiply(&prepared.qt,&b.iter().zip(product(&a,&corrected)).map(|(b,ax)|b-ax).collect::<Vec<_>>(),4,1,n);
        assert!(norm(&coarse_residual)<1e-12);
        let outer=crate::block_solve::Prepared::with_custom(&a,Box::new(move|lap,b|prepared.apply(lap,b))).unwrap();
        let actual=outer.solve(&lap,&b).unwrap();
        let expected=lap.solve(&dense,&b,n).unwrap();
        assert!(norm(&actual.iter().zip(expected).map(|(a,b)|a-b).collect::<Vec<_>>())<1e-9);
        assert_eq!(a,before);
    }

    #[test]
    fn dependent_seed_images_are_removed_and_the_basis_is_orthonormal() {
        let lap=Lapack::load().unwrap();
        let a=vec![vec![(0,2.)],vec![(1,3.)],vec![(2,4.)]];
        let base:BaseApply=Box::new(|_,b|Ok(vec![b[0]+b[1],b[0]+b[1],b[2]]));
        let prepared=Prepared::new_with_lift(&lap,&a,base,&[0,1,1,2],false).unwrap();
        assert_eq!(prepared.rank(),2);
        let gram=lap.multiply(&prepared.qt,&prepared.q,2,2,3);
        for (i,&value) in gram.iter().enumerate(){assert!((value-if i==0||i==3{1.}else{0.}).abs()<1e-14);}
    }

    #[test]
    fn unsafe_coarse_factor_is_rejected_for_caller_fallback() {
        let lap=Lapack::load().unwrap();
        let a=vec![vec![(0,1.)],vec![(1,1e-15)]];
        let error=Prepared::new_with_lift(&lap,&a,Box::new(|_,b|Ok(b.to_vec())),&[0,1],false).err().unwrap();
        assert!(error.contains("Ill-conditioned"));
        assert!(Prepared::new_with_lift(&lap,&a,Box::new(|_,_|Ok(vec![f64::NAN;2])),&[0],false).is_err());
        assert!(Prepared::new_with_lift(&lap,&a,Box::new(|_,b|Ok(b.to_vec())),&[2],false).is_err());
    }

    #[test]
    fn lifted_seed_reaches_sidebands_without_adding_rank_or_changing_full_outer_system() {
        let lap=Lapack::load().unwrap();
        let n=4;
        // Two central coordinates, two sideband coordinates; all couplings
        // are present, nonsymmetric, and retained by the outer solve.
        let dense=vec![2.,0.1,0.5,0.4, 0.2,3.,-0.2,0.3,
            0.7,0.4,5.,-0.1, -0.3,0.6,0.2,7.];
        let a=sparse(&dense,n);let original=a.clone();
        let base=||Box::new(|_:&Lapack,b:&[f64]|Ok(vec![b[0]/2.,b[1]/3.,b[2]/5.,b[3]/7.]))as BaseApply;
        let ordinary=Prepared::new_with_lift(&lap,&a,base(),&[0,1],false).unwrap();
        let lifted=Prepared::new_with_lift(&lap,&a,base(),&[0,1],true).unwrap();
        assert_eq!(ordinary.rank(),2);assert_eq!(lifted.rank(),2);
        assert!(ordinary.q.chunks_exact(n).all(|q|q[2..].iter().all(|&value|value==0.)));
        assert!(lifted.q.chunks_exact(n).all(|q|q[2..].iter().any(|value|value.abs()>1e-4)));
        let b=vec![1.,-2.,0.3,0.5];
        let outer=crate::block_solve::Prepared::with_custom(&a,Box::new(move|lap,b|lifted.apply(lap,b))).unwrap();
        let actual=outer.solve(&lap,&b).unwrap();
        let expected=lap.solve(&dense,&b,n).unwrap();
        assert!(norm(&actual.iter().zip(expected).map(|(a,b)|a-b).collect::<Vec<_>>())<1e-10);
        assert!(norm(&b.iter().zip(product(&a,&actual)).map(|(b,ax)|b-ax).collect::<Vec<_>>())<1e-10);
        assert_eq!(a,original);
    }
}
