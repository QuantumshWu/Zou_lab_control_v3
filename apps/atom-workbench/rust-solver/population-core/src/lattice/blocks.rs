//! The second-order block of the frame: D'_q = L0' - i nu_q + sum_s K_s over
//! every step s (`kernels`), one block for every near mode (their blocks
//! differ by -i nu_q only), built once (`frame::SwFrame::class`): the
//! metric's V^-1 and the slowest relaxation's classes.
//!
//! In eigen-coordinates of L0' a class is diag(lambda) + sum_s K~_s, whose
//! eigendecomposition V diag(mu) V^-1 diagonalizes it; the class also stores
//! the population rows of E V (each class's populations).
//!
//! A class whose steps come in pairs (s, -s) is invariant under the flip J,
//! so its matrix is real in the real eigen-coordinates of L0' (the pair
//! rule of `eigenbasis`): real eigendecompositions and LUs, about a quarter
//! of the complex ones. Most couplings of diag(lambda) + K~ are far below
//! their gaps, so the eigensystem is built by clusters of strongly coupled
//! coordinates, exact within each and perturbative between them, with the
//! mixing and the values' error over the request's horizon bounded
//! (`eigenbasis::clustered_eigensystem`, the centre's own method).
use super::*;
use super::eigenbasis::{BlockEigenbasis,ClusterControl,clustered_eigensystem};
use super::kernels::SwKernels;

#[derive(Clone)]
pub struct BlockSystem{
    /// Eigenvalues mu (the mode's own -i nu_q not included), the
    /// eigenvectors V and V^-1 (column-major d x d, eigen-coordinates of L0').
    pub values:Vec<C>,pub vectors:Vec<C>,pub inverse:Vec<C>,
    /// Population rows (column-major n x d) of E V.
    pub readout:Vec<C>,
}

impl BlockSystem{
    /// The class of every step (closed under s -> -s: the clustered real
    /// eigensystem); `horizon`: the longest output time (the clusters' error
    /// control).
    pub fn new(lapack:&Lapack,basis:&BlockEigenbasis,kernels:&SwKernels,n:usize,horizon:f64)->Result<Self,String>{
        let d=basis.dimension();
        let mut m=vec![ZERO;d*d];
        for(j,&l)in basis.lambda.iter().enumerate(){m[j+d*j]=l;}
        for s in 0..kernels.steps.len(){kernels.add_second_order(s,&mut m);}
        let first:Vec<bool>=(0..d).map(|j|basis.partner(j)>j).collect();
        let c=clustered_eigensystem(lapack,&m,&basis.lambda,&first,ClusterControl::blocks(horizon))?;
        drop(m);
        let(values,vectors,inv)=(c.values,c.vectors,c.inverse);
        let gemm=lapack.complex_gemm();let rows=basis.population_rows();
        let mut readout=vec![ZERO;n*d];gemm.multiply(n,d,d,ONE,rows,n,&vectors,d,ZERO,&mut readout,n)?;
        Ok(Self{values,vectors,inverse:inv,readout})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::kernels::KernelRule;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn frame_class_eigensystems_diagonalize_their_blocks(){
        // The class of every step (the clustered real path, to its error
        // control over a short and a long horizon): m V = V diag(mu),
        // V^-1 V = I.
        let f=fixture();let p=prepared(&f);let basis=p.eigenbasis().unwrap();let d=16;
        let k=SwKernels::new(&p,basis,KernelRule{entry:0.,horizon_rate:0.,horizon:1.,cap:f64::INFINITY}).unwrap();
        for(horizon,tolerance)in[(1.,1e-4),(1e6,1e-4)]{
            let mut m=vec![ZERO;d*d];for(j,&l)in basis.lambda.iter().enumerate(){m[j+d*j]=l;}for s in 0..k.steps.len(){k.add_second_order(s,&mut m);}
            let class=BlockSystem::new(&p.lapack,basis,&k,4,horizon).unwrap();
            let scale=m.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for c in 0..d{for r in 0..d{
                let mut mv=ZERO;for l in 0..d{mul_add(&mut mv,m[r+d*l],class.vectors[l+d*c]);}
                let vm=mul(class.vectors[r+d*c],class.values[c]);
                assert!(magnitude(C{re:mv.re-vm.re,im:mv.im-vm.im})<tolerance*scale,"horizon {horizon}: eigenpair {c}");
                let mut id=ZERO;for l in 0..d{mul_add(&mut id,class.inverse[r+d*l],class.vectors[l+d*c]);}
                assert!(magnitude(C{re:id.re-f64::from(u8::from(r==c)),im:id.im})<tolerance,"horizon {horizon}: inverse ({r}, {c})");
            }}
        }
    }

}
