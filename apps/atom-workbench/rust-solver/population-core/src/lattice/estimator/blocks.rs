//! The second-order blocks of the error propagation: per mode q of the
//! lattice, D'_q = L0' - i nu_q + sum_s K_s over the steps s whose
//! neighbour q + s is on the lattice (`kernels`). Modes with the same steps
//! share one eigensystem (their blocks differ by -i nu_q only): a class.
//!
//! In eigen-coordinates of L0' a class is diag(lambda) + sum_s K~_s, whose
//! eigendecomposition V diag(mu) V^-1 propagates any source exactly. The
//! class also stores its readouts: the population rows of E V (the mode's
//! own populations) and, per step s, of E X~_s V (what the first-order
//! transform I + S carries from the mode to its neighbour q + s).
//!
//! A class whose steps come in pairs (s, -s) is invariant under the flip J,
//! so its matrix is real in the real eigen-coordinates of L0' (the pair
//! rule of `eigenbasis`): real eigendecompositions and LUs, about a quarter
//! of the complex ones. Most couplings of diag(lambda) + K~ are far below
//! their gaps, so the eigensystem is built by clusters of strongly coupled
//! coordinates, exact within each and perturbative between them, with the
//! mixing and the values' error over the request's horizon bounded
//! (`eigenbasis::clustered_eigensystem`, the centre's own method).
use super::super::*;
use super::super::eigenbasis::{BlockEigenbasis,ClusterControl,clustered_eigensystem};
use super::super::floquet::inverse;
use super::kernels::SwKernels;

#[derive(Clone)]
pub struct BlockSystem{
    /// The steps whose second-order terms the class carries.
    pub steps:Vec<usize>,
    /// Eigenvalues mu (the mode's own -i nu_q not included), the
    /// eigenvectors V and V^-1 (column-major d x d, eigen-coordinates of L0').
    pub values:Vec<C>,pub vectors:Vec<C>,pub inverse:Vec<C>,
    /// Population rows (column-major n x d) of E V, and of E X~_s V per step
    /// (None for steps outside the class).
    pub readout:Vec<C>,pub step_readout:Vec<Option<Vec<C>>>,
    /// |V|_1 |V^-1|_1, and the largest cluster of the eigensystem.
    pub condition:f64,pub largest_cluster:usize,
}

impl BlockSystem{
    /// `horizon`: the longest output time (the clusters' error control).
    pub fn new(lapack:&Lapack,basis:&BlockEigenbasis,kernels:&SwKernels,steps:Vec<usize>,n:usize,horizon:f64)->Result<Self,String>{
        let d=basis.dimension();
        let mut m=vec![ZERO;d*d];
        for(j,&l)in basis.lambda.iter().enumerate(){m[j+d*j]=l;}
        for &s in &steps{kernels.add_second_order(s,&mut m);}
        let closed=steps.iter().all(|&s|steps.contains(&kernels.steps[s].opposite));
        let(values,vectors,inv,largest)=if closed{
            let first:Vec<bool>=(0..d).map(|j|basis.partner(j)>j).collect();
            let c=clustered_eigensystem(lapack,&m,&basis.lambda,&first,ClusterControl::blocks(horizon))?;(c.values,c.vectors,c.inverse,c.largest)
        }else{let(values,vectors)=lapack.complex_eigen_right(&m,d)?;let inv=inverse(lapack,&vectors,d)?;(values,vectors,inv,d)};
        drop(m);
        let one=|a:&[C]|(0..d).map(|c|a[c*d..(c+1)*d].iter().map(|z|magnitude(*z)).sum::<f64>()).fold(0.,f64::max);
        let condition=one(&vectors)*one(&inv);
        let gemm=lapack.complex_gemm();let rows=basis.population_rows();
        let mut readout=vec![ZERO;n*d];gemm.multiply(n,d,d,ONE,rows,n,&vectors,d,ZERO,&mut readout,n)?;
        let mut step_readout:Vec<Option<Vec<C>>>=(0..kernels.steps.len()).map(|_|None).collect();
        for &s in &steps{
            let mut rx=vec![ZERO;n*d];
            for &(i,j,x)in &kernels.steps[s].entries{for r in 0..n{mul_add(&mut rx[r+n*j],rows[r+n*i],x);}}
            let mut out=vec![ZERO;n*d];gemm.multiply(n,d,d,ONE,&rx,n,&vectors,d,ZERO,&mut out,n)?;
            step_readout[s]=Some(out);
        }
        Ok(Self{steps,values,vectors,inverse:inv,readout,step_readout,condition,largest_cluster:largest})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::kernels::KernelRule;
    use super::super::super::tests::{fixture,prepared};
    #[test]
    fn estimator_class_eigensystems_diagonalize_their_blocks(){
        // A flip-invariant class (both steps of the pair: the clustered real
        // path, to its error control over a short and a long horizon) and a
        // one-sided class (the complex path, exact): m V = V diag(mu), V^-1 V = I.
        let f=fixture();let p=prepared(&f);let basis=p.eigenbasis().unwrap();let d=16;
        let k=SwKernels::new(&p,basis,KernelRule{entry:0.,horizon_rate:0.,horizon:1.,cap:f64::INFINITY}).unwrap();
        for(steps,horizon,tolerance)in[((0..k.steps.len()).collect::<Vec<_>>(),1.,1e-4),((0..k.steps.len()).collect(),1e6,1e-4),(vec![0],1.,1e-10)]{
            let mut m=vec![ZERO;d*d];for(j,&l)in basis.lambda.iter().enumerate(){m[j+d*j]=l;}for &s in &steps{k.add_second_order(s,&mut m);}
            let class=BlockSystem::new(&p.lapack,basis,&k,steps.clone(),4,horizon).unwrap();
            let scale=m.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for c in 0..d{for r in 0..d{
                let mut mv=ZERO;for l in 0..d{mul_add(&mut mv,m[r+d*l],class.vectors[l+d*c]);}
                let vm=mul(class.vectors[r+d*c],class.values[c]);
                assert!(magnitude(C{re:mv.re-vm.re,im:mv.im-vm.im})<tolerance*scale,"steps {steps:?}, horizon {horizon}: eigenpair {c}");
                let mut id=ZERO;for l in 0..d{mul_add(&mut id,class.inverse[r+d*l],class.vectors[l+d*c]);}
                assert!(magnitude(C{re:id.re-f64::from(u8::from(r==c)),im:id.im})<tolerance,"steps {steps:?}, horizon {horizon}: inverse ({r}, {c})");
            }}
        }
    }

}
