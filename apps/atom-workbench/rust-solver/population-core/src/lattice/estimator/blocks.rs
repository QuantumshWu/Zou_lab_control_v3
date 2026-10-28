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
//! rule of `eigenbasis`): one real eigendecomposition and one real LU, about
//! a quarter of the complex ones.
use super::super::*;
use super::super::eigenbasis::BlockEigenbasis;
use super::super::floquet::inverse;
use super::kernels::SwKernels;

pub struct BlockSystem{
    /// The steps whose second-order terms the class carries.
    pub steps:Vec<usize>,
    /// Eigenvalues mu (the mode's own -i nu_q not included), the
    /// eigenvectors V and V^-1 (column-major d x d, eigen-coordinates of L0').
    pub values:Vec<C>,pub vectors:Vec<C>,pub inverse:Vec<C>,
    /// Population rows (column-major n x d) of E V, and of E X~_s V per step
    /// (None for steps outside the class).
    pub readout:Vec<C>,pub step_readout:Vec<Option<Vec<C>>>,
    /// |V|_1 |V^-1|_1.
    pub condition:f64,
}

impl BlockSystem{
    pub fn new(lapack:&Lapack,basis:&BlockEigenbasis,kernels:&SwKernels,steps:Vec<usize>,n:usize)->Result<Self,String>{
        let d=basis.dimension();
        let mut m=vec![ZERO;d*d];
        for(j,&l)in basis.lambda.iter().enumerate(){m[j+d*j]=l;}
        for &s in &steps{kernels.add_second_order(s,&mut m);}
        let closed=steps.iter().all(|&s|steps.contains(&kernels.steps[s].opposite));
        let(values,vectors,inv)=if closed{real_eigensystem(lapack,basis,&m,d)?}else{
            let(values,vectors)=lapack.complex_eigen_right(&m,d)?;let inv=inverse(lapack,&vectors,d)?;(values,vectors,inv)};
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
        Ok(Self{steps,values,vectors,inverse:inv,readout,step_readout,condition})
    }
}

/// The eigensystem of a flip-invariant class matrix m (column-major, complex
/// eigen-coordinates z of L0'). With the real coordinates x of a conjugate
/// pair (j, j + 1), z = T x, T = 1/2 [[1, -i], [1, i]], the matrix T^-1 m T
/// is real; its real eigenvectors VR give the complex ones V_c = VR S,
/// S = [[1, 1], [i, -i]] per eigenvalue pair, so V = T V_c and
/// V^-1 = S^-1 VR^-1 T^-1.
fn real_eigensystem(lapack:&Lapack,basis:&BlockEigenbasis,m:&[C],d:usize)->Result<(Vec<C>,Vec<C>,Vec<C>),String>{
    let first=|j:usize|basis.partner(j)>j;
    let i_times=|z:C|C{re:-z.im,im:z.re};
    // Y = m T (columns), then x = T^-1 Y (rows): real up to roundoff.
    let mut y=m.to_vec();
    for j in(0..d).filter(|&j|first(j)){for i in 0..d{let(a,b)=(m[i+d*j],m[i+d*(j+1)]);
        y[i+d*j]=C{re:0.5*(a.re+b.re),im:0.5*(a.im+b.im)};y[i+d*(j+1)]=i_times(C{re:0.5*(b.re-a.re),im:0.5*(b.im-a.im)});}}
    let(mut real,mut imaginary,mut size)=(vec![0.;d*d],0_f64,0_f64);
    for c in 0..d{let mut j=0;while j<d{
        if first(j){let(a,b)=(y[j+d*c],y[j+1+d*c]);let(u,v)=(C{re:a.re+b.re,im:a.im+b.im},i_times(C{re:a.re-b.re,im:a.im-b.im}));
            real[j+d*c]=u.re;real[j+1+d*c]=v.re;imaginary=imaginary.max(u.im.abs()).max(v.im.abs());size=size.max(u.re.abs()).max(v.re.abs());j+=2;}
        else{real[j+d*c]=y[j+d*c].re;imaginary=imaginary.max(y[j+d*c].im.abs());size=size.max(y[j+d*c].re.abs());j+=1;}}}
    drop(y);
    if imaginary>1e-10*size{return Err(format!("Estimator class is not flip invariant ({:e})",imaginary/size));}
    let(wr,wi,vr)=lapack.eigen(&real,d)?;drop(real);
    if vr.iter().any(|v|!v.is_finite()){return Err("Nonfinite estimator class eigenvectors".into());}
    let(lu,pivots)=lapack.factor(&vr,d)?;let mut identity=vec![0.;d*d];for k in 0..d{identity[k+d*k]=1.;}
    let vri=lapack.solve_matrix_factored(&lu,&pivots,&identity,d)?;
    let mut pair=vec![false;d];let mut l=0;while l<d{if wi[l]!=0.&&l+1<d{pair[l]=true;l+=2;}else{l+=1;}}
    let values:Vec<C>=(0..d).map(|l|if pair[l]{C{re:wr[l],im:wi[l]}}else if l>0&&pair[l-1]{C{re:wr[l],im:-wi[l-1]}}else{C{re:wr[l],im:0.}}).collect();
    // V_c = VR S (columns), V = T V_c (rows).
    let mut vc:Vec<C>=vr.iter().map(|&v|C{re:v,im:0.}).collect();
    for l in(0..d).filter(|&l|pair[l]){for i in 0..d{let(a,b)=(vr[i+d*l],vr[i+d*(l+1)]);vc[i+d*l]=C{re:a,im:b};vc[i+d*(l+1)]=C{re:a,im:-b};}}
    let mut vectors=vc.clone();
    for c in 0..d{for j in(0..d).filter(|&j|first(j)){let(a,b)=(vc[j+d*c],vc[j+1+d*c]);let ib=i_times(b);
        vectors[j+d*c]=C{re:0.5*(a.re-ib.re),im:0.5*(a.im-ib.im)};vectors[j+1+d*c]=C{re:0.5*(a.re+ib.re),im:0.5*(a.im+ib.im)};}}
    // V^-1 = S^-1 VR^-1 (rows), then T^-1 (columns).
    let mut w:Vec<C>=vri.iter().map(|&v|C{re:v,im:0.}).collect();
    for l in(0..d).filter(|&l|pair[l]){for c in 0..d{let(a,b)=(vri[l+d*c],vri[l+1+d*c]);
        w[l+d*c]=C{re:0.5*a,im:-0.5*b};w[l+1+d*c]=C{re:0.5*a,im:0.5*b};}}
    let mut inv=w.clone();
    for j in(0..d).filter(|&j|first(j)){for r in 0..d{let(a,b)=(w[r+d*j],w[r+d*(j+1)]);let ib=i_times(b);
        inv[r+d*j]=C{re:a.re+ib.re,im:a.im+ib.im};inv[r+d*(j+1)]=C{re:a.re-ib.re,im:a.im-ib.im};}}
    Ok((values,vectors,inv))
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::kernels::KernelRule;
    use super::super::super::tests::{fixture,prepared};
    #[test]
    fn estimator_class_eigensystems_diagonalize_their_blocks(){
        // A flip-invariant class (both steps of the pair: the real path) and a
        // one-sided class (the complex path): m V = V diag(mu), V^-1 V = I.
        let f=fixture();let p=prepared(&f);let basis=p.eigenbasis().unwrap();let d=16;
        let k=SwKernels::new(&p,basis,KernelRule{entry:0.,horizon_rate:0.,horizon:1.,cap:f64::INFINITY}).unwrap();
        for steps in[(0..k.steps.len()).collect::<Vec<_>>(),vec![0]]{
            let mut m=vec![ZERO;d*d];for(j,&l)in basis.lambda.iter().enumerate(){m[j+d*j]=l;}for &s in &steps{k.add_second_order(s,&mut m);}
            let class=BlockSystem::new(&p.lapack,basis,&k,steps.clone(),4).unwrap();
            let scale=m.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for c in 0..d{for r in 0..d{
                let mut mv=ZERO;for l in 0..d{mul_add(&mut mv,m[r+d*l],class.vectors[l+d*c]);}
                let vm=mul(class.vectors[r+d*c],class.values[c]);
                assert!(magnitude(C{re:mv.re-vm.re,im:mv.im-vm.im})<1e-10*scale,"steps {steps:?}: eigenpair {c}");
                let mut id=ZERO;for l in 0..d{mul_add(&mut id,class.inverse[r+d*l],class.vectors[l+d*c]);}
                assert!(magnitude(C{re:id.re-f64::from(u8::from(r==c)),im:id.im})<1e-10,"steps {steps:?}: inverse ({r}, {c})");
            }}
        }
    }
}
