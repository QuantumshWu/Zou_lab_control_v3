//! Dense complex linear algebra for the elimination's small matrices (a few dozen rows): complex
//! arithmetic, products, an LU factorization with partial pivoting and the spectral norm. Each thread
//! counts the complex multiply-adds its products, factorizations, solves and eigenvalue sweeps take
//! (their nominal counts, `counted`): a table node's work (model.rs `tabulate`), the same on any machine.
use crate::{Complex,Matrix,STATES};
use std::cell::Cell;
use std::ops::{Add,AddAssign,Div,Index,IndexMut,Mul,Neg,Sub,SubAssign};

thread_local!{static WORK:Cell<f64>=const{Cell::new(0.)};}
/// The complex multiply-adds this thread's dense algebra has taken so far (nominal counts).
pub fn counted()->f64{WORK.with(|w|w.get())}
fn count(work:usize){WORK.with(|w|w.set(w.get()+work as f64));}

impl Complex{
    pub const ZERO:Complex=Complex{re:0.,im:0.};
    pub const ONE:Complex=Complex{re:1.,im:0.};
    pub const I:Complex=Complex{re:0.,im:1.};
    pub const fn new(re:f64,im:f64)->Complex{Complex{re,im}}
    pub fn conj(self)->Complex{Complex{re:self.re,im:-self.im}}
    pub fn norm_sqr(self)->f64{self.re*self.re+self.im*self.im}
    pub fn abs(self)->f64{self.re.hypot(self.im)}
    /// e^{iφ}.
    pub fn cis(phase:f64)->Complex{let(sin,cos)=phase.sin_cos();Complex{re:cos,im:sin}}
    pub fn exp(self)->Complex{Complex::cis(self.im)*self.re.exp()}
}
impl Add for Complex{type Output=Complex;fn add(self,b:Complex)->Complex{Complex{re:self.re+b.re,im:self.im+b.im}}}
impl Sub for Complex{type Output=Complex;fn sub(self,b:Complex)->Complex{Complex{re:self.re-b.re,im:self.im-b.im}}}
impl Mul for Complex{type Output=Complex;fn mul(self,b:Complex)->Complex{Complex{re:self.re*b.re-self.im*b.im,im:self.re*b.im+self.im*b.re}}}
impl Mul<f64> for Complex{type Output=Complex;fn mul(self,s:f64)->Complex{Complex{re:self.re*s,im:self.im*s}}}
impl Div for Complex{type Output=Complex;fn div(self,b:Complex)->Complex{self*b.conj()*(1./b.norm_sqr())}}
impl Neg for Complex{type Output=Complex;fn neg(self)->Complex{Complex{re:-self.re,im:-self.im}}}
impl AddAssign for Complex{fn add_assign(&mut self,b:Complex){*self=*self+b;}}
impl SubAssign for Complex{fn sub_assign(&mut self,b:Complex){*self=*self-b;}}

/// A dense complex matrix, row-major.
#[derive(Clone,Debug,PartialEq)]
pub struct Dense{pub rows:usize,pub cols:usize,pub entries:Vec<Complex>}
impl Index<(usize,usize)> for Dense{type Output=Complex;fn index(&self,(i,j):(usize,usize))->&Complex{&self.entries[i*self.cols+j]}}
impl IndexMut<(usize,usize)> for Dense{fn index_mut(&mut self,(i,j):(usize,usize))->&mut Complex{&mut self.entries[i*self.cols+j]}}
impl Dense{
    pub fn zeros(rows:usize,cols:usize)->Dense{Dense{rows,cols,entries:vec![Complex::ZERO;rows*cols]}}
    pub fn from_fn(rows:usize,cols:usize,mut entry:impl FnMut(usize,usize)->Complex)->Dense{
        Dense{rows,cols,entries:(0..rows*cols).map(|k|entry(k/cols,k%cols)).collect()}
    }
    /// The rows × cols block of a model matrix.
    pub fn block(matrix:&Matrix,rows:&[usize],cols:&[usize])->Dense{
        Dense::from_fn(rows.len(),cols.len(),|i,j|{let k=rows[i]*STATES+cols[j];Complex{re:matrix.real[k],im:matrix.imag[k]}})
    }
    pub fn adjoint(&self)->Dense{Dense::from_fn(self.cols,self.rows,|i,j|self[(j,i)].conj())}
    /// self · b.
    pub fn product(&self,b:&Dense)->Dense{
        assert_eq!(self.cols,b.rows);let mut out=Dense::zeros(self.rows,b.cols);count(self.rows*self.cols*b.cols);
        for i in 0..self.rows{for k in 0..self.cols{let x=self[(i,k)];if x==Complex::ZERO{continue;}
            for j in 0..b.cols{out[(i,j)]+=x*b[(k,j)];}}}
        out
    }
    /// self† · b.
    pub fn adjoint_product(&self,b:&Dense)->Dense{
        assert_eq!(self.rows,b.rows);let mut out=Dense::zeros(self.cols,b.cols);count(self.rows*self.cols*b.cols);
        for k in 0..self.rows{for i in 0..self.cols{let x=self[(k,i)].conj();if x==Complex::ZERO{continue;}
            for j in 0..b.cols{out[(i,j)]+=x*b[(k,j)];}}}
        out
    }
    /// self += z b.
    pub fn add_scaled(&mut self,z:Complex,b:&Dense){
        assert_eq!((self.rows,self.cols),(b.rows,b.cols));for(x,&y)in self.entries.iter_mut().zip(&b.entries){*x+=z*y;}
    }
    pub fn scaled(&self,z:Complex)->Dense{Dense{rows:self.rows,cols:self.cols,entries:self.entries.iter().map(|&x|z*x).collect()}}
    /// The largest entry's modulus.
    pub fn largest(&self)->f64{self.entries.iter().fold(0.,|a,x|a.max(x.abs()))}
    /// ‖self‖₂, the largest singular value.
    pub fn spectral_norm(&self)->f64{
        let gram=if self.rows>=self.cols{self.adjoint_product(self)}else{self.product(&self.adjoint())};
        hermitian_eigenvalues(&gram).into_iter().fold(0.,f64::max).sqrt()
    }
}

/// The LU factorization with partial pivoting of a square matrix: its rows permuted, A = L U.
#[derive(Clone,Debug)]
pub struct Lu{factors:Dense,pivots:Vec<usize>}
impl Lu{
    /// None when a pivot vanishes: the matrix is singular.
    pub fn new(matrix:&Dense)->Option<Lu>{
        let n=matrix.rows;assert_eq!(n,matrix.cols);let mut a=matrix.clone();let mut pivots=vec![0;n];
        lu_factor(&mut a.entries,n,&mut pivots).then_some(Lu{factors:a,pivots})
    }
    /// b ← A⁻¹ b.
    pub fn solve(&self,b:&mut [Complex]){lu_solve(&self.factors.entries,&self.pivots,self.factors.rows,b)}
    /// A⁻¹ b for every column of b.
    pub fn solve_columns(&self,b:&Dense)->Dense{
        let mut out=b.clone();let mut column=vec![Complex::ZERO;b.rows];
        for j in 0..b.cols{for i in 0..b.rows{column[i]=b[(i,j)];}self.solve(&mut column);for i in 0..b.rows{out[(i,j)]=column[i];}}
        out
    }
}

/// The LU factorization with partial pivoting of the n × n matrix in `a`, in place (row-major), its pivots; false when a pivot vanishes.
pub(crate) fn lu_factor(a:&mut [Complex],n:usize,pivots:&mut [usize])->bool{
    count(n*n*n/3);
    for k in 0..n{
        let p=(k..n).max_by(|&i,&j|a[i*n+k].norm_sqr().total_cmp(&a[j*n+k].norm_sqr())).unwrap();
        if a[p*n+k]==Complex::ZERO{return false;}
        pivots[k]=p;for j in 0..n{a.swap(k*n+j,p*n+j);}
        let pivot=a[k*n+k];
        for i in k+1..n{let factor=a[i*n+k]/pivot;a[i*n+k]=factor;for j in k+1..n{let u=a[k*n+j];a[i*n+j]-=factor*u;}}
    }
    true
}
/// b ← A⁻¹ b from `lu_factor`'s factors and pivots.
pub(crate) fn lu_solve(a:&[Complex],pivots:&[usize],n:usize,b:&mut [Complex]){
    count(n*n);
    for(k,&p)in pivots.iter().enumerate(){b.swap(k,p);}
    for i in 0..n{let mut x=b[i];for j in 0..i{x-=a[i*n+j]*b[j];}b[i]=x;}
    for i in (0..n).rev(){let mut x=b[i];for j in i+1..n{x-=a[i*n+j]*b[j];}b[i]=x/a[i*n+i];}
}
/// A term of a sum of size `scale` at or within this is that sum's rounding, no term: 64 ε of the scale.
pub(crate) fn round_off(scale:f64)->f64{64.*f64::EPSILON*scale}
/// The same at one ε of `scale`: the sites that read a term against a single number's precision.
pub(crate) fn unit_round_off(scale:f64)->f64{f64::EPSILON*scale}
/// The same at the kernel's single precision, one ε of `f32`: what the GPU's generator cannot hold.
pub(crate) fn single_round_off(scale:f64)->f64{f32::EPSILON as f64*scale}
/// One Jacobi rotation of a real symmetric 3 × 3 matrix in the plane (p, q), a[p][q] ≠ 0, its eigenvectors the rows of v.
fn rotate_symmetric(a:&mut [[f64;3];3],v:&mut [[f64;3];3],p:usize,q:usize){
    let theta=(a[q][q]-a[p][p])/(2.*a[p][q]);let t=theta.signum()/(theta.abs()+(theta*theta+1.).sqrt());let c=1./(t*t+1.).sqrt();let s=t*c;
    for k in 0..3{let(x,y)=(a[k][p],a[k][q]);a[k][p]=c*x-s*y;a[k][q]=s*x+c*y;}
    for k in 0..3{let(x,y)=(a[p][k],a[q][k]);a[p][k]=c*x-s*y;a[q][k]=s*x+c*y;}
    for k in 0..3{let(x,y)=(v[p][k],v[q][k]);v[p][k]=c*x-s*y;v[q][k]=s*x+c*y;}
}
/// The eigen-decomposition of a real symmetric 3 × 3 matrix: its eigenvalues and orthonormal eigenvectors (rows), by cyclic Jacobi rotations.
pub(crate) fn symmetric_eigen(m:[[f64;3];3])->([f64;3],[[f64;3];3]){
    let mut a=m;let mut v=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
    for _ in 0..64{
        let off=a[0][1].abs()+a[0][2].abs()+a[1][2].abs();if off<=1e-15*(a[0][0].abs()+a[1][1].abs()+a[2][2].abs()){break;}
        for(p,q)in [(0,1),(0,2),(1,2)]{if a[p][q]!=0.{rotate_symmetric(&mut a,&mut v,p,q);}}
    }
    ([a[0][0],a[1][1],a[2][2]],v)
}
/// The same with the largest off-diagonal entry rotated first, to one ε: the order the CPU's `part_weights` (engine.rs) takes. Either order on the other's matrices moves every chaotic trajectory's digits, none of their statistics.
pub(crate) fn symmetric_eigen_pivoted(m:[[f64;3];3])->([f64;3],[[f64;3];3]){
    let mut a=m;let mut v=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
    for _ in 0..64{
        let(mut p,mut q,mut big)=(0,1,0.);for i in 0..3{for j in i+1..3{if a[i][j].abs()>big{big=a[i][j].abs();p=i;q=j;}}}
        if !(big>f64::EPSILON*(a[0][0].abs()+a[1][1].abs()+a[2][2].abs())){break;}
        rotate_symmetric(&mut a,&mut v,p,q);
    }
    ([a[0][0],a[1][1],a[2][2]],v)
}
/// `x` emptied and filled with `len` zeros: a buffer kept for its capacity.
pub(crate) fn zeroed<T:Clone+Default>(x:&mut Vec<T>,len:usize){x.clear();x.resize(len,T::default());}

/// The eigenvalues of a Hermitian matrix: cyclic Jacobi rotations until the off-diagonal part is
/// below the double precision of the matrix or stops shrinking.
pub fn hermitian_eigenvalues(matrix:&Dense)->Vec<f64>{jacobi(matrix,None,&|_,_|true)}
/// The eigendecomposition of a Hermitian matrix, H = U diag(λ) U†: the same rotations, U their product --
/// its columns orthonormal to round-off, in a degenerate eigenspace too.
pub fn hermitian_eigen(matrix:&Dense)->(Vec<f64>,Dense){
    let mut u=Dense::zeros(matrix.rows,matrix.rows);for i in 0..matrix.rows{u[(i,i)]=Complex::ONE;}
    let values=jacobi(matrix,Some(&mut u),&|_,_|true);(values,u)
}
/// The same as far as a coupling matters over a width w: the rotations of a pair only while |H_pq|² exceeds share² (w² + (H_pp − H_qq)²), its mixing's
/// size whether its separation or the width bounds it, until a sweep rotates none.
pub fn hermitian_eigen_within(matrix:&Dense,width:f64,share:f64)->(Vec<f64>,Dense){
    let mut u=Dense::zeros(matrix.rows,matrix.rows);for i in 0..matrix.rows{u[(i,i)]=Complex::ONE;}
    let values=jacobi(matrix,Some(&mut u),&|coupling,apart|coupling>share*share*(width*width+apart));(values,u)
}
/// Cyclic Jacobi rotations of the pairs `matters` keeps by |H_pq|² and (H_pp − H_qq)².
fn jacobi(matrix:&Dense,mut vectors:Option<&mut Dense>,matters:&dyn Fn(f64,f64)->bool)->Vec<f64>{
    let n=matrix.rows;let mut a=matrix.clone();
    let total:f64=a.entries.iter().map(|x|x.norm_sqr()).sum();
    let mut previous=f64::INFINITY;
    loop{
        let off:f64=(0..n).map(|i|(0..n).filter(|&j|j!=i).map(|j|a[(i,j)].norm_sqr()).sum::<f64>()).sum();
        if off<=f64::EPSILON*f64::EPSILON*total||!(off<previous){break;}
        previous=off;count(2*n*n*n);let mut rotated=false;
        for p in 0..n{for q in p+1..n{
            let modulus=a[(p,q)].abs();if modulus==0.||!matters(modulus*modulus,(a[(p,p)].re-a[(q,q)].re).powi(2)){continue;}rotated=true;
            // A unitary phase on q makes a_pq real and positive; a real rotation in (p, q) then zeroes it.
            // The phase divides by the modulus, whose reciprocal overflows when it is subnormal.
            let phase=Complex::new(a[(p,q)].re/modulus,a[(p,q)].im/modulus);
            for k in 0..n{a[(k,q)]=a[(k,q)]*phase.conj();a[(q,k)]=a[(q,k)]*phase;}
            if let Some(u)=vectors.as_deref_mut(){for k in 0..n{u[(k,q)]=u[(k,q)]*phase.conj();}}
            let tau=(a[(q,q)].re-a[(p,p)].re)/(2.*modulus);
            let t=tau.signum()/(tau.abs()+(1.+tau*tau).sqrt());let c=1./(1.+t*t).sqrt();let s=t*c;
            for k in 0..n{let(x,y)=(a[(k,p)],a[(k,q)]);a[(k,p)]=x*c-y*s;a[(k,q)]=x*s+y*c;}
            for k in 0..n{let(x,y)=(a[(p,k)],a[(q,k)]);a[(p,k)]=x*c-y*s;a[(q,k)]=x*s+y*c;}
            if let Some(u)=vectors.as_deref_mut(){for k in 0..n{let(x,y)=(u[(k,p)],u[(k,q)]);u[(k,p)]=x*c-y*s;u[(k,q)]=x*s+y*c;}}
            a[(p,q)]=Complex::ZERO;a[(q,p)]=Complex::ZERO;
        }}
        if !rotated{break;}
    }
    (0..n).map(|i|a[(i,i)].re).collect()
}

/// A reproducible stream of numbers uniform in [−½, ½), for the tests.
#[cfg(test)]
pub(crate) fn uniform(seed:u64)->impl FnMut()->f64{
    let mut state=seed;
    move||{state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(state>>11) as f64/(1u64<<53) as f64-0.5}
}

#[cfg(test)]
mod tests{
    use super::*;
    /// A reproducible matrix of unit-scale entries.
    fn sample(rows:usize,cols:usize,seed:u64)->Dense{let mut next=uniform(seed);Dense::from_fn(rows,cols,|_,_|Complex::new(next(),next()))}
    #[test]
    fn the_lu_solves_and_the_jacobi_eigenvalues_give_the_spectral_norm(){
        let a=sample(24,24,1);let b=sample(24,3,2);
        let x=Lu::new(&a).unwrap().solve_columns(&b);let mut residual=a.product(&x);residual.add_scaled(-Complex::ONE,&b);
        assert!(residual.largest()<=1e-12*b.largest(),"{}",residual.largest());
        // A Hermitian matrix with known eigenvalues: U diag(d) U†, U unitary from a Hermitian H's Cayley transform.
        let h=sample(6,6,3);let h=Dense::from_fn(6,6,|i,j|(h[(i,j)]+h[(j,i)].conj())*0.5);
        let mut plus=h.scaled(Complex::I);let mut minus=h.scaled(-Complex::I);for i in 0..6{plus[(i,i)]+=Complex::ONE;minus[(i,i)]+=Complex::ONE;}
        let u=Lu::new(&plus).unwrap().solve_columns(&minus);
        let d=[3.,-1.,0.5,2.,-4.,1e-3];let diagonal=Dense::from_fn(6,6,|i,j|if i==j{Complex::new(d[i],0.)}else{Complex::ZERO});
        let mut eigen=hermitian_eigenvalues(&u.product(&diagonal).product(&u.adjoint()));eigen.sort_by(f64::total_cmp);
        let mut expected=d.to_vec();expected.sort_by(f64::total_cmp);
        assert!(eigen.iter().zip(&expected).all(|(x,y)|(x-y).abs()<=1e-13),"{eigen:?}");
        assert!((u.product(&diagonal).spectral_norm()-4.).abs()<=1e-13);
        // A subnormal off-diagonal element, whose reciprocal overflows: rotated away, no NaN.
        let tiny=Dense::from_fn(2,2,|i,j|if i==j{Complex::new(1.+i as f64,0.)}else{Complex::new(5e-320,if i<j{3e-320}else{-3e-320})});
        assert_eq!(hermitian_eigenvalues(&tiny),vec![1.,2.]);
    }
    /// As far as a coupling matters: a pair within the share over the width is left, one beyond it rotated away; the matrix is U diag(λ) U† but for what is left.
    #[test]
    fn the_eigenstates_within_a_share_leave_the_couplings_that_do_not_matter(){
        let n=4;let mut h=Dense::zeros(n,n);for(i,e)in [0.,3.,50.,50.2].iter().enumerate(){h[(i,i)]=Complex::new(*e,0.);}
        let mut couple=|a:usize,b:usize,z:Complex|{h[(a,b)]=z;h[(b,a)]=z.conj();};
        couple(0,1,Complex::new(0.,2.));couple(2,3,Complex::new(0.3,0.1));couple(0,2,Complex::new(1e-4,0.));
        let(share,width)=(1e-3,1.);let(values,u)=hermitian_eigen_within(&h,width,share);
        let diagonal=Dense::from_fn(n,n,|i,j|if i==j{Complex::new(values[i],0.)}else{Complex::ZERO});
        let mut rest=u.product(&diagonal).product(&u.adjoint());rest.add_scaled(-Complex::ONE,&h);
        assert!(rest.largest()<=2e-4,"{}",rest.largest());
        let exact=hermitian_eigenvalues(&h);let mut a=values.clone();let mut b=exact.clone();a.sort_by(f64::total_cmp);b.sort_by(f64::total_cmp);
        for(x,y)in a.iter().zip(&b){assert!((x-y).abs()<=1e-6,"{a:?} {b:?}");}
    }
}
