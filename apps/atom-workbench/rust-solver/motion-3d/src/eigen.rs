//! The eigendecomposition of a small complex matrix (row-major n × n), H = R Λ R⁻¹: Householder
//! reduction to Hessenberg form, the shifted QR iteration to the complex Schur form T = Q† H Q
//! (Wilkinson shifts, deflation at the double precision of the neighbouring diagonal), the
//! eigenvectors of T by back substitution and R = Q Y, its columns of unit norm, and R⁻¹ by LU.
//! Its condition, ‖R‖₁‖R⁻¹‖₁, bounds the rounding of what is computed through it: near an exceptional
//! point it grows without bound, and the exact step (engine.rs) then takes its grid instead.
use crate::linear::{lu_factor,lu_solve,zeroed};
use crate::Complex;

/// H = R diag(λ) R⁻¹ with the condition of R, and the buffers its computation works in: one kept for its next generator takes no memory (the exact step's, a decomposition per step).
#[derive(Default)]
pub struct Eigen{pub n:usize,pub values:Vec<Complex>,pub vectors:Vec<Complex>,pub inverse:Vec<Complex>,pub condition:f64,a:Vec<Complex>,q:Vec<Complex>,v:Vec<Complex>,y:Vec<Complex>,factors:Vec<Complex>,pivots:Vec<usize>}

fn abs1(z:Complex)->f64{z.re.abs()+z.im.abs()}

/// The eigendecomposition of an n × n matrix, or None when the QR iteration does not converge or R is
/// singular.
pub fn eigen(h:&[Complex],n:usize)->Option<Eigen>{let mut e=Eigen::default();decompose(&mut e,h,n).then_some(e)}
/// `eigen` into a kept `Eigen`; false when the QR iteration does not converge or R is singular.
pub fn decompose(e:&mut Eigen,h:&[Complex],n:usize)->bool{
    e.n=n;let Eigen{values,vectors:r,inverse,condition,a,q,v,y,factors,pivots,..}=e;
    a.clear();a.extend_from_slice(&h[..n*n]);zeroed(q,n*n);for i in 0..n{q[i*n+i]=Complex::ONE;}
    // Householder reduction to upper Hessenberg form, A ← P A P, Q ← Q P.
    zeroed(v,n);
    for k in 0..n.saturating_sub(2){
        let alpha:f64=(k+1..n).map(|i|a[i*n+k].norm_sqr()).sum::<f64>().sqrt();if alpha==0.{continue;}
        let x0=a[(k+1)*n+k];let phase=if x0.abs()>0.{x0*(1./x0.abs())}else{Complex::ONE};
        for i in 0..n{v[i]=Complex::ZERO;}
        for i in k+1..n{v[i]=a[i*n+k];}v[k+1]+=phase*alpha;
        let vv:f64=(k+1..n).map(|i|v[i].norm_sqr()).sum();if vv==0.{continue;}let beta=2./vv;
        // A ← (I − β v v†) A (I − β v v†); Q ← Q (I − β v v†).
        for j in 0..n{let s=(k+1..n).fold(Complex::ZERO,|s,i|s+v[i].conj()*a[i*n+j])*beta;for i in k+1..n{a[i*n+j]-=v[i]*s;}}
        for i in 0..n{let s=(k+1..n).fold(Complex::ZERO,|s,j|s+a[i*n+j]*v[j])*beta;for j in k+1..n{a[i*n+j]-=s*v[j].conj();}}
        for i in 0..n{let s=(k+1..n).fold(Complex::ZERO,|s,j|s+q[i*n+j]*v[j])*beta;for j in k+1..n{q[i*n+j]-=s*v[j].conj();}}
        for i in k+2..n{a[i*n+k]=Complex::ZERO;}
    }
    // The shifted QR iteration on the active block [lo, hi], Givens rotations accumulated into Q.
    let scale=a.iter().map(|z|abs1(*z)).fold(0.,f64::max).max(f64::MIN_POSITIVE);
    let mut hi=n;let mut iterations=0;
    while hi>1{
        let mut lo=hi-1;
        while lo>0{let s=abs1(a[(lo-1)*n+lo-1])+abs1(a[lo*n+lo]);if abs1(a[lo*n+lo-1])<=f64::EPSILON*if s>0.{s}else{scale}{a[lo*n+lo-1]=Complex::ZERO;break;}lo-=1;}
        if lo==hi-1{hi-=1;iterations=0;continue;}
        iterations+=1;if iterations>60*n{return false;}
        // The Wilkinson shift: the eigenvalue of the trailing 2 × 2 block nearer its corner (an
        // exceptional shift every tenth iteration).
        let(p,qq,r,s)=(a[(hi-2)*n+hi-2],a[(hi-2)*n+hi-1],a[(hi-1)*n+hi-2],a[(hi-1)*n+hi-1]);
        let mut mu=if iterations%10==0{s+Complex::new(abs1(a[(hi-1)*n+hi-2]),0.)}else{
            let half=(p-s)*0.5;let d=csqrt(half*half+qq*r);let(m1,m2)=(s-(d-half),s+(d+half));
            let(e1,e2)=(m1-s,m2-s);if abs1(e1)<=abs1(e2){m1}else{m2}};
        if !(mu.re.is_finite()&&mu.im.is_finite()){mu=s;}
        // One QR sweep of (A − μ) on [lo, hi) by Givens rotations, then RQ + μ.
        for k in lo..hi-1{
            let x=if k==lo{a[k*n+k]-mu}else{a[k*n+k-1]};let y=if k==lo{a[(k+1)*n+k]}else{a[(k+1)*n+k-1]};
            let(c,sn)=givens(x,y);
            // Rows k, k+1 from column (k−1 or lo) on.
            let start=if k==lo{lo}else{k-1};
            for j in start..n{let(u,w)=(a[k*n+j],a[(k+1)*n+j]);a[k*n+j]=u*c+w*sn;a[(k+1)*n+j]=w*c-u*sn.conj();}
            // Columns k, k+1 of the rows above (up to k + 2).
            let end=(k+3).min(hi);
            for i in 0..end{let(u,w)=(a[i*n+k],a[i*n+k+1]);a[i*n+k]=u*c+w*sn.conj();a[i*n+k+1]=w*c-u*sn;}
            for i in 0..n{let(u,w)=(q[i*n+k],q[i*n+k+1]);q[i*n+k]=u*c+w*sn.conj();q[i*n+k+1]=w*c-u*sn;}
            if k>lo{a[(k+1)*n+k-1]=Complex::ZERO;}
        }
    }
    // The eigenvectors of T by back substitution, near-equal eigenvalues kept apart by ε‖T‖.
    values.clear();values.extend((0..n).map(|i|a[i*n+i]));
    let floor=f64::EPSILON*scale;
    zeroed(y,n*n);
    for k in 0..n{
        y[k*n+k]=Complex::ONE;
        for j in (0..k).rev(){
            let s=(j+1..=k).fold(Complex::ZERO,|s,m|s+a[j*n+m]*y[m*n+k]);
            let mut d=a[j*n+j]-values[k];if abs1(d)<floor{d=Complex::new(floor,0.);}
            y[j*n+k]=-(s/d);
        }
    }
    // R = Q Y, columns of unit norm.
    zeroed(r,n*n);
    for i in 0..n{for k in 0..n{r[i*n+k]=(0..=k).fold(Complex::ZERO,|s,m|s+q[i*n+m]*y[m*n+k]);}}
    for k in 0..n{let norm=(0..n).map(|i|r[i*n+k].norm_sqr()).sum::<f64>().sqrt();if !(norm>0.){return false;}for i in 0..n{r[i*n+k]=r[i*n+k]*(1./norm);}}
    // R⁻¹, column by column; v the column.
    factors.clear();factors.extend_from_slice(r);zeroed(pivots,n);if !lu_factor(factors,n,pivots){return false;}
    zeroed(inverse,n*n);
    for j in 0..n{for i in 0..n{v[i]=if i==j{Complex::ONE}else{Complex::ZERO};}lu_solve(factors,pivots,n,&mut v[..n]);for i in 0..n{inverse[i*n+j]=v[i];}}
    let norm1=|m:&[Complex]|(0..n).map(|j|(0..n).map(|i|m[i*n+j].abs()).sum::<f64>()).fold(0.,f64::max);
    *condition=norm1(r)*norm1(inverse);
    condition.is_finite()
}

/// √z on the principal branch.
pub fn csqrt(z:Complex)->Complex{
    let r=z.abs();if r==0.{return Complex::ZERO;}
    let re=((r+z.re)*0.5).max(0.).sqrt();let im=((r-z.re)*0.5).max(0.).sqrt();
    Complex::new(re,if z.im<0.{-im}else{im})
}
/// The Givens rotation G = [[c, s], [−s*, c]] with G [x, y]ᵀ = [r, 0]ᵀ (c real).
fn givens(x:Complex,y:Complex)->(f64,Complex){
    let(ax,ay)=(x.abs(),y.abs());if ay==0.{return(1.,Complex::ZERO);}if ax==0.{return(0.,(y.conj())*(1./ay));}
    let norm=ax.hypot(ay);let c=ax/norm;let phase=x*(1./ax);(c,phase*y.conj()*(1./norm))
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::linear::uniform;
    /// R Λ R⁻¹ gives H back to the double precision of its condition, for random non-Hermitian matrices
    /// of 1 to 32 states (an H_eff's shape: Hermitian plus −(i/2) a positive part), and for a defective
    /// one nearly (an exceptional point: the condition large, the residual still small).
    #[test]
    fn the_eigendecomposition_gives_the_matrix_back(){
        for n in [1,2,3,5,8,12,13,20,32]{for seed in 1..=3{
            let mut next=uniform(seed*100+n as u64);
            let mut h=vec![Complex::ZERO;n*n];
            for i in 0..n{for j in 0..=i{let z=Complex::new(next()*50.,next()*50.);if i==j{h[i*n+i]=Complex::new(z.re*8.,-19.*(next()+0.5));}else{h[i*n+j]=z;h[j*n+i]=z.conj();}}}
            let e=eigen(&h,n).expect("converges");
            let mut worst=0_f64;let scale=h.iter().map(|z|z.abs()).fold(0.,f64::max);
            for i in 0..n{for j in 0..n{let z=(0..n).fold(Complex::ZERO,|s,k|s+e.vectors[i*n+k]*e.values[k]*e.inverse[k*n+j]);worst=worst.max((z-h[i*n+j]).abs());}}
            assert!(worst<=1e-12*scale*e.condition.max(1.),"n {n} seed {seed}: residual {worst:e} of {scale:e}, condition {:.1e}",e.condition);
        }}
        // A two-level atom near its exceptional point, Ω = Γ/2: [[0, Ω/2], [Ω/2, −iΓ/2]].
        let g=38.;let h=[Complex::ZERO,Complex::new(g/4.*(1.+1e-9),0.),Complex::new(g/4.*(1.+1e-9),0.),Complex::new(0.,-g/2.)];
        let e=eigen(&h,2).unwrap();assert!(e.condition>1e3,"condition {}",e.condition);
    }
}
