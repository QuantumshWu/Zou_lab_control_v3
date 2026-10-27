//! The eigenbasis of the folded centre generator L0' = L0 + K_0^(0), which
//! propagates error sources through every mode's dressed block
//! B_q = L0' - i nu_q exactly (estimates-math.md section 3.3; the near
//! couplings between modes are left out of the propagation, the source
//! carries them).
//!
//! L0' maps Hermitian matrices to Hermitian matrices, so in the orthonormal
//! Hermitian basis (E_ii; (E_ij + E_ji)/sqrt2 and i (E_ij - E_ji)/sqrt2 for
//! i < j) it is a real d x d matrix: one real eigendecomposition, about a
//! quarter of the complex one's cost, and complex blocks take complex
//! coordinates in the same basis. A complex pair lambda, conj(lambda) with
//! the real eigenvector columns v_r, v_i has the eigenvectors v_r +- i v_i,
//! and a vector a v_r + b v_i has the coefficients (a -+ i b)/2 on them.
//!
//! Propagation: a source s_q(t) = sum_k beta_k e^{mu_k t} on mode q gives
//!   e_q(t) = sum_j E_j sum_k beta_jk (e^{mu_k t} - e^{zeta_j t}) / (mu_k - zeta_j),
//! zeta_j = lambda_j - i nu_q, read out on the populations with the phase
//! e^{i nu_q t} (a near-resonant pair takes the limit t e^{mu_k t}).
use super::*;
use super::operator::LatticeOperator;
use crate::parallel::Execution;

/// |mu_k - zeta_j| T below which a pair is treated as resonant.
const RESONANT:f64=1e-8;

pub struct BlockEigenbasis{
    n:usize,d:usize,lapack:Arc<Lapack>,
    pub lambda:Vec<C>,
    /// E_R^-1 (column-major d x d): real coefficients on the real eigenvector
    /// columns of Hermitian-basis coordinates.
    inverse:Vec<f64>,
    /// Per column: 0 a real eigenvalue, 1 and 2 the real and imaginary
    /// columns of a complex pair.
    kind:Vec<u8>,
    /// Population rows of the complex eigenvectors (column-major n x d).
    readout:Vec<C>,
    /// |E_R|_1 |E_R^-1|_1.
    pub condition:f64,
    pub milliseconds:f64,
}

/// The coordinates of a row-major n x n block in the Hermitian basis.
fn coordinates(x:&[C],n:usize)->Vec<C>{
    let mut c=Vec::with_capacity(n*n);let r=std::f64::consts::FRAC_1_SQRT_2;
    for i in 0..n{c.push(x[i*n+i]);}
    for i in 0..n{for j in i+1..n{let(a,b)=(x[i*n+j],x[j*n+i]);
        c.push(C{re:r*(a.re+b.re),im:r*(a.im+b.im)});
        // -i (a - b)/sqrt2.
        c.push(C{re:r*(a.im-b.im),im:-r*(a.re-b.re)});}}
    c
}

impl BlockEigenbasis{
    pub fn new(prepared:&LatticePrepared)->Result<Self,String>{
        let clock=Instant::now();let lapack=&prepared.lapack;
        let n=prepared.generator.n;let d=n*n;let l=prepared.centre_generator();
        // Real matrix: column k is the coordinate vector of L0' applied to
        // basis element k.
        let r=std::f64::consts::FRAC_1_SQRT_2;
        let column=|k:usize|->Vec<C>{(0..d).map(|row|l[row*d+k]).collect()};
        let mut real=vec![0_f64;d*d];
        let mut put=|k:usize,y:Vec<C>|{for(i,v)in coordinates(&y,n).into_iter().enumerate(){real[i+d*k]=v.re;}};
        for i in 0..n{put(i,column(i*n+i));}
        let mut k=n;
        for i in 0..n{for j in i+1..n{
            let(a,b)=(column(i*n+j),column(j*n+i));
            put(k,a.iter().zip(&b).map(|(x,y)|C{re:r*(x.re+y.re),im:r*(x.im+y.im)}).collect());
            put(k+1,a.iter().zip(&b).map(|(x,y)|C{re:-r*(x.im-y.im),im:r*(x.re-y.re)}).collect());
            k+=2;}}
        drop(l);
        let(re,im,vectors)=lapack.eigen(&real,d)?;drop(real);
        if vectors.iter().any(|v|!v.is_finite()){return Err("Nonfinite centre-generator eigenvectors".into());}
        let mut kind=vec![0_u8;d];let mut lambda=vec![ZERO;d];let mut j=0;
        while j<d{
            if im[j]!=0.&&j+1<d{kind[j]=1;kind[j+1]=2;lambda[j]=C{re:re[j],im:im[j]};lambda[j+1]=C{re:re[j],im:-im[j]};j+=2;}
            else{lambda[j]=C{re:re[j],im:0.};j+=1;}
        }
        let mut readout=vec![ZERO;n*d];
        for i in 0..n{for j in 0..d{let v=vectors[i+d*j];
            readout[i+n*j]=match kind[j]{0=>C{re:v,im:0.},1=>C{re:v,im:vectors[i+d*(j+1)]},_=>C{re:vectors[i+d*(j-1)],im:-v}};}}
        let(lu,pivots)=lapack.factor(&vectors,d)?;
        let mut identity=vec![0.;d*d];for i in 0..d{identity[i+d*i]=1.;}
        let inverse=lapack.solve_matrix_factored(&lu,&pivots,&identity,d)?;
        let one=|a:&[f64]|(0..d).map(|c|a[c*d..(c+1)*d].iter().map(|x|x.abs()).sum::<f64>()).fold(0.,f64::max);
        let condition=one(&vectors)*one(&inverse);
        Ok(Self{n,d,lapack:Arc::clone(lapack),lambda,inverse,kind,readout,condition,milliseconds:clock.elapsed().as_secs_f64()*1e3})
    }
    /// Coefficients on the complex eigenvectors of row-major blocks
    /// (column-major d x count): one real GEMM with E_R^-1, then the pair rule.
    fn coefficients(&self,blocks:&[&[C]])->Result<Vec<C>,String>{
        let(n,d)=(self.n,self.d);let count=blocks.len();
        let mut parts=vec![0.;d*2*count];
        for(c,block)in blocks.iter().enumerate(){for(k,v)in coordinates(block,n).into_iter().enumerate(){parts[k+d*c]=v.re;parts[k+d*(count+c)]=v.im;}}
        let mut y=vec![0.;d*2*count];self.lapack.gemm(false,false,d,2*count,d,1.,&self.inverse,d,&parts,d,0.,&mut y,d)?;
        let at=|j:usize,c:usize|C{re:y[j+d*c],im:y[j+d*(count+c)]};
        let mut z=vec![ZERO;d*count];
        for c in 0..count{let mut j=0;
            while j<d{
                if self.kind[j]==1{let(a,b)=(at(j,c),at(j+1,c));
                    // (a -+ i b)/2.
                    z[j+d*c]=C{re:0.5*(a.re+b.im),im:0.5*(a.im-b.re)};z[j+1+d*c]=C{re:0.5*(a.re-b.im),im:0.5*(a.im+b.re)};j+=2;}
                else{z[j+d*c]=at(j,c);j+=1;}
            }}
        Ok(z)
    }
    /// Population trajectories (one row of n per time) of e' = B e + s(t),
    /// e(0) = 0, with B the block-diagonal dressed generator of the mode set
    /// and s(t) = sum_c fields[c] psi_c(t), psi_c(t) = sum_k alpha[c][k]
    /// e^{mu_k t}; the fields are Hermitian-symmetric on the modes of `op`.
    /// Per mode, complex GEMMs: beta = Z alpha, the readout of
    /// beta / (mu - zeta), and the two exponential sums.
    pub fn propagate(&self,op:&LatticeOperator<'_>,fields:&[Vec<C>],alpha:&[Vec<C>],mu:&[C],times:&[f64])->Result<Vec<Vec<f64>>,String>{
        let(n,d)=(self.n,self.d);let(c_count,k_count,t_count)=(fields.len(),mu.len(),times.len());
        if c_count==0||k_count==0{return Ok(vec![vec![0.;n];t_count]);}
        let horizon=times.iter().fold(0_f64,|a,&t|a.max(t.abs())).max(1e-300);
        let cexp=|z:C,t:f64|{let m=(z.re*t).exp();C{re:m*(z.im*t).cos(),im:m*(z.im*t).sin()}};
        // e^{mu_k t} (K x T) and e^{lambda_j t} (d x T), column-major.
        let ritz:Vec<C>=times.iter().flat_map(|&t|mu.iter().map(move|&z|cexp(z,t))).collect();
        let block:Vec<C>=times.iter().flat_map(|&t|self.lambda.iter().map(move|&z|cexp(z,t))).collect();
        let mut alpha_cm=vec![ZERO;c_count*k_count];for(c,a)in alpha.iter().enumerate(){for k in 0..k_count{alpha_cm[c+c_count*k]=a[k];}}
        let gemm=self.lapack.complex_gemm();
        let computed:Vec<usize>=(0..op.modes.len()).filter(|&q|op.canonical[q]).collect();
        let parts=Execution::new()?.map(computed,|q|->Result<Vec<f64>,String>{
            let weight=if op.mirror[q]==q{1.}else{2.};let nu=op.nu[q];
            let blocks:Vec<&[C]>=fields.iter().map(|f|&f[q*d..(q+1)*d]).collect();
            let z=self.coefficients(&blocks)?;
            let mut beta=vec![ZERO;d*k_count];gemm.multiply(d,k_count,c_count,ONE,&z,d,&alpha_cm,c_count,ZERO,&mut beta,d)?;
            // G_jk = beta_jk / (mu_k - zeta_j), zeta_j = lambda_j - i nu;
            // resonant pairs take the limit t e^{mu_k t}.
            let mut gamma=vec![ZERO;d];let mut resonant=Vec::new();
            for k in 0..k_count{for j in 0..d{
                let b=beta[j+d*k];if !nonzero(b){continue;}
                let den=C{re:mu[k].re-self.lambda[j].re,im:mu[k].im-self.lambda[j].im+nu};
                if magnitude(den)*horizon<RESONANT{resonant.push((j,k,b));beta[j+d*k]=ZERO;continue;}
                let s=den.re*den.re+den.im*den.im;let v=mul(b,C{re:den.re/s,im:-den.im/s});beta[j+d*k]=v;add(&mut gamma[j],v);
            }}
            let mut m1=vec![ZERO;n*k_count];gemm.multiply(n,k_count,d,ONE,&self.readout,n,&beta,d,ZERO,&mut m1,n)?;
            let mut first=vec![ZERO;n*t_count];gemm.multiply(n,t_count,k_count,ONE,&m1,n,&ritz,k_count,ZERO,&mut first,n)?;
            // e^{i nu t} e^{zeta_j t} = e^{lambda_j t}.
            let mut weighted=self.readout.clone();for j in 0..d{for i in 0..n{weighted[i+n*j]=mul(weighted[i+n*j],gamma[j]);}}
            let mut second=vec![ZERO;n*t_count];gemm.multiply(n,t_count,d,ONE,&weighted,n,&block,d,ZERO,&mut second,n)?;
            let mut out=vec![0.;t_count*n];
            for(ti,&t)in times.iter().enumerate(){
                let phase=cexp(C{re:0.,im:nu},t);
                for i in 0..n{
                    let mut v=first[i+n*ti];
                    for &(j,k,b)in &resonant{mul_add(&mut v,mul(self.readout[i+n*j],b),scaled(ritz[k+k_count*ti],t));}
                    out[ti*n+i]=weight*(mul(phase,v).re-second[i+n*ti].re);
                }
            }
            Ok(out)
        });
        let mut rows=vec![vec![0.;n];t_count];
        for part in parts{let part=part?;for(ti,row)in rows.iter_mut().enumerate(){for(r,&v)in row.iter_mut().zip(&part[ti*n..(ti+1)*n]){*r+=v;}}}
        Ok(rows)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_block_eigenbasis_propagates_a_source_exactly(){
        // One mode (the centre), a constant source f (mu = 0): e(t) =
        // (e^{L0' t} - 1) L0'^-1 f on the trace-free part; checked against a
        // fine explicit integration of e' = L0' e + f.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;
        let basis=BlockEigenbasis::new(&p).unwrap();
        let op=LatticeOperator::new(&p,&[p.generator.zero.clone()]).unwrap();
        let mut source=vec![ZERO;d];for(i,v)in[(0,0.3),(5,-0.2),(10,-0.1),(15,0.0)]{source[i]=C{re:v,im:0.};}
        source[1]=C{re:0.05,im:0.02};source[4]=C{re:0.05,im:-0.02};
        let times=[0.,0.5,1.,1.5,2.];
        let rows=basis.propagate(&op,&[source.clone()],&[vec![ONE]],&[ZERO],&times).unwrap();
        let l=p.centre_generator();let steps=200000;let dt=2./steps as f64;let mut e=vec![ZERO;d];let mut check=Vec::new();
        for s in 0..=steps{
            if s%(steps/4)==0{check.push((0..n).map(|i|e[i*n+i].re).collect::<Vec<f64>>());}
            // Midpoint step.
            let rate=|e:&[C]|->Vec<C>{(0..d).map(|r|{let mut v=source[r];for c in 0..d{mul_add(&mut v,l[r*d+c],e[c]);}v}).collect()};
            let k1=rate(&e);let half:Vec<C>=e.iter().zip(&k1).map(|(a,b)|C{re:a.re+0.5*dt*b.re,im:a.im+0.5*dt*b.im}).collect();
            let k2=rate(&half);for(a,b)in e.iter_mut().zip(&k2){a.re+=dt*b.re;a.im+=dt*b.im;}
        }
        for(t,(row,expected))in rows.iter().zip(&check).enumerate(){
            let error=row.iter().zip(expected).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
            assert!(error<1e-6,"time {t}: {error:e}, {row:?} vs {expected:?}");
        }
    }
}
