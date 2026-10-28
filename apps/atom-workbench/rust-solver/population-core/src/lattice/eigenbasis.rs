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
//!
//! Block solves: (L0' - i nu_q - xi)^-1 on any blocks, exact for every mode
//! and shift (real or complex): the blocks' coordinates, E_R^-1 (one real
//! GEMM over every block's real and imaginary parts), the pair rule, the
//! division by lambda_j - i nu_q - xi, the pair rule back, E_R (one GEMM),
//! about 4 d^2 real multiply-adds per block.
//!
//! Dressed coordinates: E_R^-1 applied to a block's Hermitian-basis
//! coordinates (`dressed`), where the block-diagonal part of the lattice is
//! diagonal (2 x 2 rotations for complex pairs) with Re <= 0: the inner
//! product of the reduced model. The adjoint solves give the norm of a
//! block's resolvent (`resolvent_norm`, the ring condition).
//!
//! The decomposition runs per connected block of the real matrix (the
//! elements its nonzero entries connect: exact zeros exploited for speed
//! only, the result is the full eigendecomposition): E_R and E_R^-1 are
//! block diagonal up to the permutation.
//!
//! The basis is request-level (`LatticePrepared::eigenbasis`): computed once,
//! shared by every pipeline that needs it.
use super::*;
use super::operator::LatticeOperator;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// |mu_k - zeta_j| T below which a pair is treated as resonant.
const RESONANT:f64=1e-8;
/// Power iterations of `resolvent_norms`.
const RESOLVENT_ITERATIONS:usize=20;
/// Fewest blocks per task of the dressed coordinates.
const DRESSED_BLOCKS:usize=16;

pub struct BlockEigenbasis{
    n:usize,d:usize,lapack:Arc<Lapack>,
    pub lambda:Vec<C>,
    /// E_R and E_R^-1 (column-major d x d): the real eigenvector columns of
    /// Hermitian-basis coordinates and their inverse.
    vectors:Vec<f64>,inverse:Vec<f64>,
    /// Per column: 0 a real eigenvalue, 1 and 2 the real and imaginary
    /// columns of a complex pair.
    kind:Vec<u8>,
    /// Population rows of the complex eigenvectors (column-major n x d).
    pub readout:Vec<C>,
    /// |E_R|_1 |E_R^-1|_1.
    pub condition:f64,
    /// Connected blocks and the largest one.
    pub blocks:usize,pub largest_block:usize,
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

/// The row-major n x n block of Hermitian-basis coordinates (the inverse of
/// `coordinates`): a = x_ij = (s + i t)/sqrt2, b = x_ji = (s - i t)/sqrt2 for
/// the symmetric and antisymmetric coordinates s, t of i < j.
fn from_coordinates(c:&[C],n:usize)->Vec<C>{
    let mut x=vec![ZERO;n*n];let r=std::f64::consts::FRAC_1_SQRT_2;
    for i in 0..n{x[i*n+i]=c[i];}
    let mut k=n;
    for i in 0..n{for j in i+1..n{let(s,t)=(c[k],c[k+1]);
        x[i*n+j]=C{re:r*(s.re-t.im),im:r*(s.im+t.re)};x[j*n+i]=C{re:r*(s.re+t.im),im:r*(s.im-t.re)};k+=2;}}
    x
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
        // Connected blocks: union-find over the nonzero entries, each block's
        // elements ascending, the blocks by their first element.
        fn root(parent:&mut[usize],mut i:usize)->usize{while parent[i]!=i{parent[i]=parent[parent[i]];i=parent[i];}i}
        let mut parent:Vec<usize>=(0..d).collect();
        for k in 0..d{for i in 0..d{if i!=k&&real[i+d*k]!=0.{let(a,b)=(root(&mut parent,i),root(&mut parent,k));if a!=b{parent[a.max(b)]=a.min(b);}}}}
        let mut groups:BTreeMap<usize,Vec<usize>>=BTreeMap::new();
        for i in 0..d{let r=root(&mut parent,i);groups.entry(r).or_default().push(i);}
        let(mut re,mut im,mut vectors,mut inverse)=(vec![0.;d],vec![0.;d],vec![0.;d*d],vec![0.;d*d]);
        let mut column=0;let largest_block=groups.values().map(|g|g.len()).max().unwrap_or(0);
        for members in groups.values(){
            let b=members.len();let mut sub=vec![0.;b*b];
            for(c,&k)in members.iter().enumerate(){for(i,&row)in members.iter().enumerate(){sub[i+b*c]=real[row+d*k];}}
            let(br,bi,bv)=lapack.eigen(&sub,b)?;
            let(lu,pivots)=lapack.factor(&bv,b)?;let mut identity=vec![0.;b*b];for i in 0..b{identity[i+b*i]=1.;}
            let bw=lapack.solve_matrix_factored(&lu,&pivots,&identity,b)?;
            for c in 0..b{re[column+c]=br[c];im[column+c]=bi[c];
                for(i,&row)in members.iter().enumerate(){vectors[row+d*(column+c)]=bv[i+b*c];inverse[(column+c)+d*row]=bw[c+b*i];}}
            column+=b;
        }
        drop(real);
        if vectors.iter().chain(&inverse).any(|v|!v.is_finite()){return Err("Nonfinite centre-generator eigenvectors".into());}
        let mut kind=vec![0_u8;d];let mut lambda=vec![ZERO;d];let mut j=0;
        while j<d{
            if im[j]!=0.&&j+1<d{kind[j]=1;kind[j+1]=2;lambda[j]=C{re:re[j],im:im[j]};lambda[j+1]=C{re:re[j],im:-im[j]};j+=2;}
            else{lambda[j]=C{re:re[j],im:0.};j+=1;}
        }
        let mut readout=vec![ZERO;n*d];
        for i in 0..n{for j in 0..d{let v=vectors[i+d*j];
            readout[i+n*j]=match kind[j]{0=>C{re:v,im:0.},1=>C{re:v,im:vectors[i+d*(j+1)]},_=>C{re:vectors[i+d*(j-1)],im:-v}};}}
        let one=|a:&[f64]|(0..d).map(|c|a[c*d..(c+1)*d].iter().map(|x|x.abs()).sum::<f64>()).fold(0.,f64::max);
        let condition=one(&vectors)*one(&inverse);
        Ok(Self{n,d,lapack:Arc::clone(lapack),lambda,vectors,inverse,kind,readout,condition,blocks:groups.len(),largest_block,milliseconds:clock.elapsed().as_secs_f64()*1e3})
    }
    /// x_k <- (L0' - i nu_k - shift)^-1 x_k for row-major blocks x_k (exact;
    /// see the module documentation), spread over the workers.
    pub fn solve_blocks(&self,nu:&[f64],shift:C,blocks:&mut[C])->Result<(),String>{self.solve_with(nu,shift,blocks,false)}
    /// The adjoint solve x_k <- (L0' - i nu_k - shift)^-H x_k (Frobenius
    /// inner product): E_R^-T, the conjugated denominators, E_R^T.
    pub fn solve_blocks_adjoint(&self,nu:&[f64],shift:C,blocks:&mut[C])->Result<(),String>{self.solve_with(nu,shift,blocks,true)}
    fn solve_with(&self,nu:&[f64],shift:C,blocks:&mut[C],adjoint:bool)->Result<(),String>{
        let(n,d)=(self.n,self.d);if blocks.len()!=nu.len()*d{return Err("Eigenbasis block solve dimensions differ".into());}
        let work=4*d*d*nu.len()/2;
        // Forward: E_R^-1 then E_R; adjoint: E_R^T then E_R^-T.
        let(first_matrix,second_matrix)=if adjoint{(&self.vectors,&self.inverse)}else{(&self.inverse,&self.vectors)};
        Execution::new()?.for_mode_chunks(blocks,d,work,MINIMUM_WORK_PER_TASK,&mut Vec::<()>::new(),|first,chunk,_|{
            let count=chunk.len()/d;
            // Real and imaginary parts of the coordinates, column-major d x 2 count.
            let mut parts=vec![0.;d*2*count];
            for(c,block)in chunk.chunks_exact(d).enumerate(){for(k,v)in coordinates(block,n).into_iter().enumerate(){parts[k+d*c]=v.re;parts[k+d*(count+c)]=v.im;}}
            let mut y=vec![0.;d*2*count];self.lapack.gemm(adjoint,false,d,2*count,d,1.,first_matrix,d,&parts,d,0.,&mut y,d)?;
            for c in 0..count{
                let at=|j:usize,y:&[f64]|C{re:y[j+d*c],im:y[j+d*(count+c)]};let shift_q=C{re:shift.re,im:shift.im+nu[first+c]};
                let divide=|z:C,l:C|{let mut den=C{re:l.re-shift_q.re,im:l.im-shift_q.im};if adjoint{den.im=-den.im;}let s=den.re*den.re+den.im*den.im;mul(z,C{re:den.re/s,im:-den.im/s})};
                let mut j=0;
                while j<d{
                    if self.kind[j]==1{
                        // a v_r + b v_i = z+ (v_r + i v_i) + z- (v_r - i v_i), z+- = (a -+ i b)/2.
                        let(a,b)=(at(j,&y),at(j+1,&y));
                        let plus=divide(C{re:0.5*(a.re+b.im),im:0.5*(a.im-b.re)},self.lambda[j]);let minus=divide(C{re:0.5*(a.re-b.im),im:0.5*(a.im+b.re)},self.lambda[j+1]);
                        // Back: a' = z+ + z-, b' = i (z+ - z-).
                        let(a2,b2)=(C{re:plus.re+minus.re,im:plus.im+minus.im},C{re:-(plus.im-minus.im),im:plus.re-minus.re});
                        y[j+d*c]=a2.re;y[j+d*(count+c)]=a2.im;y[j+1+d*c]=b2.re;y[j+1+d*(count+c)]=b2.im;j+=2;}
                    else{let z=divide(at(j,&y),self.lambda[j]);y[j+d*c]=z.re;y[j+d*(count+c)]=z.im;j+=1;}
                }
            }
            self.lapack.gemm(adjoint,false,d,2*count,d,1.,second_matrix,d,&y,d,0.,&mut parts,d)?;
            for(c,block)in chunk.chunks_exact_mut(d).enumerate(){
                let coordinate:Vec<C>=(0..d).map(|k|C{re:parts[k+d*c],im:parts[k+d*(count+c)]}).collect();block.copy_from_slice(&from_coordinates(&coordinate,n));}
            Ok(())
        })
    }
    /// ||(L0' - i nu)^-1||_2 for every nu by power iteration on the normal
    /// form, all frequencies in one batch (a few percent: enough for the ring
    /// condition's threshold).
    pub fn resolvent_norms(&self,nu:&[f64])->Result<Vec<f64>,String>{
        let d=self.d;let mut x:Vec<C>=(0..d*nu.len()).map(|k|{let k=k%d;C{re:((k*7+3)%29)as f64/29.-0.5,im:((k*11+5)%31)as f64/31.-0.5}}).collect();
        let mut value=vec![0.;nu.len()];
        let norm=|b:&[C]|b.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
        for _ in 0..RESOLVENT_ITERATIONS{
            for b in x.chunks_exact_mut(d){let size=norm(b);for z in b.iter_mut(){*z=scaled(*z,1./size);}}
            self.solve_blocks(nu,ZERO,&mut x)?;self.solve_blocks_adjoint(nu,ZERO,&mut x)?;
            for(v,b)in value.iter_mut().zip(x.chunks_exact(d)){*v=norm(b);}
        }
        Ok(value.iter().map(|v|v.sqrt()).collect())
    }
    /// Dressed coordinates of row-major blocks: E_R^-1 applied to each
    /// block's Hermitian-basis coordinates (complex entries; the blocks of a
    /// Hermitian-symmetric field's mirror modes have the conjugate ones).
    /// One real GEMM over every block, spread over the workers.
    pub fn dressed(&self,blocks:&[C])->Result<Vec<C>,String>{
        let(n,d)=(self.n,self.d);let mut out=blocks.to_vec();let count=blocks.len()/d;
        // At least DRESSED_BLOCKS blocks per task: the GEMM streams E_R^-1
        // once per task.
        let execution=Execution::new()?;let tasks=execution.threads().min(count.div_ceil(DRESSED_BLOCKS)).max(1);
        let work=2*d*d*count;
        execution.for_mode_chunks(&mut out,d,work,work/tasks,&mut Vec::<()>::new(),|_,chunk,_|{
            let count=chunk.len()/d;let mut parts=vec![0.;d*2*count];
            for(c,block)in chunk.chunks_exact(d).enumerate(){for(k,v)in coordinates(block,n).into_iter().enumerate(){parts[k+d*c]=v.re;parts[k+d*(count+c)]=v.im;}}
            let mut y=vec![0.;d*2*count];self.lapack.gemm(false,false,d,2*count,d,1.,&self.inverse,d,&parts,d,0.,&mut y,d)?;
            for(c,block)in chunk.chunks_exact_mut(d).enumerate(){for(k,v)in block.iter_mut().enumerate(){*v=C{re:y[k+d*c],im:y[k+d*(count+c)]};}}
            Ok(())
        })?;
        Ok(out)
    }
    /// Coefficients on the complex eigenvectors from dressed coordinates
    /// (the pair rule).
    pub fn complex_coefficients(&self,y:&[C])->Vec<C>{
        let d=self.d;let mut z=vec![ZERO;d];let mut j=0;
        while j<d{
            if self.kind[j]==1{let(a,b)=(y[j],y[j+1]);
                z[j]=C{re:0.5*(a.re+b.im),im:0.5*(a.im-b.re)};z[j+1]=C{re:0.5*(a.re-b.im),im:0.5*(a.im+b.re)};j+=2;}
            else{z[j]=y[j];j+=1;}
        }
        z
    }
    /// The population weight of each dressed centre mode in a block:
    /// |coefficient| times the norm of the mode's population rows.
    pub fn mode_weights(&self,block:&[C])->Result<Vec<f64>,String>{
        let(n,d)=(self.n,self.d);let z=self.coefficients(&[block])?;
        Ok((0..d).map(|j|magnitude(z[j])*(0..n).map(|i|{let v=self.readout[i+n*j];v.re*v.re+v.im*v.im}).sum::<f64>().sqrt()).collect())
    }
    /// Eigenvalues of modulus below `floor`: the conserved quantities of the
    /// folded centre block.
    pub fn kernel_count(&self,floor:f64)->usize{self.lambda.iter().filter(|l|magnitude(**l)<floor).count()}
    /// The block of the eigenvalue of least modulus (the dressed centre's
    /// steady state when it is unique), trace one.
    pub fn kernel_block(&self)->Result<Vec<C>,String>{
        let(n,d)=(self.n,self.d);
        let j=(0..d).min_by(|&a,&b|magnitude(self.lambda[a]).total_cmp(&magnitude(self.lambda[b]))).ok_or("Empty eigenbasis")?;
        if self.kind[j]!=0{return Err("The dressed centre's slowest mode is not real".into());}
        let coordinate:Vec<C>=(0..d).map(|k|C{re:self.vectors[k+d*j],im:0.}).collect();
        let block=from_coordinates(&coordinate,n);let trace:f64=(0..n).map(|i|block[i*n+i].re).sum();
        if !(trace.abs()>0.){return Err("The dressed centre's slowest mode has no trace".into());}
        Ok(block.iter().map(|z|scaled(*z,1./trace)).collect())
    }
    pub fn dimension(&self)->usize{self.d}
    /// The conjugate partner of eigen-coordinate j: J e_j = e_partner(j) for
    /// the Hermitian flip J (x -> x^dagger), whose coordinates are the
    /// complex conjugates; a real eigenvalue's vector is its own partner.
    pub fn partner(&self,j:usize)->usize{match self.kind[j]{1=>j+1,2=>j-1,_=>j}}
    /// Population rows of the complex eigenvectors (column-major n x d).
    pub fn population_rows(&self)->&[C]{&self.readout}
    /// The complex eigenvectors as row-major n x n blocks, d consecutive
    /// blocks: v_r + i v_i for the first of a pair, v_r - i v_i for the
    /// second.
    pub fn eigenvector_blocks(&self)->Vec<C>{
        let(n,d)=(self.n,self.d);let mut out=Vec::with_capacity(d*d);
        for j in 0..d{
            let column=|k:usize|&self.vectors[d*k..d*(k+1)];
            let coordinate:Vec<C>=match self.kind[j]{
                0=>column(j).iter().map(|&v|C{re:v,im:0.}).collect(),
                1=>column(j).iter().zip(column(j+1)).map(|(&a,&b)|C{re:a,im:b}).collect(),
                _=>column(j-1).iter().zip(column(j)).map(|(&a,&b)|C{re:a,im:-b}).collect()};
            out.extend(from_coordinates(&coordinate,n));
        }
        out
    }
    /// Coefficients on the complex eigenvectors of row-major blocks
    /// (column-major d x count): one real GEMM with E_R^-1, then the pair rule.
    pub fn coefficients(&self,blocks:&[&[C]])->Result<Vec<C>,String>{
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
    fn lattice_block_eigenbasis_solves_shifted_blocks_exactly(){
        // (L0' - i nu - xi)^-1 on general complex blocks against a dense LU,
        // for real and complex shifts and several modes.
        let f=fixture();let p=prepared(&f);let d=16;let basis=p.eigenbasis().unwrap();let l=p.centre_generator();
        let nu=[0.,3.,-6.];
        for shift in[C{re:20.,im:0.},C{re:0.5,im:13.},C{re:1e-3,im:0.}]{
            let b:Vec<C>=(0..3*d).map(|k|C{re:((k*7+3)%29)as f64/29.-0.5,im:((k*11+5)%31)as f64/31.-0.5}).collect();
            let mut x=b.clone();basis.solve_blocks(&nu,shift,&mut x).unwrap();
            for(q,&w)in nu.iter().enumerate(){let(xq,bq)=(&x[q*d..(q+1)*d],&b[q*d..(q+1)*d]);
                let mut r=bq.to_vec();for i in 0..d{for k in 0..d{let v=mul(l[i*d+k],xq[k]);r[i].re-=v.re;r[i].im-=v.im;}let v=mul(C{re:-shift.re,im:-shift.im-w},xq[i]);r[i].re-=v.re;r[i].im-=v.im;}
                let error=r.iter().map(|z|magnitude(*z)).fold(0.,f64::max);assert!(error<1e-11,"{shift:?} nu {w}: residual {error:e}");}
            // The adjoint: <y, D^-1 x> = <D^-H y, x>.
            let y:Vec<C>=(0..3*d).map(|k|C{re:((k*5+1)%23)as f64/23.-0.5,im:((k*3+2)%19)as f64/19.-0.5}).collect();
            let mut z=y.clone();basis.solve_blocks_adjoint(&nu,shift,&mut z).unwrap();
            let dot=|a:&[C],b:&[C]|a.iter().zip(b).fold(ZERO,|s,(u,v)|{let mut s=s;mul_add(&mut s,conj(*u),*v);s});
            let(left,right)=(dot(&y,&x),dot(&z,&b));
            assert!(magnitude(C{re:left.re-right.re,im:left.im-right.im})<1e-11*magnitude(left).max(1.),"{shift:?}: adjoint {left:?} {right:?}");
        }
        // The resolvent norm against the dense inverse's largest singular value (power iteration on a dense LU).
        let nu0=3.;let mut m=l.clone();for i in 0..d{m[i*d+i].im-=nu0;}
        let mut col=vec![ZERO;d*d];for r in 0..d{for c in 0..d{col[r+d*c]=m[r*d+c];}}
        let inv=super::super::floquet::inverse(&p.lapack,&col,d).unwrap();
        let mut v=vec![ONE;d];let mut sigma=0.;
        for _ in 0..200{let w:Vec<C>=(0..d).map(|r|(0..d).fold(ZERO,|s,c|{let mut s=s;mul_add(&mut s,inv[r+d*c],v[c]);s})).collect();
            let u:Vec<C>=(0..d).map(|c|(0..d).fold(ZERO,|s,r|{let mut s=s;mul_add(&mut s,conj(inv[r+d*c]),w[r]);s})).collect();
            let size=u.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();sigma=size.sqrt();v=u.iter().map(|z|scaled(*z,1./size)).collect();}
        let norm=basis.resolvent_norms(&[nu0]).unwrap()[0];
        assert!((norm/sigma-1.).abs()<0.05,"resolvent norm {norm:e} against {sigma:e}");
    }
    #[test]
    fn lattice_block_eigenbasis_propagates_a_source_exactly(){
        // One mode (the centre), a constant source f (mu = 0): e(t) =
        // (e^{L0' t} - 1) L0'^-1 f on the trace-free part; checked against a
        // fine explicit integration of e' = L0' e + f.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;
        let basis=p.eigenbasis().unwrap();
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
