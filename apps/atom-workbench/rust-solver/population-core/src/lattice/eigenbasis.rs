//! The eigenbasis of the folded centre generator L0' = L0 + K_0^(0): the
//! dressed blocks B_q = L0' - i nu_q of every mode and their solves, the
//! eigen-coordinates of the Schrieffer-Wolff frame (`frame`) and of the
//! error estimate's propagation (`estimator`), and the dressed coordinates
//! of the pole ordering's error transform (`reduced`).
//!
//! L0' maps Hermitian matrices to Hermitian matrices, so in the orthonormal
//! Hermitian basis (E_ii; (E_ij + E_ji)/sqrt2 and i (E_ij - E_ji)/sqrt2 for
//! i < j) it is a real d x d matrix: real eigensystems, about a quarter of
//! the complex ones' cost, and complex blocks take complex coordinates in
//! the same basis. A complex pair lambda, conj(lambda) with the real
//! eigenvector columns v_r, v_i has the eigenvectors v_r +- i v_i, and a
//! vector a v_r + b v_i has the coefficients (a -+ i b)/2 on them.
//!
//! The eigensystem by clusters (`clustered_eigensystem`, shared with the
//! estimator's second-order blocks): L0' X = NL' X + X NR' + J X, the
//! separable part diagonal on X_ab = p_a p_b^H (the Floquet block frame,
//! NL' = P diag(lambda) P^-1, NR' = NL'^H + c: the value lambda_a + mu_b),
//! J the rest (the recycling, the fold's scattering). In the coordinates of
//! the X_ab most couplings are far below their gaps (coherences of states
//! at different frequencies), so the strongly coupled ones (the
//! populations, the coherences of near-degenerate states) form clusters,
//! each diagonalized exactly, the couplings between them at first
//! (vectors) and second (values) order, with the mixing and the values'
//! error over the horizon bounded (merged clusters otherwise). The
//! pair (X_ab, X_ba) is the flip pair of the real form. The centre's
//! eigenbasis is its exact limit (`exact`) in the Hermitian basis: the
//! connected blocks of the real form, each whole (one block of 1024 on the
//! tweezer cases); the clusters serve the estimator's second-order blocks
//! (`estimator::blocks`). The
//! sector grading of the frame does not decouple the centre: the separable
//! part's eigenvectors mix ground and excited states by 6% (three and five
//! lasers + tweezer) to 36% (four lasers, no trap) (near drives at charge 0),
//! and couplings between the sectors reach 0.04-1 of their gaps.
//!
//! Block solves: (L0' - i nu_q - xi)^-1 on any blocks, for every mode and
//! shift (real or complex), to the eigensystem's accuracy (the clusters'
//! mixing squared; exact with one cluster): the blocks' coordinates, E_R^-1
//! (one real GEMM over every block's real and imaginary parts), the pair
//! rule, the division by lambda_j - i nu_q - xi, the pair rule back, E_R (one
//! GEMM), about 4 d^2 real multiply-adds per block.
//!
//! Dressed coordinates: E_R^-1 applied to a block's Hermitian-basis
//! coordinates (`dressed`), where the block-diagonal part of the lattice is
//! diagonal (2 x 2 rotations for complex pairs) with Re <= 0. The adjoint
//! solves give the norm of a block's resolvent (`resolvent_norms`, the ring
//! condition).
//!
//! The basis is request-level (`LatticePrepared::eigenbasis`: the
//! screening, the frame, the estimator and the pole ordering), computed
//! once and shared.
use super::*;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK,REFERENCE_WORKERS};

/// Power iterations of `resolvent_norms`.
const RESOLVENT_ITERATIONS:usize=20;
/// Columns per task of a large eigenvector matrix's inverse.
const INVERSE_PANEL:usize=64;
/// Fewest blocks per task of the dressed coordinates.
const DRESSED_BLOCKS:usize=16;
/// The complex coordinates z+- = (a -+ i b)/2 of a conjugate pair from its
/// real coordinates (a, b): a v_r + b v_i = z+ (v_r + i v_i) + z- (v_r - i v_i).
pub(crate) fn pair_coordinates(a:C,b:C)->(C,C){(C{re:0.5*(a.re+b.im),im:0.5*(a.im-b.re)},C{re:0.5*(a.re-b.im),im:0.5*(a.im+b.re)})}
/// LAPACK's real eigenvalues (re, im), each conjugate pair adjacent with its
/// positive imaginary part first: per value 0 (real), 1 (a pair's first) or
/// 2 (its second), and the complex values.
pub(crate) fn conjugate_pairs(re:&[f64],im:&[f64])->(Vec<u8>,Vec<C>){
    let d=re.len();let mut kind=vec![0_u8;d];let mut values=vec![ZERO;d];let mut j=0;
    while j<d{
        if im[j]!=0.&&j+1<d{kind[j]=1;kind[j+1]=2;values[j]=C{re:re[j],im:im[j]};values[j+1]=C{re:re[j],im:-im[j]};j+=2;}
        else{values[j]=C{re:re[j],im:0.};j+=1;}
    }
    (kind,values)
}

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
    /// The eigensystem's clusters, the largest one and the merge passes.
    pub clusters:usize,pub largest_cluster:usize,pub passes:usize,
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
    /// Exact: the clustered method's exact limit (every coupling joins its
    /// coordinates) in the Hermitian basis, where the clusters are the
    /// connected blocks of the real form (exact zeros, for speed only), each
    /// diagonalized whole in real arithmetic (`real_inverse`).
    pub fn exact(prepared:&LatticePrepared)->Result<Self,String>{
        let clock=Instant::now();let lapack=&prepared.lapack;
        let n=prepared.generator.n;let d=n*n;let l=prepared.centre_generator();
        // The real form: column k is the coordinate vector of L0' applied to
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
        let mut column=0;let largest=groups.values().map(|g|g.len()).max().unwrap_or(0);
        for members in groups.values(){
            let b=members.len();let mut sub=vec![0.;b*b];
            for(c,&k)in members.iter().enumerate(){for(i,&row)in members.iter().enumerate(){sub[i+b*c]=real[row+d*k];}}
            let(br,bi,bv)=lapack.eigen(&sub,b)?;let bw=real_inverse(lapack,&bv,b)?;
            for c in 0..b{re[column+c]=br[c];im[column+c]=bi[c];
                for(i,&row)in members.iter().enumerate(){vectors[row+d*(column+c)]=bv[i+b*c];inverse[(column+c)+d*row]=bw[c+b*i];}}
            column+=b;
        }
        drop(real);
        let(kind,lambda)=conjugate_pairs(&re,&im);
        Self::finish(n,d,lapack,lambda,vectors,inverse,kind,[groups.len(),largest,1],clock)
    }
    /// The readout and the condition of E_R, E_R^-1 and the values.
    #[allow(clippy::too_many_arguments)]
    fn finish(n:usize,d:usize,lapack:&Arc<Lapack>,lambda:Vec<C>,vectors:Vec<f64>,inverse:Vec<f64>,kind:Vec<u8>,clusters:[usize;3],clock:Instant)->Result<Self,String>{
        if vectors.iter().chain(&inverse).any(|v|!v.is_finite()){return Err("Nonfinite centre-generator eigenvectors".into());}
        let mut readout=vec![ZERO;n*d];
        for i in 0..n{for j in 0..d{let v=vectors[i+d*j];
            readout[i+n*j]=match kind[j]{0=>C{re:v,im:0.},1=>C{re:v,im:vectors[i+d*(j+1)]},_=>C{re:vectors[i+d*(j-1)],im:-v}};}}
        let one=|a:&[f64]|(0..d).map(|c|a[c*d..(c+1)*d].iter().map(|x|x.abs()).sum::<f64>()).fold(0.,f64::max);
        let condition=one(&vectors)*one(&inverse);
        Ok(Self{n,d,lapack:Arc::clone(lapack),lambda,vectors,inverse,kind,readout,condition,clusters:clusters[0],largest_cluster:clusters[1],passes:clusters[2],
            milliseconds:clock.elapsed().as_secs_f64()*1e3})
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
                        let(plus,minus)=pair_coordinates(a,b);let(plus,minus)=(divide(plus,self.lambda[j]),divide(minus,self.lambda[j+1]));
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
        // At least DRESSED_BLOCKS blocks per task (the GEMM streams E_R^-1
        // once per task), at most one task per reference worker: independent
        // of the thread count.
        let execution=Execution::new()?;let tasks=REFERENCE_WORKERS.min(count.div_ceil(DRESSED_BLOCKS)).max(1);
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
            if self.kind[j]==1{(z[j],z[j+1])=pair_coordinates(y[j],y[j+1]);j+=2;}
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
                if self.kind[j]==1{(z[j+d*c],z[j+1+d*c])=pair_coordinates(at(j,c),at(j+1,c));j+=2;}
                else{z[j+d*c]=at(j,c);j+=1;}
            }}
        Ok(z)
    }
    /// The adjoint of `coefficients`: row-major blocks (count of them, one
    /// after the other) from complex eigen-coordinates w (column-major
    /// d x count): the pair rule's adjoint, E_R^-T (one real GEMM), the
    /// Hermitian-basis coordinates' inverse (they are orthonormal).
    pub fn coefficients_adjoint(&self,w:&[C],count:usize)->Result<Vec<C>,String>{
        let(n,d)=(self.n,self.d);
        let mut parts=vec![0.;d*2*count];
        for c in 0..count{let mut j=0;let col=&w[c*d..(c+1)*d];
            while j<d{
                if self.kind[j]==1{let(u,v)=(col[j],col[j+1]);
                    // ((u + v)/2, i (u - v)/2).
                    let(a,b)=(C{re:0.5*(u.re+v.re),im:0.5*(u.im+v.im)},C{re:-0.5*(u.im-v.im),im:0.5*(u.re-v.re)});
                    parts[j+d*c]=a.re;parts[j+d*(count+c)]=a.im;parts[j+1+d*c]=b.re;parts[j+1+d*(count+c)]=b.im;j+=2;}
                else{parts[j+d*c]=col[j].re;parts[j+d*(count+c)]=col[j].im;j+=1;}
            }}
        let mut y=vec![0.;d*2*count];self.lapack.gemm(true,false,d,2*count,d,1.,&self.inverse,d,&parts,d,0.,&mut y,d)?;
        let mut out=Vec::with_capacity(d*count);
        for c in 0..count{let coordinate:Vec<C>=(0..d).map(|k|C{re:y[k+d*c],im:y[k+d*(count+c)]}).collect();out.extend(from_coordinates(&coordinate,n));}
        Ok(out)
    }
}


/// Couplings below CLUSTER_ENTRY times their gap, |m_ij| < CLUSTER_ENTRY
/// |base_i - base_j|, leave coordinates i, j in different clusters; the
/// clusters are diagonalized exactly and the couplings between them at first
/// order (vectors) and second order (values). Measured on the estimator's
/// second-order blocks against the whole eigensystem on three, four and five
/// lasers (20 us, 10 ms): the estimates within 0.35%, the largest cluster
/// 156-362 of 1024 (1e-3: 330-552), the block built in 0.2-0.33 s instead
/// of 0.55-1.2 s.
const CLUSTER_ENTRY:f64=1e-2;
/// Two clusters merge when a first-order mixing between them exceeds this,
/// or when the third-order error of a value, over the time its mode lives
/// (the horizon, or 1 / |Re mu| when shorter), exceeds CLUSTER_PHASE.
/// Measured with CLUSTER_ENTRY (above): the estimates within 0.35% of the
/// whole eigensystem's; the vectors' error is about the mixing squared.
const CLUSTER_MIXING:f64=0.05;
const CLUSTER_PHASE:f64=1e-2;
/// Merge passes before the whole matrix is diagonalized at once.
const CLUSTER_PASSES:usize=4;
/// Clusters of at least CLUSTER_SPLIT coordinates run their products over
/// the whole dimension in panels of CLUSTER_PANEL (on the workers).
const CLUSTER_SPLIT:usize=32;
const CLUSTER_PANEL:usize=128;
/// Coordinates per tile of the second-order check (K'_ab and K'_ba).
const CHECK_TILE:usize=64;

/// The error control of `clustered_eigensystem`: couplings below `entry`
/// times their gap leave two coordinates in different clusters; two
/// clusters merge when a first-order mixing between them exceeds `mixing`,
/// or when the third-order error of a value, over the time its mode lives
/// (`horizon`, or 1 / |Re mu| when shorter), exceeds `phase`.
#[derive(Clone,Copy,Debug)]
pub struct ClusterControl{pub entry:f64,pub mixing:f64,pub phase:f64,pub horizon:f64}
impl ClusterControl{
    /// The estimator's second-order blocks over a request's horizon.
    pub fn blocks(horizon:f64)->Self{Self{entry:CLUSTER_ENTRY,mixing:CLUSTER_MIXING,phase:CLUSTER_PHASE,horizon}}
}

/// An eigensystem m V = V diag(values) (`clustered_eigensystem`): V and V^-1
/// column-major; `pair[l]`: eigen-indices l, l + 1 are a conjugate pair
/// (values[l + 1] = conj values[l], V[:, l + 1] the flip of V[:, l]). The
/// clusters, the largest one and the merge passes.
pub struct Clustered{pub values:Vec<C>,pub vectors:Vec<C>,pub inverse:Vec<C>,pub pair:Vec<bool>,pub clusters:usize,pub largest:usize,pub passes:usize}

/// The eigensystem of a flip-invariant matrix m (column-major d x d, complex
/// coordinates where `first[j]` marks a conjugate pair (j, j + 1), the rest
/// real) by clusters of strongly coupled coordinates (closed under the
/// pairing; `base`: the values the gaps are measured with): each cluster
/// exactly (`real_eigensystem`), the couplings K' between clusters, in the
/// clusters' eigencoordinates, by first-order mixing T_ba = K'_ba / (mu_a -
/// mu_b) and the second-order values mu_a + sum_b K'_ab T_ba. V = V_c (1 + T),
/// V^-1 = (1 - T) V_c^-1 (error O(T^2)). A cluster pair whose mixing or
/// third-order value error over `horizon` is above its bound is merged and
/// the whole is redone; one cluster is the exact eigensystem.
pub fn clustered_eigensystem(lapack:&Lapack,m:&[C],base:&[C],first:&[bool],control:ClusterControl)->Result<Clustered,String>{
    let ClusterControl{entry,mixing:mixing_bound,phase,horizon}=control;
    let d=base.len();
    // Atoms: a pair (j, j + 1) or a single coordinate.
    let mut atom=vec![0;d];let mut atoms=0;let mut j=0;
    while j<d{atom[j]=atoms;if first[j]{atom[j+1]=atoms;j+=2;}else{j+=1;}atoms+=1;}
    let mut parent:Vec<usize>=(0..atoms).collect();
    fn root(p:&mut [usize],mut i:usize)->usize{while p[i]!=i{p[i]=p[p[i]];i=p[i];}i}
    fn join(p:&mut [usize],a:usize,b:usize){let(x,y)=(root(p,a),root(p,b));if x!=y{p[x]=y;}}
    for j in 0..d{for i in 0..d{if atom[i]==atom[j]{continue;}let v=magnitude(m[i+d*j]);if v==0.{continue;}
        let gap=magnitude(C{re:base[i].re-base[j].re,im:base[i].im-base[j].im});
        if v>=entry*gap{join(&mut parent,atom[i],atom[j]);}}}
    let execution=Execution::new()?;
    let mut kept:BTreeMap<Vec<usize>,RealSystem>=BTreeMap::new();let mut passes=0;
    for pass in 0..CLUSTER_PASSES{passes=pass+1;
        // Clusters (sorted coordinates) and their exact eigensystems.
        let mut members:BTreeMap<usize,Vec<usize>>=BTreeMap::new();
        for j in 0..d{let r=root(&mut parent,atom[j]);members.entry(r).or_default().push(j);}
        let clusters:Vec<Vec<usize>>=members.into_values().collect();
        let largest=clusters.iter().map(|c|c.len()).max().unwrap_or(0);
        if clusters.len()==1{break;}
        // The clusters a merge left unchanged keep their eigensystems.
        let missing:Vec<Vec<usize>>=clusters.iter().filter(|c|!kept.contains_key(*c)).cloned().collect();
        let built=execution.map(missing.clone(),|c|->Result<RealSystem,String>{
            let n=c.len();let local:Vec<C>=(0..n*n).map(|k|m[c[k%n]+d*c[k/n]]).collect();
            let pairs:Vec<bool>=c.iter().map(|&j|first[j]).collect();
            real_eigensystem(lapack,&pairs,&local,n)});
        for(c,system)in missing.into_iter().zip(built){kept.insert(c,system?);}
        let systems:Vec<RealSystem>=clusters.iter().map(|c|kept.remove(c).unwrap()).collect();
        // Eigen-index a of cluster k: offset[k] + l.
        let mut offset=Vec::with_capacity(clusters.len());let mut owner=vec![0;d];let mut total=0;
        for(k,c)in clusters.iter().enumerate(){offset.push(total);for a in total..total+c.len(){owner[a]=k;}total+=c.len();}
        let values:Vec<C>=systems.iter().flat_map(|s|s.values.iter().copied()).collect();
        // Clusters no entry couples (exact zeros, for speed only): the exact
        // eigensystem, block diagonal.
        let mut cluster_of=vec![0;d];for(k,c)in clusters.iter().enumerate(){for &j in c{cluster_of[j]=k;}}
        if (0..d).all(|j|(0..d).all(|i|cluster_of[i]==cluster_of[j]||!nonzero(m[i+d*j]))){
            let(mut vectors,mut inverse)=(vec![ZERO;d*d],vec![ZERO;d*d]);
            for(k,c)in clusters.iter().enumerate(){let(n,system)=(c.len(),&systems[k]);
                for l in 0..n{for(jl,&j)in c.iter().enumerate(){vectors[j+d*(offset[k]+l)]=system.vectors[jl+n*l];inverse[offset[k]+l+d*j]=system.inverse[l+n*jl];}}}
            let pair:Vec<bool>=systems.iter().flat_map(|s|s.pair.iter().copied()).collect();
            return Ok(Clustered{values,vectors,inverse,pair,clusters:clusters.len(),largest,passes:pass+1});
        }
        // Y = m V_c (d x d, eigen-index columns), K' = V_c^-1 Y: per cluster,
        // a large one in panels of the other dimension.
        let tasks:Vec<(usize,usize,usize)>=(0..clusters.len()).flat_map(|k|{let step=if clusters[k].len()>=CLUSTER_SPLIT{CLUSTER_PANEL}else{d};
            (0..d).step_by(step).map(move|s|(k,s,(s+step).min(d)))}).collect();
        let slabs=execution.map(tasks.clone(),|(k,s,e)|->Result<Vec<C>,String>{let(c,n,w)=(&clusters[k],clusters[k].len(),e-s);
            let mut columns=vec![ZERO;w*n];for(l,&j)in c.iter().enumerate(){columns[l*w..(l+1)*w].copy_from_slice(&m[j*d+s..j*d+e]);}
            let mut out=vec![ZERO;w*n];lapack.complex_gemm().multiply(w,n,n,ONE,&columns,w,&systems[k].vectors,n,ZERO,&mut out,w)?;Ok(out)});
        let mut y=vec![ZERO;d*d];
        for(&(k,s,e),slab)in tasks.iter().zip(slabs){let(slab,w)=(slab?,e-s);for l in 0..clusters[k].len(){let a=offset[k]+l;y[a*d+s..a*d+e].copy_from_slice(&slab[l*w..(l+1)*w]);}}
        // K' in panels of columns over every cluster.
        let columns:Vec<usize>=(0..d).step_by(CLUSTER_PANEL).collect();
        let panels=execution.map(columns.clone(),|s|->Result<Vec<C>,String>{let e=(s+CLUSTER_PANEL).min(d);let w=e-s;let gemm=lapack.complex_gemm();
            let mut panel=vec![ZERO;d*w];
            for(k,c)in clusters.iter().enumerate(){let n=c.len();
                let mut rows=vec![ZERO;n*w];for a in s..e{for(l,&j)in c.iter().enumerate(){rows[l+n*(a-s)]=y[j+d*a];}}
                let mut out=vec![ZERO;n*w];gemm.multiply(n,w,n,ONE,&systems[k].inverse,n,&rows,n,ZERO,&mut out,n)?;
                for a in 0..w{panel[offset[k]+d*a..offset[k]+n+d*a].copy_from_slice(&out[n*a..n*(a+1)]);}}
            Ok(panel)});
        drop(y);
        let mut coupling=Vec::with_capacity(d*d);for panel in panels{coupling.extend(panel?);}
        // First-order mixing between clusters, second-order values, and the
        // checks: per eigenvector a, its column of T, its value's shift, the
        // clusters that dominate them and the merge it asks for (K'_ab and
        // K'_ba by tiles of b).
        let(coupling_ref,values_ref,owner_ref)=(&coupling,&values,&owner);
        let checked=execution.map(columns.clone(),|s|{let e=(s+CLUSTER_PANEL).min(d);let w=e-s;let mut t_part=vec![ZERO;d*w];
            // Per a: the most mixed cluster and its mixing, the most shifting one and its shift, the shifts' sum, the shift.
            let mut state=vec![(usize::MAX,0_f64,usize::MAX,0_f64,0_f64,ZERO);w];
            for b0 in(0..d).step_by(CHECK_TILE){let b1=(b0+CHECK_TILE).min(d);
                for a in s..e{let(most_mixed,largest_mix,most_shifting,largest_shift,shifts,shift_a)=&mut state[a-s];
                    for b in b0..b1{if owner_ref[b]==owner_ref[a]{continue;}
                        let kba=coupling_ref[b+d*a];if !nonzero(kba){continue;}
                        let gap=C{re:values_ref[a].re-values_ref[b].re,im:values_ref[a].im-values_ref[b].im};let g=gap.re*gap.re+gap.im*gap.im;
                        let t=if g>0.{mul(kba,C{re:gap.re/g,im:-gap.im/g})}else{C{re:f64::INFINITY,im:0.}};
                        t_part[b+d*(a-s)]=t;let shift=mul(coupling_ref[a+d*b],t);add(shift_a,shift);
                        let(tm,sm)=(magnitude(t),magnitude(shift));*shifts+=sm;
                        if tm>*largest_mix{*largest_mix=tm;*most_mixed=owner_ref[b];}
                        if sm>*largest_shift{*largest_shift=sm;*most_shifting=owner_ref[b];}}}}
            let out:Vec<(C,Option<(usize,usize)>)>=state.into_iter().enumerate().map(|(i,(most_mixed,largest_mix,most_shifting,_,shifts,shift_a))|{let a=s+i;
                let lives=if values_ref[a].re<0.{horizon.min(1./-values_ref[a].re)}else{horizon};
                let merge=if largest_mix>mixing_bound{Some((owner_ref[a],most_mixed))}else if shifts*largest_mix*lives>phase{Some((owner_ref[a],most_shifting))}else{None};
                (shift_a,merge)}).collect();
            (t_part,out)});
        drop(coupling);
        let mut mixing=Vec::with_capacity(d*d);let mut shifted=values.clone();let mut merge:Vec<(usize,usize)>=Vec::new();
        for(&s,(t_part,out))in columns.iter().zip(checked){mixing.extend(t_part);for(i,(shift,wanted))in out.into_iter().enumerate(){add(&mut shifted[s+i],shift);merge.extend(wanted);}}
        if !merge.is_empty(){
            if pass+1==CLUSTER_PASSES{break;}
            for(x,y)in merge{let(a,b)=(clusters[x][0],clusters[y][0]);join(&mut parent,atom[a],atom[b]);}
            kept=clusters.into_iter().zip(systems).collect();
            continue;
        }
        // V = V_c (1 + T): rows of cluster k (in panels of columns); V^-1 =
        // (1 - T) V_c^-1: columns of cluster k (in panels of rows).
        let panels=execution.map(columns,|s|->Result<Vec<C>,String>{let e=(s+CLUSTER_PANEL).min(d);let w=e-s;let gemm=lapack.complex_gemm();
            let mut panel=vec![ZERO;d*w];
            for(k,c)in clusters.iter().enumerate(){let n=c.len();
                let mut right=vec![ZERO;n*w];for a in s..e{right[n*(a-s)..n*(a-s+1)].copy_from_slice(&mixing[offset[k]+d*a..offset[k]+n+d*a]);
                    if a>=offset[k]&&a<offset[k]+n{right[a-offset[k]+n*(a-s)]=ONE;}}
                let mut rows=vec![ZERO;n*w];gemm.multiply(n,w,n,ONE,&systems[k].vectors,n,&right,n,ZERO,&mut rows,n)?;
                for a in 0..w{for(l,&j)in c.iter().enumerate(){panel[j+d*a]=rows[l+n*a];}}}
            Ok(panel)});
        let parts=execution.map(tasks.clone(),|(k,s,e)|->Result<Vec<C>,String>{let(n,w)=(clusters[k].len(),e-s);
            let mut left=vec![ZERO;w*n];for l in 0..n{let b=offset[k]+l;for a in s..e{let t=mixing[a+d*b];left[(a-s)+w*l]=if a==b{ONE}else{C{re:-t.re,im:-t.im}};}}
            let mut columns=vec![ZERO;w*n];lapack.complex_gemm().multiply(w,n,n,ONE,&left,w,&systems[k].inverse,n,ZERO,&mut columns,w)?;
            Ok(columns)});
        let mut vectors=Vec::with_capacity(d*d);for panel in panels{vectors.extend(panel?);}
        let mut inverse=vec![ZERO;d*d];
        for(&(k,s,e),part)in tasks.iter().zip(parts){let(columns,c)=(part?,&clusters[k]);let w=e-s;
            for(l,&j)in c.iter().enumerate(){inverse[j*d+s..j*d+e].copy_from_slice(&columns[l*w..(l+1)*w]);}}
        let pair:Vec<bool>=systems.iter().flat_map(|s|s.pair.iter().copied()).collect();
        // A pair's values are conjugate.
        for l in 0..d{if pair[l]{shifted[l+1]=conj(shifted[l]);}}
        return Ok(Clustered{values:shifted,vectors,inverse,pair,clusters:clusters.len(),largest,passes:pass+1});
    }
    // One cluster, or the couplings still too strong after CLUSTER_PASSES merges: the whole.
    let whole=real_eigensystem(lapack,first,m,d)?;
    Ok(Clustered{values:whole.values,vectors:whole.vectors,inverse:whole.inverse,pair:whole.pair,clusters:1,largest:d,passes})
}

/// A cluster's eigensystem (`real_eigensystem`).
struct RealSystem{values:Vec<C>,vectors:Vec<C>,inverse:Vec<C>,pair:Vec<bool>}

/// The eigensystem of a flip-invariant matrix m (column-major, complex
/// coordinates z, `pairs[j]` the first of a conjugate pair (j, j + 1)). With
/// the real coordinates x of a pair, z = T x, T = 1/2 [[1, -i], [1, i]], the
/// matrix T^-1 m T is real; its real eigenvectors VR give the complex ones
/// V_c = VR S, S = [[1, 1], [i, -i]] per eigenvalue pair, so V = T V_c and
/// V^-1 = S^-1 VR^-1 T^-1.
fn real_eigensystem(lapack:&Lapack,pairs:&[bool],m:&[C],d:usize)->Result<RealSystem,String>{
    let first=|j:usize|pairs[j];
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
    if imaginary>1e-10*size{return Err(format!("Eigensystem matrix is not flip invariant ({:e})",imaginary/size));}
    let(wr,wi,vr)=lapack.eigen(&real,d)?;drop(real);
    if vr.iter().any(|v|!v.is_finite()){return Err("Nonfinite eigenvectors".into());}
    let vri=real_inverse(lapack,&vr,d)?;
    let(kind,values)=conjugate_pairs(&wr,&wi);let pair:Vec<bool>=kind.iter().map(|&k|k==1).collect();
    // V_c = VR S (columns), V = T V_c (rows).
    let mut vc:Vec<C>=vr.iter().map(|&v|C{re:v,im:0.}).collect();
    for l in(0..d).filter(|&l|pair[l]){for i in 0..d{let(a,b)=(vr[i+d*l],vr[i+d*(l+1)]);vc[i+d*l]=C{re:a,im:b};vc[i+d*(l+1)]=C{re:a,im:-b};}}
    let mut vectors=vc.clone();
    for c in 0..d{for j in(0..d).filter(|&j|first(j)){(vectors[j+d*c],vectors[j+1+d*c])=pair_coordinates(vc[j+d*c],vc[j+1+d*c]);}}
    // V^-1 = S^-1 VR^-1 (rows), then T^-1 (columns).
    let mut w:Vec<C>=vri.iter().map(|&v|C{re:v,im:0.}).collect();
    for l in(0..d).filter(|&l|pair[l]){for c in 0..d{let(a,b)=(vri[l+d*c],vri[l+1+d*c]);
        w[l+d*c]=C{re:0.5*a,im:-0.5*b};w[l+1+d*c]=C{re:0.5*a,im:0.5*b};}}
    let mut inverse=w.clone();
    for j in(0..d).filter(|&j|first(j)){for r in 0..d{let(a,b)=(w[r+d*j],w[r+d*(j+1)]);let ib=i_times(b);
        inverse[r+d*j]=C{re:a.re+ib.re,im:a.im+ib.im};inverse[r+d*(j+1)]=C{re:a.re-ib.re,im:a.im-ib.im};}}
    Ok(RealSystem{values,vectors,inverse,pair})
}

/// The inverse of a real matrix (column-major d x d): one LU, the identity's
/// columns solved in panels on the workers.
fn real_inverse(lapack:&Lapack,a:&[f64],d:usize)->Result<Vec<f64>,String>{
    let(lu,pivots)=lapack.factor(a,d)?;let(lu,pivots)=(&lu,&pivots);
    let panels:Vec<usize>=(0..d).step_by(INVERSE_PANEL).collect();
    let parts=Execution::new()?.map(panels,|start|->Result<Vec<f64>,String>{let width=INVERSE_PANEL.min(d-start);
        let mut columns=vec![0.;d*width];for c in 0..width{columns[start+c+d*c]=1.;}lapack.solve_matrix_factored(lu,pivots,&columns,d)});
    let mut out=Vec::with_capacity(d*d);for part in parts{out.extend(part?);}
    Ok(out)
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn clustered_eigensystems_are_exact_within_clusters_and_second_order_between(){
        // A flip-invariant matrix (real coordinates x, pairs z = T x): two
        // strongly coupled groups and weak couplings eps between them. Clusters
        // form; m V = V diag(mu) and V^-1 V = I hold to O(eps^2) and the
        // values to O(eps^3) against the whole eigensystem.
        let f=fixture();let p=prepared(&f);let lapack=&p.lapack;
        let first=[false,true,false,false,true,false,false];let d=first.len();
        let eps=1e-3;
        // Real form: a single (0), a pair (1, 2) as a rotation block, a single (3),
        // a pair (4, 5) and a single (6); 0-1-2 strongly coupled, 3-4-5-6 strongly
        // coupled, everything else at eps of the gaps.
        let mut r=vec![0_f64;d*d];
        let diag=[-1.,-3.,-3.,-20.,-25.,-25.,-24.];for j in 0..d{r[j+d*j]=diag[j];}
        r[1+d*2]=-7.;r[2+d*1]=7.;r[4+d*5]=-30.;r[5+d*4]=30.;
        let strong=[(0,1,2.),(2,0,1.5),(3,4,4.),(5,6,3.),(6,3,2.5)];for &(i,j,v)in &strong{r[i+d*j]=v;}
        for i in 0..d{for j in 0..d{if r[i+d*j]==0.{r[i+d*j]=eps*(((i*5+j*3)%7)as f64-3.);}}}
        // m = T R T^-1 per pair: z_j = (x_j - i x_j+1)/2, z_j+1 = (x_j + i x_j+1)/2.
        let mut t=vec![ZERO;d*d];let mut ti=vec![ZERO;d*d];let mut j=0;
        while j<d{if first[j]{t[j+d*j]=C{re:0.5,im:0.};t[j+d*(j+1)]=C{re:0.,im:-0.5};t[j+1+d*j]=C{re:0.5,im:0.};t[j+1+d*(j+1)]=C{re:0.,im:0.5};
            ti[j+d*j]=ONE;ti[j+d*(j+1)]=ONE;ti[j+1+d*j]=C{re:0.,im:1.};ti[j+1+d*(j+1)]=C{re:0.,im:-1.};j+=2;}else{t[j+d*j]=ONE;ti[j+d*j]=ONE;j+=1;}}
        let product=|a:&[C],b:&[C]|->Vec<C>{let mut c=vec![ZERO;d*d];for i in 0..d{for k in 0..d{for l in 0..d{mul_add(&mut c[i+d*k],a[i+d*l],b[l+d*k]);}}}c};
        let rc:Vec<C>=r.iter().map(|&v|C{re:v,im:0.}).collect();let m=product(&product(&t,&rc),&ti);
        let base:Vec<C>=(0..d).map(|j|m[j+d*j]).collect();
        let c=clustered_eigensystem(lapack,&m,&base,&first,ClusterControl::blocks(1.)).unwrap();
        assert!(c.clusters==2&&c.largest==4,"clusters {} largest {}",c.clusters,c.largest);
        let(mv,vi)=(product(&m,&c.vectors),product(&c.inverse,&c.vectors));
        let scale=m.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        for a in 0..d{for i in 0..d{
            let e=C{re:mv[i+d*a].re-(c.vectors[i+d*a].re*c.values[a].re-c.vectors[i+d*a].im*c.values[a].im),im:mv[i+d*a].im-(c.vectors[i+d*a].re*c.values[a].im+c.vectors[i+d*a].im*c.values[a].re)};
            assert!(magnitude(e)<=100.*eps*eps*scale,"eigenpair {a}: {e:?}");
            assert!(magnitude(C{re:vi[i+d*a].re-f64::from(u8::from(i==a)),im:vi[i+d*a].im})<=100.*eps*eps,"inverse ({i}, {a})");}}
        // The values against the whole eigensystem (conjugate pairs adjacent).
        let whole=real_eigensystem(lapack,&first,&m,d).unwrap();
        for v in &c.values{let nearest=whole.values.iter().map(|w|magnitude(C{re:v.re-w.re,im:v.im-w.im})).fold(f64::INFINITY,f64::min);
            assert!(nearest<=100.*eps*eps*eps*scale,"value {v:?}: {nearest:e}");}
        for l in 0..d{if c.pair[l]{assert!(c.values[l+1].re==c.values[l].re&&c.values[l+1].im==-c.values[l].im);}}
    }
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
}
