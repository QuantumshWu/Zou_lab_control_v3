//! The reduced model of the space and every output from it.
//!
//! Galerkin in the dressed eigen-coordinates: the inner product of the
//! reduced model is Re <T a, T b> with T = blockdiag(E_R^-1) on Hermitian-
//! basis coordinates (`eigenbasis::dressed`), where the lattice's
//! block-diagonal part is diagonal (rotations for complex pairs) with
//! Re <= 0, so only the couplings between blocks push the reduced numerical
//! range to the right (in the Euclidean product the conditioning of E_R
//! does too). With the Gram matrices K = (TW)^T TW, H = (TW)^T T(AW) and
//! the ordered Cholesky K = L L^T: X = L^-T, G = X^T H X, c0 = X^T K e_0 |x0|
//! (w_0 = x0 / |x0|); x(t) ~ W X u(t), u' = G u, u(0) = c0. The dressed
//! columns and the Gram matrices grow with the space (append-only), on the
//! canonical modes (a mirror mode's dressed block is the conjugate).
//!
//! Trace correction: tr x_q evolves by -i nu_q tr x_q exactly; G is replaced
//! by the Frobenius-closest matrix with the trace rows of W X as left
//! eigenvectors, on the significant rows only (above TRACE_ROWS of the
//! largest): the other rows are inner-solve roundoff, and fitting them
//! turns roundoff into O(1) changes of G.
//!
//! Ritz decomposition G S = S diag(theta), a = S^-1 c0. The kernel holds
//! |theta| < KERNEL and always the value of least modulus (the trace
//! correction pins one exact zero). Growing modes by amplitude (the mode's
//! largest population contribution): growth below GROWTH_NEGLIGIBLE over the
//! longest output time clipped to the axis; amplitude up to GROWTH_BUDGET x
//! tolerance projected out (charged); otherwise reflected (an unconverged
//! pseudo-mode the estimate sees).
//!
//! Outputs from the one G: the trajectory p(t) = sum_q Re e^{i nu_q t}
//! diag(W X u(t))_q (the local time average); the state at a horizon T, its
//! centre block (the laboratory time average at T); the steady state from
//! the bordered small system (G + c0 tau^T) r = c0 (tau the centre trace
//! row; a simple kernel), else the kernel's spectral projection; the slowest
//! relaxation among the converged Ritz values and the conserved count.
//!
//! The error transform (R1) orders the poles: the Laplace transform of the
//! population error with the residual propagated through the dressed blocks,
//! p_err(xi) = sum_q W_d (xi - Lambda)^-1 Z_q (xi - i nu_q - Theta)^-1 1,
//! Z_q = W^-1 (R S diag a)_q; sigma |p_err(sigma + i omega)| is the mean
//! error near t = 1/sigma at frequency omega.
use super::*;
use super::eigenbasis::BlockEigenbasis;
use super::operator::LatticeOperator;
use crate::parallel::Execution;

/// Trace rows below this fraction of the largest are left unconstrained.
const TRACE_ROWS:f64=1e-6;
/// Singular values of the trace rows below this fraction are dropped.
const TRACE_RANK:f64=1e-10;
/// |theta| below this (1/us) is a conserved quantity.
pub const KERNEL:f64=1e-9;
/// Growth (e^{Re theta T} - 1) x amplitude below this is clipped.
const GROWTH_NEGLIGIBLE:f64=1e-6;
/// Amplitude (times the tolerance) a growing mode may have to be projected
/// out.
const GROWTH_BUDGET:f64=0.1;
/// Dependent columns of the dressed Gram matrix (squared ratio).
const DEPENDENT:f64=1e-14;
/// A Ritz pair counts as converged (slowest relaxation) when its relative
/// residual is below this share of |Re theta|.
const RITZ_CONVERGED:f64=0.1;

/// Row-major (r x k) (k x c), the column-major (A B)^T = B^T A^T.
pub(crate) fn product(lapack:&Lapack,a:&[f64],b:&[f64],r:usize,k:usize,c:usize)->Result<Vec<f64>,String>{
    let mut out=vec![0.;r*c];if r*k*c>0{lapack.gemm(false,false,c,r,k,1.,b,c,a,k,0.,&mut out,c)?;}Ok(out)
}
pub(crate) fn transpose_rows(a:&[f64],r:usize,c:usize)->Vec<f64>{let mut t=vec![0.;r*c];for i in 0..r{for j in 0..c{t[j*r+i]=a[i*c+j];}}t}
/// A real view of complex data (Complex64 is repr(C) {re, im}).
fn real_view(x:&[C])->&[f64]{unsafe{std::slice::from_raw_parts(x.as_ptr()as *const f64,2*x.len())}}
/// Re (A^H B) for column-major complex A (length x m) and B (length x c):
/// row-major m x c, the length split over the workers (partial sums added
/// in a fixed order).
fn real_gram(lapack:&Lapack,a:&[C],m:usize,b:&[C],c:usize,length:usize)->Result<Vec<f64>,String>{
    if m==0||c==0{return Ok(vec![0.;m*c]);}
    let(ar,br)=(real_view(a),real_view(b));let total=2*length;let execution=Execution::new()?;
    let chunk=total.div_ceil(execution.threads().max(1)).max(4096);
    let starts:Vec<usize>=(0..total).step_by(chunk).collect();
    let parts=execution.map(starts,|start|->Result<Vec<f64>,String>{let end=(start+chunk).min(total);let mut out=vec![0.;m*c];
        lapack.gemm(true,false,m,c,end-start,1.,&ar[start..],total,&br[start..],total,0.,&mut out,m)?;Ok(out)});
    let mut gram=vec![0.;m*c];for part in parts{let part=part?;for i in 0..m{for j in 0..c{gram[i*c+j]+=part[i+m*j];}}}
    Ok(gram)
}

/// The dressed columns of the space and their Gram matrices.
pub struct Dressed{
    n:usize,d:usize,
    /// Canonical modes: (mode, weight, nu); the centre's position.
    pub canonical:Vec<(usize,f64,f64)>,centre:usize,
    /// sqrt(weight)-scaled dressed coordinates on the canonical modes,
    /// column-major (length L = canonical x d, one column per basis vector),
    /// of the basis and of its images.
    basis:Vec<C>,images:Vec<C>,length:usize,
    /// Per column: the canonical modes' diagonals (canonical x n) and the
    /// trace functionals [centre, then Re and Im per other canonical mode].
    populations:Vec<Vec<C>>,traces:Vec<Vec<f64>>,
    /// K, H and M = T(AW)^T T(AW), row-major over the covered columns.
    k:Vec<f64>,h:Vec<f64>,m:Vec<f64>,covered:usize,
    pub milliseconds:f64,
}
impl Dressed{
    pub fn new(op:&LatticeOperator<'_>)->Self{
        let n=op.n;let d=n*n;
        let canonical:Vec<(usize,f64,f64)>=(0..op.modes.len()).filter(|&q|op.canonical[q]).map(|q|(q,if op.mirror[q]==q{1.}else{2.},op.nu[q])).collect();
        let centre=canonical.iter().position(|c|c.0==op.centre).unwrap_or(0);
        let length=canonical.len()*d;
        Self{n,d,canonical,centre,basis:vec![],images:vec![],length,populations:vec![],traces:vec![],k:vec![],h:vec![],m:vec![],covered:0,milliseconds:0.}
    }
    pub fn columns(&self)->usize{self.covered}
    /// Takes the space's columns not yet covered.
    pub fn ingest(&mut self,lapack:&Lapack,eigen:&BlockEigenbasis,basis:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        let clock=Instant::now();let(old,new)=(self.covered,basis.len());if new==old{return Ok(());}
        let(n,d)=(self.n,self.d);let l=self.length;
        // Every new column and image in one batch of dressed coordinates.
        let blocks:Vec<C>=(old..new).flat_map(|j|[&basis[j],&images[j]]).flat_map(|field|self.canonical.iter().flat_map(move|&(q,_,_)|field[q*d..(q+1)*d].iter().copied())).collect();
        let mut dressed=eigen.dressed(&blocks)?;
        for(c,&(_,w,_))in self.canonical.iter().enumerate(){let s=w.sqrt();for column in dressed.chunks_exact_mut(l){for v in &mut column[c*d..(c+1)*d]{*v=scaled(*v,s);}}}
        for(j,pair)in(old..new).zip(dressed.chunks_exact(2*l)){
            self.basis.extend_from_slice(&pair[..l]);self.images.extend_from_slice(&pair[l..]);
            let pops:Vec<C>=self.canonical.iter().flat_map(|&(q,_,_)|(0..n).map(move|i|basis[j][q*d+i*n+i])).collect();
            let mut traces=vec![0.;2*self.canonical.len()-1];let mut at=1;
            for(c,_)in self.canonical.iter().enumerate(){let t=(0..n).fold(ZERO,|s,i|{let v=pops[c*n+i];C{re:s.re+v.re,im:s.im+v.im}});
                if c==self.centre{traces[0]=t.re;}else{traces[at]=t.re;traces[at+1]=t.im;at+=2;}}
            self.populations.push(pops);self.traces.push(traces);
        }
        let width=new-old;
        let(b_new,a_new)=(&self.basis[old*l..new*l],&self.images[old*l..new*l]);
        let kb=real_gram(lapack,&self.basis,new,b_new,width,l)?;           // K[:, old..new]
        let hb=real_gram(lapack,&self.basis,new,a_new,width,l)?;           // H[:, old..new]
        let hr=real_gram(lapack,b_new,width,&self.images[..old*l],old,l)?; // H[old..new, ..old]
        let mb=real_gram(lapack,&self.images,new,a_new,width,l)?;          // M[:, old..new]
        let grow=|old_matrix:&[f64]|->Vec<f64>{let mut g=vec![0.;new*new];for i in 0..old{for j in 0..old{g[i*new+j]=old_matrix[i*old+j];}}g};
        let(mut k,mut h,mut m)=(grow(&self.k),grow(&self.h),grow(&self.m));
        for i in 0..new{for c in 0..width{let j=old+c;
            k[i*new+j]=kb[i*width+c];k[j*new+i]=kb[i*width+c];m[i*new+j]=mb[i*width+c];m[j*new+i]=mb[i*width+c];h[i*new+j]=hb[i*width+c];}}
        for c in 0..width{for j in 0..old{h[(old+c)*new+j]=hr[c*old+j];}}
        self.k=k;self.h=h;self.m=m;self.covered=new;
        self.milliseconds+=clock.elapsed().as_secs_f64()*1e3;Ok(())
    }
}

/// The reduced model of one checkpoint.
pub struct Model{
    /// Reduced dimension, X (k x r row-major), G (r x r), c0.
    pub r:usize,pub x:Vec<f64>,pub g:Vec<f64>,pub c0:Vec<f64>,
    /// Ritz values, eigenvectors (r x r row-major, complex), coefficients,
    /// and the values and coefficients the outputs use.
    pub theta:Vec<C>,pub vectors:Vec<C>,pub coefficients:Vec<C>,
    pub used_theta:Vec<C>,pub used_coefficients:Vec<C>,pub kernel:Vec<bool>,
    /// Relative Ritz residuals (dressed metric) and mode amplitudes.
    pub residuals:Vec<f64>,pub amplitudes:Vec<f64>,
    pub growing:usize,pub projected:usize,pub reflected:usize,pub growth_charge:f64,
    /// The reduced steady state (kernel part of c0).
    pub steady:Vec<f64>,pub simple_kernel:bool,
    /// Per canonical mode: populations of W X (n x r, row-major complex).
    readout:Vec<Vec<C>>,
    pub gram_condition:f64,
}

impl Dressed{
    /// The reduced model over the covered columns; `initial_norm` = |x0|,
    /// `longest` the longest output time (growth), `tolerance` the
    /// population tolerance.
    pub fn model(&self,lapack:&Lapack,initial_norm:f64,longest:f64,tolerance:f64)->Result<Model,String>{
        let k=self.covered;let n=self.n;
        let(x,r,gram_condition)=super::krylov::orthonormal_coordinates(&self.k,k,DEPENDENT);
        let xt=transpose_rows(&x,k,r);
        let g0=product(lapack,&product(lapack,&xt,&self.h,r,k,k)?,&x,r,k,r)?;
        let first:Vec<f64>=(0..k).map(|i|initial_norm*self.k[i*k]).collect();
        let c0=product(lapack,&xt,&first,r,k,1)?;
        // Trace correction on the significant rows.
        let kt=self.traces[0].len();
        let trace_rows:Vec<f64>=(0..kt).flat_map(|t|(0..k).map(move|j|(t,j))).map(|(t,j)|self.traces[j][t]).collect();
        let tw=product(lapack,&trace_rows,&x,kt,k,r)?;
        let row_norm=|t:usize|tw[t*r..(t+1)*r].iter().map(|v|v*v).sum::<f64>();
        let charge_norm:Vec<f64>=(0..self.canonical.len()).map(|c|if c==self.centre{row_norm(0)}else{let a=1+2*(c-usize::from(c>self.centre));row_norm(a)+row_norm(a+1)}).map(f64::sqrt).collect();
        let largest=charge_norm.iter().fold(0_f64,|a,&b|a.max(b));
        let mut kept:Vec<(usize,usize,f64)>=Vec::new();// (row, partner row or usize::MAX, nu)
        for(c,&(_,_,nu))in self.canonical.iter().enumerate(){if !(charge_norm[c]>TRACE_ROWS*largest){continue;}
            if c==self.centre{kept.push((0,usize::MAX,0.));}else{let a=1+2*(c-usize::from(c>self.centre));kept.push((a,a+1,nu));kept.push((a+1,a,-nu));}}
        let mut g=g0.clone();
        if !kept.is_empty(){
            let rows=kept.len();let t:Vec<f64>=kept.iter().flat_map(|&(row,_,_)|tw[row*r..(row+1)*r].iter().copied()).collect();
            // Omega: d(re)/dt = nu im, d(im)/dt = -nu re; rows in kept order.
            let position=|row:usize|kept.iter().position(|e|e.0==row);
            let tg=product(lapack,&t,&g0,rows,r,r)?;let mut defect=tg;
            for(i,&(_,partner,nu))in kept.iter().enumerate(){if partner==usize::MAX{continue;}let Some(p)=position(partner)else{continue};
                for c in 0..r{defect[i*r+c]-=nu*t[p*r+c];}}
            let gram=product(lapack,&t,&transpose_rows(&t,rows,r),rows,r,rows)?;
            let complex:Vec<C>=gram.iter().map(|&v|C{re:v,im:0.}).collect();
            let(values,vectors)=lapack.hermitian_eigen(&complex,rows)?;let top=values.iter().fold(0_f64,|a,&v|a.max(v));
            let mut pseudo=vec![0.;rows*rows];
            for j in 0..rows{if values[j]>TRACE_RANK*TRACE_RANK*top{for a in 0..rows{for b in 0..rows{pseudo[a*rows+b]+=mul(vectors[a+rows*j],conj(vectors[b+rows*j])).re/values[j];}}}}
            let correction=product(lapack,&product(lapack,&transpose_rows(&t,rows,r),&pseudo,r,rows,rows)?,&defect,r,rows,r)?;
            for(a,b)in g.iter_mut().zip(&correction){*a-=b;}
        }
        // Ritz decomposition.
        let(re,im,ev)=lapack.eigen(&transpose_rows(&g,r,r),r)?;
        let mut theta=vec![ZERO;r];let mut s=vec![ZERO;r*r];let mut pair=vec![0_u8;r];let mut j=0;
        while j<r{if im[j]!=0.&&j+1<r{theta[j]=C{re:re[j],im:im[j]};theta[j+1]=C{re:re[j],im:-im[j]};pair[j]=1;pair[j+1]=2;
                for i in 0..r{let(a,b)=(ev[i+r*j],ev[i+r*(j+1)]);s[i*r+j]=C{re:a,im:b};s[i*r+j+1]=C{re:a,im:-b};}j+=2;}
            else{theta[j]=C{re:re[j],im:0.};for i in 0..r{s[i*r+j]=C{re:ev[i+r*j],im:0.};}j+=1;}}
        let(lu,pivots)=lapack.factor(&ev,r)?;let y=lapack.solve_matrix_factored(&lu,&pivots,&c0,r)?;
        let mut a=vec![ZERO;r];let mut j=0;
        while j<r{if pair[j]==1{a[j]=C{re:0.5*y[j],im:-0.5*y[j+1]};a[j+1]=C{re:0.5*y[j],im:0.5*y[j+1]};j+=2;}else{a[j]=C{re:y[j],im:0.};j+=1;}}
        // Readout of W X per canonical mode, and the modes' amplitudes.
        let readout:Vec<Vec<C>>=(0..self.canonical.len()).map(|c|{
            let mut out=vec![ZERO;n*r];for i in 0..n{for jj in 0..k{let p=self.populations[jj][c*n+i];if !nonzero(p){continue;}
                for col in 0..r{let v=x[jj*r+col];if v!=0.{out[i*r+col].re+=p.re*v;out[i*r+col].im+=p.im*v;}}}}out}).collect();
        let amplitudes:Vec<f64>=(0..r).map(|jj|{let mut largest=0_f64;
            for block in &readout{for i in 0..n{let mut v=ZERO;for col in 0..r{mul_add(&mut v,block[i*r+col],s[col*r+jj]);}largest=largest.max(magnitude(v));}}
            magnitude(a[jj])*largest}).collect();
        // Relative Ritz residuals in the dressed metric: y = X s_j,
        // |T(R s)|^2 = y^H M y - 2 Re(conj(theta) y^H H y) + |theta|^2 y^H K y.
        let gemm=lapack.complex_gemm();
        let to_complex=|a:&[f64]|->Vec<C>{a.iter().map(|&v|C{re:v,im:0.}).collect()};
        // Y = X S (k x r) and the quadratic forms y_j^H M y_j etc. (column-major).
        let(xc,sc)=(to_complex(&transpose_rows(&x,k,r)),(0..r*r).map(|e|s[(e%r)*r+e/r]).collect::<Vec<C>>());
        let mut ys=vec![ZERO;k*r];gemm.multiply(k,r,r,ONE,&xc,k,&sc,r,ZERO,&mut ys,k)?;
        let quadratic=|m:&[f64]|->Result<Vec<C>,String>{let mc=to_complex(&transpose_rows(m,k,k));let mut my=vec![ZERO;k*r];gemm.multiply(k,r,k,ONE,&mc,k,&ys,k,ZERO,&mut my,k)?;
            Ok((0..r).map(|j|(0..k).fold(ZERO,|acc,i|{let mut acc=acc;mul_add(&mut acc,conj(ys[i+k*j]),my[i+k*j]);acc})).collect())};
        let(ym,yh,yk)=(quadratic(&self.m)?,quadratic(&self.h)?,quadratic(&self.k)?);
        let residuals:Vec<f64>=(0..r).map(|jj|{let t=theta[jj];
            let value=ym[jj].re-2.*mul(conj(t),yh[jj]).re+(t.re*t.re+t.im*t.im)*yk[jj].re;
            value.max(0.).sqrt()/yk[jj].re.max(f64::MIN_POSITIVE).sqrt()}).collect();
        // Kernel and growing modes.
        let least=(0..r).min_by(|&p,&q|magnitude(theta[p]).total_cmp(&magnitude(theta[q]))).ok_or("Empty reduced model")?;
        let kernel:Vec<bool>=(0..r).map(|jj|jj==least||magnitude(theta[jj])<KERNEL).collect();
        let(mut used_theta,mut used_coefficients)=(theta.clone(),a.clone());
        let(mut growing,mut projected,mut reflected,mut growth_charge)=(0,0,0,0.);
        for jj in 0..r{
            if kernel[jj]{used_theta[jj]=ZERO;continue;}
            if !(theta[jj].re>0.){continue;}
            let growth=((theta[jj].re*longest).min(700.).exp()-1.)*amplitudes[jj];
            if growth<=GROWTH_NEGLIGIBLE{used_theta[jj].re=0.;continue;}
            growing+=1;
            if amplitudes[jj]<=GROWTH_BUDGET*tolerance{used_theta[jj]=ZERO;used_coefficients[jj]=ZERO;projected+=1;growth_charge+=amplitudes[jj];}
            else{used_theta[jj].re=-theta[jj].re;reflected+=1;}
        }
        // The steady state.
        let simple_kernel=kernel.iter().filter(|&&b|b).count()==1;
        let tau:Vec<f64>=tw[..r].to_vec();
        let steady:Vec<f64>=if simple_kernel{
            let mut bordered=g.clone();for i in 0..r{for c in 0..r{bordered[i*r+c]+=c0[i]*tau[c];}}
            let solution=lapack.solve(&transpose_rows(&bordered,r,r),&c0,r)?;
            let trace:f64=tau.iter().zip(&c0).map(|(a,b)|a*b).sum();solution.iter().map(|v|v*trace).collect()
        }else{
            (0..r).map(|i|(0..r).filter(|&jj|kernel[jj]).fold(0.,|acc,jj|acc+mul(s[i*r+jj],a[jj]).re)).collect()
        };
        if steady.iter().chain(&g).any(|v|!v.is_finite()){return Err("Nonfinite reduced model".into());}
        Ok(Model{r,x,g,c0,theta,vectors:s,coefficients:a,used_theta,used_coefficients,kernel,residuals,amplitudes,
            growing,projected,reflected,growth_charge,steady,simple_kernel,readout,gram_condition})
    }
}

fn cexp(z:C,t:f64)->C{let m=(z.re*t).exp();C{re:m*(z.im*t).cos(),im:m*(z.im*t).sin()}}
impl Model{
    /// u(t) = Re S diag(e^{used theta t}) used coefficients.
    pub fn state(&self,t:f64)->Vec<f64>{
        let r=self.r;let e:Vec<C>=(0..r).map(|j|mul(cexp(self.used_theta[j],t),self.used_coefficients[j])).collect();
        (0..r).map(|i|(0..r).fold(0.,|acc,j|acc+mul(self.vectors[i*r+j],e[j]).re)).collect()
    }
    /// Populations of the reduced vector u at time t (lab phases), or on the
    /// centre alone (`centre`: the laboratory time average).
    fn populations(&self,dressed:&Dressed,u:&[f64],t:f64,centre:bool)->Vec<f64>{
        let(n,r)=(dressed.n,self.r);let mut p=vec![0.;n];
        for(c,&(_,w,nu))in dressed.canonical.iter().enumerate(){
            if centre&&c!=dressed.centre{continue;}let phase=cexp(C{re:0.,im:nu},t);let block=&self.readout[c];
            for(i,pi)in p.iter_mut().enumerate(){let mut v=ZERO;for(col,&uc)in u.iter().enumerate(){let b=block[i*r+col];v.re+=b.re*uc;v.im+=b.im*uc;}
                *pi+=if centre{v.re}else{w*mul(phase,v).re};}
        }
        p
    }
    /// The trajectory's populations on the grid.
    pub fn trajectory(&self,dressed:&Dressed,times:&[f64])->Vec<Vec<f64>>{
        Execution::new().map(|e|e.map(times.to_vec(),|t|self.populations(dressed,&self.state(t),t,false)))
            .unwrap_or_else(|_|times.iter().map(|&t|self.populations(dressed,&self.state(t),t,false)).collect())
    }
    /// The laboratory time average at the horizon T (the centre block).
    pub fn horizon(&self,dressed:&Dressed,t:f64)->Vec<f64>{self.populations(dressed,&self.state(t),t,true)}
    pub fn steady_populations(&self,dressed:&Dressed)->Vec<f64>{self.populations(dressed,&self.steady,0.,true)}
    /// Basis coefficients X u of a reduced vector.
    pub fn coordinates(&self,u:&[f64])->Vec<f64>{let r=self.r;let k=self.x.len()/r.max(1);(0..k).map(|i|(0..r).map(|c|self.x[i*r+c]*u[c]).sum()).collect()}
    pub fn conserved(&self)->usize{self.kernel.iter().filter(|&&b|b).count().max(1)}
    /// The slowest relaxation rate (1/us) among the converged decaying Ritz
    /// values.
    pub fn slowest_rate(&self)->Option<f64>{
        (0..self.r).filter(|&j|!self.kernel[j]&&self.theta[j].re<0.&&self.residuals[j]<=RITZ_CONVERGED*(-self.theta[j].re))
            .map(|j|-self.theta[j].re).reduce(f64::min)
    }
    /// The error transform per pole xi (populations, n each): see the module
    /// documentation.
    pub fn error_transform(&self,dressed:&Dressed,eigen:&BlockEigenbasis,lapack:&Lapack,points:&[C])->Result<Vec<Vec<C>>,String>{
        let(n,d,r,l)=(dressed.n,dressed.d,self.r,dressed.length);let k=dressed.covered;let count=points.len();
        if count==0{return Ok(vec![]);}
        let gemm=lapack.complex_gemm();
        // G S (r x r complex), X (k x r real as complex).
        let gs:Vec<C>=(0..r*r).map(|e|{let(i,j)=(e/r,e%r);(0..r).fold(ZERO,|acc,c|{let mut acc=acc;mul_add(&mut acc,C{re:self.g[i*r+c],im:0.},self.vectors[c*r+j]);acc})}).collect();
        let modes:Vec<(usize,bool)>=(0..dressed.canonical.len()).flat_map(|c|{let mirrored=dressed.canonical[c].1>1.;
            std::iter::once((c,false)).chain(mirrored.then_some((c,true)))}).collect();
        let parts=Execution::new()?.map(modes,|(c,mirror)|->Result<Vec<C>,String>{
            let(_,w,nu)=dressed.canonical[c];let nu=if mirror{-nu}else{nu};let unscale=1./w.sqrt();
            // h = X S (a e / (xi - i nu - theta)), h' = X G S (...), k x count (column-major).
            let(mut h,mut h2)=(vec![ZERO;k*count],vec![ZERO;k*count]);
            for(p,&xi)in points.iter().enumerate(){
                let coefficient:Vec<C>=(0..r).map(|j|{let den=C{re:xi.re-self.used_theta[j].re,im:xi.im-nu-self.used_theta[j].im};
                    let s2=den.re*den.re+den.im*den.im;mul(self.used_coefficients[j],C{re:den.re/s2,im:-den.im/s2})}).collect();
                let(mut v,mut v2)=(vec![ZERO;r],vec![ZERO;r]);
                for i in 0..r{for j in 0..r{mul_add(&mut v[i],self.vectors[i*r+j],coefficient[j]);mul_add(&mut v2[i],gs[i*r+j],coefficient[j]);}}
                for i in 0..k{let(mut a,mut b)=(ZERO,ZERO);for col in 0..r{let xv=self.x[i*r+col];a.re+=xv*v[col].re;a.im+=xv*v[col].im;b.re+=xv*v2[col].re;b.im+=xv*v2[col].im;}
                    // A mirror mode's dressed block is the conjugate: conj(D) h = conj(D conj(h)).
                    h[i+k*p]=if mirror{conj(a)}else{a};h2[i+k*p]=if mirror{conj(b)}else{b};}
            }
            let mut y=vec![ZERO;d*count];
            gemm.multiply(d,count,k,ONE,&dressed.images[c*d..],l,&h,k,ZERO,&mut y,d)?;
            gemm.multiply(d,count,k,C{re:-1.,im:0.},&dressed.basis[c*d..],l,&h2,k,ONE,&mut y,d)?;
            let mut out=vec![ZERO;n*count];
            for(p,&xi)in points.iter().enumerate(){
                let column:Vec<C>=y[p*d..(p+1)*d].iter().map(|z|{let z=scaled(*z,unscale);if mirror{conj(z)}else{z}}).collect();
                let z=eigen.complex_coefficients(&column);
                for(jj,&zj)in z.iter().enumerate(){if !nonzero(zj){continue;}let den=C{re:xi.re-eigen.lambda[jj].re,im:xi.im-eigen.lambda[jj].im};
                    let s2=den.re*den.re+den.im*den.im;let f=mul(zj,C{re:den.re/s2,im:-den.im/s2});
                    for i in 0..n{mul_add(&mut out[i+n*p],eigen.readout[i+n*jj],f);}}
            }
            Ok(out)
        });
        let mut total=vec![vec![ZERO;n];count];
        for part in parts{let part=part?;for(p,row)in total.iter_mut().enumerate(){for(i,v)in row.iter_mut().enumerate(){add(v,part[i+n*p]);}}}
        Ok(total)
    }
}
