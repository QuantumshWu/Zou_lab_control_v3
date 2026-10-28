//! The reduced model of the space and every output from it.
//!
//! The projection: a Galerkin model in the inner product
//! Re <x, M y> of the Schrieffer-Wolff / Floquet frame's metric
//! M = T^H T (`frame::SwMetric`): T = V^-1 (I - S~) E^-1 removes the
//! first-order couplings between the modes and leaves the dissipative
//! second-order blocks, so the reduced numerical range is pushed right only
//! at second order (no strict guarantee: the provable bound is the
//! Hilbert-Schmidt one, Re theta <= Gamma/2). One model for every output
//! that propagates (the trajectory, the horizons); the Euclidean Galerkin
//! model (M = I) grew spurious near-zero slow pairs there (`frame`). A
//! request that propagates nothing (the steady state alone) takes M = I as
//! a speed case: it skips about 1 s of frame build, and its kernel does not
//! depend on the metric (the steady column is in the space). Append-only: per basis
//! column w_j only M w_j is kept (on the canonical modes), and the Gram
//! matrices K = Re W^H M W and H = Re (M W)^H A W grow by one column (and
//! row) each; G = K^-1 H in the coordinates of W (the basis is
//! Euclidean-orthonormal), x(t) ~ W u(t), u' = G u, u(0) = c0 = |x0| e_0
//! (w_0 = x0 / |x0|). The Euclidean Grams E = Re W^T A W and
//! F = Re (AW)^T AW give the residual's norm for the error transform.
//!
//! Trace correction: tr x_q evolves by -i nu_q tr x_q exactly; G is replaced
//! by the Frobenius-closest matrix with the trace rows of W as left
//! eigenvectors, on the significant rows only (above TRACE_ROWS of the
//! largest): the other rows are inner-solve roundoff, and fitting them
//! turns roundoff into O(1) changes of G.
//!
//! Ritz decomposition G S = S diag(theta), a = S^-1 c0. The kernel holds
//! |theta| < KERNEL and always the value of least modulus (the trace
//! correction pins one exact zero). Growing modes (Re theta > 0) are never
//! reflected: a mode of amplitude (its largest population contribution) up
//! to GROWTH_BUDGET x tolerance is projected out and charged; any other is
//! clipped to the imaginary axis, the minimal change of the residual
//! (|Re theta| |W y|), which the error estimate reads from the images.
//!
//! Outputs from the one G: the trajectory p(t) = sum_q Re e^{i nu_q t}
//! diag(W u(t))_q (the local time average); the state at a horizon T, its
//! centre block (the laboratory time average at T); the steady state from
//! the bordered small system (G + c0 tau^T) r = c0 (tau the centre trace
//! row; a simple kernel), else the kernel's spectral projection; the slow
//! Ritz values the slowest relaxation is read from (`pipeline` checks their
//! residuals against the images and the frame's classes) and the conserved
//! count.
//!
//! The error transform orders the poles: the Laplace transform of the
//! population error with the residual propagated through the dressed blocks,
//! p_err(xi) = sum_q W_d (xi - Lambda)^-1 Z_q (xi - i nu_q - Theta)^-1 1,
//! Z_q = W^-1 (R S diag a)_q; sigma |p_err(sigma + i omega)| is the mean
//! error near t = 1/sigma at frequency omega.
use super::*;
use super::eigenbasis::{BlockEigenbasis,conjugate_pairs};
use super::frame::SwMetric;
use super::krylov::Vectors;
use super::operator::LatticeOperator;

use crate::parallel::Execution;

/// Trace rows below this fraction of the largest are left unconstrained.
const TRACE_ROWS:f64=1e-6;
/// A direction of the trace rows is enforced while its share of the
/// correction, |u_j^T D| / s_j (D the defect T G0 - Omega T, s_j its
/// singular value), stays below this fraction of |G0| (the correction stays
/// a correction), and always while s_j is at least the population
/// tolerance times the largest (a trace the space carries above the
/// tolerance). Measured with the runs continued to 40 solves (five lasers
/// + tweezer at 20 us, five lasers without the trap at 10 ms, four lasers
/// without the trap at 20 us with a 1 s horizon): the determined directions
/// take shares of 1e-6 to 8.6e-3 (singular values down to 5e-4 of the
/// largest), the directions of inner-solve roundoff (at most 1.6e-5 of the
/// largest) 0.18 to 4e4; 3e-2 lies 3.5x above the first and 6x below the
/// second. A direction inverted at roundoff turned the defect into a
/// correction of up to 1e6 |G0|: spurious growing modes (Re theta up to
/// 3e4 /us) once the space reached new modes' trace rows.
const TRACE_CORRECTION:f64=3e-2;
/// Singular values below this fraction of the largest are roundoff.
const TRACE_ROUNDOFF:f64=1e-14;
/// |theta| below this (1/us) is a conserved quantity.
pub const KERNEL:f64=1e-9;
/// The largest exponent of a growth factor e^{theta T} (e^700 ~ 1e304,
/// below the double range's 1.8e308: the factor stays finite).
const EXPONENT_LIMIT:f64=700.;
/// Growth (e^{Re theta T} - 1) x amplitude below this is clipped (not
/// counted as growing): a deliberate margin, 2e-4 of the tolerance.
const GROWTH_NEGLIGIBLE:f64=1e-6;
/// Amplitude (times the tolerance) a growing mode may have to be projected
/// out (its amplitude is charged to every output). Measured: projecting
/// regardless of amplitude removed a three-laser slow mode of amplitude
/// 8.9e-2 (trajectory off by 9e-2); a tenth of the tolerance is a margin.
const GROWTH_BUDGET:f64=0.1;
/// A slow Ritz pair is resolved when its residual |A W y - theta W y| / |W y|
/// is below this share of its rate (and no slower class is missing,
/// `SlowRate`): a deliberate margin, the relaxation time within about 10%.
pub const SLOW_ACCURACY:f64=0.1;
/// Residual components of the error transform: at most TRANSFORM_RANK,
/// down to TRANSFORM_COMPONENT of the largest singular value. Margins: the
/// residual's rank in exact arithmetic is at most 2 + 2 per translation pair
/// (14 for six pairs), below TRANSFORM_RANK; the transform only ranks the
/// poles, so components below 1e-3 of the largest are left out.
const TRANSFORM_RANK:usize=24;
const TRANSFORM_COMPONENT:f64=1e-3;

/// Row-major (r x k) (k x c), the column-major (A B)^T = B^T A^T.
pub(crate) fn product(lapack:&Lapack,a:&[f64],b:&[f64],r:usize,k:usize,c:usize)->Result<Vec<f64>,String>{
    let mut out=vec![0.;r*c];if r*k*c>0{lapack.gemm(false,false,c,r,k,1.,b,c,a,k,0.,&mut out,c)?;}Ok(out)
}
pub(crate) fn transpose_rows(a:&[f64],r:usize,c:usize)->Vec<f64>{let mut t=vec![0.;r*c];for i in 0..r{for j in 0..c{t[j*r+i]=a[i*c+j];}}t}

/// The metric columns M W of the space and the Gram matrices.
pub struct Projection{
    n:usize,d:usize,
    /// Canonical modes: (mode, weight, nu); the centre's position.
    pub canonical:Vec<(usize,f64,f64)>,centre:usize,
    /// M w_j on the canonical modes, weighted (`SwMetric::apply`),
    /// column-major (length L = canonical x d per column).
    metric:Vec<C>,length:usize,
    /// Per column: the canonical modes' diagonals (canonical x n) and the
    /// trace functionals [centre, then Re and Im per other canonical mode].
    populations:Vec<Vec<C>>,traces:Vec<Vec<f64>>,
    /// K = Re W^H M W, H = Re (M W)^H A W, E = Re W^T A W and
    /// F = Re (AW)^T A W, row-major over the covered columns; whether
    /// M = I (no metric: G0 = E, K and H are not built).
    k:Vec<f64>,h:Vec<f64>,e:Vec<f64>,f:Vec<f64>,covered:usize,euclidean:bool,
    vectors:Vectors,
    pub milliseconds:f64,
}
/// Grows a row-major old x old matrix to new x new (zeros in the new part).
fn grow(old_matrix:&[f64],old:usize,new:usize)->Vec<f64>{let mut g=vec![0.;new*new];for i in 0..old{for j in 0..old{g[i*new+j]=old_matrix[i*old+j];}}g}
impl Projection{
    /// `euclidean`: M = I (a request that propagates nothing).
    pub fn new(op:&LatticeOperator<'_>,euclidean:bool)->Result<Self,String>{
        let n=op.n;let d=n*n;
        let canonical:Vec<(usize,f64,f64)>=(0..op.modes.len()).filter(|&q|op.canonical[q]).map(|q|(q,if op.mirror[q]==q{1.}else{2.},op.nu[q])).collect();
        let centre=canonical.iter().position(|c|c.0==op.centre).unwrap_or(0);
        let length=canonical.len()*d;
        Ok(Self{n,d,canonical,centre,metric:vec![],length,populations:vec![],traces:vec![],k:vec![],h:vec![],e:vec![],f:vec![],covered:0,euclidean,
            vectors:Vectors::new(op,1)?,milliseconds:0.})
    }
    pub fn columns(&self)->usize{self.covered}
    /// Re <z_a, field_c> on the canonical modes for metric columns z
    /// (column-major, `count` of them) and fields (row-major |fields| x
    /// count): one task per field, a fixed order of summation.
    fn metric_dots(&self,z:&[C],count:usize,fields:&[&[C]])->Result<Vec<f64>,String>{
        let(d,l)=(self.d,self.length);
        let rows=Execution::new()?.map(fields.to_vec(),|field|{
            (0..count).map(|a|{let column=&z[a*l..(a+1)*l];let mut s=0.;
                for(c,&(q,_,_))in self.canonical.iter().enumerate(){for(x,y)in column[c*d..(c+1)*d].iter().zip(&field[q*d..(q+1)*d]){s+=x.re*y.re+x.im*y.im;}}s}).collect::<Vec<f64>>()});
        Ok(rows.concat())
    }
    /// Takes the space's columns not yet covered; `metric` None exactly when
    /// the projection is Euclidean.
    pub fn ingest(&mut self,metric:Option<&SwMetric<'_>>,basis:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        let clock=Instant::now();let(old,new)=(self.covered,basis.len());if new==old{return Ok(());}
        let(n,d,l)=(self.n,self.d,self.length);let width=new-old;
        if self.euclidean!=metric.is_none(){return Err("Galerkin projection: the metric differs from the projection's".into());}
        if let Some(metric)=metric{let fresh:Vec<&[C]>=basis[old..new].iter().map(|v|v.as_slice()).collect();self.metric.extend(metric.apply(&fresh)?);}
        for column in &basis[old..new]{
            let pops:Vec<C>=self.canonical.iter().flat_map(|&(q,_,_)|(0..n).map(move|i|column[q*d+i*n+i])).collect();
            let mut traces=vec![0.;2*self.canonical.len()-1];let mut at=1;
            for(c,_)in self.canonical.iter().enumerate(){let t=(0..n).fold(ZERO,|s,i|{let v=pops[c*n+i];C{re:s.re+v.re,im:s.im+v.im}});
                if c==self.centre{traces[0]=t.re;}else{traces[at]=t.re;traces[at+1]=t.im;at+=2;}}
            self.populations.push(pops);self.traces.push(traces);
        }
        let all_basis:Vec<&[C]>=basis[..new].iter().map(|v|v.as_slice()).collect();
        let all_images:Vec<&[C]>=images[..new].iter().map(|v|v.as_slice()).collect();
        let new_images=&all_images[old..];
        let(kb,hc,hr)=if self.euclidean{(vec![],vec![],vec![])}else{
            let z_new=&self.metric[old*l..new*l];
            (self.metric_dots(z_new,width,&all_basis)?,                     // K[i, old + c] at i * width + c
             self.metric_dots(&self.metric,new,new_images)?,               // H[i, old + c] at c * new + i
             self.metric_dots(z_new,width,&all_images[..old])?)};          // H[old + c, j] at j * width + c
        let ec=self.vectors.gram(&all_basis,new_images)?;                 // E[:, old..new]
        let er=self.vectors.gram(&all_basis[old..],&all_images[..old])?;  // E[old..new, ..old]
        let fc=self.vectors.gram(&all_images,new_images)?;                // F[:, old..new]
        let(mut k,mut h)=if self.euclidean{(Vec::new(),Vec::new())}else{(grow(&self.k,old,new),grow(&self.h,old,new))};
        let(mut e,mut f)=(grow(&self.e,old,new),grow(&self.f,old,new));
        for i in 0..new{for c in 0..width{let j=old+c;
            if !self.euclidean{k[i*new+j]=kb[i*width+c];k[j*new+i]=kb[i*width+c];h[i*new+j]=hc[c*new+i];}
            e[i*new+j]=ec[i*width+c];
            f[i*new+j]=fc[i*width+c];f[j*new+i]=fc[i*width+c];}}
        for c in 0..width{for j in 0..old{if !self.euclidean{h[(old+c)*new+j]=hr[j*width+c];}e[(old+c)*new+j]=er[c*old+j];}}
        self.k=k;self.h=h;self.e=e;self.f=f;self.covered=new;
        self.milliseconds+=clock.elapsed().as_secs_f64()*1e3;Ok(())
    }
}

/// The reduced model of one checkpoint.
pub struct Model{
    /// Reduced dimension (the basis' columns: W is orthonormal), G (r x r).
    pub r:usize,pub g:Vec<f64>,
    /// Ritz values, eigenvectors (r x r row-major, complex), coefficients,
    /// and the values and coefficients the outputs use; per mode the kernel
    /// and a growing mode projected out (used theta = a = 0).
    pub theta:Vec<C>,pub vectors:Vec<C>,pub coefficients:Vec<C>,
    pub used_theta:Vec<C>,pub used_coefficients:Vec<C>,pub kernel:Vec<bool>,pub projected_out:Vec<bool>,
    /// Mode amplitudes (largest population contribution).
    pub amplitudes:Vec<f64>,
    pub growing:usize,pub projected:usize,pub clipped:usize,pub growth_charge:f64,
    /// The reduced steady state (kernel part of c0).
    pub steady:Vec<f64>,
    /// Per canonical mode: populations of W (n x r, row-major complex).
    readout:Vec<Vec<C>>,
}

impl Projection{
    /// The reduced model over the covered columns; `initial_norm` = |x0|,
    /// `longest` the longest output time (growth), `tolerance` the
    /// population tolerance.
    pub fn model(&self,lapack:&Lapack,initial_norm:f64,longest:f64,tolerance:f64)->Result<Model,String>{
        let k=self.covered;let n=self.n;let r=k;
        // G0 = K^-1 H (K symmetric: its row-major form is its column-major one).
        let g0=if self.euclidean{self.e.clone()}else{let(lu,pivots)=lapack.factor(&self.k,k)?;
            transpose_rows(&lapack.solve_matrix_factored(&lu,&pivots,&transpose_rows(&self.h,k,k),k)?,k,k)};
        let mut c0=vec![0.;k];c0[0]=initial_norm;
        // Trace correction on the significant rows.
        let kt=self.traces[0].len();
        let tw:Vec<f64>=(0..kt).flat_map(|t|(0..k).map(move|j|(t,j))).map(|(t,j)|self.traces[j][t]).collect();
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
            // Per direction j of the rows (T T^T u_j = s_j^2 u_j) its share of
            // the correction, T^T u_j (u_j^T D) / s_j^2, of Frobenius norm
            // |u_j^T D| / s_j: kept while that stays below TRACE_CORRECTION |G0|
            // or while its trace content s_j is at least the tolerance times
            // the largest (a space too small to conserve a trace it carries
            // needs an O(1) correction: two columns of no laser, share 1).
            let size=|m:&[f64]|m.iter().map(|v|v*v).sum::<f64>().sqrt();let g0_size=size(&g0);
            let singular:Vec<f64>=values.iter().map(|v|v.max(0.).sqrt()).collect();let largest_singular=top.max(0.).sqrt();
            let share:Vec<f64>=(0..rows).map(|j|{let mut p=vec![ZERO;r];for a in 0..rows{let u=conj(vectors[a+rows*j]);for c in 0..r{mul_add(&mut p[c],u,C{re:defect[a*r+c],im:0.});}}
                p.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()/singular[j].max(f64::MIN_POSITIVE)}).collect();
            let keep:Vec<bool>=(0..rows).map(|j|singular[j]>TRACE_ROUNDOFF*largest_singular&&(share[j]<=TRACE_CORRECTION*g0_size||singular[j]>=tolerance*largest_singular)).collect();
            let mut pseudo=vec![0.;rows*rows];
            for j in 0..rows{if keep[j]{for a in 0..rows{for b in 0..rows{pseudo[a*rows+b]+=mul(vectors[a+rows*j],conj(vectors[b+rows*j])).re/values[j];}}}}
            let correction=product(lapack,&product(lapack,&transpose_rows(&t,rows,r),&pseudo,r,rows,rows)?,&defect,r,rows,r)?;
            for(a,b)in g.iter_mut().zip(&correction){*a-=b;}
        }
        // Ritz decomposition.
        let(re,im,ev)=lapack.eigen(&transpose_rows(&g,r,r),r)?;
        let(pair,theta)=conjugate_pairs(&re,&im);let mut s=vec![ZERO;r*r];
        for j in 0..r{match pair[j]{
            1=>for i in 0..r{let(a,b)=(ev[i+r*j],ev[i+r*(j+1)]);s[i*r+j]=C{re:a,im:b};s[i*r+j+1]=C{re:a,im:-b};},
            0=>for i in 0..r{s[i*r+j]=C{re:ev[i+r*j],im:0.};},
            _=>{}}}
        let(lu,pivots)=lapack.factor(&ev,r)?;let y=lapack.solve_matrix_factored(&lu,&pivots,&c0,r)?;
        let mut a=vec![ZERO;r];let mut j=0;
        while j<r{if pair[j]==1{a[j]=C{re:0.5*y[j],im:-0.5*y[j+1]};a[j+1]=C{re:0.5*y[j],im:0.5*y[j+1]};j+=2;}else{a[j]=C{re:y[j],im:0.};j+=1;}}
        // Readout of W per canonical mode, and the modes' amplitudes.
        let readout:Vec<Vec<C>>=(0..self.canonical.len()).map(|c|{
            let mut out=vec![ZERO;n*r];for i in 0..n{for jj in 0..k{out[i*r+jj]=self.populations[jj][c*n+i];}}out}).collect();
        let amplitudes:Vec<f64>=(0..r).map(|jj|{let mut largest=0_f64;
            for block in &readout{for i in 0..n{let mut v=ZERO;for col in 0..r{mul_add(&mut v,block[i*r+col],s[col*r+jj]);}largest=largest.max(magnitude(v));}}
            magnitude(a[jj])*largest}).collect();
        // Kernel and growing modes.
        let least=(0..r).min_by(|&p,&q|magnitude(theta[p]).total_cmp(&magnitude(theta[q]))).ok_or("Empty reduced model")?;
        let kernel:Vec<bool>=(0..r).map(|jj|jj==least||magnitude(theta[jj])<KERNEL).collect();
        let(mut used_theta,mut used_coefficients)=(theta.clone(),a.clone());
        let(mut growing,mut projected,mut clipped,mut growth_charge)=(0,0,0,0.);let mut projected_out=vec![false;r];
        for jj in 0..r{
            if kernel[jj]{used_theta[jj]=ZERO;continue;}
            if !(theta[jj].re>0.){continue;}
            let growth=((theta[jj].re*longest).min(EXPONENT_LIMIT).exp()-1.)*amplitudes[jj];
            if growth<=GROWTH_NEGLIGIBLE{used_theta[jj].re=0.;continue;}
            growing+=1;
            if amplitudes[jj]<=GROWTH_BUDGET*tolerance{used_theta[jj]=ZERO;used_coefficients[jj]=ZERO;projected_out[jj]=true;projected+=1;growth_charge+=amplitudes[jj];}
            else{used_theta[jj].re=0.;clipped+=1;}
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
        Ok(Model{r,g,theta,vectors:s,coefficients:a,used_theta,used_coefficients,kernel,projected_out,amplitudes,
            growing,projected,clipped,growth_charge,steady,readout})
    }
}

impl Model{
    /// u(t) = Re S diag(e^{used theta t}) used coefficients.
    pub fn state(&self,t:f64)->Vec<f64>{
        let r=self.r;let e:Vec<C>=(0..r).map(|j|mul(cexp(self.used_theta[j],t),self.used_coefficients[j])).collect();
        (0..r).map(|i|(0..r).fold(0.,|acc,j|acc+mul(self.vectors[i*r+j],e[j]).re)).collect()
    }
    /// Populations of the reduced vector u at time t (lab phases), or on the
    /// centre alone (`centre`: the laboratory time average).
    fn populations(&self,projection:&Projection,u:&[f64],t:f64,centre:bool)->Vec<f64>{
        let(n,r)=(projection.n,self.r);let mut p=vec![0.;n];
        for(c,&(_,w,nu))in projection.canonical.iter().enumerate(){
            if centre&&c!=projection.centre{continue;}let phase=cexp(C{re:0.,im:nu},t);let block=&self.readout[c];
            for(i,pi)in p.iter_mut().enumerate(){let mut v=ZERO;for(col,&uc)in u.iter().enumerate(){let b=block[i*r+col];v.re+=b.re*uc;v.im+=b.im*uc;}
                *pi+=if centre{v.re}else{w*mul(phase,v).re};}
        }
        p
    }
    /// The trajectory's populations on the grid.
    pub fn trajectory(&self,projection:&Projection,times:&[f64])->Vec<Vec<f64>>{
        Execution::new().map(|e|e.map(times.to_vec(),|t|self.populations(projection,&self.state(t),t,false)))
            .unwrap_or_else(|_|times.iter().map(|&t|self.populations(projection,&self.state(t),t,false)).collect())
    }
    /// The laboratory time average at the horizon T (the centre block).
    pub fn horizon(&self,projection:&Projection,t:f64)->Vec<f64>{self.populations(projection,&self.state(t),t,true)}
    pub fn steady_populations(&self,projection:&Projection)->Vec<f64>{self.populations(projection,&self.steady,0.,true)}
    pub fn conserved(&self)->usize{self.kernel.iter().filter(|&&b|b).count().max(1)}
    /// The slowest relaxation's candidates: the decaying
    /// Ritz values outside the kernel (rate at least KERNEL: slower ones are
    /// undamped precession; growing ones are projected out or clipped) of
    /// population weight (`amplitudes`) at least `weight`, one of each
    /// conjugate pair, slowest first: (index, rate).
    pub fn slow_candidates(&self,weight:f64)->Vec<(usize,f64)>{
        let mut out:Vec<(usize,f64)>=(0..self.r).filter(|&j|!self.kernel[j]&&-self.theta[j].re>=KERNEL&&self.theta[j].im>=0.&&self.amplitudes[j]>=weight)
            .map(|j|(j,-self.theta[j].re)).collect();
        out.sort_by(|a,b|a.1.total_cmp(&b.1));out
    }
    /// Basis coefficients s_j of a Ritz vector (complex).
    pub fn ritz_coordinates(&self,j:usize)->Vec<C>{let r=self.r;(0..r).map(|i|self.vectors[i*r+j]).collect()}
    /// The error transform per pole xi (populations, n each): see the module
    /// documentation. The residual R = A W - W G enters through its leading
    /// singular components (TRANSFORM_RANK, down to TRANSFORM_COMPONENT of
    /// the largest): R = sum_i u_i v_i^T with u_i = R v_i, from the
    /// eigenvectors v_i of R^T R = F - E^T G - G^T E + G^T G (W^T W = I),
    /// each u_i in the dressed coordinates.
    #[allow(clippy::too_many_arguments)]
    pub fn error_transform(&self,projection:&Projection,basis:&[Vec<C>],images:&[Vec<C>],eigen:&BlockEigenbasis,lapack:&Lapack,points:&[C])->Result<Vec<Vec<C>>,String>{
        let(n,d,r)=(projection.n,projection.d,self.r);let k=projection.covered;let count=points.len();
        if count==0{return Ok(vec![]);}
        let gemm=lapack.complex_gemm();
        // The residual's leading components.
        let(et,gt)=(transpose_rows(&projection.e,k,k),transpose_rows(&self.g,k,k));
        let(a1,a2,a3)=(product(lapack,&et,&self.g,k,k,k)?,product(lapack,&gt,&projection.e,k,k,k)?,product(lapack,&gt,&self.g,k,k,k)?);
        let gram:Vec<C>=(0..k*k).map(|e|C{re:projection.f[e]-a1[e]-a2[e]+a3[e],im:0.}).collect();
        let(values,vectors)=lapack.hermitian_eigen(&gram,r)?;
        let top=values.iter().fold(0_f64,|a,&v|a.max(v));
        let chosen:Vec<usize>=(0..r).rev().filter(|&j|values[j]>TRANSFORM_COMPONENT*TRANSFORM_COMPONENT*top).take(TRANSFORM_RANK).collect();
        let p=chosen.len();if p==0{return Ok(vec![vec![ZERO;n];count]);}
        // Real singular vectors (up to a phase per column), V (r x p).
        let v:Vec<f64>={let mut v=vec![0.;r*p];for(c,&j)in chosen.iter().enumerate(){let col=&vectors[r*j..r*(j+1)];
            let pivot=col.iter().copied().max_by(|a,b|magnitude(*a).total_cmp(&magnitude(*b))).unwrap_or(ONE);let s=magnitude(pivot).max(f64::MIN_POSITIVE);
            let phase=C{re:pivot.re/s,im:-pivot.im/s};for i in 0..r{v[i*p+c]=mul(col[i],phase).re;}}v};
        // u_i = A W v_i - W G v_i, then its dressed coordinates on every canonical mode (L x p).
        let gv=product(lapack,&self.g,&v,r,r,p)?;
        let fields:Vec<&[C]>=images[..k].iter().chain(&basis[..k]).map(|f|f.as_slice()).collect();
        let coefficients:Vec<Vec<f64>>=(0..p).map(|i|(0..k).map(|j|v[j*p+i]).chain((0..k).map(|j|-gv[j*p+i])).collect()).collect();
        let residuals=projection.vectors.combinations(&fields,&coefficients)?;
        let blocks:Vec<C>=projection.canonical.iter().flat_map(|&(q,_,_)|residuals.iter().flat_map(move|u|u[q*d..(q+1)*d].iter().copied())).collect();
        let u=eigen.dressed(&blocks)?;drop(residuals);
        // V^T S diag(a) (p x r, complex).
        let vs:Vec<C>=(0..p*r).map(|e|{let(i,j)=(e/r,e%r);let mut s=ZERO;for m in 0..r{mul_add(&mut s,C{re:v[m*p+i],im:0.},self.vectors[m*r+j]);}mul(s,self.used_coefficients[j])}).collect();
        let modes:Vec<(usize,bool)>=(0..projection.canonical.len()).flat_map(|c|{let mirrored=projection.canonical[c].1>1.;
            std::iter::once((c,false)).chain(mirrored.then_some((c,true)))}).collect();
        let parts=Execution::new()?.map(modes,|(c,mirror)|->Result<Vec<C>,String>{
            let(_,_,nu)=projection.canonical[c];let nu=if mirror{-nu}else{nu};
            // Coefficients of the components per point: (V^T S diag(a)) / (xi - i nu - theta), p x count.
            let mut cm=vec![ZERO;p*count];
            for(q,&xi)in points.iter().enumerate(){
                let f:Vec<C>=(0..r).map(|j|{let den=C{re:xi.re-self.used_theta[j].re,im:xi.im-nu-self.used_theta[j].im};let s2=den.re*den.re+den.im*den.im;C{re:den.re/s2,im:-den.im/s2}}).collect();
                for i in 0..p{let mut s=ZERO;for j in 0..r{mul_add(&mut s,vs[i*r+j],f[j]);}
                    // A mirror mode's dressed block is the conjugate: conj(U) c = conj(U conj(c)).
                    cm[i+p*q]=if mirror{conj(s)}else{s};}
            }
            let mut y=vec![ZERO;d*count];gemm.multiply(d,count,p,ONE,&u[c*d*p..],d,&cm,p,ZERO,&mut y,d)?;
            // Complex eigen-coefficients scaled by 1/(xi - lambda), then the readout (n x d) by GEMM.
            let mut z=vec![ZERO;d*count];
            for(q,&xi)in points.iter().enumerate(){
                let column:Vec<C>=y[q*d..(q+1)*d].iter().map(|e|if mirror{conj(*e)}else{*e}).collect();
                for(jj,zj)in eigen.complex_coefficients(&column).into_iter().enumerate(){
                    let den=C{re:xi.re-eigen.lambda[jj].re,im:xi.im-eigen.lambda[jj].im};let s2=den.re*den.re+den.im*den.im;
                    z[jj+d*q]=mul(zj,C{re:den.re/s2,im:-den.im/s2});}
            }
            let mut out=vec![ZERO;n*count];gemm.multiply(n,count,d,ONE,&eigen.readout,n,&z,d,ZERO,&mut out,n)?;
            Ok(out)
        });
        let mut total=vec![vec![ZERO;n];count];
        for part in parts{let part=part?;for(q,row)in total.iter_mut().enumerate(){for(i,v)in row.iter_mut().enumerate(){add(v,part[i+n*q]);}}}
        Ok(total)
    }
}

/// A slow relaxation rate (1/us), its residual (rad/us) and whether every
/// class of the frame's second-order block slower than it is matched by a
/// Ritz pair (`pipeline`; None: not checked): resolved when it is and the
/// residual is
/// within SLOW_ACCURACY of the rate, else only the range rate +- residual
/// (a slower class the space has not reached included).
#[derive(Clone,Copy,Debug)]
pub struct SlowRate{pub rate:f64,pub residual:f64,pub matched:Option<bool>}
impl SlowRate{
    pub fn resolved(&self)->bool{self.matched==Some(true)&&self.residual<=SLOW_ACCURACY*self.rate}
}
