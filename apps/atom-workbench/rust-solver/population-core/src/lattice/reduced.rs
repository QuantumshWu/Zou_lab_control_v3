//! The reduced model of the space and every output from it.
//!
//! The projection: a metric Petrov-Galerkin model. The trial space is the
//! basis W (Euclidean-orthonormal, its first column the trace carrier w_0 =
//! u / |u| and every other in K: `space`), the test space Z = M (A - tau) W
//! with M = T^H T the Schrieffer-Wolff / Floquet frame's metric
//! (`frame::SwMetric`: T = V^-1 (I - S~) E^-1 removes the first-order
//! couplings between the modes and leaves the dissipative second-order
//! blocks) and tau = LADDER_TOP / t1, the ladder's top (`space`): the
//! residual A W u - W u' is made M-orthogonal to (A - tau) W, which weighs
//! the fast part of the residual (|A w| large) as the shift-and-invert rule
//! weighs the earliest output time. x(t) ~ W u(t) with u_0 fixed at |u| (the
//! trace, exactly) and the K coordinates by the condition on the K columns
//! of Z alone: with S = Z^T W and R = Z^T A W, the rows 1.. of u' = G u are
//! S'^-1 R[1.., ..] (S' = S[1.., 1..]) and G's first row is zero (A W lies
//! in K, so no image leaks into the carrier); u(0) = (|u|, c), S' c = Z^T x0
//! [1..] - S[1.., 0] |u|. Every block trace is conserved by construction: no
//! trace correction.
//!
//! Only three Gram matrices are formed: K = Re W^H M W, H = Re (M W)^H A W
//! and Fm = Re (A W)^H M (A W), so S = H^T - tau K and R = Fm - tau H. They
//! grow by one column (and row) per basis column; no image A w and no M w is
//! stored: a new column's image is applied once, and its rows against the
//! old columns come from the adjoint apply, Re <z, A w_j> = Re <A^H z, w_j>
//! (`LatticeOperator::apply_adjoint`). Z^T x0 = W^T (A^H - tau) M x0 takes
//! one adjoint application for the whole run.
//!
//! Ritz decomposition G S = S diag(theta), a = S^-1 u(0). G's zero first row
//! makes 0 an eigenvalue, its eigenvector the one carrying the trace (its
//! carrier coordinate nonzero, every other eigenvector's zero): that mode is
//! the kernel, with every further |theta| < KERNEL (more conserved
//! quantities). Growing modes (Re theta > 0) are never reflected: a mode of
//! amplitude (its largest population contribution) up to GROWTH_BUDGET x
//! tolerance is projected out and charged to every output; any other is
//! clipped to the imaginary axis.
//!
//! Outputs from the one G: the trajectory p(t) = sum_q Re e^{i nu_q t}
//! diag(W u(t))_q (the local time average); the state at a horizon T, its
//! centre block (the laboratory time average at T); the slow Ritz values the
//! slowest relaxation is read from (`pipeline` checks their residuals and the
//! frame's classes).
use super::*;
use super::eigenbasis::conjugate_pairs;
use super::frame::SwMetric;
use super::krylov::Vectors;
use super::poles::Lattice;

use crate::parallel::Execution;

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

pub(crate) fn transpose_rows(a:&[f64],r:usize,c:usize)->Vec<f64>{let mut t=vec![0.;r*c];for i in 0..r{for j in 0..c{t[j*r+i]=a[i*c+j];}}t}

/// The Gram matrices of the space.
pub struct Projection<'m>{
    n:usize,d:usize,
    /// Canonical modes: (mode, weight, nu); the centre's position.
    pub canonical:Vec<(usize,f64,f64)>,centre:usize,
    /// Length of a metric column (canonical x d).
    length:usize,
    /// Per column: the canonical modes' diagonals (canonical x n).
    populations:Vec<Vec<C>>,
    /// K = Re W^H M W, H = Re (M W)^H A W and Fm = Re (A W)^H M (A W),
    /// row-major over the covered columns.
    k:Vec<f64>,h:Vec<f64>,fm:Vec<f64>,covered:usize,
    /// The Petrov-Galerkin shift tau and Z^T x0 per covered column.
    tau:f64,zx:Vec<f64>,
    /// (A^H - tau) M x0 as the dot products take it: A^H M x0 (a field) and
    /// M x0 (its weighted canonical blocks).
    adjoint_x0:Vec<C>,metric_x0:Vec<C>,
    frame_metric:&'m SwMetric<'m>,
    vectors:Vectors,
    pub milliseconds:f64,
}
/// Grows a row-major old x old matrix to new x new (zeros in the new part).
fn grow(old_matrix:&[f64],old:usize,new:usize)->Vec<f64>{let mut g=vec![0.;new*new];for i in 0..old{for j in 0..old{g[i*new+j]=old_matrix[i*old+j];}}g}
impl<'m> Projection<'m>{
    /// The projection of the lattice's fields with the shift tau, from x0 (a
    /// field on the lattice).
    pub fn new(lattice:&Lattice<'_>,metric:&'m SwMetric<'m>,tau:f64,x0:&[C])->Result<Self,String>{
        let op=lattice.operator();let n=op.n;let d=n*n;
        let canonical:Vec<(usize,f64,f64)>=(0..op.modes.len()).filter(|&q|op.canonical[q]).map(|q|(q,if op.mirror[q]==q{1.}else{2.},op.nu[q])).collect();
        let centre=canonical.iter().position(|c|c.0==op.centre).unwrap_or(0);
        let length=canonical.len()*d;
        let mut projection=Self{n,d,canonical,centre,length,populations:vec![],k:vec![],h:vec![],fm:vec![],covered:0,tau,zx:vec![],
            adjoint_x0:vec![],metric_x0:vec![],frame_metric:metric,vectors:Vectors::new(op)?,milliseconds:0.};
        projection.metric_x0=metric.apply(&[x0])?;
        let field=projection.field(&projection.metric_x0,0,lattice);projection.adjoint_x0=op.apply_adjoint(&field)?;
        Ok(projection)
    }
    pub fn columns(&self)->usize{self.covered}
    /// Metric column c of `z` (weighted canonical blocks) as a
    /// Hermitian-symmetric field (unweighted, mirrored).
    fn field(&self,z:&[C],c:usize,lattice:&Lattice<'_>)->Vec<C>{
        let op=lattice.operator();let(n,d,l)=(self.n,self.d,self.length);let mut f=vec![ZERO;op.dimension()];
        for(k,&(q,w,_))in self.canonical.iter().enumerate(){for e in 0..d{f[q*d+e]=scaled(z[c*l+k*d+e],1./w);}}
        mirror_field(&mut f,n,&op.canonical,&op.mirror);f
    }
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
    /// Takes the space's columns not yet covered: their images from the
    /// lattice, K, H and Fm's new columns against every column and the new
    /// rows of H against the old columns by the adjoint apply, Z^T x0's new
    /// entries.
    pub fn ingest(&mut self,basis:&[Vec<C>],lattice:&mut Lattice<'_>)->Result<(),String>{
        let clock=Instant::now();let(old,new)=(self.covered,basis.len());if new==old{return Ok(());}
        let(n,d)=(self.n,self.d);let width=new-old;
        let fresh:Vec<&[C]>=basis[old..new].iter().map(|v|v.as_slice()).collect();
        let images:Vec<Vec<C>>=fresh.iter().map(|w|{let mut a=vec![ZERO;w.len()];lattice.apply(w,&mut a).map(|_|a)}).collect::<Result<_,_>>()?;
        let new_images:Vec<&[C]>=images.iter().map(|v|v.as_slice()).collect();
        for column in &basis[old..new]{self.populations.push(self.canonical.iter().flat_map(|&(q,_,_)|(0..n).map(move|i|column[q*d+i*n+i])).collect());}
        let all_basis:Vec<&[C]>=basis[..new].iter().map(|v|v.as_slice()).collect();
        let z_new=self.frame_metric.apply(&fresh)?;let za=self.frame_metric.apply(&new_images)?;
        // M w_c and M A w_c as fields, their adjoint applications over the
        // workers, then one Gram of all four against the basis (one sweep of
        // the basis, one GEMM per canonical block): K[i, old + c] =
        // Re <M w_c, w_i>, H[i, old + c] = Re <M A w_c, w_i>, and on the old
        // columns H[old + c, j] = Re <A^H M w_c, w_j>, Fm[old + c, j] =
        // Re <A^H M A w_c, w_j>.
        let lattice_ref=&*lattice;
        let mut rows:Vec<Vec<C>>=(0..width).flat_map(|c|[self.field(&z_new,c,lattice_ref),self.field(&za,c,lattice_ref)]).collect();
        let op=lattice.operator();
        let adjoints=Execution::new()?.map((0..rows.len()).collect(),|i|op.apply_adjoint(&rows[i])).into_iter().collect::<Result<Vec<_>,_>>()?;
        rows.extend(adjoints);
        let refs:Vec<&[C]>=rows.iter().map(|v|v.as_slice()).collect();let g=self.vectors.gram(&refs,&all_basis)?;drop(refs);drop(rows);
        let at=|row:usize,j:usize|g[row*new+j];
        let kb:Vec<f64>=(0..new*width).map(|x|at(2*(x%width),x/width)).collect();
        let hc:Vec<f64>=(0..new*width).map(|x|at(2*(x%width)+1,x/width)).collect();
        let mut hr=vec![0.;old*width];let mut fmc=vec![0.;new*width];
        for c in 0..width{for j in 0..old{hr[j*width+c]=at(2*width+2*c,j);fmc[j*width+c]=at(2*width+2*c+1,j);}}
        let fmnew=self.metric_dots(&za,width,&new_images)?;for a in 0..width{for c in 0..width{fmc[(old+a)*width+c]=fmnew[a*width+c];}}
        let(mut k,mut h,mut fm)=(grow(&self.k,old,new),grow(&self.h,old,new),grow(&self.fm,old,new));
        for i in 0..new{for c in 0..width{let j=old+c;
            k[i*new+j]=kb[i*width+c];k[j*new+i]=kb[i*width+c];h[i*new+j]=hc[i*width+c];
            fm[i*new+j]=fmc[i*width+c];fm[j*new+i]=fmc[i*width+c];}}
        for c in 0..width{for j in 0..old{h[(old+c)*new+j]=hr[j*width+c];}}
        // Z^T x0 = Re <w, A^H M x0> - tau Re <w, M x0>.
        let one=self.vectors.dots(&fresh,&self.adjoint_x0);let two=self.metric_dots(&self.metric_x0,1,&fresh)?;
        self.zx.extend(one.iter().zip(&two).map(|(a,b)|a-self.tau*b));
        self.k=k;self.h=h;self.fm=fm;self.covered=new;
        self.milliseconds+=clock.elapsed().as_secs_f64()*1e3;Ok(())
    }
}

/// The reduced model of one checkpoint.
pub struct Model{
    /// Reduced dimension (the basis' columns: W is orthonormal).
    pub r:usize,
    /// Ritz values and eigenvectors (r x r row-major, complex), the values
    /// and coefficients the outputs use; per mode the kernel.
    pub theta:Vec<C>,pub vectors:Vec<C>,
    pub used_theta:Vec<C>,pub used_coefficients:Vec<C>,pub kernel:Vec<bool>,
    /// Mode amplitudes: the largest population contribution on any mode.
    pub amplitudes:Vec<f64>,
    pub growing:usize,pub projected:usize,pub clipped:usize,pub growth_charge:f64,
    /// Per canonical mode: populations of W (n x r, row-major complex).
    readout:Vec<Vec<C>>,
}

impl<'m> Projection<'m>{
    /// The reduced model over the covered columns; `initial` x0's leading
    /// coordinates (`Space::initial`: the carrier's |u| fixed), `longest` the
    /// longest output time (growth), `tolerance` the population tolerance.
    pub fn model(&self,lapack:&Lapack,initial:&[f64],longest:f64,tolerance:f64)->Result<Model,String>{
        let k=self.covered;let n=self.n;let r=k;
        let mut g=vec![0.;r*r];let mut c0=vec![0.;k];for(c,&v)in c0.iter_mut().zip(initial){*c=v;}
        if r>1{
            // S = Z^T W = H^T - tau K, R = Z^T A W = Fm - tau H; rows 1.. of G
            // are S'^-1 R[1.., :] and S' c[1..] = Z^T x0 [1..] - S[1.., 0] c_0.
            let(tau,m)=(self.tau,r-1);
            let s=|i:usize,j:usize|self.h[j*r+i]-tau*self.k[i*r+j];
            let t=|i:usize,j:usize|self.fm[i*r+j]-tau*self.h[i*r+j];
            let sp:Vec<f64>=(0..m*m).map(|x|s(1+x%m,1+x/m)).collect();
            let b:Vec<f64>=(0..m*r).map(|x|t(1+x%m,x/m)).collect();
            let(lu,piv)=lapack.factor(&sp,m)?;let x=lapack.solve_matrix_factored(&lu,&piv,&b,m)?;
            for i in 1..r{for j in 0..r{g[i*r+j]=x[(i-1)+m*j];}}
            let rhs:Vec<f64>=(1..r).map(|i|self.zx[i]-s(i,0)*c0[0]).collect();let c=lapack.solve_matrix_factored(&lu,&piv,&rhs,m)?;
            c0[1..r].copy_from_slice(&c);
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
        // Kernel and growing modes: the eigenvector carrying the trace (the
        // largest carrier coordinate; every other eigenvector's is zero) and
        // every further conserved quantity.
        let carrying=(0..r).max_by(|&p,&q|magnitude(s[p]).total_cmp(&magnitude(s[q]))).ok_or("Empty reduced model")?;
        let kernel:Vec<bool>=(0..r).map(|jj|jj==carrying||magnitude(theta[jj])<KERNEL).collect();
        let(mut used_theta,mut used_coefficients)=(theta.clone(),a.clone());
        let(mut growing,mut projected,mut clipped,mut growth_charge)=(0,0,0,0.);
        for jj in 0..r{
            if kernel[jj]{used_theta[jj]=ZERO;continue;}
            if !(theta[jj].re>0.){continue;}
            let growth=((theta[jj].re*longest).min(EXPONENT_LIMIT).exp()-1.)*amplitudes[jj];
            if growth<=GROWTH_NEGLIGIBLE{used_theta[jj].re=0.;continue;}
            growing+=1;
            if amplitudes[jj]<=GROWTH_BUDGET*tolerance{used_theta[jj]=ZERO;used_coefficients[jj]=ZERO;projected+=1;growth_charge+=amplitudes[jj];}
            else{used_theta[jj].re=0.;clipped+=1;}
        }
        if g.iter().any(|v|!v.is_finite()){return Err("Nonfinite reduced model".into());}
        Ok(Model{r,theta,vectors:s,used_theta,used_coefficients,kernel,amplitudes,growing,projected,clipped,growth_charge,readout})
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
}

/// A slow relaxation rate (1/us), its residual (rad/us) and whether every
/// class of the frame's second-order block slower than it is matched by a
/// Ritz pair (`pipeline`): resolved when it is and the residual is within
/// SLOW_ACCURACY of the rate, else only the range rate +- residual (a
/// slower class the space has not reached included).
#[derive(Clone,Copy,Debug)]
pub struct SlowRate{pub rate:f64,pub residual:f64,pub matched:bool}
impl SlowRate{
    pub fn resolved(&self)->bool{self.matched&&self.residual<=SLOW_ACCURACY*self.rate}
}
