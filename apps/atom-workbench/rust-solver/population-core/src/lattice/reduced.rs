//! The reduced model of the space and every output from it.
//!
//! The projection: Galerkin. The basis W is orthonormal in the real inner
//! product of Hermitian-symmetric fields (`krylov::Vectors`), its first column
//! the trace carrier w_0 = u / |u| and every other in K, the fields of zero
//! block traces (`space`). The model is u' = H u with H = W^T A W, formed
//! from the images A W, which are stored as the basis grows (one application
//! per column, its new row and column of H by two Gram sweeps). A maps every
//! field into K, so H's carrier row is zero: every block trace is conserved
//! exactly and the carrier coefficient stays |u|. u(0) = W^T x0 = (|u|,
//! |x0 - u|, 0, ...): the space starts from x0 (`Space::initial`).
//!
//! The Ritz decomposition H s_j = theta_j s_j, u(0) = sum_j a_j s_j, gives the
//! model's modes x(t) = sum_j a_j e^{theta_j t} v_j, v_j = W s_j. A mode with
//! |theta_j| T within the a-priori share of the tolerance is conserved over
//! the run (theta_j = 0); a growing one (Re theta_j > 0, an artefact of the
//! projection: the generator's spectrum is in the left half plane) is
//! reflected, Re theta_j -> -Re theta_j. Clipped to the imaginary axis it
//! would live undamped to the horizon (r21's mean at 1 s: 2.0e-2 off with
//! two clipped modes, 1.6e-8 reflected).
//!
//! Outputs from the modes: the trajectory p(t) = sum_q Re e^{i nu_q t}
//! diag(x_q(t)) (the local time average), the state at a horizon T and its
//! centre block (the laboratory time average at T), and the slowest
//! relaxation: the slowest decaying Ritz value of the outputs' modes.
use super::*;
use super::eigenbasis::conjugate_pairs;
use super::krylov::Vectors;
use super::poles::Lattice;
use super::accuracy::{A_PRIORI_SHARE,POPULATION_TOLERANCE};
use crate::parallel::Execution;

/// The Galerkin projection of the space: the images and H.
pub struct Galerkin{
    n:usize,d:usize,
    /// Canonical modes: (mode, weight, nu).
    canonical:Vec<(usize,f64,f64)>,centre:usize,
    images:Vec<Vec<C>>,
    /// H = W^T A W, row-major over the covered columns.
    h:Vec<f64>,
    /// Per column: the canonical modes' diagonals (canonical x n).
    populations:Vec<Vec<C>>,
    vectors:Vectors,
    pub milliseconds:f64,
}
impl Galerkin{
    pub fn new(lattice:&Lattice<'_>)->Result<Self,String>{
        let op=lattice.operator();let n=op.n;
        let canonical:Vec<(usize,f64,f64)>=(0..op.modes.len()).filter(|&q|op.canonical[q]).map(|q|(q,if op.mirror[q]==q{1.}else{2.},op.nu[q])).collect();
        let centre=canonical.iter().position(|c|c.0==op.centre).unwrap_or(0);
        Ok(Self{n,d:n*n,canonical,centre,images:vec![],h:vec![],populations:vec![],vectors:Vectors::new(op)?,milliseconds:0.})
    }
    pub fn columns(&self)->usize{self.images.len()}
    /// Takes the space's columns not yet covered: their images, H's new
    /// columns (the images against every column) and rows (the new columns
    /// against every old image).
    pub fn ingest(&mut self,basis:&[Vec<C>],lattice:&mut Lattice<'_>)->Result<(),String>{
        let clock=Instant::now();let(old,new)=(self.images.len(),basis.len());if new==old{return Ok(());}
        let(n,d)=(self.n,self.d);
        for w in &basis[old..new]{let mut a=vec![ZERO;w.len()];lattice.apply(w,&mut a)?;self.images.push(a);
            self.populations.push(self.canonical.iter().flat_map(|&(q,_,_)|(0..n).map(move|i|w[q*d+i*n+i])).collect());}
        let all:Vec<&[C]>=basis[..new].iter().map(|v|v.as_slice()).collect();
        let fresh_images:Vec<&[C]>=self.images[old..].iter().map(|v|v.as_slice()).collect();
        // H[i, old + c] = Re <w_i, A w_c> for every i.
        let columns=self.vectors.gram(&fresh_images,&all)?;
        // H[old + c, j] = Re <w_c, A w_j> for the old j.
        let fresh:Vec<&[C]>=basis[old..new].iter().map(|v|v.as_slice()).collect();
        let old_images:Vec<&[C]>=self.images[..old].iter().map(|v|v.as_slice()).collect();
        let rows=self.vectors.gram(&fresh,&old_images)?;
        let width=new-old;let mut h=vec![0.;new*new];
        for i in 0..old{for j in 0..old{h[i*new+j]=self.h[i*old+j];}}
        for c in 0..width{for i in 0..new{h[i*new+old+c]=columns[c*new+i];}}
        for c in 0..width{for j in 0..old{h[(old+c)*new+j]=rows[c*old+j];}}
        // A maps into K: the carrier row is zero.
        for v in &mut h[..new]{*v=0.;}
        self.h=h;
        self.milliseconds+=clock.elapsed().as_secs_f64()*1e3;Ok(())
    }
    /// The model over the covered columns; `initial` x0's leading coordinates
    /// (`Space::initial`), `longest` the run's longest output time.
    pub fn model(&self,lapack:&Lapack,initial:&[f64],longest:f64)->Result<Model,String>{
        let r=self.images.len();let n=self.n;
        if self.h.iter().any(|v|!v.is_finite()){return Err("Nonfinite reduced model".into());}
        let mut c0=vec![0.;r];for(c,&v)in c0.iter_mut().zip(initial){*c=v;}
        let(re,im,ev)=lapack.eigen(&transpose_rows(&self.h,r,r),r)?;
        let(pair,theta)=conjugate_pairs(&re,&im);
        // Complex eigenvectors s_j (column-major), a = S^-1 u(0) by the real
        // solve and the pair rule.
        let(lu,pivots)=lapack.factor(&ev,r)?;let y=lapack.solve_matrix_factored(&lu,&pivots,&c0,r)?;
        let mut s=vec![ZERO;r*r];let mut a=vec![ZERO;r];let mut j=0;
        while j<r{
            if pair[j]==1{for i in 0..r{let(p,q)=(ev[i+r*j],ev[i+r*(j+1)]);s[i+r*j]=C{re:p,im:q};s[i+r*(j+1)]=C{re:p,im:-q};}
                a[j]=C{re:0.5*y[j],im:-0.5*y[j+1]};a[j+1]=C{re:0.5*y[j],im:0.5*y[j+1]};j+=2;}
            else{for i in 0..r{s[i+r*j]=C{re:ev[i+r*j],im:0.};}a[j]=C{re:y[j],im:0.};j+=1;}
        }
        // The modes' population rows per canonical mode: populations(W) s_j.
        let readout:Vec<Vec<C>>=Execution::new()?.map((0..self.canonical.len()).collect(),|c|{
            let mut out=vec![ZERO;n*r];
            for jj in 0..r{for i in 0..n{let mut v=ZERO;for col in 0..r{let p=self.populations[col][c*n+i];if nonzero(p){mul_add(&mut v,p,s[col+r*jj]);}}out[i+n*jj]=v;}}
            out});
        Ok(Model::new(theta,a,readout,self.canonical.iter().map(|c|(c.1,c.2)).collect(),self.centre,s,longest))
    }
}
fn transpose_rows(a:&[f64],r:usize,c:usize)->Vec<f64>{let mut t=vec![0.;r*c];for i in 0..r{for j in 0..c{t[j*r+i]=a[i*c+j];}}t}

/// The modes of one checkpoint: x(t) = sum_j a_j e^{theta_j t} v_j.
pub struct Model{
    pub r:usize,
    /// Ritz values, and as the outputs use them (conserved and reflected).
    pub theta:Vec<C>,pub used:Vec<C>,
    /// a_j, and per mode its largest population contribution |a_j| max|v_j|.
    pub coefficients:Vec<C>,pub amplitudes:Vec<f64>,
    /// Growing modes (Re theta > 0), each reflected into the left half plane.
    pub growing:usize,
    /// Per canonical mode (weight, nu) the population rows of every v_j
    /// (column-major n x r); the centre's position.
    readout:Vec<Vec<C>>,modes:Vec<(f64,f64)>,centre:usize,
    /// s_j in the space's coordinates (column-major r x r): v_j = W s_j.
    pub vectors:Vec<C>,
}
impl Model{
    fn new(theta:Vec<C>,coefficients:Vec<C>,readout:Vec<Vec<C>>,modes:Vec<(f64,f64)>,centre:usize,vectors:Vec<C>,longest:f64)->Self{
        let r=theta.len();let n=readout.first().map_or(0,|b|b.len()/r.max(1));
        let amplitudes:Vec<f64>=(0..r).map(|j|magnitude(coefficients[j])*readout.iter().map(|b|b[n*j..n*(j+1)].iter().map(|z|magnitude(*z)).fold(0_f64,f64::max)).fold(0_f64,f64::max)).collect();
        // Conserved within the run: |theta| T within the a-priori share.
        let conserved=A_PRIORI_SHARE*POPULATION_TOLERANCE/longest.max(f64::MIN_POSITIVE);
        let mut growing=0;
        let used:Vec<C>=theta.iter().map(|&t|{
            if magnitude(t)<=conserved{return ZERO;}
            if t.re>0.{growing+=1;return C{re:-t.re,im:t.im};}
            t}).collect();
        Self{r,theta,used,coefficients,amplitudes,growing,readout,modes,centre,vectors}
    }
    /// The complex mode weights at t, a_j e^{used_j t}.
    pub fn weights(&self,t:f64)->Vec<C>{(0..self.r).map(|j|mul(self.coefficients[j],cexp(self.used[j],t))).collect()}
    /// The population rows of canonical mode c at t.
    fn populations(&self,c:usize,weights:&[C])->Vec<C>{
        let n=self.readout[c].len()/self.r.max(1);let mut p=vec![ZERO;n];
        for(j,&w)in weights.iter().enumerate(){if !nonzero(w){continue;}for(i,pi)in p.iter_mut().enumerate(){mul_add(pi,self.readout[c][i+n*j],w);}}
        p
    }
    /// The laboratory populations at t (the local time average).
    pub fn at(&self,t:f64)->Vec<f64>{
        let weights=self.weights(t);let mut out:Vec<f64>=Vec::new();
        for(c,&(w,nu))in self.modes.iter().enumerate(){
            let p=self.populations(c,&weights);let phase=cexp(C{re:0.,im:nu},t);
            if out.is_empty(){out=vec![0.;p.len()];}
            for(o,v)in out.iter_mut().zip(&p){*o+=w*mul(phase,*v).re;}
        }
        out
    }
    /// The trajectory's populations on the grid.
    pub fn trajectory(&self,times:&[f64])->Vec<Vec<f64>>{
        Execution::new().map(|e|e.map(times.to_vec(),|t|self.at(t))).unwrap_or_else(|_|times.iter().map(|&t|self.at(t)).collect())
    }
    /// The laboratory time average at the horizon T (the centre block).
    pub fn horizon(&self,t:f64)->Vec<f64>{self.populations(self.centre,&self.weights(t)).iter().map(|z|z.re).collect()}
    /// The slowest relaxation rate (1/us): the slowest decaying mode as the
    /// outputs use it (not conserved over the run, a growing one reflected)
    /// of population weight at least `weight`.
    pub fn slowest(&self,weight:f64)->Option<f64>{
        (0..self.r).filter(|&j|self.used[j].re<0.&&self.amplitudes[j]>=weight).map(|j|-self.used[j].re).min_by(f64::total_cmp)
    }
}
