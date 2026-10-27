//! The error lattice of one run: the error equation E' = A1 E + R1 u(t) on
//! the space's modes P u Q, the omitted ring R = ring(P u Q) slaved, and its
//! Schrieffer-Wolff propagation (module documentation of `estimator`).
//!
//! Per basis column w_j (with its image A w_j), once:
//! - the ring: y_j = -D_R^-1 (C w_j)|_R at z = 0 (exact block solves, D_R
//!   the ring's diagonal blocks), its feedback g_j = (B y_j)|_{P u Q} and
//!   the ring's own populations of y_j;
//! - the class coordinates of the first-order transform of both source
//!   parts, z_w = V_c^-1 E^-1 (I - S) w_j and z_a = V_c^-1 E^-1 (I - S)(A w_j + g_j)
//!   per mode.
//! Per checkpoint, with the Ritz vectors W b_m (b_m = X s_m), values theta_m
//! and coefficients a_m the outputs use, the source of mode m is
//! (A W + G - W theta_m) b_m a_m e^{theta_m t}: on every mode beta =
//! (z_a b - z_w b theta) a, then the closed-form convolution per eigenvalue
//! of the mode's class and the readout (`propagate`).
use super::super::*;
use super::super::eigenbasis::BlockEigenbasis;
use super::super::operator::{LatticeOperator,Parts};
use super::blocks::BlockSystem;
use super::kernels::{KernelRule,SwKernels};
use crate::parallel::Execution;

/// Refinement passes of the ring's block solves for the diagonal fold's
/// frequency term i nu_r K^(1) (left out of the eigenbasis), and their
/// relative residual target.
const RING_REFINEMENTS:usize=12;
const RING_RESIDUAL:f64=1e-12;
/// Blocks per task of the coordinate changes.
const BLOCKS_PER_TASK:usize=4;

/// Which second-order terms a mode's block carries: those of the steps
/// whose neighbour is on the lattice (the error equation on P u Q as R2
/// wrote it, one eigensystem per neighbour pattern), or every step (the
/// infinite lattice's averaging: one flip-invariant eigensystem for every
/// mode, the edge modes' outward paths included).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Classes{Neighbours,Full}

#[derive(Clone,Debug,Default)]
pub struct ErrorWork{
    /// Lattice operator applications counted per block (the ring's
    /// couplings and refinements), exact block solves, and coordinate
    /// changes (one d x d product per block: E^-1 and V^-1).
    pub block_operations:usize,pub block_solves:usize,pub frame_blocks:usize,
    /// Columns processed and their time (ms).
    pub columns:usize,pub column_ms:f64,
}

pub struct ErrorLattice<'p>{
    pub(super) prepared:&'p LatticePrepared,pub(super) basis:&'p BlockEigenbasis,
    pub n:usize,pub d:usize,
    /// P u Q (the space's modes), their frequencies and the centre.
    pub modes:Vec<Charge>,pub nu:Vec<f64>,pub centre:usize,
    pub(super) index:BTreeMap<Charge,usize>,pub(super) mirror:Vec<usize>,
    /// The omitted ring R = ring(P u Q).
    pub ring:Vec<Charge>,pub(super) ring_nu:Vec<f64>,
    /// The lattice operator on P u Q u R (the ring's couplings).
    outer:LatticeOperator<'p>,
    pub kernels:SwKernels,
    /// (target, source, step): target = source + step, both in P u Q.
    pub(super) pairs:Vec<(usize,usize,usize)>,
    pub classes:Vec<BlockSystem>,pub(super) class_of:Vec<usize>,
    /// Per mode, column-major d x k: class coordinates of the transformed
    /// basis columns and of their transformed images (+ ring feedback).
    pub(super) basis_coordinates:Vec<Vec<C>>,pub(super) image_coordinates:Vec<Vec<C>>,
    /// Ring populations of y_j, column-major (|R| n) x k.
    pub(super) ring_populations:Vec<C>,
    pub columns:usize,
    pub work:ErrorWork,
    /// Build times (ms): kernels, classes.
    pub build_ms:[f64;2],
}

impl<'p> ErrorLattice<'p>{
    pub fn new(prepared:&'p LatticePrepared,modes:&[Charge],rule:KernelRule,classes:Classes)->Result<Self,String>{
        let basis=prepared.eigenbasis()?;
        let n=prepared.generator.n;let d=n*n;
        let index:BTreeMap<Charge,usize>=modes.iter().enumerate().map(|(i,q)|(q.clone(),i)).collect();
        let centre=*index.get(&prepared.generator.zero).ok_or("Error lattice without the centre")?;
        let mirror=modes.iter().map(|q|index.get(&opposite(q)).copied().ok_or("Error lattice not closed under q -> -q")).collect::<Result<Vec<_>,_>>()?;
        let nu=modes.iter().map(|q|prepared.nu(q)).collect::<Result<Vec<_>,_>>()?;
        let ring=super::super::estimate_api::ring(prepared,modes)?;
        let ring_nu=ring.iter().map(|q|prepared.nu(q)).collect::<Result<Vec<_>,_>>()?;
        let all:Vec<Charge>=modes.iter().chain(&ring).cloned().collect();
        let outer=LatticeOperator::new(prepared,&all)?;
        let clock=Instant::now();
        let kernels=SwKernels::new(prepared,basis,rule)?;
        let kernel_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
        let mut pairs=Vec::new();let mut keys:Vec<Vec<usize>>=Vec::new();
        for(a,q)in modes.iter().enumerate(){
            let mut steps=Vec::new();
            for(s,step)in kernels.steps.iter().enumerate(){
                if let Some(&b)=index.get(&charge_minus(q,&step.charge)?){pairs.push((a,b,s));}
                if classes==Classes::Full||index.contains_key(&charge_plus(q,&step.charge)?){steps.push(s);}
            }
            keys.push(steps);
        }
        let mut distinct:Vec<Vec<usize>>=keys.clone();distinct.sort();distinct.dedup();
        let class_of=keys.iter().map(|k|distinct.iter().position(|x|x==k).unwrap()).collect();
        let lapack=&prepared.lapack;
        let built=Execution::new()?.map(distinct,|steps|BlockSystem::new(lapack,basis,&kernels,steps,n));
        let classes=built.into_iter().collect::<Result<Vec<_>,_>>()?;
        let class_ms=clock.elapsed().as_secs_f64()*1e3;
        let count=modes.len();
        Ok(Self{prepared,basis,n,d,modes:modes.to_vec(),nu,centre,index,mirror,ring,ring_nu,outer,kernels,pairs,classes,class_of,
            basis_coordinates:vec![Vec::new();count],image_coordinates:vec![Vec::new();count],ring_populations:Vec::new(),columns:0,
            work:ErrorWork::default(),build_ms:[kernel_ms,class_ms]})
    }
    pub fn dimension(&self)->usize{self.modes.len()*self.d}
    /// Process the basis columns not seen yet (columns are append-only).
    pub fn add_columns(&mut self,basis:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        let clock=Instant::now();
        for j in self.columns..basis.len(){self.add_column(&basis[j],&images[j])?;}
        self.work.column_ms+=clock.elapsed().as_secs_f64()*1e3;
        Ok(())
    }
    fn add_column(&mut self,w:&[C],aw:&[C])->Result<(),String>{
        if w.len()!=self.dimension()||aw.len()!=w.len(){return Err("Estimator: basis column dimensions differ from the lattice".into());}
        let(n,d)=(self.n,self.d);
        let y=self.slave(w)?;
        for r in 0..self.ring.len(){for i in 0..n{self.ring_populations.push(y[r*d+i*n+i]);}}
        let g=self.feedback(&y)?;
        let image:Vec<C>=aw.iter().zip(&g).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
        let zw=self.coordinates(w)?;let za=self.coordinates(&image)?;
        for b in 0..self.modes.len(){
            self.basis_coordinates[b].extend_from_slice(&zw[b*d..(b+1)*d]);
            self.image_coordinates[b].extend_from_slice(&za[b*d..(b+1)*d]);
        }
        self.columns+=1;self.work.columns+=1;
        Ok(())
    }
    /// The slaved ring of a field x on P u Q: y = -D_R^-1 (C x)|_R.
    pub(super) fn slave(&mut self,x:&[C])->Result<Vec<C>,String>{
        let d=self.d;let(inner,count)=(self.modes.len(),self.ring.len());
        if count==0{return Ok(Vec::new());}
        let mut field=vec![ZERO;(inner+count)*d];field[..inner*d].copy_from_slice(x);
        let mut out=vec![ZERO;field.len()];
        self.outer.apply(&field,&mut out)?;self.work.block_operations+=inner+count;
        let source=out[inner*d..].to_vec();let size=norm(&source);
        let mut y=source.clone();self.basis.solve_blocks(&self.ring_nu,ZERO,&mut y)?;self.work.block_solves+=count;
        for _ in 0..RING_REFINEMENTS{
            if size==0.{break;}
            field.fill(ZERO);field[inner*d..].copy_from_slice(&y);
            self.outer.apply_parts(&field,&mut out,Parts::Diagonal)?;self.work.block_operations+=inner+count;
            let mut r:Vec<C>=source.iter().zip(&out[inner*d..]).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect();
            if norm(&r)<=RING_RESIDUAL*size{break;}
            self.basis.solve_blocks(&self.ring_nu,ZERO,&mut r)?;self.work.block_solves+=count;
            for(a,b)in y.iter_mut().zip(&r){add(a,*b);}
        }
        for v in &mut y{v.re=-v.re;v.im=-v.im;}
        Ok(y)
    }
    /// The ring's feedback on P u Q: (B y)|_{P u Q}.
    pub(super) fn feedback(&mut self,y:&[C])->Result<Vec<C>,String>{
        let d=self.d;let inner=self.modes.len();
        if y.is_empty(){return Ok(vec![ZERO;inner*d]);}
        let mut field=vec![ZERO;(inner+self.ring.len())*d];field[inner*d..].copy_from_slice(y);
        let mut out=vec![ZERO;field.len()];
        self.outer.apply(&field,&mut out)?;self.work.block_operations+=inner+self.ring.len();
        out.truncate(inner*d);Ok(out)
    }
    /// Class coordinates V_c^-1 E^-1 (I - S) v of a field on P u Q, per mode.
    fn coordinates(&mut self,v:&[C])->Result<Vec<C>,String>{
        let d=self.d;let count=self.modes.len();
        let execution=Execution::new()?;
        let chunks:Vec<usize>=(0..count).step_by(BLOCKS_PER_TASK).collect();
        let basis=self.basis;
        let parts=execution.map(chunks.clone(),|first|{
            let blocks:Vec<&[C]>=v[first*d..(first+BLOCKS_PER_TASK).min(count)*d].chunks_exact(d).collect();
            basis.coefficients(&blocks)});
        let mut c=Vec::with_capacity(count*d);for p in parts{c.extend(p?);}
        let mut t=c.clone();
        for &(a,b,s)in &self.pairs{self.kernels.subtract_kernel(s,&c[b*d..(b+1)*d],&mut t[a*d..(a+1)*d]);}
        let(classes,class_of,lapack)=(&self.classes,&self.class_of,&self.prepared.lapack);
        let t=&t;
        let parts=execution.map(chunks,|first|->Result<Vec<C>,String>{
            let gemm=lapack.complex_gemm();let last=(first+BLOCKS_PER_TASK).min(count);let mut out=vec![ZERO;(last-first)*d];
            for b in first..last{gemm.multiply(d,1,d,ONE,&classes[class_of[b]].inverse,d,&t[b*d..(b+1)*d],d,ZERO,&mut out[(b-first)*d..(b-first+1)*d],d)?;}
            Ok(out)});
        let mut z=Vec::with_capacity(count*d);for p in parts{z.extend(p?);}
        self.work.frame_blocks+=2*count;
        Ok(z)
    }
    /// The Hermitian-symmetric part (x + J x)/2 of a field on P u Q.
    pub(super) fn hermitian_part(&self,x:&[C])->Vec<C>{
        let(n,d)=(self.n,self.d);let mut out=vec![ZERO;x.len()];
        for(q,&m)in self.mirror.iter().enumerate(){for i in 0..n{for j in 0..n{
            let(a,b)=(x[q*d+i*n+j],x[m*d+j*n+i]);out[q*d+i*n+j]=C{re:0.5*(a.re+b.re),im:0.5*(a.im-b.im)};}}}
        out
    }
}

pub(super) fn norm(x:&[C])->f64{x.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()}
