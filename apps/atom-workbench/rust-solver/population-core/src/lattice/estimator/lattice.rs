//! The error lattice of one run: the space's modes P u Q, the
//! Schrieffer-Wolff blocks of the fast Ritz modes near the centre, and the
//! omitted ring R = ring(P u Q) for the slow modes' sources.
//!
//! Fast part (R2 section 11.2): only the centre, its one-step neighbours and
//! the explicit set P (the near modes) get class coordinates; their sources are transformed by
//! I - S with the first-order transfer from every lattice neighbour (whose
//! eigen-coordinates of L0' are therefore read), and their errors are read
//! out on themselves and, through I + S, on their neighbours. Per basis
//! column w_j (with its image A w_j), once:
//!   z_w = V^-1 E^-1 (I - S) w_j,  z_a = V^-1 E^-1 (I - S) A w_j  on the near modes.
//! The other lattice neighbours of the near modes (the outer modes) carry
//! their error with the dressed blocks of L0' alone, from the
//! eigen-coordinates the transfer reads anyway (five lasers: the error on
//! the ring Q is a large part at 20 us).
//! Per checkpoint the source of Ritz mode m is (A W - W theta_m) b_m a_m
//! e^{theta_m t}: beta = (z_a b - z_w b theta) a, then the closed-form
//! convolution and the readout (`propagate`).
//!
//! Ring (`ring_feedback`): for a field x on P u Q and a shift z, the ring
//! slaved at z, y = (z - D_R)^-1 (C x)|_R (exact block solves, refined for
//! the diagonal fold's i nu_r K^(1)), and its feedback (B y)|_{P u Q}.
//! Complex fields are split into their Hermitian-symmetric parts
//! (x = a + i b, a = (x + J x)/2, b = (x - J x)/2i), the parts the lattice
//! operator acts on.
use super::super::*;
use super::super::eigenbasis::BlockEigenbasis;
use super::super::operator::{LatticeOperator,Mask,Parts};
use super::blocks::BlockSystem;
use super::kernels::{KernelRule,SwKernels};
use crate::parallel::Execution;

/// Refinement passes of the ring's block solves for the diagonal fold's
/// frequency term i nu_r K^(1) (left out of the eigenbasis), and their
/// relative residual target: the slaved field enters the sources linearly,
/// so a few digits beyond the estimate's own are enough (the term is about
/// 1e-4 of the block, one refinement reaches it).
const RING_REFINEMENTS:usize=12;
const RING_RESIDUAL:f64=1e-8;
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
    /// Lattice operator applications counted per computed block (the
    /// ring's couplings and refinements), exact block solves, and
    /// coordinate changes (one d x d product per block: E^-1 and V^-1).
    pub block_operations:usize,pub block_solves:usize,pub frame_blocks:usize,
    /// Columns processed and their time (ms).
    pub columns:usize,pub column_ms:f64,
}

/// The Hermitian-symmetric parts (a, b) of a field x = a + i b on modes with
/// the mirror map: a = (x + J x)/2, b = (x - J x)/2i, (J x)_q = x_-q^dagger.
pub(super) fn split(x:&[C],mirror:&[usize],n:usize)->(Vec<C>,Vec<C>){
    let d=n*n;let(mut a,mut b)=(vec![ZERO;x.len()],vec![ZERO;x.len()]);
    for(q,&m)in mirror.iter().enumerate(){for i in 0..n{for j in 0..n{
        let(u,v)=(x[q*d+i*n+j],conj(x[m*d+j*n+i]));
        a[q*d+i*n+j]=C{re:0.5*(u.re+v.re),im:0.5*(u.im+v.im)};
        // (u - v) / 2i = -i (u - v) / 2.
        b[q*d+i*n+j]=C{re:0.5*(u.im-v.im),im:-0.5*(u.re-v.re)};}}}
    (a,b)
}
/// a + i b.
pub(super) fn join(a:&[C],b:&[C])->Vec<C>{a.iter().zip(b).map(|(x,y)|C{re:x.re-y.im,im:x.im+y.re}).collect()}
pub(super) fn norm(x:&[C])->f64{x.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()}

pub struct ErrorLattice<'p>{
    pub(super) prepared:&'p LatticePrepared,pub(super) basis:&'p BlockEigenbasis,
    pub n:usize,pub d:usize,
    /// P u Q (the space's modes), their frequencies, the centre and the
    /// mirror map.
    pub modes:Vec<Charge>,pub nu:Vec<f64>,pub centre:usize,
    pub(super) index:BTreeMap<Charge,usize>,pub(super) mirror:Vec<usize>,
    /// The omitted ring R = ring(P u Q) and the lattice operator on
    /// P u Q u R, applied with masks: P u Q -> R for the outflow, R -> R for
    /// the ring's diagonal blocks, R -> P u Q for the feedback.
    pub ring:Vec<Charge>,ring_nu:Vec<f64>,
    outer:LatticeOperator<'p>,inner_mask:Vec<bool>,ring_mask:Vec<bool>,edge_modes:usize,
    pub kernels:SwKernels,
    /// The near modes (the centre and its one-step neighbours), and the (I - S)
    /// pairs into them: (target, source, step), target = source + step.
    pub near:Vec<usize>,pub(super) pairs:Vec<(usize,usize,usize)>,
    /// The modes whose eigen-coordinates the transform reads, and those of
    /// them outside the near modes (the outer modes): their errors are
    /// propagated by the dressed blocks of L0' alone, from the coordinates
    /// the transform reads anyway.
    frames:Vec<usize>,pub outer_modes:Vec<usize>,
    pub classes:Vec<BlockSystem>,pub(super) class_of:Vec<usize>,
    /// Per near mode (in `near` order), column-major d x k: class
    /// coordinates of the transformed basis columns and images.
    pub(super) basis_coordinates:Vec<Vec<C>>,pub(super) image_coordinates:Vec<Vec<C>>,
    /// Per outer mode, column-major d x k: eigen-coordinates of L0' of the
    /// basis columns and images.
    pub(super) outer_basis:Vec<Vec<C>>,pub(super) outer_image:Vec<Vec<C>>,
    pub columns:usize,
    /// Whether the fast part's sources carry the ring slaved at z = 0 (its
    /// feedback added to every image), and whether the outer modes are
    /// propagated.
    pub ring_in_fast:bool,pub outer_bare:bool,
    pub work:ErrorWork,
    /// Build times (ms): kernels, classes.
    pub build_ms:[f64;2],
}

impl<'p> ErrorLattice<'p>{
    /// `explicit`: the explicit set P, whose modes join the near ones.
    pub fn new(prepared:&'p LatticePrepared,modes:&[Charge],explicit:&[Charge],rule:KernelRule,classes:Classes)->Result<Self,String>{
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
        let inner_mask:Vec<bool>=(0..all.len()).map(|q|q<modes.len()).collect();let ring_mask:Vec<bool>=inner_mask.iter().map(|&b|!b).collect();
        let structural=prepared.plan.structural_steps();
        let edge_modes=modes.iter().filter(|q|ring.iter().any(|r|charge_minus(r,q).is_ok_and(|s|structural.contains(&s)))).count();
        let clock=Instant::now();
        let kernels=SwKernels::new(prepared,basis,rule)?;
        let kernel_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
        // Near modes: the centre, every lattice mode one kernel step away and
        // the explicit set (by construction where the state has weight).
        let mut near=vec![centre];
        for q in kernels.steps.iter().map(|s|&s.charge).chain(explicit){if let Some(&q)=index.get(q){if !near.contains(&q){near.push(q);}}}
        let mut pairs=Vec::new();let mut keys:Vec<Vec<usize>>=Vec::new();let mut frames=near.clone();
        for &a in &near{
            let q=&modes[a];let mut steps=Vec::new();
            for(s,step)in kernels.steps.iter().enumerate(){
                if let Some(&b)=index.get(&charge_minus(q,&step.charge)?){pairs.push((a,b,s));if !frames.contains(&b){frames.push(b);}}
                if classes==Classes::Full||index.contains_key(&charge_plus(q,&step.charge)?){steps.push(s);}
            }
            keys.push(steps);
        }
        let mut distinct:Vec<Vec<usize>>=keys.clone();distinct.sort();distinct.dedup();
        let mut class_of=vec![usize::MAX;modes.len()];
        for(&a,k)in near.iter().zip(&keys){class_of[a]=distinct.iter().position(|x|x==k).unwrap();}
        let lapack=&prepared.lapack;
        let built=Execution::new()?.map(distinct,|steps|BlockSystem::new(lapack,basis,&kernels,steps,n));
        let classes=built.into_iter().collect::<Result<Vec<_>,_>>()?;
        let class_ms=clock.elapsed().as_secs_f64()*1e3;
        let count=near.len();
        let outer_modes:Vec<usize>=frames.iter().copied().filter(|q|!near.contains(q)).collect();let outer_count=outer_modes.len();
        Ok(Self{prepared,basis,n,d,modes:modes.to_vec(),nu,centre,index,mirror,ring,ring_nu,outer,inner_mask,ring_mask,edge_modes,kernels,
            near,pairs,frames,outer_modes,classes,class_of,basis_coordinates:vec![Vec::new();count],image_coordinates:vec![Vec::new();count],
            outer_basis:vec![Vec::new();outer_count],outer_image:vec![Vec::new();outer_count],columns:0,ring_in_fast:false,outer_bare:true,
            work:ErrorWork::default(),build_ms:[kernel_ms,class_ms]})
    }
    pub fn dimension(&self)->usize{self.modes.len()*self.d}
    /// Process the basis columns not seen yet (columns are append-only).
    pub fn add_columns(&mut self,basis:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        let clock=Instant::now();
        for j in self.columns..basis.len(){
            if basis[j].len()!=self.dimension()||images[j].len()!=basis[j].len(){return Err("Estimator: basis column dimensions differ from the lattice".into());}
            let(zw,ow)=self.coordinates(&basis[j])?;
            let(za,oa)=if self.ring_in_fast{
                let g=self.ring_feedback(&basis[j],ZERO)?;let image:Vec<C>=images[j].iter().zip(&g).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
                self.coordinates(&image)?}else{self.coordinates(&images[j])?};
            let d=self.d;
            for(slot,(w,a))in self.basis_coordinates.iter_mut().zip(self.image_coordinates.iter_mut()).enumerate(){
                w.extend_from_slice(&zw[slot*d..(slot+1)*d]);a.extend_from_slice(&za[slot*d..(slot+1)*d]);}
            for(slot,(w,a))in self.outer_basis.iter_mut().zip(self.outer_image.iter_mut()).enumerate(){
                w.extend_from_slice(&ow[slot*d..(slot+1)*d]);a.extend_from_slice(&oa[slot*d..(slot+1)*d]);}
            self.columns+=1;self.work.columns+=1;
        }
        self.work.column_ms+=clock.elapsed().as_secs_f64()*1e3;
        Ok(())
    }
    /// Class coordinates V_c^-1 E^-1 (I - S) v of a field on P u Q per near
    /// mode (in `near` order), and its coordinates E^-1 v per outer mode.
    fn coordinates(&mut self,v:&[C])->Result<(Vec<C>,Vec<C>),String>{
        let d=self.d;let execution=Execution::new()?;let basis=self.basis;
        let chunks:Vec<Vec<usize>>=self.frames.chunks(BLOCKS_PER_TASK).map(|c|c.to_vec()).collect();
        let parts=execution.map(chunks.clone(),|modes|{let blocks:Vec<&[C]>=modes.iter().map(|&q|&v[q*d..(q+1)*d]).collect();basis.coefficients(&blocks)});
        let mut c=vec![ZERO;self.modes.len()*d];
        for(modes,part)in chunks.iter().zip(parts){let part=part?;for(i,&q)in modes.iter().enumerate(){c[q*d..(q+1)*d].copy_from_slice(&part[i*d..(i+1)*d]);}}
        let mut t:Vec<C>=self.near.iter().flat_map(|&a|c[a*d..(a+1)*d].iter().copied()).collect();
        for &(a,b,s)in &self.pairs{let slot=self.near.iter().position(|&x|x==a).unwrap();self.kernels.subtract_kernel(s,&c[b*d..(b+1)*d],&mut t[slot*d..(slot+1)*d]);}
        let(classes,class_of,lapack,near)=(&self.classes,&self.class_of,&self.prepared.lapack,&self.near);
        let t=&t;
        let slots:Vec<usize>=(0..near.len()).collect();
        let parts=execution.map(slots,|slot|->Result<Vec<C>,String>{
            let mut out=vec![ZERO;d];
            lapack.complex_gemm().multiply(d,1,d,ONE,&classes[class_of[near[slot]]].inverse,d,&t[slot*d..(slot+1)*d],d,ZERO,&mut out,d)?;Ok(out)});
        let mut z=Vec::with_capacity(near.len()*d);for p in parts{z.extend(p?);}
        let outer:Vec<C>=self.outer_modes.iter().flat_map(|&q|c[q*d..(q+1)*d].iter().copied()).collect();
        self.work.frame_blocks+=self.frames.len()+self.near.len();
        Ok((z,outer))
    }
    /// The ring's feedback on P u Q of a field x at the shift z:
    /// (B (z - D_R)^-1 C x)|_{P u Q}.
    pub fn ring_feedback(&mut self,x:&[C],shift:C)->Result<Vec<C>,String>{
        let d=self.d;let(inner,count)=(self.modes.len(),self.ring.len());
        if count==0{return Ok(vec![ZERO;inner*d]);}
        let(xa,xb)=split(x,&self.mirror,self.n);
        let outflow=|this:&mut Self,part:&[C]|->Result<Vec<C>,String>{
            if norm(part)==0.{return Ok(vec![ZERO;count*d]);}
            let mut field=vec![ZERO;(inner+count)*d];field[..inner*d].copy_from_slice(part);let mut out=vec![ZERO;field.len()];
            this.outer.apply_masked(&field,&mut out,Parts::Full,Some(Mask{targets:&this.ring_mask,sources:&this.inner_mask}))?;this.work.block_operations+=count;
            Ok(out[inner*d..].to_vec())};
        let(oa,ob)=(outflow(self,&xa)?,outflow(self,&xb)?);let source=join(&oa,&ob);let size=norm(&source);
        if size==0.{return Ok(vec![ZERO;inner*d]);}
        // (D_R - z) y = source, then -y.
        let mut y=source.clone();self.basis.solve_blocks(&self.ring_nu,shift,&mut y)?;self.work.block_solves+=count;
        for _ in 0..RING_REFINEMENTS{
            let mut r=source.clone();
            let applied=self.ring_parts(&y,Parts::Diagonal,true)?;
            for((t,a),v)in r.iter_mut().zip(&applied).zip(&y){let w=mul(shift,*v);t.re-=a.re-w.re;t.im-=a.im-w.im;}
            if norm(&r)<=RING_RESIDUAL*size{break;}
            self.basis.solve_blocks(&self.ring_nu,shift,&mut r)?;self.work.block_solves+=count;
            for(a,b)in y.iter_mut().zip(&r){add(a,*b);}
        }
        for v in &mut y{v.re=-v.re;v.im=-v.im;}
        self.ring_parts(&y,Parts::Full,false)
    }
    /// The operator's parts on a (complex) ring field: its diagonal blocks
    /// back onto the ring (`ring_rows`) or its couplings onto P u Q.
    fn ring_parts(&mut self,y:&[C],parts:Parts,ring_rows:bool)->Result<Vec<C>,String>{
        let d=self.d;let(inner,count)=(self.modes.len(),self.ring.len());
        let ring_mirror:Vec<usize>=(0..count).map(|r|self.outer.mirror[inner+r]-inner).collect();
        let(ya,yb)=split(y,&ring_mirror,self.n);
        let mut results=Vec::new();
        for part in[&ya,&yb]{
            if norm(part)==0.{results.push(vec![ZERO;if ring_rows{count*d}else{inner*d}]);continue;}
            let mut field=vec![ZERO;(inner+count)*d];field[inner*d..].copy_from_slice(part);let mut out=vec![ZERO;field.len()];
            let mask=if ring_rows{Mask{targets:&self.ring_mask,sources:&self.ring_mask}}else{Mask{targets:&self.inner_mask,sources:&self.ring_mask}};
            self.outer.apply_masked(&field,&mut out,parts,Some(mask))?;self.work.block_operations+=if ring_rows{count}else{self.edge_modes};
            results.push(if ring_rows{out[inner*d..].to_vec()}else{out[..inner*d].to_vec()});
        }
        Ok(join(&results[0],&results[1]))
    }
}
