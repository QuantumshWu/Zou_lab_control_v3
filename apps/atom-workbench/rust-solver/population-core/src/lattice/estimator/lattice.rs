//! The error lattice of one run: the space's modes P u Q, the
//! Schrieffer-Wolff blocks of the fast Ritz modes near the centre, and the
//! omitted ring R = ring(P u Q) for the slow modes' sources.
//!
//! Fast part: only the centre, its one-step neighbours and
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
//! Flip symmetry: the columns are Hermitian-symmetric (x_-q = x_q^dagger)
//! and the Ritz data come in conjugate pairs, so a mirrored mode's
//! coordinates are the flip of its partner's, c_-q[i] = conj c_q[i*]
//! (`partner`), and its population errors are the conjugates of its
//! partner's. Only the canonical member of each pair is transformed and
//! propagated (counted twice); the mirrored frame modes' coordinates, which
//! the transfer reads, are flipped copies.
//!
//! Ring (`ring_feedback`): for a field x on P u Q and a shift z, the ring
//! slaved at z, y = (z - D_R)^-1 (C x)|_R (exact block solves, refined for
//! the diagonal fold's i nu_r K^(1)), and its feedback (B y)|_{P u Q}; per
//! ring charge, the size of its own amplitude's feedback
//! (`ring_attribution`, the completion's).
//! Complex fields are split into their Hermitian-symmetric parts
//! (x = a + i b, a = (x + J x)/2, b = (x - J x)/2i), the parts the lattice
//! operator acts on.
use super::super::*;
use super::super::eigenbasis::BlockEigenbasis;
use super::super::operator::{LatticeOperator,Mask,Parts};
use super::blocks::BlockSystem;
use super::kernels::SwKernels;
use super::super::frame::SwFrame;
use std::sync::OnceLock;
use crate::parallel::Execution;

/// Refinement passes of the ring's block solves for the diagonal fold's
/// frequency term i nu_r K^(1) (left out of the eigenbasis), and their
/// relative residual target: the slaved field enters the sources linearly,
/// so a few digits beyond the estimate's own are enough (the term is about
/// 1e-4 of the block, one refinement reaches it). RING_REFINEMENTS is a cap,
/// not a working count: it bounds the passes should the term be a large
/// share of a block, where each pass gains less.
const RING_REFINEMENTS:usize=12;
const RING_RESIDUAL:f64=1e-8;
/// Blocks per task of the coordinate changes (E^-1 and V^-1 streamed once
/// per task).
const BLOCKS_PER_TASK:usize=64;

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
    pub(super) prepared:&'p LatticePrepared,
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
    /// The request's frame: its kernels and blocks.
    frame:&'p SwFrame<'p>,
    /// The near modes (the centre and its one-step neighbours), and the (I - S)
    /// pairs into them: (target, source, step), target = source + step.
    pub near:Vec<usize>,pub(super) pairs:Vec<(usize,usize,usize)>,
    /// The modes whose eigen-coordinates the transform reads, and the
    /// canonical ones of them outside the near modes (the outer modes):
    /// their errors are propagated by the dressed blocks of L0' alone, from
    /// the coordinates the transform reads anyway.
    frames:Vec<usize>,pub outer_modes:Vec<usize>,
    /// Per mode whether it is the canonical member of its mirror pair (the
    /// centre is its own mirror), the near slots of the canonical near
    /// modes, and the flip's permutation of the eigen-coordinates.
    pub(super) canonical:Vec<bool>,pub(super) near_canonical:Vec<usize>,partner:OnceLock<Vec<usize>>,
    /// Per near mode (in `near` order; the canonical ones filled),
    /// column-major d x k: class coordinates of the transformed basis
    /// columns and images.
    pub(super) basis_coordinates:Vec<Vec<C>>,pub(super) image_coordinates:Vec<Vec<C>>,
    /// Per outer mode, column-major d x k: eigen-coordinates of L0' of the
    /// basis columns and images.
    pub(super) outer_basis:Vec<Vec<C>>,pub(super) outer_image:Vec<Vec<C>>,
    pub columns:usize,
    pub work:ErrorWork,
}

impl<'p> ErrorLattice<'p>{
    /// `explicit`: the explicit set P, whose modes join the near ones.
    pub fn new(prepared:&'p LatticePrepared,fold:&'p super::super::fold::FoldLibrary,modes:&[Charge],explicit:&[Charge],frame:&'p SwFrame<'p>)->Result<Self,String>{
        let n=prepared.generator.n;let d=n*n;
        let ring=super::super::estimate_api::ring(prepared,modes)?;
        let all:Vec<Charge>=modes.iter().chain(&ring).cloned().collect();
        let outer=LatticeOperator::new(prepared,fold,&all)?;
        // P u Q comes first in the operator: its index, mirrors (closed on
        // P u Q) and frequencies, the ring's after them.
        let count=modes.len();let centre=outer.centre;
        let index:BTreeMap<Charge,usize>=outer.index.iter().filter(|(_,&i)|i<count).map(|(q,&i)|(q.clone(),i)).collect();
        let mirror=outer.mirror[..count].to_vec();
        if mirror.iter().any(|&m|m>=count){return Err("Error lattice not closed under q -> -q".into());}
        let(nu,ring_nu)=(outer.nu[..count].to_vec(),outer.nu[count..].to_vec());
        let inner_mask:Vec<bool>=(0..all.len()).map(|q|q<modes.len()).collect();let ring_mask:Vec<bool>=inner_mask.iter().map(|&b|!b).collect();
        let structural=prepared.plan.structural_steps();
        let edge_modes=modes.iter().filter(|q|ring.iter().any(|r|charge_minus(r,q).is_ok_and(|s|structural.contains(&s)))).count();
        // Near modes: the centre, every lattice mode one kernel step away and
        // the explicit set (by construction where the state has weight).
        let mut near=vec![centre];
        for q in frame.steps.iter().chain(explicit){if let Some(&q)=index.get(q){if !near.contains(&q){near.push(q);}}}
        let mut pairs=Vec::new();let mut frames=near.clone();
        for &a in &near{
            let q=&modes[a];
            for(s,step)in frame.steps.iter().enumerate(){
                if let Some(&b)=index.get(&charge_minus(q,step)?){pairs.push((a,b,s));if !frames.contains(&b){frames.push(b);}}
            }
        }
        let canonical:Vec<bool>=outer.canonical[..count].to_vec();
        let count=near.len();
        let near_canonical:Vec<usize>=(0..count).filter(|&slot|canonical[near[slot]]).collect();
        let outer_modes:Vec<usize>=frames.iter().copied().filter(|&q|!near.contains(&q)&&canonical[q]).collect();let outer_count=outer_modes.len();
        Ok(Self{prepared,n,d,modes:modes.to_vec(),nu,centre,index,mirror,ring,ring_nu,outer,inner_mask,ring_mask,edge_modes,frame,
            near,pairs,frames,outer_modes,canonical,near_canonical,partner:OnceLock::new(),basis_coordinates:vec![Vec::new();count],image_coordinates:vec![Vec::new();count],
            outer_basis:vec![Vec::new();outer_count],outer_image:vec![Vec::new();outer_count],columns:0,
            work:ErrorWork::default()})
    }
    /// The frame's kernels.
    pub fn kernels(&self)->Result<&'p SwKernels,String>{self.frame.kernels()}
    /// The exact eigenbasis of L0' (the frame's; the fast part's
    /// coordinates; the ring's block solves): request-level, built by the
    /// pipeline for every request (`pipeline::solve`).
    pub fn basis(&self)->Result<&'p BlockEigenbasis,String>{self.prepared.eigenbasis()}
    /// The flip partner of each eigen-coordinate (`BlockEigenbasis::partner`).
    fn partner(&self)->Result<&[usize],String>{
        let basis=self.basis()?;Ok(self.partner.get_or_init(||(0..self.d).map(|j|basis.partner(j)).collect()))
    }
    /// The near modes' second-order block, the frame's: every near mode
    /// carries every step's second-order term (the infinite lattice's
    /// averaging, the edge modes' outward paths included), one
    /// flip-invariant eigensystem. Measured against per-neighbour classes
    /// (9-19 complex eigensystems, 10-18 s on three/four lasers): the same
    /// estimate to 0.5% on every two/three/four-laser checkpoint, the build
    /// 1.0-1.2 s.
    pub fn class(&self)->Result<&'p BlockSystem,String>{self.frame.class()}
    pub fn dimension(&self)->usize{self.modes.len()*self.d}
    /// Process the basis columns not seen yet (columns are append-only):
    /// every new column and its image, transformed together.
    pub fn add_columns(&mut self,basis:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        let clock=Instant::now();
        if basis.len()<=self.columns{return Ok(());}
        let d=self.d;let first=self.columns;
        let mut fields:Vec<&[C]>=Vec::with_capacity(2*(basis.len()-first));
        for j in first..basis.len(){
            if basis[j].len()!=self.dimension()||images[j].len()!=basis[j].len(){return Err("Estimator: basis column dimensions differ from the lattice".into());}
            fields.push(&basis[j]);fields.push(&images[j]);
        }
        let(z,o)=self.coordinates(&fields)?;
        let(near,outer)=(self.near.len(),self.outer_modes.len());
        for i in 0..basis.len()-first{
            let(w,a)=(2*i,2*i+1);
            for &slot in &self.near_canonical{
                self.basis_coordinates[slot].extend_from_slice(&z[(w*near+slot)*d..(w*near+slot+1)*d]);
                self.image_coordinates[slot].extend_from_slice(&z[(a*near+slot)*d..(a*near+slot+1)*d]);}
            for slot in 0..outer{
                self.outer_basis[slot].extend_from_slice(&o[(w*outer+slot)*d..(w*outer+slot+1)*d]);
                self.outer_image[slot].extend_from_slice(&o[(a*outer+slot)*d..(a*outer+slot+1)*d]);}
        }
        let added=basis.len()-first;self.columns=basis.len();self.work.columns+=added;
        self.work.column_ms+=clock.elapsed().as_secs_f64()*1e3;
        Ok(())
    }
    /// For Hermitian-symmetric fields on P u Q: the class coordinates
    /// V_c^-1 E^-1 (I - S) v per canonical near mode, (field, near slot)
    /// consecutive (the other slots zero), and the coordinates E^-1 v per
    /// outer mode, (field, outer slot) consecutive. The coordinate changes
    /// run as large GEMMs over every field and mode (E^-1 streamed once per
    /// task).
    fn coordinates(&mut self,fields:&[&[C]])->Result<(Vec<C>,Vec<C>),String>{
        let d=self.d;let execution=Execution::new()?;let basis=self.basis()?;let partner=self.partner()?;
        let(count,near,outer)=(fields.len(),self.near.len(),self.outer_modes.len());
        // E^-1 on every (field, canonical frame mode).
        let canonical=&self.canonical;
        let computed:Vec<usize>=self.frames.iter().copied().filter(|&q|canonical[q]).collect();
        let items:Vec<(usize,usize)>=(0..count).flat_map(|f|computed.iter().map(move|&q|(f,q))).collect();
        let chunks:Vec<Vec<(usize,usize)>>=items.chunks(BLOCKS_PER_TASK).map(|c|c.to_vec()).collect();
        let parts=execution.map(chunks.clone(),|chunk|{let blocks:Vec<&[C]>=chunk.iter().map(|&(f,q)|&fields[f][q*d..(q+1)*d]).collect();basis.coefficients(&blocks)});
        let modes=self.modes.len();let mut c=vec![ZERO;count*modes*d];
        for(chunk,part)in chunks.iter().zip(parts){let part=part?;for(i,&(f,q))in chunk.iter().enumerate(){c[(f*modes+q)*d..(f*modes+q+1)*d].copy_from_slice(&part[i*d..(i+1)*d]);}}
        // The mirrored frame modes: c_-q[i] = conj c_q[partner i].
        for f in 0..count{for &q in self.frames.iter().filter(|&&q|!canonical[q]){
            let source=c[(f*modes+self.mirror[q])*d..(f*modes+self.mirror[q]+1)*d].to_vec();
            for(i,v)in c[(f*modes+q)*d..(f*modes+q+1)*d].iter_mut().enumerate(){*v=conj(source[partner[i]]);}}}
        // (I - S) into the canonical near modes.
        let slot_of:Vec<usize>=(0..modes).map(|q|self.near.iter().position(|&x|x==q).unwrap_or(usize::MAX)).collect();
        let(kernels,c_ref)=(self.kernels()?,&c);
        let rows=execution.map((0..count).collect(),|f|{
            let mut t=vec![ZERO;near*d];
            for &slot in &self.near_canonical{let a=self.near[slot];t[slot*d..(slot+1)*d].copy_from_slice(&c_ref[(f*modes+a)*d..(f*modes+a+1)*d]);}
            for &(a,b,s)in self.pairs.iter().filter(|p|canonical[p.0]){let slot=slot_of[a];kernels.subtract_kernel(s,&c_ref[(f*modes+b)*d..(f*modes+b+1)*d],&mut t[slot*d..(slot+1)*d]);}
            t});
        let t:Vec<C>=rows.concat();
        // V^-1 over every (field, canonical near slot).
        let(class,lapack)=(self.class()?,&self.prepared.lapack);
        let items:Vec<usize>=(0..count).flat_map(|f|self.near_canonical.iter().map(move|&slot|f*near+slot)).collect();
        let tasks:Vec<Vec<usize>>=items.chunks(BLOCKS_PER_TASK).map(|ch|ch.to_vec()).collect();
        let t=&t;
        let parts=execution.map(tasks.clone(),|items|->Result<Vec<C>,String>{
            let m=items.len();let mut input=vec![ZERO;d*m];for(i,&item)in items.iter().enumerate(){input[i*d..(i+1)*d].copy_from_slice(&t[item*d..(item+1)*d]);}
            let mut out=vec![ZERO;d*m];lapack.complex_gemm().multiply(d,m,d,ONE,&class.inverse,d,&input,d,ZERO,&mut out,d)?;Ok(out)});
        let mut z=vec![ZERO;count*near*d];
        for(items,part)in tasks.iter().zip(parts){let part=part?;for(i,&item)in items.iter().enumerate(){z[item*d..(item+1)*d].copy_from_slice(&part[i*d..(i+1)*d]);}}
        let mut o=vec![ZERO;count*outer*d];
        for f in 0..count{for(slot,&q)in self.outer_modes.iter().enumerate(){o[(f*outer+slot)*d..(f*outer+slot+1)*d].copy_from_slice(&c[(f*modes+q)*d..(f*modes+q+1)*d]);}}
        self.work.frame_blocks+=count*(computed.len()+self.near_canonical.len());
        Ok((z,o))
    }
    /// The ring slaved to a field x on P u Q at the shift z,
    /// y = (z - D_R)^-1 (C x)|_R; None when x feeds no ring charge.
    fn slaved(&mut self,x:&[C],shift:C)->Result<Option<Vec<C>>,String>{
        let d=self.d;let(inner,count)=(self.modes.len(),self.ring.len());
        if count==0{return Ok(None);}
        let(xa,xb)=split(x,&self.mirror,self.n);
        let outflow=|this:&mut Self,part:&[C]|->Result<Vec<C>,String>{
            if norm(part)==0.{return Ok(vec![ZERO;count*d]);}
            let mut field=vec![ZERO;(inner+count)*d];field[..inner*d].copy_from_slice(part);let mut out=vec![ZERO;field.len()];
            this.outer.apply_masked(&field,&mut out,Parts::Full,Some(Mask{targets:&this.ring_mask,sources:&this.inner_mask}))?;this.work.block_operations+=count;
            Ok(out[inner*d..].to_vec())};
        let(oa,ob)=(outflow(self,&xa)?,outflow(self,&xb)?);let source=join(&oa,&ob);let size=norm(&source);
        if size==0.{return Ok(None);}
        // (D_R - z) y = source, then -y.
        // The eigenbasis's block solves, refined for the ring blocks' frequency term.
        let basis=self.prepared.eigenbasis()?;
        let mut y=source.clone();basis.solve_blocks(&self.ring_nu,shift,&mut y)?;self.work.block_solves+=count;
        for _ in 0..RING_REFINEMENTS{
            let mut r=source.clone();
            let applied=self.ring_parts(&y,Parts::Diagonal,true)?;
            for((t,a),v)in r.iter_mut().zip(&applied).zip(&y){let w=mul(shift,*v);t.re-=a.re-w.re;t.im-=a.im-w.im;}
            if norm(&r)<=RING_RESIDUAL*size{break;}
            basis.solve_blocks(&self.ring_nu,shift,&mut r)?;self.work.block_solves+=count;
            for(a,b)in y.iter_mut().zip(&r){add(a,*b);}
        }
        for v in &mut y{v.re=-v.re;v.im=-v.im;}
        Ok(Some(y))
    }
    /// The ring's feedback on P u Q of a field x at the shift z:
    /// (B (z - D_R)^-1 C x)|_{P u Q}.
    pub fn ring_feedback(&mut self,x:&[C],shift:C)->Result<Vec<C>,String>{
        match self.slaved(x,shift)?{Some(y)=>self.ring_parts(&y,Parts::Full,false),None=>Ok(vec![ZERO;self.dimension()])}
    }
    /// Per ring charge r (in `ring` order), the feedback onto P u Q of its
    /// slaved amplitude alone (the mirror with it), |B y_r|, for a field x at
    /// the shift z.
    pub fn ring_attribution(&mut self,x:&[C],shift:C)->Result<Vec<f64>,String>{
        let(d,inner,count)=(self.d,self.modes.len(),self.ring.len());
        let Some(y)=self.slaved(x,shift)? else{return Ok(vec![0.;count]);};
        let mut per=Vec::with_capacity(count);
        for r in 0..count{let mut yr=vec![ZERO;count*d];yr[r*d..(r+1)*d].copy_from_slice(&y[r*d..(r+1)*d]);
            let m=self.outer.mirror[inner+r]-inner;if m!=r{yr[m*d..(m+1)*d].copy_from_slice(&y[m*d..(m+1)*d]);}
            per.push(norm(&self.ring_parts(&yr,Parts::Full,false)?));}
        Ok(per)
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
