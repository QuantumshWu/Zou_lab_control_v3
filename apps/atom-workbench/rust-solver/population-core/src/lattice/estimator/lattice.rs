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
use super::super::operator::{LatticeOperator,Parts,Restriction};
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
/// Blocks per task of the coordinate changes (E^-1 and V^-1 streamed once
/// per task).
const BLOCKS_PER_TASK:usize=64;

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
    /// P u Q u R restricted to its pieces: P u Q -> R (the outflow), R's
    /// diagonal blocks, R -> P u Q (the feedback); the modes of P u Q with a
    /// ring neighbour.
    pub ring:Vec<Charge>,ring_nu:Vec<f64>,
    outflow_op:LatticeOperator<'p>,ring_op:LatticeOperator<'p>,feedback_op:LatticeOperator<'p>,ring_mirror:Vec<usize>,edge_modes:usize,
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
        let inner_mask:Vec<bool>=(0..all.len()).map(|q|q<modes.len()).collect();let ring_mask:Vec<bool>=inner_mask.iter().map(|&b|!b).collect();
        let piece=|rows:&[bool],columns:&[bool],diagonal:bool|LatticeOperator::restricted(prepared,&all,Some(&Restriction{rows:rows.to_vec(),columns:columns.to_vec(),diagonal}));
        let(outflow_op,ring_op,feedback_op)=(piece(&ring_mask,&inner_mask,false)?,piece(&ring_mask,&ring_mask,true)?,piece(&inner_mask,&ring_mask,false)?);
        let ring_mirror:Vec<usize>=(0..ring.len()).map(|r|outflow_op.mirror[modes.len()+r]-modes.len()).collect();
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
        Ok(Self{prepared,basis,n,d,modes:modes.to_vec(),nu,centre,index,mirror,ring,ring_nu,outflow_op,ring_op,feedback_op,ring_mirror,edge_modes,kernels,
            near,pairs,frames,outer_modes,classes,class_of,basis_coordinates:vec![Vec::new();count],image_coordinates:vec![Vec::new();count],
            outer_basis:vec![Vec::new();outer_count],outer_image:vec![Vec::new();outer_count],columns:0,ring_in_fast:false,outer_bare:true,
            work:ErrorWork::default(),build_ms:[kernel_ms,class_ms]})
    }
    pub fn dimension(&self)->usize{self.modes.len()*self.d}
    /// Process the basis columns not seen yet (columns are append-only).
    pub fn add_columns(&mut self,basis:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        let clock=Instant::now();
        if basis.len()<=self.columns{return Ok(());}
        let d=self.d;let first=self.columns;
        // Every new column and its image (with the ring's feedback where the
        // fast part carries it), transformed together.
        let mut fields:Vec<Vec<C>>=Vec::with_capacity(2*(basis.len()-first));
        for j in first..basis.len(){
            if basis[j].len()!=self.dimension()||images[j].len()!=basis[j].len(){return Err("Estimator: basis column dimensions differ from the lattice".into());}
            fields.push(basis[j].clone());
            if self.ring_in_fast{let g=self.ring_feedback(&basis[j],ZERO)?;fields.push(images[j].iter().zip(&g).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect());}
            else{fields.push(images[j].clone());}
        }
        let refs:Vec<&[C]>=fields.iter().map(|f|&f[..]).collect();
        let(z,o)=self.coordinates(&refs)?;
        let(near,outer)=(self.near.len(),self.outer_modes.len());
        for(i,_)in(first..basis.len()).enumerate(){
            let(w,a)=(2*i,2*i+1);
            for slot in 0..near{
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
    /// For fields on P u Q: the class coordinates V_c^-1 E^-1 (I - S) v per
    /// near mode, (field, near slot) consecutive, and the coordinates E^-1 v
    /// per outer mode, (field, outer slot) consecutive. The coordinate
    /// changes run as large GEMMs over every field and mode (E^-1 streamed
    /// once per task).
    fn coordinates(&mut self,fields:&[&[C]])->Result<(Vec<C>,Vec<C>),String>{
        let d=self.d;let execution=Execution::new()?;let basis=self.basis;
        let(count,frames,near,outer)=(fields.len(),self.frames.len(),self.near.len(),self.outer_modes.len());
        // E^-1 on every (field, frame mode).
        let items:Vec<(usize,usize)>=(0..count).flat_map(|f|self.frames.iter().map(move|&q|(f,q))).collect();
        let chunks:Vec<Vec<(usize,usize)>>=items.chunks(BLOCKS_PER_TASK).map(|c|c.to_vec()).collect();
        let parts=execution.map(chunks.clone(),|chunk|{let blocks:Vec<&[C]>=chunk.iter().map(|&(f,q)|&fields[f][q*d..(q+1)*d]).collect();basis.coefficients(&blocks)});
        let modes=self.modes.len();let mut c=vec![ZERO;count*modes*d];
        for(chunk,part)in chunks.iter().zip(parts){let part=part?;for(i,&(f,q))in chunk.iter().enumerate(){c[(f*modes+q)*d..(f*modes+q+1)*d].copy_from_slice(&part[i*d..(i+1)*d]);}}
        let _=frames;
        // (I - S) into the near modes.
        let slot_of:Vec<usize>=(0..modes).map(|q|self.near.iter().position(|&x|x==q).unwrap_or(usize::MAX)).collect();
        let mut t=vec![ZERO;count*near*d];
        for f in 0..count{
            for(slot,&a)in self.near.iter().enumerate(){t[(f*near+slot)*d..(f*near+slot+1)*d].copy_from_slice(&c[(f*modes+a)*d..(f*modes+a+1)*d]);}
            for &(a,b,s)in &self.pairs{let slot=slot_of[a];let source=&c[(f*modes+b)*d..(f*modes+b+1)*d];
                self.kernels.subtract_kernel(s,source,&mut t[(f*near+slot)*d..(f*near+slot+1)*d]);}
        }
        // V_c^-1 per class over every (field, near slot) of the class.
        let(classes,class_of,lapack)=(&self.classes,&self.class_of,&self.prepared.lapack);
        let mut groups:Vec<(usize,Vec<usize>)>=Vec::new();
        for f in 0..count{for(slot,&a)in self.near.iter().enumerate(){let c=class_of[a];let item=f*near+slot;
            match groups.iter_mut().find(|g|g.0==c){Some(g)=>g.1.push(item),None=>groups.push((c,vec![item]))}}}
        let tasks:Vec<(usize,Vec<usize>)>=groups.iter().flat_map(|(c,items)|items.chunks(BLOCKS_PER_TASK).map(move|ch|(*c,ch.to_vec()))).collect();
        let t=&t;
        let parts=execution.map(tasks.clone(),|(c,items)|->Result<Vec<C>,String>{
            let m=items.len();let mut input=vec![ZERO;d*m];for(i,&item)in items.iter().enumerate(){input[i*d..(i+1)*d].copy_from_slice(&t[item*d..(item+1)*d]);}
            let mut out=vec![ZERO;d*m];lapack.complex_gemm().multiply(d,m,d,ONE,&classes[c].inverse,d,&input,d,ZERO,&mut out,d)?;Ok(out)});
        let mut z=vec![ZERO;count*near*d];
        for((_,items),part)in tasks.iter().zip(parts){let part=part?;for(i,&item)in items.iter().enumerate(){z[item*d..(item+1)*d].copy_from_slice(&part[i*d..(i+1)*d]);}}
        let mut o=vec![ZERO;count*outer*d];
        for f in 0..count{for(slot,&q)in self.outer_modes.iter().enumerate(){o[(f*outer+slot)*d..(f*outer+slot+1)*d].copy_from_slice(&c[(f*modes+q)*d..(f*modes+q+1)*d]);}}
        self.work.frame_blocks+=count*(self.frames.len()+near);
        Ok((z,o))
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
            this.outflow_op.apply(&field,&mut out)?;this.work.block_operations+=count;
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
        let(ya,yb)=split(y,&self.ring_mirror,self.n);
        let mut results=Vec::new();
        for part in[&ya,&yb]{
            if norm(part)==0.{results.push(vec![ZERO;if ring_rows{count*d}else{inner*d}]);continue;}
            let mut field=vec![ZERO;(inner+count)*d];field[inner*d..].copy_from_slice(part);let mut out=vec![ZERO;field.len()];
            if ring_rows{self.ring_op.apply_parts(&field,&mut out,parts)?;self.work.block_operations+=count;}
            else{self.feedback_op.apply_parts(&field,&mut out,parts)?;self.work.block_operations+=self.edge_modes;}
            results.push(if ring_rows{out[inner*d..].to_vec()}else{out[..inner*d].to_vec()});
        }
        Ok(join(&results[0],&results[1]))
    }
}
