//! The lattice P u Q (the explicit charges, then their ring) and its solves
//! at the poles of the space: (A - xi) w = f for a real or complex pole xi,
//! and the bordered steady system (A + u tau0^T) w = f (u = I/n on the
//! centre, tau0 its trace) for the residue at 0. Every column of the
//! rational Krylov space is such a solve, to the inner tolerance.
//!
//! Inner solves: GMRES on all of P u Q with the Floquet-frame preconditioner
//! on every block (`preconditioner`), one batch per pole, the kernels on
//! every mode (a pole serves few solves, too few to probe the reaches).
//! Measured against eliminating the ring Q exactly at the pole (GMRES on P
//! with the Schur complement A_PP + A_PQ (xi - A_QQ)^-1 A_QP, the ring by
//! batched eigenbasis block solves): on two, three and four lasers + tweezer
//! the whole-lattice solves took 0.25, 0.20 and 0.15 of the time with fewer
//! inner steps (119 / 108, 270 / 342, 94 / 121), because an exact 1024 x 1024
//! block solve costs about sixteen block applications and the Floquet
//! frames precondition the P-Q edges the Schur complement leaves to GMRES.
//!
//! Complex poles act on the real form (a pair of Hermitian-symmetric fields
//! w_1 + i w_2).
use super::*;
use super::estimate_api::{LatticeSets,SteadySolves};
use super::operator::LatticeOperator;
use super::preconditioner::{FloquetBatch,Preconditioner,Preconditioning,Shift};
use super::floquet::Reach;
use super::krylov::{self,KrylovOptions,KrylovTiming,System};

/// A pole of the rational Krylov space: the bordered steady system (the
/// residue at 0), a real or a complex shift.
#[derive(Clone,Copy,Debug)]
pub enum Pole{Zero,Real(f64),Complex(C)}
impl PartialEq for Pole{
    fn eq(&self,other:&Self)->bool{
        match(self,other){(Pole::Zero,Pole::Zero)=>true,(Pole::Real(a),Pole::Real(b))=>a==b,(Pole::Complex(a),Pole::Complex(b))=>a.re==b.re&&a.im==b.im,_=>false}
    }
}
impl Pole{
    pub fn shift(&self)->Shift{match *self{Pole::Zero=>Shift::Steady,Pole::Real(s)=>Shift::Real(s),Pole::Complex(s)=>Shift::Complex(s)}}
    pub fn sigma(&self)->C{self.shift().sigma()}
    pub fn copies(&self)->usize{self.shift().copies()}
}
impl std::fmt::Display for Pole{
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
        match self{Pole::Zero=>write!(f,"0 (bordered)"),Pole::Real(s)=>write!(f,"{s:.4e}"),Pole::Complex(s)=>write!(f,"{:.4e}{:+.4e}i",s.re,s.im)}
    }
}

/// Work of the solves, per lattice block (one n x n block of an operator or
/// preconditioner application = 1).
#[derive(Clone,Debug,Default)]
pub struct InnerWork{
    pub solves:usize,pub steps:usize,
    pub applied_blocks:usize,pub preconditioned_blocks:usize,
    /// Preconditioner batches built (one per pole) and their time.
    pub batches:usize,pub batch_ms:f64,
    /// Complex multiply-adds of one operator application (blocks and edges,
    /// fold rests).
    pub operator_work:[usize;2],
    pub milliseconds:f64,
    pub timing:KrylovTiming,
}
impl InnerWork{pub fn block_operations(&self)->usize{self.applied_blocks+self.preconditioned_blocks}}

/// Bytes of the steady solve's inner directions kept for the estimate.
const RECORDED_BYTES:usize=1<<28;

/// The lattice of a run with its preconditioner batches per pole.
pub struct Lattice<'a>{
    pub sets:LatticeSets,pub modes:Vec<Charge>,
    op:LatticeOperator<'a>,
    preconditioning:Preconditioning<'a>,
    batches:Vec<(Pole,FloquetBatch<'a>)>,
    work:InnerWork,
    /// The inner directions z and their images A z of the first steady solve
    /// (the pipeline's pole-0 column; append-only), and whether it ran.
    directions:Vec<Vec<C>>,images:Vec<Vec<C>>,recorded:bool,
}
impl<'a> Lattice<'a>{
    pub fn new(prepared:&'a LatticePrepared,sets:LatticeSets)->Result<Self,String>{
        if sets.centre()!=Some(0){return Err("The explicit set must start with the zero charge".into());}
        let modes=sets.modes();let op=LatticeOperator::new(prepared,&modes)?;let operator_work=op.work_split();
        Ok(Self{sets,modes,op,preconditioning:Preconditioning::new(prepared)?,batches:Vec::new(),work:InnerWork{operator_work,..InnerWork::default()},directions:Vec::new(),images:Vec::new(),recorded:false})
    }
    /// The preconditioner batch of a pole (built at its first use).
    fn batch(&mut self,pole:Pole)->Result<usize,String>{
        if let Some(i)=self.batches.iter().position(|(p,_)|*p==pole){return Ok(i);}
        let start=Instant::now();
        let batch=self.preconditioning.batch(&self.op,pole.shift(),Some(Reach::All))?;self.batches.push((pole,batch));
        self.work.batches+=1;self.work.batch_ms+=start.elapsed().as_secs_f64()*1e3;Ok(self.batches.len()-1)
    }
    pub fn operator(&self)->&LatticeOperator<'a>{&self.op}
    /// y = A x.
    pub fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{self.op.apply(x,y)}
    pub fn work(&self)->&InnerWork{&self.work}
    /// w = (A - xi)^-1 f at a real or complex pole (f and w: one field, or
    /// the real form's two), or (A + u tau0^T)^-1 f at Pole::Zero; with the
    /// inner steps and the true relative residual.
    pub fn solve(&mut self,pole:Pole,f:&[Vec<C>],tolerance:f64)->Result<(Vec<Vec<C>>,usize,f64),String>{
        let clock=Instant::now();let copies=pole.copies();
        let i=self.batch(pole)?;
        let Self{op,batches,work,recorded,directions,images,..}=self;let batch=&mut batches[i].1;
        let count=op.modes.len();let size=op.dimension();let b:Vec<C>=f.concat();
        let mut system=System::new(op,batch,pole.shift())?;
        let record=pole==Pole::Zero&&!*recorded;
        if record{system.recorded=Some((Vec::new(),RECORDED_BYTES/(2*size*std::mem::size_of::<C>()).max(1)));}
        let solve=krylov::gmres(&mut system,&b,None,&KrylovOptions::new(tolerance))?;
        if record{*recorded=true;if let Some((list,_))=system.recorded.take(){for(z,az)in list{directions.push(z);images.push(az);}}}
        let(applications,preconditions,timing)=(system.applications,system.preconditions,system.timing.clone());
        let operator_work=system.op.work;drop(system);
        batch.record(solve.steps,solve.residual,tolerance,operator_work);
        work.applied_blocks+=applications*count;work.preconditioned_blocks+=preconditions*count*copies;work.timing.add(&timing);
        work.solves+=1;work.steps+=solve.steps;work.milliseconds+=clock.elapsed().as_secs_f64()*1e3;
        Ok(((0..copies).map(|c|solve.x[c*size..(c+1)*size].to_vec()).collect(),solve.steps,solve.residual))
    }
}
impl SteadySolves for Lattice<'_>{
    fn solve(&mut self,f:&[C],tolerance:f64)->Result<(Vec<C>,usize),String>{
        let(mut w,steps,_)=Lattice::solve(self,Pole::Zero,&[f.to_vec()],tolerance)?;Ok((w.remove(0),steps))
    }
    fn directions(&self)->(&[Vec<C>],&[Vec<C>]){(&self.directions,&self.images)}
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    use super::super::operator::symmetric_field;
    #[test]
    fn lattice_pole_solves_are_the_lattice_resolvent_columns(){
        // P = {0, +-s} with its ring on the fixture, against a dense LU of the
        // lattice P u Q: a real pole, a complex pole (real form) and the
        // bordered steady system.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;
        let s=p.plan.steps[p.plan.explicit()[0]].charge.clone();
        let mut explicit=vec![p.generator.zero.clone(),s.clone(),opposite(&s)];
        explicit.sort_by_key(|q|(q.iter().map(|&k|k.unsigned_abs()).sum::<u32>(),q.clone()));
        let ring=super::super::estimate_api::ring(&p,&explicit).unwrap();
        let mut lattice=Lattice::new(&p,LatticeSets{explicit,ring}).unwrap();
        let size=lattice.operator().dimension();let dense=lattice.operator().dense();let centre=lattice.operator().centre;
        let x=symmetric_field(lattice.operator(),1);
        for pole in[Pole::Real(0.7),Pole::Complex(C{re:0.4,im:2.5}),Pole::Zero]{
            let copies=pole.copies();
            let mut f=vec![x.clone()];if copies==2{f.push(symmetric_field(lattice.operator(),2));}
            let(w,_,residual)=lattice.solve(pole,&f,1e-12).unwrap();assert!(residual<=1e-12);
            let sigma=pole.sigma();let mut m=dense.clone();
            if pole==Pole::Zero{for i in 0..n{for j in 0..n{m[(centre*d+i*n+i)*size+centre*d+j*n+j].re+=1./n as f64;}}}
            else{for r in 0..size{m[r*size+r].re-=sigma.re;m[r*size+r].im-=sigma.im;}}
            let mut column=vec![ZERO;size*size];for r in 0..size{for c in 0..size{column[r+size*c]=m[r*size+c];}}
            let(lu,piv)=p.lapack.complex_factor(&column,size).unwrap();
            let join=|v:&[Vec<C>]|->Vec<C>{if copies==2{(0..size).map(|k|C{re:v[0][k].re-v[1][k].im,im:v[0][k].im+v[1][k].re}).collect()}else{v[0].clone()}};
            let exact=p.lapack.complex_solve_matrix_factored(&lu,&piv,&join(&f),size).unwrap();
            let got=join(&w);let scale=exact.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            let error=got.iter().zip(&exact).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
            assert!(error<1e-9*scale,"{pole}: {error:e} of {scale:e}");
        }
    }
}
