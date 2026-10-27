//! The steady error by GCRO, recycling the space.
//!
//! At t -> infinity the error is e = -A^# f with f = R u_inf + B y_inf, the
//! residual of the reduced steady state plus the slaved ring's feedback
//! (R2: exact, A (W X u_inf) = R u_inf), read as the centre populations of
//! the bordered solution M y = f, M = A + u tau0^T (f has no centre trace).
//! GCRO minimizes |f - M D x| over the directions D whose images M D are
//! known:
//! - every basis column w_j, M w_j = A w_j + u tau0(w_j) from the images;
//! - every earlier augmentation direction with its image, kept across
//!   checkpoints: the first one is the pipeline's own bordered solve of the
//!   first checkpoint's f (`SteadySolves`), the later ones preconditioned
//!   residuals (the shared Floquet preconditioner of the steady shift).
//! The images are orthonormalized as they arrive (M D = Q R, CGS2), so a
//! checkpoint costs the projections of f and, usually, a few steps. It
//! stops when the residual is below STEADY_RESIDUAL |f| and the readout has
//! settled (two consecutive changes below STABLE of its size).
use super::super::*;
use super::super::estimate_api::SteadySolves;
use super::super::operator::LatticeOperator;
use super::super::preconditioner::{FloquetBatch,Preconditioner,Preconditioning,Shift};
use super::super::floquet::Reach;

const STEADY_RESIDUAL:f64=1e-4;
const STABLE:f64=0.1;
const MAXIMUM_STEPS:usize=60;
/// An image whose part outside the earlier ones is below this fraction of
/// its size adds no direction.
const DEPENDENT:f64=1e-10;

#[derive(Clone,Debug,Default)]
pub struct SteadyWork{pub applications:usize,pub preconditions:usize,pub pipeline_solves:usize,pub inner_steps:usize,pub steps:usize}

pub struct SteadyRecycler<'p>{
    op:LatticeOperator<'p>,batch:FloquetBatch<'p>,
    n:usize,watch:Vec<usize>,
    /// Orthonormal images and the triangular factor (column c has c + 1
    /// entries), in the order the directions arrived.
    q:Vec<Vec<C>>,r:Vec<Vec<f64>>,
    /// Augmentation directions so far.
    augmentations:usize,
    /// Centre populations of every direction.
    populations:Vec<Vec<f64>>,
    columns:usize,
    pub work:SteadyWork,
}

fn dot(a:&[C],b:&[C])->f64{a.iter().zip(b).map(|(x,y)|x.re*y.re+x.im*y.im).sum()}
fn axpy(a:f64,x:&[C],y:&mut[C]){for(t,v)in y.iter_mut().zip(x){t.re+=a*v.re;t.im+=a*v.im;}}

impl<'p> SteadyRecycler<'p>{
    pub fn new(prepared:&'p LatticePrepared,modes:&[Charge])->Result<Self,String>{
        let op=LatticeOperator::new(prepared,modes)?;
        let batch=Preconditioning::new(prepared)?.batch(&op,Shift::Steady,Some(Reach::All))?;
        let n=op.n;let c0=op.centre*n*n;let watch=(0..n).map(|i|c0+i*n+i).collect();
        Ok(Self{op,batch,n,watch,q:Vec::new(),r:Vec::new(),augmentations:0,populations:Vec::new(),columns:0,work:SteadyWork::default()})
    }
    fn trace_term(&self,x:&[C],y:&mut[C]){
        let t=self.watch.iter().fold(ZERO,|s,&k|C{re:s.re+x[k].re,im:s.im+x[k].im});
        for &k in &self.watch{add(&mut y[k],scaled(t,1./self.n as f64));}
    }
    fn apply(&mut self,x:&[C])->Result<Vec<C>,String>{
        let mut y=vec![ZERO;x.len()];self.op.apply(x,&mut y)?;self.trace_term(x,&mut y);self.work.applications+=1;Ok(y)
    }
    fn centre_populations(&self,x:&[C])->Vec<f64>{self.watch.iter().map(|&k|x[k].re).collect()}
    /// Adds a direction with its image M x: the image's part outside the
    /// earlier ones joins Q (none when it is dependent).
    fn push(&mut self,field:&[C],mut image:Vec<C>)->bool{
        let size=dot(&image,&image).sqrt();
        let mut h=vec![0.;self.q.len()];
        for _ in 0..2{for(c,q)in self.q.iter().enumerate(){let a=dot(q,&image);h[c]+=a;axpy(-a,q,&mut image);}}
        let eta=dot(&image,&image).sqrt();
        if !(eta>DEPENDENT*size){return false;}
        for v in &mut image{v.re/=eta;v.im/=eta;}
        h.push(eta);self.q.push(image);self.r.push(h);
        self.populations.push(self.centre_populations(field));
        true
    }
    /// The solution's centre populations for the coordinates c of f on Q.
    fn readout(&self,c:&[f64])->Vec<f64>{
        let m=c.len();let mut x=c.to_vec();
        for j in(0..m).rev(){x[j]/=self.r[j][j];for i in 0..j{x[i]-=self.r[j][i]*x[j];}}
        let mut out=vec![0.;self.n];for(j,&a)in x.iter().enumerate(){for(o,&p)in out.iter_mut().zip(&self.populations[j]){*o+=a*p;}}
        out
    }
    /// The centre populations of the steady error -A^# f (f Hermitian-symmetric).
    pub fn error(&mut self,basis:&[Vec<C>],images:&[Vec<C>],f:&[C],steady:&mut dyn SteadySolves)->Result<Vec<f64>,String>{
        for j in self.columns..basis.len(){
            let mut image=images[j].clone();self.trace_term(&basis[j],&mut image);
            self.push(&basis[j],image);
        }
        self.columns=basis.len();
        let fsize=dot(f,f).sqrt();if fsize==0.{return Ok(vec![0.;self.n]);}
        let mut res=f.to_vec();let mut c=Vec::with_capacity(self.q.len());
        for q in &self.q{let a=dot(q,&res);c.push(a);axpy(-a,q,&mut res);}
        if self.augmentations==0{
            let(y,steps)=steady.solve(f,STEADY_RESIDUAL)?;self.work.pipeline_solves+=1;self.work.inner_steps+=steps;
            let image=self.apply(&y)?;self.augmentations+=1;
            if self.push(&y,image){let q=self.q.last().unwrap();let a=dot(q,&res);c.push(a);axpy(-a,q,&mut res);}
        }
        let mut history=vec![self.readout(&c)];
        let mut steps=0;
        loop{
            let rn=dot(&res,&res).sqrt();
            if rn<=1e-13*fsize{break;}
            if history.len()>=3&&rn<=STEADY_RESIDUAL*fsize{
                let h=&history;let last=&h[h.len()-1];let size=last.iter().fold(0_f64,|a,&b|a.max(b.abs()));
                let change=|a:&[f64],b:&[f64]|a.iter().zip(b).fold(0_f64,|m,(x,y)|m.max((x-y).abs()));
                if change(last,&h[h.len()-2])<=STABLE*size&&change(&h[h.len()-2],&h[h.len()-3])<=STABLE*size{break;}
            }
            if steps>=MAXIMUM_STEPS{break;}
            let scaled_res:Vec<C>=res.iter().map(|v|C{re:v.re/rn,im:v.im/rn}).collect();
            let mut z=vec![ZERO;res.len()];self.batch.apply(&scaled_res,&mut z)?;self.work.preconditions+=1;
            let image=self.apply(&z)?;self.augmentations+=1;
            if self.push(&z,image){let q=self.q.last().unwrap();let a=dot(q,&res);c.push(a);axpy(-a,q,&mut res);}
            steps+=1;history.push(self.readout(&c));
        }
        self.work.steps+=steps;
        Ok(history.last().unwrap().iter().map(|v|-v).collect())
    }
    /// Directions with independent images (basis columns and augmentations).
    pub fn directions(&self)->usize{self.q.len()}
}
