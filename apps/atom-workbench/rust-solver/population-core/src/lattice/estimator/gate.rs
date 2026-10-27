//! The gate in front of the full estimate: the change of the outputs since
//! the previous checkpoint.
//!
//! The outputs of a checkpoint (trajectory, horizons, steady state) are read
//! from the reduced model with the basis columns' population rows alone. If
//! they moved by delta since the previous checkpoint, then the errors e_k,
//! e_k-1 of the two satisfy delta <= e_k + e_k-1, so delta > 4 eps means one
//! of them exceeds 2 eps: the full estimate is skipped (and delta returned)
//! unless the previous full estimate was itself within GATE_FACTOR eps.
//! The gate never accepts: a skipped checkpoint reports delta > 4 eps. Its
//! cost is one checkpoint late at worst, when a jump from a poor checkpoint
//! lands directly within the tolerance. The estimate's own per-column work
//! is deferred while the gate is closed, not saved.
//!
//! A residual-norm bound is no gate: the fast components dominate the
//! residual's norm and would keep it closed; R1's dressed-block propagation
//! needs a d x d coordinate change per mode and Ritz mode, as much as the
//! full estimate.
use super::super::*;
use super::super::ledger::metric;
use crate::parallel::Execution;

/// The full estimate runs when the outputs moved by at most this times the
/// tolerance, or when the previous full estimate was within it.
pub const GATE_FACTOR:f64=4.;

/// The reduced model's outputs: populations per grid time, per horizon, and
/// the steady state.
#[derive(Clone,Debug,Default)]
pub struct Outputs{pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,pub steady:Option<Vec<f64>>}

pub struct OutputGate{
    n:usize,d:usize,nu:Vec<f64>,centre:usize,
    /// Per mode the population rows of the basis columns (column-major n x k).
    rows:Vec<Vec<C>>,columns:usize,
    previous:Option<Outputs>,
    /// The latest full estimate's largest bound.
    pub last_estimate:Option<f64>,
}

fn cexp(z:C,t:f64)->C{let m=(z.re*t).exp();if m==0.{return ZERO;}C{re:m*(z.im*t).cos(),im:m*(z.im*t).sin()}}

impl OutputGate{
    pub fn new(n:usize,nu:Vec<f64>,centre:usize)->Self{let modes=nu.len();Self{n,d:n*n,nu,centre,rows:vec![Vec::new();modes],columns:0,previous:None,last_estimate:None}}
    pub fn add_columns(&mut self,basis:&[Vec<C>]){
        let(n,d)=(self.n,self.d);
        for w in &basis[self.columns..]{for(q,rows)in self.rows.iter_mut().enumerate(){for i in 0..n{rows.push(w[q*d+i*n+i]);}}}
        self.columns=basis.len();
    }
    /// The outputs for the Ritz vectors B (column-major k x r), values theta
    /// and coefficients a, and the reduced steady state's coordinates c over
    /// the basis.
    pub fn outputs(&self,lapack:&Lapack,vectors:&[C],theta:&[C],coefficients:&[C],times:&[f64],horizons:&[f64],steady:Option<&[C]>)->Result<Outputs,String>{
        let(n,k,r)=(self.n,self.columns,theta.len());
        let mut ba=vec![ZERO;k*r];for m in 0..r{for i in 0..k{ba[i+k*m]=mul(vectors[i+k*m],coefficients[m]);}}
        let modes:Vec<usize>=(0..self.nu.len()).collect();
        let lab=Execution::new()?.map(modes,|q|->Result<Vec<C>,String>{
            let gemm=lapack.complex_gemm();
            let mut y=vec![ZERO;n*r];gemm.multiply(n,r,k,ONE,&self.rows[q],n,&ba,k,ZERO,&mut y,n)?;
            let e:Vec<C>=times.iter().flat_map(|&t|theta.iter().map(move|&z|cexp(z,t))).collect();
            let mut f=vec![ZERO;n*times.len()];gemm.multiply(n,times.len(),r,ONE,&y,n,&e,r,ZERO,&mut f,n)?;
            for(ti,&t)in times.iter().enumerate(){let ph=cexp(C{re:0.,im:self.nu[q]},t);for v in &mut f[n*ti..n*(ti+1)]{*v=mul(ph,*v);}}
            Ok(f)});
        let mut total=vec![ZERO;n*times.len()];for f in lab{for(a,b)in total.iter_mut().zip(f?){add(a,b);}}
        let trajectory=(0..times.len()).map(|ti|(0..n).map(|i|total[i+n*ti].re).collect()).collect();
        let centre=&self.rows[self.centre];
        let read=|c:&[C]|->Vec<f64>{(0..n).map(|i|(0..k).map(|j|mul(centre[i+n*j],c[j]).re).sum()).collect()};
        let horizons=horizons.iter().map(|&t|{let c:Vec<C>=(0..k).map(|i|(0..r).fold(ZERO,|s,m|{let v=mul(ba[i+k*m],cexp(theta[m],t));C{re:s.re+v.re,im:s.im+v.im}})).collect();read(&c)}).collect();
        Ok(Outputs{trajectory,horizons,steady:steady.map(read)})
    }
    /// The largest change of the outputs since the previous checkpoint (None
    /// at the first), and the outputs kept for the next.
    pub fn change(&mut self,outputs:Outputs,groups:&[Vec<usize>])->Option<f64>{
        let change=self.previous.as_ref().map(|p|{
            let pairs=p.trajectory.iter().zip(&outputs.trajectory).chain(p.horizons.iter().zip(&outputs.horizons)).chain(p.steady.iter().zip(&outputs.steady));
            pairs.map(|(a,b)|metric(a,b,groups)).fold(0.,f64::max)});
        self.previous=Some(outputs);
        change
    }
}
