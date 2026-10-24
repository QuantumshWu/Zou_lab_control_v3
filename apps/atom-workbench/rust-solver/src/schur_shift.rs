//! Experimental Fourier-block preconditioner. One real Schur decomposition
//! per complete request; no previous model or computed result is retained.
use crate::lapack::Lapack;
use std::{rc::Rc,cell::Cell,time::Instant};
type Sparse=Vec<Vec<(usize,f64)>>;
pub struct Shared {
    pub d:usize,t:Vec<f64>,q:Vec<f64>,qt:Vec<f64>,weights:Vec<f64>,
    pub setup_ms:f64,pub preparations:Cell<usize>,pub applications:Cell<usize>,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shifted_schur_blocks_match_original_normal_and_transpose_systems(){
        let lap=Lapack::load().unwrap();let d=4;let n=12;
        let rows=[[0.,2.,0.,3.],[0.,-2.,0.,-3.],[0.,0.,-1.,7.],[-1.5,1.5,-7.,-1.]];
        let base:Sparse=(0..d).map(|j|(0..d).filter_map(|i|(rows[i][j]!=0.).then_some((i,rows[i][j]))).collect()).collect();
        let shared=Rc::new(Shared::new(&lap,&base,2).unwrap());
        for weighted in [false,true]{
            let sign=if weighted{-1.}else{1.};let shift=if weighted{0.15}else{0.};let omega=11.;
            let mut a=vec![Vec::new();n];
            for part in 0..3{for j in 0..d{for &(i,v)in &base[j]{a[part*d+j].push((part*d+i,sign*v*if weighted{shared.weights[i]/shared.weights[j]}else{1.}));}if shift!=0.{a[part*d+j].push((part*d+j,shift));}}}
            for i in 0..d{a[2*d+i].push((d+i,sign*omega));a[d+i].push((2*d+i,-sign*omega));}
            if !weighted{for j in 0..d{a[j].retain(|(i,_)|*i!=0);if j<2{a[j].push((0,1.));}}}
            let mut scales=vec![0_f64;n];for col in &a{for &(i,v)in col{scales[i]=scales[i].max(v.abs());}}
            let prepared=Partition::new(&lap,&a,shared.clone(),omega,shift,weighted).unwrap();
            let expected:Vec<_>=(0..n).map(|i|(i as f64+0.2).sin()).collect();
            for transpose in [false,true]{let mut rhs=vec![0.;n];for(j,col)in a.iter().enumerate(){for &(i,v)in col{if transpose{rhs[j]+=v/scales[i]*expected[i];}else{rhs[i]+=v/scales[i]*expected[j];}}}
                let actual=prepared.apply(&lap,&rhs,&scales,transpose).unwrap();
                assert!(actual.iter().zip(&expected).all(|(a,b)|(a-b).abs()<1e-10),"weighted={weighted}, transpose={transpose}: {actual:?}");
            }
        }
    }
}
impl Shared {
    pub fn new(lap:&Lapack,a:&Sparse,n:usize)->Result<Self,String>{
        let start=Instant::now();let d=n*n;let weights:Vec<_>=(0..d).map(|i|if i<n{1.}else{2_f64.sqrt()}).collect();
        let mut dense=vec![0.;d*d];for(j,col)in a.iter().enumerate(){for &(i,v)in col{dense[i+j*d]=v*weights[i]/weights[j];}}
        let(t,q)=lap.real_schur(&dense,d)?;let qt=(0..d).flat_map(|j|(0..d).map({let q=&q;move|i|q[j+i*d]})).collect();
        Ok(Self{d,t,q,qt,weights,setup_ms:start.elapsed().as_secs_f64()*1000.,preparations:Cell::new(0),applications:Cell::new(0)})
    }
}
pub struct Partition {
    shared:Rc<Shared>,t:Vec<f64>,central_lu:Vec<f64>,pivots:Vec<i32>,
    omega:f64,sign:f64,weighted:bool,k:usize,
}
impl Partition {
    pub fn new(lap:&Lapack,a:&Sparse,shared:Rc<Shared>,omega:f64,shift:f64,weighted:bool)->Result<Self,String>{
        let d=shared.d;if a.len()<3*d||(a.len()/d)%2!=1||a.len()%d!=0{return Err("Schur experiment requires complete single-frequency coordinate blocks".into());}
        let sign=if weighted{-1.}else{1.};let k=(a.len()/d-1)/2;
        let mut center=vec![0.;d*d];for j in 0..d{for &(i,v)in &a[j]{if i<d{center[i+j*d]+=v;}}}
        let(central_lu,pivots)=lap.factor(&center,d)?;
        let p=d+1;let mut t=vec![0.;p*p];t[0]=1.;for j in 0..d{for i in 0..d{t[i+1+(j+1)*p]=sign*shared.t[i+j*d]+if i==j{shift}else{0.};}}
        shared.preparations.set(shared.preparations.get()+1);
        Ok(Self{shared,t,central_lu,pivots,omega,sign,weighted,k})
    }
    /// b belongs to the row-equilibrated ORIGINAL full Fourier equation.
    pub fn apply(&self,lap:&Lapack,b:&[f64],scales:&[f64],transpose:bool)->Result<Vec<f64>,String>{
        let d=self.shared.d;let p=d+1;let mut raw=b.to_vec();if !transpose{for(v,s)in raw.iter_mut().zip(scales){*v*=s;}}
        let mut result=vec![0.;b.len()];let center=if transpose{lap.solve_factored_transpose(&self.central_lu,&self.pivots,&raw[..d],d)?}else{lap.solve_factored(&self.central_lu,&self.pivots,&raw[..d],d)?};result[..d].copy_from_slice(&center);
        let mut rhs=raw[d..].to_vec();if !self.weighted{for(j,v)in rhs.iter_mut().enumerate(){let w=self.shared.weights[j%d];*v*=if transpose{1./w}else{w};}}
        let transformed=lap.multiply(&self.shared.qt,&rhs,d,2*self.k,d);let mut solved=vec![0.;rhs.len()];
        for h in 0..self.k{
            let mut pair=vec![0.;2*p];pair[1..p].copy_from_slice(&transformed[2*h*d..(2*h+1)*d]);pair[p+1..].copy_from_slice(&transformed[(2*h+1)*d..(2*h+2)*d]);
            let pair=lap.schur_shifted_solve(&self.t,&pair,p,-self.sign*(h+1)as f64*self.omega,transpose)?;
            solved[2*h*d..(2*h+1)*d].copy_from_slice(&pair[1..p]);solved[(2*h+1)*d..(2*h+2)*d].copy_from_slice(&pair[p+1..]);
        }
        let mut back=lap.multiply(&self.shared.q,&solved,d,2*self.k,d);if !self.weighted{for(j,v)in back.iter_mut().enumerate(){let w=self.shared.weights[j%d];*v*=if transpose{w}else{1./w};}}
        result[d..].copy_from_slice(&back);if transpose{for(v,s)in result.iter_mut().zip(scales){*v*=s;}}
        self.shared.applications.set(self.shared.applications.get()+1);Ok(result)
    }
}
