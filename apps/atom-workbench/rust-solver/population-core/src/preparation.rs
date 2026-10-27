//! The supplied model in the request's own frame, as Fourier components.
//!
//! Physics is owned by the TypeScript builder. This module borrows its
//! numerical matrices, exact phase coordinates and ground/excited
//! membership; it never introduces Rb87 constants, CG coefficients, channel
//! regrouping, detuning cutoffs, frequency fitting or state elimination. H0
//! is the charge-0 component, each drive A cos(wt) + B sin(wt) contributes
//! (A - iB)/2 at its declared charge q and (A + iB)/2 at -q, and every
//! collapse channel is static in this frame. Frequencies are kept as
//! compensated (hi, lo) parts, so a beat between optical carriers keeps its
//! low-order digits.
use std::collections::BTreeMap;
use crate::C;
pub type Charge=Vec<i32>;
pub type FourierMatrices=BTreeMap<Charge,Vec<C>>;

#[derive(Clone,Copy)]
pub struct MatrixView<'a>{pub real:&'a[f64],pub imag:&'a[f64]}
#[derive(Clone,Copy)]
pub struct DriveView<'a>{pub omega:f64,pub cosine:MatrixView<'a>,pub sine:MatrixView<'a>}
pub struct ModelView<'a>{
    pub dimension:usize,
    pub hamiltonian:MatrixView<'a>,
    pub collapse:&'a[MatrixView<'a>],
    pub drives:&'a[DriveView<'a>],
    /// One explicit membership flag per supplied state; do not infer F/mF.
    pub ground:&'a[bool],
}

#[derive(Clone,Copy,Debug,Default)]
pub struct FrequencyParts{pub hi:f64,pub lo:f64}
#[derive(Clone,Debug)]
pub struct PhaseBasis{
    pub axes:Vec<FrequencyParts>,
    pub drive_indices:Vec<Charge>,
}
pub struct Source{
    pub n:usize,pub ground:Vec<usize>,
    pub phases:PhaseBasis,
    pub hamiltonian:FourierMatrices,
    /// One entry per supplied collapse channel: its Fourier components, the
    /// charge-0 one only (the channels are static in this frame), none for
    /// an all-zero channel.
    pub collapse:Vec<FourierMatrices>,
}

/// Double-double accumulator (Knuth two-sum, FMA products).
#[derive(Clone,Copy,Default)]
struct Sum{hi:f64,lo:f64}
impl Sum{
    fn add(&mut self,value:f64){let s=self.hi+value;let z=s-self.hi;let error=(self.hi-(s-z))+(value-z);
        let low=self.lo+error;let high=s+low;self.lo=low-(high-s);self.hi=high;}
    fn product(&mut self,a:f64,b:f64){let p=a*b;self.add(p);self.add(a.mul_add(b,-p));}
    fn parts(self)->FrequencyParts{FrequencyParts{hi:self.hi,lo:self.lo}}
    fn value(self)->f64{self.hi+self.lo}
}
#[inline]fn add(a:&mut C,b:C){a.re+=b.re;a.im+=b.im;}
/// Couplings below this fraction of the largest one never change the
/// lattice's structure (its explicit steps and their cost); they still act
/// exactly in every operator.
pub(crate) const STRUCTURAL_COUPLING_FLOOR:f64=1e-5;
fn opposite(q:&[i32])->Result<Charge,String>{q.iter().map(|&x|x.checked_neg().ok_or_else(||"Charge negation overflow".into())).collect()}
fn matrix(view:MatrixView<'_>,n:usize,label:&str)->Result<Vec<C>,String>{
    if n.checked_mul(n)!=Some(view.real.len())||view.imag.len()!=view.real.len(){return Err(format!("{label} matrix shape differs"));}
    view.real.iter().zip(view.imag).map(|(&re,&im)|if re.is_finite()&&im.is_finite(){Ok(C{re,im})}else{Err(format!("{label} contains a nonfinite coefficient"))}).collect()
}

impl PhaseBasis{
    pub fn from_parts(axes:&[FrequencyParts],drive_indices:&[Charge])->Result<Self,String>{
        if axes.len()>64||axes.iter().any(|f|!f.hi.is_finite()||!f.lo.is_finite())
            ||drive_indices.iter().any(|q|q.len()!=axes.len()){
            return Err("Invalid complete declared phase dimensions/parts".into());
        }
        Ok(Self{axes:axes.to_vec(),drive_indices:drive_indices.to_vec()})
    }
    /// Axes given as plain angular frequencies (test models).
    #[cfg(test)]
    pub(crate) fn from_declared(angular:&[f64],drive_indices:&[Charge])->Result<Self,String>{
        if angular.iter().any(|w|!w.is_finite()||*w<=0.){return Err("Declared axes must be positive finite frequencies".into());}
        Self::from_parts(&angular.iter().map(|&hi|FrequencyParts{hi,lo:0.}).collect::<Vec<_>>(),drive_indices)
    }
    pub fn frequency(&self,q:&[i32])->Result<FrequencyParts,String>{
        if q.len()!=self.axes.len(){return Err("Charge does not use the complete declared basis".into());}
        let mut sum=Sum::default();for(&k,f)in q.iter().zip(&self.axes){sum.product(k as f64,f.hi);sum.product(k as f64,f.lo);}
        let f=sum.parts();if !f.hi.is_finite()||!f.lo.is_finite(){return Err("Compensated frequency overflow".into());}Ok(f)
    }
    fn verify_drive(&self,drive:DriveView<'_>,q:&[i32])->Result<(),String>{
        let f=self.frequency(q)?;let mut error=Sum::default();for x in[f.hi,f.lo,-drive.omega]{error.add(x);}
        let scale=drive.omega.abs()+q.iter().zip(&self.axes).map(|(&k,f)|(k as f64).abs()*(f.hi.abs()+f.lo.abs())).sum::<f64>();
        if !drive.omega.is_finite()||!scale.is_finite()||error.value().abs()>32.*f64::EPSILON*scale{
            return Err("Drive omega differs from exact declared charge coordinates".into());}Ok(())
    }
}

impl Source{
    pub fn prepare(model:&ModelView<'_>,phases:PhaseBasis)->Result<Self,String>{
        let n=model.dimension;let axes=phases.axes.len();
        if n==0||n>64||model.ground.len()!=n||model.drives.len()!=phases.drive_indices.len(){return Err("Invalid complete Model/preparation dimensions".into());}
        let ground:Vec<_>=(0..n).filter(|&i|model.ground[i]).collect();
        if ground.is_empty()||ground.len()==n{return Err("Explicit nonempty G/E sectors are required".into());}
        let mut hamiltonian=FourierMatrices::from([(vec![0;axes],matrix(model.hamiltonian,n,"H0")?)]);
        for(&drive,q)in model.drives.iter().zip(&phases.drive_indices){
            phases.verify_drive(drive,q)?;let cos=matrix(drive.cosine,n,"drive cosine")?;let sin=matrix(drive.sine,n,"drive sine")?;
            for sign in[1.,-1.]{let key=if sign>0.{q.clone()}else{opposite(q)?};let target=hamiltonian.entry(key).or_insert_with(||vec![C::default();n*n]);
                for((value,a),b)in target.iter_mut().zip(&cos).zip(&sin){add(value,C{re:0.5*(a.re+sign*b.im),im:0.5*(a.im-sign*b.re)});}}
        }
        let mut collapse=Vec::with_capacity(model.collapse.len());
        for &operator in model.collapse{let values=matrix(operator,n,"collapse")?;let mut channel=FourierMatrices::new();
            for i in 0..n{for j in 0..n{let value=values[i*n+j];if value.re==0.&&value.im==0.{continue;}
                if !model.ground[i]||model.ground[j]{return Err("Collapse operators must act from excited to ground states".into());}
            }}
            if values.iter().any(|z|z.re!=0.||z.im!=0.){channel.insert(vec![0;axes],values);}
            collapse.push(channel);
        }
        if hamiltonian.values().flatten().any(|x|!x.re.is_finite()||!x.im.is_finite()){return Err("Complete Hamiltonian assembly overflow".into());}
        Ok(Self{n,ground,phases,hamiltonian,collapse})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn mv<'a>(re:&'a[f64],im:&'a[f64])->MatrixView<'a>{MatrixView{real:re,imag:im}}
    #[test]
    fn preparation_splits_each_drive_into_its_two_charges(){
        // A cos(7t) + B sin(7t) with A = |0><2| + h.c., B = 0.4 i (|1><2| - h.c.):
        // (A - iB)/2 at +1 and (A + iB)/2 at -1; H0 stays at 0.
        let(h,zero)=(vec![0.,0.,0.,0.,0.5,0.,0.,0.,7.],vec![0.;9]);
        let cos=vec![0.,0.,1.,0.,0.,0.,1.,0.,0.];let sin=vec![0.,0.,0.,0.,0.,0.4,0.,-0.4,0.];
        let drives=[DriveView{omega:7.,cosine:mv(&cos,&zero),sine:mv(&zero,&sin)}];
        let(cr,ci)=(vec![0.,0.,0.7,0.,0.,1e-100,0.,0.,0.],vec![0.;9]);let c=[mv(&cr,&ci)];
        let model=ModelView{dimension:3,hamiltonian:mv(&h,&zero),collapse:&c,drives:&drives,ground:&[true,true,false]};
        let source=Source::prepare(&model,PhaseBasis::from_declared(&[7.],&[vec![1]]).unwrap()).unwrap();
        assert_eq!(source.hamiltonian.keys().cloned().collect::<Vec<_>>(),vec![vec![-1],vec![0],vec![1]]);
        let(plus,minus)=(&source.hamiltonian[&vec![1]],&source.hamiltonian[&vec![-1]]);
        assert_eq!((plus[2].re,plus[5].re,minus[5].re),(0.5,0.2,-0.2));
        assert_eq!(source.hamiltonian[&vec![0]][8].re,7.);
        // A tiny collapse entry stays exactly.
        assert_eq!(source.collapse[0][&vec![0]][5].re,1e-100);
        assert_eq!(source.ground,vec![0,1]);
    }
    #[test]
    fn preparation_rejects_inconsistent_coordinates_and_collapse_support(){
        let(h,zero)=(vec![0.;9],vec![0.;9]);let cos=vec![0.,0.,1.,0.,0.,0.,1.,0.,0.];
        let drives=[DriveView{omega:7.,cosine:mv(&cos,&zero),sine:mv(&zero,&zero)}];
        let model=|c:&[MatrixView<'_>],drives:&[DriveView<'_>],phases:PhaseBasis|{
            Source::prepare(&ModelView{dimension:3,hamiltonian:mv(&h,&zero),collapse:c,drives,ground:&[true,true,false]},phases).map(|_|())};
        assert!(model(&[],&drives,PhaseBasis::from_declared(&[7.],&[vec![1]]).unwrap()).is_ok());
        assert!(model(&[],&drives,PhaseBasis::from_declared(&[7.5],&[vec![1]]).unwrap()).is_err());
        let up=vec![0.,0.,0.,0.,0.,0.,1.,0.,0.];
        assert!(model(&[mv(&up,&zero)],&[],PhaseBasis::from_declared(&[],&[]).unwrap()).is_err());
        assert!(PhaseBasis::from_declared(&[7.],&[vec![1,0]]).is_err());
    }
}
