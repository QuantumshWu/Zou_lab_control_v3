//! First native preparation layer: COMPLETE supplied Model -> signed H/C.
//!
//! Physics remains owned by the existing TS builder. This module borrows its
//! numerical matrices, exact phase coordinates and G/E membership. It never
//! introduces Rb87 constants, CG coefficients, channel regrouping, detuning
//! cutoffs, frequency fitting, state elimination or a time-evolution answer.
//!
//! Existing native integration should pass authoritative phase high/low parts
//! from harmonics::model_phase_frequency_parts. The standalone validation
//! constructor below implements the same FMA compensated unit conversion,
//! without importing the old crate's private Model or integer-relation search.
//! The validated maximum-score-gap forest is available here. Additional
//! coordinate selectors must supply explicit integer gauges until ported and
//! compared independently; they must not silently fall back to another policy.
use std::collections::BTreeMap;
pub use crate::lapack::Complex64;
type C=Complex64;
pub type Charge=Vec<i32>;
pub type FourierMatrices=BTreeMap<Charge,Vec<C>>;
#[path="preparation_hilbert.rs"]
pub mod hilbert;
#[path="preparation_primitives.rs"]
pub mod primitives;
#[path="preparation_gauge.rs"]
pub mod gauge;

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
    pub initial_index:usize,
}

#[derive(Clone,Copy,Debug,Default)]
pub struct FrequencyParts{pub hi:f64,pub lo:f64}
impl FrequencyParts{
    pub fn phase(self,time:f64)->Result<C,String>{
        if !time.is_finite(){return Err("Nonfinite physical evaluation time".into());}
        let mut angle=Sum::default();angle.product(self.hi,time);angle.product(self.lo,time);let angle=angle.parts();
        if !angle.hi.is_finite()||!angle.lo.is_finite(){return Err("Physical phase product overflow".into());}
        let(sh,ch)=angle.hi.sin_cos();let(sl,cl)=angle.lo.sin_cos();Ok(C{re:ch*cl-sh*sl,im:sh*cl+ch*sl})
    }
}
#[derive(Clone,Debug)]
pub struct PhaseBasis{
    pub axes:Vec<FrequencyParts>,
    pub drive_indices:Vec<Charge>,
    pub authoritative_mhz:bool,
}
#[derive(Clone,Copy)]
pub enum GaugePolicy<'a>{Identity,MaximumScoreGapForest,UnsafeResonantForest,UnsafeComponentAligned,Explicit(&'a[Charge])}
pub struct PreparationDiagnostics{
    pub original_states:usize,pub original_channels:usize,pub phase_axes:usize,
    pub original_hamiltonian_charges:usize,pub gauged_hamiltonian_charges:usize,
    pub collapse_charge_blocks:usize,pub collapse_coefficient_pairs:usize,
    pub retained_tree_edges:usize,pub complete_tree_edges:usize,
    pub policy:&'static str,
    pub coordinate_selection:Option<gauge::GaugeDiagnostics>,
}
pub struct GaugedSource{
    pub n:usize,pub ground:Vec<usize>,pub excited:Vec<usize>,
    pub phases:PhaseBasis,pub gauges:Vec<Charge>,pub state_frequencies:Vec<FrequencyParts>,
    pub hamiltonian:FourierMatrices,
    /// Original channel grouping is the OUTER Vec. Different charges inside
    /// one channel still interfere in the complete Lindblad equation.
    pub collapse:Vec<FourierMatrices>,
    pub diagnostics:PreparationDiagnostics,
}
pub struct RadiationCoefficients{
    /// Compact EE gamma: e-by-e, row-major. All same-channel q,r included.
    pub gamma:FourierMatrices,
    /// Complete EE -> GG tensor: row (a*g+b), column (i*e+j).
    pub recycling:FourierMatrices,
    pub ground_indices:Vec<usize>,pub excited_indices:Vec<usize>,
    pub coefficient_pairs:usize,
}

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
#[inline]fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
#[inline]fn conj(a:C)->C{C{re:a.re,im:-a.im}}
#[inline]fn nonzero(a:&C)->bool{a.re!=0.||a.im!=0.}
fn opposite(q:&[i32])->Result<Charge,String>{q.iter().map(|&x|x.checked_neg().ok_or_else(||"Charge negation overflow".into())).collect()}
fn difference(a:&[i32],b:&[i32])->Result<Charge,String>{
    if a.len()!=b.len(){return Err("Charge dimensions differ".into());}
    a.iter().zip(b).map(|(&a,&b)|a.checked_sub(b).ok_or_else(||"Charge subtraction overflow".into())).collect()
}
fn plus(a:&[i32],b:&[i32])->Result<Charge,String>{
    if a.len()!=b.len(){return Err("Charge dimensions differ".into());}
    a.iter().zip(b).map(|(&a,&b)|a.checked_add(b).ok_or_else(||"Charge addition overflow".into())).collect()
}
fn matrix(view:MatrixView<'_>,n:usize,label:&str)->Result<Vec<C>,String>{
    if n.checked_mul(n)!=Some(view.real.len())||view.imag.len()!=view.real.len(){return Err(format!("{label} matrix shape differs"));}
    view.real.iter().zip(view.imag).map(|(&re,&im)|if re.is_finite()&&im.is_finite(){Ok(C{re,im})}else{Err(format!("{label} contains a nonfinite coefficient"))}).collect()
}
fn keep_exact_nonzero(values:&mut FourierMatrices){values.retain(|_,a|a.iter().any(nonzero));}

impl PhaseBasis{
    pub fn from_parts(axes:&[FrequencyParts],drive_indices:&[Charge])->Result<Self,String>{
        if axes.len()>64||axes.iter().any(|f|!f.hi.is_finite()||!f.lo.is_finite())
            ||drive_indices.iter().any(|q|q.len()!=axes.len()){
            return Err("Invalid complete declared phase dimensions/parts".into());
        }
        Ok(Self{axes:axes.to_vec(),drive_indices:drive_indices.to_vec(),authoritative_mhz:false})
    }
    /// No axis reduction or frequency relation search. Redundant angular
    /// values validate arithmetic only; declared phase provenance wins.
    pub fn from_declared(angular:&[f64],mhz:Option<&[f64]>,drive_indices:&[Charge])->Result<Self,String>{
        if angular.iter().any(|w|!w.is_finite()||*w<=0.){return Err("Declared axes must be positive finite frequencies".into());}
        let mut axes=Vec::with_capacity(angular.len());
        if let Some(source)=mhz{
            if source.len()!=angular.len()||source.iter().any(|w|!w.is_finite()||*w<=0.){return Err("Authoritative MHz axes differ".into());}
            for(&f,&omega)in source.iter().zip(angular){let hi=std::f64::consts::TAU*f;let lo=std::f64::consts::TAU.mul_add(f,-hi);
                let mut error=Sum::default();for v in[hi,lo,-omega]{error.add(v);}
                if !hi.is_finite()||!lo.is_finite()||error.value().abs()>16.*f64::EPSILON*(hi.abs()+omega.abs()){
                    return Err("Declared MHz and redundant angular frequency disagree".into());}
                axes.push(FrequencyParts{hi,lo});}
        }else{axes.extend(angular.iter().map(|&hi|FrequencyParts{hi,lo:0.}));}
        let mut result=Self::from_parts(&axes,drive_indices)?;result.authoritative_mhz=mhz.is_some();Ok(result)
    }
    pub fn frequency(&self,q:&[i32])->Result<FrequencyParts,String>{
        if q.len()!=self.axes.len(){return Err("Charge does not use the complete declared basis".into());}
        let mut sum=Sum::default();for(&k,f)in q.iter().zip(&self.axes){sum.product(k as f64,f.hi);sum.product(k as f64,f.lo);}
        let f=sum.parts();if !f.hi.is_finite()||!f.lo.is_finite(){return Err("Compensated frequency overflow".into());}Ok(f)
    }
    /// exp(i*q.omega*t), preserving the low parts of both integer-frequency
    /// and actual-time products before evaluating the two small factors.
    pub fn phase(&self,q:&[i32],time:f64)->Result<C,String>{
        self.frequency(q)?.phase(time)
    }
    fn verify_drive(&self,drive:DriveView<'_>,q:&[i32])->Result<(),String>{
        let f=self.frequency(q)?;let mut error=Sum::default();for x in[f.hi,f.lo,-drive.omega]{error.add(x);}
        let scale=drive.omega.abs()+q.iter().zip(&self.axes).map(|(&k,f)|(k as f64).abs()*(f.hi.abs()+f.lo.abs())).sum::<f64>();
        if !drive.omega.is_finite()||!scale.is_finite()||error.value().abs()>32.*f64::EPSILON*scale{
            return Err("Drive omega differs from exact declared charge coordinates".into());}Ok(())
    }
}

#[derive(Clone)]
struct Edge{score:f64,i:usize,j:usize,q:Charge}
fn forest(original:&FourierMatrices,phases:&PhaseBasis,n:usize,initial:usize)->Result<(Vec<Charge>,usize,usize),String>{
    let zero=vec![0;phases.axes.len()];let h0=&original[&zero];let mut edges=Vec::new();
    for(q,h)in original{let omega=phases.frequency(q)?;for i in 0..n{for j in i+1..n{
        let amplitude=h[i*n+j].re.hypot(h[i*n+j].im);if amplitude==0.{continue;}
        let mut delta=Sum::default();for x in[h0[i*n+i].re,-h0[j*n+j].re,omega.hi,omega.lo]{delta.add(x);}
        let score=amplitude/(1.+delta.value().abs());
        if !score.is_finite(){return Err("Coordinate edge score overflow".into());}
        edges.push(Edge{score,i,j,q:q.clone()});
    }}}
    edges.sort_by(|a,b|b.score.total_cmp(&a.score).then_with(||a.i.cmp(&b.i)).then_with(||a.j.cmp(&b.j)).then_with(||a.q.cmp(&b.q)));
    let mut groups:Vec<_>=(0..n).collect();let mut tree=Vec::new();
    for edge in edges{let(a,b)=(groups[edge.i],groups[edge.j]);if a==b{continue;}
        for group in &mut groups{if *group==b{*group=a;}}tree.push(edge);}
    let mut keep=tree.len();let mut greatest=0.;
    for i in 0..tree.len().saturating_sub(1){let(a,b)=(tree[i].score,tree[i+1].score);
        if a>b&&b>0.{let gap=a.ln()-b.ln();if gap>greatest{greatest=gap;keep=i+1;}}}
    let mut groups:Vec<_>=(0..n).collect();let mut gauges=vec![zero;n];
    for edge in &tree[..keep]{let(left,right)=(groups[edge.i],groups[edge.j]);
        let shift=difference(&difference(&gauges[edge.i],&gauges[edge.j])?,&edge.q)?;
        for i in 0..n{if groups[i]==right{gauges[i]=plus(&gauges[i],&shift)?;groups[i]=left;}}
    }
    for group in 0..n{let members:Vec<_>=(0..n).filter(|&i|groups[i]==group).collect();if members.is_empty(){continue;}
        let anchor=if members.contains(&initial){initial}else{members[0]};let shift=gauges[anchor].clone();
        for i in members{gauges[i]=difference(&gauges[i],&shift)?;}}
    Ok((gauges,keep,tree.len()))
}

impl GaugedSource{
    pub fn prepare(model:&ModelView<'_>,phases:PhaseBasis,policy:GaugePolicy<'_>)->Result<Self,String>{
        Self::prepare_internal(model,phases,policy,None)
    }
    pub fn prepare_with_lapack(model:&ModelView<'_>,phases:PhaseBasis,policy:GaugePolicy<'_>,lap:&crate::lapack::Lapack)->Result<Self,String>{
        Self::prepare_internal(model,phases,policy,Some(lap))
    }
    fn prepare_internal(model:&ModelView<'_>,phases:PhaseBasis,policy:GaugePolicy<'_>,lap:Option<&crate::lapack::Lapack>)->Result<Self,String>{
        let n=model.dimension;let axes=phases.axes.len();
        if n==0||n>64||model.ground.len()!=n||model.initial_index>=n||model.drives.len()!=phases.drive_indices.len(){return Err("Invalid complete Model/preparation dimensions".into());}
        let ground:Vec<_>=(0..n).filter(|&i|model.ground[i]).collect();let excited:Vec<_>=(0..n).filter(|&i|!model.ground[i]).collect();
        if ground.is_empty()||excited.is_empty(){return Err("Explicit nonempty G/E sectors are required".into());}
        let zero=vec![0;axes];let mut original=FourierMatrices::from([(zero.clone(),matrix(model.hamiltonian,n,"H0")?)]);
        for(&drive,q)in model.drives.iter().zip(&phases.drive_indices){
            phases.verify_drive(drive,q)?;let cos=matrix(drive.cosine,n,"drive cosine")?;let sin=matrix(drive.sine,n,"drive sine")?;
            for sign in[1.,-1.]{let key=if sign>0.{q.clone()}else{opposite(q)?};let target=original.entry(key).or_insert_with(||vec![C::default();n*n]);
                for((value,a),b)in target.iter_mut().zip(&cos).zip(&sin){add(value,C{re:0.5*(a.re+sign*b.im),im:0.5*(a.im-sign*b.re)});}}
        }
        let mut coordinate_selection=None;
        let(gauges,kept,total,label)=match policy{
            GaugePolicy::Identity=>(vec![zero.clone();n],0,0,"identity"),
            GaugePolicy::MaximumScoreGapForest=>{let(g,k,t)=forest(&original,&phases,n,model.initial_index)?;(g,k,t,"maximum-tree-score-gap-forest")},
            GaugePolicy::UnsafeResonantForest|GaugePolicy::UnsafeComponentAligned=>{
                let lap=lap.ok_or("Unsafe/component gauge selection requires the existing small-Hilbert LAPACK; no silent policy substitution")?;
                let aligned=matches!(policy,GaugePolicy::UnsafeComponentAligned);
                let selection=gauge::select(lap,model,&original,&phases,aligned)?;
                let count=selection.diagnostics.tree_edges;coordinate_selection=Some(selection.diagnostics);
                (selection.gauges,count,count,if aligned{"unsafe-component-aligned"}else{"unsafe-resonant-forest"})
            },
            GaugePolicy::Explicit(g)=>{if g.len()!=n||g.iter().any(|q|q.len()!=axes){return Err("Explicit integer gauge dimensions differ".into());}(g.to_vec(),0,0,"caller-explicit-validated-integer-gauge")},
        };
        let mut hamiltonian=FourierMatrices::from([(zero.clone(),vec![C::default();n*n])]);
        for(q,h)in &original{for i in 0..n{for j in 0..n{let value=h[i*n+j];if !nonzero(&value){continue;}
            let charge=plus(q,&difference(&gauges[j],&gauges[i])?)?;
            add(&mut hamiltonian.entry(charge).or_insert_with(||vec![C::default();n*n])[i*n+j],value);
        }}}
        let state_frequencies=gauges.iter().map(|q|phases.frequency(q)).collect::<Result<Vec<_>,_>>()?;
        for(i,f)in state_frequencies.iter().enumerate(){let target=&mut hamiltonian.get_mut(&zero).unwrap()[i*n+i];
            let mut value=Sum::default();for x in[target.re,f.hi,f.lo]{value.add(x);}target.re=value.value();}
        let mut collapse=Vec::new();let mut blocks=0;let mut pairs=0;
        for &operator in model.collapse{let values=matrix(operator,n,"collapse")?;let mut channel=FourierMatrices::new();
            for i in 0..n{for j in 0..n{let value=values[i*n+j];if !nonzero(&value){continue;}
                if !model.ground[i]||model.ground[j]{return Err("Original C has nonzero support outside E->G; retain the complete generic route".into());}
                let q=difference(&gauges[j],&gauges[i])?;channel.entry(q).or_insert_with(||vec![C::default();n*n])[i*n+j]=value;
            }}
            blocks+=channel.len();pairs+=channel.len()*channel.len();collapse.push(channel);
        }
        if hamiltonian.values().flatten().any(|x|!x.re.is_finite()||!x.im.is_finite()){return Err("Complete Hamiltonian assembly overflow".into());}
        let diagnostics=PreparationDiagnostics{original_states:n,original_channels:collapse.len(),phase_axes:axes,
            original_hamiltonian_charges:original.len(),gauged_hamiltonian_charges:hamiltonian.len(),collapse_charge_blocks:blocks,
            collapse_coefficient_pairs:pairs,retained_tree_edges:kept,complete_tree_edges:total,policy:label,coordinate_selection};
        Ok(Self{n,ground,excited,phases,gauges,state_frequencies,hamiltonian,collapse,diagnostics})
    }
    /// Complete Gamma_q and R_q in the supplied state basis, before the next
    /// block-diagonal Hilbert preparation. Noncontiguous original G/E indices
    /// are explicitly mapped; neither Hilbert manifold nor C channel is lost.
    pub fn radiation(&self)->Result<RadiationCoefficients,String>{
        let(g,e,n)=(self.ground.len(),self.excited.len(),self.n);let mut gamma=FourierMatrices::new();let mut recycling=FourierMatrices::new();let mut pairs=0;
        for channel in &self.collapse{for(qp,cp)in channel{for(qr,cr)in channel{
            let q=difference(qp,qr)?;let gam=gamma.entry(q.clone()).or_insert_with(||vec![C::default();e*e]);
            let rec=recycling.entry(q).or_insert_with(||vec![C::default();g*g*e*e]);
            for a in 0..g{for i in 0..e{let p=cp[self.ground[a]*n+self.excited[i]];
                for j in 0..e{let r=cr[self.ground[a]*n+self.excited[j]];add(&mut gam[j*e+i],mul(conj(r),p));}
                for b in 0..g{for j in 0..e{let r=cr[self.ground[b]*n+self.excited[j]];add(&mut rec[(a*g+b)*e*e+i*e+j],mul(p,conj(r)));}}
            }}pairs+=1;
        }}}
        keep_exact_nonzero(&mut gamma);keep_exact_nonzero(&mut recycling);
        if gamma.values().chain(recycling.values()).flatten().any(|z|!z.re.is_finite()||!z.im.is_finite()){return Err("Complete radiation coefficient overflow".into());}
        Ok(RadiationCoefficients{gamma,recycling,ground_indices:self.ground.clone(),excited_indices:self.excited.clone(),coefficient_pairs:pairs})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn mv<'a>(re:&'a[f64],im:&'a[f64])->MatrixView<'a>{MatrixView{real:re,imag:im}}
    #[test]
    fn preparation_zero_axes_and_noncontiguous_ground_keep_tiny_radiation(){
        let n=4;let h=vec![0.;16];let mut cr=vec![0.;16];let ci=vec![0.;16];cr[1*n]=0.7;cr[3*n+2]=1e-100;
        let c=[mv(&cr,&ci)];let phases=PhaseBasis::from_declared(&[],None,&[]).unwrap();
        let model=ModelView{dimension:n,hamiltonian:mv(&h,&h),collapse:&c,drives:&[],ground:&[false,true,false,true],initial_index:1};
        let result=GaugedSource::prepare(&model,phases,GaugePolicy::MaximumScoreGapForest).unwrap();
        assert_eq!(result.ground,vec![1,3]);assert_eq!(result.excited,vec![0,2]);assert!(result.gauges.iter().all(Vec::is_empty));
        let radiation=result.radiation().unwrap();assert_eq!(radiation.coefficient_pairs,1);
        let gamma=&radiation.gamma[&vec![]];assert!(gamma[3].re>0.);assert!((gamma[3].re/1e-200-1.).abs()<4.*f64::EPSILON);
        let recycling=&radiation.recycling[&vec![]];assert!(recycling[3*4+3].re>0.);assert!((recycling[3*4+3].re/1e-200-1.).abs()<4.*f64::EPSILON);
    }
    #[test]
    fn preparation_keeps_original_channel_cross_charge_interference(){
        let h=vec![0.;9];let cos=vec![0.,0.,1.,0.,0.,0.,1.,0.,0.];let sin=vec![0.;9];let cr=vec![0.,0.,0.7,0.,0.,0.2,0.,0.,0.];let ci=vec![0.;9];
        let drives=[DriveView{omega:7.,cosine:mv(&cos,&sin),sine:mv(&sin,&sin)}];let c=[mv(&cr,&ci)];
        let model=ModelView{dimension:3,hamiltonian:mv(&h,&h),collapse:&c,drives:&drives,ground:&[true,true,false],initial_index:0};
        let phases=PhaseBasis::from_declared(&[7.],None,&[vec![1]]).unwrap();let gauge=[vec![0],vec![1],vec![2]];
        let result=GaugedSource::prepare(&model,phases,GaugePolicy::Explicit(&gauge)).unwrap();
        assert_eq!(result.collapse.len(),1);assert_eq!(result.collapse[0].len(),2);
        assert_eq!(result.collapse[0][&vec![2]][2].re,0.7);
        assert!(result.hamiltonian.contains_key(&vec![3]));assert!(result.hamiltonian.contains_key(&vec![-3]));
        assert_eq!(result.hamiltonian[&vec![0]][4].re,7.);assert_eq!(result.hamiltonian[&vec![0]][8].re,14.);
        let radiation=result.radiation().unwrap();assert_eq!(radiation.coefficient_pairs,4);
        assert!((radiation.recycling[&vec![1]][1].re-0.14).abs()<2.*f64::EPSILON);
        assert!((radiation.recycling[&vec![-1]][2].re-0.14).abs()<2.*f64::EPSILON);
    }
    #[test]
    fn preparation_authoritative_phase_low_part_and_no_axis_merging(){
        let frequencies=[1e12,1e12+0.01];let angular=frequencies.map(|f|std::f64::consts::TAU*f);
        let basis=PhaseBasis::from_declared(&angular,Some(&frequencies),&[vec![-1,1]]).unwrap();
        assert_eq!(basis.axes.len(),2);assert!(basis.authoritative_mhz);
        let f=basis.frequency(&[-1,1]).unwrap();let expected=std::f64::consts::TAU*(frequencies[1]-frequencies[0]);
        assert!(((f.hi+f.lo)-expected).abs()<1e-12);
        assert!((basis.axes[1].hi-basis.axes[0].hi-expected).abs()>1e-6);
        assert!(PhaseBasis::from_declared(&[7.],None,&[vec![1,0]]).is_err());
    }
}
