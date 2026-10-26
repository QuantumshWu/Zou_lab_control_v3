//! Request-owned complete B0 inverse. Two small Hilbert frames and transformed
//! U are shared across domains. Exact finite complex shift keys own immutable
//! RU/small feedback factors. Runtime applies a fixed prepared operator only.
use super::*;
use super::b0_factor::B0Factor;

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum P0Policy{EigenPreferred,SchurOnly}
#[derive(Clone,Debug)]
pub struct FrameDiagnostics{
    pub backend:&'static str,pub left_condition_upper:Option<f64>,pub right_condition_upper:Option<f64>,
    pub left_reconstruction_hs:f64,pub right_reconstruction_hs:f64,
    pub transformed_u_columns:usize,pub hilbert_factorizations:usize,
}
#[derive(Clone,Debug)]
pub struct ShiftDiagnostics{
    pub shift:C,pub backend:&'static str,pub fallback_reason:Option<String>,
    pub minimum_denominator:Option<f64>,pub eigen_perturbation_indicator:Option<f64>,
    pub feedback_condition_upper:Option<f64>,pub feedback_inverse_residual_inf:Option<f64>,
    pub all_feedback_columns:usize,
}
#[derive(Clone,Debug,Default)]
pub struct RequestPreparationProfile{
    pub factor_ms:f64,pub initial_frame_ms:f64,pub lazy_schur_frame_ms:f64,
    pub shift_factors_ms:f64,pub shift_cache_hits:usize,pub unique_shifts:usize,pub feedback_columns:usize,
}
enum FrameKind{Eigen{a:Vec<C>,b:Vec<C>},Schur{ta:Vec<C>,tb:Vec<C>}}
struct Frame{
    left_in:Vec<C>,right_in_t:Vec<C>,left_out:Vec<C>,right_out_t:Vec<C>,
    kind:FrameKind,uhat:Vec<C>,diagnostics:FrameDiagnostics,
}
enum ShiftSolve{Eigen{reciprocal:Vec<C>},Schur{left:Vec<C>}}
enum Feedback{Empty,Inverse(Vec<C>),Lu{matrix:Vec<C>,pivots:Vec<i32>}}
struct ShiftFactor{frame:Arc<Frame>,solve:ShiftSolve,ru:Vec<C>,feedback:Feedback,diagnostics:ShiftDiagnostics}

pub struct RequestP0{
    factor:Arc<B0Factor>,lapack:Arc<Lapack>,eigen:Option<Arc<Frame>>,schur:Option<Arc<Frame>>,
    eigen_failure:Option<String>,entries:BTreeMap<(u64,u64),Arc<ShiftFactor>>,profile:RequestPreparationProfile,
}
pub struct P0Batch{
    factor:Arc<B0Factor>,lapack:Arc<Lapack>,entries:Vec<Arc<ShiftFactor>>,max_rhs:usize,workspace:P0Workspace,
}
#[derive(Default)]
struct P0Workspace{block:Vec<C>,reduced:Vec<C>,base:Vec<C>,pack_a:Vec<C>,pack_b:Vec<C>,middle:Vec<C>,last:Vec<C>,beta:Vec<C>,amount:Vec<C>,correction:Vec<C>}

fn add(a:&mut C,b:C){a.re+=b.re;a.im+=b.im;}
fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
fn conj(a:C)->C{C{re:a.re,im:-a.im}}
fn minus(a:C,b:C)->C{C{re:a.re-b.re,im:a.im-b.im}}
fn norm(a:&[C])->f64{a.iter().fold(0_f64,|v,z|v.hypot(z.re.hypot(z.im)))}
fn column(a:&[C],rows:usize,cols:usize)->Vec<C>{(0..rows*cols).map(|i|a[(i%rows)*cols+i/rows]).collect()}
fn transpose(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|i|a[(i%n)*n+i/n]).collect()}
fn adjoint(a:&[C],n:usize)->Vec<C>{transpose(a,n).into_iter().map(conj).collect()}
fn identity(n:usize)->Vec<C>{let mut x=vec![C::default();n*n];for i in 0..n{x[i+i*n].re=1.;}x}
fn sum3(a:f64,b:f64,c:f64)->f64{
    let mut hi=0.;let mut lo=0.;for value in[a,b,c]{let s=hi+value;let z=s-hi;lo+=(hi-(s-z))+(value-z);hi=s;}
    hi+lo
}
fn reciprocal(z:C)->Result<C,String>{let scale=z.re.hypot(z.im);
    if scale==0.||!scale.is_finite(){return Err("Unresolved shifted Sylvester denominator".into());}
    let value=C{re:(z.re/scale)/scale,im:-(z.im/scale)/scale};
    if !value.finite(){return Err("Shifted Sylvester reciprocal overflow".into());}Ok(value)
}
fn resize(a:&mut Vec<C>,length:usize){a.resize(length,C::default());}
fn inverse(lap:&Lapack,a:&[C],n:usize)->Result<(Vec<C>,Vec<C>,Vec<i32>),String>{
    if a.iter().any(|z|!z.finite()){return Err("Nonfinite complete small matrix before LU".into());}
    let(lu,pivots)=lap.complex_factor(a,n)?;let mut result=vec![C::default();n*n];
    for col in 0..n{let mut rhs=vec![C::default();n];rhs[col].re=1.;
        result[col*n..(col+1)*n].copy_from_slice(&lap.complex_solve_factored(&lu,&pivots,&rhs,n)?);}
    if result.iter().any(|z|!z.finite()){return Err("Nonfinite complete small inverse".into());}Ok((result,lu,pivots))
}
fn norms_one_inf(a:&[C],n:usize)->(f64,f64){let mut one=0_f64;let mut inf=0_f64;
    for i in 0..n{let mut col=0.;let mut row=0.;for j in 0..n{col+=a[j+i*n].re.hypot(a[j+i*n].im);row+=a[i+j*n].re.hypot(a[i+j*n].im);}one=one.max(col);inf=inf.max(row);}(one,inf)
}
fn condition_upper(a:&[C],inverse:&[C],n:usize)->f64{let(a1,ai)=norms_one_inf(a,n);let(b1,bi)=norms_one_inf(inverse,n);(a1*ai).sqrt()*(b1*bi).sqrt()}
fn inverse_error(lap:&Lapack,a:&[C],inverse:&[C],n:usize)->Result<f64,String>{
    let mut product=lap.multiply_complex(a,inverse,n,n,n)?;for i in 0..n{product[i+i*n].re-=1.;}Ok(norms_one_inf(&product,n).1)
}
fn transform(lap:&Lapack,n:usize,left:&[C],right_t:&[C],input:&[C],rhs:usize,output:&mut Vec<C>,
    pack_a:&mut Vec<C>,pack_b:&mut Vec<C>,middle:&mut Vec<C>,last:&mut Vec<C>)->Result<(),String>{
    let length=product(&[n,n,rhs])?;if input.len()!=length{return Err("Hilbert column transform dimensions differ".into());}
    for a in [&mut *output,&mut *pack_a,&mut *pack_b,&mut *middle,&mut *last]{resize(a,length);}
    if rhs==0{return Ok(());}
    for b in 0..rhs{for i in 0..n{for j in 0..n{pack_a[i+(b*n+j)*n]=input[(i*n+j)*rhs+b];}}}
    lap.multiply_complex_into(left,pack_a,middle,n,n*rhs,n)?;
    for b in 0..rhs{for i in 0..n{for j in 0..n{pack_b[j+(b*n+i)*n]=middle[i+(b*n+j)*n];}}}
    lap.multiply_complex_into(right_t,pack_b,last,n,n*rhs,n)?;
    for b in 0..rhs{for i in 0..n{for j in 0..n{output[(i*n+j)*rhs+b]=last[j+(b*n+i)*n];}}}Ok(())
}
fn dense_u(factor:&B0Factor)->Vec<C>{let d=factor.dimension();let r=factor.rank();let mut u=vec![C::default();d*r];
    for i in 0..d{for at in factor.u.row[i]..factor.u.row[i+1]{u[i*r+factor.u.col[at]]=factor.u.value[at];}}u}

impl Frame{
    fn eigen(factor:&B0Factor,lap:&Lapack)->Result<Self,String>{
        let n=factor.hilbert_dimension();let a=column(&factor.left,n,n);let b=column(&factor.right,n,n);
        let(ea,va)=lap.complex_eigen_right(&a,n)?;let(eb,vb)=lap.complex_eigen_right(&b,n)?;
        let(wa,_,_)=inverse(lap,&va,n)?;let(wb,_,_)=inverse(lap,&vb,n)?;
        let ca=condition_upper(&va,&wa,n);let cb=condition_upper(&vb,&wb,n);
        if !ca.is_finite()||!cb.is_finite()||ca.max(cb)>1e4{return Err(format!("Complete eigenframe condition bound unsuitable: {ca:e}, {cb:e}"));}
        if inverse_error(lap,&va,&wa,n)?>1e-8||inverse_error(lap,&vb,&wb,n)?>1e-8{return Err("Eigenframe inverse residual unresolved".into());}
        let reconstruction=|original:&[C],v:&[C],eigen:&[C],w:&[C]|->Result<f64,String>{
            let scaled:Vec<_>=v.iter().enumerate().map(|(i,&z)|mul(z,eigen[i/n])).collect();let represented=lap.multiply_complex(&scaled,w,n,n,n)?;
            Ok(norm(&represented.iter().zip(original).map(|(&a,&b)|minus(a,b)).collect::<Vec<_>>()))};
        let da=reconstruction(&a,&va,&ea,&wa)?;let db=reconstruction(&b,&vb,&eb,&wb)?;
        // ||A||F/sqrt(n)<=||A||2, so this sufficient check cannot weaken the
        // existing relative spectral reconstruction threshold.
        if !da.is_finite()||!db.is_finite()||(da/(norm(&a)/(n as f64).sqrt()).max(1.)).max(db/(norm(&b)/(n as f64).sqrt()).max(1.))>1e-12{
            return Err("Complete independent no-jump eigen reconstruction unresolved".into());}
        let mut frame=Self{left_in:wa,right_in_t:transpose(&vb,n),left_out:va,right_out_t:transpose(&wb,n),kind:FrameKind::Eigen{a:ea,b:eb},uhat:vec![],
            diagnostics:FrameDiagnostics{backend:"eigen",left_condition_upper:Some(ca),right_condition_upper:Some(cb),left_reconstruction_hs:da,right_reconstruction_hs:db,
                transformed_u_columns:factor.rank(),hilbert_factorizations:2}};
        frame.prepare_u(factor,lap)?;Ok(frame)
    }
    fn schur(factor:&B0Factor,lap:&Lapack)->Result<Self,String>{
        let n=factor.hilbert_dimension();let a=column(&factor.left,n,n);let b=column(&factor.right,n,n);
        let(ta,qa)=lap.complex_schur(&a,n)?;
        // Existing mature kernel solves N/C. Factor the independently supplied
        // B^H so that Tb^H represents B exactly; never substitute A^H for B.
        let(tb,qb)=lap.complex_schur(&adjoint(&b,n),n)?;let qah=adjoint(&qa,n);let qbh=adjoint(&qb,n);
        let ra=lap.multiply_complex(&lap.multiply_complex(&qa,&ta,n,n,n)?,&qah,n,n,n)?;
        let rb=lap.multiply_complex(&lap.multiply_complex(&qb,&adjoint(&tb,n),n,n,n)?,&qbh,n,n,n)?;
        let da=norm(&ra.iter().zip(&a).map(|(&x,&y)|minus(x,y)).collect::<Vec<_>>());let db=norm(&rb.iter().zip(&b).map(|(&x,&y)|minus(x,y)).collect::<Vec<_>>());
        if !da.is_finite()||!db.is_finite(){return Err("Nonfinite complete independent Schur reconstruction".into());}
        let mut frame=Self{left_in:qah,right_in_t:transpose(&qb,n),left_out:qa,right_out_t:transpose(&qbh,n),kind:FrameKind::Schur{ta,tb},uhat:vec![],
            diagnostics:FrameDiagnostics{backend:"complete-Schur",left_condition_upper:None,right_condition_upper:None,left_reconstruction_hs:da,right_reconstruction_hs:db,
                transformed_u_columns:factor.rank(),hilbert_factorizations:2}};
        frame.prepare_u(factor,lap)?;Ok(frame)
    }
    fn prepare_u(&mut self,factor:&B0Factor,lap:&Lapack)->Result<(),String>{
        let mut w=P0Workspace::default();transform(lap,factor.hilbert_dimension(),&self.left_in,&self.right_in_t,&dense_u(factor),factor.rank(),&mut self.uhat,
            &mut w.pack_a,&mut w.pack_b,&mut w.middle,&mut w.last)
    }
}
fn solve_reduced(frame:&Frame,solve:&ShiftSolve,lap:&Lapack,n:usize,rhs:usize,reduced:&mut[C])->Result<(),String>{
    match(solve,&frame.kind){
        (ShiftSolve::Eigen{reciprocal},FrameKind::Eigen{..})=>{for i in 0..n*n{for b in 0..rhs{reduced[i*rhs+b]=mul(reduced[i*rhs+b],reciprocal[i]);}}},
        (ShiftSolve::Schur{left},FrameKind::Schur{tb,..})=>{for b in 0..rhs{
            let mut column=vec![C::default();n*n];for i in 0..n{for j in 0..n{column[i+j*n]=reduced[(i*n+j)*rhs+b];}}
            let solved=lap.sylvester_kernel().complex_sylvester(left,tb,&column,n)?;
            for i in 0..n{for j in 0..n{reduced[(i*n+j)*rhs+b]=solved[i+j*n];}}
        }},
        _=>return Err("Shift factor does not belong to its no-jump frame".into()),
    }
    if reduced.iter().any(|z|!z.finite()){return Err("Nonfinite complete shifted no-jump action".into());}Ok(())
}
impl ShiftFactor{
    fn prepare(frame:Arc<Frame>,factor:&B0Factor,lap:&Lapack,shift:C,fallback_reason:Option<String>)->Result<Self,String>{
        let n=factor.hilbert_dimension();let d=n*n;let rank=factor.rank();let mut minimum=None;let mut indicator=None;
        let solve=match &frame.kind{
            FrameKind::Eigen{a,b}=>{let mut values=vec![C::default();d];let mut smallest=f64::INFINITY;
                for i in 0..n{for j in 0..n{let denominator=C{re:sum3(shift.re,-a[i].re,-b[j].re),im:sum3(shift.im,-a[i].im,-b[j].im)};
                    smallest=smallest.min(denominator.re.hypot(denominator.im));values[i*n+j]=reciprocal(denominator)?;}}
                let diag=&frame.diagnostics;let perturb=diag.left_condition_upper.unwrap()*diag.right_condition_upper.unwrap()
                    *(diag.left_reconstruction_hs+diag.right_reconstruction_hs)/smallest;
                if !perturb.is_finite()||perturb>1e-4{return Err(format!("Eigen reconstruction/shift perturbation indicator unresolved: {perturb:e}"));}
                minimum=Some(smallest);indicator=Some(perturb);ShiftSolve::Eigen{reciprocal:values}
            },
            FrameKind::Schur{ta,..}=>{let mut left:Vec<_>=ta.iter().map(|z|C{re:-z.re,im:-z.im}).collect();for i in 0..n{add(&mut left[i+i*n],shift);}
                if left.iter().any(|z|!z.finite()){return Err("Complete shifted Schur matrix overflow".into());}ShiftSolve::Schur{left}},
        };
        let mut reduced=frame.uhat.clone();solve_reduced(&frame,&solve,lap,n,rank,&mut reduced)?;let mut w=P0Workspace::default();let mut ru_rows=Vec::new();
        transform(lap,n,&frame.left_out,&frame.right_out_t,&reduced,rank,&mut ru_rows,&mut w.pack_a,&mut w.pack_b,&mut w.middle,&mut w.last)?;
        let ru=column(&ru_rows,d,rank);let mut small=identity(rank);
        for i in 0..rank{for at in factor.v.row[i]..factor.v.row[i+1]{for j in 0..rank{
            let value=mul(factor.v.value[at],ru[factor.v.col[at]+j*d]);small[i+j*rank].re-=value.re;small[i+j*rank].im-=value.im;
        }}}
        if small.iter().any(|z|!z.finite()){return Err("Nonfinite complete Woodbury feedback before factorization".into());}
        let(mut condition,mut residual)=(None,None);
        let feedback=if rank==0{Feedback::Empty}else if matches!(solve,ShiftSolve::Eigen{..}){
            let(inv,_,_)=inverse(lap,&small,rank)?;let cond=condition_upper(&small,&inv,rank);let err=inverse_error(lap,&small,&inv,rank)?;
            if !cond.is_finite()||cond>1e8||!err.is_finite()||err>1e-8{return Err(format!("Complete small eigenframe feedback unresolved: condition={cond:e}, residual={err:e}"));}
            condition=Some(cond);residual=Some(err);Feedback::Inverse(inv)
        }else{let(matrix,pivots)=lap.complex_factor(&small,rank)?;Feedback::Lu{matrix,pivots}};
        let diagnostics=ShiftDiagnostics{shift,backend:frame.diagnostics.backend,fallback_reason,minimum_denominator:minimum,
            eigen_perturbation_indicator:indicator,feedback_condition_upper:condition,feedback_inverse_residual_inf:residual,all_feedback_columns:rank};
        Ok(Self{frame,solve,ru,feedback,diagnostics})
    }
}
impl RequestP0{
    fn new(factor:Arc<B0Factor>,lapack:Arc<Lapack>,policy:P0Policy)->Result<Self,String>{
        let began=Instant::now();let rank=factor.rank();
        let(eigen,eigen_failure)=if policy==P0Policy::EigenPreferred{match Frame::eigen(&factor,&lapack){Ok(frame)=>(Some(Arc::new(frame)),None),Err(reason)=>(None,Some(reason))}}
            else{(None,Some("Explicit complete-Schur policy".into()))};
        let mut result=Self{factor,lapack,eigen,schur:None,eigen_failure,entries:BTreeMap::new(),profile:RequestPreparationProfile{feedback_columns:rank,..RequestPreparationProfile::default()}};
        if result.eigen.is_none(){result.schur=Some(Arc::new(Frame::schur(&result.factor,&result.lapack)?));}
        result.profile.initial_frame_ms=began.elapsed().as_secs_f64()*1000.;Ok(result)
    }
    pub fn factor(&self)->&B0Factor{&self.factor}
    pub fn belongs_to(&self,prepared:&OwnedPrepared)->bool{Arc::ptr_eq(&self.factor.owner,&prepared.primitives)}
    pub fn prepared_shift_count(&self)->usize{self.entries.len()}
    pub fn preparation_profile(&self)->&RequestPreparationProfile{&self.profile}
    pub(super) fn left_selector_frame(&self)->Option<(Vec<C>,Vec<C>,Vec<C>)>{
        self.eigen.as_ref().and_then(|frame|if let FrameKind::Eigen{a,..}=&frame.kind{Some((a.clone(),frame.left_out.clone(),frame.left_in.clone()))}else{None})
    }
    pub fn eigen_frame_diagnostics(&self)->Option<&FrameDiagnostics>{self.eigen.as_ref().map(|x|&x.diagnostics)}
    pub fn schur_frame_diagnostics(&self)->Option<&FrameDiagnostics>{self.schur.as_ref().map(|x|&x.diagnostics)}
    pub fn prepare_batch(&mut self,shifts:&[C],max_rhs:usize)->Result<P0Batch,String>{
        if shifts.is_empty()||max_rhs==0||shifts.iter().any(|z|!z.finite()){return Err("Expected nonempty finite complex shifts and positive max_rhs".into());}
        let mut entries=Vec::with_capacity(shifts.len());
        for &shift in shifts{let key=(shift.re.to_bits(),shift.im.to_bits());
            if !self.entries.contains_key(&key){
                let began=Instant::now();let mut extra_frame_ms=0.;
                let candidate=if let Some(frame)=&self.eigen{ShiftFactor::prepare(Arc::clone(frame),&self.factor,&self.lapack,shift,None)}
                    else{Err(self.eigen_failure.clone().unwrap_or_else(||"Eigenframe not selected".into()))};
                let value=match candidate{Ok(value)=>value,Err(reason)=>{
                    if self.schur.is_none(){let frame_begin=Instant::now();self.schur=Some(Arc::new(Frame::schur(&self.factor,&self.lapack)?));
                        extra_frame_ms=frame_begin.elapsed().as_secs_f64()*1000.;self.profile.lazy_schur_frame_ms+=extra_frame_ms;}
                    ShiftFactor::prepare(Arc::clone(self.schur.as_ref().unwrap()),&self.factor,&self.lapack,shift,Some(reason))?
                }};
                self.entries.insert(key,Arc::new(value));
                self.profile.shift_factors_ms+=(began.elapsed().as_secs_f64()*1000.-extra_frame_ms).max(0.);self.profile.unique_shifts=self.entries.len();
            }else{self.profile.shift_cache_hits+=1;}
            entries.push(Arc::clone(&self.entries[&key]));
        }
        Ok(P0Batch{factor:Arc::clone(&self.factor),lapack:Arc::clone(&self.lapack),entries,max_rhs,workspace:P0Workspace::default()})
    }
}
impl P0Batch{
    pub fn shift_diagnostics(&self)->Vec<&ShiftDiagnostics>{self.entries.iter().map(|x|&x.diagnostics).collect()}
    pub fn dimension(&self)->usize{self.entries.len()*self.factor.dimension()}
    pub fn apply_into(&mut self,input:&[C],rhs:usize,output:&mut[C])->Result<(),String>{
        if rhs==0||rhs>self.max_rhs||input.len()!=product(&[self.dimension(),rhs])?||output.len()!=input.len()||input.iter().any(|z|!z.finite()){
            return Err("Invalid complete P0 batch shape, RHS count or finite input".into());}
        let n=self.factor.hilbert_dimension();let d=n*n;let rank=self.factor.rank();let w=&mut self.workspace;
        resize(&mut w.block,d*rhs);resize(&mut w.beta,rank*rhs);resize(&mut w.amount,rank*rhs);resize(&mut w.correction,d*rhs);
        for(mode,entry)in self.entries.iter().enumerate(){
            w.block.copy_from_slice(&input[mode*d*rhs..(mode+1)*d*rhs]);let frame=&entry.frame;
            transform(&self.lapack,n,&frame.left_in,&frame.right_in_t,&w.block,rhs,&mut w.reduced,&mut w.pack_a,&mut w.pack_b,&mut w.middle,&mut w.last)?;
            solve_reduced(frame,&entry.solve,&self.lapack,n,rhs,&mut w.reduced)?;
            transform(&self.lapack,n,&frame.left_out,&frame.right_out_t,&w.reduced,rhs,&mut w.base,&mut w.pack_a,&mut w.pack_b,&mut w.middle,&mut w.last)?;
            w.beta.fill(C::default());for i in 0..rank{for at in self.factor.v.row[i]..self.factor.v.row[i+1]{for b in 0..rhs{
                add(&mut w.beta[i+b*rank],mul(self.factor.v.value[at],w.base[self.factor.v.col[at]*rhs+b]));
            }}}
            match &entry.feedback{
                Feedback::Empty=>{},
                Feedback::Inverse(inv)=>self.lapack.multiply_complex_into(inv,&w.beta,&mut w.amount,rank,rhs,rank)?,
                Feedback::Lu{matrix,pivots}=>{for b in 0..rhs{w.amount[b*rank..(b+1)*rank].copy_from_slice(&self.lapack.complex_solve_factored(matrix,pivots,&w.beta[b*rank..(b+1)*rank],rank)?);}},
            }
            if rank>0{self.lapack.multiply_complex_into(&entry.ru,&w.amount,&mut w.correction,d,rhs,rank)?;
                for i in 0..d{for b in 0..rhs{add(&mut w.base[i*rhs+b],w.correction[i+b*d]);}}}
            if w.base.iter().any(|z|!z.finite()){return Err("Prepared complete P0 produced nonfinite output; backend is not changed during apply".into());}
            output[mode*d*rhs..(mode+1)*d*rhs].copy_from_slice(&w.base);
        }Ok(())
    }
}
impl OwnedPrepared{
    pub fn prepare_p0(&self,policy:P0Policy)->Result<RequestP0,String>{
        let began=Instant::now();let factor=Arc::new(self.prepare_b0_factor()?);let factor_ms=began.elapsed().as_secs_f64()*1000.;
        let mut request=RequestP0::new(factor,Arc::clone(&self.preparation_lapack),policy)?;request.profile.factor_ms=factor_ms;Ok(request)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn known_solution_check(request:&mut RequestP0,shifts:&[C]){
        let d=request.factor.dimension();let rhs=3;let known:Vec<_>=(0..shifts.len()*d*rhs).map(|i|C{re:((i*7+3)%31)as f64/37.,im:((i*17+5)%43)as f64/-47.}).collect();
        let mut input=vec![C::default();known.len()];let mut action=vec![C::default();d*rhs];
        for(mode,&shift)in shifts.iter().enumerate(){let x=&known[mode*d*rhs..(mode+1)*d*rhs];request.factor.apply_into(x,rhs,&mut action).unwrap();
            for i in 0..d*rhs{input[mode*d*rhs+i]=minus(mul(shift,x[i]),action[i]);}}
        let mut batch=request.prepare_batch(shifts,rhs).unwrap();let mut actual=vec![C::default();known.len()];batch.apply_into(&input,rhs,&mut actual).unwrap();
        let difference=norm(&actual.iter().zip(&known).map(|(&a,&b)|minus(a,b)).collect::<Vec<_>>())/norm(&known);assert!(difference<2e-10,"complete B0 solve error {difference:e}");
        let previous=actual.clone();batch.apply_into(&vec![C::default();input.len()],rhs,&mut actual).unwrap();assert!(actual.iter().all(|z|z.re==0.&&z.im==0.));assert!(previous.iter().any(|z|z.re!=0.));
    }
    #[test]
    fn preparation_b0_inverse_complex_shifts_multirhs_and_request_owned_reuse(){
        let owner=super::super::tests::fixture();let mut request=owner.prepare_p0(P0Policy::EigenPreferred).unwrap();
        assert!(request.eigen_frame_diagnostics().is_some(),"Normal fixture must exercise the eigenframe, not only its fallback");
        let shifts=[C{re:8.,im:19.},C{re:8.,im:-31.},C{re:8.,im:19.}];known_solution_check(&mut request,&shifts);assert_eq!(request.prepared_shift_count(),2);
        let a=request.prepare_batch(&shifts,3).unwrap();let b=request.prepare_batch(&shifts[1..],3).unwrap();
        assert!(a.shift_diagnostics().iter().all(|d|d.backend=="eigen"));
        assert!(Arc::ptr_eq(&a.entries[0],&a.entries[2]));assert!(Arc::ptr_eq(&a.entries[1],&b.entries[0]));
        let separate=owner.prepare_p0(P0Policy::SchurOnly).unwrap();assert!(!Arc::ptr_eq(&request.factor,&separate.factor));
        let mut schur=separate;known_solution_check(&mut schur,&shifts);assert!(schur.eigen_frame_diagnostics().is_none());
        let mut retained=a;drop(owner);drop(request);let mut out=vec![C::default();retained.dimension()];retained.apply_into(&vec![C{re:1.,im:0.2};out.len()],1,&mut out).unwrap();
    }
    #[test]
    fn preparation_b0_inverse_defective_independent_right_uses_complete_schur(){
        let owner=super::super::tests::fixture();let mut factor=owner.prepare_b0_factor().unwrap();let n=factor.hilbert_dimension();
        factor.left.fill(C::default());factor.right.fill(C::default());
        for i in 0..n{factor.left[i*n+i]=C{re:-0.5,im:2.};factor.right[i*n+i]=C{re:-0.7-i as f64*0.1,im:-0.3*i as f64};
            if i+1<n{factor.left[i*n+i+1]=C{re:1.,im:0.2};factor.right[(i+1)*n+i]=C{re:0.1,im:-0.07};}}
        let mut request=RequestP0::new(Arc::new(factor),Arc::clone(&owner.preparation_lapack),P0Policy::EigenPreferred).unwrap();
        assert!(request.eigen_frame_diagnostics().is_none());assert!(request.schur_frame_diagnostics().is_some());
        let shifts=[C{re:8.,im:3.7},C{re:9.,im:-2.1}];known_solution_check(&mut request,&shifts);
        assert!(request.prepare_batch(&shifts,1).unwrap().shift_diagnostics().iter().all(|d|d.backend=="complete-Schur"&&d.fallback_reason.is_some()));
    }
}
