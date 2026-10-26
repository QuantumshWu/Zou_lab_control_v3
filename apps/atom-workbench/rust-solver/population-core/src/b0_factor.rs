//! The SAME numerical FactorB0 reference as the current CPU solver.
//! It is NOT the physical generator and NOT E0^{-1}F0. Complete original
//! E/F remains the independent physical action. No feedback rank is dropped.
use super::*;

#[derive(Clone,Debug)]
pub struct B0Diagnostics{
    pub hilbert_dimension:usize,pub ground_feedback_columns:usize,pub feedback_columns:usize,
    pub optical_feedback_groups:usize,pub feedback_modes:usize,
    pub oscillating_gamma_feedback_hs:f64,
    pub left_right_adjoint_difference_hs:f64,
    pub original_physical_generator_changed:bool,
}
pub struct B0Factor{
    pub(super) owner:Arc<PrimitivePreparation>,
    pub(super) left:Vec<C>,pub(super) right:Vec<C>,
    pub(super) u:Coefficient,pub(super) v:Coefficient,
    pub(super) diagnostics:B0Diagnostics,
}
fn add(a:&mut C,b:C){a.re+=b.re;a.im+=b.im;}
fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
fn scaled(a:C,s:f64)->C{C{re:a.re*s,im:a.im*s}}
fn minus_i(a:C)->C{C{re:a.im,im:-a.re}}
fn plus_i(a:C)->C{C{re:-a.im,im:a.re}}
fn conj(a:C)->C{C{re:a.re,im:-a.im}}
fn norm(a:&[C])->f64{a.iter().fold(0_f64,|v,z|v.hypot(z.re.hypot(z.im)))}
fn row_to_column(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|i|a[(i%n)*n+i/n]).collect()}
fn mm(lap:&Lapack,a:&[C],b:&[C],n:usize)->Result<Vec<C>,String>{
    let result=lap.multiply_complex(&row_to_column(a,n),&row_to_column(b,n),n,n,n)?;Ok(row_to_column(&result,n))
}
fn comm(lap:&Lapack,a:&[C],b:&[C],n:usize)->Result<Vec<C>,String>{
    let mut result=mm(lap,a,b,n)?;let right=mm(lap,b,a,n)?;for(x,y)in result.iter_mut().zip(right){x.re-=y.re;x.im-=y.im;}Ok(result)
}
fn add_family(out:&mut FourierMatrices,q:Charge,value:&[C]){let target=out.entry(q).or_insert_with(||vec![C::default();value.len()]);for(a,&b)in target.iter_mut().zip(value){add(a,b);}}
fn comm_family(lap:&Lapack,a:&FourierMatrices,b:&FourierMatrices,n:usize)->Result<FourierMatrices,String>{
    let mut output=FourierMatrices::new();for(q,x)in a{for(p,y)in b{add_family(&mut output,add_charge(q,p,false)?,&comm(lap,x,y,n)?);}}Ok(output)
}
fn derivative(a:&FourierMatrices,phases:&PhaseBasis)->Result<FourierMatrices,String>{
    a.iter().map(|(q,value)|{let w=phases.frequency(q)?;let omega=w.hi+w.lo;
        Ok((q.clone(),value.iter().map(|&z|scaled(plus_i(z),omega)).collect()))}).collect()
}
/// This finite Hamiltonian expression is used solely by the existing P0.
/// Its approximation is NOT substituted into the original physical action.
fn reference_hamiltonian(lap:&Lapack,p:&PrimitivePreparation)->Result<Vec<C>,String>{
    let source=&p.hilbert.source;let n=source.n;let zero=vec![0;source.phases.axes.len()];
    let ds=derivative(&p.s,&source.phases)?;let hs=comm_family(lap,&source.hamiltonian,&p.s,n)?;
    let hss=comm_family(lap,&hs,&p.s,n)?;let dss=comm_family(lap,&ds,&p.s,n)?;
    let mut result=source.hamiltonian.get(&zero).cloned().unwrap_or_else(||vec![C::default();n*n]);
    for(family,weight)in[(&hs,C{re:1.,im:0.}),(&hss,C{re:0.5,im:0.}),(&ds,C{re:0.,im:-1.}),(&dss,C{re:0.,im:-0.5})]{
        if let Some(value)=family.get(&zero){for(a,&b)in result.iter_mut().zip(value){add(a,mul(weight,b));}}
    }Ok(result)
}

impl B0Factor{
    pub(super) fn prepare(owner:Arc<PrimitivePreparation>,lap:&Lapack)->Result<Self,String>{
        let p=&owner;let source=&p.hilbert.source;let n=source.n;let d=n*n;let g=source.ground.len();let e=n-g;let e2=e*e;
        let zero=vec![0;source.phases.axes.len()];let rank=p.gg_rows.len();let optical=p.optical_rows.len();
        let h0=reference_hamiltonian(lap,p)?;let gamma0=p.hilbert.radiation.gamma.get(&zero);
        let mut left:Vec<_>=h0.iter().copied().map(minus_i).collect();let mut right:Vec<_>=h0.iter().copied().map(plus_i).collect();
        if let Some(gamma)=gamma0{for i in 0..e{for j in 0..e{let a=scaled(gamma[i*e+j],-0.5);add(&mut left[(g+i)*n+g+j],a);add(&mut right[(g+i)*n+g+j],a);}}}
        if left.iter().chain(&right).any(|z|!z.finite()){return Err("Nonfinite complete B0 left/right reference".into());}
        let mut gg_ee=p.selected_recycling.get(&zero).cloned().unwrap_or_else(||vec![C::default();rank*e2]);
        let mut gg_optical=vec![C::default();rank*optical];
        let mut extra_u=Vec::<Vec<C>>::new();let mut extra_v=Vec::<Vec<C>>::new();let mut groups=0;let mut gamma_size=0.;
        for(q,k)in &p.selected_k{
            let opposite:Charge=q.iter().map(|&x|x.checked_neg().ok_or("B0 feedback charge overflow")).collect::<Result<_,_>>()?;
            let h=source.hamiltonian.get(&opposite);let gamma=p.hilbert.radiation.gamma.get(&opposite);
            if let Some(value)=gamma{gamma_size+=norm(value);}
            let get_h=|i:usize,j:usize|h.map_or(C::default(),|v|v[i*n+j]);
            let get_gamma=|i:usize,j:usize|gamma.map_or(C::default(),|v|v[i*e+j]);
            // Complete L_GG,-q K_q, with the previously established exact
            // GG closure. No raw support assumption or new rank truncation.
            if let Some(lgg)=p.gg_hamiltonian.get(&opposite){for a in 0..rank{for b in 0..rank{let value=lgg[a*rank+b];
                for ij in 0..e2{add(&mut gg_ee[a*e2+ij],mul(value,k[b*e2+ij]));}
            }}}
            // -K_q L_EE,-q retains ALL original oscillating Gamma entries.
            for row in 0..rank{for i in 0..e{for j in 0..e{let mut value=C::default();
                for a in 0..e{
                    let mut aa=minus_i(get_h(g+a,g+i));add(&mut aa,scaled(get_gamma(a,i),-0.5));
                    let mut bb=plus_i(get_h(g+j,g+a));add(&mut bb,scaled(get_gamma(j,a),-0.5));
                    add(&mut value,mul(k[row*e2+a*e+j],aa));add(&mut value,mul(k[row*e2+i*e+a],bb));
                }add(&mut gg_ee[row*e2+i*e+j],scaled(value,-1.));
            }}}
            // -K_q L_EE,O,-q is a complex-linear functional. No conjugation
            // of independent GE/EG coordinates is permitted here.
            for row in 0..rank{for(col,&index)in p.optical_rows.iter().enumerate(){let(i,j)=(index/n,index%n);let mut value=C::default();
                if i<g{for a in 0..e{add(&mut value,plus_i(mul(k[row*e2+a*e+j-g],get_h(g+a,i))));}}
                else{for b in 0..e{add(&mut value,minus_i(mul(k[row*e2+(i-g)*e+b],get_h(j,g+b))));}}
                add(&mut gg_optical[row*optical+col],value);
            }}
            // The optical feedback factorization keeps each exact active
            // coordinate column, never an SVD/threshold-selected rank.
            let mut group_added=false;if let Some(og)=p.optical_hamiltonian.get(&opposite){for row in 0..rank{
                let active_left=(0..optical).any(|i|{let z=og[i*rank+row];z.re!=0.||z.im!=0.});
                let active_right=k[row*e2..(row+1)*e2].iter().any(|z|z.re!=0.||z.im!=0.);
                if active_left&&active_right{
                    let mut u=vec![C::default();d];let mut v=vec![C::default();d];
                    for(i,&index)in p.optical_rows.iter().enumerate(){u[index]=og[i*rank+row];}
                    for i in 0..e{for j in 0..e{v[(g+i)*n+g+j]=k[row*e2+i*e+j];}}
                    extra_u.push(u);extra_v.push(v);group_added=true;
                }
            }}if group_added{groups+=1;}
        }
        let all_rank=rank+extra_u.len();let mut u=vec![C::default();product(&[d,all_rank])?];let mut v=vec![C::default();product(&[all_rank,d])?];
        for(row,&ab)in p.gg_rows.iter().enumerate(){u[((ab/g)*n+ab%g)*all_rank+row].re=1.;
            for i in 0..e{for j in 0..e{v[row*d+(g+i)*n+g+j]=gg_ee[row*e2+i*e+j];}}
            for(i,&index)in p.optical_rows.iter().enumerate(){v[row*d+index]=gg_optical[row*optical+i];}
        }
        for(j,(a,b))in extra_u.into_iter().zip(extra_v).enumerate(){for i in 0..d{u[i*all_rank+rank+j]=a[i];v[(rank+j)*d+i]=b[i];}}
        let difference:Vec<_>=(0..d).map(|i|{let a=conj(left[(i%n)*n+i/n]);C{re:right[i].re-a.re,im:right[i].im-a.im}}).collect();
        let diagnostics=B0Diagnostics{hilbert_dimension:n,ground_feedback_columns:rank,feedback_columns:all_rank,
            optical_feedback_groups:groups,feedback_modes:p.selected_k.len(),oscillating_gamma_feedback_hs:gamma_size,
            left_right_adjoint_difference_hs:norm(&difference),original_physical_generator_changed:false};
        Ok(Self{owner,left,right,u:coefficient(&u,d,all_rank,KernelPolicy::Csr)?,v:coefficient(&v,all_rank,d,KernelPolicy::Csr)?,diagnostics})
    }
    pub fn diagnostics(&self)->&B0Diagnostics{&self.diagnostics}
    pub fn dimension(&self)->usize{self.left.len()}
    pub fn hilbert_dimension(&self)->usize{self.owner.hilbert.source.n}
    pub fn rank(&self)->usize{self.v.rows}
    /// Independent complete B0 application; useful for preparation/linear
    /// inverse validation only. It is never a substitute for physical E/F.
    pub fn apply_into(&self,input:&[C],rhs:usize,output:&mut[C])->Result<(),String>{
        let n=self.hilbert_dimension();let d=n*n;
        if rhs==0||input.len()!=product(&[d,rhs])?||output.len()!=input.len()||input.iter().any(|z|!z.finite()){
            return Err("Invalid complete B0 column dimensions/input".into());}
        output.fill(C::default());for i in 0..n{for j in 0..n{for k in 0..n{for b in 0..rhs{
            add(&mut output[(i*n+j)*rhs+b],mul(self.left[i*n+k],input[(k*n+j)*rhs+b]));
            add(&mut output[(i*n+j)*rhs+b],mul(input[(i*n+k)*rhs+b],self.right[k*n+j]));
        }}}}
        let mut amount=vec![C::default();self.rank()*rhs];
        for i in 0..self.rank(){for at in self.v.row[i]..self.v.row[i+1]{for b in 0..rhs{
            add(&mut amount[i*rhs+b],mul(self.v.value[at],input[self.v.col[at]*rhs+b]));
        }}}
        for i in 0..d{for at in self.u.row[i]..self.u.row[i+1]{for b in 0..rhs{
            add(&mut output[i*rhs+b],mul(self.u.value[at],amount[self.u.col[at]*rhs+b]));
        }}}
        if output.iter().any(|z|!z.finite()){return Err("Nonfinite complete B0 action".into());}Ok(())
    }
}

impl OwnedPrepared{
    pub fn prepare_b0_factor(&self)->Result<B0Factor,String>{B0Factor::prepare(Arc::clone(&self.primitives),&self.preparation_lapack)}
}

#[cfg(test)]
mod tests{
    use super::*;use crate::preparation::MatrixView;
    pub(super) fn fixture()->OwnedPrepared{
        let n=4;let mut h=vec![0.;16];let zero=vec![0.;16];for(i,v)in[0.,0.6,3.,4.].iter().enumerate(){h[i*n+i]=*v;}
        for(i,j,v)in[(0,2,0.1),(1,3,0.2),(0,1,0.02)]{h[i*n+j]=v;h[j*n+i]=v;}
        let mut c=vec![0.;16];c[2]=0.7;c[3]=0.1;c[6]=0.2;c[7]=0.8;
        let channels=[MatrixView{real:&c,imag:&zero}];let model=ModelView{dimension:n,hamiltonian:MatrixView{real:&h,imag:&zero},
            collapse:&channels,drives:&[],ground:&[true,true,false,false],initial_index:0};
        let phases=PhaseBasis::from_declared(&[7.],None,&[]).unwrap();let gauge=[vec![0],vec![1],vec![0],vec![2]];
        OwnedPrepared::prepare(&model,phases,GaugePolicy::Explicit(&gauge),KernelPolicy::Csr).unwrap()
    }
    fn k_action(k:&[C],x:&[C],n:usize,g:usize)->Vec<C>{let e=n-g;let mut y=vec![C::default();n*n];
        for a in 0..g{for b in 0..g{for i in 0..e{for j in 0..e{add(&mut y[a*n+b],mul(k[(a*g+b)*e*e+i*e+j],x[(g+i)*n+g+j]));}}}}y}
    fn physical_l(p:&PrimitivePreparation,q:&Charge,x:&[C],lap:&Lapack)->Vec<C>{
        let n=p.hilbert.source.n;let g=p.hilbert.source.ground.len();let e=n-g;let mut out=vec![C::default();n*n];
        if let Some(h)=p.hilbert.source.hamiltonian.get(q){out=comm(lap,h,x,n).unwrap().into_iter().map(minus_i).collect();}
        if let Some(ge)=p.hilbert.radiation.gamma.get(q){let mut full=vec![C::default();n*n];for i in 0..e{for j in 0..e{full[(g+i)*n+g+j]=ge[i*e+j];}}
            let a=mm(lap,&full,x,n).unwrap();let b=mm(lap,x,&full,n).unwrap();for i in 0..n*n{add(&mut out[i],scaled(a[i],-0.5));add(&mut out[i],scaled(b[i],-0.5));}}
        if let Some(r)=p.hilbert.radiation.recycling.get(q){for(a,b)in out.iter_mut().zip(k_action(r,x,n,g)){add(a,b);}}out
    }
    #[test]
    fn preparation_b0_factor_matches_complete_original_feedback_action(){
        let owner=fixture();let factor=owner.prepare_b0_factor().unwrap();let p=owner.primitives();let lap=&owner.preparation_lapack;
        let n=p.hilbert.source.n;let d=n*n;let g=p.hilbert.source.ground.len();let zero=vec![0;p.hilbert.source.phases.axes.len()];
        assert!(factor.diagnostics.oscillating_gamma_feedback_hs>0.);assert!(factor.diagnostics.optical_feedback_groups>0);
        let rhs=3;let x:Vec<_>=(0..d*rhs).map(|i|C{re:((i*7+1)%23)as f64/17.,im:((i*13+2)%31)as f64/-29.}).collect();let mut actual=vec![C::default();x.len()];factor.apply_into(&x,rhs,&mut actual).unwrap();
        for b in 0..rhs{let field:Vec<_>=(0..d).map(|i|x[i*rhs+b]).collect();let mut expected=mm(lap,&factor.left,&field,n).unwrap();let right=mm(lap,&field,&factor.right,n).unwrap();
            for i in 0..d{add(&mut expected[i],right[i]);}
            if let Some(r)=p.hilbert.radiation.recycling.get(&zero){for(a,z)in expected.iter_mut().zip(k_action(r,&field,n,g)){add(a,z);}}
            for(q,k)in &p.k_hat{let opposite=q.iter().map(|&i|-i).collect();let kx=k_action(k,&field,n,g);
                let lk=physical_l(p,&opposite,&kx,lap);let lx=physical_l(p,&opposite,&field,lap);let kl=k_action(k,&lx,n,g);
                for i in 0..d{add(&mut expected[i],lk[i]);add(&mut expected[i],scaled(kl[i],-1.));}
            }
            for i in 0..d{let a=actual[i*rhs+b];let z=expected[i];assert!((a.re-z.re).hypot(a.im-z.im)<3e-12*(1.+z.re.hypot(z.im)));}
        }
        let before=actual.clone();factor.apply_into(&vec![C::default();x.len()],rhs,&mut actual).unwrap();assert!(actual.iter().all(|z|z.re==0.&&z.im==0.));assert!(before.iter().any(|z|z.re!=0.));
    }
}
