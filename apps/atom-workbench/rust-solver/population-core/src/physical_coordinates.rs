//! Original Model initial data and complete physical observations for the
//! SAME current-request coordinates T(theta)=(I+Khat(theta)) J2(S(theta)).
//! No population is clipped, normalized, made real, or fed back into evolution.
use super::*;

/// Matches the existing production Model protocol exactly. `index` is the
/// GLOBAL state index, not an index into the ground-state subset. Both full
/// row-major real/imaginary arrays must be present together or both absent.
#[derive(Clone,Copy)]
pub struct InitialView<'a>{pub index:usize,pub real:Option<&'a[f64]>,pub imag:Option<&'a[f64]>}

#[derive(Clone,Debug)]
pub struct InitialDiagnostics{
    pub full_density_supplied:bool,
    pub original_trace_error:f64,pub original_hermiticity_error:f64,
    pub point_s_antihermitian_error_hs:f64,pub minimum_j2_eigenvalue_modulus:f64,
    pub original_coordinate_forward_error:f64,
    pub original_density_modified:bool,
}
pub struct InitialCoordinates{original:Vec<C>,point:Vec<C>,diagnostics:InitialDiagnostics}
impl InitialCoordinates{
    pub fn original(&self)->&[C]{&self.original}
    /// Constant-envelope initial y0. Its physical image at theta=0 equals
    /// original rho0; away from that phase this is a numerical extension.
    pub fn point(&self)->&[C]{&self.point}
    pub fn diagnostics(&self)->&InitialDiagnostics{&self.diagnostics}
}

pub struct PhysicalCoordinates{primitives:Arc<PrimitivePreparation>,lapack:Arc<Lapack>}
fn add(a:&mut C,b:C){a.re+=b.re;a.im+=b.im;}
fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
fn conj(a:C)->C{C{re:a.re,im:-a.im}}
fn scaled(a:C,b:f64)->C{C{re:a.re*b,im:a.im*b}}
fn norm(a:&[C])->f64{a.iter().fold(0_f64,|v,z|v.hypot(z.re.hypot(z.im)))}
fn maximum(a:&[C])->f64{a.iter().map(|z|z.re.hypot(z.im)).fold(0_f64,f64::max)}
fn row_to_column(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|i|a[(i%n)*n+i/n]).collect()}
fn column_to_row(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|i|a[(i%n)*n+i/n]).collect()}
fn matmul(lap:&Lapack,a:&[C],b:&[C],n:usize)->Result<Vec<C>,String>{
    Ok(column_to_row(&lap.multiply_complex(&row_to_column(a,n),&row_to_column(b,n),n,n,n)?,n))
}
fn commutator(lap:&Lapack,s:&[C],x:&[C],n:usize)->Result<Vec<C>,String>{
    let mut left=matmul(lap,s,x,n)?;let right=matmul(lap,x,s,n)?;
    for(a,b)in left.iter_mut().zip(right){a.re-=b.re;a.im-=b.im;}Ok(left)
}
fn evaluate(family:&FourierMatrices,phases:&PhaseBasis,time:f64,width:usize)->Result<Vec<C>,String>{
    let mut result=vec![C::default();width];
    for(q,value)in family{if value.len()!=width{return Err("Physical coefficient shape differs from its owner".into());}
        let phase=phases.phase(q,time)?;for(a,&b)in result.iter_mut().zip(value){add(a,mul(phase,b));}}
    if result.iter().any(|z|!z.finite()){return Err("Nonfinite physical coefficient evaluation".into());}Ok(result)
}
fn shear_action(k:&[C],x:&[C],n:usize,g:usize)->Vec<C>{
    let e=n-g;let mut output=vec![C::default();n*n];
    for a in 0..g{for b in 0..g{for i in 0..e{for j in 0..e{
        add(&mut output[a*n+b],mul(k[(a*g+b)*e*e+i*e+j],x[(g+i)*n+g+j]));
    }}}}output
}
fn restore_hilbert(lap:&Lapack,b:&[C],x:&[C],n:usize)->Result<Vec<C>,String>{
    let adjoint:Vec<_>=(0..n*n).map(|i|conj(b[(i%n)*n+i/n])).collect();
    crate::preparation::hilbert::transform_matrix(lap,&adjoint,x,n)
}

impl PhysicalCoordinates{
    pub(super) fn new(primitives:Arc<PrimitivePreparation>,lapack:Arc<Lapack>)->Self{Self{primitives,lapack}}
    pub fn dimension(&self)->usize{self.primitives.hilbert.source.n}
    pub fn initial(&self,initial:InitialView<'_>)->Result<InitialCoordinates,String>{
        let n=self.dimension();let d=n*n;let full=initial.real.is_some();
        let original=match(initial.real,initial.imag){
            (Some(re),Some(im))=>{
                if re.len()!=d||im.len()!=d{return Err("Full initial density must be n*n row-major real/imaginary arrays".into());}
                let value:Vec<_>=re.iter().zip(im).map(|(&re,&im)|C{re,im}).collect();
                if value.iter().any(|z|!z.finite()){return Err("Nonfinite full initial density".into());}value
            },
            (None,None)=>{if initial.index>=n{return Err("initialIndex is outside the GLOBAL state space".into());}
                let mut value=vec![C::default();d];value[initial.index*n+initial.index].re=1.;value},
            _=>return Err("initialReal and initialImag must be provided together".into()),
        };
        let mut trace=C{re:-1.,im:0.};let mut herm=0_f64;
        for i in 0..n{add(&mut trace,original[i*n+i]);for j in 0..n{
            let(a,b)=(original[i*n+j],conj(original[j*n+i]));herm=herm.max((a.re-b.re).hypot(a.im-b.im));}}
        let trace_error=trace.re.hypot(trace.im);
        // Fixed original-input invariants, independent of any population ε.
        if trace_error>1e-10||herm>1e-10{return Err("Original initial density fails fixed trace/Hermitian input checks".into());}
        let p=&self.primitives;let g=p.hilbert.source.ground.len();let e=n-g;let phases=&p.hilbert.source.phases;
        let rho=crate::preparation::hilbert::transform_matrix(&self.lapack,&p.hilbert.basis_column_major,&original,n)?;
        let k=evaluate(&p.k_hat,phases,0.,g*g*e*e)?;let correction=shear_action(&k,&rho,n,g);
        let before:Vec<_>=rho.iter().zip(correction).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect();
        let s=evaluate(&p.s,phases,0.,d)?;let anti:Vec<_>=(0..d).map(|i|{
            let z=conj(s[(i%n)*n+i/n]);C{re:s[i].re+z.re,im:s[i].im+z.im}}).collect();
        let anti_error=norm(&anti);let construction_scale=p.s.values().map(|a|norm(a)).sum::<f64>();
        if anti_error>256.*f64::EPSILON*n as f64*construction_scale.max(f64::MIN_POSITIVE){
            return Err("Point S is not anti-Hermitian within coefficient accumulation roundoff".into());}
        // The same lower-Hermitian eigenselector as scipy.linalg.eigh(-i*S).
        // It is only used for inversion. Forward validation uses UNMODIFIED S.
        let mut selector:Vec<_>=s.iter().map(|z|C{re:z.im,im:-z.re}).collect();
        for i in 0..n{selector[i*n+i].im=0.;for j in i+1..n{selector[i*n+j]=conj(selector[j*n+i]);}}
        let(energies,vectors)=self.lapack.hermitian_eigen(&row_to_column(&selector,n),n)?;
        let mut transformed=crate::preparation::hilbert::transform_matrix(&self.lapack,&vectors,&before,n)?;
        let mut minimum=f64::INFINITY;
        for i in 0..n{for j in 0..n{let delta=energies[i]-energies[j];let denominator=C{re:1.-0.5*delta*delta,im:delta};
            let modulus=denominator.re.hypot(denominator.im);minimum=minimum.min(modulus);
            if !modulus.is_finite()||modulus==0.{return Err("Unresolved J2 point inverse denominator".into());}
            let normalized=scaled(denominator,1./modulus);transformed[i*n+j]=scaled(mul(transformed[i*n+j],conj(normalized)),1./modulus);
        }}
        let point=restore_hilbert(&self.lapack,&vectors,&transformed,n)?;
        let reconstructed=self.from_envelope_at(&point,0.)?;
        let residual:Vec<_>=reconstructed.iter().zip(&original).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect();
        let forward_error=maximum(&residual);
        if forward_error>1e-10{return Err(format!("Original T point inverse fails fixed initial-state check: {forward_error:e}"));}
        Ok(InitialCoordinates{original,point,diagnostics:InitialDiagnostics{full_density_supplied:full,
            original_trace_error:trace_error,original_hermiticity_error:herm,point_s_antihermitian_error_hs:anti_error,
            minimum_j2_eigenvalue_modulus:minimum,original_coordinate_forward_error:forward_error,original_density_modified:false}})
    }
    /// Complete T(theta), B and state gauge restore for an arbitrary COMPLEX
    /// envelope. No Hermitian-input assumption and no output repair is used.
    pub fn from_envelope_at(&self,envelope:&[C],time:f64)->Result<Vec<C>,String>{
        let p=&self.primitives;let n=self.dimension();let d=n*n;let g=p.hilbert.source.ground.len();let e=n-g;
        if envelope.len()!=d||envelope.iter().any(|z|!z.finite()){return Err("Expected a finite complete complex envelope".into());}
        let phases=&p.hilbert.source.phases;let s=evaluate(&p.s,phases,time,d)?;
        let once=commutator(&self.lapack,&s,envelope,n)?;let twice=commutator(&self.lapack,&s,&once,n)?;
        let mut j=envelope.to_vec();for i in 0..d{add(&mut j[i],once[i]);add(&mut j[i],scaled(twice[i],0.5));}
        let k=evaluate(&p.k_hat,phases,time,g*g*e*e)?;let z=shear_action(&k,&j,n,g);for(a,b)in j.iter_mut().zip(z){add(a,b);}
        let mut physical=restore_hilbert(&self.lapack,&p.hilbert.basis_column_major,&j,n)?;
        let state_phases=p.hilbert.source.gauges.iter().map(|q|phases.phase(q,time)).collect::<Result<Vec<_>,_>>()?;
        for i in 0..n{for j in 0..n{physical[i*n+j]=mul(mul(state_phases[i],physical[i*n+j]),conj(state_phases[j]));}}
        if physical.iter().any(|z|!z.finite()){return Err("Nonfinite physical reconstruction".into());}Ok(physical)
    }
    pub fn field_at(&self,modes:&[Charge],field:&[C],time:f64)->Result<Vec<C>,String>{
        let n=self.dimension();let phases=&self.primitives.hilbert.source.phases;
        mode_set(modes,phases.axes.len())?;if field.len()!=product(&[modes.len(),n,n])?||field.iter().any(|z|!z.finite()){
            return Err("Expected finite complete signed Fourier field".into());}
        let mut envelope=vec![C::default();n*n];
        for(q,values)in modes.iter().zip(field.chunks_exact(n*n)){let phase=phases.phase(q,time)?;for(a,&b)in envelope.iter_mut().zip(values){add(a,mul(phase,b));}}
        self.from_envelope_at(&envelope,time)
    }
    pub fn population_observation(&self)->Result<PopulationObservation,String>{PopulationObservation::prepare(self)}
}

/// Complex-linear bilinear rows O_ab X_ab; X is NEVER conjugated.
pub struct PopulationObservation{
    owner:Arc<PrimitivePreparation>,
    /// Dual charges and, per charge, the sparse rows (state, coordinate, value).
    charges:Vec<Charge>,rows:Vec<Vec<(usize,usize,C)>>,
}
pub struct PopulationBasis{owner:Arc<PrimitivePreparation>,modes:Vec<Charge>,columns:usize,coefficients:Vec<C>}
/// Fixed current-domain/actual-time operator table, never trajectory values.
pub struct PopulationPhasePlan{owner:Arc<PrimitivePreparation>,modes:Vec<Charge>,times:Arc<[f64]>,phases:Vec<C>}
#[derive(Clone)]
pub struct PopulationRows{owner:Arc<PrimitivePreparation>,times:Arc<[f64]>,states:usize,columns:usize,values:Arc<[C]>}
/// Append-only operator-row blocks. Old projected columns are shared without
/// copying or reapplying T; snapshot compacts them only when publishing one
/// immutable Candidate. No propagated population is an input.
#[derive(Clone)]
pub struct PopulationRowBlocks{chunks:Vec<PopulationRows>,columns:usize}

fn add_dual(family:&mut FourierMatrices,q:Charge,value:&[C],scale:f64){
    let out=family.entry(q).or_insert_with(||vec![C::default();value.len()]);for(a,&b)in out.iter_mut().zip(value){add(a,scaled(b,scale));}}
struct DualCommutator{
    n:usize,coefficients:Vec<(Charge,Vec<C>,Vec<C>)>,
    packed:Vec<C>,transposed:Vec<C>,left:Vec<C>,right:Vec<C>,
}
impl DualCommutator{
    fn new(s:&FourierMatrices,n:usize)->Self{
        // S(row-major) is S^T in column-major. For the right action,
        // (Q S^T)^T=S Q^T, so its shared factor is S(column-major).
        let coefficients=s.iter().map(|(q,a)|(q.clone(),a.clone(),row_to_column(a,n))).collect();let length=n*n*n;
        Self{n,coefficients,packed:vec![C::default();length],transposed:vec![C::default();length],left:vec![C::default();length],right:vec![C::default();length]}
    }
    fn apply(&mut self,dual:&FourierMatrices,lap:&Lapack)->Result<FourierMatrices,String>{
        let n=self.n;let d=n*n;let mut out=FourierMatrices::new();
        for(q,values)in dual{if values.len()!=n*d{return Err("Complete observable-row shape differs".into());}
            // Pack ALL goals once for this q, reuse for every S charge.
            for goal in 0..n{for i in 0..n{for j in 0..n{let value=values[goal*d+i*n+j];
                self.packed[i+(goal*n+j)*n]=value;self.transposed[j+(goal*n+i)*n]=value;
            }}}
            for(p,st,s)in &self.coefficients{
                lap.multiply_complex_into(st,&self.packed,&mut self.left,n,n*n,n)?;
                lap.multiply_complex_into(s,&self.transposed,&mut self.right,n,n*n,n)?;
                let target=out.entry(add_charge(q,p,false)?).or_insert_with(||vec![C::default();n*d]);
                for goal in 0..n{for i in 0..n{for j in 0..n{let a=self.left[i+(goal*n+j)*n];let b=self.right[j+(goal*n+i)*n];
                    add(&mut target[goal*d+i*n+j],C{re:a.re-b.re,im:a.im-b.im});
                }}}
            }
        }Ok(out)
    }
}
impl PopulationObservation{
    fn prepare(coordinates:&PhysicalCoordinates)->Result<Self,String>{
        let p=&coordinates.primitives;let n=p.hilbert.source.n;let d=n*n;let g=p.hilbert.source.ground.len();let e=n-g;
        let basis=&p.hilbert.basis_column_major;let zero=vec![0;p.hilbert.source.phases.axes.len()];let mut observable=vec![C::default();n*d];
        for state in 0..n{for a in 0..n{for b in 0..n{observable[state*d+a*n+b]=mul(basis[state+a*n],conj(basis[state+b*n]));}}}
        let mut before=FourierMatrices::from([(zero,observable.clone())]);
        for(q,k)in &p.k_hat{let mut contribution=vec![C::default();n*d];
            for state in 0..n{for i in 0..e{for j in 0..e{for a in 0..g{for b in 0..g{
                add(&mut contribution[state*d+(g+i)*n+g+j],mul(observable[state*d+a*n+b],k[(a*g+b)*e*e+i*e+j]));
            }}}}}add_dual(&mut before,q.clone(),&contribution,1.);
        }
        let mut commutator=DualCommutator::new(&p.s,n);
        let once=commutator.apply(&before,&coordinates.lapack)?;let twice=commutator.apply(&once,&coordinates.lapack)?;
        let mut complete=before;for(q,a)in once{add_dual(&mut complete,q,&a,1.);}for(q,a)in twice{add_dual(&mut complete,q,&a,0.5);}
        // Entries below 1e-15 of the largest dual entry are accumulated
        // roundoff of the S/K pull-back. Dropping them changes a population
        // by at most 1e-15*max|D|*(number of dropped entries)*|x|, far below
        // any requested tolerance, and removes ~3/4 of the readout work.
        let largest=complete.values().flat_map(|a|a.iter()).map(|z|z.re.hypot(z.im)).fold(0_f64,f64::max);
        let floor=1e-15*largest;
        let mut charges=Vec::new();let mut rows=Vec::new();
        for(q,a)in complete{
            let entries:Vec<_>=(0..n).flat_map(|s|(0..d).map(move|c|(s,c))).map(|(s,c)|(s,c,a[s*d+c]))
                .filter(|(_,_,z)|z.re.hypot(z.im)>floor).collect();
            if !entries.is_empty(){charges.push(q);rows.push(entries);}
        }
        Ok(Self{owner:Arc::clone(p),charges,rows})
    }
    pub fn coefficient_modes(&self)->usize{self.charges.len()}
    /// Output modes reached by the readout from the given domain modes.
    pub fn output_modes(&self,input_modes:&[Charge])->Result<Vec<Charge>,String>{
        let inputs=mode_set(input_modes,self.owner.hilbert.source.phases.axes.len())?;
        let support:Modes=self.charges.iter().cloned().collect();
        Ok(combinations(&support,&inputs,false)?.into_iter().collect())
    }
    pub fn frequency(&self,q:&[i32])->Result<crate::preparation::FrequencyParts,String>{self.owner.hilbert.source.phases.frequency(q)}
    pub fn prepare_phase_plan(&self,input_modes:&[Charge],times:&[f64])->Result<PopulationPhasePlan,String>{
        let inputs=mode_set(input_modes,self.owner.hilbert.source.phases.axes.len())?;
        let support:Modes=self.charges.iter().cloned().collect();let modes:Vec<_>=combinations(&support,&inputs,false)?.into_iter().collect();
        PopulationPhasePlan::prepare(Arc::clone(&self.owner),modes,times)
    }
    /// Readout coefficients of `columns` signed trial columns `[row, columns]`:
    /// coefficient[(mode*n+state)*columns+j] = sum over dual charges q and
    /// input modes i with q+i = mode of D_q[state,:] . x_i[:,j].
    /// One ZGEMM per input mode applies every dual charge at once.
    pub fn project(&self,input_modes:&[Charge],basis:&[C],columns:usize)->Result<PopulationBasis,String>{
        let n=self.owner.hilbert.source.n;let d=n*n;let phases=&self.owner.hilbert.source.phases;
        let inputs=mode_set(input_modes,phases.axes.len())?;
        if columns==0||basis.len()!=product(&[input_modes.len(),d,columns])?{
            return Err("Expected complete signed Fourier trial columns [mode*n*n,columns]".into());}
        let support:Modes=self.charges.iter().cloned().collect();let modes:Vec<_>=combinations(&support,&inputs,false)?.into_iter().collect();
        let at:BTreeMap<_,_>=modes.iter().enumerate().map(|(i,q)|(q,i)).collect();
        let block=n*columns;let mut coefficients=vec![C::default();product(&[modes.len(),block])?];
        for(i,input)in input_modes.iter().enumerate(){
            let x=&basis[i*d*columns..(i+1)*d*columns];
            if x.iter().all(|z|z.re==0.&&z.im==0.){continue;}
            for(q,entries)in self.charges.iter().zip(&self.rows){
                let target=at[&add_charge(q,input,false)?];let out=&mut coefficients[target*block..(target+1)*block];
                for &(state,c,value)in entries{
                    let source=&x[c*columns..(c+1)*columns];let output=&mut out[state*columns..(state+1)*columns];
                    for(o,&z)in output.iter_mut().zip(source){add(o,mul(value,z));}
                }
            }
        }
        if coefficients.iter().any(|z|!z.finite()){return Err("Nonfinite complete population projection".into());}
        Ok(PopulationBasis{owner:Arc::clone(&self.owner),modes,columns,coefficients})
    }
}
impl PopulationBasis{
    pub fn modes(&self)->&[Charge]{&self.modes}
    /// Row-major `[(mode*n+state), column]` complex coefficients.
    pub fn coefficients(&self)->&[C]{&self.coefficients}
    pub fn columns(&self)->usize{self.columns}
    pub fn at_times(&self,times:&[f64])->Result<PopulationRows,String>{
        self.at_phase_plan(&PopulationPhasePlan::prepare(Arc::clone(&self.owner),self.modes.clone(),times)?)
    }
    pub fn at_phase_plan(&self,plan:&PopulationPhasePlan)->Result<PopulationRows,String>{
        if !Arc::ptr_eq(&self.owner,&plan.owner)||self.modes!=plan.modes{return Err("Population phase plan belongs to another Model or output-mode union".into());}
        let times=&plan.times;
        if times.is_empty()||times.iter().any(|t|!t.is_finite()){return Err("Expected a nonempty finite actual time grid".into());}
        let n=self.owner.hilbert.source.n;let width=n*self.columns;
        let mut values=vec![C::default();product(&[times.len(),width])?];
        for it in 0..times.len(){for m in 0..self.modes.len(){let phase=plan.phases[it*self.modes.len()+m];
            for k in 0..width{add(&mut values[it*width+k],mul(phase,self.coefficients[m*width+k]));}
        }}
        if values.iter().any(|z|!z.finite()){return Err("Nonfinite current-basis population rows".into());}
        Ok(PopulationRows{owner:Arc::clone(&self.owner),times:Arc::clone(&plan.times),states:n,columns:self.columns,values:values.into()})
    }
}
impl PopulationPhasePlan{
    fn prepare(owner:Arc<PrimitivePreparation>,modes:Vec<Charge>,times:&[f64])->Result<Self,String>{
        if times.is_empty()||times.iter().any(|t|!t.is_finite()){return Err("Expected a nonempty finite actual phase grid".into());}
        let mut phases=vec![C::default();product(&[modes.len(),times.len()])?];
        for(m,q)in modes.iter().enumerate(){let frequency=owner.hilbert.source.phases.frequency(q)?;
            for(it,&time)in times.iter().enumerate(){phases[it*modes.len()+m]=frequency.phase(time)?;}}
        Ok(Self{owner,modes,times:times.to_vec().into(),phases})
    }
    pub fn times(&self)->&[f64]{&self.times}
    pub fn modes(&self)->&[Charge]{&self.modes}
    pub fn phase_evaluations(&self)->usize{self.phases.len()}
}
impl PopulationRows{
    pub fn belongs_to(&self,prepared:&OwnedPrepared)->bool{Arc::ptr_eq(&self.owner,&prepared.primitives)}
    pub fn times(&self)->&[f64]{&self.times}
    pub fn columns(&self)->usize{self.columns}
    /// Exact current-basis observation coefficients [time,state,column], not
    /// populations or a cache of a previously propagated solution.
    pub fn coefficients(&self)->&[C]{&self.values}
    /// small is row-major [column,time]; output is [time,state] COMPLEX.
    pub fn evaluate_into(&self,small:&[C],output:&mut[C])->Result<(),String>{
        let nt=self.times.len();if small.len()!=product(&[self.columns,nt])?||output.len()!=product(&[nt,self.states])?
            ||small.iter().any(|z|!z.finite()){return Err("Population small coefficients/output dimensions or finite check failed".into());}
        output.fill(C::default());for t in 0..nt{for state in 0..self.states{for j in 0..self.columns{
            add(&mut output[t*self.states+state],mul(self.values[(t*self.states+state)*self.columns+j],small[j*nt+t]));
        }}}
        if output.iter().any(|z|!z.finite()){return Err("Nonfinite complex population output".into());}Ok(())
    }
}

impl PopulationRowBlocks{
    pub fn new(first:PopulationRows)->Self{let columns=first.columns;Self{chunks:vec![first],columns}}
    pub fn columns(&self)->usize{self.columns}
    pub fn chunk_count(&self)->usize{self.chunks.len()}
    pub fn append(&mut self,rows:PopulationRows)->Result<(),String>{
        let first=&self.chunks[0];
        if !Arc::ptr_eq(&first.owner,&rows.owner)||first.states!=rows.states||first.times.len()!=rows.times.len()
            ||!first.times.iter().zip(rows.times.iter()).all(|(a,b)|a.to_bits()==b.to_bits()){
            return Err("Observation-row append requires the same current Model and exact time grid".into());}
        self.columns=self.columns.checked_add(rows.columns).ok_or("Observation column count overflow")?;self.chunks.push(rows);Ok(())
    }
    pub fn snapshot(&self)->Result<PopulationRows,String>{
        let first=&self.chunks[0];if self.chunks.len()==1{return Ok(first.clone());}
        let rows=product(&[first.times.len(),first.states])?;let mut values=vec![C::default();product(&[rows,self.columns])?];let mut offset=0;
        for chunk in &self.chunks{for i in 0..rows{values[i*self.columns+offset..i*self.columns+offset+chunk.columns]
            .copy_from_slice(&chunk.values[i*chunk.columns..(i+1)*chunk.columns]);}offset+=chunk.columns;}
        Ok(PopulationRows{owner:Arc::clone(&first.owner),times:Arc::clone(&first.times),states:first.states,columns:self.columns,values:values.into()})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn preparation_observation_batched_dual_and_fixed_phase_plan_keep_complete_oracle(){
        let owner=super::super::tests::fixture();let coordinates=owner.physical_coordinates();let n=coordinates.dimension();let d=n*n;let p=owner.primitives();let lap=&owner.preparation_lapack;
        let raw:FourierMatrices=[vec![0],vec![2]].into_iter().enumerate().map(|(m,q)|(q,(0..n*d).map(|i|C{re:((i*13+m*7)%31)as f64/37.,im:((i*7+m*11)%29)as f64/-41.}).collect())).collect();
        let oracle=|input:&FourierMatrices|{let mut out=FourierMatrices::new();for(q,values)in input{for(r,s)in &p.s{
            let st:Vec<_>=(0..d).map(|i|s[(i%n)*n+i/n]).collect();let mut result=vec![C::default();n*d];
            for state in 0..n{result[state*d..(state+1)*d].copy_from_slice(&commutator(lap,&st,&values[state*d..(state+1)*d],n).unwrap());}
            add_dual(&mut out,add_charge(q,r,false).unwrap(),&result,1.);
        }}out};
        let mut plan=DualCommutator::new(&p.s,n);let first=plan.apply(&raw,lap).unwrap();let old=oracle(&raw);let second=plan.apply(&first,lap).unwrap();let old_second=oracle(&old);
        for(actual,expected)in[(&first,&old),(&second,&old_second)]{assert_eq!(actual.len(),expected.len());for(q,a)in actual{let b=&expected[q];for(x,y)in a.iter().zip(b){assert!((x.re-y.re).hypot(x.im-y.im)<5e-13*(1.+y.re.hypot(y.im)));}}}
        let observer=coordinates.population_observation().unwrap();let inputs=[vec![1],vec![0],vec![-1]];let times=[0.,0.137,1e6+0.123456789];
        let phases=observer.prepare_phase_plan(&inputs,&times).unwrap();let fields:Vec<_>=(0..inputs.len()*d*2).map(|i|C{re:(i as f64*0.17).sin(),im:(i as f64*0.13).cos()}).collect();
        let basis=observer.project(&inputs,&fields,2).unwrap();let cached=basis.at_phase_plan(&phases).unwrap();let direct=basis.at_times(&times).unwrap();
        assert_eq!(phases.phase_evaluations(),phases.modes().len()*times.len());
        for(a,b)in cached.coefficients().iter().zip(direct.coefficients()){assert_eq!(a.re.to_bits(),b.re.to_bits());assert_eq!(a.im.to_bits(),b.im.to_bits());}
        let other=super::super::tests::fixture();let foreign=other.physical_coordinates().population_observation().unwrap().prepare_phase_plan(&inputs,&times).unwrap();
        assert!(basis.at_phase_plan(&foreign).is_err());
    }
    #[test]
    fn preparation_coordinates_initial_preserves_global_index_and_full_density(){
        let owner=super::super::tests::fixture();let coordinates=owner.physical_coordinates();let n=coordinates.dimension();
        let index=coordinates.initial(InitialView{index:3,real:None,imag:None}).unwrap();assert_eq!(index.original()[15].re,1.);
        assert!(index.diagnostics().original_coordinate_forward_error<2e-13);
        let psi=[C{re:0.5,im:0.},C{re:0.,im:0.5},C{re:-0.5,im:0.},C{re:0.,im:-0.5}];
        let density:Vec<_>=(0..n*n).map(|i|mul(psi[i/n],conj(psi[i%n]))).collect();let re:Vec<_>=density.iter().map(|z|z.re).collect();let im:Vec<_>=density.iter().map(|z|z.im).collect();
        let initial=coordinates.initial(InitialView{index:0,real:Some(&re),imag:Some(&im)}).unwrap();
        assert!(initial.diagnostics().full_density_supplied);assert!(!initial.diagnostics().original_density_modified);
        assert!(initial.diagnostics().original_coordinate_forward_error<3e-13);
        for(a,b)in initial.original().iter().zip(&density){assert_eq!(a.re.to_bits(),b.re.to_bits());assert_eq!(a.im.to_bits(),b.im.to_bits());}
        assert!(coordinates.initial(InitialView{index:0,real:Some(&re),imag:None}).is_err());
        assert!(coordinates.initial(InitialView{index:4,real:None,imag:None}).is_err());
        let mut bad=re.clone();bad[0]+=1e-7;assert!(coordinates.initial(InitialView{index:0,real:Some(&bad),imag:Some(&im)}).is_err());
    }
    #[test]
    fn preparation_coordinates_population_full_union_complex_and_gauge_restore(){
        let owner=super::super::tests::fixture();let coordinates=owner.physical_coordinates();let n=coordinates.dimension();let d=n*n;
        let observer=coordinates.population_observation().unwrap();let modes=[vec![2],vec![-1],vec![0]];let columns=2;
        let basis:Vec<_>=(0..modes.len()*d*columns).map(|i|C{re:((i*13+3)%37)as f64/43.,im:((i*17+7)%41)as f64/-47.}).collect();
        let projected=observer.project(&modes,&basis,columns).unwrap();assert!(projected.modes().len()>modes.len());
        let times=[0.,0.137,1e6+0.123456789];let rows=projected.at_times(&times).unwrap();let nt=times.len();
        let small:Vec<_>=(0..columns*nt).map(|i|C{re:0.2+i as f64*0.07,im:-0.3+i as f64*0.11}).collect();let mut population=vec![C::default();nt*n];rows.evaluate_into(&small,&mut population).unwrap();
        for(t,&time)in times.iter().enumerate(){let mut field=vec![C::default();modes.len()*d];
            for i in 0..field.len(){for j in 0..columns{add(&mut field[i],mul(basis[i*columns+j],small[j*nt+t]));}}
            let physical=coordinates.field_at(&modes,&field,time).unwrap();
            for state in 0..n{let(a,b)=(population[t*n+state],physical[state*n+state]);assert!((a.re-b.re).hypot(a.im-b.im)<3e-12*(1.+b.re.hypot(b.im)));}
        }
        assert!(population.iter().any(|z|z.im.abs()>0.01)); // no hidden real projection
        // Per-column incremental projection is identical to projecting a block.
        for j in 0..columns{let column:Vec<_>=(0..modes.len()*d).map(|i|basis[i*columns+j]).collect();
            let single=observer.project(&modes,&column,1).unwrap().at_times(&times).unwrap();
            for t in 0..nt{for state in 0..n{let(a,b)=(single.coefficients()[t*n+state],rows.coefficients()[(t*n+state)*columns+j]);assert_eq!(a.re.to_bits(),b.re.to_bits());assert_eq!(a.im.to_bits(),b.im.to_bits());}}
        }
        assert!(observer.project(&[vec![0],vec![0]],&basis,columns).is_err());
    }
}
