//! Exact radiation eigenframe for batched complete-density propagation.
//!
//! A unitary basis change diagonalizes G=sum C†C on each exact connected
//! source block. No small off-diagonal entry is discarded. In this frame
//! the no-jump exponential is elementwise, and the complete recycling map
//! is a real target² by source² matrix applied to all RHS by one DGEMM.
use crate::{complex_input,coordinates,lapack::Complex64 as Z,Coordinate,Lapack,Model,C};
use std::{collections::{BTreeMap,BTreeSet},time::Instant};

fn z(value:C)->Z{Z{re:value.re,im:value.im}}
fn c(value:Z)->C{C::new(value.re,value.im)}
fn adjoint(matrix:&[Z],n:usize)->Vec<Z>{(0..n*n).map(|i|{let value=matrix[i/n+(i%n)*n];Z{re:value.re,im:-value.im}}).collect()}
fn transpose(matrix:&[Z],n:usize)->Vec<Z>{(0..n*n).map(|i|matrix[i/n+(i%n)*n]).collect()}
fn finite(matrix:&[Z])->bool{matrix.iter().all(|value|value.re.is_finite()&&value.im.is_finite())}

/// Contiguous column-major n×n densities: i+j*n+rhs*n². These two large
/// GEMMs conjugate all RHS at once, with an explicit tensor transpose for
/// right multiplication. No loop of n² tiny GEMMs is used.
fn conjugate_batch(lap:&Lapack,left:&[Z],right:&[Z],batch:&[Z],n:usize)->Result<Vec<Z>,String>{
    let d=n*n;if batch.len()%d!=0||!finite(batch){return Err("Invalid radiation-frame batch".into());}
    let count=batch.len()/d;if count==0{return Ok(Vec::new());}
    let multiplied=lap.multiply_complex(left,batch,n,n*count,n)?;
    let mut packed=vec![Z::default();batch.len()];
    for rhs in 0..count{let offset=rhs*d;for i in 0..n{for j in 0..n{packed[offset+j+i*n]=multiplied[offset+i+j*n];}}}
    let multiplied=lap.multiply_complex(&transpose(right,n),&packed,n,n*count,n)?;
    let mut result=vec![Z::default();batch.len()];
    for rhs in 0..count{let offset=rhs*d;for i in 0..n{for j in 0..n{result[offset+i+j*n]=multiplied[offset+j+i*n];}}}
    Ok(result)
}

pub struct Frame {
    n:usize,
    vectors:Vec<Z>,
    adjoint:Vec<Z>,
    gamma:Vec<f64>,
    source_coordinates:Vec<Coordinate>,
    target_coordinates:Vec<Coordinate>,
    source_rates:Vec<f64>,
    recycling:Vec<f64>,
}

pub struct FrameStep {
    n:usize,
    decay:Vec<f64>,
    gain:Vec<f64>,
    source_coordinates:Vec<Coordinate>,
    target_coordinates:Vec<Coordinate>,
    trace_error:f64,
}

#[derive(Default)]
pub struct ApplyProfile {
    pub decay_ms:f64,
    pub gather_ms:f64,
    pub gain_gemm_ms:f64,
    pub scatter_ms:f64,
    pub total_ms:f64,
}

/// Reused real source/target batches for one fixed density batch shape.
/// Half, full, and signed steps from the same Frame share these dimensions.
pub struct BatchWorkspace {
    n:usize,
    source_rows:usize,
    target_rows:usize,
    count:usize,
    source:Vec<f64>,
    target:Vec<f64>,
}
impl BatchWorkspace {
    pub fn new(n:usize,source_rows:usize,target_rows:usize,count:usize)->Result<Self,String>{
        if n==0||n>64{return Err("Invalid radiation workspace Hilbert dimension".into());}
        let source_len=source_rows.checked_mul(count).ok_or("Radiation source workspace overflow")?;
        let target_len=target_rows.checked_mul(count).ok_or("Radiation target workspace overflow")?;
        n.checked_mul(n).and_then(|d|d.checked_mul(count)).ok_or("Radiation complex workspace overflow")?;
        Ok(Self{n,source_rows,target_rows,count,source:vec![0.;source_len],target:vec![0.;target_len]})
    }
}

fn mapped(coordinate:Coordinate,indices:&[usize])->Coordinate{match coordinate{
    Coordinate::Population(i)=>Coordinate::Population(indices[i]),
    Coordinate::Real(i,j)=>Coordinate::Real(indices[i],indices[j]),
    Coordinate::Imag(i,j)=>Coordinate::Imag(indices[i],indices[j]),
}}
fn matrix_entry(coordinate:Coordinate)->(usize,usize){match coordinate{
    Coordinate::Population(i)=>(i,i),Coordinate::Real(i,j)|Coordinate::Imag(i,j)=>(i,j),
}}

impl Frame {
    pub fn new(lap:&Lapack,model:&Model)->Result<Self,String>{
        let n=model.dimension;if n==0||n>64{return Err("Invalid radiation-frame Hilbert dimension".into());}
        let mut sources=BTreeSet::new();let mut targets=BTreeSet::new();let mut operators=Vec::new();
        for operator in &model.collapse{
            let matrix=complex_input(&operator.real,&operator.imag,n*n,"Radiation-frame collapse")?;
            for(i,value)in matrix.iter().enumerate(){if value.re!=0.||value.im!=0.{targets.insert(i/n);sources.insert(i%n);}}
            operators.push(matrix);
        }
        if !sources.is_disjoint(&targets){return Err("Radiation eigenframe requires disjoint collapse source/target sectors".into());}
        let sources:Vec<_>=sources.into_iter().collect();let targets:Vec<_>=targets.into_iter().collect();
        let m=sources.len();let k=targets.len();let mut gram=vec![C::default();m*m];
        for operator in &operators{for(a,&i)in sources.iter().enumerate(){for(b,&j)in sources.iter().enumerate(){for &g in &targets{gram[a+b*m]+=operator[g*n+i].conj()*operator[g*n+j];}}}}
        // Build components using exact represented nonzeros, including tiny
        // entries. This preserves exact source blocks and leaves targets fixed.
        let mut parent:Vec<_>=(0..m).collect();
        fn root(parent:&mut[usize],mut i:usize)->usize{while parent[i]!=i{parent[i]=parent[parent[i]];i=parent[i];}i}
        for a in 0..m{for b in 0..a{let value=gram[a+b*m];if value.re!=0.||value.im!=0.{let x=root(&mut parent,a);let y=root(&mut parent,b);parent[x]=y;}}}
        let mut blocks=BTreeMap::<usize,Vec<usize>>::new();for i in 0..m{let key=root(&mut parent,i);blocks.entry(key).or_default().push(i);}
        let mut vectors=vec![Z::default();n*n];for i in 0..n{vectors[i+i*n].re=1.;}
        let mut gamma=vec![0.;n];let gram_norm=gram.iter().map(|value|value.abs()).fold(0.,f64::max);
        for block in blocks.values(){
            let length=block.len();
            if length==1{gamma[sources[block[0]]]=gram[block[0]+block[0]*m].re;continue;}
            let matrix:Vec<_>=(0..length*length).map(|i|z(gram[block[i%length]+block[i/length]*m])).collect();
            let (values,q)=lap.hermitian_eigen(&matrix,length)?;
            for (j,&index) in block.iter().enumerate(){let column=sources[index];gamma[column]=values[j];for(i,&row)in block.iter().enumerate(){vectors[sources[row]+column*n]=q[i+j*length];}}
        }
        if gamma.iter().any(|value|!value.is_finite()||*value< -128.*f64::EPSILON*n as f64*gram_norm.max(1.)){
            return Err("Unresolved radiation Gram eigensystem".into());
        }
        // C'=C V, because V acts only on source states and is identity on all
        // target states. Keep every original collapse channel independently.
        let transformed:Vec<Vec<C>>=operators.iter().map(|operator|{
            let mut result=vec![C::default();k*m];
            for(g,&target)in targets.iter().enumerate(){for(a,&column)in sources.iter().enumerate(){for &source in &sources{
                result[g*m+a]+=operator[target*n+source]*c(vectors[source+column*n]);
            }}}
            result
        }).collect();
        let source_local=coordinates(m);let target_local=coordinates(k);let ns=m*m;let nt=k*k;
        let mut recycling=vec![0.;nt*ns];let mut source_rates=Vec::with_capacity(ns);
        for (j,&input) in source_local.iter().enumerate(){
            let(a,b)=matrix_entry(input);source_rates.push(0.5*(gamma[sources[a]]+gamma[sources[b]]));
            for (i,&output) in target_local.iter().enumerate(){
                let(g,h)=matrix_entry(output);let mut value=C::default();
                for operator in &transformed{
                    let first=operator[g*m+a]*operator[h*m+b].conj();
                    value+=match input{
                        Coordinate::Population(_)=>first,
                        Coordinate::Real(_,_)=>first+operator[g*m+b]*operator[h*m+a].conj(),
                        Coordinate::Imag(_,_)=>{let difference=first-operator[g*m+b]*operator[h*m+a].conj();C::new(-difference.im,difference.re)},
                    };
                }
                recycling[i+j*nt]=if matches!(output,Coordinate::Imag(_, _)){value.im}else{value.re};
            }
        }
        if recycling.iter().any(|value|!value.is_finite()){return Err("Non-finite transformed radiation recycling".into());}
        let adjoint=adjoint(&vectors,n);
        Ok(Self{n,vectors,adjoint,gamma,
            source_coordinates:source_local.into_iter().map(|coord|mapped(coord,&sources)).collect(),
            target_coordinates:target_local.into_iter().map(|coord|mapped(coord,&targets)).collect(),
            source_rates,recycling})
    }
    pub fn dimension(&self)->usize{self.n*self.n}
    pub fn gamma(&self)->&[f64]{&self.gamma}
    pub fn into_frame_batch(&self,lap:&Lapack,batch:&[Z])->Result<Vec<Z>,String>{conjugate_batch(lap,&self.adjoint,&self.vectors,batch,self.n)}
    pub fn out_of_frame_batch(&self,lap:&Lapack,batch:&[Z])->Result<Vec<Z>,String>{conjugate_batch(lap,&self.vectors,&self.adjoint,batch,self.n)}
    pub fn rotate_unitary(&self,lap:&Lapack,unitary:&[Z])->Result<Vec<Z>,String>{
        if unitary.len()!=self.n*self.n||!finite(unitary){return Err("Invalid radiation-frame unitary".into());}
        lap.multiply_complex(&lap.multiply_complex(&self.adjoint,unitary,self.n,self.n,self.n)?,&self.vectors,self.n,self.n,self.n)
    }
    /// Signed finite durations are for explicit high-order composition only.
    /// Negative-time steps are exact inverses of the dissipative exponential,
    /// but they are NOT CPTP; no such claim is made or positivity repair used.
    pub fn step(&self,time:f64)->Result<FrameStep,String>{
        if !time.is_finite(){return Err("Invalid radiation-frame duration".into());}
        let n=self.n;let decay:Vec<_>=(0..n*n).map(|i|(-0.5*(self.gamma[i%n]+self.gamma[i/n])*time).exp()).collect();
        let nt=self.target_coordinates.len();let ns=self.source_coordinates.len();let mut gain=self.recycling.clone();
        for (j,&rate) in self.source_rates.iter().enumerate(){
            let integral=if rate==0.{time}else{-(-rate*time).exp_m1()/rate};
            for value in &mut gain[j*nt..(j+1)*nt]{*value*=integral;}
        }
        if decay.iter().chain(&gain).any(|value|!value.is_finite()){return Err("Non-finite radiation-frame exponential".into());}
        let mut trace_error=0_f64;
        for j in 0..ns{
            let recycled:f64=self.target_coordinates.iter().enumerate().filter_map(|(i,coordinate)|matches!(coordinate,Coordinate::Population(_)).then_some(gain[i+j*nt])).sum();
            let expected=match self.source_coordinates[j]{Coordinate::Population(i)=>1.-decay[i+i*n],_=>0.};
            trace_error=trace_error.max((recycled-expected).abs());
        }
        let scale=decay.iter().map(|value|value.abs()).fold(1.,f64::max);
        if trace_error>2e-11*scale{return Err(format!("Radiation eigenframe trace identity unresolved: {trace_error:e}"));}
        Ok(FrameStep{n,decay,gain,source_coordinates:self.source_coordinates.clone(),target_coordinates:self.target_coordinates.clone(),trace_error})
    }
}

impl FrameStep {
    pub fn trace_error(&self)->f64{self.trace_error}
    pub fn workspace(&self,count:usize)->Result<BatchWorkspace,String>{
        BatchWorkspace::new(self.n,self.source_coordinates.len(),self.target_coordinates.len(),count)
    }
    /// Trusted inner-loop application. The caller validates numeric inputs
    /// before its prepared propagation and validates the completed output.
    /// Shape checks remain O(1); no allocation or finite-value scan occurs.
    pub fn apply_batch_into(&self,lap:&Lapack,input:&[Z],output:&mut[Z],workspace:&mut BatchWorkspace)->Result<ApplyProfile,String>{
        self.apply_batch_layout_into(lap,input,output,workspace,false)
    }
    /// The transposed layout stores rho^T rather than rho. Diagonal decay is
    /// unchanged; raw Hermitian imaginary coordinates change sign at gather
    /// and scatter. This is a storage transformation, not D^T or D's adjoint.
    pub fn apply_batch_layout_into(&self,lap:&Lapack,input:&[Z],output:&mut[Z],workspace:&mut BatchWorkspace,transposed:bool)->Result<ApplyProfile,String>{
        let started=Instant::now();let mut profile=ApplyProfile::default();
        let n=self.n;let d=n*n;let ns=self.source_coordinates.len();let nt=self.target_coordinates.len();
        if input.len()!=output.len()||input.len()%d!=0||workspace.n!=n||workspace.source_rows!=ns||workspace.target_rows!=nt||workspace.count!=input.len()/d{
            return Err("Trusted radiation batch/workspace dimension mismatch".into());
        }
        let count=workspace.count;let decay_started=Instant::now();
        for (result,density) in output.chunks_exact_mut(d).zip(input.chunks_exact(d)){
            for ((value,&original),&decay) in result.iter_mut().zip(density).zip(&self.decay){value.re=original.re*decay;value.im=original.im*decay;}
        }
        profile.decay_ms=decay_started.elapsed().as_secs_f64()*1000.;
        if count==0||ns==0||nt==0{profile.total_ms=started.elapsed().as_secs_f64()*1000.;return Ok(profile);}
        let gather_started=Instant::now();
        for rhs in 0..count{let offset=rhs*d;for (i,&coordinate) in self.source_coordinates.iter().enumerate(){workspace.source[i+rhs*ns]=match coordinate{
            Coordinate::Population(i)=>input[offset+i+i*n].re,
            Coordinate::Real(i,j)=>input[offset+i+j*n].re,
            Coordinate::Imag(i,j)=>if transposed{-input[offset+i+j*n].im}else{input[offset+i+j*n].im},
        };}}
        profile.gather_ms=gather_started.elapsed().as_secs_f64()*1000.;
        let gain_started=Instant::now();
        lap.multiply_into(&self.gain,&workspace.source,&mut workspace.target,nt,count,ns)?;
        profile.gain_gemm_ms=gain_started.elapsed().as_secs_f64()*1000.;
        let scatter_started=Instant::now();
        for rhs in 0..count{let offset=rhs*d;for(i,&coordinate)in self.target_coordinates.iter().enumerate(){let value=workspace.target[i+rhs*nt];match coordinate{
            Coordinate::Population(i)=>output[offset+i+i*n].re+=value,
            Coordinate::Real(i,j)=>{output[offset+i+j*n].re+=value;output[offset+j+i*n].re+=value;},
            Coordinate::Imag(i,j)=>{let value=if transposed{-value}else{value};output[offset+i+j*n].im+=value;output[offset+j+i*n].im-=value;},
        }}}
        profile.scatter_ms=scatter_started.elapsed().as_secs_f64()*1000.;
        profile.total_ms=started.elapsed().as_secs_f64()*1000.;Ok(profile)
    }
    pub fn apply_batch(&self,lap:&Lapack,batch:&[Z])->Result<Vec<Z>,String>{
        self.apply_batch_profiled(lap,batch).map(|(result,_)|result)
    }
    pub fn apply_batch_profiled(&self,lap:&Lapack,batch:&[Z])->Result<(Vec<Z>,ApplyProfile),String>{
        let started=Instant::now();let mut profile=ApplyProfile::default();
        let n=self.n;let d=n*n;if batch.len()%d!=0||!finite(batch){return Err("Invalid diagonal radiation batch".into());}
        let count=batch.len()/d;let ns=self.source_coordinates.len();let nt=self.target_coordinates.len();
        let decay_started=Instant::now();
        let mut result=batch.to_vec();for density in result.chunks_exact_mut(d){for (value,&decay) in density.iter_mut().zip(&self.decay){value.re*=decay;value.im*=decay;}}
        profile.decay_ms=decay_started.elapsed().as_secs_f64()*1000.;
        if count==0||ns==0||nt==0{profile.total_ms=started.elapsed().as_secs_f64()*1000.;return Ok((result,profile));}
        let gather_started=Instant::now();
        let mut input=vec![0.;ns*count];
        for rhs in 0..count{let offset=rhs*d;for (i,&coordinate) in self.source_coordinates.iter().enumerate(){input[i+rhs*ns]=match coordinate{
            Coordinate::Population(i)=>batch[offset+i+i*n].re,
            Coordinate::Real(i,j)=>batch[offset+i+j*n].re,
            Coordinate::Imag(i,j)=>batch[offset+i+j*n].im,
        };}}
        profile.gather_ms=gather_started.elapsed().as_secs_f64()*1000.;
        let gain_started=Instant::now();
        let gain=lap.multiply(&self.gain,&input,nt,count,ns);
        profile.gain_gemm_ms=gain_started.elapsed().as_secs_f64()*1000.;
        let scatter_started=Instant::now();
        for rhs in 0..count{let offset=rhs*d;for(i,&coordinate)in self.target_coordinates.iter().enumerate(){let value=gain[i+rhs*nt];match coordinate{
            Coordinate::Population(i)=>result[offset+i+i*n].re+=value,
            Coordinate::Real(i,j)=>{result[offset+i+j*n].re+=value;result[offset+j+i*n].re+=value;},
            Coordinate::Imag(i,j)=>{result[offset+i+j*n].im+=value;result[offset+j+i*n].im-=value;},
        }}}
        profile.scatter_ms=scatter_started.elapsed().as_secs_f64()*1000.;
        if !finite(&result){return Err("Non-finite diagonal radiation batch result".into());}
        profile.total_ms=started.elapsed().as_secs_f64()*1000.;Ok((result,profile))
    }
}

#[cfg(test)]
mod tests{
    use super::*;use crate::{build_generator,expand_density,Operator};
    fn basis_batch(n:usize)->Vec<Z>{
        let coords=coordinates(n);let d=n*n;let mut batch=Vec::with_capacity(d*d);
        for j in 0..d{let mut unit=vec![0.;d];unit[j]=1.;let rho=expand_density(&coords,&unit,n);batch.extend((0..d).map(|i|z(rho[i/n+(i%n)*n])));}batch
    }
    fn model()->Model{
        Model{ frequency_coordinates:None,dimension:3,states:vec![],h_real:vec![0.;9],h_imag:vec![0.;9],collapse:vec![
            Operator{real:vec![0.,1.,0.3,0.,0.,0.,0.,0.,0.],imag:vec![0.,0.,0.2,0.,0.,0.,0.,0.,0.]},
            Operator{real:vec![0.,0.,0.7,0.,0.,0.,0.,0.,0.],imag:vec![0.;9]}],
            initial_index:1,initial_real:None,initial_imag:None,time_dependent:false,drives:vec![]}
    }
    #[test]
    fn diagonal_frame_radiation_matches_complete_dense_lindblad_on_every_basis_column(){
        let lap=Lapack::load().unwrap();let model=model();let n=model.dimension;let d=n*n;let g=build_generator(&model).unwrap();
        let mut dense=vec![0.;d*d];for(j,col)in g.matrix.iter().enumerate(){for &(i,value)in col{dense[i+j*d]+=value;}}
        let frame=Frame::new(&lap,&model).unwrap();let batch=basis_batch(n);let transformed=frame.into_frame_batch(&lap,&batch).unwrap();
        for time in [0.4,-0.15]{
            let step=frame.step(time).unwrap();assert!(step.trace_error()<1e-12);
            let output=frame.out_of_frame_batch(&lap,&step.apply_batch(&lap,&transformed).unwrap()).unwrap();
            for j in 0..d{let mut unit=vec![0.;d];unit[j]=1.;let expected=crate::dense_expm::action(&lap,&dense,&unit,&[time]).unwrap();
                let rho=expand_density(&g.coords,&expected[0],n);let error=(0..d).map(|i|(c(output[j*d+i])-rho[i/n+(i%n)*n]).abs()).fold(0.,f64::max);
                assert!(error<3e-12,"column={j}, time={time}, error={error:e}");
            }
        }
    }
    #[test]
    fn tiny_offdiagonal_gram_is_rotated_not_dropped_and_batch_roundtrip_is_exact(){
        let lap=Lapack::load().unwrap();let mut model=model();model.collapse[0].real[2]=1e-11;model.collapse[0].imag[2]=2e-11;
        let frame=Frame::new(&lap,&model).unwrap();
        assert!(frame.vectors[1+2*3].re!=0.||frame.vectors[1+2*3].im!=0.);
        let batch=basis_batch(3);let rotated=frame.into_frame_batch(&lap,&batch).unwrap();let restored=frame.out_of_frame_batch(&lap,&rotated).unwrap();
        let error=batch.iter().zip(restored).map(|(&a,b)|(c(a)-c(b)).abs()).fold(0.,f64::max);assert!(error<2e-14);
        let probe=frame.step(0.2).unwrap().apply_batch(&lap,&rotated).unwrap();let output=frame.out_of_frame_batch(&lap,&probe).unwrap();
        // Population in source1 feeds an off-diagonal source coherence through
        // the complete G exponential. An entry-wise diagonal approximation
        // would make this exactly zero.
        assert!(c(output[1*9+1+2*3]).abs()>1e-13);
    }
    #[test]
    fn reusable_batch_workspace_matches_allocating_oracle_for_positive_and_signed_steps(){
        let lap=Lapack::load().unwrap();let frame=Frame::new(&lap,&model()).unwrap();
        let input=frame.into_frame_batch(&lap,&basis_batch(3)).unwrap();
        let mut workspace=frame.step(0.1).unwrap().workspace(9).unwrap();let mut output=vec![Z::default();input.len()];
        let source_pointer=workspace.source.as_ptr();let target_pointer=workspace.target.as_ptr();
        for time in [0.1,-0.17,0.3]{
            let step=frame.step(time).unwrap();let expected=step.apply_batch(&lap,&input).unwrap();
            step.apply_batch_into(&lap,&input,&mut output,&mut workspace).unwrap();
            let error=expected.iter().zip(&output).map(|(&a,&b)|(c(a)-c(b)).abs()).fold(0.,f64::max);assert!(error<1e-14);
            assert_eq!(source_pointer,workspace.source.as_ptr());assert_eq!(target_pointer,workspace.target.as_ptr());
        }
        assert!(frame.step(0.1).unwrap().apply_batch_into(&lap,&input[..9],&mut output,&mut workspace).is_err());
    }
    #[test]
    fn transposed_storage_radiation_matches_full_complex_recycling(){
        let lap=Lapack::load().unwrap();let frame=Frame::new(&lap,&model()).unwrap();let n=3;let d=n*n;
        let input=frame.into_frame_batch(&lap,&basis_batch(n)).unwrap();
        let transposed:Vec<_>=input.chunks_exact(d).flat_map(|rho|transpose(rho,n)).collect();
        let mut workspace=frame.step(0.1).unwrap().workspace(d).unwrap();let mut output=vec![Z::default();input.len()];
        for time in [0.1,-0.17,0.3]{
            let step=frame.step(time).unwrap();let expected=step.apply_batch(&lap,&input).unwrap();
            step.apply_batch_layout_into(&lap,&transposed,&mut output,&mut workspace,true).unwrap();
            let restored:Vec<_>=output.chunks_exact(d).flat_map(|rho|transpose(rho,n)).collect();
            let error=expected.iter().zip(restored).map(|(&a,b)|(c(a)-c(b)).abs()).fold(0.,f64::max);
            assert!(error<1e-14,"Transposed storage changed radiation by {error:e}");
        }
    }
}
