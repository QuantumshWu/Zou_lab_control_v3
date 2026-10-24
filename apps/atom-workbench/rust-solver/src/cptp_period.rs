//! Structure-preserving full-density period action.
//!
//! Each fixed step is exact spontaneous emission for h/2, exact unitary
//! evolution under H(t+h/2) for h, then exact spontaneous emission for h/2.
//! Strang splitting is a second-order NUMERICAL discretization; no state,
//! Hamiltonian entry, decay channel, or off-diagonal C^H C entry is removed.
//! The outer solver must refine the grid at the requested time horizon.
//! The OPTIONAL symmetric fourth-order constructor includes negative-time
//! dissipative factors. It is NOT a CPTP map at finite grid spacing; callers
//! must inspect is_cptp() and perform independent physical-result validation.
use crate::{add_superterm, build_generator, complex_input, coordinate_map, coordinates, expand_density,
    lapack::Complex64 as Z, Coordinate, Lapack, Model, C};
use std::{cell::RefCell,collections::BTreeSet,time::Instant};

type Sparse = Vec<Vec<(usize,f64)>>;
struct HamiltonianDrive { omega:f64, omega_low:f64, cos:Vec<C>, sin:Vec<C> }
struct Hamiltonian { n:usize, base:Vec<C>, drives:Vec<HamiltonianDrive> }
struct Bath {
    n:usize,
    sources:Vec<usize>,
    targets:Vec<usize>,
    gamma:Vec<f64>,
    vectors:Vec<Z>,
    // Exact nonzero-connected components of G on the source space.
    // Their size depends on the supplied operators, not state labels/cutoffs.
    components:Vec<Vec<usize>>,
    // Each complete collapse operator, restricted to target x source,
    // transformed only on the source side into the eigenbasis of G.
    transformed:Vec<Vec<C>>,
}
struct NoJumpBlock { indices:Vec<usize>, matrix:Vec<C> }
struct Dissipation { blocks:Vec<NoJumpBlock>, gain:Sparse }
struct Unitary { matrix:Vec<Z>, adjoint:Vec<Z>, conjugate:Vec<Z> }
struct ComplexWorkspace { left:Vec<Z>,packed:Vec<Z>,right:Vec<Z> }
impl ComplexWorkspace{
    fn new(length:usize)->Self{Self{left:vec![Z::default();length],packed:vec![Z::default();length],right:vec![Z::default();length]}}
}
struct FourthOrder {
    middle_dissipation:Dissipation,
    full_boundary_dissipation:Dissipation,
}
struct BathFrameAction {
    frame:crate::bath_frame::Frame,
    unitaries:Vec<Unitary>,
    half:crate::bath_frame::FrameStep,
    interior:crate::bath_frame::FrameStep,
    middle:Option<crate::bath_frame::FrameStep>,
}
struct SingleWorkspace {
    state:Vec<Z>,
    complex:ComplexWorkspace,
    radiation:crate::bath_frame::BatchWorkspace,
}
#[derive(Clone,Debug,Default,serde::Serialize)]
#[serde(rename_all="camelCase")]
pub struct BuildProfile {
    pub initialization_ms:f64,
    pub frame_initialization_ms:f64,
    pub frame_transform_ms:f64,
    pub frame_decay_ms:f64,
    pub frame_gather_ms:f64,
    pub frame_gain_gemm_ms:f64,
    pub frame_scatter_ms:f64,
    pub preparation_ms:f64,
    pub workspace_preparation_ms:f64,
    pub radiation_ms:f64,
    pub first_unitary_gemm_ms:f64,
    pub packing_ms:f64,
    pub second_unitary_gemm_ms:f64,
    pub coordinate_conversion_ms:f64,
    pub total_ms:f64,
    pub columns:usize,
    pub physical_dimension:usize,
    pub macro_steps:usize,
    pub unitary_steps:usize,
    pub radiation_applications:usize,
    pub bath_frame:bool,
    pub workspace:bool,
    pub alternating_layout:bool,
    pub bath_frame_coordinates:bool,
}
pub struct Action {
    n:usize,
    coords:Vec<Coordinate>,
    period:f64,
    steps:usize,
    step:f64,
    hamiltonian:Hamiltonian,
    bath:Bath,
    half_dissipation:Dissipation,
    unitaries:Vec<Unitary>,
    fourth:Option<FourthOrder>,
    initialization_ms:f64,
    build_profile:RefCell<Option<BuildProfile>>,
    bath_frame:Option<BathFrameAction>,
    frame_initialization_ms:f64,
    single_workspace:RefCell<Option<SingleWorkspace>>,
}
fn z(value:C)->Z { Z{re:value.re,im:value.im} }
fn c(value:Z)->C { C::new(value.re,value.im) }
fn adjoint(a:&[Z],n:usize)->Vec<Z> {
    (0..n*n).map(|i|{let value=a[i/n+(i%n)*n];Z{re:value.re,im:-value.im}}).collect()
}
fn coordinates_from_matrix(coords:&[Coordinate],rho:&[Z],n:usize)->Vec<f64> {
    coords.iter().map(|&coordinate|match coordinate {
        Coordinate::Population(i)=>rho[i+i*n].re,
        Coordinate::Real(i,j)=>rho[i+j*n].re,
        Coordinate::Imag(i,j)=>rho[i+j*n].im,
    }).collect()
}
fn density_column(coords:&[Coordinate],x:&[f64],n:usize)->Vec<Z> {
    let row=expand_density(coords,x,n);
    (0..n*n).map(|k|z(row[(k%n)*n+k/n])).collect()
}
fn compress(matrix:&mut Sparse) {
    for col in matrix {
        col.sort_by_key(|&(i,_)|i);let mut result:Vec<(usize,f64)>=Vec::new();
        for &(i,a)in col.iter(){if let Some(last)=result.last_mut().filter(|last|last.0==i){last.1+=a;}else{result.push((i,a));}}
        result.retain(|&(_,a)|a!=0.);*col=result;
    }
}
fn hermitian_input(real:&[f64],imag:&[f64],n:usize,name:&str)->Result<Vec<C>,String>{
    let result=complex_input(real,imag,n*n,name)?;
    for i in 0..n{for j in 0..n{
        if(result[i*n+j]-result[j*n+i].conj()).abs()>1e-10*(1.+result[i*n+j].abs()){
            return Err(format!("{name} is not Hermitian"));
        }
    }}Ok(result)
}
impl Hamiltonian {
    fn new(model:&Model)->Result<Self,String>{
        let n=model.dimension;let base=hermitian_input(&model.h_real,&model.h_imag,n,"Period Hamiltonian")?;
        let frequency_parts=crate::harmonics::model_drive_frequency_parts(model)?;
        let drives=model.drives.iter().zip(frequency_parts).map(|(drive,(omega,omega_low))|{
            if !drive.omega.is_finite()||drive.omega<=0.{return Err("Invalid period-action drive frequency".into());}
            Ok(HamiltonianDrive{omega,omega_low,
                cos:hermitian_input(&drive.cos.real,&drive.cos.imag,n,"Period cosine drive")?,
                sin:hermitian_input(&drive.sin.real,&drive.sin.imag,n,"Period sine drive")?})
        }).collect::<Result<Vec<_>,String>>()?;
        Ok(Self{n,base,drives})
    }
    fn unitary(&self,lap:&Lapack,time:f64,duration:f64)->Result<Unitary,String>{
        let n=self.n;let mut h=self.base.clone();
        for drive in &self.drives{
            let high=drive.omega*time;let low=drive.omega_low.mul_add(time,drive.omega.mul_add(time,-high));
            let(s,c)=high.sin_cos();let(ds,dc)=low.sin_cos();let cosine=c*dc-s*ds;let sine=s*dc+c*ds;
            for(i,value)in h.iter_mut().enumerate(){*value+=drive.cos[i]*cosine+drive.sin[i]*sine;}
        }
        let column:Vec<_>=(0..n*n).map(|i|z(h[(i%n)*n+i/n])).collect();
        let(energies,vectors)=lap.hermitian_eigen(&column,n)?;let vh=adjoint(&vectors,n);
        let mut phased=vectors.clone();
        for j in 0..n{
            let angle=-energies[j]*duration;let tail=(-energies[j]).mul_add(duration,-angle);
            let(s,c)=angle.sin_cos();let(ds,dc)=tail.sin_cos();let phase=C::new(c*dc-s*ds,s*dc+c*ds);
            for i in 0..n{phased[i+j*n]=z(cplx(vectors[i+j*n])*phase);}
        }
        let matrix=lap.multiply_complex(&phased,&vh,n,n,n)?;
        let conjugate=matrix.iter().map(|value|Z{re:value.re,im:-value.im}).collect();
        Ok(Unitary{adjoint:adjoint(&matrix,n),matrix,conjugate})
    }
}
// A separate spelling avoids shadowing the converter by local cosine values.
fn cplx(value:Z)->C { c(value) }

impl Bath {
    fn new(lap:&Lapack,model:&Model)->Result<Self,String>{
        let n=model.dimension;let mut sources=BTreeSet::new();let mut targets=BTreeSet::new();let mut operators=Vec::new();
        for operator in &model.collapse{
            let values=complex_input(&operator.real,&operator.imag,n*n,"Period collapse")?;
            for(i,value)in values.iter().enumerate(){if value.re!=0.||value.im!=0.{targets.insert(i/n);sources.insert(i%n);}}
            operators.push(values);
        }
        if !sources.is_disjoint(&targets){return Err("Exact radiation step requires disjoint collapse source/target Hilbert sectors".into());}
        let sources:Vec<_>=sources.into_iter().collect();let targets:Vec<_>=targets.into_iter().collect();let m=sources.len();
        if m==0{return Ok(Self{n,sources,targets,gamma:Vec::new(),vectors:Vec::new(),components:Vec::new(),transformed:Vec::new()});}
        let mut gram=vec![C::default();m*m];
        for operator in &operators{for(i,&a)in sources.iter().enumerate(){for(j,&b)in sources.iter().enumerate(){
            for &g in &targets{gram[i+j*m]+=operator[g*n+a].conj()*operator[g*n+b];}
        }}}
        let mut components=Vec::new();let mut seen=vec![false;m];
        for start in 0..m{if seen[start]{continue;}let mut component=Vec::new();let mut queue=vec![start];seen[start]=true;
            while let Some(i)=queue.pop(){component.push(i);for j in 0..m{
                let a=gram[i+j*m];let b=gram[j+i*m];
                if !seen[j]&&(a.re!=0.||a.im!=0.||b.re!=0.||b.im!=0.){seen[j]=true;queue.push(j);}
            }}component.sort_unstable();components.push(component);
        }
        let mut gamma=vec![0.;m];let mut vectors=vec![Z::default();m*m];
        for component in &components{let k=component.len();
            if k==1{let i=component[0];gamma[i]=gram[i+i*m].re;vectors[i+i*m].re=1.;continue;}
            let matrix:Vec<_>=(0..k*k).map(|i|z(gram[component[i%k]+component[i/k]*m])).collect();
            let(values,basis)=lap.hermitian_eigen(&matrix,k)?;
            for(j,&b)in component.iter().enumerate(){gamma[b]=values[j];for(i,&a)in component.iter().enumerate(){vectors[a+b*m]=basis[i+j*k];}}
        }
        let norm=gram.iter().map(|x|x.abs()).fold(0.,f64::max);
        if gamma.iter().any(|&g|g< -64.*f64::EPSILON*m as f64*norm.max(1.)){
            return Err("Radiation Gram eigensystem has an unresolved negative eigenvalue".into());
        }
        // Tiny signed eigensolver roundoff is retained, not clipped. The
        // channel remains a sum of Kraus-integral terms; trace is checked below.
        let transformed=operators.iter().map(|operator|{
            let mut result=vec![C::default();targets.len()*m];
            for(g,&target)in targets.iter().enumerate(){for a in 0..m{for(i,&source)in sources.iter().enumerate(){
                result[g*m+a]+=operator[target*n+source]*c(vectors[i+a*m]);
            }}}
            result
        }).collect();
        Ok(Self{n,sources,targets,gamma,vectors,components,transformed})
    }
    fn step(&self,lap:&Lapack,time:f64)->Result<Dissipation,String>{
        if !time.is_finite()||time<0.{return Err("Invalid exact radiation duration".into());}
        self.signed_step(lap,time)
    }
    /// Exact exponential of the supplied dissipator for a signed numerical
    /// substep. Negative time is an inverse linear map, NOT a quantum channel.
    /// This private entry point is used only by the labelled S4 experiment.
    fn signed_step(&self,lap:&Lapack,time:f64)->Result<Dissipation,String>{
        if !time.is_finite(){return Err("Invalid signed radiation duration".into());}
        let n=self.n;let m=self.sources.len();let d=n*n;let coords=coordinates(n);let map=coordinate_map(&coords,n);
        let mut no_jump=vec![Z::default();d];for i in 0..n{no_jump[i+i*n].re=1.;}
        let mut gain=vec![Vec::new();d];
        if m==0||time==0.{return Ok(Dissipation{blocks:Vec::new(),gain});}
        let vh=adjoint(&self.vectors,m);let mut scaled=self.vectors.clone();
        for j in 0..m{let decay=(-0.5*self.gamma[j]*time).exp();for i in 0..m{scaled[i+j*m].re*=decay;scaled[i+j*m].im*=decay;}}
        let e=lap.multiply_complex(&scaled,&vh,m,m,m)?;
        for(i,&a)in self.sources.iter().enumerate(){for(j,&b)in self.sources.iter().enumerate(){no_jump[a+b*n]=e[i+j*m];}}
        let blocks=self.components.iter().map(|component|{
            let matrix=component.iter().flat_map(|&i|component.iter().map({let e=&e;move|&j|c(e[i+j*m])})).collect();
            NoJumpBlock{indices:component.iter().map(|&i|self.sources[i]).collect(),matrix}
        }).collect();
        let mut integrals=vec![0.;m*m];
        for a in 0..m{for b in 0..m{
            let rate=0.5*(self.gamma[a]+self.gamma[b]);
            integrals[a+b*m]=if rate==0.{time}else{-(-rate*time).exp_m1()/rate};
        }}
        let conjugate:Vec<_>=self.vectors.iter().map(|value|Z{re:value.re,im:-value.im}).collect();
        let transpose:Vec<_>=(0..m*m).map(|i|self.vectors[i/m+(i%m)*m]).collect();
        let mut imaginary=vec![0.;n*d];
        for(g,&ground)in self.targets.iter().enumerate(){for(h,&other)in self.targets.iter().enumerate(){
            if ground>other{continue;}
            let mut kernel=vec![Z::default();m*m];
            for a in 0..m{for b in 0..m{
                let mut value=C::default();for operator in &self.transformed{value+=operator[g*m+a]*operator[h*m+b].conj();}
                kernel[a+b*m]=z(value*integrals[a+b*m]);
            }}
            // rho_eigen=U^H rho U. The ORIGINAL rho_ij coefficient is
            // (conj(U) K U^T)_ij, not U K U^H.
            let original=lap.multiply_complex(&lap.multiply_complex(&conjugate,&kernel,m,m,m)?,&transpose,m,m,m)?;
            for(i,&a)in self.sources.iter().enumerate(){for(j,&b)in self.sources.iter().enumerate(){
                add_superterm(&mut gain,&mut imaginary,&map,ground*n+other,a*n+b,c(original[i+j*m]),n);
            }}
        }}
        if imaginary.iter().any(|value|value.abs()>1e-10){return Err("Exact radiation gain failed Hermiticity invariance".into());}
        compress(&mut gain);let eh=adjoint(&no_jump,n);
        let metric=lap.multiply_complex(&eh,&no_jump,n,n,n)?;
        let mut trace_coefficients=coords.iter().map(|&coordinate|match coordinate{
            Coordinate::Population(i)=>metric[i+i*n].re,
            Coordinate::Real(i,j)=>2.*metric[i+j*n].re,
            Coordinate::Imag(i,j)=>2.*metric[i+j*n].im,
        }).collect::<Vec<_>>();
        for(j,column)in gain.iter().enumerate(){for &(i,value)in column{if i<n{trace_coefficients[j]+=value;}}}
        let trace_error=trace_coefficients.iter().enumerate().map(|(i,&v)|(v-if i<n{1.}else{0.}).abs()).fold(0.,f64::max);
        if trace_error>2e-11{return Err(format!("Exact radiation trace identity unresolved: {trace_error:e}"));}
        Ok(Dissipation{blocks,gain})
    }
}
impl Dissipation {
    fn apply(&self,_lap:&Lapack,coords:&[Coordinate],initial:&[f64],n:usize)->Result<Vec<f64>,String>{
        let rho=expand_density(coords,initial,n);let mut left=rho.clone();
        // E is identity off the source sector and block diagonal on exact G
        // components. All entries inside every block, including tiny ones,
        // participate in both products; nothing is thresholded.
        for block in &self.blocks{let k=block.indices.len();for(a,&row)in block.indices.iter().enumerate(){for col in 0..n{
            let mut value=C::default();for(b,&input)in block.indices.iter().enumerate(){value+=block.matrix[a*k+b]*rho[input*n+col];}
            left[row*n+col]=value;
        }}}
        let mut output=left.clone();
        for block in &self.blocks{let k=block.indices.len();for row in 0..n{for(b,&col)in block.indices.iter().enumerate(){
            let mut value=C::default();for(a,&input)in block.indices.iter().enumerate(){value+=left[row*n+input]*block.matrix[b*k+a].conj();}
            output[row*n+col]=value;
        }}}
        let mut result=coords.iter().map(|&coordinate|match coordinate{
            Coordinate::Population(i)=>output[i*n+i].re,
            Coordinate::Real(i,j)=>output[i*n+j].re,
            Coordinate::Imag(i,j)=>output[i*n+j].im,
        }).collect::<Vec<_>>();
        for(j,column)in self.gain.iter().enumerate(){for &(i,value)in column{result[i]+=value*initial[j];}}
        Ok(result)
    }
    fn apply_batch(&self,initial:&[Z],coords:&[Coordinate],n:usize,count:usize)->Result<Vec<Z>,String>{
        let d=n*n;if initial.len()!=d*count{return Err("Invalid radiation batch dimensions".into());}
        let mut left=initial.to_vec();
        for column in 0..count{let offset=column*d;
            for block in &self.blocks{let k=block.indices.len();for(a,&row)in block.indices.iter().enumerate(){for col in 0..n{
                let mut value=C::default();for(b,&input)in block.indices.iter().enumerate(){value+=block.matrix[a*k+b]*c(initial[offset+input+col*n]);}
                left[offset+row+col*n]=z(value);
            }}}
        }
        let mut output=left.clone();
        for column in 0..count{let offset=column*d;
            for block in &self.blocks{let k=block.indices.len();for row in 0..n{for(b,&col)in block.indices.iter().enumerate(){
                let mut value=C::default();for(a,&input)in block.indices.iter().enumerate(){value+=c(left[offset+row+input*n])*block.matrix[b*k+a].conj();}
                output[offset+row+col*n]=z(value);
            }}}
            for(j,entries)in self.gain.iter().enumerate(){if entries.is_empty(){continue;}
                let input=match coords[j]{Coordinate::Population(i)=>initial[offset+i+i*n].re,
                    Coordinate::Real(i,k)=>initial[offset+i+k*n].re,Coordinate::Imag(i,k)=>initial[offset+i+k*n].im};
                for &(row,coefficient)in entries{let value=coefficient*input;match coords[row]{
                    Coordinate::Population(i)=>output[offset+i+i*n].re+=value,
                    Coordinate::Real(i,j)=>{output[offset+i+j*n].re+=value;output[offset+j+i*n].re+=value;},
                    Coordinate::Imag(i,j)=>{output[offset+i+j*n].im+=value;output[offset+j+i*n].im-=value;},
                }}
            }
        }
        if output.iter().any(|value|!value.re.is_finite()||!value.im.is_finite()){return Err("Non-finite radiation batch result".into());}
        Ok(output)
    }
}
impl Unitary {
    fn apply(&self,lap:&Lapack,coords:&[Coordinate],initial:&[f64],n:usize)->Result<Vec<f64>,String>{
        let rho=density_column(coords,initial,n);
        let evolved=lap.multiply_complex(&lap.multiply_complex(&self.matrix,&rho,n,n,n)?,&self.adjoint,n,n,n)?;
        Ok(coordinates_from_matrix(coords,&evolved,n))
    }
    fn apply_batch(&self,lap:&Lapack,initial:&[Z],n:usize,count:usize,profile:&mut BuildProfile)->Result<Vec<Z>,String>{
        let d=n*n;if initial.len()!=d*count{return Err("Invalid unitary batch dimensions".into());}
        // Concatenated column-major density matrices are an n x (n*count)
        // matrix: one large GEMM applies U to every left index.
        let phase=Instant::now();let left=lap.multiply_complex(&self.matrix,initial,n,n*count,n)?;
        profile.first_unitary_gemm_ms+=phase.elapsed().as_secs_f64()*1000.;
        // Move the right density index into the GEMM column dimension. Each
        // (left index,basis column) pair becomes one row, avoiding count tiny
        // GEMMs without forming the n^2 x n^2 unitary superoperator.
        let phase=Instant::now();let rows=n*count;let mut packed=vec![Z::default();d*count];
        for column in 0..count{for j in 0..n{for i in 0..n{
            packed[i+column*n+j*rows]=left[i+j*n+column*d];
        }}}
        profile.packing_ms+=phase.elapsed().as_secs_f64()*1000.;
        let phase=Instant::now();let right=lap.multiply_complex(&packed,&self.adjoint,rows,n,n)?;
        profile.second_unitary_gemm_ms+=phase.elapsed().as_secs_f64()*1000.;
        let phase=Instant::now();let mut result=vec![Z::default();d*count];
        for column in 0..count{for j in 0..n{for i in 0..n{
            result[i+j*n+column*d]=right[i+column*n+j*rows];
        }}}
        profile.packing_ms+=phase.elapsed().as_secs_f64()*1000.;profile.unitary_steps+=1;
        Ok(result)
    }
    fn apply_batch_into(&self,lap:&Lapack,state:&mut[Z],n:usize,count:usize,workspace:&mut ComplexWorkspace,profile:&mut BuildProfile)->Result<(),String>{
        let d=n*n;let length=d*count;
        if state.len()!=length||workspace.left.len()<length||workspace.packed.len()<length||workspace.right.len()<length{
            return Err("Invalid reusable unitary-batch dimensions".into());
        }
        // Inputs/operators were validated at preparation/batch entry. These
        // products use beta=0 to overwrite reused destinations. Transposing
        // each small density locally makes right multiplication another wide
        // left GEMM: (U rho U^H)^T = conj(U) (U rho)^T.
        let phase=Instant::now();
        lap.multiply_complex_into(&self.matrix,state,&mut workspace.left[..length],n,n*count,n)?;
        profile.first_unitary_gemm_ms+=phase.elapsed().as_secs_f64()*1000.;
        let phase=Instant::now();
        for column in 0..count{let offset=column*d;for i in 0..n{for j in 0..n{
            workspace.packed[offset+j+i*n]=workspace.left[offset+i+j*n];
        }}}
        profile.packing_ms+=phase.elapsed().as_secs_f64()*1000.;let phase=Instant::now();
        lap.multiply_complex_into(&self.conjugate,&workspace.packed[..length],&mut workspace.right[..length],n,n*count,n)?;
        profile.second_unitary_gemm_ms+=phase.elapsed().as_secs_f64()*1000.;let phase=Instant::now();
        for column in 0..count{let offset=column*d;for i in 0..n{for j in 0..n{
            state[offset+i+j*n]=workspace.right[offset+j+i*n];
        }}}
        profile.packing_ms+=phase.elapsed().as_secs_f64()*1000.;profile.unitary_steps+=1;Ok(())
    }
    /// Keep the second GEMM's transposed output and alternate representation
    /// on the next step. Normal input: conj(U)*(U*rho)^T = rho_new^T.
    /// Transposed input: U*(conj(U)*rho^T)^T = rho_new. Thus only ONE local
    /// transpose is needed per step, with the exact same two GEMMs.
    fn apply_batch_alternating_into(&self,lap:&Lapack,state:&mut Vec<Z>,n:usize,count:usize,transposed:&mut bool,
        workspace:&mut ComplexWorkspace,profile:&mut BuildProfile)->Result<(),String>{
        let d=n*n;let length=d.checked_mul(count).ok_or("Alternating unitary batch overflow")?;
        if state.len()!=length||workspace.left.len()!=length||workspace.packed.len()!=length||workspace.right.len()!=length{
            return Err("Invalid alternating unitary-batch dimensions".into());
        }
        let(first,second)=if *transposed{(&self.conjugate,&self.matrix)}else{(&self.matrix,&self.conjugate)};
        let phase=Instant::now();
        lap.multiply_complex_into(first,state,&mut workspace.left,n,n*count,n)?;
        profile.first_unitary_gemm_ms+=phase.elapsed().as_secs_f64()*1000.;let phase=Instant::now();
        for column in 0..count{let offset=column*d;for i in 0..n{for j in 0..n{
            workspace.packed[offset+j+i*n]=workspace.left[offset+i+j*n];
        }}}
        profile.packing_ms+=phase.elapsed().as_secs_f64()*1000.;let phase=Instant::now();
        lap.multiply_complex_into(second,&workspace.packed,&mut workspace.right,n,n*count,n)?;
        profile.second_unitary_gemm_ms+=phase.elapsed().as_secs_f64()*1000.;
        std::mem::swap(state,&mut workspace.right);*transposed=!*transposed;profile.unitary_steps+=1;Ok(())
    }
}
fn record_frame_timing(profile:&mut BuildProfile,timing:crate::bath_frame::ApplyProfile){
    profile.radiation_ms+=timing.total_ms;profile.frame_decay_ms+=timing.decay_ms;
    profile.frame_gather_ms+=timing.gather_ms;profile.frame_gain_gemm_ms+=timing.gain_gemm_ms;
    profile.frame_scatter_ms+=timing.scatter_ms;profile.radiation_applications+=1;
}
fn apply_frame_radiation(lap:&Lapack,step:&crate::bath_frame::FrameStep,batch:&[Z],profile:&mut BuildProfile)->Result<Vec<Z>,String>{
    let(result,timing)=step.apply_batch_profiled(lap,batch)?;
    record_frame_timing(profile,timing);
    Ok(result)
}
fn frame_radiation_into(lap:&Lapack,step:&crate::bath_frame::FrameStep,state:&mut Vec<Z>,complex:&mut ComplexWorkspace,
    radiation:&mut crate::bath_frame::BatchWorkspace,profile:&mut BuildProfile)->Result<(),String>{
    let timing=step.apply_batch_into(lap,state,&mut complex.left,radiation)?;
    record_frame_timing(profile,timing);std::mem::swap(state,&mut complex.left);Ok(())
}
impl BathFrameAction{
    fn apply_prefix_into(&self,lap:&Lapack,state:&mut Vec<Z>,n:usize,count:usize,unitary_count:usize,
        complex:&mut ComplexWorkspace,radiation:&mut crate::bath_frame::BatchWorkspace,profile:&mut BuildProfile)->Result<(),String>{
        if unitary_count>self.unitaries.len(){return Err("Invalid prepared period-stage count".into());}
        if unitary_count==0{return Ok(());}
        frame_radiation_into(lap,&self.half,state,complex,radiation,profile)?;
        for(i,unitary)in self.unitaries.iter().take(unitary_count).enumerate(){
            unitary.apply_batch_into(lap,state,n,count,complex,profile)?;
            let dissipation=if i+1==unitary_count{&self.half}
                else if let Some(middle)=&self.middle{if i%3==2{&self.interior}else{middle}}
                else{&self.interior};
            frame_radiation_into(lap,dissipation,state,complex,radiation,profile)?;
        }
        Ok(())
    }
    fn apply_prefix_alternating_into(&self,lap:&Lapack,state:&mut Vec<Z>,n:usize,count:usize,unitary_count:usize,
        complex:&mut ComplexWorkspace,radiation:&mut crate::bath_frame::BatchWorkspace,profile:&mut BuildProfile)->Result<(),String>{
        if unitary_count>self.unitaries.len(){return Err("Invalid alternating period-stage count".into());}
        if unitary_count==0{return Ok(());}
        frame_radiation_into(lap,&self.half,state,complex,radiation,profile)?;
        let mut transposed=false;
        for(i,unitary)in self.unitaries.iter().take(unitary_count).enumerate(){
            unitary.apply_batch_alternating_into(lap,state,n,count,&mut transposed,complex,profile)?;
            let dissipation=if i+1==unitary_count{&self.half}
                else if let Some(middle)=&self.middle{if i%3==2{&self.interior}else{middle}}
                else{&self.interior};
            let timing=dissipation.apply_batch_layout_into(lap,state,&mut complex.left,radiation,transposed)?;
            record_frame_timing(profile,timing);std::mem::swap(state,&mut complex.left);
        }
        // Normalize once at the public boundary, never after every unitary.
        if transposed{let phase=Instant::now();let d=n*n;
            for column in 0..count{let offset=column*d;for i in 0..n{for j in i+1..n{state.swap(offset+i+j*n,offset+j+i*n);}}}
            profile.packing_ms+=phase.elapsed().as_secs_f64()*1000.;
        }
        Ok(())
    }
}
impl Action {
    pub fn new(lap:&Lapack,model:&Model,period:f64,steps:usize)->Result<Self,String>{
        Self::prepare(lap,model,period,steps,false)
    }
    /// Yoshida's symmetric triple composition of the FULL second-order step.
    /// The negative dissipative substeps mean finite-grid CPTP is NOT claimed.
    pub fn new_symmetric_fourth_order(lap:&Lapack,model:&Model,period:f64,steps:usize)->Result<Self,String>{
        Self::prepare(lap,model,period,steps,true)
    }
    fn prepare(lap:&Lapack,model:&Model,period:f64,steps:usize,fourth_order:bool)->Result<Self,String>{
        let started=Instant::now();
        if !period.is_finite()||period<=0.||steps==0||steps>100_000{return Err("CPTP period action requires a positive finite period and1..100000steps".into());}
        build_generator(model)?;let n=model.dimension;let step=period/steps as f64;
        if step==0.{return Err("CPTP period step underflow".into());}
        let hamiltonian=Hamiltonian::new(model)?;let bath=Bath::new(lap,model)?;
        let(half_dissipation,unitaries,fourth)=if fourth_order{
            let w1=1./(2.-2_f64.cbrt());let w0=1.-2.*w1;
            let half=bath.step(lap,0.5*w1*step)?;
            let middle=bath.signed_step(lap,0.5*(1.-w1)*step)?;
            let full=bath.step(lap,w1*step)?;
            let mut unitaries=Vec::with_capacity(3*steps);
            for i in 0..steps{let start=i as f64*step;
                for(center,duration)in[(0.5*w1,w1),(0.5,w0),(1.-0.5*w1,w1)]{
                    unitaries.push(hamiltonian.unitary(lap,start+center*step,duration*step)?);
                }
            }
            (half,unitaries,Some(FourthOrder{middle_dissipation:middle,full_boundary_dissipation:full}))
        }else{
            let half=bath.step(lap,0.5*step)?;
            let unitaries=(0..steps).map(|i|hamiltonian.unitary(lap,(i as f64+0.5)*step,step)).collect::<Result<Vec<_>,_>>()?;
            (half,unitaries,None)
        };
        let mut result=Self{n,coords:coordinates(n),period,steps,step,hamiltonian,bath,half_dissipation,unitaries,fourth,
            initialization_ms:0.,build_profile:RefCell::new(None),bath_frame:None,frame_initialization_ms:0.,single_workspace:RefCell::new(None)};
        if std::env::var("ATOM_PERIOD_BATH_FRAME").is_ok_and(|value|value=="1"){result.prepare_bath_frame(lap,model)?;}
        result.initialization_ms=started.elapsed().as_secs_f64()*1000.;Ok(result)
    }
    fn prepare_bath_frame(&mut self,lap:&Lapack,model:&Model)->Result<(),String>{
        let started=Instant::now();let frame=crate::bath_frame::Frame::new(lap,model)?;
        let unitaries=self.unitaries.iter().map(|unitary|{
            let matrix=frame.rotate_unitary(lap,&unitary.matrix)?;
            let conjugate=matrix.iter().map(|value|Z{re:value.re,im:-value.im}).collect();
            Ok(Unitary{adjoint:adjoint(&matrix,self.n),matrix,conjugate})
        }).collect::<Result<Vec<_>,String>>()?;
        let(half,interior,middle)=if self.fourth.is_some(){let w1=1./(2.-2_f64.cbrt());
            (frame.step(0.5*w1*self.step)?,frame.step(w1*self.step)?,Some(frame.step(0.5*(1.-w1)*self.step)?))
        }else{(frame.step(0.5*self.step)?,frame.step(self.step)?,None)};
        self.bath_frame=Some(BathFrameAction{frame,unitaries,half,interior,middle});
        *self.single_workspace.borrow_mut()=None;
        self.frame_initialization_ms=started.elapsed().as_secs_f64()*1000.;Ok(())
    }
    pub fn dimension(&self)->usize{self.n*self.n}
    pub fn is_cptp(&self)->bool{self.fourth.is_none()}
    pub fn order(&self)->usize{if self.fourth.is_some(){4}else{2}}
    pub fn name(&self)->&'static str{if self.fourth.is_some(){"Symmetric fourth-order full-density splitting (not finite-grid CPTP)"}
        else{"Strang CPTP full-density splitting"}}
    pub fn full_matrix_profile(&self)->Option<BuildProfile>{self.build_profile.borrow().clone()}
    /// Complete period map, column-major d x d in ORIGINAL real Hermitian
    /// coordinates. All basis columns evolve under the same fixed linear
    /// map. No Krylov projection, state selection, or HS weighting occurs.
    pub fn full_matrix(&self,lap:&Lapack)->Result<Vec<f64>,String>{
        self.full_matrix_impl(lap,std::env::var("ATOM_PERIOD_WORKSPACE").is_ok_and(|value|value=="1"))
    }
    fn full_matrix_impl(&self,lap:&Lapack,reuse_workspace:bool)->Result<Vec<f64>,String>{
        self.full_matrix_options(lap,reuse_workspace,false,std::env::var("ATOM_PERIOD_ALTERNATING_LAYOUT").is_ok_and(|value|value=="1"))
    }
    pub fn has_bath_frame(&self)->bool{self.bath_frame.is_some()}
    /// Same complete map in the bath eigenbasis. Build its identity columns
    /// directly in that basis instead of conjugating all n^2 input/output
    /// basis matrices. The caller must transform its state at the boundary.
    pub fn full_matrix_in_bath_frame(&self,lap:&Lapack)->Result<Vec<f64>,String>{
        self.full_matrix_options(lap,std::env::var("ATOM_PERIOD_WORKSPACE").is_ok_and(|value|value=="1"),true,
            std::env::var("ATOM_PERIOD_ALTERNATING_LAYOUT").is_ok_and(|value|value=="1"))
    }
    pub fn state_into_bath_frame(&self,lap:&Lapack,state:&[f64])->Result<Vec<f64>,String>{
        self.validate(state)?;let frame=self.bath_frame.as_ref().ok_or("Radiation eigenbasis is not prepared")?;
        let transformed=frame.frame.into_frame_batch(lap,&density_column(&self.coords,state,self.n))?;
        Ok(coordinates_from_matrix(&self.coords,&transformed,self.n))
    }
    fn full_matrix_options(&self,lap:&Lapack,reuse_workspace:bool,frame_coordinates:bool,alternating:bool)->Result<Vec<f64>,String>{
        if frame_coordinates&&self.bath_frame.is_none(){return Err("Cannot construct a bath-coordinate period map without a radiation eigenbasis".into());}
        *self.build_profile.borrow_mut()=None;let started=Instant::now();let phase=Instant::now();
        let n=self.n;let d=self.dimension();let mut batch=vec![Z::default();d*d];
        let mut profile=BuildProfile{initialization_ms:self.initialization_ms,frame_initialization_ms:self.frame_initialization_ms,
            columns:d,physical_dimension:n,macro_steps:self.steps,bath_frame:self.bath_frame.is_some(),
            workspace:reuse_workspace&&self.bath_frame.is_some(),alternating_layout:alternating&&reuse_workspace&&self.bath_frame.is_some(),
            bath_frame_coordinates:frame_coordinates,..BuildProfile::default()};
        for(column,&coordinate)in self.coords.iter().enumerate(){let offset=column*d;match coordinate{
            Coordinate::Population(i)=>batch[offset+i+i*n].re=1.,
            Coordinate::Real(i,j)=>{batch[offset+i+j*n].re=1.;batch[offset+j+i*n].re=1.;},
            Coordinate::Imag(i,j)=>{batch[offset+i+j*n].im=1.;batch[offset+j+i*n].im=-1.;},
        }}
        // Consecutive exact dissipative halves compose to exp(D*h). Fuse only
        // those adjacent factors; Hamiltonian and emission order is unchanged.
        let full_dissipation=if self.fourth.is_none()&&self.bath_frame.is_none(){Some(self.bath.step(lap,self.step)?)}else{None};
        profile.preparation_ms=phase.elapsed().as_secs_f64()*1000.;
        if let Some(frame)=&self.bath_frame{
            if !frame_coordinates{let phase=Instant::now();batch=frame.frame.into_frame_batch(lap,&batch)?;
                profile.frame_transform_ms+=phase.elapsed().as_secs_f64()*1000.;}
            if reuse_workspace{
                let phase=Instant::now();let mut workspace=ComplexWorkspace::new(batch.len());
                let mut radiation_workspace=frame.half.workspace(d)?;
                profile.workspace_preparation_ms=phase.elapsed().as_secs_f64()*1000.;
                if alternating{frame.apply_prefix_alternating_into(lap,&mut batch,n,d,frame.unitaries.len(),&mut workspace,&mut radiation_workspace,&mut profile)?;}
                else{frame.apply_prefix_into(lap,&mut batch,n,d,frame.unitaries.len(),&mut workspace,&mut radiation_workspace,&mut profile)?;}
            }else{
                batch=apply_frame_radiation(lap,&frame.half,&batch,&mut profile)?;
                for(i,unitary)in frame.unitaries.iter().enumerate(){
                    batch=unitary.apply_batch(lap,&batch,n,d,&mut profile)?;
                    let dissipation=if i+1==frame.unitaries.len(){&frame.half}
                        else if let Some(middle)=&frame.middle{if i%3==2{&frame.interior}else{middle}}
                        else{&frame.interior};
                    batch=apply_frame_radiation(lap,dissipation,&batch,&mut profile)?;
                }
            }
            if batch.iter().any(|value|!value.re.is_finite()||!value.im.is_finite()){
                return Err("Non-finite complete period map at trusted batch boundary".into());
            }
            if !frame_coordinates{let phase=Instant::now();batch=frame.frame.out_of_frame_batch(lap,&batch)?;
                profile.frame_transform_ms+=phase.elapsed().as_secs_f64()*1000.;}
        }else{
            let phase=Instant::now();
            batch=self.half_dissipation.apply_batch(&batch,&self.coords,n,d)?;
            profile.radiation_ms+=phase.elapsed().as_secs_f64()*1000.;profile.radiation_applications+=1;
            for(i,unitary)in self.unitaries.iter().enumerate(){
                batch=unitary.apply_batch(lap,&batch,n,d,&mut profile)?;
                let phase=Instant::now();
                batch=if i+1==self.unitaries.len(){self.half_dissipation.apply_batch(&batch,&self.coords,n,d)?}
                    else if let Some(fourth)=&self.fourth{
                        let dissipation=if i%3==2{&fourth.full_boundary_dissipation}else{&fourth.middle_dissipation};
                        dissipation.apply_batch(&batch,&self.coords,n,d)?
                    }else{full_dissipation.as_ref().unwrap().apply_batch(&batch,&self.coords,n,d)?};
                profile.radiation_ms+=phase.elapsed().as_secs_f64()*1000.;profile.radiation_applications+=1;
            }
        }
        let phase=Instant::now();let mut result=Vec::with_capacity(d*d);
        for column in 0..d{result.extend(coordinates_from_matrix(&self.coords,&batch[column*d..(column+1)*d],n));}
        if result.iter().any(|value|!value.is_finite()){return Err("Non-finite original-coordinate complete period map".into());}
        profile.coordinate_conversion_ms=phase.elapsed().as_secs_f64()*1000.;profile.total_ms=started.elapsed().as_secs_f64()*1000.;
        if std::env::var_os("ATOM_PERIOD_PROFILE").is_some(){eprintln!("Period matrix profile {}",serde_json::to_string(&profile).unwrap_or_default());}
        *self.build_profile.borrow_mut()=Some(profile);
        Ok(result)
    }
    fn validate(&self,x:&[f64])->Result<(),String>{
        if x.len()!=self.dimension()||x.iter().any(|v|!v.is_finite()){return Err("Invalid Hermitian-coordinate CPTP action vector".into());}Ok(())
    }
    fn full_step(&self,lap:&Lapack,x:&[f64],i:usize)->Result<Vec<f64>,String>{
        let before=self.half_dissipation.apply(lap,&self.coords,x,self.n)?;
        let coherent=if let Some(fourth)=&self.fourth{
            let first=self.unitaries[3*i].apply(lap,&self.coords,&before,self.n)?;
            let first=fourth.middle_dissipation.apply(lap,&self.coords,&first,self.n)?;
            let second=self.unitaries[3*i+1].apply(lap,&self.coords,&first,self.n)?;
            let second=fourth.middle_dissipation.apply(lap,&self.coords,&second,self.n)?;
            self.unitaries[3*i+2].apply(lap,&self.coords,&second,self.n)?
        }else{self.unitaries[i].apply(lap,&self.coords,&before,self.n)?};
        self.half_dissipation.apply(lap,&self.coords,&coherent,self.n)
    }
    pub fn apply(&self,lap:&Lapack,x:&[f64])->Result<Vec<f64>,String>{
        self.validate(x)?;
        if self.bath_frame.is_some()&&std::env::var("ATOM_PERIOD_WORKSPACE").is_ok_and(|value|value=="1"){
            return self.apply_frame_workspace(lap,x,self.steps,0.);
        }
        let mut result=x.to_vec();
        for i in 0..self.steps{result=self.full_step(lap,&result,i)?;}
        if result.iter().any(|v|!v.is_finite()){return Err("Non-finite CPTP period result".into());}Ok(result)
    }
    /// Phase is a duration from zero, in microseconds. Reuse all complete
    /// prepared steps, constructing only the final fractional step.
    pub fn partial(&self,lap:&Lapack,x:&[f64],phase:f64)->Result<Vec<f64>,String>{
        self.validate(x)?;if !phase.is_finite()||phase<0.||phase>self.period{return Err("CPTP partial phase is outside[0,period]".into());}
        if phase==0.{return Ok(x.to_vec());}if phase==self.period{return self.apply(lap,x);}
        let full=(phase/self.step).floor()as usize;let full=full.min(self.steps);
        let remaining=phase-full as f64*self.step;
        if self.bath_frame.is_some()&&std::env::var("ATOM_PERIOD_WORKSPACE").is_ok_and(|value|value=="1"){
            return self.apply_frame_workspace(lap,x,full,remaining.max(0.));
        }
        let mut result=x.to_vec();
        for i in 0..full{result=self.full_step(lap,&result,i)?;}
        if remaining>0.{
            let start=full as f64*self.step;
            if self.fourth.is_some(){
                let w1=1./(2.-2_f64.cbrt());let w0=1.-2.*w1;
                let half=self.bath.step(lap,0.5*w1*remaining)?;
                let middle=self.bath.signed_step(lap,0.5*(1.-w1)*remaining)?;
                result=half.apply(lap,&self.coords,&result,self.n)?;
                for(stage,(center,duration))in[(0.5*w1,w1),(0.5,w0),(1.-0.5*w1,w1)].into_iter().enumerate(){
                    let unitary=self.hamiltonian.unitary(lap,start+center*remaining,duration*remaining)?;
                    result=unitary.apply(lap,&self.coords,&result,self.n)?;
                    if stage<2{result=middle.apply(lap,&self.coords,&result,self.n)?;}
                }
                result=half.apply(lap,&self.coords,&result,self.n)?;
            }else{
                let dissipation=self.bath.step(lap,0.5*remaining)?;
                let unitary=self.hamiltonian.unitary(lap,start+0.5*remaining,remaining)?;
                result=dissipation.apply(lap,&self.coords,&result,self.n)?;
                result=unitary.apply(lap,&self.coords,&result,self.n)?;
                result=dissipation.apply(lap,&self.coords,&result,self.n)?;
            }
        }
        if result.iter().any(|v|!v.is_finite()){return Err("Non-finite CPTP partial result".into());}Ok(result)
    }
    /// Evaluate different exact phase durations from different NEW starting
    /// states. Complete Strang steps are shared across an active RHS prefix;
    /// phases are never rounded or binned. The caller opts into this path.
    pub fn partial_many(&self,lap:&Lapack,initial_states:&[Vec<f64>],phases:&[f64])->Result<Vec<Vec<f64>>,String>{
        self.partial_many_impl(lap,initial_states,phases,false)
    }
    /// Period-boundary states are already in the bath eigenbasis; return all
    /// sampled states in the ORIGINAL physical basis after phase evolution.
    pub fn partial_many_from_bath_frame(&self,lap:&Lapack,initial_states:&[Vec<f64>],phases:&[f64])->Result<Vec<Vec<f64>>,String>{
        self.partial_many_impl(lap,initial_states,phases,true)
    }
    fn partial_many_impl(&self,lap:&Lapack,initial_states:&[Vec<f64>],phases:&[f64],input_in_bath_frame:bool)->Result<Vec<Vec<f64>>,String>{
        if initial_states.len()!=phases.len(){return Err("Partial-period state/phase count mismatch".into());}
        for(state,&phase)in initial_states.iter().zip(phases){self.validate(state)?;
            if !phase.is_finite()||phase<0.||phase>self.period{return Err("Partial-period duration is outside[0,period]".into());}}
        if initial_states.is_empty(){return Ok(Vec::new());}
        // The signed high-order experiment retains its existing single-state
        // oracle. This batching path is currently only the positive S2 map.
        let Some(frame)=self.bath_frame.as_ref().filter(|_|self.fourth.is_none())else{
            if input_in_bath_frame{
                let frame=self.bath_frame.as_ref().ok_or("Partial inputs require the prepared radiation eigenbasis")?;
                let batch:Vec<_>=initial_states.iter().flat_map(|state|density_column(&self.coords,state,self.n)).collect();
                let original=frame.frame.out_of_frame_batch(lap,&batch)?;
                let states:Vec<_>=original.chunks_exact(self.dimension()).map(|state|coordinates_from_matrix(&self.coords,state,self.n)).collect();
                return self.partial_many_impl(lap,&states,phases,false);
            }
            return initial_states.iter().zip(phases).map(|(state,&phase)|self.partial(lap,state,phase)).collect();
        };
        struct Case{original:usize,whole:usize,tail:f64}
        let mut cases=phases.iter().enumerate().map(|(original,&phase)|{
            if phase==self.period{return Case{original,whole:self.steps,tail:0.};}
            let mut whole=((phase/self.step).floor()as usize).min(self.steps);
            // Correct only the integer decomposition at floating boundaries;
            // the requested phase itself is unchanged, including its tail.
            while whole>0&&whole as f64*self.step>phase{whole-=1;}
            while whole<self.steps&&(whole+1)as f64*self.step<=phase{whole+=1;}
            Case{original,whole,tail:phase-whole as f64*self.step}
        }).collect::<Vec<_>>();
        cases.sort_by(|a,b|b.whole.cmp(&a.whole).then(a.original.cmp(&b.original)));
        let n=self.n;let d=self.dimension();let count=cases.len();let length=d.checked_mul(count).ok_or("Partial batch dimension overflow")?;
        let mut batch=vec![Z::default();length];
        for(column,case)in cases.iter().enumerate(){let offset=column*d;
            for(&coordinate,&value)in self.coords.iter().zip(&initial_states[case.original]){match coordinate{
                Coordinate::Population(i)=>batch[offset+i+i*n].re=value,
                Coordinate::Real(i,j)=>{batch[offset+i+j*n].re=value;batch[offset+j+i*n].re=value;},
                Coordinate::Imag(i,j)=>{batch[offset+i+j*n].im=value;batch[offset+j+i*n].im=-value;},
            }}
        }
        if !input_in_bath_frame{batch=frame.frame.into_frame_batch(lap,&batch)?;}
        let mut completed=vec![Z::default();length];let mut workspace=ComplexWorkspace::new(length);
        let mut profile=BuildProfile::default();let mut active=cases.iter().take_while(|case|case.whole>0).count();
        for(column,case)in cases.iter().enumerate().skip(active){
            completed[case.original*d..(case.original+1)*d].copy_from_slice(&batch[column*d..(column+1)*d]);
        }
        if active>0{
            let mut radiation=frame.half.workspace(active)?;
            frame.half.apply_batch_into(lap,&batch[..active*d],&mut workspace.left[..active*d],&mut radiation)?;
            std::mem::swap(&mut batch,&mut workspace.left);
        }
        for step in 0..cases[0].whole{
            if active==0{break;}
            frame.unitaries[step].apply_batch_into(lap,&mut batch[..active*d],n,active,&mut workspace,&mut profile)?;
            let continuing=cases[..active].iter().take_while(|case|case.whole>step+1).count();
            let finished=active-continuing;
            if finished>0{
                let mut radiation=frame.half.workspace(finished)?;
                frame.half.apply_batch_into(lap,&batch[continuing*d..active*d],&mut workspace.left[..finished*d],&mut radiation)?;
                for local in 0..finished{let case=&cases[continuing+local];
                    completed[case.original*d..(case.original+1)*d].copy_from_slice(&workspace.left[local*d..(local+1)*d]);
                }
            }
            if continuing>0{
                let mut radiation=frame.interior.workspace(continuing)?;
                frame.interior.apply_batch_into(lap,&batch[..continuing*d],&mut workspace.left[..continuing*d],&mut radiation)?;
                std::mem::swap(&mut batch,&mut workspace.left);
            }
            active=continuing;
        }
        let mut tail_state=vec![Z::default();d];let mut tail_workspace=ComplexWorkspace::new(d);
        let mut tail_radiation=frame.half.workspace(1)?;
        for case in &cases{if case.tail>0.{
            tail_state.copy_from_slice(&completed[case.original*d..(case.original+1)*d]);
            self.advance_frame_fraction_into(lap,frame,&mut tail_state,&mut tail_workspace,&mut tail_radiation,
                case.whole as f64*self.step,case.tail,&mut profile)?;
            completed[case.original*d..(case.original+1)*d].copy_from_slice(&tail_state);
        }}
        if completed.iter().any(|value|!value.re.is_finite()||!value.im.is_finite()){return Err("Non-finite batched partial-period state".into());}
        let original=frame.frame.out_of_frame_batch(lap,&completed)?;
        Ok((0..count).map(|i|if phases[i]==0.&&!input_in_bath_frame{initial_states[i].clone()}
            else{coordinates_from_matrix(&self.coords,&original[i*d..(i+1)*d],n)}).collect())
    }
    fn advance_frame_fraction_into(&self,lap:&Lapack,frame:&BathFrameAction,state:&mut Vec<Z>,complex:&mut ComplexWorkspace,
        radiation:&mut crate::bath_frame::BatchWorkspace,start:f64,fraction:f64,profile:&mut BuildProfile)->Result<(),String>{
        let(w1,w0)=if self.fourth.is_some(){let w1=1./(2.-2_f64.cbrt());(w1,1.-2.*w1)}else{(1.,0.)};
        let half=frame.frame.step(0.5*w1*fraction)?;
        let middle=if self.fourth.is_some(){Some(frame.frame.step(0.5*(1.-w1)*fraction)?)}else{None};
        frame_radiation_into(lap,&half,state,complex,radiation,profile)?;
        let times:Vec<_>=if self.fourth.is_some(){vec![(0.5*w1,w1),(0.5,w0),(1.-0.5*w1,w1)]}else{vec![(0.5,1.)]};
        for(i,&(center,duration))in times.iter().enumerate(){
            let original=self.hamiltonian.unitary(lap,start+center*fraction,duration*fraction)?;
            let matrix=frame.frame.rotate_unitary(lap,&original.matrix)?;
            let conjugate=matrix.iter().map(|value|Z{re:value.re,im:-value.im}).collect();
            let unitary=Unitary{adjoint:adjoint(&matrix,self.n),matrix,conjugate};
            unitary.apply_batch_into(lap,state,self.n,1,complex,profile)?;
            if i+1<times.len(){frame_radiation_into(lap,middle.as_ref().unwrap(),state,complex,radiation,profile)?;}
        }
        frame_radiation_into(lap,&half,state,complex,radiation,profile)
    }
    /// Reuse numerical scratch only. Every call copies its NEW full input
    /// vector and recomputes the complete fixed-grid period/partial action.
    fn apply_frame_workspace(&self,lap:&Lapack,x:&[f64],macro_steps:usize,fraction:f64)->Result<Vec<f64>,String>{
        self.validate(x)?;let frame=self.bath_frame.as_ref().ok_or("Radiation frame is not prepared")?;
        if macro_steps>self.steps||!fraction.is_finite()||fraction<0.||fraction>self.step{
            return Err("Invalid frame-workspace period interval".into());
        }
        let mut saved=self.single_workspace.borrow_mut();
        if saved.is_none(){*saved=Some(SingleWorkspace{state:vec![Z::default();self.dimension()],
            complex:ComplexWorkspace::new(self.dimension()),radiation:frame.half.workspace(1)?});}
        let SingleWorkspace{state,complex,radiation}=saved.as_mut().unwrap();
        state.fill(Z::default());
        for(&coordinate,&value)in self.coords.iter().zip(x){match coordinate{
            Coordinate::Population(i)=>state[i+i*self.n].re=value,
            Coordinate::Real(i,j)=>{state[i+j*self.n].re=value;state[j+i*self.n].re=value;},
            Coordinate::Imag(i,j)=>{state[i+j*self.n].im=value;state[j+i*self.n].im=-value;},
        }}
        let transformed=frame.frame.into_frame_batch(lap,state)?;state.copy_from_slice(&transformed);
        let mut profile=BuildProfile::default();
        let stages=macro_steps*if self.fourth.is_some(){3}else{1};
        frame.apply_prefix_into(lap,state,self.n,1,stages,complex,radiation,&mut profile)?;
        if fraction>0.{
            let start=macro_steps as f64*self.step;
            self.advance_frame_fraction_into(lap,frame,state,complex,radiation,start,fraction,&mut profile)?;
        }
        if state.iter().any(|value|!value.re.is_finite()||!value.im.is_finite()){return Err("Non-finite frame-workspace period result".into());}
        let original=frame.frame.out_of_frame_batch(lap,state)?;
        Ok(coordinates_from_matrix(&self.coords,&original,self.n))
    }
}

#[cfg(test)]
mod tests{
    use super::*;use crate::{Operator,Drive,Options};
    fn damping_model()->Model{
        Model{ frequency_coordinates:None,dimension:3,states:vec![],h_real:vec![0.;9],h_imag:vec![0.;9],
            collapse:vec![Operator{real:vec![0.,1.,0.4,0.,0.,0.,0.,0.,0.],imag:vec![0.,0.,0.2,0.,0.,0.,0.,0.,0.]}],
            initial_index:1,initial_real:None,initial_imag:None,time_dependent:false,drives:vec![]}
    }
    #[test]
    fn exact_radiation_keeps_offdiagonal_gram_matches_dense_and_preserves_positivity(){
        let lap=Lapack::load().unwrap();let model=damping_model();let g=build_generator(&model).unwrap();
        let mut dense=vec![0.;81];for(j,col)in g.matrix.iter().enumerate(){for &(i,a)in col{dense[i+j*9]+=a;}}
        let action=Action::new(&lap,&model,0.7,5).unwrap();let actual=action.apply(&lap,&g.initial).unwrap();
        let expected=crate::dense_expm::action(&lap,&dense,&g.initial,&[0.7]).unwrap();
        let error=actual.iter().zip(&expected[0]).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
        assert!(error<2e-12,"Exact damping density discrepancy {error:e}");
        assert!((actual[..3].iter().sum::<f64>()-1.).abs()<2e-13);
        assert!(crate::positivity(&lap,&expand_density(&g.coords,&actual,3),3).unwrap()<2e-13);
    }
    fn driven()->Model{
        Model{ frequency_coordinates:None,dimension:2,states:vec![],h_real:vec![0.,0.3,0.3,1.3],h_imag:vec![0.;4],
            collapse:vec![Operator{real:vec![0.,1.,0.,0.],imag:vec![0.;4]}],initial_index:0,
            initial_real:None,initial_imag:None,time_dependent:true,drives:vec![Drive{omega:3.7,
                cos:Operator{real:vec![0.,0.2,0.2,0.],imag:vec![0.;4]},
                sin:Operator{real:vec![0.;4],imag:vec![0.,0.1,-0.1,0.]}}]}
    }
    #[test]
    fn cptp_midpoint_converges_to_full_time_dependent_density(){
        let lap=Lapack::load().unwrap();let model=driven();let period=std::f64::consts::TAU/3.7;
        let action=Action::new(&lap,&model,period,256).unwrap();let initial=vec![1.,0.,0.,0.];
        let actual=action.apply(&lap,&initial).unwrap();
        let options=Options{duration_us:period,samples:2,include_asymptotic:false,..Options::default()};
        let reference=crate::floquet::solve(&lap,&model,&options).unwrap();
        for i in 0..2{assert!((actual[i]-reference["populations"][1][i].as_f64().unwrap()).abs()<2e-5);}
        let partial=action.partial(&lap,&initial,0.371*period).unwrap();
        for row in [&actual,&partial]{assert!((row[..2].iter().sum::<f64>()-1.).abs()<2e-12);
            assert!(crate::positivity(&lap,&expand_density(&coordinates(2),row,2),2).unwrap()<2e-12);}
    }
    #[test]
    fn cptp_action_is_linear_on_arbitrary_hermitian_vectors(){
        let lap=Lapack::load().unwrap();let model=driven();let action=Action::new(&lap,&model,0.4,12).unwrap();
        let x=vec![0.4,-0.2,0.13,0.01];let y=vec![-0.1,0.7,-0.08,0.2];let combination:Vec<_>=x.iter().zip(&y).map(|(a,b)|1.7*a-0.2*b).collect();
        let a=action.apply(&lap,&x).unwrap();let b=action.apply(&lap,&y).unwrap();let combined=action.apply(&lap,&combination).unwrap();
        assert!(combined.iter().zip(a.iter().zip(b)).all(|(v,(a,b))|(v-(1.7*a-0.2*b)).abs()<2e-13));
        assert_eq!(action.partial(&lap,&x,0.).unwrap(),x);
        assert_eq!(action.partial(&lap,&x,0.4).unwrap(),a);
    }
    fn disconnected_bath_model()->Model{
        let mut first=Operator{real:vec![0.;25],imag:vec![0.;25]};first.real[1]=0.8;first.real[2]=0.2;first.imag[2]=0.3;
        let mut second=Operator{real:vec![0.;25],imag:vec![0.;25]};second.real[23]=0.6;
        Model{ frequency_coordinates:None,dimension:5,states:vec![],h_real:vec![0.;25],h_imag:vec![0.;25],collapse:vec![first,second],
            initial_index:0,initial_real:None,initial_imag:None,time_dependent:false,drives:vec![]}
    }
    #[test]
    fn exact_gram_blocks_match_dense_radiation_on_arbitrary_hermitian_input(){
        let lap=Lapack::load().unwrap();let model=disconnected_bath_model();let action=Action::new(&lap,&model,0.37,3).unwrap();
        assert_eq!(action.bath.components.iter().map(Vec::len).collect::<Vec<_>>(),vec![2,1]);
        let g=build_generator(&model).unwrap();let mut dense=vec![0.;625];
        for(j,col)in g.matrix.iter().enumerate(){for &(i,a)in col{dense[i+j*25]+=a;}}
        let initial:Vec<_>=(0..25).map(|i|((i as f64+0.2)*0.37).cos()).collect();
        let actual=action.apply(&lap,&initial).unwrap();let expected=crate::dense_expm::action(&lap,&dense,&initial,&[0.37]).unwrap();
        assert!(actual.iter().zip(&expected[0]).all(|(a,b)|(a-b).abs()<2e-12));
    }
    #[test]
    fn exact_gram_components_retain_arbitrarily_small_nonzero_connections(){
        let lap=Lapack::load().unwrap();let mut model=disconnected_bath_model();
        model.collapse[0].real[3]=1e-18;
        let bath=Bath::new(&lap,&model).unwrap();
        assert_eq!(bath.components.iter().map(Vec::len).collect::<Vec<_>>(),vec![3]);
    }
    #[test]
    fn batched_complete_period_map_matches_every_original_basis_column(){
        let lap=Lapack::load().unwrap();
        for model in [driven(),disconnected_bath_model()]{
            let action=Action::new(&lap,&model,0.37,7).unwrap();let d=action.dimension();
            let matrix=action.full_matrix(&lap).unwrap();assert_eq!(matrix.len(),d*d);
            for j in 0..d{let mut basis=vec![0.;d];basis[j]=1.;let expected=action.apply(&lap,&basis).unwrap();
                assert!(matrix[j*d..(j+1)*d].iter().zip(expected).all(|(a,b)|(a-b).abs()<2e-12),"basis column {j}");}
        }
    }
    #[test]
    fn symmetric_fourth_order_converges_on_complete_time_dependent_density(){
        let lap=Lapack::load().unwrap();let model=driven();let duration=0.7;
        let options=Options{duration_us:duration,samples:2,include_asymptotic:false,return_density:true,..Options::default()};
        let reference=crate::floquet::solve(&lap,&model,&options).unwrap();
        let re=reference["densityReal"][1].as_array().unwrap();let im=reference["densityImag"][1].as_array().unwrap();
        let expected=vec![re[0].as_f64().unwrap(),re[3].as_f64().unwrap(),re[1].as_f64().unwrap(),im[1].as_f64().unwrap()];
        let initial=vec![1.,0.,0.,0.];let mut errors=Vec::new();
        for steps in [4,8]{let action=Action::new_symmetric_fourth_order(&lap,&model,duration,steps).unwrap();
            assert!(!action.is_cptp());assert_eq!(action.order(),4);assert!(action.name().contains("not finite-grid CPTP"));
            let actual=action.apply(&lap,&initial).unwrap();
            errors.push(actual.iter().zip(&expected).map(|(a,b)|(a-b).abs()).fold(0.,f64::max));
            assert!((actual[..2].iter().sum::<f64>()-1.).abs()<2e-12);
        }
        assert!(errors[0]>8.*errors[1],"Fourth-order refinement {errors:?}");
        assert!(errors[1]<2e-5,"Fine full-density error {errors:?}");
        let positive=Action::new(&lap,&model,duration,4).unwrap();assert!(positive.is_cptp());assert_eq!(positive.order(),2);
    }
    #[test]
    fn signed_fourth_order_batch_and_fraction_match_direct_actions(){
        let lap=Lapack::load().unwrap();
        for model in [driven(),damping_model()]{
            let action=Action::new_symmetric_fourth_order(&lap,&model,0.4,4).unwrap();let d=action.dimension();
            let matrix=action.full_matrix(&lap).unwrap();
            for j in 0..d{let mut basis=vec![0.;d];basis[j]=1.;let expected=action.apply(&lap,&basis).unwrap();
                assert!(matrix[j*d..(j+1)*d].iter().zip(expected).all(|(a,b)|(a-b).abs()<3e-12));}
            let initial:Vec<_>=(0..d).map(|i|(i as f64+0.2).sin()).collect();
            assert_eq!(action.partial(&lap,&initial,0.).unwrap(),initial);
            let half=action.partial(&lap,&initial,0.2).unwrap();
            let expected=Action::new_symmetric_fourth_order(&lap,&model,0.2,2).unwrap().apply(&lap,&initial).unwrap();
            assert!(half.iter().zip(expected).all(|(a,b)|(a-b).abs()<2e-13));
            // A proper fractional macrostep samples its three Hamiltonians at
            // the shifted nonautonomous times, including the negative stage.
            assert!(action.partial(&lap,&initial,0.237).unwrap().iter().all(|x|x.is_finite()));
        }
    }
    #[test]
    fn radiation_eigenframe_matches_original_basis_complete_period_maps(){
        let lap=Lapack::load().unwrap();
        for model in [driven(),damping_model()]{for fourth in [false,true]{
            let mut original=Action::prepare(&lap,&model,0.37,5,fourth).unwrap();original.bath_frame=None;
            let expected=original.full_matrix(&lap).unwrap();
            let mut rotated=Action::prepare(&lap,&model,0.37,5,fourth).unwrap();
            rotated.prepare_bath_frame(&lap,&model).unwrap();
            let actual=rotated.full_matrix(&lap).unwrap();
            assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<3e-12));
            let profile=rotated.full_matrix_profile().unwrap();assert!(profile.bath_frame);
            assert_eq!(profile.unitary_steps,if fourth{15}else{5});
            assert_eq!(profile.radiation_applications,profile.unitary_steps+1);
        }}
    }
    #[test]
    fn reusable_local_transpose_workspace_matches_allocating_period_maps(){
        let lap=Lapack::load().unwrap();
        for model in [driven(),damping_model()]{for fourth in [false,true]{
            let mut action=Action::prepare(&lap,&model,0.37,5,fourth).unwrap();
            action.prepare_bath_frame(&lap,&model).unwrap();
            let expected=action.full_matrix_impl(&lap,false).unwrap();
            let actual=action.full_matrix_impl(&lap,true).unwrap();
            let error=actual.iter().zip(expected).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
            assert!(error<3e-12,"Workspace/local-transpose full-map discrepancy {error:e}");
            let profile=action.full_matrix_profile().unwrap();assert!(profile.workspace&&profile.bath_frame);
            assert_eq!(profile.unitary_steps,if fourth{15}else{5});
        }}
    }
    #[test]
    fn single_vector_workspace_reuses_scratch_and_recomputes_each_full_input(){
        let lap=Lapack::load().unwrap();
        for model in [driven(),damping_model()]{for fourth in [false,true]{
            let mut original=Action::prepare(&lap,&model,0.4,4,fourth).unwrap();original.bath_frame=None;
            let mut action=Action::prepare(&lap,&model,0.4,4,fourth).unwrap();action.prepare_bath_frame(&lap,&model).unwrap();
            let d=action.dimension();let mut pointers=None;
            for phase in [0.2,0.7]{
                let input:Vec<_>=(0..d).map(|i|(i as f64+phase).sin()).collect();
                let expected=original.apply(&lap,&input).unwrap();
                let actual=action.apply_frame_workspace(&lap,&input,4,0.).unwrap();
                assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<3e-12));
                let expected=original.partial(&lap,&input,0.237).unwrap();
                let actual=action.apply_frame_workspace(&lap,&input,2,0.037).unwrap();
                assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<3e-12));
                let saved=action.single_workspace.borrow();let saved=saved.as_ref().unwrap();
                let mut current=[saved.state.as_ptr()as usize,saved.complex.left.as_ptr()as usize,
                    saved.complex.packed.as_ptr()as usize,saved.complex.right.as_ptr()as usize];current.sort_unstable();
                if let Some(previous)=pointers{assert_eq!(current,previous);}else{pointers=Some(current);}
            }
        }}
    }
    #[test]
    fn batched_partial_phases_match_individual_arbitrary_hermitian_states(){
        let lap=Lapack::load().unwrap();let phases=vec![0.237,0.,0.4,0.1,0.037,0.3,0.237];
        for model in [driven(),damping_model()]{for fourth in [false,true]{
            let mut original=Action::prepare(&lap,&model,0.4,4,fourth).unwrap();original.bath_frame=None;
            let mut action=Action::prepare(&lap,&model,0.4,4,fourth).unwrap();action.prepare_bath_frame(&lap,&model).unwrap();
            let initial:Vec<Vec<f64>>=(0..phases.len()).map(|j|(0..action.dimension()).map(|i|(i as f64+0.2*(j+1)as f64).sin()).collect()).collect();
            let expected:Vec<_>=initial.iter().zip(&phases).map(|(state,&phase)|original.partial(&lap,state,phase).unwrap()).collect();
            let actual=action.partial_many(&lap,&initial,&phases).unwrap();
            assert_eq!(actual[1],initial[1]);assert_eq!(actual.len(),phases.len());
            let error=actual.iter().zip(&expected).flat_map(|(a,b)|a.iter().zip(b).map(|(a,b)|(a-b).abs())).fold(0.,f64::max);
            assert!(error<4e-12,"Batched exact-phase discrepancy {error:e}, fourth={fourth}");
            assert!(action.partial_many(&lap,&[],&[]).unwrap().is_empty());
            assert!(action.partial_many(&lap,&initial,&phases[..1]).is_err());
        }}
    }
    fn complex_bath_and_hamiltonian()->Model{
        let mut model=damping_model();model.h_real=vec![0.2,0.3,-0.1,0.3,0.7,0.4,-0.1,0.4,-0.3];
        model.h_imag=vec![0.,0.11,-0.07,-0.11,0.,0.2,0.07,-0.2,0.];
        model
    }
    #[test]
    fn alternating_storage_matches_original_full_maps_with_odd_and_even_stage_counts(){
        let lap=Lapack::load().unwrap();let model=complex_bath_and_hamiltonian();
        for fourth in [false,true]{for steps in [4,5]{
            let mut action=Action::prepare(&lap,&model,0.37,steps,fourth).unwrap();action.prepare_bath_frame(&lap,&model).unwrap();
            let expected=action.full_matrix_options(&lap,true,false,false).unwrap();
            let actual=action.full_matrix_options(&lap,true,false,true).unwrap();
            let error=actual.iter().zip(expected).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
            assert!(error<4e-12,"Alternating storage map changed by {error:e}, fourth={fourth}, steps={steps}");
            let profile=action.full_matrix_profile().unwrap();assert!(profile.alternating_layout);assert!(!profile.bath_frame_coordinates);
        }}
    }
    #[test]
    fn bath_coordinate_period_powers_and_exact_phases_match_original_basis(){
        let lap=Lapack::load().unwrap();let model=complex_bath_and_hamiltonian();let phases=vec![0.,0.035,0.2,0.369,0.37];
        for fourth in [false,true]{
            let mut action=Action::prepare(&lap,&model,0.37,5,fourth).unwrap();action.prepare_bath_frame(&lap,&model).unwrap();let d=action.dimension();
            let original_map=action.full_matrix_options(&lap,true,false,false).unwrap();
            let frame_map=action.full_matrix_options(&lap,true,true,true).unwrap();
            let profile=action.full_matrix_profile().unwrap();assert!(profile.bath_frame_coordinates);assert_eq!(profile.frame_transform_ms,0.);
            let mut original:Vec<_>=(0..d).map(|i|(0.31+i as f64*0.7).sin()).collect();
            let mut framed=action.state_into_bath_frame(&lap,&original).unwrap();
            for _ in 0..3{original=lap.multiply(&original_map,&original,d,1,d);framed=lap.multiply(&frame_map,&framed,d,1,d);}
            let expected=action.partial_many(&lap,&vec![original;phases.len()],&phases).unwrap();
            let actual=action.partial_many_from_bath_frame(&lap,&vec![framed;phases.len()],&phases).unwrap();
            let error=actual.iter().zip(expected).flat_map(|(a,b)|a.iter().zip(b).map(|(a,b)|(a-b).abs())).fold(0.,f64::max);
            assert!(error<8e-12,"Bath-coordinate powers/phases changed by {error:e}, fourth={fourth}");
        }
    }
}
