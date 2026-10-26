//! Original Model initial data and complete physical observations for the
//! SAME current-request coordinates T(theta)=(I+Khat(theta)) J2(S(theta)).
//! No population is clipped, normalized, made real, or fed back into evolution.
use super::*;
use crate::parallel::Execution;
use crate::pencil::{Commutator, Gemm, Scratch};

/// States pulled back together through the shear layers (see
/// `PopulationObservation::prepare`).
const STATE_CHUNK:usize=4;

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
        // Invert the shear layers at theta = 0, outermost first.
        let(mut point,mut anti_error,mut minimum)=(before,0_f64,f64::INFINITY);
        for layer in p.layers.iter().rev(){
            let points:Vec<Vec<C>>=layer.iter().map(|class|evaluate(class,phases,0.,d)).collect::<Result<_,_>>()?;
            for(s,class)in points.iter().zip(layer){let anti:Vec<_>=(0..d).map(|i|{
                let z=conj(s[(i%n)*n+i/n]);C{re:s[i].re+z.re,im:s[i].im+z.im}}).collect();
                let error=norm(&anti);anti_error=anti_error.max(error);let construction_scale=class.values().map(|a|norm(a)).sum::<f64>();
                if error>256.*f64::EPSILON*n as f64*construction_scale.max(f64::MIN_POSITIVE){
                    return Err("Point S is not anti-Hermitian within coefficient accumulation roundoff".into());}}
            if points.len()!=1{
                // A sum of class shears: invert I + sum_c D_c by fixed-point
                // iteration (each D_c is small), checked against the input.
                let target=point.clone();let scale=maximum(&target).max(f64::MIN_POSITIVE);let mut converged=false;
                for _ in 0..200{let mut image=point.clone();
                    for s in &points{let once=commutator(&self.lapack,s,&point,n)?;let twice=commutator(&self.lapack,s,&once,n)?;
                        for i in 0..d{add(&mut image[i],once[i]);add(&mut image[i],scaled(twice[i],0.5));}}
                    let mut change=0_f64;for i in 0..d{let r=C{re:target[i].re-image[i].re,im:target[i].im-image[i].im};add(&mut point[i],r);change=change.max(r.re.hypot(r.im));}
                    if change<=4.*f64::EPSILON*scale{converged=true;break;}}
                if !converged{return Err("Shear layer point inverse did not converge".into());}
                continue;
            }
            let s=&points[0];
            // The same lower-Hermitian eigenselector as scipy.linalg.eigh(-i*S).
            // It is only used for inversion. Forward validation uses UNMODIFIED S.
            let mut selector:Vec<_>=s.iter().map(|z|C{re:z.im,im:-z.re}).collect();
            for i in 0..n{selector[i*n+i].im=0.;for j in i+1..n{selector[i*n+j]=conj(selector[j*n+i]);}}
            let(energies,vectors)=self.lapack.hermitian_eigen(&row_to_column(&selector,n),n)?;
            let mut transformed=crate::preparation::hilbert::transform_matrix(&self.lapack,&vectors,&point,n)?;
            for i in 0..n{for j in 0..n{let delta=energies[i]-energies[j];let denominator=C{re:1.-0.5*delta*delta,im:delta};
                let modulus=denominator.re.hypot(denominator.im);minimum=minimum.min(modulus);
                if !modulus.is_finite()||modulus==0.{return Err("Unresolved J2 point inverse denominator".into());}
                let normalized=scaled(denominator,1./modulus);transformed[i*n+j]=scaled(mul(transformed[i*n+j],conj(normalized)),1./modulus);
            }}
            point=restore_hilbert(&self.lapack,&vectors,&transformed,n)?;
        }
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
        let phases=&p.hilbert.source.phases;let mut j=envelope.to_vec();
        for layer in &p.layers{let mut next=j.clone();for class in layer{let s=evaluate(class,phases,time,d)?;
            let once=commutator(&self.lapack,&s,&j,n)?;let twice=commutator(&self.lapack,&s,&once,n)?;
            for i in 0..d{add(&mut next[i],once[i]);add(&mut next[i],scaled(twice[i],0.5));}}j=next;}
        let k=evaluate(&p.k_hat,phases,time,g*g*e*e)?;let z=shear_action(&k,&j,n,g);for(a,b)in j.iter_mut().zip(z){add(a,b);}
        let mut physical=restore_hilbert(&self.lapack,&p.hilbert.basis_column_major,&j,n)?;
        let state_phases=p.hilbert.source.gauges.iter().map(|q|phases.phase(q,time)).collect::<Result<Vec<_>,_>>()?;
        for i in 0..n{for j in 0..n{physical[i*n+j]=mul(mul(state_phases[i],physical[i*n+j]),conj(state_phases[j]));}}
        if physical.iter().any(|z|!z.finite()){return Err("Nonfinite physical reconstruction".into());}Ok(physical)
    }
    /// One Fourier coefficient of `E x` (row-major, Hilbert frame) in the
    /// original state basis: the `restore_hilbert` step of `from_envelope_at`,
    /// without the time-dependent state-gauge phases.
    pub fn hilbert_to_states(&self,matrix:&[C])->Result<Vec<C>,String>{
        let n=self.dimension();
        if matrix.len()!=n*n||matrix.iter().any(|z|!z.finite()){return Err("Expected a finite complete Hilbert-frame matrix".into());}
        restore_hilbert(&self.lapack,&self.primitives.hilbert.basis_column_major,matrix,n)
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
pub struct PopulationBasis{modes:Vec<Charge>,columns:usize,coefficients:Vec<C>}
/// Readout duals of `states` state functionals over an ordered list of
/// charges. Each charge holds them as one mode block with the states as
/// right-hand sides: entry (i, j) of local state s sits at i + n*(s + states*j).
struct Duals{charges:Vec<Charge>,states:usize,values:Vec<C>}
impl Duals{
    fn zeros(charges:Vec<Charge>,n:usize,states:usize)->Self{let length=charges.len()*n*n*states;Self{charges,states,values:vec![C::default();length]}}
    /// The sum of several dual families over the union of their charges.
    fn sum(parts:&[&Duals],n:usize)->Self{
        let charges:Vec<Charge>=parts.iter().flat_map(|d|d.charges.iter().cloned()).collect::<Modes>().into_iter().collect();
        let index:BTreeMap<&Charge,usize>=charges.iter().enumerate().map(|(i,q)|(q,i)).collect();
        let states=parts[0].states;let size=n*n*states;let mut out=Self::zeros(charges.clone(),n,states);
        for part in parts{for(m,q)in part.charges.iter().enumerate(){let target=index[q]*size;
            for(a,&b)in out.values[target..target+size].iter_mut().zip(part.block(n,m)){add(a,b);}}}
        out
    }
    fn block(&self,n:usize,index:usize)->&[C]{let size=n*n*self.states;&self.values[index*size..(index+1)*size]}
}
fn dual_at(n:usize,states:usize,s:usize,i:usize,j:usize)->usize{i+n*(s+states*j)}
/// Hermitian pairing of a charge set closed under q -> -q: [zero, positive q
/// (sorted), -q in the same order] and the number of leading canonical
/// charges. None when the set is not closed.
fn paired_order(set:&Modes)->Option<(Vec<Charge>,usize)>{
    let negate=|q:&Charge|->Option<Charge>{q.iter().map(|&x|x.checked_neg()).collect()};
    if !set.iter().all(|q|negate(q).is_some_and(|m|set.contains(&m))){return None;}
    let positive=|q:&Charge|q.iter().copied().find(|&x|x!=0).is_some_and(|x|x>0);
    let central:Vec<Charge>=set.iter().filter(|q|q.iter().all(|&x|x==0)).cloned().collect();
    let positives:Vec<Charge>=set.iter().filter(|q|positive(q)).cloned().collect();
    let negatives:Vec<Charge>=positives.iter().map(|q|negate(q).unwrap()).collect();
    let canonical=central.len()+positives.len();
    Some(([central,positives,negatives].concat(),canonical))
}
/// Pull-back through ad_S: Q -> S^T Q - Q S^T is the bilinear adjoint of
/// X -> [S, X], and a dual at charge q reaches charge q+p through S_p. Runs on
/// the pencil's blocked commutator kernel.
struct DualCommutator{n:usize,ground:usize,transposed:FourierMatrices,lapack:Arc<Lapack>,execution:Execution,scratch:Vec<Scratch>}
impl DualCommutator{
    fn new(s:&FourierMatrices,n:usize,ground:usize,lapack:Arc<Lapack>)->Result<Self,String>{
        // row_to_column of a row-major matrix is its transpose (row-major).
        let transposed=s.iter().map(|(q,a)|(q.clone(),row_to_column(a,n))).collect();
        Ok(Self{n,ground,transposed,lapack,execution:Execution::new()?,scratch:Vec::new()})
    }
    /// alpha * ad_S^T(input), on every charge input+p it reaches. With
    /// `paired` the input is Hermitian-paired, D_{-q}[s] = D_q[s]^dagger (the
    /// readout duals are: O_s is Hermitian, S_{-p} = -S_p^dagger and K pairs
    /// the same way), so only canonical charges are computed and their
    /// partners mirrored.
    fn apply(&mut self,input:&Duals,alpha:f64,paired:bool)->Result<Duals,String>{
        let(n,states)=(self.n,input.states);let size=n*n*states;
        let sources:Modes=input.charges.iter().cloned().collect();let steps:Modes=self.transposed.keys().cloned().collect();
        let reached=combinations(&sources,&steps,false)?;
        let(charges,canonical)=match paired.then(||paired_order(&reached)).flatten(){
            Some(order)=>order,None=>{let length=reached.len();(reached.into_iter().collect(),length)}};
        let mut output=Duals::zeros(charges,n,states);
        let stage=Commutator::new(&self.transposed,&input.charges,&output.charges[..canonical],n,self.ground)?;
        let gemm=Gemm::Blas(self.lapack.complex_gemm());
        stage.apply(&self.execution,gemm,&input.values,&mut output.values[..canonical*size],states,alpha,&mut self.scratch)?;
        let pairs=output.charges.len()-canonical;let central=canonical-pairs;
        let(computed,mirrored)=output.values.split_at_mut(canonical*size);
        for(k,target)in mirrored.chunks_exact_mut(size).enumerate(){let source=&computed[(central+k)*size..(central+k+1)*size];
            for state in 0..states{for j in 0..n{for i in 0..n{target[dual_at(n,states,state,i,j)]=conj(source[dual_at(n,states,state,j,i)]);}}}}
        Ok(output)
    }
}
impl PopulationObservation{
    /// Readout duals D = T_0^T ... T_last^T (I + K^T) O, with
    /// T^T = I + sum_c (ad_{S_c}^T + (ad_{S_c}^T)^2/2) per shear layer and
    /// O_s the population functional of state s in the Hilbert frame. States
    /// whose (I + K^T) O_s are nonzero on the same charges are pulled back
    /// together, `STATE_CHUNK` at a time.
    fn prepare(coordinates:&PhysicalCoordinates)->Result<Self,String>{
        let p=&coordinates.primitives;let n=p.hilbert.source.n;let g=p.hilbert.source.ground.len();let e=n-g;
        let basis=&p.hilbert.basis_column_major;let zero=vec![0;p.hilbert.source.phases.axes.len()];
        let at=|s:usize,i:usize,j:usize|dual_at(n,n,s,i,j);
        // O at charge 0 and its K^T pull-back (EE entries) at every K charge.
        let mut first:Modes=p.k_hat.keys().cloned().collect();first.insert(zero.clone());
        let mut before=Duals::zeros(first.into_iter().collect(),n,n);let size=n*n*n;
        let index:BTreeMap<Charge,usize>=before.charges.iter().cloned().enumerate().map(|(i,q)|(q,i)).collect();
        let central=index[&zero];
        for state in 0..n{for a in 0..n{for b in 0..n{
            before.values[central*size+at(state,a,b)]=mul(basis[state+a*n],conj(basis[state+b*n]));
        }}}
        // D_q[s]_EE(i,j) += sum_{a,b<g} O_s[a,b] K_q[(a,b),(i,j)]: K_q row-major
        // (g*g x e*e) is column-major (e*e x g*g), so one GEMM per charge.
        let mut ground=vec![C::default();g*g*n];
        for state in 0..n{for a in 0..g{for b in 0..g{ground[a*g+b+g*g*state]=before.values[central*size+at(state,a,b)];}}}
        let gemm=coordinates.lapack.complex_gemm();let mut pulled=vec![C::default();e*e*n];
        for(q,k)in &p.k_hat{
            gemm.multiply(e*e,n,g*g,C{re:1.,im:0.},k,e*e,&ground,g*g,C::default(),&mut pulled,e*e)?;
            let target=index[q]*size;
            for state in 0..n{for i in 0..e{for j in 0..e{add(&mut before.values[target+at(state,g+i,g+j)],pulled[i*e+j+e*e*state]);}}}
        }
        // Group states by the charges on which their functional is nonzero.
        let nonzero=|m:usize,state:usize|(0..n).any(|i|(0..n).any(|j|{let z=before.values[m*size+at(state,i,j)];z.re!=0.||z.im!=0.}));
        let mut patterns=BTreeMap::<Vec<usize>,Vec<usize>>::new();
        for state in 0..n{let key:Vec<usize>=(0..before.charges.len()).filter(|&m|nonzero(m,state)).collect();
            if !key.is_empty(){patterns.entry(key).or_default().push(state);}}
        // Entries below 1e-15 of the largest functional entry are accumulated
        // roundoff of the S/K pull-back (which changes entries by O(|S|)
        // relative). Dropping them changes a population by at most
        // 1e-15*max|O|*(number of dropped entries)*|x|, far below any
        // requested tolerance, and removes most of the readout work.
        let magnitude=|z:C|z.re*z.re+z.im*z.im;
        let largest=before.values.iter().map(|&z|magnitude(z)).fold(0_f64,f64::max).sqrt();
        let floor=1e-15*largest;let floor_squared=floor*floor;
        if largest>0.&&!floor_squared.is_normal(){return Err("Readout dual magnitudes outside the normal floating-point range".into());}
        // Pulled back through the shear layers, outermost first:
        // D <- D + ad^T D + (ad^T)^2 D / 2 per layer. The states go through
        // STATE_CHUNK at a time and each chunk keeps only its entries above
        // the floor: with a strong far laser the layers reach thousands of
        // charges, where dense duals of all states would take ~1 GB while
        // under 1 % of their entries are nonzero.
        let mut commutators=p.layers.iter().rev().map(|layer|layer.iter().map(|class|DualCommutator::new(class,n,g,Arc::clone(&coordinates.lapack))).collect::<Result<Vec<_>,_>>()).collect::<Result<Vec<_>,_>>()?;
        let execution=Execution::new()?;let execution=&execution;let minimum=crate::parallel::MINIMUM_WORK_PER_TASK;
        let mut collected=BTreeMap::<Charge,Vec<(usize,usize,C)>>::new();
        for(key,states)in patterns{for chunk in states.chunks(STATE_CHUNK){let r=chunk.len();
            let mut own=Duals::zeros(key.iter().map(|&m|before.charges[m].clone()).collect(),n,r);
            for(local_charge,&m)in key.iter().enumerate(){for(local,&state)in chunk.iter().enumerate(){for i in 0..n{for j in 0..n{
                own.values[local_charge*n*n*r+dual_at(n,r,local,i,j)]=before.values[m*size+at(state,i,j)];
            }}}}
            for layer in &mut commutators{let mut total:Option<Duals>=None;
                for commutator in layer.iter_mut(){let once=commutator.apply(&own,1.,true)?;let twice=commutator.apply(&once,0.5,true)?;
                    total=Some(Duals::sum(&[total.as_ref().unwrap_or(&own),&once,&twice],n));}
                if let Some(total)=total{own=total;}
            }
            let mut rows=vec![Vec::new();own.charges.len()];let work=own.charges.len()*n*n*r*4;
            execution.for_mode_chunks(&mut rows,1,work,minimum,&mut Vec::<()>::new(),|first,part,_|{
                for(k,entries)in part.iter_mut().enumerate(){let block=own.block(n,first+k);
                    for(local,&state)in chunk.iter().enumerate(){for i in 0..n{for j in 0..n{let z=block[dual_at(n,r,local,i,j)];
                        if magnitude(z)>floor_squared{entries.push((state,i*n+j,z));}
                    }}}
                }Ok(())
            })?;
            for(q,entries)in own.charges.iter().zip(rows){if !entries.is_empty(){collected.entry(q.clone()).or_default().extend(entries);}}
        }}
        let(charges,rows):(Vec<_>,Vec<_>)=collected.into_iter().map(|(q,mut entries)|{entries.sort_by_key(|&(state,at,_)|(state,at));(q,entries)}).unzip();
        Ok(Self{owner:Arc::clone(p),charges,rows})
    }
    /// Output modes reached by the readout from the given domain modes.
    pub fn output_modes(&self,input_modes:&[Charge])->Result<Vec<Charge>,String>{
        let inputs=mode_set(input_modes,self.owner.hilbert.source.phases.axes.len())?;
        let support:Modes=self.charges.iter().cloned().collect();
        Ok(combinations(&support,&inputs,false)?.into_iter().collect())
    }
    pub fn frequency(&self,q:&[i32])->Result<crate::preparation::FrequencyParts,String>{self.owner.hilbert.source.phases.frequency(q)}
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
        Ok(PopulationBasis{modes,columns,coefficients})
    }
}
impl PopulationBasis{
    pub fn modes(&self)->&[Charge]{&self.modes}
    pub fn columns(&self)->usize{self.columns}
    /// Row-major `[(mode*n+state), column]` complex coefficients.
    pub fn coefficients(&self)->&[C]{&self.coefficients}
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn preparation_observation_batched_dual_and_fixed_phase_plan_keep_complete_oracle(){
        let owner=super::super::tests::fixture();let coordinates=owner.physical_coordinates();let n=coordinates.dimension();let d=n*n;let p=owner.primitives();let lap=&owner.preparation_lapack;
        let raw:FourierMatrices=[vec![0],vec![2]].into_iter().enumerate().map(|(m,q)|(q,(0..n*d).map(|i|C{re:((i*13+m*7)%31)as f64/37.,im:((i*7+m*11)%29)as f64/-41.}).collect())).collect();
        let add_dual=|family:&mut FourierMatrices,q:Charge,value:&[C]|{
            let out=family.entry(q).or_insert_with(||vec![C::default();value.len()]);for(a,&b)in out.iter_mut().zip(value){add(a,b);}};
        let oracle=|input:&FourierMatrices|{let mut out=FourierMatrices::new();for(q,values)in input{for(r,s)in &p.s{
            let st:Vec<_>=(0..d).map(|i|s[(i%n)*n+i/n]).collect();let mut result=vec![C::default();n*d];
            for state in 0..n{result[state*d..(state+1)*d].copy_from_slice(&commutator(lap,&st,&values[state*d..(state+1)*d],n).unwrap());}
            add_dual(&mut out,add_charge(q,r,false).unwrap(),&result);
        }}out};
        // Row-major per-state duals <-> mode-block layout with states as rhs.
        let at=|s:usize,i:usize,j:usize|i+n*(s+n*j);
        let to_duals=|f:&FourierMatrices|{let mut duals=Duals::zeros(f.keys().cloned().collect(),n,n);
            for(m,values)in f.values().enumerate(){for s in 0..n{for i in 0..n{for j in 0..n{duals.values[m*n*d+at(s,i,j)]=values[s*d+i*n+j];}}}}duals};
        let from_duals=|duals:&Duals|->FourierMatrices{duals.charges.iter().enumerate().map(|(m,q)|(q.clone(),
            (0..n*d).map(|k|duals.block(n,m)[at(k/d,(k%d)/n,k%n)]).collect())).collect()};
        let g=p.hilbert.source.ground.len();let mut plan=DualCommutator::new(&p.s,n,g,Arc::clone(lap)).unwrap();
        let first=plan.apply(&to_duals(&raw),1.,false).unwrap();let second=plan.apply(&first,1.,false).unwrap();
        let(first,second)=(from_duals(&first),from_duals(&second));let old=oracle(&raw);let old_second=oracle(&old);
        for(actual,expected)in[(&first,&old),(&second,&old_second)]{assert_eq!(actual.len(),expected.len());for(q,a)in actual{let b=&expected[q];for(x,y)in a.iter().zip(b){assert!((x.re-y.re).hypot(x.im-y.im)<5e-13*(1.+y.re.hypot(y.im)));}}}
        // Hermitian-paired duals over a symmetric charge set: the mirrored
        // partners must equal the full oracle as well.
        let one:Vec<C>=(0..n*d).map(|i|C{re:((i*5+3)%17)as f64/19.,im:((i*3+1)%13)as f64/-23.}).collect();
        let adjoint=|a:&[C]|->Vec<C>{(0..n*d).map(|k|{let(st,i,j)=(k/d,(k%d)/n,k%n);conj(a[st*d+j*n+i])}).collect()};
        let hermitian:Vec<C>=one.iter().zip(adjoint(&one)).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
        let paired_raw:FourierMatrices=[(vec![-1],adjoint(&one)),(vec![0],hermitian),(vec![1],one.clone())].into_iter().collect();
        let mirrored=from_duals(&plan.apply(&to_duals(&paired_raw),1.,true).unwrap());let expected=oracle(&paired_raw);
        assert_eq!(mirrored.len(),expected.len());
        for(q,a)in &mirrored{let b=&expected[q];for(x,y)in a.iter().zip(b){assert!((x.re-y.re).hypot(x.im-y.im)<5e-13*(1.+y.re.hypot(y.im)));}}
        // Readout coefficients reproduce the diagonal of the complete physical
        // reconstruction T(theta) at arbitrary times, for arbitrary complex fields.
        let observer=coordinates.population_observation().unwrap();let inputs=[vec![1],vec![0],vec![-1]];
        let fields:Vec<_>=(0..inputs.len()*d*2).map(|i|C{re:(i as f64*0.17).sin(),im:(i as f64*0.13).cos()}).collect();
        let basis=observer.project(&inputs,&fields,2).unwrap();let block=n*2;
        for time in[0.,0.137,1e3+0.123456789]{
            for j in 0..2{
                let column:Vec<_>=(0..inputs.len()*d).map(|i|fields[i*2+j]).collect();
                let physical=coordinates.field_at(&inputs,&column,time).unwrap();
                for state in 0..n{
                    let mut value=C::default();
                    for(mode,q)in basis.modes().iter().enumerate(){
                        let phase=p.hilbert.source.phases.phase(q,time).unwrap();
                        add(&mut value,mul(phase,basis.coefficients()[mode*block+state*2+j]));
                    }
                    let expected=physical[state*n+state];
                    assert!((value.re-expected.re).hypot(value.im-expected.im)<1e-11*(1.+expected.re.hypot(expected.im)));
                }
            }
        }
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
        let times=[0.,0.137,1e6+0.123456789];let nt=times.len();let block=n*columns;
        let small:Vec<_>=(0..columns*nt).map(|i|C{re:0.2+i as f64*0.07,im:-0.3+i as f64*0.11}).collect();
        for(t,&time)in times.iter().enumerate(){let mut field=vec![C::default();modes.len()*d];
            for i in 0..field.len(){for j in 0..columns{add(&mut field[i],mul(basis[i*columns+j],small[j*nt+t]));}}
            let physical=coordinates.field_at(&modes,&field,time).unwrap();
            for state in 0..n{
                let mut value=C::default();
                for(mode,q)in projected.modes().iter().enumerate(){let phase=owner.primitives().hilbert.source.phases.phase(q,time).unwrap();
                    for j in 0..columns{add(&mut value,mul(phase,mul(projected.coefficients()[mode*block+state*columns+j],small[j*nt+t])));}}
                let b=physical[state*n+state];assert!((value.re-b.re).hypot(value.im-b.im)<3e-12*(1.+b.re.hypot(b.im)));
            }
        }
        // Per-column incremental projection is identical to projecting a block.
        for j in 0..columns{let column:Vec<_>=(0..modes.len()*d).map(|i|basis[i*columns+j]).collect();
            let single=observer.project(&modes,&column,1).unwrap();
            for row in 0..projected.modes().len()*n{let(a,b)=(single.coefficients()[row],projected.coefficients()[row*columns+j]);
                assert_eq!(a.re.to_bits(),b.re.to_bits());assert_eq!(a.im.to_bits(),b.im.to_bits());}
        }
        assert!(observer.project(&[vec![0],vec![0]],&basis,columns).is_err());
    }
}
