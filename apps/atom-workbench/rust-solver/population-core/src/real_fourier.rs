//! REAL orthonormal Hermitian/Fourier coordinates <-> complete signed fields.
//! This is an isometry, not a density repair. The complex extension and adjoint
//! preserve arbitrary complex data; physical operator callers must retain the
//! pulled imaginary residual rather than silently accepting its real part.
use super::*;

#[derive(Clone,Debug)]
pub struct HarmonicBlock{pub index:Charge,pub offset:usize,pub has_imaginary_block:bool}
#[derive(Clone,Copy)]struct Entry{row:usize,value:C}
pub struct RealFourierMap{
    n:usize,modes:Vec<Charge>,harmonics:Vec<HarmonicBlock>,offsets:Vec<usize>,entries:Vec<Entry>,
}
#[derive(Clone,Debug)]
pub struct ProposalProjectionDiagnostics{pub imaginary_hs_by_column:Vec<f64>,pub input_hs_by_column:Vec<f64>,pub real_part_defines_numerical_proposal_only:bool}
fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
fn add(a:&mut C,b:C){a.re+=b.re;a.im+=b.im;}
fn conj(a:C)->C{C{re:a.re,im:-a.im}}
impl RealFourierMap{
    pub fn new(n:usize,modes:&[Charge])->Result<Self,String>{
        if n==0||n>64||modes.is_empty(){return Err("Invalid complete REAL Fourier dimensions".into());}
        let axes=modes[0].len();let set=mode_set(modes,axes)?;let zero=vec![0;axes];
        if !set.contains(&zero){return Err("REAL Fourier domain must include its central mode".into());}
        for q in &set{let other=q.iter().map(|&x|x.checked_neg().ok_or("REAL Fourier partner overflow")).collect::<Result<Charge,_>>()?;
            if !set.contains(&other){return Err("REAL Fourier domain must include every exact signed partner".into());}}
        let d=product(&[n,n])?;let positions:BTreeMap<_,_>=modes.iter().enumerate().map(|(i,q)|(q,i)).collect();
        let mut order=vec![zero.clone()];order.extend(modes.iter().filter(|q|q.iter().copied().find(|&v|v!=0).is_some_and(|v|v>0)).cloned());
        // One normalized Hermitian basis for every complete complex n*n matrix.
        let r=std::f64::consts::FRAC_1_SQRT_2;let mut base=Vec::<Vec<Entry>>::new();
        for i in 0..n{base.push(vec![Entry{row:i*n+i,value:C{re:1.,im:0.}}]);}
        for i in 0..n{for j in i+1..n{
            base.push(vec![Entry{row:i*n+j,value:C{re:r,im:0.}},Entry{row:j*n+i,value:C{re:r,im:0.}}]);
            base.push(vec![Entry{row:i*n+j,value:C{re:0.,im:r}},Entry{row:j*n+i,value:C{re:0.,im:-r}}]);
        }}assert_eq!(base.len(),d);
        let mut entries=Vec::new();let mut offsets=vec![0];let mut harmonics=Vec::new();
        for q in order{let central=q==zero;let offset=offsets.len()-1;harmonics.push(HarmonicBlock{index:q.clone(),offset,has_imaginary_block:!central});
            if central{for column in &base{for e in column{entries.push(Entry{row:positions[&q]*d+e.row,value:e.value});}offsets.push(entries.len());}}
            else{let negative:Charge=q.iter().map(|&x|-x).collect();
                for imaginary in[false,true]{for column in &base{
                    for(mode,phase)in[(&q,if imaginary{C{re:0.,im:r}}else{C{re:r,im:0.}}),
                        (&negative,if imaginary{C{re:0.,im:-r}}else{C{re:r,im:0.}})]{
                        for e in column{entries.push(Entry{row:positions[mode]*d+e.row,value:mul(phase,e.value)});}
                    }offsets.push(entries.len());
                }}
            }
        }
        if offsets.len()-1!=product(&[modes.len(),d])?{return Err("Incomplete REAL Fourier coordinate count".into());}
        Ok(Self{n,modes:modes.to_vec(),harmonics,offsets,entries})
    }
    pub fn dimension(&self)->usize{self.offsets.len()-1}
    pub fn physical_dimension(&self)->usize{self.n}
    pub fn modes(&self)->&[Charge]{&self.modes}
    pub fn harmonics(&self)->&[HarmonicBlock]{&self.harmonics}
    fn shape(&self,source:usize,target:usize,rhs:usize)->Result<(),String>{
        if rhs==0||source!=product(&[self.dimension(),rhs])?||target!=source{return Err("REAL map requires complete [coordinate,rhs] buffers".into());}Ok(())
    }
    pub fn push_into(&self,input:&[f64],rhs:usize,output:&mut[C])->Result<(),String>{
        self.shape(input.len(),output.len(),rhs)?;if input.iter().any(|x|!x.is_finite()){return Err("Nonfinite REAL Fourier input".into());}
        output.fill(C::default());for column in 0..self.dimension(){for e in &self.entries[self.offsets[column]..self.offsets[column+1]]{for b in 0..rhs{
            let value=input[column*rhs+b];add(&mut output[e.row*rhs+b],C{re:e.value.re*value,im:e.value.im*value});
        }}}if output.iter().any(|z|!z.finite()){return Err("REAL Fourier push overflow".into());}Ok(())
    }
    /// Complex-linear extension of the same isometry, used by small-system
    /// algebra. Complex input here is not declared a Hermitian density.
    pub fn push_complex_into(&self,input:&[C],rhs:usize,output:&mut[C])->Result<(),String>{
        self.shape(input.len(),output.len(),rhs)?;if input.iter().any(|x|!x.finite()){return Err("Nonfinite complex Fourier coordinates".into());}
        output.fill(C::default());for column in 0..self.dimension(){for e in &self.entries[self.offsets[column]..self.offsets[column+1]]{for b in 0..rhs{
            add(&mut output[e.row*rhs+b],mul(e.value,input[column*rhs+b]));
        }}}if output.iter().any(|z|!z.finite()){return Err("Complex Fourier push overflow".into());}Ok(())
    }
    /// Full T^dagger result. Imaginary parts are intentionally retained so an
    /// original physical E/F/M guard can detect wrong or unresolved actions.
    pub fn pull_complex_into(&self,input:&[C],rhs:usize,output:&mut[C])->Result<(),String>{
        self.shape(input.len(),output.len(),rhs)?;if input.iter().any(|x|!x.finite()){return Err("Nonfinite complete signed Fourier field".into());}
        output.fill(C::default());for column in 0..self.dimension(){for e in &self.entries[self.offsets[column]..self.offsets[column+1]]{for b in 0..rhs{
            add(&mut output[column*rhs+b],mul(conj(e.value),input[e.row*rhs+b]));
        }}}if output.iter().any(|z|!z.finite()){return Err("Complex Fourier pull overflow".into());}Ok(())
    }
    /// |T^dagger| times the complete native arithmetic envelope. It is used
    /// only for the same entered-basis reality guard, not population accuracy.
    pub fn pull_envelope_into(&self,input:&[f64],rhs:usize,output:&mut[f64])->Result<(),String>{
        self.shape(input.len(),output.len(),rhs)?;
        if input.iter().any(|x|!x.is_finite()||*x<0.){return Err("Invalid nonnegative Fourier operation envelope".into());}
        output.fill(0.);for column in 0..self.dimension(){for e in &self.entries[self.offsets[column]..self.offsets[column+1]]{let weight=e.value.re.hypot(e.value.im);
            for b in 0..rhs{output[column*rhs+b]+=weight*input[e.row*rhs+b];}}}
        if output.iter().any(|x|!x.is_finite()){return Err("Fourier operation envelope overflow".into());}Ok(())
    }
    /// Explicit P0/proposal convention Re(T^dagger P T x). This is not an
    /// accepted physical action and must never replace its complex residual.
    pub fn pull_real_proposal_into(&self,input:&[C],rhs:usize,output:&mut[f64])->Result<ProposalProjectionDiagnostics,String>{
        self.shape(input.len(),output.len(),rhs)?;let mut full=vec![C::default();output.len()];self.pull_complex_into(input,rhs,&mut full)?;
        let mut imaginary=vec![0_f64;rhs];let mut source=vec![0_f64;rhs];for i in 0..self.dimension(){for b in 0..rhs{
            let at=i*rhs+b;output[at]=full[at].re;imaginary[b]=imaginary[b].hypot(full[at].im);source[b]=source[b].hypot(input[at].re.hypot(input[at].im));
        }}Ok(ProposalProjectionDiagnostics{imaginary_hs_by_column:imaginary,input_hs_by_column:source,real_part_defines_numerical_proposal_only:true})
    }
}
impl OwnedPrepared{
    pub fn real_fourier_map(&self,modes:&[Charge])->Result<RealFourierMap,String>{
        if modes.iter().any(|q|q.len()!=self.primitives.hilbert.source.phases.axes.len()){return Err("REAL map uses another phase coordinate system".into());}
        RealFourierMap::new(self.primitives.hilbert.source.n,modes)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn preparation_real_fourier_isometry_permuted_modes_and_arbitrary_complex(){
        let modes=[vec![0,-1],vec![1,0],vec![0,0],vec![-1,0],vec![0,1]];let map=RealFourierMap::new(3,&modes).unwrap();let rows=map.dimension();let rhs=2;
        let real:Vec<_>=(0..rows*rhs).map(|i|((i*13+7)%43)as f64/47.).collect();let mut signed=vec![C::default();real.len()];map.push_into(&real,rhs,&mut signed).unwrap();
        let mut pulled=vec![C::default();real.len()];map.pull_complex_into(&signed,rhs,&mut pulled).unwrap();
        for(i,a)in pulled.iter().enumerate(){assert!((a.re-real[i]).abs()<6e-16);assert!(a.im.abs()<3e-16);}
        for(m,q)in modes.iter().enumerate(){let partner=modes.iter().position(|p|p.iter().zip(q).all(|(a,b)|a+b==0)).unwrap();for i in 0..3{for j in 0..3{for b in 0..rhs{
            let a=signed[(m*9+i*3+j)*rhs+b];let c=conj(signed[(partner*9+j*3+i)*rhs+b]);assert_eq!(a.re,c.re);assert_eq!(a.im,c.im);
        }}}}
        let input:Vec<_>=(0..rows*rhs).map(|i|C{re:real[i],im:(i as f64*0.17).sin()}).collect();map.push_complex_into(&input,rhs,&mut signed).unwrap();map.pull_complex_into(&signed,rhs,&mut pulled).unwrap();
        for(a,b)in pulled.iter().zip(&input){assert!((a.re-b.re).hypot(a.im-b.im)<1.1e-15);}
        assert_eq!(map.harmonics()[0].offset,0);assert!(!map.harmonics()[0].has_imaginary_block);
        assert!(RealFourierMap::new(3,&[vec![0],vec![1]]).is_err());assert!(RealFourierMap::new(3,&[vec![0],vec![0]]).is_err());
    }
    #[test]
    fn preparation_real_fourier_pull_keeps_imaginary_physical_defect(){
        let map=RealFourierMap::new(2,&[vec![0]]).unwrap();let input=[C{re:0.,im:1.},C::default(),C::default(),C::default()];let mut full=vec![C::default();4];map.pull_complex_into(&input,1,&mut full).unwrap();assert_eq!(full[0].im,1.);
        let mut proposal=vec![0.;4];let diagnostic=map.pull_real_proposal_into(&input,1,&mut proposal).unwrap();assert_eq!(diagnostic.imaginary_hs_by_column[0],1.);assert!(proposal.iter().all(|&x|x==0.));
        let zero_axis=RealFourierMap::new(2,&[vec![]]).unwrap();assert_eq!(zero_axis.dimension(),4);
    }
}
