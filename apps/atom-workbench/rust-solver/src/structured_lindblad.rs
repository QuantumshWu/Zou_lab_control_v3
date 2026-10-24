//! Fixed-Hilbert-space products of the COMPLETE supplied Lindblad equation.
//!
//! For a Hermitian density-coordinate vector, one sparse off-diagonal H*rho
//! product suffices: [H,rho]=H*rho-(H*rho)^H. Diagonal energies are handled as
//! differences BEFORE multiplication, avoiding cancellation of huge common
//! optical offsets. Every collapse contribution is retained in the existing
//! builder's compact real dissipative matrix. No rate equation, elimination,
//! detuning threshold, positivity restriction, or result cache is used.
use crate::{build_generator,complex_input,Model,C};

type Sparse = Vec<Vec<(usize,f64)>>;
struct Hamiltonian {
    n:usize,
    rows:Vec<Vec<(usize,C)>>,
    gaps:Vec<f64>,
    off_diagonal:bool,
}
struct Drive { cos:Hamiltonian,sin:Hamiltonian }
pub struct Prepared {
    n:usize,
    hamiltonian:Hamiltonian,
    dissipative:Sparse,
    drives:Vec<Drive>,
}
/// Scratch only; every input and intermediate is overwritten on each call.
pub struct Workspace {
    n:usize,
    density:Vec<C>,
    product:Vec<C>,
    metric_input:Vec<f64>,
}

impl Hamiltonian {
    fn new(real:&[f64],imag:&[f64],n:usize,name:&str)->Result<Self,String>{
        let values=complex_input(real,imag,n*n,name)?;
        // The one-product identity uses represented Hermitian symmetry. Do
        // not silently symmetrize a merely tolerance-close input: a caller
        // can retain its original general sparse product instead.
        for i in 0..n{for j in 0..n{
            let a=values[i*n+j];let b=values[j*n+i];
            if a.re!=b.re||a.im!= -b.im{
                return Err(format!("{name} lacks exact represented Hermitian symmetry; retain the general generator"));
            }
        }}
        let rows=(0..n).map(|i|(0..n).filter_map(|j|{
            let value=values[i*n+j];
            (i!=j&&(value.re!=0.||value.im!=0.)).then_some((j,value))
        }).collect::<Vec<_>>()).collect::<Vec<_>>();
        let off_diagonal=rows.iter().any(|row|!row.is_empty());
        let gaps=(0..n).flat_map(|i|(i+1..n).map({let values=&values;move|j|values[i*n+i].re-values[j*n+j].re})).collect::<Vec<_>>();
        if gaps.iter().any(|value|!value.is_finite()){return Err("Non-finite Hamiltonian energy difference".into());}
        Ok(Self{n,rows,gaps,off_diagonal})
    }
    fn apply_into(&self,x:&[f64],out:&mut[f64],density:&mut[C],product:&mut[C]){
        let n=self.n;out.fill(0.);
        if self.off_diagonal{
            density.fill(C::default());product.fill(C::default());
            for i in 0..n{density[i*n+i].re=x[i];}
            let mut index=n;
            for i in 0..n{for j in i+1..n{
                let value=C::new(x[index],x[index+1]);density[i*n+j]=value;
                density[j*n+i]=value.conj();index+=2;
            }}
            for(i,entries)in self.rows.iter().enumerate(){
                let output=&mut product[i*n..(i+1)*n];
                for &(source,value)in entries{
                    let source_density=&density[source*n..(source+1)*n];
                    for(target,&rho)in output.iter_mut().zip(source_density){*target+=value*rho;}
                }
            }
            for i in 0..n{out[i]=2.*product[i*n+i].im;}
            let mut index=n;
            for i in 0..n{for j in i+1..n{
                let upper=product[i*n+j];let lower=product[j*n+i];
                out[index]=upper.im+lower.im;
                out[index+1]=lower.re-upper.re;
                index+=2;
            }}
        }
        // -i(E_i-E_j)*rho_ij, exactly the original diagonal commutator.
        for(pair,&gap)in self.gaps.iter().enumerate(){let index=n+2*pair;
            out[index]+=gap*x[index+1];out[index+1]-=gap*x[index];
        }
    }
    fn apply_transpose_into(&self,x:&[f64],out:&mut[f64],workspace:&mut Workspace){
        // Raw Hermitian coordinates have Hilbert--Schmidt metric
        // G=diag(1 on populations, 2 on Re/Im coherences). The Hamiltonian
        // generator is G-skew-adjoint, so its EUCLIDEAN transpose is
        // A_H^T=-G A_H G^-1, not simply -A_H. This is also valid for signed,
        // non-positive Hermitian inputs used by inverse-norm estimation.
        workspace.metric_input[..self.n].copy_from_slice(&x[..self.n]);
        for(input,&value)in workspace.metric_input[self.n..].iter_mut().zip(&x[self.n..]){*input=0.5*value;}
        self.apply_into(&workspace.metric_input,out,&mut workspace.density,&mut workspace.product);
        for value in &mut out[..self.n]{*value= -*value;}
        for value in &mut out[self.n..]{*value*= -2.;}
    }
}
impl Prepared {
    pub fn new(model:&Model)->Result<Self,String>{
        let n=model.dimension;if n==0||n>64{return Err("Invalid structured Lindblad Hilbert dimension".into());}
        let hamiltonian=Hamiltonian::new(&model.h_real,&model.h_imag,n,"Static structured Hamiltonian")?;
        let mut zero_h=model.clone();zero_h.h_real=vec![0.;n*n];zero_h.h_imag=vec![0.;n*n];
        zero_h.drives.clear();zero_h.time_dependent=false;
        // Same builder and identical C arrays as the reference equation,
        // including every C rho C^H and off-diagonal C^H C term.
        let dissipative=build_generator(&zero_h)?.matrix;
        if dissipative.iter().flatten().any(|(_,value)|!value.is_finite()){
            return Err("Non-finite complete dissipative operator".into());
        }
        let drives=model.drives.iter().map(|drive|{
            if !drive.omega.is_finite()||drive.omega<=0.{return Err("Invalid structured drive frequency".into());}
            Ok(Drive{
                cos:Hamiltonian::new(&drive.cos.real,&drive.cos.imag,n,"Structured cosine Hamiltonian")?,
                sin:Hamiltonian::new(&drive.sin.real,&drive.sin.imag,n,"Structured sine Hamiltonian")?,
            })
        }).collect::<Result<Vec<_>,String>>()?;
        Ok(Self{n,hamiltonian,dissipative,drives})
    }
    pub fn dimension(&self)->usize{self.n*self.n}
    pub fn drive_count(&self)->usize{self.drives.len()}
    pub fn dissipative_nonzeros(&self)->usize{self.dissipative.iter().map(Vec::len).sum()}
    pub fn workspace(&self)->Workspace{Workspace{n:self.n,density:vec![C::default();self.dimension()],product:vec![C::default();self.dimension()],metric_input:vec![0.;self.dimension()]}}
    fn validate(&self,x:&[f64],out:&[f64],workspace:&Workspace)->Result<(),String>{
        if x.len()!=self.dimension()||out.len()!=self.dimension()||workspace.n!=self.n
            ||workspace.density.len()!=self.dimension()||workspace.product.len()!=self.dimension()||workspace.metric_input.len()!=self.dimension(){
            return Err("Structured Lindblad coordinate/workspace dimensions differ".into());
        }
        if x.iter().any(|value|!value.is_finite()){return Err("Non-finite structured Lindblad input".into());}
        Ok(())
    }
    fn finite(out:&[f64])->Result<(),String>{
        if out.iter().any(|value|!value.is_finite()){Err("Non-finite structured Lindblad product".into())}else{Ok(())}
    }
    pub fn apply_static_into(&self,x:&[f64],out:&mut[f64],workspace:&mut Workspace)->Result<(),String>{
        self.validate(x,out,workspace)?;self.hamiltonian.apply_into(x,out,&mut workspace.density,&mut workspace.product);
        for(j,column)in self.dissipative.iter().enumerate(){let input=x[j];
            if input==0.{continue;}for &(i,value)in column{out[i]+=value*input;}
        }
        Self::finite(out)
    }
    pub fn apply_cos_into(&self,index:usize,x:&[f64],out:&mut[f64],workspace:&mut Workspace)->Result<(),String>{
        self.validate(x,out,workspace)?;let drive=self.drives.get(index).ok_or("Structured cosine drive index is out of range")?;
        drive.cos.apply_into(x,out,&mut workspace.density,&mut workspace.product);Self::finite(out)
    }
    pub fn apply_sin_into(&self,index:usize,x:&[f64],out:&mut[f64],workspace:&mut Workspace)->Result<(),String>{
        self.validate(x,out,workspace)?;let drive=self.drives.get(index).ok_or("Structured sine drive index is out of range")?;
        drive.sin.apply_into(x,out,&mut workspace.density,&mut workspace.product);Self::finite(out)
    }
    /// Ordinary Euclidean transpose in the original, unweighted real coordinates.
    /// Overwrites every output coordinate and reuses only scratch storage.
    pub fn apply_static_transpose_into(&self,x:&[f64],out:&mut[f64],workspace:&mut Workspace)->Result<(),String>{
        self.validate(x,out,workspace)?;self.hamiltonian.apply_transpose_into(x,out,workspace);
        // A column of the complete original dissipator is a row of D^T.
        // Keep every entry, including recycling and off-diagonal Gram terms.
        for(j,column)in self.dissipative.iter().enumerate(){
            for &(i,value)in column{out[j]+=value*x[i];}
        }
        Self::finite(out)
    }
    pub fn apply_cos_transpose_into(&self,index:usize,x:&[f64],out:&mut[f64],workspace:&mut Workspace)->Result<(),String>{
        self.validate(x,out,workspace)?;let drive=self.drives.get(index).ok_or("Structured cosine drive index is out of range")?;
        drive.cos.apply_transpose_into(x,out,workspace);Self::finite(out)
    }
    pub fn apply_sin_transpose_into(&self,index:usize,x:&[f64],out:&mut[f64],workspace:&mut Workspace)->Result<(),String>{
        self.validate(x,out,workspace)?;let drive=self.drives.get(index).ok_or("Structured sine drive index is out of range")?;
        drive.sin.apply_transpose_into(x,out,workspace);Self::finite(out)
    }
    pub fn apply_static(&self,x:&[f64])->Result<Vec<f64>,String>{
        let mut out=vec![0.;self.dimension()];let mut workspace=self.workspace();self.apply_static_into(x,&mut out,&mut workspace)?;Ok(out)
    }
    pub fn apply_cos(&self,index:usize,x:&[f64])->Result<Vec<f64>,String>{
        let mut out=vec![0.;self.dimension()];let mut workspace=self.workspace();self.apply_cos_into(index,x,&mut out,&mut workspace)?;Ok(out)
    }
    pub fn apply_sin(&self,index:usize,x:&[f64])->Result<Vec<f64>,String>{
        let mut out=vec![0.;self.dimension()];let mut workspace=self.workspace();self.apply_sin_into(index,x,&mut out,&mut workspace)?;Ok(out)
    }
    pub fn apply(&self,x:&[f64])->Result<Vec<f64>,String>{self.apply_static(x)}
    pub fn apply_into(&self,x:&[f64],out:&mut[f64],workspace:&mut Workspace)->Result<(),String>{self.apply_static_into(x,out,workspace)}
}

#[cfg(test)]
mod tests{
    use super::*;use crate::{Drive as ModelDrive,Operator};
    fn model(common:f64)->Model{
        Model{ frequency_coordinates:None,dimension:3,states:vec![],h_real:vec![common,0.7,0.2,0.7,common+0.125,-0.4,0.2,-0.4,common-0.25],
            h_imag:vec![0.,0.2,-0.3,-0.2,0.,0.1,0.3,-0.1,0.],
            collapse:vec![Operator{real:vec![0.,0.8,0.3,0.,0.,0.1,0.,0.,0.],imag:vec![0.,0.2,-0.1,0.,0.,0.3,0.,0.,0.]},
                Operator{real:vec![0.1,0.,0.,0.,-0.2,0.,0.,0.,0.3],imag:vec![0.;9]}],
            initial_index:0,initial_real:None,initial_imag:None,time_dependent:true,
            drives:vec![ModelDrive{omega:7.2,
                cos:Operator{real:vec![0.5,0.3,0.2,0.3,-0.2,0.1,0.2,0.1,0.8],imag:vec![0.,0.1,-0.2,-0.1,0.,0.3,0.2,-0.3,0.]},
                sin:Operator{real:vec![0.,-0.4,0.1,-0.4,0.,0.2,0.1,0.2,0.],imag:vec![0.,0.2,0.3,-0.2,0.,-0.1,-0.3,0.1,0.]}}]}
    }
    fn product(matrix:&Sparse,x:&[f64])->Vec<f64>{
        let mut out=vec![0.;x.len()];for(j,col)in matrix.iter().enumerate(){for &(i,a)in col{out[i]+=a*x[j];}}out
    }
    fn transpose(matrix:&Sparse)->Sparse{
        let mut out=vec![vec![];matrix.len()];
        for(j,column)in matrix.iter().enumerate(){for &(i,value)in column{out[i].push((j,value));}}out
    }
    #[test]
    fn structured_full_static_and_drives_match_every_original_basis_and_signed_input(){
        let model=model(0.);let prepared=Prepared::new(&model).unwrap();let d=prepared.dimension();
        assert_eq!(d,9);assert_eq!(prepared.drive_count(),1);
        let mut references=vec![build_generator(&model).unwrap().matrix];
        for operator in [&model.drives[0].cos,&model.drives[0].sin]{
            let mut pure=model.clone();pure.h_real=operator.real.clone();pure.h_imag=operator.imag.clone();
            pure.collapse.clear();pure.drives.clear();pure.time_dependent=false;references.push(build_generator(&pure).unwrap().matrix);
        }
        let mut inputs=(0..d).map(|j|{let mut x=vec![0.;d];x[j]=1.;x}).collect::<Vec<_>>();
        inputs.push((0..d).map(|j|(0.2+1.1*j as f64).sin()).collect());
        let mut workspace=prepared.workspace();let mut out=vec![f64::NAN;d];
        for x in inputs{for(which,reference)in references.iter().enumerate(){
            match which{0=>prepared.apply_static_into(&x,&mut out,&mut workspace),1=>prepared.apply_cos_into(0,&x,&mut out,&mut workspace),_=>prepared.apply_sin_into(0,&x,&mut out,&mut workspace)}.unwrap();
            let expected=product(reference,&x);let error=out.iter().zip(expected).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
            assert!(error<2e-14,"operator {which}, error {error:e}");
        }}
    }
    #[test]
    fn optical_common_offset_is_canceled_before_coherence_products(){
        let model=model(1e12);let prepared=Prepared::new(&model).unwrap();let reference=build_generator(&model).unwrap();
        let x:Vec<_>=(0..9).map(|j|(0.37+j as f64).cos()).collect();
        let actual=prepared.apply_static(&x).unwrap();let expected=product(&reference.matrix,&x);
        assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<2e-14));
    }
    #[test]
    fn structured_transpose_matches_original_explicit_transpose_on_every_basis(){
        // Includes complex H, complex C, off-diagonal C^H C, recycling,
        // dephasing, and signed vectors. The oracle is the unchanged full
        // generator, explicitly transposed in ordinary raw coordinates.
        for common in [0.,1e12]{
            let model=model(common);let prepared=Prepared::new(&model).unwrap();let d=prepared.dimension();
            let mut references=vec![transpose(&build_generator(&model).unwrap().matrix)];
            for operator in [&model.drives[0].cos,&model.drives[0].sin]{
                let mut pure=model.clone();pure.h_real=operator.real.clone();pure.h_imag=operator.imag.clone();
                pure.collapse.clear();pure.drives.clear();pure.time_dependent=false;
                references.push(transpose(&build_generator(&pure).unwrap().matrix));
            }
            let mut inputs=(0..d).map(|j|{let mut x=vec![0.;d];x[j]=1.;x}).collect::<Vec<_>>();
            inputs.push((0..d).map(|j|(0.13+0.73*j as f64).cos()).collect());inputs.push(vec![0.;d]);
            let mut workspace=prepared.workspace();let mut out=vec![f64::NAN;d];
            let buffers=(workspace.density.as_ptr(),workspace.product.as_ptr(),workspace.metric_input.as_ptr());
            for x in inputs{for(which,reference)in references.iter().enumerate(){
                // Interleave forward calls to ensure no previous product or
                // coordinate scaling survives into the next transpose call.
                prepared.apply_static_into(&x,&mut out,&mut workspace).unwrap();out.fill(f64::NAN);
                match which{
                    0=>prepared.apply_static_transpose_into(&x,&mut out,&mut workspace),
                    1=>prepared.apply_cos_transpose_into(0,&x,&mut out,&mut workspace),
                    _=>prepared.apply_sin_transpose_into(0,&x,&mut out,&mut workspace),
                }.unwrap();
                let expected=product(reference,&x);
                let error=out.iter().zip(expected).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
                assert!(error<2e-14,"transpose operator {which}, common {common:e}, error {error:e}");
                assert_eq!(buffers,(workspace.density.as_ptr(),workspace.product.as_ptr(),workspace.metric_input.as_ptr()));
            }}
        }
    }
    #[test]
    fn structured_transpose_diagonal_hamiltonian_and_input_validation(){
        let mut model=model(1e12);let n=model.dimension;
        for i in 0..n{for j in 0..n{if i!=j{model.h_real[i*n+j]=0.;model.h_imag[i*n+j]=0.;}}}
        let prepared=Prepared::new(&model).unwrap();let mut workspace=prepared.workspace();
        let reference=transpose(&build_generator(&model).unwrap().matrix);
        let x:Vec<_>=(0..n*n).map(|j|(0.8+j as f64).sin()).collect();let mut out=vec![f64::NAN;n*n];
        prepared.apply_static_transpose_into(&x,&mut out,&mut workspace).unwrap();
        assert!(out.iter().zip(product(&reference,&x)).all(|(a,b)|(a-b).abs()<2e-14));
        assert!(prepared.apply_static_transpose_into(&[0.;8],&mut out,&mut workspace).is_err());
        assert!(prepared.apply_cos_transpose_into(1,&x,&mut out,&mut workspace).is_err());
        assert!(prepared.apply_sin_transpose_into(1,&x,&mut out,&mut workspace).is_err());
        let mut bad=x.clone();bad[3]=f64::NAN;
        assert!(prepared.apply_static_transpose_into(&bad,&mut out,&mut workspace).is_err());
        workspace.metric_input.pop();
        assert!(prepared.apply_static_transpose_into(&x,&mut out,&mut workspace).is_err());
    }
    #[test]
    fn structured_product_rejects_unsupported_symmetry_and_wrong_workspace_without_modifying_model(){
        let mut bad=model(0.);bad.h_real[1]+=1e-12;
        assert!(Prepared::new(&bad).err().unwrap().contains("represented Hermitian"));
        let prepared=Prepared::new(&model(0.)).unwrap();let mut workspace=prepared.workspace();
        assert!(prepared.apply_static_into(&[0.;8],&mut[0.;9],&mut workspace).is_err());
        assert!(prepared.apply_cos(1,&[0.;9]).is_err());
    }
}
