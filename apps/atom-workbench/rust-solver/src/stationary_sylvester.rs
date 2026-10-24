//! Optional stationary PRECONDITIONER, never a replacement stationary equation.
//!
//! Each numerical block is L0-alpha I-i omega, alpha>0. Every Hamiltonian and
//! collapse entry stays in L0. A Sherman-Morrison row replacement gives the
//! central block the same normalization row as the full stationary system.
//! The caller must retain the original full Fourier forward/transpose products,
//! residual, condition and density checks. Alpha changes only this inverse
//! device; it is not a physical decay or an assertion of a unique dark state.
use crate::{Lapack,Model};
use crate::sylvester_recycling::HarmonicBlocks;

pub struct Prepared {
    blocks:HarmonicBlocks,
    trace_row:usize,
    trace_coordinates:Vec<usize>,
    inverse_unit:Vec<f64>,
    transpose_correction:Vec<f64>,
    denominator:f64,
}

impl Prepared {
    /// Input/output are ORIGINAL unweighted real Hermitian Fourier coordinates.
    /// `indices[local]` maps a retained coordinate to the full Fourier layout.
    /// The unique central block starts at zero; trace_row is a LOCAL index of
    /// one central population coordinate. Row equilibration belongs to caller.
    pub fn new(
        lap:&Lapack,model:&Model,alpha:f64,harmonics:&[(usize,f64,bool)],
        extended_dimension:usize,indices:&[usize],trace_row:usize,
    )->Result<Self,String>{
        let d=model.dimension.checked_mul(model.dimension).ok_or("Stationary Sylvester dimension overflow")?;
        if !alpha.is_finite()||alpha<=0.||d==0||extended_dimension<d||indices.is_empty()
            ||trace_row>=indices.len()||indices[trace_row]>=model.dimension{
            return Err("Invalid stationary Sylvester normalization or regularizer".into());
        }
        let mut covered=vec![false;extended_dimension];let mut central=0;
        for &(offset,frequency,sine)in harmonics{
            let width=d.checked_mul(if sine{2}else{1}).ok_or("Stationary Sylvester block overflow")?;
            let end=offset.checked_add(width).ok_or("Stationary Sylvester block overflow")?;
            if end>extended_dimension||!frequency.is_finite()||(!sine&&(offset!=0||frequency!=0.)){
                return Err("Invalid stationary Sylvester harmonic layout".into());
            }
            if !sine{central+=1;}
            for flag in &mut covered[offset..end]{if *flag{return Err("Overlapping stationary Sylvester blocks".into());}*flag=true;}
        }
        if central!=1||covered.iter().any(|v|!*v){return Err("Incomplete stationary Sylvester harmonic layout".into());}
        let mut seen=vec![false;extended_dimension];
        for &index in indices{if index>=extended_dimension||seen[index]{return Err("Invalid stationary Sylvester support mapping".into());}seen[index]=true;}
        let trace_coordinates:Vec<_>=indices.iter().enumerate().filter_map(|(local,&global)|(global<model.dimension).then_some(local)).collect();
        let blocks=HarmonicBlocks::new(lap,model,alpha,harmonics,d,indices,&vec![1.;extended_dimension])?;
        let mut unit=vec![0.;indices.len()];unit[trace_row]=1.;
        let mut inverse_unit=blocks.apply_preconditioner(lap,&unit)?;
        for value in &mut inverse_unit{*value= -*value;}
        let denominator=trace_coordinates.iter().map(|&i|inverse_unit[i]).sum::<f64>();
        let scale=inverse_unit.iter().map(|v|v.abs()).sum::<f64>();
        if !denominator.is_finite()||denominator.abs()<=64.*f64::EPSILON*scale.max(1e-300){
            return Err("Stationary Sylvester normalization correction is singular".into());
        }
        let mut trace=vec![0.;indices.len()];for &i in &trace_coordinates{trace[i]=1.;}
        // K=B^-1=-R, u=K e_t, P=B+e_t(c^T-e_t^T B).
        // P^-1=K+u(e_t^T-c^T K)/(c^T u).
        // P^-T=K^T+(e_t-K^T c)u^T/(c^T u).
        let mut transpose_correction=blocks.apply_preconditioner_transpose(lap,&trace)?;
        transpose_correction[trace_row]+=1.;
        Ok(Self{blocks,trace_row,trace_coordinates,inverse_unit,transpose_correction,denominator})
    }

    pub fn dimension(&self)->usize{self.inverse_unit.len()}

    pub fn apply(&self,lap:&Lapack,rhs:&[f64],transpose:bool)->Result<Vec<f64>,String>{
        if rhs.len()!=self.dimension()||rhs.iter().any(|v|!v.is_finite()){
            return Err("Invalid stationary Sylvester RHS".into());
        }
        let mut result=if transpose{self.blocks.apply_preconditioner_transpose(lap,rhs)?}
            else{self.blocks.apply_preconditioner(lap,rhs)?};
        for value in &mut result{*value= -*value;}
        if transpose{
            let amount=self.inverse_unit.iter().zip(rhs).map(|(u,b)|u*b).sum::<f64>()/self.denominator;
            for(value,correction)in result.iter_mut().zip(&self.transpose_correction){*value+=amount*correction;}
        }else{
            let trace=self.trace_coordinates.iter().map(|&i|result[i]).sum::<f64>();
            let amount=(rhs[self.trace_row]-trace)/self.denominator;
            for(value,u)in result.iter_mut().zip(&self.inverse_unit){*value+=amount*u;}
        }
        if result.iter().any(|v|!v.is_finite()){return Err("Non-finite stationary Sylvester inverse".into());}
        Ok(result)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::{Operator,build_generator};

    fn model(n:usize)->Model{
        let (h_real,h_imag,collapse)=if n==2{
            (vec![0.2,0.6,0.6,1.8],vec![0.,0.23,-0.23,0.],vec![
                Operator{real:vec![0.,0.7,0.,0.],imag:vec![0.,0.2,0.,0.]},
                Operator{real:vec![0.1,0.,0.,-0.1],imag:vec![0.;4]},
            ])
        }else{
            (vec![0.,0.3,0.2,0.3,1.2,-0.4,0.2,-0.4,3.4],vec![0.,0.1,-0.2,-0.1,0.,0.25,0.2,-0.25,0.],vec![
                // Shared target gives an off-diagonal, complex C^H C. The
                // transpose oracle would expose an incorrect C <-> C^H bath.
                Operator{real:vec![0.,0.4,0.7,0.,0.,0.,0.,0.,0.],imag:vec![0.,0.2,-0.3,0.,0.,0.,0.,0.,0.]},
                Operator{real:vec![0.,0.,0.,0.,0.,0.2,0.,0.,0.],imag:vec![0.;9]},
            ])
        };
        Model{frequency_coordinates:None,dimension:n,states:vec![],h_real,h_imag,collapse,
            initial_index:0,initial_real:None,initial_imag:None,time_dependent:false,drives:vec![]}
    }

    fn dense_numerical_matrix(model:&Model,alpha:f64,blocks:&[(usize,f64,bool)],size:usize,trace_row:usize)->Vec<f64>{
        let g=build_generator(model).unwrap();let d=model.dimension*model.dimension;let mut result=vec![0.;size*size];
        for &(offset,frequency,sine)in blocks{
            for part in 0..if sine{2}else{1}{
                let base=offset+part*d;
                for(j,column)in g.matrix.iter().enumerate(){for &(i,value)in column{result[base+i+(base+j)*size]+=value;}}
                for i in 0..d{result[base+i+(base+i)*size]-=alpha;}
            }
            if sine{for i in 0..d{
                result[offset+i+(offset+d+i)*size]+=frequency;
                result[offset+d+i+(offset+i)*size]-=frequency;
            }}
        }
        for column in 0..size{result[trace_row+column*size]=if column<model.dimension{1.}else{0.};}
        result
    }

    #[test]
    fn stationary_regularized_blocks_match_complete_two_and_three_level_dense_oracles(){
        let lap=Lapack::load().unwrap();
        for n in [2,3]{
            let model=model(n);let d=n*n;let size=5*d;let blocks=[(0,0.,false),(d,1.7,true),(3*d,-2.3,true)];
            for alpha in [1e-3,0.7]{
                let trace_row=1;let indices:Vec<_>=(0..size).collect();
                let prepared=Prepared::new(&lap,&model,alpha,&blocks,size,&indices,trace_row).unwrap();
                let matrix=dense_numerical_matrix(&model,alpha,&blocks,size,trace_row);
                let(lu,pivots)=lap.factor(&matrix,size).unwrap();
                for transpose in [false,true]{
                    let expected:Vec<_>=(0..size).map(|i|(0.2+i as f64).sin()).collect();
                    let rhs:Vec<_>=(0..size).map(|i|(0..size).map(|j|matrix[if transpose{j+i*size}else{i+j*size}]*expected[j]).sum()).collect();
                    let result=prepared.apply(&lap,&rhs,transpose).unwrap();
                    let dense=if transpose{lap.solve_factored_transpose(&lu,&pivots,&rhs,size)}else{lap.solve_factored(&lu,&pivots,&rhs,size)}.unwrap();
                    let error=result.iter().zip(&dense).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
                    assert!(error<2e-8,"n={n} alpha={alpha} transpose={transpose} error={error:e}");
                    assert!(result.iter().zip(&expected).all(|(a,b)|(a-b).abs()<2e-8));
                    if !transpose{assert!((result[..n].iter().sum::<f64>()-rhs[trace_row]).abs()<2e-11);}
                }
            }
        }
    }

    #[test]
    fn stationary_regularizer_keeps_a_dark_static_nullspace_out_of_the_physical_equation(){
        let lap=Lapack::load().unwrap();let mut model=model(2);model.h_real.fill(0.);model.h_imag.fill(0.);model.collapse.clear();
        let original=build_generator(&model).unwrap();assert!(original.matrix.iter().flatten().all(|(_,v)|*v==0.));
        let blocks=[(0,0.,false)];let indices=[0,1,2,3];let prepared=Prepared::new(&lap,&model,1e-3,&blocks,4,&indices,0).unwrap();
        let matrix=dense_numerical_matrix(&model,1e-3,&blocks,4,0);let(lu,pivots)=lap.factor(&matrix,4).unwrap();
        let rhs=[0.3,-0.4,0.5,-0.6];
        for transpose in [false,true]{
            let actual=prepared.apply(&lap,&rhs,transpose).unwrap();
            let expected=if transpose{lap.solve_factored_transpose(&lu,&pivots,&rhs,4)}else{lap.solve_factored(&lu,&pivots,&rhs,4)}.unwrap();
            assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-9));
        }
        assert!(Prepared::new(&lap,&model,0.,&blocks,4,&indices,0).is_err());
        // The constructor only borrows the original model; its zero physical
        // generator remains zero, so this inverse does not select a mean state.
        assert!(build_generator(&model).unwrap().matrix.iter().flatten().all(|(_,v)|*v==0.));
    }
}
