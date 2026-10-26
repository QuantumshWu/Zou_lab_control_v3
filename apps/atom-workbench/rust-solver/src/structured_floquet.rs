//! Exact action of the EXISTING finite real Fourier operator. Layout, selected
//! indices and Hilbert--Schmidt weights are supplied by floquet::extend and
//! its caller; this module discovers no frequencies, truncation or support.
use crate::{Model,structured_lindblad::{Prepared as Physical,Workspace as PhysicalWorkspace}};
use std::{cell::RefCell,collections::{BTreeSet,HashMap}};

#[derive(Clone,Debug)]
pub struct HarmonicSpec{pub index:Vec<i32>,pub offset:usize,pub imaginary:bool}
#[derive(Clone,Debug)]
pub struct Layout{
    pub frequencies:Vec<f64>,
    pub drive_indices:Vec<Vec<i32>>,
    pub harmonics:Vec<HarmonicSpec>,
    /// Existing reachable-coordinate order, without sorting or recomputing it.
    pub indices:Vec<usize>,
    /// FULL extended-coordinate weights, not merely the selected slice.
    pub weights:Vec<f64>,
}
struct Neighbor{drive:usize,target:usize,target_imaginary:bool,source:usize,source_imaginary:bool,direction:f64,conjugate:f64}
struct Needs{cosine:Vec<usize>,sine:Vec<usize>}
struct DriveProducts{cosine:Vec<f64>,sine:Vec<f64>}
struct Workspace{input:Vec<f64>,output:Vec<f64>,drives:Vec<DriveProducts>,physical:PhysicalWorkspace}
pub struct Prepared{
    physical:Physical,layout:Layout,dimension:usize,chunks:Vec<usize>,
    rotations:Vec<(usize,f64)>,neighbors:Vec<Neighbor>,needs:Vec<Needs>,transpose_needs:Vec<Needs>,workspace:RefCell<Workspace>,
}

fn canonical(index:Vec<i32>)->Result<(Vec<i32>,f64),String>{
    if index.iter().find(|&&value|value!=0).is_some_and(|&value|value<0){
        Ok((index.into_iter().map(|value|value.checked_neg().ok_or("Fourier index negation overflow")).collect::<Result<Vec<_>,_>>()?,-1.))
    }else{Ok((index,1.))}
}

impl Prepared{
    pub fn new(model:&Model,layout:Layout)->Result<Self,String>{
        let dimension=model.dimension.checked_mul(model.dimension).ok_or("Structured Floquet physical dimension overflow")?;
        let size=layout.weights.len();let axes=layout.frequencies.len();
        if dimension==0||size==0||layout.harmonics.is_empty()||layout.indices.is_empty()||
            layout.frequencies.iter().any(|f|!f.is_finite()||*f<=0.)||
            layout.weights.iter().any(|w|!w.is_finite()||*w<=0.)||
            layout.drive_indices.len()!=model.drives.len()||
            layout.drive_indices.iter().any(|index|index.len()!=axes||index.iter().all(|&v|v==0)){
            return Err("Invalid supplied structured Floquet metadata".into());
        }
        let mut selected=vec![false;size];
        for &index in &layout.indices{
            if index>=size||selected[index]{return Err("Invalid or repeated selected Floquet coordinate".into());}
            selected[index]=true;
        }
        let mut offsets=HashMap::new();let mut occupied=vec![false;size];let mut chunks=Vec::new();let mut rotations=Vec::new();let mut zero_count=0;
        for h in &layout.harmonics{
            let zero=h.index.iter().all(|&i|i==0);
            if h.index.len()!=axes||h.imaginary==zero||canonical(h.index.clone())?.1<0.||offsets.insert(h.index.clone(),h.offset).is_some(){
                return Err("Invalid supplied canonical harmonic block".into());
            }
            if zero{zero_count+=1;if h.offset!=0{return Err("The supplied central harmonic must start at zero".into());}}
            let width=dimension.checked_mul(if h.imaginary{2}else{1}).ok_or("Fourier block size overflow")?;
            let end=h.offset.checked_add(width).ok_or("Fourier block offset overflow")?;
            if end>size||occupied[h.offset..end].iter().any(|v|*v){return Err("Overlapping or out-of-bounds supplied Fourier blocks".into());}
            occupied[h.offset..end].fill(true);chunks.push(h.offset);
            if h.imaginary{
                chunks.push(h.offset+dimension);
                // Identical order and expression to extend(); no frequency
                // relation is rediscovered or rounded here.
                let omega=h.index.iter().zip(&layout.frequencies).map(|(&i,&f)|i as f64*f).sum::<f64>();
                if !omega.is_finite(){return Err("Non-finite supplied harmonic angular frequency".into());}
                rotations.push((h.offset,omega));
            }
        }
        if zero_count!=1||occupied.iter().any(|v|!*v){return Err("Incomplete supplied real Fourier layout".into());}
        let mut neighbors=Vec::new();
        let mut needed:Vec<_>=(0..model.drives.len()).map(|_|(BTreeSet::new(),BTreeSet::new())).collect();
        let mut transpose_needed:Vec<_>=(0..model.drives.len()).map(|_|(BTreeSet::new(),BTreeSet::new())).collect();
        for h in &layout.harmonics{for(drive,step)in layout.drive_indices.iter().enumerate(){for direction in [-1,1]{
            let source=h.index.iter().zip(step).map(|(&n,&s)|if direction==1{n.checked_add(s)}else{n.checked_sub(s)})
                .collect::<Option<Vec<_>>>().ok_or("Fourier neighbor index overflow")?;
            let source_imaginary=source.iter().any(|&i|i!=0);let(key,conjugate)=canonical(source)?;
            let Some(&offset)=offsets.get(&key)else{continue;}; // Same retained lattice boundary as extend().
            needed[drive].0.insert(offset);
            if source_imaginary{
                needed[drive].1.insert(offset+dimension);
                if h.imaginary{needed[drive].0.insert(offset+dimension);}
            }
            if h.imaginary{needed[drive].1.insert(offset);}
            // Transposition reverses every EXISTING edge. Its input is the
            // original target chunk, not a newly inferred lattice neighbor.
            transpose_needed[drive].0.insert(h.offset);
            if source_imaginary{
                transpose_needed[drive].1.insert(h.offset);
                if h.imaginary{transpose_needed[drive].0.insert(h.offset+dimension);}
            }
            if h.imaginary{transpose_needed[drive].1.insert(h.offset+dimension);}
            neighbors.push(Neighbor{drive,target:h.offset,target_imaginary:h.imaginary,source:offset,source_imaginary,direction:direction as f64,conjugate});
        }}}
        let physical=Physical::new(model)?;
        if physical.dimension()!=dimension||physical.drive_count()!=model.drives.len(){return Err("Structured physical and Fourier dimensions differ".into());}
        let workspace=Workspace{input:vec![0.;size],output:vec![0.;size],physical:physical.workspace(),
            drives:(0..model.drives.len()).map(|_|DriveProducts{cosine:vec![0.;size],sine:vec![0.;size]}).collect()};
        let needs=needed.into_iter().map(|(cosine,sine)|Needs{cosine:cosine.into_iter().collect(),sine:sine.into_iter().collect()}).collect();
        let transpose_needs=transpose_needed.into_iter().map(|(cosine,sine)|Needs{cosine:cosine.into_iter().collect(),sine:sine.into_iter().collect()}).collect();
        Ok(Self{physical,layout,dimension,chunks,rotations,neighbors,needs,transpose_needs,workspace:RefCell::new(workspace)})
    }
    pub fn selected_dimension(&self)->usize{self.layout.indices.len()}

    fn shifted_side(&self,x:&[f64],out:&mut[f64],shift:f64)->Result<(),String>{
        let d=self.dimension;let mut workspace=self.workspace.borrow_mut();
        let Workspace{input,output,drives,physical}=&mut *workspace;
        input.fill(0.);output.fill(0.);
        for(local,&global)in self.layout.indices.iter().enumerate(){input[global]=x[local]/self.layout.weights[global];}
        for &offset in &self.chunks{
            self.physical.apply_static_into(&input[offset..offset+d],&mut output[offset..offset+d],physical)?;
        }
        // Each consumed Ac/As source chunk is recomputed exactly once THIS
        // product, then reused only for its multiple outgoing neighbor terms.
        for(j,values)in drives.iter_mut().enumerate(){
            for &offset in &self.needs[j].cosine{self.physical.apply_cos_into(j,&input[offset..offset+d],&mut values.cosine[offset..offset+d],physical)?;}
            for &offset in &self.needs[j].sine{self.physical.apply_sin_into(j,&input[offset..offset+d],&mut values.sine[offset..offset+d],physical)?;}
        }
        for &(offset,omega)in &self.rotations{for i in 0..d{
            output[offset+i]+=omega*input[offset+d+i];output[offset+d+i]-=omega*input[offset+i];
        }}
        for edge in &self.neighbors{
            let values=&drives[edge.drive];
            for i in 0..d{
                output[edge.target+i]+=0.5*values.cosine[edge.source+i];
                if edge.source_imaginary{output[edge.target+i]-=0.5*edge.direction*edge.conjugate*values.sine[edge.source+d+i];}
                if edge.target_imaginary{
                    if edge.source_imaginary{output[edge.target+d+i]+=0.5*edge.conjugate*values.cosine[edge.source+d+i];}
                    output[edge.target+d+i]+=0.5*edge.direction*values.sine[edge.source+i];
                }
            }
        }
        for(local,&global)in self.layout.indices.iter().enumerate(){out[local]=shift*x[local]-self.layout.weights[global]*output[global];}
        Ok(())
    }

    fn shifted_transpose_side(&self,x:&[f64],out:&mut[f64],shift:f64)->Result<(),String>{
        let d=self.dimension;let mut workspace=self.workspace.borrow_mut();
        let Workspace{input,output,drives,physical}=&mut *workspace;
        input.fill(0.);output.fill(0.);
        // A = W L W^-1 on the supplied selected coordinates, hence
        // A^T = W^-1 L^T W. Restriction still zero-fills unselected inputs.
        for(local,&global)in self.layout.indices.iter().enumerate(){input[global]=self.layout.weights[global]*x[local];}
        for &offset in &self.chunks{
            self.physical.apply_static_transpose_into(&input[offset..offset+d],&mut output[offset..offset+d],physical)?;
        }
        // Every consumed physical transpose is computed once in THIS call.
        // The same buffers may previously hold forward products; every chunk
        // read below is overwritten according to transpose_needs first.
        for(j,values)in drives.iter_mut().enumerate(){
            for &offset in &self.transpose_needs[j].cosine{self.physical.apply_cos_transpose_into(j,&input[offset..offset+d],&mut values.cosine[offset..offset+d],physical)?;}
            for &offset in &self.transpose_needs[j].sine{self.physical.apply_sin_transpose_into(j,&input[offset..offset+d],&mut values.sine[offset..offset+d],physical)?;}
        }
        for &(offset,omega)in &self.rotations{for i in 0..d{
            output[offset+i]-=omega*input[offset+d+i];output[offset+d+i]+=omega*input[offset+i];
        }}
        for edge in &self.neighbors{
            let values=&drives[edge.drive];
            for i in 0..d{
                output[edge.source+i]+=0.5*values.cosine[edge.target+i];
                if edge.source_imaginary{output[edge.source+d+i]-=0.5*edge.direction*edge.conjugate*values.sine[edge.target+i];}
                if edge.target_imaginary{
                    if edge.source_imaginary{output[edge.source+d+i]+=0.5*edge.conjugate*values.cosine[edge.target+d+i];}
                    output[edge.source+i]+=0.5*edge.direction*values.sine[edge.target+d+i];
                }
            }
        }
        for(local,&global)in self.layout.indices.iter().enumerate(){out[local]=shift*x[local]-output[global]/self.layout.weights[global];}
        Ok(())
    }

    /// The original (NOT row-equilibrated) shifted operator used by Resolvent.
    /// None: shift I-L_F on r coordinates. Some(nu): its complex shift as the
    /// real2r block [[A,-nu I],[nu I,A]]. The external complex component is
    /// independent of the Fourier real/imaginary chunks inside each r-vector.
    pub fn try_product_shifted(&self,x:&[f64],shift:f64,external_frequency:Option<f64>)->Result<Vec<f64>,String>{
        let r=self.selected_dimension();let size=r.checked_mul(if external_frequency.is_some(){2}else{1}).ok_or("Augmented product size overflow")?;
        if x.len()!=size||!shift.is_finite()||external_frequency.is_some_and(|f|!f.is_finite())||x.iter().any(|v|!v.is_finite()){
            return Err("Invalid structured shifted Fourier product input".into());
        }
        let mut result=vec![0.;size];self.shifted_side(&x[..r],&mut result[..r],shift)?;
        if let Some(frequency)=external_frequency{
            self.shifted_side(&x[r..],&mut result[r..],shift)?;
            for i in 0..r{result[i]-=frequency*x[r+i];result[r+i]+=frequency*x[i];}
        }
        if result.iter().any(|v|!v.is_finite()){return Err("Non-finite structured shifted Fourier product".into());}
        Ok(result)
    }
    /// Infallible callback adapter. Invalid numeric data produce a NaN vector,
    /// which the GMRES and mandatory explicit-original residual gates reject;
    /// no incomplete or reduced physical product is substituted on failure.
    pub fn product_shifted(&self,x:&[f64],shift:f64,external_frequency:Option<f64>)->Vec<f64>{
        self.try_product_shifted(x,shift,external_frequency).unwrap_or_else(|_|vec![f64::NAN;x.len()])
    }

    /// Euclidean transpose of the SAME original weighted shifted operator.
    /// The augmented complex pole reverses its external cross-block signs:
    /// [[A,-nu I],[nu I,A]]^T = [[A^T,+nu I],[-nu I,A^T]].
    pub fn try_product_shifted_transpose(&self,x:&[f64],shift:f64,external_frequency:Option<f64>)->Result<Vec<f64>,String>{
        let r=self.selected_dimension();let size=r.checked_mul(if external_frequency.is_some(){2}else{1}).ok_or("Augmented transpose product size overflow")?;
        if x.len()!=size||!shift.is_finite()||external_frequency.is_some_and(|f|!f.is_finite())||x.iter().any(|v|!v.is_finite()){
            return Err("Invalid structured shifted Fourier transpose product input".into());
        }
        let mut result=vec![0.;size];self.shifted_transpose_side(&x[..r],&mut result[..r],shift)?;
        if let Some(frequency)=external_frequency{
            self.shifted_transpose_side(&x[r..],&mut result[r..],shift)?;
            for i in 0..r{result[i]+=frequency*x[r+i];result[r+i]-=frequency*x[i];}
        }
        if result.iter().any(|v|!v.is_finite()){return Err("Non-finite structured shifted Fourier transpose product".into());}
        Ok(result)
    }
    /// Transpose callback adapter with the same explicit-residual fallback
    /// contract as product_shifted; no forward substitute is used on error.
    pub fn product_shifted_transpose(&self,x:&[f64],shift:f64,external_frequency:Option<f64>)->Vec<f64>{
        self.try_product_shifted_transpose(x,shift,external_frequency).unwrap_or_else(|_|vec![f64::NAN;x.len()])
    }
}
