//! One prepared request: the gauge-pencil operator library, the initial
//! state, the population observation duals and the P0 preconditioner factory.
//! Everything here depends only on the Model, never on a trial space.
use super::*;
use super::b0_inverse::{P0Policy,RequestP0};
use super::coordinates::{InitialCoordinates,InitialView,PhysicalCoordinates,PopulationObservation};
use super::real_fourier::RealFourierMap;

/// Defect of the REPRESENTED prepared coefficients under Hermitian-Fourier
/// conjugacy. These triangle bounds concern numerical REAL arithmetic only;
/// they are neither a forward model error nor a population-error certificate.
#[derive(Clone,Debug)]
pub(super) struct PreparedRealityBudget{pub e_norm:f64,pub e_defect:f64,pub f_base_defect:f64,
    pub s_defect:f64,pub k_defect:f64,pub nojump_defect:f64,pub recycling_defect:f64,pub gg_action_defect:f64}
fn complex_norm(a:&[C])->f64{a.iter().fold(0_f64,|x,z|x.hypot(z.re.hypot(z.im)))}
fn complex_family_norm(f:&FourierMatrices)->f64{f.values().map(|a|complex_norm(a)).sum()}
fn hilbert_pair_defect(a:&FourierMatrices,b:&FourierMatrices,n:usize,sign:f64)->f64{
    let keys:Modes=a.keys().cloned().chain(b.keys().map(|q|q.iter().map(|x|-x).collect())).collect();let mut total=0.;
    for q in keys{let other:Charge=q.iter().map(|x|-x).collect();let x=a.get(&q);let y=b.get(&other);let mut hs=0_f64;
        for i in 0..n{for j in 0..n{let x=x.map_or(C::default(),|v|v[i*n+j]);let y=y.map_or(C::default(),|v|v[j*n+i]);hs=hs.hypot((x.re-sign*y.re).hypot(x.im+sign*y.im));}}total+=hs;
    }total
}
fn recycling_pair_defect(a:&FourierMatrices,g:usize,e:usize)->f64{
    let keys:Modes=a.keys().cloned().chain(a.keys().map(|q|q.iter().map(|x|-x).collect())).collect();let mut total=0.;
    for q in keys{let other:Charge=q.iter().map(|x|-x).collect();let x=a.get(&q);let y=a.get(&other);let mut hs=0_f64;
        for ga in 0..g{for gb in 0..g{for i in 0..e{for j in 0..e{let x=x.map_or(C::default(),|v|v[(ga*g+gb)*e*e+i*e+j]);let y=y.map_or(C::default(),|v|v[(gb*g+ga)*e*e+j*e+i]);hs=hs.hypot((x.re-y.re).hypot(x.im+y.im));}}}}total+=hs;
    }total
}
impl PreparedRealityBudget{
    pub(super) fn prepare(p:&PrimitivePreparation)->Result<Self,String>{
        let n=p.hilbert.source.n;let g=p.hilbert.source.ground.len();let e=n-g;
        // ||ad S|| <=2 sum||S_q||F. Partner S is -S_-q^dagger.
        let s=2.*complex_family_norm(&p.s);let ds=2.*hilbert_pair_defect(&p.s,&p.s,n,-1.);
        let k=complex_family_norm(&p.k_hat);let dk=recycling_pair_defect(&p.k_hat,g,e);
        let r=complex_family_norm(&p.hilbert.radiation.recycling);let dr=recycling_pair_defect(&p.hilbert.radiation.recycling,g,e);
        let nj=complex_family_norm(&p.no_jump_left)+complex_family_norm(&p.no_jump_right);
        // Conjugacy swaps the executed left/right no-jump factors.
        let dn=2.*hilbert_pair_defect(&p.no_jump_left,&p.no_jump_right,n,1.);
        // L on a GG-supported matrix uses only H[:,G] and H[G,:], not the
        // irrelevant large EE carrier. This covers complete LGG plus LOG.
        let h=&p.hilbert.source.hamiltonian;let keys:Modes=h.keys().cloned().chain(h.keys().map(|q|q.iter().map(|x|-x).collect())).collect();let(mut l,mut dl)=(0.,0.);
        for q in keys{let other:Charge=q.iter().map(|x|-x).collect();let a=h.get(&q);let b=h.get(&other);let(mut columns,mut rows,mut dc,mut dr0)=(0_f64,0_f64,0_f64,0_f64);
            for i in 0..n{for j in 0..n{let x=a.map_or(C::default(),|v|v[i*n+j]);let y=b.map_or(C::default(),|v|v[j*n+i]);let delta=(x.re-y.re).hypot(x.im+y.im);
                if j<g{columns=columns.hypot(x.re.hypot(x.im));dc=dc.hypot(delta);}if i<g{rows=rows.hypot(x.re.hypot(x.im));dr0=dr0.hypot(delta);}
            }}l+=columns+rows;dl+=dc+dr0;
        }
        let j=1.+s+0.5*s*s;let dj=ds*(1.+s);
        let e_norm=(1.+k)*j;let e_defect=(1.+k)*dj+dk*j;
        let f_base_defect=(dn+dr)*j+(nj+r)*dj+dl*k*j+l*(k*dj+dk*j);
        if [s,ds,k,dk,r,dr,nj,dn,l,dl,e_norm,e_defect,f_base_defect].iter().any(|x|!x.is_finite()||*x<0.){return Err("Prepared coefficient REAL-defect budget overflow".into());}
        Ok(Self{e_norm,e_defect,f_base_defect,s_defect:ds,k_defect:dk,nojump_defect:dn,recycling_defect:dr,gg_action_defect:dl})
    }
}
pub struct RuntimeCore{
    owned:OwnedPrepared,
    physical:PhysicalCoordinates,
    initial:InitialCoordinates,
    observer:Arc<PopulationObservation>,
    p0:RequestP0,
    timing:[f64;6],
}

impl RuntimeCore{
    pub fn from_owned(owned:OwnedPrepared,initial:InitialView<'_>,p0_policy:P0Policy)->Result<Self,String>{
        let physical=owned.physical_coordinates();
        let clock=Instant::now();
        let initial=physical.initial(initial)?;
        let initial_ms=clock.elapsed().as_secs_f64()*1000.;let clock=Instant::now();
        let observer=Arc::new(physical.population_observation()?);
        let observer_ms=clock.elapsed().as_secs_f64()*1000.;let clock=Instant::now();
        let p0=owned.prepare_p0(p0_policy)?;
        let p0_ms=clock.elapsed().as_secs_f64()*1000.;
        let [gauge,primitives,library]=owned.timing;
        Ok(Self{owned,physical,initial,observer,p0,timing:[gauge,primitives,library,initial_ms,observer_ms,p0_ms]})
    }
    /// Milliseconds: gauge/source, S/K primitives, operator library, initial
    /// point, readout duals, P0 factor.
    pub fn preparation_timing(&self)->[f64;6]{self.timing}
    pub fn prepare(model:&ModelView<'_>,initial:InitialView<'_>,phases:PhaseBasis,gauge:GaugePolicy<'_>,kernel:KernelPolicy,p0:P0Policy)->Result<Self,String>{
        Self::from_owned(OwnedPrepared::prepare(model,phases,gauge,kernel)?,initial,p0)
    }
    pub fn prepared(&self)->&OwnedPrepared{&self.owned}
    pub fn initial(&self)->&InitialCoordinates{&self.initial}
    pub fn physical(&self)->&PhysicalCoordinates{&self.physical}
    pub fn observer(&self)->&Arc<PopulationObservation>{&self.observer}
    pub fn lapack(&self)->&Arc<Lapack>{&self.owned.preparation_lapack}
    pub(crate) fn p0_mut(&mut self)->&mut RequestP0{&mut self.p0}
    /// Operator library and P0 factory together, for benchmarks and tools.
    pub fn operator_context(&mut self)->(&OwnedPrepared,&mut RequestP0){(&self.owned,&mut self.p0)}
    /// REAL coordinates of the constant initial envelope on `modes`: the
    /// initial point sits on the central mode, all other modes are zero.
    pub fn initial_real(&self,map:&RealFourierMap,modes:&[Charge])->Result<Vec<f64>,String>{
        let n=map.physical_dimension();let d=n*n;let rows=map.dimension();
        let central=modes.iter().position(|q|q.iter().all(|&x|x==0)).ok_or("The domain lacks its central mode")?;
        let mut signed=vec![C::default();rows];signed[central*d..(central+1)*d].copy_from_slice(self.initial.point());
        let mut pulled=vec![C::default();rows];map.pull_complex_into(&signed,1,&mut pulled)?;
        let scale=signed.iter().map(|z|z.re.hypot(z.im)).fold(0_f64,f64::max).max(1.);
        if pulled.iter().any(|z|z.im.abs()>1e-12*scale){return Err("The initial point is not a Hermitian envelope".into());}
        Ok(pulled.iter().map(|z|z.re).collect())
    }
}
