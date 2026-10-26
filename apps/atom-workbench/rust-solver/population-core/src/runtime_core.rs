//! One prepared request: the gauge-pencil operator library, the initial
//! state, the population observation duals and the P0 preconditioner factory.
//! Everything here depends only on the Model, never on a trial space.
use super::*;
use super::b0_inverse::{P0Policy,RequestP0};
use super::coordinates::{InitialCoordinates,InitialView,PhysicalCoordinates,PopulationObservation};
use super::real_fourier::RealFourierMap;

/// Fourier steps whose first-order mixing amplitude `|H_q| / |omega_q|` is
/// below this fraction of the largest one do not shape starting domains. This
/// is a cost choice only: the halo estimates cover every step, and a mode
/// reached through a weak step is added when its indicator asks for it.
const DOMAIN_STEP_AMPLITUDE:f64=1e-3;

/// The Fourier steps of the gauged Hamiltonian: every nonzero charge (`all`,
/// the modes one coherent coupling reaches from the central mode) and those
/// with a significant mixing amplitude (`domain`).
pub struct FourierSteps{pub axes:usize,pub all:Vec<Charge>,pub domain:Vec<Charge>}

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
    pub fn fourier_steps(&self)->Result<FourierSteps,String>{
        let source=&self.owned.primitives().hilbert.source;
        let(mut all,mut amplitudes)=(Vec::new(),Vec::new());
        for(q,matrix)in &source.hamiltonian{
            if q.iter().all(|&k|k==0){continue;}
            let coupling=matrix.iter().map(|z|z.re.hypot(z.im)).fold(0_f64,f64::max);
            let frequency=source.phases.frequency(q).map(|f|(f.hi+f.lo).abs())?;
            amplitudes.push(if frequency>0.{coupling/frequency}else{f64::INFINITY});
            all.push(q.clone());
        }
        let largest=amplitudes.iter().copied().fold(0_f64,f64::max);
        let domain=all.iter().zip(&amplitudes).filter(|&(_,&a)|a>=DOMAIN_STEP_AMPLITUDE*largest).map(|(q,_)|q.clone()).collect();
        Ok(FourierSteps{axes:source.phases.axes.len(),all,domain})
    }
    pub fn initial(&self)->&InitialCoordinates{&self.initial}
    pub fn physical(&self)->&PhysicalCoordinates{&self.physical}
    pub fn observer(&self)->&Arc<PopulationObservation>{&self.observer}
    pub fn lapack(&self)->&Arc<Lapack>{&self.owned.preparation_lapack}
    pub(crate) fn p0_mut(&mut self)->&mut RequestP0{&mut self.p0}
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
