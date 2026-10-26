//! One prepared request: the gauge-pencil operator library, the initial
//! state, the population observation duals and the P0 preconditioner factory.
//! Everything here depends only on the Model, never on a trial space.
use super::*;
use std::collections::BTreeSet;
use super::b0_inverse::{P0Policy,RequestP0};
use super::coordinates::{InitialCoordinates,InitialView,PhysicalCoordinates,PopulationObservation};
use super::real_fourier::RealFourierMap;

/// Starting-domain steps are the modes the TRANSFORMED pencil still couples
/// to the central mode with a mode amplitude `|(F x)_q| / (|omega_q| |x|)` of
/// at least this fraction of the largest one, and at least
/// `DOMAIN_STEP_FLOOR`. A sheared coupling contributes only its residual
/// (second order, or its unsafe entries); an unsheared one its full first
/// order. This is a cost choice only: the halo estimates cover every mode
/// outside the domain, and a mode is added when its indicator asks for it.
const DOMAIN_STEP_AMPLITUDE:f64=1e-3;
const DOMAIN_STEP_FLOOR:f64=1e-4;

/// The Fourier steps of the gauged Hamiltonian: every nonzero charge (`all`,
/// the modes one coherent coupling reaches from the central mode) and the
/// modes the transformed pencil couples significantly to it (`domain`).
pub struct FourierSteps{pub axes:usize,pub all:Vec<Charge>,pub domain:Vec<Charge>}

/// The prepared operators, initial point and readout duals are immutable and
/// shared by every `fork`; the P0 factor caches its shift factors per core.
pub struct RuntimeCore{
    owned:Arc<OwnedPrepared>,
    physical:PhysicalCoordinates,
    initial:Arc<InitialCoordinates>,
    observer:Arc<PopulationObservation>,
    p0:RequestP0,
    p0_policy:P0Policy,
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
        Ok(Self{owned:Arc::new(owned),physical,initial:Arc::new(initial),observer,p0,p0_policy,
            timing:[gauge,primitives,library,initial_ms,observer_ms,p0_ms]})
    }
    /// A second core of the same request: it shares the prepared operators,
    /// the initial point and the readout duals, and has its own P0 factor
    /// (trial spaces add shift factors to it), so two solves of one request
    /// can run concurrently. Costs one P0 preparation.
    pub fn fork(&self)->Result<Self,String>{
        let clock=Instant::now();
        let p0=self.owned.prepare_p0(self.p0_policy)?;
        let mut timing=self.timing;timing[5]=clock.elapsed().as_secs_f64()*1000.;
        Ok(Self{owned:Arc::clone(&self.owned),physical:self.owned.physical_coordinates(),initial:Arc::clone(&self.initial),
            observer:Arc::clone(&self.observer),p0,p0_policy:self.p0_policy,timing})
    }
    /// Milliseconds: gauge/source, S/K primitives, operator library, initial
    /// point, readout duals, P0 factor.
    pub fn preparation_timing(&self)->[f64;6]{self.timing}
    pub fn prepare(model:&ModelView<'_>,initial:InitialView<'_>,phases:PhaseBasis,gauge:GaugePolicy<'_>,kernel:KernelPolicy,p0:P0Policy)->Result<Self,String>{
        Self::from_owned(OwnedPrepared::prepare(model,phases,gauge,kernel)?,initial,p0)
    }
    pub fn prepared(&self)->&OwnedPrepared{&self.owned}
    pub fn fourier_steps(&self)->Result<FourierSteps,String>{
        let source=&self.owned.primitives().hilbert.source;let axes=source.phases.axes.len();let zero=vec![0;axes];
        let all:Vec<Charge>=source.hamiltonian.keys().filter(|q|q.iter().any(|&k|k!=0)).cloned().collect();
        // Probe the central mode with slow content: the initial point, the
        // identity and a fixed dense Hermitian matrix whose coherence (i, j)
        // is weighted by Gamma / (Gamma + |E_i - E_j|) (a coherence much
        // faster than the linewidth is not populated by the dynamics). Read
        // the first-order envelope amplitude |(F x)_q| / (|omega_q| |x|) of
        // every reachable mode.
        let n=source.n;let d=n*n;let probes=3;
        // Candidate modes: every Hamiltonian charge (first order) and every
        // sum of two (second-order and cross couplings).
        let steps:BTreeSet<Charge>=source.hamiltonian.keys().filter(|q|q.iter().any(|&k|k!=0)).cloned().collect();
        let mut candidates=steps.clone();
        for a in &steps{for b in &steps{let sum:Charge=a.iter().zip(b).map(|(x,y)|x+y).collect();if sum.iter().any(|&k|k!=0){candidates.insert(sum);}}}
        let outputs:Vec<Charge>=candidates.into_iter().collect();
        if outputs.is_empty(){return Ok(FourierSteps{axes,all,domain:Vec::new()});}
        let h0=&source.hamiltonian[&zero];let gamma=self.owned.primitives().hilbert.radiation.gamma.get(&zero)
            .map_or(0_f64,|g|{let e=source.excited.len();(0..e).map(|i|g[i*e+i].re).fold(0_f64,f64::max)});
        let mut columns=vec![self.initial.point().to_vec(),(0..d).map(|k|C{re:if k/n==k%n{1.}else{0.},im:0.}).collect::<Vec<_>>(),Vec::with_capacity(d)];
        for i in 0..n{for j in 0..n{let(a,b)=(i.min(j),i.max(j));let phase=((a*31+b*17)%23)as f64*0.273;
            let weight=if i==j{1.}else{gamma/(gamma+(h0[i*n+i].re-h0[j*n+j].re).abs())};
            columns[2].push(if i==j{C{re:1.,im:0.}}else{C{re:weight*phase.cos(),im:weight*if i<j{phase.sin()}else{-phase.sin()}}});}}
        let mut input=vec![C::default();d*probes];let mut norms=[0_f64;3];
        for(b,column)in columns.iter().enumerate(){for k in 0..d{input[k*probes+b]=column[k];norms[b]=norms[b].hypot(column[k].re.hypot(column[k].im));}}
        let mut pencil=self.owned.pencil(&[zero.clone()],&outputs,probes)?;
        let(mut mass,mut force)=(vec![C::default();outputs.len()*d*probes],vec![C::default();outputs.len()*d*probes]);
        pencil.apply_into(&input,probes,&mut mass,&mut force)?;
        // Each component (i, j) of mode q responds through its own diagonal
        // rate NL_ii + NR_jj - i omega_q: a slow (Raman-resonant) mode is
        // reached far more strongly than its frequency |omega_q| suggests.
        let p=self.owned.primitives();
        let(nl,nr)=(&p.no_jump_left[&zero],&p.no_jump_right[&zero]);
        let mut amplitudes=Vec::with_capacity(outputs.len());
        for(m,q)in outputs.iter().enumerate(){
            let f=source.phases.frequency(q)?;let frequency=f.hi+f.lo;
            let mut a=0_f64;for b in 0..probes{if norms[b]==0.{continue;}
                let size=(0..d).fold(0_f64,|s,k|{let(i,j)=(k/n,k%n);let z=force[(m*d+k)*probes+b];
                    let rate=C{re:nl[i*n+i].re+nr[j*n+j].re,im:nl[i*n+i].im+nr[j*n+j].im-frequency};
                    let denominator=rate.re.hypot(rate.im);s.hypot(if denominator>0.{z.re.hypot(z.im)/denominator}else{f64::INFINITY})});
                a=a.max(size/norms[b]);}
            amplitudes.push(a);
        }
        let largest=amplitudes.iter().copied().fold(0_f64,f64::max);
        let threshold=(DOMAIN_STEP_AMPLITUDE*largest).max(DOMAIN_STEP_FLOOR);
        let domain=outputs.iter().zip(&amplitudes).filter(|&(_,&a)|a>=threshold).map(|(q,_)|q.clone()).collect();
        Ok(FourierSteps{axes,all,domain})
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
