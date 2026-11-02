//! Exact free flight: no beam acts and the model leaves the internal state as it is -- its
//! generator vanishes on the initial density -- so only the positions move, and they start
//! uniform. The populations, the Maxwellian and its temperature keep their initial values.
use crate::{Complex,Matrix,Momentum,Request,Solution,STATES};

/// k_B (J/K), exact in the SI.
const BOLTZMANN:f64=1.380649e-23;
/// Points per thermal width σ = sqrt(k_B T/m) of the reported Maxwellian, which reaches out to
/// where its density falls below the double precision of its peak. Its variance is σ² to that
/// precision: at this spacing the discrete Gaussian's aliasing is e^{−2π²·64}.
const POINTS_PER_WIDTH:f64=8.;
const METHOD:&str="Exact free flight: no beam acts and the internal state is stationary under the model, so the populations and the Maxwellian momentum distribution keep their initial values";

/// The free flight of a request, or none when a beam acts or the internal state evolves.
pub(crate) fn solve(request:&Request)->Option<Solution>{
    if !request.beams.is_empty()||!stationary(request){return None;}
    let times=request.times_us.len();
    let populations:Vec<f64>=(0..STATES).map(|i|request.initial_density[i*STATES+i].re).collect();
    let width=(BOLTZMANN*request.temperature_uk*1e-6/request.mass_kg).sqrt();
    let reach=(POINTS_PER_WIDTH*(2.*(1./f64::EPSILON).ln()).sqrt()).ceil() as i64;
    let weights:Vec<f64>=(-reach..=reach).map(|j|(-0.5*(j as f64/POINTS_PER_WIDTH).powi(2)).exp()).collect();
    let total:f64=weights.iter().sum();
    let maxwellian:Vec<f64>=weights.iter().map(|w|w/total).collect();
    let samples:Vec<usize>=if times>1{vec![0,times-1]}else{vec![0]};
    Some(Solution{populations:vec![populations;times],sample_errors:vec![0.;times],temperature_uk:vec![request.temperature_uk;times],mean_velocity_mps:vec![0.;times],
        momentum:Momentum{velocities_mps:(-reach..=reach).map(|j|width*j as f64/POINTS_PER_WIDTH).collect(),probabilities:vec![maxwellian;samples.len()],samples},
        method:METHOD.into(),reason:None})
}

/// Whether the model leaves the initial density ρ as it is: every entry of
/// −i[H0, ρ] + Σ_c (C_c ρ C_c† − ½{C_c†C_c, ρ}) is zero.
fn stationary(request:&Request)->bool{
    let rho=&request.initial_density;let h=entries(&request.hamiltonian);
    let mut generator=sum(&product(&h,rho),&product(rho,&h),-1.);
    generator=generator.iter().map(|z|Complex{re:z.im,im:-z.re}).collect();
    for c in &request.emission{
        let c=entries(c);let rate=product(&adjoint(&c),&c);
        generator=sum(&generator,&product(&product(&c,rho),&adjoint(&c)),1.);
        generator=sum(&generator,&sum(&product(&rate,rho),&product(rho,&rate),1.),-0.5);
    }
    generator.iter().all(|z|z.re==0.&&z.im==0.)
}
fn entries(m:&Matrix)->Vec<Complex>{m.real.iter().zip(&m.imag).map(|(&re,&im)|Complex{re,im}).collect()}
fn product(a:&[Complex],b:&[Complex])->Vec<Complex>{
    let mut out=vec![Complex::default();STATES*STATES];
    for i in 0..STATES{for k in 0..STATES{let x=a[i*STATES+k];
        for j in 0..STATES{let y=b[k*STATES+j];let z=&mut out[i*STATES+j];z.re+=x.re*y.re-x.im*y.im;z.im+=x.re*y.im+x.im*y.re;}}}
    out
}
fn adjoint(a:&[Complex])->Vec<Complex>{(0..STATES*STATES).map(|k|{let z=a[(k%STATES)*STATES+k/STATES];Complex{re:z.re,im:-z.im}}).collect()}
/// a + s b.
fn sum(a:&[Complex],b:&[Complex],s:f64)->Vec<Complex>{a.iter().zip(b).map(|(x,y)|Complex{re:x.re+s*y.re,im:x.im+s*y.im}).collect()}
