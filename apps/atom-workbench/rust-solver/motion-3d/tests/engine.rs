//! The 3D engine's checks on the app's tweezer PGC example (tests/fixtures/tweezer_pgc.json: its
//! motion block along x, the frame y, z, x): the generator's consistency at random points, the
//! Ehrenfest force against the energy's gradient, the dipole pattern's moments, the trap, a
//! thermal state that stays thermal with the light off (its T over a trap period as well, with
//! smaller errors), a group's Latin hypercube of initial momenta, and a short solve whose t = 0 is
//! the request's initial state.
use atom_motion_3d::engine::{dipole_direction,Engine,Settings};
use atom_motion_3d::model::Model;
use atom_motion_3d::rng::{normal_quantile,Rng};
use atom_motion_3d::stats::{summarize,Scales,KT_PER_UK};
use atom_motion_3d::{initial_trajectory,simulate,Initial,Stratum};
use atom_motion_3d::solve::solve;
use atom_motion_3d::{Beam,Complex,Matrix,Request,STATES};
use serde_json::Value;

fn fixture()->Value{serde_json::from_str(include_str!("fixtures/tweezer_pgc.json")).unwrap()}
fn matrix(v:&Value)->Matrix{
    let mut m=Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]};
    for e in v.as_array().unwrap(){let(i,j)=(e[0].as_u64().unwrap() as usize,e[1].as_u64().unwrap() as usize);m.real[i*STATES+j]=e[2].as_f64().unwrap();m.imag[i*STATES+j]=e[3].as_f64().unwrap();}
    m
}
fn floats(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
/// The fixture's request, its beams kept by `keep`.
fn request(keep:&dyn Fn(&Value)->bool)->Request{
    let r=&fixture()["request"];let states:Vec<String>=r["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect();
    let initial=states.iter().position(|s|s==r["initialStateId"].as_str().unwrap()).unwrap();
    let mut density=vec![Complex::ZERO;STATES*STATES];density[initial*STATES+initial]=Complex::ONE;
    let beams=r["beams"].as_array().unwrap().iter().filter(|b|keep(b)).map(|b|{let d=floats(&b["direction"]);Beam{id:b["id"].as_str().unwrap().into(),name:b["name"].as_str().unwrap().into(),
        direction:[d[0],d[1],d[2]],wave_number_per_um:b["waveNumberPerUm"].as_f64().unwrap(),waist_um:b["waistUm"].as_f64(),intensity_mw_cm2:b["intensityMWcm2"].as_f64().unwrap(),
        frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),coupling:matrix(&b["coupling"]),
        counter_coupling:matrix(&b["counterCoupling"]),outside_shift:matrix(&b["outsideShift"])}}).collect::<Vec<Beam>>();
    let trap_beam=beams.iter().position(|b|b.waist_um.is_some()).unwrap_or(0);
    let e=r["emission"].as_array().unwrap();
    Request{states,hamiltonian:matrix(&r["hamiltonian"]),frame_frequencies_mhz:floats(&r["frameFrequenciesMHz"]),frequencies_mhz:floats(&r["frequenciesMHz"]),beams,
        emission:[matrix(&e[0]),matrix(&e[1]),matrix(&e[2])],field_vector_g:[0.;3],mass_kg:r["massKg"].as_f64().unwrap(),initial_density:density,
        temperature_uk:r["temperatureUK"].as_f64().unwrap(),trap_beam,times_us:vec![]}
}
/// A reproducible uniform stream in [−½, ½).
fn uniform(seed:u64)->impl FnMut()->f64{let mut rng=Rng::new(seed,0);move||rng.uniform()-0.5}
/// Random points within the trap and normalized states of each block.
fn samples(model:&Model,seed:u64)->Vec<(usize,[f64;3],[f64;3],[Complex;5])>{
    let mut next=uniform(seed);
    (0..12).map(|i|{let block=i%model.blocks.len();let n=model.dimension(block);
        let r=[0.3*next(),0.8*next(),0.3*next()];let kappa=[60.*next(),60.*next(),60.*next()];
        let mut psi=[Complex::ZERO;5];for z in psi[..n].iter_mut(){*z=Complex::new(next(),next());}
        let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        (block,r,kappa,psi)}).collect()
}

/// At any point the photons carry the whole decay of the norm: ⟨ψ|Γ|ψ⟩ (the anti-Hermitian part of
/// H_eff) is Σ_c Σ_a ‖C_a R_c ψ‖² over every ground state (the kinetic shift off, so both are the
/// responses at rest).
#[test]
fn the_photons_carry_the_whole_decay(){
    let mut model=Model::new(&request(&|_|true)).unwrap();model.doppler=false;
    for(block,r,kappa,psi)in samples(&model,1){
        let n=model.dimension(block);let p=model.point(r,kappa);let local=model.local(block,&p);
        let mut decay=0.;for a in 0..n{for b in 0..n{decay+=(psi[a].conj()*local.hamiltonian[a*n+b]*psi[b]).im;}}let decay=-2.*decay;
        let photons=model.photons(block,&p,&psi);
        let total:f64=photons.amplitude.iter().map(|c|c.iter().map(|axis|axis.iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>()).sum();
        assert!((decay-total).abs()<=1e-6*total.max(1e-12),"block {block} at {r:?}: {decay:e} against {total:e}");
    }
}

/// The Ehrenfest force is minus the gradient of the energy ⟨ψ|H_herm|ψ⟩ (central differences),
/// the trap's and the light's, the Doppler term's own gradient aside (the kinetic shift off).
#[test]
fn the_force_is_the_energy_s_gradient(){
    let mut model=Model::new(&request(&|_|true)).unwrap();model.doppler=false;
    let energy=|block:usize,r:[f64;3],psi:&[Complex;5]|->f64{
        let n=model.dimension(block);let local=model.local(block,&model.point(r,[0.;3]));let mut e=local.scalar;
        for a in 0..n{for b in 0..n{e+=(psi[a].conj()*local.hamiltonian[a*n+b]*psi[b]).re;}}e
    };
    for(block,r,_,psi)in samples(&model,2){
        let p=model.point(r,[0.;3]);let local=model.local(block,&p);let force=model.force(block,&p,&local,&psi);let h=1e-5;
        for a in 0..3{
            let(mut plus,mut minus)=(r,r);plus[a]+=h;minus[a]-=h;let numeric=-(energy(block,plus,&psi)-energy(block,minus,&psi))/(2.*h);
            assert!((force[a]-numeric).abs()<=1e-5*numeric.abs().max(1.),"block {block} at {r:?}, axis {a}: {} against {numeric}",force[a]);
        }
    }
}

/// The photon's direction follows the dipole pattern: ⟨n_i n_j⟩ = ((2/5) δ_ij tr G − (1/5) Re G_ij)/tr G
/// for a σ dipole about z with a linear admixture (a Hermitian G of rank 2).
#[test]
fn the_emission_follows_the_dipole_pattern(){
    let u=[[Complex::new(0.6,0.),Complex::new(0.,0.2)],[Complex::new(0.,0.6),Complex::new(0.1,0.)],[Complex::new(0.3,0.1),Complex::new(0.,0.)]];
    let gram:[[Complex;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|u[a].iter().zip(&u[b]).fold(Complex::ZERO,|s,(x,y)|s+x.conj()**y)));
    let trace=gram[0][0].re+gram[1][1].re+gram[2][2].re;let mut rng=Rng::new(7,1);let count=400_000;let mut second=[[0.;3];3];
    for _ in 0..count{let n=dipole_direction(&mut rng,&gram);for i in 0..3{for j in 0..3{second[i][j]+=n[i]*n[j]/count as f64;}}}
    for i in 0..3{for j in 0..3{
        let expected=((if i==j{0.4*trace}else{0.})-0.2*gram[i][j].re)/trace;
        assert!((second[i][j]-expected).abs()<=4e-3,"<n{i} n{j}> {} against {expected}",second[i][j]);
    }}
}

/// The trap: the tweezer's frequencies at the focus, 100.4 kHz across it and 19.3 kHz along it (the
/// pipeline study's 100.55 kHz radial with the tweezer alone; the cooling light's own trap lowers
/// it), the same with the tweezer as the only light -- then the frame's carrier, a near beam, not
/// a fold.
#[test]
fn the_trap_is_the_tweezer_s(){
    let khz=|model:&Model|{let initial=Initial::new(model,&request(&|_|true)).unwrap();let(b,x,_)=initial.states[0];
        model.trap_frequencies(b,x).map(|w|w/std::f64::consts::TAU*1e3)};
    let all=khz(&Model::new(&request(&|_|true)).unwrap());
    assert!((all[0]-100.41).abs()<0.05&&(all[2]-100.41).abs()<0.05&&(all[1]-19.26).abs()<0.05,"{all:?}");
    let alone=Model::new(&request(&|b|b["waistUm"].is_number())).unwrap();
    let only=khz(&alone);assert!((only[0]-100.55).abs()<0.05&&(only[1]-19.28).abs()<0.05,"{only:?}");
}

/// With the light off the thermal state stays thermal: T on every axis keeps its initial value (no
/// jumps), within three standard errors at every sample over 200 us.
#[test]
fn a_thermal_state_stays_thermal_with_the_light_off(){
    let r=request(&|b|b["waistUm"].is_number());let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let(b,x,_)=initial.states[0];let frequencies=model.trap_frequencies(b,x);assert_eq!(model.focused.len(),1);
    let settings=Settings{dt:0.25,free:[true;3],escape_intensity:1./20.,frequencies,truncation:f64::EPSILON,average:false};
    let outcome=simulate(&model,&settings,&initial,5,0,1024,8,40,21,&[]);assert_eq!(outcome.jumps,0);
    let e=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies});
    for a in 0..3{let start=e[0].temperature[a];for x in &e{
        assert!((x.temperature[a]-start).abs()<=3.*(x.temperature_error[a].hypot(e[0].temperature_error[a])),"axis {a}: {} against {start}",x.temperature[a]);
        assert!((x.temperature[a]-r.temperature_uk).abs()<=4.*x.temperature_error[a],"axis {a}: {} against {}",x.temperature[a],r.temperature_uk);
    }}
    let _=(Engine{model:&model,settings},initial_trajectory);
}

#[test]
fn the_normal_quantile_is_the_inverse_distribution(){
    // Reference quantiles of the standard normal, and its symmetry.
    for(p,x)in [(0.5,0.),(0.975,1.959963984540054),(0.99,2.3263478740408408),(0.999,3.090232306167813),(1e-10,-6.361340902404056)]{
        assert!((normal_quantile(p)-x).abs()<=1e-13*x.abs().max(1.),"{p}: {} against {x}",normal_quantile(p));
        // 1 − p is exact in double precision for these p but the deepest tail's.
        if p>=1e-3{assert!((normal_quantile(1.-p)+normal_quantile(p)).abs()<=1e-13*x.abs().max(1.),"{p}");}
    }
}

#[test]
fn a_group_s_initial_momenta_take_one_stratum_each(){
    // In a group of eight, each momentum component of the members lies in another eighth of the
    // Maxwell distribution (its strata bounded by the normal's quantiles at k/8).
    let r=request(&|_|true);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let(b,x,_)=initial.states[0];let frequencies=model.trap_frequencies(b,x);
    let settings=Settings{dt:0.25,free:[true;3],escape_intensity:1./20.,frequencies,truncation:f64::EPSILON,average:false};
    let(size,kt,hm)=(8_u64,KT_PER_UK*r.temperature_uk,model.hbar_over_mass);
    for first in [0,8,96]{
        let group:Vec<[f64;3]>=(first..first+size).map(|i|initial_trajectory(&model,&settings,&initial,3,i,Stratum{first,size}).kappa).collect();
        for a in 0..3{
            let half=frequencies[a]/(2.*kt);let width=(kt/hm).sqrt()*(half/half.tanh()).sqrt();
            let mut strata:Vec<usize>=group.iter().map(|k|(1..size as usize).filter(|&j|k[a]/width>=normal_quantile(j as f64/size as f64)).count()).collect();
            strata.sort();assert_eq!(strata,(0..size as usize).collect::<Vec<_>>(),"group {first}, axis {a}");
        }
    }
}

#[test]
fn a_thermal_state_s_temperature_over_a_trap_period_is_its_temperature_with_smaller_errors(){
    // The light off: T over one trap period of each axis against T at the time, the same atoms.
    let r=request(&|b|b["waistUm"].is_number());let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let(b,x,_)=initial.states[0];let frequencies=model.trap_frequencies(b,x);
    let settings=|average:bool|Settings{dt:0.25,free:[true;3],escape_intensity:1./20.,frequencies,truncation:f64::EPSILON,average};
    let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies};
    let run=|average:bool|summarize(&simulate(&model,&settings(average),&initial,5,0,1024,1,40,21,&[]).groups,&scales);
    let(at,over)=(run(false),run(true));let mut ratio=0.;
    for i in 1..at.len(){for a in 0..3{
        let(x,y)=(&at[i],&over[i]);
        assert!((x.temperature[a]-y.temperature[a]).abs()<=4.*x.temperature_error[a],"output {i}, axis {a}: {} against {}",y.temperature[a],x.temperature[a]);
        ratio+=y.temperature_error[a]/x.temperature_error[a]/(3*(at.len()-1)) as f64;
    }}
    // An oscillator's mean kinetic energy over a period is half its energy: the variance of T halves.
    assert!(ratio<0.85,"standard errors over a period / at the time: {ratio:.2}");
}

#[test]
fn a_solve_starts_from_the_request_s_initial_state(){
    // Ten microseconds of the tweezer PGC: t = 0 holds the initial populations and, per axis, the
    // thermal Wigner temperature of the initial trap, without error; the targets are met after it.
    let mut r=request(&|_|true);r.times_us=vec![0.,5.,10.];
    let s=solve(&r,&mut |_,_|{}).unwrap();assert!(s.reason.is_none());
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();let(b,x,_)=initial.states[0];
    let(frequencies,kt)=(model.trap_frequencies(b,x),KT_PER_UK*r.temperature_uk);
    for a in 0..3{let half=frequencies[a]/(2.*kt);assert_eq!(s.temperature_uk[0][a],r.temperature_uk*half/half.tanh());assert_eq!(s.temperature_errors_uk[0][a],0.);}
    assert_eq!(s.mean_velocity_mps[0],[0.;3]);assert_eq!(s.sample_errors[0],0.);
    for i in 0..STATES{assert_eq!(s.populations[0][i],r.initial_density[i*STATES+i].re);}
    for e in &s.temperature_errors_uk[1..]{for a in 0..3{assert!(e[a]>0.);}}
}
