//! The 3D engine's checks on the app's tweezer PGC example (tests/fixtures/tweezer_pgc.json: its
//! motion block along x, the frame y, z, x): the tables against the front end's dense generator at
//! random points and the photons' decay, the Ehrenfest force against the energy's gradient, the
//! dipole pattern's moments, the trap, a thermal state that stays thermal and held with the light
//! off (its T over a trap period as well, with smaller errors), a group's Latin hypercube of initial
//! momenta, and a short solve whose t = 0 is the request's initial state.
use atom_motion_3d::control::{Harmonic,Twin};
use atom_motion_3d::engine::{dipole_direction,Engine,Settings};
use atom_motion_3d::ground::Path;
use atom_motion_3d::ledger::BUDGET;
use atom_motion_3d::model::{Model,MOST_MIXED};
use atom_motion_3d::rng::{normal_quantile,Purpose,Rng};
use atom_motion_3d::stats::{summarize,Scales,KT_PER_UK};
use atom_motion_3d::{initial_trajectory,simulate,Initial,Stratum};
use atom_motion_3d::solve::{solve,Device};
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
fn uniform(seed:u64)->impl FnMut()->f64{let mut rng=Rng::new(seed,Purpose::Initial,0,0);move||rng.uniform()-0.5}
/// Random points within the trap, momenta and times, and normalized states of each cluster.
fn samples(model:&Model,seed:u64)->Vec<(usize,[f64;3],[f64;3],f64,[Complex;MOST_MIXED])>{
    let mut next=uniform(seed);
    (0..12).map(|i|{let cluster=i%model.clusters();let n=model.dimension(cluster);
        let r=[0.3*next(),0.8*next(),0.3*next()];let kappa=[60.*next(),60.*next(),60.*next()];let t=10.*(next()+0.5);
        let mut psi=[Complex::ZERO;MOST_MIXED];for z in psi[..n].iter_mut(){*z=Complex::new(next(),next());}
        let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        (cluster,r,kappa,t,psi)}).collect()
}
/// ⟨ψ|M|ψ⟩ of an n × n row-major matrix.
fn expectation(m:&[Complex],psi:&[Complex],n:usize)->Complex{let mut s=Complex::ZERO;for a in 0..n{for b in 0..n{s+=psi[a].conj()*m[a*n+b]*psi[b];}}s}

/// The model's tables against the front end's dense generator at random points and times (the
/// kinetic shift off, the van Vleck paths the tables hold), for the clusters of ground states and
/// those with explicit excited states alike: H_eff and every class's photons, to the tables'
/// interpolation; and the photons carry the whole decay but the paths' (their second order).
#[test]
fn the_tables_give_the_front_end_s_generator_and_the_photons_carry_the_decay(){
    // Every drive folded, and the drives kept until each ground state's folded rest is within the budget.
    for allowance in [f64::INFINITY,BUDGET]{
    let mut model=Model::with(&request(&|_|true),allowance).unwrap();model.doppler=false;let g=&model.ground;
    let tabulated=|p:&Path|model.carriers[p.via.carrier].beams.len()==1;
    assert_eq!((0..model.clusters()).any(|k|model.mixed(k)),allowance<f64::INFINITY,"explicit excited states at the allowance {allowance}");
    for(k,r,_,t,psi)in samples(&model,1){
        let n=model.dimension(k);let p=model.point(r,[0.;3]);let local=model.generator(k,&p,t);
        let e=g.eliminate(&model.hamiltonian_at_point(r)).unwrap();let dense=g.local_with(&e,k,r,t,&tabulated).unwrap();
        let scale=dense.effective.largest();
        for a in 0..n{for b in 0..n{let x=local.hamiltonian[a*n+b]+if a==b{Complex::new(local.scalar,0.)}else{Complex::ZERO};
            assert!((x-dense.effective[(a,b)]).abs()<=1e-3*scale,"cluster {k} at {r:?}, ({a}, {b}): {x:?} against {:?}",dense.effective[(a,b)]);}}
        let photons=model.photons(k,&p,&psi,t,None);
        let fast:f64=photons.amplitude.iter().zip(&g.clusters[k].classes).map(|(u,c)|{let nt=model.ground_dimension(c.target);u.iter().map(|axis|axis[..nt].iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>()}).sum();
        let exact:f64=dense.jumps.iter().map(|j|j.iter().map(|m|{let v:Vec<Complex>=(0..m.rows).map(|i|(0..n).fold(Complex::ZERO,|s,b|s+m[(i,b)]*psi[b])).collect();v.iter().map(|z|z.norm_sqr()).sum::<f64>()}).sum::<f64>()).sum();
        assert!((fast-exact).abs()<=1e-3*exact.max(1e-12),"cluster {k}: photons {fast:e} against {exact:e}");
        let decay=-2.*expectation(&local.hamiltonian[..n*n],&psi,n).im;
        let shift=-2.*expectation(&dense.shift.entries,&psi,n).im;
        assert!((decay-shift-fast).abs()<=1e-3*fast.max(1e-12),"cluster {k} at {r:?}: decay {decay:e} less the paths' {shift:e} against {fast:e}");
    }
    }
}

/// The Ehrenfest force is minus the gradient of the energy Re⟨ψ|H_eff|ψ⟩ (central differences),
/// the trap's and the light's -- with explicit excited states (the drives kept within the budget)
/// their own potential and the kept drives' stimulated force --, the Doppler term's own gradient
/// aside (the kinetic shift off).
#[test]
fn the_force_is_the_energy_s_gradient(){
    for allowance in [f64::INFINITY,BUDGET]{
    let mut model=Model::with(&request(&|_|true),allowance).unwrap();model.doppler=false;
    let energy=|cluster:usize,r:[f64;3],t:f64,psi:&[Complex;MOST_MIXED]|->f64{
        let n=model.dimension(cluster);let local=model.generator(cluster,&model.point(r,[0.;3]),t);local.scalar+expectation(&local.hamiltonian[..n*n],psi,n).re
    };
    for(cluster,r,_,t,psi)in samples(&model,2){
        let p=model.point(r,[0.;3]);let force=model.generator(cluster,&p,t).force(&psi);let h=1e-5;
        for a in 0..3{
            let(mut plus,mut minus)=(r,r);plus[a]+=h;minus[a]-=h;let numeric=-(energy(cluster,plus,t,&psi)-energy(cluster,minus,t,&psi))/(2.*h);
            assert!((force[a]-numeric).abs()<=1e-5*numeric.abs().max(1.),"cluster {cluster} at {r:?}, axis {a}: {} against {numeric}",force[a]);
        }
    }
    }
}

/// The photon's direction follows the dipole pattern: ⟨n_i n_j⟩ = ((2/5) δ_ij tr G − (1/5) Re G_ij)/tr G
/// for a σ dipole about z with a linear admixture (a Hermitian G of rank 2).
#[test]
fn the_emission_follows_the_dipole_pattern(){
    let u=[[Complex::new(0.6,0.),Complex::new(0.,0.2)],[Complex::new(0.,0.6),Complex::new(0.1,0.)],[Complex::new(0.3,0.1),Complex::new(0.,0.)]];
    let gram:[[Complex;3];3]=std::array::from_fn(|a|std::array::from_fn(|b|u[a].iter().zip(&u[b]).fold(Complex::ZERO,|s,(x,y)|s+x.conj()**y)));
    let trace=gram[0][0].re+gram[1][1].re+gram[2][2].re;let mut rng=Rng::new(7,Purpose::Jump,1,0);let count=400_000;let mut second=[[0.;3];3];
    for _ in 0..count{let n=dipole_direction(&mut rng,&gram);for i in 0..3{for j in 0..3{second[i][j]+=n[i]*n[j]/count as f64;}}}
    for i in 0..3{for j in 0..3{
        let expected=((if i==j{0.4*trace}else{0.})-0.2*gram[i][j].re)/trace;
        assert!((second[i][j]-expected).abs()<=4e-3,"<n{i} n{j}> {} against {expected}",second[i][j]);
    }}
}

/// The trap: the tweezer's frequencies at the focus, 100.4 kHz across it and 19.3 kHz along it (the
/// pipeline study's 100.55 kHz radial with the tweezer alone; the cooling light's own trap lowers
/// it), the same with the tweezer as the only light; its principal axes are the frame's.
#[test]
fn the_trap_is_the_tweezer_s(){
    let trap=|model:&Model|{let initial=Initial::new(model,&request(&|_|true)).unwrap();initial.traps[0]};
    let all=trap(&Model::new(&request(&|_|true)).unwrap());let khz=all.lab.map(|w|w/std::f64::consts::TAU*1e3);
    assert!((khz[0]-100.41).abs()<0.05&&(khz[2]-100.41).abs()<0.05&&(khz[1]-19.26).abs()<0.05,"{khz:?}");
    let mut principal=all.frequencies;principal.sort_by(f64::total_cmp);let mut lab=all.lab;lab.sort_by(f64::total_cmp);
    assert!(principal.iter().zip(&lab).all(|(a,b)|(a-b).abs()<=1e-6*b),"{principal:?} against {lab:?}");
    let only=trap(&Model::new(&request(&|b|b["waistUm"].is_number())).unwrap()).lab.map(|w|w/std::f64::consts::TAU*1e3);
    assert!((only[0]-100.55).abs()<0.05&&(only[1]-19.28).abs()<0.05,"{only:?}");
}

/// With the light off the thermal state stays thermal and held: T on every axis keeps its initial
/// value (no jumps, no atom lost), within three standard errors at every sample over 200 us.
#[test]
fn a_thermal_state_stays_thermal_with_the_light_off(){
    let r=request(&|b|b["waistUm"].is_number());let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let trap=initial.traps[0];assert_eq!(model.focused.len(),1);
    let settings=Settings::new(0.25,&trap,f64::EPSILON,false);
    let outcome=simulate(&model,&settings,&initial,5,0,1024,8,40,21,&[],None);assert_eq!(outcome.jumps,0);
    let e=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab},&[]);
    for a in 0..3{let start=e[0].temperature[a];for x in &e{
        assert!((x.temperature[a]-start).abs()<=3.*(x.temperature_error[a].hypot(e[0].temperature_error[a])),"axis {a}: {} against {start}",x.temperature[a]);
        assert!((x.temperature[a]-r.temperature_uk).abs()<=4.*x.temperature_error[a],"axis {a}: {} against {}",x.temperature[a],r.temperature_uk);
        assert_eq!(x.loss,0.);
    }}
    let _=(Engine::new(&model,settings),initial_trajectory);
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
    // Maxwell distribution (its strata bounded by the normal's quantiles at k/8), on the trap's
    // principal axes.
    let r=request(&|_|true);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let trap=initial.traps[0];let settings=Settings::new(0.25,&trap,f64::EPSILON,false);
    let(size,kt,hm)=(8_u64,KT_PER_UK*r.temperature_uk,model.hbar_over_mass);
    for first in [0,8,96]{
        let group:Vec<[f64;3]>=(first..first+size).map(|i|initial_trajectory(&model,&settings,&initial,3,i,Stratum{first,size}).kappa).collect();
        for i in 0..3{
            let half=trap.frequencies[i]/(2.*kt);let width=(kt/hm).sqrt()*(half/half.tanh()).sqrt();
            let along=|k:&[f64;3]|(0..3).map(|a|k[a]*trap.axes[i][a]).sum::<f64>();
            let mut strata:Vec<usize>=group.iter().map(|k|(1..size as usize).filter(|&j|along(k)/width>=normal_quantile(j as f64/size as f64)).count()).collect();
            strata.sort();assert_eq!(strata,(0..size as usize).collect::<Vec<_>>(),"group {first}, axis {i}");
        }
    }
}

#[test]
fn a_thermal_state_s_temperature_over_a_trap_period_is_its_temperature_with_smaller_errors(){
    // The light off: T over one trap period of each axis against T at the time, the same atoms.
    let r=request(&|b|b["waistUm"].is_number());let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let trap=initial.traps[0];
    let settings=|average:bool|Settings::new(0.25,&trap,f64::EPSILON,average);
    let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab};
    let run=|average:bool|summarize(&simulate(&model,&settings(average),&initial,5,0,1024,1,40,21,&[],None).groups,&scales,&[]);
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
    let s=solve(&r,Device::Cpu,&mut |_,_|{}).unwrap();assert!(s.reason.is_none());
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();let trap=initial.traps[0];let kt=KT_PER_UK*r.temperature_uk;
    for a in 0..3{let expected:f64=(0..3).map(|i|{let half=trap.frequencies[i]/(2.*kt);trap.axes[i][a]*trap.axes[i][a]*r.temperature_uk*half/half.tanh()}).sum();
        assert!((s.temperature_uk[0][a]-expected).abs()<=1e-12*expected,"axis {a}: {} against {expected}",s.temperature_uk[0][a]);assert_eq!(s.temperature_errors_uk[0][a],0.);}
    assert_eq!(s.mean_velocity_mps[0],[0.;3]);assert_eq!(s.sample_errors[0],0.);assert_eq!(s.temperature_total_errors_uk[0],0.);
    for i in 0..STATES{assert_eq!(s.populations[0][i],r.initial_density[i*STATES+i].re);}
    for e in &s.temperature_errors_uk[1..]{for a in 0..3{assert!(e[a]>0.);}}
    for(t,e)in s.temperature_total_uk.iter().zip(&s.temperature_total_errors_uk).skip(1){assert!(*t>0.&&*e>0.);}
}

/// The harmonic control's mean (control.rs) against the initial states' own: its value from 4096
/// drawn initial states at outputs over a trap period, within four standard errors of its exact mean.
#[test]
fn the_harmonic_control_s_mean_is_exact(){
    let r=request(&|_|true);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();let trap=initial.traps[0];
    let settings=Settings::new(0.25,&trap,f64::EPSILON,true);let(steps,outputs)=(8,41);
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);
    let starts:Vec<([f64;3],[f64;3])>=(0..4096).map(|i|{let t=initial_trajectory(&model,&settings,&initial,11,i,Stratum::single(i));(t.r,t.kappa)}).collect();
    for i in (0..outputs).step_by(5){for a in 0..3{
        let v:Vec<f64>=starts.iter().map(|&(r0,k0)|harmonic.value(i,r0,k0)[a]).collect();let n=v.len() as f64;
        let m=v.iter().sum::<f64>()/n;let se=(v.iter().map(|x|(x-m).powi(2)).sum::<f64>()/(n-1.)/n).sqrt();
        assert!((m-harmonic.mean[i][a]).abs()<=4.*se,"output {i} axis {a}: {m} ± {se} against {}",harmonic.mean[i][a]);
    }}
}

/// The held twin's exact means (control.rs: the engine's generator and jumps at its point, by the
/// superoperator's exponential) against the dense front end's master equation at the same point
/// (ground.rs `local_at`, RK4), from the initial state over 20 us: within the tables' accuracy.
#[test]
fn the_held_twin_s_master_equation_is_the_front_end_s(){
    let r=request(&|_|true);let model=Model::new(&r).unwrap();let g=&model.ground;
    let twin=Twin::new(&model,0.9).expect("a static generator");let point=twin.held.point.r;
    let(interval,outputs)=(1.,21);let means=twin.means(&model,&r.initial_density,interval,outputs);
    let e=g.eliminate(&model.hamiltonian_at_point(point)).unwrap();
    let locals:Vec<_>=(0..g.clusters.len()).map(|k|g.local_at(&e,k,point,0.).unwrap()).collect();
    let mut rho:Vec<atom_motion_3d::linear::Dense>=g.clusters.iter().map(|c|{let n=c.states.len();atom_motion_3d::linear::Dense::from_fn(n,n,|i,j|if i==j{let s=g.ground_states[c.states[i]];r.initial_density[s*STATES+s]}else{Complex::ZERO})}).collect();
    let derivative=|rho:&[atom_motion_3d::linear::Dense]|->Vec<atom_motion_3d::linear::Dense>{
        let mut out:Vec<_>=rho.iter().zip(&locals).map(|(p,x)|{let h=&x.effective;let mut d=h.product(p).scaled(Complex::new(0.,-1.));d.add_scaled(Complex::new(0.,1.),&p.product(&h.adjoint()));d}).collect();
        for(k,x)in locals.iter().enumerate(){for(class,jumps)in g.clusters[k].classes.iter().zip(&x.jumps){for j in jumps{out[class.target].add_scaled(Complex::ONE,&j.product(&rho[k]).product(&j.adjoint()));}}}
        out
    };
    let h=0.002;let mut worst=0_f64;
    for i in 1..outputs{
        for _ in 0..(interval/h).round() as usize{
            let add=|a:&[atom_motion_3d::linear::Dense],b:&[atom_motion_3d::linear::Dense],s:f64|->Vec<atom_motion_3d::linear::Dense>{a.iter().zip(b).map(|(x,y)|{let mut z=x.clone();z.add_scaled(Complex::new(s,0.),y);z}).collect()};
            let k1=derivative(&rho);let k2=derivative(&add(&rho,&k1,h/2.));let k3=derivative(&add(&rho,&k2,h/2.));let k4=derivative(&add(&rho,&k3,h));
            for(j,x)in rho.iter_mut().enumerate(){for(k,w)in [(&k1,1.),(&k2,2.),(&k3,2.),(&k4,1.)]{x.add_scaled(Complex::new(h*w/6.,0.),&k[j]);}}
        }
        let mut p=[0.;STATES];for(k,x)in locals.iter().enumerate(){for(a,b)in p.iter_mut().zip(g.populations(k,x,&rho[k])){*a+=b;}}
        worst=worst.max((0..STATES).map(|s|(p[s]-means[i][s]).abs()).fold(0.,f64::max));
    }
    assert!(worst<=2e-3,"largest |Δp| {worst:.2e}");
}

/// The held engine (engine.rs `Held`: the twin's generator and jumps computed once, control.rs)
/// follows the plain engine's atom held at the same point: from the same initial state and streams,
/// the same jumps and the same internal state over 2000 steps -- without the photons' recoil term,
/// where the engine's decay is its jumps' (the twin's by construction).
#[test]
fn the_held_engine_is_the_plain_engine_at_rest(){
    let r=request(&|_|true);let mut model=Model::new(&r).unwrap();model.doppler=false;let initial=Initial::new(&model,&r).unwrap();let trap=initial.traps[0];
    let twin=Twin::new(&model,0.9).expect("a static generator");
    let mut settings=Settings::new(0.2,&trap,f64::EPSILON,false);settings.free=[false;3];settings.loss=false;
    let(plain,held)=(Engine::new(&model,settings.clone()),Engine::held(&model,&settings,&twin.held));
    for index in 0..8{
        let mut a=initial_trajectory(&model,&settings,&initial,13,index,Stratum::single(index));a.r=twin.held.point.r;a.kappa=[0.;3];let mut b=a.clone();
        for _ in 0..2000{plain.step(&mut a);held.step(&mut b);}
        assert_eq!((a.jumps,a.cluster),(b.jumps,b.cluster),"trajectory {index}");
        let n=model.dimension(a.cluster);let(na,nb):(f64,f64)=(a.psi[..n].iter().map(|z|z.norm_sqr()).sum(),b.psi[..n].iter().map(|z|z.norm_sqr()).sum());
        for i in 0..n{assert!((a.psi[i].norm_sqr()/na-b.psi[i].norm_sqr()/nb).abs()<1e-6,"trajectory {index} state {i}");}
    }
}

/// The held twin is the unraveling of a Lindblad equation: its generator's decay is its jumps' own
/// Σ L†L (the photons' first order in the recoil kept), so its master equation (`Twin::means`) keeps
/// the trace one -- and every correction of the populations (stats.rs) keeps their sum -- over
/// 100 us of a probe on the tweezer.
#[test]
fn the_held_twin_is_a_lindblad_unraveling_and_keeps_the_trace(){
    let r=request(&|b|b["id"]=="tweezer"||b["id"]=="cool-plus");let model=Model::new(&r).unwrap();
    let twin=Twin::new(&model,0.93).expect("a static generator");
    for k in 0..model.clusters(){
        let n=model.dimension(k);let h=&twin.held.locals[k].hamiltonian;let(mut worst,mut scale)=(0_f64,0_f64);
        for p in 0..n{for q in 0..n{
            let decay=(h[p*n+q]-h[q*n+p].conj())*Complex::new(0.,1.);let mut jumps=Complex::ZERO;
            for(class,l)in model.ground.clusters[k].classes.iter().zip(&twin.held.jumps[k]){let nt=model.dimension(class.target);for x in 0..3{for i in 0..nt{jumps+=l[x][i*n+p].conj()*l[x][i*n+q];}}}
            worst=worst.max((decay-jumps).abs());scale=scale.max(jumps.abs());
        }}
        assert!(worst<=1e-12*scale.max(f64::MIN_POSITIVE),"cluster {k}: |Γ − ΣL†L| {worst:.2e} of {scale:.2e}");
    }
    let means=twin.means(&model,&r.initial_density,1.,101);
    let worst=means.iter().map(|p|(p.iter().sum::<f64>()-1.).abs()).fold(0.,f64::max);
    assert!(worst<=1e-10,"the trace off by {worst:.2e}");
}
