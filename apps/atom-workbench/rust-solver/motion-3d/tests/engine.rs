//! The 3D engine's checks on the app's tweezer PGC example (tests/fixtures/tweezer_pgc.json: its
//! motion block along x, the frame y, z, x): the tables against the front end's dense generator at
//! random points and the photons' decay, the Ehrenfest force against the energy's gradient, the
//! dipole pattern's moments, the trap, a thermal state that stays thermal and held with the light
//! off (its T over a trap period as well, with smaller errors), the initial states' bound ensemble and
//! a group's Latin hypercube of it, and a short solve whose t = 0 is the request's initial state.
use atom_motion_3d::control::{Harmonic,Twin};
use atom_motion_3d::engine::{dipole_direction,Engine,Settings,Trajectory,QUADRATURE};
use atom_motion_3d::ensemble::kept;
use atom_motion_3d::ground::{ExcitedKind,Ground};
use atom_motion_3d::ledger::BUDGET;
use atom_motion_3d::model::{Model,MOST_MIXED};
use atom_motion_3d::rng::{normal_quantile,Purpose,Rng};
use atom_motion_3d::stats::{summarize,Means,Scales,KT_PER_UK};
use atom_motion_3d::{initial_trajectory,sample_states,simulate,simulate_to,Initial,Stratum,Variates};
use atom_motion_3d::solve::{grid_phase,jumps,solve,Device,BIAS_SHARE,NUMERICAL_SHARE,POPULATION_TARGET,TEMPERATURE_TARGET};
use atom_motion_3d::{Beam,Complex,Matrix,Request,STATES};
use serde_json::Value;
#[path="../examples/scene/mod.rs"] mod scene;

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
/// kinetic shift off), for the clusters of ground states and those with explicit excited states
/// alike: H_eff and every class's photons, to the tables' interpolation; and the photons carry the
/// whole decay (the dressed responses', ground.rs # The near light's dressing).
#[test]
fn the_tables_give_the_front_end_s_generator_and_the_photons_carry_the_decay(){
    // Every drive folded, and the drives kept until each ground state's folded rest is within the budget.
    for allowance in [f64::INFINITY,BUDGET]{
    let mut model=Model::with(&request(&|_|true),allowance).unwrap();model.doppler=false;let g=&model.ground;
    assert_eq!((0..model.clusters()).any(|k|model.mixed(k)),allowance<f64::INFINITY,"explicit excited states at the allowance {allowance}");
    for(k,r,_,t,psi)in samples(&model,1){
        let n=model.dimension(k);let p=model.point(r,[0.;3]);let local=model.generator(k,&p,t);
        let e=g.eliminate(&model.intensities_at_point(r)).unwrap();let dense=g.local_at(&e,k,r,t).unwrap();
        let scale=dense.effective.largest();
        for a in 0..n{for b in 0..n{let x=local.hamiltonian[a*n+b]+if a==b{Complex::new(local.scalar,0.)}else{Complex::ZERO};
            assert!((x-dense.effective[(a,b)]).abs()<=1e-3*scale,"cluster {k} at {r:?}, ({a}, {b}): {x:?} against {:?}",dense.effective[(a,b)]);}}
        let photons=model.photons(k,&p,&psi,t,None);
        let fast:f64=photons.amplitude.iter().zip(&g.clusters[k].classes).map(|(u,c)|{let nt=model.ground_dimension(c.target);u.iter().map(|axis|axis[..nt].iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>()}).sum();
        let exact:f64=dense.jumps.iter().map(|j|j.iter().map(|m|{let v:Vec<Complex>=(0..m.rows).map(|i|(0..n).fold(Complex::ZERO,|s,b|s+m[(i,b)]*psi[b])).collect();v.iter().map(|z|z.norm_sqr()).sum::<f64>()}).sum::<f64>()).sum();
        assert!((fast-exact).abs()<=1e-3*exact.max(1e-12),"cluster {k}: photons {fast:e} against {exact:e}");
        let decay=-2.*expectation(&local.hamiltonian[..n*n],&psi,n).im;
        assert!((decay-fast).abs()<=1e-3*fast.max(1e-12),"cluster {k} at {r:?}: decay {decay:e} against {fast:e}");
    }
    }
}

/// The fourth order's tables (model.rs `FourthLayout`) against the dense fourth-order fold (ground.rs
/// `Ground::fourth`) at random points and times, every drive folded, the kinetic shift off: H_eff's
/// Hermitian part with H^(4), every class's photons with −C W3 − ½ C R R†R, and the post-jump slow part
/// Π, to the tables' interpolation.
#[test]
fn the_fourth_order_tables_give_the_dense_fourth_order_fold(){
    let mut model=Model::with_order(&request(&|_|true),f64::INFINITY,true).unwrap();model.doppler=false;let g=&model.ground;
    let herm=|m:&dyn Fn(usize,usize)->Complex,a:usize,b:usize|(m(a,b)+m(b,a).conj())*0.5;
    for(k,r,_,t,psi)in samples(&model,5){
        let n=model.dimension(k);let p=model.point(r,[0.;3]);let local=model.local(k,&p,t);
        let e=g.eliminate(&model.intensities_at_point(r)).unwrap();let dense=g.fourth(&e,k,r,t).unwrap();
        let scale=dense.effective.largest();
        // The local generator and the beats' components (`Model::generator`, a beating cluster's path).
        let generator=model.generator(k,&p,t);
        for a in 0..n{for b in 0..n{
            let x=herm(&|i,j|local.hamiltonian[i*n+j]+if i==j{Complex::new(local.scalar,0.)}else{Complex::ZERO},a,b);let y=herm(&|i,j|dense.effective[(i,j)],a,b);
            assert!((x-y).abs()<=2e-3*scale,"cluster {k} at {r:?}, ({a}, {b}): {x:?} against {y:?}");
            let z=herm(&|i,j|generator.hamiltonian[i*n+j]+if i==j{Complex::new(generator.scalar,0.)}else{Complex::ZERO},a,b);
            assert!((z-y).abs()<=2e-3*scale,"cluster {k} at {r:?}, ({a}, {b}), components: {z:?} against {y:?}");
        }}
        let photons=model.photons(k,&p,&psi,t,None);
        for(c,(u,j))in photons.amplitude.iter().zip(&dense.jumps).enumerate(){let nt=model.ground_dimension(g.clusters[k].classes[c].target);
            let norm:f64=j.iter().map(|m|(0..m.rows).map(|i|(0..n).fold(Complex::ZERO,|s,q|s+m[(i,q)]*psi[q]).norm_sqr()).sum::<f64>()).sum::<f64>().sqrt().max(1e-12);
            for x in 0..3{for i in 0..nt{let v=(0..n).fold(Complex::ZERO,|s,q|s+j[x][(i,q)]*psi[q]);
                assert!((u[x][i]-v).abs()<=2e-3*norm,"cluster {k} class {c} axis {x} row {i}: {:?} against {v:?}",u[x][i]);}}
        }
        let slow=model.slow_part(k,&p,t).unwrap();
        for a in 0..n{for b in 0..n{assert!((slow[a*n+b]-dense.slow[(a,b)]).abs()<=2e-3,"cluster {k} Π ({a}, {b}): {:?} against {:?}",slow[a*n+b],dense.slow[(a,b)]);}}
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

/// The trap's principal-axis Wigner factors at T (lib.rs `initial_trajectory`) taken out of a vector.
fn classical(trap:&atom_motion_3d::model::Trap,kt:f64,v:[f64;3])->[f64;3]{
    std::array::from_fn(|a|(0..3).map(|i|{let half=trap.frequencies[i]/(2.*kt);let quantum=if half>0.{(half/half.tanh()).sqrt()}else{1.};
        (0..3).map(|b|v[b]*trap.axes[i][b]).sum::<f64>()/quantum*trap.axes[i][a]}).sum())
}
/// The 31 inner edges of 32 bins of equal probability of a distribution function on [lo, hi] (bisection).
fn quantiles(distribution:&(dyn Fn(f64)->f64+Sync),lo:f64,hi:f64)->Vec<f64>{
    use rayon::prelude::*;
    (1..32).into_par_iter().map(|j|{let target=j as f64/32.;let(mut a,mut b)=(lo,hi);for _ in 0..60{let m=0.5*(a+b);if distribution(m)<target{a=m;}else{b=m;}}0.5*(a+b)}).collect()
}
/// χ² of samples in the bins between `edges` against equal probabilities.
fn chi_square(samples:&[f64],edges:&[f64])->(f64,Vec<usize>){
    let mut hits=vec![0usize;edges.len()+1];for &x in samples{hits[edges.partition_point(|&e|e<x)]+=1;}
    let expected=samples.len() as f64/hits.len() as f64;(hits.iter().map(|&h|(h as f64-expected).powi(2)/expected).sum(),hits)
}

/// The initial states are the trap's bound ensemble, exactly (ensemble.rs: the lowered isothermal f ∝
/// e^{−E/k_BT} − 1 over E < 0): the tweezer alone -- its trap potential −D A(r), A the Gaussian beam's
/// profile, to the tables' round-off -- at a tenth of its depth and at two thirds of it, 2¹⁶
/// independent initial states (the Wigner factors taken out) against the exact distributions of the
/// position's depth A and of the energy E/D, from the beam's volume element ∝ (Z + Z³/3)/A per dA (Z² =
/// 1/A − 1) and the momenta's `kept` at each A: each in 32 bins of equal probability, χ² within its
/// 0.1 % point (61.1 at 31 degrees of freedom). No state lies above the far level, by the trap's measure
/// or the loss's; deep in the trap the ensemble is Boltzmann's to e^{−D/k_BT}, shallow its kinetic
/// temperature lies well below T.
#[test]
fn the_initial_states_are_the_trap_s_bound_ensemble(){
    use rayon::prelude::*;
    let mut r=request(&|b|b["waistUm"].is_number());let model=Model::new(&r).unwrap();let hm=model.hbar_over_mass;
    let depth={let i=Initial::new(&model,&r).unwrap();i.ensembles[0].as_ref().unwrap().depth};
    let beam=&model.geometry[model.trap_beam];let profile=|q:[f64;3]|beam.intensity(q)/beam.intensity([0.;3]);
    for ratio in [10.,1.5]{
        r.temperature_uk=depth/ratio/KT_PER_UK;let initial=Initial::new(&model,&r).unwrap();assert_eq!(initial.states.len(),1);
        let(cluster,state,_)=initial.states[0];let trap=initial.traps[0];let kt=KT_PER_UK*r.temperature_uk;let x=ratio;
        // The exact distributions over τ = √(1 − A) (smooth at the focus): the position's weight in A,
        // w(xA)(Z + Z³/3)/A dA, and the energy's below e (in D): each A's momenta below k = x(A + e).
        let nodes=4000;let grid:Vec<(f64,f64)>=(0..nodes).map(|i|{let tau=(i as f64+0.5)/nodes as f64;let a=1.-tau*tau;let z=(1./a-1.).sqrt();
            (a,kept(1.5,x*a,x*a,x)*(z+z*z*z/3.)/a*2.*tau)}).collect();
        let total:f64=grid.iter().map(|g|g.1).sum();
        let position=|a:f64|grid.iter().filter(|g|g.0<=a).map(|g|g.1).sum::<f64>()/total;
        let energy=|e:f64|grid.iter().map(|&(a,w)|{let k=(x*(a+e)).clamp(0.,x*a);w*kept(1.5,x*a,k,x)/kept(1.5,x*a,x*a,x)}).sum::<f64>()/total;
        // The sampled initial states (each trajectory's own draw).
        let settings=Settings::new(0.25,&trap,f64::EPSILON,true);let count=1u64<<16;
        let mut psi=[Complex::ZERO;MOST_MIXED];psi[state]=Complex::ONE;
        let drawn:Vec<(f64,f64,f64,f64)>=(0..count).into_par_iter().map(|i|{let t=initial_trajectory(&model,&settings,&initial,7,i,Stratum::single(i));
            let(q,k)=(classical(&trap,kt,t.r),classical(&trap,kt,t.kappa));let kinetic=0.5*hm*k.iter().map(|v|v*v).sum::<f64>();
            let loss=0.5*hm*t.kappa.iter().map(|v|v*v).sum::<f64>()+model.trap_potential(cluster,&model.point(t.r,t.kappa),&psi,0.);
            (profile(q),(kinetic+model.trap_level(cluster,state,q))/depth,loss,t.kappa.iter().map(|v|v*v).sum::<f64>()*hm/(3.*KT_PER_UK))}).collect();
        assert!(drawn.iter().all(|d|d.1<0.&&d.2<0.),"depth/kT {ratio}: a state above the far level");
        for(name,statistic,distribution,lo,hi)in [("A",0,&position as &(dyn Fn(f64)->f64+Sync),0.,1.),("E/D",1,&energy,-1.,0.)]{
            let samples:Vec<f64>=drawn.iter().map(|d|if statistic==0{d.0}else{d.1}).collect();
            let(chi,hits)=chi_square(&samples,&quantiles(distribution,lo,hi));
            println!("depth/kT {ratio}: {name} χ² {chi:.1}");
            assert!(chi<=61.1,"depth/kT {ratio}, {name}: χ² {chi:.1} over {hits:?}");
        }
        // The kinetic temperature: the ensemble's exact one, below T where the trap is shallow.
        let n=count as f64;let m=drawn.iter().map(|d|d.3).sum::<f64>()/n;let se=(drawn.iter().map(|d|(d.3-m).powi(2)).sum::<f64>()/(n-1.)/n).sqrt();
        let exact=initial.temperature().iter().sum::<f64>()/3.;
        println!("depth/kT {ratio}: kinetic T {m:.3} ± {se:.3} uK against {exact:.3} (T {:.1}), the ensemble's share left out {:.4}",r.temperature_uk,initial.cut);
        assert!((m-exact).abs()<=4.*se,"depth/kT {ratio}: kinetic T {m} ± {se} against {exact}");
        if ratio==10.{assert!((initial.kinetic()-1.).abs()<0.05&&initial.cut<0.05,"{} {}",initial.kinetic(),initial.cut);}else{assert!(initial.kinetic()<0.5,"{}",initial.kinetic());}
    }
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
    // In a group of eight, each member's kinetic energy lies in another eighth of its distribution at
    // the position it took (ensemble.rs: kept(3/2, y, k)/kept(3/2, y, k_max), the Wigner factors taken
    // out) and its momentum's direction in another eighth of the polar cosine's.
    let r=request(&|_|true);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let trap=initial.traps[0];let settings=Settings::new(0.25,&trap,f64::EPSILON,false);
    let(size,kt,hm)=(8_u64,KT_PER_UK*r.temperature_uk,model.hbar_over_mass);let(cluster,state,_)=initial.states[0];
    let mut psi=[Complex::ZERO;MOST_MIXED];psi[state]=Complex::ONE;
    for first in [0,8,96]{
        let group:Vec<([f64;3],[f64;3])>=(first..first+size).map(|i|{let t=initial_trajectory(&model,&settings,&initial,3,i,Stratum{first,size});(classical(&trap,kt,t.r),classical(&trap,kt,t.kappa))}).collect();
        let quantile=|(q,k):&([f64;3],[f64;3])|{
            let y=-model.trap_level(cluster,state,*q)/kt;let reach=y.min(-model.trap_potential(cluster,&model.point(*q,[0.;3]),&psi,0.)/kt);
            kept(1.5,y,0.5*hm*k.iter().map(|v|v*v).sum::<f64>()/kt,y)/kept(1.5,y,reach,y)};
        let cosine=|(_,k):&([f64;3],[f64;3])|0.5*(1.+k[2]/k.iter().map(|v|v*v).sum::<f64>().sqrt());
        for(name,value)in [("kinetic energy",&quantile as &dyn Fn(&([f64;3],[f64;3]))->f64),("direction",&cosine)]{
            let mut strata:Vec<usize>=group.iter().map(|m|((value(m)*size as f64).floor() as usize).min(size as usize-1)).collect();
            strata.sort();assert_eq!(strata,(0..size as usize).collect::<Vec<_>>(),"group {first}, {name}");
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
    // initial ensemble's kinetic temperature (lib.rs `Initial::temperature`) without error -- in this
    // trap 28 k_BT deep the thermal Wigner temperature of the initial trap to 10⁻⁶; the targets are met
    // after it.
    let mut r=request(&|_|true);r.times_us=vec![0.,5.,10.];
    let s=solve(&r,Device::Cpu,&mut |_,_|{}).unwrap();assert!(s.reason.is_none());
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();let trap=initial.traps[0];let kt=KT_PER_UK*r.temperature_uk;
    let exact=initial.temperature();
    for a in 0..3{let wigner:f64=(0..3).map(|i|{let half=trap.frequencies[i]/(2.*kt);trap.axes[i][a]*trap.axes[i][a]*r.temperature_uk*half/half.tanh()}).sum();
        assert!((s.temperature_uk[0][a]-exact[a]).abs()<=1e-12*exact[a],"axis {a}: {} against {}",s.temperature_uk[0][a],exact[a]);
        assert!((exact[a]-wigner).abs()<=1e-6*wigner,"axis {a}: {} against the Wigner temperature {wigner}",exact[a]);assert_eq!(s.temperature_errors_uk[0][a],0.);}
    assert_eq!(s.mean_velocity_mps[0],[0.;3]);assert_eq!(s.sample_errors[0],0.);assert_eq!(s.temperature_total_errors_uk[0],0.);
    for i in 0..STATES{assert_eq!(s.populations[0][i],r.initial_density[i*STATES+i].re);}
    for e in &s.temperature_errors_uk[1..]{for a in 0..3{assert!(e[a]>0.);}}
    for(t,e)in s.temperature_total_uk.iter().zip(&s.temperature_total_errors_uk).skip(1){assert!(*t>0.&&*e>0.);}
}

/// The fold's choice on the shallow trap's cycling control (solve.rs, # The fold's choice): every drive stays
/// folded, the cheapest by the candidates' counted costs. The line took the first explicit rung uncosted (6
/// excited states exact, dt 0.094 us, 2.6 times the folded model's CPU time): the pilot's fold charge was the
/// extreme of its states grouped by their times' last bits (6.3 % against their mean 2.9 %), and left the
/// folded model no room.
#[test]
fn the_shallow_cycle_keeps_its_drives_folded(){
    let fixture:Value=serde_json::from_str(include_str!("../../tests/fixtures/shallow_cycle_motion.json")).unwrap();
    let mut r=scene::request(&fixture["payload"]);r.times_us=(0..31).map(|i|10.*i as f64).collect();
    let s=solve(&r,Device::Cpu,&mut |_,_|{}).unwrap();
    println!("{} trajectories, dt {:.4} us; {}",s.trajectories,s.time_step_us,s.method);
    assert!(s.method.contains("消去激发态后的基态内态"),"{}",s.method);
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

/// The harmonic control at any step (control.rs `Harmonic::new`: its window means in closed form): at
/// 2⁴⁰ steps per output -- the running sums it once kept per step would have taken 10¹⁴ bytes -- its
/// means are those of 2²⁴ steps, both the continuous time's window means to about the step's share of
/// a trap period.
#[test]
fn the_harmonic_control_keeps_no_sums_per_step(){
    let r=request(&|_|true);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();let trap=initial.traps[0];
    let(interval,outputs)=(2.,41);
    let at=|steps:usize|Harmonic::new(&model,&Settings::new(interval/steps as f64,&trap,f64::EPSILON,true),&initial,steps,outputs);
    let(fine,coarse)=(at(1<<40),at(1<<24));
    for i in 0..outputs{for x in 0..3{
        assert!(fine.mean[i][x].is_finite()&&(fine.mean[i][x]-coarse.mean[i][x]).abs()<=1e-6*coarse.mean[i][x].abs(),"output {i} axis {x}: {} against {}",fine.mean[i][x],coarse.mean[i][x]);
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
    let e=g.eliminate(&model.intensities_at_point(point)).unwrap();
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
/// Σ L†L (the photons' first order in the recoil kept; at fourth order K = Π L̃, the post-jump map
/// in its jumps, control.rs), so its master equation (`Twin::means`) keeps the trace one -- and every
/// correction of the populations (stats.rs) keeps their sum -- over 100 us of a probe on the
/// tweezer, at second and at fourth order.
#[test]
fn the_held_twin_is_a_lindblad_unraveling_and_keeps_the_trace(){
    let r=request(&|b|b["id"]=="tweezer"||b["id"]=="cool-plus");
    for fourth in [false,true]{
    let model=Model::with_order(&r,f64::INFINITY,fourth).unwrap();
    let twin=Twin::new(&model,0.93).expect("a static generator");
    assert_eq!(twin.slow.iter().any(|s|s.is_some()),fourth,"the post-jump map at fourth order only");
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
}

/// The held twin's populations average to its master equation's at fourth order: 4096 twins of a
/// probe on the tweezer (their bare initial state's slow part at the twin's point, their streams the
/// trajectories'), held 30 us in steps of 0.2 us, against `Twin::means` at every output within four
/// standard errors -- the post-jump map in the jumps, K = Π L̃, and the plain threshold make the
/// unraveling exactly that Lindblad equation's (control.rs, # The twin at fourth order).
#[test]
fn the_held_twin_at_fourth_order_averages_to_its_master_equation(){
    let r=request(&|b|b["id"]=="tweezer"||b["id"]=="cool-plus");let model=Model::with_order(&r,f64::INFINITY,true).unwrap();
    let initial=Initial::new(&model,&r).unwrap();let twin=Twin::new(&model,0.93).expect("a static generator");
    assert!(twin.slow.iter().any(|s|s.is_some()),"fourth-order tables");
    let mut settings=Settings::new(0.2,&initial.traps[0],f64::EPSILON,false);settings.free=[false;3];settings.loss=false;
    let held=Engine::held(&model,&settings,&twin.held);
    let(steps,outputs,count)=(5,31,4096u64);let means=twin.means(&model,&r.initial_density,1.,outputs);
    let(cluster,state,_)=initial.states[0];
    let runs:Vec<Vec<[f64;STATES]>>=(0..count).map(|index|{
        let mut psi=[Complex::ZERO;MOST_MIXED];psi[state]=Complex::ONE;
        let mut t=atom_motion_3d::engine::Trajectory{cluster,psi,r:twin.held.point.r,kappa:[0.;3],time:0.,threshold:atom_motion_3d::engine::jump_stream(17,index,0).open_uniform(),
            alive:true,jumps:0,doppler:0.,beats:0.,seed:17,index};
        twin.start(&mut t);
        let mut out=vec![twin.populations(t.cluster,&t.psi)];
        for _ in 1..outputs{for _ in 0..steps{held.step(&mut t);}out.push(twin.populations(t.cluster,&t.psi));}
        out}).collect();
    let mut worst=(0_f64,0usize,0usize);
    for i in 0..outputs{for s in 0..STATES{
        let v:Vec<f64>=runs.iter().map(|x|x[i][s]).collect();let n=v.len() as f64;let m=v.iter().sum::<f64>()/n;
        // The spread at least a rare event's (the exact mean's share of the twins once each), where
        // few twins or none have jumped yet.
        let se=(v.iter().map(|x|(x-m).powi(2)).sum::<f64>()/(n-1.)/n).sqrt().max((means[i][s].abs()/n).sqrt());
        if se>0.{let z=(m-means[i][s]).abs()/se;if z>worst.0{worst=(z,i,s);}}
    }}
    println!("largest |z| {:.2} (output {}, state {})",worst.0,worst.1,worst.2);
    assert!(worst.0<=4.,"output {} state {}: |z| {:.2}",worst.1,worst.2,worst.0);
}

/// A shallow trap's loss against the step (rust-solver/tests/fixtures/shallow_probe_motion.json: the
/// default scene's tweezer at 0.6 mW, 128 uK deep, its probe on F = 2 → F' = 2 at 1 mW/cm², the atom
/// at 30 uK, 500 us): the probe pumps every atom into F = 1 within the first sample, dark from then on
/// and its energy kept, so that past it an atom is lost only by the step. 1024 trajectories on shared
/// streams at 1.25 us, about the step the solve takes (1.43 us), and at a quarter of it. The loss was judged on the end's
/// κ with the middle's potential -- first order in dt: 2.7 % of the atoms lost in the dark at 2 us --; on
/// the middle's state alone the losses at the two steps differ by no more than their discordant
/// trajectories' spread (three standard errors of the sign test), with the trap's region (the solve's,
/// engine.rs `Settings::region`) as without. The bound initial ensemble (ensemble.rs) holds atoms up to
/// the trap's top, some far out on its axis: the dark atoms the step loses each started within 5 % of
/// the depth below the top -- with the region (the harmonic part weighted down beyond it, engine.rs # Beyond
/// the trap's region) 3 at 1.25 us and 1 at 0.31 us, the one at every step down to 0.125 us; without it
/// 9 and 3, none at 0.125 us; 20 at the 2 us the line took with the thermal ensemble before it.
#[test]
fn a_shallow_trap_s_dark_atoms_stay_whatever_the_step(){
    use rayon::prelude::*;
    let fixture:Value=serde_json::from_str(include_str!("../../tests/fixtures/shallow_probe_motion.json")).unwrap();
    let mut r=scene::request(&fixture["payload"]);let(duration,outputs,count)=(500.,26,1024u64);
    r.times_us=(0..outputs).map(|i|duration*i as f64/(outputs-1) as f64).collect();
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();
    let interval=duration/(outputs-1) as f64;
    let(cluster,state,_)=initial.states[reference];let depth=initial.ensembles[reference].as_ref().unwrap().depth;let hm=model.hbar_over_mass;
    // Each trajectory's first sample it is lost by (None: held to the end) and its initial energy (in D),
    // `steps` steps per sample.
    let lost=|steps:usize,region:f64|->Vec<(Option<usize>,f64)>{
        let mut s=Settings::new(interval/steps as f64,&initial.traps[reference],f64::EPSILON,true);s.region=region;let engine=Engine::new(&model,s.clone());
        (0..count).into_par_iter().map(|index|{let mut t=initial_trajectory(&model,&s,&initial,0x3D_5EED,index,Stratum::single(index));
            let energy=(0.5*hm*t.kappa.iter().map(|k|k*k).sum::<f64>()+model.trap_level(cluster,state,t.r))/depth;
            for i in 1..outputs{for _ in 0..steps{if t.alive{engine.step(&mut t);}}if !t.alive{return(Some(i),energy);}}(None,energy)}).collect()};
    for region in [0.,initial.depths[reference]]{
        let(coarse,fine)=(lost(16,region),lost(64,region));
        let total=|v:&[(Option<usize>,f64)]|v.iter().filter(|x|x.0.is_some()).count() as f64;
        let dark:Vec<f64>=coarse.iter().filter(|x|x.0.map_or(false,|i|i>=2)).map(|x|x.1).collect();
        let discordant=coarse.iter().zip(&fine).filter(|(a,b)|a.0.is_some()!=b.0.is_some()).count() as f64;
        println!("region {region:.1}: lost {} at 1.25 us ({} in the dark, from E/D {dark:.3?}), {} at 0.31 us, {discordant} discordant",total(&coarse),dark.len(),total(&fine));
        assert!(dark.iter().all(|&e|e>-0.05),"region {region}: dark atoms lost from below 5 % of the depth: {dark:?}");
        assert!((total(&coarse)-total(&fine)).abs()<=3.*discordant.sqrt(),"region {region}: lost {} at 1.25 us, {} at 0.31 us, {discordant} discordant",total(&coarse),total(&fine));
    }
}

/// The same shallow trap's step (solve.rs, # Time step): the line took 2 us, where the impulse's and the
/// populations' criteria are met and T is 2.1–2.7 % high against 0.25 us (4096 trajectories on shared
/// streams, its share 1 %); T's own step error, the orbits' over a trap period (`orbits`), bounds the step (0.6–0.9 %
/// at 1 us, as dt²; the step's share s of T's room beside its a-priori estimates, solve.rs # Shares, now about 2 %
/// there), and T over the outputs lies within its share (at most s of the temperature target) of a quarter of that
/// step, plus three standard errors, on shared streams.
#[test]
fn a_shallow_trap_s_step_holds_t_within_its_share(){
    let fixture:Value=serde_json::from_str(include_str!("../../tests/fixtures/shallow_probe_motion.json")).unwrap();
    let mut r=scene::request(&fixture["payload"]);let(duration,outputs,count)=(500.,26,1024u64);
    r.times_us=(0..outputs).map(|i|duration*i as f64/(outputs-1) as f64).collect();
    let dt=solve(&r,Device::Cpu,&mut |_,_|{}).unwrap().time_step_us;
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();let trap=initial.traps[reference];
    let interval=duration/(outputs-1) as f64;let steps=(interval/dt).round() as usize;
    let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab};
    // T per output at `steps` per interval, the solve's seed and groups (the same trajectories at either step).
    let temperatures=|steps:usize|{
        let mut s=Settings::new(interval/steps as f64,&trap,f64::EPSILON,true);s.region=initial.depths[reference];
        let harmonic=Harmonic::new(&model,&s,&initial,steps,outputs);
        let outcome=simulate(&model,&s,&initial,0x3D_5EED,0,count,8,steps,outputs,&[],Some(&Variates{harmonic:&harmonic,twin:None}));
        let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:harmonic.mean[i],twin:None}).collect();
        summarize(&outcome.groups,&scales,&means)};
    let(coarse,fine)=(temperatures(steps),temperatures(4*steps));
    // Per axis the mean ΔT/T over the outputs and the mean of its standard errors (the outputs' errors are
    // correlated along the trajectories: the mean's is at most theirs).
    let share=BIAS_SHARE*TEMPERATURE_TARGET;let n=(outputs-1) as f64;
    for a in 0..3{
        let(d,e)=(1..outputs).fold((0.,0.),|(d,e),i|{let t=fine[i].temperature[a];(d+(coarse[i].temperature[a]-t)/t,e+coarse[i].temperature_error[a].hypot(fine[i].temperature_error[a])/t)});
        println!("axis {a}: dt {dt:.3} us against {:.3} us, mean ΔT/T {:+.2} % ± {:.2} %",dt/4.,100.*d/n,100.*e/n);
        assert!((d/n).abs()<=share+3.*e/n,"axis {a}: mean ΔT/T {:.2} % ± {:.2} %",100.*d/n,100.*e/n);
    }
}

/// The shallow trap's initial ensemble (rust-solver/tests/fixtures/shallow_probe_motion.json: the
/// default scene's tweezer at 0.6 mW, 128 uK deep, the atom at 30 uK, its probe): the bound states only
/// (ensemble.rs) -- the thermal distribution F7 drew left 15 % of the atoms above the trap's top, held
/// until they left its region. 4096 initial states in the solve's groups of eight: none unbound by the
/// loss's own measure (the trap potential of its state, engine.rs # Loss), and their kinetic
/// temperature per lab axis the ensemble's exact one (`Initial::temperature`: 16.8 uK, the shallow
/// trap's bound atoms' -- none of its bound ensembles is hotter than 0.166 D/k_B = 21 uK) within four
/// standard errors.
#[test]
fn a_shallow_trap_s_initial_atoms_are_bound(){
    use rayon::prelude::*;
    let fixture:Value=serde_json::from_str(include_str!("../../tests/fixtures/shallow_probe_motion.json")).unwrap();
    let r=scene::request(&fixture["payload"]);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();
    let settings=Settings::new(0.25,&initial.traps[reference],f64::EPSILON,true);let hm=model.hbar_over_mass;let count=4096u64;
    let drawn:Vec<(f64,[f64;3])>=(0..count).into_par_iter().map(|i|{
        let t=initial_trajectory(&model,&settings,&initial,0x3D_5EED,i,Stratum{first:i/8*8,size:8});
        (0.5*hm*t.kappa.iter().map(|k|k*k).sum::<f64>()+model.trap_potential(t.cluster,&model.point(t.r,t.kappa),&t.psi,0.),t.kappa.map(|k|k*k*hm/KT_PER_UK))}).collect();
    let unbound=drawn.iter().filter(|d|d.0>=0.).count();
    let exact=initial.temperature();let n=count as f64;
    for a in 0..3{
        let m=drawn.iter().map(|d|d.1[a]).sum::<f64>()/n;let se=(drawn.iter().map(|d|(d.1[a]-m).powi(2)).sum::<f64>()/(n-1.)/n).sqrt();
        println!("axis {a}: kinetic T {m:.3} ± {se:.3} uK against {:.3}",exact[a]);
        assert!((m-exact[a]).abs()<=4.*se,"axis {a}: kinetic T {m} ± {se} against {}",exact[a]);
    }
    assert_eq!(unbound,0,"{unbound} of {count} initial atoms unbound");
    assert!(exact.iter().all(|&t|t<0.6*r.temperature_uk),"{exact:?}");
}

/// The shallow trap's bound ensemble is an equilibrium of the conservative motion (Jeans' theorem, its
/// distribution a function of the energy alone, ensemble.rs): the fixture's tweezer alone at 30 uK, 4.3
/// k_BT deep, 1024 trajectories over 2 ms (nine periods of its slowest axis) at 0.5 us and 0.125 us, the
/// solve's region: T on every axis keeps its initial value within four standard errors at every one of
/// the 126 samples (each axis' at a time correlated along the trajectories), and no atom is lost -- the barely bound ones far out on the axis too, where the harmonic part about
/// the focus is weighted down (engine.rs, # Beyond the trap's region; without the weight 4 of 1024 were
/// lost at 0.25 us, every one within 0.2 % of the depth below the top).
#[test]
fn a_shallow_trap_s_bound_ensemble_stays_with_the_light_off(){
    let fixture:Value=serde_json::from_str(include_str!("../../tests/fixtures/shallow_probe_motion.json")).unwrap();
    let mut r=scene::request(&fixture["payload"]);let trap_id=fixture["payload"]["model"]["motion"]["trapBeamId"].as_str().unwrap().to_owned();
    r.beams.retain(|b|b.id==trap_id);r.trap_beam=0;
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();let trap=initial.traps[0];let hm=model.hbar_over_mass;
    assert!(initial.kinetic()<0.6,"{}",initial.kinetic());
    let exact=initial.temperature();
    for(dt,steps)in [(0.5,200),(0.125,800)]{
        let mut settings=Settings::new(dt,&trap,f64::EPSILON,false);settings.region=initial.depths[0];
        let outcome=simulate(&model,&settings,&initial,5,0,1024,8,steps,21,&[],None);assert_eq!(outcome.jumps,0);
        let e=summarize(&outcome.groups,&Scales{hbar_over_mass:hm,frequencies:trap.lab},&[]);
        for(i,x)in e.iter().enumerate(){
            assert_eq!(x.loss,0.,"dt {dt}, output {i}");
            for a in 0..3{assert!((x.temperature[a]-exact[a]).abs()<=4.*x.temperature_error[a],"dt {dt}, output {i}, axis {a}: {} ± {} against {}",x.temperature[a],x.temperature_error[a],exact[a]);}
        }
    }
}

/// A second far beam crossing the tweezer (the validation's F9, its far beams alone) is part of the trap the bound
/// ensemble is at equilibrium in (ensemble.rs): its depth the two beams', T stationary from t = 0 -- at the trap beam's
/// own equilibrium the cloud, far too long along the tweezer's axis for the crossed trap, breathed from 19 up to 360 uK
/// within 4 us. A few k_BT deep the draws from the two beams' mixture have the quadrature's kinetic temperature, and
/// every atom is bound.
#[test]
fn a_crossed_far_beam_is_part_of_the_bound_ensemble(){
    let fixture:Value=serde_json::from_str(include_str!("fixtures/crossed_far_beam_motion.json")).unwrap();
    let mut r=scene::request(&fixture);r.times_us=vec![0.,1.];
    let model=Model::new(&r).unwrap();assert_eq!(model.trap_modes().len(),2);
    let initial=Initial::new(&model,&r).unwrap();let trap=initial.traps[0];let(cluster,state,_)=initial.states[0];let hm=model.hbar_over_mass;
    let e=initial.ensembles[0].as_ref().unwrap();assert!((e.depth/initial.depths[0]-1.).abs()<=1e-6,"{} {}",e.depth,initial.depths[0]);
    let exact=initial.temperature();
    let mut settings=Settings::new(0.05,&trap,f64::EPSILON,false);settings.region=initial.depths[0];
    let outcome=simulate(&model,&settings,&initial,5,0,1024,8,20,11,&[],None);assert_eq!(outcome.jumps,0);
    let estimates=summarize(&outcome.groups,&Scales{hbar_over_mass:hm,frequencies:trap.lab},&[]);
    for(i,x)in estimates.iter().enumerate(){assert_eq!(x.loss,0.);
        for a in 0..3{assert!((x.temperature[a]-exact[a]).abs()<=4.*x.temperature_error[a],"output {i}, axis {a}: {} ± {} against {}",x.temperature[a],x.temperature_error[a],exact[a]);}}
    // A few k_BT deep.
    let kt=initial.depths[0]/4.;r.temperature_uk=kt/KT_PER_UK;
    let model=Model::new(&r).unwrap();
    let initial=Initial::new(&model,&r).unwrap();let settings=Settings::new(0.05,&initial.traps[0],f64::EPSILON,false);
    let draws:Vec<(f64,f64)>=(0..16384u64).map(|i|{let t=initial_trajectory(&model,&settings,&initial,9,i,Stratum::single(i));
        let kinetic=0.5*hm*t.kappa.iter().map(|k|k*k).sum::<f64>();(kinetic,model.trap_level(cluster,state,t.r))}).collect();
    assert!(draws.iter().all(|x|x.0+x.1<0.),"an unbound atom");
    let n=draws.len() as f64;let mean=draws.iter().map(|x|x.0).sum::<f64>()/n;let se=(draws.iter().map(|x|(x.0-mean).powi(2)).sum::<f64>()/(n-1.)/n).sqrt();
    let wanted=1.5*kt*initial.kinetic();assert!((mean-wanted).abs()<=4.*se,"kinetic energy {mean} ± {se} against the quadrature's {wanted}");
}

/// Far beams alone leave a ground state lone in its cluster, its H nothing but the rest of its table entry's rounding (the
/// imaginary part 3e-20 of 70 rad/us at a node, under the crossed beams' complex fields): at every temperature, the reach's
/// nodes moving with it, that rounding is no gain and the guard passes the model (model.rs `Model::lindblad`).
#[test]
fn a_lone_state_s_round_off_is_no_gain(){
    let fixture:Value=serde_json::from_str(include_str!("fixtures/crossed_far_beam_motion.json")).unwrap();
    let mut r=scene::request(&fixture);r.times_us=vec![0.,1.];
    let model=Model::new(&r).unwrap();let depth=Initial::new(&model,&r).unwrap().depths[0];
    for f in [40.,20.,10.,8.,6.,5.,4.,3.,2.]{r.temperature_uk=depth/f/KT_PER_UK;
        Model::new(&r).unwrap_or_else(|e|panic!("k_BT = D/{f}: {e}"));}
}

/// The near light comes on at t = 0, and its fold's shift is no part of the ensemble's energy: on the tweezer PGC's
/// all-folded model the 10 MHz probe's denominator closes on a shell about the focus, where with it the envelope's top
/// stood at 137 k_BT against the focus's 34 and no proposal in e^100 was kept (the solve never reached its first
/// result). The envelope sits at the focus's depth, and the draws take no time.
#[test]
fn a_near_probe_s_fold_stays_out_of_the_ensemble_s_energy(){
    let fixture:Value=serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap();
    let mut r=scene::request(&fixture);r.times_us=vec![0.,1.];
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let e=initial.ensembles[0].as_ref().unwrap();let ratio=e.top/(e.depth/(KT_PER_UK*r.temperature_uk));
    assert!(ratio<=1.01,"the envelope's top {ratio} times the depth");
    let settings=Settings::new(0.05,&initial.traps[0],f64::EPSILON,false);
    for i in 0..256u64{let t=initial_trajectory(&model,&settings,&initial,9,i,Stratum::single(i));assert!(t.r.iter().all(|x|x.is_finite()));}
}

/// A standing wave's neighbouring sites: their atoms start about their own sites' bottoms (lib.rs `initial_trajectory`:
/// scaled about the focus site 1's stood 67 times as high up its well as the centre's), and the harmonic part about each
/// atom's own site (engine.rs, # A standing wave's sites) keeps T, no atom lost, at a step of ω dt = 1.5: the kick
/// no longer takes the site's stiffness (about the focus T_x rose to 3.6 times at a step of that order).
#[test]
fn a_standing_wave_s_sites_hold_their_atoms_as_the_centre_does(){
    let(_,mut r)=retro(0.);r.times_us=vec![0.,1.];let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let trap=initial.traps[0];let(cluster,state,_)=initial.states[0];let hm=model.hbar_over_mass;
    let lattice=trap.lattice.expect("the tweezer and its reflection are a standing wave");
    let settings=Settings::new(0.01,&trap,f64::EPSILON,false);
    let mut levels=[(0.,0_u32);2];
    for i in 0..8192u64{let t=initial_trajectory(&model,&settings,&initial,9,i,Stratum::single(i));
        let z=(0..3).map(|a|t.r[a]*lattice.axis[a]).sum::<f64>();let n=lattice.nearest(z).0;let m=n.abs() as usize;if m>1{continue;}
        let bottom:[f64;3]=std::array::from_fn(|a|t.r[a]+(lattice.site(n)-z)*lattice.axis[a]);
        levels[m].0+=model.trap_level(cluster,state,t.r)-model.trap_level(cluster,state,bottom);levels[m].1+=1;}
    let(centre,neighbours)=(levels[0].0/levels[0].1 as f64,levels[1].0/levels[1].1 as f64);
    assert!(levels[1].1>64&&(neighbours/centre-1.).abs()<0.3,"{} atoms' level above their bottoms at sites ±1 {neighbours}, at the centre {centre}",levels[1].1);
    let exact=initial.temperature();let dt=1.5/trap.frequencies.iter().copied().fold(0.,f64::max);
    let mut s=Settings::new(dt,&trap,f64::EPSILON,false);s.region=initial.depths[0];
    let steps=(1./dt).ceil() as usize;s.dt=1./steps as f64;
    let outcome=simulate(&model,&s,&initial,5,0,1024,8,steps,11,&[],None);assert_eq!(outcome.jumps,0);
    let estimates=summarize(&outcome.groups,&Scales{hbar_over_mass:hm,frequencies:trap.lab},&[]);
    for(i,x)in estimates.iter().enumerate(){assert_eq!(x.loss,0.);
        for a in 0..3{assert!((x.temperature[a]-exact[a]).abs()<=5.*x.temperature_error[a],"output {i}, axis {a}: {} ± {} against {}",x.temperature[a],x.temperature_error[a],exact[a]);}}
}

/// The fixture's tweezer alone and with a reflection along its axis, in phase at the focus (F5): the
/// reflection's field the tweezer's with `turn` of the cooling beam's polarization added (relative to
/// the tweezer's coupling's size, its outside shift the tweezer's).
fn retro(turn:f64)->(Request,Request){
    let alone=request(&|b|b["id"]=="tweezer");let cool=request(&|b|b["id"]=="cool-plus").beams[0].clone();
    let mut reflection=alone.beams[0].clone();(reflection.id,reflection.name)=("reflection".into(),"反射光".into());reflection.direction=reflection.direction.map(|x|-x);
    let norm=|m:&Matrix|m.real.iter().zip(&m.imag).map(|(a,b)|a*a+b*b).sum::<f64>().sqrt();let scale=turn*norm(&reflection.coupling)/norm(&cool.coupling);
    for(m,c)in [(&mut reflection.coupling,&cool.coupling),(&mut reflection.counter_coupling,&cool.counter_coupling)]{for k in 0..STATES*STATES{m.real[k]+=scale*c.real[k];m.imag[k]+=scale*c.imag[k];}}
    let mut retro=alone.clone();retro.beams.push(reflection);(alone,retro)
}

/// A retro-reflected tweezer folds as its standing wave (F5, ground.rs `Unit`): the fixture's tweezer
/// and its reflection, one field, shift the states as the tweezer alone at twice its amplitude does at
/// their summed field's intensity |a₁ + a₂|²/4 -- four times the tweezer's own shift at the focus, where
/// the release folded each beam on its own (twice), and cos²(kζ) along the axis --, to round-off at
/// every point; their trap along the axis is the standing wave's lattice site, tens of times the
/// tweezer's 19 kHz.
#[test]
fn a_retro_reflected_tweezer_folds_as_its_standing_wave(){
    let(alone,retro)=retro(0.);
    let mut double=alone.clone();
    {let b=&mut double.beams[0];for(m,x)in [(&mut b.coupling,2.),(&mut b.counter_coupling,2.),(&mut b.outside_shift,4.)]{for v in m.real.iter_mut().chain(m.imag.iter_mut()){*v*=x;}}}
    let(g,reference)=(Ground::new(&retro).unwrap(),Ground::new(&double).unwrap());
    let n=retro.beams[0].direction;
    for r in (0..9).map(|i|n.map(|x|0.05*i as f64*x)).chain([[0.3,0.1,0.2],[-0.2,0.37,0.1]]){
        let a=g.amplitudes_at(r);let(h,expected)=(g.hamiltonian_at(r),reference.hamiltonian_with(&[(a[0]+a[1]).norm_sqr()/4.]));
        let mut d=h.clone();d.add_scaled(-Complex::ONE,&expected);
        assert!(d.largest()<=1e-12*expected.largest(),"r {r:?}: {:.2e} of {:.2e}",d.largest(),expected.largest());
    }
    let along=|r:&Request|{let model=Model::new(r).unwrap();Initial::new(&model,r).unwrap().traps[0].lab[1]/std::f64::consts::TAU*1e3};
    let(single,lattice)=(along(&alone),along(&retro));
    assert!(lattice>30.*single,"the trap along the beam: {lattice:.1} kHz against the tweezer's {single:.1} kHz");
}

/// A reflection a little off the tweezer's polarization (a millionth of the cooling beam's added) is
/// the standing wave's unit with its remainder booked (ground.rs `coherence`), to its smallness: its
/// shifts within 1e-5 of the exact reflection's at every point and its interference item far within
/// the budget -- where the beams taken apart would book the whole interference. A polarization that
/// differs (as much of the cooling beam's) books an interference beyond the budget: the solve is
/// flagged, not silently wrong.
#[test]
fn a_reflection_a_little_off_the_tweezer_s_polarization_is_its_standing_wave_within_its_bound(){
    let(_,exact)=retro(0.);let(_,turned)=retro(1e-6);let(_,crossed)=retro(1.);
    let(e,g,c)=(Ground::new(&exact).unwrap(),Ground::new(&turned).unwrap(),Ground::new(&crossed).unwrap());
    let estimate=|g:&Ground|g.interference.iter().map(|x|x.estimate(g)).fold(0.,f64::max);
    assert!(g.units.iter().map(|u|u.beams.clone()).eq([vec![0,1]])&&g.interference[0].coherent,"{:?}",g.interference);
    assert!(estimate(&g)<=1e-2*BUDGET,"the remainder's item {:.2e}",estimate(&g));
    assert!(estimate(&c)>BUDGET,"the crossed polarizations' item {:.2e}",estimate(&c));
    let n=exact.beams[0].direction;
    for r in (0..9).map(|i|n.map(|x|0.05*i as f64*x)).chain([[0.3,0.1,0.2]]){
        let(h,reference)=(g.hamiltonian_at(r),e.hamiltonian_at(r));let mut light=reference.clone();light.add_scaled(-Complex::ONE,&e.hamiltonian);
        let mut d=h.clone();d.add_scaled(-Complex::ONE,&reference);
        assert!(d.largest()<=1e-5*light.largest(),"r {r:?}: {:.2e} of {:.2e}",d.largest(),light.largest());
    }
}

/// A fixture's motion request, its matrices sparse ([row, column, re, im]), its initial state diagonal
/// (tests/fixtures/*_probe_motion.json: validate/probe_scan payloads with the population core's rows
/// recorded at the focus).
fn sparse_request(v:&Value)->Request{
    let beams=v["beams"].as_array().unwrap().iter().map(|b|{let d=floats(&b["direction"]);Beam{id:b["id"].as_str().unwrap().into(),name:b["name"].as_str().unwrap().into(),
        direction:[d[0],d[1],d[2]],wave_number_per_um:b["waveNumberPerUm"].as_f64().unwrap(),waist_um:b["waistUm"].as_f64(),intensity_mw_cm2:b["intensityMWcm2"].as_f64().unwrap(),
        frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),coupling:matrix(&b["coupling"]),
        counter_coupling:matrix(&b["counterCoupling"]),outside_shift:matrix(&b["outsideShift"])}}).collect::<Vec<Beam>>();
    let trap_beam=beams.iter().position(|b|b.id==v["trapBeamId"].as_str().unwrap()).unwrap();
    let initial=floats(&v["initial"]);let mut density=vec![Complex::ZERO;STATES*STATES];for i in 0..STATES{density[i*STATES+i]=Complex::new(initial[i],0.);}
    let(e,f)=(v["emission"].as_array().unwrap(),floats(&v["fieldVectorG"]));
    Request{states:v["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect(),hamiltonian:matrix(&v["hamiltonian"]),
        frame_frequencies_mhz:floats(&v["frameFrequenciesMHz"]),frequencies_mhz:floats(&v["frequenciesMHz"]),beams,emission:[matrix(&e[0]),matrix(&e[1]),matrix(&e[2])],
        field_vector_g:[f[0],f[1],f[2]],mass_kg:v["massKg"].as_f64().unwrap(),initial_density:density,temperature_uk:v["temperatureUK"].as_f64().unwrap(),trap_beam,times_us:vec![]}
}
/// The frozen atom's F = 1 population each microsecond, its eliminated master equation's at the focus:
/// the held twin's exact means there (control.rs), every drive folded, at second or fourth order.
fn frozen_f1(r:&Request,fourth:bool)->Vec<f64>{
    let model=Model::with_order(r,f64::INFINITY,fourth).unwrap();let twin=Twin::new(&model,1.).expect("a static generator");assert_eq!(twin.held.point.r,[0.;3]);
    twin.means(&model,&r.initial_density,1.,11).iter().map(|row|(0..STATES).filter(|&i|r.states[i].starts_with("ground-1-")).map(|i|row[i]).sum()).collect()
}

/// A strong beam 3 GHz off a probe's line dresses the probe (F10; ground.rs, # The near light's
/// dressing): with the folded pairs' far channels in the responses' denominators -- where the release
/// added their first order outside the jumps, 3.2 rad/us of gain (tests/lindblad.rs) -- the frozen
/// atom's F = 1 population, the strong beam's Raman scattering, is the population core's at every
/// microsecond to 0.005 (0.2008 at 10 us; the release's 0.000).
#[test]
fn a_strong_beam_s_raman_rate_through_its_dressing_is_the_core_s(){
    let v:Value=serde_json::from_str(include_str!("fixtures/far_beam_probe_motion.json")).unwrap();let r=sparse_request(&v);
    for(i,(a,b))in frozen_f1(&r,false).iter().zip(floats(&v["core"]["f1"])).enumerate(){assert!((a-b).abs()<=5e-3,"{i} us: {a} against the core's {b}");}
}

/// An 808 nm tweezer, a near carrier, moves a D1 probe's line by its light shift (F1; ground.rs, # The
/// near light's dressing): the probe 32 MHz above the bare line is on the shifted one, and the frozen
/// atom's F = 1 population is the population core's at every microsecond to 0.005 (0.549 at 10 us;
/// the release's line sat at +8 MHz, 0.98 here).
#[test]
fn a_near_tweezer_s_shift_moves_a_probe_s_line_to_the_core_s(){
    let v:Value=serde_json::from_str(include_str!("fixtures/near_tweezer_probe_motion.json")).unwrap();let r=sparse_request(&v);
    for(i,(a,b))in frozen_f1(&r,false).iter().zip(floats(&v["core"]["f1"])).enumerate(){assert!((a-b).abs()<=5e-3,"{i} us: {a} against the core's {b}");}
}

/// The fourth order on a dressed probe (ground.rs, # Fourth order): its W3 without what the near light's dressing
/// already holds -- Σ's first order through the far channels, z's light shifts by the other carriers -- and with the
/// dressed denominators, the frozen atom's F = 1 population is the population core's at every microsecond to 0.005,
/// as the second order's: the strong beam 3 GHz off the probe's line (F10, a carrier of the fourth order: 0.0013 at
/// most) and the 808 nm tweezer (F1, below the budget's square, the probe's dressing alone: 0.0029). The line's fourth
/// order counted the dressing twice where the dressing's carrier is the fourth order's: F10's 0.0100 off at 2 us.
#[test]
fn a_dressed_probe_at_fourth_order_is_the_core_s(){
    for fixture in [include_str!("fixtures/far_beam_probe_motion.json"),include_str!("fixtures/near_tweezer_probe_motion.json")]{
        let v:Value=serde_json::from_str(fixture).unwrap();let r=sparse_request(&v);
        for(i,(a,b))in frozen_f1(&r,true).iter().zip(floats(&v["core"]["f1"])).enumerate(){assert!((a-b).abs()<=5e-3,"{i} us: {a} against the core's {b}");}
    }
}

/// b1/109's fourth order, every drive folded (its F = 2 sublevels −2..0 one cluster, the strong beam's Zeeman pairs
/// folded): the folded photons a class joins without a kept coupling (ground.rs `FoldedCross`) meet in the jumps' products
/// at fourth order too -- through the metric's bend and W3 --, and so do the photons of the carriers below the budget's
/// square sharing their classes. The tables' decay takes them over the landing states the class shares
/// (model.rs `FourthLayout::landing`), the triples' frequencies from their carriers' integer combinations: at the focus,
/// a node, the generator is a Lindblad one to round-off at t = 0 and at a generic instant. It lacked 2.4e-3 rad/us of the
/// jumps' cross terms (of 1.1).
#[test]
fn a_fourth_order_decay_holds_its_folded_photons_cross_terms(){
    let v:Value=serde_json::from_str(include_str!("fixtures/explicit_rung_motion.json")).unwrap();let r=sparse_request(&v);
    let model=Model::with_order(&r,f64::INFINITY,true).unwrap();let p=model.point([0.;3],[0.;3]);
    for k in 0..model.clusters(){for t in [0.,0.618]{
        let(low,_,scale)=model.decay_balance(k,&p,t);let round=64.*f64::EPSILON*model.dimension(k) as f64*scale;
        assert!(low>=-round,"cluster {k} at t = {t}: i(H − H†) − Σ L̃†L̃ reaches {low:.3e} rad/us (round-off {round:.1e})");
    }}
}

/// b1/109's explicit rung: its F = 2 sublevels joined through D1 and D2 explicit states whose frames carry the D1
/// carrier's detuning, −4.5e7 rad/us in the D2-framed excited manifold. Each frame is its first state's energy plus an
/// integer combination of the carriers' detunings (ground.rs `Cluster::combinations`): two sublevels' frames along
/// θ + δ − δ are equal, not 3.2e-9 rad/us apart, and the residuals the photons rotate at are the generator's beats.
/// At the focus, a node, the generator is a Lindblad one to round-off at a generic instant; the drift was 3.7e-9
/// rad/us (of 38) at t = 0.618 us, growing with t.
#[test]
fn an_explicit_rung_s_frames_are_exact(){
    let v:Value=serde_json::from_str(include_str!("fixtures/explicit_rung_motion.json")).unwrap();let mut r=sparse_request(&v);
    // The case's own run, 100 us in 21 samples: its duration sets the rule's resonance floor and the rung.
    r.times_us=(0..21).map(|i|5.*i as f64).collect();
    let allowance=Ground::allowances(&r).unwrap()[1];let model=Model::with(&r,allowance).unwrap();let p=model.point([0.;3],[0.;3]);
    for k in 0..model.clusters(){for t in [0.618,61.8]{
        let(low,_,scale)=model.decay_balance(k,&p,t);let round=64.*f64::EPSILON*model.dimension(k) as f64*scale;
        assert!(low>=-round,"cluster {k} at t = {t}: i(H − H†) − Σ L†L reaches {low:.3e} rad/us (round-off {round:.1e})");
    }}
}

/// The phase a solve measures for a beating grid (solve.rs `grid_phase`) lies far above the quadrature's bound, and a step
/// at it meets both shares against a grid far finer than the bound's, by another path than the doubling that chose it: the
/// whole step from thermal states a few microseconds into the PGC with a resonant probe, its 12-state cluster's beats at 36.4 and 72.8 MHz.
#[test]
fn a_beating_grid_s_measured_phase_meets_its_shares(){
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..101).map(|i|i as f64).collect();
    let model=Model::with(&r,Ground::allowances(&r).unwrap()[1]).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let(dt,waiting)=(0.25,0.6);let share=NUMERICAL_SHARE*POPULATION_TARGET*dt/waiting;
    let mut at=Settings::new(dt,&initial.traps[0],share,true);at.horizon=waiting;let engine=Engine::new(&model,at.clone());
    let states:Vec<Trajectory>=(0..64).map(|i|{let mut t=initial_trajectory(&model,&at,&initial,7,i,Stratum::single(i));for _ in 0..8{engine.step(&mut t);}t}).collect();
    let phase=grid_phase(&model,&states,&at);assert!(phase>4.*at.node_phase,"{phase} against the bound's {}",at.node_phase);
    let(mut coarse,mut fine)=(at.clone(),at.clone());coarse.node_phase=phase;fine.node_phase=at.node_phase/2.;
    let(coarse,fine)=(Engine::new(&model,coarse),Engine::new(&model,fine));
    let(mut populations,mut impulse,mut size)=(0.,0.,0.);
    for t in &states{
        let n=model.dimension(t.cluster);let(a,b)=(coarse.coast(t,dt,1),fine.coast(t,dt,1));
        let norm=|x:&Trajectory|x.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>();let(x,y)=(norm(&a),norm(&b));
        populations+=(0..n).map(|i|(a.psi[i].norm_sqr()/x-b.psi[i].norm_sqr()/y).abs()).fold(0.,f64::max);
        impulse+=(0..3).map(|k|(a.kappa[k]-b.kappa[k]).powi(2)).sum::<f64>().sqrt();size+=(0..3).map(|k|(b.kappa[k]-t.kappa[k]).powi(2)).sum::<f64>().sqrt();
    }
    let m=states.len() as f64;assert!(populations/m<=share,"populations {:e} against the share {share:e}",populations/m);
    assert!(impulse/size<=QUADRATURE,"the impulse {:e} against {QUADRATURE:e}",impulse/size);
}

/// The ladder's rungs keep more as the allowance falls, and the closure keeps what it must first: on the PGC with a resonant probe
/// the first rung holds the probe's five drives and the strongest cooling drive, the cooling beams' others folded. (With the
/// closure's estimates taken under the folded probe's light, and no rung above each ground state's first kept drive, the first rung
/// held 7 excited states, its T 5 % above the fully explicit F = 2 cluster's at 100 us, where this rung's is within its 2 % of statistics.)
#[test]
fn the_ladder_keeps_more_as_the_allowance_falls_and_the_near_drives_first(){
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..101).map(|i|i as f64).collect();
    let ladder=Ground::allowances(&r).unwrap();assert!(ladder[0]==f64::INFINITY&&ladder.windows(2).all(|w|w[0]>w[1]),"{ladder:?}");
    let grounds:Vec<Ground>=ladder.iter().take(12).map(|&a|Ground::with(&r,a).unwrap()).collect();
    let explicit:Vec<usize>=grounds.iter().map(|g|g.clusters.iter().map(|c|c.excited.len()).sum()).collect();
    assert!(explicit[0]==0&&explicit.windows(2).all(|w|w[0]<=w[1]),"explicit states along the ladder: {explicit:?}");
    let g=&grounds[1];let drives:Vec<usize>=g.excited_items.iter().filter(|x|!x.folded&&x.size>1e-6).filter_map(|x|match x.kind{ExcitedKind::Drive(d)=>Some(d.carrier),_=>None}).collect();
    let probe=drives.iter().filter(|&&c|c==drives[0]).count();
    assert!(explicit[1]==5&&drives.len()==6&&(probe==5||probe==1),"the first rung: {} explicit states, its drives' carriers {drives:?}",explicit[1]);
}

/// The atom frozen at the focus: its eliminated master equation there (ground.rs `local_at`, every drive
/// folded, the beats' phases at each instant), RK4 at 1/250 of the outputs' interval, the populations
/// at the given times up to `last`.
fn frozen_rows(r:&Request,times:&[f64],last:usize)->Vec<[f64;STATES]>{
    use atom_motion_3d::linear::Dense;
    let model=Model::new(r).unwrap();let g=&model.ground;let point=[0.;3];
    let e=g.eliminate(&model.intensities_at_point(point)).unwrap();
    let locals=|t:f64|->Vec<_>{(0..g.clusters.len()).map(|k|g.local_at(&e,k,point,t).unwrap()).collect()};
    let mut rho:Vec<Dense>=g.clusters.iter().map(|c|{let n=c.dimension();Dense::from_fn(n,n,|i,j|if i==j&&i<c.states.len(){let s=g.ground_states[c.states[i]];r.initial_density[s*STATES+s]}else{Complex::ZERO})}).collect();
    let derivative=|t:f64,rho:&[Dense]|->Vec<Dense>{
        let x=locals(t);
        let mut out:Vec<Dense>=rho.iter().zip(&x).map(|(p,l)|{let h=&l.effective;let mut d=h.product(p).scaled(Complex::new(0.,-1.));d.add_scaled(Complex::new(0.,1.),&p.product(&h.adjoint()));d}).collect();
        for(k,l)in x.iter().enumerate(){for(class,jumps)in g.clusters[k].classes.iter().zip(&l.jumps){for j in jumps{out[class.target].add_scaled(Complex::ONE,&j.product(&rho[k]).product(&j.adjoint()));}}}
        out
    };
    let add=|a:&[Dense],b:&[Dense],s:f64|->Vec<Dense>{a.iter().zip(b).map(|(x,y)|{let mut z=x.clone();z.add_scaled(Complex::new(s,0.),y);z}).collect()};
    let readout=|t:f64,rho:&[Dense]|{let mut p=[0.;STATES];for(k,l)in locals(t).iter().enumerate(){for(a,b)in p.iter_mut().zip(g.populations(k,l,&rho[k])){*a+=b;}}p};
    let mut rows=vec![readout(times[0],&rho)];let mut t=times[0];
    for i in 1..=last{
        let steps=250;let h=(times[i]-times[i-1])/steps as f64;
        for _ in 0..steps{
            let k1=derivative(t,&rho);let k2=derivative(t+h/2.,&add(&rho,&k1,h/2.));let k3=derivative(t+h/2.,&add(&rho,&k2,h/2.));let k4=derivative(t+h,&add(&rho,&k3,h));
            for(j,x)in rho.iter_mut().enumerate(){for(k,w)in [(&k1,1.),(&k2,2.),(&k3,2.),(&k4,1.)]{x.add_scaled(Complex::new(h*w/6.,0.),&k[j]);}}
            t+=h;
        }
        rows.push(readout(times[i],&rho));
    }
    rows
}

/// A circular tweezer across the field mixes the Zeeman sublevels by its vector light shift, 6.7 rad/us
/// against their 0.7 rad/us splitting (F6; ground.rs, # The static ground block): the probes' responses
/// are those of the static block's eigenstates, each at its own energy, and the frozen atom's
/// populations at the focus are the population core's -- every state's, at every output until the
/// core's own error exceeds 0.005, to 0.01 beyond it (the release's 0.066 at 35 us).
#[test]
fn a_tweezer_s_vector_shift_across_the_field_moves_the_responses_to_the_static_block_s_eigenstates(){
    let v:Value=serde_json::from_str(include_str!("fixtures/tensor_tweezer_probe_motion.json")).unwrap();let r=sparse_request(&v);
    let times=floats(&v["core"]["timesUs"]);let errors=floats(&v["core"]["errors"]);
    let last=errors.iter().position(|&x|x>5e-3).unwrap_or(errors.len())-1;
    let rows=frozen_rows(&r,&times,last);let mut worst=(0.,0.);
    for(i,(row,core))in rows.iter().zip(v["core"]["populations"].as_array().unwrap()).enumerate(){
        if errors[i]>5e-3{continue;}
        let core=floats(core);let error=(0..STATES).map(|s|(row[s]-core[s]).abs()).fold(0.,f64::max);
        println!("{} us: {error:.4}",times[i]);if error>worst.0{worst=(error,times[i]);}
    }
    assert!(worst.0<=1e-2+5e-3,"the frozen atom's populations {:.4} off the core's at {} us",worst.0,worst.1);
}

/// Tables whose function is not smooth on any grid's scale where the atoms go -- a folded drive's pole, the PGC with a
/// resonant probe with every drive folded -- stop refining at the first doubling that does not shrink the error, whose next
/// one is predicted to save nothing (model.rs `tabulate`), the last grid kept and its error booked: 33 nodes per axis, its error 1.8.
#[test]
fn tables_that_do_not_converge_stop_refining_with_their_error_booked(){
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..101).map(|i|i as f64).collect();
    let model=Model::with(&r,f64::INFINITY).unwrap();
    println!("refined to {:?} nodes per axis, its error {:.3}, stalled {}",model.table_limit,model.table_error,model.table_stalled);
    assert!(model.table_stalled&&model.table_limit.is_some(),"refined to {:?} nodes per axis (stalled {})",model.table_limit,model.table_stalled);
    assert!(model.table_error>0.,"the error booked: {:.2e}",model.table_error);
}

/// The retro-reflected tweezer with a resonant probe (tests/fixtures/retro_probe_motion.json, the validation's F3
/// repro; its 951 kHz lattice axis turns the velocity by 0.3 rad per step): the step holds a step's jump hazard at
/// its middle, and the heating the jumps carry errs at second order (solve.rs `jumps`) -- at the step the line took,
/// 0.054 us, T −3.1 ± 1.9 % at its binding outputs against 0.0132 us (4096 trajectories on shared streams), the
/// jumps' net −0.064 µK per jump of their 2.15 against the hazard taken along the step, −0.071 predicted. On states
/// along 32 trajectories at that step the jumps' criterion lies beyond T's share there, and within it at a quarter
/// of it (it goes as dt²).
#[test]
fn the_jumps_criterion_on_a_retro_reflected_tweezer_reads_the_step_as_its_square(){
    let fixture:Value=serde_json::from_str(include_str!("fixtures/retro_probe_motion.json")).unwrap();
    let(outputs,interval)=(21,5.);let mut r=scene::request(&fixture);r.times_us=(0..outputs).map(|i|interval*i as f64).collect();
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();
    let settings=|steps:usize|{let mut s=Settings::new(interval/steps as f64,&initial.traps[reference],f64::EPSILON,true);s.doppler=BUDGET;s.region=initial.depths[reference];s};
    let steps=93;let harmonic=Harmonic::new(&model,&settings(steps),&initial,steps,outputs);
    let(states,_,_)=sample_states(&model,&settings(steps),&initial,0x3D_5EED,32,steps,outputs,&harmonic);
    // A fixed bar: a fifth of the T target, near the criterion at the step the line once took there; a quarter of that step reads a sixteenth of it.
    let bar=0.2*TEMPERATURE_TARGET;
    let(line,quarter)=(jumps(&Engine::new(&model,settings(steps)),&states)/bar,jumps(&Engine::new(&model,settings(4*steps)),&states)/bar);
    println!("the jumps' criterion over the bar: {line:.2} at 0.054 us, {quarter:.2} at 0.0134 us ({} states)",states.len());
    assert!(line>0.5&&line/quarter>8.&&line/quarter<32.,"{line:.3} at the line's step, {quarter:.3} at a quarter");
}

/// A group run only through an output (solve.rs # Batches) sums what the whole run sums there -- T's windows half a period past it
/// included --, and counts no trajectories beyond; the estimates at an output rest on the groups that reach it.
#[test]
fn a_group_run_only_through_an_output_sums_what_the_whole_run_sums_there(){
    let fixture:Value=serde_json::from_str(include_str!("fixtures/retro_probe_motion.json")).unwrap();
    let(outputs,interval,steps)=(11,5.,20);let mut r=scene::request(&fixture);r.times_us=(0..outputs).map(|i|interval*i as f64).collect();
    let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();let trap=initial.traps[reference];
    let mut s=Settings::new(interval/steps as f64,&trap,f64::EPSILON,true);s.doppler=BUDGET;s.region=initial.depths[reference];
    let harmonic=Harmonic::new(&model,&s,&initial,steps,outputs);let v=Variates{harmonic:&harmonic,twin:None};
    let whole=simulate(&model,&s,&initial,0x3D_5EED,0,24,8,steps,outputs,&[],Some(&v));
    let ends=[10,7,3];let cut=simulate_to(&model,&s,&initial,0x3D_5EED,0,24,8,steps,outputs,&[],Some(&v),&ends);
    for(g,&end)in ends.iter().enumerate(){for i in 0..outputs{let(a,b)=(&whole.groups[g].sums[i],&cut.groups[g].sums[i]);
        if i<=end{assert!(a.held==b.held&&a.kappa2==b.kappa2&&a.populations==b.populations&&a.harmonic==b.harmonic&&a.trajectories==b.trajectories,"group {g} output {i}");}
        else{assert!(b.trajectories==0.&&b.held==0.,"group {g} output {i}");}}}
    let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:harmonic.mean[i],twin:None}).collect();let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab};
    let e=summarize(&cut.groups,&scales,&means);
    assert_eq!(e[3].trajectories,24.);assert_eq!(e[4].trajectories,16.);assert_eq!(e[10].trajectories,8.);
}
