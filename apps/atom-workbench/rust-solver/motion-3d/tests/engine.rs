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
use atom_motion_3d::model::{Model,MOST_MIXED,MOST_STATES};
use atom_motion_3d::rng::{normal_quantile,Purpose,Rng};
use atom_motion_3d::stats::{summarize,Means,Scales,KT_PER_UK};
use atom_motion_3d::{initial_trajectory,sample_states,simulate,simulate_to,Initial,Stratum,Variates};
use atom_motion_3d::solve::{bound_probes,drift,grid_phase,jumps,solve,Device,BIAS_SHARE,NUMERICAL_SHARE,POPULATION_TARGET,TEMPERATURE_TARGET};
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
/// their own potential and the kept drives' stimulated force --, at rest and with the kinetic shift's first
/// order at the momentum held (model.rs # The photons and the Doppler term).
#[test]
fn the_force_is_the_energy_s_gradient(){
    for(allowance,doppler)in [(f64::INFINITY,false),(BUDGET,false),(f64::INFINITY,true),(BUDGET,true)]{
    let mut model=Model::with(&request(&|_|true),allowance).unwrap();model.doppler=doppler;
    let energy=|cluster:usize,r:[f64;3],kappa:[f64;3],t:f64,psi:&[Complex;MOST_MIXED]|->f64{
        let n=model.dimension(cluster);let local=model.generator(cluster,&model.point(r,kappa),t);local.scalar+expectation(&local.hamiltonian[..n*n],psi,n).re
    };
    for(cluster,r,kappa,t,psi)in samples(&model,2){
        let p=model.point(r,kappa);let force=model.generator(cluster,&p,t).force(&psi);let h=1e-5;
        for a in 0..3{
            let(mut plus,mut minus)=(r,r);plus[a]+=h;minus[a]-=h;let numeric=-(energy(cluster,plus,kappa,t,&psi)-energy(cluster,minus,kappa,t,&psi))/(2.*h);
            assert!((force[a]-numeric).abs()<=1e-5*numeric.abs().max(1.),"cluster {cluster} at {r:?}, doppler {doppler}, axis {a}: {} against {numeric}",force[a]);
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

/// The held engine (engine.rs `Held`: the twin's generator and jumps computed once, control.rs) is the plain engine's atom held at the
/// same point, but for the jump's transient, which the twin leaves out: along the atom's states over 2000 steps the same internal state after a step without a jump and the same
/// photon of every class, to the tables' interpolation between the plain engine's decay and its jumps' Σ L†L (the twin's by construction) --
/// without the photons' recoil term, where the two are one at the nodes.
#[test]
fn the_held_engine_is_the_plain_engine_at_rest(){
    let r=request(&|_|true);let mut model=Model::new(&r).unwrap();model.doppler=false;let initial=Initial::new(&model,&r).unwrap();let trap=initial.traps[0];
    let twin=Twin::new(&model,0.9).expect("a static generator");
    let mut settings=Settings::new(0.2,&trap,f64::EPSILON,false);settings.free=[false;3];settings.loss=false;
    let(plain,held)=(Engine::new(&model,settings.clone()),Engine::held(&model,&settings,&twin.held));
    for index in 0..8{
        let mut a=initial_trajectory(&model,&settings,&initial,13,index,Stratum::single(index));a.r=twin.held.point.r;a.kappa=[0.;3];
        for step in 0..2000{
            plain.step(&mut a);if step%100!=0{continue;}
            let(k,n)=(a.cluster,model.dimension(a.cluster));let norm:f64=a.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();let mut from=a.clone();for z in from.psi[..n].iter_mut(){*z=*z*(1./norm);}from.threshold=0.;
            let(mut x,mut y)=(from.clone(),from.clone());plain.step(&mut x);held.step(&mut y);
            let(nx,ny):(f64,f64)=(x.psi[..n].iter().map(|z|z.norm_sqr()).sum(),y.psi[..n].iter().map(|z|z.norm_sqr()).sum());
            for i in 0..n{assert!((x.psi[i].norm_sqr()/nx-y.psi[i].norm_sqr()/ny).abs()<=model.table_error,"trajectory {index} state {i}: {} against the tables' error {}",(x.psi[i].norm_sqr()/nx-y.psi[i].norm_sqr()/ny).abs(),model.table_error);}
            let photons=model.photons(k,&twin.held.point,&from.psi,from.time,None);let rate:f64=photons.amplitude.iter().zip(&model.ground.clusters[k].classes).map(|(u,c)|u.iter().map(|v|v[..model.ground_dimension(c.target)].iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>()).sum();
            for(c,(u,class))in photons.amplitude.iter().zip(&model.ground.clusters[k].classes).enumerate(){let nt=model.ground_dimension(class.target);
                for axis in 0..3{for i in 0..nt{let z=(0..n).fold(Complex::ZERO,|s,b|s+twin.held.jumps[k][c][axis][i*n+b]*from.psi[b]);
                    assert!((z-u[axis][i]).abs()<=model.table_error*rate.sqrt(),"trajectory {index} class {c} axis {axis} row {i}: {z:?} against {:?}",u[axis][i]);}}}
        }
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
            alive:true,jumps:0,beats:0.,seed:17,index};
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
/// the trap's top, some far out on its axis: the dark atoms lost start within 5 % of the depth below the top, but for at most
/// three of the 1024 on a stream (0-3 over six seeds, as far down as 34 % of the depth with the region, most of those at every step:
/// the region's, the harmonic part weighted down beyond it, engine.rs # Beyond the trap's region, rather than the step's) --
/// with the region 1 at 1.25, 0.31 and 0.125 us, the same atom's; without it 7, 1 and none; 20 at the 2 us the line took with the thermal ensemble before it.
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
        assert!(dark.iter().filter(|&&e|e<=-0.05).count()<=3,"region {region}: dark atoms lost from below 5 % of the depth: {dark:?}");
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

/// A near tweezer's far cross terms stay out of a cluster's dressing (ground.rs, # One rule over the ground and the excited states): a D1 gray molasses in an 808 nm tweezer keeps the Raman resonance of its cooling and repump
/// beams in one cluster and folds the pairs at the 6.8 GHz splitting, whose mixing is 1e-9: bounds of 2.9 GHz at a beat of 6 THz gave it a dressing past that splitting, a beat on a grid of 30 000 nodes per us.
#[test]
fn a_near_tweezer_s_far_cross_terms_leave_the_splitting_folded(){
    let v:Value=serde_json::from_str(include_str!("fixtures/near_tweezer_gray_molasses_motion.json")).unwrap();let r=sparse_request(&v);
    let model=Model::new(&r).unwrap();
    assert_eq!(model.clusters(),1);assert_eq!(model.dimension(0),8);assert!(!model.beating(0),"{:?}",model.beats(0).0);
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

/// A weak folded drive across an explicit state's line within the trap (the validation's F2, case 109, its first rungs): the state's shift
/// by it is bounded by its width (ground.rs `Elimination::shift_denominator`), and the tables hold it like any other: the bare denominator's pole, of no width, read
/// 22 % at any grid from 33 nodes on, the refinement stalled there.
#[test]
fn a_folded_drive_across_an_explicit_state_s_line_leaves_its_tables_smooth(){
    let v:Value=serde_json::from_str(include_str!("fixtures/folded_drive_across_explicit_line_motion.json")).unwrap();let mut r=sparse_request(&v);r.times_us=(0..21).map(|i|5.*i as f64).collect();
    let model=Model::with(&r,Ground::allowances(&r).unwrap()[1]).unwrap();
    assert!(model.ground.clusters.iter().any(|c|!c.excited.is_empty()),"the rung holds no explicit state");
    assert!(!model.table_stalled&&model.table_error<1e-3,"its error {:.2e}, stalled {}",model.table_error,model.table_stalled);
}

/// A folded drive's line narrower than the first grids where the atoms go: the retro-reflected tweezer with a resonant probe, every drive folded, on a line of half its amplitude's width: an error of about 1 from 33 to
/// 129 nodes, rising on the way, and then the cubic's order: the refinement goes on through it (model.rs `tabulate`, `STALLED_DOUBLINGS`).
#[test]
fn a_line_the_first_grids_do_not_resolve_does_not_stop_the_refinement(){
    let f:Value=serde_json::from_str(include_str!("fixtures/retro_probe_motion.json")).unwrap();let mut r=scene::request(&f);r.times_us=(0..21).map(|i|5.*i as f64).collect();
    for m in r.emission.iter_mut(){for x in m.real.iter_mut().chain(m.imag.iter_mut()){*x*=0.5;}}
    let model=Model::with(&r,f64::INFINITY).unwrap();
    assert!(!model.table_stalled&&model.table_error<0.05,"refined to {:?} nodes per axis, its error {:.2e}, stalled {}",model.table_limit,model.table_error,model.table_stalled);
}
/// The same tweezer at a mirror phase of 30°: the standing wave's antinode a fraction of a wavelength off the focus, nothing else moved. Its rung that keeps the probe holds its tables' error within the targets, without a flag.
#[test]
fn a_mirror_phase_leaves_the_tables_converging(){
    let f:Value=serde_json::from_str(include_str!("fixtures/retro_probe_motion.json")).unwrap();let mut r=scene::request(&f);r.times_us=(0..21).map(|i|5.*i as f64).collect();
    let phase=30_f64.to_radians();let turn=|m:&mut Matrix,sign:f64|{let(s,c)=(sign*phase).sin_cos();for(x,y)in m.real.iter_mut().zip(m.imag.iter_mut()){(*x,*y)=(*x*c-*y*s,*x*s+*y*c);}};
    let reflection=r.beams.iter_mut().find(|b|b.id.contains("reflection")).unwrap();turn(&mut reflection.coupling,1.);turn(&mut reflection.counter_coupling,-1.);
    let model=Model::with(&r,Ground::allowances(&r).unwrap()[1]).unwrap();
    assert!(!model.table_stalled&&model.table_error<0.05,"refined to {:?} nodes per axis, its error {:.2e}, stalled {}",model.table_limit,model.table_error,model.table_stalled);
}

/// The PGC with a resonant probe at other intensities (mW/cm²), its first microseconds: its couplings scaled by the root of the intensity, the outside shifts by the intensity.
fn probed_at(probe:f64,cooling:f64,duration:usize)->Request{
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..=duration).map(|i|i as f64).collect();
    for b in r.beams.iter_mut(){
        let to=if b.id=="probe"{probe}else if b.id.starts_with("cool"){cooling}else{continue};let f=(to/b.intensity_mw_cm2).sqrt();b.intensity_mw_cm2=to;
        for m in [&mut b.coupling,&mut b.counter_coupling]{for x in m.real.iter_mut().chain(m.imag.iter_mut()){*x*=f;}}
        for x in b.outside_shift.real.iter_mut().chain(b.outside_shift.imag.iter_mut()){*x*=f*f;}
    }
    r
}
/// A pilot whose atoms are all lost before the first output, its model's own estimates far past the targets (the resonant probe of 100 mW/cm² with cooling of 30, every drive folded: a charge of 5e8, a
/// resonance of 53 %), hands over to the next rung: the solve took that model, every atom lost by 1 us, where its 3+12 rung loses none by 2 us (its more explicit rungs agree, 63 % by 100 us). Its run
/// reads the rung's trap and region: 100 kHz and 136 rad/us, against the all-folded 78 kHz and a region of −53.
#[test]
fn a_pilot_that_keeps_no_atom_hands_over_to_the_next_rung(){
    let s=solve(&probed_at(100.,30.,3),Device::Cpu,&mut |_,_|{}).unwrap();
    assert!(s.method.contains("显式"),"{}",s.method);assert!(s.loss_probability[2]<0.1,"the loss at 2 us: {:?}",s.loss_probability);
    assert!(s.trap_frequency_khz[0]>99.,"the trap: {:?} kHz",s.trap_frequency_khz);
}
/// A near beam whose fold anti-traps the reference state (the resonant probe of 300 mW/cm² with cooling of 30, every drive folded: 146 % of resonance) is no reason to refuse while a rung keeps that drive.
#[test]
fn a_model_with_no_trap_hands_over_to_the_next_rung(){
    let s=solve(&probed_at(300.,30.,3),Device::Cpu,&mut |_,_|{}).unwrap();
    assert!(s.method.contains("显式"),"{}",s.method);assert!(s.trap_frequency_khz[0]>99.,"the trap: {:?} kHz",s.trap_frequency_khz);
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
/// whole step from thermal states a few microseconds into the PGC with a resonant probe, its 10-state cluster's beat at 36.4 MHz.
#[test]
fn a_beating_grid_s_measured_phase_meets_its_shares(){
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..101).map(|i|i as f64).collect();
    let model=Model::with(&r,Ground::allowances(&r).unwrap()[1]).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let(dt,waiting)=(0.25,0.6);let share=NUMERICAL_SHARE*POPULATION_TARGET*dt/waiting;
    let mut at=Settings::new(dt,&initial.traps[0],share,true);at.horizon=waiting;let engine=Engine::new(&model,at.clone());
    let states:Vec<Trajectory>=(0..64).map(|i|{let mut t=initial_trajectory(&model,&at,&initial,7,i,Stratum::single(i));for _ in 0..8{engine.step(&mut t);}t}).collect();
    let phase=grid_phase(&model,&states,&at,None);assert!(phase>4.*at.node_phase,"{phase} against the bound's {}",at.node_phase);
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

/// What a beating grid lands between the groups of ground states over the run (solve.rs `grid_phase`, `landed_excess`), on the same PGC: at the rung a solve takes there it leaves the phase its shares measure; at the rung with
/// three states of F' = 2 explicit, which those shares let the grid kick through the beat, it takes the phase down by a third. A step within a spacing has nothing to read beyond the grids' coinciding: the phase stops there (it ran to 1e28).
#[test]
fn a_beating_grid_s_phase_holds_what_it_lands_over_the_run(){
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..101).map(|i|i as f64).collect();
    let ladder=Ground::allowances(&r).unwrap();let groups=Model::with(&r,f64::INFINITY).unwrap().ground.cluster.clone();
    let phases=|rung:usize,dt:f64|{
        let model=Model::with(&r,ladder[rung]).unwrap();let initial=Initial::new(&model,&r).unwrap();let waiting=0.7;
        let mut at=Settings::new(dt,&initial.traps[0],NUMERICAL_SHARE*POPULATION_TARGET*dt/waiting,true);at.horizon=waiting;let engine=Engine::new(&model,at.clone());
        let states:Vec<Trajectory>=(0..64).map(|i|{let mut t=initial_trajectory(&model,&at,&initial,7,i,Stratum::single(i));for _ in 0..((2.5/dt) as usize).min(8){engine.step(&mut t);}t}).collect();
        (grid_phase(&model,&states,&at,None),grid_phase(&model,&states,&at,Some((&groups,100.))))};
    let(shares,landed)=phases(1,1./3.);assert_eq!(landed,shares,"the chosen rung's phase");
    let(shares,landed)=phases(8,1./3.);assert!(landed<0.8*shares,"{landed} against the shares' {shares}");
    let(shares,_)=phases(1,0.002);assert!(shares<4.,"a step within a spacing: {shares}");
}

/// The ladder's rungs keep more as the allowance falls, and the closure keeps what it must first: on the PGC with a resonant probe
/// the first rung holds the probe's five drives and, into those five states, the cooling beams' (their pairs with the probe's lie within the dressing's reach of their 36 MHz, `Ground::keep_partners`),
/// the cooling beams' others folded. (With the closure's estimates taken under the folded probe's light, and no rung above each ground state's first kept drive, the first rung
/// held 7 excited states, its T 5 % above the fully explicit F = 2 cluster's at 100 us, where this rung's is within its 2 % of statistics.)
#[test]
fn the_ladder_keeps_more_as_the_allowance_falls_and_the_near_drives_first(){
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..101).map(|i|i as f64).collect();
    let ladder=Ground::allowances(&r).unwrap();assert!(ladder[0]==f64::INFINITY&&ladder.windows(2).all(|w|w[0]>w[1]),"{ladder:?}");
    let grounds:Vec<Ground>=ladder.iter().take(12).map(|&a|Ground::with(&r,a).unwrap()).collect();
    let explicit:Vec<usize>=grounds.iter().map(|g|g.clusters.iter().map(|c|c.excited.len()).sum()).collect();
    assert!(explicit[0]==0&&explicit.windows(2).all(|w|w[0]<=w[1]),"explicit states along the ladder: {explicit:?}");
    let g=&grounds[1];let drives:Vec<usize>=g.excited_items.iter().filter(|x|!x.folded&&x.size>1e-6).filter_map(|x|match x.kind{ExcitedKind::Drive(d)=>Some(d.carrier),_=>None}).collect();
    let probe=drives.iter().filter(|&&c|c==drives[0]).count().min(drives.iter().filter(|&&c|c!=drives[0]).count());
    assert!(explicit[1]==5&&probe==5,"the first rung: {} explicit states, its drives' carriers {drives:?}",explicit[1]);
}

/// What a rung holds between its ground states, its explicit states eliminated at second order, against the second order of every excited state at a point:
/// the largest difference of H_eff's Hermitian part (less each one's mean diagonal), and of the rates between the bare ground states, each frequency's photons as the rung's classes
/// join them. In the frameless picture a cluster's entry at residual Ω beats at ν = Ω − (θ_x − θ_y), a drive of carrier c at −δ_c.
fn second_order_gaps(model:&Model,k:usize,point:[f64;3])->(f64,f64){
    use atom_motion_3d::linear::{Dense,Lu};
    let g=&model.ground;let cl=&g.clusters[k];let(n0,n)=(cl.states.len(),cl.dimension());let(ng,ne)=(g.ground_states.len(),g.excited_states.len());
    let a=g.amplitudes_at(point);let e=g.eliminate(&model.intensities_at_point(point)).unwrap();
    let c=model.components(k,&model.point(point,[0.;3]));
    let entry=|x:usize,y:usize,nu:f64|->Complex{let apart=cl.frame(x)-cl.frame(y);let mut s=Complex::ZERO;
        if (apart+nu).abs()<1e-6{s+=c.statics.hamiltonian[x*n+y];}
        for b in &c.beats{if (b.omega-apart-nu).abs()<1e-6{s+=b.hamiltonian[x*n+y];}}s};
    let diagonal=|x:usize|c.statics.scalar+cl.frame(x);
    let shifted=|m:&Dense,z:f64|Lu::new(&Dense::from_fn(m.rows,m.cols,|i,j|if i==j{m[(i,j)]-Complex::new(z,0.)}else{m[(i,j)]})).unwrap();
    let block=Dense::from_fn(n-n0,n-n0,|i,j|entry(n0+i,n0+j,0.)+Complex::new(if i==j{diagonal(n0+i)}else{0.},0.));
    let(mut exact,mut held)=(Dense::zeros(n0,n0),Dense::zeros(n0,n0));let mut worst=0_f64;
    for i in 0..n0{exact[(i,i)]=Complex::new(e.energies[cl.states[i]],0.);for j in 0..n0{held[(i,j)]=entry(i,j,0.);}held[(i,i)]+=Complex::new(diagonal(i),0.);}
    let own:Vec<f64>=(0..n0).map(|i|held[(i,i)].re).collect();
    // An explicit state's photon, dressed by H's folded couplings (ground.rs `Ground::dressing`).
    let flags:Vec<bool>=(0..ne).map(|f|cl.excited.contains(&f)).collect();
    let dressed:Vec<Vec<Complex>>=cl.excited.iter().map(|&f|g.dressing(&e,f,&flags).unwrap()).collect();
    let photon=|f:usize,ax:usize,to:usize|(0..ne).fold(g.emission[ax][(to,cl.excited[f])],|s,q|s+g.emission[ax][(to,q)]*dressed[f][q]);
    for(j,&b)in cl.states.iter().enumerate(){
        let(mut rates,mut photons)=(vec![0.;ng],vec![0.;ng]);
        for carrier in &g.carriers{
            let d=carrier.detuning;
            let field=|m:&[Dense],i:usize,b:usize|carrier.beams.iter().fold(Complex::ZERO,|s,&l|s+a[l]*m[l][(i,b)]);
            let mut x:Vec<Complex>=(0..ne).map(|i|field(&g.couplings,i,b)).collect();shifted(&e.excited_hamiltonian,e.energies[b]+d).solve(&mut x);
            for(i,&t)in cl.states.iter().enumerate(){exact[(i,j)]=exact[(i,j)]-(0..ne).fold(Complex::ZERO,|s,q|s+field(&g.couplings,q,t).conj()*x[q]);}
            for to in 0..ng{rates[to]+=(0..3).map(|ax|(0..ne).fold(Complex::ZERO,|s,q|s+g.emission[ax][(to,q)]*x[q]).norm_sqr()).sum::<f64>();}
            let mut y:Vec<Complex>=(n0..n).map(|f|entry(f,j,-d)).collect();
            if n>n0{shifted(&block,own[j]+d).solve(&mut y);}
            for i in 0..n0{held[(i,j)]=held[(i,j)]-(n0..n).fold(Complex::ZERO,|s,f|s+entry(i,f,d)*y[f-n0]);}
            for class in &cl.classes{for to in 0..ng{for ax in 0..3{
                let mut u=Complex::ZERO;
                if class.photons.iter().any(|p|std::ptr::eq(&g.carriers[p.0.carrier],carrier)&&p.0.from==b&&p.0.to==to){for q in 0..ne{u+=g.emission[ax][(to,q)]*field(&e.responses,q,b);}}
                for &(f,t,_)in &class.direct{if t==to{u+=photon(f-n0,ax,to)*y[f-n0];}}
                photons[to]+=u.norm_sqr();
            }}}
        }
        for to in 0..ng{worst=worst.max((rates[to]-photons[to]).abs());}
    }
    let hermitian=|m:&Dense,i:usize,j:usize|(m[(i,j)]+m[(j,i)].conj())*0.5;let mean=|m:&Dense|(0..n0).map(|i|m[(i,i)].re).sum::<f64>()/n0 as f64;
    let(m0,m1)=(mean(&exact),mean(&held));
    let mut gap=0_f64;for i in 0..n0{for j in 0..n0{
        let shift=|m:f64|Complex::new(if i==j{m}else{0.},0.);gap=gap.max(((hermitian(&exact,i,j)-shift(m0))-(hermitian(&held,i,j)-shift(m1))).abs());
    }}
    (gap,worst)
}

/// Every rung of the PGC with a resonant probe holds the Raman couplings and the rates of the second order of every excited state, whatever its frames. The folded partner of a
/// kept drive into one state, its pair near, and an explicit state's photon of a frequency its frame does not carry, were lost where a probe's and a cooling drive's one state split the
/// ground states' frames: couplings 1.09 rad/us off of the largest 4-5, a rate 0.040 /us off of 0.30 (T up to 10 % high at a fold's charge of 0.07 %, ground.rs); here within 0.052 and 0.011.
#[test]
fn a_rung_holds_the_raman_couplings_and_the_photons_of_the_second_order(){
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..101).map(|i|i as f64).collect();
    for(i,&allowance)in Ground::allowances(&r).unwrap().iter().enumerate().skip(1).take(8){
        let model=Model::with(&r,allowance).unwrap();let g=&model.ground;
        let k=(0..model.clusters()).max_by_key(|&k|model.dimension(k)).unwrap();let cl=&g.clusters[k];
        // No explicit state takes a kept and a folded drive of one carrier from the cluster's ground states.
        for &e in &cl.excited{for c in 0..g.carriers.len(){
            let verdicts:Vec<bool>=g.excited_items.iter().filter_map(|x|match x.kind{ExcitedKind::Drive(d)if d.carrier==c&&d.excited==e&&cl.states.contains(&d.ground)&&x.size/x.distance>f64::EPSILON=>Some(x.folded),_=>None}).collect();
            assert!(verdicts.iter().all(|&v|v)||verdicts.iter().all(|&v|!v),"rung {i}: carrier {c} into {}: {verdicts:?}",r.states[g.excited_states[e]]);
        }}
        for point in [[0.1,0.05,0.2],[0.3,-0.1,0.5],[-0.2,0.2,-0.4]]{
            let(coupling,rate)=second_order_gaps(&model,k,point);
            assert!(coupling<0.1&&rate<0.015,"rung {i} at {point:?}: H_eff {coupling:.3} rad/us off, a rate {rate:.4} /us off");
        }
    }
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

/// A table that holds only round-off is no term (model.rs `Model::prune`, ground.rs `rounded`, `partition`): the PGC with a resonant probe with every drive folded held Raman groups of 1e-53, products of two round-offs a
/// selection rule leaves, whose relative "error" of 1.8 read as a pole no grid resolves and stopped its refinement at 33 nodes; the rung of 3+10 at 100 mW/cm² 24 of its 36 groups, 147 %. What holds anything interpolates within 1e-3 there.
#[test]
fn a_table_of_round_off_is_no_error(){
    let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=(0..101).map(|i|i as f64).collect();
    let model=Model::with(&r,f64::INFINITY).unwrap();
    println!("limit {:?}, its error {:.3e}, stalled {}",model.table_limit,model.table_error,model.table_stalled);
    assert!(!model.table_stalled&&model.table_error<1e-3&&model.table_error>0.,"its error {:.2e}, stalled {}",model.table_error,model.table_stalled);
    let v=probed_at(100.,3.,100);let rung=Model::with(&v,Ground::allowances(&v).unwrap()[1]).unwrap();
    assert!(!rung.table_stalled&&rung.table_error<1e-3,"the rung's error {:.2e}, stalled {}",rung.table_error,rung.table_stalled);
    assert!(rung.work(1).pairs<=64,"the rung's pairs of fields: {} (141 with those of round-off)",rung.work(1).pairs);
}

/// A σ⁺σ⁻ molasses on an F = 1 → F′ = 2 line along z, a detuning σ below its centre: the ground states' energies m Ω, the excited states' m′ Ω′, the σ⁺ beam along +z at +δ/2, the σ⁻
/// beam along −z at −δ/2 (ground-2-* and D2-3-* stand for the two levels).
fn larmor(omega:f64,excited:f64,delta:f64)->Request{
    let(g,ground,upper)=(3.,|m:i32|format!("ground-2-{m}"),|m:i32|format!("D2-3-{m}"));
    let strength=|m:i32,q:i32|->f64{match q{0=>if m==0{2./3.}else{0.5},_=>match m*q{1=>1.,0=>0.5,_=>1./6.}}};
    let names:Vec<(String,String,i32,i32)>=(-1..=1).flat_map(|m|[-1,0,1].map(|q|(m,q))).map(|(m,q)|(upper(m+q),ground(m),m,q)).collect();
    let levels:Vec<(String,f64)>=(-1..=1).map(|m|(ground(m),m as f64*omega)).chain((-2..=2).map(|m|(upper(m),SIGMA+m as f64*excited))).collect();
    let levels:Vec<(&str,f64)>=levels.iter().map(|(s,e)|(s.as_str(),*e)).collect();
    let z=Complex::new(1.,0.);
    let emissions:Vec<(&str,&str,[Complex;3],f64)>=names.iter().map(|(e,b,m,q)|(e.as_str(),b.as_str(),match q{0=>[Complex::ZERO,Complex::ZERO,z],q=>circular(*q as f64)},LINE*strength(*m,*q))).collect();
    let beam=|q:i32|names.iter().filter(|x|x.3==q).map(|(e,b,m,_)|(e.as_str(),b.as_str(),g*strength(*m,q).sqrt())).collect::<Vec<_>>();
    let mut r=synthetic(&levels,&emissions,&[([0.,0.,1.],beam(1)),([0.,0.,-1.],beam(-1))],"ground-2-0");
    r.frequencies_mhz[1]=0.5*delta.abs()/std::f64::consts::TAU;
    if delta!=0.{let sign=delta.signum() as i32;r.beams[0].frequency_index=vec![0,sign];r.beams[1].frequency_index=vec![0,-sign];}
    r
}
/// The steady populations of the molasses' three ground states, frozen at the focus: the null vector of its folded generator and jumps on the clusters that hold them.
fn larmor_steady(r:&Request)->[f64;3]{
    use atom_motion_3d::linear::{Dense,Lu};
    let model=Model::new(r).unwrap();let g=&model.ground;let at=|m:i32|{let s=r.states.iter().position(|x|*x==format!("ground-2-{m}")).unwrap();g.ground_states.iter().position(|&i|i==s).unwrap()};
    let held=[at(-1),at(0),at(1)];let focus=[0.;3];let e=g.eliminate(&model.intensities_at_point(focus)).unwrap();
    let clusters:Vec<usize>=(0..g.clusters.len()).filter(|&k|g.clusters[k].states.iter().any(|b|held.contains(b))).collect();
    let sizes:Vec<usize>=clusters.iter().map(|&k|g.clusters[k].states.len()).collect();let offsets:Vec<usize>=sizes.iter().scan(0,|o,&n|{let x=*o;*o+=n*n;Some(x)}).collect();let d:usize=sizes.iter().map(|n|n*n).sum();
    let mut generator=Dense::zeros(d,d);
    for(i,&k)in clusters.iter().enumerate(){let n=sizes[i];let local=g.local_at(&e,k,focus,0.).unwrap();for column in 0..n*n{
        let rho=Dense::from_fn(n,n,|a,b|if a*n+b==column{Complex::ONE}else{Complex::ZERO});
        let mut image:Vec<Dense>=sizes.iter().map(|&m|Dense::zeros(m,m)).collect();
        image[i].add_scaled(Complex::new(0.,-1.),&local.effective.product(&rho));image[i].add_scaled(Complex::I,&rho.product(&local.effective.adjoint()));
        for(class,jumps)in g.clusters[k].classes.iter().zip(&local.jumps){if let Some(b)=clusters.iter().position(|&x|x==class.target){for j in jumps{image[b].add_scaled(Complex::ONE,&j.product(&rho).product(&j.adjoint()));}}}
        for(j,x)in image.iter().enumerate(){for(q,z)in x.entries.iter().enumerate(){generator[(offsets[j]+q,offsets[i]+column)]=*z;}}
    }}
    // The trace's row in the place of the first.
    for j in 0..d{generator[(0,j)]=Complex::ZERO;}for(i,&n)in sizes.iter().enumerate(){for a in 0..n{generator[(0,offsets[i]+a*n+a)]=Complex::ONE;}}
    let mut x=vec![Complex::ZERO;d];x[0]=Complex::ONE;Lu::new(&generator).unwrap().solve(&mut x);
    held.map(|b|{let k=g.cluster[b];let i=clusters.iter().position(|&c|c==k).unwrap();let a=g.clusters[k].position(b).unwrap();x[offsets[i]+a*sizes[i]+a].re})
}
/// Larmor's theorem: with the excited states' g factor the ground states', a field along the beams and the beams' difference 2Ω is the field-free molasses in the frame turning at Ω,
/// exactly, by another route: two carriers, their photons in classes of their frequencies, its Raman couplings static in the frames alone. So its frozen atom's populations do not depend on
/// the field, to double precision, and stand within the fold's own order of the one carrier's (the saturation 2(g/|σ − iΓ/2|)² = 0.9 % here; 8e-4: the two carriers' light dresses each
/// other's response, one carrier's does not), while the field without the difference leaves them; and the excited states' g factor apart, the populations lean to the low m in proportion.
#[test]
fn a_bias_field_compensated_by_the_beams_is_larmor_s_frame(){
    let free=larmor_steady(&larmor(0.,0.,0.));
    println!("field-free {free:?}");assert!((free[0]-free[2]).abs()<=1e-12,"{free:?}");
    let saturation=2.*9./(SIGMA*SIGMA+LINE*LINE/4.);let first=larmor_steady(&larmor(0.3,0.3,0.6));
    for omega in [0.3,1.,3.,8.,30.]{
        let p=larmor_steady(&larmor(omega,omega,2.*omega));
        let worst=(0..3).map(|i|(p[i]-free[i]).abs()).fold(0.,f64::max);println!("Ω {omega}: {p:?}, |Δp| {worst:.2e}");
        assert!(worst<=0.2*saturation,"Ω {omega}: {p:?} against {free:?}");
        assert!((0..3).all(|i|(p[i]-first[i]).abs()<=1e-12),"Ω {omega}: {p:?} against {first:?}");
        // Without the difference the Raman coupling is off by 2Ω.
        if omega>=1.{let off=larmor_steady(&larmor(omega,omega,0.));let moved=(0..3).map(|i|(off[i]-free[i]).abs()).fold(0.,f64::max);println!("  without the difference {off:?}");assert!(moved>=5.*worst.max(1e-4),"Ω {omega}: {off:?}");}
    }
    let lean=|xi:f64|{let p=larmor_steady(&larmor(1.,1.+xi,2.));p[0]-p[2]};
    let(one,two)=(lean(0.1),lean(0.2));println!("the excited g factor apart: p(-1) - p(+1) {one:.5} at 0.1 rad/us, {two:.5} at 0.2");
    assert!(one>0.&&(two/one-2.).abs()<=0.1,"{one} {two}");
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

/// T's step error over the run (solve.rs `drift`): the retro-reflected tweezer's lattice axis heats all run long and a rate's error adds up,
/// so at a step whose error over one period lies within T's tolerance more than the tolerance stands over the 100 us, and at half that step less than half of that.
#[test]
fn the_t_error_over_the_run_adds_up_on_a_heating_axis_and_goes_with_the_step(){
    let fixture:Value=serde_json::from_str(include_str!("fixtures/retro_probe_motion.json")).unwrap();
    let(outputs,interval)=(21,5.);let mut r=scene::request(&fixture);r.times_us=(0..outputs).map(|i|interval*i as f64).collect();
    // The rung its solve takes, at the waiting time and Doppler budget it takes them with.
    let model=Model::with(&r,Ground::allowances(&r).unwrap()[5]).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();let waiting=0.47;
    let settings=|steps:usize|{let dt=interval/steps as f64;let mut s=Settings::new(dt,&initial.traps[reference],NUMERICAL_SHARE*POPULATION_TARGET*dt/waiting,true);s.doppler=0.02;s.region=initial.depths[reference];s.horizon=waiting;s};
    let steps=48;let harmonic=Harmonic::new(&model,&settings(steps),&initial,steps,outputs);
    let(states,_,_)=sample_states(&model,&settings(steps),&initial,0x3D_5EED,32,steps,outputs,&harmonic);let probes=bound_probes(&model,&states,&vec![1.;states.len()]);
    let period=std::f64::consts::TAU/initial.traps[reference].frequencies.iter().copied().fold(f64::INFINITY,f64::min);
    let tolerance=BIAS_SHARE*TEMPERATURE_TARGET;let limit=|_:f64|tolerance;
    let(one,whole)=(drift(&model,&settings,steps,&probes,period,period,&limit,0.,f64::INFINITY),drift(&model,&settings,steps,&probes,period,100.,&limit,0.,f64::INFINITY));let half=drift(&model,&settings,2*steps,&probes,period,100.,&limit,whole.span,f64::INFINITY);
    println!("over a period {one:?}, over the run {whole:?}, at half the step {half:?} ({} us period)",period);
    assert!(one.excess<1.&&whole.excess>1.&&half.stands<0.5*whole.stands,"{one:?} {whole:?} {half:?}");
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

/// A synthetic atom on the model's 32 states, its levels' energies in the frame (rad/us; the excited states not named far off), its
/// emissions (excited state, ground state, dipole, rate), its near beams at the frame's frequency (direction, couplings g:
/// excited state, ground state, rad/us), a tweezer far off and weak, and a mass a million times the atom's: at any velocity an atom
/// held to it, its recoil's shift nothing.
fn synthetic(levels:&[(&str,f64)],emissions:&[(&str,&str,[Complex;3],f64)],beams:&[([f64;3],Vec<(&str,&str,f64)>)],initial:&str)->Request{
    let mut r=request(&|b|b["waistUm"].is_number());let at=|s:&str|r.states.iter().position(|x|x==s).unwrap();
    let zero=||Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]};
    let mut h=zero();for s in 0..STATES{if !r.states[s].starts_with("ground"){h.real[s*STATES+s]=5000.+s as f64;}}
    for &(s,e)in levels{let i=at(s);h.real[i*STATES+i]=e;}
    let mut emission=[zero(),zero(),zero()];
    for &(e,g,d,rate)in emissions{let(i,j)=(at(g),at(e));for a in 0..3{let c=d[a]*rate.sqrt();emission[a].real[i*STATES+j]=c.re;emission[a].imag[i*STATES+j]=c.im;}}
    let frames:Vec<f64>=r.states.iter().map(|s|if s.starts_with("ground"){0.}else{3.8e8}).collect();
    let mut list:Vec<Beam>=beams.iter().enumerate().map(|(n,(d,couplings))|{let mut x=zero();for &(e,g,v)in couplings{x.real[at(e)*STATES+at(g)]=v;}
        Beam{id:format!("b{n}"),name:format!("b{n}"),direction:*d,wave_number_per_um:WAVE,waist_um:None,intensity_mw_cm2:1.,frequency_index:vec![0,0],coupling:x,counter_coupling:zero(),outside_shift:zero()}}).collect();
    let mut x=zero();x.real[at("D1-1-0")*STATES+at(initial)]=1e-3;
    list.push(Beam{id:"tweezer".into(),name:"tweezer".into(),direction:[1.,0.,0.],wave_number_per_um:7.4,waist_um:Some(1.),intensity_mw_cm2:1.,frequency_index:vec![1,0],coupling:x,counter_coupling:zero(),outside_shift:zero()});
    let i=at(initial);let mut density=vec![Complex::ZERO;STATES*STATES];density[i*STATES+i]=Complex::ONE;
    r.trap_beam=list.len()-1;r.mass_kg*=1e6;(r.hamiltonian,r.emission,r.frame_frequencies_mhz,r.frequencies_mhz,r.beams,r.initial_density,r.field_vector_g,r.times_us)=(h,emission,frames,vec![1e7,0.],list,density,[0.;3],vec![0.,100.]);r
}
const WAVE:f64=8.05;const LINE:f64=38.1;const SIGMA:f64=40.;
/// A circular dipole about z.
fn circular(sign:f64)->[Complex;3]{let r=0.5f64.sqrt();[Complex::new(r,0.),Complex::new(0.,sign*r),Complex::ZERO]}
/// Γ g²/(σ² + Γ²/4): the scattering rate of a drive g at a detuning σ.
fn lorentzian(g:f64,sigma:f64)->f64{LINE*g*g/(sigma*sigma+LINE*LINE/4.)}
/// A ground state's cluster and place.
fn place(model:&Model,request:&Request,state:&str)->(usize,usize){let s=request.states.iter().position(|x|x==state).unwrap();let b=model.ground.ground_states.iter().position(|&i|i==s).unwrap();let k=model.ground.cluster[b];(k,model.ground.clusters[k].position(b).unwrap())}
/// The engine's mean force on a ground state at (r, κ), from its parts: the Ehrenfest force, the jumps' mean Σ_a Im⟨u_a|∇u_a⟩/Σ_a ‖u_a‖² (engine.rs
/// `Engine::kick`, over the photons' directions and polarizations) and the hazard the generator gives them.
fn parts(model:&Model,cluster:usize,state:usize,r:[f64;3],kappa:[f64;3])->([f64;3],[f64;3],f64){
    let mut psi=[Complex::ZERO;MOST_MIXED];psi[state]=Complex::ONE;
    let p=model.point(r,kappa);let g=model.generator(cluster,&p,0.);let n=g.n;
    let hazard=-2.*expectation(&g.hamiltonian[..n*n],&psi,n).im;let photons=model.photons(cluster,&p,&psi,0.,None);let(mut weight,mut mean)=(0.,[0.;3]);
    for(c,(class,u))in model.ground.clusters[cluster].classes.iter().zip(&photons.amplitude).enumerate(){let nt=model.ground_dimension(class.target);
        weight+=u.iter().map(|v|v[..nt].iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>();
        for a in 0..3{let mut e=[0.;3];e[a]=1.;let(gradient,_)=photons.absorption(c,e,&[Complex::ZERO;MOST_STATES]);for x in 0..3{for i in 0..nt{mean[x]+=(u[a][i].conj()*gradient[i][x]).im;}}}}
    (g.force(&psi),mean.map(|x|x/weight),hazard)
}
/// Atoms from a ground state, held at velocity v about r0 over `time` after `skip`: per axis the mean of κ's rate and its standard error, the jumps' rate
/// and the final populations.
fn held_force(model:&Model,request:&Request,state:&str,r0:[f64;3],v:[f64;3],count:u64,time:f64,dt:f64,skip:f64)->([f64;3],[f64;3],f64,Vec<f64>){
    use rayon::prelude::*;
    let(hm,(cluster,at))=(model.hbar_over_mass,place(model,request,state));let k0=v.map(|x|x/hm);let start:[f64;3]=std::array::from_fn(|a|r0[a]-v[a]*time/2.);
    let trap=atom_motion_3d::model::Trap{lattice:None,axes:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],frequencies:[0.;3],lab:[0.;3]};
    let mut s=Settings::new(dt,&trap,f64::EPSILON,false);s.loss=false;let engine=Engine::new(model,s);let(steps,first)=((time/dt).round() as usize,(skip/dt).round() as usize);
    let out:Vec<([f64;3],u64,[f64;STATES])>=(0..count).into_par_iter().map(|i|{
        let mut psi=[Complex::ZERO;MOST_MIXED];psi[at]=Complex::ONE;
        let mut t=atom_motion_3d::engine::Trajectory{cluster,psi,r:start,kappa:k0,time:0.,threshold:atom_motion_3d::engine::jump_stream(7,i,0).open_uniform(),alive:true,jumps:0,beats:0.,seed:7,index:i};
        let(mut base,mut jumps)=(k0,0);for n in 0..steps{if n==first{base=t.kappa;jumps=t.jumps;}engine.step(&mut t);}
        (std::array::from_fn(|a|t.kappa[a]-base[a]),t.jumps-jumps,model.populations(t.cluster,&model.point(t.r,t.kappa),&t.psi,t.time))}).collect();
    let(n,span)=(count as f64,time-skip);let mean:[f64;3]=std::array::from_fn(|a|out.iter().map(|x|x.0[a]).sum::<f64>()/n);
    let error=std::array::from_fn(|a|(out.iter().map(|x|(x.0[a]-mean[a]).powi(2)).sum::<f64>()/(n-1.)/n).sqrt()/span);
    (mean.map(|x|x/span),error,out.iter().map(|x|x.1 as f64).sum::<f64>()/n/span,(0..STATES).map(|s|out.iter().map(|x|x.2[s]).sum::<f64>()/n).collect())
}
/// A request's model with every drive folded and with every drive kept.
fn folded_and_explicit(r:&Request)->[Model;2]{[Model::new(r).unwrap(),Model::with(r,*Ground::allowances(r).unwrap().last().unwrap()).unwrap()]}

/// A running wave's force is ħk R(σ + kv), the rate at the Doppler shifted detuning (model.rs # The photons and the Doppler term): on a closed line, g = 3 rad/us, σ = 40
/// rad/us, its hazard the first order's and its mean kick ħk whatever the velocity -- with the first order in the rate at rest and the Doppler term's own
/// gradient left out the slope of F(v) had the wrong sign, a blue probe's Doppler heating its cooling --, folded as with the state explicit, within 4 standard errors.
#[test]
fn a_running_wave_s_force_follows_its_doppler_shifted_rate(){
    let z=[0.,0.,1.];let r=synthetic(&[("D2-3-3",SIGMA)],&[("D2-3-3","ground-2-2",circular(1.),LINE)],&[(z,vec![("D2-3-3","ground-2-2",3.)])],"ground-2-2");
    let[folded,explicit]=folded_and_explicit(&r);
    let(cluster,state)=place(&folded,&r,"ground-2-2");let hm=folded.hbar_over_mass;let rate=lorentzian(3.,SIGMA);let slope=2.*SIGMA/(SIGMA*SIGMA+LINE*LINE/4.);
    for v in [-0.4,-0.2,-0.05,0.05,0.2,0.4]{
        let w=WAVE*v;let(force,mean,hazard)=parts(&folded,cluster,state,[0.,0.,0.3],[0.,0.,v/hm]);
        assert!((hazard/(rate*(1.-w*slope))-1.).abs()<=1e-3,"v {v}: hazard {hazard} against {}",rate*(1.-w*slope));
        assert!((mean[2]-WAVE).abs()<=1e-9&&mean[0].abs()+mean[1].abs()<=1e-9,"v {v}: the mean kick {mean:?}");
        assert!(force[2].abs()<=1e-6,"v {v}: the Ehrenfest force {force:?}");
    }
    for(model,count)in [(&folded,8192),(&explicit,2048)]{for v in [0.1,0.4]{
        let(a,b)=(held_force(model,&r,"ground-2-2",[0.;3],[0.,0.,v],count,100.,0.1,5.),held_force(model,&r,"ground-2-2",[0.;3],[0.,0.,-v],count,100.,0.1,5.));
        let exact=WAVE*(lorentzian(3.,SIGMA+WAVE*v)-lorentzian(3.,SIGMA-WAVE*v))/2.;let(odd,error)=((a.0[2]-b.0[2])/2.,0.5*a.1[2].hypot(b.1[2]));
        println!("v {v}: odd part of F {odd:.4} ± {error:.4} against k(R₊ − R₋)/2 {exact:.4}");
        assert!((odd-exact).abs()<=4.*error,"v {v}: odd part of F {odd} ± {error} against {exact}");
    }}
}
/// A balanced pair's friction is Doppler's (the J = 0 → J′ = 1 molasses, σ⁺ from +z and σ⁻ from −z on orthogonal dipoles, no interference): F = k(R₊ − R₋), to 4 standard
/// errors, folded as with the states explicit. Each beam's part in the photon draws its kick; with the term's own gradient left out half of it.
#[test]
fn a_balanced_pair_s_friction_is_doppler_s(){
    let(z,up,down)=([0.,0.,1.],"D2-3-3","D2-3-1");let g="ground-2-2";
    let r=synthetic(&[(up,SIGMA),(down,SIGMA)],&[(up,g,circular(1.),LINE),(down,g,circular(-1.),LINE)],&[(z,vec![(up,g,3.)]),(z.map(|x|-x),vec![(down,g,3.)])],g);
    for(model,count)in folded_and_explicit(&r).iter().zip([16384,2048]){for v in [0.1,0.4]{
        let(force,error,_,_)=held_force(model,&r,g,[0.;3],[0.,0.,v],count,100.,0.1,5.);let exact=WAVE*(lorentzian(3.,SIGMA+WAVE*v)-lorentzian(3.,SIGMA-WAVE*v));
        println!("v {v}: F {:.4} ± {:.4} against k(R₊ − R₋) {exact:.4}",force[2],error[2]);
        assert!((force[2]-exact).abs()<=4.*error[2],"v {v}: F {} ± {} against {exact}",force[2],error[2]);
    }}
}
/// The optical Bloch equations of a two-level atom moving at v through a standing wave, from far back along its path to z (RK4): the force −⟨∂_z H⟩ at z.
fn bloch_force(z:f64,v:f64,g:f64)->f64{
    let(back,steps)=(4.,8000);let h=back/steps as f64;let field=|z:f64|2.*g*(WAVE*z).cos();
    let derivative=|t:f64,ee:f64,eg:Complex|->(f64,Complex){let om=Complex::new(field(z+v*t),0.);
        ((Complex::new(-LINE*ee,0.)+Complex::new(0.,-1.)*(om*eg.conj()-om.conj()*eg)).re,Complex::new(-0.5*LINE,-SIGMA)*eg+Complex::new(0.,-1.)*om*(1.-2.*ee))};
    let(mut ee,mut eg,mut t)=(0.,Complex::ZERO,-back);
    for _ in 0..steps{
        let(a,b)=derivative(t,ee,eg);let(c,d)=derivative(t+h/2.,ee+h/2.*a,eg+b*(h/2.));let(e,f)=derivative(t+h/2.,ee+h/2.*c,eg+d*(h/2.));let(x,y)=derivative(t+h,ee+h*e,eg+f*h);
        ee+=h/6.*(a+2.*c+2.*e+x);eg+=(b+d*2.+f*2.+y)*(h/6.);t+=h;
    }
    -2.*(-2.*g*WAVE*(WAVE*z).sin())*eg.re
}
/// A standing wave's friction on a two-level atom goes as sin²(kz) -- the textbook's, 8 k² v g² σ Γ |G|⁴ sin²(kz), and the optical Bloch equations' at every phase --: nothing at the
/// antinode, twice the running waves' mean at the node. The jumps' mean gives that mean everywhere, the first order's Hermitian part's gradient the rest: an atom a trap holds
/// within the wavelength (kx ~ 0.7 in the tweezer's radial axes at 30 uK) takes the friction of its place. Where the rate at rest vanishes the first-order rate falls out of the mean, a
/// second-order share of the path (kv|G|). The first order's fast atom stays a decay (`Model::first_shift`).
#[test]
fn a_standing_wave_s_friction_goes_as_the_square_of_its_sine(){
    let(z,e,g)=([0.,0.,1.],"D2-3-3","ground-2-2");let c=0.5;
    let r=synthetic(&[(e,SIGMA)],&[(e,g,circular(1.),LINE)],&[(z,vec![(e,g,c)]),(z.map(|x|-x),vec![(e,g,c)])],g);
    let model=Model::new(&r).unwrap();let(cluster,state)=place(&model,&r,g);let hm=model.hbar_over_mass;let v=0.02;
    let denominator=SIGMA*SIGMA+LINE*LINE/4.;
    for i in 0..8{
        let phase=i as f64*std::f64::consts::PI/8.+0.05;let at=[0.,0.,phase/WAVE];
        let(a,b)=(parts(&model,cluster,state,at,[0.,0.,v/hm]),parts(&model,cluster,state,at,[0.,0.,-v/hm]));
        let odd=(a.0[2]-b.0[2]+a.1[2]*a.2-b.1[2]*b.2)/2.;
        let textbook=-8.*WAVE*WAVE*v*c*c*SIGMA*LINE/(denominator*denominator)*phase.sin().powi(2);
        let bloch=(bloch_force(at[2],v,c)-bloch_force(at[2],-v,c))/2.;
        println!("kz {phase:.3}: {odd:.6}, the textbook's {textbook:.6}, the optical Bloch equations' {bloch:.6}");
        assert!((odd-textbook).abs()<=5e-3*textbook.abs()+1e-7,"kz {phase}: {odd} against {textbook}");
        assert!((odd-bloch).abs()<=3e-2*bloch.abs()+1e-6,"kz {phase}: {odd} against the optical Bloch equations' {bloch}");
    }
    let hot=parts(&model,cluster,state,[0.,0.,0.3/WAVE],[0.,0.,1e3*SIGMA/WAVE/hm]);assert!(hot.2>=0.,"a fast atom's hazard {}",hot.2);
}
/// Optical pumping turns a rate's first order into a population's, whose rest force is a force of the same order: two ground states, σ⁺ from +z pumps the first to the second and σ⁻ from −z
/// the second to the first, equal branching: the rate equations' p₁ = R₋/(R₊ + R₋) balance the scattering from either beam, F = 0 at any velocity. The generator's first order gives their populations
/// and no force; the rest rate with each jump's mean reweighted by the moving rate (correct on one state: R₀ ρ b̄ = R(v) b̄) leaves p₁ = ½ and the running waves' friction.
#[test]
fn optical_pumping_holds_a_force_of_the_rate_s_order(){
    let(z,up,down,a,b)=([0.,0.,1.],"D2-3-3","D2-3-1","ground-2-2","ground-2-0");let half=0.5*LINE;
    let r=synthetic(&[(up,SIGMA),(down,SIGMA)],&[(up,a,circular(1.),half),(up,b,circular(1.),half),(down,b,circular(-1.),half),(down,a,circular(-1.),half)],&[(z,vec![(up,a,3.)]),(z.map(|x|-x),vec![(down,b,3.)])],a);
    let model=Model::new(&r).unwrap();let(first,second)=(r.states.iter().position(|s|s==a).unwrap(),r.states.iter().position(|s|s==b).unwrap());
    for v in [0.2,0.4]{
        let(force,error,_,p)=held_force(&model,&r,a,[0.;3],[0.,0.,v],8192,300.,0.1,50.);
        let(plus,minus)=(lorentzian(3.,SIGMA+WAVE*v),lorentzian(3.,SIGMA-WAVE*v));let share=minus/(plus+minus);
        println!("v {v}: F {:.4} ± {:.4}, p {:.4} {:.4} against the rate equations' {share:.4} {:.4}",force[2],error[2],p[first],p[second],1.-share);
        assert!(force[2].abs()<=4.*error[2],"v {v}: F {} ± {}",force[2],error[2]);
        assert!((p[first]/(p[first]+p[second])-share).abs()<=0.015,"v {v}: p {} {} against {share}",p[first],p[second]);
    }
}
/// The generator's first order in the kinetic shifts (model.rs # The photons and the Doppler term) is the exact responses' (`Ground::response`): on the tweezer PGC's clusters, every drive folded
/// and at its first explicit rung, at random points, momenta and times, the part of H_eff odd in κ against the dense generator's at the exact responses, to the third order and the tables' interpolation,
/// and the decay is the jumps' Σ L†L to first order -- i(H − H†) − Σ L†L odd in κ is nothing -- where the transfer keeps all it leaves (at a T that makes every entry count, `Model::transfer_tables`).
/// At the request's T what it leaves stays within its share of T: either's error at the thermal velocity, as the friction it makes against the recoil heating (F·v/(R ħk²/m), R the cluster's largest rate there).
#[test]
fn the_generator_s_first_order_is_the_exact_response_s(){
    for allowance in [f64::INFINITY,BUDGET]{for hot in [true,false]{
    let mut r=request(&|_|true);r.times_us=vec![0.,1000.];if hot{r.temperature_uk=1e9;}let model=Model::with(&r,allowance).unwrap();let g=&model.ground;
    let(hm,wave)=(model.hbar_over_mass,model.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max));let thermal=KT_PER_UK*r.temperature_uk*hm;let share=NUMERICAL_SHARE*TEMPERATURE_TARGET;
    for(k,at,kappa,t,psi)in samples(&model,3){
        let n=model.dimension(k);let n0=model.ground_dimension(k);let kappa=kappa.map(|x|x*0.1);let opposite=kappa.map(|x|-x);
        // The generator with the scalar it leaves out, at κ.
        let of=|q:[f64;3]|->Vec<Complex>{let x=model.generator(k,&model.point(at,q),t);let mut h=x.hamiltonian;for a in 0..n{h[a*n+a]+=Complex::new(x.scalar,0.);}h};
        let dense=|q:[f64;3]|->Vec<Complex>{let mut e=g.eliminate(&model.intensities_at_point(at)).unwrap();
            for c in &g.carriers{for &l in &c.beams{e.responses[l]=g.response(&e,l,model.kinetic_shift(l,q)).unwrap();}}
            let d=g.local_at(&e,k,at,t).unwrap().effective;(0..n*n).map(|i|d[(i/n,i%n)]).collect()};
        let(h1,h2,d1,d2)=(of(kappa),of(opposite),dense(kappa),dense(opposite));
        let odd:Vec<Complex>=h1.iter().zip(&h2).map(|(a,b)|(*a-*b)*0.5).collect();let exact:Vec<Complex>=d1.iter().zip(&d2).map(|(a,b)|(*a-*b)*0.5).collect();
        let size=exact.iter().map(|z|z.abs()).fold(0.,f64::max);
        if size==0.{continue;}
        // The jumps at ±κ: the first order of the decay against the photons' on ψ.
        let hazard=|h:&[Complex]|-2.*expectation(&h[..n*n],&psi,n).im;
        let rate=|q:[f64;3],state:&[Complex]|->f64{let p=model.point(at,q);let u=model.photons(k,&p,state,t,None);u.amplitude.iter().zip(&g.clusters[k].classes).map(|(u,c)|{let nt=model.ground_dimension(c.target);u.iter().map(|v|v[..nt].iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>()}).sum()};
        let photons=|q:[f64;3]|rate(q,&psi);
        let(odd_hazard,odd_photons)=((hazard(&h1)-hazard(&h2))/2.,(photons(kappa)-photons(opposite))/2.);
        let entries=(0..n0*n0).map(|i|{let j=(i/n0)*n+i%n0;(odd[j]-exact[j]).abs()}).fold(0.,f64::max);
        if hot{
            assert!(entries<=2e-2*size+1e-12,"cluster {k} at {at:?}: {entries:e} of {size:e}");
            assert!((odd_hazard-odd_photons).abs()<=1e-2*odd_photons.abs().max(1e-3*photons(kappa)),"cluster {k}: the decay's first order {odd_hazard:e} against the photons' {odd_photons:e}");
        }else{
            let largest=(0..n0).map(|b|{let mut e=[Complex::ZERO;MOST_MIXED];e[b]=Complex::ONE;rate(kappa,&e)}).fold(0.,f64::max);
            let speed=hm*kappa.iter().map(|x|x*x).sum::<f64>().sqrt();let on_t=|error:f64|error*thermal/speed/(largest*hm*wave);
            assert!(on_t(entries)<=share&&on_t((odd_hazard-odd_photons).abs())<=share,"cluster {k} at {at:?}: on T {:e} and {:e} of its share {share:e}",on_t(entries),on_t((odd_hazard-odd_photons).abs()));
        }
    }
    }}
}
/// The mean force on a multi-level atom's state is the adiabatic expansion's to first order in v (model.rs # The photons and the Doppler term): −2 Re⟨c_e|∇V|ψ⟩, c_e = −Σ_l a_l (R_l − w_l R2_l) ψ, on the
/// fixture's F = 2 → D2 cooling light and a third beam across it (rank 3), the tweezer weak so that no intensity moves, at random points, velocities and states of its cluster:
/// the part of the engine's force odd in κ -- the Ehrenfest force's and the jumps' mean at the generator's hazard -- less the exact expansion's is the derivative along the path of the momentum the
/// excited amplitude holds, Σ_l′l (k_l′ + k_l) w_l Im M_l′l, M_l′l = a_l′* a_l ⟨R_l′ψ|R_lψ⟩ -- of mean zero: every beam's first order in its place.
#[test]
fn the_mean_force_on_a_state_is_the_adiabatic_expansion_s(){
    let mut r=request(&|b|b["id"]=="tweezer"||b["id"]=="cool-plus"||b["id"]=="cool-minus");
    for b in r.beams.iter_mut(){if b.waist_um.is_some(){for v in b.coupling.real.iter_mut().chain(b.coupling.imag.iter_mut()).chain(b.counter_coupling.real.iter_mut()).chain(b.counter_coupling.imag.iter_mut()){*v*=1e-3;}
        for v in b.outside_shift.real.iter_mut().chain(b.outside_shift.imag.iter_mut()){*v*=1e-6;}}}
    let mut across=r.beams.iter().find(|b|b.id=="cool-plus").unwrap().clone();across.id="across".into();across.name="across".into();across.direction=[0.6,0.8,0.];r.beams.push(across);
    let model=Model::new(&r).unwrap();let g=&model.ground;let carrier=&g.carriers[0];assert_eq!(g.carriers.len(),1);
    let mut next=uniform(11);let(ne,ng)=(g.excited_states.len(),g.ground_states.len());
    for k in (0..model.clusters()).filter(|&k|model.dimension(k)>=2){for _ in 0..8{
        let n=model.dimension(k);let at=[next(),next(),next()];let v=[0.2*next(),0.2*next(),0.2*next()];let kappa=v.map(|x|x/model.hbar_over_mass);
        let mut psi=[Complex::ZERO;MOST_MIXED];for z in psi[..n].iter_mut(){*z=Complex::new(next(),next());}let norm:f64=psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();for z in psi[..n].iter_mut(){*z=*z*(1./norm);}
        let e=g.eliminate(&model.intensities_at_point(at)).unwrap();let states=&g.clusters[k].states;
        // Per beam ζ_l = a_l R_l ψ, its first order η_l = a_l w_l R2_l ψ and ∇a_l V_l ψ (per axis).
        let on=|m:&atom_motion_3d::linear::Dense,i:usize|(0..n).fold(Complex::ZERO,|s,x|s+m[(i,states[x])]*psi[x]);let _=ng;
        let force=|q:[f64;3]|->([f64;3],[f64;3]){
            let p=model.point(at,q);let local=model.generator(k,&p,0.);let ehrenfest=local.force(&psi);let hazard=-2.*expectation(&local.hamiltonian[..n*n],&psi,n).im;
            let photons=model.photons(k,&p,&psi,0.,None);let(mut weight,mut mean)=(0.,[0.;3]);
            for(c,(class,u))in g.clusters[k].classes.iter().zip(&photons.amplitude).enumerate(){let nt=model.ground_dimension(class.target);
                weight+=u.iter().map(|x|x[..nt].iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>();
                for a in 0..3{let mut ax=[0.;3];ax[a]=1.;let(gradient,_)=photons.absorption(c,ax,&[Complex::ZERO;MOST_STATES]);for x in 0..3{for i in 0..nt{mean[x]+=(u[a][i].conj()*gradient[i][x]).im;}}}}
            let engine:[f64;3]=std::array::from_fn(|x|ehrenfest[x]+mean[x]*hazard/weight);
            let mut exact=[0.;3];let mut amplitude=vec![Complex::ZERO;ne];
            for &l in &carrier.beams{let(a,_)=model.geometry[l].field(at);let w=model.kinetic_shift(l,q);for i in 0..ne{amplitude[i]-=a*(on(&e.responses[l],i)-on(&e.derivatives[l],i)*w);}}
            for &l in &carrier.beams{let(_,slope)=model.geometry[l].field(at);for x in 0..3{for i in 0..ne{exact[x]-=2.*(amplitude[i].conj()*slope[x]*on(&g.folded[l],i)).re;}}}
            (engine,exact)};
        let(a,b)=(force(kappa),force(kappa.map(|x|-x)));
        // The momentum the excited amplitude holds, its derivative along the path at the state ψ (odd in κ).
        let mut held=[0.;3];
        for &l1 in &carrier.beams{for &l in &carrier.beams{
            let(a1,_)=model.geometry[l1].field(at);let(a2,_)=model.geometry[l].field(at);let(k1,k2)=(model.geometry[l1].wave_vector(),model.geometry[l].wave_vector());
            let w=(model.kinetic_shift(l,kappa)-model.kinetic_shift(l,kappa.map(|x|-x)))/2.;
            let m=(0..ne).fold(Complex::ZERO,|s,i|s+(a1*on(&e.responses[l1],i)).conj()*(a2*on(&e.responses[l],i)));
            for x in 0..3{held[x]+=(k1[x]+k2[x])*w*m.im;}
        }}
        let odd:[f64;3]=std::array::from_fn(|x|(a.0[x]-b.0[x])/2.);let exact:[f64;3]=std::array::from_fn(|x|(a.1[x]-b.1[x])/2.);
        let size=odd.iter().map(|x|x.abs()).fold(0.,f64::max).max(exact.iter().map(|x|x.abs()).fold(0.,f64::max));
        for x in 0..3{let residual=odd[x]-exact[x];
            assert!((residual-held[x]).abs()<=1e-2*size+1e-7,"cluster {k} at {at:?}, v {v:?}, axis {x}: engine {} less exact {} is {residual}, against the amplitude's momentum {}",odd[x],exact[x],held[x]);}
    }}
}

// The transient after a jump (engine.rs, # The transient).

/// The tetrahedral MOT on a tweezer (tests/fixtures/tetrahedral_mot_motion.json): its near beams 11 Γ off the tensor-split F′ = 3 at the focus, 88.6 rad/us (2.3 Γ) across it.
fn tetrahedral()->(Value,Request){
    let v:Value=serde_json::from_str(include_str!("fixtures/tetrahedral_mot_motion.json")).unwrap();let mut r=sparse_request(&v);
    r.times_us=(0..41).map(|i|5.*i as f64).collect();(v,r)
}
/// The steady ground populations of a frozen atom's master equation at the focus with every jump's transient (model.rs `Model::after_jump`): the null vector of its superoperator on the clusters' densities; and its folded excited population (the elimination's η²).
fn frozen_steady(model:&Model)->(Vec<f64>,f64){
    use atom_motion_3d::linear::{Dense,Lu};
    let g=&model.ground;let r=[0.;3];let e=g.eliminate(&model.intensities_at_point(r)).unwrap();let point=model.point(r,[0.;3]);let count=g.clusters.len();
    let fourths:Vec<_>=(0..count).map(|k|(model.fourth&&!g.mixed(k)).then(||g.fourth(&e,k,r,0.).unwrap())).collect();
    let locals:Vec<_>=(0..count).map(|k|{let mut l=g.local_at(&e,k,r,0.).unwrap();if let Some(f)=&fourths[k]{l.effective=f.effective.clone();l.jumps=f.jumps.clone();}l}).collect();
    let transients:Vec<_>=(0..count).map(|k|model.transient(k,&point,0.)).collect();
    let rings:Vec<Vec<_>>=(0..count).map(|k|(0..g.excited_groups.len()).map(|m|model.ringing(k,r,m)).collect()).collect();
    let sizes:Vec<usize>=g.clusters.iter().map(|c|c.dimension()).collect();let offsets:Vec<usize>=sizes.iter().scan(0,|o,&n|{let x=*o;*o+=n*n;Some(x)}).collect();let d:usize=sizes.iter().map(|n|n*n).sum();
    let mut generator=Dense::zeros(d,d);
    for(k,&n)in sizes.iter().enumerate(){for column in 0..n*n{
        let rho=Dense::from_fn(n,n,|a,b|if a*n+b==column{Complex::ONE}else{Complex::ZERO});
        let h=&locals[k].effective;let mut image:Vec<Dense>=sizes.iter().map(|&m|Dense::zeros(m,m)).collect();
        image[k].add_scaled(Complex::new(0.,-1.),&h.product(&rho));image[k].add_scaled(Complex::I,&rho.product(&h.adjoint()));
        for(class,jumps)in g.clusters[k].classes.iter().zip(&locals[k].jumps){
            let b=class.target;let nt=g.clusters[b].states.len();let mut sigma=Dense::zeros(nt,nt);
            for j in jumps{sigma.add_scaled(Complex::ONE,&j.product(&rho).product(&j.adjoint()));}
            for(c,m)in model.after_jump(&e,r,b,&sigma,&transients[b],&rings[b],fourths[b].as_ref().map(|f|&f.slow),0.).iter().enumerate(){for i in 0..m.rows{for j in 0..m.cols{image[c][(i,j)]+=m[(i,j)];}}}
        }
        for(j,x)in image.iter().enumerate(){for(q,z)in x.entries.iter().enumerate(){generator[(offsets[j]+q,offsets[k]+column)]=*z;}}
    }}
    // The trace's row in the place of the first, the singular generator's one dependent row.
    for j in 0..d{generator[(0,j)]=Complex::ZERO;}for(k,&n)in sizes.iter().enumerate(){for a in 0..n{generator[(0,offsets[k]+a*n+a)]=Complex::ONE;}}
    let mut x=vec![Complex::ZERO;d];x[0]=Complex::ONE;Lu::new(&generator).unwrap().solve(&mut x);
    let(mut populations,mut excited)=(vec![0.;STATES],0.);
    for(k,&n)in sizes.iter().enumerate(){let rho=Dense::from_fn(n,n,|a,b|x[offsets[k]+a*n+b]);
        let p=match &fourths[k]{Some(f)=>g.populations_fourth(k,f,&rho),None=>g.populations(k,&locals[k],&rho)};for(a,b)in populations.iter_mut().zip(p){*a+=b;}}
    for &i in &g.excited_states{excited+=populations[i];}
    (populations,excited)
}
/// The defect these fix (engine.rs, # The transient): TET's frozen atom in its slow mode, |2,1⟩, against the rung with F′ = 3 explicit. The ringing photon at each pair's true coherence takes
/// the second order from 0.0134 to 0.0029 and the fourth from 0.0150 to 0.0004 (the bare line's second photon, fully coherent over the tensor split, came back into the mode 96 % of the time, the explicit rung's 66 %),
/// and the fourth is within its booking, S² + ς S, S = 2η² (the fold's charge, solve.rs `fold_charges`).
#[test]
fn a_tensor_split_line_s_second_photon_lands_as_the_explicit_rung_s(){
    let(_,r)=tetrahedral();let ladder=Ground::allowances(&r).unwrap();
    let(explicit,_)=frozen_steady(&Model::with(&r,ladder[4]).unwrap());let ground:Vec<usize>=(0..STATES).filter(|&i|r.states[i].starts_with("ground")).collect();
    let error=|p:&[f64]|ground.iter().map(|&i|(p[i]-explicit[i]).abs()).fold(0.,f64::max);
    let(second,_)=frozen_steady(&Model::with_order(&r,f64::INFINITY,false).unwrap());assert!(error(&second)<=0.0035,"second order: {}",error(&second));
    let model=Model::with_order(&r,f64::INFINITY,true).unwrap();let(fourth,excited)=frozen_steady(&model);let s=2.*excited;
    assert!(error(&fourth)<=0.0006,"fourth order: {}",error(&fourth));
    assert!(error(&fourth)<=s*s+model.ringing_spread*s,"fourth order: {} against S {s} and the spread {}",error(&fourth),model.ringing_spread);
}
/// The engine's unraveling of the transient keeps its master equation's populations: TET's frozen atom over t ≥ 100 us against the population core's, its |2,1⟩ at either order within 0.006 --
/// 0.7987, the engine without this photon 0.8134 at second order and with a photon coherent over the tensor split 0.8139 at fourth --, its statistics about 0.0015.
#[test]
fn a_tetrahedral_mot_s_frozen_atom_keeps_the_core_s_populations(){
    use atom_motion_3d::simulate_from;
    let(v,r)=tetrahedral();let states:Vec<&str>=v["core"]["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap()).collect();let core=floats(&v["core"]["populations"]);
    let at=r.states.iter().position(|s|s=="ground-2-1").unwrap();let expected=core[states.iter().position(|&s|s=="ground-2-1").unwrap()];
    for fourth in [false,true]{
        let model=Model::with_order(&r,f64::INFINITY,fourth).unwrap();let initial=Initial::new(&model,&r).unwrap();
        let mut s=Settings::new(0.5,&initial.traps[0],1e-6,false);s.free=[false;3];s.loss=false;
        let outcome=simulate_from(&model,&s,&initial,7,0,16384,8,10,41,&[],[0.;3]);
        let means=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:[1.;3]},&[]);
        let p:f64=means[20..].iter().map(|e|e.populations[at]).sum::<f64>()/(means.len()-20) as f64;
        assert!((p-expected).abs()<=0.006,"fourth {fourth}: {p} against the core's {expected}");
    }
}
/// The ringing photon's mean over its time (`Model::ringing_photon`, τ ~ Γ e^{−Γτ}) is its landing (`Model::ringing_landing`): each pair's coherence Γ/(Γ + iδ), through H's off-diagonal part
/// and the landing states' frames -- on the tetrahedral MOT and on a molasses with a probe, whose cluster's frames differ --, and its trace the excited amplitude ‖v‖².
#[test]
fn the_ringing_photon_s_mean_over_its_time_is_its_landing(){
    let probe:Value=serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap();let mut with_probe=scene::request(&probe);with_probe.times_us=vec![0.,1.];
    for r in [tetrahedral().1,with_probe]{
        let model=Model::new(&r).unwrap();let g=&model.ground;let mut next=uniform(3);let time=0.37;let p=model.point([0.05,0.1,0.2],[0.;3]);
        for k in 0..model.clusters(){
            let q=model.transient(k,&p,time);let(n0,ne)=(q.n0,q.ne);let mut psi=vec![Complex::ZERO;n0];for z in psi.iter_mut(){*z=Complex::new(next(),next());}
            let norm:f64=psi.iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();for z in psi.iter_mut(){*z=*z*(1./norm);}
            let sigma=atom_motion_3d::linear::Dense::from_fn(n0,n0,|a,b|psi[a]*psi[b].conj());let rings:Vec<_>=(0..g.excited_groups.len()).map(|m|model.ringing(k,p.r,m)).collect();
            let landing=model.ringing_landing(&q,&rings,&sigma,time);
            // The photon's density over the quantiles of its time, each group's and manifold's.
            let mut mean:Vec<atom_motion_3d::linear::Dense>=g.clusters.iter().map(|c|atom_motion_3d::linear::Dense::zeros(c.states.len(),c.states.len())).collect();let count=4096;
            for x in &q.groups{
                let v:Vec<Complex>=(0..ne).map(|i|(0..n0).fold(Complex::ZERO,|s,b|s+x[i*n0+b]*psi[b])).collect();
                for ring in &rings{
                    for j in 0..count{
                        let tau=-(1.-(j as f64+0.5)/count as f64).ln()/ring.width;let ops=model.ringing_photon(ring,tau,time);
                        for op in &ops{for(c,cluster)in g.clusters.iter().enumerate(){
                            let u:Vec<Complex>=cluster.states.iter().map(|&to|ring.levels.iter().fold(Complex::ZERO,|s,&i|s+op[(to,i)]*v[i])).collect();
                            for a in 0..u.len(){for b in 0..u.len(){mean[c][(a,b)]+=u[a]*u[b].conj()*(1./(ring.width*count as f64));}}
                        }}
                    }
                }
            }
            let total=|m:&[atom_motion_3d::linear::Dense]|m.iter().map(|x|(0..x.rows).map(|a|x[(a,a)].re).sum::<f64>()).sum::<f64>();
            let amplitude:f64=q.groups.iter().map(|x|(0..ne).map(|i|(0..n0).fold(Complex::ZERO,|s,b|s+x[i*n0+b]*psi[b]).norm_sqr()).sum::<f64>()).sum();
            assert!((total(&landing)-amplitude).abs()<=1e-3*amplitude,"cluster {k}: trace {} against ‖v‖² {amplitude}",total(&landing));
            assert!(amplitude<=model.ringing_bound(k,&p,&psi)*(1.+model.table_error),"cluster {k}: ‖v‖² {amplitude} beyond the bound {}",model.ringing_bound(k,&p,&psi));
            for(c,(a,b))in landing.iter().zip(&mean).enumerate(){for i in 0..a.rows{for j in 0..a.cols{
                assert!((a[(i,j)]-b[(i,j)]).abs()<=3e-3*amplitude,"cluster {k}, landing {c} ({i}, {j}): {:?} against the photon's mean {:?}, of ‖v‖² {amplitude}",a[(i,j)],b[(i,j)]);}}}
        }
    }
}
/// The delay's weight (engine.rs, # The transient): at fourth order the slow part's weight and the ringing's probability exceed one by what the steady amplitude's photons lose while it builds, A + p − 1 = Σ (1 − κ_i) |v_i|²,
/// 1 − κ_i = Γ_i²/(Δ_i² + Γ_i²/4) (ground.rs `state_factor`), within the fold's next order and never below it.
#[test]
fn the_slow_part_s_weight_and_the_ringing_s_sum_to_the_delay(){
    let(_,r)=tetrahedral();let model=Model::with_order(&r,f64::INFINITY,true).unwrap();let g=&model.ground;let e=g.eliminate(&model.intensities_at_point([0.;3])).unwrap();let point=model.point([0.;3],[0.;3]);
    for k in 0..model.clusters(){
        let n=model.dimension(k);let f=g.fourth(&e,k,[0.;3],0.).unwrap();let q=model.transient(k,&point,0.);let ne=q.ne;
        for b in 0..n{
            let a:f64=(0..n).map(|i|f.slow[(i,b)].norm_sqr()).sum();
            let(mut p,mut w)=(0.,0.);
            for(x,group)in q.groups.iter().zip(&g.clusters[k].coherent){let(ch,_)=group[0];for i in 0..ne{let v=x[i*n+b].norm_sqr();p+=v;w+=(1.-g.state_factor(&e,i,ch.carrier,ch.state))*v;}}
            let excess=a+p-1.;
            assert!(excess>=0.&&(excess-w).abs()<=0.05*p*p.max(0.1),"cluster {k} state {b}: A {a} + p {p} − 1 = {excess} against Σ (1 − κ)|v|² {w}");
        }
    }
}
