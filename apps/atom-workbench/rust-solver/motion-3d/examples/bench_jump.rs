//! A jump's parts on one thread (a diagnostic): at states along a scene's trajectories, the whole jump
//! (`Engine::jump`), the photons of every class (`Model::photons`), and for the class of the largest
//! weight its absorbed light (`Photons::absorption`); and the whole step with its jumps.
//! `cargo run --profile iterate -p atom-motion-3d --example bench_jump -- <scene> <dt_us> <truncation> [repeats]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::{dipole_direction,Engine,Settings};
use atom_motion_3d::rng::{Purpose,Rng};
use atom_motion_3d::Complex;
use atom_motion_3d::model::Model;
use atom_motion_3d::{initial_trajectory,Initial,Stratum};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();let name=args[1].clone();let dt:f64=args[2].parse().unwrap();let truncation:f64=args[3].parse().unwrap();
    let repeats:usize=args.get(4).map_or(7,|s|s.parse().unwrap());
    let request=scene::scene(&name);let model=Model::new(&request).unwrap();let initial=Initial::new(&model,&request).unwrap();
    let mut settings=Settings::new(dt,&initial.traps[0],truncation,false);settings.loss=false;
    let engine=Engine::new(&model,settings.clone());
    let(mut states,mut jumps,mut steps)=(vec![],0u64,0u64);let clock=Instant::now();
    for index in 0..16u64{let mut t=initial_trajectory(&model,&settings,&initial,9,index,Stratum::single(index));
        for k in 0..2000{let before=t.jumps;engine.step(&mut t);jumps+=t.jumps-before;steps+=1;if k%25==0{states.push(t.clone());}}}
    let whole=clock.elapsed().as_secs_f64()/steps as f64*1e6;let per_step=jumps as f64/steps as f64;
    let normalized:Vec<_>=states.iter().map(|t|{let n=model.dimension(t.cluster);let norm:f64=t.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();let mut psi=t.psi;for z in psi[..n].iter_mut(){*z=*z*(1./norm);}psi}).collect();
    let points:Vec<_>=states.iter().map(|t|model.point(t.r,t.kappa)).collect();
    let mut sink=0.;
    let mut time=|f:&mut dyn FnMut(usize)->f64|->f64{let mut best=f64::INFINITY;for _ in 0..repeats{let clock=Instant::now();for i in 0..states.len(){sink+=f(i);}best=best.min(clock.elapsed().as_secs_f64());}best/states.len() as f64*1e6};
    let jump=time(&mut|i|{let mut t=states[i].clone();let r=t.r;let time=t.time;engine.jump(&mut t,r,time,false);t.kappa[0]});
    let photons=time(&mut|i|{let t=&states[i];model.photons(t.cluster,&points[i],&normalized[i],t.time,None).amplitude.len() as f64});
    // The class of the largest weight at each state.
    let picks:Vec<usize>=states.iter().enumerate().map(|(i,t)|{let p=model.photons(t.cluster,&points[i],&normalized[i],t.time,None);
        let classes=&model.ground.clusters[t.cluster].classes;(0..classes.len()).max_by(|&a,&b|{let w=|c:usize|{let nt=model.dimension(classes[c].target);(0..3).map(|x|p.amplitude[c][x][..nt].iter().map(|z|z.norm_sqr()).sum::<f64>()).sum::<f64>()};w(a).total_cmp(&w(b))}).unwrap()}).collect();
    let absorption=time(&mut|i|{let t=&states[i];let p=model.photons(t.cluster,&points[i],&normalized[i],t.time,None);let(g,s)=p.absorption(picks[i],[0.6,0.,0.8]);g[0][0].re+s.len() as f64});
    // The jump's other parts: its point, the dipole pattern's rejection on the picked class's Gram
    // matrix, and an advance (a jump redoes two in Engine::step: to the crossing and the rest).
    let grams:Vec<[[Complex;3];3]>=states.iter().enumerate().map(|(i,t)|{let p=model.photons(t.cluster,&points[i],&normalized[i],t.time,None);let u=&p.amplitude[picks[i]];
        let nt=model.dimension(model.ground.clusters[t.cluster].classes[picks[i]].target);std::array::from_fn(|a|std::array::from_fn(|b|(0..nt).fold(Complex::ZERO,|s,j|s+u[a][j].conj()*u[b][j])))}).collect();
    let point=time(&mut|i|{let t=&states[i];model.point(t.r,t.kappa).r[0]});
    let mut rng=Rng::new(3,Purpose::Jump,0,0);
    let direction=time(&mut|i|dipole_direction(&mut rng,&grams[i])[0]);
    let advance=time(&mut|i|{let t=&states[i];engine.coast(t,dt,1).kappa[0]});
    let classes=states.iter().map(|t|model.ground.clusters[t.cluster].classes.len()).sum::<usize>() as f64/states.len() as f64;
    let photons_per_class=states.iter().zip(&picks).map(|(t,&c)|model.ground.clusters[t.cluster].classes[c].photons.len()).sum::<usize>() as f64/states.len() as f64;
    println!("{name} dt {dt}: whole step {whole:.3} us, jumps/step {per_step:.4}; classes {classes:.1}, the picked class's photons {photons_per_class:.1}");
    println!("  jump {jump:.3} us: photons {photons:.3} us, + absorption {:.3} us, rest {:.3} us (its point {point:.3} us, the dipole pattern's rejection {direction:.3} us); an advance {advance:.3} us (a jump redoes two) [{sink:.1e}]",absorption-photons,jump-absorption);
}
