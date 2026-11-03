//! The step's parts against their counted work on one thread (a diagnostic, for the step choice's
//! cost model, solve.rs): states along a scene's trajectories at dt; per state the point, the
//! generator, the force, a half step's propagation and a jump's photons, timed, and their work
//! (`Model::work`, the series' sub-steps × terms × n²); and the whole step with its jumps.
//! `cargo run --profile iterate -p atom-motion-3d --example bench_cost -- <scene> <dt_us> <truncation> [repeats]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::{column_norm,propagate,series,Engine,Settings};
use atom_motion_3d::model::Model;
use atom_motion_3d::{initial_trajectory,Initial,Stratum};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();let name=args[1].clone();let dt:f64=args[2].parse().unwrap();let truncation:f64=args[3].parse().unwrap();
    let repeats:usize=args.get(4).map_or(7,|s|s.parse().unwrap());
    let request=scene::scene(&name);let model=Model::new(&request).unwrap();let initial=Initial::new(&model,&request).unwrap();
    let mut settings=Settings::new(dt,&initial.traps[0],truncation,false);settings.loss=false;
    let engine=Engine::new(&model,settings.clone());
    // States along 16 trajectories, every 25 steps over 2000; the whole step timed there with its jumps.
    let(mut states,mut jumps,mut steps)=(vec![],0u64,0u64);let clock=Instant::now();
    for index in 0..16u64{let mut t=initial_trajectory(&model,&settings,&initial,9,index,Stratum::single(index));
        for k in 0..2000{let before=t.jumps;engine.step(&mut t);jumps+=t.jumps-before;steps+=1;if k%25==0{states.push(t.clone());}}}
    let whole=clock.elapsed().as_secs_f64()/steps as f64*1e6;let per_step=jumps as f64/steps as f64;
    let points:Vec<_>=states.iter().map(|t|model.point(t.r,t.kappa)).collect();
    let locals:Vec<_>=states.iter().zip(&points).map(|(t,p)|model.local(t.cluster,p,t.time)).collect();
    let mut sink=0.;
    let mut time=|f:&mut dyn FnMut(usize)->f64|->f64{let mut best=f64::INFINITY;for _ in 0..repeats{let clock=Instant::now();for i in 0..states.len(){sink+=f(i);}best=best.min(clock.elapsed().as_secs_f64());}best/states.len() as f64*1e6};
    let point=time(&mut|i|{let t=&states[i];model.point([t.r[0]+1e-9,t.r[1],t.r[2]],t.kappa).r[0]});
    let local=time(&mut|i|{let t=&states[i];model.local(t.cluster,&points[i],t.time).hamiltonian[0].re});
    let force=time(&mut|i|{let t=&states[i];model.force(t.cluster,&locals[i],&t.psi)[0]});
    let half=time(&mut|i|{let t=&states[i];let n=model.dimension(t.cluster);let mut psi=t.psi;propagate(&locals[i].hamiltonian,n,0.5*dt,&mut psi,truncation);psi[0].re});
    let photons=time(&mut|i|{let t=&states[i];model.photons(t.cluster,&points[i],&t.psi,t.time,None).amplitude.len() as f64});
    let advance=time(&mut|i|{let t=&states[i];engine.coast(t,dt,1).kappa[0]});
    let mean=|f:&dyn Fn(usize)->f64|->f64{(0..states.len()).map(f).sum::<f64>()/states.len() as f64};
    let(generator,forces,photon)=(mean(&|i|model.work(states[i].cluster).generator),mean(&|i|model.work(states[i].cluster).force),mean(&|i|model.work(states[i].cluster).photons));
    let taylor=mean(&|i|{let t=&states[i];let n=model.dimension(t.cluster);let(parts,terms)=series(column_norm(&locals[i].hamiltonian,n),0.5*dt,truncation);(parts*terms*n*n) as f64});
    println!("{name} dt {dt} truncation {truncation:.1e}: whole step {whole:.3} us, jumps/step {per_step:.4}");
    println!("  time (us): point {point:.3}, generator {local:.3}, force {force:.3}, half step {half:.3}, photons {photons:.3}");
    println!("  work (MACs): generator {generator:.0}, force {forces:.0}, half step {taylor:.0}, photons {photon:.0}");
    println!("  ns per MAC: generator {:.3}, force {:.3}, half step {:.3}, photons {:.3}; advance {advance:.3} us, its rest {:.3} us; beams {} [{sink:.1e}]",1e3*local/generator,1e3*force/forces,1e3*half/taylor,1e3*photons/photon,
        advance-(point+local+force+2.*half),model.geometry.len());
}
