//! The cost of one step's parts on one thread: the point, the generator, the propagation and the
//! force, for a thermal atom of a scene.
//! `cargo run --profile iterate -p atom-motion-3d --example bench_step -- <scene> [steps]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::{propagate,Engine,Settings};
use atom_motion_3d::model::Model;
use atom_motion_3d::{initial_trajectory,Initial};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();let name=args.get(1).cloned().unwrap_or("TW2".into());
    let steps:usize=args.get(2).map_or(20000,|s|s.parse().unwrap());
    let request=scene::scene(&name);
    let model=Model::new(&request).unwrap();
    let settings=Settings{dt:0.2,free:[true;3],escape_intensity:1e-2,frequencies:[0.;3]};
    let initial=Initial::new(&model,&request).unwrap();
    let mut t=initial_trajectory(&model,&settings,&initial,1,0);
    let n=model.dimension(t.block);
    let clock=Instant::now();let mut sink=0.;
    for i in 0..steps{let p=model.point([t.r[0]+1e-6*i as f64,t.r[1],t.r[2]],t.kappa);sink+=p.fields[0].e[0].re;}
    let point=clock.elapsed().as_secs_f64()/steps as f64;
    let p=model.point(t.r,t.kappa);
    let clock=Instant::now();
    for _ in 0..steps{let l=model.local(t.block,&p);sink+=l.hamiltonian[0].re;}
    let local=clock.elapsed().as_secs_f64()/steps as f64;
    let l=model.local(t.block,&p);
    let clock=Instant::now();
    for _ in 0..steps{let mut psi=t.psi;propagate(&l.hamiltonian,n,0.1,&mut psi);sink+=psi[0].re;}
    let prop=clock.elapsed().as_secs_f64()/steps as f64;
    let clock=Instant::now();
    for _ in 0..steps{let f=model.force(t.block,&p,&l,&t.psi);sink+=f[0];}
    let force=clock.elapsed().as_secs_f64()/steps as f64;
    let clock=Instant::now();
    for _ in 0..steps/10{let ph=model.photons(t.block,&p,&t.psi);sink+=ph.amplitude[0][0][0].re;}
    let photons=clock.elapsed().as_secs_f64()/(steps/10) as f64;
    let clock=Instant::now();
    for _ in 0..steps/10{let q=model.populations(t.block,&p,&t.psi);sink+=q[0];}
    let populations=clock.elapsed().as_secs_f64()/(steps/10) as f64;
    let engine=Engine{model:&model,settings};
    let clock=Instant::now();
    for _ in 0..steps{engine.step(&mut t);}
    let step=clock.elapsed().as_secs_f64()/steps as f64;
    println!("{name}: point {:.2} us, local {:.2} us, propagate (dt/2) {:.2} us, force {:.2} us, photons {:.2} us, populations {:.2} us; whole step {:.2} us ({} jumps) [{sink:.1e}]",
        point*1e6,local*1e6,prop*1e6,force*1e6,photons*1e6,populations*1e6,step*1e6,t.jumps);
}
