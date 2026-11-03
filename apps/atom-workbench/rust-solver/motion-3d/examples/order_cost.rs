//! The fold's second and fourth order on a scene: CPU time per step of the same initial trajectories
//! (one thread each, run back to back), and the jumps.
//! `cargo run --profile iterate -p atom-motion-3d --example order_cost -- <scene> [steps] [dt_us]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::{Engine,Settings};
use atom_motion_3d::model::Model;
use atom_motion_3d::{initial_trajectory,Initial,Stratum};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();let name=args.get(1).cloned().unwrap_or("PGC3D".into());
    let steps:usize=args.get(2).map_or(5000,|s|s.parse().unwrap());let dt:f64=args.get(3).map_or(0.23,|s|s.parse().unwrap());
    let request=scene::scene(&name);
    for fourth in [false,true]{
        let model=Model::with_order(&request,f64::INFINITY,fourth).unwrap();let initial=Initial::new(&model,&request).unwrap();
        let mut settings=Settings::new(dt,&initial.traps[0],1e-6,false);settings.loss=false;let engine=Engine::new(&model,settings.clone());
        let(mut seconds,mut jumps)=(0.,0);
        for index in 0..8{
            let mut t=initial_trajectory(&model,&settings,&initial,0x3D_5EED,index,Stratum::single(index));
            let clock=Instant::now();for _ in 0..steps{engine.step(&mut t);}seconds+=clock.elapsed().as_secs_f64();jumps+=t.jumps;
        }
        println!("{name} fourth {fourth}: {:.2} us per step, {:.3} jumps per step",seconds/(8*steps) as f64*1e6,jumps as f64/(8*steps) as f64);
    }
}
