//! A scene's motion at given time steps, as a solve runs it (T over a trap period, the solve's seed
//! and groups) but at a fixed number of trajectories: T per axis and the F = 1 population at a few
//! times for each step, to see a step's bias against its standard errors.
//! `cargo run --profile iterate -p atom-motion-3d --example steps -- <scene> <trajectories> <t_end_us> <interval_us> <dt_us>...`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::Settings;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Scales};
use atom_motion_3d::{simulate,Initial};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();
    let name=args[1].clone();let count:u64=args[2].parse().unwrap();let t_end:f64=args[3].parse().unwrap();let interval:f64=args[4].parse().unwrap();
    let request=scene::scene(&name);let model=Model::new(&request).unwrap();let initial=Initial::new(&model,&request).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();let trap=initial.traps[reference];
    let outputs=(t_end/interval).round() as usize+1;let threads=rayon::current_num_threads() as u64;
    let f1:Vec<usize>=request.states.iter().enumerate().filter(|(_,s)|s.starts_with("ground-1-")).map(|(i,_)|i).collect();
    for dt in args[5..].iter().map(|x|x.parse::<f64>().unwrap()){
        let steps=(interval/dt).round().max(1.) as usize;let settings=Settings::new(interval/steps as f64,&trap,f64::EPSILON,true);
        let clock=Instant::now();let outcome=simulate(&model,&settings,&initial,0x3D_5EED,0,count,(count/(2*threads)).max(1),steps,outputs,&[]);
        let e=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab});
        println!("{name} dt {:.4} us: {:.1} s",settings.dt,clock.elapsed().as_secs_f64());
        for i in (0..outputs).step_by((outputs/8).max(1)).chain([outputs-1]){
            let x=&e[i];let p:f64=f1.iter().map(|&k|x.populations[k]).sum();
            println!("  t {:7.0}  T {:6.2}±{:4.2} {:6.2}±{:4.2} {:6.2}±{:4.2}  Ttot {:6.2}±{:4.2}  pF1 {:.4}  loss {:.1e}",i as f64*interval,
                x.temperature[0],x.temperature_error[0],x.temperature[1],x.temperature_error[1],x.temperature[2],x.temperature_error[2],x.temperature_total,x.temperature_total_error,p,x.loss);
        }
    }
}
