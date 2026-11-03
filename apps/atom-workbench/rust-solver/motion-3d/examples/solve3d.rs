//! A full 3D solve as the app makes it (`solve::solve`): the step-doubling check, the batches to the
//! targets, the checkpoints with their remaining-time estimates; the result per axis at a few times.
//! `cargo run --profile iterate -p atom-motion-3d --example solve3d -- <scene> [duration_us] [samples] [cpu|gpu]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::solve::{solve,Device};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let name=arg(1,"PGC3");let duration:f64=arg(2,"10000").parse().unwrap();let samples:usize=arg(3,"201").parse().unwrap();
    let device=if arg(4,"cpu")=="gpu"{Device::Gpu}else{Device::Cpu};
    let mut request=scene::scene(&name);
    request.times_us=(0..samples).map(|i|duration*i as f64/(samples-1) as f64).collect();
    let started=Instant::now();
    let mut observe=|s:&atom_motion_3d::solve::Solution,remaining:Option<f64>|{
        println!("checkpoint at {:.1} s: {} trajectories, dt {:.4} us, estimated remaining {:.1} s",started.elapsed().as_secs_f64(),s.trajectories,s.time_step_us,remaining.unwrap_or(f64::NAN)/1e3);
    };
    let s=solve(&request,device,&mut observe).unwrap();
    let wall=started.elapsed().as_secs_f64();
    println!("{name}: {:.1} s wall on {:?}{}, {} trajectories, dt {:.4} us, trap {:?} kHz; {}",wall,s.device,s.device_note.as_ref().map_or(String::new(),|n|format!(" ({n})")),
        s.trajectories,s.time_step_us,s.trap_frequency_khz,s.reason.clone().unwrap_or_default());
    let pick=|label:&str|request.states.iter().position(|x|x==label).unwrap();
    let(m22,f1)=(pick("ground-2-2"),[pick("ground-1--1"),pick("ground-1-0"),pick("ground-1-1")]);
    for i in (0..samples).step_by((samples/10).max(1)).chain([samples-1]){
        let t=&s.temperature_uk[i];let e=&s.temperature_errors_uk[i];let n=&s.mean_occupation[i];
        println!("t {:7.0}  T (frame 0,1,2) {:6.2}±{:4.2} {:6.2}±{:4.2} {:6.2}±{:4.2}  nbar {:5.2} {:5.2} {:5.2}  m22 {:.4}  pF1 {:.4}  loss {:.1e}  2SE(pop) {:.3}  classical {}",
            request.times_us[i],t[0],e[0],t[1],e[1],t[2],e[2],n[0],n[1],n[2],s.populations[i][m22],f1.iter().map(|&k|s.populations[i][k]).sum::<f64>(),s.loss_probability[i],2.*s.sample_errors[i],s.classical[i]);
    }
}
