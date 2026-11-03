//! The GPU kernel's cost per part on a scene: a solve on the GPU with the kernels compiled with their
//! profiling spans (`compile_ptx -- <nvrtc64_120_0.dll> --profile <file>`), and per trajectory the
//! cycles each part took -- per advance, per jump, per output -- and its share of the warp's time.
//! `cargo run --profile iterate -p atom-motion-3d --example profile -- <profiled.ptx> <scene> <duration_us> <samples>`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::gpu::{self,SPANS};
use atom_motion_3d::solve::{solve,Device};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();
    let ptx=std::fs::read_to_string(&args[1]).unwrap();let name=&args[2];let duration:f64=args[3].parse().unwrap();let samples:usize=args[4].parse().unwrap();
    gpu::install_profiled(&ptx).unwrap();
    let mut request=scene::scene(name);request.times_us=(0..samples).map(|i|duration*i as f64/(samples-1) as f64).collect();
    let clock=Instant::now();let s=solve(&request,Device::Gpu,&mut |_,_|{}).unwrap();let wall=clock.elapsed().as_secs_f64();
    let g=gpu::gpu().unwrap();let g=g.lock().unwrap();let c=g.spans().expect("a profiled kernel");
    let at=|n:&str|c[SPANS.iter().position(|&x|x==n).unwrap()] as f64;
    let n=s.trajectories as f64;let(total,advances,jumps)=(at("total"),at("advances"),at("jumps"));
    println!("{name} {duration} us on the GPU ({:?}{}): {wall:.2} s, {} trajectories, dt {:.4} us; per trajectory {:.0} advances, {:.1} jumps, {:.3e} cycles",
        s.device,s.device_note.as_ref().map_or(String::new(),|x|format!(": {x}")),s.trajectories,s.time_step_us,advances/n,jumps/n,total/n);
    let share=|x:f64|100.*x/total;
    for part in ["point","generator","exponential","evolve"]{println!("  {part:12} {:8.0} cycles per advance  {:5.1}%",at(part)/advances,share(at(part)));}
    println!("  {:12} {:8.0} cycles per jump     {:5.1}%","jump",at("jump")/jumps.max(1.),share(at("jump")));
    println!("  {:12} {:8.0} cycles per advance  {:5.1}%","loss",at("loss")/advances,share(at("loss")));
    println!("  {:12} {:8.0} cycles per advance  {:5.1}%  (all of a step: the parts above and its control)","step",at("step")/advances,share(at("step")));
    println!("  {:12} {:8.0} cycles per output   {:5.1}%","readout",at("readout")/(n*samples as f64),share(at("readout")));
    println!("  {:12} {:8.0} cycles per output   {:5.1}%  (the windows, the block's sums and its waits)","windows",at("windows")/(n*samples as f64),share(at("windows")));
}
