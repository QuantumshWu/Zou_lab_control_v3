//! A scene solved on both devices back to back (`solve::solve`), their wall times and their agreement:
//! per output after t = 0, every population's and the loss's difference against the two solves'
//! combined standard error (each the largest over the populations and the loss, `sample_errors`),
//! and each T's (per axis and in total) against theirs.
//! `cargo run --profile iterate -p atom-motion-3d --example devices -- <scene> <duration_us> <samples> [gpu-first|warm]`
//! (`warm`: the GPU started before either solve, as in a worker that has served one).
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::gpu;
use atom_motion_3d::solve::{solve,Device,Solution};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();
    let name=args[1].clone();let duration:f64=args[2].parse().unwrap();let samples:usize=args[3].parse().unwrap();let gpu_first=args.get(4).is_some_and(|a|a=="gpu-first");
    if args.get(4).is_some_and(|a|a=="warm"){let _=gpu::gpu();}
    let mut request=scene::scene(&name);
    request.times_us=(0..samples).map(|i|duration*i as f64/(samples-1) as f64).collect();
    let run=|device:Device|->(Solution,f64,f64){
        let(clock,cpu)=(Instant::now(),scene::cpu_seconds());let s=solve(&request,device,&mut |_,_|{}).unwrap();(s,clock.elapsed().as_secs_f64(),scene::cpu_seconds()-cpu)
    };
    let order=if gpu_first{[Device::Gpu,Device::Cpu]}else{[Device::Cpu,Device::Gpu]};
    let mut out=vec![];for d in order{out.push(run(d));}
    let(cpu,gpu_run)=if gpu_first{(&out[1],&out[0])}else{(&out[0],&out[1])};
    let mut kernel=0.;
    if let Ok(g)=gpu::gpu(){let g=g.lock().unwrap();let(r,l,b)=g.resources().unwrap();kernel=g.kernel_seconds();
        println!("GPU {} (compute {}.{}, {} SMs), driver CUDA {}: startup driver {:.0} ms, context {:.0} ms, module {:.0} ms; kernel {r} registers, {l} B spilled, {b} blocks/SM",
            g.info.name,g.info.compute_capability.0,g.info.compute_capability.1,g.info.multiprocessors,g.driver_version,g.startup.driver_ms,g.startup.context_ms,g.startup.module_ms);}
    for(label,(s,wall,cpu_time))in [("CPU",cpu),("GPU",gpu_run)]{
        let on_gpu=if label=="GPU"{format!(", GPU kernels {kernel:.3} s")}else{String::new()};
        println!("{name} {label}: wall {wall:.3} s, CPU {cpu_time:.2} s{on_gpu}, on {:?}{}, {} trajectories, dt {:.4} us",s.device,s.device_note.as_ref().map_or(String::new(),|n|format!(" ({n})")),s.trajectories,s.time_step_us);
    }
    let(c,g)=(&cpu.0,&gpu_run.0);
    let z=|d:f64,a:f64,b:f64|d.abs()/(a*a+b*b).sqrt().max(1e-300);
    let(mut pz,mut pcount,mut pover,mut pmax)=(0f64,0usize,0usize,0f64);
    for i in 1..samples{for k in 0..c.populations[i].len(){
        let d=g.populations[i][k]-c.populations[i][k];let x=z(d,c.sample_errors[i],g.sample_errors[i]);pz=pz.max(x);pcount+=1;pover+=(x>2.) as usize;pmax=pmax.max(d.abs());
    }}
    let(mut lz,mut lmax)=(0f64,0f64);for i in 1..samples{let d=g.loss_probability[i]-c.loss_probability[i];lz=lz.max(z(d,c.sample_errors[i],g.sample_errors[i]));lmax=lmax.max(d.abs());}
    println!("populations: largest |Δp| {pmax:.4}, largest z {pz:.2}, z > 2 in {pover} of {pcount}; loss: largest |Δ| {lmax:.2e}, z {lz:.2}");
    for a in 0..4{
        let(mut tz,mut over,mut count,mut rel)=(0f64,0,0,0f64);
        for i in 1..samples{
            let(tc,tg,ec,eg)=if a<3{(c.temperature_uk[i][a],g.temperature_uk[i][a],c.temperature_errors_uk[i][a],g.temperature_errors_uk[i][a])}
                else{(c.temperature_total_uk[i],g.temperature_total_uk[i],c.temperature_total_errors_uk[i],g.temperature_total_errors_uk[i])};
            if !(ec.is_finite()&&eg.is_finite()){continue;}
            let x=z(tg-tc,ec,eg);tz=tz.max(x);count+=1;over+=(x>2.) as usize;rel=rel.max((tg-tc).abs()/tc);
        }
        let mean=|s:&Solution|->f64{(1..samples).map(|i|if a<3{s.temperature_uk[i][a]}else{s.temperature_total_uk[i]}).sum::<f64>()/(samples-1) as f64};
        println!("T {}: mean {:.2} μK (CPU) against {:.2} μK (GPU); largest relative |ΔT| {:.3}, largest z {tz:.2}, z > 2 in {over} of {count}",
            ["x","y","z","total"][a],mean(c),mean(g),rel);
    }
}
