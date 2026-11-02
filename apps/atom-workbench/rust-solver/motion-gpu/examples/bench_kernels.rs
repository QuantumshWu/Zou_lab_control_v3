//! Times the kinetic and local passes on synthetic data: `cargo run --profile iterate --example bench_kernels`.
use atom_motion_gpu::driver::{Buffer,Event};
use atom_motion_gpu::gpu::Gpu;

fn main(){
    let gpu=Gpu::new().unwrap();
    println!("{} ({} SMs, L2 {} MiB)",gpu.context.info.name,gpu.context.info.multiprocessors,gpu.context.info.l2_cache>>20);
    for(n,trajectories)in [(64usize,1000usize),(64,100),(32,1000),(32,400),(16,1000)]{
        let points=n*n;let planes=trajectories*5;let bytes=planes*points*8;
        let state=Buffer::new(bytes).unwrap();gpu.stream.zero(&state).unwrap();
        let components=Buffer::new(trajectories*4).unwrap();gpu.stream.upload(&components,&vec![5i32;trajectories]).unwrap();
        let scale=Buffer::new(trajectories*4).unwrap();gpu.stream.upload(&scale,&vec![1f32;trajectories]).unwrap();
        let table=Buffer::new(n*8).unwrap();gpu.stream.upload(&table,&vec![[1f32/n as f32,0.];n]).unwrap();
        let v5=Buffer::new(25*points*8).unwrap();gpu.stream.zero(&v5).unwrap();let v3=Buffer::new(9*points*8).unwrap();gpu.stream.zero(&v3).unwrap();
        let tiles=points.div_ceil(256);let partial=Buffer::new(trajectories*tiles*4).unwrap();
        let kinetic=gpu.function(&format!("kinetic_{n}x{n}")).unwrap();let shared=n*(n+1)*8;kinetic.allow_shared(shared).unwrap();
        let local=gpu.function("local_step").unwrap();
        let(start,end)=(Event::new().unwrap(),Event::new().unwrap());
        let reps=50;
        let time=|gpu:&Gpu,f:&dyn Fn(&Gpu)|->f64{f(gpu);gpu.stream.record(&start).unwrap();for _ in 0..reps{f(gpu);}gpu.stream.record(&end).unwrap();end.synchronize().unwrap();end.since(&start).unwrap()/reps as f64};
        let k=time(&gpu,&|g:&Gpu|g.stream.launch(&kinetic,(planes as u32,1,1),256,shared,&[&state.pointer,&components.pointer,&scale.pointer,&table.pointer,&table.pointer]).unwrap());
        let l=time(&gpu,&|g:&Gpu|g.stream.launch(&local,(tiles as u32,trajectories.div_ceil(8) as u32,1),256,0,&[&state.pointer,&(points as i32),&(trajectories as i32),&components.pointer,&v5.pointer,&v3.pointer,&partial.pointer]).unwrap());
        let(rk,rl)=(kinetic.resources().unwrap(),local.resources().unwrap());
        println!("{n}x{n}, {trajectories} trajectories ({:.0} MB): kinetic {:.3} ms ({:.0} GB/s, regs {:?}), local {:.3} ms ({:.0} GB/s, regs {:?}); per trajectory-step {:.3} us",
            bytes as f64/1e6,k,2.*bytes as f64/k/1e6,rk,l,2.*bytes as f64/l/1e6,rl,(k+l)*1e3/trajectories as f64);
    }
}
