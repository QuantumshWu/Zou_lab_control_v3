//! The step-doubling error of the deterministic step on states sampled along trajectories: for each
//! dt, the relative error of the step's non-harmonic impulse, Σ|Δκ|/Σ|kick|, the population error per
//! step and the relative error of the norm's decay (the jumps' timing).
//! `cargo run --profile iterate -p atom-motion-3d --example local_error -- <scene> [trajectories] [t_end_us]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::{Engine,Settings};
use atom_motion_3d::model::Model;
use atom_motion_3d::{initial_trajectory,Initial,Stratum};

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let name=arg(1,"TW2");let count:u64=arg(2,"64").parse().unwrap();let t_end:f64=arg(3,"1000").parse().unwrap();
    let request=scene::scene(&name);
    let model=Model::new(&request).unwrap();let initial=Initial::new(&model,&request).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();
    let settings=Settings::new(0.1,&initial.traps[reference],f64::EPSILON,false);let engine=Engine{model:&model,settings:settings.clone()};
    // States along trajectories (dt 0.1), every 10 us.
    let mut states=vec![];
    for index in 0..count{let mut t=initial_trajectory(&model,&settings,&initial,3,index,Stratum::single(index));
        for _ in 0..(t_end/10.) as usize{for _ in 0..100{if t.alive{engine.step(&mut t);}}if t.alive{states.push(t.clone());}}}
    println!("{name}: {} sampled states",states.len());
    for dt in [0.05,0.1,0.2,0.3,0.5,1.0]{
        let(mut dk,mut kick,mut dp,mut dl,mut l)=(0.,0.,0_f64,0.,0.);
        for t in &states{let(a,b)=engine.doubling(t,dt);dk+=a[0];kick+=a[1];dp=dp.max(b[0]);dl+=b[1];l+=b[2];}
        let mean_dp:f64=states.iter().map(|t|engine.doubling(t,dt).1[0]).sum::<f64>()/states.len() as f64;
        println!("dt {dt:4}: impulse {:.2e} (rel), population per step mean {:.2e} max {:.2e}, decay {:.2e} (rel)",dk/kick,mean_dp,dp,dl/l);
    }
}
