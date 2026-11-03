//! The step-doubling error of the deterministic step on states sampled along trajectories: for each
//! dt, the relative error of the step's non-harmonic impulse, Σ|Δκ|/Σ|kick|, the population error per
//! step, the relative error of the norm's decay (the jumps' timing), of the friction and of the
//! heating (the momentum's covariance increment).
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
        let(mut dk,mut kick,mut dp,mut dl,mut l,mut de,mut e)=(0.,0.,0_f64,0.,0.,0.,0.);
        let(mut dv,mut v,mut sum_dp)=(0.,0.,0.);
        for t in &states{let d=engine.doubling(t,dt);dk+=d.momentum[0];kick+=d.momentum[1];dp=dp.max(d.internal[0]);sum_dp+=d.internal[0];dl+=d.internal[1];l+=d.internal[2];
            de+=d.energy[0];e+=d.energy[1];dv+=d.heating[0];v+=d.heating[1];}
        println!("dt {dt:4}: impulse {:.2e} (rel), population per step mean {:.2e} max {:.2e}, decay {:.2e} (rel), friction {:.2e} (rel; mean energy change {:.2e} rad/us per step), heating {:.2e} (rel)",
            dk/kick,sum_dp/states.len() as f64,dp,dl/l,(de/e).abs(),e/states.len() as f64,dv/v);
    }
}
