//! The Latin hypercube of initial states (lib.rs `Stratum`) against independent draws: the standard
//! errors of T per axis at a few times for one number of trajectories, groups of 1 (independent)
//! and larger, several seeds.
//! `cargo run --profile iterate -p atom-motion-3d --example strata -- <scene> [trajectories] [t_end_us] [dt]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::Settings;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Scales};
use atom_motion_3d::{simulate,Initial};

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let name=arg(1,"TW2_3D");let count:u64=arg(2,"1024").parse().unwrap();let t_end:f64=arg(3,"300").parse().unwrap();let dt:f64=arg(4,"0.227").parse().unwrap();
    let request=scene::scene(&name);let model=Model::new(&request).unwrap();let initial=Initial::new(&model,&request).unwrap();
    let(block,state,_)=*initial.states.iter().max_by(|a,b|a.2.total_cmp(&b.2)).unwrap();let frequencies=model.trap_frequencies(block,state);
    let settings=Settings{dt,free:[true;3],escape_intensity:1./20.,frequencies,truncation:f64::EPSILON};
    let interval=20.;let steps=(interval/dt).round() as usize;let outputs=(t_end/interval).round() as usize+1;
    let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies};
    for group in [1,8,32]{
        // The standard error per axis at each output, averaged over seeds as a variance.
        let seeds=4;let mut variance=vec![[0.;3];outputs];
        for seed in 0..seeds{
            let outcome=simulate(&model,&settings,&initial,100+seed,0,count,group,steps,outputs,&[]);
            for(v,e)in variance.iter_mut().zip(summarize(&outcome.groups,&scales)){for a in 0..3{v[a]+=(e.temperature_error[a]/e.temperature[a]).powi(2)/seeds as f64;}}
        }
        let at=|i:usize|format!("t {:4.0}: {}",i as f64*interval,(0..3).map(|a|format!("{:.2}%",100.*variance[i][a].sqrt())).collect::<Vec<_>>().join(" "));
        println!("{name} groups of {group:2}: relative SE of T (x y z) {} | {} | {} | {}",at(0),at(outputs/4),at(outputs/2),at(outputs-1));
    }
}
