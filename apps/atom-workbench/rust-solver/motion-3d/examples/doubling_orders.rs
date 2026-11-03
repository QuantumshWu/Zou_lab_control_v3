//! The step-doubling population error of the fold's second and fourth order on the same states (sampled
//! along second-order trajectories from a seed): per dt the mean and largest population error per step of
//! either, the measure the solve's step check takes -- a beat the fourth order keeps but the step cannot
//! follow shows as a fourth-order error far above the second's.
//! `cargo run --profile iterate -p atom-motion-3d --example doubling_orders -- <scene> [trajectories] [t_end_us] [seed] [dt_us...]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::{Engine,Settings};
use atom_motion_3d::model::Model;
use atom_motion_3d::{initial_trajectory,Complex,Initial,Stratum};

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let name=arg(1,"THREE");let count:u64=arg(2,"32").parse().unwrap();let t_end:f64=arg(3,"500").parse().unwrap();let seed:u64=arg(4,"1").parse().unwrap();
    let dts:Vec<f64>=if args.len()>5{args[5..].iter().map(|x|x.parse().unwrap()).collect()}else{vec![0.02,0.05,0.1,0.2,0.4]};
    let request=scene::scene(&name);
    let second=Model::new(&request).unwrap();let fourth=Model::with_order(&request,f64::INFINITY,true).unwrap();
    let initial=Initial::new(&second,&request).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();
    let settings=Settings::new(0.05,&initial.traps[reference],f64::EPSILON,true);let engine=Engine::new(&second,settings.clone());
    let mut states=vec![];
    for index in 0..count{let mut t=initial_trajectory(&second,&settings,&initial,seed,index,Stratum::single(index));
        for _ in 0..(t_end/10.) as usize{for _ in 0..200{if t.alive{engine.step(&mut t);}}if t.alive{states.push(t.clone());}}}
    let tables:usize=(0..fourth.clusters()).filter(|&k|fourth.slow_part(k,&fourth.point([0.;3],[0.;3]),0.).is_some()).count();
    println!("{name} seed {seed}: {} states; fourth-order clusters {tables}, H^(4) entries left out Σm² {:.2e}; beating {:?}",states.len(),fourth.fourth_dropped,
        (0..fourth.clusters()).map(|k|fourth.beating(k)).collect::<Vec<_>>());
    // The beats' components (a beating cluster's generator) against the local generator at the states' times.
    let mut worst=0_f64;
    for t in &states{let p=fourth.point(t.r,t.kappa);let n=fourth.dimension(t.cluster);
        let(g,l)=(fourth.generator(t.cluster,&p,t.time),fourth.local(t.cluster,&p,t.time));
        let scale=l.hamiltonian[..n*n].iter().map(|z|z.abs()).fold(0.,f64::max);
        for a in 0..n{for b in 0..n{let shift=Complex::new(if a==b{g.scalar-l.scalar}else{0.},0.);worst=worst.max((g.hamiltonian[a*n+b]+shift-l.hamiltonian[a*n+b]).abs()/scale);}}}
    println!("fourth order: components against the local generator, largest relative difference {worst:.1e}");
    // The generator against its jumps at the states (model.rs `Model::decay_balance`), either order.
    for(label,model)in [("second",&second),("fourth",&fourth)]{
        let(mut gain,mut rate)=(0_f64,0_f64);
        for t in &states{let(low,r,_)=model.decay_balance(t.cluster,&model.point(t.r,t.kappa),t.time);gain=gain.max(-low);rate=rate.max(r);}
        println!("{label} order: largest −λ(i(H − H†) − ΣL†L) {gain:.2e} rad/us, largest jump rate {rate:.2e}: relative {:.1e}",gain/rate.max(f64::MIN_POSITIVE));
    }
    for dt in dts{
        let mut line=format!("dt {dt:6.3}:");
        for(label,model)in [("second",&second),("fourth",&fourth)]{
            let e=Engine::new(model,Settings::new(dt,&initial.traps[reference],f64::EPSILON,true));
            let errors:Vec<f64>=states.iter().map(|t|e.doubling(t,dt).1[0]).collect();
            let mean=errors.iter().sum::<f64>()/errors.len().max(1) as f64;let most=errors.iter().copied().fold(0.,f64::max);
            line+=&format!("  {label} mean {mean:.2e} max {most:.2e}");
        }
        println!("{line}");
    }
}
