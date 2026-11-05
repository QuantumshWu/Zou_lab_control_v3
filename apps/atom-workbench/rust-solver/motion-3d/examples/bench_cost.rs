//! The step's parts against their counted work on one thread, for the step choice's cost model
//! (solve.rs `interval_work`): states along a scene's trajectories at dt; per state an
//! advance (`Engine::coast` of one step), a half step's propagation, a whole jump (`Engine::jump`:
//! the photons of every class, the picked class's absorbed light and the rest), timed; and their work (`Model::work`, the series' sub-steps × terms × n²). Load only adds
//! time: every unit is timed once per round, the rounds interleaved, and each unit's least time
//! over the rounds kept -- pin the process to one core.
//! `cargo run --profile iterate -p atom-motion-3d --example bench_cost -- <scene> <dt_us> <truncation> [rounds]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::{column_norm,propagate,series,Engine,Settings};
use atom_motion_3d::model::Model;
use atom_motion_3d::{initial_trajectory,Initial,Stratum};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();let name=args[1].clone();let dt:f64=args[2].parse().unwrap();let truncation:f64=args[3].parse().unwrap();
    let rounds:usize=args.get(4).map_or(40,|s|s.parse().unwrap());
    let request=scene::scene(&name);let model=Model::new(&request).unwrap();let initial=Initial::new(&model,&request).unwrap();
    let mut settings=Settings::new(dt,&initial.traps[0],truncation,false);settings.loss=false;
    let engine=Engine::new(&model,settings.clone());
    // States along 16 trajectories, every 25 steps over 2000.
    let mut states=vec![];
    for index in 0..16u64{let mut t=initial_trajectory(&model,&settings,&initial,9,index,Stratum::single(index));
        for k in 0..2000{engine.step(&mut t);if k%25==0{states.push(t.clone());}}}
    let locals:Vec<_>=states.iter().map(|t|model.local(t.cluster,&model.point(t.r,t.kappa),t.time)).collect();
    // The units, a round each over every state; the least time of each over the rounds (us per call).
    let units=["advance","half step","jump"];let mut best=[f64::INFINITY;3];let mut sink=0.;
    for _ in 0..rounds{for(u,b)in best.iter_mut().enumerate(){
        let clock=Instant::now();
        for(i,t)in states.iter().enumerate(){sink+=match u{
            0=>engine.coast(t,dt,1).kappa[0],
            1=>{let n=model.dimension(t.cluster);let mut psi=t.psi;propagate(&locals[i].hamiltonian,n,0.5*dt,&mut psi,truncation);psi[0].re}
            _=>{let mut x=t.clone();let(r,time)=(x.r,x.time);engine.jump(&mut x,r,time,false);x.kappa[0]}
        };}
        *b=b.min(clock.elapsed().as_secs_f64()/states.len() as f64*1e6);
    }}
    let mean=|f:&dyn Fn(usize)->f64|->f64{(0..states.len()).map(f).sum::<f64>()/states.len() as f64};
    let w=|i:usize|model.work(states[i].cluster);
    let(macs,pairs,photons,absorption)=(mean(&|i|w(i).generator+w(i).force),mean(&|i|w(i).pairs as f64),mean(&|i|w(i).photons),mean(&|i|w(i).absorption));
    let series_at=|i:usize|{let t=&states[i];let n=model.dimension(t.cluster);let(p,k)=series(column_norm(&locals[i].hamiltonian,n),0.5*dt,truncation);((p*k) as f64,(p*k*n*n) as f64)};
    let(terms,taylor)=(mean(&|i|series_at(i).0),mean(&|i|series_at(i).1));
    let n2=mean(&|i|{let n=model.dimension(states[i].cluster);(n*n) as f64});
    println!("{name} dt {dt} truncation {truncation:.1e}, {} states, {rounds} rounds [{sink:.1e}]",states.len());
    println!("  least time (us): {}",units.iter().zip(best).map(|(u,b)|format!("{u} {b:.4}")).collect::<Vec<_>>().join(", "));
    println!("  work: generator+force {macs:.1}, pairs {pairs:.2}, half step terms {terms:.2} and multiply-adds {taylor:.1}, photons {photons:.1}, absorption {absorption:.1}, n² {n2:.2}");
}
