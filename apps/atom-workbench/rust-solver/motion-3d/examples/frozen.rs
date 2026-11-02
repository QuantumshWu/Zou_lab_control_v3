//! The atom held still at a point: the engine's jump trajectories against the local master
//! equation of the front end (ground.rs `local_at`), integrated directly (RK4), from the same
//! initial state: the populations over time.
//! `cargo run --profile iterate -p atom-motion-3d --example frozen -- <scene> [x_um along the beams] [trajectories] [t_end_us]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::Settings;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Scales};
use atom_motion_3d::{simulate,Initial};
use atom_motion_core::Complex;
use atom_motion_core::linear::Dense;

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let name=arg(1,"TW2");let x:f64=arg(2,"0").parse().unwrap();let count:u64=arg(3,"1024").parse().unwrap();let t_end:f64=arg(4,"200").parse().unwrap();
    let request=scene::scene(&name);
    let model=Model::new(&request).unwrap();let g=&model.ground;
    let r=[0.,0.,x];
    // The exact local master equation at r, from |2,+2>.
    let e=g.eliminate(&g.hamiltonian_at(r)).unwrap();let local=g.local_at(&e,r);let h=local.effective_hamiltonian(&e);let ng=g.ground_states.len();
    let index=request.states.iter().position(|s|s=="ground-2-2").unwrap();let b22=g.ground_states.iter().position(|&i|i==index).unwrap();
    let mut rho=Dense::zeros(ng,ng);rho[(b22,b22)]=Complex::ONE;
    let derivative=|rho:&Dense|->Dense{
        let hr=h.product(rho);let mut out=hr.scaled(Complex::new(0.,-1.));out.add_scaled(Complex::new(0.,1.),&rho.product(&h.adjoint()));
        for l in &local.jumps{out.add_scaled(Complex::ONE,&l.product(rho).product(&l.adjoint()));}out
    };
    let dt=0.005;let interval=10.;let outputs=(t_end/interval).round() as usize+1;let mut exact=vec![g.populations(&local,&rho)];
    for _ in 1..outputs{for _ in 0..(interval/dt).round() as usize{
        let k1=derivative(&rho);let mut y=rho.clone();y.add_scaled(Complex::new(dt/2.,0.),&k1);let k2=derivative(&y);
        let mut y=rho.clone();y.add_scaled(Complex::new(dt/2.,0.),&k2);let k3=derivative(&y);let mut y=rho.clone();y.add_scaled(Complex::new(dt,0.),&k3);let k4=derivative(&y);
        for(k,w)in [(k1,1.),(k2,2.),(k3,2.),(k4,1.)]{rho.add_scaled(Complex::new(dt*w/6.,0.),&k);}
    }exact.push(g.populations(&local,&rho));}
    // The engine with the atom held at r: every trajectory starts there (no axis free).
    let settings=Settings{dt:0.2,free:[false;3],escape_intensity:0.,frequencies:[0.;3],truncation:f64::EPSILON,average:false};
    let initial=Initial::new(&model,&request).unwrap();
    let mut model_at=model.clone();let _=&mut model_at;
    let outcome=atom_motion_3d::simulate_from(&model,&settings,&initial,7,0,count,8,(interval/0.2).round() as usize,outputs,&[],r);
    let est=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:[1.;3]});
    let labels=["ground-1--1","ground-1-0","ground-1-1","ground-2--2","ground-2--1","ground-2-0","ground-2-1","ground-2-2"];
    let ids:Vec<usize>=labels.iter().map(|l|request.states.iter().position(|s|s==l).unwrap()).collect();
    let mut worst=0_f64;
    for i in (0..outputs).step_by((outputs/10).max(1)){
        let ex=&exact[i];let mc=&est[i];
        let z=ids.iter().map(|&k|((mc.populations[k]-ex[k])/mc.population_errors[k].max(1e-9)).abs()).fold(0.,f64::max);worst=worst.max(z);
        println!("t {:5.0} {}  exc {:.5}/{:.5}  worst {:.1} SE",i as f64*interval,ids.iter().map(|&k|format!("{:.4}/{:.4}",mc.populations[k],ex[k])).collect::<Vec<_>>().join(" "),
            g.excited_states.iter().map(|&k|mc.populations[k]).sum::<f64>(),g.excited_states.iter().map(|&k|ex[k]).sum::<f64>(),z);
    }
    let _=simulate;println!("largest deviation {worst:.1} SE");
}
