//! The atom held still at a point: the engine's jump trajectories against the eliminated master
//! equation of the front end (ground.rs `local_at`: every cluster's H_eff and jumps), integrated
//! directly (RK4 on the clusters' densities), from the same initial state: the populations over
//! time. Writes both, with the trajectories' standard errors, to <out>.json for the comparison with
//! the population core's internal solve of the same scene.
//! `cargo run --profile iterate -p atom-motion-3d --example frozen -- <scene> [trajectories] [t_end_us] [interval_us] [out.json] [x y z (um)] [dt_us]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::Settings;
use atom_motion_3d::linear::Dense;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Scales};
use atom_motion_3d::{simulate_from,Complex,Initial,STATES};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let name=arg(1,"PGC3D");let count:u64=arg(2,"2048").parse().unwrap();let t_end:f64=arg(3,"200").parse().unwrap();let interval:f64=arg(4,"5").parse().unwrap();
    let out=arg(5,&format!("frozen_{name}.json"));
    let r=[arg(6,"0").parse::<f64>().unwrap(),arg(7,"0").parse::<f64>().unwrap(),arg(8,"0").parse::<f64>().unwrap()];
    let request=scene::scene(&name);
    let clock=Instant::now();let model=Model::new(&request).unwrap();let g=&model.ground;
    println!("{name}: model {:.2} s; clusters {:?}; {} items ({} kept), cut {:.3e} rad/us, pole {:.3e} rad/us, scale {:.3e} rad/us; focused {:?}, table error {:.1e}, quartic {:.1e}",
        clock.elapsed().as_secs_f64(),g.clusters.iter().map(|c|c.states.len()).collect::<Vec<_>>(),g.items.len(),g.items.iter().filter(|i|!i.folded).count(),g.cut,g.pole,g.generator_scale,
        model.focused,model.table_error,model.quartic);
    let(far,near)=g.fold_estimates();println!("folds: far Σm² {far:.2e}, couplings Σm² {near:.2e}; paths {}",g.clusters.iter().map(|c|c.paths.len()).sum::<usize>());
    // The eliminated master equation at r, from the initial ground populations (each in its cluster).
    let e=g.eliminate(&model.hamiltonian_at_point(r)).unwrap();
    let static_generator=g.clusters.iter().all(|c|c.groups.iter().all(|x|x.entries.iter().all(|e|e.2==0.))&&c.paths.iter().all(|p|p.residual==0.)
        &&c.classes.iter().all(|k|k.photons.iter().all(|p|p.1==0.)));
    let locals=|t:f64|->Vec<atom_motion_3d::ground::Local>{(0..g.clusters.len()).map(|k|g.local_at(&e,k,r,t).unwrap()).collect()};
    let mut rho:Vec<Dense>=g.clusters.iter().map(|c|{let n=c.states.len();Dense::from_fn(n,n,|i,j|if i==j{let s=g.ground_states[c.states[i]];request.initial_density[s*STATES+s]}else{Complex::ZERO})}).collect();
    let derivative=|rho:&[Dense],l:&[atom_motion_3d::ground::Local]|->Vec<Dense>{
        let mut out:Vec<Dense>=rho.iter().zip(l).map(|(p,x)|{let h=&x.effective;let mut d=h.product(p).scaled(Complex::new(0.,-1.));d.add_scaled(Complex::new(0.,1.),&p.product(&h.adjoint()));d}).collect();
        for(k,x)in l.iter().enumerate(){for(class,jumps)in g.clusters[k].classes.iter().zip(&x.jumps){for j in jumps{out[class.target].add_scaled(Complex::ONE,&j.product(&rho[k]).product(&j.adjoint()));}}}
        out
    };
    let readout=|rho:&[Dense],l:&[atom_motion_3d::ground::Local]|->Vec<f64>{let mut p=vec![0.;STATES];for(k,x)in l.iter().enumerate(){for(a,b)in p.iter_mut().zip(g.populations(k,x,&rho[k])){*a+=b;}}p};
    let dt=interval/((interval/0.005).round().max(1.));let outputs=(t_end/interval).round() as usize+1;let steps=(interval/dt).round() as usize;
    let fixed=locals(0.);let at=|t:f64|if static_generator{fixed.clone()}else{locals(t)};
    let mut exact=vec![readout(&rho,&fixed)];let clock=Instant::now();let mut t=0.;
    for _ in 1..outputs{for _ in 0..steps{
        let(l0,lh,l1)=(at(t),at(t+0.5*dt),at(t+dt));
        let add=|a:&[Dense],b:&[Dense],s:f64|->Vec<Dense>{a.iter().zip(b).map(|(x,y)|{let mut z=x.clone();z.add_scaled(Complex::new(s,0.),y);z}).collect()};
        let k1=derivative(&rho,&l0);let k2=derivative(&add(&rho,&k1,dt/2.),&lh);let k3=derivative(&add(&rho,&k2,dt/2.),&lh);let k4=derivative(&add(&rho,&k3,dt),&l1);
        for(i,x)in rho.iter_mut().enumerate(){for(k,w)in [(&k1,1.),(&k2,2.),(&k3,2.),(&k4,1.)]{x.add_scaled(Complex::new(dt*w/6.,0.),&k[i]);}}
        t+=dt;
    }exact.push(readout(&rho,&at(t)));}
    println!("eliminated master equation: {:.1} s (static {static_generator})",clock.elapsed().as_secs_f64());
    // The engine with the atom held at r: every trajectory starts there (no axis free, no loss).
    let initial=Initial::new(&model,&request).unwrap();let trap=initial.traps[0];
    let engine_steps=(interval/arg(9,"0.2").parse::<f64>().unwrap()).round().max(1.) as usize;
    let mut settings=Settings::new(interval/engine_steps as f64,&trap,f64::EPSILON,false);settings.free=[false;3];settings.loss=false;
    let clock=Instant::now();
    let outcome=simulate_from(&model,&settings,&initial,7,0,count,8,engine_steps,outputs,&[],r);
    let est=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:[1.;3]});
    println!("engine: {count} trajectories, {:.1} s, {} jumps",clock.elapsed().as_secs_f64(),outcome.jumps);
    let level=|p:&[f64],f:&str|request.states.iter().enumerate().filter(|(_,s)|s.starts_with(f)).map(|(i,_)|p[i]).sum::<f64>();
    let mut worst=0_f64;
    for i in (0..outputs).step_by((outputs/10).max(1)).chain([outputs-1]){
        let(ex,mc)=(&exact[i],&est[i]);
        let z=(0..STATES).map(|k|((mc.populations[k]-ex[k])/mc.population_errors[k].max(1e-9)).abs()).fold(0.,f64::max);worst=worst.max(z);
        let d=(0..STATES).map(|k|(mc.populations[k]-ex[k]).abs()).fold(0.,f64::max);
        println!("t {:6.0}  F1 {:.4}/{:.4}  F2 {:.4}/{:.4}  exc {:.5}/{:.5}  max |Δp| {d:.4} ({z:.1} SE)",i as f64*interval,
            level(&mc.populations,"ground-1-"),level(ex,"ground-1-"),level(&mc.populations,"ground-2-"),level(ex,"ground-2-"),
            1.-level(&mc.populations,"ground-"),1.-level(ex,"ground-"));
    }
    println!("largest deviation {worst:.1} SE");
    let json=format!("{{\"scene\":\"{name}\",\"r\":{r:?},\"states\":{:?},\"timesUs\":{:?},\"eliminated\":{:?},\"engine\":{:?},\"engineErrors\":{:?},\"trajectories\":{count}}}",
        request.states,(0..outputs).map(|i|i as f64*interval).collect::<Vec<_>>(),exact,est.iter().map(|e|e.populations.clone()).collect::<Vec<_>>(),est.iter().map(|e|e.population_errors.clone()).collect::<Vec<_>>());
    std::fs::write(&out,json).unwrap();println!("written {out}");
}
