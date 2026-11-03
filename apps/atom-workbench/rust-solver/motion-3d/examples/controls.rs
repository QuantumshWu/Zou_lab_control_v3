//! The control variates (control.rs) against their trajectories, as a solve runs them: the held
//! twin's populations against the master equation at its point, the harmonic control against its
//! exact mean (z per output, jackknife over the groups); and the estimates with the controls against
//! those without on the same trajectories -- their difference against its standard error, and the
//! standard errors' ratio.
//! `cargo run --profile release -p atom-motion-3d --example controls -- <scene> <duration_us> <samples> <trajectories> <dt_us>`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::control::{Harmonic,Twin};
use atom_motion_3d::engine::Settings;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Means,Scales};
use atom_motion_3d::{sample_states,simulate,Initial,Variates,STATES};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();
    let name=args[1].clone();let duration:f64=args[2].parse().unwrap();let samples:usize=args[3].parse().unwrap();
    let count:u64=args[4].parse().unwrap();let dt:f64=args[5].parse().unwrap();
    let request=scene::scene(&name);let model=Model::new(&request).unwrap();let initial=Initial::new(&model,&request).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();let trap=initial.traps[reference];
    let interval=duration/(samples-1) as f64;let steps=((interval/dt).round() as usize).max(1);let outputs=samples;
    let settings=Settings::new(interval/steps as f64,&trap,f64::EPSILON,true);let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab};
    let threads=rayon::current_num_threads() as u64;let seed=0x3D_5EED;
    // The twin where the trap beam's intensity is the pilot's mean, as a solve places it.
    let(states,_)=sample_states(&model,&settings,&initial,seed,2*threads,steps,outputs);
    let intensity=states.iter().map(|t|model.geometry[model.trap_beam].intensity(t.r)).sum::<f64>()/states.len().max(1) as f64;
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);let twin=Twin::new(&model,intensity);
    let clock=Instant::now();let exact=twin.as_ref().map(|t|t.means(&model,&request.initial_density,interval,outputs));
    if let Some(e)=&exact{println!("master equation: largest |Σ populations − 1| over the outputs {:.2e}",e.iter().map(|p|(p.iter().sum::<f64>()-1.).abs()).fold(0.,f64::max));}
    println!("{name}: dt {:.4} us, {count} trajectories; twin {} (intensity {intensity:.4}), master equation {:.3} s",settings.dt,
        twin.as_ref().map_or("none (beats)".into(),|t|format!("at {:?}",t.held.point.r)),clock.elapsed().as_secs_f64());
    let clock=Instant::now();
    let out=simulate(&model,&settings,&initial,seed,0,count,(count/(2*threads)).max(1),steps,outputs,&[],Some(&Variates{harmonic:&harmonic,twin:twin.as_ref()}));
    println!("run {:.1} s; thread-seconds: trajectories {:.1}, twins {:.1} ({:.2} of the trajectories')",clock.elapsed().as_secs_f64(),out.seconds[0],out.seconds[1],out.seconds[1]/out.seconds[0]);
    // z of a per-trajectory mean against its exact value, by the jackknife over the groups.
    let z=|i:usize,value:&dyn Fn(&atom_motion_3d::stats::Sums)->f64,exact:f64|->(f64,f64){
        let total=|skip:Option<usize>|{let(mut v,mut n)=(0.,0.);for(g,x)in out.groups.iter().enumerate(){if Some(g)!=skip{v+=value(&x.sums[i]);n+=x.sums[i].trajectories;}}v/n};
        let m=total(None);let k=out.groups.len();let leave:Vec<f64>=(0..k).map(|g|total(Some(g))).collect();let lm=leave.iter().sum::<f64>()/k as f64;
        let se=((k-1) as f64/k as f64*leave.iter().map(|x|(x-lm).powi(2)).sum::<f64>()).sqrt();(m-exact,if se>0.{(m-exact)/se}else{0.})};
    let(mut worst,mut worst_harmonic,mut largest,mut worst_difference)=((0_f64,0.,0.),(0_f64,0.,0.),0_f64,(0_f64,0.,0.));
    for i in 1..outputs{
        // z where the twins' spread is resolved (an absolute SE above 1e-4: before the first jumps the
        // twins agree to rounding); the largest difference anywhere, with its z.
        if let Some(e)=&exact{for s in 0..STATES{let(d,zz)=z(i,&|x|x.twin[s],e[i][s]);let se=if zz!=0.{d/zz}else{0.};
            if d.abs()>largest{largest=d.abs();worst_difference=(d,zz,i as f64*interval);}if se.abs()>1e-4&&zz.abs()>worst.0.abs(){worst=(zz,d,i as f64*interval);}}}
        for a in 0..3{let(d,zz)=z(i,&|x|x.harmonic[a][0],harmonic.mean[i][a]);if zz.abs()>worst_harmonic.0.abs(){worst_harmonic=(zz,d/harmonic.mean[i][a],i as f64*interval);}}
    }
    if exact.is_some(){println!("twin against the master equation: largest |z| {:.2} ({:+.5} at {:.1} us); largest |difference| {:+.5} ({:.2} SE at {:.1} us)",worst.0,worst.1,worst.2,worst_difference.0,worst_difference.1,worst_difference.2);}
    println!("harmonic control against its mean: largest |z| {:.2} ({:+.2e} relative at {:.1} us)",worst_harmonic.0,worst_harmonic.1,worst_harmonic.2);
    // A systematic bias: per state, the twins' mean less the master equation's averaged over the
    // second half of the outputs, its standard error by the jackknife over the groups.
    if let Some(e)=&exact{
        let late:Vec<usize>=(outputs/2..outputs).collect();
        let bias=|s:usize,skip:Option<usize>|->f64{late.iter().map(|&i|{let(mut v,mut n)=(0.,0.);for(g,x)in out.groups.iter().enumerate(){if Some(g)!=skip{v+=x.sums[i].twin[s];n+=x.sums[i].trajectories;}}v/n-e[i][s]}).sum::<f64>()/late.len() as f64};
        let k=out.groups.len();let mut worst=(0_f64,0_f64,0usize);
        for s in 0..STATES{let b=bias(s,None);let leave:Vec<f64>=(0..k).map(|g|bias(s,Some(g))).collect();let m=leave.iter().sum::<f64>()/k as f64;
            let se=((k-1) as f64/k as f64*leave.iter().map(|x|(x-m).powi(2)).sum::<f64>()).sqrt();if se>0.&&(b/se).abs()>worst.0.abs(){worst=(b/se,b,s);}}
        println!("twin's late-time mean against the master equation: largest |z| {:.2} ({:+.5}, state {})",worst.0,worst.1,request.states[worst.2]);
    }
    // The estimates with the controls and without, on the same trajectories.
    let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:harmonic.mean[i],twin:exact.as_ref().map(|e|e[i])}).collect();
    let(with,without)=(summarize(&out.groups,&scales,&means),summarize(&out.groups,&scales,&[]));
    let pick:Vec<usize>={let mut v:Vec<usize>=(1..=12).map(|j|((outputs-1) as f64*(j as f64/12.).powi(2)).round() as usize).filter(|&i|i>0).collect();v.dedup();v};
    println!("{:>9} | {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7} | largest |Δp|/SE, |ΔT|/SE","t_us","SEp","SEp cv","ratio","SET%","SET% cv","ratio");
    let(mut dp,mut dt_)=(0_f64,0_f64);
    for i in 1..outputs{
        let(a,b)=(&with[i],&without[i]);
        let sp=b.population_errors.iter().fold(0_f64,|m,&x|m.max(x));let spc=a.population_errors.iter().fold(0_f64,|m,&x|m.max(x));
        let st=(0..3).map(|x|b.temperature_error[x]/b.temperature[x]).fold(0_f64,f64::max);let stc=(0..3).map(|x|a.temperature_error[x]/a.temperature[x]).fold(0_f64,f64::max);
        let p=(0..STATES).map(|s|(a.populations[s]-b.populations[s]).abs()/b.population_errors[s].max(1e-4)).fold(0_f64,f64::max);
        let t=(0..3).map(|x|(a.temperature[x]-b.temperature[x]).abs()/b.temperature_error[x].max(1e-12)).fold(0_f64,f64::max);dp=dp.max(p);dt_=dt_.max(t);
        if pick.contains(&i){println!("{:9.1} | {sp:8.5} {spc:8.5} {:7.3} | {:8.3} {:8.3} {:7.3} | {p:.2}, {t:.2}",i as f64*interval,spc/sp,100.*st,100.*stc,stc/st);}
    }
    println!("over every output: the controlled estimates within {dp:.2} (populations) and {dt_:.2} (T) plain standard errors of the plain ones");
}
