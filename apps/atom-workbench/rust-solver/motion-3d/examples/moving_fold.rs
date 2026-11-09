//! The moving atom with the drives folded at second and at fourth order against the same scene with
//! the drives kept explicit (the rule's candidate at `allowance`): the same seeds, initial states and
//! step, every trajectory moving in the trap. Per output the F = 1 and F = 2 populations, the most
//! populated ground state's and T_total, each with its standard error, and the folds' differences
//! from the explicit model; the folded excited population 2η² along the way. A negative allowance
//! leaves the explicit model out (the second order against the fourth alone).
//! `cargo run --profile iterate -p atom-motion-3d --example moving_fold -- <scene> [trajectories] [duration_us] [outputs] [dt_us] [allowance]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::Settings;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Estimate,Scales};
use atom_motion_3d::{simulate,Initial};
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let name=arg(1,"PGC3D");let count:u64=arg(2,"2048").parse().unwrap();let duration:f64=arg(3,"2000").parse().unwrap();
    let outputs:usize=arg(4,"21").parse().unwrap();let dt:f64=arg(5,"0.2304").parse().unwrap();let allowance:f64=arg(6,"0.0095").parse().unwrap();
    let mut request=scene::scene(&name);request.times_us=(0..outputs).map(|i|duration*i as f64/(outputs-1) as f64).collect();
    let interval=duration/(outputs-1) as f64;let steps=(interval/dt).round().max(1.) as usize;
    let threads=rayon::current_num_threads() as u64;
    let mut runs:Vec<(&str,Model)>=vec![("second",Model::with_order(&request,f64::INFINITY,false).unwrap()),("fourth",Model::with_order(&request,f64::INFINITY,true).unwrap())];
    if allowance>=0.{runs.push(("explicit",Model::with_order(&request,allowance,false).unwrap()));}
    let mut results:Vec<(String,Vec<Estimate>)>=vec![];
    for(label,model)in &runs{
        let initial=Initial::new(model,&request).unwrap();let trap=initial.traps[0];
        let settings=Settings::new(interval/steps as f64,&trap,1e-6,true);
        let sizes:Vec<String>=(0..model.clusters()).map(|k|format!("{}({})",model.dimension(k),model.ground.clusters[k].excited.len())).collect();
        let clock=Instant::now();let cpu=scene::cpu_seconds();
        let outcome=simulate(model,&settings,&initial,0x3D_5EED,0,count,(count/(2*threads)).max(1),steps,outputs,&[],None);
        let estimates=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab},&[]);
        println!("{label}: clusters {}, {count} trajectories, {:.1} s wall, {:.0} CPU-s, {} jumps",sizes.join("+"),clock.elapsed().as_secs_f64(),scene::cpu_seconds()-cpu,outcome.jumps);
        results.push((label.to_string(),estimates));
    }
    let pick=|label:&str|request.states.iter().position(|x|x==label).unwrap();
    let group=|e:&Estimate,prefix:&str|->(f64,f64){let idx:Vec<usize>=request.states.iter().enumerate().filter(|(_,s)|s.starts_with(prefix)).map(|(i,_)|i).collect();
        (idx.iter().map(|&i|e.populations[i]).sum(),idx.iter().map(|&i|e.population_errors[i].error).fold(0.,f64::max))};
    let m22=pick("ground-2-2");
    // Per output: each model's F = 1 total, |m=2⟩ and T_total with standard errors; the second order
    // less the fourth, against their combined standard error; the largest |Δp| over the 32 states.
    let k=results.len();
    println!("t (us) | F1 per model (SE) | m22 per model (SE) | T_total uK per model (SE) | 2η² | second − fourth: F1, m22, max|Δp| over states (combined SE)");
    for i in 1..outputs{
        let e=|j:usize|&results[j].1[i];
        let f1:Vec<(f64,f64)>=(0..k).map(|j|group(e(j),"ground-1-")).collect();
        let fmt=|v:Vec<(f64,f64)>,d:usize|v.iter().map(|x|format!("{:.prec$}±{:.prec$}",x.0,x.1,prec=d)).collect::<Vec<_>>().join(" ");
        let dmax=(0..e(0).populations.len()).map(|s|((e(0).populations[s]-e(1).populations[s]).abs(),(e(0).population_errors[s].error.powi(2)+e(1).population_errors[s].error.powi(2)).sqrt()))
            .fold((0.,0.),|a:(f64,f64),b|if b.0>a.0{b}else{a});
        println!("{:6.0} | {} | {} | {} | {:.4} | {:+.4} {:+.4} {:.4} (SE {:.4} {:.4} {:.4})",i as f64*interval,fmt(f1.clone(),4),
            fmt((0..k).map(|j|(e(j).populations[m22],e(j).population_errors[m22].error)).collect(),4),fmt((0..k).map(|j|(e(j).temperature_total,e(j).temperature_total_error.error)).collect(),2),
            2.*e(0).folded,f1[0].0-f1[1].0,e(0).populations[m22]-e(1).populations[m22],dmax.0,(f1[0].1.powi(2)+f1[1].1.powi(2)).sqrt(),
            (e(0).population_errors[m22].error.powi(2)+e(1).population_errors[m22].error.powi(2)).sqrt(),dmax.1);
    }
}
