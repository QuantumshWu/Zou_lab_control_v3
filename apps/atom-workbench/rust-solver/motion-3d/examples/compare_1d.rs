//! The semiclassical engine on the line through the focus (the beams' axis free, the others held
//! at 0) against the 1D GMW references LONGREF_TW1/TW2: T, n̄, the ground populations, the
//! excited population and the loss.
//! `cargo run --profile iterate -p atom-motion-3d --example compare_1d -- <TW1|TW2> <label> [trajectories] [t_end_us] [dt] [seed] [flow|drift] [truncation] [group] [average|instant]`
//! (the exponentials' truncation bound, double precision by default, the Latin hypercube's group,
//! eight by default -- a solve's are its share of the step budget and a batch over 2 threads -- and
//! T over one trap period, as a solve takes it, or at the output time.)
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::Settings;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Scales};
use atom_motion_3d::{simulate,Initial};
use serde_json::Value;

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let(name,variant)=(arg(1,"TW2"),arg(2,"exact"));
    let(count,t_end,dt,seed):(u64,f64,f64,u64)=(arg(3,"512").parse().unwrap(),arg(4,"1000").parse().unwrap(),arg(5,"0.2").parse().unwrap(),arg(6,"1").parse().unwrap());
    let request=scene::scene(&name);
    let reference:Value=scene::reference(&name);
    let rows=reference["rows"].as_array().unwrap();
    let start=std::time::Instant::now();
    let model=Model::new(&request).unwrap();
    println!("model {:.2} s: carriers {:?} ranks {:?}, focused {:?}, table error {:.1e}",start.elapsed().as_secs_f64(),
        model.carriers.iter().map(|c|c.beams.clone()).collect::<Vec<_>>(),model.carriers.iter().map(|c|c.rank).collect::<Vec<_>>(),model.focused,model.table_error);
    // The motion axis: the beams' axis (the frame's third), the reference's absorbing layer at 1.1606 um.
    let axis=2;let mut free=[false;3];free[axis]=true;
    let w=model.geometry[model.focused[0]].waist.unwrap();let cap=1.1606160660169471;
    let index=request.states.iter().position(|s|s=="ground-2-2").unwrap();let b22=model.ground.ground_states.iter().position(|&i|i==index).unwrap();
    let(block22,state22)={let k=model.ground.cluster[b22];(k,model.ground.clusters[k].position(b22).unwrap())};
    let trap=model.trap(block22,state22);let frequencies=trap.lab;let _=cap;
    // The trap's harmonic part moved exactly (flow), or free flight with the whole trap as kicks (drift).
    let flow=arg(7,"flow")=="flow";let truncation:f64=arg(8,"2.2e-16").parse().unwrap();let group:u64=arg(9,"8").parse().unwrap();let average=arg(10,"average")=="average";
    let mut settings=Settings::new(dt,&trap,truncation,average);settings.free=free;if !flow{settings.lab=[0.;3];}let _=w;
    let initial=Initial::new(&model,&request).unwrap();
    println!("trap {:?} kHz",frequencies.map(|f|f/std::f64::consts::TAU*1e3));
    let interval=10.;let steps=(interval/dt).round() as usize;let outputs=(t_end/interval).round() as usize+1;
    let run=std::time::Instant::now();
    let outcome=simulate(&model,&settings,&initial,seed,0,count,group,steps,outputs,&[outputs-1],None);
    let wall=run.elapsed().as_secs_f64();
    let estimates=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies},&[]);
    let labels=["ground-1--1","ground-1-0","ground-1-1","ground-2--2","ground-2--1","ground-2-0","ground-2-1","ground-2-2"];
    let ground_index:Vec<usize>=labels.iter().map(|l|request.states.iter().position(|s|s==l).unwrap()).collect();
    let report_every=((t_end/10.)/interval).round().max(1.) as usize;
    let(mut dt_sum,mut dt_n,mut worst_pop)=(0.,0.,0_f64);
    for(i,e)in estimates.iter().enumerate(){
        let t=i as f64*interval;let row=rows.iter().min_by(|a,b|(a["t_us"].as_f64().unwrap()-t).abs().total_cmp(&(b["t_us"].as_f64().unwrap()-t).abs())).unwrap();
        let tr=row["T_uK"].as_f64().unwrap();let rp:Vec<f64>=row["pops_ground"].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect();
        let pops:Vec<f64>=ground_index.iter().map(|&k|e.populations[k]).collect();
        let excited:f64=model.ground.excited_states.iter().map(|&k|e.populations[k]).sum();
        let dpop=pops.iter().zip(&rp).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
        if t>0.{dt_sum+=e.temperature[axis]/tr-1.;dt_n+=1.;worst_pop=worst_pop.max(dpop);}
        if i%report_every==0||i+1==estimates.len(){
            println!("t {:6.0}  T {:6.2}±{:4.2} ref {:6.2} ({:+5.1}%)  nbar {:5.2} ref {:5.2}  pF1 {:.4} ref {:.4}  m22 {:.4} ref {:.4}  m20 {:.4} ref {:.4}  exc {:.5} ref {:.5}  max|dpop| {:.4}  loss {:.1e} ref {:.1e}",
                t,e.temperature[axis],e.temperature_error[axis].error,tr,100.*(e.temperature[axis]/tr-1.),e.occupation[axis],row["nbar"].as_f64().unwrap(),
                pops[..3].iter().sum::<f64>(),row["pF1"].as_f64().unwrap(),pops[7],rp[7],pops[5],rp[5],excited,row["excited"].as_f64().unwrap(),dpop,e.loss,row["loss"].as_f64().unwrap());
        }
    }
    // The position and momentum distributions at the end against the reference's nearest listed ones.
    let dists=reference["distributions"].as_array().unwrap();
    let dref=dists.iter().min_by(|a,b|(a["t_us"].as_f64().unwrap()-t_end).abs().total_cmp(&(b["t_us"].as_f64().unwrap()-t_end).abs())).unwrap();
    let k_ref=reference["physics"]["k_ref_rad_per_um"].as_f64().unwrap();
    for(key,grid_key,scale,which)in [("P_x","x_nm",1e3,0),("P_p","p_hbark",k_ref,1)]{
        let grid:Vec<f64>=dref[grid_key].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect();let pr:Vec<f64>=dref[key].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect();
        let step=grid[1]-grid[0];let mut hist=vec![0.;grid.len()];let samples=&outcome.samples[0];
        for(r,kappa)in samples{let v=if which==0{r[axis]*scale}else{kappa[axis]/scale};let i=((v-grid[0])/step).round();if i>=0.&&(i as usize)<grid.len(){hist[i as usize]+=1./samples.len() as f64;}}
        let mid=grid.len()/2;let span=8;
        println!("{key} at {} us (bin {step}): {}",dref["t_us"],(mid-span..=mid+span).map(|i|format!("{:.0}:{:.3}/{:.3}",grid[i],hist[i],pr[i])).collect::<Vec<_>>().join(" "));
    }
    println!("{name} {variant}: {count} trajectories, dt {dt}, {:.1} s wall ({:.2} us per trajectory-step); mean dT {:+.1}%, worst |dpop| {:.4}; jumps per trajectory {:.0}",
        wall,wall*1e6/(count as f64*(outputs-1) as f64*steps as f64),100.*dt_sum/dt_n,worst_pop,outcome.jumps as f64/count as f64);
}
