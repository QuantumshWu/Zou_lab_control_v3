//! The multirate exact step (engine.rs `Segment`) on a scene's fold candidates (ground.rs
//! `Ground::allowances`; `<candidate>` its index, 0 all folded, or `x` the first with explicit states):
//! - `beats <scene>`: every candidate's clusters, their beats and periods;
//! - `check <scene> <candidate>`: each cluster's components at random points and times against the
//!   tables' `local` (a cluster of ground states) and the front end's dense generator;
//! - `held <scene> <candidate> <trajectories> <t_end> <interval> <dt> <out.json> [truncation]`: the atom
//!   held at the focus, its populations (the frozen example's JSON, for the population core's);
//! - `solve <scene> <candidate> <duration> <samples> <trajectories> [dt]`: the solve's step choice (its pilot,
//!   the bound rule and the measured criterion, solve.rs) for the candidate, then its trajectories with the
//!   harmonic twin: T and the populations; with dt, that step instead of the choice;
//! - `moving <scene> <candidate> <trajectories> <t_end> <outputs> <dt,...> [truncation]`: atoms from the same initial
//!   states and jump streams per step (common random numbers): T per axis from ⟨κ²⟩ and F = 1, 2.
//! - `doubling <scene> <candidate> <pilot trajectories> <pilot us> <dt,...>`: the step-doubling check's parts on a
//!   short pilot's states: the momentum's and the impulse's sums, the populations', and the mechanical energy's
//!   difference and change split into the kinetic and the internal (Re⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ and the scalar) parts.
//! - `profile <scene> <allowance> <dt> <truncation> <trajectories> <duration us> [horizon us] [beats' share]`: the exact step's cost on one thread
//!   along trajectories from the solve's seed: per cluster its segments' paths, nodes and beats (each pair's bound
//!   against the share, left out or not), and the microseconds of its components, its segment and the whole step.
//! - `work <scene> <candidate> <trajectories> <duration us> <dt> <truncation>`: along the trajectories, the folded
//!   saturation S = 2η² (the readout's folded excited population, twice) weighted by the force's work |F·v| per
//!   step, ⟨S |F·v|⟩/⟨|F·v|⟩, against its plain mean, per tenth of the run; the explicit states' population; and
//!   the kinetic energy's change per atom and us over the steps with jumps and those without.
//! `cargo run --profile iterate -p atom-motion-3d --example multirate -- <mode> <scene> ...`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::control::Harmonic;
use atom_motion_3d::engine::{Engine,Settings,Trajectory};
use atom_motion_3d::ground::{Ground,Path};
use atom_motion_3d::ledger::BUDGET;
use atom_motion_3d::model::{Generator,Model};
use atom_motion_3d::solve::{CONFIDENCE,POPULATION_TARGET,TEMPERATURE_TARGET};
use atom_motion_3d::stats::{summarize,Means,Scales,KT_PER_UK};
use atom_motion_3d::{initial_trajectory,sample_states,simulate,simulate_from,Complex,Initial,Request,Stratum,Variates,STATES};
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;
use rayon::prelude::*;
use std::f64::consts::TAU;
use std::time::Instant;

/// The candidate's model: its index in the ladder, or the first with explicit excited states.
fn candidate(request:&Request,which:&str)->(f64,Model){
    let ladder=Ground::allowances(request).unwrap();
    if which=="x"{for &a in &ladder{let m=Model::with(request,a).unwrap();if (0..m.clusters()).any(|k|m.mixed(k)){return(a,m);}}panic!("no candidate keeps explicit states");}
    let a=ladder[which.parse::<usize>().unwrap()];(a,Model::with(request,a).unwrap())
}
fn describe(m:&Model)->String{
    (0..m.clusters()).map(|k|{let(beats,period)=m.beats(k);let half=beats.len()/2;
        format!("{}({}){}",m.dimension(k),m.dimension(k)-m.ground_dimension(k),if half>0{format!(" beats {:?} MHz period {}",beats[half..].iter().map(|w|(w/TAU*1e4).round()/1e4).collect::<Vec<_>>(),
            period.map_or("none".into(),|t|format!("{t:.5} us")))}else{String::new()})}).collect::<Vec<_>>().join(", ")
}

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let mode=arg(1,"beats");let name=arg(2,"GMD1C");let request=scene::scene(&name);
    match mode.as_str(){
        "beats"=>{
            for(i,&a)in Ground::allowances(&request).unwrap().iter().enumerate(){
                let clock=Instant::now();match Model::with(&request,a){
                    Ok(m)=>println!("{name} candidate {i} (allowance {a:.3e}, {:.1} s): {}",clock.elapsed().as_secs_f64(),describe(&m)),
                    Err(e)=>println!("{name} candidate {i} (allowance {a:.3e}): {e}")}
            }
        }
        "check"=>check(&request,&arg(3,"0")),
        "held"=>held(&name,&request,&arg(3,"0"),arg(4,"2048").parse().unwrap(),arg(5,"200").parse().unwrap(),arg(6,"5").parse().unwrap(),arg(7,"0.2").parse().unwrap(),&arg(8,"held.json"),arg(9,"1e-12").parse().unwrap()),
        "solve"=>solve(&name,request,&arg(3,"x"),arg(4,"2000").parse().unwrap(),arg(5,"11").parse().unwrap(),arg(6,"512").parse().unwrap(),args.get(7).map(|x|x.parse().unwrap())),
        "moving"=>moving(&request,&arg(3,"0"),arg(4,"256").parse().unwrap(),arg(5,"40").parse().unwrap(),arg(6,"8").parse().unwrap(),&arg(7,"0.21739"),arg(8,"1e-12").parse().unwrap()),
        "profile"=>profile(&request,arg(3,"0.0095").parse().unwrap(),arg(4,"0.2304").parse().unwrap(),arg(5,"1e-6").parse().unwrap(),arg(6,"32").parse().unwrap(),arg(7,"200").parse().unwrap(),args.get(8).map(|x|x.parse().unwrap()),args.get(9).map(|x|x.parse().unwrap())),
        "work"=>work(&request,&arg(3,"0"),arg(4,"256").parse().unwrap(),arg(5,"2000").parse().unwrap(),arg(6,"0.2924").parse().unwrap(),arg(7,"1e-6").parse().unwrap()),
        "doubling"=>doubling(&request,&arg(3,"x"),arg(4,"16").parse().unwrap(),arg(5,"200").parse().unwrap(),&arg(6,"0.24,0.05,0.005,0.0005")),
        other=>panic!("unknown mode {other}"),
    }
}

/// The components at random points and times against `local` and the dense generator: the largest
/// differences of H (its scalar restored) and of the gradients, relative to their scales.
fn check(request:&Request,which:&str){
    let(_,mut model)=candidate(request,which);model.doppler=false;let g=&model.ground;println!("clusters {}",describe(&model));
    // Each beat at the focus: its frequency, its size ‖H_j‖₁, its mixing ‖H_j‖₁/|Ω| and its largest entries.
    let p=model.point([0.;3],[0.;3]);
    for k in 0..model.clusters(){
        let c=model.components(k,&p);let n=c.n();let cl=&g.clusters[k];
        let label=|i:usize|if i<cl.states.len(){request.states[g.ground_states[cl.states[i]]].clone()}else{request.states[g.excited_states[cl.excited[i-cl.states.len()]]].clone()};
        let norm=|h:&[Complex]|(0..n).map(|b|(0..n).map(|a|h[a*n+b].abs()).sum::<f64>()).fold(0.,f64::max);
        println!("cluster {k}: static ‖H₀‖₁ {:.3e} rad/us",norm(&c.statics.hamiltonian));
        for b in c.beats.iter().filter(|b|b.omega>0.){
            let mut entries:Vec<(f64,usize,usize)>=(0..n*n).map(|i|(b.hamiltonian[i].abs(),i/n,i%n)).filter(|x|x.0>0.).collect();entries.sort_by(|x,y|y.0.total_cmp(&x.0));
            println!("  beat {:.4} MHz: ‖H_j‖₁ {:.3e} rad/us, mixing {:.2e}; largest {:?}",b.omega/TAU,norm(&b.hamiltonian),norm(&b.hamiltonian)/b.omega,
                entries.iter().take(6).map(|&(v,a,bb)|format!("{} <- {}: {:.2e}",label(a),label(bb),v)).collect::<Vec<_>>());
        }
    }
    let tabulated=|p:&Path|model.carriers[p.via.carrier].beams.len()==1;
    let mut seed=0x5EED_u64;let mut next=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;(seed>>11) as f64/(1u64<<53) as f64-0.5};
    let(mut worst_local,mut worst_dense,mut worst_gradient)=(0_f64,0_f64,0_f64);
    for _ in 0..8{
        let r=[0.4*next(),0.4*next(),1.2*next()];let p=model.point(r,[0.;3]);let e=g.eliminate(&model.hamiltonian_at_point(r)).unwrap();
        for k in 0..model.clusters(){
            let n=model.dimension(k);let c=model.components(k,&p);
            for _ in 0..4{
                let t=20.*(next()+0.5);let x=c.at(t);
                let full=|h:&Generator,a:usize,b:usize|h.hamiltonian[a*n+b]+if a==b{Complex::new(h.scalar,0.)}else{Complex::ZERO};
                let dense=g.local_with(&e,k,r,t,&tabulated).unwrap();let scale=dense.effective.largest();
                for a in 0..n{for b in 0..n{worst_dense=worst_dense.max((full(&x,a,b)-dense.effective[(a,b)]).abs()/scale);}}
                if !model.mixed(k){
                    let l=Generator::of(&model.local(k,&p,t),n);let gs=l.gradient.iter().flat_map(|v|v.iter()).map(|z|z.abs()).fold(0.,f64::max);
                    for a in 0..n{for b in 0..n{worst_local=worst_local.max((full(&x,a,b)-full(&l,a,b)).abs()/scale);
                        for q in 0..3{worst_gradient=worst_gradient.max((x.gradient[q][a*n+b]-l.gradient[q][a*n+b]).abs()/gs);}}}
                }
            }
        }
    }
    println!("components against local: H {worst_local:.1e}, gradient {worst_gradient:.1e}; against the dense generator: H {worst_dense:.1e} (the tables' interpolation)");
}

/// The atom held at the focus (no axis free, no loss): its populations with their standard errors.
fn held(name:&str,request:&Request,which:&str,count:u64,t_end:f64,interval:f64,dt:f64,out:&str,truncation:f64){
    let(a,model)=candidate(request,which);println!("{name} candidate {which} (allowance {a:.3e}): {}",describe(&model));
    let initial=Initial::new(&model,request).unwrap();let trap=initial.traps[0];
    let(outputs,steps)=((t_end/interval).round() as usize+1,(interval/dt).round().max(1.) as usize);
    let mut settings=Settings::new(interval/steps as f64,&trap,truncation,false);settings.free=[false;3];settings.loss=false;
    let clock=Instant::now();
    let outcome=simulate_from(&model,&settings,&initial,7,0,count,8,steps,outputs,&[],[0.;3]);
    let est=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:[1.;3]},&[]);
    println!("engine: {count} trajectories, dt {:.4} us, {:.1} s wall, {} jumps",settings.dt,clock.elapsed().as_secs_f64(),outcome.jumps);
    let level=|p:&[f64],f:&str|request.states.iter().enumerate().filter(|(_,s)|s.starts_with(f)).map(|(i,_)|p[i]).sum::<f64>();
    for i in (0..outputs).step_by((outputs/10).max(1)).chain([outputs-1]){let p=&est[i].populations;
        println!("t {:6.1}  F1 {:.4}  F2 {:.4}  excited {:.5}",i as f64*interval,level(p,"ground-1-"),level(p,"ground-2-"),1.-level(p,"ground-"));}
    let pops:Vec<Vec<f64>>=est.iter().map(|e|e.populations.clone()).collect();
    let json=format!("{{\"scene\":\"{name}\",\"states\":{:?},\"timesUs\":{:?},\"eliminated\":{pops:?},\"engine\":{pops:?},\"engineErrors\":{:?},\"trajectories\":{count}}}",
        request.states,(0..outputs).map(|i|i as f64*interval).collect::<Vec<_>>(),est.iter().map(|e|e.population_errors.clone()).collect::<Vec<_>>());
    std::fs::write(out,json).unwrap();println!("written {out}");
}

/// A checked step: its steps per interval and its ratios to the population's and the motion's shares.
#[derive(Clone,Copy,Debug)]
struct Check{steps:usize,population:f64,motion:f64}
impl Check{fn excess(&self)->f64{self.population.max(self.motion)}fn met(&self)->bool{self.excess()<=1.}fn decided_by_population(&self)->bool{self.motion<=1.&&self.population>self.motion}}
/// solve.rs `step_excess`.
fn step_excess(engine:&Engine,states:&[Trajectory],waiting:f64)->[f64;2]{
    let dt=engine.settings.dt;let richardson=4./3.;
    let doubled:Vec<([f64;2],[f64;3],[f64;2])>=states.par_iter().map(|t|engine.doubling(t,dt)).collect();
    let(mut population,mut difference,mut impulse,mut energy,mut friction)=(0.,0.,0.,0.,0.);
    for(motion,internal,work)in &doubled{difference+=motion[0];impulse+=motion[1];population+=internal[0];energy+=work[0];friction+=work[1];}
    let bound=richardson*population/states.len().max(1) as f64*(waiting/dt).max(1.);
    let relative=if impulse>0.{richardson*difference/impulse}else{0.};
    let dissipated=if friction!=0.{richardson*(energy/friction).abs()}else{0.};
    [bound/(A_PRIORI_SHARE*POPULATION_TARGET),relative.max(dissipated)/(A_PRIORI_SHARE*TEMPERATURE_TARGET)]
}
/// solve.rs `accumulated_excess`.
fn accumulated_excess(engine:&Engine,states:&[Trajectory],horizon:f64,subsample:usize)->f64{
    let dt=engine.settings.dt;let richardson=4./3.;let count=((horizon/dt).ceil() as usize).max(1);
    let every=(states.len()/subsample.max(1)).max(1);let picked:Vec<&Trajectory>=states.iter().step_by(every).collect();
    let errors:Vec<f64>=picked.par_iter().map(|t|{
        let n=engine.model.dimension(t.cluster);let(one,two)=(engine.coast(t,dt,count),engine.coast(t,0.5*dt,2*count));
        let(a,b)=(one.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>(),two.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>());
        richardson*(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max)
    }).collect();
    let m=errors.len().max(1) as f64;let mean=errors.iter().sum::<f64>()/m;
    let spread=(errors.iter().map(|e|(e-mean).powi(2)).sum::<f64>()/(m*(m-1.).max(1.))).sqrt();
    (mean+CONFIDENCE*spread)/(A_PRIORI_SHARE*POPULATION_TARGET)
}

/// The solve's step choice for a candidate (solve.rs `run`: its pilot, the bound rule, the measured
/// population criterion where it decides; the largest step that met the targets kept), then its
/// trajectories in one batch with the harmonic twin.
fn solve(name:&str,mut request:Request,which:&str,duration:f64,samples:usize,count:u64,forced:Option<f64>){
    request.times_us=(0..samples).map(|i|duration*i as f64/(samples-1) as f64).collect();
    let clock=Instant::now();let(a,model)=candidate(&request,which);let initial=Initial::new(&model,&request).unwrap();
    println!("{name} candidate {which} (allowance {a:.3e}, model {:.1} s): {}",clock.elapsed().as_secs_f64(),describe(&model));
    let(outputs,interval)=(samples,duration/(samples-1) as f64);let seed=0x3D_5EED;let threads=rayon::current_num_threads() as u64;let pilot=2*threads;
    let trap=initial.traps[(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap()];
    let thermal=(KT_PER_UK*request.temperature_uk*model.hbar_over_mass).sqrt();
    let pattern=model.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max)*thermal;
    let rate=if pattern>0.{pattern}else{trap.frequencies.iter().copied().fold(0.,f64::max)};
    let first=((interval*rate/(A_PRIORI_SHARE*TEMPERATURE_TARGET).sqrt()).ceil() as usize).max(1);
    let plain=|steps:usize,truncation:f64|Settings::new(interval/steps as f64,&trap,truncation,true);
    let clock=Instant::now();
    // The pilot as the solve takes it: its exponentials to double precision, the beats left out within the
    // step's population budget over the whole run.
    let piloting=|steps:usize|{let mut s=plain(steps,f64::EPSILON);s.beats=A_PRIORI_SHARE*A_PRIORI_SHARE*POPULATION_TARGET*s.dt/duration;s.horizon=duration;s};
    let(states,jumps)=sample_states(&model,&piloting(first),&initial,seed,pilot,first,outputs);
    let waiting=pilot as f64*duration/(jumps.max(1) as f64);let horizon=waiting.min(duration);
    println!("pilot: {pilot} trajectories at dt {:.4} us, {:.1} s; waiting {waiting:.2} us; {} states",interval/first as f64,clock.elapsed().as_secs_f64(),states.len());
    let settings=|steps:usize|{let dt=interval/steps as f64;let mut s=plain(steps,A_PRIORI_SHARE*A_PRIORI_SHARE*POPULATION_TARGET*dt/waiting);s.doppler=BUDGET;s.horizon=waiting;s};
    let clock=Instant::now();
    let doubled=|steps:usize|{let r=step_excess(&Engine::new(&model,settings(steps)),&states,waiting);Check{steps,population:r[0],motion:r[1]}};
    let search=|first:usize,measured:bool,known:&[Check],checked:&mut Vec<Check>|{
        let mut steps=first;
        for _ in 0..8{
            if checked.iter().any(|c|c.steps==steps){break;}
            let mut c=known.iter().find(|c|c.steps==steps).copied().unwrap_or_else(||doubled(steps));
            if measured&&c.decided_by_population(){c.population=accumulated_excess(&Engine::new(&model,settings(steps)),&states,horizon,pilot as usize);}
            println!("  {} check: dt {:.5} us, population {:.3}, motion {:.3}",if measured{"measured"}else{"bound"},interval/steps as f64,c.population,c.motion);
            checked.push(c);steps=((steps as f64*c.excess().sqrt()).ceil() as usize).max(1);
        }
    };
    let mut bound=vec![];if let Some(dt)=forced{let steps=(interval/dt).round().max(1.) as usize;bound.push(Check{steps,population:0.,motion:0.});println!("  forced dt {:.4} us",interval/steps as f64);}else{search(first,false,&[],&mut bound);}
    let grown=*bound.iter().filter(|c|c.met()).min_by_key(|c|c.steps).unwrap_or(&bound[bound.len()-1]);
    let mut measured=vec![];if grown.decided_by_population(){search(grown.steps,true,&bound,&mut measured);}
    let chosen=bound.iter().chain(&measured).filter(|c|c.met()).min_by_key(|c|c.steps).copied().unwrap_or(grown);
    let steps=chosen.steps;let s=settings(steps);
    println!("step: dt {:.4} us (the bound rule's {:.4} us), {:.1} s",s.dt,interval/grown.steps as f64,clock.elapsed().as_secs_f64());
    let harmonic=Harmonic::new(&model,&s,&initial,steps,outputs);
    let clock=Instant::now();
    let outcome=simulate(&model,&s,&initial,seed,0,count,(count/(2*threads)).max(1),steps,outputs,&[],Some(&Variates{harmonic:&harmonic,twin:None}));
    let wall=clock.elapsed().as_secs_f64();
    let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:harmonic.mean[i],twin:None}).collect();
    let est=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab},&means);
    println!("{count} trajectories: {wall:.1} s wall, {} jumps, {:.1} ms per trajectory-ms; beats left out: their bound over the horizon {:.2e} (pilot {:.2e})",outcome.jumps,
        wall*1e3*threads as f64/(count as f64*duration*1e-3),outcome.beats*horizon,states.iter().map(|t|t.beats).fold(0.,f64::max)*horizon);
    let pick=|label:&str|request.states.iter().position(|x|x==label).unwrap();let m22=pick("ground-2-2");
    let level=|p:&[f64],f:&str|request.states.iter().enumerate().filter(|(_,s)|s.starts_with(f)).map(|(i,_)|p[i]).sum::<f64>();
    for i in 1..outputs{let e=&est[i];let t=&e.temperature;let te=&e.temperature_error;
        println!("t {:7.0}  T {:6.2}±{:4.2} {:6.2}±{:4.2} {:6.2}±{:4.2}  m22 {:.4}  pF1 {:.4}±{:.4}  excited {:.5}  loss {:.1e}",request.times_us[i],t[0],te[0],t[1],te[1],t[2],te[2],
            e.populations[m22],level(&e.populations,"ground-1-"),e.population_errors.iter().fold(0_f64,|m,&x|m.max(x)),1.-level(&e.populations,"ground-"),e.loss);}
}

/// Atoms from the same initial states and jump streams at each step (the research's floquet_motion
/// setup: seed 0xF10, one stratum each, exponentials to 1e-12 by default): T per axis from ⟨κ²⟩, F = 1 and F = 2,
/// and against the first step's the differences of the same atoms.
fn moving(request:&Request,which:&str,count:u64,t_end:f64,outputs:usize,steps:&str,truncation:f64){
    let(a,model)=candidate(request,which);println!("candidate {which} (allowance {a:.3e}): {}",describe(&model));
    let initial=Initial::new(&model,request).unwrap();
    let trap=initial.traps[(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap()];
    let interval=t_end/outputs as f64;let seed=0xF10_u64;
    let f1:Vec<usize>=request.states.iter().enumerate().filter(|(_,s)|s.starts_with("ground-1-")).map(|(i,_)|i).collect();
    let f2:Vec<usize>=request.states.iter().enumerate().filter(|(_,s)|s.starts_with("ground-2-")).map(|(i,_)|i).collect();
    let to_t=|k2:f64|k2*model.hbar_over_mass/KT_PER_UK;
    let mut runs:Vec<Vec<Vec<([f64;3],f64,f64,bool)>>>=vec![];
    for dt in steps.split(',').map(|x|x.parse::<f64>().unwrap()){
        let n=(interval/dt).round().max(1.) as usize;let settings=Settings::new(interval/n as f64,&trap,truncation,false);
        let clock=Instant::now();
        let rows:Vec<Vec<([f64;3],f64,f64,bool)>>=(0..count).into_par_iter().map(|index|{
            let engine=Engine::new(&model,settings.clone());let mut t=initial_trajectory(&model,&settings,&initial,seed,index,Stratum::single(index));
            let read=|t:&Trajectory|{let p=model.point(t.r,t.kappa);let q:[f64;STATES]=model.populations(t.cluster,&p,&t.psi,t.time);
                (t.kappa.map(|k|k*k),f1.iter().map(|&i|q[i]).sum::<f64>(),f2.iter().map(|&i|q[i]).sum::<f64>(),t.alive)};
            let mut rows=vec![read(&t)];for _ in 0..outputs{for _ in 0..n{if t.alive{engine.step(&mut t);}}rows.push(read(&t));}rows}).collect();
        println!("dt {:.5} us: {:.1} s wall",settings.dt,clock.elapsed().as_secs_f64());
        for o in 0..=outputs{
            let alive:Vec<usize>=(0..count as usize).filter(|&i|rows[i][o].3).collect();let k=alive.len() as f64;
            let stat=|f:&dyn Fn(usize)->f64|{let v:Vec<f64>=alive.iter().map(|&i|f(i)).collect();let m=v.iter().sum::<f64>()/k;(m,(v.iter().map(|x|(x-m)*(x-m)).sum::<f64>()/(k-1.).max(1.)/k).sqrt())};
            let t:Vec<(f64,f64)>=(0..3).map(|a|stat(&|i|to_t(rows[i][o].0[a]))).collect();let(p1,p2)=(stat(&|i|rows[i][o].1),stat(&|i|rows[i][o].2));
            let mut line=format!("  t {:6.2}  T {:6.2}±{:4.2} {:6.2}±{:4.2} {:6.2}±{:4.2}  F1 {:.4}±{:.4}  F2 {:.4}±{:.4}",o as f64*interval,t[0].0,t[0].1,t[1].0,t[1].1,t[2].0,t[2].1,p1.0,p1.1,p2.0,p2.1);
            if let Some(base)=runs.first(){
                let both:Vec<usize>=(0..count as usize).filter(|&i|rows[i][o].3&&base[i][o].3).collect();let k=both.len() as f64;
                let stat=|f:&dyn Fn(usize)->f64|{let v:Vec<f64>=both.iter().map(|&i|f(i)).collect();let m=v.iter().sum::<f64>()/k;(m,(v.iter().map(|x|(x-m)*(x-m)).sum::<f64>()/(k-1.).max(1.)/k).sqrt())};
                let d:Vec<(f64,f64)>=(0..3).map(|a|stat(&|i|to_t(rows[i][o].0[a])-to_t(base[i][o].0[a]))).collect();let d1=stat(&|i|rows[i][o].1-base[i][o].1);
                line+=&format!("  vs first: ΔT {:+.2}±{:.2} {:+.2}±{:.2} {:+.2}±{:.2}  ΔF1 {:+.4}±{:.4}",d[0].0,d[0].1,d[1].0,d[1].1,d[2].0,d[2].1,d1.0,d1.1);
            }
            println!("{line}");
        }
        runs.push(rows);
    }
}

/// The step-doubling check's parts per step (solve.rs `step_excess`) on a short pilot's states.
fn doubling(request:&Request,which:&str,count:u64,duration:f64,steps:&str){
    let(a,model)=candidate(request,which);println!("candidate {which} (allowance {a:.3e}): {}",describe(&model));
    let initial=Initial::new(&model,request).unwrap();
    let trap=initial.traps[(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap()];
    let outputs=11;let interval=duration/(outputs-1) as f64;let first=((interval/0.24).ceil() as usize).max(1);
    let mut pilot=Settings::new(interval/first as f64,&trap,f64::EPSILON,true);pilot.beats=A_PRIORI_SHARE*A_PRIORI_SHARE*POPULATION_TARGET*pilot.dt/duration;
    let clock=Instant::now();let(states,jumps)=sample_states(&model,&pilot,&initial,0x3D_5EED,count,first,outputs);
    let waiting=count as f64*duration/(jumps.max(1) as f64);
    println!("pilot: {count} x {duration} us at dt {:.4}, {:.1} s; waiting {waiting:.1} us; {} states",pilot.dt,clock.elapsed().as_secs_f64(),states.len());
    let hm=model.hbar_over_mass;let kinetic=|t:&Trajectory|0.5*hm*t.kappa.iter().map(|k|k*k).sum::<f64>();
    for dt in steps.split(',').map(|x|x.parse::<f64>().unwrap()){
        let mut s=Settings::new(dt,&trap,A_PRIORI_SHARE*A_PRIORI_SHARE*POPULATION_TARGET*dt/waiting,true);s.doppler=BUDGET;let engine=Engine::new(&model,s);
        let clock=Instant::now();
        // Per state: |Δκ|, |kick|, max |Δp|, and (one − two, two − start) of the kinetic and the internal energy.
        let rows:Vec<[f64;7]>=states.par_iter().map(|t|{
            let(one,two)=(engine.coast(t,dt,1),engine.coast(t,0.5*dt,2));
            let dk=(0..3).map(|a|(one.kappa[a]-two.kappa[a]).powi(2)).sum::<f64>().sqrt();
            let kick=(0..3).map(|a|(two.kappa[a]-t.kappa[a]).powi(2)).sum::<f64>().sqrt();
            let n=model.dimension(t.cluster);let(x,y)=(one.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>(),two.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>());
            let dp=(0..n).map(|i|(one.psi[i].norm_sqr()/x-two.psi[i].norm_sqr()/y).abs()).fold(0.,f64::max);
            let(e0,e1,e2)=(engine.energy(t),engine.energy(&one),engine.energy(&two));let(k0,k1,k2)=(kinetic(t),kinetic(&one),kinetic(&two));
            [dk,kick,dp,k1-k2,(e1-k1)-(e2-k2),k2-k0,(e2-k2)-(e0-k0)]}).collect();
        let sum=|i:usize|rows.iter().map(|r|r[i]).sum::<f64>();let abs=|i:usize|rows.iter().map(|r|r[i].abs()).sum::<f64>();let m=rows.len() as f64;
        println!("dt {dt:.5} ({:.1} s): |Δκ|/|kick| {:.3e} ({:.2e}/{:.2e}); mean max|Δp| {:.2e}; ΔE kinetic Σ {:+.2e} (Σ|·| {:.2e}), internal Σ {:+.2e} (Σ|·| {:.2e}); E2−E0 kinetic Σ {:+.2e} (Σ|·| {:.2e}), internal Σ {:+.2e} (Σ|·| {:.2e})",
            clock.elapsed().as_secs_f64(),sum(0)/sum(1),sum(0)/m,sum(1)/m,sum(2)/m,sum(3),abs(3),sum(4),abs(4),sum(5),abs(5),sum(6),abs(6));
    }
}

/// The exact step's cost on one thread along trajectories of a model at `allowance` (`Engine::probe`).
fn profile(request:&Request,allowance:f64,dt:f64,truncation:f64,count:u64,duration:f64,horizon:Option<f64>,beats:Option<f64>){
    // The request over the duration (the fold's rule takes its 1/T floor), 21 outputs as the cost probe's.
    let mut request=request.clone();request.times_us=(0..21).map(|i|duration*i as f64/20.).collect();let request=&request;
    let model=Model::with(request,allowance).unwrap();println!("allowance {allowance:.3e}: {}",describe(&model));
    let initial=Initial::new(&model,request).unwrap();let trap=initial.traps[0];
    let mut settings=Settings::new(dt,&trap,truncation,true);if let Some(h)=horizon{settings.horizon=h;}if let Some(b)=beats{settings.beats=b;}let engine=Engine::new(&model,settings.clone());let steps=(duration/dt).round() as usize;
    // Per cluster: steps, the step's, the components' and the segment's seconds, nodes, paths, and per beat (Ω) its largest bound and drops.
    struct Acc{steps:usize,step:f64,components:f64,segment:f64,nodes:usize,paths:Vec<(&'static str,usize)>,beats:Vec<(f64,f64,usize,usize)>,weighted:usize,counted:usize}
    let mut acc:Vec<Acc>=(0..model.clusters()).map(|_|Acc{steps:0,step:0.,components:0.,segment:0.,nodes:0,paths:vec![],beats:vec![],weighted:0,counted:0}).collect();
    let mut share=0.;
    for index in 0..count{
        let mut t=initial_trajectory(&model,&settings,&initial,0x3D_5EED,index,Stratum::single(index));
        for _ in 0..steps{
            if !t.alive{break;}let k=t.cluster;let a=&mut acc[k];
            if model.mixed(k)||model.beating(k){
                // The probe at the step's half-step point is near the step's own; its build times stand for the step's.
                let cpu=scene::cpu_seconds();let p=model.point(t.r,t.kappa);let probe=engine.probe(k,&p,t.time,dt,&t.psi);share=probe.share;if let Some(w)=probe.weighted{a.weighted+=w;a.counted+=1;}
                let built=scene::cpu_seconds()-cpu;let wall=probe.seconds[0]+probe.seconds[1];
                // The probe's CPU time shared by its two parts as their wall times.
                a.components+=built*probe.seconds[0]/wall.max(1e-300);a.segment+=built*probe.seconds[1]/wall.max(1e-300);a.nodes+=probe.nodes;
                match a.paths.iter_mut().find(|x|x.0==probe.path){Some(x)=>x.1+=1,None=>a.paths.push((probe.path,1))}
                for(w,bound,left)in probe.beats{match a.beats.iter_mut().find(|x|(x.0-w).abs()<=1e-9*w){Some(x)=>{x.1=x.1.max(bound);x.2+=1;x.3+=left as usize;}None=>a.beats.push((w,bound,1,left as usize))}}
            }
            let cpu=scene::cpu_seconds();engine.step(&mut t);a.step+=scene::cpu_seconds()-cpu;a.steps+=1;
        }
    }
    for(k,a)in acc.iter().enumerate(){
        if a.steps==0{continue;}let m=a.steps as f64;
        println!("cluster {k} {}({}): {} steps, {:.1} CPU-us per step; components {:.1} us, segment {:.1} us, the rest {:.1} us; nodes {:.1} per step; paths {:?}",
            model.dimension(k),model.dimension(k)-model.ground_dimension(k),a.steps,1e6*a.step/m,1e6*a.components/m,1e6*a.segment/m,1e6*(a.step-a.components-a.segment)/m,a.nodes as f64/m,a.paths);
        if a.counted>0{println!("    amplitude-weighted spacing: {:.1} nodes per step on average (the rule's {:.1})",a.weighted as f64/a.counted as f64,a.nodes as f64/m);}
        for &(w,bound,seen,left)in &a.beats{println!("    beat {:.4} MHz: largest bound {:.2e} (share {:.1e}), left out {}/{}",w/TAU,bound,share,left,seen);}
    }
}

/// The fold's saturation where the force works (`work` mode): per tenth of the run Σ S|F·v|, Σ|F·v|, Σ S and the
/// explicit states' population over every step of every trajectory, F the Ehrenfest force after the step at its point.
fn work(request:&Request,which:&str,count:u64,duration:f64,dt:f64,truncation:f64){
    let(a,model)=candidate(request,which);println!("candidate {which} (allowance {a:.3e}): {}",describe(&model));
    let initial=Initial::new(&model,request).unwrap();let trap=initial.traps[0];
    let settings=Settings::new(dt,&trap,truncation,true);let steps=(duration/dt).round() as usize;let hm=model.hbar_over_mass;let bins=10;
    let clock=Instant::now();
    let sums:Vec<Vec<[f64;10]>>=(0..count).into_par_iter().map(|index|{
        let engine=Engine::new(&model,settings.clone());let mut t=initial_trajectory(&model,&settings,&initial,0x3D_5EED,index,Stratum::single(index));
        let mut out=vec![[0.;10];bins];let kinetic=|t:&Trajectory|0.5*hm*t.kappa.iter().map(|k|k*k).sum::<f64>();
        for i in 0..steps{
            if !t.alive{break;}let(e0,j0)=(kinetic(&t),t.jumps);engine.step(&mut t);let de=kinetic(&t)-e0;
            {let b=&mut out[(i*bins/steps).min(bins-1)];if t.jumps>j0{b[6]+=de;b[7]+=(t.jumps-j0) as f64;}else{b[8]+=de;b[9]+=1.;}}
            let p=model.point(t.r,t.kappa);let g=model.generator(t.cluster,&p,t.time);let f=g.force(&t.psi);
            let w=(0..3).map(|x|f[x]*hm*t.kappa[x]).sum::<f64>().abs();
            let(pops,folded)=model.readout(t.cluster,&p,&t.psi,t.time);let s=2.*folded;
            let explicit:f64=model.ground.excited_states.iter().map(|&e|pops[e]).sum::<f64>()-folded;
            let b=&mut out[(i*bins/steps).min(bins-1)];b[0]+=s*w;b[1]+=w;b[2]+=s;b[3]+=1.;b[4]+=explicit*w;b[5]+=explicit;
        }
        out}).collect();
    println!("{count} trajectories, {:.1} s wall",clock.elapsed().as_secs_f64());
    let mut total=[0.;10];let uk=|e:f64|e/KT_PER_UK;
    for b in 0..bins{let mut x=[0.;10];for s in &sums{for k in 0..10{x[k]+=s[b][k];}}for k in 0..10{total[k]+=x[k];}
        let span=duration/bins as f64*count as f64;
        println!("  t {:6.0}-{:6.0} us: S work-weighted {:.4}, mean {:.4}; explicit excited work-weighted {:.4}, mean {:.4}; kinetic energy per atom-us: steps with jumps {:+.4} uK ({:.4} jumps), without {:+.4} uK",
            duration*b as f64/bins as f64,duration*(b+1) as f64/bins as f64,x[0]/x[1],x[2]/x[3],x[4]/x[1],x[5]/x[3],uk(x[6]/span),x[7]/span,uk(x[8]/span));}
    println!("  whole run: S work-weighted {:.4}, mean {:.4}; explicit excited work-weighted {:.4}, mean {:.4}",total[0]/total[1],total[2]/total[3],total[4]/total[1],total[5]/total[3]);
}
