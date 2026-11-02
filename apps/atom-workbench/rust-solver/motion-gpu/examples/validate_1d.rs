//! The trajectories on one row (the line through the focus, ny = 1) against the 1D GMW-o1
//! reference LONGREF_TW2 (the tweezer study's box engine: dt 0.2 us, o1 every 1 us, box ±1.365 um
//! with an absorbing ramp from 1.161 um): T, n̄, loss and the populations.
//! `cargo run --profile iterate --example validate_1d -- [t_end_us] [trajectories] [seed] [dt] [nx]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_core::{Complex,STATES};
use atom_motion_gpu::engine::{Cap,Engine,Internal,Settings};
use atom_motion_gpu::gpu::Gpu;
use atom_motion_gpu::model::{Grid,Model};
use atom_motion_gpu::stats::{occupation,sample};
use serde_json::Value;

const REFERENCE:&str=r"C:\Users\eadri\AppData\Local\Temp\claude\C--Users-eadri--claude-spares-spare-cmNOsa\8308d66c-699b-4483-ab9b-7305d93a9f7e\scratchpad\pgc1d\tweezer\refs\LONGREF_TW2.json";
const BOX:f64=1.3654306659022907;
const ABSORB:f64=1.1606160660169471;

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:f64|args.get(i).map_or(d,|s|s.parse().unwrap());
    let(t_end,b,seed,dt,nx,half)=(arg(1,1000.),arg(2,4096.) as usize,arg(3,1.) as u64,arg(4,0.2),arg(5,512.) as usize,arg(6,BOX));
    let reference:Value=serde_json::from_str(&std::fs::read_to_string(REFERENCE).unwrap()).unwrap();
    let rows=reference["rows"].as_array().unwrap();
    let mut request=scene::turned(scene::load(scene::PGC));
    // DIAGNOSTIC: free space -- the tweezer removed, the plane waves kept (a periodic box).
    if std::env::var("ATOM_GPU_FREE_SPACE").is_ok(){request.beams.retain(|b|b.waist_um.is_none());}
    let mut model=Model::new(&request).unwrap();
    let grid=Grid{nx,ny:1,dx:2.*half/nx as f64};
    // DIAGNOSTIC: a magic tweezer -- its fold and outside shift kept on the ground states only.
    if std::env::var("ATOM_GPU_MAGIC").is_ok(){
        let excited=model.ground.excited_states.clone();
        for m in model.ground.folds.iter_mut().chain(model.ground.outside.iter_mut()){for &i in &excited{for &j in &excited{m[(i,j)]=atom_motion_core::Complex::ZERO;}}}
    }
    model.tabulate(grid.reach()).unwrap();
    // DIAGNOSTIC: the light independent of the radius (the focus's responses everywhere), the
    // tweezer's trap kept: what the light's radial dependence does, apart from the trap's.
    if std::env::var("ATOM_GPU_UNIFORM_LIGHT").is_ok(){
        let e0=model.radial.eliminations[0].clone();
        for e in model.radial.eliminations.iter_mut(){e.responses=e0.responses.clone();e.derivatives=e0.derivatives.clone();}
    }
    // DIAGNOSTIC: a harmonic trap and light independent of the radius -- separable in x and y;
    // with ATOM_GPU_HARMONIC_TRAP the trap alone harmonic, the light's radial dependence kept.
    let harmonic_trap=std::env::var("ATOM_GPU_HARMONIC_TRAP").is_ok();
    if std::env::var("ATOM_GPU_HARMONIC").is_ok()||harmonic_trap{
        let step=model.radial.step;let e0=model.radial.eliminations[0].clone();let e1=model.radial.eliminations[1].clone();
        for(k,e)in model.radial.eliminations.iter_mut().enumerate(){
            let rho=k as f64*step;
            for b in 0..e.energies.len(){
                let c=2.*(e1.energies[b]-e0.energies[b])/(step*step);let u=e0.energies[b]+0.5*c*rho*rho;
                let shift=u-e.energies[b];e.energies[b]=u;e.hamiltonian[(b,b)].re+=shift;
            }
            if !harmonic_trap{e.responses=e0.responses.clone();e.derivatives=e0.derivatives.clone();}
        }
    }
    let g=&model.ground;
    let doppler_every=(1./dt).round() as usize;
    let settings=Settings{grid,dt,doppler_every,trajectories:b,seed,cap:(half>ABSORB).then_some(Cap{start:ABSORB,rate:20.}),scene:0};
    let start=std::time::Instant::now();
    let mut engine=Engine::new(std::rc::Rc::new(Gpu::new().unwrap()),&model,settings).unwrap();
    println!("{} x {}, dx {:.2} nm (k_max {:.0} hbar k), dt {dt} us, o1 every {doppler_every}, {b} trajectories; tables {:.2} s",
        grid.nx,grid.ny,grid.dx*1e3,std::f64::consts::PI/grid.dx/8.0528,start.elapsed().as_secs_f64());
    let index=request.states.iter().position(|s|s=="ground-2-2").unwrap();
    let b22=g.ground_states.iter().position(|&i|i==index).unwrap();
    let c22=model.block_states[1].iter().position(|&s|s==b22).unwrap();
    let mut amplitudes=vec![Complex::ZERO;5];amplitudes[c22]=Complex::ONE;
    engine.initialize(&vec![Internal{block:1,amplitudes};b],30.,&model).unwrap();
    let omega=model.trap_frequency(b22,0).max(1e-9);
    println!("trap {:.2} kHz",omega/std::f64::consts::TAU*1e3);
    let interval_us=50.;let interval=(interval_us/dt).round() as usize;let outputs=(t_end/interval_us).round() as usize;
    let report=|engine:&Engine,t:f64|{
        let(records,components)=engine.read_records().unwrap();
        let s=sample(&model,&records,&components,&engine.trajectory_weights);
        let row=rows.iter().min_by(|a,b|(a["t_us"].as_f64().unwrap()-t).abs().total_cmp(&(b["t_us"].as_f64().unwrap()-t).abs())).unwrap();
        let rp:Vec<f64>=row["pops_ground"].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect();
        let pf1:f64=g.ground_states[..3].iter().map(|&i|s.populations[i]).sum();
        let pf1e=(g.ground_states[..3].iter().map(|&i|s.population_errors[i].powi(2)).sum::<f64>()).sqrt();
        let excited:f64=g.excited_states.iter().map(|&i|s.populations[i]).sum();
        let worst=(0..8).map(|k|{let i=g.ground_states[k];((s.populations[i]-rp[k])/s.population_errors[i].max(1e-6)).abs()}).fold(0.,f64::max);
        let nbar=occupation(&model,s.momentum_square[0],s.position_variance[0],omega);
        println!("t {:6.0}  T {:6.2}±{:4.2} ({:6.2}, {:+5.1}%)  nbar {:5.2} ({:5.2})  loss {:.1e} ({:.1e})  pF1 {:.4}±{:.4} ({:.4})  exc {:.4} ({:.4})  pops worst {:.1} SE  tails {:.1e} {:.1e}",
            t,s.temperature[0],s.temperature_errors[0],row["T_uK"].as_f64().unwrap(),100.*(s.temperature[0]/row["T_uK"].as_f64().unwrap()-1.),
            nbar,row["nbar"].as_f64().unwrap(),s.loss,row["loss"].as_f64().unwrap(),pf1,pf1e,row["pF1"].as_f64().unwrap(),excited,row["excited"].as_f64().unwrap(),worst,s.tails[0],s.tails[2]);
        let _=STATES;
    };
    engine.observe_initial(false).unwrap();report(&engine,0.);
    if std::env::var("PROFILE").is_ok(){engine.set_profiling(true).unwrap();}
    let run=std::time::Instant::now();
    for o in 1..=outputs{for s in 0..interval{engine.step(s+1==interval,false).unwrap();}report(&engine,o as f64*interval_us);}
    // The hottest trajectories' share of ⟨κ²⟩: the tail the temperature's mean rests on.
    {let(records,components)=engine.read_records().unwrap();let w=&engine.trajectory_weights;
     let mut k2:Vec<(f64,usize)>=(0..b).filter(|&t|components[t]>0).map(|t|(w[t]*records[t*atom_motion_gpu::engine::RECORD+atom_motion_gpu::engine::REC_KX+2] as f64,t)).collect();
     let total:f64=k2.iter().map(|x|x.0).sum();k2.sort_by(|a,b|b.0.total_cmp(&a.0));
     println!("hottest trajectories' shares of the summed <k^2>: {}",k2.iter().take(8).map(|x|format!("{:.3}",x.0/total)).collect::<Vec<_>>().join(" "));}
    for(phase,ms)in engine.profile(){println!("  {phase}: {:.3} ms per step",ms/(outputs*interval) as f64);}
    println!("{} steps: {:.1} s ({:.3} us per trajectory-step)",outputs*interval,run.elapsed().as_secs_f64(),run.elapsed().as_secs_f64()*1e6/(outputs*interval*b) as f64);
}
