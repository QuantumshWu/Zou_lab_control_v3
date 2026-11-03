//! The tweezer PGC (beams along x only) in the plane, against the 1D reference along the beams
//! (LONGREF_TW2): T_x, n̄_x and the populations; T_y heats by the photons' recoil alone. With
//! `swap` 1 the frame's plane axes are exchanged (the beams along y): T_y then follows the
//! reference, testing the column path.
//! `cargo run --profile iterate --example validate_2d -- [t_end_us] [trajectories] [seed] [swap] [n] [dt]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_core::Complex;
use atom_motion_gpu::engine::{Engine,Internal,Settings};
use atom_motion_gpu::gpu::Gpu;
use atom_motion_gpu::model::{Grid,Model};
use atom_motion_gpu::stats::{occupation,sample};
use serde_json::Value;

const REFERENCE:&str=r"C:\Users\eadri\AppData\Local\Temp\claude\C--Users-eadri--claude-spares-spare-cmNOsa\8308d66c-699b-4483-ab9b-7305d93a9f7e\scratchpad\pgc1d\tweezer\refs\LONGREF_TW2.json";

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:f64|args.get(i).map_or(d,|s|s.parse().unwrap());
    let(t_end,b,seed,swap,n,dt,dark)=(arg(1,500.),arg(2,1024.) as usize,arg(3,1.) as u64,arg(4,0.)>0.,arg(5,64.) as usize,arg(6,0.2),arg(7,0.)>0.);
    let reference:Value=serde_json::from_str(&std::fs::read_to_string(REFERENCE).unwrap()).unwrap();
    let rows=reference["rows"].as_array().unwrap();
    let mut request=scene::turned(scene::load(scene::PGC));
    if swap{for beam in &mut request.beams{beam.direction.swap(0,1);}request.emission.swap(0,1);}
    // dark: the tweezer alone, a stationary thermal state.
    if dark{request.beams.retain(|b|b.waist_um.is_some());}
    // DIAGNOSTIC: free space -- the tweezer removed, the plane waves kept (a periodic box).
    let free=std::env::var("ATOM_GPU_FREE_SPACE").is_ok();
    if free{request.beams.retain(|b|b.waist_um.is_none());}
    let mut model=Model::new(&request).unwrap();
    let index=request.states.iter().position(|s|s=="ground-2-2").unwrap();
    let b22=model.ground.ground_states.iter().position(|&i|i==index).unwrap();
    let omega=model.trap_frequency(b22,0).max(1e-9);
    // The grid balanced for the trap: half the box and the largest momentum the same number of
    // thermal widths, (L/2)/x_T = κ_max/κ_T, so dx² = 2π ħ/(N m ω).
    let dx=if free{0.0075378}else{(std::f64::consts::TAU*model.hbar_over_mass/(n as f64*omega)).sqrt()};
    let grid=Grid{nx:n,ny:n,dx,dy:dx};
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
    let period=std::f64::consts::TAU/omega;
    let settings=Settings{grid,dt,period,trajectories:b,seed,cap:None,scene:0};
    let start=std::time::Instant::now();
    let mut engine=Engine::new(std::rc::Rc::new(Gpu::new().unwrap()),&model,settings).unwrap();
    let thermal=|t_uk:f64|((t_uk*1e-6*atom_motion_gpu::model::BOLTZMANN/(atom_motion_gpu::model::HBAR*omega*1e6))*model.hbar_over_mass/omega).sqrt();
    println!("{n} x {n}, dx {:.2} nm, box ±{:.0} nm ({:.1} x_T at 30 uK), k_max {:.1} hbar k, dt {dt}, o1 kick every {period:.2} us, {b} trajectories, swap {swap}; tables {:.2} s",
        dx*1e3,grid.half_width(0)*1e3,grid.half_width(0)/thermal(30.),std::f64::consts::PI/dx/8.0528,start.elapsed().as_secs_f64());
    let c22=model.block_states[1].iter().position(|&s|s==b22).unwrap();
    let mut amplitudes=vec![Complex::ZERO;5];amplitudes[c22]=Complex::ONE;
    engine.initialize(&vec![Internal{block:1,amplitudes};b],30.,&model).unwrap();
    let(along,across)=if swap{(1,0)}else{(0,1)};
    let interval_us=50.;let interval=(interval_us/dt).round() as usize;let outputs=(t_end/interval_us).round() as usize;
    let report=|engine:&Engine,t:f64|{
        let(records,components)=engine.read_records().unwrap();
        let s=sample(&model,&records,&components,&engine.trajectory_weights);
        let row=rows.iter().min_by(|a,b|(a["t_us"].as_f64().unwrap()-t).abs().total_cmp(&(b["t_us"].as_f64().unwrap()-t).abs())).unwrap();
        let rp:Vec<f64>=row["pops_ground"].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect();
        let pf1:f64=g.ground_states[..3].iter().map(|&i|s.populations[i]).sum();
        let worst=(0..8).map(|k|{let i=g.ground_states[k];((s.populations[i]-rp[k])/s.population_errors[i].max(1e-6)).abs()}).fold(0.,f64::max);
        let nb=|d:usize|occupation(&model,s.momentum_square[d],s.position_variance[d],omega);
        let tr=row["T_uK"].as_f64().unwrap();
        println!("t {:5.0}  T_along {:6.2}±{:4.2} ({:6.2}, {:+5.1}%)  T_across {:6.2}±{:4.2}  nbar {:5.2} ({:5.2}) / {:5.2}  loss {:.1e}  pF1 {:.4} ({:.4})  pops worst {:.1} SE  tails x {:.1e} {:.1e} k {:.1e} {:.1e}",
            t,s.temperature[along],s.temperature_errors[along],tr,100.*(s.temperature[along]/tr-1.),s.temperature[across],s.temperature_errors[across],
            nb(along),row["nbar"].as_f64().unwrap(),nb(across),s.loss,pf1,row["pF1"].as_f64().unwrap(),worst,s.tails[0],s.tails[1],s.tails[2],s.tails[3]);
    };
    engine.observe_initial(false).unwrap();report(&engine,0.);
    if std::env::var("PROFILE").is_ok(){engine.set_profiling(true).unwrap();}
    let run=std::time::Instant::now();
    for o in 1..=outputs{for s in 0..interval{engine.step(((o-1)*interval+s) as u64,s+1==interval,false).unwrap();}report(&engine,o as f64*interval_us);}
    for(phase,ms)in engine.profile(){println!("  {phase}: {:.3} ms per step",ms/(outputs*interval) as f64);}
    println!("{} steps: {:.1} s ({:.3} us per trajectory-step)",outputs*interval,run.elapsed().as_secs_f64(),run.elapsed().as_secs_f64()*1e6/(outputs*interval*b) as f64);
}
