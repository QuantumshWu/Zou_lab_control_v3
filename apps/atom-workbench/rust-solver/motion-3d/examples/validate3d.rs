//! 3D validations of the semiclassical engine (all three axes free; frame axes 0, 1, 2 are lab y, z, x
//! for the exported scenes, the tweezer along lab z):
//! - `off`: the tweezer alone; the thermal state stays stationary (T and n̄ per axis).
//! - `one`: the tweezer and one circularly polarized beam along x (cycling, Doppler term off): the
//!   heating rate per axis against the recoil formula (ħk)²/(2m k_B) R (δ_ax + ⟨n_a²⟩), R from the
//!   jumps, ⟨n_x²⟩ = 2/5 and ⟨n_y²⟩ = ⟨n_z²⟩ = 3/10 for σ emission about x.
//! - `line`: TW1 or TW2 (beams along ±x) in 3D: x against the 1D reference, y and z against the
//!   emission-only heating with R from the jumps.
//! `cargo run --profile iterate -p atom-motion-3d --example validate3d -- <off|one|line> [scene] [trajectories] [t_end_us] [dt]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::engine::Settings;
use atom_motion_3d::model::Model;
use atom_motion_3d::stats::{summarize,Scales,KT_PER_UK};
use atom_motion_3d::{simulate,Initial};
use serde_json::Value;

/// The least-squares slope of y against x.
fn slope(x:&[f64],y:&[f64])->f64{let n=x.len() as f64;let(mx,my)=(x.iter().sum::<f64>()/n,y.iter().sum::<f64>()/n);
    x.iter().zip(y).map(|(a,b)|(a-mx)*(b-my)).sum::<f64>()/x.iter().map(|a|(a-mx).powi(2)).sum::<f64>()}

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:&str|args.get(i).cloned().unwrap_or(d.into());
    let kind=arg(1,"off");let name=arg(2,match kind.as_str(){"off"=>"OFF","one"=>"ONE",_=>"TW2"});
    let(count,t_end,dt):(u64,f64,f64)=(arg(3,"4096").parse().unwrap(),arg(4,"1000").parse().unwrap(),arg(5,"0.25").parse().unwrap());
    let request=scene::scene(&name);
    let mut model=Model::new(&request).unwrap();model.doppler=kind!="one";
    let index=request.states.iter().position(|s|s=="ground-2-2").unwrap();let b22=model.ground.ground_states.iter().position(|&i|i==index).unwrap();
    let(block22,state22)={let k=model.ground.cluster[b22];(k,model.ground.clusters[k].position(b22).unwrap())};
    let trap=model.trap(block22,state22);let frequencies=trap.lab;
    println!("{name}: trap {:?} kHz (frame axes: lab y, z, x), table nodes error {:.1e}",frequencies.map(|f|f/std::f64::consts::TAU*1e3),model.table_error);
    // Optional: the free axes (a mask "yzx" of 0/1) and the harmonic flow (flow|drift).
    let mask:Vec<bool>=arg(6,"111").chars().map(|c|c=='1').collect();let free=[mask[0],mask[1],mask[2]];
    let flow=arg(7,"flow")=="flow";
    let mut settings=Settings::new(dt,&trap,f64::EPSILON,false);settings.free=free;if !flow{settings.lab=[0.;3];settings.frequencies=[0.;3];}
    let initial=Initial::new(&model,&request).unwrap();
    let interval=10.;let steps=(interval/dt).round() as usize;let outputs=(t_end/interval).round() as usize+1;
    let clock=std::time::Instant::now();
    // (r, κ) of every held atom at every tenth output: the mechanical energy.
    let listed:Vec<usize>=(0..outputs).step_by((outputs/10).max(1)).collect();
    let outcome=simulate(&model,&settings,&initial,7,0,count,8,steps,outputs,&listed);
    let wall=clock.elapsed().as_secs_f64();
    let e=summarize(&outcome.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies});
    let lab=["y","z","x"];
    let every=(outputs/10).max(1);
    for(i,x)in e.iter().enumerate().step_by(every){
        println!("t {:6.0}  T(y,z,x) {:6.2}±{:4.2} {:6.2}±{:4.2} {:6.2}±{:4.2}  nbar {:5.2} {:5.2} {:5.2}  loss {:.1e}",i as f64*interval,
            x.temperature[0],x.temperature_error[0],x.temperature[1],x.temperature_error[1],x.temperature[2],x.temperature_error[2],x.occupation[0],x.occupation[1],x.occupation[2],x.loss);
    }
    let rate=outcome.jumps as f64/count as f64/t_end;
    println!("{count} trajectories, dt {dt}, {wall:.1} s wall; scattering rate {rate:.4}/us");
    let times:Vec<f64>=(0..outputs).map(|i|i as f64*interval).collect();
    let recoil=|k:f64|(k*k)*model.hbar_over_mass/2./KT_PER_UK;
    match kind.as_str(){
        "off"=>{for a in 0..3{let t:Vec<f64>=e.iter().map(|x|x.temperature[a]).collect();let mean=t.iter().sum::<f64>()/t.len() as f64;
            println!("{} axis: T mean {mean:.3} uK, drift {:+.4} uK/ms (T0 {:.3})",lab[a],slope(&times,&t)*1e3,t[0]);}}
        "one"=>{
            let k=model.carriers[0].wave_number;let c=recoil(k);
            // The mechanical energy per atom (kinetic plus the reference state's trap potential, μK): it
            // rises by 2 E_r per scattering whatever the trap's anharmonicity (the absorbed ħk along x and
            // the emitted photon, ⟨n²⟩ = 1 summed over the axes).
            let(block22,state22)={let k=model.ground.cluster[b22];(k,model.ground.clusters[k].position(b22).unwrap())};
            let energy:Vec<f64>=outcome.samples.iter().map(|atoms|atoms.iter().map(|(r,kappa)|
                (0..3).map(|a|model.hbar_over_mass*kappa[a]*kappa[a]/2.).sum::<f64>()+model.potential(block22,state22,*r)).sum::<f64>()/atoms.len() as f64/KT_PER_UK).collect();
            let at:Vec<f64>=listed.iter().map(|&i|times[i]).collect();
            println!("mechanical energy: dE/dt {:.5} uK/us, 2 E_r R {:.5} uK/us ({:+.1}%)",slope(&at,&energy),2.*c*rate,100.*(slope(&at,&energy)/(2.*c*rate)-1.));
            for(a,extra)in[(0,0.3),(1,0.3),(2,1.4)]{let t:Vec<f64>=e.iter().map(|x|x.temperature[a]).collect();let measured=slope(&times,&t);let expected=c*rate*extra;
                println!("{} axis: dT/dt {:.5} uK/us, recoil formula {:.5} uK/us ({:+.1}%)",lab[a],measured,expected,100.*(measured/expected-1.));}
        }
        _=>{
            let reference:Value=scene::reference(&name);
            let rows=reference["rows"].as_array().unwrap();
            for(i,x)in e.iter().enumerate().step_by(every){let t=i as f64*interval;
                let row=rows.iter().min_by(|a,b|(a["t_us"].as_f64().unwrap()-t).abs().total_cmp(&(b["t_us"].as_f64().unwrap()-t).abs())).unwrap();
                println!("t {:6.0}  T_x {:6.2} ref {:6.2} ({:+5.1}%)",t,x.temperature[2],row["T_uK"].as_f64().unwrap(),100.*(x.temperature[2]/row["T_uK"].as_f64().unwrap()-1.));}
            // y and z: heated by the emitted photons alone, (ħk)²/(2m k_B) R ⟨n_a²⟩ with the pattern's ⟨n_a²⟩ = 1/3 on average.
            let k=model.carriers[0].wave_number;let c=recoil(k);
            for a in 0..2{let t:Vec<f64>=e.iter().map(|x|x.temperature[a]).collect();println!("{} axis: dT/dt {:.5} uK/us; emission alone with <n^2> = 1/3: {:.5} uK/us",lab[a],slope(&times,&t),c*rate/3.);}
        }
    }
}
