//! The GPU kernel against the CPU engine (needs an NVIDIA GPU; without one each check says so and
//! passes): the structs' layouts, the random streams, one step, and a jump with the first-order and
//! the exact response, on the app's tweezer PGC example (tests/fixtures/tweezer_pgc.json: clusters of
//! 3 and 5 states) and on a D1 gray molasses (tests/fixtures/tweezer_gray_molasses.json: one cluster
//! of all 8 ground states, the kernel's largest).
use atom_motion_3d::engine::{Engine,Settings,Trajectory};
use atom_motion_3d::gpu::{self,driver::Buffer,program};
use atom_motion_3d::model::Model;
use atom_motion_3d::rng::{Purpose,Rng};
#[path="../examples/scene/mod.rs"] mod scene;
use atom_motion_3d::control::{Harmonic,Twin};
use atom_motion_3d::stats::{summarize,Means,Scales};
use atom_motion_3d::{initial_trajectory,Beam,Complex,Initial,Matrix,Request,Stratum,Variates,STATES};
use serde_json::Value;

fn matrix(v:&Value)->Matrix{
    let mut m=Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]};
    for e in v.as_array().unwrap(){let(i,j)=(e[0].as_u64().unwrap() as usize,e[1].as_u64().unwrap() as usize);m.real[i*STATES+j]=e[2].as_f64().unwrap();m.imag[i*STATES+j]=e[3].as_f64().unwrap();}
    m
}
fn floats(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
const PGC:&str=include_str!("fixtures/tweezer_pgc.json");
const MOLASSES:&str=include_str!("fixtures/tweezer_gray_molasses.json");
/// A fixture's request (tests/engine.rs `request`).
fn request(fixture:&str)->Request{
    let fixture:Value=serde_json::from_str(fixture).unwrap();let r=&fixture["request"];
    let states:Vec<String>=r["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect();
    let initial=states.iter().position(|s|s==r["initialStateId"].as_str().unwrap()).unwrap();
    let mut density=vec![Complex::ZERO;STATES*STATES];density[initial*STATES+initial]=Complex::ONE;
    let beams=r["beams"].as_array().unwrap().iter().map(|b|{let d=floats(&b["direction"]);Beam{id:b["id"].as_str().unwrap().into(),name:b["name"].as_str().unwrap().into(),
        direction:[d[0],d[1],d[2]],wave_number_per_um:b["waveNumberPerUm"].as_f64().unwrap(),waist_um:b["waistUm"].as_f64(),intensity_mw_cm2:b["intensityMWcm2"].as_f64().unwrap(),
        frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),coupling:matrix(&b["coupling"]),
        counter_coupling:matrix(&b["counterCoupling"]),outside_shift:matrix(&b["outsideShift"])}}).collect::<Vec<Beam>>();
    let trap_beam=beams.iter().position(|b|b.waist_um.is_some()).unwrap_or(0);
    let e=r["emission"].as_array().unwrap();
    Request{states,hamiltonian:matrix(&r["hamiltonian"]),frame_frequencies_mhz:floats(&r["frameFrequenciesMHz"]),frequencies_mhz:floats(&r["frequenciesMHz"]),beams,
        emission:[matrix(&e[0]),matrix(&e[1]),matrix(&e[2])],field_vector_g:[0.;3],mass_kg:r["massKg"].as_f64().unwrap(),initial_density:density,
        temperature_uk:r["temperatureUK"].as_f64().unwrap(),trap_beam,times_us:vec![]}
}
/// The GPU, or None (said) on a machine without one.
fn device()->Option<&'static std::sync::Mutex<gpu::Gpu>>{match gpu::gpu(){Ok(g)=>Some(g),Err(e)=>{println!("no GPU ({e}): nothing to check");None}}}

/// The kernel's structs have their Rust twins' sizes and offsets.
#[test]
fn the_kernel_s_structs_match_their_twins(){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let f=gpu.function("layout").unwrap();let out=Buffer::new(128).unwrap();gpu.stream().launch(&f,1,32,&[&out.pointer]).unwrap();
    let mut words=[0u32;24];gpu.stream().download(&out,&mut words).unwrap();
    assert_eq!(words,program::layout());
}

/// The kernel's streams are rng.rs's: the same words, deviates and directions.
#[test]
fn the_kernel_s_streams_are_the_cpu_s(){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());let f=gpu.function("rng_stream").unwrap();
    let(words,deviates)=(Buffer::new(40).unwrap(),Buffer::new(48).unwrap());
    for(seed,purpose,trajectory,event)in [(0x3D_5EED_u64,Purpose::Jump,0u64,0u32),(u64::MAX-3,Purpose::Jump,123_456_789_012,77),(42,Purpose::Initial,5,0)]{
        gpu.stream().launch(&f,1,32,&[&seed,&(purpose as u32),&trajectory,&event,&words.pointer,&deviates.pointer]).unwrap();
        let(mut w,mut d)=([0u32;10],[0f64;6]);gpu.stream().download(&words,&mut w).unwrap();gpu.stream().download(&deviates,&mut d).unwrap();
        let mut a=Rng::new(seed,purpose,trajectory,event as u64);assert_eq!(w,std::array::from_fn(|_|a.next_u32()));
        let mut b=Rng::new(seed,purpose,trajectory,event as u64);let(u,o,n,v)=(b.uniform(),b.open_uniform(),b.direction(),b.uniform());
        assert_eq!([d[0],d[1],d[5]],[u,o,v]);for i in 0..3{assert!((d[2+i]-n[i]).abs()<=1e-15,"{:?} against {n:?}",&d[2..5]);}
    }
}

/// A batch of the fixture's thermal trajectories and the step's settings (a solve's: the truncation
/// a share of the step's budget, the loss on), with `threshold` (when given) as every one's.
fn setup(fixture:&str,threshold:Option<f64>,allowance:f64)->(Model,Settings,Vec<Trajectory>){
    let r=request(fixture);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let mut settings=Settings::new(0.2,&initial.traps[0],1e-6,false);settings.doppler=allowance;
    let trajectories=(0..64).map(|index|{let mut t=initial_trajectory(&model,&settings,&initial,0x3D_5EED,index,Stratum::single(index));if let Some(x)=threshold{t.threshold=x;}t}).collect();
    (model,settings,trajectories)
}
/// The CPU's and the GPU's trajectories after one step from the same states agree to single
/// precision: the same jumps (the same choices), positions, momenta, internal states and norms.
fn compare(fixture:&str,threshold:Option<f64>,allowance:f64,jumps:bool){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let(model,settings,trajectories)=setup(fixture,threshold,allowance);
    let program=gpu.program(&model,&settings).unwrap();
    let got=gpu.advance(&program,&settings,&trajectories,1).unwrap();
    let engine=Engine::new(&model,settings.clone());
    let mut jumped=0;
    for(start,g)in trajectories.iter().zip(&got){
        let mut c=start.clone();engine.step(&mut c);
        let at=format!("trajectory {} (cluster {} → {}, jumps {})",start.index,start.cluster,c.cluster,c.jumps);
        assert_eq!((g.cluster,g.jumps,g.alive),(c.cluster,c.jumps,c.alive),"{at}");jumped+=(c.jumps>0) as usize;
        assert_eq!(g.threshold,c.threshold,"{at}: the new threshold is the stream's");
        let scale=c.kappa.iter().map(|x|x.abs()).fold(1.,f64::max);
        for a in 0..3{
            assert!((g.r[a]-c.r[a]).abs()<=2e-6*c.r.iter().map(|x|x.abs()).fold(1e-3,f64::max),"{at}: r {:?} against {:?}",g.r,c.r);
            assert!((g.kappa[a]-c.kappa[a]).abs()<=2e-5*scale,"{at}: κ {:?} against {:?}",g.kappa,c.kappa);
        }
        let n=model.dimension(c.cluster);let(gn,cn)=(g.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>(),c.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>());
        // Each jump redoes part of the step from a crossing found in single precision.
        assert!((gn.ln()-cn.ln()).abs()<=2e-5*(1+c.jumps) as f64,"{at}: ln‖ψ‖² {} against {}",gn.ln(),cn.ln());
        for x in 0..n{let(a,b)=(g.psi[x]*(1./gn.sqrt()),c.psi[x]*(1./cn.sqrt()));assert!((a-b).abs()<=2e-5,"{at}: ψ[{x}] {a:?} against {b:?}");}
    }
    if jumps{assert_eq!(jumped,trajectories.len(),"every trajectory jumps in the step");}
    println!("{} trajectories, {jumped} jumped",trajectories.len());
}

/// One step from thermal states, the jumps where their thresholds fall.
#[test]
fn a_step_is_the_cpu_s(){compare(PGC,None,f64::INFINITY,false);}
/// A jump in the first step (every threshold just below one): the class, the photon's direction and
/// polarization, the recoils and the new threshold are the CPU's.
#[test]
fn a_jump_is_the_cpu_s(){compare(PGC,Some(0.999),f64::INFINITY,true);}
/// The same with the exact response (an allowance of zero: every step beyond the first order).
#[test]
fn a_jump_with_the_exact_response_is_the_cpu_s(){compare(PGC,Some(0.999),0.,true);}
/// On a cluster of all 8 ground states (two entries of each matrix per lane): a step, and a jump with
/// the first-order and the exact response.
#[test]
fn on_eight_states_a_step_and_a_jump_are_the_cpu_s(){
    compare(MOLASSES,None,f64::INFINITY,false);compare(MOLASSES,Some(0.999),f64::INFINITY,true);compare(MOLASSES,Some(0.999),0.,true);
}

/// A batch on the GPU sums what the same batch sums on the CPU (lib.rs `simulate`), to single
/// precision, with the control variates (control.rs): 20 trajectories in two groups -- tasks of 8, 2,
/// 8 and 2 (blocks with idle warps) --, 3 steps between 6 outputs, the motion's means over each
/// output's window of half a trap period (the windows closing up to the reach past the last output,
/// over several launches), the samples kept at two outputs, every trajectory's harmonic control and,
/// where the generator is static, its held twin.
fn batch(fixture:&str)->bool{
    let Some(gpu)=device()else{return false};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let r=request(fixture);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let settings=Settings::new(0.2,&initial.traps[0],1e-6,true);
    let(seed,first,count,group,steps,outputs,listed)=(0x3D_5EED,40,20,10,3,6,[2,5]);
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);
    let twin=Twin::new(&model,0.5*model.geometry[model.trap_beam].intensity([0.;3]));
    let variates=Variates{harmonic:&harmonic,twin:twin.as_ref()};
    let program=gpu.program(&model,&settings).unwrap();
    let g=gpu.simulate(&program,&model,&settings,&initial,seed,first,count,group,steps,outputs,&listed,Some(&variates)).unwrap();
    let c=atom_motion_3d::simulate(&model,&settings,&initial,seed,first,count,group,steps,outputs,&listed,Some(&variates));
    assert_eq!(g.groups.len(),c.groups.len());
    let close=|a:f64,b:f64,scale:f64,what:&str|assert!((a-b).abs()<=1e-3*b.abs().max(scale),"{what}: {a} against {b}");
    for(k,(gg,cg))in g.groups.iter().zip(&c.groups).enumerate(){for(i,(x,y))in gg.sums.iter().zip(&cg.sums).enumerate(){
        let at=format!("group {k}, output {i}");
        assert_eq!((x.trajectories,x.held),(y.trajectories,y.held),"{at}: trajectories and held");
        for a in 0..3{
            close(x.kappa[a],y.kappa[a],(y.kappa2[a]*y.held).sqrt(),&format!("{at}: Σκ[{a}]"));close(x.kappa2[a],y.kappa2[a],0.,&format!("{at}: Σκ²[{a}]"));
            close(x.position[a],y.position[a],(y.position2[a]*y.held).sqrt(),&format!("{at}: Σr[{a}]"));close(x.position2[a],y.position2[a],0.,&format!("{at}: Σr²[{a}]"));
            for j in 0..3{close(x.harmonic[a][j],y.harmonic[a][j],1e-9*y.harmonic[a][1].sqrt(),&format!("{at}: harmonic[{a}][{j}]"));}
        }
        for s in 0..STATES{close(x.populations[s],y.populations[s],y.held,&format!("{at}: Σp[{s}]"));close(x.twin[s],y.twin[s],y.trajectories,&format!("{at}: twin Σc[{s}]"));}
        for j in 0..2{close(x.twin2[j],y.twin2[j],y.trajectories,&format!("{at}: twin2[{j}]"));}
        close(x.profile,y.profile,1e-6*y.held,&format!("{at}: profile"));close(x.doppler,y.doppler,1e-9*y.held,&format!("{at}: (kv/σ)²"));
    }}
    assert_eq!(g.jumps,c.jumps,"jumps");
    for(gs,cs)in g.samples.iter().zip(&c.samples){
        assert_eq!(gs.len(),cs.len());
        for(x,y)in gs.iter().zip(cs){for a in 0..3{close(x.0[a],y.0[a],1e-3,"a sample's r");close(x.1[a],y.1[a],1.,"a sample's κ");}}
    }
    println!("{} groups of {} outputs, {} jumps, held twin {}",g.groups.len(),outputs,g.jumps,twin.is_some());
    twin.is_some()
}
/// The batches of the tweezer PGC and the gray molasses; one of them has a held twin.
#[test]
fn a_batch_s_sums_are_the_cpu_s(){
    let twins=[batch(PGC),batch(MOLASSES)];
    if device().is_some(){assert!(twins.iter().any(|&t|t),"a fixture with a held twin");}
}

/// The default scene's motion as the app sends it (rust-solver/tests/fixtures/default_scene_motion.json)
/// with the held twin kept: a GPU batch's estimates, the twin's control correcting the populations,
/// have rows that sum to one within 1e-9 -- the atom's and the twin's readouts in double precision
/// (the adapter checks 1e-7).
#[test]
fn the_default_scene_s_rows_sum_to_one_with_the_held_twin(){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let fixture:Value=serde_json::from_str(include_str!("../../tests/fixtures/default_scene_motion.json")).unwrap();
    let r=scene::request(&fixture["payload"]);let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();let trap=initial.traps[reference];
    let(outputs,steps,interval)=(101,10,1.);
    let settings=Settings::new(interval/steps as f64,&trap,1e-7,true);
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);
    let twin=Twin::new(&model,0.5*model.geometry[model.trap_beam].intensity([0.;3])).expect("the default scene's generator is static");
    let means:Vec<Means>=twin.means(&model,&r.initial_density,interval,outputs).into_iter().enumerate().map(|(i,m)|Means{harmonic:harmonic.mean[i],twin:Some(m)}).collect();
    let program=gpu.program(&model,&settings).unwrap();
    let out=gpu.simulate(&program,&model,&settings,&initial,0x3D_5EED,0,256,32,steps,outputs,&[],Some(&Variates{harmonic:&harmonic,twin:Some(&twin)})).unwrap();
    let estimates=summarize(&out.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab},&means);
    let worst=estimates.iter().filter(|e|e.loss<1.).map(|e|(e.populations.iter().sum::<f64>()-1.).abs()).fold(0.,f64::max);
    println!("the rows' largest |sum - 1|: {worst:.2e}");
    assert!(worst<=1e-9,"a row's sum is off by {worst:.2e}");
}
