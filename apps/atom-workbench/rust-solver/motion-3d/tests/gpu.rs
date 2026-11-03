//! The GPU kernel against the CPU engine (needs an NVIDIA GPU; without one each check says so and
//! passes): the structs' layouts, the random streams, one step, and a jump with the first-order and
//! the exact response, on the app's tweezer PGC example (tests/fixtures/tweezer_pgc.json).
use atom_motion_3d::engine::{Engine,Settings,Trajectory};
use atom_motion_3d::gpu::{self,driver::Buffer,program};
use atom_motion_3d::model::Model;
use atom_motion_3d::rng::{Purpose,Rng};
use atom_motion_3d::{initial_trajectory,Beam,Complex,Initial,Matrix,Request,Stratum,STATES};
use serde_json::Value;

fn matrix(v:&Value)->Matrix{
    let mut m=Matrix{real:vec![0.;STATES*STATES],imag:vec![0.;STATES*STATES]};
    for e in v.as_array().unwrap(){let(i,j)=(e[0].as_u64().unwrap() as usize,e[1].as_u64().unwrap() as usize);m.real[i*STATES+j]=e[2].as_f64().unwrap();m.imag[i*STATES+j]=e[3].as_f64().unwrap();}
    m
}
fn floats(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
/// The fixture's request (tests/engine.rs `request`).
fn request()->Request{
    let fixture:Value=serde_json::from_str(include_str!("fixtures/tweezer_pgc.json")).unwrap();let r=&fixture["request"];
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
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap();
    let f=gpu.function("layout").unwrap();let out=Buffer::new(64).unwrap();gpu.stream().launch(&f,1,32,&[&out.pointer]).unwrap();
    let mut words=[0u32;16];gpu.stream().download(&out,&mut words).unwrap();
    assert_eq!(words,program::layout());
}

/// The kernel's streams are rng.rs's: the same words, deviates and directions.
#[test]
fn the_kernel_s_streams_are_the_cpu_s(){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap();let f=gpu.function("rng_stream").unwrap();
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
fn setup(threshold:Option<f64>,allowance:f64)->(Model,Settings,Vec<Trajectory>){
    let r=request();let model=Model::new(&r).unwrap();let initial=Initial::new(&model,&r).unwrap();
    let mut settings=Settings::new(0.2,&initial.traps[0],1e-6,false);settings.doppler=allowance;
    let trajectories=(0..64).map(|index|{let mut t=initial_trajectory(&model,&settings,&initial,0x3D_5EED,index,Stratum::single(index));if let Some(x)=threshold{t.threshold=x;}t}).collect();
    (model,settings,trajectories)
}
/// The CPU's and the GPU's trajectories after one step from the same states agree to single
/// precision: the same jumps (the same choices), positions, momenta, internal states and norms.
fn compare(threshold:Option<f64>,allowance:f64,jumps:bool){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap();
    let(model,settings,trajectories)=setup(threshold,allowance);
    let program=gpu.program(&model,&settings).unwrap();
    let got=gpu.advance(&program,&settings,&trajectories,1).unwrap();
    let engine=Engine{model:&model,settings:settings.clone()};
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
        assert!((gn.ln()-cn.ln()).abs()<=2e-5,"{at}: ln‖ψ‖² {} against {}",gn.ln(),cn.ln());
        for x in 0..n{let(a,b)=(g.psi[x]*(1./gn.sqrt()),c.psi[x]*(1./cn.sqrt()));assert!((a-b).abs()<=2e-5,"{at}: ψ[{x}] {a:?} against {b:?}");}
    }
    if jumps{assert_eq!(jumped,trajectories.len(),"every trajectory jumps in the step");}
    println!("{} trajectories, {jumped} jumped",trajectories.len());
}

/// One step from thermal states, the jumps where their thresholds fall.
#[test]
fn a_step_is_the_cpu_s(){compare(None,f64::INFINITY,false);}
/// A jump in the first step (every threshold just below one): the class, the photon's direction and
/// polarization, the recoils and the new threshold are the CPU's.
#[test]
fn a_jump_is_the_cpu_s(){compare(Some(0.999),f64::INFINITY,true);}
/// The same with the exact response (an allowance of zero: every step beyond the first order).
#[test]
fn a_jump_with_the_exact_response_is_the_cpu_s(){compare(Some(0.999),0.,true);}
