//! The GPU kernel against the CPU engine (needs an NVIDIA GPU; without one each check says so and
//! passes): the structs' layouts, the random streams, one step, and a jump with the first-order and
//! the exact response, on the app's tweezer PGC example (tests/fixtures/tweezer_pgc.json: clusters of
//! 3 and 5 states) and on a D1 gray molasses (tests/fixtures/tweezer_gray_molasses.json: one cluster
//! of all 8 ground states, the kernel's largest).
use atom_motion_3d::engine::{Engine,Settings,Trajectory};
use atom_motion_3d::gpu::{self,driver::Buffer,program};
use atom_motion_3d::ground::Ground;
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
/// The tweezer PGC with the tweezer's reflection along its axis, in phase at the focus (F5, ground.rs
/// `Unit`): the two one unit, their standing wave's intensity |a₁ + a₂|²/4 an axis of the tables, both
/// the trap's beams (its far level with the two off).
fn retro()->Request{
    let mut r=request(PGC);let mut reflection=r.beams[r.trap_beam].clone();
    (reflection.id,reflection.name)=("reflection".into(),"反射光".into());reflection.direction=reflection.direction.map(|x|-x);r.beams.push(reflection);r
}
/// The GPU, or None (said) on a machine without one.
fn device()->Option<&'static std::sync::Mutex<gpu::Gpu>>{match gpu::gpu(){Ok(g)=>Some(g),Err(e)=>{println!("no GPU ({e}): nothing to check");None}}}

/// The kernel's structs have their Rust twins' sizes and offsets.
#[test]
fn the_kernel_s_structs_match_their_twins(){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let f=gpu.function("layout").unwrap();let out=Buffer::new(128).unwrap();gpu.stream().launch(&f,1,32,&[&out.pointer]).unwrap();
    let mut words=[0u32;25];gpu.stream().download(&out,&mut words).unwrap();
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

/// A fixture's model, folded at fourth order where `fourth` (model.rs # Fourth order).
fn model(r:&Request,fourth:bool)->Model{Model::with_order(r,f64::INFINITY,fourth).unwrap()}
/// A request's model at its fold ladder's first rung with explicit excited states (ground.rs
/// `Ground::allowances`): its clusters take the exact step (engine.rs # The exact step).
fn explicit(r:&Request)->Model{Model::with(r,Ground::allowances(r).unwrap()[1]).unwrap()}
/// The PGC with a resonant probe 10 MHz off the cooling light (the floquet study's PGCP_10), as the app
/// sends it: its folds' saturation far beyond one, the solve keeps 7 excited states explicit with the 5
/// ground states of F = 2, a cluster beating at 36.4 and 72.8 MHz that its exact step takes by CF4.
fn probed()->Request{let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=vec![0.,1.];r}
/// A batch of the fixture's thermal trajectories and the step's settings (a solve's: the truncation
/// a share of the step's budget, the loss on), with `threshold` (when given) as every one's.
fn setup(r:&Request,model:Model,dt:f64,threshold:Option<f64>,allowance:f64)->(Model,Settings,Vec<Trajectory>){
    let initial=Initial::new(&model,r).unwrap();
    // The exact step's beats left out within a solve's share over a solve's horizon (engine.rs `Settings`:
    // η² ε dt/waiting, a waiting time of microseconds): a tweezer's hyperfine beat (6.6 GHz, 1e-9 of the
    // generator) is left out as a solve leaves it -- kept, its 13,000 single-precision CF4 spacings a step
    // drift ln‖ψ‖² by 1e-4, the CPU's cached propagators by none.
    let mut settings=Settings::new(dt,&initial.traps[0],1e-6,false);settings.doppler=allowance;settings.beats=1e-4;settings.horizon=10.;
    let trajectories=(0..64).map(|index|{let mut t=initial_trajectory(&model,&settings,&initial,0x3D_5EED,index,Stratum::single(index));if let Some(x)=threshold{t.threshold=x;}t}).collect();
    (model,settings,trajectories)
}
/// `compare_model` on the fixture's model, folded at fourth order where `fourth`, at a step of 0.2 us.
fn compare(r:&Request,threshold:Option<f64>,allowance:f64,jumps:bool,fourth:bool){compare_model(r,model(r,fourth),0.2,threshold,allowance,jumps);}
/// The CPU's and the GPU's trajectories after one step from the same states agree to single
/// precision: the same jumps (the same choices), positions, momenta, internal states and norms; the
/// new threshold the stream's (at fourth order divided by the slow start's A, single precision). The
/// exact step (explicit states, beats) takes ψ through hundreds of exponentials and a static cluster's
/// propagator as many times, in single precision -- its ψ, norm and momentum to 5e-4, 5e-4 and 5e-5.
fn compare_model(r:&Request,model:Model,dt:f64,threshold:Option<f64>,allowance:f64,jumps:bool){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let fourth=model.fourth;let exact=(0..model.clusters()).any(|k|model.mixed(k)||model.beating(k));let(tp,tn,tk)=if exact{(5e-4,5e-4,5e-5)}else{(2e-5,2e-5,2e-5)};
    let(model,settings,trajectories)=setup(r,model,dt,threshold,allowance);
    let program=gpu.program(&model,&settings).unwrap();
    let got=gpu.advance(&program,&settings,&trajectories,1).unwrap();
    let engine=Engine::new(&model,settings.clone());
    let mut jumped=0;
    for(start,g)in trajectories.iter().zip(&got){
        let mut c=start.clone();engine.step(&mut c);
        let at=format!("trajectory {} (cluster {} → {}, jumps {})",start.index,start.cluster,c.cluster,c.jumps);
        assert_eq!((g.cluster,g.jumps,g.alive),(c.cluster,c.jumps,c.alive),"{at}");jumped+=(c.jumps>0) as usize;
        assert!((g.threshold-c.threshold).abs()<=if fourth{1e-5*c.threshold}else{0.},"{at}: the new threshold {} against {}",g.threshold,c.threshold);
        let scale=c.kappa.iter().map(|x|x.abs()).fold(1.,f64::max);
        for a in 0..3{
            // A momentum's difference moves the position over the step's flow by (ħ/m) dt Δκ.
            assert!((g.r[a]-c.r[a]).abs()<=2e-6*c.r.iter().map(|x|x.abs()).fold(1e-3,f64::max)+if exact{model.hbar_over_mass*dt*tk*scale}else{0.},"{at}: r {:?} against {:?}, κ {:?} against {:?}",g.r,c.r,g.kappa,c.kappa);
            assert!((g.kappa[a]-c.kappa[a]).abs()<=tk*scale,"{at}: κ {:?} against {:?}",g.kappa,c.kappa);
        }
        let n=model.dimension(c.cluster);let(gn,cn)=(g.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>(),c.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>());
        // Each jump redoes part of the step from a crossing found in single precision.
        assert!((gn.ln()-cn.ln()).abs()<=tn*(1+c.jumps) as f64,"{at}: ln‖ψ‖² {} against {}",gn.ln(),cn.ln());
        for x in 0..n{let(a,b)=(g.psi[x]*(1./gn.sqrt()),c.psi[x]*(1./cn.sqrt()));assert!((a-b).abs()<=tp,"{at}: ψ[{x}] {a:?} against {b:?}");}
    }
    if jumps{assert_eq!(jumped,trajectories.len(),"every trajectory jumps in the step");}
    println!("{} trajectories, {jumped} jumped",trajectories.len());
}

/// One step from thermal states, the jumps where their thresholds fall.
#[test]
fn a_step_is_the_cpu_s(){compare(&request(PGC),None,f64::INFINITY,false,false);}
/// A jump in the first step (every threshold just below one): the class, the photon's direction and
/// polarization, the recoils and the new threshold are the CPU's.
#[test]
fn a_jump_is_the_cpu_s(){compare(&request(PGC),Some(0.999),f64::INFINITY,true,false);}
/// The same with the exact response (an allowance of zero: every step beyond the first order).
#[test]
fn a_jump_with_the_exact_response_is_the_cpu_s(){compare(&request(PGC),Some(0.999),0.,true,false);}
/// On a cluster of all 8 ground states (two entries of each matrix per lane): a step, and a jump with
/// the first-order response.
#[test]
fn on_eight_states_a_step_and_a_jump_are_the_cpu_s(){
    let r=request(MOLASSES);compare(&r,None,f64::INFINITY,false,false);compare(&r,Some(0.999),f64::INFINITY,true,false);
}
/// The same jump with the exact response: the gray molasses' near carriers dress each other's responses
/// (ground.rs, # The near light's dressing), which the kernel's exact response leaves out -- its bound
/// booked (solve.rs), 1e-3 of the generator here; the CPU's ψ 5e-5 off after the jump.
#[test]
#[ignore="the kernel's exact response is the undressed one: its port of the near light's dressing is pending"]
fn on_eight_states_a_jump_with_the_exact_response_is_the_cpu_s(){compare(&request(MOLASSES),Some(0.999),0.,true,false);}
/// With explicit excited states (engine.rs # The exact step), on the tweezer PGC's first explicit rung
/// (clusters of 3 ground and 4 explicit states and of 2 and 3, their hyperfine beat left out: the static
/// path, the CPU's eigendecomposition against the GPU's series), the gray molasses' and the probed PGC's (7
/// explicit states beating at 36.4 and 72.8 MHz: CF4, at its solve's 0.1 us):
/// - a jump-free step is the CPU's to single precision;
/// - a jump at each thermal state -- the explicit states' photons and their share of the absorbed light,
///   and into a cluster with explicit states its transient -- is the CPU's to single precision;
/// - a step with its jumps where their thresholds fall makes the same jumps, landings and losses. Its
///   crossing is found on the single-precision norm: in a slowly decaying state (0.02/us on the PGC's
///   explicit rung) its time moves by 1e-3 of itself, and the state after it by as much (unbiased).
#[test]
fn with_explicit_states_a_step_and_a_jump_are_the_cpu_s(){
    let Some(gpu)=device()else{return};
    for(r,dt)in [(request(PGC),0.2),(request(MOLASSES),0.2),(probed(),0.1)]{
        let m=explicit(&r);assert!((0..m.clusters()).any(|k|m.mixed(k)),"explicit states");
        println!("clusters {:?}",(0..m.clusters()).map(|k|(m.dimension(k),m.ground_dimension(k),m.beating(k))).collect::<Vec<_>>());
        compare_model(&r,m.clone(),dt,Some(1e-300),f64::INFINITY,false);
        let(model,settings,trajectories)=setup(&r,m.clone(),dt,None,f64::INFINITY);
        {
            let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());let program=gpu.program(&model,&settings).unwrap();let engine=Engine::new(&model,settings.clone());
            for t in &trajectories{
                let mut c=t.clone();let(r0,t0)=(c.r,c.time);engine.jump(&mut c,r0,t0,false);let g=gpu.jump_probe(&program,&settings,t).unwrap();
                let at=format!("the jump of trajectory {} (cluster {} → {})",t.index,t.cluster,c.cluster);
                assert_eq!((g.cluster,g.jumps),(c.cluster,c.jumps),"{at}");assert_eq!(g.threshold,c.threshold,"{at}");
                let scale=c.kappa.iter().map(|x|x.abs()).fold(1.,f64::max);
                for a in 0..3{assert!((g.kappa[a]-c.kappa[a]).abs()<=1e-6*scale,"{at}: κ {:?} against {:?}",g.kappa,c.kappa);}
                let n=model.dimension(c.cluster);let(gn,cn)=(g.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt(),c.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt());
                for x in 0..n{let(a,b)=(g.psi[x]*(1./gn),c.psi[x]*(1./cn));assert!((a-b).abs()<=1e-5,"{at}: ψ[{x}] {a:?} against {b:?}");}
            }
            // A step with its jumps where their thresholds fall: the same jumps, landings and losses.
            let got=gpu.advance(&program,&settings,&trajectories,1).unwrap();let mut jumped=0;
            for(t,g)in trajectories.iter().zip(&got){let mut c=t.clone();engine.step(&mut c);jumped+=c.jumps as usize;
                assert_eq!((g.cluster,g.jumps,g.alive),(c.cluster,c.jumps,c.alive),"trajectory {}",t.index);}
            println!("{} jumps in a step, the same on both devices",jumped);
        }
    }
}
/// With a retro-reflected tweezer (`retro`): a step, and a jump with the first-order and the exact
/// response (its base and the standing wave's block per unit, gpu/program.rs).
#[test]
fn with_a_retro_reflected_tweezer_a_step_and_a_jump_are_the_cpu_s(){
    let r=retro();let m=model(&r,false);
    assert!(m.ground.units.iter().any(|u|u.beams.len()==2)&&m.trap_beams.len()==2,"the tweezer and its reflection one unit, both the trap's");
    compare(&r,None,f64::INFINITY,false,false);compare(&r,Some(0.999),f64::INFINITY,true,false);compare(&r,Some(0.999),0.,true,false);
}
/// The same at fourth order (model.rs # Fourth order): H^(4) in the generator; at a jump the photons'
/// −C W3 − ½ C R R†R and the slow start ψ ← Π f/√A, θ ← θ/A; the first order's and the exact response.
/// On the tweezer PGC, whose clusters each lie in one hyperfine level: the gray molasses' spans F = 1
/// and F = 2, and at fourth order H^(4)'s entries between them beat at the hyperfine splitting, which
/// the multirate exact step takes on the CPU (gpu/program.rs refuses it).
#[test]
#[ignore="fourth2's FourthLayout (slabs, slow pairs, jump slabs): the kernel's port is pending"]
fn at_fourth_order_a_step_and_a_jump_are_the_cpu_s(){
    let r=request(PGC);compare(&r,None,f64::INFINITY,false,true);compare(&r,Some(0.999),f64::INFINITY,true,true);compare(&r,Some(0.999),0.,true,true);
}

/// The trajectories of a batch (`first`, `count` and `group` as lib.rs `simulate` takes them) whose
/// jumps have fallen in different steps on the two devices after any of `after` steps (ascending),
/// as (trajectory, steps): each device's own trajectories from the batch's initial states, the GPU's
/// in a launch of that many steps from the start. The GPU's ln‖ψ‖² drifts from the CPU's at single
/// precision (about 1e-7 a step); a crossing on a slowly decaying stretch moves by that drift over
/// the decay rate, and the fast decay after its jump turns the move back into ln‖ψ‖² (the gray
/// molasses' trajectory 44 from F7's initial positions on: 4.6e-6 after 26 steps, 1.2e-5 after its
/// first jump, beyond the CPU's margin of 8.8e-4 two jumps later, in step 28 where the CPU's fell in
/// 29). So now and then a crossing near a step's end falls in the next step on one of them, at any
/// step -- a smaller step does not help, the drift does not shrink with it --: it is that and nothing
/// else, the counts never differ by more than one.
fn straddles(gpu:&gpu::Gpu,program:&program::Program,model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,after:&[usize])->Vec<(u64,usize)>{
    let engine=Engine::new(model,settings.clone());
    let start:Vec<Trajectory>=(first..first+count).map(|i|{let s=first+(i-first)/group*group;
        initial_trajectory(model,settings,initial,seed,i,Stratum{first:s,size:(s+group).min(first+count)-s})}).collect();
    let(mut cpu,mut done,mut out)=(start.clone(),0,vec![]);
    for &k in after{
        for _ in done..k{for t in cpu.iter_mut(){if t.alive{engine.step(t);}}}done=k;
        for(c,g)in cpu.iter().zip(gpu.advance(program,settings,&start,k).unwrap()){
            if(c.jumps,c.alive)!=(g.jumps,g.alive){
                assert!(c.jumps.abs_diff(g.jumps)<=1,"trajectory {} after {k} steps: {} jumps against the CPU's {}",c.index,g.jumps,c.jumps);
                out.push((c.index,k));
            }
        }
    }
    out
}

/// A batch on the GPU sums what the same batch sums on the CPU (lib.rs `simulate`), to single
/// precision, with the control variates (control.rs): 20 trajectories in two groups -- tasks of 8, 2,
/// 8 and 2 (blocks with idle warps) --, 3 steps between 6 outputs, the motion's means over each
/// output's window of half a trap period (the windows closing up to the reach past the last output,
/// over several launches), the samples kept at two outputs, every trajectory's harmonic control and,
/// where the generator is static, its held twin. The batch is the first 20 trajectories from the
/// 40th on, by 20, whose jumps fall in the same steps on both devices (`straddles`) wherever the
/// sums can tell: after every step within the windows, and at the run's end, where the jumps are
/// counted. A jump a step apart moves its recoil by a step in a window (1.6e-3 of the gray molasses'
/// Σκ_x² at its last output, trajectory 44 from the 40th). The held twin's jumps are not checked so
/// (no entry steps it alone on the GPU): each fixture's fall in the same steps.
fn batch(r:&Request,fourth:bool)->bool{
    let Some(gpu)=device()else{return false};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let model=model(r,fourth);let initial=Initial::new(&model,r).unwrap();
    let settings=Settings::new(0.2,&initial.traps[0],1e-6,true);
    let(seed,count,group,steps,outputs,listed)=(0x3D_5EED,20,10,3,6,[2,5]);
    let program=gpu.program(&model,&settings).unwrap();
    // The run: the outputs, then the windows' reach past the last, half the slowest trap period
    // (lib.rs `half_windows`). A window takes as many steps after its output as before it (lib.rs
    // `Windows::means`), so the last output's ends `last` past it at most.
    let last=(outputs-1)*steps;
    let reach=settings.lab.iter().filter(|&&w|settings.average&&w>0.).map(|&w|(std::f64::consts::PI/w/settings.dt).round() as usize).max().unwrap_or(0);
    let(windows,run)=(last+reach.min(last),last+reach);
    let after:Vec<usize>=(1..=windows).chain((run>windows).then_some(run)).collect();
    let first=(40u64..).step_by(20).take(8).find(|&first|{
        let s=straddles(&gpu,&program,&model,&settings,&initial,seed,first,count,group,&after);
        if !s.is_empty(){println!("trajectories {first} to {}: jumps a step apart (trajectory, step) {s:?}",first+count-1);}
        s.is_empty()
    }).expect("a batch whose jumps fall in the same steps on both devices");
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);
    let twin=Twin::new(&model,0.5*model.geometry[model.trap_beam].intensity([0.;3]));
    let variates=Variates{harmonic:&harmonic,twin:twin.as_ref()};
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
        close(x.folded,y.folded,1e-9*y.held,&format!("{at}: folded η²"));
    }}
    assert_eq!(g.jumps,c.jumps,"jumps");
    for(gs,cs)in g.samples.iter().zip(&c.samples){
        assert_eq!(gs.len(),cs.len());
        for(x,y)in gs.iter().zip(cs){for a in 0..3{close(x.0[a],y.0[a],1e-3,"a sample's r");close(x.1[a],y.1[a],1.,"a sample's κ");}}
    }
    println!("trajectories {first} to {}: {} groups of {} outputs, {} jumps, held twin {}",first+count-1,g.groups.len(),outputs,g.jumps,twin.is_some());
    twin.is_some()
}
/// The batches of the tweezer PGC, the gray molasses and the retro-reflected tweezer; one of them has
/// a held twin.
#[test]
fn a_batch_s_sums_are_the_cpu_s(){
    let twins=[batch(&request(PGC),false),batch(&request(MOLASSES),false),batch(&retro(),false)];
    if device().is_some(){assert!(twins.iter().any(|&t|t),"a fixture with a held twin");}
}
/// The same at fourth order, on the tweezer PGC, with the held twin's start and post-jump map (control.rs
/// # The twin at fourth order).
#[test]
#[ignore="fourth2's FourthLayout (slabs, slow pairs, jump slabs): the kernel's port is pending"]
fn at_fourth_order_a_batch_s_sums_are_the_cpu_s(){batch(&request(PGC),true);}

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

