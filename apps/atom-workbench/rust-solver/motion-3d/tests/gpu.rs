//! The GPU kernel against the CPU engine (needs an NVIDIA GPU; without one each check says so and
//! passes): the structs' layouts, the random streams, one step, and a jump with the first-order and
//! the exact response, on the app's tweezer PGC example (tests/fixtures/tweezer_pgc.json: clusters of
//! 3 and 5 states) and on a D1 gray molasses (tests/fixtures/tweezer_gray_molasses.json: one cluster
//! of all 8 ground states, the kernel's largest).
use atom_motion_3d::engine::{jump_stream,Engine,Settings,Trajectory,QUADRATURE};
use atom_motion_3d::gpu::{self,driver::Buffer,program};
use atom_motion_3d::ground::Ground;
use atom_motion_3d::linear::Dense;
use atom_motion_3d::model::{Exact,Model,GROUND,MOST_MIXED};
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
/// The tetrahedral MOT on a tweezer (tests/fixtures/tetrahedral_mot_motion.json): its near beams 11 Γ off the F′ = 3 the tweezer's tensor shift splits by 2.3 Γ, from |2,2⟩.
fn tetrahedral()->Request{
    let v:Value=serde_json::from_str(include_str!("fixtures/tetrahedral_mot_motion.json")).unwrap();
    let beams=v["beams"].as_array().unwrap().iter().map(|b|{let d=floats(&b["direction"]);Beam{id:b["id"].as_str().unwrap().into(),name:b["name"].as_str().unwrap().into(),
        direction:[d[0],d[1],d[2]],wave_number_per_um:b["waveNumberPerUm"].as_f64().unwrap(),waist_um:b["waistUm"].as_f64(),intensity_mw_cm2:b["intensityMWcm2"].as_f64().unwrap(),
        frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),coupling:matrix(&b["coupling"]),
        counter_coupling:matrix(&b["counterCoupling"]),outside_shift:matrix(&b["outsideShift"])}}).collect::<Vec<Beam>>();
    let trap_beam=beams.iter().position(|b|b.id==v["trapBeamId"].as_str().unwrap()).unwrap();
    let initial=floats(&v["initial"]);let mut density=vec![Complex::ZERO;STATES*STATES];for i in 0..STATES{density[i*STATES+i]=Complex::new(initial[i],0.);}
    let(e,f)=(v["emission"].as_array().unwrap(),floats(&v["fieldVectorG"]));
    Request{states:v["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect(),hamiltonian:matrix(&v["hamiltonian"]),
        frame_frequencies_mhz:floats(&v["frameFrequenciesMHz"]),frequencies_mhz:floats(&v["frequenciesMHz"]),beams,emission:[matrix(&e[0]),matrix(&e[1]),matrix(&e[2])],
        field_vector_g:[f[0],f[1],f[2]],mass_kg:v["massKg"].as_f64().unwrap(),initial_density:density,temperature_uk:v["temperatureUK"].as_f64().unwrap(),trap_beam,times_us:vec![]}
}
/// The GPU, or None (said) on a machine without one.
fn device()->Option<&'static std::sync::Mutex<gpu::Gpu>>{match gpu::gpu(){Ok(g)=>Some(g),Err(e)=>{println!("no GPU ({e}): nothing to check");None}}}

/// The kernel's structs have their Rust twins' sizes and offsets.
#[test]
fn the_kernel_s_structs_match_their_twins(){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let f=gpu.function("layout").unwrap();let out=Buffer::new(128).unwrap();gpu.stream().launch(&f,1,32,&[&out.pointer]).unwrap();
    let mut words=[0u32;28];gpu.stream().download(&out,&mut words).unwrap();
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
/// The first rung with explicit states that beat at two frequencies.
fn beating_twice(r:&Request)->Model{
    Ground::allowances(r).unwrap().into_iter().map(|a|Model::with(r,a).unwrap()).find(|m|(0..m.clusters()).any(|k|m.mixed(k)&&m.beats(k).0.len()>=4)).expect("a rung beating twice")
}
/// The PGC with a resonant probe 10 MHz off the cooling light (the floquet study's PGCP_10), as the app
/// sends it: its folds' saturation far beyond one, its first rung keeps 5 excited states explicit with the 5
/// ground states of F = 2, a cluster beating at 36.4 MHz that its exact step takes by CF4.
fn probed()->Request{let mut r=scene::request(&serde_json::from_str(include_str!("fixtures/pgc_probe_10mhz.json")).unwrap());r.times_us=vec![0.,1.];r}
/// The same with its last cooling beam a line of its own, at twice the cooling light's offset from the probe: three carriers, rungs that beat at 36.4 and 72.8 MHz, and a photon's near
/// parts whose recoils may leave their covariance an eigenvalue of round-off.
fn probed_twice()->Request{let mut r=probed();let beam=r.beams.last_mut().unwrap();beam.frequency_index[0]*=2;r}
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
/// new threshold the stream's (at fourth order scaled by the transient's (1 − p)/A, single precision). The
/// exact step (explicit states, beats) takes ψ through hundreds of exponentials and a static cluster's
/// propagator as many times, in single precision -- its ψ, norm and momentum to 5e-4, 5e-4 and 5e-5.
fn compare_model(r:&Request,model:Model,dt:f64,threshold:Option<f64>,allowance:f64,jumps:bool){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let(model,settings,trajectories)=setup(r,model,dt,threshold,allowance);
    compare_from(&gpu,&model,&settings,&trajectories,jumps,0.);
}
/// `compare_model` from given states. `crossing`: how far off ln‖ψ‖² the GPU's crossing may lie (`a_step_with_its_jump`), per jump of the step, its tolerances then with what that moves on the CPU itself, a step at each of its thresholds moved so.
fn compare_from(gpu:&gpu::Gpu,model:&Model,settings:&Settings,trajectories:&[Trajectory],jumps:bool,crossing:f64){
    let dt=settings.dt;let fourth=model.fourth;let exact=(0..model.clusters()).any(|k|model.mixed(k)||model.beating(k));let(tp,tn,tk)=if exact{(5e-4,5e-4,5e-5)}else{(2e-5,2e-5,2e-5)};
    let program=gpu.program(model,settings).unwrap();
    let got=gpu.advance(&program,settings,trajectories,1).unwrap();
    let engine=Engine::new(model,settings.clone());
    let mut jumped=0;
    for(start,g)in trajectories.iter().zip(&got){
        let mut c=start.clone();engine.step(&mut c);
        let at=format!("trajectory {} (cluster {} → {}, jumps {})",start.index,start.cluster,c.cluster,c.jumps);
        assert_eq!((g.cluster,g.jumps,g.alive),(c.cluster,c.jumps,c.alive),"{at}");jumped+=(c.jumps>0) as usize;
        let scale=c.kappa.iter().map(|x|x.abs()).fold(1.,f64::max);
        let n=model.dimension(c.cluster);let(gn,cn)=(g.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>(),c.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>());
        // What the crossing's own error moves on the CPU: r, κ, ln‖ψ‖² and ψ.
        let(mut dr,mut dk,mut dn,mut dp)=([0_f64;3],[0_f64;3],0_f64,[0_f64;MOST_MIXED]);
        if crossing>0.{for sign in [-1.,1.]{
            let mut d=start.clone();d.threshold*=(sign*crossing*c.jumps as f64).exp();engine.step(&mut d);
            if d.cluster!=c.cluster{continue;}
            let dd=d.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>();dn=dn.max((dd.ln()-cn.ln()).abs());
            for a in 0..3{dr[a]=dr[a].max((d.r[a]-c.r[a]).abs());dk[a]=dk[a].max((d.kappa[a]-c.kappa[a]).abs());}
            for x in 0..n{dp[x]=dp[x].max((d.psi[x]*(1./dd.sqrt())-c.psi[x]*(1./cn.sqrt())).abs());}
        }}
        assert!((g.threshold-c.threshold).abs()<=if fourth{1e-5*c.threshold}else{0.},"{at}: the new threshold {} against {}",g.threshold,c.threshold);
        for a in 0..3{
            // The exact step's impulse on its node grid within its quadrature share (engine.rs `QUADRATURE`), where the CPU's
            // static cluster may take the spectral path exactly; a momentum's difference moves the position over the step's
            // flow by (ħ/m) dt Δκ.
            let k=tk*scale+if exact{QUADRATURE*(c.kappa[a]-start.kappa[a]).abs()}else{0.};
            assert!((g.r[a]-c.r[a]).abs()<=2e-6*c.r.iter().map(|x|x.abs()).fold(1e-3,f64::max)+if exact{model.hbar_over_mass*dt*k}else{0.}+dr[a],"{at}: r {:?} against {:?}, κ {:?} against {:?}, the crossing's own {dr:?}",g.r,c.r,g.kappa,c.kappa);
            assert!((g.kappa[a]-c.kappa[a]).abs()<=k+dk[a],"{at}: κ {:?} against {:?}, the crossing's own {dk:?}",g.kappa,c.kappa);
        }
        // Each jump redoes part of the step from a crossing found in single precision: the norm and the state alike.
        assert!((gn.ln()-cn.ln()).abs()<=tn*(1+c.jumps) as f64+dn,"{at}: ln‖ψ‖² {} against {}",gn.ln(),cn.ln());
        for x in 0..n{let(a,b)=(g.psi[x]*(1./gn.sqrt()),c.psi[x]*(1./cn.sqrt()));assert!((a-b).abs()<=tp*(1+c.jumps) as f64+dp[x],"{at}: ψ[{x}] {a:?} against {b:?}");}
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
/// On a cluster of all 8 ground states (two entries of each matrix per lane): a jump-free step, a jump at each thermal state with the first-order response, and a step with a jump
/// inside it (`steps_and_jumps`). Past a step's own crossing the continuous quantities agree only as far as the crossing does (`a_step_with_its_jump`): it lies on the
/// single-precision norm and a ringing photon lands at its cluster's frames at the crossing's time, F = 1 and F = 2 4.3e4 rad/us apart. A threshold moved by that round-off (6e-8) on the CPU alone moves 10 of trajectories
/// 0 to 63 (at a threshold of 0.999) by 4e-5 to 3e-4 in the state and up to 1.5e-3 in the momentum, the others by 1.4e-7 and 8e-7; the phase is as good as the photon's own draw, its mixture over τ the coherence Γ/(Γ + iδ) of 8e-4.
#[test]
fn on_eight_states_a_step_and_a_jump_are_the_cpu_s(){let r=request(MOLASSES);steps_and_jumps(&r,model(&r,false),0.2,&[false]);}
/// The same jump with the exact response: the gray molasses' static ground block mixes its responses (ground.rs,
/// # The static ground block), the kernel's by its eigenstates (`carrier_responses`).
#[test]
fn on_eight_states_a_jump_with_the_exact_response_is_the_cpu_s(){let r=request(MOLASSES);steps_and_jumps(&r,model(&r,false),0.2,&[true]);}
/// With explicit excited states (engine.rs # The exact step), on the tweezer PGC's first explicit rung
/// (clusters of 3 ground and 4 explicit states and of 2 and 3, their hyperfine beat left out: the static
/// path, the CPU's eigendecomposition against the GPU's series), the gray molasses', and the probed PGC's (`probed_twice`) first
/// beating at two frequencies (5 explicit states at 36.4 and 72.8 MHz: CF4, at its solve's 0.1 us):
/// - a jump-free step is the CPU's to single precision;
/// - a jump at each thermal state -- the explicit states' photons and their share of the absorbed light,
///   and its transient --, with the first-order and with the exact
///   response (K decoupled from the explicit states, their dressing in the source), is the CPU's to single
///   precision: with the exact response, the resolvents' error (`the_exact_responses_are_the_cpu_s`, 5e-7
///   of a column's largest entry here) times the cancellation in a class's amplitude, Σ|terms|/|u| up to 19
///   on the PGC's explicit rung, whose beams all lie along one axis (κ 5e-6 off at most); with the first order, the
///   tables' single-precision slopes times that cancellation (its nearest state's κ_z scatters by 7e-7 of κ, up to
///   2e-6, unbiased, as the position moves by 2e-5 um);
/// - a step with a jump inside it, at every place of the step, makes the same jump, landing and loss, and leaves the same r, κ and ψ
///   (`a_step_with_its_jump`): the step redone to the jump's crossing, in the same order. The crossing is found on the single-precision norm (1.7e-6 to 5.4e-6 of
///   ln‖ψ‖² off here): in a slowly decaying state (0.02/us on the PGC's explicit rung) its time moves by 1e-3 of itself, and the state after it by as much (unbiased).
#[test]
fn with_explicit_states_a_step_and_a_jump_are_the_cpu_s(){
    let(pgc,molasses,probe)=(request(PGC),request(MOLASSES),probed_twice());
    explicit_rung(&pgc,explicit(&pgc),0.2,&[false,true]);
    explicit_rung(&molasses,explicit(&molasses),0.2,&[false]);
    explicit_rung(&probe,beating_twice(&probe),0.1,&[false,true]);
}
/// The gray molasses' explicit rung's jump with the exact response: its block mixes (the kernel's `carrier_responses`).
#[test]
fn with_explicit_states_a_jump_with_a_mixed_block_s_exact_response_is_the_cpu_s(){let r=request(MOLASSES);explicit_rung(&r,explicit(&r),0.2,&[true]);}
/// A request's explicit rung (`with_explicit_states_a_step_and_a_jump_are_the_cpu_s`), its jumps with the responses
/// `exact` lists: true, the exact one; false, the first order's.
fn explicit_rung(r:&Request,m:Model,dt:f64,exact:&[bool]){
    assert!((0..m.clusters()).any(|k|m.mixed(k)),"explicit states");
    println!("clusters {:?}",(0..m.clusters()).map(|k|(m.dimension(k),m.ground_dimension(k),m.beating(k))).collect::<Vec<_>>());
    steps_and_jumps(r,m,dt,exact);
}
/// A jump at a state on both devices, at its own position and time, so at one crossing, with the first-order or the exact response: the same landing, the same threshold (at fourth order
/// scaled by the transient's (1 − p)/A in single precision), its momentum within `kappa` of its largest and the state it leaves within `psi`. What they lie apart by.
fn same_jump(gpu:&gpu::Gpu,program:&program::Program,engine:&Engine,model:&Model,settings:&Settings,t:&Trajectory,exact:bool,kappa:f64,psi:f64)->[f64;2]{
    let mut c=t.clone();let(r0,t0)=(c.r,c.time);engine.jump(&mut c,r0,t0,exact);let g=gpu.jump_probe(program,settings,t,exact).unwrap();
    let at=format!("the jump of trajectory {} (cluster {} → {}, exact {exact})",t.index,t.cluster,c.cluster);
    assert_eq!((g.cluster,g.jumps),(c.cluster,c.jumps),"{at}");
    assert!((g.threshold-c.threshold).abs()<=if model.fourth{1e-5*c.threshold}else{0.},"{at}: the threshold {} against {}",g.threshold,c.threshold);
    let scale=c.kappa.iter().map(|x|x.abs()).fold(1.,f64::max);let mut off=[0_f64;2];
    for a in 0..3{off[0]=off[0].max((g.kappa[a]-c.kappa[a]).abs()/scale);assert!((g.kappa[a]-c.kappa[a]).abs()<=kappa*scale,"{at}: κ {:?} against {:?}",g.kappa,c.kappa);}
    let n=model.dimension(c.cluster);let(gn,cn)=(g.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt(),c.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt());
    for x in 0..n{let(a,b)=(g.psi[x]*(1./gn),c.psi[x]*(1./cn));off[1]=off[1].max((a-b).abs());assert!((a-b).abs()<=psi,"{at}: ψ[{x}] {a:?} against {b:?}");}
    off
}
/// A model's jump-free step, its jump at each thermal state with the responses `exact` lists, and a step with a jump inside it.
fn steps_and_jumps(r:&Request,m:Model,dt:f64,exact:&[bool]){
    let Some(gpu)=device()else{return};
    compare_model(r,m.clone(),dt,Some(1e-300),f64::INFINITY,false);
    let(model,settings,trajectories)=setup(r,m,dt,None,f64::INFINITY);
    let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());let program=gpu.program(&model,&settings).unwrap();let engine=Engine::new(&model,settings.clone());
    for(t,&exact)in trajectories.iter().flat_map(|t|exact.iter().map(move|e|(t,e))){same_jump(&gpu,&program,&engine,&model,&settings,t,exact,if exact{2e-5}else{5e-6},1e-5);}
    a_step_with_its_jump(&gpu,&model,&settings,&trajectories);
}
/// A step with a jump inside it, on both devices the same jump, landing and loss, and r, κ and ψ after it: each state's threshold at its own share of the state's fall over a jump-free step, the shares evenly over
/// the step, so the crossings lie everywhere in it. The position says where in the step the jump was made: held at the step's middle through its jump, as the exact step once was, the recoil moves r by (ħ/m) κ_jump (τ/2 − t*). The crossing lies on the
/// single-precision norm: within the GPU's largest ln‖ψ‖² off the CPU's in that jump-free step, and what that moves on the CPU itself goes into the tolerances (`compare_from`); a state that falls by less than that has no crossing to place.
fn a_step_with_its_jump(gpu:&gpu::Gpu,model:&Model,settings:&Settings,thermal:&[Trajectory]){
    let engine=Engine::new(model,settings.clone());let program=gpu.program(model,settings).unwrap();
    let norm=|t:&Trajectory|t.psi[..model.dimension(t.cluster)].iter().map(|z|z.norm_sqr()).sum::<f64>().ln();
    let free:Vec<Trajectory>=thermal.iter().map(|t|{let mut x=t.clone();x.threshold=1e-300;x}).collect();
    let ends:Vec<f64>=free.iter().map(|t|{let mut c=t.clone();engine.step(&mut c);norm(&c)}).collect();
    let round=gpu.advance(&program,settings,&free,1).unwrap().iter().zip(&ends).map(|(g,e)|(norm(g)-e).abs()).fold(0.,f64::max);
    let last=thermal.len() as f64;
    let placed:Vec<Trajectory>=thermal.iter().zip(&ends).enumerate().filter_map(|(i,(t,e))|{let share=(i as f64+1.)/(last+1.);let fall=norm(t)-e;
        (fall*(1.-share)>2.*round).then(||{let mut x=t.clone();x.threshold=(norm(t)-share*fall).exp();x})}).collect();
    assert!(2*placed.len()>=thermal.len(),"{} of {} states fall by more than the norm's {round:.1e} in a step",placed.len(),thermal.len());
    compare_from(gpu,model,settings,&placed,true,round);
    println!("{} of {} states' jumps within a step, the crossing within {round:.1e}",placed.len(),thermal.len());
}
/// The exact responses at thermal states (the kernel's `exact_responses`, model.rs `Exact`): every near
/// beam's C_a R_l at each cluster column, dressed by the near light (ground.rs, # The near light's
/// dressing), within 1e-5 of the column's largest entry of the CPU's -- single precision's resolvent, NE ε =
/// 1.4e-6 times its condition (3.2e-6 at most here) --; on the probed PGC's explicit rung (K decoupled from 7 explicit
/// states, their dressing in the source), where the dressing moves them by 6.9e-4.
#[test]
fn the_exact_responses_are_the_cpu_s(){let r=probed();let m=explicit(&r);exact_responses(&r,m);}
/// On the gray molasses (one cluster of all 8 ground states), whose static block mixes its responses (the kernel's
/// `mixed_responses`).
#[test]
fn on_a_mixed_block_the_exact_responses_are_the_cpu_s(){let r=request(MOLASSES);let m=model(&r,false);exact_responses(&r,m);}
/// A model's exact responses on both devices (`the_exact_responses_are_the_cpu_s`).
fn exact_responses(r:&Request,m:Model){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let mut bare=m.clone();bare.ground.dressing=false;
    let(model,settings,trajectories)=setup(r,m,0.2,None,0.);let program=gpu.program(&model,&settings).unwrap();
    let g=&model.ground;let near:Vec<usize>=(0..g.beams.len()).filter(|&l|g.carriers.iter().any(|c|c.beams.contains(&l))).collect();
    let(mut worst,mut dressing)=(0_f64,0_f64);
    for t in &trajectories{
        let got=gpu.response_probe(&program,&settings,t).unwrap();
        let(cpu,undressed)=(Exact::new(&model,t.r,t.kappa).unwrap(),Exact::new(&bare,t.r,t.kappa).unwrap());
        // A column at single precision's round-off of the state's largest entry (a coupling at the round-off of the
        // data) has no relative accuracy to check.
        let largest=near.iter().flat_map(|&l|(0..3).map(move|a|(l,a))).map(|(l,a)|g.emission[a].product(&cpu.responses[l]).entries.iter().map(|z|z.abs()).fold(0.,f64::max)).fold(0.,f64::max);
        for(j,&l)in near.iter().enumerate(){
            let(c,u):(Vec<Dense>,Vec<Dense>)=(0..3).map(|a|(g.emission[a].product(&cpu.responses[l]),g.emission[a].product(&undressed.responses[l]))).unzip();
            for(col,&b)in g.clusters[t.cluster].states.iter().enumerate(){
                let scale=(0..3).flat_map(|a|(0..GROUND).map(move|to|(a,to))).map(|(a,to)|c[a][(to,b)].abs()).fold(f32::EPSILON as f64*largest,f64::max);if scale==0.{continue;}
                for a in 0..3{for to in 0..GROUND{
                    let x=got[((j*program::MAXN+col)*3+a)*GROUND+to];
                    worst=worst.max((x-c[a][(to,b)]).abs()/scale);dressing=dressing.max((u[a][(to,b)]-c[a][(to,b)]).abs()/scale);
                }}
            }
        }
    }
    println!("{} states: the GPU's {worst:.1e} off, the dressing {dressing:.1e}",trajectories.len());
    assert!(worst<=1e-5,"the exact responses {worst:.2e} off the CPU's");assert!(dressing>=10.*1e-5,"the dressing ({dressing:.1e}) within the check's reach");
}
/// A model's every cluster from the thermal (r, κ) (model.rs, # The photons and the Doppler term: each cluster's groups take their own transfers and leave their own
/// entries), a superposition of its states in place of the initial one: a jump-free step, and a jump at the state, its photon's
/// recoil and the state it leaves relative to the photon's own size.
fn every_cluster(r:&Request,m:Model,dt:f64){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let(model,settings,thermal)=setup(r,m,dt,Some(1e-300),f64::INFINITY);
    let states:Vec<Trajectory>=thermal.iter().enumerate().map(|(i,t)|{
        let k=i%model.clusters();let n=model.dimension(k);let mut x=t.clone();x.cluster=k;x.psi=[Complex::ZERO;MOST_MIXED];
        for b in 0..n{x.psi[b]=Complex::cis(1.7*((b+1)*(i+1)) as f64)*(1./(n as f64).sqrt());}x}).collect();
    compare_from(&gpu,&model,&settings,&states,false,0.);
    let program=gpu.program(&model,&settings).unwrap();let engine=Engine::new(&model,settings.clone());
    for t in &states{
        let mut c=t.clone();let(r0,t0)=(c.r,c.time);engine.jump(&mut c,r0,t0,false);let g=gpu.jump_probe(&program,&settings,t,false).unwrap();
        let at=format!("the jump of trajectory {} (cluster {} → {})",t.index,t.cluster,c.cluster);
        assert_eq!((g.cluster,g.jumps),(c.cluster,c.jumps),"{at}");assert_eq!(g.threshold,c.threshold,"{at}");
        let scale=c.kappa.iter().map(|x|x.abs()).fold(1.,f64::max);
        for a in 0..3{assert!((g.kappa[a]-c.kappa[a]).abs()<=5e-5*scale,"{at}: κ {:?} against {:?}",g.kappa,c.kappa);}
        let n=model.dimension(c.cluster);let(gn,cn)=(g.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt(),c.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt());
        for x in 0..n{let(a,b)=(g.psi[x]*(1./gn),c.psi[x]*(1./cn));assert!((a-b).abs()<=1e-4,"{at}: ψ[{x}] {a:?} against {b:?}");}
    }
}
/// What the first order's transfer leaves (model.rs `Model::transfer_tables`), at its entries and their mirrors in the generator: the default scene's static clusters,
/// its explicit rung's and the probed PGC's beating ones, where a group between two carriers keeps entries at either beat.
#[test]
fn with_what_the_transfer_leaves_a_step_and_a_jump_are_the_cpu_s(){
    let fixture:Value=serde_json::from_str(include_str!("../../tests/fixtures/default_scene_motion.json")).unwrap();
    let(default,probe)=(scene::request(&fixture["payload"]),probed());
    for(r,m,dt)in [(&default,model(&default,false),0.2),(&default,explicit(&default),0.2),(&probe,model(&probe,false),0.1)]{
        assert!((0..m.clusters()).any(|k|m.kept_entries(k)>0),"entries the transfer leaves");every_cluster(r,m,dt);
    }
}
/// A cluster's exact-step components at thermal points (the kernel's `exact_components`, model.rs `Model::components`), every part's H_q and G_q entry by entry -- with the
/// first order's coefficients on Q at the part and its mirror, what the transfer leaves and the cross terms' X2_k --: H_q within 1e-5 of the part's largest or of the generator's
/// single-precision round-off, H₀'s also within 1e-6 of the scalar it left out (h0's single-precision round-off in a state whose light has all but gone), G_q within 1e-4 of its largest (the tables' single-precision
/// slopes) or 1e-5 of that scalar (close to the focus the intensity's gradient is what is left of the wave's k a a*, 2e-5 rad/us/um of 0.13 there).
#[test]
fn the_exact_components_are_the_cpu_s(){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let fixture:Value=serde_json::from_str(include_str!("../../tests/fixtures/default_scene_motion.json")).unwrap();
    let(default,pgc,molasses,probe,twice)=(scene::request(&fixture["payload"]),request(PGC),request(MOLASSES),probed(),probed_twice());
    for(r,m)in [(&default,explicit(&default)),(&pgc,explicit(&pgc)),(&molasses,explicit(&molasses)),(&probe,model(&probe,false)),(&probe,explicit(&probe)),(&twice,beating_twice(&twice))]{
        let(model,settings,thermal)=setup(r,m,0.1,None,f64::INFINITY);let program=gpu.program(&model,&settings).unwrap();let mut worst=[0_f64;2];
        for(i,t)in thermal.iter().enumerate(){
            let k=i%model.clusters();let mut x=t.clone();x.cluster=k;
            let c=model.components(k,&model.point(x.r,x.kappa));let n=c.n();let nn=n*n;let parts=1+c.omegas().len();
            let got=gpu.exact_probe(&program,&model,&settings,&x).unwrap();
            let round=f32::EPSILON as f64*c.beats().flat_map(|b|b.hamiltonian).chain(c.hamiltonian()).map(|z|z.abs()).fold(1e30*f32::MIN_POSITIVE as f64,f64::max);
            for q in 0..parts{
                let(h,g)=if q==0{(c.hamiltonian(),[c.gradient(0),c.gradient(1),c.gradient(2)])}else{let b=c.beat(q-1);(b.hamiltonian,b.gradient)};
                let check=|cpu:&[Complex],gpu:&[Complex],floor:f64,within:f64,what:String,worst:&mut [f64;2],side:usize|{let largest=cpu.iter().map(|z|z.abs()).fold(floor,f64::max);
                    for(e,(a,b))in cpu.iter().zip(gpu).enumerate(){let off=(*a-*b).abs()/largest;worst[side]=worst[side].max(off);assert!(off<=within,"cluster {k}, {what}, entry {e}: {b:?} against {a:?} of {largest}");}};
                check(h,&got[q*nn..(q+1)*nn],if q==0{0.1*c.scalar.abs()}else{round},1e-5,format!("H_{q}"),&mut worst,0);
                for a in 0..3{check(g[a],&got[(parts+3*q+a)*nn..(parts+3*q+a+1)*nn],(0.1*c.scalar.abs()).max(round),1e-4,format!("G_{q}[{a}]"),&mut worst,1);}
            }
        }
        println!("clusters {:?}: the GPU's H {:.1e} and G {:.1e} off",(0..model.clusters()).map(|k|(model.dimension(k),model.ground_dimension(k),model.beats(k).0.len())).collect::<Vec<_>>(),worst[0],worst[1]);
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
/// −C W3 − ½ C R R†R and the slow part ψ ← Π f/√A, θ ← θ (1 − p)/A; the first order's and the exact response.
/// On the tweezer PGC, whose clusters each lie in one hyperfine level: the gray molasses' spans F = 1
/// and F = 2, and at fourth order H^(4)'s entries between them beat at the hyperfine splitting, which
/// the multirate exact step takes.
#[test]
fn at_fourth_order_a_step_and_a_jump_are_the_cpu_s(){
    let r=request(PGC);compare(&r,None,f64::INFINITY,false,true);compare(&r,Some(0.999),f64::INFINITY,true,true);compare(&r,Some(0.999),0.,true,true);
}

/// The same on the gray molasses, whose H^(4) beats at the hyperfine splitting: its slabs in the parts of the exact step. A jump-free
/// step, and a jump at each thermal state: past a step's own crossing the slow part's and the photon's phases at its time, 4.3e4 rad/us between F = 1 and F = 2, are as good as a draw (a crossing moved by 1e-4 us moves (1 − p)/A by 0.5 % and κ by 0.6/um).
#[test]
fn at_fourth_order_a_beating_step_and_jump_are_the_cpu_s(){
    let Some(gpu)=device()else{return};
    let r=request(MOLASSES);let m=model(&r,true);assert!((0..m.clusters()).any(|k|m.beating(k)),"its fourth order beats");
    compare_model(&r,m.clone(),0.2,Some(1e-300),f64::INFINITY,false);
    let(model,settings,trajectories)=setup(&r,m,0.2,None,f64::INFINITY);
    let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());let program=gpu.program(&model,&settings).unwrap();let engine=Engine::new(&model,settings.clone());
    for t in &trajectories{same_jump(&gpu,&program,&engine,&model,&settings,t,false,5e-6,1e-5);}
}

/// The ringing's photon (engine.rs, # The transient; the kernel's `ringing_photon`), a jump's as its draw falls within its probability p (1e-3 to 5e-3 on the tweezer PGC, so no jump above takes it there): each
/// thermal state on the first jump number of its stream whose draw lies under `RINGING_DRAW`, below p on most: the group and manifold by their weights, the eigenstates of H on the manifold, the photon's time, its landing cluster,
/// direction and polarization, its kicks through the channels' tables, and the slow start of the state it leaves; the devices at one crossing, within 4.1e-7 of the momentum and 3.8e-6 of the state at most (`same_jump`).
fn ringing_photons(r:&Request,m:Model,dt:f64,exact:bool){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let(model,settings,thermal)=setup(r,m,dt,None,f64::INFINITY);let program=gpu.program(&model,&settings).unwrap();let engine=Engine::new(&model,settings.clone());
    let(mut off,mut reached)=([0_f64;2],0);
    for t in &thermal{
        let mut x=t.clone();x.jumps=(1..).find(|&j|{let mut rng=jump_stream(t.seed,t.index,j);for _ in 0..4{rng.uniform();}rng.uniform()<RINGING_DRAW}).unwrap()-1;
        let o=same_jump(&gpu,&program,&engine,&model,&settings,&x,exact,2e-6,1e-5);off=[off[0].max(o[0]),off[1].max(o[1])];
        // The state it leaves has about as large a p: most of them went through the photon.
        let mut c=x.clone();let(r0,t0)=(c.r,c.time);engine.jump(&mut c,r0,t0,exact);let point=model.point(c.r,c.kappa);
        let p:f64=model.ringing_amplitudes(c.cluster,&model.responses(c.cluster,&point),&c.psi,c.time).iter().map(|v|v.iter().map(|z|z.norm_sqr()).sum::<f64>()).sum();reached+=(p>RINGING_DRAW) as usize;
    }
    println!("clusters {:?}, manifolds {:?}, exact {exact}: {reached} of {} states with p over the draw; κ {:.1e}, ψ {:.1e} off",(0..model.clusters()).map(|k|(model.dimension(k),model.ground_dimension(k))).collect::<Vec<_>>(),model.ground.excited_groups.iter().map(|x|x.len()).collect::<Vec<_>>(),thermal.len(),off[0],off[1]);
    assert!(2*reached>=thermal.len(),"{reached} of {} states with p over the draw",thermal.len());
}
/// The draw `ringing_photons` takes.
const RINGING_DRAW:f64=1e-4;
/// On the tweezer PGC at either order, each cluster in one hyperfine level, and on the gray molasses, a cluster over both, its landing states' frames apart.
#[test]
fn a_ringing_photon_is_the_cpu_s(){
    let(pgc,molasses)=(request(PGC),request(MOLASSES));
    for fourth in [false,true]{ringing_photons(&pgc,model(&pgc,fourth),0.2,false);ringing_photons(&molasses,model(&molasses,fourth),0.2,false);}
    ringing_photons(&pgc,model(&pgc,false),0.2,true);
}
/// On the tetrahedral MOT, whose tweezer's tensor shift the photon's eigenstates take: H's couplings across F′ mix 3 % there, and a jump's photon lands 66 % back in the mode with them, 96 % coherent over the split (engine.rs, # The transient).
#[test]
fn a_tensor_split_line_s_ringing_photon_is_the_cpu_s(){
    let r=tetrahedral();
    for fourth in [false,true]{ringing_photons(&r,model(&r,fourth),0.2,false);}
}
/// With explicit states, the folded amplitude's H decoupled from them: on the tweezer PGC's first explicit rung and the probed PGC's (`probed_twice`) beating at two frequencies (the gray molasses' has explicit what drives it: p below the draw).
#[test]
fn with_explicit_states_a_ringing_photon_is_the_cpu_s(){
    let(pgc,probe)=(request(PGC),probed_twice());
    ringing_photons(&pgc,explicit(&pgc),0.2,false);ringing_photons(&probe,beating_twice(&probe),0.1,false);
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
/// step -- a smaller step does not help, the drift does not shrink with it --: within `within` steps it is that and nothing
/// else, the counts differ by one at most. Further on a jump itself can part them, on either device: the mean recoil of a state
/// the light nearly leaves dark is as sensitive as its photon is small (the gray molasses' trajectory 45 from the 40th at 0.1 us, on the CPU alone,
/// r and κ moved by 1e-7 of themselves: Δκ 3e-4 after 180 steps, 0.06 at 200, three jumps on, 13/um and two jumps apart at 280, as on the GPU).
fn straddles(gpu:&gpu::Gpu,program:&program::Program,model:&Model,settings:&Settings,initial:&Initial,seed:u64,first:u64,count:u64,group:u64,after:&[usize],within:usize)->Vec<(u64,usize)>{
    let engine=Engine::new(model,settings.clone());
    let start:Vec<Trajectory>=(first..first+count).map(|i|{let s=first+(i-first)/group*group;
        initial_trajectory(model,settings,initial,seed,i,Stratum{first:s,size:(s+group).min(first+count)-s})}).collect();
    let(mut cpu,mut done,mut out)=(start.clone(),0,vec![]);
    for &k in after{
        for _ in done..k{for t in cpu.iter_mut(){if t.alive{engine.step(t);}}}done=k;
        for(c,g)in cpu.iter().zip(gpu.advance(program,settings,&start,k).unwrap()){
            if(c.jumps,c.alive)!=(g.jumps,g.alive){
                assert!(k>within||c.jumps.abs_diff(g.jumps)<=1,"trajectory {} after {k} steps: {} jumps against the CPU's {}",c.index,g.jumps,c.jumps);
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
/// counted -- but where a cluster's states' frames lie apart (`frames_apart`): there a ringing photon's landing takes their phase at its crossing's time, in single precision (`on_eight_states_a_step_and_a_jump_are_the_cpu_s`),
/// and over the half trap period past the windows 1 to 3 of every 20 trajectories part by a jump (the gray molasses' batches from the 40th to the 180th), so that its total is left out. A jump a step apart moves its recoil by a step in a window (1.6e-3 of the gray molasses'
/// Σκ_x² at its last output, trajectory 44 from the 40th). The held twin's jumps are not checked so
/// (no entry steps it alone on the GPU): each fixture's fall in the same steps. The trap's region is
/// the solve's (its depth: the loss's, and the harmonic part's weight beyond it, engine.rs), the step
/// `dt` one the batch's trajectories converge at: the devices' precisions part where they do not.
fn batch(r:&Request,fourth:bool,dt:f64)->bool{
    let Some(gpu)=device()else{return false};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let model=model(r,fourth);let initial=Initial::new(&model,r).unwrap();
    let mut settings=Settings::new(dt,&initial.traps[0],1e-6,true);settings.region=initial.depths[0];
    let(seed,count,group,steps,outputs,listed)=(0x3D_5EED,20,10,3,6,[2,5]);
    let program=gpu.program(&model,&settings).unwrap();
    // The run: the outputs, then the windows' reach past the last, half the slowest trap period
    // (lib.rs `half_windows`). A window takes as many steps after its output as before it (lib.rs
    // `Windows::means`), so the last output's ends `last` past it at most.
    let last=(outputs-1)*steps;
    let reach=settings.lab.iter().filter(|&&w|settings.average&&w>0.).map(|&w|(std::f64::consts::PI/w/settings.dt).round() as usize).max().unwrap_or(0);
    let(windows,run)=(last+reach.min(last),last+reach);
    let apart=frames_apart(&model);let after:Vec<usize>=(1..=windows).chain((run>windows&&!apart).then_some(run)).collect();
    let first=(40u64..).step_by(20).take(8).find(|&first|{
        let s=straddles(&gpu,&program,&model,&settings,&initial,seed,first,count,group,&after,windows);
        if !s.is_empty(){println!("trajectories {first} to {}: jumps a step apart (trajectory, step) {s:?}",first+count-1);}
        s.is_empty()
    }).expect("a batch whose jumps fall in the same steps on both devices");
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);
    let twin=Twin::new(&model,0.5*model.geometry[model.trap_beam].intensity([0.;3]));
    let variates=Variates{harmonic:&harmonic,twin:twin.as_ref()};
    let g=gpu.simulate(&program,&model,&settings,&initial,seed,first,count,group,steps,outputs,&listed,Some(&variates),None).unwrap();
    let c=atom_motion_3d::simulate(&model,&settings,&initial,seed,first,count,group,steps,outputs,&listed,Some(&variates));
    assert_eq!(g.groups.len(),c.groups.len());
    let close=|a:f64,b:f64,scale:f64,what:&str|assert!((a-b).abs()<=1e-3*b.abs().max(scale),"{what}: {a} against {b}");
    for(k,(gg,cg))in g.groups.iter().zip(&c.groups).enumerate(){for(i,(x,y))in gg.sums.iter().zip(&cg.sums).enumerate(){
        let at=format!("group {k}, output {i}");
        assert_eq!((x.trajectories,x.held),(y.trajectories,y.held),"{at}: trajectories and held");
        for a in 0..3{
            close(x.kappa[a],y.kappa[a],(y.kappa2[a]*y.held).sqrt(),&format!("{at}: Σκ[{a}]"));close(x.kappa2[a],y.kappa2[a],0.,&format!("{at}: Σκ²[{a}]"));
            for j in 0..3{close(x.harmonic[a][j],y.harmonic[a][j],1e-9*y.harmonic[a][1].sqrt(),&format!("{at}: harmonic[{a}][{j}]"));}
        }
        for s in 0..STATES{close(x.populations[s],y.populations[s],y.held,&format!("{at}: Σp[{s}]"));close(x.twin[s],y.twin[s],y.trajectories,&format!("{at}: twin Σc[{s}]"));}
        for j in 0..2{close(x.twin2[j],y.twin2[j],y.trajectories,&format!("{at}: twin2[{j}]"));}
        close(x.profile,y.profile,1e-6*y.held,&format!("{at}: profile"));close(x.doppler,y.doppler,1e-9*y.held,&format!("{at}: (kv/σ)²"));
        close(x.folded,y.folded,1e-9*y.held,&format!("{at}: folded η²"));
    }}
    if !apart{assert_eq!(g.jumps,c.jumps,"jumps");}
    for(gs,cs)in g.samples.iter().zip(&c.samples){
        assert_eq!(gs.len(),cs.len());
        for(x,y)in gs.iter().zip(cs){for a in 0..3{close(x.0[a],y.0[a],1e-3,"a sample's r");close(x.1[a],y.1[a],1.,"a sample's κ");}}
    }
    println!("trajectories {first} to {}: {} groups of {} outputs, {} jumps, held twin {}",first+count-1,g.groups.len(),outputs,g.jumps,twin.is_some());
    twin.is_some()
}
/// Whether a cluster's ground states' frames lie apart.
fn frames_apart(model:&Model)->bool{model.ground.clusters.iter().any(|c|c.frames[..c.states.len()].iter().any(|&w|w!=c.frames[0]))}
/// The batches of the tweezer PGC, the gray molasses and the retro-reflected tweezer; one of them has
/// a held twin. The retro-reflected tweezer's at 0.05 us: its initial atoms fill the standing wave's
/// neighbouring sites too (ensemble.rs, 40 % of them), whose own wells are as stiff as the central
/// site's while the harmonic part about the focus there is ten times the depth (weighted down,
/// engine.rs) -- their κ after 3 us moves by up to 14 % between 0.2 and 0.05 us, by 1 % between 0.1
/// and 0.05 us, the central site's by 10⁻⁴.
#[test]
fn a_batch_s_sums_are_the_cpu_s(){
    let twins=[batch(&request(PGC),false,0.1),batch(&request(MOLASSES),false,0.1),batch(&retro(),false,0.05)];
    if device().is_some(){assert!(twins.iter().any(|&t|t),"a fixture with a held twin");}
}
/// A batch's groups each through its own last output (lib.rs `simulate_to`, solve.rs # Batches) on the GPU are the CPU's: the same sums at the outputs a group reaches -- with its windows' far halves
/// past its last --, none counted beyond, and the samples of those that reach a listed output.
#[test]
fn a_group_stops_at_its_last_output_as_on_the_cpu(){
    let Some(gpu)=device()else{return};let gpu=gpu.lock().unwrap_or_else(|e|e.into_inner());
    let r=request(PGC);let model=model(&r,false);let initial=Initial::new(&model,&r).unwrap();
    let mut settings=Settings::new(0.1,&initial.traps[0],1e-6,true);settings.region=initial.depths[0];
    let(seed,first,count,group,steps,outputs,listed,ends)=(0x3D_5EED,40,32,8,3,6,[2,5],[5,3,1,5]);
    let program=gpu.program(&model,&settings).unwrap();
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);let variates=Variates{harmonic:&harmonic,twin:None};
    let g=gpu.simulate(&program,&model,&settings,&initial,seed,first,count,group,steps,outputs,&listed,Some(&variates),Some(&ends)).unwrap();
    let c=atom_motion_3d::simulate_to(&model,&settings,&initial,seed,first,count,group,steps,outputs,&listed,Some(&variates),&ends);
    let close=|a:f64,b:f64,what:&str|assert!((a-b).abs()<=1e-3*b.abs().max(1.),"{what}: {a} against {b}");
    for(k,(gg,cg))in g.groups.iter().zip(&c.groups).enumerate(){for(i,(x,y))in gg.sums.iter().zip(&cg.sums).enumerate(){
        let at=format!("group {k}, output {i}");assert_eq!((x.trajectories,x.held),(y.trajectories,y.held),"{at}");
        for a in 0..3{close(x.kappa2[a],y.kappa2[a],&format!("{at}: Σκ²[{a}]"));for j in 0..3{close(x.harmonic[a][j],y.harmonic[a][j],&format!("{at}: harmonic[{a}][{j}]"));}}
        for s in 0..STATES{close(x.populations[s],y.populations[s],&format!("{at}: Σp[{s}]"));}
        if i>ends[k]{assert_eq!(x.held,0.,"{at}: none beyond its last output");}
    }}
    for(gs,cs)in g.samples.iter().zip(&c.samples){assert_eq!(gs.len(),cs.len());}
}
/// The same at fourth order, on the tweezer PGC, with the held twin's start and post-jump map (control.rs
/// # The twin at fourth order).
#[test]
fn at_fourth_order_a_batch_s_sums_are_the_cpu_s(){batch(&request(PGC),true,0.2);}

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
    let out=gpu.simulate(&program,&model,&settings,&initial,0x3D_5EED,0,256,32,steps,outputs,&[],Some(&Variates{harmonic:&harmonic,twin:Some(&twin)}),None).unwrap();
    let estimates=summarize(&out.groups,&Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab},&means);
    let worst=estimates.iter().filter(|e|e.loss<1.).map(|e|(e.populations.iter().sum::<f64>()-1.).abs()).fold(0.,f64::max);
    println!("the rows' largest |sum - 1|: {worst:.2e}");
    assert!(worst<=1e-9,"a row's sum is off by {worst:.2e}");
}

