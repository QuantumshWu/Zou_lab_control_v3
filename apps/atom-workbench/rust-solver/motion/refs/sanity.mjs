// node sanity.mjs --worker=<atom-lindblad.exe> [--scenes=S1,S2,S2Q,S3] [--duration=5000] [--old] [--app=<apps/atom-workbench>]
//                 [--save=<summary.json>]
//
// The two most basic motion scenes, built exactly as the app builds them and solved by a worker over its JSONL protocol
// as the app drives it, judged against physics (the user, 2026-11-11: "最最基础的两个场景……这些基本的sanity check得过啊"):
//   S1  the app's default tweezer (852 nm, 4 mW, w0 1.2 µm, along +x, π) and one probe added with ＋ (types.ts newLaser:
//       50 µW in a 1 mm waist, 3.18 mW/cm²) set σ+ along the bias field (the app's σ+ preset), at the free-space F=2 → F'=3
//       frequency (the default probe of the default scene removed); initial |F=2, m=2⟩, motion at 30 µK (the defaults);
//   S2  S1 + the probe's reflection by the app's mirror (reflection.ts addReflection, default mirror: R 1, φ 0, no λ/4)
//       + a repump added with ＋ and set to F=1 → F'=2 (along +x, π);
//   S2Q S2 with the mirror's λ/4 plate passed twice (reflection quarterWave: true): σ+ and σ− about the field, the
//       PGC-like polarization gradient; no longer a two-level problem (the atom is pumped over the m sublevels).
//   S3  the user's experiment (2026-11-12) in the lab's geometry, with the app's default tweezer (along +x), field and
//       detunings: a probe added with ＋ along +y tilted 4° toward the tweezer's axis, set π, 1 mW in a 1 mm waist
//       (63.7 mW/cm²), its F=2 → F'=3 offset −20 MHz (the user, 2026-11-12: "探测光应该是失谐的，对自由空间也红失谐差不多20MHz"),
//       the tweezer at 7.25 mW for its 1.5 mK depth (the user: "光镊差不多1.5mK吧"; the default 4 mW makes 828 µK),
//       its reflection by the mirror with the λ/4 plate at its default 45° (lin⊥lin), S2's repump; 100 ms
//       (or --duration), 201 samples. Report mode: the run's checks only, its T and held share printed beside the user's
//       statement (held over 100 ms, little heating -- a physical fact the engine is to reproduce).
// What the check expects is physics, derived independently of every engine: sanity/model.py (its header holds the
// derivation; its output sanity/model.json is read here, python is not run): light shifts from a sum over states on the
// app's atomic tables, and a classical-atom Monte Carlo in two independent formulations (K: scattering events at the
// two-level steady-state rate at the local light-shifted, Doppler-shifted detuning with dipole-pattern kicks; F: the
// Gordon–Ashkin Fokker–Planck coefficients from the optical Bloch equations, with the standing wave's interference).
// The expectation is their mean; their difference and statistical errors are its uncertainty.
// Physics' verdict on the scenes (model.json): S1 heats and loses the atom (half lost at 1.84 ms, 97 % at 5 ms); S2 as the
// app builds it is a pure σ+ standing wave on a closed two-level cycle -- Doppler cooling along z only (the light-shifted
// line puts the probe at −3.8Γ, T_D ≈ 410 µK, half the trap depth), x and y heated by spontaneous emission at twice
// S1's photon rate, the repump idle -- so it loses the atom FASTER than S1 (half at 1.17 ms), not slower.
// S2Q is judged against model.py's multilevel rate model (quantum jumps over the 8 ground sublevels, each beam's rate on
// each |F',m'⟩ at its own light-shifted, Zeeman-shifted, Doppler-shifted detuning, dipole branching; no ground
// coherences: the 3 G field dephases them -- Larmor 2.1 MHz per Δm against a ground light shift of ~0.4 MHz -- which
// also moves the σ+σ− sub-Doppler resonance to v = ω_L/k ≈ 1.6 m/s, far above the atoms' 0.05–0.3 m/s). The atom is
// pumped mostly into |2,−2⟩ (56 %), whose σ− line the 3 G field puts 8.4 MHz nearer than σ+'s: −2.45Γ at the centre,
// +0.69Γ (blue: heating, pushing) at the trap's edge. Rate model: half lost at 0.52 ms. Bounds for whatever the
// model leaves out along z (sub-Doppler of either sign): z frozen (the best any z-only cooling can do: half at 2.99 ms)
// and z without friction (half at 0.45 ms). x and y have no cooling beam in any variant.
//
// Criteria (tolerances: the user's accuracy targets -- loss 0.05 absolute, T 10 % -- plus twice the combined standard
// error of engine and model; they exceed the model's systematic uncertainty, ≤ 0.01 in L and ≤ 2 % in T):
//   run      the solve completes in the 5-minute limit, the app's reader accepts every checkpoint and the result;
//   trap     ω/2π along x (axial) and y (radial) within 3 % of the tweezer's paraxial Gaussian (the near light's
//            softening ≤ 2 %);
//   closed   the held atom's population outside |2,2⟩ and |3',3'⟩ ≤ 0.01 while L ≤ 0.9 (physics ≲ 1e-4; the app's tolerance);
//   heating  T_total at 0.25, 0.5, 1, 2 ms (while the model keeps ≥ 10 % of the atoms) within 10 % + 2σ;
//   loss     L at 1, 2, 3, 4, 5 ms within 0.05 + 2σ;
//   lost     L at the end ≥ 0.9 (both scenes: model 0.97 and 1.00);
//   S2 vs S1 t½(S1)/t½(S2) in [1.25, 2.0] (model 1.57; each t½ is known to ~7 % from the loss tolerance);
//   S2Q      loss at 0.25, 0.5, 1, 2, 3, 5 ms inside the band [z frozen − tol, z frictionless + tol] (tol 0.05 + 2σ; a loose
//            criterion, as wide as the model's ignorance along z); T_total at 0.25, 0.5 ms inside [min − 10 %, max + 10 %]
//            of the three variants; the held atom's |2,−2⟩ and |2,2⟩ shares at 0.25 and 0.5 ms within 0.05 + 2σ of the
//            rate model's (insensitive to the z variants: they agree to 0.03).
// A run shorter than 5 ms (--duration) is judged on the times it reaches. The script ends with exit code 1 on any FAIL.
// --old: the previous release's worker (old request format: no outside forms, beams without electricField; its result
// read raw). Run under the guard (one heavy job at a time); one scene takes seconds to minutes.
import {spawn} from 'node:child_process';
import {existsSync,readFileSync,writeFileSync,statSync,readdirSync} from 'node:fs';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';

const here=dirname(fileURLToPath(import.meta.url));
const args=process.argv.slice(2),opt=name=>args.find(a=>a.startsWith(`--${name}=`))?.slice(name.length+3),flag=name=>args.includes(`--${name}`);
const exe=opt('worker');
if(!exe){console.error('usage: node sanity.mjs --worker=<exe> [--scenes=S1,S2,S2Q,S3] [--duration=5000] [--old] [--app=<dir>] [--save=<file>]');process.exit(2);}
const app=resolve(opt('app')??resolve(here,'../../..')),old=flag('old'),durationUs=Number(opt('duration')??5000),wanted=(opt('scenes')??'S1,S2,S2Q').split(',');
const from=path=>import(pathToFileURL(join(app,path)).href);
/** The app's dist must be newer than its sources (node scripts/build.mjs). */
const newest=folder=>Math.max(...readdirSync(folder,{withFileTypes:true}).map(e=>e.isDirectory()?newest(join(folder,e.name)):statSync(join(folder,e.name)).mtimeMs));
if(statSync(join(app,'dist/js/atom-workbench/src/types.js')).mtimeMs<newest(join(app,'src'))){console.error(`the app's dist is older than its sources: node scripts/build.mjs in ${app}`);process.exit(2);}
const types=await from('dist/js/atom-workbench/src/types.js'),physics=await from('dist/js/atom-workbench/src/physics.js');
const reflection=await from('dist/js/atom-workbench/src/reflection.js'),intensity=await from('dist/js/atom-workbench/src/laser-intensity.js');
const spectroscopy=await from('dist/js/atom-workbench/src/spectroscopy.js'),sceneFile=await from('dist/js/atom-workbench/src/scene-file.js');
const {motionRequest}=await from('dist/js/atom-workbench/src/motion.js'),{normalizeDynamicsResult}=await from('dist/js/atom-workbench/src/solver-client.js');
const {NATIVE_PROTOCOL,FULL_MODEL_SCHEMA}=await from('dist/js/atom-workbench/src/solver-protocol.js');
const {prepareNativeSolve,nativeCapabilities}=await from('scripts/native-protocol.mjs'),{nativeChildEnvironment}=await from('scripts/native-child-env.mjs');
const model=JSON.parse(readFileSync(join(here,'sanity/model.json'),'utf8'));

// ---------------------------------------------------------------------------------------------------------------------
// The scenes, as the operator makes them in the app.
const SAMPLE_US=50;
function s1(){
  const scene=types.defaultScene();
  // The default probe (F=2 → F'=2, 1 mW/cm²) deleted; a new laser added with ＋, made σ+ along the field by the preset.
  const tweezer=scene.lasers.find(l=>l.id===scene.tweezerId),probe=types.newLaser('laser-probe','激光 1','D2');
  scene.lasers=[tweezer,probe];
  physics.setPolarizationPreset(probe,scene.fields,'sigma-plus');
  Object.assign(scene.dynamics,{initialStateId:'ground-2-2',durationUs,durationUnit:'μs',samples:Math.round(durationUs/SAMPLE_US)+1});
  delete scene.dynamics.initialState;
  scene.name='sanity S1: 光镊 + σ+ 探测光';
  return scene;
}
function s2(quarterWave=false){
  const scene=s1(),probe=scene.lasers.find(l=>l.id==='laser-probe');
  physics.resolveSceneFrequencies(scene);
  const back=reflection.addReflection(scene,probe);
  if(quarterWave)back.reflection.quarterWave=true;
  const repump=types.newLaser('laser-repump','激光 2','D2');
  repump.frequency={base:{kind:'transition',line:'D2',Fg:1,Fe:2},offsetMHz:0};
  scene.lasers.push(repump);
  scene.name=`sanity S2${quarterWave?'Q':''}: 光镊 + σ+ 探测光 + 反射${quarterWave?'（λ/4）':''} + 回泵光`;
  return scene;
}
/** S3, the user's experiment as far as it is known (report mode): S1's tweezer (along +x), field and initial state; a
 * probe added with ＋, in the lab's geometry across the tweezer -- along +y, tilted 4° toward its axis (the user: roughly
 * perpendicular, a few degrees toward the axis; the ＋ default runs along the tweezer's own axis) -- and set π (the app's
 * π preset), 1 mW in a 1 mm waist (63.7 mW/cm²), its F=2 → F'=3 transition's offset −20 MHz as the operator sets it (the user:
 * about 20 MHz red of the free-space line); the tweezer's power raised to 7.25 mW, its depth 1.5 mK as the user gives it
 * (the default 4 mW at 852 nm and 1.2 µm makes 828 µK, the depth linear in the power); its reflection by the app's
 * mirror with the λ/4 plate at its default fast axis (45°: π comes back ⊥, σ+ and σ− half each about the field --
 * lin⊥lin); S2's repump. Its own duration (S3_US, or --duration) and samples (S3_SAMPLES). */
const S3_US=100_000,S3_SAMPLES=201,S3_TILT_DEG=4,S3_OFFSET_MHZ=-20,S3_TWEEZER_MW=7.25;
function s3(){
  const scene=types.defaultScene();
  const tweezer=scene.lasers.find(l=>l.id===scene.tweezerId),probe=types.newLaser('laser-probe','激光 1','D2');
  scene.lasers=[tweezer,probe];
  // The π preset points a probe across the field along +x; the lab's direction after it, its linear polarization turned
  // to the field (π: 90° in this direction's frame, checked below as the scene's polarization).
  physics.setPolarizationPreset(probe,scene.fields,'pi');
  Object.assign(probe,{direction:{reference:{kind:'axis',axis:'+y'},thetaDeg:S3_TILT_DEG,phiDeg:0},polarizationAngleDeg:90});
  Object.assign(probe,{powerMW:1,waistUm:1000});
  probe.frequency.offsetMHz=S3_OFFSET_MHZ;
  tweezer.powerMW=S3_TWEEZER_MW;
  physics.resolveSceneFrequencies(scene);
  reflection.addReflection(scene,probe).reflection.quarterWave=true;
  const repump=types.newLaser('laser-repump','激光 2','D2');
  repump.frequency={base:{kind:'transition',line:'D2',Fg:1,Fe:2},offsetMHz:0};
  scene.lasers.push(repump);
  const duration=Number(opt('duration')??S3_US);
  Object.assign(scene.dynamics,{initialStateId:'ground-2-2',durationUs:duration,durationUnit:'μs',samples:S3_SAMPLES});
  delete scene.dynamics.initialState;
  scene.name='sanity S3: 光镊 + π 探测光 1 mW + 反射（λ/4 45°）+ 回泵光';
  return scene;
}
/** Resolved and read back through the app's importer (a saved file opened). */
function finish(scene){physics.resolveSceneFrequencies(scene);scene.lasers.forEach(intensity.normalizeLaserIntensity);return sceneFile.validateScene(JSON.parse(JSON.stringify(scene)));}
const SCENES={S1:()=>finish(s1()),S2:()=>finish(s2()),S2Q:()=>finish(s2(true)),S3:()=>finish(s3())};

/** What the app makes of a scene: polarizations, the levels chart's light shifts of the tweezer alone (all-order Floquet,
 * spectroscopy.ts) and the detunings at the trap centre. */
function facts(scene){
  const a=physics.analyze(scene),s=JSON.parse(JSON.stringify(scene));for(const l of s.lasers)l.showLightShift=l.id===scene.tweezerId;
  const chart=spectroscopy.spectroscopyAnalysis(s,undefined,false),levels=Object.fromEntries([...chart.ground,...chart.D1,...chart.D2].map(l=>[l.id,l]));
  const abs=id=>physics.absoluteEnergyMHz(levels[id]),bare=id=>abs(id)-levels[id].starkShiftMHz;
  const lasers=scene.lasers.map(l=>{const p=a.polarizations[l.id];return {id:l.id,name:l.name,k:p.k,sigmaMinus:p.minus,pi:p.pi,sigmaPlus:p.plus,intensityMWcm2:intensity.laserIntensityMWcm2(l),MHz:physics.laserFrequencyMHz(l),reflection:l.reflection}});
  const probe=lasers.find(l=>l.id==='laser-probe'),repump=lasers.find(l=>l.id==='laser-repump');
  const out={status:chart.spectroscopy?.status,lasers,groundShiftMHz:levels['ground-2-2'].starkShiftMHz,excitedShiftMHz:levels['D2-3-3'].starkShiftMHz,
    probeCentreMHz:probe.MHz-(abs('D2-3-3')-abs('ground-2-2')),probeFreeMHz:probe.MHz-(bare('D2-3-3')-bare('ground-2-2'))};
  if(repump)out.repumpCentreMHz=[-1,0,1].map(m=>({m,MHz:repump.MHz-(abs(`D2-2-${m}`)-abs(`ground-1-${m}`))}));
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// The worker, driven as the app drives it (scripts/serve.mjs, src/solver-client.ts): capabilities, warm-up, a trajectory.
const LIMIT_MS=290_000;
function environment(){
  const env=nativeChildEnvironment(process.env),beside=join(dirname(resolve(exe)),'libscipy_openblas.dll');
  const tools='C:/Users/eadri/Dropbox/WorkCode/Github/Zou_lab_control_v3/.tools/science-env/Lib/site-packages/scipy.libs/libscipy_openblas-64eda39e79589aedb16f58e5547eb599.dll';
  if(!env.ATOM_OPENBLAS_DLL)env.ATOM_OPENBLAS_DLL=existsSync(beside)?beside:tools;
  return env;
}
function worker(){
  const child=spawn(resolve(exe),[],{stdio:['pipe','pipe','pipe'],windowsHide:true,env:environment()});
  child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
  let buffer='',stderr='',waiting=null;
  child.stderr.on('data',t=>{stderr+=t;if(stderr.length>200000)stderr=stderr.slice(-100000);});
  child.stdout.on('data',t=>{buffer+=t;for(let end;(end=buffer.indexOf('\n'))>=0;){
    const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(!line.trim())continue;
    const message=JSON.parse(line),job=waiting;if(!job)continue;
    if(message.progress!==undefined){try{job.onCheckpoint?.(message);}catch(e){job.reject(e);}continue;}
    waiting=null;clearTimeout(job.timer);job.resolve({message,ms:performance.now()-job.sent});
  }});
  child.on('close',code=>{if(waiting){clearTimeout(waiting.timer);waiting.reject(new Error(`the worker ended (exit code ${code}): ${stderr.trim().slice(-300)}`));}});
  return {child,
    ask:(message,onCheckpoint)=>new Promise((accept,reject)=>{
      const job={resolve:accept,reject,onCheckpoint,sent:performance.now(),timer:setTimeout(()=>{child.kill();reject(new Error(`no reply in ${LIMIT_MS/1000} s: aborted (the 5-minute rule)`));},LIMIT_MS)};
      waiting=job;child.stdin.write(JSON.stringify(message)+'\n');}),
    end:()=>new Promise(done=>{if(child.exitCode!==null)return done();child.once('close',done);child.stdin.end();setTimeout(()=>child.kill(),2000).unref();})};
}
/** The old engine's request: what it knows of the motion block. */
function oldFormat(payload){
  const p=structuredClone(payload),keep=['id','name','direction','waveNumberPerUm','waistUm','outsideShift','intensityMWcm2','frequencyIndex','coupling','counterCoupling'];
  delete p.model.motion.outsideForms;p.model.motion.beams=p.model.motion.beams.map(b=>Object.fromEntries(Object.entries(b).filter(([k])=>keep.includes(k))));
  return p;
}
/** A result's motion series, from the app's reader (new) or raw (--old). */
function series(result,raw,states){
  if(result){const m=result.motion;
    return {times:result.timesUs,T:m.temperatureUK,TSE:m.temperatureErrorUK,Ttotal:m.temperatureTotalUK,TtotalSE:m.temperatureTotalErrorUK,loss:m.lossProbability,
      trajectories:m.trajectories,stepUs:m.timeStepUs,trapKHz:m.trapFrequencyKHz,flagged:m.approximations?.flagged??'',note:m.approximations?.note??'',rows:raw.populations,states};}
  const m=raw.motion,axes=['x','y','z'],ax=v=>v&&!Array.isArray(v)?axes.map(a=>v[a]):v;
  const T=ax(m.temperatureUK),TSE=ax(m.temperatureErrorUK)??T.map(a=>a.map(()=>null));
  const Ttotal=m.temperatureTotalUK??T[0].map((_,k)=>(T[0][k]+T[1][k]+T[2][k])/3);
  return {times:raw.timesUs,T,TSE,Ttotal,TtotalSE:m.temperatureTotalErrorUK??Ttotal.map(()=>null),loss:m.lossProbability??Ttotal.map(()=>0),
    trajectories:m.trajectories,stepUs:m.timeStepUs,trapKHz:ax(m.trapFrequencyKHz),flagged:'',note:'',rows:raw.populations,states};
}
async function solve(scene,name){
  const w=worker(),started=performance.now(),settings=scene.dynamics,motion=motionRequest(settings);
  const options={initialStateId:settings.initialStateId,initialState:types.canonicalInitialState(types.resolveInitialState(settings)),durationUs:settings.durationUs,samples:settings.samples,
    populationTolerance:types.POPULATION_TOLERANCE,meanHorizonUs:settings.meanHorizonUs,includeMean:false,meanOnly:false,motion,device:settings.motion.device};
  const native=prepareNativeSolve({modelSchema:FULL_MODEL_SCHEMA,scene,options}),token={clientId:'sanity',sequence:2};
  const read=r=>normalizeDynamicsResult(native.respond(r,token),scene,types.resolveInitialState(settings),settings.meanHorizonUs,motion);
  // A checkpoint the app's reader rejects is recorded (the app would show "求解结果不完整或无效") and the run read on.
  let checkpoints=0;const rejected=[];
  try{
    if(!old){
      nativeCapabilities((await w.ask({op:'capabilities',id:0})).message);
      const warm=await w.ask({id:-1,op:'warmup'});
      if(warm.message.error||warm.message.ready!==true||warm.message.protocol!==NATIVE_PROTOCOL)throw new Error(`warmup: ${JSON.stringify(warm.message).slice(0,200)}`);
    }
    const t0=performance.now();
    const reply=await w.ask({id:2,...(old?oldFormat(native.payload):native.payload),progress:true},m=>{checkpoints++;if(!old)try{read(m.result);}catch(e){rejected.push(`checkpoint ${checkpoints}: ${e.message}`);}});
    const solveS=(performance.now()-t0)/1000;
    if(reply.message.error)return {name,error:`the worker answered an error: ${reply.message.error}`,solveS,checkpoints};
    const raw=reply.message.result;
    return {name,solveS,checkpoints,rejected,engine:raw.diagnostics?.engine,...series(old?null:read(raw),raw,raw.states??native.payload.model.states)};
  }catch(error){return {name,error:error.message,solveS:(performance.now()-started)/1000,checkpoints};}
  finally{await w.end();}
}

// ---------------------------------------------------------------------------------------------------------------------
// The expectation and the criteria.
/** The model runs a scene's expectation is made of (model.json): S1, S2 the mean of the two formulations; S2Q the rate
 * model, with the band of its z variants. */
const EXPECT={S1:{central:['S1K','S1F']},S2:{central:['S2K','S2F']},S2Q:{central:['S2QR'],low:'S2QZ',high:'S2QV'}};
/** The model's expectation at sample k: the mean of the central runs; u = half their difference ⊕ their standard errors. */
function expectation(scene,k){
  const runs=EXPECT[scene].central.map(id=>model.runs[id]);
  const one=(key,seKey)=>{const xs=runs.map(r=>({v:r[key][k],se:r[seKey]?.[k]??0}));if(xs.some(x=>x.v==null))return null;
    const v=xs.reduce((s,x)=>s+x.v,0)/xs.length,spread=xs.length>1?Math.abs(xs[0].v-xs[1].v)/2:0;
    return {v,u:Math.hypot(spread,Math.hypot(...xs.map(x=>x.se))/xs.length)};};
  return {t:runs[0].timesUs[k],loss:one('loss','lossSE'),T:one('Ttotal','TtotalSE')};
}
const half=(t,L)=>{const i=L.findIndex(x=>x>=0.5);return i>0?t[i-1]+(0.5-L[i-1])*(t[i]-t[i-1])/(L[i]-L[i-1]):NaN;};
const modelHalf=scene=>{const runs=EXPECT[scene].central.map(id=>model.runs[id]);return half(runs[0].timesUs,runs[0].loss.map((_,k)=>runs.reduce((s,r)=>s+r.loss[k],0)/runs.length));};
const ok=b=>b?'PASS':'FAIL';
function judge(sceneId,f,r){
  const out=[],add=(pass,what,detail)=>out.push({pass,what,detail});
  if(r.error){add(false,'run',r.error);return out;}
  add(true,'run',`completed in ${r.solveS.toFixed(1)} s, ${r.checkpoints} checkpoint(s)`);
  if(!old)add(!r.rejected.length,'run: the app\'s reader accepts every checkpoint and the result',r.rejected.length?`${r.rejected.length} rejected -- ${r.rejected[0]} (the app shows this error)`:'');
  const finite=[...r.Ttotal,...r.loss].every(x=>x===null||Number.isFinite(x));add(finite,'run: numbers finite','');
  if(sceneId==='S3')return out;
  const p=model.parameters,wx=p.omegaAx/2/Math.PI*1e3,wy=p.omegaR/2/Math.PI*1e3;
  if(r.trapKHz){add(Math.abs(r.trapKHz[0]/wx-1)<=0.03,'trap x (axial)',`${r.trapKHz[0].toFixed(2)} kHz vs ${wx.toFixed(2)} ± 3 %`);
    add(Math.abs(r.trapKHz[1]/wy-1)<=0.03,'trap y (radial)',`${r.trapKHz[1].toFixed(2)} kHz vs ${wy.toFixed(2)} ± 3 %`);}
  const N=r.trajectories,index=t=>r.times.findIndex(x=>Math.abs(x-t)<1e-6),stateIndex=id=>r.states.findIndex(s=>(s.id??s)===id);
  if(sceneId==='S2Q')return judgeS2Q(r,add,N,index,stateIndex),out;
  if(r.rows){
    // Time-averaged while ≥ 10 % of the atoms are held: a single sample is granular (one depumped trajectory is 1/N, or
    // 2/N for a pair), its dwell outside the cycle ~10–20 µs. Physics (model.json multilevel.depumpingS2): the repump's
    // off-resonant excitation of |3',2'⟩, |2',2'⟩ from 6.5–6.8 GHz away, 2.1/s to |2,1⟩ and 1.2/s to |1,1⟩ at the centre,
    // back within ~20 µs: ~4e-5; S1 has no path (0). Limit 1e-3: 25× the physical rate.
    const keepIdx=['ground-2-2','D2-3-3'].map(stateIndex),ks=r.rows.map((_,k)=>k).filter(k=>k>0&&r.loss[k]<=0.9);
    const other=k=>1-keepIdx.reduce((s,i)=>s+r.rows[k][i],0),avg=ks.reduce((s,k)=>s+other(k),0)/Math.max(1,ks.length),kw=ks.reduce((a,k)=>other(k)>other(a)?k:a,ks[0]??0);
    const top=r.rows[kw].map((p,i)=>({id:r.states[i].id??r.states[i],p})).filter((_,i)=>!keepIdx.includes(i)).sort((a,b)=>b.p-a.p).slice(0,2);
    add(avg<=1e-3,'closed cycle',`population outside |2,2⟩, |3',3'⟩ of the held atom, time average ${avg.toExponential(2)} (largest sample ${other(kw).toExponential(1)} at ${(r.times[kw]/1e3).toFixed(2)} ms: ${top.map(s=>`${s.id} ${s.p.toExponential(1)}`).join(', ')}); physics ${sceneId==='S2'?'~4e-5':'0'}, limit 1e-3`);}
  for(const t of [250,500,1000,2000]){const k=index(t),e=expectation(sceneId,Math.round(t/SAMPLE_US));
    if(k<0||!e.T||e.loss.v>0.9)continue;
    const se=r.TtotalSE[k]??0,tol=0.1*e.T.v+2*Math.hypot(se,e.T.u),T=r.Ttotal[k];
    add(T!=null&&Math.abs(T-e.T.v)<=tol,`heating: T_total(${t/1e3} ms)`,`${T?.toFixed(1)} ± ${se?.toFixed(1)} µK vs ${e.T.v.toFixed(1)} ± ${tol.toFixed(1)} (model ± ${e.T.u.toFixed(1)})`);}
  for(const t of [1000,2000,3000,4000,5000]){const k=index(t);if(k<0)continue;const e=expectation(sceneId,Math.round(t/SAMPLE_US)),L=r.loss[k];
    const se=Math.sqrt(Math.max(L*(1-L),1/N)/N),tol=0.05+2*Math.hypot(se,e.loss.u);
    add(Math.abs(L-e.loss.v)<=tol,`loss: L(${t/1e3} ms)`,`${L.toFixed(3)} ± ${se.toFixed(3)} vs ${e.loss.v.toFixed(3)} ± ${tol.toFixed(3)}`);}
  const kEnd=r.times.length-1,e=expectation(sceneId,Math.round(r.times[kEnd]/SAMPLE_US));
  if(r.times[kEnd]>=5000-1e-6)add(r.loss[kEnd]>=0.9,'lost: L(end) ≥ 0.9',`${r.loss[kEnd].toFixed(3)} at ${r.times[kEnd]/1e3} ms (model ${e.loss.v.toFixed(3)})`);
  return out;
}
/** S2Q: the loss and T inside the band of the rate model's z variants; the held atom's ground populations the rate model's. */
function judgeS2Q(r,add,N,index,stateIndex){
  const {low,high}=EXPECT.S2Q,lo=model.runs[low],hi=model.runs[high],mid=model.runs.S2QR;
  for(const t of [250,500,1000,2000,3000,5000]){const k=index(t),j=Math.round(t/SAMPLE_US);if(k<0)continue;
    const L=r.loss[k],se=Math.sqrt(Math.max(L*(1-L),1/N)/N),a=lo.loss[j]-0.05-2*Math.hypot(se,lo.lossSE[j]),b=hi.loss[j]+0.05+2*Math.hypot(se,hi.lossSE[j]);
    add(L>=a&&L<=b,`loss: L(${t/1e3} ms) in the band`,`${L.toFixed(3)} ± ${se.toFixed(3)} vs [${Math.max(0,a).toFixed(3)}, ${Math.min(1,b).toFixed(3)}] (z frozen ${lo.loss[j].toFixed(3)}, rate model ${mid.loss[j].toFixed(3)}, z frictionless ${hi.loss[j].toFixed(3)})`);}
  for(const t of [250,500]){const k=index(t),j=Math.round(t/SAMPLE_US);if(k<0)continue;
    const vs=[lo,mid,hi].map(m=>m.Ttotal[j]).filter(v=>v!=null),T=r.Ttotal[k],se=r.TtotalSE[k]??0,a=0.9*Math.min(...vs)-2*se,b=1.1*Math.max(...vs)+2*se;
    add(T!=null&&T>=a&&T<=b,`heating: T_total(${t/1e3} ms) in the band`,`${T?.toFixed(1)} ± ${se.toFixed(1)} µK vs [${a.toFixed(1)}, ${b.toFixed(1)}] (z frozen ${vs[0]?.toFixed(1)}, rate model ${mid.Ttotal[j]?.toFixed(1)}, z frictionless ${hi.Ttotal[j]?.toFixed(1)})`);}
  if(r.rows)for(const t of [250,500]){const k=index(t),j=Math.round(t/SAMPLE_US);if(k<0)continue;
    // The engine's rows hold the excited states too: its ground shares are taken among the ground states (the rate
    // model's atom is always in a ground state between its jumps).
    const ground=mid.groundStates.map(stateIndex),sum=ground.reduce((s,i)=>s+r.rows[k][i],0),held=N*(1-r.loss[k]),heldModel=mid.atoms*(1-mid.loss[j]);
    for(const id of ['ground-2--2','ground-2-2']){const p=r.rows[k][stateIndex(id)]/sum,q=mid.groundPopulations[j][mid.groundStates.indexOf(id)];
      const tol=0.05+2*Math.hypot(Math.sqrt(p*(1-p)/Math.max(held,1)),Math.sqrt(q*(1-q)/Math.max(heldModel,1)));
      add(Math.abs(p-q)<=tol,`populations: ${id} of the held atom's ground states at ${t/1e3} ms`,`${p.toFixed(3)} vs ${q.toFixed(3)} ± ${tol.toFixed(3)}`);}}
}

/** S3 in report mode, beside the user's experiment (2026-11-12, verbatim: "我很确定我们的pi偏振的，差不多半径1mm, 功率1mW的
 * probe光，在带lambda/4反射镜反射之后，加上repump，对于光镊里面的原子的加热效果非常小，吹100ms也是基本吹不飞的，这是物理实验结果"):
 * the held share and T over the run; no criterion until the experiment's numbers (its tweezer, field and detunings) are
 * known -- the scene takes the app's defaults for them. */
function reportS3(r){
  console.log(`     t ms | T_total ± SE µK (x/y/z)                 | L ± SE`);
  for(const t of [0,500,1000,2000,5000,10000,20000,50000,100000]){const k=r.times.findIndex(x=>Math.abs(x-t)<1e-6);if(k<0)continue;
    const L=r.loss[k],se=Math.sqrt(Math.max(L*(1-L),1/r.trajectories)/r.trajectories);
    console.log(`   ${(t/1e3).toFixed(1).padStart(6)} | ${(r.Ttotal[k]==null?'—':`${r.Ttotal[k].toFixed(1)} ± ${(r.TtotalSE[k]??0).toFixed(1)}`).padEnd(16)} (${r.T.map(a=>a[k]==null?'—':a[k].toFixed(0)).join('/')})`.padEnd(54)+`| ${L.toFixed(3)} ± ${se.toFixed(3)}`);}
  const end=r.times.length-1;
  console.log(`   REPORT experiment: held over 100 ms ("基本吹不飞"), little heating; engine: held ${(1-r.loss[end]).toFixed(3)} at ${(r.times[end]/1e3).toFixed(1)} ms, T_total ${r.Ttotal[end]?.toFixed(1)??'—'} µK (from ${r.Ttotal[0]?.toFixed(1)} µK)`);
}

// ---------------------------------------------------------------------------------------------------------------------
console.log(`worker ${resolve(exe)}${old?' (old request format)':''}\napp    ${app}\nmodel  sanity/model.json (${model.generator}; ${Object.values(model.runs)[0].atoms} atoms per model)`);
const ls=model.lightShifts,summary={worker:resolve(exe),old,durationUs,scenes:{}};
let failures=0;const results={};
for(const id of wanted){
  const scene=SCENES[id](),f=facts(scene);
  console.log(`\n== ${id}: ${scene.name}; ${scene.dynamics.durationUs} µs, ${scene.dynamics.samples} samples, motion at ${scene.dynamics.motion.temperatureUK} µK (device ${scene.dynamics.motion.device}), initial ${scene.dynamics.initialStateId}`);
  for(const l of f.lasers)console.log(`   ${l.name.padEnd(10)} k ${l.k.map(x=>x.toFixed(0)).join(',').padEnd(7)} σ− ${l.sigmaMinus.toFixed(3)} π ${l.pi.toFixed(3)} σ+ ${l.sigmaPlus.toFixed(3)}  ${l.intensityMWcm2.toPrecision(4)} mW/cm²${l.reflection?`  mirror R ${l.reflection.reflectivity} φ ${l.reflection.phaseDeg}° λ/4 ${l.reflection.quarterWave}`:''}`);
  const both=f.lasers.filter(l=>l.id.startsWith('laser-probe'));
  if(both.length===2)console.log(`   the probe pair on the field's axis: ${both.map(l=>l.sigmaPlus>0.999?'σ+':l.sigmaMinus>0.999?'σ−':'mixed').join(' + ')} (${both[0].sigmaPlus>0.999&&both[1].sigmaPlus>0.999?'σ+σ+: a pure σ+ standing wave, no polarization gradient':'opposite circular: a polarization gradient'})`);
  // model.json's shifts are the default 4 mW tweezer's, linear in the power (S3's tweezer: S3_TWEEZER_MW)
  const power=id==='S3'?S3_TWEEZER_MW/4:1,lsg=ls.groundShiftMHz*power,lse=ls.excitedShiftMHz*power;
  console.log(`   tweezer's light shift at the focus (levels chart): |2,2⟩ ${f.groundShiftMHz.toFixed(4)} MHz (model ${lsg.toFixed(4)}), |3',3'⟩ ${f.excitedShiftMHz.toFixed(4)} MHz (model ${lse.toFixed(4)})`);
  console.log(`   probe detuning from |2,2⟩ → |3',3'⟩: ${f.probeCentreMHz.toFixed(3)} MHz at the trap centre (${(f.probeCentreMHz/(model.parameters.gamma/2/Math.PI)).toFixed(2)}Γ), ${f.probeFreeMHz.toFixed(3)} MHz without the tweezer (3 G Zeeman)`);
  if(f.repumpCentreMHz)console.log(`   repump detuning from F=1 → F'=2 (π) at the centre: ${f.repumpCentreMHz.map(x=>`m ${x.m}: ${x.MHz.toFixed(2)}`).join(', ')} MHz`);
  const sceneChecks=[[Math.abs(f.groundShiftMHz/lsg-1)<0.005&&Math.abs(f.excitedShiftMHz/lse-1)<0.005,'scene: the levels chart\'s shifts within 0.5 % of the independent sum over states'],
    [Math.abs(f.probeFreeMHz-model.chart.zeemanDetuningMHz-(id==='S3'?S3_OFFSET_MHZ:0))<1e-3,`scene: the probe's free detuning is the model's${id==='S3'?` with S3's offset ${S3_OFFSET_MHZ} MHz`:''}`],
    [f.lasers.filter(l=>l.id.startsWith('laser-probe')).every(l=>id==='S3'?(l.id.endsWith('reflection')?Math.abs(l.sigmaMinus-0.5)<1e-3&&Math.abs(l.sigmaPlus-0.5)<1e-3:l.pi>0.999):id==='S2Q'&&l.id.endsWith('reflection')?l.sigmaMinus>0.999:l.sigmaPlus>0.999),'scene: the probe beams\' polarization as described']];
  for(const [pass,what] of sceneChecks){console.log(`   ${ok(pass)} ${what}`);if(!pass)failures++;}
  const r=await solve(scene,id);results[id]=r;
  if(r.error)console.log(`   engine: ERROR after ${r.solveS?.toFixed(1)} s: ${r.error}`);
  else{
    console.log(`   engine ${r.engine??''}: solve ${r.solveS.toFixed(1)} s, ${r.trajectories} trajectories, step ${r.stepUs} µs, trap ${r.trapKHz?.map(x=>x.toFixed(2)).join('/')} kHz (model ${(model.parameters.omegaAx/2/Math.PI*1e3).toFixed(2)}/${(model.parameters.omegaR/2/Math.PI*1e3).toFixed(2)}/${(model.parameters.omegaR/2/Math.PI*1e3).toFixed(2)} bare)`);
    if(r.flagged)console.log(`   flagged: ${r.flagged.slice(0,300)}`);
  }
  if(!r.error&&id==='S3')reportS3(r);
  else if(!r.error){
    const band=EXPECT[id].low?[model.runs[EXPECT[id].low],model.runs[EXPECT[id].high]]:null;
    console.log(`     t ms | T_total engine (x/y/z)          | model T  | L engine | model L${band?' (band: z frozen – z frictionless)':''}`);
    for(const t of [0,250,500,1000,1500,2000,3000,4000,5000]){const k=r.times.findIndex(x=>Math.abs(x-t)<1e-6);if(k<0)continue;const j=Math.round(t/SAMPLE_US),e=expectation(id,j);
      console.log(`   ${(t/1e3).toFixed(2).padStart(5)} | ${(r.Ttotal[k]==null?'—':`${r.Ttotal[k].toFixed(1)} ± ${(r.TtotalSE[k]??0).toFixed(1)}`).padEnd(12)} (${r.T.map(a=>a[k]==null?'—':a[k].toFixed(0)).join('/')})`.padEnd(46)+`| ${e?.T?e.T.v.toFixed(1).padStart(7):'      —'}  | ${r.loss[k].toFixed(3)}    | ${e?e.loss.v.toFixed(3):'—'}${band?` (${band[0].loss[j].toFixed(3)} – ${band[1].loss[j].toFixed(3)})`:''}`);}
    const th=half(r.times,r.loss);console.log(`   t½ engine ${Number.isFinite(th)?(th/1e3).toFixed(2)+' ms':'not reached'}, model ${(modelHalf(id)/1e3).toFixed(2)} ms${band?` (band ${(half(band[1].timesUs,band[1].loss)/1e3).toFixed(2)}–${(half(band[0].timesUs,band[0].loss)/1e3).toFixed(2)} ms)`:''}`);
  }
  const checks=judge(id,f,r);
  for(const c of checks){console.log(`   ${ok(c.pass)} ${c.what}${c.detail?`: ${c.detail}`:''}`);if(!c.pass)failures++;}
  summary.scenes[id]={facts:f,run:r.error?{error:r.error,solveS:r.solveS}:{solveS:r.solveS,trajectories:r.trajectories,stepUs:r.stepUs,trapKHz:r.trapKHz,flagged:r.flagged,rejected:r.rejected,times:r.times,Ttotal:r.Ttotal,TtotalSE:r.TtotalSE,T:r.T,loss:r.loss},checks};
}
if(results.S1&&results.S2&&!results.S1.error&&!results.S2.error){
  const a=half(results.S1.times,results.S1.loss),b=half(results.S2.times,results.S2.loss),ratio=a/b,pass=Number.isFinite(ratio)&&ratio>=1.25&&ratio<=2.0;
  console.log(`\n${ok(pass)} S2 vs S1: t½(S1)/t½(S2) = ${Number.isFinite(ratio)?ratio.toFixed(2):'undefined'} (model 1.57, accepted 1.25–2.0: S2 as built loses the atom faster than S1)`);
  if(!pass)failures++;summary.ratio=ratio;
}
if(results.S1&&results.S2Q&&!results.S1.error&&!results.S2Q.error){
  const a=half(results.S1.times,results.S1.loss),b=half(results.S2Q.times,results.S2Q.loss);
  console.log(`\n   S2Q vs S1 (information): t½(S1)/t½(S2Q) = ${(a/b).toFixed(2)}; physics: rate model ${(modelHalf('S1')/modelHalf('S2Q')).toFixed(2)}, and even with z frozen ${(modelHalf('S1')/half(model.runs.S2QZ.timesUs,model.runs.S2QZ.loss)).toFixed(2)} -- the λ/4 mirror does not make S2 slow either`);
}
if(opt('save'))writeFileSync(opt('save'),JSON.stringify(summary));
console.log(`\n${failures?`FAILED: ${failures} criteria`:'PASS'}`);
process.exit(failures?1:0);
