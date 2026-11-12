// node sanity.mjs --worker=<atom-lindblad.exe> [--scenes=S1,S2] [--duration=5000] [--old] [--app=<apps/atom-workbench>]
//                 [--save=<summary.json>]
//
// The two most basic motion scenes, built exactly as the app builds them and solved by a worker over its JSONL protocol
// as the app drives it, judged against physics (the user, 2026-11-11: "最最基础的两个场景……这些基本的sanity check得过啊"):
//   S1  the app's default tweezer (852 nm, 4 mW, w0 1.2 µm, along +x, π) and one probe added with ＋ (types.ts newLaser:
//       50 µW in a 1 mm waist, 3.18 mW/cm²) set σ+ along the bias field (the app's σ+ preset), at the free-space F=2 → F'=3
//       frequency (the default probe of the default scene removed); initial |F=2, m=2⟩, motion at 30 µK (the defaults);
//   S2  S1 + the probe's reflection by the app's mirror (reflection.ts addReflection, default mirror: R 1, φ 0, no λ/4)
//       + a repump added with ＋ and set to F=1 → F'=2 (along +x, π);
//   S2Q (information only, no criteria) S2 with the mirror's λ/4 plate (σ+σ− about the field: not a two-level problem).
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
//   S2 vs S1 t½(S1)/t½(S2) in [1.25, 2.0] (model 1.57; each t½ is known to ~7 % from the loss tolerance).
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
if(!exe){console.error('usage: node sanity.mjs --worker=<exe> [--scenes=S1,S2] [--duration=5000] [--old] [--app=<dir>] [--save=<file>]');process.exit(2);}
const app=resolve(opt('app')??resolve(here,'../../..')),old=flag('old'),durationUs=Number(opt('duration')??5000),wanted=(opt('scenes')??'S1,S2').split(',');
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
/** Resolved and read back through the app's importer (a saved file opened). */
function finish(scene){physics.resolveSceneFrequencies(scene);scene.lasers.forEach(intensity.normalizeLaserIntensity);return sceneFile.validateScene(JSON.parse(JSON.stringify(scene)));}
const SCENES={S1:()=>finish(s1()),S2:()=>finish(s2()),S2Q:()=>finish(s2(true))};

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
  let checkpoints=0;
  try{
    if(!old){
      nativeCapabilities((await w.ask({op:'capabilities',id:0})).message);
      const warm=await w.ask({id:-1,op:'warmup'});
      if(warm.message.error||warm.message.ready!==true||warm.message.protocol!==NATIVE_PROTOCOL)throw new Error(`warmup: ${JSON.stringify(warm.message).slice(0,200)}`);
    }
    const t0=performance.now();
    const reply=await w.ask({id:2,...(old?oldFormat(native.payload):native.payload),progress:true},m=>{checkpoints++;if(!old)read(m.result);});
    const solveS=(performance.now()-t0)/1000;
    if(reply.message.error)return {name,error:`the worker answered an error: ${reply.message.error}`,solveS,checkpoints};
    const raw=reply.message.result;
    return {name,solveS,checkpoints,engine:raw.diagnostics?.engine,...series(old?null:read(raw),raw,raw.states??native.payload.model.states)};
  }catch(error){return {name,error:error.message,solveS:(performance.now()-started)/1000,checkpoints};}
  finally{await w.end();}
}

// ---------------------------------------------------------------------------------------------------------------------
// The expectation and the criteria.
/** The model's expectation at sample k: the mean of K and F; u = half their difference ⊕ their standard errors. */
function expectation(scene,k){
  const [a,b]=[`${scene}K`,`${scene}F`].map(id=>model.runs[id]);
  const pick=(r,key,seKey)=>({v:r[key][k],se:r[seKey]?.[k]??0});
  const one=(key,seKey)=>{const x=pick(a,key,seKey),y=pick(b,key,seKey);if(x.v==null||y.v==null)return null;
    return {v:(x.v+y.v)/2,u:Math.hypot(Math.abs(x.v-y.v)/2,Math.hypot(x.se,y.se)/2)};};
  return {t:a.timesUs[k],loss:one('loss','lossSE'),T:one('Ttotal','TtotalSE')};
}
const half=(t,L)=>{const i=L.findIndex(x=>x>=0.5);return i>0?t[i-1]+(0.5-L[i-1])*(t[i]-t[i-1])/(L[i]-L[i-1]):NaN;};
const ok=b=>b?'PASS':'FAIL';
function judge(sceneId,f,r){
  const out=[],add=(pass,what,detail)=>out.push({pass,what,detail});
  if(r.error){add(false,'run',r.error);return out;}
  add(true,'run',`completed in ${r.solveS.toFixed(1)} s, ${r.checkpoints} checkpoint(s)${old?'':' all read by the app\'s reader'}`);
  const finite=[...r.Ttotal,...r.loss].every(x=>x===null||Number.isFinite(x));add(finite,'run: numbers finite','');
  const p=model.parameters,wx=p.omegaAx/2/Math.PI*1e3,wy=p.omegaR/2/Math.PI*1e3;
  if(r.trapKHz){add(Math.abs(r.trapKHz[0]/wx-1)<=0.03,'trap x (axial)',`${r.trapKHz[0].toFixed(2)} kHz vs ${wx.toFixed(2)} ± 3 %`);
    add(Math.abs(r.trapKHz[1]/wy-1)<=0.03,'trap y (radial)',`${r.trapKHz[1].toFixed(2)} kHz vs ${wy.toFixed(2)} ± 3 %`);}
  if(r.rows){
    // While at least 10 % of the atoms are held (the rest of the series is a few trajectories' noise).
    const keepIdx=['ground-2-2','D2-3-3'].map(id=>r.states.findIndex(s=>(s.id??s)===id));
    const other=r.rows.map((row,k)=>r.loss[k]<=0.9?1-keepIdx.reduce((s,i)=>s+row[i],0):0),worst=Math.max(...other),kw=other.indexOf(worst),at=r.times[kw];
    const top=r.rows[kw].map((p,i)=>({id:r.states[i].id??r.states[i],p})).filter((_,i)=>!keepIdx.includes(i)).sort((a,b)=>b.p-a.p).slice(0,3);
    add(worst<=0.01,'closed cycle',`largest population outside |2,2⟩, |3',3'⟩ of the held atom ${worst.toExponential(2)} at ${(at/1e3).toFixed(2)} ms while L ≤ 0.9 (${top.map(s=>`${s.id} ${s.p.toExponential(1)}`).join(', ')}; physics: off-resonant leaks only, ≲ 1e-4; limit 0.01, the app's population tolerance)`);}
  const N=r.trajectories,index=t=>r.times.findIndex(x=>Math.abs(x-t)<1e-6);
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

// ---------------------------------------------------------------------------------------------------------------------
console.log(`worker ${resolve(exe)}${old?' (old request format)':''}\napp    ${app}\nmodel  sanity/model.json (${model.generator}; ${Object.values(model.runs)[0].atoms} atoms per model)`);
const ls=model.lightShifts,summary={worker:resolve(exe),old,durationUs,scenes:{}};
let failures=0;const results={};
for(const id of wanted){
  const scene=SCENES[id](),f=facts(scene);
  console.log(`\n== ${id}: ${scene.name}; ${durationUs} µs, ${scene.dynamics.samples} samples, motion at ${scene.dynamics.motion.temperatureUK} µK (device ${scene.dynamics.motion.device}), initial ${scene.dynamics.initialStateId}`);
  for(const l of f.lasers)console.log(`   ${l.name.padEnd(10)} k ${l.k.map(x=>x.toFixed(0)).join(',').padEnd(7)} σ− ${l.sigmaMinus.toFixed(3)} π ${l.pi.toFixed(3)} σ+ ${l.sigmaPlus.toFixed(3)}  ${l.intensityMWcm2.toPrecision(4)} mW/cm²${l.reflection?`  mirror R ${l.reflection.reflectivity} φ ${l.reflection.phaseDeg}° λ/4 ${l.reflection.quarterWave}`:''}`);
  const both=f.lasers.filter(l=>l.id.startsWith('laser-probe'));
  if(both.length===2)console.log(`   the probe pair on the field's axis: ${both.map(l=>l.sigmaPlus>0.999?'σ+':l.sigmaMinus>0.999?'σ−':'mixed').join(' + ')} (${both[0].sigmaPlus>0.999&&both[1].sigmaPlus>0.999?'σ+σ+: a pure σ+ standing wave, no polarization gradient':'opposite circular: a polarization gradient'})`);
  console.log(`   tweezer's light shift at the focus (levels chart): |2,2⟩ ${f.groundShiftMHz.toFixed(4)} MHz (model ${ls.groundShiftMHz.toFixed(4)}), |3',3'⟩ ${f.excitedShiftMHz.toFixed(4)} MHz (model ${ls.excitedShiftMHz.toFixed(4)})`);
  console.log(`   probe detuning from |2,2⟩ → |3',3'⟩: ${f.probeCentreMHz.toFixed(3)} MHz at the trap centre (${(f.probeCentreMHz/(model.parameters.gamma/2/Math.PI)).toFixed(2)}Γ), ${f.probeFreeMHz.toFixed(3)} MHz without the tweezer (3 G Zeeman)`);
  if(f.repumpCentreMHz)console.log(`   repump detuning from F=1 → F'=2 (π) at the centre: ${f.repumpCentreMHz.map(x=>`m ${x.m}: ${x.MHz.toFixed(2)}`).join(', ')} MHz`);
  const sceneChecks=[[Math.abs(f.groundShiftMHz/ls.groundShiftMHz-1)<0.005&&Math.abs(f.excitedShiftMHz/ls.excitedShiftMHz-1)<0.005,'scene: the levels chart\'s shifts within 0.5 % of the independent sum over states'],
    [Math.abs(f.probeFreeMHz-model.chart.zeemanDetuningMHz)<1e-3,'scene: the probe\'s free detuning is the model\'s'],
    [f.lasers.filter(l=>l.id.startsWith('laser-probe')).every(l=>id==='S2Q'&&l.id.endsWith('reflection')?l.sigmaMinus>0.999:l.sigmaPlus>0.999),'scene: the probe beams\' polarization as described']];
  for(const [pass,what] of sceneChecks){console.log(`   ${ok(pass)} ${what}`);if(!pass)failures++;}
  const r=await solve(scene,id);results[id]=r;
  if(r.error)console.log(`   engine: ERROR after ${r.solveS?.toFixed(1)} s: ${r.error}`);
  else{
    console.log(`   engine ${r.engine??''}: solve ${r.solveS.toFixed(1)} s, ${r.trajectories} trajectories, step ${r.stepUs} µs, trap ${r.trapKHz?.map(x=>x.toFixed(2)).join('/')} kHz (model ${(model.parameters.omegaAx/2/Math.PI*1e3).toFixed(2)}/${(model.parameters.omegaR/2/Math.PI*1e3).toFixed(2)}/${(model.parameters.omegaR/2/Math.PI*1e3).toFixed(2)} bare)`);
    if(r.flagged)console.log(`   flagged: ${r.flagged.slice(0,300)}`);
    const hasModel=!!model.runs[`${id}K`];
    console.log(`     t ms | T_total engine (x/y/z)          | model T  | L engine | model L`);
    for(const t of [0,250,500,1000,1500,2000,3000,4000,5000]){const k=r.times.findIndex(x=>Math.abs(x-t)<1e-6);if(k<0)continue;const e=hasModel?expectation(id,Math.round(t/SAMPLE_US)):null;
      console.log(`   ${(t/1e3).toFixed(2).padStart(5)} | ${(r.Ttotal[k]==null?'—':`${r.Ttotal[k].toFixed(1)} ± ${(r.TtotalSE[k]??0).toFixed(1)}`).padEnd(12)} (${r.T.map(a=>a[k]==null?'—':a[k].toFixed(0)).join('/')})`.padEnd(46)+`| ${e?.T?e.T.v.toFixed(1).padStart(7):'      —'}  | ${r.loss[k].toFixed(3)}    | ${e?e.loss.v.toFixed(3):'—'}`);}
    const th=half(r.times,r.loss);console.log(`   t½ engine ${Number.isFinite(th)?(th/1e3).toFixed(2)+' ms':'not reached'}${hasModel?`, model ${(half(model.runs[`${id}K`].timesUs,model.runs[`${id}K`].loss.map((x,k)=>(x+model.runs[`${id}F`].loss[k])/2))/1e3).toFixed(2)} ms`:''}`);
  }
  if(model.runs[`${id}K`]){for(const c of judge(id,f,r)){console.log(`   ${ok(c.pass)} ${c.what}${c.detail?`: ${c.detail}`:''}`);if(!c.pass)failures++;}}
  else console.log('   (information only: no physics criteria for this scene)');
  summary.scenes[id]={facts:f,run:r.error?{error:r.error,solveS:r.solveS}:{solveS:r.solveS,trajectories:r.trajectories,stepUs:r.stepUs,trapKHz:r.trapKHz,flagged:r.flagged,times:r.times,Ttotal:r.Ttotal,TtotalSE:r.TtotalSE,T:r.T,loss:r.loss},
    checks:model.runs[`${id}K`]?judge(id,f,r):[]};
}
if(results.S1&&results.S2&&!results.S1.error&&!results.S2.error){
  const a=half(results.S1.times,results.S1.loss),b=half(results.S2.times,results.S2.loss),ratio=a/b,pass=Number.isFinite(ratio)&&ratio>=1.25&&ratio<=2.0;
  console.log(`\n${ok(pass)} S2 vs S1: t½(S1)/t½(S2) = ${Number.isFinite(ratio)?ratio.toFixed(2):'undefined'} (model 1.57, accepted 1.25–2.0: S2 as built loses the atom faster than S1)`);
  if(!pass)failures++;summary.ratio=ratio;
}
if(opt('save'))writeFileSync(opt('save'),JSON.stringify(summary));
console.log(`\n${failures?`FAILED: ${failures} criteria`:'PASS'}`);
process.exit(failures?1:0);
