// Motion (ui-design.md §6.20–§6.21): the tweezer the operator designates, the scene file, the model's
// motion block, the request with its device, the strict reader of a motion result (populations of the
// held atom with 丢失 beside them), the temperature chart and the examples.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as pause} from 'node:timers/promises';
import {DEFAULT_MOTION_TEMPERATURE_UK,NEW_LASER_POWER_MW,NEW_LASER_WAIST_UM,defaultMotion,defaultScene,newLaser} from '../dist/js/atom-workbench/src/types.js';
import {RB87,analyze,composeMagneticField,laserFrequencyMHz,laserIntensityMWcm2,polarization,resolveSceneFrequencies} from '../dist/js/atom-workbench/src/physics.js';
import {laserWaistUm,normalizeLaserIntensity,setLaserIntensityMode,writeLaserPower} from '../dist/js/atom-workbench/src/laser-intensity.js';
import {directionVector,invert,rotate} from '../dist/js/atom-workbench/src/direction.js';
import {MOTION_FRAME,motionRefusal,motionRequest,motionTemperatureReading,readMotionDevice,readMotionRequest,setMotionTemperature} from '../dist/js/atom-workbench/src/motion.js';
import {keepTweezer,setTweezer,tweezerOf,tweezerRefusal} from '../dist/js/atom-workbench/src/tweezer.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {EXAMPLES,EXAMPLE_DURATION_US,EXAMPLE_SAMPLES,EXAMPLE_TEMPERATURE_UK,F2_LARMOR_MHZ_PER_G,LAB,PGC1D,PGC3D,STARTUP_EXAMPLE,exampleScene} from '../dist/js/atom-workbench/src/examples.js';
import {meanInputsKey,trajectoryInputsKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {plannedSolve} from '../dist/js/atom-workbench/src/solve-plan.js';
import {DynamicsClient,normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {solverText} from '../dist/js/atom-workbench/src/solver-words.js';
import {exceedsNote,ledgerRows,outsideRuns,partialSentence,resultTolerance} from '../dist/js/atom-workbench/src/accuracy.js';
import {definitionNotes,modelNotes} from '../dist/js/atom-workbench/src/model-notes.js';
import {POPULATION_TOLERANCE} from '../dist/js/atom-workbench/src/types.js';
import {SEMICLASSICAL_POPULATION_TARGET} from '../dist/js/atom-workbench/src/solver-types.js';
import {typedDigits} from '../dist/js/atom-workbench/src/unit-input.js';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';
import {populationChart,populationColor} from '../dist/js/atom-workbench/src/population-chart.js';
import {LOSS_CURVE,recommendedCurves,withPopulationGroups} from '../dist/js/atom-workbench/src/population-selection.js';
import {TEMPERATURE_WINDOW,TOTAL_INK,TOTAL_WIDTH,lineReading,motionSeries,seriesChart} from '../dist/js/atom-workbench/src/motion-chart.js';
import {MOTION_DIAGNOSTICS,motionBlock,motionResult} from './native-result.mjs';
import {spherical} from './directions.mjs';
import {readFileSync,readdirSync} from 'node:fs';
import {chinese,english,filled,rustLiterals} from './rust-sentences.mjs';

const N=32,C=299792458;
const axis=name=>({reference:{kind:'axis',axis:name},thetaDeg:0,phiDeg:0});
/** The 3D PGC example: the 852 nm tweezer along z, σ+σ− cooling pairs along x, y and z (a beam and its λ/4 reflection each), a
 * repump, B = 0, motion on at EXAMPLE_TEMPERATURE_UK. */
const pgc=()=>exampleScene('pgc3d');
const roundtrip=scene=>validateScene(JSON.parse(JSON.stringify(scene)));
/** The motion request of a scene's settings. */
const request=scene=>motionRequest(scene.dynamics);

test('motion is on by default: the default scene\'s 852 nm tweezer is designated and on, new lasers are given by power and waist',()=>{
  const scene=defaultScene();
  assert.deepEqual(scene.dynamics.motion,{enabled:true,temperatureUK:DEFAULT_MOTION_TEMPERATURE_UK,temperatureUnit:'μK',device:'gpu'});
  assert.equal(tweezerOf(scene)?.id,'tweezer');assert.equal(tweezerOf(scene).enabled,true);assert.equal(motionRefusal(scene),'');
  // A new laser: a cooling or probe beam, 50 μW in a 1 mm waist -- about 3.2 mW/cm² at the atom.
  const laser=normalizeLaserIntensity(newLaser('new','激光 1','D2'));
  assert.deepEqual([laser.intensityMode,laser.powerMW,laser.waistUm,laser.powerUnit,laser.waistUnit,laser.intensityUnit],['power',NEW_LASER_POWER_MW,NEW_LASER_WAIST_UM,'μW','mm','mW/cm²']);
  assert.ok(Math.abs(laserIntensityMWcm2(laser)-2*.05/(Math.PI*.1**2))<1e-12,`${laserIntensityMWcm2(laser)}`);
  assert.deepEqual(laser.frequency,{base:{kind:'transition',line:'D2',Fg:2,Fe:3},offsetMHz:0});
  assert.deepEqual(newLaser('d1','激光 2','D1').frequency,{base:{kind:'transition',line:'D1',Fg:2,Fe:2},offsetMHz:0});
});

test('the tweezer is the one laser designated: designating another moves it, removing it drops it, and motion needs it',()=>{
  const scene=defaultScene(),[probe,tweezer]=scene.lasers;
  setTweezer(scene,probe);assert.equal(tweezerOf(scene),probe,'designating another moves it');
  setTweezer(scene,undefined);assert.equal(scene.tweezerId,undefined);assert.equal(tweezerOf(scene),undefined);
  assert.throws(()=>setTweezer(scene,{...tweezer,id:'elsewhere'}),/不是场景中的激光/);
  // Motion needs it, on, bright and focused at the atom; each refusal names the laser.
  assert.equal(motionRefusal(scene),'运动需要光镊：请在一束激光的参数中打开“设为光镊”，或在布居演化页关闭“运动”。');
  setTweezer(scene,tweezer);assert.equal(motionRefusal(scene),'');
  tweezer.enabled=false;assert.equal(tweezerRefusal(scene),'运动需要光镊：光镊「光镊」已关闭。');tweezer.enabled=true;
  writeLaserPower(tweezer,0,'mW');assert.equal(tweezerRefusal(scene),'运动需要光镊：光镊「光镊」的强度为零。');writeLaserPower(tweezer,4,'mW');
  setLaserIntensityMode(tweezer,'intensity');assert.match(tweezerRefusal(scene),/^运动需要光镊：光镊「光镊」按局域强度给出，是平面波；/);
  setLaserIntensityMode(tweezer,'na');assert.equal(tweezerRefusal(scene),'','NA mode is a Gaussian beam focused at the atom');
  // The model refuses motion with the same words; off, nothing is refused.
  delete scene.tweezerId;assert.throws(()=>buildFullMasterModel(scene,scene.dynamics,{axis:'3d',temperatureUK:30}),/^Error: 运动需要光镊：请在一束激光的参数中打开“设为光镊”/);
  scene.dynamics.motion.enabled=false;assert.equal(motionRefusal(scene),'');
  // A laser leaving the scene takes its designation with it; another leaving keeps it.
  const kept=defaultScene();kept.lasers=kept.lasers.filter(l=>l.id!=='probe');keepTweezer(kept);assert.equal(kept.tweezerId,'tweezer');
  kept.lasers=[];keepTweezer(kept);assert.equal(kept.tweezerId,undefined);
});

test('the scene file keeps the tweezer and the motion with its device; another file is refused with the field named',()=>{
  const scene=defaultScene(),loaded=roundtrip(scene);
  assert.equal(loaded.tweezerId,'tweezer');assert.deepEqual(loaded.dynamics.motion,scene.dynamics.motion);
  scene.dynamics.motion={enabled:false,temperatureUK:1500,temperatureUnit:'mK',device:'cpu'};assert.deepEqual(roundtrip(scene).dynamics.motion,scene.dynamics.motion);
  const none=defaultScene();delete none.tweezerId;assert.equal(Object.hasOwn(roundtrip(none),'tweezerId'),false,'no tweezer: none');
  const file=()=>JSON.parse(JSON.stringify(defaultScene()));
  for(const [label,change,reason] of [
    ['a tweezer that is no laser',f=>{f.tweezerId='nowhere';},/tweezerId 应为场景中一束激光的 ID/],['a tweezer that is no text',f=>{f.tweezerId=7;},/tweezerId 应为 1 至 64 个字符的文本/],
    ['no motion',f=>{delete f.dynamics.motion;},/dynamics 缺少 motion/],['a motion axis',f=>{f.dynamics.motion.axis='x';},/dynamics\.motion 含有未知字段 axis/],
    ['no device',f=>{delete f.dynamics.motion.device;},/dynamics\.motion 缺少 device/],['another device',f=>{f.dynamics.motion.device='tpu';},/dynamics\.motion\.device/],
    ['no temperature',f=>{f.dynamics.motion.temperatureUK=0;},/dynamics\.motion\.temperatureUK/],['a temperature unit',f=>{f.dynamics.motion.temperatureUnit='nK';},/dynamics\.motion\.temperatureUnit/],
    ['a version-2 file',f=>{f.version=2;},/^Error: 场景文件不符合当前格式（version 3）：version 为 2，应为 3$/],
  ]){const f=file();change(f);assert.throws(()=>validateScene(f),reason,label);}
  // 丢失 is a curve a scene may show.
  const curves=defaultScene();curves.dynamics.monitorMode='manual';curves.dynamics.monitoredGroupIds=[LOSS_CURVE.id,'F:ground:1'];
  assert.deepEqual(roundtrip(curves).dynamics.monitoredGroupIds,['loss','F:ground:1']);
});

test('the initial temperature keeps its digits when its unit changes (§0.4); it is finite and positive',()=>{
  const motion=defaultMotion();assert.deepEqual(motionTemperatureReading(motion),{value:30,unit:'μK'});
  setMotionTemperature(motion,typedDigits(motionTemperatureReading(motion).value),'mK');
  assert.deepEqual([motion.temperatureUK,motion.temperatureUnit],[30000,'mK']);assert.deepEqual(motionTemperatureReading(motion),{value:30,unit:'mK'});
  for(const value of [0,-1,NaN,Infinity])assert.throws(()=>setMotionTemperature(motion,value,'μK'),/温度必须是有限正数/);
  assert.deepEqual([motion.temperatureUK,motion.temperatureUnit],[30000,'mK'],'a refused temperature leaves it as it was');
});

/** Σ_j drive.cos + (H0 − H0 without beams) = H(t = 0, r = 0), from the model; and from the beams. */
function hamiltonianAtOrigin(model){
  const re=model.hReal.map((h,k)=>h-model.motion.hamiltonian.real[k]+model.drives.reduce((s,d)=>s+d.cos.real[k],0));
  const im=model.hImag.map((h,k)=>h-model.motion.hamiltonian.imag[k]+model.drives.reduce((s,d)=>s+d.cos.imag[k],0));
  const fromBeams={re:Array(N*N).fill(0),im:Array(N*N).fill(0)};
  for(const b of model.motion.beams)for(const X of [b.coupling,b.counterCoupling])for(let i=0;i<N;i++)for(let j=0;j<N;j++){
    const k=i*N+j,t=j*N+i;fromBeams.re[k]+=X.real[k]+X.real[t];fromBeams.im[k]+=X.imag[k]-X.imag[t];
  }
  return {model:{re,im},beams:fromBeams};
}
const largest=(a,b)=>Math.max(...a.re.map((x,k)=>Math.hypot(x-(b?b.re[k]:0),a.im[k]-(b?b.im[k]:0))));

test('the motion block: 3D in the lab frame, the trap beam named, a waist exactly on the Gaussian beams; at the atom the beams add up to the drives',()=>{
  // A cooling beam given by power and waist is a Gaussian beam too: it carries its waist, and is no trap.
  const scene=pgc(),cooling=scene.lasers.find(l=>l.id==='cool-x');setLaserIntensityMode(cooling,'power');resolveSceneFrequencies(scene);
  const model=buildFullMasterModel(scene,scene.dynamics,request(scene)),m=model.motion,f=model.frequencyCoordinates.frequenciesMHz;
  assert.deepEqual([m.axis,m.trapBeamId],['3d','tweezer']);assert.equal(m.beams.length,8);
  const {model:h,beams}=hamiltonianAtOrigin(model);assert.ok(largest(h,beams)<=1e-12*largest(beams),`${largest(h,beams)}`);
  const reference=scene.lasers.find(l=>l.id===model.referenceLaserId);assert.equal(f.at(-1),laserFrequencyMHz(reference));
  for(const b of m.beams){
    const laser=scene.lasers.find(l=>l.id===b.id),detuning=b.frequencyIndex.reduce((s,n,k)=>s+n*f[k],0);
    assert.ok(Math.abs(detuning-(laserFrequencyMHz(laser)-laserFrequencyMHz(reference)))<1e-6,b.id);
    assert.deepEqual([b.name,b.waveNumberPerUm,b.intensityMWcm2],[laser.name,2*Math.PI*laserFrequencyMHz(laser)/C,laserIntensityMWcm2(laser)]);
    // The direction on the lab axes (a zero's sign aside: a reflection's opposite direction has −0 components); a waist
    // for a beam given by power and waist or NA, none for a plane wave.
    assert.deepEqual(b.direction.map(x=>x+0),directionVector(scene.fields,laser.direction).map(x=>x+0),b.id);assert.equal(b.waistUm,laser.intensityMode==='intensity'?undefined:laserWaistUm(laser),b.id);
    assert.ok(b.coupling.real.every((x,k)=>x===0&&b.coupling.imag[k]===0||(Math.floor(k/N)>=8&&k%N<8)),b.id);
  }
  assert.deepEqual(m.beams.filter(b=>b.waistUm!==undefined).map(b=>[b.id,b.waistUm]),[['tweezer',1],['cool-x',laserWaistUm(cooling)],['cool-x-reflection',laserWaistUm(cooling)]]);
  assert.deepEqual(m.beams.find(b=>b.id==='tweezer').direction,[0,0,1]);assert.deepEqual(m.beams.find(b=>b.id==='cool-x-reflection').direction,[-1,0,0]);
  // H0 without beams is diagonal; the excited states' frame is the reference carrier.
  assert.ok(m.hamiltonian.real.every((x,k)=>x===0||k%(N+1)===0)&&m.hamiltonian.imag.every(x=>x===0));
  assert.deepEqual(m.frameFrequenciesMHz,model.frameFrequenciesMHz);assert.equal(m.massKg,RB87.massKg);assert.equal(m.temperatureUK,EXAMPLE_TEMPERATURE_UK);
  // Another laser designated moves the trap; without motion the model is as before.
  setTweezer(scene,cooling);assert.equal(buildFullMasterModel(scene,scene.dynamics,request(scene)).motion.trapBeamId,'cool-x');
  const plain=buildFullMasterModel(scene,scene.dynamics);assert.equal(plain.motion,undefined);
  assert.deepEqual({h:plain.hReal,d:plain.drives,c:plain.collapse},{h:model.hReal,d:model.drives,c:model.collapse});
});

/** C ρ C† summed over operators. */
function dissipated(operators,rho){
  const out={re:Array(N*N).fill(0),im:Array(N*N).fill(0)};
  for(const c of operators)for(let i=0;i<N;i++)for(let j=0;j<N;j++)for(let k=0;k<N;k++){
    const a={re:c.real[i*N+k],im:c.imag[i*N+k]};if(!a.re&&!a.im)continue;
    for(let l=0;l<N;l++){const b={re:c.real[j*N+l],im:-c.imag[j*N+l]};if(!b.re&&!b.im)continue;const r={re:rho.re[k*N+l],im:rho.im[k*N+l]};
      const ar={re:a.re*r.re-a.im*r.im,im:a.re*r.im+a.im*r.re};out.re[i*N+j]+=ar.re*b.re-ar.im*b.im;out.im[i*N+j]+=ar.re*b.im+ar.im*b.re;}
  }
  return out;
}
test('the Cartesian emission operators on the lab axes are the model\'s spontaneous emission, in the drives\' polarization convention',()=>{
  // A tilted field: the quantization frame is not the lab's.
  const scene=pgc();Object.assign(scene.fields[0],{magnitudeG:2.5,direction:{reference:{kind:'angles',thetaDeg:37,phiDeg:-21},thetaDeg:0,phiDeg:0}});
  const model=buildFullMasterModel(scene,scene.dynamics,request(scene)),rho={re:Array(N*N).fill(0),im:Array(N*N).fill(0)};
  for(let i=0;i<N;i++)for(let j=0;j<=i;j++){const a=Math.sin(3*i+7*j),b=i===j?0:Math.cos(5*i-2*j);rho.re[i*N+j]=rho.re[j*N+i]=a;rho.im[i*N+j]=b;rho.im[j*N+i]=-b;}
  const cartesian=dissipated(model.motion.emission,rho),spherical$=dissipated(model.collapse,rho);
  assert.ok(largest(cartesian,spherical$)<=1e-13*largest(spherical$));
  // The convention: a laser's amplitudes are e_q*·ε on the quantization axis (q = −1, 0, +1), the
  // basis the emission operators are combined in.
  const frame=composeMagneticField(scene.fields).frame;
  for(const [theta,phi,psi,chi] of [[0,0,0,45],[90,30,10,-20],[133,-71,-40,7],[180,0,25,45]]){
    const laser={...structuredClone(scene.lasers[1]),direction:spherical(theta,phi),polarizationAngleDeg:psi,ellipticityDeg:chi},p=polarization(laser,scene.fields);
    const c=Math.cos(chi*Math.PI/180),s=Math.sin(chi*Math.PI/180),re=p.u.map(x=>c*x),im=p.v.map(x=>s*x);
    const [xr,yr,zr]=rotate(invert(frame),re),[xi,yi,zi]=rotate(invert(frame),im),h=Math.SQRT1_2;
    const expected=[{re:h*(xr-yi),im:h*(xi+yr)},{re:zr,im:zi},{re:-h*(xr+yi),im:-h*(xi-yr)}];
    expected.forEach((e,q)=>assert.ok(Math.hypot(e.re-p.amplitudes[q].re,e.im-p.amplitudes[q].im)<1e-12,`${theta},${phi},${psi},${chi}: q=${q-1}`));
  }
  // Along the field (z, at B along z) the z component is the π operator itself.
  const along=pgc();Object.assign(along.fields[0],{magnitudeG:3,direction:axis('+z')});
  const pi=buildFullMasterModel(along,along.dynamics,request(along)),axial=pi.motion.emission[2];
  assert.ok(axial.real.every((x,k)=>x===pi.collapse[1].real[k])&&axial.imag.every((x,k)=>x===pi.collapse[1].imag[k]));
  assert.deepEqual(MOTION_FRAME,[[1,0,0],[0,1,0],[0,0,1]]);
});

test('a motion trajectory is its own request: its key holds the motion, the device, the tweezer and the waists; the mean\'s key does not',()=>{
  const scene=pgc(),off=structuredClone(scene);off.dynamics.motion.enabled=false;
  const key=s=>trajectoryInputsKey(s,s.dynamics),changed=change=>{const s=structuredClone(scene);change(s);return key(s);};
  assert.notEqual(key(scene),key(off));
  for(const [label,change] of [['the temperature',s=>{s.dynamics.motion.temperatureUK=40;}],['the device',s=>{s.dynamics.motion.device='cpu';}],
    ['the tweezer',s=>{s.tweezerId='cool-x';}],['a waist',s=>{s.lasers[0].waistUm=1.5;s.lasers[0].intensityAnchor=undefined;}]])assert.notEqual(changed(change),key(scene),label);
  // The same intensity at the atom as a plane wave: the internal model is the same, the motion's is not.
  const plane=structuredClone(scene),offPlane=structuredClone(off);for(const s of [plane,offPlane])setLaserIntensityMode(s.lasers[0],'intensity');
  assert.equal(laserIntensityMWcm2(plane.lasers[0]),laserIntensityMWcm2(scene.lasers[0]));
  assert.notEqual(key(plane),key(scene));assert.equal(key(offPlane),key(off));
  // Off, the device and the tweezer are no part of it; the mean is the internal state's either way.
  assert.equal(changed(s=>{s.dynamics.motion.enabled=false;s.dynamics.motion.device='cpu';delete s.tweezerId;}),key(off));
  assert.equal(meanInputsKey(scene,scene.dynamics),meanInputsKey(off,off.dynamics));
  assert.deepEqual(request(scene),{axis:'3d',temperatureUK:EXAMPLE_TEMPERATURE_UK});assert.equal(request(off),undefined);
  assert.equal(plannedSolve(true,{trajectory:false,mean:false},true),'trajectory');
  assert.equal(plannedSolve(true,{trajectory:true,mean:false},true),undefined);
  assert.equal(plannedSolve(false,{trajectory:false,mean:false},true),'mean','the levels\' p̄: the atom at rest, the internal state');
});

test('the native payload carries the motion block with its trap beam, and the device; a malformed motion or device is refused',()=>{
  const scene=pgc(),{motion:_,...dynamics}=scene.dynamics,options={...dynamics,samples:3,includeMean:false,meanOnly:false,motion:request(scene),device:'gpu'};
  const {payload}=prepareNativeSolve({scene,options});
  assert.deepEqual(Object.keys(payload.model.motion).sort(),['axis','beams','emission','fieldVectorG','frameFrequenciesMHz','hamiltonian','massKg','outsideForms','temperatureUK','trapBeamId']);
  assert.deepEqual([payload.model.motion.axis,payload.model.motion.trapBeamId,payload.options.device],['3d','tweezer','gpu']);
  assert.deepEqual(Object.keys(payload.model.motion.beams.find(b=>b.id==='tweezer')).sort(),
    ['counterCoupling','coupling','direction','electricField','frequencyIndex','id','intensityMWcm2','name','outsideShift','waistUm','waveNumberPerUm']);
  // Each beam's static shift through the states outside the model: Hermitian; the tweezer's on |2,+2⟩ about −0.11 MHz at its focus.
  const shift=payload.model.motion.beams.find(b=>b.id==='tweezer').outsideShift,two=payload.model.states.indexOf('ground-2-2');
  const size=Math.max(...shift.real.map(Math.abs)),transposed=k=>(k%N)*N+Math.floor(k/N);
  assert.ok(shift.real.every((x,k)=>Math.abs(x-shift.real[transposed(k)])<=1e-12*size)&&shift.imag.every((x,k)=>Math.abs(x+shift.imag[transposed(k)])<=1e-12*size));
  assert.ok(Math.abs(shift.real[two*N+two]/(2*Math.PI)+.113)<.01,`${shift.real[two*N+two]/(2*Math.PI)} MHz`);
  assert.equal(prepareNativeSolve({scene,options:{...options,device:'cpu'}}).payload.options.device,'cpu');
  // No motion: no block and no device.
  const plain=prepareNativeSolve({scene,options:{...options,motion:undefined,device:undefined}}).payload;
  assert.equal(plain.model.motion,undefined);assert.equal(Object.hasOwn(plain.options,'device'),false);
  for(const [label,change] of [['an axis',o=>{o.motion={axis:'x',temperatureUK:30};}],['a plane',o=>{o.motion={axis:'xy',temperatureUK:30};}],['no temperature',o=>{o.motion={axis:'3d',temperatureUK:-1};}],
    ['an unknown motion key',o=>{o.motion={axis:'3d',temperatureUK:30,trajectories:64};}],['no motion object',o=>{o.motion=null;}],['no device',o=>{delete o.device;}],['another device',o=>{o.device='tpu';}],
    ['a device without motion',o=>{delete o.motion;}],['a mean',o=>{o.includeMean=true;}]]){
    const o=structuredClone(options);change(o);assert.throws(()=>prepareNativeSolve({scene,options:o}),error=>error.code==='invalid-options',label);
  }
  assert.deepEqual(readMotionRequest({axis:'3d',temperatureUK:12.5}),{axis:'3d',temperatureUK:12.5});assert.equal(readMotionDevice('cpu'),'cpu');
  // Without a tweezer the model is refused, saying why.
  const loose=pgc();delete loose.tweezerId;assert.throws(()=>prepareNativeSolve({scene:loose,options}),/运动需要光镊/);
});

test('the client sends the motion and its device with a trajectory alone; neither with a mean (the atom at rest) or without motion',async t=>{
  const calls=[],client=new DynamicsClient({fetch:async(url,init)=>new Promise(resolve=>calls.push({resolve,body:JSON.parse(init.body)})),debounceMs:0,clientId:'motion'});t.after(()=>client.dispose());
  const scene=pgc();scene.dynamics.samples=3;scene.dynamics.motion.device='cpu';
  const pending=client.solve(scene,scene.dynamics,{includeMean:false});
  for(let i=0;i<100&&!calls.length;i++)await pause(2);
  const call=calls[0];assert.deepEqual([call.body.options.motion,call.body.options.device,call.body.options.includeMean],[{axis:'3d',temperatureUK:EXAMPLE_TEMPERATURE_UK},'cpu',false]);
  const states=buildFullMasterModel(scene,scene.dynamics).states,timesUs=[0,scene.dynamics.durationUs/2,scene.dynamics.durationUs];
  const data=motionResult({states,initialIndex:states.findIndex(s=>s.id===scene.dynamics.initialStateId),timesUs});
  call.resolve(new Response(JSON.stringify({...data,request:call.body.request,service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA]}}),{status:200,headers:{'Content-Type':'application/json'}}));
  const result=await pending;assert.deepEqual([result.motion.temperatureTotalUK,result.motion.device],[[31,31,31],'cpu']);
  // The levels' p̄(T) with motion on: the internal state's mean, no motion asked for.
  const mean=client.solve(scene,scene.dynamics,{meanOnly:true}).catch(e=>e);for(let i=0;i<100&&calls.length<2;i++)await pause(2);
  assert.deepEqual([calls[1].body.options.meanOnly,calls[1].body.options.motion,calls[1].body.options.device],[true,undefined,undefined]);client.cancel();await mean;
  scene.dynamics.motion.enabled=false;const plain=client.solve(scene,scene.dynamics).catch(e=>e);for(let i=0;i<100&&calls.length<3;i++)await pause(2);
  assert.deepEqual([calls[2].body.options.motion,calls[2].body.options.device],[undefined,undefined]);client.cancel();await plain;
});

/** The 3D example's motion result on three samples (0, 5 and 10 ms), changed by `change`, read as the client reads it. */
function readMotion(change=()=>{}){
  const scene=pgc();scene.dynamics.samples=3;const states=buildFullMasterModel(scene,scene.dynamics).states;
  const data=motionResult({states,initialIndex:states.findIndex(s=>s.id===scene.dynamics.initialStateId),timesUs:[0,5000,10000]});change(data);
  return normalizeDynamicsResult(data,scene,undefined,undefined,request(scene));
}
test('the ledger\'s table lists its items, the largest first, at the largest one\'s precision, and those that read zero there together',()=>{
  const item=estimate=>({label:`${estimate}`,estimate}),rows=estimates=>{const {places,listed,rest}=ledgerRows(estimates.map(item));return [places,listed.map(i=>i.estimate),rest.map(i=>i.estimate)];};
  // 0.59 % is written to its second digit, the second decimal: 0.005 % and above reads above zero there.
  assert.deepEqual(rows([5.9e-3,2e-4,5.1e-5,4.9e-5,1.3e-17,0]),[2,[5.9e-3,2e-4,5.1e-5],[4.9e-5,1.3e-17,0]]);
  // The place follows the largest item: 2.1 % to the first decimal, 12 % to the unit, 0.0031 % to the fourth decimal.
  assert.deepEqual(rows([.021,6e-4,4e-4]),[1,[.021,6e-4],[4e-4]]);
  assert.deepEqual(rows([.12,.006,.004]),[0,[.12,.006],[.004]]);
  assert.deepEqual(rows([3.1e-5,1e-9]),[4,[3.1e-5],[1e-9]]);
  // The largest rounded to two digits may gain a place (0.0996 % reads 0.10 %): its place is the rounded reading's.
  assert.deepEqual(rows([9.96e-4,6e-6]),[2,[9.96e-4],[6e-6]]);
  // Nothing above zero: nothing listed; no item: nothing at all; an item below every decimal the language writes is not listed.
  assert.deepEqual(rows([0,0]),[1,[],[0,0]]);assert.deepEqual(rows([]),[1,[],[]]);assert.deepEqual(rows([1e-200]),[100,[],[1e-200]]);
  // The fixture's ledger: both items read above zero at 0.12 %'s place.
  assert.equal(ledgerRows(MOTION_DIAGNOSTICS.approximations.items).rest.length,0);
});
test('a motion result is read strictly: the 3D block with the total temperature, the device, a fallback\'s note and the approximation ledger',()=>{
  const result=readMotion(),m=result.motion,block=motionBlock(3);
  assert.deepEqual([m.temperatureUK,m.temperatureErrorUK],[[block.temperatureUK.x,block.temperatureUK.y,block.temperatureUK.z],[[1.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]]]);
  assert.deepEqual([m.temperatureTotalUK,m.temperatureTotalErrorUK,m.trapFrequencyKHz,m.classical],[[31,31,31],[.9,.9,.9],[100.5,100.5,19.3],[true,true,false]]);
  assert.deepEqual([m.device,m.deviceNote,m.approximations],[MOTION_DIAGNOSTICS.device,MOTION_DIAGNOSTICS.deviceNote,MOTION_DIAGNOSTICS.approximations]);
  assert.equal(result.populationAccuracy.requestedTolerance,.05);assert.equal(result.mean.status,'not-requested');
  const few=readMotion(d=>{d.motion.temperatureErrorUK.x[1]=null;d.motion.temperatureTotalErrorUK[1]=null;}).motion;
  assert.deepEqual([few.temperatureErrorUK[0],few.temperatureTotalErrorUK],[[1.5,null,1.5],[.9,null,.9]],'a sample with fewer than two atoms held has no T of its own and no error');
  const flagged=readMotion(d=>{d.diagnostics.approximations.flagged='t ≥ 8 ms 的输出超出近似误差预算';delete d.diagnostics.deviceNote;}).motion;
  assert.equal(flagged.approximations.flagged,'t ≥ 8 ms 的输出超出近似误差预算');assert.equal(Object.hasOwn(flagged,'deviceNote'),false);
  const noted=readMotion(d=>{d.diagnostics.approximations.note='初始热分布中 84% 原子不计入';}).motion;
  assert.equal(noted.approximations.note,'初始热分布中 84% 原子不计入');assert.equal(Object.hasOwn(noted.approximations,'flagged'),false,'the setup\'s note is no flag');
  for(const [label,change] of [
    ['no block',d=>{delete d.motion;}],['an axis',d=>{d.motion.axis='x';}],['no total temperature',d=>{delete d.motion.temperatureTotalUK;}],
    ['a total short of a sample',d=>{d.motion.temperatureTotalUK.pop();}],['a negative total error',d=>{d.motion.temperatureTotalErrorUK[1]=-.1;}],
    ['one axis missing',d=>{delete d.motion.temperatureUK.z;}],['no regime',d=>{delete d.motion.classical;}],['no trajectories',d=>{d.motion.trajectories=0;}],
    ['a loss that decreases',d=>{d.motion.lossProbability[2]=0;}],['an unknown key',d=>{d.motion.axes=['x','y','z'];}],
    ['no device',d=>{delete d.diagnostics.device;}],['another device',d=>{d.diagnostics.device='tpu';}],['an empty note',d=>{d.diagnostics.deviceNote=' ';}],
    ['no ledger',d=>{delete d.diagnostics.approximations;}],['a ledger key',d=>{d.diagnostics.approximations.total=1;}],['no totals',d=>{delete d.diagnostics.approximations.totals;}],['a negative total',d=>{d.diagnostics.approximations.totals.temperature.error=-1;}],['a negative estimate',d=>{d.diagnostics.approximations.items[0].estimate=-1;}],['items not from the largest down',d=>{d.diagnostics.approximations.items.reverse();}],
    ['an item without its action',d=>{delete d.diagnostics.approximations.items[1].action;}],['an empty flag',d=>{d.diagnostics.approximations.flagged='';}],['an empty ledger note',d=>{d.diagnostics.approximations.note='';}],
  ])assert.throws(()=>readMotion(change),/运动结果/,label);
  assert.throws(()=>readMotion(d=>{d.populationAccuracy.requestedTolerance=.01;}),/布居精度/);
  // A block nobody asked for is refused.
  const scene=pgc(),states=buildFullMasterModel(scene,scene.dynamics).states;
  const unasked=motionResult({states,initialIndex:states.findIndex(s=>s.id===scene.dynamics.initialStateId),timesUs:[0,1,2]});unasked.populationAccuracy.requestedTolerance=.01;
  assert.throws(()=>normalizeDynamicsResult(unasked,scene),/运动结果/);
});

test('a motion trajectory is tested against its own target, 5 %, and every sentence about its samples names that target',()=>{
  // A result's sample errors with the status the adapter writes for them (motion_adapter.rs).
  const errors=values=>d=>{const a=d.populationAccuracy,worst=Math.max(...values),partial=worst>a.requestedTolerance;
    Object.assign(a,{sampleErrors:values,estimatedMaximumError:worst,status:partial?'estimated-partial':'estimated-converged'});if(partial)a.reason='test budget';else delete a.reason;};
  const initial=pgc().dynamics.initialStateId,chart=r=>populationChart(r,[initial],{[initial]:'初态'},{showTotalExcited:false});
  // 3 %: beyond the population core's ε, within the motion's own target -- nothing marked, nothing said.
  const within=readMotion(errors([0,.03,.01])),tolerance=resultTolerance(within);
  assert.equal(tolerance,SEMICLASSICAL_POPULATION_TARGET);assert.notEqual(tolerance,POPULATION_TOLERANCE);
  assert.deepEqual(outsideRuns(within.populationAccuracy.sampleErrors,tolerance),[]);
  assert.equal(partialSentence(within.timesUs,within.populationAccuracy.sampleErrors,'ms',tolerance),'');
  assert.doesNotMatch(chart(within),/accuracy-band|超过/);
  // 6.2 %: marked, and the words name the 5 % it was tested against, never ε.
  const over=readMotion(errors([0,.062,.01])),limit=resultTolerance(over);
  assert.equal(partialSentence(over.timesUs,over.populationAccuracy.sampleErrors,'ms',limit),'t ≤ 5 ms 估计误差最大 6.2%（超过 5%），其余时刻 ≤ 5%');
  assert.equal(exceedsNote(.062,limit),'（估计误差 6.2%，超过 5%）');assert.equal(exceedsNote(null,limit),'（估计误差 未知，超过 5%）');
  assert.match(chart(over),/超过 5%/);assert.doesNotMatch(chart(over),/超过 1%/);
  // A trajectory of the population core is tested against ε.
  assert.equal(resultTolerance({populationAccuracy:{requestedTolerance:POPULATION_TOLERANCE}}),POPULATION_TOLERANCE);
  // The app's own sentences (app.ts: the status line, a checkpoint's words, the levels chart's notes) take a trajectory's
  // tolerance from the result: ε is written only in what is said of p̄(T), the population core's mean.
  const app=readFileSync(new URL('../src/app.ts',import.meta.url),'utf8');
  const naming=app.split('\n').filter(line=>/(errorPercent|small)\(POPULATION_TOLERANCE\)/.test(line));
  assert.ok(naming.length>0&&naming.every(line=>/mean/.test(line)),naming.join('\n'));
});

test('the notes and the chart titles say what the engine computes: the loss, the trap light, T\'s window, n̄, the initial ensemble, the statistics, the target on T',()=>{
  const notes=definitionNotes()+modelNotes(),solver=new URL('../rust-solver/motion/src/',import.meta.url),source=path=>readFileSync(new URL(path,solver),'utf8');
  // The loss (model.rs `Model::lost`): first outside the trap light's region AND its loss energy above the far value; an
  // atom outside the region but bound there is held. The region's words are the engine's (units.rs REGION_INTENSITY).
  assert.match(source('model.rs'),/self\.light\.outside\(at\.r\)\s*&&\s*self\.energy_with\([^)]*\)\s*>\s*self\.asymptote\(/);
  assert.match(source('units.rs'),/pub const REGION_INTENSITY: f64 = 1\.0 \/ 20\.0;/);
  const region=/model-notes\.ts: "([^"]+)"/.exec(source('units.rs'))?.[1];
  assert.ok(region&&region.includes('1/20')&&notes.includes(region),region);
  assert.match(notes,/1\/20）[^。]*能量[^。]*高于远处的值[^。]*区域之外仍被束缚的原子不算丢失/);
  // The trap light is the whole field of the tweezer's frequency (field.rs `Light::outside`, SPEC §1.3).
  assert.match(source('field.rs'),/self\.frequencies\[self\.trap\]\.beams\.iter\(\)\.all/);
  assert.match(notes,/“设为光镊”的那一束光，连同与它同频的各束光/);
  // T and n̄ over one trap period about t, cut at t = 0 and kept centred (estimate.rs `half_window`): from 0 to 2t at first.
  assert.match(source('estimate.rs'),/pub fn half_window\(frequency: f64, t: f64\) -> f64 \{\s*\(0\.5 \* TAU \/ frequency\)\.min\(t\)/);
  const [temperature,occupation]=motionSeries(readMotion().motion);
  assert.ok(temperature.title.includes(TEMPERATURE_WINDOW)&&TEMPERATURE_WINDOW.includes('一个阱周期')&&TEMPERATURE_WINDOW.includes('0 到 2t'),temperature.title);
  assert.match(notes,/一个阱周期内的平均（t 不到半个周期时窗口截到 t = 0、仍以 t 为中心，即 0 到 2t/);
  // n̄ of the held atoms' mean κ², the mean velocity in it (estimate.rs `summarize`): k_BT/ħω − ½ only for atoms at rest as a whole.
  assert.match(source('estimate.rs'),/occupation: std::array::from_fn\(\|a\| occupation\(hm \* all\.kappa2\[a\], n, scales\.frequencies\[a\]\)\)/);
  assert.match(occupation.title,/n̄ = m⟨v²⟩\/ħω − ½/);assert.match(notes,/⟨v²⟩ 含原子整体的平均速度：没有平均速度时它就是 k<sub>B<\/sub>T\/ħω − ½/);
  // The initial ensemble: the lowered isothermal of the bound atoms (ensemble.rs), not a Boltzmann factor cut at the escape.
  assert.match(source('ensemble.rs'),/f\(r, κ\) ∝ e\^\{−E\/k_BT\} − 1 for E < 0/);
  assert.ok(notes.includes('降低的等温分布')&&notes.includes('e<sup>−E/k<sub>B</sub>T</sup> − 1'));
  // The statistics: Student's t at the normal's tail beyond z (estimate.rs `Spread::interval_over`), wider than two standard
  // errors over few groups.
  assert.match(source('estimate.rs'),/quantile\(tail_among\(looks\), freedom\) \* self\.error/);
  assert.match(notes,/区间按组数取 Student t，组少时比两倍标准误差宽/);
  // The total T's target: its three parts weighed by the held share (solve.rs `accounts`; SPEC §0).
  assert.match(source('solve.rs'),/Account \{ model: h \* model, step: h \* bound, statistics: h \* relative \}/);
  assert.match(notes,/按仍在光阱中的份额 1 − L 计权后与 10% 比较/);
  // A motion trajectory's sample errors are its total estimates against its own target.
  assert.match(notes,/按运动自己的目标 5% 判断与标出/);
  // The mirror's λ/4 plate: the field mirrored about the fast axis (reflection.ts).
  assert.match(notes,/ψ′ = ψ − 2θ、χ′ = χ[^。]*θ = ψ 时不变，θ = ψ ± 45° 时转 90°/);
  // A motion solve's assumptions are the motion's: no word of the master equation's unfolded drives or micromotion.
  const scene=pgc(),moving=buildFullMasterModel(scene,scene.dynamics,request(scene)).assumptions,still=buildFullMasterModel(scene,scene.dynamics).assumptions;
  assert.ok(still.some(t=>t.includes('不作 RWA'))&&still.some(t=>t.includes('光频微运动，幅度')));
  assert.ok(!moving.some(t=>t.includes('不作 RWA')||t.includes('光频微运动，幅度')),moving.join('\n'));
  assert.ok(moving.some(t=>t.includes('反向旋转项按裸能级二阶绝热消去'))&&moving.some(t=>t.includes('不含光频微运动')));
});

test('the populations of a motion result are the held atom\'s times 1 − L: with 丢失 they add up to one',()=>{
  const result=readMotion(),loss=result.motion.lossProbability,initial=pgc().dynamics.initialStateId;
  assert.deepEqual(loss,[0,.1,.2]);
  result.timesUs.forEach((_,k)=>{
    const states=result.states.reduce((sum,s)=>sum+result.populations[s.id][k],0);
    assert.ok(Math.abs(states+loss[k]-1)<1e-12,`sample ${k}: ${states} + ${loss[k]}`);
  });
  assert.deepEqual(result.populations[initial],[1,.9,.8]);assert.deepEqual(result.totalExcited,[0,0,0]);
  // 丢失 is a curve: shown when asked for, ranked among the states for 初末各 3 态, drawn in the neutral ink.
  const shown=withPopulationGroups(result,[LOSS_CURVE.id]);assert.deepEqual(shown.populations.loss,loss);assert.equal(result.populations.loss,undefined,'the solver\'s data stay');
  assert.deepEqual(recommendedCurves(result,initial),[initial,'loss']);
  const svg=populationChart(shown,[initial,'loss'],{[initial]:'初态',loss:'丢失'},{showTotalExcited:false});
  assert.match(svg,/<polyline data-series-id="loss"[^>]* stroke="#6a6866"/);assert.equal(populationColor('loss'),'#6a6866');
  // Without motion there is no loss to show.
  const plain={...result};delete plain.motion;assert.equal(withPopulationGroups(plain,[LOSS_CURVE.id]).populations.loss,undefined);
});

test('温度与 n̄: a line per axis and the total temperature apart from them; no standard-error band, the readout says T ± 误差',()=>{
  const result=readMotion(),[temperature,occupation]=motionSeries(result.motion),chart=seriesChart(result.timesUs,[temperature,occupation],'ms');
  assert.deepEqual(temperature.lines.map(l=>[l.name,l.ink,l.width]),[['T（x 轴）','#517FA6',undefined],['T（y 轴）','#B17D4B',undefined],['T（z 轴）','#568C79',undefined],['总温度',TOTAL_INK,TOTAL_WIDTH]]);
  assert.deepEqual(occupation.lines.map(l=>l.name),['n̄（x 轴）','n̄（y 轴）','n̄（z 轴）']);assert.match(occupation.title,/x 100\.5、y 100\.5、z 19\.3 kHz/);
  assert.match(chart,new RegExp(`<polyline points="[^"]+" fill="none" stroke="${TOTAL_INK}" stroke-width="${TOTAL_WIDTH}" data-series-id="temperature-3"`));
  assert.doesNotMatch(chart,/<polygon/,'no band');assert.doesNotMatch(chart,/⟨v⟩|平均速度/,'no velocity');
  assert.equal(lineReading(temperature.lines[3],1,temperature.unit),'31 ± 0.9 μK');assert.equal(lineReading(temperature.lines[0],1,temperature.unit),'30 ± 1.5 μK');
  assert.equal(lineReading(occupation.lines[2],1,occupation.unit),'4.5');
  const few=motionSeries(readMotion(d=>{d.motion.temperatureErrorUK.x[1]=null;}).motion)[0];
  assert.equal(lineReading(few.lines[0],1,few.unit),'30 μK（误差未知）');assert.equal(lineReading(few.lines[0],0,few.unit),'30 ± 1.5 μK');
  // A sample outside the validated regime lies on the grey band of both panels, from half-way to its neighbour.
  const bands=svg=>[...svg.matchAll(/<rect class="reference-band" x="([\d.]+)" y="(\d+)" width="([\d.]+)" height="(\d+)"/g)].map(m=>m.slice(1).map(Number));
  const reference=result.motion.classical.map(inside=>!inside);
  assert.deepEqual(bands(seriesChart(result.timesUs,[temperature,occupation],'ms',{reference})).map(([x,y,w,h])=>[x,x+w,y,y+h]),[[599.5,778,32,176],[599.5,778,212,300]]);
  assert.deepEqual(bands(chart),[]);
});

test('the examples: three scenes of the atom moving in its designated tweezer from |2,+2⟩ at 100 μK on D2; a retro-reflection is the app\'s mirror',()=>{
  assert.deepEqual(EXAMPLES.map(e=>e.key),['pgc3d','probe','pgc1d']);assert.equal(STARTUP_EXAMPLE,'pgc3d');
  assert.throws(()=>exampleScene('three'),/没有示例/);
  const scenes=Object.fromEntries(EXAMPLES.map(e=>[e.key,roundtrip(exampleScene(e.key))]));
  for(const {key,label} of EXAMPLES){
    const s=scenes[key];assert.equal(s.name,label);assert.equal(s.tweezerId,'tweezer',key);assert.equal(motionRefusal(s),'',key);
    assert.deepEqual(s.dynamics.motion,{...defaultMotion(),temperatureUK:EXAMPLE_TEMPERATURE_UK},key);
    assert.deepEqual([s.dynamics.initialStateId,s.dynamics.durationUs,s.dynamics.durationUnit,s.dynamics.samples],['ground-2-2',EXAMPLE_DURATION_US,'ms',EXAMPLE_SAMPLES],key);
    assert.ok(s.lasers.every(l=>l.line==='D2'),key);assert.equal(buildFullMasterModel(s,s.dynamics,request(s)).motion.trapBeamId,'tweezer',key);
  }
  const laser=(s,id)=>s.lasers.find(l=>l.id===id),round=x=>Math.round(x*1e9)/1e9+0;
  const weights=(s,id)=>{const p=polarization(laser(s,id),s.fields);return [p.minus,p.pi,p.plus].map(round);},vector=(s,id)=>directionVector(s.fields,laser(s,id).direction).map(round);
  const mirrors=s=>s.lasers.map(l=>[l.id,l.reflection?.of??null,l.reflection?.quarterWave??null]),D2=(Fg,Fe,offsetMHz)=>({base:{kind:'transition',line:'D2',Fg,Fe},offsetMHz});
  // 3D PGC (the startup scene): B = 0 (its axis along x); the tweezer along z, π; σ+σ− cooling pairs along x, y and z, each a beam and its
  // reflection through the λ/4 plate (the handedness about its own k kept: σ+ and σ− about the pair's axis), the z pair
  // of the other helicity; a repump. Eight lasers.
  const pgc3d=scenes.pgc3d,gamma=RB87.D2.gammaHz/(2*Math.PI*1e6),trap=pgc3d.lasers[0];assert.equal(pgc3d.fields[0].magnitudeG,0);
  assert.deepEqual([trap.id,trap.frequency,trap.intensityMode,trap.powerMW,trap.waistUm,vector(pgc3d,'tweezer'),weights(pgc3d,'tweezer')],['tweezer',{wavelengthNm:852},'power',PGC3D.tweezerMW,PGC3D.tweezerWaistUm,[0,0,1],[0,1,0]]);
  assert.deepEqual(mirrors(pgc3d),[['tweezer',null,null],['cool-x',null,null],['cool-x-reflection','cool-x',true],['cool-y',null,null],['cool-y-reflection','cool-y',true],['cool-z',null,null],['cool-z-reflection','cool-z',true],['repump',null,null]]);
  const cooling=pgc3d.lasers.filter(l=>l.id.startsWith('cool-')),free=D2(2,3,PGC3D.coolingDetuningGamma*gamma);
  for(const l of cooling)assert.deepEqual([l.reflection?laser(pgc3d,l.reflection.of).frequency:l.frequency,laserFrequencyMHz(l),laserIntensityMWcm2(l)],[free,laserFrequencyMHz(laser(pgc3d,'cool-x')),PGC3D.coolingMWcm2],l.id);
  assert.deepEqual(cooling.map(l=>vector(pgc3d,l.id)),[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]);
  assert.deepEqual(cooling.map(l=>l.ellipticityDeg),[45,45,45,45,-45,-45]);assert.deepEqual([weights(pgc3d,'cool-x'),weights(pgc3d,'cool-x-reflection')],[[0,0,1],[1,0,0]]);
  assert.deepEqual([laser(pgc3d,'repump').frequency,laser(pgc3d,'repump').intensityMWcm2],[D2(1,2,0),PGC3D.repumpMWcm2]);
  // 光镊 + 探测光 + 回泵光 at zero field: the probe π, 1 mW in a 1 mm waist, 20 MHz red, across the
  // tweezer; its reflection through the λ/4 plate at 45° comes back linear across the axis (lin⊥lin); the tweezer at 1.5 mK.
  const probe=scenes.probe,p=laser(probe,'probe');assert.equal(probe.fields[0].magnitudeG,0);
  assert.deepEqual(mirrors(probe),[['probe',null,null],['probe-reflection','probe',true],['tweezer',null,null],['repump',null,null]]);
  assert.deepEqual([p.frequency,p.intensityMode,p.powerMW,p.waistUm],[D2(2,3,LAB.probeOffsetMHz),'power',LAB.probeMW,LAB.probeWaistUm]);
  assert.deepEqual([weights(probe,'probe'),weights(probe,'probe-reflection'),weights(probe,'tweezer'),laser(probe,'tweezer').powerMW],[[0,1,0],[.5,0,.5],[0,1,0],LAB.tweezerMW]);
  assert.deepEqual(laser(probe,'repump').frequency,D2(1,2,0));
  // 1D PGC in the field: B along +z; the probe σ+ along it and the PGC beam σ− against it, of the probe's power and waist,
  // red of it by the |2,m⟩ → |2,m+2⟩ splitting -- the Raman resonance of the pair for the atom at rest.
  const field=scenes.pgc1d,[f,g]=['probe','pgc'].map(id=>laser(field,id));
  assert.deepEqual([field.fields[0].magnitudeG,directionVector(field.fields,field.fields[0].direction)],[PGC1D.fieldG,[0,0,1]]);
  assert.deepEqual(mirrors(field).map(([id,of])=>[id,of]),[['probe',null],['pgc',null],['tweezer',null],['repump',null]]);
  assert.deepEqual([weights(field,'probe'),weights(field,'pgc'),weights(field,'tweezer'),vector(field,'probe'),vector(field,'pgc')],[[0,0,1],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]]);
  assert.deepEqual([f.frequency,g.frequency,g.powerMW,g.waistUm],[D2(2,3,LAB.probeOffsetMHz),D2(2,3,LAB.probeOffsetMHz-2*F2_LARMOR_MHZ_PER_G*PGC1D.fieldG),f.powerMW,f.waistUm]);
  const E=Object.fromEntries(analyze(field).ground.map(l=>[l.id,l.energyMHz])),raman=laserFrequencyMHz(f)-laserFrequencyMHz(g);
  for(const m of [-2,-1,0])assert.ok(Math.abs(raman-(E[`ground-2-${m+2}`]-E[`ground-2-${m}`]))<5e-3,`${raman} MHz against |2,${m}⟩ → |2,${m+2}⟩`);
  assert.ok(Math.abs(F2_LARMOR_MHZ_PER_G-.6996)<1e-4,`${F2_LARMOR_MHZ_PER_G} MHz/G`);
});

test('the motion engine\'s sentences are shown in Chinese',()=>{
  const solver=new URL('../rust-solver/',import.meta.url),source=path=>readFileSync(new URL(path,solver),'utf8');
  // Shown: written in Chinese, or an English sentence solver-words.ts translates -- with 10 in each placeholder, the
  // longest duration in seconds as its translation has it.
  const shown=text=>chinese(text)||solverText(filled(text,'10'))!==filled(text,'10');
  // The engine (rust-solver/motion/src) and its adapter: every sentence of the shipped code -- a refusal, the method,
  // the ledger's items, a flag, a note -- is shown. What an assertion says goes with a fault in the program, to no user.
  const ASSERTED=['assert!','assert_eq!','assert_ne!','debug_assert!','debug_assert_eq!','debug_assert_ne!','expect','unreachable!','panic!','compile_error!'];
  const files=['src/motion_adapter.rs',...readdirSync(new URL('motion/src/',solver),{recursive:true}).map(name=>`motion/src/${name.replaceAll('\\','/')}`)
    .filter(path=>path.endsWith('.rs')&&!path.endsWith('/tests.rs')&&path!=='motion/src/request.rs')];
  assert.ok(files.length>20,`${files.length} files`);
  let sentences=0;
  for(const file of files)for(const {text,within} of rustLiterals(source(file))){
    if(!chinese(text)&&!(english(text)&&!within.some(name=>ASSERTED.includes(name))))continue;
    sentences++;assert.ok(shown(text),`${file}: ${text}`);
  }
  assert.ok(sentences>60,`${sentences} sentences`);
  // The request (motion/src/request.rs). What a scene or a setting can meet is shown: motion along an axis or in a
  // plane, a tweezer that is a plane wave and an initial state on an excited state in Chinese; the mean, the duration
  // and the samples in the words the population core's adapter refuses them with, translated. Its other refusals are of
  // requests off the contract, which the app's own model (full-master-model.ts) does not send: in English, as written.
  const request=source('motion/src/request.rs'),constant=name=>new RegExp(`const ${name}: &str = "([^"]+)";`).exec(request)?.[1]??'';
  for(const name of ['ONLY_3D','NO_WAIST','ON_GROUND'])assert.ok(chinese(constant(name)),name);
  const translated=rustLiterals(request).map(l=>l.text).filter(text=>english(text)&&shown(text));
  assert.deepEqual(translated,[constant('TRAJECTORY_ALONE'),'The trajectory duration must be a positive finite time',
    'The trajectory duration is limited to {} s (requested {} s)','A trajectory has 2 to {MOST_SAMPLES} samples (requested {samples})']);
  assert.equal(solverText(constant('TRAJECTORY_ALONE')),'运动只计算时间演化，不计算长期布居。');
  // The rows' check the adapter shares with the population core's.
  assert.ok(shown(/fn row_errors[^]*?Err\("([^"]+)"/.exec(source('src/lattice_core_adapter.rs'))?.[1]??''));
  // A request for the GPU is computed on the CPU, and the result says so in the adapter's words.
  assert.equal(/pub const GPU_ABSENT: &str = "([^"]+)";/.exec(source('src/motion_adapter.rs'))?.[1],MOTION_DIAGNOSTICS.deviceNote);
});
