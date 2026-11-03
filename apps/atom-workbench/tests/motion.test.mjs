// Motion (ui-design.md §6.20–§6.21): the tweezer the operator designates, the scene file, the model's
// motion block, the request with its device, the strict reader of a motion result (populations of the
// held atom with 丢失 beside them), the temperature chart and the examples.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as pause} from 'node:timers/promises';
import {DEFAULT_MOTION_TEMPERATURE_UK,NEW_LASER_POWER_MW,NEW_LASER_WAIST_UM,defaultMotion,defaultScene,newLaser} from '../dist/js/atom-workbench/src/types.js';
import {RB87,composeMagneticField,laserFrequencyMHz,laserIntensityMWcm2,polarization} from '../dist/js/atom-workbench/src/physics.js';
import {laserWaistUm,normalizeLaserIntensity,setLaserIntensityMode,writeLaserPower} from '../dist/js/atom-workbench/src/laser-intensity.js';
import {directionVector,invert,rotate} from '../dist/js/atom-workbench/src/direction.js';
import {MOTION_FRAME,motionRefusal,motionRequest,motionTemperatureReading,readMotionDevice,readMotionRequest,setMotionTemperature} from '../dist/js/atom-workbench/src/motion.js';
import {keepTweezer,setTweezer,tweezerOf,tweezerRefusal} from '../dist/js/atom-workbench/src/tweezer.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {EXAMPLES,LAMBDA_TWEEZER,TWEEZER_PGC,exampleScene} from '../dist/js/atom-workbench/src/examples.js';
import {meanInputsKey,trajectoryInputsKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {plannedSolve} from '../dist/js/atom-workbench/src/solve-plan.js';
import {DynamicsClient,normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {solverText} from '../dist/js/atom-workbench/src/solver-words.js';
import {typedDigits} from '../dist/js/atom-workbench/src/unit-input.js';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';
import {populationChart,populationColor} from '../dist/js/atom-workbench/src/population-chart.js';
import {LOSS_CURVE,recommendedCurves,withPopulationGroups} from '../dist/js/atom-workbench/src/population-selection.js';
import {TOTAL_INK,TOTAL_WIDTH,lineReading,motionSeries,seriesChart} from '../dist/js/atom-workbench/src/motion-chart.js';
import {MOTION_DIAGNOSTICS,motionBlock,motionResult} from './native-result.mjs';
import {spherical} from './directions.mjs';
import {readFileSync} from 'node:fs';

const N=32,C=299792458;
const axis=name=>({reference:{kind:'axis',axis:name},thetaDeg:0,phiDeg:0});
/** The 3D tweezer PGC example: the 852 nm tweezer along z, σ+σ− cooling pairs along x, y and z, a repump, B = 0, motion on. */
const pgc=()=>exampleScene('pgc3d').scene;
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
  const scene=pgc(),cooling=scene.lasers.find(l=>l.id==='cool-plus-x');setLaserIntensityMode(cooling,'power');
  const model=buildFullMasterModel(scene,scene.dynamics,request(scene)),m=model.motion,f=model.frequencyCoordinates.frequenciesMHz;
  assert.deepEqual([m.axis,m.trapBeamId],['3d','tweezer']);assert.equal(m.beams.length,8);
  const {model:h,beams}=hamiltonianAtOrigin(model);assert.ok(largest(h,beams)<=1e-12*largest(beams),`${largest(h,beams)}`);
  const reference=scene.lasers.find(l=>l.id===model.referenceLaserId);assert.equal(f.at(-1),laserFrequencyMHz(reference));
  for(const b of m.beams){
    const laser=scene.lasers.find(l=>l.id===b.id),detuning=b.frequencyIndex.reduce((s,n,k)=>s+n*f[k],0);
    assert.ok(Math.abs(detuning-(laserFrequencyMHz(laser)-laserFrequencyMHz(reference)))<1e-6,b.id);
    assert.deepEqual([b.name,b.waveNumberPerUm,b.intensityMWcm2],[laser.name,2*Math.PI*laserFrequencyMHz(laser)/C,laserIntensityMWcm2(laser)]);
    // The direction on the lab axes; a waist for a beam given by power and waist or NA, none for a plane wave.
    assert.deepEqual(b.direction,directionVector(scene.fields,laser.direction),b.id);assert.equal(b.waistUm,laser.intensityMode==='intensity'?undefined:laserWaistUm(laser),b.id);
    assert.ok(b.coupling.real.every((x,k)=>x===0&&b.coupling.imag[k]===0||(Math.floor(k/N)>=8&&k%N<8)),b.id);
  }
  assert.deepEqual(m.beams.filter(b=>b.waistUm!==undefined).map(b=>[b.id,b.waistUm]),[['tweezer',1],['cool-plus-x',laserWaistUm(cooling)]]);
  assert.deepEqual(m.beams.find(b=>b.id==='tweezer').direction,[0,0,1]);assert.deepEqual(m.beams.find(b=>b.id==='cool-minus-x').direction,[-1,0,0]);
  // H0 without beams is diagonal; the excited states' frame is the reference carrier.
  assert.ok(m.hamiltonian.real.every((x,k)=>x===0||k%(N+1)===0)&&m.hamiltonian.imag.every(x=>x===0));
  assert.deepEqual(m.frameFrequenciesMHz,model.frameFrequenciesMHz);assert.equal(m.massKg,RB87.massKg);assert.equal(m.temperatureUK,30);
  // Another laser designated moves the trap; without motion the model is as before.
  setTweezer(scene,cooling);assert.equal(buildFullMasterModel(scene,scene.dynamics,request(scene)).motion.trapBeamId,'cool-plus-x');
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
    ['the tweezer',s=>{s.tweezerId='cool-plus-x';}],['a waist',s=>{s.lasers[0].waistUm=1.5;s.lasers[0].intensityAnchor=undefined;}]])assert.notEqual(changed(change),key(scene),label);
  // The same intensity at the atom as a plane wave: the internal model is the same, the motion's is not.
  const plane=structuredClone(scene),offPlane=structuredClone(off);for(const s of [plane,offPlane])setLaserIntensityMode(s.lasers[0],'intensity');
  assert.equal(laserIntensityMWcm2(plane.lasers[0]),laserIntensityMWcm2(scene.lasers[0]));
  assert.notEqual(key(plane),key(scene));assert.equal(key(offPlane),key(off));
  // Off, the device and the tweezer are no part of it; the mean is the internal state's either way.
  assert.equal(changed(s=>{s.dynamics.motion.enabled=false;s.dynamics.motion.device='cpu';delete s.tweezerId;}),key(off));
  assert.equal(meanInputsKey(scene,scene.dynamics),meanInputsKey(off,off.dynamics));
  assert.deepEqual(request(scene),{axis:'3d',temperatureUK:30});assert.equal(request(off),undefined);
  assert.equal(plannedSolve(true,{trajectory:false,mean:false},true),'trajectory');
  assert.equal(plannedSolve(true,{trajectory:true,mean:false},true),undefined);
  assert.equal(plannedSolve(false,{trajectory:false,mean:false},true),'mean','the levels\' p̄: the atom at rest, the internal state');
});

test('the native payload carries the motion block with its trap beam, and the device; a malformed motion or device is refused',()=>{
  const scene=pgc(),{motion:_,...dynamics}=scene.dynamics,options={...dynamics,samples:3,includeMean:false,meanOnly:false,motion:request(scene),device:'gpu'};
  const {payload}=prepareNativeSolve({scene,options});
  assert.deepEqual(Object.keys(payload.model.motion).sort(),['axis','beams','emission','fieldVectorG','frameFrequenciesMHz','hamiltonian','massKg','temperatureUK','trapBeamId']);
  assert.deepEqual([payload.model.motion.axis,payload.model.motion.trapBeamId,payload.options.device],['3d','tweezer','gpu']);
  assert.deepEqual(Object.keys(payload.model.motion.beams.find(b=>b.id==='tweezer')).sort(),
    ['counterCoupling','coupling','direction','frequencyIndex','id','intensityMWcm2','name','outsideShift','waistUm','waveNumberPerUm']);
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
  const call=calls[0];assert.deepEqual([call.body.options.motion,call.body.options.device,call.body.options.includeMean],[{axis:'3d',temperatureUK:30},'cpu',false]);
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
test('a motion result is read strictly: the 3D block with the total temperature, the device, a fallback\'s note and the approximation ledger',()=>{
  const result=readMotion(),m=result.motion,block=motionBlock(3);
  assert.deepEqual([m.temperatureUK,m.temperatureErrorUK],[[block.temperatureUK.x,block.temperatureUK.y,block.temperatureUK.z],[[1.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]]]);
  assert.deepEqual([m.temperatureTotalUK,m.temperatureTotalErrorUK,m.trapFrequencyKHz,m.classical],[[31,31,31],[.9,.9,.9],[100.5,100.5,19.3],[true,true,false]]);
  assert.deepEqual([m.device,m.deviceNote,m.approximations],[MOTION_DIAGNOSTICS.device,MOTION_DIAGNOSTICS.deviceNote,MOTION_DIAGNOSTICS.approximations]);
  assert.equal(result.populationAccuracy.requestedTolerance,.05);assert.equal(result.mean.status,'not-requested');
  const flagged=readMotion(d=>{d.diagnostics.approximations.flagged='t ≥ 8 ms 的输出超出近似误差预算';delete d.diagnostics.deviceNote;}).motion;
  assert.equal(flagged.approximations.flagged,'t ≥ 8 ms 的输出超出近似误差预算');assert.equal(Object.hasOwn(flagged,'deviceNote'),false);
  for(const [label,change] of [
    ['no block',d=>{delete d.motion;}],['an axis',d=>{d.motion.axis='x';}],['no total temperature',d=>{delete d.motion.temperatureTotalUK;}],
    ['a total short of a sample',d=>{d.motion.temperatureTotalUK.pop();}],['a negative total error',d=>{d.motion.temperatureTotalErrorUK[1]=-.1;}],
    ['one axis missing',d=>{delete d.motion.temperatureUK.z;}],['no regime',d=>{delete d.motion.classical;}],['no trajectories',d=>{d.motion.trajectories=0;}],
    ['a loss that decreases',d=>{d.motion.lossProbability[2]=0;}],['an unknown key',d=>{d.motion.axes=['x','y','z'];}],
    ['no device',d=>{delete d.diagnostics.device;}],['another device',d=>{d.diagnostics.device='tpu';}],['an empty note',d=>{d.diagnostics.deviceNote=' ';}],
    ['no ledger',d=>{delete d.diagnostics.approximations;}],['a ledger key',d=>{d.diagnostics.approximations.total=1;}],['a negative estimate',d=>{d.diagnostics.approximations.items[0].estimate=-1;}],
    ['an item without its action',d=>{delete d.diagnostics.approximations.items[1].action;}],['an empty flag',d=>{d.diagnostics.approximations.flagged='';}],
  ])assert.throws(()=>readMotion(change),/运动结果/,label);
  assert.throws(()=>readMotion(d=>{d.populationAccuracy.requestedTolerance=.01;}),/布居精度/);
  // A block nobody asked for is refused.
  const scene=pgc(),states=buildFullMasterModel(scene,scene.dynamics).states;
  const unasked=motionResult({states,initialIndex:states.findIndex(s=>s.id===scene.dynamics.initialStateId),timesUs:[0,1,2]});unasked.populationAccuracy.requestedTolerance=.01;
  assert.throws(()=>normalizeDynamicsResult(unasked,scene),/运动结果/);
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
  // A sample outside the validated regime lies on the grey band of both panels, from half-way to its neighbour.
  const bands=svg=>[...svg.matchAll(/<rect class="reference-band" x="([\d.]+)" y="(\d+)" width="([\d.]+)" height="(\d+)"/g)].map(m=>m.slice(1).map(Number));
  const reference=result.motion.classical.map(inside=>!inside);
  assert.deepEqual(bands(seriesChart(result.timesUs,[temperature,occupation],'ms',{reference})).map(([x,y,w,h])=>[x,x+w,y,y+h]),[[599.5,778,32,176],[599.5,778,212,300]]);
  assert.deepEqual(bands(chart),[]);
});

test('the examples: 3D PGC and 光镊 + 探测光 + 回泵光 run motion in their designated tweezer; the internal examples switch it off',()=>{
  assert.equal(EXAMPLES.some(e=>e.key==='pgc'||/径向|1D/.test(e.label)),false,'no 1D example');
  // 3D PGC: the 852 nm tweezer along z, its six σ+σ− cooling beams and one repump, |2,+2⟩ at 30 μK for 10 ms.
  const {scene:pgc3d,view}=exampleScene('pgc3d'),loaded=roundtrip(pgc3d);assert.equal(view,'dynamics');
  assert.equal(loaded.tweezerId,'tweezer');assert.deepEqual(loaded.dynamics.motion,defaultMotion());assert.equal(motionRefusal(loaded),'');
  const trap=loaded.lasers[0];assert.deepEqual([trap.id,trap.frequency,trap.intensityMode,trap.powerMW,trap.waistUm,directionVector(loaded.fields,trap.direction)],['tweezer',{wavelengthNm:852},'power',3.5,1,[0,0,1]]);
  assert.deepEqual([loaded.fields[0].magnitudeG,loaded.dynamics.initialStateId,loaded.dynamics.durationUs,loaded.dynamics.samples],[0,'ground-2-2',TWEEZER_PGC.durationUs,1001]);
  const cooling=loaded.lasers.filter(l=>l.id.startsWith('cool-')),gamma=RB87.D2.gammaHz/(2*Math.PI*1e6);
  assert.deepEqual(cooling.map(l=>directionVector(loaded.fields,l.direction)),[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]);
  for(const l of cooling)assert.deepEqual([l.frequency,l.intensityMWcm2],[{base:{kind:'transition',line:'D2',Fg:2,Fe:3},offsetMHz:TWEEZER_PGC.coolingDetuningGamma*gamma},TWEEZER_PGC.coolingMWcm2]);
  assert.deepEqual(cooling.map(l=>l.ellipticityDeg),[45,45,45,45,-45,-45]);
  const [plus,minus]=cooling.map(l=>polarization(l,loaded.fields));assert.ok(Math.abs(plus.plus-1)<1e-12&&Math.abs(minus.minus-1)<1e-12,`${plus.plus} ${minus.minus}`);
  const repump=loaded.lasers.at(-1);assert.deepEqual([repump.frequency,repump.intensityMWcm2],[{base:{kind:'transition',line:'D2',Fg:1,Fe:2},offsetMHz:0},TWEEZER_PGC.repumpMWcm2]);
  // 光镊 + 探测光 + 回泵光: a Raman-resonant Λ -- the probe on F=2 → F′=2, the repump on F=1 → F′=2, both on resonance -- in the default tweezer, in 3D.
  const three=exampleScene('three'),lambda=roundtrip(three.scene);assert.equal(three.view,'dynamics');assert.equal(lambda.name,'光镊 + 探测光 + 回泵光');
  assert.equal(lambda.tweezerId,'tweezer');assert.equal(tweezerOf(lambda).enabled,true);assert.deepEqual(lambda.dynamics.motion,defaultMotion());assert.equal(motionRefusal(lambda),'');
  assert.deepEqual(lambda.lasers.map(l=>[l.id,l.enabled]),[['probe',true],['tweezer',true],['repump',true]]);
  assert.deepEqual(lambda.lasers.find(l=>l.id==='probe').frequency,{base:{kind:'transition',line:'D2',Fg:2,Fe:2},offsetMHz:0});
  assert.deepEqual(lambda.lasers.find(l=>l.id==='repump').frequency,{base:{kind:'transition',line:'D2',Fg:1,Fe:2},offsetMHz:0});
  assert.deepEqual([lambda.dynamics.durationUs,lambda.dynamics.durationUnit,lambda.dynamics.samples],[LAMBDA_TWEEZER.durationUs,LAMBDA_TWEEZER.durationUnit,LAMBDA_TWEEZER.samples]);
  assert.equal(buildFullMasterModel(lambda,lambda.dynamics,request(lambda)).motion.trapBeamId,'tweezer');
  // The internal physics: the tweezer off, and motion with it.
  for(const key of ['open','cycling','f3-linear','rabi','dark','both','tilted']){
    const s=roundtrip(exampleScene(key).scene);assert.equal(s.dynamics.motion.enabled,false,key);assert.equal(s.lasers.find(l=>l.id==='tweezer')?.enabled,false,key);
  }
});

test('the motion engine\'s sentences are shown in Chinese',()=>{
  assert.equal(solverText('A motion request is a trajectory alone: meanOnly and includeMean are false'),'运动只计算时间演化，不计算长期布居。');
  // The 3D engine's refusals and its device note are written in Chinese: motion along an axis or in
  // a plane, a request for the GPU, and a scene whose light does not confine the atom.
  const adapter=readFileSync(new URL('../rust-solver/src/motion3d_adapter.rs',import.meta.url),'utf8');
  for(const name of ['ONLY_3D','GPU_PENDING'])assert.match(new RegExp(`pub const ${name}:&str="([^"]+)";`).exec(adapter)?.[1]??'',/^[一-鿿]/,name);
  const solve=readFileSync(new URL('../rust-solver/motion-3d/src/solve.rs',import.meta.url),'utf8');
  assert.match(/pub const NOT_CONFINED:&str="([^"]+)";/.exec(solve)?.[1]??'',/^[一-鿿]/);
  // A result short of its statistical target says why in Chinese.
  const partial=[...adapter.matchAll(/format!\("(The statistical error is[^"]*)"\)/g)].map(m=>m[1].replace(/\{[A-Za-z_]*(?::[^}]*)?\}/g,'1'));
  assert.equal(partial.length,1);assert.equal(solverText(partial[0]),'统计误差为 1，目标 1。');
});
