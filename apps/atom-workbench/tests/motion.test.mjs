// 1D motion (ui-design.md §6.20): which beams take part, the model's motion block, the scene file,
// the request and the strict reader of a motion result.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as pause} from 'node:timers/promises';
import {DEFAULT_MOTION_TEMPERATURE_UK,defaultMotion,defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {RB87,composeMagneticField,laserFrequencyMHz,polarization,resolveSceneFrequencies} from '../dist/js/atom-workbench/src/physics.js';
import {invert,rotate} from '../dist/js/atom-workbench/src/direction.js';
import {MOTION_FRAME,motionBeams,motionRefusal,motionRequest,motionTemperatureReading,readMotionRequest,setMotionTemperature} from '../dist/js/atom-workbench/src/motion.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {exampleScene} from '../dist/js/atom-workbench/src/examples.js';
import {meanInputsKey,trajectoryInputsKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {plannedSolve} from '../dist/js/atom-workbench/src/solve-plan.js';
import {DynamicsClient,normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {solverText} from '../dist/js/atom-workbench/src/solver-words.js';
import {typedDigits} from '../dist/js/atom-workbench/src/unit-input.js';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';
import {nativeResult} from './native-result.mjs';
import {spherical} from './directions.mjs';
import {readFileSync} from 'node:fs';

const N=32,C=299792458;
const axis=name=>({reference:{kind:'axis',axis:name},thetaDeg:0,phiDeg:0});
/** The PGC example: cooling and repump pairs along ±z, B = 0. */
const pgc=()=>exampleScene('pgc').scene;
/** The PGC example with motion along `along` (its own settings otherwise). */
function moving(scene,along='z'){scene.dynamics.motion={...defaultMotion(),axis:along};return scene;}

test('the beams along the axis are ±1, a perpendicular beam 0, decided exactly; the PGC example takes part whole',()=>{
  const scene=pgc();
  assert.deepEqual(motionBeams(scene,'z').map(b=>[b.laser.id,b.axial]),[['cool-plus',1],['cool-minus',-1],['repump-plus',1],['repump-minus',-1]]);
  // Along x the same beams are perpendicular to the axis: uniform along it.
  assert.deepEqual(motionBeams(scene,'x').map(b=>b.axial),[0,0,0,0]);
  // A beam in the xy plane at any azimuth is exactly perpendicular to z; a laser that is off, or dark, takes no part.
  scene.lasers.push({...structuredClone(scene.lasers[0]),id:'side',name:'侧光',direction:spherical(90,37)},{...structuredClone(scene.lasers[0]),id:'off',name:'关闭',enabled:false,direction:spherical(45,0)},
    {...structuredClone(scene.lasers[0]),id:'dark',name:'无光',intensityMWcm2:0,direction:spherical(45,0)});
  resolveSceneFrequencies(scene);
  assert.deepEqual(motionBeams(scene,'z').map(b=>[b.laser.id,b.axial]).slice(4),[['side',0]]);
});

test('an oblique beam, and a beam along the axis off the D2 line, are refused by name',()=>{
  const scene=pgc();scene.lasers[2].direction=spherical(45,0);
  assert.throws(()=>motionBeams(scene,'z'),/^Error: 1D 运动不支持当前光束：「回泵光 \+z」（与 z 轴成 45°）与运动轴既不平行也不垂直/);
  // The 852 nm tweezer along the axis: another wavelength line; the same tweezer across the axis takes part.
  const tweezer=moving(defaultScene());tweezer.lasers[0].enabled=false;tweezer.lasers[1].direction=axis('+z');resolveSceneFrequencies(tweezer);
  assert.match(motionRefusal(tweezer),/「光镊」（852 nm）沿运动轴传播，但不在 D2 线上/);
  tweezer.dynamics.motion.axis='y';assert.equal(motionRefusal(tweezer),'');
  // A D1 beam along the axis likewise.
  const d1=pgc();d1.lasers[0].frequency={base:{kind:'transition',line:'D1',Fg:2,Fe:2},offsetMHz:0};delete d1.lasers[0].frequencyAnchor;resolveSceneFrequencies(d1);
  assert.match(motionRefusal(moving(d1)),/「冷却光 \+z」（795 nm）沿运动轴传播，但不在 D2 线上/);
  // Off, nothing is refused.
  d1.dynamics.motion.enabled=false;assert.equal(motionRefusal(d1),'');
});

/** Σ_j drive.cos + (H0 − H0 without beams) = H(t = 0, x = 0), from the model; and from the beams. */
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

test('the motion block keeps the beams apart: at x = 0 they add up to the model\'s drives, each on its own frequency coordinate',()=>{
  // Two lasers of one frequency per pair: merged in the drives, two beams each here.
  const scene=pgc(),model=buildFullMasterModel(scene,scene.dynamics,{axis:'z',temperatureUK:30}),m=model.motion,f=model.frequencyCoordinates.frequenciesMHz;
  assert.deepEqual(m.beams.map(b=>[b.id,b.axial]).sort(),[['cool-minus',-1],['cool-plus',1],['repump-minus',-1],['repump-plus',1]]);
  const {model:h,beams}=hamiltonianAtOrigin(model);assert.ok(largest(h,beams)<=1e-12*largest(beams),`${largest(h,beams)}`);
  const reference=scene.lasers.find(l=>l.id===model.referenceLaserId);assert.equal(f.at(-1),laserFrequencyMHz(reference));
  for(const b of m.beams){
    const laser=scene.lasers.find(l=>l.id===b.id),detuning=b.frequencyIndex.reduce((s,n,k)=>s+n*f[k],0);
    assert.ok(Math.abs(detuning-(laserFrequencyMHz(laser)-laserFrequencyMHz(reference)))<1e-6,b.id);
    assert.equal(b.waveNumberPerUm,2*Math.PI*laserFrequencyMHz(laser)/C);
    // Only excited ← ground entries.
    assert.ok(b.coupling.real.every((x,k)=>x===0&&b.coupling.imag[k]===0||(Math.floor(k/N)>=8&&k%N<8)),b.id);
  }
  // H0 without beams is diagonal; the excited states' frame is the reference carrier.
  assert.ok(m.hamiltonian.real.every((x,k)=>x===0||k%(N+1)===0)&&m.hamiltonian.imag.every(x=>x===0));
  assert.deepEqual(m.frameFrequenciesMHz,model.frameFrequenciesMHz);assert.equal(m.massKg,RB87.massKg);assert.equal(m.temperatureUK,30);
  // Without motion the model is as before: no motion block, the same matrices.
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
test('the Cartesian emission operators are the model\'s spontaneous emission, in the drives\' polarization convention',()=>{
  // A tilted field: the quantization frame is not the lab's.
  const scene=pgc();Object.assign(scene.fields[0],{magnitudeG:2.5,direction:{reference:{kind:'angles',thetaDeg:37,phiDeg:-21},thetaDeg:0,phiDeg:0}});
  for(const along of ['x','y','z']){
    const model=buildFullMasterModel(scene,scene.dynamics,{axis:along,temperatureUK:30}),rho={re:Array(N*N).fill(0),im:Array(N*N).fill(0)};
    for(let i=0;i<N;i++)for(let j=0;j<=i;j++){const a=Math.sin(3*i+7*j),b=i===j?0:Math.cos(5*i-2*j);rho.re[i*N+j]=rho.re[j*N+i]=a;rho.im[i*N+j]=b;rho.im[j*N+i]=-b;}
    const cartesian=dissipated(model.motion.emission,rho),spherical$=dissipated(model.collapse,rho);
    assert.ok(largest(cartesian,spherical$)<=1e-13*largest(spherical$),along);
  }
  // The convention: a laser's amplitudes are e_q*·ε on the quantization axis (q = −1, 0, +1), the
  // basis the emission operators are combined in.
  const frame=composeMagneticField(scene.fields).frame;
  for(const [theta,phi,psi,chi] of [[0,0,0,45],[90,30,10,-20],[133,-71,-40,7],[180,0,25,45]]){
    const laser={...structuredClone(scene.lasers[0]),direction:spherical(theta,phi),polarizationAngleDeg:psi,ellipticityDeg:chi},p=polarization(laser,scene.fields);
    const c=Math.cos(chi*Math.PI/180),s=Math.sin(chi*Math.PI/180),re=p.u.map(x=>c*x),im=p.v.map(x=>s*x);
    const [xr,yr,zr]=rotate(invert(frame),re),[xi,yi,zi]=rotate(invert(frame),im),h=Math.SQRT1_2;
    const expected=[{re:h*(xr-yi),im:h*(xi+yr)},{re:zr,im:zi},{re:-h*(xr+yi),im:-h*(xi-yr)}];
    expected.forEach((e,q)=>assert.ok(Math.hypot(e.re-p.amplitudes[q].re,e.im-p.amplitudes[q].im)<1e-12,`${theta},${phi},${psi},${chi}: q=${q-1}`));
  }
  // Along the field (here z, at B along z) the axial component is the π operator itself.
  const along=pgc();along.fields[0].magnitudeG=3;
  const model=buildFullMasterModel(along,along.dynamics,{axis:'z',temperatureUK:30});
  const pi=model.collapse[1],axial=model.motion.emission[2];
  assert.ok(axial.real.every((x,k)=>x===pi.real[k])&&axial.imag.every((x,k)=>x===pi.imag[k]));
  assert.deepEqual(MOTION_FRAME.z,[[1,0,0],[0,1,0],[0,0,1]]);
});

test('the scene keeps 1D motion, on or off, with its unit; an invalid one is refused with the field named',()=>{
  const roundtrip=scene=>validateScene(JSON.parse(JSON.stringify(scene)));
  assert.equal(defaultScene().dynamics.motion,undefined,'the default scene has no motion');
  const scene=pgc();assert.deepEqual(roundtrip(scene).dynamics.motion,{enabled:true,axis:'z',temperatureUK:DEFAULT_MOTION_TEMPERATURE_UK,temperatureUnit:'μK'});
  scene.dynamics.motion={enabled:false,axis:'y',temperatureUK:1500,temperatureUnit:'mK'};assert.deepEqual(roundtrip(scene).dynamics.motion,scene.dynamics.motion);
  for(const [change,reason] of [
    [m=>{m.axis='w';},/dynamics\.motion\.axis/],[m=>{m.temperatureUK=0;},/dynamics\.motion\.temperatureUK/],[m=>{m.temperatureUnit='nK';},/dynamics\.motion\.temperatureUnit/],
    [m=>{delete m.enabled;},/dynamics\.motion 缺少 enabled/],[m=>{m.trajectories=64;},/dynamics\.motion 含有未知字段 trajectories/],
  ]){const bad=pgc();change(bad.dynamics.motion);assert.throws(()=>roundtrip(bad),reason);}
});

test('the initial temperature keeps its digits when its unit changes (§0.4); it is finite and positive',()=>{
  const motion=defaultMotion();assert.deepEqual(motionTemperatureReading(motion),{value:30,unit:'μK'});
  setMotionTemperature(motion,typedDigits(motionTemperatureReading(motion).value),'mK');
  assert.deepEqual([motion.temperatureUK,motion.temperatureUnit],[30000,'mK']);assert.deepEqual(motionTemperatureReading(motion),{value:30,unit:'mK'});
  for(const value of [0,-1,NaN,Infinity])assert.throws(()=>setMotionTemperature(motion,value,'μK'),/温度必须是有限正数/);
  assert.deepEqual([motion.temperatureUK,motion.temperatureUnit],[30000,'mK'],'a refused temperature leaves it as it was');
});

test('a motion trajectory is its own request: its own key, no long-time mean, the mean\'s key unchanged',()=>{
  const scene=pgc(),off=structuredClone(scene);off.dynamics.motion.enabled=false;
  const none=structuredClone(scene);delete none.dynamics.motion;
  assert.equal(trajectoryInputsKey(off,off.dynamics),trajectoryInputsKey(none,none.dynamics),'motion off is no part of the key');
  assert.notEqual(trajectoryInputsKey(scene,scene.dynamics),trajectoryInputsKey(off,off.dynamics));
  const hotter=structuredClone(scene);hotter.dynamics.motion.temperatureUK=40;assert.notEqual(trajectoryInputsKey(hotter,hotter.dynamics),trajectoryInputsKey(scene,scene.dynamics));
  assert.equal(meanInputsKey(scene,scene.dynamics),meanInputsKey(off,off.dynamics));
  assert.deepEqual(motionRequest(scene.dynamics),{axis:'z',temperatureUK:30});assert.equal(motionRequest(off.dynamics),undefined);
  assert.equal(plannedSolve(true,{trajectory:false,mean:false},true),'trajectory');
  assert.equal(plannedSolve(true,{trajectory:true,mean:false},true),undefined);
  assert.equal(plannedSolve(false,{trajectory:false,mean:false},true),'mean','the levels\' long-time mode keeps its internal mean');
});

/** A contract-valid motion block: T and ⟨v⟩ per sample, distributions at the first and last sample. */
function motionBlock(samples,axis='z'){
  const velocities=[-.1,-.05,0,.05,.1],p=[.1,.2,.4,.2,.1];
  return {axis,temperatureUK:Array(samples).fill(30),meanVelocityMps:Array(samples).fill(0),momentum:{velocitiesMps:velocities,samples:[0,samples-1],probabilities:[p,p]}};
}
function motionResult(scene,samples=3){
  const timesUs=Array.from({length:samples},(_,i)=>scene.dynamics.durationUs*i/(samples-1)),model=buildFullMasterModel(scene,scene.dynamics);
  const data=nativeResult({states:model.states,initialIndex:model.states.findIndex(s=>s.id===scene.dynamics.initialStateId),timesUs,mean:false});
  return {...data,motion:motionBlock(samples),assumptions:['test']};
}
test('a motion result is read strictly: its block is there exactly when asked, on the request\'s axis and samples',()=>{
  const scene=pgc(),request=motionRequest(scene.dynamics),read=data=>normalizeDynamicsResult(data,scene,undefined,undefined,request);
  const result=read(motionResult(scene));assert.deepEqual(result.motion,motionBlock(3));assert.equal(result.mean.status,'not-requested');
  const without=motionResult(scene);delete without.motion;assert.throws(()=>read(without),/运动结果/);
  assert.throws(()=>normalizeDynamicsResult(motionResult(scene),scene),/运动结果/,'a block nobody asked for');
  for(const [label,change] of [
    ['another axis',m=>{m.axis='x';}],['a temperature short of a sample',m=>{m.temperatureUK.pop();}],['a negative temperature',m=>{m.temperatureUK[1]=-1;}],
    ['a nonfinite velocity',m=>{m.meanVelocityMps[0]=NaN;}],['a grid out of order',m=>{m.momentum.velocitiesMps[1]=-.2;}],
    ['no distribution at the last sample',m=>{m.momentum.samples=[0,1];}],['a distribution not summing to one',m=>{m.momentum.probabilities[1]=[.1,.2,.4,.2,.2];}],
    ['a negative probability',m=>{m.momentum.probabilities[0]=[-.1,.4,.4,.2,.1];}],['an unknown field',m=>{m.trajectories=64;}],
  ]){const data=motionResult(scene);change(data.motion);assert.throws(()=>read(data),/运动结果/,label);}
});

test('the client asks for the motion with a trajectory alone, and reads its result',async t=>{
  const calls=[],client=new DynamicsClient({fetch:async(url,init)=>new Promise(resolve=>calls.push({resolve,body:JSON.parse(init.body)})),debounceMs:0,clientId:'motion'});t.after(()=>client.dispose());
  const scene=pgc();scene.dynamics.samples=3;
  const pending=client.solve(scene,scene.dynamics,{includeMean:false});
  for(let i=0;i<100&&!calls.length;i++)await pause(2);
  const call=calls[0];assert.deepEqual(call.body.options.motion,{axis:'z',temperatureUK:30});assert.equal(call.body.options.includeMean,false);
  call.resolve(new Response(JSON.stringify({...motionResult(scene),request:call.body.request,service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA]}}),{status:200,headers:{'Content-Type':'application/json'}}));
  assert.deepEqual((await pending).motion.temperatureUK,[30,30,30]);
  // Off, no motion is asked for.
  scene.dynamics.motion.enabled=false;const plain=client.solve(scene,scene.dynamics).catch(e=>e);for(let i=0;i<100&&calls.length<2;i++)await pause(2);
  assert.equal(calls[1].body.options.motion,undefined);client.cancel();await plain;
});

test('the native payload of a motion solve carries the motion block; a malformed motion or one with a mean is refused',()=>{
  const scene=pgc(),options={...scene.dynamics,samples:3,includeMean:false,meanOnly:false,motion:{axis:'z',temperatureUK:30}};
  const {payload}=prepareNativeSolve({scene,options});
  assert.deepEqual(Object.keys(payload.model.motion).sort(),['axis','beams','emission','fieldVectorG','frameFrequenciesMHz','hamiltonian','massKg','temperatureUK']);
  assert.equal(payload.model.motion.beams.length,4);assert.equal(payload.model.motion.emission.length,3);
  assert.equal(prepareNativeSolve({scene,options:{...options,motion:undefined}}).payload.model.motion,undefined);
  for(const motion of [{axis:'w',temperatureUK:30},{axis:'z',temperatureUK:-1},{axis:'z',temperatureUK:30,trajectories:64},null]){
    assert.throws(()=>prepareNativeSolve({scene,options:{...options,motion}}),error=>error.code==='invalid-options');
  }
  assert.throws(()=>prepareNativeSolve({scene,options:{...options,includeMean:true}}),error=>error.code==='invalid-options');
  assert.deepEqual(readMotionRequest({axis:'y',temperatureUK:12.5}),{axis:'y',temperatureUK:12.5});
  // Beams with no 1D form are refused by the model, naming them.
  const oblique=pgc();oblique.lasers[0].direction=spherical(30,0);
  assert.throws(()=>prepareNativeSolve({scene:oblique,options}),/「冷却光 \+z」（与 z 轴成 30°）/);
});

test('the 1D PGC example: lin⊥lin cooling pair 3Γ below F=2 → F′=3, a resonant repump pair, B = 0, |2,+2⟩ at 30 μK for 1 ms',()=>{
  const {scene,view}=exampleScene('pgc');assert.equal(view,'dynamics');
  const loaded=validateScene(JSON.parse(JSON.stringify(scene)));
  assert.equal(loaded.fields[0].magnitudeG,0);assert.equal(loaded.dynamics.initialStateId,'ground-2-2');
  assert.deepEqual([loaded.dynamics.durationUs,loaded.dynamics.durationUnit,loaded.dynamics.samples],[1000,'ms',201]);
  assert.deepEqual(loaded.dynamics.motion,{enabled:true,axis:'z',temperatureUK:30,temperatureUnit:'μK'});
  const [plus,minus,repumpPlus,repumpMinus]=loaded.lasers;
  assert.deepEqual([plus.intensityMWcm2,minus.intensityMWcm2,repumpPlus.intensityMWcm2,repumpMinus.intensityMWcm2],[.7,.7,.1,.1]);
  assert.equal(plus.frequency.offsetMHz,-3*RB87.D2.gammaHz/(2*Math.PI*1e6));assert.deepEqual(plus.frequency.base,{kind:'transition',line:'D2',Fg:2,Fe:3});
  assert.deepEqual(repumpPlus.frequency,{base:{kind:'transition',line:'D2',Fg:1,Fe:2},offsetMHz:0});
  // lin⊥lin: the two cooling beams' polarizations are x and y.
  const u=l=>polarization(l,loaded.fields).u.map(x=>Math.round(x*1e12)/1e12+0);
  assert.deepEqual([u(plus),u(minus)],[[1,0,0],[0,1,0]]);
});

test('the motion core\'s sentences are shown in Chinese',()=>{
  const method=/const METHOD:&str="([^"]+)";/.exec(readFileSync(new URL('../rust-solver/motion-core/src/free_flight.rs',import.meta.url),'utf8'))?.[1];
  assert.ok(method);assert.match(solverText(method),/^精确的自由飞行/);
  const adapter=readFileSync(new URL('../rust-solver/src/motion_core_adapter.rs',import.meta.url),'utf8');
  const partial=[...adapter.matchAll(/format!\("(The error bound is[^"]*)"/g)].map(m=>m[1].replace(/\{[A-Za-z_]*(?::[^}]*)?\}/g,'1'));
  assert.equal(partial.length,1);assert.match(solverText(partial[0]),/^误差界为 1，误差目标 1。$/);
  assert.match(solverText('A motion request is a trajectory alone: meanOnly and includeMean are false'),/1D 运动/);
});
