// 1D motion (ui-design.md §6.20): the radial axis through the tweezer focus, the model's motion
// block, the scene file, the request and the strict reader of a motion result.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as pause} from 'node:timers/promises';
import {DEFAULT_MOTION_TEMPERATURE_UK,defaultMotion} from '../dist/js/atom-workbench/src/types.js';
import {RB87,composeMagneticField,laserFrequencyMHz,laserIntensityMWcm2,polarization,resolveSceneFrequencies} from '../dist/js/atom-workbench/src/physics.js';
import {laserWaistUm,setLaserIntensityMode} from '../dist/js/atom-workbench/src/laser-intensity.js';
import {directionVector,invert,rotate} from '../dist/js/atom-workbench/src/direction.js';
import {MOTION_FRAME,focusedBeams,motionAxis,motionRefusal,motionRequest,motionTemperatureReading,radialAxes,readMotionRequest,setMotionTemperature} from '../dist/js/atom-workbench/src/motion.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {TWEEZER_PGC,exampleScene} from '../dist/js/atom-workbench/src/examples.js';
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
/** The tweezer PGC example: the 852 nm tweezer along z, the cooling and repump pairs along ±x, B = 0, motion along x. */
const pgc=()=>exampleScene('pgc').scene;
/** Another focused beam: a copy of the example's tweezer along `direction`. */
const tweezer=(scene,id,name,direction)=>({...structuredClone(scene.lasers.find(l=>l.id==='tweezer')),id,name,direction});

test('the atom moves along a lab axis through the focus perpendicular to every focused beam; refused by name when there is none',()=>{
  const scene=pgc();
  assert.deepEqual(focusedBeams(scene).map(l=>l.id),['tweezer']);assert.deepEqual(radialAxes(scene),['x','y']);
  // The chosen axis when it is radial, else the first radial one.
  scene.dynamics.motion.axis='y';assert.equal(motionAxis(scene,scene.dynamics.motion),'y');
  scene.dynamics.motion.axis='z';assert.equal(motionAxis(scene,scene.dynamics.motion),'x');assert.deepEqual(motionRequest(scene,scene.dynamics),{axis:'x',temperatureUK:30});
  // Plane waves take part at any angle.
  scene.lasers[1].direction=spherical(37,21);resolveSceneFrequencies(scene);assert.deepEqual(radialAxes(scene),['x','y']);assert.equal(motionRefusal(scene),'');
  // A tweezer off the lab axes: the axes perpendicular to it, exactly; none at all is refused, naming it.
  const tilted=pgc();tilted.lasers[0].direction=spherical(90,30);assert.deepEqual(radialAxes(tilted),['z']);
  tilted.lasers[0].direction=spherical(40,30);assert.throws(()=>radialAxes(tilted),/^Error: 1D 运动沿穿过焦点、与聚焦光垂直的坐标轴：没有垂直于「光镊」的坐标轴。$/);
  assert.match(motionRefusal(tilted),/「光镊」/);
  // Every focused beam: two leave one axis, three none.
  const two=pgc();two.lasers.push(tweezer(two,'tweezer-x','光镊 x',axis('+x')));assert.deepEqual(radialAxes(two),['y']);
  two.lasers.push(tweezer(two,'tweezer-y','光镊 y',axis('-y')));assert.throws(()=>radialAxes(two),/没有同时垂直于「光镊」、「光镊 x」、「光镊 y」的坐标轴/);
  // No focused beam: no tweezer. Off, nothing is refused.
  const plain=pgc();setLaserIntensityMode(plain.lasers[0],'intensity');assert.throws(()=>radialAxes(plain),/没有聚焦在原子处的光/);
  plain.dynamics.motion.enabled=false;assert.equal(motionRefusal(plain),'');
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

test('the motion block keeps the beams apart with their geometry; at the atom they add up to the model\'s drives',()=>{
  // Two lasers of one frequency per pair: merged in the drives, two beams each here.
  const scene=pgc(),model=buildFullMasterModel(scene,scene.dynamics,{axis:'x',temperatureUK:30}),m=model.motion,f=model.frequencyCoordinates.frequenciesMHz;
  assert.deepEqual(m.beams.map(b=>b.id).sort(),['cool-minus','cool-plus','repump-minus','repump-plus','tweezer']);
  const {model:h,beams}=hamiltonianAtOrigin(model);assert.ok(largest(h,beams)<=1e-12*largest(beams),`${largest(h,beams)}`);
  const reference=scene.lasers.find(l=>l.id===model.referenceLaserId);assert.equal(f.at(-1),laserFrequencyMHz(reference));
  for(const b of m.beams){
    const laser=scene.lasers.find(l=>l.id===b.id),detuning=b.frequencyIndex.reduce((s,n,k)=>s+n*f[k],0),k=directionVector(scene.fields,laser.direction);
    assert.ok(Math.abs(detuning-(laserFrequencyMHz(laser)-laserFrequencyMHz(reference)))<1e-6,b.id);
    assert.deepEqual([b.name,b.waveNumberPerUm,b.intensityMWcm2],[laser.name,2*Math.PI*laserFrequencyMHz(laser)/C,laserIntensityMWcm2(laser)]);
    // The direction in the motion frame (MOTION_FRAME.x: y, z, then x); a waist for a beam focused at the atom, none for a plane wave.
    assert.deepEqual(b.direction,[k[1],k[2],k[0]],b.id);assert.equal(b.waistUm,laser.intensityMode==='intensity'?undefined:laserWaistUm(laser),b.id);
    // Only excited ← ground entries.
    assert.ok(b.coupling.real.every((x,k)=>x===0&&b.coupling.imag[k]===0||(Math.floor(k/N)>=8&&k%N<8)),b.id);
  }
  assert.deepEqual(m.beams.filter(b=>b.waistUm!==undefined).map(b=>[b.id,b.direction,b.waistUm]),[['tweezer',[0,1,0],1]]);
  assert.deepEqual(m.beams.find(b=>b.id==='cool-minus').direction,[0,0,-1]);
  // H0 without beams is diagonal; the excited states' frame is the reference carrier.
  assert.ok(m.hamiltonian.real.every((x,k)=>x===0||k%(N+1)===0)&&m.hamiltonian.imag.every(x=>x===0));
  assert.deepEqual(m.frameFrequenciesMHz,model.frameFrequenciesMHz);assert.equal(m.massKg,RB87.massKg);assert.equal(m.temperatureUK,30);
  // An axis that is not radial is refused.
  assert.throws(()=>buildFullMasterModel(scene,scene.dynamics,{axis:'z',temperatureUK:30}),/1D 运动的运动轴 z 不与每束聚焦光都垂直（可选：x、y）/);
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
  // A tilted field: the quantization frame is not the lab's. Each axis with the tweezer across it.
  for(const [along,direction] of [['x','+z'],['y','+z'],['z','+x']]){
    const scene=pgc();Object.assign(scene.fields[0],{magnitudeG:2.5,direction:{reference:{kind:'angles',thetaDeg:37,phiDeg:-21},thetaDeg:0,phiDeg:0}});scene.lasers[0].direction=axis(direction);
    const model=buildFullMasterModel(scene,scene.dynamics,{axis:along,temperatureUK:30}),rho={re:Array(N*N).fill(0),im:Array(N*N).fill(0)};
    for(let i=0;i<N;i++)for(let j=0;j<=i;j++){const a=Math.sin(3*i+7*j),b=i===j?0:Math.cos(5*i-2*j);rho.re[i*N+j]=rho.re[j*N+i]=a;rho.im[i*N+j]=b;rho.im[j*N+i]=-b;}
    const cartesian=dissipated(model.motion.emission,rho),spherical$=dissipated(model.collapse,rho);
    assert.ok(largest(cartesian,spherical$)<=1e-13*largest(spherical$),along);
  }
  // The convention: a laser's amplitudes are e_q*·ε on the quantization axis (q = −1, 0, +1), the
  // basis the emission operators are combined in.
  const scene=pgc(),frame=composeMagneticField(scene.fields).frame;
  for(const [theta,phi,psi,chi] of [[0,0,0,45],[90,30,10,-20],[133,-71,-40,7],[180,0,25,45]]){
    const laser={...structuredClone(scene.lasers[1]),direction:spherical(theta,phi),polarizationAngleDeg:psi,ellipticityDeg:chi},p=polarization(laser,scene.fields);
    const c=Math.cos(chi*Math.PI/180),s=Math.sin(chi*Math.PI/180),re=p.u.map(x=>c*x),im=p.v.map(x=>s*x);
    const [xr,yr,zr]=rotate(invert(frame),re),[xi,yi,zi]=rotate(invert(frame),im),h=Math.SQRT1_2;
    const expected=[{re:h*(xr-yi),im:h*(xi+yr)},{re:zr,im:zi},{re:-h*(xr+yi),im:-h*(xi-yr)}];
    expected.forEach((e,q)=>assert.ok(Math.hypot(e.re-p.amplitudes[q].re,e.im-p.amplitudes[q].im)<1e-12,`${theta},${phi},${psi},${chi}: q=${q-1}`));
  }
  // Along the field (here z, at B along z; the tweezer along x) the axial component is the π operator itself.
  const along=pgc();Object.assign(along.fields[0],{magnitudeG:3,direction:axis('+z')});along.lasers[0].direction=axis('+x');
  const model=buildFullMasterModel(along,along.dynamics,{axis:'z',temperatureUK:30});
  const pi=model.collapse[1],axial=model.motion.emission[2];
  assert.ok(axial.real.every((x,k)=>x===pi.real[k])&&axial.imag.every((x,k)=>x===pi.imag[k]));
  assert.deepEqual(MOTION_FRAME.z,[[1,0,0],[0,1,0],[0,0,1]]);
});

test('the scene keeps 1D motion, on or off, with its unit; an invalid one is refused with the field named',()=>{
  const roundtrip=scene=>validateScene(JSON.parse(JSON.stringify(scene)));
  const scene=pgc();assert.deepEqual(roundtrip(scene).dynamics.motion,{enabled:true,axis:'x',temperatureUK:DEFAULT_MOTION_TEMPERATURE_UK,temperatureUnit:'μK'});
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

test('a motion trajectory is its own request: its own key with the beams\' waists, no long-time mean, the mean\'s key unchanged',()=>{
  const scene=pgc(),off=structuredClone(scene);off.dynamics.motion.enabled=false;
  const none=structuredClone(scene);delete none.dynamics.motion;
  assert.equal(trajectoryInputsKey(off,off.dynamics),trajectoryInputsKey(none,none.dynamics),'motion off is no part of the key');
  assert.notEqual(trajectoryInputsKey(scene,scene.dynamics),trajectoryInputsKey(off,off.dynamics));
  const hotter=structuredClone(scene);hotter.dynamics.motion.temperatureUK=40;assert.notEqual(trajectoryInputsKey(hotter,hotter.dynamics),trajectoryInputsKey(scene,scene.dynamics));
  // The same intensity at the atom as a plane wave: the internal model is the same, the motion's is not.
  const plane=structuredClone(scene),offPlane=structuredClone(off);for(const s of [plane,offPlane])setLaserIntensityMode(s.lasers[0],'intensity');
  assert.equal(laserIntensityMWcm2(plane.lasers[0]),laserIntensityMWcm2(scene.lasers[0]));
  assert.notEqual(trajectoryInputsKey(plane,plane.dynamics),trajectoryInputsKey(scene,scene.dynamics));
  assert.equal(trajectoryInputsKey(offPlane,offPlane.dynamics),trajectoryInputsKey(off,off.dynamics));
  assert.equal(meanInputsKey(scene,scene.dynamics),meanInputsKey(off,off.dynamics));
  assert.deepEqual(motionRequest(scene,scene.dynamics),{axis:'x',temperatureUK:30});assert.equal(motionRequest(off,off.dynamics),undefined);
  assert.equal(plannedSolve(true,{trajectory:false,mean:false},true),'trajectory');
  assert.equal(plannedSolve(true,{trajectory:true,mean:false},true),undefined);
  assert.equal(plannedSolve(false,{trajectory:false,mean:false},true),'mean','the levels\' long-time mode keeps its internal mean');
});

/** A contract-valid motion block: per sample T, ⟨v⟩, n̄ and the loss; distributions at the first and last sample. */
function motionBlock(samples,axis='x'){
  const velocities=[-.1,-.05,0,.05,.1],positions=[-.4,-.2,0,.2,.4],p=[.1,.2,.4,.2,.1];
  return {axis,temperatureUK:Array(samples).fill(30),meanVelocityMps:Array(samples).fill(0),trapFrequencyKHz:123.4,meanOccupation:Array(samples).fill(4.5),
    lossProbability:Array.from({length:samples},(_,i)=>i*.01),momentum:{velocitiesMps:velocities,samples:[0,samples-1],probabilities:[p,p]},
    position:{positionsUm:positions,samples:[0,samples-1],probabilities:[p,p]}};
}
function motionResult(scene,samples=3){
  const timesUs=Array.from({length:samples},(_,i)=>scene.dynamics.durationUs*i/(samples-1)),model=buildFullMasterModel(scene,scene.dynamics);
  const data=nativeResult({states:model.states,initialIndex:model.states.findIndex(s=>s.id===scene.dynamics.initialStateId),timesUs,mean:false});
  return {...data,motion:motionBlock(samples),assumptions:['test']};
}
test('a motion result is read strictly: its block is there exactly when asked, on the request\'s axis and samples',()=>{
  const scene=pgc(),request=motionRequest(scene,scene.dynamics),read=data=>normalizeDynamicsResult(data,scene,undefined,undefined,request);
  const result=read(motionResult(scene)),block=motionBlock(3);
  assert.deepEqual(result.motion,{...block,momentum:{grid:block.momentum.velocitiesMps,samples:[0,2],probabilities:block.momentum.probabilities},
    position:{grid:block.position.positionsUm,samples:[0,2],probabilities:block.position.probabilities}});
  assert.equal(result.mean.status,'not-requested');
  const without=motionResult(scene);delete without.motion;assert.throws(()=>read(without),/运动结果/);
  assert.throws(()=>normalizeDynamicsResult(motionResult(scene),scene),/运动结果/,'a block nobody asked for');
  for(const [label,change] of [
    ['another axis',m=>{m.axis='y';}],['a temperature short of a sample',m=>{m.temperatureUK.pop();}],['a negative temperature',m=>{m.temperatureUK[1]=-1;}],
    ['a nonfinite velocity',m=>{m.meanVelocityMps[0]=NaN;}],['no trap frequency',m=>{delete m.trapFrequencyKHz;}],['a trap frequency of zero',m=>{m.trapFrequencyKHz=0;}],
    ['an occupation short of a sample',m=>{m.meanOccupation.pop();}],['a loss above one',m=>{m.lossProbability[2]=1.5;}],['a loss that decreases',m=>{m.lossProbability[2]=0;}],
    ['a grid out of order',m=>{m.momentum.velocitiesMps[1]=-.2;}],['a position grid out of order',m=>{m.position.positionsUm[1]=-.5;}],
    ['no distribution at the last sample',m=>{m.momentum.samples=[0,1];}],['no position at the last sample',m=>{m.position.samples=[0,1];}],
    ['a distribution not summing to one',m=>{m.momentum.probabilities[1]=[.1,.2,.4,.2,.2];}],['a negative probability',m=>{m.position.probabilities[0]=[-.1,.4,.4,.2,.1];}],
    ['no position',m=>{delete m.position;}],['an unknown field',m=>{m.trajectories=64;}],
  ]){const data=motionResult(scene);change(data.motion);assert.throws(()=>read(data),/运动结果/,label);}
});

test('the client asks for the motion with a trajectory alone, along the radial axis, and reads its result',async t=>{
  const calls=[],client=new DynamicsClient({fetch:async(url,init)=>new Promise(resolve=>calls.push({resolve,body:JSON.parse(init.body)})),debounceMs:0,clientId:'motion'});t.after(()=>client.dispose());
  const scene=pgc();scene.dynamics.samples=3;scene.dynamics.motion.axis='z';
  const pending=client.solve(scene,scene.dynamics,{includeMean:false});
  for(let i=0;i<100&&!calls.length;i++)await pause(2);
  const call=calls[0];assert.deepEqual(call.body.options.motion,{axis:'x',temperatureUK:30});assert.equal(call.body.options.includeMean,false);
  call.resolve(new Response(JSON.stringify({...motionResult(scene),request:call.body.request,service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA]}}),{status:200,headers:{'Content-Type':'application/json'}}));
  const result=await pending;assert.deepEqual([result.motion.temperatureUK,result.motion.trapFrequencyKHz],[[30,30,30],123.4]);
  // Off, no motion is asked for.
  scene.dynamics.motion.enabled=false;const plain=client.solve(scene,scene.dynamics).catch(e=>e);for(let i=0;i<100&&calls.length<2;i++)await pause(2);
  assert.equal(calls[1].body.options.motion,undefined);client.cancel();await plain;
});

test('the native payload of a motion solve carries the motion block; a malformed motion, one with a mean or no radial axis is refused',()=>{
  const scene=pgc(),options={...scene.dynamics,samples:3,includeMean:false,meanOnly:false,motion:{axis:'x',temperatureUK:30}};
  const {payload}=prepareNativeSolve({scene,options});
  assert.deepEqual(Object.keys(payload.model.motion).sort(),['axis','beams','emission','fieldVectorG','frameFrequenciesMHz','hamiltonian','massKg','temperatureUK']);
  assert.equal(payload.model.motion.beams.length,5);assert.equal(payload.model.motion.emission.length,3);
  assert.deepEqual(Object.keys(payload.model.motion.beams.find(b=>b.id==='tweezer')).sort(),
    ['counterCoupling','coupling','direction','frequencyIndex','id','intensityMWcm2','name','outsideShift','waistUm','waveNumberPerUm']);
  // Each beam's static shift through the states outside the model: Hermitian; the tweezer's on |2,+2⟩ about −0.11 MHz at its focus.
  const shift=payload.model.motion.beams.find(b=>b.id==='tweezer').outsideShift,two=payload.model.states.indexOf('ground-2-2');
  const size=Math.max(...shift.real.map(Math.abs)),transposed=k=>(k%N)*N+Math.floor(k/N);
  assert.ok(shift.real.every((x,k)=>Math.abs(x-shift.real[transposed(k)])<=1e-12*size)&&shift.imag.every((x,k)=>Math.abs(x+shift.imag[transposed(k)])<=1e-12*size));
  assert.ok(Math.abs(shift.real[two*N+two]/(2*Math.PI)+.113)<.01,`${shift.real[two*N+two]/(2*Math.PI)} MHz`);
  assert.equal(prepareNativeSolve({scene,options:{...options,motion:undefined}}).payload.model.motion,undefined);
  for(const motion of [{axis:'w',temperatureUK:30},{axis:'x',temperatureUK:-1},{axis:'x',temperatureUK:30,trajectories:64},null]){
    assert.throws(()=>prepareNativeSolve({scene,options:{...options,motion}}),error=>error.code==='invalid-options');
  }
  assert.throws(()=>prepareNativeSolve({scene,options:{...options,includeMean:true}}),error=>error.code==='invalid-options');
  assert.deepEqual(readMotionRequest({axis:'y',temperatureUK:12.5}),{axis:'y',temperatureUK:12.5});
  // A plane wave at any angle takes part; a tweezer with no lab axis across it refuses the motion, naming it.
  const oblique=pgc();oblique.lasers[1].direction=spherical(30,0);assert.equal(prepareNativeSolve({scene:oblique,options}).payload.model.motion.beams.length,5);
  oblique.lasers[0].direction=spherical(40,30);assert.throws(()=>prepareNativeSolve({scene:oblique,options}),/没有垂直于「光镊」的坐标轴/);
});

test('the tweezer PGC example: the 852 nm tweezer along z (π to the x quantization axis), lin⊥lin cooling and repump pairs along ±x, B = 0, |2,+2⟩ at 30 μK, motion along x',()=>{
  const {scene,view}=exampleScene('pgc');assert.equal(view,'dynamics');
  const loaded=validateScene(JSON.parse(JSON.stringify(scene)));
  assert.equal(loaded.fields[0].magnitudeG,0);assert.equal(loaded.dynamics.initialStateId,'ground-2-2');
  assert.deepEqual(directionVector(loaded.fields,loaded.fields[0].direction),[1,0,0],'the quantization axis is the motion axis');
  assert.deepEqual([loaded.dynamics.durationUs,loaded.dynamics.durationUnit,loaded.dynamics.samples],[TWEEZER_PGC.durationUs,'ms',1001]);
  assert.deepEqual(loaded.dynamics.motion,{enabled:true,axis:'x',temperatureUK:30,temperatureUnit:'μK'});
  const [trap,plus,minus,repumpPlus,repumpMinus]=loaded.lasers;
  assert.deepEqual([trap.id,trap.frequency,trap.intensityMode,trap.powerMW,trap.waistUm,directionVector(loaded.fields,trap.direction)],['tweezer',{wavelengthNm:852},'power',3.5,1,[0,0,1]]);
  assert.deepEqual(polarization(trap,loaded.fields).u.map(x=>Math.round(x*1e12)/1e12+0),[1,0,0]);
  assert.deepEqual([plus,minus,repumpPlus,repumpMinus].map(l=>directionVector(loaded.fields,l.direction)[0]),[1,-1,1,-1]);
  assert.deepEqual([plus.intensityMWcm2,minus.intensityMWcm2,repumpPlus.intensityMWcm2,repumpMinus.intensityMWcm2],[3,3,.5,.5]);
  const gamma=RB87.D2.gammaHz/(2*Math.PI*1e6);
  assert.deepEqual(plus.frequency,{base:{kind:'transition',line:'D2',Fg:2,Fe:3},offsetMHz:TWEEZER_PGC.coolingDetuningGamma*gamma});
  assert.deepEqual(repumpPlus.frequency,{base:{kind:'transition',line:'D2',Fg:1,Fe:2},offsetMHz:0});
  // lin⊥lin: the two cooling beams' linear polarizations are perpendicular, both across x.
  const u=l=>polarization(l,loaded.fields).u;const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  assert.ok(Math.abs(dot(u(plus),u(minus)))<1e-12&&Math.abs(u(plus)[0])<1e-12&&Math.abs(u(minus)[0])<1e-12);
  assert.deepEqual(radialAxes(loaded),['x','y']);
});

test('the motion core\'s sentences are shown in Chinese',()=>{
  const adapter=readFileSync(new URL('../rust-solver/src/motion_core_adapter.rs',import.meta.url),'utf8');
  const partial=[...adapter.matchAll(/format!\("(The error bound is[^"]*)"/g)].map(m=>m[1].replace(/\{[A-Za-z_]*(?::[^}]*)?\}/g,'1'));
  assert.equal(partial.length,1);assert.match(solverText(partial[0]),/^误差界为 1，误差目标 1。$/);
  assert.match(solverText('A motion request is a trajectory alone: meanOnly and includeMean are false'),/1D 运动/);
  // The core's refusals are written in Chinese.
  const core=readFileSync(new URL('../rust-solver/motion-core/src/lib.rs',import.meta.url),'utf8');
  for(const name of ['NO_TRAP','NOT_CONFINED'])assert.match(new RegExp(`pub const ${name}:&str="([^"]+)";`).exec(core)?.[1]??'',/^[一-鿿]/,name);
});
