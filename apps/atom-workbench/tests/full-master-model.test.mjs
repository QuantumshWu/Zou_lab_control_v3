import test from 'node:test';import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {STATIC_MIXING_LIMIT,displayMixing,spectroscopyAnalysis,spectroscopyCoupling} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {classifyLasers,farValidity} from '../dist/js/atom-workbench/src/laser-regime.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {normalizeLaserFrequency} from '../dist/js/atom-workbench/src/laser-frequency.js';
import {levelsChart} from '../dist/js/atom-workbench/src/charts.js';
import {RB87,accurateSum,laserFrequencyDifferenceMHz,laserFrequencyMHz,laserIntensityMWcm2} from '../dist/js/atom-workbench/src/physics.js';
import {CORE_POLARIZABILITY_AU,VIRTUAL_MANIFOLDS} from '../dist/js/atom-workbench/src/spectroscopy-data.js';

function threeNearBeams(){
 const scene=defaultScene(),probe=scene.lasers[0];
 probe.referenceTransition={line:'D2',Fg:2,Fe:3};probe.ellipticityDeg=17;probe.thetaDeg=7;normalizeLaserFrequency(probe);
 const repump={...probe,id:'repump',referenceTransition:{line:'D2',Fg:1,Fe:2},phaseDeg:19};normalizeLaserFrequency(repump);
 const third={...probe,id:'third',referenceTransition:{line:'D2',Fg:2,Fe:2},phaseDeg:-31};normalizeLaserFrequency(third);
 scene.lasers.push(repump,third);return scene;
}
function hamiltonianAt(model,time){
 const phases=model.frequencyCoordinates.frequencies.map(frequency=>frequency*time);
 return {real:model.hReal.map((value,i)=>value+model.drives.reduce((sum,drive,j)=>{
  const phase=accurateSum(model.frequencyCoordinates.driveIndices[j].map((integer,k)=>integer*phases[k]));
  return sum+drive.cos.real[i]*Math.cos(phase)+drive.sin.real[i]*Math.sin(phase);
 },0)),imag:model.hImag.map((value,i)=>value+model.drives.reduce((sum,drive,j)=>{
  const phase=accurateSum(model.frequencyCoordinates.driveIndices[j].map((integer,k)=>integer*phases[k]));
  return sum+drive.cos.imag[i]*Math.cos(phase)+drive.sin.imag[i]*Math.sin(phase);
 },0))};
}

test('three near carriers retain their exact two-axis phase provenance including cross-F differences',()=>{
 const model=buildFullMasterModel(threeNearBeams()),coordinates=model.frequencyCoordinates;
 assert.equal(model.laserCount,3);assert.equal(coordinates.frequencies.length,2);assert.equal(model.drives.length,3);
 assert.ok(coordinates.frequencies.every(frequency=>frequency>0&&Number.isFinite(frequency)));
 assert.deepEqual(coordinates.frequencies,coordinates.frequenciesMHz.map(frequency=>2*Math.PI*frequency));
 assert.equal(coordinates.driveIndices.length,model.drives.length);
 for(const [j,index]of coordinates.driveIndices.entries()){
  assert.equal(index.length,2);assert.ok(index.every(Number.isInteger));assert.ok(index.some(value=>value!==0));
  assert.equal(model.drives[j].omega,accurateSum(index.map((value,k)=>value*coordinates.frequencies[k])));
 }
 const ordered=model.drives.map((drive,j)=>({omega:drive.omega,index:coordinates.driveIndices[j]})).sort((a,b)=>a.omega-b.omega);
 assert.deepEqual(ordered[2].index,ordered[0].index.map((value,k)=>value+ordered[1].index[k]),'cross-F beat must use the original two laser phase axes');
 assert.equal(coordinates.frameIndices.length,32);assert.equal(coordinates.frameSectors.length,32);
 for(const [i,state]of model.states.entries()){
  assert.equal(coordinates.frameIndices[i].length,2);assert.ok(coordinates.frameIndices[i].every(Number.isInteger));
  assert.equal(coordinates.frameSectors[i],state.manifold==='ground'?0:1);
  if(state.manifold!=='ground')assert.deepEqual(coordinates.frameIndices[i],[0,0]);
 }
 const groundOne=model.states.findIndex(state=>state.manifold==='ground'&&state.F===1);
 assert.ok(coordinates.frameIndices[groundOne].some(value=>value!==0));
});

test('integer phase coordinates reproduce individual laser Hamiltonians and both rotating frames',()=>{
 const scene=threeNearBeams(),near=scene.lasers.filter(laser=>laser.enabled&&laser.regime==='near');
 const common=buildFullMasterModel(scene,undefined,'common'),resonant=buildFullMasterModel(scene);
 const individual=near.map(laser=>buildFullMasterModel({...scene,lasers:[laser,...scene.lasers.filter(beam=>beam.regime==='far')]}));
 for(const time of [0,0.000013,0.137,20]){
  const commonH=hamiltonianAt(common,time),resonantH=hamiltonianAt(resonant,time);
  for(let e=8;e<32;e++)for(let g=0;g<8;g++){
   const k=e*32+g;let expectedReal=0,expectedImag=0,scale=1;
   for(let j=0;j<near.length;j++){
    const phase=-2*Math.PI*laserFrequencyDifferenceMHz(near[j],near[0])*time;
    const re=individual[j].hReal[k],im=individual[j].hImag[k];scale+=Math.hypot(re,im);
    expectedReal+=re*Math.cos(phase)-im*Math.sin(phase);expectedImag+=re*Math.sin(phase)+im*Math.cos(phase);
   }
   assert.ok(Math.abs(commonH.real[k]-expectedReal)<1e-8*scale);assert.ok(Math.abs(commonH.imag[k]-expectedImag)<1e-8*scale);
   const frameDifference=resonant.frequencyCoordinates.frameIndices[e].map((integer,j)=>integer-resonant.frequencyCoordinates.frameIndices[g][j]);
   const phase=accurateSum(frameDifference.map((integer,j)=>integer*resonant.frequencyCoordinates.frequencies[j]*time));
   const re=commonH.real[k]*Math.cos(phase)-commonH.imag[k]*Math.sin(phase),im=commonH.real[k]*Math.sin(phase)+commonH.imag[k]*Math.cos(phase);
   assert.ok(Math.abs(resonantH.real[k]-re)<1e-8*scale);assert.ok(Math.abs(resonantH.imag[k]-im)<1e-8*scale);
  }
 }
 assert.deepEqual(resonant.collapse,common.collapse);
});

test('a true tiny D1 pair beat survives a remote D2 reference without a phase cutoff',()=>{
 const scene=defaultScene(),probe=scene.lasers[0];
 const first={...probe,id:'D1-a',line:'D1',referenceTransition:{line:'D1',Fg:1,Fe:2},detuningMHz:0};normalizeLaserFrequency(first);
 const second={...first,id:'D1-b',detuningMHz:1e-10,phaseDeg:27};normalizeLaserFrequency(second);
 scene.lasers.push(first,second);
 const physicalBeat=laserFrequencyDifferenceMHz(second,first);assert.ok(physicalBeat>0&&physicalBeat<1e-9);
 // The old reference-offset subtraction loses this beat altogether.
 assert.equal(laserFrequencyDifferenceMHz(first,probe),laserFrequencyDifferenceMHz(second,probe));
 const model=buildFullMasterModel(scene),coordinates=model.frequencyCoordinates;
 assert.equal(coordinates.frequencies.length,2);
 const tinyAxis=coordinates.frequencies.indexOf(2*Math.PI*physicalBeat);assert.notEqual(tinyAxis,-1);
 assert.ok(model.drives.some((drive,j)=>drive.omega===2*Math.PI*physicalBeat&&coordinates.driveIndices[j][tinyAxis]===1&&coordinates.driveIndices[j].every((value,k)=>k===tinyAxis||value===0)));
});

test('exact co-carriers share phase coordinates while their optical phases remain in the Hamiltonian',()=>{
 const scene=defaultScene(),first=scene.lasers[0];
 const second={...first,id:'same-frequency',phaseDeg:90};scene.lasers.push(second);
 const model=buildFullMasterModel(scene),a=buildFullMasterModel({...scene,lasers:[first,...scene.lasers.filter(l=>l.regime==='far')]}),b=buildFullMasterModel({...scene,lasers:[second,...scene.lasers.filter(l=>l.regime==='far')]});
 assert.deepEqual(model.frequencyCoordinates.frequencies,[]);assert.deepEqual(model.frequencyCoordinates.driveIndices,[]);assert.equal(model.drives.length,0);
 for(let e=8;e<32;e++)for(let g=0;g<8;g++){const k=e*32+g;assert.equal(model.hReal[k],accurateSum([a.hReal[k],b.hReal[k]]));assert.equal(model.hImag[k],accurateSum([a.hImag[k],b.hImag[k]]));}
});

test('authoritative MHz tree coordinates retain exact 100 and 300 MHz carrier relations before TAU conversion',()=>{
 const scene=defaultScene(),probe=scene.lasers[0];
 scene.lasers.push({...probe,id:'plus100',detuningMHz:100},{...probe,id:'plus400',detuningMHz:400});
 const model=buildFullMasterModel(scene),coordinates=model.frequencyCoordinates;
 assert.deepEqual(coordinates.frequenciesMHz,[100,300]);
 assert.deepEqual(coordinates.frequencies,[2*Math.PI*100,2*Math.PI*300]);
 const ordered=model.drives.map((drive,i)=>({omega:drive.omega,index:coordinates.driveIndices[i]})).sort((a,b)=>a.omega-b.omega);
 assert.deepEqual(ordered.map(drive=>drive.index),[[1,0],[1,1]]);
 assert.equal(ordered[0].omega,2*Math.PI*100);assert.equal(ordered[1].omega,accurateSum(coordinates.frequencies));
});

test('each near beam retains every allowed dipole in both ground F and both fine manifolds',()=>{
 for(const intensity of [1,2.2104853207207683e8]){
  const s=defaultScene();s.fields[0].thetaDeg=23;s.lasers[0].ellipticityDeg=17;s.lasers[0].intensityMWcm2=intensity;
  const model=buildFullMasterModel(s),a=spectroscopyAnalysis(s),seen=new Set();assert.equal(model.drives.length,0);
  for(let g=0;g<8;g++)for(let e=8;e<32;e++){
   const expected=spectroscopyCoupling(a,s.lasers[0],model.states[g].id,model.states[e].id).rabiMHz;
   const actual=Math.hypot(model.hReal[e*32+g],model.hImag[e*32+g])/Math.PI;
   assert.ok(Math.abs(actual-expected)<1e-9*Math.max(1,expected),`${model.states[g].id} → ${model.states[e].id}`);
   if(actual>1e-8)seen.add(`${model.states[g].F}:${model.states[e].manifold}`);
  }assert.equal(seen.size,4);
 }
});

test('resonant and common frames have identical physical driving at multiple phases',()=>{
 const s=defaultScene(),repump={...s.lasers[0],id:'repump',referenceTransition:{line:'D2',Fg:1,Fe:2},phaseDeg:19};normalizeLaserFrequency(repump);s.lasers.push(repump);
 const common=buildFullMasterModel(s,undefined,'common'),resonant=buildFullMasterModel(s);assert.equal(common.drives.length,1);assert.equal(resonant.drives.length,1);
 const at=(model,t)=>({real:model.hReal.map((v,i)=>v+model.drives.reduce((sum,d)=>sum+d.cos.real[i]*Math.cos(d.omega*t)+d.sin.real[i]*Math.sin(d.omega*t),0)),imag:model.hImag.map((v,i)=>v+model.drives.reduce((sum,d)=>sum+d.cos.imag[i]*Math.cos(d.omega*t)+d.sin.imag[i]*Math.sin(d.omega*t),0))});
 for(const t of [0,.000013,.17]){const a=at(common,t),b=at(resonant,t);for(let i=0;i<32;i++)for(let j=0;j<32;j++){
  const deltaI=(resonant.frameFrequenciesMHz[i]-common.frameFrequenciesMHz[i])*2*Math.PI,deltaJ=(resonant.frameFrequenciesMHz[j]-common.frameFrequenciesMHz[j])*2*Math.PI,phase=(deltaI-deltaJ)*t,k=i*32+j;
  const re=a.real[k]*Math.cos(phase)-a.imag[k]*Math.sin(phase)-(i===j?deltaI:0),im=a.real[k]*Math.sin(phase)+a.imag[k]*Math.cos(phase);
  assert.ok(Math.abs(re-b.real[k])<1e-7*Math.max(1,Math.abs(re)));assert.ok(Math.abs(im-b.imag[k])<1e-7*Math.max(1,Math.abs(im)));
 }}
});

test('the regime decides the physics: a far laser is only a static shift and is part of the dynamics key',()=>{
 const s=defaultScene(),model=buildFullMasterModel(s);
 assert.equal(model.laserCount,1);assert.ok(model.drives.every(d=>!d.laserIds.includes('tweezer')));
 // The far tweezer moves H0 (and its basis), nothing else.
 const dark=buildFullMasterModel({...s,lasers:s.lasers.filter(l=>l.id!=='tweezer')});
 assert.equal(dark.laserCount,1);assert.notDeepEqual(dark.hReal.filter((_,k)=>k%33===0),model.hReal.filter((_,k)=>k%33===0));
 const display=spectroscopyAnalysis(s,undefined,false);
 assert.deepEqual(model.states.map(l=>l.energyMHz),[...display.ground,...display.D1,...display.D2].map(l=>l.energyMHz),'the levels chart shows the H0 the dynamics uses');
 const key=scene=>dynamicsCacheKey(scene,scene.dynamics,false),flipped=structuredClone(s);flipped.lasers[1].regime='near';
 assert.notEqual(key(flipped),key(s));
});

test('a far laser near resonance and a near laser far from resonance are refused, as on main',()=>{
 const far=defaultScene();far.lasers[0].regime='far';
 const c=classifyLasers(far).find(x=>x.id==='probe');assert.equal(c.valid,false);assert.match(c.message,/远失谐条件/);
 assert.throws(()=>buildFullMasterModel(far),/远失谐条件/);
 const near=defaultScene();near.lasers[1].regime='near';
 assert.equal(classifyLasers(near).find(x=>x.id==='tweezer').valid,false);assert.throws(()=>buildFullMasterModel(near),/近共振范围/);
 const off=defaultScene();off.lasers[0].regime='far';off.lasers[0].enabled=false;assert.ok(classifyLasers(off).every(x=>x.valid));
});

test('a far laser is refused as on main: the detuning rule and the 1% static mixing |Ω/2Δ|, both in farValidity',()=>{
 const s=defaultScene(),tweezer=s.lasers.find(l=>l.id==='tweezer'),base=farValidity(s,'tweezer'),intensity=laserIntensityMWcm2(tweezer);
 assert.equal(STATIC_MIXING_LIMIT,.01);assert.equal(base.mixingLimit,.01);assert.ok(base.valid&&base.detuningRule&&base.withinGap&&base.mixing>0&&base.mixing<.01&&base.message==='');
 assert.equal(base.mixing,displayMixing(s,'tweezer').ratio);
 const at=ratio=>{const scene=structuredClone(s),t=scene.lasers.find(l=>l.id==='tweezer');t.intensityMode='intensity';t.intensityMWcm2=intensity*(ratio/base.mixing)**2;return scene;};
 // Main's detuning rule |Δ| ≥ 100 Ω/2π binds first for this tweezer, near |Ω/2Δ| = 0.53%.
 const ok=at(.005),rule=at(.006),above=at(.012);
 assert.ok(farValidity(ok,'tweezer').valid);assert.equal(buildFullMasterModel(ok).laserCount,1);
 const r=farValidity(rule,'tweezer');assert.ok(!r.valid&&!r.detuningRule&&r.mixing<.01);
 assert.throws(()=>buildFullMasterModel(rule),error=>error.message===r.message&&/远失谐条件/.test(r.message));
 // Above 1% the static shift itself is refused too: the levels chart has no static shift for it.
 const a=farValidity(above,'tweezer');assert.ok(!a.valid&&Math.abs(a.mixing-.012)<1e-9);
 assert.equal(spectroscopyAnalysis({...above,lasers:above.lasers.filter(l=>l.id==='tweezer')}).spectroscopy.status,'unresolved');
 assert.throws(()=>buildFullMasterModel(above));
 const probe=farValidity(s,'probe');assert.ok(!probe.valid&&!probe.withinGap&&!probe.detuningRule,'a resonant probe cannot be far');
 assert.deepEqual(farValidity(s,'nobody'),{mixing:0,mixingLimit:.01,withinGap:true,detuningRule:true,valid:true,message:''});
});

test('the far static shift includes the Rb+ core: the mean ground shift is the scalar polarizability sum',()=>{
 // pi-polarized 852 nm tweezer alone. The trace over the 8 ground states removes
 // hyperfine and Zeeman structure to first order: what is left is the scalar
 // shift -alpha |E|^2 / 4 of D1 + D2 (co- and counter-rotating), the listed
 // outside states and the core.
 const s=defaultScene();s.lasers=s.lasers.filter(l=>l.id==='tweezer');Object.assign(s.lasers[0],{thetaDeg:90,phiDeg:0,ellipticityDeg:0,polarizationAngleDeg:0});
 const model=buildFullMasterModel(s),ground=model.states.slice(0,8),mean=accurateSum(ground.map(l=>l.starkShiftMHz))/8;
 const H=6.62607015e-34,hartree=4.3597447222071e-18,eV=1.602176634e-19,ea0=eV*5.29177210903e-11,omega=H*laserFrequencyMHz(s.lasers[0])*1e6/hartree;
 const term=(reduced,de)=>reduced*reduced*de/(de*de-omega*omega)/3;
 // Edmonds reduced elements: sqrt(2J+1) times Steck's <J||er||J'>; RB87.D2.dipole is the stretched-state dipole, 1/sqrt(2) of it.
 const dLines=term(Math.SQRT2*RB87.D1.dipole/ea0,H*RB87.D1.frequencyHz/hartree)+term(2*RB87.D2.dipole/ea0,H*RB87.D2.frequencyHz/hartree);
 const valence=VIRTUAL_MANIFOLDS.ground.reduce((sum,v)=>sum+term(v.reducedEA0,v.energyEV*eV/hartree),0),core=CORE_POLARIZABILITY_AU.ground;
 const E2=2*laserIntensityMWcm2(s.lasers[0])*10/(8.8541878128e-12*299792458),shift=alpha=>-alpha*1.64877727436e-41*E2/(4*H*1e6);
 assert.ok(Math.abs(mean/shift(dLines+valence+core)-1)<1e-6,`mean ground shift ${mean} MHz vs ${shift(dLines+valence+core)} MHz`);
 assert.ok(Math.abs(shift(core)/mean)>1e-3&&Math.abs(shift(valence)/mean)>1e-4,'the test resolves the core and the outside states');
 assert.ok(model.assumptions.some(text=>text.includes('Rb+ 芯极化')));
});

test('emission is secular per D line and ground F: one operator per line, photon polarization and F, total rate Gamma',()=>{
 const model=buildFullMasterModel(defaultScene()),states=model.states;assert.equal(model.collapse.length,12);
 // One D line per operator: the far tweezer mixes the two fine-structure manifolds of the
 // background states (through the ground copies), so the other line carries ~1e-12 of the rate.
 for(const c of model.collapse){const rows=new Set(),weight={D1:0,D2:0};
  for(let g=0;g<8;g++)for(let e=8;e<32;e++)if(c.real[g*32+e]!==0||c.imag[g*32+e]!==0){rows.add(states[g].F);weight[states[e].manifold]+=c.real[g*32+e]**2+c.imag[g*32+e]**2;}
  assert.equal(rows.size,1);assert.ok(Math.min(weight.D1,weight.D2)<1e-9*Math.max(weight.D1,weight.D2));
  for(let i=0;i<32;i++)for(let j=0;j<32;j++)if(!(i<8&&j>=8))assert.equal(c.real[i*32+j],0);
 }
 for(let e=8;e<32;e++){const rate=model.collapse.reduce((sum,c)=>sum+Array.from({length:8},(_,g)=>c.real[g*32+e]**2+c.imag[g*32+e]**2).reduce((a,b)=>a+b,0),0);
  const gamma=states[e].manifold==='D1'?36.129:38.117;assert.ok(Math.abs(rate/gamma-1)<1e-3,`${states[e].id} ${rate}`);}
});

test('initial-state specs build a diagonal unit-trace density in the model basis',()=>{
 const s=defaultScene(),states=buildFullMasterModel(s).states,diag=m=>states.map((_,i)=>m.initialReal[i*32+i]);
 const check=(m)=>{assert.ok(Math.abs(diag(m).reduce((a,b)=>a+b,0)-1)<1e-14);assert.ok(m.initialImag.every(v=>v===0));for(let i=0;i<32;i++)for(let j=0;j<32;j++)if(i!==j)assert.equal(m.initialReal[i*32+j],0);assert.equal(m.initialIndex,diag(m).indexOf(Math.max(...diag(m))));};
 const one=buildFullMasterModel(s);check(one);assert.equal(diag(one)[states.findIndex(x=>x.id===s.dynamics.initialStateId)],1);
 assert.equal(one.initialIndex,states.findIndex(x=>x.id===s.dynamics.initialStateId));
 const all=buildFullMasterModel(s,{...s.dynamics,initialState:{kind:'unpolarized',manifold:'ground'}});check(all);
 states.forEach((x,i)=>assert.equal(diag(all)[i],x.manifold==='ground'?1/8:0));
 const f1=buildFullMasterModel(s,{...s.dynamics,initialState:{kind:'unpolarized',manifold:'F1'}});check(f1);
 states.forEach((x,i)=>assert.ok(Math.abs(diag(f1)[i]-(x.manifold==='ground'&&x.F===1?1/3:0))<1e-15));
 const hot=buildFullMasterModel(s,{...s.dynamics,initialState:{kind:'thermal',temperatureK:1e-3}});check(hot);
 const g20=states.findIndex(x=>x.id==='ground-2-0'),g10=states.findIndex(x=>x.id==='ground-1-0');
 const ratio=diag(hot)[g20]/diag(hot)[g10],expected=Math.exp(-(states[g20].energyMHz-states[g10].energyMHz)*6.62607015e-28/1.380649e-23/1e-3);
 assert.ok(Math.abs(ratio/expected-1)<1e-9);assert.ok(diag(hot)[g20]<diag(hot)[g10]);
 const flat=buildFullMasterModel(s,{...s.dynamics,initialState:{kind:'thermal',temperatureK:1e9}});states.forEach((x,i)=>assert.ok(Math.abs(diag(flat)[i]-(x.manifold==='ground'?1/8:0))<1e-6));
 const mixed=buildFullMasterModel(s,{...s.dynamics,initialState:{kind:'weights',weights:{'ground-2-0':3,'D2-3-0':1}}});check(mixed);
 assert.equal(diag(mixed)[g20],0.75);assert.equal(mixed.initialPopulations[states.findIndex(x=>x.id==='D2-3-0')],0.25);
 assert.throws(()=>buildFullMasterModel(s,{...s.dynamics,initialState:{kind:'weights',weights:{'ground-2-0':0}}}));
 assert.ok(mixed.assumptions.some(text=>text.startsWith('初态：')));
});

test('many ground references stay in the ground region instead of being packed into excited manifolds',()=>{
 const s=defaultScene();s.references=Array.from({length:12},(_,i)=>({id:`ref-${i}`,name:`reference ${i}`,anchorId:'ground-1-0',offsetMHz:i,anchorMode:'follow'}));
 const svg=levelsChart(s,spectroscopyAnalysis(s),undefined,{line:'D2',selectedIds:[],initialId:s.dynamics.initialStateId});
 const ys=s.references.map(r=>Number(svg.split(`data-state="${r.id}"`)[1].match(/<line x1="911" y1="([^"]+)"/)[1]));
 assert.ok(ys.every(y=>y>=535));for(let i=1;i<ys.length;i++)assert.ok(Math.abs(ys[i]-ys[i-1])>=51.9);
});
