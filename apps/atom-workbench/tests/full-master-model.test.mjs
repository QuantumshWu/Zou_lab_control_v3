import test from 'node:test';import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {spectroscopyAnalysis,spectroscopyCoupling} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {normalizeLaserFrequency} from '../dist/js/atom-workbench/src/laser-frequency.js';
import {levelsChart} from '../dist/js/atom-workbench/src/charts.js';
import {accurateSum,laserFrequencyDifferenceMHz} from '../dist/js/atom-workbench/src/physics.js';

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

test('many ground references stay in the ground region instead of being packed into excited manifolds',()=>{
 const s=defaultScene();s.references=Array.from({length:12},(_,i)=>({id:`ref-${i}`,name:`reference ${i}`,anchorId:'ground-1-0',offsetMHz:i,anchorMode:'follow'}));
 const svg=levelsChart(s,spectroscopyAnalysis(s),undefined,{line:'D2',selectedIds:[],initialId:s.dynamics.initialStateId});
 const ys=s.references.map(r=>Number(svg.split(`data-state="${r.id}"`)[1].match(/<line x1="911" y1="([^"]+)"/)[1]));
 assert.ok(ys.every(y=>y>=535));for(let i=1;i<ys.length;i++)assert.ok(Math.abs(ys[i]-ys[i-1])>=51.9);
});
