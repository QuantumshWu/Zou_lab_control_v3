import test from 'node:test';import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {spectroscopyAnalysis,spectroscopyCoupling,staticModelBackground} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {classifyLasers} from '../dist/js/atom-workbench/src/laser-regime.js';
import {normalizeLaserFrequency} from '../dist/js/atom-workbench/src/laser-frequency.js';
import {levelsChart} from '../dist/js/atom-workbench/src/charts.js';
import {accurateSum,analyze,laserFrequencyDifferenceMHz,laserFrequencyMHz,laserIntensityMWcm2} from '../dist/js/atom-workbench/src/physics.js';
import {VIRTUAL_MANIFOLDS} from '../dist/js/atom-workbench/src/spectroscopy-data.js';

const N=32,TAU=2*Math.PI;
function threeNearBeams(){
 const scene=defaultScene(),probe=scene.lasers[0];
 probe.referenceTransition={line:'D2',Fg:2,Fe:3};probe.ellipticityDeg=17;probe.thetaDeg=7;normalizeLaserFrequency(probe);
 const repump={...probe,id:'repump',referenceTransition:{line:'D2',Fg:1,Fe:2},phaseDeg:19};normalizeLaserFrequency(repump);
 const third={...probe,id:'third',referenceTransition:{line:'D2',Fg:2,Fe:2},phaseDeg:-31};normalizeLaserFrequency(third);
 scene.lasers.push(repump,third);return scene;
}
/** Coefficient of e^{+i omega t} of drive j: (Hcos - i Hsin)/2, entry k. */
function plus(drive,k){return {re:(drive.cos.real[k]+drive.sin.imag[k])/2,im:(drive.cos.imag[k]-drive.sin.real[k])/2};}
const carrierIndex=model=>model.frequencyCoordinates.driveIndices.map(index=>index.at(-1));

test('every enabled laser is a co- and counter-rotating drive in exact integer coordinates with the reference carrier as an axis',()=>{
 const scene=threeNearBeams(),model=buildFullMasterModel(scene),c=model.frequencyCoordinates;
 assert.equal(model.laserCount,4,'probe, tweezer, repump, third: the far tweezer is a drive too');
 assert.equal(c.frequenciesMHz.length,4);assert.equal(c.frequenciesMHz.at(-1),laserFrequencyMHz(scene.lasers.find(l=>l.id===model.referenceLaserId)));
 assert.equal(model.carrierMHz,c.frequenciesMHz.at(-1));assert.ok(model.carrierMHz>3.8e8);
 assert.deepEqual(c.frequencies,c.frequenciesMHz.map(f=>TAU*f));
 for(const [j,index] of c.driveIndices.entries()){
  assert.ok(index.every(Number.isInteger)&&index.some(v=>v!==0));
  assert.equal(model.drives[j].omega,accurateSum(index.map((n,k)=>n*c.frequencies[k])));assert.ok(model.drives[j].omega>0);
 }
 // Counter-rotating terms: carrier index 2 (nu_l + nu_ref), one per laser.
 assert.equal(carrierIndex(model).filter(n=>n===2).length,4);assert.ok(carrierIndex(model).every(n=>n===0||n===2));
 for(const [i,s] of model.states.entries()){assert.equal(c.frameSectors[i],s.manifold==='ground'?0:1);assert.deepEqual(c.frameIndices[i],[0,0,0,0]);}
 assert.equal(model.frameFrequenciesMHz.find((_,i)=>model.states[i].manifold!=='ground'),model.carrierMHz);
 // The most resonant beam defines the frame; the 852 nm tweezer does not.
 assert.notEqual(model.referenceLaserId,'tweezer');
});

test('integer phase coordinates reproduce the full laboratory dipole coupling of each beam',()=>{
 const scene=threeNearBeams();scene.lasers=scene.lasers.filter(l=>l.id!=='tweezer');
 const model=buildFullMasterModel(scene),reference=scene.lasers.find(l=>l.id===model.referenceLaserId),c=model.frequencyCoordinates;
 // A beam alone is its own reference: its co-rotating coupling is static and its counter-rotating one sits at 2 nu_l.
 const single=scene.lasers.map(l=>buildFullMasterModel({...scene,lasers:[l]}));
 // Single-beam bases differ by the (tiny) static shifts of the other beams: compare on the global coupling scale.
 const scale=Math.max(...single.flatMap(m=>m.hReal.map((v,k)=>Math.hypot(v,m.hImag[k]))));
 for(const t of [0,1.3e-7,2.9e-6]){
  for(let e=8;e<N;e++)for(let g=0;g<8;g++){
   const k=e*N+g;let re=model.hReal[k],im=model.hImag[k],expectedRe=0,expectedIm=0;
   for(const [j,d] of model.drives.entries()){const phase=accurateSum(c.driveIndices[j].map((n,a)=>n*c.frequencies[a]*t)),cs=Math.cos(phase),sn=Math.sin(phase);re+=d.cos.real[k]*cs+d.sin.real[k]*sn;im+=d.cos.imag[k]*cs+d.sin.imag[k]*sn;}
   for(const [l,laser] of scene.lasers.entries()){
    const m=single[l],co={re:m.hReal[k],im:m.hImag[k]},counter=plus(m.drives[0],k),beat=TAU*laserFrequencyDifferenceMHz(laser,reference)*t,sum=TAU*(laserFrequencyMHz(laser)+laserFrequencyMHz(reference))*t;
    expectedRe+=co.re*Math.cos(beat)+co.im*Math.sin(beat)+counter.re*Math.cos(sum)-counter.im*Math.sin(sum);
    expectedIm+=co.im*Math.cos(beat)-co.re*Math.sin(beat)+counter.re*Math.sin(sum)+counter.im*Math.cos(sum);
   }
   assert.ok(Math.abs(re-expectedRe)<1e-7*scale&&Math.abs(im-expectedIm)<1e-7*scale,`${model.states[g].id} -> ${model.states[e].id} at ${t}`);
  }
 }
});

test('the counter-rotating amplitude drives the opposite helicity with the conjugate optical phase',()=>{
 const build=phaseDeg=>{const scene=defaultScene();scene.lasers=scene.lasers.slice(0,1);Object.assign(scene.lasers[0],{thetaDeg:0,phiDeg:0,ellipticityDeg:45,phaseDeg});return buildFullMasterModel(scene);};
 const model=build(37),zero=build(0),states=model.states;assert.equal(model.drives.length,1);assert.deepEqual(model.frequencyCoordinates.driveIndices,[[2]]);
 const coDelta=new Set(),counterDelta=new Set(),phi=37*Math.PI/180;
 for(let e=8;e<N;e++)for(let g=0;g<8;g++){const k=e*N+g,co={re:model.hReal[k],im:model.hImag[k]},cr=plus(model.drives[0],k);
  if(Math.hypot(co.re,co.im)>1e-9){coDelta.add(states[e].m-states[g].m);const ref={re:zero.hReal[k],im:zero.hImag[k]};
   assert.ok(Math.hypot(co.re-(ref.re*Math.cos(phi)-ref.im*Math.sin(phi)),co.im-(ref.re*Math.sin(phi)+ref.im*Math.cos(phi)))<1e-9*Math.hypot(co.re,co.im));}
  if(Math.hypot(cr.re,cr.im)>1e-9){counterDelta.add(states[e].m-states[g].m);const ref=plus(zero.drives[0],k);
   assert.ok(Math.hypot(cr.re-(ref.re*Math.cos(phi)+ref.im*Math.sin(phi)),cr.im-(ref.im*Math.cos(phi)-ref.re*Math.sin(phi)))<1e-9*Math.hypot(cr.re,cr.im));}
 }
 assert.deepEqual([...coDelta],[1]);assert.deepEqual([...counterDelta],[-1]);
});

test('a true tiny D1 pair beat survives a remote D2 reference without a phase cutoff',()=>{
 const scene=defaultScene(),probe=scene.lasers[0];
 const first={...probe,id:'D1-a',line:'D1',referenceTransition:{line:'D1',Fg:1,Fe:2},detuningMHz:0};normalizeLaserFrequency(first);
 const second={...first,id:'D1-b',detuningMHz:1e-10,phaseDeg:27};normalizeLaserFrequency(second);
 scene.lasers.push(first,second);
 const physicalBeat=laserFrequencyDifferenceMHz(second,first);assert.ok(physicalBeat>0&&physicalBeat<1e-9);
 const model=buildFullMasterModel(scene),coordinates=model.frequencyCoordinates;
 const tinyAxis=coordinates.frequencies.indexOf(2*Math.PI*physicalBeat);assert.notEqual(tinyAxis,-1);
 assert.ok(model.drives.some((drive,j)=>drive.omega===2*Math.PI*physicalBeat&&coordinates.driveIndices[j][tinyAxis]===1&&coordinates.driveIndices[j].every((value,k)=>k===tinyAxis||value===0)));
});

test('exact co-carriers share phase coordinates and cancel exactly in opposite phase',()=>{
 const scene=defaultScene(),first=scene.lasers[0];scene.lasers=[first,{...first,id:'same-frequency',phaseDeg:90}];
 const model=buildFullMasterModel(scene),a=buildFullMasterModel({...scene,lasers:[first]}),b=buildFullMasterModel({...scene,lasers:[scene.lasers[1]]});
 assert.equal(model.frequencyCoordinates.frequencies.length,1);assert.deepEqual(model.frequencyCoordinates.driveIndices,[[2]]);
 const scale=Math.max(...a.hReal.map((v,k)=>Math.hypot(v,a.hImag[k])));
 for(let e=8;e<N;e++)for(let g=0;g<8;g++){const k=e*N+g;assert.ok(Math.abs(model.hReal[k]-accurateSum([a.hReal[k],b.hReal[k]]))<1e-9*scale);assert.ok(Math.abs(model.hImag[k]-accurateSum([a.hImag[k],b.hImag[k]]))<1e-9*scale);}
 const cancel=buildFullMasterModel({...scene,lasers:[first,{...first,id:'opposite',phaseDeg:180}]});
 assert.equal(cancel.drives.length,0,'co- and counter-rotating parts both cancel exactly');
 for(let e=8;e<N;e++)for(let g=0;g<8;g++){assert.equal(cancel.hReal[e*N+g],0);assert.equal(cancel.hImag[e*N+g],0);}
});

test('authoritative MHz tree coordinates retain exact 100 and 300 MHz carrier relations before TAU conversion',()=>{
 const scene=defaultScene(),probe=scene.lasers[0];scene.lasers=[probe,{...probe,id:'plus100',detuningMHz:100},{...probe,id:'plus400',detuningMHz:400}];
 const model=buildFullMasterModel(scene),coordinates=model.frequencyCoordinates;
 assert.deepEqual(coordinates.frequenciesMHz,[100,300,laserFrequencyMHz(probe)]);
 assert.deepEqual(coordinates.frequencies,coordinates.frequenciesMHz.map(f=>2*Math.PI*f));
 const co=model.drives.map((drive,i)=>({omega:drive.omega,index:coordinates.driveIndices[i]})).filter(d=>d.index[2]===0).sort((a,b)=>a.omega-b.omega);
 assert.deepEqual(co.map(drive=>drive.index),[[1,0,0],[1,1,0]]);assert.equal(co[0].omega,2*Math.PI*100);
 assert.deepEqual(model.frequencyCoordinates.driveIndices.filter(i=>i[2]!==0).sort(),[[0,0,2],[1,0,2],[1,1,2]]);
});

test('the reference beam retains every allowed dipole in both ground F and both fine manifolds',()=>{
 for(const intensity of [1,2.2104853207207683e8]){
  const s=defaultScene();s.lasers=s.lasers.slice(0,1);s.fields[0].thetaDeg=23;s.lasers[0].ellipticityDeg=17;s.lasers[0].intensityMWcm2=intensity;
  // Couplings in the model basis (the static outside-model shifts of this beam mix m slightly).
  const model=buildFullMasterModel(s),bg=staticModelBackground(s),a={...bg.analysis,ground:bg.levels.slice(0,8),D1:bg.levels.slice(8,16),D2:bg.levels.slice(16)},seen=new Set();assert.equal(model.drives.length,1);
  for(let g=0;g<8;g++)for(let e=8;e<32;e++){
   const expected=spectroscopyCoupling(a,s.lasers[0],model.states[g].id,model.states[e].id).rabiMHz;
   const actual=Math.hypot(model.hReal[e*32+g],model.hImag[e*32+g])/Math.PI;
   assert.ok(Math.abs(actual-expected)<1e-9*Math.max(1,expected),`${model.states[g].id} → ${model.states[e].id}`);
   if(actual>1e-8)seen.add(`${model.states[g].F}:${model.states[e].manifold}`);
  }assert.equal(seen.size,4);
 }
});

test('no laser regime is invalid: a far-labelled resonant beam is an ordinary explicit drive',()=>{
 const s=defaultScene();s.lasers[0].regime='far';s.lasers[1].regime='near';
 assert.ok(classifyLasers(s).every(c=>c.valid&&c.message===''));
 const model=buildFullMasterModel(s);assert.equal(model.laserCount,2);assert.equal(model.referenceLaserId,'probe');
 const relabelled=buildFullMasterModel({...s,lasers:s.lasers.map(l=>({...l,regime:l.regime==='far'?'near':'far'}))});
 assert.deepEqual(relabelled.hReal,model.hReal);assert.deepEqual(relabelled.drives,model.drives);
});

test('emission: one operator per photon polarization with both D lines and both ground-F channels, static in the frame',()=>{
 const s=defaultScene(),model=buildFullMasterModel(s),states=model.states;assert.equal(model.collapse.length,3);
 for(const c of model.collapse){const rows=new Set(),lines=new Set();
  for(let g=0;g<8;g++)for(let e=8;e<N;e++)if(c.real[g*N+e]!==0||c.imag[g*N+e]!==0){rows.add(states[g].F);lines.add(states[e].manifold);}
  assert.deepEqual([...rows].sort(),[1,2]);assert.deepEqual([...lines].sort(),['D1','D2']);
  for(let i=0;i<N;i++)for(let j=0;j<N;j++)if(!(i<8&&j>=8))assert.equal(c.real[i*N+j],0);
 }
 // Total decay rate of every excited state equals its line width.
 for(let e=8;e<N;e++){const rate=model.collapse.reduce((sum,c)=>sum+Array.from({length:8},(_,g)=>c.real[g*N+e]**2+c.imag[g*N+e]**2).reduce((a,b)=>a+b,0),0);
  const gamma=(states[e].manifold==='D1'?36.129:38.117);assert.ok(Math.abs(rate/gamma-1)<1e-3,`${states[e].id} ${rate}`);}
});

test('far-detuned spin-flip Raman scattering keeps the D1/D2 interference of the emission amplitudes',()=>{
 // 852 nm tweezer, pi-polarized, out of ground-2-0: adiabatic excited amplitude
 // a_e = H_eg / (E_g - E_e) (static co-rotating coupling in its own frame).
 const s=defaultScene();s.lasers=s.lasers.slice(1);Object.assign(s.lasers[0],{thetaDeg:90,phiDeg:0,ellipticityDeg:0,polarizationAngleDeg:0});
 const m=buildFullMasterModel(s),g=m.states.findIndex(x=>x.id==='ground-2-0'),E=i=>m.hReal[i*N+i],a=[];
 for(let e=0;e<N;e++)a.push(e<8?{re:0,im:0}:{re:m.hReal[e*N+g]/(E(g)-E(e)),im:m.hImag[e*N+g]/(E(g)-E(e))});
 const rates=ops=>Array.from({length:8},(_,f)=>ops.reduce((sum,c)=>{let re=0,im=0;for(let e=8;e<N;e++){const cr=c.real[f*N+e],ci=c.imag[f*N+e];re+=cr*a[e].re-ci*a[e].im;im+=cr*a[e].im+ci*a[e].re;}return sum+re*re+im*im;},0)*1e6);
 const combined=rates(m.collapse),total=combined.reduce((x,y)=>x+y,0),raman=total-combined[g];
 // Separate D1 and D2 operators (a secular approximation across 7 THz) lose the cancellation.
 const split=[];for(const c of m.collapse)for(const line of ['D1','D2'])split.push({real:c.real.map((v,k)=>m.states[k%N].manifold===line?v:0),imag:c.imag.map((v,k)=>m.states[k%N].manifold===line?v:0)});
 const separate=rates(split),ramanSeparate=separate.reduce((x,y)=>x+y,0)-separate[g];
 assert.ok(total>15&&total<30,`total scattering ${total}/s`);
 assert.ok(raman>0.2&&raman<0.3,`spin-flip Raman ${raman}/s`);assert.ok(ramanSeparate/raman>30,`separate lines ${ramanSeparate}/s`);
});

test('the static H0 keeps only states outside the model and the Rb+ core; the D-line light shift is dynamical',()=>{
 const s=defaultScene();s.lasers=s.lasers.slice(1);s.lasers[0].regime='far';
 const bare=analyze(s),model=staticModelBackground(s),display=spectroscopyAnalysis(s);
 const id='ground-2-0',e0=bare.ground.find(l=>l.id===id).energyMHz,staticShift=model.levels.find(l=>l.id===id).energyMHz-e0,displayShift=display.ground.find(l=>l.id===id).energyMHz-e0;
 // Independent scalar sum: core (9.1 - 0.26 au) plus the listed 6-9P states, co- and counter-rotating.
 const hartree=4.3597447222071e-18,omega=6.62607015e-34*laserFrequencyMHz(s.lasers[0])*1e6/hartree;
 const valence=VIRTUAL_MANIFOLDS.ground.reduce((sum,v)=>{const de=v.energyEV*1.602176634e-19/hartree;return sum+v.reducedEA0**2*de/(de*de-omega*omega)/3;},0);
 const E2=2*laserIntensityMWcm2(s.lasers[0])*10/(8.8541878128e-12*299792458),expected=-(9.1-0.26+valence)*1.64877727436e-41*E2/(4*6.62607015e-34*1e6);
 assert.ok(displayShift< -10,'the display includes the D-line light shift');
 assert.ok(valence>1&&valence<3);assert.ok(Math.abs(staticShift/expected-1)<1e-3,`static ${staticShift} vs ${expected}`);
 const master=buildFullMasterModel(s),k=master.states.findIndex(l=>l.id===id);assert.equal(master.states[k].energyMHz,model.levels.find(l=>l.id===id).energyMHz);
 assert.ok(master.assumptions.some(text=>text.includes('Rb+ 芯极化')));
});

test('initial-state specs build a diagonal unit-trace density in the model basis',()=>{
 const s=defaultScene(),states=buildFullMasterModel(s).states,diag=m=>states.map((_,i)=>m.initialReal[i*N+i]);
 const check=(m)=>{assert.ok(Math.abs(diag(m).reduce((a,b)=>a+b,0)-1)<1e-14);assert.ok(m.initialImag.every(v=>v===0));for(let i=0;i<N;i++)for(let j=0;j<N;j++)if(i!==j)assert.equal(m.initialReal[i*N+j],0);assert.equal(m.initialIndex,diag(m).indexOf(Math.max(...diag(m))));};
 const one=buildFullMasterModel(s);check(one);assert.equal(diag(one)[states.findIndex(x=>x.id===s.dynamics.initialStateId)],1);
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
