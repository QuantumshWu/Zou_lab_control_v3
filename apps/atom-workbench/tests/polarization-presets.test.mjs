import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {composeMagneticField,polarization,laserDirectionAxis,setLaserDirectionAxis,relativeDirectionAxis,setPolarizationPreset,direction} from '../dist/js/atom-workbench/src/physics.js';
import {spectroscopyAnalysis} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {buildEffectiveMasterModel} from '../dist/js/atom-workbench/src/effective-master-model.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
const near=(a,b)=>assert.ok(Math.abs(a-b)<2e-13,`${a} != ${b}`);
const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);

test('a transverse plane wave parallel or antiparallel to B never has pi, for any ellipse',()=>{
 for(const theta of [0,37.123,90,127,180])for(const mode of ['parallel','anti'])for(const chi of [-45,-17,0,21,45]){
  const s=defaultScene();Object.assign(s.fields[0],{thetaDeg:theta,phiDeg:31});const f=composeMagneticField(s.fields),l=s.lasers[0];
  setLaserDirectionAxis(l,relativeDirectionAxis(f.fieldAxis,mode),true);l.ellipticityDeg=chi;l.polarizationAngleDeg=19;
  const p=polarization(l,f.fieldVector,f.fieldAxis);near(p.pi,0);near(p.minus+p.plus,1);near(dot(p.k,p.u),0);near(dot(p.k,p.v),0);
  if(chi===0){near(p.minus,.5);near(p.plus,.5);}
  if(Math.abs(chi)===45)near(chi*(mode==='parallel'?1:-1)>0?p.plus:p.minus,1);
 }
});

test('atomic presets configure exact pure components, survive import, and do not follow later B edits',()=>{
 for(const theta of [0,13,37.123456789,89.999999,90,91,127,180])for(const preset of ['pi','sigma-minus','sigma-plus']){
  let s=defaultScene();Object.assign(s.fields[0],{thetaDeg:theta,phiDeg:17.234});setPolarizationPreset(s.lasers[0],s.fields,preset);
  s=validateScene(JSON.parse(JSON.stringify(s)));const f=composeMagneticField(s.fields),p=polarization(s.lasers[0],f.fieldVector,f.fieldAxis);
  const weights=[p.minus,p.pi,p.plus],chosen=preset==='pi'?1:preset==='sigma-minus'?0:2;
  weights.forEach((v,i)=>i===chosen?near(v,1):assert.equal(v,0));
  const saved=laserDirectionAxis(s.lasers[0]);s.fields[0].thetaDeg=73;assert.deepEqual(laserDirectionAxis(s.lasers[0]),saved);
 }
 for(const preset of ['pi','sigma-minus','sigma-plus']){const s=defaultScene();s.fields[0].enabled=false;setPolarizationPreset(s.lasers[0],s.fields,preset);const p=polarization(s.lasers[0],[0,0,0]);near(preset==='pi'?p.pi:preset==='sigma-minus'?p.minus:p.plus,1);}
});

test('shape presets retain k and main axis; perpendicular linear orientations select pi or sigma mixture',()=>{
 const s=defaultScene(),l=s.lasers[0],f=composeMagneticField(s.fields);l.polarizationAngleDeg=27;
 for(const shape of ['linear','circular-plus','circular-minus']){const before=laserDirectionAxis(l);setPolarizationPreset(l,s.fields,shape);assert.deepEqual(laserDirectionAxis(l),before);assert.equal(l.polarizationAngleDeg,27);}
 setLaserDirectionAxis(l,{thetaDeg:90,phiDeg:0});setPolarizationPreset(l,s.fields,'linear');
 for(const [psi,pi] of [[0,1],[45,.5],[90,0]]){l.polarizationAngleDeg=psi;const p=polarization(l,f.fieldVector,f.fieldAxis);near(p.pi,pi);near(p.minus,(1-pi)/2);near(p.plus,(1-pi)/2);}
});

test('oblique weights match direct lab-vector projection independently of spherical-coordinate formulas',()=>{
 for(const [bt,bp,kt,kp,psi,chi] of [[32,17,67,-23,19,21],[120,-70,91,31,-34,-45],[9,110,150,-9,83,0]]){
  const s=defaultScene();Object.assign(s.fields[0],{thetaDeg:bt,phiDeg:bp});Object.assign(s.lasers[0],{thetaDeg:kt,phiDeg:kp,polarizationAngleDeg:psi,ellipticityDeg:chi});
  const f=composeMagneticField(s.fields),p=polarization(s.lasers[0],f.fieldVector,f.fieldAxis),b=direction(bt,bp),c=Math.cos(chi*Math.PI/180),d=Math.sin(chi*Math.PI/180);
  near(p.pi,(c*dot(p.u,b))**2+(d*dot(p.v,b))**2);near(p.minus+p.pi+p.plus,1);near(dot(p.k,p.u),0);near(dot(p.k,p.v),0);
 }
});

test('true pi has exactly zero nonzero-delta-m drive in default background and in the effective Hamiltonian',()=>{
 const s=defaultScene();setPolarizationPreset(s.lasers[0],s.fields,'pi');
 for(const trap of [true,false]){s.lasers[1].enabled=trap;const a=spectroscopyAnalysis(s,'probe');for(const t of a.transitions.filter(t=>t.laserId==='probe'&&t.me!==t.mg))assert.equal(t.rabiMHz,0);
  const model=buildEffectiveMasterModel(s);model.states.forEach((g,i)=>model.states.forEach((e,j)=>{if(g.manifold==='ground'&&e.manifold!=='ground'&&g.m!==e.m){assert.equal(model.hReal[i*32+j],0);assert.equal(model.hImag[i*32+j],0);}}));
 }
});

test('mixed-m background labels do not misidentify the actual pi driving component',()=>{
 const s=defaultScene();setPolarizationPreset(s.lasers[0],s.fields,'pi');s.lasers[1].polarizationAngleDeg=90;
 const a=spectroscopyAnalysis(s,'probe'),mixed=a.transitions.filter(t=>t.laserId==='probe'&&t.me!==t.mg&&t.rabiMHz>1e-8);
 assert.ok(mixed.length>0);for(const t of mixed){assert.equal(t.q,0);assert.equal(t.mixedPolarization,false);assert.equal(t.mLabelApproximate,true);}
});
