import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {composeMagneticField,polarization,setPolarizationPreset,direction,analyze} from '../dist/js/atom-workbench/src/physics.js';
import {shortcutDirection} from '../dist/js/atom-workbench/src/direction.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {spectroscopyAnalysis} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {along,spherical} from './directions.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<2e-13,`${a} != ${b}`);
const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
const roundtrip=s=>validateScene(JSON.parse(JSON.stringify(s)));

test('a transverse plane wave along or against its linked field never has pi, for any ellipse',()=>{
 for(const theta of [0,37.123,90,127,180])for(const mode of ['parallel','anti'])for(const chi of [-45,-17,0,21,45]){
  const s=defaultScene();s.fields[0].direction=along(theta,31);const l=s.lasers[0];
  l.direction=shortcutDirection(s.fields[0],mode);l.ellipticityDeg=chi;l.polarizationAngleDeg=19;
  const p=polarization(l,s.fields);near(p.pi,0);near(p.minus+p.pi+p.plus,1);near(dot(p.k,p.u),0);near(dot(p.k,p.v),0);
  if(chi===0){near(p.minus,.5);near(p.plus,.5);}
  if(Math.abs(chi)===45)near(chi*(mode==='parallel'?1:-1)>0?p.plus:p.minus,1);
 }
});

test('atomic presets link the light to the field: exactly pure, kept on import, following the field\'s reference axis',()=>{
 for(const theta of [0,13,37.123456789,89.999999,90,91,127,180])for(const preset of ['pi','sigma-minus','sigma-plus']){
  let s=defaultScene();s.fields[0].direction=along(theta,17.234);setPolarizationPreset(s.lasers[0],s.fields,preset);
  assert.deepEqual(s.lasers[0].direction.reference,{kind:'field',id:'bias'});
  s=roundtrip(s);const chosen=preset==='pi'?1:preset==='sigma-minus'?0:2;
  const pure=()=>{const p=polarization(s.lasers[0],s.fields);[p.minus,p.pi,p.plus].forEach((v,i)=>i===chosen?near(v,1):assert.equal(v,0));};
  pure();
  // A new reference axis of the field carries the linked light with it: still pure.
  s.fields[0].direction.reference.thetaDeg=73;pure();
 }
 // Without a field the presets read on +z.
 for(const preset of ['pi','sigma-minus','sigma-plus']){const s=defaultScene();s.fields=[];setPolarizationPreset(s.lasers[0],s.fields,preset);const p=polarization(s.lasers[0],s.fields);near(preset==='pi'?p.pi:preset==='sigma-minus'?p.minus:p.plus,1);}
});

test('shape presets retain k and the main axis; perpendicular linear orientations select pi or a sigma mixture',()=>{
 const s=defaultScene(),l=s.lasers[0];l.polarizationAngleDeg=27;
 for(const shape of ['linear','circular-plus','circular-minus']){const before=structuredClone(l.direction);setPolarizationPreset(l,s.fields,shape);assert.deepEqual(l.direction,before);assert.equal(l.polarizationAngleDeg,27);}
 l.direction=spherical(90,0);setPolarizationPreset(l,s.fields,'linear');
 for(const [psi,pi] of [[0,1],[45,.5],[90,0]]){l.polarizationAngleDeg=psi;const p=polarization(l,s.fields);near(p.pi,pi);near(p.minus,(1-pi)/2);near(p.plus,(1-pi)/2);}
});

test('a linked light keeps its place when the field moves by its own offset, and its own edits stay independent',()=>{
 const s=defaultScene(),l=s.lasers[0];s.fields[0].direction=along(37,17);
 setPolarizationPreset(l,s.fields,'sigma-plus');
 const geometry=()=>structuredClone({direction:l.direction,psi:l.polarizationAngleDeg,chi:l.ellipticityDeg});
 const clicked=geometry(),k=polarization(l,s.fields).k;
 // The field turns away by its own offset: the light, linked to its reference axis, stays.
 Object.assign(s.fields[0].direction,{thetaDeg:24,phiDeg:-23});analyze(s);spectroscopyAnalysis(s,'probe');buildFullMasterModel(s);
 assert.deepEqual(geometry(),clicked);assert.deepEqual(polarization(l,s.fields).k,k);
 let p=polarization(l,s.fields);assert.ok(p.plus<.99);assert.ok(p.pi>0);
 l.direction.thetaDeg=41;l.polarizationAngleDeg=21;l.ellipticityDeg=9;const edited=geometry();
 analyze(s);spectroscopyAnalysis(s,'probe');buildFullMasterModel(s);assert.deepEqual(geometry(),edited);
 p=polarization(l,s.fields);near(dot(p.k,p.u),0);near(dot(p.k,p.v),0);
 const reloaded=roundtrip(s);assert.deepEqual({direction:reloaded.lasers[0].direction,psi:reloaded.lasers[0].polarizationAngleDeg,chi:reloaded.lasers[0].ellipticityDeg},edited);
});

test('presets are commands, not saved state: a file with a preset or follow label is refused',()=>{
 for(const [key,value] of [['polarizationPreset','pi'],['polarizationMode','relative-to-B'],['followMagneticField',true]]){
  const raw=JSON.parse(JSON.stringify(defaultScene()));raw.lasers[0][key]=value;
  assert.throws(()=>validateScene(raw),error=>error.message.endsWith(`lasers[0] 含有未知字段 ${key}`));
 }
});

test('oblique weights match direct lab-vector projection independently of spherical-coordinate formulas',()=>{
 for(const [bt,bp,kt,kp,psi,chi] of [[32,17,67,-23,19,21],[120,-70,91,31,-34,-45],[9,110,150,-9,83,0]]){
  const s=defaultScene();s.fields[0].direction=along(bt,bp);Object.assign(s.lasers[0],{direction:spherical(kt,kp),polarizationAngleDeg:psi,ellipticityDeg:chi});
  const p=polarization(s.lasers[0],s.fields),b=direction(bt,bp),c=Math.cos(chi*Math.PI/180),d=Math.sin(chi*Math.PI/180);
  near(p.pi,(c*dot(p.u,b))**2+(d*dot(p.v,b))**2);near(p.minus+p.pi+p.plus,1);near(dot(p.k,p.u),0);near(dot(p.k,p.v),0);
 }
});

test('true pi has exactly zero nonzero-delta-m drive in default background and in the effective Hamiltonian',()=>{
 const s=defaultScene();setPolarizationPreset(s.lasers[0],s.fields,'pi');
 for(const trap of [true,false]){s.lasers[1].enabled=trap;const a=spectroscopyAnalysis(s,'probe');for(const t of a.transitions.filter(t=>t.laserId==='probe'&&t.me!==t.mg))assert.equal(t.rabiMHz,0);
  const model=buildFullMasterModel(s);model.states.forEach((g,i)=>model.states.forEach((e,j)=>{if(g.manifold==='ground'&&e.manifold!=='ground'&&g.m!==e.m){assert.equal(model.hReal[i*32+j],0);assert.equal(model.hImag[i*32+j],0);}}));
 }
});

test('mixed-m background labels do not misidentify the actual pi driving component',()=>{
 const s=defaultScene();setPolarizationPreset(s.lasers[0],s.fields,'pi');s.lasers[1].polarizationAngleDeg=90;
 const a=spectroscopyAnalysis(s,'probe'),mixed=a.transitions.filter(t=>t.laserId==='probe'&&t.me!==t.mg&&t.rabiMHz>1e-8);
 assert.ok(mixed.length>0);for(const t of mixed){assert.equal(t.q,0);assert.equal(t.mixedPolarization,false);assert.equal(t.mLabelApproximate,true);}
});

test('the composed field and its quantization axis are what every analysis reads',()=>{
 const s=defaultScene(),f=composeMagneticField(s.fields),a=analyze(s);
 assert.deepEqual(a.fieldVector,f.fieldVector);assert.deepEqual(a.fieldAxis,f.fieldAxis);
 assert.deepEqual(a.polarizations.probe.amplitudes,polarization(s.lasers[0],s.fields).amplitudes);
});
