import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze,laserFrequencyMHz,resolveSceneFrequencies} from '../dist/js/atom-workbench/src/physics.js';
import {referenceEnergyMHz} from '../dist/js/atom-workbench/src/reference-energy.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {followers,releaseFollowers,setFrequencyBase,setFrequencyValue} from '../dist/js/atom-workbench/src/laser-frequency.js';

const roundtrip=scene=>validateScene(JSON.parse(JSON.stringify(scene)));
const hz=scene=>Object.fromEntries(scene.lasers.map(l=>[l.id,laserFrequencyMHz(l)]));

test('every beam has one frequency expression, kept bit for bit by the scene file',()=>{
  const scene=defaultScene();resolveSceneFrequencies(scene);const before=hz(scene);
  const [probe,tweezer]=scene.lasers;
  assert.deepEqual(probe.frequency,{base:{kind:'transition',line:'D2',Fg:2,Fe:2},offsetMHz:0});
  assert.deepEqual(tweezer.frequency,{wavelengthNm:852});
  assert.deepEqual(hz(roundtrip(scene)),before,'export and import keep every bit');
});

test('a laser can follow another laser or a reference frequency (an offset lock)',()=>{
  const scene=roundtrip(defaultScene()),[probe]=scene.lasers;
  scene.referenceFrequencies=[{id:'lock',name:'锁频点',frequency:{base:{kind:'transition',line:'D2',Fg:2,Fe:3},offsetMHz:-12},frequencyUnit:'MHz'}];
  const repump={...structuredClone(probe),id:'repump',name:'回泵光'};scene.lasers.push(repump);
  setFrequencyBase(scene,repump,{kind:'laser',id:'probe'});setFrequencyValue(scene,repump,6.568,'GHz');
  const offset=laserFrequencyMHz(repump)-laserFrequencyMHz(probe);assert.ok(Math.abs(offset-6568)<1e-6);
  setFrequencyValue(scene,probe,10);assert.ok(Math.abs(laserFrequencyMHz(repump)-laserFrequencyMHz(probe)-6568)<1e-6,'it follows the probe');
  assert.deepEqual(followers(scene,'laser:probe'),['laser:repump']);
  // Neither itself nor its follower can be a base: no cycle.
  assert.throws(()=>setFrequencyBase(scene,probe,{kind:'laser',id:'repump'}),/跟随/);
  assert.throws(()=>setFrequencyBase(scene,probe,{kind:'laser',id:'probe'}),/跟随/);
  const before=laserFrequencyMHz(probe);setFrequencyBase(scene,probe,{kind:'reference',id:'lock'});
  assert.equal(laserFrequencyMHz(probe),before,'re-expressed, not moved');
  assert.deepEqual(probe.frequency.base,{kind:'reference',id:'lock'});
  const back=roundtrip(scene);assert.deepEqual(hz(back),hz(scene));
  // A saved cycle or a missing base is refused.
  const cyclic=JSON.parse(JSON.stringify(scene));cyclic.referenceFrequencies[0].frequency={base:{kind:'laser',id:'repump'},offsetMHz:0};
  assert.throws(()=>validateScene(cyclic),/成环/);
  const missing=JSON.parse(JSON.stringify(scene));missing.lasers[1].frequency={base:{kind:'laser',id:'gone'},offsetMHz:0};missing.lasers[1].frequencyUnit='MHz';
  assert.throws(()=>validateScene(missing),/不存在/);
});

test('switching base never re-rounds; removing a base releases its followers unchanged',()=>{
  const scene=roundtrip(defaultScene()),[probe]=scene.lasers,repump={...structuredClone(probe),id:'repump',name:'回泵光'};scene.lasers.push(repump);
  setFrequencyValue(scene,probe,-17.3);setFrequencyBase(scene,repump,{kind:'laser',id:'probe'});setFrequencyValue(scene,repump,6568.1234567);
  const p=laserFrequencyMHz(probe),r=laserFrequencyMHz(repump);
  for(let i=0;i<5;i++){setFrequencyBase(scene,probe,{kind:'absolute'});setFrequencyBase(scene,probe,{kind:'transition',line:'D1',Fg:1,Fe:2});}
  assert.equal(laserFrequencyMHz(probe),p);assert.equal(laserFrequencyMHz(repump),r);
  // A wavelength the operator wrote, re-expressed against a transition and back, keeps its frequency.
  setFrequencyBase(scene,probe,{kind:'absolute'});setFrequencyValue(scene,probe,780.241,'nm');const w=laserFrequencyMHz(probe),rw=laserFrequencyMHz(repump);
  for(let i=0;i<5;i++){setFrequencyBase(scene,probe,{kind:'transition',line:'D2',Fg:2,Fe:3});setFrequencyBase(scene,probe,{kind:'absolute'});}
  setFrequencyBase(scene,probe,{kind:'transition',line:'D1',Fg:1,Fe:2});assert.equal(laserFrequencyMHz(probe),w);assert.equal(laserFrequencyMHz(repump),rw);
  assert.deepEqual(releaseFollowers(scene,probe),['回泵光']);scene.lasers=scene.lasers.filter(l=>l!==probe);resolveSceneFrequencies(scene);
  assert.equal(laserFrequencyMHz(repump),rw);assert.equal(repump.frequency.base.kind,'transition','its base is the deleted beam\'s own zero-field base');
});


test('a reference level linked to a photon follows it, and survives the scene file',()=>{
  const scene=roundtrip(defaultScene()),[probe]=scene.lasers,a=analyze(scene),levels=[...a.ground,...a.D1,...a.D2];
  scene.references=[{id:'reference-1',name:'虚拟中间态',anchorId:'ground-2-0',anchorMode:'follow',offsetMHz:-5,offsetUnit:'MHz',photon:{source:{kind:'laser',id:'probe'},direction:'absorb'}}];
  const anchor=levels.find(l=>l.id==='ground-2-0').energyMHz,energy=()=>referenceEnergyMHz(scene.references[0],levels,scene);
  assert.ok(Math.abs(energy()-(anchor+laserFrequencyMHz(probe)-5))<1e-6);
  setFrequencyValue(scene,probe,100);assert.ok(Math.abs(energy()-(anchor+laserFrequencyMHz(probe)-5))<1e-6,'it moves with the probe');
  scene.references[0].photon.direction='emit';assert.ok(Math.abs(energy()-(anchor-laserFrequencyMHz(probe)-5))<1e-6);
  assert.equal(referenceEnergyMHz(scene.references[0],levels),undefined,'a linked energy needs the scene');
  const back=roundtrip(scene);assert.deepEqual(back.references[0].photon,scene.references[0].photon);
  const broken=JSON.parse(JSON.stringify(scene));broken.references[0].photon.source.id='gone';
  assert.throws(()=>validateScene(broken),/不存在/);
});

test('how a frequency is read is saved with the scene, as a display preference only',()=>{
  const scene=roundtrip(defaultScene());scene.display={readouts:{pair:{unit:'GHz',base:'laser:probe'},'laser-frequency':{base:'absolute'}}};
  assert.deepEqual(roundtrip(scene).display,scene.display);
  assert.throws(()=>validateScene({...JSON.parse(JSON.stringify(scene)),display:{readouts:{pair:{unit:'parsec'}}}}),/display.readouts.pair.unit/);
});
