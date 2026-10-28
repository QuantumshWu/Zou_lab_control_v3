import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze} from '../dist/js/atom-workbench/src/physics.js';
import {startupScene,validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {laserReferenceStateId} from '../dist/js/atom-workbench/src/transition-view.js';

const initial=defaultScene(),analysis=analyze(initial),ids=[...analysis.ground,...analysis.D1,...analysis.D2].map(state=>state.id);
const roundtrip=scene=>validateScene(JSON.parse(JSON.stringify(scene)));

test('scene files preserve selected monitor order, more than eight states, all 32 and the explicitly empty set',()=>{
  for(const count of [0,9,16,32]){
    const scene=defaultScene();scene.dynamics.monitoredStateIds=ids.slice().reverse().slice(0,count);scene.dynamics.initialStateId='D2-3-3';
    const loaded=roundtrip(scene),reloaded=roundtrip(loaded);
    assert.deepEqual(loaded.dynamics.monitoredStateIds,scene.dynamics.monitoredStateIds);assert.deepEqual(reloaded.dynamics.monitoredStateIds,scene.dynamics.monitoredStateIds);
    assert.equal(reloaded.dynamics.initialStateId,'D2-3-3');
  }
});

test('monitor imports reject duplicate IDs, nonphysical references and oversized lists instead of silently dropping states',()=>{
  for(const invalid of [[ids[0],ids[0]],[...ids,'reference-level'],['reference-level'],[...ids,ids[0]]]){
    const scene=defaultScene();scene.dynamics.monitoredStateIds=invalid;assert.throws(()=>roundtrip(scene),/dynamics\.monitoredStateIds/);
  }
});

test('scene duration and sample imports enforce the same explicit UI bounds without rounding or truncation',()=>{
  for(const duration of [.001,.02,.2,2000,100000,5000000]){const scene=defaultScene();scene.dynamics.durationUs=duration;assert.equal(roundtrip(scene).dynamics.durationUs,duration);}
  for(const duration of [0,-1,NaN]){const scene=defaultScene();scene.dynamics.durationUs=duration;assert.throws(()=>roundtrip(scene),/dynamics\.durationUs/);}
  const fractional=defaultScene();fractional.dynamics.samples=161.5;assert.throws(()=>roundtrip(fractional),/dynamics\.samples 应为整数/);
});

// One scene schema (types.ts SCENE_VERSION): a file is the current schema exactly, or refused with the reason.
test('a file in an older format is refused cleanly, naming what does not match',()=>{
  const file=()=>JSON.parse(JSON.stringify(defaultScene()));
  // A version-1 file: an F/m initial state, a laser given by its detuning, a field by its angles.
  const old={version:1,name:'old',initialF:2,initialM:2,fields:[{id:'b',name:'b',enabled:true,magnitudeG:3,thetaDeg:0,phiDeg:0}],
    lasers:[{id:'probe',name:'probe',enabled:true,line:'D2',detuningMHz:0,intensityMWcm2:1,referenceTransition:{line:'D2',Fg:2,Fe:2},thetaDeg:0,phiDeg:0,polarizationAngleDeg:0,ellipticityDeg:0}]};
  assert.throws(()=>validateScene(old),/^Error: 场景文件不符合当前格式（version 2）：version 为 1，应为 2$/);
  for(const [label,change,reason] of [
    ['no dynamics (an F/m-only file)',f=>{delete f.dynamics;},/场景 缺少 dynamics/],
    ['an initial F and m',f=>{f.initialF=2;f.initialM=0;},/场景 含有未知字段 initialF/],
    ['a quality setting',f=>{f.dynamics.quality='verified';},/dynamics 含有未知字段 quality/],
    ['a population tolerance',f=>{f.dynamics.populationTolerance=.005;},/dynamics 含有未知字段 populationTolerance/],
    ['a dynamics laser',f=>{f.dynamics.laserId='probe';},/dynamics 含有未知字段 laserId/],
    ['no time T',f=>{delete f.dynamics.meanHorizonUs;},/dynamics 缺少 meanHorizonUs/],
    ['no duration unit',f=>{delete f.dynamics.durationUnit;},/dynamics 缺少 durationUnit/],
    ['a laser regime',f=>{f.lasers[0].regime='far';},/lasers\[0\] 含有未知字段 regime/],
    ['a legacy detuning',f=>{f.lasers[0].detuningMHz=0;},/lasers\[0\] 含有未知字段 detuningMHz/],
    ['no 显示光移 switch',f=>{delete f.lasers[0].showLightShift;},/lasers\[0\] 缺少 showLightShift/],
    ['no intensity mode',f=>{delete f.lasers[0].intensityMode;},/lasers\[0\] 缺少 intensityMode/],
    ['no power unit',f=>{delete f.lasers[1].powerUnit;},/lasers\[1\] 缺少 powerUnit/],
    ['no phase',f=>{delete f.lasers[0].phaseDeg;},/lasers\[0\] 缺少 phaseDeg/],
    ['a field given by its angles',f=>{f.fields[0]={...f.fields[0],thetaDeg:0,phiDeg:0};delete f.fields[0].direction;},/fields\[0\]/],
    ['a laser with a preset label',f=>{f.lasers[0].polarizationPreset='sigma-plus';},/含有未知字段 polarizationPreset/],
    ['a wavelength written in THz',f=>{f.lasers[1].frequencyUnit='THz';},/lasers\[1\]\.frequencyUnit/],
    ['a reference level without its offset unit',f=>{f.references=[{id:'r',name:'r',anchorId:'ground-2-0',anchorMode:'follow',offsetMHz:0}];},/references\[0\] 缺少 offsetUnit/],
    ['a thermal state without its unit',f=>{f.dynamics.initialState={kind:'thermal',temperatureK:1e-3};},/热分布初态缺少温度单位/],
    ['a temperature unit without a thermal state',f=>{f.dynamics.temperatureUnit='mK';},/只用于热分布初态/],
  ]){const f=file();change(f);assert.throws(()=>validateScene(f),reason,label);}
  // Display preferences are genuinely optional; derived anchors may be absent (they are resolved).
  const bare=file();delete bare.display;for(const l of bare.lasers){delete l.frequencyAnchor;delete l.intensityAnchor;}assert.equal(validateScene(bare).lasers.length,2);
});

test('a fresh start always shows the default scene; scene data an earlier version kept is removed',()=>{
  const prefix='rb87-workbench-demo-v4',edited=roundtrip(defaultScene());edited.name='Edited';edited.fields[0].magnitudeG=7;
  const storage=entries=>{const items=new Map(entries);return {items,removeItem:key=>items.delete(key)};};
  for(const saved of [JSON.stringify(edited),'{"version":1,"name":"old"}','not json']){
    const s=storage([[prefix,saved],[`${prefix}-unreadable`,saved],[`${prefix}-baseline`,'{}'],[`${prefix}-state-labels`,'{"first":"ground-2-0"}'],[`${prefix}-solver-failures`,'[]']]);
    assert.deepEqual(startupScene(s,prefix),defaultScene(),saved);
    assert.deepEqual([...s.items.keys()],[`${prefix}-solver-failures`],'only the solver-failure log stays: it is not scene data');
  }
  assert.deepEqual(startupScene(storage([]),prefix),defaultScene(),'nothing stored');
  assert.deepEqual(startupScene(undefined,prefix),defaultScene(),'no storage at all');
  assert.deepEqual(startupScene({removeItem(){throw new Error('denied');}},prefix),defaultScene(),'storage that refuses');
});

test('independent per-laser reference states roundtrip without changing rho0, Hamiltonian or physical cache identity',()=>{
  const s=roundtrip(defaultScene()),initial=s.dynamics.initialStateId,before=buildFullMasterModel(s),key=dynamicsCacheKey(s,s.dynamics,true);
  for(const id of ids){s.lasers[0].referenceStateId=id;const loaded=roundtrip(s);assert.equal(loaded.lasers[0].referenceStateId,id);assert.equal(laserReferenceStateId(loaded,loaded.lasers[0]),id);assert.equal(laserReferenceStateId(loaded,loaded.lasers[1]),initial);assert.equal(loaded.dynamics.initialStateId,initial);assert.equal(dynamicsCacheKey(loaded,loaded.dynamics,true),key);const after=buildFullMasterModel(loaded);assert.deepEqual(after.hReal,before.hReal);assert.deepEqual(after.hImag,before.hImag);assert.deepEqual(after.drives,before.drives);assert.equal(after.initialIndex,before.initialIndex);}
  delete s.lasers[0].referenceStateId;assert.equal(laserReferenceStateId(roundtrip(s),s.lasers[0]),initial);
  for(const invalid of ['reference-1','',42]){s.lasers[0].referenceStateId=invalid;assert.throws(()=>roundtrip(s));}
});

test('mixed initial states round-trip, a single state is initialStateId alone, and the request identity includes the spec',()=>{
  const base=roundtrip(defaultScene()),key=s=>dynamicsCacheKey(s,s.dynamics,false);
  const asState=structuredClone(base);asState.dynamics.initialState={kind:'state',id:'ground-1-1'};assert.throws(()=>roundtrip(asState),/dynamics\.initialState/);
  const keys=new Set([key(base)]);
  for(const spec of [{kind:'unpolarized',manifold:'ground'},{kind:'unpolarized',manifold:'F1'},{kind:'unpolarized',manifold:'F2'},{kind:'thermal',temperatureK:1e-5},{kind:'thermal',temperatureK:300},{kind:'weights',weights:{'ground-2-0':1,'D2-3-3':0.5}}]){
    const s=structuredClone(base);s.dynamics.initialState=spec;if(spec.kind==='thermal')s.dynamics.temperatureUnit='K';const loaded=roundtrip(s);
    assert.deepEqual(loaded.dynamics.initialState,spec);assert.equal(loaded.dynamics.initialStateId,base.dynamics.initialStateId);
    keys.add(key(loaded));
    const model=buildFullMasterModel(loaded,loaded.dynamics);assert.deepEqual(model.initialState,spec);
  }
  assert.equal(keys.size,7,'every spec is a distinct request');
  const a=structuredClone(base),b=structuredClone(base);a.dynamics.initialState={kind:'weights',weights:{'ground-2-0':1,'ground-1-0':2}};b.dynamics.initialState={kind:'weights',weights:{'ground-1-0':2,'ground-2-0':1}};
  assert.equal(key(a),key(b),'weight order is not part of the identity');
  for(const bad of [{kind:'state',id:'ground-3-0'},{kind:'unpolarized',manifold:'F3'},{kind:'thermal',temperatureK:0},{kind:'thermal',temperatureK:-1},{kind:'weights',weights:{}},{kind:'weights',weights:{'ground-2-0':0}},{kind:'weights',weights:{'nope':1}},{kind:'weights',weights:{'ground-2-0':-1}},{kind:'mystery'}]){
    const s=structuredClone(base);s.dynamics.initialState=bad;if(bad.kind==='thermal')s.dynamics.temperatureUnit='K';assert.throws(()=>roundtrip(s),/dynamics\.initialState/,JSON.stringify(bad));
  }
});

test('显示光移 is saved with the scene and is no part of the dynamics request identity',()=>{
  const s=roundtrip(defaultScene()),key=dynamicsCacheKey(s,s.dynamics,false);
  assert.deepEqual(s.lasers.map(l=>l.showLightShift),[false,true]);
  s.lasers[1].showLightShift=false;assert.equal(dynamicsCacheKey(s,s.dynamics,false),key);
  assert.equal(roundtrip(s).lasers[1].showLightShift,false,'the switch survives a save, also for a laser called tweezer');
  s.lasers[1].showLightShift='yes';assert.throws(()=>roundtrip(s),/showLightShift/);
});

