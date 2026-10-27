import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze} from '../dist/js/atom-workbench/src/physics.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
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
    assert.equal(reloaded.dynamics.initialStateId,'D2-3-3');assert.equal(Object.hasOwn(reloaded.dynamics,'quality'),false);
  }
});

test('monitor imports reject duplicate IDs, nonphysical references and oversized lists instead of silently dropping states',()=>{
  for(const invalid of [[ids[0],ids[0]],[...ids,'reference-level'],['reference-level'],[...ids,ids[0]]]){
    const scene=defaultScene();scene.dynamics.monitoredStateIds=invalid;assert.throws(()=>roundtrip(scene),/动力学态选择无效/);
  }
});

test('scene duration and sample imports enforce the same explicit UI bounds without rounding or truncation',()=>{
  for(const duration of [.001,.02,.2,2000,100000,5000000]){const scene=defaultScene();scene.dynamics.durationUs=duration;assert.equal(roundtrip(scene).dynamics.durationUs,duration);}
  for(const duration of [0,-1,NaN]){const scene=defaultScene();scene.dynamics.durationUs=duration;assert.throws(()=>roundtrip(scene),/durationUs 超出范围/);}
  const fractional=defaultScene();fractional.dynamics.samples=161.5;assert.throws(()=>roundtrip(fractional),/samples 必须为整数/);
});

test('legacy F/m-only files set and monitor their actual initial state; old quality remains inert metadata',()=>{
  const legacy=defaultScene();delete legacy.dynamics;legacy.initialF=2;legacy.initialM=2;
  const loaded=roundtrip(legacy);assert.equal(loaded.dynamics.initialStateId,'ground-2-2');assert.equal(loaded.dynamics.monitoredStateIds[0],'ground-2-2');
  assert.equal(Object.hasOwn(loaded.dynamics,'quality'),false);
  const historical=defaultScene();historical.dynamics.quality='verified';assert.equal(roundtrip(historical).dynamics.quality,'verified');
});

test('independent per-laser reference states roundtrip without changing rho0, Hamiltonian or physical cache identity',()=>{
  const s=roundtrip(defaultScene()),initial=s.dynamics.initialStateId,before=buildFullMasterModel(s),key=dynamicsCacheKey(s,s.dynamics,true);
  for(const id of ids){s.lasers[0].referenceStateId=id;const loaded=roundtrip(s);assert.equal(loaded.lasers[0].referenceStateId,id);assert.equal(laserReferenceStateId(loaded,loaded.lasers[0]),id);assert.equal(laserReferenceStateId(loaded,loaded.lasers[1]),initial);assert.equal(loaded.dynamics.initialStateId,initial);assert.equal(dynamicsCacheKey(loaded,loaded.dynamics,true),key);const after=buildFullMasterModel(loaded);assert.deepEqual(after.hReal,before.hReal);assert.deepEqual(after.hImag,before.hImag);assert.deepEqual(after.drives,before.drives);assert.equal(after.initialIndex,before.initialIndex);}
  delete s.lasers[0].referenceStateId;assert.equal(laserReferenceStateId(roundtrip(s),s.lasers[0]),initial);
  for(const invalid of ['reference-1','',42]){s.lasers[0].referenceStateId=invalid;assert.throws(()=>roundtrip(s));}
});

test('initial-state specs round-trip, a state spec folds into initialStateId, and the request identity includes the spec',()=>{
  const base=roundtrip(defaultScene()),key=s=>dynamicsCacheKey(s,s.dynamics,false);
  const asState=structuredClone(base);asState.dynamics.initialState={kind:'state',id:'ground-1-1'};
  const folded=roundtrip(asState);assert.equal(folded.dynamics.initialStateId,'ground-1-1');assert.equal(Object.hasOwn(folded.dynamics,'initialState'),false);
  assert.equal(folded.initialF,1);assert.equal(folded.initialM,1);
  const keys=new Set([key(base)]);
  for(const spec of [{kind:'unpolarized',manifold:'ground'},{kind:'unpolarized',manifold:'F1'},{kind:'unpolarized',manifold:'F2'},{kind:'thermal',temperatureK:1e-5},{kind:'thermal',temperatureK:300},{kind:'weights',weights:{'ground-2-0':1,'D2-3-3':0.5}}]){
    const s=structuredClone(base);s.dynamics.initialState=spec;const loaded=roundtrip(s);
    assert.deepEqual(loaded.dynamics.initialState,spec);assert.equal(loaded.dynamics.initialStateId,base.dynamics.initialStateId);
    keys.add(key(loaded));
    const model=buildFullMasterModel(loaded,loaded.dynamics);assert.deepEqual(model.initialState,spec);
  }
  assert.equal(keys.size,7,'every spec is a distinct request');
  const a=structuredClone(base),b=structuredClone(base);a.dynamics.initialState={kind:'weights',weights:{'ground-2-0':1,'ground-1-0':2}};b.dynamics.initialState={kind:'weights',weights:{'ground-1-0':2,'ground-2-0':1}};
  assert.equal(key(a),key(b),'weight order is not part of the identity');
  for(const bad of [{kind:'state',id:'ground-3-0'},{kind:'unpolarized',manifold:'F3'},{kind:'thermal',temperatureK:0},{kind:'thermal',temperatureK:-1},{kind:'weights',weights:{}},{kind:'weights',weights:{'ground-2-0':0}},{kind:'weights',weights:{'nope':1}},{kind:'weights',weights:{'ground-2-0':-1}},{kind:'mystery'}]){
    const s=structuredClone(base);s.dynamics.initialState=bad;assert.throws(()=>roundtrip(s),/初态/);
  }
});

test('显示光移 is saved with the scene and is no part of the dynamics request identity',()=>{
  const s=roundtrip(defaultScene()),key=dynamicsCacheKey(s,s.dynamics,false);
  assert.deepEqual(s.lasers.map(l=>l.showLightShift),[false,true]);
  s.lasers[1].showLightShift=false;assert.equal(dynamicsCacheKey(s,s.dynamics,false),key);
  assert.equal(roundtrip(s).lasers[1].showLightShift,false,'the switch survives a save, also for a laser called tweezer');
  s.lasers[1].showLightShift='yes';assert.throws(()=>roundtrip(s),/showLightShift/);
});

// A scene saved with the 处理方式 control: every laser is an explicit drive, and a laser that
// was 静态光移 ('far') shows its light shift. Nothing else changes and nothing is said.
test('a scene with regimes loads into the explicit model, its 静态光移 lasers shown',()=>{
  const old=JSON.parse(JSON.stringify(defaultScene()));
  old.lasers.forEach(l=>delete l.showLightShift);old.lasers[0].regime='near';old.lasers[1].regime='far';
  old.lasers.push({...old.lasers[0],id:'second',name:'second',regime:'far'},{...old.lasers[0],id:'third',name:'third'});delete old.lasers[3].regime;
  const loaded=validateScene(old);
  assert.deepEqual(loaded.lasers.map(l=>l.showLightShift),[false,true,true,false]);
  assert.ok(loaded.lasers.every(l=>!('regime' in l)),'no regime is kept');
  // A tweezer saved before either field: shown.
  const first=JSON.parse(JSON.stringify(defaultScene()));first.lasers.forEach(l=>delete l.showLightShift);
  assert.deepEqual(validateScene(first).lasers.map(l=>l.showLightShift),[false,true]);
  // The model is the explicit one whatever the old regime said.
  const explicit=defaultScene(),model=buildFullMasterModel(loaded,loaded.dynamics),reference=buildFullMasterModel({...loaded,lasers:loaded.lasers.map(l=>({...l,showLightShift:false}))},loaded.dynamics);
  assert.equal(model.laserCount,4);assert.deepEqual(model.drives,reference.drives);assert.deepEqual(model.hReal,reference.hReal);
  assert.equal(buildFullMasterModel(explicit).laserCount,2);
});

test('a saved population tolerance, whatever its value, loads silently and is not kept',()=>{
  for(const value of [.001,.005,.02,.05,0,-1,NaN,'0.005',null,{}]){
    const old=JSON.parse(JSON.stringify(defaultScene()));old.dynamics.populationTolerance=value;
    const loaded=validateScene(old);
    assert.ok(!('populationTolerance' in loaded.dynamics),String(value));
    assert.equal(dynamicsCacheKey(loaded,loaded.dynamics,false),dynamicsCacheKey(roundtrip(defaultScene()),roundtrip(defaultScene()).dynamics,false));
  }
});
