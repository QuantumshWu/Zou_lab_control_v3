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

test('a display-only regime change keeps the dynamics request identity',()=>{
  const s=roundtrip(defaultScene()),key=dynamicsCacheKey(s,s.dynamics,false);
  s.lasers[1].regime='near';assert.equal(dynamicsCacheKey(s,s.dynamics,false),key);
});
