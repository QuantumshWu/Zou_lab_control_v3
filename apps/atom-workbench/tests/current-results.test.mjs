import test from 'node:test';
import assert from 'node:assert/strict';
import {CurrentResults} from '../dist/js/atom-workbench/src/current-results.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {meanInputsKey,trajectoryInputsKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';

test('current display survives pure UI edits, but changing physics, duration or points erases all historical results',()=>{
  for(const change of [s=>s.fields[0].magnitudeG+=1,s=>s.dynamics.durationUs+=1,s=>s.dynamics.samples+=1]){
    const scene=defaultScene(),store=new CurrentResults(),key=()=>trajectoryInputsKey(scene,scene.dynamics),original=key(),epoch=store.activate(original),record={populations:[1]};
    store.set(original,epoch,'trajectory',record);scene.name='Renamed';scene.marks={'ground-2-0':'Marked'};scene.dynamics.monitoredStateIds=['ground-1-0'];
    assert.equal(store.activate(key()),epoch);assert.strictEqual(store.get(key(),'trajectory'),record);
    const before=structuredClone(scene);change(scene);store.activate(key());assert.equal(store.get(original,'trajectory'),undefined);
    Object.assign(scene,before);store.activate(key());assert.equal(key(),original);assert.equal(store.get(original,'trajectory'),undefined);
    assert.equal(store.set(original,epoch,'trajectory',record),false,'late result from the first visit cannot resurrect history');
  }
});

/** The app's mean cache: the means of the current mean inputs, one slot per time T. */
function filled(){
  const scene=defaultScene(),cache=new CurrentResults(),key=()=>meanInputsKey(scene,scene.dynamics),epoch=cache.activate(key());
  const means=Object.fromEntries([1e5,1e6,null].map(T=>[String(T),{horizon:T}]));
  for(const T of [1e5,1e6,null])assert.equal(cache.set(key(),epoch,T,means[String(T)]),true);
  return {scene,cache,key,epoch,means};
}

test('mean cache hit: another T, another duration or other samples find the mean already there',()=>{
  const {scene,cache,key,means}=filled();
  for(const T of [null,1e5,1e6]){scene.dynamics.meanHorizonUs=T;cache.activate(key());assert.strictEqual(cache.get(key(),T),means[String(T)],`T = ${T}`);}
  const trajectory=trajectoryInputsKey(scene,scene.dynamics);
  scene.dynamics.durationUs*=3;scene.dynamics.samples=51;cache.activate(key());
  assert.notEqual(trajectoryInputsKey(scene,scene.dynamics),trajectory,'the trajectory is solved again');
  assert.strictEqual(cache.get(key(),scene.dynamics.meanHorizonUs),means[String(scene.dynamics.meanHorizonUs)],'its mean is not');
  scene.name='Renamed';scene.lasers[1].showLightShift=!scene.lasers[1].showLightShift;scene.dynamics.monitorMode='manual';cache.activate(key());
  assert.strictEqual(cache.get(key(),1e6),means['1000000'],'display settings leave the mean');
});

test('mean cache miss: a T no result carried yet, or other inputs',()=>{
  const scene=defaultScene(),cache=new CurrentResults(),key=meanInputsKey(scene,scene.dynamics),epoch=cache.activate(key);
  assert.equal(cache.get(key,1e6),undefined,'nothing solved yet');
  cache.set(key,epoch,1e6,{horizon:1e6});
  assert.equal(cache.get(key,null),undefined,'only the T a result carried');
  assert.equal(cache.get('other inputs',1e6),undefined);
});

test('mean cache invalidation: exactly a change of the physics or of the initial state drops every T',()=>{
  for(const [label,change] of [
    ['field',s=>{s.fields[0].magnitudeG+=1;}],['laser intensity',s=>{s.lasers[0].intensityMWcm2*=2;}],['laser off',s=>{s.lasers[1].enabled=false;}],
    ['initial state',s=>{s.dynamics.initialStateId='ground-1-0';}],
  ]){
    const {scene,cache,key,epoch}=filled(),original=key(),before=structuredClone(scene);
    change(scene);assert.notEqual(key(),original,label);cache.activate(key());
    for(const T of [1e5,1e6,null])assert.equal(cache.get(key(),T),undefined,`${label}: T = ${T}`);
    // Back to the first inputs: nothing comes back, and a late result of the first visit is refused.
    Object.assign(scene,before);assert.equal(key(),original);cache.activate(key());
    assert.equal(cache.get(original,1e6),undefined,label);assert.equal(cache.set(original,epoch,1e6,{horizon:1e6}),false,label);
  }
});
