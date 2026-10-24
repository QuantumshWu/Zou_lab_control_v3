import test from 'node:test';
import assert from 'node:assert/strict';
import {CurrentDynamicsResults} from '../dist/js/atom-workbench/src/current-dynamics-results.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';

test('current display survives pure UI edits, but changing physics, duration or points erases all historical results',()=>{
  for(const change of [s=>s.fields[0].magnitudeG+=1,s=>s.dynamics.durationUs+=1,s=>s.dynamics.samples+=1]){
    const scene=defaultScene(),store=new CurrentDynamicsResults(),key=()=>dynamicsCacheKey(scene,scene.dynamics,false),original=key(),epoch=store.activate(original),record={populations:[1]};
    store.set(original,epoch,'trajectory',record);scene.name='Renamed';scene.marks={'ground-2-0':'Marked'};scene.dynamics.monitoredStateIds=['ground-1-0'];
    assert.equal(store.activate(key()),epoch);assert.strictEqual(store.get(key(),'trajectory'),record);
    const before=structuredClone(scene);change(scene);store.activate(key());assert.equal(store.get(original,'trajectory'),undefined);
    Object.assign(scene,before);store.activate(key());assert.equal(key(),original);assert.equal(store.get(original,'trajectory'),undefined);
    assert.equal(store.set(original,epoch,'trajectory',record),false,'late result from the first visit cannot resurrect history');
  }
});

test('a streamed current mean is displayable without masquerading as a completed finite trajectory',()=>{
  const store=new CurrentDynamicsResults(),epoch=store.activate('current'),mean={times:[0]},full={times:[0,1,2]};
  store.set('current',epoch,'mean',mean);store.set('current',epoch,'progress',mean);
  assert.strictEqual(store.get('current','mean'),mean);assert.strictEqual(store.get('current','progress'),mean);assert.equal(store.get('current','trajectory'),undefined);
  store.set('current',epoch,'trajectory',full);assert.strictEqual(store.get('current','trajectory'),full);assert.equal(store.get('current','progress'),undefined);
  store.clear();assert.equal(store.get('current','mean'),undefined);assert.equal(store.get('current','trajectory'),undefined);
});
