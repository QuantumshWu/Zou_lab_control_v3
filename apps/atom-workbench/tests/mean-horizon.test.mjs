import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene,resolveMeanHorizon} from '../dist/js/atom-workbench/src/types.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';
import {nativeResult} from './native-result.mjs';

test('the long-time horizon is a dynamics setting: 1 s by default, saved, migrated and keyed',()=>{
  const scene=defaultScene();assert.equal(scene.dynamics.meanHorizonUs,1e6);
  const old=JSON.parse(JSON.stringify(scene));delete old.dynamics.meanHorizonUs;
  assert.equal(validateScene(old).dynamics.meanHorizonUs,1e6,'a scene from before reads at 1 s');
  for(const horizon of [1e5,1e6,null]){const s=structuredClone(scene);s.dynamics.meanHorizonUs=horizon;assert.equal(validateScene(JSON.parse(JSON.stringify(s))).dynamics.meanHorizonUs,horizon);}
  assert.throws(()=>validateScene({...old,dynamics:{...old.dynamics,meanHorizonUs:3e6}}),/长期布居的时刻 T 无效/);
  assert.equal(resolveMeanHorizon({}),1e6);assert.equal(resolveMeanHorizon({meanHorizonUs:null}),null);
  const other={...scene.dynamics,meanHorizonUs:null};
  assert.notEqual(dynamicsCacheKey(scene,scene.dynamics,true),dynamicsCacheKey(scene,other,true),'the mean depends on it');
  assert.equal(dynamicsCacheKey(scene,scene.dynamics,false,false),dynamicsCacheKey(scene,other,false,false),'a trajectory does not');
});

test('the horizon goes to the solver and a solver that echoes it must echo the one asked for',()=>{
  const scene=defaultScene();
  assert.equal(prepareNativeSolve({scene,options:{meanHorizonUs:null}}).payload.options.meanHorizonUs,null);
  assert.equal(prepareNativeSolve({scene,options:{meanHorizonUs:1e5}}).payload.options.meanHorizonUs,1e5);
  assert.equal(Object.hasOwn(prepareNativeSolve({scene,options:{}}).payload.options,'meanHorizonUs'),false);
  const states=buildFullMasterModel(scene,scene.dynamics).states,index=states.findIndex(s=>s.id===scene.dynamics.initialStateId);
  const data={...nativeResult({states,initialIndex:index,timesUs:[0]}),assumptions:[]};
  assert.equal(normalizeDynamicsResult(data,scene).steady.horizonUs,undefined,'not echoed: t → ∞');
  assert.equal(normalizeDynamicsResult({...data,horizonUs:1e6},scene).steady.horizonUs,1e6);
  assert.throws(()=>normalizeDynamicsResult({...data,horizonUs:42},scene),/长期布居时刻 T/);
});
