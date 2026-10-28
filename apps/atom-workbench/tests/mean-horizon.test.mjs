import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';
import {nativeResult} from './native-result.mjs';

test('the time T of p̄(T) is a dynamics setting: 1 s by default, saved, required and keyed',()=>{
  const scene=defaultScene();assert.equal(scene.dynamics.meanHorizonUs,1e6);
  for(const horizon of [1e5,1e6,null]){const s=structuredClone(scene);s.dynamics.meanHorizonUs=horizon;assert.equal(validateScene(JSON.parse(JSON.stringify(s))).dynamics.meanHorizonUs,horizon);}
  const file=JSON.parse(JSON.stringify(scene));
  assert.throws(()=>validateScene({...file,dynamics:{...file.dynamics,meanHorizonUs:3e6}}),/dynamics\.meanHorizonUs/);
  delete file.dynamics.meanHorizonUs;assert.throws(()=>validateScene(file),/dynamics 缺少 meanHorizonUs/);
  const other={...scene.dynamics,meanHorizonUs:null};
  assert.notEqual(dynamicsCacheKey(scene,scene.dynamics,true),dynamicsCacheKey(scene,other,true),'the mean depends on it');
  assert.equal(dynamicsCacheKey(scene,scene.dynamics,false,false),dynamicsCacheKey(scene,other,false,false),'a trajectory does not');
});

test('the time T goes to the solver, and a computed mean echoes the one asked for',()=>{
  const scene=defaultScene();
  assert.equal(prepareNativeSolve({scene,options:{meanHorizonUs:null}}).payload.options.meanHorizonUs,null);
  assert.equal(prepareNativeSolve({scene,options:{meanHorizonUs:1e5}}).payload.options.meanHorizonUs,1e5);
  assert.equal(Object.hasOwn(prepareNativeSolve({scene,options:{}}).payload.options,'meanHorizonUs'),false);
  const states=buildFullMasterModel(scene,scene.dynamics).states,index=states.findIndex(s=>s.id===scene.dynamics.initialStateId);
  const data={...nativeResult({states,initialIndex:index,timesUs:[0]}),assumptions:[]};
  assert.equal(normalizeDynamicsResult(data,scene).steady.horizonUs,1e6);
  assert.equal(normalizeDynamicsResult({...data,horizonUs:null},scene).steady.horizonUs,null);
  const {horizonUs:_,...unechoed}=data;assert.throws(()=>normalizeDynamicsResult(unechoed,scene),/长期布居时刻 T/,'a computed mean says its T');
  assert.throws(()=>normalizeDynamicsResult({...data,horizonUs:42},scene),/长期布居时刻 T/);
});
