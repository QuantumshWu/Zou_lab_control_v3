import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {meanInputsKey,trajectoryInputsKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';
import {HORIZONS,nativeResult} from './native-result.mjs';

test('the time T of p̄(T) is a dynamics setting: 1 s by default, saved and required; no input key holds it',()=>{
  const scene=defaultScene();assert.equal(scene.dynamics.meanHorizonUs,1e6);
  for(const horizon of [1e5,1e6]){const s=structuredClone(scene);s.dynamics.meanHorizonUs=horizon;assert.equal(validateScene(JSON.parse(JSON.stringify(s))).dynamics.meanHorizonUs,horizon);}
  const file=JSON.parse(JSON.stringify(scene));
  assert.throws(()=>validateScene({...file,dynamics:{...file.dynamics,meanHorizonUs:3e6}}),/dynamics\.meanHorizonUs/);
  assert.throws(()=>validateScene({...file,dynamics:{...file.dynamics,meanHorizonUs:null}}),/dynamics\.meanHorizonUs/,'there is no t → ∞');
  delete file.dynamics.meanHorizonUs;assert.throws(()=>validateScene(file),/dynamics 缺少 meanHorizonUs/);
  const other={...scene.dynamics,meanHorizonUs:1e5};
  // A mean run returns every T: the means are cached per T under inputs without it.
  assert.equal(meanInputsKey(scene,scene.dynamics),meanInputsKey(scene,other),'the mean inputs do not hold T');
  assert.equal(trajectoryInputsKey(scene,scene.dynamics),trajectoryInputsKey(scene,other),'a trajectory does not depend on it');
});

test('the time T goes to the solver, and the reader takes that T\'s mean',()=>{
  const scene=defaultScene();
  assert.equal(prepareNativeSolve({scene,options:{meanHorizonUs:1e5}}).payload.options.meanHorizonUs,1e5);
  assert.equal(Object.hasOwn(prepareNativeSolve({scene,options:{}}).payload.options,'meanHorizonUs'),false);
  const states=buildFullMasterModel(scene,scene.dynamics).states,index=states.findIndex(s=>s.id===scene.dynamics.initialStateId);
  const data={...nativeResult({states,initialIndex:index,timesUs:[0]}),assumptions:[]};
  assert.equal(normalizeDynamicsResult(data,scene).steady.horizonUs,1e6,'the scene\'s T unless said');
  assert.equal(normalizeDynamicsResult(data,scene,undefined,1e5).steady.horizonUs,1e5);
  assert.throws(()=>normalizeDynamicsResult(data,scene,undefined,42),/长期布居时刻 T/,'no mean at another T');
});

/** A mean result at T `horizonUs` with the mean at both T (asymptoticHorizons). */
function meanData(options={}){
  const scene=defaultScene(),states=buildFullMasterModel(scene,scene.dynamics).states,index=states.findIndex(s=>s.id===scene.dynamics.initialStateId);
  return {scene,data:{...nativeResult({states,initialIndex:index,timesUs:[0],...options}),assumptions:[]}};
}
const entry=(data,T)=>data.asymptoticHorizons.find(e=>e.horizonUs===T);

test('every mean run returns the mean at both standard T: the requested T\'s is the mean itself',()=>{
  const {scene,data}=meanData({horizonUs:1e5,meanError:.001,horizonErrors:{'1000000':.009}});
  const result=normalizeDynamicsResult(data,scene,undefined,1e5);
  assert.deepEqual(result.horizonMeans.map(s=>s.horizonUs),HORIZONS,'one per T, in order');
  assert.deepEqual(HORIZONS,[1e5,1e6],'100 ms and 1 s: no t → ∞');
  assert.strictEqual(result.horizonMeans[0],result.steady,'the requested T\'s is the primary mean, its diagnostics and all');
  assert.deepEqual(result.horizonMeans.map(s=>[s.estimatedError,s.withinTolerance]),[[.001,true],[.009,false]]);
  assert.equal(result.horizonMeans[1].converged,true,'outside ε is still computed, and shown with its error');
  assert.equal(result.horizonMeans[1].method,'test mean');
  assert.ok(result.horizonMeans[1].density,'each T keeps its complex density');
  // The order of the list is the solver's; the reader keys it by T.
  assert.equal(normalizeDynamicsResult({...data,asymptoticHorizons:[...data.asymptoticHorizons].reverse()},scene,undefined,1e5).steady.estimatedError,.001);
  // A trajectory alone carries no mean, and no list.
  const {data:none}=meanData({mean:false});assert.equal(normalizeDynamicsResult(none,scene).horizonMeans,undefined);
});

test('asymptoticHorizons is read strictly, every T under the same rules',()=>{
  const {scene,data}=meanData({horizonUs:1e6,meanError:.001});
  const variants=[
    ['a T missing',d=>{d.asymptoticHorizons.pop();}],
    ['a T twice',d=>{d.asymptoticHorizons[1]={...d.asymptoticHorizons[0]};}],
    ['a third entry',d=>{d.asymptoticHorizons.push({...d.asymptoticHorizons[0],horizonUs:3e6});}],
    ['a t → ∞ entry beside them',d=>{d.asymptoticHorizons.push({...d.asymptoticHorizons[0],horizonUs:null});}],
    ['an entry at t → ∞ for a T',d=>{entry(d,1e6).horizonUs=null;}],
    ['another T',d=>{entry(d,1e6).horizonUs=3e6;}],
    ['T absent',d=>{delete entry(d,1e6).horizonUs;}],
    ['a status of the t → ∞ limit',d=>{entry(d,1e5).status='initial-dependent-time-average';}],
    ['a stationary state',d=>{entry(d,1e5).status='stationary';}],
    ['not an object',d=>{d.asymptoticHorizons[0]=[1];}],
    ['an unknown status',d=>{entry(d,1e5).status='periodic-mean';}],
    ['not requested in the list',d=>{entry(d,1e5).status='not-requested';}],
    ['a reason that is no text',d=>{entry(d,1e5).reason=7;}],
    ['within ε claimed for an error above it',d=>{Object.assign(entry(d,1e5),{estimatedError:.02,withinTolerance:true});}],
    ['not within for an error within ε',d=>{Object.assign(entry(d,1e5),{estimatedError:.0007,withinTolerance:false});}],
    ['within said other than true or false',d=>{entry(d,1e5).withinTolerance='yes';}],
    ['an error that is no bound',d=>{entry(d,1e5).estimatedError=-1;}],
    ['no error for a computed mean',d=>{delete entry(d,1e5).estimatedError;}],
    ['populations that are no probabilities',d=>{entry(d,1e5).populations=entry(d,1e5).populations.map(()=>.5);}],
    ['a density off its diagonal',d=>{entry(d,1e5).densityReal=entry(d,1e5).densityReal.map((v,i)=>i===0?v+.1:v);}],
    ['no density',d=>{delete entry(d,1e5).densityImag;}],
    ['populations without a computed mean',d=>{Object.assign(entry(d,1e5),{status:'not-converged',estimatedError:null,withinTolerance:false});}],
  ];
  for(const [label,change] of variants){const d=structuredClone(data);change(d);assert.throws(()=>normalizeDynamicsResult(d,scene),/长期/,label);}
});

test('a T the run did not resolve is said so, and the requested mean may be that T',()=>{
  const {scene,data}=meanData({horizonUs:1e6,meanError:.001});
  Object.assign(entry(data,1e5),{status:'not-converged',reason:'no mean resolved at 100 ms',populations:null,densityReal:null,densityImag:null,estimatedError:null,withinTolerance:false});
  let result=normalizeDynamicsResult(data,scene);
  assert.equal(result.horizonMeans[0].converged,false);assert.equal(result.horizonMeans[0].populations,null);assert.equal(result.horizonMeans[0].reason,'no mean resolved at 100 ms');
  delete entry(data,1e5).withinTolerance;delete entry(data,1e5).estimatedError;delete entry(data,1e5).populations;
  result=normalizeDynamicsResult(data,scene);assert.equal(result.horizonMeans[0].estimatedError,null,'absent and null are the same here');
  // The mean asked for at 100 ms, not resolved: the requested mean says so, and 1 s stands.
  result=normalizeDynamicsResult(data,scene,undefined,1e5);
  assert.equal(result.steady.converged,false);assert.strictEqual(result.horizonMeans[0],result.steady);assert.equal(result.horizonMeans[1].converged,true);
});
