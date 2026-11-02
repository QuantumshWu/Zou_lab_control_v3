// Partial results (ui-design.md §6.16): every computed result is read and shown; samples whose
// estimated error exceeds ε, and a p̄(T) outside it, are marked rather than withheld.
import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {POPULATION_TOLERANCE as E,defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {populationChart} from '../dist/js/atom-workbench/src/population-chart.js';
import {errorPercent,outsideRuns,partialSentence,partialShort} from '../dist/js/atom-workbench/src/accuracy.js';
import {remainingWords} from '../dist/js/atom-workbench/src/format.js';
import {HORIZONS,entryAt,nativeResult} from './native-result.mjs';

// Every error is written relative to the tolerance E.
const P=errorPercent;

const scene=defaultScene(),states=buildFullMasterModel(scene,scene.dynamics).states,initialIndex=states.findIndex(s=>s.id===scene.dynamics.initialStateId);
const times=n=>Array.from({length:n},(_,i)=>i*scene.dynamics.durationUs/(n-1));
const result=options=>({...nativeResult({states,initialIndex,...options}),assumptions:[]});
const read=data=>normalizeDynamicsResult(data,scene);

test('a converged trajectory reads its per-sample errors',()=>{
  const r=read(result({timesUs:times(5),sampleErrors:[0,.8*E,.6*E,.4*E,.2*E]}));
  assert.equal(r.populationAccuracy.status,'estimated-converged');assert.deepEqual(r.populationAccuracy.sampleErrors,[0,.8*E,.6*E,.4*E,.2*E]);
  assert.equal(r.populationAccuracy.estimatedMaximumError,.8*E);assert.equal(r.populationAccuracy.reason,undefined);
});

test('a partial trajectory is read, not rejected: where it misses ε, by how much, and why',()=>{
  const errors=[0,1.22*E,1.04*E,.8*E,.2*E],data=result({timesUs:times(5),sampleErrors:errors});
  const r=read(data);assert.equal(r.populationAccuracy.status,'estimated-partial');assert.equal(r.populationAccuracy.reason,'test budget');
  assert.equal(r.populationAccuracy.estimatedMaximumError,1.22*E);assert.deepEqual(r.populationAccuracy.sampleErrors,errors);
  assert.equal(r.timesUs.length,5);assert.equal(r.populations[scene.dynamics.initialStateId].length,5,'the curves are all there');
  for(const [label,change] of [
    ['no reason',d=>{delete d.populationAccuracy.reason;}],
    ['maximum not that of the samples',d=>{d.populationAccuracy.estimatedMaximumError=1.4*E;}],
    ['one error short',d=>{d.populationAccuracy.sampleErrors.pop();}],
    ['a negative error',d=>{d.populationAccuracy.sampleErrors[2]=-1;}],
    ['a converged status over ε',d=>{d.populationAccuracy.status='estimated-converged';delete d.populationAccuracy.reason;}],
    ['an unknown status',d=>{d.populationAccuracy.status='estimated-unknown';}],
  ]){const bad=structuredClone(data);change(bad);assert.throws(()=>read(bad),/布居精度元数据/,label);}
  const converged=result({timesUs:times(3)});converged.populationAccuracy.reason='none';assert.throws(()=>read(converged),/布居精度元数据/,'a reason only when partial');
});

test('converged and within ε are the bounds within ε, exactly: a status that disagrees is refused',()=>{
  // The trajectory: partial with every sample within ε.
  const data=result({timesUs:times(5),sampleErrors:[0,.2*E,.4*E,.2*E,.2*E]});
  Object.assign(data.populationAccuracy,{status:'estimated-partial',reason:'stopped'});assert.throws(()=>read(data),/布居精度元数据/);
  // The mean: not within for a bound within ε, within for one above it, or said other than true or false.
  const unclaimed=result({timesUs:[0],meanError:.14*E});entryAt(unclaimed).withinTolerance=false;assert.throws(()=>read(unclaimed),/长期布居误差估计/);
  const claimed=result({timesUs:[0],meanError:1.44*E});entryAt(claimed).withinTolerance=true;assert.throws(()=>read(claimed),/长期布居误差估计/);
  const vague=result({timesUs:[0],meanError:.14*E});delete entryAt(vague).withinTolerance;assert.throws(()=>read(vague),/长期布居误差估计/);
});

test('the old format, with no per-sample errors or mean error, is no longer read',()=>{
  const old=result({timesUs:times(3)});delete old.populationAccuracy.sampleErrors;assert.throws(()=>read(old),/布居精度元数据/);
  const mean=result({timesUs:[0]});delete entryAt(mean).estimatedError;delete entryAt(mean).withinTolerance;assert.throws(()=>read(mean),/长期布居误差估计/);
});

test('a p̄(T) outside the tolerance is computed and shown, with its estimated error',()=>{
  const r=read(result({timesUs:[0],meanError:1.44*E}));
  assert.equal(r.mean.converged,true);assert.equal(r.mean.withinTolerance,false);assert.equal(r.mean.estimatedError,1.44*E);assert.ok(r.mean.populations);
  const within=read(result({timesUs:[0],meanError:.2*E}));assert.equal(within.mean.withinTolerance,true);
  for(const change of [e=>{e.withinTolerance=true;},e=>{e.estimatedError=NaN;},e=>{e.withinTolerance='no';}]){
    const bad=result({timesUs:[0],meanError:1.44*E});change(entryAt(bad));assert.throws(()=>read(bad),/长期布居误差估计/);
  }
  // 'not-converged': nothing usable computed -- no populations, never within the tolerance.
  const none=result({timesUs:[0],mean:false});none.meanHorizons=HORIZONS.map(horizonUs=>({horizonUs,status:'not-converged',withinTolerance:false}));
  assert.equal(read(none).mean.converged,false);
  entryAt(none).withinTolerance=true;assert.throws(()=>read(none),/长期布居误差估计/);
});

test('the status sentence says where the trajectory misses ε',()=>{
  const t=[0,.01,.021,.03,.04,.05];
  assert.deepEqual(outsideRuns([0,1.22*E,1.04*E,.8*E,.2*E,0]),[[1,2]]);
  assert.equal(partialSentence(t,[0,1.22*E,1.04*E,.8*E,.2*E,0],'μs'),`t ≤ 0.021 μs 估计误差最大 ${P(1.22*E)}（超过 ${P(E)}），其余时刻 ≤ ${P(E)}`);
  assert.equal(partialSentence(t,[0,0,0,1.2*E,.2*E,0],'μs'),`t = 0.03 μs 估计误差最大 ${P(1.2*E)}（超过 ${P(E)}），其余时刻 ≤ ${P(E)}`);
  assert.equal(partialSentence(t,[0,0,0,1.2*E,1.4*E,1.6*E],'μs'),`t ≥ 0.03 μs 估计误差最大 ${P(1.6*E)}（超过 ${P(E)}），其余时刻 ≤ ${P(E)}`);
  assert.equal(partialSentence(t,[0,1.2*E,0,0,1.8*E,0],'μs'),`2 段时刻（最早 t = 0.01 μs） 估计误差最大 ${P(1.8*E)}（超过 ${P(E)}），其余时刻 ≤ ${P(E)}`);
  assert.equal(partialSentence(t,[0,.2*E,.2*E,.2*E,.2*E,.2*E],'μs'),'');assert.equal(errorPercent(.0061),'0.61%');
  // Beside p̄(T)'s error the status line says it short; the full sentence stays in its tooltip.
  assert.equal(partialShort(t,[0,1.22*E,1.04*E,.8*E,.2*E,0],'μs'),`t ≤ 0.021 μs 误差最大 ${P(1.22*E)}`);
  assert.equal(partialShort(t,[0,1.2*E,0,0,1.8*E,0],'μs'),`2 段时刻误差最大 ${P(1.8*E)}`,'several runs by their number; the band shows where');
  assert.equal(partialShort(t,[0,.2*E,.2*E,.2*E,.2*E,.2*E],'μs'),'');
});

test('while a solve runs the status line counts its latest estimate of the time left down',()=>{
  // Whole seconds below 10 s (never 0 s), 5 s steps below a minute, whole minutes beyond; each
  // with the time until it says the next value.
  assert.deepEqual(remainingWords(7400,0),{words:'预计剩余约 7 s',changesInMs:901});
  assert.deepEqual(remainingWords(7400,1000),{words:'预计剩余约 6 s',changesInMs:901},'counted down since the checkpoint arrived');
  assert.deepEqual(remainingWords(1200,0),{words:'预计剩余约 1 s',changesInMs:1201});
  assert.deepEqual(remainingWords(300,0),{words:'预计剩余约 1 s',changesInMs:301});
  assert.deepEqual(remainingWords(12400,0),{words:'预计剩余约 10 s',changesInMs:2901},'below 9.5 s it says seconds');
  assert.deepEqual(remainingWords(13000,0),{words:'预计剩余约 15 s',changesInMs:501});
  assert.deepEqual(remainingWords(58000,0),{words:'预计剩余约 1 min',changesInMs:501});
  assert.deepEqual(remainingWords(100000,0),{words:'预计剩余约 2 min',changesInMs:10001});
  assert.deepEqual(remainingWords(100000,0,true),{words:'预计剩余不超过 2 min',changesInMs:10001},'an upper bound says so');
  // The run stops at its checkpoint: the result is on its way.
  assert.deepEqual(remainingWords(0,5000),{words:'即将完成'});
  // A countdown that has run out says nothing until the next checkpoint brings an estimate.
  assert.deepEqual(remainingWords(5000,5000),{words:''});assert.deepEqual(remainingWords(5000,6000),{words:''});
});

test('the chart draws every curve, its samples above ε on a light band, dashed and faded there',()=>{
  // 100 samples, the first 2% (after t = 0) above ε.
  const n=100,timesUs=times(n),errors=timesUs.map((_,i)=>i===0?0:i<=2?1.22*E:.4*E),r=read(result({timesUs,sampleErrors:errors}));
  const id=scene.dynamics.initialStateId,svg=populationChart(r,[id],{[id]:'初态'},{showTotalExcited:false});
  const bands=svg.match(/<rect class="accuracy-band"[^>]*>/g)??[];assert.equal(bands.length,1);
  const x=Number(/x="([\d.]+)"/.exec(bands[0])[1]),width=Number(/width="([\d.]+)"/.exec(bands[0])[1]);
  // From half-way between t = 0 (exact) and the first sample above ε, to half-way past the last one.
  const step=714/(n-1);assert.ok(Math.abs(x-(64+step/2))<1e-3,`${x}`);assert.ok(Math.abs(width-2*step)<1e-3,`${width}`);
  const lines=svg.match(/<polyline[^>]*>/g);assert.equal(lines.length,2,'one dashed stretch, one solid');
  assert.match(lines[0],/stroke-dasharray="4 3" stroke-opacity="0.55" class="outside-tolerance"/);assert.doesNotMatch(lines[1],/stroke-dasharray/);
  assert.ok(svg.includes(`估计误差最大 ${P(1.22*E)}，超过 ${P(E)}`));
  // A converged trajectory: one solid curve, no band.
  const clean=populationChart(read(result({timesUs,sampleErrors:timesUs.map((_,i)=>i?.4*E:0)})),[id],{[id]:'初态'},{showTotalExcited:false});
  assert.equal((clean.match(/<polyline/g)||[]).length,1);assert.ok(!clean.includes('accuracy-band')&&!clean.includes('stroke-dasharray'));
});
