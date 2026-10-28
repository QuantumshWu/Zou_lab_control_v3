// Partial results (ui-design.md §6.16): every computed result is read and shown; samples whose
// estimated error exceeds ε, and a p̄(T) outside it, are marked rather than withheld.
import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {populationChart} from '../dist/js/atom-workbench/src/population-chart.js';
import {errorPercent,outsideRuns,partialSentence,partialShort} from '../dist/js/atom-workbench/src/accuracy.js';
import {nativeResult} from './native-result.mjs';

const scene=defaultScene(),states=buildFullMasterModel(scene,scene.dynamics).states,initialIndex=states.findIndex(s=>s.id===scene.dynamics.initialStateId);
const times=n=>Array.from({length:n},(_,i)=>i*scene.dynamics.durationUs/(n-1));
const result=options=>({...nativeResult({states,initialIndex,...options}),assumptions:[]});
const read=data=>normalizeDynamicsResult(data,scene);

test('a converged trajectory reads its per-sample errors',()=>{
  const r=read(result({timesUs:times(5),sampleErrors:[0,.004,.003,.002,.001]}));
  assert.equal(r.populationAccuracy.status,'estimated-converged');assert.deepEqual(r.populationAccuracy.sampleErrors,[0,.004,.003,.002,.001]);
  assert.equal(r.populationAccuracy.estimatedMaximumError,.004);assert.equal(r.populationAccuracy.reason,undefined);
});

test('a partial trajectory is read, not rejected: where it misses ε, by how much, and why',()=>{
  const errors=[0,.0061,.0052,.004,.001],data=result({timesUs:times(5),sampleErrors:errors});
  const r=read(data);assert.equal(r.populationAccuracy.status,'estimated-partial');assert.equal(r.populationAccuracy.reason,'test budget');
  assert.equal(r.populationAccuracy.estimatedMaximumError,.0061);assert.deepEqual(r.populationAccuracy.sampleErrors,errors);
  assert.equal(r.timesUs.length,5);assert.equal(r.populations[scene.dynamics.initialStateId].length,5,'the curves are all there');
  for(const [label,change] of [
    ['no reason',d=>{delete d.populationAccuracy.reason;}],
    ['partial with every sample within ε',d=>{d.populationAccuracy.sampleErrors=[0,.001,.001,.001,.001];d.populationAccuracy.estimatedMaximumError=.001;}],
    ['maximum not that of the samples',d=>{d.populationAccuracy.estimatedMaximumError=.007;}],
    ['one error short',d=>{d.populationAccuracy.sampleErrors.pop();}],
    ['a negative error',d=>{d.populationAccuracy.sampleErrors[2]=-1;}],
    ['a converged status over ε',d=>{d.populationAccuracy.status='estimated-converged';delete d.populationAccuracy.reason;}],
    ['an unknown status',d=>{d.populationAccuracy.status='estimated-unknown';}],
  ]){const bad=structuredClone(data);change(bad);assert.throws(()=>read(bad),/布居精度元数据/,label);}
  const converged=result({timesUs:times(3)});converged.populationAccuracy.reason='none';assert.throws(()=>read(converged),/布居精度元数据/,'a reason only when partial');
});

test('the old format, with no per-sample errors or mean error, is no longer read',()=>{
  const old=result({timesUs:times(3)});delete old.populationAccuracy.sampleErrors;assert.throws(()=>read(old),/布居精度元数据/);
  const mean=result({timesUs:[0]});delete mean.asymptoticEstimatedError;delete mean.asymptoticWithinTolerance;assert.throws(()=>read(mean),/长期布居误差估计/);
});

test('a p̄(T) outside the tolerance is computed and shown, with its estimated error',()=>{
  const r=read(result({timesUs:[0],meanError:.0072}));
  assert.equal(r.steady.converged,true);assert.equal(r.steady.withinTolerance,false);assert.equal(r.steady.estimatedError,.0072);assert.ok(r.steady.populations);
  const within=read(result({timesUs:[0],meanError:.001}));assert.equal(within.steady.withinTolerance,true);
  for(const change of [d=>{d.asymptoticWithinTolerance=true;},d=>{d.asymptoticEstimatedError=NaN;},d=>{d.asymptoticWithinTolerance='no';}]){
    const bad=result({timesUs:[0],meanError:.0072});change(bad);assert.throws(()=>read(bad),/长期布居误差估计/);
  }
  // 'not-converged': nothing usable computed -- no populations, never within the tolerance.
  const none=result({timesUs:[0],mean:false});none.asymptoticStatus='not-converged';assert.equal(read(none).steady.converged,false);
  none.asymptoticWithinTolerance=true;assert.throws(()=>read(none),/长期布居误差估计/);
});

test('the status sentence says where the trajectory misses ε',()=>{
  const t=[0,.01,.021,.03,.04,.05];
  assert.deepEqual(outsideRuns([0,.0061,.0052,.004,.001,0]),[[1,2]]);
  assert.equal(partialSentence(t,[0,.0061,.0052,.004,.001,0],'μs'),'t ≤ 0.021 μs 估计误差最大 0.61%（超过 0.5%），其余时刻 ≤ 0.5%');
  assert.equal(partialSentence(t,[0,0,0,.006,.001,0],'μs'),'t = 0.03 μs 估计误差最大 0.6%（超过 0.5%），其余时刻 ≤ 0.5%');
  assert.equal(partialSentence(t,[0,0,0,.006,.007,.008],'μs'),'t ≥ 0.03 μs 估计误差最大 0.8%（超过 0.5%），其余时刻 ≤ 0.5%');
  assert.equal(partialSentence(t,[0,.006,0,0,.009,0],'μs'),'2 段时刻（最早 t = 0.01 μs） 估计误差最大 0.9%（超过 0.5%），其余时刻 ≤ 0.5%');
  assert.equal(partialSentence(t,[0,.001,.001,.001,.001,.001],'μs'),'');assert.equal(errorPercent(.0061),'0.61%');
  // Beside p̄(T)'s error the status line says it short; the full sentence stays in its tooltip.
  assert.equal(partialShort(t,[0,.0061,.0052,.004,.001,0],'μs'),'t ≤ 0.021 μs 误差最大 0.61%');
  assert.equal(partialShort(t,[0,.006,0,0,.009,0],'μs'),'2 段时刻（最早 t = 0.01 μs） 误差最大 0.9%');
  assert.equal(partialShort(t,[0,.001,.001,.001,.001,.001],'μs'),'');
});

test('the chart draws every curve, its samples above ε on a light band, dashed and faded there',()=>{
  // 100 samples, the first 2% (after t = 0) above ε.
  const n=100,timesUs=times(n),errors=timesUs.map((_,i)=>i===0?0:i<=2?.0061:.002),r=read(result({timesUs,sampleErrors:errors}));
  const id=scene.dynamics.initialStateId,svg=populationChart(r,[id],{[id]:'初态'},{showTotalExcited:false});
  const bands=svg.match(/<rect class="accuracy-band"[^>]*>/g)??[];assert.equal(bands.length,1);
  const x=Number(/x="([\d.]+)"/.exec(bands[0])[1]),width=Number(/width="([\d.]+)"/.exec(bands[0])[1]);
  // From half-way between t = 0 (exact) and the first sample above ε, to half-way past the last one.
  const step=714/(n-1);assert.ok(Math.abs(x-(64+step/2))<1e-3,`${x}`);assert.ok(Math.abs(width-2*step)<1e-3,`${width}`);
  const lines=svg.match(/<polyline[^>]*>/g);assert.equal(lines.length,2,'one dashed stretch, one solid');
  assert.match(lines[0],/stroke-dasharray="4 3" stroke-opacity="0.55" class="outside-tolerance"/);assert.doesNotMatch(lines[1],/stroke-dasharray/);
  assert.match(svg,/估计误差最大 0.61%，超过 0.5%/);
  // A converged trajectory: one solid curve, no band.
  const clean=populationChart(read(result({timesUs,sampleErrors:timesUs.map((_,i)=>i?.002:0)})),[id],{[id]:'初态'},{showTotalExcited:false});
  assert.equal((clean.match(/<polyline/g)||[]).length,1);assert.ok(!clean.includes('accuracy-band')&&!clean.includes('stroke-dasharray'));
});
