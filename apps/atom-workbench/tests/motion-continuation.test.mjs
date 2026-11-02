// A 1D motion result's continuation (ui-design.md §6.20): `motion.continuedFromUs`, the last
// simulated time when the later samples are the engine's prediction from the simulated history,
// read strictly and marked in every time view.
import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {continuedAt} from '../dist/js/atom-workbench/src/solver-types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {exampleScene} from '../dist/js/atom-workbench/src/examples.js';
import {motionRequest} from '../dist/js/atom-workbench/src/motion.js';
import {populationChart} from '../dist/js/atom-workbench/src/population-chart.js';
import {motionSeries,seriesChart} from '../dist/js/atom-workbench/src/motion-chart.js';
import {nativeResult} from './native-result.mjs';

/** The tweezer example's motion result on 5 samples over its 10 ms, the samples after `from` (us) continued. */
const scene=exampleScene('pgc').scene;scene.dynamics.samples=5;
const motion=motionRequest(scene,scene.dynamics),duration=scene.dynamics.durationUs,timesUs=[0,.25,.5,.75,1].map(f=>f*duration);
function result(from){
  const states=buildFullMasterModel(scene,scene.dynamics).states,p=[.1,.2,.4,.2,.1];
  return {...nativeResult({states,initialIndex:states.findIndex(s=>s.id===scene.dynamics.initialStateId),timesUs,mean:false}),assumptions:['test'],
    motion:{axis:motion.axis,temperatureUK:[30,20,12,8,6],meanVelocityMps:[0,0,0,0,0],trapFrequencyKHz:100,meanOccupation:[6,4,3,2,1.5],lossProbability:[0,0,.01,.01,.02],
      momentum:{velocitiesMps:[-.1,-.05,0,.05,.1],samples:[0,4],probabilities:[p,p]},position:{positionsUm:[-.4,-.2,0,.2,.4],samples:[0,4],probabilities:[p,p]},
      ...(from===undefined?{}:{continuedFromUs:from})}};
}
const read=data=>normalizeDynamicsResult(data,scene,undefined,undefined,motion);

test('the continuation is read strictly: a time of the run before its last sample, or none',()=>{
  assert.equal(read(result()).motion.continuedFromUs,undefined,'every sample simulated');
  assert.equal(read(result(.25*duration)).motion.continuedFromUs,.25*duration);
  assert.equal(read(result(0)).motion.continuedFromUs,0,'all but t = 0 predicted');
  for(const bad of [duration,1.5*duration,-1,NaN,Infinity,null,'2500'])assert.throws(()=>read(result(bad)),/运动结果/,String(bad));
  const motionOf=read(result(.25*duration)).motion;
  assert.deepEqual(timesUs.map(t=>continuedAt(motionOf,t)),[false,false,true,true,true],'the samples after the last simulated time');
  assert.deepEqual(timesUs.map(t=>continuedAt(read(result()).motion,t)),[false,false,false,false,false]);
});

test('every time view shades the continuation from the last simulated time to the end, labelled 延拓',()=>{
  const band=svg=>{const rect=/<rect class="continuation-band" x="([\d.]+)" y="(\d+)" width="([\d.]+)" height="(\d+)"/.exec(svg);return rect&&rect.slice(1).map(Number);};
  // The populations: from x(2.5 ms) to the plot's right edge (64 + 714 · t / T), over the whole plot height.
  const r=read(result(.25*duration)),id=scene.dynamics.initialStateId;
  const populations=populationChart(r,[id],{[id]:'初态'},{showTotalExcited:false,continuedFromUs:.25*duration});
  const [x,y,width,height]=band(populations);
  assert.ok(Math.abs(x-(64+714*.25))<1e-3&&Math.abs(x+width-778)<1e-3&&y===32&&height===268,`${[x,y,width,height]}`);
  assert.match(populations,/>延拓<\/text>/);assert.match(populations,/<title>t &gt; 2\.5 ms：延拓，由模拟历史预测，不是逐步模拟<\/title>/);
  assert.equal(band(populationChart(r,[id],{[id]:'初态'},{showTotalExcited:false})),null,'none without a continuation');
  // T and ⟨v⟩, n̄ and the loss: across both panels.
  for(const kind of ['temperature','trap']){
    const chart=seriesChart(timesUs,motionSeries(r.motion,kind),'ms',r.motion.continuedFromUs),[sx,sy,sw,sh]=band(chart);
    assert.ok(Math.abs(sx-(64+714*.25))<1e-3&&Math.abs(sx+sw-778)<1e-3&&sy===32&&sh===268,kind);
    assert.equal(band(seriesChart(timesUs,motionSeries(r.motion,kind),'ms')),null,`${kind}: none without`);
  }
});
