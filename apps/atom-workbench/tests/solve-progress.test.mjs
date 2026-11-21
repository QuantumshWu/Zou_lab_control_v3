// A running motion solve's course (solve-progress.ts): the pipeline's steps as the engine's stages say them -- done
// with the time they took, in progress with the time so far --, what the step in progress is at, the checkpoints, and
// the gauge of how far the run stands from its targets.
import test from 'node:test';
import assert from 'node:assert/strict';
import {addMark,clockChangesInMs,clockWords,courseMark,courseView,currentStep} from '../dist/js/atom-workbench/src/solve-progress.js';
import {SEMICLASSICAL_POPULATION_TARGET} from '../dist/js/atom-workbench/src/solver-types.js';

const mark=(atMs,trajectories,error,stepUs=1.6667)=>({atMs,trajectories,stepUs,error});
const course=(stages,marks=[])=>({startedAt:0,stages:stages.map(([atMs,stage])=>({atMs,stage})),marks});
const shown=view=>view.nodes.map(n=>`${n.label}|${n.detail}|${n.state}`);
const STAGES=[[200,{step:'select'}],[3200,{step:'model'}],[5000,{step:'scout',stepUs:3.3333}],[9000,{step:'scout',stepUs:1.6667}],[12000,{step:'pilot',stepUs:1.6667}],
  [60000,{step:'product',stepUs:1.6667,batch:{number:1,trajectories:88,want:176,worst:1.85}}],[95000,{step:'product',stepUs:1.6667,batch:{number:2,trajectories:176,want:352,worst:1.2}}]];

test('a checkpoint\'s mark: when it came, its trajectories, its step and its largest population error; none without motion; the last again is no new one',()=>{
  const result={motion:{trajectories:264,timeStepUs:.7125},populationAccuracy:{estimatedMaximumError:.072}};
  assert.deepEqual(courseMark(result,1234),{atMs:1234,trajectories:264,stepUs:.7125,error:.072});
  assert.equal(courseMark({populationAccuracy:{estimatedMaximumError:.01}},5),undefined,'a result without motion has no course');
  const c=course([]);addMark(c,mark(1,88,.1));addMark(c,mark(61,88,.1));addMark(c,mark(90,176,.08));
  assert.deepEqual(c.marks.map(m=>m.atMs),[1,90],'the engine\'s last result sent again (Messages::again) is the same checkpoint');
});

test('before any stage the selection is in progress from the request on; then each step as its stage says',()=>{
  const before=courseView(course([]),2500);
  assert.deepEqual(shown(before),['选态|已用 2 s|current','建模型||pending','侦察||pending','先导||pending','正式批次||pending','结果||pending']);
  assert.equal(before.gauge,undefined);assert.equal(before.words,'尚无检查点');
  const model=courseView(course(STAGES.slice(0,2)),4500);
  assert.deepEqual(shown(model),['选态|3 s|done','建模型|已用 1 s|current','侦察||pending','先导||pending','正式批次||pending','结果||pending']);
  assert.match(model.nodes[0].title,/^选态：.*三点.*（开始后 0 s 开始，用时 3 s）$/);
  // The scout moved down a rung: its words say both.
  const scout=courseView(course(STAGES.slice(0,4)),10500);
  assert.equal(scout.nodes[2].detail,'已用 5 s');assert.equal(scout.words,'侦察 · 时间步长 3.33 → 1.67 μs · 尚无检查点');
  assert.deepEqual([currentStep(course(STAGES.slice(0,4))),currentStep(course([]))],[{step:'scout',sinceMs:5000},{step:'select',sinceMs:0}]);
});

test('in the pilot the gauge is the last checkpoint\'s populations; in the product the batch\'s own, its largest total over target',()=>{
  const pilot=courseView(course(STAGES.slice(0,5),[mark(70000,88,.072)]),72000);
  assert.deepEqual(shown(pilot).slice(2,5),['侦察|7 s|done','先导|已用 1 min 0 s|current','正式批次||pending']);
  assert.equal(pilot.gauge.fraction,SEMICLASSICAL_POPULATION_TARGET/.072);assert.equal(pilot.gauge.words,'布居误差 7.2%，目标 5%');
  assert.equal(pilot.words,'先导 · 时间步长 1.67 μs · 检查点 1');
  const product=courseView(course(STAGES,[mark(60000,88,.072),mark(95000,176,.061)]),125500);
  assert.deepEqual(shown(product).slice(3),['先导|48 s|done','正式批次|已用 1 min 5 s|current','结果||pending']);
  assert.equal(product.gauge.fraction,1/1.2);assert.equal(product.gauge.words,'误差最大的一项是目标的 1.2 倍');
  assert.equal(product.words,'第 2 批：176 → 352 条轨迹 · 时间步长 1.67 μs · 检查点 2');
  assert.match(product.nodes[4].title,/^正式批次：.*（开始后 1 min 0 s 开始，已用 1 min 5 s；时间步长 1.67 μs）$/);
});

test('the course\'s clock: whole seconds, then minutes and seconds, then hours and minutes; it changes at the next whole second of the step in progress',()=>{
  assert.deepEqual([0,999,59999,60000,3599999,3723000].map(ms=>clockWords(ms)),['0 s','0 s','59 s','1 min 0 s','59 min 59 s','1 h 2 min']);
  assert.deepEqual([0,1500,2999].map(ms=>clockChangesInMs(ms)),[1000,500,1]);
  assert.equal(clockChangesInMs(5300,5000),700,'the step began at 5 s: its clock turns at 6 s');
});
