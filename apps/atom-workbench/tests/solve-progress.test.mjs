// A running motion solve's course (solve-progress.ts): the pipeline's steps as the engine's stages came -- done with
// the time they took, in progress with the time so far, a step again after another a node of its own --, those to come
// in their order, what the step in progress is at, the checkpoints, and the gauge of how far the run stands from its
// targets.
import test from 'node:test';
import assert from 'node:assert/strict';
import {addMark,clockChangesInMs,clockWords,courseMark,courseView,currentStep} from '../dist/js/atom-workbench/src/solve-progress.js';
import {SEMICLASSICAL_POPULATION_TARGET,SOLVE_STEPS} from '../dist/js/atom-workbench/src/solver-types.js';

const mark=(atMs,trajectories,error,stepUs=1.6667)=>({atMs,trajectories,stepUs,error});
const course=(stages,marks=[])=>({startedAt:0,stages:stages.map(([atMs,stage])=>({atMs,stage})),marks});
const shown=view=>view.nodes.map(n=>`${n.label}|${n.detail}|${n.state}`);
const batch=(number,trajectories,want,worst)=>({number,trajectories,want,...(worst===undefined?{}:{worst})});
// A new selection that chooses the all-ground model: the model, the scout (down a rung), the pilot, the selection, the
// product's batches.
const GROUND=[[300,{step:'model'}],[5000,{step:'scout',stepUs:3.3333}],[9000,{step:'scout',stepUs:1.6667}],[12000,{step:'pilot',stepUs:1.6667}],
  [40000,{step:'select'}],[52000,{step:'product',stepUs:1.6667,batch:batch(1,88,176,1.85)}],[80000,{step:'product',stepUs:1.6667,batch:batch(2,176,352,1.2)}]];
// A new selection that chooses another model: the first round on the all-ground one, the selection, then the model,
// the scout and the pilot again on the chosen one, and the product.
const OTHER=[...GROUND.slice(0,5),[52400,{step:'model'}],[55000,{step:'scout',stepUs:1.6667}],[58000,{step:'scout',stepUs:.8333}],[66000,{step:'pilot',stepUs:.8333}],
  [90000,{step:'product',stepUs:.8333,batch:batch(1,88,176,1.5)}]];
// The selection kept for the scene: the selection at once, then the model, the scout, the pilot and the product.
const KEPT=[[100,{step:'select'}],[400,{step:'model'}],[3000,{step:'scout',stepUs:3.3333}],[7000,{step:'scout',stepUs:1.6667}],[10000,{step:'pilot',stepUs:1.6667}],
  [40000,{step:'product',stepUs:1.6667,batch:batch(1,88,176)}]];

test('a checkpoint\'s mark: when it came, its trajectories, its step and its largest population error; none without motion; the last again is no new one',()=>{
  const result={motion:{trajectories:264,timeStepUs:.7125},populationAccuracy:{estimatedMaximumError:.072}};
  assert.deepEqual(courseMark(result,1234),{atMs:1234,trajectories:264,stepUs:.7125,error:.072});
  assert.equal(courseMark({populationAccuracy:{estimatedMaximumError:.01}},5),undefined,'a result without motion has no course');
  const c=course([]);addMark(c,mark(1,88,.1));addMark(c,mark(61,88,.1));addMark(c,mark(90,176,.08));
  assert.deepEqual(c.marks.map(m=>m.atMs),[1,90],'the engine\'s last result sent again (Messages::again) is the same checkpoint');
});

test('the steps\' order is a new selection\'s: the model, the scout, the pilot, the selection, the product',()=>{
  assert.deepEqual([...SOLVE_STEPS],['model','scout','pilot','select','product']);
});

test('before any stage the steps are all to come in their order, the model in progress from the request on',()=>{
  const before=courseView(course([]),2500);
  assert.deepEqual(shown(before),['建模型|已用 2 s|current','侦察||pending','先导||pending','选态||pending','正式批次||pending','结果||pending']);
  assert.match(before.nodes[0].title,/^建模型：.*全基态模型.*（开始后 0 s 开始，已用 2 s）$/);
  assert.equal(before.nodes.slice(1).some(n=>n.title.includes('开始后')),false,'a step to come says what it does, no times');
  assert.equal(before.gauge,undefined);assert.equal(before.words,'尚无检查点');
  assert.deepEqual(currentStep(course([])),{step:'model',sinceMs:0});
});

test('a new selection that chooses the all-ground model: the selection after the pilot, the product after it; the gauge the checkpoint\'s, then the batch\'s',()=>{
  // The model's stage came 0.3 s after the request: the first step counts from the request.
  const scout=courseView(course(GROUND.slice(0,3)),10500);
  assert.deepEqual(shown(scout),['建模型|5 s|done','侦察|已用 5 s|current','先导||pending','选态||pending','正式批次||pending','结果||pending']);
  assert.match(scout.nodes[0].title,/^建模型：.*（开始后 0 s 开始，用时 5 s）$/);
  assert.equal(scout.words,'侦察 · 时间步长 3.33 → 1.67 μs · 尚无检查点');
  assert.deepEqual(currentStep(course(GROUND.slice(0,3))),{step:'scout',sinceMs:5000});
  // The pilot: the gauge is the last checkpoint's populations.
  const pilot=courseView(course(GROUND.slice(0,4),[mark(30000,88,.072)]),32000);
  assert.deepEqual(shown(pilot).slice(1),['侦察|7 s|done','先导|已用 20 s|current','选态||pending','正式批次||pending','结果||pending']);
  assert.equal(pilot.gauge.fraction,SEMICLASSICAL_POPULATION_TARGET/.072);assert.equal(pilot.gauge.words,'布居误差 7.2%，目标 5%');
  assert.equal(pilot.words,'先导 · 时间步长 1.67 μs · 检查点 1');
  // The selection: the pilot, the scout and the model done, nothing of them to come again.
  const select=courseView(course(GROUND.slice(0,5),[mark(30000,88,.072)]),45500);
  assert.deepEqual(shown(select),['建模型|5 s|done','侦察|7 s|done','先导|28 s|done','选态|已用 5 s|current','正式批次||pending','结果||pending']);
  assert.match(select.nodes[2].title,/^先导：.*为选态提供原子走过的路径（开始后 12 s 开始，用时 28 s；时间步长 1.67 μs）$/);
  assert.match(select.nodes[3].title,/^选态：沿先导原子走过的路径.*32 态参照.*总代价最小.*沿用之前的选择（开始后 40 s 开始，已用 5 s）$/);
  assert.equal(select.nodes.some(n=>/三点|模型误差/.test(n.title)),false,'no step says the old selection\'s words');
  assert.equal(select.words,'检查点 1');
  assert.deepEqual(currentStep(course(GROUND.slice(0,5))),{step:'select',sinceMs:40000});
  // The product: the batch's own gauge, its largest total over target.
  const product=courseView(course(GROUND,[mark(30000,88,.072),mark(80000,176,.061)]),110500);
  assert.deepEqual(shown(product),['建模型|5 s|done','侦察|7 s|done','先导|28 s|done','选态|12 s|done','正式批次|已用 58 s|current','结果||pending']);
  assert.equal(product.gauge.fraction,1/1.2);assert.equal(product.gauge.words,'误差最大的一项是目标的 1.2 倍');
  assert.equal(product.words,'第 2 批：176 → 352 条轨迹 · 时间步长 1.67 μs · 检查点 2');
  assert.match(product.nodes[4].title,/^正式批次：.*（开始后 52 s 开始，已用 58 s；时间步长 1.67 μs）$/);
  assert.deepEqual(currentStep(course(GROUND)),{step:'product',sinceMs:52000});
});

test('a new selection that chooses another model: its model, scout and pilot again are nodes of their own, each with its own start, time and rungs',()=>{
  const model=courseView(course(OTHER.slice(0,6)),53500);
  assert.deepEqual(shown(model),['建模型|5 s|done','侦察|7 s|done','先导|28 s|done','选态|12 s|done','建模型|已用 1 s|current','正式批次||pending','结果||pending']);
  assert.deepEqual(model.nodes.map(n=>n.key),['model','scout','pilot','select','model-2','product','result']);
  assert.match(model.nodes[4].title,/^建模型：.*（开始后 52 s 开始，已用 1 s）$/);
  // The step in progress began with its own stages, not at the step's first time: its clock turns at its own second.
  assert.deepEqual(currentStep(course(OTHER.slice(0,6))),{step:'model',sinceMs:52400});
  assert.equal(clockChangesInMs(53500,currentStep(course(OTHER.slice(0,6))).sinceMs),900);
  const scout=courseView(course(OTHER.slice(0,8)),60500);
  assert.deepEqual(shown(scout).slice(4),['建模型|2 s|done','侦察|已用 5 s|current','正式批次||pending','结果||pending']);
  assert.equal(scout.words,'侦察 · 时间步长 1.67 → 0.833 μs · 尚无检查点','the second scout\'s rungs alone');
  assert.match(scout.nodes[1].title,/（开始后 5 s 开始，用时 7 s；时间步长 3.33 → 1.67 μs）$/);
  assert.match(scout.nodes[5].title,/^侦察：.*（开始后 55 s 开始，已用 5 s；时间步长 1.67 → 0.833 μs）$/);
  assert.deepEqual(currentStep(course(OTHER.slice(0,8))),{step:'scout',sinceMs:55000});
  const product=courseView(course(OTHER,[mark(30000,88,.072),mark(90000,88,.05,.8333)]),100500);
  assert.deepEqual(shown(product),['建模型|5 s|done','侦察|7 s|done','先导|28 s|done','选态|12 s|done','建模型|2 s|done','侦察|11 s|done','先导|24 s|done','正式批次|已用 10 s|current','结果||pending']);
  assert.deepEqual(product.nodes.map(n=>n.key),['model','scout','pilot','select','model-2','scout-2','pilot-2','product','result']);
  assert.match(product.nodes[6].title,/^先导：.*（开始后 1 min 6 s 开始，用时 24 s；时间步长 0.833 μs）$/);
  assert.equal(product.words,'第 1 批：88 → 176 条轨迹 · 时间步长 0.833 μs · 检查点 2');
  assert.equal(product.gauge.fraction,1/1.5);
  assert.deepEqual(currentStep(course(OTHER)),{step:'product',sinceMs:90000});
});

test('the selection kept for the scene: the selection first, at once, then the model, the scout, the pilot and the product',()=>{
  const select=courseView(course(KEPT.slice(0,1)),200);
  assert.deepEqual(shown(select),['选态|已用 0 s|current','正式批次||pending','结果||pending']);
  assert.match(select.nodes[0].title,/^选态：.*（开始后 0 s 开始，已用 0 s）$/);
  assert.deepEqual(currentStep(course(KEPT.slice(0,1))),{step:'select',sinceMs:0});
  const model=courseView(course(KEPT.slice(0,2)),1500);
  assert.deepEqual(shown(model),['选态|0 s|done','建模型|已用 1 s|current','侦察||pending','先导||pending','正式批次||pending','结果||pending']);
  assert.match(model.nodes[0].title,/（开始后 0 s 开始，用时 0 s）$/);
  assert.deepEqual(currentStep(course(KEPT.slice(0,2))),{step:'model',sinceMs:400});
  const product=courseView(course(KEPT,[mark(40000,88,.061)]),45000);
  assert.deepEqual(shown(product),['选态|0 s|done','建模型|2 s|done','侦察|7 s|done','先导|30 s|done','正式批次|已用 5 s|current','结果||pending']);
  // A batch with no worst: the gauge is the last checkpoint's populations.
  assert.equal(product.gauge.words,'布居误差 6.1%，目标 5%');
  assert.equal(product.words,'第 1 批：88 → 176 条轨迹 · 时间步长 1.67 μs · 检查点 1');
  assert.deepEqual(currentStep(course(KEPT)),{step:'product',sinceMs:40000});
});

test('the course\'s clock: whole seconds, then minutes and seconds, then hours and minutes; it changes at the next whole second of the step in progress',()=>{
  assert.deepEqual([0,999,59999,60000,3599999,3723000].map(ms=>clockWords(ms)),['0 s','0 s','59 s','1 min 0 s','59 min 59 s','1 h 2 min']);
  assert.deepEqual([0,1500,2999].map(ms=>clockChangesInMs(ms)),[1000,500,1]);
  assert.equal(clockChangesInMs(5300,5000),700,'the step began at 5 s: its clock turns at 6 s');
});
