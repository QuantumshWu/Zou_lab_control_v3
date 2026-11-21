// A running motion solve's course (solve-progress.ts): the opening, the checkpoints in the order they came with their
// errors, the run going on, and the gauge of the last checkpoint's error against the target.
import test from 'node:test';
import assert from 'node:assert/strict';
import {SHOWN_CHECKPOINTS,clockChangesInMs,clockWords,courseMark,courseView} from '../dist/js/atom-workbench/src/solve-progress.js';
import {SEMICLASSICAL_POPULATION_TARGET} from '../dist/js/atom-workbench/src/solver-types.js';

const mark=(atMs,trajectories,error,stepUs=.7125)=>({atMs,trajectories,stepUs,error});
const keys=view=>view.nodes.map(n=>n.key),states=view=>view.nodes.map(n=>n.state);

test('a checkpoint\'s mark: when it came, its trajectories, its step and its largest population error; none without motion',()=>{
  const result={motion:{trajectories:264,timeStepUs:.7125},populationAccuracy:{estimatedMaximumError:.072}};
  assert.deepEqual(courseMark(result,1234),{atMs:1234,trajectories:264,stepUs:.7125,error:.072});
  assert.equal(courseMark({populationAccuracy:{estimatedMaximumError:.01}},5),undefined,'a result without motion has no course');
});

test('before its first checkpoint the course is in its opening, its clock running; no gauge',()=>{
  const view=courseView({startedAt:1000,marks:[]},43500);
  assert.deepEqual(keys(view),['opening','result']);assert.deepEqual(states(view),['current','pending']);
  assert.equal(view.nodes[0].detail,'已用 42 s');assert.match(view.nodes[0].title,/建模型、选态.*侦察与先导/);
  assert.equal(view.gauge,undefined);assert.equal(view.words,'尚无检查点');
});

test('with checkpoints: the opening done at the first, each checkpoint its error, the run in progress, the gauge the target over the last error',()=>{
  const marks=[mark(42000,88,.12),mark(102000,176,.081),mark(150000,264,.072)];
  const view=courseView({startedAt:0,marks},192500);
  assert.deepEqual(keys(view),['opening','checkpoint-0','checkpoint-1','checkpoint-2','running','result']);
  assert.deepEqual(states(view),['done','done','done','done','current','pending']);
  assert.deepEqual(view.nodes.map(n=>[n.label,n.detail]),[['开头','42 s'],['检查点 1','12%'],['检查点 2','8.1%'],['检查点 3','7.2%'],['计算中','已用 3 min 12 s'],['结果','']]);
  assert.equal(view.nodes[3].title,'检查点 3（开始后 2 min 30 s）：264 条轨迹，时间步长 0.713 μs；布居的估计误差最大 7.2%，目标 5%');
  assert.equal(view.gauge.fraction,SEMICLASSICAL_POPULATION_TARGET/.072);assert.equal(view.gauge.words,'布居误差 7.2%，目标 5%');assert.match(view.gauge.title,/是目标的 1\.4 倍/);
  assert.equal(view.words,'264 条轨迹 · 时间步长 0.713 μs');
  // Within the target the gauge is full; the run goes on until the temperature is within its allowance too.
  const within=courseView({startedAt:0,marks:[...marks,mark(200000,352,.041)]},201000);
  assert.equal(within.gauge.fraction,1);assert.match(within.gauge.title,/已在目标内/);
});

test(`beyond ${SHOWN_CHECKPOINTS} checkpoints the first and the last ones stand by themselves, those between them as one`,()=>{
  const marks=Array.from({length:7},(_,i)=>mark(60000*(i+1),88*(i+1),.2/(i+1)));
  const view=courseView({startedAt:0,marks},430000);
  assert.deepEqual(keys(view),['opening','checkpoint-0','hidden','checkpoint-5','checkpoint-6','running','result']);
  const hidden=view.nodes[2];assert.deepEqual([hidden.label,hidden.detail,hidden.title],['…','4 个','检查点 2–5']);
  assert.deepEqual(view.nodes.slice(3,5).map(n=>n.label),['检查点 6','检查点 7']);
  assert.deepEqual(keys(courseView({startedAt:0,marks:marks.slice(0,SHOWN_CHECKPOINTS)},1)).filter(k=>k.startsWith('checkpoint')).length,SHOWN_CHECKPOINTS,'as many as it shows: each by itself');
});

test('the course\'s clock: whole seconds, then minutes and seconds, then hours and minutes; it changes at the next whole second',()=>{
  assert.deepEqual([0,999,59999,60000,3599999,3723000].map(clockWords),['0 s','0 s','59 s','1 min 0 s','59 min 59 s','1 h 2 min']);
  assert.deepEqual([0,1500,2999].map(clockChangesInMs),[1000,500,1]);
});
