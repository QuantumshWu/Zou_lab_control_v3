import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {formatNumber,formatElapsed,stateLabel,idLabel,labelStateIds,signedInteger,MINUS} from '../dist/js/atom-workbench/src/format.js';
import {solverText,steadyStatusText} from '../dist/js/atom-workbench/src/solver-words.js';
import {stateTree} from '../dist/js/atom-workbench/src/choice-trees.js';
import {POPULATION_GROUPS} from '../dist/js/atom-workbench/src/population-selection.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze} from '../dist/js/atom-workbench/src/physics.js';

const NNBSP='\u202f';
test('numbers use significant digits, the true minus and SI grouping from five integer digits',()=>{
  assert.equal(formatNumber(384.228115203271),'384.228');
  assert.equal(formatNumber(3,{digits:4}),'3');
  assert.equal(formatNumber(-37.552,{places:3}),`${MINUS}37.552`);
  assert.equal(formatNumber(-7123689.76,{places:3}),`${MINUS}7${NNBSP}123${NNBSP}689.760`);
  assert.equal(formatNumber(8083.4,{digits:5}),'8083.4');
  assert.equal(formatNumber(6.66e-16,{digits:2}),'6.7 × 10⁻¹⁶');
  assert.equal(formatNumber(2.5e9,{digits:3}),'2.5 × 10⁹');
  assert.equal(formatNumber(-0.0001,{places:3}),'0.000','a value rounded to zero carries no sign');
  assert.equal(formatNumber(0),'0');
  assert.equal(formatNumber(4.2,{signed:true}),'+4.2');
  assert.equal(formatNumber(NaN),'—');assert.equal(formatNumber(Infinity),'∞');
  assert.equal(formatElapsed(741.62),'741.6 ms');assert.equal(formatElapsed(8083.4),'8.083 s');
});
test('every state is spelled one way, in trees, groups and labels',()=>{
  assert.equal(stateLabel({manifold:'ground',F:2,m:-1}),`5S₁/₂ · F=2 · m=${MINUS}1`);
  assert.equal(stateLabel({manifold:'D2',F:3,m:1}),'5P₃/₂ · F′=3 · m=+1');
  assert.equal(stateLabel({manifold:'D1',F:1,m:0}),'5P₁/₂ · F″=1 · m=0');
  assert.equal(idLabel('ground-1--1'),`5S₁/₂ · F=1 · m=${MINUS}1`);
  assert.equal(idLabel('reference-1',()=>'中间态'),'中间态');
  assert.equal(labelStateIds('最近允许跃迁 ground-2--2 → D2-3-3，失谐'),`最近允许跃迁 5S₁/₂ · F=2 · m=${MINUS}2 → 5P₃/₂ · F′=3 · m=+3，失谐`);
  assert.equal(signedInteger(-2),`${MINUS}2`);
  const a=analyze(defaultScene()),tree=stateTree([...a.ground,...a.D1,...a.D2]);
  const ground=tree[0].children[0];
  assert.equal(ground.heading,'F=1');
  assert.deepEqual(ground.leaves.map(l=>l.label),[`m=${MINUS}1`,'m=0','m=+1']);
  assert.equal(ground.leaves[0].text,stateLabel({manifold:'ground',F:1,m:-1}));
  assert.equal(POPULATION_GROUPS.find(g=>g.id==='F:D2:3').label,'5P₃/₂ · F′=3');
});
test('solver statuses and known sentences are shown in Chinese; unknown text is kept',()=>{
  assert.equal(steadyStatusText('time-average'),'长期时间平均');
  assert.equal(steadyStatusText('not-converged'),'未收敛');
  assert.equal(solverText('an unforeseen native message'),'an unforeseen native message');
});
test('the method sentences the native adapter writes are shown in Chinese',()=>{
  const adapter=readFileSync(new URL('../rust-solver/src/lattice_core_adapter.rs',import.meta.url),'utf8');
  for(const name of ['FINITE_METHOD','MEAN_METHOD','STATIC_METHOD']){
    const sentence=new RegExp(`const ${name}:&str="([^"]+)";`).exec(adapter)?.[1];
    assert.ok(sentence,name);
    assert.match(solverText(sentence),/^完整 32 态/,name);
  }
});
