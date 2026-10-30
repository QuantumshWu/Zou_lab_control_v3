import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {formatNumber,formatElapsed,stateLabel,idLabel,labelStateIds,signedInteger,MINUS} from '../dist/js/atom-workbench/src/format.js';
import {solverText} from '../dist/js/atom-workbench/src/solver-words.js';
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
test('known solver sentences are shown in Chinese; unknown text is kept',()=>{
  assert.equal(solverText('an unforeseen native message'),'an unforeseen native message');
});
// The reasons the solver formats, read from its source: every format
// string with sample values for its placeholders, in order.
const pipelineSource=readFileSync(new URL('../rust-solver/population-core/src/lattice/pipeline.rs',import.meta.url),'utf8');
const adapterSource=readFileSync(new URL('../rust-solver/src/lattice_core_adapter.rs',import.meta.url),'utf8');
const formatStrings=source=>[...source.matchAll(/format!\("((?:[^"\\]|\\.)*)"/g)].map(m=>m[1]);
const formatted=(source,start,values)=>{
  const format=formatStrings(source).find(f=>f.startsWith(start));assert.ok(format,start);
  let i=0;return format.replace(/\{[A-Za-z_]*(?::[^}]*)?\}/g,()=>String(values[i++]));
};
const chinese=(english,name)=>{const text=solverText(english);
  assert.notEqual(text,english,name);assert.match(text,/[\u4e00-\u9fff]/,name);assert.doesNotMatch(text,/\b(the|is|of|above|after)\b/,name);return text;};
test('the stop and checkpoint reasons the solver formats are shown in Chinese',()=>{
  const whys=[['the error bound stopped improving',[2,3,48]],['the basis memory was spent',[30]],['the solve budget',[48]]]
    .map(([start,values])=>formatted(pipelineSource,start,values));
  const samples=formatted(pipelineSource,'; {} of {} samples above it',[5,1000,'0.0200','0.1001']);
  const means=formatted(pipelineSource,'; {means} mean output(s) above it',[1]);
  for(const why of whys){
    for(const tail of [['',''],[samples,''],['',means],[samples,means]]){
      chinese(formatted(pipelineSource,'Stopped because',[why,24,'1.89e-2','5e-3',...tail]),why);
    }
  }
  chinese(formatted(adapterSource,'Checkpoint at',[8,'1.20e-2','5e-3']),'checkpoint');
  chinese(formatted(adapterSource,"The mean's error bound is",['3.61e0']),'uninformative mean');
  // A request stopped before its first solve: nothing computed, the reason alone.
  chinese(formatted(pipelineSource,'The lattice of this request has',[150,'5.3','5']),'memory guard');
  chinese(formatted(pipelineSource,'An output time is at most',['1e7','1.5e7']),'an output time beyond 10 s');
  // A grid the solver does not solve: refused with the reason.
  assert.match(chinese(formatted(adapterSource,'The trajectory duration is limited',['15']),'too long'),/10 s.*15 s/);
  assert.match(chinese(formatted(adapterSource,'A trajectory has 2 to',[2000,2001]),'too many samples'),/2 至 2000.*2001/);
  chinese('The trajectory duration must be a positive finite time','no duration');
  chinese('meanHorizonUs must be 1e5 or 1e6 us (100 ms or 1 s)','another T');
});
test('every sentence the pipeline and the adapter format is shown in Chinese (a new one fails here)',()=>{
  const sentences=[...formatStrings(pipelineSource),...formatStrings(adapterSource)].filter(f=>/^[A-Z][a-z]/.test(f));
  assert.ok(sentences.length>=5,`${sentences.length} sentences`);
  for(const format of sentences)chinese(format.replace(/\{[A-Za-z_]*(?::[^}]*)?\}/g,'1'),format);
});
test('every mean reason the native adapter writes is shown in Chinese, within the tolerance and outside it',()=>{
  const outside=/const OUTSIDE:&str="([^"]+)";/.exec(adapterSource)?.[1];assert.ok(outside,'OUTSIDE');
  const reasons=[...adapterSource.matchAll(/"((?:Phase average|State at)[^"]*\.)"/g)].map(m=>m[1]);
  assert.equal(reasons.length,2,`${reasons.length} reasons: the driven and the undriven mean at T`);
  for(const reason of reasons){chinese(reason,reason);assert.match(chinese(reason+outside,reason),/未收敛/,reason);}
});
test('the method sentences the native adapter writes are shown in Chinese',()=>{
  const adapter=readFileSync(new URL('../rust-solver/src/lattice_core_adapter.rs',import.meta.url),'utf8');
  for(const name of ['METHOD']){
    const sentence=new RegExp(`const ${name}:&str="([^"]+)";`).exec(adapter)?.[1];
    assert.ok(sentence,name);
    assert.match(solverText(sentence),/^完整 32 态/,name);
  }
});
