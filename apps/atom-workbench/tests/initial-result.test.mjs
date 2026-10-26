import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {solverText} from '../dist/js/atom-workbench/src/solver-words.js';
import {nativeResult} from './native-result.mjs';
import {initialAnchor,initialSpec,initialSummary,setInitialSpec,startingSpec} from '../dist/js/atom-workbench/src/initial-state.js';
import {analyze} from '../dist/js/atom-workbench/src/physics.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';

// A mixed initial state's first row is its diagonal, not a one-hot vector at initialStateId.
function mixed(spec){
  const scene=defaultScene();scene.dynamics.samples=3;scene.dynamics.durationUs=.2;scene.dynamics.initialState=spec;
  const model=buildFullMasterModel(scene,scene.dynamics),states=model.states,p=model.initialPopulations;
  const data=nativeResult({states,initialIndex:model.initialIndex,timesUs:[0,.1,.2],mean:false});
  data.populations=data.populations.map(()=>p.slice());
  Object.assign(data,{assumptions:[],initialState:model.initialState,initialPopulations:p.slice()});
  return {scene,data,p,states};
}

test('a mixed initial state is checked against its own populations',()=>{
  for(const spec of [{kind:'unpolarized',manifold:'ground'},{kind:'unpolarized',manifold:'F1'},{kind:'thermal',temperatureK:1e-4},{kind:'weights',weights:{'ground-2-0':3,'D2-3-0':1}}]){
    const f=mixed(spec),result=normalizeDynamicsResult(f.data,f.scene);
    assert.deepEqual(f.states.map(s=>result.populations[s.id][0]),f.p,spec.kind);
  }
  const f=mixed({kind:'unpolarized',manifold:'ground'});
  // A single state where a mixture was asked for is not the requested state.
  const one=structuredClone(f.data);one.populations[0]=f.p.map((_,i)=>+(i===f.states.findIndex(s=>s.id==='ground-2-0')));
  assert.throws(()=>normalizeDynamicsResult(one,f.scene),/布居初态/);
  const other=structuredClone(f.data);other.initialState={kind:'unpolarized',manifold:'F2'};
  assert.throws(()=>normalizeDynamicsResult(other,f.scene),/请求初态/);
  const sent=structuredClone(f.data);sent.initialPopulations=sent.initialPopulations.map((p,i)=>i===0?p+1e-9:i===1?p-1e-9:p);
  assert.throws(()=>normalizeDynamicsResult(sent,f.scene),/请求初态/);
  // An id still means that one state.
  assert.throws(()=>normalizeDynamicsResult(f.data,f.scene,'ground-2-0'),/请求初态|布居初态/);
});

test('the solver\'s formatted failure reasons read in Chinese',()=>{
  assert.equal(solverText('2 conserved quantities together with a relaxation of 1.5e3 us cannot be resolved by the near-zero pole; the tolerance was not relaxed'),
    '2 个守恒量与 1500 μs 的弛豫无法由近零极点分开；误差目标未放宽。');
  assert.match(solverText('The resolvent solve at pole 1e-4/us stalled at relative residual 3.2e-6 (requested 1e-8); the tolerance was not relaxed'),/^近零极点 .+\/μs 处的预解求解停滞在相对残差 .+（要求 .+）；误差目标未放宽。$/);
  assert.equal(solverText('Nonfinite population error series'),'布居误差级数出现非有限值。');
  assert.equal(solverText('Initial-dependent long-time mean did not reach the tolerance; the tolerance was not relaxed'),'依赖初态的长期平均未达到误差目标；误差目标未放宽。');
  assert.equal(solverText('an unknown sentence'),'an unknown sentence');
});

test('one initial-state description: a single state clears a mixture, a mixture keeps its anchor',()=>{
  const scene=defaultScene(),a=analyze(scene),levels=[...a.ground,...a.D1,...a.D2];
  setInitialSpec(scene,{kind:'unpolarized',manifold:'F1'},levels);
  assert.deepEqual(initialSpec(scene),{kind:'unpolarized',manifold:'F1'});
  assert.match(scene.dynamics.initialStateId,/^ground-1-/,'the anchor is a state of the mixture');
  // A single state chosen afterwards is the initial state: no loaded mixture overrides it.
  setInitialSpec(scene,{kind:'state',id:'ground-2-1'},levels);
  assert.deepEqual(initialSpec(scene),{kind:'state',id:'ground-2-1'});assert.equal(scene.dynamics.initialState,undefined);
  assert.equal(scene.initialF,2);assert.equal(scene.initialM,1);
  assert.equal(initialAnchor({kind:'weights',weights:{'ground-2-0':1,'D2-3-0':3}},levels),'D2-3-0');
  assert.deepEqual(startingSpec('weights',{kind:'state',id:'ground-2-1'},levels),{kind:'weights',weights:{'ground-2-1':1}});
  assert.deepEqual(startingSpec('state',{kind:'weights',weights:{'ground-1-0':1,'ground-2-2':2}},levels),{kind:'state',id:'ground-2-2'});
  assert.equal(initialSummary({kind:'thermal',temperatureK:2.5e-4}),'热分布 · T = 250 μK');
  assert.equal(initialSummary({kind:'unpolarized',manifold:'ground'}),'未极化 · 全部基态');
  assert.throws(()=>setInitialSpec(scene,{kind:'thermal',temperatureK:0},levels));
  assert.deepEqual(initialSpec(scene),{kind:'state',id:'ground-2-1'},'a refused description leaves the old one');
});

test('a mixed initial state survives the scene file; a single one is stored as initialStateId',()=>{
  const scene=defaultScene(),a=analyze(scene),levels=[...a.ground,...a.D1,...a.D2];
  setInitialSpec(scene,{kind:'thermal',temperatureK:3e-4},levels);
  assert.deepEqual(initialSpec(validateScene(JSON.parse(JSON.stringify(scene)))),{kind:'thermal',temperatureK:3e-4});
  setInitialSpec(scene,{kind:'state',id:'ground-1-1'},levels);
  const back=validateScene(JSON.parse(JSON.stringify(scene)));
  assert.equal(back.dynamics.initialState,undefined);assert.equal(back.dynamics.initialStateId,'ground-1-1');
});
