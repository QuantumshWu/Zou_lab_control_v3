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
  assert.equal(solverText('8 conserved quantities: the mean depends on the initial state, which the lattice core does not serve'),
    '8 个守恒量：长期平均依赖初态，求解器不计算这种长期平均。');
  assert.equal(solverText('The mean at the horizon is estimated at 6.2e-3; the tolerance was not relaxed'),'观察时长处的长期平均误差估计为 0.0062；误差目标未放宽。');
  assert.match(solverText('Lattice steady state not accepted up to near order 6 (estimated error 7.5e-3)'),/^长期平均到近阶 6 仍未达到误差目标（误差估计 .+）；误差目标未放宽。$/);
  assert.match(solverText('Finite-time estimate 5.1e-3 not below the tolerance with 120 columns'),/^时间演化的误差估计 .+ 在 120 列时仍未低于误差目标；误差目标未放宽。$/);
  assert.equal(solverText('Error equation truncation did not stabilise in 40 augmentation steps'),'截断误差方程在 40 步增广内未稳定；误差目标未放宽。');
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
