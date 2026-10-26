import test from 'node:test';
import assert from 'node:assert/strict';
import {DynamicsClient,normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {POPULATION_GROUPS} from '../dist/js/atom-workbench/src/population-selection.js';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {nativeResult} from './native-result.mjs';

const response=data=>({...data,assumptions:['test assumption'],
  service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA],binarySha256:'binary',buildSha256:'build'}});
function fixture(){
  const scene=defaultScene();scene.dynamics.samples=3;scene.dynamics.durationUs=.2;
  const states=buildFullMasterModel(scene,scene.dynamics).states,index=states.findIndex(s=>s.id===scene.dynamics.initialStateId);
  const data=response(nativeResult({states,initialIndex:index,timesUs:[0,.1,.2],error:.003,charge:.0001,mean:false}));
  return {scene,data,index,states};
}
function meanFixture(f){
  const mean=response(nativeResult({states:f.states,initialIndex:f.index,timesUs:[0]}));
  Object.assign(mean.diagnostics,{asymptoticTraceError:2e-7,asymptoticPositivityError:1e-5});
  return mean;
}

test('a finite result accepts complete all-group accuracy without claiming a density',()=>{
  const f=fixture();f.data.diagnostics.rawDensityPositivityError=.2;
  const result=normalizeDynamicsResult(f.data,f.scene);
  assert.equal(result.populationAccuracy.requestedTolerance,.005);
  assert.equal(result.populationAccuracy.includesProjectionCharge,true);
  assert.equal(result.populationAccuracy.scope.groups.length,POPULATION_GROUPS.length);
  assert.equal(result.steady.density,undefined);
  f.data.populationAccuracy.scope.groups.pop();
  assert.equal(result.populationAccuracy.scope.groups.length,POPULATION_GROUPS.length);
});

test('a finite trajectory cannot bypass complete accuracy and projection-charge qualification',()=>{
  const f=fixture();
  for(const change of [
    d=>delete d.populationAccuracy,d=>d.populationAccuracy=null,
    d=>d.populationAccuracy.status='unresolved',d=>d.populationAccuracy.includesProjectionCharge=false,
    d=>delete d.populationAccuracy.projectionCharge,d=>d.populationAccuracy.projectionCharge=.004,
    d=>d.populationAccuracy.projectionCharge=-1,d=>d.populationAccuracy.estimatedMaximumError=.006,
    d=>d.populationAccuracy.estimatedMaximumError=NaN,d=>d.populationAccuracy.requestedTolerance=0,
    d=>d.populationAccuracy.requestedTolerance=.021,d=>d.populationAccuracy.requestedTolerance='0.005',
    d=>d.populationAccuracy.scope.states=31,d=>d.populationAccuracy.scope.samples=2,
    d=>d.populationAccuracy.scope.groups.pop(),d=>d.populationAccuracy.scope.groups[0]=d.populationAccuracy.scope.groups[1],
    d=>d.populationAccuracy.scope.groups[0]='unknown',
  ]){
    const data=structuredClone(f.data);change(data);
    assert.throws(()=>normalizeDynamicsResult(data,f.scene),/不完整或无效/);
  }
});

test('a finite result retains finite, probability, trace, initial and no-density gates',()=>{
  const f=fixture();
  for(const change of [
    d=>d.densityReal=null,d=>d.densityImag=[],d=>d.populations[1][0]=NaN,
    d=>{d.populations[1][f.index]+=1e-7;d.populations[1][(f.index+1)%32]=-1e-7;},
    d=>d.populations[1].fill(0),d=>d.diagnostics.traceError=6e-6,
    d=>delete d.diagnostics.initialStateError,d=>d.diagnostics.initialStateError=2e-10,
    d=>{d.populations[0][f.index]=.99;d.populations[0][(f.index+1)%32]=.01;},
    d=>d.timesUs[0]=.01,d=>{d.timesUs=[0];d.populations=[d.populations[0]];d.populationAccuracy.scope.samples=1;},
  ]){
    const data=structuredClone(f.data);change(data);
    assert.throws(()=>normalizeDynamicsResult(data,f.scene),/不完整或无效/);
  }
});

test('mean trace, positivity and complex-density gates hold in mean and finite results alike',()=>{
  const f=fixture(),mean=meanFixture(f);
  assert.equal(normalizeDynamicsResult(mean,f.scene).steady.converged,true);
  for(const [key,value] of [['asymptoticPositivityError',undefined],['asymptoticPositivityError',.0002],['asymptoticPositivityError',NaN],['asymptoticTraceError',6e-6]]){
    const bad=structuredClone(mean);bad.diagnostics[key]=value;
    assert.throws(()=>normalizeDynamicsResult(bad,f.scene),/长期密度数值验证信息/);
  }
  // As the writer merges: the mean's asymptotic fields; finite diagnostics win.
  const combined={...f.data,asymptoticStatus:mean.asymptoticStatus,asymptoticReason:mean.asymptoticReason,asymptoticPopulations:mean.asymptoticPopulations,
    asymptoticDensityReal:mean.asymptoticDensityReal,asymptoticDensityImag:mean.asymptoticDensityImag,
    diagnostics:{...mean.diagnostics,...f.data.diagnostics}};
  assert.equal(normalizeDynamicsResult(combined,f.scene).steady.converged,true);
  const missing=structuredClone(combined);delete missing.diagnostics.asymptoticPositivityError;
  assert.throws(()=>normalizeDynamicsResult(missing,f.scene),/长期密度/);
  const invalid=structuredClone(combined);invalid.asymptoticDensityImag[f.index*32+f.index]=.01;
  assert.throws(()=>normalizeDynamicsResult(invalid,f.scene),/长期复密度/);
});

test('transport binds population accuracy to the actual requested tolerance and initial state',async t=>{
  const f=fixture();f.scene.dynamics.populationTolerance=.001;
  let reply=f.data,calls=0;
  const client=new DynamicsClient({clientId:'population-contract',debounceMs:0,fetch:async(_url,init)=>{
    calls++;const body=JSON.parse(init.body);
    return new Response(JSON.stringify({...reply,request:body.request}),{headers:{'Content-Type':'application/json'}});
  }});t.after(()=>client.dispose());
  await assert.rejects(client.solve(f.scene,f.scene.dynamics),/误差目标与本次请求不匹配/);
  reply=structuredClone(f.data);reply.populationAccuracy.requestedTolerance=.001;reply.populationAccuracy.estimatedMaximumError=.0006;
  assert.equal((await client.solve(f.scene,f.scene.dynamics)).populationAccuracy.requestedTolerance,.001);
  const other=f.states.find(state=>state.id!==f.scene.dynamics.initialStateId).id;
  await assert.rejects(client.solve(f.scene,{...f.scene.dynamics,initialStateId:other}),/布居初态/);
  assert.equal(calls,3);
});
