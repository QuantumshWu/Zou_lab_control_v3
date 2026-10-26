import test from 'node:test';
import assert from 'node:assert/strict';
import {DynamicsClient,normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {POPULATION_GROUPS} from '../dist/js/atom-workbench/src/population-selection.js';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';

function fixture(){
  const scene=defaultScene();scene.dynamics.samples=3;scene.dynamics.durationUs=.2;
  const states=buildFullMasterModel(scene,scene.dynamics).states,index=states.findIndex(s=>s.id===scene.dynamics.initialStateId);
  const row=Array(32).fill(0);row[index]=1;
  const data={modelSchema:FULL_MODEL_SCHEMA,protocol:NATIVE_PROTOCOL,states,timesUs:[0,.1,.2],populations:[row.slice(),row.slice(),row.slice()],
    asymptoticPopulations:null,asymptoticStatus:'not-requested',
    service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA],binarySha256:'binary',buildSha256:'build'},
    diagnostics:{validationKind:'population',traceError:0,initialStateError:0,elapsedMs:1},
    populationAccuracy:{status:'estimated-converged',requestedTolerance:.005,estimatedMaximumError:.003,
      projectionCharge:.0001,includesProjectionCharge:true,scope:{states:32,samples:3,groups:POPULATION_GROUPS.map(g=>g.id)}}};
  return {scene,data,index,states};
}
function meanFixture(f){
  const mean=structuredClone(f.data);delete mean.populationAccuracy;
  mean.timesUs=[0];mean.populations=[mean.populations[0]];mean.asymptoticPopulations=mean.populations[0].slice();mean.asymptoticStatus='stationary';
  mean.asymptoticDensityReal=Array(32*32).fill(0);mean.asymptoticDensityReal[f.index*32+f.index]=1;mean.asymptoticDensityImag=Array(32*32).fill(0);
  mean.diagnostics={traceError:2e-7,positivityError:1e-5,elapsedMs:1,method:'mean test'};
  return mean;
}

test('population-only result accepts complete all-group accuracy without claiming density PSD',()=>{
  const f=fixture();f.data.diagnostics.rawDensityPositivityError=.2;
  const result=normalizeDynamicsResult(f.data,f.scene);
  assert.equal(result.populationAccuracy.requestedTolerance,.005);
  assert.equal(result.populationAccuracy.includesProjectionCharge,true);
  assert.equal(result.populationAccuracy.scope.groups.length,POPULATION_GROUPS.length);
  assert.equal(Object.hasOwn(result.validation,'positivityError'),false);
  f.data.populationAccuracy.scope.groups.pop();
  assert.equal(result.populationAccuracy.scope.groups.length,POPULATION_GROUPS.length);
});

test('population marker cannot bypass complete accuracy and projection-charge qualification',()=>{
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

test('population-only result retains finite, probability, trace, initial and no-density gates',()=>{
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

test('legacy PSD and separate mean-density validation remain required',()=>{
  const f=fixture(),mean=meanFixture(f);
  assert.equal(normalizeDynamicsResult(mean,f.scene).validation.positivityError,1e-5);
  for(const positivity of [undefined,.0002,NaN]){
    const bad=structuredClone(mean);bad.diagnostics.positivityError=positivity;
    assert.throws(()=>normalizeDynamicsResult(bad,f.scene),/数值验证信息/);
  }
  const combined={...f.data,asymptoticStatus:mean.asymptoticStatus,asymptoticPopulations:mean.asymptoticPopulations,
    asymptoticDensityReal:mean.asymptoticDensityReal,asymptoticDensityImag:mean.asymptoticDensityImag,
    diagnostics:{...f.data.diagnostics,asymptoticTraceError:2e-7,asymptoticPositivityError:1e-5}};
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
