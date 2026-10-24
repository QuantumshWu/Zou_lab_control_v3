// Transport and validation tests. Controlled replies exercise races only;
// numerical correctness is tested separately against the actual Rust worker.
import assert from 'node:assert/strict';
import test from 'node:test';
import {setTimeout as pause} from 'node:timers/promises';
import {DynamicsClient,normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildMasterModel} from '../dist/js/atom-workbench/src/master-model.js';
import {EFFECTIVE_MODEL_SCHEMA as ATOMIC_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';

function fixture(scene=defaultScene(),settings=scene.dynamics){
  const states=buildMasterModel(scene,settings).states,initial=states.findIndex(s=>s.id===settings.initialStateId);
  const row=Array.from({length:32},(_,i)=>i===initial?1:0);
  return {modelSchema:ATOMIC_MODEL_SCHEMA,protocol:NATIVE_PROTOCOL,service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[ATOMIC_MODEL_SCHEMA]},
    states,timesUs:[0,settings.durationUs/2,settings.durationUs],populations:[row.slice(),row.slice(),row.slice()],
    asymptoticPopulations:row.slice(),asymptoticStatus:'stationary',diagnostics:{traceError:0,positivityError:0,elapsedMs:1}};
}
function scenario(){const s=defaultScene();s.dynamics.samples=3;s.dynamics.durationUs=.2;return s;}

function streamReply(call){let controller;const body=new ReadableStream({start(c){controller=c;}});call.resolve(new Response(body,{headers:{'Content-Type':'application/x-ndjson'}}));const encoder=new TextEncoder();return {send:event=>controller.enqueue(encoder.encode(JSON.stringify(event)+'\n')),close:()=>controller.close()};}
function meanFixture(scene,request){const result=fixture(scene);return {...result,request,timesUs:[0],populations:[result.populations[0]]};}
test('verified mean arrives before the trajectory and remains available when evolution fails',async t=>{
 const mock=transport(),progress=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'partial-mean'});t.after(()=>client.dispose());
 const scene=scenario();let finished=false;const pending=client.solve(scene,scene.dynamics,{onProgress:value=>progress.push(value)});const rejected=assert.rejects(pending,/unstable evolution/).then(()=>finished=true);
 await waitCalls(mock,1);assert.equal(mock.calls[0].body.stream,true);const stream=streamReply(mock.calls[0]);
 stream.send({type:'asymptotic',data:meanFixture(scene,mock.calls[0].body.request)});await pause(15);
 assert.equal(progress.length,1);assert.equal(progress[0].steady.converged,true);assert.equal(progress[0].timesUs.length,1);assert.equal(finished,false);
 stream.send({type:'error',status:422,data:{error:'unstable evolution',code:'numerical-failure'}});stream.close();await rejected;
 assert.equal(progress[0].steady.converged,true);assert.equal(mock.calls.length,1);
});
test('streamed mean and final trajectory share request identity and preserve actual samples',async t=>{
 const mock=transport(),progress=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'stream-success'});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics,{onProgress:value=>progress.push(value)});await waitCalls(mock,1);const stream=streamReply(mock.calls[0]),request=mock.calls[0].body.request;
 stream.send({type:'asymptotic',data:meanFixture(scene,request)});stream.send({type:'result',data:{...fixture(scene),request}});stream.close();
 const result=await pending;assert.equal(progress.length,1);assert.equal(result.timesUs.length,3);assert.equal(result.timesUs.at(-1),scene.dynamics.durationUs);
});
test('late streamed progress cannot overwrite a superseding scene',async t=>{
 const mock=transport(),progress=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'stream-race'});t.after(()=>client.dispose());
 const scene=scenario(),old=client.solve(scene,scene.dynamics,{onProgress:value=>progress.push(value)}).catch(e=>e);await waitCalls(mock,1);const stream=streamReply(mock.calls[0]);
 const next=client.solve(scene,scene.dynamics);assert.equal((await old).name,'AbortError');await waitCalls(mock,2);
 stream.send({type:'asymptotic',data:meanFixture(scene,mock.calls[0].body.request)});stream.send({type:'result',data:{...fixture(scene),request:mock.calls[0].body.request}});stream.close();
 reply(mock.calls[1],fixture(scene));await next;await pause(15);assert.equal(progress.length,0);
});

test('transient connection failure retries once with fresh identity and records the immutable scene',async t=>{
 const mock=transport(),records=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'recover',onFailure:r=>records.push(r)});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics);await waitCalls(mock,1);scene.fields[0].thetaDeg=99;
 mock.calls[0].reject(new TypeError('Network disconnected'));await pause(230);await waitCalls(mock,2);
 assert.ok(mock.calls[1].body.request.sequence>mock.calls[0].body.request.sequence);
 assert.equal(mock.calls[1].body.scene.fields[0].thetaDeg,0);assert.equal(records[0].scene.fields[0].thetaDeg,0);assert.equal(records[0].willRetry,true);
 reply(mock.calls[1],fixture(mock.calls[1].body.scene));assert.equal((await pending).steady.converged,true);
});

test('temporary service failures have a bounded retry; supersession cancels the queued retry',async t=>{
 const mock=transport(),records=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'bounded',onFailure:r=>records.push(r)});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics),rejected=assert.rejects(pending,/temporarily unavailable/);
 await waitCalls(mock,1);mock.calls[0].resolve(new Response(JSON.stringify({error:'temporarily unavailable',code:'solver-unavailable'}),{status:503}));
 await pause(230);await waitCalls(mock,2);mock.calls[1].resolve(new Response(JSON.stringify({error:'temporarily unavailable',code:'solver-unavailable'}),{status:503}));await rejected;
 await pause(230);assert.equal(mock.calls.length,2);assert.deepEqual(records.map(r=>r.willRetry),[true,false]);
 const obsolete=client.solve(scene,scene.dynamics).catch(e=>e);await waitCalls(mock,3);mock.calls[2].reject(new TypeError('temporary'));await pause(20);
 const latest=client.solve(scene,scene.dynamics);assert.equal((await obsolete).name,'AbortError');await waitCalls(mock,4);reply(mock.calls[3],fixture(scene));await latest;
 await pause(230);assert.equal(mock.calls.length,4);
});

test('numerical failures and unresolved means are recorded without blind retries or invented populations',async t=>{
 const mock=transport(),records=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'numerical',onFailure:r=>records.push(r)});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics),rejected=assert.rejects(pending,/numerical failure/);await waitCalls(mock,1);
 mock.calls[0].resolve(new Response(JSON.stringify({error:'numerical failure',code:'numerical-failure',service:{binarySha256:'test-binary'}}),{status:422}));await rejected;
 assert.equal(records[0].willRetry,false);assert.equal(records[0].response.service.binarySha256,'test-binary');
 const next=client.solve(scene,scene.dynamics);await waitCalls(mock,2);const data=fixture(scene);data.asymptoticStatus='unresolved-slow-modes';data.asymptoticPopulations=null;reply(mock.calls[1],data);
 assert.equal((await next).steady.converged,false);assert.equal(records[1].response.asymptoticStatus,'unresolved-slow-modes');await pause(230);assert.equal(mock.calls.length,2);
});
function transport(){const calls=[];return {calls,fetch:async(url,init)=>new Promise((resolve,reject)=>calls.push({url,init,resolve,reject,body:JSON.parse(init.body)}))};}
const reply=(call,data)=>call.resolve(new Response(JSON.stringify({...data,request:call.body.request}),{status:200,headers:{'Content-Type':'application/json'}}));
async function waitCalls(mock,count){for(let i=0;i<100&&mock.calls.length<count;i++)await pause(2);assert.equal(mock.calls.length,count);}

test('malformed native payloads never become fabricated populations',()=>{
  const scene=scenario(),valid=fixture(scene);
  const changes=[
    data=>{data.populations.pop();}, data=>{data.populations[1].pop();}, data=>{data.populations[1][0]=NaN;},
    data=>{data.populations[0].fill(0);}, data=>{data.populations[0][0]=-.2;data.populations[0][1]+=.2;},
    data=>{data.states[1].id=data.states[0].id;},data=>{data.states[0].energyMHz=Infinity;},
    data=>{data.states[0].id='arbitrary-reference';},data=>{data.states[0].F=99;},
    data=>{data.timesUs[1]=data.timesUs[0];},data=>{data.asymptoticPopulations.pop();},
    data=>{data.asymptoticPopulations[0]=Infinity;},data=>{data.asymptoticPopulations.fill(0);},
    data=>{data.asymptoticPopulations=null;},data=>{data.diagnostics.traceError=NaN;},data=>{delete data.diagnostics.positivityError;},
    data=>{data.diagnostics.positivityError=.1;},data=>{data.diagnostics.residual=NaN;},data=>{data.warnings={text:'not an array'};},
  ];
  for(const change of changes){const data=structuredClone(valid);change(data);assert.throws(()=>normalizeDynamicsResult(data,scene),/不完整或无效/);}
  for(const data of [null,undefined,[],{}])assert.throws(()=>normalizeDynamicsResult(data,scene),/不完整或无效/);
});

test('initial-dependent mean is converged, retains its reason and is marked initial dependent',()=>{
  const scene=scenario(),data=fixture(scene);data.asymptoticStatus='initial-dependent-mean';data.diagnostics.asymptoticReason='Specified initial mixture retained.';
  const result=normalizeDynamicsResult(data,scene);assert.equal(result.steady.converged,true);assert.equal(result.steady.initialDependent,true);assert.equal(result.steady.reason,data.diagnostics.asymptoticReason);
});

test('old atomic models and unnegotiated or missing service capabilities are rejected',async t=>{
  const scene=scenario();
  for(const change of [r=>delete r.modelSchema,r=>r.modelSchema='rb87-dipole/1',r=>delete r.protocol,r=>r.protocol='atom-lindblad-jsonl/1']){
    const r=fixture(scene);change(r);assert.throws(()=>normalizeDynamicsResult(r,scene),/版本不兼容/);
  }
  const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'protocol'});t.after(()=>client.dispose());
  for(const [i,change]of [r=>delete r.service,r=>r.service.nativeProtocol='atom-lindblad-jsonl/1',r=>r.service.capabilities=[],r=>r.service.protocol='atom-workbench-http/1'].entries()){
    const pending=client.solve(scene,scene.dynamics),rejected=assert.rejects(pending,/版本不兼容/);await waitCalls(mock,i+1);const r=fixture(scene);change(r);reply(mock.calls[i],r);await rejected;
  }
});

test('normalization reports the larger finite-time or separately cached mean convergence error',()=>{
  const scene=scenario(),data=fixture(scene);data.diagnostics.harmonicConvergenceError=8e-9;data.diagnostics.asymptoticHarmonicConvergenceError=1e-10;
  assert.equal(normalizeDynamicsResult(data,scene).validation.convergenceMaxDelta,8e-9);
  data.diagnostics.harmonicConvergenceError=1e-12;assert.equal(normalizeDynamicsResult(data,scene).validation.convergenceMaxDelta,1e-10);
  data.diagnostics.asymptoticFrameConvergenceError=9e-9;assert.equal(normalizeDynamicsResult(data,scene).validation.convergenceMaxDelta,9e-9);
  data.diagnostics.frameUnitarityBound=Infinity;assert.throws(()=>normalizeDynamicsResult(data,scene),/数值诊断/);
});

test('steady residual uses the mean proof without reporting an unrelated finite-trajectory residual',()=>{
  const scene=scenario(),data=fixture(scene);Object.assign(data.diagnostics,{residual:8e-8,populationResidual:9e-9,asymptoticResidual:3e-13,asymptoticPopulationResidual:2e-13});
  assert.equal(normalizeDynamicsResult(data,scene).steady.residual,2e-13);
  data.diagnostics.asymptoticPopulationResidual=null;assert.equal(normalizeDynamicsResult(data,scene).steady.residual,3e-13);
  data.diagnostics.asymptoticResidual=null;assert.equal(normalizeDynamicsResult(data,scene).steady.residual,9e-9);
  data.diagnostics.populationResidual=null;assert.equal(normalizeDynamicsResult(data,scene).steady.residual,8e-8);
  data.diagnostics.asymptoticPopulationResidual=0;assert.equal(normalizeDynamicsResult(data,scene).steady.residual,0);
  data.diagnostics.asymptoticPopulationResidual=NaN;assert.throws(()=>normalizeDynamicsResult(data,scene),/数值诊断/);
  data.diagnostics.asymptoticPopulationResidual=null;data.diagnostics.asymptoticResidual=Infinity;assert.throws(()=>normalizeDynamicsResult(data,scene),/数值诊断/);
});

test('rapid edits debounce to only the last immutable scene snapshot',async t=>{
  const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:20,clientId:'debounce'});t.after(()=>client.dispose());
  const scene=scenario();scene.dynamics.quality='verified';const first=client.solve(scene,scene.dynamics).catch(error=>error);
  scene.lasers[0].id='latest-laser';const second=client.solve(scene,scene.dynamics);
  scene.lasers[0].id='mutated-after-submit';scene.dynamics.durationUs=9;
  assert.equal((await first).name,'AbortError');await waitCalls(mock,1);
  const call=mock.calls[0];assert.equal(call.body.scene.lasers[0].id,'latest-laser');assert.equal(call.body.options.durationUs,.2);assert.equal(Object.hasOwn(call.body.options,'quality'),false);
  reply(call,fixture(call.body.scene,call.body.options));const result=await second;assert.equal(result.activeLaserIds[0],'latest-laser');assert.equal(result.timesUs.at(-1),.2);
});

test('in-flight stale fetch is aborted; late success or failure cannot settle the new promise',async t=>{
  const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'race'});t.after(()=>client.dispose());
  const scene=scenario(),first=client.solve(scene,scene.dynamics).catch(error=>error);await waitCalls(mock,1);
  const second=client.solve(scene,scene.dynamics);assert.equal(mock.calls[0].init.signal.aborted,true);assert.equal((await first).name,'AbortError');await waitCalls(mock,2);
  reply(mock.calls[1],fixture(scene));const result=await second;assert.equal(result.timesUs.length,3);
  const obsolete=fixture(scene);delete obsolete.modelSchema;delete obsolete.protocol;delete obsolete.service;
  reply(mock.calls[0],obsolete);await pause(0);
  const third=client.solve(scene,scene.dynamics).catch(error=>error);await waitCalls(mock,3);
  const fourth=client.solve(scene,scene.dynamics);await waitCalls(mock,4);mock.calls[2].reject(new TypeError('late network failure'));
  assert.equal((await third).name,'AbortError');reply(mock.calls[3],fixture(scene));assert.equal((await fourth).timesUs.length,3);
});

test('cancel and dispose clear both scheduled and active work without blocking a later solve',async t=>{
  const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:5,clientId:'cancel'});t.after(()=>client.dispose());const scene=scenario();
  const first=client.solve(scene,scene.dynamics).catch(error=>error);client.cancel();assert.equal((await first).name,'AbortError');await pause(10);assert.equal(mock.calls.length,0);
  const second=client.solve(scene,scene.dynamics).catch(error=>error);await waitCalls(mock,1);client.cancel();assert.equal((await second).name,'AbortError');assert.equal(mock.calls[0].init.signal.aborted,true);
  const third=client.solve(scene,scene.dynamics);await waitCalls(mock,2);reply(mock.calls[1],fixture(scene));await third;
  const last=client.solve(scene,scene.dynamics).catch(error=>error);client.dispose();assert.equal((await last).name,'AbortError');await assert.rejects(client.solve(scene,scene.dynamics),{name:'AbortError'});
});

test('mismatched transport identity and missing requested samples are rejected',async t=>{
  const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'identity'});t.after(()=>client.dispose());const scene=scenario();
  const first=client.solve(scene,scene.dynamics);const rejected=assert.rejects(first,/请求不匹配/);await waitCalls(mock,1);
  mock.calls[0].resolve(new Response(JSON.stringify({...fixture(scene),request:{clientId:'wrong',sequence:1}})));await rejected;
  const second=client.solve(scene,scene.dynamics);const missing=assert.rejects(second,/采样与请求不匹配/);await waitCalls(mock,2);
  const data=fixture(scene);data.timesUs=[0];data.populations=[data.populations[0]];reply(mock.calls[1],data);await missing;
  const third=client.solve(scene,scene.dynamics),noIdentity=assert.rejects(third,/请求不匹配/);await waitCalls(mock,3);
  mock.calls[2].resolve(new Response(JSON.stringify(fixture(scene))));await noIdentity;
});
