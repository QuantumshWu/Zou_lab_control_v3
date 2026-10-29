// Transport and validation tests. Controlled replies exercise races only;
// numerical correctness is tested separately against the actual Rust worker.
import assert from 'node:assert/strict';
import test from 'node:test';
import {setTimeout as pause} from 'node:timers/promises';
import {DynamicsClient,normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {POPULATION_TOLERANCE,defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {HORIZONS,entryAt,nativeResult} from './native-result.mjs';
import {CurrentResults} from '../dist/js/atom-workbench/src/current-results.js';
import {horizonRecords,keptCheckpoint} from '../dist/js/atom-workbench/src/solve-plan.js';

/** The HTTP response for this scene: a native result plus transport metadata. */
function fixture(scene=defaultScene(),settings=scene.dynamics,timesUs=[0,settings.durationUs/2,settings.durationUs]){
  const states=buildFullMasterModel(scene,settings).states,initialIndex=states.findIndex(s=>s.id===settings.initialStateId);
  return {...nativeResult({states,initialIndex,timesUs,tolerance:POPULATION_TOLERANCE}),assumptions:['test assumption'],
    service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA]}};
}
/** A trajectory alone (includeAsymptotic false): no mean. */
const trajectoryOnly=data=>{delete data.asymptoticHorizons;return data;};
function scenario(){const s=defaultScene();s.dynamics.samples=3;s.dynamics.durationUs=.2;return s;}

function streamReply(call){let controller;const body=new ReadableStream({start(c){controller=c;}});call.resolve(new Response(body,{headers:{'Content-Type':'application/x-ndjson'}}));const encoder=new TextEncoder();return {send:event=>controller.enqueue(encoder.encode(JSON.stringify(event)+'\n')),close:()=>controller.close()};}
function meanFixture(scene,request){return {...fixture(scene,scene.dynamics,[0]),request};}
/** A streamed checkpoint (ui-design.md §6.16): a full result so far, flagged as progress, with its per-sample errors. */
function checkpoint(scene,request,sampleErrors){const settings=scene.dynamics,timesUs=[0,settings.durationUs/2,settings.durationUs],states=buildFullMasterModel(scene,settings).states;
 return {...nativeResult({states,initialIndex:states.findIndex(s=>s.id===settings.initialStateId),timesUs,sampleErrors}),assumptions:['test assumption'],
   service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA]},progress:true,request};}
test('a checkpoint arrives before the final result and stays available when evolution fails',async t=>{
 const mock=transport(),progress=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'partial-mean'});t.after(()=>client.dispose());
 const scene=scenario();let finished=false;const pending=client.solve(scene,scene.dynamics,{onProgress:value=>progress.push(value)});const rejected=assert.rejects(pending,/unstable evolution/).then(()=>finished=true);
 await waitCalls(mock,1);assert.equal(mock.calls[0].body.stream,true);const stream=streamReply(mock.calls[0]);
 stream.send({type:'checkpoint',data:checkpoint(scene,mock.calls[0].body.request,[0,.012,.003])});await pause(15);
 assert.equal(progress.length,1);assert.equal(progress[0].timesUs.length,3);assert.equal(progress[0].populationAccuracy.status,'estimated-partial');assert.deepEqual(progress[0].populationAccuracy.sampleErrors,[0,.012,.003]);assert.equal(finished,false);
 stream.send({type:'error',status:422,data:{error:'unstable evolution',code:'numerical-failure'}});stream.close();await rejected;
 assert.equal(progress[0].steady.converged,true);assert.equal(mock.calls.length,1);
});
test('two checkpoints, then the final result: each delivered in turn, the final one without a progress flag',async t=>{
 const mock=transport(),progress=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'stream-success'});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics,{onProgress:value=>progress.push(value)});await waitCalls(mock,1);const stream=streamReply(mock.calls[0]),request=mock.calls[0].body.request;
 stream.send({type:'checkpoint',data:checkpoint(scene,request,[0,.02,.009])});stream.send({type:'checkpoint',data:checkpoint(scene,request,[0,.007,.001])});
 const final=checkpoint(scene,request,[0,.002,.001]);delete final.progress;stream.send({type:'result',data:final});stream.close();
 const result=await pending;assert.equal(progress.length,2);assert.deepEqual(progress.map(p=>p.populationAccuracy.estimatedMaximumError),[.02,.007]);
 assert.equal(result.populationAccuracy.status,'estimated-converged');assert.equal(result.timesUs.length,3);assert.equal(result.timesUs.at(-1),scene.dynamics.durationUs);
});
test('two checkpoints, then 停止: the solve ends and the last checkpoint is what remains, marked partial',async t=>{
 const mock=transport(),progress=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'stream-stop'});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics,{onProgress:value=>progress.push(value)}).catch(e=>e);await waitCalls(mock,1);const stream=streamReply(mock.calls[0]),request=mock.calls[0].body.request;
 stream.send({type:'checkpoint',data:checkpoint(scene,request,[0,.03,.004])});stream.send({type:'checkpoint',data:checkpoint(scene,request,[0,.011,.002])});await pause(15);
 client.cancel();assert.equal((await pending).name,'AbortError');
 const last=progress.at(-1);assert.equal(progress.length,2);assert.equal(last.populationAccuracy.status,'estimated-partial');assert.deepEqual(last.populationAccuracy.sampleErrors,[0,.011,.002]);
 // The page drew neither; on 停止 the last one becomes its result: the trajectory and its mean at every T.
 const trajectories=new CurrentResults(),means=new CurrentResults(),epochs=[trajectories.activate('trajectory inputs'),means.activate('mean inputs')];
 assert.equal(trajectories.get('trajectory inputs','trajectory'),undefined);
 const kept=keptCheckpoint(last,'full',()=>false);assert.deepEqual(kept,{trajectory:true,horizons:[1e5,1e6]});
 trajectories.set('trajectory inputs',epochs[0],'trajectory',last);
 for(const mean of horizonRecords({result:last,wallMs:1,horizon:1e6}))if(kept.horizons.includes(mean.horizon))means.set('mean inputs',epochs[1],mean.horizon,mean);
 assert.equal(trajectories.get('trajectory inputs','trajectory'),last);assert.deepEqual([1e5,1e6].map(T=>means.get('mean inputs',T)?.result.steady.horizonUs),[1e5,1e6]);
});
test('the stream speaks checkpoints and one final result only',async t=>{
 const scene=scenario();
 for(const [label,events] of [
   ['a retired asymptotic event',request=>[{type:'asymptotic',data:meanFixture(scene,request)}]],
   ['a checkpoint without its progress flag',request=>[{type:'checkpoint',data:{...checkpoint(scene,request,[0,.001,.001]),progress:undefined}}]],
   ['a final result flagged as progress',request=>[{type:'result',data:checkpoint(scene,request,[0,.001,.001])}]],
   ['a checkpoint on another time grid',request=>[{type:'checkpoint',data:{...meanFixture(scene,request),progress:true}}]],
 ]){
   const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'stream-contract'});
   const pending=client.solve(scene,scene.dynamics,{onProgress:()=>{}});await waitCalls(mock,1);const stream=streamReply(mock.calls[0]);
   for(const event of events(mock.calls[0].body.request))stream.send(event);stream.close();
   await assert.rejects(pending,/未知求解进度|不匹配|进度标记/,label);client.dispose();
 }
});
test('a mean request streams checkpoints of its single t = 0 row',async t=>{
 const mock=transport(),progress=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'stream-mean'});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics,{steadyOnly:true,onProgress:value=>progress.push(value)});await waitCalls(mock,1);const stream=streamReply(mock.calls[0]),request=mock.calls[0].body.request;
 stream.send({type:'checkpoint',data:{...meanFixture(scene,request),progress:true}});stream.send({type:'result',data:meanFixture(scene,request)});stream.close();
 const result=await pending;assert.equal(progress.length,1);assert.equal(progress[0].timesUs.length,1);assert.equal(result.steady.converged,true);
});
test('late streamed progress cannot overwrite a superseding scene',async t=>{
 const mock=transport(),progress=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'stream-race'});t.after(()=>client.dispose());
 const scene=scenario(),old=client.solve(scene,scene.dynamics,{onProgress:value=>progress.push(value)}).catch(e=>e);await waitCalls(mock,1);const stream=streamReply(mock.calls[0]);
 const next=client.solve(scene,scene.dynamics);assert.equal((await old).name,'AbortError');await waitCalls(mock,2);
 stream.send({type:'checkpoint',data:checkpoint(scene,mock.calls[0].body.request,[0,.001,.001])});stream.send({type:'result',data:{...fixture(scene),request:mock.calls[0].body.request}});stream.close();
 reply(mock.calls[1],fixture(scene));await next;await pause(15);assert.equal(progress.length,0);
});

test('transient connection failure retries once with fresh identity and records the immutable scene',async t=>{
 const mock=transport(),records=[],client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'recover',onFailure:r=>records.push(r)});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics);await waitCalls(mock,1);scene.fields[0].direction.thetaDeg=99;
 mock.calls[0].reject(new TypeError('Network disconnected'));await pause(230);await waitCalls(mock,2);
 assert.ok(mock.calls[1].body.request.sequence>mock.calls[0].body.request.sequence);
 assert.equal(mock.calls[1].body.scene.fields[0].direction.thetaDeg,0);assert.equal(records[0].scene.fields[0].direction.thetaDeg,0);assert.equal(records[0].willRetry,true);
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
 const next=client.solve(scene,scene.dynamics);await waitCalls(mock,2);const data=fixture(scene);
 data.asymptoticHorizons=HORIZONS.map(horizonUs=>({horizonUs,status:'not-converged',reason:'slow modes unresolved',populations:null,densityReal:null,densityImag:null,estimatedError:null,withinTolerance:false}));reply(mock.calls[1],data);
 assert.equal((await next).steady.converged,false);assert.deepEqual(records[1].response.means,HORIZONS.map(horizonUs=>({horizonUs,status:'not-converged'})));await pause(230);assert.equal(mock.calls.length,2);
});
function transport(){const calls=[];return {calls,fetch:async(url,init)=>new Promise((resolve,reject)=>calls.push({url,init,resolve,reject,body:JSON.parse(init.body)}))};}
const reply=(call,data)=>call.resolve(new Response(JSON.stringify({...data,request:call.body.request}),{status:200,headers:{'Content-Type':'application/json'}}));
async function waitCalls(mock,count){for(let i=0;i<100&&mock.calls.length<count;i++)await pause(2);assert.equal(mock.calls.length,count);}

test('every POST asks for the fixed population accuracy, whatever the settings carry',async t=>{
 const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'population-tolerance'});t.after(()=>client.dispose());
 for(const [index,stray] of [undefined,.003731,.02,'0.005'].entries()){
  const scene=scenario();if(stray!==undefined)scene.dynamics.populationTolerance=stray;
  const pending=client.solve(scene,scene.dynamics);await waitCalls(mock,index+1);
  assert.equal(mock.calls[index].body.options.populationTolerance,.005);
  reply(mock.calls[index],fixture(scene));await pending;
 }
});

test('malformed native payloads never become fabricated populations',()=>{
  const scene=scenario(),valid=fixture(scene);
  const changes=[
    data=>{data.populations.pop();}, data=>{data.populations[1].pop();}, data=>{data.populations[1][0]=NaN;},
    data=>{data.populations[0].fill(0);}, data=>{data.populations[0][0]=-.2;data.populations[0][1]+=.2;},
    data=>{data.states[1].id=data.states[0].id;},data=>{data.states[0].energyMHz=Infinity;},
    data=>{data.states[0].id='arbitrary-reference';},data=>{data.states[0].F=99;},
    data=>{data.timesUs[1]=data.timesUs[0];},data=>{entryAt(data).populations.pop();},
    data=>{entryAt(data).populations[0]=Infinity;},data=>{entryAt(data).populations.fill(0);},
    data=>{entryAt(data).populations=null;},data=>{delete entryAt(data).densityReal;},
    data=>{entryAt(data).status='not-converged';},data=>{entryAt(data).status='periodic-mean';},data=>{entryAt(data).reason=7;},
    data=>{data.diagnostics.traceError=NaN;},data=>{delete data.diagnostics.asymptoticPositivityError;},data=>{data.diagnostics.asymptoticPositivityError=.1;},
    data=>{data.diagnostics.asymptoticTraceError=6e-6;},data=>{delete data.diagnostics.asymptoticMethod;},
    data=>{delete data.diagnostics.method;},data=>{delete data.populationAccuracy;},data=>{delete data.diagnostics.elapsedMs;},
    data=>{data.assumptions={text:'not an array'};},data=>{delete data.assumptions;},
  ];
  for(const change of changes){const data=structuredClone(valid);change(data);assert.throws(()=>normalizeDynamicsResult(data,scene),/不完整或无效/);}
  for(const data of [null,undefined,[],{}])assert.throws(()=>normalizeDynamicsResult(data,scene),/不完整或无效/);
});

test('a computed mean is the phase average at T, and nothing else: the statuses of the t → ∞ limit are refused',()=>{
  const scene=scenario(),data=fixture(scene);for(const entry of data.asymptoticHorizons)entry.reason='its reason';
  const result=normalizeDynamicsResult(data,scene);
  assert.equal(result.steady.converged,true);assert.equal(result.steady.status,'time-average');assert.equal(result.steady.reason,'its reason');
  for(const status of ['stationary','population-stationary','initial-dependent','initial-dependent-time-average','initial-dependent-mean','nonunique-initial-dependent','periodic-mean']){
    const data=fixture(scene);entryAt(data).status=status;assert.throws(()=>normalizeDynamicsResult(data,scene),/不完整或无效/,status);
  }
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

test('both methods each come from one contract key; no steady residual or conserved count is read',()=>{
  const scene=scenario(),data=fixture(scene);
  const result=normalizeDynamicsResult(data,scene);
  assert.equal(result.method,'test trajectory');assert.equal(result.steady.method,'test mean');
  const mean=normalizeDynamicsResult(meanFixture(scene),scene);assert.equal(mean.method,undefined);assert.equal(mean.steady.method,'test mean');
  const finiteOnly=normalizeDynamicsResult(trajectoryOnly(fixture(scene)),scene);assert.equal(finiteOnly.steady.converged,false);
  // The t → ∞ fields are not part of the contract: present or not, the mean reads the same.
  const extra=fixture(scene);Object.assign(extra.diagnostics,{residual:NaN,conservedQuantities:3});
  const read=normalizeDynamicsResult(extra,scene);assert.equal(Object.hasOwn(read.steady,'residual'),false);assert.equal(Object.hasOwn(read.steady,'conservedQuantities'),false);
});

test('a trajectory alone: includeAsymptotic false is sent, and its answer carries no mean',async t=>{
 const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'trajectory-alone'});t.after(()=>client.dispose());
 const scene=scenario(),pending=client.solve(scene,scene.dynamics,{includeAsymptotic:false});await waitCalls(mock,1);
 assert.equal(mock.calls[0].body.options.includeAsymptotic,false);assert.equal(mock.calls[0].body.options.steadyOnly,false);
 reply(mock.calls[0],trajectoryOnly(fixture(scene)));const result=await pending;
 assert.equal(result.steady.status,'not-requested');assert.equal(result.horizonMeans,undefined);assert.equal(result.timesUs.length,3);
 // A mean nobody asked for, or none where one was asked for, is not this request's answer.
 const extra=client.solve(scene,scene.dynamics,{includeAsymptotic:false}),refused=assert.rejects(extra,/长期布居与本次请求不符/);await waitCalls(mock,2);reply(mock.calls[1],fixture(scene));await refused;
 const missing=client.solve(scene,scene.dynamics),absent=assert.rejects(missing,/长期布居与本次请求不符/);await waitCalls(mock,3);reply(mock.calls[2],trajectoryOnly(fixture(scene)));await absent;
 await assert.rejects(client.solve(scene,scene.dynamics,{steadyOnly:true,includeAsymptotic:false}),/mean-only solve asks for its mean/);assert.equal(mock.calls.length,3);
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
  reply(mock.calls[1],fixture(scene,scene.dynamics,[0]));await missing;
  const third=client.solve(scene,scene.dynamics),noIdentity=assert.rejects(third,/请求不匹配/);await waitCalls(mock,3);
  mock.calls[2].resolve(new Response(JSON.stringify(fixture(scene))));await noIdentity;
});
