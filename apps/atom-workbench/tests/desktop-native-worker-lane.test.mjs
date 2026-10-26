import test from 'node:test';
import assert from 'node:assert/strict';
import {createModelWorker} from '../desktop-native/ui/model-worker.mjs';
import {IpcNativeLane} from '../desktop-native/ui/ipc-lane.mjs';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {FULL_MODEL_SCHEMA,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {nativeResult} from './native-result.mjs';
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
const capabilities={id:0,protocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA]};
const warm={id:-1,protocol:NATIVE_PROTOCOL,ready:true};

test('IPC lane preserves handshake, cancellation and stale-generation isolation',async()=>{
  const commands=[],finished=[];const owner={binarySha256:'binary',build:{sha256:'build'},startupTimeoutMs:1000,workerIdleMs:1000,sendNative:c=>commands.push(c),finish:(job,error,value)=>{job.settled=true;finished.push({job,error,value});}};
  const lane=new IpcNativeLane(owner,'trajectory');
  const first={nativeId:3,payload:{},settled:false};lane.enqueue(first);await tick();
  assert.deepEqual(commands[0],{op:'open',lane:'trajectory',generation:1});
  lane.receive({kind:'opened',generation:1,binarySha256:'binary',pid:123});assert.equal(commands.at(-1).message.op,'capabilities');
  lane.receive({kind:'message',generation:1,message:capabilities});assert.equal(commands.at(-1).message.id,-1);
  lane.receive({kind:'message',generation:1,message:warm});await tick();assert.equal(commands.at(-1).message.id,3);
  const cancellation=Error('cancel');lane.cancel(first,cancellation);assert.equal(commands.at(-1).op,'close');assert.equal(finished[0].error,cancellation);
  lane.receive({kind:'closed',generation:1});
  const second={nativeId:4,payload:{},settled:false};lane.enqueue(second);await tick();assert.equal(commands.at(-1).generation,2);
  lane.receive({kind:'message',generation:1,message:{id:3,result:{}}});assert.equal(finished.length,1);
  lane.receive({kind:'opened',generation:2,binarySha256:'binary',pid:124});lane.receive({kind:'message',generation:2,message:capabilities});lane.receive({kind:'message',generation:2,message:warm});await tick();
  lane.receive({kind:'message',generation:2,message:{id:999,result:{}}});assert.equal(finished[1].error.code,'invalid-native-response');assert.equal(commands.at(-1).op,'close');
  lane.receive({kind:'closed',generation:2});await lane.close();
});

const fakeResult=({model,options})=>nativeResult({states:model.states,initialIndex:model.initialIndex,tolerance:options.populationTolerance,mean:options.includeAsymptotic,
  timesUs:options.steadyOnly?[0]:Array.from({length:options.samples},(_,i)=>options.durationUs*i/(options.samples-1))});

test('Worker sends the identical complete model, finite jobs to the trajectory lane and means to the steady lane',async()=>{
  const events=[],requests=[];let worker;
  const host=command=>{
    const event=message=>queueMicrotask(()=>void worker.receive({kind:'native',event:{lane:command.lane,generation:command.generation,...message}}));
    if(command.op==='open')event({kind:'opened',pid:100+requests.length,binarySha256:'binary'});
    if(command.op==='close')event({kind:'closed'});
    if(command.op==='write'){
      const request=command.message;
      if(request.id===0)event({kind:'message',message:capabilities});
      else if(request.id===-1)event({kind:'message',message:warm});
      else{requests.push({...request,lane:command.lane});event({kind:'message',message:{id:request.id,result:fakeResult(request)}});}
    }
  };
  worker=createModelWorker(message=>{events.push(message);if(message.kind==='native')host(message.command);});
  await worker.receive({kind:'solve',channel:'invalid',request:{}});
  await worker.receive({kind:'native',event:{kind:'configuration',binarySha256:'binary',buildSha256:'build',defaults:{ATOM_STEADY_BLAS_THREADS:'1',ATOM_TRAJECTORY_BLAS_THREADS:'8'}}});
  await tick();assert.equal(events.find(e=>e.channel==='invalid').status,400);
  const scene=defaultScene();scene.lasers[1].enabled=false;scene.lasers.push({...scene.lasers[0],id:'second',detuningMHz:100});
  const options={durationUs:0.1,samples:3,populationTolerance:.003731,steadyOnly:false,includeAsymptotic:true};const request={scene,modelSchema:FULL_MODEL_SCHEMA,stream:true,request:{clientId:'worker-test',sequence:1},options};
  await worker.receive({kind:'solve',channel:'valid',request});
  for(let i=0;i<50&&!events.some(e=>e.channel==='valid'&&e.event?.type==='result');i++)await tick();
  const messages=events.filter(e=>e.channel==='valid');assert.deepEqual(messages.map(e=>e.event?.type),['result']);
  // One native job: the finite trajectory carries its own inline mean.
  assert.equal(requests.length,1);assert.deepEqual(requests.map(r=>r.lane),['trajectory']);
  assert.equal(requests[0].options.includeAsymptotic,true);assert.equal(requests[0].progress,true);
  const model=buildFullMasterModel(scene,{...scene.dynamics,...options});
  for(const sent of requests){
    assert.equal(sent.options.populationTolerance,.003731);
    assert.equal(sent.modelSchema,FULL_MODEL_SCHEMA);
    // Exactly the fields the worker reads; e.g. no frameFrequenciesMHz: the
    // native solver returns the laboratory-frame mean itself.
    assert.deepEqual(Object.keys(sent).filter(key=>key!=='lane').sort(),['id','model','modelSchema','options','progress']);
    assert.deepEqual(Object.keys(sent.model).sort(),['collapse','dimension','drives','frequencyCoordinates','hImag','hReal','initialIndex','states']);
    for(const key of ['dimension','hReal','hImag','collapse','initialIndex','drives','frequencyCoordinates'])assert.deepEqual(sent.model[key],model[key],key);
    assert.deepEqual(sent.model.states,model.states.map(s=>s.id));
  }
  assert.equal(messages.at(-1).event.data.service.lane,'trajectory');
  assert.equal(messages.at(-1).event.data.service.asymptoticCacheHit,false);
  // The Worker's response is exactly what the UI reader accepts.
  const finite=normalizeDynamicsResult(messages.at(-1).event.data,scene);
  assert.equal(finite.timesUs.length,3);assert.equal(finite.steady.converged,true);assert.deepEqual(finite.assumptions,model.assumptions);
  const legacy=structuredClone(scene);delete legacy.dynamics.populationTolerance;
  await worker.receive({kind:'solve',channel:'legacy',request:{scene:legacy,modelSchema:FULL_MODEL_SCHEMA,
    request:{clientId:'worker-test',sequence:2},options:{durationUs:.1,samples:3,steadyOnly:true,includeAsymptotic:true}}});
  for(let i=0;i<50&&!events.some(e=>e.channel==='legacy'&&e.kind==='response');i++)await tick();
  assert.equal(requests.at(-1).options.populationTolerance,.005);assert.equal(requests.at(-1).lane,'steady');
  const mean=normalizeDynamicsResult(events.find(e=>e.channel==='legacy'&&e.kind==='response').data,legacy);
  assert.equal(mean.timesUs.length,1);assert.equal(mean.steady.converged,true);
  const count=requests.length;
  await worker.receive({kind:'solve',channel:'invalid-tolerance',request:{scene,modelSchema:FULL_MODEL_SCHEMA,
    request:{clientId:'worker-test',sequence:3},options:{populationTolerance:.021}}});
  await tick();assert.equal(events.find(e=>e.channel==='invalid-tolerance').status,400);
  assert.equal(requests.length,count,'invalid tolerance never reaches a native lane');
  await worker.receive({kind:'shutdown'});assert.equal(events.at(-1).kind,'shutdown-complete');
});

test('Worker reports native startup failure and cancels requests queued before configuration',async()=>{
  const events=[],worker=createModelWorker(message=>events.push(message));
  await worker.receive({kind:'solve',channel:'canceled',request:{}});await worker.receive({kind:'cancel',channel:'canceled'});
  await worker.receive({kind:'solve',channel:'pending',request:{}});
  await worker.receive({kind:'native',event:{kind:'fatal',error:'Missing native solver'}});
  assert.equal(events.length,1);assert.equal(events[0].channel,'pending');assert.equal(events[0].status,503);assert.match(events[0].data.error,/Missing native/);
  await worker.receive({kind:'solve',channel:'retry',request:{}});assert.equal(events.at(-1).status,503);
  await worker.receive({kind:'shutdown'});
});
