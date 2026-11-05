import test from 'node:test';
import assert from 'node:assert/strict';
import {createModelWorker} from '../desktop-native/ui/model-worker.mjs';
import {IpcNativeLane} from '../desktop-native/ui/ipc-lane.mjs';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {POPULATION_TOLERANCE,defaultScene} from '../dist/js/atom-workbench/src/types.js';
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

const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const started=()=>{const commands=[],finished=[],owner={binarySha256:'binary',build:{sha256:'build'},startupTimeoutMs:1000,workerIdleMs:1,sendNative:c=>commands.push(c),finish:(job,error,value)=>{job.settled=true;finished.push({job,error,value});}};
  return {commands,finished,lane:new IpcNativeLane(owner,'trajectory')};};
const handshake=async(lane,generation=1)=>{lane.receive({kind:'opened',generation,binarySha256:'binary',pid:123});lane.receive({kind:'message',generation,message:capabilities});lane.receive({kind:'message',generation,message:warm});await tick();};

test('A compile holds the trajectory lane: nothing is written to its worker, the worker is not retired as idle, then the queue goes on',async()=>{
  const {commands,lane}=started();
  let answer='none';const start=lane.startGpu().then(why=>{answer=why;});await tick();
  assert.deepEqual(commands[0],{op:'open',lane:'trajectory',generation:1});
  await handshake(lane);
  assert.deepEqual(commands.at(-1).message,{id:-2,op:'gpu'},'the worker told its lane no more than its warmup');
  assert.deepEqual(commands.find(c=>c.message?.id===-1).message,{id:-1,op:'warmup'});
  const job={nativeId:7,payload:{},settled:false};lane.enqueue(job);
  // The idle retirement is 1 ms here: far past it the worker stands, and nothing went to it.
  await sleep(30);
  assert.equal(commands.at(-1).message.id,-2);assert.equal(commands.some(c=>c.op==='close'),false);assert.equal(answer,'none');
  lane.receive({kind:'message',generation:1,message:{id:-2,protocol:NATIVE_PROTOCOL,ready:true}});
  await start;assert.equal(answer,undefined,'started: no reason');
  await tick();assert.equal(commands.at(-1).message.id,7,'then the job');
  lane.receive({kind:'message',generation:1,message:{id:7,error:'end'}});
  lane.receive({kind:'closed',generation:1});await lane.close();
});

test('A GPU that cannot run answers why and takes the lane no further; a second start asks its worker again',async()=>{
  const {commands,lane}=started();
  const first=lane.startGpu();await tick();await handshake(lane);
  lane.receive({kind:'message',generation:1,message:{id:-2,protocol:NATIVE_PROTOCOL,error:'没有 NVIDIA 显卡'}});
  assert.equal(await first,'没有 NVIDIA 显卡');
  const second=lane.startGpu();await tick();assert.equal(commands.filter(c=>c.message?.id===-2).length,2);
  lane.receive({kind:'message',generation:1,message:{id:-2,protocol:NATIVE_PROTOCOL,error:'没有 NVIDIA 显卡'}});
  assert.equal(await second,'没有 NVIDIA 显卡');
  lane.receive({kind:'closed',generation:1});await lane.close();
});

test('A worker that ends during the compile rejects the start, with the queue, and the next start has a worker of its own',async()=>{
  const {commands,finished,lane}=started();
  const crash=lane.startGpu();await tick();await handshake(lane);
  const job={nativeId:8,payload:{},settled:false};lane.enqueue(job);await tick();
  lane.receive({kind:'closed',generation:1,error:'exit code 1'});
  await assert.rejects(crash,/Rust 后端退出。exit code 1/);assert.match(finished[0].error.message,/exit code 1/);
  const next=lane.startGpu();await tick();assert.deepEqual(commands.at(-1),{op:'open',lane:'trajectory',generation:2});
  await handshake(lane,2);assert.equal(commands.at(-1).message.id,-2);
  // A worker that cannot start at all ends the start: it does not spawn another for ever.
  lane.receive({kind:'error',generation:2,error:'write failed'});
  await assert.rejects(next,/write failed/);assert.equal(commands.at(-1).op,'close');
  lane.receive({kind:'closed',generation:2});await tick();assert.equal(commands.filter(c=>c.op==='open').length,2);
  await lane.close();
});

test('A start is the lane\'s last job: a cancelled solve\'s worker going does not lose it, and the service\'s close ends it',async()=>{
  const {commands,lane}=started();
  const job={nativeId:9,payload:{},settled:false};lane.enqueue(job);await tick();await handshake(lane);assert.equal(commands.at(-1).message.id,9);
  // The solve that runs is cancelled for the one that asks for the GPU: its worker goes, the next has the start.
  const start=lane.startGpu();lane.cancel(job,Error('cancelled'));await tick();assert.equal(commands.at(-1).op,'close');
  lane.receive({kind:'closed',generation:1});await tick();assert.deepEqual(commands.at(-1),{op:'open',lane:'trajectory',generation:2});
  await handshake(lane,2);assert.equal(commands.at(-1).message.id,-2);
  const closing=lane.close();await assert.rejects(start,/shutting down/);
  lane.receive({kind:'closed',generation:2});await closing;
});

test('IPC lane fails the worker on any progress but a checkpoint: the deleted 2D engine\'s running report is no message',async()=>{
  const commands=[],finished=[];const owner={binarySha256:'binary',build:{sha256:'build'},startupTimeoutMs:1000,workerIdleMs:1000,sendNative:c=>commands.push(c),finish:(job,error,value)=>{job.settled=true;finished.push({job,error,value});}};
  const lane=new IpcNativeLane(owner,'trajectory');
  const job={nativeId:5,payload:{},settled:false,onProgress:()=>{}};lane.enqueue(job);await tick();
  lane.receive({kind:'opened',generation:1,binarySha256:'binary',pid:123});lane.receive({kind:'message',generation:1,message:capabilities});lane.receive({kind:'message',generation:1,message:warm});await tick();
  lane.receive({kind:'message',generation:1,message:{id:5,protocol:NATIVE_PROTOCOL,progress:'running',running:{simulatedUs:250,durationUs:1000,trajectories:16,estimatedRemainingMs:4200}}});
  assert.equal(finished[0].error.code,'invalid-native-response');assert.equal(commands.at(-1).op,'close');
  lane.receive({kind:'closed',generation:1});await lane.close();
});

const fakeResult=({model,options})=>nativeResult({states:model.states,initialIndex:model.initialIndex,tolerance:options.populationTolerance,mean:options.includeMean,
  timesUs:options.meanOnly?[0]:Array.from({length:options.samples},(_,i)=>options.durationUs*i/(options.samples-1))});

test('Worker sends the identical complete model, finite jobs to the trajectory lane and means to the mean lane',async()=>{
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
  await worker.receive({kind:'native',event:{kind:'configuration',binarySha256:'binary',buildSha256:'build',defaults:{ATOM_SOLVER_THREADS:'8'}}});
  await tick();assert.equal(events.find(e=>e.channel==='invalid').status,400);
  const scene=defaultScene();scene.lasers[1].enabled=false;scene.lasers.push({...scene.lasers[0],id:'second',frequency:{base:{kind:'transition',line:'D2',Fg:2,Fe:2},offsetMHz:100}});
  const options={durationUs:0.1,samples:3,populationTolerance:.003731,meanOnly:false,includeMean:true};const request={scene,modelSchema:FULL_MODEL_SCHEMA,stream:true,request:{clientId:'worker-test',sequence:1},options};
  await worker.receive({kind:'solve',channel:'valid',request});
  for(let i=0;i<50&&!events.some(e=>e.channel==='valid'&&e.event?.type==='result');i++)await tick();
  const messages=events.filter(e=>e.channel==='valid');assert.deepEqual(messages.map(e=>e.event?.type),['result']);
  // One native job: the finite trajectory carries its own inline mean.
  assert.equal(requests.length,1);assert.deepEqual(requests.map(r=>r.lane),['trajectory']);
  assert.equal(requests[0].options.includeMean,true);assert.equal(requests[0].progress,true);
  const model=buildFullMasterModel(scene,{...scene.dynamics,...options});
  for(const sent of requests){
    // A tolerance in the options is not read: every native request asks for POPULATION_TOLERANCE.
    assert.equal(sent.options.populationTolerance,POPULATION_TOLERANCE);
    assert.equal(sent.modelSchema,FULL_MODEL_SCHEMA);
    // Exactly the fields the worker reads; e.g. no frameFrequenciesMHz: the
    // native solver returns the laboratory-frame mean itself.
    assert.deepEqual(Object.keys(sent).filter(key=>key!=='lane').sort(),['id','model','modelSchema','options','progress']);
    assert.deepEqual(Object.keys(sent.model).sort(),['collapse','dimension','drives','frequencyCoordinates','hImag','hReal','initialImag','initialIndex','initialReal','states']);
    for(const key of ['dimension','hReal','hImag','collapse','initialIndex','initialReal','initialImag','drives','frequencyCoordinates'])assert.deepEqual(sent.model[key],model[key],key);
    assert.deepEqual(sent.model.states,model.states.map(s=>s.id));
  }
  assert.equal(messages.at(-1).event.data.service.lane,'trajectory');
  assert.equal(messages.at(-1).event.data.service.meanCacheHit,false);
  // The Worker's response is exactly what the UI reader accepts.
  const finite=normalizeDynamicsResult(messages.at(-1).event.data,scene);
  assert.equal(finite.timesUs.length,3);assert.equal(finite.mean.converged,true);assert.deepEqual(finite.assumptions,model.assumptions);
  // A scene that still carries a tolerance of its own: the mean asks for POPULATION_TOLERANCE too.
  const old=structuredClone(scene);old.dynamics.populationTolerance=.021;
  await worker.receive({kind:'solve',channel:'mean',request:{scene:old,modelSchema:FULL_MODEL_SCHEMA,
    request:{clientId:'worker-test',sequence:2},options:{durationUs:.1,samples:3,meanOnly:true,includeMean:true}}});
  for(let i=0;i<50&&!events.some(e=>e.channel==='mean'&&e.kind==='response');i++)await tick();
  assert.equal(requests.at(-1).options.populationTolerance,POPULATION_TOLERANCE);assert.equal(requests.at(-1).lane,'mean');
  const mean=normalizeDynamicsResult(events.find(e=>e.channel==='mean'&&e.kind==='response').data,old);
  assert.equal(mean.timesUs.length,1);assert.equal(mean.mean.converged,true);
  await worker.receive({kind:'shutdown'});assert.equal(events.at(-1).kind,'shutdown-complete');
});

test('Worker answers a GPU start asked before its configuration: with why it cannot run, with the failure that ended its worker',async()=>{
  const events=[];let worker;const answers=[{id:-2,protocol:NATIVE_PROTOCOL,error:'没有 NVIDIA 显卡'}];
  const host=command=>{
    const event=message=>queueMicrotask(()=>void worker.receive({kind:'native',event:{lane:command.lane,generation:command.generation,...message}}));
    if(command.op==='open')event({kind:'opened',pid:100,binarySha256:'binary'});
    if(command.op==='close')event({kind:'closed'});
    if(command.op==='write'){
      const request=command.message;
      if(request.id===0)event({kind:'message',message:capabilities});
      else if(request.id===-1)event({kind:'message',message:warm});
      else if(request.id===-2){assert.equal(command.lane,'trajectory');const answer=answers.shift();event(answer?{kind:'message',message:answer}:{kind:'closed',error:'exit code 1'});}
    }
  };
  worker=createModelWorker(message=>{events.push(message);if(message.kind==='native')host(message.command);});
  const answered=async channel=>{for(let i=0;i<100&&!events.some(e=>e.channel===channel);i++)await tick();return events.find(e=>e.channel===channel);};
  await worker.receive({kind:'gpu',channel:'early'});
  assert.equal(events.some(e=>e.channel==='early'),false,'it waits for the configuration');
  await worker.receive({kind:'native',event:{kind:'configuration',binarySha256:'binary',buildSha256:'build',defaults:{}}});
  const cannot=await answered('early');assert.equal(cannot.status,200);assert.equal(cannot.data.reason,'没有 NVIDIA 显卡');
  await worker.receive({kind:'gpu',channel:'crash'});
  const crashed=await answered('crash');assert.equal(crashed.status,503);assert.match(crashed.data.error,/exit code 1/);
  await worker.receive({kind:'shutdown'});
  await worker.receive({kind:'gpu',channel:'closed'});assert.equal(events.at(-1).status,503);
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
