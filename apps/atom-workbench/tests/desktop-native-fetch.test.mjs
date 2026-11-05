import test from 'node:test';import assert from 'node:assert/strict';import {installNativeFetch} from '../desktop-native/ui/native-fetch.mjs';
class FakeWorker extends EventTarget{
 messages=[];terminated=0;autoAck=true;
 postMessage(message){this.messages.push(structuredClone(message));if(message.kind==='shutdown'&&this.autoAck)queueMicrotask(()=>this.emit({kind:'shutdown-complete'}));}
 terminate(){this.terminated++;}
 emit(data){this.dispatchEvent(new MessageEvent('message',{data}));}
 crash(message='Worker crashed'){const event=new Event('error',{cancelable:true});Object.defineProperty(event,'message',{value:message});this.dispatchEvent(event);}
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const input=(sequence=1,stream=false)=>({scene:{name:'same physical scene'},stream,request:{clientId:'test',sequence},options:{samples:3,durationUs:20}});
function fixture(extra={}){const worker=new FakeWorker(),commands=[],fallback=[];
 const adapter=installNativeFetch({worker,postNative:command=>commands.push(command),originalFetch:(...args)=>{fallback.push(args);return Promise.resolve(new Response('fallback'));},timeoutMs:1000,shutdownTimeoutMs:20,...extra});
 return {worker,commands,fallback,adapter};}
const solve=(adapter,body=input(),signal)=>adapter.fetch('http://127.0.0.1:4180/api/solve',{method:'POST',body:JSON.stringify(body),signal});
async function posted(worker){await tick();return worker.messages.filter(message=>message.kind==='solve').at(-1);}
const line=chunk=>JSON.parse(new TextDecoder().decode(chunk).trim());

test('a GPU start has no deadline, keeps its answer, and ends with the adapter',async()=>{
 const f=fixture({timeoutMs:5});try{
  const pending=f.adapter.fetch('http://127.0.0.1:4180/api/gpu',{method:'POST'});await tick();
  const message=f.worker.messages.at(-1);assert.equal(message.kind,'gpu');
  // Far past the solves' transport deadline it stands.
  await new Promise(resolve=>setTimeout(resolve,40));let settled=false;void pending.finally(()=>{settled=true;});await tick();assert.equal(settled,false);
  f.worker.emit({kind:'response',channel:message.channel,status:200,data:{reason:'没有 NVIDIA 显卡'}});
  const response=await pending;assert.equal(response.status,200);assert.equal((await response.json()).reason,'没有 NVIDIA 显卡');
  // A model Worker that dies takes the start with it: nothing waits for ever.
  const lost=f.adapter.fetch('http://127.0.0.1:4180/api/gpu',{method:'POST'});await tick();f.worker.crash('Worker crashed');
  await assert.rejects(lost,/Worker crashed/);
 }finally{await f.adapter.dispose();}
});

test('native JSON response preserves identity and other URL paths use the original fetch',async()=>{
 const f=fixture();try{
  assert.equal(await(await f.adapter.fetch('/data.json')).text(),'fallback');assert.equal(f.fallback.length,1);
  const pending=solve(f.adapter),message=await posted(f.worker);assert.deepEqual(message.request,input());
  f.adapter.receiveNative({kind:'configuration',binarySha256:'native'});assert.equal(f.worker.messages.at(-1).kind,'native');
  f.worker.emit({kind:'native',command:{op:'open',lane:'trajectory',generation:1}});assert.equal(f.commands.at(-1).op,'open');
  const data={request:input().request,populations:[[1,0]],protocol:'same-protocol'};f.worker.emit({kind:'response',channel:message.channel,status:200,data});
  const response=await pending;assert.equal(response.status,200);assert.deepEqual(await response.json(),data);
  f.worker.emit({kind:'response',channel:message.channel,status:500,data:{error:'late'}});
 }finally{await f.adapter.dispose();}assert.equal(f.worker.terminated,1);assert.equal(f.commands.at(-1).op,'close-all');
});

test('mean then final result form one ordered NDJSON stream and late messages are ignored',async()=>{
 const f=fixture();try{
  const pending=solve(f.adapter,input(1,true)),message=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:message.channel,event:{type:'checkpoint',data:{request:input().request,mean:0.2}}});
  const response=await pending;assert.match(response.headers.get('Content-Type'),/application\/x-ndjson/);const reader=response.body.getReader();
  assert.equal(line((await reader.read()).value).type,'checkpoint');
  f.worker.emit({kind:'stream',channel:message.channel,event:{type:'result',data:{request:input().request,populations:[[1,0]]}},done:true});
  assert.equal(line((await reader.read()).value).type,'result');assert.equal((await reader.read()).done,true);
  f.worker.emit({kind:'stream',channel:message.channel,event:{type:'checkpoint',data:{late:true}}});
  assert.equal(f.worker.messages.filter(message=>message.kind==='cancel').length,0);
 }finally{await f.adapter.dispose();}
});

test('abort before Worker dispatch never starts a native request',async()=>{
 const f=fixture();try{const controller=new AbortController(),pending=solve(f.adapter,input(),controller.signal);controller.abort();
  await assert.rejects(pending,{name:'AbortError'});await tick();assert.equal(f.worker.messages.filter(message=>message.kind==='solve').length,0);
 }finally{await f.adapter.dispose();}
});

test('abort after mean errors the body, cancels once, and stale final cannot affect the next request',async()=>{
 const f=fixture();try{
  const controller=new AbortController(),pending=solve(f.adapter,input(1,true),controller.signal),old=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:old.channel,event:{type:'checkpoint',data:{mean:0.4}}});
  const reader=(await pending).body.getReader();await reader.read();const blocked=reader.read();controller.abort();
  await assert.rejects(blocked,{name:'AbortError'});assert.equal(f.worker.messages.filter(message=>message.kind==='cancel'&&message.channel===old.channel).length,1);
  const next=solve(f.adapter,input(2)),fresh=await posted(f.worker);assert.notEqual(fresh.channel,old.channel);
  f.worker.emit({kind:'stream',channel:old.channel,event:{type:'result',data:{wrong:true}},done:true});
  f.worker.emit({kind:'response',channel:fresh.channel,status:200,data:{request:input(2).request,correct:true}});
  assert.equal((await(await next).json()).correct,true);
 }finally{await f.adapter.dispose();}
});

test('canceling the response reader cancels native work without waiting for AbortSignal',async()=>{
 const f=fixture();try{const pending=solve(f.adapter,input(1,true)),message=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:message.channel,event:{type:'checkpoint',data:{}}});const response=await pending;
  await response.body.cancel();assert.equal(f.worker.messages.filter(value=>value.kind==='cancel').length,1);
  f.worker.emit({kind:'stream',channel:message.channel,event:{type:'result',data:{}},done:true});
 }finally{await f.adapter.dispose();}
});

test('timeout cancels only its channel and ignores the late terminal response',async()=>{
 const f=fixture({timeoutMs:15});try{const pending=solve(f.adapter),rejected=assert.rejects(pending,{name:'TimeoutError'}),message=await posted(f.worker);
  await rejected;assert.equal(f.worker.messages.filter(value=>value.kind==='cancel'&&value.channel===message.channel).length,1);
  f.worker.emit({kind:'response',channel:message.channel,status:200,data:{late:true}});
 }finally{await f.adapter.dispose();}
});

test('the checkpoints of a motion solve stream on and restart its deadline; a stream may not end on one',async()=>{
 const f=fixture({timeoutMs:60});try{
  const body={...input(1,true),options:{...input().options,motion:{axis:'3d',temperatureUK:30},device:'cpu'}};
  const pending=solve(f.adapter,body),message=await posted(f.worker);
  const beat=()=>f.worker.emit({kind:'stream',channel:message.channel,event:{type:'checkpoint',data:{request:input().request,progress:true,populations:[[1,0]]}}});
  beat();const response=await pending,reader=response.body.getReader();assert.equal(line((await reader.read()).value).type,'checkpoint');
  // Checkpoints 40 ms apart keep a 60 ms deadline from expiring for 160 ms.
  for(let k=0;k<4;k++){await new Promise(resolve=>setTimeout(resolve,40));beat();assert.equal(line((await reader.read()).value).type,'checkpoint');}
  f.worker.emit({kind:'stream',channel:message.channel,event:{type:'result',data:{request:input().request,populations:[[1,0]]}},done:true});
  assert.equal(line((await reader.read()).value).type,'result');assert.equal((await reader.read()).done,true);
  const ended=solve(f.adapter,input(2,true)),next=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:next.channel,event:{type:'checkpoint',data:{request:input(2).request,progress:true,populations:[[1,0]]}},done:true});
  await assert.rejects(ended);
  // The running report of the deleted 2D engine is no event any more.
  const stale=solve(f.adapter,input(3,true)),last=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:last.channel,event:{type:'running',data:{request:input(3).request,running:{simulatedUs:1,durationUs:2,trajectories:0}}}});
  await assert.rejects(stale);
 }finally{await f.adapter.dispose();}
});
test('Worker failure rejects pending fetch/body and synchronously closes native children',async()=>{
 const f=fixture();try{
  const pending=solve(f.adapter),waiting=assert.rejects(pending,/Worker crashed/);await posted(f.worker);
  const streaming=solve(f.adapter,input(2,true)),message=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:message.channel,event:{type:'checkpoint',data:{}}});const reader=(await streaming).body.getReader();await reader.read();
  const bodyFailure=assert.rejects(reader.read(),/Worker crashed/);f.worker.crash();
  assert.equal(f.commands.at(-1).op,'close-all');await waiting;await bodyFailure;await tick();assert.equal(f.worker.terminated,1);
  await assert.rejects(solve(f.adapter,input(3)),{name:'AbortError'});
 }finally{await f.adapter.dispose();}
});

test('dispose closes the host directly, suppresses late opens, and bounds a stuck Worker shutdown',async()=>{
 const f=fixture({shutdownTimeoutMs:8});f.worker.autoAck=false;
 const pending=solve(f.adapter),rejected=assert.rejects(pending,{name:'AbortError'});await posted(f.worker);
 const closing=f.adapter.dispose();assert.equal(f.commands[0].op,'close-all');
 f.worker.emit({kind:'native',command:{op:'open',lane:'mean',generation:7}});
 f.worker.emit({kind:'native',command:{op:'write',lane:'trajectory',generation:2,line:'old'}});
 assert.equal(f.commands.length,1);await rejected;await closing;assert.equal(f.worker.terminated,1);
 await f.adapter.dispose();assert.equal(f.worker.terminated,1);
});

test('invalid JSON/method and premature stream termination fail without inventing a result',async()=>{
 const f=fixture();try{
  assert.equal((await f.adapter.fetch('/api/solve',{method:'GET'})).status,405);
  assert.equal((await f.adapter.fetch('/api/solve',{method:'POST',body:'{'})).status,400);
  assert.equal(f.worker.messages.filter(message=>message.kind==='solve').length,0);
  const pending=solve(f.adapter,input(1,true)),rejected=assert.rejects(pending,/without a final result/),message=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:message.channel,event:{type:'checkpoint',data:{}},done:true});await rejected;
 }finally{await f.adapter.dispose();}
});

test('a numerical stream error is terminal only for that request and a fresh retry can succeed',async()=>{
 const f=fixture();try{
  const first=solve(f.adapter,input(1,true)),old=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:old.channel,event:{type:'checkpoint',data:{request:input(1).request,mean:0.4}}});
  const reader=(await first).body.getReader();assert.equal(line((await reader.read()).value).type,'checkpoint');
  f.worker.emit({kind:'stream',channel:old.channel,event:{type:'error',status:422,data:{request:input(1).request,error:'Numerical refinement failed',code:'numerical-failure'}},done:true});
  assert.equal(line((await reader.read()).value).data.code,'numerical-failure');assert.equal((await reader.read()).done,true);
  assert.equal(f.worker.terminated,0);assert.equal(f.commands.length,0);
  const retry=solve(f.adapter,input(2)),fresh=await posted(f.worker);
  f.worker.emit({kind:'stream',channel:old.channel,event:{type:'result',data:{wrong:true}},done:true});
  f.worker.emit({kind:'response',channel:fresh.channel,status:200,data:{request:input(2).request,correct:true}});
  assert.deepEqual(await(await retry).json(),{request:input(2).request,correct:true});assert.equal(f.worker.terminated,0);
 }finally{await f.adapter.dispose();}
});
