/** Experimental browser transport for the existing DynamicsClient protocol.
 * This is a native desktop bridge, NOT a pure-HTML/WASM population solver.
 * Only /api/solve is intercepted. No historical result cache is maintained. */
export function installNativeFetch({worker,postNative,originalFetch=globalThis.fetch?.bind(globalThis),timeoutMs=120000,shutdownTimeoutMs=500}){
 if(!worker?.postMessage||!worker?.addEventListener||!worker?.terminate||typeof postNative!=='function')throw new TypeError('Native fetch requires a Worker and native bridge');
 if(!Number.isFinite(timeoutMs)||timeoutMs<=0||!Number.isFinite(shutdownTimeoutMs)||shutdownTimeoutMs<=0)throw new TypeError('Invalid native fetch timeout');
 const pending=new Map(),encoder=new TextEncoder();let sequence=0,phase='active',shutdownTimer,closePromise,closeResolve;
 const prefix=globalThis.crypto?.randomUUID?.()??`native-${Date.now()}-${Math.random().toString(36).slice(2)}`;
 const aborted=message=>new DOMException(message??'The native solve was aborted.','AbortError');
 const expired=()=>new DOMException('The native solve exceeded its transport deadline.','TimeoutError');
 const isCurrent=job=>!job.finished&&pending.get(job.channel)===job;
 const release=job=>{if(!isCurrent(job))return false;job.finished=true;pending.delete(job.channel);clearTimeout(job.timer);job.signal?.removeEventListener('abort',job.abort);return true;};
 function cancelWorker(job){if(job.dispatched&&phase!=='closed'){try{worker.postMessage({kind:'cancel',channel:job.channel});}catch{/* Direct close-all on fatal disposal remains available. */}}}
 function fail(job,error,{cancel=true,errorBody=true}={}){
  if(!release(job))return;if(cancel)cancelWorker(job);
  if(job.controller){if(errorBody){try{job.controller.error(error);}catch{/* Already canceled by the reader. */}}}
  else job.reject(error);
 }
 function finishShutdown(){
  if(phase==='closed')return;phase='closed';clearTimeout(shutdownTimer);
  worker.removeEventListener('message',onMessage);worker.removeEventListener('error',onError);worker.removeEventListener('messageerror',onMessageError);
  try{worker.terminate();}finally{closeResolve?.();}
 }
 function shutdown(reason){
  if(phase!=='active')return closePromise??Promise.resolve();
  phase='closing';closePromise=new Promise(resolve=>{closeResolve=resolve;});
  // A Worker may be inside synchronous model work or already have failed.
  // Send this directly, BEFORE waiting for its acknowledgement, so native
  // processes cannot be orphaned by Worker termination.
  try{postNative({op:'close-all'});}catch{/* The host also owns process cleanup on application exit. */}
  for(const job of [...pending.values()])fail(job,reason);
  try{worker.postMessage({kind:'shutdown'});}catch{finishShutdown();}
  if(phase!=='closed')shutdownTimer=setTimeout(finishShutdown,shutdownTimeoutMs);
  return closePromise;
 }
 function streamResponse(job){
  const body=new ReadableStream({start(controller){job.controller=controller;},cancel(reason){
   fail(job,reason instanceof Error?reason:aborted('The response reader canceled the native solve.'),{errorBody:false});
  }});
  job.responded=true;job.resolve(new Response(body,{status:200,headers:{'Content-Type':'application/x-ndjson; charset=utf-8','Cache-Control':'no-store'}}));
 }
 function onMessage(message){
  const value=message.data;if(!value||typeof value!=='object')return;
  if(value.kind==='shutdown-complete'){if(phase==='closing')finishShutdown();return;}
  if(value.kind==='native'){
   // Late queued open/write commands must not restart a process after the
   // direct close-all above. Graceful lane-close commands may still finish.
   if(phase==='closed'||phase==='closing'&&!['close','close-all'].includes(value.command?.op))return;
   try{postNative(value.command);}catch(error){void shutdown(error instanceof Error?error:new Error(String(error)));}return;
  }
  if(phase!=='active')return;const job=pending.get(value.channel);if(!job||!isCurrent(job))return;
  try{
   if(value.kind==='response'){
    if(job.responded)throw new Error('Native Worker changed an active NDJSON response into JSON.');
    const status=value.status??200;if(!Number.isInteger(status)||status<200||status>599)throw new Error('Invalid native Worker response status.');
    const response=new Response(JSON.stringify(value.data),{status,headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'}});
    release(job);job.resolve(response);return;
   }
   if(value.kind==='stream'){
    const event=value.event;if(!event||!['asymptotic','result','error'].includes(event.type))throw new Error('Unknown native Worker stream event.');
    if(value.done&&event.type==='asymptotic')throw new Error('Native Worker ended a stream without a final result.');
    if(!job.responded)streamResponse(job);
    job.controller.enqueue(encoder.encode(JSON.stringify(event)+'\n'));
    if(event.type==='result'||event.type==='error'){release(job);job.controller.close();}return;
   }
  }catch(error){fail(job,error instanceof Error?error:new Error(String(error)));}
 }
 function onError(event){event.preventDefault?.();void shutdown(new Error(event.message||'The native model Worker failed.'));}
 function onMessageError(){void shutdown(new Error('The native model Worker sent an unreadable message.'));}
 worker.addEventListener('message',onMessage);worker.addEventListener('error',onError);worker.addEventListener('messageerror',onMessageError);
 function fetch(input,init){
  let url,request;
  try{url=new URL(input instanceof Request?input.url:String(input),globalThis.location?.href??'http://127.0.0.1:4180/');}
  catch(error){return Promise.reject(error);}
  if(url.pathname!=='/api/solve')return originalFetch?originalFetch(input,init):Promise.reject(new TypeError('No fallback fetch is available.'));
  if(phase!=='active')return Promise.reject(aborted('The native fetch adapter is closed.'));
  try{request=input instanceof Request?new Request(input,init):new Request(url,init);}
  catch(error){return Promise.reject(error);}
  if(request.method!=='POST')return Promise.resolve(new Response(JSON.stringify({error:'Native solve requires POST.',code:'invalid-request'}),{status:405,headers:{'Content-Type':'application/json'}}));
  return new Promise((resolve,reject)=>{
   const job={channel:`${prefix}:${++sequence}`,resolve,reject,signal:request.signal,finished:false,responded:false,dispatched:false,controller:null,timer:null,abort:null};
   job.abort=()=>fail(job,aborted());pending.set(job.channel,job);
   if(job.signal.aborted){fail(job,aborted());return;}
   job.signal.addEventListener('abort',job.abort,{once:true});job.timer=setTimeout(()=>fail(job,expired()),timeoutMs);
   void request.text().then(text=>{
    if(!isCurrent(job)||phase!=='active')return;let body;
    try{body=JSON.parse(text);}catch{
     release(job);resolve(new Response(JSON.stringify({error:'Invalid JSON solve request.',code:'invalid-request'}),{status:400,headers:{'Content-Type':'application/json'}}));return;
    }
    job.dispatched=true;worker.postMessage({kind:'solve',channel:job.channel,request:body});
   }).catch(error=>fail(job,error instanceof Error?error:new Error(String(error))));
  });
 }
 function receiveNative(event){if(phase==='closed')return;try{worker.postMessage({kind:'native',event});}catch(error){void shutdown(error instanceof Error?error:new Error(String(error)));}}
 return {fetch,receiveNative,dispose:()=>shutdown(aborted('The native fetch adapter was disposed.'))};
}
