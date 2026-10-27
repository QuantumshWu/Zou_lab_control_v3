import {createServer} from 'node:http';
import {failureRecorder} from './solver-failure-log.mjs';
import {createNativeServiceClass} from './native-service-core.mjs';
import {readFile,stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {dirname,resolve,extname,sep} from 'node:path';
import {spawn} from 'node:child_process';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {nativeCapabilities,prepareNativeSolve,validateAtomicResult} from './native-protocol.mjs';
import {nativeChildEnvironment} from './native-child-env.mjs';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';

const app=resolve(dirname(fileURLToPath(import.meta.url)),'..'),root=resolve(app,'dist');
const defaultFailureRecorder=failureRecorder(resolve(app,'artifacts/diagnostics/solver-failures.json'));
export {HTTP_PROTOCOL,NATIVE_PROTOCOL};
const hash=data=>createHash('sha256').update(data).digest('hex');
const failure=(message,status=503,code='solver-unavailable')=>Object.assign(new Error(message),{status,code});
const setting=(value,fallback,min,max)=>{const result=Number(value??fallback);if(!Number.isFinite(result)||result<min||result>max)throw new Error('Invalid solver service setting.');return result;};

async function fileIdentity(path){const [data,info]=await Promise.all([readFile(path),stat(path)]);return {path,sha256:hash(data),size:info.size,modifiedAt:info.mtime.toISOString()};}
async function buildIdentity(){
  const files=['physics.js','laser-frequency.js','laser-intensity.js','types.js','solver-protocol.js','mean-density.js','spectroscopy.js','spectroscopy-data.js','full-master-model.js','dynamics-cache-key.js','floquet-background.js'];
  const entries=await Promise.all(files.map(async name=>({name,sha256:hash(await readFile(resolve(root,'js/atom-workbench/src',name)))})));
  entries.push({name:'scripts/serve.mjs',sha256:hash(await readFile(fileURLToPath(import.meta.url)))});
  entries.push({name:'scripts/solver-failure-log.mjs',sha256:hash(await readFile(resolve(app,'scripts/solver-failure-log.mjs')))});
  entries.push({name:'scripts/native-protocol.mjs',sha256:hash(await readFile(resolve(app,'scripts/native-protocol.mjs')))});
  entries.push({name:'scripts/native-service-core.mjs',sha256:hash(await readFile(resolve(app,'scripts/native-service-core.mjs')))});
  entries.push({name:'scripts/native-child-env.mjs',sha256:hash(await readFile(resolve(app,'scripts/native-child-env.mjs')))});
  return {sha256:hash(JSON.stringify(entries)),files:entries,node:process.version};
}

/** One persistent native process per lane. Only one request is ever written
 * ahead of its reply; cancellation kills that exact process, not a process
 * name or a global PID group. Pending work remains in bounded JavaScript data. */
class NativeLane {
  constructor(owner,name){this.owner=owner;this.name=name;this.worker=null;this.active=null;this.queue=[];this.generation=0;this.error='';this.pumpTimer=null;this.idleTimer=null;this.closed=false;}
  snapshot(){const w=this.worker;return {name:this.name,status:this.closed?'closed':w?.retiring?'restarting':this.active?'busy':w?.ready?'ready':w?'starting':'idle',
    pid:w?.child?.pid??null,generation:this.generation,queued:this.queue.length,active:this.active?.token??null,binary:w?.identity??null,
    nativeProtocol:w?.capabilities?.protocol??null,capabilities:w?.capabilities?.capabilities??[],error:this.error||undefined};}
  enqueue(job){clearTimeout(this.idleTimer);this.idleTimer=null;job.queuedAt=performance.now();this.queue.push(job);this.schedule();}
  schedule(delay=this.owner.restartDebounceMs){
    if(this.closed||this.pumpTimer)return;
    const run=()=>{this.pumpTimer=null;void this.pump();};
    if(delay===0){this.pumpTimer={microtask:true};queueMicrotask(run);}else this.pumpTimer=setTimeout(run,delay);
  }
  async start(){
    if(this.closed||this.worker)return;
    const w={child:null,ready:false,retiring:false,buffer:'',generation:++this.generation,identity:null,capabilities:null,startTimer:null,killTimer:null,exit:null};
    this.worker=w;
    try{
      w.identity=await fileIdentity(this.owner.executable);
      if(this.closed||w.retiring||this.worker!==w){this.worker=null;return;}
      const child=spawn(this.owner.executable,[],{cwd:app,windowsHide:true,stdio:['pipe','pipe','pipe'],env:nativeChildEnvironment(process.env)});
      w.child=child;w.exit=new Promise(accept=>child.once('close',accept));
      child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
      child.stdout.on('data',data=>this.receive(w,data));
      child.stderr.on('data',data=>{this.error=String(data).trim().slice(-2000);});
      child.stdin.on('error',error=>this.failWorker(w,error));
      child.on('error',error=>this.failWorker(w,error));
      child.once('close',()=>{
        clearTimeout(w.startTimer);clearTimeout(w.killTimer);if(this.worker!==w)return;
        this.worker=null;
        if(!w.retiring){const error=failure('Rust 后端退出。'+(this.error||''));if(this.active){const active=this.active;this.active=null;this.owner.finish(active,error);}this.rejectQueue(error);}
        if(this.queue.length&&!this.closed)this.schedule();
      });
      w.startTimer=setTimeout(()=>this.failWorker(w,failure('Rust worker startup timed out.')),this.owner.startupTimeoutMs);
      child.stdin.write(JSON.stringify({op:'capabilities',id:0})+'\n');
    }catch(error){this.error=error.message;if(this.worker===w)this.worker=null;this.rejectQueue(failure('Rust 后端无法启动：'+error.message));}
  }
  receive(w,data){
    if(this.worker!==w||w.retiring)return;
    w.buffer+=data;
    if(w.buffer.length>64*1024*1024){this.failWorker(w,failure('Native response exceeded 64 MiB.',502,'invalid-native-response'));return;}
    for(let end;(end=w.buffer.indexOf('\n'))>=0;){
      const line=w.buffer.slice(0,end);w.buffer=w.buffer.slice(end+1);if(!line.trim())continue;
      let message;try{message=JSON.parse(line);}catch{this.failWorker(w,failure('Invalid native JSON response.',502,'invalid-native-response'));return;}
      if(!w.ready){
        if(!w.capabilities){
          try{w.capabilities=nativeCapabilities(message);}catch(error){this.failWorker(w,failure(error.message,503,'incompatible-native-protocol'));return;}
          // Capability negotiation comes first. The warmup operation then
          // checks the numerical runtime (a LAPACK self-check).
          w.child.stdin.write(JSON.stringify({id:-1,op:'warmup'})+'\n');
          continue;
        }
        if(message.id!==-1||message.error||message.protocol!==NATIVE_PROTOCOL||message.ready!==true){
          this.failWorker(w,failure('Native numerical startup check failed: '+(message.error??'invalid warmup response'),503,'invalid-native-startup'));return;
        }
        // Pair the handshake with the executable that was actually opened.
        // If a build replaced the path during startup, do not mislabel its PID.
        void fileIdentity(this.owner.executable).then(identity=>{
          if(this.worker!==w||w.retiring)return;
          if(identity.sha256!==w.identity.sha256){this.failWorker(w,failure('Native executable changed during worker startup.'));return;}
          clearTimeout(w.startTimer);w.ready=true;this.error='';this.owner.verified=true;this.schedule(0);
        }).catch(error=>this.failWorker(w,error));continue;
      }
      const job=this.active;
      if(!job||message.id!==job.nativeId){this.failWorker(w,failure('Native response identity mismatch.',502,'invalid-native-response'));return;}
      if(message.progress!==undefined){
        try{
          if(message.progress!=='asymptotic'||!job.onProgress)throw Error('Unexpected native progress message.');
          const result=message.result;validateAtomicResult(result,job.payload);
          job.onProgress({...result,service:{protocol:HTTP_PROTOCOL,nativeProtocol:w.capabilities.protocol,capabilities:w.capabilities.capabilities,lane:this.name,generation:w.generation,binarySha256:w.identity.sha256,buildSha256:this.owner.build.sha256,asymptoticCacheHit:false}});
        }catch(error){this.failWorker(w,failure(error.message,502,'invalid-native-response'));return;}
        continue;
      }
      this.active=null;
      if(message.error)this.owner.finish(job,failure(message.error,422,'numerical-failure'));
      else if(!message.result||typeof message.result!=='object')this.owner.finish(job,failure('Missing native result.',502,'invalid-native-response'));
      else {
        try{
          const result=message.result;
          validateAtomicResult(result,job.payload);
          this.owner.finish(job,null,{...result,service:{protocol:HTTP_PROTOCOL,nativeProtocol:w.capabilities.protocol,capabilities:w.capabilities.capabilities,lane:this.name,generation:w.generation,binarySha256:w.identity.sha256,buildSha256:this.owner.build.sha256,
            asymptoticCacheHit:false}});
        }catch(error){this.owner.finish(job,failure(error.message,502,'invalid-native-response'));this.retire();}
      }
      this.schedule(0);
    }
  }
  async pump(){
    if(this.closed||this.active||this.worker?.retiring)return;
    this.queue=this.queue.filter(job=>!job.settled);
    if(!this.queue.length){
      if(this.name==='trajectory'&&this.worker?.ready&&!this.idleTimer){this.idleTimer=setTimeout(()=>{this.idleTimer=null;if(!this.active&&!this.queue.length)this.retire();},this.owner.workerIdleMs);}
      return;
    }
    if(!this.worker){await this.start();return;}
    if(!this.worker.ready)return;
    const job=this.queue.shift();if(job.settled){this.schedule(0);return;}
    this.active=job;
    job.startedAt=performance.now();
    // Each calculation is fresh. An explicitly independent trajectory may
    // have a separate fresh mean child owned by the same HTTP request.
    const options=job.payload.options;
    this.worker.child.stdin.write(JSON.stringify({id:job.nativeId,...job.payload,options,progress:!!job.onProgress})+'\n');
  }
  cancel(job,error){
    this.queue=this.queue.filter(item=>item!==job);
    if(this.active===job){this.active=null;this.retire();}
    this.owner.finish(job,error);this.schedule();
  }
  rejectQueue(error){const jobs=this.queue.splice(0);for(const job of jobs)this.owner.finish(job,error);}
  failWorker(w,error){
    if(this.worker!==w||w.retiring)return;if(!error.status)error=failure('Rust 后端连接失败：'+error.message);this.error=error.message;
    if(this.active){const job=this.active;this.active=null;this.owner.finish(job,error);}
    this.rejectQueue(error);this.retire();
  }
  retire(){
    clearTimeout(this.idleTimer);this.idleTimer=null;
    const w=this.worker;if(!w||w.retiring)return;w.retiring=true;w.ready=false;clearTimeout(w.startTimer);
    if(w.child){w.child.kill();w.killTimer=setTimeout(()=>w.child.kill('SIGKILL'),2000);}
  }
  async close(){
    this.closed=true;clearTimeout(this.pumpTimer);clearTimeout(this.idleTimer);
    const error=failure('Local solver service is shutting down.');
    if(this.active){const job=this.active;this.active=null;this.owner.finish(job,error);}this.rejectQueue(error);
    this.retire();const child=this.worker?.exit;if(child)await child;
  }
}

export const NativeService=createNativeServiceClass(NativeLane);

const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.svg':'image/svg+xml','.woff2':'font/woff2','.woff':'font/woff'};
export async function createAtomServer(options={}){
  const recordFailure=options.failureLogPath===false?async()=>{}:options.failureLogPath?failureRecorder(options.failureLogPath):defaultFailureRecorder;
  const executable=resolve(options.executable??process.env.ZLC_ATOM_SOLVER??resolve(app,'rust-solver/target/x86_64-pc-windows-msvc/release/atom-lindblad.exe'));
  const [build,configuredBinary]=await Promise.all([buildIdentity(),fileIdentity(executable)]);
  buildFullMasterModel(defaultScene()); // Initialize atomic tables before accepting UI requests.
  const service=new NativeService({executable,build,
    solveTimeoutMs:setting(options.solveTimeoutMs??process.env.ZLC_ATOM_SOLVE_TIMEOUT_MS,120000,10,600000),
    startupTimeoutMs:setting(options.startupTimeoutMs??process.env.ZLC_ATOM_STARTUP_TIMEOUT_MS,10000,100,120000),
    workerIdleMs:setting(options.workerIdleMs??process.env.ZLC_ATOM_WORKER_IDLE_MS,60000,20,600000),
    restartDebounceMs:setting(options.restartDebounceMs??process.env.ZLC_ATOM_RESTART_DEBOUNCE_MS,0,0,1000)});
  let legacySequence=0;
  const server=createServer(async(request,response)=>{
    const origin=request.headers.origin;
    if(origin&&origin!=='null'&&!/^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(origin)){response.writeHead(403);response.end('Local application only');return;}
    if(origin){response.setHeader('Access-Control-Allow-Origin',origin);response.setHeader('Vary','Origin');}
    response.setHeader('Cache-Control','no-store');response.setHeader('X-Content-Type-Options','nosniff');
    if(request.method==='OPTIONS'){response.writeHead(204,{'Access-Control-Allow-Methods':'GET,POST,OPTIONS','Access-Control-Allow-Headers':'Content-Type'});response.end();return;}
    let streaming=false;
    const streamEvent=event=>{if(response.destroyed||response.writableEnded)return;if(!response.headersSent)response.writeHead(200,{'Content-Type':'application/x-ndjson; charset=utf-8'});response.write(JSON.stringify(event)+'\n');};
    const send=(code,value)=>{if(response.destroyed||response.writableEnded)return;if(streaming){streamEvent({type:code===200?'result':'error',status:code,data:value});response.end();return;}response.writeHead(code,{'Content-Type':'application/json; charset=utf-8'});response.end(JSON.stringify(value));};
    let pathname;try{pathname=decodeURIComponent(new URL(request.url,'http://localhost').pathname);}catch{send(400,{error:'Invalid URL'});return;}
    if(pathname==='/api/health'){
      const workers=Object.values(service.lanes).map(lane=>lane.snapshot()),ready=!service.closed&&workers.some(lane=>lane.status==='ready'||lane.status==='busy');
      const negotiated=workers.find(worker=>worker.status==='ready'||worker.status==='busy');
      send(ready?200:503,{ready,accepting:!service.closed,engine:'Rust master equation',protocol:HTTP_PROTOCOL,
        nativeProtocol:negotiated?.nativeProtocol??null,capabilities:negotiated?.capabilities??[],requiredNativeProtocol:NATIVE_PROTOCOL,
        modelSchema:FULL_MODEL_SCHEMA,configuredBinary,build,workers,pending:service.current.size,asymptoticCache:{enabled:false,size:0}});return;
    }
    if(pathname==='/api/solve'&&request.method==='POST'){
      if(!request.headers['content-type']?.startsWith('application/json')){send(415,{error:'JSON required'});return;}
      let task,submitted;const disconnected=()=>{if(!response.writableEnded)task?.cancel();};response.once('close',disconnected);
      try{
        const chunks=[];let bytes=0;for await(const part of request){bytes+=part.length;if(bytes>1024*1024)throw failure('Scene exceeds 1 MiB',413,'invalid-scene');chunks.push(part);}
        if(response.destroyed)return;
        const payload=JSON.parse(Buffer.concat(chunks).toString('utf8'));if(!payload.scene)throw failure('Scene required',400,'invalid-scene');
        submitted=payload;
        if(payload.stream!==undefined&&typeof payload.stream!=='boolean')throw failure('Invalid stream option',400,'invalid-options');
        streaming=payload.stream===true;
        let token=payload.request;
        if(token!==undefined&&(!token||typeof token.clientId!=='string'||!/^[-\w]{1,96}$/.test(token.clientId)||!Number.isSafeInteger(token.sequence)||token.sequence<1))throw failure('Invalid request identity',400,'invalid-request');
        token=token?{clientId:token.clientId,sequence:token.sequence}:{clientId:`legacy-${++legacySequence}`,sequence:1};
        const native=prepareNativeSolve(payload),options=native.payload.options;
        task=service.submit(native.payload,token,streaming?result=>streamEvent({type:'asymptotic',data:native.respond(result,token)}):undefined);
        const result=await task.promise;
        if(options.includeAsymptotic&&!result.asymptoticPopulations)await recordFailure({time:new Date().toISOString(),request:token,scene:payload.scene,options,buildSha256:build.sha256,binarySha256:result.service?.binarySha256??configuredBinary.sha256,status:result.asymptoticStatus,diagnostics:result.diagnostics}).catch(()=>{});
        send(200,native.respond(result,token));
      }catch(error){const evidence={time:new Date().toISOString(),request:submitted?.request,scene:submitted?.scene,options:submitted?.options,buildSha256:build.sha256,binarySha256:configuredBinary.sha256,error:error.message,code:error.code??'invalid-request',httpStatus:error.status??400};if(submitted?.scene&&error.code!=='superseded')await recordFailure(evidence).catch(()=>{});send(error.status??400,{error:error.message,code:error.code??'invalid-request',service:{buildSha256:build.sha256,binarySha256:configuredBinary.sha256}});}
      finally{response.removeListener('close',disconnected);}return;
    }
    if(request.method!=='GET'&&request.method!=='HEAD'){send(405,{error:'Method not allowed'});return;}
    const path=resolve(root,'.'+(pathname==='/'?'/index.html':pathname));
    if(!path.startsWith(root+sep)){send(403,{error:'Forbidden'});return;}
    try{if(!(await stat(path)).isFile())throw new Error('Not a file');const data=await readFile(path);response.writeHead(200,{'Content-Type':mime[extname(path)]||'application/octet-stream'});response.end(request.method==='HEAD'?undefined:data);}
    catch{send(404,{error:'Not found'});}
  });
  server.requestTimeout=15000;
  let closing;
  const close=()=>closing??=Promise.all([service.close(),new Promise(accept=>{server.close(accept);server.closeAllConnections();})]).then(()=>{});
  await Promise.all([service.lanes.steady.start(),service.lanes.trajectory.start()]);
  return {server,service,close};
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const args=process.argv.slice(2),portAt=args.indexOf('--port'),port=Number(portAt>=0?args[portAt+1]:process.env.PORT||4180);
  if(!Number.isInteger(port)||port<1||port>65535)throw new Error('Invalid port.');
  const running=await createAtomServer();
  running.server.on('error',async error=>{console.error(error.message);await running.close();process.exitCode=1;});
  running.server.listen(port,'127.0.0.1',()=>console.log(`Local: http://127.0.0.1:${port}`));
  for(const signal of ['SIGINT','SIGTERM'])process.once(signal,async()=>{await running.close();process.exit(0);});
  process.once('exit',()=>{for(const lane of Object.values(running.service.lanes))lane.worker?.child?.kill();});
}
