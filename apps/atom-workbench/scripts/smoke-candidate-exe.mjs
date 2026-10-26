// No browser/WebView/window is opened. The real model Worker and IPC lane
// communicate with private children of the candidate EXE over its JSONL API.
import assert from 'node:assert/strict';
import {spawn,execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {readFile,writeFile,mkdir,stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve,dirname} from 'node:path';
import {createModelWorker} from '../desktop-native/ui/model-worker.mjs';
import {DESKTOP_SOLVER_DEFAULTS} from './desktop-solver-settings.mjs';
import {nativeChildEnvironment} from './native-child-env.mjs';
import {defaultScene,resolvePopulationTolerance} from '../dist/js/atom-workbench/src/types.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {normalizeDynamicsResult} from '../dist/js/atom-workbench/src/solver-client.js';
import {EFFECTIVE_MODEL_SCHEMA as SCHEMA} from '../dist/js/atom-workbench/src/solver-protocol.js';

const option=name=>{const i=process.argv.indexOf(name);return i<0?undefined:process.argv[i+1];};
if(process.argv.includes('--help')){
  console.log('node scripts/smoke-candidate-exe.mjs --exe <candidate.exe> --out <fresh-report.json> [--request <HTTP-scene-request.json>] [--timeout-ms 120000]');
  console.log('Default: fresh single-near-laser scene, 0.02 us, 3 samples, epsilon=.005. No GUI or answer cache.');
  process.exit(0);
}
if(!option('--exe')||!option('--out'))throw Error('--exe and a fresh --out path are required');
const exe=resolve(option('--exe')),out=resolve(option('--out')),timeout=Number(option('--timeout-ms')??120000);
if(!Number.isSafeInteger(timeout)||timeout<1000||timeout>600000)throw Error('Invalid smoke timeout');
try{await stat(out);throw Error('Smoke report already exists; choose a fresh path');}catch(error){if(error.code!=='ENOENT')throw error;}
const sha=data=>createHash('sha256').update(data).digest('hex'),binary=await readFile(exe),binarySha256=sha(binary);
let scene=defaultScene(),settings;
if(option('--request')){
  const value=JSON.parse(await readFile(resolve(option('--request')),'utf8'));
  if(!value.scene)throw Error('--request must contain a current HTTP scene request, not a stored numerical answer');
  scene=validateScene(value.scene);settings={...scene.dynamics,...value.options};
}else{
  scene.lasers=scene.lasers.filter(laser=>(laser.regime??'near')==='near');
  settings={...scene.dynamics,durationUs:.02,samples:3,populationTolerance:.005};
}
settings.populationTolerance=resolvePopulationTolerance(settings.populationTolerance);
const baseEnv={...process.env};
// Use the release defaults, not inherited research experiment switches.
for(const key of Object.keys(baseEnv))if(key.toUpperCase().startsWith('ATOM_'))delete baseEnv[key];
Object.assign(baseEnv,DESKTOP_SOLVER_DEFAULTS);
const report={status:'starting',executable:exe,binarySha256,bytes:binary.length,windowOpened:false,
  scope:'Candidate EXE/backend-info plus normal Worker/IPC/JSONL finite and mean requests; no independent physics oracle',
  requestSource:option('--request')?resolve(option('--request')):'Generated single-near-laser smoke fixture',
  requestSha256:sha(JSON.stringify({scene,settings})),options:settings,processes:[],workerRequests:[],historicalAnswerRead:false};
const children=new Map(),pending=new Map();let worker,sequence=0;
const forward=(command,event)=>Promise.resolve(worker.receive({kind:'native',event:{lane:command.lane,generation:command.generation,...event}})).catch(error=>{
  for(const wait of pending.values())wait.reject(error);
});
function host(command){
  const key=`${command.lane}:${command.generation}`;
  if(command.op==='open'){
    const child=spawn(exe,['--worker'],{windowsHide:true,stdio:['pipe','pipe','pipe'],env:nativeChildEnvironment(command.lane,baseEnv)});
    const record={lane:command.lane,generation:command.generation,pid:child.pid,args:['--worker'],stderr:'',exitCode:null};
    report.processes.push(record);children.set(key,child);let buffer='';
    child.once('spawn',()=>void forward(command,{kind:'opened',pid:child.pid,binarySha256}));
    child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
    child.stderr.on('data',chunk=>record.stderr=(record.stderr+chunk).slice(-12000));
    child.stdout.on('data',chunk=>{
      buffer+=chunk;
      if(buffer.length>64*1024*1024){void forward(command,{kind:'error',error:'JSONL response exceeded smoke buffer limit'});child.kill();return;}
      for(let end;(end=buffer.indexOf('\n'))>=0;){
        const line=buffer.slice(0,end).trim();buffer=buffer.slice(end+1);if(!line)continue;
        try{void forward(command,{kind:'message',message:JSON.parse(line)});}
        catch(error){void forward(command,{kind:'error',error:'Invalid worker JSONL: '+error.message});child.kill();}
      }
    });
    child.once('error',error=>void forward(command,{kind:'error',error:error.message}));
    child.stdin.on('error',error=>void forward(command,{kind:'error',error:error.message}));
    child.once('close',(code,signal)=>{record.exitCode=code;record.signal=signal;record.closed=true;children.delete(key);void forward(command,{kind:'closed',error:code?record.stderr:undefined});});
  }else if(command.op==='write'){
    const child=children.get(key);if(!child)throw Error('IPC write has no matching owned process');
    if(command.message.id>0)report.workerRequests.push({lane:command.lane,id:command.message.id,options:command.message.options,
      modelSha256:sha(JSON.stringify(command.message.model)),model:command.message.model});
    child.stdin.write(JSON.stringify(command.message)+'\n');
  }else if(command.op==='close')children.get(key)?.kill();
  else if(command.op==='close-all')for(const child of children.values())child.kill();
  else throw Error('Unknown native host command: '+command.op);
}
function request(kind,options){
  const channel=`smoke-${kind}-${++sequence}`,token={clientId:'candidate-exe-smoke',sequence};
  return new Promise((resolveReply,reject)=>{
    const timer=setTimeout(()=>{void worker.receive({kind:'cancel',channel});pending.delete(channel);reject(Error(`${kind} smoke request timed out; no retry started`));},timeout);
    pending.set(channel,{resolve:data=>{clearTimeout(timer);pending.delete(channel);resolveReply(data);},reject:error=>{clearTimeout(timer);pending.delete(channel);reject(error);}});
    void worker.receive({kind:'solve',channel,request:{scene,modelSchema:SCHEMA,request:token,options,stream:false}});
  });
}
const began=performance.now();
try{
  const info=await promisify(execFile)(exe,['--backend-info'],{windowsHide:true,env:baseEnv,timeout:Math.min(timeout,15000),maxBuffer:8*1024*1024});
  report.backendInfo=JSON.parse(info.stdout.trim());assert.equal(report.backendInfo.staticSolver,true);
  assert.ok(report.backendInfo.numericSource&&report.backendInfo.v3IconSha256,'Embedded solver source and v3 icon provenance are required');
  worker=createModelWorker(message=>{
    if(message.kind==='native'){host(message.command);return;}
    if(message.kind==='response'){
      const wait=pending.get(message.channel);if(!wait)return;
      if(message.status!==200)wait.reject(Error(message.data?.error??`IPC status ${message.status}`));else wait.resolve(message.data);
    }
  });
  await worker.receive({kind:'native',event:{kind:'configuration',binarySha256,
    buildSha256:sha(JSON.stringify(report.backendInfo.numericSource)),defaults:DESKTOP_SOLVER_DEFAULTS}});
  const finite=await request('finite',{...settings,steadyOnly:false,includeAsymptotic:true});
  const parsed=normalizeDynamicsResult(finite,scene,settings.initialStateId);
  assert.equal(finite.diagnostics.validationKind,'population');
  assert.equal(parsed.populationAccuracy.requestedTolerance,settings.populationTolerance);
  assert.equal(parsed.timesUs.length,settings.samples);assert.equal(parsed.timesUs.at(-1),settings.durationUs);
  assert.equal(parsed.steady.converged,true,'The finite request carries its qualified inline mean');
  assert.equal(Object.hasOwn(parsed.validation,'positivityError'),false,'Population output must not fabricate a density PSD guarantee');
  const mean=await request('mean',{...settings,steadyOnly:true,includeAsymptotic:true});
  const meanParsed=normalizeDynamicsResult(mean,scene,settings.initialStateId);
  assert.equal(meanParsed.steady.converged,true,'Smoke mean must be qualified, never the finite endpoint');
  assert.equal(report.workerRequests.length,2);assert.deepEqual(report.workerRequests.map(r=>r.lane),['trajectory','steady']);
  assert.equal(report.workerRequests[0].modelSha256,report.workerRequests[1].modelSha256,'Both fresh results must use the identical current Model');
  const meanDelta=Math.max(...finite.asymptoticPopulations.map((p,i)=>Math.abs(p-mean.asymptoticPopulations[i])));
  assert.ok(meanDelta<1e-9,`Inline and steadyOnly means solve the same equation (difference ${meanDelta})`);
  report.status='passed';report.validation={backendStatic:true,normalIpcAndJsonl:true,finitePopulationAccuracy:parsed.populationAccuracy,
    meanStatus:meanParsed.steady.status,finiteNativeMs:finite.diagnostics.elapsedMs,meanNativeMs:mean.diagnostics.elapsedMs};
}catch(error){report.status='failed';report.error=error.stack??error.message;process.exitCode=1;}
finally{
  await worker?.receive({kind:'shutdown'});
  for(const child of children.values())child.kill();
  for(let i=0;i<100&&children.size;i++)await new Promise(resolveWait=>setTimeout(resolveWait,20));
  report.allOwnedWorkersClosed=children.size===0;report.wallMs=performance.now()-began;
  if(children.size){report.status='failed';process.exitCode=1;}
  for(const request of report.workerRequests)delete request.model;
  await mkdir(dirname(out),{recursive:true});await writeFile(out,JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({status:report.status,binarySha256,wallMs:report.wallMs,allOwnedWorkersClosed:report.allOwnedWorkersClosed,report:out}));
}
