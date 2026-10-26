import {buildFullMasterModel} from '../../dist/js/atom-workbench/src/full-master-model.js';
import {DEFAULT_DYNAMICS_SAMPLES,resolvePopulationTolerance} from '../../dist/js/atom-workbench/src/types.js';
import {EFFECTIVE_MODEL_SCHEMA as ATOMIC_MODEL_SCHEMA} from '../../dist/js/atom-workbench/src/solver-protocol.js';
import {createNativeServiceClass} from '../../scripts/native-service-core.mjs';
import {IpcNativeLane} from './ipc-lane.mjs';
const NativeService=createNativeServiceClass(IpcNativeLane);

export function createModelWorker(post){
  let service,closed=false,unavailable;const active=new Map(),pending=[];
  const sendNative=command=>post({kind:'native',command});
  const finish=(channel,status,data)=>post({kind:'response',channel,status,data});
  async function solve(channel,input){
    if(closed){finish(channel,503,{error:'Native bridge is closed.',code:'solver-unavailable'});return;}
    if(unavailable){finish(channel,503,{error:unavailable,code:'solver-unavailable'});return;}
    if(!service){if(pending.length>=8)finish(channel,503,{error:'Native bridge is not ready.',code:'solver-unavailable'});else pending.push({channel,input});return;}
    let task,streaming=false,token;
    const event=(type,data,status)=>{streaming=true;post({kind:'stream',channel,event:{type,data,...(status===undefined?{}:{status})},done:type!=='asymptotic'});};
    try{
      if(!input?.scene||typeof input.scene!=='object')throw Error('Scene required');
      if(input.modelSchema!==undefined&&input.modelSchema!==ATOMIC_MODEL_SCHEMA)throw Error('Unsupported atomic model schema.');
      token=input.request;
      if(!token||typeof token.clientId!=='string'||!/^[-\w]{1,96}$/.test(token.clientId)||!Number.isSafeInteger(token.sequence)||token.sequence<1)throw Error('Invalid request identity');
      if(input.stream!==undefined&&typeof input.stream!=='boolean')throw Error('Invalid stream option');
      const settings={...input.scene.dynamics,...input.options};const populationTolerance=resolvePopulationTolerance(settings.populationTolerance);const modelStarted=performance.now();
      const model=buildFullMasterModel(input.scene,settings),modelBuildMs=performance.now()-modelStarted;
      const options={durationUs:settings.durationUs??5,samples:settings.samples??DEFAULT_DYNAMICS_SAMPLES,populationTolerance,includeAsymptotic:input.options?.includeAsymptotic??true,steadyOnly:input.options?.steadyOnly??false};
      if(typeof options.steadyOnly!=='boolean'||typeof options.includeAsymptotic!=='boolean')throw Error('Invalid solver options');
      const payload={modelSchema:model.modelSchema,radiation:model.radiation,model:{dimension:model.dimension,hReal:model.hReal,hImag:model.hImag,collapse:model.collapse,initialIndex:model.initialIndex,timeDependent:model.timeDependent,drives:model.drives,frameFrequenciesMHz:model.frameFrequenciesMHz,frequencyCoordinates:model.frequencyCoordinates,states:model.states.map(state=>state.id)},options};
      const decorate=result=>({...result,request:token,states:model.states,service:{...result.service,modelBuildMs,modelCacheHit:model.buildCacheHit},engine:'Rust master equation',assumptions:model.assumptions,decayChannels:model.decayChannels});
      task=service.submit(payload,token,input.stream?result=>event('asymptotic',decorate(result)):undefined);active.set(channel,task);
      const result=await task.promise;
      if(active.get(channel)!==task)return;
      const data=decorate(result);if(input.stream)event('result',data);else finish(channel,200,data);
    }catch(error){
      if(task&&active.get(channel)!==task)return;
      const status=error.status??400,data={error:error.message,code:error.code??'invalid-request',request:token};
      if(streaming)event('error',data,status);else finish(channel,status,data);
    }finally{if(active.get(channel)===task)active.delete(channel);}
  }
  async function receive(message){
    if(message.kind==='native'){
      const event=message.event;
      if(event.kind==='fatal'){
        unavailable=event.error??'Native bridge is unavailable.';
        for(const item of pending.splice(0))finish(item.channel,503,{error:unavailable,code:'solver-unavailable'});
        await service?.close();return;
      }
      if(event.kind==='configuration'){
        if(service)return;
        service=new NativeService({binarySha256:event.binarySha256,build:{sha256:event.buildSha256},solveTimeoutMs:120000,startupTimeoutMs:15000,workerIdleMs:60000,sendNative});
        // Match createAtomServer: initialize both native lanes independently
        // of the first user solve, without blocking UI/static spectroscopy.
        for(const lane of Object.values(service.lanes))void lane.start();
        for(const item of pending.splice(0))void solve(item.channel,item.input);return;
      }
      service?.lanes[event.lane]?.receive(event);return;
    }
    if(message.kind==='solve'){void solve(message.channel,message.request);return;}
    if(message.kind==='cancel'){
      for(let i=pending.length-1;i>=0;i--)if(pending[i].channel===message.channel)pending.splice(i,1);
      const task=active.get(message.channel);active.delete(message.channel);task?.cancel();return;
    }
    if(message.kind==='shutdown'){
      closed=true;pending.length=0;for(const task of active.values())task.cancel();active.clear();await service?.close();
      post({kind:'shutdown-complete'});
    }
  }
  return {receive};
}

if(typeof WorkerGlobalScope!=='undefined'&&globalThis instanceof WorkerGlobalScope){
  const worker=createModelWorker(message=>postMessage(message));
  onmessage=event=>{worker.receive(event.data).catch(error=>setTimeout(()=>{throw error;}));};
}
