// Shared request-local scheduling for Node and native IPC transports.
import {hasVerifiedFreshMean,mergeFreshMean} from "./fresh-mean.mjs";
import {EFFECTIVE_MODEL_SCHEMA as ATOMIC_MODEL_SCHEMA} from "../dist/js/atom-workbench/src/solver-protocol.js";
const failure=(message,status=503,code="solver-unavailable")=>Object.assign(new Error(message),{status,code});
const cancelled=()=>failure("已取消被更新参数替代的求解。",409,"superseded");
export function parallelMeanSettings(env){
  const mode=env.ATOM_FULL_PERIOD,monodromy=env.ATOM_MONODROMY==='1';
  return {parallelMean:env.ATOM_PARALLEL_MEAN==='1'&&(mode==='1'||mode==='auto'||monodromy),
    parallelMeanAuto:mode==='auto'&&!monodromy,
    longFourierSerial:env.ATOM_LONG_FOURIER_SERIAL==='1'&&mode==='auto'&&env.ATOM_MONODROMY===undefined&&env.ATOM_FIXED_ETD===undefined};
}

// Match the native auto full-period gate. Long horizons remain a single
// Floquet job, which computes its own mean instead of duplicating that work.
export function shortPeriodAutoEligible(payload){
  const {model,options}=payload;
  if(model.dimension!==32||options.steadyOnly||model.drives?.length!==1||!Number.isFinite(options.durationUs)||options.durationUs<=0)return false;
  const omega=model.drives[0].omega;
  if(!Number.isFinite(omega)||omega<=0)return false;
  const period=2*Math.PI/omega;
  return period<=0.02&&options.durationUs<=100;
}

// Thread/lane selection only. The entire finite payload, samples and physical
// operator are unchanged; sparse long-time work avoids an eight-thread BLAS
// launch per small operation. The existing steady lane accepts finite jobs.
export function serialLongFourierEligible(payload){
  const {model,options}=payload;
  return model.dimension===32&&!options.steadyOnly&&model.drives?.length===1&&
    Number.isFinite(model.drives[0].omega)&&model.drives[0].omega>0&&
    Number.isFinite(options.durationUs)&&options.durationUs>0&&Number.isSafeInteger(options.samples)&&options.samples>=2&&
    options.durationUs/(options.samples-1)>10&&!shortPeriodAutoEligible(payload);
}


export function createNativeServiceClass(NativeLane){
return class NativeService {
  constructor(options){Object.assign(this,options);this.sequence=0;this.closed=false;this.verified=false;this.current=new Map();this.seen=new Map();this.lanes={steady:new NativeLane(this,'steady'),trajectory:new NativeLane(this,'trajectory')};}
  submit(payload,token,onProgress){
    if(this.closed)return {promise:Promise.reject(failure('Service is shutting down.')),cancel:()=>{}};
    const previousSequence=this.seen.get(token.clientId);
    if(previousSequence!==undefined&&token.sequence<=previousSequence)return {promise:Promise.reject(cancelled()),cancel:()=>{}};
    const previous=this.current.get(token.clientId);if(previous)this.cancel(previous,cancelled());
    if(this.current.size>=8)return {promise:Promise.reject(failure('求解请求过多，请稍后重试。',429,'queue-full')),cancel:()=>{}};
    this.seen.delete(token.clientId);this.seen.set(token.clientId,token.sequence);
    if(this.seen.size>128)for(const id of this.seen.keys())if(!this.current.has(id)){this.seen.delete(id);break;}
    if(this.parallelMean&&payload.modelSchema===ATOMIC_MODEL_SCHEMA&&payload.model.dimension===32&&
      !payload.options.steadyOnly&&payload.options.includeAsymptotic===true&&payload.model.drives?.length===1&&
      Number.isFinite(payload.model.drives[0].omega)&&payload.model.drives[0].omega>0&&
      (!this.parallelMeanAuto||shortPeriodAutoEligible(payload))){
      return this.submitParallel(payload,token,onProgress);
    }
    const lane=this.lanes[payload.options.steadyOnly||(this.longFourierSerial&&serialLongFourierEligible(payload))?'steady':'trajectory'];
    let job;const promise=new Promise((resolve,reject)=>{
      job={token,nativeId:++this.sequence,payload,lane,resolve,reject,onProgress,settled:false,timer:null};this.current.set(token.clientId,job);
      job.timer=setTimeout(()=>lane.cancel(job,failure('求解超时，已终止对应 Rust 进程；可修改参数后重试。',504,'solver-timeout')),this.solveTimeoutMs);
      lane.enqueue(job);
    });
    return {promise,cancel:()=>{if(!job.settled)lane.cancel(job,cancelled());}};
  }
  submitParallel(payload,token,onProgress){
    let parent;
    const promise=new Promise((resolve,reject)=>{
      parent={token,payload,onProgress,resolve,reject,settled:false,children:[],outcomes:{},startedAt:performance.now(),timer:null};
      this.current.set(token.clientId,parent);
      const add=(role,options)=>{
        const lane=this.lanes[role==='mean'?'steady':'trajectory'];
        const child={token,nativeId:++this.sequence,payload:{...payload,options},lane,role,settled:false,timer:null,
          resolve:value=>this.finishParallelChild(parent,role,null,value),reject:error=>this.finishParallelChild(parent,role,error)};
        parent.children.push(child);lane.enqueue(child);
      };
      parent.timer=setTimeout(()=>this.cancel(parent,failure('求解超时，已终止对应 Rust 进程；可修改参数后重试。',504,'solver-timeout')),this.solveTimeoutMs);
      // One immutable model object is serialized to each child; no second
      // atomic model construction and no history/result lookup occur here.
      add('mean',{...payload.options,steadyOnly:true,includeAsymptotic:true});
      add('trajectory',{...payload.options,steadyOnly:false,includeAsymptotic:false});
    });
    return {promise,cancel:()=>this.cancel(parent,cancelled())};
  }
  finishParallelChild(parent,role,error,value){
    if(parent.settled||this.current.get(parent.token.clientId)!==parent)return;
    parent.outcomes[role]={error,value};
    try{
      if(role==='mean'){
        if(error){this.cancel(parent,error);return;}
        if(hasVerifiedFreshMean(value,parent.payload.model))parent.onProgress?.(value);
      }
      const mean=parent.outcomes.mean,trajectory=parent.outcomes.trajectory;
      if(!mean||!trajectory)return;
      // A failed finite solve still lets this request's verified mean arrive,
      // matching the existing mean-then-error stream behavior.
      if(trajectory.error){this.finish(parent,trajectory.error);return;}
      const timing=role=>{const job=parent.children.find(child=>child.role===role),outcome=parent.outcomes[role];return {
        queueMs:Math.max(0,(job.startedAt??job.finishedAt)-(job.queuedAt??parent.startedAt)),
        wallMs:job.finishedAt-(job.startedAt??job.queuedAt??parent.startedAt),nativeMs:outcome.value?.diagnostics?.elapsedMs};};
      const result=mergeFreshMean(trajectory.value,mean.value,parent.payload.model,{
        elapsedMs:performance.now()-parent.startedAt,meanTiming:timing('mean'),trajectoryTiming:timing('trajectory')});
      this.finish(parent,null,result);
    }catch(error){this.cancel(parent,failure(error.message,502,'invalid-native-response'));}
  }
  cancel(job,error){
    if(job.settled)return;
    if(!job.children){job.lane.cancel(job,error);return;}
    // Settle the parent FIRST so no synchronously rejected child can stream a
    // stale mean while cancellation removes queued work or retires a worker.
    this.finish(job,error);
    for(const child of job.children)if(!child.settled)child.lane.cancel(child,error);
  }
  finish(job,error,value){if(job.settled)return;job.settled=true;job.finishedAt=performance.now();clearTimeout(job.timer);if(this.current.get(job.token.clientId)===job)this.current.delete(job.token.clientId);if(error)job.reject(error);else job.resolve(value);}
  async close(){this.closed=true;for(const job of [...this.current.values()])this.cancel(job,failure('Local solver service is shutting down.'));await Promise.all(Object.values(this.lanes).map(lane=>lane.close()));}
}
;
}
