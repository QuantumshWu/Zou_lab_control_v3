// Shared request-local scheduling for Node and native IPC transports.
const failure=(message,status=503,code="solver-unavailable")=>Object.assign(new Error(message),{status,code});
const cancelled=()=>failure("已取消被更新参数替代的求解。",409,"superseded");

export function createNativeServiceClass(NativeLane){
return class NativeService {
  constructor(options){Object.assign(this,options);this.sequence=0;this.closed=false;this.verified=false;this.current=new Map();this.seen=new Map();this.lanes={mean:new NativeLane(this,'mean'),trajectory:new NativeLane(this,'trajectory')};}
  submit(payload,token,onProgress){
    if(this.closed)return {promise:Promise.reject(failure('Service is shutting down.')),cancel:()=>{}};
    const previousSequence=this.seen.get(token.clientId);
    if(previousSequence!==undefined&&token.sequence<=previousSequence)return {promise:Promise.reject(cancelled()),cancel:()=>{}};
    const previous=this.current.get(token.clientId);if(previous)this.cancel(previous,cancelled());
    if(this.current.size>=8)return {promise:Promise.reject(failure('求解请求过多，请稍后重试。',429,'queue-full')),cancel:()=>{}};
    this.seen.delete(token.clientId);this.seen.set(token.clientId,token.sequence);
    if(this.seen.size>128)for(const id of this.seen.keys())if(!this.current.has(id)){this.seen.delete(id);break;}
    // Requests are serial per lane: meanOnly means run on the mean lane,
    // finite trajectories (with their inline mean) on the trajectory lane.
    const lane=this.lanes[payload.options.meanOnly?'mean':'trajectory'];
    let job;const promise=new Promise((resolve,reject)=>{
      job={token,nativeId:++this.sequence,payload,lane,resolve,reject,onProgress,settled:false,timer:null};this.current.set(token.clientId,job);
      job.timer=setTimeout(()=>lane.cancel(job,failure('求解超时，已终止对应 Rust 进程；可修改参数后重试。',504,'solver-timeout')),this.solveTimeoutMs);
      lane.enqueue(job);
    });
    return {promise,cancel:()=>{if(!job.settled)lane.cancel(job,cancelled());}};
  }
  cancel(job,error){
    if(job.settled)return;
    job.lane.cancel(job,error);
  }
  finish(job,error,value){if(job.settled)return;job.settled=true;job.finishedAt=performance.now();clearTimeout(job.timer);if(this.current.get(job.token.clientId)===job)this.current.delete(job.token.clientId);if(error)job.reject(error);else job.resolve(value);}
  async close(){this.closed=true;for(const job of [...this.current.values()])this.cancel(job,failure('Local solver service is shutting down.'));await Promise.all(Object.values(this.lanes).map(lane=>lane.close()));}
}
;
}
