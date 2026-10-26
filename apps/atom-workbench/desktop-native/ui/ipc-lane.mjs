import {nativeCapabilities,validateAtomicResult} from '../../scripts/native-protocol.mjs';
import {HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../../dist/js/atom-workbench/src/solver-protocol.js';

const failure=(message,status=503,code='solver-unavailable')=>Object.assign(new Error(message),{status,code});

// Browser Worker-side adapter for the unchanged NativeService scheduler.
// The native host owns processes; this class owns request/generation semantics.
export class IpcNativeLane {
  constructor(owner,name){this.owner=owner;this.name=name;this.worker=null;this.active=null;this.queue=[];this.generation=0;this.error='';this.closed=false;this.pumpTimer=null;this.idleTimer=null;}
  enqueue(job){clearTimeout(this.idleTimer);job.queuedAt=performance.now();this.queue.push(job);this.schedule();}
  schedule(){if(this.closed||this.pumpTimer)return;this.pumpTimer=true;queueMicrotask(()=>{this.pumpTimer=null;void this.pump();});}
  async start(){
    if(this.closed||this.worker)return;
    const w={generation:++this.generation,ready:false,retiring:false,pid:null,capabilities:null,startTimer:null,openSentAt:performance.now(),startup:{}};
    w.exit=new Promise(resolve=>w.exited=resolve);this.worker=w;
    w.startTimer=setTimeout(()=>this.failWorker(w,failure('Native solver startup timed out.')),this.owner.startupTimeoutMs);
    this.owner.sendNative({op:'open',lane:this.name,generation:w.generation});
  }
  write(w,message){this.owner.sendNative({op:'write',lane:this.name,generation:w.generation,message});}
  receive(event){
    const w=this.worker;if(!w||event.generation!==w.generation)return;
    if(event.kind==='closed'){
      clearTimeout(w.startTimer);w.exited();this.worker=null;
      if(!w.retiring){const error=failure('Rust 后端退出。'+(event.error??this.error));if(this.active){const job=this.active;this.active=null;this.owner.finish(job,error);}this.rejectQueue(error);}
      if(this.queue.length)this.schedule();return;
    }
    if(w.retiring)return;
    if(event.kind==='error'){this.failWorker(w,failure(event.error??'Native process communication failed.'));return;}
    if(event.kind==='opened'){
      if(event.binarySha256!==this.owner.binarySha256){this.failWorker(w,failure('Native executable changed.'));return;}
      w.pid=event.pid;w.startup.openRoundTripMs=performance.now()-w.openSentAt;w.startup.hostSpawnMs=event.hostSpawnMs;w.startup.openblasThreads=event.openblasThreads;
      w.capabilitySentAt=performance.now();this.write(w,{op:'capabilities',id:0});return;
    }
    if(event.kind!=='message'){this.failWorker(w,failure('Unexpected native transport event.'));return;}
    const message=event.message;
    if(!w.ready){
      if(!w.capabilities){
        w.startup.capabilitiesRoundTripMs=performance.now()-w.capabilitySentAt;w.startup.capabilitiesNativeIoMs=event.transport?.nativeIoMs;
        try{w.capabilities=nativeCapabilities(message);}catch(error){this.failWorker(w,failure(error.message,503,'incompatible-native-protocol'));return;}
        w.warmSentAt=performance.now();
        this.write(w,{id:-1,op:'warmup'});return;
      }
      if(message.id!==-1||message.error||message.protocol!==NATIVE_PROTOCOL||message.ready!==true){
        this.failWorker(w,failure('Native numerical startup check failed.'));return;
      }
      w.startup.numericalReadyRoundTripMs=performance.now()-w.warmSentAt;w.startup.numericalReadyNativeIoMs=event.transport?.nativeIoMs;w.startup.workerStartupMs=performance.now()-w.openSentAt;
      clearTimeout(w.startTimer);w.ready=true;this.error='';this.owner.verified=true;this.schedule();return;
    }
    const job=this.active;if(!job||message.id!==job.nativeId){this.failWorker(w,failure('Native response identity mismatch.',502,'invalid-native-response'));return;}
    const convert=()=>{
      const result=message.result;
      validateAtomicResult(result,job.payload);
      return {...result,service:{protocol:HTTP_PROTOCOL,nativeProtocol:w.capabilities.protocol,capabilities:w.capabilities.capabilities,lane:this.name,generation:w.generation,binarySha256:this.owner.binarySha256,buildSha256:this.owner.build.sha256,asymptoticCacheHit:false,startup:{...w.startup}}};
    };
    if(message.progress!==undefined){
      try{if(message.progress!=='asymptotic'||!job.onProgress)throw Error('Unexpected native progress message.');job.onProgress(convert());}
      catch(error){this.failWorker(w,failure(error.message,502,'invalid-native-response'));}return;
    }
    this.active=null;
    if(message.error)this.owner.finish(job,failure(message.error,422,'numerical-failure'));
    else{try{this.owner.finish(job,null,convert());}catch(error){this.owner.finish(job,failure(error.message,502,'invalid-native-response'));this.retire();}}
    this.schedule();
  }
  async pump(){
    if(this.closed||this.active||this.worker?.retiring)return;
    this.queue=this.queue.filter(job=>!job.settled);
    if(!this.queue.length){
      if(this.name==='trajectory'&&this.worker?.ready&&!this.idleTimer)this.idleTimer=setTimeout(()=>{this.idleTimer=null;if(!this.active&&!this.queue.length)this.retire();},this.owner.workerIdleMs);
      return;
    }
    if(!this.worker){await this.start();return;}if(!this.worker.ready)return;
    const job=this.queue.shift();if(job.settled){this.schedule();return;}this.active=job;job.startedAt=performance.now();
    this.write(this.worker,{id:job.nativeId,...job.payload,progress:!!job.onProgress});
  }
  cancel(job,error){this.queue=this.queue.filter(item=>item!==job);if(this.active===job){this.active=null;this.retire();}this.owner.finish(job,error);this.schedule();}
  rejectQueue(error){for(const job of this.queue.splice(0))this.owner.finish(job,error);}
  failWorker(w,error){
    if(this.worker!==w||w.retiring)return;this.error=error.message;
    if(this.active){const job=this.active;this.active=null;this.owner.finish(job,error);}this.rejectQueue(error);this.retire();
  }
  retire(){clearTimeout(this.idleTimer);this.idleTimer=null;const w=this.worker;if(!w||w.retiring)return;w.retiring=true;w.ready=false;clearTimeout(w.startTimer);this.owner.sendNative({op:'close',lane:this.name,generation:w.generation});}
  async close(){
    this.closed=true;clearTimeout(this.idleTimer);const error=failure('Local solver service is shutting down.');
    if(this.active){const job=this.active;this.active=null;this.owner.finish(job,error);}this.rejectQueue(error);const w=this.worker;this.retire();
    if(w)await Promise.race([w.exit,new Promise(resolve=>setTimeout(resolve,1500))]);
  }
}
