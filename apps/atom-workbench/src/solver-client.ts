import type {DynamicsResult,SteadyStatus,PopulationAccuracy} from './solver-types.js';
import {CONVERGED_MEAN_STATUSES,STEADY_STATUSES,isInitialDependent,readPopulationAccuracy} from './solver-types.js';
import type {DynamicsSettings,InitialStateSpec,Scene,Level} from './types.js';
import {MEAN_HORIZONS_US,POPULATION_TOLERANCE,canonicalInitialState,resolveInitialState,resolveMeanHorizon} from './types.js';
import {initialPopulations} from './full-master-model.js';
import {laserIntensityMWcm2} from './physics.js';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from './solver-protocol.js';
import {readMeanDensity} from './mean-density.js';

export interface SolverFailureRecord {time:string;request:{clientId:string;sequence:number};scene:Scene;options:object;attempt:number;willRetry:boolean;error:string;status?:number;code?:string;response?:unknown;page?:string;}
class RequestFailure extends Error {constructor(message:string,readonly status:number,readonly code:string,readonly retryable:boolean){super(message);}}
type Job={id:number;attempt:number;scene:Scene;settings:DynamicsSettings;steadyOnly:boolean;controller:AbortController;
  resolve:(value:DynamicsResult)=>void;reject:(error:Error)=>void;onProgress?:(value:DynamicsResult)=>void;started:number;timer?:ReturnType<typeof setTimeout>};
type ClientOptions={baseUrl?:string;debounceMs?:number;fetch?:typeof fetch;clientId?:string;onFailure?:(record:SolverFailureRecord)=>void};

/** The production solver is a native Rust process. QuTiP is an offline oracle only. */
export class DynamicsClient {
  private sequence=0;private current:Job|null=null;private disposed=false;
  private readonly base:string;private readonly debounceMs:number;private readonly transport:typeof fetch;private readonly clientId:string;
  lastRoundTripMs=0;
  private readonly onFailure:ClientOptions['onFailure'];
  constructor(options:ClientOptions={}){
    this.onFailure=options.onFailure;
    this.base=options.baseUrl??(globalThis.location?.protocol==='file:'?'http://127.0.0.1:4180':'');
    this.debounceMs=options.debounceMs??60;this.transport=options.fetch??globalThis.fetch.bind(globalThis);
    this.clientId=options.clientId??globalThis.crypto.randomUUID();
    if(!Number.isFinite(this.debounceMs)||this.debounceMs<0)throw new Error('Invalid solver debounce interval.');
  }
  solve(scene:Scene,settings:DynamicsSettings,control:{steadyOnly?:boolean;onProgress?:(value:DynamicsResult)=>void}={}):Promise<DynamicsResult>{
    if(this.disposed)return Promise.reject(new DOMException('Disposed.','AbortError'));
    this.abortCurrent('Superseded.');
    const id=++this.sequence;
    return new Promise((resolve,reject)=>{
      const job:Job={id,attempt:0,scene:structuredClone(scene),settings:structuredClone(settings),steadyOnly:control.steadyOnly??false,
        controller:new AbortController(),resolve,reject,onProgress:control.onProgress,started:performance.now()};
      this.current=job;job.timer=setTimeout(()=>{job.timer=undefined;void this.run(job);},this.debounceMs);
    });
  }
  private async run(job:Job):Promise<void>{
    if(this.current!==job||job.controller.signal.aborted)return;
    job.attempt++;let evidence:unknown;
    try{
      const response=await this.transport(this.base+'/api/solve',{method:'POST',signal:job.controller.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({modelSchema:FULL_MODEL_SCHEMA,stream:!!job.onProgress,request:{clientId:this.clientId,sequence:job.id},scene:job.scene,options:{
        initialStateId:job.settings.initialStateId,initialState:canonicalInitialState(resolveInitialState(job.settings,job.scene)),durationUs:job.settings.durationUs,samples:job.settings.samples,
        populationTolerance:POPULATION_TOLERANCE,meanHorizonUs:resolveMeanHorizon(job.settings),
        includeAsymptotic:true,steadyOnly:job.steadyOnly,
      }})}).catch(error=>{if(error instanceof TypeError)throw new RequestFailure('本机 Rust 求解服务连接暂时失败。',0,'network-error',true);throw error;});
      if(!response.ok){const detail=await response.json().catch(()=>({error:''}));evidence=detail;throw new RequestFailure(detail.error||`Local solver: ${response.status}`,response.status,detail.code??'',(response.status===502||response.status===503)&&!/invalid|incompatible|numerical/.test(detail.code??''));}
      const data=await this.readResponse(response,job);if(!data||typeof data!=='object'||Array.isArray(data))throw new Error('求解结果不完整或无效：响应对象。');
      evidence={protocol:data.protocol,modelSchema:data.modelSchema,service:data.service,diagnostics:data.diagnostics,asymptoticStatus:data.asymptoticStatus};if(data.error)throw new Error(data.error);
      if(this.current!==job||job.controller.signal.aborted)return;
      if(data.service?.protocol!==HTTP_PROTOCOL||data.service?.nativeProtocol!==NATIVE_PROTOCOL||
        !Array.isArray(data.service?.capabilities)||!data.service.capabilities.includes(FULL_MODEL_SCHEMA))throw new Error('本机求解服务版本不兼容：需要 rb87-full/2 模型，旧模型结果未显示。');
      if(!data.request||data.request.clientId!==this.clientId||data.request.sequence!==job.id)throw new Error('求解响应与当前请求不匹配。');
      const result=normalizeDynamicsResult(data,job.scene,resolveInitialState(job.settings,job.scene));
      this.checkHorizon(result,job);
      const expected=job.steadyOnly?1:job.settings.samples;
      if(result.timesUs.length!==expected||(!job.steadyOnly&&Math.abs(result.timesUs.at(-1)!-job.settings.durationUs)>1e-10*Math.max(1,job.settings.durationUs)))throw new Error('求解响应的采样与请求不匹配。');
      if(!result.steady?.converged)this.recordFailure(job,new Error(result.steady?.reason??'长期布居未解析'),false,evidence);
      this.current=null;this.lastRoundTripMs=performance.now()-job.started;job.resolve(result);
    }catch(error){
      if(this.current!==job)return;
      if(!job.controller.signal.aborted){
        const retry=error instanceof RequestFailure&&error.retryable&&job.attempt<2;
        this.recordFailure(job,error,retry,evidence);
        if(retry){job.id=++this.sequence;job.timer=setTimeout(()=>{job.timer=undefined;void this.run(job);},200);return;}
      }
      this.current=null;
      job.reject(job.controller.signal.aborted?new DOMException('Cancelled.','AbortError'):new Error(error instanceof TypeError?'本机 Rust 求解服务未连接。请从 http://127.0.0.1:4180 打开工作台。':error instanceof Error?error.message:String(error)));
    }
  }
  private async readResponse(response:Response,job:Job):Promise<any>{
    if(!response.headers?.get('Content-Type')?.includes('application/x-ndjson'))return response.json();
    if(!response.body)throw new Error('求解流为空。');
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',final:any;
    const consume=(line:string)=>{if(!line.trim())return;const event=JSON.parse(line);
      if(event.type==='error')throw new RequestFailure(event.data?.error??'时间演化失败。',event.status??422,event.data?.code??'numerical-failure',false);
      if(final!==undefined)throw new Error('求解流包含多余结果。');
      if(event.type==='result'){final=event.data;return;}
      if(event.type!=='asymptotic')throw new Error('未知求解进度。');
      if(this.current!==job||job.controller.signal.aborted)return;
      const data=event.data;
      if(data?.request?.clientId!==this.clientId||data.request.sequence!==job.id||data.service?.protocol!==HTTP_PROTOCOL||data.service?.nativeProtocol!==NATIVE_PROTOCOL||!data.service?.capabilities?.includes(FULL_MODEL_SCHEMA))throw new Error('稳态进度与当前请求不匹配。');
      const mean=normalizeDynamicsResult(data,job.scene,resolveInitialState(job.settings,job.scene));
      this.checkHorizon(mean,job);
      if(mean.timesUs.length!==1||!mean.steady.converged)throw new Error('稳态进度缺少已验证的长期布居。');
      job.onProgress?.(mean);
    };
    try{while(true){const {done,value}=await reader.read();buffer+=decoder.decode(value,{stream:!done});if(buffer.length>64*1024*1024)throw new Error('求解流超过大小限制。');let end;while((end=buffer.indexOf('\n'))>=0){consume(buffer.slice(0,end));buffer=buffer.slice(end+1);}if(done)break;}if(buffer.trim())consume(buffer);}
    finally{reader.releaseLock();}
    if(final===undefined)throw new Error('时间演化响应提前结束；已收到的稳态结果仍可查看。');return final;
  }
  private recordFailure(job:Job,error:unknown,willRetry:boolean,response:unknown):void{
    try{this.onFailure?.({time:new Date().toISOString(),request:{clientId:this.clientId,sequence:job.id},scene:structuredClone(job.scene),options:{...job.settings,steadyOnly:job.steadyOnly,includeAsymptotic:true},attempt:job.attempt,willRetry,error:error instanceof Error?error.message:String(error),...(error instanceof RequestFailure?{status:error.status,code:error.code}:{}),response,page:globalThis.location?.href});}catch{/* Diagnostics cannot block or replace a physics result. */}
  }
  private checkHorizon(result:DynamicsResult,job:Job):void{
    if(result.steady.horizonUs!==undefined&&result.steady.horizonUs!==resolveMeanHorizon(job.settings))throw new Error('求解响应的长期布居观察时长与本次请求不匹配。');
  }
  private abortCurrent(reason:string):void{
    const job=this.current;if(!job)return;this.current=null;clearTimeout(job.timer);job.controller.abort();job.reject(new DOMException(reason,'AbortError'));
  }
  cancel():void{++this.sequence;this.abortCurrent('Cancelled.');}
  dispose():void{this.disposed=true;this.cancel();}
}

/** Reads the one native result contract (documented in
 * rust-solver/src/lattice_core_adapter.rs) and rejects anything else.
 * `initial` is the requested initial state (an id is a single state). */
export function normalizeDynamicsResult(data:any,scene:Scene,initial:string|InitialStateSpec=resolveInitialState(scene.dynamics,scene)):DynamicsResult{
  const invalid=(detail:string):never=>{throw new Error(`求解结果不完整或无效：${detail}；未绘制替代数据。`);};
  const bounded=(value:unknown,limit=Infinity)=>typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=limit;
  const probabilities=(row:unknown,slack:number):row is number[]=>Array.isArray(row)&&row.length===32&&
    row.every(p=>typeof p==='number'&&p>=-slack&&p<=1+slack)&&Math.abs(row.reduce((a:number,b:number)=>a+b,0)-1)<=5e-6;
  if(!data||typeof data!=='object'||Array.isArray(data))invalid('响应对象');
  if(data.modelSchema!==FULL_MODEL_SCHEMA||data.protocol!==NATIVE_PROTOCOL)invalid('模型版本不兼容，需要 rb87-full/2');
  const states=data.states as Level[],timesUs=data.timesUs as number[],rows=data.populations,d=data.diagnostics;
  if(!Array.isArray(states)||states.length!==32||new Set(states.map(s=>s?.id)).size!==32||states.some(s=>typeof s?.id!=='string'||!['ground','D1','D2'].includes(s.manifold)||!Number.isFinite(s.energyMHz)||!Number.isInteger(s.F)||!Number.isInteger(s.m)||s.m < -s.F||s.m>s.F||!(s.manifold==='D2'?[0,1,2,3]:[1,2]).includes(s.F)||s.id!==`${s.manifold}-${s.F}-${s.m}`))invalid('32 态元数据');
  if(!Array.isArray(timesUs)||timesUs[0]!==0||timesUs.some((t,i)=>!Number.isFinite(t)||(i>0&&t<=timesUs[i-1]!)))invalid('时间采样');
  // Every row is a projected probability vector: no negative or >1 entries.
  if(!Array.isArray(rows)||rows.length!==timesUs.length||!rows.every(row=>probabilities(row,0)))invalid('布居矩阵');
  if(!d||typeof d!=='object'||!bounded(d.traceError,5e-6)||!bounded(d.initialStateError,1e-10)||!bounded(d.elapsedMs))invalid('布居数值验证信息');
  // The first row is the requested initial state's diagonal in the returned basis: a mixture
  // as much as a single state. The model's own populations, when sent, must be the same.
  const spec=typeof initial==='string'?{kind:'state' as const,id:initial}:initial;
  let expected:number[]=[];try{expected=initialPopulations(spec,states);}catch{invalid('请求初态');}
  if(data.initialState!==undefined&&JSON.stringify(canonicalInitialState(data.initialState))!==JSON.stringify(canonicalInitialState(spec)))invalid('请求初态');
  if(data.initialPopulations!==undefined&&(!probabilities(data.initialPopulations,0)||data.initialPopulations.some((p:number,i:number)=>Math.abs(p-expected[i]!)>1e-12)))invalid('请求初态');
  const initialError=Math.max(...rows[0].map((p:number,i:number)=>Math.abs(p-expected[i]!)));
  if(initialError>1e-10||initialError>d.initialStateError+64*Number.EPSILON)invalid('布居初态');
  // A finite trajectory (more than one sample) carries its estimated population
  // error; the single t = 0 row of a mean result is the initial state itself.
  // Neither claims a finite-time density.
  if(Object.hasOwn(data,'densityReal')||Object.hasOwn(data,'densityImag'))invalid('布居结果契约');
  let populationAccuracy:PopulationAccuracy|undefined;
  if(timesUs.length>1){
    if(typeof d.method!=='string')invalid('时间演化方法');
    try{populationAccuracy=readPopulationAccuracy(data.populationAccuracy,timesUs.length);}catch{invalid('布居精度元数据');}
  }else if(Object.hasOwn(data,'populationAccuracy'))invalid('布居精度元数据');
  const status=data.asymptoticStatus as SteadyStatus,reason=data.asymptoticReason as string|undefined;
  if(!STEADY_STATUSES.includes(status)||(reason!==undefined&&typeof reason!=='string'))invalid('长期布居状态');
  const converged=(CONVERGED_MEAN_STATUSES as readonly SteadyStatus[]).includes(status);
  let steadyPopulations:Record<string,number>|null=null,density;
  if(converged){
    const mean=data.asymptoticPopulations;
    if(!probabilities(mean,5e-6))invalid('长期布居矩阵');
    if(!bounded(d.asymptoticTraceError,5e-6)||!bounded(d.asymptoticPositivityError,1e-4)||!bounded(d.residual)||typeof d.asymptoticMethod!=='string')invalid('长期密度数值验证信息');
    try{density=readMeanDensity(data,32,mean);}catch{invalid('长期复密度矩阵');}
    if(!density)invalid('长期复密度矩阵');
    steadyPopulations=Object.fromEntries(states.map((state,i)=>[state.id,mean[i]!]));
  }else if(data.asymptoticPopulations!=null||data.asymptoticDensityReal!=null||data.asymptoticDensityImag!=null)invalid('未收敛结果携带长期布居');
  const horizon=data.horizonUs;if(horizon!==undefined&&!MEAN_HORIZONS_US.includes(horizon))invalid('长期布居观察时长');
  const assumptions=data.assumptions;
  if(!Array.isArray(assumptions)||assumptions.some((text:unknown)=>typeof text!=='string'))invalid('模型假设');
  const populations=Object.fromEntries(states.map((state,i)=>[state.id,rows.map((row:number[])=>row[i]!)]));
  const active=scene.lasers.filter(l=>l.enabled&&laserIntensityMWcm2(l)>0);
  return {states,timesUs,populations,totalExcited:timesUs.map((_,i)=>states.filter(s=>s.manifold!=='ground').reduce((sum,s)=>sum+populations[s.id]![i]!,0)),
    activeLaserIds:active.map(l=>l.id),
    steady:{converged,status,populations:steadyPopulations,density,residual:converged?d.residual:null,method:converged?d.asymptoticMethod:undefined,reason,
      initialDependent:isInitialDependent(status),
      slowestRelaxationUs:converged&&typeof d.slowestRelaxationUs==='number'&&Number.isFinite(d.slowestRelaxationUs)&&d.slowestRelaxationUs>0?d.slowestRelaxationUs:null,
      conservedQuantities:converged&&Number.isSafeInteger(d.conservedQuantities)&&d.conservedQuantities>0?d.conservedQuantities:null,
      ...(horizon!==undefined?{horizonUs:horizon as number|null}:{})},
    metrics:{totalMs:d.elapsedMs},validation:{maxTraceError:d.traceError,minPopulation:Math.min(...rows.flat())},
    ...(populationAccuracy?{populationAccuracy,method:d.method}:{}),assumptions};
}
