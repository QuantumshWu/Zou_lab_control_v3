/** Full-app integration observer, inert unless the native TEST host explicitly
 * injects __atomSelfTest into an isolated profile. It uses the actual app
 * buttons/DynamicsClient and never replaces a scene, request, or result. */
export function installSelfTest({adapter,postNative}){
 const config=globalThis.__atomSelfTest;if(!config)return adapter.fetch;
 const report={runtimeVersion:config.runtimeVersion??globalThis.__runtimeVersion,hiddenRafShim:config.hiddenRafShim===true,
  origin:location.origin,url:location.href,nodeRuntime:false,ui:'unchanged full app bundle',requests:[],streamEvents:[],transportErrors:[]};
 let finished=false,finalizing=false,firstFetchAt,clickAt,finalResult;
 const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
 const expected=config.request;
 const snapshot=()=>({dynamicsSelected:document.querySelector('.tab[data-view="dynamics"]')?.getAttribute('aria-selected')==='true',
  chartSvg:!!document.querySelector('#population-chart svg'),populationFresh:document.querySelector('#app')?.dataset.populationFresh,
  summary:document.querySelector('#summary')?.textContent?.slice(0,600),chartCaption:document.querySelector('#population-caption')?.textContent?.slice(0,600),
  storedScenePresent:localStorage.getItem('rb87-workbench-demo-v4')!==null,storageKeys:Object.keys(localStorage),
  width:innerWidth,height:innerHeight,visibility:document.visibilityState});
 async function finish(ok,error){
  if(finished)return;finished=true;clearTimeout(watchdog);
  try{report.dom=snapshot();}catch(snapshotError){report.snapshotError=String(snapshotError);}
  await adapter.dispose();
  postNative({op:'self-test',result:{...report,ok,...(error?{error:String(error.stack??error)}:{}),...(finalResult?{finalResult}:{}),
   elapsedMs:firstFetchAt===undefined?null:performance.now()-firstFetchAt}});
 }
 const watchdog=setTimeout(()=>void finish(false,new Error('Full-app hidden self-test did not receive and display a valid final result.')),config.timeoutMs??120000);
 function assertToken(data,body){
  if(data?.request?.clientId!==body.request?.clientId||data?.request?.sequence!==body.request?.sequence)throw Error('Native progress/final token differs from the actual UI request.');
 }
 function validateRequest(body){
  if(!body?.scene||!body.request||!body.options)throw Error('The real app sent an incomplete solve request.');
  if(body.options.steadyOnly!==false)throw Error('Full-app test expected finite evolution, not the initial levels-tab mean.');
  if(body.options.includeAsymptotic!==true)throw Error('Full-app test must exercise the complete mean + trajectory result.');
  if(expected?.modelSchema!==undefined&&body.modelSchema!==expected.modelSchema)throw Error('UI model schema differs from test input.');
  for(const key of ['durationUs','samples','initialStateId']){
   if(expected?.options?.[key]!==undefined&&body.options[key]!==expected.options[key])throw Error(`UI ${key} differs from the seeded reference request.`);
  }
 }
 async function acceptFinal(data,body,started){
  if(finished||finalizing)return;assertToken(data,body);
  if(!Array.isArray(data.states)||data.states.length!==32||!Array.isArray(data.timesUs)||data.timesUs.length!==body.options.samples
    ||!Array.isArray(data.populations)||data.populations.length!==body.options.samples)throw Error('Actual full-app response dimensions are incomplete.');
  if(data.timesUs[0]!==0||Math.abs(data.timesUs.at(-1)-body.options.durationUs)>1e-10*Math.max(1,body.options.durationUs))throw Error('Actual UI duration is inconsistent with its response.');
  finalizing=true;finalResult=data;report.actualRequest=body;report.finalResultMs=performance.now()-started;
  report.fetchToFinalMs=performance.now()-firstFetchAt;report.clickToFinalMs=clickAt===undefined?null:performance.now()-clickAt;
  // Observe the actual client’s parsing/state update and DOM paint, rather
  // than treating the transport result alone as successful UI delivery.
  await delay(60);const dom=snapshot();
  if(!dom.dynamicsSelected||!dom.chartSvg||dom.populationFresh!=='true')throw Error('Validated response did not become a fresh Population chart in the real app: '+JSON.stringify(dom));
  await finish(true);
 }
 async function observe(response,body,started,requestIndex){
  if(!response.ok){const data=await response.json().catch(()=>({}));report.transportErrors.push({requestIndex,status:response.status,data});
   const retryable=(response.status===502||response.status===503)&&!/invalid|incompatible|numerical/.test(data.code??'');
   if(!retryable||report.transportErrors.filter(item=>item.status===502||item.status===503).length>=2)await finish(false,new Error(data.error??`Native solve returned HTTP ${response.status}`));return;}
  if(!response.headers.get('Content-Type')?.includes('application/x-ndjson')){
   const data=await response.json();report.streamEvents.push({type:'json-result',requestIndex,atMs:performance.now()-started,request:data.request});
   await acceptFinal(data,body,started);return;
  }
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',final,terminal=false;
  const consume=line=>{
   if(!line.trim())return;if(terminal)throw Error('Extra native stream event followed a terminal result.');
   const event=JSON.parse(line),atMs=performance.now()-started;
   if(!['checkpoint','result','error'].includes(event.type))throw Error('Unknown native NDJSON event.');
   if(event.type==='error'){terminal=true;report.streamEvents.push({...event,requestIndex,atMs});throw Error(event.data?.error??'Native stream failed.');}
   assertToken(event.data,body);
   report.streamEvents.push({type:event.type,requestIndex,atMs,request:event.data.request,...(event.type==='checkpoint'?{progress:{solves:event.data.diagnostics?.solves,estimatedMaximumError:event.data.populationAccuracy?.estimatedMaximumError}}:{} )});
   if(event.type==='checkpoint')report.firstCheckpointMs??=performance.now()-firstFetchAt;
   else{final=event.data;terminal=true;}
  };
  try{
   while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value,{stream:!done});if(buffer.length>64*1024*1024)throw Error('Native self-test stream exceeded its limit.');
    let end;while((end=buffer.indexOf('\n'))>=0){consume(buffer.slice(0,end));buffer=buffer.slice(end+1);}if(done)break;
   }
   if(buffer.trim())consume(buffer);if(final===undefined)throw Error('Native stream ended without its final result.');
   await acceptFinal(final,body,started);
  }finally{reader.releaseLock();}
 }
 function wrappedFetch(input,init){
  const url=new URL(input instanceof Request?input.url:String(input),location.href);
  if(url.pathname!=='/api/solve')return adapter.fetch(input,init);
  const started=performance.now();firstFetchAt??=started;const requestIndex=report.requests.length;
  const record={requestIndex,startedMs:started-firstFetchAt};report.requests.push(record);
  const bodyPromise=typeof init?.body==='string'?Promise.resolve().then(()=>JSON.parse(init.body))
   :input instanceof Request?input.clone().json():Promise.reject(new Error('Cannot observe actual UI request body.'));
  // Register the observer BEFORE the caller's await resumes, but return the
  // original Response to the real DynamicsClient. The clone is test-only.
  return adapter.fetch(input,init).then(response=>{
   const observed=response.clone();
   void bodyPromise.then(body=>{validateRequest(body);record.request=body.request;record.options=body.options;return observe(observed,body,started,requestIndex);})
    .catch(error=>{if(error?.name==='AbortError'){report.transportErrors.push({requestIndex,error:'AbortError'});return;}void finish(false,error);});
   return response;
  },error=>{report.transportErrors.push({requestIndex,error:String(error)});throw error;});
 }
 async function waitFor(selector){
  const started=performance.now();while(!finished){const element=document.querySelector(selector);if(element)return element;
   if(performance.now()-started>(config.domTimeoutMs??10000))throw Error('Full app did not create '+selector);await delay(16);}
  throw Error('Self-test ended before the app was ready.');
 }
 async function start(){try{
  if(!expected?.scene)throw Error('Full-app self-test is missing the reference scene.');
  // v3 TabWidget pages turn on the tab's press.
  const tab=await waitFor('.tab[data-view="dynamics"]');clickAt=performance.now();tab.dispatchEvent(new MouseEvent('mousedown',{bubbles:true,button:0}));
  const run=await waitFor('#run-dynamics');if(run.disabled)throw Error('The actual Population solve button is disabled.');run.click();
 }catch(error){await finish(false,error);}}
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>void start(),{once:true});else queueMicrotask(()=>void start());
 return wrappedFetch;
}
