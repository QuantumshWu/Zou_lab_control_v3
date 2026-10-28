// What the app asks the solver for (solve-plan.ts): a cached mean at the chosen T is shown with no
// request; a trajectory whose mean is cached is asked for alone (includeAsymptotic false).
import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as pause} from 'node:timers/promises';
import {covers,delivers,horizonRecords,keptCheckpoint,plannedSolve,solveOptions} from '../dist/js/atom-workbench/src/solve-plan.js';
import {CurrentResults} from '../dist/js/atom-workbench/src/current-results.js';
import {DynamicsClient} from '../dist/js/atom-workbench/src/solver-client.js';
import {meanInputsKey,trajectoryInputsKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {POPULATION_TOLERANCE,defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {FULL_MODEL_SCHEMA,HTTP_PROTOCOL,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {nativeResult} from './native-result.mjs';

test('the planned request: nothing when the page has its trajectory and its mean at T',()=>{
  assert.equal(plannedSolve(true,{trajectory:false,mean:false}),'full');
  assert.equal(plannedSolve(true,{trajectory:false,mean:true}),'trajectory','the mean cached: the trajectory alone');
  assert.equal(plannedSolve(true,{trajectory:true,mean:false}),'mean');
  assert.equal(plannedSolve(true,{trajectory:true,mean:true}),undefined);
  assert.equal(plannedSolve(false,{trajectory:false,mean:true}),undefined,'the levels\' long-time mode needs the mean alone');
  assert.equal(plannedSolve(false,{trajectory:false,mean:false}),'mean');
  assert.deepEqual(solveOptions('full'),{steadyOnly:false,includeAsymptotic:true});
  assert.deepEqual(solveOptions('trajectory'),{steadyOnly:false,includeAsymptotic:false});
  assert.deepEqual(solveOptions('mean'),{steadyOnly:true,includeAsymptotic:true});
});

test('a running request goes on while it brings what the page needs: a mean run brings every T',()=>{
  const here={trajectoryKey:'X',meanKey:'M'},longer={trajectoryKey:'Y',meanKey:'M'},other={trajectoryKey:'Z',meanKey:'N'};
  assert.equal(covers('full',here,'mean',here),true,'the levels opened during a full solve');
  assert.equal(covers('mean',here,'mean',here),true,'another T during a mean run');
  assert.equal(covers('mean',here,'full',here),false,'a mean run brings no trajectory');
  assert.equal(covers('trajectory',here,'mean',here),false);
  assert.equal(covers('full',here,'full',longer),false,'another duration: its trajectory is another');
  assert.equal(covers('full',here,'mean',longer),true,'but its mean still stands');
  assert.deepEqual(delivers('full',here,other),{trajectory:false,mean:false},'other physics: obsolete');
  assert.deepEqual(delivers('trajectory',here,longer),{trajectory:false,mean:false});
});

const steady=(T,error)=>({converged:true,status:'stationary',populations:{},estimatedError:error,withinTolerance:error<=POPULATION_TOLERANCE,residual:null,initialDependent:false,horizonUs:T});
test('a result\'s means become one record per T; 停止 keeps a checkpoint\'s where no final mean is',()=>{
  const result={timesUs:[0,1,2],steady:steady(1e6,.001),horizonMeans:[steady(1e5,.002),steady(1e6,.001),steady(null,.003)]};
  const records=horizonRecords({result,wallMs:5,horizon:1e6});
  assert.deepEqual(records.map(r=>[r.horizon,r.result.steady.estimatedError,r.wallMs]),[[1e5,.002,5],[1e6,.001,5],[null,.003,5]]);
  assert.deepEqual(horizonRecords({result:{...result,horizonMeans:undefined},wallMs:5,horizon:1e6}),[],'a trajectory alone carries none');
  assert.deepEqual(keptCheckpoint(result,'full',()=>false),{trajectory:true,horizons:[1e5,1e6,null]});
  assert.deepEqual(keptCheckpoint(result,'full',T=>T===null),{trajectory:true,horizons:[1e5,1e6]},'a final mean already here stays');
  assert.deepEqual(keptCheckpoint(result,'trajectory',()=>false),{trajectory:true,horizons:[]});
  assert.deepEqual(keptCheckpoint({timesUs:[0],steady:steady(1e6,.001),horizonMeans:result.horizonMeans},'mean',()=>false),{trajectory:false,horizons:[1e5,1e6,null]},'a mean run\'s checkpoint is never a trajectory');
});

/** The mock solver: every call is answered by the test. */
function transport(){const calls=[];return {calls,fetch:async(url,init)=>new Promise((resolve,reject)=>calls.push({url,init,resolve,reject,body:JSON.parse(init.body)}))};}
const reply=(call,data)=>call.resolve(new Response(JSON.stringify({...data,request:call.body.request}),{status:200,headers:{'Content-Type':'application/json'}}));
async function waitCalls(mock,count){for(let i=0;i<100&&mock.calls.length<count;i++)await pause(2);assert.equal(mock.calls.length,count);}
function answer(scene,options){
  const settings=scene.dynamics,timesUs=options.steadyOnly?[0]:[0,settings.durationUs/2,settings.durationUs];
  const states=buildFullMasterModel(scene,settings).states,initialIndex=states.findIndex(s=>s.id===settings.initialStateId);
  // Each T its own error, so the test sees which T it is shown.
  return {...nativeResult({states,initialIndex,timesUs,mean:options.includeAsymptotic,horizonUs:options.meanHorizonUs,meanError:.0011,horizonErrors:{'100000':.0012,'1000000':.0011,'null':.0013}}),
    assumptions:['test assumption'],service:{protocol:HTTP_PROTOCOL,nativeProtocol:NATIVE_PROTOCOL,capabilities:[FULL_MODEL_SCHEMA]}};
}

test('request flow: a horizon switch makes no request; a new duration asks for the trajectory alone and keeps the mean',async t=>{
  const mock=transport(),client=new DynamicsClient({fetch:mock.fetch,debounceMs:0,clientId:'horizon-flow'});t.after(()=>client.dispose());
  const scene=defaultScene();scene.dynamics.samples=3;scene.dynamics.durationUs=.2;scene.dynamics.meanHorizonUs=1e6;
  const trajectories=new CurrentResults(),means=new CurrentResults();
  const inputs=()=>({trajectoryKey:trajectoryInputsKey(scene,scene.dynamics),meanKey:meanInputsKey(scene,scene.dynamics)});
  // The app on every change (app.ts invalidateObsoletePopulation, requestSolve): the current
  // inputs' stores, then the request the dynamics page needs.
  const plan=()=>{const i=inputs();trajectories.activate(i.trajectoryKey);means.activate(i.meanKey);
    return plannedSolve(true,{trajectory:!!trajectories.get(i.trajectoryKey,'trajectory'),mean:!!means.get(i.meanKey,scene.dynamics.meanHorizonUs)});};
  const solve=async kind=>{
    const i=inputs(),epochs={trajectory:trajectories.activate(i.trajectoryKey),mean:means.activate(i.meanKey)},before=mock.calls.length;
    const pending=client.solve(scene,scene.dynamics,solveOptions(kind));await waitCalls(mock,before+1);
    const call=mock.calls.at(-1);reply(call,answer(scene,call.body.options));
    const record={result:await pending,wallMs:1,horizon:scene.dynamics.meanHorizonUs};
    if(kind!=='mean')trajectories.set(i.trajectoryKey,epochs.trajectory,'trajectory',record);
    if(kind!=='trajectory')for(const mean of horizonRecords(record))means.set(i.meanKey,epochs.mean,mean.horizon,mean);
    return call.body.options;
  };
  const shownError=()=>means.get(inputs().meanKey,scene.dynamics.meanHorizonUs)?.result.steady.estimatedError;

  assert.equal(plan(),'full');
  assert.deepEqual((({steadyOnly,includeAsymptotic,meanHorizonUs})=>({steadyOnly,includeAsymptotic,meanHorizonUs}))(await solve('full')),{steadyOnly:false,includeAsymptotic:true,meanHorizonUs:1e6});
  assert.equal(shownError(),.0011);
  // ∞, then 1 s, then 100 ms: each mean is here already.
  for(const [T,error] of [[null,.0013],[1e6,.0011],[1e5,.0012]]){scene.dynamics.meanHorizonUs=T;assert.equal(plan(),undefined,`T = ${T}`);assert.equal(shownError(),error,`T = ${T}`);}
  assert.equal(mock.calls.length,1,'no request for a horizon switch');
  // A longer trajectory: asked for alone; its mean is the cached one, the very same.
  const cached=means.get(inputs().meanKey,1e5);scene.dynamics.durationUs=.4;
  assert.equal(plan(),'trajectory');
  const sent=await solve('trajectory');
  assert.equal(sent.includeAsymptotic,false);assert.equal(sent.steadyOnly,false);assert.equal(sent.durationUs,.4);
  assert.equal(trajectories.get(inputs().trajectoryKey,'trajectory').result.steady.status,'not-requested','the trajectory alone');
  assert.equal(plan(),undefined);assert.strictEqual(means.get(inputs().meanKey,1e5),cached,'the mean unchanged');
  // New samples: the same.
  scene.dynamics.samples=5;assert.equal(plan(),'trajectory');
  // Other physics: the mean and the trajectory are solved again, together.
  scene.fields[0].magnitudeG+=1;assert.equal(plan(),'full');assert.equal(mock.calls.length,2);
});
