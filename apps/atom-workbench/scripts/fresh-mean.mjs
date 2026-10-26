import {readMeanDensity} from '../dist/js/atom-workbench/src/mean-density.js';
import {readPopulationAccuracy} from '../dist/js/atom-workbench/src/solver-types.js';

const verified=new Set(['stationary','population-stationary','periodic-mean','initial-dependent-mean','nonunique-initial-dependent']);
const finite=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
function validatePopulationTrajectory(result,model){
  const n=model.dimension,d=result.diagnostics,times=result.timesUs,rows=result.populations;
  if(n!==32||!Array.isArray(times)||times.length<2||times[0]!==0||times.some((t,i)=>!Number.isFinite(t)||t<0||(i>0&&t<=times[i-1]))||
    !Array.isArray(rows)||rows.length!==times.length||rows.some(row=>!Array.isArray(row)||row.length!==n||row.some(p=>!Number.isFinite(p)||p<0||p>1)||Math.abs(row.reduce((a,b)=>a+b,0)-1)>5e-6)||
    Object.hasOwn(result,'densityReal')||Object.hasOwn(result,'densityImag')||
    !finite(d?.traceError)||d.traceError>5e-6||!finite(d?.elapsedMs)||!finite(d?.initialStateError)||d.initialStateError>1e-10){
    throw Error('Finite population-only result failed its explicit validation contract.');
  }
  readPopulationAccuracy(result.populationAccuracy,times.length);
  let initial;
  if(model.initialReal!=null){
    if(!Array.isArray(model.initialReal)||model.initialReal.length!==n*n||model.initialReal.some(value=>!Number.isFinite(value)))throw Error('Invalid original initial density.');
    initial=Array.from({length:n},(_,i)=>model.initialReal[i*n+i]);
  }else{
    const index=model.initialIndex??0;
    if(!Number.isInteger(index)||index<0||index>=n)throw Error('Invalid original initial state.');
    initial=Array.from({length:n},(_,i)=>+(i===index));
  }
  const error=Math.max(...rows[0].map((p,i)=>Math.abs(p-initial[i])));
  if(error>1e-10||error>d.initialStateError+64*Number.EPSILON)throw Error('Finite population-only result has the wrong initial population.');
}
export function hasVerifiedFreshMean(result,model){
  if(!verified.has(result?.asymptoticStatus))return false;
  const row=result.asymptoticPopulations,d=result.diagnostics,n=model.dimension;
  if(JSON.stringify(result.states)!==JSON.stringify(model.states)||!Array.isArray(result.timesUs)||result.timesUs.length!==1||result.timesUs[0]!==0||
    !Array.isArray(row)||row.length!==n||
    row.some(p=>!Number.isFinite(p)||p< -5e-6||p>1+5e-6)||Math.abs(row.reduce((a,b)=>a+b,0)-1)>5e-6||
    !finite(d?.traceError)||d.traceError>5e-6||!finite(d?.positivityError)||d.positivityError>1e-4||!finite(d?.elapsedMs)){
    throw Error('Independent mean failed the existing density validation gates.');
  }
  readMeanDensity(result,n,row);
  return true;
}

/** Combines two fresh children of ONE request; this module has no result store.
 * Both children have already had physicalMeanDensity applied by their lane. */
export function mergeFreshMean(trajectory,mean,model,{elapsedMs,meanTiming,trajectoryTiming}){
  hasVerifiedFreshMean(mean,model);
  for(const result of [trajectory,mean]){
    if(JSON.stringify(result.states)!==JSON.stringify(model.states))throw Error('Independent results use different state orders.');
  }
  for(const key of ['protocol','modelSchema'])if(trajectory[key]!==mean[key])throw Error('Independent results use different model protocols.');
  for(const key of ['binarySha256','buildSha256','nativeProtocol']){
    if(!trajectory.service?.[key]||trajectory.service[key]!==mean.service?.[key])throw Error('Independent workers do not share the same verified binary and build.');
  }
  const fields={};
  for(const key of ['asymptoticPopulations','asymptoticDensityReal','asymptoticDensityImag','asymptoticScatteringRatePerUs','asymptoticStatus','asymptoticReason']){
    if(Object.hasOwn(mean,key))fields[key]=structuredClone(mean[key]);
  }
  const source=mean.diagnostics??{},diagnostics={...trajectory.diagnostics};
  const populationOnly=diagnostics.validationKind==='population';
  if(populationOnly)validatePopulationTrajectory(trajectory,model);
  for(const key of ['asymptoticStatus','asymptoticReason','densityLimitExists','initialDependent','uniquenessScope','globalUniquenessEstablished','initialDependenceScope','meanDensityFrame']){
    if(Object.hasOwn(source,key))diagnostics[key]=structuredClone(source[key]);
  }
  for(const key of ['residual','populationResidual','harmonicOrder','harmonicConvergenceError','boundaryForcingMax','frameConvergenceError','frameUnitarityBound','frameSchrodingerResidual','traceError','positivityError','hermiticityError']){
    if(Object.hasOwn(source,key))diagnostics[`asymptotic${key[0].toUpperCase()}${key.slice(1)}`]=structuredClone(source[key]);
  }
  if(populationOnly){
    // Mean PSD was independently checked above. The finite population output
    // has no repaired density, and must not receive a fabricated PSD value.
    if(finite(source.traceError))diagnostics.traceError=Math.max(diagnostics.traceError,source.traceError);
  }else{
    for(const key of ['traceError','positivityError']){
      if(!finite(diagnostics[key]))throw Error('Finite trajectory validation is missing.');
      if(finite(source[key]))diagnostics[key]=Math.max(diagnostics[key],source[key]);
    }
  }
  diagnostics.steadyMethod=source.steadyMethod??source.method;
  diagnostics.asymptoticDiagnostics=structuredClone(source);
  diagnostics.trajectorySolveMs=trajectory.diagnostics.elapsedMs;
  diagnostics.stationarySolveMs=source.elapsedMs;
  diagnostics.elapsedMs=elapsedMs;
  diagnostics.elapsedScope='service wall time for the two fresh concurrent solves; excludes HTTP model construction';
  diagnostics.asymptoticServiceCached=false;
  return {...trajectory,...fields,diagnostics,service:{...trajectory.service,lane:'parallel',executionMode:'parallel-independent-mean',
    elapsedMs,mean:{...mean.service,...meanTiming},trajectory:{...trajectory.service,...trajectoryTiming},asymptoticCacheHit:false}};
}
