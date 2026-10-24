import {readMeanDensity} from '../dist/js/atom-workbench/src/mean-density.js';

const verified=new Set(['stationary','population-stationary','periodic-mean','initial-dependent-mean','nonunique-initial-dependent']);
const finite=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
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
  for(const key of ['asymptoticStatus','asymptoticReason','densityLimitExists','initialDependent','uniquenessScope','globalUniquenessEstablished','initialDependenceScope','meanDensityFrame']){
    if(Object.hasOwn(source,key))diagnostics[key]=structuredClone(source[key]);
  }
  for(const key of ['residual','populationResidual','harmonicOrder','harmonicConvergenceError','boundaryForcingMax','frameConvergenceError','frameUnitarityBound','frameSchrodingerResidual','traceError','positivityError','hermiticityError']){
    if(Object.hasOwn(source,key))diagnostics[`asymptotic${key[0].toUpperCase()}${key.slice(1)}`]=structuredClone(source[key]);
  }
  for(const key of ['traceError','positivityError']){
    if(!finite(diagnostics[key]))throw Error('Finite trajectory validation is missing.');
    if(finite(source[key]))diagnostics[key]=Math.max(diagnostics[key],source[key]);
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
