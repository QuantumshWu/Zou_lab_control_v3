// One contract-valid native result (rust-solver/src/lattice_core_adapter.rs)
// for the transport and reader tests: populations stay in the initial state.
import {POPULATION_GROUPS} from '../dist/js/atom-workbench/src/population-selection.js';
import {FULL_MODEL_SCHEMA,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';

/** The standard times T of a mean run, in the order the reader returns them. */
export const HORIZONS=[1e5,1e6,null];

/** A finite result over timesUs (a mean result when timesUs is [0]) with its per-sample errors
 * (`sampleErrors`, else `error` after t = 0); `mean` adds a computed stationary long-time mean at
 * every standard T (`asymptoticHorizons`): its estimated error `meanError` at the requested T
 * `horizonUs`, and `horizonErrors` maps another T to its own. */
export function nativeResult({states,initialIndex,timesUs,tolerance=.005,error=0,mean=true,sampleErrors=timesUs.map((_,i)=>i?error:0),meanError=0,horizonUs=1e6,horizonErrors={}}){
  const row=Array.from({length:32},(_,i)=>+(i===initialIndex)),real=Array(32*32).fill(0),finite=timesUs.length>1;
  real[initialIndex*33]=1;
  const entry=T=>{const e=T===horizonUs?meanError:horizonErrors[String(T)]??meanError;
    return {horizonUs:T,status:'stationary',reason:'test mean',populations:row.slice(),densityReal:real.slice(),densityImag:Array(32*32).fill(0),estimatedError:e,withinTolerance:e<=tolerance};};
  return {protocol:NATIVE_PROTOCOL,modelSchema:FULL_MODEL_SCHEMA,states,timesUs,populations:timesUs.map(()=>row.slice()),
    ...(finite?{populationAccuracy:{status:Math.max(...sampleErrors)>tolerance?'estimated-partial':'estimated-converged',requestedTolerance:tolerance,estimatedMaximumError:Math.max(...sampleErrors),
      sampleErrors:sampleErrors.slice(),...(Math.max(...sampleErrors)>tolerance?{reason:'test budget'}:{}),
      scope:{states:32,samples:timesUs.length,groups:POPULATION_GROUPS.map(group=>group.id)}}}:{}),
    ...(mean?{asymptoticHorizons:HORIZONS.map(entry)}:{}),
    diagnostics:{traceError:0,initialStateError:0,elapsedMs:1,...(finite?{method:'test trajectory'}:{}),
      ...(mean?{asymptoticMethod:'test mean',asymptoticTraceError:0,asymptoticPositivityError:0,residual:0}:{})}};
}
/** A result's mean at T (the scene's default, 1 s, unless said). */
export function entryAt(data,T=1e6){return data.asymptoticHorizons.find(e=>e.horizonUs===T);}
