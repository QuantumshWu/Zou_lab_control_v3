// One contract-valid native result (rust-solver/src/lattice_core_adapter.rs)
// for the transport and reader tests: populations stay in the initial state.
import {POPULATION_GROUPS} from '../dist/js/atom-workbench/src/population-selection.js';
import {FULL_MODEL_SCHEMA,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';

/** A finite result over timesUs (a mean result when timesUs is [0]) with its per-sample errors
 * (`sampleErrors`, else `error` after t = 0); `mean` adds a computed stationary long-time mean with
 * its estimated error `meanError` at the time T `horizonUs`. */
export function nativeResult({states,initialIndex,timesUs,tolerance=.005,error=0,charge=0,mean=true,sampleErrors=timesUs.map((_,i)=>i?error:0),meanError=0,horizonUs=1e6}){
  const row=Array.from({length:32},(_,i)=>+(i===initialIndex)),real=Array(32*32).fill(0),finite=timesUs.length>1;
  real[initialIndex*33]=1;
  return {protocol:NATIVE_PROTOCOL,modelSchema:FULL_MODEL_SCHEMA,states,timesUs,populations:timesUs.map(()=>row.slice()),
    ...(finite?{populationAccuracy:{status:Math.max(...sampleErrors)>tolerance?'estimated-partial':'estimated-converged',requestedTolerance:tolerance,estimatedMaximumError:Math.max(...sampleErrors),
      sampleErrors:sampleErrors.slice(),...(Math.max(...sampleErrors)>tolerance?{reason:'test budget'}:{}),projectionCharge:charge,
      includesProjectionCharge:true,scope:{states:32,samples:timesUs.length,groups:POPULATION_GROUPS.map(group=>group.id)}}}:{}),
    ...(mean?{asymptoticStatus:'stationary',asymptoticReason:'test mean',asymptoticPopulations:row.slice(),
      asymptoticDensityReal:real,asymptoticDensityImag:Array(32*32).fill(0),asymptoticEstimatedError:meanError,asymptoticWithinTolerance:meanError<=tolerance,horizonUs}:{asymptoticStatus:'not-requested'}),
    diagnostics:{traceError:0,initialStateError:0,elapsedMs:1,...(finite?{method:'test trajectory'}:{}),
      ...(mean?{asymptoticMethod:'test mean',asymptoticTraceError:0,asymptoticPositivityError:0,residual:0}:{})}};
}
