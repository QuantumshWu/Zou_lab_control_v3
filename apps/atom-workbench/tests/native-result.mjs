// One contract-valid native result (rust-solver/src/lattice_core_adapter.rs, and with motion
// motion3d_adapter.rs) for the transport and reader tests: populations stay in the initial state.
import {POPULATION_GROUPS} from '../dist/js/atom-workbench/src/population-selection.js';
import {FULL_MODEL_SCHEMA,NATIVE_PROTOCOL} from '../dist/js/atom-workbench/src/solver-protocol.js';
import {POPULATION_TOLERANCE} from '../dist/js/atom-workbench/src/types.js';
import {SEMICLASSICAL_POPULATION_TARGET} from '../dist/js/atom-workbench/src/solver-types.js';

/** The standard times T of a mean run, in the order the reader returns them. */
export const HORIZONS=[1e5,1e6];

/** A finite result over timesUs (a mean result when timesUs is [0]) with its per-sample errors
 * (`sampleErrors`, else `error` after t = 0); `mean` adds a computed long-time mean p̄(T) at
 * both standard T (`meanHorizons`): its estimated error `meanError` at the requested T
 * `horizonUs`, and `horizonErrors` maps another T to its own. */
export function nativeResult({states,initialIndex,timesUs,tolerance=POPULATION_TOLERANCE,error=0,mean=true,sampleErrors=timesUs.map((_,i)=>i?error:0),meanError=0,horizonUs=1e6,horizonErrors={}}){
  const row=Array.from({length:32},(_,i)=>+(i===initialIndex)),real=Array(32*32).fill(0),finite=timesUs.length>1;
  real[initialIndex*33]=1;
  const entry=T=>{const e=T===horizonUs?meanError:horizonErrors[String(T)]??meanError;
    return {horizonUs:T,status:'time-average',reason:'test mean',populations:row.slice(),densityReal:real.slice(),densityImag:Array(32*32).fill(0),estimatedError:e,withinTolerance:e<=tolerance};};
  return {protocol:NATIVE_PROTOCOL,modelSchema:FULL_MODEL_SCHEMA,states,timesUs,populations:timesUs.map(()=>row.slice()),
    ...(finite?{populationAccuracy:{status:Math.max(...sampleErrors)>tolerance?'estimated-partial':'estimated-converged',requestedTolerance:tolerance,estimatedMaximumError:Math.max(...sampleErrors),
      sampleErrors:sampleErrors.slice(),...(Math.max(...sampleErrors)>tolerance?{reason:'test budget'}:{}),
      scope:{states:32,samples:timesUs.length,groups:POPULATION_GROUPS.map(group=>group.id)}}}:{}),
    ...(mean?{meanHorizons:HORIZONS.map(entry)}:{}),
    diagnostics:{traceError:0,initialStateError:0,elapsedMs:1,...(finite?{method:'test trajectory'}:{}),
      ...(mean?{meanMethod:'test mean',meanTraceError:0,meanPositivityError:0}:{})}};
}
/** A result's mean at T (the scene's default, 1 s, unless said). */
export function entryAt(data,T=1e6){return data.meanHorizons.find(e=>e.horizonUs===T);}
/** A contract-valid motion block on `samples` samples: per lab axis T (x, y, z: 30, 28 and 35 μK)
 * with its standard error, ⟨v⟩, n̄, the trap frequency and the distributions at the first and the
 * last sample; the total temperature (31 μK) with its standard error; the loss, rising by `loss`
 * per sample from 0; the regime per sample, the last outside it. */
export function motionBlock(samples,loss=.1){
  const velocities=[-.1,-.05,0,.05,.1],positions=[-.4,-.2,0,.2,.4],p=[.1,.2,.4,.2,.1],series=value=>Array(samples).fill(value);
  const axes=make=>({x:make(),y:make(),z:make()});
  return {axis:'3d',temperatureUK:{x:series(30),y:series(28),z:series(35)},temperatureErrorUK:axes(()=>series(1.5)),temperatureTotalUK:series(31),temperatureTotalErrorUK:series(.9),
    meanVelocityMps:axes(()=>series(0)),meanOccupation:axes(()=>series(4.5)),trapFrequencyKHz:{x:100.5,y:100.5,z:19.3},lossProbability:Array.from({length:samples},(_,i)=>i*loss),
    classical:Array.from({length:samples},(_,i)=>i<samples-1),momentum:axes(()=>({velocitiesMps:velocities,samples:[0,samples-1],probabilities:[p,p]})),
    position:axes(()=>({positionsUm:positions,samples:[0,samples-1],probabilities:[p,p]})),trajectories:512,timeStepUs:.25};
}
/** The motion entries of a motion result's diagnostics: the device used, a fallback's note and the
 * approximation ledger. */
export const MOTION_DIAGNOSTICS={device:'cpu',deviceNote:'尚无 GPU 版本：本次在 CPU 上计算',
  approximations:{budget:.005,used:.0021,items:[{label:'谐振近似',estimate:.0012,action:'计入预算'},{label:'一阶多普勒',estimate:.0009,action:'计入预算'}]}};
/** A motion result over timesUs: the held atom in its initial state, at the method's own target. */
export function motionResult({states,initialIndex,timesUs,block=motionBlock(timesUs.length),diagnostics=MOTION_DIAGNOSTICS}){
  const data=nativeResult({states,initialIndex,timesUs,mean:false,tolerance:SEMICLASSICAL_POPULATION_TARGET});
  return {...data,motion:block,diagnostics:{...data.diagnostics,...structuredClone(diagnostics)},assumptions:['test assumption']};
}
