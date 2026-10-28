import type {DynamicsResult} from './solver-types.js';

/** What a request computes: the mean alone ('mean': one mean run, every standard time T), the
 * trajectory with its mean ('full'), or the trajectory alone ('trajectory', includeAsymptotic
 * false: its mean is already at hand). */
export type SolveKind='mean'|'full'|'trajectory';
/** The inputs a request was made for: its trajectory's key and its mean's key. */
export interface SolveInputs{trajectoryKey:string;meanKey:string}

/** The request a page needs now, none when what it shows is here: the mean at T on every page,
 * the trajectory on all but the levels' long-time mode. */
export function plannedSolve(showsTrajectory:boolean,have:{trajectory:boolean;mean:boolean}):SolveKind|undefined{
  return !showsTrajectory||have.trajectory?(have.mean?undefined:'mean'):have.mean?'trajectory':'full';
}
/** The client options of a request. */
export function solveOptions(kind:SolveKind):{steadyOnly:boolean;includeAsymptotic:boolean}{
  return {steadyOnly:kind==='mean',includeAsymptotic:kind!=='trajectory'};
}
/** What a request made for `ran` still brings the current inputs: its trajectory while the
 * trajectory inputs stand, its means (every T) while the mean inputs do. */
export function delivers(kind:SolveKind,ran:SolveInputs,current:SolveInputs):{trajectory:boolean;mean:boolean}{
  return {trajectory:kind!=='mean'&&ran.trajectoryKey===current.trajectoryKey,mean:kind!=='trajectory'&&ran.meanKey===current.meanKey};
}
/** A running request goes on when it brings everything the planned one would. */
export function covers(kind:SolveKind,ran:SolveInputs,need:SolveKind,current:SolveInputs):boolean{
  const brings=delivers(kind,ran,current);
  return (need==='trajectory'||brings.mean)&&(need==='mean'||brings.trajectory);
}
/** A result's mean at every standard T, each as a record of its own (the result with that mean). */
export function horizonRecords<R extends {result:DynamicsResult;horizon:number|null}>(record:R):R[]{
  return (record.result.horizonMeans??[]).map(steady=>({...record,result:{...record.result,steady},horizon:steady.horizonUs??null}));
}
/** What the last checkpoint of a stopped (or failed) request stands as: the trajectory when it is
 * one, and its mean at each T where no final mean is here. */
export function keptCheckpoint(checkpoint:DynamicsResult,kind:SolveKind,finalMean:(horizon:number|null)=>boolean):{trajectory:boolean;horizons:(number|null)[]}{
  return {trajectory:kind!=='mean'&&checkpoint.timesUs.length>1,
    horizons:kind==='trajectory'?[]:(checkpoint.horizonMeans??[]).map(steady=>steady.horizonUs??null).filter(horizon=>!finalMean(horizon))};
}
