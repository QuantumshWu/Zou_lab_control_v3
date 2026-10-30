import type {Level} from './types.js';
import {POPULATION_TOLERANCE} from './types.js';
import {POPULATION_GROUPS} from './population-selection.js';
/** A trajectory's estimated population error (ui-design.md §6.16). Every computed trajectory is
 * shown: 'estimated-converged' when every sample's bound is within the requested tolerance ε,
 * 'estimated-partial' otherwise -- the samples above ε are marked. */
export interface PopulationAccuracy {
  status:'estimated-converged'|'estimated-partial';requestedTolerance:number;estimatedMaximumError:number;
  /** One absolute population-error bound per sample; their maximum is estimatedMaximumError. */
  sampleErrors:number[];
  /** Why the solver stopped short of ε (partial only). */
  reason?:string;
  scope:{states:32;samples:number;groups:string[]};
}
/** Explicit population-output qualification; never a density/PSD claim. */
export function readPopulationAccuracy(value:unknown,samples:number):PopulationAccuracy {
  const invalid=():never=>{throw new Error('Invalid complete populationAccuracy contract');};
  if(!value||typeof value!=='object'||Array.isArray(value))invalid();
  const a=value as Record<string,unknown>,partial=a.status==='estimated-partial';
  // Every request asks for POPULATION_TOLERANCE; the echo must be that one.
  if((a.status!=='estimated-converged'&&!partial)||a.requestedTolerance!==POPULATION_TOLERANCE)invalid();
  const error=a.estimatedMaximumError,finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0;
  // Partial exactly when the largest bound is above ε.
  if(!finite(error)||partial!==(error>POPULATION_TOLERANCE))invalid();
  // The per-sample bounds: one for each sample, their maximum the trajectory's. A partial result
  // says why the solver stopped short of ε.
  const errors=a.sampleErrors;
  if(!Array.isArray(errors)||errors.length!==samples||!errors.every(finite)||Math.max(...errors)!==error)invalid();
  if(partial?typeof a.reason!=='string':a.reason!==undefined)invalid();
  if(!a.scope||typeof a.scope!=='object'||Array.isArray(a.scope))invalid();
  const scope=a.scope as Record<string,unknown>,groups=scope.groups;
  if(scope.states!==32||scope.samples!==samples||!Number.isSafeInteger(samples)||samples<1||!Array.isArray(groups)||groups.length!==POPULATION_GROUPS.length||new Set(groups).size!==groups.length||groups.some(id=>typeof id!=='string'||!POPULATION_GROUPS.some(group=>group.id===id)))invalid();
  return {status:partial?'estimated-partial':'estimated-converged',requestedTolerance:POPULATION_TOLERANCE,estimatedMaximumError:error as number,
    sampleErrors:[...(errors as number[])],...(partial?{reason:a.reason as string}:{}),
    scope:{states:32,samples,groups:[...(groups as string[])]}};
}
/** The native mean `status` values (`meanHorizons`): a computed mean p̄(T), the phase
 * average at T ('time-average'), or none. */
export const CONVERGED_MEAN_STATUSES=['time-average'] as const;
export type MeanStatus=typeof CONVERGED_MEAN_STATUSES[number]|'not-converged'|'not-requested';
export const MEAN_STATUSES:readonly MeanStatus[]=[...CONVERGED_MEAN_STATUSES,'not-converged','not-requested'];
/** The long-time mean p̄(T) at one time T. */
export interface MeanResult {
  /** A mean was computed (its status one of CONVERGED_MEAN_STATUSES). */
  converged:boolean;status:MeanStatus;populations:Record<string,number>|null;
  /** The computed result's estimated population error (null: nothing computed). */
  estimatedError:number|null;
  /** Within the requested tolerance: its bound is within ε. A computed result not within is still
   * shown, with its error above ε. */
  withinTolerance:boolean;
  /** Native full complex long-time mean, row-major in DynamicsResult.states. */
  density?:{real:number[];imag:number[]};
  method?:string;reason?:string;
  /** Slowest relaxation time (us) of the mean run's decaying modes (null: none resolved or
   * reported): p̄(T) is still relaxing at a T shorter than it. */
  slowestRelaxationUs?:number|null;
  /** The time T of p̄(T) (us); undefined when no mean was requested. */
  horizonUs?:number;
}
export interface DynamicsResult {
  states:Level[];timesUs:number[];populations:Record<string,number[]>;totalExcited:number[];
  activeLaserIds:string[];mean:MeanResult;
  /** The mean at every standard time T (MEAN_HORIZONS_US order), all from the one mean run;
   * the requested T's is `mean` itself. Absent when no mean was requested. */
  horizonMeans?:MeanResult[];
  metrics:{totalMs:number};
  validation:{maxTraceError:number;minPopulation:number};
  /** Estimated error of returned populations and all declared groups only. */
  populationAccuracy?:PopulationAccuracy;
  /** The finite trajectory's method; a mean-only result has none. */
  method?:string;assumptions:string[];
}
