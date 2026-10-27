import type {Level} from './types.js';
import {POPULATION_TOLERANCE} from './types.js';
import {POPULATION_GROUPS} from './population-selection.js';
export interface PopulationAccuracy {
  status:'estimated-converged';requestedTolerance:number;estimatedMaximumError:number;
  projectionCharge:number;includesProjectionCharge:true;
  scope:{states:32;samples:number;groups:string[]};
}
/** Explicit population-output qualification; never a density/PSD claim. */
export function readPopulationAccuracy(value:unknown,samples:number):PopulationAccuracy {
  const invalid=():never=>{throw new Error('Invalid complete populationAccuracy contract');};
  if(!value||typeof value!=='object'||Array.isArray(value))invalid();
  const a=value as Record<string,unknown>;
  // Every request asks for POPULATION_TOLERANCE; the echo must be that one.
  if(a.status!=='estimated-converged'||a.includesProjectionCharge!==true||a.requestedTolerance!==POPULATION_TOLERANCE)invalid();
  const error=a.estimatedMaximumError,charge=a.projectionCharge;
  if(typeof error!=='number'||!Number.isFinite(error)||error<0||error>POPULATION_TOLERANCE||typeof charge!=='number'||!Number.isFinite(charge)||charge<0||charge>error)invalid();
  if(!a.scope||typeof a.scope!=='object'||Array.isArray(a.scope))invalid();
  const scope=a.scope as Record<string,unknown>,groups=scope.groups;
  if(scope.states!==32||scope.samples!==samples||!Number.isSafeInteger(samples)||samples<1||!Array.isArray(groups)||groups.length!==POPULATION_GROUPS.length||new Set(groups).size!==groups.length||groups.some(id=>typeof id!=='string'||!POPULATION_GROUPS.some(group=>group.id===id)))invalid();
  return {status:'estimated-converged',requestedTolerance:POPULATION_TOLERANCE,estimatedMaximumError:error as number,
    projectionCharge:charge as number,includesProjectionCharge:true,
    scope:{states:32,samples,groups:[...(groups as string[])]}};
}
/** The native `asymptoticStatus` values: a converged long-time mean (static:
 * stationary / population-stationary / initial-dependent; driven:
 * time-average / initial-dependent-time-average), or none. */
export const CONVERGED_MEAN_STATUSES=['stationary','population-stationary','initial-dependent','time-average','initial-dependent-time-average'] as const;
export type SteadyStatus=typeof CONVERGED_MEAN_STATUSES[number]|'not-converged'|'not-requested';
export const STEADY_STATUSES:readonly SteadyStatus[]=[...CONVERGED_MEAN_STATUSES,'not-converged','not-requested'];
/** A driven model's mean is a long-time phase average, not a time-independent state. */
export function isTimeAverage(status:SteadyStatus|undefined):boolean{return status==='time-average'||status==='initial-dependent-time-average';}
/** A degenerate null space: the mean depends on the specified initial state. */
export function isInitialDependent(status:SteadyStatus|undefined):boolean{return status==='initial-dependent'||status==='initial-dependent-time-average';}
export interface SteadyResult {
  converged:boolean;status:SteadyStatus;populations:Record<string,number>|null;
  /** Native full complex long-time mean, row-major in DynamicsResult.states. */
  density?:{real:number[];imag:number[]};
  residual:number|null;method?:string;reason?:string;initialDependent:boolean;
  /** Slowest relaxation time (us) of a converged mean's decaying modes: the
   * limit is reached after a few of them (null: none resolved or reported). */
  slowestRelaxationUs?:number|null;
  /** Conserved quantities (trace included) found by the slow generator; more
   * than one makes the mean initial-dependent (null: not reported). */
  conservedQuantities?:number|null;
  /** The horizon T of p̄(T) the solver echoes (us; null: t → ∞). Undefined: a solver that
   * does not echo one returns t → ∞. */
  horizonUs?:number|null;
}
export interface DynamicsResult {
  states:Level[];timesUs:number[];populations:Record<string,number[]>;totalExcited:number[];
  activeLaserIds:string[];steady:SteadyResult;
  metrics:{totalMs:number};
  validation:{maxTraceError:number;minPopulation:number};
  /** Estimated error of returned populations and all declared groups only. */
  populationAccuracy?:PopulationAccuracy;
  /** The finite trajectory's method; a mean-only result has none. */
  method?:string;assumptions:string[];
}
