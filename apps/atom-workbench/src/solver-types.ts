import type {Level} from './types.js';
import {POPULATION_TOLERANCE} from './types.js';
import {POPULATION_GROUPS} from './population-selection.js';
/** A trajectory's estimated population error (ui-design.md §6.16). Every computed trajectory is
 * shown: 'estimated-converged' when the solver accepted the run (every sample within the requested
 * tolerance ε and every slow class it needs resolved), 'estimated-partial' otherwise -- the samples
 * above ε are marked, and a run not accepted with every sample within ε is said unconfirmed. */
export interface PopulationAccuracy {
  status:'estimated-converged'|'estimated-partial';requestedTolerance:number;estimatedMaximumError:number;
  /** One absolute population-error bound per sample; their maximum is estimatedMaximumError. */
  sampleErrors:number[];
  /** Why the solver stopped short of ε (partial only). */
  reason?:string;
  projectionCharge:number;includesProjectionCharge:true;
  scope:{states:32;samples:number;groups:string[]};
}
/** Explicit population-output qualification; never a density/PSD claim. */
export function readPopulationAccuracy(value:unknown,samples:number):PopulationAccuracy {
  const invalid=():never=>{throw new Error('Invalid complete populationAccuracy contract');};
  if(!value||typeof value!=='object'||Array.isArray(value))invalid();
  const a=value as Record<string,unknown>,partial=a.status==='estimated-partial';
  // Every request asks for POPULATION_TOLERANCE; the echo must be that one.
  if((a.status!=='estimated-converged'&&!partial)||a.includesProjectionCharge!==true||a.requestedTolerance!==POPULATION_TOLERANCE)invalid();
  const error=a.estimatedMaximumError,charge=a.projectionCharge,finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0;
  if(!finite(error)||!finite(charge)||charge>error||(!partial&&error>POPULATION_TOLERANCE))invalid();
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
    projectionCharge:charge as number,includesProjectionCharge:true,
    scope:{states:32,samples,groups:[...(groups as string[])]}};
}
/** The native `asymptoticStatus` values: a converged long-time mean (static:
 * stationary / population-stationary / initial-dependent; driven:
 * time-average / initial-dependent-time-average), or none. */
export const CONVERGED_MEAN_STATUSES=['stationary','population-stationary','initial-dependent','time-average','initial-dependent-time-average'] as const;
export type SteadyStatus=typeof CONVERGED_MEAN_STATUSES[number]|'not-converged'|'not-requested';
export const STEADY_STATUSES:readonly SteadyStatus[]=[...CONVERGED_MEAN_STATUSES,'not-converged','not-requested'];
/** A driven model's mean is a phase average at T (its fast oscillations averaged out), not a time-independent state. */
export function isTimeAverage(status:SteadyStatus|undefined):boolean{return status==='time-average'||status==='initial-dependent-time-average';}
/** A degenerate null space: the mean depends on the specified initial state. */
export function isInitialDependent(status:SteadyStatus|undefined):boolean{return status==='initial-dependent'||status==='initial-dependent-time-average';}
export interface SteadyResult {
  /** A long-time result was computed (its status one of CONVERGED_MEAN_STATUSES). */
  converged:boolean;status:SteadyStatus;populations:Record<string,number>|null;
  /** The computed result's estimated population error (null: nothing computed). */
  estimatedError:number|null;
  /** Within the requested tolerance: the solver accepted the run and its bound is within ε. A
   * computed result not within is still shown, with its error: above ε, or unconfirmed. */
  withinTolerance:boolean;
  /** Native full complex long-time mean, row-major in DynamicsResult.states. */
  density?:{real:number[];imag:number[]};
  residual:number|null;method?:string;reason?:string;initialDependent:boolean;
  /** Slowest relaxation time (us) of a converged mean's decaying modes: the
   * limit is reached after a few of them (null: none resolved or reported). */
  slowestRelaxationUs?:number|null;
  /** Conserved quantities (trace included) found by the slow generator; more
   * than one makes the mean initial-dependent (null: not reported). */
  conservedQuantities?:number|null;
  /** The time T of the computed p̄(T), as the solver echoes it (us; null: t → ∞); undefined
   * when nothing was computed. */
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
