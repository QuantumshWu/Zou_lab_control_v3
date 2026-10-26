import type {Level, SpectralLine} from './types.js';
import {resolvePopulationTolerance} from './types.js';
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
  if(a.status!=='estimated-converged'||a.includesProjectionCharge!==true||typeof a.requestedTolerance!=='number')invalid();
  const tolerance=resolvePopulationTolerance(a.requestedTolerance);
  const error=a.estimatedMaximumError,charge=a.projectionCharge;
  if(typeof error!=='number'||!Number.isFinite(error)||error<0||error>tolerance||typeof charge!=='number'||!Number.isFinite(charge)||charge<0||charge>error)invalid();
  if(!a.scope||typeof a.scope!=='object'||Array.isArray(a.scope))invalid();
  const scope=a.scope as Record<string,unknown>,groups=scope.groups;
  if(scope.states!==32||scope.samples!==samples||!Number.isSafeInteger(samples)||samples<1||!Array.isArray(groups)||groups.length!==POPULATION_GROUPS.length||new Set(groups).size!==groups.length||groups.some(id=>typeof id!=='string'||!POPULATION_GROUPS.some(group=>group.id===id)))invalid();
  return {status:'estimated-converged',requestedTolerance:tolerance,estimatedMaximumError:error as number,
    projectionCharge:charge as number,includesProjectionCharge:true,
    scope:{states:32,samples,groups:[...(groups as string[])]}};
}
export type SteadyStatus='stationary'|'population-stationary'|'periodic-mean'|'initial-dependent-mean'|'nonunique-initial-dependent'|'time-dependent'|'not-converged'|'oscillatory'|'nonstationary-limit'|'unresolved-slow-modes';
export interface SteadyResult {
  converged:boolean;status:SteadyStatus;populations:Record<string,number>|null;
  /** Native full complex long-time mean, row-major in DynamicsResult.states. */
  density?:{real:number[];imag:number[]};
  residual:number|null;method:string;reason?:string;initialDependent?:boolean;
}
export interface DynamicsResult {
  states:Level[];timesUs:number[];populations:Record<string,number[]>;totalExcited:number[];
  line:SpectralLine|'both';activeLaserIds:string[];steady:SteadyResult;
  metrics:{buildMs:number;propagateMs:number;totalMs:number;dimension:number;fullDimension:number;steps:number;cached:boolean;kernel:string};
  validation:{maxTraceError:number;minPopulation:number;convergenceMaxDelta:number|null;positivityError?:number};
  /** Estimated error of returned populations and all declared groups only. */
  populationAccuracy?:PopulationAccuracy;
  warnings:string[];method:string;
}
