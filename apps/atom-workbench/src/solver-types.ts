import type {Level, SpectralLine} from './types.js';
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
  warnings:string[];method:string;
}
