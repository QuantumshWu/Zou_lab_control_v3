import type {Level,MotionAxis,MotionPlane} from './types.js';
import {POPULATION_TOLERANCE} from './types.js';
import {POPULATION_GROUPS} from './population-selection.js';
import {isPlane,planeAxes} from './motion.js';
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
/** A motion result's distributions along an axis: on one grid (strictly increasing), at some of
 * the sample times -- `samples`, increasing indices into timesUs, the last sample among them --
 * each the probability at every grid point, summing to one. */
export interface Distributions {grid:number[];samples:number[];probabilities:number[][]}
/** The position distributions of 2D motion, as Distributions on `grid` -- uniform -- along both
 * axes of the plane: each a map, row-major -- for each grid position along the plane's second axis
 * a row of the probabilities at every grid position along its first -- summing to one. */
export interface PositionMaps {grid:number[];samples:number[];probabilities:number[][]}
/** The motion of a 1D motion result (ui-design.md §6.20): a single atom in the tweezer moving
 * radially along its axis. At every sample the temperature T = m·Var(v)/k_B (μK), the mean
 * velocity (m/s), n̄ relative to the trap frequency at the focus (`trapFrequencyKHz`, ω/2π, the one
 * the engine used) and the probability that the atom has left the trap by then; the momentum
 * distributions (grid: velocities, m/s) and the position distributions (grid: positions from the
 * focus, μm). Everything but the loss describes the atom still held. */
export interface LineMotionResult {
  axis:MotionAxis;temperatureUK:number[];meanVelocityMps:number[];trapFrequencyKHz:number;meanOccupation:number[];lossProbability:number[];
  momentum:Distributions;position:Distributions;
}
/** The motion of a 2D motion result (ui-design.md §6.21): a single atom in the tweezer moving in
 * the plane `axis` through the focus perpendicular to the tweezer, from `trajectories` quantum
 * trajectories. Per axis of the plane (`axes`, in order) and sample: T = m·Var(v)/k_B (μK) with its
 * standard error, the mean velocity (m/s) and n̄ relative to that axis's trap frequency at the focus
 * (`trapFrequencyKHz`, ω/2π); per sample the probability that the atom has left the trap by then;
 * the velocity marginal along each axis (both at the same samples) and the position maps (μm from
 * the focus). Everything but the loss describes the atom still held. */
export interface PlaneMotionResult {
  axis:MotionPlane;axes:[MotionAxis,MotionAxis];temperatureUK:[number[],number[]];temperatureErrorUK:[number[],number[]];
  meanVelocityMps:[number[],number[]];trapFrequencyKHz:[number,number];meanOccupation:[number[],number[]];lossProbability:number[];
  momentum:[Distributions,Distributions];position:PositionMaps;trajectories:number;
}
export type MotionResult=LineMotionResult|PlaneMotionResult;
/** The motion is in a plane (2D). */
export function isPlaneMotion(motion:MotionResult):motion is PlaneMotionResult{return isPlane(motion.axis);}
/** The motion block of a result with `samples` samples, along the requested axis or in the
 * requested plane, read strictly (rust-solver/src/motion_core_adapter.rs, motion_gpu_adapter.rs). */
export function readMotionResult(value:unknown,samples:number,axis:MotionAxis|MotionPlane):MotionResult {
  const invalid=():never=>{throw new Error('Invalid motion result');};
  const object=(v:unknown,keys:readonly string[]):Record<string,unknown>=>{if(!v||typeof v!=='object'||Array.isArray(v)||Object.keys(v).length!==keys.length||!keys.every(key=>Object.hasOwn(v,key)))invalid();return v as Record<string,unknown>;};
  const series=(v:unknown,length:number,admits:(x:number)=>boolean):number[]=>{if(!Array.isArray(v)||v.length!==length||!v.every(x=>typeof x==='number'&&Number.isFinite(x)&&admits(x)))invalid();return [...(v as number[])];};
  const pair=<T,>(v:unknown,read:(item:unknown)=>T):[T,T]=>Array.isArray(v)&&v.length===2?[read(v[0]),read(v[1])]:invalid();
  // Distributions of `points` probabilities each: the grid's count along an axis, its square for a map of the plane.
  const distributions=(v:unknown,gridKey:string,map=false):Distributions=>{
    const p=object(v,[gridKey,'samples','probabilities']),values=p[gridKey];
    const grid=Array.isArray(values)?series(values,values.length,()=>true):invalid();
    const at=Array.isArray(p.samples)?series(p.samples,p.samples.length,i=>Number.isInteger(i)&&i>=0&&i<samples):invalid();
    if(grid.length<2||grid.some((x,i)=>i>0&&x<=grid[i-1]!)||!at.length||at.some((i,k)=>k>0&&i<=at[k-1]!)||at.at(-1)!==samples-1)invalid();
    // Each distribution is a probability vector, its sum one as the population rows' are.
    const points=map?grid.length**2:grid.length;
    const rows=Array.isArray(p.probabilities)&&p.probabilities.length===at.length?p.probabilities.map(row=>series(row,points,x=>x>=0&&x<=1)):invalid();
    if(rows.some(row=>Math.abs(row.reduce((a,b)=>a+b,0)-1)>5e-6))invalid();
    return {grid,samples:at,probabilities:rows};
  };
  const loss=(v:unknown):number[]=>{const p=series(v,samples,x=>x>=0&&x<=1);if(p.some((x,i)=>i>0&&x<p[i-1]!))invalid();return p;};
  const frequency=(v:unknown):number=>typeof v==='number'&&Number.isFinite(v)&&v>0?v:invalid();
  if(isPlane(axis)){
    const m=object(value,['axis','axes','temperatureUK','temperatureErrorUK','meanVelocityMps','trapFrequencyKHz','meanOccupation','lossProbability','momentum','position','trajectories']);
    const axes=planeAxes(axis);
    if(m.axis!==axis||!Array.isArray(m.axes)||m.axes.length!==2||m.axes.some((name,d)=>name!==axes[d])||!Number.isSafeInteger(m.trajectories)||!((m.trajectories as number)>0))invalid();
    // The two marginals, drawn together, are on one grid at the same samples.
    const momentum=pair(m.momentum,v=>distributions(v,'velocitiesMps')),same=(a:number[],b:number[])=>a.length===b.length&&a.every((x,i)=>x===b[i]);
    if(!same(momentum[0].grid,momentum[1].grid)||!same(momentum[0].samples,momentum[1].samples))invalid();
    // The maps are images of the plane: their grid is uniform.
    const position=distributions(m.position,'positionsUm',true),grid=position.grid,step=(grid.at(-1)!-grid[0]!)/(grid.length-1);
    if(grid.some((x,i)=>Math.abs(x-(grid[0]!+i*step))>1e-9*step))invalid();
    return {axis,axes,temperatureUK:pair(m.temperatureUK,v=>series(v,samples,x=>x>=0)),temperatureErrorUK:pair(m.temperatureErrorUK,v=>series(v,samples,x=>x>=0)),
      meanVelocityMps:pair(m.meanVelocityMps,v=>series(v,samples,()=>true)),trapFrequencyKHz:pair(m.trapFrequencyKHz,frequency),
      meanOccupation:pair(m.meanOccupation,v=>series(v,samples,()=>true)),lossProbability:loss(m.lossProbability),momentum,position,trajectories:m.trajectories as number};
  }
  const m=object(value,['axis','temperatureUK','meanVelocityMps','trapFrequencyKHz','meanOccupation','lossProbability','momentum','position']);
  if(m.axis!==axis)invalid();
  return {axis,temperatureUK:series(m.temperatureUK,samples,x=>x>=0),meanVelocityMps:series(m.meanVelocityMps,samples,()=>true),
    trapFrequencyKHz:frequency(m.trapFrequencyKHz),meanOccupation:series(m.meanOccupation,samples,()=>true),lossProbability:loss(m.lossProbability),
    momentum:distributions(m.momentum,'velocitiesMps'),position:distributions(m.position,'positionsUm')};
}
export interface DynamicsResult {
  states:Level[];timesUs:number[];populations:Record<string,number[]>;totalExcited:number[];
  activeLaserIds:string[];mean:MeanResult;
  /** A motion solve's motion, along an axis or in a plane; a motion result has no long-time mean. */
  motion?:MotionResult;
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
