import type {Level,MotionAxis,MotionChoice} from './types.js';
import {POPULATION_TOLERANCE} from './types.js';
import {POPULATION_GROUPS} from './population-selection.js';
import {isPlane,planeAxes} from './motion.js';
/** The semiclassical 3D motion's population accuracy target (absolute): its statistics are run
 * until two standard errors of every population lie within it (rust-solver/motion-3d solve.rs). */
export const SEMICLASSICAL_POPULATION_TARGET=0.05;
/** The semiclassical 3D motion's validated regime: k_B T ≥ SEMICLASSICAL_REGIME ħω on every axis
 * (rust-solver/motion-3d solve.rs CLASSICAL); its result marks each sample (`classical`). */
export const SEMICLASSICAL_REGIME=4;
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
/** Explicit population-output qualification; never a density/PSD claim. `tolerance` is the
 * target the solve was run to: POPULATION_TOLERANCE, or the semiclassical 3D motion's own
 * (SEMICLASSICAL_POPULATION_TARGET), against which it reports two standard errors per sample. */
export function readPopulationAccuracy(value:unknown,samples:number,tolerance=POPULATION_TOLERANCE):PopulationAccuracy {
  const invalid=():never=>{throw new Error('Invalid complete populationAccuracy contract');};
  if(!value||typeof value!=='object'||Array.isArray(value))invalid();
  const a=value as Record<string,unknown>,partial=a.status==='estimated-partial';
  // The echo is the tolerance asked for.
  if((a.status!=='estimated-converged'&&!partial)||a.requestedTolerance!==tolerance)invalid();
  const error=a.estimatedMaximumError,finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0;
  // Partial exactly when the largest bound is above it.
  if(!finite(error)||partial!==(error>tolerance))invalid();
  // The per-sample bounds: one for each sample, their maximum the trajectory's. A partial result
  // says why the solver stopped short of ε.
  const errors=a.sampleErrors;
  if(!Array.isArray(errors)||errors.length!==samples||!errors.every(finite)||Math.max(...errors)!==error)invalid();
  if(partial?typeof a.reason!=='string':a.reason!==undefined)invalid();
  if(!a.scope||typeof a.scope!=='object'||Array.isArray(a.scope))invalid();
  const scope=a.scope as Record<string,unknown>,groups=scope.groups;
  if(scope.states!==32||scope.samples!==samples||!Number.isSafeInteger(samples)||samples<1||!Array.isArray(groups)||groups.length!==POPULATION_GROUPS.length||new Set(groups).size!==groups.length||groups.some(id=>typeof id!=='string'||!POPULATION_GROUPS.some(group=>group.id===id)))invalid();
  return {status:partial?'estimated-partial':'estimated-converged',requestedTolerance:tolerance,estimatedMaximumError:error as number,
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
/** A motion result's distributions along one axis: on one grid (strictly increasing), at some of
 * the sample times -- `samples`, increasing indices into timesUs, the last sample among them --
 * each the probability at every grid point, summing to one. */
export interface Distributions {grid:number[];samples:number[];probabilities:number[][]}
/** The position distributions of 2D motion, as Distributions on `grid` -- uniform -- along both
 * axes of the plane: each a map, row-major -- for each grid position along the plane's second axis
 * a row of the probabilities at every grid position along its first -- summing to one. */
export interface PositionMaps {grid:number[];samples:number[];probabilities:number[][]}
/** The motion of a motion result (ui-design.md §6.20–§6.22): a single atom in the tweezer, along
 * one lab axis (1D), in a plane (2D) or in 3D, its series per axis (`axes`: the motion axis, the
 * plane's two axes, or x, y and z). At every sample
 * per axis the temperature T = m·Var(v)/k_B (μK), the mean velocity (m/s), n̄ relative to that
 * axis's trap frequency at the focus (`trapFrequencyKHz`, ω/2π, the one the engine used); the
 * probability that the atom has left the trap by then; per axis the momentum distributions (grid:
 * velocities, m/s) and the position distributions (grid: positions from the focus, μm).
 * Everything but the loss describes the atom still held. A 3D result (semiclassical) gives T's
 * standard error, whether each sample lies in the method's validated regime (`classical`: k_B T ≥
 * SEMICLASSICAL_REGIME ħω on every axis; outside it the result is a reference only), its
 * trajectories and time step. A 2D result (quantum trajectories) gives T's standard error, its
 * trajectories, and its positions as maps of the plane (`positionMap`; `position` holds their
 * marginals). A 1D result may end in its engine's continuation (`continuedFromUs`). */
export interface MotionResult {
  axis:MotionChoice;axes:MotionAxis[];temperatureUK:number[][];meanVelocityMps:number[][];trapFrequencyKHz:number[];meanOccupation:number[][];lossProbability:number[];
  momentum:Distributions[];position:Distributions[];
  temperatureErrorUK?:number[][];classical?:boolean[];trajectories?:number;timeStepUs?:number;positionMap?:PositionMaps;
  /** The last simulated time (us) when the samples after it are the engine's continuation (its
   * prediction from the simulated history), not simulated; absent when every sample was (1D only). */
  continuedFromUs?:number;
}
/** A sample time the continuation's: after the last simulated one. */
export function continuedAt(motion:Pick<MotionResult,'continuedFromUs'>,timeUs:number):boolean{return motion.continuedFromUs!==undefined&&timeUs>motion.continuedFromUs;}
/** The motion is in a plane (2D). */
export function isPlaneMotion(motion:MotionResult):boolean{return isPlane(motion.axis);}
/** The motion block of a result on the grid `timesUs`, read strictly (rust-solver/src/motion_core_adapter.rs,
 * for 2D motion_gpu_adapter.rs and for 3D motion3d_adapter.rs). */
export function readMotionResult(value:unknown,timesUs:readonly number[],axis:MotionChoice):MotionResult {
  const invalid=():never=>{throw new Error('Invalid motion result');};
  const samples=timesUs.length;
  const object=(v:unknown,keys:readonly string[],optional:readonly string[]=[]):Record<string,unknown>=>{
    if(!v||typeof v!=='object'||Array.isArray(v)||!keys.every(key=>Object.hasOwn(v,key))||!Object.keys(v).every(key=>keys.includes(key)||optional.includes(key)))invalid();return v as Record<string,unknown>;};
  const series=(v:unknown,length:number,admits:(x:number)=>boolean):number[]=>{if(!Array.isArray(v)||v.length!==length||!v.every(x=>typeof x==='number'&&Number.isFinite(x)&&admits(x)))invalid();return [...(v as number[])];};
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
  const loss=(v:unknown)=>{const l=series(v,samples,x=>x>=0&&x<=1);if(l.some((p,i)=>i>0&&p<l[i-1]!))invalid();return l;};
  const frequency=(v:unknown):number=>typeof v==='number'&&Number.isFinite(v)&&v>0?v:invalid();
  if(axis==='3d'){
    const axes:MotionAxis[]=['x','y','z'];
    const m=object(value,['axis','temperatureUK','temperatureErrorUK','meanVelocityMps','meanOccupation','trapFrequencyKHz','lossProbability','classical','momentum','position','trajectories','timeStepUs']);
    function perAxis<T>(v:unknown,read:(x:unknown)=>T):T[]{const o=object(v,axes);return axes.map(a=>read(o[a]));}
    if(m.axis!=='3d'||!Array.isArray(m.classical)||m.classical.length!==samples||!m.classical.every(x=>typeof x==='boolean'))invalid();
    if(!Number.isSafeInteger(m.trajectories)||!((m.trajectories as number)>0)||typeof m.timeStepUs!=='number'||!(m.timeStepUs>0))invalid();
    return {axis,axes,temperatureUK:perAxis(m.temperatureUK,v=>series(v,samples,x=>x>=0)),temperatureErrorUK:perAxis(m.temperatureErrorUK,v=>series(v,samples,x=>x>=0)),
      meanVelocityMps:perAxis(m.meanVelocityMps,v=>series(v,samples,()=>true)),meanOccupation:perAxis(m.meanOccupation,v=>series(v,samples,()=>true)),
      trapFrequencyKHz:perAxis(m.trapFrequencyKHz,frequency),lossProbability:loss(m.lossProbability),classical:[...(m.classical as boolean[])],
      momentum:perAxis(m.momentum,v=>distributions(v,'velocitiesMps')),position:perAxis(m.position,v=>distributions(v,'positionsUm')),
      trajectories:m.trajectories as number,timeStepUs:m.timeStepUs as number};
  }
  if(isPlane(axis)){
    const m=object(value,['axis','axes','temperatureUK','temperatureErrorUK','meanVelocityMps','trapFrequencyKHz','meanOccupation','lossProbability','momentum','position','trajectories']);
    const axes=planeAxes(axis);
    if(m.axis!==axis||!Array.isArray(m.axes)||m.axes.length!==2||m.axes.some((name,d)=>name!==axes[d])||!Number.isSafeInteger(m.trajectories)||!((m.trajectories as number)>0))invalid();
    function pair<T>(v:unknown,read:(item:unknown)=>T):T[]{return Array.isArray(v)&&v.length===2?[read(v[0]),read(v[1])]:invalid();}
    // The two velocity marginals are on one grid at the same samples.
    const momentum=pair(m.momentum,v=>distributions(v,'velocitiesMps')),same=(a:number[],b:number[])=>a.length===b.length&&a.every((x,i)=>x===b[i]);
    if(!same(momentum[0]!.grid,momentum[1]!.grid)||!same(momentum[0]!.samples,momentum[1]!.samples))invalid();
    // The maps are images of the plane on a uniform grid; their marginals are the positions along each axis.
    const map=distributions(m.position,'positionsUm',true),grid=map.grid,n=grid.length,step=(grid.at(-1)!-grid[0]!)/(n-1);
    if(grid.some((x,i)=>Math.abs(x-(grid[0]!+i*step))>1e-9*step))invalid();
    const marginal=(along:number):Distributions=>({grid,samples:map.samples,probabilities:map.probabilities.map(row=>Array.from({length:n},(_,i)=>{
      let sum=0;for(let j=0;j<n;j++)sum+=along===0?row[j*n+i]!:row[i*n+j]!;return sum;}))});
    return {axis,axes,temperatureUK:pair(m.temperatureUK,v=>series(v,samples,x=>x>=0)),temperatureErrorUK:pair(m.temperatureErrorUK,v=>series(v,samples,x=>x>=0)),
      meanVelocityMps:pair(m.meanVelocityMps,v=>series(v,samples,()=>true)),trapFrequencyKHz:pair(m.trapFrequencyKHz,frequency),
      meanOccupation:pair(m.meanOccupation,v=>series(v,samples,()=>true)),lossProbability:loss(m.lossProbability),momentum,position:[marginal(0),marginal(1)],positionMap:map,
      trajectories:m.trajectories as number};
  }
  const m=object(value,['axis','temperatureUK','meanVelocityMps','trapFrequencyKHz','meanOccupation','lossProbability','momentum','position'],['continuedFromUs']);
  if(m.axis!==axis)invalid();
  // The continuation starts at a time of the run before its last sample: some samples are its.
  const from=m.continuedFromUs;if(from!==undefined&&!(typeof from==='number'&&Number.isFinite(from)&&from>=0&&from<timesUs.at(-1)!))invalid();
  return {axis,axes:[axis],temperatureUK:[series(m.temperatureUK,samples,x=>x>=0)],meanVelocityMps:[series(m.meanVelocityMps,samples,()=>true)],
    trapFrequencyKHz:[frequency(m.trapFrequencyKHz)],meanOccupation:[series(m.meanOccupation,samples,()=>true)],lossProbability:loss(m.lossProbability),
    momentum:[distributions(m.momentum,'velocitiesMps')],position:[distributions(m.position,'positionsUm')],...(from===undefined?{}:{continuedFromUs:from as number})};
}
/** A running solve's report between its checkpoints (a 2D motion solve's): the running batch's
 * simulated time of the duration (us), the trajectories done and, once estimated, the wall time
 * left (ms). */
export interface RunningReport {simulatedUs:number;durationUs:number;trajectories:number;estimatedRemainingMs?:number}
/** A running report, read strictly. */
export function readRunningReport(value:unknown):RunningReport{
  const v=value as Record<string,unknown>|null,finite=(x:unknown):x is number=>typeof x==='number'&&Number.isFinite(x)&&x>=0;
  if(!v||typeof v!=='object'||Array.isArray(v)||!Object.keys(v).every(k=>['simulatedUs','durationUs','trajectories','estimatedRemainingMs'].includes(k))
    ||!finite(v.simulatedUs)||!finite(v.durationUs)||!Number.isSafeInteger(v.trajectories)||(v.trajectories as number)<0
    ||(v.estimatedRemainingMs!==undefined&&!finite(v.estimatedRemainingMs)))throw new Error('Invalid running report');
  return {simulatedUs:v.simulatedUs,durationUs:v.durationUs,trajectories:v.trajectories as number,...(v.estimatedRemainingMs===undefined?{}:{estimatedRemainingMs:v.estimatedRemainingMs as number})};
}
export interface DynamicsResult {
  states:Level[];timesUs:number[];populations:Record<string,number[]>;totalExcited:number[];
  activeLaserIds:string[];mean:MeanResult;
  /** A motion solve's motion: along an axis, in a plane or in 3D; a motion result has no long-time mean. */
  motion?:MotionResult;
  /** The mean at every standard time T (MEAN_HORIZONS_US order), all from the one mean run;
   * the requested T's is `mean` itself. Absent when no mean was requested. */
  horizonMeans?:MeanResult[];
  /** The solve's wall time so far (ms), and on a checkpoint its run's estimate of the wall time
   * left until the final result (ms; absent before the run's history supports one), with
   * `estimatedRemainingAtMost` when it is an upper bound (the run may stop earlier). */
  metrics:{totalMs:number;estimatedRemainingMs?:number;estimatedRemainingAtMost?:true};
  validation:{maxTraceError:number;minPopulation:number};
  /** Estimated error of returned populations and all declared groups only. */
  populationAccuracy?:PopulationAccuracy;
  /** The finite trajectory's method; a mean-only result has none. */
  method?:string;assumptions:string[];
}
