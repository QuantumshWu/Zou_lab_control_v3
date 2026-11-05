import type {Level,MotionDevice} from './types.js';
import {POPULATION_TOLERANCE} from './types.js';
import {POPULATION_GROUPS} from './population-selection.js';
import {MOTION_AXES} from './motion.js';
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
/** A motion solve's approximation ledger (`diagnostics.approximations`): the error budget its
 * approximations may use, the part they use, and each approximation -- what it is, its estimated
 * error and what the engine did about it; `flagged`, when present, says where the ledger is not
 * kept, shown as the validated regime is (a reference only). */
export interface ApproximationLedger {budget:number;used:number;items:{label:string;estimate:number;action:string}[];flagged?:string;note?:string}
/** The motion of a motion result (ui-design.md §6.20): a single atom in the tweezer moving in 3D,
 * solved semiclassically, its series per lab axis x, y, z (motion.ts MOTION_AXES). At every sample
 * per axis the temperature T = m·Var(v)/k_B (μK) with its standard error and n̄ relative to that
 * axis's trap frequency at the trap's bottom (`trapFrequencyKHz`, ω/2π, the one the engine used), m⟨v²⟩/ħω − ½; the total
 * temperature (T_x + T_y + T_z)/3 with its own standard error, both the engine's; the probability
 * that the atom has left the trap by then; whether the sample lies in the method's validated regime
 * (`classical`: k_B T ≥ SEMICLASSICAL_REGIME ħω on every axis; outside it the result is a reference
 * only); per axis the momentum distributions (grid: velocities, m/s) and the position distributions
 * (grid: positions from the focus, μm). Everything but the loss describes the atom still held. Its
 * trajectories and time step; the device it ran on, why that is not the one asked for (a
 * fallback's `deviceNote`), and its approximation ledger. */
export interface MotionResult {
  temperatureUK:number[][];temperatureErrorUK:(number|null)[][];temperatureTotalUK:number[];temperatureTotalErrorUK:(number|null)[];
  meanOccupation:number[][];trapFrequencyKHz:number[];lossProbability:number[];classical:boolean[];
  momentum:Distributions[];position:Distributions[];trajectories:number;timeStepUs:number;
  device:MotionDevice;deviceNote?:string;approximations:ApproximationLedger;
}
/** The motion block of a result on the grid `timesUs` and the motion entries of its diagnostics,
 * read strictly (rust-solver/src/motion3d_adapter.rs). */
export function readMotionResult(value:unknown,diagnostics:Record<string,unknown>,timesUs:readonly number[]):MotionResult {
  const invalid=():never=>{throw new Error('Invalid motion result');};
  const samples=timesUs.length;
  const object=(v:unknown,keys:readonly string[],optional:readonly string[]=[]):Record<string,unknown>=>{
    if(!v||typeof v!=='object'||Array.isArray(v)||!keys.every(key=>Object.hasOwn(v,key))||!Object.keys(v).every(key=>keys.includes(key)||optional.includes(key)))invalid();return v as Record<string,unknown>;};
  const series=(v:unknown,length:number,admits:(x:number)=>boolean):number[]=>{if(!Array.isArray(v)||v.length!==length||!v.every(x=>typeof x==='number'&&Number.isFinite(x)&&admits(x)))invalid();return [...(v as number[])];};
  // A temperature's standard errors: null where it has none, a sample with fewer than two atoms held (the engine's `assemble`).
  const errors=(v:unknown):(number|null)[]=>{if(!Array.isArray(v)||v.length!==samples||!v.every(x=>x===null||(typeof x==='number'&&Number.isFinite(x)&&x>=0)))invalid();return [...(v as (number|null)[])];};
  const text=(v:unknown,empty=false):string=>typeof v==='string'&&(empty||v.trim()!=='')?v:invalid();
  const size=(v:unknown):number=>typeof v==='number'&&Number.isFinite(v)&&v>=0?v:invalid();
  // Distributions: on a strictly increasing grid, at increasing samples ending with the last, each a
  // probability vector, its sum one as the population rows' are.
  const distributions=(v:unknown,gridKey:string):Distributions=>{
    const p=object(v,[gridKey,'samples','probabilities']),values=p[gridKey];
    const grid=Array.isArray(values)?series(values,values.length,()=>true):invalid();
    const at=Array.isArray(p.samples)?series(p.samples,p.samples.length,i=>Number.isInteger(i)&&i>=0&&i<samples):invalid();
    if(grid.length<2||grid.some((x,i)=>i>0&&x<=grid[i-1]!)||!at.length||at.some((i,k)=>k>0&&i<=at[k-1]!)||at.at(-1)!==samples-1)invalid();
    const rows=Array.isArray(p.probabilities)&&p.probabilities.length===at.length?p.probabilities.map(row=>series(row,grid.length,x=>x>=0&&x<=1)):invalid();
    if(rows.some(row=>Math.abs(row.reduce((a,b)=>a+b,0)-1)>5e-6))invalid();
    return {grid,samples:at,probabilities:rows};
  };
  const m=object(value,['axis','temperatureUK','temperatureErrorUK','temperatureTotalUK','temperatureTotalErrorUK','meanVelocityMps','meanOccupation','trapFrequencyKHz',
    'lossProbability','classical','momentum','position','trajectories','timeStepUs']);
  function perAxis<T>(v:unknown,read:(x:unknown)=>T):T[]{const o=object(v,MOTION_AXES);return MOTION_AXES.map(a=>read(o[a]));}
  if(m.axis!=='3d'||!Array.isArray(m.classical)||m.classical.length!==samples||!m.classical.every(x=>typeof x==='boolean'))invalid();
  if(!Number.isSafeInteger(m.trajectories)||!((m.trajectories as number)>0)||typeof m.timeStepUs!=='number'||!(m.timeStepUs>0))invalid();
  // The loss: from 0 to 1, never decreasing.
  const loss=series(m.lossProbability,samples,x=>x>=0&&x<=1);if(loss.some((p,i)=>i>0&&p<loss[i-1]!))invalid();
  // The mean velocity is in the contract; no view shows it.
  perAxis(m.meanVelocityMps,v=>series(v,samples,()=>true));
  // The device it ran on, a fallback's note, and the approximation ledger.
  const device:MotionDevice=diagnostics.device==='gpu'||diagnostics.device==='cpu'?diagnostics.device:invalid();
  const note=diagnostics.deviceNote===undefined?undefined:text(diagnostics.deviceNote);
  const ledger=object(diagnostics.approximations,['budget','used','items'],['flagged','note']);
  const items=Array.isArray(ledger.items)?ledger.items.map(item=>{const i=object(item,['label','estimate','action']);return {label:text(i.label),estimate:size(i.estimate),action:text(i.action,true)};}):invalid();
  return {temperatureUK:perAxis(m.temperatureUK,v=>series(v,samples,x=>x>=0)),temperatureErrorUK:perAxis(m.temperatureErrorUK,errors),
    temperatureTotalUK:series(m.temperatureTotalUK,samples,x=>x>=0),temperatureTotalErrorUK:errors(m.temperatureTotalErrorUK),
    meanOccupation:perAxis(m.meanOccupation,v=>series(v,samples,()=>true)),trapFrequencyKHz:perAxis(m.trapFrequencyKHz,v=>typeof v==='number'&&Number.isFinite(v)&&v>0?v:invalid()),
    lossProbability:loss,classical:[...(m.classical as boolean[])],
    momentum:perAxis(m.momentum,v=>distributions(v,'velocitiesMps')),position:perAxis(m.position,v=>distributions(v,'positionsUm')),
    trajectories:m.trajectories as number,timeStepUs:m.timeStepUs as number,device,...(note===undefined?{}:{deviceNote:note}),
    approximations:{budget:size(ledger.budget),used:size(ledger.used),items,...(ledger.flagged===undefined?{}:{flagged:text(ledger.flagged)}),...(ledger.note===undefined?{}:{note:text(ledger.note)})}};
}
export interface DynamicsResult {
  states:Level[];timesUs:number[];populations:Record<string,number[]>;totalExcited:number[];
  activeLaserIds:string[];mean:MeanResult;
  /** A motion solve's motion; a motion result has no long-time mean. Its populations are the
   * probabilities that the atom is in each state and still held, p·(1 − L): with the loss L
   * (`motion.lossProbability`, 丢失) they add up to one. */
  motion?:MotionResult;
  /** The mean at every standard time T (MEAN_HORIZONS_US order), all from the one mean run;
   * the requested T's is `mean` itself. Absent when no mean was requested. */
  horizonMeans?:MeanResult[];
  /** The solve's wall time so far (ms), and on a checkpoint its run's estimate of the wall time
   * left until the final result (ms; absent before the run's history supports one). */
  metrics:{totalMs:number;estimatedRemainingMs?:number};
  validation:{maxTraceError:number;minPopulation:number};
  /** Estimated error of returned populations and all declared groups only. */
  populationAccuracy?:PopulationAccuracy;
  /** The finite trajectory's method; a mean-only result has none. */
  method?:string;assumptions:string[];
}
