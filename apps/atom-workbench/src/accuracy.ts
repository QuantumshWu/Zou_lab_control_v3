// Where a computed result misses the population tolerance ε (ui-design.md §6.16). Every computed
// result is shown; the samples whose estimated error exceeds ε are marked, never withheld.
import {POPULATION_TOLERANCE,type DurationUnit} from './types.js';
import {formatNumber} from './format.js';
import {timeDisplayUnit} from './duration.js';

/** A population error (an absolute probability) as a percentage: "0.61%". */
export function errorPercent(error:number):string{return `${formatNumber(error*100,{digits:2})}%`;}
/** The tolerance a result's trajectory was solved to and is marked against (ui-design.md §6.16, §6.20): the one it
 * echoes, ε for the population core and the semiclassical 3D motion's own target (5%) for a motion result; ε where it
 * states none (a mean alone). Every sentence about a trajectory's samples names this one: the test and the words take
 * the same number. */
export function resultTolerance(result:{populationAccuracy?:{requestedTolerance?:number}}):number{return result.populationAccuracy?.requestedTolerance??POPULATION_TOLERANCE;}
/** A value marked beyond its tolerance, as the note beside it: its estimated error (unknown where it has none) and the
 * tolerance it was tested against ("（估计误差 6.2%，超过 5%）"). */
export function exceedsNote(error:number|null|undefined,tolerance:number):string{return `（估计误差 ${error==null?'未知':errorPercent(error)}，超过 ${errorPercent(tolerance)}）`;}

/** The most decimals `Number.prototype.toFixed` writes (the language's bound). */
const MOST_PLACES=100;
/** A ledger's items as its table lists them (ui-design.md §6.20), in the engine's order: from the largest down
 * (solver-types.ts `readMotionResult`). The column has one precision, its largest entry's: the decimal place (of a
 * percentage) of the first item's second significant digit, the digits `errorPercent` writes. `listed` are the items
 * that read above zero at that place; `rest`, those that read zero there, are not listed one by one: the table says
 * how many they are and which is the largest (its first). */
export function ledgerRows<T extends {estimate:number}>(items:readonly T[]):{places:number;listed:T[];rest:T[]}{
  const exponent=Number(((items[0]?.estimate??0)*100).toExponential(1).split('e')[1]);
  const places=Math.min(MOST_PLACES,Math.max(0,1-exponent)),shown=(item:T):boolean=>Number((item.estimate*100).toFixed(places))>0;
  return {places,listed:items.filter(shown),rest:items.filter(item=>!shown(item))};
}

/** A trajectory's per-sample error bounds, when its solver sent them. */
export function sampleErrors(result:{timesUs:readonly number[];populationAccuracy?:{sampleErrors?:number[]}}):number[]|undefined{
  const errors=result.populationAccuracy?.sampleErrors;return errors&&errors.length===result.timesUs.length?errors:undefined;
}

/** The runs of consecutive samples whose error exceeds ε, as [first, last] sample indices. */
export function outsideRuns(errors:readonly number[],tolerance=POPULATION_TOLERANCE):[number,number][]{
  const runs:[number,number][]=[];
  errors.forEach((error,i)=>{if(!(error>tolerance))return;const last=runs.at(-1);if(last&&last[1]===i-1)last[1]=i;else runs.push([i,i]);});
  return runs;
}

/** Where a trajectory misses ε, and its largest error there; undefined when it does nowhere. */
function partialSummary(timesUs:readonly number[],errors:readonly number[],unit?:DurationUnit,tolerance=POPULATION_TOLERANCE):{where:string;runs:number;worst:number}|undefined{
  const runs=outsideRuns(errors,tolerance);if(!runs.length)return undefined;
  const scale=timeDisplayUnit(timesUs.at(-1)!,unit),at=(i:number)=>`${formatNumber(timesUs[i]!*scale.factor,{digits:3})} ${scale.name}`;
  const worst=Math.max(...runs.flatMap(([a,b])=>errors.slice(a,b+1))),last=timesUs.length-1,[a,b]=runs[0]!;
  // t = 0 is the initial state itself: a run from the first or second sample starts at the start.
  return {worst,runs:runs.length,where:runs.length>1?`${runs.length} 段时刻（最早 t = ${at(a)}）`:a<=1?`t ≤ ${at(b)}`:b===last?`t ≥ ${at(a)}`:a===b?`t = ${at(a)}`:`t = ${at(a)}–${at(b)}`};
}
/** One sentence for a trajectory some of whose samples miss ε: where, its largest error, and that
 * the rest are within ε ("t ≤ 0.021 μs 估计误差最大 1.2%（超过 1%），其余时刻 ≤ 1%"); empty when none does. */
export function partialSentence(timesUs:readonly number[],errors:readonly number[],unit?:DurationUnit,tolerance=POPULATION_TOLERANCE):string{
  const s=partialSummary(timesUs,errors,unit,tolerance);
  return s?`${s.where} 估计误差最大 ${errorPercent(s.worst)}（超过 ${errorPercent(tolerance)}），其余时刻 ≤ ${errorPercent(tolerance)}`:'';
}
/** The same, short, for a line that says more (p̄(T)'s error, a stopped solve): "t ≤ 0.021 μs 误差最大
 * 1.2%"; several runs only by their number ("2 段时刻误差最大 1.8%"), the chart's band shows where. */
export function partialShort(timesUs:readonly number[],errors:readonly number[],unit?:DurationUnit,tolerance=POPULATION_TOLERANCE):string{
  const s=partialSummary(timesUs,errors,unit,tolerance);return s?`${s.runs>1?`${s.runs} 段时刻`:`${s.where} `}误差最大 ${errorPercent(s.worst)}`:'';
}
