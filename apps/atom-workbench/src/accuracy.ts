// Where a computed result misses the population tolerance ε (ui-design.md §6.16). Every computed
// result is shown; the samples whose estimated error exceeds ε are marked, never withheld.
import {POPULATION_TOLERANCE,type DurationUnit} from './types.js';
import {formatNumber} from './format.js';
import {timeDisplayUnit} from './duration.js';

/** A population error (an absolute probability) as a percentage: "0.61%". */
export function errorPercent(error:number):string{return `${formatNumber(error*100,{digits:2})}%`;}

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
 * the rest are within ε ("t ≤ 0.021 μs 估计误差最大 0.61%（超过 0.5%），其余时刻 ≤ 0.5%"); empty when none does. */
export function partialSentence(timesUs:readonly number[],errors:readonly number[],unit?:DurationUnit,tolerance=POPULATION_TOLERANCE):string{
  const s=partialSummary(timesUs,errors,unit,tolerance);
  return s?`${s.where} 估计误差最大 ${errorPercent(s.worst)}（超过 ${errorPercent(tolerance)}），其余时刻 ≤ ${errorPercent(tolerance)}`:'';
}
/** The same, short, for a line that says more (p̄(T)'s error, a stopped solve): "t ≤ 0.021 μs 误差最大
 * 0.61%"; several runs only by their number ("2 段时刻误差最大 0.9%"), the chart's band shows where. */
export function partialShort(timesUs:readonly number[],errors:readonly number[],unit?:DurationUnit,tolerance=POPULATION_TOLERANCE):string{
  const s=partialSummary(timesUs,errors,unit,tolerance);return s?`${s.runs>1?`${s.runs} 段时刻`:`${s.where} `}误差最大 ${errorPercent(s.worst)}`:'';
}
