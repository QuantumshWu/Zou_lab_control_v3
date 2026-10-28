import type {DynamicsSettings,DurationUnit} from './types.js';
import type {Reading} from './unit-input.js';
export const DURATION_UNITS:DurationUnit[]=['ns','μs','ms','s'];
export const DURATION_US:Record<DurationUnit,number>={ns:.001,'μs':1,ms:1000,s:1e6};
/** The unit a time reads naturally in. */
export function naturalDurationUnit(us:number):DurationUnit{return us>=1e6?'s':us>=1000?'ms':us<1?'ns':'μs';}
export function durationValue(settings:Pick<DynamicsSettings,'durationUs'|'durationUnit'>):number{return settings.durationUs/DURATION_US[settings.durationUnit];}
/** The duration as the operator wrote it: the number and its unit. */
export function durationReading(settings:Pick<DynamicsSettings,'durationUs'|'durationUnit'>):Reading<DurationUnit>{return {value:durationValue(settings),unit:settings.durationUnit};}
/** The operator's duration: `value` in `unit` (its own unit by default). A unit picked for the
 * duration keeps its digits, so 10 μs read in ms is 10 ms (ui-design.md §0.4). */
export function setDuration(settings:DynamicsSettings,value:number,unit:DurationUnit=settings.durationUnit):void{
 if(!DURATION_UNITS.includes(unit))throw new Error('时间单位无效');
 const us=value*DURATION_US[unit];if(!Number.isFinite(us)||!(us>0))throw new Error('时长必须是正的有限数值');
 settings.durationUs=us;settings.durationUnit=unit;
}
export function timeDisplayUnit(us:number,preferred?:DurationUnit):{name:DurationUnit;factor:number}{
 const unit=preferred??naturalDurationUnit(us);return {name:unit,factor:1/DURATION_US[unit]};
}
export function durationText(us:number,preferred?:DurationUnit):string{const unit=timeDisplayUnit(us,preferred);return `${Number((us*unit.factor).toPrecision(6))} ${unit.name}`;}
