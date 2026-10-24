import type {DynamicsSettings,DurationUnit} from './types.js';
export const DURATION_UNITS:DurationUnit[]=['ns','μs','ms','s'];
export const DURATION_US:Record<DurationUnit,number>={ns:.001,'μs':1,ms:1000,s:1e6};
export function durationUnit(settings:Pick<DynamicsSettings,'durationUs'|'durationUnit'>):DurationUnit{
 return settings.durationUnit??(settings.durationUs>=1e6?'s':settings.durationUs>=1000?'ms':settings.durationUs<1?'ns':'μs');
}
export function durationValue(settings:Pick<DynamicsSettings,'durationUs'|'durationUnit'>):number{return settings.durationUs/DURATION_US[durationUnit(settings)];}
export function setDurationUnit(settings:DynamicsSettings,unit:DurationUnit):number{
 if(!DURATION_UNITS.includes(unit))throw new Error('时间单位无效');
 const value=settings.durationUs/DURATION_US[unit];if(!Number.isFinite(value)||value<=0)throw new Error('此单位无法表示当前时长');
 settings.durationUnit=unit;return value;
}
export function setDurationValue(settings:DynamicsSettings,value:number):void{
 const us=value*DURATION_US[durationUnit(settings)];if(!Number.isFinite(us)||us<=0)throw new Error('时长必须是正的有限数值');
 settings.durationUs=us;
}
export function timeDisplayUnit(us:number,preferred?:DurationUnit):{name:DurationUnit;factor:number}{
 const unit=preferred??durationUnit({durationUs:us});return {name:unit,factor:1/DURATION_US[unit]};
}
export function durationText(us:number,preferred?:DurationUnit):string{const unit=timeDisplayUnit(us,preferred);return `${Number((us*unit.factor).toPrecision(6))} ${unit.name}`;}
