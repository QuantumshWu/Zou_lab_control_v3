// Temperatures as the operator writes them (ui-design.md §0.4): a number and its unit, K, mK or
// μK -- a thermal initial state's and the initial temperature of 1D motion.
import type {TemperatureUnit} from './types.js';
import type {Reading} from './unit-input.js';

export const TEMPERATURE_UNITS:readonly TemperatureUnit[]=['K','mK','μK'];
export const TEMPERATURE_K:Readonly<Record<TemperatureUnit,number>>={K:1,mK:1e-3,'μK':1e-6};

export function naturalTemperatureUnit(kelvin:number):TemperatureUnit{return kelvin>=1?'K':kelvin>=1e-3?'mK':'μK';}
/** A temperature as the operator wrote it: the number in its unit. */
export function temperatureReading(kelvin:number,unit:TemperatureUnit):Reading<TemperatureUnit>{return {value:kelvin/TEMPERATURE_K[unit],unit};}
