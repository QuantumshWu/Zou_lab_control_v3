// Motion (ui-design.md §6.20): a single atom in the tweezer -- the laser the operator designates
// (tweezer.ts) -- moving in 3D (semiclassical); what a motion solve asks for, and the initial
// temperature as the operator writes it (§0.4).
import type {DynamicsSettings,MotionAxis,MotionDevice,MotionSettings,Scene,TemperatureUnit,Vec3} from './types.js';
import {TEMPERATURE_UK,TEMPERATURE_UNITS} from './temperature.js';
import {tweezerRefusal} from './tweezer.js';
import type {Reading} from './unit-input.js';

/** The lab axes a motion result's per-axis series run along, in order. */
export const MOTION_AXES:readonly MotionAxis[]=['x','y','z'];
/** The frame of the motion block's beam directions and Cartesian emission operators
 * (full-master-model.ts): the lab axes x, y, z. */
export const MOTION_FRAME:readonly [Vec3,Vec3,Vec3]=[[1,0,0],[0,1,0],[0,0,1]];
/** Where a motion solve may run, as 计算设备 lists them: the GPU first, the default. */
export const MOTION_DEVICES:readonly MotionDevice[]=['gpu','cpu'];

/** Why the scene's motion cannot be solved, or empty: nothing to refuse while 运动 is off. Motion
 * needs the designated tweezer, on, bright and focused at the atom (tweezer.ts tweezerRefusal). */
export function motionRefusal(scene:Scene):string{
  if(!scene.dynamics.motion.enabled)return '';
  try{return tweezerRefusal(scene);}catch(error){return error instanceof Error?error.message:String(error);}
}

/** What a motion solve asks for (the request options' `motion`): 3D, and the initial temperature (μK). */
export interface MotionRequest {axis:'3d';temperatureUK:number}
/** The motion of a solve with these settings: none while 运动 is off. */
export function motionRequest(settings:Pick<DynamicsSettings,'motion'>):MotionRequest|undefined{
  const motion=settings.motion;return motion.enabled?{axis:'3d',temperatureUK:motion.temperatureUK}:undefined;
}
/** A solve with these settings asks for the GPU: 运动 on, 计算设备 GPU. */
export function asksGpu(settings:Pick<DynamicsSettings,'motion'>):boolean{return settings.motion.enabled&&settings.motion.device==='gpu';}
/** A request's motion as sent: 3D and a finite positive temperature, nothing else. */
export function readMotionRequest(value:unknown):MotionRequest{
  const m=value as Record<string,unknown>|null;
  if(!m||typeof m!=='object'||Array.isArray(m)||Object.keys(m).some(key=>key!=='axis'&&key!=='temperatureUK')||m.axis!=='3d'||
    typeof m.temperatureUK!=='number'||!Number.isFinite(m.temperatureUK)||!(m.temperatureUK>0))throw new Error('Invalid motion options');
  return {axis:'3d',temperatureUK:m.temperatureUK};
}
/** A request's device as sent (the request options' `device`, with a motion only): the GPU or the CPU. */
export function readMotionDevice(value:unknown):MotionDevice{
  if(!MOTION_DEVICES.includes(value as MotionDevice))throw new Error('Invalid motion device');
  return value as MotionDevice;
}

/** The initial temperature as the operator wrote it: the number in its unit. */
export function motionTemperatureReading(motion:MotionSettings):Reading<TemperatureUnit>{return {value:motion.temperatureUK/TEMPERATURE_UK[motion.temperatureUnit],unit:motion.temperatureUnit};}
/** The operator's initial temperature: `value` in `unit`, written with its unit. A unit picked for
 * it keeps its digits, so 30 μK read in mK is 30 mK (§0.4). */
export function setMotionTemperature(motion:MotionSettings,value:number,unit:TemperatureUnit):void{
  if(!TEMPERATURE_UNITS.includes(unit))throw new Error('温度单位无效');
  const uK=value*TEMPERATURE_UK[unit];if(!Number.isFinite(uK)||!(uK>0))throw new Error('温度必须是有限正数');
  motion.temperatureUK=uK;motion.temperatureUnit=unit;
}
