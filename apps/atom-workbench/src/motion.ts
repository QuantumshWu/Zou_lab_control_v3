// 1D motion (ui-design.md §6.20): a single atom in an optical tweezer moving radially -- along a lab
// axis through the focus, perpendicular to every beam focused at the atom -- what a motion solve asks
// for, and the initial temperature as the operator writes it (§0.4).
import type {DynamicsSettings,Laser,MotionAxis,MotionSettings,Scene,TemperatureUnit,Vec3} from './types.js';
import {directionVector} from './direction.js';
import {drivingLasers} from './physics.js';
import {TEMPERATURE_UK,TEMPERATURE_UNITS} from './temperature.js';
import type {Reading} from './unit-input.js';

export const MOTION_AXES:readonly MotionAxis[]=['x','y','z'];
const COMPONENT:Readonly<Record<MotionAxis,number>>={x:0,y:1,z:2};
/** Two lab axes perpendicular to the motion axis and the axis itself, a right-handed triple: the
 * frame of the motion block's beam directions and Cartesian emission operators (full-master-model.ts). */
export const MOTION_FRAME:Readonly<Record<MotionAxis,readonly [Vec3,Vec3,Vec3]>>={
  x:[[0,1,0],[0,0,1],[1,0,0]],y:[[0,0,1],[1,0,0],[0,1,0]],z:[[1,0,0],[0,1,0],[0,0,1]],
};

/** The beams focused at the atom: every driving laser whose intensity is given by power and waist or
 * by NA -- a Gaussian beam with its focus at the atom. A laser given by its local intensity is a
 * plane wave. */
export function focusedBeams(scene:Scene):Laser[]{return drivingLasers(scene).filter(laser=>laser.intensityMode!=='intensity');}
/** The lab axes the atom moves along: through the focus, perpendicular to every focused beam --
 * exactly, as directions are exact rotations (direction.ts). With no focused beam there is no
 * tweezer, and with no perpendicular axis no radial line: refused with the reason, naming the beams. */
export function radialAxes(scene:Scene):MotionAxis[]{
  const focused=focusedBeams(scene);
  if(!focused.length)throw new Error('1D 运动计算光镊中单个原子的径向运动：没有聚焦在原子处的光（按功率与束腰或按数值孔径给出强度的光）。');
  const axes=MOTION_AXES.filter(axis=>focused.every(laser=>directionVector(scene.fields,laser.direction)[COMPONENT[axis]]===0));
  if(!axes.length)throw new Error(`1D 运动沿穿过焦点、与聚焦光垂直的坐标轴：没有${focused.length>1?'同时':''}垂直于${focused.map(l=>`「${l.name}」`).join('、')}的坐标轴。`);
  return axes;
}
/** The axis a motion runs along: the chosen one when it is radial, else the first radial axis (the
 * chosen one when the scene has none, as its motion is refused). */
export function motionAxis(scene:Scene,motion:MotionSettings):MotionAxis{
  let axes:MotionAxis[];try{axes=radialAxes(scene);}catch{return motion.axis;}
  return axes.includes(motion.axis)?motion.axis:axes[0]!;
}
/** Why the scene's motion cannot be solved, or empty: nothing to refuse while 1D 运动 is off. */
export function motionRefusal(scene:Scene):string{
  if(!scene.dynamics.motion?.enabled)return '';
  try{radialAxes(scene);return '';}catch(error){return error instanceof Error?error.message:String(error);}
}

/** What a motion solve asks for (the request options' `motion`): the axis and the initial temperature (μK). */
export interface MotionRequest {axis:MotionAxis;temperatureUK:number}
/** The motion of a solve with these settings in this scene: none while 1D 运动 is off. */
export function motionRequest(scene:Scene,settings:Pick<DynamicsSettings,'motion'>):MotionRequest|undefined{
  const motion=settings.motion;return motion?.enabled?{axis:motionAxis(scene,motion),temperatureUK:motion.temperatureUK}:undefined;
}
/** A request's motion as sent: an axis and a finite positive temperature, nothing else. */
export function readMotionRequest(value:unknown):MotionRequest{
  const m=value as Record<string,unknown>|null;
  if(!m||typeof m!=='object'||Array.isArray(m)||Object.keys(m).some(key=>key!=='axis'&&key!=='temperatureUK')||!MOTION_AXES.includes(m.axis as MotionAxis)||
    typeof m.temperatureUK!=='number'||!Number.isFinite(m.temperatureUK)||!(m.temperatureUK>0))throw new Error('Invalid motion options');
  return {axis:m.axis as MotionAxis,temperatureUK:m.temperatureUK};
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
