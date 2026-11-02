// Motion (ui-design.md §6.20–§6.22): a single atom in an optical tweezer moving radially -- along a
// lab axis through the focus perpendicular to every beam focused at the atom (1D), or in the plane
// through the focus perpendicular to the tweezer when every focused beam runs along one lab axis
// (2D 径向) -- or in 3D (semiclassical); what a motion solve asks for, and the initial temperature
// as the operator writes it (§0.4).
import type {DynamicsSettings,Laser,MotionAxis,MotionChoice,MotionPlane,MotionSettings,Scene,TemperatureUnit,Vec3} from './types.js';
import {directionVector} from './direction.js';
import {drivingLasers} from './physics.js';
import {TEMPERATURE_UK,TEMPERATURE_UNITS} from './temperature.js';
import type {Reading} from './unit-input.js';

export const MOTION_AXES:readonly MotionAxis[]=['x','y','z'];
const COMPONENT:Readonly<Record<MotionAxis,number>>={x:0,y:1,z:2};
/** The frame of the motion block's beam directions and Cartesian emission operators
 * (full-master-model.ts): for 1D two lab axes perpendicular to the motion axis and the axis itself,
 * a right-handed triple -- also the frame of 2D motion across a tweezer along that axis, in the plane
 * of the first two; for 3D the lab axes x, y, z. */
export const MOTION_FRAME:Readonly<Record<MotionAxis|'3d',readonly [Vec3,Vec3,Vec3]>>={
  x:[[0,1,0],[0,0,1],[1,0,0]],y:[[0,0,1],[1,0,0],[0,1,0]],z:[[1,0,0],[0,1,0],[0,0,1]],'3d':[[1,0,0],[0,1,0],[0,0,1]],
};
/** The plane perpendicular to each lab axis, named by the first two axes of its MOTION_FRAME in
 * order: the plane of 2D motion across a tweezer along that axis. */
export const MOTION_PLANES:Readonly<Record<MotionAxis,MotionPlane>>={x:'yz',y:'zx',z:'xy'};
/** Every motion a solve may compute: along a lab axis, in a plane, or 3D. */
export const MOTION_CHOICES:readonly MotionChoice[]=[...MOTION_AXES,...MOTION_AXES.map(axis=>MOTION_PLANES[axis]),'3d'];
/** A plane: 2D motion. */
export function isPlane(axis:MotionChoice):axis is MotionPlane{return axis==='yz'||axis==='zx'||axis==='xy';}
/** A plane's two lab axes, in its frame's order. */
export function planeAxes(plane:MotionPlane):[MotionAxis,MotionAxis]{return [plane[0] as MotionAxis,plane[1] as MotionAxis];}
/** The frame of a motion: MOTION_FRAME of its axis (or 3D), or of the axis its plane is
 * perpendicular to. */
export function motionFrame(axis:MotionChoice):readonly [Vec3,Vec3,Vec3]{
  return MOTION_FRAME[isPlane(axis)?MOTION_AXES.find(normal=>MOTION_PLANES[normal]===axis)!:axis];
}

/** The beams focused at the atom: every driving laser whose intensity is given by power and waist or
 * by NA -- a Gaussian beam with its focus at the atom. A laser given by its local intensity is a
 * plane wave. */
export function focusedBeams(scene:Scene):Laser[]{return drivingLasers(scene).filter(laser=>laser.intensityMode!=='intensity');}
/** The lab axes the atom moves along: through the focus, perpendicular to every focused beam --
 * exactly, as directions are exact rotations (direction.ts). With no focused beam there is no
 * tweezer, and with no perpendicular axis no radial line: refused with the reason, naming the beams. */
export function radialAxes(scene:Scene):MotionAxis[]{
  const focused=focusedBeams(scene);
  if(!focused.length)throw new Error('运动计算光镊中的单个原子：没有聚焦在原子处的光（按功率与束腰或按数值孔径给出强度的光）。');
  const axes=MOTION_AXES.filter(axis=>focused.every(laser=>directionVector(scene.fields,laser.direction)[COMPONENT[axis]]===0));
  if(!axes.length)throw new Error(`径向运动沿穿过焦点、与聚焦光垂直的坐标轴：没有${focused.length>1?'同时':''}垂直于${focused.map(l=>`「${l.name}」`).join('、')}的坐标轴。`);
  return axes;
}
/** Where the scene lets the atom move radially: its radial axes, then -- when they are two, every
 * focused beam running along the third axis -- the plane they span (none when its radial motion is
 * refused). 3D is chosen besides. */
export function radialChoices(scene:Scene):(MotionAxis|MotionPlane)[]{
  let axes:MotionAxis[];try{axes=radialAxes(scene);}catch{return [];}
  return axes.length===2?[...axes,MOTION_PLANES[MOTION_AXES.find(axis=>!axes.includes(axis))!]]:axes;
}
/** The motion a solve computes: 3D when chosen; else the chosen axis or plane when the scene allows
 * it; a plane it does not, its plane when it has one; else its first radial axis (the chosen one
 * when the scene has none, as its motion is refused). */
export function motionAxis(scene:Scene,motion:MotionSettings):MotionChoice{
  if(motion.axis==='3d')return '3d';
  const choices=radialChoices(scene);if(!choices.length||choices.includes(motion.axis))return motion.axis;
  return (isPlane(motion.axis)&&choices.find(isPlane))||choices[0]!;
}
/** Why the scene's motion cannot be solved, or empty: nothing to refuse while 运动 is off. 3D needs
 * a beam focused at the atom (the tweezer); 1D and 2D a radial axis besides. */
export function motionRefusal(scene:Scene):string{
  const motion=scene.dynamics.motion;if(!motion?.enabled)return '';
  try{if(motion.axis==='3d'){if(!focusedBeams(scene).length)radialAxes(scene);}else radialAxes(scene);return '';}catch(error){return error instanceof Error?error.message:String(error);}
}

/** What a motion solve asks for (the request options' `motion`): the motion and the initial temperature (μK). */
export interface MotionRequest {axis:MotionChoice;temperatureUK:number}
/** The motion of a solve with these settings in this scene: none while 运动 is off. */
export function motionRequest(scene:Scene,settings:Pick<DynamicsSettings,'motion'>):MotionRequest|undefined{
  const motion=settings.motion;return motion?.enabled?{axis:motionAxis(scene,motion),temperatureUK:motion.temperatureUK}:undefined;
}
/** A request's motion as sent: a motion and a finite positive temperature, nothing else. */
export function readMotionRequest(value:unknown):MotionRequest{
  const m=value as Record<string,unknown>|null;
  if(!m||typeof m!=='object'||Array.isArray(m)||Object.keys(m).some(key=>key!=='axis'&&key!=='temperatureUK')||!MOTION_CHOICES.includes(m.axis as MotionChoice)||
    typeof m.temperatureUK!=='number'||!Number.isFinite(m.temperatureUK)||!(m.temperatureUK>0))throw new Error('Invalid motion options');
  return {axis:m.axis as MotionChoice,temperatureUK:m.temperatureUK};
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
