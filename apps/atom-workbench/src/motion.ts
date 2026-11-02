// 1D motion along one lab axis (ui-design.md §6.20): which lasers take part and how -- each beam's
// wave vector runs along the axis, against it or perpendicular to it -- what a motion solve asks
// for, and the initial temperature as the operator writes it (§0.4).
import type {DynamicsSettings,Laser,MotionAxis,MotionSettings,Scene,TemperatureUnit,Vec3} from './types.js';
import {directionVector} from './direction.js';
import {captureLaserFrequencyAnchor,drivingLasers,laserFrequencyMHz} from './physics.js';
import {TEMPERATURE_K,TEMPERATURE_UNITS} from './temperature.js';
import {formatNumber} from './format.js';
import type {Reading} from './unit-input.js';

export const MOTION_AXES:readonly MotionAxis[]=['x','y','z'];
const COMPONENT:Readonly<Record<MotionAxis,number>>={x:0,y:1,z:2};
/** Two lab axes perpendicular to the motion axis and the axis itself, a right-handed triple: the
 * order of the Cartesian emission operators (full-master-model.ts). */
export const MOTION_FRAME:Readonly<Record<MotionAxis,readonly [Vec3,Vec3,Vec3]>>={
  x:[[0,1,0],[0,0,1],[1,0,0]],y:[[0,0,1],[1,0,0],[0,1,0]],z:[[1,0,0],[0,1,0],[0,0,1]],
};

/** What a motion solve asks for (the request options' `motion`): the axis and the initial temperature (μK). */
export interface MotionRequest {axis:MotionAxis;temperatureUK:number}
/** The motion of a solve with these settings: none while 1D 运动 is off. */
export function motionRequest(settings:Pick<DynamicsSettings,'motion'>):MotionRequest|undefined{
  const motion=settings.motion;return motion?.enabled?{axis:motion.axis,temperatureUK:motion.temperatureUK}:undefined;
}
/** A request's motion as sent: an axis and a finite positive temperature, nothing else. */
export function readMotionRequest(value:unknown):MotionRequest{
  const m=value as Record<string,unknown>|null;
  if(!m||typeof m!=='object'||Array.isArray(m)||Object.keys(m).some(key=>key!=='axis'&&key!=='temperatureUK')||!MOTION_AXES.includes(m.axis as MotionAxis)||
    typeof m.temperatureUK!=='number'||!Number.isFinite(m.temperatureUK)||!(m.temperatureUK>0))throw new Error('Invalid motion options');
  return {axis:m.axis as MotionAxis,temperatureUK:m.temperatureUK};
}

/** A laser of a motion solve and its wave vector on the axis, k·axis/|k|: +1 or −1, a travelling
 * wave along or against the axis; 0, a beam perpendicular to it, uniform along the axis. */
export interface MotionBeam {laser:Laser;axial:-1|0|1}
/** The beams of a motion solve: every laser that drives the dynamics (drivingLasers). Each runs
 * along the axis, against it or perpendicular to it -- decided exactly, as directions are exact
 * rotations (direction.ts): no angle tolerance. Any other angle has no 1D form, and the beams with
 * an axial component are travelling waves of one photon momentum ħk: each is on the D2 line.
 * Otherwise the motion is refused with the reason, naming the lasers. */
export function motionBeams(scene:Scene,axis:MotionAxis):MotionBeam[]{
  const along=COMPONENT[axis],oblique:string[]=[],offLine:string[]=[];
  const beams=drivingLasers(scene).map(laser=>{
    const k=directionVector(scene.fields,laser.direction),transverse=k.filter((_,i)=>i!==along);
    if(transverse.every(c=>c===0)){
      const axial=Math.sign(k[along]!) as -1|1;
      if(captureLaserFrequencyAnchor(laser).line!=='D2')offLine.push(`「${laser.name}」（${formatNumber(299792458e3/laserFrequencyMHz(laser),{digits:4})} nm）`);
      return {laser,axial};
    }
    if(k[along]!==0)oblique.push(`「${laser.name}」（与 ${axis} 轴成 ${formatNumber(Math.acos(Math.max(-1,Math.min(1,k[along]!)))*180/Math.PI,{digits:4})}°）`);
    return {laser,axial:0 as const};
  });
  const problems=[...(oblique.length?[`${oblique.join('、')}与运动轴既不平行也不垂直，1D 运动只含沿运动轴（正向或反向）传播或垂直于它的光`]:[]),
    ...(offLine.length?[`${offLine.join('、')}沿运动轴传播，但不在 D2 线上；沿运动轴传播的光必须都在 D2 线上`]:[])];
  if(problems.length)throw new Error(`1D 运动不支持当前光束：${problems.join('；')}。`);
  return beams;
}
/** Why the scene's motion cannot be solved, or empty: nothing to refuse while 1D 运动 is off. */
export function motionRefusal(scene:Scene):string{
  const request=motionRequest(scene.dynamics);if(!request)return '';
  try{motionBeams(scene,request.axis);return '';}catch(error){return error instanceof Error?error.message:String(error);}
}

/** The unit factor: how many μK one `unit` is. */
const microkelvin=(unit:TemperatureUnit):number=>TEMPERATURE_K[unit]/TEMPERATURE_K['μK'];
/** The initial temperature as the operator wrote it: the number in its unit. */
export function motionTemperatureReading(motion:MotionSettings):Reading<TemperatureUnit>{return {value:motion.temperatureUK/microkelvin(motion.temperatureUnit),unit:motion.temperatureUnit};}
/** The operator's initial temperature: `value` in `unit`, written with its unit. A unit picked for
 * it keeps its digits, so 30 μK read in mK is 30 mK (§0.4). */
export function setMotionTemperature(motion:MotionSettings,value:number,unit:TemperatureUnit):void{
  if(!TEMPERATURE_UNITS.includes(unit))throw new Error('温度单位无效');
  const uK=value*microkelvin(unit);if(!Number.isFinite(uK)||!(uK>0))throw new Error('温度必须是有限正数');
  motion.temperatureUK=uK;motion.temperatureUnit=unit;
}
