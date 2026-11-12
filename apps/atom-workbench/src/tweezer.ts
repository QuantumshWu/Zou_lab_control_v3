// The tweezer (ui-design.md §6.21): the laser the operator designates (设为光镊) as the trap motion
// holds the atom in. A scene names at most one (Scene.tweezerId); designating another moves it.
// The designation says nothing about the beam: a laser given by power and waist or by NA is a
// Gaussian beam focused at the atom whether it is the tweezer or not, and every motion solve keeps
// each beam's profile; it only says which beam is the trap -- with the beams of its frequency (its reflection, a
// crossed beam of the same laser), one coherent field: the motion engine's trap light (rust-solver/motion SPEC §1.3).
import type {Laser,Scene} from './types.js';
import {laserIntensityMWcm2} from './laser-intensity.js';

type Lasers=Pick<Scene,'lasers'|'tweezerId'>;

/** The designated tweezer, if the scene names one of its lasers. */
export function tweezerOf(scene:Lasers):Laser|undefined{return scene.tweezerId===undefined?undefined:scene.lasers.find(l=>l.id===scene.tweezerId);}
/** Designates `laser` as the tweezer -- the one designated before no longer is -- or, undefined, none. */
export function setTweezer(scene:Lasers,laser:Laser|undefined):void{
  if(!laser){delete scene.tweezerId;return;}
  if(!scene.lasers.includes(laser))throw new Error(`「${laser.name}」不是场景中的激光。`);
  scene.tweezerId=laser.id;
}
/** After lasers leave the scene: a designation that names none of its lasers is dropped. */
export function keepTweezer(scene:Lasers):void{if(scene.tweezerId!==undefined&&!tweezerOf(scene))delete scene.tweezerId;}
/** Why the scene has no tweezer to hold the atom, or empty: none designated, or the designated one
 * off, dark, or a plane wave (given by its local intensity) -- named. */
export function tweezerRefusal(scene:Lasers):string{
  const tweezer=tweezerOf(scene),head='运动需要光镊：';
  if(!tweezer)return `${head}请在一束激光的参数中打开“设为光镊”，或在布居演化页关闭“运动”。`;
  if(!tweezer.enabled)return `${head}光镊「${tweezer.name}」已关闭。`;
  if(!(laserIntensityMWcm2(tweezer)>0))return `${head}光镊「${tweezer.name}」的强度为零。`;
  if(tweezer.intensityMode==='intensity')return `${head}光镊「${tweezer.name}」按局域强度给出，是平面波；请按功率与束腰或功率与数值孔径给出它的强度（焦点在原子处的高斯光束）。`;
  return '';
}
