import {captureLaserFrequencyAnchor,composeMagneticField,laserIntensityMWcm2} from './physics.js';
import {directionFrame} from './direction.js';
import type {DynamicsSettings,Scene} from './types.js';
import {canonicalInitialState,resolveInitialState} from './types.js';

/** The identity of a dynamics request: every physical input of the master model and the
 * solve settings. The explicit quantization axis and frame are part of the operator
 * convention, even when two field descriptions happen to round to the same Cartesian vector;
 * each laser's resolved frame (its links followed) and the full initial state
 * (InitialStateSpec) are part of it. Display settings (a laser's 显示光移, a unit) are not.
 * `withHorizon` false: the key of what does not depend on the mean's horizon (a trajectory). */
export function dynamicsCacheKey(scene:Scene,settings:DynamicsSettings,steadyOnly:boolean,withHorizon=true):string {
  const {fieldVector,fieldAxis,frame}=composeMagneticField(scene.fields);
  return JSON.stringify({field:fieldVector,fieldAxis,frame,lasers:scene.lasers.filter(l=>l.enabled).map(l=>{const anchor=captureLaserFrequencyAnchor(l);return [l.id,anchor.line,anchor.offsetMHz,laserIntensityMWcm2(l),directionFrame(scene.fields,l.direction),l.polarizationAngleDeg,l.ellipticityDeg,l.phaseDeg??0];}),initialState:canonicalInitialState(resolveInitialState(settings)),durationUs:settings.durationUs,samples:settings.samples,...(withHorizon?{meanHorizonUs:settings.meanHorizonUs}:{}),steadyOnly});
}
