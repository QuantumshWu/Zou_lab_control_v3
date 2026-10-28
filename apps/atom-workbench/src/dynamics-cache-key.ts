import {captureLaserFrequencyAnchor,composeMagneticField,laserIntensityMWcm2} from './physics.js';
import {directionFrame} from './direction.js';
import type {DynamicsSettings,Scene} from './types.js';
import {canonicalInitialState,resolveInitialState} from './types.js';

/** What a long-time mean depends on, besides its time T: every physical input of the master
 * model and the initial state. The explicit quantization axis and frame are part of the
 * operator convention, even when two field descriptions happen to round to the same Cartesian
 * vector; each laser's resolved frame (its links followed) and the full initial state
 * (InitialStateSpec) are part of it. Display settings (a laser's 显示光移, a unit) are not, and
 * neither are the trajectory's duration and samples: a mean run returns every standard T. */
export function meanInputsKey(scene:Scene,settings:DynamicsSettings):string {
  const {fieldVector,fieldAxis,frame}=composeMagneticField(scene.fields);
  return JSON.stringify({field:fieldVector,fieldAxis,frame,lasers:scene.lasers.filter(l=>l.enabled).map(l=>{const anchor=captureLaserFrequencyAnchor(l);return [l.id,anchor.line,anchor.offsetMHz,laserIntensityMWcm2(l),directionFrame(scene.fields,l.direction),l.polarizationAngleDeg,l.ellipticityDeg,l.phaseDeg??0];}),initialState:canonicalInitialState(resolveInitialState(settings))});
}
/** What a finite trajectory depends on: the mean's inputs, the duration and the samples (not T). */
export function trajectoryInputsKey(scene:Scene,settings:DynamicsSettings):string {
  return JSON.stringify({mean:meanInputsKey(scene,settings),durationUs:settings.durationUs,samples:settings.samples});
}
