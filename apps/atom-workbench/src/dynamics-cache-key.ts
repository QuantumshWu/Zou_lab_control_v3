import {captureLaserFrequencyAnchor,composeMagneticField,laserIntensityMWcm2,laserDirectionAxis} from './physics.js';
import type {DynamicsSettings,Scene} from './types.js';
import {canonicalInitialState,resolveInitialState,resolveMeanHorizon} from './types.js';

/** The identity of a dynamics request: every physical input of the master model and the
 * solve settings. The explicit quantization axis is part of the operator convention, even
 * when two field descriptions happen to round to the same Cartesian vector; the full
 * initial state (InitialStateSpec) is part of it. Display settings (a laser's 显示光移)
 * are not. `withHorizon` false: the key of what does not depend on the mean's horizon
 * (a trajectory). */
export function dynamicsCacheKey(scene:Scene,settings:DynamicsSettings,steadyOnly:boolean,withHorizon=true):string {
  const {fieldVector,fieldAxis}=composeMagneticField(scene.fields);
  return JSON.stringify({field:fieldVector,fieldAxis,lasers:scene.lasers.filter(l=>l.enabled).map(l=>{const anchor=captureLaserFrequencyAnchor(l);return [l.id,anchor.line,anchor.offsetMHz,laserIntensityMWcm2(l),laserDirectionAxis(l),l.polarizationAngleDeg,l.ellipticityDeg,l.phaseDeg??0];}),initialState:canonicalInitialState(resolveInitialState(settings,scene)),durationUs:settings.durationUs,samples:settings.samples,...(withHorizon?{meanHorizonUs:resolveMeanHorizon(settings)}:{}),steadyOnly});
}
