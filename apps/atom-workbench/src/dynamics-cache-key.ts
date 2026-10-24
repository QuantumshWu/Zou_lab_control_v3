import {captureLaserFrequencyAnchor,composeMagneticField,laserIntensityMWcm2,laserDirectionAxis} from './physics.js';
import type {DynamicsSettings,Scene} from './types.js';

/** The explicit quantization axis is part of the operator convention, even
 * when two field descriptions happen to round to the same Cartesian vector. */
export function dynamicsCacheKey(scene:Scene,settings:DynamicsSettings,steadyOnly:boolean):string {
  const {fieldVector,fieldAxis}=composeMagneticField(scene.fields);
  return JSON.stringify({field:fieldVector,fieldAxis,lasers:scene.lasers.filter(l=>l.enabled).map(l=>{const anchor=captureLaserFrequencyAnchor(l);return [l.id,l.regime??'near',anchor.line,anchor.offsetMHz,laserIntensityMWcm2(l),laserDirectionAxis(l),l.polarizationAngleDeg,l.ellipticityDeg,l.phaseDeg??0];}),initialStateId:settings.initialStateId,durationUs:settings.durationUs,samples:settings.samples,steadyOnly});
}
