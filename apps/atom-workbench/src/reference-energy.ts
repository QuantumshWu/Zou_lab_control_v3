import { RB87, absoluteEnergyMHz, atomicSystem, laserFrequencyMHz } from './physics.js';
import type { DetuningUnit, Level, ReferenceLevel, Scene } from './types.js';
import { DETUNING_UNITS, DETUNING_UNIT_MHZ } from './laser-frequency.js';
import type { Reading } from './unit-input.js';

/** The linked photon's frequency in MHz (0 without a link; undefined when the source is gone
 * or no scene is given to find it). */
export function photonFrequencyMHz(reference:ReferenceLevel,scene?:Pick<Scene,'lasers'|'referenceFrequencies'>):number|undefined {
  const link=reference.photon;if(!link)return 0;if(!scene)return undefined;
  if(link.source.kind==='laser'){const laser=scene.lasers.find(l=>l.id===link.source.id);return laser?laserFrequencyMHz(laser):undefined;}
  const r=scene.referenceFrequencies.find(r=>r.id===link.source.id);return r?.frequencyAnchor?RB87[r.frequencyAnchor.line].frequencyHz/1e6+r.frequencyAnchor.offsetMHz:undefined;
}

const zeroFieldLevels=new Map<string,Level>();
/** A state with no field and no light: its zero-field hyperfine level. */
export function zeroFieldLevel(level:Pick<Level,'id'|'manifold'>):Level|undefined {
  if(!zeroFieldLevels.has(level.id)) {
    const system=atomicSystem(level.manifold==='D1'?'D1':'D2',0);
    for(const state of [...system.ground,...system.excited])zeroFieldLevels.set(state.id,state);
  }
  return zeroFieldLevels.get(level.id);
}

/** An annotation energy only; references never enter the dynamical basis:
 * E = E(anchor) ± hν(photon) + h·offset. */
export function referenceEnergyMHz(reference:ReferenceLevel,currentLevels:readonly Level[],scene?:Pick<Scene,'lasers'|'referenceFrequencies'>):number|undefined {
  let anchor=currentLevels.find(level=>level.id===reference.anchorId);
  if(!anchor)return undefined;
  if(reference.anchorMode==='zero-field')anchor=zeroFieldLevel(anchor);
  const photon=photonFrequencyMHz(reference,scene);
  return anchor&&photon!==undefined?absoluteEnergyMHz(anchor)+(reference.photon?.direction==='emit'?-photon:photon)+reference.offsetMHz:undefined;
}

/** A reference level's offset δ as the operator wrote it: the number in its unit. */
export function offsetReading(reference:ReferenceLevel):Reading<DetuningUnit>{
  return {value:reference.offsetMHz/DETUNING_UNIT_MHZ[reference.offsetUnit],unit:reference.offsetUnit};
}
/** The operator's offset: `value` in `unit`, written with its unit (a unit picked for it keeps its
 * digits, ui-design.md §0.4). A refused number changes nothing. */
export function setReferenceOffset(reference:ReferenceLevel,value:number,unit:DetuningUnit):void{
  if(!DETUNING_UNITS.includes(unit))throw new Error('频率单位无效');
  const mhz=value*DETUNING_UNIT_MHZ[unit];if(!Number.isFinite(mhz)||Math.abs(mhz)>1e12)throw new Error('偏移必须是有限数值（最大 ±1 000 000 THz）');
  reference.offsetMHz=mhz;reference.offsetUnit=unit;
}
