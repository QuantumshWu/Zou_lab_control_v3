import { absoluteEnergyMHz, atomicSystem } from './physics.js';
import type { Level, ReferenceLevel } from './types.js';

const zeroFieldAnchors=new Map<string,Level>();
/** An annotation energy only; references never enter the dynamical basis. */
export function referenceEnergyMHz(reference:ReferenceLevel,currentLevels:readonly Level[]):number|undefined {
  let anchor=currentLevels.find(level=>level.id===reference.anchorId);
  if(!anchor)return undefined;
  if(reference.anchorMode==='zero-field') {
    if(!zeroFieldAnchors.has(anchor.id)) {
      const system=atomicSystem(anchor.manifold==='D1'?'D1':'D2',0);
      for(const level of [...system.ground,...system.excited])zeroFieldAnchors.set(level.id,level);
    }
    anchor=zeroFieldAnchors.get(anchor.id);
  }
  return anchor?absoluteEnergyMHz(anchor)+reference.offsetMHz:undefined;
}
