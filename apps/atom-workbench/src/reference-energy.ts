import { RB87, absoluteEnergyMHz, atomicSystem, laserFrequencyMHz } from './physics.js';
import type { Level, ReferenceLevel, Scene } from './types.js';

/** The linked photon's frequency in MHz (0 without a link; undefined when the source is gone
 * or no scene is given to find it). */
export function photonFrequencyMHz(reference:ReferenceLevel,scene?:Pick<Scene,'lasers'|'referenceFrequencies'>):number|undefined {
  const link=reference.photon;if(!link)return 0;if(!scene)return undefined;
  if(link.source.kind==='laser'){const laser=scene.lasers.find(l=>l.id===link.source.id);return laser?laserFrequencyMHz(laser):undefined;}
  const r=scene.referenceFrequencies?.find(r=>r.id===link.source.id);return r?.frequencyAnchor?RB87[r.frequencyAnchor.line].frequencyHz/1e6+r.frequencyAnchor.offsetMHz:undefined;
}

const zeroFieldAnchors=new Map<string,Level>();
/** An annotation energy only; references never enter the dynamical basis:
 * E = E(anchor) ± hν(photon) + h·offset. */
export function referenceEnergyMHz(reference:ReferenceLevel,currentLevels:readonly Level[],scene?:Pick<Scene,'lasers'|'referenceFrequencies'>):number|undefined {
  let anchor=currentLevels.find(level=>level.id===reference.anchorId);
  if(!anchor)return undefined;
  if(reference.anchorMode==='zero-field') {
    if(!zeroFieldAnchors.has(anchor.id)) {
      const system=atomicSystem(anchor.manifold==='D1'?'D1':'D2',0);
      for(const level of [...system.ground,...system.excited])zeroFieldAnchors.set(level.id,level);
    }
    anchor=zeroFieldAnchors.get(anchor.id);
  }
  const photon=photonFrequencyMHz(reference,scene);
  return anchor&&photon!==undefined?absoluteEnergyMHz(anchor)+(reference.photon?.direction==='emit'?-photon:photon)+reference.offsetMHz:undefined;
}
