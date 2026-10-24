import type {Scene,Laser,Level} from './types.js';
import {laserReferenceStateId} from './transition-view.js';
import {laserFrequencyMHz} from './physics.js';
/** Flatten a reference-source photon endpoint onto its physical anchor. */
export function referenceFromLaser(scene:Scene,laser:Laser,states:readonly Level[]){
 const id=laserReferenceStateId(scene,laser),reference=scene.references?.find(r=>r.id===id),state=states.find(s=>s.id===id);
 if(!reference&&!state)return undefined;
 const emission=laser.referenceProcess==='emission'||(!reference&&state?.manifold!=='ground');
 return {anchorId:reference?.anchorId??state!.id,anchorMode:reference?.anchorMode??'follow',offsetMHz:(reference?.offsetMHz??0)+(emission?-1:1)*laserFrequencyMHz(laser),emission};
}
