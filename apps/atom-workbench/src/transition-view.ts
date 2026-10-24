import type {Analysis,Laser,Level,Scene,SpectralLine,Transition} from './types.js';
import {absoluteEnergyMHz,laserCarrierOffsetMHz,laserFrequencyMHz} from './physics.js';

/** A spectroscopy origin belongs to one beam. It never changes rho(0). */
export function laserReferenceStateId(scene:Scene,laser?:Laser):string {
  return laser?.referenceStateId??scene.dynamics?.initialStateId??`ground-${scene.initialF}-${scene.initialM}`;
}

/** Card coordinates: ground-state energy relative to the reference, and
 * excited-state optical detuning. Arrow tooltips retain the absorption / 
 * stimulated-emission detuning defined below. */
export function levelReadoutMHz(reference:Level,target:Level,laser:Laser):number|undefined {
  if(reference.manifold!=='ground'){
    if(target.manifold==='ground')return targetDetuningMHz(reference,target,laser);
    return target.manifold===reference.manifold?reference.energyMHz-target.energyMHz:undefined;
  }
  if(target.manifold==='ground')return reference.energyMHz-target.energyMHz;
  return laserCarrierOffsetMHz(laser,target.manifold)-(target.energyMHz-reference.energyMHz);
}

/** Canonical g/e matrix elements describe both directions of the Hermitian
 * light-matter Hamiltonian. Only their displayed origin/destination changes. */
export function transitionEndpoints(transition:Transition,initialId:string){
  if(transition.groundId===initialId)return {fromId:transition.groundId,toId:transition.excitedId,emission:false,deltaM:transition.me-transition.mg};
  if(transition.excitedId===initialId)return {fromId:transition.excitedId,toId:transition.groundId,emission:true,deltaM:transition.mg-transition.me};
  return undefined;
}

export function incidentTransitions(analysis:Analysis,initialId:string,laserId:string|undefined,line:SpectralLine):Transition[]{
  return analysis.transitions.filter(transition=>transition.laserId===laserId&&transition.line===line&&transitionEndpoints(transition,initialId)!==undefined);
}

/** An excited initial state has downward stimulated-emission routes in the
 * represented 5S/5P basis. Absorption into higher, absent manifolds is not drawn
 * as a simulated transition. Local carrier subtraction preserves small offsets. */
export function photonEndpoint(initial:Level,laser:Laser){
  const emission=initial.manifold!=='ground';
  return {emission,target:emission?'ground' as const:'excited' as const,
    energyMHz:initial.manifold==='ground'?absoluteEnergyMHz(initial)+laserFrequencyMHz(laser):initial.energyMHz-laserCarrierOffsetMHz(laser,initial.manifold)};
}

/** Use Δ=νL−νeg for both absorption and stimulated emission, independent of
 * arrow direction. This avoids subtracting two large absolute optical numbers. */
export function targetDetuningMHz(initial:Level,target:Level,laser:Laser):number|undefined{
  if(initial.manifold==='ground'&&target.manifold!=='ground')return laserCarrierOffsetMHz(laser,target.manifold)-(target.energyMHz-initial.energyMHz);
  if(initial.manifold!=='ground'&&target.manifold==='ground')return laserCarrierOffsetMHz(laser,initial.manifold)-(initial.energyMHz-target.energyMHz);
  return undefined;
}
