// Object identity colours (ui-design.md §0.2): every laser keeps one colour,
// from the workbench palette tokens (style.css --identity-0 … --identity-7), in
// the scene list, its level-diagram arrows and transitions, and its 3D beam and
// polarization; magnetic fields are always --field-ink.
import type {Scene} from './types.js';

export const IDENTITY_SLOTS=8;
/** A laser's slot: its place among the scene's lasers that are no reflection; a reflection wears its
 * source's (reflection.ts), so adding or removing one moves no other colour. A reference frequency's
 * slot follows the lasers'. */
export function laserSlot(scene:Pick<Scene,'lasers'|'referenceFrequencies'>,id:string):number{
  const own=scene.lasers.filter(l=>!l.reflection),of=scene.lasers.find(l=>l.id===id)?.reflection?.of??id;
  const index=own.findIndex(l=>l.id===of);if(index>=0)return index%IDENTITY_SLOTS;
  const reference=scene.referenceFrequencies.findIndex(r=>r.id===id);return reference<0?0:(own.length+reference)%IDENTITY_SLOTS;
}
export function identityToken(slot:number):string{return `--identity-${((slot%IDENTITY_SLOTS)+IDENTITY_SLOTS)%IDENTITY_SLOTS}`;}
/** For CSS and inline SVG: var(--identity-n). */
export function laserColour(scene:Pick<Scene,'lasers'|'referenceFrequencies'>,id:string):string{return `var(${identityToken(laserSlot(scene,id))})`;}
/** For a canvas: the token's value (#rrggbb) on `node`. */
export function resolveColour(node:Element,token:string,fallback='#808080'):string{
  const value=getComputedStyle(node).getPropertyValue(token).trim();
  return /^#[0-9a-f]{6}$/i.test(value)?value:/^#[0-9a-f]{3}$/i.test(value)?`#${[...value.slice(1)].map(c=>c+c).join('')}`:fallback;
}
