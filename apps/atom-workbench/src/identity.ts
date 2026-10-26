// Object identity colours (ui-design.md §0.2): every laser keeps one colour,
// from the workbench palette tokens (style.css --identity-0 … --identity-7), in
// the scene list, its level-diagram arrows and transitions, and its 3D beam and
// polarization; magnetic fields are always --field-ink.
import type {Scene} from './types.js';

export const IDENTITY_SLOTS=8;
/** A laser's slot: its place in the scene's laser list. */
export function laserSlot(scene:Pick<Scene,'lasers'>,id:string):number{const index=scene.lasers.findIndex(l=>l.id===id);return (index<0?0:index)%IDENTITY_SLOTS;}
export function identityToken(slot:number):string{return `--identity-${((slot%IDENTITY_SLOTS)+IDENTITY_SLOTS)%IDENTITY_SLOTS}`;}
/** For CSS and inline SVG: var(--identity-n). */
export function laserColour(scene:Pick<Scene,'lasers'>,id:string):string{return `var(${identityToken(laserSlot(scene,id))})`;}
/** For a canvas: the token's value (#rrggbb) on `node`. */
export function resolveColour(node:Element,token:string,fallback='#808080'):string{
  const value=getComputedStyle(node).getPropertyValue(token).trim();
  return /^#[0-9a-f]{6}$/i.test(value)?value:/^#[0-9a-f]{3}$/i.test(value)?`#${[...value.slice(1)].map(c=>c+c).join('')}`:fallback;
}
