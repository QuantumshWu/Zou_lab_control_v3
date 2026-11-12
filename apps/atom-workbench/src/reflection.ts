// Retro-reflections (ui-design.md §6.23): a laser may be the reflection of another -- a mirror far
// away, perpendicular to that laser's beam, sends it straight back through the atom. The mirror
// (its reflectivity, the phase the round trip adds, a λ/4 plate passed twice) is the reflection's
// own; its beam is derived from the source wherever the scene is resolved (physics.ts
// resolveSceneFrequencies), so it always follows the source. A resolved reflection is an ordinary
// laser: what reads a scene -- the analysis, the master model, every motion engine -- knows no
// reflections.
import type {FrequencyExpr,Laser,LaserReflection,Scene} from './types.js';
import {DEFAULT_PLATE_ANGLE_DEG,MAXIMUM_LASERS} from './types.js';
import {captureLaserFrequencyAnchor,expressionKey,resolveSceneFrequencies} from './physics.js';
import {laserIntensityMWcm2,laserIntensityViewKey} from './laser-intensity.js';
import {oppositeDirection} from './direction.js';

/** The mirror a new reflection starts with: a perfect one, the reflection in phase with the source
 * at the atom, no wave plate (its fast axis at 45° once it is put in). */
const DEFAULT_MIRROR:Omit<LaserReflection,'of'>={reflectivity:1,phaseDeg:0,quarterWave:false,plateAngleDeg:DEFAULT_PLATE_ANGLE_DEG};
type Lasers=Pick<Scene,'lasers'>;

/** The reflection of `source`, when it has one (a source has at most one). */
export function reflectionOf(scene:Lasers,source:Laser):Laser|undefined{return scene.lasers.find(l=>l.reflection?.of===source.id);}
/** The laser a reflection reflects. */
export function sourceOf(scene:Lasers,reflection:Laser):Laser|undefined{const of=reflection.reflection?.of;return of===undefined?undefined:scene.lasers.find(l=>l.id===of);}
/** The other beam of the source–reflection pair `laser` is in, if it is in one. */
export function partnerOf(scene:Lasers,laser:Laser):Laser|undefined{return laser.reflection?sourceOf(scene,laser):reflectionOf(scene,laser);}

/** The reflection's polarization in its own frame, which keeps the source's x and reverses its y and
 * k (direction.ts oppositeDirection): the source's major axis u is the reflection's at ψ′ = −ψ, and
 * its v = k × u is the reflection's −v.
 * - A plain mirror at normal incidence keeps the lab-frame field, cos χ u + i sin χ v (up to an
 *   overall phase, the mirror's φ): in the reflection's frame that is χ′ = −χ, the handedness about
 *   the reversed k flips (σ⁺ comes back σ⁻ about its own k, still driving Δm = +1 about the source's).
 * - A double pass through a quarter-wave plate whose fast axis lies at θ in the source's frame is a
 *   half-wave plate there: it mirrors the lab field about that axis, R(θ) diag(1, −1) R(−θ). The
 *   ellipse R(ψ)(cos χ, i sin χ) becomes R(2θ − ψ)(cos χ, −i sin χ), which the reflection's frame reads
 *   as ψ′ = ψ − 2θ, χ′ = χ -- the handedness about the beam's own k stays (the MOT retro-reflection: σ⁺
 *   comes back σ⁺), and linear light comes back mirrored about the fast axis: kept at θ = ψ, turned by
 *   90° at θ = ψ ± 45° (lin⊥lin). The plate's own phase, common to both axes, is the mirror's φ.
 *   An axis ψ′ outside −90°..90° is the same axis turned by 180°, the field's sign reversed: the sign
 *   goes to the phase.
 * The optical phase at the atom is the source's plus the mirror's. */
function reflectedPolarization(source:Pick<Laser,'polarizationAngleDeg'|'ellipticityDeg'|'phaseDeg'>,mirror:Pick<LaserReflection,'phaseDeg'|'quarterWave'|'plateAngleDeg'>):Pick<Laser,'polarizationAngleDeg'|'ellipticityDeg'|'phaseDeg'>{
  // 0 − x: the negation of a zero is a zero, never −0.
  let axis=mirror.quarterWave?source.polarizationAngleDeg-2*mirror.plateAngleDeg:0-source.polarizationAngleDeg,phase=source.phaseDeg+mirror.phaseDeg;
  while(axis>90){axis-=180;phase+=180;}
  while(axis<-90){axis+=180;phase+=180;}
  while(phase>180)phase-=360;
  while(phase<-180)phase+=360;
  return {polarizationAngleDeg:axis,ellipticityDeg:mirror.quarterWave?source.ellipticityDeg:0-source.ellipticityDeg,phaseDeg:phase};
}

/** Writes a reflection's beam from its (resolved) source: the source's carrier, which it follows at
 * +0 MHz; its switch, 显示光移 and levels-chart origin; the opposite direction; the polarization of
 * reflectedPolarization; R times its power and local intensity, with its waist or NA -- a plane wave
 * stays one, and a beam focused at the atom is taken as focused back at the atom (as by a curved
 * mirror or a lens before the mirror). The reflection's display units stay its own. */
function deriveBeam(reflection:Laser,source:Laser):void{
  const mirror=reflection.reflection!,R=mirror.reflectivity,carrier=captureLaserFrequencyAnchor(source);
  const frequency:FrequencyExpr={base:{kind:'laser',id:source.id},offsetMHz:0};
  Object.assign(reflection,{enabled:source.enabled,showLightShift:source.showLightShift,line:carrier.line,frequency,frequencyUnit:'MHz',
    frequencyAnchor:{line:carrier.line,offsetMHz:carrier.offsetMHz,viewKey:expressionKey(frequency,carrier)},
    intensityMode:source.intensityMode,direction:oppositeDirection(source.direction),...reflectedPolarization(source,mirror)});
  for(const key of ['referenceStateId','referenceProcess','waistUm','numericalAperture'] as const){if(source[key]===undefined)delete reflection[key];else Object.assign(reflection,{[key]:source[key]});}
  if(source.powerMW===undefined)delete reflection.powerMW;else reflection.powerMW=source.powerMW*R;
  // The source's exact local intensity times R (its anchor, not a recomputation from rounded inputs).
  const intensity=laserIntensityMWcm2(source)*R;
  reflection.intensityMWcm2=intensity;reflection.intensityAnchor={intensityMWcm2:intensity,viewKey:laserIntensityViewKey(reflection)};
}
/** Every reflection of the scene takes its beam from its source, whose frequency is resolved. */
export function resolveReflections(scene:Lasers):void{
  for(const laser of scene.lasers){
    if(!laser.reflection)continue;
    const source=sourceOf(scene,laser);
    if(!source||source===laser||source.reflection)throw new Error(`「${laser.name}」反射的激光不存在，或本身是反射光。`);
    deriveBeam(laser,source);
  }
}

/** The name a new reflection gets: the source's, marked (within the 64 characters of a name). */
function reflectionName(source:Pick<Laser,'name'>):string{return `${source.name.slice(0,60)}（反射）`;}
/** Adds the reflection of `source` right under it: the default mirror, named after the source.
 * Refused for a reflection, for a source that has one and at the laser limit. */
export function addReflection(scene:Pick<Scene,'lasers'|'referenceFrequencies'>,source:Laser):Laser{
  if(source.reflection)throw new Error('反射光不能再被反射。');
  if(reflectionOf(scene,source))throw new Error(`「${source.name}」已有反射光。`);
  if(scene.lasers.length>=MAXIMUM_LASERS)throw new Error(`最多 ${MAXIMUM_LASERS} 束激光（反射光也计入）。`);
  const used=new Set([...scene.lasers,...scene.referenceFrequencies].map(o=>o.id)),own=`${source.id}-reflection`;
  const id=own.length<=64&&!used.has(own)?own:`reflection-${Date.now()}`;
  const reflection:Laser={...structuredClone(source),id,name:reflectionName(source),reflection:{of:source.id,...DEFAULT_MIRROR}};
  scene.lasers.splice(scene.lasers.indexOf(source)+1,0,reflection);
  try{resolveSceneFrequencies(scene);}catch(error){scene.lasers.splice(scene.lasers.indexOf(reflection),1);throw error;}
  return reflection;
}
/** 解除关联: the reflection becomes an independent laser with the beam it has now, its frequency
 * defined as its source's is (the same carrier, bit for bit); it moves to the end of the list, so
 * every other laser keeps its place and colour. */
export function detachReflection(scene:Pick<Scene,'lasers'|'referenceFrequencies'>,reflection:Laser):void{
  const source=sourceOf(scene,reflection);if(!source)throw new Error(`「${reflection.name}」反射的激光不存在。`);
  resolveSceneFrequencies(scene);
  Object.assign(reflection,{frequency:structuredClone(source.frequency),frequencyUnit:source.frequencyUnit});
  if(source.frequencyAnchor)reflection.frequencyAnchor={...source.frequencyAnchor};else delete reflection.frequencyAnchor;
  delete reflection.reflection;
  scene.lasers.splice(scene.lasers.indexOf(reflection),1);scene.lasers.push(reflection);
  resolveSceneFrequencies(scene);
}
