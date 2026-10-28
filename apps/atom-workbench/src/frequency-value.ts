// Frequencies as the workbench reads them (ui-design.md §1.2–1.4): an optical
// frequency is kept as a large fixed part (a fine-structure centroid, or 0)
// plus a small offset, so a difference of two frequencies subtracts the large
// parts first and keeps Hz-level offsets on a ~380 THz carrier.
import type {DetuningUnit,FrequencyBaseSpec,Laser,LaserFrequencyAnchor,Level,ReferenceLevel,Scene,SpectralLine,ZeroFieldTransition} from './types.js';
import {RB87,captureLaserFrequencyAnchor,transitionReferenceMHz} from './physics.js';
import {referenceEnergyMHz,zeroFieldLevel} from './reference-energy.js';
import {formatNumber,TERMS,fSymbol} from './format.js';

export type FrequencyValue={centroidMHz:number;offsetMHz:number};
export type WavelengthUnit='nm'|'μm';
/** An energy difference read as a temperature, ΔE/k_B. */
export type TemperatureEquivalentUnit='mK';
export type FrequencyUnit=DetuningUnit|WavelengthUnit|TemperatureEquivalentUnit;
export const FREQUENCY_UNITS:readonly DetuningUnit[]=['Hz','kHz','MHz','GHz','THz'];
export const WAVELENGTH_UNITS:readonly WavelengthUnit[]=['nm','μm'];
export const TEMPERATURE_EQUIVALENT_UNITS:readonly TemperatureEquivalentUnit[]=['mK'];
export const UNIT_MHZ:Readonly<Record<DetuningUnit,number>>={Hz:1e-6,kHz:1e-3,MHz:1,GHz:1e3,THz:1e6};
export const UNIT_NM:Readonly<Record<WavelengthUnit,number>>={nm:1,'μm':1e3};
/** c in nm·MHz: λ/nm = C_NM_MHZ / (ν/MHz). */
export const C_NM_MHZ=299792458000;
/** k_B/h in MHz per mK, the ratio of the exact SI constants (k_B = 1.380 649 × 10⁻²³ J/K,
 * h = 6.626 070 15 × 10⁻³⁴ J·s): 1 mK ↔ 20.836 619 MHz. */
export const MHZ_PER_MK=1.380649e-23/6.62607015e-34/1e9;
export const isWavelengthUnit=(unit:string):unit is WavelengthUnit=>unit==='nm'||unit==='μm';
export const isTemperatureEquivalentUnit=(unit:string):unit is TemperatureEquivalentUnit=>unit==='mK';
export const isDetuningUnit=(unit:string):unit is DetuningUnit=>(FREQUENCY_UNITS as readonly string[]).includes(unit);

const centroid=(line:SpectralLine)=>RB87[line].frequencyHz/1e6;
export const value=(centroidMHz:number,offsetMHz:number):FrequencyValue=>({centroidMHz,offsetMHz});
export function total(v:FrequencyValue):number{return v.centroidMHz+v.offsetMHz;}
/** a − b, the large parts first. */
export function difference(a:FrequencyValue,b:FrequencyValue):number{return (a.centroidMHz-b.centroidMHz)+(a.offsetMHz-b.offsetMHz);}
export function gap(from:FrequencyValue,to:FrequencyValue):FrequencyValue{return value(to.centroidMHz-from.centroidMHz,to.offsetMHz-from.offsetMHz);}

export function laserValue(laser:Laser):FrequencyValue{return anchorValue(captureLaserFrequencyAnchor(laser));}
/** A resolved carrier (a laser's or a reference frequency's) as a value. */
export function anchorValue(anchor:Pick<LaserFrequencyAnchor,'line'|'offsetMHz'>):FrequencyValue{return value(centroid(anchor.line),anchor.offsetMHz);}
export function transitionValue(t:ZeroFieldTransition):FrequencyValue{
  const fromD1=t.fromManifold==='D1';
  return value(centroid(t.line)-(fromD1?centroid('D1'):0),transitionReferenceMHz(t.line,t.Fg,t.Fe,t.fromManifold)+(fromD1?centroid('D1'):0));
}
export function levelValue(level:Pick<Level,'manifold'|'energyMHz'>):FrequencyValue{return value(level.manifold==='ground'?0:centroid(level.manifold),level.energyMHz);}
export function referenceLevelValue(reference:ReferenceLevel,levels:readonly Level[],scene?:FrequencyScene):FrequencyValue|undefined{
  const energy=referenceEnergyMHz(reference,levels,scene),anchor=levels.find(l=>l.id===reference.anchorId);
  if(energy===undefined||!anchor)return undefined;
  const base=anchor.manifold==='ground'?0:centroid(anchor.manifold);return value(base,energy-base);
}
type StateScene=FrequencyScene&Pick<Scene,'references'>;
/** A state's energy as a frequency, E/h: a real state, or a reference level by its definition. */
export function stateValue(id:string,levels:readonly Level[],scene:StateScene):FrequencyValue|undefined{
  const level=levels.find(l=>l.id===id);if(level)return levelValue(level);
  const r=scene.references?.find(r=>r.id===id);return r?referenceLevelValue(r,levels,scene):undefined;
}
/** The same state with no field and no light (ui-design.md §1.8): a real state's zero-field
 * hyperfine level; a reference level's own definition on its anchor's zero-field level -- its
 * photon and offset δ define it, they are not a shift. */
export function zeroFieldStateValue(id:string,levels:readonly Level[],scene:StateScene):FrequencyValue|undefined{
  const level=levels.find(l=>l.id===id);if(level){const zero=zeroFieldLevel(level);return zero&&levelValue(zero);}
  const r=scene.references?.find(r=>r.id===id);return r?referenceLevelValue({...r,anchorMode:'zero-field'},levels,scene):undefined;
}

// ------------------------------------------------------------ bases
/** What a frequency is read relative to: nothing (the absolute frequency), a zero-field transition, a laser. */
export type FrequencyBase=FrequencyBaseSpec;
type FrequencyScene=Pick<Scene,'lasers'|'referenceFrequencies'>;
export function transitionKey(t:ZeroFieldTransition):string{return `transition:${t.fromManifold??'ground'}-${t.line}-${t.Fg}-${t.Fe}`;}
export function baseKey(base:FrequencyBase):string{return base.kind==='absolute'?'absolute':base.kind==='laser'||base.kind==='reference'?`${base.kind}:${base.id}`:transitionKey(base);}
export function parseBase(key:string):FrequencyBase|undefined{
  if(key==='absolute')return {kind:'absolute'};
  const laser=/^laser:(.+)$/.exec(key);if(laser)return {kind:'laser',id:laser[1]!};
  const reference=/^reference:(.+)$/.exec(key);if(reference)return {kind:'reference',id:reference[1]!};
  const t=/^transition:(ground|D1)-(D1|D2)-([12])-(\d)$/.exec(key);
  return t?{kind:'transition',...(t[1]==='D1'?{fromManifold:'D1' as const}:{}),line:t[2] as SpectralLine,Fg:Number(t[3]) as 1|2,Fe:Number(t[4])}:undefined;
}
export function baseValue(base:FrequencyBase,scene:FrequencyScene):FrequencyValue|undefined{
  if(base.kind==='absolute')return value(0,0);
  if(base.kind==='transition')return transitionValue(base);
  if(base.kind==='reference'){const r=scene.referenceFrequencies?.find(r=>r.id===base.id);return r?.frequencyAnchor?anchorValue(r.frequencyAnchor):undefined;}
  const laser=scene.lasers.find(l=>l.id===base.id);return laser?laserValue(laser):undefined;
}
export function transitionName(t:ZeroFieldTransition):string{
  const from=t.fromManifold??'ground';return `${TERMS[from]} ${fSymbol(from)}=${t.Fg} → ${TERMS[t.line]} ${fSymbol(t.line)}=${t.Fe}`;
}
/** A transition's photon frequency |ν| (an emission's gap read the other way). */
export function photon(v:FrequencyValue):FrequencyValue{return total(v)<0?value(-v.centroidMHz,-v.offsetMHz):v;}
/** What a column reading this base and unit says it is: "ν − ν(探测光) / MHz". */
export function readingHeading(base:FrequencyBase,scene:FrequencyScene,unit:FrequencyUnit):string{
  return base.kind==='absolute'?`ν / ${unit}`:`ν − ν(${baseName(base,scene)}) / ${unit}`;
}
export function baseName(base:FrequencyBase,scene:FrequencyScene):string{
  return base.kind==='absolute'?'绝对':base.kind==='transition'?transitionName(base):base.kind==='reference'?scene.referenceFrequencies?.find(r=>r.id===base.id)?.name??base.id:scene.lasers.find(l=>l.id===base.id)?.name??base.id;
}

// ------------------------------------------------------------ reading
/** A frequency in `unit` (a wavelength of an absolute positive frequency, NaN otherwise; a
 * temperature ΔE/k_B of an energy difference). */
export function inUnit(mhz:number,unit:FrequencyUnit):number{
  if(isWavelengthUnit(unit))return mhz>0?C_NM_MHZ/mhz/UNIT_NM[unit]:NaN;
  if(isTemperatureEquivalentUnit(unit))return mhz/MHZ_PER_MK;
  return mhz/UNIT_MHZ[unit];
}
/** A readout's words: twelve significant digits at most, grouped in threes on
 * both sides of the point (SI), "≈ " when the value had more; a plus sign on a
 * relative reading. */
export function readoutText(reading:number,signed:boolean):string{
  if(!Number.isFinite(reading))return '—';
  const twelve=Number(reading.toPrecision(12)),rounded=Math.abs(twelve-reading)>Math.abs(reading)*1e-14;
  const text=formatNumber(twelve,{digits:12,signed}).replace(/\.(\d+)/,(_,digits:string)=>'.'+digits.replace(/(\d{3})(?=\d)/g,'$1 '));
  return `${rounded?'≈ ':''}${text}`;
}
