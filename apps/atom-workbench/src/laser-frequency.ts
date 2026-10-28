import { RB87, captureLaserFrequencyAnchor, expressionAnchor, expressionKey, resolveSceneFrequencies } from './physics.js';
import type { DetuningUnit, FrequencyBaseSpec, FrequencyDisplayUnit, FrequencyExpr, Laser, LaserFrequencyAnchor, Scene, SpectralLine, ZeroFieldTransition } from './types.js';

export const DETUNING_UNITS: readonly DetuningUnit[] = ['Hz','kHz','MHz','GHz','THz'];
export const DETUNING_UNIT_MHZ: Readonly<Record<DetuningUnit,number>> = {
  Hz: 1e-6, kHz: 1e-3, MHz: 1, GHz: 1e3, THz: 1e6,
};
/** The scene file's bound on a detuning (±5000 THz), in any unit it is written in. */
export const MAX_DETUNING_MHZ = 5e9;
const C_NM_MHZ = 299792458000;

/** All existing zero-field hyperfine energy differences across D-line manifolds. The transition
 * reference is independent of the selected initial state and magnetic field. */
export function zeroFieldTransitionOptions(line?:SpectralLine):(ZeroFieldTransition & {label:string})[] {
  const result:(ZeroFieldTransition & {label:string})[]=[];
  for(const item of (line?[line]:['D2','D1']) as SpectralLine[]) {
    for(const Fg of [1,2] as const)for(let Fe=item==='D1'?1:0;Fe<=(item==='D1'?2:3);Fe++) {
      result.push({line:item,Fg,Fe,label:`${item} · F=${Fg} → ${item==='D1'?'F″':'F′'}=${Fe}`});
    }
  }
  if(!line||line==='D2')for(const Fg of [1,2] as const)for(const Fe of [0,1,2,3])result.push({fromManifold:'D1',line:'D2',Fg,Fe,label:`5P₁/₂ F″=${Fg} → 5P₃/₂ F′=${Fe}`});
  return result;
}

function checkReference(reference:ZeroFieldTransition):void {
  if(!zeroFieldTransitionOptions().some(item=>item.line===reference.line&&item.Fg===reference.Fg&&item.Fe===reference.Fe&&(item.fromManifold??'ground')===(reference.fromManifold??'ground')))
    throw new Error('请选择两个存在的零场超精细能级。');
}

/** The unit a frequency difference reads naturally in: the largest whose reading is at least 1. */
export function naturalDetuningUnit(detuningMHz:number):DetuningUnit {
  if(!Number.isFinite(detuningMHz))throw new Error('失谐必须是有限数值。');
  const magnitude=Math.abs(detuningMHz);
  return magnitude>=1e6?'THz':magnitude>=1e3?'GHz':magnitude>=1||magnitude===0?'MHz':magnitude>=1e-3?'kHz':'Hz';
}

// ---------------------------------------------------------------- the frequency expression (ui-design.md §1.2–1.3)
/** A laser or a reference frequency: an object whose frequency is defined. */
export type DefinedFrequency={id:string;name:string;frequency:FrequencyExpr;frequencyUnit:FrequencyDisplayUnit;frequencyAnchor?:LaserFrequencyAnchor};
type FrequencyScene=Pick<Scene,'lasers'|'referenceFrequencies'>;
type Carrier={line:SpectralLine;offsetMHz:number};
const WAVELENGTH_NM:Readonly<Record<'nm'|'μm',number>>={nm:1,'μm':1e3};
const isWavelength=(unit:FrequencyDisplayUnit|undefined):unit is 'nm'|'μm'=>unit==='nm'||unit==='μm';

/** The key of a defined object in the scene's frequency graph. */
export function objectKey(scene:FrequencyScene,object:DefinedFrequency):string {return scene.lasers.includes(object as Laser)?`laser:${object.id}`:`reference:${object.id}`;}
export function objectOf(scene:FrequencyScene,key:string):DefinedFrequency|undefined {
  const [kind,id]=[key.slice(0,key.indexOf(':')),key.slice(key.indexOf(':')+1)];
  return kind==='laser'?scene.lasers.find(l=>l.id===id):kind==='reference'?scene.referenceFrequencies.find(r=>r.id===id):undefined;
}
const followsKey=(object:DefinedFrequency):string|undefined=>{const f=object.frequency;return f&&'base' in f&&(f.base.kind==='laser'||f.base.kind==='reference')?`${f.base.kind}:${f.base.id}`:undefined;};
/** The base an expression is written against ('absolute' for a wavelength). */
export function expressionBase(expr:FrequencyExpr):FrequencyBaseSpec {return 'wavelengthNm' in expr?{kind:'absolute'}:expr.base;}
/** Every object that follows `key`, directly or through others. */
export function followers(scene:FrequencyScene,key:string):string[] {
  const all=[...scene.lasers,...scene.referenceFrequencies],found:string[]=[],queue=[key];
  while(queue.length){const at=queue.shift()!;for(const o of all){const k=objectKey(scene,o);if(followsKey(o)===at&&!found.includes(k)){found.push(k);queue.push(k);}}}
  return found;
}
/** The object's carrier, resolved with the scene. */
export function objectAnchor(scene:FrequencyScene,object:DefinedFrequency):Carrier {
  resolveSceneFrequencies(scene);
  if(scene.lasers.includes(object as Laser))return captureLaserFrequencyAnchor(object as Laser);
  if(!object.frequencyAnchor)throw new Error('频率尚未解析。');return object.frequencyAnchor;
}
const difference=(a:Carrier,b:Carrier)=>(a.offsetMHz-b.offsetMHz)+(a.line===b.line?0:(RB87[a.line].frequencyHz-RB87[b.line].frequencyHz)/1e6);
const absolute=(a:Carrier)=>RB87[a.line].frequencyHz/1e6+a.offsetMHz;
/** Writes an object's expression and resolves the scene; a refused one leaves the old.
 * `keep`: the same frequency in another form -- its carrier is kept bit for bit. */
function writeExpression(scene:FrequencyScene,object:DefinedFrequency,expr:FrequencyExpr,unit?:FrequencyDisplayUnit,keep=false):void {
  const before={frequency:object.frequency,frequencyUnit:object.frequencyUnit,frequencyAnchor:object.frequencyAnchor};
  const carrier=keep?objectAnchor(scene,object):undefined,follows='base' in expr&&(expr.base.kind==='laser'||expr.base.kind==='reference')?`${expr.base.kind}:${expr.base.id}`:undefined;
  const base=carrier&&follows?objectAnchor(scene,objectOf(scene,follows)!):undefined;
  object.frequency=expr;if(unit)object.frequencyUnit=unit;
  if(carrier)object.frequencyAnchor={line:carrier.line,offsetMHz:carrier.offsetMHz,viewKey:expressionKey(expr,base)};
  try{resolveSceneFrequencies(scene);}catch(error){Object.assign(object,before);resolveSceneFrequencies(scene);throw error;}
}
/** Re-expresses the object against another base without changing its frequency (§1.3).
 * An object follows neither itself nor anything that follows it. */
export function setFrequencyBase(scene:FrequencyScene,object:DefinedFrequency,base:FrequencyBaseSpec):void {
  const key=objectKey(scene,object),target=base.kind==='laser'||base.kind==='reference'?`${base.kind}:${base.id}`:undefined;
  if(target&&(target===key||followers(scene,key).includes(target)))throw new Error('不能跟随自己，也不能跟随跟随它的对象。');
  const here=objectAnchor(scene,object);
  if(base.kind==='absolute'){
    // Absolute: a wavelength stays one; a frequency reads in its unit.
    if(isWavelength(object.frequencyUnit))writeExpression(scene,object,{wavelengthNm:C_NM_MHZ/absolute(here)},undefined,true);
    else writeExpression(scene,object,{base:{kind:'absolute'},offsetMHz:absolute(here)},'THz',true);
    return;
  }
  if(base.kind==='transition'){const {kind:_k,...t}=base;checkReference(t);}
  const other=target?objectOf(scene,target):undefined;if(target&&!other)throw new Error('频率基准不存在。');
  const offset=difference(here,base.kind==='transition'?expressionAnchor({base,offsetMHz:0}):objectAnchor(scene,other!));
  // A relative value keeps its unit; one coming from an absolute frequency or a wavelength reads naturally.
  const wasRelative=!!object.frequency&&'base' in object.frequency&&object.frequency.base.kind!=='absolute';
  const unit=wasRelative&&object.frequencyUnit&&!isWavelength(object.frequencyUnit)?object.frequencyUnit:naturalDetuningUnit(offset);
  writeExpression(scene,object,{base,offsetMHz:offset},unit,true);
}
/** The definition's value in its unit: the offset, the absolute frequency or the wavelength. */
export function frequencyValue(object:DefinedFrequency):{value:number;unit:FrequencyDisplayUnit} {
  const f=object.frequency;
  if('wavelengthNm' in f){const unit=isWavelength(object.frequencyUnit)?object.frequencyUnit:'nm';return {value:f.wavelengthNm/WAVELENGTH_NM[unit],unit};}
  const unit=object.frequencyUnit&&!isWavelength(object.frequencyUnit)?object.frequencyUnit:naturalDetuningUnit(f.offsetMHz);
  return {value:f.offsetMHz/DETUNING_UNIT_MHZ[unit],unit};
}
/** The operator's number, in `unit` (the definition's own by default). A unit picked for the
 * number keeps its digits (ui-design.md §0.4): the offset becomes that many of the new unit; on an
 * absolute base a wavelength unit makes it that vacuum wavelength and a frequency unit that
 * frequency. A number the scene refuses (a frequency not positive, past ±5000 THz) changes nothing. */
export function setFrequencyValue(scene:FrequencyScene,object:DefinedFrequency,value:number,unit:FrequencyDisplayUnit=frequencyValue(object).unit):void {
  if(!Number.isFinite(value))throw new Error('频率必须是有限数值。');
  const f=object.frequency;
  if(isWavelength(unit)){
    if(!('wavelengthNm' in f)&&f.base.kind!=='absolute')throw new Error('只有绝对频率可以写成波长。');
    if(!(value>0))throw new Error('真空波长必须是正数。');
    writeExpression(scene,object,{wavelengthNm:value*WAVELENGTH_NM[unit]},unit);return;
  }
  if(!DETUNING_UNITS.includes(unit))throw new Error('未知的频率单位。');
  const offset=value*DETUNING_UNIT_MHZ[unit];if(Math.abs(offset)>MAX_DETUNING_MHZ)throw new Error('频率超出 ±5000 THz。');
  writeExpression(scene,object,{base:'wavelengthNm' in f?{kind:'absolute'}:f.base,offsetMHz:offset},unit);
}
/** What the definition's value is called. */
export function frequencyValueLabel(object:DefinedFrequency):string {
  const f=object.frequency;
  if('wavelengthNm' in f)return '波长 λ';
  return f.base.kind==='absolute'?'频率 ν':f.base.kind==='transition'?'失谐 Δ':'偏频 Δ';
}
/** The base at the end of an object's chain: a zero-field transition or absolute. */
function rootBase(scene:FrequencyScene,object:DefinedFrequency):FrequencyBaseSpec {
  for(let at:DefinedFrequency|undefined=object,steps=0;at&&steps<64;steps++){const key=followsKey(at);if(!key)return at.frequency?expressionBase(at.frequency):{kind:'absolute'};at=objectOf(scene,key);}
  return {kind:'absolute'};
}
/** Before an object goes (§1.3): those that follow it directly are written against the
 * zero-field or absolute base its own chain ends in, their frequencies unchanged. Their names. */
export function releaseFollowers(scene:FrequencyScene,object:DefinedFrequency):string[] {
  const key=objectKey(scene,object),root=rootBase(scene,object),names:string[]=[];
  for(const o of [...scene.lasers,...scene.referenceFrequencies])if(followsKey(o)===key){setFrequencyBase(scene,o,root);names.push(o.name);}
  return names;
}
