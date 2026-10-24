import { captureLaserFrequencyAnchor, laserCarrierOffsetMHz, laserFrequencyMHz, laserFrequencyViewKey, transitionReferenceMHz,
  zeroFieldTransitionFrequencyMHz } from './physics.js';
import type { DetuningUnit, Laser, SpectralLine, ZeroFieldTransition } from './types.js';

export { zeroFieldTransitionFrequencyMHz };
export const DETUNING_UNITS: readonly DetuningUnit[] = ['Hz','kHz','MHz','GHz','THz'];
export const DETUNING_UNIT_MHZ: Readonly<Record<DetuningUnit,number>> = {
  Hz: 1e-6, kHz: 1e-3, MHz: 1, GHz: 1e3, THz: 1e6,
};
export const MAX_DETUNING_VALUE = 5000;
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

export function laserReferenceTransition(laser:Laser):ZeroFieldTransition {
  return {...(laser.referenceTransition??{line:laser.line,Fg:2,Fe:laser.line==='D1'?2:3})};
}

function checkReference(reference:ZeroFieldTransition):void {
  if(!zeroFieldTransitionOptions().some(item=>item.line===reference.line&&item.Fg===reference.Fg&&item.Fe===reference.Fe&&(item.fromManifold??'ground')===(reference.fromManifold??'ground')))
    throw new Error('请选择两个存在的零场超精细能级。');
}

/** MHz is always the stored detuning unit; this also reads wavelength-mode
 * lasers against the selected transition without changing their frequency. */
export function laserDetuningMHz(laser:Laser):number {
  if(laser.frequencyMode!=='wavelength')return laser.detuningMHz;
  const ref=laserReferenceTransition(laser);
  return laserCarrierOffsetMHz(laser,ref.line)-transitionReferenceMHz(ref.line,ref.Fg,ref.Fe,ref.fromManifold);
}

export function detuningInUnit(laser:Laser,unit:DetuningUnit=laser.detuningUnit??'MHz'):number {
  return laserDetuningMHz(laser)/DETUNING_UNIT_MHZ[unit];
}

/** Prefer the chosen unit, escalating only if ±5000 cannot represent the
 * physical detuning. Never clamp or silently change an optical frequency. */
export function suitableDetuningUnit(detuningMHz:number,preferred:DetuningUnit='MHz'):DetuningUnit {
  if(!Number.isFinite(detuningMHz))throw new Error('失谐必须是有限数值。');
  const index=DETUNING_UNITS.indexOf(preferred);
  if(index<0)throw new Error('未知的失谐单位。');
  for(let i=index;i<DETUNING_UNITS.length;i++) {
    const unit=DETUNING_UNITS[i];
    if(Math.abs(detuningMHz)<=MAX_DETUNING_VALUE*DETUNING_UNIT_MHZ[unit])return unit;
  }
  throw new Error('当前失谐超出 ±5000 THz。');
}

export function setLaserDetuningUnit(laser:Laser,unit:DetuningUnit):DetuningUnit {
  return laser.detuningUnit=suitableDetuningUnit(laserDetuningMHz(laser),unit);
}

export function setLaserDetuningValue(laser:Laser,value:number):void {
  if(!Number.isFinite(value)||Math.abs(value)>MAX_DETUNING_VALUE)throw new Error('失谐输入范围为 −5000 至 +5000（当前单位）。');
  const unit=laser.detuningUnit??'MHz',delta=value*DETUNING_UNIT_MHZ[unit],ref=laserReferenceTransition(laser);
  checkReference(ref);
  if(!(zeroFieldTransitionFrequencyMHz(ref.line,ref.Fg,ref.Fe,ref.fromManifold)+delta>0))throw new Error('失谐后的光学频率必须大于零。');
  // Entering a detuning explicitly selects the transition parametrization.
  const next:Laser={...laser,detuningMHz:delta,frequencyMode:'transition',referenceTransition:ref,line:ref.line,frequencyAnchor:undefined};
  next.frequencyAnchor=captureLaserFrequencyAnchor(next);
  Object.assign(laser,next);
}

/** An explicit wavelength entry is a new physical input, including when its
 * displayed digits happen to equal a converted reference-mode wavelength. */
export function setLaserWavelengthNm(laser:Laser,value:number):void {
  if(!Number.isFinite(value)||!(value>0))throw new Error('真空波长必须是正的有限数值。');
  const next:Laser={...laser,wavelengthNm:value,frequencyMode:'wavelength',frequencyAnchor:undefined};
  normalizeLaserFrequency(next);
  Object.assign(laser,next);
}

/** Change the frequency reference, not the actual laser. Even switching D lines
 * is a relabeling; the larger detuning is represented by a suitable unit. */
export function setLaserTransitionReference(laser:Laser,reference:ZeroFieldTransition):void {
  checkReference(reference);
  const anchor=captureLaserFrequencyAnchor(laser);
  const delta=laserCarrierOffsetMHz(laser,reference.line)-transitionReferenceMHz(reference.line,reference.Fg,reference.Fe,reference.fromManifold);
  const unit=suitableDetuningUnit(delta,laser.detuningUnit??'MHz');
  if(!(zeroFieldTransitionFrequencyMHz(reference.line,reference.Fg,reference.Fe,reference.fromManifold)+delta>0))throw new Error('光学频率必须大于零。');
  laser.line=reference.line;laser.referenceTransition={...reference};laser.detuningMHz=delta;laser.detuningUnit=unit;
  laser.frequencyAnchor={...anchor,viewKey:laserFrequencyViewKey(laser)};
}

export function setLaserFrequencyMode(laser:Laser,mode:'transition'|'wavelength'):void {
  const anchor=captureLaserFrequencyAnchor(laser);
  const frequency=laserFrequencyMHz(laser);
  if(!Number.isFinite(frequency)||!(frequency>0))throw new Error('光学频率必须是正的有限数值。');
  if(mode==='wavelength')laser.wavelengthNm=C_NM_MHZ/frequency;
  else {
    const delta=laserDetuningMHz(laser),unit=suitableDetuningUnit(delta,laser.detuningUnit??'MHz');
    laser.detuningMHz=delta;laser.detuningUnit=unit;
  }
  laser.frequencyMode=mode;
  laser.frequencyAnchor={...anchor,viewKey:laserFrequencyViewKey(laser)};
}

/** Upgrade a legacy beam explicitly: a missing reference always means its
 * original fixed D-line reference, so existing exported scenes keep frequency.
 * The caller validates the remaining scene fields independently. */
export function normalizeLaserFrequency(laser:Laser):Laser {
  const ref=laserReferenceTransition(laser);checkReference(ref);
  if(ref.line!==laser.line)throw new Error('跃迁参考与激光 D 线不一致。');
  if(laser.frequencyMode!==undefined&&laser.frequencyMode!=='transition'&&laser.frequencyMode!=='wavelength')throw new Error('未知的频率输入模式。');
  const frequency=laserFrequencyMHz(laser);
  if(!Number.isFinite(frequency)||!(frequency>0))throw new Error('光学频率必须是正的有限数值。');
  const delta=laserDetuningMHz(laser),unit=suitableDetuningUnit(delta,laser.detuningUnit??'MHz');
  const anchor=captureLaserFrequencyAnchor(laser);
  laser.referenceTransition=ref;laser.frequencyMode??='transition';laser.detuningMHz=delta;laser.detuningUnit=unit;
  laser.frequencyAnchor={...anchor,viewKey:laserFrequencyViewKey(laser)};
  return laser;
}
