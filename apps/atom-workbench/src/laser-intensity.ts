import type {IntensityMode,IntensityUnit,Laser,LaserIntensityAnchor,PowerUnit,WaistUnit} from './types.js';
import type {Reading} from './unit-input.js';
import {laserFrequencyMHz} from './physics.js';

/** The units power, waist and local intensity are written or read in, each with its size in the
 * stored unit (mW, μm, mW/cm²). An input's number keeps its digits when its unit changes (a new
 * quantity); a computed one is converted (ui-design.md §0.4). */
export const POWER_UNITS:readonly PowerUnit[]=['W','mW','μW','nW'];
export const POWER_MW:Readonly<Record<PowerUnit,number>>={W:1e3,mW:1,'μW':1e-3,nW:1e-6};
export const WAIST_UNITS:readonly WaistUnit[]=['mm','μm','nm'];
export const WAIST_UM:Readonly<Record<WaistUnit,number>>={mm:1e3,'μm':1,nm:1e-3};
export const INTENSITY_UNITS:readonly IntensityUnit[]=['kW/cm²','W/cm²','mW/cm²','μW/cm²'];
export const INTENSITY_MWCM2:Readonly<Record<IntensityUnit,number>>={'kW/cm²':1e6,'W/cm²':1e3,'mW/cm²':1,'μW/cm²':1e-3};
/** How a local intensity is given: directly; by power and waist; by power and the numerical
 * aperture the beam fills, its waist following from the wavelength. */
export const INTENSITY_MODES:readonly IntensityMode[]=['intensity','power','na'];
/** What a mode starts from when the laser has none yet. */
export const DEFAULT_WAIST_UM=1.2,DEFAULT_NUMERICAL_APERTURE=.5;
/** c in μm·MHz: λ/μm = C_UM_MHZ / (ν/MHz). */
const C_UM_MHZ=299792458;

/** The largest unit the value reads at least 1 in (the base unit for zero, the smallest below it). */
function naturalUnit<U extends string>(value:number,size:Readonly<Record<U,number>>,units:readonly U[],base:U):U{
  const magnitude=Math.abs(value);if(magnitude===0||!Number.isFinite(magnitude))return base;
  return units.find(u=>magnitude>=size[u])??units.at(-1)!;
}
/** A value that may not be computable now (a frequency not yet resolved): NaN then. */
const attempt=(value:()=>number):number=>{try{return value();}catch{return NaN;}};
/** The unit each is shown in: the operator's, else the natural one. */
export const powerUnit=(l:Laser):PowerUnit=>l.powerUnit??naturalUnit(l.powerMW??0,POWER_MW,POWER_UNITS,'mW');
export const waistUnit=(l:Laser):WaistUnit=>l.waistUnit??naturalUnit(attempt(()=>laserWaistUm(l)),WAIST_UM,WAIST_UNITS,'μm');
export const intensityUnit=(l:Laser):IntensityUnit=>l.intensityUnit??naturalUnit(attempt(()=>laserIntensityMWcm2(l)),INTENSITY_MWCM2,INTENSITY_UNITS,'mW/cm²');

export function checkNumericalAperture(value:number):number{
  if(!Number.isFinite(value)||!(value>0)||value>1)throw new Error('数值孔径必须满足 0 < NA ≤ 1（原子在真空中）。');return value;
}
/** The laser's vacuum wavelength λ = c/ν, in μm. */
export function laserWavelengthUm(laser:Laser):number{return C_UM_MHZ/laserFrequencyMHz(laser);}
/** The 1/e² intensity radius at the atom, in μm: the operator's in power mode; in NA mode the
 * focus of a Gaussian beam filling the aperture, w₀ = λ/(π·NA) -- the paraxial relation with NA
 * the sine of the 1/e² far-field half-angle, approximate at large NA. */
export function laserWaistUm(laser:Laser):number{
  return laser.intensityMode==='na'?laserWavelengthUm(laser)/(Math.PI*checkNumericalAperture(laser.numericalAperture!)):laser.waistUm??DEFAULT_WAIST_UM;
}

/** Active input fields only. Direct legacy edits invalidate the canonical value; inactive
 * representation fields do not override the selected input. NA mode includes the wavelength,
 * so a frequency change moves its waist and intensity. */
export function laserIntensityViewKey(laser:Laser):string {
  const mode=laser.intensityMode;
  return JSON.stringify(mode==='power'?['power',laser.powerMW,laser.waistUm]:mode==='na'?['na',laser.powerMW,laser.numericalAperture,laserWavelengthUm(laser)]:['intensity',laser.intensityMWcm2]);
}
function finiteIntensity(value:number):number {
  if(!Number.isFinite(value)||value<0)throw new Error('原子处强度必须是非负有限数值。');return value;
}
function checkInputs(laser:Laser):void{
  if(laser.intensityMode==='power'&&(!Number.isFinite(laser.powerMW)||laser.powerMW!<0||!Number.isFinite(laser.waistUm)||!(laser.waistUm!>0)))
    throw new Error('光功率必须非负，束腰必须大于零。');
  if(laser.intensityMode==='na'){
    if(!Number.isFinite(laser.powerMW)||laser.powerMW!<0)throw new Error('光功率必须是非负有限数值。');
    checkNumericalAperture(laser.numericalAperture!);
  }
}
export function captureLaserIntensityAnchor(laser:Laser):LaserIntensityAnchor {
  checkInputs(laser);
  const viewKey=laserIntensityViewKey(laser);
  if(laser.intensityAnchor?.viewKey===viewKey)return {intensityMWcm2:finiteIntensity(laser.intensityAnchor.intensityMWcm2),viewKey};
  const intensity=laser.intensityMode==='power'||laser.intensityMode==='na'?2*laser.powerMW!/(Math.PI*laserWaistUm(laser)**2)*1e8:laser.intensityMWcm2;
  return {intensityMWcm2:finiteIntensity(intensity),viewKey};
}
/** Local on-axis intensity in mW/cm², preserving exact physical inputs across
 * representation changes so coherent cancellation is not spoiled by 1 ULP. */
export function laserIntensityMWcm2(laser:Laser):number{return captureLaserIntensityAnchor(laser).intensityMWcm2;}
export function normalizeLaserIntensity(laser:Laser):Laser {
  if(laser.intensityMode!==undefined&&!INTENSITY_MODES.includes(laser.intensityMode))throw new Error('未知的强度输入模式。');
  const anchor=captureLaserIntensityAnchor(laser);laser.intensityMode??='intensity';laser.intensityMWcm2=anchor.intensityMWcm2;
  laser.intensityAnchor={intensityMWcm2:anchor.intensityMWcm2,viewKey:laserIntensityViewKey(laser)};return laser;
}
/** Another way to give the same local intensity: the new mode's waist or NA is the laser's own
 * (or the default), and the power is what gives the intensity there. */
export function setLaserIntensityMode(laser:Laser,mode:IntensityMode):void {
  if(!INTENSITY_MODES.includes(mode))throw new Error('未知的强度输入模式。');
  const intensity=laserIntensityMWcm2(laser),next:Laser={...laser,intensityMode:mode,intensityMWcm2:intensity};
  if(mode==='power')next.waistUm??=DEFAULT_WAIST_UM;
  if(mode==='na')next.numericalAperture??=DEFAULT_NUMERICAL_APERTURE;
  if(mode!=='intensity'){
    const waist=laserWaistUm(next);
    if(!Number.isFinite(waist)||!(waist>0))throw new Error('束腰必须是正的有限数值。');
    next.powerMW=(intensity/1e8)*(Math.PI*waist**2/2);
    if(!Number.isFinite(next.powerMW)||(intensity>0&&next.powerMW===0))throw new Error('当前强度与束腰的功率表示超出浮点数范围。');
  }
  next.intensityAnchor={intensityMWcm2:intensity,viewKey:laserIntensityViewKey(next)};Object.assign(laser,next);
}
export function setLaserLocalIntensity(laser:Laser,value:number):void {
  const next:Laser={...laser,intensityMode:'intensity',intensityMWcm2:finiteIntensity(value),intensityAnchor:undefined};
  normalizeLaserIntensity(next);Object.assign(laser,next);
}
/** The power is an input of power and NA modes; given in intensity mode it selects power mode. */
export function setLaserPowerMW(laser:Laser,value:number):void {
  if(!Number.isFinite(value)||value<0)throw new Error('光功率必须是非负有限数值。');
  const mode=laser.intensityMode==='na'?'na':'power';
  const next:Laser={...laser,intensityMode:mode,powerMW:value,...(mode==='power'?{waistUm:laser.waistUm??DEFAULT_WAIST_UM}:{}),intensityAnchor:undefined};
  normalizeLaserIntensity(next);Object.assign(laser,next);
}
export function setLaserWaistUm(laser:Laser,value:number):void {
  if(!Number.isFinite(value)||!(value>0))throw new Error('束腰必须是正的有限数值。');
  const next:Laser={...laser,waistUm:value,intensityAnchor:undefined};normalizeLaserIntensity(next);Object.assign(laser,next);
}
export function setLaserNumericalAperture(laser:Laser,value:number):void {
  const next:Laser={...laser,numericalAperture:checkNumericalAperture(value),intensityAnchor:undefined};normalizeLaserIntensity(next);Object.assign(laser,next);
}

// ------------------------------------------------------------ in the operator's units
/** Power, waist and local intensity as the inspector shows them: the number in its unit. */
export const powerReading=(l:Laser):Reading<PowerUnit>=>{const unit=powerUnit(l);return {value:(l.powerMW??0)/POWER_MW[unit],unit};};
export const waistReading=(l:Laser):Reading<WaistUnit>=>{const unit=waistUnit(l);return {value:laserWaistUm(l)/WAIST_UM[unit],unit};};
export const intensityReading=(l:Laser):Reading<IntensityUnit>=>{const unit=intensityUnit(l);return {value:laserIntensityMWcm2(l)/INTENSITY_MWCM2[unit],unit};};
const known=<U extends string>(unit:U,units:readonly U[]):U=>{if(!units.includes(unit))throw new Error('单位无效。');return unit;};
/** The operator's number in `unit`, and the unit with it: a unit picked for an input keeps its
 * digits, so the quantity changes with it. A refused number changes nothing. */
export function writeLaserPower(laser:Laser,value:number,unit:PowerUnit):void{setLaserPowerMW(laser,value*POWER_MW[known(unit,POWER_UNITS)]);laser.powerUnit=unit;}
export function writeLaserWaist(laser:Laser,value:number,unit:WaistUnit):void{setLaserWaistUm(laser,value*WAIST_UM[known(unit,WAIST_UNITS)]);laser.waistUnit=unit;}
export function writeLaserIntensity(laser:Laser,value:number,unit:IntensityUnit):void{setLaserLocalIntensity(laser,value*INTENSITY_MWCM2[known(unit,INTENSITY_UNITS)]);laser.intensityUnit=unit;}
