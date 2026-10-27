import type {IntensityUnit,Laser,LaserIntensityAnchor,PowerUnit,WaistUnit} from './types.js';

/** The units power, waist and local intensity are read in, each with its size in the stored unit
 * (mW, μm, mW/cm²). A unit only changes how the value is read. */
export const POWER_UNITS:readonly PowerUnit[]=['W','mW','μW','nW'];
export const POWER_MW:Readonly<Record<PowerUnit,number>>={W:1e3,mW:1,'μW':1e-3,nW:1e-6};
export const WAIST_UNITS:readonly WaistUnit[]=['mm','μm','nm'];
export const WAIST_UM:Readonly<Record<WaistUnit,number>>={mm:1e3,'μm':1,nm:1e-3};
export const INTENSITY_UNITS:readonly IntensityUnit[]=['kW/cm²','W/cm²','mW/cm²','μW/cm²'];
export const INTENSITY_MWCM2:Readonly<Record<IntensityUnit,number>>={'kW/cm²':1e6,'W/cm²':1e3,'mW/cm²':1,'μW/cm²':1e-3};
/** The largest unit the value reads at least 1 in (the base unit for zero, the smallest below it). */
function naturalUnit<U extends string>(value:number,size:Readonly<Record<U,number>>,units:readonly U[],base:U):U{
  const magnitude=Math.abs(value);if(magnitude===0||!Number.isFinite(magnitude))return base;
  return units.find(u=>magnitude>=size[u])??units.at(-1)!;
}
/** The unit each is shown in: the operator's, else the natural one. */
export const powerUnit=(l:Laser):PowerUnit=>l.powerUnit??naturalUnit(l.powerMW??0,POWER_MW,POWER_UNITS,'mW');
export const waistUnit=(l:Laser):WaistUnit=>l.waistUnit??naturalUnit(l.waistUm??0,WAIST_UM,WAIST_UNITS,'μm');
export const intensityUnit=(l:Laser):IntensityUnit=>l.intensityUnit??naturalUnit(laserIntensityMWcm2(l),INTENSITY_MWCM2,INTENSITY_UNITS,'mW/cm²');

/** Active input fields only. Direct legacy edits invalidate the canonical
 * value; inactive representation fields do not override the selected input. */
export function laserIntensityViewKey(laser:Laser):string {
  return JSON.stringify(laser.intensityMode==='power'?['power',laser.powerMW,laser.waistUm]:['intensity',laser.intensityMWcm2]);
}
function finiteIntensity(value:number):number {
  if(!Number.isFinite(value)||value<0)throw new Error('原子处强度必须是非负有限数值。');return value;
}
export function captureLaserIntensityAnchor(laser:Laser):LaserIntensityAnchor {
  const viewKey=laserIntensityViewKey(laser);
  if(laser.intensityMode==='power'&&(!Number.isFinite(laser.powerMW)||laser.powerMW!<0||!Number.isFinite(laser.waistUm)||!(laser.waistUm!>0)))
    throw new Error('光功率必须非负，束腰必须大于零。');
  if(laser.intensityAnchor?.viewKey===viewKey)return {intensityMWcm2:finiteIntensity(laser.intensityAnchor.intensityMWcm2),viewKey};
  const intensity=laser.intensityMode==='power'?2*laser.powerMW!/(Math.PI*laser.waistUm!**2)*1e8:laser.intensityMWcm2;
  return {intensityMWcm2:finiteIntensity(intensity),viewKey};
}
/** Local on-axis intensity in mW/cm², preserving exact physical inputs across
 * representation changes so coherent cancellation is not spoiled by 1 ULP. */
export function laserIntensityMWcm2(laser:Laser):number{return captureLaserIntensityAnchor(laser).intensityMWcm2;}
export function normalizeLaserIntensity(laser:Laser):Laser {
  if(laser.intensityMode!==undefined&&laser.intensityMode!=='intensity'&&laser.intensityMode!=='power')throw new Error('未知的强度输入模式。');
  const anchor=captureLaserIntensityAnchor(laser);laser.intensityMode??='intensity';laser.intensityMWcm2=anchor.intensityMWcm2;
  laser.intensityAnchor={intensityMWcm2:anchor.intensityMWcm2,viewKey:laserIntensityViewKey(laser)};return laser;
}
export function setLaserIntensityMode(laser:Laser,mode:'intensity'|'power'):void {
  const intensity=laserIntensityMWcm2(laser),next:Laser={...laser,intensityMode:mode,intensityMWcm2:intensity};
  if(mode==='power'){
    next.waistUm??=1.2;
    if(!Number.isFinite(next.waistUm)||!(next.waistUm>0))throw new Error('束腰必须是正的有限数值。');
    next.powerMW=(intensity/1e8)*(Math.PI*next.waistUm**2/2);
    if(!Number.isFinite(next.powerMW)||(intensity>0&&next.powerMW===0))throw new Error('当前强度与束腰的功率表示超出浮点数范围。');
  }
  next.intensityAnchor={intensityMWcm2:intensity,viewKey:laserIntensityViewKey(next)};Object.assign(laser,next);
}
export function setLaserLocalIntensity(laser:Laser,value:number):void {
  const next:Laser={...laser,intensityMode:'intensity',intensityMWcm2:finiteIntensity(value),intensityAnchor:undefined};
  normalizeLaserIntensity(next);Object.assign(laser,next);
}
export function setLaserPowerMW(laser:Laser,value:number):void {
  if(!Number.isFinite(value)||value<0)throw new Error('光功率必须是非负有限数值。');
  const next:Laser={...laser,intensityMode:'power',powerMW:value,waistUm:laser.waistUm??1.2,intensityAnchor:undefined};
  normalizeLaserIntensity(next);Object.assign(laser,next);
}
export function setLaserWaistUm(laser:Laser,value:number):void {
  if(!Number.isFinite(value)||!(value>0))throw new Error('束腰必须是正的有限数值。');
  const next:Laser={...laser,waistUm:value,intensityAnchor:undefined};normalizeLaserIntensity(next);Object.assign(laser,next);
}
