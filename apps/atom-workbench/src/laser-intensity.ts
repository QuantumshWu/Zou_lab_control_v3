import type {Laser,LaserIntensityAnchor} from './types.js';

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
