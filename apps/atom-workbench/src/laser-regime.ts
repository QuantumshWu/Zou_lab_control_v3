import type {Scene} from './types.js';
import {atomicSystem,complexPolarization,composeMagneticField,laserCarrierOffsetMHz,laserIntensityMWcm2,RB87} from './physics.js';
import {STATIC_MIXING_LIMIT,displayMixing} from './spectroscopy.js';
import {idLabel,withUnit} from './format.js';
/** A detuning as the workbench writes one: its natural unit, four digits. */
const frequency=(mhz:number):string=>{const a=Math.abs(mhz),[scale,unit]=a>=1e6?[1e6,'THz']:a>=1e3?[1e3,'GHz']:a>=1?[1,'MHz']:[1e-3,'kHz'];return withUnit(mhz/scale,unit,{digits:4});};
/** A laser's regime decides its physics (full-master-model.ts): 'near' is a
 * rotating-wave drive of the master equation, 'far' only a static
 * second-order light shift. `valid` is the regime's own condition on the
 * nearest allowed transition:
 * - far: |Δ| ≥ 100 max(Γ/2π, Ω/2π) for every allowed dipole (the static
 *   shift's own mixing check is in spectroscopy.ts; farValidity has both);
 * - near: the nearest transition within max(1 GHz, 10 Ω/2π).
 * A disabled or dark laser is always valid. */
export interface LaserClassification {id:string;regime:'near'|'far';nearestMHz:number;transition:string;maximumMixing:number;valid:boolean;message:string;line:'D1'|'D2';groundF:1|2}
/** Classification uses actual allowed dipoles and polarization, not a diagram origin. */
export function classifyLasers(scene:Scene):LaserClassification[]{
 const {fieldMagnitudeG,fieldVector,fieldAxis}=composeMagneticField(scene.fields);
 return scene.lasers.map(l=>{
  let nearest=Infinity,transition='',maxMix=0,farValid=true,nearScale=1000,line:'D1'|'D2'=l.line,F:1|2=2;
  const intensity=laserIntensityMWcm2(l),pol=complexPolarization(l,fieldVector,fieldAxis),electric=Math.sqrt(2*intensity*10/(8.8541878128e-12*299792458));
  for(const branch of ['D1','D2'] as const){const sys=atomicSystem(branch,fieldMagnitudeG);for(let e=0;e<sys.excited.length;e++)for(let g=0;g<8;g++){
   const a=sys.excited[e]!,b=sys.ground[g]!,q=a.m-b.m;if(q < -1||q>1)continue;
   const rabi=Math.abs(sys.dipoles[e*8+g]!)*RB87.D2.dipole*electric/6.62607015e-34/1e6*Math.hypot(pol[q+1]!.re,pol[q+1]!.im);
   if(rabi===0)continue;
   const delta=Math.abs(laserCarrierOffsetMHz(l,branch)-(a.energyMHz-b.energyMHz));
   if(delta<nearest){nearest=delta;transition=`${idLabel(b.id)} → ${idLabel(a.id)}`;line=branch;F=b.F as 1|2;nearScale=Math.max(1000,10*rabi);}
   maxMix=Math.max(maxMix,rabi/Math.max(Number.MIN_VALUE,delta));
   if(delta<100*Math.max(RB87[branch].gammaHz/(2*Math.PI*1e6),rabi))farValid=false;
  }}
  const regime=l.regime??'near',valid=!l.enabled||intensity===0||(regime==='far'?farValid:nearest<=nearScale);
  return {id:l.id,regime,nearestMHz:nearest,transition,maximumMixing:maxMix,valid,line,groundF:F,message:valid?'':regime==='far'
    ?`${l.name}：离最近的允许跃迁（${transition}）只有 ${frequency(nearest)}，不满足“静态光移”的远失谐条件 |Δ| ≥ 100·max(Γ/2π, Ω/2π)。请改为“进入主方程”，或调整频率、强度。`
    :`${l.name}：离最近的允许跃迁（${transition}）有 ${frequency(nearest)}，超出“进入主方程”的近共振范围 ${frequency(nearScale)}（max(1 GHz, 10·Ω/2π)）。离共振这样远的光请改为“静态光移”，或把频率调到跃迁附近。`};
 });
}
export function assertLaserRegimes(scene:Scene):void{const invalid=classifyLasers(scene).find(c=>!c.valid);if(invalid)throw new Error(invalid.message);}
/** Whether a laser may be 'far' (a static shift), by the master model's two checks
 * for it alone at its carrier: main's detuning rule above (detuningRule) and the static
 * shift's mixing |Ω/2Δ| ≤ STATIC_MIXING_LIMIT (1%) with its minimum gaps (spectroscopy.ts
 * displayMixing). For the default 852 nm tweezer the detuning rule binds first, at
 * |Ω/2Δ| ≈ 0.53%. Beams of one carrier enter the model together, so their sum is
 * what the model checks. It changes nothing. */
export function farValidity(scene:Scene,laserId:string):{mixing:number;mixingLimit:number;withinGap:boolean;detuningRule:boolean;valid:boolean;message:string}{
 const laser=scene.lasers.find(l=>l.id===laserId);
 if(!laser||laserIntensityMWcm2(laser)===0)return {mixing:0,mixingLimit:STATIC_MIXING_LIMIT,withinGap:true,detuningRule:true,valid:true,message:''};
 const rule=classifyLasers({...scene,lasers:[{...laser,enabled:true,regime:'far'}]})[0]!,m=displayMixing(scene,laserId);
 const mixingValid=m.ratio<=STATIC_MIXING_LIMIT&&m.withinGap,valid=rule.valid&&mixingValid;
 const message=!rule.valid?rule.message:mixingValid?'':!m.withinGap?`${laser.name}：静态光移的虚跃迁离共振太近（D 线 < 100 MHz 或高态 < 10 GHz），超出静态色散模型。请改为“进入主方程”，或调整频率。`:`${laser.name}：静态光移的混合比 |Ω/2Δ| = ${(100*m.ratio).toPrecision(3)}%，超过 ${100*STATIC_MIXING_LIMIT}% 的静态色散模型范围。请改为“进入主方程”，或调整频率、强度。`;
 return {mixing:m.ratio,mixingLimit:STATIC_MIXING_LIMIT,withinGap:m.withinGap,detuningRule:rule.valid,valid,message};
}
