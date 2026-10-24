import type {Laser,Scene} from './types.js';
import {atomicSystem,complexPolarization,composeMagneticField,laserCarrierOffsetMHz,laserIntensityMWcm2,RB87} from './physics.js';
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
   if(delta<nearest){nearest=delta;transition=`${b.id} → ${a.id}`;line=branch;F=b.F as 1|2;nearScale=Math.max(1000,10*rabi);}
   maxMix=Math.max(maxMix,rabi/Math.max(Number.MIN_VALUE,delta));
   if(delta<100*Math.max(RB87[branch].gammaHz/(2*Math.PI*1e6),rabi))farValid=false;
  }}
  const regime=l.regime??'near',valid=!l.enabled||intensity===0||(regime==='far'?farValid:nearest<=nearScale);
  return {id:l.id,regime,nearestMHz:nearest,transition,maximumMixing:maxMix,valid,line,groundF:F,message:valid?'':`${l.name}：最近允许跃迁 ${transition}，失谐 ${nearest.toPrecision(5)} MHz；${regime==='far'?'不满足远失谐条件 |Δ| ≥ 100 max(Γ/2π, Ω/2π)，请改为近共振光或调整频率/强度。':`超出近共振范围 ${nearScale.toPrecision(4)} MHz，请改为远失谐光或调整频率。`}`};
 });
}
export function assertLaserRegimes(scene:Scene):void{const invalid=classifyLasers(scene).find(c=>!c.valid);if(invalid)throw new Error(invalid.message);}
