import type {Scene} from './types.js';
import {atomicSystem,complexPolarization,composeMagneticField,laserCarrierOffsetMHz,laserIntensityMWcm2,RB87} from './physics.js';
/** Display metadata of a laser: its nearest allowed D-line transition and its
 * largest Rabi/detuning ratio. `regime` only chooses whether the levels chart
 * shows this laser's light shift ('far'); every enabled laser is an explicit
 * drive of the master equation whatever its detuning, so no regime is
 * invalid (`valid` is always true and `message` empty). */
export interface LaserClassification {id:string;regime:'near'|'far';nearestMHz:number;transition:string;maximumMixing:number;valid:boolean;message:string;line:'D1'|'D2';groundF:1|2}
/** Classification uses actual allowed dipoles and polarization, not a diagram origin. */
export function classifyLasers(scene:Scene):LaserClassification[]{
 const {fieldMagnitudeG,fieldVector,fieldAxis}=composeMagneticField(scene.fields);
 return scene.lasers.map(l=>{
  let nearest=Infinity,transition='',maxMix=0,line:'D1'|'D2'=l.line,F:1|2=2;
  const intensity=laserIntensityMWcm2(l),pol=complexPolarization(l,fieldVector,fieldAxis),electric=Math.sqrt(2*intensity*10/(8.8541878128e-12*299792458));
  for(const branch of ['D1','D2'] as const){const sys=atomicSystem(branch,fieldMagnitudeG);for(let e=0;e<sys.excited.length;e++)for(let g=0;g<8;g++){
   const a=sys.excited[e]!,b=sys.ground[g]!,q=a.m-b.m;if(q < -1||q>1)continue;
   const rabi=Math.abs(sys.dipoles[e*8+g]!)*RB87.D2.dipole*electric/6.62607015e-34/1e6*Math.hypot(pol[q+1]!.re,pol[q+1]!.im);
   if(rabi===0)continue;
   const delta=Math.abs(laserCarrierOffsetMHz(l,branch)-(a.energyMHz-b.energyMHz));
   if(delta<nearest){nearest=delta;transition=`${b.id} → ${a.id}`;line=branch;F=b.F as 1|2;}
   maxMix=Math.max(maxMix,rabi/Math.max(Number.MIN_VALUE,delta));
  }}
  return {id:l.id,regime:l.regime??'near',nearestMHz:nearest,transition,maximumMixing:maxMix,valid:true,message:'',line,groundF:F};
 });
}
