// Exact angle arithmetic shared by the direction model (direction.ts) and the
// physics (physics.ts): sines and cosines of degrees on one canonical octant,
// error-free sums, and angles carried as a high part and a low residual so a
// physical tilt smaller than one ULP of 90° or 180° is never rounded away.
import type {AxisAngles,Vec3} from './types.js';

export const DEG=Math.PI/180;

/** Evaluate one canonical octant so exact geometric symmetries use the same
 * floating-point argument. No near-angle tolerance or amplitude cutoff. */
export function sinCosDegrees(angle:number):{sin:number;cos:number}{
  const reduced=angle%360;let canonical=Math.abs(reduced),sinSign=reduced<0?-1:1,cosSign=1;
  if(canonical>180){canonical=360-canonical;sinSign=-sinSign;}
  if(canonical>90){canonical=180-canonical;cosSign=-1;}
  const swapped=canonical>45;if(swapped)canonical=90-canonical;
  const small=canonical===0?0:canonical===45?Math.SQRT1_2:Math.sin(canonical*DEG);
  const large=canonical===0?1:canonical===45?Math.SQRT1_2:Math.cos(canonical*DEG);
  const sin=sinSign*(swapped?large:small),cos=cosSign*(swapped?small:large);
  return {sin:sin===0?0:sin,cos:cos===0?0:cos};
}

/** The unit vector at polar angle θ and azimuth φ (degrees). */
export function direction(thetaDeg:number,phiDeg:number):Vec3{
  const t=sinCosDegrees(thetaDeg),p=sinCosDegrees(phiDeg);
  return [t.sin*p.cos,t.sin*p.sin,t.cos];
}

/** Error-free partial sums retain a small real value between cancelling large
 * inputs. This is numerical summation, never a near-zero cutoff. */
export function exactSumParts(values:readonly number[]):number[]{
  const partials:number[]=[];
  for(let value of values){let count=0;for(let other of partials){if(Math.abs(value)<Math.abs(other))[value,other]=[other,value];const sum=value+other,error=other-(sum-value);if(error!==0)partials[count++]=error;value=sum;}partials.length=count;partials.push(value);}
  return partials;
}
/** Accumulate represented finite terms without losing a small residual between
 * cancelling large terms. Shared by magnetic and coherent optical sums. */
export function accurateSum(values:readonly number[]):number{return exactSumParts(values).reduce((sum,value)=>sum+value,0);}
/** True when the exact sum of `values` is zero. */
export function exactlyZero(values:readonly number[]):boolean{return exactSumParts(values).every(value=>value===0);}

/** base + offset as a rounded high part and its exact low residual. */
export function angleParts(base:number,offset:number):[number,number]{const sum=base+offset,other=sum-base;return [sum,(base-(sum-other))+(offset-other)];}
/** The exact sum of `values` as a high part and its residual. */
export function combinedAngle(values:readonly number[]):[number,number]{const parts=exactSumParts(values),high=parts.pop()??0;return angleParts(high,accurateSum(parts));}

/** atan2 expressed around the nearest cardinal direction. The low component
 * retains physical tilts smaller than one ULP of 90 or 180 degrees. */
export function vectorAxis(vector:Vec3):AxisAngles{
  const [x,y,z]=vector,rho=Math.hypot(x,y);
  const t=rho<=Math.abs(z)?(z>=0?angleParts(0,Math.atan2(rho,z)/DEG):angleParts(180,-Math.atan2(rho,-z)/DEG)):angleParts(90,-Math.atan2(z,rho)/DEG);
  const p=Math.abs(x)>=Math.abs(y)?(x>=0?angleParts(0,Math.atan2(y,x)/DEG):angleParts(y<0?-180:180,-Math.atan2(y,-x)/DEG)):(y>=0?angleParts(90,-Math.atan2(x,y)/DEG):angleParts(-90,Math.atan2(x,-y)/DEG));
  return withResiduals(t,p);
}
/** Angles from [high, residual] pairs; a zero residual is not written. */
export function withResiduals(theta:[number,number],phi:[number,number]):AxisAngles{
  return {thetaDeg:theta[0],phiDeg:phi[0],...(theta[1]!==0?{thetaResidualDeg:theta[1]}:{}),...(phi[1]!==0?{phiResidualDeg:phi[1]}:{})};
}

/** sin and cos of (Σ angles)/divisor, composed over the exact partials. */
export function sinCosSum(angles:readonly number[],divisor=1):{sin:number;cos:number}{
  let sin=0,cos=1;
  for(const part of exactSumParts(angles)){const next=sinCosDegrees(part/divisor),s=sin*next.cos+cos*next.sin,c=cos*next.cos-sin*next.sin;sin=s;cos=c;}
  return {sin,cos};
}
