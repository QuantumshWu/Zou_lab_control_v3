import type { Analysis, AxisAngles, FrequencyExpr, MagneticField, Laser, LaserFrequencyAnchor, Level, Polarization, Scene, SpectralLine, Vec3 } from './types.js';
import {laserIntensityMWcm2} from './laser-intensity.js';
export {laserIntensityMWcm2} from './laser-intensity.js';

/** Data: Steck, Rubidium 87 D Line Data, revision 2.3.4 (2025-08-08), Tables 3–7.
 * https://steck.us/alkalidata/rubidium87numbers.pdf
 * Internal Hamiltonian is exact within separate fine-structure manifolds, magnetic dipole
 * + electric quadrupole HFS, and linear electronic/nuclear Zeeman operators. No diamagnetism,
 * coupling between fine-structure manifolds, optical dressing, or magnetic octupole term.
 */
export const RB87 = {
  I: 1.5, groundA: 3417.341305452145, muBOverH: 1.39962449361,
  gI: -0.0009951414, groundGJ: 2.002331070,
  D1: { J: 0.5, A: 407.25, B: 0, gJ: 0.666, frequencyHz: 377.107463380e12, gammaHz: 36.129e6, dipole: 2.5377e-29 },
  D2: { J: 1.5, A: 84.7185, B: 12.4965, gJ: 1.33410, frequencyHz: 384.2304844685e12, gammaHz: 38.117e6, dipole: 2.53444e-29 },
} as const;
const H = 6.62607015e-34, C = 299792458;
const EPS = 8.8541878128e-12;
const DEG = Math.PI / 180, TAU = 2 * Math.PI;
const dot = (a: Vec3, b: Vec3) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const scale = (a: Vec3, s: number): Vec3 => [a[0]*s,a[1]*s,a[2]*s];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

/** Evaluate one canonical octant so exact geometric symmetries use the same
 * floating-point argument. No near-angle tolerance or amplitude cutoff. */
export function sinCosDegrees(angle: number): { sin: number; cos: number } {
  const reduced=angle%360;let canonical=Math.abs(reduced),sinSign=reduced<0?-1:1,cosSign=1;
  if(canonical>180){canonical=360-canonical;sinSign=-sinSign;}
  if(canonical>90){canonical=180-canonical;cosSign=-1;}
  const swapped=canonical>45;if(swapped)canonical=90-canonical;
  const small=canonical===0?0:canonical===45?Math.SQRT1_2:Math.sin(canonical*DEG);
  const large=canonical===0?1:canonical===45?Math.SQRT1_2:Math.cos(canonical*DEG);
  const sin=sinSign*(swapped?large:small),cos=cosSign*(swapped?small:large);
  return {sin:sin===0?0:sin,cos:cos===0?0:cos};
}

function ellipticitySinCos(angle: number): { sin: number; cos: number } {
  if (angle === 45) return { sin: Math.SQRT1_2, cos: Math.SQRT1_2 };
  if (angle === -45) return { sin: -Math.SQRT1_2, cos: Math.SQRT1_2 };
  return sinCosDegrees(angle);
}

export function direction(thetaDeg: number, phiDeg: number): Vec3 {
  const t = sinCosDegrees(thetaDeg), p = sinCosDegrees(phiDeg);
  return [t.sin*p.cos, t.sin*p.sin, t.cos];
}

/** Error-free partial sums retain a small real field between cancelling large
 * inputs. This is numerical summation, never a near-zero field cutoff. */
function exactSumParts(values:number[]):number[] {
  const partials:number[]=[];
  for(let value of values){let count=0;for(let other of partials){if(Math.abs(value)<Math.abs(other))[value,other]=[other,value];const sum=value+other,error=other-(sum-value);if(error!==0)partials[count++]=error;value=sum;}partials.length=count;partials.push(value);}
  return partials;
}
/** Accumulate represented finite terms without losing a small residual between
 * cancelling large terms. Shared by magnetic and coherent optical sums. */
export function accurateSum(values:number[]):number {return exactSumParts(values).reduce((sum,value)=>sum+value,0);}
const fieldSum=accurateSum;
function angleParts(base:number,offset:number):[number,number] {const sum=base+offset,other=sum-base;return [sum,(base-(sum-other))+(offset-other)];}
/** atan2 expressed around the nearest cardinal direction. The low component
 * retains physical tilts smaller than one ULP of 90 or 180 degrees. */
function vectorAxis(vector:Vec3):AxisAngles {
  const [x,y,z]=vector,rho=Math.hypot(x,y);
  const t=rho<=Math.abs(z)?(z>=0?angleParts(0,Math.atan2(rho,z)/DEG):angleParts(180,-Math.atan2(rho,-z)/DEG)):angleParts(90,-Math.atan2(z,rho)/DEG);
  const p=Math.abs(x)>=Math.abs(y)?(x>=0?angleParts(0,Math.atan2(y,x)/DEG):angleParts(y<0?-180:180,-Math.atan2(y,-x)/DEG)):(y>=0?angleParts(90,-Math.atan2(x,y)/DEG):angleParts(-90,Math.atan2(x,-y)/DEG));
  return {thetaDeg:t[0],phiDeg:p[0],...(t[1]!==0?{thetaResidualDeg:t[1]}:{}),...(p[1]!==0?{phiResidualDeg:p[1]}:{})};
}
function sinCosSum(angles:number[],divisor=1):{sin:number;cos:number} {
  let sin=0,cos=1;
  for(const part of exactSumParts(angles)){const next=sinCosDegrees(part/divisor),s=sin*next.cos+cos*next.sin,c=cos*next.cos-sin*next.sin;sin=s;cos=c;}
  return {sin,cos};
}
export function laserDirectionViewKey(laser:Laser):string{return JSON.stringify([laser.thetaDeg,laser.phiDeg]);}
export function laserDirectionAxis(laser:Laser):AxisAngles {
  const anchor=laser.directionAnchor;
  if(anchor?.viewKey===laserDirectionViewKey(laser)){
    if(![anchor.thetaDeg,anchor.phiDeg,anchor.thetaResidualDeg??0,anchor.phiResidualDeg??0].every(Number.isFinite))throw new Error('Laser direction must be finite.');
    return {thetaDeg:anchor.thetaDeg,phiDeg:anchor.phiDeg,...(anchor.thetaResidualDeg?{thetaResidualDeg:anchor.thetaResidualDeg}:{}),...(anchor.phiResidualDeg?{phiResidualDeg:anchor.phiResidualDeg}:{})};
  }
  return {thetaDeg:laser.thetaDeg,phiDeg:laser.phiDeg};
}
/** This is a fixed lab direction snapshot, not a live constraint on B. */
export function setLaserDirectionAxis(laser:Laser,axis:AxisAngles,exact=false):void {
  laser.thetaDeg=axis.thetaDeg;laser.phiDeg=axis.phiDeg;
  if(exact)laser.directionAnchor={...axis,viewKey:laserDirectionViewKey(laser)};else delete laser.directionAnchor;
}
function combinedAngle(values:number[]):[number,number]{const parts=exactSumParts(values),high=parts.pop()??0;return angleParts(high,fieldSum(parts));}
export function relativeDirectionAxis(axis:AxisAngles,mode:'parallel'|'anti'|'perpendicular'):AxisAngles {
  if(mode==='parallel')return {...axis};
  const t=mode==='anti'?combinedAngle([180,-axis.thetaDeg,-(axis.thetaResidualDeg??0)]):[90,0];
  const p=combinedAngle([axis.phiDeg,axis.phiResidualDeg??0,mode==='anti'?180:90]);
  if(p[0]!>180)[p[0],p[1]]=combinedAngle([p[0]!,p[1]!,-360]);
  return {thetaDeg:t[0]!,phiDeg:p[0]!,...(t[1]!==0?{thetaResidualDeg:t[1]}:{}),...(p[1]!==0?{phiResidualDeg:p[1]}:{})};
}
type FieldGroup={axis:AxisAngles;unit:Vec3;magnitudes:number[]};
function fieldGroups(fields:MagneticField[]):FieldGroup[] {
  const groups:FieldGroup[]=[];
  for(const field of fields)if(field.enabled&&field.magnitudeG!==0){
    if(![field.magnitudeG,field.thetaDeg,field.phiDeg].every(Number.isFinite))throw new Error('Magnetic field inputs must be finite.');
    const unit=direction(field.thetaDeg,field.phiDeg);
    // Exact component equality only. In particular, no angular epsilon can
    // silently absorb a genuine transverse 1e-20-degree perturbation.
    const group=groups.find(g=>g.unit.every((x,i)=>x===unit[i])||g.unit.every((x,i)=>x===-unit[i]!));
    if(group){const sign=group.unit.every((x,i)=>x===unit[i])?1:-1;group.magnitudes.push(sign*field.magnitudeG);}
    else groups.push({axis:{thetaDeg:field.thetaDeg,phiDeg:field.phiDeg},unit,magnitudes:[field.magnitudeG]});
  }
  return groups;
}
function axisForGroups(groups:FieldGroup[],vector:Vec3):AxisAngles {
  if(Math.hypot(...vector)===0)return {thetaDeg:0,phiDeg:0};
  const active=groups.map(group=>({group,magnitude:fieldSum(group.magnitudes)})).filter(g=>g.magnitude!==0);
  if(active.length===1){const {group,magnitude}=active[0]!,axis=magnitude>0?{...group.axis}:{thetaDeg:180-group.axis.thetaDeg,phiDeg:group.axis.phiDeg>0?group.axis.phiDeg-180:group.axis.phiDeg+180};if(sinCosDegrees(axis.thetaDeg).sin===0)axis.phiDeg=0;return axis;}
  return vectorAxis(vector);
}
/** One composition convention for the spectrum, Hamiltonian and result key.
 * Exactly collinear sources share their original axis, avoiding atan2 roundoff
 * that would otherwise open forbidden optical transitions. */
export function composeMagneticField(fields:MagneticField[]):{fieldVector:Vec3;fieldAxis:AxisAngles;fieldMagnitudeG:number} {
  const groups=fieldGroups(fields),contributions=groups.map(g=>scale(g.unit,fieldSum(g.magnitudes))),fieldVector=[0,1,2].map(i=>fieldSum(contributions.map(v=>v[i]!))) as Vec3;
  return {fieldVector,fieldAxis:axisForGroups(groups,fieldVector),fieldMagnitudeG:Math.hypot(...fieldVector)};
}
export type PolarizationPreset='linear'|'circular-minus'|'circular-plus'|'pi'|'sigma-minus'|'sigma-plus';
/** Shape presets keep the lab propagation direction. Atomic presets configure
 * both k and E relative to the current quantization axis (z at B=0). They are
 * lab-frame snapshots: changing B afterwards must not silently rotate a beam. */
export function setPolarizationPreset(laser:Laser,fields:MagneticField[],preset:PolarizationPreset):void {
  if(preset==='linear'||preset==='circular-minus'||preset==='circular-plus'){
    laser.ellipticityDeg=preset==='linear'?0:preset==='circular-minus'?-45:45;return;
  }
  const axis=composeMagneticField(fields).fieldAxis;
  if(preset==='pi'){
    // Stay in B's meridional plane: E_theta is then exactly parallel to B.
    // Preserve the compensated 90-degree offset so rounding cannot create a
    // spurious sigma component and a false infinite-time pumping channel.
    const [thetaDeg,thetaResidualDeg]=combinedAngle([axis.thetaDeg,axis.thetaResidualDeg??0,axis.thetaDeg<=90?90:-90]);
    setLaserDirectionAxis(laser,{...axis,thetaDeg,thetaResidualDeg},true);
    laser.ellipticityDeg=0;
  }else{
    setLaserDirectionAxis(laser,axis,true);
    laser.ellipticityDeg=preset==='sigma-plus'?45:-45;
  }
  laser.polarizationAngleDeg=0;
}
export function quantizationAxis(fields:MagneticField[],vector:Vec3):AxisAngles {return axisForGroups(fieldGroups(fields),vector);}
function fieldAngles(vector:Vec3,axis?:AxisAngles):AxisAngles {
  if(Math.hypot(...vector)===0)return {thetaDeg:0,phiDeg:0};
  return axis??vectorAxis(vector);
}
function relativePolarization(laser:Laser,bVector:Vec3,axis?:AxisAngles):{q:[ComplexAmplitude,ComplexAmplitude,ComplexAmplitude];angleDeg:number} {
  const b=fieldAngles(bVector,axis),k=laserDirectionAxis(laser),dphi=sinCosSum([k.phiDeg,k.phiResidualDeg??0,-b.phiDeg,-(b.phiResidualDeg??0)],2);
  const delta=sinCosSum([k.thetaDeg,k.thetaResidualDeg??0,-b.thetaDeg,-(b.thetaResidualDeg??0)],2),sum=sinCosSum([k.thetaDeg,k.thetaResidualDeg??0,b.thetaDeg,b.thetaResidualDeg??0],2);
  // Quaternion for Ry(-theta_B) Rz(phi_k-phi_B) Ry(theta_k). Its spin-1
  // circular components are quadratic polynomials: no 1-cos(delta) loss can
  // erase a real, very small opposite-helicity coupling.
  const w=dphi.cos*delta.cos,x=-dphi.sin*sum.sin,y=dphi.cos*delta.sin,z=dphi.sin*sum.cos;
  const alpha=sinCosDegrees(laser.polarizationAngleDeg);
  const plus=sinCosDegrees(laser.ellipticityDeg-45).cos,minus=sinCosDegrees(laser.ellipticityDeg+45).cos;
  const ap={re:plus*alpha.cos,im:-plus*alpha.sin},am={re:minus*alpha.cos,im:minus*alpha.sin};
  const mul=(a:ComplexAmplitude,b:ComplexAmplitude):ComplexAmplitude=>({re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re});
  const combine=(a:ComplexAmplitude,b:ComplexAmplitude):ComplexAmplitude=>({re:a.re+b.re,im:a.im+b.im});
  const small={re:x*x-y*y,im:2*x*y},large={re:w*w-z*z,im:2*w*z};
  const pi={re:Math.SQRT2*(x*z-w*y),im:Math.SQRT2*(y*z+w*x)};
  const qm=combine(mul(ap,small),mul(am,large));
  const q0=combine(mul(ap,pi),mul(am,{re:pi.re,im:-pi.im}));
  const qp=combine(mul(ap,{re:-large.re,im:large.im}),mul(am,{re:-small.re,im:small.im}));
  return {q:[qm,q0,qp],angleDeg:2*Math.atan2(Math.hypot(x,y),Math.hypot(w,z))/DEG};
}

/** e_theta, e_phi form a right-handed transverse frame with k. At the north pole,
 * phi=0 gives u=x, v=y. alpha rotates this frame about k. Positive chi gives
 * E=(u+i v)/sqrt(2) at chi=45deg and drives delta-m=+1 when k is parallel to B.
 * Weights are invariant under the arbitrary azimuthal choice of the B-frame.
 */
export function polarization(laser: Laser, bVector: Vec3,axis?:AxisAngles): Polarization {
  const axisK=laserDirectionAxis(laser),t=sinCosSum([axisK.thetaDeg,axisK.thetaResidualDeg??0]),p=sinCosSum([axisK.phiDeg,axisK.phiResidualDeg??0]),a=sinCosDegrees(laser.polarizationAngleDeg);
  const k:Vec3=[t.sin*p.cos,t.sin*p.sin,t.cos];
  const et: Vec3 = [t.cos*p.cos, t.cos*p.sin, -t.sin];
  const ep: Vec3 = [-p.sin, p.cos, 0];
  const u = add(scale(et, a.cos), scale(ep, a.sin));
  const v = add(scale(et, -a.sin), scale(ep, a.cos));
  const relative=relativePolarization(laser,bVector,axis),weight=relative.q.map(z=>clamp(z.re*z.re+z.im*z.im,0,1));
  return { minus:weight[0]!, pi:weight[1]!, plus:weight[2]!, k, u, v, angleToBDeg:relative.angleDeg };
}

type Matrix = number[][];
export interface BasisState { mi: number; mj: number }
export interface Eigenstate extends Level { basis: BasisState[]; vector: number[] }
const matrix = (n: number): Matrix => Array.from({length:n},()=>Array(n).fill(0));

/** Real symmetric Jacobi eigensolver. Fixed-m blocks have dimension at most four. */
function eigen(matrixIn: Matrix): { value: number; vector: number[] }[] {
  const a = matrixIn.map(row=>row.slice()), n = a.length, v = matrix(n);
  for (let i=0;i<n;i++) v[i]![i]=1;
  for (let iteration=0;iteration<100;iteration++) {
    let p=0,q=0,max=0;
    for(let i=0;i<n;i++) for(let j=i+1;j<n;j++) if(Math.abs(a[i]![j]!)>max) {max=Math.abs(a[i]![j]!);p=i;q=j;}
    if(max<1e-11) break;
    const apq=a[p]![q]!, app=a[p]![p]!, aqq=a[q]![q]!;
    const theta=0.5*Math.atan2(2*apq,aqq-app), c=Math.cos(theta), s=Math.sin(theta);
    for(let k=0;k<n;k++) {
      if(k!==p && k!==q) {
        const akp=a[k]![p]!, akq=a[k]![q]!;
        a[k]![p]=a[p]![k]=c*akp-s*akq;
        a[k]![q]=a[q]![k]=s*akp+c*akq;
      }
      const vkp=v[k]![p]!, vkq=v[k]![q]!;
      v[k]![p]=c*vkp-s*vkq;v[k]![q]=s*vkp+c*vkq;
    }
    a[p]![p]=c*c*app-2*c*s*apq+s*s*aqq;
    a[q]![q]=s*s*app+2*c*s*apq+c*c*aqq;
    a[p]![q]=a[q]![p]=0;
  }
  return Array.from({length:n},(_,i)=>({value:a[i]![i]!,vector:v.map(row=>row[i]!)})).sort((a,b)=>a.value-b.value);
}

function hfsEnergy(J: number, A: number, B: number, F: number): number {
  const I=RB87.I, x=(F*(F+1)-I*(I+1)-J*(J+1))/2;
  return A*x + (J>0.5 ? B*(3*x*x+1.5*x-I*(I+1)*J*(J+1))/(2*I*(2*I-1)*J*(2*J-1)) : 0);
}

function spectrum(manifold: Level['manifold'], fieldG: number): Eigenstate[] {
  const {J,A,B,gJ} = manifold==='ground' ? {J:0.5,A:RB87.groundA,B:0,gJ:RB87.groundGJ} : RB87[manifold];
  const I=RB87.I, states: Eigenstate[]=[];
  for(let m=-(I+J);m<=I+J;m++) {
    const basis: BasisState[]=[];
    for(let mi=-I;mi<=I;mi++) for(let mj=-J;mj<=J;mj++) if(Math.abs(mi+mj-m)<1e-9) basis.push({mi,mj});
    const n=basis.length, x=matrix(n), h=matrix(n);
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) {
      const bi=basis[i]!, bj=basis[j]!;
      if(i===j) x[i]![j]=bi.mi*bi.mj;
      else if(bj.mi===bi.mi+1 && bj.mj===bi.mj-1) {
        x[i]![j]=x[j]![i]=0.5*Math.sqrt((I*(I+1)-bi.mi*(bi.mi+1))*(J*(J+1)-bi.mj*(bi.mj-1)));
      }
    }
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) {
      let q=0;
      if(J>0.5) {
        let xx=0;for(let k=0;k<n;k++) xx+=x[i]![k]!*x[k]![j]!;
        q=B*(3*xx+1.5*x[i]![j]!-(i===j ? I*(I+1)*J*(J+1):0))/(2*I*(2*I-1)*J*(2*J-1));
      }
      h[i]![j]=A*x[i]![j]!+q+(i===j?RB87.muBOverH*fieldG*(gJ*basis[i]!.mj+RB87.gI*basis[i]!.mi):0);
    }
    const eig=eigen(h), minF=Math.max(Math.abs(I-J),Math.abs(m));
    eig.forEach((e,index)=>{
      const F=minF+index;
      states.push({id:`${manifold}-${F}-${m}`,manifold,F,m,energyMHz:e.value,
        shiftMHz:e.value-hfsEnergy(J,A,B,F),basis,vector:e.vector});
    });
  }
  return states.sort((a,b)=>a.F-b.F || a.m-b.m);
}

function factorial(n: number): number { if(n<0 || Math.abs(n-Math.round(n))>1e-8)return Infinity;let r=1;for(let i=2;i<=Math.round(n);i++)r*=i;return r; }
/** Racah formula; all factorial arguments are integers for valid angular momenta. */
export function wigner3j(j1:number,j2:number,j3:number,m1:number,m2:number,m3:number):number {
  if(Math.abs(m1+m2+m3)>1e-8 || Math.abs(m1)>j1 || Math.abs(m2)>j2 || Math.abs(m3)>j3 || j3>j1+j2 || j3<Math.abs(j1-j2)) return 0;
  const tri=factorial(j1+j2-j3)*factorial(j1-j2+j3)*factorial(-j1+j2+j3)/factorial(j1+j2+j3+1);
  const pref=(-1)**Math.round(j1-j2-m3)*Math.sqrt(tri*factorial(j1+m1)*factorial(j1-m1)*factorial(j2+m2)*factorial(j2-m2)*factorial(j3+m3)*factorial(j3-m3));
  let sum=0;
  const lo=Math.max(0,j2-j3-m1,j1-j3+m2), hi=Math.min(j1+j2-j3,j1-m1,j2+m2);
  for(let z=Math.ceil(lo);z<=Math.floor(hi);z++) sum+=(-1)**z/(factorial(z)*factorial(j1+j2-j3-z)*factorial(j1-m1-z)*factorial(j2+m2-z)*factorial(j3-j2+m1+z)*factorial(j3-j1-m2+z));
  return pref*sum;
}
function clebsch(j1:number,m1:number,j2:number,m2:number,j3:number,m3:number):number {
  return (-1)**Math.round(j1-j2+m3)*Math.sqrt(2*j3+1)*wigner3j(j1,j2,j3,m1,m2,-m3);
}
/** Signed dipole amplitude in units of the D2 stretched-state dipole. */
export function transitionDipole(g:Eigenstate,e:Eigenstate,line:SpectralLine,q:number):number {
  let amplitude=0;
  g.basis.forEach((gb,i)=>e.basis.forEach((eb,j)=>{
    if(gb.mi===eb.mi && Math.abs(eb.mj-gb.mj-q)<1e-9)
      amplitude+=g.vector[i]!*e.vector[j]!*clebsch(0.5,gb.mj,1,q,RB87[line].J,eb.mj);
  }));
  return amplitude*(RB87[line].dipole/RB87.D2.dipole);
}
function transitionStrength(g:Eigenstate,e:Eigenstate,line:SpectralLine,q:number):number {
  return transitionDipole(g,e,line,q)**2;
}

export function transitionReferenceMHz(line:SpectralLine,Fg:1|2=2,Fe:number=line==='D1'?2:3,fromManifold:'ground'|'D1'='ground'):number {
  const d=RB87[line];
  if ((Fg!==1&&Fg!==2) || !Number.isInteger(Fe) || Fe < (line==='D1'?1:0) || Fe> (line==='D1'?2:3))
    throw new Error('Invalid zero-field hyperfine reference.');
  if(fromManifold==='D1'&&line!=='D2')throw new Error('Energy reference must connect different electronic manifolds.');
  return hfsEnergy(d.J,d.A,d.B,Fe)-(fromManifold==='D1'?RB87.D1.frequencyHz/1e6+hfsEnergy(.5,RB87.D1.A,0,Fg):hfsEnergy(0.5,RB87.groundA,0,Fg));
}
export function zeroFieldTransitionFrequencyMHz(line:SpectralLine,Fg:1|2,Fe:number,fromManifold:'ground'|'D1'='ground'):number {
  return RB87[line].frequencyHz/1e6+transitionReferenceMHz(line,Fg,Fe,fromManifold);
}
export function absoluteEnergyMHz(level:Level):number {
  return level.energyMHz+(level.manifold==='ground'?0:RB87[level.manifold].frequencyHz/1e6);
}
/** Only physical input fields are fingerprinted, not units, power or geometry. */
export function laserFrequencyViewKey(laser:Laser):string {
  if(laser.frequency)return JSON.stringify(['expr',laser.frequency]);
  const reference=laser.referenceTransition;
  return JSON.stringify([laser.frequencyMode??'transition',laser.line,reference?.line??laser.line,
    reference?.Fg??2,reference?.Fe??(laser.line==='D1'?2:3),
    laser.frequencyMode==='wavelength'?laser.wavelengthNm:laser.detuningMHz,...(reference?.fromManifold==='D1'?['D1']:[])]);
}

/** A frequency expression's carrier: a D-line centroid and the offset from it. A base
 * frequency is needed for an expression relative to a laser or a reference frequency. */
export function expressionAnchor(expr:FrequencyExpr,base?:{line:SpectralLine;offsetMHz:number}):{line:SpectralLine;offsetMHz:number}{
  if('wavelengthNm' in expr){
    if(!Number.isFinite(expr.wavelengthNm)||!(expr.wavelengthNm>0))throw new Error('Vacuum wavelength must be positive and finite.');
    return {line:'D2',offsetMHz:C*1000/expr.wavelengthNm-RB87.D2.frequencyHz/1e6};
  }
  const b=expr.base,offset=expr.offsetMHz;if(!Number.isFinite(offset))throw new Error('频率偏移必须是有限数值。');
  if(b.kind==='absolute'){const line:SpectralLine=Math.abs(offset-RB87.D1.frequencyHz/1e6)<Math.abs(offset-RB87.D2.frequencyHz/1e6)?'D1':'D2';return {line,offsetMHz:offset-RB87[line].frequencyHz/1e6};}
  if(b.kind==='transition')return {line:b.line,offsetMHz:transitionReferenceMHz(b.line,b.Fg,b.Fe,b.fromManifold)+offset};
  if(!base)throw new Error('频率基准尚未解析。');
  return {line:base.line,offsetMHz:base.offsetMHz+offset};
}
type FrequencyObject={id:string;name:string;frequency?:FrequencyExpr;frequencyAnchor?:LaserFrequencyAnchor};
/** An expression's fingerprint, with its base's carrier when it follows another object. A
 * saved carrier of the same fingerprint is kept bit for bit: re-expressing a frequency
 * against another base never re-rounds it. */
export function expressionKey(expr:FrequencyExpr,base?:{line:SpectralLine;offsetMHz:number}):string{return JSON.stringify(base?['expr',expr,[base.line,base.offsetMHz]]:['expr',expr]);}
const sameExpression=(viewKey:string,expr:FrequencyExpr):boolean=>{try{const key=JSON.parse(viewKey);return key[0]==='expr'&&JSON.stringify(key[1])===JSON.stringify(expr);}catch{return false;}};
/** Resolves every defined frequency of the scene in dependency order (reference frequencies
 * and lasers may follow one another): each object's derived carrier, and a laser's D line.
 * A base that does not exist, a cycle, or a frequency that is not positive is refused. */
export function resolveSceneFrequencies(scene:Pick<Scene,'lasers'|'referenceFrequencies'>):void{
  const objects=new Map<string,{object:FrequencyObject;laser?:Laser}>();
  for(const r of scene.referenceFrequencies??[])objects.set(`reference:${r.id}`,{object:r});
  for(const l of scene.lasers)objects.set(`laser:${l.id}`,{object:l,laser:l});
  const done=new Map<string,{line:SpectralLine;offsetMHz:number}>(),visiting=new Set<string>();
  const visit=(key:string):{line:SpectralLine;offsetMHz:number}=>{
    const saved=done.get(key);if(saved)return saved;
    const entry=objects.get(key);if(!entry)throw new Error('频率基准不存在。');
    const {object,laser}=entry;
    if(!object.frequency){const anchor=captureLaserFrequencyAnchor(laser!);done.set(key,anchor);return anchor;}
    if(visiting.has(key))throw new Error(`「${object.name}」的频率定义成环。`);
    visiting.add(key);
    const b='base' in object.frequency?object.frequency.base:undefined,dependsOn=b&&(b.kind==='laser'||b.kind==='reference')?`${b.kind}:${b.id}`:undefined;
    if(dependsOn&&!objects.has(dependsOn))throw new Error(`「${object.name}」的频率基准不存在。`);
    const base=dependsOn?visit(dependsOn):undefined,viewKey=expressionKey(object.frequency,base);
    const anchor=object.frequencyAnchor?.viewKey===viewKey?object.frequencyAnchor:expressionAnchor(object.frequency,base);
    visiting.delete(key);
    if(!Number.isFinite(anchor.offsetMHz)||!(RB87[anchor.line].frequencyHz/1e6+anchor.offsetMHz>0))throw new Error(`「${object.name}」的光频必须是有限正数。`);
    object.frequencyAnchor={line:anchor.line,offsetMHz:anchor.offsetMHz,viewKey};if(laser)laser.line=anchor.line;
    done.set(key,anchor);return anchor;
  };
  for(const key of objects.keys())visit(key);
}
export function captureLaserFrequencyAnchor(laser:Laser):LaserFrequencyAnchor {
  const viewKey=laserFrequencyViewKey(laser),saved=laser.frequencyAnchor;
  if(laser.frequency){
    // An expression relative to another object is resolved with the scene (resolveSceneFrequencies).
    const f=laser.frequency,own='wavelengthNm' in f||f.base.kind==='absolute'||f.base.kind==='transition';
    const anchor=own?(saved?.viewKey===viewKey?saved:{...expressionAnchor(f),viewKey}):saved&&sameExpression(saved.viewKey,f)?saved:undefined;
    if(!anchor)throw new Error(`「${laser.name}」的频率跟随其它对象，尚未解析。`);
    if((anchor.line!=='D1'&&anchor.line!=='D2')||!Number.isFinite(anchor.offsetMHz)||!(RB87[anchor.line].frequencyHz/1e6+anchor.offsetMHz>0))throw new Error('Invalid canonical optical frequency.');
    return {...anchor};
  }
  if(saved?.viewKey===viewKey) {
    if((saved.line!=='D1'&&saved.line!=='D2')||!Number.isFinite(saved.offsetMHz)
      ||!(RB87[saved.line].frequencyHz/1e6+saved.offsetMHz>0))throw new Error('Invalid canonical optical frequency.');
    return {...saved};
  }
  if(laser.frequencyMode==='wavelength') {
    if(!Number.isFinite(laser.wavelengthNm)||!(laser.wavelengthNm!>0))throw new Error('Vacuum wavelength must be positive and finite.');
    return {line:'D2',offsetMHz:C*1000/laser.wavelengthNm!-RB87.D2.frequencyHz/1e6,viewKey};
  }
  const reference=laser.referenceTransition;
  if(reference&&reference.line!==laser.line)throw new Error('The selected transition reference and laser D line disagree.');
  return {line:laser.line,offsetMHz:transitionReferenceMHz(laser.line,reference?.Fg,reference?.Fe,reference?.fromManifold)+(laser.detuningMHz??0),viewKey};
}

export function laserFrequencyMHz(laser:Laser):number {
  const coordinate=captureLaserFrequencyAnchor(laser);
  return RB87[coordinate.line].frequencyHz/1e6+coordinate.offsetMHz;
}

/** Optical frequency relative to a fine-structure centre of gravity. Keeping
 * same-line arithmetic local preserves sub-Hz detunings before adding the
 * ~380 THz absolute optical frequency. */
export function laserCarrierOffsetMHz(laser:Laser,line:SpectralLine=laser.line):number {
  const coordinate=captureLaserFrequencyAnchor(laser);
  return coordinate.offsetMHz+(line===coordinate.line?0:(RB87[coordinate.line].frequencyHz-RB87[line].frequencyHz)/1e6);
}

/** Beat frequency (first minus second), subtracting small offsets before the
 * fine-structure centroids. In particular, a D1 sub-Hz beat must not first be
 * rounded by converting both frequencies to offsets ~7 THz from D2. */
export function laserFrequencyDifferenceMHz(first:Laser,second:Laser):number {
  const a=captureLaserFrequencyAnchor(first),b=captureLaserFrequencyAnchor(second);
  return (a.offsetMHz-b.offsetMHz)+(a.line===b.line?0:(RB87[a.line].frequencyHz-RB87[b.line].frequencyHz)/1e6);
}

export interface ComplexAmplitude {re:number;im:number}
/** Complex absorption amplitudes q=-1,0,+1 in a deterministic B-aligned frame.
 * Their squared moduli equal polarization().minus/pi/plus. The q=+1
 * convention is consistent with the documented positive helicity convention.
 */
export function complexPolarization(laser:Laser,bVector:Vec3,axis?:AxisAngles):[ComplexAmplitude,ComplexAmplitude,ComplexAmplitude] {
  return relativePolarization(laser,bVector,axis).q;
}

let systemCacheKey='';
let systemCache:{ground:Eigenstate[];excited:Eigenstate[];dipoles:Float64Array;line:SpectralLine;fieldG:number};
/** Dipoles are row-major excited × ground, including their sign; each pair has
 * at most one q in the field eigenbasis. All fine-structure states are retained. */
export function atomicSystem(line:SpectralLine,fieldG:number):typeof systemCache {
  const key=`${line}:${fieldG}`;
  if(key===systemCacheKey)return systemCache;
  const ground=spectrum('ground',fieldG),excited=spectrum(line,fieldG),dipoles=new Float64Array(ground.length*excited.length);
  for(let e=0;e<excited.length;e++)for(let g=0;g<ground.length;g++) {
    const q=excited[e].m-ground[g].m;
    // Exact zero-field angular-momentum identities. Preserve them algebraically
    // rather than allowing eigenvector-roundoff to open a false decay/drive
    // channel in long-time dynamics. Nonzero B may genuinely mix these states.
    if(fieldG===0 && (Math.abs(excited[e].F-ground[g].F)>1 ||
      (q===0 && ground[g].m===0 && excited[e].F===ground[g].F))) continue;
    if(q===-1||q===0||q===1)dipoles[e*ground.length+g]=transitionDipole(ground[g],excited[e],line,q);
  }
  systemCacheKey=key;return systemCache={ground,excited,dipoles,line,fieldG};
}

export function dipoleBetweenStates(firstId:string,secondId:string,fieldMagnitudeG=0):{allowed:boolean;q:number;amplitudeCM:number;strength:number;line:SpectralLine|null} {
  const line:SpectralLine|null=firstId.startsWith('D1-')||secondId.startsWith('D1-')?'D1':firstId.startsWith('D2-')||secondId.startsWith('D2-')?'D2':null;
  if(!line)return {allowed:false,q:0,amplitudeCM:0,strength:0,line:null};
  const a=atomicSystem(line,fieldMagnitudeG),g=a.ground.find(s=>s.id===firstId||s.id===secondId),e=a.excited.find(s=>s.id===firstId||s.id===secondId);
  if(!g||!e)return {allowed:false,q:0,amplitudeCM:0,strength:0,line};
  const q=e.m-g.m,d=q===-1||q===0||q===1?a.dipoles[a.excited.indexOf(e)*a.ground.length+a.ground.indexOf(g)]:0;
  return {allowed:d!==0,q,amplitudeCM:d*RB87.D2.dipole,strength:d*d,line};
}

/** Pair readout independent of the scene's initial-state selector. This is the
 * isolated-beam Rabi magnitude; coherent sums of several beams are in dynamics. */
export function couplingBetweenStates(firstId:string,secondId:string,laser:Laser,fieldVector:Vec3,axis?:AxisAngles):{
  allowed:boolean;q:number;dipoleCM:number;strength:number;rabiMHz:number;offsetMHz:number;detuningMHz:number;line:SpectralLine|null
} {
  const dipole=dipoleBetweenStates(firstId,secondId,Math.hypot(...fieldVector));
  const a=dipole.line?atomicSystem(dipole.line,Math.hypot(...fieldVector)):null;
  const g=a?.ground.find(s=>s.id===firstId||s.id===secondId),e=a?.excited.find(s=>s.id===firstId||s.id===secondId);
  const reference=laser.referenceTransition;
  const offsetMHz=g&&e&&dipole.line?(RB87[dipole.line].frequencyHz-RB87[laser.line].frequencyHz)/1e6+e.energyMHz-g.energyMHz-transitionReferenceMHz(laser.line,reference?.Fg,reference?.Fe,reference?.fromManifold):NaN;
  const detuningMHz=g&&e&&dipole.line?laserCarrierOffsetMHz(laser,dipole.line)-(e.energyMHz-g.energyMHz):NaN;
  const pol=complexPolarization(laser,fieldVector,axis),p=dipole.q>=-1&&dipole.q<=1?pol[dipole.q+1]:{re:0,im:0};
  const rabiMHz=Math.abs(dipole.amplitudeCM)*Math.sqrt(2*Math.max(0,laserIntensityMWcm2(laser))*10/(EPS*C))*Math.hypot(p.re,p.im)/H/1e6;
  return {allowed:dipole.allowed,q:dipole.q,dipoleCM:dipole.amplitudeCM,strength:dipole.strength,rabiMHz,offsetMHz,detuningMHz,line:dipole.line};
}

/** Exact m=0 Breit–Rabi frequency and derivative within the ground-manifold model. */
export function clockFrequencyMHz(fieldG:number): number {
  const delta=2*RB87.groundA, slope=RB87.muBOverH*(RB87.groundGJ-RB87.gI);
  return Math.hypot(delta,slope*fieldG);
}
export function clockSensitivityHzPerMG(fieldG:number):number {
  const slope=RB87.muBOverH*(RB87.groundGJ-RB87.gI);
  return slope*slope*fieldG/clockFrequencyMHz(fieldG)*1000;
}

let cachedField=NaN;
let cachedSpectra:{ground:Eigenstate[];D1:Eigenstate[];D2:Eigenstate[]};

export function analyze(scene:Scene):Analysis {
  resolveSceneFrequencies(scene);
  const start=performance.now();
  const {fieldVector,fieldMagnitudeG,fieldAxis}=composeMagneticField(scene.fields);
  if(fieldMagnitudeG!==cachedField) {
    cachedSpectra={ground:spectrum('ground',fieldMagnitudeG),D1:spectrum('D1',fieldMagnitudeG),D2:spectrum('D2',fieldMagnitudeG)};
    cachedField=fieldMagnitudeG;
  }
  const {ground,D1,D2}=cachedSpectra;
  // The spectroscopy view follows the same initial state as the density
  // evolution. Legacy F/m fields are only a fallback for older scene files.
  const initialId=scene.dynamics?.initialStateId??`ground-${scene.initialF}-${scene.initialM}`;
  const initial=[...ground,...D1,...D2].find(state=>state.id===initialId);
  if(!initial)throw new Error(`Unknown initial state: ${initialId}`);
  const polarizations:Analysis['polarizations']={},transitions:Analysis['transitions']=[];
  for(const laser of scene.lasers) {
    const pol=polarization(laser,fieldVector,fieldAxis);polarizations[laser.id]=pol;
    if(!laser.enabled) continue;
    const ref=laser.referenceTransition;
    const reference=transitionReferenceMHz(laser.line,ref?.Fg,ref?.Fe,ref?.fromManifold);
    const intensity=laserIntensityMWcm2(laser);
    // The selected reference is a frequency coordinate, never a selection of
    // which physical dipole manifold the electromagnetic field acts on.
    for(const line of ['D1','D2'] as const) {
      if(initial.manifold!=='ground'&&initial.manifold!==line)continue;
      const levels=initial.manifold==='ground'?(line==='D1'?D1:D2):[initial];
      const grounds=initial.manifold==='ground'?[initial]:ground;
      const carrier=laserCarrierOffsetMHz(laser,line);
      const referenceOffset=(RB87[line].frequencyHz-RB87[laser.line].frequencyHz)/1e6-reference;
      for(const g of grounds)for(const e of levels) {
        const q=e.m-g.m;
        if(q!==-1 && q!==0 && q!==1) continue;
        if(fieldMagnitudeG===0&&(Math.abs(e.F-g.F)>1||(q===0&&g.m===0&&e.F===g.F)))continue;
        const strength=transitionStrength(g,e,line,q);
        if(strength===0) continue;
        const weight=q===-1?pol.minus:q===0?pol.pi:pol.plus;
        const offsetMHz=e.energyMHz-g.energyMHz+referenceOffset,detuningMHz=carrier-(e.energyMHz-g.energyMHz);
        const rabiMHz=RB87.D2.dipole*Math.sqrt(2*Math.max(0,intensity)*10/(EPS*C)*strength*weight)/H/1e6;
        transitions.push({id:`${laser.id}:${g.id}:${e.id}`,laserId:laser.id,line,groundId:g.id,excitedId:e.id,
          Fg:g.F,mg:g.m,Fe:e.F,me:e.m,q,offsetMHz,detuningMHz,strength,weight,rabiMHz});
      }
    }
  }
  const clockMHz=clockFrequencyMHz(fieldMagnitudeG);
  // Strip internal eigenvectors from the UI-facing result.
  const publicLevel=({basis:_basis,vector:_vector,...level}:Eigenstate):Level=>level;
  return {fieldVector,fieldMagnitudeG,fieldAxis,ground:ground.map(publicLevel),D1:D1.map(publicLevel),D2:D2.map(publicLevel),polarizations,transitions,
    clockMHz,clockShiftKHz:(clockMHz-2*RB87.groundA)*1000,clockSensitivityHzPerMG:clockSensitivityHzPerMG(fieldMagnitudeG),
    computeMs:performance.now()-start};
}
