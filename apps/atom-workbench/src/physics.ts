import type { Analysis, AxisAngles, ComplexAmplitude, FrequencyExpr, MagneticField, Laser, LaserFrequencyAnchor, Level, Polarization, Scene, SpectralLine, Vec3 } from './types.js';
import {laserIntensityMWcm2} from './laser-intensity.js';
import {accurateSum,sinCosDegrees,sinCosSum,vectorAxis} from './angles.js';
import {directionFrame,frameAxis,invert,referenceFrame,rotate,shortcutDirection,shortcutField,simplify,sphericalFrame,withoutTrailingZ,type Frame} from './direction.js';
export {laserIntensityMWcm2} from './laser-intensity.js';
export {accurateSum,direction,sinCosDegrees} from './angles.js';

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
const DEG = Math.PI / 180;
const scale = (a: Vec3, s: number): Vec3 => [a[0]*s,a[1]*s,a[2]*s];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const fieldSum=accurateSum;

type FieldGroup={frame:Frame;unit:Vec3;magnitudes:number[]};
/** Enabled fields with a magnitude; exactly collinear ones (equal or opposite unit vectors) share a group. */
function fieldGroups(fields:readonly MagneticField[]):FieldGroup[] {
  const groups:FieldGroup[]=[];
  for(const field of fields)if(field.enabled&&field.magnitudeG!==0){
    if(!Number.isFinite(field.magnitudeG))throw new Error('Magnetic field inputs must be finite.');
    const frame=withoutTrailingZ(directionFrame(fields,field.direction)),unit=rotate(frame,[0,0,1]);
    // Exact component equality only. In particular, no angular epsilon can
    // silently absorb a genuine transverse 1e-20-degree perturbation.
    const group=groups.find(g=>g.unit.every((x,i)=>x===unit[i])||g.unit.every((x,i)=>x===-unit[i]!));
    if(group){const sign=group.unit.every((x,i)=>x===unit[i])?1:-1;group.magnitudes.push(sign*field.magnitudeG);}
    else groups.push({frame,unit,magnitudes:[field.magnitudeG]});
  }
  return groups;
}
/** The frame a polarization is read in, about its axis: no trailing turn about z, and on a pole
 * no azimuth (there a turn about z only turns the frame about its own axis). */
function quantizationFrame(frame:Frame):Frame {
  const chain=withoutTrailingZ(frame);
  return chain.length===2&&chain[0]!.axis==='z'&&sinCosSum(chain[1]!.parts).sin===0?simplify(chain.slice(1)):chain;
}
/** The quantization axis and frame: one collinear group's own frame (so a light linked to the
 * field's axis is exactly along it), several groups' resultant, and at zero field the first
 * field's reference axis (+z without fields). */
function quantization(fields:readonly MagneticField[],groups:FieldGroup[],vector:Vec3):{axis:AxisAngles;frame:Frame} {
  let frame:Frame;
  if(Math.hypot(...vector)===0)frame=fields[0]?quantizationFrame(referenceFrame(fields,fields[0].direction.reference)):[];
  else{
    const active=groups.map(group=>({group,magnitude:fieldSum(group.magnitudes)})).filter(g=>g.magnitude!==0);
    if(active.length!==1){const axis=vectorAxis(vector);return {axis,frame:sphericalFrame(axis)};}
    const {group,magnitude}=active[0]!;
    frame=quantizationFrame(magnitude>0?group.frame:simplify([...group.frame,{axis:'y',parts:[180]}]));
  }
  return {axis:frameAxis(frame),frame};
}
/** One composition convention for the spectrum, Hamiltonian and result key.
 * Exactly collinear sources share their original frame, avoiding atan2 roundoff
 * that would otherwise open forbidden optical transitions. */
export function composeMagneticField(fields:readonly MagneticField[]):{fieldVector:Vec3;fieldAxis:AxisAngles;fieldMagnitudeG:number;frame:Frame} {
  const groups=fieldGroups(fields),contributions=groups.map(g=>scale(g.unit,fieldSum(g.magnitudes))),fieldVector=[0,1,2].map(i=>fieldSum(contributions.map(v=>v[i]!))) as Vec3;
  const {axis,frame}=quantization(fields,groups,fieldVector);
  return {fieldVector,fieldAxis:axis,fieldMagnitudeG:Math.hypot(...fieldVector),frame};
}
export type PolarizationPreset='linear'|'circular-minus'|'circular-plus'|'pi'|'sigma-minus'|'sigma-plus';
/** Shape presets keep the direction. Atomic presets link the light to a field's reference axis
 * (the 沿 B shortcuts, direction.ts): σ± along it with χ = ±45°, π perpendicular to it with E
 * along it. Being links, they follow that field's reference axis, not its own offset. */
export function setPolarizationPreset(laser:Laser,fields:readonly MagneticField[],preset:PolarizationPreset):void {
  if(preset==='linear'||preset==='circular-minus'||preset==='circular-plus'){
    laser.ellipticityDeg=preset==='linear'?0:preset==='circular-minus'?-45:45;return;
  }
  laser.direction=shortcutDirection(shortcutField(fields,laser.direction),preset==='pi'?'perpendicular':'parallel');
  laser.ellipticityDeg=preset==='pi'?0:preset==='sigma-plus'?45:-45;
  laser.polarizationAngleDeg=0;
}

type Quaternion={w:number;x:number;y:number;z:number};
/** A frame's quaternion. Its leading Ry(a) Rz(b) Ry(c) is written with the half-angle sum and
 * difference of c and a, so a light along B is exactly along it and a tiny tilt keeps its
 * size; the turns after it are multiplied on. */
function quaternion(frame:Frame):Quaternion {
  let i=0;const take=(axis:'y'|'z')=>frame[i]?.axis===axis?frame[i++]!.parts:[];
  const a=take('y'),b=take('z'),c=take('y');
  const dphi=sinCosSum(b,2),delta=sinCosSum([...c,...a],2),sum=sinCosSum([...c,...a.map(v=>-v)],2);
  let q={w:dphi.cos*delta.cos,x:-dphi.sin*sum.sin,y:dphi.cos*delta.sin,z:dphi.sin*sum.cos};
  for(;i<frame.length;i++){
    const turn=frame[i]!,h=sinCosSum(turn.parts,2),r=turn.axis==='y'?{w:h.cos,x:0,y:h.sin,z:0}:{w:h.cos,x:0,y:0,z:h.sin};
    q={w:q.w*r.w-q.x*r.x-q.y*r.y-q.z*r.z,x:q.w*r.x+q.x*r.w+q.y*r.z-q.z*r.y,y:q.w*r.y-q.x*r.z+q.y*r.w+q.z*r.x,z:q.w*r.z+q.x*r.y-q.y*r.x+q.z*r.w};
  }
  return q;
}
type Composition=ReturnType<typeof composeMagneticField>;
/** The light's frame read in the quantization frame, and the spherical amplitudes of its Jones vector there. */
function relativePolarization(laser:Laser,fields:readonly MagneticField[],field:Composition):{q:[ComplexAmplitude,ComplexAmplitude,ComplexAmplitude];angleDeg:number;frame:Frame} {
  const frame=directionFrame(fields,laser.direction),{w,x,y,z}=quaternion(simplify([...invert(field.frame),...frame]));
  // The spin-1 circular components are quadratic polynomials of the quaternion: no
  // 1-cos(delta) loss can erase a real, very small opposite-helicity coupling.
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
  return {q:[qm,q0,qp],angleDeg:2*Math.atan2(Math.hypot(x,y),Math.hypot(w,z))/DEG,frame};
}

/** The x, y of the light's frame are its transverse axes; ψ turns them about k to u, v. Positive
 * χ gives E=(u+i v)/sqrt(2) at χ=45° and drives Δm=+1 when k is parallel to B. The amplitudes
 * are on the quantization axis (composeMagneticField); their weights do not depend on the
 * quantization frame's azimuth about it. */
export function polarization(laser:Laser,fields:readonly MagneticField[],field:Composition=composeMagneticField(fields)):Polarization {
  const relative=relativePolarization(laser,fields,field),a=sinCosDegrees(laser.polarizationAngleDeg);
  const k=rotate(relative.frame,[0,0,1]),ex=rotate(relative.frame,[1,0,0]),ey=rotate(relative.frame,[0,1,0]);
  const u=add(scale(ex,a.cos),scale(ey,a.sin)),v=add(scale(ex,-a.sin),scale(ey,a.cos));
  const weight=relative.q.map(z=>clamp(z.re*z.re+z.im*z.im,0,1));
  return {minus:weight[0]!,pi:weight[1]!,plus:weight[2]!,amplitudes:relative.q,k,u,v,angleToBDeg:relative.angleDeg};
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
type FrequencyObject={id:string;name:string;frequency:FrequencyExpr;frequencyAnchor?:LaserFrequencyAnchor};
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
  for(const r of scene.referenceFrequencies)objects.set(`reference:${r.id}`,{object:r});
  for(const l of scene.lasers)objects.set(`laser:${l.id}`,{object:l,laser:l});
  const done=new Map<string,{line:SpectralLine;offsetMHz:number}>(),visiting=new Set<string>();
  const visit=(key:string):{line:SpectralLine;offsetMHz:number}=>{
    const saved=done.get(key);if(saved)return saved;
    const entry=objects.get(key);if(!entry)throw new Error('频率基准不存在。');
    const {object,laser}=entry;
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
  // An expression relative to another object is resolved with the scene (resolveSceneFrequencies).
  const f=laser.frequency,saved=laser.frequencyAnchor,viewKey=expressionKey(f),own='wavelengthNm' in f||f.base.kind==='absolute'||f.base.kind==='transition';
  const anchor=own?(saved?.viewKey===viewKey?saved:{...expressionAnchor(f),viewKey}):saved&&sameExpression(saved.viewKey,f)?saved:undefined;
  if(!anchor)throw new Error(`「${laser.name}」的频率跟随其它对象，尚未解析。`);
  if((anchor.line!=='D1'&&anchor.line!=='D2')||!Number.isFinite(anchor.offsetMHz)||!(RB87[anchor.line].frequencyHz/1e6+anchor.offsetMHz>0))throw new Error('Invalid canonical optical frequency.');
  return {...anchor};
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
export function couplingBetweenStates(firstId:string,secondId:string,laser:Laser,fields:readonly MagneticField[]):{
  allowed:boolean;q:number;dipoleCM:number;strength:number;rabiMHz:number;offsetMHz:number;detuningMHz:number;line:SpectralLine|null
} {
  const field=composeMagneticField(fields),dipole=dipoleBetweenStates(firstId,secondId,field.fieldMagnitudeG);
  const a=dipole.line?atomicSystem(dipole.line,field.fieldMagnitudeG):null;
  const g=a?.ground.find(s=>s.id===firstId||s.id===secondId),e=a?.excited.find(s=>s.id===firstId||s.id===secondId);
  const offsetMHz=g&&e&&dipole.line?(RB87[dipole.line].frequencyHz-RB87[laser.line].frequencyHz)/1e6+e.energyMHz-g.energyMHz-transitionReferenceMHz(laser.line):NaN;
  const detuningMHz=g&&e&&dipole.line?laserCarrierOffsetMHz(laser,dipole.line)-(e.energyMHz-g.energyMHz):NaN;
  const pol=polarization(laser,fields,field).amplitudes,p=dipole.q>=-1&&dipole.q<=1?pol[dipole.q+1]!:{re:0,im:0};
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
  const field=composeMagneticField(scene.fields),{fieldVector,fieldMagnitudeG,fieldAxis}=field;
  if(fieldMagnitudeG!==cachedField) {
    cachedSpectra={ground:spectrum('ground',fieldMagnitudeG),D1:spectrum('D1',fieldMagnitudeG),D2:spectrum('D2',fieldMagnitudeG)};
    cachedField=fieldMagnitudeG;
  }
  const {ground,D1,D2}=cachedSpectra;
  // The spectroscopy view follows the same initial state as the density evolution.
  const initialId=scene.dynamics.initialStateId;
  const initial=[...ground,...D1,...D2].find(state=>state.id===initialId);
  if(!initial)throw new Error(`Unknown initial state: ${initialId}`);
  const polarizations:Analysis['polarizations']={},transitions:Analysis['transitions']=[];
  for(const laser of scene.lasers) {
    const pol=polarization(laser,scene.fields,field);polarizations[laser.id]=pol;
    if(!laser.enabled) continue;
    const reference=transitionReferenceMHz(laser.line);
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
