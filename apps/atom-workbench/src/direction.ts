// Directions of magnetic fields and lasers: a reference axis and a relative
// offset (θ′, φ′) from it. The reference axis is given by angles, by a coordinate
// axis, or by a link to a magnetic field's reference axis (never to the field's
// own offset). The offset is read in the frame that the minimal rotation taking z
// to the reference axis carries: φ′ = 0 is the x it carries, and on the axis z the
// offset is the usual spherical (θ, φ).
//
// A direction's frame is a chain of rotations about y and z whose angles are kept
// as exact sums. Turns by exactly zero are the identity and adjacent turns about
// one axis add, so a light linked to a field's axis meets that field exactly: no
// tolerance decides that two directions coincide.
import type {AxisAngles,AxisName,Direction,MagneticField,ReferenceAxis,Vec3} from './types.js';
import {combinedAngle,exactlyZero,sinCosSum,vectorAxis,withResiduals} from './angles.js';

export const AXIS_NAMES:readonly AxisName[]=['+x','-x','+y','-y','+z','-z'];
export const AXES:Readonly<Record<AxisName,AxisAngles>>={
  '+x':{thetaDeg:90,phiDeg:0},'-x':{thetaDeg:90,phiDeg:180},'+y':{thetaDeg:90,phiDeg:90},
  '-y':{thetaDeg:90,phiDeg:-90},'+z':{thetaDeg:0,phiDeg:0},'-z':{thetaDeg:180,phiDeg:0},
};
/** How an axis is written: with the true minus sign. */
export const axisWords=(axis:AxisName):string=>axis.replace('-','−');

/** One rotation of a frame: about y or z by the exact sum of `parts` (degrees). */
export type Turn={axis:'y'|'z';parts:number[]};
/** A rotation as a chain of turns, applied right to left: R = T₁ T₂ … Tₙ. */
export type Frame=Turn[];

type FieldReference=Pick<MagneticField,'id'|'direction'>;
const parts=(value:number,residual?:number):number[]=>residual?[value,residual]:[value];
const negated=(values:readonly number[]):number[]=>values.map(v=>-v);

/** Drops turns by exactly zero and adds adjacent turns about one axis, until neither applies. */
export function simplify(turns:readonly Turn[]):Frame{
  let out=[...turns];
  for(let changed=true;changed;){
    changed=false;const next:Turn[]=[];
    for(const turn of out){
      if(exactlyZero(turn.parts)){changed=true;continue;}
      const last=next.at(-1);
      // The right-hand turn's angles first, as the relative frame of a light and B is written.
      if(last?.axis===turn.axis){next[next.length-1]={axis:turn.axis,parts:[...turn.parts,...last.parts]};changed=true;}
      else next.push(turn);
    }
    out=next;
  }
  return out;
}
export function invert(frame:Frame):Frame{return frame.slice().reverse().map(turn=>({axis:turn.axis,parts:negated(turn.parts)}));}
/** Without its trailing turns about z, which leave the frame's z axis where it is. */
export function withoutTrailingZ(frame:Frame):Frame{const out=frame.slice();while(out.at(-1)?.axis==='z')out.pop();return out;}
/** R v. */
export function rotate(frame:Frame,v:Vec3):Vec3{
  let [x,y,z]=v;
  for(let i=frame.length-1;i>=0;i--){
    const turn=frame[i]!,{sin,cos}=sinCosSum(turn.parts);
    if(turn.axis==='y')[x,z]=[x*cos+z*sin,-x*sin+z*cos];else [x,y]=[x*cos-y*sin,x*sin+y*cos];
  }
  return [x,y,z];
}

/** Polar angle into [0°, 180°] and azimuth into (−180°, 180°], from exact sums; φ = 0 on the poles. */
function sphericalAxis(thetaParts:readonly number[],phiParts:readonly number[]):AxisAngles{
  let theta=combinedAngle(thetaParts),phi=combinedAngle(phiParts);
  if(theta[0]<0||theta[0]>=360)theta=combinedAngle([...theta,-360*Math.floor(theta[0]/360)]);
  if(theta[0]>180){theta=combinedAngle([360,-theta[0],-theta[1]]);phi=combinedAngle([...phi,180]);}
  if(phi[0]>180||phi[0]<=-180)phi=combinedAngle([...phi,-360*Math.ceil((phi[0]-180)/360)]);
  if(sinCosSum(theta).sin===0)phi=[0,0];
  return withResiduals(theta,phi);
}
/** The direction R z of a frame as angles: exact for a spherical chain Rz(φ) Ry(θ), else from its vector. */
export function frameAxis(frame:Frame):AxisAngles{
  const chain=withoutTrailingZ(frame);
  if(!chain.length)return {thetaDeg:0,phiDeg:0};
  if(chain.length===1)return sphericalAxis(chain[0]!.parts,[]);
  if(chain.length===2&&chain[0]!.axis==='z')return sphericalAxis(chain[1]!.parts,chain[0]!.parts);
  return vectorAxis(rotate(chain,[0,0,1]));
}
/** The spherical frame Rz(φ) Ry(θ) of an axis: its x is e_θ, its y e_φ. */
export function sphericalFrame(axis:AxisAngles):Frame{
  return simplify([{axis:'z',parts:parts(axis.phiDeg,axis.phiResidualDeg)},{axis:'y',parts:parts(axis.thetaDeg,axis.thetaResidualDeg)}]);
}
/** The minimal rotation taking z to the axis: Rz(φ) Ry(θ) Rz(−φ). */
export function minimalFrame(axis:AxisAngles):Frame{
  const phi=parts(axis.phiDeg,axis.phiResidualDeg);
  return simplify([{axis:'z',parts:phi},{axis:'y',parts:parts(axis.thetaDeg,axis.thetaResidualDeg)},{axis:'z',parts:negated(phi)}]);
}

/** A reference axis as angles, its links followed; a missing field or a cycle is refused. */
export function referenceAxis(fields:readonly FieldReference[],reference:ReferenceAxis,seen:ReadonlySet<string>=new Set()):AxisAngles{
  if(reference.kind==='axis'){const axis=AXES[reference.axis];if(!axis)throw new Error('未知的坐标轴。');return {...axis};}
  if(reference.kind==='angles'){
    if(![reference.thetaDeg,reference.phiDeg,reference.thetaResidualDeg??0,reference.phiResidualDeg??0].every(Number.isFinite))throw new Error('参考轴的角度必须是有限数值。');
    return {thetaDeg:reference.thetaDeg,phiDeg:reference.phiDeg,...(reference.thetaResidualDeg?{thetaResidualDeg:reference.thetaResidualDeg}:{}),...(reference.phiResidualDeg?{phiResidualDeg:reference.phiResidualDeg}:{})};
  }
  if(seen.has(reference.id))throw new Error('磁场参考轴的链接成环。');
  const field=fields.find(f=>f.id===reference.id);if(!field)throw new Error('链接的磁场不存在。');
  return referenceAxis(fields,field.direction.reference,new Set([...seen,reference.id]));
}
/** The frame a direction's offset is read in: the minimal rotation to its reference axis. */
export function referenceFrame(fields:readonly FieldReference[],reference:ReferenceAxis):Frame{return minimalFrame(referenceAxis(fields,reference));}
/** A direction's own frame: its reference frame, then the offset Rz(φ′) Ry(θ′). Its z is the
 * actual direction; a laser's polarization angle ψ is measured from its x. */
export function directionFrame(fields:readonly FieldReference[],direction:Direction):Frame{
  if(![direction.thetaDeg,direction.phiDeg].every(Number.isFinite))throw new Error('方向的偏移角必须是有限数值。');
  return simplify([...referenceFrame(fields,direction.reference),{axis:'z',parts:[direction.phiDeg]},{axis:'y',parts:[direction.thetaDeg]}]);
}
export const directionAxis=(fields:readonly FieldReference[],direction:Direction):AxisAngles=>frameAxis(directionFrame(fields,direction));
export const directionVector=(fields:readonly FieldReference[],direction:Direction):Vec3=>rotate(withoutTrailingZ(directionFrame(fields,direction)),[0,0,1]);

/** A reference written as the coordinate axis it is exactly, else as angles. */
export function referenceOf(axis:AxisAngles):ReferenceAxis{
  const name=!axis.thetaResidualDeg&&!axis.phiResidualDeg?AXIS_NAMES.find(n=>AXES[n].thetaDeg===axis.thetaDeg&&AXES[n].phiDeg===axis.phiDeg):undefined;
  return name?{kind:'axis',axis:name}:{kind:'angles',...axis};
}
/** Where a field's reference axis link leads: the fields it can name without a cycle. */
export function linkableFields<T extends FieldReference>(fields:readonly T[],ownerId?:string):T[]{
  const reaches=(field:T,seen=new Set<string>()):boolean=>{const ref=field.direction.reference;if(ref.kind!=='field')return false;if(ref.id===ownerId)return true;if(seen.has(ref.id))return false;const next=fields.find(f=>f.id===ref.id);return !!next&&reaches(next,new Set([...seen,ref.id]));};
  return fields.filter(f=>f.id!==ownerId&&!reaches(f));
}
/** Everything whose reference axis links to the field. */
export function linkedTo<T extends {direction:Direction}>(objects:readonly T[],fieldId:string):T[]{return objects.filter(o=>o.direction.reference.kind==='field'&&o.direction.reference.id===fieldId);}
/** Before a field is deleted: what links to it keeps its reference axis, as angles. */
export function releaseLinks(fields:readonly FieldReference[],objects:readonly {direction:Direction}[],fieldId:string):void{
  for(const object of linkedTo(objects,fieldId))object.direction={...object.direction,reference:referenceOf(referenceAxis(fields,object.direction.reference))};
}

export type DirectionShortcut='parallel'|'anti'|'perpendicular';
/** The field the 沿 B shortcuts link to: the one the direction already links to, else the first
 * enabled field with a magnitude, else the first field. */
export function shortcutField<T extends FieldReference&Pick<MagneticField,'enabled'|'magnitudeG'>>(fields:readonly T[],current?:Direction):T|undefined{
  const linked=current?.reference.kind==='field'?fields.find(f=>f.id===(current.reference as {id:string}).id):undefined;
  return linked??fields.find(f=>f.enabled&&f.magnitudeG!==0)??fields[0];
}
/** Along, against or perpendicular to a field: a link to its reference axis, the offset placing
 * the light against the field's own offset (perpendicular in the field's meridian). */
export function shortcutDirection(field:FieldReference|undefined,mode:DirectionShortcut):Direction{
  const own=field?.direction??{reference:{kind:'axis',axis:'+z'},thetaDeg:0,phiDeg:0},theta=own.thetaDeg,phi=own.phiDeg;
  const reference:ReferenceAxis=field?{kind:'field',id:field.id}:own.reference;
  if(mode==='parallel')return {reference,thetaDeg:theta,phiDeg:phi};
  if(mode==='anti')return {reference,thetaDeg:180-theta,phiDeg:phi>0?phi-180:phi+180};
  return {reference,thetaDeg:theta<=90?theta+90:theta-90,phiDeg:phi};
}
export function sameReference(a:ReferenceAxis,b:ReferenceAxis):boolean{
  if(a.kind==='axis')return b.kind==='axis'&&a.axis===b.axis;
  if(a.kind==='field')return b.kind==='field'&&a.id===b.id;
  return b.kind==='angles'&&a.thetaDeg===b.thetaDeg&&a.phiDeg===b.phiDeg&&(a.thetaResidualDeg??0)===(b.thetaResidualDeg??0)&&(a.phiResidualDeg??0)===(b.phiResidualDeg??0);
}
export function sameDirection(a:Direction,b:Direction):boolean{return sameReference(a.reference,b.reference)&&a.thetaDeg===b.thetaDeg&&a.phiDeg===b.phiDeg;}
