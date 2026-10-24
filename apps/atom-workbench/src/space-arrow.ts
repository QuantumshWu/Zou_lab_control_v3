import type {Vec3} from './types.js';

export type ArrowFacet={vertices:Vec3[];normal:Vec3};
const add=(a:Vec3,b:Vec3):Vec3=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const mul=(a:Vec3,s:number):Vec3=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a:Vec3,b:Vec3)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a:Vec3,b:Vec3):Vec3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=(a:Vec3):Vec3=>mul(a,1/Math.hypot(...a));

/** A closed cylinder/cone arrow in world coordinates. All radii shrink with
 * the vector near zero; camera foreshortening never switches its geometry.
 * A transverse hint keeps the tessellation stable as a polarization rotates. */
export function arrowGeometry(from:Vec3,to:Vec3,desiredRadius:number,transverse?:Vec3):ArrowFacet[]{
  const delta=add(to,mul(from,-1)),length=Math.hypot(...delta);
  if(length<1e-12||desiredRadius<=0)return [];
  const axis=mul(delta,1/length);
  const hint=transverse??(Math.abs(axis[2])<.9?[0,0,1]:[0,1,0]);
  let radial=add(hint,mul(axis,-dot(hint,axis)));
  if(Math.hypot(...radial)<1e-8){const fallback:Vec3=Math.abs(axis[2])<.9?[0,0,1]:[0,1,0];radial=add(fallback,mul(axis,-dot(fallback,axis)));}
  const u=unit(radial),v=cross(axis,u),radius=Math.min(desiredRadius,length*.045);
  const headLength=Math.min(desiredRadius*8.5,length*.42),headRadius=Math.min(desiredRadius*2.9,length*.16);
  const neck=add(to,mul(axis,-headLength)),facets:ArrowFacet[]=[],sides=24;
  const radialAt=(t:number):Vec3=>add(mul(u,Math.cos(t)),mul(v,Math.sin(t)));
  for(let i=0;i<sides;i++){
    const a=radialAt(i*2*Math.PI/sides),b=radialAt((i+1)*2*Math.PI/sides),normal=radialAt((i+.5)*2*Math.PI/sides);
    const startA=add(from,mul(a,radius)),startB=add(from,mul(b,radius));
    const endA=add(neck,mul(a,radius)),endB=add(neck,mul(b,radius));
    const headA=add(neck,mul(a,headRadius)),headB=add(neck,mul(b,headRadius));
    facets.push({vertices:[startA,endA,endB,startB],normal});
    facets.push({vertices:[from,startB,startA],normal:mul(axis,-1)});
    facets.push({vertices:[headA,to,headB],normal:unit(add(mul(normal,headLength),mul(axis,headRadius)))});
    facets.push({vertices:[neck,headB,headA],normal:mul(axis,-1)});
  }
  return facets;
}

export function shadeArrow(color:string,normal:Vec3,light:Vec3):string{
  const factor=.78+.22*Math.max(0,dot(normal,light));
  return '#'+[1,3,5].map(i=>Math.round(parseInt(color.slice(i,i+2),16)*factor).toString(16).padStart(2,'0')).join('');
}
