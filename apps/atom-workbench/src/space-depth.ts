import type {Vec3} from './types.js';
export function depthOpacity(z:number):number{return .52+.48*Math.max(0,Math.min(1,(z+2.2)/4.4));}
export function splitSegment(a:Vec3,b:Vec3,step=.12):[Vec3,Vec3][]{
 const count=Math.max(1,Math.ceil(Math.hypot(...a.map((v,i)=>b[i]!-v))/step));
 const at=(t:number)=>a.map((v,i)=>v+(b[i]!-v)*t)as Vec3;
 return Array.from({length:count},(_,i)=>[at(i/count),at((i+1)/count)]);
}
export function rayDepth(a:{z:number;scale?:number},b:{z:number;scale?:number},t:number):number{const u=Math.max(0,Math.min(1,t)),sa=a.scale??1,sb=b.scale??1;return ((1-u)*a.z*sa+u*b.z*sb)/((1-u)*sa+u*sb);}
