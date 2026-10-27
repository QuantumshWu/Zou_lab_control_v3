// Directions for tests, written as the scene file writes them (direction.ts).
/** The usual spherical (θ, φ): the reference axis z and the offset. */
export const spherical=(thetaDeg,phiDeg)=>({reference:{kind:'axis',axis:'+z'},thetaDeg,phiDeg});
/** A reference axis given by angles, no offset: how a field saved as (θ, φ) loads. */
export const along=(thetaDeg,phiDeg)=>({reference:{kind:'angles',thetaDeg,phiDeg},thetaDeg:0,phiDeg:0});
/** A field of `magnitudeG` along (θ, φ). */
export const field=(id,magnitudeG,thetaDeg=0,phiDeg=0)=>({id,name:id,enabled:true,magnitudeG,direction:along(thetaDeg,phiDeg)});
/** One field along a lab vector (its direction as angles). */
export function fieldAlong(vector,id='b'){
  const [x,y,z]=vector,r=Math.hypot(x,y,z);
  return field(id,r,Math.acos(z/r)*180/Math.PI,Math.atan2(y,x)*180/Math.PI);
}
