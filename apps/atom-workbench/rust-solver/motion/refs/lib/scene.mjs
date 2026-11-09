// Scene building blocks on the app's own scene schema (types.ts Scene, version 3): every reference scene is an ordinary
// app scene -- what an operator could set up -- built from the app's defaults and read back through its importer
// (app.mjs finish). Directions are lab axes; the quantization axis is the bias field's.
import {app} from './app.mjs';

export const axis=name=>({reference:{kind:'axis',axis:name},thetaDeg:0,phiDeg:0});

/** The app's default scene emptied of lasers: a bias field of `fieldG` along `fieldAxis`, motion on (CPU), the given
 * initial state, temperature (µK), duration (µs) and samples. */
export async function base({name,fieldG=3,fieldAxis='+z',initialStateId='ground-2-2',temperatureUK=30,durationUs,samples,device='cpu'}){
  const {types}=await app();
  const s=types.defaultScene();
  s.name=name;s.lasers=[];delete s.tweezerId;
  Object.assign(s.fields[0],{magnitudeG:fieldG,direction:axis(fieldAxis),enabled:true});
  Object.assign(s.dynamics,{initialStateId,durationUs,samples,durationUnit:'μs',monitoredStateIds:[initialStateId],monitorMode:'auto'});
  delete s.dynamics.initialState;
  Object.assign(s.dynamics.motion,{enabled:true,temperatureUK,temperatureUnit:'μK',device});
  return s;
}

/** A tweezer (the designated trap): a Gaussian beam focused at the atom, given by power and waist, of vacuum wavelength
 * `nm`, along `along`; polarization angle ψ and ellipticity χ as the app defines them (χ = 45°: E = (u + iv)/√2,
 * helicity +1 about its own k). */
export async function tweezer(scene,{id='tweezer',name='光镊',nm,powerMW,waistUm,along,polarizationAngleDeg=0,ellipticityDeg=0}){
  const {types}=await app();
  const t={...types.defaultTweezer(),id,name,frequency:{wavelengthNm:nm},frequencyUnit:'nm',intensityMode:'power',powerMW,waistUm,
    direction:axis(along),polarizationAngleDeg,ellipticityDeg,phaseDeg:0,showLightShift:true};
  delete t.intensityAnchor;delete t.frequencyAnchor;
  scene.lasers.push(t);scene.tweezerId=id;return t;
}

/** A near-resonant beam on a zero-field transition (`line`, F → F′) offset by `offsetMHz`: a plane wave of local intensity
 * `intensityMWcm2` (intensity mode), or a Gaussian beam focused at the atom when `waistUm` is given (power mode). */
export async function beam(scene,{id,name=id,line,Fg,Fe,offsetMHz=0,intensityMWcm2,waistUm,along,polarizationAngleDeg=0,ellipticityDeg=0,phaseDeg=0,wavelengthNm}){
  const l={id,name,enabled:true,showLightShift:false,line,frequency:wavelengthNm?{wavelengthNm}:{base:{kind:'transition',line,Fg,Fe},offsetMHz},frequencyUnit:wavelengthNm?'nm':'MHz',
    intensityMode:waistUm?'power':'intensity',intensityMWcm2,powerUnit:'mW',waistUnit:'μm',intensityUnit:'mW/cm²',
    direction:axis(along),polarizationAngleDeg,ellipticityDeg,phaseDeg};
  if(waistUm){l.waistUm=waistUm;l.powerMW=intensityMWcm2*Math.PI*waistUm*waistUm/2e8;}
  scene.lasers.push(l);return l;
}

/** The retro-reflection of `of` (reflection.ts): a mirror far away sending it straight back, reflectivity R, the
 * reflected beam's phase at the atom relative to the source's, with or without a double pass through a quarter-wave
 * plate. Every physical field is derived from the source when the scene is resolved. */
export async function reflection(scene,of,{reflectivity=1,phaseDeg=0,quarterWave=false}={}){
  const {reflection:r}=await app();
  const source=scene.lasers.find(l=>l.id===of);
  const back=r.addReflection(scene,source);
  back.reflection={of,reflectivity,phaseDeg,quarterWave};
  return back;
}
