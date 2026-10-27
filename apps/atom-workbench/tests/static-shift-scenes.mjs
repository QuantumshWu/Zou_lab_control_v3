// Scenes for the static background checks: far-detuned lasers shown in the levels chart, by name.
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {laserIntensityMWcm2,setPolarizationPreset} from '../dist/js/atom-workbench/src/physics.js';
import {displayMixing} from '../dist/js/atom-workbench/src/spectroscopy.js';

/** A far beam at a vacuum wavelength, local intensity in mW/cm². */
function farBeam(id,wavelengthNm,intensityMWcm2){
 return {id,name:id,enabled:true,showLightShift:true,line:'D2',frequencyMode:'wavelength',wavelengthNm,referenceTransition:{line:'D2',Fg:2,Fe:3},detuningMHz:0,detuningUnit:'THz',
  intensityMode:'intensity',intensityMWcm2,thetaDeg:90,phiDeg:0,polarizationAngleDeg:0,ellipticityDeg:0,phaseDeg:0};
}
/** Scale a far beam's intensity until its static mixing |Ω/2Δ| is `ratio`. */
function atMixing(scene,id,ratio){
 const beam=scene.lasers.find(l=>l.id===id),base=displayMixing(scene,id).ratio,intensity=laserIntensityMWcm2(beam);
 Object.assign(beam,{intensityMode:'intensity',intensityMWcm2:intensity*(ratio/base)**2,intensityAnchor:undefined,powerMW:undefined,waistUm:undefined});
 return scene;
}
function tweezer({polarization='pi',fieldG=3,mixing}={}){
 const s=defaultScene();s.fields[0].magnitudeG=fieldG;s.lasers=s.lasers.filter(l=>l.id==='tweezer');
 setPolarizationPreset(s.lasers[0],s.fields,polarization==='pi'?'pi':'sigma-plus');
 return mixing===undefined?s:atMixing(s,'tweezer',mixing);
}
/** 5P3/2 -> 4D5/2 lies at 196.0239 THz, 5S -> 6P3/2 at 713.2839 THz (spectroscopy-data.ts). */
const C=299792458;
const nearLine=(id,lineTHz,offsetGHz)=>farBeam(id,C/((lineTHz+offsetGHz*1e-3)*1e12)*1e9,1);
export const STATIC_SHIFT_SCENES={
 'pi-3G':()=>tweezer(),
 'pi-3G-high':()=>tweezer({mixing:.009}),
 'sigma-3G':()=>tweezer({polarization:'sigma'}),
 'sigma-3G-high':()=>tweezer({polarization:'sigma',mixing:.009}),
 'pi-0G':()=>tweezer({fieldG:0}),
 'pi-0G-high':()=>tweezer({fieldG:0,mixing:.009}),
 'tweezer+1064':()=>{const s=tweezer();s.lasers.push(farBeam('lattice',1064,2e7));return s;},
 'near-4D5/2':()=>{const s=tweezer({fieldG:3});s.lasers=[nearLine('telecom',196.0239,60)];return atMixing(s,'telecom',.003);},
 'near-6P3/2':()=>{const s=tweezer({fieldG:3});s.lasers=[nearLine('blue',713.2839,80)];return atMixing(s,'blue',.003);},
};
/** Beyond the 1% static-mixing limit: the solver itself, outside the levels chart's validity rule. */
export const BEYOND_LIMIT_SCENES={
 'pi-3G-2%':()=>tweezer({mixing:.02}),
 'sigma-3G-2%':()=>tweezer({polarization:'sigma',mixing:.02}),
};
