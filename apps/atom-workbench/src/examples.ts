// 示例…: ready-made scenes. Loading one replaces the whole scene; the caller
// asks first when the current scene holds edits.
import {defaultDynamics,defaultScene,defaultTweezer} from './types.js';
import type {AxisName,Direction,FrequencyExpr,Laser,Scene,SpectralLine} from './types.js';
import {RB87,resolveSceneFrequencies} from './physics.js';
import {normalizeLaserIntensity} from './laser-intensity.js';
import {tweezerRefusal} from './tweezer.js';

export type ExampleView='levels'|'dynamics'|'space';
export const EXAMPLES:readonly {key:string;label:string}[]=[
  {key:'three',label:'光镊 + 探测光 + 回泵光'},
  {key:'open',label:'单束光 · F=2 → F′=2'},
  {key:'cycling',label:'D2 · 循环跃迁'},
  {key:'f3-linear',label:'F=2 → F′=3 · 线偏振'},
  {key:'rabi',label:'D2 · Rabi 振荡'},
  {key:'dark',label:'D1 · F=2 → F=1 光抽运'},
  {key:'both',label:'D1 + D2 · 同时驱动'},
  {key:'tilted',label:'倾斜磁场'},
  {key:'pgc3d',label:'光镊 · 3D PGC'},
];
/** The tweezer PGC example (typical in-trap PGC): the tweezer's power (mW) and waist (μm); the
 * cooling beams' detuning from F=2 → F′=3 and the repump's from F=1 → F′=2 (in units of the D2 Γ,
 * as in free space), and each beam's intensity (mW/cm²); the duration (μs). */
export const TWEEZER_PGC={tweezerMW:3.5,tweezerWaistUm:1,coolingDetuningGamma:-6,repumpDetuningGamma:0,coolingMWcm2:3,repumpMWcm2:1,durationUs:1e4};
/** The Raman-resonant Λ example (光镊 + 探测光 + 回泵光): its duration (μs, written in ms) and samples. */
export const LAMBDA_TWEEZER={durationUs:1e3,durationUnit:'ms',samples:1001} as const;

const axis=(name:AxisName):Direction=>({reference:{kind:'axis',axis:name},thetaDeg:0,phiDeg:0});
const angles=(thetaDeg:number,phiDeg:number):Direction=>({reference:{kind:'angles',thetaDeg,phiDeg},thetaDeg:0,phiDeg:0});

/** A detuning from a zero-field transition (by default the line's F=2 → F′=3, or F″=2 on D1). */
const detuned=(line:SpectralLine,offsetMHz:number,Fg:1|2=2,Fe=line==='D1'?2:3):FrequencyExpr=>({base:{kind:'transition',line,Fg,Fe},offsetMHz});

/** The probe of every example runs along the bias field (linked to its reference axis), except
 * where the example is about a light that does not: D1 + D2 and the tilted field. Motion runs where
 * the example has a tweezer to hold the atom (光镊 + 探测光 + 回泵光, 3D PGC); the examples of the
 * internal physics alone switch the tweezer off, and motion with it. */
export function exampleScene(key:string):{scene:Scene;view:ExampleView;line:SpectralLine}{
  const next=defaultScene(),probe=next.lasers[0]!;let view:ExampleView='levels',line:SpectralLine='D2';
  probe.frequency=detuned('D2',4.2);
  next.lasers.filter(l=>l.id==='tweezer').forEach(l=>l.enabled=false);
  next.fields[0]!.direction=axis('+z');next.fields[0]!.magnitudeG=3;next.dynamics.durationUs=1;
  next.name=EXAMPLES.find(example=>example.key===key)?.label??next.name;
  if(key==='three'){
    // A Raman-resonant Λ in the tweezer: the probe on F=2 → F′=2 and the repump on F=1 → F′=2, both
    // on resonance (free space), the atom moving in 3D in the default 852 nm tweezer.
    next.lasers[1]!.enabled=true;
    Object.assign(probe,{name:'探测光',frequency:detuned('D2',0,2,2)});
    next.lasers.push({...structuredClone(probe),id:'repump',name:'回泵光',frequency:detuned('D2',0,1,2)});
    next.dynamics={...defaultDynamics(),...LAMBDA_TWEEZER};
    view='dynamics';
  }
  if(key==='open'){
    Object.assign(probe,{frequency:detuned('D2',-266.65),intensityMWcm2:1,ellipticityDeg:0});
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-1--1','ground-1-0','ground-1-1','ground-2-0','D2-2-1'],durationUs:20};
  }
  if(key==='cycling'){
    Object.assign(probe,{ellipticityDeg:45,intensityMWcm2:.35});
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-2',monitoredStateIds:['ground-2-2','D2-3-3','ground-1-0'],durationUs:5};
  }
  if(key==='f3-linear'){
    Object.assign(probe,{frequency:detuned('D2',0),intensityMWcm2:1,ellipticityDeg:0});
    next.dynamics={...defaultDynamics(),durationUs:20,initialStateId:'ground-2-0'};
  }
  if(key==='rabi'){next.fields[0]!.magnitudeG=0;Object.assign(probe,{frequency:detuned('D2',0),intensityMWcm2:6.7});next.dynamics.durationUs=.25;view='dynamics';}
  if(key==='dark'){
    Object.assign(probe,{frequency:detuned('D1',0),intensityMWcm2:2});
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-1--1','ground-1-0','ground-1-1','ground-2-0','D1-2-1'],durationUs:3};
    view='dynamics';line='D1';
  }
  if(key==='both'){
    Object.assign(next.fields[0]!,{magnitudeG:1.5,direction:angles(32,0)});
    Object.assign(probe,{name:'D2 驱动',frequency:detuned('D2',0),intensityMWcm2:2.5,ellipticityDeg:18,direction:axis('+z')});
    next.lasers.push({...structuredClone(probe),id:'d1-drive',name:'D1 驱动',frequency:detuned('D1',0),ellipticityDeg:0,direction:axis('+x')});
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-2-0','D1-2-1','D1-2-0','D2-3-1','D2-3-0'],durationUs:.5};
    view='dynamics';
  }
  if(key==='tilted'){next.fields[0]!.direction=angles(67,35);Object.assign(probe,{ellipticityDeg:28,direction:axis('+z')});view='space';}
  if(key==='pgc3d'){
    // Polarization-gradient cooling of one atom in the 852 nm tweezer, which runs along z polarized
    // along x (π to the quantization axis: no field, its reference axis along x). The MOT's six
    // beams kept on for PGC: a σ+σ− D2 cooling pair along each of x, y and z below F=2 → F′=3 (both
    // beams of a pair of one helicity about their own k; the z pair, along the quadrupole axis, of
    // the other helicity than the x and y pairs; TWEEZER_PGC). A pair's two circular beams make a
    // polarization turning along the pair at constant intensity, so the three pairs need no choice
    // of linear axes. One repump beam along +x on F=1 → F′=2 (a scene holds eight lasers). The atom
    // starts in |2,+2⟩ at 30 μK and moves in 3D. Every beam's phase is zero at the focus: the six
    // beams' fixed interference there sets how well each axis cools (with these phases x the least;
    // another phase on one beam moves the pattern), as a frozen snapshot of the drifting phases of
    // an experiment's beams.
    Object.assign(next.fields[0]!,{magnitudeG:0,direction:axis('+x')});
    const gamma=RB87.D2.gammaHz/(2*Math.PI*1e6),cooling=detuned('D2',TWEEZER_PGC.coolingDetuningGamma*gamma),repump=detuned('D2',TWEEZER_PGC.repumpDetuningGamma*gamma,1,2);
    const beam=(id:string,name:string,frequency:FrequencyExpr,intensityMWcm2:number,along:AxisName,polarizationAngleDeg:number,ellipticityDeg:number):Laser=>
      ({...structuredClone(probe),id,name,frequency,intensityMWcm2,direction:axis(along),polarizationAngleDeg,ellipticityDeg,phaseDeg:0});
    const pair=(name:'x'|'y'|'z',helicity:1|-1)=>(['+','-'] as const).map(sign=>
      beam(`cool-${sign==='+'?'plus':'minus'}-${name}`,`冷却光 ${sign==='+'?'+':'−'}${name}`,cooling,TWEEZER_PGC.coolingMWcm2,`${sign}${name}` as AxisName,0,45*helicity));
    next.lasers=[{...defaultTweezer(),direction:axis('+z'),powerMW:TWEEZER_PGC.tweezerMW,waistUm:TWEEZER_PGC.tweezerWaistUm},
      ...pair('x',1),...pair('y',1),...pair('z',-1),beam('repump','回泵光 +x',repump,TWEEZER_PGC.repumpMWcm2,'+x',0,0)];
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-2',durationUs:TWEEZER_PGC.durationUs,durationUnit:'ms',samples:1001};
    view='dynamics';
  }
  resolveSceneFrequencies(next);next.lasers.forEach(normalizeLaserIntensity);
  next.dynamics.motion.enabled=!tweezerRefusal(next);
  return {scene:next,view,line};
}
