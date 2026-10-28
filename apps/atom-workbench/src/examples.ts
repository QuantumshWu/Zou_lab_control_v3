// 示例…: ready-made scenes. Loading one replaces the whole scene; the caller
// asks first when the current scene holds edits.
import {defaultDynamics,defaultScene} from './types.js';
import type {Direction,FrequencyExpr,Scene,SpectralLine} from './types.js';
import {resolveSceneFrequencies} from './physics.js';
import {normalizeLaserIntensity} from './laser-intensity.js';

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
];

const axis=(name:'+x'|'+z'):Direction=>({reference:{kind:'axis',axis:name},thetaDeg:0,phiDeg:0});
const angles=(thetaDeg:number,phiDeg:number):Direction=>({reference:{kind:'angles',thetaDeg,phiDeg},thetaDeg:0,phiDeg:0});

/** A detuning from a zero-field transition (by default the line's F=2 → F′=3, or F″=2 on D1). */
const detuned=(line:SpectralLine,offsetMHz:number,Fg:1|2=2,Fe=line==='D1'?2:3):FrequencyExpr=>({base:{kind:'transition',line,Fg,Fe},offsetMHz});

/** The probe of every example runs along the bias field (linked to its reference axis), except
 * where the example is about a light that does not: D1 + D2 and the tilted field. */
export function exampleScene(key:string):{scene:Scene;view:ExampleView;line:SpectralLine}{
  const next=defaultScene(),probe=next.lasers[0]!;let view:ExampleView='levels',line:SpectralLine='D2';
  probe.frequency=detuned('D2',4.2);
  next.lasers.filter(l=>l.id==='tweezer').forEach(l=>l.enabled=false);
  next.fields[0]!.direction=axis('+z');next.fields[0]!.magnitudeG=3;next.dynamics.durationUs=1;
  next.name=EXAMPLES.find(example=>example.key===key)?.label??next.name;
  if(key==='three'){
    next.lasers[1]!.enabled=true;
    Object.assign(probe,{name:'探测光',frequency:detuned('D2',0,2,2)});
    next.lasers.push({...structuredClone(probe),id:'repump',name:'回泵光',frequency:detuned('D2',0,1,2)});
    next.dynamics={...defaultDynamics(),durationUs:20};
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
  resolveSceneFrequencies(next);next.lasers.forEach(normalizeLaserIntensity);
  return {scene:next,view,line};
}
