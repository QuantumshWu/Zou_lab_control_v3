// 示例…: ready-made scenes. Loading one replaces the whole scene; the caller
// asks first when the current scene holds edits.
import {defaultDynamics,defaultScene} from './types.js';
import type {Scene,SpectralLine} from './types.js';
import {normalizeLaserFrequency} from './laser-frequency.js';

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

export function exampleScene(key:string):{scene:Scene;view:ExampleView;line:SpectralLine}{
  const next=defaultScene(),probe=next.lasers[0]!;let view:ExampleView='levels',line:SpectralLine='D2';
  delete probe.referenceTransition;probe.frequencyMode='transition';probe.detuningUnit='MHz';
  next.lasers.filter(l=>l.id==='tweezer').forEach(l=>l.enabled=false);
  next.fields[0]!.thetaDeg=0;next.fields[0]!.magnitudeG=3;next.dynamics!.durationUs=1;probe.detuningMHz=4.2;
  next.name=EXAMPLES.find(example=>example.key===key)?.label??next.name;
  if(key==='three'){
    next.lasers[1]!.enabled=true;
    Object.assign(probe,{regime:'near',name:'探测光',referenceTransition:{line:'D2',Fg:2,Fe:2},detuningMHz:0});
    delete probe.frequencyAnchor;normalizeLaserFrequency(probe);
    const repump={...probe,id:'repump',name:'回泵光',referenceTransition:{line:'D2' as const,Fg:1 as const,Fe:2},detuningMHz:0};
    delete repump.frequencyAnchor;normalizeLaserFrequency(repump);next.lasers.push(repump);
    next.dynamics={...defaultDynamics(),durationUs:20};
  }
  if(key==='open'){
    Object.assign(next.fields[0]!,{magnitudeG:3,thetaDeg:0,phiDeg:0});next.initialM=0;
    Object.assign(probe,{line:'D2',detuningMHz:-266.65,intensityMWcm2:1,thetaDeg:0,phiDeg:0,ellipticityDeg:0});
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-1--1','ground-1-0','ground-1-1','ground-2-0','D2-2-1'],durationUs:20};
  }
  if(key==='cycling'){
    next.initialM=2;Object.assign(probe,{thetaDeg:0,phiDeg:0,ellipticityDeg:45,intensityMWcm2:.35});
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-2',monitoredStateIds:['ground-2-2','D2-3-3','ground-1-0'],durationUs:5};
  }
  if(key==='f3-linear'){
    next.initialM=0;Object.assign(probe,{detuningMHz:0,intensityMWcm2:1,ellipticityDeg:0});
    next.dynamics={...defaultDynamics(),durationUs:20,initialStateId:'ground-2-0'};
  }
  if(key==='rabi'){next.fields[0]!.magnitudeG=0;Object.assign(probe,{detuningMHz:0,intensityMWcm2:6.7});next.dynamics!.durationUs=.25;view='dynamics';}
  if(key==='dark'){
    Object.assign(probe,{line:'D1',detuningMHz:0,intensityMWcm2:2});next.initialM=0;
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-1--1','ground-1-0','ground-1-1','ground-2-0','D1-2-1'],durationUs:3};
    view='dynamics';line='D1';
  }
  if(key==='both'){
    Object.assign(next.fields[0]!,{magnitudeG:1.5,thetaDeg:32});next.initialM=0;
    Object.assign(probe,{name:'D2 驱动',detuningMHz:0,intensityMWcm2:2.5,ellipticityDeg:18});
    next.lasers.push({id:'d1-drive',name:'D1 驱动',enabled:true,line:'D1',detuningMHz:0,intensityMWcm2:2.5,thetaDeg:90,phiDeg:0,polarizationAngleDeg:0,ellipticityDeg:0,phaseDeg:0});
    next.dynamics={...defaultDynamics(),initialStateId:'ground-2-0',monitoredStateIds:['ground-2-0','D1-2-1','D1-2-0','D2-3-1','D2-3-0'],durationUs:.5};
    view='dynamics';
  }
  if(key==='tilted'){Object.assign(next.fields[0]!,{thetaDeg:67,phiDeg:35});probe.ellipticityDeg=28;next.initialM=0;view='space';}
  next.lasers.forEach(l=>normalizeLaserFrequency(l));
  return {scene:next,view,line};
}
