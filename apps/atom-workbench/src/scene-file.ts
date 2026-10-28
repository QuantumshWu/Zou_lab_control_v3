// The one scene schema (types.ts Scene, version SCENE_VERSION). A file is validated strictly
// against it: every required field present and in range, no other field. A file written by an
// earlier version is refused with the reason -- nothing is migrated.
import type {AxisName,Direction,DynamicsSettings,FrequencyDisplayUnit,FrequencyExpr,InitialStateSpec,Laser,LaserFrequencyAnchor,LaserIntensityAnchor,
  MagneticField,ReferenceAxis,ReferenceFrequency,ReferenceLevel,Scene} from './types.js';
import {MEAN_HORIZONS_US,SCENE_VERSION,defaultScene} from './types.js';
import {POPULATION_GROUPS} from './population-selection.js';
import {DURATION_UNITS} from './duration.js';
import {analyze,resolveSceneFrequencies} from './physics.js';
import {DETUNING_UNITS} from './laser-frequency.js';
import {INTENSITY_MODES,INTENSITY_UNITS,POWER_UNITS,WAIST_UNITS,checkNumericalAperture,normalizeLaserIntensity} from './laser-intensity.js';
import {TEMPERATURE_UNITS} from './initial-state.js';
import {AXIS_NAMES,referenceAxis} from './direction.js';
import {initialPopulations} from './full-master-model.js';

type Obj=Record<string,unknown>;
const FREQUENCY_UNITS:readonly string[]=[...DETUNING_UNITS,'nm','μm'];
/** A readout's units: a frequency's, and an energy difference read as a temperature. */
const READOUT_UNITS:readonly string[]=[...FREQUENCY_UNITS,'mK'];

/** Reads one object of the file at `path` ("lasers[1].frequency"): its fields, checked as read. */
class Reader{
  constructor(readonly o:Obj,readonly path:string){}
  static of(value:unknown,path:string,required:readonly string[],optional:readonly string[]=[]):Reader{
    if(!value||typeof value!=='object'||Array.isArray(value))fail(path,'应为对象');
    const o=value as Obj;
    for(const key of required)if(!Object.hasOwn(o,key))fail(path,`缺少 ${key}`);
    for(const key of Object.keys(o))if(!required.includes(key)&&!optional.includes(key))fail(path,`含有未知字段 ${key}`);
    return new Reader(o,path);
  }
  at(key:string):string{return `${this.path}.${key}`;}
  has(key:string):boolean{return Object.hasOwn(this.o,key);}
  number(key:string,min=-Number.MAX_VALUE,max=Number.MAX_VALUE):number{const v=this.o[key];if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max)fail(this.at(key),`应为 ${min}…${max} 内的数值`);return v as number;}
  integer(key:string,min:number,max:number):number{const v=this.number(key,min,max);if(!Number.isInteger(v))fail(this.at(key),'应为整数');return v;}
  text(key:string,max=64):string{const v=this.o[key];if(typeof v!=='string'||!v.trim()||v.length>max)fail(this.at(key),`应为 1 至 ${max} 个字符的文本`);return v as string;}
  boolean(key:string):boolean{const v=this.o[key];if(typeof v!=='boolean')fail(this.at(key),'应为布尔值');return v as boolean;}
  choice<T>(key:string,choices:readonly T[]):T{const v=this.o[key] as T;if(!choices.includes(v))fail(this.at(key),`应为 ${choices.map(c=>JSON.stringify(c)).join('、')} 之一`);return v;}
  list(key:string,max:number):unknown[]{const v=this.o[key];if(!Array.isArray(v)||v.length>max)fail(this.at(key),`应为至多 ${max} 项的数组`);return v as unknown[];}
}
function fail(path:string,what:string):never{throw new Error(`${path} ${what}`);}

function direction(value:unknown,path:string):Direction{
  const d=Reader.of(value,path,['reference','thetaDeg','phiDeg']),r=Reader.of(d.o.reference,d.at('reference'),['kind'],['axis','id','thetaDeg','phiDeg','thetaResidualDeg','phiResidualDeg']);
  let reference:ReferenceAxis;
  const kind=r.choice('kind',['axis','field','angles'] as const);
  if(kind==='axis'){Reader.of(r.o,r.path,['kind','axis']);reference={kind,axis:r.choice('axis',AXIS_NAMES) as AxisName};}
  else if(kind==='field'){Reader.of(r.o,r.path,['kind','id']);reference={kind,id:r.text('id')};}
  else{
    Reader.of(r.o,r.path,['kind','thetaDeg','phiDeg'],['thetaResidualDeg','phiResidualDeg']);
    reference={kind,thetaDeg:r.number('thetaDeg',0,180),phiDeg:r.number('phiDeg',-180,180),
      ...(r.has('thetaResidualDeg')?{thetaResidualDeg:r.number('thetaResidualDeg')}:{}),...(r.has('phiResidualDeg')?{phiResidualDeg:r.number('phiResidualDeg')}:{})};
  }
  return {reference,thetaDeg:d.number('thetaDeg',0,180),phiDeg:d.number('phiDeg',-180,180)};
}
/** A frequency expression (ui-design.md §1.2) and its unit; that its base exists and no cycle forms is checked once every object is known. */
function frequency(owner:Reader):{frequency:FrequencyExpr;frequencyUnit:FrequencyDisplayUnit}{
  const path=owner.at('frequency'),raw=owner.o.frequency as Obj|undefined;
  if(raw&&typeof raw==='object'&&Object.hasOwn(raw,'wavelengthNm')){
    const f=Reader.of(raw,path,['wavelengthNm']);
    return {frequency:{wavelengthNm:f.number('wavelengthNm',Number.MIN_VALUE)},frequencyUnit:owner.choice('frequencyUnit',['nm','μm'] as const)};
  }
  const f=Reader.of(raw,path,['base','offsetMHz']),b=Reader.of(f.o.base,f.at('base'),['kind'],['id','line','Fg','Fe','fromManifold']),offsetMHz=f.number('offsetMHz',-5e9,5e9);
  const unit=owner.choice('frequencyUnit',DETUNING_UNITS);
  const kind=b.choice('kind',['absolute','transition','laser','reference'] as const);
  if(kind==='absolute'){Reader.of(b.o,b.path,['kind']);return {frequency:{base:{kind},offsetMHz},frequencyUnit:unit};}
  if(kind==='laser'||kind==='reference'){Reader.of(b.o,b.path,['kind','id']);return {frequency:{base:{kind,id:b.text('id')},offsetMHz},frequencyUnit:unit};}
  Reader.of(b.o,b.path,['kind','line','Fg','Fe'],['fromManifold']);
  const fromManifold=b.has('fromManifold')?b.choice('fromManifold',['D1'] as const):undefined;
  return {frequency:{base:{kind,line:b.choice('line',['D1','D2'] as const),Fg:b.choice('Fg',[1,2] as const),Fe:b.integer('Fe',0,3),...(fromManifold?{fromManifold}:{})},offsetMHz},frequencyUnit:unit};
}
function frequencyAnchor(value:unknown,path:string):LaserFrequencyAnchor{
  const a=Reader.of(value,path,['line','offsetMHz','viewKey']);
  return {line:a.choice('line',['D1','D2'] as const),offsetMHz:a.number('offsetMHz'),viewKey:a.text('viewKey',2048)};
}
function intensityAnchor(value:unknown,path:string):LaserIntensityAnchor{
  const a=Reader.of(value,path,['intensityMWcm2','viewKey']);return {intensityMWcm2:a.number('intensityMWcm2',0),viewKey:a.text('viewKey',2048)};
}
const LASER_REQUIRED=['id','name','enabled','line','frequency','frequencyUnit','showLightShift','intensityMode','intensityMWcm2','powerUnit','waistUnit','intensityUnit',
  'direction','polarizationAngleDeg','ellipticityDeg','phaseDeg'];
const LASER_OPTIONAL=['referenceStateId','referenceProcess','frequencyAnchor','intensityAnchor','powerMW','waistUm','numericalAperture'];
function laser(value:unknown,path:string,id:(r:Reader)=>string):Laser{
  const l=Reader.of(value,path,LASER_REQUIRED,LASER_OPTIONAL),mode=l.choice('intensityMode',INTENSITY_MODES);
  const result:Laser={id:id(l),name:l.text('name'),enabled:l.boolean('enabled'),line:l.choice('line',['D1','D2'] as const),...frequency(l),
    showLightShift:l.boolean('showLightShift'),intensityMode:mode,intensityMWcm2:l.number('intensityMWcm2',0),
    powerUnit:l.choice('powerUnit',POWER_UNITS),waistUnit:l.choice('waistUnit',WAIST_UNITS),intensityUnit:l.choice('intensityUnit',INTENSITY_UNITS),
    direction:direction(l.o.direction,l.at('direction')),polarizationAngleDeg:l.number('polarizationAngleDeg',-90,90),ellipticityDeg:l.number('ellipticityDeg',-45,45),phaseDeg:l.number('phaseDeg',-180,180)};
  if(l.has('referenceStateId'))result.referenceStateId=l.text('referenceStateId');
  if(l.has('referenceProcess'))result.referenceProcess=l.choice('referenceProcess',['absorption','emission'] as const);
  if(l.has('frequencyAnchor'))result.frequencyAnchor=frequencyAnchor(l.o.frequencyAnchor,l.at('frequencyAnchor'));
  if(l.has('intensityAnchor'))result.intensityAnchor=intensityAnchor(l.o.intensityAnchor,l.at('intensityAnchor'));
  if(l.has('powerMW'))result.powerMW=l.number('powerMW',0);
  if(l.has('waistUm'))result.waistUm=l.number('waistUm',Number.MIN_VALUE);
  if(l.has('numericalAperture')){const na=l.number('numericalAperture');try{result.numericalAperture=checkNumericalAperture(na);}catch(error){fail(l.at('numericalAperture'),(error as Error).message);}}
  // The mode's own inputs are there.
  if(mode==='power'&&(result.powerMW===undefined||result.waistUm===undefined))fail(path,'功率与束腰方式缺少 powerMW 或 waistUm');
  if(mode==='na'&&(result.powerMW===undefined||result.numericalAperture===undefined))fail(path,'功率与 NA 方式缺少 powerMW 或 numericalAperture');
  return result;
}
function referenceFrequency(value:unknown,path:string,id:(r:Reader)=>string):ReferenceFrequency{
  const r=Reader.of(value,path,['id','name','frequency','frequencyUnit'],['frequencyAnchor']);
  return {id:id(r),name:r.text('name',32),...frequency(r),...(r.has('frequencyAnchor')?{frequencyAnchor:frequencyAnchor(r.o.frequencyAnchor,r.at('frequencyAnchor'))}:{})};
}
function referenceLevel(value:unknown,path:string,id:(r:Reader)=>string):ReferenceLevel{
  const r=Reader.of(value,path,['id','name','anchorId','anchorMode','offsetMHz','offsetUnit'],['photon']);
  const level:ReferenceLevel={id:id(r),name:r.text('name',32),anchorId:r.text('anchorId'),anchorMode:r.choice('anchorMode',['follow','zero-field'] as const),
    offsetMHz:r.number('offsetMHz',-1e12,1e12),offsetUnit:r.choice('offsetUnit',DETUNING_UNITS)};
  if(r.has('photon')){
    const p=Reader.of(r.o.photon,r.at('photon'),['source','direction']),s=Reader.of(p.o.source,p.at('source'),['kind','id']);
    level.photon={source:{kind:s.choice('kind',['laser','reference'] as const),id:s.text('id')},direction:p.choice('direction',['absorb','emit'] as const)};
  }
  return level;
}
/** A mixed initial state (a single state is dynamics.initialStateId alone); ids must be physical model states. */
function initialState(value:unknown,path:string,physical:Set<string>):InitialStateSpec{
  const kind=Reader.of(value,path,['kind'],['manifold','temperatureK','weights']).choice('kind',['unpolarized','thermal','weights'] as const);
  if(kind==='unpolarized')return {kind,manifold:Reader.of(value,path,['kind','manifold']).choice('manifold',['ground','F1','F2'] as const)};
  if(kind==='thermal')return {kind,temperatureK:Reader.of(value,path,['kind','temperatureK']).number('temperatureK',Number.MIN_VALUE)};
  const w=Reader.of(Reader.of(value,path,['kind','weights']).o.weights,`${path}.weights`,[],[...physical]),weights:Record<string,number>={};
  for(const id of Object.keys(w.o))weights[id]=w.number(id,0);
  if(!Object.keys(weights).length||!(Object.values(weights).reduce((a,b)=>a+b,0)>0))fail(w.path,'至少一个态的权重应为正');
  return {kind,weights};
}
const DYNAMICS_REQUIRED=['initialStateId','monitoredStateIds','monitorMode','monitoredGroupIds','durationUs','durationUnit','samples','meanHorizonUs'];

/** A scene file, validated strictly against the current schema; throws why it is not one. */
export function validateScene(input:unknown):Scene {
  try{return readScene(input);}catch(error){throw new Error(`场景文件不符合当前格式（version ${SCENE_VERSION}）：${error instanceof Error?error.message:String(error)}`);}
}
function readScene(input:unknown):Scene {
  // A file of another version is said to be one, before anything it lacks or has more.
  if(input&&typeof input==='object'&&!Array.isArray(input)&&(input as Obj).version!==SCENE_VERSION)fail('version',`为 ${JSON.stringify((input as Obj).version)}，应为 ${SCENE_VERSION}`);
  const s=Reader.of(input,'场景',['version','name','fields','lasers','referenceFrequencies','references','marks','dynamics'],['display']);
  const ids=new Set<string>(),id=(r:Reader):string=>{const v=r.text('id');if(!/^[A-Za-z0-9_-]+$/.test(v)||['__proto__','constructor','prototype'].includes(v)||ids.has(v))fail(r.at('id'),'非法或重复');ids.add(v);return v;};
  const fields:MagneticField[]=s.list('fields',8).map((v,i)=>{const f=Reader.of(v,`fields[${i}]`,['id','name','enabled','magnitudeG','direction']);
    return {id:id(f),name:f.text('name'),enabled:f.boolean('enabled'),magnitudeG:f.number('magnitudeG',0,20),direction:direction(f.o.direction,f.at('direction'))};});
  const lasers=s.list('lasers',8).map((v,i)=>laser(v,`lasers[${i}]`,id));
  const referenceFrequencies=s.list('referenceFrequencies',8).map((v,i)=>referenceFrequency(v,`referenceFrequencies[${i}]`,id));
  const references=s.list('references',12).map((v,i)=>referenceLevel(v,`references[${i}]`,id));
  // Every linked field exists and no link forms a cycle.
  for(const object of [...fields,...lasers])try{referenceAxis(fields,object.direction.reference);}catch(error){throw new Error(`「${object.name}」的参考轴无效：${(error as Error).message}`);}
  const d=Reader.of(s.o.dynamics,'dynamics',DYNAMICS_REQUIRED,['initialState','temperatureUnit']),marks=s.o.marks;
  if(!marks||typeof marks!=='object'||Array.isArray(marks))fail('marks','应为对象');
  const scene:Scene={version:SCENE_VERSION,name:s.text('name',100),fields,lasers,referenceFrequencies,references,marks:{},
    dynamics:{initialStateId:d.text('initialStateId'),monitoredStateIds:[],monitorMode:d.choice('monitorMode',['auto','manual'] as const),monitoredGroupIds:[],
      durationUs:d.number('durationUs',Number.MIN_VALUE),durationUnit:d.choice('durationUnit',DURATION_UNITS),samples:d.integer('samples',2,2001),
      meanHorizonUs:d.choice('meanHorizonUs',MEAN_HORIZONS_US)}};
  if(s.has('display')){
    const display=Reader.of(s.o.display,'display',['readouts']),readouts=display.o.readouts;
    if(!readouts||typeof readouts!=='object'||Array.isArray(readouts)||Object.keys(readouts).length>16)fail('display.readouts','应为至多 16 项的对象');
    scene.display={readouts:Object.fromEntries(Object.entries(readouts as Obj).map(([key,v])=>{
      if(key.length>64)fail('display.readouts','键过长');const pref=Reader.of(v,`display.readouts.${key}`,[],['unit','base']);
      return [key,{...(pref.has('unit')?{unit:pref.choice('unit',READOUT_UNITS)}:{}),...(pref.has('base')?{base:pref.text('base',128)}:{})}];}))};
  }
  // Every base exists, no frequency follows itself, every frequency is positive; then each intensity.
  resolveSceneFrequencies(scene);
  for(const [i,l] of lasers.entries())try{normalizeLaserIntensity(l);}catch(error){fail(`lasers[${i}]`,(error as Error).message);}
  const analysis=analyze(scene),levels=[...analysis.ground,...analysis.D1,...analysis.D2],physical=new Set(levels.map(level=>level.id));
  for(const [key,v] of Object.entries(marks as Obj)){if(!physical.has(key)||typeof v!=='string'||v.length>32)fail(`marks.${key}`,'应为真实态的至多 32 个字符的标记');scene.marks[key]=v;}
  for(const [i,r] of references.entries()){
    if(!physical.has(r.anchorId))fail(`references[${i}].anchorId`,'应为真实态');
    if(physical.has(r.id))fail(`references[${i}].id`,'不能使用真实态 ID');
    if(r.photon&&!(r.photon.source.kind==='laser'?lasers:referenceFrequencies).some(o=>o.id===r.photon!.source.id))fail(`references[${i}].photon`,'链接的频率不存在');
  }
  for(const [i,l] of lasers.entries())if(l.referenceStateId!==undefined&&!physical.has(l.referenceStateId)&&!references.some(r=>r.id===l.referenceStateId))fail(`lasers[${i}].referenceStateId`,'参考态不存在');
  const dynamics=scene.dynamics as DynamicsSettings;
  if(!physical.has(dynamics.initialStateId))fail('dynamics.initialStateId','应为真实态');
  const monitored=d.list('monitoredStateIds',32);
  if(new Set(monitored).size!==monitored.length||monitored.some(v=>typeof v!=='string'||!physical.has(v)))fail('dynamics.monitoredStateIds','应为不重复的真实态');
  dynamics.monitoredStateIds=[...monitored] as string[];
  const groups=d.list('monitoredGroupIds',POPULATION_GROUPS.length);
  if(new Set(groups).size!==groups.length||groups.some(v=>!POPULATION_GROUPS.some(g=>g.id===v)))fail('dynamics.monitoredGroupIds','应为不重复的 F 布居分组');
  dynamics.monitoredGroupIds=[...groups] as string[];
  if(d.has('initialState')){
    const spec=initialState(d.o.initialState,'dynamics.initialState',physical);
    try{initialPopulations(spec,levels);}catch(error){fail('dynamics.initialState',(error as Error).message);}
    dynamics.initialState=spec;
  }
  // A thermal initial state has its temperature's unit, and only it has one.
  if((dynamics.initialState?.kind==='thermal')!==d.has('temperatureUnit'))fail('dynamics.temperatureUnit',dynamics.initialState?.kind==='thermal'?'热分布初态缺少温度单位':'只用于热分布初态');
  if(d.has('temperatureUnit'))dynamics.temperatureUnit=d.choice('temperatureUnit',TEMPERATURE_UNITS);
  return scene;
}

/** The autosaved scene: the saved one when it is a valid current scene, else the default scene
 * and, briefly, why the saved one was refused (the caller keeps the original for download). */
export function loadSavedScene(text:string|null):{scene:Scene;failure?:string}{
  if(text===null)return {scene:defaultScene()};
  let input:unknown;try{input=JSON.parse(text);}catch{return {scene:defaultScene(),failure:'不是 JSON'};}
  try{return {scene:readScene(input)};}catch(error){return {scene:defaultScene(),failure:error instanceof Error?error.message:String(error)};}
}
