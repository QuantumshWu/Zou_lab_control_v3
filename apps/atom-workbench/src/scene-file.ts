import type {Scene,Laser,MagneticField} from './types.js';
import {POPULATION_GROUPS} from './population-selection.js';
import {DURATION_UNITS} from './duration.js';
import type {DurationUnit,InitialStateSpec} from './types.js';
import {DEFAULT_MEAN_HORIZON_US,MEAN_HORIZONS_US,defaultDynamics,resolvePopulationTolerance} from './types.js';
import {analyze} from './physics.js';
import {normalizeLaserFrequency} from './laser-frequency.js';
import {normalizeLaserIntensity} from './laser-intensity.js';

/** Validates a saved initial-state spec; ids must be physical model states. */
function initialStateSpec(input:unknown,physicalIds:Set<string>):InitialStateSpec {
  const invalid=():never=>{throw new Error('动力学初态设置无效');};
  if(!input||typeof input!=='object'||Array.isArray(input))invalid();
  const o=input as Record<string,unknown>;
  if(o.kind==='state'){if(typeof o.id!=='string'||!physicalIds.has(o.id))invalid();return {kind:'state',id:o.id as string};}
  if(o.kind==='unpolarized'){if(o.manifold!=='ground'&&o.manifold!=='F1'&&o.manifold!=='F2')invalid();return {kind:'unpolarized',manifold:o.manifold as 'ground'|'F1'|'F2'};}
  if(o.kind==='thermal'){const t=o.temperatureK;if(typeof t!=='number'||!Number.isFinite(t)||!(t>0))invalid();return {kind:'thermal',temperatureK:t as number};}
  if(o.kind==='weights'){
    const w=o.weights;if(!w||typeof w!=='object'||Array.isArray(w))invalid();
    const entries=Object.entries(w as Record<string,unknown>);
    if(!entries.length||entries.length>32||entries.some(([id,v])=>!physicalIds.has(id)||typeof v!=='number'||!Number.isFinite(v)||v<0))invalid();
    const total=entries.reduce((sum,[,v])=>sum+(v as number),0);if(!(total>0)||!Number.isFinite(total))invalid();
    return {kind:'weights',weights:Object.fromEntries(entries) as Record<string,number>};
  }
  return invalid();
}

export function validateScene(input:unknown):Scene {
  const obj=(v:unknown):Record<string,unknown>=>{if(!v||typeof v!=='object'||Array.isArray(v))throw new Error('场景结构不正确');return v as Record<string,unknown>;};
  const val=(o:Record<string,unknown>,k:string,min:number,max:number)=>{const n=o[k];if(typeof n!=='number'||!Number.isFinite(n)||n<min||n>max)throw new Error(`${k} 超出范围 ${min}…${max}`);return n;};
  const string=(o:Record<string,unknown>,k:string,max=64)=>{const v=o[k];if(typeof v!=='string'||!v.trim()||v.length>max)throw new Error(`${k} 不是有效文本`);return v;};
  const enabled=(o:Record<string,unknown>)=>{if(typeof o.enabled!=='boolean')throw new Error('enabled 必须为布尔值');return o.enabled;};
  const s=obj(input);if(s.version!==1)throw new Error('仅支持 version 1 场景');
  if(!Array.isArray(s.fields)||s.fields.length>8||!Array.isArray(s.lasers)||s.lasers.length>8)throw new Error('磁场和激光分别最多 8 个');
  const ids=new Set<string>(),id=(o:Record<string,unknown>)=>{const v=string(o,'id');if(!/^[A-Za-z0-9_-]+$/.test(v)||['__proto__','constructor','prototype'].includes(v)||ids.has(v))throw new Error('对象 ID 非法或重复');ids.add(v);return v;};
  const fields:MagneticField[]=s.fields.map(v=>{const f=obj(v);return {id:id(f),name:string(f,'name'),enabled:enabled(f),magnitudeG:val(f,'magnitudeG',0,20),thetaDeg:val(f,'thetaDeg',0,180),phiDeg:val(f,'phiDeg',-180,180)};});
  const lasers:Laser[]=s.lasers.map(v=>{const l=obj(v);if(l.line!=='D1'&&l.line!=='D2')throw new Error('光谱线必须为 D1 或 D2');
    // Only authored lab geometry and Jones parameters are restored. Legacy
    // preset/follow labels are intentionally outside this whitelist: shortcuts
    // are one-time commands and loading a scene must never apply one again.
    const result:Laser={id:id(l),name:string(l,'name'),enabled:enabled(l),line:l.line,detuningMHz:val(l,'detuningMHz',-5e9,5e9),intensityMWcm2:val(l,'intensityMWcm2',0,Number.MAX_VALUE),thetaDeg:val(l,'thetaDeg',0,180),phiDeg:val(l,'phiDeg',-180,180),polarizationAngleDeg:val(l,'polarizationAngleDeg',-90,90),ellipticityDeg:val(l,'ellipticityDeg',-45,45),phaseDeg:l.phaseDeg===undefined?0:val(l,'phaseDeg',-180,180)};
    if(l.frequencyAnchor!==undefined){const anchor=obj(l.frequencyAnchor);if(anchor.line!=='D1'&&anchor.line!=='D2')throw new Error('频率锚点 D 线无效');result.frequencyAnchor={line:anchor.line,offsetMHz:val(anchor,'offsetMHz',-Number.MAX_VALUE,Number.MAX_VALUE),viewKey:string(anchor,'viewKey',2048)};}
    if(l.intensityAnchor!==undefined){const anchor=obj(l.intensityAnchor);result.intensityAnchor={intensityMWcm2:val(anchor,'intensityMWcm2',0,Number.MAX_VALUE),viewKey:string(anchor,'viewKey',2048)};}
    if(l.directionAnchor!==undefined){const anchor=obj(l.directionAnchor);result.directionAnchor={thetaDeg:val(anchor,'thetaDeg',0,180),phiDeg:val(anchor,'phiDeg',-180,180),viewKey:string(anchor,'viewKey',2048),...(anchor.thetaResidualDeg!==undefined?{thetaResidualDeg:val(anchor,'thetaResidualDeg',-Number.MAX_VALUE,Number.MAX_VALUE)}:{}),...(anchor.phiResidualDeg!==undefined?{phiResidualDeg:val(anchor,'phiResidualDeg',-Number.MAX_VALUE,Number.MAX_VALUE)}:{})};}
    if(l.referenceTransition!==undefined){const ref=obj(l.referenceTransition);if(ref.line!=='D1'&&ref.line!=='D2')throw new Error('跃迁参考的 D 线无效');const Fg=val(ref,'Fg',1,2),Fe=val(ref,'Fe',0,3);if(!Number.isInteger(Fg)||!Number.isInteger(Fe))throw new Error('跃迁参考 F 必须为整数');if(ref.fromManifold!==undefined&&ref.fromManifold!=='ground'&&ref.fromManifold!=='D1')throw new Error('参考起始电子流形无效');result.referenceTransition={line:ref.line,Fg:Fg as 1|2,Fe,...(ref.fromManifold==='D1'?{fromManifold:'D1' as const}:{})};}
    if(l.regime!==undefined&&l.regime!=='near'&&l.regime!=='far')throw new Error('激光类型必须是 near 或 far');
    result.regime=(l.regime??(l.id==='tweezer'?'far':'near')) as 'near'|'far';
    if(l.referenceProcess!==undefined&&l.referenceProcess!=='absorption'&&l.referenceProcess!=='emission')throw new Error('光子方向无效');
    result.referenceProcess=l.referenceProcess as Laser['referenceProcess'];
    if(l.referenceStateId!==undefined)result.referenceStateId=string(l,'referenceStateId');
    if(l.frequencyMode!==undefined){if(l.frequencyMode!=='transition'&&l.frequencyMode!=='wavelength')throw new Error('频率模式无效');result.frequencyMode=l.frequencyMode;}
    if(l.wavelengthNm!==undefined)result.wavelengthNm=val(l,'wavelengthNm',Number.MIN_VALUE,Number.MAX_VALUE);
    if(l.detuningUnit!==undefined){if(!['Hz','kHz','MHz','GHz','THz'].includes(String(l.detuningUnit)))throw new Error('失谐单位无效');result.detuningUnit=l.detuningUnit as Laser['detuningUnit'];}
    if(l.intensityMode!==undefined){if(l.intensityMode!=='intensity'&&l.intensityMode!=='power')throw new Error('强度模式无效');result.intensityMode=l.intensityMode;}
    if(l.powerMW!==undefined)result.powerMW=val(l,'powerMW',0,Number.MAX_VALUE);if(l.waistUm!==undefined)result.waistUm=val(l,'waistUm',Number.MIN_VALUE,Number.MAX_VALUE);
    if(result.frequencyMode==='wavelength'&&result.wavelengthNm===undefined)throw new Error('请指定真空波长');if(result.intensityMode==='power'&&(result.powerMW===undefined||result.waistUm===undefined))throw new Error('请指定功率与束腰');return normalizeLaserIntensity(normalizeLaserFrequency(result));
  });
  const F=val(s,'initialF',1,2);if(F!==1&&F!==2)throw new Error('F 必须为 1 或 2');const m=val(s,'initialM',-F,F);if(!Number.isInteger(m))throw new Error('mF 必须为整数');
  const legacyInitial=`ground-${F}-${m}`,legacyDefaults=defaultDynamics();
  const next:Scene={version:1,name:string(s,'name',100),fields,lasers,initialF:F,initialM:m,marks:{},references:[],dynamics:{...legacyDefaults,initialStateId:legacyInitial,monitoredStateIds:[legacyInitial,...legacyDefaults.monitoredStateIds.filter(id=>id!==legacyInitial)]}};
  const validatedAnalysis=analyze(next),physicalIds=new Set([...validatedAnalysis.ground,...validatedAnalysis.D1,...validatedAnalysis.D2].map(l=>l.id));

  if(s.marks!==undefined){const marks=obj(s.marks);for(const [k,v]of Object.entries(marks)){if(!physicalIds.has(k)||typeof v!=='string'||v.length>32)throw new Error('能级标记无效');next.marks![k]=v;}}
  if(s.references!==undefined){if(!Array.isArray(s.references)||s.references.length>12)throw new Error('最多 12 个参考能级');next.references=s.references.map(v=>{const r=obj(v),anchorId=string(r,'anchorId'),referenceId=id(r);if(!physicalIds.has(anchorId))throw new Error('参考能级的锚点无效');if(physicalIds.has(referenceId))throw new Error('参考能级不能使用真实态 ID');if(r.anchorMode!==undefined&&r.anchorMode!=='follow'&&r.anchorMode!=='zero-field')throw new Error('参考能级锚点模式无效');return {id:referenceId,name:string(r,'name',32),anchorId,offsetMHz:val(r,'offsetMHz',-1e12,1e12),anchorMode:r.anchorMode as 'follow'|'zero-field'|undefined};});}
  for(const laser of lasers)if(laser.referenceStateId!==undefined&&!physicalIds.has(laser.referenceStateId)&&!next.references!.some(r=>r.id===laser.referenceStateId))throw new Error('激光参考态不存在');
  if(s.dynamics!==undefined){
    const d=obj(s.dynamics),laserId=typeof d.laserId==='string'?d.laserId.slice(0,64):'';
    // A 'state' spec is the same thing as initialStateId: fold it in, so each
    // initial state has one representation. Mixed kinds keep their spec.
    const spec=d.initialState===undefined?undefined:initialStateSpec(d.initialState,physicalIds),initialStateId=spec?.kind==='state'?spec.id:string(d,'initialStateId');
    if(!physicalIds.has(initialStateId)||!Array.isArray(d.monitoredStateIds)||d.monitoredStateIds.length>32||new Set(d.monitoredStateIds).size!==d.monitoredStateIds.length||d.monitoredStateIds.some(v=>typeof v!=='string'||!physicalIds.has(v)))throw new Error('动力学态选择无效：最多 32 个不重复的物理态');
    const samples=val(d,'samples',2,2001);if(!Number.isInteger(samples))throw new Error('samples 必须为整数');
    // Scenes from before the horizon read at the 1 s default.
    const meanHorizonUs=d.meanHorizonUs===undefined?DEFAULT_MEAN_HORIZON_US:d.meanHorizonUs;if(!MEAN_HORIZONS_US.includes(meanHorizonUs as number|null))throw new Error('长期布居的观察时长无效');
    if(d.durationUnit!==undefined&&!DURATION_UNITS.includes(d.durationUnit as DurationUnit))throw new Error('时间单位无效');
    if(d.monitorMode!==undefined&&d.monitorMode!=='auto'&&d.monitorMode!=='manual')throw new Error('观察态选择模式无效');
    const groups=d.monitoredGroupIds??[];if(!Array.isArray(groups)||groups.length>POPULATION_GROUPS.length||new Set(groups).size!==groups.length||groups.some(id=>!POPULATION_GROUPS.some(g=>g.id===id)))throw new Error('F 布居分组无效');
    next.dynamics={laserId,initialStateId,...(spec&&spec.kind!=='state'?{initialState:spec}:{}),monitorMode:d.monitorMode==='manual'?'manual':'auto',monitoredGroupIds:[...groups] as string[],monitoredStateIds:[...d.monitoredStateIds] as string[],durationUs:val(d,'durationUs',Number.MIN_VALUE,Number.MAX_VALUE),...(d.durationUnit!==undefined?{durationUnit:d.durationUnit as DurationUnit}:{}),samples,populationTolerance:resolvePopulationTolerance(d.populationTolerance),meanHorizonUs:meanHorizonUs as number|null,...(d.quality==='verified'||d.quality==='preview'?{quality:d.quality}:{}),autoRun:d.autoRun!==false};
  }
  const initial=[...validatedAnalysis.ground,...validatedAnalysis.D1,...validatedAnalysis.D2].find(l=>l.id===next.dynamics?.initialStateId);if(initial?.manifold==='ground'){next.initialF=initial.F as 1|2;next.initialM=initial.m;}
  return next;
}
