import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze,composeMagneticField,laserIntensityMWcm2,laserFrequencyMHz,laserDirectionAxis,setLaserDirectionAxis,relativeDirectionAxis} from '../dist/js/atom-workbench/src/physics.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {setLaserIntensityMode,setLaserLocalIntensity,setLaserPowerMW,setLaserWaistUm} from '../dist/js/atom-workbench/src/laser-intensity.js';
import {setLaserDetuningUnit,setLaserDetuningValue,setLaserFrequencyMode,setLaserWavelengthNm} from '../dist/js/atom-workbench/src/laser-frequency.js';
const roundtrip=scene=>validateScene(JSON.parse(JSON.stringify(scene)));
const field=(id,magnitudeG,thetaDeg=61,phiDeg=19)=>({id,name:id,enabled:true,magnitudeG,thetaDeg,phiDeg});
const key=scene=>dynamicsCacheKey(scene,scene.dynamics,true);
function cycling(){const s=defaultScene();s.fields=[field('b',3)];s.lasers=s.lasers.slice(0,1);Object.assign(s.lasers[0],{thetaDeg:61,phiDeg:19,ellipticityDeg:45});s.dynamics.initialStateId='ground-2-2';return s;}
function connected(model){const found=new Set([model.initialIndex]);for(let previous=-1;previous!==found.size;){previous=found.size;for(const i of found)for(let j=0;j<model.dimension;j++)if(model.hReal[j*model.dimension+i]!==0||model.hImag[j*model.dimension+i]!==0)found.add(j);}return [...found].sort((a,b)=>a-b);}

test('power/intensity conversion preserves exact local I, edits replace the canonical value, and JSON roundtrips retain it',()=>{
  let s=defaultScene(),l=s.lasers[1];setLaserIntensityMode(l,'intensity');setLaserLocalIntensity(l,1);
  for(let i=0;i<100;i++){setLaserIntensityMode(l,'power');assert.equal(laserIntensityMWcm2(l),1);assert.notEqual(l.powerMW,4);s=roundtrip(s);l=s.lasers[1];setLaserIntensityMode(l,'intensity');assert.equal(laserIntensityMWcm2(l),1);}
  setLaserIntensityMode(l,'power');const p=l.powerMW;setLaserWaistUm(l,2.4);assert.equal(l.powerMW,p);assert.equal(laserIntensityMWcm2(l),2*p/(Math.PI*2.4**2)*1e8);
  setLaserPowerMW(l,4);assert.equal(laserIntensityMWcm2(l),2*4/(Math.PI*2.4**2)*1e8);
  const before=laserIntensityMWcm2(l);l.powerMW=2;assert.equal(laserIntensityMWcm2(l),before/2,'legacy direct edit invalidates its anchor');
  setLaserIntensityMode(l,'intensity');const current=laserIntensityMWcm2(l);setLaserWaistUm(l,1.2);assert.equal(laserIntensityMWcm2(l),current,'inactive waist does not change local-I input');
  setLaserLocalIntensity(l,1e14);setLaserIntensityMode(l,'power');assert.equal(laserIntensityMWcm2(roundtrip(s).lasers[1]),1e14,'converted values do not encounter unrelated parser caps');
  setLaserWaistUm(l,.05);setLaserPowerMW(l,10000);const edgeI=laserIntensityMWcm2(l);assert.ok(edgeI>1e12);setLaserIntensityMode(l,'intensity');assert.equal(l.intensityMWcm2,edgeI);s=roundtrip(s);l=s.lasers[1];setLaserIntensityMode(l,'power');assert.equal(laserIntensityMWcm2(l),edgeI);
});

test('pure input-mode changes cannot spoil two-beam destructive interference by one ULP',()=>{
  let s=cycling();setLaserLocalIntensity(s.lasers[0],.12345678912345678);s.lasers.push({...structuredClone(s.lasers[0]),id:'cancel',phaseDeg:180});
  const baseline=buildFullMasterModel(s);for(let step=0;step<100;step++){setLaserIntensityMode(s.lasers[1],step%2?'intensity':'power');s=roundtrip(s);}
  const result=buildFullMasterModel(s);assert.deepEqual(result.hReal,baseline.hReal);assert.deepEqual(result.hImag,baseline.hImag);assert.deepEqual(result.drives,baseline.drives);
  for(let g=0;g<8;g++)for(let e=8;e<32;e++){assert.equal(result.hReal[g*32+e],0);assert.equal(result.hImag[g*32+e],0);}
  assert.equal(laserIntensityMWcm2(s.lasers[0]),laserIntensityMWcm2(s.lasers[1]));
});

test('all positive supported detunings survive wavelength/transition, export/import and direct wavelength edits',()=>{
  for(const detuning of [-350,1000,5000]){let s=defaultScene(),l=s.lasers[0];setLaserDetuningUnit(l,'THz');setLaserDetuningValue(l,detuning);const frequency=laserFrequencyMHz(l),baseKey=key(s);for(let i=0;i<10;i++){setLaserFrequencyMode(l,'wavelength');s=roundtrip(s);l=s.lasers[0];assert.equal(laserFrequencyMHz(l),frequency);assert.equal(key(s),baseKey);setLaserFrequencyMode(l,'transition');s=roundtrip(s);l=s.lasers[0];assert.equal(laserFrequencyMHz(l),frequency);}}
  const s=defaultScene(),l=s.lasers[0];setLaserWavelengthNm(l,8758.729242094549);assert.equal(roundtrip(s).lasers[0].wavelengthNm,8758.729242094549);
  const before=JSON.stringify(l);for(const invalid of [0,-1,Number.MIN_VALUE,Infinity,NaN]){assert.throws(()=>setLaserWavelengthNm(l,invalid));assert.equal(JSON.stringify(l),before,'failed input is atomic');}
});

test('single B and split collinear B give bitwise identical full H, radiative metadata and exact cycling support',()=>{
  const s=cycling(),single=buildFullMasterModel(s),singleKey=key(s);assert.deepEqual(connected(single),[7,31]);
  s.fields=[field('a',1.5),field('b',1.5)];const split=buildFullMasterModel(s);assert.deepEqual(analyze(s).fieldAxis,{thetaDeg:61,phiDeg:19});assert.deepEqual(split.hReal,single.hReal);assert.deepEqual(split.hImag,single.hImag);assert.deepEqual(split.collapse,single.collapse);assert.deepEqual(connected(split),[7,31]);assert.equal(key(s),singleKey);
  s.fields=[field('a',4.5),field('opposite',1.5,119,-161)];const subtract=buildFullMasterModel(s);assert.deepEqual(subtract.hReal,single.hReal);assert.deepEqual(subtract.hImag,single.hImag);assert.deepEqual(connected(subtract),[7,31]);assert.equal(key(s),singleKey);
});

test('field cancellation is order robust without losing a genuine 1e-20 G remainder or transverse angle',()=>{
  for(const magnitudes of [[1,1e-20,-1],[1,-1,1e-20],[1e-20,1,-1],[-1,1e-20,1]]){
    const fields=magnitudes.map((m,i)=>field(String(i),m)),actual=composeMagneticField(fields),expected=composeMagneticField([field('tiny',1e-20)]);assert.deepEqual(actual,expected);
    const s=cycling();s.fields=fields;const t=cycling();t.fields=[field('tiny',1e-20)];assert.deepEqual(buildFullMasterModel(s).hReal,buildFullMasterModel(t).hReal);assert.deepEqual(buildFullMasterModel(s).hImag,buildFullMasterModel(t).hImag);assert.equal(key(s),key(t));
  }
  const zero=composeMagneticField([field('a',1),field('b',1,119,-161)]);assert.deepEqual(zero.fieldVector,[0,0,0]);assert.equal(zero.fieldMagnitudeG,0);
  const perturbed=composeMagneticField([field('tiny-angle',1,1e-20,0),field('opposite',1,180,0)]);assert.ok(perturbed.fieldVector[0]>0);assert.equal(perturbed.fieldVector[2],0);assert.equal(perturbed.fieldAxis.thetaDeg,90);
  const s=cycling();s.fields=[field('z',1,0,0),field('tiny-angle',1,1e-20,0)];Object.assign(s.lasers[0],{thetaDeg:0,phiDeg:0});const real=buildFullMasterModel(s);assert.ok(connected(real).length>2,'real tilt still opens forbidden-at-exact-alignment channels');
  const noTilt=structuredClone(s);noTilt.fields[1].thetaDeg=0;assert.notDeepEqual(real.hReal,buildFullMasterModel(noTilt).hReal);assert.notEqual(key(s),key(noTilt));
});

test('the cache identity explicitly includes the operator quantization-axis convention',()=>{
  const s=cycling(),parsed=JSON.parse(key(s));assert.deepEqual(parsed.fieldAxis,analyze(s).fieldAxis);assert.deepEqual(parsed.field,buildFullMasterModel(s).fieldVector);
});

test('true sub-ULP tilts survive spherical-angle conversion at both poles, equator and azimuth branch boundaries',()=>{
  for(const [theta,phi,perturbTheta,perturbPhi] of [[0,0,90,0],[180,0,90,0],[90,0,0,0],[90,180,90,90],[90,90,90,0],[90,-90,90,0]]){
    const s=cycling();s.fields=[field('base',1,theta,phi)];Object.assign(s.lasers[0],{thetaDeg:theta,phiDeg:phi});const pure=buildFullMasterModel(s);assert.deepEqual(connected(pure),[7,31]);const oldKey=key(s);
    s.fields.push(field('physical-perturbation',1e-20,perturbTheta,perturbPhi));const tilted=buildFullMasterModel(s),a=analyze(s);assert.notDeepEqual([tilted.hReal,tilted.hImag],[pure.hReal,pure.hImag],`lost transverse field at ${theta},${phi}`);assert.ok(connected(tilted).length>2);assert.notEqual(key(s),oldKey);
    const laserPi=a.polarizations[s.lasers[0].id].pi;assert.ok(laserPi>0&&laserPi<1e-38,`tiny physical pi weight lost or exaggerated: ${laserPi}`);
    assert.ok(Math.abs(laserPi/5e-41-1)<1e-14);assert.ok(Math.abs(a.polarizations[s.lasers[0].id].minus/6.25e-82-1)<1e-14,'opposite-helicity delta^4 strength must survive');
    assert.deepEqual(JSON.parse(key(s)).fieldAxis,a.fieldAxis);
  }
});

test('relative direction shortcuts preserve their exact fixed lab axis across JSON, B edits, and explicit angle entry',()=>{
  for(const [theta,phi,pt,pp] of [[180,0,90,0],[90,180,90,90],[90,90,90,0]]){
    let s=cycling();s.fields=[field('base',1,theta,phi),field('tiny',1e-20,pt,pp)];const axis=analyze(s).fieldAxis;
    setLaserDirectionAxis(s.lasers[0],relativeDirectionAxis(axis,'parallel'),true);const aligned=buildFullMasterModel(s);assert.deepEqual(connected(aligned),[7,31]);assert.equal(analyze(s).polarizations.probe.pi,0);const alignedKey=key(s);
    s=roundtrip(s);assert.deepEqual(laserDirectionAxis(s.lasers[0]),axis);assert.deepEqual(buildFullMasterModel(s).hReal,aligned.hReal);assert.deepEqual(buildFullMasterModel(s).hImag,aligned.hImag);assert.equal(key(s),alignedKey);
    const pol=analyze(s).polarizations.probe,vector=composeMagneticField(s.fields).fieldVector;assert.ok(pol.k.some(v=>v!==0&&Math.abs(v)<1e-18),'3D propagation retains tiny component');for(let i=0;i<3;i++)assert.ok(Math.abs(pol.k[i]-vector[i])<1e-35||pol.k[i]===vector[i]);
    const fixed=laserDirectionAxis(s.lasers[0]);s.fields[1].magnitudeG=2e-20;assert.deepEqual(laserDirectionAxis(s.lasers[0]),fixed);assert.ok(analyze(s).polarizations.probe.pi>0);assert.ok(connected(buildFullMasterModel(s)).length>2,'changing B does not auto-rotate the saved lab direction');
    setLaserDirectionAxis(s.lasers[0],{thetaDeg:s.lasers[0].thetaDeg,phiDeg:s.lasers[0].phiDeg});assert.equal(s.lasers[0].directionAnchor,undefined);assert.ok(analyze(s).polarizations.probe.pi>0,'explicit displayed angles remove low residual');
    setLaserDirectionAxis(s.lasers[0],relativeDirectionAxis(axis,'anti'),true);const anti=analyze(s).polarizations.probe;assert.ok(anti.minus>.999999999999);assert.ok(anti.pi<1e-38);
  }
});
