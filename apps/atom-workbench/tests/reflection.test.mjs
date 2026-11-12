// Retro-reflections (reflection.ts, ui-design.md §6.23): the beam a reflection takes from its source,
// its polarization in the code's conventions (physics.ts: E ∝ cos χ u + i sin χ v, v = k × u), the
// scene file, and the export -- an ordinary beam, the master model of the same beam written by hand.
import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_PLATE_ANGLE_DEG,MAXIMUM_LASERS,defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze,captureLaserFrequencyAnchor,drivingLasers,laserIntensityMWcm2,polarization,resolveSceneFrequencies} from '../dist/js/atom-workbench/src/physics.js';
import {addReflection,detachReflection,partnerOf,reflectionOf,sourceOf} from '../dist/js/atom-workbench/src/reflection.js';
import {directionVector} from '../dist/js/atom-workbench/src/direction.js';
import {laserWaistUm,setLaserIntensityMode,writeLaserIntensity} from '../dist/js/atom-workbench/src/laser-intensity.js';
import {setFrequencyBase} from '../dist/js/atom-workbench/src/laser-frequency.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {setTweezer} from '../dist/js/atom-workbench/src/tweezer.js';
import {laserSlot} from '../dist/js/atom-workbench/src/identity.js';
import {along,field} from './directions.mjs';

const DEG=Math.PI/180;
const roundtrip=scene=>validateScene(JSON.parse(JSON.stringify(scene)));
const axis=name=>({reference:{kind:'axis',axis:name},thetaDeg:0,phiDeg:0});
/** A beam's lab-frame Jones vector e^{iφ}(cos χ u + i sin χ v): per lab axis [re, im]. */
function jones(laser,fields){
  const p=polarization(laser,fields),c=Math.cos(laser.ellipticityDeg*DEG),s=Math.sin(laser.ellipticityDeg*DEG),ph=laser.phaseDeg*DEG;
  return [0,1,2].map(i=>{const re=c*p.u[i],im=s*p.v[i];return [re*Math.cos(ph)-im*Math.sin(ph),re*Math.sin(ph)+im*Math.cos(ph)];});
}
/** The helicity about the beam's own k, i (ε × ε*)·k for a unit ε: +1 σ⁺, −1 σ⁻, 0 linear. */
function helicity(laser,fields){
  const e=jones(laser,fields),k=polarization(laser,fields).k;
  // i (ε × ε*) = 2 Im(ε*) × Re(ε)... written out: i(ε×ε*)_a = 2 (Re ε_b Im ε_c − Im ε_b Re ε_c) over (a, b, c) cyclic.
  return [0,1,2].reduce((sum,a)=>{const b=(a+1)%3,c=(a+2)%3;return sum+2*(e[b][0]*e[c][1]-e[b][1]*e[c][0])*k[a];},0);
}
const close=(a,b,tol,what)=>assert.ok(Math.abs(a-b)<=tol,`${what}: ${a} vs ${b}`);
/** Components with a zero written +0: a turn by 180° may leave a −0. */
const plain=v=>v.map(x=>x+0);
/** The scene of one source along `direction` (the default scene's probe) with the given polarization, and its reflection. */
function pair(direction,{psi=0,chi=0,phase=0,quarterWave=false,plate=psi,reflectivity=1,mirrorPhase=0,fields}={}){
  const scene=defaultScene();if(fields)scene.fields=fields;
  const source=scene.lasers[0];Object.assign(source,{direction,polarizationAngleDeg:psi,ellipticityDeg:chi,phaseDeg:phase});
  // The probe alone: no tweezer.
  scene.lasers=[source];delete scene.tweezerId;const reflection=addReflection(scene,source);
  Object.assign(reflection.reflection,{quarterWave,plateAngleDeg:plate,reflectivity,phaseDeg:mirrorPhase});resolveSceneFrequencies(scene);
  return {scene,source,reflection};
}
const GEOMETRIES=[['along the bias field (a link)',{reference:{kind:'field',id:'bias'},thetaDeg:0,phiDeg:0}],['along +x',axis('+x')],
  ['tilted, angles reference with an offset',{reference:{kind:'angles',thetaDeg:37,phiDeg:21},thetaDeg:28,phiDeg:-117}],['a field link with an offset',{reference:{kind:'field',id:'bias'},thetaDeg:133,phiDeg:180}]];

test('a reflection follows its source: the carrier bit for bit, the switch, the opposite k, R times the power and intensity at the same waist',()=>{
  const scene=defaultScene(),probe=scene.lasers[0],tweezer=scene.lasers[1];
  const a=addReflection(scene,probe),b=addReflection(scene,tweezer);
  assert.deepEqual(scene.lasers.map(l=>l.id),['probe','probe-reflection','tweezer','tweezer-reflection'],'each reflection right under its source');
  assert.equal(a.name,'探测光（反射）');assert.equal(sourceOf(scene,a),probe);assert.equal(reflectionOf(scene,probe),a);assert.equal(partnerOf(scene,probe),a);assert.equal(partnerOf(scene,a),probe);
  b.reflection.reflectivity=.5;resolveSceneFrequencies(scene);
  for(const [source,reflection] of [[probe,a],[tweezer,b]]){
    assert.deepEqual(captureLaserFrequencyAnchor(reflection).offsetMHz,captureLaserFrequencyAnchor(source).offsetMHz);assert.equal(reflection.line,source.line);
    assert.deepEqual(plain(directionVector(scene.fields,reflection.direction)),directionVector(scene.fields,source.direction).map(v=>0-v),'exactly opposite');
  }
  assert.equal(laserIntensityMWcm2(a),laserIntensityMWcm2(probe));
  assert.equal(laserIntensityMWcm2(b),laserIntensityMWcm2(tweezer)*.5);assert.equal(b.powerMW,tweezer.powerMW*.5);assert.equal(laserWaistUm(b),laserWaistUm(tweezer));assert.equal(b.intensityMode,'power');
  // Edits to the source reach the reflection where the scene is resolved; no light in, no reflection.
  Object.assign(probe,{ellipticityDeg:30,polarizationAngleDeg:-20,intensityMWcm2:3,intensityAnchor:undefined,enabled:false});probe.frequency={...probe.frequency,offsetMHz:-12};resolveSceneFrequencies(scene);
  assert.deepEqual([a.enabled,a.ellipticityDeg,a.polarizationAngleDeg,laserIntensityMWcm2(a)],[false,-30,20,3]);
  assert.equal(captureLaserFrequencyAnchor(a).offsetMHz,captureLaserFrequencyAnchor(probe).offsetMHz);
  assert.ok(!drivingLasers(scene).includes(a));
  // The colour is the source's; a reflection added or detached moves no other laser's.
  assert.deepEqual(scene.lasers.map(l=>laserSlot(scene,l.id)),[0,0,1,1]);
  detachReflection(scene,a);assert.deepEqual(scene.lasers.map(l=>[l.id,laserSlot(scene,l.id)]),[['probe',0],['tweezer',1],['tweezer-reflection',1],['probe-reflection',2]]);
  assert.equal(a.reflection,undefined);assert.deepEqual(a.frequency,probe.frequency);assert.equal(captureLaserFrequencyAnchor(a).offsetMHz,captureLaserFrequencyAnchor(probe).offsetMHz);
  probe.ellipticityDeg=0;resolveSceneFrequencies(scene);assert.equal(a.ellipticityDeg,-30,'a detached laser is its own');
});

test("the intensity is the source's exact one in every mode, and the carrier follows an offset-locked source",()=>{
  const scene=defaultScene(),probe=scene.lasers[0];
  writeLaserIntensity(probe,1.2345678901,'mW/cm²');setLaserIntensityMode(probe,'na');setFrequencyBase(scene,probe,{kind:'laser',id:'tweezer'});
  const reflection=addReflection(scene,probe);
  assert.equal(laserIntensityMWcm2(reflection),1.2345678901,'the typed intensity, not one recomputed from P and NA');assert.equal(reflection.numericalAperture,probe.numericalAperture);
  scene.lasers[2].frequency={wavelengthNm:850};resolveSceneFrequencies(scene);
  assert.equal(captureLaserFrequencyAnchor(reflection).offsetMHz,captureLaserFrequencyAnchor(probe).offsetMHz,'it follows its source as the source follows the tweezer');
  assert.equal(laserWaistUm(reflection),laserWaistUm(probe),'NA mode: the waist follows the wavelength');assert.equal(laserIntensityMWcm2(reflection),laserIntensityMWcm2(probe));
  assert.equal(laserIntensityMWcm2(roundtrip(scene).lasers[1]),laserIntensityMWcm2(reflection));
});

test('the limits: a reflection counts toward eight lasers, a laser has one, a reflection is not reflected',()=>{
  const scene=defaultScene(),probe=scene.lasers[0];
  const reflection=addReflection(scene,probe);
  assert.throws(()=>addReflection(scene,probe),/已有反射光/);assert.throws(()=>addReflection(scene,reflection),/反射光不能再被反射/);
  while(scene.lasers.length<MAXIMUM_LASERS)scene.lasers.push({...structuredClone(probe),id:`extra-${scene.lasers.length}`});
  assert.throws(()=>addReflection(scene,scene.lasers[2]),/最多 8 束激光（反射光也计入）/);assert.equal(scene.lasers.length,MAXIMUM_LASERS);
});

test('polarization: linear stays linear on the same lab axis; a plain mirror sends σ⁺ back σ⁻ about its own k, a λ/4 double pass σ⁺',()=>{
  const fields=[field('bias',3,0,0)];fields[0].direction=axis('+z');
  for(const [where,direction] of GEOMETRIES)for(const quarterWave of [false,true]){
    const words=`${where}, ${quarterWave?'λ/4':'plain'}`;
    for(const psi of [0,30,-71]){
      const {scene,source,reflection}=pair(direction,{psi,quarterWave,fields});
      const s=polarization(source,scene.fields),r=polarization(reflection,scene.fields);
      s.k.forEach((v,i)=>close(r.k[i],-v,1e-15,`${words}: k reversed`));
      // The same lab axis: u′ = u (the major axis), the field linear along it.
      s.u.forEach((v,i)=>close(r.u[i],v,1e-15,`${words}: ψ ${psi}° keeps its axis`));
      close(helicity(reflection,scene.fields),0,1e-15,`${words}: still linear`);
    }
    // σ⁺ about the source's k.
    const {scene,source,reflection}=pair(direction,{psi:12,chi:45,quarterWave,fields});
    close(helicity(source,scene.fields),1,1e-15,'σ⁺ source');
    close(helicity(reflection,scene.fields),quarterWave?1:-1,1e-15,`${words}: helicity about its own k`);
  }
  // Along the field: the plain mirror's σ⁻-about-k reflection still drives Δm = +1 about B (the
  // lab-frame field is the source's); the λ/4 one drives Δm = −1 (the MOT pair).
  for(const quarterWave of [false,true]){
    const {scene,source,reflection}=pair(GEOMETRIES[0][1],{chi:45,quarterWave,fields}),a=analyze(scene);
    assert.equal(a.polarizations[source.id].plus,1);
    assert.deepEqual([a.polarizations[reflection.id].minus,a.polarizations[reflection.id].pi,a.polarizations[reflection.id].plus],quarterWave?[1,0,0]:[0,0,1]);
  }
});

test('the lab-frame field: a plain mirror returns the source\'s Jones vector times e^{iφ}, a λ/4 double pass its mirror image about the plate\'s fast axis',()=>{
  const fields=[field('bias',2,0,0)];fields[0].direction=along(61,19);
  for(const [where,direction] of GEOMETRIES)for(const [quarterWave,plate] of [[false,45],[true,24],[true,69],[true,-80],[true,90]])for(const [chi,mirrorPhase,phase] of [[17,0,0],[-33,73,40],[45,180,-150],[0,0,170]]){
    const {scene,source,reflection}=pair(direction,{psi:24,chi,phase,quarterWave,plate,mirrorPhase,fields}),words=`${where}, ${quarterWave?`λ/4 at ${plate}°`:'plain'}, χ ${chi}°`;
    const p=polarization(source,scene.fields),c=Math.cos(chi*DEG),s=Math.sin(chi*DEG),total=(phase+mirrorPhase)*DEG;
    // The half-wave plate R(θ) diag(1, −1) R(−θ) in the source's frame mirrors u, v about the fast axis, at θ − ψ from u:
    // u″ = cos 2(θ − ψ) u + sin 2(θ − ψ) v, v″ = sin 2(θ − ψ) u − cos 2(θ − ψ) v; the field e^{i(φ₀+φ)} (cos χ u″ + i sin χ v″).
    // The plain mirror keeps u and v.
    const t=quarterWave?2*(plate-24)*DEG:0,cu=Math.cos(t),su=Math.sin(t);
    const u=[0,1,2].map(i=>quarterWave?cu*p.u[i]+su*p.v[i]:p.u[i]),v=[0,1,2].map(i=>quarterWave?su*p.u[i]-cu*p.v[i]:p.v[i]);
    const expected=[0,1,2].map(i=>{const re=c*u[i],im=s*v[i];return [re*Math.cos(total)-im*Math.sin(total),re*Math.sin(total)+im*Math.cos(total)];});
    jones(reflection,scene.fields).forEach(([re,im],i)=>{close(re,expected[i][0],1e-14,`${words}: Re ε_${i}`);close(im,expected[i][1],1e-14,`${words}: Im ε_${i}`);});
  }
});

test('a linear source and the λ/4 plate at 45° to it: the reflection comes back polarized at right angles (lin⊥lin); π about the field comes back half σ⁺, half σ⁻',()=>{
  const fields=[field('bias',3,0,0)];fields[0].direction=axis('+z');
  for(const [where,direction] of GEOMETRIES)for(const psi of [0,30,-71])for(const plate of [psi+45,psi-45].filter(t=>Math.abs(t)<=90)){
    const {scene,source,reflection}=pair(direction,{psi,quarterWave:true,plate,fields});
    const s=polarization(source,scene.fields),r=polarization(reflection,scene.fields);
    close(s.u.reduce((sum,v,i)=>sum+v*r.u[i],0),0,1e-15,`${where}, ψ ${psi}°, θ ${plate}°: u′ ⊥ u`);
    close(helicity(reflection,scene.fields),0,1e-15,`${where}: still linear`);
  }
  // The π probe across the field (the app's π preset: along +x, polarized along B) and its reflection through the default plate.
  const {scene,reflection}=pair(axis('+x'),{psi:0,quarterWave:true,plate:45,fields}),a=analyze(scene);
  const q=a.polarizations[reflection.id];close(q.pi,0,1e-15,'no π');close(q.minus,.5,1e-15,'σ⁻ half');close(q.plus,.5,1e-15,'σ⁺ half');
  // A new mirror's plate sits at the default.
  const fresh=defaultScene();assert.equal(addReflection(fresh,fresh.lasers[0]).reflection.plateAngleDeg,DEFAULT_PLATE_ANGLE_DEG);
});

test('the scene file: a reflection round-trips, its beam is derived again on reading, and what is not a reflection is refused',()=>{
  const scene=defaultScene(),reflection=addReflection(scene,scene.lasers[1]);Object.assign(reflection.reflection,{reflectivity:.81,phaseDeg:-45,quarterWave:true});
  scene.lasers[1].ellipticityDeg=20;resolveSceneFrequencies(scene);
  const loaded=roundtrip(scene),back=loaded.lasers.find(l=>l.id===reflection.id);
  assert.deepEqual(back,reflection);assert.deepEqual(roundtrip(loaded),loaded);
  // A file's beam for a reflection is what its source gives, whatever the file says.
  const stale=JSON.parse(JSON.stringify(scene)),at=stale.lasers.findIndex(l=>l.id===reflection.id);
  Object.assign(stale.lasers[at],{enabled:false,ellipticityDeg:-7,powerMW:99,polarizationAngleDeg:5});
  assert.deepEqual(validateScene(stale).lasers[at],reflection);
  // A file written before the plate had an angle reads with the default's.
  const older=JSON.parse(JSON.stringify(scene));delete older.lasers[at].reflection.plateAngleDeg;
  assert.equal(validateScene(older).lasers[at].reflection.plateAngleDeg,DEFAULT_PLATE_ANGLE_DEG);
  // A scene without reflections reads as before: no field added.
  assert.ok(roundtrip(defaultScene()).lasers.every(l=>!Object.hasOwn(l,'reflection')));
  const file=()=>JSON.parse(JSON.stringify(scene));
  for(const [label,change,reason] of [
    ['a missing source',f=>{f.lasers[at].reflection.of='nobody';},/lasers\[2\]\.reflection\.of 应为场景中另一束激光/],
    ['itself',f=>{f.lasers[at].reflection.of=f.lasers[at].id;},/应为场景中另一束激光/],
    ['a reflection of a reflection',f=>{const again={...structuredClone(f.lasers[at]),id:'again'};again.reflection.of=f.lasers[at].id;f.lasers.push(again);},/不能是反射光/],
    ['two reflections of one laser',f=>{f.lasers.push({...structuredClone(f.lasers[at]),id:'second'});},/「光镊」已有反射光/],
    ['a reflectivity above one',f=>{f.lasers[at].reflection.reflectivity=1.5;},/reflection\.reflectivity/],
    ['a negative reflectivity',f=>{f.lasers[at].reflection.reflectivity=-.1;},/reflection\.reflectivity/],
    ['a mirror phase out of range',f=>{f.lasers[at].reflection.phaseDeg=200;},/reflection\.phaseDeg/],
    ['no wave-plate switch',f=>{delete f.lasers[at].reflection.quarterWave;},/reflection 缺少 quarterWave/],
    ['a plate angle out of range',f=>{f.lasers[at].reflection.plateAngleDeg=120;},/reflection\.plateAngleDeg/],
    ['an unknown mirror field',f=>{f.lasers[at].reflection.angleDeg=45;},/reflection 含有未知字段 angleDeg/],
    ['a laser following a reflection',f=>{f.lasers[0].frequency={base:{kind:'laser',id:f.lasers[at].id},offsetMHz:5};f.lasers[0].frequencyUnit='MHz';},/lasers\[0\]\.frequency\.base 不能跟随反射光/],
    ['a ninth laser',f=>{for(let i=0;i<6;i++)f.lasers.push({...structuredClone(f.lasers[0]),id:`more-${i}`});},/lasers 应为至多 8 项的数组/],
  ]){const f=file();change(f);assert.throws(()=>validateScene(f),reason,label);}
});

/** The reflection written by hand as an independent laser, by the rules of reflection.ts. */
function byHand(source,reflection){
  const {reflection:mirror,...own}=structuredClone(reflection),d=source.direction;
  let axis=mirror.quarterWave?source.polarizationAngleDeg-2*mirror.plateAngleDeg:0-source.polarizationAngleDeg,phase=source.phaseDeg+mirror.phaseDeg;
  if(axis>90){axis-=180;phase+=180;}else if(axis<-90){axis+=180;phase+=180;}
  phase=phase>180?phase-360:phase<-180?phase+360:phase;
  return {...own,frequency:structuredClone(source.frequency),frequencyUnit:source.frequencyUnit,frequencyAnchor:structuredClone(source.frequencyAnchor),
    direction:{reference:structuredClone(d.reference),thetaDeg:180-d.thetaDeg,phiDeg:d.phiDeg>0?d.phiDeg-180:d.phiDeg+180},
    polarizationAngleDeg:axis,ellipticityDeg:mirror.quarterWave?source.ellipticityDeg:0-source.ellipticityDeg,phaseDeg:phase};
}
test('the export: a reflection is an ordinary beam -- its master model, with motion too, is that of the same beam written by hand',()=>{
  for(const quarterWave of [false,true]){
    const {scene,source,reflection}=pair({reference:{kind:'angles',thetaDeg:37,phiDeg:21},thetaDeg:28,phiDeg:-117},{psi:24,chi:31,phase:40,quarterWave,plate:-63,reflectivity:.64,mirrorPhase:73});
    const hand=structuredClone(scene);hand.lasers[1]=byHand(source,reflection);
    assert.deepEqual(buildFullMasterModel(scene),buildFullMasterModel(hand),quarterWave?'λ/4':'plain');
  }
  // A tweezer focused at the atom along +x and its reflection: two Gaussian beams of the motion
  // block, the tweezer the trap. The reflection may be the one designated: it traps as written by hand.
  const scene=defaultScene();scene.lasers=[scene.lasers[1]];const tweezer=scene.lasers[0],reflection=addReflection(scene,tweezer);
  const hand=structuredClone(scene);hand.lasers[1]=byHand(tweezer,reflection);
  const motion={axis:'3d',temperatureUK:30},model=buildFullMasterModel(scene,scene.dynamics,motion);
  assert.deepEqual(model,buildFullMasterModel(hand,hand.dynamics,motion));assert.equal(model.motion.trapBeamId,'tweezer');
  const [forward,back]=model.motion.beams;
  assert.deepEqual(plain(back.direction),forward.direction.map(v=>0-v));assert.equal(back.waistUm,forward.waistUm);assert.equal(back.intensityMWcm2,forward.intensityMWcm2);
  setTweezer(scene,reflection);setTweezer(hand,hand.lasers[1]);
  const mirrored=buildFullMasterModel(scene,scene.dynamics,motion);assert.equal(mirrored.motion.trapBeamId,reflection.id);
  assert.deepEqual(mirrored,buildFullMasterModel(hand,hand.dynamics,motion));
});

test('the MOT pair written by hand -- σ⁺ about its own k along +z and along −z, the second at phase 180° in its own frame -- is the λ/4 reflection; σ⁻ along −z is the plain one',()=>{
  // The −z axis's frame turns +z's x by 180° where the reflection keeps it: the same field is the
  // hand-written beam at the opposite sign, φ = 180°.
  for(const quarterWave of [false,true]){
    const {scene,source,reflection}=pair(axis('+z'),{chi:45,quarterWave,fields:[{...field('bias',3),direction:axis('+z')}]});
    const hand=structuredClone(scene),{reflection:_,...own}=structuredClone(reflection);
    hand.lasers[1]={...own,frequency:structuredClone(source.frequency),frequencyUnit:source.frequencyUnit,frequencyAnchor:structuredClone(source.frequencyAnchor),
      direction:axis('-z'),polarizationAngleDeg:0,ellipticityDeg:quarterWave?45:-45,phaseDeg:180};
    const a=buildFullMasterModel(scene),b=buildFullMasterModel(hand),scale=Math.max(...a.drives.flatMap(d=>[...d.cos.real,...d.cos.imag].map(Math.abs)));
    assert.equal(a.drives.length,b.drives.length);
    a.drives.forEach((drive,j)=>{for(const part of ['cos','sin'])for(const re of ['real','imag'])drive[part][re].forEach((v,k)=>close(v,b.drives[j][part][re][k],1e-12*scale,`${quarterWave?'λ/4':'plain'} drive ${j} ${part}.${re}[${k}]`));});
    for(const key of ['hReal','hImag'])a[key].forEach((v,k)=>close(v,b[key][k],1e-12*Math.max(1,Math.abs(v)),`${key}[${k}]`));
    assert.deepEqual(a.collapse,b.collapse);
  }
});
