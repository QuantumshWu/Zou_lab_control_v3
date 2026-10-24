import test from 'node:test';
import assert from 'node:assert/strict';
import { analyze, RB87, polarization, direction, clockFrequencyMHz, sinCosDegrees } from '../dist/js/atom-workbench/src/physics.js';
import { defaultScene } from '../dist/js/atom-workbench/src/types.js';

const near=(actual,expected,tol=1e-8)=>assert.ok(Math.abs(actual-expected)<=tol,`${actual} differs from ${expected} (tolerance ${tol})`);
function zeroScene() {
  const s=defaultScene();s.fields[0].magnitudeG=0;s.lasers[0].thetaDeg=0;
  // Isolate one beam for these spectroscopy sum-rule tests; the product's
  // default scene also includes an enabled far-detuned tweezer.
  s.lasers=s.lasers.filter(l=>l.id==='probe');delete s.lasers[0].referenceTransition;
  s.initialF=2;s.initialM=2;s.dynamics.initialStateId='ground-2-2';s.lasers[0].ellipticityDeg=45;
  return s;
}

test('exact opposite entered field directions cancel without erasing real tiny deviations',()=>{
  for(const [theta,phi]of [[37,25],[61,-17],[145,173]]){
    const scene=zeroScene();Object.assign(scene.fields[0],{magnitudeG:3,thetaDeg:theta,phiDeg:phi});
    scene.fields.push({...scene.fields[0],id:'opposite',thetaDeg:180-theta,phiDeg:phi>0?phi-180:phi+180});
    const exact=analyze(scene);assert.equal(exact.fieldMagnitudeG,0);assert.deepEqual(exact.fieldAxis,{thetaDeg:0,phiDeg:0});
    scene.fields[1].thetaDeg+=1e-8;assert.ok(analyze(scene).fieldMagnitudeG>0);
  }
  const tiny=sinCosDegrees(1e-20);assert.ok(tiny.sin>0,'a represented nonzero angle is not rounded to a cardinal direction');
  for(let angle=-720;angle<=720;angle+=7){const actual=sinCosDegrees(angle);near(actual.sin,Math.sin(angle*Math.PI/180),3e-15);near(actual.cos,Math.cos(angle*Math.PI/180),3e-15);}
});

test('zero-field degeneracies, D1 and D2 hyperfine intervals, centre of gravity',()=>{
  const a=analyze(zeroScene());
  assert.equal(a.ground.length,8);assert.equal(a.D1.length,8);assert.equal(a.D2.length,16);
  near(a.clockMHz,6834.68261090429,1e-9);near(a.clockShiftKHz,0,1e-7);
  for(const states of [a.ground,a.D1,a.D2]) {
    near(states.reduce((sum,s)=>sum+s.energyMHz,0),0,1e-8);
    for(const level of states)near(level.shiftMHz,0,1e-9);
  }
  const e=(line,F)=>a[line].find(s=>s.F===F).energyMHz;
  near(e('D1',2)-e('D1',1),814.5);
  // Measured intervals (MHz); the A/B-only fit differs by a few kHz from individual
  // intervals. These tolerances do not imply a precision optical-frequency model.
  near(e('D2',3)-e('D2',2),266.6500,0.01);
  near(e('D2',2)-e('D2',1),156.9470,0.01);
  near(e('D2',1)-e('D2',0),72.2180,0.01);
});

test('clock shift and exact eigenvalue gap match Breit–Rabi',()=>{
  for(const b of [0.01,1,3,30,200,1000]) {
    const s=zeroScene();s.fields[0].magnitudeG=b;const a=analyze(s);
    const gap=a.ground.find(v=>v.F===2&&v.m===0).energyMHz-a.ground.find(v=>v.F===1&&v.m===0).energyMHz;
    near(gap,clockFrequencyMHz(b),2e-9);
  }
  const s=zeroScene();s.fields[0].magnitudeG=1;const a=analyze(s);
  near(a.clockShiftKHz*1000,575.15,0.01);
  near(a.clockSensitivityHzPerMG,1.1503,0.00003);
});

test('field rotation preserves energies; disabled fields do not contribute',()=>{
  const s=defaultScene(), a=analyze(s);
  s.fields[0].thetaDeg=131;s.fields[0].phiDeg=67;
  s.fields.push({id:'off',name:'off',enabled:false,magnitudeG:1000,thetaDeg:14,phiDeg:97});
  const b=analyze(s);near(b.fieldMagnitudeG,3);
  a.D2.forEach((v,i)=>near(v.energyMHz,b.D2[i].energyMHz));
  s.fields[0].enabled=false;near(analyze(s).fieldMagnitudeG,0);
});

test('Jones helicity, transverse geometry and polarization conservation',()=>{
  const s=zeroScene(),l=s.lasers[0];
  let p=polarization(l,[0,0,1]);near(p.plus,1);near(p.pi,0);near(p.minus,0);
  l.ellipticityDeg=-45;p=polarization(l,[0,0,1]);near(p.minus,1);
  l.ellipticityDeg=0;p=polarization(l,[0,0,1]);near(p.plus,0.5);near(p.minus,0.5);
  p=polarization(l,[1,0,0]);near(p.pi,1);
  for(let i=0;i<30;i++) {
    l.thetaDeg=i*6;l.phiDeg=i*13;l.polarizationAngleDeg=i*7;l.ellipticityDeg=i*3-45;
    p=polarization(l,direction(73,11));near(p.plus+p.pi+p.minus,1,1e-12);
    near(p.k.reduce((sum,v,i)=>sum+v*p.u[i],0),0,1e-12);
    near(p.k.reduce((sum,v,i)=>sum+v*p.v[i],0),0,1e-12);
  }
  // Common rotation about lab z rotates the beam's e_theta/e_phi and B together.
  const before=polarization(l,direction(73,11));l.phiDeg+=42;
  const after=polarization(l,direction(73,53));near(before.plus,after.plus);near(before.pi,after.pi);
});

test('cycling line strength, zero-field reference and Rabi convention',()=>{
  const s=zeroScene();s.lasers[0].intensityMWcm2=1.66933;s.lasers[0].detuningMHz=0;
  const a=analyze(s),t=a.transitions.find(t=>t.Fe===3&&t.me===3);
  near(t.strength,1);near(t.weight,1);near(t.offsetMHz,0,1e-9);
  near(t.rabiMHz,RB87.D2.gammaHz/(2*Math.PI)/1e6/Math.sqrt(2),0.001);
  s.lasers[0].intensityMWcm2*=4;const t4=analyze(s).transitions.find(t=>t.Fe===3&&t.me===3);
  near(t4.rabiMHz/t.rabiMHz,2);
});

test('complete excited manifolds satisfy dipole sum rules, including mixed states',()=>{
  for(const B of [0,3,250]) for(const F of [1,2]) for(let m=-F;m<=F;m++) {
    const s=zeroScene();s.fields[0].magnitudeG=B;s.initialF=F;s.initialM=m;s.dynamics.initialStateId=`ground-${F}-${m}`;
    const transitions=analyze(s).transitions;
    const d2=transitions.filter(t=>t.line==='D2').reduce((sum,t)=>sum+t.strength,0);
    near(d2,2,2e-10);
    const d1=transitions.filter(t=>t.line==='D1').reduce((sum,t)=>sum+t.strength,0);
    near(d1,(RB87.D1.dipole/RB87.D2.dipole)**2,2e-10);
  }
});

test('disabled light has no transitions; zero field uses a deterministic +z axis',()=>{
  const s=zeroScene();const a=analyze(s);near(a.polarizations.probe.plus,1);
  s.lasers[0].enabled=false;assert.equal(analyze(s).transitions.length,0);
});
