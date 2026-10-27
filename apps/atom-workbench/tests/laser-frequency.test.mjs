import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultScene, defaultTweezer } from '../dist/js/atom-workbench/src/types.js';
import { analyze, atomicSystem, couplingBetweenStates, laserCarrierOffsetMHz, laserFrequencyDifferenceMHz, laserFrequencyMHz,
  laserIntensityMWcm2, RB87, transitionReferenceMHz, zeroFieldTransitionFrequencyMHz } from '../dist/js/atom-workbench/src/physics.js';
import { DETUNING_UNITS, detuningInUnit, laserDetuningMHz, naturalDetuningUnit, normalizeLaserFrequency,
  setLaserDetuningUnit, setLaserDetuningValue, setLaserFrequencyMode, setLaserTransitionReference,
  setLaserWavelengthNm, zeroFieldTransitionOptions } from '../dist/js/atom-workbench/src/laser-frequency.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';

const near=(actual,expected,tol=1e-8)=>assert.ok(Math.abs(actual-expected)<=tol,`${actual} != ${expected} (tol ${tol})`);
const laser=()=>({...defaultScene().lasers[0]});

test('all electronic-manifold energy references include E1-forbidden pairs match independent field-Hamiltonian eigenvalues',()=>{
  const references=zeroFieldTransitionOptions();
  assert.equal(references.length,20);
  assert.equal(zeroFieldTransitionOptions('D1').length,4);
  assert.equal(zeroFieldTransitionOptions('D2').length,16);
  for(const ref of references) {
    const system=atomicSystem(ref.line,0);
    const ground=(ref.fromManifold==='D1'?atomicSystem('D1',0).excited:system.ground).find(s=>s.F===ref.Fg),excited=system.excited.find(s=>s.F===ref.Fe);
    near(transitionReferenceMHz(ref.line,ref.Fg,ref.Fe,ref.fromManifold),excited.energyMHz-ground.energyMHz-(ref.fromManifold==='D1'?RB87.D1.frequencyHz/1e6:0),1e-7);
    near(zeroFieldTransitionFrequencyMHz(ref.line,ref.Fg,ref.Fe,ref.fromManifold),
      RB87[ref.line].frequencyHz/1e6+excited.energyMHz-ground.energyMHz-(ref.fromManifold==='D1'?RB87.D1.frequencyHz/1e6:0),1e-7);
  }
});

test('legacy scene migration preserves original absolute laser frequencies',()=>{
  for(const line of ['D1','D2'])for(const delta of [-8000,-266.65,0,13.713,8000]) {
    const beam=laser();delete beam.referenceTransition;delete beam.frequencyMode;delete beam.detuningUnit;
    Object.assign(beam,{line,detuningMHz:delta});
    const expected=RB87[line].frequencyHz/1e6+transitionReferenceMHz(line)+delta;
    normalizeLaserFrequency(beam);
    near(laserFrequencyMHz(beam),expected,1e-7);
    assert.deepEqual(beam.referenceTransition,{line,Fg:2,Fe:line==='D1'?2:3});
    assert.equal(beam.detuningUnit,naturalDetuningUnit(delta),'a legacy beam without a unit is read in its natural unit');
  }
});

test('reference and unit changes preserve actual frequency and keep the operator unit',()=>{
  const beam=laser();beam.detuningMHz=13.781234;
  const initial=laserFrequencyMHz(beam),carrier=laserCarrierOffsetMHz(beam);
  for(const ref of zeroFieldTransitionOptions('D2')) {
    setLaserTransitionReference(beam,ref);
    near(laserCarrierOffsetMHz(beam),carrier,2e-12);
    near(laserFrequencyMHz(beam),initial,1e-7);
    assert.equal(beam.detuningUnit,'MHz');
  }
  for(const ref of zeroFieldTransitionOptions()) {
    setLaserTransitionReference(beam,ref);
    near(laserFrequencyMHz(beam),initial,1e-7);
    for(const unit of DETUNING_UNITS) {
      const delta=beam.detuningMHz;
      setLaserDetuningUnit(beam,unit);
      assert.equal(beam.detuningMHz,delta);
      assert.equal(beam.detuningUnit,unit);
      near(laserFrequencyMHz(beam),initial,1e-7);
    }
  }
});

test('a detuning has no per-unit range: any finite value with a positive optical frequency, Hz precision kept',()=>{
  const beam=laser();
  for(const unit of ['Hz','kHz','MHz','GHz']) {
    beam.detuningMHz=0;setLaserDetuningUnit(beam,unit);
    for(const value of [-5000,5000,-12345.678,98765.4321]) {setLaserDetuningValue(beam,value);near(detuningInUnit(beam),value,Math.abs(value)*1e-15);assert.equal(beam.detuningUnit,unit);}
  }
  beam.detuningMHz=0;setLaserDetuningUnit(beam,'Hz');setLaserDetuningValue(beam,.125);
  assert.equal(beam.detuningMHz,.125e-6);
  setLaserDetuningUnit(beam,'THz');const before=structuredClone(beam);
  assert.throws(()=>setLaserDetuningValue(beam,-5000),/大于零/);
  assert.deepEqual(beam,before);
  assert.throws(()=>setLaserDetuningValue(beam,5001),/±5000 THz/,'the scene file bound, not a unit range');
  assert.throws(()=>setLaserDetuningValue(beam,NaN),/有限/);
  setLaserDetuningValue(beam,1.5);near(laserDetuningMHz(beam),1.5e6,1e-6);
});

test('852 nm tweezer is present, enabled and physically specified as ordinary power-mode laser',()=>{
  const scene=defaultScene(),tweezer=scene.lasers.find(l=>l.id==='tweezer');
  assert.ok(tweezer?.enabled);assert.equal(tweezer.frequencyMode,'wavelength');
  near(laserFrequencyMHz(tweezer),299792458000/852,1e-7);
  near(laserIntensityMWcm2(tweezer),2*.004/(Math.PI*(1.2e-6)**2)/10,1e-7);
  assert.ok(laserCarrierOffsetMHz(tweezer,'D2')< -31e6);
  assert.ok(laserCarrierOffsetMHz(tweezer,'D1')< -24e6);
  const before=laserFrequencyMHz(tweezer);normalizeLaserFrequency(tweezer);
  near(laserFrequencyMHz(tweezer),before,1e-7);
  near(detuningInUnit(tweezer),laserDetuningMHz(tweezer)/1e6);
  tweezer.intensityMWcm2=0;assert.ok(laserIntensityMWcm2(tweezer)>1e8,'stored compatibility value must not override power-mode intensity');
});

test('frequency-mode switching round trips optical frequency, including cross-line reference changes',()=>{
  for(const beam of [laser(),defaultTweezer()]) {
    const frequency=laserFrequencyMHz(beam);
    for(const mode of ['wavelength','transition','wavelength','transition']) {
      setLaserFrequencyMode(beam,mode);
      near(laserFrequencyMHz(beam),frequency,1e-7);
      setLaserTransitionReference(beam,{line:'D1',Fg:1,Fe:2});
      near(laserFrequencyMHz(beam),frequency,1e-7);
    }
  }
});

test('spectroscopy and pair readout follow selected reference and wavelength without stale detuning/intensity caches',()=>{
  const scene=defaultScene();scene.fields[0].magnitudeG=0;
  const beam=scene.lasers[0];beam.detuningMHz=17.25;
  let analysis=analyze(scene),transition=analysis.transitions.find(t=>t.laserId===beam.id&&t.line==='D2'&&t.Fe===2&&t.me===1);
  near(transition.detuningMHz,17.25,1e-9);near(transition.offsetMHz,0,1e-9);
  const pair=couplingBetweenStates(transition.groundId,transition.excitedId,beam,[0,0,0]);
  near(pair.detuningMHz,transition.detuningMHz,1e-9);near(pair.rabiMHz,transition.rabiMHz,1e-12);
  const originalDetuning=transition.detuningMHz;
  setLaserTransitionReference(beam,{line:'D2',Fg:1,Fe:1});analysis=analyze(scene);
  transition=analysis.transitions.find(t=>t.laserId===beam.id&&t.line==='D2'&&t.Fe===2&&t.me===1);
  near(transition.detuningMHz,originalDetuning,1e-9);
  const tweezer=scene.lasers[1],a=analysis.transitions.find(t=>t.laserId===tweezer.id);
  const tweezerPair=couplingBetweenStates(a.groundId,a.excitedId,tweezer,[0,0,0]);
  near(tweezerPair.detuningMHz,a.detuningMHz,1e-8);near(tweezerPair.rabiMHz,a.rabiMHz,1e-8);
  const rabi=a.rabiMHz;tweezer.powerMW*=4;
  near(analyze(scene).transitions.find(t=>t.id===a.id).rabiMHz,2*rabi,1e-8);
});

test('every beam acts on both D manifolds and changing frequency reference cannot alter physical couplings',()=>{
  const scene=defaultScene(),before=analyze(scene);
  for(const beam of scene.lasers)for(const line of ['D1','D2']) {
    const entries=before.transitions.filter(t=>t.laserId===beam.id&&t.line===line);
    assert.ok(entries.length>0,`${beam.name} is missing ${line}`);
    assert.ok(entries.some(t=>t.rabiMHz>0));
    for(const t of entries) {
      const pair=couplingBetweenStates(t.groundId,t.excitedId,beam,before.fieldVector);
      near(t.detuningMHz,pair.detuningMHz,1e-8);near(t.rabiMHz,pair.rabiMHz,1e-8);
    }
    setLaserTransitionReference(beam,{line:'D1',Fg:1,Fe:2});
    const after=analyze(scene);
    for(const prior of before.transitions.filter(t=>t.laserId===beam.id)) {
      const next=after.transitions.find(t=>t.id===prior.id);
      assert.ok(next);near(next.detuningMHz,prior.detuningMHz,1e-7);
      assert.equal(next.rabiMHz,prior.rabiMHz);assert.equal(next.strength,prior.strength);
    }
  }
});

test('100 cross-line reference, wavelength and serialization round trips preserve canonical frequency and create no false beat',()=>{
  for(const original of [laser(),defaultTweezer()]) {
    normalizeLaserFrequency(original);
    const offset=laserCarrierOffsetMHz(original,'D2'),absolute=laserFrequencyMHz(original);
    let converted=structuredClone(original);
    for(let i=0;i<100;i++) {
      const line=i%2?'D2':'D1';
      setLaserTransitionReference(converted,{line,Fg:i%2?1:2,Fe:2});
      setLaserFrequencyMode(converted,'wavelength');
      converted=JSON.parse(JSON.stringify(converted));normalizeLaserFrequency(converted);
      setLaserFrequencyMode(converted,'transition');
      assert.equal(laserCarrierOffsetMHz(converted,'D2'),offset);
      assert.equal(laserFrequencyMHz(converted),absolute);
      const scene=defaultScene();scene.lasers=[original,{...converted,id:'converted',phaseDeg:180}];
      const model=buildFullMasterModel(scene);
      // Opposite-phase copies cancel exactly, co- and counter-rotating alike.
      assert.equal(model.drives.length,0,`iteration ${i} generated a fictitious beat`);
      for(let e=8;e<32;e++)for(let g=0;g<8;g++) {
        assert.equal(model.hReal[e*32+g],0);assert.equal(model.hImag[e*32+g],0);
      }
    }
  }
});

test('explicit frequency inputs rebuild the anchor, direct legacy edits invalidate it, and a real sub-Hz beat survives',()=>{
  const beam=laser();normalizeLaserFrequency(beam);
  const original=laserCarrierOffsetMHz(beam,'D2');
  const edited=structuredClone(beam);edited.detuningMHz+=.125e-6;
  near(laserCarrierOffsetMHz(edited,'D2')-original,.125e-6,3e-13);
  const scene=defaultScene();scene.lasers=[beam,{...edited,id:'real-beat'}];
  const model=buildFullMasterModel(scene),beats=model.drives.filter((_,j)=>model.frequencyCoordinates.driveIndices[j].at(-1)===0);
  assert.equal(beats.length,1);assert.ok(beats[0].omega>0);
  near(beats[0].omega/(2*Math.PI),.125e-6,3e-13);
  setLaserFrequencyMode(edited,'wavelength');const prior=laserFrequencyMHz(edited);
  setLaserWavelengthNm(edited,850);near(laserFrequencyMHz(edited),299792458000/850,1e-7);
  assert.notEqual(laserFrequencyMHz(edited),prior);
  setLaserFrequencyMode(edited,'transition');setLaserDetuningUnit(edited,'THz');setLaserDetuningValue(edited,-12);
  near(laserDetuningMHz(edited),-12e6);
  const snapshot=structuredClone(edited);
  assert.throws(()=>setLaserWavelengthNm(edited,0),/正/);assert.deepEqual(edited,snapshot);
});

test('D1 and D2 sub-Hz beats retain local-offset precision before optical-frame conversion',()=>{
  for(const line of ['D1','D2']) {
    const first=laser();first.line=line;first.referenceTransition={line,Fg:2,Fe:2};
    normalizeLaserFrequency(first);
    const second=structuredClone(first);second.detuningMHz+=.125e-6;
    near(laserFrequencyDifferenceMHz(second,first),.125e-6,3e-13);
    assert.equal(laserFrequencyDifferenceMHz(first,first),0);
  }
});
