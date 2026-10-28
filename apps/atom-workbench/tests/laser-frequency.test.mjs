import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultScene } from '../dist/js/atom-workbench/src/types.js';
import { analyze, atomicSystem, couplingBetweenStates, laserCarrierOffsetMHz, laserFrequencyDifferenceMHz, laserFrequencyMHz,
  laserIntensityMWcm2, RB87, resolveSceneFrequencies, transitionReferenceMHz, zeroFieldTransitionFrequencyMHz } from '../dist/js/atom-workbench/src/physics.js';
import { frequencyValue, setFrequencyBase, setFrequencyValue, zeroFieldTransitionOptions } from '../dist/js/atom-workbench/src/laser-frequency.js';
import { validateScene } from '../dist/js/atom-workbench/src/scene-file.js';
import { buildFullMasterModel } from '../dist/js/atom-workbench/src/full-master-model.js';

const near=(actual,expected,tol=1e-8)=>assert.ok(Math.abs(actual-expected)<=tol,`${actual} != ${expected} (tol ${tol})`);
/** The default scene, resolved: the probe (D2 F=2 → F′=2, Δ = 0) and the 852 nm tweezer. */
const scene=()=>{const s=defaultScene();resolveSceneFrequencies(s);return s;};
const roundtrip=s=>validateScene(JSON.parse(JSON.stringify(s)));

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

test('a base change keeps the actual frequency and the offset unit',()=>{
  const s=scene(),beam=s.lasers[0];setFrequencyValue(s,beam,13.781234,'MHz');
  const initial=laserFrequencyMHz(beam),carrier=laserCarrierOffsetMHz(beam);
  for(const ref of zeroFieldTransitionOptions('D2')){setFrequencyBase(s,beam,{kind:'transition',...ref});near(laserCarrierOffsetMHz(beam),carrier,2e-12);near(laserFrequencyMHz(beam),initial,1e-7);assert.equal(beam.frequencyUnit,'MHz');}
  for(const ref of zeroFieldTransitionOptions()){setFrequencyBase(s,beam,{kind:'transition',...ref});near(laserFrequencyMHz(beam),initial,1e-7);}
});

test('a detuning has no per-unit range: any finite value with a positive optical frequency, Hz precision kept',()=>{
  const s=scene(),beam=s.lasers[0];
  for(const unit of ['Hz','kHz','MHz','GHz'])for(const value of [-5000,5000,-12345.678,98765.4321]){setFrequencyValue(s,beam,value,unit);near(frequencyValue(beam).value,value,Math.abs(value)*1e-15);assert.equal(beam.frequencyUnit,unit);}
  setFrequencyValue(s,beam,.125,'Hz');assert.equal(beam.frequency.offsetMHz,.125e-6);
  const before=JSON.stringify(beam);
  assert.throws(()=>setFrequencyValue(s,beam,-5000,'THz'),/光频必须是有限正数/);assert.equal(JSON.stringify(beam),before,'a refused value changes nothing');
  assert.throws(()=>setFrequencyValue(s,beam,5001,'THz'),/±5000 THz/,'the scene file bound, not a unit range');
  assert.throws(()=>setFrequencyValue(s,beam,NaN,'THz'),/有限/);
  setFrequencyValue(s,beam,1.5,'THz');near(beam.frequency.offsetMHz,1.5e6,1e-6);
});

test('852 nm tweezer is present, enabled and physically specified as an ordinary power-mode laser',()=>{
  const s=scene(),tweezer=s.lasers.find(l=>l.id==='tweezer');
  assert.ok(tweezer?.enabled);assert.deepEqual(tweezer.frequency,{wavelengthNm:852});assert.equal(tweezer.intensityMode,'power');
  near(laserFrequencyMHz(tweezer),299792458000/852,1e-7);
  near(laserIntensityMWcm2(tweezer),2*.004/(Math.PI*(1.2e-6)**2)/10,1e-7);
  assert.ok(laserCarrierOffsetMHz(tweezer,'D2')< -31e6);
  assert.ok(laserCarrierOffsetMHz(tweezer,'D1')< -24e6);
  tweezer.intensityMWcm2=0;assert.ok(laserIntensityMWcm2(tweezer)>1e8,'the intensity field does not override the power-mode inputs');
});

test('absolute and transition bases round-trip the optical frequency, cross-line included',()=>{
  for(const index of [0,1]){
    const s=scene(),beam=s.lasers[index],frequency=laserFrequencyMHz(beam);
    for(let i=0;i<2;i++){setFrequencyBase(s,beam,{kind:'absolute'});near(laserFrequencyMHz(beam),frequency,1e-7);setFrequencyBase(s,beam,{kind:'transition',line:'D1',Fg:1,Fe:2});near(laserFrequencyMHz(beam),frequency,1e-7);}
  }
});

test('spectroscopy and pair readout follow the frequency without stale detuning or intensity caches',()=>{
  const s=scene();s.fields[0].magnitudeG=0;
  const beam=s.lasers[0];setFrequencyValue(s,beam,17.25,'MHz');
  let analysis=analyze(s),transition=analysis.transitions.find(t=>t.laserId===beam.id&&t.line==='D2'&&t.Fe===2&&t.me===1);
  near(transition.detuningMHz,17.25,1e-9);near(transition.offsetMHz,transitionReferenceMHz('D2',2,2)-transitionReferenceMHz('D2'),1e-9);
  const pair=couplingBetweenStates(transition.groundId,transition.excitedId,beam,s.fields);
  near(pair.detuningMHz,transition.detuningMHz,1e-9);near(pair.rabiMHz,transition.rabiMHz,1e-12);
  const originalDetuning=transition.detuningMHz;
  setFrequencyBase(s,beam,{kind:'transition',line:'D2',Fg:1,Fe:1});analysis=analyze(s);
  transition=analysis.transitions.find(t=>t.laserId===beam.id&&t.line==='D2'&&t.Fe===2&&t.me===1);
  near(transition.detuningMHz,originalDetuning,1e-9);
  const tweezer=s.lasers[1],a=analysis.transitions.find(t=>t.laserId===tweezer.id);
  const tweezerPair=couplingBetweenStates(a.groundId,a.excitedId,tweezer,s.fields);
  near(tweezerPair.detuningMHz,a.detuningMHz,1e-8);near(tweezerPair.rabiMHz,a.rabiMHz,1e-8);
  const rabi=a.rabiMHz;tweezer.powerMW*=4;
  near(analyze(s).transitions.find(t=>t.id===a.id).rabiMHz,2*rabi,1e-8);
});

test('every beam acts on both D manifolds and a base change cannot alter physical couplings',()=>{
  const s=scene(),before=analyze(s);
  for(const beam of s.lasers)for(const line of ['D1','D2']) {
    const entries=before.transitions.filter(t=>t.laserId===beam.id&&t.line===line);
    assert.ok(entries.length>0,`${beam.name} is missing ${line}`);
    assert.ok(entries.some(t=>t.rabiMHz>0));
    for(const t of entries) {
      const pair=couplingBetweenStates(t.groundId,t.excitedId,beam,s.fields);
      near(t.detuningMHz,pair.detuningMHz,1e-8);near(t.rabiMHz,pair.rabiMHz,1e-8);
    }
    setFrequencyBase(s,beam,{kind:'transition',line:'D1',Fg:1,Fe:2});
    const after=analyze(s);
    for(const prior of before.transitions.filter(t=>t.laserId===beam.id)) {
      const next=after.transitions.find(t=>t.id===prior.id);
      assert.ok(next);near(next.detuningMHz,prior.detuningMHz,1e-7);
      assert.equal(next.rabiMHz,prior.rabiMHz);assert.equal(next.strength,prior.strength);
    }
  }
});

test('100 cross-line base changes and scene-file round trips keep the carrier bit for bit and create no false beat',()=>{
  for(const index of [0,1]){
    let s=scene();const original=s.lasers[index];s.lasers=[original,{...structuredClone(original),id:'converted',phaseDeg:180}];resolveSceneFrequencies(s);
    const offset=laserCarrierOffsetMHz(original,'D2'),absolute=laserFrequencyMHz(original);
    for(let i=0;i<100;i++){
      const converted=s.lasers[1],line=i%2?'D2':'D1';
      setFrequencyBase(s,converted,{kind:'transition',line,Fg:i%2?1:2,Fe:2});setFrequencyBase(s,converted,{kind:'absolute'});
      s=roundtrip(s);
      assert.equal(laserCarrierOffsetMHz(s.lasers[1],'D2'),offset);assert.equal(laserFrequencyMHz(s.lasers[1]),absolute);
      // Opposite-phase copies cancel exactly, co- and counter-rotating alike.
      const model=buildFullMasterModel(s);
      assert.equal(model.drives.length,0,`iteration ${i} generated a fictitious beat`);
      for(let e=8;e<32;e++)for(let g=0;g<8;g++){assert.equal(model.hReal[e*32+g],0);assert.equal(model.hImag[e*32+g],0);}
    }
  }
});

test('frequency inputs rebuild the carrier, and a real sub-Hz beat survives',()=>{
  const s=scene(),beam=s.lasers[0],original=laserCarrierOffsetMHz(beam,'D2');
  const edited={...structuredClone(beam),id:'real-beat'};s.lasers=[beam,edited];setFrequencyValue(s,edited,.125,'Hz');
  near(laserCarrierOffsetMHz(edited,'D2')-original,.125e-6,3e-13);
  const model=buildFullMasterModel(s),beats=model.drives.filter((_,j)=>model.frequencyCoordinates.driveIndices[j].at(-1)===0);
  assert.equal(beats.length,1);assert.ok(beats[0].omega>0);
  near(beats[0].omega/(2*Math.PI),.125e-6,3e-13);
  setFrequencyBase(s,edited,{kind:'absolute'});const prior=laserFrequencyMHz(edited);
  setFrequencyValue(s,edited,850,'nm');near(laserFrequencyMHz(edited),299792458000/850,1e-7);assert.notEqual(laserFrequencyMHz(edited),prior);
  setFrequencyBase(s,edited,{kind:'transition',line:'D2',Fg:2,Fe:2});setFrequencyValue(s,edited,-12,'THz');
  near(laserCarrierOffsetMHz(edited,'D2')-transitionReferenceMHz('D2',2,2),-12e6,1e-6);
  setFrequencyBase(s,edited,{kind:'absolute'});setFrequencyValue(s,edited,800,'nm');const snapshot=JSON.stringify(edited);
  assert.throws(()=>setFrequencyValue(s,edited,0,'nm'),/正/);assert.equal(JSON.stringify(edited),snapshot);
});

test('D1 and D2 sub-Hz beats retain local-offset precision before optical-frame conversion',()=>{
  for(const line of ['D1','D2']) {
    const s=scene(),first=s.lasers[0];first.frequency={base:{kind:'transition',line,Fg:2,Fe:2},offsetMHz:0};
    const second={...structuredClone(first),id:'second'};second.frequency={base:{kind:'transition',line,Fg:2,Fe:2},offsetMHz:.125e-6};
    s.lasers=[first,second];resolveSceneFrequencies(s);
    near(laserFrequencyDifferenceMHz(second,first),.125e-6,3e-13);
    assert.equal(laserFrequencyDifferenceMHz(first,first),0);
  }
});
