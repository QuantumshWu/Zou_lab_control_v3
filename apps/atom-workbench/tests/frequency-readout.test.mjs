import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultScene } from '../dist/js/atom-workbench/src/types.js';
import { RB87, analyze, laserFrequencyMHz, zeroFieldTransitionFrequencyMHz } from '../dist/js/atom-workbench/src/physics.js';
import { spectroscopyAnalysis } from '../dist/js/atom-workbench/src/spectroscopy.js';
import { setLaserDetuningUnit, setLaserDetuningValue, setLaserTransitionReference,
  zeroFieldTransitionOptions } from '../dist/js/atom-workbench/src/laser-frequency.js';
import { C_NM_MHZ, baseKey, baseValue, difference, gap, inUnit, laserValue, levelValue, parseBase, photon, readingHeading,
  readoutText, stateValue, total, transitionKey, transitionValue, value, zeroFieldStateValue } from '../dist/js/atom-workbench/src/frequency-value.js';
import { frequencyBaseTree } from '../dist/js/atom-workbench/src/choice-trees.js';
import { laserEmits, levelReadoutMHz, referenceReadoutMHz, setLaserReferenceState } from '../dist/js/atom-workbench/src/transition-view.js';

const near=(actual,expected,tol)=>assert.ok(Math.abs(actual-expected)<=tol,`${actual} != ${expected} (tol ${tol})`);
const leaves=groups=>groups.flatMap(g=>[...g.leaves,...leaves(g.children??[])]);

test('a frequency difference subtracts the carriers first and keeps Hz on an optical carrier',()=>{
  const laser=structuredClone(defaultScene().lasers[0]);
  setLaserTransitionReference(laser,{line:'D2',Fg:2,Fe:3});setLaserDetuningUnit(laser,'Hz');setLaserDetuningValue(laser,1);
  // 1 Hz = 1e-6 MHz on ~384 THz: the plain difference of totals only keeps ~6e-8 MHz.
  near(difference(laserValue(laser),transitionValue({line:'D2',Fg:2,Fe:3})),1e-6,1e-12);
  near(total(laserValue(laser)),laserFrequencyMHz(laser),1e-6);
});

test('every zero-field transition is a carrier plus a small offset with its full frequency',()=>{
  for(const t of zeroFieldTransitionOptions()){
    const v=transitionValue(t);
    near(total(v),zeroFieldTransitionFrequencyMHz(t.line,t.Fg,t.Fe,t.fromManifold),1e-6);
    assert.ok(Math.abs(v.offsetMHz)<1e5,`${t.label}: offset ${v.offsetMHz}`);
  }
});

test('bases round-trip through their keys, and every offered base parses',()=>{
  const scene=defaultScene();
  for(const t of zeroFieldTransitionOptions()){const base={kind:'transition',...t};delete base.label;assert.deepEqual(parseBase(baseKey(base)),base);assert.equal(transitionKey(t),baseKey(base));}
  assert.deepEqual(parseBase('absolute'),{kind:'absolute'});
  assert.deepEqual(parseBase(`laser:${scene.lasers[0].id}`),{kind:'laser',id:scene.lasers[0].id});
  assert.equal(parseBase('transition:ground-D3-2-3'),undefined);
  const offered=leaves(frequencyBaseTree({absolute:'光频',lasers:scene.lasers})).map(leaf=>leaf.key);
  for(const key of offered)assert.ok(parseBase(key)&&baseValue(parseBase(key),scene),key);
  assert.equal(offered.filter(key=>key.startsWith('transition:')).length,zeroFieldTransitionOptions().length);
  assert.ok(offered.includes(`laser:${scene.lasers[0].id}`)&&offered.includes('absolute'));
  assert.equal(baseValue({kind:'laser',id:'gone'},scene),undefined);
  assert.equal(readingHeading({kind:'laser',id:scene.lasers[0].id},scene,'MHz'),`ν − ν(${scene.lasers[0].name}) / MHz`);
  assert.equal(readingHeading({kind:'absolute'},scene,'THz'),'ν / THz');
});

// Digit groups are U+202F (narrow no-break space).
test('a readout spells twelve significant digits in groups of three, ≈ when it had more',()=>{
  assert.equal(readoutText(6834.6826109,true),'+6834.682\u202F610\u202F9');
  assert.equal(readoutText(384230484.468562123,false),'≈ 384\u202F230\u202F484.469');
  assert.equal(readoutText(-12.25,true),'−12.25');
  assert.equal(readoutText(Number.NaN,false),'—');
  near(inUnit(C_NM_MHZ/780.241,'nm'),780.241,1e-9);
  near(inUnit(C_NM_MHZ/780.241,'μm'),0.780241,1e-12);
  assert.ok(Number.isNaN(inUnit(-1,'nm')),'a relative reading has no wavelength');
  near(inUnit(6834.68,'GHz'),6.83468,1e-12);
});

// ui-design.md §1.8: the two-state readout's "自动" is the same two states with no field and no
// light, so Δν = ν₁₂ − ν₁₂(零场) is what B and the lasers shown (显示光移) move the transition by.
test('"自动" reads a pair against the same two states at zero field',()=>{
  const scene=defaultScene(),a=spectroscopyAnalysis(scene,undefined,false),levels=[...a.ground,...a.D1,...a.D2],get=id=>levels.find(s=>s.id===id);
  assert.equal(a.spectroscopy.status,'dispersive','3 G and the far tweezer shift the levels');
  const pair=(read,from,to)=>photon(gap(read(from,levels,scene),read(to,levels,scene)));
  const shift=(from,to)=>difference(pair(stateValue,from,to),pair(zeroFieldStateValue,from,to));
  // Two ground states of different F: the zero-field ν₁₂ is the hyperfine splitting 2A.
  near(total(pair(zeroFieldStateValue,'ground-1-0','ground-2-1')),2*RB87.groundA,1e-9);
  near(shift('ground-1-0','ground-2-1'),get('ground-2-1').totalShiftMHz-get('ground-1-0').totalShiftMHz,1e-6);
  assert.ok(Math.abs(shift('ground-1-0','ground-2-1'))>1,'the Zeeman and light shifts are read, not hidden');
  // Ground to excited: the zero-field D2 hyperfine line; swapping the endpoints reads the same photon.
  near(difference(pair(zeroFieldStateValue,'ground-2-0','D2-3-1'),transitionValue({line:'D2',Fg:2,Fe:3})),0,1e-6);
  near(shift('D2-3-1','ground-2-0'),shift('ground-2-0','D2-3-1'),1e-9);
  near(shift('ground-2-0','D2-3-1'),get('D2-3-1').totalShiftMHz-get('ground-2-0').totalShiftMHz,1e-6);
  // A reference level keeps its photon and δ; only its anchor goes to zero field.
  scene.references=[{id:'virtual',name:'虚能级',anchorId:'ground-2-0',anchorMode:'follow',offsetMHz:-12,photon:{source:{kind:'laser',id:'probe'},direction:'absorb'}}];
  near(total(pair(zeroFieldStateValue,'ground-2-0','virtual')),laserFrequencyMHz(scene.lasers[0])-12,1e-6);
  near(shift('ground-2-0','virtual'),0,1e-6);
  near(shift('ground-1-0','virtual'),get('ground-2-0').totalShiftMHz-get('ground-1-0').totalShiftMHz,1e-6);
  scene.references[0].anchorMode='zero-field';
  near(shift('ground-2-0','virtual'),-get('ground-2-0').totalShiftMHz,1e-6);
});

test('a transition read from its upper state is the same photon',()=>{
  const ground=value(0,-4271.7),excited=value(384230484.4685,193.7);
  assert.deepEqual(photon(gap(excited,ground)),gap(ground,excited));
});


test('the photon direction belongs to a reference level and is cleared with it',()=>{
  const scene=defaultScene(),laser=scene.lasers[0],states=[{id:'ground-2-0',manifold:'ground'},{id:'D2-3-0',manifold:'D2'}];
  scene.references=[{id:'ref-a',name:'A',anchorId:'D2-3-0',offsetMHz:0}];
  setLaserReferenceState(scene,laser,'ref-a');laser.referenceProcess='emission';assert.equal(laserEmits(scene,laser,states),true);
  setLaserReferenceState(scene,laser,'ref-a');assert.equal(laser.referenceProcess,'emission','re-picking the same reference keeps its direction');
  // Back to a real ground state: the stale emission flag goes, and absorption is read.
  setLaserReferenceState(scene,laser,'ground-2-0');assert.equal(laser.referenceProcess,undefined);assert.equal(laserEmits(scene,laser,states),false);
  setLaserReferenceState(scene,laser,'D2-3-0');assert.equal(laserEmits(scene,laser,states),true,'a real excited origin emits');
  laser.referenceProcess='emission';setLaserReferenceState(scene,laser,undefined);assert.equal(laser.referenceStateId,undefined);assert.equal(laser.referenceProcess,undefined);
});

// ui-design.md §1.5: with a laser selected a card says how far its state lies above its
// anchor (the reference state on its side, the photon's end on the other); +x MHz is higher.
test('a state card reads its transition frequency minus the laser frequency: + is higher',()=>{
  const scene=defaultScene(),laser=scene.lasers[0],a=analyze(scene),get=id=>[...a.ground,...a.D1,...a.D2].find(s=>s.id===id);
  const ref=get('ground-2-0'),lower=get('ground-1-0'),excited=get('D2-3-0');
  near(levelReadoutMHz(ref,lower,laser),lower.energyMHz-ref.energyMHz,1e-9);
  assert.ok(levelReadoutMHz(ref,lower,laser)<-6800,'F=1 below the F=2 reference reads about −6.83 GHz');
  // The excited card is the two-state readout relative to the laser: ν(参考态→e) − ν_L.
  const card=levelReadoutMHz(ref,excited,laser),readout=difference(photon(gap(levelValue(ref),levelValue(excited))),laserValue(laser));
  near(card,readout,1e-6);
  // Stimulated emission from the excited state: the ground card is its height above the photon's end.
  near(levelReadoutMHz(excited,ref,laser),-levelReadoutMHz(ref,excited,laser),1e-9);
  near(levelReadoutMHz(excited,get('D2-2-0'),laser),get('D2-2-0').energyMHz-excited.energyMHz,1e-9);
  assert.equal(levelReadoutMHz(excited,get('D1-2-0'),laser),undefined);
});

test('a reference level is read by the rule of the manifold its energy lies in',()=>{
  const absorbing={sourceEnergyMHz:-4000,sourceManifold:'ground',emission:false,photonEnergyMHz:384230000};
  assert.equal(referenceReadoutMHz(absorbing,-4100,'ground'),-100,'ground side: (E − E_ref)/h');
  assert.equal(referenceReadoutMHz(absorbing,384230050,'D2'),50,'excited side: the level above the photon end');
  const emitting={sourceEnergyMHz:384230000,sourceManifold:'D2',emission:true,photonEnergyMHz:-4000};
  assert.equal(referenceReadoutMHz(emitting,-3990,'ground'),10,'ground side: the level above the photon end');
  assert.equal(referenceReadoutMHz(emitting,384229900,'D2'),-100,'own side: (E − E_ref)/h');
  assert.equal(referenceReadoutMHz(emitting,377107000,'D1'),undefined);
  assert.equal(referenceReadoutMHz({...absorbing,sourceEnergyMHz:undefined},0,'ground'),undefined);
});
