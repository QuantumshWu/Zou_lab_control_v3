import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {OUTSIDE_GAP_MHZ,STATIC_MIXING_LIMIT,modelScope,spectroscopyAnalysis} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';

const C=299792458;
/** A beam at a vacuum wavelength (nm), local intensity in mW/cm². */
function beam(id,wavelengthNm,intensityMWcm2=1,extra={}){
  return {id,name:id,enabled:true,showLightShift:false,line:'D2',frequency:{wavelengthNm},frequencyUnit:'nm',
    intensityMode:'intensity',intensityMWcm2,powerUnit:'mW',waistUnit:'μm',intensityUnit:'mW/cm²',direction:{reference:{kind:'axis',axis:'+x'},thetaDeg:0,phiDeg:0},
    polarizationAngleDeg:0,ellipticityDeg:0,phaseDeg:0,...extra};
}
/** A beam `offsetGHz` above an outside-state line at `lineTHz` (spectroscopy-data.ts: 5P₃/₂ → 4D₅/₂ at 196.0239 THz). */
const nearLine=(id,lineTHz,offsetGHz,intensity=1)=>beam(id,C/((lineTHz+offsetGHz*1e-3)*1e12)*1e9,intensity);
const sceneWith=(...lasers)=>{const s=defaultScene();s.lasers.push(...lasers);return s;};

test('the default scene is within the model: no scope note',()=>{
  assert.deepEqual(modelScope(defaultScene()),[]);
});

test('a laser that photoionizes 5P is named, with the thresholds it is below; the model is built and solved all the same',()=>{
  const s=sceneWith(beam('蓝光',450));
  const notes=modelScope(s);
  assert.equal(notes.length,1);assert.equal(notes[0].kind,'photoionization');assert.deepEqual(notes[0].laserIds,['蓝光']);
  assert.match(notes[0].text,/「蓝光」的波长 450 nm 短于 5P₃\/₂ 和 5P₁\/₂ 的光电离阈值（479\.1 nm、473\.7 nm）：光电离不在模型内/);
  assert.doesNotThrow(()=>buildFullMasterModel(s,s.dynamics),'a warning, not a refusal');
  // Between the two thresholds only 5P₃/₂ ionizes; above both, nothing.
  assert.match(modelScope(sceneWith(beam('b',476)))[0].text,/短于 5P₃\/₂ 的光电离阈值 479\.1 nm/);
  assert.deepEqual(modelScope(sceneWith(beam('c',480))),[]);
  // A laser that is off, or dark, does nothing.
  assert.deepEqual(modelScope(sceneWith(beam('d',450,1,{enabled:false}))),[]);
  assert.deepEqual(modelScope(sceneWith(beam('e',450,0))),[]);
});

test('a laser near a resonance of a state outside the model is named with it; the dynamics keeps it, the levels chart draws no static shift',()=>{
  // 5 GHz from 5P₃/₂ → 4D₅/₂: within OUTSIDE_GAP_MHZ, so the second-order shift does not represent the state.
  const close=sceneWith(nearLine('telecom',196.0239,5));
  const notes=modelScope(close);
  assert.equal(notes.length,1);assert.equal(notes[0].kind,'outside-resonance');assert.deepEqual(notes[0].laserIds,['telecom']);
  assert.match(notes[0].text,/「telecom」离 4D5\/2 的虚跃迁失谐只有 \S+ GHz（不足 10 GHz）：32 态以外的能级（6P、nS、nD）只作为二阶静态光移进入模型，这里超出它的适用范围/);
  assert.equal(OUTSIDE_GAP_MHZ,1e4);
  assert.doesNotThrow(()=>buildFullMasterModel(close,close.dynamics),'the release refused this model; now it is built and the result shown');
  // The levels chart's rule for a static shift is the same one: shown there, it is not drawn.
  const shown=structuredClone(close);shown.lasers.at(-1).showLightShift=true;
  assert.equal(spectroscopyAnalysis(shown).spectroscopy.status,'unresolved');
  // 60 GHz away it is a static shift again, until the mixing passes 1%.
  const far=sceneWith(nearLine('telecom',196.0239,60));assert.deepEqual(modelScope(far),[]);
  const strong=sceneWith(nearLine('telecom',196.0239,60,1e5));
  const strongNotes=modelScope(strong);assert.equal(strongNotes.length,1);
  assert.match(strongNotes[0].text,/对 4D5\/2 的混合比 \|Ω\/2Δ\| 为 \S+%（超过 1%）/);assert.equal(STATIC_MIXING_LIMIT,.01);
  const strongShown=structuredClone(strong);strongShown.lasers.at(-1).showLightShift=true;
  assert.equal(spectroscopyAnalysis(strongShown).spectroscopy.status,'unresolved');
  assert.doesNotThrow(()=>buildFullMasterModel(strong,strong.dynamics));
});

test('a laser of both kinds is named once for each reason',()=>{
  // 5S → 6P₃/₂ at 713.2839 THz (420.3 nm): 5 GHz above it is near that resonance and below both 5P thresholds.
  const notes=modelScope(sceneWith(nearLine('blue',713.2839,5)));
  assert.deepEqual(notes.map(note=>note.kind),['outside-resonance','photoionization']);
  assert.ok(notes.every(note=>note.text.startsWith('「blue」')));
});
