import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {laserIntensityMWcm2} from '../dist/js/atom-workbench/src/physics.js';
import {displayMixing,spectroscopyAnalysis} from '../dist/js/atom-workbench/src/spectroscopy.js';

// The hint beside 显示光移 is the display model's own small parameter, not a second one.
test('the mixing hint is the ratio the levels-chart model refuses above 1%',()=>{
  const scene=defaultScene(),probe=scene.lasers.find(l=>l.id==='probe'),tweezer=scene.lasers.find(l=>l.id==='tweezer');
  assert.equal(displayMixing(scene,probe.id).withinGap,false,'a resonant probe is too close');
  const base=displayMixing(scene,tweezer.id),intensity=laserIntensityMWcm2(tweezer);
  assert.ok(base.withinGap&&base.ratio>0&&base.ratio<.01);
  const at=ratio=>{const s=structuredClone(scene),t=s.lasers.find(l=>l.id==='tweezer');t.intensityMode='intensity';t.intensityMWcm2=intensity*(ratio/base.ratio)**2;return s;};
  for(const [ratio,status] of [[.008,'dispersive'],[.012,'unresolved']]){
    const s=at(ratio);assert.ok(Math.abs(displayMixing(s,'tweezer').ratio-ratio)<1e-9*ratio);
    assert.equal(spectroscopyAnalysis({...s,lasers:s.lasers.filter(l=>l.id==='tweezer')}).spectroscopy.status,status);
  }
  assert.deepEqual(displayMixing(scene,'nobody'),{ratio:0,withinGap:true});
});
