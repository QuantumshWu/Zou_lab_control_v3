import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene,defaultDynamics,defaultTweezer} from '../dist/js/atom-workbench/src/types.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {laserFrequencyMHz} from '../dist/js/atom-workbench/src/physics.js';
import {spectroscopyAnalysis} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {combinedLevelsChart} from '../dist/js/atom-workbench/src/combined-level-chart.js';

test('new/reset scenes use 1000 points and a physically specified 852 nm tweezer',()=>{
  assert.equal(defaultDynamics().samples,1000);
  assert.equal(defaultScene().dynamics.samples,1000);
  assert.equal(defaultTweezer().wavelengthNm,852);
  assert.ok(Math.abs(laserFrequencyMHz(defaultTweezer())-299792458000/852)<1e-7);
  assert.match(defaultScene().name,/852 nm/);
});

test('loading an existing scene preserves its sample count and 850 nm carrier',()=>{
  const old=defaultScene();old.dynamics.samples=161;old.lasers.find(x=>x.id==='tweezer').wavelengthNm=850;
  const loaded=validateScene(JSON.parse(JSON.stringify(old)));
  assert.equal(loaded.dynamics.samples,161);
  const tweezer=loaded.lasers.find(x=>x.id==='tweezer');
  assert.equal(tweezer.frequency.wavelengthNm,850,'migrated to the wavelength as written');
  assert.ok(Math.abs(laserFrequencyMHz(tweezer)-299792458000/850)<1e-7);
});

test('mF bands include the lowest complete shifted state card and lower padding',()=>{
  for(const field of [0,.2,20]){
    const scene=defaultScene();scene.fields[0].magnitudeG=field;
    if(field===20)scene.references=[{id:'reference-low',name:'参考',anchorId:'ground-1--1',offsetMHz:-1e5,anchorMode:'follow'}];
    const analysis=spectroscopyAnalysis(scene,'probe',false);
    const svg=combinedLevelsChart(scene,analysis,undefined,{line:'D2',selectedIds:[],initialId:'ground-2-0'});
    const attr=(tag,name)=>Number(tag.match(new RegExp(`\\b${name}="([^"]+)"`))?.[1]);
    const cards=[...svg.matchAll(/<rect class="state-population-background"[^>]*>/g)].map(x=>attr(x[0],'y')+attr(x[0],'height'));
    const bands=[...svg.matchAll(/<g class="mf-band"[^>]*><rect[^>]*>/g)].map(x=>attr(x[0],'y')+attr(x[0],'height'));
    assert.equal(bands.length,7);assert.ok(cards.length>=32);
    for(const bottom of bands)assert.ok(bottom>=Math.max(...cards)+11.99,`${field} G clips a state card`);
    const height=Number(svg.match(/viewBox="0 0 [\d.]+ ([\d.]+)"/)[1]);
    assert.ok(height>=Math.max(...bands)+23.99);
  }
});
