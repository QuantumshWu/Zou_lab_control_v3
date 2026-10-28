import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze} from '../dist/js/atom-workbench/src/physics.js';
import {displayMixing,spectroscopyAnalysis,staticBackgroundProblem} from '../dist/js/atom-workbench/src/spectroscopy.js';
import {solveStaticBackground} from '../dist/js/atom-workbench/src/floquet-background.js';
import {spherical} from './directions.mjs';

const levels=a=>[...a.ground,...a.D1,...a.D2];
const lattice=()=>({id:'lattice',name:'晶格光',enabled:true,showLightShift:true,line:'D2',frequency:{wavelengthNm:1064},intensityMode:'intensity',intensityMWcm2:2e7,
  direction:spherical(60,20),polarizationAngleDeg:10,ellipticityDeg:15,phaseDeg:0});

// ui-design.md §4, §6.8: 显示光移 puts a laser's all-order static shift into the levels chart's background.
test('a laser shown in the levels chart adds its all-order static shift to the background',()=>{
  const s=defaultScene(),a=spectroscopyAnalysis(s),solution=solveStaticBackground(staticBackgroundProblem(s));
  assert.equal(a.spectroscopy.status,'dispersive');assert.deepEqual(a.spectroscopy.backgroundLaserIds,['tweezer']);
  levels(a).forEach((level,i)=>{assert.equal(level.energyMHz,solution.levels[i].energyMHz,level.id);assert.deepEqual(level.shiftBound,{floquetMHz:solution.levels[i].floquetBoundMHz,outsideMHz:solution.levels[i].outsideBoundMHz});});
  assert.ok(a.spectroscopy.numerics.floquetBoundMHz+a.spectroscopy.numerics.outsideBoundMHz<1e-6,'the shift is known to well below 1 Hz');
  assert.ok(levels(a).find(l=>l.id==='ground-2-0').totalShiftMHz< -10,'the 852 nm tweezer lowers the ground state by tens of MHz');
  // Two shown lasers: one Floquet problem with both carriers.
  const both=defaultScene();both.lasers.push(lattice());
  const b=spectroscopyAnalysis(both);assert.deepEqual(b.spectroscopy.backgroundLaserIds,['tweezer','lattice']);
  assert.equal(staticBackgroundProblem(both).carriers.length,2);
  levels(b).forEach((level,i)=>assert.equal(level.energyMHz,solveStaticBackground(staticBackgroundProblem(both)).levels[i].energyMHz));
});

test('with the switch off the laser is not in the background, and the probe is never shown by default',()=>{
  const s=defaultScene();s.lasers[1].showLightShift=false;
  const a=spectroscopyAnalysis(s),bare=analyze(s);
  assert.equal(a.spectroscopy.status,'bare');assert.deepEqual(a.spectroscopy.backgroundLaserIds,[]);
  levels(a).forEach((level,i)=>{assert.equal(level.energyMHz,levels(bare)[i].energyMHz);assert.equal(level.starkShiftMHz,0);});
  assert.equal(staticBackgroundProblem(s).carriers.length,0);
  assert.equal(defaultScene().lasers.find(l=>l.id==='probe').showLightShift,false);
});

test('a near-resonant laser shown in the chart is too close: its static shift is left undefined, not drawn',()=>{
  const s=defaultScene();s.lasers[0].showLightShift=true;
  const m=displayMixing(s,'probe');assert.equal(m.withinGap,false);
  const a=spectroscopyAnalysis(s);
  assert.equal(a.spectroscopy.status,'unresolved');
  assert.match(a.spectroscopy.warnings[0],/探测光.*(虚跃迁失谐小于|混合比)/);
  assert.ok(levels(a).every(l=>l.totalShiftMHz===null),'bare positions, shift undefined');
  // The tweezer alone at 852 nm is far: well within the 1% mixing limit.
  const tweezer=displayMixing(defaultScene(),'tweezer');assert.ok(tweezer.withinGap&&tweezer.ratio>0&&tweezer.ratio<.01);
});
