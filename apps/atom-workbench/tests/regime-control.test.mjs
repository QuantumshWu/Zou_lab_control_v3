import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {classifyLasers,farValidity} from '../dist/js/atom-workbench/src/laser-regime.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';

const scene=()=>validateScene(JSON.parse(JSON.stringify(defaultScene())));

// ui-design.md §6.8: the inspector shows the model's own refusal, in the control's words.
test('an 852 nm beam typed in as 进入主方程 is refused plainly, with the solver\'s words',()=>{
  const s=scene(),tweezer=s.lasers.find(l=>l.id==='tweezer');tweezer.regime='near';
  const refusal=classifyLasers(s).find(c=>c.id==='tweezer');
  assert.equal(refusal.valid,false);
  assert.match(refusal.message,/离最近的允许跃迁（5[SP].*）有 .* THz，超出“进入主方程”的近共振范围 .*（max\(1 GHz, 10·Ω\/2π\)）/);
  assert.match(refusal.message,/请改为“静态光移”/);
  assert.equal(tweezer.regime,'near','the regime is never changed for the operator');
  assert.throws(()=>buildFullMasterModel(s),error=>error.message===refusal.message);
});

test('a 静态光移 beam too close to resonance is refused with farValidity\'s message by the model too',()=>{
  const s=scene(),probe=s.lasers.find(l=>l.id==='probe');probe.regime='far';
  const check=farValidity(s,'probe');
  assert.equal(check.valid,false);assert.match(check.message,/“静态光移”的远失谐条件/);assert.match(check.message,/请改为“进入主方程”/);
  assert.throws(()=>buildFullMasterModel(s),error=>error.message===check.message);
  // The default tweezer is a valid static shift.
  assert.equal(farValidity(scene(),'tweezer').valid,true);
});
