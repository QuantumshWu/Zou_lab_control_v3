import test from 'node:test';
import assert from 'node:assert/strict';
import {idMappingError} from '../dist/js/atom-workbench/src/population-basis.js';
import {analyze} from '../dist/js/atom-workbench/src/physics.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';

const levels=()=>{const a=analyze(defaultScene());return [...a.ground,...a.D1,...a.D2].map(l=>({...l}));};
const unit=i=>({real:Array.from({length:32},(_,k)=>+(k===i)),imag:Array(32).fill(0)});

test('p(t) carried over by id is exact in one basis and bounded by the overlap otherwise',()=>{
  const model=levels(),display=levels();
  assert.equal(idMappingError(model,display),0);
  // Display states 0 and 1 rotated by θ: δ = sin²θ, so the bound is 2 sin²θ + 2 |sin θ|.
  const theta=1e-3,c=Math.cos(theta),s=Math.sin(theta);
  display[0].backgroundState={real:unit(0).real.map((v,k)=>k===0?c:k===1?s:0),imag:Array(32).fill(0)};
  display[1].backgroundState={real:unit(1).real.map((v,k)=>k===0?-s:k===1?c:0),imag:Array(32).fill(0)};
  const bound=2*s*s+2*Math.abs(s);
  assert.ok(Math.abs(idMappingError(model,display)-bound)<1e-12);
  assert.equal(idMappingError(model,display.map((l,i)=>i===5?{...l,id:'nowhere'}:l)),Infinity);
});
