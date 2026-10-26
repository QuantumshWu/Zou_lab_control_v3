import test from 'node:test';
import assert from 'node:assert/strict';
import {cameraAlong} from '../dist/js/atom-workbench/src/space3d.js';
import {laserSlot,identityToken,laserColour} from '../dist/js/atom-workbench/src/identity.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';

const forward=({yaw,pitch})=>[Math.cos(pitch)*Math.cos(yaw),Math.cos(pitch)*Math.sin(yaw),Math.sin(pitch)];
test('a view along a direction puts that direction toward the viewer, x right at the poles',()=>{
  for(const d of [[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[.3,-.4,.866]]){
    const length=Math.hypot(...d),f=forward(cameraAlong(d));
    f.forEach((v,i)=>assert.ok(Math.abs(v-d[i]/length)<1e-12,`${d}`));
  }
  assert.ok(Math.abs(cameraAlong([0,0,1]).yaw+Math.PI/2)<1e-12);
  assert.deepEqual(cameraAlong([0,0,0]),{yaw:-1.03,pitch:.5});
});
test('each laser keeps one identity colour slot, by its place in the list',()=>{
  const scene=defaultScene();
  assert.equal(laserSlot(scene,'probe'),0);assert.equal(laserSlot(scene,'tweezer'),1);
  assert.equal(laserColour(scene,'tweezer'),'var(--identity-1)');
  assert.equal(identityToken(9),'--identity-1');assert.equal(identityToken(-1),'--identity-7');
});
