import test from 'node:test';
import assert from 'node:assert/strict';
import {lineMesh,triangleMesh,rasterizeDepth} from '../dist/js/atom-workbench/src/space-raster.js';
import {rayDepth} from '../dist/js/atom-workbench/src/space-depth.js';
import {readoutBaseline} from '../dist/js/atom-workbench/src/readout-alignment.js';
import {clickState,keepLabels,labelled,swapLabels,withLabel} from '../dist/js/atom-workbench/src/transition-choice.js';
import {referenceFromLaser} from '../dist/js/atom-workbench/src/reference-draft.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {analyze,laserFrequencyMHz} from '../dist/js/atom-workbench/src/physics.js';

test('a line crossing a circle is behind at one intersection and in front at the other, regardless of submission order',()=>{
 const ring=Array.from({length:128},(_,i)=>({x:50+30*Math.cos(i*Math.PI/64),y:50+30*Math.sin(i*Math.PI/64),z:0}));
 const outline=ring.flatMap((p,i)=>lineMesh(p,ring[(i+1)%ring.length],5,'#0000ff'));
 const beam=lineMesh({x:5,y:50,z:-2},{x:95,y:50,z:2},5,'#ff0000');
 const pixel=(result,x)=>Array.from(result.pixels.slice((50*100+x)*4,(50*100+x)*4+3));
 for(const triangles of [[...outline,...beam],[...beam,...outline]]){
  const result=rasterizeDepth(triangles,100,100,8);
  assert.deepEqual(pixel(result,20),[0,0,255]);assert.deepEqual(pixel(result,80),[255,0,0]);
 }
 const filled=rasterizeDepth([...beam,...triangleMesh(ring,'#0000ff',.5)],100,100,8);
 assert.deepEqual(pixel(filled,25),[128,0,128]);assert.deepEqual(pixel(filled,75),[255,0,0]);
});
test('screen-space picking uses perspective-correct depth',()=>{
 const a={z:-2,scale:1/10},b={z:2,scale:1/6};
 assert.ok(Math.abs(rayDepth(a,b,.5)-.5)<1e-12);
});
test('population and shift ink have the same distance from the energy line',()=>{
 const ink={top:-11,bottom:3},line=100;
 assert.equal(line-(readoutBaseline(line,'above',ink)+ink.bottom),8);
 assert.equal(readoutBaseline(line,'below',ink)+ink.top-line,8);
});
test('state labels: nothing preselected, 1 then 2 in click order, a labelled click removes its label, a third replaces 2',()=>{
 let labels={};
 labels=clickState(labels,'ground-2-0');assert.deepEqual(labels,{first:'ground-2-0'});
 labels=clickState(labels,'D2-3-1');assert.deepEqual(labels,{first:'ground-2-0',second:'D2-3-1'});
 labels=clickState(labels,'D2-2-0');assert.deepEqual(labels,{first:'ground-2-0',second:'D2-2-0'},'a third state replaces 2 and keeps 1');
 assert.deepEqual(labelled(labels),[{id:'ground-2-0',label:1},{id:'D2-2-0',label:2}]);
 // A click on a labelled state removes that label; nothing moves to fill it.
 labels=clickState(labels,'ground-2-0');assert.deepEqual(labels,{second:'D2-2-0'},'no automatic swap: 2 stays 2');
 labels=clickState(labels,'ground-1-1');assert.deepEqual(labels,{first:'ground-1-1',second:'D2-2-0'},'the free label 1 is taken first');
 labels=clickState(labels,'D2-2-0');assert.deepEqual(labels,{first:'ground-1-1'});
 labels=clickState(labels,'ground-1-1');assert.deepEqual(labels,{});
});
test('state labels: a picker sets one label, a state holds one label, the swap is explicit, missing states drop out',()=>{
 let labels=withLabel({},'second','D1-2-1');assert.deepEqual(labels,{second:'D1-2-1'});
 labels=withLabel(labels,'first','D1-2-1');assert.deepEqual(labels,{first:'D1-2-1'},'taking the other label moves the state, never duplicates it');
 labels=withLabel(labels,'second','reference-1');assert.deepEqual(swapLabels(labels),{first:'reference-1',second:'D1-2-1'});
 assert.deepEqual(swapLabels({first:'a'}),{second:'a'});
 assert.deepEqual(keepLabels({first:'a',second:'reference-gone'},id=>id!=='reference-gone'),{first:'a'});
 assert.deepEqual(withLabel({first:'a',second:'b'},'first',undefined),{second:'b'});
});
test('a photon endpoint from a reference preserves the physical anchor and anchor mode',()=>{
 const scene=defaultScene(),laser={...scene.lasers[0],referenceStateId:'reference-1'};
 scene.references=[{id:'reference-1',name:'Reference',anchorId:'ground-2-0',anchorMode:'zero-field',offsetMHz:123,offsetUnit:'MHz'}];
 const a=analyze(scene),states=[...a.ground,...a.D1,...a.D2],result=referenceFromLaser(scene,laser,states);
 assert.equal(result.anchorId,'ground-2-0');assert.equal(result.anchorMode,'zero-field');assert.equal(result.offsetMHz,123+laserFrequencyMHz(laser));
 assert.equal(referenceFromLaser(scene,{...laser,referenceProcess:'emission'},states).offsetMHz,123-laserFrequencyMHz(laser));
});
