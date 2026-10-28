import test from 'node:test';import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {durationReading,durationText,durationValue,setDuration} from '../dist/js/atom-workbench/src/duration.js';
import {typedDigits} from '../dist/js/atom-workbench/src/unit-input.js';
import {trajectoryInputsKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {populationLayout} from '../dist/js/atom-workbench/src/population-chart.js';
test('the duration is an input: a unit picked for it keeps its digits, a new duration solved anew',()=>{
 const s=defaultScene(),d=s.dynamics;setDuration(d,10,'μs');const key=trajectoryInputsKey(s,d);
 // 总时长 10 μs read in ms is 10 ms: the digits stay, the quantity and the request change.
 setDuration(d,durationReading(d).value,'ms');assert.equal(d.durationUs,1e4);assert.deepEqual(durationReading(d),{value:10,unit:'ms'});assert.notEqual(trajectoryInputsKey(s,d),key);
 for(const [unit,us] of [['s',1e7],['ns',.01],['μs',10]]){setDuration(d,typedDigits(durationValue(d)),unit);assert.equal(d.durationUs,us);assert.equal(d.durationUnit,unit);}
 assert.equal(trajectoryInputsKey(s,d),key,'back in μs: the same duration and request');
 // Without a unit the value is written in the duration's own unit.
 setDuration(d,5,'s');setDuration(d,7);assert.equal(d.durationUs,7e6);assert.equal(d.durationUnit,'s');
 const restored=validateScene(JSON.parse(JSON.stringify(s)));assert.equal(restored.dynamics.durationUs,7e6);assert.equal(restored.dynamics.durationUnit,'s');
 // A refused value or unit changes nothing.
 for(const value of [0,-1,Infinity,NaN])assert.throws(()=>setDuration(d,value),/正的有限/);
 assert.throws(()=>setDuration(d,1e308,'s'),/正的有限/);assert.throws(()=>setDuration(d,1,'min'),/单位/);assert.equal(d.durationUs,7e6);assert.equal(d.durationUnit,'s');
});
test('time displays convert: the same time in another unit',()=>{
 assert.equal(durationText(1500,'ms'),'1.5 ms');assert.equal(durationText(1500,'μs'),'1500 μs');assert.equal(durationText(1500,'s'),'0.0015 s');assert.equal(durationText(2.5e6),'2.5 s');
});
test('sample count roundtrips and changes the output request independently of time units',()=>{
 const s=defaultScene(),before=trajectoryInputsKey(s,s.dynamics);for(const count of [2,51,1001,2001]){s.dynamics.samples=count;assert.equal(validateScene(JSON.parse(JSON.stringify(s))).dynamics.samples,count);assert.notEqual(trajectoryInputsKey(s,s.dynamics),before);}
 for(const count of [1,2002,3.5]){s.dynamics.samples=count;assert.throws(()=>validateScene(JSON.parse(JSON.stringify(s))));}
});
test('chart axes follow the selected time unit for long simulations',()=>{
 const data={timesUs:[0,2.5e6,5e6],populations:{g:[1,1,1]},totalExcited:[0,0,0]};
 const layout=populationLayout(data,['g'],{timeUnit:'s'});assert.equal(layout.unit.name,'s');assert.equal(layout.maxTime*layout.unit.factor,5);
});
