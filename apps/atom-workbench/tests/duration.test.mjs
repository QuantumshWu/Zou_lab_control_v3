import test from 'node:test';import assert from 'node:assert/strict';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {durationUnit,durationValue,setDurationUnit,setDurationValue} from '../dist/js/atom-workbench/src/duration.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {populationLayout} from '../dist/js/atom-workbench/src/population-chart.js';
test('unit switches retain the physical time and numerical request identity exactly',()=>{
 const s=defaultScene(),d=s.dynamics,key=dynamicsCacheKey(s,d,false),time=d.durationUs;
 for(const unit of ['s','ns','ms','μs']){setDurationUnit(d,unit);assert.equal(d.durationUs,time);assert.equal(dynamicsCacheKey(s,d,false),key);}
 setDurationUnit(d,'s');setDurationValue(d,5);assert.equal(d.durationUs,5e6);assert.equal(durationValue(d),5);assert.equal(durationUnit(d),'s');
 const restored=validateScene(JSON.parse(JSON.stringify(s)));assert.equal(restored.dynamics.durationUs,5e6);assert.equal(restored.dynamics.durationUnit,'s');
 for(const value of [0,-1,Infinity,NaN])assert.throws(()=>setDurationValue(d,value),/正的有限/);
});
test('sample count roundtrips and changes the output request independently of time units',()=>{
 const s=defaultScene(),before=dynamicsCacheKey(s,s.dynamics,false);for(const count of [2,51,1001,2001]){s.dynamics.samples=count;assert.equal(validateScene(JSON.parse(JSON.stringify(s))).dynamics.samples,count);assert.notEqual(dynamicsCacheKey(s,s.dynamics,false),before);}
 for(const count of [1,2002,3.5]){s.dynamics.samples=count;assert.throws(()=>validateScene(JSON.parse(JSON.stringify(s))));}
});
test('chart axes follow the selected time unit for long simulations',()=>{
 const data={timesUs:[0,2.5e6,5e6],populations:{g:[1,1,1]},totalExcited:[0,0,0]};
 const layout=populationLayout(data,['g'],{timeUnit:'s'});assert.equal(layout.unit.name,'s');assert.equal(layout.maxTime*layout.unit.factor,5);
});
