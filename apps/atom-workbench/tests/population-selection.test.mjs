import test from 'node:test';
import assert from 'node:assert/strict';
import {recommendedPopulationStates,withPopulationGroups,populationDisplayIds} from '../dist/js/atom-workbench/src/population-selection.js';
import {defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {populationChart} from '../dist/js/atom-workbench/src/population-chart.js';

test('automatic curves use occupied initial/final top three, not intermediate peaks or steady ranking',()=>{
 const populations={a:[.6,0,.1],b:[.3,0,.2],c:[.1,0,.3],d:[0,0,.4],peak:[0,1,0],noise:[1e-12,0,1e-12]};
 const result={states:Object.keys(populations).map(id=>({id})),timesUs:[0,1,2],populations,steady:{populations:{peak:1}}};
 assert.deepEqual(recommendedPopulationStates(result,'a'),['a','b','c','d']);
 assert.deepEqual(recommendedPopulationStates(undefined,'a'),['a']);
 result.populations={a:[1,1,1]};assert.deepEqual(recommendedPopulationStates(result,'a'),['a']);
});

test('F curves sum all constituent states, including hidden ones, without modifying solver data',()=>{
 const result={timesUs:[0,1],populations:{'ground-1--1':[.2,.1],'ground-1-0':[.3,.2],'ground-1-1':[.5,.4]},totalExcited:[0,.3],steady:{populations:{'ground-1--1':.1,'ground-1-0':.2,'ground-1-1':.4}}};
 const before=JSON.stringify(result),grouped=withPopulationGroups(result,['F:ground:1']);
 assert.deepEqual(grouped.populations['F:ground:1'],[1,.7000000000000001]);
 assert.ok(Math.abs(grouped.steady.populations['F:ground:1']-.7)<1e-14);
 assert.equal(JSON.stringify(result),before);
 const svg=populationChart(grouped,['F:ground:1'],{'F:ground:1':'F=1'},{showTotalExcited:false});
 assert.match(svg,/data-series-id="F:ground:1"/);assert.equal((svg.match(/<polyline/g)||[]).length,1);assert.ok(!svg.includes('stroke-dasharray'));
});

test('manual F and individual choices persist and do not trigger a different physical calculation',()=>{
 const scene=defaultScene(),d=scene.dynamics,key=dynamicsCacheKey(scene,d,false);
 d.monitorMode='manual';d.monitoredStateIds=['ground-2-0'];d.monitoredGroupIds=['F:ground:1','F:D2:3'];
 const loaded=validateScene(JSON.parse(JSON.stringify(scene)));
 assert.equal(loaded.dynamics.monitorMode,'manual');assert.deepEqual(populationDisplayIds(loaded.dynamics),['ground-2-0','F:ground:1','F:D2:3']);
 assert.equal(dynamicsCacheKey(loaded,loaded.dynamics,false),key);
 for(const groups of [['unknown'],['F:ground:1','F:ground:1']]){d.monitoredGroupIds=groups;assert.throws(()=>validateScene(JSON.parse(JSON.stringify(scene))));}
});
