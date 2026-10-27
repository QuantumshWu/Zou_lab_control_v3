import test from 'node:test';
import assert from 'node:assert/strict';
import {POPULATION_TOLERANCE,defaultDynamics,defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';

// The population accuracy is fixed at 0.5%: no setting, no part of a scene or of a request's identity.
test('every native request asks for populationTolerance 0.005, whatever the scene or the options carry',()=>{
  assert.equal(POPULATION_TOLERANCE,.005);assert.ok(!('populationTolerance' in defaultDynamics()));
  for(const stray of [undefined,.001,.02,.05,'0.005',null]){
    const scene=defaultScene();scene.lasers=scene.lasers.slice(0,1);if(stray!==undefined)scene.dynamics.populationTolerance=stray;
    for(const options of [{},{populationTolerance:stray},{steadyOnly:true}]){
      const {payload}=prepareNativeSolve({scene,options});
      assert.equal(payload.options.populationTolerance,.005,`${stray} ${JSON.stringify(options)}`);
    }
  }
});

test('a stray tolerance is no part of the request identity',()=>{
  const scene=defaultScene(),settings=scene.dynamics,before=dynamicsCacheKey(scene,settings,false);
  for(const stray of [.001,.02,'x'])assert.equal(dynamicsCacheKey(scene,{...settings,populationTolerance:stray},false),before);
});
