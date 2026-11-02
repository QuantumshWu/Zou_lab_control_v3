import test from 'node:test';
import assert from 'node:assert/strict';
import {POPULATION_TOLERANCE,defaultDynamics,defaultScene} from '../dist/js/atom-workbench/src/types.js';
import {trajectoryInputsKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';

// The population accuracy is fixed (POPULATION_TOLERANCE, 1%): no setting, no part of a scene or of a request's identity.
test('every native request asks for the fixed populationTolerance, whatever the scene or the options carry',()=>{
  assert.equal(POPULATION_TOLERANCE,.01);assert.ok(!('populationTolerance' in defaultDynamics()));
  for(const stray of [undefined,.001,.02,.05,'0.01',null]){
    const scene=defaultScene();scene.lasers=scene.lasers.slice(0,1);if(stray!==undefined)scene.dynamics.populationTolerance=stray;
    for(const options of [{},{populationTolerance:stray},{meanOnly:true}]){
      const {payload}=prepareNativeSolve({scene,options});
      assert.equal(payload.options.populationTolerance,POPULATION_TOLERANCE,`${stray} ${JSON.stringify(options)}`);
    }
  }
});

test('a stray tolerance is no part of the request identity',()=>{
  const scene=defaultScene(),settings=scene.dynamics,before=trajectoryInputsKey(scene,settings);
  for(const stray of [.001,.02,'x'])assert.equal(trajectoryInputsKey(scene,{...settings,populationTolerance:stray}),before);
});
