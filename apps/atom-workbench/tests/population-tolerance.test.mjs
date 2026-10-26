import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene,defaultDynamics,resolvePopulationTolerance} from '../dist/js/atom-workbench/src/types.js';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {dynamicsCacheKey} from '../dist/js/atom-workbench/src/dynamics-cache-key.js';

test('population tolerance defaults old scenes and preserves valid custom probability values',()=>{
  assert.equal(defaultDynamics().populationTolerance,.005);
  const legacy=defaultScene();delete legacy.dynamics.populationTolerance;
  assert.equal(validateScene(legacy).dynamics.populationTolerance,.005);
  delete legacy.dynamics;
  assert.equal(validateScene(legacy).dynamics.populationTolerance,.005);
  for(const epsilon of [.001,.005,.02,.003731,1e-12]){
    const scene=defaultScene();scene.dynamics.populationTolerance=epsilon;
    const loaded=validateScene(JSON.parse(JSON.stringify(scene)));
    assert.equal(loaded.dynamics.populationTolerance,epsilon);
    assert.equal(validateScene(JSON.parse(JSON.stringify(loaded))).dynamics.populationTolerance,epsilon);
  }
  for(const invalid of [0,-.001,.0200001,NaN,Infinity,null,'0.005']){
    assert.throws(()=>resolvePopulationTolerance(invalid),/populationTolerance/);
    const scene=defaultScene();scene.dynamics.populationTolerance=invalid;
    assert.throws(()=>validateScene(scene),/populationTolerance/);
  }
});

test('population tolerance is part of current request identity with legacy default equivalence',()=>{
  const scene=defaultScene(),settings=scene.dynamics;
  const before=dynamicsCacheKey(scene,settings,false);
  delete settings.populationTolerance;
  assert.equal(dynamicsCacheKey(scene,settings,false),before);
  settings.populationTolerance=.003731;
  assert.notEqual(dynamicsCacheKey(scene,settings,false),before);
});
