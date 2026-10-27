import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {validateScene} from '../dist/js/atom-workbench/src/scene-file.js';
import {buildFullMasterModel} from '../dist/js/atom-workbench/src/full-master-model.js';
import {prepareNativeSolve} from '../scripts/native-protocol.mjs';

// The requests the population core's lattice was validated on: scenes saved with the
// 处理方式 control and the request research/population/models/gen_model.mjs (branch
// claude/atom-research) built from each, as SHA-256 digests (tests/fixtures/research-requests.json).
const fixture=JSON.parse(readFileSync(new URL('./fixtures/research-requests.json',import.meta.url),'utf8'));
const sha=value=>createHash('sha256').update(value).digest('hex');
const FIELDS=['hReal','hImag','collapse','initialReal','initialImag','drives','frequencyCoordinates'];
/** gen_model.mjs's payload(): the builder's model with each drive's omega, cos and sin. */
function payload(scene){
  const m=buildFullMasterModel(scene,scene.dynamics);
  return {modelSchema:m.modelSchema,options:{},model:{dimension:m.dimension,states:m.states.map(s=>s.id),hReal:m.hReal,hImag:m.hImag,collapse:m.collapse,
    initialIndex:m.initialIndex,initialReal:m.initialReal,initialImag:m.initialImag,drives:m.drives.map(d=>({omega:d.omega,cos:d.cos,sin:d.sin})),frequencyCoordinates:m.frequencyCoordinates}};
}
/** The scene as gen_model.mjs reads it (its durationUnit 'us' is today's 'μs'). */
function load(raw){const copy=structuredClone(raw);if(copy.dynamics?.durationUnit==='us')copy.dynamics.durationUnit='μs';return validateScene(copy);}

for(const [name,{scene:raw,request:expected}] of Object.entries(fixture.scenes)){
  test(`${name}: the request is bit for bit the one the lattice core was validated on`,()=>{
    const scene=load(raw),request=payload(scene),m=request.model;
    assert.equal(m.drives.length,expected.drives);assert.equal(m.collapse.length,expected.collapse);
    assert.deepEqual(m.frequencyCoordinates.frequenciesMHz,expected.frequencyCoordinates.frequenciesMHz);
    assert.deepEqual(m.frequencyCoordinates.driveIndices,expected.frequencyCoordinates.driveIndices);
    assert.deepEqual(m.states,expected.states);assert.equal(m.initialIndex,expected.initialIndex);
    for(const key of FIELDS)assert.equal(sha(JSON.stringify(m[key])),expected.fieldSha256[key],key);
    assert.equal(sha(JSON.stringify(request)),expected.sha256);
  });
  test(`${name}: the app's native request carries the same model, with the fixed accuracy`,()=>{
    const scene=load(raw),{payload:native}=prepareNativeSolve({scene,options:{durationUs:scene.dynamics.durationUs,samples:scene.dynamics.samples}}),m=native.model;
    for(const key of FIELDS.filter(key=>key!=='drives'))assert.equal(sha(JSON.stringify(m[key])),expected.fieldSha256[key],key);
    // The app's drives also name their lasers; omega, cos and sin are the validated ones.
    assert.equal(sha(JSON.stringify(m.drives.map(d=>({omega:d.omega,cos:d.cos,sin:d.sin})))),expected.fieldSha256.drives);
    assert.ok(m.drives.every(d=>Array.isArray(d.laserIds)&&d.laserIds.length>0));
    assert.equal(native.options.populationTolerance,.005);
  });
}
