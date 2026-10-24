import test from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {configureDesktopSolver} from '../scripts/desktop-solver-settings.mjs';

test('desktop enables verified defaults and pins DLLs while honoring explicit algorithm/thread overrides',()=>{
  const executable=resolve('test-package/solver/atom-lindblad.exe');
  const defaults=configureDesktopSolver(executable,{});
  assert.equal(defaults.ATOM_FULL_PERIOD,'auto');assert.equal(defaults.ATOM_PARALLEL_MEAN,'1');
  assert.equal(defaults.ATOM_STEADY_BLAS_THREADS,'1');assert.equal(defaults.ATOM_TRAJECTORY_BLAS_THREADS,'8');
  assert.equal(defaults.ATOM_NATIVE_ILUTP,'1');assert.equal(defaults.ATOM_PERIOD_POPULATION_TOLERANCE,'0.0005');
  assert.equal(defaults.ATOM_AFFINE_CENTER,'auto');assert.equal(defaults.ATOM_REUSE_HARMONIC_ILUTP,'1');assert.equal(defaults.ATOM_REUSE_MEAN_ILUTP,'1');
  const env={ATOM_FULL_PERIOD:'0',ATOM_PARALLEL_MEAN:'0',ATOM_TRAJECTORY_BLAS_THREADS:'2',ATOM_KRYLOV_NO_RECYCLE:'0',ATOM_OPENBLAS_DLL:'external.dll',ATOM_SUPERLU_DLL:'other.dll'};
  assert.strictEqual(configureDesktopSolver(executable,env),env);
  assert.equal(env.ATOM_FULL_PERIOD,'0');assert.equal(env.ATOM_PARALLEL_MEAN,'0');assert.equal(env.ATOM_TRAJECTORY_BLAS_THREADS,'2');
  assert.equal(Object.hasOwn(env,'ATOM_KRYLOV_NO_RECYCLE'),false);
  assert.equal(env.ATOM_OPENBLAS_DLL,resolve('test-package/solver/libscipy_openblas.dll'));
  assert.equal(env.ATOM_SUPERLU_DLL,resolve('test-package/solver/atom_superlu.dll'));
});
