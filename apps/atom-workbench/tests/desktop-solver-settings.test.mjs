import test from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {configureDesktopSolver,DESKTOP_SOLVER_DEFAULTS} from '../scripts/desktop-solver-settings.mjs';

test('desktop enables thread defaults and pins DLLs while honoring explicit overrides',()=>{
  const executable=resolve('test-package/solver/atom-lindblad.exe');
  const defaults=configureDesktopSolver(executable,{});
  assert.equal(defaults.ATOM_STEADY_BLAS_THREADS,'1');assert.equal(defaults.ATOM_TRAJECTORY_BLAS_THREADS,'8');
  assert.equal(defaults.ATOM_PENCIL_THREADS,'8');assert.equal(defaults.ATOM_POPULATION_BLAS_THREADS,'1');
  // Algorithm choices are fixed in the Rust solver; only thread settings remain.
  assert.deepEqual(Object.keys(DESKTOP_SOLVER_DEFAULTS).sort(),['ATOM_PENCIL_THREADS','ATOM_POPULATION_BLAS_THREADS','ATOM_STEADY_BLAS_THREADS','ATOM_TRAJECTORY_BLAS_THREADS']);
  const env={ATOM_STEADY_BLAS_THREADS:'3',ATOM_TRAJECTORY_BLAS_THREADS:'2',ATOM_OPENBLAS_DLL:'external.dll'};
  assert.strictEqual(configureDesktopSolver(executable,env),env);
  assert.equal(env.ATOM_STEADY_BLAS_THREADS,'3');assert.equal(env.ATOM_TRAJECTORY_BLAS_THREADS,'2');assert.equal(env.ATOM_PENCIL_THREADS,'8');
  assert.equal(env.ATOM_OPENBLAS_DLL,resolve('test-package/solver/libscipy_openblas.dll'));
});
