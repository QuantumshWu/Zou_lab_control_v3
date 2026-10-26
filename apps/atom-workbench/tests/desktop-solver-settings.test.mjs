import test from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {configureDesktopSolver,DESKTOP_SOLVER_DEFAULTS} from '../scripts/desktop-solver-settings.mjs';

test('desktop enables routing/thread defaults and pins DLLs while honoring explicit overrides',()=>{
  const executable=resolve('test-package/solver/atom-lindblad.exe');
  const defaults=configureDesktopSolver(executable,{});
  assert.equal(defaults.ATOM_FULL_PERIOD,'auto');assert.equal(defaults.ATOM_PARALLEL_MEAN,'1');
  assert.equal(defaults.ATOM_STEADY_BLAS_THREADS,'1');assert.equal(defaults.ATOM_TRAJECTORY_BLAS_THREADS,'8');
  assert.equal(defaults.ATOM_PENCIL_THREADS,'8');assert.equal(defaults.ATOM_POPULATION_BLAS_THREADS,'1');assert.equal(defaults.ATOM_LONG_FOURIER_SERIAL,'1');
  // Former algorithm switches are hard-wired in the Rust solver; only routing and thread settings remain.
  assert.deepEqual(Object.keys(DESKTOP_SOLVER_DEFAULTS).sort(),['ATOM_FULL_PERIOD','ATOM_LONG_FOURIER_SERIAL','ATOM_PARALLEL_MEAN','ATOM_PENCIL_THREADS',
    'ATOM_POPULATION_BLAS_THREADS','ATOM_STEADY_BLAS_THREADS','ATOM_TRAJECTORY_BLAS_THREADS']);
  const env={ATOM_FULL_PERIOD:'0',ATOM_PARALLEL_MEAN:'0',ATOM_TRAJECTORY_BLAS_THREADS:'2',ATOM_OPENBLAS_DLL:'external.dll',ATOM_SUPERLU_DLL:'other.dll'};
  assert.strictEqual(configureDesktopSolver(executable,env),env);
  assert.equal(env.ATOM_FULL_PERIOD,'0');assert.equal(env.ATOM_PARALLEL_MEAN,'0');assert.equal(env.ATOM_TRAJECTORY_BLAS_THREADS,'2');
  assert.equal(env.ATOM_OPENBLAS_DLL,resolve('test-package/solver/libscipy_openblas.dll'));
  assert.equal(env.ATOM_SUPERLU_DLL,resolve('test-package/solver/atom_superlu.dll'));
});
