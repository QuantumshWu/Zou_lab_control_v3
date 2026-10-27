import test from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {configureDesktopSolver,DESKTOP_SOLVER_DEFAULTS} from '../scripts/desktop-solver-settings.mjs';

test('desktop enables thread defaults and pins DLLs while honoring explicit overrides',()=>{
  const executable=resolve('test-package/solver/atom-lindblad.exe');
  const defaults=configureDesktopSolver(executable,{});
  assert.equal(defaults.ATOM_SOLVER_THREADS,'8');
  // Algorithm choices are fixed in the Rust solver; only the pool size remains.
  assert.deepEqual(Object.keys(DESKTOP_SOLVER_DEFAULTS),['ATOM_SOLVER_THREADS']);
  const env={ATOM_SOLVER_THREADS:'2',ATOM_OPENBLAS_DLL:'external.dll'};
  assert.strictEqual(configureDesktopSolver(executable,env),env);
  assert.equal(env.ATOM_SOLVER_THREADS,'2');
  assert.equal(env.ATOM_OPENBLAS_DLL,resolve('test-package/solver/libscipy_openblas.dll'));
});
