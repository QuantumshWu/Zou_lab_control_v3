import {dirname,resolve} from 'node:path';

// Thread settings only: every numerical algorithm choice is fixed in Rust.
// The population kernels parallelize over output modes; BLAS is always
// single-threaded inside the worker.
export const DESKTOP_SOLVER_DEFAULTS=Object.freeze({
  ATOM_SOLVER_THREADS:'8',
});

// Only the desktop entry applies these defaults; the development service keeps
// its environment untouched.
export function configureDesktopSolver(executable,env){
  for(const [name,value]of Object.entries(DESKTOP_SOLVER_DEFAULTS))if(env[name]===undefined)env[name]=value;
  const folder=dirname(executable);
  env.ATOM_OPENBLAS_DLL=resolve(folder,'libscipy_openblas.dll');
  return env;
}
