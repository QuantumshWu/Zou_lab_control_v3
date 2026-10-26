import {dirname,resolve} from 'node:path';

export const DESKTOP_SOLVER_DEFAULTS=Object.freeze({
  ATOM_FULL_PERIOD:'auto',
  ATOM_PARALLEL_MEAN:'1',
  ATOM_STEADY_BLAS_THREADS:'1',
  ATOM_TRAJECTORY_BLAS_THREADS:'8',
  ATOM_PENCIL_THREADS:'8',
  ATOM_POPULATION_BLAS_THREADS:'1',
  ATOM_LONG_FOURIER_SERIAL:'1',
});

// Only the desktop entry applies these defaults; the development service keeps
// its environment untouched for controlled algorithm comparisons.
export function configureDesktopSolver(executable,env){
  for(const [name,value]of Object.entries(DESKTOP_SOLVER_DEFAULTS))if(env[name]===undefined)env[name]=value;
  const folder=dirname(executable);
  env.ATOM_OPENBLAS_DLL=resolve(folder,'libscipy_openblas.dll');
  env.ATOM_SUPERLU_DLL=resolve(folder,'atom_superlu.dll');
  return env;
}
