import {dirname,resolve} from 'node:path';

export const DESKTOP_SOLVER_DEFAULTS=Object.freeze({
  ATOM_FULL_PERIOD:'auto',
  ATOM_PARALLEL_MEAN:'1',
  ATOM_STEADY_BLAS_THREADS:'1',
  ATOM_TRAJECTORY_BLAS_THREADS:'8',
  ATOM_PENCIL_THREADS:'8',
  ATOM_POPULATION_BLAS_THREADS:'1',
  ATOM_LONG_FOURIER_SERIAL:'1',
  ATOM_PERIOD_BATH_FRAME:'1',
  ATOM_PERIOD_WORKSPACE:'1',
  ATOM_PERIOD_BATCH_PARTIAL:'1',
  ATOM_PERIOD_ALTERNATING_LAYOUT:'1',
  ATOM_PERIOD_FRAME_POWERS:'1',
  ATOM_PERIOD_POPULATION_TOLERANCE:'0.0005',
  ATOM_NATIVE_ILUTP:'1',
  ATOM_STRUCTURED_FLOQUET_PRODUCT:'1',
  ATOM_STRUCTURED_OUTER_PRODUCT:'1',
  ATOM_INCREMENTAL_PROJECTED:'1',
  ATOM_STRUCTURED_MEAN_PRODUCT:'1',
  ATOM_GMRES_TRUE_CHECK:'1',
  ATOM_KRYLOV_NO_RECYCLE:'1',
  ATOM_AFFINE_CENTER:'auto',
  ATOM_REUSE_HARMONIC_ILUTP:'1',
  ATOM_REUSE_MEAN_ILUTP:'1',
});

// Only the desktop entry applies these defaults; the development service keeps
// its environment untouched for controlled algorithm comparisons.
export function configureDesktopSolver(executable,env){
  for(const [name,value]of Object.entries(DESKTOP_SOLVER_DEFAULTS))if(env[name]===undefined)env[name]=value;
  // Native recycling currently uses presence, rather than a boolean parser.
  if(env.ATOM_KRYLOV_NO_RECYCLE==='0')delete env.ATOM_KRYLOV_NO_RECYCLE;
  const folder=dirname(executable);
  env.ATOM_OPENBLAS_DLL=resolve(folder,'libscipy_openblas.dll');
  env.ATOM_SUPERLU_DLL=resolve(folder,'atom_superlu.dll');
  return env;
}
