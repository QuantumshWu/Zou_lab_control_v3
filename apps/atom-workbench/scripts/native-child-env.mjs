// OpenBLAS sizes its thread pool from OPENBLAS_NUM_THREADS when the library
// initializes, before Rust applies ATOM_BLAS_THREADS. Give the private child
// the same count so no idle pool or per-thread buffers are created. An invalid
// ATOM_BLAS_THREADS is passed through unchanged for the worker to reject.
export function nativeChildEnvironment(parent){
  const raw=String(parent.ATOM_BLAS_THREADS||'1');
  const threads=/^\d+$/.test(raw)&&Number(raw)>=1&&Number(raw)<=32?raw:'1';
  const child={...parent};
  // Windows environment keys are case insensitive. Avoid duplicate spellings
  // competing with the explicit private-child values passed to spawn().
  for(const key of Object.keys(child))if(['ATOM_BLAS_THREADS','OPENBLAS_NUM_THREADS'].includes(key.toUpperCase()))delete child[key];
  child.ATOM_BLAS_THREADS=raw;
  child.OPENBLAS_NUM_THREADS=threads;
  return child;
}
