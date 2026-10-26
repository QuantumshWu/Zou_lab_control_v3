// The worker runs OpenBLAS single-threaded. OpenBLAS sizes its thread pool
// and per-thread buffers from OPENBLAS_NUM_THREADS when it initializes, so the
// private child gets 1 instead of the machine's core count.
export function nativeChildEnvironment(parent){
  const child={...parent};
  // Windows environment keys are case insensitive. Avoid duplicate spellings
  // competing with the explicit private-child value passed to spawn().
  for(const key of Object.keys(child))if(key.toUpperCase()==='OPENBLAS_NUM_THREADS')delete child[key];
  child.OPENBLAS_NUM_THREADS='1';
  return child;
}
