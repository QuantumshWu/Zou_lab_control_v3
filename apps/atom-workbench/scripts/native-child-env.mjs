// The DLL reads OPENBLAS_NUM_THREADS during load, before Rust can call
// scipy_openblas_set_num_threads. Match lapack::load's i32 parse/clamp here.
export function nativeChildEnvironment(lane,parent){
  const raw=String(parent[lane==='steady'?'ATOM_STEADY_BLAS_THREADS':'ATOM_TRAJECTORY_BLAS_THREADS']||parent.ATOM_BLAS_THREADS||'1');
  const value=/^[+-]?\d+$/.test(raw)?Number(raw):NaN;
  const parsed=Number.isInteger(value)&&value>=-2147483648&&value<=2147483647?value:1;
  const child={...parent};
  // Windows environment keys are case insensitive. Avoid duplicate spellings
  // competing with the explicit private-child values passed to spawn().
  for(const key of Object.keys(child))if(['ATOM_BLAS_THREADS','OPENBLAS_NUM_THREADS'].includes(key.toUpperCase()))delete child[key];
  child.ATOM_BLAS_THREADS=raw;
  child.OPENBLAS_NUM_THREADS=String(Math.min(32,Math.max(1,parsed)));
  return child;
}
