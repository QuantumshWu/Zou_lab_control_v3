export interface MeanDensity { real:number[]; imag:number[] }

/** Preserve the native complex mean, including coherences needed for a change
 * of measurement basis. The native solver checks positivity of the full matrix. */
export function readMeanDensity(mean:{densityReal?:unknown;densityImag?:unknown},n:number,populations:number[]):MeanDensity|undefined {
  const real=mean.densityReal,imag=mean.densityImag;
  if(real==null&&imag==null)return undefined;
  if(populations.length!==n||populations.some(p=>!Number.isFinite(p)))throw new Error('Mean density requires its population diagonal');
  if(!Array.isArray(real)||!Array.isArray(imag)||real.length!==n*n||imag.length!==n*n||
    [...real,...imag].some(v=>typeof v!=='number'||!Number.isFinite(v)))throw new Error('Invalid complex mean density dimensions or entries');
  for(let i=0;i<n;i++){
    if(Math.abs(real[i*n+i]-populations[i]!)>5e-6||Math.abs(imag[i*n+i])>5e-6)throw new Error('Mean density disagrees with population diagonal');
    for(let j=0;j<n;j++){
      if(Math.abs(real[i*n+j]-real[j*n+i])>5e-6||Math.abs(imag[i*n+j]+imag[j*n+i])>5e-6)throw new Error('Mean density is not Hermitian');
      if(real[i*n+j]**2+imag[i*n+j]**2>Math.max(0,populations[i]!)*Math.max(0,populations[j]!)+5e-6)throw new Error('Mean density violates a positivity minor');
    }
  }
  return {real:real.slice(),imag:imag.slice()};
}
