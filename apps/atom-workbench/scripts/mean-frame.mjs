/** Undo rotating coordinates before averaging a density. A nonzero harmonic
 * can become a DC ground-state coherence; simply deleting it is incorrect. */
function declaredFrames(coordinates,transform,n,dimension){
 if(!coordinates)return null;
 const source=coordinates.frequencies?.length;
 if(!Number.isInteger(source)||!Array.isArray(coordinates.frameIndices)||coordinates.frameIndices.length!==n||
    !Array.isArray(coordinates.frameSectors)||coordinates.frameSectors.length!==n||coordinates.frameSectors.some(x=>!Number.isSafeInteger(x))||
    !Array.isArray(transform)||transform.length!==source||transform.some(row=>!Array.isArray(row)||row.length!==dimension||row.some(x=>!Number.isSafeInteger(x)))){
  throw Error('Invalid declared carrier-frame coordinate mapping');
 }
 const indices=coordinates.frameIndices.map(row=>{
  if(!Array.isArray(row)||row.length!==source||row.some(x=>!Number.isSafeInteger(x)))throw Error('Invalid declared frame indices');
  const output=Array(dimension).fill(0);
  row.forEach((coefficient,i)=>transform[i].forEach((value,j)=>{output[j]+=coefficient*value;}));
  if(output.some(x=>!Number.isSafeInteger(x)))throw Error('Carrier-frame integer coordinate overflow');
  return output;
 });
 return {indices,sectors:coordinates.frameSectors};
}
export function physicalMeanDensity(result,frame,n,coordinates){
 if(!Array.isArray(result.asymptoticDensityReal)||!Array.isArray(result.asymptoticDensityImag))return result;
 if(!coordinates&&(!Array.isArray(frame)||frame.length!==n))return result;
 const oldReal=result.asymptoticDensityReal,oldImag=result.asymptoticDensityImag;
 const real=Array(n*n).fill(0),imag=Array(n*n).fill(0),fourier=result.asymptoticFourierDensity;
 if(fourier){
  const frequencies=fourier.frequenciesRadPerUs;
  if(!Array.isArray(frequencies)||!Array.isArray(fourier.harmonics))throw Error('Invalid mean-density Fourier data');
  const declared=declaredFrames(coordinates,fourier.sourceToBasis,n,frequencies.length);
  for(const h of fourier.harmonics){
   if(h.index.length!==frequencies.length||h.real.length!==n*n||h.imag.length!==n*n)throw Error('Invalid mean-density Fourier dimensions');
   const frequency=h.index.reduce((sum,k,i)=>sum+k*frequencies[i],0),central=h.index.every(k=>k===0);
   for(let i=0;i<n;i++)for(let j=0;j<n;j++){
    const target=declared?declared.indices[i].map((x,k)=>x-declared.indices[j][k]):2*Math.PI*(frame[i]-frame[j]),k=i*n+j;
    // Exact represented-frequency identity; no near-frequency rounding turns
    // a long beat into a DC term at infinite time.
    // New models declare carrier relationships BEFORE floating-point
    // conversion. Integer matching also retains noncentral DC coherences.
    const match=sign=>declared?declared.sectors[i]===declared.sectors[j]&&h.index.every((v,a)=>sign*v===target[a]):sign*frequency===target;
    if(match(1)){real[k]+=h.real[k];imag[k]+=h.imag[k];}
    if(!central&&match(-1)){real[k]+=h.real[j*n+i];imag[k]-=h.imag[j*n+i];}
   }
  }
 }else {
  const transform=result.diagnostics?.carrierSourceToBasis;
  const declared=declaredFrames(coordinates,transform,n,transform?.[0]?.length??0);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
   const same=declared?declared.sectors[i]===declared.sectors[j]&&declared.indices[i].every((v,k)=>v===declared.indices[j][k]):frame[i]===frame[j];
   if(same){const k=i*n+j;real[k]=oldReal[k];imag[k]=oldImag[k];}
  }
 }
 const changed=real.some((v,i)=>Math.abs(v-oldReal[i])+Math.abs(imag[i]-oldImag[i])>1e-8);
 if(changed&&result.asymptoticStatus==='stationary'){
  result.asymptoticStatus='population-stationary';result.diagnostics.asymptoticStatus='population-stationary';result.diagnostics.densityLimitExists=false;
  result.diagnostics.asymptoticReason='Populations converge in the background basis; exported density is the laboratory-frame time average.';
 }
 result.asymptoticDensityReal=real;result.asymptoticDensityImag=imag;
 result.diagnostics.meanDensityFrame='laboratory time average in the background eigenbasis';
 delete result.asymptoticFourierDensity;return result;
}
