// Which basis a population is in. The solver returns populations in the model basis (the
// eigenstates of the static H0: B and the far lasers' static background, spectroscopy.ts);
// the levels chart shows the display basis (the same background without a selected far
// laser). A long-time mean carries its full density and is
// projected exactly (spectroscopy.ts projectSpectroscopyPopulations). A finite-time
// p(t) carries populations only: read on a display state by its id it is off by at most
// 2√δ + 2δ, δ = 1 − |⟨model state|display state⟩|² of that id -- the bound below.
import type {Level} from './types.js';

type Vector={real:number[];imag:number[]};
const N=32;
const unit=(index:number):Vector=>({real:Array.from({length:N},(_,i)=>+(i===index)),imag:Array(N).fill(0)});

/** The largest change carrying p(t) over by id can make to a display state's population
 * (0 when the bases agree; Infinity when a display state has no model state of its id). */
export function idMappingError(model:readonly Level[],display:readonly Level[]):number{
  if(model.length!==N||display.length!==N)return Infinity;
  let worst=0;
  display.forEach((d,a)=>{
    const j=model.findIndex(s=>s.id===d.id);if(j<0){worst=Infinity;return;}
    const u=model[j]!.backgroundState??unit(j),v=d.backgroundState??unit(a);
    let re=0,im=0;for(let i=0;i<N;i++){re+=u.real[i]!*v.real[i]!+u.imag[i]!*v.imag[i]!;im+=u.real[i]!*v.imag[i]!-u.imag[i]!*v.real[i]!;}
    // |p_a − ρ_aa| ≤ δ ρ_aa + 2√((1−δ)δ) |ρ| + δ ≤ 2δ + 2√δ for a density ρ.
    const delta=Math.max(0,1-(re*re+im*im));worst=Math.max(worst,Math.min(1,2*delta+2*Math.sqrt(delta)));
  });
  return worst;
}
