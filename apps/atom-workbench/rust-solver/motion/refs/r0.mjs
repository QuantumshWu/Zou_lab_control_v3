// node r0.mjs [ids...] -- R0, the static limit (contract §6): for each scene of the R0 set (scenes.mjs R0_SET) the
// population core (the atom-lindblad worker of rust-solver/src: the complete 32-state Lindblad equation of the atom at
// rest at the focus, ε = 0.01) solves the scene's static request (out/R0/<id>.static.json, the same model without the
// motion block); its rows are the expectation of the motion solver run on out/R0/<id>.json (the same scene with the mass
// ×10⁶ and 1e-3 µK: m → ∞ decouples the motion, and the jump ensemble's mean is the Lindblad equation at the focus).
// Written to expect/R0_<id>.json: every state that reaches 1e-3 and the nine groups (F manifolds and all excited),
// u = the core's own per-time estimated error; L ≡ 0.
//
// The worker: lib/worker.mjs (the app's own binary; ATOM_LINDBLAD overrides), its LAPACK from ATOM_OPENBLAS_DLL.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {R0_SET,HELD} from './scenes.mjs';
import {runWorker} from './lib/worker.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const read=f=>JSON.parse(fs.readFileSync(path.join(here,f),'utf8'));

/** The nine population groups of the result's scope (contract §2.2): the eight F manifolds and all excited states. */
export function groups(states){
  const out=[];
  for(const [m,Fs] of [['ground',[1,2]],['D1',[1,2]],['D2',[0,1,2,3]]])for(const F of Fs)out.push({label:`${m} F=${F}`,states:states.filter(s=>s.startsWith(`${m}-${F}-`))});
  out.push({label:'F:excited',states:states.filter(s=>!s.startsWith('ground-'))});
  return out;
}

async function run(id){
  const still=read(`out/R0/${id}.static.json`),started=performance.now(),{final}=await runWorker(still,{progress:false});
  if(final.error)throw new Error(`${id}: ${final.error}`);
  const r=final.result,states=r.states,P=r.populations,u=r.populationAccuracy.sampleErrors,indices=P.map((_,i)=>i);
  const series=[];
  states.forEach((s,k)=>{if(Math.max(...P.map(row=>row[k]))<1e-3)return;series.push({quantity:'population',state:s,indices,values:P.map(row=>row[k]),u,target:{absolute:0.05}});});
  for(const g of groups(states)){const ix=g.states.map(s=>states.indexOf(s));const v=P.map(row=>ix.reduce((a,k)=>a+row[k],0));if(Math.max(...v)<1e-3)continue;
    series.push({quantity:'group',label:g.label,states:g.states,indices,values:v,u,target:{absolute:0.05}});}
  series.push({quantity:'loss',indices,values:indices.map(()=>0),u:indices.map(()=>0),target:{absolute:0.05}});
  const e={id:`R0_${id}`,reference:'R0',kind:'pass-fail',title:`R0 静止极限：${id}（种群核心的 32 态 Lindblad 轨迹）`,request:`out/R0/${id}.json`,
    durationUs:still.options.durationUs,samples:still.options.samples,
    derivation:'m → ∞ (mass ×10⁶, 1e-3 µK) holds the atom at the focus: the motion decouples and the quantum-jump ensemble\'s mean is the Lindblad equation there, which the population core solves for the same model (static request) to its estimated error.',
    conditions:[{what:'held atom',value:HELD},{what:'population core status',value:r.populationAccuracy.status},{what:'core estimated maximum error',value:r.populationAccuracy.estimatedMaximumError,limit:'≤ 0.01'}],
    series,info:{coreMethod:r.diagnostics?.method,coreMs:performance.now()-started}};
  fs.mkdirSync(path.join(here,'expect'),{recursive:true});fs.writeFileSync(path.join(here,'expect',`R0_${id}.json`),JSON.stringify(e,null,1));
  console.log(`R0_${id}`,r.populationAccuracy.status,'max err',r.populationAccuracy.estimatedMaximumError.toExponential(2),`${(performance.now()-started).toFixed(0)} ms`,series.length,'series');
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const ids=process.argv.slice(2);
  for(const id of Object.keys(R0_SET))if(!ids.length||ids.includes(id))await run(id);
}
