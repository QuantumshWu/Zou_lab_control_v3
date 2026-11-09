// The native worker (rust-solver/src main.rs, `atom-lindblad`): one JSONL request on stdin, its checkpoint lines
// ({progress:"checkpoint", result}) then one final line ({result} or {error}) on stdout (contract §2.1). The same binary
// serves the population core (a request without a motion block) and the motion solver (with one) -- the app's exact
// path, no second serializer. A run longer than the limit (the user's 5-minute rule) is killed and its last
// checkpoint kept as a partial result.
//
// The binary: ATOM_LINDBLAD, else this worktree's release build target/x86_64-pc-windows-msvc/release/atom-lindblad.exe,
// else its iterate build; LAPACK from ATOM_OPENBLAS_DLL (as the cargo script sets it).
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const target=path.resolve(here,'../../../target/x86_64-pc-windows-msvc');
export const WORKER=process.env.ATOM_LINDBLAD??['release','iterate'].map(p=>path.join(target,p,'atom-lindblad.exe')).find(f=>fs.existsSync(f));
/** The user's limit for a single run (seconds): over it a run is aborted and reported as a finding. */
export const LIMIT_S=300;

/** Runs one request line; resolves {final, checkpoints:[{s, result}], wallS, aborted, stderr}. `final` is the worker's
 * last line ({result} or {error}); when aborted, the last checkpoint stands in for it. */
export function runWorker(line,{limitS=LIMIT_S,progress=true}={}){
  if(!WORKER)throw new Error('No atom-lindblad worker: build it (cargo build --release --target x86_64-pc-windows-msvc -p atom-lindblad) or set ATOM_LINDBLAD');
  return new Promise((resolve,reject)=>{
    const started=performance.now(),child=spawn(WORKER,[],{stdio:['pipe','pipe','pipe'],windowsHide:true});
    const checkpoints=[];let buffer='',stderr='',final=null,aborted=false;
    const timer=setTimeout(()=>{aborted=true;child.kill();},limitS*1000);
    child.stdout.on('data',d=>{buffer+=d;let nl;while((nl=buffer.indexOf('\n'))>=0){const text=buffer.slice(0,nl);buffer=buffer.slice(nl+1);if(!text.trim())continue;
      const msg=JSON.parse(text),s=(performance.now()-started)/1000;
      if(msg.progress==='checkpoint')checkpoints.push({s,result:msg.result});else{final=msg;child.stdin.end();}}});
    child.stderr.on('data',d=>{stderr+=d;});
    child.on('error',e=>{clearTimeout(timer);reject(e);});
    child.on('close',()=>{clearTimeout(timer);const wallS=(performance.now()-started)/1000;
      if(!final&&!aborted)return reject(new Error(`worker ended without a result (stderr: ${stderr.slice(-400)})`));
      resolve({final:final??(checkpoints.length?{result:checkpoints.at(-1).result,partial:'aborted at the limit'}:{error:'aborted at the limit before any checkpoint'}),
        checkpoints:checkpoints.map(c=>({s:c.s})),wallS,aborted,stderr});});
    child.stdin.write(JSON.stringify({...line,progress})+'\n');
  });
}
