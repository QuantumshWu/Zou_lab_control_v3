// The native worker (rust-solver/src main.rs, `atom-lindblad`): one JSONL request on stdin, its progress lines then one
// final line ({result} or {error}) on stdout (contract §2.1). The same binary serves the population core (a request
// without a motion block) and the motion solver (with one) -- the app's exact path, no second serializer. A run longer
// than the limit (the user's 5-minute rule) is killed and its last checkpoint kept as a partial result.
//
// The progress lines are read as the app reads them (scripts/serve.mjs, src/solver-client.ts): a line with `progress`
// is never the final line. `checkpoint` carries a result; `stage` (motion_adapter.rs `stage`: where a motion solve's
// pipeline stands) carries no result and is read by the app's own `readStage` (src/solver-types.ts, from this worktree's
// dist, compiled by lib/app.mjs when a source is newer); any other progress, or a stage the app cannot read, fails the
// solve as it fails the app's (the worker is stopped, the run's final line is that error). The app's transport deadline
// (120 s) restarts on a checkpoint and on a stage alike (desktop-native/ui/native-fetch.mjs).
//
// The binary: ATOM_LINDBLAD, else this worktree's release build target/x86_64-pc-windows-msvc/release/atom-lindblad.exe,
// else its iterate build; LAPACK from ATOM_OPENBLAS_DLL (as the cargo script sets it).
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const target=path.resolve(here,'../../../target/x86_64-pc-windows-msvc');
export const WORKER=process.env.ATOM_LINDBLAD??['release','iterate'].map(p=>path.join(target,p,'atom-lindblad.exe')).find(f=>fs.existsSync(f));
/** The user's limit for a single run (seconds): over it a run is aborted and reported as a finding. */
export const LIMIT_S=300;

/** The app's reader of a stage (src/solver-types.ts `readStage`), loaded once. */
let stageReader;
const readStage=()=>stageReader??=import('./app.mjs').then(async app=>{await app.build();
  return (await import(pathToFileURL(path.join(app.APP,'dist/js/atom-workbench/src/solver-types.js')).href)).readStage;});

/** Runs one request line; resolves {final, checkpoints:[{s}], stages:[{s, stage}], wallS, aborted, stderr} (s: seconds
 * from the request). `final` is the worker's last line ({result} or {error}); when aborted, the last checkpoint stands in
 * for it; a progress line the app would refuse makes it {error} (the module's header). */
export async function runWorker(line,{limitS=LIMIT_S,progress=true}={}){
  if(!WORKER)throw new Error('No atom-lindblad worker: build it (cargo build --release --target x86_64-pc-windows-msvc -p atom-lindblad) or set ATOM_LINDBLAD');
  const read=progress?await readStage():null;
  return new Promise((resolve,reject)=>{
    const started=performance.now(),child=spawn(WORKER,[],{stdio:['pipe','pipe','pipe'],windowsHide:true});
    const checkpoints=[],stages=[];let buffer='',stderr='',final=null,aborted=false;
    const timer=setTimeout(()=>{aborted=true;child.kill();},limitS*1000);
    const refuse=why=>{if(final)return;final={error:why};clearTimeout(timer);child.kill();};
    child.stdout.on('data',d=>{buffer+=d;let nl;while((nl=buffer.indexOf('\n'))>=0){const text=buffer.slice(0,nl);buffer=buffer.slice(nl+1);if(!text.trim()||final)continue;
      const msg=JSON.parse(text),s=(performance.now()-started)/1000;
      if(msg.progress==='checkpoint')checkpoints.push({s,result:msg.result});
      else if(msg.progress==='stage'){try{read(msg.stage);}catch{refuse(`求解进度无效：the app cannot read the stage ${JSON.stringify(msg.stage).slice(0,200)}`);continue;}stages.push({s,stage:msg.stage});}
      else if(msg.progress!==undefined)refuse(`Unexpected native progress message: ${JSON.stringify(msg.progress).slice(0,80)}`);
      else{final=msg;child.stdin.end();}}});
    child.stderr.on('data',d=>{stderr+=d;});
    child.on('error',e=>{clearTimeout(timer);reject(e);});
    child.on('close',()=>{clearTimeout(timer);const wallS=(performance.now()-started)/1000;
      if(!final&&!aborted)return reject(new Error(`worker ended without a result (stderr: ${stderr.slice(-400)})`));
      resolve({final:final??(checkpoints.length?{result:checkpoints.at(-1).result,partial:'aborted at the limit'}:{error:'aborted at the limit before any checkpoint'}),
        checkpoints:checkpoints.map(c=>({s:c.s})),stages,wallS,aborted,stderr});});
    child.stdin.write(JSON.stringify({...line,progress})+'\n');
  });
}
