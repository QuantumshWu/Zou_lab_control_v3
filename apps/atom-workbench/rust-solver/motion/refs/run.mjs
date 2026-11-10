// node run.mjs <id>... | --r | --r0 | --regress [--limit=<s>] -- runs references through the app's own worker
// (lib/worker.mjs: the request line on stdin, its checkpoints and the final result on stdout, contract §2.1) and keeps
// each run in out/results/<id>.json: the final line (or, aborted at the limit, the last checkpoint), every checkpoint's
// wall time (the app kills a solve that sends none for 120 s, contract §2.5), the wall time. An id names an expectation
// (expect/<id>.json or expect/regress/<id>.json), whose `request` is run. Runs go one at a time on all cores; a run over
// the limit (5 minutes, the user's rule) is aborted and reported as a finding. Then: node check.mjs <id>...
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {LIMIT_S,WORKER,runWorker} from './lib/worker.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const expectationFile=id=>[path.join(here,'expect',`${id}.json`),path.join(here,'expect','regress',`${id}.json`)].find(f=>fs.existsSync(f));
const listed=folder=>fs.existsSync(folder)?fs.readdirSync(folder).filter(f=>f.endsWith('.json')).map(f=>f.slice(0,-5)):[];

const args=process.argv.slice(2),limit=Number(args.find(a=>a.startsWith('--limit='))?.slice(8)??LIMIT_S);
let ids=args.filter(a=>!a.startsWith('--'));
if(args.includes('--r'))ids.push(...listed(path.join(here,'expect')).filter(id=>/^R[1-9]/.test(id)));
if(args.includes('--r0'))ids.push(...listed(path.join(here,'expect')).filter(id=>id.startsWith('R0_')));
if(args.includes('--regress'))ids.push(...listed(path.join(here,'expect','regress')));
console.log(`worker ${WORKER}, limit ${limit} s, ${ids.length} runs`);
for(const id of ids){
  const file=expectationFile(id);if(!file){console.log(id,'no expectation');continue;}
  const e=JSON.parse(fs.readFileSync(file,'utf8'));
  if(e.excluded){console.log(id,'excluded:',e.excluded);continue;}
  if(e.members){console.log(id,'a relation of',e.members.map(m=>m.id).join(', '),'(run under their own ids)');continue;}
  const line=JSON.parse(fs.readFileSync(path.join(here,e.request),'utf8'));
  const run=await runWorker(line,{limitS:limit});
  const out={id,request:e.request,worker:WORKER,wallS:run.wallS,aborted:run.aborted,firstCheckpointS:run.checkpoints[0]?.s??null,checkpoints:run.checkpoints,final:run.final,stderr:run.stderr.slice(-2000)};
  fs.mkdirSync(path.join(here,'out','results'),{recursive:true});fs.writeFileSync(path.join(here,'out','results',`${id}.json`),JSON.stringify(out));
  const r=run.final.result;
  console.log(id.padEnd(22),run.final.error?`ERROR ${run.final.error}`:`${r.populationAccuracy?.status} traj ${r.motion?.trajectories??'-'}`,
    `${run.wallS.toFixed(1)} s`,run.aborted?'ABORTED (limit)':'',`checkpoints ${run.checkpoints.length}`,run.checkpoints.length?`first ${run.checkpoints[0].s.toFixed(1)} s`:'');
}
