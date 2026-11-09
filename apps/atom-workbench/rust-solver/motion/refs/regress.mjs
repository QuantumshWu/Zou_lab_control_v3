// node regress.mjs [import|requests|truths] -- the old landing gate as a regression net (contract §5.2, §7 Q2, Q10):
// - import: each gate scene's source (the app scene the gate's payload was made from; the Codex re-export's
//   <id>.scene.json) copied to regress/<scene>.scene.json, with its provenance in regress/sources.json and the check that
//   the app's re-export of it reproduces the gate's payload bit for bit (couplings, emission, H0, initial state): the old
//   truths' state-resolved populations are then in the same basis as the new requests. Scenes without a source are
//   listed there (they drop out).
// - requests: out/regress/<run>.json, each gate run's request as the app sends it now (the run's duration and samples;
//   a GPU run and its CPU twin share one truth and one request, run on the CPU).
// - truths: expect/regress/<run>.json, the run's old truth (G/truth/<run>.json: at its three outputs T per axis, loss and
//   the held atoms' populations, each value ± hypot(SE, last step residual)) as a regression expectation, classed by
//   contract §5.2: the anchors (T, populations, loss), the population/loss anchors, consistency data (reported, not
//   judged) and the excluded (wrong) truths; T where k_BT < 4ħω (the truth's weight w < 1) is never a physical reference.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {finish,motionRequest} from './lib/app.mjs';
import {checkRequest} from './lib/contract.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const SP='C:/Users/eadri/AppData/Local/Temp/claude/C--Users-eadri-Dropbox-WorkCode-Github-Zou-lab-control-v3/8308d66c-699b-4483-ab9b-7305d93a9f7e/scratchpad';
const NS='C:/Users/eadri/AppData/Local/Temp/claude/c0815e347c54866c/8308d66c-699b-4483-ab9b-7305d93a9f7e/scratchpad';
const GATE=`${SP}/gate`,CODEX=`${NS}/codex_resume/scenes`;
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const write=(f,v)=>{fs.mkdirSync(path.dirname(f),{recursive:true});fs.writeFileSync(f,JSON.stringify(v,null,1));};
const sha=s=>createHash('sha256').update(s).digest('hex');

/** The gate's scenes and where their sources are: Codex's re-export kept each source as <name>.scene.json. F9_CROSSED_P
 * is F9_CROSSED's scene (the gate's only payload already with fields); F2's truth ran the payload
 * validate/repro/F2_default_plus_weak_D1_beam.json, whose source Codex kept as WEAK_D1_FAR. */
const SOURCE={F9_CROSSED_P:'F9_CROSSED',F2:'WEAK_D1_FAR'};
const runs=()=>{const r=read(`${GATE}/runs.json`).runs;return [...r,{id:'F2_100us',scene:'F2',durationUs:100,samples:21,device:'cpu'}];};

/** contract §5.2. */
const WRONG={R5059_cpu:'热原子穿越 D1 F′=2 共振壳（a ≈ 1.3–1.9），折叠不成立；全显式 X8 给出 10 ms 损失 0.228，真值 0.383',
  F9_CROSSED:'旧载荷不带场，秩 2 的远光按标量拟合，app 从不发这种请求'};
const ANCHOR=['RETRO_100us','F5_RETRO80','R5051_cpu','R5031_cpu','SHALLOW_PROBE','F10_28MHz','CONTROL_CYCLE','R5038_cpu'];
const ANCHOR_P=['R5033_cpu','R5066_cpu','R5040_cpu','R5004_cpu'];
const SUSPICIOUS_P=['PGCP_10_100us_cpu','PGCP_10_100us_gpu'];

async function importSources(){
  const scenes=[...new Set(runs().map(r=>r.scene))],sources={},missing=[];
  for(const id of scenes){
    const name=SOURCE[id]??id,file=`${CODEX}/${name}.scene.json`;
    if(!fs.existsSync(file)){missing.push(id);continue;}
    const text=fs.readFileSync(file,'utf8'),scene=JSON.parse(text);
    // The gate's payload, against which the re-export is compared (the gate's own copy; F2's from validate/repro).
    const gate=id==='F2'?`${SP}/validate/repro/F2_default_plus_weak_D1_beam.json`:`${GATE}/scenes/${id}.json`;
    const old=read(gate),now=await motionRequest(id,await finish(structuredClone(scene)),{device:'cpu'});
    const same=['hReal','hImag','initialReal','initialImag'].every(k=>JSON.stringify(old.model[k])===JSON.stringify(now.model[k]))&&
      JSON.stringify(old.model.motion.emission)===JSON.stringify(now.model.motion.emission)&&
      old.model.motion.beams.every(b=>{const n=now.model.motion.beams.find(x=>x.id===b.id);return n&&JSON.stringify(b.coupling)===JSON.stringify(n.coupling)&&JSON.stringify(b.counterCoupling)===JSON.stringify(n.counterCoupling);});
    write(path.join(here,'regress',`${id}.scene.json`),scene);
    sources[id]={source:file,sha256:sha(text),gatePayload:gate,reproducesGatePayload:same};
    console.log(id.padEnd(16),name.padEnd(12),same?'bit-identical model':'MODEL DIFFERS');
  }
  write(path.join(here,'regress','sources.json'),{note:'Sources of the old gate scenes (contract §7 Q2): Codex re-export (codex_resume/export_scenes.mjs) kept each as <name>.scene.json. reproducesGatePayload: the app of this worktree re-exports the gate payload bit for bit (couplings, emission, H0, initial state), so the old truths compare state by state.',sources,missing});
  console.log('missing sources:',missing.length?missing.join(', '):'none');
}

/** The CPU runs (a GPU run's truth is its CPU twin's; a GPU-only run is run on the CPU). */
function cpuRuns(){
  const seen=new Map();
  for(const r of runs()){const key=`${r.scene}:${r.durationUs}:${r.samples}`;if(!seen.has(key)||r.device==='cpu')seen.set(key,r);}
  return [...seen.values()];
}
async function requests(){
  for(const r of cpuRuns()){
    const file=path.join(here,'regress',`${r.scene}.scene.json`);if(!fs.existsSync(file)){console.log(r.id,'no source');continue;}
    const src=read(file),s=await finish({...src,dynamics:{...src.dynamics,durationUs:r.durationUs,samples:r.samples}});
    const line=await motionRequest(r.id,s,{device:'cpu'});checkRequest(line);
    write(path.join(here,'out','regress',`${r.id}.json`),line);console.log(r.id,r.durationUs,r.samples);
  }
}

function truths(){
  const out={};
  for(const r of cpuRuns()){
    const file=`${GATE}/truth/${r.id}.json`;if(!fs.existsSync(file)){console.log(r.id,'no truth');continue;}
    const t=read(file),kind=WRONG[r.id]?'excluded':ANCHOR.includes(r.id)?'regression-anchor':ANCHOR_P.includes(r.id)?'regression-anchor-populations':'regression-consistency';
    const u=x=>Math.hypot(x.se,x.residual??0);
    const series=[];
    const judged=kind==='regression-anchor'||kind==='regression-anchor-populations';
    // T per axis where the truth's regime weight w ≥ 1 (k_BT ≥ 4ħω): judged only for the full anchors.
    for(let a=0;a<3;a++){
      const outs=t.outputs.filter(o=>o.T&&o.T[a]&&Number.isFinite(o.T[a].value)&&o.held>0);
      if(outs.length)series.push({quantity:'T',axis:'xyz'[a],indices:outs.map(o=>o.index),values:outs.map(o=>o.T[a].value),u:outs.map(o=>u(o.T[a])),
        weights:outs.map(o=>o.weight),judged:outs.map(o=>kind==='regression-anchor'&&o.weight>=1),target:{relative:0.10}});
    }
    series.push({quantity:'loss',indices:t.outputs.map(o=>o.index),values:t.outputs.map(o=>o.loss.value),u:t.outputs.map(o=>u(o.loss)),judged:t.outputs.map(()=>judged),target:{absolute:0.05}});
    const states=[...new Set(t.outputs.flatMap(o=>Object.keys(o.populations)))];
    for(const s of states){
      const outs=t.outputs.filter(o=>o.populations[s]&&o.held>0);
      series.push({quantity:'population',state:s,indices:outs.map(o=>o.index),values:outs.map(o=>o.populations[s].value),u:outs.map(o=>u(o.populations[s])),
        judged:outs.map(()=>judged&&!SUSPICIOUS_P.includes(r.id)),target:{absolute:0.05}});
    }
    const e={id:r.id,reference:'regress',kind,title:`旧门 ${r.id} 的真值（线头 ${t.lineHead.commit} 的模型 ${t.lineHead.model} 收敛值）`,
      request:`out/regress/${r.id}.json`,durationUs:r.durationUs,samples:r.samples,
      derivation:'The old engine\'s own model at its line head, run to convergence (gate/truth/README.md): a regression value, not a physical one. u = hypot(SE, last step-halving residual).',
      excluded:WRONG[r.id]??null,series,info:{lineHead:t.lineHead,truthLevel:t.truthLevel,converged:t.converged}};
    write(path.join(here,'expect','regress',`${r.id}.json`),e);out[r.id]=kind;
  }
  console.log(JSON.stringify(out,null,1));
}

const what=process.argv[2];
if(!what||what==='import')await importSources();
if(!what||what==='requests')await requests();
if(!what||what==='truths')truths();
