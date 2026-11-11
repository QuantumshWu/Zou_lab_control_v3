// node r4d.mjs force | tz -- the two references that the app's worker cannot run (kind "tool": expect/R4D_FORCE.json,
// expect/R4D_TW2.json, written by expect.mjs; their derivations are there): they step the engine through the example
// binary (motion/examples/diffusion.rs: the model and the step as the solve calls them, nothing changed) beside the
// exact tools of refs/diffusion. Build it first:
//   cargo build -p atom-motion --example diffusion --release --target x86_64-pc-windows-msvc
// force (judged): the engine's mean force at constant velocities against the master equation of its own model,
//   python diffusion/uniform_engine.py <velocities> <trajectories> <span> <τ> 1 rows.json  (R4D_PYTHON, else `python`:
//   one with numpy and scipy), then check.mjs's rule 0 on F_engine − F_exact: the count beyond 2σ and the mean deviation,
//   σ² = SE² + u², u the harmonic balance's truncation. The momentum diffusion of the same runs is printed, not judged.
// tz (reported): T_z(t) of the 1D trapped problem, `diffusion free R4_TW2 1d ...`, against the same model's phase-space
//   equation and its master equation in the position representation (data/R4D_TW2.json: recorded runs of
//   diffusion/pde.py, exported by diffusion/export.py). Never judged.
// Each run is one command on all cores, killed at the limit of lib/worker.mjs (the user's 5 minutes); results in
// out/results/<id>.json.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {Z,consistency} from './check.mjs';
import {KT_PER_UK} from './lib/phys.mjs';
import {LIMIT_S} from './lib/worker.mjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const EX=path.resolve(here,'../../target/x86_64-pc-windows-msvc/release/examples/diffusion.exe');
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const scratch=path.join(here,'out','r4d'),results=path.join(here,'out','results');
for(const d of [scratch,results])fs.mkdirSync(d,{recursive:true});
function run(command,args){
  const started=Date.now(),r=spawnSync(command,args,{cwd:scratch,encoding:'utf8',timeout:LIMIT_S*1000,maxBuffer:1<<26});
  if(r.error||r.status!==0)throw new Error(`${command} ${args.join(' ')}: ${r.error?.message??`exit ${r.status}`}\n${(r.stderr??'').slice(-2000)}`);
  return (Date.now()-started)/1000;
}

function force(){
  const e=read(path.join(here,'expect','R4D_FORCE.json')),t=e.tool,rows=path.join(scratch,'force_rows.json');
  const wall=run(process.env.R4D_PYTHON??'python',[path.join(here,'diffusion','uniform_engine.py'),t.velocities.join(','),String(t.trajectories),String(t.spanUs),String(t.tauUs),'1',rows]);
  const out=read(rows),test=consistency(out.map(r=>({d:r.engine-r.exact,u:r.u,fixed:0,random:Z*r.se}))),verdict=test.reject?'FAIL':'PASS';
  console.log(`${e.id} [${e.reference}, ${e.kind}] ${e.title}`);
  console.log(`  ${verdict}: ${out.length} velocities, ${t.trajectories} trajectories each, τ ${t.tauUs} µs, ${wall.toFixed(0)} s; beyond 2σ: ${test.beyond} (n_eff ${test.neff.toFixed(1)}, p ${test.p.toPrecision(2)}${test.count?'!':''}); mean deviation ${test.dbar.toExponential(2)} against ${test.M.toExponential(2)}${test.common?'!':''}`);
  console.log('  v µm/µs    F exact       u    F engine      se   d/σ  |  2D exact  2D engine      se  (reported)');
  for(const r of out)console.log('  '+[r.v.toFixed(3).padStart(7),r.exact.toFixed(5).padStart(10),r.u.toExponential(1).padStart(8),r.engine.toFixed(5).padStart(10),r.se.toFixed(5).padStart(8),((r.engine-r.exact)/Math.hypot(r.se,r.u)).toFixed(2).padStart(6),
    ' | ',r.exact2D.toFixed(3).padStart(8),r.engine2D.toFixed(3).padStart(9),r.engine2Dse.toFixed(3).padStart(8),`  ${(100*(r.engine2D/r.exact2D-1)).toFixed(0)} %`].join(' '));
  fs.writeFileSync(path.join(results,`${e.id}.json`),JSON.stringify({id:e.id,verdict,wallS:wall,test,rows:out}));
  return verdict;
}

function tz(){
  const e=read(path.join(here,'expect','R4D_TW2.json')),t=e.tool,file=path.join(scratch,'free_1d.json'),data=read(path.join(here,t.data));
  const wall=run(EX,['free',t.scene,'1d',String(t.temperatureUK),String(t.tauUs),String(t.atoms),String(t.spanUs),String(t.everyUs),file]);
  // The rows of `diffusion free`: [t, ⟨κ_z²⟩, its standard error, ⟨κ_z⟩, ...] over the window that ends at t.
  const d=read(file),T=d.rows.map(r=>({t:r[0],T:d.hbar_over_mass*(r[1]-r[3]**2)/KT_PER_UK,se:d.hbar_over_mass*r[2]/KT_PER_UK}));
  const at=(series,x)=>{const i=series.timesUs.findIndex(y=>Math.abs(y-x)<1e-6);return i<0?NaN:series.TzUK[i];};
  console.log(`${e.id} [${e.reference}, ${e.kind}] ${e.title}`);
  console.log(`  REPORT (never judged): ${t.atoms} atoms, τ ${t.tauUs} µs, ${wall.toFixed(0)} s; x and y held at the trap's centre`);
  console.log('   t µs  engine T_z      se | 2nd-order eq.     gap | quantum (same model)     gap');
  const rows=[];
  for(const x of e.tool.shownUs){const r=T.find(q=>Math.abs(q.t-x)<1e-6);if(!r)continue;const b=at(data.secondOrder,x),q=at(data.quantum,x);rows.push({t:x,engine:r.T,se:r.se,secondOrder:b,quantum:q});
    console.log('  '+[String(x).padStart(5),r.T.toFixed(2).padStart(9),r.se.toFixed(2).padStart(7),'|',b.toFixed(2).padStart(13),`${(100*(r.T/b-1)).toFixed(1)} %`.padStart(8),'|',q.toFixed(2).padStart(13),`${(100*(r.T/q-1)).toFixed(1)} %`.padStart(14)].join(' '));}
  fs.writeFileSync(path.join(results,`${e.id}.json`),JSON.stringify({id:e.id,verdict:'REPORT',wallS:wall,rows}));
  return 'REPORT';
}

const which=process.argv[2];
if(!fs.existsSync(EX)){console.error(`no ${EX}: build the example first (this file's header)`);process.exitCode=2;}
else if(which==='force')process.exitCode=force()==='FAIL'?1:0;
else if(which==='tz')tz();
else{console.error('node r4d.mjs force | tz');process.exitCode=2;}
