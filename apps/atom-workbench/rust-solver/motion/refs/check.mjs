// node check.mjs <id>... | --r | --r0 | --regress [--result=<file>] -- applies the references' acceptance rules (contract
// §6, SPEC §6) to the solver's results (out/results/<id>.json from run.mjs, or any file holding the contract's result:
// the bare result, a native line {result}, or run.mjs's {final:{result}}) and prints a compact table.
//
// Per output (a series' value at an output index): d = solver − reference, σ = √(SE² + u²) with u the reference's
// standard uncertainty and SE the solver's statistical standard error:
//   T            temperatureErrorUK (the result's own SE; null: no estimate, the output is not judged);
//   loss         √(L(1 − L)/N), N = motion.trajectories (a Bernoulli mean);
//   population   √(p(1 − p)/N_held), N_held = N(1 − L): the variance of a [0, 1]-valued mean is at most p(1 − p), so
//   and group    this bounds SE from above -- it can only make the first rule more lenient, by at most 1/√N_held;
//   meanVelocity √(k_B T_a/m)/√N_held; meanPosition the distribution's own standard deviation over √N_held;
//   trapFrequency 0 (deterministic).
// Rule 1 (accuracy): fail where |d| − 2σ > target, the user's targets unrelaxed: populations, groups and loss 0.05,
//   T 10 % of the reference; a derived quantity carries its own target in the expectation (trap frequencies 0: they
//   must agree within the reference's uncertainty).
// Rule 2 (coverage): fail where more than 5 % of the outputs have |d| > c + 2u, c the solver's own error band at z = 2:
//   populations, groups and loss the result's sampleErrors (the total error E, statistics at z = 2: "2 × the
//   self-reported error" with the self-reported error E/2), T hypot(2 SE, ledger total × T) (the T error the result
//   reports is statistical only; the ledger's totals.temperature.error, relative, holds the rest). Derived quantities
//   the result reports no error for (trap frequency, mean velocity, mean position) are judged by rule 1 alone.
// Power: an output with σ > 0.3 × target is underpowered (contract §6: run enough trajectories for σ ≤ 0.3 target);
// it is judged, but a pass there says little. T series marked regime "classical" are judged only where the result's
// `classical` flag is true; outputs an expectation marks unjudged (old truths outside their class, T below 4ħω) are
// reported, not judged. Kinds "characterization" and "regression-consistency" are reported, never failed.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const KB=1.380649e-23,Z=2,SHARE=0.05,POWER=0.3;
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const expectationFile=id=>fs.existsSync(id)?id:[path.join(here,'expect',`${id}.json`),path.join(here,'expect','regress',`${id}.json`)].find(f=>fs.existsSync(f));
const resultOf=v=>v.final?.result??v.result??(v.populations?v:null);

/** One expectation against one result: rows per series, totals, the verdict. */
export function check(e,result,request){
  const r=result,m=r.motion??{},N=m.trajectories??NaN,loss=m.lossProbability??[],errors=r.populationAccuracy?.sampleErrors??[];
  const tLedger=r.diagnostics?.approximations?.totals?.temperature?.error??0,mass=request?.model?.motion?.massKg;
  const k=s=>r.states.indexOf(s),held=i=>N*(1-(loss[i]??0));
  const rows=[];let judged=0,exceeded=0,failed=0,under=0,skipped=0;
  if(r.timesUs?.length!==e.samples||Math.abs(r.timesUs.at(-1)-e.durationUs)>1e-9*e.durationUs)return {error:`grid: ${r.timesUs?.length} times to ${r.timesUs?.at(-1)} µs, expected ${e.samples} to ${e.durationUs}`};
  for(const s of e.series){
    const name=s.quantity+(s.axis?` ${s.axis}`:'')+(s.state?` ${s.state}`:'')+(s.label?` ${s.label}`:'');
    const outputs=s.indices?s.indices.map((i,j)=>({i,ref:s.values[j],u:s.u[j],judge:s.judged?s.judged[j]:true})):[{i:null,ref:s.value,u:s.u,judge:true}];
    const row={name,n:0,maxD:0,worst:-Infinity,worstAt:null,fail:0,exceed:0,under:0,skipped:0};
    for(const o of outputs){
      let x,se,band=null;const i=o.i;
      switch(s.quantity){
        case 'population':x=r.populations[i][k(s.state)];se=Math.sqrt(x*(1-x)/held(i));band=errors[i];break;
        case 'group':x=s.states.reduce((t,st)=>t+r.populations[i][k(st)],0);se=Math.sqrt(Math.max(0,x*(1-x))/held(i));band=errors[i];break;
        case 'loss':x=loss[i];se=Math.sqrt(x*(1-x)/N);band=errors[i];break;
        case 'T':{x=m.temperatureUK[s.axis][i];const t=m.temperatureErrorUK[s.axis][i];se=t==null?Infinity:t;band=Math.hypot(Z*se,tLedger*x);
          if(s.regime==='classical'&&!m.classical[i])o.judge=false;break;}
        case 'trapFrequency':x=m.trapFrequencyKHz[s.axis];se=0;break;
        case 'meanVelocity':x=m.meanVelocityMps[s.axis][i];se=Math.sqrt(KB*m.temperatureUK[s.axis][i]*1e-6/mass)/Math.sqrt(held(i));break;
        case 'meanPosition':{const p=m.position?.[s.axis],j=p?.samples.indexOf(i);if(!p||j<0){x=undefined;break;}
          const w=p.probabilities[j],mean=w.reduce((t,q,c)=>t+q*p.positionsUm[c],0),v=w.reduce((t,q,c)=>t+q*(p.positionsUm[c]-mean)**2,0);x=mean;se=Math.sqrt(v/held(i));break;}
        case 'meanPositionAverage':{const p=m.position?.[s.axis];if(!p){x=undefined;break;}
          // The mean over the distributions after t = 0 of their means; its SE bounded by the mean of theirs (fully correlated).
          const rs=p.samples.map((i2,j)=>[i2,p.probabilities[j]]).filter(([i2])=>i2>0).map(([i2,w])=>{const mean=w.reduce((t,q,c)=>t+q*p.positionsUm[c],0);
            return [mean,Math.sqrt(w.reduce((t,q,c)=>t+q*(p.positionsUm[c]-mean)**2,0)/held(i2))];});
          x=rs.reduce((t,q)=>t+q[0],0)/rs.length;se=rs.reduce((t,q)=>t+q[1],0)/rs.length;break;}
        default:throw new Error(`unknown quantity ${s.quantity}`);
      }
      if(x===undefined||!Number.isFinite(se)&&s.quantity!=='T'){continue;}
      if(!o.judge||!Number.isFinite(se)){row.skipped++;skipped++;continue;}
      const target=s.target.absolute??(s.target.relative*Math.abs(o.ref)),d=x-o.ref,sigma=Math.hypot(se,o.u);
      row.n++;judged++;row.maxD=Math.max(row.maxD,Math.abs(d));
      const score=target>0?(Math.abs(d)-Z*sigma)/target:(Math.abs(d)-Z*sigma);
      if(score>row.worst){row.worst=score;row.worstAt={i,t:r.timesUs[i??0],x,ref:o.ref,se,u:o.u};}
      if(Math.abs(d)-Z*sigma>target){row.fail++;failed++;}
      if(band!==null){if(Math.abs(d)>band+Z*o.u){row.exceed++;exceeded++;}}
      if(target>0&&sigma>POWER*target){row.under++;under++;}
    }
    rows.push(row);
  }
  const covered=e.series.filter(s=>['population','group','loss','T'].includes(s.quantity)).length?exceeded/Math.max(1,rows.filter(r=>/^(population|group|loss|T)/.test(r.name)).reduce((t,r)=>t+r.n,0)):0;
  const judgedKind=!['characterization','regression-consistency'].includes(e.kind);
  const verdict=!judged?'NOT JUDGED':!judgedKind?'REPORT':failed||covered>SHARE?'FAIL':'PASS';
  return {rows,judged,failed,exceeded,coverage:covered,under,skipped,verdict};
}

function print(e,c,run){
  const head=`${e.id} [${e.reference}, ${e.kind}] ${e.title}`;
  if(c.error){console.log(`${head}\n  ${c.error}\n`);return;}
  console.log(`${head}\n  ${c.verdict}: ${c.judged} outputs judged, ${c.failed} beyond target (rule 1), ${(100*c.coverage).toFixed(1)} % outside the solver's own error (rule 2, ≤ 5 %), ${c.under} underpowered, ${c.skipped} not judged`+
    (run?` | run ${run.wallS.toFixed(1)} s${run.aborted?' ABORTED':''}, first checkpoint ${run.firstCheckpointS?.toFixed(1)??'-'} s, ${run.final?.result?.motion?.trajectories??'-'} trajectories`:''));
  console.log('  '+['series'.padEnd(30),'n'.padStart(4),'max|d|'.padStart(10),'worst (|d|−2σ)/tgt'.padStart(20),'fail'.padStart(5),'out'.padStart(5),'weak'.padStart(5),'  worst output (t: solver vs ref ± se, u)'].join(''));
  for(const r of c.rows){const w=r.worstAt;
    console.log('  '+[r.name.slice(0,30).padEnd(30),String(r.n).padStart(4),r.maxD.toPrecision(3).padStart(10),(Number.isFinite(r.worst)?r.worst.toFixed(2):'-').padStart(20),String(r.fail).padStart(5),String(r.exceed).padStart(5),String(r.under).padStart(5),
      w?`  t=${w.t.toPrecision(4)}: ${w.x.toPrecision(5)} vs ${w.ref.toPrecision(5)} ± ${w.se.toPrecision(2)}, ${w.u.toPrecision(2)}`:''].join(''));}
  console.log('');
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const args=process.argv.slice(2),one=args.find(a=>a.startsWith('--result='))?.slice(9);
  const listed=f=>fs.existsSync(f)?fs.readdirSync(f).filter(x=>x.endsWith('.json')).map(x=>x.slice(0,-5)):[];
  let ids=args.filter(a=>!a.startsWith('--'));
  if(args.includes('--r'))ids.push(...listed(path.join(here,'expect')).filter(id=>/^R[1-9]/.test(id)));
  if(args.includes('--r0'))ids.push(...listed(path.join(here,'expect')).filter(id=>id.startsWith('R0_')));
  if(args.includes('--regress'))ids.push(...listed(path.join(here,'expect','regress')));
  const summary=[];let bad=0;
  for(const id of ids){
    const ef=expectationFile(id);if(!ef){console.log(id,'no expectation');continue;}
    const e=read(ef),rf=one??path.join(here,'out','results',`${e.id}.json`);
    if(!fs.existsSync(rf)){summary.push([e.id,'NO RESULT']);continue;}
    const raw=read(rf),r=resultOf(raw);if(!r){summary.push([e.id,raw.final?.error?`ERROR ${raw.final.error}`:'NO RESULT']);continue;}
    const c=check(e,r,read(path.join(here,e.request)));print(e,c,raw.final?raw:null);summary.push([e.id,c.error?'GRID MISMATCH':c.verdict]);if(c.verdict==='FAIL'||c.error)bad++;
  }
  if(summary.length>1){console.log('summary');for(const [id,v] of summary)console.log(`  ${id.padEnd(26)} ${v}`);}
  process.exitCode=bad?1:0;
}
