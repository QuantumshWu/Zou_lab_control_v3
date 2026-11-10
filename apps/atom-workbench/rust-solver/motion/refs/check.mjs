// node check.mjs <id>... | --r | --r0 | --regress [--result=<file>] [--results=<file>,<file>,...] -- applies the
// references' acceptance rules (contract §6, SPEC §6) to the solver's results (out/results/<id>.json from run.mjs, or
// any file holding the contract's result: the bare result, a native line {result}, or run.mjs's {final:{result}}) and
// prints a compact table. A relation (kind "relation": one statement about several scenes' results, `relation` below)
// reads its members' results (out/results/<member>.json, or --results in the members' order).
//
// Per output (a series' value at an output index): d = solver − reference, σ = √(SE² + u²) with u the reference's
// standard uncertainty and SE the solver's statistical standard error:
//   T            temperatureErrorUK (the result's own SE; null: no estimate, the output is not judged);
//   loss         √(L(1 − L)/N), N = motion.trajectories (a Bernoulli mean);
//   population   √(p(1 − p)/N_held), N_held = N(1 − L): the variance of a [0, 1]-valued mean is at most p(1 − p), so
//   and group    this bounds SE from above -- it can only make the first rule more lenient, by at most 1/√N_held;
//   meanVelocity √(k_B T_a/m)/√N_held; meanPosition(Average) the distribution's own spread over √N_held;
//   trapFrequency 0 (deterministic).
// Rule 1 (accuracy): fail where |d| − zσ > target (z = 2), the user's targets unrelaxed: populations, groups and loss
//   0.05, T 10 % of the reference; a derived quantity carries its target in the expectation (trap frequencies 0).
// Rule 2 (coverage, SPEC §6): an output exceeds when |d| > 2·(reported error ⊕ ledger total) + 2u -- populations,
//   groups and loss: the result's sampleErrors (their total, ledger included); T: temperatureErrorUK ⊕ the ledger's
//   totals.temperature.error × T. Per series the exceedances are counted against effectively independent outputs:
//   from the standardized residuals z_i = d_i/σ_i in time order, the autocorrelation ρ_k, n_eff = n/(1 + 2Σρ_k) summed
//   to the first non-positive ρ_k, and the count scaled by n_eff/n; a one-sided binomial test (Bin(n_eff, 5 %), the
//   regularized incomplete beta for non-integer counts) fails the series when the tail probability is below Φ(−z).
//   Quantities the result reports no error for (trap frequency, mean velocity and position) are judged by rule 1 only.
// Power: an output with σ > 0.3 × target is underpowered (contract §6: enough trajectories for σ ≤ 0.3 target): judged,
// but a pass there says little. T series of regime "classical" are judged only where the result's `classical` is
// true; outputs an expectation marks unjudged (old truths outside their class, T below 4ħω) are reported, not judged.
// Kinds "characterization" and "regression-consistency" are reported, never failed.
// A series marked `reported` (another engine's old numeric result, an axis whose share a symmetry leaves open) is
// compared and printed like any other, marked *, and never judged: its outputs enter no total and fail nothing.
// T and Tincrease take one axis, or `axes`: their mean, its SE bounded by the mean of theirs (fully correlated).
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const KB=1.380649e-23,HBAR=6.62607015e-34/(2*Math.PI),Z=2,SHARE=0.05,POWER=0.3;
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const expectationFile=id=>fs.existsSync(id)?id:[path.join(here,'expect',`${id}.json`),path.join(here,'expect','regress',`${id}.json`)].find(f=>fs.existsSync(f));
const resultOf=v=>v.final?.result??v.result??(v.populations?v:null);

// ln Γ (Lanczos, g = 7, n = 9: ~1e-15) and the regularized incomplete beta I_x(a, b) (Numerical Recipes betacf).
const LANCZOS=[0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
function lnGamma(x){if(x<0.5)return Math.log(Math.PI/Math.sin(Math.PI*x))-lnGamma(1-x);x-=1;let a=LANCZOS[0];const t=x+7.5;for(let i=1;i<9;i++)a+=LANCZOS[i]/(x+i);return 0.5*Math.log(2*Math.PI)+(x+0.5)*Math.log(t)-t+Math.log(a);}
function betacf(a,b,x){const tiny=1e-300;let c=1,d=1-(a+b)*x/(a+1);if(Math.abs(d)<tiny)d=tiny;d=1/d;let h=d;
  for(let m=1;m<500;m++){const m2=2*m;let aa=m*(b-m)*x/((a+m2-1)*(a+m2));d=1+aa*d;if(Math.abs(d)<tiny)d=tiny;c=1+aa/c;if(Math.abs(c)<tiny)c=tiny;d=1/d;h*=d*c;
    aa=-(a+m)*(a+b+m)*x/((a+m2)*(a+m2+1));d=1+aa*d;if(Math.abs(d)<tiny)d=tiny;c=1+aa/c;if(Math.abs(c)<tiny)c=tiny;d=1/d;const del=d*c;h*=del;if(Math.abs(del-1)<1e-15)break;}
  return h;}
function betai(a,b,x){if(x<=0)return 0;if(x>=1)return 1;const bt=Math.exp(lnGamma(a+b)-lnGamma(a)-lnGamma(b)+a*Math.log(x)+b*Math.log(1-x));
  return x<(a+1)/(a+b+2)?bt*betacf(a,b,x)/a:1-bt*betacf(b,a,1-x)/b;}
/** P(X ≥ k) for X ~ Bin(n, p), continued to real k, n: I_p(k, n − k + 1). */
export const binomialTail=(k,n,p)=>k<=0?1:k>n?0:betai(k,n-k+1,p);
/** Φ(−z), the normal's upper tail (erfc by its continued fraction is not needed at z = 2: Abramowitz–Stegun 7.1.26 is
 * too coarse, so by the series of erf). */
function upperTail(z){const x=z/Math.SQRT2;let term=x,sum=x;for(let n=1;n<200;n++){term*=-x*x/n;const add=term/(2*n+1);sum+=add;if(Math.abs(add)<1e-17)break;}return 0.5*(1-2/Math.sqrt(Math.PI)*sum);}
/** The effective number of independent outputs of a residual series: n/(1 + 2Σρ_k), summed to the first ρ_k ≤ 0. */
export function effectiveCount(z){
  const n=z.length;if(n<3)return n;const mean=z.reduce((a,b)=>a+b,0)/n,c0=z.reduce((a,b)=>a+(b-mean)**2,0);if(!(c0>0))return 1;
  let s=0;for(let k=1;k<n;k++){let c=0;for(let i=0;i+k<n;i++)c+=(z[i]-mean)*(z[i+k]-mean);const rho=c/c0;if(!(rho>0))break;s+=rho;}
  return Math.min(n,Math.max(1,n/(1+2*s)));
}

/** One expectation against one result: rows per series, totals, the verdict. */
export function check(e,result,request){
  const r=result,m=r.motion??{},N=m.trajectories??NaN,loss=m.lossProbability??[],errors=r.populationAccuracy?.sampleErrors??[];
  const tLedger=r.diagnostics?.approximations?.totals?.temperature?.error??0,mass=request?.model?.motion?.massKg;
  const k=s=>r.states.indexOf(s),held=i=>N*(1-(loss[i]??0)),alpha=upperTail(Z);
  const temperature=(s,i)=>{const axes=s.axes??[s.axis],se=axes.map(a=>m.temperatureErrorUK[a][i]);
    return {x:axes.reduce((t,a)=>t+m.temperatureUK[a][i],0)/axes.length,se:se.some(e=>e==null)?Infinity:se.reduce((t,e)=>t+e,0)/axes.length};};
  const rows=[];let judged=0,failed=0,under=0,skipped=0,rejected=0,shown=0;
  if(r.timesUs?.length!==e.samples||Math.abs(r.timesUs.at(-1)-e.durationUs)>1e-9*e.durationUs)return {error:`grid: ${r.timesUs?.length} times to ${r.timesUs?.at(-1)} µs, expected ${e.samples} to ${e.durationUs}`};
  for(const s of e.series){
    const name=(s.reported?'* ':'')+s.quantity+(s.axis?` ${s.axis}`:'')+(s.axes?` ${s.axes.join('+')}`:'')+(s.state?` ${s.state}`:'')+(s.label?` ${s.label}`:''),count=!s.reported;
    const outputs=s.indices?s.indices.map((i,j)=>({i,ref:s.values[j],u:s.u[j],judge:s.judged?s.judged[j]:true})):[{i:null,ref:s.value,u:s.u,judge:true}];
    const row={name,n:0,maxD:0,worst:-Infinity,worstAt:null,fail:0,exceed:0,under:0,skipped:0,z:[],neff:null,p:null,reject:false};
    for(const o of outputs){
      let x,se,reported=null;const i=o.i;
      switch(s.quantity){
        case 'population':x=r.populations[i][k(s.state)];se=Math.sqrt(x*(1-x)/held(i));reported=errors[i];break;
        case 'group':x=s.states.reduce((t,st)=>t+r.populations[i][k(st)],0);se=Math.sqrt(Math.max(0,x*(1-x))/held(i));reported=errors[i];break;
        case 'loss':x=loss[i];se=Math.sqrt(x*(1-x)/N);reported=errors[i];break;
        case 'T':{({x,se}=temperature(s,i));reported=Math.hypot(se,tLedger*x);
          if(s.regime==='classical'&&!m.classical[i])o.judge=false;break;}
        case 'Tincrease':{({x,se}=temperature(s,i));x-=temperature(s,0).x;break;}
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
      if(x===undefined)continue;
      if(!o.judge||!Number.isFinite(se)){row.skipped++;if(count)skipped++;continue;}
      const target=s.target.absolute??(s.target.relative*Math.abs(o.ref)),d=x-o.ref,sigma=Math.hypot(se,o.u);
      row.n++;if(count)judged++;else shown++;row.maxD=Math.max(row.maxD,Math.abs(d));
      const score=target>0?(Math.abs(d)-Z*sigma)/target:(Math.abs(d)-Z*sigma);
      if(score>row.worst){row.worst=score;row.worstAt={i,t:r.timesUs[i??0],x,ref:o.ref,se,u:o.u};}
      if(Math.abs(d)-Z*sigma>target){row.fail++;if(count)failed++;}
      if(reported!==null){row.z.push(sigma>0?d/sigma:0);if(Math.abs(d)>2*reported+2*o.u)row.exceed++;}
      if(target>0&&sigma>POWER*target){row.under++;if(count)under++;}
    }
    if(row.z.length){row.neff=effectiveCount(row.z);row.p=binomialTail(row.exceed*row.neff/row.z.length,row.neff,SHARE);row.reject=row.p<alpha;if(row.reject&&count)rejected++;}
    delete row.z;rows.push(row);
  }
  const judgedKind=!['characterization','regression-consistency'].includes(e.kind);
  const verdict=!judged?'NOT JUDGED':!judgedKind?'REPORT':failed||rejected?'FAIL':'PASS';
  return {rows,judged,failed,rejected,under,skipped,shown,verdict};
}

/** A relation between the steady states of several results (kind "relation": R4's scaling law, expect.mjs
 * r4linear): D = Σ c_k T̄_k, T̄_k the mean of member k's T on the relation's axis over its window, against `value`.
 * A window mean's SE is bounded by the mean of its outputs' (fully correlated in time); the members are separate runs,
 * so SE_D² = Σ c_k² SE_k². The scale is the T̄ of the member `scale`: the target and u are relative to it, and u adds to
 * the scenes' part what only the results tell -- the trap light's line shift over the cloud against the detuning,
 * κk_BT̄/ħ|δ|, and the polarization pattern's first harmonic over the coldest cloud, e^{−2k²σ²}, σ² = k_BT̄/mω².
 * The law is of steady states in the method's validated regime: a member whose window is not classical throughout, or
 * whose T̄ over the window's two halves differ by more than Z of their standard errors, leaves the relation NOT
 * JUDGED (said with the numbers). Then rule 1 as for an output: fail when |D − value| − Zσ > target; the coverage of
 * rule 2 (|D| against 2·(SE ⊕ Σ|c_k|·ledger_k·T̄_k) + 2u) is one output and only reported. */
export function relation(e,results){
  const mean=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
  const members=e.members.map((mb,k)=>{
    const r=results[k],m=r?.motion;if(!m)return {id:mb.id,error:'no result'};
    if(r.timesUs?.length!==mb.samples||Math.abs(r.timesUs.at(-1)-mb.durationUs)>1e-9*mb.durationUs)return {id:mb.id,error:`grid: ${r.timesUs?.length} times to ${r.timesUs?.at(-1)} µs, expected ${mb.samples} to ${mb.durationUs}`};
    const [lo,hi]=mb.window,idx=Array.from({length:hi-lo+1},(_,j)=>lo+j),mid=idx.length>>1;
    const part=ix=>{const se=ix.map(i=>m.temperatureErrorUK[e.axis][i]);return {x:mean(ix.map(i=>m.temperatureUK[e.axis][i])),se:se.some(s=>s==null)?Infinity:mean(se)};};
    const whole=part(idx),early=part(idx.slice(0,mid)),late=part(idx.slice(mid));
    return {id:mb.id,...whole,early,late,from:r.timesUs[lo],start:m.temperatureUK[e.axis][0],classical:idx.every(i=>m.classical[i]),steady:Math.abs(late.x-early.x)<=Z*Math.hypot(early.se,late.se),
      ledger:r.diagnostics?.approximations?.totals?.temperature?.error??0,trajectories:m.trajectories,step:m.timeStepUs};
  });
  if(members.some(m=>m.error))return {members,verdict:'NO RESULT'};
  const c=e.coefficients,D=members.reduce((t,m,k)=>t+c[k]*m.x,0),se=Math.hypot(...members.map((m,k)=>c[k]*m.se)),scale=members[e.scale].x,i=e.info,kT=KB/HBAR*1e-12;
  const shift=i.kappa*kT*scale/Math.abs(i.deltaRadPerUs),pattern=Math.max(...members.map(m=>Math.exp(-2*i.k*i.k*i.hbarOverMass*kT*m.x/(i.trapZ*i.trapZ))));
  const u=scale*(e.u.relative+shift+pattern),sigma=Math.hypot(se,u),target=e.target.relative*scale,d=D-e.value;
  const reported=Math.hypot(se,members.reduce((t,m,k)=>t+Math.abs(c[k])*m.ledger*m.x,0));
  const reasons=[...members.filter(m=>!m.classical).map(m=>`${m.id}: below 4ħω in its window (the semiclassical limit)`),...members.filter(m=>!m.steady).map(m=>`${m.id}: not steady (T̄ ${m.early.x.toFixed(2)} ± ${m.early.se.toFixed(2)} then ${m.late.x.toFixed(2)} ± ${m.late.se.toFixed(2)} µK over the window's halves)`),
    ...members.filter(m=>!Number.isFinite(m.se)).map(m=>`${m.id}: no error estimate`)];
  return {members,D,se,u,terms:{scenes:e.u.relative,shift,pattern},sigma,target,score:(Math.abs(d)-Z*sigma)/target,covered:Math.abs(d)<=2*reported+2*u,under:sigma>POWER*target,reasons,
    verdict:reasons.length?'NOT JUDGED':Math.abs(d)-Z*sigma>target?'FAIL':'PASS'};
}

function printRelation(e,c){
  console.log(`${e.id} [${e.reference}, ${e.kind}] ${e.title}`);
  for(const m of c.members)console.log('  '+m.id.padEnd(12)+(m.error??`T̄_${e.axis} ${m.x.toFixed(3)} ± ${m.se.toFixed(3)} µK from ${m.from} µs (halves ${m.early.x.toFixed(2)}, ${m.late.x.toFixed(2)}; T(0) ${m.start.toFixed(2)}) | ${m.classical?'classical':'BELOW 4ħω'}, ${m.steady?'steady':'NOT STEADY'} | ${m.trajectories} trajectories, step ${m.step} µs, ledger ${(100*m.ledger).toFixed(1)} %`));
  if(c.verdict==='NO RESULT'){console.log('  NO RESULT\n');return;}
  const U=e.info?.modulationUK,first=c.members[0],last=c.members.at(-1),slope=U?(last.x-first.x)/(U.at(-1)-U[0]):null;
  console.log(`  ${c.verdict}: D = ${e.coefficients.map((q,k)=>`${q>0?'+':'−'}${Math.abs(q)}·T̄(${e.members[k].id})`).join(' ')} = ${c.D.toFixed(3)} µK against ${e.value}; SE ${c.se.toFixed(3)}, u ${c.u.toFixed(3)} (relative to T̄ ${c.members[e.scale].x.toFixed(2)}: scenes ${c.terms.scenes.toFixed(4)}, line shift ${c.terms.shift.toFixed(4)}, pattern ${c.terms.pattern.toExponential(1)}), target ${c.target.toFixed(3)}`);
  console.log(`  (|D| − 2σ)/target = ${c.score.toFixed(2)}${c.under?'; underpowered (σ > 0.3 target)':''}; reported errors ${c.covered?'cover':'DO NOT cover'} D`+(slope===null?'':`; line through the outer two: T̄ = ${slope.toFixed(4)}·U + ${(first.x-slope*U[0]).toFixed(2)} µK (U the light potential's modulation)`));
  for(const r of c.reasons)console.log(`  not judged: ${r}`);
  console.log('');
}

function print(e,c,run){
  const head=`${e.id} [${e.reference}, ${e.kind}] ${e.title}`;
  if(c.error){console.log(`${head}\n  ${c.error}\n`);return;}
  console.log(`${head}\n  ${c.verdict}: ${c.judged} outputs judged; rule 1 (beyond target): ${c.failed}; rule 2 (coverage rejected): ${c.rejected} series; ${c.under} underpowered; ${c.skipped} not judged${c.shown?`; ${c.shown} reported (*)`:''}`+
    (run?` | run ${run.wallS.toFixed(1)} s${run.aborted?' ABORTED':''}, first checkpoint ${run.firstCheckpointS?.toFixed(1)??'-'} s, ${run.final?.result?.motion?.trajectories??'-'} trajectories`:''));
  console.log('  '+['series'.padEnd(30),'n'.padStart(4),'max|d|'.padStart(10),'(|d|−2σ)/tgt'.padStart(14),'fail'.padStart(5),'out'.padStart(5),'n_eff'.padStart(7),'p'.padStart(8),'weak'.padStart(5),'  worst output (t: solver vs ref ± se, u)'].join(''));
  for(const r of c.rows){const w=r.worstAt;
    console.log('  '+[r.name.slice(0,30).padEnd(30),String(r.n).padStart(4),r.maxD.toPrecision(3).padStart(10),(Number.isFinite(r.worst)?r.worst.toFixed(2):'-').padStart(14),String(r.fail).padStart(5),String(r.exceed).padStart(5),
      (r.neff===null?'-':r.neff.toFixed(1)).padStart(7),(r.p===null?'-':r.p.toPrecision(2)+(r.reject?'!':'')).padStart(8),String(r.under).padStart(5),
      w?`  t=${w.t.toPrecision(4)}: ${w.x.toPrecision(5)} vs ${w.ref.toPrecision(5)} ± ${w.se.toPrecision(2)}, ${w.u.toPrecision(2)}`:''].join(''));}
  console.log('');
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const args=process.argv.slice(2),one=args.find(a=>a.startsWith('--result='))?.slice(9),many=args.find(a=>a.startsWith('--results='))?.slice(10).split(',');
  const listed=f=>fs.existsSync(f)?fs.readdirSync(f).filter(x=>x.endsWith('.json')).map(x=>x.slice(0,-5)):[];
  let ids=args.filter(a=>!a.startsWith('--'));
  if(args.includes('--r'))ids.push(...listed(path.join(here,'expect')).filter(id=>/^R[1-9]/.test(id)));
  if(args.includes('--r0'))ids.push(...listed(path.join(here,'expect')).filter(id=>id.startsWith('R0_')));
  if(args.includes('--regress'))ids.push(...listed(path.join(here,'expect','regress')));
  const summary=[];let bad=0;
  for(const id of ids){
    const ef=expectationFile(id);if(!ef){console.log(id,'no expectation');continue;}
    const e=read(ef);
    if(e.kind==='relation'){
      const files=many??e.members.map(m=>path.join(here,'out','results',`${m.id}.json`)),c=relation(e,e.members.map((_,k)=>fs.existsSync(files[k]??'')?resultOf(read(files[k])):null));
      printRelation(e,c);summary.push([e.id,c.verdict]);if(c.verdict==='FAIL')bad++;continue;
    }
    const rf=one??path.join(here,'out','results',`${e.id}.json`);
    if(!fs.existsSync(rf)){summary.push([e.id,'NO RESULT']);continue;}
    const raw=read(rf),r=resultOf(raw);if(!r){summary.push([e.id,raw.final?.error?`ERROR ${raw.final.error}`:'NO RESULT']);continue;}
    const c=check(e,r,read(path.join(here,e.request)));print(e,c,raw.final?raw:null);summary.push([e.id,c.error?'GRID MISMATCH':c.verdict]);if(c.verdict==='FAIL'||c.error)bad++;
  }
  if(summary.length>1){console.log('summary');for(const [id,v] of summary)console.log(`  ${id.padEnd(26)} ${v}`);}
  process.exitCode=bad?1:0;
}
