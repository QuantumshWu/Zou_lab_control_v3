// node selftest.mjs -- the constructed checks of check.mjs's two tests of a series against its claimed errors
// (`consistency`: the count beyond the bound, the mean deviation) and of a scene's family of them (rule 3). Synthetic
// series only; a fixed seed. Each line: the case, what the rule must do, what it does. Exit code 1 where a case that
// must be rejected is not, or honest series or scenes are rejected more often than the rule states.
import {check,consistency,quantile,ALPHA,BEYOND} from './check.mjs';

const Z=2;
let seed=0x1234ABCD;const next=()=>{seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
const normal=()=>Math.sqrt(-2*Math.log(1-next()))*Math.cos(2*Math.PI*next());
/** The share of `trials` series from `make` (a list of deviations in units of the reported bound E at the confidence
 * z, all of it statistics: an honest output is N(0, (E/z)²)) that the tests reject: [either, the count's, the mean's]. */
function rate(make,trials){
  let either=0,count=0,common=0;
  for(let t=0;t<trials;t++){const c=consistency(make().map(d=>({d,u:0,fixed:0,random:1})));either+=c.reject;count+=c.count;common+=c.common;}
  return [either,count,common].map(x=>x/trials);
}
const iid=n=>()=>Array.from({length:n},()=>normal()/Z);
/** AR(1) of unit variance and correlation ρ between neighbours. */
const ar=(n,rho)=>()=>{let x=normal();return Array.from({length:n},()=>{const y=x;x=rho*x+Math.sqrt(1-rho*rho)*normal();return y/Z;});};
/** One sinusoid of a random phase and a Rayleigh amplitude (each output N(0, 1) over the ensemble: a sample's breathing
 * at twice its trap frequency, not decaying), `turn` radians an output, plus white noise of the share `white` of the variance. */
const breathing=(n,turn,white)=>()=>{const a=normal(),b=normal();return Array.from({length:n},(_,i)=>(Math.sqrt(1-white)*(a*Math.cos(turn*i)+b*Math.sin(turn*i))+Math.sqrt(white)*normal())/Z);};
const percent=x=>(100*x).toFixed(2)+' %';

let bad=0;
const must=(name,ok,text)=>{console.log(`${ok?'ok  ':'BAD '} ${name}: ${text}`);if(!ok)bad++;};
// 1. A bias common to every output, twice the reported error -- alone, with honest noise on it, and where the reported
//    error is all systematic: rejected, whatever the correlation.
for(const [name,make,claim] of [['a common bias of twice the reported error, nothing else, 100 outputs',()=>Array(100).fill(2),{fixed:0,random:1}],
  ['the same on one output',()=>[2],{fixed:0,random:1}],['the same with honest independent noise on it',()=>iid(100)().map(d=>d+2),{fixed:0,random:1}],
  ['the same with fully correlated honest noise on it',()=>{const e=normal()/Z;return Array(100).fill(2+e);},{fixed:0,random:1}],
  ['the same where the reported error is the model\'s (no statistics)',()=>Array(100).fill(2),{fixed:1,random:0}]]){
  let rejected=0;const trials=2000;for(let t=0;t<trials;t++)rejected+=consistency(make().map(d=>({d,u:0,...claim}))).reject;
  // Fully correlated noise of standard deviation E/z on a bias 2E: |d̄| ≤ E only where the noise is below −z: Φ(−z).
  must(name,rejected/trials>=1-0.03,`rejected ${percent(rejected/trials)} (must: all; with fully correlated noise all but Φ(−2) = 2.28 %)`);
}
// 2. A bias common to every output within the reported error: not rejected (the claim allows it).
must('a common bias of 0.9 of the reported error, 100 outputs',!consistency(Array(100).fill({d:0.9,u:0,fixed:0,random:1})).reject,'not rejected');
// 3. Honest independent residuals: the count test at no more than Φ(−z) = 2.28 %, the mean test not at all.
for(const n of [10,49,101,201,401]){const [either,count,common]=rate(iid(n),20000);
  must(`honest independent residuals, ${n} outputs`,either<=0.0228+3*Math.sqrt(0.0228*0.9772/20000),`rejected ${percent(either)} (count ${percent(count)}, mean ${percent(common)}; stated: ≤ 2.28 %)`);}
// 4. Honest correlated residuals: a drifting series (AR(1)) and a breathing one are not rejected more often.
for(const [name,make] of [['honest AR(1) residuals, ρ = 0.9, 201 outputs',ar(201,0.9)],['honest AR(1) residuals, ρ = 0.98, 201 outputs',ar(201,0.98)],
  ['an honest breathing sample (a sinusoid, 1.9 rad an output), 201 outputs',breathing(201,1.9,0)],['the same with half its variance white',breathing(201,1.9,0.5)]]){
  const [either,count,common]=rate(make,20000);
  must(name,either<=0.0228+0.0455+3*Math.sqrt(0.07*0.93/20000),`rejected ${percent(either)} (count ${percent(count)}, mean ${percent(common)}; stated: count ≤ 2.28 %, mean ≤ 2Φ(−2) = 4.55 % for one common draw)`);}
// 5. An understated error, independent residuals of 1.5 times the reported size: rejected by the count (the power).
{const [either]=rate(()=>iid(201)().map(d=>1.5*d),5000);must('independent residuals of 1.5 times the reported error, 201 outputs',either>0.9,`rejected ${percent(either)} (power; must: most)`);}
// 6. A scene of m honest series, each one draw for its run (rule 3; the class R1_CROSSED at 717cb386 fell in: every
//    axis's T of a conservative trap), its standard error a jackknife's over g groups: the series' deviation is the mean
//    of g independent group values less their expectation, common to its n outputs, with a white remainder of a tenth
//    of its standard error; its claim the result's own interval at z, t_{g−1}·SE (estimate.rs `Spread::interval`).
//    The scene fails -- some series rejected, each at the family's tails -- no more often than 2Φ(−z) + Φ(−z), the two
//    tests' own levels, where the group values are normal (Student's t then the jackknife-t's exact law); the rule
//    before (z·SE, each series at its own level) shown beside: 1 − (1 − P(|t_{g−1}| > z))^m.
//    Skewed group values -- each the mean of 8 exponential energies, an atom's energy along one axis of a thermal trap
//    (measured in R1_CROSSED: per atom var/mean² 0.93–1.00, skewness 1.8–2.2) -- make the jackknife-t's low tail heavier
//    than Student's (Edgeworth: by γ(2q² + 1)φ(q)/(6√g) at q, γ the group mean's skewness), the more so at the family's
//    far tails: the result's symmetric interval of a right-skewed T, a property of the solver's statistics the rule
//    reads as stated. R1's own family -- four such series of T among fifteen (the trap frequencies' and populations'
//    claims hold u, and their deviations none of T's skew) -- must keep the stated rate; a scene of fifteen skewed T
//    series on eleven groups is shown (note), not held to it.
const scene=(m,g,n,group,random=m,shift=0)=>{
  const tq=quantile(BEYOND,g-1);let now=0,before=0;
  return trials=>{for(let t=0;t<trials;t++){let fails=false,failed=false;
    for(let s=0;s<m;s++){const v=Array.from({length:g},s<random?group:normal),mean=v.reduce((a,b)=>a+b,0)/g,se=Math.sqrt(v.reduce((a,b)=>a+(b-mean)**2,0)/(g-1)/g);
      const e=mean+(s===0?shift:0),d=Array.from({length:n},()=>e+0.1*se*normal());
      fails||=consistency(d.map(x=>({d:x,u:0,fixed:0,random:tq*se,freedom:g-1})),m).reject;
      failed||=consistency(d.map(x=>({d:x,u:0,fixed:0,random:Z*se}))).reject;}
    now+=fails;before+=failed;}
    return [now/trials,before/trials];};
};
const exponential=()=>-Math.log(1-next()),skewed=()=>Array.from({length:8},exponential).reduce((a,b)=>a+b,0)/8-1,stated=BEYOND+ALPHA;
for(const [m,g,group,random,label] of [[15,11,normal,15,'R1 at the pilot: 15 series on 11 groups, normal group values'],[4,24,normal,4,'4 series on 24 groups, normal group values'],
  [1,3,normal,1,'one series on 3 groups, normal group values'],[15,11,skewed,4,'R1\'s family on 11 groups: 4 skewed series of T among 15'],[15,24,skewed,4,'the same on 24 groups']]){
  const trials=4000,[now,before]=scene(m,g,49,group,random)(trials);
  must(`an honest scene, ${label}, 49 outputs`,now<=stated+3*Math.sqrt(stated*(1-stated)/trials),`fails ${percent(now)} (stated: ≤ 2Φ(−2) + Φ(−2) = ${percent(stated)}; the rule before: ${percent(before)})`);}
{const [now,before]=scene(15,11,49,skewed)(4000);console.log(`note an honest scene of 15 skewed series of T on 11 groups, 49 outputs: fails ${percent(now)} (the solver's symmetric interval of a skewed T; the rule before: ${percent(before)})`);}
// 7. Its power: one series of the scene off by twice the family's bound for its expected standard error, the others
//    honest: rejected, by the mean test of that series.
{const m=15,g=11,shift=2*quantile(1-(1-BEYOND)**(1/m),g-1)/Math.sqrt(g),[now]=scene(m,g,49,normal,m,shift)(2000);
  must('one series of fifteen off by twice the family\'s bound, 11 groups',now>0.9,`rejected ${percent(now)} (power; must: most)`);}
// 8. A result's claim read through `check` (rule 0, one axis's T, a family of one): R1_CROSSED's T_y at 717cb386 -- one
//    draw 2.12 standard errors low for the whole run, 88 trajectories in groups of 8 -- lies inside the result's own
//    interval t₁₀ = 2.28 and passes; the same result stating no grouping is read at z and rejected.
{const n=21,times=Array.from({length:n},(_,i)=>50*i),se=2.27,dev=-2.12*se;
  const e={samples:n,durationUs:1000,kind:'pass-fail',series:[{quantity:'T',axis:'y',target:null,indices:times.map((_,i)=>i).slice(1),values:Array(n-1).fill(30),u:Array(n-1).fill(0)}]};
  const result=method=>({timesUs:times,states:[],populations:[],populationAccuracy:{sampleErrors:[]},diagnostics:{method,approximations:{totals:{temperature:{error:0}}}},
    motion:{trajectories:88,lossProbability:Array(n).fill(0),temperatureUK:{y:times.map((_,i)=>i?30+dev+0.1*se*(-1)**i:30)},temperatureErrorUK:{y:times.map((_,i)=>i?se:0)}}});
  const grouped=check(e,result('88 条轨迹（每 8 条一组，组内按拉丁超立方抽取初始原子）'),{}).verdict,plain=check(e,result('88 条轨迹'),{}).verdict;
  must('a one-draw T 2.12 standard errors off on eleven groups, through check',grouped==='PASS'&&plain==='FAIL',`${grouped} at the result's t₁₀ (must: PASS); ${plain} with no grouping stated, at z (must: FAIL)`);}
process.exitCode=bad?1:0;
