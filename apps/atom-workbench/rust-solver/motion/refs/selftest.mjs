// node selftest.mjs -- the constructed checks of check.mjs's two tests of a series against its claimed errors
// (`consistency`: the count beyond the bound, the mean deviation). Synthetic series only; a fixed seed. Each line: the
// case, what the rule must do, what it does. Exit code 1 where a case that must be rejected is not, or an honest series
// is rejected more often than the rule states.
import {consistency} from './check.mjs';

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
process.exitCode=bad?1:0;
