// node check.mjs <id>... | --r | --r0 | --regress [--result=<file>] [--results=<file>,<file>,...] -- applies the
// references' acceptance rules (contract §6, SPEC §6) to the solver's results (out/results/<id>.json from run.mjs, or
// any file holding the contract's result: the bare result, a native line {result}, or run.mjs's {final:{result}}) and
// prints a compact table. A relation (kind "relation": one statement about several scenes' results, `relation` below)
// reads its members' results (out/results/<member>.json, or --results in the members' order).
//
// Per output (a series' value at an output index): d = solver − reference, u the reference's standard uncertainty, SE
// the solver's statistical standard error, and the band at the confidence z the two intervals combined,
// √((q·SE)² + (zu)²), q the quantile of SE's own law at the two-sided tail 2Φ(−z) -- z itself for a standard error of
// known variance, Student's t at the jackknife's degrees of freedom for T's (`quantile`):
//   T            one axis: temperatureErrorUK; `total`: temperatureTotalUK with temperatureTotalErrorUK, the result's
//                own total (T_x + T_y + T_z)/3 and its jackknife; `axes`: their mean, its SE bounded by the mean of
//                theirs (fully correlated); null: no estimate, the output is not judged. Each is the delete-one-group
//                jackknife over the groups that hold an atom (estimate.rs, # Errors), whose interval at the confidence
//                z the result states as Student's t at one degree of freedom fewer than those groups times SE (Tukey's
//                jackknife t; estimate.rs `Spread::interval`; the app's text: "区间按组数取 Student t，组少时比两倍
//                标准误差宽"): q = t_{g−1}, g the groups -- the result's trajectories come in groups of the size its
//                method states ("每 G 条一组"), an output's held atoms lie in at most min(N/G, N_held) of them, and that
//                most is read, so that the band is never wider than the result's own interval. z·SE in its place
//                understated the claim by t_{g−1}/z (14 % at the pilot's eleven groups, 2.2 times at three) and
//                rejected what the result had not claimed (R1_CROSSED at 717cb386: T_y 2.12 SE off on eleven groups,
//                inside the result's t₁₀ = 2.28, beyond z = 2). A result that states no grouping is read at z;
//   loss         √(L(1 − L)/N), N = motion.trajectories (a Bernoulli mean);
//   population   √(p(1 − p)/N_held), N_held = N(1 − L): the variance of a [0, 1]-valued mean is at most p(1 − p), so
//   and group    this bounds SE from above -- it can only make the first rule more lenient, by at most 1/√N_held;
//   meanVelocity √(k_B T_a/m)/√N_held; meanPosition(Average) the distribution's own spread over √N_held;
//   trapFrequency 0 (deterministic).
// Every constant below is z = 2, the confidence of the solver's own statistics (SPEC §0), or follows from it: the
// share of outputs a Gaussian error leaves beyond zσ, 2Φ(−z) = 4.55 %; the level a test rejects at, Φ(−z) = 2.28 %;
// either taken over a scene's family of tests by Šidák (rule 3).
//
// A series with a user's target (SPEC §0: populations, groups and loss 0.05 absolute; T 10 % of the reference -- on the
// total temperature, solve.rs "# T's target is on the total temperature") is judged by two rules.
// Rule 1 (accuracy): an output fails where |d| − band > target, the target unrelaxed. An output is resolved where
//   band ≤ target: an error as large as the target is then told from none at the confidence z. Where band > target and
//   the output does not fail, target + band is more noise than target: the output is not judged (column `unres`).
// Rule 2 (coverage, SPEC §6: the reported error covers the deviation). The result's error E of an output is its bound
//   at the confidence z (ledger.rs: E² = A² + b² + S², A and b the model's and the step's parts, S the statistics'
//   interval at z): populations, groups and loss: the result's sampleErrors (the total; of it S is taken as z·SE of
//   the Bernoulli bound above, no more than the total, the rest systematic); T: S = t_{g−1}·SE (above), A ⊕ b = the
//   ledger's totals.temperature.error × T. With the reference's own u the claim is |d| ≤ B, B² = E² + (zu)². Two tests:
//   (a) the count K of outputs with |d| > B against effectively independent outputs. The null: the standardized
//       residuals r_i = z d_i/B_i (in standard deviations of the claim: beyond is |r_i| > z), in time order, are a
//       stationary Gaussian series of mean zero, a variance of no more than one, and an autocorrelation ρ_k that the
//       claim leaves open -- a result states the size of its errors, not what the outputs of one run share. Then
//       E K ≤ nq, q = 2Φ(−z), and the indicators 1{|r_i| > z} of two outputs k apart have the covariance
//       q(1 − q)κ(ρ_k), κ(ρ) = (1/π)∫_0^{asin|ρ|} [e^{−z²/(1 + sin θ)} − e^{−z²/(1 − sin θ)}] dθ / q(1 − q) (Plackett:
//       dP(X > h, Y > k)/dρ is the bivariate density at (h, k); the four corners of the two-sided event, t = sin θ):
//       even in ρ, 0 at ρ = 0, 1 at |ρ| = 1, 0.54ρ² for small ρ and no more than ρ² (Mehler: a two-sided event holds
//       only the even Hermite orders) -- a residual that oscillates counts as one that drifts (the kinetic energy of
//       N atoms in a harmonic trap carries a part at twice the trap frequency that does not decay: the sample's own
//       breathing, of the size of its standard error). So
//       Var K = q(1 − q)[n + 2Σ_k (n − k)κ(ρ_k)]: that of n_eff = n/(1 + 2Σ_k (1 − k/n)κ(ρ_k)) independent outputs,
//       each counted n/n_eff times. The count scaled by n_eff/n; a one-sided binomial test (Bin(n_eff, q), the
//       regularized incomplete beta for non-integer counts) rejects where the tail probability is below Φ(−z). The
//       binomial has the null's mean and variance, and is the null's own law at both ends: independent outputs
//       (n_eff = n), and outputs that are one draw (n_eff = 1: K is n with the probability q and 0 otherwise, and the
//       tail I_q(K/n, 2 − K/n) is never below q: one event of 4.55 % is no evidence, however many outputs it shows
//       on). Between the ends the shape is the binomial's, not derived: measured (the end of (b)).
//       ρ_k is the sample's about ZERO, the null's mean, each lag from its own n − k products:
//       ρ̂_k = [Σ_i r_i r_{i+k}/(n − k)]/[Σ_i r_i²/n], its square no more than 1; of each ρ̂_k² only what exceeds
//       z²(1 − ρ̂_k²)²/(n − k) enters -- a correlation from n − k pairs has the standard deviation (1 − ρ²)/√(n − k)
//       (1/√(n − k) for independent outputs, none for outputs that are one draw), and what lies within z of it is
//       not told from none. About zero, not about the run's mean: an offset the whole run shares is, under the null,
//       noise whose correlation time is the run or longer (ρ_k → 1: a held trap conserves each atom's energy, and a
//       sample's energy is one draw for the run); taken about the run's mean it is removed, the small remainder
//       decorrelates fast, and the exceedances that one offset causes are counted as so many independent events
//       (the rule before: over 64 seeds of R10_HELD_ANTINODE, whose reported errors are the seed-to-seed scatter
//       within 0.96 ± 0.07, the count rejected 10 of the 192 judged T series and the mean deviation 6, with tail
//       probabilities to 1e-37 for one offset of 1.2 bounds; this rule: 0 and the same 6). The ratio to the series'
//       own mean square, not to the null's variance: the estimate is of the structure, whatever the size.
//       What (a) sees: more exceedances than the residual's own structure leaves room for -- errors understated on
//       outputs that are many draws (noise that decorrelates within the run, a breathing the errors do not state,
//       populations at separated times). What it cannot see: a residual that is one draw or a few (n_eff → 1) -- any
//       number of its outputs beyond the bound is one event, and whether that draw is too large is (b)'s question,
//       for the part common to the run: of a series that is one draw a single run tells an error understated f times
//       only with the probability 2Φ(−z/f). A slow residual that changes sign within the run is a few draws to (a)
//       and averages out of (b): it is seen where it is large enough to be more than those few draws allow, or by
//       rule 1;
//   (b) the mean deviation d̄ = Σd_i/n against the bound the claims leave it: M² = (mean A⊕b)² + (z·mean u)² +
//       (mean S)²; rejected where |d̄| > M. Derivation: d_i = a_i + v_i + e_i, three independent sources: the solver's
//       systematic error (claimed |a_i| ≤ A_i ⊕ b_i; common to the outputs at worst, so |ā| ≤ mean A⊕b), the
//       reference's (standard deviation u_i; common at worst: that of v̄ ≤ mean u) and the statistics' (standard
//       deviation S_i/z; the outputs of one run are read from the same trajectories, and what their errors share the
//       result does not say -- at worst all of it: in a conservative trap the sample's energy is one draw for the
//       whole run, R1 --, so that of ē ≤ mean S/z), combined in quadrature at the confidence z as the ledger combines
//       its own parts. Nothing is granted for averaging over the outputs (with a 1/√n_eff of the residuals about
//       their mean, R1's total T, 0.7 standard errors off for the whole run, was rejected: measured, not kept). An
//       honest error whose outputs are one draw is beyond M with the probability 2Φ(−z) = 4.55 % a series (less
//       where the ledger or u is part of M): that, and no more, is what a run of honest errors fails by. A series whose every output is beyond its bound on one side
//       has |d̄| > mean B ≥ M (the norm of a mean is no more than the mean of the norms): rejected whatever its
//       correlation -- test (a) alone cannot: its n_eff is 1 for a constant residual, where a count of one in one is
//       no evidence. On independent residuals of the reported size (a) is the binomial test of n outputs and (b)
//       does not reject (the mean of n ≥ 2 such outputs has the standard deviation S/z√n: it reaches M = S with the
//       probability 2Φ(−z√n) ≤ 0.5 %): measured by `node selftest.mjs`. So the two divide a series between them:
//       (b) answers for the part the whole run shares, at 2Φ(−z) a series for honest errors; (a) for what varies
//       within the run: honest errors -- independent, drifting, breathing (selftest.mjs: 0.02 to 1.4 % of series),
//       or one draw with a remainder -- it rejects less often than Φ(−z), and a series that is one draw with a
//       remainder next to never where (b) does not (constructed as selftest's, 49 and 201 outputs, a white or AR(1)
//       remainder of 0.02 to 0.8 of the variance: (a) alone 0.13 % at most, where the rule before had 2 to 15 %).
// A series without a user's target (`target: null`: trap frequencies, a mean position or velocity, one axis's T --
//   each axis's T is shown with its statistics and held to nothing by the solver) is judged against the reference's own
//   uncertainty and the result's statistics alone, by the same two tests with B = the band (no systematic part of the
//   solver's: A ⊕ b = 0, S = q·SE): rule 0 (consistency). A single deterministic output (a trap frequency): |d| ≤ zu.
// Rule 3 (the family): a scene fails where any of its series' tests rejects, so an honest result fails by the sum of
//   its tests' chances. Each test of rules 0 and 2 is at its own level, and (b) rejects a series that is one draw for
//   the whole run (a conservative trap's sample energy, R1: every axis's T and the total T) with the probability
//   2Φ(−z) whatever the trajectories -- more of them shrink the bound and the draw alike --: a scene of m such series
//   failed an honest result 1 − (1 − 2Φ(−z))^m of the time, 50 to 66 % over the landing gate's fifteen one-axis T
//   series of R1 at eleven to twenty-four groups (R1_CROSSED: v23 passed and 717cb386 failed on the same 88 pilot
//   atoms, T_y 25.23 ± 2.27 µK in both; v23 had gone on to 192). The family's tests are taken at Šidák's tails,
//   1 − (1 − p)^{1/m} for each test's own p (the count's Φ(−z), the mean's 2Φ(−z)), which keeps the chance that some
//   test of the scene rejects an honest result at p -- exactly for independent series, at most p for the correlated
//   ones of one run (its outputs and axes share their trajectories: Šidák's inequality for the centred multivariate
//   normal) --, the rule the solver takes for a decision over many reads (estimate.rs, # Many looks). m counts the
//   series judged whose claims hold a statistical part (S > 0 or u > 0 at some output): a claim of a systematic part
//   alone is not met by chance, and its rejection is evidence by itself. At the mean's tail the bound M takes each
//   part at its own quantile there -- S by its law (Student's t for T, its interval at z rescaled), u by the
//   normal's --, A ⊕ b as it is (a bound, not a spread); the count's bound B and its null rate 2Φ(−z) stay the
//   claim's at z, its level is Šidák's. What it costs: a series is told at the family's confidence -- a one-draw
//   series among R1's fifteen at 2.96 standard errors of known variance, t₁₀ = 3.85 on eleven groups, instead of 2 --,
//   the least a single run can tell without failing honest results by more than 2Φ(−z) a scene.
// T series of regime "classical" are compared only where the result's `classical` is true; outputs an expectation
// marks unjudged are reported, not judged. Kinds "characterization" and "regression-consistency" are reported, never
// failed. A series marked `reported` (another engine's old numeric result, an axis whose share a symmetry leaves open)
// is compared and printed like any other, marked *, and never judged: its outputs enter no total and fail nothing.
// Kind "tool" (R4D_FORCE, R4D_TW2: the engine stepped through the example binary beside an exact tool, r4d.mjs) is run
// and judged by its own command; here its last verdict is listed.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const KB=1.380649e-23,HBAR=6.62607015e-34/(2*Math.PI);
/** z, the confidence of the solver's own statistics (SPEC §0). */
export const Z=2;
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
/** Φ(−z), the level of a one-sided test at the confidence z; 2Φ(−z), the share of outputs a Gaussian error leaves
 * beyond zσ. */
export const ALPHA=upperTail(Z),BEYOND=2*ALPHA;
/** The two-sided quantile at the tail p of Student's t at `freedom` degrees of freedom, the normal's at Infinity: the q
 * with P(|t| > q) = p -- I_{ν/(ν + q²)}(ν/2, ½) for Student's (Abramowitz–Stegun 26.7.1 with 26.5.27), 2Φ(−q) for the
 * normal's -- by bisection on the tail to the last bit, kept per (p, freedom). A standard error's interval at the tail
 * p (the header): z at 2Φ(−z) for the normal's, t_{g−1} for a jackknife over g groups (estimate.rs `quantile`, the same
 * definition). */
const kept=new Map();
export function quantile(p,freedom=Infinity){
  const key=`${p} ${freedom}`;if(kept.has(key))return kept.get(key);
  const tail=q=>Number.isFinite(freedom)?betai(freedom/2,0.5,freedom/(freedom+q*q)):2*upperTail(q);
  let lo=0,hi=1;while(tail(hi)>p){lo=hi;hi*=2;}
  for(;;){const m=0.5*(lo+hi);if(!(m>lo&&m<hi))break;if(tail(m)>p)lo=m;else hi=m;}
  kept.set(key,hi);return hi;
}
/** Šidák's tail of one of `looks` tests that keeps the family's chance of a rejection at p (rule 3). */
export const sidak=(p,looks)=>-Math.expm1(Math.log1p(-p)/looks);
/** κ(ρ), the correlation of the indicators 1{|X| > z}, 1{|Y| > z} of two standard normal variables of correlation
 * ρ in [0, 1] (the header's (a)): (1/π)∫_0^{asin ρ} [e^{−z²/(1 + sin θ)} − e^{−z²/(1 − sin θ)}] dθ over q(1 − q), by
 * Simpson's rule on 32 panels (the integrand is smooth to the end: within 1e-7 of the integral for every ρ). */
export function exceedanceCorrelation(rho){
  const panels=32,h=Math.asin(rho)/panels,f=t=>{const s=Math.sin(t);return Math.exp(-Z*Z/(1+s))-Math.exp(-Z*Z/(1-s));};
  let sum=0;for(let i=0;i<=panels;i++)sum+=(i===0||i===panels?1:i%2?4:2)*f(i*h);
  return sum*h/3/Math.PI/(BEYOND*(1-BEYOND));
}
/** The effective number of independent outputs of a residual series for a count of its two-sided exceedances (the
 * header's (a)): n/(1 + 2Σ_k (1 − k/n)κ(ρ_k)), ρ_k² = max(0, ρ̂_k² − z²(1 − ρ̂_k²)²/(n − k)), ρ̂_k the autocorrelation
 * about zero from the n − k products of the lag, ρ̂_k² ≤ 1; n for a series of no deviation at all (nothing to count). */
export function effectiveCount(r){
  const n=r.length,c0=r.reduce((a,b)=>a+b*b,0)/n;if(!(c0>0))return n;
  let s=0;for(let k=1;k<n;k++){const m=n-k;let c=0;for(let i=0;i<m;i++)c+=r[i]*r[i+k];const e2=Math.min(1,(c/m/c0)**2),rho2=e2-(Z*(1-e2))**2/m;if(rho2>0)s+=m/n*exceedanceCorrelation(Math.sqrt(rho2));}
  return n/(1+2*s);
}
/** The two tests of a series against the errors claimed for it (the header's rule 2; rule 0 with no systematic part of
 * the solver's), as one of a family of `looks` series (rule 3; one by default): per output the deviation d, the
 * systematic part of its bound at the confidence z (`fixed`: A ⊕ b), the statistics' interval at z (`random`: S) and
 * the degrees of freedom of its law (`freedom`, the normal's where absent), and the reference's standard uncertainty u.
 * Returns the count of outputs beyond their bound, the count test's n_eff and tail probability p, the mean deviation
 * and its bound M at the family's tail, and whether either rejects. */
export function consistency(outputs,looks=1){
  const n=outputs.length,B=outputs.map(o=>Math.hypot(o.fixed,o.random,Z*o.u)),r=outputs.map((o,i)=>B[i]>0?Z*o.d/B[i]:o.d?Math.sign(o.d)*Infinity:0);
  const beyond=outputs.filter((o,i)=>Math.abs(o.d)>B[i]).length,mean=f=>outputs.reduce((t,o)=>t+f(o),0)/n;
  const neff=r.every(Number.isFinite)?effectiveCount(r):1,p=binomialTail(neff*(beyond/n),neff,BEYOND);
  // Each part of M at the family's tail: S by its own law (its interval at z rescaled), u by the normal's, A ⊕ b as it is.
  const tail=sidak(BEYOND,looks),at=o=>o.random&&o.random*quantile(tail,o.freedom)/quantile(BEYOND,o.freedom);
  const dbar=mean(o=>o.d),M=Math.hypot(mean(o=>o.fixed),quantile(tail)*mean(o=>o.u),mean(at)),level=sidak(ALPHA,looks);
  return {n,beyond,neff,p,dbar,M,looks,count:p<level,common:Math.abs(dbar)>M,reject:p<level||Math.abs(dbar)>M};
}

/** One expectation against one result: rows per series, totals, the verdict. */
export function check(e,result,request){
  const r=result,m=r.motion??{},N=m.trajectories??NaN,loss=m.lossProbability??[],errors=r.populationAccuracy?.sampleErrors??[];
  const tLedger=r.diagnostics?.approximations?.totals?.temperature?.error??0,mass=request?.model?.motion?.massKg;
  const k=s=>r.states.indexOf(s),held=i=>N*(1-(loss[i]??0));
  // The degrees of freedom of T's jackknife at an output (the header): the most groups its held atoms lie in, less one.
  const G=Number(/每 (\d+) 条一组/.exec(r.diagnostics?.method??'')?.[1]),freedom=i=>G>0?Math.max(1,Math.min(N/G,held(i))-1):Infinity;
  const temperature=(s,i)=>{
    if(s.total){const se=m.temperatureTotalErrorUK?.[i];return {x:m.temperatureTotalUK?.[i],se:se==null?Infinity:se};}
    const axes=s.axes??[s.axis],se=axes.map(a=>m.temperatureErrorUK[a][i]);
    return {x:axes.reduce((t,a)=>t+m.temperatureUK[a][i],0)/axes.length,se:se.some(e=>e==null)?Infinity:se.reduce((t,e)=>t+e,0)/axes.length};};
  // The result's total E (populations, groups, loss) in its statistical part, z·SE and no more than E, and the rest.
  const split=(E,se)=>{const random=Math.min(E,Z*se);return {fixed:Math.sqrt(Math.max(0,E*E-random*random)),random};};
  const rows=[];let judged=0,failed=0,unresolved=0,skipped=0,rejected=0,shown=0;
  if(r.timesUs?.length!==e.samples||Math.abs(r.timesUs.at(-1)-e.durationUs)>1e-9*e.durationUs)return {error:`grid: ${r.timesUs?.length} times to ${r.timesUs?.at(-1)} µs, expected ${e.samples} to ${e.durationUs}`};
  for(const s of e.series){
    const name=(s.reported?'* ':'')+s.quantity+(s.total?' total':'')+(s.axis?` ${s.axis}`:'')+(s.axes?` ${s.axes.join('+')}`:'')+(s.state?` ${s.state}`:'')+(s.label?` ${s.label}`:''),count=!s.reported;
    const outputs=s.indices?s.indices.map((i,j)=>({i,ref:s.values[j],u:s.u[j],judge:s.judged?s.judged[j]:true})):[{i:null,ref:s.value,u:s.u,judge:true}];
    const row={name,own:!s.target,counted:count,n:0,maxD:0,worst:-Infinity,worstAt:null,fail:0,unresolved:0,skipped:0,claims:[],test:null};
    for(const o of outputs){
      let x,se,claim=null,nu=Infinity;const i=o.i;
      switch(s.quantity){
        case 'population':x=r.populations[i][k(s.state)];se=Math.sqrt(x*(1-x)/held(i));claim=split(errors[i],se);break;
        case 'group':x=s.states.reduce((t,st)=>t+r.populations[i][k(st)],0);se=Math.sqrt(Math.max(0,x*(1-x))/held(i));claim=split(errors[i],se);break;
        case 'loss':x=loss[i];se=Math.sqrt(x*(1-x)/N);claim=split(errors[i],se);break;
        case 'T':{({x,se}=temperature(s,i));nu=freedom(i);claim={fixed:tLedger*x,random:quantile(BEYOND,nu)*se,freedom:nu};
          if(s.regime==='classical'&&!m.classical[i])o.judge=false;break;}
        case 'Tincrease':{({x,se}=temperature(s,i));x-=temperature(s,0).x;nu=freedom(i);break;}
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
      const d=x-o.ref,band=Math.hypot(quantile(BEYOND,nu)*se,Z*o.u);
      row.n++;row.maxD=Math.max(row.maxD,Math.abs(d));if(!count)shown++;
      // Rule 0: against its own errors (the tests below); the score is |d| over the band. Rules 1 and 2: the target, then the claim.
      const target=row.own?null:s.target.absolute??(s.target.relative*Math.abs(o.ref)),score=row.own?(band>0?Math.abs(d)/band:d?Infinity:0):(Math.abs(d)-band)/target;
      if(score>row.worst){row.worst=score;row.worstAt={i,t:r.timesUs[i??0],x,ref:o.ref,se,u:o.u};}
      if(row.own){row.claims.push({d,u:o.u,fixed:0,random:quantile(BEYOND,nu)*se,freedom:nu});if(count)judged++;continue;}
      if(claim)row.claims.push({d,u:o.u,...claim});
      if(score>1){row.fail++;if(count){failed++;judged++;}}
      else if(band>target){row.unresolved++;if(count)unresolved++;}
      else if(count)judged++;
    }
    rows.push(row);
  }
  // Rule 3: the family -- the series judged whose claims hold a statistical part -- and each one's tests at its tails.
  const chance=row=>row.counted&&row.claims.some(c=>c.random>0||c.u>0),family=rows.filter(chance).length;
  for(const row of rows){
    if(row.claims.length){row.test=consistency(row.claims,chance(row)?family:1);if(row.test.reject&&row.counted)rejected++;}
    delete row.claims;
  }
  const judgedKind=!['characterization','regression-consistency'].includes(e.kind);
  const verdict=!judged?'NOT JUDGED':!judgedKind?'REPORT':failed||rejected?'FAIL':'PASS';
  return {rows,judged,failed,rejected,unresolved,skipped,shown,family,verdict};
}

/** A relation between the steady states of several results (kind "relation": R4's scaling law, expect.mjs
 * r4linear): D = Σ c_k T̄_k, T̄_k the mean of member k's T on the relation's axis over its window, against `value`.
 * A window mean's SE is bounded by the mean of its outputs' (fully correlated in time); the members are separate runs,
 * so SE_D² = Σ c_k² SE_k². The scale is the T̄ of the member `scale`: u is relative to it, and adds to the scenes' part
 * what only the results tell -- the trap light's line shift over the cloud against the detuning, κk_BT̄/ħ|δ|, and the
 * polarization pattern's first harmonic over the coldest cloud, e^{−2k²σ²}, σ² = k_BT̄/mω².
 * D is no temperature and has no user's target: it is judged against the reference's own allowance and the results'
 * statistics alone (the header's rule 0 for one output), FAIL where |D − value| > z√(SE_D² + u²). What that resolves
 * is printed: zσ over the scale, and the exponents α of T ∝ I^α whose D it could not tell from zero.
 * The statement is of steady states and has content only where T changes along the members: a member whose T̄ over
 * its window's two halves differ by more than z of their standard errors, one the result does not call classical
 * throughout its window, or a rise T̄_last − T̄_first within z of its standard error (the second difference of a
 * constant vanishes whatever the physics) leaves the relation NOT JUDGED, said with the numbers; a fall beyond z of
 * its standard error contradicts the law (c > 0): FAIL. The coverage of rule 2 (|D| against the members' reported
 * errors, B² = (z SE_D)² + (Σ|c_k|·ledger_k·T̄_k)² + (zu)²) is one output and only reported. */
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
  const u=scale*(e.u.relative+shift+pattern),sigma=Math.hypot(se,u),d=D-e.value;
  const B=Math.hypot(Z*se,members.reduce((t,m,k)=>t+Math.abs(c[k])*m.ledger*m.x,0),Z*u);
  const first=members[0],last=members.at(-1),rise=last.x-first.x,riseSe=Math.hypot(first.se,last.se);
  const reasons=[...members.filter(m=>!m.classical).map(m=>`${m.id}: the result does not call its window classical throughout`),...members.filter(m=>!m.steady).map(m=>`${m.id}: not steady (T̄ ${m.early.x.toFixed(2)} ± ${m.early.se.toFixed(2)} then ${m.late.x.toFixed(2)} ± ${m.late.se.toFixed(2)} µK over the window's halves)`),
    ...members.filter(m=>!Number.isFinite(m.se)).map(m=>`${m.id}: no error estimate`),...(Math.abs(rise)<=Z*riseSe?[`no rise resolved: T̄(${last.id}) − T̄(${first.id}) = ${rise.toFixed(2)} ± ${riseSe.toFixed(2)} µK (the second difference of a constant says nothing)`]:[])];
  // T ∝ I^α over the members' intensities gives D/T_scale = Σ c_k (I_k/I_scale)^α: the α the allowance cannot tell from 0.
  const I=i.intensities,power=a=>c.reduce((t,q,k)=>t+q*(I[k]/I[e.scale])**a,0),grid=Array.from({length:401},(_,j)=>j/100),open=grid.filter(a=>Math.abs(power(a))<=Z*sigma/scale);
  const falls=rise<-Z*riseSe;
  return {members,D,se,u,terms:{scenes:e.u.relative,shift,pattern},sigma,resolves:Z*sigma/scale,exponents:[open[0],open.at(-1)],rise,riseSe,covered:Math.abs(d)<=B,reasons,
    verdict:falls?'FAIL':reasons.length?'NOT JUDGED':Math.abs(d)>Z*sigma?'FAIL':'PASS',falls};
}

function printRelation(e,c){
  console.log(`${e.id} [${e.reference}, ${e.kind}] ${e.title}`);
  for(const m of c.members)console.log('  '+m.id.padEnd(12)+(m.error??`T̄_${e.axis} ${m.x.toFixed(3)} ± ${m.se.toFixed(3)} µK from ${m.from} µs (halves ${m.early.x.toFixed(2)}, ${m.late.x.toFixed(2)}; T(0) ${m.start.toFixed(2)}) | ${m.classical?'classical':'NOT CLASSICAL'}, ${m.steady?'steady':'NOT STEADY'} | ${m.trajectories} trajectories, step ${m.step} µs, ledger ${(100*m.ledger).toFixed(1)} %`));
  if(c.verdict==='NO RESULT'){console.log('  NO RESULT\n');return;}
  const U=e.info?.modulationUK,first=c.members[0],last=c.members.at(-1),slope=U?(last.x-first.x)/(U.at(-1)-U[0]):null;
  console.log(`  ${c.verdict}: D = ${e.coefficients.map((q,k)=>`${q>0?'+':'−'}${Math.abs(q)}·T̄(${e.members[k].id})`).join(' ')} = ${c.D.toFixed(3)} µK against ${e.value}; SE ${c.se.toFixed(3)}, u ${c.u.toFixed(3)} (relative to T̄ ${c.members[e.scale].x.toFixed(2)}: scenes ${c.terms.scenes.toFixed(4)}, line shift ${c.terms.shift.toFixed(4)}, pattern ${c.terms.pattern.toExponential(1)}); allowance zσ = ${(Z*c.sigma).toFixed(3)} µK`);
  console.log(`  |D|/zσ = ${(Math.abs(c.D-e.value)/(Z*c.sigma)).toFixed(2)}; resolves a second difference of ${(100*c.resolves).toFixed(1)} % of T̄(${e.members[e.scale].id}): T ∝ I^α is not told from the law for α in [${c.exponents[0]}, ${c.exponents[1]}]; rise ${c.rise.toFixed(2)} ± ${c.riseSe.toFixed(2)} µK${c.falls?' (FALLS)':''}; reported errors ${c.covered?'cover':'DO NOT cover'} D`+(slope===null?'':`; line through the outer two: T̄ = ${slope.toFixed(4)}·U + ${(first.x-slope*U[0]).toFixed(2)} µK (U the light potential's modulation)`));
  for(const r of c.reasons)console.log(`  not judged: ${r}`);
  console.log('');
}

function print(e,c,run){
  const head=`${e.id} [${e.reference}, ${e.kind}] ${e.title}`;
  if(c.error){console.log(`${head}\n  ${c.error}\n`);return;}
  console.log(`${head}\n  ${c.verdict}: ${c.judged} outputs judged; rule 1 (beyond target): ${c.failed}; rules 0 and 2 (inconsistent with the errors): ${c.rejected} series of a family of ${c.family}; ${c.unresolved} unresolved (band > target); ${c.skipped} not judged${c.shown?`; ${c.shown} reported (*)`:''}`+
    (run?` | run ${run.wallS.toFixed(1)} s${run.aborted?' ABORTED':''}, first checkpoint ${run.firstCheckpointS?.toFixed(1)??'-'} s, ${run.final?.result?.motion?.trajectories??'-'} trajectories`:''));
  console.log('  '+['series'.padEnd(30),'n'.padStart(4),'max|d|'.padStart(10),'score'.padStart(8),'fail'.padStart(5),'unres'.padStart(6),'out'.padStart(5),'n_eff'.padStart(7),'p'.padStart(9),'|d̄|/M'.padStart(8),'  worst output (t: solver vs ref ± se, u)'].join(''));
  for(const r of c.rows){const w=r.worstAt,t=r.test;
    console.log('  '+[r.name.slice(0,30).padEnd(30),String(r.n).padStart(4),r.maxD.toPrecision(3).padStart(10),(Number.isFinite(r.worst)?r.worst.toFixed(2)+(r.own?'°':' '):'-').padStart(8),(r.own?'-':String(r.fail)).padStart(5),(r.own?'-':String(r.unresolved)).padStart(6),
      (t?String(t.beyond):'-').padStart(5),(t?t.neff.toFixed(1):'-').padStart(7),(t?t.p.toPrecision(2)+(t.count?'!':' '):'-').padStart(9),(t?(t.M>0?(Math.abs(t.dbar)/t.M).toFixed(2):t.dbar?'inf':'0.00')+(t.common?'!':' '):'-').padStart(8),
      w?`  t=${w.t.toPrecision(4)}: ${w.x.toPrecision(5)} vs ${w.ref.toPrecision(5)} ± ${w.se.toPrecision(2)}, ${w.u.toPrecision(2)}`:''].join(''));}
  console.log('  score: (|d| − band)/target, beyond 1 fails, the band q·SE ⊕ 2u (q: 2, or Student\'s t at the groups less one for T); ° no user\'s target: |d|/band, judged by the two tests at the family\'s Šidák tails (out, n_eff, p: the count beyond the bound against n_eff independent outputs, n_eff from the residual\'s autocorrelation about zero -- an offset the whole run shares is one draw; |d̄|/M: the mean deviation over its bound; ! rejects)\n');
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
    // A reference the worker cannot run (kind "tool"): its own command runs and judges it, and keeps the verdict.
    if(e.kind==='tool'){const rf=path.join(here,'out','results',`${e.id}.json`),v=fs.existsSync(rf)?read(rf).verdict:'NO RESULT';
      console.log(`${e.id} [${e.reference}, ${e.kind}] ${e.title}\n  ${v} (${e.tool.command}${e.reported?'; reported, never judged':''})\n`);summary.push([e.id,v]);if(v==='FAIL')bad++;continue;}
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
