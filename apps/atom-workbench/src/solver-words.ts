// The solver's words in the workbench's vocabulary.  The native result keeps
// its own (English) status names and method sentences; only what is shown is
// translated here, so the result contract and its validation are untouched.
import {formatNumber} from './format.js';

/** Sentences the native solver is known to write (rust-solver/src/lattice_core_adapter.rs and
 * motion_adapter.rs; rust-solver/motion), and the client's own. */
const SENTENCES:Record<string,string>={
  "Full32 density-matrix Lindblad dynamics on the folded Fourier lattice (explicit charges by mixing, seed amplitude and ring condition, and their ring; far harmonics folded statically, the pieces negligible over the run left out): one rational Krylov space from the initial state and its translations, every output read from one Galerkin reduced model, each output's error estimated by its change between consecutive checkpoints":
    '完整 32 态密度矩阵在折叠 Fourier 晶格上的 Lindblad 动力学（显式电荷按混合、种子幅度与环条件选取，连同其环；远离的谐波静态折叠，在本次计算时长内可忽略的部分略去）：从初态及其平移出发的一个有理 Krylov 空间，每个输出都由同一个 Galerkin 约化模型读出，误差按输出在相邻两个检查点之间的变化估计。',
  'Phase average at the horizon (horizonUs) of the full driven Lindblad equation from the specified initial density: the state at T with only its fast oscillations (beats, optical micromotion) averaged out, not a time-independent density.':
    '完整受驱 Lindblad 方程从给定初始密度矩阵出发在时刻 T 的相位平均：T 时刻的态只平均掉快振荡（拍频与光频微运动），不是不随时间变化的密度矩阵。',
  'State at the horizon (horizonUs) of the undriven Lindblad equation from the specified initial density, its optical coherences averaged out in the laboratory frame.':
    '无驱动 Lindblad 方程从给定初始密度矩阵出发在时刻 T 的态，在实验室系中平均掉光学相干。',
  'Cancelled.':'计算已取消。',
  'The trajectory duration must be a positive finite time':'轨迹时长必须是正的有限时间。',
  'meanHorizonUs must be 1e5 or 1e6 us (100 ms or 1 s)':'长期布居的时刻 T 只能是 100 ms 或 1 s。',
  'A motion request is a trajectory alone: meanOnly and includeMean are false':'运动只计算时间演化，不计算长期布居。',
  'Returned population probabilities failed fixed trace/initial checks':'算出的布居没有通过检查（各态布居之和不是 1，或初始时刻与初态不符），结果未显示。',
};
/** What a mean's reason adds when its bound is above the tolerance (lattice_core_adapter.rs `OUTSIDE`). */
const OUTSIDE=' Not converged: its error estimate (estimatedError) is above the tolerance, so the populations are an estimate.';
const OUTSIDE_WORDS='未收敛：误差估计高于误差目标，布居只是估计值。';
/** A number the solver wrote (Rust {:e} or plain), as the workbench writes numbers. */
const num=(text:string):string=>{const value=Number(text);return Number.isFinite(value)?formatNumber(value,{digits:3}):text;};
/** Why a run stopped (rust-solver population-core lattice/pipeline.rs). */
const WHY:readonly [RegExp,(...groups:string[])=>string][]=[
  [/^the space stopped growing after (\d+) solves$/,n=>`空间在 ${n} 次求解后不再增长`],
  [/^the basis memory was spent after (\d+) solves$/,n=>`基的内存在 ${n} 次求解后用尽`],
  [/^the solve budget \((\d+)\) was spent$/,n=>`求解次数用满（${n} 次）`],
];
const why=(text:string):string=>{for(const [pattern,words] of WHY){const match=pattern.exec(text);if(match)return words(...match.slice(1));}return text;};
/** Reasons the native solver formats (pipeline.rs, lattice_core_adapter.rs). */
const PATTERNS:readonly [RegExp,(...groups:string[])=>string][]=[
  [/^Stopped because (.+?): the best checkpoint \((\d+) solves\) has an error estimate of (\S+) against the tolerance (\S+?)(?:; (\d+) of (\d+) samples above it, t = (\S+) to (\S+) us)?(?:; (\d+) mean output\(s\) above it)?$/,
    (reason,n,bound,tolerance,over,samples,from,to,means)=>[`${why(reason)}，计算停止：最好的检查点（${n} 次求解）误差估计为 ${num(bound)}，误差目标 ${num(tolerance)}`,
      ...(over?[`${samples} 个采样点中有 ${over} 个超过误差目标（t = ${from===to?num(from):`${num(from)} 到 ${num(to)}`} µs）`]:[]),...(means?[`${means} 个长期布居输出超过误差目标`]:[])].join('；')+'。'],
  [/^Checkpoint at (\d+) solves: the error estimate is (\S+) against the tolerance (\S+); the run continues$/,
    (n,bound,tolerance)=>`第 ${n} 次求解的检查点：误差估计 ${num(bound)}，误差目标 ${num(tolerance)}；计算继续。`],
  [/^The mean's error estimate is (\S+), at least 1: it carries no information, so no populations are shown$/,
    bound=>`长期布居的误差估计为 ${num(bound)}（不小于 1），不含任何信息，不显示布居。`],
  [/^Far shell at (\S+) rad\/us is not separated from L0's spectrum \(\|L0\| <= (\S+) rad\/us\): not solved$/,
    (at,bound)=>`未求解（超出模型范围）：远失谐光的组合在 ${num(at)} rad/µs 处落入中心生成元的谱（‖L0‖ ≤ ${num(bound)} rad/µs），远壳不再是远壳。`],
  [/^The far shell of this request would need about (\S+) GB, above its (\S+) GB share of the memory budget: not solved$/,
    (need,budget)=>`未求解（不收敛）：此请求的远壳约需 ${need} GB 内存，超过它所分得的 ${budget} GB 内存预算。`],
  [/^The lattice of this request has (\d+) modes and would need about (\S+) GB, above the (\S+) GB memory budget: not solved$/,
    (modes,need,budget)=>`未求解（不收敛）：此请求的晶格有 ${modes} 个模式，约需 ${need} GB 内存，超过 ${budget} GB 的内存预算。`],
  [/^The trajectory duration is limited to 10 s \(requested (\S+) s\)$/,seconds=>`轨迹时长最长 10 s（请求为 ${num(seconds)} s），未计算。`],
  [/^A trajectory has 2 to (\d+) samples \(requested (\d+)\)$/,(most,samples)=>`轨迹的采样点数应为 2 至 ${most}（请求为 ${samples}），未计算。`],
  [/^An output time is at most (\S+) us \(10 s; requested (\S+) us\)$/,(most,requested)=>`输出时刻最长 10 s（${num(most)} µs；请求为 ${num(requested)} µs），未计算。`],
];
/** A solver sentence in Chinese; one this table does not know is shown as written. */
export function solverText(text:string):string{
  const trimmed=text.trim(),known=SENTENCES[trimmed];if(known)return known;
  if(trimmed.endsWith(OUTSIDE)){const base=solverText(trimmed.slice(0,-OUTSIDE.length));if(base!==trimmed.slice(0,-OUTSIDE.length))return base+OUTSIDE_WORDS;}
  for(const [pattern,words] of PATTERNS){const match=pattern.exec(trimmed);if(match)return words(...match.slice(1));}
  return text;
}
