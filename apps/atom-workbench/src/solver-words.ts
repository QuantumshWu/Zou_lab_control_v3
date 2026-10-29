// The solver's words in the workbench's vocabulary.  The native result keeps
// its own (English) status names and method sentences; only what is shown is
// translated here, so the result contract and its validation are untouched.
import {formatNumber} from './format.js';

/** Sentences the native solver is known to write (rust-solver/src/lattice_core_adapter.rs), and the client's own. */
const SENTENCES:Record<string,string>={
  "Full32 density-matrix Lindblad dynamics on the folded Fourier lattice (explicit charges by mixing, seed amplitude and ring condition, and their ring; far harmonics folded statically): one rational Krylov space from the initial state, every output read from one Galerkin reduced model in the Schrieffer-Wolff/Floquet frame's metric, its error estimated per output from the reduced model's residual, the omitted ring and the fold rests the lattice leaves out":
    '完整 32 态密度矩阵在折叠 Fourier 晶格上的 Lindblad 动力学（显式电荷按混合、种子幅度与环条件选取，连同其环；远离的谐波静态折叠）：从初态出发的一个有理 Krylov 空间，每个输出都由 Schrieffer–Wolff/Floquet 框架度量下的同一个 Galerkin 约化模型读出，误差按输出由约化模型的残差、省略的环和晶格留下的折叠余项估计。',
  'Phase average at the horizon (horizonUs) of the full driven Lindblad equation from the specified initial density: the state at T with only its fast oscillations (beats, optical micromotion) averaged out, not a time-independent density.':
    '完整受驱 Lindblad 方程从给定初始密度矩阵出发在时刻 T 的相位平均：T 时刻的态只平均掉快振荡（拍频与光频微运动），不是不随时间变化的密度矩阵。',
  'State at the horizon (horizonUs) of the undriven Lindblad equation from the specified initial density, its optical coherences averaged out in the laboratory frame.':
    '无驱动 Lindblad 方程从给定初始密度矩阵出发在时刻 T 的态，在实验室系中平均掉光学相干。',
  'Cancelled.':'计算已取消。',
};
/** What a mean's reason adds when its bound is above the tolerance (lattice_core_adapter.rs `OUTSIDE`). */
const OUTSIDE=' Not converged: its error bound (estimatedError) is above the tolerance, so the populations are an estimate.';
const OUTSIDE_WORDS='未收敛：误差界高于误差目标，布居只是估计值。';
/** A number the solver wrote (Rust {:e} or plain), as the workbench writes numbers. */
const num=(text:string):string=>{const value=Number(text);return Number.isFinite(value)?formatNumber(value,{digits:3}):text;};
/** Why a run stopped (rust-solver population-core lattice/pipeline.rs). */
const WHY:readonly [RegExp,(...groups:string[])=>string][]=[
  [/^the error bound stopped improving \((\S+)x over (\d+) checkpoints\) after (\d+) solves$/,(gain,k,n)=>`误差界在最近 ${k} 个检查点内没有改善到 1/${gain}（共 ${n} 次求解）`],
  [/^the basis memory was spent after (\d+) solves$/,n=>`基的内存在 ${n} 次求解后用尽`],
  [/^the pole ladder was exhausted after (\d+) solves$/,n=>`极点阶梯在 ${n} 次求解后用尽`],
  [/^the solve budget \((\d+)\) was spent$/,n=>`求解次数预算（${n} 次）已用完`],
];
const why=(text:string):string=>{for(const [pattern,words] of WHY){const match=pattern.exec(text);if(match)return words(...match.slice(1));}return text;};
/** Reasons the native solver formats (pipeline.rs, lattice_core_adapter.rs). */
const PATTERNS:readonly [RegExp,(...groups:string[])=>string][]=[
  [/^Stopped because (.+?): the best checkpoint \((\d+) solves\) has an error bound of (\S+) against the tolerance (\S+?)(?:; (\d+) of (\d+) samples above it, t = (\S+) to (\S+) us)?(?:; (\d+) mean output\(s\) above it)?$/,
    (reason,n,bound,tolerance,over,samples,from,to,means)=>[`${why(reason)}，计算停止：最好的检查点（${n} 次求解）误差界为 ${num(bound)}，误差目标 ${num(tolerance)}`,
      ...(over?[`${samples} 个采样点中有 ${over} 个超过误差目标（t = ${from===to?num(from):`${num(from)} 到 ${num(to)}`} µs）`]:[]),...(means?[`${means} 个长期布居输出超过误差目标`]:[])].join('；')+'。'],
  [/^Checkpoint at (\d+) solves: the error bound is (\S+) against the tolerance (\S+); the run continues$/,
    (n,bound,tolerance)=>`第 ${n} 次求解的检查点：误差界 ${num(bound)}，误差目标 ${num(tolerance)}；计算继续。`],
  [/^The mean's error bound is (\S+), at least 1: it carries no information, so no populations are shown$/,
    bound=>`长期布居的误差界为 ${num(bound)}（不小于 1），不含任何信息，不显示布居。`],
  [/^The lattice of this request has (\d+) modes and would need about (\S+) GB, above the (\S+) GB memory budget: not solved$/,
    (modes,need,budget)=>`未求解（不收敛）：此请求的晶格有 ${modes} 个模式，约需 ${need} GB 内存，超过 ${budget} GB 的内存预算。`],
  [/^Static fold expansion parameter (\S+) is above (\S+): the fold's frequency expansion is not small on this lattice, not solved$/,
    (parameter,limit)=>`未求解（不收敛）：静态折叠的展开参数为 ${num(parameter)}，高于 ${num(limit)}，折叠的频率展开在此晶格上不够小。`],
];
/** A solver sentence in Chinese; one this table does not know is shown as written. */
export function solverText(text:string):string{
  const trimmed=text.trim(),known=SENTENCES[trimmed];if(known)return known;
  if(trimmed.endsWith(OUTSIDE)){const base=solverText(trimmed.slice(0,-OUTSIDE.length));if(base!==trimmed.slice(0,-OUTSIDE.length))return base+OUTSIDE_WORDS;}
  for(const [pattern,words] of PATTERNS){const match=pattern.exec(trimmed);if(match)return words(...match.slice(1));}
  return text;
}
