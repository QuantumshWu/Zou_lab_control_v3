// The solver's words in the workbench's vocabulary.  The native result keeps
// its own (English) status names and method sentences; only what is shown is
// translated here, so the result contract and its validation are untouched.
import type {SteadyStatus} from './solver-types.js';
import {formatNumber} from './format.js';

const STATUS:Record<SteadyStatus,string>={
  'stationary':'稳态',
  'population-stationary':'布居稳态',
  'initial-dependent':'稳态（依赖初态）',
  'time-average':'长期时间平均',
  'initial-dependent-time-average':'长期时间平均（依赖初态）',
  'not-converged':'未收敛',
  'not-requested':'未计算',
};
/** The long-time result's kind, as a word. */
export function steadyStatusText(status:SteadyStatus|undefined):string{return status?STATUS[status]??status:STATUS['not-converged'];}

/** Sentences the native solver is known to write (rust-solver/src/population_core_adapter.rs). */
const SENTENCES:Record<string,string>={
  'Full32 density-matrix Lindblad evolution with adaptive population refinement':
    '完整 32 态密度矩阵 Lindblad 演化，布居误差自适应加密。',
  'Full32 Cesàro mean of the gauge pencil: defect-corrected null vector with a near-zero pole, halo-condensed Fourier truncation':
    '完整 32 态规范束的 Cesàro 平均：近零极点的零空间向量经缺陷校正，Fourier 截断凝聚晕模。',
  'Full32 Cesàro mean of the gauge pencil: defect-corrected null vector with a near-zero pole below the slowest relaxation of the slow generator, halo-condensed Fourier truncation':
    '完整 32 态规范束的 Cesàro 平均：取低于慢生成元最慢弛豫的近零极点，零空间向量经缺陷校正，Fourier 截断凝聚晕模。',
  'Converged long-time phase average of the full driven Lindblad equation (unique limit: only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time); populations are a time average, not a time-independent density.':
    '完整受驱 Lindblad 方程的长期相位平均已收敛（极限唯一：只有迹守恒，最慢弛豫时间见上）；布居是时间平均，不是不随时间变化的密度矩阵。',
  'Unique full-density Lindblad stationary solution (only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time).':
    '完整密度矩阵的 Lindblad 稳态唯一（只有迹守恒，最慢弛豫时间见上）。',
  'Converged long-time phase average of the full driven Lindblad equation (one-dimensional null space); populations are a time average, not a time-independent density.':
    '完整受驱 Lindblad 方程的长期相位平均已收敛（一维零空间）；布居是时间平均，不是不随时间变化的密度矩阵。',
  'Converged long-time phase average of the specified initial density; several conserved quantities make it initial-state dependent.':
    '给定初始密度矩阵的长期相位平均已收敛；存在多个守恒量，结果依赖初态。',
  'Long-time phase average at the observation horizon (horizonUs) of the specified initial density: slow modes still relaxing at the horizon (or conserved quantities) keep it initial-state dependent.':
    '给定初始密度矩阵在观察时长 T 处的长期相位平均：到 T 仍在弛豫的慢模式（或守恒量）使结果依赖初态。',
  'Lindblad mean at the observation horizon (horizonUs) of the specified initial density: slow modes still relaxing at the horizon (or conserved quantities) keep it initial-state dependent.':
    '给定初始密度矩阵在观察时长 T 处的 Lindblad 平均：到 T 仍在弛豫的慢模式（或守恒量）使结果依赖初态。',
  'Populations converge in the background basis; exported density is the laboratory-frame time average.':
    '布居在背景基中收敛；导出的密度矩阵是实验室系的时间平均。',
  'Unique full-density Lindblad stationary solution within the resolved rates.':
    '在已分辨的速率内，完整密度矩阵的 Lindblad 稳态唯一。',
  'Full Lindblad stationary projection from the specified initial density.':
    '由给定初始密度矩阵得到的完整 Lindblad 稳态投影。',
  'Cancelled.':'计算已取消。',
};
/** A number the solver wrote (Rust {:e} or plain), as the workbench writes numbers. */
const num=(text:string):string=>{const value=Number(text);return Number.isFinite(value)?formatNumber(value,{digits:3}):text;};
const KEPT='误差目标未放宽。';
/** Failure reasons the native solver formats (rust-solver/population-core). */
const PATTERNS:readonly [RegExp,(...groups:string[])=>string][]=[
  [/^(\d+) conserved quantities together with a relaxation of (\S+) us cannot be resolved by the near-zero pole; the tolerance was not relaxed$/,(n,t)=>`${n} 个守恒量与 ${num(t)} μs 的弛豫无法由近零极点分开；${KEPT}`],
  [/^The resolvent solve at pole (\S+)\/us stalled at relative residual (\S+) \(requested (\S+)\); the tolerance was not relaxed$/,(p,r,t)=>`近零极点 ${num(p)}/μs 处的预解求解停滞在相对残差 ${num(r)}（要求 ${num(t)}）；${KEPT}`],
  [/^Long-time mean estimate (\S+) exceeds the tolerance after the density projection; the tolerance was not relaxed$/,e=>`密度投影后长期平均的误差估计 ${num(e)} 超过误差目标；${KEPT}`],
  [/^Long-time mean work limit reached with (\d+) columns on (\d+) Fourier modes; the tolerance was not relaxed$/,(c,m)=>`长期平均达到工作量上限（${m} 个 Fourier 模式、${c} 列）；${KEPT}`],
  [/^Long-time mean needs more than (\d+) Fourier modes; the tolerance was not relaxed$/,m=>`长期平均需要超过 ${m} 个 Fourier 模式；${KEPT}`],
  [/^Long-time mean trial space limit of (\d+) columns reached; the tolerance was not relaxed$/,c=>`长期平均的试探空间达到 ${c} 列上限；${KEPT}`],
  [/^The long-time mean trial space cannot be extended further$/,()=>'长期平均的试探空间无法再扩展。'],
  [/^Initial-dependent long-time mean did not reach the tolerance; the tolerance was not relaxed$/,()=>`依赖初态的长期平均未达到误差目标；${KEPT}`],
  [/^Population work limit reached after (\S+) s with (\d+) Fourier modes and (\d+) columns; the tolerance was not relaxed$/,(s,m,c)=>`布居计算 ${num(s)} s 后达到工作量上限（${m} 个 Fourier 模式、${c} 列）；${KEPT}`],
  [/^Trial space limit of (\d+) columns reached with (\d+) Fourier modes; the tolerance was not relaxed$/,(c,m)=>`试探空间达到 ${c} 列上限（${m} 个 Fourier 模式）；${KEPT}`],
  [/^The trial space cannot be extended further$/,()=>'试探空间无法再扩展。'],
  [/^Nonfinite population error series$/,()=>'布居误差级数出现非有限值。'],
  [/^Nonfinite population error estimate$/,()=>'布居误差估计出现非有限值。'],
  [/^Reduced generator eigenbasis is unresolved \(defect (\S+), amplification (\S+)\)$/,(d,a)=>`约化生成元的本征基未解析（缺陷 ${num(d)}，放大 ${num(a)}）。`],
];
/** A solver sentence in Chinese; one this table does not know is shown as written. */
export function solverText(text:string):string{
  const trimmed=text.trim(),known=SENTENCES[trimmed];if(known)return known;
  for(const [pattern,words] of PATTERNS){const match=pattern.exec(trimmed);if(match)return words(...match.slice(1));}
  return text;
}
