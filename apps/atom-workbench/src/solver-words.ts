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

/** Sentences the native solver is known to write (rust-solver/src/lattice_core_adapter.rs). */
const SENTENCES:Record<string,string>={
  "Full32 density-matrix Lindblad evolution on the folded Fourier lattice: rational Krylov with lattice-translation augmentation, its basis, near-shell and fold errors estimated by propagating their sources through the dressed blocks; beyond 1 ms the slow eigenmodes' expansion from the Krylov state at a switch time":
    '完整 32 态密度矩阵在折叠 Fourier 晶格上的 Lindblad 演化：带晶格平移增广的有理 Krylov 方法，基、近壳层与折叠误差由各自的源经缀饰块传播估计；1 ms 以后用切换时刻 Krylov 态的慢本征模展开。',
  "Full32 time average of the folded Fourier lattice (near charges explicit, far harmonics folded statically at second order, higher orders charged): dressed-block preconditioned null vector, its truncation and fold errors estimated in the solver's Krylov space":
    '完整 32 态折叠 Fourier 晶格上的时间平均（近共振电荷显式，远离的谐波按二阶静态折叠，更高阶计入误差）：缀饰块预条件的零空间向量，截断与折叠误差在求解器的 Krylov 空间中估计。',
  'Converged long-time phase average of the full driven Lindblad equation (unique limit: only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time); populations are a time average, not a time-independent density.':
    '完整受驱 Lindblad 方程的长期相位平均已收敛（极限唯一：只有迹守恒，最慢弛豫时间见上）；布居是时间平均，不是不随时间变化的密度矩阵。',
  'Unique full-density Lindblad stationary solution (only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time).':
    '完整密度矩阵的 Lindblad 稳态唯一（只有迹守恒，最慢弛豫时间见上）。',
  'Long-time phase average at the observation horizon (horizonUs) of the specified initial density: slow modes still relaxing at the horizon (or conserved quantities) keep it initial-state dependent.':
    '给定初始密度矩阵在观察时长 T 处的长期相位平均：到 T 仍在弛豫的慢模式（或守恒量）使结果依赖初态。',
  'Lindblad mean at the observation horizon (horizonUs) of the specified initial density: slow modes still relaxing at the horizon (or conserved quantities) keep it initial-state dependent.':
    '给定初始密度矩阵在观察时长 T 处的 Lindblad 平均：到 T 仍在弛豫的慢模式（或守恒量）使结果依赖初态。',
  'Populations converge in the background basis; exported density is the laboratory-frame time average.':
    '布居在背景基中收敛；导出的密度矩阵是实验室系的时间平均。',
  'Cancelled.':'计算已取消。',
};
/** A number the solver wrote (Rust {:e} or plain), as the workbench writes numbers. */
const num=(text:string):string=>{const value=Number(text);return Number.isFinite(value)?formatNumber(value,{digits:3}):text;};
const KEPT='误差目标未放宽。';
/** The lattice's error equations, by their labels. */
const EQUATION:Record<string,string>={truncation:'截断',fold:'折叠','third-order':'三阶折叠',correction:'壳层修正'};
/** Failure reasons the native solver formats (rust-solver). */
const PATTERNS:readonly [RegExp,(...groups:string[])=>string][]=[
  [/^(\d+) conserved quantities: the mean depends on the initial state, which the lattice core does not serve$/,n=>`${n} 个守恒量：长期平均依赖初态，求解器不计算这种长期平均。`],
  [/^The mean at the horizon is estimated at (\S+); the tolerance was not relaxed$/,e=>`观察时长处的长期平均误差估计为 ${num(e)}；${KEPT}`],
  [/^Finite-time populations estimated at (\S+); the tolerance was not relaxed$/,e=>`时间演化布居的误差估计为 ${num(e)}；${KEPT}`],
  [/^Lattice steady state not accepted up to near order (\d+) \(estimated error (\S+)\)$/,(k,e)=>`长期平均到近阶 ${k} 仍未达到误差目标（误差估计 ${num(e)}）；${KEPT}`],
  [/^Lattice steady state not accepted: estimated error (\S+) above (\S+) with no near shell to add$/,e=>`长期平均的误差估计 ${num(e)} 超过误差目标，且没有可加入的近壳层；${KEPT}`],
  [/^Finite-time populations not accepted up to near order (\d+)$/,k=>`时间演化布居到近阶 ${k} 仍未达到误差目标；${KEPT}`],
  [/^Error equation (truncation|fold|third-order|correction) did not stabilise in (\d+) augmentation steps$/,(label,n)=>`${EQUATION[label]}误差方程在 ${n} 步增广内未稳定；${KEPT}`],
  [/^Finite-time estimate (\S+) not below the tolerance with (\d+) columns$/,(e,c)=>`时间演化的误差估计 ${num(e)} 在 ${c} 列时仍未低于误差目标；${KEPT}`],
];
/** A solver sentence in Chinese; one this table does not know is shown as written. */
export function solverText(text:string):string{
  const trimmed=text.trim(),known=SENTENCES[trimmed];if(known)return known;
  for(const [pattern,words] of PATTERNS){const match=pattern.exec(trimmed);if(match)return words(...match.slice(1));}
  return text;
}
