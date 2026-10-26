// The solver's words in the workbench's vocabulary.  The native result keeps
// its own (English) status names and method sentences; only what is shown is
// translated here, so the result contract and its validation are untouched.
import type {SteadyStatus} from './solver-types.js';

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
  'Converged long-time phase average of the full driven Lindblad equation (one-dimensional null space); populations are a time average, not a time-independent density.':
    '完整受驱 Lindblad 方程的长期相位平均已收敛（一维零空间）；布居是时间平均，不是不随时间变化的密度矩阵。',
  'Converged long-time phase average of the specified initial density; several conserved quantities make it initial-state dependent.':
    '给定初始密度矩阵的长期相位平均已收敛；存在多个守恒量，结果依赖初态。',
  'Populations converge in the background basis; exported density is the laboratory-frame time average.':
    '布居在背景基中收敛；导出的密度矩阵是实验室系的时间平均。',
  'Unique full-density Lindblad stationary solution within the resolved rates.':
    '在已分辨的速率内，完整密度矩阵的 Lindblad 稳态唯一。',
  'Full Lindblad stationary projection from the specified initial density.':
    '由给定初始密度矩阵得到的完整 Lindblad 稳态投影。',
  'Cancelled.':'计算已取消。',
};
/** A solver sentence in Chinese; one this table does not know is shown as written. */
export function solverText(text:string):string{return SENTENCES[text.trim()]??text;}
