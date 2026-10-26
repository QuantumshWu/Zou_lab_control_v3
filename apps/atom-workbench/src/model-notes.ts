// 模型说明: the model, units and scope, one section per topic. Each page's ⓘ
// opens the dialog at its own section; the levels page adds the scope of the
// spectrum it is showing now.
import {math} from './math.js';

export type ModelSection='states'|'field'|'population'|'figure';

export function modelNotes():string{
  return `<section id="model-states"><h3>32 态密度矩阵</h3><p>基态 8 个、D1 激发态 8 个、D2 激发态 16 个。动力学由本机 Rust 求解器计算 Lindblad 主方程，保留密度矩阵的布居与相干项。激光相干驱动和自发辐射均参与演化。用户指定远失谐光或近共振光。远失谐光仅产生背景光移；近共振光对两个基态 F 及 D1、D2 的全部允许电偶极跃迁进入密度矩阵驱动；不同频率的拍频项显式保留。光镊不再作为显式光学驱动。模型限于 32 个内部态，Zeeman 混合在各精细结构流形内计算。显式驱动和辐射采用电偶极光学 RWA 与各向同性自由空间 Markov 真空；自发衰变算符随背景本征基变换；区分不同基态超精细发射频率，同一通道保留相干。远失谐背景 Hamiltonian 包含 D 线反旋转项及列明高电子态的二阶 DC 光移；近共振光只采用光学 RWA，不截断跨 F 驱动，也不重复加微扰 Stark 项。高态未加入传播的 Hilbert 空间，其布居和额外散射通道未被显式求解；还未包含原子运动、技术噪声和驱动引起的辐射 Lamb shift。数值收敛误差不等于这些物理近似的误差。</p>`+
`<p class="model-formula">${math('\\dot\\rho=-i[H/\\hbar,\\rho]+\\sum_j\\mathcal D[L_j]\\rho')}</p>`+
`<h4>850 nm 远失谐光的适用范围</h4><p>此前的独立弱场 Kramers–Heisenberg 对照中，850 nm 反旋转辐射项使总散射率比光学 RWA 结果高约 7.84%。这个数值针对该弱场辐射对照，并不是任意场景的误差上界。当前新增的二阶 DC 光移修正不会补上缺失的辐射通道；更小的数值残差也不会消除该散射模型偏差。</p></section>`+
`<section id="model-field"><h3>磁场与自发衰变</h3><p>超精细和 Zeeman Hamiltonian 对角化给出当前磁场下本征态。驱动和辐射偶极算子在同一个本征基中变换；辐射分支因本征态混合而改变。通向相同发射光子的 D1、D2 路径保留相干叠加。具体数值残差和求解状态在“计算详情”内查看。</p></section>`+
`<section id="model-population"><h3>布居着色</h3><p>“稳态”仅使用通过收敛检查的长期布居极限。多频驱动若通过收敛检查，则显示“长期时间平均”；单频周期驱动时，它对应周期平均。若存在多个长期平均解，则保留给定初态的投影，详情中标明初态依赖。颜色使用固定的 0–100% 布居色标，不随其他态的布居或选中状态变化。未获得稳态时不绘制虚构的无限时结果。基态上方始终显示已验证的长期布居。背景基发生混合时，布居由完整长期密度矩阵的 v†ρv 得到，保留相干项；缺少完整密度矩阵时显示未知。“指定时刻”的着色只在现有瞬态数据足以支持所示基底时使用；布居演化视图保留原始磁场本征态的 p(t)。非零稳态布居本身不等同于严格量子暗态。</p><p>布居演化曲线只连接实际求解的样点，不推断未采样的运动；样点数和间隔写在图下方。</p></section>`+
`<section id="model-figure"><h3>图形约定</h3><p>纵向色带按 |m| 区分磁子能级；态卡片的蓝色深浅是固定的 0–100% 布居色标，灰色卡片是初态。每个精细流形内按能量线性排列，光学频率间隔省略。</p><p>选中激光时，光谱图把其余启用外场作为背景，排除所选光束的自作用；取消选择时显示全部外场的总位移。可定义的远失谐背景采用静态 Stark 光谱模型，并在同一背景基中计算能差、矩元和布居。近共振、强混合或不可静态约化的多频背景没有这里使用的单一静态光移，图中保留裸态几何并将相应数值标为未定义；不会把裸 Zeeman 能量冒充总光移。该光谱图模型不替代完整 Rust 密度矩阵演化，所有启用光束始终参与动力学。每束光可单独指定图示参考态，缺省跟随动力学初态；基态参考画吸收上箭头，激发态参考画受激辐射下箭头。改变参考态或选择对象不改变 ρ₀。参考能级只参与能差计算。</p><p>参数框的 Δ = νL − νref，以所选零场跃迁为参考；图和矩元读数的 ΔL = νL − νeg，νeg 为当前背景下 g↔e 的正频率。频率单位切换保持实际频率不变。偏振主轴 ψ 相对光束球坐标的 eθ 方向，横向另一轴为 eφ；χ = +45° 在 k 与 B 同向时对应 σ⁺。零磁场以 +z 为显示参考轴。σ 分量采用光场约定；受激辐射的原子 Δm 与对应吸收相反。</p></section>`+
`<section id="model-data"><h3>数据</h3><p>原子数据：<a href="https://steck.us/alkalidata/rubidium87numbers.pdf" target="_blank" rel="noopener noreferrer">Steck · Rubidium 87 Data</a>；独立校验：<a href="https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-master.html" target="_blank" rel="noopener noreferrer">QuTiP</a>。</p></section>`;
}
