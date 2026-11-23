# atom-motion：3D 半经典运动求解器（CPU 版）规格索引

规格就是代码的模块文档：每个模块文件头的 `//!` 与它的 `# ` 小节写着这个模块定义的量、规则、推导和"直到某日如何"的历史。本文件只是索引：
按代码里引用的节号（"SPEC §x.y"）指到现在定义它的模块与小节，另记三样没有模块可归的东西——原则（§0）、不做的事（§7）、
给用户的已知限制（§8）。推导和证据在 `C:\Users\eadri\WorkCode\motion3d_status\rewrite\` 的稿里（contract、PIPELINE_V3_FINAL、
study 等）。

改规则就改模块文档；本文件只在模块或小节改名、增删时跟着改，不复述模块文档的内容（2026-11-22 之前它复述，
有约 25 处指向已删的模块：`model::elim`、`model::reach`、`model::tabulate`、`model::photon`、`table`……）。
"文件 # 小节"指那个文件模块文档里的小节。

## 0 范围与原则
- 取代 `atom-motion-3d`（冻结于 62d4d206，只读参考）。与界面的契约不变（§9）。只做 CPU。
- 硬约束：没有特判，没有拍定的数，一个量只有一种实现（新实现取代旧实现），没有冗余。
- 只允许四条近似（scene.rs 模块文档；校验在 request.rs `Request::validate`）：光镊光与其余光分开处理；其余光用旋转波近似；
  其余光不考虑 32 态以外的能级；其余光只限 D1/D2。其余的取舍只按舍入，或按严格界记账（model/classify.rs # The tiers）。
- 用户的目标：布居与损失 0.05（绝对），温度 0.10（相对，读法见 solve.rs # T's target）；只描述被俘获原子的量（温度、
  它的误差）按持有份额 1 − L 计权，损失按自身——这是用户对目标的定义，不是推导（units.rs、estimate.rs 模块文档）。
- 误差账：每个输出时刻 E² = A² + c² + S² ≤ 目标²——A 模型误差、c 步长误差、S 统计误差（ledger.rs # The total error）。
  误差控制是 QuTiP 式的点估计加收敛检查。
- 允许的常数及其来源（用户目标、格式的阶、置信度 z、物理常数、浮点格式、由它们推导的值、契约的定义与上限）：units.rs。
- 单位：units.rs 模块文档；记号（态、发射算符）：model.rs # Notation。

## 1 物理模型
### 1.1 原子
32 个内态（8 个基态在前）、H0 在模型基里对角、发射算符：request.rs（`STATES`、`GROUND`）、model.rs # Notation。

### 1.2 光场
光束振幅及其解析导数、每个频率一个相干场：field.rs # Beams、# One field per frequency。耦合、外层形式、H(r, t)：scene.rs 模块文档。

### 1.3 势与阱
- 阱光的势 V = Σ_b p_b W_b 与它的展开：scene.rs `trap_potential`；报告的阱（其余光的平滑光移）：scene.rs、model/selected.rs、
  solve.rs `reported`。
- 从焦点下降到阱心：model/potential.rs # The descent。
- 区域（光强 1/20）：field.rs # The region。它不参与损失（§1.7）。

### 1.4 内态生成元
- H 按项：scene.rs 模块文档。
- 唯一的折叠（Reiter–Sørensen）：model/rs.rs # The fold、# The two folds；结构与缩并：model/rs/fold.rs；折叠块的谱：
  model/rs/spectrum.rs。
- 轨迹的模型（簇、帧、拍、节点、光子）：model/selected.rs 模块文档；帧：floquet/frame.rs # The frame。

### 1.5 模型的取舍与误差
- 每一项的四档（零、删、折叠、保留），删项的严格界：model/classify.rs # The tiers、# The drop tier。
- 折叠的误差沿原子路径测：paths.rs 模块文档；候选与选择：select.rs 模块文档；流程：solve.rs # The model、# The selection。

### 1.6 跳跃与光给的动量
step.rs # Jumps、# The absorbed light's momentum、# The optical sector's noise；光子的类与唯一的光子规则：model.rs `Moments`、
model/selected.rs # The photons；光学扇区的动量噪声：model/optical.rs。

### 1.7 读出与输出
- 显示量是期望（布居、损失、温度）：estimate.rs # The outputs are expectations、# The cloud's own motion、# Estimators。
- 损失：能量越出束缚集 H = {E ≤ 0} 的首次通过，不看区域：step.rs `Dynamics::lost`（model/selected.rs 里的实现）、adapt.rs # The reads。

### 1.8 一步的定义域
step.rs # The domain；未分辨的原子的记账：solve.rs # The unresolved atoms。

## 2 数值方法
### 2.1 一步
分裂 A(τ/2) B(τ) A(τ/2)、参考、冲量、路径：step.rs 模块文档；步中点生成元的相互作用绘景：spectral.rs 模块文档；
节点与精确块：model/selected.rs # The nodes、# Exact blocks；线性代数算到舍入：linalg.rs # To round-off。

### 2.2 步长
全程一个常数步长：adapt.rs # The step；它的选取（两层、三层、阶梯）：solve.rs # The step；步长误差的读数：estimate.rs # The step's error。

### 2.3 表
折叠块按光镊光强建的表：model/rs/fold_table.rs 模块文档。

### 2.4 随机数
random.rs。

### 2.5 初始系综
ensemble.rs 模块文档（降低的等温分布、Wigner 因子、拉丁超立方的组 `GROUP`）。

## 3 统计
estimate.rs # Estimators、# Errors、# Many looks、# What no trajectory saw；停止：solve.rs # Stopping；温度的目标与允许量：
solve.rs # T's target、# T's allowance。

### 3.1 分析时刻
analysis.rs # The rule；接入：solve.rs # The analysis times。

## 4 账本
ledger.rs # The total error、# The contract's shape。

## 5 决策
### 5.1 形式
一个读数、三种结果、多个读数、按计数的工作量：decide.rs # A decision、# Many reads、# Work。

### 5.2 模型
model/classify.rs、select.rs、paths.rs；流程：solve.rs # The model、# The selection。

### 5.3 步长
solve.rs # The step。

### 5.4 表
model/rs/fold_table.rs # Where it pays。

### 5.5 轨迹数
§3。

## 6 验证参照
少而物理上不同、有解析或近似解析解的参照（R0 静止极限、R1 保守阱、R2 二能级辐射压与反冲、R3 一维多普勒极限、R4 西西弗斯、
R5 光泵、R6 拉曼 Λ、R7 共振壳穿越、R9 损失与逃逸）：场景与公式在 contract.md §6 与 `refs/`；通过条件的定义与推导在
`refs/check.mjs` 文件头（常数都由 z = 2 得出）。旧真值表只作回归。

## 7 不做（旧实现里有，新实现不要）
- 旧的全局 dt 验证办法：D3、漂移读、轨道探针、冲量与跳跃加热判据、布居界、网格相位读、着陆读、`largest_met`。
- 逐轨迹的接受/拒绝步长控制、按单位时间分摊的容差、逃逸能量密度及其回退、修正能量。
- 梯子上的共享 allowance、BUDGET、采样极值当界、类的统一截断、先验记账的光子相干。
- 两束驻波的特例参考、区域换接权重、起点参考。
- 一阶多普勒的钳位与半切换。
- 多个阱势函数、两套基构造、多处生成元组装、多处力与危险率。
- 标定的单价表、手写运算数、`stake`。
- 运行时 Lindblad 拒绝路径（改为构造保证加测试）。
- 设备两套规则、GPU 相关的一切、旧请求兼容层（无载荷）。
- 只供测试的模式与入口（冻结轴、多普勒开关、验证用的稠密参考）：测试的参照只随测试编译（`#[cfg(test)]`）；没有测试再读的
  就删（2026-11-23 删了 krylov 与 floquet 的装配，连同 faer）。
- 随样本估计变化的 T 目标、经典区系数 C。
- 控制变量与其他方差缩减（实测至多省 4–32 %，study C §3.1）。

## 8 已知限制（照实写给用户）
- **点粒子（Ehrenfest）假设：亚多普勒冷却的温度偏低。这是模型自身的缺陷，不是数值误差，不进误差预算（用户 2026-11-11 定）。**
  - 是什么：一条轨迹是一个点（r, κ）带一个内态 ψ，质心按当前内态上的平均力 ⟨F̂⟩ 运动。精确的半经典理论里内态的各分量各受
    各的力，在动量上彼此分开，再经光泵与飞行反过来影响受力；点粒子带不了这个关联（缺的一项是 ½(|χ⟩⟨ψ| + |ψ⟩⟨χ|)，
    χ = (F̂ − ⟨F̂⟩)ψ）。平均力不受影响——摩擦是精确的（参考 R4D_FORCE）；受影响的是力的慢关联，也就是温度。
  - 何时为零：跳跃后的态是力的本征态时（多普勒冷却、反冲、驻波中的闭合跃迁：R2、R3、R10）。偏振梯度里基态带相干的冷却
    （F = 2 → 3 的 lin⊥lin、σ⁺σ⁻、灰色黏团）都有。
  - 多大（实测，同一模型的全量子参考对真实引擎）：
    - 光镊紧束缚轴上的 lin⊥lin，一维：−2 %（100 µs）、−9 %（500 µs）、−15 %（1 ms）（参考 R4D_TW2）；R4_LIN 场景一维
      −18 % 至 −27 %。
    - 一维自由空间，−6Γ：lin⊥lin 每束 3–12 mW/cm² 时低 11–21 %，48 mW/cm² 时只有真值的 0.44；σ⁺σ⁻ 在 6–12 mW/cm² 时只有
      真值的 1/4 到 1/3，48 mW/cm² 时 0.42；3 mW/cm² 时真值是两群原子（核心 10.8 µK，一成很热），引擎的核心是 2.5 µK。
      σ⁺σ⁻ 的差距里混着半经典理论自身收敛慢的部分，没有分开。
    - 三维六束光：没有可用的量子参考，偏差未知；只能由实验数据判定。
  - 为什么不修：点粒子轨迹带不了这项关联；带上一阶矩的闭合（M、M+N）线性不稳定；既精确又为正的做法是在紧束缚轴上给每条
    轨迹一个波函数（一维代价 ×7.5，三维每量子化一个轴再 ×64–128），超出这个引擎的定位。推导与数值在
    `motion3d_status/rewrite/diffusion.status.md`、`mscheme.status.md`。
  - 三维光镊里两个简并的紧束缚轴经阱的非谐势交换能量是模型内真实的物理（引擎与经典二维运动一致），拿三维结果去比一维参考
    不是有效的检验。
  - 经典区 k_BT_a ≥ 4ħω_a 只作报告，不参与任何决定（units.rs `CLASSICAL`）；同一模型的精确半经典解在 4ħω 处与全量子只差
    1.4 %，阱的量子化不是这里的主要限制。
- 傍轴近似，不含纵向场；所有光束共焦于原点（field.rs # Beams）。
- 同频光束相干，相位取请求的值（field.rs # One field per frequency）。
- 原子冷到被局域在近共振光的晶格格点里时，经典区判据只反映光镊尺度的量子化，不反映格点的。

## 9 接口契约
契约在 contract.md。引擎这一侧：
- 请求的映射与校验、拒绝的消息（中文，界面原样显示）：request.rs（`Request::from_native`、`Request::validate`）。
- 流水线与结果、检查点、消息：solve.rs # The pipeline、# Checkpoints、# Exits；结果的账本形状：ledger.rs # The contract's shape；
  与协议的桥：`../src/motion_adapter.rs`。
- 公开面：`request::Request`、`solve::solve`、`solve::Solution`、账本类型、`solve::POPULATION_TARGET`。

## 10 模块（每个量的唯一定义处；各自的模块文档是规格）

| 模块 | 内容 |
|---|---|
| `units` | 常数与推导值：用户目标、阶、z、份额、物理常数、契约的上限 |
| `linalg` | 复数、小稠密矩阵、LU、厄米与一般本征、指数，算到舍入；向量内核 |
| `special` | 不完全 Γ 函数（级数与连分式）；高斯–勒让德的节点与权 |
| `random` | Philox 与流 |
| `request` | 契约的请求类型、映射与校验 |
| `field` | 光束振幅及其解析导数、每个频率的相干场、区域（光强判据） |
| `scene` | 请求的哈密顿量的唯一构造处（按项、按频率分量）、阱光的势、报告的阱 |
| `floquet` | 模型按频率分量的形式（`Hilbert`）与帧（`frame`）：折叠的扇区、轨迹模型的帧矢量 |
| `model` | 模型的共用类型（拍、点、阱、簇、光子的类）；`classify` 每一项的四档；`rs` 唯一的折叠（`fold`、`fold_table`、`spectrum`）；`selected` 轨迹的模型；`optical` 光学扇区的动量噪声；`potential` 阱心的下降 |
| `spectral` | 步中点生成元的相互作用绘景：音调、分裂、步内任意时刻的态 |
| `step` | 一步（A、B、参考、冲量）、跳跃、吸收的光的动量、定义域 |
| `adapt` | 常数步长的推进（步、步内读数、跳跃时刻、布朗路径） |
| `ensemble` | 初始系综 |
| `estimate` | 输出、估计量、jackknife、Student t、Wilson、步长误差的读数 |
| `decide` | 读数、三种结果、工作量计数 |
| `ledger` | 账本 |
| `analysis` | 分析时刻 |
| `paths` | 候选模型沿原子路径的误差 |
| `select` | 选 P：候选、路径、模型误差 A、最小总代价 |
| `solve` | 流水线：请求 → 模型与选择 → 先导与步长 → 批次 → 检查点 → 结果 |
| `prof` | 剖析计数（只随 `prof` 特性编译，测量用） |
