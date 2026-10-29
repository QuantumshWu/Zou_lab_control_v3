# 分组曲线交互失效修复

分支：`codex/series-hover-scan`，基于 `d4ca239e`；未合并、未发布。

## 实际复现与根因

使用用户提供的扫描 NPZ，仅只读加载。其保存的绘图是 FacetGrid/Image，
facet 为单个 pair，且 `facet_focus=0`。按原步骤把 Cell kind 改为 Curve，
再把 dipole 轴设为 Group 后，原代码会丢失 hover 和点击锁定。

1. 配置解析把 facet 焦点与 cell 的 X/Y 角色兼容性绑定。Image→Curve 改变了
   Y 轴含义，因此误清了仍然有效的 cell 焦点，进入不接受 series 交互的概览模式。
2. dipole 坐标有 21/22 的不等间距，涉及函数坐标变换。旧命中代码只使用后半段
   仿射矩阵，未先执行轴变换，因此鼠标命中位置与实际曲线可能错位。

v2 底层替换 spec 也会清焦点，但它的 Workbench/Viewer 界面随后会用
PanelState 恢复 `facet_focus`。v2 的 series 命中使用完整的 Matplotlib
`transData`，并处理 fit 隐藏源线后显示的散点。因此不能只看底层清焦点语句，
就认定 v2 的正常界面也有相同失效。

## 修复范围

- 分离 facet 身份与 cell 的 X/Y/group 角色；保持同源、同义 cell 的焦点，
  同时拒绝把旧位置索引带到改变了 facet 枚举的坐标、scope 或数据源中。
- 显式退出聚焦仍有效；不开放新的概览交互模式。
- 命中计算采用与绘制一致的线性、log 和函数坐标变换；柱形图两轴正确逆变换。
- fit 源数据按可见散点命中，不把散点之间虚构成可选线段。
- press、滚轮余量及读数绑定完整的轴、facet 和 series 目标；普通实时刷新保留
  正在进行的交互，切换来源或目标时不沿用旧状态。
- 解锁不会被旧的 locked 读数恢复；换源不隐式继承旧 hover/lock，显式保存的
  交互状态仍可恢复。

## 验证

- 38 项自动回归通过：facet 转换 3、series 命中 7、状态兼容 4，及已有
  interaction 9、navigation 14、restart configuration 1。
- 用户原文件另行通过 `saved_scan_cell_curve_and_dipole_group_remain_interactive`：
  验证焦点、hover、锁定和 RGBA 图像实际变化；原文件未修改，未提交实验数据。
- 原文件 SHA-256：`10f270b6bc2e6dc7a99a8d802c1247b70894cf30885b95551c1e128be05c9a72`。
- 用户叫停 computer use 后，验证仅使用代码和无界面测试。
