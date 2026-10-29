# 远端快照、数值编辑与 Scan Plan 状态

## 远端 snapshot

通用 fabric 现在转发设备自身的完整 snapshot，而不是从控制面板拼一份字段较少的替代记录。
本地和远端共用相同的设备实现，包含 values、units、provenance 及设备专用字段。
设备字段与 apparatus 的身份字段冲突时，共享记录路径保留冲突字段，不静默丢弃；
远端连接来源另放在 `apparatus.device_transports`，不覆盖设备自己的字段。

能力位向后兼容：旧发布端没有 snapshot 能力时明确提示需要更新，不伪造数据。
使用端和发布端都需要更新程序，新的采集记录才能保存远端状态。旧 NPZ 中未采集的历史设置不能补造。

Start 的快照等待有总时限。普通快照超时不触发 watchdog、SAFE 或夺取 owner；
尚未发送的过期请求会取消，已开始的本地管道写入会完成帧并丢弃迟到结果，每设备最多一项在途。
TCP 已写出部分帧后失败属于真实连接损坏，仍需断开以保护协议；零字节写入超时保留连接。
显式关闭 host 时，超出原有关闭预算才结束本 host 所属的残留进程。

SLM 本机 head 和网络 head 当前使用同一个 adapter，都未提供 Start snapshot，仍如实显示 unavailable。
Sequencer 已有专用完整 snapshot，未改变。两个记录的 Pulse 入口及其语义均保留。

## Pulse 文件按钮

对照 v2 `InfoTree._add_action`：使用普通 FluentButton 的高度与内边距，按钮左对齐，
树行只添加上下留白。去掉 v3 对按钮本身高度的特殊压缩/扩张规则。
长名称不截断，通过树的横向滚动读取；过滤按按钮可见文字匹配。

## 数值滚轮

共享 SpinBox 按十进制格点计算商、格点和乘积，最后一次转换为 Number，
避免浮点乘除累积及过宽的格点容差。步进后的显示和提交值一致。
DG4000 和通用 tunable 的标量单位转换共用精确前缀转换，step 与 bounds 同步换算。
真实高精度输入、非线性转换和硬件量化回读不做全局四舍五入。

## Scan Plan 缓存

同一节点编辑上下文按完整 port 身份记忆 Range、Values、mode、unit 和步长。
频率→时长→频率会恢复之前编辑的频率配置，普通重绘、读数与运行状态更新不会清空。
节点、Pulse 文件、设备绑定或原生单位契约不同不会串用缓存。
成功 Revert、Load/New/Clear 等明确替换会清理；取消或失败保留。
缓存仅驻留编辑器，不写入最终 scan plan，也不新增持久化文件。

## 验证范围

- 144 项相关 Rust 回归通过；交互式原生 picker 用例保持 ignored，本次没有运行。
- 142 项 headless DOM 回归、21 项 Python 数值/生命周期回归通过。
- 本地 TCP/mock 设备比较完整 local/remote snapshot，覆盖专用数据、字段冲突、unsupported、非 owner、失败及超时。
- 本地假进程覆盖堵塞 writer、迟到 snapshot、重复请求与显式 shutdown；没有访问实际实验设备。
- Python 数值/生命周期回归，DOM 数值控件、Pulse 按钮布局与 scan cache 回归。
- 没有使用 computer use，没有修改本机 VPN、系统代理或 Git 网络配置。
