# 安装器网络处理与按图打开 Figure

本次修改位于 `codex/installer-network-figure-picker`，基于 `5cfbf76e`。
按要求保留在独立 worktree，不合并、不发布 release。

## 安装器：静态代码确认的问题

- Git fetch 与 PowerShell HTTP 下载使用不同的代理来源。旧代码只给
  .NET HTTP 设置系统代理，没有让 Git 在缺少显式配置时使用同一系统代理。
- 在线检查先查最新 release、再查硬编码仓库的 compare API、再 fetch；
  正式更新又 fetch，原上限为 300 秒。compare API 的分支不一定等于实际 upstream。
- Git 的超时只限制父进程等待；父进程退出后，继承输出管道的子进程仍可能让
  ReadToEndAsync 一直等。安装器自己的检查也存在先读到 EOF 再等待退出的问题。
- 先输出进度 JSON、随后失败，可能被当成成功；中间的 fetching 状态未必得到终态，
  界面可能一直显示检查中。
- 安装当前源码对应的 release 与先 Pull 绑定，GitHub Desktop 手动更新以后，
  安装器仍可能被另一次 Git 检查/拉取失败拦住。

## 安装器：修改后的行为

- 代理策略尊重 NO_PROXY、Git remote/URL 配置（包括显式直连）、代理环境变量；
  Windows 系统代理仅补缺失配置。Git 配置只注入子进程环境，不写用户或仓库配置。
  显式 SOCKS 通过 Windows curl 处理，禁用 curlrc 干扰；日志隐藏凭证。
- 移除多余 compare API，按实际 upstream 判断状态，失败保持“未知”，不假报最新。
  检查 fetch 上限为 30 秒，正式 Pull 的 fetch 上限为 120 秒。
- 进程树及输出管道纳入有界等待；检查失败、超时和不完整 JSON 都会结束检查状态。
  无法建立进程监督时立即停止子进程，不进入无法取消的运行状态。
- 新增 **Install checked-out release**。手动通过 GitHub Desktop 更新代码后，
  可直接安装该 checkout 的 release；在线检查未完成或失败也不阻止这个入口。
  普通 Update 仍保留先更新源码、失败即停止的语义。
- 下载增加读取期限、总时限及失败时的部分文件清理；已通过长度和 SHA-256 校验的
  本地包可以复用。

## Figure Viewer：按图片选择数据

参考 `Confocal-GUIv2/Confocal_GUIv2/gui/gui_individual.py::on_load`：
它在原生文件窗口中同时列出 NPZ/JPG，再通过文件名前缀查找数据。

v3 的 Figure 专用选择器同时显示 NPZ、PNG、JPEG 等图片，启用大缩略图与系统预览面板。
选择图片时，只打开**同目录、完整同名**的 NPZ，不使用模糊前缀或通配符。
找不到配对时明确报错，保留当前 Figure。直接选择 NPZ、输入路径、CLI/remote open
共用同一解析入口；Refresh 使用已经打开的 NPZ 路径。

缩略图及右侧预览由 Windows Shell 提供，支持程度取决于对应图片格式的系统处理器。
本次没有实现或注册 NPZ 的系统缩略图处理器，也没有修改文件关联。

## 验证与边界

- Rust 桌面测试：35 项通过；交互式文件窗口用例在自动运行中保持 ignored，随后单独运行并通过。
- TypeScript 构建及本地 DOM 测试：118 项通过，包括文件选择提示/刷新和安装当前版本按钮。
- `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File tools/tests/network.ps1`
  在生产使用的 Windows PowerShell 5.1 上通过：注入假代理、假 Git/curl、内存响应与本地进程。
- 原生窗口离线验证：临时 PNG/JPG 和 NPZ 同屏显示，图片缩略图及右侧预览可见；
  选择 PNG 后成功解析同名 NPZ。没有启动实验设备或真实安装器。
- 未执行真实 GitHub 拉取/下载测试，未读取或修改本机 VPN/代理设置。
  上述为源码确认的缺陷与离线验证，不能据此断言另一台电脑的 Clash 状态或唯一触发原因。
