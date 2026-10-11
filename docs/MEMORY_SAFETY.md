# 内存安全防线（MEMORY_SAFETY）

> 本文记录 2026-10-11 WSL2 OOM 事故的根因与四道防线的使用方法（变更
> `wsl2-oom-hardening`）。适用于 Linux/WSL2 部署；非 Linux 平台仅防线二/三
> 适用（内存门禁为 Linux-only）。

## 事故摘要

2026-10-11 08:21-08:23，WSL2 宿主（66G 内存 + 17G swap）内存耗尽：内核 OOM
杀掉占 45G 匿名内存的 vecboost 服务，37 秒后 VM 进入关机流程。

根因链（三层叠加，缺一不崩）：

1. **翻译引擎权重倍增（主因）**：全部翻译 pair 在启动期为每 pair 预加载
   `TRANSLATION_PARALLEL_WORKERS`(默认 4) 套并行 worker Session 集。ONNX
   Runtime 对标准 .onnx 的每个 Session 独占一份权重堆拷贝（跨 Session 不
   共享内存页），4 pair × 4 worker 全 FP32 → 服务 idle 零请求仍常驻 30G+。
2. **加载无内存门禁**：Session 构建不检查 MemAvailable，内存不足时静默
   加载直至系统级 OOM，无 fail-fast。
3. **系统侧无早期 OOM 防护**：WSL2 默认无 earlyoom/systemd-oomd；崩溃时刻
   并发的 release LTO 构建（单 rustc 峰值数十 GB）+ 多个 agent CLI 进程把
   RAM+swap 打穿，VM 冻结而非快速失败。

## 四道防线

| # | 防线 | 落点 | 生效条件 |
|---|------|------|----------|
| 1 | Session 内存门禁 | `session.rs`（构建前检查 MemAvailable） | 默认开启 |
| 2 | 并行 worker 惰性加载 | `onnx_translate.rs`（首次长输入并行请求才补建附加集） | 默认开启 |
| 3 | 构建内存上限 | `scripts/build-memcap.sh`（systemd --user scope） | 手动使用 |
| 4 | 用户级 earlyoom | `scripts/mem-guard.sh` + `mem-guard-install.sh` | 安装后常驻 |
| 5 | 进程内存上限（通用） | `scripts/run-memcap.sh`（服务自爆保整机） | 手动使用 |
| 6 | jemalloc decay 烘焙 | `.cargo/config.toml [env]`（大请求后 RSS 秒级回落） | 默认生效 |

### 防线 1：Session 内存门禁

`build_session` 在创建任何 ORT 分配前检查 MemAvailable；需求按模型文件
大小 ×3/2 估算。拒绝时 fail-fast（`ModelLoadError`，消息含 available/need/
floor 三个数字与处置建议），杜绝静默加载到系统 OOM。所有引擎（embedding/
rerank/decision/translate/transcribe）统一生效。

```bash
SESSION_MEMORY_GUARD=off          # 关停门禁（默认开）
SESSION_MEMORY_FLOOR_MB=4096      # 可用量安全地板，默认 2048
```

### 防线 2：并行 worker 惰性加载

翻译并行 worker Session 集不再启动期预载：idle 驻留从
`pair 数 × worker 数` 套降为每 pair 一套主集；首次长输入并行请求（≥600 字
符且 ≥2 句）按需补建，P7 时延契约不变（长输入时延 ÷ worker 数）。首次并行
请求会多付一次性构建耗时（每套秒级）。

```bash
TRANSLATION_PARALLEL_WORKERS=2    # worker 数，默认 4，钳 1..=8；1=关闭并行
```

### 防线 3：构建内存上限（免 root）

把 cargo 构建关进用户级 cgroup：超限时 rustc/ld.mold 被 cgroup 杀掉 =
构建显性失败，而非与常驻服务叠加打穿 RAM+swap。

```bash
scripts/build-memcap.sh build --release   # 代替 cargo build
VECBOOST_BUILD_MEMMAX=26G VECBOOST_BUILD_MEMHIGH=20G VECBOOST_BUILD_SWAPMAX=4G \
    VECBOOST_BUILD_JOBS=8 \
    scripts/build-memcap.sh build --release --features transcribe
```

已验证：scope 内 `memory.max` 内核级真实生效（26G）；cgroup v2 语义下总预
算 = MemoryMax + MemorySwapMax，超总预算即 cgroup OOM 击杀。仅设 MemoryMax
而 swap 无限制时，超限页换出即可绕过上限（实测 16M 上限下 133M 分配可存
活），故必须同设 SwapMax（实测 5G 分配在 16M+4G 预算下被击杀）。

### 防线 4a：进程内存上限（免 root，r2）

以用户级 cgroup 包裹**任意命令**（主要用途：常驻推理服务）——服务异常膨胀时先被自身上限击杀，而不是等整机逼近极限：

```bash
scripts/run-memcap.sh target/release/vecboost config-e2e-transcribe.toml
VECBOOST_RUN_MEMMAX=40G VECBOOST_RUN_MEMHIGH=36G VECBOOST_RUN_SWAPMAX=4G \
    scripts/run-memcap.sh <command> [args...]
```

已验证：scope 内 `memory.max=40G`、`memory.swap.max=4G` 内核级生效；总预算
= MemoryMax + MemorySwapMax，超总预算即 cgroup OOM 击杀。

### 防线 4b：jemalloc decay 烘焙（r2）

jemalloc 默认 dirty/muzzy decay 10s，大请求（长翻译/批量）释放的解码缓冲
滞留 RSS。经 `.cargo/config.toml [env] JEMALLOC_SYS_WITH_MALLOC_CONF` 在
jemalloc-sys 编译期烘焙为 1000ms（对全部 arena 含未来生效），已实测
`opt.dirty_decay_ms=1000`。三条注入路径的排除证据链（运行时 mallctl 不存在、
`je_malloc_conf` 被 hidden visibility 否决）见 design D3。运行期可用
`MALLOC_CONF` 环境变量覆盖（优先级更高，运维逃生口）：

```bash
MALLOC_CONF="dirty_decay_ms:0,muzzy_decay_ms:0" target/release/vecboost config.toml  # 立即归还
```

### 防线 4：用户级 earlyoom（免 root）

MemAvailable 低于阈值时，在目标列表内按 RSS 杀最大者（SIGTERM → 10s 后仍
存活升级 SIGKILL），默认只针对 `vecboost`，绝不波及列表外进程；在 OOM
killer 介入前抢跑，避免 VM 冻结级连锁。

```bash
scripts/mem-guard-install.sh      # 安装并以 systemd --user 常驻（幂等）
systemctl --user status vecboost-mem-guard
tail ~/.local/state/vecboost-mem-guard.log

# 旋钮（写在 service 的 Environment= 或导出后重启服务）
MEM_GUARD_MIN_MB=3416             # 阈值，默认 max(2048, 5%×MemTotal)
MEM_GUARD_TARGETS=vecboost        # 逗号分隔 comm 名
MEM_GUARD_INTERVAL_S=5            # 轮询间隔
```

实测证据：正向（阈值拉高 + 目标 sleep → 假进程被 TERM 并留痕）、反向
（目标不匹配 → 仅告警零误杀）。

## 旋钮总表

| 旋钮 | 默认 | 含义 |
|------|------|------|
| `SESSION_MEMORY_GUARD` | on | Session 加载内存门禁开关（`off` 关停） |
| `SESSION_MEMORY_FLOOR_MB` | 2048 | 门禁安全地板（MB） |
| `TRANSLATION_PARALLEL_WORKERS` | 4 | 翻译并行 worker 数（1..=8，1=关闭） |
| `VECBOOST_BUILD_MEMMAX` / `_MEMHIGH` / `_SWAPMAX` | 26G / 20G / 4G | 构建硬/软/swap 上限 |
| `VECBOOST_RUN_MEMMAX` / `_MEMHIGH` / `_SWAPMAX` | 40G / 36G / 4G | 通用进程硬/软/swap 上限 |
| `JEMALLOC_SYS_WITH_MALLOC_CONF`（cargo [env]） | dirty/muzzy 1000ms | jemalloc decay 编译期烘焙 |
| `MALLOC_CONF`（运行期 env） | 未设 | jemalloc 配置运行期覆盖（优先级最高） |
| `VECBOOST_BUILD_JOBS` | 8 | 构建并行度 |
| `MEM_GUARD_MIN_MB` / `_TARGETS` / `_INTERVAL_S` / `_LOG` | 见上 | 守护参数 |

## WSL2 加固附录（需用户侧操作，未自动执行）

- **root earlyoom（可选，推荐）**：`sudo apt install earlyoom && sudo systemctl enable --now earlyoom`——用户级守护只能杀自己的进程，root 版可兜底任意失控进程。
- **`.wslconfig`（Windows 侧，r2 已落地待生效）**：已追加
  `kernelCommandLine = vm.min_free_kbytes=262144`（原 44MB → 256MB，内核更早
  启动回收/OOM 决策，减少「打穿后才反应」的冻结窗口；原文件备份在
  `temp/wslconfig.backup-20261011`）。**下次 `wsl --shutdown` 重启后生效，
  需用户执行**。
- **纪律**：重构建（尤其 `--release`，本仓 `lto="fat"`）与满载服务并发时，
  用防线 3 封装构建或错峰执行。

## 验证清单

```bash
cargo test --features onnx --lib            # 引擎测试（含门禁/惰性/allocator 纯函数与烘焙验证）
systemctl --user is-active vecboost-mem-guard
scripts/build-memcap.sh --version           # scope 内 cargo 可用
scripts/run-memcap.sh /bin/true             # 通用进程封装可用
cargo test --features onnx --lib allocator_tune  # decay 烘焙生效值 = (1000,1000)
```
