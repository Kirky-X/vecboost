# ../base 生态吸收能力分析（2026-09-16）

> 目标：盘点 `../base` 七库（trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge）
> 相对 vecboost 已接线子集的能力差距，评估可吸收项并排序。
> 方法：kueiku RICE 优先级框架 × 本仓库「接线三问」过滤（装配处构造注入 / config 段真实解析 / 指标热路径观测）。
> 数据基线：各库 Cargo.toml `[features]` 全表 + README/CHANGELOG（能力增量集中在 rc.3，rc.4/rc.5 以发布卫生为主）。
>
> **实施状态（2026-09-16）**：P0 四项已全部落地——
> ① RateLimit-\* 响应头（`[rate_limit] headers_enabled`，adapter `check_rate_limit_detailed`
> + 中间件注入，回补 limiteron `tower-middleware` feature）；
> ② 文件日志采样（`[logging.sampling]`，main.rs 手工构建 FileSink 并按需包 SamplingSink）；
> ③ 请求上下文（启用 sdforge `context`，替换手写 x-request-id 闭包；debug 级请求日志
> 携带 request_id/trace_id/耗时；`log::` 全量自动注入 trace_id 受 inklog LogAdapter
> 上游限制——其硬编码 `trace_id: None`，待上游在 LogAdapter 消费环境上下文后打通）；
> ④ 能力版本协商（trait-kit `negotiate` feature；AsyncKit 侧协商为本仓库新增补齐：
> register 登记声明版本/最低要求，build() 拓扑校验前 semver 校验；vecboost 17 个
> 生产模块已在 ModuleMeta 声明 `VERSION = CARGO_PKG_VERSION`）。
> 随附修复：limiteron 适配器 `global_requests_per_minute = 0` 时 Governor 空规则集
> 构建 panic（改用高容量哨兵规则）；sdforge `config/app.rs` 因外部并行编辑删除
> `ValidateConfig` 导入而编译失败（调用点改 UFCS 全限定，不依赖 trait 导入）。

## 结论速览

| Tier | 候选 | 来源库 | RICE | 一句话理由 |
|---|---|---|---|---|
| **P0 立即吸收（✅ 已落地）** | 限流预检 `peek()` + IETF `RateLimit-*` 响应头 | limiteron（默认代码） | 20 | 每个 HTTP 请求可见，纯增量无新依赖，`inject_rate_limit_headers` 无 feature 门控 |
| **P0 立即吸收（✅ 已落地）** | 日志采样器 `SamplingSink` | inklog（默认代码） | 11 | 零 feature 成本；嵌入 API 日志量大，采样直接降盘压 |
| **P0 立即吸收（✅ 已落地）** | 请求上下文 `context` + inklog `trace_id` 关联 | sdforge+inklog | 10 | request_id/trace_id 跨协议注入 → 响应头回显 + 请求日志关联 |
| **P0 立即吸收（✅ 已落地）** | 能力版本协商 `negotiate` | trait-kit | 10 | 直击本仓库反复发生的 ../base 漂移痛点（build() 时 semver 校验 fail-fast） |
| **P1 按运维路线吸收** | 构建/健康报告导出 `report` | trait-kit | 5 | BuildReport JSON + 模块图导出，运维诊断利器，成本半天 |
| **P1 按运维路线吸收** | 金丝雀渐进热重载 `progressive-reload` | confers | 5 | 叠加已接线的 watch：新配置金丝雀验证 + 健康检查不过自动回滚 |
| **P1 按运维路线吸收** | 集中日志转发 `net-sink` | inklog | 5 | TCP(TLS)/UDP 断线缓冲重连，部署上集中日志时的标准件 |
| **P1 按运维路线吸收** | 配置快照 `snapshot` + 审计链 `audit` | confers | 4 | 变更前快照可回退 + HMAC 链留痕，合规与事故复盘双收益 |
| **P1 按运维路线吸收** | OTLP 导出 `otlp` | inklog（或 sdforge/dbnexus 同名） | 3 | 零新依赖接 APM；等有真实 APM 需求时启用 |
| **P2 Backlog** | `adaptive-limiting` / `bulkhead` | limiteron | 2 | AIMD 自适应限流、舱壁隔离——推理负载治理好料，接线面大需专项 |
| **P2 Backlog** | 声明式校验 `validate` | sdforge | 2 | `#[forge(validate)]` 删手写样板；建议随下次 API 面改动顺带做 |
| **P2 Backlog** | `prepare-cache` | dbnexus | 2 | 语句级 LRU，对嵌入式 sqlite 收益有限，postgres 接线后再评估 |
| **P2 Backlog** | 跨实例失效 `invalidation` / `ban-sync` / `red-lock` | oxcache/limiteron | 1 | **前置条件缺失**：vecboost 无 Redis 接线（Cargo.toml Redis 段为空），且须与 confers change-stream、trait-kit EventBus 整链评估，不可单点吸收 |
| **P2 Backlog** | `etag` / `paginate` / `interpolation` / `security-rules` | sdforge/confers | 1-2 | 嵌入 API 场景弱（分页/ETag）或收益边际（配置 DRY、配置体检） |

## 不吸收清单（重叠能力，防止双机器漂移）

| 能力 | 来源 | 不吸收理由 |
|---|---|---|
| `/healthz` `/readyz` 自动探针 | sdforge `health` | vecboost 已有自研 `/health?depth=full`（含引擎 dummy 推理探测 + 500ms 缓存），更强 |
| `/metrics` Prometheus 文本 | sdforge `metrics` | 已用 prometheus + axum-prometheus，指标体系已收敛 |
| SIGTERM 三阶段优雅停机 | sdforge `graceful` | 已有 AsyncShutdownCoordinator 分级停机 + gRPC drain 窗口（run_server_lifecycle） |
| JWT/CSRF/审计等安全能力 | confers/dbnexus `authentication` 等 | garrison 已全量接线并深度定制，双安全栈必然漂移 |
| inklog `http`/`cli` | inklog | `cli` 有价值但非服务能力（离线查档工具），按需单独引入；`http` 与自研 /metrics、/health 重叠 |

## P0 候选接线要点（三问预检）

1. **RateLimit-\* 响应头**（limiteron）
   - ① 装配：限流中间件在 `Governor::check()` 得到 `Decision` 后经 `middleware::headers::inject_rate_limit_headers` 注入响应（`peek` 为 RateLimiter trait 默认方法，无需新 feature）。
   - ② config：`[rate_limit]` 增加 `headers_enabled`（默认 false 保持兼容）。
   - ③ 指标：Decision 快照已含 remaining/limit，无需新增热路径观测。
2. **日志采样器**（inklog）
   - ① 装配：LoggerManager 构建 file sink 后包一层 `SamplingSink::new(inner, Sampler::new(...))`。
   - ② config：`[logging] sampling` 段（rate + per-level 覆盖），接入 AppConfig。
   - ③ 指标：采样丢弃计数可选接入 InferenceCollector。
3. **context + trace_id**（sdforge+inklog）
   - ① 装配：启用 `sdforge/context`（request_id/trace_id 跨协议注入）；inklog 侧 trace_id/span_id 已内建，只需中间件把上下文写入 task-local。
   - ② config：`[server] request_context_enabled`。
   - ③ 指标：可与 x-request-id 响应头（已存在）打通。
4. **negotiate**（trait-kit）
   - ① 装配：AsyncKit build 前对各 Module 注册能力版本断言；版本源用各库 CARGO_PKG_VERSION。
   - ② config：无新段（编译期/启动期行为）。
   - ③ 指标：协商失败 → 启动失败（fail-fast），正是目标行为。

## 多实例一致性链路（整链评估提示）

confers `change-stream` ↔ trait-kit `EventBus` ↔ oxcache `invalidation`(+redis) ↔ limiteron `ban-sync` 构成同一条「配置/缓存/封禁变更传播」链。vecboost 当前单实例且无 Redis，整链价值为负；一旦引入水平扩展，应作为**单条 epic** 整链设计，不要按单 feature 碎做。
