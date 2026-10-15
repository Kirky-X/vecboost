# trait-kit 统一封装 SDK 接口评估

> 评估时间：2026-10-14。评估对象：能否用 trait-kit 的 trait / 能力协商统一封装
> 本项目使用的外部 SDK 接口（sdforge / garrison / oxcache / limiteron / dbnexus /
> confers / inklog）。
>
> **结论摘要**：技术可行（feasible），但按「≤3 文件、≤400 行、风险 low、门禁全绿」
> 的发布前实施门槛，全量统一不满足；**verdict = defer_with_proposal**。
> 理由与后续提案见 §3、§4，风险与前置条件见 §5。

---

## 1. 现状分析

### 1.1 协议 SDK 面（sdforge）：已统一，无重复适配层

`src/api/` 全部协议能力经 sdforge `#[forge]` 宏注册（`src/api/mod.rs:1-33`），
每个业务端点只有一份协议无关的处理逻辑，HTTP/CLI/gRPC（及 MCP）各自只保留一个
薄包装：

- 协议无关 handler：`src/api/embedding.rs:310-343`（`embed_handler`）等；
- HTTP：`src/api/embedding.rs:879`（`forge_embed`）、CLI：`:1186`（`cli_embed`）、
  gRPC：`:1233`（`grpc_embed`）——三者都只调 `embed_handler(req)`；
- 同类模式在 `src/api/decisions.rs:1-12`、`src/api/transcribe.rs:1-10` 的模块注释中
  被显式声明为项目约定。

结论：协议 SDK 面**没有**重复适配层，不需要再包一层 trait-kit trait。

### 1.2 infra 六个 SDK 的接入点：每个 SDK 恰好一个适配点

| SDK | 接入点 | 形态 |
| --- | --- | --- |
| garrison | `src/auth/mod.rs:24-44`（`GarrisonHandle`）+ `src/auth/interface.rs`（`VecBoostInterface` 实现 garrison `GarrisonInterface`） | 薄适配 + 若干 re-export（`:17-127`） |
| limiteron | `src/rate_limit/limiteron_adapter.rs:112-147`（`LimiteronAdapter`：Governor + per-key 桶 + `health_cache`） | 厚适配（577 行，真实业务逻辑） |
| oxcache | `src/cache/oxcache_backend.rs`（1070 行，bloom + WAL 持久层）+ `src/cache/semantic_cache.rs`（915 行） | 厚适配；实例由服务持有：`src/service/embedding.rs:49,52`、`src/service/rerank.rs:23` |
| dbnexus | `src/db/mod.rs:16-73`（`DbPool` wrapper，错误类型转换） | 薄适配 |
| confers | 配置加载 + `ConfigWatcherModule`（`src/registry/impl_.rs:744-794`，`WatcherGuard` 能力） | 模块化接入 |
| inklog | `src/logger/impl_.rs:19-39`（`LoggerModule`，能力 = `Arc<inklog::LoggerManager>`） | 薄适配 |

结论：**不存在「同一 SDK 被重复适配」**（没有第二份 limiteron/oxcache/dbnexus 包装）。
重复面不在 SDK 适配器，而在模块样板与装配序列（见 §1.5）。

### 1.3 trait-kit 注册表现状：21 个 Module，全部「预构建注入」+ 空依赖

- `src/registry/impl_.rs` 含 20 个 `ModuleMeta` + `AsyncAutoBuilder` 实现，
  `src/logger/impl_.rs:21-39` 1 个（其中 auth/csrf 2 个受 feature 门控），共 21 个。
- 每个模块的 `build()` 都是同一句能力查表：
  `Box::pin(async move { kit.config::<<Self as AsyncAutoBuilder>::Capability>() })`
  （`src/registry/impl_.rs:64-68`；同类实现遍布 :113-117、:157-161、…、:753-762）。
  模块本身不构造任何资源——对象在 `main.rs` 预构建后经 `kit.set_config()` 注入。
- 依赖声明全部为空：`fn dependencies() -> &[(&str, TypeId)] { &[] }`
  （`src/registry/impl_.rs:55-57` 等 21 处），并有测试把「全空」钉死
  （`src/registry/impl_.rs:918-934`，`test_all_modules_have_no_dependencies`）。
  即 `build()` 的拓扑校验当前是空转：不存在真实依赖边。
- 能力形状不统一：`Arc<RwLock<EmbeddingService>>`、`Arc<LimiteronAdapter>`、
  `bool`（`CacheModule`，`src/registry/impl_.rs:428-442`）、`Vec<String>`、
  `Option<Arc<T>>`、`DetectService` 值类型混合使用。
- 消费侧有两种检索通道并存：`state().kit.require::<M>()`
  （`src/api/embedding.rs:337-341`，全仓 161 处）与 Axum `FromRef`（`src/lib.rs:135-231`，
  22 处 `require`），后者是前者的 axum 适配。

### 1.4 生命周期 / 健康检查 / 版本协商的真实接线度

- lifecycle 已挂 8 个（`src/main.rs:2794-2801`），health 已挂 7 个（`src/main.rs:2802-2808`）。
  其中确有真实行为：`AuditModule` 关闭时 flush（`src/registry/impl_.rs:512-532`）、
  `ConfigWatcherModule` 关闭时带 5s 超时关闭 watcher（`:764-794`）、
  `RateLimitModule` 读缓存健康位（`:377-388`）、`TranscribeModule` 未装配报 Degraded
  （`:310-321`）。其余模块的 lifecycle 只是日志、health 恒 Healthy。
- 能力版本协商处于**空转**状态：
  - 21 个模块的 `VERSION` 全部取 `env!("CARGO_PKG_VERSION")`
    （`src/registry/impl_.rs:53` 等 21 处）——这是 **vecboost 自身版本**，不是模块或
    对接 SDK 的版本；
  - `required_versions()` 全仓 0 处实现（`grep -rn required_versions src/ Cargo.toml` 无命中）；
  - `Cargo.toml:54` 仍启用 `negotiate`（version-negotiation 别名），`Cargo.toml:49-53`
    注释自述 rc.7-rc.9 引入的 probe/report/presets/di/compose/encryption 等面
    「全部零接线不启用」。
  - trait-kit 侧语义：`ModuleMeta::VERSION` + `required_versions` 在 `build()` 做 semver
    校验（`trait-kit/src/core/meta.rs:13-56`；README.md:93、:313）。
    对**单一静态二进制**来说，这些版本在编译期已由 `Cargo.lock` 钉死，
    `build()` 再校验没有增量信息——见 §2.3。
- trait-kit 的高级面在 vecboost 零使用：`override_module` / `register_as` / `resolve` /
  `probe_aggregate` / `run_probes` / `health_aggregate` / `build_report` /
  `contract_manifest` 全部 grep 无命中（`AsyncKit` 甚至没有 `register_as/resolve`，
  di 只在同步 `Kit` 上：`trait-kit/src/kit/kit.rs:587,2171`）。

### 1.5 重复面在别处：装配序列、手写健康检查、全局逃逸

1. **装配序列复制 4 处**（同一「set_config × N + register × N + build」模式）：
   - 生产：`src/main.rs:2695-2815`（`build_module_registry`）+ `:2638-2690`
     （`register_all_modules`）；
   - 测试：`src/lib.rs:294-390`（`make_app_state_with_options`）；
   - 库模式：`src/library/mod.rs:101-148`（`VecBoostLibrary::new`）；
   - 模块化集成：`src/library/mod.rs:377-418`（`VecBoostModuleBuilder::build`）。
   注释自称「kit 是 single source of truth」（`src/main.rs:2727`），但只有生产装配
   是完整的，其余三处各自维护子集，且注册数量注释已漂移（`src/lib.rs:108` 写
   「18 个 Module」，实际注册 21 个）。
2. **健康检查手写重复**：`src/api/embedding.rs:767-864` 对 4 个模块逐一手写
   `health_check → match → i18n 文案`；深度就绪探测 `:528-587`
   （`ENGINE_PROBE_CACHE` + `run_deep_health_checks`）手写了 db/engine/rate_limit
   三个探测，其中 500ms 缓存、超时语义都是自建实现——与 trait-kit `probe` 特性
   （`ServiceProbe` + `run_probes_with_timeout`，`trait-kit/src/core/probe.rs:59`、
   `trait-kit/src/kit/async_kit.rs:1513-1660`）能力重叠。
3. **绕过 kit 的进程级全局**（kit 并非全部资源的事实源）：
   - db 连接池：`static GLOBAL_POOL: OnceLock<Arc<DbPool>>`（`src/db/mod.rs:76-93`），
     `DbModule` 能力只是 `bool`（`src/registry/impl_.rs:473-482`）；
   - Prometheus 写入口：`set_global_collector`（`src/main.rs:2757-2760`，注释明言
     「worker/服务热路径拿不到 kit 状态」）；
   - API 状态单例：`api::init::STATE: OnceLock<VecboostState>`（`src/api/init.rs:15-28`）。

### 1.6 门禁现状（本次实测）

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| i18n 一致性 | `python3 scripts/i18n_consistency_check.py` | **通过**（en/zh 各 535 键齐平，141 文件无硬编码中文） |
| 文档一致性 | `python3 scripts/doc_consistency_check.py` | **失败**（exit 1，8 处 `VECBOOST_BUILD_*/RUN_*` env 声明不一致；`docs/I18N_MISSING_AUDIT.md:605` 记录过 7 处同类遗留，属预存已知项，与 trait-kit 无关） |
| cargo fmt / clippy / test | — | **未运行**（本评估未改动代码，不适用） |

→ 发布前实施门槛之一「全部门禁保持绿」**当前不成立**（doc 门禁红）。
完整门禁清单见 `docs/MAINTENANCE.md:56-63` 与 `.github/workflows/health-check.yml:57-131`。

---

## 2. 设计草案（若统一封装）

### 2.1 分层原则

统一封装的正确切面**不是**「给每个 SDK 造一个 trait」，而是「明确 kit 拥有什么资源、
谁负责关闭、健康/就绪从哪里来」。建议分四层，**只动 L2/L3，L0/L1 保持现状**：

| 层 | 内容 | 建议 |
| --- | --- | --- |
| L0 协议面 | sdforge `#[forge]` HTTP/GPRC/MCP/CLI | 不动（已统一，§1.1） |
| L1 领域服务 | `src/service/*`（EmbeddingService 等具体类型） | 不动（`InferenceEngine` trait 已提供替身缝） |
| L2 能力契约 | 6 个 infra SDK 的资源能力 trait + 模块样板收敛 | 只对「kit 应拥有其生命周期的资源」建 trait：oxcache 后端、limiteron、garrison handle、dbnexus 池、inklog manager、confers watcher |
| L3 装配/编排 | 单一路径构建 kit；健康/就绪统一走 health/probe | 收敛 §1.5 的 4 处装配复制 + 手写探测 |

### 2.2 关键 trait 签名草案（标注：未实施，仅为提案）

```rust
// 提案落点：src/registry/contract.rs（新增文件）——本次未实施
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// kit 拥有生命周期的「资源型能力」最小契约。
/// 只服务检索/健康/关闭三件事；业务方法（check_rate_limit 等）仍留在具体适配器上。
pub trait ManagedResource: Send + Sync + 'static {
    /// 稳定标识，与 ModuleMeta::NAME 对齐（健康聚合、日志用）。
    const NAME: &'static str;
    /// 提供方版本（诊断用途；协商语义见 §2.3——不建议做运行时协商）。
    const VERSION: &'static str;
    /// 同步健康快照：只读缓存态，禁止网络 I/O（对齐 trait-kit
    /// AsyncHealthCheck::check 的「报告路径同步」契约，health.rs:225）。
    fn health(&self) -> trait_kit::HealthStatus;
    /// 关闭钩子：由 AsyncKit::shutdown_async() 逆拓扑序调用。
    fn shutdown<'a>(&'a self) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

// 样板收敛：把 21 份同构 Module 收敛为泛型模块（装配模型不变，仍是预构建注入）。
pub struct ResourceModule<R: ManagedResource>(std::marker::PhantomData<R>);

impl<R: ManagedResource> trait_kit::ModuleMeta for ResourceModule<R> {
    const NAME: &'static str = R::NAME;
    const VERSION: &'static str = R::VERSION;
}

impl<R: ManagedResource> trait_kit::AsyncAutoBuilder for ResourceModule<R> {
    type Capability = Arc<R>;
    type Error = trait_kit::TraitKitError;
    fn build<'a>(
        kit: &'a trait_kit::AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}
```

两点工程事实（读 trait-kit 源码确认）：

- **不需要 `di` feature**：能力类型直接取 `Arc<dyn Trait>` 即可类型擦除检索；
  `di` 的 `register_as/resolve` 只存在于同步 `Kit`（`trait-kit/src/kit/kit.rs:587,2171`），
  `AsyncKit` 无对应 API，本仓用不上。
- **真实 I/O 探测需要 `probe` feature**（未启用）：`ServiceProbe`
  （`trait-kit/src/core/probe.rs:59`）+ `run_probes_with_timeout`
  （`trait-kit/src/kit/async_kit.rs:1534`）对应 §1.5-2 的手写探测；
  启用它会引入 `futures-timer` 可选依赖（trait-kit Cargo.toml `probe = ["health", "dep:futures-timer"]`），
  属于**改依赖声明**。

### 2.3 逐项价值判断（回答「真实价值 vs 无谓间接层」）

| 动机 | 现状证据 | 判断 |
| --- | --- | --- |
| 单一装配入口 | 生产侧已有唯一入口 `main.rs:2692-2815`；但序列复制 4 处、全局逃逸 3 处 | **部分真实**：值得收敛重复，但不需要新 trait，只需抽公共构建函数 |
| 生命周期/健康检查 | 8+7 个已挂载，其中 4 个有真实行为；db/inklog/cache WAL/engine 探测在 kit 之外 | **真实且价值最高**：把缺位资源纳入 kit 的 shutdown/health，才名副其实 |
| 能力版本协商 | `VERSION` = 自身 crate 版本 ×21；`required_versions` 0 实现；单静态二进制、版本编译期钉死 | **纯间接层**：同一 build 内 semver 自校验无增量信息。只有动态加载/跨进程 provider 场景才有意义。建议：要么把 `VERSION` 语义修正为「模块 API 版本」仅供诊断（build_report/graph），要么退掉 `negotiate` |
| 可测试替身 | `InferenceEngine` trait + mock 已存在（`src/library/mod.rs:443-486`）；infra 侧测试直接用真件（`src/registry/tests.rs:144` 真 `LimiteronAdapter`、`src/lib.rs:340-347` 真 inklog） | **弱-中等**：`Arc<dyn Trait>` 能力可让 API 层测试用假件，但当前没有以替身为阻塞的测试需求 |
| 类型安全检索 | `require::<M>()` 已用（161 处） | 已有价值，继续保留 |

**综合判断**：以「统一封装 SDK 接口」为名增设一层 trait，如果落成 §2.2 草案中
**资源契约 + 生命周期归属**，有真实价值（关闭/健康缺口、样板收敛）；
如果落成**能力版本协商层**或为唯一实现造 dyn 接口，则是无谓间接层。

---

## 3. 结论与理由（verdict = defer_with_proposal）

**不做（本发布前不实施）**，理由三条：

1. **规模超界**：任何有意义的统一都 ≥4 个文件——
   最小切片（把 §1.5-2 手写探测/健康收敛进 kit）也要
   `registry/impl_.rs` + `main.rs` + `api/embedding.rs`，若用 `probe` 还加
   `Cargo.toml`；全量统一（契约 trait + 泛型模块 + 装配收敛 + 探测迁移）为
   `Cargo.toml`、`registry/mod.rs`、`registry/impl_.rs`、`main.rs`、
   `api/embedding.rs`、`lib.rs`、`library/mod.rs`、3 个适配器，共 **6-10 文件、
   估 400-900 行**。均超出「≤3 文件、≤400 行」。
2. **触碰依赖声明**：真正用于治理健康/就绪缺口的 `probe`（及可选 `report`）feature
   未启用，`Cargo.toml:49-54` 注释明确「零接线不启用」；启用即改依赖图
   （`futures-timer` 等），与本轮「自研依赖刚升级完毕，勿再改依赖」冲突。
3. **风险与门禁前提不满足**：健康/关闭语义重排不是 low risk（见 §5）；
   且 `doc_consistency_check.py` 当前红（§1.6），「全部门禁保持绿」不成立。

**技术可行性**（feasible = true）：trait-kit 的 `ModuleMeta` / `AsyncAutoBuilder` /
`AsyncHealthCheck` / `AsyncLifecycle` / `ServiceProbe` 原语齐备，且 `Arc<dyn Trait>`
可直接作为能力类型，无需 `di`（§2.2）；改造不动 sdforge 协议面与
`InferenceEngine` 服务面，风险可控——适合作为后续迭代而非发布前改动。

---

## 4. 后续提案（defer 之后按序执行）

| 提案 | 内容 | 文件/行数估计 | 风险 | 前置条件 |
| --- | --- | --- | --- | --- |
| **P1 元数据诚实化** | 21 处 `VERSION` 改为模块真实 API 版本或对接 SDK 版本（不再是 vecboost 自身版本）；补 `dependencies()` 真实边（至少声明真实装配顺序：如服务类模块 → CacheModule/ConfigModule）；删除 `negotiate` 空转或注明仅诊断用途 | 1-2 文件，~100 行 | low | 无（不改依赖） |
| **P2 健康/就绪统一** | 把 `run_deep_health_checks`（db/engine/rate_limit）迁到 `ServiceProbe` + `run_probes_with_timeout`；`health_handler` 的 4 段重复 match 收敛为循环列表；db 池/inklog/cache WAL 纳入 kit 生命周期 | 4 文件（含 Cargo.toml），~200-350 行 | medium | ① 允许启用 `probe` feature 的窗口；② `/health?depth=full` wire 语义（503+JSON）保持不变，有 e2e 覆盖；③ `AsyncHealthCheck::check` 保持同步 → 探测走后台缓存态，不得把网络 I/O 塞进报告路径 |
| **P3 装配入口收敛** | 抽出 `build_runtime_kit(config, prebuilt…)`（或 `VecBoostKitBuilder`），供 `main.rs` / `library` / 测试共用，消除 4 处复制序列与注释漂移 | 2-4 文件，~150 行 | medium（测试面广） | 先补齐 kit 装配的单测覆盖（现 `src/registry/tests.rs` 887 行主要测样板） |
| **P4 资源契约 + 泛型模块（可选）** | 实施 §2.2 `ManagedResource` + `ResourceModule`，把厚资源（oxcache/limiteron/dbnexus/inklog/watcher）收敛为 `Arc<R>` 能力 | 3-6 文件，~300 行 | medium | P2 落地后；保持具体适配器业务方法不迁走 |

执行纪律（任一提案）：每步单独提交；跑 `docs/MAINTENANCE.md:56-63` 全部门禁
（fmt / clippy `-D warnings -W clippy::unwrap_used` / lib+集成测试 / 两个 python 检查）。

---

## 5. 风险与前置条件

**风险**

1. `/health` wire 契约：`health_handler` 是协议面端点（`forge_health`，
   `src/api/embedding.rs:947`），收敛探测会触及 503 状态码、`retry_after`、
   JSON 结构与 i18n 文案（`:855-864`）；有回归测试但需逐一核对。
2. 关闭顺序：trait-kit 的 `on_shutdown` 按逆拓扑序执行，而当前关键关闭依赖
   `main.rs` 手工顺序（如 `ConfigWatcherModule`）；迁移前需确认排序语义等价，
   否则 watcher 关闭 / audit flush 可能互相踩。
3. 替身层 dyn 化：`Arc<dyn Trait>` 会隐藏具体类型方法（`LimiteronAdapter::check_rate_limit_detailed`
   在中间件热路径直接使用，`src/api/embedding.rs:577` 也直接调 `check_health`），
   误用会迫使业务方法也走 dyn 转发，引入装箱开销与接口膨胀。
4. 版本协商的语义陷阱：`ModuleMeta::VERSION` 默认 `"0.0.0"` 表示「未声明」，
   未声明 provider 被要求最低版本时 `build()` 直接失败（`trait-kit/src/core/meta.rs:46-50`）；
   若下游 kit 消费者（`VecBoostModuleBuilder` 路径）误加 `required_versions`，
   会在别人项目里炸启动。不建议启用真协商。
5. 门禁：`doc_consistency_check.py` 现红（§1.6），任何「发布前实施」主张都需先修复或
   显式豁免该 8 处遗留。

**前置条件**

- 需要 owner 批准一个「允许改 `Cargo.toml` feature 集合（probe/report）」的窗口；
  若不批准，则 P2 降级为「用现有 health feature + 后台状态缓存」自建方案，价值打折。
- 修复 `doc_consistency_check.py` 的 8 处 env 声明不一致（或将其定义为非阻断门禁）。
- `/health` 与 `/health?depth=full` 的 e2e 用例先行（本仓已有 e2e 骨架：
  `tests/integration/api_test.rs`），确认探测迁移不改变 wire。
- 每次改动跑全部门禁并贴证据（规则 10）。

---

## 附：AI 使用声明

本文由 AI agent（自动化 workflow 子任务）基于对本仓库与 `trait-kit` 兄弟仓的只读
代码阅读生成；未执行 `cargo fmt/clippy/test`。仅执行了两个 python 一致性脚本
（结果见 §1.6），未做任何代码修改。证据均为 `file:line` 引用，可在对应文件核对。

---

## 6. 发布决策

> 决策时间：2026-10-14。决策对象：本次 0.3.0-rc.1 发布是否实施 §2 统一封装提案。

**结论：本次 0.3.0-rc.1 发布不实施该统一封装。**

**理由：技术可行（feasible = true，§3），但发布前不实施：**

1. **规模超界**：有意义的最小切片也 ≥4 文件（`registry/impl_.rs` + `main.rs` +
   `api/embedding.rs`，若用 `probe` 再加 `Cargo.toml`，§3-1）；全量统一 6-10 文件、
   估 400-900 行，超出「≤3 文件、≤400 行」的发布前实施门槛。
2. **触碰依赖声明**：治理健康/就绪缺口所需的 `probe`（及可选 `report`）feature
   未启用（`Cargo.toml:49-54` 注释自述「零接线不启用」）；启用即改依赖图
   （引入 `futures-timer` 等，§2.2），与本轮「自研依赖刚升级完毕，勿再改依赖」冲突。
3. **风险非 low 且门禁前提不成立**：健康/关闭语义重排会触及 `/health` wire 契约
   （503 / `retry_after` / JSON 结构 / i18n 文案，§5-1）与逆拓扑序关闭顺序（§5-2）；
   且实测 `python3 scripts/doc_consistency_check.py` 当前失败（exit 1，8 处预存
   `VECBOOST_BUILD_*/RUN_*` 不一致，`docs/I18N_MISSING_AUDIT.md:605` 记录过同类遗留），
   「全部门禁保持绿」不满足。

**真实价值判断**（逐项，证据见 §1.4、§1.5、§2.3）：

- **单一装配入口——部分真实**：生产侧已有唯一入口，但序列复制 4 处、3 处全局逃逸。
- **生命周期/健康检查——真实且价值最高**：已有 8+7 个挂载，`AuditModule` flush /
  `ConfigWatcherModule` shutdown 真实工作；缺口在 db 池 / inklog / cache WAL /
  engine 探测。
- **能力版本协商——纯间接层**：同一静态二进制内 semver 自校验无增量。
- **可测试替身——弱-中等**：`InferenceEngine` 已有 mock，infra 无替身且当前不
  以替身为阻塞。

**何时再评估的前置条件**：即评估给出的前置条件（§5）——① owner 批准允许启用
`probe`/`report` feature 的窗口；② 修复（或显式豁免）`doc_consistency_check.py`
的 8 处 env 声明不一致；③ `/health` 与 `/health?depth=full` 的 e2e 用例先行；
④ 每次改动跑全部门禁并贴证据。前置条件满足后，按 §4 P1→P2→P3→(P4) 顺序重估。

> 2026-10-15 更新：前置条件②已解决——8 处 `VECBOOST_BUILD_*/RUN_*` 系
> `scripts/{build,run}-memcap.sh` 真实消费、检查器只扫 Rust 面的误报，env 消费面
> 已纳入 `scripts/`（另过滤族前缀写法截断），门禁转绿；其余前置条件不变。
