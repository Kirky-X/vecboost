# Spec — module-registry

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖 trait-kit 模块管理能力域需求。

## Requirements

### R-module-registry-001: Kit typestate 模块注册

`src/module_registry/mod.rs` 定义 6 个模块(`EmbeddingModule`/`AuthModule`/`RateLimitModule`/`CacheModule`/`DbModule`/`LoggerModule`),每个实现 `trait_kit::prelude::ModuleMeta` + `AutoBuilder` trait。`Kit<Unbuilt>` 注册后调用 `.build()` 返回 `Kit<Ready>`,构建失败返回 `TraitKitError`。

**验收标准:**
- 6 个模块全部注册成功,`Kit::new().register::<...>().build()` 返回 `Ok(Kit<Ready>)`
- 缺失依赖时(如 `AuthModule` 依赖 `DbModule` 但未注册)返回 `Err(TraitKitError::MissingDependency)`
- 循环依赖检测:`ModuleA` 依赖 `ModuleB`,`ModuleB` 依赖 `ModuleA` 返回 `Err(TraitKitError::CircularDependency)`

### R-module-registry-002: AppState 重构

`src/lib.rs` 中 `AppState` 从 14 字段手动堆叠改为单字段 `pub kit: Arc<trait_kit::Kit<trait_kit::Ready>>`。移除 6 个 `FromRef` impl,改为提供辅助方法 `pub fn require<M: ModuleMeta>(&self) -> Result<Arc<M::Capability>, _>`。

**v0.2.0 实际实施状态(未完成,推迟到 v0.3.0)**:
- `AppState` 仍保留 14 字段(`src/lib.rs:66-88`),未改为单 `kit` 字段
- 6 个 `FromRef` impl 仍保留(`src/lib.rs:90-145`)
- `main.rs:304-321` 注释明确:"当前保持现有 AppState 结构不变,module_registry 作为未来重构的基础"
- 原因:`Kit` 内部基于 `RefCell`(`!Send + !Sync`),无法满足 Axum `AppState: Send + Sync` 要求;改用 `AsyncKit` 后完整重构影响面过大,推迟到 v0.3.0
- 详见 `design.md` D1 决策 + "实施偏离记录"

**验收标准(目标,v0.3.0 实现):**
- `AppState` struct 仅含 `kit` 字段
- `grep -n "FromRef" src/lib.rs` 输出为空
- 所有 routes handler 通过 `state.require::<EmbeddingModule>()?` 获取服务,而非 `state.service.clone()`

### R-module-registry-003: mod.rs 加固

`src/module_registry/mod.rs` 仅包含:`pub mod` 声明、`pub use` 重导出、`ModuleMeta` trait 别名定义。所有 `impl AutoBuilder for XxxModule` 实现移至 `src/module_registry/impl.rs`。

**验收标准:**
- `grep -E "^\s*impl|^\s*fn " src/module_registry/mod.rs` 输出为空
- `grep -E "^\s*pub (trait|enum|struct)" src/module_registry/mod.rs` 输出 ≥ 1(允许 trait/enum/struct 定义)

## Constraints

- trait-kit 始终启用(非 optional),作为模块管理核心
- `Kit<Ready>` 不可变,模块能力通过 `kit.require::<M>()` 检索
- 构建错误必须在应用启动前暴露(编译期 + 启动期,非运行时 panic)

## Out of Scope

- 不实现异步模块(`AsyncKit` 留待下个 change,本轮用同步 `Kit`)
- 不引入 confers 集成 feature(本轮 `confers-macros` 仅在配置模块用)
- 不实现 ICU4X 国际化(`i18n` feature 不启用)
