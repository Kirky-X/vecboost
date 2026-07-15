// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Module Registry — trait-kit AsyncKit 集成模块
//!
//! 通过 trait-kit 0.3 的 `AsyncKit`（`Send + Sync`，基于 `Arc<RwLock>`）管理
//! 模块依赖关系和能力构建。`AsyncKit` 可以放入 `Arc<AsyncKit<Ready>>` 共享
//! 跨线程，也可存入 `VecboostState`。
//!
//! 模块采用"预构建能力注入"模式：需要异步构造的复杂对象（如 EmbeddingService）
//! 在 main.rs 中预构建后通过 `kit.set_config()` 注入，模块的 `build()` 从 config 检索。
//! `AsyncKit::build().await` 按拓扑序异步调用各模块的 `AsyncAutoBuilder::build`。

mod impl_;
#[cfg(test)]
mod tests;

/// 嵌入服务模块 — 提供 `Arc<RwLock<EmbeddingService>>` 能力
pub struct EmbeddingModule;

/// 认证模块 — 提供 `Option<Arc<JwtManager>>` 能力（需要 auth feature）
#[cfg(feature = "auth")]
pub struct AuthModule;

/// 限流模块 — 提供 `Arc<LimiteronAdapter>` 能力
pub struct RateLimitModule;

/// 缓存模块 — 提供缓存启用状态能力
pub struct CacheModule;

/// 数据库模块 — 提供数据库启用状态能力（db 集成见 src/db/mod.rs）
pub struct DbModule;

/// 审计模块 — 提供 `Option<Arc<AuditLogger>>` 能力
pub struct AuditModule;

// ---------------------------------------------------------------------------
// v0.3.0 D3 重构：覆盖 VecboostState 剩余字段的 13 个 Module
//
// 设计原则（与现有 EmbeddingModule/AuthModule 等保持一致）：
//   - 模块 Capability 类型 = 字段类型（1:1 映射，无派生逻辑）
//   - 复杂对象通过 `kit.set_config(capability)` 预构建后注入，build() 从 config 检索
//   - bool 字段用 newtype 包装以避免 TypeMap 中 bool TypeId 冲突
// ---------------------------------------------------------------------------

/// 用户存储模块（auth）— 提供 `Option<Arc<UserStore>>` 能力
#[cfg(feature = "auth")]
pub struct UserStoreModule;

/// 认证启用模块 — 提供 `bool` 能力（读取 `AuthEnabled` newtype 配置）
pub struct AuthEnabledModule;

/// CSRF 配置模块（auth）— 提供 `Option<Arc<CsrfConfig>>` 能力
#[cfg(feature = "auth")]
pub struct CsrfConfigModule;

/// CSRF token 存储模块（auth）— 提供 `Option<Arc<CsrfTokenStore>>` 能力
#[cfg(feature = "auth")]
pub struct CsrfTokenStoreModule;

/// 指标收集器模块 — 提供 `Option<Arc<InferenceCollector>>` 能力
pub struct MetricsCollectorModule;

/// Prometheus 收集器模块 — 提供 `Option<Arc<PrometheusCollector>>` 能力
pub struct PrometheusCollectorModule;

/// IP 白名单模块 — 提供 `Vec<String>` 能力
pub struct IpWhitelistModule;

/// 限流启用模块 — 提供 `bool` 能力（读取 `RateLimitEnabled` newtype 配置）
pub struct RateLimitEnabledModule;

/// 管道启用模块 — 提供 `bool` 能力（读取 `PipelineEnabled` newtype 配置）
pub struct PipelineEnabledModule;

/// 管道队列模块 — 提供 `Arc<PriorityRequestQueue>` 能力
pub struct PipelineQueueModule;

/// 响应通道模块 — 提供 `Arc<ResponseChannel>` 能力
pub struct ResponseChannelModule;

/// 优先级计算器模块 — 提供 `Arc<PriorityCalculator>` 能力
pub struct PriorityCalculatorModule;

/// Worker 管理器模块 — 提供 `Arc<WorkerManager>` 能力
pub struct WorkerManagerModule;

// ---------------------------------------------------------------------------
// 配置类型（通过 `Kit::set_config` 注入）
// ---------------------------------------------------------------------------

/// 缓存配置
#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub enabled: bool,
    pub size: usize,
}

/// 数据库配置
#[derive(Clone, Debug, Default)]
pub struct DbConfig {
    pub enabled: bool,
}

/// 认证启用配置（newtype 包装以避免 bool TypeId 冲突）
#[derive(Clone, Copy, Debug)]
pub struct AuthEnabled(pub bool);

/// 限流启用配置（newtype 包装以避免 bool TypeId 冲突）
#[derive(Clone, Copy, Debug)]
pub struct RateLimitEnabled(pub bool);

/// 管道启用配置（newtype 包装以避免 bool TypeId 冲突）
#[derive(Clone, Copy, Debug)]
pub struct PipelineEnabled(pub bool);
