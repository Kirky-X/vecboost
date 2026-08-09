// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Auth 模块 — garrison 集成层。
//!
//! 本模块将 VecBoost 的认证鉴权完全委托给 garrison 框架：
//! - `config` — AuthConfig → GarrisonConfig 映射
//! - `interface` — VecBoostInterface（GarrisonInterface 实现）
//! - `middleware` — axum 中间件（外壳保留，内部委托 garrison）
//! - `types` — HTTP 请求/响应类型（serde 序列化，无业务逻辑）

pub mod config;
pub mod interface;
pub mod middleware;
pub mod types;

// Re-export garrison 核心类型
pub use garrison::prelude::{GarrisonConfig, GarrisonManager, GarrisonUtil};

// Re-export VecBoost 适配类型
pub use config::map_auth_config_to_garrison;
pub use interface::VecBoostInterface;

/// Garrison 初始化标记类型。
///
/// garrison 使用全局单例模式（`GarrisonManager::init()` 后通过 `GarrisonUtil` 静态方法访问），
/// 无需在 kit 中持有 `GarrisonManager` 实例。此 newtype 作为 AuthModule 的 Capability，
/// `Some` 表示 garrison 已初始化，`None` 表示 auth 未启用。
#[derive(Clone, Debug)]
pub struct GarrisonHandle;

// Re-export 保留的 HTTP 类型（API 契约不变）
pub use types::{AuthResponse, LoginRequest, Permission, RefreshTokenRequest, User, validate_username_format};

// Re-export garrison 密码哈希（替代手写 argon2 实现）
pub use garrison::account::credential::password::{Argon2Hasher, PasswordHasher};

// Re-export garrison CSRF 类型与工具函数
#[cfg(feature = "auth")]
pub use garrison::web::csrf::{
    CsrfConfig as GarrisonCsrfConfig, generate_csrf_token, validate_csrf_token,
};

// Re-export garrison DAO（内存实现，供示例/测试使用）
#[cfg(feature = "auth")]
pub use garrison::dao::GarrisonDaoOxcache;

// Re-export middleware 函数
pub use middleware::{
    auth_middleware, auth_rate_limit_middleware, csrf_combined_middleware, csrf_middleware,
    csrf_origin_middleware, require_role_middleware,
};
