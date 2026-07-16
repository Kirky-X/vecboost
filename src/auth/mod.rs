// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod csrf;
pub mod jwt;
pub mod middleware;
pub mod token_store;
pub mod types;
pub mod user_store;

pub use csrf::{CsrfConfig, CsrfProtection, CsrfToken, CsrfTokenStore, OriginValidator};
pub use jwt::JwtManager;
pub use middleware::{
    auth_middleware, auth_rate_limit_middleware, csrf_combined_middleware, csrf_middleware,
    csrf_origin_middleware, require_role_middleware,
};
pub use token_store::{MemoryTokenStore, TokenStore, TokenStoreFactory};
pub use types::{AuthResponse, LoginRequest, Permission, RefreshTokenRequest, User};
pub use user_store::{
    CreateUserRequest, StoredUser, UpdateUserRequest, UserStore, create_default_admin_user,
    hash_password, validate_password_complexity, validate_username_format,
};
