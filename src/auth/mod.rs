pub mod csrf;
pub mod handlers;
pub mod jwt;
pub mod middleware;
pub mod token_store;
pub mod types;
pub mod user_store;

pub use csrf::{CsrfConfig, CsrfProtection, CsrfToken, CsrfTokenStore, OriginValidator};
pub use handlers::{login_handler, logout_handler, refresh_token_handler};
pub use jwt::JwtManager;
pub use middleware::{
    auth_middleware, csrf_combined_middleware, csrf_middleware, csrf_origin_middleware,
    require_role_middleware,
};
pub use token_store::{MemoryTokenStore, TokenStore, TokenStoreFactory};
pub use types::{AuthResponse, LoginRequest, Permission, RefreshTokenRequest, User};
pub use user_store::{
    CreateUserRequest, StoredUser, UpdateUserRequest, UserStore, create_default_admin_user,
    hash_password, validate_password_complexity, validate_username_format,
};

// Re-export audit logger types for convenience (TODO: Implement audit logging)
// pub use crate::audit::{AuditLogger, SecurityEvent, SecurityEventType};
