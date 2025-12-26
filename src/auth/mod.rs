pub mod handlers;
pub mod jwt;
pub mod middleware;
pub mod types;
pub mod user_store;

pub use handlers::{login_handler, logout_handler, me_handler, refresh_token_handler};
pub use jwt::{Claims, JwtManager};
pub use middleware::{JwtAuthLayer, auth_middleware};
pub use types::{AuthRequest, AuthResponse, LoginRequest};
pub use user_store::{StoredUser, UserStore, create_default_admin_user, create_default_user};
