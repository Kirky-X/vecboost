// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::{net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tower_http::{set_header::SetResponseHeaderLayer, trace::TraceLayer};
use vecboost::{
    AppConfig, AppState,
    audit::{AuditConfig, AuditLogger},
    auth::{
        CsrfConfig, CsrfTokenStore, JwtManager, UserStore, create_default_admin_user,
        validate_password_complexity,
    },
    config::model::{EngineType, ModelConfig},
    engine::AnyEngine,
    grpc::server::GrpcServer,
    pipeline::{PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel},
    rate_limit::{MemoryRateLimitStore, RateLimiter},
    security::{KeyStore, KeyType, SecretKey, SecurityConfig, StorageType, create_key_store},
    service::embedding::EmbeddingService,
};

// 使用 vecboost crate 中的路由模块
use vecboost::routes;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Rust Embedding Service...");

    let config = AppConfig::load()?;
    tracing::info!(
        "Configuration loaded: {} auth={} audit={}",
        if config.auth.enabled {
            "auth enabled"
        } else {
            "auth disabled"
        },
        config.auth.enabled,
        config.audit.enabled
    );

    let model_config = ModelConfig {
        name: config.model.model_repo.clone(),
        engine_type: EngineType::Candle,
        model_path: std::path::PathBuf::from(&config.model.model_repo),
        tokenizer_path: None,
        device: if config.model.use_gpu {
            vecboost::config::model::DeviceType::Cuda
        } else {
            vecboost::config::model::DeviceType::Cpu
        },
        max_batch_size: config.model.batch_size,
        pooling_mode: None,
        expected_dimension: config.model.expected_dimension,
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
    };

    tracing::info!("Initializing Inference Engine (this may take a while to download models)...");
    let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(AnyEngine::new(
        &model_config,
        EngineType::Candle,
        vecboost::config::model::Precision::Fp32,
    )?));

    let cache_enabled = config.embedding.cache_enabled;
    let cache_size = config.embedding.cache_size;

    let service = if cache_enabled && cache_size > 0 {
        tracing::info!("KV Cache enabled with size: {}", cache_size);
        EmbeddingService::with_cache(engine, Some(model_config), cache_size)
    } else {
        tracing::info!("KV Cache disabled");
        EmbeddingService::new(engine, Some(model_config))
    };
    let service = Arc::new(RwLock::new(service));

    // 创建限流器
    let rate_limiter = Arc::new(RateLimiter::new(Arc::new(MemoryRateLimitStore::new())));

    let (jwt_manager, user_store): (Option<Arc<JwtManager>>, Option<Arc<UserStore>>) = if config
        .auth
        .enabled
    {
        let security_config = SecurityConfig {
            storage_type: match config.auth.security.storage_type.as_str() {
                "encrypted_file" => StorageType::EncryptedFile,
                _ => StorageType::Environment,
            },
            encryption_key: config.auth.security.encryption_key.clone(),
            key_file_path: config.auth.security.key_file_path.clone(),
        };

        let key_store: Arc<dyn KeyStore> = {
            let boxed = create_key_store(&security_config).await?;
            Arc::from(boxed)
        };

        let jwt_secret_name = "jwt_secret";
        let _jwt_secret = if let Some(secret) = config.auth.jwt_secret {
            // 验证 JWT 密钥强度（至少 32 字节）
            if secret.len() < 32 {
                return Err(anyhow::anyhow!(
                    "JWT secret must be at least 32 characters long for security. Current length: {}",
                    secret.len()
                ));
            }
            let key = SecretKey::new(KeyType::JwtSecret, jwt_secret_name, secret);
            key_store.set(&key).await?;
            key.value
        } else {
            return Err(anyhow::anyhow!(
                "JWT secret is required when authentication is enabled. \
                     Please provide a strong JWT secret (at least 32 characters) in the configuration."
            ));
        };

        let jwt_manager = Arc::new(
            JwtManager::new_with_key_store(Arc::clone(&key_store), Some(jwt_secret_name))
                .await?
                .with_expiration(config.auth.token_expiration_hours.unwrap_or(24)),
        );

        let user_store = Arc::new(UserStore::new());

        // 强制用户提供管理员凭证，不再使用默认值
        let admin_username = config.auth.default_admin_username.ok_or_else(|| {
            anyhow::anyhow!(
                "Administrator username is required when authentication is enabled. \
                     Please set 'default_admin_username' in the configuration."
            )
        })?;

        let admin_password = config.auth.default_admin_password.ok_or_else(|| {
            anyhow::anyhow!(
                "Administrator password is required when authentication is enabled. \
                     Please set 'default_admin_password' in the configuration."
            )
        })?;

        // 验证密码复杂度
        validate_password_complexity(&admin_password)
            .map_err(|e| anyhow::anyhow!("Administrator password validation failed: {}", e))?;

        let admin_user = create_default_admin_user(&admin_username, &admin_password)
            .map_err(|e| anyhow::anyhow!("Failed to create default admin user: {}", e))?;
        user_store
            .add_user(admin_user)
            .map_err(|e| anyhow::anyhow!("Failed to add default admin user: {}", e))?;

        tracing::info!(
            "JWT authentication enabled with {} storage",
            config.auth.security.storage_type
        );
        tracing::info!("Default admin user created: {}", admin_username);

        (Some(jwt_manager), Some(user_store))
    } else {
        tracing::info!("JWT authentication disabled");
        (None, None)
    };

    // Initialize CSRF protection
    let (csrf_config, csrf_token_store): (Option<Arc<CsrfConfig>>, Option<Arc<CsrfTokenStore>>) =
        if config.auth.csrf.enabled {
            tracing::info!("CSRF protection enabled");

            let csrf_config =
                CsrfConfig::new(config.auth.csrf.allowed_origins.clone().unwrap_or_default())
                    .with_token_validation(config.auth.csrf.token_validation_enabled)
                    .with_token_expiration(config.auth.csrf.token_expiration_secs.unwrap_or(3600))
                    .with_allow_same_origin(config.auth.csrf.allow_same_origin);

            let csrf_token_store = if config.auth.csrf.token_validation_enabled {
                tracing::info!("CSRF token validation enabled");
                Some(Arc::new(CsrfTokenStore::new()))
            } else {
                tracing::info!("CSRF token validation disabled (using Origin validation only)");
                None
            };

            (Some(Arc::new(csrf_config)), csrf_token_store)
        } else {
            tracing::info!("CSRF protection disabled");
            (None, None)
        };

    // Initialize audit logging
    let audit_logger = if config.audit.enabled {
        tracing::info!("Audit logging enabled");
        let audit_config = AuditConfig {
            enabled: true,
            log_file_path: std::path::PathBuf::from(&config.audit.log_file_path),
            log_level: config.audit.log_level.clone(),
            max_file_size_mb: config.audit.max_file_size_mb,
            max_files: config.audit.max_files,
            async_write: true,
        };
        Some(Arc::new(AuditLogger::new(audit_config)))
    } else {
        tracing::warn!("Audit logging is DISABLED - security events will not be logged!");
        None
    };

    // Initialize pipeline if enabled
    let (pipeline_enabled, pipeline_queue, response_channel, priority_calculator) =
        if config.pipeline.enabled {
            tracing::info!(
                "Request pipeline enabled with queue_size={}",
                config.pipeline.queue.max_queue_size
            );

            let pipeline_queue = Arc::new(PriorityRequestQueue::new(
                config.pipeline.queue.max_queue_size,
            ));
            let response_channel = Arc::new(ResponseChannel::new());
            let priority_config = PriorityConfig {
                base_priority: config.pipeline.priority.base_priority,
                timeout_boost_factor: config.pipeline.priority.timeout_boost_factor,
                user_tier_weights: config.pipeline.priority.user_tier_weights,
                source_weights: config.pipeline.priority.source_weights,
            };
            let priority_calculator = Arc::new(PriorityCalculator::new(priority_config));

            tracing::info!("Pipeline components initialized successfully");

            (true, pipeline_queue, response_channel, priority_calculator)
        } else {
            tracing::info!("Request pipeline disabled");
            (
                false,
                Arc::new(PriorityRequestQueue::new(0)),
                Arc::new(ResponseChannel::new()),
                Arc::new(PriorityCalculator::new(PriorityConfig::default())),
            )
        };

    let app_state = AppState {
        service,
        jwt_manager: jwt_manager.clone(),
        user_store: user_store.clone(),
        auth_enabled: config.auth.enabled,
        csrf_config,
        csrf_token_store,
        metrics_collector: Some(Arc::new(vecboost::metrics::InferenceCollector::new())),
        prometheus_collector: Some(Arc::new(
            vecboost::metrics::PrometheusCollector::new()
                .map_err(|e| anyhow::anyhow!("Failed to create PrometheusCollector: {}", e))?,
        )),
        rate_limiter,
        ip_whitelist: config.rate_limit.ip_whitelist,
        rate_limit_enabled: config.rate_limit.enabled,
        audit_logger,
        pipeline_enabled,
        pipeline_queue,
        response_channel,
        priority_calculator,
    };

    let grpc_service = app_state.service.clone();

    // Using the new routing module to create the router
    let app = routes::create_router(app_state);

    let app = app
        .layer(TraceLayer::new_for_http())
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_CONTENT_TYPE_OPTIONS,
            axum::http::HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_FRAME_OPTIONS,
            axum::http::HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_XSS_PROTECTION,
            axum::http::HeaderValue::from_static("1; mode=block"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::STRICT_TRANSPORT_SECURITY,
            axum::http::HeaderValue::from_static("max-age=31536000; includeSubDomains"),
        ));

    // ConnectInfo is automatically available when using axum::serve with a TcpListener
    // No additional layer needed

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Server listening on {}", addr);

    if config.server.grpc_enabled {
        let grpc_host = config
            .server
            .grpc_host
            .unwrap_or_else(|| config.server.host.clone());
        let grpc_port = config.server.grpc_port.unwrap_or(50051);
        let grpc_addr: SocketAddr = format!("{}:{}", grpc_host, grpc_port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid gRPC address: {}", e))?;

        let grpc_server = GrpcServer::new(grpc_addr, grpc_service);

        tokio::spawn(async move {
            if let Err(e) = grpc_server.run().await {
                tracing::error!("gRPC server error: {}", e);
            }
        });

        tracing::info!("gRPC server enabled on {}", grpc_addr);
    }

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}
