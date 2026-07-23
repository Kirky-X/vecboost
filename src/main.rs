// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#[cfg(feature = "cli")]
use std::collections::HashMap;
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tower_http::{set_header::SetResponseHeaderLayer, trace::TraceLayer};
use vecboost::AppConfig;
use vecboost::module_registry::RateLimitModule;
#[cfg(feature = "auth")]
use vecboost::module_registry::{
    AuthModule, CsrfConfigModule, CsrfTokenStoreModule, UserStoreModule,
};
use vecboost::{
    VecboostState,
    audit::{AuditConfig, AuditLogger},
    config::model::{EngineType, ModelConfig},
    engine::AnyEngine,
    module_registry::{
        AuditModule, AuthEnabled, AuthEnabledModule, CacheConfig, CacheModule, DbConfig, DbModule,
        EmbeddingModule, IpWhitelistModule, MetricsCollectorModule, PipelineEnabled,
        PipelineEnabledModule, PipelineQueueModule, PriorityCalculatorModule,
        PrometheusCollectorModule, RateLimitEnabled, RateLimitEnabledModule, ResponseChannelModule,
        WorkerManagerModule,
    },
    pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    },
    rate_limit::LimiteronAdapter,
    service::embedding::EmbeddingService,
};

#[cfg(feature = "cli")]
use sdforge::cli::{CliBuilder, CliCommandRegistration, CliHandlerRegistration};

#[cfg(feature = "db")]
use vecboost::db::{DbPool, init_schema};

#[cfg(feature = "auth")]
use vecboost::{
    auth::{
        CsrfConfig, CsrfTokenStore, JwtManager, UserStore, create_default_admin_user,
        validate_password_complexity,
    },
    security::{KeyStore, KeyType, SecretKey, SecurityConfig, StorageType, create_key_store},
};

#[cfg(feature = "grpc")]
use vecboost::grpc::server::GrpcServer;

// metrics 端点（Prometheus text/plain, forge 不支持非 JSON 响应, 保留手写）
use vecboost::metrics::metrics_endpoint;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 日志初始化:inklog 完全接管日志输出(通过 log crate 宏 + inklog LogLogger 适配器)
    let _logger_manager = inklog::LoggerManager::builder()
        .level("info")
        .console(true)
        .file("logs/vecboost.log")
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize inklog logger: {}", e))?;
    // _logger_manager 保持存活至 main 结束,避免 LoggerManager shutdown 导致日志停止

    log::info!("Starting Rust Embedding Service...");

    // 确保所有 sdforge inventory（HTTP/MCP/CLI）被链接器保留
    #[cfg(any(feature = "http", feature = "mcp", feature = "cli"))]
    {
        let _counts = sdforge::init_all_plugins();
    }

    let config = AppConfig::load_via_confers()
        .map_err(|e| anyhow::anyhow!("Failed to load config via confers: {}", e))?;
    log::info!(
        "Configuration loaded: {} auth={} audit={}",
        if config.auth.enabled {
            "auth enabled"
        } else {
            "auth disabled"
        },
        config.auth.enabled,
        config.audit.enabled
    );

    // 初始化数据库连接池（db feature 启用时）
    #[cfg(feature = "db")]
    let db_pool = {
        log::info!(
            "Initializing database pool with url={}",
            config
                .database
                .url
                .rsplit_once('@')
                .map(|(prefix, host)| {
                    let scheme = prefix.split("://").next().unwrap_or("db");
                    format!("{}://***@{}", scheme, host)
                })
                .unwrap_or_else(|| config.database.url.clone())
        );
        let pool = DbPool::new(&config.database.url)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create database pool: {}", e))?;
        init_schema(&pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize database schema: {}", e))?;
        log::info!("Database pool initialized and schema verified");
        pool
    };

    let model_config = ModelConfig {
        name: config.model.model_repo.clone(),
        engine_type: EngineType::Candle,
        model_path: match &config.model.model_path {
            Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
            _ => std::path::PathBuf::from(&config.model.model_repo),
        },
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

    log::info!("Initializing Inference Engine (this may take a while to download models)...");
    let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(
        vecboost::engine::EngineFactory::create(EngineType::Candle, &model_config)?,
    ));

    let cache_enabled = config.embedding.cache_enabled;
    let cache_size = config.embedding.cache_size;

    let service = if cache_enabled && cache_size > 0 {
        log::info!("KV Cache enabled with size: {}", cache_size);
        EmbeddingService::with_cache(engine, Some(model_config), cache_size)
    } else {
        log::info!("KV Cache disabled");
        EmbeddingService::new(engine, Some(model_config))
    };
    let service = Arc::new(RwLock::new(service));

    // MCP stdio run-mode: when `--mcp` is passed, serve the Model Context Protocol
    // over stdio and do NOT start the HTTP/gRPC servers (stdout must stay clean for
    // the JSON-RPC stream). Tools are generated by sdforge's `#[forge(tool_name =
    // ...)]` macros and collected via `sdforge::mcp::build()`.
    #[cfg(feature = "mcp")]
    if std::env::args().any(|a| a == "--mcp") {
        use rmcp::{ServiceExt, transport::io::stdio};

        log::info!("Starting VecBoost MCP server over stdio");
        // 最小 kit：仅 EmbeddingModule，供 forge handler 通过 state().kit.require 访问
        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(service.clone());
        kit.register::<EmbeddingModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
        let kit = kit
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to build AsyncKit: {}", e))?;
        vecboost::api::init_state(VecboostState::new(Arc::new(kit)));
        let server = sdforge::mcp::build();
        let running = server.serve(stdio()).await?;
        running.waiting().await?;
        return Ok(());
    }

    // CLI dispatch: sdforge CliBuilder 构建命令树 + 手写 dispatch
    // sdforge 只构建 clap::Command,不提供 dispatch;此处手动查找 handler 并调用
    #[cfg(feature = "cli")]
    {
        let cli_cmd = CliBuilder::new().with_name("vecboost").build();
        let first_arg = std::env::args().nth(1);
        let is_cli = first_arg
            .as_ref()
            .map(|cmd| {
                cli_cmd
                    .get_subcommands()
                    .any(|sc| sc.get_name() == cmd.as_str())
            })
            .unwrap_or(false);

        if is_cli {
            // 最小 kit：仅 EmbeddingModule，供 forge handler 通过 state().kit.require 访问
            let mut kit = trait_kit::AsyncKit::new();
            kit.set_config(service.clone());
            kit.register::<EmbeddingModule>()
                .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
            let kit = kit
                .build()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to build AsyncKit: {}", e))?;
            vecboost::api::init_state(VecboostState::new(Arc::new(kit)));
            let matches = cli_cmd.get_matches_from(std::env::args());

            if let Some((name, sub_matches)) = matches.subcommand() {
                // 从 ArgMatches 提取参数到 HashMap<String, String>
                let mut args_map = HashMap::new();
                for reg in sdforge::inventory::iter::<CliCommandRegistration>() {
                    if reg.name == name {
                        for arg in reg.args {
                            if let Some(val) = sub_matches.get_one::<String>(arg.name) {
                                args_map.insert(arg.name.to_string(), val.clone());
                            }
                        }
                        break;
                    }
                }

                // 查找并调用 handler
                let handler = sdforge::inventory::iter::<CliHandlerRegistration>()
                    .find(|h| h.name == name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("No handler registered for CLI command: {}", name)
                    })?;

                (handler.handler)(args_map, None)
                    .await
                    .map_err(|e| anyhow::anyhow!("CLI command '{}' failed: {:?}", name, e))?;
                return Ok(());
            }
            return Ok(());
        }
    }

    // 创建限流器
    let rate_limiter = Arc::new(LimiteronAdapter::with_default_config());

    #[cfg(feature = "auth")]
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
        let _jwt_secret = if let Some(ref secret) = config.auth.jwt_secret {
            // 验证 JWT 密钥强度（至少 32 字节）
            if secret.len() < 32 {
                return Err(anyhow::anyhow!(
                    "JWT secret must be at least 32 characters long for security. Current length: {}",
                    secret.len()
                ));
            }
            let key = SecretKey::new(KeyType::JwtSecret, jwt_secret_name, secret.clone());
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

        #[cfg(feature = "db")]
        let user_store = Arc::new(UserStore::new(Arc::new(db_pool.clone())));
        #[cfg(not(feature = "db"))]
        let user_store = Arc::new(UserStore::new());

        // 强制用户提供管理员凭证，不再使用默认值
        let admin_username = config.auth.default_admin_username.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "Administrator username is required when authentication is enabled. \
                     Please set 'default_admin_username' in the configuration."
            )
        })?;

        let admin_password = config.auth.default_admin_password.clone().ok_or_else(|| {
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
            .await
            .map_err(|e| anyhow::anyhow!("Failed to add default admin user: {}", e))?;

        log::info!(
            "JWT authentication enabled with {} storage",
            config.auth.security.storage_type
        );
        log::info!("Default admin user created: {}", admin_username);

        (Some(jwt_manager), Some(user_store))
    } else {
        log::info!("JWT authentication disabled");
        (None, None)
    };

    // Initialize CSRF protection
    #[cfg(feature = "auth")]
    let (csrf_config, csrf_token_store): (
        Option<Arc<CsrfConfig>>,
        Option<Arc<CsrfTokenStore>>,
    ) = if config.auth.csrf.enabled {
        log::info!("CSRF protection enabled");

        let csrf_config =
            CsrfConfig::new(config.auth.csrf.allowed_origins.clone().unwrap_or_default())
                .with_token_validation(config.auth.csrf.token_validation_enabled)
                .with_token_expiration(config.auth.csrf.token_expiration_secs.unwrap_or(3600))
                .with_allow_same_origin(config.auth.csrf.allow_same_origin);

        let csrf_token_store = if config.auth.csrf.token_validation_enabled {
            log::info!("CSRF token validation enabled");
            Some(Arc::new(CsrfTokenStore::new()))
        } else {
            log::info!("CSRF token validation disabled (using Origin validation only)");
            None
        };

        (Some(Arc::new(csrf_config)), csrf_token_store)
    } else {
        log::info!("CSRF protection disabled");
        (None, None)
    };

    // T017/T027: Warn on dangerous CSRF config combination — enabled but neither
    // token validation nor allowed_origins are configured, leaving CSRF
    // protection ineffective. Logic extracted to auth::csrf::check_csrf_dangerous_config
    // for unit-test coverage (T027).
    #[cfg(feature = "auth")]
    if let Some(warning) = vecboost::auth::csrf::check_csrf_dangerous_config(&config.auth.csrf) {
        log::warn!("{}", warning);
    }

    // Initialize audit logging
    let audit_logger = if config.audit.enabled {
        log::info!("Audit logging enabled");
        let audit_config = AuditConfig {
            enabled: true,
            log_file_path: std::path::PathBuf::from(&config.audit.log_file_path),
            log_level: config.audit.log_level.clone(),
            max_file_size_mb: config.audit.max_file_size_mb,
            max_files: config.audit.max_files,
            async_write: true,
        };
        #[cfg(feature = "db")]
        {
            Some(Arc::new(AuditLogger::new_with_db(
                audit_config,
                db_pool.clone(),
            )))
        }
        #[cfg(not(feature = "db"))]
        Some(Arc::new(AuditLogger::new(audit_config)))
    } else {
        log::warn!("Audit logging is DISABLED - security events will not be logged!");
        None
    };

    // Initialize pipeline if enabled
    let (pipeline_enabled, pipeline_queue, response_channel, priority_calculator, worker_manager) =
        if config.pipeline.enabled {
            log::info!(
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

            // Create WorkerManager with EmbeddingService
            let worker_config = vecboost::pipeline::WorkerConfig {
                min_workers: config.pipeline.worker.min_workers,
                max_workers: config.pipeline.worker.max_workers,
                scale_up_threshold: config.pipeline.worker.scale_up_threshold,
                scale_down_threshold: config.pipeline.worker.scale_down_threshold,
                scale_check_interval_secs: config.pipeline.worker.scale_check_interval_secs,
                idle_timeout_secs: config.pipeline.worker.idle_timeout_secs,
            };

            let worker_manager = Arc::new(WorkerManager::new(
                pipeline_queue.clone(),
                response_channel.clone(),
                worker_config.clone(),
                service.clone(), // Pass the Arc<RwLock<EmbeddingService>>
            ));

            // Start minimum workers
            for _ in 0..worker_config.min_workers {
                worker_manager.spawn_worker().await;
            }

            log::info!("Pipeline components initialized successfully");

            (
                true,
                pipeline_queue,
                response_channel,
                priority_calculator,
                worker_manager,
            )
        } else {
            log::info!("Request pipeline disabled");

            (
                false,
                Arc::new(PriorityRequestQueue::new(0)),
                Arc::new(ResponseChannel::new()),
                Arc::new(PriorityCalculator::new(PriorityConfig::default())),
                Arc::new(WorkerManager::new(
                    Arc::new(PriorityRequestQueue::new(0)),
                    Arc::new(ResponseChannel::new()),
                    WorkerConfig::default(),
                    service.clone(), // Pass the Arc<RwLock<EmbeddingService>>
                )),
            )
        };

    // ---------------------------------------------------------------------------
    // Module Registry (trait-kit AsyncKit) — D1 集成
    //
    // 使用 trait-kit 0.3 的 AsyncKit 构建模块依赖图。AsyncKit 是 Send + Sync
    // （基于 Arc<RwLock>），可安全存入 VecboostState 并跨线程共享。
    //
    // 模块采用"预构建能力注入"模式：需要异步构造的复杂对象（如 EmbeddingService）
    // 在上方已预构建，此处通过 kit.set_config() 注入，模块的 build() 从 config 检索。
    // ---------------------------------------------------------------------------

    let mut kit = trait_kit::AsyncKit::new();

    // 注入预构建的能力对象（kit 是 single source of truth）
    kit.set_config(service.clone());
    kit.set_config(rate_limiter.clone());
    kit.set_config(CacheConfig {
        enabled: config.embedding.cache_enabled,
        size: config.embedding.cache_size,
    });
    kit.set_config(DbConfig {
        enabled: cfg!(feature = "db"),
    });
    kit.set_config(audit_logger.clone());
    // v0.3.0 D3: 注入 13 个新 Module 的能力配置
    kit.set_config(Some(Arc::new(vecboost::metrics::InferenceCollector::new())));
    kit.set_config(Some(Arc::new(
        vecboost::metrics::PrometheusCollector::new()
            .map_err(|e| anyhow::anyhow!("Failed to create PrometheusCollector: {}", e))?,
    )));
    kit.set_config(config.rate_limit.ip_whitelist.clone());
    kit.set_config(config.embedding.clone());
    kit.set_config(AuthEnabled(config.auth.enabled));
    // T013: Inject AuthConfig for `trusted_proxies` (XFF trust boundary) access
    // via `kit.config::<AuthConfig>()` in `auth_middleware` (see lib.rs `FromRef` impl).
    kit.set_config(config.auth.clone());
    kit.set_config(RateLimitEnabled(config.rate_limit.enabled));
    kit.set_config(PipelineEnabled(pipeline_enabled));
    kit.set_config(pipeline_queue.clone());
    kit.set_config(response_channel.clone());
    kit.set_config(priority_calculator.clone());
    kit.set_config(worker_manager.clone());
    #[cfg(feature = "auth")]
    {
        if let Some(ref jwt) = jwt_manager {
            kit.set_config(Some(Arc::clone(jwt)) as Option<Arc<vecboost::auth::JwtManager>>);
        } else {
            kit.set_config(None::<Arc<vecboost::auth::JwtManager>>);
        }
        kit.set_config(user_store.clone());
        kit.set_config(csrf_config.clone());
        kit.set_config(csrf_token_store.clone());
    }

    // 注册模块（15 个非 auth + 4 个 auth feature = 19 个 Module）
    kit.register::<EmbeddingModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
    kit.register::<RateLimitModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RateLimitModule: {}", e))?;
    kit.register::<CacheModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register CacheModule: {}", e))?;
    kit.register::<DbModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register DbModule: {}", e))?;
    kit.register::<AuditModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register AuditModule: {}", e))?;
    // v0.3.0 D3: 注册 13 个新 Module
    kit.register::<MetricsCollectorModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register MetricsCollectorModule: {}", e))?;
    kit.register::<PrometheusCollectorModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register PrometheusCollectorModule: {}", e))?;
    kit.register::<IpWhitelistModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register IpWhitelistModule: {}", e))?;
    kit.register::<AuthEnabledModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register AuthEnabledModule: {}", e))?;
    kit.register::<RateLimitEnabledModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RateLimitEnabledModule: {}", e))?;
    kit.register::<PipelineEnabledModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register PipelineEnabledModule: {}", e))?;
    kit.register::<PipelineQueueModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register PipelineQueueModule: {}", e))?;
    kit.register::<ResponseChannelModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register ResponseChannelModule: {}", e))?;
    kit.register::<PriorityCalculatorModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register PriorityCalculatorModule: {}", e))?;
    kit.register::<WorkerManagerModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register WorkerManagerModule: {}", e))?;
    #[cfg(feature = "auth")]
    {
        kit.register::<AuthModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register AuthModule: {}", e))?;
        kit.register::<UserStoreModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register UserStoreModule: {}", e))?;
        kit.register::<CsrfConfigModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register CsrfConfigModule: {}", e))?;
        kit.register::<CsrfTokenStoreModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register CsrfTokenStoreModule: {}", e))?;
    }

    let kit = kit
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to build AsyncKit: {}", e))?;
    let kit = Arc::new(kit);

    log::info!("AsyncKit module registry built successfully");

    // v0.3.0 D3: VecboostState 仅持有 kit 单字段，所有能力通过 kit.require 查询
    let app_state = VecboostState::new(kit);

    let _grpc_service = app_state
        .kit()
        .require::<EmbeddingModule>()
        .map_err(|e| anyhow::anyhow!("Failed to require EmbeddingModule for gRPC: {}", e))?
        .clone();

    // 注入 state 到 api 模块（统一入口：所有 forge handler 通过 state().kit.require 访问）
    vecboost::api::init_state(app_state.clone());

    // sdforge #[forge] 路由（Router<()>，从 inventory 收集所有 forge 函数注册的路由）
    let app = sdforge::http::build();

    // metrics 端点（手写例外：Prometheus text/plain 响应，forge 不支持非 JSON）
    let metrics_router = axum::Router::new()
        .route("/metrics", axum::routing::get(metrics_endpoint))
        .with_state(app_state.clone());
    let app = app.merge(metrics_router);

    // auth_middleware：应用到所有路由，内部用路径白名单放行公开端点
    // (/health, /api/v1/auth/login, /api/v1/auth/refresh)
    #[cfg(feature = "auth")]
    let app = if config.auth.enabled {
        use axum::middleware::from_fn_with_state;
        app.layer(from_fn_with_state(
            app_state.clone(),
            vecboost::auth::auth_middleware,
        ))
        .layer(from_fn_with_state(
            app_state.clone(),
            vecboost::auth::auth_rate_limit_middleware,
        ))
    } else {
        app
    };

    // CSRF 保护（条件性应用：auth 启用且 csrf 启用时）
    #[cfg(feature = "auth")]
    let app = if config.auth.enabled && config.auth.csrf.enabled {
        use axum::middleware::from_fn_with_state;
        let csrf_config = app_state
            .kit()
            .require::<CsrfConfigModule>()
            .map_err(|e| anyhow::anyhow!("Failed to require CsrfConfigModule: {}", e))?;
        let csrf_token_store = app_state
            .kit()
            .require::<CsrfTokenStoreModule>()
            .map_err(|e| anyhow::anyhow!("Failed to require CsrfTokenStoreModule: {}", e))?;
        match (csrf_config, csrf_token_store) {
            (Some(cfg), Some(store)) if cfg.token_validation_enabled => {
                app.layer(from_fn_with_state(
                    (cfg, store),
                    vecboost::auth::middleware::csrf_combined_middleware,
                ))
            }
            (Some(cfg), _) => app.layer(from_fn_with_state(
                cfg,
                vecboost::auth::csrf_origin_middleware,
            )),
            _ => app,
        }
    } else {
        app
    };

    // 安全 headers + trace
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
    log::info!("Server listening on {}", addr);

    #[cfg(feature = "grpc")]
    if config.server.grpc_enabled {
        let grpc_host = config
            .server
            .grpc_host
            .unwrap_or_else(|| config.server.host.clone());
        let grpc_port = config.server.grpc_port.unwrap_or(50051);
        let grpc_addr: SocketAddr = format!("{}:{}", grpc_host, grpc_port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid gRPC address: {}", e))?;

        let grpc_server = GrpcServer::new(grpc_addr, _grpc_service);

        tokio::spawn(async move {
            if let Err(e) = grpc_server.run().await {
                log::error!("gRPC server error: {}", e);
            }
        });

        log::info!("gRPC server enabled on {}", grpc_addr);
    }

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}
