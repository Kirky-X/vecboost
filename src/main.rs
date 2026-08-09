// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#[cfg(feature = "cli")]
use std::collections::HashMap;
use std::{net::SocketAddr, sync::Arc};
use std::time::Duration;
use tokio::sync::RwLock;
use trait_kit::prelude::{AsyncShutdownCoordinator, BuildObserver, ShutdownPhase};
use tower_http::{set_header::SetResponseHeaderLayer, trace::TraceLayer};
use vecboost::AppConfig;
use vecboost::module_registry::RateLimitModule;
use vecboost::logger::LoggerModule;
#[cfg(feature = "auth")]
use vecboost::module_registry::{
    AuthModule, CsrfConfigModule,
};
use vecboost::{
    VecboostState,
    audit::{AuditConfig, AuditLogger},
    config::model::{EngineType, ModelConfig},
    engine::AnyEngine,
    module_registry::{
        AuditModule, AuthEnabled, CacheConfig, CacheModule,
        ConfigWatcherModule, DbConfig, DbModule, EmbeddingModule, IpWhitelistModule,
        MetricsCollectorModule, PipelineEnabled, PipelineQueueModule,
        PriorityCalculatorModule, PrometheusCollectorModule, RateLimitEnabled,
        RerankModule, ResponseChannelModule, WorkerManagerModule,
    },
    pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    },
    rate_limit::LimiteronAdapter,
    service::{embedding::EmbeddingService, rerank::RerankService},
};

#[cfg(feature = "cli")]
use sdforge::cli::{CliBuilder, CliCommandRegistration, CliHandlerRegistration};

#[cfg(feature = "db")]
use vecboost::db::{DbPool, init_schema};

#[cfg(feature = "auth")]
use vecboost::{
    auth::{
        GarrisonHandle, GarrisonCsrfConfig, VecBoostInterface,
        garrison_csrf_middleware, map_auth_config_to_garrison,
    },
};

#[cfg(feature = "grpc")]
use sdforge::grpc::{GrpcServerConfig, build_server_with_config};
#[cfg(all(feature = "grpc", feature = "auth"))]
use sdforge::security::BearerAuth;
#[cfg(feature = "grpc")]
use sdforge::security::ratelimit::LimiteronAdapter as SdforgeLimiteronAdapter;

// metrics 端点（Prometheus text/plain, forge 不支持非 JSON 响应, 保留手写）
use vecboost::metrics::metrics_endpoint;

/// Build observer that logs per-module build timing (T017a, observer feature).
struct LoggingObserver;

impl BuildObserver for LoggingObserver {
    fn on_module_start(&self, module_name: &'static str) {
        log::info!("Building module: {}", module_name);
    }

    fn on_module_built(&self, module_name: &'static str, elapsed: Duration) {
        if elapsed.as_secs_f64() > 1.0 {
            log::warn!(
                "Module {} built in {:.2}s (>1s)",
                module_name,
                elapsed.as_secs_f64()
            );
        } else {
            log::info!(
                "Module {} built in {:.2}ms",
                module_name,
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 日志初始化:inklog 完全接管日志输出(通过 log crate 宏 + inklog LogLogger 适配器)
    let logger_manager = Arc::new(
        inklog::LoggerManager::builder()
            .level("info")
            .console(true)
            .file("logs/vecboost.log")
            .file_compress(true) // T021: zstd compression for log files (inklog compression feature)
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize inklog logger: {}", e))?,
    );
    // logger_manager 通过 Arc 注入 kit，由 LoggerModule 管理生命周期，保持存活至 main 结束

    log::info!("Starting Rust Embedding Service...");

    // 确保所有 sdforge inventory（HTTP/MCP/CLI/gRPC）被链接器保留
    #[cfg(any(feature = "http", feature = "mcp", feature = "cli", feature = "grpc"))]
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
    let (db_pool, db_metrics) = {
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
        // T024-T025: Use DbPool::with_config to enable retry policy + pool-health-check
        // pool-health-check starts automatically in DbPool::with_config() (background task)
        let mut db_config = dbnexus::DbConfig {
            url: config.database.url.clone(),
            ..Default::default()
        };
        // T025: Enable retry policy for idempotent database operations
        // (dbnexus `retry` feature is always enabled when vecboost `db` feature is active)
        db_config.retry_policy = Some(dbnexus::RetryPolicy {
            max_retries: 3,
            ..Default::default()
        });
        log::info!("Database retry policy enabled (max_retries=3, exponential backoff)");
        let pool = DbPool::with_config(db_config)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create database pool: {}", e))?;
        init_schema(&pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize database schema: {}", e))?;
        log::info!("Database pool initialized and schema verified");
        // T044: Create standalone dbnexus MetricsCollector for Prometheus endpoint
        // (dbnexus pool internal metrics_collector is not yet settable from outside;
        // this standalone collector is wired to /metrics and ready for future pool integration)
        let db_metrics = Arc::new(dbnexus::MetricsCollector::new());
        log::info!("dbnexus MetricsCollector created (T044 observability wiring)");
        (pool, db_metrics)
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
        EmbeddingService::with_cache(engine.clone(), Some(model_config.clone()), cache_size)
    } else {
        log::info!("KV Cache disabled");
        EmbeddingService::new(engine.clone(), Some(model_config.clone()))
    };
    let service = Arc::new(RwLock::new(service));

    // Rerank service — reuses the same engine
    let rerank_service = Arc::new(RwLock::new(
        RerankService::new(engine.clone(), Some(model_config)),
    ));

    // MCP stdio run-mode: when `--mcp` is passed, serve the Model Context Protocol
    // over stdio and do NOT start the HTTP/gRPC servers (stdout must stay clean for
    // the JSON-RPC stream). Tools are generated by sdforge's `#[forge(tool_name =
    // ...)]` macros and collected via `sdforge::mcp::build()`.
    #[cfg(feature = "mcp")]
    if std::env::args().any(|a| a == "--mcp") {
        use sdforge::rmcp::{ServiceExt, transport::io::stdio};

        log::info!("Starting VecBoost MCP server over stdio");
        // kit：EmbeddingModule + RerankModule，供 forge handler 通过 state().kit.require 访问
        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(service.clone());
        kit.set_config(rerank_service.clone());
        kit.set_config(config.rerank.clone());
        kit.register::<EmbeddingModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
        kit.register::<RerankModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register RerankModule: {}", e))?;
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
            // kit：EmbeddingModule + RerankModule，供 forge handler 通过 state().kit.require 访问
            let mut kit = trait_kit::AsyncKit::new();
            kit.set_config(service.clone());
            kit.set_config(rerank_service.clone());
            kit.set_config(config.rerank.clone());
            kit.register::<EmbeddingModule>()
                .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
            kit.register::<RerankModule>()
                .map_err(|e| anyhow::anyhow!("Failed to register RerankModule: {}", e))?;
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
    let rate_limiter = Arc::new(LimiteronAdapter::with_defaults().await);

    // Garrison 认证初始化（替代手写 JWT/UserStore/CSRF）
    #[cfg(feature = "auth")]
    let garrison_handle: Option<Arc<GarrisonHandle>> = if config.auth.enabled {
        // 验证 JWT 密钥强度（至少 32 字节）
        if let Some(ref secret) = config.auth.jwt_secret {
            if secret.len() < 32 {
                return Err(anyhow::anyhow!(
                    "JWT secret must be at least 32 characters long for security. Current length: {}",
                    secret.len()
                ));
            }
        } else {
            return Err(anyhow::anyhow!(
                "JWT secret is required when authentication is enabled. \
                     Please provide a strong JWT secret (at least 32 characters) in the configuration."
            ));
        };

        // 创建 garrison DAO（内存缓存）
        let dao = garrison::dao::GarrisonDaoOxcache::new()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create GarrisonDaoOxcache: {}", e))?;

        // 映射 VecBoost AuthConfig → GarrisonConfig
        let garrison_config = map_auth_config_to_garrison(&config.auth);

        // 初始化 garrison 全局单例
        garrison::prelude::GarrisonManager::init(
            Arc::new(dao),
            Arc::new(garrison_config),
            Arc::new(VecBoostInterface::new(
                config.auth.default_admin_username.clone().unwrap_or_else(|| "admin".to_string()),
            )),
        )
        .map_err(|e| anyhow::anyhow!("Failed to init GarrisonManager: {}", e))?;

        log::info!("Garrison authentication enabled (JWT + session)");

        Some(Arc::new(GarrisonHandle))
    } else {
        log::info!("Authentication disabled");
        None
    };

    // Garrison CSRF 配置
    #[cfg(feature = "auth")]
    let garrison_csrf_config: Option<Arc<GarrisonCsrfConfig>> = if config.auth.csrf.enabled {
        let csrf = GarrisonCsrfConfig::default();
        log::info!("CSRF protection enabled (garrison)");
        Some(Arc::new(csrf))
    } else {
        log::info!("CSRF protection disabled");
        None
    };

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

    // T017a: Register build observer for per-module build timing
    kit.with_observer(Arc::new(LoggingObserver));

    // 注入预构建的能力对象（kit 是 single source of truth）
    kit.set_config(service.clone());
    kit.set_config(rerank_service.clone());
    kit.set_config(config.rerank.clone());
    kit.set_config(rate_limiter.clone());
    kit.set_config(CacheConfig {
        enabled: config.embedding.cache_enabled,
        size: config.embedding.cache_size,
    });
    kit.set_config(DbConfig {
        enabled: cfg!(feature = "db"),
    });
    // T044: Inject dbnexus MetricsCollector for Prometheus endpoint integration
    #[cfg(feature = "db")]
    kit.set_config(Some(db_metrics.clone()));
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
    // LoggerModule: Arc<inklog::LoggerManager> 能力注入
    kit.set_config(logger_manager.clone());
    #[cfg(feature = "auth")]
    {
        kit.set_config(garrison_handle.clone());
        kit.set_config(garrison_csrf_config.clone());
    }

    // 注册模块（15 个非 auth + 2 个 auth feature = 17 个 Module）
    kit.register::<EmbeddingModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
    kit.register::<RateLimitModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RateLimitModule: {}", e))?;
    kit.register::<RerankModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RerankModule: {}", e))?;
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
    kit.register::<PipelineQueueModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register PipelineQueueModule: {}", e))?;
    kit.register::<ResponseChannelModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register ResponseChannelModule: {}", e))?;
    kit.register::<PriorityCalculatorModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register PriorityCalculatorModule: {}", e))?;
    kit.register::<WorkerManagerModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register WorkerManagerModule: {}", e))?;
    // T034-T035: ConfigWatcherModule — monitors config.toml for hot reload
    let watcher_guard = Arc::new(confers::watcher::WatcherGuard::new());
    kit.set_config(watcher_guard);
    kit.register::<ConfigWatcherModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register ConfigWatcherModule: {}", e))?;
    kit.register::<LoggerModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register LoggerModule: {}", e))?;
    #[cfg(feature = "auth")]
    {
        kit.register::<AuthModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register AuthModule: {}", e))?;
        kit.register::<CsrfConfigModule>()
            .map_err(|e| anyhow::anyhow!("Failed to register CsrfConfigModule: {}", e))?;
    }

    // T012-T016: Register lifecycle and health check for key modules
    kit.register_lifecycle::<EmbeddingModule>();
    kit.register_lifecycle::<RerankModule>();
    kit.register_lifecycle::<RateLimitModule>();
    kit.register_lifecycle::<AuditModule>();
    kit.register_lifecycle::<ConfigWatcherModule>();
    kit.register_health_check::<EmbeddingModule>();
    kit.register_health_check::<RerankModule>();
    kit.register_health_check::<RateLimitModule>();
    kit.register_health_check::<CacheModule>();

    let kit = kit
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to build AsyncKit: {}", e))?;
    let kit = Arc::new(kit);

    // T017: AsyncShutdownCoordinator — phased graceful shutdown
    let shutdown_coordinator = AsyncShutdownCoordinator::new();
    shutdown_coordinator.set_global_timeout(Duration::from_secs(30));
    {
        let kit_for_shutdown = Arc::clone(&kit);
        shutdown_coordinator
            .register_hook(ShutdownPhase::CloseConnections, move || {
                Box::pin(async move {
                    // Manually invoke async on_shutdown for lifecycle modules
                    // (AsyncKit::shutdown() is sync and cannot call async fns)
                    if let Ok(audit_cap) = kit_for_shutdown.require::<AuditModule>() {
                        <AuditModule as trait_kit::prelude::AsyncLifecycle>::on_shutdown(&audit_cap)
                            .await;
                    }
                })
            })
            .map_err(|e| anyhow::anyhow!("Failed to register shutdown hook: {}", e))?;
    }
    // T035: Register ConfigWatcherModule shutdown hook
    {
        let kit_for_watcher_shutdown = Arc::clone(&kit);
        shutdown_coordinator
            .register_hook(ShutdownPhase::DrainQueue, move || {
                Box::pin(async move {
                    if let Ok(watcher_cap) = kit_for_watcher_shutdown.require::<ConfigWatcherModule>() {
                        <ConfigWatcherModule as trait_kit::prelude::AsyncLifecycle>::on_shutdown(
                            &watcher_cap,
                        )
                        .await;
                    }
                })
            })
            .map_err(|e| anyhow::anyhow!("Failed to register config watcher shutdown hook: {}", e))?;
    }

    log::info!("AsyncKit module registry built successfully");

    // T034: Spawn config file watcher task for hot reload
    // Watches config.toml and reloads configuration on file changes,
    // injecting new config through kit.set_config().
    {
        let kit_for_watch = Arc::clone(&kit);
        tokio::spawn(async move {
            // FsWatcher requires the file to exist; skip gracefully if not
            let config_path = "config.toml";
            let mut fs_watcher = match confers::watcher::FsWatcher::new(config_path, 200).await {
                Ok(w) => {
                    log::info!("Config file watcher started for {}", config_path);
                    w
                }
                Err(e) => {
                    log::warn!(
                        "Config file watcher not started ({} not found or error: {:?})",
                        config_path,
                        e
                    );
                    return;
                }
            };
            while let Some(changed_path) = fs_watcher.recv().await {
                log::info!("Config file changed: {:?}, reloading...", changed_path);
                match AppConfig::load_via_confers() {
                    Ok(new_config) => {
                        kit_for_watch.set_config(new_config);
                        log::info!("Configuration reloaded successfully");
                    }
                    Err(e) => {
                        log::error!("Failed to reload configuration: {}", e);
                    }
                }
            }
            log::info!("Config file watcher stopped");
        });
    }

    // v0.3.0 D3: VecboostState 仅持有 kit 单字段，所有能力通过 kit.require 查询
    let app_state = VecboostState::new(kit);

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
    // 直接使用 garrison garrison_csrf_middleware（包含 Origin + Token 双重校验）
    #[cfg(feature = "auth")]
    let app = if config.auth.enabled && config.auth.csrf.enabled {
        use axum::middleware::from_fn_with_state;
        let csrf_config = app_state
            .kit()
            .require::<CsrfConfigModule>()
            .map_err(|e| anyhow::anyhow!("Failed to require CsrfConfigModule: {}", e))?;
        if let Some(cfg) = csrf_config {
            app.layer(from_fn_with_state(
                cfg,
                garrison_csrf_middleware,
            ))
        } else {
            app
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

    // T018: Signal-aware graceful shutdown (SIGINT + SIGTERM)
    let signal = async {
        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    log::info!("Received SIGINT, initiating graceful shutdown");
                }
                _ = sigterm.recv() => {
                    log::info!("Received SIGTERM, initiating graceful shutdown");
                }
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install CTRL-C handler");
            log::info!("Received CTRL-C, initiating graceful shutdown");
        }
    };

    #[cfg(feature = "grpc")]
    if config.server.grpc_enabled {
        let grpc_host = config
            .server
            .grpc_host
            .clone()
            .unwrap_or_else(|| config.server.host.clone());
        let grpc_port = config.server.grpc_port.unwrap_or(50051);
        let grpc_addr = format!("{}:{}", grpc_host, grpc_port);

        // Secure default: require auth unless explicitly disabled via config.
        // config.server.grpc_require_auth defaults to Some(true) in ServerConfig::default().
        let require_auth = config.server.grpc_require_auth.unwrap_or(true);

        // Build BearerAuth when auth is enabled and a JWT secret is configured.
        // sdforge's GrpcServerConfig requires `auth: Option<BearerAuth>` (gated by
        // sdforge/security feature, which vecboost's grpc feature pulls in).
        let bearer_auth = if require_auth {
            #[cfg(feature = "auth")]
            {
                if config.auth.enabled {
                    if let Some(secret) = config.auth.jwt_secret.as_ref() {
                        match BearerAuth::try_new(secret.clone()) {
                            Ok(b) => {
                                log::info!("gRPC BearerAuth enabled (auth.enabled=true)");
                                Some(b)
                            }
                            Err(e) => {
                                anyhow::bail!(
                                    "gRPC require_auth=true but BearerAuth creation failed: {}. \
                                     Set VECBOOST_JWT_SECRET (>=32 chars) or set \
                                     [server] grpc_require_auth = false for dev",
                                    e
                                );
                            }
                        }
                    } else {
                        anyhow::bail!(
                            "gRPC require_auth=true but auth.jwt_secret is None. \
                             Set VECBOOST_JWT_SECRET env var or set \
                             [server] grpc_require_auth = false for dev"
                        );
                    }
                } else {
                    anyhow::bail!(
                        "gRPC require_auth=true but auth.enabled=false. \
                         Enable [auth] enabled = true or set [server] grpc_require_auth = false"
                    );
                }
            }
            #[cfg(not(feature = "auth"))]
            {
                anyhow::bail!(
                    "gRPC require_auth=true but vecboost `auth` feature is not enabled. \
                     Enable `auth` feature or set [server] grpc_require_auth = false in config"
                );
            }
        } else {
            log::warn!(
                "gRPC server starting with require_auth=false — \
                 this is insecure; use only for development behind network isolation"
            );
            None
        };

        // Build sdforge rate_limiter (gated by sdforge/ratelimit feature, which
        // vecboost's grpc feature pulls in). Uses default config (100 burst, 10 req/s).
        // `new()` panics only on invalid default config (should never happen).
        let rate_limiter: Option<std::sync::Arc<dyn sdforge::security::ratelimit::RateLimiter>> = {
            let limiter = SdforgeLimiteronAdapter::new().await
                .map_err(|e| anyhow::anyhow!("Failed to create gRPC rate limiter: {}", e))?;
            log::info!(
                "gRPC rate_limiter enabled (sdforge LimiteronAdapter, default config: 100 burst / 10 req/s)"
            );
            Some(std::sync::Arc::new(limiter))
        };

        let grpc_config = GrpcServerConfig {
            max_connections: config.server.grpc_max_connections.unwrap_or(1000),
            timeout_seconds: config.server.grpc_timeout_seconds.unwrap_or(30),
            require_auth,
            auth: bearer_auth,
            state: None,
            rate_limiter,
        };

        log::info!("gRPC server enabled on {}", grpc_addr);
        tokio::spawn(async move {
            if let Err(e) = build_server_with_config(&grpc_addr, grpc_config).await {
                log::error!("gRPC server error: {}", e);
            }
        });
    }

    // Wait for server to complete (graceful shutdown on signal)
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(signal)
    .await?;

    // Execute phased shutdown coordinator after server stops
    log::info!("Server stopped, executing phased shutdown...");
    let shutdown_result = shutdown_coordinator.shutdown().await;
    if !shutdown_result.is_ok() {
        log::warn!(
            "Shutdown timed out on phases: {:?}",
            shutdown_result.timed_out_phases()
        );
    }
    log::info!("VecBoost shutdown complete");

    Ok(())
}
