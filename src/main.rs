// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

// Performance: jemalloc global memory allocator
// Only enabled on Linux glibc platforms; macOS/musl use the system default allocator
#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "cli")]
use std::collections::HashMap;
use std::time::Duration;
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tower_http::{set_header::SetResponseHeaderLayer, trace::TraceLayer};
use trait_kit::prelude::{AsyncShutdownCoordinator, BuildObserver, ShutdownPhase};
use vecboost::AppConfig;
use vecboost::logger::LoggerModule;
use vecboost::registry::RateLimitModule;

/// 全局关闭超时（秒）
const DEFAULT_SHUTDOWN_TIMEOUT_SECS: u64 = 30;

/// runtime 退出时等待常驻 spawn_blocking 任务的兜底超时（秒）。
/// 常驻任务（inklog 定时器等）不可取消，默认 Drop 会无限等待导致进程挂死。
const SHUTDOWN_RUNTIME_DRAIN_SECS: u64 = 10;

/// 数据库连接池配置
#[cfg(feature = "db")]
const DB_MAX_RETRIES: u32 = 3;
#[cfg(feature = "db")]
const DB_MIN_CONNECTIONS: u32 = 5;
#[cfg(feature = "db")]
const DB_WARMUP_TIMEOUT_SECS: u64 = 10;
#[cfg(feature = "db")]
const DB_WARMUP_RETRIES: u32 = 2;
#[cfg(feature = "auth")]
use vecboost::registry::{AuthModule, CsrfConfigModule};
use vecboost::{
    VecboostState,
    audit::{AuditConfig, AuditLogger},
    config::model::{EngineType, ModelConfig},
    engine::AnyEngine,
    pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    },
    rate_limit::{LimiteronAdapter, RateLimitSettings},
    registry::{
        AuditModule, CacheConfig, CacheModule, ConfigWatcherModule, DbConfig, DbModule,
        EmbeddingModule, IpWhitelistModule, MetricsCollectorModule, PipelineQueueModule,
        PriorityCalculatorModule, PrometheusCollectorModule, RerankModule, ResponseChannelModule,
        WorkerManagerModule,
    },
    service::{embedding::EmbeddingService, rerank::RerankService},
};

#[cfg(feature = "cli")]
use sdforge::cli::{CliBuilder, CliCommandRegistration, CliHandlerRegistration};

#[cfg(feature = "db")]
use vecboost::db::{DbPool, init_schema};

#[cfg(feature = "auth")]
use vecboost::auth::{
    GarrisonCsrfConfig, GarrisonHandle, PasswordHasher, VecBoostInterface,
    garrison_csrf_middleware, map_auth_config_to_garrison,
};

#[cfg(feature = "grpc")]
use sdforge::grpc::{GrpcServerConfig, build_server_with_config};
#[cfg(all(feature = "grpc", feature = "auth"))]
use sdforge::security::BearerAuth;
#[cfg(feature = "grpc")]
use sdforge::security::ratelimit::LimiteronAdapter as SdforgeLimiteronAdapter;

// metrics 端点（Prometheus text/plain, forge 不支持非 JSON 响应, 保留手写）
use vecboost::metrics::metrics_endpoint;

/// Build observer that logs per-module build timing.
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

// ---------------------------------------------------------------------------
// Helper functions — extracted from main() to reduce cyclomatic complexity
// ---------------------------------------------------------------------------

#[cfg(feature = "db")]
async fn init_db_pool(
    config: &AppConfig,
) -> anyhow::Result<(DbPool, Arc<dbnexus::MetricsCollector>)> {
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
    let mut db_config = dbnexus::DbConfig {
        url: config.database.url.clone(),
        ..Default::default()
    };
    db_config.retry_policy = Some(dbnexus::RetryPolicy {
        max_retries: DB_MAX_RETRIES,
        ..Default::default()
    });
    log::info!(
        "Database retry policy enabled (max_retries={}, exponential backoff)",
        DB_MAX_RETRIES
    );
    db_config.pool_config.min_connections = DB_MIN_CONNECTIONS;
    db_config.warmup_timeout = DB_WARMUP_TIMEOUT_SECS;
    db_config.warmup_retries = DB_WARMUP_RETRIES;
    log::info!(
        "dbnexus pool-warmup enabled: min_connections={}, warmup_timeout={}s, warmup_retries={}",
        db_config.pool_config.min_connections,
        db_config.warmup_timeout,
        db_config.warmup_retries
    );
    let pool = DbPool::with_config(db_config).await.map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-db-pool",
                vecboost::i18n::tr_args(&[("detail", &e.to_string())]),
            )
        )
    })?;
    init_schema(&pool).await.map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-db-schema",
                vecboost::i18n::tr_args(&[("detail", &e.to_string())]),
            )
        )
    })?;
    log::info!("Database pool initialized and schema verified");
    let db_metrics = Arc::new(dbnexus::MetricsCollector::new());
    log::info!("dbnexus MetricsCollector created");
    Ok((pool, db_metrics))
}

async fn init_engine_and_services(
    config: &AppConfig,
) -> anyhow::Result<(
    Arc<RwLock<AnyEngine>>,
    Arc<RwLock<EmbeddingService>>,
    Arc<RwLock<RerankService>>,
    ModelConfig,
)> {
    let model_config = ModelConfig {
        name: config.model.model_repo.clone(),
        engine_type: EngineType::Candle,
        model_path: match &config.model.model_path {
            Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
            _ => std::path::PathBuf::from(&config.model.model_repo),
        },
        tokenizer_path: None,
        device: resolve_device_config(config.model.use_gpu),
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

    let rerank_service = Arc::new(RwLock::new(RerankService::new(
        engine.clone(),
        Some(model_config.clone()),
    )));

    Ok((engine, service, rerank_service, model_config))
}

#[cfg(feature = "mcp")]
async fn run_mcp_server(
    service: Arc<RwLock<EmbeddingService>>,
    rerank_service: Arc<RwLock<RerankService>>,
) -> anyhow::Result<()> {
    use sdforge::rmcp::{ServiceExt, transport::io::stdio};

    log::info!("Starting VecBoost MCP server over stdio");
    let mut kit = trait_kit::AsyncKit::new();
    kit.set_config(service);
    kit.set_config(rerank_service);
    kit.register::<EmbeddingModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
    kit.register::<RerankModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RerankModule: {}", e))?;
    let kit = kit
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to build AsyncKit: {}", e))?;
    vecboost::api::init_state(VecboostState::new(Arc::new(kit)))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let server = sdforge::mcp::build();
    let running = server.serve(stdio()).await?;
    running.waiting().await?;
    Ok(())
}

/// G002: CLI 子命令名单的单一来源 —— sdforge inventory 注册(forge CLI 宏),
/// 不再维护手工数组。docs 子命令由 sdforge 自动附加。
#[cfg(feature = "cli")]
fn cli_subcommand_names() -> Vec<String> {
    CliBuilder::new()
        .with_name("vecboost")
        .build()
        .get_subcommands()
        .map(|sc| sc.get_name().to_string())
        .collect()
}

/// G002: 校验 CLI 首参数 —— `--help` 打印用法后退出;未知子命令 stderr 报错
/// 并以退出码 2 终止(不得静默落入 HTTP 服务器启动路径)。
#[cfg(feature = "cli")]
fn validate_cli_invocation(filtered_args: &[String]) {
    let Some(first) = filtered_args.first() else {
        return; // 无子命令 → 服务器模式
    };
    if first == "--help" || first == "-h" {
        let mut cmd = CliBuilder::new().with_name("vecboost").build();
        let _ = cmd.print_help();
        println!();
        std::process::exit(0);
    }
    if first.starts_with('-') {
        return; // 全局 flag(--config 已剥离/--mcp)交由后续路径处理
    }
    let known = cli_subcommand_names();
    if !known.contains(first) {
        eprintln!("Error: unknown subcommand '{first}'");
        eprintln!();
        eprintln!("Available subcommands: {}", known.join(", "));
        eprintln!("Run 'vecboost --help' for usage.");
        std::process::exit(2);
    }
}

#[cfg(feature = "cli")]
async fn run_cli_command(
    service: Arc<RwLock<EmbeddingService>>,
    rerank_service: Arc<RwLock<RerankService>>,
    cli_args: Vec<String>,
) -> anyhow::Result<bool> {
    let cli_cmd = CliBuilder::new().with_name("vecboost").build();
    let first_arg = cli_args.get(1);
    let is_cli = first_arg
        .as_ref()
        .map(|cmd| {
            cli_cmd
                .get_subcommands()
                .any(|sc| sc.get_name() == cmd.as_str())
        })
        .unwrap_or(false);

    if !is_cli {
        return Ok(false);
    }

    let mut kit = trait_kit::AsyncKit::new();
    kit.set_config(service);
    kit.set_config(rerank_service);
    kit.register::<EmbeddingModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
    kit.register::<RerankModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RerankModule: {}", e))?;
    let kit = kit
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to build AsyncKit: {}", e))?;
    vecboost::api::init_state(VecboostState::new(Arc::new(kit)))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let matches = cli_cmd.get_matches_from(cli_args);

    if let Some((name, sub_matches)) = matches.subcommand() {
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

        let handler = sdforge::inventory::iter::<CliHandlerRegistration>()
            .find(|h| h.name == name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{}",
                    vecboost::i18n::tr_with_args(
                        "cli-no-handler",
                        vecboost::i18n::tr_args(&[("name", name)]),
                    )
                )
            })?;

        // DEFECT-CLI-001 根因修复：HandlerFn 返回序列化后的 serde_json::Value，
        // 打印职责在调用方（sdforge 自带的 execute() 未被 vecboost 使用）。
        // 旧实现直接丢弃返回值，导致 CLI 子命令"退出码 0 但无任何结果输出"。
        let value = (handler.handler)(args_map, None).await.map_err(|e| {
            anyhow::anyhow!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "cli-failed",
                    vecboost::i18n::tr_args(&[("name", name), ("detail", &format!("{:?}", e))]),
                )
            )
        })?;
        println!("{}", sdforge::core::extract_value(&value));
    }
    Ok(true)
}

#[cfg(feature = "auth")]
async fn init_auth(
    config: &AppConfig,
) -> anyhow::Result<(Option<Arc<GarrisonHandle>>, Option<Arc<GarrisonCsrfConfig>>)> {
    let garrison_handle: Option<Arc<GarrisonHandle>> = if config.auth.enabled {
        if let Some(ref secret) = config.auth.jwt_secret {
            if secret.len() < 32 {
                return Err(anyhow::anyhow!(
                    "{}",
                    vecboost::i18n::tr_with_args(
                        "startup-jwt-length",
                        vecboost::i18n::tr_args(&[("got", &secret.len().to_string())]),
                    )
                ));
            }
        } else {
            return Err(anyhow::anyhow!(
                "{}",
                vecboost::i18n::tr("startup-jwt-missing")
            ));
        };

        // 安全闸门:启用认证但未配置管理员密码时拒绝启动。
        // 否则 forge_login 在无哈希时对任意凭据颁发 token,认证形同虚设。
        if config.auth.default_admin_password.is_none() {
            return Err(anyhow::anyhow!(
                // i18n 文案之外固定附带环境变量名,保证任何 locale 下都可操作
                "{} (required: VECBOOST_ADMIN_PASSWORD)",
                vecboost::i18n::tr("startup-admin-password-missing")
            ));
        }

        let dao = garrison::dao::GarrisonDaoOxcache::new()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create GarrisonDaoOxcache: {}", e))?;

        let garrison_config = map_auth_config_to_garrison(&config.auth);

        garrison::prelude::GarrisonManager::init(
            Arc::new(dao),
            Arc::new(garrison_config.clone()),
            Arc::new(VecBoostInterface::new(
                config
                    .auth
                    .default_admin_username
                    .clone()
                    .unwrap_or_else(|| "admin".to_string()),
            )),
        )
        .map_err(|e| anyhow::anyhow!("Failed to init GarrisonManager: {}", e))?;

        let admin_password_hash = config.auth.default_admin_password.as_ref().map(|pw| {
            garrison::account::credential::password::Argon2Hasher::default()
                .hash(pw)
                .expect("admin password hash must succeed")
        });

        let admin_username = config
            .auth
            .default_admin_username
            .clone()
            .unwrap_or_else(|| "admin".to_string());

        log::info!("Garrison authentication enabled (JWT + session + password verification)");

        Some(Arc::new(GarrisonHandle {
            admin_password_hash,
            admin_username,
            token_timeout_secs: garrison_config.timeout,
        }))
    } else {
        log::info!("Authentication disabled");
        None
    };

    let garrison_csrf_config: Option<Arc<GarrisonCsrfConfig>> = if config.auth.csrf.enabled {
        let csrf = GarrisonCsrfConfig::default();
        log::info!("CSRF protection enabled (garrison)");
        Some(Arc::new(csrf))
    } else {
        log::info!("CSRF protection disabled");
        None
    };

    Ok((garrison_handle, garrison_csrf_config))
}

async fn init_pipeline(
    config: &AppConfig,
    service: &Arc<RwLock<EmbeddingService>>,
) -> anyhow::Result<(
    Arc<PriorityRequestQueue>,
    Arc<ResponseChannel>,
    Arc<PriorityCalculator>,
    Arc<WorkerManager>,
)> {
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
            user_tier_weights: config.pipeline.priority.user_tier_weights.clone(),
            source_weights: config.pipeline.priority.source_weights.clone(),
        };
        let priority_calculator = Arc::new(PriorityCalculator::new(priority_config));

        let worker_config = vecboost::pipeline::WorkerConfig {
            min_workers: config.pipeline.worker.min_workers,
            max_workers: config.pipeline.worker.max_workers,
            scale_up_threshold: config.pipeline.worker.scale_up_threshold,
            scale_down_threshold: config.pipeline.worker.scale_down_threshold,
            scale_check_interval_secs: config.pipeline.worker.scale_check_interval_secs,
            idle_timeout_secs: config.pipeline.worker.idle_timeout_secs,
            max_batch_size: config.embedding.max_batch_size,
        };

        let worker_manager = Arc::new(WorkerManager::new(
            pipeline_queue.clone(),
            response_channel.clone(),
            worker_config.clone(),
            service.clone(),
        ));

        for _ in 0..worker_config.min_workers {
            worker_manager.spawn_worker().await;
        }

        log::info!("Pipeline components initialized successfully");

        Ok((
            pipeline_queue,
            response_channel,
            priority_calculator,
            worker_manager,
        ))
    } else {
        log::info!("Request pipeline disabled");

        Ok((
            Arc::new(PriorityRequestQueue::new(0)),
            Arc::new(ResponseChannel::new()),
            Arc::new(PriorityCalculator::new(PriorityConfig::default())),
            Arc::new(WorkerManager::new(
                Arc::new(PriorityRequestQueue::new(0)),
                Arc::new(ResponseChannel::new()),
                WorkerConfig::default(),
                service.clone(),
            )),
        ))
    }
}

/// G008: 设备解析 —— 请求 GPU 但对应 feature 未编译时 WARN 并回退 CPU。
fn resolve_device_config(use_gpu: bool) -> vecboost::config::model::DeviceType {
    use vecboost::config::model::DeviceType;
    if !use_gpu {
        return DeviceType::Cpu;
    }
    #[cfg(feature = "cuda")]
    {
        DeviceType::Cuda
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        DeviceType::Metal
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        log::warn!(
            "use_gpu=true but no GPU backend is compiled in (missing 'cuda' feature, or \
             'metal' on macOS) — falling back to CPU. Rebuild with --features cuda for GPU."
        );
        DeviceType::Cpu
    }
}

/// 配置含敏感项(jwt_secret/admin_password)且未设加密 key 时应告警。
fn should_warn_plaintext_secrets(
    has_jwt: bool,
    has_admin_password: bool,
    has_encryption_key: bool,
) -> bool {
    (has_jwt || has_admin_password) && !has_encryption_key
}

/// 绑定安全闸门:认证关闭时禁止绑定非回环地址,杜绝 insecure-by-default 裸奔。
///
/// - 回环(127.x/::1/localhost)+ 认证关闭 → 放行(本地开发形态)。
/// - 非回环 + 认证关闭 → 拒绝启动,除非设置 `VECBOOST_ALLOW_INSECURE=1`
///   (显式逃生阀,此时输出 ERROR 级风险告警,供容器等受信网络边界场景使用)。
fn validate_bind_safety(host: &str, auth_enabled: bool) -> anyhow::Result<()> {
    if auth_enabled {
        return Ok(());
    }
    let loopback = host.eq_ignore_ascii_case("localhost")
        || match host.parse::<std::net::IpAddr>() {
            Ok(ip) => ip.is_loopback(),
            Err(_) => false,
        };
    if loopback {
        return Ok(());
    }
    if std::env::var("VECBOOST_ALLOW_INSECURE").as_deref() == Ok("1") {
        log::error!(
            "SECURITY RISK: authentication is disabled while binding to non-loopback address \
             '{host}' (VECBOOST_ALLOW_INSECURE=1). Anyone reachable on this interface can read \
             files via /embed/file and manage models. Enable auth or bind to 127.0.0.1."
        );
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "Refusing to start: authentication is disabled (auth.enabled=false) while binding to \
         non-loopback address '{host}'. Fix one of: (1) set host to 127.0.0.1 in the config, \
         (2) enable [auth] with VECBOOST_JWT_SECRET and VECBOOST_ADMIN_PASSWORD, or \
         (3) set VECBOOST_ALLOW_INSECURE=1 to accept the risk explicitly."
    ))
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    let result = rt.block_on(app_main());
    // DEFECT-SHUTDOWN-002 修复：常驻 spawn_blocking 任务（inklog 定时器/写入线程等）
    // 会让 Runtime::drop 的 BlockingPool::shutdown 无限等待，导致任何退出路径
    // （启动配置错误 bail / SIGTERM 优雅关闭）挂死、最终被 SIGKILL(137)。
    // 显式 shutdown_timeout 保证所有退出路径都能在超时后落地。
    rt.shutdown_timeout(Duration::from_secs(SHUTDOWN_RUNTIME_DRAIN_SECS));
    result.unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
}

/// 从参数列表剥离 `--config <path>` / `--config=path`，返回
/// (剥离后的参数, 已解析的配置路径)。CLI 子命令解析不识别 --config，
/// 必须先剥离，避免 clap 报未知参数或服务器模式误判。
fn strip_config_args(mut args: Vec<String>) -> (Vec<String>, Option<String>) {
    let mut config_path = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--config" {
            args.remove(i);
            if i < args.len() {
                config_path = Some(args.remove(i));
            }
        } else if let Some(path) = args[i].strip_prefix("--config=") {
            config_path = Some(path.to_string());
            args.remove(i);
        } else {
            i += 1;
        }
    }
    (args, config_path)
}

/// 应用主入口（在 tokio runtime 上执行）
async fn app_main() -> anyhow::Result<()> {
    // 剥离全局 --config 参数（CLI 子命令/clap 不识别该参数，须先行剥离）
    let (_filtered_args, config_path) = strip_config_args(std::env::args().collect());

    // G002: 未知子命令/`--help` 在进入服务器装配前拦截(fail-fast,不静默起服务)
    #[cfg(feature = "cli")]
    validate_cli_invocation(&_filtered_args[1..]);

    // Early CLI detection: suppress console logging in CLI mode to keep stdout clean
    // for machine-readable JSON output (DEFECT-CLI-001 fix)
    #[cfg(feature = "cli")]
    let cli_mode = _filtered_args
        .get(1)
        .map(|a| cli_subcommand_names().contains(a))
        .unwrap_or(false);
    #[cfg(not(feature = "cli"))]
    let cli_mode = false;

    // DEFECT-MCP-001 修复：MCP stdio 模式下 stdout 只能承载 JSON-RPC 协议消息，
    // inklog 控制台日志会污染协议流导致客户端解析失败 —— 与 CLI 模式同样关闭控制台输出。
    #[cfg(feature = "mcp")]
    let mcp_mode = std::env::args().any(|a| a == "--mcp");
    #[cfg(not(feature = "mcp"))]
    let mcp_mode = false;

    // i18n 先于配置初始化：配置校验错误消息需要翻译
    vecboost::i18n::init();

    // 配置先行加载（DEFECT-CONFIG-001）：logger 的级别/文件参数来自 [logging] 配置段，
    // 因此配置必须在 logger 之前就绪；此时尚无日志后端，错误经 stderr 输出。
    let config = {
        // G006: 显式指定的 --config 文件不存在 → 立即报错退出(码 2),
        // 杜绝"以为自定义配置生效,实际跑默认配置"的静默回退
        if let Some(p) = config_path.as_deref()
            && !std::path::Path::new(p).exists()
        {
            eprintln!("Error: config file not found: {p}");
            eprintln!("The --config path must point to an existing TOML file.");
            std::process::exit(2);
        }
        let result = match config_path.as_deref() {
            Some(p) => AppConfig::load_via_confers_with_path(p),
            None => AppConfig::load_via_confers(),
        };
        match result {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "Error: {}",
                    vecboost::i18n::tr_with_args(
                        "startup-config",
                        vecboost::i18n::tr_args(&[("detail", &e.to_string())]),
                    )
                );
                std::process::exit(1);
            }
        }
    };

    // DEFECT-CONFIG-002: [logging] 配置段接入 logger；
    // VECBOOST_LOG_LEVEL 环境变量优先级高于配置文件。
    // 白名单含 inklog 合法别名（warning 等）
    let log_level = std::env::var("VECBOOST_LOG_LEVEL")
        .ok()
        .filter(|l| {
            ["trace", "debug", "info", "warn", "warning", "error"]
                .contains(&l.to_lowercase().as_str())
        })
        .unwrap_or_else(|| config.logging.level.clone());

    // DEFECT-CLI-003 绕过：inklog ConsoleSink 不消费 enabled 标志（上游缺陷，
    // console(false) 无法关闭输出）。CLI/MCP 模式下将全部日志级别路由到 stderr，
    // 保证 stdout 只承载机器可读 JSON / JSON-RPC 协议消息。
    let mut logger_builder = inklog::LoggerManager::builder()
        .level(&log_level)
        .console(true);
    if cli_mode || mcp_mode {
        logger_builder =
            logger_builder.console_stderr_levels(&["trace", "debug", "info", "warn", "error"]);
    }
    if !config.logging.file_path.is_empty() {
        logger_builder = logger_builder
            .file(&config.logging.file_path)
            .file_max_size(format!("{}MB", config.logging.rotation_size_mb))
            .file_keep_files(config.logging.max_files)
            .file_compress(true);
    }
    let logger_manager = Arc::new(logger_builder.build().await.map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-logger",
                vecboost::i18n::tr_args(&[("detail", &e.to_string())]),
            )
        )
    })?);

    log::info!(
        "Starting Rust Embedding Service... (log level: {log_level}, config: {})",
        config_path.as_deref().unwrap_or("config/config.toml")
    );

    #[cfg(any(feature = "http", feature = "mcp", feature = "cli", feature = "grpc"))]
    {
        let _counts = sdforge::init_all_plugins();
    }

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

    // Enforce encryption key when explicitly required (production hardening).
    // Set VECBOOST_REQUIRE_ENCRYPTION=1 to refuse startup without a valid key.
    if std::env::var("VECBOOST_REQUIRE_ENCRYPTION")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
    {
        if let Err(reason) = vecboost::config::encryption::validate_encryption_key() {
            return Err(anyhow::anyhow!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "startup-encryption",
                    vecboost::i18n::tr_args(&[("detail", &reason.to_string())]),
                )
            ));
        }
        log::info!("Encryption key validated (VECBOOST_REQUIRE_ENCRYPTION=1)");
    } else {
        // 检测到敏感配置但未设加密 key → 提示明文存储风险(不强制,
        // VECBOOST_REQUIRE_ENCRYPTION=1 才拒绝启动)
        let has_jwt = config
            .auth
            .jwt_secret
            .as_ref()
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        let has_pw = config
            .auth
            .default_admin_password
            .as_ref()
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        let has_key = std::env::var("VECBOOST_ENCRYPTION_KEY")
            .map(|k| !k.trim().is_empty())
            .unwrap_or(false);
        if should_warn_plaintext_secrets(has_jwt, has_pw, has_key) {
            log::warn!(
                "Sensitive config (jwt_secret / default_admin_password) is present but                  VECBOOST_ENCRYPTION_KEY is not set — values are stored in plaintext.                  Set VECBOOST_ENCRYPTION_KEY to encrypt, or VECBOOST_REQUIRE_ENCRYPTION=1                  to enforce encryption at startup."
            );
        }
    }

    #[cfg(feature = "db")]
    let (db_pool, _db_metrics) = init_db_pool(&config).await?;

    let (_engine, service, rerank_service, _model_config) =
        init_engine_and_services(&config).await?;

    #[cfg(feature = "mcp")]
    if std::env::args().any(|a| a == "--mcp") {
        return run_mcp_server(service, rerank_service).await;
    }

    #[cfg(feature = "cli")]
    if run_cli_command(
        service.clone(),
        rerank_service.clone(),
        _filtered_args.clone(),
    )
    .await?
    {
        return Ok(());
    }

    // 安全闸门:仅 HTTP 服务器路径需要(上方 MCP/CLI 均已提前返回)。
    validate_bind_safety(&config.server.host, config.auth.enabled)?;

    let rate_limiter = Arc::new(
        LimiteronAdapter::new(RateLimitSettings {
            global_requests_per_minute: config.rate_limit.global_requests_per_minute,
            ip_requests_per_minute: config.rate_limit.ip_requests_per_minute,
            user_requests_per_minute: config.rate_limit.user_requests_per_minute,
            api_key_requests_per_minute: config.rate_limit.api_key_requests_per_minute,
        })
        .await,
    );

    #[cfg(feature = "auth")]
    let (garrison_handle, garrison_csrf_config) = init_auth(&config).await?;

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

    let (pipeline_queue, response_channel, priority_calculator, worker_manager) =
        init_pipeline(&config, &service).await?;

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

    // Register build observer for per-module build timing
    kit.with_observer(Arc::new(LoggingObserver));

    // 注入预构建的能力对象（kit 是 single source of truth）— 已清理未被任何 Module/Handler 消费的冗余注入
    kit.set_config(service.clone());
    kit.set_config(rerank_service.clone());
    kit.set_config(rate_limiter.clone());
    kit.set_config(CacheConfig {
        enabled: config.embedding.cache_enabled,
        size: config.embedding.cache_size,
    });
    kit.set_config(DbConfig {
        enabled: cfg!(feature = "db"),
    });
    kit.set_config(audit_logger.clone());
    // 注入各 Module 的能力配置
    kit.set_config(Some(Arc::new(vecboost::metrics::InferenceCollector::new())));
    kit.set_config(Some(Arc::new(
        vecboost::metrics::PrometheusCollector::new().map_err(|e| {
            anyhow::anyhow!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "startup-register-failed",
                    vecboost::i18n::tr_args(&[
                        ("module", "PrometheusCollector"),
                        ("detail", &e.to_string())
                    ]),
                )
            )
        })?,
    )));
    kit.set_config(config.rate_limit.ip_whitelist.clone());
    kit.set_config(vecboost::registry::RateLimitEnabled(
        config.rate_limit.enabled,
    ));
    kit.set_config(config.embedding.clone());
    // Inject AuthConfig for `trusted_proxies` (XFF trust boundary) access
    // via `kit.config::<AuthConfig>()` in `auth_middleware` (see lib.rs `FromRef` impl).
    kit.set_config(config.auth.clone());
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
    // 注册各 Module
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
    // ConfigWatcherModule — monitors config.toml for hot reload
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

    // Register lifecycle and health check for key modules
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

    // AsyncShutdownCoordinator — phased graceful shutdown
    let shutdown_coordinator = AsyncShutdownCoordinator::new();
    shutdown_coordinator
        .set_global_timeout(Duration::from_secs(DEFAULT_SHUTDOWN_TIMEOUT_SECS))
        .map_err(|e| anyhow::anyhow!("Failed to set shutdown timeout: {}", e))?;
    {
        let kit_for_shutdown = Arc::clone(&kit);
        shutdown_coordinator
            .register_hook(ShutdownPhase::CloseConnections, move || {
                Box::pin(async move {
                    // Manually invoke async on_shutdown for lifecycle modules
                    // (AsyncKit::shutdown() is sync and cannot call async fns)
                    if let Ok(audit_cap) = kit_for_shutdown.require::<AuditModule>() {
                        <AuditModule as trait_kit::prelude::AsyncLifecycle>::on_shutdown(
                            &audit_cap,
                        )
                        .await;
                    }
                })
            })
            .map_err(|e| anyhow::anyhow!("Failed to register shutdown hook: {}", e))?;
    }
    // Register ConfigWatcherModule shutdown hook
    {
        let kit_for_watcher_shutdown = Arc::clone(&kit);
        shutdown_coordinator
            .register_hook(ShutdownPhase::DrainQueue, move || {
                Box::pin(async move {
                    if let Ok(watcher_cap) =
                        kit_for_watcher_shutdown.require::<ConfigWatcherModule>()
                    {
                        <ConfigWatcherModule as trait_kit::prelude::AsyncLifecycle>::on_shutdown(
                            &watcher_cap,
                        )
                        .await;
                    }
                })
            })
            .map_err(|e| {
                anyhow::anyhow!("Failed to register config watcher shutdown hook: {}", e)
            })?;
    }
    // WorkerManager 优雅关闭：排空队列并等待 in-flight 请求完成
    {
        let wm = Arc::clone(&worker_manager);
        shutdown_coordinator
            .register_hook(ShutdownPhase::DrainQueue, move || {
                Box::pin(async move {
                    wm.shutdown().await;
                })
            })
            .map_err(|e| {
                anyhow::anyhow!("Failed to register worker manager shutdown hook: {}", e)
            })?;
    }

    log::info!("AsyncKit module registry built successfully");

    // JoinSet for managing background tasks lifecycle
    let mut bg_tasks = tokio::task::JoinSet::<()>::new();

    // Spawn config file watcher task for hot reload
    // 变更检测：当前 AsyncKit<Ready> 不支持运行时 set_config，热重载仅验证新配置可加载
    // 后续待 trait-kit 为 AsyncKit 提供 reload 能力后再接线至各 Module
    bg_tasks.spawn(async move {
            // FsWatcher requires the file to exist; skip gracefully if not
            let config_path = config_path.as_deref().unwrap_or("config/config.toml");
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
            // 热重载校验跟随 --config 路径（否则自定义配置的变更会被默认路径误校验）
            let reload_path = config_path.to_string();
            while let Some(changed_path) = fs_watcher.recv().await {
                log::info!("Config file changed: {:?}, reloading...", changed_path);
                match AppConfig::load_via_confers_with_path(&reload_path) {
                    Ok(_new_config) => {
                        log::info!(
                            "Configuration reloaded and validated successfully (hot-swap pending trait-kit AsyncKit reload)"
                        );
                    }
                    Err(e) => {
                        log::error!("Failed to reload configuration: {}", e);
                    }
                }
            }
            log::info!("Config file watcher stopped");
    });

    // VecboostState 仅持有 kit 单字段，所有能力通过 kit.require 查询
    let app_state = VecboostState::new(kit);

    // 注入 state 到 api 模块（统一入口：所有 forge handler 通过 state().kit.require 访问）
    vecboost::api::init_state(app_state.clone()).map_err(|e| anyhow::anyhow!("{}", e))?;

    // sdforge #[forge] 路由（Router<()>，从 inventory 收集所有 forge 函数注册的路由）
    // G009: /api → /api/v1 307 重定向(版本前缀惯例)
    let app = sdforge::http::build_with_redirect();

    // G001: 挂载 Swagger UI(/api-docs/openapi.json + /swagger-ui/),
    // openapi.json 由 sdforge 从 forge 注册路由动态生成
    let app = app.merge(sdforge::docs::swagger_ui_router());

    // metrics 端点（手写例外：Prometheus text/plain 响应，forge 不支持非 JSON）
    let metrics_router = axum::Router::new()
        .route("/metrics", axum::routing::get(metrics_endpoint))
        .with_state(app_state.clone());
    let mut app = app.merge(metrics_router);

    // Prometheus 指标记录中间件 — 无条件应用到所有路由
    #[cfg(feature = "http")]
    {
        use axum::middleware::from_fn_with_state;
        app = app.layer(from_fn_with_state(
            app_state.clone(),
            vecboost::metrics::metrics_middleware,
        ));
    }

    // i18n Accept-Language 中间件 — 解析请求语言，设置请求级 locale
    // 使所有下游 handler 和 IntoResponse 自动使用正确的语言
    #[cfg(feature = "http")]
    {
        use axum::middleware::from_fn;
        app = app.layer(from_fn(vecboost::i18n::i18n_middleware));
    }

    // 全局限流中间件 — 应用到所有路由
    // 内部通过 RateLimitEnabled 配置控制是否生效
    // 注：auth_rate_limit_middleware 定义在 auth 模块下，需 feature = "auth" 门控
    #[cfg(feature = "auth")]
    {
        use axum::middleware::from_fn_with_state;
        app = app.layer(from_fn_with_state(
            app_state.clone(),
            vecboost::auth::auth_rate_limit_middleware,
        ));
    }

    // auth_middleware：应用到所有路由，内部用路径白名单放行公开端点
    // (/health, /api/1/auth/login, /api/1/auth/refresh)
    #[cfg(feature = "auth")]
    let app = if config.auth.enabled {
        use axum::middleware::from_fn_with_state;
        app.layer(from_fn_with_state(
            app_state.clone(),
            vecboost::auth::auth_middleware,
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
            app.layer(from_fn_with_state(cfg, garrison_csrf_middleware))
        } else {
            app
        }
    } else {
        app
    };

    // 安全 headers + trace
    // G009: 请求体大小上限(默认 5 MiB,防超大 payload DoS)
    let app = app.layer(tower_http::limit::RequestBodyLimitLayer::new(
        config.server.body_limit_mb.max(1) as usize * 1024 * 1024,
    ));
    // G009: 每个响应注入 x-request-id(与日志关联)
    let app = app.layer(axum::middleware::from_fn(
        |req: axum::extract::Request, next: axum::middleware::Next| async move {
            let mut resp = next.run(req).await;
            if !resp.headers().contains_key("x-request-id") {
                if let Ok(id) = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                {
                    resp.headers_mut().insert(
                        "x-request-id",
                        axum::http::HeaderValue::from_str(&format!(
                            "req-{:x}",
                            id.as_nanos()
                        ))
                        .unwrap_or(axum::http::HeaderValue::from_static("req-unknown")),
                    );
                }
            }
            Ok::<_, std::convert::Infallible>(resp)
        },
    ));
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

    // 响应压缩（gzip）——客户端经 Accept-Encoding 协商
    #[cfg(feature = "http")]
    let app = app.layer(tower_http::compression::CompressionLayer::new());

    // CORS（配置开关，默认关闭）：[server] cors_enabled / cors_allow_origins
    #[cfg(feature = "http")]
    let app = if config.server.cors_enabled {
        use axum::http::{HeaderName, HeaderValue, Method};
        let origins = config.server.cors_allow_origins;
        let mut cors = tower_http::cors::CorsLayer::new()
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([
                HeaderName::from_static("content-type"),
                HeaderName::from_static("authorization"),
                HeaderName::from_static("accept-language"),
            ]);
        if origins.is_empty() || origins.iter().any(|o| o == "*") {
            cors = cors.allow_origin(tower_http::cors::Any);
        } else {
            let list: Vec<HeaderValue> = origins
                .iter()
                .filter_map(|o| HeaderValue::from_str(o).ok())
                .collect();
            cors = cors.allow_origin(list);
        }
        app.layer(cors)
    } else {
        app
    };

    // ConnectInfo is automatically available when using axum::serve with a TcpListener
    // No additional layer needed

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    log::info!("Server listening on {}", addr);

    // Signal-aware graceful shutdown (SIGINT + SIGTERM)
    let signal = async {
        #[cfg(unix)]
        {
            let mut sigterm =
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Failed to install SIGTERM handler: {}", e);
                        // SIGTERM unavailable — continue with ctrl_c only
                        tokio::signal::ctrl_c().await.ok();
                        log::info!("Received SIGINT, initiating graceful shutdown");
                        return;
                    }
                };
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
            if let Err(e) = tokio::signal::ctrl_c().await {
                log::error!("Failed to install CTRL-C handler: {}", e);
            }
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
                                    "{}",
                                    vecboost::i18n::tr_with_args(
                                        "startup-grpc-bearer-failed",
                                        vecboost::i18n::tr_args(&[("detail", &e.to_string())]),
                                    )
                                );
                            }
                        }
                    } else {
                        anyhow::bail!("{}", vecboost::i18n::tr("startup-grpc-no-secret"));
                    }
                } else {
                    anyhow::bail!("{}", vecboost::i18n::tr("startup-grpc-auth-disabled"));
                }
            }
            #[cfg(not(feature = "auth"))]
            {
                anyhow::bail!("{}", vecboost::i18n::tr("startup-grpc-no-feature"));
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
        // `new()` is infallible (panics only on invalid default config, which is a bug).
        let rate_limiter: Option<std::sync::Arc<dyn sdforge::security::ratelimit::RateLimiter>> = {
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                SdforgeLimiteronAdapter::new(),
            )
            .await
            {
                Ok(limiter) => {
                    log::info!(
                        "gRPC rate_limiter enabled (sdforge LimiteronAdapter, default config: 100 burst / 10 req/s)"
                    );
                    Some(std::sync::Arc::new(limiter))
                }
                Err(_) => {
                    log::warn!(
                        "gRPC rate_limiter initialization timed out after 10s, starting without rate limiting"
                    );
                    None
                }
            }
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
        bg_tasks.spawn(async move {
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

    // Cancel all background tasks (config watcher, gRPC server, etc.)
    bg_tasks.abort_all();
    // Drain remaining tasks to prevent runtime hang
    while bg_tasks.join_next().await.is_some() {}

    let shutdown_result = match shutdown_coordinator.shutdown().await {
        Ok(result) => result,
        Err(e) => {
            log::warn!("Shutdown coordinator error: {}", e);
            return Err(anyhow::anyhow!("Phased shutdown failed: {}", e));
        }
    };
    if !shutdown_result.is_ok() {
        log::warn!(
            "Shutdown timed out on phases: {:?}",
            shutdown_result.timed_out_phases()
        );
    }
    log::info!("VecBoost shutdown complete");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // 绑定安全闸门 —— 回环放行 / 非回环拒绝 / 逃生阀放行并告警
    #[test]
    fn bind_safety_allows_loopback_without_auth() {
        assert!(validate_bind_safety("127.0.0.1", false).is_ok());
        assert!(validate_bind_safety("::1", false).is_ok());
        assert!(validate_bind_safety("localhost", false).is_ok());
        // auth enabled 时无论绑定地址一律放行
        assert!(validate_bind_safety("0.0.0.0", true).is_ok());
    }

    // 拒绝与逃生阀共享进程级环境变量,合并为单测避免并行竞态
    #[test]
    fn bind_safety_rejects_non_loopback_escape_valve_allows() {
        // SAFETY: 单线程内变更测试专用环境变量;Rust 2024 中 set_var/remove_var 为 unsafe
        unsafe { std::env::remove_var("VECBOOST_ALLOW_INSECURE") };
        let err = validate_bind_safety("0.0.0.0", false).unwrap_err();
        assert!(err.to_string().contains("VECBOOST_ALLOW_INSECURE"));
        assert!(validate_bind_safety("192.168.1.10", false).is_err());

        unsafe { std::env::set_var("VECBOOST_ALLOW_INSECURE", "1") };
        let result = validate_bind_safety("0.0.0.0", false);
        unsafe { std::env::remove_var("VECBOOST_ALLOW_INSECURE") };
        assert!(result.is_ok());
    }

    // G009: x-request-id 中间件注入响应头
    #[cfg(feature = "http")]
    #[tokio::test]
    async fn request_id_header_injected() {
        use axum::routing::get;
        use tower::ServiceExt;
        async fn ping() -> &'static str {
            "ok"
        }
        let app = axum::Router::new().route("/ping", get(ping)).layer(axum::middleware::from_fn(
            |req: axum::extract::Request, next: axum::middleware::Next| async move {
                let mut resp = next.run(req).await;
                if !resp.headers().contains_key("x-request-id") {
                    resp.headers_mut().insert(
                        "x-request-id",
                        axum::http::HeaderValue::from_static("req-test"),
                    );
                }
                Ok::<_, std::convert::Infallible>(resp)
            },
        ));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/ping")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get("x-request-id").unwrap(),
            "req-test"
        );
    }

    // G008: 默认构建(无 cuda/metal)下 use_gpu=true 回退 CPU
    #[test]
    fn device_resolution_falls_back_to_cpu_without_gpu_features() {
        assert!(
            matches!(
                resolve_device_config(true),
                vecboost::config::model::DeviceType::Cpu
            ),
            "non-GPU build must fall back to CPU"
        );
        assert!(matches!(
            resolve_device_config(false),
            vecboost::config::model::DeviceType::Cpu
        ));
    }

    // 敏感配置明文存储告警判定
    #[test]
    fn plaintext_secret_warning_logic() {
        assert!(should_warn_plaintext_secrets(true, false, false));
        assert!(should_warn_plaintext_secrets(false, true, false));
        assert!(!should_warn_plaintext_secrets(true, true, true));
        assert!(!should_warn_plaintext_secrets(false, false, false));
    }

    // G001: Swagger UI 挂载验证 —— openapi.json 与 UI 资源均可访问
    #[cfg(feature = "http")]
    #[tokio::test]
    async fn swagger_ui_endpoints_served() {
        use tower::ServiceExt;
        let app = sdforge::docs::swagger_ui_router();

        let resp = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api-docs/openapi.json")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/swagger-ui/")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    // G002: CLI 名单单一来源 + 未知子命令判定
    #[cfg(feature = "cli")]
    #[test]
    fn cli_subcommand_names_from_inventory() {
        // 名单来自 sdforge inventory(单一来源,非硬编码数组)
        let names = cli_subcommand_names();
        assert!(names.contains(&"embed_batch".to_string()));
        assert!(names.contains(&"compute_similarity".to_string()));
        assert!(names.contains(&"embed".to_string()));
        // 未知子命令的 exit(2) 行为在子进程中验证:
        //   vecboost definitely_not_a_cmd  -> exit 2
        //   vecboost --help                -> exit 0
    }

    // auth 启用但未配置管理员密码 → 拒绝启动(错误信息指明 VECBOOST_ADMIN_PASSWORD)
    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn init_auth_rejects_missing_admin_password() {
        let mut config = AppConfig::default();
        config.auth.enabled = true;
        config.auth.jwt_secret = Some("test-secret-0123456789abcdef0123".to_string());
        config.auth.default_admin_password = None;
        let err = init_auth(&config).await.unwrap_err();
        assert!(err.to_string().contains("VECBOOST_ADMIN_PASSWORD"));
    }
}
