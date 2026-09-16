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
#[cfg(not(feature = "auth"))]
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::trace::TraceLayer;
use trait_kit::prelude::{AsyncShutdownCoordinator, BuildObserver, ShutdownPhase};
use vecboost::AppConfig;
use vecboost::logger::LoggerModule;
use vecboost::registry::RateLimitModule;

/// 全局关闭超时（秒）
const DEFAULT_SHUTDOWN_TIMEOUT_SECS: u64 = 30;

/// runtime 退出时等待常驻 spawn_blocking 任务的兜底超时（秒）。
/// 常驻任务（inklog 定时器等）不可取消，默认 Drop 会无限等待导致进程挂死。
const SHUTDOWN_RUNTIME_DRAIN_SECS: u64 = 10;

/// 线程调优：解析优先级 显式配置 > 物理核检测 > num_cpus 回退。
/// `VECBOOST_NO_THREAD_TUNE=1` 时跳过检测直接回退。
fn resolve_runtime_threads(explicit: Option<usize>) -> (usize, Option<usize>, usize) {
    if std::env::var("VECBOOST_NO_THREAD_TUNE").as_deref() == Ok("1") {
        let fallback = num_cpus::get().max(1);
        return (
            explicit.filter(|&v| v > 0).unwrap_or(fallback),
            None,
            fallback,
        );
    }
    let detected = vecboost::thread_tune::detect_physical_cores();
    let fallback = num_cpus::get().max(1);
    let effective = vecboost::thread_tune::resolve_worker_threads(explicit, detected, fallback);
    (effective, detected, fallback)
}

/// 探测采集：RAM（sys-info）、物理核（thread_tune）、模型目录大小。
/// 任一探针失败即为 None，不参与规划（不猜测）；GPU 探测暂无 bin 可达路径，记 None。
fn gather_probes(config: &AppConfig) -> vecboost::planner::Probes {
    let avail_ram_mb = sys_info::mem_info().ok().map(|m| m.avail / 1024);
    let physical_cores = vecboost::thread_tune::detect_physical_cores();
    let logical_cores = Some(num_cpus::get());
    let model_resident_mb = config
        .model
        .model_path
        .as_deref()
        .map(model_dir_size_mb)
        .unwrap_or(None);
    vecboost::planner::Probes {
        avail_ram_mb,
        physical_cores,
        logical_cores,
        gpu_present: None,
        model_resident_mb,
    }
}

/// 估算模型目录常驻大小（MB）：递归累加，失败/缺失返回 None。
fn model_dir_size_mb(path: &str) -> Option<u64> {
    fn walk(dir: &std::path::Path, acc: &mut u64, depth: usize) {
        if depth > 4 {
            return;
        }
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    *acc = acc.saturating_add(meta.len());
                } else if meta.is_dir() {
                    walk(&p, acc, depth + 1);
                }
            }
        }
    }
    let mut bytes = 0u64;
    walk(std::path::Path::new(path), &mut bytes, 0);
    if bytes == 0 {
        None
    } else {
        Some(bytes / (1024 * 1024))
    }
}

/// NUMA 检测：多 socket 时返回建议文本，单 socket/解析失败返回 None。
fn numa_advice() -> Option<String> {
    let out = std::process::Command::new("lscpu").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let sockets = vecboost::thread_tune::parse_lscpu_sockets(&text)?;
    if sockets >= 2 {
        Some(format!(
            "NUMA detected: {} sockets; 建议使用 `numactl --interleave=all` 或 `--cpunodebind` 启动以均衡内存带宽（不做进程内绑定）",
            sockets
        ))
    } else {
        None
    }
}

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

// gRPC 服务器经 tonic Server 直接装配（garrison GarrisonGrpcAuthLayer server 级挂载，
// 服务本体复用 sdforge pub 类型）—— build_server_with_config 的 auth 槽位为
// BearerAuth 具体类型，无 layer 挂载点，不再使用
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
        quantized: config.model.quantized,
    };

    log::info!("Initializing Inference Engine (this may take a while to download models)...");
    let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(
        vecboost::engine::EngineFactory::create(EngineType::Candle, &model_config)?,
    ));

    let cache_enabled = config.embedding.cache_enabled;
    let cache_size = config.embedding.cache_size;

    // [embedding] persist_path 配置时启用 WAL 持久层并在启动时回放。
    const DEFAULT_PERSIST_MAX_BYTES: u64 = 1 << 30;
    let service = if cache_enabled && cache_size > 0 {
        log::info!("KV Cache enabled with size: {}", cache_size);
        match config.embedding.persist_path.as_deref() {
            Some(path) => {
                let max_bytes = config
                    .embedding
                    .persist_max_bytes
                    .unwrap_or(DEFAULT_PERSIST_MAX_BYTES);
                log::info!(
                    "Embedding cache WAL persistence enabled: {} (compact threshold {} bytes)",
                    path,
                    max_bytes
                );
                EmbeddingService::with_cache_persist(
                    engine.clone(),
                    Some(model_config.clone()),
                    cache_size,
                    std::path::PathBuf::from(path),
                    max_bytes,
                )
            }
            None => {
                EmbeddingService::with_cache(engine.clone(), Some(model_config.clone()), cache_size)
            }
        }
    } else {
        log::info!("KV Cache disabled");
        EmbeddingService::new(engine.clone(), Some(model_config.clone()))
    };

    // server 模式装配 ModelManager——LFRU 驻留配置 + heat warmstart。
    // 此前 service 的 model_manager 恒 None：switch/unload 在服务模式不可用，
    // [model] max_resident_models / resident_memory_budget_mb 无人消费。
    let model_manager = {
        let loader = Arc::new(vecboost::model_management::LocalModelLoader::new(
            std::path::PathBuf::from("models"),
        ));
        let manager = vecboost::model_management::ModelManager::with_loader(loader);
        match (
            config.model.max_resident_models,
            config.model.resident_memory_budget_mb,
        ) {
            (None, None) => manager,
            (max, budget) => {
                log::info!(
                    "model residency: max_resident_models={} budget_mb={:?}",
                    max.unwrap_or(usize::MAX),
                    budget
                );
                manager.with_residency(max.unwrap_or(usize::MAX), budget)
            }
        }
    };
    model_manager
        .load_heat_file(std::path::Path::new(
            vecboost::model_management::DEFAULT_HEAT_PATH,
        ))
        .await;
    let service = service.with_model_manager(Arc::new(model_manager));

    // server 模式注入语义缓存（[semantic_cache] enabled=true 时；
    // 默认 false 行为不变）。comparison_mode 非法值启动报错，不静默回退。
    let service = if config.semantic_cache.enabled {
        let mode: vecboost::ComparisonMode = config
            .semantic_cache
            .comparison_mode
            .parse()
            .map_err(|e: String| anyhow::anyhow!("[semantic_cache] comparison_mode 无效: {e}"))?;
        log::info!(
            "semantic cache enabled: threshold={} capacity={} mode={}",
            config.semantic_cache.similarity_threshold,
            config.semantic_cache.capacity,
            config.semantic_cache.comparison_mode
        );
        let semantic = Arc::new(
            vecboost::SemanticCache::with_capacity(
                config.semantic_cache.similarity_threshold,
                config.semantic_cache.capacity,
            )
            .with_comparison_mode(mode),
        );
        service.with_semantic_cache(semantic)
    } else {
        service
    };

    let service = Arc::new(RwLock::new(service));
    // 启动回放 WAL 重建缓存（未启用持久层时为 no-op）。
    service.read().await.load_persisted_cache().await;

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

/// CLI 子命令名单的单一来源 —— sdforge inventory 注册(forge CLI 宏),
/// 不再维护手工数组。docs 子命令由 sdforge 自动附加。
/// `doctor` 例外：在服务装配前短路运行（诊断必须能在模型损坏时工作），
/// 不走需要 EmbeddingService 的 inventory 分发路径。
#[cfg(feature = "cli")]
fn cli_subcommand_names() -> Vec<String> {
    let mut names: Vec<String> = CliBuilder::new()
        .with_name("vecboost")
        .build()
        .get_subcommands()
        .map(|sc| sc.get_name().to_string())
        .collect();
    names.push("doctor".to_string());
    names
}

/// 校验 CLI 首参数 —— `--help` 打印用法后退出;未知子命令 stderr 报错
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

        // 根因修复：HandlerFn 返回序列化后的 serde_json::Value，
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

        // garrison 0.9：init(dao, config, interface) → builder 链（build() 启动全局单例
        // 后台 task）；firewall-* feature 启用时 builder 自动注入防火墙检查钩子（CRIT-010）
        garrison::prelude::GarrisonManager::builder()
            .dao(Arc::new(dao))
            .config(Arc::new(garrison_config.clone()))
            .interface(Arc::new(VecBoostInterface::new(
                config
                    .auth
                    .default_admin_username
                    .clone()
                    .unwrap_or_else(|| "admin".to_string()),
            )))
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to init GarrisonManager: {}", e))?;

        // garrison metrics-prometheus：eager 创建 OnceLock 单例，把 garrison_* auth 域
        // 指标注册到 prometheus default_registry（/metrics 端点合并导出；
        // 惰性路径会漏掉首次记录前的零值序列）
        let _ = garrison::observability::GarrisonMetrics::new();

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

        let mut worker_config = vecboost::pipeline::WorkerConfig {
            min_workers: config.pipeline.worker.min_workers,
            max_workers: config.pipeline.worker.max_workers,
            scale_up_threshold: config.pipeline.worker.scale_up_threshold,
            scale_down_threshold: config.pipeline.worker.scale_down_threshold,
            scale_check_interval_secs: config.pipeline.worker.scale_check_interval_secs,
            idle_timeout_secs: config.pipeline.worker.idle_timeout_secs,
            max_batch_size: config.embedding.max_batch_size,
            batch_wait_ms: config.pipeline.worker.batch_wait_ms,
        };

        // [device] auto_plan 为 true 时，用硬件探测计划填充未显式配置字段。
        // 显式判定：server.workers（Option，Some 即显式）；其余字段以"与编译期
        // 默认不同"近似判定（显式设为默认值会被视为未显式，见注释与启动日志）。
        if config.device.auto_plan {
            let probes = gather_probes(config);
            let hw_plan = vecboost::planner::plan(&probes);
            log::info!(
                "auto_plan: 硬件规划 worker_threads={} max_batch_size={} batch_wait_ms={} \
                 quantized_recommended={} bottleneck={} rationale={}",
                hw_plan.worker_threads,
                hw_plan.max_batch_size,
                hw_plan.batch_wait_ms,
                hw_plan.quantized_recommended,
                hw_plan.bottleneck,
                hw_plan.rationale
            );
            let d = vecboost::pipeline::WorkerConfig::default();
            let explicit = vecboost::planner::PlanOverride {
                worker_threads: config.server.workers,
                max_batch_size: (config.embedding.max_batch_size != d.max_batch_size)
                    .then_some(config.embedding.max_batch_size),
                batch_wait_ms: (config.pipeline.worker.batch_wait_ms != d.batch_wait_ms)
                    .then_some(config.pipeline.worker.batch_wait_ms),
            };
            let mut wt = hw_plan.worker_threads; // 仅计划日志用（见下）
            let mut bs = worker_config.max_batch_size;
            let mut bw = worker_config.batch_wait_ms;
            vecboost::planner::apply_plan(&mut wt, &mut bs, &mut bw, &hw_plan, &explicit);
            // worker_threads 语义：plan 值为 tokio/rayon 级线程建议，此处仅日志；
            // 实际 runtime 线程数在 main() 启动时已按 确定（显式优先）。
            log::info!(
                "auto_plan: 应用结果 max_batch_size={} batch_wait_ms={} \
                 （计划 max_batch_size={} batch_wait_ms={} worker_threads 建议={}）",
                bs,
                bw,
                hw_plan.max_batch_size,
                hw_plan.batch_wait_ms,
                wt
            );
            worker_config.max_batch_size = bs;
            worker_config.batch_wait_ms = bw;
        }

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

/// 设备解析 —— 请求 GPU 但对应 feature 未编译时 WARN 并回退 CPU。
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
    // tokio runtime 线程数取物理核检测（显式 server.workers 优先，
    // VECBOOST_NO_THREAD_TUNE=1 回退 num_cpus）。配置在运行时前 best-effort
    // 预读，失败则由 app_main 内正式加载路径报错。
    let pre_explicit: Option<usize> = AppConfig::load_via_confers()
        .ok()
        .and_then(|c| c.server.workers)
        .or_else(|| {
            std::env::args()
                .collect::<Vec<_>>()
                .windows(2)
                .find(|w| w[0] == "--workers")
                .and_then(|w| w[1].parse().ok())
        });
    let (effective_threads, detected, fallback) = resolve_runtime_threads(pre_explicit);
    eprintln!(
        "[thread-tune] 物理核={} 逻辑核={} 生效线程数={}（显式配置={}）",
        detected
            .map(|v| v.to_string())
            .unwrap_or_else(|| "未知(回退)".to_string()),
        fallback,
        effective_threads,
        pre_explicit
            .map(|v| v.to_string())
            .unwrap_or_else(|| "未设置".to_string()),
    );
    if let Some(advice) = numa_advice() {
        eprintln!("[thread-tune] {}", advice);
    }
    // rayon 全局池与 tokio 对齐（仅初始化一次，失败则记录并继续）。
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(effective_threads)
        .build_global()
        .map_err(|e| eprintln!("[thread-tune] rayon 全局池已初始化，跳过：{}", e));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(effective_threads)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    let result = rt.block_on(app_main());
    // 修复：常驻 spawn_blocking 任务（inklog 定时器/写入线程等）
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
            } else {
                // fail-fast:`--config` 悬空（缺路径参数）不得静默回落默认配置启动,
                // 否则用户以为在改自定义配置、实际跑的是默认值。退出码与未知子命令一致。
                eprintln!("Error: --config requires a path argument (--config <path>)");
                std::process::exit(2);
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

/// 解析 `--warmup N` / `--warmup=N`。默认 0 = 不预热；
/// 缺参/非法值一律按 0（不猜）。
fn parse_warmup_count(args: &[String]) -> u32 {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--warmup" {
            return args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0);
        }
        if let Some(v) = args[i].strip_prefix("--warmup=") {
            return v.parse().unwrap_or(0);
        }
        i += 1;
    }
    0
}

#[cfg(test)]
mod warmup_arg_tests {
    use super::parse_warmup_count;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn warmup_flag_parsing() {
        assert_eq!(parse_warmup_count(&args(&[])), 0);
        assert_eq!(parse_warmup_count(&args(&["--warmup", "8"])), 8);
        assert_eq!(parse_warmup_count(&args(&["--warmup=4"])), 4);
        assert_eq!(parse_warmup_count(&args(&["--warmup"])), 0, "缺参按 0");
        assert_eq!(
            parse_warmup_count(&args(&["--warmup", "abc"])),
            0,
            "非法按 0"
        );
        assert_eq!(parse_warmup_count(&args(&["--warmup", "0"])), 0);
        assert_eq!(
            parse_warmup_count(&args(&["--config", "x.toml", "--warmup", "2"])),
            2,
            "与 --config 共存"
        );
    }
}

/// 应用主入口（在 tokio runtime 上执行）
async fn app_main() -> anyhow::Result<()> {
    // 剥离全局 --config 参数（CLI 子命令/clap 不识别该参数，须先行剥离）
    let (_filtered_args, config_path) = strip_config_args(std::env::args().collect());

    // 未知子命令/`--help` 在进入服务器装配前拦截(fail-fast,不静默起服务)
    #[cfg(feature = "cli")]
    validate_cli_invocation(&_filtered_args[1..]);

    // Early CLI detection: suppress console logging in CLI mode to keep stdout clean
    // for machine-readable JSON output
    #[cfg(feature = "cli")]
    let cli_mode = _filtered_args
        .get(1)
        .map(|a| cli_subcommand_names().contains(a))
        .unwrap_or(false);
    #[cfg(not(feature = "cli"))]
    let cli_mode = false;

    // 修复：MCP stdio 模式下 stdout 只能承载 JSON-RPC 协议消息，
    // inklog 控制台日志会污染协议流导致客户端解析失败 —— 与 CLI 模式同样关闭控制台输出。
    #[cfg(feature = "mcp")]
    let mcp_mode = std::env::args().any(|a| a == "--mcp");
    #[cfg(not(feature = "mcp"))]
    let mcp_mode = false;

    // i18n 先于配置初始化：配置校验错误消息需要翻译
    vecboost::i18n::init();

    // 配置先行加载：logger 的级别/文件参数来自 [logging] 配置段，
    // 因此配置必须在 logger 之前就绪；此时尚无日志后端，错误经 stderr 输出。
    let config = {
        // 显式指定的 --config 文件不存在 → 立即报错退出(码 2),
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

    // doctor 只读诊断 ——在 logger/引擎装配之前短路运行。
    // 诊断必须能在模型损坏、依赖缺失时工作，因此不走需要
    // EmbeddingService 的 run_cli_command 分发路径。
    #[cfg(feature = "cli")]
    if _filtered_args.get(1).map(|a| a.as_str()) == Some("doctor") {
        vecboost::doctor::DoctorReport::run(&config)
            .await
            .print_and_exit();
    }

    // [logging] 配置段接入 logger；
    // VECBOOST_LOG_LEVEL 环境变量优先级高于配置文件。
    // 白名单含 inklog 合法别名（warning 等）
    let log_level = std::env::var("VECBOOST_LOG_LEVEL")
        .ok()
        .filter(|l| {
            ["trace", "debug", "info", "warn", "warning", "error"]
                .contains(&l.to_lowercase().as_str())
        })
        .unwrap_or_else(|| config.logging.level.clone());

    // 绕过：inklog ConsoleSink 不消费 enabled 标志（上游缺陷，
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
        // 文件 sink 手工构建：以便按需以 SamplingSink 包装（[logging.sampling]）。
        // FileSinkConfig 字段与旧 builder.file/file_max_size/file_keep_files/
        // file_compress 链式调用一一对应。
        use inklog::FileSinkConfig;
        use inklog::sink::file::FileSink;
        use inklog::sink::sampling::{Sampler, SamplingSink};

        let file_config = FileSinkConfig {
            enabled: true,
            path: config.logging.file_path.clone().into(),
            max_size: format!("{}MB", config.logging.rotation_size_mb),
            keep_files: config.logging.max_files,
            compress: true,
            ..Default::default()
        };
        let file_sink: Arc<dyn inklog::sink::AsyncSink> = if config.logging.sampling.enabled {
            let sampler = Sampler::new(
                &config.logging.sampling.min_level,
                config.logging.sampling.sample_every_n,
                config.logging.sampling.keyword_whitelist.clone(),
            )
            .map_err(|e| anyhow::anyhow!("Invalid [logging.sampling] config: {}", e))?;
            let inner = FileSink::new(file_config)
                .map_err(|e| anyhow::anyhow!("File log sink build failed: {}", e))?;
            Arc::new(SamplingSink::new(Arc::new(inner), Arc::new(sampler)))
        } else {
            Arc::new(
                FileSink::new(file_config)
                    .map_err(|e| anyhow::anyhow!("File log sink build failed: {}", e))?,
            )
        };
        logger_builder = logger_builder.add_sink(file_sink);
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
    // 供 /health?depth=full 的 DB 就绪探测
    #[cfg(feature = "db")]
    let _ = vecboost::db::register_global_pool(std::sync::Arc::new(db_pool.clone()));

    let (_engine, service, rerank_service, _model_config) =
        init_engine_and_services(&config).await?;

    // `--warmup N` 启动预热 ——N 条合成短文本推理，预热 mkl/代码路径/
    // tokenizer 缓存。放在 MCP/CLI 分流之前，三种模式均受益。
    let warmup = parse_warmup_count(&_filtered_args);
    if warmup > 0 {
        let start = std::time::Instant::now();
        for i in 0..warmup {
            let text = format!(
                "vecboost warmup sentence {i}: the quick brown fox jumps over the lazy dog"
            );
            let guard = _engine.read().await;
            let _ = vecboost::engine::InferenceEngine::embed(&*guard, &text);
        }
        log::info!("warmup: {warmup} 次预热推理完成（{:?}）", start.elapsed());
    }

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

    let kit = build_module_registry(
        &config,
        service.clone(),
        rerank_service.clone(),
        rate_limiter.clone(),
        audit_logger.clone(),
        pipeline_queue.clone(),
        response_channel.clone(),
        priority_calculator.clone(),
        worker_manager.clone(),
        logger_manager.clone(),
        #[cfg(feature = "auth")]
        garrison_handle.clone(),
        #[cfg(feature = "auth")]
        garrison_csrf_config.clone(),
    )
    .await?;

    let shutdown_coordinator = AsyncShutdownCoordinator::new();
    shutdown_coordinator
        .set_global_timeout(Duration::from_secs(DEFAULT_SHUTDOWN_TIMEOUT_SECS))
        .map_err(|e| anyhow::anyhow!("Failed to set shutdown timeout: {}", e))?;
    register_shutdown_hooks(&shutdown_coordinator, &kit, &worker_manager)?;

    log::info!("AsyncKit module registry built successfully");

    // JoinSet for managing background tasks lifecycle
    let mut bg_tasks = tokio::task::JoinSet::<()>::new();

    spawn_config_watcher(&mut bg_tasks, config_path);

    // VecboostState 仅持有 kit 单字段，所有能力通过 kit.require 查询
    let app_state = VecboostState::new(kit);

    // 注入 state 到 api 模块（统一入口：所有 forge handler 通过 state().kit.require 访问）
    vecboost::api::init_state(app_state.clone()).map_err(|e| anyhow::anyhow!("{}", e))?;

    // sdforge #[forge] 路由（Router<()>，从 inventory 收集所有 forge 函数注册的路由）
    /// HTTP 路由装配(sdforge 路由 + Swagger/metrics/中间件/CORS)。
    async fn build_http_router(config: &AppConfig) -> anyhow::Result<axum::Router> {
        // app_state 由调用方保证已 init_state(所有能力经 kit.require 获取)
        let app_state = vecboost::api::state().map_err(|e| anyhow::anyhow!("{e}"))?;
        // 版本前缀:路由注册为 /api/1/*(forge version=1)。
        // 实施期决策:不启用 build_with_redirect —— 它会把 /api/1/* 重定向到
        // 不存在的 /api/v1/*,破坏全部 API 路由(实测 301)。
        let app = sdforge::http::build();

        // 挂载 Swagger UI(/api-docs/openapi.json + /swagger-ui/),
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

        // Extractor 拒绝规范化（R-1 审计建议）— axum Json extractor 拒绝的
        // text/plain 错误体改写为 handler 层同构的结构化错误 JSON
        #[cfg(feature = "http")]
        {
            use axum::middleware::from_fn;
            app = app.layer(from_fn(
                vecboost::api::rejection_normalize::rejection_normalizer,
            ));
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
        // 请求体大小上限(默认 5 MiB,防超大 payload DoS)
        let app = app.layer(tower_http::limit::RequestBodyLimitLayer::new(
            config.server.body_limit_mb.max(1) as usize * 1024 * 1024,
        ));
        // HTTP 全局请求超时(默认 60s,配置 server.request_timeout_seconds)
        let app = app.layer(tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            std::time::Duration::from_secs(config.server.request_timeout_seconds.max(1)),
        ));
        // 请求上下文（sdforge context 吸收）：入口解析/生成 X-Request-Id 与
        // X-Trace-Id（兼容 W3C traceparent），task-local 贯穿请求链，响应回显
        // 双头；替代原先手写的 x-request-id 响应头闭包。此处包一层观测：
        // debug 级请求生命周期日志携带双 id + 耗时，实现日志按请求关联。
        let app = app.layer(axum::middleware::from_fn(
            |req: axum::extract::Request, next: axum::middleware::Next| async move {
                let started = std::time::Instant::now();
                let method = req.method().clone();
                let path = req.uri().path().to_string();
                let resp = sdforge::context::context_middleware(req, next).await;
                let (request_id, trace_id) = sdforge::context::current()
                    .map(|c| (c.request_id().to_string(), c.trace_id().to_string()))
                    .unwrap_or_else(|| ("-".to_string(), "-".to_string()));
                log::debug!(
                    "http_request method={method} path={path} status={} duration_ms={} request_id={request_id} trace_id={trace_id}",
                    resp.status().as_u16(),
                    started.elapsed().as_millis()
                );
                resp
            },
        ));
        let app = app.layer(TraceLayer::new_for_http());
        // 安全响应头（garrison 0.9 web-security-headers 吸收）：auth 构建改用
        // garrison::web::security_headers 中间件，替代原 4 个手写 SetResponseHeaderLayer，
        // 并新增 Cache-Control: no-store / Pragma: no-cache（防敏感认证响应被缓存）；
        // HSTS 仅 garrison tls feature 下注入（明文 HTTP 注入无意义且可被降级利用，
        // 原 stack 无条件注入系过度设置）。非 auth 构建无 garrison 依赖，保留手写头栈。
        #[cfg(feature = "auth")]
        let app = app.layer(axum::middleware::from_fn(
            garrison::web::security_headers::security_headers_middleware,
        ));
        #[cfg(not(feature = "auth"))]
        let app = app
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
            let origins = config.server.cors_allow_origins.clone();
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
        Ok(app)
    }

    let app = build_http_router(&config).await?;

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
        spawn_grpc_server(&config, &mut bg_tasks).await?;
    }

    run_server_lifecycle(
        listener,
        app,
        signal,
        &config,
        bg_tasks,
        shutdown_coordinator,
    )
    .await?;

    Ok(())
}

/// 服务器生命周期 —— 阻塞等待退出信号,执行分级优雅关闭。
#[cfg(feature = "http")]
async fn run_server_lifecycle(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    signal: impl std::future::Future<Output = ()> + Send + 'static,
    _config: &AppConfig,
    mut bg_tasks: tokio::task::JoinSet<()>,
    shutdown_coordinator: AsyncShutdownCoordinator,
) -> anyhow::Result<()> {
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(signal)
    .await?;

    // Execute phased shutdown coordinator after server stops
    log::info!("Server stopped, executing phased shutdown...");

    // gRPC drain 窗口 —— HTTP 已停止;给在途 gRPC 调用一个有界完成窗口
    // 再 abort。sdforge server API 无 shutdown 注入口,无法做 tonic 级
    // graceful drain(复制鉴权拦截器有安全漂移风险,实施期决策)。
    #[cfg(feature = "grpc")]
    if _config.server.grpc_enabled {
        log::info!("gRPC drain window: waiting up to 30s for in-flight calls");
        let deadline = tokio::time::sleep(std::time::Duration::from_secs(
            _config.server.grpc_timeout_seconds.unwrap_or(5).min(30),
        ));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = &mut deadline => break,
                _ = bg_tasks.join_next(), if !bg_tasks.is_empty() => {}
                else => break,
            }
        }
    }

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

/// 注册分级优雅关闭钩子(CloseConnections/DrainQueue 各阶段)。
fn register_shutdown_hooks(
    shutdown_coordinator: &AsyncShutdownCoordinator,
    kit: &Arc<trait_kit::AsyncKit<trait_kit::AsyncReady>>,
    worker_manager: &Arc<WorkerManager>,
) -> anyhow::Result<()> {
    {
        let kit_for_shutdown = Arc::clone(kit);
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
        let kit_for_watcher_shutdown = Arc::clone(kit);
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
        let wm = Arc::clone(worker_manager);
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

    Ok(())
}

/// 装配并启动 gRPC 服务器(BearerAuth/限流/连接上限/超时)。
#[cfg(feature = "grpc")]
async fn spawn_grpc_server(
    config: &AppConfig,
    bg_tasks: &mut tokio::task::JoinSet<()>,
) -> anyhow::Result<()> {
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

    // garrison 0.9 吸收：gRPC 鉴权由 GarrisonGrpcAuthLayer 承担 —— check_login 全异步
    // 会话校验（过期/吊销/防火墙），与 HTTP 侧共用同一 GarrisonDaoOxcache 会话宇宙。
    // 替代 sdforge BearerAuth（无状态 JWT 校验 + 独立内存黑名单：HTTP revoke_token
    // 吊销的 token 此前在 gRPC 侧 exp 前仍有效，双协议鉴权语义分裂）。
    // vuln-0006 fail-fast 语义保持：require_auth=true 时 auth 未启用/无 secret 拒绝启动。
    if require_auth {
        #[cfg(feature = "auth")]
        {
            if !config.auth.enabled {
                anyhow::bail!("{}", vecboost::i18n::tr("startup-grpc-auth-disabled"));
            }
            if config.auth.jwt_secret.is_none() {
                anyhow::bail!("{}", vecboost::i18n::tr("startup-grpc-no-secret"));
            }
            log::info!(
                "gRPC garrison auth layer enabled (full session validation, \
                 revocation shared with HTTP)"
            );
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
    }

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

    // 服务装配与 build_server_with_config 内部一致（state=None + rate_limiter +
    // 4MiB 解码上限），复用 sdforge pub 类型自行装配 —— GrpcServerConfig 的 auth
    // 槽位是 BearerAuth 具体类型、无 layer 挂载点，garrison 鉴权层在 server 级挂载
    //（GarrisonGrpcAuthLayer 未实现 NamedService，无法 per-service 包裹）。
    let service =
        sdforge::grpc::SdForgeGrpcService::with_state_and_rate_limiter(None, rate_limiter);
    let forge_service =
        sdforge::grpc::sdforge_v1::sd_forge_service_server::SdForgeServiceServer::new(service)
            .max_decoding_message_size(4 * 1024 * 1024);

    let max_connections = config.server.grpc_max_connections.unwrap_or(1000);
    let timeout_seconds = config.server.grpc_timeout_seconds.unwrap_or(30);

    log::info!("gRPC server enabled on {}", grpc_addr);
    bg_tasks.spawn(async move {
        // 地址校验沿用 build_server_with_config 的安全修复（先解析后绑定，错误仅记日志）
        let addr: std::net::SocketAddr = match grpc_addr.parse() {
            Ok(a) => a,
            Err(e) => {
                log::error!("Invalid gRPC server address format: {}", e);
                return;
            }
        };
        // layer() 改变 Server 泛型类型，鉴权/非鉴权两条 builder 链无法共用变量，
        // 按 require_auth 分支装配（true 时上方 fail-fast 已保证 garrison 就绪）
        let served = if require_auth {
            #[cfg(feature = "auth")]
            {
                let mut builder = tonic::transport::Server::builder()
                    .layer(garrison::grpc::GarrisonGrpcAuthLayer);
                if max_connections > 0 {
                    builder = builder.concurrency_limit_per_connection(max_connections);
                }
                if timeout_seconds > 0 {
                    builder = builder.timeout(std::time::Duration::from_secs(timeout_seconds));
                }
                builder.add_service(forge_service).serve(addr).await
            }
            #[cfg(not(feature = "auth"))]
            {
                drop(forge_service);
                Ok::<(), tonic::transport::Error>(())
            }
        } else {
            let mut builder = tonic::transport::Server::builder();
            if max_connections > 0 {
                builder = builder.concurrency_limit_per_connection(max_connections);
            }
            if timeout_seconds > 0 {
                builder = builder.timeout(std::time::Duration::from_secs(timeout_seconds));
            }
            builder.add_service(forge_service).serve(addr).await
        };
        if let Err(e) = served {
            log::error!("gRPC server error: {}", e);
        }
    });

    Ok(())
}

/// 启动配置文件监视任务(热重载校验;变更后重启生效,见 README)。
fn spawn_config_watcher(bg_tasks: &mut tokio::task::JoinSet<()>, config_path: Option<String>) {
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
}

/// 构建模块注册中心(trait-kit AsyncKit)—— 预构建能力注入 + 17 个
/// Module 注册 + 生命周期/健康检查挂载。从 `app_main` 拆出。
#[allow(clippy::too_many_arguments)]
async fn build_module_registry(
    config: &AppConfig,
    service: Arc<RwLock<EmbeddingService>>,
    rerank_service: Arc<RwLock<RerankService>>,
    rate_limiter: Arc<LimiteronAdapter>,
    audit_logger: Option<Arc<AuditLogger>>,
    pipeline_queue: Arc<PriorityRequestQueue>,
    response_channel: Arc<ResponseChannel>,
    priority_calculator: Arc<PriorityCalculator>,
    worker_manager: Arc<WorkerManager>,
    logger_manager: Arc<inklog::LoggerManager>,
    #[cfg(feature = "auth")] garrison_handle: Option<Arc<vecboost::auth::GarrisonHandle>>,
    #[cfg(feature = "auth")] garrison_csrf_config: Option<Arc<vecboost::auth::GarrisonCsrfConfig>>,
) -> anyhow::Result<Arc<trait_kit::AsyncKit<trait_kit::AsyncReady>>> {
    // ---------------------------------------------------------------------------
    // Module Registry (trait-kit AsyncKit)
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
    let prometheus_collector =
        Arc::new(vecboost::metrics::PrometheusCollector::new().map_err(|e| {
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
        })?);
    // 全局桥 ——worker/服务热路径拿不到 kit 状态，经 OnceLock 取指标写入口。
    let _ =
        vecboost::metrics::prometheus_exporter::set_global_collector(prometheus_collector.clone());
    kit.set_config(Some(prometheus_collector));
    kit.set_config(config.rate_limit.ip_whitelist.clone());
    kit.set_config(vecboost::registry::RateLimitEnabled(
        config.rate_limit.enabled,
    ));
    kit.set_config(vecboost::registry::RateLimitHeadersEnabled(
        config.rate_limit.headers_enabled,
    ));
    kit.set_config(config.embedding.clone());
    // ServerConfig 注入：/embed/file 的 PathValidator 与 model_path_validator
    // 经 kit.config::<ServerConfig>() 读取 grpc_allowed_roots；缺失时前者
    // 422(CONFIG_HINT)、后者回落相对根 "models"(拒绝绝对路径 → 500)
    kit.set_config(config.server.clone());
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
    Ok(Arc::new(kit))
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

    // 请求上下文中间件（sdforge context 吸收）：回显 X-Request-Id/X-Trace-Id，
    // 且入站 X-Request-Id 优先（客户端关联）而非覆盖
    #[cfg(feature = "http")]
    #[tokio::test]
    async fn request_id_header_injected() {
        use axum::routing::get;
        use tower::ServiceExt;
        async fn ping() -> &'static str {
            "ok"
        }
        let app = axum::Router::new()
            .route("/ping", get(ping))
            .layer(axum::middleware::from_fn(
                sdforge::context::context_middleware,
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
        let request_id = resp
            .headers()
            .get("x-request-id")
            .expect("响应应携带 X-Request-Id")
            .to_str()
            .unwrap();
        assert!(
            request_id.starts_with("req-"),
            "生成 id 应带 req- 前缀: {request_id}"
        );
        assert!(
            resp.headers().get("x-trace-id").is_some(),
            "响应应携带 X-Trace-Id"
        );

        // 入站 X-Request-Id 应被保留（客户端关联语义）
        let app2 = axum::Router::new()
            .route("/ping", get(ping))
            .layer(axum::middleware::from_fn(
                sdforge::context::context_middleware,
            ));
        let resp2 = app2
            .oneshot(
                axum::http::Request::builder()
                    .uri("/ping")
                    .header("x-request-id", "my-client-id-42")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp2.headers().get("x-request-id").unwrap(),
            "my-client-id-42"
        );
    }

    // 默认构建(无 cuda/metal)下 use_gpu=true 回退 CPU
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

    // Swagger UI 挂载验证 —— openapi.json 与 UI 资源均可访问
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

    // CLI 名单单一来源 + 未知子命令判定
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
