// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

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
use vecboost::VecboostConfig;
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
fn gather_probes(config: &VecboostConfig) -> vecboost::planner::Probes {
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
        Some(vecboost::i18n::tr_with_args(
            "numa-advice",
            vecboost::i18n::tr_args(&[("sockets", &sockets.to_string())]),
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
    config::model::{CheckpointPreset, EngineType, ModelConfig, ModelTask},
    engine::AnyEngine,
    pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    },
    rate_limit::{LimiteronAdapter, RateLimitSettings},
    registry::{
        AuditModule, CacheConfig, CacheModule, ConfigWatcherModule, DbConfig, DbModule,
        DecisionModule, DetectModule, EmbeddingModule, IpWhitelistModule, MetricsCollectorModule,
        PipelineQueueModule, PriorityCalculatorModule, PrometheusCollectorModule, RerankModule,
        ResponseChannelModule, TranscribeModule, TranslateModule, WorkerManagerModule,
    },
    service::{
        decision::DecisionService, detect::DetectService, embedding::EmbeddingService,
        rerank::RerankService, transcribe::TranscribeService, translate::TranslateService,
    },
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
        log::info!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-building-module",
                vecboost::i18n::tr_args(&[("module", module_name)])
            )
        );
    }

    fn on_module_built(&self, module_name: &'static str, elapsed: Duration) {
        if elapsed.as_secs_f64() > 1.0 {
            log::warn!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "log-main-module-built-slow",
                    vecboost::i18n::tr_args(&[
                        ("module", module_name),
                        ("seconds", &format!("{:.2}", elapsed.as_secs_f64()))
                    ])
                )
            );
        } else {
            log::info!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "log-main-module-built",
                    vecboost::i18n::tr_args(&[
                        ("module", module_name),
                        ("ms", &format!("{:.2}", elapsed.as_secs_f64() * 1000.0))
                    ])
                )
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions — extracted from main() to reduce cyclomatic complexity
// ---------------------------------------------------------------------------

#[cfg(feature = "db")]
async fn init_db_pool(
    config: &VecboostConfig,
) -> anyhow::Result<(DbPool, Arc<dbnexus::MetricsCollector>)> {
    log::info!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-db-pool-init",
            vecboost::i18n::tr_args(&[(
                "url",
                &config
                    .database
                    .url
                    .rsplit_once('@')
                    .map(|(prefix, host)| {
                        let scheme = prefix.split("://").next().unwrap_or("db");
                        format!("{}://***@{}", scheme, host)
                    })
                    .unwrap_or_else(|| config.database.url.clone())
            )])
        )
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
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-db-retry",
            vecboost::i18n::tr_args(&[("retries", &DB_MAX_RETRIES.to_string())])
        )
    );
    db_config.pool_config.min_connections = DB_MIN_CONNECTIONS;
    db_config.warmup_timeout = DB_WARMUP_TIMEOUT_SECS;
    db_config.warmup_retries = DB_WARMUP_RETRIES;
    log::info!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-db-warmup",
            vecboost::i18n::tr_args(&[
                ("min", &db_config.pool_config.min_connections.to_string()),
                ("timeout", &db_config.warmup_timeout.to_string()),
                ("retries", &db_config.warmup_retries.to_string())
            ])
        )
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
    log::info!("{}", vecboost::i18n::tr("log-main-db-ready"));
    let db_metrics = Arc::new(dbnexus::MetricsCollector::new());
    log::info!("{}", vecboost::i18n::tr("log-main-db-metrics"));
    Ok((pool, db_metrics))
}

/// 引擎级 `ModelConfig` 装配（装配三问①：`[model]` 配置面 → 运行时
/// ModelConfig）。onnx-primary：`[model].engine_type` 未配置时默认 onnx
/// （主推理引擎 ort）；`"candle"` 需 feature `candle` 显式启用，未编译时
/// 启动期显性拒绝（禁止静默降级 onnx）。未知值启动期显性报错；
/// `[model].tokenizer_path` 仅用户显式配置时注入，路径缺失在引擎加载期
/// 显性报错，不代赋。
fn resolve_engine_model_config(config: &VecboostConfig) -> anyhow::Result<ModelConfig> {
    let engine_type = match config.model.engine_type.as_deref() {
        // 未配置 → 默认主引擎 onnx；仅 onnx feature 缺席的定制构建
        // （--no-default-features）才落 candle 变体（该值在工厂加载期
        // 显性拒绝，行为与显式配置 candle 一致）
        None => {
            #[cfg(feature = "onnx")]
            {
                EngineType::Onnx
            }
            #[cfg(not(feature = "onnx"))]
            {
                EngineType::Candle
            }
        }
        Some("candle") => {
            // candle 回退后端 opt-in：未编译时启动期显性拒绝（禁止静默降级
            // onnx——engine_type 是用户显式选择）
            #[cfg(feature = "candle")]
            {
                EngineType::Candle
            }
            #[cfg(not(feature = "candle"))]
            {
                anyhow::bail!(
                    "{}",
                    vecboost::i18n::tr("startup-engine-candle-not-compiled")
                );
            }
        }
        #[cfg(feature = "onnx")]
        Some("onnx") => EngineType::Onnx,
        Some(other) => {
            let hint = if other == "onnx" && cfg!(not(feature = "onnx")) {
                vecboost::i18n::tr("startup-engine-onnx-rebuild-hint")
            } else {
                String::new()
            };
            anyhow::bail!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "startup-engine-type-unknown",
                    vecboost::i18n::tr_args(&[("got", other), ("hint", &hint)]),
                )
            );
        }
    };
    let model_path = match &config.model.model_path {
        Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::PathBuf::from(&config.model.model_repo),
    };
    // tokenizer_path 消费者：onnx 本地 bundle 分支（按 engine_type）+ 决策
    // 管线（task=decision 时工厂按 engine_type 分派决策后端——candle 原生
    // 决策头与 onnx DecisionPipeline 两路均消费）；candle embedding 路径与
    // onnx HF 拉取分支均无消费者，配置值会静默 no-op——沿用 use_gpu 无
    // feature 回退告警惯例，启动期显性提示。本地判定与 onnx_engine
    // with_device 同口径（exists() && is_dir()）。
    if config.model.tokenizer_path.is_some()
        && tokenizer_path_is_noop(&engine_type, &config.model.task, &model_path)
    {
        log::warn!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-tokenizer-noop",
                vecboost::i18n::tr_args(&[
                    ("engine", &engine_type.to_string()),
                    ("task", &config.model.task.to_string())
                ])
            )
        );
    }
    Ok(ModelConfig {
        name: config.model.model_repo.clone(),
        engine_type,
        model_path,
        tokenizer_path: config
            .model
            .tokenizer_path
            .as_deref()
            .map(std::path::PathBuf::from),
        device: resolve_device_config(config.model.use_gpu),
        max_batch_size: config.model.batch_size,
        pooling_mode: None,
        expected_dimension: config.model.expected_dimension,
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
        task: config.model.task,
        quantized: config.model.quantized,
        decision_params: None,
        translation_params: None,
        transcription_params: None,
    })
}

/// `[model].tokenizer_path` 在该引擎/任务/模型路径组合下是否静默 no-op。
/// 两个真实消费者：onnx 本地 bundle 分支（`resolve_local_bundle` 的
/// explicit 路径，按 engine_type）与决策管线（工厂 task=decision 分派臂
/// 按 engine_type 选后端——candle 原生决策头与 onnx `DecisionPipeline`
/// 两路的 explicit 路径均消费）；本地判定与 `onnx_engine` 的
/// `with_device` 同口径（`exists() && is_dir()`）。其余组合（candle
/// embedding、onnx HF 拉取分支、任何非本地路径）不读该字段。
/// `[model.checkpoints]` 启动 fail-fast（沿用既有装配口径：配置错误拒绝
/// 启动而非静默忽略，对齐 [semantic_cache].comparison_mode 校验位）：
/// 合并内置 `laya-multilingual` 预设后逐条目校验 + TOML 键与条目 name
/// 一致性校验（内置条目同样过闸，代码常量漂移由测试钉兜住）。返回表供
/// kit.set_config 注入与 EmbeddingService switch 查表共用（装配点单一事实源）。
fn resolve_checkpoints_fail_fast(
    config: &VecboostConfig,
) -> anyhow::Result<std::collections::BTreeMap<String, CheckpointPreset>> {
    let resolved =
        vecboost::config::model::merge_builtin_checkpoints(config.model.checkpoints.clone())
            .map_err(|e| anyhow::anyhow!("[model.checkpoints] {}", e.error_detail()))?;
    vecboost::config::model::validate_checkpoint_map(&resolved)
        .map_err(|e| anyhow::anyhow!("[model.checkpoints] {}", e.error_detail()))?;
    Ok(resolved)
}

fn tokenizer_path_is_noop(
    engine_type: &EngineType,
    task: &ModelTask,
    model_path: &std::path::Path,
) -> bool {
    if *task == ModelTask::Decision {
        return !(model_path.exists() && model_path.is_dir());
    }
    match engine_type {
        EngineType::Candle => true,
        #[cfg(feature = "onnx")]
        EngineType::Onnx => !(model_path.exists() && model_path.is_dir()),
    }
}

/// [embedding] 缓存配置 → EmbeddingService 构建分流：persist_path 配置时
/// 启用 WAL 持久层（compact 阈值缺省 1 GiB），cache 关闭时走无缓存构造。
async fn build_embedding_service(
    engine: &Arc<RwLock<AnyEngine>>,
    config: &VecboostConfig,
    model_config: Option<ModelConfig>,
) -> anyhow::Result<EmbeddingService> {
    let cache_enabled = config.embedding.cache_enabled;
    let cache_size = config.embedding.cache_size;

    const DEFAULT_PERSIST_MAX_BYTES: u64 = 1 << 30;
    if cache_enabled && cache_size > 0 {
        log::info!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-kvcache-enabled",
                vecboost::i18n::tr_args(&[("size", &cache_size.to_string())])
            )
        );
        if config.embedding.cache_backend == vecboost::config::app::CacheBackendKind::Redis {
            #[cfg(feature = "distributed-redis")]
            {
                let url = config
                    .embedding
                    .redis_url
                    .as_deref()
                    .filter(|url| !url.trim().is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!(vecboost::i18n::tr("startup-cache-redis-url-missing"))
                    })?;
                return EmbeddingService::with_cache_redis(
                    engine.clone(),
                    model_config,
                    cache_size,
                    url,
                )
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()));
            }
            #[cfg(not(feature = "distributed-redis"))]
            {
                return Err(anyhow::anyhow!(vecboost::i18n::tr(
                    "startup-cache-redis-feature-missing"
                )));
            }
        }
        match config.embedding.persist_path.as_deref() {
            Some(path) => {
                let max_bytes = config
                    .embedding
                    .persist_max_bytes
                    .unwrap_or(DEFAULT_PERSIST_MAX_BYTES);
                log::info!(
                    "{}",
                    vecboost::i18n::tr_with_args(
                        "log-main-kvcache-wal",
                        vecboost::i18n::tr_args(&[
                            ("path", path),
                            ("bytes", &max_bytes.to_string())
                        ])
                    )
                );
                Ok(EmbeddingService::with_cache_persist(
                    engine.clone(),
                    model_config,
                    cache_size,
                    std::path::PathBuf::from(path),
                    max_bytes,
                ))
            }
            None => Ok(EmbeddingService::with_cache(
                engine.clone(),
                model_config,
                cache_size,
            )),
        }
    } else {
        log::info!("{}", vecboost::i18n::tr("log-main-kvcache-disabled"));
        Ok(EmbeddingService::new(engine.clone(), model_config))
    }
}

/// server 模式装配 ModelManager——LFRU 驻留配置 + heat warmstart。
/// 此前 service 的 model_manager 恒 None：switch/unload 在服务模式不可用，
/// [model] max_resident_models / resident_memory_budget_mb 无人消费。
async fn attach_model_manager(
    service: EmbeddingService,
    config: &VecboostConfig,
) -> EmbeddingService {
    let loader = Arc::new(vecboost::model_management::LocalModelLoader::new(
        std::path::PathBuf::from("models"),
    ));
    let manager = vecboost::model_management::ModelManager::with_loader(loader);
    let manager = match (
        config.model.max_resident_models,
        config.model.resident_memory_budget_mb,
    ) {
        (None, None) => manager,
        (max, budget) => {
            log::info!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "log-main-residency",
                    vecboost::i18n::tr_args(&[
                        ("max", &max.unwrap_or(usize::MAX).to_string()),
                        ("budget", &format!("{:?}", budget))
                    ])
                )
            );
            manager.with_residency(max.unwrap_or(usize::MAX), budget)
        }
    };
    manager
        .load_heat_file(std::path::Path::new(
            vecboost::model_management::DEFAULT_HEAT_PATH,
        ))
        .await;
    service.with_model_manager(Arc::new(manager))
}

/// server 模式注入语义缓存（[semantic_cache] enabled=true 时；默认 false
/// 行为不变）。comparison_mode 非法值启动报错，不静默回退；取值本身已
/// 废弃不消费（解析保留兼容，非法值仍启动报错）。
fn attach_semantic_cache(service: EmbeddingService, config: &VecboostConfig) -> EmbeddingService {
    if !config.semantic_cache.enabled {
        return service;
    }
    let mode: vecboost::ComparisonMode = config
        .semantic_cache
        .comparison_mode
        .parse()
        .map_err(|e: String| {
            anyhow::anyhow!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "startup-semantic-mode-invalid",
                    vecboost::i18n::tr_args(&[("detail", &e)]),
                )
            )
        })
        .expect("semantic_cache.comparison_mode validated at parse");
    if mode != vecboost::ComparisonMode::Exact {
        log::warn!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-semantic-mode-deprecated",
                vecboost::i18n::tr_args(&[("mode", &format!("{:?}", mode))])
            )
        );
    }
    log::info!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-semantic-enabled",
            vecboost::i18n::tr_args(&[
                (
                    "threshold",
                    &config.semantic_cache.similarity_threshold.to_string()
                ),
                ("fuzzy", &config.semantic_cache.fuzzy_threshold.to_string()),
                ("capacity", &config.semantic_cache.capacity.to_string())
            ])
        )
    );
    let semantic = Arc::new(
        vecboost::SemanticCache::with_capacity(
            config.semantic_cache.similarity_threshold,
            config.semantic_cache.capacity,
        )
        .with_fuzzy_threshold(config.semantic_cache.fuzzy_threshold),
    );
    service.with_semantic_cache(semantic)
}

/// 转录服务装配：`[transcribe] enabled=true` 时经公开门面
/// `vecboost::transcribe::open` 急切构建会话（bundle 缺失 fail-fast，与
/// translation pairs 同口径）；否则未装配（端点注册、请求 400
/// transcribe-not-configured）。
#[cfg(feature = "transcribe")]
fn build_transcribe_service(config: &VecboostConfig) -> anyhow::Result<TranscribeService> {
    if !config.transcribe.enabled {
        return Ok(TranscribeService::unconfigured());
    }
    let opts = vecboost::transcribe::TranscribeOpts {
        intra_threads: config.transcribe.intra_threads,
        no_spin: config.transcribe.no_spin,
        cuda: config.transcribe.cuda,
        vad_threshold: config.transcribe.vad_threshold,
    };
    let session =
        vecboost::transcribe::open(std::path::Path::new(&config.transcribe.models_dir), &opts)
            .map_err(|e| {
                anyhow::anyhow!(
                    "{}",
                    vecboost::i18n::tr_with_args(
                        "startup-transcribe-session-open",
                        vecboost::i18n::tr_args(&[("detail", &e.to_string())]),
                    )
                )
            })?;
    Ok(TranscribeService::new(session))
}

#[cfg(not(feature = "transcribe"))]
fn build_transcribe_service(_config: &VecboostConfig) -> anyhow::Result<TranscribeService> {
    Ok(TranscribeService::unconfigured())
}

/// 翻译路由表：启动期消费 [translation].pairs 逐 pair 急切构建
/// （EngineFactory task=translate 分派臂），任一 bundle 缺失 fail-fast；
/// 空 pairs = 空表服务（端点注册、请求 400 translate-not-configured）。
/// 评审 PERF-4：按 pair 数分摊 per-Session intra 线程预算（防多 pair
/// Session 全核过订阅）；单 pair/空表回退全核。
#[allow(unused_variables)]
fn build_translate_pairs(
    config: &VecboostConfig,
    device: vecboost::config::model::DeviceType,
) -> anyhow::Result<
    std::collections::HashMap<(String, String), vecboost::service::translate::TranslatePairRoute>,
> {
    #[cfg_attr(not(feature = "onnx"), allow(unused_mut))]
    let mut translate_pairs = std::collections::HashMap::new();

    #[cfg(feature = "onnx")]
    if config.translation.pairs.len() > 1 {
        let cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let divisor = config.translation.pairs.len().clamp(1, 4);
        let budget = (cores / divisor).clamp(1, cores);
        vecboost::engine::set_session_intra_threads(budget);
        log::info!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-translation-threads",
                vecboost::i18n::tr_args(&[
                    ("budget", &budget.to_string()),
                    ("pairs", &config.translation.pairs.len().to_string())
                ])
            )
        );
    }
    #[cfg(not(feature = "onnx"))]
    if !config.translation.pairs.is_empty() {
        return Err(anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr("startup-translation-no-onnx")
        ));
    }

    #[cfg(feature = "onnx")]
    for (index, pair) in config.translation.pairs.iter().enumerate() {
        // 缺省值取 TranslationParams::default() 单一事实源（评审 ARC-2：
        // 禁止 512 字面量第二处来源，default 变更时自动联动）
        let defaults = vecboost::config::model::TranslationParams::default();
        let mut translation_params = vecboost::config::model::TranslationParams::new(
            pair.max_source_tokens.unwrap_or(defaults.max_source_tokens),
            pair.max_new_tokens.unwrap_or(defaults.max_new_tokens),
            pair.min_new_tokens.unwrap_or(defaults.min_new_tokens),
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "startup-translate-pair-invalid",
                    vecboost::i18n::tr_args(&[
                        ("index", &index.to_string()),
                        ("src", pair.source_language.as_str()),
                        ("tgt", pair.target_language.as_str()),
                        ("detail", &e.to_string()),
                    ]),
                )
            )
        })?;
        translation_params.output_fixes = pair.output_fixes.clone();
        let pair_config = vecboost::config::model::ModelConfig {
            name: format!(
                "translate-{}-{}",
                pair.source_language, pair.target_language
            ),
            engine_type: vecboost::config::model::EngineType::Onnx,
            model_path: pair.model_path.clone(),
            tokenizer_path: pair.tokenizer_path.clone(),
            device: device.clone(),
            max_batch_size: 1,
            oom_fallback_enabled: true,
            task: vecboost::config::model::ModelTask::Translate,
            translation_params: Some(translation_params.clone()),
            transcription_params: None,
            ..Default::default()
        };
        translate_pairs.insert(
            (pair.source_language.clone(), pair.target_language.clone()),
            vecboost::service::translate::TranslatePairRoute::new(
                Arc::new(RwLock::new(
                    vecboost::engine::EngineFactory::create(
                        vecboost::config::model::EngineType::Onnx,
                        &pair_config,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "{}",
                            vecboost::i18n::tr_with_args(
                                "startup-translate-pair-load-failed",
                                vecboost::i18n::tr_args(&[
                                    ("index", &index.to_string()),
                                    ("src", pair.source_language.as_str()),
                                    ("tgt", pair.target_language.as_str()),
                                    ("detail", &e.to_string()),
                                ]),
                            )
                        )
                    })?,
                )),
                translation_params,
            ),
        );
    }

    Ok(translate_pairs)
}

async fn init_engine_and_services(
    config: &VecboostConfig,
    checkpoints: &std::collections::BTreeMap<String, CheckpointPreset>,
) -> anyhow::Result<(
    Arc<RwLock<AnyEngine>>,
    Arc<RwLock<EmbeddingService>>,
    Arc<RwLock<RerankService>>,
    Arc<RwLock<DecisionService>>,
    DetectService,
    Arc<TranslateService>,
    Arc<TranscribeService>,
    ModelConfig,
)> {
    let model_config = resolve_engine_model_config(config)?;
    let engine_type = model_config.engine_type.clone();

    log::info!("{}", vecboost::i18n::tr("log-main-engine-init"));
    let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(
        vecboost::engine::EngineFactory::create(engine_type, &model_config)?,
    ));

    let service = build_embedding_service(&engine, config, Some(model_config.clone())).await?;
    let service = attach_model_manager(service, config).await;

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
                    "{}",
                    vecboost::i18n::tr_with_args(
                        "log-main-residency",
                        vecboost::i18n::tr_args(&[
                            ("max", &max.unwrap_or(usize::MAX).to_string()),
                            ("budget", &format!("{:?}", budget))
                        ])
                    )
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

    let service = attach_semantic_cache(service, config);

    // [model.checkpoints] 预设表注入（MCP/CLI/server 三模式共用——switch_model
    // 按名命中预设时应用参数；查表消费唯一入口为 EmbeddingService 持有的该表）
    let service = service.with_checkpoints(Arc::new(checkpoints.clone()));

    service.init_memory_limit().await;
    let service = Arc::new(RwLock::new(service));
    // 启动回放 WAL 重建缓存（未启用持久层时为 no-op）。
    service.read().await.load_persisted_cache().await;

    let rerank_service = Arc::new(RwLock::new(RerankService::new(
        engine.clone(),
        Some(model_config.clone()),
    )));

    // 决策服务无条件预构建（零成本，三种模式签名统一）：非 decision 配置下
    // decide 经 trait 默认臂/能力门返回 UnsupportedTask（HTTP 400），端点
    // 始终可注册应答，不随 engine_type/task 配置缺省而消失
    let decision_service = Arc::new(RwLock::new(DecisionService::new(engine.clone())));

    // 检测服务无条件预构建（零配置可用）：无引擎无状态（µs 级 whatlang
    // 统计），端点始终可注册应答
    let detect_service = DetectService::new();

    let translate_service = Arc::new(TranslateService::new(build_translate_pairs(
        config,
        model_config.device.clone(),
    )?));

    log::info!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-translation-ready",
            vecboost::i18n::tr_args(&[(
                "pairs",
                &translate_service.configured_pairs().len().to_string()
            )])
        )
    );

    // 转录服务：enabled=false 或 transcribe feature 未启用 → 端点注册但不装配
    // （transcribe-not-configured 400 显性拒绝）。引擎经 [transcribe].models_dir
    // 急切装配（bundle 缺失 fail-fast，与 translation pairs 同口径）。
    let transcribe_service = Arc::new(build_transcribe_service(config)?);
    if transcribe_service.is_configured() {
        log::info!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-transcribe-ready",
                vecboost::i18n::tr_args(&[("bundle", &config.transcribe.models_dir)])
            )
        );
    }

    Ok((
        engine,
        service,
        rerank_service,
        decision_service,
        detect_service,
        translate_service,
        transcribe_service,
        model_config,
    ))
}

#[cfg(feature = "mcp")]
async fn run_mcp_server(
    service: Arc<RwLock<EmbeddingService>>,
    rerank_service: Arc<RwLock<RerankService>>,
    decision_service: Arc<RwLock<DecisionService>>,
    detect_service: DetectService,
    translate_service: Arc<TranslateService>,
    transcribe_service: Arc<TranscribeService>,
) -> anyhow::Result<()> {
    use sdforge::rmcp::{ServiceExt, transport::io::stdio};

    log::info!("{}", vecboost::i18n::tr("log-main-mcp-start"));
    let mut kit = trait_kit::AsyncKit::new();
    kit.set_config(service);
    kit.set_config(rerank_service);
    kit.set_config(decision_service);
    kit.set_config(detect_service);
    kit.set_config(translate_service.clone());
    kit.set_config(transcribe_service.clone());
    kit.register::<EmbeddingModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
    kit.register::<RerankModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RerankModule: {}", e))?;
    kit.register::<DecisionModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "DecisionModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
    kit.register::<DetectModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[("module", "DetectModule"), ("detail", &e.to_string()),]),
            )
        )
    })?;
    kit.register::<TranslateModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "TranslateModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
    kit.register::<TranscribeModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "TranscribeModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
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
        eprintln!(
            "{}",
            vecboost::i18n::tr_with_args(
                "cli-unknown-subcommand",
                vecboost::i18n::tr_args(&[("name", first)]),
            )
        );
        eprintln!();
        eprintln!(
            "{}",
            vecboost::i18n::tr_with_args(
                "cli-available-subcommands",
                vecboost::i18n::tr_args(&[("list", &known.join(", "))]),
            )
        );
        eprintln!("{}", vecboost::i18n::tr("cli-usage-hint"));
        std::process::exit(2);
    }
}

#[cfg(feature = "cli")]
async fn run_cli_command(
    service: Arc<RwLock<EmbeddingService>>,
    rerank_service: Arc<RwLock<RerankService>>,
    decision_service: Arc<RwLock<DecisionService>>,
    detect_service: DetectService,
    translate_service: Arc<TranslateService>,
    transcribe_service: Arc<TranscribeService>,
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
    kit.set_config(decision_service);
    kit.set_config(detect_service);
    kit.set_config(translate_service.clone());
    kit.set_config(transcribe_service.clone());
    kit.register::<EmbeddingModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
    kit.register::<RerankModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RerankModule: {}", e))?;
    kit.register::<DecisionModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "DecisionModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
    kit.register::<DetectModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[("module", "DetectModule"), ("detail", &e.to_string()),]),
            )
        )
    })?;
    kit.register::<TranslateModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "TranslateModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
    kit.register::<TranscribeModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "TranscribeModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
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
async fn build_auth_dao(
    config: &VecboostConfig,
    #[cfg(feature = "db")] db_pool: Option<&vecboost::db::DbPool>,
) -> anyhow::Result<Arc<dyn garrison::dao::GarrisonDao>> {
    match config.auth.session_backend {
        vecboost::config::app::SessionBackend::Memory => Ok(Arc::new(
            garrison::dao::GarrisonDaoOxcache::new()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create GarrisonDaoOxcache: {}", e))?,
        )),
        vecboost::config::app::SessionBackend::Database => {
            #[cfg(feature = "db")]
            {
                let Some(pool) = db_pool else {
                    return Err(anyhow::anyhow!(
                        "{}",
                        vecboost::i18n::tr("startup-auth-session-backend-db-pool-unavailable")
                    ));
                };
                #[cfg(any(feature = "sqlite", feature = "postgres"))]
                {
                    let dao = vecboost::auth::atomic_kv_dao::build_atomic_dao(pool.inner().clone())
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to build atomic garrison dao: {e}"))?;
                    Ok(Arc::new(dao))
                }
                #[cfg(not(any(feature = "sqlite", feature = "postgres")))]
                {
                    let _ = pool;
                    Err(anyhow::anyhow!(
                        "{}",
                        vecboost::i18n::tr("startup-auth-session-backend-db-feature-missing")
                    ))
                }
            }
            #[cfg(not(feature = "db"))]
            {
                Err(anyhow::anyhow!(
                    "{}",
                    vecboost::i18n::tr("startup-auth-session-backend-db-feature-missing")
                ))
            }
        }
    }
}

/// 会话后端日志标签（backend, table）——`database` 固定 `garrison_kv`；
/// 仅返回字面量，不携带数据库连接串/凭据。
#[cfg(feature = "auth")]
fn session_backend_labels(
    backend: vecboost::config::app::SessionBackend,
) -> (&'static str, &'static str) {
    match backend {
        vecboost::config::app::SessionBackend::Memory => ("memory", "-"),
        vecboost::config::app::SessionBackend::Database => ("database", "garrison_kv"),
    }
}

/// Redis URL 脱敏标签：仅保留 scheme + host:port，剔除 userinfo/path/query。
fn redis_endpoint_label(url: Option<&str>) -> String {
    let Some(url) = url else {
        return "-".to_string();
    };
    let (scheme, rest) = url.split_once("://").unwrap_or(("redis", url));
    // 先剔 userinfo（rsplit 以最后一个 @ 为界——密码可能含未编码的 / ? #，
    // 顺序反了会把密码片段当 host 输出），再剔 path/query/fragment。
    let no_userinfo = rest.rsplit('@').next().unwrap_or(rest);
    let host_port = no_userinfo.split(['/', '?', '#']).next().unwrap_or("");
    if host_port.is_empty() {
        "-".to_string()
    } else {
        format!("{scheme}://{host_port}")
    }
}

/// 启动日志：限流/缓存后端类型 + 脱敏 endpoint（不输出连接串凭据）。
fn log_shared_state_backends(config: &VecboostConfig) {
    use vecboost::config::app::{CacheBackendKind, RateLimitBackend};

    let (rate_backend, rate_endpoint) = match config.rate_limit.backend {
        RateLimitBackend::Memory => ("memory", "-".to_string()),
        RateLimitBackend::Redis => (
            "redis",
            redis_endpoint_label(config.rate_limit.redis_url.as_deref()),
        ),
    };
    log::info!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-rate-limit-backend",
            vecboost::i18n::tr_args(&[
                ("backend", rate_backend),
                ("endpoint", rate_endpoint.as_str()),
            ])
        )
    );

    let (cache_backend, cache_endpoint) = match config.embedding.cache_backend {
        CacheBackendKind::Memory => ("memory", "-".to_string()),
        CacheBackendKind::Redis => (
            "redis",
            redis_endpoint_label(config.embedding.redis_url.as_deref()),
        ),
    };
    log::info!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-cache-backend",
            vecboost::i18n::tr_args(&[
                ("backend", cache_backend),
                ("endpoint", cache_endpoint.as_str()),
            ])
        )
    );
}

#[cfg(feature = "auth")]
async fn init_auth(
    config: &VecboostConfig,
    #[cfg(feature = "db")] db_pool: Option<&vecboost::db::DbPool>,
) -> anyhow::Result<(Option<Arc<GarrisonHandle>>, Option<Arc<GarrisonCsrfConfig>>)> {
    // 会话后端校验不分 auth.enabled 分支：database+auth 关闭属配置自相矛盾，
    // 必须在任何装配前拒启动（不能因 auth 关闭而静默跳过）。
    vecboost::auth::config::validate_session_backend(&config.auth)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

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

        let dao = build_auth_dao(
            config,
            #[cfg(feature = "db")]
            db_pool,
        )
        .await?;

        let (session_backend, session_table) = session_backend_labels(config.auth.session_backend);
        log::info!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-auth-session-backend",
                vecboost::i18n::tr_args(&[("backend", session_backend), ("table", session_table),])
            )
        );

        let garrison_config = map_auth_config_to_garrison(&config.auth);

        // garrison 0.9：init(dao, config, interface) → builder 链（build() 启动全局单例
        // 后台 task）；firewall-* feature 启用时 builder 自动注入防火墙检查钩子
        garrison::prelude::GarrisonManager::builder()
            .dao(dao)
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

        log::info!("{}", vecboost::i18n::tr("log-main-auth-enabled"));

        Some(Arc::new(GarrisonHandle {
            admin_password_hash,
            admin_username,
            token_timeout_secs: garrison_config.timeout,
        }))
    } else {
        log::info!("{}", vecboost::i18n::tr("log-main-auth-disabled"));
        None
    };

    let garrison_csrf_config: Option<Arc<GarrisonCsrfConfig>> = if config.auth.csrf.enabled {
        let csrf = GarrisonCsrfConfig::default();
        log::info!("{}", vecboost::i18n::tr("log-main-csrf-enabled"));
        Some(Arc::new(csrf))
    } else {
        log::info!("{}", vecboost::i18n::tr("log-main-csrf-disabled"));
        None
    };

    Ok((garrison_handle, garrison_csrf_config))
}

/// [device] auto_plan：硬件探测计划填充 WorkerConfig 未显式配置字段。
/// 显式判定：server.workers（Option，Some 即显式）；其余字段以"与编译期
/// 默认不同"近似判定（显式设为默认值会被视为未显式，见注释与启动日志）。
fn apply_auto_plan(config: &VecboostConfig, worker_config: &mut vecboost::pipeline::WorkerConfig) {
    if !config.device.auto_plan {
        return;
    }
    let probes = gather_probes(config);
    let hw_plan = vecboost::planner::plan(&probes);
    log::info!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-plan-hardware",
            vecboost::i18n::tr_args(&[
                ("threads", &hw_plan.worker_threads.to_string()),
                ("batch", &hw_plan.max_batch_size.to_string()),
                ("wait", &hw_plan.batch_wait_ms.to_string()),
                ("quantized", &hw_plan.quantized_recommended.to_string()),
                ("bottleneck", &hw_plan.bottleneck.to_string()),
                ("rationale", &hw_plan.rationale)
            ])
        )
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
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-plan-applied",
            vecboost::i18n::tr_args(&[
                ("batch", &bs.to_string()),
                ("wait", &bw.to_string()),
                ("planned_batch", &hw_plan.max_batch_size.to_string()),
                ("planned_wait", &hw_plan.batch_wait_ms.to_string()),
                ("threads", &wt.to_string())
            ])
        )
    );
    worker_config.max_batch_size = bs;
    worker_config.batch_wait_ms = bw;
}

async fn init_pipeline(
    config: &VecboostConfig,
    service: &Arc<RwLock<EmbeddingService>>,
    decision_service: &Arc<RwLock<DecisionService>>,
) -> anyhow::Result<(
    Arc<PriorityRequestQueue>,
    Arc<ResponseChannel>,
    Arc<PriorityCalculator>,
    Arc<WorkerManager>,
)> {
    if config.pipeline.enabled {
        log::info!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-pipeline-enabled",
                vecboost::i18n::tr_args(&[(
                    "size",
                    &config.pipeline.queue.max_queue_size.to_string()
                )])
            )
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
            drain_timeout_secs: config.pipeline.worker.drain_timeout_secs,
            max_batch_size: config.embedding.max_batch_size,
            batch_wait_ms: config.pipeline.worker.batch_wait_ms,
        };

        // [device] auto_plan 为 true 时，用硬件探测计划填充未显式配置字段。
        // 显式判定：server.workers（Option，Some 即显式）；其余字段以"与编译期
        // 默认不同"近似判定（显式设为默认值会被视为未显式，见注释与启动日志）。
        apply_auto_plan(config, &mut worker_config);

        let worker_manager = Arc::new(
            WorkerManager::new(
                pipeline_queue.clone(),
                response_channel.clone(),
                worker_config.clone(),
                service.clone(),
            )
            // 决策请求经共享队列出队后由 worker 执行（P2-3 决策任务入队）
            .with_decision_service(decision_service.clone()),
        );

        // start() = spawn min_workers + 启动扩缩容 monitor（
        // 此前只 spawn min_workers，max_workers/scale 阈值全是死配置）
        if let Err(e) = worker_manager.start().await {
            log::error!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "log-main-worker-start-failed",
                    vecboost::i18n::tr_args(&[("detail", &e.to_string())])
                )
            );
        }

        log::info!("{}", vecboost::i18n::tr("log-main-pipeline-ready"));

        Ok((
            pipeline_queue,
            response_channel,
            priority_calculator,
            worker_manager,
        ))
    } else {
        log::info!("{}", vecboost::i18n::tr("log-main-pipeline-disabled"));

        Ok((
            Arc::new(PriorityRequestQueue::new(0)),
            Arc::new(ResponseChannel::new()),
            Arc::new(PriorityCalculator::new(PriorityConfig::default())),
            Arc::new(
                WorkerManager::new(
                    Arc::new(PriorityRequestQueue::new(0)),
                    Arc::new(ResponseChannel::new()),
                    WorkerConfig::default(),
                    service.clone(),
                )
                .with_decision_service(decision_service.clone()),
            ),
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
        log::warn!("{}", vecboost::i18n::tr("log-main-gpu-no-backend"));
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
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-insecure-warning",
                vecboost::i18n::tr_args(&[("host", host)])
            )
        );
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-insecure-refusal",
            vecboost::i18n::tr_args(&[("host", host)])
        )
    ))
}

fn main() {
    // i18n 必须先于一切用户可见输出：thread-tune 提示（numa_advice）与
    // CLI fail-fast 文案（未知子命令/--config 校验）均已接线 FTL，早于 app_main。
    vecboost::i18n::init();
    // jemalloc decay 调优（wsl2-oom-hardening-r2 D3）：boot 期 je_malloc_conf
    // 强符号已于首次分配前生效，此处只做生效性观测与记录
    #[cfg(all(target_os = "linux", not(target_env = "musl")))]
    match vecboost::allocator_tune::effective_boot_conf() {
        Ok((dirty, muzzy)) => {
            log::info!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "log-main-jemalloc-decay",
                    vecboost::i18n::tr_args(&[
                        ("dirty", &dirty.to_string()),
                        ("muzzy", &muzzy.to_string())
                    ])
                )
            );
        }
        Err(e) => log::warn!(
            "{}",
            vecboost::i18n::tr_with_args(
                "log-main-jemalloc-probe-failed",
                vecboost::i18n::tr_args(&[("detail", &e.to_string())])
            )
        ),
    }
    // tokio runtime 线程数取物理核检测（显式 server.workers 优先，
    // VECBOOST_NO_THREAD_TUNE=1 回退 num_cpus）。配置在运行时前 best-effort
    // 预读，失败则由 app_main 内正式加载路径报错。
    let pre_explicit: Option<usize> = VecboostConfig::load_via_confers()
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
        "{}",
        vecboost::i18n::tr_with_args(
            "startup-thread-tune",
            vecboost::i18n::tr_args(&[
                (
                    "physical",
                    &detected
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| vecboost::i18n::tr("startup-thread-unknown")),
                ),
                ("logical", &fallback.to_string()),
                ("effective", &effective_threads.to_string()),
                (
                    "explicit",
                    &pre_explicit
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| vecboost::i18n::tr("startup-thread-unset")),
                ),
            ]),
        )
    );
    if let Some(advice) = numa_advice() {
        eprintln!("[thread-tune] {}", advice);
    }
    // rayon 全局池与 tokio 对齐（仅初始化一次，失败则记录并继续）。
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(effective_threads)
        .build_global()
        .map_err(|e| {
            eprintln!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "startup-rayon-skipped",
                    vecboost::i18n::tr_args(&[("detail", &e.to_string())]),
                )
            );
        });
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
        eprintln!(
            "{}",
            vecboost::i18n::tr_with_args(
                "cli-error-prefix",
                vecboost::i18n::tr_args(&[("detail", &e.to_string())])
            )
        );
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
                eprintln!("{}", vecboost::i18n::tr("cli-config-requires-path"));
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
    let config = load_startup_config(config_path.as_deref());

    // doctor 只读诊断 ——在 logger/引擎装配之前短路运行。
    // 诊断必须能在模型损坏、依赖缺失时工作，因此不走需要
    // EmbeddingService 的 run_cli_command 分发路径。
    #[cfg(feature = "cli")]
    if _filtered_args.get(1).map(|a| a.as_str()) == Some("doctor") {
        vecboost::doctor::DoctorReport::run(&config)
            .await
            .print_and_exit();
    }

    let (logger_manager, log_level) = init_logger(&config, cli_mode, mcp_mode).await?;

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

    // 会话后端校验提前到 DB 池初始化之前：database + auth 关闭等矛盾配置
    // 在对部署库产生连接/DDL 副作用前即拒绝（init_auth 内保留二次校验纵深）。
    #[cfg(feature = "auth")]
    vecboost::auth::config::validate_session_backend(&config.auth)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    #[cfg(not(feature = "auth"))]
    vecboost::config::app::validate_session_backend_without_auth(&config.auth)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // 共享状态后端校验提前到任何连接副作用之前：redis + persist_path 等
    // 矛盾配置在建连前拒绝；CLI/MCP 分发路径同样消费（不再静默容忍）。
    config
        .rate_limit
        .validate_backend()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    config
        .embedding
        .validate_cache_backend()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    enforce_encryption_gate(&config)?;

    #[cfg(feature = "db")]
    let (db_pool, _db_metrics) = init_db_pool(&config).await?;
    // 供 /health?depth=full 的 DB 就绪探测
    #[cfg(feature = "db")]
    let _ = vecboost::db::register_global_pool(std::sync::Arc::new(db_pool.clone()));

    // [model.checkpoints] 预设表启动 fail-fast：非法条目在此拒绝启动，
    // resolved 表供 server kit 注入与 EmbeddingService switch 查表共用
    let checkpoints = resolve_checkpoints_fail_fast(&config)?;
    log_checkpoints_resolved(&config, &checkpoints);

    let (
        _engine,
        service,
        rerank_service,
        decision_service,
        detect_service,
        translate_service,
        transcribe_service,
        _model_config,
    ) = init_engine_and_services(&config, &checkpoints).await?;

    // `--warmup N` 启动预热 ——N 条合成短文本推理，预热 mkl/代码路径/
    // tokenizer 缓存。放在 MCP/CLI 分流之前，三种模式均受益。
    let warmup = parse_warmup_count(&_filtered_args);
    if warmup > 0 {
        run_startup_warmup(&_engine, warmup).await;
    }

    if dispatch_non_server_modes(
        &service,
        &rerank_service,
        &decision_service,
        &detect_service,
        &translate_service,
        &transcribe_service,
        &_filtered_args,
    )
    .await?
    {
        return Ok(());
    }

    let security = init_security_stack(
        &config,
        #[cfg(feature = "db")]
        db_pool.clone(),
    )
    .await?;

    assemble_and_serve_http(
        &config,
        config_path,
        service,
        rerank_service,
        decision_service,
        detect_service,
        translate_service,
        transcribe_service,
        logger_manager,
        security,
        #[cfg(feature = "db")]
        db_pool,
    )
    .await
}

/// 预设表解析结果盘点日志：同名显式条目对内置做字段级覆盖、不产生
/// 第二条目（merge 语义），单列计数避免运维据「总 resolved + explicit」
/// 盘点得出重复总数。
fn log_checkpoints_resolved(
    config: &VecboostConfig,
    checkpoints: &std::collections::BTreeMap<String, CheckpointPreset>,
) {
    let builtin_overrides = config
        .model
        .checkpoints
        .contains_key(vecboost::config::model::BUILTIN_MULTILINGUAL_CHECKPOINT)
        as usize;
    log::info!(
        "{}",
        vecboost::i18n::tr_with_args(
            "log-main-model-checkpoints-resolved",
            vecboost::i18n::tr_args(&[
                ("a1", &(checkpoints.len()).to_string()),
                ("a2", &(config.model.checkpoints.len()).to_string()),
                ("a3", &(builtin_overrides).to_string())
            ])
        )
    );
}

/// server 路径安全栈：绑定安全闸门（fail-fast）→ 限流适配器 → auth
/// （garrison handle + CSRF 配置，auth feature）→ 审计日志。
/// 仅 HTTP 服务器路径需要（MCP/CLI 均已提前返回）。
struct SecurityStack {
    rate_limiter: Arc<LimiteronAdapter>,
    audit_logger: Option<Arc<AuditLogger>>,
    #[cfg(feature = "auth")]
    garrison_handle: Option<Arc<vecboost::auth::GarrisonHandle>>,
    #[cfg(feature = "auth")]
    garrison_csrf_config: Option<Arc<vecboost::auth::GarrisonCsrfConfig>>,
}

async fn init_security_stack(
    config: &VecboostConfig,
    #[cfg(feature = "db")] db_pool: crate::DbPool,
) -> anyhow::Result<SecurityStack> {
    validate_bind_safety(&config.server.host, config.auth.enabled)?;

    // 多副本共享状态后端校验：memory 恒放行；redis 缺 feature/URL 启动期显性拒绝
    config
        .rate_limit
        .validate_backend()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    config
        .embedding
        .validate_cache_backend()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    log_shared_state_backends(config);

    let rate_limiter = Arc::new(
        LimiteronAdapter::try_new(RateLimitSettings {
            global_requests_per_minute: config.rate_limit.global_requests_per_minute,
            ip_requests_per_minute: config.rate_limit.ip_requests_per_minute,
            user_requests_per_minute: config.rate_limit.user_requests_per_minute,
            api_key_requests_per_minute: config.rate_limit.api_key_requests_per_minute,
            backend: config.rate_limit.backend,
            redis_url: config.rate_limit.redis_url.clone(),
            global_window_ms: config.rate_limit.global_window_ms,
        })
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?,
    );

    #[cfg(feature = "auth")]
    let (garrison_handle, garrison_csrf_config) = init_auth(
        config,
        #[cfg(feature = "db")]
        Some(&db_pool),
    )
    .await?;

    let audit_logger = init_audit_logger(
        config,
        #[cfg(feature = "db")]
        db_pool,
    );

    Ok(SecurityStack {
        rate_limiter,
        audit_logger,
        #[cfg(feature = "auth")]
        garrison_handle,
        #[cfg(feature = "auth")]
        garrison_csrf_config,
    })
}

/// 启动配置加载：显式指定的 --config 文件不存在 → 立即报错退出(码 2)，
/// 杜绝"以为自定义配置生效,实际跑默认配置"的静默回退；加载失败以
/// stderr 输出后退出(码 1)——此时尚无日志后端。
fn load_startup_config(config_path: Option<&str>) -> VecboostConfig {
    if let Some(p) = config_path
        && !std::path::Path::new(p).exists()
    {
        eprintln!(
            "{}",
            vecboost::i18n::tr_with_args(
                "cli-config-not-found",
                vecboost::i18n::tr_args(&[("path", p)]),
            )
        );
        eprintln!("{}", vecboost::i18n::tr("cli-config-not-file-hint"));
        std::process::exit(2);
    }
    let result = match config_path {
        Some(p) => VecboostConfig::load_via_confers_with_path(p),
        None => VecboostConfig::load_via_confers(),
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
}

/// [logging] 配置段接入 logger；VECBOOST_LOG_LEVEL 环境变量优先级高于
/// 配置文件（白名单含 inklog 合法别名 warning 等）。CLI/MCP 模式将全部
/// 日志级别路由到 stderr——stdout 只承载机器可读 JSON / JSON-RPC 协议
/// 消息（inklog rc.8 已修复 ConsoleSink 忽略 enabled=false 的上游缺陷；
/// 此处的 stderr 路由与其无关，只为协议纯净性）。
async fn init_logger(
    config: &VecboostConfig,
    cli_mode: bool,
    mcp_mode: bool,
) -> anyhow::Result<(Arc<inklog::LoggerManager>, String)> {
    let log_level = std::env::var("VECBOOST_LOG_LEVEL")
        .ok()
        .filter(|l| {
            ["trace", "debug", "info", "warn", "warning", "error"]
                .contains(&l.to_lowercase().as_str())
        })
        .unwrap_or_else(|| config.logging.level.clone());

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
    Ok((logger_manager, log_level))
}

/// 加密闸门（production hardening）：VECBOOST_REQUIRE_ENCRYPTION=1 时
/// 无有效 key 拒绝启动；未强制时检测到敏感配置仅提示明文存储风险。
fn enforce_encryption_gate(config: &VecboostConfig) -> anyhow::Result<()> {
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
    Ok(())
}

/// `--warmup N` 启动预热：N 条合成短文本推理，预热 mkl/代码路径/
/// tokenizer 缓存。按 task 分派：task=decision 时 embed 恒
/// UnsupportedTask（决策引擎不产向量），改用单题 noul decide 预热全链
/// （预处理→5 张量推理→后处理）；预热失败显性记录，不再以成功日志掩盖
/// 零次执行（规则 11）。
async fn run_startup_warmup(engine: &Arc<RwLock<AnyEngine>>, warmup: u32) {
    let start = std::time::Instant::now();
    let guard = engine.read().await;
    let decision_pipeline = vecboost::engine::InferenceEngine::supports_task(
        &*guard,
        vecboost::config::model::ModelTask::Decision,
    );
    let failures = if decision_pipeline {
        let req: vecboost::domain::DecisionRequest = serde_json::from_str(
            r#"{"state":"vecboost warmup","questions":[{"name":"warmup","qtype":"noul","instructions":"warm up the decision pipeline"}]}"#,
        )
        .expect("static warmup decision request");
        match vecboost::engine::InferenceEngine::decide(&*guard, &req) {
            Ok(_) => 0,
            Err(e) => {
                log::warn!(
                    "{}",
                    vecboost::i18n::tr_with_args(
                        "warmup-decision-pipeline-failed",
                        vecboost::i18n::tr_args(&[("e", &e.to_string())])
                    )
                );
                1
            }
        }
    } else {
        (0..warmup)
            .filter(|i| {
                let text = format!(
                    "vecboost warmup sentence {i}: the quick brown fox jumps over the lazy dog"
                );
                vecboost::engine::InferenceEngine::embed(&*guard, &text).is_err()
            })
            .count()
    };
    let mode = if decision_pipeline {
        vecboost::i18n::tr("warmup-mode-decide")
    } else {
        vecboost::i18n::tr("warmup-mode-embed")
    };
    if failures > 0 {
        log::warn!(
            "{}",
            vecboost::i18n::tr_with_args(
                "warmup-incomplete",
                vecboost::i18n::tr_args(&[
                    ("mode", &mode),
                    ("failures", &failures.to_string()),
                    ("elapsed", &format!("{:?}", start.elapsed())),
                ])
            )
        );
    } else {
        log::info!(
            "{}",
            vecboost::i18n::tr_with_args(
                "warmup-complete",
                vecboost::i18n::tr_args(&[
                    ("mode", &mode),
                    ("elapsed", &format!("{:?}", start.elapsed())),
                ])
            )
        );
    }
}

/// 审计日志装配：[audit] enabled 时构建（db 构建下挂 DB 双写），禁用时
/// 显性 warn（安全事件不留痕必须可感知）。
#[allow(unused_variables)]
fn init_audit_logger(
    config: &VecboostConfig,
    #[cfg(feature = "db")] db_pool: crate::DbPool,
) -> Option<Arc<AuditLogger>> {
    if !config.audit.enabled {
        log::warn!("Audit logging is DISABLED - security events will not be logged!");
        return None;
    }
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
        Some(Arc::new(AuditLogger::new_with_db(audit_config, db_pool)))
    }
    #[cfg(not(feature = "db"))]
    Some(Arc::new(AuditLogger::new(audit_config)))
}

/// Signal-aware graceful shutdown（SIGINT + SIGTERM）；SIGTERM 安装失败
/// 时降级仅 ctrl_c（不阻断启动）。
async fn install_signal_handlers() {
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
}

/// 非 server 模式分流：`--mcp`（stdio JSON-RPC）与 CLI 子命令。命中即
/// 消费服务句柄并返回 true（调用方直接返回）；server 路径继续装配。
/// CLI 判定消费 &args 克隆；MCP 分流置 stdio 抑制日志。
#[allow(unused_variables)]
async fn dispatch_non_server_modes(
    service: &Arc<RwLock<EmbeddingService>>,
    rerank_service: &Arc<RwLock<RerankService>>,
    decision_service: &Arc<RwLock<DecisionService>>,
    detect_service: &DetectService,
    translate_service: &Arc<TranslateService>,
    transcribe_service: &Arc<TranscribeService>,
    _filtered_args: &[String],
) -> anyhow::Result<bool> {
    #[cfg(feature = "mcp")]
    if std::env::args().any(|a| a == "--mcp") {
        // MCP 启动失败必须传播（拆分前的 `return ...await` 语义）：
        // 吞掉错误会以 exit 0 假成功退出，客户端只看到流中断
        run_mcp_server(
            service.clone(),
            rerank_service.clone(),
            decision_service.clone(),
            *detect_service,
            translate_service.clone(),
            transcribe_service.clone(),
        )
        .await?;
        return Ok(true);
    }

    #[cfg(feature = "cli")]
    if run_cli_command(
        service.clone(),
        rerank_service.clone(),
        decision_service.clone(),
        *detect_service,
        translate_service.clone(),
        transcribe_service.clone(),
        _filtered_args.to_vec(),
    )
    .await?
    {
        return Ok(true);
    }

    Ok(false)
}

/// HTTP 服务器模式装配与启动：pipeline → AsyncKit 模块注册表 → 分级
/// 关停协调器 → 后台任务 → VecboostState/api 注入 → 路由 → 监听 →
/// 生命周期（分级优雅关闭）。MCP/CLI 分流之后才会走到这里。
#[allow(clippy::too_many_arguments)]
async fn assemble_and_serve_http(
    config: &VecboostConfig,
    config_path: Option<String>,
    service: Arc<RwLock<EmbeddingService>>,
    rerank_service: Arc<RwLock<RerankService>>,
    decision_service: Arc<RwLock<DecisionService>>,
    detect_service: DetectService,
    translate_service: Arc<TranslateService>,
    transcribe_service: Arc<TranscribeService>,
    logger_manager: Arc<inklog::LoggerManager>,
    security: SecurityStack,
    #[cfg(feature = "db")] _db_pool: crate::DbPool,
) -> anyhow::Result<()> {
    let audit_logger = security.audit_logger.clone();

    let (pipeline_queue, response_channel, priority_calculator, worker_manager) =
        init_pipeline(config, &service, &decision_service).await?;

    let kit = build_module_registry(
        config,
        service.clone(),
        rerank_service.clone(),
        decision_service.clone(),
        detect_service,
        translate_service,
        transcribe_service,
        security.rate_limiter.clone(),
        audit_logger.clone(),
        pipeline_queue.clone(),
        response_channel.clone(),
        priority_calculator.clone(),
        worker_manager.clone(),
        logger_manager,
        #[cfg(feature = "auth")]
        security.garrison_handle.clone(),
        #[cfg(feature = "auth")]
        security.garrison_csrf_config.clone(),
    )
    .await?;

    let shutdown_coordinator = AsyncShutdownCoordinator::new();
    shutdown_coordinator
        .set_global_timeout(Duration::from_secs(DEFAULT_SHUTDOWN_TIMEOUT_SECS))
        .map_err(|e| anyhow::anyhow!("Failed to set shutdown timeout: {}", e))?;
    register_shutdown_hooks(&shutdown_coordinator, &kit, &worker_manager)?;

    log::info!("AsyncKit module registry built successfully");

    let mut bg_tasks = tokio::task::JoinSet::<()>::new();

    spawn_config_watcher(&mut bg_tasks, config_path);

    // VecboostState 仅持有 kit 单字段，所有能力通过 kit.require 查询
    let app_state = VecboostState::new(kit);

    // 注入 state 到 api 模块（统一入口：所有 forge handler 通过 state().kit.require 访问）
    vecboost::api::init_state(app_state.clone()).map_err(|e| anyhow::anyhow!("{}", e))?;

    let app = build_http_router(config).await?;

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    log::info!("Server listening on {}", addr);

    // Signal-aware graceful shutdown (SIGINT + SIGTERM)
    let signal = install_signal_handlers();
    #[cfg(feature = "grpc")]
    if config.server.grpc_enabled {
        spawn_grpc_server(config, &mut bg_tasks).await?;
    }

    run_server_lifecycle(
        listener,
        app,
        signal,
        config,
        bg_tasks,
        shutdown_coordinator,
    )
    .await
}

#[cfg(feature = "http")]
/// 请求级观测/本地化中间件：Prometheus 指标记录（无条件应用）、
/// Accept-Language → 请求级 locale、axum Json extractor 拒绝体规范化
/// （审计建议：text/plain 错误改写为 handler 同构结构化 JSON）。
fn apply_request_middlewares(app: axum::Router, app_state: crate::VecboostState) -> axum::Router {
    // Prometheus 指标记录中间件 — 无条件应用到所有路由
    use axum::middleware::from_fn_with_state;
    let app = app.layer(from_fn_with_state(
        app_state.clone(),
        vecboost::metrics::metrics_middleware,
    ));

    // i18n Accept-Language 中间件 — 解析请求语言，设置请求级 locale
    // 使所有下游 handler 和 IntoResponse 自动使用正确的语言
    use axum::middleware::from_fn;
    let app = app.layer(from_fn(vecboost::i18n::i18n_middleware));

    // Extractor 拒绝规范化（审计建议）— axum Json extractor 拒绝的
    // text/plain 错误体改写为 handler 层同构的结构化错误 JSON
    app.layer(from_fn(
        vecboost::api::rejection_normalize::rejection_normalizer,
    ))
}

/// auth 中间件栈：全局限流（内部按 RateLimitEnabled 配置决定生效）→
/// 认证（路径白名单放行 /health、/api/1/auth/login、/api/1/auth/refresh）→
/// CSRF（auth+csrf 双启用时，garrison Origin+Token 双重校验）。
#[cfg(feature = "auth")]
fn apply_auth_middlewares(
    app: axum::Router,
    config: &VecboostConfig,
    app_state: crate::VecboostState,
) -> anyhow::Result<axum::Router> {
    use axum::middleware::from_fn_with_state;

    // 全局限流中间件 — 应用到所有路由
    // 内部通过 RateLimitEnabled 配置控制是否生效
    // 注：auth_rate_limit_middleware 定义在 auth 模块下，需 feature = "auth" 门控
    let app = app.layer(from_fn_with_state(
        app_state.clone(),
        vecboost::auth::auth_rate_limit_middleware,
    ));

    // auth_middleware：应用到所有路由，内部用路径白名单放行公开端点
    // (/health, /api/1/auth/login, /api/1/auth/refresh)
    let app = if config.auth.enabled {
        app.layer(from_fn_with_state(
            app_state.clone(),
            vecboost::auth::auth_middleware,
        ))
    } else {
        app
    };

    // CSRF 保护（条件性应用：auth 启用且 csrf 启用时）
    // 直接使用 garrison garrison_csrf_middleware（包含 Origin + Token 双重校验）
    // require 失败必须中止装配（fail-closed）：吞掉错误会以无 CSRF 防护的
    // app 静默起服——恢复拆分前 `?` 传播语义。
    if config.auth.enabled && config.auth.csrf.enabled {
        use axum::middleware::from_fn_with_state;
        let csrf_config = app_state
            .kit()
            .require::<CsrfConfigModule>()
            .map_err(|e| anyhow::anyhow!("Failed to require CsrfConfigModule: {}", e))?;
        if let Some(cfg) = csrf_config {
            Ok(app.layer(from_fn_with_state(cfg, garrison_csrf_middleware)))
        } else {
            Ok(app)
        }
    } else {
        Ok(app)
    }
}

/// 非 auth 构建的 auth 中间件栈恒等函数（保持调用点 cfg-free）。
#[cfg(not(feature = "auth"))]
fn apply_auth_middlewares(
    app: axum::Router,
    config: &VecboostConfig,
    app_state: crate::VecboostState,
) -> anyhow::Result<axum::Router> {
    let _ = (config, app_state);
    Ok(app)
}

/// 传输面层栈：请求体大小上限（默认 5 MiB，防超大 payload DoS）→ 全局
/// 请求超时 → 请求上下文（X-Request-Id/X-Trace-Id，W3C traceparent 兼容）
/// → trace → 安全响应头（auth 构建走 garrison web-security-headers；
/// 非 auth 构建保留手写头栈，HSTS 仅 garrison tls feature 注入）→ 压缩 →
/// CORS（配置开关，默认关闭）。
fn apply_transport_layers(app: axum::Router, config: &VecboostConfig) -> axum::Router {
    // 安全 headers + trace
    // 请求体大小上限(默认 5 MiB,防超大 payload DoS)
    let body_limit_bytes = config.server.body_limit_mb.max(1) as usize * 1024 * 1024;
    let app = app.layer(tower_http::limit::RequestBodyLimitLayer::new(
        body_limit_bytes,
    ));
    // axum 提取器层默认 body 上限（2MB）独立于上者：`Json`/`bytes` 提取器
    // 受 DefaultBodyLimit 约束，会先于 RequestBodyLimitLayer 拒绝——导致
    // `body_limit_mb > 2` 的配置静默失效（转录大样本 413 实测先例）。
    // 两层统一到同一配置口径，提取器层不再构成隐藏硬顶。
    let app = app.layer(axum::extract::DefaultBodyLimit::max(body_limit_bytes));
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

    app
}

async fn build_http_router(config: &VecboostConfig) -> anyhow::Result<axum::Router> {
    // app_state 由调用方保证已 init_state(所有能力经 kit.require 获取)
    let app_state = vecboost::api::state().map_err(|e| anyhow::anyhow!("{e}"))?;
    // 版本前缀:路由注册为 /api/1/*(forge version=1)。
    // 实施期决策:不启用 build_with_redirect —— 它会把 /api/1/* 重定向到
    // 不存在的 /api/v1/*,破坏全部 API 路由(实测 301)。
    let app = sdforge::http::build();

    // Swagger UI(/api-docs/openapi.json + /swagger-ui/) 与 ReDoc(/redoc)：
    // feature `swagger-ui` 显式 opt-in（默认构建不编译；挂主服务端口，
    // 单一二进制一次构建完成）
    #[cfg(feature = "swagger-ui")]
    let app = app
        .merge(sdforge::docs::swagger_ui_router())
        .merge(vecboost::api::redoc_router());

    // metrics 端点（手写例外：Prometheus text/plain 响应，forge 不支持非 JSON）
    let metrics_router = axum::Router::new()
        .route("/metrics", axum::routing::get(metrics_endpoint))
        .with_state(app_state.clone());
    let mut app = app.merge(metrics_router);

    // 404 fallback：未知路由返回结构化 JSON（与错误契约一致）
    app = app.fallback(|| async {
        use axum::http::{StatusCode, header};
        use axum::response::IntoResponse;
        let body = serde_json::json!({
            "type": "NotFound",
            "message": vecboost::i18n::tr("route-not-found"),
            "field": null,
            "value": null,
        });
        (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "application/json")],
            body.to_string(),
        )
            .into_response()
    });

    // Prometheus 指标记录/i18n/拒绝规范化中间件
    let app = apply_request_middlewares(app, app_state.clone());

    // auth 限流/认证/CSRF 中间件（内部按路径白名单放行公开端点）
    let app = apply_auth_middlewares(app, config, app_state)?;

    // 请求体上限/超时/请求上下文/trace/安全响应头/压缩/CORS
    let app = apply_transport_layers(app, config);

    // ConnectInfo is automatically available when using axum::serve with a TcpListener
    // No additional layer needed
    Ok(app)
}

/// 服务器生命周期 —— 阻塞等待退出信号,执行分级优雅关闭。
#[cfg(feature = "http")]
async fn run_server_lifecycle(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    signal: impl std::future::Future<Output = ()> + Send + 'static,
    _config: &VecboostConfig,
    mut bg_tasks: tokio::task::JoinSet<()>,
    shutdown_coordinator: AsyncShutdownCoordinator,
) -> anyhow::Result<()> {
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(signal)
    .await?;

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
    config: &VecboostConfig,
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
    // 会话校验（过期/吊销/防火墙），与 HTTP 侧共用同一 garrison 会话宇宙。
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
                match VecboostConfig::load_via_confers_with_path(&reload_path) {
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

/// 注册 16 个非 auth + 2 个 auth feature 的全部 Module（ConfigWatcher
/// 的 WatcherGuard 能力注入在此一并完成）。
fn register_all_modules(kit: &mut trait_kit::AsyncKit) -> anyhow::Result<()> {
    // 注册模块（16 个非 auth + 2 个 auth feature = 18 个 Module）
    kit.register::<EmbeddingModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register EmbeddingModule: {}", e))?;
    kit.register::<RateLimitModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RateLimitModule: {}", e))?;
    kit.register::<RerankModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register RerankModule: {}", e))?;
    kit.register::<DecisionModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "DecisionModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
    kit.register::<DetectModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[("module", "DetectModule"), ("detail", &e.to_string()),]),
            )
        )
    })?;
    kit.register::<TranslateModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "TranslateModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
    kit.register::<TranscribeModule>().map_err(|e| {
        anyhow::anyhow!(
            "{}",
            vecboost::i18n::tr_with_args(
                "startup-register-failed",
                vecboost::i18n::tr_args(&[
                    ("module", "TranscribeModule"),
                    ("detail", &e.to_string()),
                ]),
            )
        )
    })?;
    kit.register::<CacheModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register CacheModule: {}", e))?;
    kit.register::<DbModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register DbModule: {}", e))?;
    kit.register::<AuditModule>()
        .map_err(|e| anyhow::anyhow!("Failed to register AuditModule: {}", e))?;
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

    Ok(())
}

/// 构建模块注册中心(trait-kit AsyncKit)—— 预构建能力注入 + 18 个
/// Module 注册 + 生命周期/健康检查挂载。从 `app_main` 拆出。
#[allow(clippy::too_many_arguments)]
async fn build_module_registry(
    config: &VecboostConfig,
    service: Arc<RwLock<EmbeddingService>>,
    rerank_service: Arc<RwLock<RerankService>>,
    decision_service: Arc<RwLock<DecisionService>>,
    detect_service: DetectService,
    translate_service: Arc<TranslateService>,
    transcribe_service: Arc<TranscribeService>,
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

    kit.with_observer(Arc::new(LoggingObserver));

    // 注入预构建的能力对象（kit 是 single source of truth）— 已清理未被任何 Module/Handler 消费的冗余注入
    kit.set_config(service.clone());
    kit.set_config(rerank_service.clone());
    kit.set_config(decision_service.clone());
    kit.set_config(detect_service);
    kit.set_config(translate_service);
    kit.set_config(transcribe_service);
    kit.set_config(rate_limiter.clone());
    kit.set_config(CacheConfig {
        enabled: config.embedding.cache_enabled,
        size: config.embedding.cache_size,
    });
    kit.set_config(DbConfig {
        enabled: cfg!(feature = "db"),
    });
    kit.set_config(audit_logger.clone());
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
    // PipelineEnabled 真接线（装配三问）：embed/decision handler 的入队分支
    // 按此开关分流；此前仅库内 make_app_state 注入 false，生产 server 缺注入
    // 使 config() 恒 Err、队列路径不可达（P2-3 决策入队 e2e 实抓）
    kit.set_config(vecboost::registry::PipelineEnabled(config.pipeline.enabled));
    // LoggerModule: Arc<inklog::LoggerManager> 能力注入
    kit.set_config(logger_manager.clone());
    #[cfg(feature = "auth")]
    {
        kit.set_config(garrison_handle.clone());
        kit.set_config(garrison_csrf_config.clone());
    }

    register_all_modules(&mut kit)?;

    kit.register_lifecycle::<EmbeddingModule>();
    kit.register_lifecycle::<RerankModule>();
    kit.register_lifecycle::<DecisionModule>();
    kit.register_lifecycle::<DetectModule>();
    kit.register_lifecycle::<TranslateModule>();
    kit.register_lifecycle::<RateLimitModule>();
    kit.register_lifecycle::<AuditModule>();
    kit.register_lifecycle::<ConfigWatcherModule>();
    kit.register_health_check::<EmbeddingModule>();
    kit.register_health_check::<RerankModule>();
    kit.register_health_check::<DecisionModule>();
    kit.register_health_check::<DetectModule>();
    kit.register_health_check::<TranslateModule>();
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
        // FTL 文案断言前置:未 init 时 tr 回退键名,消息不含插值与环境变量名
        vecboost::i18n::init();
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

    // 装配三问①：[model].tokenizer_path 透传进引擎级 ModelConfig（spec
    // R-model-bundle-fetch-002），注入值原样可达引擎加载分支
    #[test]
    fn engine_model_config_passes_through_tokenizer_path() {
        let mut config = VecboostConfig::default();
        config.model.tokenizer_path = Some("models/laya/tokenizer/tokenizer.json".to_string());
        let model_config = resolve_engine_model_config(&config).expect("config resolves");
        assert_eq!(
            model_config.tokenizer_path.as_deref(),
            Some(std::path::Path::new("models/laya/tokenizer/tokenizer.json")),
            "[model].tokenizer_path 必须透传进引擎级 ModelConfig"
        );
    }

    // 未配置 tokenizer_path 时保持 None（探测走根目录 → tokenizer/ 子目录链）
    #[test]
    fn engine_model_config_tokenizer_path_defaults_to_none() {
        let config = VecboostConfig::default();
        let model_config = resolve_engine_model_config(&config).expect("config resolves");
        assert!(model_config.tokenizer_path.is_none());
    }

    // ── 装配三问②：[model.checkpoints] 预设表启动 fail-fast ──

    fn checkpoint_preset(name: &str) -> CheckpointPreset {
        CheckpointPreset {
            name: name.to_string(),
            model_path: std::path::PathBuf::from("models/laya"),
            task: ModelTask::Decision,
            engine_type: None,
            tokenizer_path: None,
            head_max_len: None,
            max_len: None,
        }
    }

    #[test]
    fn checkpoints_fail_fast_merges_builtin_into_empty_table() {
        let config = VecboostConfig::default();
        let resolved = resolve_checkpoints_fail_fast(&config).expect("空配置必须通过");
        assert_eq!(resolved.len(), 1, "内置 laya-multilingual 零配置可用");
        let builtin = &resolved[vecboost::config::model::BUILTIN_MULTILINGUAL_CHECKPOINT];
        assert_eq!(builtin.task, ModelTask::Decision);
        assert_eq!(builtin.head_max_len, Some(256));
        assert_eq!(builtin.max_len, Some(256));
    }

    #[test]
    fn checkpoints_fail_fast_accepts_valid_entry() {
        let mut config = VecboostConfig::default();
        config
            .model
            .checkpoints
            .insert("laya".to_string(), checkpoint_preset("laya"));
        let resolved = resolve_checkpoints_fail_fast(&config).expect("合法条目必须通过");
        assert_eq!(resolved.len(), 2, "内置 + 用户显式条目并存");
        assert_eq!(resolved["laya"].name, "laya");
        assert_eq!(resolved["laya"].task, ModelTask::Decision);
        // 用户显式配置同名条目覆盖内置（显式优先）
        let mut config = VecboostConfig::default();
        let mut preset =
            checkpoint_preset(vecboost::config::model::BUILTIN_MULTILINGUAL_CHECKPOINT);
        preset.head_max_len = Some(512);
        preset.max_len = Some(512);
        config.model.checkpoints.insert(preset.name.clone(), preset);
        let resolved = resolve_checkpoints_fail_fast(&config).expect("覆盖内置必须通过");
        assert_eq!(resolved.len(), 1, "同名覆盖不产生双条目");
        assert_eq!(
            resolved[vecboost::config::model::BUILTIN_MULTILINGUAL_CHECKPOINT].head_max_len,
            Some(512)
        );
    }

    #[test]
    fn checkpoints_fail_fast_rejects_key_name_mismatch() {
        let mut config = VecboostConfig::default();
        config
            .model
            .checkpoints
            .insert("wrong-key".to_string(), checkpoint_preset("laya"));
        let err = resolve_checkpoints_fail_fast(&config).unwrap_err();
        assert!(
            err.to_string().contains("[model.checkpoints]")
                && err.to_string().contains("wrong-key"),
            "启动 fail-fast 必须拒绝键名漂移并点名配置段与键，got {err}"
        );
    }

    // 内置名不得绕过成对闸门：单向写法经字段级 .or 继承会与内置 256 拼成
    // 合法对（head 384/state 256 非对称预算静默生效），merge 前必须对用户
    // 原始条目先跑成对校验，与非内置名同语义启动期拒绝
    #[test]
    fn checkpoints_fail_fast_rejects_one_sided_params_on_builtin_name() {
        let mut config = VecboostConfig::default();
        let mut preset =
            checkpoint_preset(vecboost::config::model::BUILTIN_MULTILINGUAL_CHECKPOINT);
        preset.head_max_len = Some(384);
        config.model.checkpoints.insert(preset.name.clone(), preset);
        let err = resolve_checkpoints_fail_fast(&config).unwrap_err();
        assert!(
            err.to_string().contains("[model.checkpoints]")
                && err.to_string().contains("head_max_len")
                && err.to_string().contains("must be set together"),
            "内置名单向 head_max_len 必须与非内置名同语义启动期拒绝，got {err}"
        );
    }

    #[test]
    fn checkpoints_fail_fast_rejects_unknown_engine_type() {
        // FTL 文案断言前置:未 init 时 tr 回退键名,消息不含非法取值插值
        vecboost::i18n::init();
        let mut config = VecboostConfig::default();
        let mut preset = checkpoint_preset("laya");
        preset.engine_type = Some("tensorrt".to_string());
        config.model.checkpoints.insert("laya".to_string(), preset);
        let err = resolve_checkpoints_fail_fast(&config).unwrap_err();
        assert!(
            err.to_string().contains("tensorrt"),
            "engine_type 非法取值必须启动期拒绝，got {err}"
        );
    }

    // tokenizer_path no-op 判定覆盖面：消费者有两类——onnx 本地 bundle
    // 分支（按 engine_type）与决策管线（task=decision 时按 engine_type
    // 分派后端，candle/onnx 两路均消费，判定本身不分 engine_type）；
    // 其余（candle embedding、onnx HF 拉取、任何非本地目录）一律 no-op
    // 纳入启动期 warn（防配置值静默失效无感知）
    #[test]
    fn tokenizer_path_noop_covers_candle_and_onnx_hf_pull() {
        let bundle = std::env::temp_dir().join(format!("vb_tok_noop_{}", std::process::id()));
        std::fs::create_dir_all(&bundle).expect("create bundle dir");
        let hf_repo = std::path::Path::new("BAAI/bge-small-en-v1.5");

        assert!(
            tokenizer_path_is_noop(&EngineType::Candle, &ModelTask::Embedding, &bundle),
            "candle embedding 路径无消费者，本地目录亦 no-op"
        );
        assert!(
            !tokenizer_path_is_noop(&EngineType::Candle, &ModelTask::Decision, &bundle),
            "决策管线按 engine_type 分派后端（candle/onnx 两路均消费），candle+decision+本地目录非 no-op"
        );
        assert!(
            tokenizer_path_is_noop(&EngineType::Candle, &ModelTask::Decision, hf_repo),
            "决策管线非本地路径无 bundle 可探测，no-op 告警保留"
        );

        #[cfg(feature = "onnx")]
        {
            assert!(
                !tokenizer_path_is_noop(&EngineType::Onnx, &ModelTask::Embedding, &bundle),
                "onnx 本地 bundle 目录（exists && is_dir）有消费者"
            );
            assert!(
                tokenizer_path_is_noop(&EngineType::Onnx, &ModelTask::Embedding, hf_repo),
                "onnx HF 拉取分支（非本地目录）必须按 no-op 告警"
            );
            assert!(
                !tokenizer_path_is_noop(&EngineType::Onnx, &ModelTask::Decision, &bundle),
                "onnx+decision+本地目录经决策臂消费，非 no-op"
            );
        }

        let _ = std::fs::remove_dir_all(&bundle);
    }

    // Swagger UI 挂载验证 —— openapi.json 与 UI 资源均可访问
    // （随 feature `swagger-ui` 门控：默认构建不编译 docs 链）
    #[cfg(feature = "swagger-ui")]
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

    // ReDoc 挂载验证 —— /redoc 返回内嵌 HTML（与 Swagger UI 共用 OpenAPI 规范）
    #[cfg(feature = "swagger-ui")]
    #[tokio::test]
    async fn redoc_endpoint_served_with_docs_router() {
        use tower::ServiceExt;
        let app = sdforge::docs::swagger_ui_router().merge(vecboost::api::redoc_router());

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/redoc")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let content_type = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default();
        assert!(
            content_type.contains("text/html"),
            "unexpected content-type: {content_type}"
        );
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
        let mut config = VecboostConfig::default();
        config.auth.enabled = true;
        config.auth.jwt_secret = Some("test-secret-0123456789abcdef0123".to_string());
        config.auth.default_admin_password = None;
        let err = init_auth(
            &config,
            #[cfg(feature = "db")]
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("VECBOOST_ADMIN_PASSWORD"));
    }

    /// 会话后端日志标签：database → garrison_kv；memory → "-"（无表）。
    #[cfg(feature = "auth")]
    #[test]
    fn session_backend_labels_map_memory_and_database() {
        use vecboost::config::app::SessionBackend;
        assert_eq!(
            session_backend_labels(SessionBackend::Memory),
            ("memory", "-")
        );
        assert_eq!(
            session_backend_labels(SessionBackend::Database),
            ("database", "garrison_kv")
        );
    }

    /// Redis endpoint 日志脱敏：剔除 userinfo/path/query。
    #[test]
    fn redis_endpoint_label_strips_credentials_and_path() {
        assert_eq!(redis_endpoint_label(None), "-");
        assert_eq!(
            redis_endpoint_label(Some("redis://user:pass@host:6380/0?x=1")),
            "redis://host:6380"
        );
        assert_eq!(redis_endpoint_label(Some("host:6379")), "redis://host:6379");
        // 密码含未编码分隔符：先剔 userinfo 再剔 path，密码片段不得入日志
        assert_eq!(
            redis_endpoint_label(Some("redis://user:pa/ss@host:6379")),
            "redis://host:6379"
        );
        assert_eq!(
            redis_endpoint_label(Some("redis://user:p?x#h@host:6379")),
            "redis://host:6379"
        );
    }
}
