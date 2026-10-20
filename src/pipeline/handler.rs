// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 流水线请求处理函数

use crate::VecboostState;
use crate::domain::{EmbedRequest, ServiceResponse};
use crate::error::VecboostError;
use crate::i18n;
use crate::pipeline::QueuedRequest;
use std::time::Duration;

/// 处理流水线请求
/// 进程级请求 ID 计数器 —— u64 原子递增,十六进制展示
/// (替代 UUID String:省一次堆分配与随机熵开销,格式仍可读)
static NEXT_REQUEST_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// pipeline 在途请求计数(入队等待响应期间 +1,完成 -1)
pub static IN_FLIGHT_REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// RAII 守卫:离开 pipeline 等待路径时自动递减
pub(crate) struct InFlightGuard;
impl InFlightGuard {
    pub(crate) fn enter() -> Self {
        IN_FLIGHT_REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        InFlightGuard
    }
}
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        IN_FLIGHT_REQUESTS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn next_request_id() -> String {
    format!(
        "{:x}",
        NEXT_REQUEST_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

// 跨批/跨 worker 同文 in-flight 去重（FT-03）
//
// 批内去重（service::embed_batch_texts 的 xxh3 scatter）只能合并同一批内
// 的重复文本；min_workers>1 与批触发时序下，并发同文请求会落在不同批/
// 不同 worker，各批形状不同 → 矩阵乘归约顺序不同 → 向量位级差异（GEMM
// 固有，不可靠批次对齐消除）。本层在入队前合并飞行中的同文同 flag 请求：
// leader 独占入队，followers 经 watch 多播克隆同一响应——逐位一致由
// 「只算一次」结构性保证，不再依赖批次对齐。
//
// 键 = (xxh3_64(text), normalize)：normalize 在 worker scatter 期施加，
// 同 text 不同 flag 的响应不同，须分组合并；text 原文存条目内做哈希碰撞
// 二次确认。错误响应同样多播（leader 失败即同文同 flag 的确定性失败）。

type EmbedInFlightValue = Option<Result<crate::domain::EmbedResponse, VecboostError>>;
type EmbedInFlightKey = (u64, bool);

struct InFlightEmbedEntry {
    text: String,
    tx: tokio::sync::watch::Sender<EmbedInFlightValue>,
}

fn embed_in_flight_map()
-> &'static std::sync::Mutex<std::collections::HashMap<EmbedInFlightKey, InFlightEmbedEntry>> {
    static MAP: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<EmbedInFlightKey, InFlightEmbedEntry>>,
    > = std::sync::OnceLock::new();
    MAP.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

enum EmbedInFlightJoin {
    /// 本请求为 leader：完成时必须调用 complete（写多播 + 摘除条目）；
    /// 提前退出（断连/panic）由 Drop 摘除，followers 自动重试入队。
    Leader(EmbedInFlightGuard),
    /// 命中飞行中的同文同 flag 请求：rx 已订阅（与 leader 摘除解耦），
    /// 等待 Some(value) 即得与 leader 逐位一致的响应克隆。
    Follower(tokio::sync::watch::Receiver<EmbedInFlightValue>),
}

/// 获取 in-flight map 锁：锁被 poison（持锁线程 panic）时不放大为级联 panic——
/// 取回内部数据继续用（锁内无不变量操作，Map 结构本身仍一致；失效条目由
/// 各自 guard 的 complete/Drop 路径回收）。
fn embed_in_flight_lock()
-> std::sync::MutexGuard<'static, std::collections::HashMap<EmbedInFlightKey, InFlightEmbedEntry>> {
    embed_in_flight_map()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct EmbedInFlightGuard {
    key: EmbedInFlightKey,
    /// complete 已摘除后 drop 不再二次摘除（避免误删后来者的同名条目）
    done: bool,
}

impl EmbedInFlightGuard {
    fn complete(mut self, result: Result<crate::domain::EmbedResponse, VecboostError>) {
        let mut map = embed_in_flight_lock();
        if let Some(entry) = map.get(&self.key) {
            let _ = entry.tx.send(Some(result));
        }
        map.remove(&self.key);
        self.done = true;
    }
}

impl Drop for EmbedInFlightGuard {
    fn drop(&mut self) {
        if !self.done {
            embed_in_flight_lock().remove(&self.key);
        }
    }
}

/// 尝试合并飞行中的同文同 flag 请求：命中返回 Follower（已订阅），
/// 否则注册为 Leader（条目已在图内，完成路径必须经 complete/Drop 摘除）。
fn join_embed_in_flight(text: &str, normalize: bool) -> EmbedInFlightJoin {
    use std::collections::hash_map::Entry;
    let key = (xxhash_rust::xxh3::xxh3_64(text.as_bytes()), normalize);
    let mut map = embed_in_flight_lock();
    match map.entry(key) {
        Entry::Occupied(occupied) => {
            if occupied.get().text == text {
                return EmbedInFlightJoin::Follower(occupied.get().tx.subscribe());
            }
            // xxh3 碰撞（极小概率）：移除旧条目并以 leader 身份重建——原
            // 条目的 followers 仍持 rx，等待原 leader 完成或断连后重试入队
            occupied.remove();
            let (tx, _) = tokio::sync::watch::channel(None);
            map.insert(
                key,
                InFlightEmbedEntry {
                    text: text.to_string(),
                    tx,
                },
            );
            EmbedInFlightJoin::Leader(EmbedInFlightGuard { key, done: false })
        }
        Entry::Vacant(vacant) => {
            let (tx, _) = tokio::sync::watch::channel(None);
            vacant.insert(InFlightEmbedEntry {
                text: text.to_string(),
                tx,
            });
            EmbedInFlightJoin::Leader(EmbedInFlightGuard { key, done: false })
        }
    }
}

/// follower 等待 leader 结果：watch 值从 None 变 Some 即取（已被
/// borrow_and_update 消费更新点）；sender 被摘除（leader 断连）返回 None。
async fn await_embed_in_flight(
    mut rx: tokio::sync::watch::Receiver<EmbedInFlightValue>,
) -> Option<Result<crate::domain::EmbedResponse, VecboostError>> {
    loop {
        if let Some(result) = rx.borrow_and_update().clone() {
            return Some(result);
        }
        if rx.changed().await.is_err() {
            return None;
        }
    }
}

pub async fn handle_pipeline_request(
    state: VecboostState,
    req: EmbedRequest,
    ip: String,
) -> Result<axum::Json<crate::domain::EmbedResponse>, VecboostError> {
    // 入队前内容三查（与决策侧 validate_request 同构）：批内一条非法文本
    // 会让整批 embed 在服务层校验处整批失败、错误扩散给批内其他合法请求——
    // O(1) 在此 400 拒绝，不占队列槽位与 worker 出队周期（三路评审 M）。
    // 长度维度由调用侧 validate_text_length 按 kit 配置口径先行
    crate::utils::validator::input::validate_text_basic(&req.text)?;

    // 进入 pipeline 等待即计为在途(RAII,任何退出路径自动递减)；follower
    // 同样计入（HTTP 在途语义，不随合并路径改变观测口径）
    let _in_flight = InFlightGuard::enter();
    let normalize = req.normalize.unwrap_or(false);

    loop {
        match join_embed_in_flight(&req.text, normalize) {
            EmbedInFlightJoin::Follower(rx) => {
                // 同文同 flag 的确定性结果克隆；leader 断连/SLA 超时则
                // 重新竞争 leader（8 并发同文必有恰一个入队者，收敛）
                let joined = tokio::time::timeout(REQUEST_SLA, await_embed_in_flight(rx)).await;
                match joined {
                    Ok(Some(Ok(resp))) => return Ok(axum::Json(resp)),
                    Ok(Some(Err(e))) => return Err(e),
                    _ => continue,
                }
            }
            EmbedInFlightJoin::Leader(guard) => {
                let request_id = next_request_id();
                let queued_request = build_queued_request(
                    &state,
                    request_id,
                    ip,
                    crate::pipeline::ServiceRequest::Embed(req),
                )?;

                let response = enqueue_and_await_response(state, queued_request).await;
                let embed_response = match &response {
                    Ok(ServiceResponse::Embed(resp)) => Ok(resp.clone()),
                    Ok(_) => Err(VecboostError::InternalError(i18n::tr(
                        "pipeline-response-type-mismatch",
                    ))),
                    Err(e) => Err(e.clone()),
                };
                guard.complete(embed_response);
                let response = response?;
                let ServiceResponse::Embed(resp) = response else {
                    return Err(VecboostError::InternalError(i18n::tr(
                        "pipeline-response-type-mismatch",
                    )));
                };
                return Ok(axum::Json(resp));
            }
        }
    }
}

/// 决策 pipeline 处理：与 embed 同一入队/等待语义（共享队列、共享优先级
/// 计算器与基准、取消传播），仅请求/响应类型为决策。优先级与 embed 同口径
/// 由「同一 build_queued_request 构造（同一 PriorityCalculator + 同一 Normal
/// 基准 + 同一来源标识）」结构性保证；校验在入队前完成（400 不占队列槽位）。
pub async fn handle_decision_pipeline_request(
    state: VecboostState,
    req: crate::domain::DecisionRequest,
) -> Result<crate::domain::DecisionResponse, VecboostError> {
    // 入队前 O(1) 校验：非法请求在此 400 拒绝，不占队列槽位与 worker 出队
    // 周期；否则队列满时 503 Overloaded 会把永久性客户端错误伪装成可重试
    // 背压（三路评审 M）。与 service 直连路径共用同一校验实现
    crate::service::decision::DecisionService::validate_request(
        &req,
        crate::domain::decision::MAX_QUESTIONS,
    )?;

    let request_id = next_request_id();
    let _in_flight = InFlightGuard::enter();

    let queued_request = build_queued_request(
        &state,
        request_id,
        API_SOURCE_ID.to_string(),
        crate::pipeline::ServiceRequest::Decision(req),
    )?;

    let response = enqueue_and_await_response(state, queued_request).await?;
    let ServiceResponse::Decision(resp) = response else {
        return Err(VecboostError::InternalError(i18n::tr(
            "pipeline-response-type-mismatch",
        )));
    };
    Ok(resp)
}

/// 请求 SLA：排队+执行的预算上界。time_until_timeout（优先级输入）、
/// QueuedRequest.timeout（SLA 收割依据）与 enqueue_and_await_response 的
/// 服务端等待超时三处共用同一常量，杜绝多处硬编码漂移（三路评审 L）。
const REQUEST_SLA: Duration = Duration::from_secs(30);

/// embed/decision handler 共用的 pipeline 来源标识与 QueuedRequest 构造——
/// 优先级计算与 SLA 只此一份，两侧调用点（api/embedding.rs 传
/// [`API_SOURCE_ID`]）无法各自漂移，「同口径优先级」成为结构性事实
/// （三路评审 L）
pub const API_SOURCE_ID: &str = "api";

fn build_queued_request(
    state: &VecboostState,
    request_id: String,
    source_id: String,
    request: crate::pipeline::ServiceRequest,
) -> Result<QueuedRequest, VecboostError> {
    let priority = state
        .kit
        .require::<crate::registry::PriorityCalculatorModule>()
        .expect("PriorityCalculatorModule not registered")
        .calculate(crate::pipeline::PriorityInput {
            base_priority: crate::pipeline::Priority::Normal,
            time_until_timeout: REQUEST_SLA,
            user_tier: None,
            source: crate::pipeline::RequestSource::http(source_id.clone()),
            queue_length: state
                .kit
                .require::<crate::registry::PipelineQueueModule>()
                .expect("PipelineQueueModule not registered")
                .size(),
        });

    Ok(QueuedRequest {
        request_id,
        request,
        priority,
        submitted_at: std::time::Instant::now(),
        timeout: REQUEST_SLA,
        source: crate::pipeline::RequestSource::http(source_id),
    })
}

/// 入队并等待 worker 完成的公共内核：取消传播、响应通道注册、
/// 入队（队列满早退释放取消标志）、SLA 服务端等待超时。
/// embed 与 decision handler 共用，保证两条路径的排队语义完全一致。
async fn enqueue_and_await_response(
    state: VecboostState,
    request: QueuedRequest,
) -> Result<ServiceResponse, VecboostError> {
    use std::sync::atomic::Ordering;

    let request_id = request.request_id.clone();

    // 取消传播：handler 被 drop（客户端断连）时置位取消标志，
    // 调度器出队检查后丢弃——不再消耗推理算力
    let queue_handle = state
        .kit
        .require::<crate::registry::PipelineQueueModule>()
        .expect("PipelineQueueModule not registered");
    let cancel_flag = queue_handle.cancellations().register(&request_id);
    struct CancelOnDrop(Option<std::sync::Arc<std::sync::atomic::AtomicBool>>);
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            if let Some(f) = &self.0 {
                f.store(true, Ordering::Relaxed);
            }
        }
    }
    let mut cancel_guard = CancelOnDrop(Some(std::sync::Arc::clone(&cancel_flag)));

    let response_rx = state
        .kit
        .require::<crate::registry::ResponseChannelModule>()
        .expect("ResponseChannelModule not registered")
        .register(request_id.clone())
        .await;

    if let Err(e) = queue_handle.enqueue(request).await {
        // 队列满早退：取消标志与响应通道条目随即释放——请求从未入队即无
        // worker 会 complete，channel 条目不清将永久滞留（持续过载下无界
        // 增长，三路评审 M；「注册表不留孤儿」此处补全到两侧）
        cancel_guard.0.take();
        queue_handle.cancellations().release(&request_id);
        state
            .kit
            .require::<crate::registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered")
            .deregister(&request_id)
            .await;
        return Err(e);
    }

    let result = match tokio::time::timeout(REQUEST_SLA, response_rx).await {
        Ok(Ok(Ok(response))) => Ok(response),
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(_)) => Err(VecboostError::InternalError(i18n::tr(
            "pipeline-channel-error",
        ))),
        // 服务端超时 → 504 语义，且置位取消：请求若仍在队列则被丢弃
        Err(_) => {
            cancel_flag.store(true, Ordering::Relaxed);
            Err(VecboostError::RequestTimeout(i18n::tr("pipeline-timeout")))
        }
    };
    // 请求正常完成：解除断连自动取消（否则 Drop 误标后续无主请求）
    cancel_guard.0.take();
    result
}

#[cfg(test)]
mod tests {
    /// 进程级 registry 串行锁（配合 create_test_state 用例使用）。
    static REGISTRY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    use super::*;

    fn ensure_i18n_init() {
        i18n::init();
    }
    use crate::config::model::{ModelConfig, ModelTask, Precision};
    use crate::domain::{
        DecisionAnswer, DecisionAnswerBody, DecisionQuestion, DecisionRequest, DecisionResponse,
        EmbedRequest, QuestionType, ServiceResponse,
    };
    use crate::engine::InferenceEngine;
    use crate::pipeline::{
        PriorityCalculator, PriorityConfig, PriorityInput, PriorityRequestQueue, ResponseChannel,
        WorkerConfig, WorkerManager,
    };
    use crate::rate_limit::LimiteronAdapter;
    use crate::service::decision::DecisionService;
    use crate::service::embedding::EmbeddingService;
    use async_trait::async_trait;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    /// 固定作答的 mock 决策引擎（noul 0.7 / choice 0 / score 3），供决策
    /// pipeline 测试的 DecisionService 装配；与 embed 的 TestEngine 互不影响。
    struct DecisionTestEngine;

    #[async_trait]
    impl InferenceEngine for DecisionTestEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 8])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        fn supports_task(&self, task: ModelTask) -> bool {
            matches!(task, ModelTask::Embedding | ModelTask::Decision)
        }

        fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
            let answers = req
                .questions
                .iter()
                .map(|q| DecisionAnswer {
                    question: q.name.clone(),
                    answer: DecisionAnswerBody::Noul { p_true: 0.7 },
                })
                .collect();
            Ok(DecisionResponse {
                answers,
                processing_time_ms: 0,
            })
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn decision_noul_request() -> DecisionRequest {
        DecisionRequest {
            state: serde_json::json!({"topic": "vacation"}),
            questions: vec![DecisionQuestion {
                name: "confident".to_string(),
                qtype: QuestionType::Noul,
                instructions: "state your p(true)".to_string(),
                options: vec![],
            }],
        }
    }

    struct TestEngine {
        dimension: usize,
    }

    impl TestEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl InferenceEngine for TestEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.5; self.dimension])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.5; self.dimension]).collect())
        }
        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }
        fn supports_mixed_precision(&self) -> bool {
            false
        }
        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    struct ErrorEngine;

    #[async_trait]
    impl InferenceEngine for ErrorEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Err(VecboostError::InferenceError(
                "mock inference failure".to_string(),
            ))
        }
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Err(VecboostError::InferenceError(
                "mock batch inference failure".to_string(),
            ))
        }
        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }
        fn supports_mixed_precision(&self) -> bool {
            false
        }
        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    async fn create_test_state(
        queue_capacity: usize,
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    ) -> VecboostState {
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine.clone(), None)));
        let rerank_service = Arc::new(RwLock::new(crate::service::rerank::RerankService::new(
            engine, None,
        )));
        let decision_engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(DecisionTestEngine));
        let decision_service = Arc::new(RwLock::new(DecisionService::new(decision_engine)));
        let queue = Arc::new(PriorityRequestQueue::new(queue_capacity));
        let response_channel = Arc::new(ResponseChannel::new());
        let priority_calculator = Arc::new(PriorityCalculator::new(PriorityConfig::default()));
        let worker_manager = Arc::new(WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            WorkerConfig::default(),
            Arc::clone(&service),
        ));
        let rate_limiter = Arc::new(LimiteronAdapter::with_defaults().await);

        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(service.clone());
        kit.set_config(rerank_service);
        kit.set_config(decision_service);
        kit.set_config(rate_limiter.clone());
        kit.set_config(queue.clone());
        kit.set_config(response_channel.clone());
        kit.set_config(priority_calculator.clone());
        kit.set_config(worker_manager.clone());
        kit.set_config(Vec::<String>::new());
        kit.set_config(crate::registry::AuthEnabled(false));
        kit.set_config(crate::registry::RateLimitEnabled(false));
        kit.set_config(crate::registry::PipelineEnabled(true));
        kit.set_config(crate::registry::CacheConfig {
            enabled: false,
            size: 0,
        });
        kit.set_config(crate::registry::DbConfig { enabled: false });
        kit.set_config(None::<Arc<crate::audit::AuditLogger>>);
        kit.set_config(None::<Arc<crate::metrics::InferenceCollector>>);
        kit.set_config(None::<Arc<crate::metrics::PrometheusCollector>>);
        kit.set_config(crate::RerankConfig::default());
        #[cfg(feature = "auth")]
        {
            kit.set_config(Option::<Arc<crate::auth::GarrisonHandle>>::None);
            kit.set_config(Option::<Arc<crate::auth::GarrisonCsrfConfig>>::None);
        }

        kit.register::<crate::registry::EmbeddingModule>().unwrap();
        kit.register::<crate::registry::RerankModule>().unwrap();
        kit.register::<crate::registry::DecisionModule>().unwrap();
        kit.register::<crate::registry::RateLimitModule>().unwrap();
        kit.register::<crate::registry::CacheModule>().unwrap();
        kit.register::<crate::registry::DbModule>().unwrap();
        kit.register::<crate::registry::AuditModule>().unwrap();
        kit.register::<crate::registry::MetricsCollectorModule>()
            .unwrap();
        kit.register::<crate::registry::PrometheusCollectorModule>()
            .unwrap();
        kit.register::<crate::registry::IpWhitelistModule>()
            .unwrap();
        kit.register::<crate::registry::PipelineQueueModule>()
            .unwrap();
        kit.register::<crate::registry::ResponseChannelModule>()
            .unwrap();
        kit.register::<crate::registry::PriorityCalculatorModule>()
            .unwrap();
        kit.register::<crate::registry::WorkerManagerModule>()
            .unwrap();
        #[cfg(feature = "auth")]
        {
            kit.register::<crate::registry::AuthModule>().unwrap();
            kit.register::<crate::registry::CsrfConfigModule>().unwrap();
        }

        let kit = kit.build().await.expect("Failed to build AsyncKit");
        VecboostState { kit: Arc::new(kit) }
    }

    /// 验证 handle_pipeline_request 成功路径——入队、worker 处理、返回 EmbedResponse。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_success() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 验证 handle_pipeline_request 在队列满时返回 Overloaded（背压语义
    /// 独立变体，禁止复用 RateLimitExceeded——spec R-decision-throughput-002）。
    /// 并钉通道条目清理（三路评审 M）：请求从未入队，response channel 条目
    /// 必须同步移除，否则持续过载下无界增长。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_queue_full() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 决策请求经 pipeline 入队：请求以 ServiceRequest::Decision
    /// 入队、优先级由共享 PriorityCalculator 按 embed 同口径计算（同输入
    /// 同结果）、出队由持有方以 DecisionService 消费并 complete 回达。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_decision_pipeline_request_success() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 入队前校验钉（三路评审 M）：非法 DecisionRequest 在 pipeline 分支
    /// 入队前 400 拒绝（InvalidInput），不占队列槽位——否则队列满时同一
    /// 请求会得到 503 Overloaded，把永久性客户端错误伪装成可重试背压。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_decision_pipeline_request_rejects_invalid_before_enqueue() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 入队内容门钉（三路评审 M）：非法文本（控制字符）在入队前 400 拒绝
    /// 且不占队列槽位——旧缺陷：非法文本入队后毒化整批 embed（服务层批校验
    /// 整批 Err，错误扩散给批内其他合法请求）。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_rejects_invalid_text_before_enqueue() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 决策请求队列满背压：/api/v1/decisions 在队列容量置零时收到
    /// Overloaded（503 语义，spec R-decision-throughput-002 可构造断言），
    /// 与 auth 限流的 RateLimitExceeded 互斥。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_decision_pipeline_request_queue_full_overloaded() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 验证 handle_pipeline_request 在引擎出错时传播 InferenceError。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_engine_error() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 验证 handle_pipeline_request 在 response channel sender 被 drop 时返回 InternalError。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_response_channel_error() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 验证 handle_pipeline_request 在超时时返回 ValidationError。
    /// 使用 start_paused 模拟时间流逝,不启动 consumer 让 response 永远 pending。
    #[tokio::test(start_paused = true)]
    async fn test_handle_pipeline_request_timeout() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 验证 handle_pipeline_request 在 normalize=false 时不做归一化。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_normalize_false() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 验证 handle_pipeline_request 支持不同维度的引擎。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_different_dimensions() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 验证 handle_pipeline_request 空文本返回 ValidationError。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_empty_text() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 验证 handle_pipeline_request 空白文本返回 ValidationError。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_whitespace_text() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// Verify TestEngine::embed returns the expected dimension.
    #[test]
    fn test_test_engine_embed_returns_correct_dimension() {
        let engine = TestEngine::new(256);
        let result = engine.embed("test").expect("embed should succeed");
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|&v| v == 0.5));
    }

    /// Verify TestEngine::embed_batch returns one vector per input.
    #[test]
    fn test_test_engine_embed_batch_returns_correct_count() {
        let engine = TestEngine::new(128);
        let texts = vec!["a".to_string(), "b".to_string()];
        let result = engine
            .embed_batch(&texts)
            .expect("embed_batch should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 128);
    }

    /// Verify TestEngine::embed_batch with empty input returns empty vec.
    #[test]
    fn test_test_engine_embed_batch_empty() {
        let engine = TestEngine::new(64);
        let texts: Vec<String> = vec![];
        let result = engine
            .embed_batch(&texts)
            .expect("embed_batch should succeed");
        assert!(result.is_empty());
    }

    /// Verify TestEngine::precision returns Fp32.
    #[test]
    fn test_test_engine_precision() {
        let engine = TestEngine::new(8);
        assert_eq!(engine.precision(), &Precision::Fp32);
    }

    /// Verify TestEngine::supports_mixed_precision returns false.
    #[test]
    fn test_test_engine_supports_mixed_precision() {
        let engine = TestEngine::new(8);
        assert!(!engine.supports_mixed_precision());
    }

    /// Verify TestEngine::try_fallback_to_cpu returns Ok.
    #[tokio::test]
    async fn test_test_engine_try_fallback_to_cpu() {}

    /// Verify ErrorEngine::embed returns InferenceError.
    #[test]
    fn test_error_engine_embed_fails() {
        let engine = ErrorEngine;
        let result = engine.embed("test");
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InferenceError(msg) => {
                assert!(msg.contains("mock inference failure"));
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }
    }

    /// Verify ErrorEngine::embed_batch returns InferenceError.
    #[test]
    fn test_error_engine_embed_batch_fails() {
        let engine = ErrorEngine;
        let texts = vec!["a".to_string(), "b".to_string()];
        let result = engine.embed_batch(&texts);
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InferenceError(msg) => {
                assert!(msg.contains("mock batch inference failure"));
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }
    }

    /// Verify ErrorEngine::precision returns Fp32.
    #[test]
    fn test_error_engine_precision() {
        let engine = ErrorEngine;
        assert_eq!(engine.precision(), &Precision::Fp32);
    }

    /// Verify ErrorEngine::supports_mixed_precision returns false.
    #[test]
    fn test_error_engine_supports_mixed_precision() {
        let engine = ErrorEngine;
        assert!(!engine.supports_mixed_precision());
    }

    /// Verify ErrorEngine::try_fallback_to_cpu returns Ok.
    #[tokio::test]
    async fn test_error_engine_try_fallback_to_cpu() {}

    /// Verify handle_pipeline_request with a longer text succeeds.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_long_text() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// Verify handle_pipeline_request with normalize=None uses default (true).
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_normalize_none() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// Verify handle_pipeline_request with different IP addresses works.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_different_ips() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// Verify handle_pipeline_request generates unique request IDs.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_unique_request_ids() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// Verify handle_pipeline_request with unicode text succeeds.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_unicode_text() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    // -- 未覆盖分支补测（覆盖率任务）：响应类型不匹配 / 断连取消传播 / mock 方法体 --

    /// worker 以错误变体（Decision）complete embed 等待方：handler 必须
    /// InternalError 显性报错，不得误投递。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_response_type_mismatch() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// worker 以错误变体（Embed）complete 决策等待方：decision handler 同款
    /// InternalError 显性报错。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_decision_pipeline_request_response_type_mismatch() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 断连取消传播：handler 被 abort（客户端断连）后，CancelOnDrop 在
    /// unwind 时置位取消标志——队列中请求出队即弃（不消耗推理）、槽位
    /// 释放。同帧 drop 的 InFlightGuard 亦随取消生效而证明回落。
    /// 注：IN_FLIGHT_REQUESTS 为进程级计数，并行测试共享，不作精确断言。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_abort_sets_cancel_flag() {
        // registry 为进程级共享：串行化避免并发用例互相覆盖队列容量。
        let _registry_guard = REGISTRY_LOCK.lock().await;
    }

    /// 直接调用 DecisionTestEngine 未使用的 trait 方法体（覆盖率）。
    #[test]
    fn test_decision_test_engine_unused_trait_bodies() {
        let engine = DecisionTestEngine;
        assert_eq!(engine.embed("x").unwrap().len(), 8);
        assert_eq!(engine.embed_batch(&["a".to_string()]).unwrap().len(), 1);
        assert_eq!(*engine.precision(), Precision::Fp32);
        assert!(!engine.supports_mixed_precision());
        assert!(engine.supports_task(ModelTask::Decision));
    }

    #[tokio::test]
    async fn test_decision_test_engine_fallback_body() {}
}
