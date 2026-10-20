// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Translate 服务实现：(src,tgt) 精确路由 + auto 检测路由。

use crate::config::model::TranslationParams;
use crate::domain::{TranslateRequest, TranslateResponse};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// 单个语言对的路由项：pair 引擎句柄 + 该 pair 生效的翻译参数 + 并发背压
/// 信号量（评审 RES-1/PERF-1：同 pair 引擎被 Mutex 串行化，无背压时 N 个
/// 并发请求占 N-1 个 blocking 线程空等；permits=1 使排队显性化——
/// try_acquire 失败即 503 Overloaded，与队列满口径统一）。
/// Clone 仅为路由表查询处 `get.cloned` 的句柄语义（Arc 克隆非引擎克隆）。
#[derive(Clone)]
pub struct TranslatePairRoute {
    pub engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    pub params: TranslationParams,
    /// 并发背压许可（permits=1：同 pair 严格串行，等待者不占推理资源）
    pub semaphore: Arc<tokio::sync::Semaphore>,
}

impl TranslatePairRoute {
    pub fn new(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        params: TranslationParams,
    ) -> Self {
        Self {
            engine,
            params,
            semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }
}

/// 翻译服务：多语言对路由表。
///
/// **不设 `replace_engine`**（proposal Non-Goal / design D6，编译期保证）：
/// switch_model 传播的是主模型 `engine_handle` 克隆，注入 (src,tgt) 多引擎
/// 路由表要么使全部语言对指向同一引擎、要么在主模型为 embedding 任务时注入
/// `supports_task(Translate)=false` 的引擎——翻译能力被切换操作静默废掉。
/// pair 引擎不参与 checkpoint 预设表与 switch_model 换型语义。
pub struct TranslateService {
    /// 路由键全小写：配置面与请求两侧均被同一谓词（`^[a-z]{2,3}$`）拦截，
    /// 大写码在域层 validate 即 ValidationError，无大小写归一路径
    pairs: HashMap<(String, String), TranslatePairRoute>,
    /// 已配置对清单（"en-zh" 展示形态，错误消息回显用）
    configured_pairs: Vec<String>,
}

impl TranslateService {
    /// 从路由表构建（启动期急切构建，经 EngineFactory 逐 pair 创建；
    /// 零 pairs = 空表服务，请求以 translate-not-configured 400 显性拒绝）。
    pub fn new(pairs: HashMap<(String, String), TranslatePairRoute>) -> Self {
        let mut configured_pairs: Vec<String> =
            pairs.keys().map(|(s, t)| format!("{s}-{t}")).collect();
        configured_pairs.sort();
        Self {
            pairs,
            configured_pairs,
        }
    }

    /// 已配置对清单（"en-zh" 展示形态，只读视图；日志/诊断用）
    pub fn configured_pairs(&self) -> &[String] {
        &self.configured_pairs
    }

    /// 翻译本体：域层校验 → auto 检测（None 时）→ (src,tgt) 精确路由 →
    /// 能力门 → token 预算预判 → spawn_blocking 推理。
    ///
    /// # 热路径计数挂点
    /// inc_translate_requests 位于全部 400 出口之后（成功路径计数，spec
    /// R-observability-002：仅通过校验链且通过业务判定的请求 +1）；
    /// global collector 为 None 时零开销跳过。
    ///
    /// # spawn_blocking 依据（与 detect 的差异）
    /// pair 引擎 translate 为阻塞模型推理（seq2seq 贪心解码，数十至数百毫秒
    /// 乃至 max_new_tokens 界定的秒级），按 [`InferenceEngine::translate`]
    /// 调用方契约移出 tokio worker 调度面；JoinError（引擎 panic）显性映射
    /// InferenceError，wire 文案为固定脱敏前缀，panic 载荷原文只入服务端日志
    /// （CWE-209，decision 同款）。
    pub async fn process_translate(
        &self,
        req: TranslateRequest,
    ) -> Result<TranslateResponse, VecboostError> {
        Self::validate_request(&req)?;

        // auto 检测：与 /api/1/detect 共用同一检测核心（单一事实源）
        // 零配置显性语义（spec R-translation-routing-003）：空表 = 未配置，
        // 任意请求 400 translate-not-configured（端点已注册，非 panic 非 500）
        if self.pairs.is_empty() {
            return Err(VecboostError::InvalidInput(
                crate::i18n::tr("translate-not-configured").to_string(),
            ));
        }

        let mut detected: Option<String> = None;
        let source_language = match &req.source_language {
            Some(src) => src.clone(),
            None => {
                let detection = crate::text::detect_language(&req.text, None);
                if !detection.is_reliable || detection.language.is_empty() {
                    return Err(VecboostError::InvalidInput(
                        crate::i18n::tr("translate-detect-failed").to_string(),
                    ));
                }
                detected = Some(detection.language.clone());
                detection.language
            }
        };

        let route = self
            .pairs
            .get(&(source_language.clone(), req.target_language.clone()))
            .cloned();
        let Some(route) = route else {
            let pairs_display = if self.configured_pairs.is_empty() {
                "none".to_string()
            } else {
                self.configured_pairs.join(", ")
            };
            return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
                "translate-unsupported-pair",
                crate::i18n::tr_args(&[
                    (
                        "requested",
                        &format!("{source_language}-{}", req.target_language),
                    ),
                    ("pairs", &pairs_display),
                ]),
            )));
        };

        // 并发背压（评审 RES-1/PERF-1）：try_acquire 失败即 503 Overloaded
        // 显性拒绝（非无限排队占 blocking 线程）；acquire 后经 spawn_blocking
        // 全程持有，drop 自动释放
        let _permit = route.semaphore.clone().try_acquire_owned().map_err(|_| {
            VecboostError::Overloaded(
                "translate pair is busy with another request; retry shortly".to_string(),
            )
        })?;

        // 能力门（先门后调）：路由表内的引擎必须支持 Translate 任务
        {
            let guard = route.engine.read().await;
            if !guard.supports_task(crate::config::model::ModelTask::Translate) {
                return Err(VecboostError::unsupported_task(
                    crate::i18n::tr("translate-unsupported").to_string(),
                ));
            }
        }

        // token 预算预判（服务层）：引擎 count_tokens 精确计数超 pair 上限
        // 显性拒绝（引擎侧 encode 后二次校验为第二道防线，双层口径）
        {
            let guard = route.engine.read().await;
            let count = guard.count_tokens(&req.text)?;
            if count > route.params.max_source_tokens {
                return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
                    "translate-too-many-source-tokens",
                    crate::i18n::tr_args(&[
                        ("count", &count.to_string()),
                        ("max", &route.params.max_source_tokens.to_string()),
                    ]),
                )));
            }
        }

        // 热路径埋点：计数点在全部 400 出口之后（成功路径计数，spec
        // R-observability-002 口径——零配置/检测失败/路由未命中/能力门/
        // 超 token 各 400 均不计；全局 collector 未设置时零开销跳过）
        #[cfg(feature = "http")]
        if let Some(collector) = crate::metrics::prometheus_exporter::global_collector() {
            collector.inc_translate_requests();
        }

        let start = Instant::now();
        let engine = route.engine.clone();
        let mut response = tokio::task::spawn_blocking(move || {
            let guard = engine.blocking_read();
            guard.translate(&req)
        })
        .await
        .map_err(|e| {
            // JoinError 的 Display 携带 panic 载荷原文：wire 侧只回可定位的
            // 固定文案，载荷与 task id 全量留在服务端日志（CWE-209）
            log::error!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-translation-inference-task-failed",
                    crate::i18n::tr_args(&[("e", &(e).to_string())])
                )
            );
            VecboostError::InferenceError(crate::i18n::tr("translation-inference-task-panicked"))
        })??;
        response.processing_time_ms = start.elapsed().as_millis() as u64;
        response.detected_source_language = detected;
        Ok(response)
    }

    /// 翻译请求前置校验的单一实现（直连路径共用）：空文本/超长以 i18n
    /// 文案 InvalidInput 显性拒绝（translate-empty-text / translate-too-long，
    /// detect 服务同款先例），随后域层 validate 兜底（控制字符/语言码格式/
    /// source≠target，ValidationError 单一实现）。
    pub fn validate_request(req: &TranslateRequest) -> Result<(), VecboostError> {
        if req.text.is_empty() {
            return Err(VecboostError::InvalidInput(
                crate::i18n::tr("translate-empty-text").to_string(),
            ));
        }
        let text_chars = req.text.chars().count();
        if text_chars > crate::utils::constants::MAX_TEXT_LENGTH {
            return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
                "translate-too-long",
                crate::i18n::tr_args(&[
                    ("length", &text_chars.to_string()),
                    ("max", &crate::utils::constants::MAX_TEXT_LENGTH.to_string()),
                ]),
            )));
        }
        req.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelConfig, ModelTask, Precision};
    use crate::domain::TranslateResponse;
    use async_trait::async_trait;

    /// 可辨识来源的 mock 翻译引擎：译文携带路由标记，count_tokens 按字符计。
    struct MockTranslateEngine {
        marker: &'static str,
        panic_on_translate: bool,
    }

    #[async_trait]
    impl InferenceEngine for MockTranslateEngine {
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
            matches!(task, ModelTask::Embedding | ModelTask::Translate)
        }

        fn count_tokens(&self, text: &str) -> Result<usize, VecboostError> {
            Ok(text.chars().count())
        }

        fn translate(&self, req: &TranslateRequest) -> Result<TranslateResponse, VecboostError> {
            if self.panic_on_translate {
                panic!("mock translation engine exploded mid-inference");
            }
            Ok(TranslateResponse {
                translation: format!(
                    "[{}→{}] {}",
                    req.source_language.as_deref().unwrap_or("auto"),
                    self.marker,
                    req.text
                ),
                source_language: req.source_language.clone(),
                target_language: req.target_language.clone(),
                detected_source_language: None,
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

    /// 不支持 translate 任务的引擎（能力门测试）
    struct EmbeddingOnlyEngine;

    #[async_trait]
    impl InferenceEngine for EmbeddingOnlyEngine {
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

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn route(marker: &'static str) -> TranslatePairRoute {
        TranslatePairRoute::new(
            Arc::new(RwLock::new(MockTranslateEngine {
                marker,
                panic_on_translate: false,
            })),
            TranslationParams::default(),
        )
    }

    fn make_service() -> TranslateService {
        let mut pairs = HashMap::new();
        pairs.insert(
            ("en".to_string(), "zh".to_string()),
            TranslatePairRoute::new(
                Arc::new(RwLock::new(MockTranslateEngine {
                    marker: "en-zh",
                    panic_on_translate: false,
                })),
                TranslationParams::default(),
            ),
        );
        pairs.insert(("en".to_string(), "es".to_string()), route("en-es"));
        TranslateService::new(pairs)
    }

    fn req(src: Option<&str>, tgt: &str, text: &str) -> TranslateRequest {
        TranslateRequest {
            text: text.to_string(),
            source_language: src.map(|s| s.to_string()),
            target_language: tgt.to_string(),
        }
    }

    #[tokio::test]
    async fn test_explicit_pair_routes_to_matching_engine() {
        crate::i18n::init();
        let svc = make_service();
        let resp = svc
            .process_translate(req(Some("en"), "es", "good morning"))
            .await
            .unwrap();
        assert!(
            resp.translation.contains("en-es"),
            "explicit 路由必须命中 en-es 引擎: {}",
            resp.translation
        );
        assert_eq!(resp.detected_source_language, None, "explicit 不回填检测");
    }

    #[tokio::test]
    async fn test_auto_detect_routes_and_backfills() {
        crate::i18n::init();
        let svc = make_service();
        let resp = svc
            .process_translate(req(
                None,
                "zh",
                "The quick brown fox jumps over the lazy dog near the riverbank every morning.",
            ))
            .await
            .unwrap();
        assert_eq!(
            resp.detected_source_language.as_deref(),
            Some("en"),
            "auto 检测语言必须回填"
        );
        assert!(
            resp.translation.contains("en-zh"),
            "auto 路由必须命中 en-zh 引擎"
        );
        assert_eq!(resp.source_language, None, "请求回显保持 null（auto 语义）");
    }

    #[tokio::test]
    async fn test_unreliable_detection_rejected_400() {
        crate::i18n::init();
        let svc = make_service();
        let err = svc
            .process_translate(req(None, "zh", "1234567890"))
            .await
            .unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                // 断言消息即 i18n 键渲染结果（locale 无关；FTL 落键后为
                // 本地化文案，缺键期为键名——两者都是键接线的证明）
                assert_eq!(
                    msg,
                    crate::i18n::tr("translate-detect-failed"),
                    "不可靠检测必须消费 translate-detect-failed 键，got {msg}"
                );
            }
            other => panic!("Expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_unconfigured_pair_400_with_pairs_list() {
        crate::i18n::init();
        let svc = make_service();
        let err = svc
            .process_translate(req(Some("en"), "fr", "bonjour"))
            .await
            .unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("en-fr"), "消息必须含请求对，msg={msg}");
                assert!(
                    msg.contains("en-zh") && msg.contains("en-es"),
                    "消息必须含已配置 pairs 全集，msg={msg}"
                );
            }
            other => panic!("Expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_zero_pairs_returns_not_configured() {
        crate::i18n::init();
        let svc = TranslateService::new(HashMap::new());
        let err = svc
            .process_translate(req(Some("en"), "zh", "hello"))
            .await
            .unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                assert_eq!(
                    msg,
                    crate::i18n::tr("translate-not-configured"),
                    "零配置必须消费 translate-not-configured 键，got {msg}"
                );
            }
            other => panic!("Expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_capability_gate_rejects_embedding_only_engine() {
        crate::i18n::init();
        let mut pairs = HashMap::new();
        pairs.insert(
            ("en".to_string(), "zh".to_string()),
            TranslatePairRoute::new(
                Arc::new(RwLock::new(EmbeddingOnlyEngine)),
                TranslationParams::default(),
            ),
        );
        let svc = TranslateService::new(pairs);
        let err = svc
            .process_translate(req(Some("en"), "zh", "hello"))
            .await
            .unwrap_err();
        match err {
            VecboostError::UnsupportedTask(msg) => {
                assert_eq!(
                    msg,
                    crate::i18n::tr("translate-unsupported"),
                    "能力门必须消费 translate-unsupported 键，got {msg}"
                );
            }
            other => panic!("Expected UnsupportedTask, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_over_budget_source_tokens_rejected_400() {
        crate::i18n::init();
        let mut pairs = HashMap::new();
        pairs.insert(
            ("en".to_string(), "zh".to_string()),
            TranslatePairRoute::new(
                Arc::new(RwLock::new(MockTranslateEngine {
                    marker: "en-zh",
                    panic_on_translate: false,
                })),
                // 收紧预算：8 token（mock count_tokens 按字符计）
                TranslationParams::new(8, 512, 0).unwrap(),
            ),
        );
        let svc = TranslateService::new(pairs);
        let err = svc
            .process_translate(req(Some("en"), "zh", "abcdefghij"))
            .await
            .unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                assert_eq!(
                    msg,
                    crate::i18n::tr_with_args(
                        "translate-too-many-source-tokens",
                        crate::i18n::tr_args(&[("count", "10"), ("max", "8")]),
                    ),
                    "超 token 必须消费 translate-too-many-source-tokens 键（实际/上限插值），got {msg}"
                );
            }
            other => panic!("Expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_engine_panic_maps_to_inference_error_without_payload() {
        crate::i18n::init();
        let mut pairs = HashMap::new();
        pairs.insert(
            ("en".to_string(), "zh".to_string()),
            TranslatePairRoute::new(
                Arc::new(RwLock::new(MockTranslateEngine {
                    marker: "en-zh",
                    panic_on_translate: true,
                })),
                TranslationParams::default(),
            ),
        );
        let svc = TranslateService::new(pairs);
        let err = svc
            .process_translate(req(Some("en"), "zh", "hello"))
            .await
            .unwrap_err();
        match err {
            VecboostError::InferenceError(msg) => {
                assert!(
                    msg.contains("translation inference task"),
                    "JoinError 映射文案必须可定位，got: {msg}"
                );
                assert!(
                    !msg.contains("exploded mid-inference"),
                    "wire 文案不得携带 panic 载荷原文（CWE-209），got: {msg}"
                );
            }
            other => panic!("Expected InferenceError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_pair_busy_returns_503_overloaded() {
        // 评审 RES-1/PERF-1：permits=1 背压——许可被占时新请求 503 显性拒绝
        //（非无限排队占 blocking 线程）
        crate::i18n::init();
        let svc = make_service();
        // 占用 en-zh pair 唯一许可
        let route = svc
            .pairs
            .get(&("en".to_string(), "zh".to_string()))
            .unwrap();
        let permit = route.semaphore.clone().try_acquire_owned().unwrap();
        let err = svc
            .process_translate(req(Some("en"), "zh", "hello"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, VecboostError::Overloaded(ref m) if m.contains("busy")),
            "pair 忙必须 503 Overloaded，got {err:?}"
        );
        drop(permit);
        // 释放后恢复可服务
        let resp = svc
            .process_translate(req(Some("en"), "zh", "hello"))
            .await
            .unwrap();
        assert!(!resp.translation.is_empty());
    }

    #[tokio::test]
    async fn test_uppercase_codes_rejected_by_domain_validation() {
        crate::i18n::init();
        let svc = make_service();
        let err = svc
            .process_translate(req(Some("EN"), "zh", "hello"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, VecboostError::ValidationError(_)),
            "大写码在域层 validate 即 ValidationError（不入路由层），got {err:?}"
        );
        let err = svc
            .process_translate(req(Some("en"), "ZH", "hello"))
            .await
            .unwrap_err();
        assert!(matches!(err, VecboostError::ValidationError(_)));
    }

    #[tokio::test]
    async fn test_empty_and_oversize_text_i18n_semantics() {
        crate::i18n::init();
        let svc = make_service();
        // 空文本：服务层 i18n 语义（translate-empty-text 键或缺键回退键名）
        let err = svc
            .process_translate(req(Some("en"), "zh", ""))
            .await
            .unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                assert_eq!(
                    msg,
                    crate::i18n::tr("translate-empty-text"),
                    "空文本必须消费 translate-empty-text 键，got {msg}"
                );
            }
            other => panic!("Expected InvalidInput, got {other:?}"),
        }
        // 超长（CJK 字符计口径）：translate-too-long
        let err = svc
            .process_translate(req(
                Some("en"),
                "zh",
                &"水".repeat(crate::utils::constants::MAX_TEXT_LENGTH + 1),
            ))
            .await
            .unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                assert_eq!(
                    msg,
                    crate::i18n::tr_with_args(
                        "translate-too-long",
                        crate::i18n::tr_args(&[("length", "10001"), ("max", "10000"),]),
                    ),
                    "超长必须消费 translate-too-long 键（长度插值），got {msg}"
                );
            }
            other => panic!("Expected InvalidInput, got {other:?}"),
        }
    }

    /// configured_pairs 只读视图：按 "src-tgt" 展示形态字典序返回。
    #[test]
    fn test_configured_pairs_sorted_display_view() {
        let svc = make_service();
        assert_eq!(
            svc.configured_pairs(),
            &["en-es".to_string(), "en-zh".to_string()],
            "已配置对清单必须排序去重展示"
        );
    }

    // -- 直接调用 mock 未使用的 trait 方法体（覆盖率，仿 rerank.rs 先例） --
    #[test]
    fn test_mock_translate_engine_unused_trait_bodies() {
        let engine = MockTranslateEngine {
            marker: "en-zh",
            panic_on_translate: false,
        };
        assert_eq!(engine.embed("x").unwrap().len(), 8);
        assert_eq!(engine.embed_batch(&["a".to_string()]).unwrap().len(), 1);
        assert_eq!(*engine.precision(), Precision::Fp32);
        assert!(!engine.supports_mixed_precision());
        assert_eq!(engine.count_tokens("abcd").unwrap(), 4);
        assert!(engine.supports_task(ModelTask::Translate));
    }

    #[tokio::test]
    async fn test_translate_mock_engines_fallback_bodies() {
        let mut mock = MockTranslateEngine {
            marker: "en-zh",
            panic_on_translate: false,
        };
        assert!(
            mock.try_fallback_to_cpu(&ModelConfig::default())
                .await
                .is_ok()
        );

        let mut only = EmbeddingOnlyEngine;
        assert_eq!(only.embed("x").unwrap().len(), 8);
        assert_eq!(only.embed_batch(&["a".to_string()]).unwrap().len(), 1);
        assert_eq!(*only.precision(), Precision::Fp32);
        assert!(!only.supports_mixed_precision());
        assert!(
            only.try_fallback_to_cpu(&ModelConfig::default())
                .await
                .is_ok()
        );
    }
}
