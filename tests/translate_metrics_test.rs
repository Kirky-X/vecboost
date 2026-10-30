// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! translate 热路径计数语义测试。
//!
//! 独立测试二进制：进程内无其他测试共享全局 collector（lib 测试二进制中
//! api/forge 成功路径会并发 inc 同一计数器，计数差断言有竞态），故计数差
//! 断言放本文件——单 #[test] 内顺序完成，无竞态。
//!
//! 口径（spec R-observability-002）：仅通过服务层校验链且通过业务判定的
//! 请求 +1；零配置 / 未配置 pair 等 400 拒绝不计数。
//!
//! prometheus_exporter 模块由 http feature 门控，非 http 构建下本文件整体
//! 不编译（#![cfg]）。

#![cfg(feature = "http")]

use std::sync::Arc;

use tokio::sync::RwLock;

use async_trait::async_trait;
#[cfg(feature = "onnx")]
use std::path::Path;
#[cfg(feature = "onnx")]
use vecboost::config::model::{DeviceType, EngineType, ModelTask};
use vecboost::config::model::{ModelConfig, TranslationParams};
use vecboost::domain::TranslateRequest;
#[cfg(feature = "onnx")]
use vecboost::engine::EngineFactory;
use vecboost::engine::InferenceEngine;
use vecboost::metrics::prometheus_exporter::{PrometheusCollector, set_global_collector};
use vecboost::service::translate::{TranslatePairRoute, TranslateService};

#[cfg(feature = "onnx")]
const OPUS_EN_ZH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/opus-mt-en-zh");

#[cfg(feature = "onnx")]
fn opus_ready() -> bool {
    Path::new(OPUS_EN_ZH)
        .join("onnx/encoder_model.onnx")
        .exists()
        && Path::new(OPUS_EN_ZH)
            .join("onnx/decoder_model_merged.onnx")
            .exists()
}

fn read_counter(collector: &PrometheusCollector) -> u64 {
    let families = collector.registry().gather();
    families
        .iter()
        .find(|m| m.name() == "vecboost_translate_requests_total")
        .expect("vecboost_translate_requests_total registered")
        .get_metric()[0]
        .get_counter()
        .value() as u64
}

fn req(src: Option<&str>, tgt: &str, text: &str) -> TranslateRequest {
    TranslateRequest {
        text: text.to_string(),
        source_language: src.map(|s| s.to_string()),
        target_language: tgt.to_string(),
    }
}

/// 400 拒绝段聚合断言：零配置 / auto 检测失败 / 路由未命中 / 能力门四类
/// 拒绝均不得计数；返回含 embed-only 路由的服务句柄（onnx 成功段复用
/// collector 而非本服务）。
async fn assert_rejection_segments_not_counted(
    collector: &Arc<PrometheusCollector>,
    c0: u64,
) -> (TranslateService, u64) {
    // ── 零配置 400：translate-not-configured，不计 ──
    let empty = TranslateService::new(std::collections::HashMap::new());
    let err = empty
        .process_translate(req(Some("en"), "zh", "hello"))
        .await
        .expect_err("zero config must 400");
    assert!(matches!(
        err,
        vecboost::error::VecboostError::InvalidInput(_)
    ));
    assert_eq!(
        read_counter(collector),
        c0,
        "零配置 400（translate-not-configured）不得计数"
    );

    // ── auto 检测失败 400：translate-detect-failed，不计 ──
    let mut pairs = std::collections::HashMap::new();
    pairs.insert(
        ("en".to_string(), "zh".to_string()),
        TranslatePairRoute::new(
            Arc::new(RwLock::new(EmbeddingOnlyEngine)),
            TranslationParams::default(),
        ),
    );
    let svc = TranslateService::new(pairs);
    let err = svc
        .process_translate(req(None, "zh", "1234567890"))
        .await
        .expect_err("unreliable detection must 400");
    assert!(matches!(
        err,
        vecboost::error::VecboostError::InvalidInput(_)
    ));
    assert_eq!(
        read_counter(collector),
        c0,
        "auto 检测失败 400（translate-detect-failed）不得计数"
    );

    // ── 路由未命中 400：translate-unsupported-pair，不计 ──
    let err = svc
        .process_translate(req(Some("en"), "fr", "hello"))
        .await
        .expect_err("unconfigured pair must 400");
    assert!(matches!(
        err,
        vecboost::error::VecboostError::InvalidInput(_)
    ));
    assert_eq!(
        read_counter(collector),
        c0,
        "路由未命中 400（translate-unsupported-pair）不得计数"
    );

    // ── 能力门 400：translate-unsupported，不计 ──
    let err = svc
        .process_translate(req(Some("en"), "zh", "hello"))
        .await
        .expect_err("embedding-only engine must be rejected");
    assert!(matches!(
        err,
        vecboost::error::VecboostError::UnsupportedTask(_)
    ));
    assert_eq!(
        read_counter(collector),
        c0,
        "能力门 400（translate-unsupported）不得计数"
    );

    (svc, c0)
}

#[tokio::test]
async fn translate_counter_counts_only_successful_requests() {
    vecboost::i18n::init();

    // 本测试二进制无其他 set_global_collector 调用方，set 必成功 →
    // 计数差断言无并发竞态
    let collector = Arc::new(PrometheusCollector::new().expect("collector"));
    assert!(
        set_global_collector(collector.clone()),
        "独立测试二进制内 collector 必可注册"
    );
    let c0 = read_counter(&collector);

    let _svc = assert_rejection_segments_not_counted(&collector, c0).await;

    // ── 成功请求：+1（真机 bundle 守卫；缺失时跳过该段，拒绝段断言不受
    // 影响。dylib 探测必须在 Session 构建前：ort/load-dynamic 缺库为
    // 非 unwinding panic 会 SIGABRT 拖垮测试进程——decision.rs 同款先例）──
    // 段内 EngineType::Onnx 由 onnx feature 门控，无 onnx 构建下整段不编译。
    #[cfg(feature = "onnx")]
    {
        // ort/load-dynamic 默认 dylib 按目标平台选扩展名（与 src/engine/session.rs
        // ort_dylib_default 同口径；集成测试独立 crate，cfg 门控声明）。
        #[cfg(target_os = "macos")]
        let default_dylib = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnxruntime/libonnxruntime.dylib"
        )
        .to_string();
        #[cfg(target_os = "windows")]
        let default_dylib = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnxruntime/onnxruntime.dll"
        )
        .to_string();
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let default_dylib = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnxruntime/libonnxruntime.so"
        )
        .to_string();
        let dylib = std::env::var("ORT_DYLIB_PATH").unwrap_or(default_dylib);
        if !Path::new(&dylib).is_file() {
            println!("SKIP [success-path segment]: onnxruntime dylib 缺失（{dylib}）");
            return;
        }
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            // SAFETY: 测试进程内单线程时点回填一次
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &dylib);
            }
        }
        if !opus_ready() {
            println!("SKIP [success-path segment]: opus-mt-en-zh bundle 缺失");
            return;
        }
        let any = EngineFactory::create(
            EngineType::Onnx,
            &ModelConfig {
                name: "counter-e2e".to_string(),
                engine_type: EngineType::Onnx,
                model_path: std::path::PathBuf::from(OPUS_EN_ZH),
                device: DeviceType::Cpu,
                max_batch_size: 1,
                oom_fallback_enabled: false,
                task: ModelTask::Translate,
                translation_params: Some(TranslationParams::new(512, 512, 0).unwrap()),
                transcription_params: None,
                ..Default::default()
            },
        )
        .expect("bundle 就绪时必须可加载");
        let engine: Arc<RwLock<dyn InferenceEngine>> = Arc::new(RwLock::new(any));
        assert!(engine.read().await.supports_task(ModelTask::Translate));

        let mut pairs = std::collections::HashMap::new();
        pairs.insert(
            ("en".to_string(), "zh".to_string()),
            TranslatePairRoute::new(engine, TranslationParams::default()),
        );
        let svc = TranslateService::new(pairs);
        let resp = svc
            .process_translate(req(Some("en"), "zh", "hello"))
            .await
            .expect("成功路径必须 200");
        assert!(!resp.translation.trim().is_empty());
        assert_eq!(
            read_counter(&collector),
            c0 + 1,
            "成功请求必须恰好 +1（400 类不得贡献计数）"
        );

        // 二次成功：再 +1
        svc.process_translate(req(Some("en"), "zh", "world"))
            .await
            .expect("second success");
        assert_eq!(read_counter(&collector), c0 + 2, "第二次成功再 +1");
    }
}

struct EmbeddingOnlyEngine;

// EmbeddingOnlyEngine 的 trait 实现（能力门 400 场景：不支持 Translate）
#[async_trait]
impl InferenceEngine for EmbeddingOnlyEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, vecboost::error::VecboostError> {
        Ok(vec![0.0; 8])
    }
    fn embed_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, vecboost::error::VecboostError> {
        Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
    }

    fn precision(&self) -> &vecboost::config::model::Precision {
        static FP32: vecboost::config::model::Precision = vecboost::config::model::Precision::Fp32;
        &FP32
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    async fn try_fallback_to_cpu(
        &mut self,
        _config: &ModelConfig,
    ) -> Result<(), vecboost::error::VecboostError> {
        Ok(())
    }
}
