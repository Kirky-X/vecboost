// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 共享测试 fixtures
//!
//! 提供集成测试与性能测试共用的工具函数、`MockEngine` 与 `RealTestEngine`。
//! 本模块合并自原 `tests/integration/real_engine.rs`。

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use vecboost::config::model::{EngineType, ModelConfig, Precision};
use vecboost::engine::{AnyEngine, InferenceEngine};
use vecboost::error::VecboostError;

/// 默认 Mock 向量维度
#[allow(dead_code)]
pub const DEFAULT_MOCK_DIMENSION: usize = 1024;

// ---------------------------------------------------------------------------
// TestMode
// ---------------------------------------------------------------------------

/// 测试模式配置
#[derive(Debug, Clone, PartialEq)]
pub enum TestMode {
    /// Mock 模式：使用确定性哈希算法
    Mock,
    /// 轻量模式：使用小模型
    Light,
    /// 完整模式：使用完整模型
    Full,
}

impl TestMode {
    /// 从环境变量获取测试模式
    pub fn from_env() -> Self {
        match std::env::var("TEST_MODE").as_deref() {
            Ok("mock") => TestMode::Mock,
            Ok("light") | Ok("real") => TestMode::Light,
            Ok("full") => TestMode::Full,
            _ => TestMode::Mock,
        }
    }

    /// 检查是否使用 Mock 模式
    pub fn is_mock(&self) -> bool {
        matches!(self, TestMode::Mock)
    }

    /// 检查是否使用真实推理
    #[allow(dead_code)]
    pub fn is_real(&self) -> bool {
        matches!(self, TestMode::Light | TestMode::Full)
    }
}

/// 获取默认测试模型配置
pub fn get_test_model_config() -> ModelConfig {
    let mode = TestMode::from_env();

    match mode {
        TestMode::Mock => ModelConfig::default(),
        TestMode::Light => ModelConfig {
            name: "bge-small-en-v1.5".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("models/BAAI-bge-small-en-v1.5"),
            tokenizer_path: None,
            device: vecboost::config::model::DeviceType::Cpu,
            max_batch_size: 16,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
            task: vecboost::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        },
        TestMode::Full => ModelConfig {
            name: "bge-m3".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("models/bge-m3"),
            tokenizer_path: Some(PathBuf::from("models/bge-m3-tokenizer")),
            device: vecboost::config::model::DeviceType::Cpu,
            max_batch_size: 8,
            pooling_mode: None,
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
            task: vecboost::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        },
    }
}

// ---------------------------------------------------------------------------
// MockEngine
// ---------------------------------------------------------------------------

/// 轻量级 MockEngine，仅用于不需要真实引擎行为的单元/集成测试
///
/// 使用 FNV-1a 哈希 + 线性同余生成器产生确定性归一化向量。
#[derive(Clone)]
pub struct MockEngine {
    dimension: usize,
}

impl MockEngine {
    /// 创建新的 MockEngine
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// 生成确定性 Mock 向量
    pub fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.dimension];
        let bytes = text.as_bytes();

        // FNV-1a 哈希算法
        let mut hash: u64 = 1469598103934665603;
        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }

        // 线性同余生成器
        let mut state = hash;
        for val in embedding.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let float_val = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            *val = float_val;
        }

        // 归一化
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in embedding.iter_mut() {
                *val /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl InferenceEngine for MockEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        Ok(self.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| self.generate_embedding(t)).collect();
        Ok(embeddings)
    }

    fn precision(&self) -> &Precision {
        &Precision::Fp32
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RealTestEngine
// ---------------------------------------------------------------------------

/// 真实推理引擎测试包装器
///
/// 在真实推理失败时自动回退到 Mock 实现。
pub struct RealTestEngine {
    /// 真实引擎（可能为 None，如果初始化失败）
    real_engine: Option<AnyEngine>,
    /// Mock 引擎用于回退
    mock_engine: MockEngine,
    /// 当前是否使用回退
    use_fallback: bool,
    /// 期望的向量维度
    expected_dimension: usize,
}

#[allow(dead_code)]
impl RealTestEngine {
    /// 创建新的 RealTestEngine
    ///
    /// 如果真实引擎初始化失败，会自动使用 Mock 回退。
    pub fn new() -> Self {
        let mode = TestMode::from_env();
        let config = get_test_model_config();
        let expected_dimension = config.expected_dimension.unwrap_or(384);

        if mode.is_mock() {
            tracing::info!("Using Mock engine (TEST_MODE=mock)");
            Self {
                real_engine: None,
                mock_engine: MockEngine::new(expected_dimension),
                use_fallback: true,
                expected_dimension,
            }
        } else {
            match AnyEngine::new(&config, config.engine_type.clone(), Precision::Fp32) {
                Ok(engine) => {
                    tracing::info!(
                        "Using real engine: {} (dimension={})",
                        config.name,
                        expected_dimension
                    );
                    Self {
                        real_engine: Some(engine),
                        mock_engine: MockEngine::new(expected_dimension),
                        use_fallback: false,
                        expected_dimension,
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize real engine: {}. Falling back to mock.",
                        e
                    );
                    Self {
                        real_engine: None,
                        mock_engine: MockEngine::new(expected_dimension),
                        use_fallback: true,
                        expected_dimension,
                    }
                }
            }
        }
    }

    /// 创建指定维度的 RealTestEngine
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            real_engine: None,
            mock_engine: MockEngine::new(dimension),
            use_fallback: true,
            expected_dimension: dimension,
        }
    }

    /// 检查是否使用真实推理
    pub fn is_using_real_engine(&self) -> bool {
        self.real_engine.is_some() && !self.use_fallback
    }

    /// 检查是否使用回退
    pub fn is_using_fallback(&self) -> bool {
        self.use_fallback
    }

    /// 获取当前使用的引擎信息
    pub fn engine_info(&self) -> &str {
        if self.use_fallback { "mock" } else { "real" }
    }
}

impl Default for RealTestEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
#[allow(clippy::collapsible_if)]
impl InferenceEngine for RealTestEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        if let Some(ref engine) = self.real_engine
            && !self.use_fallback
        {
            match engine.embed(text) {
                Ok(embedding) => {
                    if embedding.len() == self.expected_dimension {
                        return Ok(embedding);
                    }
                    tracing::warn!(
                        "Engine returned dimension {}, expected {}. Using fallback.",
                        embedding.len(),
                        self.expected_dimension
                    );
                }
                Err(e) => {
                    tracing::warn!("Real engine embed failed: {}. Using fallback.", e);
                }
            }
        }

        Ok(self.mock_engine.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        if let Some(ref engine) = self.real_engine
            && !self.use_fallback
        {
            match engine.embed_batch(texts) {
                Ok(embeddings) => {
                    #[allow(clippy::collapsible_if)]
                    if let Some(first) = embeddings.first() {
                        if first.len() == self.expected_dimension {
                            return Ok(embeddings);
                        }
                    }
                    tracing::warn!(
                        "Engine returned unexpected dimension. Expected {}. Using fallback.",
                        self.expected_dimension
                    );
                }
                Err(e) => {
                    tracing::warn!("Real engine embed_batch failed: {}. Using fallback.", e);
                }
            }
        }

        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| self.mock_engine.generate_embedding(t))
            .collect();
        Ok(embeddings)
    }

    fn precision(&self) -> &Precision {
        if self.use_fallback {
            &Precision::Fp32
        } else if let Some(ref engine) = self.real_engine {
            engine.precision()
        } else {
            &Precision::Fp32
        }
    }

    fn supports_mixed_precision(&self) -> bool {
        if self.use_fallback {
            false
        } else if let Some(ref engine) = self.real_engine {
            engine.supports_mixed_precision()
        } else {
            false
        }
    }

    fn is_fallback_triggered(&self) -> bool {
        self.use_fallback
    }

    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError> {
        if self.use_fallback {
            return Ok(());
        }

        if let Some(ref mut engine) = self.real_engine {
            match engine.try_fallback_to_cpu(config).await {
                Ok(()) => {
                    self.use_fallback = false;
                    tracing::info!("Successfully fell back to CPU");
                    Ok(())
                }
                Err(e) => {
                    tracing::warn!("Failed to fallback to CPU: {}. Using mock fallback.", e);
                    self.use_fallback = true;
                    Ok(())
                }
            }
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

/// 创建测试引擎
///
/// 默认使用 `RealTestEngine`（mock 模式，无外部依赖）。
/// 通过 `TEST_MODE` 环境变量可切换为真实推理。
#[allow(dead_code)]
pub fn create_test_engine()
-> Result<Arc<RwLock<dyn InferenceEngine + Send + Sync>>, Box<dyn std::error::Error>> {
    let mode = TestMode::from_env();

    if mode.is_mock() {
        let engine = RealTestEngine::with_dimension(DEFAULT_MOCK_DIMENSION);
        Ok(Arc::new(RwLock::new(engine)))
    } else {
        let engine = RealTestEngine::new();
        Ok(Arc::new(RwLock::new(engine)))
    }
}

/// 创建指定维度的测试引擎
#[allow(dead_code)]
pub fn create_test_engine_with_dimension(
    dimension: usize,
) -> Result<Arc<RwLock<dyn InferenceEngine + Send + Sync>>, Box<dyn std::error::Error>> {
    let engine = RealTestEngine::with_dimension(dimension);
    Ok(Arc::new(RwLock::new(engine)))
}

// ---------------------------------------------------------------------------
// Candle 决策头对齐测试先决条件（集中一处，candle_decision_parity 复用）
// ---------------------------------------------------------------------------

/// onnxruntime 动态库的平台默认落位（`ort/load-dynamic` 运行时依赖，
/// 见 3rdparty/onnxruntime/README.md）。
#[cfg(target_os = "macos")]
const ORT_DEFAULT_DYLIB: &str = "3rdparty/onnxruntime/libonnxruntime.dylib";
#[cfg(target_os = "windows")]
const ORT_DEFAULT_DYLIB: &str = "3rdparty/onnxruntime/libonnxruntime.dll";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const ORT_DEFAULT_DYLIB: &str = "3rdparty/onnxruntime/libonnxruntime.so";

/// candle 原生决策头与 onnx `DecisionPipeline` 同题对拍的先决资产。
#[allow(dead_code)]
pub struct CandleParityAssets {
    /// 官方 `convaiinnovations/laya` checkpoint（获取步骤见
    /// docs/USER_GUIDE.md「Candle 原生决策路径」）。
    pub checkpoint: PathBuf,
    /// onnx 对拍侧会话加载所需的动态库路径。
    pub ort_dylib: PathBuf,
    /// onnx 对拍侧官方 bundle（receptron/laya-onnx 产物）。
    pub onnx_bundle: PathBuf,
}

impl CandleParityAssets {
    /// 确保进程内 `ORT_DYLIB_PATH` 环境变量就位（未显式设置时回填默认落位；
    /// ort/load-dynamic 在首个 Session 创建时按该变量加载动态库）。
    #[allow(dead_code)]
    pub fn ensure_ort_env(&self) {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &self.ort_dylib);
            }
        }
    }
}

/// 聚合探测 checkpoint safetensors、onnxruntime 动态库与 onnx 对拍 bundle；
/// `Err` 携带缺失原因清单，调用方据此 SKIP（离线红线：资产缺失不硬失败）。
#[allow(dead_code)]
pub fn candle_parity_assets() -> Result<CandleParityAssets, String> {
    let checkpoint = PathBuf::from("models/laya-pytorch/model.safetensors");
    let ort_dylib = match std::env::var("ORT_DYLIB_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(ORT_DEFAULT_DYLIB),
    };
    let onnx_bundle = PathBuf::from("models/laya");
    let mut missing = Vec::new();
    if !checkpoint.is_file() {
        missing.push(format!(
            "checkpoint 缺失：{}（获取步骤见 docs/USER_GUIDE.md）",
            checkpoint.display()
        ));
    }
    if !ort_dylib.is_file() {
        missing.push(format!(
            "onnxruntime 动态库缺失：{}（ORT_DYLIB_PATH 或默认落位）",
            ort_dylib.display()
        ));
    }
    if !onnx_bundle.join("laya.onnx").is_file() {
        missing.push(format!(
            "onnx 对拍 bundle 缺失：{}/laya.onnx（receptron/laya-onnx，获取步骤见 \
             docs/USER_GUIDE.md）",
            onnx_bundle.display()
        ));
    }
    if missing.is_empty() {
        Ok(CandleParityAssets {
            checkpoint,
            ort_dylib,
            onnx_bundle,
        })
    } else {
        Err(missing.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_engine_dimension() {
        let engine = MockEngine::new(256);
        let emb = engine.embed("hello").unwrap();
        assert_eq!(emb.len(), 256);
    }

    #[test]
    fn test_mock_engine_normalized() {
        let engine = MockEngine::new(128);
        let emb = engine.embed("world").unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mock_engine_deterministic() {
        let engine = MockEngine::new(64);
        let a = engine.embed("foo").unwrap();
        let b = engine.embed("foo").unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn test_real_test_engine_mock_fallback() {
        let engine = RealTestEngine::with_dimension(384);
        let result = engine.embed("Hello world").unwrap();

        assert_eq!(result.len(), 384);
        assert!(result.iter().all(|&x| x.is_finite()));

        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_real_test_engine_determinism() {
        let engine = RealTestEngine::with_dimension(384);

        let result1 = engine.embed("Hello world").unwrap();
        let result2 = engine.embed("Hello world").unwrap();

        assert_eq!(result1, result2);
    }

    #[tokio::test]
    async fn test_real_test_engine_batch() {
        let engine = RealTestEngine::with_dimension(384);

        let texts = vec![
            "Hello world".to_string(),
            "Machine learning".to_string(),
            "Artificial intelligence".to_string(),
        ];

        let results = engine.embed_batch(&texts).unwrap();

        assert_eq!(results.len(), 3);
        for result in &results {
            assert_eq!(result.len(), 384);
        }
    }

    #[test]
    fn test_test_mode_from_env() {
        unsafe {
            std::env::remove_var("TEST_MODE");
        }
        assert_eq!(TestMode::from_env(), TestMode::Mock);

        unsafe {
            std::env::set_var("TEST_MODE", "mock");
        }
        assert_eq!(TestMode::from_env(), TestMode::Mock);

        unsafe {
            std::env::set_var("TEST_MODE", "light");
        }
        assert_eq!(TestMode::from_env(), TestMode::Light);

        unsafe {
            std::env::set_var("TEST_MODE", "full");
        }
        assert_eq!(TestMode::from_env(), TestMode::Full);

        unsafe {
            std::env::remove_var("TEST_MODE");
        }
    }
}
