// Copyright (c) 2025 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 真实推理引擎测试包装器
//!
//! 提供 RealTestEngine 结构体，支持真实推理和 Mock 回退。

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use vecboost::config::model::{EngineType, ModelConfig, Precision};
use vecboost::engine::{AnyEngine, InferenceEngine};
use vecboost::error::AppError;

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
            _ => TestMode::Mock, // 默认使用 Mock 模式
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
            model_path: PathBuf::from("models/bge-small-en-v1.5"),
            tokenizer_path: Some(PathBuf::from("models/bge-small-en-v1.5-tokenizer")),
            device: vecboost::config::model::DeviceType::Cpu,
            max_batch_size: 16,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
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
        },
    }
}

/// Mock 引擎用于回退
#[derive(Clone)]
pub struct MockEngine {
    dimension: usize,
}

impl MockEngine {
    /// 创建新的 Mock 引擎
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// 生成确定性 Mock 向量
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
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
    fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
        Ok(self.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| self.generate_embedding(t)).collect();
        Ok(embeddings)
    }

    fn precision(&self) -> &Precision {
        &Precision::Fp32
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), AppError> {
        Ok(())
    }
}

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
            // Mock 模式：直接使用 Mock 引擎
            tracing::info!("Using Mock engine (TEST_MODE=mock)");
            Self {
                real_engine: None,
                mock_engine: MockEngine::new(expected_dimension),
                use_fallback: true,
                expected_dimension,
            }
        } else {
            // 尝试创建真实引擎
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
    fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
        if let Some(ref engine) = self.real_engine
            && !self.use_fallback
        {
            match engine.embed(text) {
                Ok(embedding) => {
                    // 验证维度
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

        // 使用 Mock 回退
        Ok(self.mock_engine.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        if let Some(ref engine) = self.real_engine
            && !self.use_fallback
        {
            match engine.embed_batch(texts) {
                Ok(embeddings) => {
                    // 验证第一个向量的维度
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

        // 使用 Mock 回退
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

    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), AppError> {
        if self.use_fallback {
            return Ok(()); // 已经使用回退
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

/// 创建测试引擎的工厂函数
///
/// 根据 TEST_MODE 环境变量选择引擎类型。
pub fn create_test_engine()
-> Result<Arc<RwLock<dyn InferenceEngine + Send + Sync>>, Box<dyn std::error::Error>> {
    // 检查测试模式
    let mode = TestMode::from_env();

    if mode.is_mock() {
        // Mock 模式：使用 1024 维（与原来一致）
        let engine = RealTestEngine::with_dimension(1024);
        Ok(Arc::new(RwLock::new(engine)))
    } else {
        // 真实模式：尝试加载真实模型
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_engine_basic() {
        let engine = RealTestEngine::with_dimension(384);
        let result = engine.embed("Hello world").unwrap();

        assert_eq!(result.len(), 384);
        assert!(result.iter().all(|&x| x.is_finite()));

        // 验证归一化
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_mock_engine_determinism() {
        let engine = RealTestEngine::with_dimension(384);

        let result1 = engine.embed("Hello world").unwrap();
        let result2 = engine.embed("Hello world").unwrap();

        assert_eq!(result1, result2);
    }

    #[tokio::test]
    async fn test_mock_engine_batch() {
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
        // 默认应该是 Mock 模式
        unsafe {
            std::env::remove_var("TEST_MODE");
        }
        assert_eq!(TestMode::from_env(), TestMode::Mock);

        // 设置为 mock
        unsafe {
            std::env::set_var("TEST_MODE", "mock");
        }
        assert_eq!(TestMode::from_env(), TestMode::Mock);

        // 设置为 light
        unsafe {
            std::env::set_var("TEST_MODE", "light");
        }
        assert_eq!(TestMode::from_env(), TestMode::Light);

        // 设置为 full
        unsafe {
            std::env::set_var("TEST_MODE", "full");
        }
        assert_eq!(TestMode::from_env(), TestMode::Full);

        // 清理
        unsafe {
            std::env::remove_var("TEST_MODE");
        }
    }
}
