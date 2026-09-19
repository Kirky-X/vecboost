// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

use crate::config::model::{EngineType, ModelConfig};
use crate::error::VecboostError;
use crate::model::loader::{LoadedModel, LocalModelLoader, ModelLoader};
use log::info;
use log::warn;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::timeout;

const DEFAULT_MODEL_LOAD_TIMEOUT_SECS: u64 = 300;

/// LFRU 决策计数衰减周期：每 1024 次 load 决策全局 heat 减半。
const HEAT_DECAY_PERIOD: u64 = 1024;
/// recency 分桶数（新近度权重 ≤ 1/256）。
const RECENCY_BUCKETS: u64 = 8;
/// 每桶覆盖的序列步长。
const RECENCY_BUCKET_SPAN: u64 = 128;

/// 驻留管理配置（port 自 colibri tier.h/LFRU）。
#[derive(Clone, Copy, Debug)]
pub struct ResidencyConfig {
    /// 最大驻留模型数。
    pub max_models: usize,
    /// 驻留内存预算（MB），None 表示不限制。
    pub memory_budget_mb: Option<u64>,
}

#[derive(Clone)]
pub struct ModelManager {
    models: Arc<RwLock<HashMap<String, Arc<dyn LoadedModel>>>>,
    loader: Arc<dyn ModelLoader>,
    default_config: ModelConfig,
    timeout_duration: Duration,
    residency: Option<ResidencyConfig>,
    /// heat 表：u32 饱和计数，粘滞不回绕；key 为模型名，unload 后保留、
    /// 经 heat.rs 持久化（warmstart 种子）。
    heat: Arc<RwLock<HashMap<String, u32>>>,
    /// recency 表：模型名 → 最后命中序列号。
    recency: Arc<RwLock<HashMap<String, u64>>>,
    seq: Arc<AtomicU64>,
    decisions: Arc<AtomicU64>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self::with_loader(
            Arc::new(LocalModelLoader::new(PathBuf::from("models"))) as Arc<dyn ModelLoader>
        )
    }

    pub fn with_loader(loader: Arc<dyn ModelLoader>) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            loader,
            default_config: ModelConfig::default(),
            timeout_duration: Duration::from_secs(DEFAULT_MODEL_LOAD_TIMEOUT_SECS),
            residency: None,
            heat: Arc::new(RwLock::new(HashMap::new())),
            recency: Arc::new(RwLock::new(HashMap::new())),
            seq: Arc::new(AtomicU64::new(0)),
            decisions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 启用 LFRU 驻留管理。默认不限制（现状行为）。
    /// `max_models = 0` 视为不限制。
    pub fn with_residency(mut self, max_models: usize, memory_budget_mb: Option<u64>) -> Self {
        if max_models == 0 {
            self.residency = None;
        } else {
            self.residency = Some(ResidencyConfig {
                max_models,
                memory_budget_mb,
            });
        }
        self
    }

    /// 查询模型当前 heat（测试与 heat.rs 落盘用）。
    pub async fn heat_of(&self, name: &str) -> u32 {
        self.heat.read().await.get(name).copied().unwrap_or(0)
    }

    /// 测试专用：预置 heat（模拟 warmstart 种子）。
    #[cfg(test)]
    pub async fn seed_heat(&self, name: &str, heat: u32) {
        self.heat.write().await.insert(name.to_string(), heat);
    }

    /// 记录一次命中：heat 饱和 +1，recency 推进。
    async fn record_hit(&self, name: &str) {
        {
            let mut heat = self.heat.write().await;
            let e = heat.entry(name.to_string()).or_insert(0);
            *e = e.saturating_add(1);
        }
        let s = self.seq.fetch_add(1, Ordering::Relaxed);
        self.recency.write().await.insert(name.to_string(), s);
    }

    /// LFRU 评分：`score = (heat << 8) | recency_bucket`。
    fn score_of(heat: u32, bucket: u64) -> u64 {
        ((heat as u64) << 8) | (bucket & 0xFF)
    }

    /// 由序列号折算 8 桶新近度（越新越大，最大 7）。
    fn bucket_for(entry_seq: u64, cur_seq: u64) -> u64 {
        let age = cur_seq.saturating_sub(entry_seq);
        (RECENCY_BUCKETS - 1).saturating_sub((age / RECENCY_BUCKET_SPAN).min(RECENCY_BUCKETS - 1))
    }

    /// 迟滞判定：候选 heat 低于现任 heat 的 25%+4 点时不驱逐（防乒乓）。
    /// 即仅当 `candidate + victim/4 + 4 >= victim` 时允许驱逐（整数运算）。
    fn hysteresis_passes(candidate_heat: u32, victim_heat: u32) -> bool {
        let margin = victim_heat / 4 + 4;
        candidate_heat.saturating_add(margin) >= victim_heat
    }

    /// 从 heat.rs 文件加载 warmstart 初始 heat（损坏/指纹不匹配即弃，见 heat.rs）。
    /// 与内存热度取 max 合并，不丢失更热的运行态。
    pub async fn load_heat_file(&self, path: &std::path::Path) {
        let loaded = super::heat::load_heat(path);
        if loaded.is_empty() {
            return;
        }
        let mut heat = self.heat.write().await;
        for (name, h) in loaded {
            let e = heat.entry(name).or_insert(0);
            *e = (*e).max(h);
        }
        info!("Model heat warmstart loaded: {} entries", heat.len());
    }

    /// 热度落盘（原子写，见 heat.rs）。
    pub async fn save_heat_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let heat = self.heat.read().await;
        super::heat::save_heat(path, &heat)
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_duration = Duration::from_secs(timeout_secs);
        self
    }

    pub async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, VecboostError> {
        let model_name = config.name.clone();

        if let Some(existing) = self.get(&model_name).await {
            info!(
                "Model {} already loaded, reusing existing instance",
                model_name
            );
            self.record_hit(&model_name).await;
            return Ok(existing);
        }

        info!(
            "Loading model: {} from {:?} (timeout: {:?})",
            model_name, config.model_path, self.timeout_duration
        );

        let load_future = self.loader.load(config);
        match timeout(self.timeout_duration, load_future).await {
            Ok(Ok(model)) => {
                let mut models = self.models.write().await;
                models.insert(model_name.clone(), Arc::clone(&model));
                drop(models);

                // 新条目：保留历史 heat（重载恢复），无历史则从 1 起步。
                {
                    let mut heat = self.heat.write().await;
                    heat.entry(model_name.clone()).or_insert(1);
                }
                let s = self.seq.fetch_add(1, Ordering::Relaxed);
                self.recency.write().await.insert(model_name.clone(), s);
                // 衰减：每 1024 次 load 决策全局减半（防旧工作负载永久冻结）。
                let d = self.decisions.fetch_add(1, Ordering::Relaxed) + 1;
                if d.is_multiple_of(HEAT_DECAY_PERIOD) {
                    let mut heat = self.heat.write().await;
                    for v in heat.values_mut() {
                        *v /= 2;
                    }
                    info!("Model heat decayed (decision #{})", d);
                }

                info!("Model {} loaded successfully", model_name);
                self.enforce_residency(&model_name).await;
                Ok(model)
            }
            Ok(Err(e)) => {
                warn!("Model {} loading failed: {}", model_name, e);
                Err(VecboostError::ModelLoadError(format!(
                    "Failed to load model {}: {}",
                    model_name, e
                )))
            }
            Err(_) => {
                warn!(
                    "Model {} loading timed out after {:?}",
                    model_name, self.timeout_duration
                );
                Err(VecboostError::ModelLoadError(format!(
                    "Model loading timed out after {} seconds: {}",
                    self.timeout_duration.as_secs(),
                    model_name
                )))
            }
        }
    }

    pub async fn get(&self, name: &str) -> Option<Arc<dyn LoadedModel>> {
        let models = self.models.read().await;
        models.get(name).map(Arc::clone)
    }

    pub async fn unload(&self, name: &str) -> Result<(), VecboostError> {
        let mut models = self.models.write().await;

        if let Some(model) = models.remove(name) {
            info!("Model {} unloaded successfully", name);
            drop(model);
            drop(models);
            // heat 保留（重载恢复/warmstart 种子），recency 清除。
            self.recency.write().await.remove(name);
            Ok(())
        } else {
            warn!("Model {} not found for unloading", name);
            Err(VecboostError::NotFound(format!(
                "Model not found: {}",
                name
            )))
        }
    }

    /// 驻留 enforcement：超驻留上限时按 LFRU 驱逐。
    ///
    /// - 受害者 = `score=(heat<<8)|recency_bucket` 最低的驻留模型；
    /// - 迟滞：新条目 heat 低于受害者 heat 的 25%+4 点时不驱逐（防乒乓，
    ///   此时允许暂时超驻留——新模型需经命中证明自己）；
    /// - 有 in-flight 引用（`Arc::strong_count > 2`）的模型跳过本轮；
    /// - 内存预算为硬约束：超预算时忽略迟滞继续驱逐（防 OOM）。
    ///
    /// 驱逐走既有 `unload` 路径，决策记日志。
    async fn enforce_residency(&self, newcomer: &str) {
        let residency = match self.residency {
            Some(r) => r,
            None => return,
        };
        // 数量上限。
        loop {
            let victim = self.pick_victim(Some(newcomer)).await;
            let Some((name, victim_heat, victim_score)) = victim else {
                break;
            };
            {
                let models = self.models.read().await;
                if models.len() <= residency.max_models {
                    break;
                }
            }
            let newcomer_heat = self.heat_of(newcomer).await;
            if !Self::hysteresis_passes(newcomer_heat, victim_heat) {
                info!(
                    "Residency: keep {} (heat {}) over {} (heat {}): hysteresis blocks eviction",
                    newcomer, newcomer_heat, name, victim_heat
                );
                break;
            }
            info!(
                "Residency: evicting {} (heat {}, score {}) for {} (heat {})",
                name, victim_heat, victim_score, newcomer, newcomer_heat
            );
            if self.unload(&name).await.is_err() {
                break;
            }
        }
        // 内存预算（硬约束，忽略迟滞；新来者同样受预算约束）。
        if let Some(budget_mb) = residency.memory_budget_mb {
            let budget_bytes = budget_mb.saturating_mul(1024 * 1024);
            loop {
                if self.resident_bytes().await <= budget_bytes {
                    break;
                }
                let victim = self.pick_victim(None).await;
                let Some((name, victim_heat, victim_score)) = victim else {
                    break;
                };
                info!(
                    "Residency: over memory budget, evicting {} (heat {}, score {})",
                    name, victim_heat, victim_score
                );
                if self.unload(&name).await.is_err() {
                    break;
                }
            }
        }
    }

    /// 选择驱逐受害者：分数最低的非 in-flight 驻留模型。
    /// （迟滞由调用方在 newcomer 与受害者之间判定，此处只负责排序与跳过。
    /// `exclude` 排除刚载入的新来者——数量上限场景不立即反杀新人；
    /// 预算场景传 None，新来者同样可被驱逐。）
    async fn pick_victim(&self, exclude: Option<&str>) -> Option<(String, u32, u64)> {
        let models = self.models.read().await;
        let heat = self.heat.read().await;
        let recency = self.recency.read().await;
        let cur = self.seq.load(Ordering::Relaxed);
        let mut best: Option<(String, u32, u64)> = None;
        for (name, model) in models.iter() {
            if Some(name.as_str()) == exclude {
                continue;
            }
            // in-flight 引用跳过本轮（map 持 1 + 调用方至多 1 = 2 为空闲）。
            if Arc::strong_count(model) > 2 {
                continue;
            }
            let h = heat.get(name).copied().unwrap_or(0);
            let bucket = recency
                .get(name)
                .map(|s| Self::bucket_for(*s, cur))
                .unwrap_or(0);
            let score = Self::score_of(h, bucket);
            if best.as_ref().map(|(_, _, bs)| score < *bs).unwrap_or(true) {
                best = Some((name.clone(), h, score));
            }
        }
        best
    }

    /// 当前驻留模型总字节数（文件缺失按 0 计）。
    async fn resident_bytes(&self) -> u64 {
        let models = self.models.read().await;
        let mut total = 0u64;
        for model in models.values() {
            if let Ok(meta) = std::fs::metadata(model.path()) {
                // GGUF 量化引擎是加载期反量化桥，运行期内存
                // ≈ 文件体积的数倍（Q8_0 ~4×、Q4_K ~7×），统一按保守 4× 估算，
                // 否则 memory_budget 对量化模型形同虚设。
                let is_quantized = model
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("gguf"))
                    .unwrap_or(false);
                let bytes = if is_quantized {
                    meta.len().saturating_mul(4)
                } else {
                    meta.len()
                };
                total = total.saturating_add(bytes);
            }
        }
        total
    }

    pub async fn unload_all(&self) {
        let mut models = self.models.write().await;
        let model_names: Vec<String> = models.keys().cloned().collect();

        for name in model_names {
            if let Some(model) = models.remove(&name) {
                info!("Unloaded model: {}", name);
                drop(model);
            }
        }
        drop(models);
        // heat 保留（warmstart 种子），recency 清除。
        self.recency.write().await.clear();

        info!("All models unloaded");
    }

    pub async fn reload(&self, name: &str) -> Result<Arc<dyn LoadedModel>, VecboostError> {
        let config = {
            let models = self.models.read().await;
            let model = models
                .get(name)
                .ok_or_else(|| VecboostError::NotFound(format!("Model not found: {}", name)))?;

            ModelConfig {
                name: name.to_string(),
                engine_type: model.engine_type(),
                model_path: model.path().to_path_buf(),
                tokenizer_path: None,
                device: crate::config::model::DeviceType::Cpu,
                max_batch_size: 32,
                pooling_mode: None,
                expected_dimension: None,
                memory_limit_bytes: None,
                oom_fallback_enabled: false,
                model_sha256: None,
                quantized: false,
            }
        };

        self.unload(name).await?;
        self.load(&config).await
    }

    pub async fn count(&self) -> usize {
        self.models.read().await.len()
    }

    pub async fn is_loaded(&self, name: &str) -> bool {
        self.models.read().await.contains_key(name)
    }

    pub async fn list_loaded(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }

    pub async fn stats(&self) -> ModelStats {
        let models = self.models.read().await;

        let candle_count = models
            .values()
            .filter(|m| m.engine_type() == EngineType::Candle)
            .count();

        let onnx_count = models
            .values()
            .filter(|_m| {
                #[cfg(feature = "onnx")]
                {
                    _m.engine_type() == EngineType::Onnx
                }
                #[cfg(not(feature = "onnx"))]
                {
                    false
                }
            })
            .count();

        let mut total_model_size = 0;
        for model in models.values() {
            if let Ok(metadata) = std::fs::metadata(model.path()) {
                total_model_size += metadata.len();
            }
        }

        ModelStats {
            total_models: models.len(),
            candle_models: candle_count,
            onnx_models: onnx_count,
            total_size_bytes: total_model_size,
        }
    }

    pub fn set_default_config(&mut self, config: ModelConfig) {
        self.default_config = config;
    }

    pub async fn load_default(&self) -> Result<Arc<dyn LoadedModel>, VecboostError> {
        self.load(&self.default_config).await
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ModelStats {
    pub total_models: usize,
    pub candle_models: usize,
    pub onnx_models: usize,
    pub total_size_bytes: u64,
}

impl ModelStats {
    pub fn total_size_mb(&self) -> f64 {
        self.total_size_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn format_size(&self) -> String {
        if self.total_size_bytes < 1024 {
            format!("{} B", self.total_size_bytes)
        } else if self.total_size_bytes < 1024 * 1024 {
            format!("{} KB", self.total_size_bytes / 1024)
        } else if self.total_size_bytes < 1024 * 1024 * 1024 {
            format!("{:.2} MB", self.total_size_bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!(
                "{:.2} GB",
                self.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    struct SlowModelLoader {
        delay_ms: u64,
    }

    impl SlowModelLoader {
        fn new(delay_ms: u64) -> Self {
            Self { delay_ms }
        }
    }

    #[async_trait]
    impl ModelLoader for SlowModelLoader {
        async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, VecboostError> {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;

            let model: Arc<dyn LoadedModel> = Arc::new(CandleModel {
                path: config.model_path.clone(),
                name: config.name.clone(),
            });

            Ok(model)
        }

        async fn get_model_path(&self, config: &ModelConfig) -> Result<PathBuf, VecboostError> {
            Ok(config.model_path.clone())
        }

        async fn is_model_cached(&self, _config: &ModelConfig) -> bool {
            true
        }
    }

    struct CandleModel {
        path: PathBuf,
        name: String,
    }

    impl LoadedModel for CandleModel {
        fn name(&self) -> &str {
            &self.name
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn engine_type(&self) -> EngineType {
            EngineType::Candle
        }

        fn reload(&self) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn create_test_model_file(path: &PathBuf) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "test model content").unwrap();
    }

    #[tokio::test]
    async fn test_model_manager_creation() {
        let manager = ModelManager::new();
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_model_manager_with_loader() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_model_manager_load_unload() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("test-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        let _model = manager.load(&config).await.unwrap();
        assert_eq!(manager.count().await, 1);
        assert!(manager.is_loaded("test-model").await);

        manager.unload("test-model").await.unwrap();
        assert_eq!(manager.count().await, 0);
        assert!(!manager.is_loaded("test-model").await);
    }

    #[tokio::test]
    async fn test_model_manager_list_loaded() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config1 = ModelConfig {
            name: "model-1".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("model-1"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        let config2 = ModelConfig {
            name: "model-2".to_string(),
            #[cfg(feature = "onnx")]
            engine_type: EngineType::Onnx,
            #[cfg(not(feature = "onnx"))]
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("model-2"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        fs::create_dir_all(&config1.model_path).unwrap();
        fs::create_dir_all(&config2.model_path).unwrap();

        manager.load(&config1).await.unwrap();
        #[cfg(feature = "onnx")]
        manager.load(&config2).await.unwrap();

        let loaded = manager.list_loaded().await;
        #[cfg(feature = "onnx")]
        {
            assert_eq!(loaded.len(), 2);
            assert!(loaded.contains(&"model-1".to_string()));
            assert!(loaded.contains(&"model-2".to_string()));
        }
        #[cfg(not(feature = "onnx"))]
        {
            assert_eq!(loaded.len(), 1);
            assert!(loaded.contains(&"model-1".to_string()));
        }
    }

    #[tokio::test]
    async fn test_model_manager_stats() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let stats = manager.stats().await;
        assert_eq!(stats.total_models, 0);
        assert_eq!(stats.candle_models, 0);
        assert_eq!(stats.onnx_models, 0);
    }

    #[tokio::test]
    async fn test_model_manager_unload_all() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        fs::create_dir_all(&config.model_path).unwrap();
        manager.load(&config).await.unwrap();
        assert_eq!(manager.count().await, 1);

        manager.unload_all().await;
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_model_manager_unload_nonexistent() {
        let manager = ModelManager::new();
        let result = manager.unload("nonexistent").await;
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("not found"));
        }
    }

    #[tokio::test]
    async fn test_model_stats_format_size() {
        let stats = ModelStats {
            total_models: 1,
            candle_models: 1,
            onnx_models: 0,
            total_size_bytes: 1024,
        };

        assert_eq!(stats.format_size(), "1 KB");
    }

    #[tokio::test]
    async fn test_model_manager_with_timeout() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader).with_timeout(1);

        assert_eq!(manager.timeout_duration.as_secs(), 1);
    }

    #[tokio::test]
    async fn test_model_manager_load_timeout_success() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader).with_timeout(30);

        let config = ModelConfig {
            name: "timeout-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("timeout-test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        fs::create_dir_all(&config.model_path).unwrap();

        let result = manager.load(&config).await;
        assert!(result.is_ok());
        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_model_manager_load_reuses_existing() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader).with_timeout(30);

        let config = ModelConfig {
            name: "reuse-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("reuse-test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        fs::create_dir_all(&config.model_path).unwrap();

        let first = manager.load(&config).await.unwrap();
        let second = manager.load(&config).await.unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_model_manager_load_timeout_failure() {
        let cache_dir = tempdir().unwrap();
        let slow_loader = Arc::new(SlowModelLoader::new(2000));
        let manager = ModelManager::with_loader(slow_loader).with_timeout(1);

        let config = ModelConfig {
            name: "slow-timeout-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("slow-timeout-test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        fs::create_dir_all(&config.model_path).unwrap();

        let result = manager.load(&config).await;

        assert!(result.is_err());
        match result {
            Err(e) => {
                let error_msg = e.to_string();
                assert!(
                    error_msg.contains("timed out"),
                    "Expected 'timed out' in error message, got: {}",
                    error_msg
                );
            }
            Ok(_) => panic!("Expected error, but got success"),
        }
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_model_manager_load_success_after_timeout() {
        let cache_dir = tempdir().unwrap();
        let slow_loader = Arc::new(SlowModelLoader::new(100));
        let manager = ModelManager::with_loader(slow_loader).with_timeout(5);

        let config = ModelConfig {
            name: "success-after-timeout-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("success-after-timeout-test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        fs::create_dir_all(&config.model_path).unwrap();

        let result = manager.load(&config).await;

        assert!(result.is_ok());
        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_get_returns_none_for_unknown_model() {
        let manager = ModelManager::new();
        assert!(manager.get("does-not-exist").await.is_none());
    }

    #[tokio::test]
    async fn test_get_returns_loaded_model() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("get-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "get-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        let loaded = manager.load(&config).await.unwrap();
        let fetched = manager.get("get-model").await;
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().name(), loaded.name());
    }

    #[tokio::test]
    async fn test_is_loaded_returns_false_for_unknown() {
        let manager = ModelManager::new();
        assert!(!manager.is_loaded("unknown").await);
    }

    #[tokio::test]
    async fn test_default_creates_empty_manager() {
        let manager = ModelManager::default();
        assert_eq!(manager.count().await, 0);
        assert!(manager.list_loaded().await.is_empty());
    }

    #[tokio::test]
    async fn test_unload_all_on_empty_manager_is_noop() {
        let manager = ModelManager::new();
        manager.unload_all().await;
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_unload_all_removes_multiple_models() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        for name in ["multi-1", "multi-2", "multi-3"] {
            let model_path = cache_dir.path().join(name);
            create_test_model_file(&model_path);
            let config = ModelConfig {
                name: name.to_string(),
                engine_type: EngineType::Candle,
                model_path,
                tokenizer_path: None,
                device: crate::config::model::DeviceType::Cpu,
                max_batch_size: 32,
                pooling_mode: None,
                expected_dimension: None,
                memory_limit_bytes: None,
                oom_fallback_enabled: false,
                model_sha256: None,
                quantized: false,
            };
            manager.load(&config).await.unwrap();
        }

        assert_eq!(manager.count().await, 3);
        manager.unload_all().await;
        assert_eq!(manager.count().await, 0);
        assert!(manager.list_loaded().await.is_empty());
    }

    #[tokio::test]
    async fn test_reload_existing_model_succeeds() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("reload-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "reload-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        let _ = manager.load(&config).await.unwrap();
        assert_eq!(manager.count().await, 1);

        let reloaded = manager.reload("reload-model").await;
        assert!(reloaded.is_ok());
        assert_eq!(manager.count().await, 1);
        assert!(manager.is_loaded("reload-model").await);
    }

    #[tokio::test]
    async fn test_reload_unknown_model_returns_not_found() {
        let manager = ModelManager::new();
        let result = manager.reload("no-such-model").await;
        assert!(result.is_err());
        match result.err().unwrap() {
            VecboostError::NotFound(msg) => {
                assert!(msg.contains("no-such-model"));
            }
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    struct FailingLoader;

    #[async_trait]
    impl ModelLoader for FailingLoader {
        async fn load(&self, _config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, VecboostError> {
            Err(VecboostError::ModelLoadError(
                "intentional failure".to_string(),
            ))
        }

        async fn get_model_path(&self, config: &ModelConfig) -> Result<PathBuf, VecboostError> {
            Ok(config.model_path.clone())
        }

        async fn is_model_cached(&self, _config: &ModelConfig) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn test_load_with_loader_failure_returns_model_load_error() {
        let loader = Arc::new(FailingLoader);
        let manager = ModelManager::with_loader(loader);

        let cache_dir = tempdir().unwrap();
        let config = ModelConfig {
            name: "fail-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("fail-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        let result = manager.load(&config).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("fail-model"));
                assert!(msg.contains("intentional failure"));
            }
            other => panic!("expected ModelLoadError, got {:?}", other),
        }
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_load_default_uses_default_config() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("default-test-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let mut manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "default-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };
        manager.set_default_config(config);

        let result = manager.load_default().await;
        assert!(result.is_ok());
        assert_eq!(manager.count().await, 1);
        assert!(manager.is_loaded("default-test-model").await);
    }

    #[tokio::test]
    async fn test_stats_counts_candle_models() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        for name in ["stat-1", "stat-2"] {
            let model_path = cache_dir.path().join(name);
            create_test_model_file(&model_path);
            let config = ModelConfig {
                name: name.to_string(),
                engine_type: EngineType::Candle,
                model_path,
                tokenizer_path: None,
                device: crate::config::model::DeviceType::Cpu,
                max_batch_size: 32,
                pooling_mode: None,
                expected_dimension: None,
                memory_limit_bytes: None,
                oom_fallback_enabled: false,
                model_sha256: None,
                quantized: false,
            };
            manager.load(&config).await.unwrap();
        }

        let stats = manager.stats().await;
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.candle_models, 2);
        assert_eq!(stats.onnx_models, 0);
        assert!(stats.total_size_bytes > 0);
    }

    #[tokio::test]
    async fn test_concurrent_loads_distinct_models() {
        let cache_dir = tempdir().unwrap();
        let cache_root = cache_dir.path().to_path_buf();

        let mut handles = Vec::new();
        let loader = Arc::new(LocalModelLoader::new(cache_root.clone()));
        let manager = std::sync::Arc::new(ModelManager::with_loader(loader));

        for i in 0..5 {
            let mgr = std::sync::Arc::clone(&manager);
            let root = cache_root.clone();
            handles.push(tokio::spawn(async move {
                let name = format!("conc-{}", i);
                let model_path = root.join(&name);
                fs::create_dir_all(model_path.parent().unwrap()).unwrap();
                fs::write(&model_path, "content").unwrap();

                let config = ModelConfig {
                    name: name.clone(),
                    engine_type: EngineType::Candle,
                    model_path,
                    tokenizer_path: None,
                    device: crate::config::model::DeviceType::Cpu,
                    max_batch_size: 32,
                    pooling_mode: None,
                    expected_dimension: None,
                    memory_limit_bytes: None,
                    oom_fallback_enabled: false,
                    model_sha256: None,
                    quantized: false,
                };
                mgr.load(&config).await.unwrap();
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(manager.count().await, 5);
        let loaded = manager.list_loaded().await;
        for i in 0..5 {
            assert!(loaded.contains(&format!("conc-{}", i)));
        }
    }

    #[test]
    fn test_model_stats_total_size_mb() {
        let stats = ModelStats {
            total_models: 1,
            candle_models: 1,
            onnx_models: 0,
            total_size_bytes: 2 * 1024 * 1024,
        };
        assert_eq!(stats.total_size_mb(), 2.0);
    }

    #[test]
    fn test_model_stats_format_size_bytes() {
        let stats = ModelStats {
            total_models: 0,
            candle_models: 0,
            onnx_models: 0,
            total_size_bytes: 512,
        };
        assert_eq!(stats.format_size(), "512 B");
    }

    #[test]
    fn test_model_stats_format_size_kb() {
        let stats = ModelStats {
            total_models: 0,
            candle_models: 0,
            onnx_models: 0,
            total_size_bytes: 2048,
        };
        assert_eq!(stats.format_size(), "2 KB");
    }

    #[test]
    fn test_model_stats_format_size_mb() {
        let stats = ModelStats {
            total_models: 0,
            candle_models: 0,
            onnx_models: 0,
            total_size_bytes: 1024 * 1024,
        };
        assert!(stats.format_size().ends_with("MB"));
    }

    #[test]
    fn test_model_stats_format_size_gb() {
        let stats = ModelStats {
            total_models: 0,
            candle_models: 0,
            onnx_models: 0,
            total_size_bytes: 1024 * 1024 * 1024,
        };
        assert!(stats.format_size().ends_with("GB"));
    }

    #[tokio::test]
    async fn test_load_already_loaded_returns_same_instance() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("reuse-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "reuse-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        let first = manager.load(&config).await.unwrap();
        let second = manager.load(&config).await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_unload_then_load_recreates_model() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("recreate-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "recreate-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };

        let _first = manager.load(&config).await.unwrap();
        manager.unload("recreate-model").await.unwrap();
        assert_eq!(manager.count().await, 0);

        let second = manager.load(&config).await;
        assert!(second.is_ok());
        assert_eq!(manager.count().await, 1);
        assert!(manager.is_loaded("recreate-model").await);
    }

    #[tokio::test]
    async fn test_slow_model_loader_get_model_path_and_cached() {
        let loader = SlowModelLoader::new(10);
        let config = ModelConfig {
            name: "slow-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/slow/path"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };
        let path = loader.get_model_path(&config).await.unwrap();
        assert_eq!(path, PathBuf::from("/slow/path"));
        assert!(loader.is_model_cached(&config).await);
    }

    #[tokio::test]
    async fn test_slow_model_loader_load_succeeds() {
        let cache_dir = tempdir().unwrap();
        let loader = SlowModelLoader::new(10);
        let config = ModelConfig {
            name: "slow-load".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("slow-load"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };
        let model = loader.load(&config).await.unwrap();
        assert_eq!(model.name(), "slow-load");
        assert_eq!(model.engine_type(), EngineType::Candle);
        assert!(model.reload().is_ok());
    }

    #[tokio::test]
    async fn test_failing_loader_get_model_path_and_cached() {
        let loader = FailingLoader;
        let config = ModelConfig {
            name: "fail-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/fail/path"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };
        let path = loader.get_model_path(&config).await.unwrap();
        assert_eq!(path, PathBuf::from("/fail/path"));
        assert!(!loader.is_model_cached(&config).await);
    }

    #[tokio::test]
    async fn test_unload_all_on_manager_with_single_model_logs() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let model_path = cache_dir.path().join("unload-single");
        create_test_model_file(&model_path);
        let config = ModelConfig {
            name: "unload-single".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        };
        manager.load(&config).await.unwrap();
        assert_eq!(manager.count().await, 1);
        manager.unload_all().await;
        assert_eq!(manager.count().await, 0);
    }

    // ===== LFRU 驻留管理测试 =====

    fn residency_config_for(name: &str, dir: &std::path::Path) -> ModelConfig {
        ModelConfig {
            name: name.to_string(),
            engine_type: EngineType::Candle,
            model_path: dir.join(name),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            quantized: false,
        }
    }

    #[tokio::test]
    async fn test_residency_evicts_lowest_score() {
        let dir = tempdir().unwrap();
        let loader = Arc::new(SlowModelLoader::new(0));
        let manager = ModelManager::with_loader(loader).with_residency(2, None);
        // A 冷（heat 10）、B 热（heat 50）；C 以 warmstart heat 100 进入。
        for name in ["model-a", "model-b"] {
            create_test_model_file(&dir.path().join(name));
        }
        manager
            .load(&residency_config_for("model-a", dir.path()))
            .await
            .unwrap();
        manager
            .load(&residency_config_for("model-b", dir.path()))
            .await
            .unwrap();
        manager.seed_heat("model-a", 10).await;
        manager.seed_heat("model-b", 50).await;
        manager.seed_heat("model-c", 100).await;
        create_test_model_file(&dir.path().join("model-c"));
        manager
            .load(&residency_config_for("model-c", dir.path()))
            .await
            .unwrap();
        // 恰好一次驱逐：最低分 A 出局，B/C 驻留。
        assert_eq!(manager.count().await, 2);
        assert!(
            !manager.is_loaded("model-a").await,
            "应驱逐 score 最低者 model-a"
        );
        assert!(manager.is_loaded("model-b").await);
        assert!(manager.is_loaded("model-c").await);
    }

    #[tokio::test]
    async fn test_residency_hysteresis_blocks_eviction() {
        // margin = victim/4 + 4（整数运算）。victim heat=100 时 margin=29，
        // 候选 heat=71 恰好放行（71+29>=100），70 则阻断（70+29<100）。
        for (cand_heat, expect_evict) in [(71u32, true), (70u32, false)] {
            let dir = tempdir().unwrap();
            let loader = Arc::new(SlowModelLoader::new(0));
            let manager = ModelManager::with_loader(loader).with_residency(1, None);
            create_test_model_file(&dir.path().join("hot"));
            manager
                .load(&residency_config_for("hot", dir.path()))
                .await
                .unwrap();
            manager.seed_heat("hot", 100).await;
            manager.seed_heat("cand", cand_heat).await;
            create_test_model_file(&dir.path().join("cand"));
            manager
                .load(&residency_config_for("cand", dir.path()))
                .await
                .unwrap();
            assert!(
                manager.is_loaded("cand").await,
                "候选 heat={} 应载入",
                cand_heat
            );
            assert_eq!(
                manager.is_loaded("hot").await,
                !expect_evict,
                "候选 heat={} 时 hot 应{}被驱逐",
                cand_heat,
                if expect_evict { "" } else { "不" }
            );
        }
    }

    #[tokio::test]
    async fn test_heat_saturates_no_wrap() {
        let dir = tempdir().unwrap();
        let loader = Arc::new(SlowModelLoader::new(0));
        let manager = ModelManager::with_loader(loader).with_residency(8, None);
        create_test_model_file(&dir.path().join("sat"));
        manager
            .load(&residency_config_for("sat", dir.path()))
            .await
            .unwrap();
        // u32 饱和不回绕。
        manager.seed_heat("sat", u32::MAX).await;
        manager
            .load(&residency_config_for("sat", dir.path()))
            .await
            .unwrap();
        assert_eq!(
            manager.heat_of("sat").await,
            u32::MAX,
            "heat 到顶不回绕不 panic"
        );
    }

    #[tokio::test]
    async fn test_inflight_model_skipped_for_eviction() {
        let dir = tempdir().unwrap();
        let loader = Arc::new(SlowModelLoader::new(0));
        let manager = ModelManager::with_loader(loader).with_residency(1, None);
        create_test_model_file(&dir.path().join("sat"));
        manager
            .load(&residency_config_for("sat", dir.path()))
            .await
            .unwrap();
        manager.seed_heat("sat", u32::MAX).await;
        // 持有两份外部引用 → strong_count = 3 > 2（in-flight）。
        let held1 = manager.get("sat").await.unwrap();
        let held2 = manager.get("sat").await.unwrap();
        assert!(Arc::strong_count(&held1) > 2);
        // 新人 heat 同样 MAX：迟滞放行，唯一阻断只能是 in-flight 跳过。
        manager.seed_heat("newbie", u32::MAX).await;
        create_test_model_file(&dir.path().join("newbie"));
        manager
            .load(&residency_config_for("newbie", dir.path()))
            .await
            .unwrap();
        assert!(
            manager.is_loaded("sat").await,
            "in-flight 模型本轮不得被驱逐"
        );
        assert!(manager.is_loaded("newbie").await);
        drop(held1);
        drop(held2);
        // 引用释放后再次触发：enforcement 恢复并收敛到上限（仅 c3 存活）。
        manager.seed_heat("c3", u32::MAX).await;
        create_test_model_file(&dir.path().join("c3"));
        manager
            .load(&residency_config_for("c3", dir.path()))
            .await
            .unwrap();
        assert_eq!(manager.count().await, 1, "引用释放后应收敛到 max_models=1");
        assert!(manager.is_loaded("c3").await);
    }

    #[tokio::test]
    async fn test_no_residency_means_no_eviction() {
        let dir = tempdir().unwrap();
        let loader = Arc::new(SlowModelLoader::new(0));
        let manager = ModelManager::with_loader(loader);
        for name in ["m1", "m2", "m3"] {
            create_test_model_file(&dir.path().join(name));
            manager
                .load(&residency_config_for(name, dir.path()))
                .await
                .unwrap();
        }
        assert_eq!(manager.count().await, 3, "默认不限制，无驱逐");
    }

    #[tokio::test]
    async fn test_heat_persist_roundtrip_via_manager() {
        let dir = tempdir().unwrap();
        let loader = Arc::new(SlowModelLoader::new(0));
        let manager = ModelManager::with_loader(loader);
        create_test_model_file(&dir.path().join("hm"));
        manager
            .load(&residency_config_for("hm", dir.path()))
            .await
            .unwrap();
        manager.seed_heat("hm", 33).await;
        let path = dir.path().join("model_heat.json");
        manager.save_heat_file(&path).await.unwrap();
        // 新 manager 经文件 warmstart 恢复。
        let loader2 = Arc::new(SlowModelLoader::new(0));
        let manager2 = ModelManager::with_loader(loader2);
        manager2.load_heat_file(&path).await;
        assert_eq!(manager2.heat_of("hm").await, 33);
    }
}
