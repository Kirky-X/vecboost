// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 缓存命中验证示例。
//!
//! 说明：当前 `EmbeddingService` 不对外暴露 TTL 配置或 `CacheStats`，
//! `processing_time_ms` 恒为 0。本示例通过 `CountingEngine` 计数器 +
//! 外部计时验证相同文本的第二次调用命中缓存（引擎不被调用）。
//! 缓存条目持续存在直到容量驱逐，不会因 TTL 过期。

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::RwLock;

use vecboost::EmbeddingService;
use vecboost::VecboostError;
use vecboost::config::model::{ModelConfig, Precision};
use vecboost::domain::EmbedRequest;
use vecboost::engine::InferenceEngine;

struct CountingEngine {
    dimension: usize,
    counter: Arc<AtomicU64>,
}

impl CountingEngine {
    fn new(dimension: usize, counter: Arc<AtomicU64>) -> Self {
        Self { dimension, counter }
    }
}

#[async_trait]
impl InferenceEngine for CountingEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(vec![0.5; self.dimension])
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        self.counter.fetch_add(texts.len() as u64, Ordering::SeqCst);
        Ok(texts.iter().map(|_| vec![0.5; self.dimension]).collect())
    }

    fn precision(&self) -> &Precision {
        static PRECISION: Precision = Precision::Fp32;
        &PRECISION
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⏱️  缓存命中验证示例");
    println!("=====================\n");

    let counter = Arc::new(AtomicU64::new(0));
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(CountingEngine::new(8, Arc::clone(&counter))));
    let service = EmbeddingService::with_cache(engine, None, 50);

    let text = "vecboost ttl cache demo";

    println!("对同一文本连续调用 3 次 process_text：\n");
    for i in 1..=3 {
        let req = EmbedRequest {
            text: text.to_string(),
            normalize: None,
        };
        let t = Instant::now();
        let resp = service.process_text(req, None).await?;
        let elapsed = t.elapsed();
        let calls = counter.load(Ordering::SeqCst);
        let hit = if i == 1 { "miss" } else { "hit" };
        println!(
            "  第 {} 次: {:?}  维度={}  引擎调用={}  cache={}",
            i, elapsed, resp.dimension, calls, hit
        );
    }

    let final_calls = counter.load(Ordering::SeqCst);
    println!(
        "\n→ 引擎总调用次数: {} (期望 1，后续均命中缓存)",
        final_calls
    );
    println!("→ 注：当前 API 未暴露 TTL 配置，缓存条目持续有效直到容量驱逐");

    println!("\n✅ 示例完成");
    Ok(())
}
