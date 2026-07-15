// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! oxcache 配置示例：对比缓存启用/禁用时的引擎调用次数。
//!
//! 由于 `EmbeddingService` 不对外暴露 `CacheStats`，`processing_time_ms` 恒为 0，
//! 本示例用 `CountingEngine`（AtomicU64 计数器）+ 外部计时验证缓存命中。

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

fn make_request(text: &str) -> EmbedRequest {
    EmbedRequest {
        text: text.to_string(),
        normalize: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 oxcache 配置示例");
    println!("====================\n");

    let text = "hello vecboost cache";
    let dim = 8;

    // 1) 缓存禁用（new）：每次都调用引擎
    let counter_off = Arc::new(AtomicU64::new(0));
    let engine_off: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(
        CountingEngine::new(dim, Arc::clone(&counter_off)),
    ));
    let service_off = EmbeddingService::new(engine_off, None);

    println!("🔒 缓存禁用 (EmbeddingService::new)");
    let t0 = Instant::now();
    let _r1 = service_off.process_text(make_request(text), None).await?;
    let d1 = t0.elapsed();
    let t1 = Instant::now();
    let _r2 = service_off.process_text(make_request(text), None).await?;
    let d2 = t1.elapsed();
    println!(
        "  第 1 次: {:?}  引擎调用计数 = {}",
        d1,
        counter_off.load(Ordering::SeqCst)
    );
    println!(
        "  第 2 次: {:?}  引擎调用计数 = {}",
        d2,
        counter_off.load(Ordering::SeqCst)
    );
    println!("  → 两次都调用引擎\n");

    // 2) 缓存启用（with_cache）：第二次命中缓存
    let counter_on = Arc::new(AtomicU64::new(0));
    let engine_on: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(
        CountingEngine::new(dim, Arc::clone(&counter_on)),
    ));
    let service_on = EmbeddingService::with_cache(engine_on, None, 100);

    println!("✅ 缓存启用 (EmbeddingService::with_cache, capacity=100)");
    let t0 = Instant::now();
    let r1 = service_on.process_text(make_request(text), None).await?;
    let d1 = t0.elapsed();
    let t1 = Instant::now();
    let r2 = service_on.process_text(make_request(text), None).await?;
    let d2 = t1.elapsed();
    println!(
        "  第 1 次: {:?}  引擎调用计数 = {}  (cache miss)",
        d1,
        counter_on.load(Ordering::SeqCst)
    );
    println!(
        "  第 2 次: {:?}  引擎调用计数 = {}  (cache hit)",
        d2,
        counter_on.load(Ordering::SeqCst)
    );
    println!(
        "  → 第二次未调用引擎，向量相同: {}",
        r1.embedding == r2.embedding
    );

    println!("\n✅ 示例完成");
    Ok(())
}
