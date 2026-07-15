// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use vecboost::VecboostError;
use vecboost::config::model::{ModelConfig, Precision};
use vecboost::engine::InferenceEngine;
use vecboost::pipeline::{PriorityRequestQueue, ResponseChannel, WorkerConfig, WorkerManager};
use vecboost::service::embedding::EmbeddingService;

struct MockEngine {
    dimension: usize,
}

impl MockEngine {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl InferenceEngine for MockEngine {
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

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 WorkerManager 启动与扩缩容示例");
    println!("==================================\n");

    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(MockEngine::new(8)));
    let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

    let queue = Arc::new(PriorityRequestQueue::new(100));
    let response_channel = Arc::new(ResponseChannel::new());
    let config = WorkerConfig {
        min_workers: 1,
        max_workers: 8,
        ..Default::default()
    };

    let manager = WorkerManager::new(queue, response_channel, config, service);

    println!(
        "阶段 1: start() 前 worker 数 = {}",
        manager.current_workers()
    );

    manager.start().await?;
    println!(
        "阶段 2: start() 后 worker 数 = {} (min_workers=1)",
        manager.current_workers()
    );

    manager.spawn_worker().await;
    manager.spawn_worker().await;
    println!(
        "阶段 3: spawn_worker() x2 后 worker 数 = {} (手动扩容至 3)",
        manager.current_workers()
    );

    tokio::time::timeout(Duration::from_secs(15), manager.shutdown())
        .await
        .expect("shutdown 应在 15s 内完成");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        if manager.current_workers() == 0 {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    println!(
        "阶段 4: shutdown() 后 worker 数 = {}",
        manager.current_workers()
    );

    println!("\n✅ WorkerManager 示例完成");
    Ok(())
}
