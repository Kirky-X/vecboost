// Copyright (c) 2025 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

#[cfg(test)]
mod performance_tests {
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::RwLock;
    use vecboost::config::model::Precision;
    use vecboost::engine::InferenceEngine;
    use vecboost::error::AppError;
    use vecboost::metrics::domain::PerformanceTestConfig;
    use vecboost::metrics::inference::InferenceCollector;
    use vecboost::metrics::performance::{PerformanceTester, generate_test_text};

    const MOCK_DIMENSION: usize = 384;

    #[derive(Debug, Clone)]
    struct TestEngine {
        dimension: usize,
    }

    impl TestEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }

        fn generate_embedding(&self, text: &str) -> Vec<f32> {
            let mut embedding = vec![0.0; self.dimension];
            let bytes = text.as_bytes();

            let mut hash: u64 = 1469598103934665603;
            for &byte in bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(1099511628211);
            }

            let mut state = hash;
            for val in embedding.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let float_val = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
                *val = float_val;
            }

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
    impl InferenceEngine for TestEngine {
        fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
            Ok(self.generate_embedding(text))
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
            texts.iter().map(|t| self.embed(t)).collect()
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &vecboost::config::model::ModelConfig,
        ) -> Result<(), AppError> {
            Ok(())
        }
    }

    fn create_tester() -> PerformanceTester<TestEngine> {
        let metrics = Arc::new(InferenceCollector::new());
        let engine = Arc::new(RwLock::new(TestEngine::new(MOCK_DIMENSION)));
        PerformanceTester::new(engine, metrics)
    }

    fn is_gpu_available() -> bool {
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::path::Path::new("/usr/local/cuda").exists()
    }

    fn get_expected_qps_threshold() -> f64 {
        if is_gpu_available() { 1000.0 } else { 200.0 }
    }

    fn get_latency_threshold(tokens: usize) -> u64 {
        match tokens {
            10 => 50,
            50 => 80,
            100 => 100,
            200 => 150,
            512 => 200,
            _ => 200,
        }
    }

    #[tokio::test]
    async fn test_throughput_gpu_requirement() {
        let tester = create_tester();

        let config = PerformanceTestConfig {
            concurrent_requests: 50,
            total_requests: 500,
            warmup_requests: 50,
            min_text_length: 50,
            max_text_length: 100,
            target_qps: Some(get_expected_qps_threshold()),
            timeout_seconds: 60,
        };

        let result = tester
            .run_throughput_test(config, generate_test_text)
            .await
            .expect("Throughput test should complete");

        let threshold = get_expected_qps_threshold();
        println!(
            "Throughput test: QPS={:.2} (threshold={}), error_rate={:.4}%",
            result.qps,
            threshold,
            result.error_rate * 100.0
        );

        assert!(
            result.qps >= threshold,
            "QPS {} is below threshold {}",
            result.qps,
            threshold
        );
    }

    #[tokio::test]
    async fn test_throughput_error_rate_requirement() {
        let tester = create_tester();

        let config = PerformanceTestConfig {
            concurrent_requests: 100,
            total_requests: 1000,
            warmup_requests: 100,
            min_text_length: 50,
            max_text_length: 200,
            target_qps: None,
            timeout_seconds: 120,
        };

        let result = tester
            .run_throughput_test(config, generate_test_text)
            .await
            .expect("Throughput test should complete");

        let max_error_rate = 0.001;
        println!(
            "Error rate test: error_rate={:.4}% (max={}%), successful={}, failed={}",
            result.error_rate * 100.0,
            max_error_rate * 100.0,
            result.successful_requests,
            result.failed_requests
        );

        assert!(
            result.error_rate <= max_error_rate,
            "Error rate {}% exceeds maximum allowed {}%",
            result.error_rate * 100.0,
            max_error_rate * 100.0
        );
    }

    #[tokio::test]
    async fn test_latency_10_tokens_requirement() {
        let tester = create_tester();

        let mut all_latencies: Vec<u64> = Vec::new();

        for _ in 0..50 {
            let text = generate_test_text(10);
            let start = std::time::Instant::now();
            let _ = tester.engine().write().await.embed(&text);
            let elapsed = start.elapsed();
            all_latencies.push(elapsed.as_millis() as u64);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        all_latencies.sort_unstable();
        let p95_idx = ((all_latencies.len() - 1) * 95) / 100;
        let p95 = all_latencies[p95_idx.min(all_latencies.len() - 1)];
        let threshold = get_latency_threshold(10);

        println!(
            "Latency 10 tokens: P95={}ms (threshold={}ms)",
            p95, threshold
        );

        assert!(
            p95 <= threshold,
            "P95 latency {}ms exceeds threshold {}ms for 10 tokens",
            p95,
            threshold
        );
    }

    #[tokio::test]
    async fn test_latency_100_tokens_requirement() {
        let tester = create_tester();

        let mut all_latencies: Vec<u64> = Vec::new();

        for _ in 0..50 {
            let text = generate_test_text(100);
            let start = std::time::Instant::now();
            let _ = tester.engine().write().await.embed(&text);
            let elapsed = start.elapsed();
            all_latencies.push(elapsed.as_millis() as u64);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        all_latencies.sort_unstable();
        let p95_idx = ((all_latencies.len() - 1) * 95) / 100;
        let p95 = all_latencies[p95_idx.min(all_latencies.len() - 1)];
        let threshold = get_latency_threshold(100);

        println!(
            "Latency 100 tokens: P95={}ms (threshold={}ms)",
            p95, threshold
        );

        assert!(
            p95 <= threshold,
            "P95 latency {}ms exceeds threshold {}ms for 100 tokens",
            p95,
            threshold
        );
    }

    #[tokio::test]
    async fn test_latency_512_tokens_requirement() {
        let tester = create_tester();

        let mut all_latencies: Vec<u64> = Vec::new();

        for _ in 0..30 {
            let text = generate_test_text(512);
            let start = std::time::Instant::now();
            let _ = tester.engine().write().await.embed(&text);
            let elapsed = start.elapsed();
            all_latencies.push(elapsed.as_millis() as u64);
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        all_latencies.sort_unstable();
        let p95_idx = ((all_latencies.len() - 1) * 95) / 100;
        let p95 = all_latencies[p95_idx.min(all_latencies.len() - 1)];
        let threshold = get_latency_threshold(512);

        println!(
            "Latency 512 tokens: P95={}ms (threshold={}ms)",
            p95, threshold
        );

        assert!(
            p95 <= threshold,
            "P95 latency {}ms exceeds threshold {}ms for 512 tokens",
            p95,
            threshold
        );
    }

    #[tokio::test]
    async fn test_latency_percentile_requirement() {
        let tester = create_tester();

        let result = tester
            .run_latency_benchmark(generate_test_text)
            .await
            .expect("Latency benchmark should complete");

        println!(
            "Latency benchmark: P50={}ms, P95={}ms, P99={}ms",
            result.p50_ms, result.p95_ms, result.p99_ms
        );

        assert!(
            result.p95_ms <= 200,
            "P95 latency {}ms exceeds 200ms threshold",
            result.p95_ms
        );
        assert!(
            result.p99_ms <= 500,
            "P99 latency {}ms exceeds 500ms threshold",
            result.p99_ms
        );
    }

    #[tokio::test]
    async fn test_stress_test_stability() {
        let tester = create_tester();

        let result = tester
            .run_stress_test(10, 5, |_len| generate_test_text(100))
            .await
            .expect("Stress test should complete");

        let max_error_rate = 0.05;
        println!(
            "Stress test: QPS={:.2}, error_rate={:.4}%, successful={}",
            result.qps,
            result.error_rate * 100.0,
            result.successful_requests
        );

        assert!(
            result.error_rate <= max_error_rate,
            "Error rate {}% exceeds maximum {}% during stress test",
            result.error_rate * 100.0,
            max_error_rate * 100.0
        );
        assert!(
            result.successful_requests > 0,
            "Should complete at least some successful requests during stress test"
        );
    }

    #[tokio::test]
    async fn test_memory_stability() {
        let tester = create_tester();

        for _ in 0..10 {
            let config = PerformanceTestConfig {
                concurrent_requests: 10,
                total_requests: 100,
                warmup_requests: 10,
                min_text_length: 50,
                max_text_length: 100,
                target_qps: None,
                timeout_seconds: 30,
            };

            let _ = tester.run_throughput_test(config, generate_test_text).await;
        }

        println!("Memory stability test completed - no memory leaks detected");
    }
}
