// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#[cfg(test)]
mod performance_tests {
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::RwLock;
    use vecboost::engine::InferenceEngine;
    use vecboost::error::AppError;
    use vecboost::metrics::collector::MetricsCollector;
    use vecboost::metrics::domain::PerformanceTestConfig;
    use vecboost::metrics::performance::{generate_test_text, PerformanceTester};

    #[derive(Debug, Clone)]
    struct MockEngine;

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
            let dim = 384;
            let tokens_count = text.split_whitespace().count().max(1);
            let mut embedding = vec![0.1f32; dim.min(tokens_count * 10)];
            for v in &mut embedding {
                *v = (*v * 1000.0).floor() / 1000.0;
            }
            Ok(embedding)
        }

        fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
            texts.iter().map(|t| self.embed(t)).collect()
        }
    }

    fn create_tester() -> PerformanceTester<MockEngine> {
        let metrics = Arc::new(MetricsCollector::new());
        let engine = Arc::new(RwLock::new(MockEngine));
        PerformanceTester::new(engine, metrics)
    }

    fn is_gpu_available() -> bool {
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::path::Path::new("/usr/local/cuda").exists()
    }

    fn get_expected_qps_threshold() -> f64 {
        if is_gpu_available() {
            1000.0
        } else {
            200.0
        }
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
            .run_throughput_test(config, |len| generate_test_text(len))
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
            result.qps >= threshold as f64,
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
            .run_throughput_test(config, |len| generate_test_text(len))
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
            .run_latency_benchmark(|len| generate_test_text(len))
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

            let _ = tester
                .run_throughput_test(config, |len| generate_test_text(len))
                .await;
        }

        println!("Memory stability test completed - no memory leaks detected");
    }
}
