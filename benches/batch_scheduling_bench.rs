use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

use vecboost::domain::EmbedRequest;
use vecboost::pipeline::{Priority, QueuedRequest, RequestSource, ServiceRequest, assemble_batch};

fn make_queued_request(id: usize) -> QueuedRequest {
    QueuedRequest {
        request_id: format!("bench-{}", id),
        request: ServiceRequest::Embed(EmbedRequest {
            text: format!("text {}", id),
            normalize: Some(true),
        }),
        priority: Priority::Normal,
        submitted_at: Instant::now(),
        timeout: Duration::from_secs(30),
        source: RequestSource::Internal,
    }
}

/// Benchmark: steady load — 100 requests arriving uniformly, assembled via
/// `assemble_batch` time-window path. Measures batch sizes collected.
fn bench_steady_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let configs = [("wait_50ms", 50u64), ("wait_20ms", 20), ("wait_5ms", 5)];

    let mut group = c.benchmark_group("batch_steady_load");
    for (label, wait_ms) in configs {
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let pending: Arc<Mutex<VecDeque<QueuedRequest>>> =
                        Arc::new(Mutex::new(VecDeque::new()));
                    // 预填 100 个请求模拟稳态到达
                    {
                        let mut guard = pending.lock().await;
                        for i in 0..100 {
                            guard.push_back(make_queued_request(i));
                        }
                    }
                    let mut total = 0usize;
                    while total < 100 {
                        let first = {
                            let mut guard = pending.lock().await;
                            match guard.pop_front() {
                                Some(req) => req,
                                None => break,
                            }
                        };
                        let batch = assemble_batch(
                            first,
                            || {
                                let pending = Arc::clone(&pending);
                                async move { pending.lock().await.pop_front() }
                            },
                            32,
                            wait_ms,
                        )
                        .await;
                        total += batch.len();
                    }
                    total
                })
            })
        });
    }
    group.finish();
}

/// Benchmark: burst load — 50 requests arriving at once, drained via
/// `assemble_batch`. Measures burst drain batching.
fn bench_burst_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let configs = [("wait_50ms", 50u64), ("wait_20ms", 20), ("wait_5ms", 5)];

    let mut group = c.benchmark_group("batch_burst_load");
    for (label, wait_ms) in configs {
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut all_sizes = Vec::new();
                    for burst in 0..3 {
                        let pending: Arc<Mutex<VecDeque<QueuedRequest>>> =
                            Arc::new(Mutex::new(VecDeque::new()));
                        {
                            let mut guard = pending.lock().await;
                            for i in 0..50 {
                                guard.push_back(make_queued_request(burst * 50 + i));
                            }
                        }
                        let mut collected = 0usize;
                        while collected < 50 {
                            let first = {
                                let mut guard = pending.lock().await;
                                match guard.pop_front() {
                                    Some(req) => req,
                                    None => break,
                                }
                            };
                            let batch = assemble_batch(
                                first,
                                || {
                                    let pending = Arc::clone(&pending);
                                    async move { pending.lock().await.pop_front() }
                                },
                                32,
                                wait_ms,
                            )
                            .await;
                            collected += batch.len();
                            all_sizes.push(batch.len());
                        }
                    }
                    all_sizes
                })
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_steady_load, bench_burst_load);
criterion_main!(benches);
