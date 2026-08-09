use criterion::{Criterion, criterion_group, criterion_main};
use std::collections::HashSet;
use std::time::Duration;

/// Trigram Jaccard similarity — local copy for baseline benchmarking.
/// Will be replaced by the actual implementation in Phase 2.
fn trigram_jaccard(a: &str, b: &str) -> f32 {
    if a.len() < 3 || b.len() < 3 {
        return 0.0;
    }
    let trigrams_a: HashSet<&[u8]> = a.as_bytes().windows(3).collect();
    let trigrams_b: HashSet<&[u8]> = b.as_bytes().windows(3).collect();
    let intersection = trigrams_a.intersection(&trigrams_b).count();
    let union_size = trigrams_a.union(&trigrams_b).count();
    if union_size == 0 {
        return 0.0;
    }
    intersection as f32 / union_size as f32
}

/// Generate paraphrased texts for benchmarking
fn generate_texts(count: usize) -> Vec<String> {
    let base_texts = [
        "今天天气怎么样",
        "机器学习很有趣",
        "Rust编程语言",
        "向量数据库搜索",
        "深度学习模型训练",
        "自然语言处理任务",
        "文本相似度计算",
        "缓存命中率优化",
        "高性能计算框架",
        "分布式系统架构",
        "GPU加速推理",
        "模型权重加载",
        "批量处理请求",
        "语义搜索算法",
        "内存池管理",
        "数据压缩存储",
        "实时流处理",
        "异步任务调度",
        "安全认证中间件",
        "API速率限制",
    ];
    (0..count)
        .map(|i| {
            let base = &base_texts[i % base_texts.len()];
            if i < base_texts.len() {
                format!("{}啊", base)
            } else {
                format!("{}的优化", base)
            }
        })
        .collect()
}

/// Benchmark: find the most similar entry in a list using trigram Jaccard
fn bench_trigram_search(c: &mut Criterion, entries: &[String], query: &str, label: &str) {
    c.bench_function(&format!("trigram_search_{}", label), |b| {
        b.iter(|| {
            let mut best_sim = 0.0f32;
            for entry in entries {
                let sim = trigram_jaccard(query, entry);
                if sim > best_sim {
                    best_sim = sim;
                }
            }
            best_sim
        })
    });
}

fn bench_exact_cache_operations(c: &mut Criterion) {
    // Benchmark exact cache hit latency (using HashMap as proxy)
    c.bench_function("exact_cache_hit", |b| {
        let mut map = std::collections::HashMap::new();
        for i in 0..1000 {
            map.insert(format!("text:hello_world_{}", i), vec![0.1f32; 128]);
        }
        b.iter(|| {
            let _ = map.get("text:hello_world_500");
        })
    });

    // Benchmark exact cache miss latency
    c.bench_function("exact_cache_miss", |b| {
        let map: std::collections::HashMap<String, Vec<f32>> = (0..1000)
            .map(|i| (format!("text:hello_world_{}", i), vec![0.1f32; 128]))
            .collect();
        b.iter(|| {
            let _ = map.get("text:completely_different_key");
        })
    });
}

fn bench_trigram_search_sizes(c: &mut Criterion) {
    let query = "今天天气怎么样啊";

    for size in [100, 1000, 10000] {
        let entries = generate_texts(size);
        bench_trigram_search(c, &entries, query, &format!("{}", size));
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(3))
        .sample_size(50);
    targets = bench_exact_cache_operations, bench_trigram_search_sizes
}
criterion_main!(benches);
