use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vecboost::utils::vector::{
    cosine_similarity, dot_product, euclidean_distance, manhattan_distance,
};

/// Generate a deterministic f32 vector for benchmarking.
fn generate_vector(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i as f32) * 0.001 + seed).sin())
        .collect()
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    for dim in [128, 384, 768, 1024] {
        let v1 = generate_vector(dim, 0.0);
        let v2 = generate_vector(dim, 1.0);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| cosine_similarity(&v1, &v2).unwrap())
        });
    }
    group.finish();
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");
    for dim in [128, 384, 768, 1024] {
        let v1 = generate_vector(dim, 0.0);
        let v2 = generate_vector(dim, 1.0);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| euclidean_distance(&v1, &v2).unwrap())
        });
    }
    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    for dim in [128, 384, 768, 1024] {
        let v1 = generate_vector(dim, 0.0);
        let v2 = generate_vector(dim, 1.0);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| dot_product(&v1, &v2).unwrap())
        });
    }
    group.finish();
}

fn bench_manhattan_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("manhattan_distance");
    for dim in [128, 384, 768, 1024] {
        let v1 = generate_vector(dim, 0.0);
        let v2 = generate_vector(dim, 1.0);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| manhattan_distance(&v1, &v2).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_euclidean_distance,
    bench_dot_product,
    bench_manhattan_distance,
);
criterion_main!(benches);
