// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod domain;
// endpoint 依赖 axum，仅在 http feature 下编译
#[cfg(feature = "http")]
pub mod endpoint;
pub mod inference;
pub mod performance;
// prometheus_exporter 依赖 prometheus crate（仅 http feature 引入），
// library 模式下不需要 Prometheus 指标收集
#[cfg(feature = "http")]
pub mod prometheus_exporter;

#[cfg(feature = "http")]
pub use endpoint::metrics_endpoint;
#[cfg(feature = "http")]
pub use endpoint::metrics_middleware;
pub use inference::InferenceCollector;
#[cfg(feature = "http")]
pub use prometheus_exporter::PrometheusCollector;

/// 批内去重率（T007）：`1 - unique/total`；空批为 0。
/// 无重复时为 0；n 条全重复（unique=1）时为 `(n-1)/n`。
pub fn inbatch_dedup_ratio(total: usize, unique: usize) -> f64 {
    if total == 0 || unique >= total {
        return 0.0;
    }
    1.0 - (unique as f64 / total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_ratio_no_dup_is_zero() {
        assert_eq!(inbatch_dedup_ratio(4, 4), 0.0);
        assert_eq!(inbatch_dedup_ratio(1, 1), 0.0);
    }

    #[test]
    fn test_dedup_ratio_all_dup() {
        let r = inbatch_dedup_ratio(4, 1);
        assert!((r - 0.75).abs() < 1e-12, "4 条全重复应为 (4-1)/4=0.75");
    }

    #[test]
    fn test_dedup_ratio_empty_is_zero() {
        assert_eq!(inbatch_dedup_ratio(0, 0), 0.0);
    }
}
