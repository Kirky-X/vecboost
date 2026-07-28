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
pub use inference::InferenceCollector;
#[cfg(feature = "http")]
pub use prometheus_exporter::PrometheusCollector;
