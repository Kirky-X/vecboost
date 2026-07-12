// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

//! 请求流水线模块
//!
//! 实现请求流水线处理，包括优先级队列、动态 Worker 管理等

pub mod config;
pub(crate) mod handler;
mod priority;
mod queue;
mod response_channel;
mod scheduler;
mod worker;

pub use config::{PipelineConfig, PriorityConfig, QueueConfig, WorkerConfig};
pub use handler::handle_pipeline_request;
pub use priority::{Priority, PriorityCalculator, PriorityInput, RequestSource};
pub use queue::{PriorityRequestQueue, QueuedRequest};
pub use response_channel::ResponseChannel;
pub use scheduler::PipelineScheduler;
pub use worker::WorkerManager;
