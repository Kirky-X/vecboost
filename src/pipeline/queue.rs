// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 优先级请求队列已下沉 `crate::domain::scheduling`。
//! 本模块仅保留兼容再导出(`super::queue::*` / `crate::pipeline::*` 调用方不受影响)。

#[allow(unused_imports)]
pub use crate::domain::scheduling::{Priority, RequestSource};
pub use crate::domain::scheduling::{PriorityRequestQueue, QueuedRequest, ServiceRequest};
