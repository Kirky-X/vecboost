// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod app;
pub mod model;

#[cfg(feature = "config")]
pub mod app_config;

#[cfg(feature = "config")]
pub use app_config::AppConfig as ConfersAppConfig;
