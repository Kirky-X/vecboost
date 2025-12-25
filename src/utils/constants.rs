// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub const MAX_TEXT_LENGTH: usize = 10_000;
pub const MIN_TEXT_LENGTH: usize = 1;
pub const MAX_BATCH_SIZE: usize = 100;
pub const MAX_SEARCH_RESULTS: usize = 1000;
pub const MAX_CONCURRENT_REQUESTS: usize = 100;
pub const DEFAULT_TOP_K: usize = 5;
pub const MAX_TOP_K: usize = 100;

pub const DEFAULT_CHUNK_SIZE: usize = 512;
pub const DEFAULT_OVERLAP_RATIO: f32 = 0.2;
pub const MIN_CHUNK_SIZE_RATIO: usize = 4;

pub const MAX_FILE_SIZE_BYTES: u64 = 100 * 1024 * 1024;
pub const MAX_FILE_SIZE_MB: usize = 100;
