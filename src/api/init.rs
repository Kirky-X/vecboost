// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Process-wide `VecboostState` singleton for forge handlers.
//!
//! All forge handlers (HTTP/MCP/CLI) access kit capabilities via
//! `state()?.kit.require::<Module>()`. Initialized once by `main.rs`
//! via `init_state(VecboostState { kit })`.

use crate::VecboostState;
use crate::error::VecboostError;
use std::sync::OnceLock;

static STATE: OnceLock<VecboostState> = OnceLock::new();

pub fn init_state(state: VecboostState) {
    let _ = STATE.set(state);
}

pub fn state() -> Result<VecboostState, VecboostError> {
    STATE
        .get()
        .cloned()
        .ok_or_else(|| VecboostError::InternalError("init_state not called".to_string()))
}
