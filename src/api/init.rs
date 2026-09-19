// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Process-wide `VecboostState` singleton for forge handlers.
//!
//! All forge handlers (HTTP/MCP/CLI) access kit capabilities via
//! `state()?.kit.require::<Module>()`. Initialized once by `main.rs`
//! via `init_state(VecboostState { kit })`.

use crate::VecboostState;
use crate::error::VecboostError;
use crate::i18n;
use std::sync::OnceLock;

static STATE: OnceLock<VecboostState> = OnceLock::new();

pub fn init_state(state: VecboostState) -> Result<(), VecboostError> {
    STATE
        .set(state)
        .map_err(|_| VecboostError::InternalError(i18n::tr("api-init-state-called")))
}

pub fn state() -> Result<VecboostState, VecboostError> {
    STATE
        .get()
        .cloned()
        .ok_or_else(|| VecboostError::InternalError(i18n::tr("api-init-state-missing")))
}
