// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Multi-protocol API layer — sdforge integration.
//!
//! `#[forge]`-annotated functions are registered via sdforge inventory for
//! HTTP/MCP/CLI protocol generation. Embedding handlers in `embedding.rs`,
//! auth handlers in `auth.rs`, state singleton in `init.rs`.

#![allow(unexpected_cfgs)]

#[cfg(feature = "auth")]
pub mod auth;
pub mod embedding;
pub mod init;
#[cfg(test)]
mod tests;

pub use init::{init_state, state};

#[cfg(all(test, feature = "mcp"))]
mod mcp_registration_tests {
    #[test]
    fn forge_macros_register_embed_tools() {
        let tools = sdforge::mcp::get_mcp_tools();
        let names: Vec<String> = tools.iter().map(|t| t.tool().name().to_string()).collect();
        assert!(
            names.iter().any(|n| n == "embed_text"),
            "embed_text not registered; tools: {:?}",
            names
        );
        assert!(
            names.iter().any(|n| n == "embed_batch"),
            "embed_batch not registered; tools: {:?}",
            names
        );
        assert!(
            names.iter().any(|n| n == "compute_similarity"),
            "compute_similarity not registered; tools: {:?}",
            names
        );
    }
}
