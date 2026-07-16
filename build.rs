// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::env;

fn main() {
    // Detect the target operating system
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Enable platform-specific features based on the target OS
    match target_os.as_str() {
        "macos" => {
            println!("cargo:warning=Building for macOS, enabling Metal support");
        }
        "linux" => {
            println!("cargo:warning=Building for Linux, CUDA support available");
        }
        "windows" => {
            println!("cargo:warning=Building for Windows, CUDA support available");
        }
        _ => {
            println!("cargo:warning=Building for unknown OS: {}", target_os);
        }
    }

    // Build gRPC service only when grpc feature is enabled
    #[cfg(feature = "grpc")]
    {
        println!("cargo:rerun-if-changed=proto/embedding.proto");

        tonic_prost_build::configure()
            .build_server(true)
            .build_client(true)
            .compile_protos(&["proto/embedding.proto"], &["proto/"])
            .expect("Failed to compile proto files");
    }
}
