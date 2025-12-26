use std::env;

fn main() {
    // Detect the target operating system
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Enable platform-specific features based on the target OS
    match target_os.as_str() {
        "macos" => {
            println!("cargo:warning=Building for macOS, enabling Metal support");
            // For now, we don't auto-enable features, but we can print warnings
            // println!("cargo:rustc-cfg=feature=\"metal\"");
        }
        "linux" => {
            println!("cargo:warning=Building for Linux, CUDA support available");
            // println!("cargo:rustc-cfg=feature=\"cuda\"");
        }
        "windows" => {
            println!("cargo:warning=Building for Windows, CUDA support available");
            // println!("cargo:rustc-cfg=feature=\"cuda\"");
        }
        _ => {
            println!("cargo:warning=Building for unknown OS: {}", target_os);
        }
    }

    // Build gRPC service if needed
    println!("cargo:rerun-if-changed=proto/embedding.proto");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/embedding.proto"], &["proto/"])
        .expect("Failed to compile proto files");
}
