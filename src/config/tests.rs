// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Tests for confers-based configuration loading (T016).
//!
//! Covers three scenarios required by the spec:
//! 1. Loading `AppConfig` from `config_minimal.toml`
//! 2. Environment variable `VECBOOST_JWT_SECRET` overriding TOML values
//! 3. Hot-reload subscribe callback via `confers::bus::InMemoryBus`

use std::io::Write;
use std::sync::Mutex;

use super::app_config::AppConfig;

/// Serialises tests that touch process-global environment variables.
///
/// Rust runs tests in parallel by default; without this lock, a test that
/// sets `VECBOOST_JWT_SECRET` can pollute another test's env-var snapshot
/// between cleanup and config load. Each env-var-touching test must hold
/// this lock for its entire body.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Helper: write a minimal TOML config to a temp file and return its path.
fn write_temp_toml(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("test_config.toml");
    let mut file = std::fs::File::create(&path).expect("failed to create temp config file");
    file.write_all(content.as_bytes())
        .expect("failed to write temp config");
    (dir, path)
}

/// Test 1: Load `AppConfig` from `config_minimal.toml` and verify key fields.
///
/// Verifies that confers correctly deserialises the TOML file into the
/// nested `AppConfig` struct, including server port, model repo, and
/// embedding settings.
#[test]
fn test_confers_load_from_minimal_toml() {
    let toml_content = r#"
[server]
host = "127.0.0.1"
port = 9002
timeout = 30

[model]
model_repo = "BAAI/bge-m3"
use_gpu = false
batch_size = 8
expected_dimension = 1024

[embedding]
default_aggregation = "mean"
similarity_metric = "cosine"
cache_enabled = false
cache_size = 0
max_batch_size = 32

[monitoring]
memory_limit_mb = 512
metrics_enabled = false
log_level = "info"

[rate_limit]
enabled = false

[pipeline]
enabled = false

[audit]
enabled = false
"#;

    let (_dir, path) = write_temp_toml(toml_content);
    let config = AppConfig::load_via_confers_with_path(&path).expect("confers load should succeed");

    // Verify server config loaded from TOML
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 9002);
    assert_eq!(config.server.timeout, Some(30));

    // Verify model config
    assert_eq!(config.model.model_repo, "BAAI/bge-m3");
    assert!(!config.model.use_gpu);
    assert_eq!(config.model.batch_size, 8);
    assert_eq!(config.model.expected_dimension, Some(1024));

    // Verify embedding config
    assert_eq!(config.embedding.default_aggregation, "mean");
    assert!(!config.embedding.cache_enabled);
    assert_eq!(config.embedding.max_batch_size, 32);

    // Verify rate_limit disabled in minimal config
    assert!(!config.rate_limit.enabled);

    // Verify audit disabled in minimal config
    assert!(!config.audit.enabled);
}

/// Test 2: Environment variable `VECBOOST_JWT_SECRET` overrides TOML value.
///
/// Sets `VECBOOST_JWT_SECRET` to a non-empty value, loads config, and
/// verifies that `auth.jwt_secret` reflects the env var rather than the
/// TOML default (None).
///
/// NOTE: Env vars are process-global; this test may race with parallel tests
/// that also touch `VECBOOST_JWT_SECRET`. The cleanup at the end minimises
/// the window.
#[test]
fn test_confers_env_var_jwt_secret_override() {
    let _guard = ENV_LOCK.lock().expect("env lock poisoned");

    // Ensure clean state before test
    unsafe {
        std::env::remove_var("VECBOOST_JWT_SECRET");
    }

    let toml_content = r#"
[server]
host = "0.0.0.0"
port = 8080

[model]
model_repo = "test/model"
use_gpu = false
batch_size = 1
expected_dimension = 128

[embedding]
default_aggregation = "mean"
similarity_metric = "cosine"
cache_enabled = false
cache_size = 0
max_batch_size = 1

[monitoring]
metrics_enabled = false

[rate_limit]
enabled = false

[pipeline]
enabled = false

[audit]
enabled = false
"#;

    let (_dir, path) = write_temp_toml(toml_content);

    // Load without env var - jwt_secret should be None (default)
    let config_no_env =
        AppConfig::load_via_confers_with_path(&path).expect("confers load should succeed");
    assert!(
        config_no_env.auth.jwt_secret.is_none(),
        "jwt_secret should be None when VECBOOST_JWT_SECRET is not set"
    );

    // Set env var and reload - jwt_secret should be overridden
    let test_secret = "test-jwt-secret-from-env-var-32+chars";
    assert!(
        test_secret.len() >= 32,
        "test secret should be at least 32 chars"
    );
    unsafe {
        std::env::set_var("VECBOOST_JWT_SECRET", test_secret);
    }

    let config_with_env =
        AppConfig::load_via_confers_with_path(&path).expect("confers load should succeed");
    assert_eq!(
        config_with_env.auth.jwt_secret,
        Some(test_secret.to_string()),
        "VECBOOST_JWT_SECRET should override TOML default"
    );

    // Cleanup
    unsafe {
        std::env::remove_var("VECBOOST_JWT_SECRET");
    }
}

/// Test 3: Hot-reload subscribe callback via `confers::bus::InMemoryBus`.
///
/// Verifies the subscribe API: create an `InMemoryBus`, subscribe to
/// config change events, publish a `ConfigChangeEvent`, and verify the
/// subscriber receives it. This models the hot-reload notification flow
/// where a watcher publishes change events and config consumers react.
#[tokio::test]
async fn test_confers_inmemorybus_subscribe_publish() {
    use confers::ConfigBus;
    use confers::bus::{ConfigChangeEvent, InMemoryBus};
    use futures::StreamExt;

    let bus = InMemoryBus::new();

    // Subscribe before publishing to ensure receipt
    let mut stream = bus
        .subscribe()
        .await
        .expect("subscribe should return a stream");

    // Publish a config change event
    let event = ConfigChangeEvent::new(
        "test-instance",
        "file://config.toml",
        vec!["server.port".to_string(), "auth.jwt_secret".to_string()],
        "checksum-abc123",
    );

    bus.publish(event.clone())
        .await
        .expect("publish should succeed");

    // Receive the event from the stream
    let received = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("should receive event within timeout")
        .expect("stream should not end");

    // Verify event content
    assert_eq!(received.instance_id, "test-instance");
    assert_eq!(received.source, "file://config.toml");
    assert_eq!(
        received.changed_keys,
        vec!["server.port".to_string(), "auth.jwt_secret".to_string()]
    );
    assert_eq!(received.checksum, "checksum-abc123");
}

/// Test 4: Defaults are applied when config file is missing.
///
/// When `file_optional` is given a non-existent path, confers should fall
/// back to `Default` implementations for all sub-configs.
#[test]
fn test_confers_defaults_when_no_file() {
    let _guard = ENV_LOCK.lock().expect("env lock poisoned");

    // Ensure clean env state
    unsafe {
        std::env::remove_var("VECBOOST_JWT_SECRET");
        std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
    }

    let non_existent = std::path::PathBuf::from("/tmp/vecboost_nonexistent_config_9999.toml");
    let config = AppConfig::load_via_confers_with_path(&non_existent)
        .expect("should load with defaults when file is missing");

    // Verify defaults from app.rs Default impls
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 3000);
    assert_eq!(config.model.model_repo, "BAAI/bge-m3");
    assert_eq!(config.model.batch_size, 32);
    assert_eq!(config.embedding.default_aggregation, "mean");
    assert!(config.embedding.cache_enabled);
    assert!(!config.auth.enabled);
    assert!(config.auth.jwt_secret.is_none());
}

/// Test 5: TOML values override struct defaults.
///
/// Loads a TOML with non-default values and verifies they take precedence
/// over the `Default` implementations.
#[test]
fn test_confers_toml_overrides_defaults() {
    let toml_content = r#"
[server]
host = "10.0.0.1"
port = 7777
grpc_enabled = true

[model]
model_repo = "custom/repo"
use_gpu = true
batch_size = 64
expected_dimension = 768

[embedding]
default_aggregation = "max"
similarity_metric = "dot"
cache_enabled = true
cache_size = 4096
max_batch_size = 128

[monitoring]
metrics_enabled = true
log_level = "debug"

[rate_limit]
enabled = true

[pipeline]
enabled = true

[audit]
enabled = true
"#;

    let (_dir, path) = write_temp_toml(toml_content);
    let config = AppConfig::load_via_confers_with_path(&path).expect("confers load should succeed");

    // Verify TOML overrode defaults
    assert_eq!(config.server.host, "10.0.0.1");
    assert_eq!(config.server.port, 7777);
    assert!(config.server.grpc_enabled);

    assert_eq!(config.model.model_repo, "custom/repo");
    assert!(config.model.use_gpu);
    assert_eq!(config.model.batch_size, 64);
    assert_eq!(config.model.expected_dimension, Some(768));

    assert_eq!(config.embedding.default_aggregation, "max");
    assert_eq!(config.embedding.similarity_metric, "dot");
    assert_eq!(config.embedding.cache_size, 4096);
    assert_eq!(config.embedding.max_batch_size, 128);

    assert!(config.rate_limit.enabled);
    assert!(config.audit.enabled);
}
