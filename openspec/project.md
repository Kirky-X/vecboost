# Project Context

## Purpose

**VecBoost** is a high-performance Rust vector embedding service optimized for production. Built with Axum 0.7 + Tokio, it supports HTTP REST and gRPC protocols with Candle (default) or ONNX Runtime for ML inference. The service provides fast, scalable embedding generation with GPU acceleration support (CUDA, Metal) and includes authentication, rate limiting, and Prometheus metrics.

## Tech Stack

- **Language**: Rust 2024 edition
- **Web Framework**: Axum 0.7 + Tokio 1.35
- **gRPC**: Tonic 0.12 + Prost
- **ML Inference Engines**: Candle 0.9.2, ONNX Runtime 2.0
- **GPU Acceleration**: CUDA (NVIDIA), Metal (Apple Silicon), cudarc 0.18.2
- **API Documentation**: Utoipa 5.0 with Swagger UI
- **Monitoring**: Prometheus 0.13 + axum-prometheus
- **Rate Limiting**: Redis (optional) for distributed rate limiting
- **Authentication**: JWT (jsonwebtoken), Argon2 password hashing, AES-GCM encryption
- **Configuration**: config 0.14 with TOML
- **Serialization**: serde + serde_json

## Project Conventions

### Code Style

- Follow KISS (Keep It Simple, Stupid) and DRY principles
- Implement only what was requested—no extra features or "future-proofing"
- Use `cargo fmt` for automatic formatting (4-space indentation, max 100 chars)
- Group imports: `std` → `external` → `internal`
- Use explicit imports, avoid glob imports
- No comments unless necessary—code should be self-documenting

**Naming Conventions**:
- **Modules**: `snake_case` (e.g., `embedding_service`)
- **Structs/Enums**: `PascalCase` (e.g., `EmbeddingRequest`)
- **Functions/Variables**: `snake_case` (e.g., `calculate_similarity`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Type parameters**: `PascalCase` (e.g., `T: Clone`)

**Type Conventions**:
- Use `Arc<RwLock<T>>` or `Arc<Mutex<T>>` for shared mutable state
- Prefer `Option<T>` over null values
- Use `Result<T, AppError>` for fallible operations
- Define errors using `thiserror` in `error.rs`
- Use `anyhow` for error propagation in `main.rs`

### Architecture Patterns

**Engine Abstraction**: Implement `InferenceEngine` trait for new engines; add variants to `AnyEngine` enum; configure in `config/model.rs` with `EngineType`

**Service Layer**: Wrap logic in service structs (e.g., `EmbeddingService`); accept `AppState` for shared access; use `Arc<RwLock<>>` for thread-safe state

**Middleware Chain**:
- Authentication: JWT token validation
- CSRF: Origin + token validation
- Rate limiting: global/IP/user/API key dimensions

**Module Visibility**:
- Public modules: `audit`, `auth`, `config`, `domain`, `engine`, `grpc`, `metrics`, `rate_limit`, `routes`, `security`, `service`, `utils`, `error`
- Internal modules: `cache`, `device`, `model`, `monitor`, `text`

### Testing Strategy

```bash
# All tests with all features
cargo test --all-features

# Unit tests only
cargo test --lib

# Integration tests only
cargo test --tests

# Run a single test by name
cargo test test_name
```

All public APIs should include unit tests covering normal and edge cases.

### Git Workflow

**Commit Style**: Conventional Commits
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Restructuring
- `perf`: Performance
- `test`: Testing
- `chore`: Build/dependencies

**Feature Flags**:
- `cuda`: NVIDIA GPU acceleration
- `metal`: Apple Silicon GPU acceleration
- `onnx`: ONNX Runtime engine
- `grpc`: gRPC server support
- `redis`: Redis support for distributed rate limiting

## Domain Context

**Vector Embedding Service**: Generates vector representations of text using ML models (Candle or ONNX Runtime). Designed for production deployment with:
- Batch processing support
- Model caching and hot reload
- GPU memory management
- Multi-model support
- REST and gRPC API access

**Default Ports**:
- HTTP API: 9002
- gRPC API: 50051
- Prometheus metrics: 9090

## Important Constraints

- JWT secret must be at least 32 characters
- All `pub` APIs must have `///` doc comments
- Code must pass `cargo clippy --all-features -- -D warnings`
- Use environment variables for all secrets—never hardcode
- Validate and sanitize all user inputs

## External Dependencies

- **Hugging Face Hub**: Model downloads via `hf-hub` library
- **Redis** (optional): Distributed rate limiting backend
- **GPU Runtimes**: CUDA toolkit (Linux), Metal (macOS), ONNX Runtime
- **Model Storage**: Local filesystem cache via `tempfile` and `home` crates
