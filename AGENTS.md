<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# AGENTS.md

This file provides guidance for agentic coding agents working in this repository.

## Project Overview

**VecBoost** is a high-performance Rust vector embedding service optimized for production. Built with Axum 0.7 + Tokio, it supports HTTP REST and gRPC protocols with Candle (default) or ONNX Runtime for ML inference.

## Essential Commands

### Build
```bash
# CPU-only release build
cargo build --release

# GPU builds (choose one)
cargo build --release --features cuda      # NVIDIA CUDA
cargo build --release --features metal     # Apple Silicon
cargo build --release --features onnx      # ONNX Runtime
cargo build --release --features grpc      # gRPC server

# All features
cargo build --release --features cuda,metal,onnx,grpc

# Run service
cargo run --release
cargo run --release -- --config config.toml
```

### Testing
```bash
# All tests with all features
cargo test --all-features

# Unit tests only
cargo test --lib

# Integration tests only
cargo test --tests

# Run a single test by name
cargo test test_name

# Run tests with output
cargo test --all-features -- --nocapture test_name
```

### Code Quality
```bash
# Format code
cargo fmt

# Lint (must pass with zero warnings)
cargo clippy --all-features -- -D warnings

# Check compilation
cargo check --all-features
```

### Examples
```bash
# Download model
cargo run --example download_model

# GPU tests (requires GPU feature)
cargo run --example gpu_candle_engine_test --features cuda
cargo run --example gpu_onnx_engine_test --features cuda,onnx
```

## Code Style Guidelines

### General Principles
- Follow KISS (Keep It Simple, Stupid) and DRY (Don't Repeat Yourself)
- Implement only what was requested—no extra features
- Avoid unnecessary abstractions or "future-proofing"
- Keep solutions minimal and maintainable

### Imports
- Group imports: std → external → internal
- Use explicit imports rather than glob imports
- Remove unused imports immediately

### Formatting
- Use `cargo fmt` for automatic formatting
- 4-space indentation (Rust default)
- Max line length: 100 characters
- No comments unless necessary (code should be self-documenting)

### Naming Conventions
- **Modules**: `snake_case` (e.g., `embedding_service`)
- **Structs/Enums**: `PascalCase` (e.g., `EmbeddingRequest`)
- **Functions/Variables**: `snake_case` (e.g., `calculate_similarity`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Type parameters**: `PascalCase` (e.g., `T: Clone`)

### Types
- Use `Arc<RwLock<T>>` or `Arc<Mutex<T>>` for shared mutable state
- Prefer `Option<T>` over null values
- Use `Result<T, AppError>` for fallible operations
- Use explicit types rather than `_` inference

### Error Handling
- All functions return `Result<T, AppError>`
- Define errors using `thiserror` in `error.rs`
- Use `anyhow` for error propagation in main.rs
- Use contextful error messages with `context()` or `with_context()`

### Documentation
- All `pub` APIs must have `///` doc comments
- Document parameters, return values, and panics
- Include usage examples in doc comments

### Async Programming
- Use Tokio runtime for async operations
- Avoid blocking calls in async contexts
- Use `.await` properly without suppressing warnings

### Module Visibility
- Public modules (used by main.rs or externally): `pub mod`
- Internal modules (crate-only): `pub(crate) mod`

Public modules: `audit`, `auth`, `config`, `domain`, `engine`, `grpc`, `metrics`, `rate_limit`, `routes`, `security`, `service`, `utils`, `error`

Internal modules: `cache`, `device`, `model`, `monitor`, `text`

### Security
- Never hardcode secrets—use environment variables
- Validate all user inputs
- Use parameterized queries for any database operations
- JWT secret must be at least 32 characters

## Architecture Patterns

### Engine Abstraction
- Implement `InferenceEngine` trait for new engines
- Add variants to `AnyEngine` enum
- Configure in `config/model.rs` with `EngineType`

### Service Layer
- Wrap logic in service structs (e.g., `EmbeddingService`)
- Accept `AppState` for shared access
- Use `Arc<RwLock<>>` for thread-safe state

### Middleware Chain
- Authentication: JWT token validation
- CSRF: Origin + token validation
- Rate limiting: global/IP/user/API key dimensions

## Feature Flags
- `cuda`: NVIDIA GPU acceleration
- `metal`: Apple Silicon GPU acceleration
- `onnx`: ONNX Runtime engine
- `grpc`: gRPC server support

## Default Ports
- HTTP API: 9002
- gRPC API: 50051
- Prometheus metrics: 9090

## Commit Style
Follow Conventional Commits:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Restructuring
- `perf`: Performance
- `test`: Testing
- `chore`: Build/dependencies

## Adding New Features

### HTTP Route
1. Define handler in `src/routes/`
2. Use `AppState` for shared state
3. Register in `src/routes/mod.rs`
4. Add OpenAPI docs with `utoipa`

### Configuration
1. Add field to `src/config/app.rs` or `src/config/model.rs`
2. Update `config.toml` example
3. Use in `main.rs`

## Debugging
```bash
# Enable verbose logging
export RUST_LOG=vecboost=debug

# Check port usage
lsof -i :9002  # HTTP
lsof -i :50051 # gRPC

# Model download progress
export RUST_LOG=hf_hub=info
```

## Agent Guidelines

### Communication Language
- Always respond to the user in **Chinese** (中文)

### Default to Action
- Implement changes rather than only suggesting them
- When user intent is unclear:
  1. Infer the most useful likely action based on context
  2. Use tools to discover missing details instead of guessing
  3. Read relevant files before making assumptions
  4. Proceed with implementation unless explicitly asked for suggestions only
  5. If truly ambiguous, briefly clarify intent before acting

### Parallel Tool Calls
- Maximize efficiency by calling independent tools simultaneously
- NEVER use parallel calls when later calls depend on earlier results
- NEVER use placeholders or guess missing parameters
- Prioritize speed through parallelization wherever safe

### Code Investigation
- **CRITICAL**: Never speculate about code you haven't opened
- Before answering ANY question about code:
  1. Read all relevant files first
  2. Understand the codebase's style, conventions, and abstractions
  3. Search thoroughly for key facts and dependencies
  4. Provide grounded, hallucination-free answers based on actual code
  5. If uncertain after investigation, explicitly state what you don't know

### Code Modification Strategy
- When editing existing code:
  1. Make surgical, minimal changes - don't rewrite entire files
  2. Preserve existing logic unless it's broken or explicitly needs changing
  3. Match the surrounding code style exactly
  4. Keep the same indentation, naming conventions, and patterns
  5. Maintain backward compatibility unless breaking changes are requested

### Error Handling & Testing
- For production code:
  - Add appropriate error handling for edge cases
  - Include input validation where needed
  - Consider failure modes and add defensive checks
  - If writing new functions, briefly verify they work as expected
- For debugging:
  - Read error messages and stack traces carefully
  - Investigate the actual error location before proposing fixes
  - Test your fix logic before applying

### Workspace Maintenance
- Maintain a clean codebase like a responsible visitor:
  - Remove temporary files, test scripts, or debug code after task completion
  - Clean up commented-out code blocks if they were part of your changes
  - Don't leave TODO comments unless explicitly requested
  - Restore any temporary modifications made during investigation
  - If you created scaffolding or helper files, remove them when done
  - **Exception**: Keep files if they're part of the deliverable or explicitly requested

### Security Awareness
- Be mindful of common security issues:
  - Don't hardcode sensitive data (API keys, passwords, tokens)
  - Validate and sanitize user inputs
  - Use parameterized queries for databases
  - Be cautious with eval() or exec() equivalents
  - Check file paths to prevent directory traversal
- However: Don't add heavy security measures unless the context suggests it's production-critical code

### Performance Consciousness
- Write reasonably efficient code:
  - Avoid obvious O(n²) solutions when O(n) is simple
  - Don't load entire large files into memory if streaming is easy
  - Cache expensive computations if they're repeated
  - Close resources (files, connections) properly
- But don't micro-optimize:
  - Premature optimization is wasteful
  - Readability > minor performance gains
  - Only optimize hot paths if there's a clear need

### Communication Practices
- Balance autonomy with clarity:
  - For straightforward requests: Just do it
  - For potentially destructive changes (deletions, major refactors): Briefly confirm intent
  - For ambiguous requests with multiple valid interpretations: Ask once, then proceed
  - After completing complex tasks: Provide a concise summary of what was changed
  - If you discover the request cannot be completed as stated: Explain why and suggest alternatives

### Dependency Management
- When working with dependencies:
  - Check existing package.json/requirements.txt/etc. before suggesting new dependencies
  - Use versions compatible with the existing stack
  - Import only what's needed - avoid wildcard imports
  - Place imports following the project's existing organization (stdlib, third-party, local)
  - Remove unused imports when you notice them
  - If adding a new dependency, mention it explicitly so the user knows to install it

### Documentation Balance
- Document thoughtfully but not excessively:
  - Add docstrings for public APIs and complex functions
  - Comment on non-obvious logic or workarounds
  - Don't comment on self-explanatory code
  - Keep comments concise and maintenance-friendly
  - Update existing comments if you change the related code
  - Use clear variable/function names to reduce need for comments
