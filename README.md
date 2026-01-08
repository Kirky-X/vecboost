<div align="center">

# 🚀 VecBoost

<p>
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build">
</p>

<p align="center">
  <strong>A high-performance Rust vector embedding service optimized for production</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#api-reference">API</a>
</p>

</div>

---

## ✨ Features

| Core Features | Advanced Features |
|--------------|-------------------|
| ✅ **High Performance** - Rust + Tokio for maximum throughput | 🚀 **Batching** - Efficient batch processing |
| ✅ **Multi-Engine** - Candle (default) and ONNX Runtime | 🔐 **Authentication** - JWT, CSRF, API Key |
| ✅ **GPU Acceleration** - CUDA, Metal, ROCm support | 📊 **Monitoring** - Prometheus metrics |
| ✅ **Dual Protocol** - HTTP REST and gRPC APIs | 📦 **Rate Limiting** - Multi-dimensional throttling |
| ✅ **Auto-Scaling** - Priority request queue with pipeline | 🔍 **Audit Logging** - Complete operation tracking |
| ✅ **Caching** - Multi-level cache (ARC, LFU, LRU) | 🛡️ **Security** - Argon2, AES-GCM encryption |

---

## 🚀 Quick Start

### Installation

```bash
# CPU-only release build
cargo build --release

# With GPU support
cargo build --release --features cuda      # NVIDIA CUDA
cargo build --release --features metal     # Apple Silicon
cargo build --release --features onnx      # ONNX Runtime
cargo build --release --features grpc      # gRPC server

# All features
cargo build --release --features cuda,metal,onnx,grpc
```

### Running

```bash
# Default configuration
cargo run --release

# Custom configuration
cargo run --release -- --config config.toml
```

### Docker

```bash
docker build -t vecboost .
docker run -p 9002:9002 -p 50051:50051 -p 9090:9090 vecboost
```

---

## 📚 Documentation

- [📖 User Guide](docs/USER_GUIDE.md)
- [🏗️ Architecture](docs/ARCHITECTURE.md)
- [📘 API Reference](docs/API_REFERENCE.md)
- [🤝 Contributing Guide](docs/CONTRIBUTING.md)

---

## 🏗️ Architecture

```mermaid
graph TB
    Client --> HTTP[HTTP API :9002]
    Client --> gRPC[gRPC API :50051]
    
    HTTP --> Auth[Auth Middleware]
    gRPC --> Auth
    
    Auth --> RateLimit[Rate Limiting]
    RateLimit --> Router[Request Router]
    
    Router --> Embedding[/embed]
    Router --> Similarity[/similarity]
    Router --> Search[/search]
    Router --> Health[/health]
    
    Embedding --> Service[Embedding Service]
    Similarity --> Service
    Search --> Service
    
    Service --> Engine[Inference Engine]
    Engine --> Candle[Candle Engine]
    Engine --> ONNX[ONNX Engine]
    
    Engine --> Device[Device Manager]
    Device --> GPU[GPU/CUDA]
    Device --> CPU[CPU]
    
    Service --> Cache[KV Cache]
    Service --> Pipeline[Priority Pipeline]
    
    Pipeline --> Queue[Request Queue]
    Queue --> Scheduler[Batch Scheduler]
    
    Service --> Metrics[Metrics Collector]
    Metrics --> Prometheus[:9090]
```

---

## ⚙️ Configuration

### Default Ports

| Service | Port |
|---------|------|
| HTTP API | 9002 |
| gRPC API | 50051 |
| Prometheus | 9090 |

### Example config.toml

```toml
[server]
host = "0.0.0.0"
port = 9002

[model]
model_repo = "BAAI/bge-m3"
use_gpu = false
batch_size = 32

[auth]
enabled = true
jwt_secret = "your-secret-key-min-32-chars"

[rate_limit]
enabled = true
global_requests_per_minute = 1000
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Throughput | 10,000+ req/sec |
| P95 Latency | < 50ms |
| P99 Latency | < 100ms |
| Error Rate | < 0.1% |

---

## 📁 Project Structure

```
vecboost/
├── src/
│   ├── audit/          # Audit logging
│   ├── auth/           # Authentication (JWT, CSRF, User Store)
│   ├── cache/          # Multi-level caching
│   ├── config/         # Configuration management
│   ├── device/         # GPU/CPU device management
│   ├── domain/         # Domain types
│   ├── engine/         # Inference engines (Candle, ONNX)
│   ├── grpc/           # gRPC server
│   ├── metrics/        # Prometheus metrics
│   ├── pipeline/       # Request queue & scheduling
│   ├── rate_limit/     # Rate limiting
│   ├── routes/         # HTTP handlers
│   ├── security/       # Encryption, key store
│   ├── service/        # Business logic
│   └── text/           # Text processing
├── tests/              # Integration & performance tests
├── examples/           # Example code
├── deployments/        # Docker, Kubernetes configs
└── docs/               # Documentation
```

---

## 🧪 Testing

```bash
# All tests with all features
cargo test --all-features

# Unit tests
cargo test --lib

# Integration tests
cargo test --tests

# Performance benchmarks
cargo test --features cuda,grpc --test performance_test
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by the VecBoost Team**

</div>
