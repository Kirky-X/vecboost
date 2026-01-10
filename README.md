# VecBoost

<p align="left">
    <img src="https://img.shields.io/badge/Rust-2024-edded?logo=rust&style=flat-square" alt="Rust Edition">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="MIT License">
    <img src="https://img.shields.io/badge/Version-0.1.0-green.svg?style=flat-square" alt="Version">
</p>

A high-performance, production-ready embedding vector service written in Rust. VecBoost provides efficient text vectorization with support for multiple inference engines, GPU acceleration, and enterprise-grade features.

## ✨ Features

- **🚀 High Performance**: Optimized Rust codebase with batch processing and concurrent request handling
- **🔧 Multiple Engines**: Support for Candle (native Rust) and ONNX Runtime inference engines
- **🎮 GPU Acceleration**: Native CUDA support (NVIDIA) and Metal support (Apple Silicon)
- **📊 Smart Caching**: Multi-tier caching with LRU, LFU, and KV cache strategies
- **🔐 Enterprise Security**: JWT authentication, CSRF protection, and audit logging
- **⚡ Rate Limiting**: Configurable rate limiting with token bucket algorithm
- **📈 Priority Queue**: Request prioritization with configurable priority weights
- **🌐 Dual APIs**: gRPC and HTTP/REST interfaces with OpenAPI documentation
- **📦 Kubernetes Ready**: Production deployment configurations included

## 🚀 Quick Start

### Prerequisites

- Rust 1.75+ (edition 2024)
- CUDA Toolkit 12.x (for GPU support on Linux)
- Metal (for GPU support on macOS)

### Installation

```bash
# Clone the repository
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost

# Build with default features (CPU only)
cargo build --release

# Build with CUDA support (Linux)
cargo build --release --features cuda

# Build with Metal support (macOS)
cargo build --release --features metal

# Build with all features
cargo build --release --features cuda,onnx,grpc,auth,redis
```

### Configuration

Copy the example configuration and customize:

```bash
cp config.toml config_custom.toml
# Edit config_custom.toml with your settings
```

### Running

```bash
# Run with default configuration
./target/release/vecboost

# Run with custom configuration
./target/release/vecboost --config config_custom.toml
```

The service will start on `http://localhost:9002` by default.

### Docker

```bash
# Build the image
docker build -t vecboost:latest .

# Run the container
docker run -p 9002:9002 -v $(pwd)/config.toml:/app/config.toml vecboost:latest
```

## 📖 Documentation

- [📋 User Guide](USER_GUIDE.md) - Detailed usage instructions
- [🔌 API Reference](API_REFERENCE.md) - REST API and gRPC documentation
- [🏗️ Architecture](ARCHITECTURE.md) - System design and components
- [🤝 Contributing](docs/CONTRIBUTING.md) - Contribution guidelines

## 🔌 API Usage

### HTTP REST API

Generate embeddings via HTTP:

```bash
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

Response:

```json
{
  "embedding": [0.123, 0.456, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

### gRPC API

The service also exposes a gRPC interface on port 50051 (configurable):

```protobuf
service EmbeddingService {
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc EmbedBatch(BatchEmbedRequest) returns (BatchEmbedResponse);
  rpc ComputeSimilarity(SimilarityRequest) returns (SimilarityResponse);
}
```

### OpenAPI Documentation

Access the interactive API documentation at:
- Swagger UI: `http://localhost:9002/swagger-ui/`
- ReDoc: `http://localhost:9002/redoc/`

## ⚙️ Configuration

### Key Configuration Options

```toml
[server]
host = "0.0.0.0"
port = 9002

[model]
model_repo = "BAAI/bge-m3"  # HuggingFace model ID
use_gpu = true
batch_size = 32
expected_dimension = 1024

[embedding]
cache_enabled = true
cache_size = 1024

[auth]
enabled = true
jwt_secret = "your-secret-key"
```

See [Configuration Guide](config.toml) for all options.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VecBoost Service                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   HTTP/gRPC │  │  Auth Layer │  │  Rate Limiting      │  │
│  │   Endpoints │  │  (JWT/CSRF) │  │  (Token Bucket)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                   │               │
│         └────────────────┴───────────────────┘               │
│                          │                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Request Pipeline                        │    │
│  │  ┌─────────┐  ┌───────────┐  ┌─────────────────┐   │    │
│  │  │ Priority│  │ Request   │  │ Response        │   │    │
│  │  │ Queue   │→ │ Workers   │→ │ Channel         │   │    │
│  │  └─────────┘  └───────────┘  └─────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Embedding Service                       │    │
│  │  ┌─────────┐  ┌───────────┐  ┌─────────────────┐   │    │
│  │  │ Text    │  │ Inference │  │ Vector Cache    │   │    │
│  │  │ Chunking│→ │ Engine    │→ │ (LRU/LFU/KV)    │   │    │
│  │  └─────────┘  └───────────┘  └─────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Inference Engine                        │    │
│  │    ┌─────────────┐  ┌─────────────┐                 │    │
│  │    │   Candle    │  │    ONNX     │                 │    │
│  │    │  (Native)   │  │  Runtime    │                 │    │
│  │    └─────────────┘  └─────────────┘                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│         ┌────────────────┼────────────────┐                 │
│         ▼                ▼                ▼                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   CPU    │    │   CUDA   │    │  Metal   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
vecboost/
├── src/
│   ├── audit/          # Audit logging
│   ├── auth/           # Authentication (JWT, CSRF)
│   ├── cache/          # Multi-tier caching (LRU, LFU, KV)
│   ├── config/         # Configuration management
│   ├── device/         # Device management (CPU, CUDA, Metal)
│   ├── engine/         # Inference engines (Candle, ONNX)
│   ├── grpc/           # gRPC server
│   ├── metrics/        # Prometheus metrics
│   ├── model/          # Model downloading and management
│   ├── pipeline/       # Request pipeline and prioritization
│   ├── rate_limit/     # Rate limiting
│   ├── routes/         # HTTP routes
│   ├── security/       # Security utilities
│   ├── service/        # Core embedding service
│   └── text/           # Text processing and tokenization
├── examples/gpu/       # GPU example programs
├── proto/              # gRPC protocol definitions
├── deployments/        # Kubernetes deployment configs
├── tests/              # Integration tests
└── config.toml         # Default configuration
```

## 🎯 Performance

| Metric | Value |
|--------|-------|
| Embedding Dimension | Up to 4096 |
| Batch Size | Up to 256 |
| Requests/Second | 1000+ (CPU) |
| Latency (p99) | < 50ms (GPU) |
| Cache Hit Ratio | > 90% (with 1024 entries) |

## 🔒 Security

- **Authentication**: JWT tokens with configurable expiration
- **Authorization**: Role-based access control
- **Audit Logging**: All requests logged with user and action details
- **Rate Limiting**: Per-IP, per-user, and global rate limits
- **Encryption**: AES-256-GCM for sensitive data at rest

## 📈 Monitoring

- **Prometheus Metrics**: `/metrics` endpoint for Prometheus scraping
- **Health Checks**: `/health` endpoint for liveness/readiness
- **OpenAPI Docs**: Swagger UI at `/swagger-ui/`
- **Grafana Dashboards**: Pre-configured dashboards in `deployments/`

## 🚀 Deployment

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f deployments/kubernetes/
```

See [Deployment Guide](deployments/kubernetes/README.md) for detailed instructions.

### Docker Compose

```yaml
services:
  vecboost:
    image: vecboost:latest
    ports:
      - "9002:9002"
    volumes:
      - ./config.toml:/app/config.toml
    environment:
      - MODEL_REPO=BAAI/bge-m3
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Candle](https://github.com/huggingface/candle) - Native Rust ML framework
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform ML inference
- [Hugging Face Hub](https://huggingface.co/models) - Model repository
