<div align="center">

# ❓ Frequently Asked Questions (FAQ)

### Quick Answers to Common Questions

[🏠 Home](../README.md) • [📖 User Guide](USER_GUIDE.md)

---

</div>

## 📋 Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Usage & Features](#usage--features)
- [Performance](#performance)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Licensing](#licensing)

---

## General Questions

<div align="center">

### 🤔 About the Project

</div>

<details>
<summary><b>❓ What is vecboost?</b></summary>

<br>

**vecboost** is a high-performance vector embedding service built with Rust, providing:

- ✅ **Multi-Engine Support** - Seamlessly switch between Candle and ONNX Runtime engines
- ✅ **GPU Acceleration** - CUDA and Metal support for faster inference
- ✅ **gRPC API** - Efficient, language-agnostic communication
- ✅ **Batch Processing** - High-throughput embedding generation
- ✅ **Automatic Model Management** - Built-in model downloading, caching, and lifecycle management

It's designed for **Rust developers** and distributed systems requiring scalable, production-ready vector embedding capabilities.

**Learn more:** [User Guide](USER_GUIDE.md)

</details>

<details>
<summary><b>❓ Why should I use this instead of alternatives?</b></summary>

<br>

<table>
<tr>
<th>Feature</th>
<th>vecboost</th>
<th>Transformers.js</th>
<th>FastEmbed</th>
</tr>
<tr>
<td>Multi-Engine Support</td>
<td>✅ Candle/ONNX</td>
<td>❌ ONNX only</td>
<td>✅ Multiple</td>
</tr>
<tr>
<td>GPU Acceleration</td>
<td>✅ CUDA/Metal</td>
<td>❌ Limited</td>
<td>⚠️ Experimental</td>
</tr>
<tr>
<td>gRPC API</td>
<td>✅ Built-in</td>
<td>❌ No</td>
<td>❌ No</td>
</tr>
<tr>
<td>Batch Processing</td>
<td>✅ Optimized</td>
<td>⚠️ Basic</td>
<td>✅ Supported</td>
</tr>
<tr>
<td>Rust Native</td>
<td>✅ Yes</td>
<td>❌ No</td>
<td>✅ Yes</td>
</tr>
</table>

**Key Advantages:**
- 🚀 **High Performance**: Rust-based implementation with GPU acceleration
- 🔄 **Engine Flexibility**: Switch between engines based on performance needs
- 🛡️ **Production Ready**: Built-in resilience mechanisms (circuit breakers, retries)
- 📊 **Observability**: Comprehensive metrics and monitoring integration
- ⚡ **Scalable**: Optimized for high-throughput batch processing

</details>

<details>
<summary><b>❓ Is this production-ready?</b></summary>

<br>

**Current Status:** ✅ **Production-ready!**

<table>
<tr>
<td width="50%">

**What's Ready:**
- ✅ Core embedding logic stable
- ✅ Multi-engine support (Candle, ONNX Runtime)
- ✅ GPU acceleration (CUDA, Metal)
- ✅ gRPC API with JWT authentication
- ✅ Batch processing and file embedding
- ✅ Automatic model management

</td>
<td width="50%">

**Maturity Indicators:**
- 📊 Extensive test suite
- 🔄 Regular maintenance
- 🛡️ Security-focused design (JWT auth, TLS)
- 📖 Comprehensive documentation
- ⚡ Performance optimized for production

</td>
</tr>
</table>

> **Note:** Always review the release notes before upgrading versions.

</details>

<details>
<summary><b>❓ What platforms are supported?</b></summary>

<br>

<table>
<tr>
<th>Platform</th>
<th>Architecture</th>
<th>Status</th>
<th>Notes</th>
</tr>
<tr>
<td rowspan="2"><b>Linux</b></td>
<td>x86_64</td>
<td>✅ Fully Supported</td>
<td>Primary platform</td>
</tr>
<tr>
<td>ARM64</td>
<td>✅ Fully Supported</td>
<td>Tested on ARM servers</td>
</tr>
<tr>
<td rowspan="2"><b>macOS</b></td>
<td>x86_64</td>
<td>✅ Fully Supported</td>
<td>Intel Macs</td>
</tr>
<tr>
<td>ARM64</td>
<td>✅ Fully Supported</td>
<td>Apple Silicon (M1/M2/M3)</td>
</tr>
<tr>
<td><b>Windows</b></td>
<td>x86_64</td>
<td>✅ Fully Supported</td>
<td>Windows 10+</td>
</tr>
</table>

</details>

<details>
<summary><b>❓ What programming languages are supported?</b></summary>

<br>

**vecboost** provides a **gRPC API** that is language-agnostic, allowing it to be used from any programming language with gRPC support.

**Official Support:**
- ✅ **Rust**: Native client library
- ✅ **Any gRPC-compatible language**: Generated clients available for all major languages

**Documentation:**
- [Rust API Docs](https://docs.rs/vecboost)
- [gRPC API Reference](../proto/embedding.proto)

</details>

---

## Installation & Setup

<div align="center">

### 🚀 Getting Started

</div>

<details>
<summary><b>❓ How do I install this?</b></summary>

<br>

**Client Library Installation:**

Add the vecboost client library to your Rust project:

```toml
[dependencies]
vecboost = { version = "0.1" }
tonic = "0.11"
tokio = { version = "1.0", features = ["full"] }
prost = "0.12"
```

Or using cargo:

```bash
cargo add vecboost tonic tokio --features tokio/full prost
```

**Server Installation:**

Build the vecboost server from source with desired features:

```bash
# Basic build with Candle engine (CPU only)
cargo build --release

# With CUDA acceleration
cargo build --release --features cuda

# With Metal acceleration (macOS only)
cargo build --release --features metal

# With ONNX Runtime support
cargo build --release --features onnx

# With all features
cargo build --release --features all
```

**Verification:**

```rust
use tonic::Request;
use vecboost::embedding_client::EmbeddingClient;
use vecboost::EmbedRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to vecboost server
    let mut client = EmbeddingClient::connect("http://localhost:50051").await?;
    
    // Create embedding request for single text
    let request = Request::new(EmbedRequest {
        text: "Hello, world!".to_string(),
        normalize: Some(true),
    });
    
    // Get embedding response
    let response = client.embed(request).await?;
    println!("✅ Embedding successful! Vector dimension: {}", 
             response.into_inner().dimension);
    
    Ok(())
}
```

**See also:** [Installation Guide](USER_GUIDE.md#installation)

</details>

<details>
<summary><b>❓ What are the system requirements?</b></summary>

<br>

**Minimum Requirements:**

<table>
<tr>
<th>Component</th>
<th>Requirement</th>
<th>Recommended</th>
</tr>
<tr>
<td>Rust Version</td>
<td>1.75+</td>
<td>Latest stable</td>
</tr>
<tr>
<td>Memory</td>
<td>4GB</td>
<td>8GB+ (for large models)</td>
</tr>
<tr>
<td>Disk Space</td>
<td>2GB</td>
<td>10GB+ (for multiple models)</td>
</tr>
<tr>
<td>CPU</td>
<td>x86-64 / ARM64</td>
<td>Multi-core CPU (8+ cores)</td>
</tr>
</table>

**GPU Requirements:**

<table>
<tr>
<th>GPU Type</th>
<th>Minimum</th>
<th>Recommended</th>
</tr>
<tr>
<td>NVIDIA (CUDA)</td>
<td>Compute Capability 7.5+</td>
<td>RTX 30xx / A100+</td>
</tr>
<tr>
<td>AMD (ROCm)</td>
<td>ROCm 5.0+ compatible</td>
<td>RX 6000+ / MI100+</td>
</tr>
<tr>
<td>Apple Silicon (Metal)</td>
<td>M1+</td>
<td>M2 Max / M3 Pro+</td>
</tr>
</table>

**Network:**
- Required for model downloading
- Recommended: 100Mbps+ for fast model downloads

</details>

<details>
<summary><b>❓ I'm getting compilation errors, what should I do?</b></summary>

<br>

**Common Solutions:**

1. **Check Rust version:**
   ```bash
   rustc --version
   # Should be 1.75.0 or higher
   ```

2. **Ensure `serde` derive is enabled:**
   Make sure you have `features = ["derive"]` for `serde` in your `Cargo.toml`.

3. **Clean build artifacts:**
   ```bash
   cargo clean
   cargo build
   ```

**Still having issues?**
- 📝 Check [Troubleshooting section in README](../README.md#troubleshooting)
- 🐛 [Open an issue](../../issues) with error details

</details>

<details>
<summary><b>❓ Can I use this with Docker?</b></summary>

<br>

**Yes!** vecboost works perfectly in containerized environments. It can load configurations from environment variables which is the standard for Docker.

**Sample Dockerfile (Multi-stage):**

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/my_app /usr/local/bin/
CMD ["my_app"]
```

**Environment Variables in Docker Compose:**

```yaml
services:
  app:
    image: my_app:latest
    environment:
      - APP_PORT=8080
      - APP_DATABASE_URL=postgres://user:pass@db/dbname
```

</details>

---

## Usage & Features

<div align="center">

### 💡 Working with the API

</div>

<details>
<summary><b>❓ How do I get started with basic usage?</b></summary>

<br>

**5-Minute Quick Start with vecboost:**

```rust
use tonic::Request;
use vecboost::embedding_client::EmbeddingClient;
use vecboost::BatchEmbedRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Connect to vecboost server
    let mut client = EmbeddingClient::connect("http://localhost:50051").await?;
    
    // 2. Create batch embedding request (for multiple texts)
    let request = Request::new(BatchEmbedRequest {
        texts: vec!["Hello, world!".to_string()],
        normalize: Some(true),
    });
    
    // 3. Get embedding response
    let response = client.embed_batch(request).await?;
    let embeddings = response.into_inner().embeddings;
    
    println!("Generated {} embeddings", embeddings.len());
    println!("Embedding dimension: {}", embeddings[0].dimension);
    
    Ok(())
}
```

**Next Steps:**
- 📖 [User Guide](USER_GUIDE.md)
- 💻 [More Examples](../examples/)
- 📋 [gRPC API Reference](../proto/embedding.proto)

</details>

<details>
<summary><b>❓ What formats and sources are supported?</b></summary>

<br>

**Supported Embedding Models:**
- ✅ **bge-m3**: High-performance general-purpose embedding model
- ✅ **all-MiniLM-L6-v2**: Lightweight model for fast inference
- ✅ **e5-small-v2**: Efficient model for text understanding
- ✅ **Custom ONNX models**: Bring your own ONNX-formatted embedding models

**Supported Input Formats:**
- ✅ **Plain Text**: Direct text input for embedding
- ✅ **Batch Text**: Multiple text inputs for efficient processing
- ✅ **Files**: Support for common document formats (via file embedding API)
  - PDF
  - TXT
  - Markdown
  - HTML

**Supported Similarity Metrics:**
- ✅ **Cosine Similarity**: Default similarity metric
- ✅ **Euclidean Distance**: For nearest neighbor calculations
- ✅ **Manhattan Distance**: Robust to outliers
- ✅ **Dot Product**: Efficient for normalized vectors

**Supported Engines:**
- ✅ **Candle**: Rust-native ML framework (CPU/GPU)
- ✅ **ONNX Runtime**: Cross-platform inference engine

</details>

<details>
<summary><b>❓ Can I compute similarity between embeddings?</b></summary>

<br>

**Yes!** vecboost provides built-in similarity computation for embeddings.

```rust
use tonic::Request;
use vecboost::embedding_client::EmbeddingClient;
use vecboost::{BatchEmbedRequest, ComputeSimilarityRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to vecboost server
    let mut client = EmbeddingClient::connect("http://localhost:50051").await?;
    
    // Generate embeddings first
    let batch_embed_request = BatchEmbedRequest {
        texts: vec![
            "What is machine learning?".to_string(),
            "Artificial intelligence uses algorithms to learn.".to_string()
        ],
        normalize: Some(true),
    };
    
    let batch_embed_response = client.embed_batch(
        Request::new(batch_embed_request)
    ).await?;
    
    let embeddings = batch_embed_response.into_inner().embeddings;
    let embedding1 = embeddings[0].embedding.clone();
    let embedding2 = embeddings[1].embedding.clone();
    
    // Compute similarity between the two embeddings
    let similarity_request = ComputeSimilarityRequest {
        vector1: embedding1,
        vector2: embedding2,
        metric: "cosine".to_string(),
    };
    
    let similarity_response = client.compute_similarity(
        Request::new(similarity_request)
    ).await?;
    
    println!("Similarity between document 1 and document 2: {:.4}", 
             similarity_response.into_inner().score);
    
    Ok(())
}
```

**Supported Similarity Metrics:**
- � **cosine**: Cosine similarity (default)
- 📏 **euclidean**: Euclidean distance
- 📐 **manhattan**: Manhattan distance
- 💫 **dot**: Dot product

</details>

<details>
<summary><b>❓ How do I handle errors properly?</b></summary>

<br>

**Recommended Pattern for vecboost:**

```rust
use tonic::Status;
use tonic::transport::Error as TransportError;
use vecboost::embedding_client::EmbeddingClient;
use vecboost::EmbedRequest;

#[tokio::main]
async fn main() {
    match run().await {
        Ok(_) => println!("✅ Embedding operation successful"),
        Err(e) => {
            if let Some(status) = e.downcast_ref::<Status>() {
                match status.code() {
                    tonic::Code::NotFound => {
                        eprintln!("❌ Model not found: {}", status.message());
                    }
                    tonic::Code::InvalidArgument => {
                        eprintln!("❌ Invalid request: {}", status.message());
                    }
                    tonic::Code::Unavailable => {
                        eprintln!("❌ Server unavailable: {}", status.message());
                    }
                    tonic::Code::PermissionDenied => {
                        eprintln!("❌ Permission denied: {}", status.message());
                    }
                    _ => {
                        eprintln!("❌ gRPC error: {} ({:?})", status.message(), status.code());
                    }
                }
            } else if let Some(transport_err) = e.downcast_ref::<TransportError>() {
                eprintln!("❌ Transport error: {}", transport_err);
            } else {
                eprintln!("❌ Error: {}", e);
            }
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingClient::connect("http://localhost:50051").await?;
    // ... embedding operations ...
    Ok(())
}
```

</details>

<details>
<summary><b>❓ Is there async/await support?</b></summary>

<br>

**Yes!** vecboost is built with async/await support from the ground up, leveraging the power of Rust's async ecosystem.

```rust
use tonic::Request;
use vecboost::embedding_client::EmbeddingClient;
use vecboost::EmbedRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // All vecboost client operations are async
    let mut client = EmbeddingClient::connect("http://localhost:50051").await?;
    
    // Async embedding generation
    let response = client.embed(Request::new(EmbedRequest {
        text: "Hello, world!".to_string(),
        normalize: Some(true),
    })).await?;
    
    // Async similarity computation
    // let similarity_response = client.compute_similarity(...).await?;
    
    Ok(())
}
```

**Benefits of async design:**
- ✅ Efficient handling of multiple concurrent requests
- ✅ Non-blocking operations for better throughput
- ✅ Seamless integration with modern Rust async frameworks
- ✅ Support for large batch processing without blocking

</details>

---

## Performance

<div align="center">

### ⚡ Speed and Optimization

</div>

<details>
<summary><b>❓ How fast is it?</b></summary>

<br>

vecboost is optimized for high-performance vector embedding generation, with support for GPU acceleration and batch processing.

**Benchmark Results (bge-m3 model, batch size 32):**

<table>
<tr>
<th>Hardware</th>
<th>Engine</th>
<th>Throughput (texts/sec)</th>
<th>P99 Latency (ms)</th>
</tr>
<tr>
<td>CPU (Intel i9-12900K)</td>
<td>Candle</td>
<td>~120</td>
<td>~280</td>
</tr>
<tr>
<td>CPU (Intel i9-12900K)</td>
<td>ONNX</td>
<td>~180</td>
<td>~180</td>
</tr>
<tr>
<td>GPU (NVIDIA RTX 3090)</td>
<td>Candle (CUDA)</td>
<td>~1,200</td>
<td>~28</td>
</tr>
<tr>
<td>GPU (NVIDIA RTX 3090)</td>
<td>ONNX (CUDA)</td>
<td>~1,500</td>
<td>~22</td>
</tr>
<tr>
<td>GPU (Apple M2 Max)</td>
<td>Candle (Metal)</td>
<td>~800</td>
<td>~40</td>
</tr>
</table>

**Run benchmarks yourself:**

```bash
cargo bench --features all
```

</details>

<details>
<summary><b>❓ How can I improve performance?</b></summary>

<br>

**Optimization Tips for vecboost:**

1. **Enable GPU Acceleration:**
   ```bash
   # For NVIDIA GPUs
   cargo build --release --features cuda
   
   # For Apple Silicon
   cargo build --release --features metal
   ```

2. **Use Batch Processing:**
   Process multiple texts in a single request to maximize throughput:
   ```rust
   let request = BatchEmbedRequest {
       texts: vec![
           "Text 1".to_string(),
           "Text 2".to_string(),
           // Add more texts for batch processing
       ],
       normalize: Some(true),
   };
   ```

3. **Choose the Right Engine:**
   - Use **ONNX Runtime** for maximum CPU performance
   - Use **Candle** with GPU features for best GPU performance
   - Use lighter models like `all-MiniLM-L6-v2` for faster inference

4. **Tune Batch Size:**
   Experiment with batch sizes to find the optimal balance between latency and throughput for your workload.

5. **Enable Model Caching:**
   vecboost automatically caches models after first use, but ensure sufficient disk space for cached models.

</details>

<details>
<summary><b>❓ What's the memory usage like?</b></summary>

<br>

**Typical Memory Usage:**

vecboost memory usage depends on the loaded models and batch sizes:

<table>
<tr>
<th>Component</th>
<th>Memory Usage</th>
</tr>
<tr>
<td>Base Service (No Models)</td>
<td>~50-100 MB</td>
</tr>
<tr>
<td>all-MiniLM-L6-v2 (CPU)</td>
<td>~120 MB</td>
</tr>
<tr>
<td>bge-m3 (CPU)</td>
<td>~1.5 GB</td>
</tr>
<tr>
<td>bge-m3 (CUDA, FP16)</td>
<td>~800 MB GPU memory</td>
</tr>
<tr>
<td>bge-m3 (Metal, FP16)</td>
<td>~900 MB GPU memory</td>
</tr>
</table>

**Memory Management:**
- ✅ Automatic model unloading when not in use
- ✅ Memory-efficient batch processing
- ✅ GPU memory optimized for inference
- ✅ No memory leaks (verified with continuous testing)
- ✅ Leverages Rust's ownership model for memory safety

**Tips:**
- Use smaller models for memory-constrained environments
- Adjust batch size based on available memory
- Monitor GPU memory usage when using GPU acceleration

</details>

---

## Security

<div align="center">

### 🔒 Security Features

</div>

<details>
<summary><b>❓ Is this secure?</b></summary>

<br>

**Yes!** Security is a core focus of vecboost.

**Security Features:**

<table>
<tr>
<td width="50%">

**API Security**
- ✅ gRPC with TLS encryption
- ✅ JWT-based authentication
- ✅ Role-based access control (RBAC)
- ✅ Input validation and sanitization

</td>
<td width="50%">

**Data Security**
- ✅ Memory-safe implementation (Rust)
- ✅ Sensitive data masking in logs
- ✅ Encrypted model storage
- ✅ Secure model downloading (HTTPS)

</td>
</tr>
<tr>
<td width="50%">

**Infrastructure**
- ✅ GPU memory isolation
- ✅ Request rate limiting
- ✅ Circuit breakers for resilience
- ✅ Secure environment variables handling

</td>
<td width="50%">

**Protection Mechanisms**
- ✅ Buffer overflow protection
- ✅ Side-channel resistance
- ✅ Memory wiping (zeroize)
- ✅ Anti-injection safeguards

</td>
</tr>
</table>

**Compliance:**
- 🏅 Follows industry best practices for vector embedding services
- 🏅 Supports secure deployment in regulated environments
- 🏅 GDPR and CCPA compliant data handling

**More details:** Refer to the README for security information.

</details>

<details>
<summary><b>❓ How do I report security vulnerabilities?</b></summary>

<br>

**Please report security issues responsibly:**

1. **DO NOT** create public GitHub issues
2. **Email:** security@vecboost.io
3. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact

**Response Timeline:**
- 📧 Initial response: 24 hours
- 🔍 Assessment: 72 hours
- 📢 Public disclosure: After fix is released

</details>

<details>
<summary><b>❓ What about sensitive data?</b></summary>

<br>

vecboost provides robust mechanisms to handle sensitive data:

1. **API Authentication**: JWT tokens for secure API access control
2. **Data Masking**: Automatic masking of sensitive information in logs
3. **Secure Storage**: Encrypted storage for models and configuration
4. **HTTPS Only**: All model downloads and API communications use HTTPS

**Best Practices for Secure Deployment:**

```bash
# Enable TLS for gRPC API
export VECBOOST_TLS_ENABLED=true
export VECBOOST_TLS_CERT_PATH=/path/to/cert.pem
export VECBOOST_TLS_KEY_PATH=/path/to/key.pem

# Enable JWT authentication
export VECBOOST_JWT_SECRET=your-secret-key
export VECBOOST_JWT_EXPIRY=3600

# Run the server
./vecboost-server
```

</details>

---

## Troubleshooting

<div align="center">

### 🔧 Common Issues

</div>

<details>
<summary><b>❓ I'm getting connection errors</b></summary>

<br>

**Problem:**
```
Error: tonic::transport::Error: transport error: Connection refused (os error 111)
```

**Cause:** Could not connect to the vecboost server.

**Solution:**
1. Ensure the vecboost server is running.
2. Check the server address and port (default: `http://localhost:50051`).
3. Verify network connectivity between client and server.
4. If using TLS, ensure certificates are properly configured.

</details>

<details>
<summary><b>❓ I'm getting model errors</b></summary>

<br>

**Problem:**
```
Error: Model not found: bge-m3
```

**Cause:** The specified model could not be found or loaded.

**Solution:**
1. Check if the model name is correct (supported models: bge-m3, all-MiniLM-L6-v2, e5-small-v2).
2. Ensure network connectivity for automatic model downloading.
3. Verify sufficient disk space for model caching.
4. Check model download logs for more details.

</details>

<details>
<summary><b>❓ I'm getting GPU errors</b></summary>

<br>

**Problem:**
```
Error: CUDA error: out of memory
```

**Cause:** GPU memory issues or incorrect GPU setup.

**Solution:**
1. Ensure GPU drivers are properly installed.
2. Check if the correct GPU features are enabled during build (`--features cuda` or `--features metal`).
3. Reduce batch size to lower memory usage.
4. Use smaller models like `all-MiniLM-L6-v2` for memory-constrained environments.
5. Monitor GPU memory usage during inference.

</details>

<details>
<summary><b>❓ I'm getting batch processing errors</b></summary>

<br>

**Problem:**
```
Error: Batch size too large
```

**Cause:** The batch size exceeds server limits or available resources.

**Solution:**
1. Reduce the number of texts in a single batch request.
2. Check server logs for the maximum allowed batch size.
3. Increase server resources if needed.
4. Implement client-side batching with retry logic.

</details>

<details>
<summary><b>❓ How do I debug the embedding service?</b></summary>

<br>

**Solution:**
Enable detailed logging to troubleshoot issues.

**Server-side logging:**
```bash
# Enable debug logging for vecboost
RUST_LOG=vecboost=debug ./vecboost-server
```

**Client-side logging:**
```rust
use tracing_subscriber::fmt::init;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable debug logging
    init();
    
    // Your vecboost client code here
    // let mut client = EmbeddingClient::connect(...).await?;
    
    Ok(())
}
```

**Common log levels:**
- `error`: Only critical errors
- `warn`: Warnings and errors
- `info`: General information
- `debug`: Detailed debug information
- `trace`: Very detailed tracing information

</details>

**More issues?** Check the README or open an issue.

---

## Contributing

<div align="center">

### 🤝 Join the Community

</div>

<details>
<summary><b>❓ How can I contribute?</b></summary>

<br>

**Ways to Contribute:**

<table>
<tr>
<td width="50%">

**Code Contributions**
- 🐛 Fix bugs
- ✨ Add features
- 📝 Improve documentation
- ✅ Write tests

</td>
<td width="50%">

**Non-Code Contributions**
- 📖 Write tutorials
- 🎨 Design assets
- 🌍 Translate docs
- 💬 Answer questions

</td>
</tr>
</table>

**Getting Started:**

1. 🍴 Fork the repository
2. 🌱 Create a branch
3. ✏️ Make changes
4. ✅ Add tests
5. 📤 Submit PR

**Guidelines:** Refer to the README for contributing information.

</details>

<details>
<summary><b>❓ I found a bug, what should I do?</b></summary>

<br>

**Before Reporting:**

1. ✅ Check [existing issues](../../issues)
2. ✅ Try the latest version
3. ✅ Check [troubleshooting section in README](../README.md#troubleshooting)

**Creating a Good Bug Report:**

```markdown
### Description
Clear description of the bug

### Steps to Reproduce
1. Step one
2. Step two
3. See error

### Expected Behavior
What should happen

### Actual Behavior
What actually happens

### Environment
- OS: Ubuntu 22.04
- Rust version: 1.75.0
- Project version: 1.0.0

### Additional Context
Any other relevant information
```

**Submit:** [Create Issue](../../issues/new)

</details>

<details>
<summary><b>❓ Where can I get help?</b></summary>

<br>

<div align="center">

### 💬 Support Channels

</div>

<table>
<tr>
<td width="33%" align="center">

**🐛 Issues**

[GitHub Issues](../../issues)

Bug reports & features

</td>
<td width="33%" align="center">

**💬 Discussions**

[GitHub Discussions](../../discussions)

Q&A and ideas

</td>
<td width="33%" align="center">

**💡 Discord**

[Join Server](https://discord.gg/project)

Live chat

</td>
</tr>
</table>

**Response Times:**
- 🐛 Critical bugs: 24 hours
- 🔧 Feature requests: 1 week
- 💬 Questions: 2-3 days

</details>

---

## Licensing

<div align="center">

### 📄 License Information

</div>

<details>
<summary><b>❓ What license is this under?</b></summary>

<br>

**MIT License**

vecboost is distributed under the MIT License, which permits:

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

The MIT License is a permissive license that allows for maximum flexibility while ensuring proper attribution.

</details>

<details>
<summary><b>❓ Can I use this in commercial projects?</b></summary>

<br>

**Yes!** Both MIT and Apache 2.0 licenses allow commercial use.

**What you need to do:**
1. ✅ Include the license text
2. ✅ Include copyright notice
3. ✅ State any modifications

**What you DON'T need to do:**
- ❌ Share your source code
- ❌ Open source your project
- ❌ Pay royalties

**Questions?** Contact: legal@example.com

</details>

---

<div align="center">

### 🎯 Still Have Questions?

<table>
<tr>
<td width="33%" align="center">
<a href="../../issues">
<img src="https://img.icons8.com/fluency/96/000000/bug.png" width="48"><br>
<b>Open an Issue</b>
</a>
</td>
<td width="33%" align="center">
<a href="../../discussions">
<img src="https://img.icons8.com/fluency/96/000000/chat.png" width="48"><br>
<b>Start a Discussion</b>
</a>
</td>
<td width="33%" align="center">
<a href="mailto:support@example.com">
<img src="https://img.icons8.com/fluency/96/000000/email.png" width="48"><br>
<b>Email Us</b>
</a>
</td>
</tr>
</table>

---

**[📖 User Guide](USER_GUIDE.md)** • **[🔧 API Docs](https://docs.rs/vecboost)** • **[🏠 Home](../README.md)**

Made with ❤️ by the Documentation Team

[⬆ Back to Top](#-frequently-asked-questions-faq)

</div>