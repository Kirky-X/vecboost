<div align="center">

# 📘 API Reference

### Complete API Documentation

[🏠 Home](../README.md) • [📖 User Guide](USER_GUIDE.md) • [🏗️ Architecture](ARCHITECTURE.md)

______________________________________________________________________

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Core API](#core-api)
  - [EmbeddingService (gRPC)](#embeddingservice-grpc)
  - [InferenceEngine](#inferenceengine)
  - [Engine Implementations](#engine-implementations)
    - [CandleEngine](#candleengine)
    - [OnnxEngine](#onnxengine)
- [Error Handling](#error-handling)
- [Type Definitions](#type-definitions)
- [Examples](#examples)

______________________________________________________________________

## Overview

<div align="center">

### 🎯 API Design Principles

</div>

<table>
<tr>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/easy.png" width="64"><br>
<b>Simple</b><br>
Intuitive and easy to use gRPC API
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/security-checked.png" width="64"><br>
<b>High Performance</b><br>
GPU-accelerated embedding generation
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/module.png" width="64"><br>
<b>Flexible</b><br>
Multi-engine support (Candle/ONNX)
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/documentation.png" width="64"><br>
<b>Well-documented</b><br>
Comprehensive API reference
</td>
</tr>
</table>

VecBoost provides a high-performance gRPC API for text embedding generation with support for multiple inference engines, GPU acceleration, and batch processing capabilities.

______________________________________________________________________

## Core API

### EmbeddingService (gRPC)

`EmbeddingService` is the main gRPC service for vecboost, providing methods for text embedding, batch processing, similarity computation, and model management.

#### `Embed(EmbedRequest)`

Generate an embedding for a single text string.

**Request:**

```rust
pub struct EmbedRequest {
  pub text: String,          // Text to embed
  pub normalize: Option<bool>,  // Whether to normalize the embedding (default: true)
}
```

**Response:**

```rust
pub struct EmbedResponse {
  pub embedding: Vec<f32>,          // The generated embedding vector
  pub dimension: usize,                   // Dimension of the embedding
  pub processing_time_ms: u128,         // Processing time in milliseconds
}
```

**Usage:**

```rust
// Rust client example (using tonic)
use vecboost::embedding_service_client::EmbeddingServiceClient;
use vecboost::EmbedRequest;

async fn embed_text() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://[::1]:50051").await?;
    
    let request = tonic::Request::new(EmbedRequest {
        text: "Hello, world!".to_string(),
        normalize: Some(true),
    });
    
    let response = client.embed(request).await?;
    println!("Embedding: {:?}", response.into_inner().embedding);
    Ok(())
}
```

#### `EmbedBatch(BatchEmbedRequest)`

Generate embeddings for multiple text strings in a batch.

**Request:**

```rust
pub struct BatchEmbedRequest {
  pub texts: Vec<String>,                // Texts to embed
  pub mode: Option<AggregationMode>,     // Aggregation mode (for document-level embeddings)
  pub normalize: Option<bool>,           // Whether to normalize embeddings (default: true)
}
```

**Response:**

```rust
pub struct BatchEmbedResponse {
  pub embeddings: Vec<BatchEmbeddingResult>,  // Generated embeddings
  pub dimension: usize,                       // Dimension of each embedding
  pub processing_time_ms: u128,               // Processing time in milliseconds
}

pub struct BatchEmbeddingResult {
  pub text_preview: String,   // Preview of the text (first few characters)
  pub embedding: Vec<f32>,    // The generated embedding vector
}
```

#### `ComputeSimilarity(SimilarityRequest)`

Compute similarity between two texts.

**Request:**

```rust
pub struct SimilarityRequest {
  pub source: String,  // Source text
  pub target: String,  // Target text
}
```

**Response:**

```rust
pub struct SimilarityResponse {
  pub score: f32,      // Similarity score
}
```

#### `EmbedFile(FileEmbedRequest)`

Generate embeddings for a file with support for different aggregation modes.

**Request:**

```rust
pub struct FileEmbedRequest {
  pub path: String,              // Path to the file
  pub mode: Option<AggregationMode>,  // Aggregation mode
}
```

**Response:**

```rust
pub struct FileEmbedResponse {
  pub mode: AggregationMode,     // Aggregation mode used
  pub stats: FileProcessingStats, // Processing statistics
  pub embedding: Option<Vec<f32>>,  // Document-level embedding (if applicable)
  pub paragraphs: Option<Vec<ParagraphEmbedding>>,  // Paragraph-level embeddings (if applicable)
}

pub struct FileProcessingStats {
  pub total_chunks: usize,       // Total chunks processed
  pub successful_chunks: usize,  // Successfully processed chunks
  pub failed_chunks: usize,      // Failed chunks
  pub processing_time_ms: u128,  // Processing time in milliseconds
}

pub struct ParagraphEmbedding {
  pub embedding: Vec<f32>,       // Paragraph embedding
  pub position: usize,           // Position in file
  pub text_preview: String,      // Preview of the paragraph
}
```

#### `ModelSwitch(ModelSwitchRequest)`

Switch to a different embedding model and/or device.

**Request:**

```rust
pub struct ModelSwitchRequest {
    pub model_name: String,                // Name of the model to switch to
    pub model_path: Option<PathBuf>,       // Path to the model files
    pub tokenizer_path: Option<PathBuf>,   // Path to the tokenizer files
    pub device: Option<DeviceType>,        // Device type: Cpu, Cuda, Metal, etc.
    pub max_batch_size: Option<usize>,     // Maximum batch size
    pub pooling_mode: Option<PoolingMode>, // Pooling mode for embeddings
    pub expected_dimension: Option<usize>, // Expected embedding dimension
    pub memory_limit_bytes: Option<u64>,   // Memory limit for the model
    pub oom_fallback_enabled: Option<bool>,// Enable OOM fallback to CPU
}
```

**Response:**

```rust
pub struct ModelSwitchResponse {
    pub previous_model: Option<String>,   // Name of the previously used model
    pub current_model: String,            // Name of the currently active model
    pub success: bool,                    // Whether the switch was successful
    pub message: String,                  // Status message
}
```

#### `GetCurrentModel(Empty)`

Get information about the currently loaded model.

**Response:**

```rust
pub struct ModelInfo {
  pub name: String,              // Model name
  pub engine_type: String,       // Engine type
  pub dimension: Option<usize>,  // Embedding dimension
  pub is_loaded: bool,           // Whether the model is loaded
}
```

#### `GetModelInfo(Empty)`

Get detailed metadata about the currently loaded model.

**Response:**

```rust
pub struct ModelMetadata {
  pub name: String,                // Model name
  pub version: String,             // Model version
  pub engine_type: String,         // Engine type
  pub dimension: Option<usize>,    // Embedding dimension
  pub max_input_length: usize,     // Maximum input length
  pub is_loaded: bool,             // Whether the model is loaded
  pub loaded_at: Option<String>,   // Timestamp when the model was loaded
}
```

#### `ListModels(Empty)`

List all available models.

**Response:**

```rust
pub struct ModelListResponse {
  pub models: Vec<ModelInfo>,    // List of available models
  pub total_count: usize,        // Total number of models
}
```

#### `HealthCheck(Empty)`

Check the health status of the service.

**Response:**

```rust
pub struct HealthResponse {
  pub status: String,            // Health status ("OK", "ERROR")
  pub version: String,           // Service version
  pub uptime: String,            // Service uptime
  pub model_loaded: Option<String>,  // Name of loaded model (if any)
}
```

______________________________________________________________________

### InferenceEngine

`InferenceEngine` is the abstract interface for all embedding engines in vecboost. It provides a unified API for generating embeddings regardless of the underlying implementation.

```rust
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Generate embedding for a single text
    fn embed(&self, text: &str) -> Result<Vec<f32>, AppError>;
    
    /// Generate embeddings for multiple texts in a batch
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError>;
    
    /// Get current precision setting
    fn precision(&self) -> &Precision;
    
    /// Check if mixed precision is supported
    fn supports_mixed_precision(&self) -> bool;
    
    /// Check if fallback to CPU has been triggered
    fn is_fallback_triggered(&self) -> bool {
        false
    }
    
    /// Try to fallback to CPU in case of OOM errors
    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), AppError>;
}
```

______________________________________________________________________

### Engine Implementations

#### CandleEngine

CandleEngine is the native Rust implementation using the Candle ML framework, optimized for performance and GPU acceleration.

**Features:**

- Native Rust implementation
- GPU acceleration (CUDA/Metal)
- Mixed precision support
- Automatic CPU fallback on OOM errors
- Model recovery mechanisms

**Usage:**

```rust
use vecboost::engine::candle_engine::CandleEngine;
use vecboost::config::model::{ModelConfig, Precision, EngineType, DeviceType, PathBuf};

fn create_candle_engine() -> Result<CandleEngine, AppError> {
    let config = ModelConfig {
        name: "BAAI/bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Candle,
        model_path: PathBuf::from("models/bge-small-en-v1.5"),
        tokenizer_path: None,
        device: DeviceType::Cuda,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: true,
        model_sha256: None,
    };
    
    let engine = CandleEngine::new(&config, Precision::Fp16)?;
    Ok(engine)
}
```

#### OnnxEngine

OnnxEngine provides support for ONNX models, enabling compatibility with models from various frameworks.

**Features:**

- ONNX model support
- Cross-framework compatibility
- CPU and GPU support
- Batch processing optimization

**Usage:**

```rust
use vecboost::engine::onnx_engine::OnnxEngine;
use vecboost::config::model::{ModelConfig, Precision, EngineType, DeviceType, PathBuf};

#[cfg(feature = "onnx")]
fn create_onnx_engine() -> Result<OnnxEngine, AppError> {
    let config = ModelConfig {
        name: "BAAI/bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Onnx,
        model_path: PathBuf::from("/path/to/model.onnx"),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: true,
        model_sha256: None,
    };
    
    let engine = OnnxEngine::new(&config, Precision::Fp32)?;
    Ok(engine)
}
```

______________________________________________________________________

## Error Handling

### `AppError`

Common error variants encountered during embedding operations.

| Variant | Description |
|---------|-------------|
| `ConfigError` | Error related to configuration |
| `ModelLoadError` | Error loading the model |
| `ModelFileCorrupted` | Corrupted model file |
| `ModelIntegrityError` | Model integrity check failed |
| `TokenizationError` | Error during text tokenization |
| `InferenceError` | Error during embedding generation |
| `OutOfMemory` | Out-of-memory error |
| `InvalidInput` | Invalid input validation |
| `NotFound` | Resource not found |
| `ModelNotLoaded` | Model not loaded |
| `AuthenticationError` | Authentication error |
| `SecurityError` | Security-related error |
| `IoError` | Input/output error |

______________________________________________________________________

## Type Definitions

### `Precision`

```rust
pub enum Precision {
    Fp32,  // Single precision float
    Fp16,  // Half precision float
    Int8,  // 8-bit integer quantization
}
```

### `EngineType`

```rust
pub enum EngineType {
    Candle, // Native Rust Candle engine
    #[cfg(feature = "onnx")]
    Onnx,   // ONNX engine (requires "onnx" feature)
}
```

### `DeviceType`

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    Cpu,     // CPU
    Cuda,    // NVIDIA GPU
    Metal,   // Apple Silicon GPU
    #[serde(rename = "amd")]
    Amd,     // AMD GPU (ROCm)
    #[serde(rename = "opencl")]
    OpenCL,  // OpenCL devices
}
```

### `AggregationMode`

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregationMode {
    Document,     // Aggregate to a single document embedding
    Paragraph,    // Generate embeddings for each paragraph
    Chunk,        // Generate embeddings for each chunk
    Average,      // Average embeddings across all chunks
    Max,          // Take max value across embeddings
    Min,          // Take min value across embeddings
}
```

### `PoolingMode`

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PoolingMode {
    Mean,         // Mean pooling (default)
    Max,          // Max pooling
    MeanMax,      // Concatenation of mean and max
    cls,          // CLS token pooling
    last,         // Last token pooling
}
```

______________________________________________________________________

## Examples

### Basic Embedding Generation

```rust
use vecboost::service::embedding::EmbeddingService;
use vecboost::engine::candle_engine::CandleEngine;
use vecboost::config::model::{ModelConfig, Precision, EngineType, DeviceType, PathBuf};
use vecboost::domain::EmbedRequest;
use std::sync::Arc;
use tokio::sync::RwLock;

async fn generate_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let config = ModelConfig {
        name: "BAAI/bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Candle,
        model_path: PathBuf::from("models/bge-small-en-v1.5"),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: true,
        model_sha256: None,
    };
    
    let engine = Arc::new(RwLock::new(CandleEngine::new(&config, Precision::Fp32)?));
    let embedding_service = EmbeddingService::new(engine, Some(config));
    
    let request = EmbedRequest {
        text: "The quick brown fox jumps over the lazy dog".to_string(),
        normalize: Some(true),
    };
    
    let response = embedding_service.process_text(request).await?;
    println!("Embedding dimension: {}", response.dimension);
    println!("Processing time: {}ms", response.processing_time_ms);
    
    Ok(())
}
```

### Batch Embedding with GPU Acceleration

```rust
use vecboost::service::embedding::EmbeddingService;
use vecboost::engine::candle_engine::CandleEngine;
use vecboost::config::model::{ModelConfig, Precision, EngineType, DeviceType, PathBuf};
use vecboost::domain::BatchEmbedRequest;
use std::sync::Arc;
use tokio::sync::RwLock;

async fn generate_batch_embeddings() -> Result<(), Box<dyn std::error::Error>> {
    let config = ModelConfig {
        name: "BAAI/bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Candle,
        model_path: PathBuf::from("models/bge-small-en-v1.5"),
        tokenizer_path: None,
        device: DeviceType::Cuda,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: true,
        model_sha256: None,
    };
    
    let engine = Arc::new(RwLock::new(CandleEngine::new(&config, Precision::Fp16)?));
    let embedding_service = EmbeddingService::new(engine, Some(config));
    
    let texts = vec![
        "First sentence".to_string(),
        "Second sentence".to_string(),
        "Third sentence".to_string(),
    ];
    
    let request = BatchEmbedRequest {
        texts,
        normalize: Some(true),
    };
    
    let response = embedding_service.process_batch(request).await?;
    println!("Generated {} embeddings", response.embeddings.len());
    println!("Total processing time: {}ms", response.processing_time_ms);
    
    Ok(())
}
```

### Model Switching

```rust
use vecboost::service::embedding::EmbeddingService;
use vecboost::engine::candle_engine::CandleEngine;
use vecboost::model::manager::ModelManager;
use vecboost::config::model::{ModelConfig, Precision, EngineType, DeviceType, PathBuf};
use vecboost::domain::ModelSwitchRequest;
use std::sync::Arc;
use tokio::sync::RwLock;

async fn switch_model() -> Result<(), Box<dyn std::error::Error>> {
    let config = ModelConfig {
        name: "BAAI/bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Candle,
        model_path: PathBuf::from("models/bge-small-en-v1.5"),
        tokenizer_path: None,
        device: DeviceType::Cuda,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: true,
        model_sha256: None,
    };
    
    let engine = Arc::new(RwLock::new(CandleEngine::new(&config, Precision::Fp16)?));
    let model_manager = Arc::new(ModelManager::new()?);
    let embedding_service = EmbeddingService::with_manager(engine, Some(config), model_manager);
    
    let request = ModelSwitchRequest {
        model_name: "BAAI/bge-large-en-v1.5".to_string(),
        model_path: Some(PathBuf::from("models/bge-large-en-v1.5")),
        tokenizer_path: None,
        device: Some(DeviceType::Cuda),
        max_batch_size: None,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: None,
    };
    
    let response = embedding_service.switch_model(request).await?;
    println!("Model switch successful: {}", response.success);
    println!("Previous model: {:?}", response.previous_model);
    println!("Current model: {}", response.current_model);
    println!("Message: {}", response.message);
    
    Ok(())
}
```
