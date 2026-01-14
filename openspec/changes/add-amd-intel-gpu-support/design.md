# Design: AMD and Intel GPU Support

## Overview

This document outlines the architectural design for adding AMD ROCm and Intel XPU support to VecBoost.

## Architecture Decisions

### Decision 1: ONNX Runtime as Primary GPU Backend

**Context**: Candle 引擎不原生支持 AMD/Intel GPU，而 ONNX Runtime 提供全面的执行提供者支持。

**Decision**: 使用 ONNX Runtime 作为 AMD/Intel GPU 加速的主要引擎。

**Pros**:
- 完整的执行提供者生态系统
- AMD ROCm EP 官方支持
- Intel XPU EP 官方支持
- 跨平台兼容

**Cons**:
- 比 Candle 有更大的运行时开销
- 需要额外依赖

**Impact**: `src/engine/onnx_engine.rs` 将成为 AMD/Intel 的主要引擎

---

### Decision 2: Unified Device Manager Pattern

**Context**: 当前有独立的 `CudaDeviceManager` 和 `AmdDeviceManager`，需要扩展支持 Intel。

**Decision**: 创建设备抽象 Trait，允许统一的设备管理。

```rust
trait GpuDeviceManager {
    fn is_available(&self) -> bool;
    fn devices(&self) -> Vec<DeviceInfo>;
    fn primary_device(&self) -> Option<DeviceInfo>;
    fn memory_usage(&self) -> GpuMemoryStats;
    fn initialize(&self) -> Result<(), AppError>;
}

struct IntelDeviceManager { /* ... */ }
impl GpuDeviceManager for IntelDeviceManager { /* ... */ }
```

**Pros**:
- 统一的设备选择逻辑
- 易于扩展新 GPU 厂商
- 清晰的职责分离

**Cons**:
- 需要重构现有代码
- 抽象可能影响性能

**Impact**: `src/device/manager.rs` 将使用 Trait 对象

---

### Decision 3: Detection Strategy

**Context**: 不同 GPU 厂商使用不同的检测机制。

**Decision**: 使用多策略检测，按优先级尝试。

#### AMD Detection Strategy
1. ROCm: `rocm-smi` (首选)
2. OpenCL: `clinfo` (备选)

#### Intel Detection Strategy
1. oneAPI/XPU: `sycl-ls` (Linux)
2. OpenCL: `clinfo` (跨平台)
3. DirectML: Windows API (Windows)

```rust
enum GpuDetectionMethod {
    Rocm,
    OpenCL,
    OneAPI,
    DirectML,
}

fn detect_amd_gpu() -> Option<Vec<AmdGpuInfo>> {
    if let Ok(devices) = detect_via_rocm() {
        return Some(devices);
    }
    if let Ok(devices) = detect_via_opencl() {
        return Some(devices);
    }
    None
}
```

---

### Decision 4: Memory Pool Architecture

**Context**: 现有 `CudaMemoryPool` 需要扩展支持 AMD/Intel。

**Decision**: 创建统一的 `GpuMemoryPool` Trait。

```rust
trait GpuMemoryPool {
    type Pointer;
    
    fn allocate(&self, size: usize) -> Result<Self::Pointer, AppError>;
    fn deallocate(&self, ptr: Self::Pointer);
    fn memory_usage(&self) -> GpuMemoryStats;
    fn clear(&self);
}

struct IntelMemoryPool { /* ... */ }
impl GpuMemoryPool for IntelMemoryPool { /* ... */ }
```

**Implementation Notes**:
- ROCm 使用 hipMalloc/hipFree
- Intel XPU 使用 oneAPI Malloc
- 提供占位符实现，后期优化

---

### Decision 5: Precision Selection

**Context**: 不同 GPU 支持不同的精度格式。

**Decision**: 根据 GPU 类型选择最优精度。

| GPU Type | Preferred Precision | Fallback |
|----------|-------------------|----------|
| NVIDIA CUDA | FP16, INT8 | FP32 |
| AMD ROCm | FP16 | FP32 |
| Intel XPU | BF16 | FP16, FP32 |
| Apple Metal | FP16 | FP32 |

```rust
fn select_precision(device: &DeviceType, config: &Precision) -> Precision {
    match device {
        DeviceType::Cuda => config, // CUDA 支持所有精度
        DeviceType::Amd => {
            if *config == Precision::Fp16 { Precision::Fp16 }
            else { Precision::Fp32 }
        }
        DeviceType::Intel => {
            if *config == Precision::Fp16 { Precision::Bf16 }
            else { Precision::Fp32 }
        }
        DeviceType::Cpu => Precision::Fp32,
    }
}
```

---

## Module Structure

```
src/device/
├── mod.rs              # 模块入口，设备 Trait 定义
├── cuda.rs             # CUDA 设备管理 (保持)
├── amd.rs              # AMD 设备管理 (增强)
├── intel.rs            # Intel 设备管理 (新建)
├── manager.rs          # 统一设备管理器 (重构)
└── memory_pool/
    ├── mod.rs          # 内存池模块
    ├── cuda_pool.rs    # CUDA 内存池 (保持)
    ├── amd_pool.rs     # AMD 内存池 (新建)
    └── intel_pool.rs   # Intel 内存池 (新建)
```

---

## Execution Provider Configuration

### ONNX Runtime EP 优先级

```rust
fn configure_execution_providers(device_type: DeviceType) -> Vec<ExecutionProvider> {
    match device_type {
        DeviceType::Cuda => {
            vec![
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ]
        }
        DeviceType::Amd | DeviceType::OpenCL => {
            vec![
                #[cfg(feature = "rocm")]
                ROCmExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ]
        }
        DeviceType::Intel => {
            vec![
                #[cfg(feature = "xpu")]
                XPUExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ]
        }
        DeviceType::Cpu => {
            vec![CPUExecutionProvider::default().build()]
        }
    }
}
```

---

## Error Handling

### Graceful Degradation

```rust
fn create_gpu_engine(
    config: &ModelConfig,
    device_type: DeviceType,
) -> Result<Box<dyn InferenceEngine>, AppError> {
    match device_type {
        DeviceType::Cuda => {
            // CUDA 完整支持
            CandleEngine::new(config, Precision::Fp16)
        }
        DeviceType::Amd => {
            // AMD: 尝试 ONNX + ROCm，回退到 CPU
            if cfg!(feature = "rocm") {
                OnnxEngine::new(config, Precision::Fp16)
            } else {
                warn!("ROCm feature not enabled, falling back to CPU");
                OnnxEngine::new(config, Precision::Fp32)
            }
        }
        DeviceType::Intel => {
            // Intel: 尝试 ONNX + XPU，回退到 CPU
            if cfg!(feature = "xpu") {
                OnnxEngine::new(config, Precision::Bf16)
            } else {
                warn!("XPU feature not enabled, falling back to CPU");
                OnnxEngine::new(config, Precision::Fp32)
            }
        }
        DeviceType::Cpu => {
            OnnxEngine::new(config, Precision::Fp32)
        }
    }
}
```

---

## Testing Strategy

### Unit Tests
- 设备检测 (模拟环境)
- 内存分配/释放
- 精度选择逻辑
- 配置解析

### Integration Tests
- ONNX Runtime EP 初始化
- 端到端 embedding 推理
- 内存池压力测试
- 多 GPU 切换

### Performance Tests
- 吞吐量基准
- 延迟测试
- 内存使用监控

---

## Open Questions

1. **ONNX Runtime ROCm/XPU 包可用性**: 需要确认 crates.io 上是否有可用的 Rust 绑定
2. **测试环境**: 是否需要 CI 环境配置 AMD ROCm/Intel XPU
3. **回退策略**: 如果 ONNX Runtime EP 不可用，是否实现纯 OpenCL 后端

---

**Created**: 2025-01-14  
**Version**: 1.0
