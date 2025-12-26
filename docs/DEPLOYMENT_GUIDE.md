# 部署配置指南

本文档介绍如何配置和构建 VecBoost 项目，包括不同硬件平台的 GPU 加速支持。

## 构建命令

### 基本构建

```bash
# 默认构建（无GPU加速，仅CPU）
cargo build --release
```

### NVIDIA GPU 支持（CUDA）

```bash
# 启用 CUDA 支持
cargo build --release --features cuda

# 或者使用 GPU 加速的 ONNX Runtime
cargo build --release --features onnx

# 同时启用 CUDA 和 ONNX
cargo build --release --features cuda,onnx
```

### Apple Silicon GPU 支持（Metal）

```bash
# 在 macOS 上启用 Metal 支持（需要 Apple Silicon 芯片）
cargo build --release --features metal
```

### AMD GPU 支持（ROCm/OpenCL）

#### 环境要求

**ROCm 支持**

- **操作系统**: Linux（推荐 Ubuntu 20.04/22.04、RHEL 8/9）
- **ROCm 版本**: 5.4 - 6.0（推荐 5.6+）
- **AMD GPU**: CDNA 架构（MI100、MI200 系列）或 RDNA 2/3（RX 6000/7000 系列）
- **内核**: 5.4+（Ubuntu）或对应发行版支持的内核版本

```bash
# 检查 ROCm 安装
rocm-smi

# 检查 GPU 检测
clinfo | grep "Device Name"
```

**OpenCL 支持**

- **操作系统**: Linux、Windows
- **AMD 驱动**: 23.10+（Windows）、AMDGPU-Pro 或 Mesa（Linux）
- **OpenCL SDK**: AMD SDK 3.0+（可选，用于开发）

```bash
# Linux 检查 OpenCL
ls -la /dev/dri/

# Windows 检查
clinfo | grep -i amd
```

#### 安装配置

**Ubuntu ROCm 安装**

```bash
# 添加 ROCm 仓库
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.6.1/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# 安装 ROCm
sudo apt update
sudo apt install rocm-dev rocm-libs

# 设置环境变量
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocprofiler/bin:/opt/rocm/opencl/bin' >> ~/.bashrc
source ~/.bashrc
```

**验证 ROCm 安装**

```bash
# 检查 ROCm 版本
rocminfo

# 检查 GPU 信息
rocm-smi

# 运行 OpenCL 测试
clinfo -l
```

**OpenCL 环境配置**

```bash
# Ubuntu 安装 OpenCL
sudo apt install ocl-icd-opencl-dev

# macOS 安装（通过 Homebrew）
brew install oclgrind

# Windows 安装 AMD 驱动后自动支持
```

#### 构建命令

```bash
# 启用 ROCm 支持（需要 ROCm 运行时）
cargo build --release --features rocm

# 使用 OpenCL 后端（跨平台 AMD 支持）
cargo build --release --features opencl

# ONNX Runtime ROCm 支持
cargo build --release --features onnx,rocm

# 完整 AMD GPU 支持（推荐）
cargo build --release --features rocm,opencl,onnx
```

#### 设备配置

```bash
# 设置 AMD GPU 设备
export VECBOOST_DEVICE=amd

# 设置推理引擎（推荐 candle）
export VECBOOST_ENGINE=candle

# 设置精度（推荐 fp16）
export VECBOOST_PRECISION=fp16

# 内存使用阈值（百分比）
export VECBOOST_MEMORY_THRESHOLD=85

# 启用 OOM 时自动回退
export VECBOOST_FALLBACK_ENABLED=true
```

#### 性能优化

**ROCm 优化参数**

```bash
# 设置批处理大小（根据显存调整）
export VECBOOST_BATCH_SIZE=16

# 启用图优化
export VECBOOST_GRAPH_OPT=true

# 设置工作负载组大小
export VECBOOST_WORKGROUP_SIZE=256
```

**OpenCL 优化参数**

```bash
# 优化内存访问模式
export VECBOOST_MEM_OPT=true

# 启用管道缓冲
export VECBOOST_PIPE_BUF=4096
```

#### 故障排除

**问题**: ROCm 设备未检测

```bash
# 检查内核模块
lsmod | grep amdgpu

# 重新加载驱动
sudo modprobe amdgpu

# 检查权限
sudo usermod -aG video $USER
sudo usermod -aG render $USER
```

**问题**: OpenCL 初始化失败

```bash
# 检查 ICD 加载器
clinfo | grep "Number of platforms"

# 验证 OpenCL 安装
ls /etc/OpenCL/vendors/

# 安装 ICD 配置文件
sudo apt install ocl-icd-libopencl1
```

**问题**: 显存不足

```bash
# 监控显存使用
rocm-smi

# 减小批处理大小
export VECBOOST_BATCH_SIZE=8

# 启用 CPU 回退
export VECBOOST_FALLBACK_ENABLED=true
export VECBOOST_MEMORY_THRESHOLD=90
```

**问题**: 构建失败（ROCm）

```bash
# 清理并重新构建
cargo clean
rm -rf target

# 使用 ROCm 环境变量
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH

# 重新构建
cargo build --release --features rocm
```

### 完整功能构建

```bash
# 在 macOS 上启用所有功能
cargo build --release --features cuda,onnx,metal

# 在 Linux 上启用所有功能（包含 AMD GPU）
cargo build --release --features cuda,onnx,rocm,opencl

# AMD GPU 专用构建
cargo build --release --features rocm,opencl,onnx
```

## 环境要求

### CUDA 支持

- **操作系统**: Linux（推荐 Ubuntu 20.04+）、Windows 10+
- **CUDA Toolkit**: 11.8 或 12.0
- **NVIDIA Driver**: 525.x 或更新版本
- **GPU**: Compute Capability 7.0+ (Volta 及更新架构)

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 计算能力
nvidia-smi
```

### Metal 支持

- **操作系统**: macOS 12.0+（Monterey 或更新版本）
- **硬件**: Apple Silicon 芯片（M1、M2、M3 系列）
- **内存**: 推荐 8GB+ 统一内存

```bash
# 检查 Metal 设备（macOS）
system_profiler SPMemoryType
```

### ONNX Runtime 支持

- **操作系统**: Linux、Windows、macOS
- **无需特殊硬件要求**
- 跨平台推理支持

## 配置文件

### 环境变量配置

创建 `.env` 文件：

```bash
# 设备配置
DEVICE=cuda        # 可选值: cpu, cuda, metal
PRECISION=fp16     # 可选值: fp32, fp16, int8

# 模型配置
MODEL_ID=BAAI/bge-m3

# 推理引擎（可选）
ENGINE=candle      # 可选值: candle, onnx
```

### 设备选择逻辑

项目支持以下设备类型自动检测：

| 配置值 | 硬件要求 | 说明 |
|--------|----------|------|
| `cpu` | 无 | 使用 CPU 推理，兼容性最好 |
| `cuda` | NVIDIA GPU + CUDA | 使用 CUDA GPU 加速 |
| `metal` | Apple Silicon | 使用 Metal GPU 加速 |
| `auto` | 自动检测 | 自动选择最佳可用设备 |

## Docker 构建

### CUDA 环境

```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu20.04

# 安装 Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.0
RUN source "$HOME/.cargo/env" && cargo install cargo-chef

# 构建项目
COPY . .
RUN source "$HOME/.cargo/env" && cargo build --release --features cuda
```

### Metal 环境（macOS）

```bash
# 在 Apple Silicon Mac 上构建
arch -arm64 cargo build --release --features metal
```

## 性能建议

### CUDA 优化

- 使用 FP16 精度可提升 2x 性能
- 批量处理可提高 GPU 利用率
- 推荐批处理大小: 16-32

```bash
cargo build --release --features cuda
```

### Metal 优化

- Apple Silicon 统一内存架构减少数据传输
- 推荐使用 FP16 精度
- 批处理大小建议 16

```bash
cargo build --release --features metal
```

### CPU 优化

- 适用于无 GPU 环境
- 使用多线程加速
- 推荐批处理大小: 8

```bash
cargo build --release
```

## 故障排除

### CUDA 问题

**问题**: `cudarc` 构建失败

**解决**: 确保 CUDA Toolkit 版本兼容
```bash
# 检查版本
nvcc --version

# 使用兼容版本重新构建
cargo clean
cargo build --release --features cuda
```

**问题**: GPU 内存不足

**解决**: 启用自动回退到 CPU
```bash
# 配置环境变量
export VECBOOST_FALLBACK_ENABLED=true
export VECBOOST_MEMORY_THRESHOLD=90
```

### Metal 问题

**问题**: Metal 编译失败（非 macOS 系统）

**解决**: Metal 仅在 macOS 上可用，在其他系统上使用 CPU 或 CUDA

### ONNX Runtime 问题

**问题**: 缺少系统依赖

**解决**: 安装 ONNX Runtime 系统依赖
```bash
# Ubuntu
sudo apt-get install libomp5

# macOS
brew install libomp
```

## CI/CD 配置示例

```yaml
# GitHub Actions
jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            features: cuda,onnx
          - os: macos-latest
            features: metal,onnx
          - os: ubuntu-latest
            features: onnx
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: cargo build --release --features ${{ matrix.features }}
```

## 运行时配置

### 命令行参数

```bash
vecboost serve --device cuda --precision fp16 --batch-size 32
```

### 程序配置

```rust
use vecboost::config::ModelConfig;

let config = ModelConfig {
    device: DeviceType::Cuda,
    precision: Precision::Fp16,
    ..Default::default()
};
```
