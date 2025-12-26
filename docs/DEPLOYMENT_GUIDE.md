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

#### 架构概述

VecBoost 提供完整的 AMD GPU 支持，通过以下两种后端实现：

- **ROCm**: AMD 的开源计算平台，提供高性能 CUDA 兼容的 GPU 加速
- **OpenCL**: 跨平台并行计算标准，提供广泛的 AMD GPU 兼容性

**设备检测策略**：
1. 优先检测 ROCm 设备（索引 0-3）
2. 如果未找到 ROCm 设备，则回退到 OpenCL 检测
3. 自动选择第一个可用设备作为主设备
4. 支持多 GPU 环境和设备切换

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

#### 设备管理

**设备初始化**

系统启动时会自动初始化 AMD GPU 设备管理器，执行以下步骤：

1. 扫描 ROCm 设备（索引 0-3）
2. 如果未找到 ROCm 设备，扫描 OpenCL 设备（索引 0-3）
3. 记录每个设备的详细信息（名称、显存、计算能力等）
4. 设置第一个可用设备为主设备
5. 初始化内存跟踪和状态监控

**设备信息**

每个 AMD GPU 设备包含以下信息：

- `name`: 设备名称（如 "AMD GPU (ROCm) - Device 0"）
- `device_id`: 设备索引
- `vram_bytes`: 显存大小（字节）
- `compute_capability`: 计算能力版本
- `opencl_version`: OpenCL 版本
- `roc_version`: ROCm 版本（如适用）
- `driver_version`: 驱动版本

**多 GPU 支持**

系统支持多 AMD GPU 环境：

```rust
// 获取所有可用设备
let devices = manager.devices().await;

// 获取主设备
let primary = manager.primary_device().await;

// 获取特定索引的设备
let device = manager.get_device(index).await;

// 设置主设备
manager.set_primary(index).await;

// 获取设备数量
let count = manager.device_count().await;
```

#### 内存管理

**显存分配与释放**

系统提供精细的显存管理：

```rust
// 分配显存（检查可用性）
let success = device.allocate(bytes);

// 释放显存
device.deallocate(bytes);

// 获取可用显存
let available = device.available_memory();

// 获取显存使用率
let usage_percent = device.memory_usage_percent();
```

**内存监控**

实时监控所有 AMD GPU 的内存使用：

```rust
// 获取总显存
let total_vram = manager.total_vram().await;

// 获取可用显存
let available_vram = manager.available_vram().await;

// 获取内存使用摘要
let summary = manager.memory_usage_summary().await;
// 输出示例: "AMD GPU Memory: 4294967296 bytes used / 17179869184 bytes total (25.0%)"
```

**OOM 防护**

系统内置显存不足防护：

- 分配前检查可用显存
- 超过显存限制时拒绝分配
- 记录警告日志
- 支持自动回退到 CPU

#### 设备状态管理

**设备忙状态**

系统跟踪每个设备的使用状态：

```rust
// 检查设备是否忙碌
let busy = device.is_busy();

// 设置设备忙碌状态
device.set_busy(true);  // 标记为忙碌
device.set_busy(false); // 标记为空闲
```

**设备重置**

重置所有设备的状态和内存：

```rust
// 重置所有设备（清除内存使用和忙碌状态）
manager.reset().await;
```

#### 精度支持

AMD GPU 支持以下精度模式：

| 精度 | 说明 | 性能 | 内存占用 |
|------|------|------|----------|
| `fp32` | 单精度浮点 | 标准 | 高 |
| `fp16` | 半精度浮点 | 2x 提升 | 50% |
| `bf16` | BFloat16 | 接近 fp16 | 50% |

```rust
// 检查设备是否支持特定精度
let supported = device.supports_precision("fp16");
```

**推荐配置**：

- **高性能场景**: 使用 `fp16` 或 `bf16`
- **高精度场景**: 使用 `fp32`
- **显存受限**: 使用 `fp16` 或 `bf16`

#### 操作支持

AMD GPU 支持以下计算操作：

- `matrix_multiply`: 矩阵乘法
- `convolution`: 卷积运算
- `activation`: 激活函数
- `normalization`: 归一化
- `reduction`: 归约操作

```rust
// 检查设备是否支持特定操作
let supported = device.supports_operation("matrix_multiply");
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

**原因分析**:
- ROCm 运行时未正确安装
- 驱动版本不兼容
- 内核模块未加载
- 用户权限不足

**诊断步骤**:

```bash
# 1. 检查 ROCm 安装
rocminfo

# 2. 检查 GPU 信息
rocm-smi

# 3. 检查内核模块
lsmod | grep amdgpu

# 4. 检查设备节点
ls -la /dev/dri/
ls -la /dev/kfd

# 5. 检查用户组权限
groups $USER
```

**解决方案**:

```bash
# 重新加载驱动
sudo modprobe -r amdgpu
sudo modprobe amdgpu

# 添加用户到必要组
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# 重新登录使权限生效
```

**问题**: OpenCL 初始化失败

**原因分析**:
- OpenCL ICD 加载器未安装
- 驱动不支持 OpenCL
- ICD 配置文件缺失

**诊断步骤**:

```bash
# 1. 检查 OpenCL 平台
clinfo -l

# 2. 检查 ICD 加载器
clinfo | grep "Number of platforms"

# 3. 检查 ICD 配置
ls /etc/OpenCL/vendors/

# 4. 检查 OpenCL 库
ldconfig -p | grep libOpenCL
```

**解决方案**:

```bash
# Ubuntu/Debian
sudo apt install ocl-icd-opencl-dev ocl-icd-libopencl1

# Fedora/RHEL
sudo dnf install ocl-icd-devel ocl-icd

# 验证安装
clinfo
```

**问题**: 显存不足（OOM）

**原因分析**:
- 批处理大小过大，单次推理显存占用超过可用显存
- 模型权重加载占用大量显存
- 多个并发任务同时执行
- 显存碎片化导致无法分配连续内存块
- 其他进程占用显存

**诊断步骤**:

```bash
# 1. 检查当前显存使用情况
rocm-smi

# 2. 查看详细显存信息
rocm-smi --showmemuse

# 3. 监控显存使用趋势
watch -n 1 rocm-smi

# 4. 检查系统日志中的 OOM 错误
journalctl -xe | grep -i "out of memory"

# 5. 检查 VecBoost 日志中的内存分配失败
grep -i "memory allocation failed" /var/log/vecboost/*.log
```

**解决方案**:

```bash
# 方案 1: 减小批处理大小
export VECBOOST_BATCH_SIZE=4  # 从默认值减小

# 方案 2: 启用 CPU 回退
export VECBOOST_FALLBACK_ENABLED=true
export VECBOOST_MEMORY_THRESHOLD=90  # 显存使用率超过 90% 时回退

# 方案 3: 使用低精度模型
export VECBOOST_PRECISION=fp16  # 或 bf16

# 方案 4: 限制并发任务数
export VECBOOST_MAX_CONCURRENT_TASKS=2

# 方案 5: 清理显存
# 重启 VecBoost 服务
sudo systemctl restart vecboost

# 或者在代码中手动释放显存
# AmdDevice::deallocate() 会自动跟踪和释放
```

**代码示例**:

```rust
// 监控显存使用
let manager = AmdDeviceManager::new().await?;
manager.initialize().await?;

// 获取显存使用摘要
let summary = manager.memory_usage_summary().await;
println!("{}", summary);

// 检查可用显存
let available = manager.available_vram().await;
if available < required_memory {
    // 减小批处理大小或启用 CPU 回退
}
```

**问题**: 设备初始化失败

**原因分析**:
- 设备索引超出范围
- 设备被其他进程占用
- 驱动版本不兼容
- 硬件故障
- 权限问题

**诊断步骤**:

```bash
# 1. 检查设备列表
rocminfo | grep "Device Name"

# 2. 检查设备状态
rocm-smi --showuse

# 3. 检查设备占用
lsof /dev/dri/card*
lsof /dev/kfd

# 4. 检查驱动版本
rocm-smi --showdriverversion

# 5. 检查内核日志
dmesg | grep -i amdgpu
```

**解决方案**:

```bash
# 方案 1: 检查并释放设备占用
# 查找占用进程
fuser -v /dev/dri/card0

# 终止占用进程（谨慎操作）
sudo kill -9 <PID>

# 方案 2: 更新驱动
# Ubuntu/Debian
sudo apt update
sudo apt install amdgpu-dkms rocm-dev

# 方案 3: 检查设备索引
# 确保设备索引在有效范围内（0-3）
# 使用 rocm-smi 查看可用设备数量

# 方案 4: 重置设备
sudo rmmod amdgpu
sudo modprobe amdgpu
```

**问题**: 性能低于预期

**原因分析**:
- 使用了 CPU 回退而非 GPU
- 批处理大小过小，GPU 利用率低
- 精度设置不当（fp32 vs fp16/bf16）
- 数据传输瓶颈（CPU-GPU）
- 多 GPU 负载不均衡

**诊断步骤**:

```bash
# 1. 检查设备使用情况
rocm-smi --showuse

# 2. 检查 GPU 利用率
rocm-smi --showgpuclocks

# 3. 监控性能指标
# 查看 VecBoost 性能日志
tail -f /var/log/vecboost/metrics.log

# 4. 检查当前配置
env | grep VECBOOST
```

**解决方案**:

```bash
# 方案 1: 确保使用 GPU
export VECBOOST_DEVICE_TYPE=gpu
export VECBOOST_FALLBACK_ENABLED=false

# 方案 2: 优化批处理大小
# 根据显存大小调整
export VECBOOST_BATCH_SIZE=16  # 8GB 显存推荐 16-32

# 方案 3: 使用半精度
export VECBOOST_PRECISION=fp16  # 提升吞吐量约 2 倍

# 方案 4: 启用缓存
export VECBOOST_CACHE_ENABLED=true
export VECBOOST_CACHE_SIZE=10000

# 方案 5: 多 GPU 负载均衡
export VECBOOST_MULTI_GPU_ENABLED=true
```

**性能优化代码示例**:

```rust
// 使用性能监控
let manager = AmdDeviceManager::new().await?;
manager.initialize().await?;

// 获取设备信息
let device = manager.primary_device().await.unwrap();
println!("Device: {}", device.name());
println!("Compute Units: {}", device.compute_units());
println!("Max Work Group Size: {}", device.max_work_group_size());

// 检查设备是否忙碌
if !device.is_busy() {
    // 执行推理任务
}
```

**问题**: 构建失败（ROCm）

**原因分析**:
- ROCm 工具链未正确安装
- CUDA feature 与 ROCm 冲突
- 依赖库版本不匹配
- 编译器版本不兼容
- 环境变量未正确设置

**诊断步骤**:

```bash
# 1. 检查 ROCm 安装
which hipcc
hipcc --version

# 2. 检查 ROCm 库
ls /opt/rocm/lib/

# 3. 检查环境变量
echo $ROCM_HOME
echo $LD_LIBRARY_PATH

# 4. 检查 Cargo feature
cargo tree -i candle-core

# 5. 查看详细编译错误
cargo build --features rocm 2>&1 | tee build.log
```

**解决方案**:

```bash
# 方案 1: 安装 ROCm 工具链
# Ubuntu/Debian
sudo apt install rocm-dev rocm-libs rocm-utils

# 设置环境变量
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# 方案 2: 确保不启用 CUDA feature
cargo build --features rocm --no-default-features

# 方案 3: 更新 Rust 工具链
rustup update
rustup default stable

# 方案 4: 清理并重新构建
cargo clean
cargo build --features rocm

# 方案 5: 检查依赖版本
# 确保 candle-core 版本支持 ROCm
cargo update -p candle-core
```

**构建配置示例**:

```toml
# Cargo.toml
[dependencies]
candle-core = { version = "0.4", features = ["rocm"] }
candle-nn = { version = "0.4", features = ["rocm"] }

# 确保不同时启用 cuda 和 rocm
# [dependencies]
# candle-core = { version = "0.4", features = ["cuda"] }  # ❌ 与 rocm 冲突
```

**常见构建错误及修复**:

```bash
# 错误: hipcc not found
# 修复: 安装 ROCm 工具链
sudo apt install rocm-dev

# 错误: undefined reference to hip*
# 修复: 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# 错误: feature conflict
# 修复: 移除冲突的 feature
cargo build --features rocm --no-default-features
```
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
