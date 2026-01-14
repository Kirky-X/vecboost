# Change Proposal: Add AMD and Intel GPU Support

## Summary

实现 AMD 和 Intel GPU 支持，特别是 Intel XPU，以扩展 VecBoost 的硬件兼容性和性能覆盖范围。

## Problem Statement

当前 VecBoost 的 GPU 支持存在以下限制：

1. **NVIDIA GPU**: ✅ 完整支持 (CUDA)
2. **AMD GPU**: ⚠️ 部分支持 - 仅设备检测，无推理加速
   - Candle 引擎不支持 AMD ROCm
   - ONNX Runtime ROCm 执行提供者未启用
3. **Intel GPU**: ❌ 未实现 - 无检测代码，无运行时支持
   - Intel XPU (集成显卡/Arc 显卡) 完全缺失
   - oneAPI / SYCL 支持未实现

### 用户需求
- AMD ROCm 加速 (Linux ROCm 环境)
- Intel XPU 加速 (Intel 集成显卡/Arc 显卡)
- 统一的设备管理抽象

## Proposed Solution

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    VecBoost 推理引擎                             │
├───────────────────────┬─────────────────────────────────────────┤
│      Candle 引擎      │            ONNX 引擎                     │
│                       │                                          │
│  • CUDA ✓            │  • CUDA ✓ (CUDAExecutionProvider)        │
│  • Metal ✓           │  • AMD ROCm ✓ (ROCmExecutionProvider)    │
│  • CPU ✓             │  • Intel XPU ✓ (XPUExecutionProvider)    │
│  • AMD ROCm ✗        │  • CoreML ✓ (CoreMLExecutionProvider)    │
└───────────────────────┴─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    设备管理层 (Device Manager)                   │
├───────────────────────┬─────────────────────────────────────────┤
│  CudaDeviceManager    │  AmdDeviceManager                       │
│  (完整)               │  • ROCm 检测 ✓                          │
│                       │  • OpenCL 检测 ✓                        │
│                       │  • 推理加速 ✗                           │
│                       │                                         │
│                       │  IntelDeviceManager (新建)              │
│                       │  • XPU/OneAPI 检测                      │
│                       │  • SYCL/DirectML 检测                   │
│                       │  • 集成显卡/Arc 检测                    │
└───────────────────────┴─────────────────────────────────────────────────────────────────┘
```

### 技术策略

#### 1. AMD GPU 加速 (Phase 1)
- **ONNX Runtime**: 启用 ROCm 执行提供者
- **依赖**: `onnxruntime-rocm` 或 `ort` 的 ROCm 支持
- **验证**: 使用 rocm-smi 检测 ROCm 环境

#### 2. Intel XPU 加速 (Phase 2)
- **ONNX Runtime**: 启用 XPU 执行提供者
- **依赖**: `onnxruntime-xpu` (Intel oneAPI)
- **检测**: 使用 `sycl-ls` 或 `clinfo` 检测 Intel XPU
- **备选**: 使用 OpenCL 检测 Intel GPU

### 实现范围

#### Phase 1: AMD ROCm 加速
- [x] 设备检测 (已有)
- [ ] 启用 ONNX Runtime ROCm 执行提供者
- [ ] 添加 ROCm 内存池支持
- [ ] 测试验证

#### Phase 2: Intel XPU 支持
- [ ] 创建 `IntelDeviceManager`
- [ ] 实现 XPU/OneAPI 设备检测
- [ ] 启用 ONNX Runtime XPU 执行提供者
- [ ] 添加 Intel 内存池支持
- [ ] 测试验证

### Out of Scope

- Candle 引擎 AMD ROCm 原生支持 (需要 Candle 社区支持)
- DirectML 支持 (Windows 特定)
- SYCL 跨平台抽象 (超出当前范围)

## Dependencies

### 外部依赖
- `onnxruntime-rocm` (可选特性)
- `onnxruntime-xpu` (可选特性)
- `rocm-smi` (AMD ROCm 环境检测)
- Intel oneAPI / SYCL 工具链 (Intel XPU)

### 内部依赖
- `src/device/mod.rs` - 设备管理模块扩展
- `src/device/manager.rs` - 统一设备管理器
- `src/engine/onnx_engine.rs` - ONNX 引擎修改
- `src/config/model.rs` - DeviceType 枚举扩展

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ONNX Runtime ROCm/XPU 包不可用 | High | 提供纯 OpenCL 回退方案 |
| 硬件测试环境不足 | Medium | 添加模拟设备检测模式 |
| 内存池实现复杂 | Medium | 使用占位符实现，后期完善 |
| ROCm/XPU 版本兼容性 | Medium | 版本检测和优雅降级 |

## Success Criteria

### AMD GPU
- [ ] ROCm 设备检测成功
- [ ] ONNX Runtime 使用 ROCm 执行提供者
- [ ] embedding 推理在 AMD GPU 上加速
- [ ] FP16 精度支持

### Intel XPU
- [ ] XPU/集成显卡/Arc 检测成功
- [ ] ONNX Runtime 使用 XPU 执行提供者
- [ ] embedding 推理在 Intel GPU 上加速
- [ ] BF16 精度支持 (Intel 优化)

### 通用
- [ ] 代码编译通过，无警告
- [ ] 单元测试覆盖率保持
- [ ] 设备管理器测试覆盖新硬件
- [ ] 文档更新

## Timeline

- **Phase 1 (AMD)**: 2-3 周
  - Week 1: ONNX Runtime ROCm 集成
  - Week 2: 测试和优化
- **Phase 2 (Intel)**: 3-4 周
  - Week 3: IntelDeviceManager 实现
  - Week 4: ONNX Runtime XPU 集成
  - Week 5: 测试和优化

## References

- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [ONNX Runtime ROCm](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
- [ONNX Runtime XPU](https://onnxruntime.ai/docs/execution-providers/XPU-ExecutionProvider.html)
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)

---

**Change ID**: `add-amd-intel-gpu-support`  
**Status**: `draft`  
**Created**: 2025-01-14  
**Author**: Sisyphus AI Agent
