# AMD and Intel GPU Support Specs

This directory contains specifications for adding AMD and Intel GPU support to VecBoost.

## Specs

- [AMD GPU Support](amd-gpu/spec.md) - ROCm detection and ONNX Runtime integration
- [Intel GPU Support](intel-gpu/spec.md) - XPU detection and ONNX Runtime integration  
- [Unified Device Abstraction](unified-device/spec.md) - Common traits for GPU managers

## Overview

These specs address the following hardware support gaps:

| GPU Vendor | Current Support | Target Support |
|------------|-----------------|----------------|
| NVIDIA | ✅ Full (CUDA) | ✅ Full |
| AMD | ⚠️ Detection only | ✅ Full (ROCm) |
| Intel | ❌ None | ✅ Full (XPU) |

## Key Requirements

### AMD GPU (Phase 1)
1. ROCm device detection via `rocm-smi`
2. ONNX Runtime ROCm Execution Provider
3. ROCm memory pool support
4. FP16 precision support

### Intel GPU (Phase 2)
1. XPU/oneAPI device detection
2. OpenCL fallback detection
3. ONNX Runtime XPU Execution Provider
4. Intel memory pool support
5. BF16 precision support

### Unified Abstraction (Phase 3)
1. GpuDeviceManager trait
2. Unified device selection
3. Memory pool abstraction
4. Capability reporting

## Dependencies

- ONNX Runtime with ROCm/XPU execution providers
- ROCm runtime (AMD)
- oneAPI/SYCL (Intel)

---

**Created**: 2025-01-14  
**Version**: 1.0
