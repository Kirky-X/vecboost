# Tasks: Add AMD and Intel GPU Support

## Phase 1: AMD ROCm Support

### Task 1.1: Enable ONNX Runtime ROCm Execution Provider
**Status**: pending  
**Priority**: High  
**Files**: `src/engine/onnx_engine.rs`  
**Validation**: ROCm EP enabled, session creation successful  

#### Steps
1. Update Cargo.toml for ROCm dependency
2. Add ROCmExecutionProvider import
3. Configure session with ROCm EP
4. Add ROCm version detection
5. Test ROCm EP initialization

#### Details
- Line 127-143: Replace AMD placeholder with ROCm EP
- Add ROCm version detection via `rocm-smi`
- Enable FP16 for ROCm if supported

---

### Task 1.2: Update AMD Device Detection
**Status**: pending  
**Priority**: Medium  
**Files**: `src/device/amd.rs`  
**Validation**: ROCm devices detected correctly  

#### Steps
1. Improve ROCm detection via `rocm-smi`
2. Add ROCm version detection
3. Get accurate VRAM info from ROCm
4. Add compute capability detection
5. Test on ROCm-enabled system

#### Details
- Lines 100-117: Improve `from_rocm()` method
- Add `detect_rocm_version()` function
- Add `detect_rocm_vram()` function improvement

---

### Task 1.3: Add ROCm Memory Pool Support
**Status**: pending  
**Priority**: Medium  
**Files**: `src/device/memory_pool/cuda_pool.rs`  
**Validation**: ROCm memory pool initialized  

#### Steps
1. Add ROCm memory pool config
2. Implement ROCm memory allocation wrapper
3. Add ROCm memory stats
4. Test memory pool initialization

---

### Task 1.4: Update Device Manager for AMD
**Status**: pending  
**Priority**: Medium  
**Files**: `src/device/manager.rs`  
**Validation**: AMD devices in device list  

#### Steps
1. Add AMD device info to unified list
2. Add AMD memory monitoring
3. Add AMD priority in device selection
4. Test device enumeration

---

### Task 1.5: Test AMD ROCm Integration
**Status**: pending  
**Priority**: High  
**Files**: `src/device/amd.rs`, `src/engine/onnx_engine.rs`  
**Validation**: All tests pass  

#### Steps
1. Add AMD device detection tests
2. Add ROCm EP initialization tests
3. Add integration tests with AMD device
4. Verify FP16 precision works
5. Performance benchmark on AMD GPU

---

## Phase 2: Intel XPU Support

### Task 2.1: Create IntelDeviceManager
**Status**: pending  
**Priority**: High  
**Files**: `src/device/intel.rs` (new)  
**Validation**: Intel XPU detected  

#### Steps
1. Create `src/device/intel.rs`
2. Implement `IntelDevice` struct
3. Implement `IntelDeviceManager` struct
4. Add XPU detection via `sycl-ls` or OpenCL
5. Add Intel GPU info struct

#### Details
- Detect Intel GPUs via OpenCL (cross-platform)
- Detect XPU via oneAPI (Linux/Windows)
- Get VRAM and compute info
- Support integrated and discrete GPUs

---

### Task 2.2: Add Intel to Device Module
**Status**: pending  
**Priority**: High  
**Files**: `src/device/mod.rs`  
**Validation**: Intel module exported  

#### Steps
1. Add `pub mod intel;` to device/mod.rs
2. Export Intel types
3. Add Intel device category
4. Update device selection logic

---

### Task 2.3: Enable ONNX Runtime XPU Execution Provider
**Status**: pending  
**Priority**: High  
**Files**: `src/engine/onnx_engine.rs`  
**Validation**: XPU EP enabled, session creation successful  

#### Steps
1. Add XPUExecutionProvider import
2. Configure session with XPU EP
3. Add Intel device detection
4. Enable BF16 precision for Intel
5. Test XPU EP initialization

#### Details
- Add XPU EP after CUDA EP
- Support BF16 on Intel XPU
- Fallback to CPU if XPU not available

---

### Task 2.4: Add Intel Memory Pool Support
**Status**: pending  
**Priority**: Medium  
**Files**: `src/device/memory_pool/intel_pool.rs` (new)  
**Validation**: Intel memory pool initialized  

#### Steps
1. Create Intel memory pool
2. Implement allocation wrapper
3. Add memory stats
4. Test memory pool

---

### Task 2.5: Test Intel XPU Integration
**Status**: pending  
**Priority**: High  
**Files**: `src/device/intel.rs`, `src/engine/onnx_engine.rs`  
**Validation**: All tests pass  

#### Steps
1. Add Intel device detection tests
2. Add XPU EP initialization tests
3. Add integration tests with Intel device
4. Verify BF16 precision works
5. Performance benchmark on Intel GPU

---

## Phase 3: Unified Device Abstraction

### Task 3.1: Refactor Device Manager
**Status**: pending  
**Priority**: Medium  
**Files**: `src/device/manager.rs`  
**Validation**: Unified device selection  

#### Steps
1. Create unified device trait
2. Refactor CUDA/AMD/Intel managers
3. Add device priority logic
4. Test unified selection

---

### Task 3.2: Update Config for New Devices
**Status**: pending  
**Priority**: Low  
**Files**: `src/config/model.rs`  
**Validation**: Config supports new devices  

#### Steps
1. Add Intel XPU to DeviceType enum
2. Update serialization
3. Add config tests

---

### Task 3.3: Documentation Update
**Status**: pending  
**Priority**: Low  
**Files**: README.md, AGENTS.md  
**Validation**: Documentation complete  

#### Steps
1. Update GPU support table
2. Add build instructions for AMD/Intel
3. Add troubleshooting section

---

## Dependencies

- Task 1.1 → Task 1.5 (sequential)
- Task 2.1 → Task 2.5 (sequential)
- Task 1.5 and 2.5 → Task 3.1 (after both phases)
- Task 3.1 → Task 3.2 → Task 3.3 (sequential)

## Total Estimated Effort

- Phase 1 (AMD): 2-3 weeks
- Phase 2 (Intel): 3-4 weeks
- Phase 3 (Unification): 1-2 weeks

**Total: 6-9 weeks**

---

**Created**: 2025-01-14  
**Status**: `draft`
