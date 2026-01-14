# Spec: AMD GPU Support

## ADDED Requirements

### REQ-001: ROCm Device Detection
**Priority**: Must Have  
**Description**: AMD GPU devices must be detected via ROCm when available.

#### Scenario: ROCm Installation Detected
- **Given** ROCm is installed on the system
- **When** VecBoost starts device enumeration
- **Then** AMD GPUs should be detected via `rocm-smi`
- **And** device info should include ROCm version
- **And** VRAM should be detected accurately

#### Scenario: OpenCL Fallback
- **Given** ROCm is not installed but OpenCL is available
- **When** ROCm detection fails
- **Then** AMD GPUs should be detected via OpenCL
- **And** device info should indicate OpenCL mode
- **And** ROCm version should be None

#### Scenario: No AMD GPU
- **Given** no AMD GPU is present in the system
- **When** VecBoost enumerates devices
- **Then** AMD device list should be empty
- **And** no error should be raised

---

### REQ-002: ONNX Runtime ROCm Execution Provider
**Priority**: Must Have  
**Description**: ONNX Runtime must use ROCm execution provider for AMD GPUs.

#### Scenario: ROCm EP Available
- **Given** `onnxruntime-rocm` feature is enabled
- **And** AMD GPU is detected
- **When** creating ONNX session
- **Then** ROCmExecutionProvider should be configured
- **And** session should use GPU for inference

#### Scenario: ROCm EP Not Available
- **Given** `onnxruntime-rocm` feature is disabled
- **When** creating ONNX session with AMD device
- **Then** warning should be logged
- **And** session should fall back to CPU
- **And** inference should still work

#### Scenario: ROCm Version Mismatch
- **Given** ROCm is installed but version incompatible
- **When** initializing ROCm execution provider
- **Then** graceful degradation to CPU should occur
- **And** warning should include version info

---

### REQ-003: ROCm Memory Pool
**Priority**: Should Have  
**Description**: Memory pool should support AMD GPU memory allocation.

#### Scenario: ROCm Memory Allocation
- **Given** ROCm is available and enabled
- **When** allocating GPU memory
- **Then** hipMalloc should be used
- **And** memory should be tracked accurately
- **And** memory stats should be available

#### Scenario: ROCm Memory Release
- **Given** allocated ROCm memory
- **When** releasing memory
- **Then** hipFree should be used
- **And** memory should be returned to pool

#### Scenario: ROCm Memory Pool Disabled
- **Given** ROCm memory pool is disabled
- **When** allocating memory
- **Then** direct allocation should be used
- **And** no pooling overhead should occur

---

### REQ-004: AMD Precision Support
**Priority**: Should Have  
**Description**: AMD GPU should support appropriate precision modes.

#### Scenario: FP16 on ROCm
- **Given** AMD GPU with ROCm support
- **When** precision is set to FP16
- **Then** FP16 inference should be used
- **And** performance should improve vs FP32

#### Scenario: FP32 Fallback
- **Given** AMD GPU without FP16 support
- **When** precision is set to FP16
- **Then** FP32 should be used as fallback
- **And** warning should be logged

---

## MODIFIED Requirements

### REQ-005: Device Manager Integration
**Status**: MODIFIED from `Implicit` to `Explicit`  
**Description**: AMD devices must be fully integrated with device manager.

#### Scenario: Unified Device List
- **Given** multiple GPU types available (NVIDIA, AMD)
- **When** listing available devices
- **Then** all devices should appear in unified list
- **And** each device should have correct type and capabilities

#### Scenario: AMD Device Selection
- **Given** AMD GPU is available
- **When** selecting device for inference
- **Then** AMD GPU should be selectable
- **And** appropriate

---

## REM engine should be chosenOVED Requirements

### REQ-006: Candle AMD Support
**Status**: REMOVED  
**Description**: Candle engine AMD support is not required.

#### Rationale
- Candle does not support AMD ROCm natively
- ONNX Runtime ROCm EP provides the acceleration
- Resources should focus on ONNX Runtime integration

---

**Created**: 2025-01-14  
**Version**: 1.0
