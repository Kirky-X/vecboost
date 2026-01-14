# Spec: Intel GPU Support

## ADDED Requirements

### REQ-001: Intel XPU Device Detection
**Priority**: Must Have  
**Description**: Intel GPU devices must be detected via XPU or OpenCL.

#### Scenario: Intel XPU via oneAPI
- **Given** Intel GPU with oneAPI/XPU support (Linux)
- **When** VecBoost starts device enumeration
- **Then** Intel GPUs should be detected via `sycl-ls`
- **And** device info should include XPU version
- **And** VRAM should be detected accurately

#### Scenario: Intel GPU via OpenCL
- **Given** Intel GPU without oneAPI but with OpenCL
- **When** XPU detection fails
- **Then** Intel GPUs should be detected via OpenCL
- **And** device info should indicate OpenCL mode

#### Scenario: Intel Integrated Graphics
- **Given** Intel CPU with integrated graphics
- **When** detecting Intel devices
- **Then** integrated GPU should be detected
- **And** shared memory should be reported correctly

#### Scenario: Intel Arc Discrete GPU
- **Given** Intel Arc discrete GPU
- **When** detecting Intel devices
- **Then** Arc GPU should be detected
- **And** dedicated VRAM should be reported

#### Scenario: No Intel GPU
- **Given** no Intel GPU in the system
- **When** VecBoost enumerates devices
- **Then** Intel device list should be empty
- **And** no error should be raised

---

### REQ-002: ONNX Runtime XPU Execution Provider
**Priority**: Must Have  
**Description**: ONNX Runtime must use XPU execution provider for Intel GPUs.

#### Scenario: XPU EP Available
- **Given** `onnxruntime-xpu` feature is enabled
- **And** Intel GPU is detected
- **When** creating ONNX session
- **Then** XPUExecutionProvider should be configured
- **And** session should use GPU for inference

#### Scenario: XPU EP Not Available
- **Given** `onnxruntime-xpu` feature is disabled
- **When** creating ONNX session with Intel device
- **Then** warning should be logged
- **And** session should fall back to CPU
- **And** inference should still work

#### Scenario: XPU Version Mismatch
- **Given** oneAPI is installed but version incompatible
- **When** initializing XPU execution provider
- **Then** graceful degradation to CPU should occur
- **And** warning should include version info

---

### REQ-003: Intel XPU Memory Pool
**Priority**: Should Have  
**Description**: Memory pool should support Intel XPU memory allocation.

#### Scenario: XPU Memory Allocation
- **Given** XPU is available and enabled
- **When** allocating GPU memory
- **Then** oneAPI Malloc should be used
- **And** memory should be tracked accurately
- **And** memory stats should be available

#### Scenario: XPU Memory Release
- **Given** allocated XPU memory
- **When** releasing memory
- **Then** oneAPI Free should be used
- **And** memory should be returned to pool

---

### REQ-004: Intel Precision Support
**Priority**: Should Have  
**Description**: Intel XPU should support BF16 precision (Intel optimized).

#### Scenario: BF16 on Intel XPU
- **Given** Intel GPU with BF16 support
- **When** precision is set to BF16
- **Then** BF16 inference should be used
- **And** performance should improve vs FP32

#### Scenario: FP16 Fallback
- **Given** Intel GPU without BF16 support
- **When** BF16 precision is requested
- **Then** FP16 should be used as fallback
- **And** warning should be logged

#### Scenario: FP16 on Intel
- **Given** Intel GPU with FP16 support
- **When** precision is set to FP16
- **Then** FP16 inference should be used

---

### REQ-005: Intel Device Manager
**Priority**: Must Have  
**Description**: IntelDeviceManager must be created and integrated.

#### Scenario: Intel Device Manager Creation
- **Given** VecBoost starts
- **When** initializing device managers
- **Then** IntelDeviceManager should be created
- **And** should attempt to detect Intel devices

#### Scenario: Intel Device Enumeration
- **Given** IntelDeviceManager is initialized
- **When** enumerating devices
- **Then** all Intel GPUs should be listed
- **And** each should have correct info (VRAM, compute units)

#### Scenario: Primary Device Selection
- **Given** multiple Intel GPUs
- **When** selecting primary device
- **Then** first available GPU should be selected
- **And** user should be able to override

---

## MODIFIED Requirements

### REQ-006: Device Type Enum Extension
**Status**: MODIFIED from `Implicit` to `Explicit`  
**Description**: DeviceType enum must explicitly include Intel.

#### Scenario: Intel Device Type
- **Given** DeviceType enum
- **When** serializing Intel device
- **Then** value should be "intel" or "xpu"
- **And** deserialization should work correctly

#### Scenario: Intel Configuration
- **Given** model configuration with Intel device
- **When** parsing config
- **Then** Intel device type should be recognized

---

**Created**: 2025-01-14  
**Version**: 1.0
