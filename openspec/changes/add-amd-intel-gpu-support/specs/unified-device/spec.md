# Spec: Unified Device Abstraction

## ADDED Requirements

### REQ-001: GpuDeviceManager Trait
**Priority**: Must Have  
**Description**: Common trait for GPU device managers must be defined.

#### Scenario: Trait Implementation
- **Given** a GPU device manager (CUDA, AMD, Intel)
- **When** implementing GpuDeviceManager trait
- **Then** all required methods must be implemented
- **And** behavior should be consistent across implementations

#### Scenario: Trait Object Creation
- **Given** GpuDeviceManager trait
- **When** creating trait object from concrete type
- **Then** it should be usable interchangeably
- **And** dynamic dispatch should work correctly

---

### REQ-002: Unified Device Selection
**Priority**: Must Have  
**Description**: Device manager should select optimal GPU automatically.

#### Scenario: Automatic GPU Selection
- **Given** multiple GPU types available
- **When** no specific GPU is requested
- **Then** best available GPU should be selected
- **And** priority should be: CUDA > ROCm > XPU > CPU

#### Scenario: Specific GPU Selection
- **Given** user specifies GPU type (e.g., "amd")
- **When** creating engine
- **Then** specified GPU type should be used
- **And** fallback should occur if not available

#### Scenario: GPU Not Available
- **Given** requested GPU type is not available
- **When** creating engine
- **Then** warning should be logged
- **And** fallback to CPU should occur
- **And** inference should still work

---

### REQ-003: Memory Pool Abstraction
**Priority**: Should Have  
**Description**: Common trait for GPU memory pools must be defined.

#### Scenario: Memory Pool Trait
- **Given** memory pools for different GPUs
- **When** implementing GpuMemoryPool trait
- **Then** allocation/deallocation should work
- **And** memory tracking should be consistent

#### Scenario: Memory Pool Selection
- **Given** selected GPU type
- **When** creating memory pool
- **Then** appropriate pool implementation should be used
- **And** GPU-specific APIs should be used

---

### REQ-004: Device Capability Reporting
**Priority**: Should Have  
**Description**: All GPUs should report capabilities consistently.

#### Scenario: Capability Query
- **Given** any GPU device
- **When** querying capabilities
- **Then** supported precisions should be reported
- **And** max memory should be reported
- **And** compute capability should be reported

#### Scenario: Precision Support
- **Given** GPU device
- **When** checking precision support
- **Then** FP16 support should be indicated
- **Then** INT8 support should be indicated (if available)
- **Then** BF16 support should be indicated (Intel)

---

## MODIFIED Requirements

### REQ-005: Device Manager Refactoring
**Status**: MODIFIED from `Independent` to `Unified`  
**Description**: Device manager should use unified trait.

#### Scenario: Unified Initialization
- **Given** DeviceManager struct
- **When** initializing for GPU support
- **Then** all GPU managers should be initialized
- **And** failures should be handled gracefully

#### Scenario: Unified Enumeration
- **Given** DeviceManager with GPU support
- **When** enumerating all devices
- **Then** devices from all GPU types should be included
- **And** each should have correct category (GPU/CPU)

---

**Created**: 2025-01-14  
**Version**: 1.0
