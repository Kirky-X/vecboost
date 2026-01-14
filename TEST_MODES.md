# 测试模式配置指南

本文档介绍 VecBoost 测试框架中的测试模式配置。

## 概述

测试框架支持三种运行模式，可以在不同场景下使用：

- **Mock 模式**（默认）：使用确定性哈希算法生成向量
- **Light 模式**：使用小模型（BGE-small-en-v1.5，384维）进行真实推理
- **Full 模式**：使用完整模型（BGE-m3，1024维）进行真实推理

## 环境变量配置

### TEST_MODE

设置测试模式：

```bash
# 使用 Mock 模式（默认）
export TEST_MODE=mock

# 使用 Light 模式
export TEST_MODE=light

# 使用 Full 模式
export TEST_MODE=full
```

### TEST_MODEL_NAME

指定测试模型名称：

```bash
export TEST_MODEL_NAME=BAAI/bge-base-en-v1.5
```

### TEST_MODEL_PATH

指定本地模型路径：

```bash
export TEST_MODEL_PATH=/path/to/local/model
```

### TEST_MODEL_DIMENSION

覆盖模型维度：

```bash
export TEST_MODEL_DIMENSION=768
```

### TEST_API_BASE_URL

设置 API 服务基础 URL（Python 测试用）：

```bash
export TEST_API_BASE_URL=http://localhost:9002
```

## Rust 测试

### 运行所有测试

```bash
# Mock 模式（默认）
cargo test --test integration_test

# Light 模式
TEST_MODE=light cargo test --test integration_test

# Full 模式
TEST_MODE=full cargo test --test integration_test
```

### 运行单个测试

```bash
cargo test --test integration_test test_e2e_text_embedding
```

### 运行单元测试

```bash
cargo test --lib
```

## Python 测试

### 运行所有测试

```bash
# Mock 模式（默认）
pytest tests/

# Light 模式
TEST_MODE=light pytest tests/

# Full 模式
TEST_MODE=full pytest tests/
```

### 运行 API 测试

```bash
pytest tests/test_api.py -v
```

## RealTestEngine 特性

`RealTestEngine` 是 Rust 测试的核心组件，具有以下特性：

### 自动回退

当真实引擎初始化失败时，自动回退到 Mock 实现：

```rust
let engine = RealTestEngine::new();

// 如果无法加载真实模型，会自动使用 Mock
let result = engine.embed("test text");
```

### 模式检测

```rust
use real_engine::{RealTestEngine, TestMode};

// 获取当前模式
let mode = TestMode::from_env();
println!("Current mode: {:?}", mode);

// 检查是否使用真实引擎
if engine.is_using_real_engine() {
    println!("Using real inference");
} else {
    println!("Using mock fallback");
}
```

## API 客户端工厂

Python 测试提供了灵活的客户端工厂：

```python
from tests.client_factory import create_client

# 使用 Mock 客户端
mock_client = create_client("mock")

# 使用 Real 客户端
real_client = create_client("real", "http://localhost:9002")

# 自适应客户端（根据配置自动选择）
adaptive_client = create_client()
```

## 回退机制

### Rust 回退

```rust
let engine = RealTestEngine::new();

if engine.is_using_fallback() {
    println!("Using mock fallback due to: {}", engine.get_fallback_reason());
}
```

### Python 回退

```python
from tests.services import AdaptiveEmbeddingService

service = AdaptiveEmbeddingService()

if service.is_using_fallback():
    print("Using mock fallback")
```

## 性能考虑

| 模式 | 单次推理时间 | 适用场景 |
|------|-------------|---------|
| Mock | < 1ms | 单元测试、CI |
| Light | 10-50ms | 快速集成测试 |
| Full | 50-200ms | 完整集成测试 |

## CI/CD 配置示例

```yaml
# GitHub Actions 示例
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run unit tests (Mock mode)
        run: cargo test --lib

      - name: Run integration tests (Mock mode)
        run: cargo test --test integration_test

      - name: Run integration tests (Light mode)
        env:
          TEST_MODE: light
        run: cargo test --test integration_test
```

## 故障排除

### 模型下载失败

如果使用 Light 或 Full 模式时模型下载失败：

1. 检查网络连接
2. 确保 HuggingFace Hub 访问权限
3. 使用 `TEST_MODEL_PATH` 指定本地模型

### 维度不匹配

如果遇到维度错误：

1. 确保 `TEST_MODEL_DIMENSION` 与实际模型匹配
2. 检查模型配置文件

### 连接拒绝（Python 测试）

如果 Real 客户端无法连接：

1. 确保 VecBoost 服务正在运行
2. 检查 `TEST_API_BASE_URL` 配置
3. 使用 Mock 模式进行本地测试
