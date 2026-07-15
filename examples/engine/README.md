# Engine 示例

本目录包含 VecBoost 推理引擎的使用示例,演示如何初始化引擎、执行嵌入推理以及在运行时切换模型。

## 示例列表

| 示例 | 说明 | 所需 feature |
|------|------|--------------|
| `candle.rs` | Candle 引擎初始化与推理(使用 MockEngine 演示 `InferenceEngine` trait) | `http` |
| `onnx.rs` | ONNX 引擎初始化(启用 `onnx` 时演示真实引擎创建,否则使用 MockEngine) | `onnx` |
| `switch.rs` | 运行时通过 `EmbeddingService::switch_model` 切换模型 | `http` |

## 运行命令

```bash
# Candle 引擎示例
cargo run -p vecboost-examples --bin candle --features http

# ONNX 引擎示例
cargo run -p vecboost-examples --bin onnx --features onnx

# 运行时模型切换示例
cargo run -p vecboost-examples --bin switch --features http
```

## 说明

- 所有示例使用 `MockEngine` 避免依赖真实模型文件,可在无 GPU/模型环境中运行。
- `InferenceEngine` trait 是引擎抽象接口;真实引擎(Candle/Onnx)通过公共 API `AnyEngine::new(config, engine_type, precision)` 创建。
- 真实引擎创建需要有效的模型路径,示例中会显式展示无模型时的错误。
