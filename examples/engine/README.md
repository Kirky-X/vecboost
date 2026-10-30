# Engine 示例

本目录包含 VecBoost 推理引擎的使用示例,演示如何初始化引擎、执行嵌入推理以及在运行时切换模型。

## 示例列表

| 示例 | 说明 | 所需 feature |
|------|------|--------------|
| `candle.rs` | Candle 引擎初始化与推理(使用 MockEngine 演示 `InferenceEngine` trait) | 无 |
| `onnx.rs` | ONNX 引擎初始化(启用 `onnx` 时演示真实引擎创建,否则使用 MockEngine) | `onnx`（`required-features`） |
| `laya_poc.rs` | Laya 决策管线 P0 数值对照(经 EngineFactory→AnyEngine→trait 全链,与 golden 概率逐值对照) | `onnx`（`required-features`）+ 本地 bundle + ONNX Runtime 动态库 |
| `switch.rs` | 运行时通过 `EmbeddingService::switch_model` 切换模型 | 无 |

## 运行命令

```bash
# Candle 引擎示例（vecboost-examples 不定义 http feature，不加 --features）
cargo run -p vecboost-examples --bin candle

# ONNX 引擎示例
cargo run -p vecboost-examples --bin onnx --features onnx

# Laya 决策管线 P0 数值对照(需本地 bundle,默认 models/laya;缺失时打印下载指引后退出;
# ONNX Runtime 动态库自动回退 3rdparty/,可用 ORT_DYLIB_PATH 覆盖)
cargo run -p vecboost-examples --bin laya_poc --features onnx

# 运行时模型切换示例
cargo run -p vecboost-examples --bin switch
```

## 说明

- 所有示例使用 `MockEngine` 避免依赖真实模型文件,可在无 GPU/模型环境中运行。
- `InferenceEngine` trait 是引擎抽象接口;真实引擎(Candle/Onnx)通过公共 API `AnyEngine::new(config, engine_type, precision)` 创建。
- 真实引擎创建需要有效的模型路径,示例中会显式展示无模型时的错误。

### laya_poc 前置条件

1. **ONNX Runtime 动态库**:`onnx` feature 走 `ort/load-dynamic` 运行期加载。
   `ORT_DYLIB_PATH` 未设时示例自动回退仓库自带 `3rdparty/onnxruntime/`
   (macOS `libonnxruntime.dylib` / Linux `libonnxruntime.so` / Windows
   `onnxruntime.dll`);兜底也缺失时示例显性报错退出。使用自装 onnxruntime
   时设 `ORT_DYLIB_PATH` 覆盖,版本与落位说明见 `3rdparty/onnxruntime/README.md`。
2. **Laya bundle**:默认 `models/laya`(环境变量 `LAYA_BUNDLE` 覆盖,相对运行目录)。
   bundle 缺失时打印下载指引后正常退出(SKIP 语义)。官方 bundle 5 文件清单与
   下载 API 见 USER_GUIDE「Laya 决策 bundle 获取」节。
3. **golden 文件(可选)**:本仓不附带 golden——`LAYA_GOLDEN_JSON` 指向 Python
   onnxruntime 复现「预处理→推理→温度校准→softmax」全管线后导出的 JSON 才做
   逐值断言(|diff| < 1e-4);未设时仅打印管线输出供人工对照,不会静默当成已验证。
4. **校准回退提示**:bundle 缺 `laya_config.json` 时库回退温度 1.2 并经
   `log::warn!` 显性提示;`log` 门面无实现时 warn 静默,故本示例内置了 stderr
   logger(Warn 级)使该提示可见。
