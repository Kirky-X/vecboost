# Spec — inference-engine

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖新引擎(TensorRT/OpenVINO)能力域需求。

## Requirements

### R-inference-engine-001: TensorRT 引擎实现

`src/engine/tensorrt_engine.rs` 实现 `pub(crate) struct TensorRtEngine` + `impl InferenceEngine for TensorRtEngine`。基于 `tensorrt-rs` crate,加载 `.engine`/`.plan` 序列化模型,支持 NVIDIA GPU 推理。

**v0.2.0 实际实施状态(stub 实现,真实 crate 集成推迟到 v0.3.0)**:
- `src/engine/tensorrt_engine.rs:29-31` 的 `TensorRtEngine::new(...)` **始终返回 `Err(VecboostError::InferenceError(RUNTIME_UNAVAILABLE.to_string()))`**
- `Cargo.toml` 中 `tensorrt = []`(空 feature,无 `dep:tensorrt-rs`)
- 原因:`tensorrt-sys` 缺 `libnvinfer.so` 编译失败,按约束#6 改为纯 stub,不依赖外部 crate
- 详见 `tasks.md` T034/T035(stub 实现 + 9 个测试)

**验收标准(按实际代码 stub 行为):**
- `TensorRtEngine::new(model_path)` 始终返回 `Err(VecboostError::InferenceError("TensorRT runtime not available"))`
- `TensorRtEngine::embed("Hello")` 返回错误
- `TensorRtEngine::embed_batch(["a", "b", "c"])` 返回错误
- 9 个测试覆盖 stub 行为(`new()`/`embed()`/`embed_batch()`/`precision`/`supports_mixed_precision`/`fallback`)

**推迟到 v0.3.0(真实 crate 集成):**
- `TensorRtEngine::new(model_path)` 加载序列化引擎,返回 `Ok(Self)`
- `TensorRtEngine::embed("Hello")` 返回 `Vec<f32>` 长度 1024
- GPU 不可用时返回 `Err(VecboostError::InferenceError("CUDA device not available"))`
- 模型路径不存在返回 `Err(VecboostError::ModelLoadError(...))`

### R-inference-engine-002: OpenVINO 引擎实现

`src/engine/openvino_engine.rs` 实现 `pub(crate) struct OpenVinoEngine` + `impl InferenceEngine for OpenVinoEngine`。基于 `openvino` crate,加载 `.xml` + `.bin` 模型,支持 Intel CPU/GPU/VPU。

**v0.2.0 实际实施状态(stub 实现,真实 crate 集成推迟到 v0.3.0)**:
- `src/engine/openvino_engine.rs:28-30` 的 `OpenVinoEngine::new(...)` **始终返回 `Err(VecboostError::InferenceError(RUNTIME_UNAVAILABLE.to_string()))`**
- `Cargo.toml` 中 `openvino = []`(空 feature,无 `dep:openvino`)
- 原因:`openvino-sys` 缺 `libopenvino_c.so` 编译失败,按约束#6 改为纯 stub
- 详见 `tasks.md` T036/T037(stub 实现 + 11 个测试)

**验收标准(按实际代码 stub 行为):**
- `OpenVinoEngine::new(model_xml, model_bin, device)` 始终返回 `Err(VecboostError::InferenceError("OpenVINO runtime not available"))`
- `OpenVinoEngine::embed("Hello")` 返回错误
- 11 个测试覆盖 stub 行为(`new()`/`embed()`/CPU/GPU device 切换/`precision`/`fallback`)

**推迟到 v0.3.0(真实 crate 集成):**
- `OpenVinoEngine::new(model_xml, model_bin, device)` 加载模型,`device` 支持 `CPU`/`GPU`/`VPU`
- `OpenVinoEngine::embed("Hello")` 返回 `Vec<f32>` 长度 1024
- 设备不可用时降级到 CPU 并记录 warning 日志
- 模型文件损坏返回 `Err(VecboostError::ModelFileCorrupted(...))`

### R-inference-engine-003: EngineFactory 多引擎选择

`src/engine/factory.rs` 实现 `EngineFactory::create(engine_type: EngineType, config: &ModelConfig) -> Result<AnyEngine, VecboostError>`。`EngineType` 枚举支持 `Candle`/`Onnx`/`TensorRt`/`OpenVino`。

**v0.2.0 实际实施状态(已实现,返回类型偏离原 spec)**:
- `src/engine/factory.rs:28` 实际签名为 `pub fn create(engine_type: EngineType, config: &ModelConfig) -> Result<AnyEngine, VecboostError>`
- 返回 `AnyEngine` 枚举(非 `Box<dyn InferenceEngine>`),`AnyEngine` 在 `src/engine/mod.rs:52-55` 定义,含 `Candle`/`Onnx`/`TensorRt`/`OpenVino` 4 个 variant
- `EngineType` 枚举在 `src/config/model.rs:141-144` 定义,`TensorRt`/`OpenVino` variant feature-gated
- `Display` impl 也 feature-gated(`model.rs:153-156`)

**验收标准(按实际代码):**
- `EngineFactory::create(EngineType::Candle, config)` 返回 `Ok(AnyEngine::Candle(...))`(默认)
- `EngineFactory::create(EngineType::TensorRt, config)` 启用 `tensorrt` feature 时返回 `Ok(AnyEngine::TensorRt(...))`(实际为 stub),否则返回 `Err`
- `EngineFactory::create(EngineType::OpenVino, config)` 启用 `openvino` feature 时返回 `Ok(AnyEngine::OpenVino(...))`(实际为 stub),否则返回 `Err`
- `EngineType` 从 `[model].engine_type` 配置读取,默认 `Candle`

**偏离说明(返回 `AnyEngine` 而非 `Box<dyn InferenceEngine>`)**:
- `AnyEngine` 枚举匹配模式更精确,编译期类型检查更严格
- 性能略优(避免动态分发开销)
- 与 `src/engine/mod.rs` 现有抽象一致

### R-inference-engine-004: 引擎切换运行时

通过 `EmbeddingService::switch_engine(engine_type)` 在运行时切换引擎,无需重启服务。切换时旧引擎优雅关闭,新引擎加载完成后接管。

**v0.2.0 实际实施状态(未实现,推迟到 v0.3.0)**:
- `grep -rn "switch_engine" src/service/` 无匹配
- `EmbeddingService` 无 `switch_engine` 方法
- 原因:运行时引擎切换涉及线程安全、并发请求处理、状态管理等复杂逻辑,v0.2.0 聚焦基础设施迁移,推迟到 v0.3.0

**验收标准(目标,v0.3.0 实现):**
- `service.switch_engine(EngineType::OpenVino)` 后,`service.embed("Hello")` 使用 OpenVINO 引擎
- 切换过程中正在处理的请求用旧引擎完成,新请求等待新引擎就绪
- 切换失败回滚到旧引擎,返回 `Err`
- 切换事件记录到审计日志

## Constraints

- TensorRT 通过 `tensorrt` feature 启用,仅支持 Linux + CUDA
- OpenVINO 通过 `openvino` feature 启用,支持 Linux/Windows/macOS
- TensorRT/OpenVINO 引擎测试用 `mockall` mock,CI 无需真实硬件
- 保留 Candle 为默认引擎,ONNX 为可选,新增 TensorRT/OpenVINO 为可选
- `InferenceEngine` trait 接口不变,仅新增实现

## Out of Scope

- 不实现 TensorRT 动态 shape(本轮仅固定 shape 1024)
- 不实现 OpenVINO 异步推理(本轮仅同步)
- 不实现引擎性能基准对比(留下个 change)
- 不实现 Triton Inference Server 集成(留下个 change)
- 不实现新 ML 模型支持(本轮聚焦引擎,模型保持 BGE-M3)
