# ONNX Runtime 动态库（`--features onnx` 运行时依赖）

`onnx` feature 使用 `ort/load-dynamic`：二进制不链接 ONNX Runtime，
启动时经 `ORT_DYLIB_PATH` 环境变量加载动态库。本目录存放官方预编译库，
**库文件本体不入 git**（`.gitignore` 的 `*.so` 规则覆盖）。

## 当前放置

- `libonnxruntime.so.1.28.2` + `libonnxruntime.so -> libonnxruntime.so.1.28.2`
- 来源：<https://github.com/microsoft/onnxruntime/releases/tag/v1.28.2>
  （`onnxruntime-linux-x64-1.28.2.tgz`，解压 `lib/libonnxruntime.so*`）

## 版本要求

- 不得低于 `ort-sys` 的 `ORT_API_VERSION`（默认 17，即 onnxruntime ≥ 1.17）；
- 不得使用 1.22：`ort 2.0.0-rc.13` 将 `GraphOptimizationLevel::Level3`
  映射为 `ORT_ENABLE_LAYOUT = 3`，1.22 及更早运行时校验该值非法
  （报 `graph_optimization_level is not valid`），1.23+ 才引入该枚举值。

## 启动方式

```bash
ORT_DYLIB_PATH=3rdparty/onnxruntime/libonnxruntime.so vecboost
```
