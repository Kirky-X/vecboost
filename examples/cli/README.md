# CLI 示例

VecBoost CLI 命令行工具使用示例。通过 `std::process::Command` 启动 `vecboost` 子进程调用 CLI 子命令,贴近真实使用场景。

## 前置条件

```bash
# 编译 vecboost 二进制(需 cli feature;workspace 默认含 onnx)
cargo build --release --features cli
```

> **⚠️ 二进制覆盖坑**:之后任何不带 `--features cli` 的 release 构建(如
> `cargo build --release`、`cargo build --release -p vecboost`)都会把
> `target/release/vecboost` 覆盖成**无 cli 版本**——未知子命令落入 HTTP
> 服务器模式,示例子进程将永久挂起。示例对此有预检(报指引性错误而非
> 挂起);被覆盖后重新执行 `cargo build --release --features cli` 恢复。

无需手动 `export PATH`:示例按 `VECBOOST_BIN`(显式覆盖)→ 仓内
`target/release/vecboost` → PATH 的顺序定位二进制。

### 运行时依赖(示例自动处理)

1. **ONNX Runtime 动态库**:`onnx` feature 走 `ort/load-dynamic` 运行期加载。
   `ORT_DYLIB_PATH` 未设时示例向子进程注入仓库自带 dylib(按平台:
   macOS `libonnxruntime.dylib` / Linux `libonnxruntime.so` /
   Windows `onnxruntime.dll`);兜底也缺失时示例报错退出。使用自装
   onnxruntime 时设 `ORT_DYLIB_PATH` 覆盖,说明见 `3rdparty/onnxruntime/README.md`。
   手动直调 `vecboost` 本体(不经本目录示例)时需自行 export,仓库根目录执行:
   ```bash
   # macOS
   export ORT_DYLIB_PATH="$PWD/3rdparty/onnxruntime/libonnxruntime.dylib"
   # Linux
   export ORT_DYLIB_PATH="$PWD/3rdparty/onnxruntime/libonnxruntime.so"
   ```
2. **配置**:示例生成临时最小配置(仅 `[model]` + `[logging]`)并显式传
   `--config`。**不要**用默认 `config/config.toml` 跑 CLI 单命令:其
   `[translation]` 段急切构建 4 个翻译语言对,需 4 个本地翻译 bundle
   (`models/m2m100_418M` 等),缺失即启动失败,连无关的 `embed` 也无法运行;
   `config/config_minimal.toml` 的 bge-m3 模型在 HF cache 布局下过不了
   onnx 外部数据路径校验,同样不可用。
3. **嵌入模型**:首次运行经 HuggingFace 下载 `BAAI/bge-small-en-v1.5`
   ONNX 权重(需网络,约 100 MB;之后走本地缓存)。

## 示例列表

| 示例 | 说明 | 子命令 |
|------|------|--------|
| `embed_cli.rs` | 单文本嵌入 | `vecboost embed --req '{"text":"hello"}'` |
| `batch_cli.rs` | 批量文件嵌入 | `vecboost embed_batch --req '{"texts":["..."]}'` |
| `rerank_cli.rs` | 文档重排序 | `vecboost rerank --req '{"query":"...","documents":[...]}'` |

## 运行方式

```bash
# 通过 cargo run(在仓库根目录执行;子进程失败时示例按子进程退出码退出)
cargo run -p vecboost-examples --bin embed_cli
cargo run -p vecboost-examples --bin batch_cli
cargo run -p vecboost-examples --bin rerank_cli
```

> 直接运行 `target/debug/<示例名>`(不经 cargo run)不会重编译——源码更新后
> 直跑旧二进制会看到旧行为;先 `cargo build -p vecboost-examples` 刷新。

## 所需 Feature

- `cli` — 启用 CLI 子命令(clap + sdforge/cli)

## CLI 子命令参考

```
vecboost embed --req '{"text":"Hello"}'              # 单文本嵌入,输出 JSON
vecboost embed_batch --req '{"texts":["a","b"]}'    # 批量嵌入(每行一条文本),输出 JSON
vecboost compute_similarity --req '{"source":"a","target":"b"}' # 计算余弦相似度,输出 JSON
vecboost rerank --req '{"query":"...","documents":["..."]}' # 文档重排序,输出 JSON
```

所有子命令输出 JSON 到 stdout,便于脚本化处理。
