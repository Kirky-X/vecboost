# CLI 示例

VecBoost CLI 命令行工具使用示例。通过 `std::process::Command` 启动 `vecboost` 子进程调用 CLI 子命令,贴近真实使用场景。

## 前置条件

```bash
# 编译 vecboost 二进制(需 cli feature)
cargo build --release --features cli

# 确保 vecboost 在 PATH 中,或使用完整路径
export PATH="$PWD/target/release:$PATH"
```

## 示例列表

| 示例 | 说明 | 子命令 |
|------|------|--------|
| `embed_cli.rs` | 单文本嵌入 | `vecboost embed --text "hello"` |
| `batch_cli.rs` | 批量文件嵌入 | `vecboost batch --input texts.txt` |

## 运行方式

```bash
# 方式一:通过 cargo run(需先在 Cargo.toml 注册 example)
cargo run -p vecboost-examples --bin embed_cli --features cli
cargo run -p vecboost-examples --bin batch_cli --features cli

# 方式二:编译后直接运行
cargo build --release --features cli --example embed_cli
./target/release/examples/embed_cli
```

## 所需 Feature

- `cli` — 启用 CLI 子命令(clap + sdforge/cli)

## CLI 子命令参考

```
vecboost embed --text "Hello"              # 单文本嵌入,输出 JSON
vecboost batch --input file.txt            # 批量嵌入(每行一条文本),输出 JSON
vecboost similarity --text1 "a" --text2 "b" # 计算余弦相似度,输出 JSON
```

所有子命令输出 JSON 到 stdout,便于脚本化处理。
