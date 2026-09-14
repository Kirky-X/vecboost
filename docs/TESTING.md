# 测试策略(TESTING)

本文档划分 vecboost 三套性能/质量验证栈的职责边界,回答"该把测试写在哪里"。

## 三层总览

| 层 | 位置 | 引擎 | 职责 | 运行方式 |
|---|---|---|---|---|
| **微基准** | `benches/`(criterion) | Mock/真实均可 | 度量原子操作的绝对性能并防回归(相似度 SIMD、批调度、语义缓存索引) | `cargo bench -p vecboost` |
| **回归阈值** | `tests/perf/performance_test.rs` | MockEngine | 断言吞吐/延迟不劣化(相对阈值),CI 可跑、不依赖真实模型 | `cargo test -p vecboost --test performance_test` |
| **真实服务 E2E** | `tests/scenario/*.py`(pytest) | 真实二进制 + models/ 本地模型 | 端到端行为契约:HTTP 矩阵/CLI/MCP/生命周期/安全;P99 延迟与真实吞吐 | `pytest tests/scenario -q`(conftest 自动拉起服务器,端口 9101-9143) |

## tests/perf 的 Python 部分

`tests/perf/` 中同时存在两类 pytest 用例,**用标记区分**:

- `@pytest.mark.sim`:打 Python `api_simulator.py` 的语义模拟,用于快速回归 API
  形状/错误路径,**不代表 Rust 服务行为**;
- 无标记 / `@pytest.mark.real`:经 `real_service.py` 打真实 `localhost:9002`。

```bash
pytest tests/perf -m "not sim"   # 只跑真实服务用例
pytest tests/perf -m sim          # 只跑模拟器用例
```

> 修改 Rust API 语义时,`sim` 用例必须同步更新;若发现 sim 与真实行为冲突,
> 以真实服务为准并修 sim。

## 模型矩阵

`tests/scenario/test_model_matrix.py` 与 `tests/scenario_sdk.rs` 按 `models/<目录名>`
存在性自动 SKIP。本机 4 模型(BGE-en 384 / MiniLM 384 / BGE-zh 512 /
multilingual-e5-small 384)齐全时覆盖 XlmRoberta + Bert 双架构、三种 pooling。

## 约定

1. **新性能断言优先写回归阈值层**(MockEngine、确定性),微基准只用于优化工作的前后对比;
2. **新端点行为契约写 scenario 层**(真实服务器 + HTTP),这同时是 API 文档的活规格;
3. **基线数字记录到 `docs/benchmarks/`**,标注环境与噪声区间;
4. 测试代码允许 `unwrap`(`clippy.toml` 已豁免),生产代码禁止。
